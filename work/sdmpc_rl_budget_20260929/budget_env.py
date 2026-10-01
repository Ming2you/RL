"""Sequential physical-interval environment for the frozen budget coordinator."""
import copy
from dataclasses import asdict
import inspect
import math
import time
from types import SimpleNamespace
import numpy as np
from budget_controller import BudgetController
from budget_runtime import plain, protocol, read, digest


def numbers(value, prefix=""):
    if isinstance(value, dict):
        for key in sorted(value):
            yield from numbers(value[key], prefix + "/" + str(key))
    elif isinstance(value, (tuple, list)):
        for index, entry in enumerate(value):
            yield from numbers(entry, prefix + "/" + str(index))
    elif isinstance(value, (float, int, np.number)):
        yield prefix, float(value)
    else:
        raise TypeError("Non-numeric observation leaf: " + prefix)


class Observation:
    """Fixed physical scaling; retain every delayed-arrival time bin."""
    def __init__(self, cfg):
        self.cfg = cfg
        net = cfg.network
        self.links = tuple(sorted(net.urban_link_storage_veh))
        sources = {str(m["origin"]) for m in net.urban_movements.values() if "origin" in m}
        self.buffer_keys = tuple(sorted(set(self.links) | sources))
        capacity = max([net.boundary_queue_max_veh, *net.urban_link_storage_veh.values()])
        self.delay = math.ceil(capacity * net.urban_avg_vehicle_length_m / 1000. /
                               net.urban_avg_speed_km_h / cfg.simulation.T_u_h) + 2
        self.names = None

    def contract(self):
        # Pin ordering and every normalization expression, as well as resolved names/bins.
        return plain(dict(version=1, implementation_sha256=digest(
            [inspect.getsource(Observation), inspect.getsource(numbers)]), names=self.names,
            links=self.links, buffer_keys=self.buffer_keys, delay=self.delay))

    def encode(self, env):
        state, cfg = env.sim.state, self.cfg
        raw = asdict(state)
        buffers = {k: raw.pop(k) for k in ("urban_arrival_buffer", "urban_storage_release_buffer")}
        entries = []
        for name, value in numbers(raw):
            scale = (14400. if name == "/time_sec" else 200. if "density" in name else
                     100. if "speed" in name else 10000. if "flow" in name else
                     float(cfg.network.freeway_lanes) if "lanes" in name else 1000.)
            entries.append(("state" + name, value / scale))
        now = int(round(state.time_sec / cfg.simulation.T_u_sec))
        for kind, buffer in buffers.items():
            unexpected = set(buffer) - set(self.buffer_keys)
            if unexpected:
                raise ValueError("Unknown buffer keys: " + str(unexpected))
            for link in self.buffer_keys:
                slots = np.zeros(self.delay + 1)
                for timestamp, count in buffer.get(link, {}).items():
                    lag = int(timestamp) - now
                    if not 0 <= lag <= self.delay:
                        raise ValueError("Arrival outside fixed physical delay window")
                    slots[lag] += count / 1000.
                entries.extend((f"{kind}/{link}/{i}", float(v)) for i, v in enumerate(slots))
        for h, demand in enumerate(env.forecast):
            for field in ("freeway_mainline", "urban_boundary", "ramp_arrival"):
                entries.extend((f"forecast/{h}/{field}/{k}", v / 10000.)
                               for k, v in sorted(getattr(demand, field).items()))
            entries.append((f"forecast/{h}/incident_factor", demand.incident_capacity_factor))
            for link in cfg.network.freeway_links:
                losses = demand.freeway_lane_loss.get(link, {})
                for i in range(cfg.network.freeway_segments_per_link):
                    entries.append((f"forecast/{h}/lane_loss/{link}/{i}", float(losses.get(i, 0.))))
        coords = env.controller.lower.coords
        for label, control in (("previous", env.previous), ("pfo", env.controller.reference["control"])):
            for i, v in enumerate(coords.encode(control)):
                entries.append((f"{label}/coordinate/{i}", float(v)))
            for signal in cfg.network.signals:
                phase = ((state.time_sec - control.offsets.get(signal, 0.)) % cfg.network.cycle_length)
                entries.append((f"{label}/phase/{signal}", phase / cfg.network.cycle_length))
        values = {"incoming_dual": env.controller.incoming.ravel() / 100.,
                  "reference_budget": env.controller.reference["budget"] / [1000., 10000.],
                  "previous_requested": env.last_requested / [1000., 10000.],
                  "previous_executed": env.last_executed / [1000., 10000.],
                  "previous_slack": env.last_slack / [1000., 10000.],
                  "reference_TTT": [env.controller.reference["evaluation"].total_ttt / 1000.],
                  "remaining": [(80 - env.k) / 75.], "fallback": [float(env.last_fallback)]}
        for label, vector in values.items():
            entries.extend((f"memory/{label}/{i}", float(v)) for i, v in enumerate(vector))
        names = tuple(k for k, _ in entries)
        if self.names is None:
            self.names = names
        elif names != self.names:
            raise ValueError("Observation schema changed")
        out = np.array([v for _, v in entries], dtype=np.float32)
        if not np.isfinite(out).all():
            raise ValueError("Nonfinite observation")
        return out


class BudgetEnv:
    def __init__(self, runtime, reward_scale=100., scenario="sweet_170_incident_w", training_seed=None):
        self.rt, self.cfg = runtime, runtime["cfg"]
        self.folder, self.protocol, self.profile = protocol(runtime, scenario)
        self.reward_scale = float(reward_scale)
        if not np.isfinite(reward_scale) or reward_scale <= 0:
            raise ValueError("Invalid reward scale")
        self.training_seed = training_seed
        self.profile_hash = digest(read(self.folder / "forecast.json"))
        if training_seed is not None:
            # A declared training-only demand realization; evaluation keeps the frozen original.
            rows = read(self.folder / "forecast.json")
            rng = np.random.default_rng(training_seed)
            multipliers = {field: rng.uniform(.98, 1.02) for field in
                           ("freeway_mainline", "urban_boundary", "ramp_arrival")}
            for row in rows:
                for field, multiplier in multipliers.items():
                    row[field] = {k: v * multiplier for k, v in row[field].items()}
            self.profile = runtime["rc"].FrozenProfile(rows, self.cfg.simulation.T_c_sec)
            self.profile_hash = digest(rows)
        self.observer = Observation(self.cfg)
        self.warm = None

    def reset(self):
        self.close()
        rc = self.rt["rc"]
        self.sim = rc.MixedTrafficSimulator(self.cfg)
        if plain(self.sim.state) != read(self.folder / "initial_state.json"):
            raise RuntimeError("Initial state drift")
        self.controller = BudgetController(self.rt)
        self.previous = rc.ControlAction.uncontrolled(self.cfg)
        self.warm = rc.make_controller("WU-FAITHFUL-FOLLOWER", self.cfg)
        self.k = 0
        self.last_requested = np.zeros(2)
        self.last_executed = np.zeros(2)
        self.last_slack = np.zeros(2)
        self.last_fallback = False
        for k in range(5):
            forecast = self.profile.horizon(self.sim.state.time_sec, 3)
            self.sim.step(self.previous, forecast[0], k)
            self.k += 1
        self.warmup_ttt = self.sim.total_ttt
        self.prepare()
        return self._observe()

    def _observe(self):
        start, cpu = time.perf_counter(), time.process_time()
        obs = self.observer.encode(self)
        self.observation_timing = dict(observation_wall_seconds=time.perf_counter() - start,
                                       observation_cpu_seconds=time.process_time() - cpu)
        return obs

    def prepare(self):
        if self.k >= 80:
            raise RuntimeError("No decisions after terminal")
        start, cpu = time.perf_counter(), time.process_time()
        self.forecast = self.profile.horizon(self.sim.state.time_sec, 3)
        forecast_wall, forecast_cpu = time.perf_counter() - start, time.process_time() - cpu
        start, cpu = time.perf_counter(), time.process_time()
        warm_control = self.warm.solve(self.sim.state.copy(), None, self.forecast, self.previous).control
        pfo_wall, pfo_cpu = time.perf_counter() - start, time.process_time() - cpu
        start, cpu = time.perf_counter(), time.process_time()
        self.controller.prepare_reference(self.sim.state, self.forecast, self.previous, warm_control)
        self.reference_timing = dict(pfo_wall_seconds=pfo_wall, pfo_cpu_seconds=pfo_cpu,
                                     forecast_wall_seconds=forecast_wall, forecast_cpu_seconds=forecast_cpu,
                                     reference_wall_seconds=time.perf_counter() - start,
                                     reference_cpu_seconds=time.process_time() - cpu)

    def step(self, action, mode="rl", actor_seconds=0., actor_cpu_seconds=None):
        if self.k >= 80:
            raise RuntimeError("Step after true terminal")
        selected, audit = self.controller.evaluate_and_commit(action, mode)
        control = selected["control"]
        before = self.sim.total_ttt
        before_time = self.sim.state.time_sec
        start, cpu = time.perf_counter(), time.process_time()
        log = self.sim.step(control, self.forecast[0], self.k)
        plant_wall, plant_cpu = time.perf_counter() - start, time.process_time() - cpu
        delta = self.sim.total_ttt - before
        if not np.isfinite(delta) or delta < 0 or abs(delta - log.freeway_ttt - log.urban_ttt) > 1e-8:
            raise RuntimeError("Interval TTT accounting mismatch")
        self.k += 1
        terminal = self.k == 80
        if terminal and self.sim.state.time_sec != 14400.:
            raise RuntimeError("Terminal duration mismatch")
        self.previous = control.copy()
        self.last_requested = np.asarray(audit["B_requested"][0]).copy()
        self.last_executed = np.asarray(audit["B_executed"]).copy()
        self.last_slack = self.last_executed - audit["G_achieved"]
        self.last_fallback = audit["selection_source"] == "PFO_reference"
        audit.update(self.reference_timing)
        audit.update(self.observation_timing)
        audit.update(step=self.k - 1, control_step=self.k - 6, time_sec=self.sim.state.time_sec,
                     action_requested=np.asarray(action).copy(), interval_ttt=delta,
                     total_ttt=self.sim.total_ttt, freeway_ttt=self.sim.freeway_ttt,
                     urban_ttt=self.sim.urban_ttt, terminated=terminal, truncated=False,
                     actor_wall_seconds=actor_seconds, actor_cpu_seconds=actor_cpu_seconds,
                     plant_wall_seconds=plant_wall, plant_cpu_seconds=plant_cpu,
                     plant_log=copy.deepcopy(plain(log)), plant_state=plain(self.sim.state),
                     queue_near_capacity_estimate=self._queue_near_capacity_estimate(
                         self.sim.state.time_sec - before_time),
                     control=control.copy(),
                     inventory=self.rt["rc"].inventory(self.sim.state, self.cfg),
                     queue_state={key: copy.deepcopy(getattr(self.sim.state, key)) for key in
                                  ("ramp_queue", "mainline_origin_queue", "boundary_queue", "urban_movement_queue")})
        phases = ("forecast", "observation", "pfo", "reference", "lower", "guard")
        audit["decision_wall_seconds"] = sum(audit[p + "_wall_seconds"] for p in phases) + actor_seconds
        audit["controller_cpu_seconds"] = sum(audit[p + "_cpu_seconds"] for p in phases)
        audit["decision_cpu_seconds"] = (None if actor_cpu_seconds is None else
                                         audit["controller_cpu_seconds"] + actor_cpu_seconds)
        if terminal:
            obs = np.zeros(len(self.observer.names), dtype=np.float32)
        else:
            self.prepare()
            obs = self._observe()
        return obs, -delta / self.reward_scale, terminal, audit

    def _queue_near_capacity_estimate(self, interval_seconds):
        threshold = .9
        definitions = dict(ramp_queue="ramp_queue_max_veh", boundary_queue="boundary_queue_max_veh")
        result = dict(method="interval_endpoint_sampled", exact_substep_exposure=False,
                      threshold_fraction=threshold, interval_seconds=float(interval_seconds),
                      duration_units="s", capacity_units="veh",
                      capacity_definition={key: "cfg.network." + value for key, value in definitions.items()})
        for kind, attribute in definitions.items():
            capacity = float(getattr(self.cfg.network, attribute))
            result[kind] = {str(key): dict(queue_veh=float(value), capacity_veh=capacity,
                                          near_capacity_seconds=float(interval_seconds) if value >= threshold * capacity else 0.)
                            for key, value in getattr(self.sim.state, kind).items()}
        return result

    def contract(self):
        to_plain_dict = self.rt["rc"].to_plain_dict
        return plain(dict(version=2, reward_divisor=self.reward_scale, cfg=to_plain_dict(self.cfg),
                          options=to_plain_dict(self.rt["options"]), source_snapshot=self.rt["snapshot_identity"],
                          observation=self.observer.contract()))

    def checkpoint(self):
        # Full PFO object preserves solver memory, including hidden predictor caches.
        return dict(contract=self.contract(), sim=self.sim.copy(),
                    warm=copy.deepcopy(self.warm), previous=self.previous.copy(),
                    k=self.k, dual=self.controller.lower.dual.copy(),
                    last_budget=copy.deepcopy(self.controller.lower.last_budget),
                    last_requested=self.last_requested.copy(), last_executed=self.last_executed.copy(),
                    last_slack=self.last_slack.copy(), last_fallback=self.last_fallback,
                    warmup_ttt=self.warmup_ttt, profile_hash=self.profile_hash,
                    observer=copy.deepcopy(self.observer),
                    prepared=copy.deepcopy(self.controller.reference) if self.controller.prepared else None,
                    # The post-prepare warm state is saved; restoring must not call PFO twice.
                    forecast=copy.deepcopy(getattr(self, "forecast", None)),
                    observation_timing=copy.deepcopy(self.observation_timing),
                    reference_timing=copy.deepcopy(getattr(self, "reference_timing", None)))

    def restore(self, checkpoint):
        if checkpoint["profile_hash"] != self.profile_hash:
            raise ValueError("Checkpoint demand mismatch")
        saved, expected = checkpoint.get("contract"), self.contract()
        if not isinstance(saved, dict) or not isinstance(saved.get("observation"), dict):
            raise ValueError("Missing checkpoint environment contract")
        if self.observer.names is None:
            expected["observation"]["names"] = saved["observation"].get("names")
        if saved != expected:
            raise ValueError("Checkpoint environment contract mismatch")
        to_plain_dict = self.rt["rc"].to_plain_dict
        if (plain(to_plain_dict(checkpoint["sim"].cfg)) != expected["cfg"] or
                plain(to_plain_dict(self.rt["cfg"])) != expected["cfg"]):
            raise ValueError("Checkpoint simulator/controller config contract mismatch")
        ref = checkpoint["prepared"]
        if checkpoint["k"] < 80 and ref is None:
            raise ValueError("Missing prepared reference")
        # Derive the receiving schema from saved inputs without a plant/PFO/lower solve.
        observer = Observation(self.cfg)
        probe = SimpleNamespace(**{name: checkpoint[name] for name in (
            "sim", "previous", "forecast", "k", "last_requested", "last_executed",
            "last_slack", "last_fallback")})
        probe.controller = SimpleNamespace(incoming=checkpoint["dual"],
            lower=SimpleNamespace(coords=self.rt["coordinates"](self.cfg, self.rt["options"], probe.previous)),
            reference=ref if ref is not None else dict(control=probe.previous, budget=np.zeros(2),
                                                       evaluation=SimpleNamespace(total_ttt=0.)))
        observer.encode(probe)
        if observer.contract() != saved["observation"]:
            raise ValueError("Checkpoint observation contract mismatch")
        self.close()
        for name in ("sim", "warm", "previous", "k", "last_requested", "last_executed",
                     "last_slack", "last_fallback", "warmup_ttt", "forecast", "reference_timing", "observation_timing"):
            setattr(self, name, copy.deepcopy(checkpoint[name]))
        self.observer = observer
        self.controller = BudgetController(self.rt)
        self.controller.lower.dual = checkpoint["dual"].copy()
        self.controller.lower.last_budget = copy.deepcopy(checkpoint["last_budget"])
        if self.k < 80:
            self.controller.prepare_reference(self.sim.state, self.forecast, self.previous, ref["control"])
        return np.zeros(len(self.observer.names), dtype=np.float32) if self.k == 80 else self.observer.encode(self)

    def close(self):
        if self.warm is not None and hasattr(self.warm, "close"):
            self.warm.close()
        self.warm = None
