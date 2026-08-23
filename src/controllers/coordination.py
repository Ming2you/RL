from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, Mapping, Protocol, Sequence

import numpy as np

from src.models.demand import DemandStep
from src.models.state import ControlAction, ExperimentConfig, TrafficState, segment_vsl


ACTION_SCHEMA_VERSION = "coordination_action_v4"
OBSERVATION_SCHEMA_VERSION = "coordination_observation_v4"
RL_RESPONSE_CONTRACT_VERSION = "rl_pstack_b13_full_segment_vsl_certificate_v3"


@dataclass(frozen=True)
class CoordinationMask:
    budget: bool = True
    green: bool = True
    offset: bool = True
    metering: bool = True
    vsl: bool = True
    linear: bool = True
    quadratic: bool = True
    cross: bool = True

    @classmethod
    def named(cls, name: str) -> "CoordinationMask":
        key = name.strip().upper().replace("_", "-")
        masks = {
            "RL-FULL": cls(),
            "RL-LINEAR": cls(quadratic=False, cross=False),
            "RL-BUDGET": cls(green=False, offset=False, metering=False, vsl=False),
            "RL-URBAN": cls(metering=False, vsl=False),
            "RL-FREEWAY": cls(green=False, offset=False),
            "RL-NO-CROSS": cls(cross=False),
            "ZERO": cls(green=False, offset=False, metering=False, vsl=False),
        }
        if key not in masks:
            raise ValueError(f"unknown coordination mask: {name!r}")
        return masks[key]

    def as_array(self) -> np.ndarray:
        return np.asarray([
            self.budget,
            self.green,
            self.offset,
            self.metering,
            self.vsl,
            self.linear,
            self.quadratic,
            self.cross,
        ], dtype=np.float32)


@dataclass(frozen=True)
class PotentialBlock:
    owner: str
    reference: tuple[float, float]
    trust_radius: tuple[float, float]
    linear: tuple[float, float] = (0.0, 0.0)
    cholesky: tuple[float, float, float] = (0.0, 0.0, 0.0)
    lever_keys: tuple[str, str] = ("", "")

    def hessian(self) -> np.ndarray:
        l11, l21, l22 = self.cholesky
        lower = np.asarray([[l11, 0.0], [l21, l22]], dtype=float)
        return lower @ lower.T

    def value(self, values: Sequence[float], mask: CoordinationMask) -> float:
        delta = np.asarray(values, dtype=float) - np.asarray(self.reference, dtype=float)
        trust = np.maximum(np.asarray(self.trust_radius, dtype=float), 1.0e-9)
        z = delta / trust
        active = np.asarray([
            mask.green if self.lever_keys[0] == "green" else mask.metering,
            mask.offset if self.lever_keys[1] == "offset" else mask.vsl,
        ], dtype=float)
        z = z * active
        value = float(np.dot(np.asarray(self.linear, dtype=float), z)) if mask.linear else 0.0
        if mask.quadratic:
            hessian = self.hessian()
            if not mask.cross:
                hessian = np.diag(np.diag(hessian))
            value += 0.5 * float(z @ hessian @ z)
        return value


@dataclass(frozen=True)
class ScalarPotentialBlock:
    owner: str
    reference: float
    trust_radius: float
    linear: float = 0.0
    cholesky: float = 0.0
    lever_key: str = "vsl"

    def curvature(self) -> float:
        return float(self.cholesky * self.cholesky)

    def value(self, value: float, mask: CoordinationMask) -> float:
        active = mask.vsl if self.lever_key == "vsl" else False
        if not active:
            return 0.0
        z = (float(value) - float(self.reference)) / max(float(self.trust_radius), 1.0e-9)
        result = float(self.linear) * z if mask.linear else 0.0
        if mask.quadratic:
            result += 0.5 * self.curvature() * z * z
        return float(result)


@dataclass(frozen=True)
class CoordinationAction:
    N_P_star: float
    N_UF_star: float
    urban_blocks: tuple[PotentialBlock, ...] = ()
    freeway_blocks: tuple[PotentialBlock, ...] = ()
    mask: CoordinationMask = field(default_factory=CoordinationMask)
    schema_version: str = ACTION_SCHEMA_VERSION
    raw_budget: tuple[float, float] | None = None
    vsl_blocks: tuple[ScalarPotentialBlock, ...] = ()
    metering_release_certified: tuple[bool, ...] | None = None

    def with_budget(self, n_p: float, n_uf: float) -> "CoordinationAction":
        return replace(self, N_P_star=float(n_p), N_UF_star=float(n_uf))


class CoordinationActionProvider(Protocol):
    def provide(
        self,
        state: TrafficState,
        forecast: Sequence[DemandStep],
        previous: ControlAction,
        controller,
    ) -> CoordinationAction:
        ...


class StaticCoordinationProvider:
    def __init__(self, action: CoordinationAction):
        self.action = action

    def provide(self, state, forecast, previous, controller) -> CoordinationAction:
        return self.action


class CoordinationActionSchema:
    """Fixed full-action layout shared by collection, training, and evaluation."""

    def __init__(
        self,
        cfg: ExperimentConfig,
        *,
        linear_scale: float = 10.0,
        cholesky_scale: float = 2.0,
    ):
        self.cfg = cfg
        self.signals = tuple(cfg.network.signals)
        self.ramps = tuple(cfg.network.ramps)
        self.linear_scale = float(linear_scale)
        self.cholesky_scale = float(cholesky_scale)
        self.freeway_vsl_keys = {
            ramp: (
                f"{cfg.network.ramp_to_freeway[ramp]}__seg"
                f"{int(cfg.network.ramp_merge_segment_index.get(ramp, 0))}"
            )
            for ramp in self.ramps
        }
        self.all_vsl_keys = tuple(
            f"{link}__seg{index}"
            for link in cfg.network.freeway_links
            for index in range(cfg.network.freeway_segments_per_link)
        )
        merge_vsl_keys = set(self.freeway_vsl_keys.values())
        self.nonmerge_vsl_keys = tuple(
            key for key in self.all_vsl_keys if key not in merge_vsl_keys
        )
        self.represented_vsl_keys = self.all_vsl_keys
        names = ["budget.N_P", "budget.N_UF"]
        for signal in self.signals:
            names.extend(f"urban.{signal}.{term}" for term in ("g_green", "g_offset", "l11", "l21", "l22"))
        for ramp in self.ramps:
            names.extend(f"freeway.{ramp}.{term}" for term in ("g_meter", "g_vsl", "l11", "l21", "l22"))
        for key in self.nonmerge_vsl_keys:
            names.extend((f"vsl.{key}.g_vsl", f"vsl.{key}.sqrt_h"))
        self.certificate_ramps = self.ramps
        names.extend(f"certificate.{ramp}.release" for ramp in self.certificate_ramps)
        self.names = tuple(names)

    @property
    def dimension(self) -> int:
        return len(self.names)

    @property
    def urban_block_count(self) -> int:
        return len(self.signals)

    @property
    def freeway_block_count(self) -> int:
        return len(self.ramps)

    @property
    def vsl_block_count(self) -> int:
        return len(self.nonmerge_vsl_keys)

    def decode(
        self,
        raw_action: Sequence[float],
        previous: ControlAction,
        mask: CoordinationMask | None = None,
    ) -> CoordinationAction:
        raw = np.clip(np.asarray(raw_action, dtype=float).reshape(-1), -1.0, 1.0)
        if raw.size != self.dimension:
            raise ValueError(f"expected action dimension {self.dimension}, got {raw.size}")
        active_mask = mask or CoordinationMask()
        n_p_lo, n_p_hi = map(float, self.cfg.leader.N_P_star_range)
        n_uf_lo, n_uf_hi = map(float, self.cfg.leader.N_UF_star_range)
        n_p = n_p_lo + 0.5 * (raw[0] + 1.0) * (n_p_hi - n_p_lo)
        n_uf = n_uf_lo + 0.5 * (raw[1] + 1.0) * (n_uf_hi - n_uf_lo)
        idx = 2
        urban = []
        green_trust = 6.0
        offset_trust = max(float(self.cfg.network.cycle_length) / 8.0, 1.0)
        for signal in self.signals:
            coeff = raw[idx:idx + 5]
            idx += 5
            reference = (
                float(previous.green_times.get(f"{signal}_p1", self.cfg.network.effective_green_total / 2.0)),
                float(previous.offsets.get(signal, 0.0)),
            )
            urban.append(PotentialBlock(
                owner=signal,
                reference=reference,
                trust_radius=(green_trust, offset_trust),
                linear=tuple((coeff[:2] * self.linear_scale).tolist()),
                cholesky=(
                    float(max(coeff[2], 0.0) * self.cholesky_scale),
                    float(coeff[3] * self.cholesky_scale),
                    float(max(coeff[4], 0.0) * self.cholesky_scale),
                ),
                lever_keys=("green", "offset"),
            ))
        freeway = []
        for ramp in self.ramps:
            coeff = raw[idx:idx + 5]
            idx += 5
            cap = float(self.cfg.network.ramp_capacity_veh_h[ramp])
            vsl_key = self.freeway_vsl_keys[ramp]
            link = self.cfg.network.ramp_to_freeway[ramp]
            seg = int(self.cfg.network.ramp_merge_segment_index.get(ramp, 0))
            reference = (
                float(previous.ramp_metering.get(ramp, cap)),
                float(segment_vsl(previous, link, seg, self.cfg)),
            )
            freeway.append(PotentialBlock(
                owner=ramp,
                reference=reference,
                trust_radius=(
                    0.2 * cap,
                    max(float(self.cfg.freeway_follower.max_vsl_step), 1.0),
                ),
                linear=tuple((coeff[:2] * self.linear_scale).tolist()),
                cholesky=(
                    float(max(coeff[2], 0.0) * self.cholesky_scale),
                    float(coeff[3] * self.cholesky_scale),
                    float(max(coeff[4], 0.0) * self.cholesky_scale),
                ),
                lever_keys=("metering", vsl_key),
            ))
        vsl_blocks = []
        for key in self.nonmerge_vsl_keys:
            coeff = raw[idx:idx + 2]
            idx += 2
            link, segment_text = key.rsplit("__seg", 1)
            vsl_blocks.append(ScalarPotentialBlock(
                owner=key,
                reference=float(segment_vsl(previous, link, int(segment_text), self.cfg)),
                trust_radius=max(float(self.cfg.freeway_follower.max_vsl_step), 1.0),
                linear=float(coeff[0] * self.linear_scale),
                cholesky=float(max(coeff[1], 0.0) * self.cholesky_scale),
                lever_key="vsl",
            ))
        certificates = tuple(bool(value > 0.0) for value in raw[idx:idx + len(self.ramps)])
        idx += len(self.ramps)
        if not active_mask.metering:
            certificates = None
        if idx != self.dimension:
            raise RuntimeError(f"action schema decode mismatch: consumed {idx}, expected {self.dimension}")
        if not active_mask.budget:
            n_p = float(previous.N_P_star)
            n_uf = float(previous.N_UF_star)
        return CoordinationAction(
            N_P_star=float(n_p),
            N_UF_star=float(n_uf),
            urban_blocks=tuple(urban),
            freeway_blocks=tuple(freeway),
            mask=active_mask,
            raw_budget=(float(n_p), float(n_uf)),
            vsl_blocks=tuple(vsl_blocks),
            metering_release_certified=certificates,
        )

    def encode(self, action: CoordinationAction) -> np.ndarray:
        n_p_lo, n_p_hi = map(float, self.cfg.leader.N_P_star_range)
        n_uf_lo, n_uf_hi = map(float, self.cfg.leader.N_UF_star_range)
        values = [
            2.0 * (float(action.N_P_star) - n_p_lo) / max(n_p_hi - n_p_lo, 1.0e-9) - 1.0,
            2.0 * (float(action.N_UF_star) - n_uf_lo) / max(n_uf_hi - n_uf_lo, 1.0e-9) - 1.0,
        ]
        by_signal = {block.owner: block for block in action.urban_blocks}
        by_ramp = {block.owner: block for block in action.freeway_blocks}
        for owner in (*self.signals, *self.ramps):
            block = by_signal.get(owner) or by_ramp.get(owner)
            if block is None:
                values.extend([0.0] * 5)
                continue
            l11, l21, l22 = block.cholesky
            values.extend([
                block.linear[0] / self.linear_scale,
                block.linear[1] / self.linear_scale,
                l11 / self.cholesky_scale,
                l21 / self.cholesky_scale,
                l22 / self.cholesky_scale,
            ])
        by_vsl = {block.owner: block for block in action.vsl_blocks}
        for key in self.nonmerge_vsl_keys:
            block = by_vsl.get(key)
            if block is None:
                values.extend((0.0, 0.0))
            else:
                values.extend((
                    block.linear / self.linear_scale,
                    block.cholesky / self.cholesky_scale,
                ))
        certificates = action.metering_release_certified
        if certificates is None:
            values.extend([-1.0] * len(self.ramps))
        else:
            if len(certificates) != len(self.ramps):
                raise ValueError("metering release certificate count does not match ramps")
            values.extend(1.0 if value else -1.0 for value in certificates)
        return np.clip(np.asarray(values, dtype=np.float32), -1.0, 1.0)

    def metadata(self) -> dict:
        return {
            "version": ACTION_SCHEMA_VERSION,
            "dimension": self.dimension,
            "names": list(self.names),
            "signals": list(self.signals),
            "ramps": list(self.ramps),
            "freeway_vsl_keys": dict(self.freeway_vsl_keys),
            "all_vsl_keys": list(self.all_vsl_keys),
            "nonmerge_vsl_keys": list(self.nonmerge_vsl_keys),
            "vsl_block_count": self.vsl_block_count,
            "certificate_ramps": list(self.certificate_ramps),
            "linear_scale": self.linear_scale,
            "cholesky_scale": self.cholesky_scale,
        }


class CoordinationObservationSchema:
    def __init__(self, cfg: ExperimentConfig):
        self.cfg = cfg
        self.coupling_keys = tuple(
            [
                f"arr_{signal}_{phase}"
                for signal in cfg.network.signals
                for phase in ("p1", "p2")
            ]
            + [f"mainline_{link}" for link in cfg.network.freeway_links]
            + [f"rho_{link}" for link in cfg.network.freeway_links]
            + [f"u_on_{ramp}" for ramp in cfg.network.ramps]
        )
        self.memory_names = (
            "np_last_sum_nin",
            "np_bias_ratio",
            "np_prev_accum",
            "np_last_real_q",
            "np_corrector_lambda",
            "np_corrector_target",
        )
        names = ["time.phase", "time.peak", "time.recovery", "urban.total_veh"]
        names.extend(f"urban.queue.{movement}" for movement in cfg.network.urban_movements)
        names.extend(f"urban.storage_pressure.{link}" for link in cfg.network.urban_links)
        for link in cfg.network.freeway_links:
            names.extend((f"freeway.{link}.rho_mean", f"freeway.{link}.rho_max", f"freeway.{link}.speed_mean"))
            for index in range(cfg.network.freeway_segments_per_link):
                names.extend((
                    f"freeway.{link}.segment.{index}.rho",
                    f"freeway.{link}.segment.{index}.speed",
                    f"freeway.{link}.segment.{index}.effective_lanes",
                ))
        for ramp in cfg.network.ramps:
            names.extend((f"ramp.{ramp}.queue", f"ramp.{ramp}.meter", f"ramp.{ramp}.headroom"))
        names.extend(("forecast.first", "forecast.mean", "forecast.peak"))
        names.extend(f"forecast.{link}.lane_loss" for link in cfg.network.freeway_links)
        for link in cfg.network.freeway_links:
            names.extend(
                f"forecast.{link}.segment.{index}.lane_loss"
                for index in range(cfg.network.freeway_segments_per_link)
            )
        names.extend(("previous.N_P", "previous.N_UF", "previous.meter_total"))
        names.extend(f"previous.green.{signal}" for signal in cfg.network.signals)
        names.extend(f"previous.offset.{signal}" for signal in cfg.network.signals)
        for link in cfg.network.freeway_links:
            names.extend(
                f"previous.vsl.{link}.segment.{index}"
                for index in range(cfg.network.freeway_segments_per_link)
            )
        names.extend(("dual.lambda_P", "dual.lambda_UF"))
        names.append("follower.coupling.present")
        names.extend(f"follower.coupling.{key}" for key in self.coupling_keys)
        for name in self.memory_names:
            names.extend((f"follower.memory.{name}.present", f"follower.memory.{name}.value"))
        self.names = tuple(names)

    @property
    def dimension(self) -> int:
        return len(self.names)

    def observe(
        self,
        state: TrafficState,
        forecast: Sequence[DemandStep],
        previous: ControlAction,
        controller=None,
        total_steps: int | None = None,
    ) -> np.ndarray:
        cfg, net = self.cfg, self.cfg.network
        total = max(int(total_steps or max(cfg.simulation.T_total / cfg.simulation.control_interval, 1)), 1)
        step = float(state.time_sec / max(cfg.simulation.control_interval, 1.0e-9))
        phase = min(max(step / total, 0.0), 1.0)
        peak = float(900.0 <= state.time_sec < 5220.0)
        recovery = float(state.time_sec >= 5220.0)
        values = [phase, peak, recovery, state.total_urban_vehicles(net) / 1000.0]
        values.extend(float(state.urban_movement_queue.get(m, 0.0)) / 100.0 for m in net.urban_movements)
        for link in net.urban_links:
            cap = max(float(net.urban_link_storage_veh.get(link, 1.0)), 1.0)
            available = float(state.urban_link_storage.get(link, cap))
            values.append(min(max((cap - available) / cap, 0.0), 2.0))
        for link in net.freeway_links:
            rho = np.asarray(state.freeway_density.get(link, [0.0]), dtype=float)
            speed = np.asarray(state.freeway_speed.get(link, [net.v_free]), dtype=float)
            values.extend((float(rho.mean()) / net.rho_crit, float(rho.max()) / net.rho_crit, float(speed.mean()) / net.v_free))
            lanes = np.asarray(
                state.freeway_effective_lanes.get(
                    link, [float(net.freeway_lanes)] * net.freeway_segments_per_link,
                ),
                dtype=float,
            )
            for index in range(net.freeway_segments_per_link):
                values.extend((
                    float(rho[index] if index < rho.size else 0.0) / max(float(net.rho_crit), 1.0e-9),
                    float(speed[index] if index < speed.size else net.v_free) / max(float(net.v_free), 1.0e-9),
                    float(lanes[index] if index < lanes.size else net.freeway_lanes)
                    / max(float(net.freeway_lanes), 1.0),
                ))
        for ramp in net.ramps:
            cap = max(float(net.ramp_capacity_veh_h[ramp]), 1.0)
            queue = float(state.ramp_queue.get(ramp, 0.0))
            values.extend((
                queue / max(float(net.ramp_queue_max_veh), 1.0),
                float(previous.ramp_metering.get(ramp, cap)) / cap,
                max(0.0, float(net.ramp_queue_max_veh) - queue) / max(float(net.ramp_queue_max_veh), 1.0),
            ))
        totals = []
        for demand in forecast:
            totals.append(float(
                sum(demand.freeway_mainline.values())
                + sum(demand.urban_boundary.values())
                + sum(demand.ramp_arrival.values())
            ))
        totals = totals or [0.0]
        values.extend((totals[0] / 10000.0, float(np.mean(totals)) / 10000.0, max(totals) / 10000.0))
        lane_loss = {
            link: [0.0] * net.freeway_segments_per_link for link in net.freeway_links
        }
        for demand in forecast:
            for key, amount in getattr(demand, "freeway_lane_loss", {}).items():
                key_text = str(key)
                link = next((
                    candidate for candidate in net.freeway_links
                    if key_text == candidate
                    or key_text.startswith(f"{candidate}__")
                    or (isinstance(key, tuple) and key and key[0] == candidate)
                ), None)
                if link is None:
                    continue
                if isinstance(amount, Mapping):
                    entries = amount.items()
                elif isinstance(key, tuple) and len(key) > 1:
                    entries = ((key[1], amount),)
                elif "__seg" in key_text:
                    entries = ((key_text.rsplit("__seg", 1)[1], amount),)
                else:
                    entries = ((index, amount) for index in range(net.freeway_segments_per_link))
                for index, loss in entries:
                    index = int(index)
                    if 0 <= index < net.freeway_segments_per_link:
                        lane_loss[link][index] = max(
                            lane_loss[link][index], float(loss),
                        )
        for link in net.freeway_links:
            values.append(max(lane_loss[link], default=0.0) / max(float(net.freeway_lanes), 1.0))
        for link in net.freeway_links:
            values.extend(
                float(loss) / max(float(net.freeway_lanes), 1.0)
                for loss in lane_loss[link]
            )
        follower = getattr(controller, "nash_solver", None)
        values.extend((
            float(previous.N_P_star) / max(max(abs(v) for v in cfg.leader.N_P_star_range), 1.0),
            float(previous.N_UF_star) / max(float(cfg.leader.N_UF_star_range[1]), 1.0),
            sum(float(v) for v in previous.ramp_metering.values()) / max(net.total_ramp_capacity, 1.0),
        ))
        values.extend(
            float(previous.green_times.get(f"{signal}_p1", net.effective_green_total / 2.0))
            / max(float(net.effective_green_total), 1.0)
            for signal in net.signals
        )
        values.extend(
            (float(previous.offsets.get(signal, 0.0)) % float(net.cycle_length))
            / max(float(net.cycle_length), 1.0)
            for signal in net.signals
        )
        max_vsl = max(float(value) for value in cfg.freeway_follower.vsl_set)
        for link in net.freeway_links:
            values.extend(
                segment_vsl(previous, link, index, cfg) / max(max_vsl, 1.0)
                for index in range(net.freeway_segments_per_link)
            )
        values.extend((
            float(getattr(follower, "_lambda_P", 0.0)),
            float(getattr(follower, "_lambda_UF", 0.0)),
        ))
        coupling = getattr(follower, "_prev_coupling", None)
        values.append(float(coupling is not None))
        coupling = coupling or {}
        for key in self.coupling_keys:
            scale = float(net.rho_crit) if key.startswith("rho_") else 10000.0
            values.append(float(coupling.get(key, 0.0)) / max(scale, 1.0e-9))
        pending = getattr(follower, "_np_corrector_pending", None)
        memory = (
            (getattr(follower, "_np_last_sum_nin", None), 1000.0),
            (getattr(follower, "_np_bias_ratio", None), 1.0),
            (getattr(follower, "_np_prev_accum", None), 1000.0),
            (getattr(follower, "_np_last_real_q", None), 1000.0),
            (pending[0] if pending is not None else None, 1.0),
            (pending[1] if pending is not None else None, 1000.0),
        )
        for value, scale in memory:
            values.extend((float(value is not None), float(value or 0.0) / scale))
        result = np.asarray(values, dtype=np.float32)
        if result.size != self.dimension:
            raise RuntimeError(f"observation schema mismatch: expected {self.dimension}, got {result.size}")
        return result

    def metadata(self) -> dict:
        return {"version": OBSERVATION_SCHEMA_VERSION, "dimension": self.dimension, "names": list(self.names)}


class RLCoordinationProvider:
    def __init__(
        self,
        actor,
        action_schema: CoordinationActionSchema,
        observation_schema: CoordinationObservationSchema,
        mask: CoordinationMask | None = None,
    ):
        self.actor = actor
        self.action_schema = action_schema
        self.observation_schema = observation_schema
        self.mask = mask or CoordinationMask()
        self.last_observation: np.ndarray | None = None
        self.last_raw_action: np.ndarray | None = None

    def provide(self, state, forecast, previous, controller) -> CoordinationAction:
        obs = self.observation_schema.observe(state, forecast, previous, controller)
        raw = np.asarray(self.actor(obs), dtype=float).reshape(-1)
        self.last_observation = obs
        self.last_raw_action = raw
        return self.action_schema.decode(raw, previous, self.mask)


class CoordinationPotentialAdapter:
    """Translate normalized PSD blocks into the follower's local objective fields."""

    @staticmethod
    def clear(follower) -> None:
        for name in (
            "signal_marginal_price",
            "offset_marginal_price",
            "metering_marginal_price",
            "vsl_marginal_price",
            "green_offset_cross_price",
            "vsl_meter_cross_price",
            "signal_quadratic_price",
            "offset_quadratic_price",
            "metering_quadratic_price",
            "vsl_quadratic_price",
            "metering_release_certified",
        ):
            setattr(follower, name, None)
        follower.offset_directive = None

    def apply(self, action: CoordinationAction, follower) -> Dict[str, float]:
        self.clear(follower)
        follower.ramp_offset_enabled = True
        follower.priced_vsl_segment_candidates_enabled = True
        mask = action.mask
        signal_linear: Dict[str, float] = {}
        offset_linear: Dict[str, float] = {}
        signal_quad: Dict[str, float] = {}
        offset_quad: Dict[str, float] = {}
        go_cross: Dict[str, float] = {}
        signal_ref: Dict[str, float] = {}
        offset_ref: Dict[str, float] = {}
        go_ref: Dict[str, tuple] = {}
        for block in action.urban_blocks:
            tg, to = block.trust_radius
            h = block.hessian()
            if not mask.cross:
                h = np.diag(np.diag(h))
            signal_ref[block.owner] = block.reference[0]
            offset_ref[block.owner] = block.reference[1]
            go_ref[block.owner] = block.reference
            signal_linear[block.owner] = block.linear[0] / tg if mask.linear and mask.green else 0.0
            offset_linear[block.owner] = block.linear[1] / to if mask.linear and mask.offset else 0.0
            signal_quad[block.owner] = h[0, 0] / (tg * tg) if mask.quadratic and mask.green else 0.0
            offset_quad[block.owner] = h[1, 1] / (to * to) if mask.quadratic and mask.offset else 0.0
            go_cross[block.owner] = h[0, 1] / (tg * to) if mask.quadratic and mask.cross and mask.green and mask.offset else 0.0
        if action.urban_blocks and (mask.green or mask.offset):
            follower.signal_marginal_price = signal_linear
            follower.offset_marginal_price = offset_linear
            follower.signal_quadratic_price = signal_quad
            follower.offset_quadratic_price = offset_quad
            follower.green_offset_cross_price = go_cross
            follower.signal_marginal_price_ref = signal_ref
            follower.offset_marginal_price_ref = offset_ref
            follower.green_offset_cross_ref = go_ref
            follower.signal_marginal_price_trust_sec = max(block.trust_radius[0] for block in action.urban_blocks)
            follower.offset_marginal_price_trust_sec = max(block.trust_radius[1] for block in action.urban_blocks)
            follower.joint_green_offset_enabled = bool(mask.green and mask.offset)

        meter_linear: Dict[str, float] = {}
        vsl_linear: Dict[str, float] = {}
        meter_quad: Dict[str, float] = {}
        vsl_quad: Dict[str, float] = {}
        vm_cross: Dict[str, float] = {}
        meter_ref: Dict[str, float] = {}
        vsl_ref: Dict[str, float] = {}
        vm_ref: Dict[str, tuple] = {}
        for block in action.freeway_blocks:
            tm, tv = block.trust_radius
            h = block.hessian()
            if not mask.cross:
                h = np.diag(np.diag(h))
            vsl_key = block.lever_keys[1]
            meter_ref[block.owner] = block.reference[0]
            vsl_ref[vsl_key] = block.reference[1]
            vm_ref[block.owner] = block.reference
            meter_linear[block.owner] = block.linear[0] / tm if mask.linear and mask.metering else 0.0
            vsl_linear[vsl_key] = vsl_linear.get(vsl_key, 0.0) + (
                block.linear[1] / tv if mask.linear and mask.vsl else 0.0
            )
            meter_quad[block.owner] = h[0, 0] / (tm * tm) if mask.quadratic and mask.metering else 0.0
            vsl_quad[vsl_key] = vsl_quad.get(vsl_key, 0.0) + (
                h[1, 1] / (tv * tv) if mask.quadratic and mask.vsl else 0.0
            )
            vm_cross[block.owner] = h[0, 1] / (tm * tv) if mask.quadratic and mask.cross and mask.metering and mask.vsl else 0.0
        for block in action.vsl_blocks:
            trust = max(float(block.trust_radius), 1.0e-9)
            vsl_ref[block.owner] = float(block.reference)
            vsl_linear[block.owner] = (
                float(block.linear) / trust if mask.linear and mask.vsl else 0.0
            )
            vsl_quad[block.owner] = (
                block.curvature() / (trust * trust) if mask.quadratic and mask.vsl else 0.0
            )
        if (action.freeway_blocks or action.vsl_blocks) and (mask.metering or mask.vsl):
            follower.metering_marginal_price = meter_linear
            follower.vsl_marginal_price = vsl_linear
            follower.metering_quadratic_price = meter_quad
            follower.vsl_quadratic_price = vsl_quad
            follower.vsl_meter_cross_price = vm_cross
            follower.metering_marginal_price_ref = meter_ref
            follower.vsl_marginal_price_ref = vsl_ref
            follower.vsl_meter_cross_ref = vm_ref
            follower.metering_marginal_price_trust_frac = 0.2
            trust_radii = [block.trust_radius[1] for block in action.freeway_blocks]
            trust_radii.extend(block.trust_radius for block in action.vsl_blocks)
            follower.vsl_marginal_price_trust_kmh = max(trust_radii, default=None)
            follower.metering_release_certified = (
                {
                    ramp: bool(certified)
                    for ramp, certified in zip(
                        (block.owner for block in action.freeway_blocks),
                        action.metering_release_certified,
                    )
                }
                if mask.metering and action.metering_release_certified is not None
                else None
            )

        return {
            "coordination_native_price_disabled": 1.0,
            "coordination_urban_block_count": float(len(action.urban_blocks)),
            "coordination_freeway_block_count": float(len(action.freeway_blocks)),
            "coordination_vsl_block_count": float(len(action.vsl_blocks)),
            "coordination_linear_active": float(mask.linear),
            "coordination_quadratic_active": float(mask.quadratic),
            "coordination_cross_active": float(mask.cross),
            "coordination_ramp_offset_active": 1.0,
            "coordination_priced_vsl_candidates_active": 1.0,
        }
