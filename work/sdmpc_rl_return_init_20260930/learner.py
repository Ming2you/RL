"""Fixed 1000/250/10 return-initialized proposal; critics retain carry continuation."""
import copy
from runtime import SPEC, SCENARIOS, np, torch, finite, tensor_hash
from data import sample, projected, network_budget


def mlp(inputs, outputs, zero=False):
    net = torch.nn.Sequential(torch.nn.Linear(inputs, 64), torch.nn.ReLU(),
        torch.nn.Linear(64, 64), torch.nn.ReLU(), torch.nn.Linear(64, outputs))
    if zero:
        torch.nn.init.zeros_(net[-1].weight)
        torch.nn.init.zeros_(net[-1].bias)
    return net


class Actor(torch.nn.Module):
    def __init__(self, observations=2367):
        super().__init__()
        self.net = mlp(observations, 2, zero=True)
        self.register_buffer("bounds", torch.tensor([.2, .1], dtype=torch.float32))

    def forward(self, obs):
        return self.bounds * torch.tanh(self.net(obs))


def residuals(heads, obs, budget):
    inputs = torch.cat((obs, budget), dim=-1)
    return torch.cat([head(inputs) for head in heads], dim=-1)


def phi_errors(phi, data):
    with torch.no_grad():
        values = phi(data["obs"]).flatten().double()
        finite(values)
        errors = (values - data["returns"]).square()
        carry = ~data["local"]
        return dict(pooled=float(errors[carry].mean()), per_scenario={name:
            float(errors[carry & (data["scenario"] == i)].mean()) for i, name in enumerate(SCENARIOS)})


def numerical_gate(initial, final):
    finite((initial, final))
    checks = dict(pooled_halved=final["pooled"] <= .5 * initial["pooled"],
        each_scenario_decreased=all(final["per_scenario"][s] < initial["per_scenario"][s] for s in SCENARIOS))
    return dict(passed=all(checks.values()), checks=checks, initial=initial, final=final,
        meaning="Numerical fit to training carry returns only; no traffic comparison or heldout calibration")


class Learner:
    def __init__(self, data):
        self.data = data
        self.rng = np.random.default_rng(7200)
        # Model initialization has its own saved Torch stream and does not consume caller RNG.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(7200)
            self.actor = Actor(data["obs"].shape[1])
            self.phi = mlp(data["obs"].shape[1], 1)
            self.critics = torch.nn.ModuleList([mlp(data["obs"].shape[1]+2, 1, zero=True) for _ in range(2)])
            self.torch_rng = torch.get_rng_state().clone()
        self.targets = copy.deepcopy(self.critics).requires_grad_(False)
        self.optimizers = {key: torch.optim.Adam(model.parameters(), lr=3e-4)
                           for key, model in self.models().items() if key != "targets"}
        self.phase = "phi"
        self.counts = dict(phi=0, critic=0, actor=0, polyak=0)
        self.losses = dict(phi=[], critic=[], actor=[])
        self.metrics = dict(phi_initial=phi_errors(self.phi, data),
            zero_actor_sha256=tensor_hash(self.actor.state_dict()),
            zero_action_sha256=tensor_hash({"u": self.actor(data["obs"]).detach()}))
        self.freeze_phase()

    def models(self):
        return dict(actor=self.actor, phi=self.phi, critics=self.critics, targets=self.targets)

    def freeze_phase(self):
        for name, model in self.models().items():
            active = ((name == "phi" and self.phase == "phi") or
                      (name == "critics" and self.phase == "critic") or
                      (name == "actor" and self.phase == "actor"))
            model.requires_grad_(active)
            model.zero_grad(set_to_none=True)

    @torch.no_grad()
    def target(self, batch):
        phi = self.phi(batch["obs"]).flatten().double()
        target = batch["returns"] - phi
        local = batch["local"]
        target[local] = batch["reward"][local] - phi[local]
        # Index true nonterminals BEFORE any next-state or next-anchor network call.
        live = local & ~batch["terminal"]
        if live.any():
            obs = batch["next_obs"][live]
            anchor = batch["next_anchor"][live]
            budget = network_budget(anchor, torch.zeros_like(anchor, dtype=torch.float32))
            value = self.phi(obs).flatten().double()
            advantage = residuals(self.targets, obs, budget).min(dim=1).values.double()
            target[live] += value + advantage
        finite(target)
        return target.detach()

    def update(self):
        if self.phase in ("done", "gate_failed"):
            raise ValueError("No updates after the fixed proposal or failed gate")
        if self.phase == "diagnostics":
            self.metrics["before_actor"] = diagnostics(self)
            self.metrics["gate"] = numerical_gate(self.metrics["phi_initial"], phi_errors(self.phi, self.data))
            self.metrics["critic_frozen_sha256"] = tensor_hash(self.critics.state_dict())
            self.metrics["targets_frozen_sha256"] = tensor_hash(self.targets.state_dict())
            self.phase = "actor" if self.metrics["gate"]["passed"] else "gate_failed"
            self.freeze_phase()
            self.validate()
            return
        phase = self.phase
        batch = sample(self.data, self.rng, phi=phase == "phi")
        if phase == "phi":
            loss = (self.phi(batch["obs"]).flatten().double() - batch["returns"]).square().mean()
            optimizer = self.optimizers["phi"]
        elif phase == "critic":
            values = residuals(self.critics, batch["obs"], network_budget(batch["anchor"], batch["action"]))
            loss = (values.double() - self.target(batch)[:, None]).square().mean()
            optimizer = self.optimizers["critics"]
        else:
            action = self.actor(batch["obs"])
            values = residuals(self.critics, batch["obs"], network_budget(batch["anchor"], action))[:, 0]
            loss = -(self.phi(batch["obs"]).flatten() + values).mean()
            optimizer = self.optimizers["actor"]
        finite(loss)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        for model in self.models().values():
            for parameter in model.parameters():
                if parameter.grad is not None:
                    finite(parameter.grad)
        optimizer.step()
        finite([m.state_dict() for m in self.models().values()])
        finite(optimizer.state_dict())
        self.counts[phase] += 1
        self.losses[phase].append(float(loss.detach()))
        if phase == "critic" and self.counts[phase] % 2 == 0:
            with torch.no_grad():
                for target, online in zip(self.targets.parameters(), self.critics.parameters()):
                    target.mul_(1. - .005).add_(online, alpha=.005)
            self.counts["polyak"] += 1
        if phase == "phi" and self.counts[phase] == SPEC["phi_updates"]:
            self.metrics["phi_frozen_sha256"] = tensor_hash(self.phi.state_dict())
            self.targets.load_state_dict(self.critics.state_dict())
            self.metrics["initial_residual_sha256"] = tensor_hash(self.critics.state_dict())
            self.metrics["initial_targets_sha256"] = tensor_hash(self.targets.state_dict())
            self.phase = "critic"
        elif phase == "critic" and self.counts[phase] == SPEC["critic_updates"]:
            self.phase = "diagnostics"
        elif phase == "actor" and self.counts[phase] == SPEC["actor_updates"]:
            self.phase = "done"
            self.metrics["final_action_audit"] = action_audit(self, self.actor(self.data["obs"]).detach())
            self.metrics["final_actor_sha256"] = tensor_hash(self.actor.state_dict())
        self.freeze_phase()
        self.validate()

    def validate(self):
        finite([m.state_dict() for m in self.models().values()])
        finite((self.metrics, self.losses, [o.state_dict() for o in self.optimizers.values()]))
        p, c, a = (self.counts[k] for k in ("phi", "critic", "actor"))
        limits = SPEC["phi_updates"], SPEC["critic_updates"], SPEC["actor_updates"]
        if any(type(x) is not int or not 0 <= x <= n for x, n in zip((p, c, a), limits)):
            raise ValueError("Fixed update limit violated")
        valid = {"phi": p < limits[0] and c == a == 0,
            "critic": p == limits[0] and c < limits[1] and a == 0,
            "diagnostics": (p, c, a) == (*limits[:2], 0),
            "actor": (p, c) == limits[:2] and a < limits[2],
            "done": (p, c, a) == limits,
            "gate_failed": (p, c, a) == (*limits[:2], 0)}
        if not valid.get(self.phase, False) or self.counts["polyak"] != c//2:
            raise ValueError("Phase/counter contract mismatch")
        for key in ("phi", "critic", "actor"):
            if len(self.losses[key]) != self.counts[key]:
                raise ValueError("Loss/counter contract mismatch")
        if self.phase != "phi" and tensor_hash(self.phi.state_dict()) != self.metrics["phi_frozen_sha256"]:
            raise ValueError("Frozen Phi changed")
        if not a and tensor_hash(self.actor.state_dict()) != self.metrics["zero_actor_sha256"]:
            raise ValueError("Actor changed before proposal")
        if self.phase in ("actor", "done", "gate_failed"):
            for key, model in (("critic", self.critics), ("targets", self.targets)):
                if tensor_hash(model.state_dict()) != self.metrics[key+"_frozen_sha256"]:
                    raise ValueError("Critic continuation weights changed during proposal")
            if self.metrics["gate"]["passed"] != (self.phase != "gate_failed"):
                raise ValueError("Numerical gate/phase mismatch")

    def state(self):
        self.validate()
        return copy.deepcopy(dict(models={k: v.state_dict() for k, v in self.models().items()},
            optimizers={k: v.state_dict() for k, v in self.optimizers.items()},
            numpy_rng=self.rng.bit_generator.state, torch_rng=self.torch_rng,
            counts=self.counts, losses=self.losses, metrics=self.metrics, phase=self.phase,
            critic_continuation="carry", spec=SPEC))

    def load(self, state):
        if state["spec"] != SPEC or state["critic_continuation"] != "carry":
            raise ValueError("Checkpoint specification/continuation mismatch")
        for key, model in self.models().items():
            model.load_state_dict(state["models"][key])
        for key, optimizer in self.optimizers.items():
            optimizer.load_state_dict(state["optimizers"][key])
        self.rng.bit_generator.state = copy.deepcopy(state["numpy_rng"])
        self.torch_rng = state["torch_rng"].clone()
        self.phase, self.counts, self.losses, self.metrics = copy.deepcopy(
            (state["phase"], state["counts"], state["losses"], state["metrics"]))
        self.freeze_phase()
        self.validate()


def action_audit(learner, action):
    data = learner.data
    with torch.no_grad():
        budgets = projected(data["anchor"], action)
        normalized = network_budget(data["anchor"], action)
        phi = learner.phi(data["obs"])
        q = phi + residuals(learner.critics, data["obs"], normalized)
        raw = data["anchor"] + action.double() * torch.tensor([50., 1000.], dtype=torch.float64)
        finite((budgets, q))
    # Gradients describe this saved state/action only, with physical projection included.
    with torch.enable_grad():
        variable = action.detach().clone().requires_grad_(True)
        values = residuals(learner.critics, data["obs"], network_budget(data["anchor"], variable))
        gradients = [torch.autograd.grad(value.sum(), variable, retain_graph=True)[0]
                     for value in (values[:, 0], values[:, 1], values.min(dim=1).values)]
    finite(gradients)
    return dict(action_sha256=tensor_hash({"u": action}), request_sha256=tensor_hash({"b": budgets}),
        network_input_sha256=tensor_hash({"b_normalized": normalized}),
        output_sha256=tensor_hash({"phi": phi, "q": q}),
        nominal_saturation=(action.abs() >= torch.tensor([.2, .1]) * .999).sum(dim=0).tolist(),
        projection_clipped=int((raw[:, 1] != budgets[:, 1]).sum()),
        projection_boundary=int(((budgets[:, 1] == 0) | (budgets[:, 1] == 6000)).sum()),
        actions=action.tolist(), requests=budgets.tolist(), q1_q2=q.tolist(),
        min_q=q.min(dim=1).values.tolist(), dq_du=torch.stack(gradients, dim=1).tolist(),
        gradient_heads=["Q1", "Q2", "minQ"],
        interpretation="Saved training states only; small incremental bounds do not bound episode-wide drift")


def diagnostics(learner):
    data = learner.data
    with torch.no_grad():
        phi = learner.phi(data["obs"]).flatten().double()
        advantage = residuals(learner.critics, data["obs"], network_budget(data["anchor"], data["action"])).double()
        q = phi[:, None] + advantage
        q = torch.cat((q, q.min(dim=1, keepdim=True).values), dim=1)
        target = learner.target(data)
        own_error = q - (phi + target)[:, None]
        return_error = q - data["returns"][:, None]
        phi_error = phi - data["returns"]
        finite((q, phi, target, own_error, return_error))
    rows, groups = [], []
    for i in range(len(phi)):
        rows.append(dict(scenario=SCENARIOS[int(data["scenario"][i])],
            behavior="local" if data["local"][i] else "carry", horizon=int(data["horizon"][i]),
            terminal=bool(data["terminal"][i]), phi=float(phi[i]), behavior_return=float(data["returns"][i]),
            phi_minus_behavior_return=float(phi_error[i]), q1_q2_min=q[i].tolist(),
            q_minus_behavior_return=return_error[i].tolist(), residual_target=float(target[i]),
            own_target_residual=own_error[i].tolist(),
            terminal_exact_error=return_error[i].tolist() if data["terminal"][i] else None))
    for scenario, name in enumerate(SCENARIOS):
        for local in (False, True):
            for low, high in ((1, 75), (51, 75), (26, 50), (1, 25), (1, 1)):
                mask = ((data["scenario"] == scenario) & (data["local"] == local) &
                        (data["horizon"] >= low) & (data["horizon"] <= high))
                if mask.any():
                    groups.append(dict(scenario=name, behavior="local" if local else "carry",
                        horizon_range=[low, high], count=int(mask.sum()),
                        phi_behavior_return_mse=float(phi_error[mask].square().mean()),
                        q_behavior_return_mse=return_error[mask].square().mean(dim=0).tolist(),
                        own_target_mse=own_error[mask].square().mean(dim=0).tolist(),
                        max_abs_own_target_error=own_error[mask].abs().max(dim=0).values.tolist()))
    zero = torch.zeros_like(data["action"])
    probes = torch.tensor([[0., 0.], [.2, 0.], [-.2, 0.], [0., .1], [0., -.1],
                           [.2, .1], [.2, -.1], [-.2, .1], [-.2, -.1]], dtype=torch.float32)
    with torch.no_grad():
        requests = projected(data["anchor"][:, None, :], probes[None, :, :])
        normalized = (requests / requests.new_tensor([1000., 10000.])).float()
        obs = data["obs"][:, None, :].expand(-1, len(probes), -1)
        values = residuals(learner.critics, obs, normalized) + phi.float()[:, None, None]
        aliases = []
        for i in range(len(phi)):
            exact, rounded = [], []
            for a in range(len(probes)):
                for b in range(a+1, len(probes)):
                    if torch.equal(normalized[i, a], normalized[i, b]):
                        rounded.append([a, b])
                    if torch.equal(requests[i, a], requests[i, b]):
                        exact.append([a, b])
                        if not torch.equal(values[i, a], values[i, b]):
                            raise ValueError("Projected action alias has unequal Q")
            aliases.append(dict(row=i, exact_projected_aliases=exact, network_float32_aliases=rounded))
        finite(values)
    return dict(rows=rows, per_scenario_horizon=groups, heads=["Q1", "Q2", "minQ"],
        zero_action=action_audit(learner, zero), replay_action=action_audit(learner, data["action"]),
        probes=probes.tolist(), probe_requests=requests.tolist(), probe_q1_q2=values.tolist(),
        probe_min_q=values.min(dim=-1).values.tolist(), projection_aliases=aliases,
        labels="Carry G and Phi fits are sampled training-only V-carry labels, never Q* or heldout values. "
               "Local behavior returns use the recorded exploration continuation; their discrepancy from "
               "carry-continuation Q is not a calibrated Q error. True terminal returns are exact rewards. "
               "Critics remain carry-continuation critics and are not Q of the proposed actor.")
