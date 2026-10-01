"""Bounded on-policy MC evaluation of one fixed actor, with continued critic Adam."""
import copy
import hashlib
from mc_common import (np, torch, SPEC, SCENARIOS, ANCESTOR_COUNTS, MODEL_SHA, rt,
    Actor, mlp, residuals, projected, finite, tensor_hash)
from mc_data import sample_indices


def state_hash(value):
    """Stable optimizer/RNG identity independent of Torch archive serialization."""
    h = hashlib.sha256()

    def visit(item):
        if isinstance(item, torch.Tensor):
            h.update(tensor_hash({"tensor": item}).encode())
        elif isinstance(item, dict):
            for key in sorted(item, key=str):
                visit(key)
                visit(item[key])
        elif isinstance(item, (tuple, list)):
            h.update(type(item).__name__.encode())
            for child in item:
                visit(child)
        else:
            h.update(repr((type(item).__name__, item)).encode())
    visit(value)
    return h.hexdigest()


class Learner:
    def __init__(self, data, parent):
        if (parent["spec"] != rt.SPEC or parent["phase"] != "done" or
                parent["counts"] != ANCESTOR_COUNTS or parent["critic_continuation"] != "carry"):
            raise ValueError("Parent continuation/specification/counters differ")
        finite((data, parent))
        self.data = data
        self.rng = np.random.Generator(np.random.PCG64(7201))
        width = data["obs"].shape[1]
        with torch.device("meta"):
            self.actor, self.phi = Actor(width), mlp(width, 1)
            self.critics = torch.nn.ModuleList([mlp(width+2, 1) for _ in range(2)])
            self.targets = torch.nn.ModuleList([mlp(width+2, 1) for _ in range(2)])
        for name, model in self.models().items():
            model.load_state_dict(copy.deepcopy(parent["models"][name]), strict=True, assign=True)
            model.requires_grad_(name == "critics")
            model.eval()
        self.initial_hashes = {name: tensor_hash(model.state_dict()) for name, model in self.models().items()}
        self.optimizer = torch.optim.Adam(self.critics.parameters(), lr=SPEC["lr"])
        old_optimizer = parent["optimizers"]["critics"]
        self.optimizer.load_state_dict(copy.deepcopy(old_optimizer))
        if state_hash(self.optimizer.state_dict()) != state_hash(old_optimizer):
            raise ValueError("Matching critic optimizer was not restored exactly")
        self.optimizer_provenance = dict(parent_model_sha256=MODEL_SHA, key="learner.optimizers.critics",
            parent_state_sha256=state_hash(old_optimizer), continued=True,
            ancestor_critic_steps=250, lr=3e-4)
        self.counts = dict(mc_critic=0)
        self.phase = "mc_fit"
        self.losses = []
        self.sample_counts = [0]*5
        self.forced_terminal_counts = [0]*5
        self.terminal_counts = [0]*5
        with torch.no_grad():
            self.phi_values = self.phi(data["obs"]).flatten().double()
            self.mc_targets = data["returns"] - self.phi_values
            self.budgets = (data["request"] / data["request"].new_tensor([1000., 10000.])).float()
        self.before = diagnostics(self, "before")
        self.validate()

    def models(self):
        return dict(actor=self.actor, phi=self.phi, critics=self.critics, targets=self.targets)

    def update(self):
        self.validate()
        if self.phase != "mc_fit":
            raise ValueError("Exactly 250 MC updates; completed learner is immutable")
        indices, forced = sample_indices(self.data, self.rng)
        values = residuals(self.critics, self.data["obs"][indices], self.budgets[indices]).double()
        errors = values - self.mc_targets[indices, None]
        head_mse = errors.square().mean(dim=0)
        loss = head_mse.sum()
        finite((values, self.mc_targets, loss))
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        for model in self.models().values():
            for parameter in model.parameters():
                if parameter.grad is not None:
                    finite(parameter.grad)
        self.optimizer.step()
        self.counts["mc_critic"] += 1
        per_scenario, forced_counts, terminals = [], [], []
        scenario_ids = self.data["scenario"][indices]
        for i in range(5):
            mask = scenario_ids == i
            n, f, t = int(mask.sum()), int(forced[mask.numpy()].sum()), int(self.data["terminal"][indices][mask].sum())
            self.sample_counts[i] += n
            self.forced_terminal_counts[i] += f
            self.terminal_counts[i] += t
            per_scenario.append(n)
            forced_counts.append(f)
            terminals.append(t)
        self.losses.append(dict(update=self.counts["mc_critic"], loss=float(loss.detach()),
            head_mse=head_mse.detach().tolist(), indices=indices.tolist(),
            forced_terminal=forced.tolist(), samples_per_scenario=per_scenario,
            forced_terminals_per_scenario=forced_counts, terminals_per_scenario=terminals,
            finite=True))
        if self.counts["mc_critic"] == 250:
            self.targets.load_state_dict(self.critics.state_dict())
            self.phase = "done"
        self.validate()

    def validate(self):
        n = self.counts["mc_critic"]
        if type(n) is not int or not 0 <= n <= 250 or self.phase != ("done" if n == 250 else "mc_fit"):
            raise ValueError("MC phase/counter mismatch")
        if (len(self.losses) != n or self.sample_counts != [8*n]*5 or
                self.forced_terminal_counts != [n]*5):
            raise ValueError("MC loss/quota mismatch")
        finite((self.mc_targets, self.before, self.losses, self.optimizer.state_dict(),
                [m.state_dict() for m in self.models().values()]))
        for name in ("actor", "phi"):
            model = self.models()[name]
            if (tensor_hash(model.state_dict()) != self.initial_hashes[name] or
                    any(p.requires_grad or p.grad is not None for p in model.parameters())):
                raise ValueError("Frozen actor/Phi changed or received gradients")
        target_hash = tensor_hash(self.targets.state_dict())
        expected = tensor_hash(self.critics.state_dict()) if n == 250 else self.initial_hashes["targets"]
        if target_hash != expected or any(p.requires_grad or p.grad is not None for p in self.targets.parameters()):
            raise ValueError("Targets updated during MC fit or final copy differs")
        groups = self.optimizer.param_groups
        if len(groups) != 1 or groups[0]["lr"] != 3e-4:
            raise ValueError("Continued Adam learning rate/groups mismatch")
        if {id(p) for p in groups[0]["params"]} != {id(p) for p in self.critics.parameters()}:
            raise ValueError("Only critic parameters may be optimized")
        if len(self.optimizer.state) != len(list(self.critics.parameters())):
            raise ValueError("Missing matching parent Adam state")
        for parameter, state in self.optimizer.state.items():
            if (float(state["step"]) != 250+n or state["exp_avg"].shape != parameter.shape or
                    state["exp_avg_sq"].shape != parameter.shape):
                raise ValueError("Continued Adam step/moment provenance differs")

    def state(self):
        self.validate()
        return copy.deepcopy(dict(format=SPEC["format"], spec=SPEC, phase=self.phase, counts=self.counts,
            ancestor_counts=ANCESTOR_COUNTS, models={k: m.state_dict() for k, m in self.models().items()},
            critic_optimizer=self.optimizer.state_dict(), optimizer_provenance=self.optimizer_provenance,
            numpy_rng=self.rng.bit_generator.state, initial_hashes=self.initial_hashes,
            losses=self.losses, sample_counts=self.sample_counts,
            forced_terminal_counts=self.forced_terminal_counts, terminal_counts=self.terminal_counts,
            before=self.before, critic_continuation=SPEC["critic_continuation"],
            target_policy_actor_sha256=self.initial_hashes["actor"],
            parent_model_sha256=MODEL_SHA, data_arrays_sha256=tensor_hash(self.data)))

    def load(self, state):
        if (state["format"] != SPEC["format"] or state["spec"] != SPEC or
                state["ancestor_counts"] != ANCESTOR_COUNTS or
                state["initial_hashes"] != self.initial_hashes or
                state["optimizer_provenance"] != self.optimizer_provenance or
                state["target_policy_actor_sha256"] != self.initial_hashes["actor"] or
                state["parent_model_sha256"] != MODEL_SHA or
                state["data_arrays_sha256"] != tensor_hash(self.data) or
                state["critic_continuation"] != SPEC["critic_continuation"] or state["before"] != self.before):
            raise ValueError("MC checkpoint source/data/model/continuation identity differs")
        for name, model in self.models().items():
            model.load_state_dict(state["models"][name], strict=True)
        self.optimizer.load_state_dict(copy.deepcopy(state["critic_optimizer"]))
        self.rng.bit_generator.state = copy.deepcopy(state["numpy_rng"])
        for key in ("phase", "counts", "losses", "sample_counts", "forced_terminal_counts", "terminal_counts"):
            setattr(self, key, copy.deepcopy(state[key]))
        # Reconcile retained sampling evidence, including incidental uniform terminal draws.
        rng = np.random.Generator(np.random.PCG64(7201))
        totals = [0]*5
        for number, loss in enumerate(self.losses, 1):
            indices, forced = sample_indices(self.data, rng)
            counts = [int(self.data["terminal"][indices][self.data["scenario"][indices] == i].sum()) for i in range(5)]
            if (loss["update"] != number or loss["indices"] != indices.tolist() or
                    loss["forced_terminal"] != forced.tolist() or loss["samples_per_scenario"] != [8]*5 or
                    loss["forced_terminals_per_scenario"] != [1]*5 or loss["terminals_per_scenario"] != counts or
                    loss["finite"] is not True or loss["loss"] != sum(loss["head_mse"])):
                raise ValueError("Retained sampling/loss evidence differs")
            totals = [a+b for a, b in zip(totals, counts)]
        if self.rng.bit_generator.state != rng.bit_generator.state or totals != self.terminal_counts:
            raise ValueError("Retained PCG64/terminal counts differ")
        self.validate()


def error_metrics(errors):
    return dict(mse=errors.square().mean(dim=0).tolist(), mae=errors.abs().mean(dim=0).tolist(),
        signed_bias=errors.mean(dim=0).tolist(), max_abs_error=errors.abs().max(dim=0).values.tolist())


@torch.no_grad()
def diagnostics(learner, phase):
    data = learner.data
    heads = learner.phi_values[:, None] + residuals(learner.critics, data["obs"], learner.budgets).double()
    q = torch.cat((heads, heads.min(dim=1, keepdim=True).values), dim=1)
    errors = q - data["returns"][:, None]
    raw = data["anchor"] + data["action"].double() * data["anchor"].new_tensor([50., 1000.])
    finite((q, errors, raw))
    rows, groups = [], []
    for i in range(len(q)):
        rows.append(dict(scenario=SCENARIOS[int(data["scenario"][i])], horizon=int(data["horizon"][i]),
            terminal=bool(data["terminal"][i]), G_pi1=float(data["returns"][i]),
            reward=float(data["reward"][i]), phi=float(learner.phi_values[i]),
            residual_target=float(learner.mc_targets[i]), q1_q2_min=q[i].tolist(),
            errors=errors[i].tolist(), terminal_reward_errors=(q[i]-data["reward"][i]).tolist()
                if data["terminal"][i] else None,
            action=data["action"][i].tolist(), anchor=data["anchor"][i].tolist(), request=data["request"][i].tolist()))
    for scenario, name in enumerate(SCENARIOS):
        for low, high in ((1, 75), (51, 75), (26, 50), (1, 25), (1, 1)):
            mask = (data["scenario"] == scenario) & (data["horizon"] >= low) & (data["horizon"] <= high)
            groups.append(dict(scenario=name, horizon_range=[low, high], count=int(mask.sum()),
                **error_metrics(errors[mask])))
    return dict(phase=phase, heads=["Q1", "Q2", "minQ"], rows=rows, per_scenario_horizon=groups,
        pooled=error_metrics(errors), terminal_reward_errors=error_metrics(q[data["terminal"]]-data["reward"][data["terminal"], None]),
        coverage=dict(action_min=data["action"].min(dim=0).values.tolist(),
            action_max=data["action"].max(dim=0).values.tolist(),
            action_99pct_saturation=(data["action"].abs() >= data["action"].new_tensor([.2, .1])*.99).sum(dim=0).tolist(),
            request_min=data["request"].min(dim=0).values.tolist(), request_max=data["request"].max(dim=0).values.tolist(),
            nuf_cap_count=int((data["request"][:, 1] == 6000.).sum()),
            nuf_zero_count=int((data["request"][:, 1] == 0.).sum()),
            nuf_projection_clipped=int((raw[:, 1] != data["request"][:, 1]).sum()),
            distinct_nuf_requests=int(data["request"][:, 1].unique().numel())),
        interpretation=("Before: Q-carry versus G-pi1 is continuation discrepancy, not independent Q-pi1 calibration."
            if phase == "before" else "After: TRAINING fit on the same 375 rows, no heldout calibration."),
        limitations="Samples label Q^pi1 at observed (s,b_pi1), not Q-star or other actions. All 375 NUF requests "
            "are 6000: no NUF action variation, so that action axis is not identified. Frozen actor/Phi means "
            "the physical policy is unchanged; critic-only fitting cannot claim improved traffic TTT.")


def final_metrics(learner):
    learner.validate()
    if learner.phase != "done":
        raise ValueError("Final metrics require exactly 250 updates")
    return dict(before=learner.before, after=diagnostics(learner, "after"), updates=learner.losses,
        initial_model_sha256=learner.initial_hashes,
        final_model_sha256={k: tensor_hash(m.state_dict()) for k, m in learner.models().items()},
        optimizer_provenance=learner.optimizer_provenance,
        final_optimizer_sha256=state_hash(learner.optimizer.state_dict()),
        counters=learner.counts, ancestor_counts=ANCESTOR_COUNTS,
        samples_per_scenario=dict(zip(SCENARIOS, learner.sample_counts)),
        forced_terminals_per_scenario=dict(zip(SCENARIOS, learner.forced_terminal_counts)),
        all_sampled_terminals_per_scenario=dict(zip(SCENARIOS, learner.terminal_counts)),
        target_policy_actor_sha256=learner.initial_hashes["actor"], target_updates_during_fit=0,
        final_target_copy=True, traffic_improvement_claim=False, heldout=False)
