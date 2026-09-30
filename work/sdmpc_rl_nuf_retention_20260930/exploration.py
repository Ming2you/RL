"""Pure carry/local exploration; the physical controller remains the executor."""

import copy
import json
from collections.abc import Mapping
from numbers import Integral, Real

import numpy as np


_FORMAT = "nuf-retention-policy-v1"
_FORMULA = {
    "mean_reversion": 0.8,
    "noise_scales": [10.0, 100.0],
    "offset_limits": [50.0, 500.0],
    "action_scales": [50.0, 1000.0],
    "action_limits": [-1.0, 1.0],
    "action_dtype": "float32",
    "budget_dtype": "float64",
    "projection": "NUF:[0,capacity];NP:signed",
    "rng": "numpy.PCG64.normal(size=2)",
    "np_rebase": ["reference_fallback", "pfo_recovery"],
    "nuf_base": "initial_actual_action_anchor",
    "nuf_realized_offset": "actual_executed_minus_retained_base_after_every_commit",
}
TREATMENT = {"id": "nuf-center-retention-v1", "policy_format": _FORMAT, "formula": _FORMULA}


def _pair(value, name):
    try:
        value = np.asarray(value)
        if value.shape != (2,) or value.dtype.kind not in "iuf":
            raise ValueError
        value = value.astype(np.float64, copy=True)
        if not np.isfinite(value).all():
            raise ValueError
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite numeric length-2 array") from exc
    return value


def _json(value):
    try:
        return json.dumps(value, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("State must contain only finite JSON values") from exc


class LocalBudgetPolicy:
    """One outstanding choice at a time; count records completed commits."""

    def __init__(self, behavior, seed, capacity):
        if not isinstance(behavior, str) or behavior not in ("carry", "local"):
            raise ValueError("behavior must be carry or local")
        if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, Integral) or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if (isinstance(capacity, (bool, np.bool_)) or not isinstance(capacity, Real)
                or not np.isfinite(capacity) or capacity <= 0):
            raise ValueError("capacity must be positive and finite")
        self.behavior, self.seed, self.capacity = behavior, int(seed), float(capacity)
        self._rng = np.random.default_rng(self.seed)
        self._base = None
        self._realized_offset = np.zeros(2, dtype=np.float64)
        self._last_executed = None
        self._pending = None
        self._count = 0

    def _project(self, budget):
        projected = budget.copy()
        projected[1] = np.clip(projected[1], 0.0, self.capacity)
        return projected

    def _choice(self, anchor, base, noise):
        scales = np.array([50.0, 1000.0])
        with np.errstate(over="raise", invalid="raise"):
            if self.behavior == "local":
                offset_raw = 0.8 * self._realized_offset + [10.0, 100.0] * noise
                offset = np.clip(offset_raw, [-50.0, -500.0], [50.0, 500.0])
                desired_raw = base + offset
                desired = self._project(desired_raw)
                raw_action = (desired - anchor) / scales
                action = np.clip(raw_action, -1.0, 1.0).astype(np.float32)
            else:
                desired_raw = anchor.copy()
                desired = self._project(anchor)
                offset = desired - base
                offset_raw = offset.copy()
                raw_action = np.zeros(2)
                action = np.zeros(2, dtype=np.float32)
            # Match residual_budget: promote the final float32 action before scaling.
            raw_request = anchor + scales * action.astype(np.float64)
            request = self._project(raw_request)
            requested_offset = request - base
        return {
            "behavior": self.behavior,
            "step": self._count,
            "committed": False,
            "action_anchor": anchor.tolist(),
            "base_before": base.tolist(),
            "base_after": None,
            "realized_offset_before": self._realized_offset.tolist(),
            "realized_offset_after": None,
            "noise": noise.tolist(),
            "desired_offset": offset.tolist(),
            "desired_budget_raw": desired_raw.tolist(),
            "desired_budget": desired.tolist(),
            "raw_action": raw_action.tolist(),
            "action": action.tolist(),
            "raw_request": raw_request.tolist(),
            "projected_request": request.tolist(),
            "requested_offset": requested_offset.tolist(),
            "offset_limited": (offset != offset_raw).tolist(),
            "rate_limited": (np.abs(raw_action) > 1.0).tolist(),
            "desired_nuf_projected": bool(desired_raw[1] != desired[1]),
            "request_nuf_projected": bool(raw_request[1] != request[1]),
            "executed_budget": None,
            "executed_offset": None,
            "rebased": False,
            "rebase_reasons": [],
        }

    def choose(self, anchor):
        if self._pending is not None:
            raise RuntimeError("commit the pending choice before choosing again")
        anchor = _pair(anchor, "anchor")
        base = anchor.copy() if self._base is None else self._base.copy()
        rng = copy.deepcopy(self._rng)
        noise = rng.normal(size=2) if self.behavior == "local" else np.zeros(2)
        try:
            audit = self._choice(anchor, base, noise)
            _json(audit)
        except FloatingPointError as exc:
            raise ValueError("Choice arithmetic must remain finite") from exc
        self._base, self._rng, self._pending = base, rng, audit
        return np.asarray(audit["action"], dtype=np.float32), copy.deepcopy(audit)

    def commit(self, row):
        if self._pending is None:
            raise RuntimeError("choose an action before committing")
        if not isinstance(row, Mapping):
            raise ValueError("row must be a mapping")
        audit = copy.deepcopy(self._pending)
        try:
            for key, expected in (("action_anchor", audit["action_anchor"]),
                                  ("action_requested", audit["action"])):
                if not np.array_equal(_pair(row[key], key), expected):
                    raise ValueError(f"{key} does not match the pending choice")
            request = _pair(row["B_requested"][0], "B_requested[0]")
            if not np.array_equal(request, audit["projected_request"]):
                raise ValueError("B_requested[0] does not match the projected request")
            executed = _pair(row["B_executed"], "B_executed")
            selection, reference = row["selection_source"], row["reference_source"]
            recovery = row["reference_recovery"]
            if (not isinstance(selection, str)
                    or selection not in ("lower_solution", "reference_fallback")):
                raise ValueError("Invalid selection_source")
            if (not isinstance(reference, str)
                    or reference not in ("previous", "pfo_initial", "pfo_recovery")):
                raise ValueError("Invalid reference_source")
            if (not isinstance(recovery, (bool, np.bool_))
                    or bool(recovery) != (reference == "pfo_recovery")):
                raise ValueError("reference_recovery must agree with reference_source")
            if not isinstance(row["guard_mode"], str) or row["guard_mode"] != "physical":
                raise ValueError("guard_mode must be physical")
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError("Incomplete or malformed execution row") from exc
        reasons = []
        if selection == "reference_fallback":
            reasons.append("reference_fallback")
        if reference == "pfo_recovery":
            reasons.append("pfo_recovery")
        try:
            with np.errstate(over="raise", invalid="raise"):
                executed_offset = executed - self._base
            base = self._base.copy()
            if reasons:
                base[0] = executed[0]
            realized = executed - base
        except FloatingPointError as exc:
            raise ValueError("Executed offset must remain finite") from exc
        audit.update(
            committed=True, base_after=base.tolist(), realized_offset_after=realized.tolist(),
            executed_budget=executed.tolist(), executed_offset=executed_offset.tolist(),
            rebased=bool(reasons), rebase_reasons=reasons, selection_source=selection,
            reference_source=reference, reference_recovery=bool(recovery), guard_mode="physical",
        )
        self._base, self._realized_offset, self._last_executed = base, realized, executed
        self._pending = None
        self._count += 1
        return audit

    def state_dict(self):
        return copy.deepcopy({
            "format": _FORMAT,
            "spec": {"behavior": self.behavior, "seed": self.seed,
                     "capacity": self.capacity, "formula": _FORMULA},
            "base": None if self._base is None else self._base.tolist(),
            "realized_offset": self._realized_offset.tolist(),
            "last_executed": None if self._last_executed is None else self._last_executed.tolist(),
            "pending": self._pending,
            "rng": self._rng.bit_generator.state,
            "count": self._count,
        })

    def load_state_dict(self, state):
        """Validate in isolation, including seeded RNG replay, before replacing state."""
        state = json.loads(_json(state))
        current = self.state_dict()
        if not isinstance(state, dict) or state.keys() != current.keys():
            raise ValueError("Invalid checkpoint fields")
        if state["format"] != _FORMAT or _json(state["spec"]) != _json(current["spec"]):
            raise ValueError("Checkpoint format/spec does not match this policy")
        count, pending = state["count"], state["pending"]
        if type(count) is not int or count < 0:
            raise ValueError("count must be a nonnegative integer")
        candidate = LocalBudgetPolicy(self.behavior, self.seed, self.capacity)
        candidate._count = count
        candidate._realized_offset = _pair(state["realized_offset"], "realized_offset")
        if state["base"] is None:
            if count != 0 or pending is not None or state["last_executed"] is not None:
                raise ValueError("Uninitialized state cannot contain execution or a choice")
        else:
            candidate._base = _pair(state["base"], "base")
            if count == 0 and pending is None:
                raise ValueError("Initial base requires a pending first choice")
        if count == 0:
            if state["last_executed"] is not None or np.any(candidate._realized_offset != 0):
                raise ValueError("Initial realized offset must be zero with no execution")
        else:
            executed = _pair(state["last_executed"], "last_executed")
            try:
                with np.errstate(over="raise", invalid="raise"):
                    realized = executed - candidate._base
            except FloatingPointError as exc:
                raise ValueError("Checkpoint offset must remain finite") from exc
            if not np.array_equal(realized, candidate._realized_offset):
                raise ValueError("Realized offset does not match last execution and base")
            candidate._last_executed = executed
        # Collection checkpoints are short (75 commits); replay also binds RNG to count/seed.
        if self.behavior == "local":
            for _ in range(count):
                candidate._rng.normal(size=2)
        if pending is not None:
            if not isinstance(pending, dict) or "action_anchor" not in pending:
                raise ValueError("Invalid pending choice")
            anchor = _pair(pending["action_anchor"], "pending action_anchor")
            if count == 0 and not np.array_equal(anchor, candidate._base):
                raise ValueError("Initial base must equal the actual first anchor")
            _, expected = candidate.choose(anchor)
            if _json(pending) != _json(expected):
                raise ValueError("Pending choice does not match checkpoint state/formula")
        if _json(state["rng"]) != _json(candidate._rng.bit_generator.state):
            raise ValueError("RNG state does not match seed/count/pending choice")
        self._base = candidate._base
        self._realized_offset = candidate._realized_offset
        self._last_executed = candidate._last_executed
        self._pending = candidate._pending
        self._rng = candidate._rng
        self._count = candidate._count
