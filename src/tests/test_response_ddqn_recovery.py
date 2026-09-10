from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from rl_leader.response_ddqn_recovery import (
    BranchOutcome,
    EnvSnapshot,
    RecoveryActionLabel,
    RecoveryBranchConfig,
    RecoveryStateEvaluation,
    _evaluate_first_action,
    capture_policy_snapshots_cached,
    capture_policy_snapshots,
    read_policy_trace,
    recovery_state_to_replay,
    select_problem_steps,
)
from rl_leader.evaluate_response_dqn_policy import (
    _step_gated_policy,
    summarize_policy_run,
)
from rl_leader.response_dqn_collect import CollectionDecision


class _FakeObservationSchema:
    def metadata(self):
        return {"version": "fake", "dimension": 2}


class _FakeEnv:
    scenario_name = "fake_scenario"
    T_total = 720.0
    warmup = 0
    n_steps = 4
    observation_schema = _FakeObservationSchema()
    experiment_contract_fingerprint = "fake-contract"

    def __init__(self, state="root", step_idx=0):
        self.state = state
        self.step_idx = step_idx
        self.sim = SimpleNamespace(
            state=SimpleNamespace(time_sec=float(step_idx * 180.0)),
        )

    def __deepcopy__(self, memo):
        return _FakeEnv(self.state, self.step_idx)

    def _observe(self):
        code = {"root": 0.0, "bad_after": 1.0, "terminal": 2.0}[self.state]
        return np.asarray([float(self.step_idx), code], dtype=np.float32)

    def reset(self):
        self.state = "root"
        self.step_idx = 0
        self.sim.state.time_sec = 0.0
        return self._observe()

    def _inventory(self):
        return 0.0

    def _anchor_context_state_fingerprint(self):
        return f"{self.state}:{self.step_idx}"


def _fake_evaluated(env):
    return SimpleNamespace(
        observation=env._observe(),
        response_mask=SimpleNamespace(
            valid_action_mask=np.asarray([True, True, True]),
            invalid_reasons=("", "", ""),
        ),
        response_features=np.zeros((3, 4), dtype=np.float32),
    )


def _fake_commit(env, evaluated, catalog, action_id):
    transitions = {
        ("root", 0): ("terminal", -10.0, True),
        ("root", 1): ("bad_after", -1.0, False),
        ("root", 2): ("terminal", -5.0, True),
        ("bad_after", 0): ("terminal", -100.0, True),
        ("bad_after", 1): ("terminal", -50.0, True),
        ("bad_after", 2): ("terminal", -1.0, True),
    }
    env.state, reward, done = transitions[(env.state, int(action_id))]
    env.step_idx += 1
    env.sim.state.time_sec = float(env.step_idx * 180.0)
    return env._observe(), reward, done, {"inventory_after": 0.0}


class ResponseDDQNRecoveryTest(unittest.TestCase):
    def test_policy_trace_remaps_legacy_action_id(self):
        catalog = SimpleNamespace(
            size=2,
            fingerprint="target-catalog",
            actions=(
                SimpleNamespace(action_id=0, key="anchor"),
                SimpleNamespace(action_id=1, key="artifact:stable-action"),
            ),
        )
        with tempfile.TemporaryDirectory() as temp:
            trace_path = Path(temp) / "trace.jsonl"
            trace_path.write_text(
                '{"control_step": 18, "selected_action_id": 103}\n',
                encoding="utf-8",
            )
            prefix = read_policy_trace(
                trace_path,
                catalog=catalog,
                action_id_remap={103: 1},
            )

        self.assertEqual(prefix, {18: 1})

    def test_policy_trace_prefers_stable_action_key(self):
        catalog = SimpleNamespace(
            size=2,
            fingerprint="target-catalog",
            actions=(
                SimpleNamespace(action_id=0, key="anchor"),
                SimpleNamespace(action_id=1, key="artifact:stable-action"),
            ),
        )
        with tempfile.TemporaryDirectory() as temp:
            trace_path = Path(temp) / "trace.jsonl"
            trace_path.write_text(
                """{"control_step": 18, "selected_action_id": 103, "selected_action_key": "artifact:stable-action", "catalog_fingerprint": "source-catalog"}\n""",
                encoding="utf-8",
            )
            prefix = read_policy_trace(trace_path, catalog=catalog)

        self.assertEqual(prefix, {18: 1})

    def test_policy_snapshot_capture_reuses_one_prefix_pass(self):
        env = _FakeEnv()
        catalog = SimpleNamespace(size=3)
        calls = []

        def fake_commit(env, catalog, action_id):
            calls.append((env.step_idx, int(action_id)))
            env.step_idx += 1
            env.state = "bad_after" if env.step_idx == 1 else "terminal"
            env.sim.state.time_sec = float(env.step_idx * 180.0)
            return env._observe(), -1.0, False, {}

        with patch(
            "rl_leader.response_ddqn_recovery.commit_catalog_action_from_anchor",
            side_effect=fake_commit,
        ):
            snapshots = capture_policy_snapshots(
                env,
                catalog,
                (1, 2),
                {0: 1, 1: 2},
            )

        self.assertEqual(sorted(snapshots), [1, 2])
        self.assertEqual(calls, [(0, 1), (1, 2)])
        self.assertEqual(snapshots[1].control_step, 1)
        self.assertEqual(snapshots[2].control_step, 2)

    def test_policy_snapshot_cache_skips_prefix_replay_on_hit(self):
        catalog = SimpleNamespace(size=3, fingerprint="fake-catalog")
        prefix = {0: 1, 1: 2}

        def fake_commit(env, catalog, action_id):
            env.step_idx += 1
            env.state = "bad_after" if env.step_idx == 1 else "terminal"
            env.sim.state.time_sec = float(env.step_idx * 180.0)
            return env._observe(), -1.0, False, {}

        with tempfile.TemporaryDirectory() as temp:
            cache_dir = Path(temp)
            with patch(
                "rl_leader.response_ddqn_recovery.commit_catalog_action_from_anchor",
                side_effect=fake_commit,
            ):
                first = capture_policy_snapshots_cached(
                    _FakeEnv(),
                    catalog,
                    (2,),
                    prefix,
                    cache_dir=cache_dir,
                )

            def fail_commit(*args, **kwargs):
                raise AssertionError("cache hit should not replay the prefix")

            with patch(
                "rl_leader.response_ddqn_recovery.commit_catalog_action_from_anchor",
                side_effect=fail_commit,
            ):
                second = capture_policy_snapshots_cached(
                    _FakeEnv(),
                    catalog,
                    (2,),
                    prefix,
                    cache_dir=cache_dir,
                )

        self.assertEqual(first[2].state_fingerprint, second[2].state_fingerprint)
        self.assertEqual(second[2].control_step, 2)

    def test_anchor_only_snapshot_cache_ignores_catalog_fingerprint(self):
        prefix = {0: 0, 1: 0}
        first_catalog = SimpleNamespace(size=3, fingerprint="first")
        second_catalog = SimpleNamespace(size=3, fingerprint="second")

        def fake_commit(env, catalog, action_id):
            env.step_idx += 1
            env.state = "bad_after" if env.step_idx == 1 else "terminal"
            env.sim.state.time_sec = float(env.step_idx * 180.0)
            return env._observe(), -1.0, False, {}

        with tempfile.TemporaryDirectory() as temp:
            cache_dir = Path(temp)
            with patch(
                "rl_leader.response_ddqn_recovery.commit_catalog_action_from_anchor",
                side_effect=fake_commit,
            ):
                first = capture_policy_snapshots_cached(
                    _FakeEnv(),
                    first_catalog,
                    (2,),
                    prefix,
                    cache_dir=cache_dir,
                )

            def fail_commit(*args, **kwargs):
                raise AssertionError("anchor-only cache hit should ignore catalog")

            with patch(
                "rl_leader.response_ddqn_recovery.commit_catalog_action_from_anchor",
                side_effect=fail_commit,
            ):
                second = capture_policy_snapshots_cached(
                    _FakeEnv(),
                    second_catalog,
                    (2,),
                    prefix,
                    cache_dir=cache_dir,
                )

        self.assertEqual(first[2].state_fingerprint, second[2].state_fingerprint)

    def test_first_action_can_be_rescued_by_recovery_action(self):
        env = _FakeEnv()
        catalog = SimpleNamespace(size=3)
        config = RecoveryBranchConfig(
            max_rollout_steps=2,
            recovery_depth=1,
            top_k=0,
        )

        def fake_tail(env, *, max_steps):
            if env.state == "bad_after":
                return BranchOutcome(-100.0, 100.0, 1, 0.0, (0,), True)
            return BranchOutcome(0.0, 0.0, 0, 0.0, (), True)

        with patch(
            "rl_leader.response_ddqn_recovery.evaluate_executable_responses",
            side_effect=lambda env, observation, catalog, workers=1, backend="serial": _fake_evaluated(env),
        ), patch(
            "rl_leader.response_ddqn_recovery.commit_action",
            side_effect=_fake_commit,
        ), patch(
            "rl_leader.response_ddqn_recovery.rollout_pstack_tail",
            side_effect=fake_tail,
        ):
            label = _evaluate_first_action(
                copy.deepcopy(env),
                _fake_evaluated(env),
                catalog,
                1,
                config,
                ensemble=(),
            )

        self.assertAlmostEqual(label.anchor_tail_return, -101.0)
        self.assertAlmostEqual(label.best_recovery_return, -2.0)
        self.assertEqual(label.best_recovery_sequence, (1, 2))
        self.assertGreater(label.recovery_gain_vs_anchor_tail, 90.0)

    def test_recovery_replay_uses_terminal_mc_targets_for_ddqn(self):
        env = _FakeEnv()
        snapshot = EnvSnapshot(
            env=env,
            observation=env._observe(),
            label="unit",
            step_idx=0,
            control_step=0,
            simulation_time_sec=0.0,
            state_fingerprint=env._anchor_context_state_fingerprint(),
            experiment_contract_sha256="fake-contract",
        )
        evaluated = _fake_evaluated(env)
        labels = (
            RecoveryActionLabel(
                action_id=0,
                valid=True,
                invalid_reason="",
                first_step_reward=-10.0,
                first_step_ttt=10.0,
                anchor_tail_return=-10.0,
                anchor_tail_ttt=10.0,
                anchor_tail_steps=1,
                anchor_tail_inventory=0.0,
                best_recovery_return=-10.0,
                best_recovery_ttt=10.0,
                best_recovery_steps=1,
                best_recovery_inventory=0.0,
                best_recovery_sequence=(0,),
            ),
            RecoveryActionLabel(
                action_id=1,
                valid=True,
                invalid_reason="",
                first_step_reward=-1.0,
                first_step_ttt=1.0,
                anchor_tail_return=-101.0,
                anchor_tail_ttt=101.0,
                anchor_tail_steps=2,
                anchor_tail_inventory=0.0,
                best_recovery_return=-2.0,
                best_recovery_ttt=2.0,
                best_recovery_steps=2,
                best_recovery_inventory=0.0,
                best_recovery_sequence=(1, 2),
            ),
        )
        state = RecoveryStateEvaluation(
            snapshot=snapshot,
            evaluated_state=evaluated,
            config=RecoveryBranchConfig(max_rollout_steps=2, recovery_depth=1),
            labels=labels,
            evaluation_seconds=0.0,
        )
        catalog = SimpleNamespace(
            size=3,
            fingerprint="catalog",
            as_manifest=lambda: {"format_version": "fake", "fingerprint": "catalog"},
        )

        replay = recovery_state_to_replay(state, catalog, source="unit-test")

        self.assertEqual(replay.size, 2)
        np.testing.assert_array_equal(replay.action_id, [0, 1])
        np.testing.assert_allclose(replay.reward, [-10.0, -2.0])
        np.testing.assert_array_equal(replay.done, [1.0, 1.0])
        self.assertEqual(
            replay.manifest["reward_semantics"],
            "terminal_best_recovery_return_negative_ttt",
        )

    def test_recovery_replay_can_store_anchor_relative_advantage_targets(self):
        env = _FakeEnv()
        snapshot = EnvSnapshot(
            env=env,
            observation=env._observe(),
            label="unit",
            step_idx=0,
            control_step=0,
            simulation_time_sec=0.0,
            state_fingerprint=env._anchor_context_state_fingerprint(),
            experiment_contract_sha256="fake-contract",
        )
        labels = (
            RecoveryActionLabel(
                action_id=0,
                valid=True,
                invalid_reason="",
                first_step_reward=-10.0,
                first_step_ttt=10.0,
                anchor_tail_return=-10.0,
                anchor_tail_ttt=10.0,
                anchor_tail_steps=1,
                anchor_tail_inventory=0.0,
                best_recovery_return=-10.0,
                best_recovery_ttt=10.0,
                best_recovery_steps=1,
                best_recovery_inventory=0.0,
                best_recovery_sequence=(0,),
            ),
            RecoveryActionLabel(
                action_id=1,
                valid=True,
                invalid_reason="",
                first_step_reward=-1.0,
                first_step_ttt=1.0,
                anchor_tail_return=-101.0,
                anchor_tail_ttt=101.0,
                anchor_tail_steps=2,
                anchor_tail_inventory=0.0,
                best_recovery_return=-2.0,
                best_recovery_ttt=2.0,
                best_recovery_steps=2,
                best_recovery_inventory=0.0,
                best_recovery_sequence=(1, 2),
            ),
        )
        state = RecoveryStateEvaluation(
            snapshot=snapshot,
            evaluated_state=_fake_evaluated(env),
            config=RecoveryBranchConfig(max_rollout_steps=2, recovery_depth=1),
            labels=labels,
            evaluation_seconds=0.0,
        )
        catalog = SimpleNamespace(
            size=3,
            fingerprint="catalog",
            as_manifest=lambda: {"format_version": "fake", "fingerprint": "catalog"},
        )

        replay = recovery_state_to_replay(
            state,
            catalog,
            source="unit-test",
            reward_mode="advantage",
        )

        np.testing.assert_allclose(replay.reward, [0.0, 8.0])
        self.assertEqual(replay.manifest["reward_mode"], "advantage")
        self.assertEqual(
            replay.manifest["reward_semantics"],
            "terminal_recovery_advantage_vs_action0_return",
        )

    def test_problem_step_selection_prefers_non_anchor_actions(self):
        rows = [
            {"control_step": 0, "selected_action_id": 0},
            {"control_step": 3, "selected_action_id": 12},
            {"control_step": 5, "selected_action_id": 0},
            {"control_step": 7, "selected_action_id": 10},
        ]

        self.assertEqual(select_problem_steps(rows, max_steps=1), [3])
        self.assertEqual(select_problem_steps(rows), [3, 7])
        self.assertEqual(
            select_problem_steps(rows, explicit_steps=(9, 3)),
            [3, 9],
        )

    def test_branch_workers_reject_nested_process_response_workers(self):
        config = RecoveryBranchConfig(
            branch_workers=2,
            response_workers=2,
            response_backend="process",
        )

        with self.assertRaisesRegex(ValueError, "nested"):
            config.validate()

    def test_branch_workers_allow_sequential_process_pools_without_recovery_tree(self):
        config = RecoveryBranchConfig(
            recovery_depth=0,
            branch_workers=2,
            response_workers=2,
            response_backend="process",
        )

        config.validate()

    def test_policy_eval_summary_compares_full_ttt_with_matching_warmup(self):
        env = _FakeEnv()
        env.scenario_name = "fake_scenario"
        env.T_total = 720.0
        env.warmup = 1
        env.sim.total_ttt = 115.0
        env.sim.urban_ttt = 40.0
        env.sim.freeway_ttt = 75.0
        rows = {
            "reward": [-10.0, -20.0],
            "action_id": [0, 2],
        }

        summary = summarize_policy_run(
            env=env,
            rows=rows,
            pstack_summary={"total_ttt": 120.0},
        )

        self.assertAlmostEqual(summary["control_ttt"], 30.0)
        self.assertAlmostEqual(summary["warmup_ttt"], 85.0)
        self.assertAlmostEqual(summary["pstack_control_ttt"], 35.0)
        self.assertTrue(summary["beats_pstack"])
        self.assertEqual(summary["action_counts"], {"0": 1, "2": 1})

    def test_step_gate_forces_anchor_outside_allowed_window(self):
        calls = []

        def base_policy(evaluated, catalog, rng):
            calls.append("base")
            return CollectionDecision(action_id=2, diagnostics={"policy": "base"})

        gated = _step_gated_policy(base_policy, {1})
        rng = np.random.default_rng(0)
        first = gated(None, None, rng)
        second = gated(None, None, rng)
        third = gated(None, None, rng)

        self.assertEqual(first.action_id, 0)
        self.assertEqual(second.action_id, 2)
        self.assertEqual(third.action_id, 0)
        self.assertEqual(calls, ["base"])
        self.assertEqual(
            first.diagnostics["fallback_reason"],
            "outside_allowed_control_steps",
        )
        self.assertTrue(second.diagnostics["step_gate_allowed"])


if __name__ == "__main__":
    unittest.main()
