"""Exact worker lifecycle tests using a tiny deterministic fake plant, never a solver."""
from contextlib import ExitStack
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from work import five_cell_worker as worker
from rl_leader.response_dqn_catalog import DiscreteLeaderAction, StructuredActionCatalog
from rl_leader.response_dqn_collect import CollectionDecision
from rl_leader.response_dqn_data import FrozenResponseReplay, make_replay_manifest


CONTRACT = "a" * 64


class TinyEnv:
    constructions = 0
    resets = 0
    contract = CONTRACT

    def __init__(self, scenario_name, T_total, warmup_nc_steps, **kwargs):
        type(self).constructions += 1
        self.scenario_name, self.T_total = scenario_name, T_total
        self.warmup, self.n_steps = warmup_nc_steps, 80
        self.step_idx = 0
        self.experiment_contract_fingerprint = self.contract
        self.action_schema = SimpleNamespace(names=("budget.N_P",))
        self.sim = SimpleNamespace(total_ttt=0.0)

    def reset(self):
        type(self).resets += 1
        self.step_idx = self.warmup
        self.sim.total_ttt = 10.0
        return self._observe()

    def _observe(self):
        return np.asarray([self.step_idx - self.warmup], dtype=np.float32)

    def _inventory(self):
        return 0.0


def catalog():
    return StructuredActionCatalog(("budget.N_P",), [
        DiscreteLeaderAction(i, "anchor" if i == 0 else "other", "anchor" if i == 0 else "freeway",
                             "test", "identity" if i == 0 else "linear", float(i),
                             "anchor" if i == 0 else "linear", (float(i),))
        for i in range(2)
    ])


def fake_replay(rows, *, env, catalog, source):
    n = len(rows["action_id"])
    manifest = make_replay_manifest(
        transition_count=n, action_count=catalog.size, catalog_fingerprint=catalog.fingerprint,
        observation_schema={}, response_contract="test", scenario=env.scenario_name, source=source,
    )
    manifest.update(reward_semantics="interval_negative_ttt", done_semantics="environment_terminal",
                    experiment_contract_sha256=env.experiment_contract_fingerprint,
                    t_total_sec=env.T_total, response_equivalence_mode=env.response_equivalence_mode)
    manifest["continuation_contract_scope"] = {
        "version": worker.MODE, "scenario": env.scenario_name,
        "expected_contract_sha256": env.experiment_contract_fingerprint,
        "experiment_contract_sha256": env.experiment_contract_fingerprint,
        "shared_contract_sha256": worker.FIVE_CELL_SHARED_CONTRACT_SHA256,
    }
    return FrozenResponseReplay(**{name: np.asarray(values) for name, values in rows.items()}, manifest=manifest)


def fake_policy(*, anchor_probability):
    def choose(evaluated, catalog, rng):
        value = 0 if anchor_probability == 1 else int(rng.integers(2))
        return CollectionDecision(value, {})
    return choose


def fake_collect(env, catalog, policy, *, rng, episode, event_group, initial_rows,
                 checkpoint_callback, initial_observation, **kwargs):
    fields = ("observation", "action_id", "reward", "next_observation", "done", "option_steps",
              "action_mask", "next_action_mask", "response_features", "next_response_features",
              "event_group", "episode", "control_step")
    rows = initial_rows if initial_rows is not None else {key: [] for key in fields}
    while env.step_idx < env.n_steps:
        step = env.step_idx - env.warmup
        decision = policy(SimpleNamespace(), catalog, rng)
        obs = env._observe()
        env.step_idx += 1
        reward = -float(1 + decision.action_id)
        env.sim.total_ttt -= reward
        done = env.step_idx == env.n_steps
        values = dict(
            observation=obs, action_id=decision.action_id, reward=reward,
            next_observation=env._observe(), done=float(done), option_steps=1,
            action_mask=[True, True], next_action_mask=[True, False] if done else [True, True],
            response_features=np.full((2, 1), step, dtype=np.float32),
            next_response_features=np.zeros((2, 1), dtype=np.float32) if done else np.full((2, 1), step+1, dtype=np.float32),
            event_group=event_group, episode=episode, control_step=step,
        )
        for key in fields:
            rows[key].append(values[key])
        with Path(kwargs["log_path"]).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({
                "format_version": worker.COLLECTOR_FORMAT, "episode": episode,
                "control_step": step, "state_id": f"state-{step}",
                "selected_action_id": decision.action_id, "interval_reward": reward,
                "catalog_fingerprint": catalog.fingerprint, "validity_gate_pass": True,
            }) + "\n")
        checkpoint_callback(rows)
    return rows


class FiveCellWorkerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        source = self.root / "source.py"
        source.write_text("pinned implementation", encoding="utf-8")
        self.payload = {
            "directory": str(self.root / "episode"), "scenario": worker.SCENARIOS[0],
            "experiment_contract_sha256": CONTRACT, "catalog": catalog().as_manifest(),
            "seed": 21, "episode": 7, "epsilon": 0.5, "checkpoints": [],
            "response_workers": 1, "max_wall_seconds": 120, "stop_file": str(self.root / "STOP"),
            "expected_source_hashes": {str(source): worker.sha256_file(source)},
            "expected_model_hashes": {},
        }
        TinyEnv.constructions = TinyEnv.resets = 0
        TinyEnv.contract = CONTRACT
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        for name, replacement in (
            ("RLLeaderEnv", TinyEnv), ("rows_to_replay", fake_replay),
            ("collect_sequential_episode", fake_collect), ("random_masked_policy", fake_policy),
            ("_configure_torch_threads", Mock()),
            ("validate_five_cell_contract", Mock()),
        ):
            self.stack.enter_context(patch.object(worker, name, replacement))
        self.shutdown = self.stack.enter_context(patch.object(worker, "_shutdown_response_process_pools"))

    def run_episode(self, **kwargs):
        return worker.collect_five_cell_episode(self.payload, **kwargs)

    def test_full_run_starts_normally_and_reconciles_75_transitions(self):
        summary = self.run_episode()
        self.assertEqual((summary["start_control_step"], summary["end_control_step"], summary["transitions"]), (0, 74, 75))
        self.assertEqual(summary["scope"], "ungated_full_run")
        self.assertEqual(summary["total_ttt"], summary["prefix_ttt"] - summary["reward_sum"])
        self.assertTrue(summary["terminal"] and summary["ttt_reconciled"])
        self.assertEqual(TinyEnv.resets, 1)
        self.shutdown.assert_called_once_with(wait=True)

    def test_resume_preserves_rng_and_never_resets_existing_state(self):
        with self.assertRaisesRegex(InterruptedError, "diagnostic transition limit"):
            self.run_episode(max_new_transitions=1)
        progress = json.loads((Path(self.payload["directory"]) / "progress.json").read_text())
        self.assertEqual(progress["transitions"], 1)
        self.assertFalse(progress["terminal"])
        resumed = self.run_episode()
        replay = worker.load_frozen_response_replay(resumed["replay"])
        self.assertEqual(TinyEnv.resets, 1)
        self.payload["directory"] = str(self.root / "uninterrupted")
        uninterrupted = self.run_episode()
        other = worker.load_frozen_response_replay(uninterrupted["replay"])
        np.testing.assert_array_equal(replay.action_id, other.action_id)
        self.assertEqual(resumed["total_ttt"], uninterrupted["total_ttt"])

    def test_complete_cache_is_verified_without_a_new_environment(self):
        first = self.run_episode()
        self.assertEqual(self.run_episode(), first)
        self.assertEqual(TinyEnv.constructions, 1)

    def test_stop_prevents_environment_construction(self):
        Path(self.payload["stop_file"]).touch()
        with self.assertRaises(InterruptedError):
            self.run_episode()
        self.assertEqual(TinyEnv.constructions, 0)
        self.shutdown.assert_called_once_with(wait=True)

    def test_stop_at_first_step_saves_exact_nonterminal_state(self):
        def stopping(*args, **kwargs):
            callback = kwargs["checkpoint_callback"]
            def save(rows):
                Path(self.payload["stop_file"]).touch()
                callback(rows)
            kwargs["checkpoint_callback"] = save
            return fake_collect(*args, **kwargs)
        with patch.object(worker, "collect_sequential_episode", stopping):
            with self.assertRaisesRegex(InterruptedError, "exact nonterminal"):
                self.run_episode()
        progress = json.loads((Path(self.payload["directory"]) / "progress.json").read_text())
        self.assertEqual(progress["transitions"], 1)
        self.assertFalse(progress["terminal"])

    def test_wall_budget_is_per_invocation_and_resumable(self):
        clock = [0.0]
        self.payload["max_wall_seconds"] = 1.0
        def expiring(*args, **kwargs):
            callback = kwargs["checkpoint_callback"]
            def save(rows):
                clock[0] = 2.0
                callback(rows)
            kwargs["checkpoint_callback"] = save
            return fake_collect(*args, **kwargs)
        with patch.object(worker.time, "perf_counter", lambda: clock[0]), patch.object(worker, "collect_sequential_episode", expiring):
            with self.assertRaisesRegex(InterruptedError, "wall budget"):
                self.run_episode()
        self.payload["max_wall_seconds"] = 120
        self.assertEqual(self.run_episode()["transitions"], 75)
        self.assertEqual(TinyEnv.resets, 1)

    def test_source_drift_fails_before_construction(self):
        Path(next(iter(self.payload["expected_source_hashes"]))).write_text("changed")
        with self.assertRaisesRegex(ValueError, "source hash mismatch"):
            self.run_episode()
        self.assertEqual(TinyEnv.constructions, 0)

    def test_contract_mismatch_fails_before_reset(self):
        TinyEnv.contract = "b" * 64
        with self.assertRaisesRegex(ValueError, "experiment contract mismatch"):
            self.run_episode()
        self.assertEqual(TinyEnv.resets, 0)

    def test_changed_configuration_rejects_checkpoint(self):
        with self.assertRaises(InterruptedError):
            self.run_episode(max_new_transitions=1)
        self.payload["epsilon"] = 0.25
        with self.assertRaisesRegex(ValueError, "configuration changed"):
            self.run_episode()

    def test_tampered_replay_and_checkpoint_fail_closed(self):
        for filename in ("replay.npz", "checkpoint.pkl"):
            with self.subTest(filename=filename):
                self.payload["directory"] = str(self.root / filename)
                self.run_episode()
                target = Path(self.payload["directory"]) / filename
                target.write_bytes(target.read_bytes() + b"changed")
                with self.assertRaisesRegex(ValueError, "hash mismatch"):
                    self.run_episode()

    def test_tampered_summary_is_not_success(self):
        self.run_episode()
        path = Path(self.payload["directory"]) / "summary.json"
        summary = json.loads(path.read_text())
        summary["total_ttt"] = 0.0
        path.write_text(json.dumps(summary))
        with self.assertRaisesRegex(ValueError, "summary integrity"):
            self.run_episode()

    def test_baseline_is_anchor_only(self):
        self.payload.update(purpose="baseline", epsilon=0)
        summary = self.run_episode()
        self.assertEqual(summary["action_support_counts"], [75, 0])

    def test_invalid_eval_and_unpinned_models_reject(self):
        self.payload.update(purpose="evaluation", epsilon=0)
        with self.assertRaisesRegex(ValueError, "evaluation requires checkpoints"):
            self.run_episode()
        self.payload["checkpoints"] = [str(self.root / "model.pt")]
        with self.assertRaisesRegex(ValueError, "model pins"):
            self.run_episode()

    def test_evaluation_uses_the_pinned_model_with_zero_exploration(self):
        path = self.root / "model.pt"
        path.write_bytes(b"model fixture")
        self.payload.update(purpose="evaluation", epsilon=0, checkpoints=[str(path)],
                            expected_model_hashes={str(path): worker.sha256_file(path)})
        model = SimpleNamespace(response_equivalence_mode=worker.MODE)
        policy = lambda evaluated, catalog, rng: CollectionDecision(1, {})
        with patch.object(worker, "load_trained_response_dqn", return_value=model) as load, patch.object(
            worker, "ensemble_greedy_policy", return_value=policy
        ) as factory:
            result = self.run_episode()
        load.assert_called_once_with(str(path.resolve()), expected_catalog_fingerprint=catalog().fingerprint)
        factory.assert_called_once_with([model], exploration_epsilon=0)
        self.assertEqual(result["action_support_counts"], [0, 75])

    def test_disconnected_next_state_is_rejected_before_completion(self):
        def corrupt(rows, **kwargs):
            replay = fake_replay(rows, **kwargs)
            if len(rows["action_id"]) > 1:
                replay.next_observation[0, 0] = -1
            return replay
        with patch.object(worker, "rows_to_replay", corrupt):
            with self.assertRaisesRegex(ValueError, "disconnected"):
                self.run_episode()
        self.assertFalse((Path(self.payload["directory"]) / "summary.json").exists())

    def test_premature_terminal_is_not_a_complete_full_run(self):
        def corrupt(rows, **kwargs):
            replay = fake_replay(rows, **kwargs)
            replay.done[-1] = 1.0
            return replay
        with patch.object(worker, "rows_to_replay", corrupt):
            with self.assertRaisesRegex(ValueError, "only full-run transition 74"):
                self.run_episode()

    def test_invalid_trace_stops_before_success_and_retains_prior_checkpoint(self):
        def invalid(*args, **kwargs):
            callback = kwargs["checkpoint_callback"]
            def save(rows):
                path = Path(kwargs["log_path"])
                item = json.loads(path.read_text().splitlines()[-1])
                item["validity_gate_pass"] = False
                path.write_text(json.dumps(item) + "\n")
                callback(rows)
            kwargs["checkpoint_callback"] = save
            return fake_collect(*args, **kwargs)
        with patch.object(worker, "collect_sequential_episode", invalid):
            with self.assertRaisesRegex(ValueError, "validity gate failed"):
                self.run_episode()
        progress = json.loads((Path(self.payload["directory"]) / "progress.json").read_text())
        self.assertEqual(progress["transitions"], 0)

    def test_resumed_duplicate_trace_must_agree_on_state_action_and_reward(self):
        with self.assertRaises(InterruptedError):
            self.run_episode(max_new_transitions=1)
        root = Path(self.payload["directory"])
        first = json.loads((root / "trace.jsonl").read_text().splitlines()[0])
        first["state_id"] = "different-state"
        (root / "trace_resume_001.jsonl").write_text(json.dumps(first) + "\n")
        with self.assertRaisesRegex(ValueError, "duplicates disagree"):
            self.run_episode()

    def test_forced_actions_and_unaudited_scenarios_reject(self):
        self.payload["first_action"] = 0
        with self.assertRaisesRegex(ValueError, "does not accept first_action"):
            self.run_episode()
        del self.payload["first_action"]
        self.payload["scenario"] = "anything"
        with self.assertRaisesRegex(ValueError, "named five-cell"):
            self.run_episode()

    def test_collector_failure_cleans_pools_and_leaves_initial_checkpoint(self):
        with patch.object(worker, "collect_sequential_episode", side_effect=ValueError("bad preview")):
            with self.assertRaisesRegex(ValueError, "bad preview"):
                self.run_episode()
        self.shutdown.assert_called_once_with(wait=True)
        progress = json.loads((Path(self.payload["directory"]) / "progress.json").read_text())
        self.assertEqual(progress["transitions"], 0)
        self.assertFalse((Path(self.payload["directory"]) / "summary.json").exists())


if __name__ == "__main__":
    unittest.main()
