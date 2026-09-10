"""No-simulator prefix recovery tests, plus one frozen real identity fixture."""
import copy
import json
import pickle
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from rl_leader.response_ddqn_recovery import commit_catalog_action_from_anchor
from rl_leader.response_dqn_collect import _encoded_continuation as real_encoded_continuation
from rl_leader.response_dqn_data import FrozenResponseReplay, make_replay_manifest
from rl_leader.response_continuation_state import AUDITED_CONTRACT_SHA256
from rl_leader.run_sequential_response_ddqn import load_verified_snapshot
from work import response_policy_prefix as prefix


class FakeCatalog:
    size = 3
    fingerprint = "fake-catalog"

    def residual(self, action):
        return np.array([action], dtype=np.float32)

    def as_manifest(self):
        return {"fingerprint": self.fingerprint, "size": self.size}


class FakeEnv:
    events = []
    after_commit = None

    def __init__(self, scenario_name, T_total, **kwargs):
        self.scenario_name, self.T_total = scenario_name, T_total
        self.warmup = self.step_idx = 2
        self.experiment_contract_fingerprint = AUDITED_CONTRACT_SHA256
        self.sim = SimpleNamespace(total_ttt=3.5, state=SimpleNamespace(time_sec=360.0))
        self.memory = self.action_sum = self.prepares = 0

    def reset(self):
        self.events.append("reset")
        return self._observe()

    def _observe(self):
        return np.array([self.step_idx - self.warmup, self.action_sum], dtype=np.float32)

    def _anchor_context_state_fingerprint(self):
        return f"state:{self._observe().tolist()}"

    def prepare_pstack_anchor_context(self):
        self.events.append("prepare")
        self.prepares += 1
        return self._anchor_context_state_fingerprint()

    def step_prepared_optimizer_anchor(self, context, sync_follower_state=True):
        assert sync_follower_state
        return (*self._advance(0), {})

    def step_anchored_candidate(self, residual, context):
        return self._advance(int(residual[0]))

    def _advance(self, action):
        self.events.append(action)
        reward = -10.123456789 - (self.step_idx - self.warmup)
        self.memory = self.memory * 10 + action + 1
        self.action_sum += action
        self.step_idx += 1
        self.sim.state.time_sec += 180.0
        self.sim.total_ttt -= reward
        if type(self).after_commit:
            type(self).after_commit(self)
        return self._observe(), reward, self.step_idx - self.warmup == 8, {"validity_gate_pass": True}


def fake_encoded(env, reward, done, valid):
    return prefix._canonical({"validity_gate_pass": valid, "continuation": {
        "experiment_contract_sha256": env.experiment_contract_fingerprint,
        "memory": env.memory, "prepares": env.prepares, "reward": reward, "done": done,
    }})


class MatchedPrefixTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.output = self.root / "prefix"
        self.trace = self.root / "trace.jsonl"
        self.stop = self.root / "STOP"
        self.config = {"scenario": "fake", "t_total": 1800,
                       "experiment_contract_sha256": AUDITED_CONTRACT_SHA256}
        self.catalog = FakeCatalog()
        FakeEnv.after_commit = None
        env = FakeEnv("fake", 1800)
        env.response_equivalence_mode = prefix.CONTINUATION_EQUIVALENCE
        observations, rewards, self.rows = [env._observe()], [], []
        actions = [0, 2, 0, 1, 0, 2, 1, 0]
        for step, action in enumerate(actions):
            state = env._anchor_context_state_fingerprint()
            obs, reward, done, info = commit_catalog_action_from_anchor(env, self.catalog, action)
            observations.append(obs)
            rewards.append(reward)
            self.rows.append({"control_step": step, "state_id": state,
                              "selected_action_id": action, "interval_reward": reward,
                              "catalog_fingerprint": self.catalog.fingerprint,
                              "response_equivalence_mode": prefix.CONTINUATION_EQUIVALENCE,
                              "candidate_continuation_identities": {
                                  str(action): json.loads(fake_encoded(env, reward, done, True))}})
        manifest = make_replay_manifest(transition_count=8, action_count=3,
            catalog_fingerprint=self.catalog.fingerprint, observation_schema={"names": ["step", "sum"]},
            response_contract="fake", scenario="fake", source="fake")
        manifest.update(experiment_contract_sha256=AUDITED_CONTRACT_SHA256, t_total_sec=1800,
            response_equivalence_mode=prefix.CONTINUATION_EQUIVALENCE,
            reward_semantics="interval_negative_ttt", done_semantics="environment_terminal")
        self.replay = FrozenResponseReplay(
            np.array(observations[:-1]), np.array(actions), np.array(rewards, dtype=np.float32),
            np.array(observations[1:]), np.array([0] * 7 + [1], dtype=np.float32), np.ones(8, dtype=int),
            np.ones((8, 3), dtype=bool), np.ones((8, 3), dtype=bool),
            np.zeros((8, 3, 1), dtype=np.float32), np.zeros((8, 3, 1), dtype=np.float32),
            np.array(["fake"] * 8), np.zeros(8, dtype=int), np.arange(8), manifest).validate()
        self.write_trace(self.rows)
        FakeEnv.events.clear()
        self.addCleanup(setattr, FakeEnv, "after_commit", None)
        for name, value in (("RLLeaderEnv", FakeEnv), ("_encoded_continuation", fake_encoded)):
            patcher = patch.object(prefix, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def write_trace(self, rows, path=None):
        (path or self.trace).write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    def capture(self, **changes):
        args = dict(environment_config=self.config, catalog=self.catalog, replay=self.replay,
                    trace_paths=[self.trace], steps=[1, 7], output_dir=self.output, stop_files=[self.stop])
        args.update(changes)
        return prefix.capture_matched_prefix(**args)

    def checkpoint(self):
        return pickle.loads((self.output / "prefix.pkl").read_bytes())

    def test_targets_are_pre_action_compatible_and_reused_without_commits(self):
        paths = self.capture()
        self.assertEqual(FakeEnv.events.count("reset"), 1)
        self.assertEqual(FakeEnv.events.count("prepare"), 7)
        self.assertEqual([event for event in FakeEnv.events if isinstance(event, int)],
                         self.replay.action_id[:7].tolist())
        for step, path in paths.items():
            snapshot = load_verified_snapshot({**self.config, "snapshot": str(path),
                                               "reference_trace": str(self.trace)})
            self.assertEqual(snapshot.control_step, step)
            np.testing.assert_array_equal(snapshot.observation, self.replay.observation[step])
        before = {p: p.read_bytes() for p in self.output.iterdir()}
        FakeEnv.events.clear()
        self.assertEqual(self.capture(), paths)
        self.assertEqual(FakeEnv.events, [])
        self.assertEqual(before, {p: p.read_bytes() for p in self.output.iterdir()})

    def test_stop_before_reset_and_on_completed_reuse(self):
        self.stop.touch()
        with self.assertRaises(InterruptedError):
            self.capture()
        self.assertFalse(self.output.exists())
        self.assertEqual(FakeEnv.events, [])
        self.stop.unlink()
        self.capture()
        FakeEnv.events.clear()
        self.stop.touch()
        with self.assertRaises(InterruptedError):
            self.capture()
        self.assertEqual(FakeEnv.events, [])

    def test_midprefix_stop_resumes_without_repeating_completed_intervals(self):
        FakeEnv.after_commit = lambda env: self.stop.touch() if env.step_idx - env.warmup == 3 else None
        with self.assertRaises(InterruptedError):
            self.capture()
        self.assertEqual(self.checkpoint()["snapshot"].control_step, 3)
        first = (self.output / "step_001.pkl").read_bytes()
        self.stop.unlink()
        FakeEnv.events.clear()
        self.capture()
        self.assertNotIn("reset", FakeEnv.events)
        self.assertEqual(FakeEnv.events.count("prepare"), 4)
        self.assertEqual(first, (self.output / "step_001.pkl").read_bytes())

    def test_crash_between_checkpoint_and_target_snapshot_is_recovered(self):
        atomic = prefix._atomic_pickle
        def crash(path, payload):
            if path.name == "step_001.pkl":
                raise OSError("injected crash")
            atomic(path, payload)
        with patch.object(prefix, "_atomic_pickle", crash), self.assertRaisesRegex(OSError, "injected crash"):
            self.capture()
        self.assertEqual(self.checkpoint()["snapshot"].control_step, 1)
        self.assertFalse((self.output / "step_001.pkl").exists())
        FakeEnv.events.clear()
        self.capture()
        self.assertEqual(FakeEnv.events.count("prepare"), 6)
        self.assertNotIn("reset", FakeEnv.events)

    def test_input_changes_refuse_overwrite(self):
        self.capture()
        before = (self.output / "prefix.pkl").read_bytes()
        altered = replace(self.replay, event_group=np.array(["changed"] * 8))
        for changes in ({"steps": [1, 6]}, {"replay": altered},
                        {"environment_config": {**self.config, "additional_pin": "changed"}}):
            with self.subTest(changes=list(changes)), self.assertRaisesRegex(ValueError, "inputs changed"):
                self.capture(**changes)
        self.trace.write_text(self.trace.read_text().replace("\n", " \n"), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "inputs changed"):
            self.capture()
        self.assertEqual(before, (self.output / "prefix.pkl").read_bytes())

    def test_pinned_file_hash_and_unrecognized_root_fail_before_reset(self):
        config = {**self.config, "input_sha256": {str(self.trace): "0" * 64}}
        with self.assertRaisesRegex(ValueError, "pinned input changed"):
            self.capture(environment_config=config)
        self.assertEqual(FakeEnv.events, [])
        unknown = self.root / "unknown"
        unknown.mkdir()
        (unknown / "other.pkl").write_bytes(b"untouched")
        with self.assertRaisesRegex(ValueError, "unrecognized"):
            self.capture(output_dir=unknown)
        self.assertEqual((unknown / "other.pkl").read_bytes(), b"untouched")

    def test_trace_holes_conflicts_and_later_segment_supersession(self):
        stale = copy.deepcopy(self.rows)
        stale[2]["state_id"] = "stale"
        self.write_trace(self.rows[:2] + self.rows[3:])
        with self.assertRaisesRegex(ValueError, "hole"):
            self.capture(output_dir=self.root / "hole")
        self.write_trace(self.rows + [stale[2]])
        with self.assertRaisesRegex(ValueError, "conflicting duplicate"):
            self.capture(output_dir=self.root / "conflict")
        self.write_trace(stale)
        tail = self.root / "trace_resume_001.jsonl"
        self.write_trace(self.rows[2:], tail)
        self.capture(trace_paths=[self.trace, tail])

    def test_bad_observation_reward_state_contract_and_identity_fail_closed(self):
        for kind in ("observation", "reward", "state", "contract", "identity", "done"):
            with self.subTest(kind=kind):
                output = self.root / kind
                def bad_commit(env, catalog, action):
                    obs, reward, done, info = commit_catalog_action_from_anchor(env, catalog, action)
                    if kind == "observation": obs[1] += 1
                    if kind == "reward": reward -= 0.25
                    if kind == "state": env.action_sum += 1
                    if kind == "contract": env.experiment_contract_fingerprint = "bad"
                    if kind == "identity": env.memory += 1
                    if kind == "done": done = True
                    return obs, reward, done, info
                with patch.object(prefix, "commit_catalog_action_from_anchor", bad_commit):
                    with self.assertRaises((AssertionError, ValueError)):
                        self.capture(output_dir=output)
                saved = pickle.loads((output / "prefix.pkl").read_bytes())
                self.assertEqual(saved["snapshot"].control_step, 0)
                self.assertFalse((output / "step_001.pkl").exists())
                evidence = json.loads(next(output.glob("failure_*.json")).read_text())
                if kind == "identity":
                    self.assertIn("expected_continuation", evidence)
                    self.assertIn("actual_continuation", evidence)

    def test_snapshot_memory_drift_and_metadata_contract_rejected_on_reuse(self):
        paths = self.capture()
        snapshot = pickle.loads(paths[1].read_bytes())
        for kind in ("memory", "contract"):
            changed = copy.deepcopy(snapshot)
            if kind == "memory":
                changed.env.memory += 1
            else:
                changed = replace(changed, experiment_contract_sha256="bad")
            paths[1].write_bytes(pickle.dumps(changed))
            with self.subTest(kind=kind), self.assertRaisesRegex(ValueError, "continuation|incompatible"):
                self.capture()

    def test_frozen_real_full_identity_matches_and_deep_field_mutation_is_rejected(self):
        directory = Path(__file__).resolve().parents[2] / (
            "results/response_dqn_170_incident/sequential_value_head_v1/finite_horizon_cost/evaluation")
        if not (directory / "checkpoint.pkl").exists():
            self.skipTest("local frozen cost-head evaluation fixture is unavailable")
        from rl_leader.env import RLLeaderEnv
        with patch.object(RLLeaderEnv, "reset", side_effect=AssertionError("no real reset")), \
             patch.object(RLLeaderEnv, "prepare_pstack_anchor_context", side_effect=AssertionError("no real step")):
            saved = pickle.loads((directory / "checkpoint.pkl").read_bytes())
            summary = json.loads((directory / "summary.json").read_text())
            rows = {}
            for name in summary["trace_segments"]:
                path = Path(__file__).resolve().parents[2] / name
                rows.update({row["control_step"]: row for row in map(json.loads, path.read_text().splitlines())})
            final = rows[max(rows)]
            self.assertTrue(saved["rows"]["done"][-1])
            identity = final["candidate_continuation_identities"][str(final["selected_action_id"])]
            actual = real_encoded_continuation(saved["env"], final["interval_reward"], True, True)
        self.assertEqual(actual, prefix._canonical(identity))
        self.assertGreater(len(actual), 100000)
        self.rows[0]["candidate_continuation_identities"]["0"] = copy.deepcopy(identity)
        self.write_trace(self.rows)
        with patch.object(prefix, "_encoded_continuation", return_value=actual):
            self.capture(steps=[1])
            changed = self.rows[0]["candidate_continuation_identities"]["0"]["continuation"]
            component = changed["field_sha256"]["rl_follower"]
            component[next(iter(component))] = "0" * 64
            self.assertEqual(changed["sha256"], identity["continuation"]["sha256"])
            self.write_trace(self.rows)
            with self.assertRaisesRegex(ValueError, "full post-commit continuation mismatch"):
                self.capture(steps=[1], output_dir=self.root / "deep-mismatch")


if __name__ == "__main__":
    unittest.main()
