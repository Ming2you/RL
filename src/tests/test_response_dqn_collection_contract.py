from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from rl_leader import response_dqn_collect as collector
from rl_leader.evaluate_response_dqn_policy import _step_gated_policy
from rl_leader.response_dqn_catalog import DiscreteLeaderAction, StructuredActionCatalog
from rl_leader.response_dqn_mask import (
    CandidateResponse,
    build_response_mask,
    response_feature_matrix,
)


def _catalog():
    return StructuredActionCatalog(("budget.N_P",), [
        DiscreteLeaderAction(
            action_id=index,
            key="anchor" if index == 0 else f"action-{index}",
            domain="anchor" if index == 0 else "freeway",
            owner="P-Stack" if index == 0 else "test-owner",
            template="identity" if index == 0 else f"template-{index}",
            magnitude=0.25 * index,
            family="anchor" if index == 0 else "linear",
            residual=(0.25 * index,),
        )
        for index in range(4)
    ])


def _model(catalog, values, support):
    return SimpleNamespace(
        catalog_fingerprint=catalog.fingerprint,
        config=SimpleNamespace(min_action_support=2),
        action_support_counts=np.asarray(support, dtype=np.int64),
        q_values=lambda observation, response: np.asarray([values], dtype=np.float32),
    )


def _evaluated(observation, catalog, anchor_context=None):
    responses = tuple(
        CandidateResponse(index, (float(observation[0]), float(index)), f"memory-{index}")
        for index in range(catalog.size)
    )
    return collector.EvaluatedState(
        observation=np.asarray(observation, dtype=np.float32).copy(),
        anchor_context=anchor_context,
        candidate_responses=responses,
        response_mask=build_response_mask(responses, catalog_size=catalog.size),
        response_features=response_feature_matrix(responses, catalog_size=catalog.size),
        raw_candidate_count=catalog.size,
        unique_response_count=catalog.size,
        evaluation_seconds=0.0,
        response_workers=1,
        response_backend="serial",
    )


class EnsembleCollectionPolicyTest(unittest.TestCase):
    def setUp(self):
        self.catalog = _catalog()
        self.evaluated = SimpleNamespace(
            observation=np.asarray([1.0], dtype=np.float32),
            response_features=np.zeros((4, 2), dtype=np.float32),
            response_mask=SimpleNamespace(
                valid_action_mask=np.asarray([True, True, True, False]),
            ),
        )

    def _sample(self, policy):
        rng = np.random.default_rng(21)
        return {
            policy(self.evaluated, self.catalog, rng).action_id
            for _ in range(128)
        }

    def test_lcb_unsupported_exploration_is_opt_in_and_respects_mask(self):
        ensemble = [_model(self.catalog, [-10, 0, 100, 1000], [10, 0, 0, 10])]
        safe = collector.ensemble_lcb_policy(ensemble, exploration_epsilon=1.0)
        expanding = collector.ensemble_lcb_policy(
            ensemble, exploration_epsilon=1.0, explore_unsupported=True,
        )
        self.assertEqual(self._sample(safe), {0})
        self.assertEqual(self._sample(expanding), {0, 1, 2})
        self.assertEqual(self._sample(collector.ensemble_lcb_policy(
            ensemble, exploration_epsilon=0.0, explore_unsupported=True,
        )), {0})

    def test_greedy_exploration_includes_unsupported_but_never_masked_actions(self):
        ensemble = [_model(self.catalog, [-10, 0, 100, 1000], [10, 0, 0, 10])]
        policy = collector.ensemble_greedy_policy(ensemble, exploration_epsilon=1.0)
        self.assertEqual(self._sample(policy), {0, 1, 2})

    def test_greedy_uses_supported_ensemble_mean_without_lcb_guard(self):
        ensemble = [
            _model(self.catalog, [-100, -200, 1000, 10000], [10, 10, 0, 10]),
            _model(self.catalog, [-100, 2, 1000, 10000], [10, 10, 20, 10]),
        ]
        rng = np.random.default_rng(0)
        greedy = collector.ensemble_greedy_policy(ensemble)(self.evaluated, self.catalog, rng)
        guarded = collector.ensemble_lcb_policy(ensemble)(self.evaluated, self.catalog, rng)
        self.assertEqual(greedy.action_id, 1)
        self.assertEqual(greedy.diagnostics["q_selected"], -99.0)
        self.assertEqual(greedy.diagnostics["fallback_reason"], "")
        self.assertEqual(guarded.action_id, 0)

    def test_greedy_anchor_competes_by_q_without_fallback(self):
        ensemble = [_model(self.catalog, [-1, -10, 100, 1000], [10, 10, 0, 10])]
        decision = collector.ensemble_greedy_policy(ensemble)(
            self.evaluated, self.catalog, np.random.default_rng(0),
        )
        self.assertEqual(decision.action_id, 0)
        self.assertFalse(decision.diagnostics["exploratory"])
        self.assertEqual(decision.diagnostics["fallback_reason"], "")

    def test_policies_reject_invalid_epsilon(self):
        ensemble = [_model(self.catalog, [0, 0, 0, 0], [10, 10, 10, 10])]
        for factory in (collector.ensemble_lcb_policy, collector.ensemble_greedy_policy):
            for epsilon in (-0.01, 1.01, float("nan"), float("inf")):
                with self.subTest(policy=factory.__name__, epsilon=epsilon):
                    with self.assertRaisesRegex(ValueError, "exploration_epsilon"):
                        factory(ensemble, exploration_epsilon=epsilon)

    def test_policies_enforce_catalog_identity(self):
        model = _model(self.catalog, [0, 0, 0, 0], [10, 10, 10, 10])
        for factory in (collector.ensemble_lcb_policy, collector.ensemble_greedy_policy):
            with self.subTest(policy=factory.__name__):
                other = _model(self.catalog, [0, 0, 0, 0], [10, 10, 10, 10])
                other.catalog_fingerprint = "different-catalog"
                with self.assertRaisesRegex(ValueError, "different action catalogs"):
                    factory([model, other])
                policy = factory([model])
                with self.assertRaisesRegex(ValueError, "runtime action catalog differ"):
                    policy(self.evaluated, SimpleNamespace(fingerprint="other"), np.random.default_rng(0))


class _ShortEnv:
    warmup = 2
    scenario_name = "test-snapshot"
    experiment_contract_fingerprint = "test-experiment-contract"
    observation_schema = SimpleNamespace(metadata=lambda: {"dimension": 2})

    def __init__(self, start=5, stop=7):
        self.step_idx = self.warmup + start
        self.n_steps = self.warmup + stop
        self.T_total = float(self.n_steps * 60)
        self.reset_count = 0
        self.actions = []
        self.preview_steps = []

    def observation(self):
        step = self.step_idx - self.warmup
        return np.asarray([step, step * step], dtype=np.float32)

    def reset(self):
        self.reset_count += 1
        self.step_idx = self.warmup
        return self.observation()

    def prepare_pstack_anchor_context(self):
        return SimpleNamespace(state_fingerprint=f"state-{self.step_idx}")

    def _advance(self, action_id):
        step = self.step_idx - self.warmup
        self.actions.append((step, action_id))
        self.step_idx += 1
        return (
            self.observation(), -10.0 - step, self.step_idx >= self.n_steps,
            {"validity_gate_pass": True},
        )

    def step_anchored_candidate(self, residual, context):
        return self._advance(int(round(float(residual[0]) * 4)))

    def step_prepared_optimizer_anchor(self, context):
        return (*self._advance(0), np.zeros(1, dtype=np.float32))


class SequentialCollectionContractTest(unittest.TestCase):
    def setUp(self):
        self.catalog = _catalog()

    def _collect(self, env, policy=None, **kwargs):
        def preview(source, observation, catalog, **options):
            source.preview_steps.append(source.step_idx - source.warmup)
            return _evaluated(observation, catalog, source.prepare_pstack_anchor_context())

        if policy is None:
            policy = lambda evaluated, catalog, rng: collector.CollectionDecision(0, {})
        with patch.object(collector, "evaluate_executable_responses", side_effect=preview):
            with contextlib.redirect_stdout(io.StringIO()):
                return collector.collect_sequential_episode(
                    env, self.catalog, policy, episode=7, event_group="event-7",
                    rng=np.random.default_rng(0), **kwargs,
                )

    def test_snapshot_without_prior_rows_preserves_real_steps_and_next_states(self):
        env = _ShortEnv()
        checkpoints = []
        rows = self._collect(
            env, initial_observation=env.observation(), checkpoint_every=1,
            checkpoint_callback=lambda rows: checkpoints.append(list(rows["control_step"])),
        )
        self.assertEqual(env.reset_count, 0)
        self.assertEqual(env.preview_steps, [5, 6])
        self.assertEqual(rows["control_step"], [5, 6])
        self.assertEqual(rows["done"], [0.0, 1.0])
        np.testing.assert_array_equal(rows["observation"], [[5, 25], [6, 36]])
        np.testing.assert_array_equal(rows["next_observation"], [[6, 36], [7, 49]])
        np.testing.assert_array_equal(rows["next_response_features"][0], rows["response_features"][1])
        np.testing.assert_array_equal(rows["next_response_features"][1], np.zeros((4, 4)))
        self.assertEqual(checkpoints, [[5], [5, 6]])

    def test_intervention_prefix_continues_until_environment_terminal(self):
        env = _ShortEnv()
        before = _evaluated(env.observation(), self.catalog, env.prepare_pstack_anchor_context())
        next_obs, reward, done, _ = collector.commit_action(env, before, self.catalog, 1)
        after = _evaluated(next_obs, self.catalog)
        prefix = {
            "observation": [before.observation.copy()],
            "action_id": [1],
            "reward": [reward],
            "next_observation": [next_obs.copy()],
            "done": [float(done)],
            "option_steps": [1],
            "action_mask": [before.response_mask.valid_action_mask.copy()],
            "next_action_mask": [after.response_mask.valid_action_mask.copy()],
            "response_features": [before.response_features.copy()],
            "next_response_features": [after.response_features.copy()],
            "event_group": ["event-7"],
            "episode": [7],
            "control_step": [5],
        }
        checkpoint = Mock()
        rows = self._collect(
            env, initial_observation=next_obs, initial_rows=prefix,
            checkpoint_every=2, checkpoint_callback=checkpoint,
        )
        self.assertIs(rows, prefix)
        self.assertEqual(env.reset_count, 0)
        self.assertEqual(env.actions, [(5, 1), (6, 0)])
        self.assertEqual(rows["done"], [0.0, 1.0])
        self.assertEqual(rows["control_step"], [5, 6])
        self.assertEqual(rows["reward"], [-15.0, -16.0])
        np.testing.assert_array_equal(rows["next_observation"], [[6, 36], [7, 49]])
        np.testing.assert_array_equal(rows["next_observation"][0], rows["observation"][1])
        checkpoint.assert_called_once_with(rows)
        replay = collector.rows_to_replay(rows, env=env, catalog=self.catalog, source="test")
        self.assertEqual(replay.manifest["reward_semantics"], "interval_negative_ttt")
        self.assertEqual(replay.manifest["done_semantics"], "environment_terminal")
        self.assertEqual(replay.manifest["experiment_contract_sha256"], env.experiment_contract_fingerprint)
        self.assertEqual(replay.manifest["t_total_sec"], env.T_total)
        np.testing.assert_array_equal(replay.done, [0, 1])
        np.testing.assert_array_equal(replay.control_step, [5, 6])
        np.testing.assert_array_equal(replay.next_observation, [[6, 36], [7, 49]])

    def test_default_reset_and_step_gate_still_start_at_zero(self):
        env = _ShortEnv(start=5, stop=2)
        policy = Mock(return_value=collector.CollectionDecision(1, {}))
        rows = self._collect(
            env, policy=_step_gated_policy(policy, {1}),
            evaluate_response_steps=lambda step: step == 1,
            anchor_only_response_feature_dim=4,
        )
        self.assertEqual(env.reset_count, 1)
        self.assertEqual(env.preview_steps, [1])
        self.assertEqual(rows["control_step"], [0, 1])
        self.assertEqual(rows["action_id"], [0, 1])
        self.assertEqual(rows["done"], [0.0, 1.0])
        policy.assert_called_once()

    def test_legacy_anchor_only_fast_path_can_log_and_checkpoint(self):
        env = _ShortEnv(start=5, stop=2)
        env._inventory = lambda: 0.0
        checkpoints = []
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl"
            rows = self._collect(
                env, evaluate_response_steps=lambda step: step == 1,
                anchor_only_response_feature_dim=4, log_path=path, checkpoint_every=1,
                checkpoint_callback=lambda rows: checkpoints.append(len(rows["done"])),
            )
            traces = [json.loads(line) for line in path.read_text().splitlines()]
        self.assertEqual(rows["done"], [0.0, 1.0])
        self.assertEqual(checkpoints, [1, 2])
        self.assertIsNone(traces[0]["physical_control_group_count"])
        self.assertEqual(traces[1]["physical_control_group_count"], 4)

    def test_checkpoint_counts_rows_and_flushes_final_partial_batch(self):
        env = _ShortEnv(start=4, stop=7)
        checkpoints = []
        self._collect(
            env, initial_observation=env.observation(), checkpoint_every=2,
            checkpoint_callback=lambda rows: checkpoints.append(list(rows["control_step"])),
        )
        self.assertEqual(checkpoints, [[4, 5], [4, 5, 6]])

    def test_prior_rows_require_an_observation_without_resetting(self):
        env = _ShortEnv()
        with self.assertRaisesRegex(ValueError, "initial_rows requires initial_observation"):
            self._collect(env, initial_rows={})
        self.assertEqual(env.reset_count, 0)

    def test_terminal_snapshot_does_not_preview_or_advance(self):
        env = _ShortEnv(start=7, stop=7)
        rows = self._collect(env, initial_observation=env.observation())
        self.assertEqual(rows["action_id"], [])
        self.assertEqual(env.reset_count, 0)
        self.assertEqual(env.preview_steps, [])
        self.assertEqual(env.actions, [])

    def test_resume_rejects_disconnected_prefix(self):
        for changed in ("step", "observation", "response"):
            with self.subTest(changed=changed):
                env = _ShortEnv()
                captured = []

                def pause(rows):
                    captured.append(rows)
                    raise InterruptedError("test pause")

                with self.assertRaises(InterruptedError):
                    self._collect(env, initial_observation=env.observation(),
                                  checkpoint_every=1, checkpoint_callback=pause)
                rows = captured[0]
                if changed == "step":
                    rows["control_step"][-1] -= 1
                elif changed == "observation":
                    rows["next_observation"][-1][0] += 1
                else:
                    rows["next_response_features"][-1][0, 0] += 1
                with self.assertRaisesRegex(ValueError, "resumed|checkpoint"):
                    self._collect(env, initial_observation=env.observation(), initial_rows=rows)


if __name__ == "__main__":
    unittest.main()
