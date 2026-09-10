from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from rl_leader.env import make_cfg
from rl_leader.response_dqn import (
    ResponseDQNConfig,
    conservative_ensemble_selection,
    load_trained_response_dqn,
    masked_argmax,
    masked_epsilon_greedy,
    train_response_dqn_member,
)
from rl_leader.response_dqn_catalog import (
    StructuredActionCatalog,
    build_structured_action_catalog,
    load_extra_action_specs,
)
from rl_leader.response_dqn_data import (
    FrozenResponseReplay,
    load_frozen_response_replay,
    make_replay_manifest,
    merge_frozen_response_replays,
)
from rl_leader.response_dqn_expansion import expansion_priority
from rl_leader.response_dqn_collect import (
    _candidate_trials,
    evaluate_executable_responses,
    evaluate_anchor_response,
)
from rl_leader.collect_response_dqn_actors import build_actor_specs
from rl_leader import train_response_dqn
from rl_leader.response_dqn_mask import (
    CandidateResponse,
    build_response_mask,
    response_feature_matrix,
)
from src.controllers.coordination import CoordinationActionSchema, CoordinationMask
from src.models.state import ControlAction


def _synthetic_replay(action_count: int = 3) -> FrozenResponseReplay:
    rng = np.random.default_rng(4)
    n, obs_dim, response_dim = 18, 5, 4
    observation = rng.normal(size=(n, obs_dim)).astype(np.float32)
    next_observation = rng.normal(size=(n, obs_dim)).astype(np.float32)
    action_mask = np.ones((n, action_count), dtype=bool)
    next_action_mask = np.ones((n, action_count), dtype=bool)
    next_action_mask[:, -1] = False
    action_id = np.arange(n, dtype=np.int64) % max(1, action_count - 1)
    response = rng.normal(size=(n, action_count, response_dim)).astype(np.float32)
    next_response = rng.normal(size=(n, action_count, response_dim)).astype(np.float32)
    manifest = make_replay_manifest(
        transition_count=n,
        action_count=action_count,
        catalog_fingerprint="catalog-test",
        observation_schema={"version": "test", "dimension": obs_dim},
        response_contract="test-response",
        scenario="test-incident",
        source="unit-test",
    )
    return FrozenResponseReplay(
        observation=observation,
        action_id=action_id,
        reward=(-np.square(observation[:, 0]) + action_id * 0.1).astype(np.float32),
        next_observation=next_observation,
        done=np.asarray([0.0] * (n - 1) + [1.0], dtype=np.float32),
        option_steps=np.ones(n, dtype=np.int64),
        action_mask=action_mask,
        next_action_mask=next_action_mask,
        response_features=response,
        next_response_features=next_response,
        event_group=np.asarray([f"event-{index // 3}" for index in range(n)]),
        episode=np.asarray([index // 6 for index in range(n)], dtype=np.int64),
        control_step=np.asarray([index % 6 for index in range(n)], dtype=np.int64),
        manifest=manifest,
    ).validate()


class StructuredCatalogTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg, _ = make_cfg("medium_demand")
        cls.schema = CoordinationActionSchema(cls.cfg)

    def test_catalog_is_stable_and_anchor_is_identity(self):
        first = build_structured_action_catalog(
            self.schema.names, magnitudes=(0.25,), owners=("C",),
        )
        second = build_structured_action_catalog(
            self.schema.names, magnitudes=(0.25,), owners=("C",),
        )
        self.assertEqual(first.fingerprint, second.fingerprint)
        self.assertEqual(first.actions, second.actions)
        self.assertEqual(first.action(0).key, "anchor")
        self.assertTrue(np.array_equal(first.residual(0), np.zeros(self.schema.dimension)))
        restored = StructuredActionCatalog.from_manifest(first.as_manifest())
        self.assertEqual(restored.fingerprint, first.fingerprint)

    def test_catalog_contains_linear_quadratic_and_cross_operators(self):
        catalog = build_structured_action_catalog(
            self.schema.names, magnitudes=(0.25,), owners=("C",),
        )
        self.assertEqual({action.family for action in catalog.actions}, {
            "anchor", "linear", "quadratic", "cross",
        })
        cross = next(action for action in catalog.actions if action.template == "cross_negative")
        indices = {name: index for index, name in enumerate(self.schema.names)}
        coordinate = 0.25 / np.sqrt(3.0)
        self.assertAlmostEqual(cross.residual[indices["urban.C.l11"]], coordinate)
        self.assertAlmostEqual(cross.residual[indices["urban.C.l21"]], -coordinate)
        self.assertAlmostEqual(cross.residual[indices["urban.C.l22"]], coordinate)
        for action in catalog.actions[1:]:
            self.assertAlmostEqual(
                float(np.linalg.norm(action.residual_array())), action.magnitude,
            )

    def test_catalog_allows_full_scale_residuals(self):
        catalog = build_structured_action_catalog(
            self.schema.names,
            magnitudes=(1.0,),
            owners=("R_F_W",),
            families=("linear",),
        )
        action = next(
            item for item in catalog.actions
            if item.template == "linear_first_positive"
        )
        self.assertEqual(action.owner, "R_F_W")
        self.assertAlmostEqual(float(np.linalg.norm(action.residual_array())), 1.0)

    def test_combo_catalog_contains_directional_nonlinear_operators(self):
        catalog = build_structured_action_catalog(
            self.schema.names,
            magnitudes=(0.25,),
            owners=("C",),
            families=("combo",),
        )
        self.assertEqual({action.family for action in catalog.actions}, {
            "anchor", "combo",
        })
        combo = next(
            action for action in catalog.actions
            if action.template == "combo_corner_pp_cross_pos"
        )
        indices = {name: index for index, name in enumerate(self.schema.names)}
        self.assertGreater(combo.residual[indices["urban.C.g_green"]], 0.0)
        self.assertGreater(combo.residual[indices["urban.C.g_offset"]], 0.0)
        self.assertGreater(combo.residual[indices["urban.C.l11"]], 0.0)
        self.assertGreater(combo.residual[indices["urban.C.l21"]], 0.0)
        self.assertGreater(combo.residual[indices["urban.C.l22"]], 0.0)
        self.assertAlmostEqual(
            float(np.linalg.norm(combo.residual_array())),
            combo.magnitude,
        )

    def test_hybrid_catalog_keeps_linear_direction_with_small_curvature(self):
        catalog = build_structured_action_catalog(
            self.schema.names,
            magnitudes=(0.25,),
            owners=("C",),
            families=("hybrid",),
        )
        self.assertEqual({action.family for action in catalog.actions}, {
            "anchor", "hybrid",
        })
        hybrid = next(
            action for action in catalog.actions
            if action.template == "hybrid_corner_pp_cross_pos"
        )
        indices = {name: index for index, name in enumerate(self.schema.names)}
        green = hybrid.residual[indices["urban.C.g_green"]]
        offset = hybrid.residual[indices["urban.C.g_offset"]]
        l11 = hybrid.residual[indices["urban.C.l11"]]
        l21 = hybrid.residual[indices["urban.C.l21"]]
        l22 = hybrid.residual[indices["urban.C.l22"]]
        self.assertGreater(green, 0.0)
        self.assertGreater(offset, 0.0)
        self.assertGreater(l11, 0.0)
        self.assertGreater(l21, 0.0)
        self.assertGreater(l22, 0.0)
        self.assertGreater(green, l11)
        self.assertGreater(offset, l22)

    def test_catalog_appends_sparse_extra_residual_action(self):
        catalog = build_structured_action_catalog(
            self.schema.names,
            magnitudes=(0.25,),
            owners=("C",),
            families=("linear",),
            extra_actions=[{
                "key": "artifact:multifield_test",
                "owner": "artifact",
                "template": "multifield_test",
                "family": "hybrid",
                "residual_nonzero": {
                    "budget.N_P": 0.25,
                    "urban.A.g_green": 1.0,
                    "freeway.R_F_W.g_meter": 0.5,
                },
            }],
        )
        extra = catalog.actions[-1]
        indices = {name: index for index, name in enumerate(self.schema.names)}
        self.assertEqual(extra.key, "artifact:multifield_test")
        self.assertEqual(extra.domain, "freeway")
        self.assertEqual(extra.family, "hybrid")
        self.assertAlmostEqual(extra.residual[indices["budget.N_P"]], 0.25)
        self.assertAlmostEqual(extra.residual[indices["urban.A.g_green"]], 1.0)
        self.assertAlmostEqual(extra.residual[indices["freeway.R_F_W.g_meter"]], 0.5)
        restored = StructuredActionCatalog.from_manifest(catalog.as_manifest())
        self.assertEqual(restored.fingerprint, catalog.fingerprint)

    def test_loads_neighborhood_manifest_as_extra_actions(self):
        indices = {name: index for index, name in enumerate(self.schema.names)}
        residual = np.zeros(self.schema.dimension, dtype=np.float32)
        residual[indices["urban.A.g_green"]] = 0.75
        payload = {
            "format_version": "residual_neighborhood_search_v1",
            "candidates": [{
                "label": "all_s0.75_at18",
                "metadata": {"family": "scaled_all", "scale": 0.75},
                "schedule": [{
                    "label": "all_s0.75_at18",
                    "residual": residual.tolist(),
                    "step": 18,
                }],
            }],
        }
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "manifest.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            specs = load_extra_action_specs((path,))
        catalog = build_structured_action_catalog(
            self.schema.names,
            magnitudes=(0.25,),
            owners=("C",),
            families=("linear",),
            extra_actions=specs,
        )
        extra = catalog.actions[-1]
        self.assertEqual(extra.key, "artifact:all_s0.75_at18")
        self.assertEqual(extra.family, "hybrid")
        self.assertEqual(extra.domain, "urban")
        self.assertAlmostEqual(extra.magnitude, 0.75)
        self.assertAlmostEqual(extra.residual[indices["urban.A.g_green"]], 0.75)

    def test_loads_cached_tail_sampler_artifact_as_extra_actions(self):
        indices = {name: index for index, name in enumerate(self.schema.names)}
        residual = np.zeros(self.schema.dimension, dtype=np.float32)
        residual[indices["freeway.R_F_W.g_meter"]] = 1.0
        residual[indices["urban.B.g_green"]] = -0.25
        payload = {
            "format_version": "cached_tail_residual_sampler_v1",
            "control_step": 18,
            "horizon_steps": 12,
            "h12_records": [{
                "candidate_id": "random:test_candidate:abc123",
                "label": "test_candidate",
                "generator": "random",
                "family": "random_multiblock",
                "continuous_residual": residual.tolist(),
                "horizon_labels": {
                    "12": {
                        "ttt_gain": 33.0,
                        "positive": True,
                    },
                },
            }],
        }
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "summary.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            specs = load_extra_action_specs((path,))
        catalog = build_structured_action_catalog(
            self.schema.names,
            magnitudes=(0.25,),
            owners=("C",),
            families=("linear",),
            extra_actions=specs,
        )
        extra = catalog.actions[-1]
        self.assertEqual(extra.key, "artifact:random:test_candidate:abc123")
        self.assertEqual(extra.owner, "random")
        self.assertEqual(extra.template, "test_candidate")
        self.assertEqual(extra.family, "hybrid")
        self.assertEqual(extra.domain, "freeway")
        self.assertAlmostEqual(extra.residual[indices["freeway.R_F_W.g_meter"]], 1.0)
        self.assertAlmostEqual(extra.residual[indices["urban.B.g_green"]], -0.25)

    def test_catalog_deduplicates_identical_extra_action_keys(self):
        indices = {name: index for index, name in enumerate(self.schema.names)}
        sparse = {
            "freeway.R_F_W.g_meter": 1.0,
            "urban.B.g_green": -0.25,
        }
        catalog = build_structured_action_catalog(
            self.schema.names,
            magnitudes=(0.25,),
            owners=("C",),
            families=("linear",),
            extra_actions=[
                {
                    "key": "artifact:duplicated_sampler_candidate",
                    "owner": "artifact",
                    "family": "hybrid",
                    "residual_nonzero": sparse,
                },
                {
                    "key": "artifact:duplicated_sampler_candidate",
                    "owner": "artifact",
                    "family": "hybrid",
                    "residual_nonzero": sparse,
                },
            ],
        )
        matches = [
            action for action in catalog.actions
            if action.key == "artifact:duplicated_sampler_candidate"
        ]
        self.assertEqual(len(matches), 1)
        self.assertAlmostEqual(matches[0].residual[indices["freeway.R_F_W.g_meter"]], 1.0)
        self.assertAlmostEqual(matches[0].residual[indices["urban.B.g_green"]], -0.25)

    def test_zero_catalog_action_preserves_native_anchor(self):
        previous = ControlAction.fixed(self.cfg)
        raw = np.linspace(-0.5, 0.5, self.schema.dimension, dtype=np.float32)
        anchor = self.schema.decode(raw, previous, CoordinationMask.named("RL-FULL"))
        catalog = build_structured_action_catalog(
            self.schema.names, magnitudes=(0.25,), owners=("C",),
        )
        replay = self.schema.decode_anchored_residual(
            catalog.residual(0), anchor, CoordinationMask.named("RL-FULL"),
        )
        self.assertEqual(replay, anchor)


class ResponseMaskTest(unittest.TestCase):
    def test_anchor_wins_its_response_group_and_invalid_is_masked(self):
        responses = [
            CandidateResponse(0, (1.0, 2.0), "memory-a"),
            CandidateResponse(1, (1.0, 2.0 + 1.0e-7), "memory-a"),
            CandidateResponse(2, (2.0, 3.0), "memory-b"),
            CandidateResponse(3, (9.0, 9.0), "memory-c", False, "infeasible"),
        ]
        result = build_response_mask(responses, catalog_size=4)
        self.assertEqual(result.valid_action_mask.tolist(), [True, False, True, False])
        self.assertEqual(result.representative_of.tolist(), [0, 0, 2, -1])
        self.assertEqual(result.invalid_reasons[1], "response_duplicate_of:0")
        features = response_feature_matrix(responses, catalog_size=4)
        self.assertEqual(features.shape, (4, 4))
        np.testing.assert_allclose(features[0, 2:], 0.0)


class ParallelResponseEvaluationTest(unittest.TestCase):
    def test_anchor_only_preview_masks_candidates_without_evaluating_them(self):
        class FakeEnv:
            def prepare_pstack_anchor_context(self):
                return SimpleNamespace(state_fingerprint="state-1")

        catalog = SimpleNamespace(
            size=3,
            actions=[SimpleNamespace(action_id=index) for index in range(3)],
        )
        calls = []

        def fake_trial(source, anchor_context, catalog, action_id):
            calls.append(int(action_id))
            return np.asarray([4.0, 7.0], dtype=np.float32), "memory-anchor", True, ""

        with patch(
            "rl_leader.response_dqn_collect._candidate_trial",
            side_effect=fake_trial,
        ):
            evaluated = evaluate_anchor_response(
                FakeEnv(),
                np.asarray([1.0, 2.0], dtype=np.float32),
                catalog,
            )

        self.assertEqual(calls, [0])
        self.assertEqual(evaluated.raw_candidate_count, 1)
        self.assertEqual(evaluated.response_backend, "anchor_only")
        self.assertEqual(
            evaluated.response_mask.valid_action_mask.tolist(),
            [True, False, False],
        )
        self.assertEqual(
            evaluated.response_mask.invalid_reasons[1],
            "preview_skipped_by_policy_gate",
        )
        self.assertEqual(evaluated.response_features.shape, (3, 4))
        np.testing.assert_allclose(evaluated.response_features[:, 2:], 0.0)

    def test_filtered_preview_evaluates_only_requested_candidates(self):
        class FakeEnv:
            def prepare_pstack_anchor_context(self):
                return SimpleNamespace(state_fingerprint="state-1")

        catalog = SimpleNamespace(
            size=4,
            actions=[SimpleNamespace(action_id=index) for index in range(4)],
        )
        calls = []

        def fake_trial(source, anchor_context, catalog, action_id):
            calls.append(int(action_id))
            return (
                np.asarray([float(action_id), 10.0], dtype=np.float32),
                f"memory-{action_id}",
                True,
                "",
            )

        with patch(
            "rl_leader.response_dqn_collect._candidate_trial",
            side_effect=fake_trial,
        ):
            evaluated = evaluate_executable_responses(
                FakeEnv(),
                np.asarray([1.0, 2.0], dtype=np.float32),
                catalog,
                action_ids=(0, 2),
            )

        self.assertEqual(calls, [0, 2])
        self.assertEqual(evaluated.raw_candidate_count, 2)
        self.assertEqual(
            evaluated.response_mask.valid_action_mask.tolist(),
            [True, False, True, False],
        )
        self.assertEqual(
            evaluated.response_mask.invalid_reasons[1],
            "preview_skipped_by_action_filter",
        )
        self.assertEqual(evaluated.response_features.shape, (4, 4))

    def test_anchor_commit_only_state_skips_response_preview(self):
        class FakeEnv:
            def prepare_pstack_anchor_context(self):
                return SimpleNamespace(state_fingerprint="state-1")

        catalog = SimpleNamespace(
            size=3,
            actions=[SimpleNamespace(action_id=index) for index in range(3)],
        )

        with patch("rl_leader.response_dqn_collect._candidate_trial") as trial:
            evaluated = evaluate_anchor_response(
                FakeEnv(),
                np.asarray([1.0, 2.0], dtype=np.float32),
                catalog,
                response_feature_dim=6,
            )

        trial.assert_not_called()
        self.assertEqual(evaluated.raw_candidate_count, 0)
        self.assertEqual(evaluated.response_backend, "anchor_commit_only")
        self.assertEqual(
            evaluated.response_mask.valid_action_mask.tolist(),
            [True, False, False],
        )
        self.assertEqual(evaluated.response_features.shape, (3, 6))
        np.testing.assert_allclose(evaluated.response_features, 0.0)

    def test_threaded_trials_preserve_action_order(self):
        action_ids = [5, 2, 9, 1]

        def fake_trial(source, anchor_context, catalog, action_id):
            time.sleep(0.002 * (10 - action_id))
            return np.asarray([action_id], dtype=np.float32), str(action_id), True, ""

        with patch(
            "rl_leader.response_dqn_collect._candidate_trial",
            side_effect=fake_trial,
        ):
            serial = _candidate_trials(
                object(), object(), object(), action_ids, workers=1,
            )
            threaded = _candidate_trials(
                object(), object(), object(), action_ids, workers=8,
            )

        self.assertEqual(list(serial), action_ids)
        self.assertEqual(list(threaded), action_ids)
        for action_id in action_ids:
            np.testing.assert_array_equal(
                serial[action_id][0], threaded[action_id][0],
            )

    def test_trials_reject_nonpositive_worker_count(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            _candidate_trials(
                object(), object(), object(), [1], workers=0,
            )

    def test_trials_reject_unknown_backend(self):
        with self.assertRaisesRegex(ValueError, "backend"):
            _candidate_trials(
                object(), object(), object(), [1], workers=1, backend="unknown",
            )

    def test_actor_specs_have_unique_episode_seed_and_paths(self):
        specs = build_actor_specs(
            scenario="incident",
            t_total=14400.0,
            episodes=3,
            episode_start=4,
            seed_start=8,
            magnitudes=(0.25,),
            families=("linear",),
            domains=("freeway",),
            owners=("R_F_E",),
            epsilon=1.0,
            anchor_probability=0.25,
            checkpoint_every=1,
            output_dir=Path("data"),
            log_dir=Path("logs"),
        )
        self.assertEqual([spec.episode for spec in specs], [4, 5, 6])
        self.assertEqual([spec.seed for spec in specs], [8, 9, 10])
        self.assertEqual(len({spec.output_path for spec in specs}), 3)
        self.assertEqual(len({spec.log_path for spec in specs}), 3)


class ReplayAndDQNTest(unittest.TestCase):
    def test_trainer_thread_configuration_is_idempotent_per_process(self):
        class FakeTorch:
            def __init__(self):
                self.thread_calls = []
                self.interop_calls = 0

            def set_num_threads(self, count):
                self.thread_calls.append(count)

            def set_num_interop_threads(self, count):
                self.interop_calls += 1
                if self.interop_calls > 1:
                    raise RuntimeError(
                        "cannot set number of interop threads after parallel work has started"
                    )

        fake = FakeTorch()
        with patch.object(train_response_dqn, "torch", fake):
            old = train_response_dqn._TORCH_INTEROP_THREADS_CONFIGURED
            try:
                train_response_dqn._TORCH_INTEROP_THREADS_CONFIGURED = False
                train_response_dqn._configure_torch_threads(2)
                train_response_dqn._configure_torch_threads(3)
            finally:
                train_response_dqn._TORCH_INTEROP_THREADS_CONFIGURED = old

        self.assertEqual(fake.thread_calls, [2, 3])
        self.assertEqual(fake.interop_calls, 1)

    def test_replay_round_trip_and_group_bootstrap(self):
        replay = _synthetic_replay()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "replay.npz"
            replay.save(path)
            loaded = load_frozen_response_replay(path)
        self.assertEqual(loaded.size, replay.size)
        self.assertEqual(set(loaded.unique_event_groups()), set(replay.unique_event_groups()))
        sampled = loaded.bootstrap_indices(np.random.default_rng(0))
        self.assertGreater(sampled.size, 0)
        for group in loaded.unique_event_groups():
            count = np.count_nonzero(loaded.event_group[sampled] == group)
            self.assertEqual(count % np.count_nonzero(loaded.event_group == group), 0)

    def test_replay_merge_preserves_contract_and_support(self):
        first = _synthetic_replay()
        second = FrozenResponseReplay(**{
            **first.__dict__,
            "event_group": np.asarray([f"second-{value}" for value in first.event_group]),
            "episode": first.episode + 10,
        }).validate()
        merged = merge_frozen_response_replays(
            [first, second], source="unit-test-merge",
        )
        self.assertEqual(merged.size, first.size * 2)
        np.testing.assert_array_equal(
            merged.action_support_counts(), first.action_support_counts() * 2,
        )
        self.assertEqual(merged.manifest["merged_batch_count"], 2)

    def test_replay_rejects_a_selected_masked_action(self):
        replay = _synthetic_replay()
        bad_mask = replay.action_mask.copy()
        row = int(np.flatnonzero(replay.action_id != 0)[0])
        bad_mask[row, replay.action_id[row]] = False
        broken = FrozenResponseReplay(**{
            **replay.__dict__, "action_mask": bad_mask,
        })
        with self.assertRaisesRegex(ValueError, "masked"):
            broken.validate()

    def test_masked_double_dqn_argmax_excludes_invalid_action(self):
        q = torch.tensor([[1.0, 99.0, 2.0], [5.0, 4.0, 100.0]])
        mask = torch.tensor([[True, False, True], [True, True, False]])
        self.assertEqual(masked_argmax(q, mask).tolist(), [2, 0])

    def test_frozen_batch_training_smoke(self):
        replay = _synthetic_replay()
        candidates = np.asarray([
            [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0],
        ], dtype=np.float32)
        model = train_response_dqn_member(
            replay,
            candidates,
            catalog_fingerprint="catalog-test",
            config=ResponseDQNConfig(
                batch_size=8,
                gradient_steps=12,
                target_update_interval=4,
                hidden=(16,),
                ensemble_size=1,
            ),
            seed=3,
            bootstrap=False,
        )
        q_values = model.q_values(
            replay.observation[0], replay.response_features[0],
        )
        self.assertEqual(q_values.shape, (1, replay.action_count))
        self.assertTrue(np.all(np.isfinite(q_values)))
        self.assertEqual(len(model.training_losses), 12)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            model.save(path)
            loaded = load_trained_response_dqn(
                path, expected_catalog_fingerprint="catalog-test",
            )
            np.testing.assert_allclose(
                loaded.q_values(replay.observation[0], replay.response_features[0]),
                q_values,
            )

    def test_training_rejects_missing_anchor_support(self):
        replay = _synthetic_replay()
        actions = np.ones_like(replay.action_id)
        unsupported = FrozenResponseReplay(**{
            **replay.__dict__, "action_id": actions,
        }).validate()
        with self.assertRaisesRegex(ValueError, "anchor lacks"):
            train_response_dqn_member(
                unsupported,
                np.ones((replay.action_count, 2), dtype=np.float32),
                catalog_fingerprint="catalog-test",
                config=ResponseDQNConfig(
                    batch_size=4, gradient_steps=1, hidden=(8,), ensemble_size=1,
                ),
                bootstrap=False,
            )

    def test_conservative_selection_and_anchor_fallback(self):
        q = np.asarray([
            [5.0, 6.0, 9.0],
            [5.0, 6.2, 1.0],
            [5.0, 6.1, 8.0],
        ])
        decision = conservative_ensemble_selection(
            q, np.asarray([True, True, True]), z_value=1.0,
            material_margin=0.5,
        )
        self.assertEqual(decision.action_id, 1)
        self.assertFalse(decision.fallback)
        fallback = conservative_ensemble_selection(
            q, np.asarray([True, True, True]), z_value=1.0,
            material_margin=2.0,
        )
        self.assertEqual(fallback.action_id, 0)
        self.assertTrue(fallback.fallback)

    def test_epsilon_greedy_and_expansion_never_use_invalid_action(self):
        rng = np.random.default_rng(10)
        mask = np.asarray([True, False, True])
        sampled = {
            masked_epsilon_greedy(
                np.asarray([0.0, 100.0, 1.0]), mask, epsilon=1.0, rng=rng,
            )
            for _ in range(50)
        }
        self.assertTrue(sampled <= {0, 2})
        priority = expansion_priority(
            np.asarray([[0.0, 3.0, 0.2], [0.0, -3.0, 0.4]]),
            np.asarray([10, 0, 2]),
            mask,
        )
        self.assertEqual(priority["score"][1], -np.inf)


if __name__ == "__main__":
    unittest.main()
