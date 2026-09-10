"""Auditable finite greedy-path backups; the default learner stays one-step."""
import contextlib
import io
import json
import random
import tempfile
import unittest
from dataclasses import fields, replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from rl_leader import response_dqn, train_response_dqn
from rl_leader.response_dqn import (
    CandidateQNetwork, FeatureNormalizer, ResponseDQNConfig, _normalized_tensors,
    build_sequential_links, greedy_consistent_ddqn_targets,
    load_trained_response_dqn, masked_argmax, train_response_dqn_member,
)
from rl_leader.response_dqn_data import FrozenResponseReplay, make_replay_manifest
from src.tests.test_response_dqn import _synthetic_replay
from src.tests.test_response_value_head import phase_replay


def chain_replay(n=6, *, terminal=True):
    observation = np.arange(n + 1, dtype=np.float32)[:, None]
    response = np.arange((n + 1) * 4, dtype=np.float32).reshape(n + 1, 4, 1)
    mask = np.ones((n + 1, 4), bool)
    done = np.zeros(n, np.float32)
    if terminal:
        done[-1] = 1
        mask[-1, 1:] = False
    manifest = make_replay_manifest(
        transition_count=n, action_count=4, catalog_fingerprint="chain-toy",
        observation_schema={"dimension": 1}, response_contract="toy",
        scenario="toy", source="unit-test",
    )
    manifest.update(reward_semantics="interval_negative_ttt", done_semantics="environment_terminal")
    return FrozenResponseReplay(
        observation=observation[:-1].copy(), next_observation=observation[1:].copy(),
        response_features=response[:-1].copy(), next_response_features=response[1:].copy(),
        action_mask=mask[:-1].copy(), next_action_mask=mask[1:].copy(),
        action_id=np.zeros(n, np.int64), reward=-np.arange(1, n + 1, dtype=np.float32),
        done=done, option_steps=np.ones(n, np.int64), event_group=np.full(n, "shared-event"),
        episode=np.zeros(n, np.int64), control_step=np.arange(n, dtype=np.int64),
        manifest=manifest,
    ).validate()


def take_rows(replay, rows):
    arrays = {field.name: getattr(replay, field.name)[rows].copy()
              for field in fields(replay) if field.name != "manifest"}
    return replace(replay, **arrays, manifest={**replay.manifest, "transition_count": len(rows)})


class TableQ(torch.nn.Module):
    def __init__(self, values):
        super().__init__()
        self.values = torch.nn.Parameter(torch.as_tensor(values, dtype=torch.float32).clone())
        self.seen = []

    def forward(self, observation, candidates, response, *, remaining_intervals=None):
        assert not torch.is_grad_enabled()
        self.seen.extend(observation[:, 0].long().tolist())
        return self.values[observation[:, 0].long()]


def table_networks(n=6):
    online = TableQ([[4, 3, 2, 1]] * (n + 1))
    # The target network prefers action 1; DDQN must still evaluate action 0.
    target = TableQ([[10 * state, 1000, 2000, 3000] for state in range(n + 1)])
    return online, target


def audit_targets(replay, online, target, *, horizon=5, indices=None, support=None, scale=1):
    normalizer = FeatureNormalizer(np.zeros(1), np.ones(1), np.zeros(1), np.ones(1))
    tensors = _normalized_tensors(replay, normalizer, torch.device("cpu"))
    return greedy_consistent_ddqn_targets(
        online, target, tensors, torch.eye(replay.action_count),
        torch.ones(replay.action_count, dtype=torch.bool) if support is None else torch.tensor(support),
        torch.as_tensor(build_sequential_links(replay)),
        torch.arange(replay.size) if indices is None else torch.tensor(indices),
        backup_horizon=horizon, reward_scale=scale,
    )


def train_tiny(replay, *, bootstrap=False, stop_files=(), **options):
    config = ResponseDQNConfig(**{
        "gamma": 1, "backup_horizon": 5, "gradient_steps": 3,
        "hidden": (8,), "batch_size": 4, "ensemble_size": 1, **options,
    })
    return train_response_dqn_member(
        replay, np.eye(replay.action_count, dtype=np.float32),
        catalog_fingerprint=replay.manifest["catalog_fingerprint"],
        config=config, seed=17, bootstrap=bootstrap, stop_files=stop_files,
    )


class SequentialLinksTest(unittest.TestCase):
    def test_unsorted_rows_link_by_episode_and_step(self):
        replay = take_rows(chain_replay(), [3, 1, 5, 0, 4, 2])
        links = build_sequential_links(replay)
        self.assertEqual(links.dtype, np.dtype("int64"))
        np.testing.assert_array_equal(links, [4, 5, -1, 1, 2, 0])

    def test_gaps_and_episode_boundaries_never_link(self):
        replay = take_rows(chain_replay(), [0, 2, 3, 5])
        np.testing.assert_array_equal(build_sequential_links(replay), [-1, 2, -1, -1])
        replay = chain_replay()
        replay.episode[3:] = 1
        np.testing.assert_array_equal(build_sequential_links(replay), [1, 2, -1, 4, 5, -1])

    def test_duplicate_keys_rejected_even_across_event_groups(self):
        replay = take_rows(chain_replay(), [0, 0])
        replay.event_group[1] = "other-event"
        with self.assertRaisesRegex(ValueError, "duplicate sequential key"):
            build_sequential_links(replay)

    def test_terminal_followed_by_data_rejected_including_gaps(self):
        for rows in ([0, 1, 2], [0, 2, 5], [5, 2, 0]):
            replay = take_rows(chain_replay(), rows)
            replay.done[replay.control_step == 0] = 1
            with self.subTest(rows=rows), self.assertRaisesRegex(ValueError, "terminal.*followed"):
                build_sequential_links(replay)

    def test_all_adjacent_next_fields_must_match_exactly(self):
        for name in ("next_observation", "next_response_features", "next_action_mask"):
            replay = chain_replay()
            values = getattr(replay, name)
            if name == "next_action_mask":
                values[1, -1] = False
            else:
                index = (1,) + (0,) * (values.ndim - 1)
                values[index] = np.nextafter(values[index], np.float32(np.inf))
            with self.subTest(field=name), self.assertRaisesRegex(ValueError, name):
                build_sequential_links(replay)

    def test_fractional_chain_keys_rejected(self):
        for name in ("episode", "control_step"):
            replay = chain_replay()
            bad = replace(replay, **{name: getattr(replay, name).astype(float) + .5})
            with self.subTest(field=name), self.assertRaisesRegex(ValueError, "integer"):
                build_sequential_links(bad)


class GreedyPathTargetTest(unittest.TestCase):
    def test_exact_one_step_ddqn_values(self):
        replay = chain_replay()
        online, target = table_networks()
        actual, horizons = audit_targets(replay, online, target, horizon=1, scale=.25)
        with torch.no_grad():
            next_states = torch.as_tensor(replay.next_observation[:, 0], dtype=torch.long)
            selected = masked_argmax(online.values[next_states], torch.tensor(replay.next_action_mask))
            q_next = target.values[next_states].gather(1, selected[:, None]).squeeze(1)
            expected = .25 * torch.tensor(replay.reward) + (1 - torch.tensor(replay.done)) * q_next
        self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(horizons.tolist(), [1] * 6)
        self.assertFalse(actual.requires_grad)

    def test_analytic_three_and_five_step_rewards_and_terminal(self):
        replay = chain_replay()
        for cap, expected, horizon in (
            (3, [24, 31, 38, -15, -11, -6], [3, 3, 3, 3, 2, 1]),
            (5, [35, -20, -18, -15, -11, -6], [5, 5, 4, 3, 2, 1]),
        ):
            with self.subTest(cap=cap):
                actual, counts = audit_targets(replay, *table_networks(), horizon=cap)
                self.assertEqual(actual.tolist(), expected)
                self.assertEqual(counts.tolist(), horizon)

    def test_cutoff_excludes_first_nongreedy_action_reward(self):
        replay = chain_replay()
        replay.action_id[2] = 1
        replay.reward[2] = -10000
        values, horizons = audit_targets(replay, *table_networks(), indices=[0, 1, 2])
        self.assertEqual(values.tolist(), [17, 18, -10015])
        # A sampled action is learned even when it is itself nongreedy.
        self.assertEqual(horizons.tolist(), [2, 1, 4])

    def test_masked_and_unsupported_high_q_never_select_or_cut_path(self):
        replay = chain_replay()
        replay.action_mask[:, 2] = False
        replay.next_action_mask[:, 2] = False
        online, target = table_networks()
        with torch.no_grad():
            online.values[:, 2] = 1.0e6
            online.values[:, 3] = 1.0e7
        values, horizons = audit_targets(
            replay, online, target, indices=[0], support=[True, True, True, False],
        )
        self.assertEqual(values.tolist(), [35])
        self.assertEqual(horizons.tolist(), [5])
        replay.action_id[1] = 3
        values, horizons = audit_targets(
            replay, online, target, indices=[0], support=[True, True, True, False],
        )
        self.assertEqual(values.tolist(), [9])
        self.assertEqual(horizons.tolist(), [1])

    def test_current_online_greedy_is_recomputed_each_call(self):
        replay = chain_replay()
        online, target = table_networks()
        first, counts = audit_targets(replay, online, target, indices=[0])
        self.assertEqual(counts.item(), 5)
        with torch.no_grad():
            online.values[1, 1] = 5
        second, counts = audit_targets(replay, online, target, indices=[0])
        self.assertEqual(first.item(), 35)
        self.assertEqual(second.item(), 999)
        self.assertEqual(counts.item(), 1)

    def test_tied_greedy_actions_use_existing_argmax_tie_break(self):
        replay = chain_replay()
        replay.action_id[1] = 1
        online, target = table_networks()
        with torch.no_grad():
            online.values[1, 1] = online.values[1, 0]
        values, horizons = audit_targets(replay, online, target, indices=[0])
        self.assertEqual(values.item(), 9)
        self.assertEqual(horizons.item(), 1)

    def test_true_terminal_never_evaluates_successor_networks(self):
        online, target = table_networks()
        values, horizons = audit_targets(chain_replay(), online, target, indices=[5])
        self.assertEqual(values.item(), -6)
        self.assertEqual(horizons.item(), 1)
        self.assertEqual(online.seen, [])
        self.assertEqual(target.seen, [])

    def test_nonterminal_truncation_bootstraps_stored_next_state(self):
        replay = take_rows(chain_replay(), [0, 1])
        values, horizons = audit_targets(replay, *table_networks(), scale=.5)
        self.assertEqual(values.tolist(), [18.5, 19])
        self.assertEqual(horizons.tolist(), [2, 1])
        np.testing.assert_array_equal(replay.done, [0, 0])

    def test_gap_and_episode_boundary_bootstrap_without_stitching(self):
        for change in ("gap", "episode"):
            replay = chain_replay()
            if change == "gap":
                replay = take_rows(replay, [0, 2, 3, 4, 5])
            else:
                replay.episode[1:] = 1
            values, horizons = audit_targets(replay, *table_networks(), indices=[0])
            with self.subTest(change=change):
                self.assertEqual(values.item(), 9)
                self.assertEqual(horizons.item(), 1)

    def test_duplicate_sampled_rows_are_independent_and_replay_is_unchanged(self):
        replay = chain_replay()
        before = take_rows(replay, list(range(replay.size)))
        values, horizons = audit_targets(replay, *table_networks(), indices=[4, 0, 4, 1])
        self.assertEqual(values.tolist(), [-11, 35, -11, -20])
        self.assertEqual(horizons.tolist(), [2, 5, 2, 5])
        for field in fields(replay):
            if field.name != "manifest":
                np.testing.assert_array_equal(getattr(replay, field.name), getattr(before, field.name))
        self.assertEqual(replay.manifest, before.manifest)


class MultistepTrainingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_horizon_positive_integer_and_gamma_guards(self):
        for value in (0, -1, True, False, 1.0, 2.5, "5", None):
            with self.subTest(horizon=value), self.assertRaisesRegex(ValueError, "positive integer"):
                ResponseDQNConfig(backup_horizon=value).validate()
        for gamma in (.99, 0, float("nan")):
            with self.subTest(gamma=gamma), self.assertRaisesRegex(ValueError, "gamma=1"):
                train_tiny(chain_replay(), gamma=gamma)
        ResponseDQNConfig().validate()
        ResponseDQNConfig(backup_horizon=5, gamma=1).validate()

    def test_metadata_and_options_fail_before_network_initialization(self):
        replay = chain_replay()
        bad_replays = []
        for field, values in (
            ("reward_semantics", (None, "monte_carlo_return", "advantage")),
            ("done_semantics", (None, "rollout_terminal", "terminal_label")),
        ):
            for value in values:
                manifest = dict(replay.manifest)
                manifest.pop(field) if value is None else manifest.update({field: value})
                bad_replays.append(replace(replay, manifest=manifest))
        for value in (2, 1.5):
            bad_replays.append(replace(replay, option_steps=np.full(replay.size, value)))
        with patch.object(response_dqn, "CandidateQNetwork", side_effect=AssertionError("too late")):
            for bad in bad_replays:
                with self.subTest(manifest=bad.manifest, options=bad.option_steps), self.assertRaises(ValueError):
                    train_tiny(bad)
            with self.assertRaisesRegex(ValueError, "bootstrap=False"):
                train_tiny(replay, bootstrap=True)

    def test_default_and_explicit_one_step_preserve_loss_weights_and_rng(self):
        replay = _synthetic_replay()
        for bootstrap in (False, True):
            models, states = [], []
            with patch.object(response_dqn, "build_sequential_links", side_effect=AssertionError("must skip")), \
                 patch.object(response_dqn, "greedy_consistent_ddqn_targets", side_effect=AssertionError("must skip")):
                for extra in ({}, {"backup_horizon": 1}):
                    config = ResponseDQNConfig(
                        gradient_steps=8, hidden=(8,), batch_size=7,
                        target_update_interval=3, conservative_alpha=.01, **extra,
                    )
                    models.append(train_response_dqn_member(
                        replay, np.eye(3, dtype=np.float32), catalog_fingerprint="catalog-test",
                        config=config, seed=17, bootstrap=bootstrap,
                    ))
                    states.append((torch.get_rng_state(), np.random.random(3), random.getstate()))
            with self.subTest(bootstrap=bootstrap):
                self.assertEqual(models[0].training_losses, models[1].training_losses)
                for name in ("online", "target"):
                    for left, right in zip(getattr(models[0], name).parameters(), getattr(models[1], name).parameters()):
                        self.assertTrue(torch.equal(left, right))
                self.assertTrue(torch.equal(states[0][0], states[1][0]))
                np.testing.assert_array_equal(states[0][1], states[1][1])
                self.assertEqual(states[0][2], states[1][2])

    def test_training_recomputes_member_targets_each_update_with_cost_head(self):
        replay = phase_replay(horizon=6, start=0)
        replay.action_id[:] = 0
        actual_helper = greedy_consistent_ddqn_targets
        calls = []

        def checked(online, target, tensors, candidates, support, links, idx, **kwargs):
            result = actual_helper(online, target, tensors, candidates, support, links, idx, **kwargs)
            self.assertIn("next_remaining_intervals", tensors)
            self.assertFalse(result[0].requires_grad)
            self.assertTrue((result[0] <= 0).all())
            self.assertTrue((result[1] <= 5).all())
            calls.append((online, target))
            return result

        with patch.object(response_dqn, "greedy_consistent_ddqn_targets", side_effect=checked):
            model = train_tiny(replay, value_parameterization="finite_horizon_cost_v1", conservative_alpha=.01)
        self.assertEqual(len(calls), 3)
        self.assertTrue(all(online is model.online and target is model.target for online, target in calls))
        self.assertTrue(np.isfinite(model.training_losses).all())

    def test_checkpoint_roundtrip_and_legacy_default_keep_inference_identical(self):
        replay = chain_replay()
        model = train_tiny(replay)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            model.save(path)
            restored = load_trained_response_dqn(path)
            self.assertEqual(restored.config.backup_horizon, 5)
            payload = model.checkpoint()
            payload["config"].pop("backup_horizon")
            torch.save(payload, path)
            legacy = load_trained_response_dqn(path)
            self.assertEqual(legacy.config.backup_horizon, 1)
        expected = model.q_values(replay.observation, replay.response_features)
        for loaded in (restored, legacy):
            np.testing.assert_array_equal(expected, loaded.q_values(replay.observation, replay.response_features))

    def test_cli_default_and_opt_in_reach_worker_and_manifest(self):
        replay = chain_replay()
        replay.manifest["catalog"] = {}
        for horizon in (1, 5):
            with self.subTest(horizon=horizon), tempfile.TemporaryDirectory() as directory:
                argv = ["train_response_dqn", "--data", "unused.npz", "--output-dir", directory,
                        "--ensemble-size", "1", "--gamma", "1", "--no-group-bootstrap"]
                if horizon != 1:
                    argv.extend(("--backup-horizon", str(horizon)))
                with patch("sys.argv", argv), \
                     patch.object(train_response_dqn, "load_frozen_response_replay", return_value=replay), \
                     patch.object(train_response_dqn.StructuredActionCatalog, "from_manifest") as catalog, \
                     patch.object(train_response_dqn, "_train_member_worker") as worker, \
                     contextlib.redirect_stdout(io.StringIO()):
                    catalog.return_value.fingerprint = "chain-toy"
                    worker.return_value = {"member": 0, "checkpoint": "mock.pt"}
                    train_response_dqn.main()
                payload = worker.call_args.args[0]
                self.assertEqual(payload["config"]["backup_horizon"], horizon)
                self.assertFalse(payload["bootstrap"])
                manifest = json.loads((Path(directory) / "ensemble_manifest.json").read_text(encoding="utf-8"))
                self.assertEqual(manifest["config"]["backup_horizon"], horizon)
                self.assertEqual(manifest["target_semantics"]["backup_horizon"], horizon)

    def test_stop_before_setup_and_during_updates(self):
        replay = chain_replay()
        with tempfile.TemporaryDirectory() as directory:
            stop = Path(directory) / "STOP"
            with patch.object(Path, "exists", return_value=True), \
                 patch.object(response_dqn, "build_sequential_links") as links:
                with self.assertRaises(InterruptedError):
                    train_tiny(replay, stop_files=(stop,))
                links.assert_not_called()
            original_exists = Path.exists
            with patch.object(torch.optim.Adam, "step") as updates:
                def exists(path):
                    return updates.call_count > 0 if path == stop else original_exists(path)

                with patch.object(Path, "exists", exists):
                    with self.assertRaises(InterruptedError):
                        train_tiny(replay, gradient_steps=101, stop_files=(stop,))
                self.assertEqual(updates.call_count, 100)


if __name__ == "__main__":
    unittest.main()
