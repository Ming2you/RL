"""Guard the controlled comparison and resumable STOP aliases."""
import copy
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from rl_leader.run_sequential_response_ddqn import _actor_stop_requested
from work.run_response_cql_ablation import (
    _screen, validate_training_recipe, validate_checkpoint_value_heads, variant_training_spec,
)
from src.tests.test_response_dqn_semantics import _replay


class ControlledRecipeTest(unittest.TestCase):
    def setUp(self):
        self.spec = {
            "common_training": {"seed": 17, "ensemble_size": 3, "gradient_steps": 6000,
                                "hidden": "128,128", "gamma": 1.0, "reward_scale": 0.01,
                                "group_resampling": False, "require_sequential_td": True},
            "data": "replay.npz", "catalog_fingerprint": "catalog",
        }
        self.manifest = {
            "config": {"seed": 17, "ensemble_size": 3, "gradient_steps": 6000,
                       "hidden": [128, 128], "gamma": 1.0, "reward_scale": 0.01,
                       "no_bootstrap": True, "require_sequential_td": True,
                       "conservative_alpha": 0.1},
            "source_dataset": "replay.npz", "catalog_fingerprint": "catalog",
            "checkpoints": ["member0.pt", "member1.pt", "member2.pt"],
        }

    def test_matching_recipe(self):
        validate_training_recipe(self.manifest, self.spec, 0.1)

    def test_value_head_override_preserves_other_recipe_and_original(self):
        original = copy.deepcopy(self.spec)
        spec = variant_training_spec(self.spec, {'value_parameterization': 'finite_horizon_cost_v1'})
        self.assertEqual(self.spec, original)
        self.assertEqual(spec['common_training'], {**original['common_training'],
                         'value_parameterization': 'finite_horizon_cost_v1'})
        with self.assertRaisesRegex(ValueError, 'value_parameterization'):
            validate_training_recipe(self.manifest, spec, .1)
        manifest = copy.deepcopy(self.manifest)
        manifest['config']['value_parameterization'] = 'finite_horizon_cost_v1'
        validate_training_recipe(manifest, spec, .1)
        with self.assertRaises(ValueError):
            variant_training_spec(self.spec, {'value_parameterization': 'unknown'})

    def test_legacy_free_head_default_and_checkpoint_mismatch(self):
        free = variant_training_spec(self.spec, {'value_parameterization': 'free_q'})
        validate_training_recipe(self.manifest, free, .1)
        legacy = SimpleNamespace(config=SimpleNamespace())
        cost = SimpleNamespace(config=SimpleNamespace(value_parameterization='finite_horizon_cost_v1'))
        validate_checkpoint_value_heads([legacy], free)
        with self.assertRaisesRegex(ValueError, 'value parameterization'):
            validate_checkpoint_value_heads([cost], free)
        variant = variant_training_spec(self.spec, {'value_parameterization': 'finite_horizon_cost_v1'})
        with self.assertRaises(ValueError):
            validate_checkpoint_value_heads([legacy], variant)
        validate_checkpoint_value_heads([cost], variant)

    def test_mask_dependent_screening_skips_legacy_replay_for_new_model(self):
        model = SimpleNamespace(response_equivalence_mode="post_commit_continuation_v1",
                                action_support_counts=[2, 2])
        report = _screen([model], {"legacy": _replay()})["legacy"]
        self.assertEqual(report["skipped_reason"], "incompatible_response_equivalence_masks")
        self.assertNotIn("selected_action_counts", report)

    def test_rejects_confounded_hyperparameters(self):
        for key, value in {"seed": 18, "gradient_steps": 6001, "hidden": [64, 64],
                           "gamma": 0.99, "reward_scale": 1.0, "conservative_alpha": 1.0,
                           "no_bootstrap": False}.items():
            with self.subTest(key=key):
                bad = copy.deepcopy(self.manifest)
                bad["config"][key] = value
                with self.assertRaises(ValueError):
                    validate_training_recipe(bad, self.spec, 0.1)

    def test_rejects_wrong_data_catalog_or_missing_member(self):
        for key, value in {"source_dataset": "other.npz", "catalog_fingerprint": "other",
                           "checkpoints": ["member0.pt"]}.items():
            with self.subTest(key=key):
                bad = {**self.manifest, key: value}
                with self.assertRaises(ValueError):
                    validate_training_recipe(bad, self.spec, 0.1)

    def test_primary_and_previous_experiment_stops(self):
        with tempfile.TemporaryDirectory() as directory:
            primary = Path(directory) / "STOP"
            previous = Path(directory) / "previous_STOP"
            payload = {"stop_file": str(primary), "additional_stop_files": [str(previous)]}
            self.assertFalse(_actor_stop_requested(payload))
            previous.touch()
            self.assertTrue(_actor_stop_requested(payload))
            previous.unlink()
            primary.touch()
            self.assertTrue(_actor_stop_requested(payload))
            self.assertTrue(_actor_stop_requested({"stop_file": str(primary)}))


if __name__ == "__main__":
    unittest.main()
