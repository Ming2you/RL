from __future__ import annotations

import inspect
import unittest

import numpy as np

from rl_leader.diagnose_candidate_ablation import (
    _dedupe_for_selector,
    _dedupe_h3_representatives,
)
from rl_leader.oracle_candidate_ablation import (
    ABLATION_GENERATORS,
    AblationConfig,
    candidate_manifest_sha256,
    generate_known_positive_canary,
    generate_owner_block_pcent_candidates,
    generate_owner_block_structured_candidates,
    generate_jacobian_secant_candidates,
    generate_orthogonal_random_candidates,
    generate_pcent_sign_candidates,
    generate_structured_candidates,
    linear_price_group_indices,
    residual_sha256,
    select_owner_block_h12_candidates,
    select_h12_candidates_without_teacher,
    synthesize_jacobian_inverse_candidates,
    validate_candidate_specs,
)


def _action_names() -> tuple[str, ...]:
    names = ["budget.N_P", "budget.N_UF"]
    for owner in "ABCDF":
        names.extend(
            f"urban.{owner}.{term}"
            for term in ("g_green", "g_offset", "l11", "l21", "l22")
        )
    for owner in ("R_D_W", "R_F_W", "R_D_E", "R_F_E"):
        names.extend(
            f"freeway.{owner}.{term}"
            for term in ("g_meter", "g_vsl", "l11", "l21", "l22")
        )
    for index in range(12):
        names.extend((f"vsl.key{index}.g_vsl", f"vsl.key{index}.sqrt_h"))
    names.extend(f"certificate.ramp{index}.release" for index in range(4))
    assert len(names) == 75
    return tuple(names)


class OracleCandidateAblationTest(unittest.TestCase):
    def setUp(self):
        self.names = _action_names()
        self.cfg = AblationConfig()

    def test_linear_price_subspace_is_exactly_canonical_30_dimensions(self):
        groups = linear_price_group_indices(self.names)
        self.assertEqual(
            {name: len(indices) for name, indices in groups.items()},
            {"green": 5, "offset": 5, "meter": 4, "vsl": 16},
        )

    def test_structured_candidates_have_equal_l2_and_fixed_nonprice_channels(self):
        specs = generate_structured_candidates(self.names, self.cfg)
        self.assertEqual(len(specs), 8)
        for spec in specs:
            residual = spec.residual_array()
            self.assertAlmostEqual(float(np.linalg.norm(residual)), 0.5, places=6)
            self.assertTrue(np.array_equal(residual[:2], [0.0, 0.0]))
            self.assertTrue(np.array_equal(residual[-4:], np.zeros(4)))
            self.assertFalse(spec.uses_pcent_target)

    def test_known_positive_canary_reproduces_exact_owner_c_cross_block(self):
        spec = generate_known_positive_canary(self.names)
        residual = spec.residual_array()
        nonzero = {
            self.names[index]: float(residual[index])
            for index in np.flatnonzero(residual)
        }
        self.assertEqual(
            nonzero,
            {"urban.C.g_green": -0.5, "urban.C.g_offset": 0.5},
        )
        self.assertAlmostEqual(float(np.linalg.norm(residual)), 2.0 ** -0.5)
        self.assertFalse(spec.uses_pcent_target)
        self.assertEqual(spec.phase, "regression_canary")

    def test_owner_block_v2_covers_all_sparse_signed_corners(self):
        specs = generate_owner_block_structured_candidates(self.names)
        self.assertEqual(len(specs), 144)
        self.assertTrue(all(not spec.uses_pcent_target for spec in specs))
        self.assertTrue(all(
            np.count_nonzero(spec.residual_array()) in (1, 2) for spec in specs
        ))
        self.assertEqual(
            {round(float(np.linalg.norm(spec.residual_array())), 6) for spec in specs},
            {0.25, 0.5, round(2.0 ** -1.5, 6), round(2.0 ** -0.5, 6)},
        )
        canary = generate_known_positive_canary(self.names)
        self.assertIn(
            residual_sha256(canary.residual),
            {residual_sha256(spec.residual) for spec in specs},
        )

    def test_owner_block_pcent_candidates_keep_privileged_provenance_separate(self):
        green = np.zeros(len(self.names), dtype=np.float32)
        offset = np.zeros(len(self.names), dtype=np.float32)
        green[self.names.index("urban.C.g_green")] = -0.2
        offset[self.names.index("urban.C.g_offset")] = 0.3
        specs = generate_owner_block_pcent_candidates(
            self.names,
            {"green": green, "offset": offset},
            include_freeway=False,
        )
        self.assertEqual(len(specs), 20)
        c_specs = [spec for spec in specs if spec.metadata["owner"] == "C"]
        self.assertTrue(all(spec.uses_pcent_target for spec in c_specs))
        toward = next(
            spec for spec in c_specs
            if ":toward:" in spec.label and spec.metadata["magnitude"] == 0.5
        )
        self.assertEqual(
            {
                self.names[index]: float(toward.residual_array()[index])
                for index in np.flatnonzero(toward.residual_array())
            },
            {"urban.C.g_green": -0.5, "urban.C.g_offset": 0.5},
        )
        structured = generate_owner_block_structured_candidates(
            self.names, include_freeway=False
        )
        self.assertEqual(
            len({residual_sha256(spec.residual) for spec in structured + specs}),
            80,
        )
        partial = generate_owner_block_pcent_candidates(
            self.names,
            {"green": green},
            include_freeway=False,
        )
        partial_c = [spec for spec in partial if spec.metadata["owner"] == "C"]
        self.assertTrue(all(spec.uses_pcent_target for spec in partial_c))
        self.assertTrue(all(spec.teacher_signal_missing for spec in partial_c))

    def test_residual_hash_canonicalizes_signed_zero(self):
        self.assertEqual(residual_sha256([0.0]), residual_sha256([-0.0]))

    def test_owner_block_h12_selector_keeps_h3_inventory_diversity_and_canary(self):
        rows = []
        for index in range(5):
            rows.append({
                "candidate_id": f"candidate-{index}",
                "candidate_label": (
                    "urban:C:green-negative_offset-positive"
                    if index == 4 else f"candidate-{index}"
                ),
                "validity_gate_pass": True,
                "h3_ttt": [9.0, 10.0, 11.0, 12.0, 13.0][index],
                "h3_terminal_inventory": [9.0, 8.0, 10.0, 11.0, 12.0][index],
                "response": [[0.1], [0.2], [3.0], [0.4], [0.5]][index],
                "residual_sha256": f"residual-{index}",
            })
        selected = select_owner_block_h12_candidates(
            rows,
            anchor_response=np.asarray([0.0]),
            known_canary_label="urban:C:green-negative_offset-positive",
        )
        self.assertEqual(
            {row["candidate_id"] for row in selected},
            {"candidate-0", "candidate-1", "candidate-2", "candidate-4"},
        )
        rows[0]["validity_gate_pass"] = False
        selected = select_owner_block_h12_candidates(
            rows,
            anchor_response=np.asarray([0.0]),
            known_canary_label="urban:C:green-negative_offset-positive",
        )
        self.assertNotIn("candidate-0", {row["candidate_id"] for row in selected})

    def test_pcent_sign_fills_missing_teacher_bundles_without_claiming_teacher_use(self):
        groups = linear_price_group_indices(self.names)
        green = np.zeros(len(self.names), dtype=np.float32)
        green[groups["green"]] = -1.0
        specs = generate_pcent_sign_candidates(
            self.names, {"green": green}, self.cfg
        )
        self.assertEqual(len(specs), 8)
        green_specs = [spec for spec in specs if spec.metadata["bundle"] == "green"]
        offset_specs = [spec for spec in specs if spec.metadata["bundle"] == "offset"]
        self.assertTrue(all(spec.uses_pcent_target for spec in green_specs))
        self.assertTrue(all(not spec.teacher_signal_missing for spec in green_specs))
        self.assertTrue(all(not spec.uses_pcent_target for spec in offset_specs))
        self.assertTrue(all(spec.teacher_signal_missing for spec in offset_specs))

    def test_random_basis_is_seeded_orthogonal_and_teacher_independent(self):
        first = generate_orthogonal_random_candidates(
            self.names, self.cfg, pre_runtime_sha256="a" * 64
        )
        repeat = generate_orthogonal_random_candidates(
            self.names, self.cfg, pre_runtime_sha256="a" * 64
        )
        changed = generate_orthogonal_random_candidates(
            self.names, self.cfg, pre_runtime_sha256="b" * 64
        )
        self.assertEqual(candidate_manifest_sha256(first), candidate_manifest_sha256(repeat))
        self.assertNotEqual(candidate_manifest_sha256(first), candidate_manifest_sha256(changed))
        positive = np.stack([first[index].residual_array() for index in range(0, 8, 2)])
        np.testing.assert_allclose(positive @ positive.T, 0.25 * np.eye(4), atol=1.0e-6)
        self.assertTrue(all(not spec.uses_pcent_target for spec in first))

    def test_jacobian_records_rank_condition_and_four_inverse_candidates(self):
        secants = generate_jacobian_secant_candidates(self.names, self.cfg)
        probes = {}
        for index, spec in enumerate(sorted(secants, key=lambda item: item.label)):
            response = np.zeros(4, dtype=float)
            response[index] = 1.0
            probes[spec.candidate_id] = {"response": response.tolist()}
        inverse, diagnostics = synthesize_jacobian_inverse_candidates(
            secants,
            probes,
            anchor_response=np.zeros(4),
            target_response=np.asarray([1.0, -0.5, 0.25, 0.1]),
            response_scales=np.ones(4),
            cfg=self.cfg,
        )
        self.assertEqual(len(inverse), 4)
        self.assertEqual(diagnostics["rank"], 4)
        self.assertAlmostEqual(diagnostics["condition_number"], 1.0)
        counts = validate_candidate_specs(
            secants + inverse, self.names, self.cfg, require_all_generators=False
        )
        self.assertEqual(counts["pcent_jacobian"], 8)
        self.assertTrue(all(spec.uses_pcent_target for spec in inverse))

    def test_all_four_generators_receive_exactly_eight_logical_slots(self):
        structured = generate_structured_candidates(self.names, self.cfg)
        pcent = generate_pcent_sign_candidates(self.names, {}, self.cfg)
        random = generate_orthogonal_random_candidates(
            self.names, self.cfg, pre_runtime_sha256="a" * 64
        )
        secants = generate_jacobian_secant_candidates(self.names, self.cfg)
        probes = {
            spec.candidate_id: {"response": [float(i == index) for i in range(4)]}
            for index, spec in enumerate(sorted(secants, key=lambda item: item.label))
        }
        inverse, _ = synthesize_jacobian_inverse_candidates(
            secants,
            probes,
            anchor_response=np.zeros(4),
            target_response=np.ones(4),
            response_scales=np.ones(4),
            cfg=self.cfg,
        )
        specs = pcent + structured + secants + inverse + random
        self.assertEqual(
            validate_candidate_specs(specs, self.names, self.cfg),
            {generator: 8 for generator in ABLATION_GENERATORS},
        )

    def test_h12_selector_has_no_teacher_argument_and_uses_common_h1_rule(self):
        self.assertNotIn(
            "target",
            inspect.signature(select_h12_candidates_without_teacher).parameters,
        )
        records = []
        for generator in ABLATION_GENERATORS:
            for index, ttt in enumerate((10.0, 9.0)):
                residual = np.zeros(75, dtype=np.float32)
                residual[2 + index] = 0.5
                records.append({
                    "generator": generator,
                    "candidate_id": f"{generator}:{index}",
                    "validity_gate_pass": True,
                    "response": [float(index + 1)],
                    "post_follower_sha256": f"memory-{index}",
                    "step_ttt": ttt,
                    "terminal_inventory": 20.0 + index,
                    "residual_sha256": residual_sha256(residual),
                })
        selected = select_h12_candidates_without_teacher(
            records,
            anchor_response=np.asarray([0.0]),
            anchor_follower_memory_sha256="anchor-memory",
            cfg=self.cfg,
        )
        self.assertEqual(len(selected), 4)
        self.assertTrue(all(row["step_ttt"] == 9.0 for row in selected))

    def test_selector_dedupe_keeps_distinct_post_step_physical_states(self):
        base = {
            "generator": "structured",
            "candidate_id": "structured:a",
            "response_memory_outcome_sha256": "outcome",
            "post_physical_sha256": "physical-a",
            "residual_sha256": "residual-a",
        }
        distinct = {
            **base,
            "candidate_id": "structured:b",
            "post_physical_sha256": "physical-b",
            "residual_sha256": "residual-b",
        }
        alias = {
            **base,
            "candidate_id": "structured:c",
            "residual_sha256": "residual-c",
        }

        deduped = _dedupe_for_selector([distinct, alias, base])

        self.assertEqual(
            {(row["candidate_id"], row["post_physical_sha256"]) for row in deduped},
            {("structured:a", "physical-a"), ("structured:b", "physical-b")},
        )

    def test_h3_outcome_dedupe_prefers_target_independent_not_canary(self):
        base = {
            "response_memory_outcome_sha256": "outcome",
            "post_physical_sha256": "physical",
        }
        guided = {
            **base,
            "candidate_id": "guided",
            "candidate_label": "guided",
            "uses_pcent_target": True,
        }
        structured = {
            **base,
            "candidate_id": "structured",
            "candidate_label": "structured",
            "uses_pcent_target": False,
        }
        canary = {
            **base,
            "candidate_id": "canary",
            "candidate_label": "urban:C:green-negative_offset-positive",
            "uses_pcent_target": False,
        }
        self.assertEqual(
            _dedupe_h3_representatives([guided, structured])[0]["candidate_id"],
            "structured",
        )
        canary["uses_pcent_target"] = True
        self.assertEqual(
            _dedupe_h3_representatives([guided, structured, canary])[0][
                "candidate_id"
            ],
            "structured",
        )

    def test_validator_rejects_budget_or_curvature_perturbations(self):
        specs = generate_structured_candidates(self.names, self.cfg)
        bad = specs[0]
        residual = bad.residual_array()
        residual[0] = 0.1
        from dataclasses import replace
        specs[0] = replace(bad, residual=tuple(map(float, residual)))
        with self.assertRaisesRegex(ValueError, "leaves the 30-D"):
            validate_candidate_specs(
                specs, self.names, self.cfg, require_all_generators=False
            )


if __name__ == "__main__":
    unittest.main()
