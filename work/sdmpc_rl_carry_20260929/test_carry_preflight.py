"""Synthetic detailed admission evidence; no traffic runtime or review approval."""
import copy
from pathlib import Path
import tempfile
from unittest.mock import patch
import unittest

from budget_runtime import read, save
from run_budget import file_hash
import build_preflight


MODES = ["v1_pfo_h3", "carry_physical"]
ACTIONS = {"zero": [0., 0.], "np_plus": [.25, 0.], "np_minus": [-.25, 0.],
           "nuf_plus": [0., .25], "nuf_minus": [0., -.25]}


def probe_fixture(root, step, pins, influential=True):
    folder = root / f"probe_step{step}"
    settings = dict(step=step, source_pins=pins, modes=MODES, actions=ACTIONS,
                    v1_controller_sha256="a" * 64, cpu_mask=1,
                    scope="matched_one_interval_diagnostic")
    save(folder / "settings.json", settings)
    summaries = []
    for mode in MODES:
        for label, action in ACTIONS.items():
            changed = influential and mode == "carry_physical" and label == "np_plus"
            point = [.35 if changed else .3, .2]
            source = "pfo_current" if mode == "v1_pfo_h3" else "pfo_initial" if step == 5 else "previous"
            calls = int(source != "previous")
            budget = [-20. + 50. * action[0], 1500. + 1000. * action[1]]
            audit = dict(execution_check=dict(physical_control_valid=True, budget_feasible=True,
                point=point, budget=budget, achieved=[-50., 1000.], ttt=9.),
                B_requested=[budget], B_executed=budget, G_achieved=[-50., 1000.],
                selected_identity=dict(kind="candidate", candidate_index=0, point=point),
                selected_TTT=9., reference_TTT=10., selection_source="lower_solution",
                fallback_reasons=[], h3_guard_would_reject=False, counts={"scalar_rollouts": 1})
            if mode == "carry_physical":
                audit.update(reference_source=source, guard_mode="physical", h3_guard_enabled=False)
            detail = dict(settings=settings, mode=mode, candidate=label, action=action, source=source,
                pfo_calls=calls, pfo_seconds=.1 * calls, prepare_seconds=.2, decision_seconds=.4,
                interval_ttt=1., full_run=False, original_state_seconds=step * 180.,
                reached_state={"time_sec": (step + 1) * 180.}, physical_point=point, audit=audit)
            save(folder / f"{mode}_{label}.json", detail)
            summaries.append(dict(mode=mode, candidate=label, action=action,
                changed_physical_control=changed, interval_ttt=1., delta_vs_zero=0.,
                requested=budget, executed=budget, source=source, selection_source="lower_solution",
                fallback_reasons=[], h3_would_reject=audit.get("h3_guard_would_reject"),
                pfo_calls=calls, decision_seconds=.4, counts=audit["counts"]))
    save(folder / "completion.json", dict(status="completed", settings=settings, step=step,
        records=summaries, all_executions_valid=True, carry_influence=influential,
        no_ordinary_pfo=True if step > 5 else None))
    return folder


class PreflightTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.pins = {"snapshot": "unchanged", "implementation": {"carry": "unchanged"}}
        save(self.root / "smoke_resume.json", dict(passed=True, source_pins=self.pins))
        for step in (5, 30, 50):
            probe_fixture(self.root, step, self.pins, influential=step != 5)
        (self.root / "preflight_tests.xml").write_text(
            '<testsuites><testsuite tests="12" errors="0" failures="0" skipped="0"/></testsuites>', encoding="utf-8")
        self.review = self.root / "review.md"
        self.review.write_text("Synthetic fixture, not an independent pilot approval.\n", encoding="utf-8")
        self.review_passes = True
        self.output = self.root / "gate.json"

    def build(self):
        original_read = Path.read_text

        def review_text(path, *args, **kwargs):
            if path == self.review:
                return "PILOT_REVIEW: " + ("PASS" if self.review_passes else "PENDING")
            return original_read(path, *args, **kwargs)

        # Only parser input is mocked; no approval marker is written to disk.
        with patch.object(build_preflight, "pins", return_value=self.pins), \
                patch.object(Path, "read_text", review_text):
            build_preflight.build(self.root, [self.review], self.output)

    def assert_rejected(self):
        with self.assertRaises(ValueError):
            self.build()
        self.assertFalse(self.output.exists())

    def reject_mutations(self, relative, changes):
        path = self.root / relative
        original = read(path)
        for index, change in enumerate(changes):
            with self.subTest(path=relative, mutation=index):
                row = copy.deepcopy(original)
                change(row)
                save(path, row)
                self.assert_rejected()
        save(path, original)

    def test_accepts_two_distinct_influential_states_and_hashes_every_detail(self):
        self.build()
        row = read(self.output)
        self.assertTrue(row["ready_for_bounded_pilot"])
        self.assertEqual(row["carry_influence_states"], 2)
        self.assertFalse(row["performance_acceptance"])
        expected = [self.root / "smoke_resume.json", self.root / "preflight_tests.xml", self.review]
        for step in (5, 30, 50):
            folder = self.root / f"probe_step{step}"
            expected.extend([folder / "completion.json", folder / "settings.json"])
            expected.extend(folder / f"{mode}_{label}.json" for mode in MODES for label in ACTIONS)
        self.assertEqual(row["evidence_sha256"], {str(path.resolve()): file_hash(path) for path in expected})

    def test_refuses_overwrite(self):
        self.build()
        before = self.output.read_bytes()
        with self.assertRaises(FileExistsError):
            self.build()
        self.assertEqual(self.output.read_bytes(), before)

    def test_rejects_copied_step30_completion_at_step50(self):
        save(self.root / "probe_step50/completion.json", read(self.root / "probe_step30/completion.json"))
        self.assert_rejected()

    def test_rejects_missing_or_wrong_step_identities(self):
        self.reject_mutations("probe_step50/completion.json", [
            lambda row: row.pop("step"), lambda row: row.update(step=30),
            lambda row: row["settings"].pop("step"), lambda row: row["settings"].update(step=30)])

    def test_rejects_wrong_mode_or_action_contract(self):
        self.reject_mutations("probe_step30/completion.json", [
            lambda row: row["settings"].update(modes=["carry_physical"]),
            lambda row: row["settings"].update(modes=["carry_physical"] * 2),
            lambda row: row["settings"]["actions"].pop("np_minus"),
            lambda row: row["settings"]["actions"].update(np_plus=[.5, 0.])])

    def test_rejects_duplicate_missing_or_mislabeled_summary_records(self):
        self.reject_mutations("probe_step30/completion.json", [
            lambda row: row["records"].pop(),
            lambda row: row["records"].append(copy.deepcopy(row["records"][0])),
            lambda row: row["records"].__setitem__(1, copy.deepcopy(row["records"][0])),
            lambda row: row["records"][1].update(candidate="nuf_minus"),
            lambda row: row["records"][1].update(mode="unknown"),
            lambda row: row["records"][1].update(action=[0., .25])])

    def test_rejects_missing_extra_or_mislabeled_detailed_records(self):
        path = self.root / "probe_step30/carry_physical_np_plus.json"
        original = read(path)
        path.unlink()
        self.assert_rejected()
        save(path, original)
        self.reject_mutations(str(path.relative_to(self.root)), [
            lambda row: row.update(candidate="zero"), lambda row: row.update(mode="v1_pfo_h3"),
            lambda row: row.update(action=[0., .25]), lambda row: row["settings"].update(step=50)])
        save(path.with_name("carry_physical_extra.json"), original)
        self.assert_rejected()

    def test_rejects_copied_or_wrong_time_state_records(self):
        path = self.root / "probe_step50/carry_physical_np_plus.json"
        save(path, read(self.root / "probe_step30/carry_physical_np_plus.json"))
        self.assert_rejected()
        probe_fixture(self.root, 50, self.pins)
        self.reject_mutations(str(path.relative_to(self.root)), [
            lambda row: row.update(original_state_seconds=5400.),
            lambda row: row.update(reached_state={"time_sec": 5400.}),
            lambda row: row.update(full_run=True)])

    def test_rejects_claimed_influence_without_distinct_executed_control(self):
        probe_fixture(self.root, 30, self.pins, influential=False)
        path = self.root / "probe_step30/completion.json"
        row = read(path)
        row["carry_influence"] = True
        save(path, row)
        self.assert_rejected()

    def test_rejects_only_one_influential_state(self):
        probe_fixture(self.root, 30, self.pins, influential=False)
        self.assert_rejected()

    def test_multiple_changed_actions_in_one_state_do_not_count_as_two_states(self):
        probe_fixture(self.root, 50, self.pins, influential=False)
        path = self.root / "probe_step30/carry_physical_nuf_plus.json"
        detail = read(path)
        detail["physical_point"] = [.4, .2]
        detail["audit"]["execution_check"]["point"] = [.4, .2]
        detail["audit"]["selected_identity"]["point"] = [.4, .2]
        save(path, detail)
        path = path.with_name("completion.json")
        completion = read(path)
        completion["records"][8]["changed_physical_control"] = True
        save(path, completion)
        with self.assertRaisesRegex(ValueError, "Insufficient executable action influence"):
            self.build()
        self.assertFalse(self.output.exists())

    def test_rejects_detail_changed_during_validation(self):
        path = self.root / "probe_step30/carry_physical_np_plus.json"
        original_read = build_preflight.read

        def changing_read(current):
            result = original_read(current)
            if current == path:
                changed = copy.deepcopy(result)
                changed["interval_ttt"] += 1.
                save(current, changed)
            return result

        with patch.object(build_preflight, "read", side_effect=changing_read), \
                self.assertRaisesRegex(ValueError, "changed while reading"):
            self.build()
        self.assertFalse(self.output.exists())

    def test_rejects_invalid_detailed_execution_despite_valid_summary(self):
        for mode in MODES:
            self.reject_mutations(f"probe_step30/{mode}_np_plus.json", [
                lambda row: row["audit"]["execution_check"].update(physical_control_valid=False),
                lambda row: row["audit"]["execution_check"].update(budget_feasible=False)])

    def test_rejects_control_audit_mismatch_and_nonfinite_points(self):
        self.reject_mutations("probe_step30/carry_physical_np_plus.json", [
            lambda row: row.update(physical_point=[.8, .2]),
            lambda row: row.update(physical_point=[float("nan"), .2]),
            lambda row: row.update(physical_point=[]),
            lambda row: row["audit"]["execution_check"].update(budget=[0., 0.]),
            lambda row: row["audit"].update(B_executed=[float("inf"), 1500.])])

    def test_rejects_actual_pfo_calls_source_and_guard_mismatches(self):
        cases = [(5, "carry_physical", {"pfo_calls": 0}),
                 (30, "carry_physical", {"pfo_calls": 1}),
                 (50, "carry_physical", {"source": "pfo_recovery"}),
                 (30, "v1_pfo_h3", {"pfo_calls": 0}),
                 (30, "carry_physical", {"pfo_seconds": .1})]
        for step, mode, changes in cases:
            self.reject_mutations(f"probe_step{step}/{mode}_np_plus.json",
                                  [lambda row, changes=changes: row.update(changes)])
        self.reject_mutations("probe_step30/carry_physical_np_plus.json", [
            lambda row: row["audit"].update(reference_source="pfo_recovery"),
            lambda row: row["audit"].update(guard_mode="h3"),
            lambda row: row["audit"].update(h3_guard_enabled=True)])

    def test_rejects_summary_detail_disagreement(self):
        self.reject_mutations("probe_step30/completion.json", [
            lambda row: row["records"][6].update(changed_physical_control=False)])

    def test_rejects_smoke_failed_or_different_source(self):
        self.reject_mutations("smoke_resume.json", [
            lambda row: row.update(passed=False),
            lambda row: row.update(source_pins={"changed": True})])

    def test_rejects_probe_failure_or_source_drift(self):
        self.reject_mutations("probe_step30/completion.json", [
            lambda row: row.update(status="paused"), lambda row: row.update(all_executions_valid=False),
            lambda row: row.update(no_ordinary_pfo=False), lambda row: row["settings"].update(source_pins={})])

    def test_rejects_bad_test_evidence_or_review(self):
        path = self.root / "preflight_tests.xml"
        for key in ("errors", "failures", "skipped"):
            path.write_text(f'<testsuite tests="12" {key}="1"/>', encoding="utf-8")
            self.assert_rejected()
        path.write_text('<testsuite tests="12"/>', encoding="utf-8")
        self.review_passes = False
        self.assert_rejected()


if __name__ == "__main__":
    unittest.main()
