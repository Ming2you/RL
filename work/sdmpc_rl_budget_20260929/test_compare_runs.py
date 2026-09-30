import copy
import tempfile
import unittest
from pathlib import Path
from budget_runtime import read, save
from compare_runs import compare
from test_run_contracts import fixture


class ComparisonTests(unittest.TestCase):
    def fixture(self, root, mode):
        folder = fixture(root, mode)
        return folder, read(folder / "episode_00_trace.json"), read(folder / "completion.json")

    def test_valid_comparison_and_no_goal_claim(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            folders = {k: self.fixture(root, k)[0] for k in ("native", "center", "rl")}
            result = compare(folders)
            self.assertEqual(result["status"], "reconciled")
            self.assertFalse(result["goal_claim"])
            self.assertEqual(result["rows"][-1]["improvement_vs_native_pct"], 0.)

    def test_reject_partial_and_contract_drift(self):
        for failure in ("partial", "accounting", "contract", "exploration", "physical", "guard"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                folders = {k: self.fixture(root, k)[0] for k in ("native", "center")}
                folder, rows, result = self.fixture(root, "rl")
                folders["rl"] = folder
                if failure == "partial":
                    rows.pop()
                elif failure == "accounting":
                    result["episodes"][0]["ttt"] = 79.
                elif failure == "contract":
                    result["source_pins"] = {"x": "changed"}
                elif failure == "exploration":
                    result["episodes"][0]["exploration"] = True
                elif failure == "physical":
                    rows[0]["execution_check"]["physical_control_valid"] = False
                else:
                    rows[0]["selected_TTT"] = 4.
                save(folder / "completion.json", result)
                save(folder / "episode_00_trace.json", rows)
                with self.assertRaises(ValueError):
                    compare(folders)


if __name__ == "__main__":
    unittest.main()
