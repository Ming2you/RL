from __future__ import annotations

import unittest

import numpy as np

from rl_leader.diagnose_pstack_parity import _price_difference, _vector_difference



class PStackParityDiagnosticTest(unittest.TestCase):
    def test_vector_difference_reports_changed_entries(self) -> None:
        result = _vector_difference(np.asarray([1.0, 2.0]), np.asarray([1.0, 5.0]))

        self.assertEqual(result["count"], 2)
        self.assertEqual(result["changed_count"], 1)
        self.assertEqual(result["mean_abs"], 1.5)
        self.assertEqual(result["max_abs"], 3.0)

    def test_price_difference_counts_native_nonzero_on_explicit_key_set(self) -> None:
        result = _price_difference(
            {"merge": 2.0, "nonmerge": -3.0},
            {"merge": 2.0},
            {"nonmerge"},
        )

        self.assertEqual(result["key_count"], 1)
        self.assertEqual(result["native_nonzero_count"], 1)
        self.assertEqual(result["changed_count"], 1)
        self.assertEqual(result["max_abs"], 3.0)


if __name__ == "__main__":
    unittest.main()
