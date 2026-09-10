import unittest

import numpy as np

from rl_leader.run_cached_pcent_guided_search import (
    pcent_guided_specs_from_candidates,
)


class CachedPcentGuidedSearchTests(unittest.TestCase):
    def test_candidate_conversion_filters_zero_and_dedupes_residuals(self):
        names = ("urban.A.g_green", "freeway.R_F_W.g_meter")
        first = np.asarray([0.0, 0.5], dtype=np.float32)
        specs = pcent_guided_specs_from_candidates(
            names,
            [
                ("anchor", np.zeros(2, dtype=np.float32)),
                ("toward:ramp:R_F_W:0.5", first),
                ("toward:ramp:R_F_W:duplicate", first.copy()),
                ("away:ramp:R_F_W:0.5", -first),
            ],
        )

        self.assertEqual(len(specs), 2)
        self.assertTrue(all(spec.generator == "pcent_cached" for spec in specs))
        self.assertEqual(specs[0].family, "pcent_ramp")
        self.assertTrue(
            any(spec.residual_array()[1] < 0.0 for spec in specs)
        )

    def test_candidate_conversion_rejects_wrong_dimension(self):
        with self.assertRaisesRegex(ValueError, "wrong dimension"):
            pcent_guided_specs_from_candidates(
                ("urban.A.g_green",),
                [("bad", np.zeros(2, dtype=np.float32))],
            )


if __name__ == "__main__":
    unittest.main()
