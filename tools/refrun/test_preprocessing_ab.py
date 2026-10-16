"""The paired decision excludes exact ties and uses a two-sided sign test."""

import unittest

from tools.refrun.preprocessing_ab import sign_test


class SignTestTests(unittest.TestCase):
    def test_original_probe_set_with_identical_pixels_has_no_preference(self):
        result = sign_test([0.0] * 349)
        self.assertEqual((result["positive"], result["negative"], result["ties"], result["p"]), (0, 0, 349, 1.0))

    def test_two_sided_probability_and_ties(self):
        self.assertEqual(sign_test([1.0] * 8)["p"], 2 / 2 ** 8)
        self.assertEqual(sign_test([-1.0] * 8 + [0.0] * 20)["p"], 2 / 2 ** 8)
        self.assertEqual(sign_test([1.0, -1.0] + [0.0] * 20)["p"], 1.0)

    def test_missing_or_nonfinite_evidence_is_refused(self):
        for rows in ([], [float("nan")], [float("inf")]):
            with self.assertRaises(ValueError):
                sign_test(rows)


class EffectiveProbeTests(unittest.TestCase):
    def test_identity_only_and_unchanged_minimum_are_refused(self):
        from tools.refrun.preprocessing_ab import require_effective_probes, VARIANTS
        rows = [{"case": "a", "variant": v, "tensor_sha256": v, "grid": [[1,4,4]]} for v in VARIANTS]
        with self.assertRaisesRegex(ValueError, "resized"):
            require_effective_probes(rows, dict.fromkeys(VARIANTS, 0))
        with self.assertRaisesRegex(ValueError, "grid"):
            require_effective_probes(rows, dict.fromkeys(VARIANTS, 1))
        rows[2]["grid"] = [[1,6,6]]
        require_effective_probes(rows, dict.fromkeys(VARIANTS, 1))

    def test_interpolation_effect_must_cover_the_scored_pair(self):
        from tools.refrun.preprocessing_ab import require_effective_probes, VARIANTS
        rows = [{"case": "a", "variant": v, "tensor_sha256": v, "grid": [[1,4,4]]} for v in VARIANTS]
        rows[2]["grid"] = [[1,6,6]]
        rows[3]["tensor_sha256"] = rows[2]["tensor_sha256"]
        with self.assertRaisesRegex(ValueError, "tensor_sha256"):
            require_effective_probes(rows, dict.fromkeys(VARIANTS, 1))
        rows[3]["tensor_sha256"] = "different"
        rows[0]["tensor_sha256"] = rows[1]["tensor_sha256"]
        require_effective_probes(rows, dict.fromkeys(VARIANTS, 1))


if __name__ == "__main__":
    unittest.main()
