"""Calibration reports accepted repeat spread and keeps the reviewed margins."""

import copy
import unittest

import kinds
import paired_calibrate
import paired_teacher
import test_paired_teacher


class CalibrationTests(unittest.TestCase):
    def setUp(self):
        fixture = test_paired_teacher.PairedTests()
        fixture.setUp()
        self.first = fixture.accepted
        self.second = copy.deepcopy(self.first)
        self.second["server"]["health"]["serving"]["startup_nonce"] = "d" * 32
        self.reference, self.policy, self.check = fixture.reference, fixture.policy, fixture.check

    def calibrate(self):
        return paired_calibrate.calibrate(self.first, self.second, self.reference, self.policy)

    def test_deterministic_repeats_report_zero_spread_and_keep_reviewed_margins(self):
        result = self.calibrate()
        self.assertEqual(result["margins"], {key: self.policy[key] for key in
                                             ("mean_margin", "category_mean_margin", "top1_delta", "tail_net_allowance")})
        self.assertEqual(result["observed_spread"], {"mean_margin": 0, "category_mean_margin": 0,
                                                     "top1_delta": 0, "tail_net_allowance": 0})

    def test_refuse_same_live_process(self):
        self.second = copy.deepcopy(self.first)
        with self.assertRaisesRegex(kinds.Insufficient, "one startup"):
            self.calibrate()

    def test_repeat_spread_uses_both_directions_without_narrowing_reviewed_margins(self):
        fixture = test_paired_teacher.PairedTests()
        fixture.setUp()
        self.second = fixture.capture(fixture.identity, 0.011)
        self.second["server"]["health"]["serving"]["startup_nonce"] = "d" * 32
        result = self.calibrate()
        self.assertAlmostEqual(result["observed_spread"]["mean_margin"], 0.001)
        self.assertAlmostEqual(result["observed_spread"]["category_mean_margin"], 0.001)
        self.assertEqual(result["margins"]["mean_margin"], 0.005)
        self.assertEqual(result["margins"]["category_mean_margin"], 0.01)
        self.first, self.second = self.second, self.first
        self.assertEqual(self.calibrate()["margins"], result["margins"])

    def test_refuse_spread_above_proposal(self):
        fixture = test_paired_teacher.PairedTests()
        fixture.setUp()
        self.second = fixture.capture(fixture.identity, 0.02)
        self.second["server"]["health"]["serving"]["startup_nonce"] = "d" * 32
        with self.assertRaisesRegex(kinds.Insufficient, "exceeds the approved proposal"):
            self.calibrate()

    def test_profile_policy_uses_live_identity(self):
        check = copy.deepcopy(self.check)
        check["constants"]["paired_teacher"] = {
            "PROFILE_POLICIES": {"value": {"tp2-ar": self.policy}, "source": "test"}}
        server = copy.deepcopy(self.first["server"])
        server["health"]["serving"]["profile"]["name"] = "tp2-ar"
        server["health"]["serving"]["profile"]["speculation"] = {"tokens": 0, "drafter": None}
        self.assertEqual(paired_teacher.policy(check, server=server), self.policy)
        server["health"]["serving"]["profile"]["tensor_parallel_size"] = 4
        with self.assertRaisesRegex(kinds.Insufficient, "no declared paired margins"):
            paired_teacher.policy(check, server=server)

    def test_effective_ar_override_does_not_select_the_default_drafter_policy(self):
        self.assertEqual(paired_teacher.profile_key({"name": "tp4-dflash", "tensor_parallel_size": 4,
                         "speculation": {"tokens": 0, "mode": "off", "drafter": None}}), "tp4-ar")
        self.assertEqual(paired_teacher.profile_key({"name": "tp2-mtp", "tensor_parallel_size": 2,
                         "speculation": {"tokens": 3, "mode": "adaptive", "drafter": "mtp"}}), "tp2-mtp")


if __name__ == "__main__":
    unittest.main()
