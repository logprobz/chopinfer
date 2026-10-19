"""Hold serving changes when host pressure or an unchanged control explains timing."""
import copy
import unittest

import verify_width_repeat as repeat


def passes():
    return [{"arm": arm, "decode": {str(c): {
        "host_load": {"idle": True, "timing": "evidence"},
        "steady": {"aggregate_tokens_per_second": 100.0 if arm == "base" or c == 32 else 110.0},
        "acceptance": {"verify_items": 0 if c == 32 else 10},
    } for c in repeat.CONCURRENCIES}} for arm in repeat.ORDER]


class RepeatedWidthTests(unittest.TestCase):
    def test_stable_exact_path_with_repeated_gain_can_advance(self):
        self.assertEqual(repeat.CONCURRENCIES, (1, 8, 16, 32))
        self.assertTrue(repeat.decision(passes())["route_d14"])

    def test_busy_host_never_qualifies_a_timing_win(self):
        rows = passes()
        rows[1]["decode"]["1"]["host_load"] = {"idle": False, "timing": "informational"}
        with self.assertRaisesRegex(ValueError, "informational"):
            repeat.decision(rows)

    def test_control_drift_regression_or_one_off_gain_holds_routing(self):
        for kind in ("control", "regression", "one-off", "verify-control"):
            rows = copy.deepcopy(passes())
            if kind == "control":
                rows[1]["decode"]["32"]["steady"]["aggregate_tokens_per_second"] = 122.0
            elif kind == "regression":
                rows[1]["decode"]["1"]["steady"]["aggregate_tokens_per_second"] = 86.0
            elif kind == "one-off":
                for c in (1, 8, 16):
                    rows[2]["decode"][str(c)]["steady"]["aggregate_tokens_per_second"] = 100.0
            else:
                rows[1]["decode"]["32"]["acceptance"]["verify_items"] = 1
            self.assertFalse(repeat.decision(rows)["route_d14"], kind)

    def test_missing_pass_or_concurrency_cannot_qualify(self):
        with self.assertRaises(ValueError):
            repeat.decision(passes()[:3])
        rows = passes()
        del rows[1]["decode"]["16"]
        with self.assertRaises(KeyError):
            repeat.decision(rows)


if __name__ == "__main__":
    unittest.main()
