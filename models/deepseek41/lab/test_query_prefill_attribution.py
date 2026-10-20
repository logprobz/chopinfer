"""Protect exclusive accounting and fail closed on empty or decode captures."""
import sqlite3
from pathlib import Path
import tempfile
import unittest
import query_prefill_attribution as attribution


class Attribution(unittest.TestCase):
    def test_empty_capture_is_flagged(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/"empty.sqlite"
            with sqlite3.connect(path):
                pass
            with self.assertRaisesRegex(ValueError, "no kernel activity"):
                attribution.analyze(path, 0)

    def test_overlap_and_owned_families(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/"rows.csv"
            path.write_text(
                "symbol,deviceId,start,end,gridX,gridY,blockX,exclusive_ms\n"
                "chop_deepseek41_q_norm_f32_v1,0,0,100000,8192,1,256,0.1\n"
                "chop_deepseek41_dense_ceil_f32_w1280_prefill_w8_v1,0,50000,150000,8192,1,256,0.05\n"
                "chop_deepseek41_moe_ceil_w3456_v1,0,150000,200000,8192,1,256,0.05\n")
            result = attribution.analyze(path, 0)
            self.assertEqual(len(result["symbols"]), 2)
            self.assertAlmostEqual(sum(r["exclusive_ms"] for r in result["symbols"]), .15)
            path.write_text(path.read_text().replace("8192", "32"))
            with self.assertRaisesRegex(ValueError, "no full 8192-row"):
                attribution.analyze(path, 0)
        self.assertIsNone(attribution.family(
            "chop_deepseek41_mxfp8_fold_m8192_n576_k5120_g1_f32_v2_admit_v1"))


if __name__ == "__main__":
    unittest.main()
