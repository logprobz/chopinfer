"""Dependency and poisoned-read checks, with integer CPU fake kernels."""

import unittest

from models.deepseek41 import prefill_cone as pc


MASK = (1 << 64) - 1


def fake_run(plan, sparse):
    n = plan["tokens"]
    # All lower layers and the final compressor's input have already run.
    hidden = {p: (p + 1) * 1234567 for p in range(n)}
    snapshots = {}
    feature_rows = set(pc.positions(plan["feature_roots"]))
    for row in plan["layers"]:
        layer = row["layer"]
        taps = set(pc.positions(row["features"])) if sparse else feature_rows
        if layer in [37, 38, 39] and plan["dspark"]:
            for p in taps:
                snapshots[("feature", layer, p)] = hidden[p]
        keys = pc.positions(row["window_kv"]) if sparse else range(n)
        # Missing hidden rows raise KeyError. No zero-fill can hide a read.
        kv = {p: ((hidden[p] * 33) ^ layer) & MASK for p in keys}
        for end in plan["captures"]:
            for p in range(max(0, end - 136), end):
                snapshots[("kv", layer, p)] = kv[p]
        output = pc.positions(row["outputs"]) if sparse else range(n)
        hidden = {p: (hidden[p] + sum(kv[k] for k in range(max(0, p - 127), p + 1))) & MASK
                  for p in output}
    return hidden[n - 1], snapshots


class Cone(unittest.TestCase):
    def test_poisoned_read_schedule_matches_full_fake_kernels_and_copied_cells(self):
        for n, limit, captures in [(521, 256, [256, 512]), (2049, 256, [1024, 2048]),
                                   (8192, 2048, [6144, 8192])]:
            for dspark in [False, True]:
                with self.subTest(n=n, dspark=dspark):
                    plan = pc.cone(n, pc.chunks(n, captures, limit), captures, dspark)
                    self.assertEqual(fake_run(plan, True), fake_run(plan, False))

    def test_stale_cell_dependency_is_outside_the_old_suffix_union(self):
        plan = pc.cold(32768, 2048, True)
        writers = plan["snapshot_writers"][30720]
        self.assertEqual(sorted(p for p in writers if p < 30592), list(range(28576, 28582)))
        self.assertIn(26290, pc.positions(plan["layers"][0]["outputs"]))
        self.assertLess(26290, min(plan["captures"]) - 2560)

    def test_lone_prefill_needs_disjoint_intervals(self):
        plan = pc.cold(32768, 8192, True)
        self.assertEqual(plan["chunk_ends"], [8192, 16384, 24576, 30720, 32768])
        self.assertEqual(plan["layers"][0]["outputs"], [(22270, 24562), (28298, 32768)])

    def test_future_boundary_reaches_back_across_chunks(self):
        plan = pc.cold(32768, 2048, True)
        self.assertIn(26290, pc.positions(plan["layers"][0]["outputs"]))
        self.assertNotIn(26624, plan["captures"])
        self.assertEqual(plan["layers"][-1]["outputs"], [(32767, 32768)])

    def test_every_prompt_logit_disables_output_skipping(self):
        plan = pc.cone(521, [256, 512, 521], [512], True, logits=list(range(521)))
        self.assertEqual(plan["upper_output_skip_fraction"], 0)

    def test_history_copies_initial_bytes_and_short_chunk_rows(self):
        writers = pc.history_writers([256, 260], [256, 260])
        self.assertEqual(writers[256].count(None), 6)
        for p in range(256, 260):
            self.assertEqual(writers[260][p % 134], p)
        with self.assertRaises(ValueError):
            pc.history_writers([256, 512], [400])


if __name__ == "__main__":
    unittest.main()
