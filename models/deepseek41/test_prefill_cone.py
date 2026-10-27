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


def fake_chunks(plan, sparse, restored=None):
    """Execute each actual chunk with persistent physical rings and history."""
    rings = {layer: [None] * 136 for layer in range(20, 40)}
    history = {layer: [None] * 134 for layer in [37, 38, 39]}
    if restored is not None:
        rings = {layer: list(cells) for layer, cells in zip(rings, restored[0])}
        history = {layer: list(cells) for layer, cells in zip(history, restored[1])}
    snapshots = []
    for chunk in plan["chunks"]:
        first, end = chunk["start"], chunk["end"]
        n = end - first
        hidden = {p: (first + p + 1) * 1234567 for p in range(n)}
        for row in chunk["layers"]:
            layer = row["layer"]
            if plan["dspark"] and layer in history:
                features = {p: hidden[p] for p in pc.positions(row["features"])} if sparse else hidden
                for p in range(n - 128 if n > 134 else 0, n):
                    history[layer][(first + p) % 134] = (first + p, features[p])
            keys = pc.positions(row["keys"]) if sparse else range(n)
            kv = {p: ((hidden[p] * 33) ^ layer) & MASK for p in keys}
            output = pc.positions(row["outputs"]) if sparse else range(n)
            next_hidden = {}
            for p in output:
                total = hidden[p]
                for position in range(max(0, first + p - 127), first + p + 1):
                    if position < first:
                        writer, value = rings[layer][position % 136]
                        assert writer == position
                        total += value
                    else:
                        total += kv[position - first]
                next_hidden[p] = total & MASK
            for p, value in kv.items():
                rings[layer][(first + p) % 136] = (first + p, value)
            hidden = next_hidden
        snapshots.append((tuple(tuple(rings[l]) for l in rings),
                          tuple(tuple(history[l]) for l in history)))
    return hidden[n - 1], snapshots


class Cone(unittest.TestCase):
    def test_extra_endpoints_keep_all_physical_cells_and_six_older_features(self):
        cuts = [64, 192, 4032, 8064]
        for dspark in [False, True]:
            plan = pc.guaranteed(8257, 8192, dspark, cuts=cuts)
            self.assertEqual([c["end"] for c in plan["chunks"]], cuts + [8192, 8257])
            sparse = fake_chunks(plan, True)
            self.assertEqual(sparse, fake_chunks(plan, False))
            for at in [1, 2]:
                hit = fake_chunks({**plan, "chunks": plan["chunks"][at + 1:]}, True, sparse[1][at])
                self.assertEqual(hit, (sparse[0], sparse[1][at + 1:]))
            if dspark:
                before = sparse[1][1][1]
                after = sparse[1][2][1]
                replaced = {p % 134 for p in range(4032 - 128, 4032)}
                preserved = set(range(134)) - replaced
                self.assertEqual(len(preserved), 6)
                for old, new in zip(before, after):
                    self.assertTrue(all(new[cell] == old[cell] for cell in preserved))
                writers = pc.history_writers([c["end"] for c in plan["chunks"]], [4032])[4032]
                self.assertEqual(tuple(cell[0] if cell is not None else None for cell in after[0]), writers)

    def test_timing_cuts_keep_uncut_outputs_and_differ_only_in_six_unread_cells(self):
        # Sixteen 512-row timing cuts to the grid and two past it, against
        # one 8192-row commit and its tail.
        cuts = list(range(512, 8192, 512)) + [8704, 8960]
        for dspark in [False, True]:
            cut = pc.guaranteed(9000, 8192, dspark, cuts=cuts)
            whole = pc.guaranteed(9000, 8192, dspark)
            self.assertEqual([c["end"] for c in cut["chunks"]], cuts[:15] + [8192, 8704, 8960, 9000])
            sparse = fake_chunks(cut, True)
            self.assertEqual(sparse, fake_chunks(cut, False))
            uncut = fake_chunks(whole, True)
            # The prompt's output is the uncut schedule's.
            self.assertEqual(sparse[0], uncut[0])
            # A restore at any cut continues to the same output and state.
            for at in [0, 7, 14, 15, 16]:
                hit = fake_chunks({**cut, "chunks": cut["chunks"][at + 1:]}, True, sparse[1][at])
                self.assertEqual(hit, (sparse[0], sparse[1][at + 1:]))
            # At the grid boundary every target ring cell holds the same
            # position and value; the raw history differs only in cells
            # 18-23, which hold 7656-7661 after the cuts and nothing after
            # one large commit.
            at_grid, whole_grid = sparse[1][15], uncut[1][0]
            self.assertEqual(at_grid[0], whole_grid[0])
            for layer, (left, right) in enumerate(zip(at_grid[1], whole_grid[1])):
                differing = [cell for cell in range(134) if left[cell] != right[cell]]
                if not dspark:
                    self.assertEqual(differing, [])
                    continue
                self.assertEqual(differing, list(range(18, 24)), layer)
                self.assertEqual([left[c][0] for c in differing], list(range(7656, 7662)))
                self.assertTrue(all(right[c] is None for c in differing))
                # Every cell a restored drafter reads, the last 128
                # positions, is equal.
                self.assertEqual({left[c] for c in range(134) if c not in differing},
                                 {right[c] for c in range(134) if c not in differing})
                self.assertTrue(all(8064 <= left[c][0] < 8192 for c in range(134) if c not in differing))
            self.assertEqual(pc.history_writers([c["end"] for c in cut["chunks"]], [8192])[8192][18:24],
                             tuple(range(7656, 7662)))
            self.assertEqual(pc.history_writers([8192, 9000], [8192])[8192][18:24], (None,) * 6)

    def test_extra_endpoints_must_be_internal_granules(self):
        for cuts in [[0], [63], [8257], [8320]]:
            with self.assertRaises(ValueError):
                pc.guaranteed(8257, 8192, True, cuts=cuts)

    def test_poisoned_read_schedule_matches_full_fake_kernels_and_copied_cells(self):
        for n, limit, captures in [(521, 256, [256, 512]), (2049, 256, [1024, 2048]),
                                   (8192, 2048, [6144, 8192])]:
            for dspark in [False, True]:
                with self.subTest(n=n, dspark=dspark):
                    plan = pc.cone(n, pc.chunks(n, captures, limit), captures, dspark)
                    self.assertEqual(fake_run(plan, True), fake_run(plan, False))

    def test_guaranteed_chunks_bound_the_work_independently_of_prompt_length(self):
        for chunk in [2048, 8192]:
            for dspark in [False, True]:
                plans = [pc.guaranteed(n, chunk, dspark) for n in [8192, 32768, 131072]]
                # Only the final chunk adds final-layer logits work.
                first = plans[0]["upper_output_rows"]
                for plan in plans[1:]:
                    multiple = plan["tokens"] // 8192
                    self.assertEqual(plan["upper_output_rows"], (first - 64) * multiple + 64)
                    self.assertGreater(plan["equal_layer_row_saving"], .20)
        self.assertEqual(pc.guaranteed(8192, 8192, True, 1)["upper_output_rows"], 24302)

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

    def test_all_choices_match_union_of_individual_boundary_cones(self):
        n = 1024
        individual = []
        for boundary in range(64, n, 64):
            captures = [boundary, n]
            individual.append(pc.cone(n, pc.chunks(n, captures, 2048), captures, True))
        envelope = pc.all_choices(n)
        for index, row in enumerate(envelope["layers"]):
            for role in ["inputs", "outputs", "window_kv", "features"]:
                self.assertEqual(row[role], pc.union(*(p["layers"][index][role] for p in individual)))

    def test_all_choices_leave_only_final_layer_output_rows_skippable(self):
        for n in [8192, 32768, 131072]:
            plan = pc.all_choices(n)
            for row in plan["layers"]:
                self.assertEqual(row["inputs"], [(0, n)])
                self.assertEqual(row["window_kv"], [(0, n)])
                expected = [(n - 1, n)] if row["layer"] == 39 else [(0, n)]
                self.assertEqual(row["outputs"], expected)
            self.assertEqual(plan["upper_output_rows"], 19 * n + 1)
            self.assertAlmostEqual(plan["upper_output_skip_fraction"], (n - 1) / (20 * n))
            self.assertAlmostEqual(plan["equal_layer_row_saving"], (n - 1) / (40 * n))
            self.assertLess(plan["equal_layer_row_saving"], 0.20)


if __name__ == "__main__":
    unittest.main()
