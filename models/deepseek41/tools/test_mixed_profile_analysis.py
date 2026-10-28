import unittest
import json
from pathlib import Path
import struct
import sqlite3
import tempfile

from mixed_profile_analysis import analyze, bounded_profile_steps, clock_epoch, cohort_readiness, exclusive, expert_census, family, moe_payload, phase_widths


class MixedProfileTest(unittest.TestCase):
    def test_long_context_scores_and_target_moe_are_not_hidden_in_glue(self):
        self.assertEqual(family("chop_position_indexer_deepseek41_r2_all_scores_v3_live_v1_mma_fp4_guard_fast_v1"), "indexer_score")
        self.assertEqual(family("chop_position_indexer_deepseek41_r1_all_select_v3_live_v1"), "indexer_topk")
        self.assertEqual(family("chop_deepseek41_moe_absorb_lanes_e384_k6_gate_small_scale_tiles_w1_v1"), "moe_gate_up")
        self.assertEqual(family("chop_moe_chain_before_gateup_e128k3_n1152_k5120_stream_small_v1"), "drafter")
        self.assertEqual(family("chop_deepseek41_mxfp8_fold_m8192_n5120_k15360_g1_f32_v2_packed_v1"), "drafter_catchup")
        self.assertEqual(family("chop_deepseek41_verify_f32_n384_k5120_v1_spans_m64_compute_v1"), "moe_router")
        self.assertEqual(family("chop_deepseek41_verify_f32_n32320_k5120_v1_spans_m64_compute_v1"), "logits")

    def test_bounded_capture_requires_full_controls_and_exact_mixed_width(self):
        for profile, key in (("tp4-ar", "decode_rows"), ("tp4-dspark", "verify_items")):
            steps = [dict(batch=i, prefill_tokens=512 if i == 3 else 0, **{key: 16})
                     for i in range(7)]
            self.assertEqual([s["batch"] for s in bounded_profile_steps(steps, 16, profile)],
                             [1, 2, 3, 4, 5])
            for broken in (steps[:-1], steps[2:], steps + [steps[3]],
                           [dict(s, **{key: 15}) if s["batch"] == 3 else s for s in steps],
                           [dict(s, prefill_tokens=256) if s["batch"] == 3 else s for s in steps]):
                with self.assertRaises(ValueError):
                    bounded_profile_steps(broken, 16, profile)

    def test_short_cohorts_cannot_pass_readiness_after_one_decoder_drains(self):
        progress = [{"emitted": 1, "finished": False} for _ in range(8)]
        progress[0] = {"emitted": 32, "finished": True}
        with self.assertRaisesRegex(ValueError, "drained"):
            cohort_readiness(progress, 8, 32)
        progress[0] = {"emitted": 31, "finished": False}
        with self.assertRaisesRegex(ValueError, "admission margin"):
            cohort_readiness(progress, 8, 32)
        self.assertEqual(cohort_readiness(progress, 8, 256)["minimum_remaining_tokens"], 225)
        with self.assertRaisesRegex(ValueError, "every decoder"):
            cohort_readiness(progress[:-1], 8, 256)

    def test_overlapping_kernels_do_not_charge_gaps_or_count_overlap_twice(self):
        rows = [{"start": 10, "end": 60, "name": "chop_moe_native_gateup_x"},
                {"start": 20, "end": 40, "name": "chop_comm_x"},
                {"start": 50, "end": 80, "name": "chop_moe_native_down_x"},
                {"start": 90, "end": 110, "name": "chop_sampling_x"}]
        result = exclusive(rows, 0, 100)
        self.assertEqual(result["exclusive_family_ns"],
                         {"moe_gate_up": 50, "collective": 0, "moe_down": 20, "sampling": 10})
        self.assertEqual(result["kernel_union_ns"], 80)
        self.assertEqual(result["no_kernel_ns"], 20)
        self.assertEqual(sum(result["raw_symbol_ns"].values()), 110)
        self.assertEqual(sum(result["exclusive_symbol_ns"].values()), 80)

    def test_mixed_widths_are_separate_launch_domains(self):
        self.assertEqual(phase_widths({"prefill_tokens": 512, "decode_rows": 8, "verify_rows": 32}),
                         {"prefill_tokens": 512, "decode_rows": 8, "verify_rows": 32})

    def test_missing_census_does_not_become_measured_bandwidth(self):
        self.assertIsNone(moe_payload(512)["weight_bytes"])
        full = moe_payload(512, 384)
        self.assertEqual(full["weight_bytes"], 1_804_861_440)
        with self.assertRaises(ValueError):
            moe_payload(1, 7)

    def test_clock_bracket_uses_monotonic_anchor_and_refuses_drift(self):
        anchors = [dict(unix_before_ns=1000, unix_after_ns=1002, monotonic_ns=100)] * 4
        self.assertEqual(clock_epoch({"systemClockNs": 10}, anchors), (911, 2))
        with self.assertRaises(ValueError):
            clock_epoch({"systemClockNs": 10}, anchors[:3])
        anchors[0] = dict(unix_before_ns=1000, unix_after_ns=20_000_000, monotonic_ns=100)
        with self.assertRaises(ValueError):
            clock_epoch({"systemClockNs": 10}, anchors)

    def test_sparse_sources_and_dense_projections_remain_distinct(self):
        self.assertEqual(family("chop_latent_sources_deepseek41_h16_d512_mixed_x"), "attention_mixed_sources")
        self.assertEqual(family("chop_position_blocks_deepseek41_r1_scores_v3_valid_v1_mma_fp4_guard_fast_v1"), "indexer_score")
        self.assertEqual(family("chop_position_blocks_deepseek41_r1_select_v3_valid_merge_v1"), "indexer_topk")
        self.assertEqual(family("chop_mxfp8_prefill_n8192_k1280_v1"), "dense_projection")
        self.assertEqual(family("chop_deepseek41_dense_f32_m8192_n384_k5120_v1"), "moe_router")
        self.assertEqual(family("new_unknown_kernel"), "unclassified")

    def test_router_taps_bound_the_union_and_require_every_layer(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "completion.json").write_text(json.dumps({"completion": "full"}))
            entries = []
            for layer in range(40):
                name = f"L{layer:02}.router.ids"
                entries.append(dict(name=name, rank=0, dtype="u32", cols=6, rows=512, first_row=0))
                ids = [row % 64 * 6 + k for row in range(512) for k in range(6)]
                (root / (name + ".r0.bin")).write_bytes(struct.pack("<3072I", *ids))
            (root / "index.jsonl").write_text("\n".join(map(json.dumps, entries)))
            result = expert_census(root, 32)
            self.assertEqual(result["layers"][39]["prefill_rows"], 64)
            self.assertEqual(result["layers"][39]["mixed_union_bounds"], [384, 384])
            self.assertEqual(sum(result["layers"][39]["route_counts"]), 384)
            (root / "index.jsonl").write_text("\n".join(map(json.dumps, entries[:-1])))
            with self.assertRaises(ValueError):
                expert_census(root, 32)

    def test_window_lookup_keeps_a_long_kernel_overlapping_shorter_ones(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "capture.sqlite"
            with sqlite3.connect(path) as db:
                db.executescript(
                    "CREATE TABLE TARGET_INFO_SESSION_START_TIME(systemClockNs INTEGER);"
                    "INSERT INTO TARGET_INFO_SESSION_START_TIME VALUES(0);"
                    "CREATE TABLE StringIds(id INTEGER,value TEXT);"
                    "INSERT INTO StringIds VALUES(1,'chop_moe_native_gateup_x');"
                    "CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INTEGER,end INTEGER,deviceId INTEGER,shortName INTEGER);"
                )
                for rank in range(4):
                    db.executemany("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(?,?,?,1)",
                                   [(0, 10000, rank), (1000, 2000, rank), (3000, 4000, rank)])
            anchors = [dict(unix_before_ns=100000, unix_after_ns=100000, monotonic_ns=0)] * 4
            result = analyze(path, [dict(completed_unix_ns=108000, execution_us=3)], anchors)
            self.assertEqual(len(result["steps"]), 1)
            for rank in result["steps"][0]["ranks"].values():
                self.assertEqual(rank["kernel_union_ns"], 3000)
                self.assertEqual(rank["no_kernel_ns"], 0)


if __name__ == "__main__":
    unittest.main()
