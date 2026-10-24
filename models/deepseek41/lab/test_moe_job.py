import copy
import json
from pathlib import Path
import unittest

import moe_job
import moe_profile


def records():
    values = []
    for label in sorted(moe_job.expected_cases()):
        bank, phase, rows, routing = label.split("-")
        experts, top_k = bank[1:].split("k")
        values.append(dict(case=label, experts=int(experts), top_k=int(top_k),
                           phase=phase, rows=int(rows), routing=routing,
                           old_ms=1.0, active_ms=1.0, compact_ms=1.1, block_warps_ms=2.0,
                           bitwise=True, dropped_join_detected=phase != "Prefill",
                           speed_passed=True, kernel_ms=dict(gateup=[0.5]*3, down=[0.2]*3)))
    for row in values:
        row.update(streamed_ms=0.8, streamed_old_ms=1.0, streamed_candidate=row["rows"]<=64 and row["phase"]!="Prefill",
                   active_experts=row["top_k"], bandwidth_tb_s=1.6,
                   kernel_order=["old", "selected", "compact", "block_warps", "streamed", "previous"])
        compact = row["experts"] == 128 and row["phase"] == "Decode" and row["rows"] == 1
        row.update(selected_compact=compact, selected_streamed=row["streamed_candidate"] and not compact,streamed_bitwise=row["streamed_candidate"],
                   previous_ms=1.0, selected_paired_ms=1.0, compared_modes=row["kernel_order"],
                   compared_outputs=12, live_replays=[row["rows"],1,0,row["rows"]])
        row["kernel_ms"] = dict(gateup=[0.5]*6, down=[0.2]*6)
        row["weight_floor_ms"] = dict(gateup=row["top_k"]*1152*5120*(0.5+1/32)/1.6e9,
                                     down=row["top_k"]*5120*576*(0.5+1/32)/1.6e9)
        row["floor_ratio"] = {name:[t/row["weight_floor_ms"][name] for t in times] for name,times in row["kernel_ms"].items()}
    return values


class TimingGate(unittest.TestCase):
    def test_complete_table_and_noise_limit(self):
        rows = records()
        self.assertEqual(len(rows), 50)
        rows[0]["active_ms"] = 1.03
        self.assertEqual(moe_job.check_records(rows), [])
        self.assertIn(rows[0]["case"], moe_job.speed_table(rows))
        rows[0]["active_ms"] = 1.031
        self.assertTrue(moe_job.check_records(rows))

    def test_parity_alone_cannot_pass_the_reported_regression(self):
        rows = records()
        rows[0].update(old_ms=0.504368, active_ms=0.847328)
        self.assertTrue(moe_job.check_records(rows))
        rows[0]["active_ms"] = float("nan")
        self.assertTrue(moe_job.check_records(rows))

    def test_previous_selection_is_a_separate_speed_gate(self):
        rows = records()
        rows[0].update(old_ms=2.0,active_ms=1.1,previous_ms=1.0,selected_paired_ms=1.031)
        self.assertTrue(moe_job.check_records(rows))
        rows[0]["selected_paired_ms"] = 1.03
        self.assertEqual(moe_job.check_records(rows),[])

    def test_claimed_streamed_parity_requires_selected_entries_and_all_replays(self):
        for field,value in [("selected_streamed",True),("selected_compact",False),("streamed_bitwise",False),
                            ("compared_outputs",11),("live_replays",[1,0,1]),("compared_modes",["old","selected"])]:
            rows = records()
            rows[0][field] = value
            self.assertTrue(moe_job.check_records(rows),field)

    def test_reported_previous_regression_still_fails_without_a_serial_regression(self):
        rows = records()
        rows[0].update(old_ms=0.488192,active_ms=0.268,previous_ms=0.260064,
                       selected_paired_ms=0.268224,speed_passed=False)
        errors = moe_job.check_records(rows)
        self.assertTrue(any("previous selection" in error for error in errors))
        self.assertFalse(any("serial baseline" in error for error in errors))
        rows[0].update(active_ms=0.257776,selected_paired_ms=0.257776,speed_passed=True)
        self.assertEqual(moe_job.check_records(rows),[])

    def test_missing_boundary_or_join_control_fails(self):
        rows = records()
        self.assertTrue(moe_job.check_records(rows[:-1]))
        self.assertTrue(moe_job.check_records(rows + [copy.deepcopy(rows[0])]))
        rows[0]["dropped_join_detected"] = False
        self.assertTrue(moe_job.check_records(rows))

    def test_missing_candidate_or_floor_evidence_fails(self):
        rows = records()
        rows[0]["streamed_candidate"] = False
        self.assertTrue(moe_job.check_records(rows))
        rows = records()
        rows[0]["floor_ratio"]["down"][4] = float("nan")
        self.assertTrue(moe_job.check_records(rows))
        rows = records()
        rows[0]["weight_floor_ms"]["gateup"] *= 2
        self.assertTrue(moe_job.check_records(rows))

    def test_profiler_covers_each_mode_and_chain_stage(self):
        locks = Path(__file__).resolve().parent.parent / "bundle/kernels.lock.d"
        symbols = []
        for family in ["routed_moe", "routed_moe_active", "routed_moe_streamed"]:
            for cubin in json.loads((locks / (family + ".lock")).read_text()):
                symbols.extend(entry["symbol"] for entry in cubin["entries"])
        report = "\n".join(symbols)
        for bank in [0,1]:
            for rows in moe_profile.ROWS:
                moe_profile.check_profile(report,bank,rows)
        for suffix in ["_active_bw4_v1", "_stream_pack_v1", "_stream_small_v1", "_stream_wide_v1", "_stream_merge_v1",
                       "_stream_small_lanes_v1", "_stream_wide_lanes_v1", "_clamp4120000041200000_lanes_v1"]:
            incomplete = "\n".join(s for s in symbols if not s.endswith(suffix))
            with self.assertRaises(AssertionError):
                moe_profile.check_profile(incomplete,0,48)

    def test_records_parse_with_libtest_prefix(self):
        row = records()[0]
        log = "test name ... MOE_RESULT " + json.dumps(row) + "\nok\n"
        self.assertEqual(moe_job.parse_records(log), [row])


if __name__ == "__main__":
    unittest.main()
