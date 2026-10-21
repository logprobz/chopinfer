"""Startup evidence rejects incomplete, failed and empty GPU uploads."""

import json
import unittest

from startup_timing import summarize


def log(**overrides):
    upload = {"status": "complete", "seconds": 10, "planned_bytes": 128,
              "uploaded_bytes": 128, **overrides}
    return ("startup weight_upload: " + json.dumps(upload) + "\n"
            "startup bundle create: 12.000s\n"
            "startup readiness: 15.000s (elapsed, includes upload)\n")


class StartupTimingTests(unittest.TestCase):
    def test_nested_phases_remain_separate(self):
        value = summarize(log() + 'startup direct_load: {"rank_seconds":40}\n')
        self.assertEqual(value["readiness_seconds"], 15)
        self.assertEqual(value["bundle_create_seconds"], 12)
        self.assertEqual(value["weight_upload"][0]["seconds"], 10)
        self.assertEqual(value["direct_load"][0]["rank_seconds"], 40)

    def test_multiple_images_are_preserved(self):
        value = summarize(log() + 'startup weight_upload: {"status":"complete",'
                          '"seconds":1,"planned_bytes":4,"uploaded_bytes":4}\n')
        self.assertEqual(len(value["weight_upload"]), 2)

    def test_package_reuse_is_reported_separately_from_nested_stages(self):
        package = {"measurements": {"total_seconds": 2, "stages": {
            "package_reuse": {"seconds": .5, "invocations": 1}}}}
        line = 'startup direct_package: ' + json.dumps(package) + '\n'
        result = summarize(log() + line)
        self.assertEqual(result["package_preparation_seconds"], 2)
        self.assertEqual(result["readiness_seconds"], 15)
        with self.assertRaises(ValueError):
            summarize(log() + line + line)

    def test_empty_sparse_failed_and_mismatched_uploads_fail(self):
        for changes in ({"uploaded_bytes": 0}, {"status": "sparse-probe"},
                        {"status": "failed"}, {"planned_bytes": 256},
                        {"seconds": float("nan")}, {"seconds": -1}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                summarize(log(**changes))

    def test_missing_phases_and_combined_process_logs_fail(self):
        for value in ("", 'startup readiness: 1.0s\n',
                      log().split("startup readiness:")[0], log() + log()):
            with self.subTest(value=value), self.assertRaises(ValueError):
                summarize(value)

    def test_failed_pipeline_is_not_hidden_by_upload_success(self):
        with self.assertRaises(ValueError):
            summarize(log() + 'startup weight_pipeline: {"status":"failed"}\n')

    def test_pipeline_bytes_and_device_timestamps_are_checked(self):
        rank = {"bytes": 128, "chunks": 2, "device_copy_seconds": .1}
        line = lambda row: 'startup weight_pipeline: ' + json.dumps(
            {"status": "complete", "ranks": [row]}) + '\n'
        summarize(log() + line(rank))
        for changed in ({**rank, "bytes": 127}, {**rank, "chunks": 0},
                        {**rank, "device_copy_seconds": float("inf")}):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                summarize(log() + line(changed))
        with self.assertRaises(ValueError):
            summarize(log() + line(rank) + line(rank))

    def test_optional_read_diagnostics_reject_invalid_counts(self):
        io = {"direct_calls": 1, "syscall_nanos": 100, "request_sizes": [1] + [0] * 7}
        line = lambda row: 'startup direct_load: ' + json.dumps(
            {"cumulative": {"io_timing": row}}) + '\n'
        summarize(log() + line(io))
        summarize(log() + line(None))
        for changed in ({**io, "direct_calls": -1}, {**io, "direct_calls": True},
                        {**io, "syscall_nanos": float("nan")},
                        {**io, "request_sizes": []}):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                summarize(log() + line(changed))


if __name__ == "__main__":
    unittest.main()
