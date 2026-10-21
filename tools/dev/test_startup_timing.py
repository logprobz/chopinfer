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


if __name__ == "__main__":
    unittest.main()
