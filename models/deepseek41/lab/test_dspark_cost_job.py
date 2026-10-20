import json
import unittest
import dspark_cost_job as job


class CostJobTests(unittest.TestCase):
    def test_memory_reports_allocations_without_calling_them_a_bound(self):
        text = "\n".join([
            "graph capture memory rank 0: 100 bytes free before, 60 after, 40 held",
            "graph capture memory rank 0: 60 bytes free before, 55 after, 5 held",
            json.dumps(dict(schema="chopinfer.graph-cache.v1", capacity=1536,
                            peak=12, released=0)),
        ])
        result = job.capture_memory(text)
        self.assertEqual(result["capture_allocated_bytes_by_rank"], {"0": 45})
        self.assertEqual(result["memory_samples_by_rank"], {"0": 2})
        self.assertEqual(result["peak_handles"], 12)
        with self.assertRaises(AssertionError):
            job.capture_memory(text.replace("5 held", "6 held"))

    def test_target_misses_belong_to_their_own_completed_step(self):
        records = [
            dict(schema="chopinfer.graph-cache.v1", event="capture", family="target"),
            dict(schema="chopinfer.step.v1", prefill_items=0, decode_rows=0, verify_items=7),
            dict(schema="chopinfer.step.v1", prefill_items=0, decode_rows=0, verify_items=8),
            dict(schema="chopinfer.graph-cache.v1", event="capture", family="catch-up"),
            dict(schema="chopinfer.step.v1", prefill_items=0, decode_rows=0, verify_items=8),
            dict(schema="chopinfer.graph-cache.v1", event="capture", family="target"),
            dict(schema="chopinfer.step.v1", prefill_items=0, decode_rows=1, verify_items=7),
        ]
        result = job.capture_hits("\n".join(map(json.dumps, records)), 8)
        self.assertEqual(result["full_verify_steps"], 3)
        self.assertEqual(result["target_capture_steps"], 1)
        self.assertAlmostEqual(result["hit_rate"], 2 / 3)
        self.assertIsNone(job.capture_hits("", 8)["hit_rate"])


if __name__ == "__main__":
    unittest.main()
