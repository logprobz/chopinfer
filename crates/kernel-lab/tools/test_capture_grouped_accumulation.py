"""CPU-only route and topology tests; no checkpoint or framework required."""
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("capture", Path(__file__).with_name("capture_grouped_accumulation.py"))
capture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(capture)


class GroupedCaptureTests(unittest.TestCase):
    def test_intermediate_and_expert_shards(self):
        self.assertEqual(capture.owned(1, 2, 2048, 256, False), ((1024, 2048), (0, 256)))
        self.assertEqual(capture.owned(2, 4, 2048, 256, False), ((1024, 1536), (0, 256)))
        self.assertEqual(capture.owned(1, 4, 640, 512, True), ((0, 640), (128, 256)))

    def test_actual_routes_only(self):
        routes = [[0, 1], [0, 2], [0, 3], [4, 3]]
        self.assertEqual(capture.choose_experts(routes, 1, 4, 2), [3, 1])
        self.assertEqual(capture.choose_experts(routes, 5, 8, 4), [])

    def test_topology_rejection(self):
        with self.assertRaises(ValueError):
            capture.owned(4, 4, 2048, 256, False)
        with self.assertRaises(ValueError):
            capture.owned(0, 8, 2048, 256, False)


if __name__ == "__main__":
    unittest.main()
