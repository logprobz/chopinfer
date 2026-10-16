"""CPU tests for capture geometry and fail-closed publication."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("capture", Path(__file__).with_name("capture_non_dense.py"))
capture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(capture)


class CaptureTests(unittest.TestCase):
    def test_absolute_segment_boundaries(self):
        self.assertEqual(capture.segment(4096), (3072, 1024, 1024))
        self.assertEqual(capture.segment(1_048_576), (786432, 262144, 262144))
        self.assertEqual(capture.segment(127), (0, 127, 128))
        self.assertEqual(capture.segment(257), (256, 1, 32))
        for keys in [1, 128, 1023, 1024, 1025, 32768, 32769, 131072]:
            start, live, padded = capture.segment(keys)
            self.assertEqual(start + live, keys)
            self.assertEqual(padded % 32, 0)
            self.assertLess(padded - live, 32)
            self.assertLessEqual(padded, 262144)

    def test_invalid_lengths_fail(self):
        for keys in [0, -1, 1_048_577]:
            with self.assertRaises(ValueError):
                capture.segment(keys)

    def test_no_empty_manifest_published(self):
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root) / "capture"
            writer = capture.Capture(directory, "0" * 64, 64)
            with self.assertRaises(RuntimeError):
                writer.finish()
            self.assertFalse((directory / "manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
