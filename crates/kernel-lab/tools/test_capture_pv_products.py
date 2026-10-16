"""CPU-only tests for PV product expansion geometry."""
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("capture", Path(__file__).with_name("capture_pv_products.py"))
capture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(capture)


class ProductShapeTests(unittest.TestCase):
    def test_site_hooks_preserve_module_inputs_and_outputs(self):
        active = []
        pre, post = capture.site_hooks(active, "block")
        inputs, output = (object(),), object()
        self.assertIsNone(pre(None, inputs))
        self.assertEqual(active, ["block"])
        self.assertIsNone(post(None, inputs, output))
        self.assertEqual(active, [])

    def test_encoder_temporary_counts(self):
        self.assertEqual(capture.expansion_shape(256, 2, 2, 64), (256, 1024, 256))
        self.assertEqual(capture.expansion_shape(4, 3, 2, 64), (64, 384, 384))

    def test_selected_key_tails(self):
        self.assertEqual(capture.expansion_shape(2051, 2, 1, 16), (2064, 4128, 32))
        self.assertEqual(capture.expansion_shape(129, 2, 1, 32), (160, 320, 64))

    def test_window_middle_and_tail_are_distinct(self):
        plans = capture.encoder_rows(256, {"window": [64, 64]})
        self.assertEqual(len(plans), 2)
        self.assertEqual(plans[0][1], list(range(128, 144)))
        self.assertEqual(plans[1][1], list(range(240, 256)))
        self.assertEqual(capture.encoder_rows(4, "full"), [("tail", [0, 1, 2, 3])])

    def test_reject_invalid_geometry(self):
        for values in [(0, 2, 1, 16), (128, 4, 1, 16), (128, 2, 3, 16)]:
            with self.assertRaises(ValueError):
                capture.expansion_shape(*values)


if __name__ == "__main__":
    unittest.main()
