"""audio_check.py helpers on synthetic rows and capture directories."""

from array import array
import contextlib
import io
from pathlib import Path
import tempfile
import unittest

from dev import audio_check
from encoder_parity import bf16_bits


class LayerTests(unittest.TestCase):
    def write(self, directory, values, codes):
        Path(directory, "audio-layer0-output-1x4.bin").write_bytes(
            bf16_bits(array("f", values)).tobytes())
        Path(directory, "audio-codes-1x4.bin").write_bytes(array("i", codes).tobytes())

    def run_layers(self, first, second):
        args = type("Args", (), {"first": first, "second": second})
        with contextlib.redirect_stdout(io.StringIO()) as output:
            status = audio_check.layers(args)
        return status, output.getvalue()

    def test_identical_and_different_captures(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            self.write(a, [1, 2, 3, 4], [1, 2, 3, 4])
            self.write(b, [1, 2, 3, 4], [1, 2, 3, 4])
            status, output = self.run_layers(a, b)
            self.assertEqual(status, 0)
            self.assertIn("identical", output)
            self.write(b, [1, 2, 3, 5], [1, 2, 3, 5])
            status, output = self.run_layers(a, b)
            self.assertEqual(status, 1)
            self.assertIn("1/4 values differ", output)
            self.assertIn("1/4 codes differ", output)


if __name__ == "__main__":
    unittest.main()
