"""Encoder parity helpers on synthetic npy files."""

from array import array
from pathlib import Path
import struct
import sys
import tempfile
import unittest

from encoder_parity import bf16_bits, read_npy

CODES = {"<f4": "f", "<i8": "q", "<i4": "i"}


def write_npy(path, descr, shape, values, fortran=False):
    header = repr({"descr": descr, "fortran_order": fortran, "shape": tuple(shape)})
    header += " " * (63 - (10 + len(header)) % 64) + "\n"
    data = array(CODES[descr], values)
    if sys.byteorder != "little":
        data.byteswap()
    path.write_bytes(b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header))
                     + header.encode("latin1") + data.tobytes())


class NpyTests(unittest.TestCase):
    def test_reads_supported_dtypes(self):
        with tempfile.TemporaryDirectory() as tmp:
            for descr, values in (("<f4", [1.5, -2.0]), ("<i8", [2**40, -1]), ("<i4", [7, -7])):
                path = Path(tmp) / "a.npy"
                write_npy(path, descr, (2, 1), values)
                shape, data = read_npy(path)
                self.assertEqual(shape, (2, 1))
                self.assertEqual(list(data), values)

    def test_rejects_fortran_order_and_other_dtypes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "a.npy"
            write_npy(path, "<f4", (1, 2), [1.0, 2.0], fortran=True)
            with self.assertRaises(ValueError):
                read_npy(path)
            path.write_bytes(path.read_bytes().replace(b"<f4", b"<f8"))
            with self.assertRaises(ValueError):
                read_npy(path)

    def test_bf16_rounds_to_nearest_even(self):
        def bits(word):
            return bf16_bits(array("f", struct.pack("<I", word)))[0]

        self.assertEqual(bits(0x3F800000), 0x3F80)
        self.assertEqual(bits(0x3F808000), 0x3F80)  # tie, even stays
        self.assertEqual(bits(0x3F818000), 0x3F82)  # tie, odd rounds up
        self.assertEqual(bits(0x3F808001), 0x3F81)
        self.assertEqual(bits(0xBF80FFFF), 0xBF81)
        self.assertEqual(bits(0x7F800000), 0x7F80)
        self.assertEqual(bits(0x7F7FFFFF), 0x7F80)  # rounds up to infinity
        self.assertEqual(bits(0x7F800001), 0x7FC0)  # NaN stays NaN


if __name__ == "__main__":
    unittest.main()


class LabReceiptTests(unittest.TestCase):
    """`lab` judges only the report its own tapdiff run writes."""

    def test_a_stale_matching_receipt_does_not_pass_an_empty_capture(self):
        import argparse
        import hashlib
        import json

        import encoder_parity
        from common import ROOT

        # Any model with an F32-policy vision encoder declaration.
        found = sorted(path for path in ROOT.glob("models/*/contract/encoders/*.json")
                       if json.loads(path.read_text()).get("precision") == "f32-pairs.v1"
                       and json.loads(path.read_text())["input"]["recipes"][0]["tag"].startswith("image."))
        self.assertTrue(found)
        model, declaration = found[0].parts[-4], str(found[0].relative_to(ROOT))
        with tempfile.TemporaryDirectory() as scratch:
            scratch = Path(scratch)
            exact, policy = scratch / "exact-run", scratch / "policy-run"
            for reference, labels in ((exact, ["exact"]), (policy, ["policy", "reference-bf16"])):
                for label in labels:
                    (reference / "taps" / label / "step0000").mkdir(parents=True)
            (policy / "run.json").write_text(json.dumps({
                "declaration": {"path": declaration,
                                "sha256": hashlib.sha256((ROOT / declaration).read_bytes()).hexdigest()},
                "modes": {"policy": {"taps": ["enc.vision.out"]}}}))
            (policy / "policy.json").write_text(json.dumps({"mode": {"precision": "f32-pairs.v1"}}))
            workload = scratch / "workload.json"
            workload.write_text(json.dumps({"sequences": [{"name": "probe"}]}))
            engine = scratch / "engine"
            engine.mkdir()
            # A receipt of an earlier, matching run at the same path.
            receipt = scratch / "receipt.json"
            receipt.write_text(json.dumps({"status": "match", "first": None,
                                           "taps": [{"tap": "enc.vision.out", "rows": 24}]}))
            args = argparse.Namespace(model=model, engine=str(engine),
                                      reference=[str(exact), str(policy)], workload=str(workload),
                                      json=str(receipt))
            self.assertEqual(encoder_parity.lab(args), 1)
