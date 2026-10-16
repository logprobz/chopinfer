"""Evidence comparisons must refuse missing ranks and changed output bits."""
import tempfile
from pathlib import Path
import unittest

import encoder_declarations
from common import ROOT
from dev.encoder_parallel import compare_arenas, compare_tokens, parse_timings, parse_phases, require_large_geometry, declared_encoders, png_variant, wav_variant


class EncoderParallelTests(unittest.TestCase):
    def test_arena_comparison_requires_every_rank_and_identical_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            left, right = (Path(directory) / name for name in ("serial", "parallel"))
            left.mkdir()
            right.mkdir()
            with self.assertRaises(ValueError):
                compare_arenas(left, right)
            for root in (left, right):
                for record in range(2):
                    for rank in range(4):
                        (root / f"request-{record}-record-0-rank-{rank}.bin").write_bytes(bytes([record, 1]))
            self.assertEqual(len(compare_arenas(left, right)), 8)
            path = right / "request-0-record-0-rank-3.bin"
            path.write_bytes(b"\x00\x02")
            with self.assertRaisesRegex(ValueError, "mismatch"):
                compare_arenas(left, right)
            path.unlink()
            with self.assertRaisesRegex(ValueError, "capture sets"):
                compare_arenas(left, right)

    def test_two_rank_captures_compare_at_tp2_only(self):
        with tempfile.TemporaryDirectory() as directory:
            left, right = (Path(directory) / name for name in ("serial", "parallel"))
            for root in (left, right):
                root.mkdir()
                for rank in range(2):
                    (root / f"request-0-record-0-rank-{rank}.bin").write_bytes(b"\x01\x02")
            self.assertEqual(len(compare_arenas(left, right, ranks=2)), 2)
            with self.assertRaisesRegex(ValueError, "all 4 ranks"):
                compare_arenas(left, right)

    def test_all_ranks_must_agree_even_when_serial_matches_parallel(self):
        with tempfile.TemporaryDirectory() as directory:
            left, right = (Path(directory) / name for name in ("serial", "parallel"))
            for root in (left, right):
                root.mkdir()
                for record in range(2):
                    for rank in range(4):
                        (root / f"request-{record}-record-0-rank-{rank}.bin").write_bytes(bytes([rank, record]))
            with self.assertRaisesRegex(ValueError, "all 4 ranks"):
                compare_arenas(left, right)

    def test_timings_use_explicit_encoder_and_fields_in_any_order(self):
        log = "Example encoder rows=5 elapsed_ms=3.5 encoder=1\n"
        log += "unrelated log\nExample encoder rows=5 encoder=1 elapsed_ms=4.5\n"
        self.assertEqual(parse_timings(log, 1), 8)
        for bad in ("", log.replace("encoder=1", "encoder=0"), log.replace("3.5", "nan")):
            with self.assertRaises(ValueError):
                parse_timings(bad, 1)

    def test_phase_timings_are_parsed_and_required(self):
        self.assertEqual(parse_phases("MiMo encoder parallel ranks=4 encoder=0 rows=128 drain_ms=3 submit_ms=2 prepare_ms=1"),
                         {"prepare_ms": 1, "submit_ms": 2, "drain_ms": 3})
        for log in ("", "MiMo encoder rank=0 encoder=0 rows=128"):
            with self.assertRaises(ValueError):
                parse_phases(log)

    def test_large_geometry_must_actually_reach_the_encoder(self):
        values = encoder_declarations.for_model("mimo26")
        values.sort(key=lambda value: value["tower"] != "vision")
        shared = ("Example encoder rows=4096 patches=16384 encoder=0 runs=2 elapsed_ms=9 prepare_ms=1 "
                  "submit_ms=2 drain_ms=3 ranks=4 mode=Relocated\n")
        log = shared + "Example encoder encoder=1 rows=407 patches=6500 runs=1\n"
        require_large_geometry(log, values)
        for bad in (log.replace("patches=16384", "patches=8192"),
                    log.replace("runs=2", "runs=1"), log.replace("patches=6500", "patches=6000")):
            with self.assertRaises(ValueError):
                require_large_geometry(bad, values)
        values[1]["sequences"]["input_rows"] = 7000
        with self.assertRaisesRegex(ValueError, "multi-segment"):
            require_large_geometry(log, values)
        self.assertEqual(parse_phases(shared), {"prepare_ms": 1, "submit_ms": 2, "drain_ms": 3})

    def test_declarations_must_cover_the_serving_model(self):
        paths = [ROOT / "models/mimo26/contract/encoders" / f"mimo-{tower}.json"
                 for tower in ("vision", "audio")]
        self.assertEqual([value["tower"] for value in declared_encoders(paths, "mimo26")], ["vision", "audio"])
        for wrong in (paths[:1], paths + paths[:1]):
            with self.assertRaises(ValueError):
                declared_encoders(wrong, "mimo26")

    def test_timing_media_differs_after_decoding(self):
        import io
        import wave
        self.assertNotEqual(png_variant(0), png_variant(1))
        with wave.open(io.BytesIO(wav_variant(0, 1))) as a, wave.open(io.BytesIO(wav_variant(1, 1))) as b:
            self.assertEqual(a.getnframes(), 24000)
            self.assertNotEqual(a.readframes(24000), b.readframes(24000))

    def test_tokens_are_required_and_compared_exactly(self):
        record = {"image-cold": {"tokens": [1, 3], "text": "blue"}}
        compare_tokens(record, record)
        for changed in ({}, {"image-cold": {"tokens": [], "text": "blue"}},
                        {"image-cold": {"tokens": [1, 4], "text": "blue"}}):
            with self.assertRaises(ValueError):
                compare_tokens(record, changed)


if __name__ == "__main__":
    unittest.main()
