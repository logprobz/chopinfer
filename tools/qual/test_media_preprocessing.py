"""The preprocessing gate refuses missing, nonfinite and out-of-bound rows."""

import unittest

import media_preprocessing as gate


class PreprocessingTests(unittest.TestCase):
    def test_reference_spread_sets_the_bound(self):
        self.assertTrue(gate.floating([1.25], [1.0], [1.125])["passed"])
        self.assertFalse(gate.floating([1.5], [1.0], [1.125])["passed"])
        self.assertFalse(gate.floating([1.0000001], [1.0], [1.0])["passed"])
        self.assertTrue(gate.floating([1.0], [1.0], [1.0])["passed"])

    def test_mel_uses_the_reference_frame_peak(self):
        import math
        result = gate.floating([math.log(2), math.log(4)], [math.log(2), math.log(4)],
                               [math.log(1), math.log(4)], bins=2)
        self.assertTrue(result["passed"])
        self.assertAlmostEqual(result["reference_f32_f64_spread"], 0.25)
        self.assertFalse(gate.floating([math.log(16), math.log(4)], [math.log(2), math.log(4)],
                                      [math.log(1), math.log(4)], bins=2)["passed"])

    def test_empty_mismatched_and_nonfinite_data_are_not_evidence(self):
        for args in (([], [], []), ([1], [1, 2], [1]), ([float("nan")], [1], [1]),
                     ([1], [float("inf")], [1])):
            with self.assertRaises(ValueError):
                gate.floating(*args)

    def test_missing_cases_are_refused(self):
        with self.assertRaises(ValueError):
            gate.compare({"cases": [{"name": "image"}]}, None, {"cases": []}, None)

    def test_image_integer_evidence_is_required(self):
        import tempfile
        import struct
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "single").write_bytes(struct.pack("<f", 1))
            (root / "double").write_bytes(struct.pack("<d", 1))
            ref = {"cases": [{"name": "image", "kind": "image", "stages": {"patches": {
                "f32": {"shape": [1], "file": "single"}, "f64": {"shape": [1], "file": "double"}}}}]}
            candidate = {"cases": [{"name": "image", "stages": {"patches": {"shape": [1], "file": "single"}}}]}
            with self.assertRaisesRegex(ValueError, "integer"):
                gate.compare(ref, root, candidate, root)

    def test_requested_exporter_must_match_the_fresh_artifact(self):
        import json
        import tempfile
        from pathlib import Path
        from types import SimpleNamespace
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as tmp:
            fresh, stale = Path(tmp) / "fresh", Path(tmp) / "stale"
            fresh.write_bytes(b"new code")
            stale.write_bytes(b"old code")
            result = SimpleNamespace(stdout=json.dumps({"reason": "compiler-artifact",
                "target": {"name": "media-preprocessing"}, "executable": str(fresh)}))
            with patch.object(gate.subprocess, "run", return_value=result) as build:
                with self.assertRaisesRegex(ValueError, "fresh Cargo build"):
                    gate.build_exporter(stale)
                self.assertEqual(gate.build_exporter(fresh)[0], fresh)
                self.assertIn("--locked", build.call_args.args[0])
                self.assertIn("--offline", build.call_args.args[0])

    def test_only_the_documented_deviations_are_accepted(self):
        def declared(value):
            return {"checks": {"media_preprocessing": {"constants": {"media_preprocessing": {
                "PROCESSOR_OVERRIDES": {"value": value}}}}}}
        for value in ({"resample": 2}, {"min_pixels": 8192}, {"image_mean": [0,0,0]}, {"max_pixels": 10}):
            with self.assertRaisesRegex(ValueError, "documented"):
                gate.preprocessing.declared_overrides(declared(value))
        self.assertEqual(gate.preprocessing.declared_overrides(declared({})), {})
        allowed = {"image_mean": [0.485, 0.456, 0.406], "image_std": [0.229, 0.224, 0.225],
                   "max_pixels": 1048576, "min_pixels": 8192}
        self.assertEqual(gate.preprocessing.declared_overrides(declared(allowed)), allowed)
        with self.assertRaisesRegex(ValueError, "documented"):
            gate.preprocessing.declared_overrides(declared({**allowed, "min_pixels": 3136}))

    def test_parallel_cargo_rebuild_cannot_replace_the_running_exporter(self):
        import contextlib
        import io
        import json
        import struct
        import tempfile
        from pathlib import Path
        from types import SimpleNamespace
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original = root / "built"
            original.write_bytes(b"fresh build")
            frontend = root / "frontend.json"
            frontend.write_text("{}")
            reference = root / "reference/audio"
            reference.mkdir(parents=True)
            (reference / "single").write_bytes(struct.pack("<f", 1))
            (reference / "double").write_bytes(struct.pack("<d", 1))
            record = {"sha256": "reference", "binding": {}, "cases": [{"name": "audio", "kind": "audio",
                "stages": {"resampled": {"f32": {"file": "single", "shape": [1]},
                                          "f64": {"file": "double", "shape": [1]}}}}]}
            (reference / "reference.json").write_text(json.dumps(record))
            def run(command, **kwargs):
                if command[0] == "cargo":
                    return SimpleNamespace(stdout=json.dumps({"reason": "compiler-artifact", "target": {
                        "name": "media-preprocessing"}, "executable": str(original)}))
                self.assertEqual(Path(command[0]).read_bytes(), b"fresh build")
                original.write_bytes(b"another Cargo build")
                output = Path(command[3])
                output.mkdir()
                (output / "single").write_bytes(struct.pack("<f", 1))
                (output / "candidate.json").write_text(json.dumps({"cases": [{"name": "audio", "stages": {
                    "resampled": {"file": "single", "shape": [1]}}}]}))
            argv = ["media_preprocessing.py", "--reference", str(reference.parent), "--frontend", str(frontend)]
            with patch.object(gate.sys, "argv", argv), patch.object(gate.subprocess, "run", side_effect=run), \
                    patch.object(gate.preprocessing, "verify", return_value=record), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(gate.main(), 0)


class ReceiptTests(unittest.TestCase):
    """Nested preprocessing rows are admitted only for the pinned full case set."""

    def setUp(self):
        import kinds
        self.kind = kinds.of("media_preprocessing")
        self.check = {"module": "media_preprocessing", "rules": [
            {"path": ["passed"], "op": "eq", "value": True},
            {"path": ["reference_bound"], "op": "eq", "value": True}]}
        row = {"case": "image", "stage": "patches", "shape_matches": True, "grid_matches": True,
               "metric": "absolute", "reference_f32_f64_spread": 0.125, "maximum_error": 0.25,
               "bound": 0.25, "integer_equal": True, "passed": True}
        self.value = {"passed": True, "reference_bound": True, "exporter_build_verified": True,
                      "exporter_sha256": "a" * 64, "frontend_sha256": "b" * 64,
                      "references": [{"reference_sha256": "c" * 64, "binding": {"processor": {}},
                                      "passed": True, "rows": [row]}]}
        self.reference = {"frontend_sha256": "b" * 64, "references": [{"sha256": "c" * 64,
            "binding": {"processor": {}}, "cases": [{"name": "image", "kind": "image",
                                                        "stages": {"patches": {"integer": {}}}}]}]}

    def test_nested_complete_receipt_passes_and_generic_guard_stays_strict(self):
        import kinds
        metrics = self.kind.score(self.value, self.reference, self.check)
        self.assertEqual(self.kind.judge(metrics, self.check)[0], "pass")
        self.assertEqual(metrics, self.kind.recorded(self.value, self.check))
        with self.assertRaisesRegex(ValueError, "no item coverage"):
            kinds.RECEIPT.score(self.value, check=self.check)

    def test_parent_gate_admits_worker_stdout_with_the_selected_policy(self):
        import json
        import tempfile
        import time
        from pathlib import Path
        from unittest.mock import patch
        import declaration
        import release_gate
        check = {**self.check, "id": "media_preprocessing", "args": [], "stdout_json": True, "timeout_seconds": 10}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "checks").mkdir()
            config = declaration.load(declaration.MECHANICS)
            config["checks"] = [check]
            path = root / "policy.json"
            path.write_text(json.dumps(config))

            def capture(command, environment, log, timeout):
                Path(log).write_text(json.dumps(self.value))
                return 0

            with patch.object(self.kind, "capture", side_effect=capture), \
                    patch.object(self.kind, "reference", return_value=self.reference):
                row = release_gate.run_check(check, [], "full", {}, root, path, time.monotonic() + 10, False, None)
            self.assertTrue(row["passed"], row)
            self.assertEqual(row["exit_code"], 0)

    def test_empty_duplicate_missing_extra_or_unbound_reports_refuse(self):
        import copy
        edits = [lambda v: v.update(references=[]),
                 lambda v: v["references"].append(copy.deepcopy(v["references"][0])),
                 lambda v: v["references"][0].update(rows=[]),
                 lambda v: v["references"][0]["rows"].append(copy.deepcopy(v["references"][0]["rows"][0])),
                 lambda v: v["references"][0]["rows"][0].update(case="unexpected"),
                 lambda v: v["references"][0].update(reference_sha256="d" * 64),
                 lambda v: v["references"][0].update(binding={}),
                 lambda v: v.update(frontend_sha256="d" * 64),
                 lambda v: v.update(reference_bound=False),
                 lambda v: v.update(exporter_build_verified=False),
                 lambda v: v.update(exporter_sha256="unrecorded")]
        for edit in edits:
            value = copy.deepcopy(self.value); edit(value)
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.kind.score(value, self.reference, self.check)

    def test_changed_bounds_nonfinite_or_missing_fields_refuse(self):
        import copy
        for field, value in (("bound", 0.5), ("maximum_error", float("nan")),
                             ("reference_f32_f64_spread", -1), ("metric", "other"),
                             ("shape_matches", 1), ("integer_equal", None)):
            receipt = copy.deepcopy(self.value)
            receipt["references"][0]["rows"][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.kind.score(receipt, self.reference, self.check)

    def test_failed_numeric_shape_and_integer_rows_cannot_claim_pass(self):
        import copy
        for field, value in (("maximum_error", 0.251), ("integer_equal", False), ("shape_matches", False)):
            receipt = copy.deepcopy(self.value)
            receipt["references"][0]["rows"][0][field] = value
            metrics = self.kind.score(receipt, self.reference, self.check)
            self.assertFalse(metrics["passed"])
            self.assertEqual(self.kind.judge(metrics, self.check)[0], "fail")

    def test_all_declared_references_and_stages_are_required(self):
        import copy
        reference = copy.deepcopy(self.reference)
        reference["references"][0]["cases"][0]["stages"]["another"] = {}
        with self.assertRaises(ValueError):
            self.kind.score(self.value, reference, self.check)
        reference = copy.deepcopy(self.reference)
        reference["references"].append({**reference["references"][0], "sha256": "d" * 64})
        with self.assertRaises(ValueError):
            self.kind.score(self.value, reference, self.check)


if __name__ == "__main__":
    unittest.main()
