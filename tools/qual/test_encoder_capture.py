"""Pinned encoder inputs and capture assembly fail closed before GPU use."""

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import encoder_capture as capture
from tools.model_sdk import diagnostic as wire
from tools.refrun import keys


class EncoderCaptureTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.repo = self.root / "repo"
        self.repo.mkdir()
        self.declared = self.repo / "encoder.json"
        self.declared.write_text('{"precision":"f32-pairs.v1"}')
        self.encoder = {"path": self.declared, "kinds": {"image"}, "tower": "vision", "precision": "f32-pairs.v1"}
        self.data = self.root / "inputs"
        self.data.mkdir()
        (self.data / "image.f32").write_bytes(b"\0" * 4)
        self.workload = {"schema": "chopinfer.refrun.workload.v1", "model": "fixture", "sequences": [
            {"name": "image", "media": [{"kind": "image", "processor": {"sha256": "a" * 64},
             "tensor": {"path": "image.f32", "dtype": "f32", "shape": [1, 1]}}]}]}
        self.write(self.data / "workload.json", self.workload)
        self.write(self.data / "body.json", {"max_tokens": 1, "temperature": 0, "messages": []})
        self.reference = self.data / "reference"
        self.reference.mkdir()
        self.source = b"the recorded encoder producer"
        declared = {"path": "encoder.json", "sha256": hashlib.sha256(self.declared.read_bytes()).hexdigest()}
        run = {"git_head": "a" * 40, "declaration": declared, "unchecked_pins": [], "modes": {}}
        for mode in capture.MODES:
            document = {"name": mode, "precision": "f32-pairs.v1" if mode == "policy" else None,
                        "arithmetic": {"exact": "f64", "policy": "pairs", "reference-bf16": "bf16"}[mode]}
            key = {"schema": keys.ENCODER_SCHEMA, "model": "fixture", "declaration": declared,
                   "checkpoint": {"source_lock_sha256": "b" * 64},
                   "mode": {"name": mode, "policy_sha256": keys.sha256(wire.canonical(document))},
                   "media": [{"kind": "image", "processor": {"sha256": "a" * 64}, "grid": None,
                              "tensor": {"dtype": "f32", "shape": [1, 1],
                                         "sha256": wire.identity(self.data / "image.f32")["sha256"]}}],
                   "code": {"files": {"producer.py": hashlib.sha256(self.source).hexdigest()}}}
            self.write(self.reference / f"{mode}.json", {"schema": "chopinfer.refrun.encoder-rows.v1", "mode": document,
                        "entries": [{"name": "image", "key": key, "key_sha256": keys.key_sha256(key)}]})
            run["modes"][mode] = {"repeat_bitwise_equal": True, "taps": ["enc.vision.out"]}
        self.write(self.reference / "run.json", run)
        self.spec = {"schema": capture.SCHEMA, "model": "fixture", "root": str(self.data),
                     "workload": {"path": "workload.json", "sha256": wire.identity(self.data / "workload.json")["sha256"]},
                     "workload_files_sha256": capture.tree_digest({"image.f32": wire.identity(self.data / "image.f32")}),
                     "requests": [{"name": "image", "path": "body.json", "sha256": wire.identity(self.data / "body.json")["sha256"]}],
                     "references": [{"path": "reference", "files_sha256": capture.tree_digest(capture.inventory(self.reference)),
                                     "source": "a" * 40}]}
        self.spec_path = self.root / "spec.json"
        self.config = {"model": "fixture"}
        for target, name, value in ((capture, "ROOT", self.repo),
                                    (capture, "specification_path", lambda model: self.spec_path),
                                    (capture.encoder_declarations, "for_model", lambda model: [self.encoder]),
                                    (capture, "checkpoint_pin", lambda config: {"source_lock_sha256": "b" * 64})):
            context = patch.object(target, name, value)
            context.start()
            self.addCleanup(context.stop)
        context = patch.object(capture.subprocess, "check_output", return_value=self.source)
        context.start()
        self.addCleanup(context.stop)

    @staticmethod
    def write(path, value):
        path.write_text(json.dumps(value))

    def preflight(self):
        self.write(self.spec_path, self.spec)
        return capture.preflight(self.config)

    def repin_reference(self):
        self.spec["references"][0]["files_sha256"] = capture.tree_digest(capture.inventory(self.reference))

    def test_admitted_inputs_assemble_and_readmit_without_relabeling_source(self):
        prepared = self.preflight()
        run = self.root / "capture"
        (run / "engine").mkdir(parents=True)
        record = {"subject": prepared["subject"], "server": {"source_commit": "c" * 40},
                  "requests": [{"name": "image"}]}
        self.write(run / "tap-capture.json", record)
        manifest = capture.assemble(run, prepared, record, self.config)
        self.assertEqual(manifest["source"], "c" * 40)
        self.assertEqual(capture.admit(run, manifest, self.config)["subject"], prepared["subject"])
        (run / "image.f32").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "captured encoder media changed"):
            capture.admit(run, manifest, self.config)

    def test_changed_workload_payload_and_request_refuse(self):
        for name in ("workload.json", "image.f32", "body.json"):
            with self.subTest(name=name):
                path = self.data / name
                saved = path.read_bytes()
                path.write_bytes(saved + b" ")
                with self.assertRaises(ValueError):
                    self.preflight()
                path.write_bytes(saved)

    def test_missing_or_duplicate_request_coverage_refuses(self):
        request = self.spec["requests"][0]
        for requests in ([], [request, request], [dict(request, name="other")]):
            with self.subTest(requests=requests):
                self.spec["requests"] = requests
                with self.assertRaisesRegex(ValueError, "request coverage"):
                    self.preflight()

    def test_reference_pin_and_semantic_key_admission_are_independent(self):
        path = self.reference / "policy.json"
        original = json.loads(path.read_text())
        changed = copy.deepcopy(original)
        changed["entries"][0]["key"]["checkpoint"]["source_lock_sha256"] = "d" * 64
        changed["entries"][0]["key_sha256"] = keys.key_sha256(changed["entries"][0]["key"])
        self.write(path, changed)
        with self.assertRaisesRegex(ValueError, "reference files changed"):
            self.preflight()
        self.repin_reference()
        with self.assertRaisesRegex(ValueError, "checkpoint lock differs"):
            self.preflight()

    def test_changed_declaration_and_producer_refuse(self):
        self.declared.write_text('{"precision":"bf16-legacy.v1"}')
        with self.assertRaisesRegex(ValueError, "declaration changed"):
            self.preflight()
        self.declared.write_text('{"precision":"f32-pairs.v1"}')
        with patch.object(capture.subprocess, "check_output", return_value=b"another producer"):
            with self.assertRaisesRegex(ValueError, "producer source bytes differ"):
                self.preflight()

    def test_missing_mode_or_declared_encoder_refuses(self):
        path = self.reference / "reference-bf16.json"
        path.unlink()
        with self.assertRaisesRegex(ValueError, "reference files changed"):
            self.preflight()
        self.spec["references"] = []
        with self.assertRaisesRegex(ValueError, "missing declared encoder"):
            self.preflight()

    def test_an_encoder_of_unqualified_media_needs_no_reference(self):
        reason = "served, no in-tree reference"
        self.config = {"model": "fixture", "capabilities": {"unqualified_media": {"image": reason}}}
        # Its reference is refused rather than scored.
        with self.assertRaisesRegex(ValueError, "encoder declaration coverage differs"):
            self.preflight()
        self.spec["references"] = []
        self.assertEqual(self.preflight()["subject"]["unqualified"], {"encoder.json": reason})
        # A qualified kind of the same encoder keeps its reference required.
        self.encoder["kinds"] = {"image", "video"}
        with self.assertRaisesRegex(ValueError, "missing declared encoder"):
            self.preflight()

    def test_links_and_path_escapes_refuse(self):
        self.spec["requests"][0]["path"] = "../outside.json"
        with self.assertRaisesRegex(ValueError, "invalid reference member"):
            self.preflight()
        self.spec["requests"][0]["path"] = "body.json"
        body = self.data / "body.json"
        body.rename(self.data / "actual.json")
        body.symlink_to("actual.json")
        with self.assertRaises(ValueError):
            self.preflight()

    def test_gate_requires_reference_admission_before_scoring(self):
        import os
        from types import SimpleNamespace
        import encoder_parity
        run = self.root / "capture"
        run.mkdir()
        self.write(run / "parity.json", {"schema": encoder_parity.PARITY_RUN, "model": "fixture"})
        with patch.object(capture, "configured", return_value=True), \
                patch.object(capture, "admit", side_effect=ValueError("reference binding differs")), \
                patch.object(encoder_parity, "qualification_config", return_value=self.config), \
                patch.dict(os.environ, {"CHOPINFER_QUAL_CONFIG": "synthetic"}), \
                self.assertRaisesRegex(ValueError, "reference binding differs"):
            encoder_parity.parity(SimpleNamespace(run=str(run), url="unused"))

    def test_supplement_cannot_substitute_reference_or_capture_source(self):
        prepared = self.preflight()
        run = self.root / "capture"
        (run / "engine").mkdir(parents=True)
        record = {"subject": prepared["subject"], "server": {"source_commit": "c" * 40},
                  "requests": [{"name": "image"}]}
        self.write(run / "tap-capture.json", record)
        manifest = capture.assemble(run, prepared, record, self.config)
        for change in ({"source": "d" * 40}, {"subject": {}}, {"references": []}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                capture.admit(run, dict(manifest, **change), self.config)



class EncoderRunnerSitesTests(unittest.TestCase):
    def test_encoder_arms_request_their_native_encoder_outputs(self):
        import release_preview as runner
        models = sorted(path.parent for path in capture.ROOT.glob("models/*/encoder-parity-inputs.json"))
        self.assertTrue(models)
        for model in models:
            with self.subTest(model=model.name):
                environment = runner.module_option("encoder_parity", "RUNNER_ENVIRONMENT", {}, model.name)
                sites = environment["encoders"]["CHOPINFER_TAPS_SITES"].split(",")
                native = json.loads((model / "taps.json").read_text())
                outputs = {tap["name"] for tap in native["taps"]
                           if tap["name"].startswith("enc.") and "{" not in tap["name"]}
                self.assertTrue(sites)
                self.assertEqual(len(sites), len(set(sites)))
                self.assertEqual(set(sites), outputs)

    def test_unconfigured_model_has_no_encoder_environment(self):
        import release_preview as runner
        with patch.object(capture, "configured", return_value=False):
            self.assertEqual(runner.module_option("encoder_parity", "RUNNER_ENVIRONMENT", {}, "fixture"), {})

if __name__ == "__main__":
    unittest.main()
