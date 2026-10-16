"""Regression evidence for authenticated media, lock shape and frontend bytes."""

from contextlib import contextmanager
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import common
import declaration
import frontend_goldens
import multimodal
import refs


@contextmanager
def locked_inputs(inputs, qualification=None):
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        config = {"schema": "chopinfer-release-gates/v1", "model": "synthetic", "inputs": inputs,
                  "qualification": {"mechanics_only": True, **(qualification or {})}}
        lock = refs.lock_path("synthetic", root)
        lock.parent.mkdir(parents=True)
        lock.write_text(json.dumps(refs.snapshot(config, root)))
        policy = root / "policy.json"
        policy.write_text(json.dumps(config))
        with patch.dict(os.environ, {"CHOPINFER_QUAL_MODEL_ROOT": str(root),
                                      "CHOPINFER_QUAL_CONFIG": str(policy)}):
            yield root, config, lock


class AuthenticatedReadRegressions(unittest.TestCase):
    def test_media_worker_rejects_tampered_expectations_before_evaluation(self):
        with tempfile.TemporaryDirectory() as temporary:
            media = Path(temporary)
            payload = b"fixed audio"
            (media / "clip.wav").write_bytes(payload)
            sha = hashlib.sha256(payload).hexdigest()
            manifest = {"files": {"clip.wav": sha}, "audio_expected_clauses": ["a short phrase"]}
            (media / "manifest.json").write_text(json.dumps(manifest))
            case = {"id": "audio", "kind": "audio", "file": "clip.wav", "prompt": "Transcribe"}
            reference = {"cases": [{"answer": "a short phrase"}]}
            with locked_inputs({"media_fixtures": str(media)}), \
                    patch.object(multimodal, "cases", return_value=[case]), \
                    patch.object(multimodal, "evaluate_case", return_value={"passed": True}) as evaluate:
                manifest["audio_expected_clauses"] = []
                (media / "manifest.json").write_text(json.dumps(manifest))
                with self.assertRaisesRegex(ValueError, "media_fixtures.*manifest.json"):
                    multimodal.run(None, {"model": "fixture"}, reference, media, {"clip.wav": sha})
                evaluate.assert_not_called()

    def test_directory_shaped_file_entry_cannot_disable_api_authentication(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Path(temporary) / "api.json"
            fixture.write_text("{}")
            with locked_inputs({"api_samples": str(fixture)}) as (_, config, lock):
                value = json.loads(lock.read_text())
                value["entries"]["api_samples"].update(directory=True, scope=[], members={})
                lock.write_text(json.dumps(value))
                fixture.write_text("[]")
                with self.assertRaisesRegex(ValueError, "lock entry api_samples"):
                    common.input_bytes("api_samples", config=config)

    def test_discovery_refuses_directory_shaped_file_entry_without_traceback(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = copy.deepcopy(declaration.load(declaration.path("mimo26")))
            config["qualification"]["mechanics_only"] = True
            for location in (config["inputs"]["chat_renders"], "fixtures/mimo26/teacher-texts.json"):
                target = root / location
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(common.ROOT / location, target)
            lock = json.loads(refs.lock_path("mimo26").read_text())
            lock["entries"]["chat_renders"].update(directory=True, scope=[], members={})
            path = refs.lock_path("mimo26", root)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(lock))
            policy = root / "gate.json"
            policy.write_text(json.dumps(config))
            environment = {key: value for key, value in os.environ.items() if not key.startswith("CHOPINFER_")}
            environment["CHOPINFER_LOCAL_ENV"] = str(root / "missing.env")
            result = subprocess.run([sys.executable, "-B", str(common.ROOT / "tools/qual/release_gate.py"),
                                     "--config", str(policy), "--model-root", str(root), "--list-inputs"],
                                    env=environment, capture_output=True, text=True, check=False)
            self.assertEqual(result.returncode, 2, result.stdout)
            self.assertIn("lock entry chat_renders", result.stderr)
            self.assertNotIn("Traceback", result.stderr)

    def test_frontend_parses_authenticated_bytes_without_reopening_the_path(self):
        with tempfile.TemporaryDirectory() as temporary:
            frontend = Path(temporary) / "frontend.json"
            frontend.write_text(json.dumps({"reasoning": {"thinking_keywords": ["enable_thinking"]}}))
            with locked_inputs({"frontend": str(frontend)}):
                verify = refs.verify_input
                calls = []

                def replace_after_verification(*args, **kwargs):
                    data = verify(*args, **kwargs)
                    calls.append(data)
                    frontend.write_text(json.dumps({"reasoning": {"thinking_keywords": ["unverified"]}}))
                    return data

                with patch.object(refs, "verify_input", side_effect=replace_after_verification):
                    self.assertEqual(common.thinking_kwargs(False), {"enable_thinking": False})
                self.assertEqual(len(calls), 1)


class ReferenceProductionTests(unittest.TestCase):
    """A lock pins bytes. Its object records (what proved or adopted each
    object) are information: no reader admits or refuses bytes by them,
    except that a numerical reference must record its runner binding
    (`refs.gate_reference_producers`)."""

    def read_with_record(self, record, tamper=False):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            path = directory / "api-shapes.json"
            path.write_text('{"sample": {"status": 200}}')
            with locked_inputs({"frontend_capture": str(directory)}) as (_, config, lock_path):
                lock = json.loads(lock_path.read_text())
                obj = lock["objects"][refs.digest(path)]
                obj.pop("production", None)
                if record is not None:
                    obj["production"] = record
                lock_path.write_text(json.dumps(lock))
                if tamper:
                    path.write_text('{"sample": {"status": 500}}')
                refs.load_lock(config)
                for name, read in {"frontend": lambda: frontend_goldens.reference_bytes(config, "frontend_capture", path),
                                   "read_member": lambda: refs.read_member(config, "frontend_capture", path)}.items():
                    with self.subTest(record=record, reader=name):
                        if tamper:
                            with self.assertRaisesRegex(ValueError, "frontend_capture: sha256 differs"):
                                read()
                        else:
                            self.assertEqual(read(), path.read_bytes())
                if tamper:
                    with self.assertRaisesRegex(ValueError, "frontend_capture: sha256 differs"):
                        refs.verify_input(config, "frontend_capture", directory)
                else:
                    refs.verify_input(config, "frontend_capture", directory)

    def test_object_records_do_not_admit_or_refuse_bytes(self):
        for record in (None, {"producer": "legacy-unknown", "adoption_commit": "0" * 40, "adoption_date": "x"},
                       {"producer": "runner-layer-reference"}, {"producer": "unrecognized-producer"}):
            self.read_with_record(record)
            self.read_with_record(record, tamper=True)

    def test_every_committed_lock_is_accepted_with_its_records(self):
        paths = [path for path in declaration.shipped() if path != declaration.MECHANICS]
        self.assertTrue(paths)
        for path in paths:
            with self.subTest(declaration=path.name):
                config = declaration.load(path)
                lock = refs.load_lock(config)
                self.assertIsNotNone(lock)
                committed = json.loads(refs.lock_path(config["model"]).read_text())
                self.assertTrue(all(lock["objects"][sha] == committed["objects"][sha] for sha in lock["objects"]))
        # Directory scopes live in each lock entry; no separate policy file declares them.
        self.assertEqual([*common.ROOT.glob("tools/qual/input-lock-policy.json"),
                          *common.ROOT.glob("models/*/input-lock-policy.json")], [])

    def test_every_shipped_gate_reference_carries_a_runner_binding(self):
        # Numerical references are runner-bound: their bytes, binding
        # included, are what the reference lock pins.
        checked = 0
        for path in declaration.shipped():
            if path == declaration.MECHANICS:
                continue
            config = declaration.load(path)
            pin = common.checkpoint_pin(config)
            for kind, spec in config["qualification"].get("references", {}).items():
                fixture = common.declared_path(spec["path"])
                if fixture is None or not fixture.is_file():
                    continue
                with self.subTest(declaration=path.name, reference=kind):
                    binding = json.loads(fixture.read_bytes()).get("binding")
                    self.assertIsNotNone(binding)
                    self.assertEqual(binding["lock"], {"path": pin["lock"], "variant": pin.get("pin")})
                    checked += 1
        self.assertGreater(checked, 0)
