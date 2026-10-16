"""Qwen3.8's exact input is its bound F32 teacher reference (the GPU
runner's), pinned like MiMo's: verify and prefix refuse a changed copy;
every gate entry point binds the live server's /health identity to the
pinned checkpoint."""

from qual_test_config import configure, qwen_probe_workload, fixture_lock

import argparse
import contextlib
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import common
import declaration
import prefix_exact
import verify_exact
from test_checkpoint_binding import LiveBindingCases

GATE = common.ROOT / "models/qwen38/qual.json"
CONFIG = declaration.load(GATE)
PIN = CONFIG["qualification"]["checkpoint"]
EXACT = common.ROOT / CONFIG["inputs"]["exact"]


def setUpModule():
    configure("qwen38")


# Audited to run once rather than under every gate config (ci/qualification.py).
QUALIFICATION_RUNS_ONCE = "exact-input pin and its replayed gate bindings"


@contextlib.contextmanager
def gate(**inputs):
    """The Qwen gate with some inputs (and the exact pin's path) moved."""
    config = copy.deepcopy(CONFIG)
    config["inputs"].update({key: str(value) for key, value in inputs.items()})
    if "exact" in inputs:
        config["qualification"]["references"]["teacher"]["path"] = str(inputs["exact"])
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory, "gate.json")
        path.write_text(json.dumps(config))
        with mock.patch.dict(os.environ, {"CHOPINFER_QUAL_CONFIG": str(path)}), fixture_lock(config, directory):
            yield Path(directory)


def changed_copy(directory):
    """The reference with one prompt token changed and its file kept valid."""
    value = json.loads(EXACT.read_text())
    value["texts"][3]["prompt_token_ids"][5] += 1
    path = Path(directory, "teacher-reference.json")
    path.write_text(json.dumps(value))
    return path


class ExactPinTests(unittest.TestCase):
    def test_the_pinned_reference_loads_everywhere(self):
        self.assertEqual(EXACT, common.ROOT / CONFIG["qualification"]["references"]["teacher"]["path"])
        self.assertEqual(common.pinned_kind(EXACT), "teacher")
        self.assertTrue(verify_exact.exact_pinned())
        self.assertEqual(len(verify_exact.load_exact(EXACT)["texts"]), 35)
        self.assertEqual(prefix_exact.load_exact(EXACT)["texts"][0]["score_from"], 1)

    def test_a_changed_reference_is_refused_by_verify_and_prefix(self):
        with tempfile.TemporaryDirectory() as directory:
            changed = changed_copy(directory)
            with gate(exact=changed):
                for label, load in (("verify", verify_exact.load_exact), ("prefix", prefix_exact.load_exact)):
                    with self.subTest(label), self.assertRaisesRegex(SystemExit, "refused: .* not the pinned"):
                        load(changed)

    def test_a_reference_of_another_revision_is_refused(self):
        value = json.loads(EXACT.read_text())
        value["provenance"]["checkpoint"]["revision"] = "0" * 40
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "exact.json")
            workload = EXACT.with_name(EXACT.stem + ".workload.json")
            path.with_name("exact.workload.json").write_bytes(workload.read_bytes())
            path.write_bytes(json.dumps(value).encode())
            config = copy.deepcopy(CONFIG)
            config["qualification"]["references"]["teacher"]["sha256"] = hashlib.sha256(
                path.read_bytes()).hexdigest()
            config["qualification"]["references"]["teacher"]["path"] = str(path)
            with fixture_lock(config, directory), self.assertRaisesRegex(ValueError, "checkpoint provenance"):
                common.pinned_reference("teacher", path, config)
            stripped = json.loads(EXACT.read_text())
            del stripped["provenance"]["variants"]["f64"]
            path.write_bytes(json.dumps(stripped).encode())
            config["qualification"]["references"]["teacher"]["sha256"] = hashlib.sha256(
                path.read_bytes()).hexdigest()
            with fixture_lock(config, directory), self.assertRaisesRegex(ValueError, "variants"):
                common.pinned_reference("teacher", path, config)

    def test_reports_refuse_records_of_another_checkpoint_and_mark_other_references(self):
        digest = hashlib.sha256(EXACT.read_bytes()).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            record = Path(directory, "prefix.json")
            record.write_text(json.dumps({"reference_sha256": digest, "server": {"model": "m"}}))
            args = argparse.Namespace(exact=None, record=[str(record)], split_record=[])
            with self.assertRaisesRegex(SystemExit, "pinned checkpoint"):
                prefix_exact.report(args)
            # A legacy record, bound by a log before /health named its process.
            record.write_text(json.dumps({"reference_sha256": digest, "server": {
                "model": "m", "checkpoint": {"checkpoint_identity": PIN["checkpoint_identity"]}}}))
            with self.assertRaisesRegex(SystemExit, "not bound live"):
                prefix_exact.report(args)
            other = Path(directory, "other.json")
            other.write_bytes(EXACT.read_bytes())
            self.assertTrue(prefix_exact.unpinned(other))
            self.assertFalse(prefix_exact.unpinned(EXACT))
            verify = {"arm": "ar", "k": 0, "concurrency": 1, "detokenized": {}, "prefill": {},
                      "reference_sha256": digest, "server": {"model": PIN["served_model_name"]}, "texts": []}
            Path(directory, "ar.json").write_text(json.dumps(verify))
            with self.assertRaisesRegex(SystemExit, "pinned checkpoint"):
                verify_exact.report(argparse.Namespace(exact=None, ar=str(Path(directory, "ar.json")),
                                                       verify=[str(Path(directory, "ar.json"))]))


class QwenLiveBindingTests(LiveBindingCases, unittest.TestCase):
    PIN = PIN
    # The image probe gate verifies its binding against a workload under $CHOPINFER_DATA.
    ARGUMENTS = {**({"multimodal": []} if qwen_probe_workload() else {}), "verify": [], "prefix": []}


if __name__ == "__main__":
    unittest.main()
