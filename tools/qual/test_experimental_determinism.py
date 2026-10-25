"""Strict diagnostic pins and refusal of experimental release evidence."""
import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import common
import declaration
import determinism
import experimental_determinism as experimental
import kinds
from test_determinism import BoundClient, startup_log


class ExperimentalDeterminismTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        config = sorted(declaration.ROOT.glob("models/*/qual.json"))[0]
        self.environment = patch.dict(os.environ, {"CHOPINFER_QUAL_CONFIG": str(config)})
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.checkpoint = common.checkpoint_pin()
        self.options = {"verify_graph_rows": 64, "enabled": True, "mode": "capture-\u03b1"}
        self.pin = {"model_options": self.options,
                    "checkpoint_identity": self.checkpoint["checkpoint_identity"],
                    "startup_nonce": "a" * 32}

    def client(self):
        client = BoundClient(self.pin["checkpoint_identity"], model=self.checkpoint["served_model_name"])
        client.serving.update(model_options=copy.deepcopy(self.options), model_options_qualified=False)
        return client

    def arguments(self, pin=None):
        path = self.root / "pin.json"
        path.write_text(json.dumps(self.pin if pin is None else pin, indent=2))
        path.chmod(0o600)
        return SimpleNamespace(
            url="http://127.0.0.1:39001", timeout=10, model=None, arm="experimental", run_id="run",
            image=True, image_reason=None,
            server_log=startup_log(self.root, self.pin["checkpoint_identity"]),
            experimental=True, experimental_pin=path,
            experimental_pin_sha256=hashlib.sha256(path.read_bytes()).hexdigest())

    def record(self, client, args):
        written = []
        with patch.object(determinism, "Client", return_value=client), \
             patch.object(determinism, "answer_control", return_value={}), \
             patch.object(determinism, "receipt", side_effect=lambda check, server, value, mechanics: written.append(
                 dict(value, check=check, server=server, mechanics_only=mechanics))):
            determinism.record(args)
        self.assertEqual(len(written), 1)
        return written[0]

    def test_capture_preserves_unqualified_status_and_all_twelve_requests(self):
        client = self.client()
        value = self.record(client, self.arguments())
        self.assertFalse(value["qualified"])
        self.assertFalse(value["server"]["health"]["serving"]["model_options_qualified"])
        self.assertEqual(value[experimental.MARKER]["pin"], self.pin)
        self.assertEqual(value["server"][experimental.MARKER], value[experimental.MARKER])
        self.assertEqual(len(value["requests"]), 12)
        self.assertEqual(len([path for path, _ in client.bodies if path != "/tokenize"]), 12)
        self.assertTrue(value["cold_repeat"]["equal"])
        self.assertTrue(value["prefix_hit"]["equal"])
        self.assertFalse(common.bound_live(value["server"]))
        compared = determinism.compare(value, value, experimental=True)
        self.assertTrue(compared["passed"])
        self.assertFalse(compared["qualified"])
        with self.assertRaisesRegex(ValueError, "not release qualification"):
            determinism.compare(value, value)

    def test_missing_extra_and_mistyped_options_fail_before_requests(self):
        for name, change in (
                ("missing", lambda p: p["model_options"].pop("enabled")),
                ("extra", lambda p: p["model_options"].update(extra=True)),
                ("bool-as-int", lambda p: p["model_options"].update(enabled=1)),
                ("int-as-float", lambda p: p["model_options"].update(verify_graph_rows=64.0)),
                ("checkpoint", lambda p: p.update(checkpoint_identity="f" * 64)),
                ("nonce", lambda p: p.update(startup_nonce="b" * 32))):
            with self.subTest(case=name):
                pin = copy.deepcopy(self.pin)
                change(pin)
                client = self.client()
                with self.assertRaises(ValueError):
                    self.record(client, self.arguments(pin))
                self.assertEqual(client.bodies, [])

    def test_pin_requires_opt_in_digest_log_and_safe_file(self):
        for name in ("opt-in", "digest", "log", "modified", "writable", "symlink", "fifo", "duplicate"):
            with self.subTest(case=name):
                args = self.arguments()
                client = self.client()
                if name == "opt-in": args.experimental = False
                elif name == "digest": args.experimental_pin_sha256 = "0" * 64
                elif name == "log": args.server_log = None
                elif name == "modified": args.experimental_pin.write_text("{}")
                elif name == "writable": args.experimental_pin.chmod(0o622)
                elif name == "symlink":
                    link = self.root / "linked-pin.json"
                    link.symlink_to(args.experimental_pin)
                    args.experimental_pin = link
                elif name == "fifo":
                    pipe = self.root / "pipe-pin.json"
                    os.mkfifo(pipe)
                    args.experimental_pin = pipe
                else:
                    args.experimental_pin.write_text('{"model_options":{},"model_options":{}}')
                    args.experimental_pin_sha256 = hashlib.sha256(args.experimental_pin.read_bytes()).hexdigest()
                with self.assertRaises((ValueError, OSError)):
                    self.record(client, args)
                self.assertEqual(client.bodies, [])

    def test_final_identity_drift_or_missing_response_identity_prevents_publication(self):
        for name in ("options", "checkpoint", "nonce", "qualification", "missing-header", "other-response"):
            with self.subTest(case=name):
                args = self.arguments()
                client = self.client()
                original = client.identity
                calls = []

                def identity(model=None):
                    value = copy.deepcopy(original(model))
                    calls.append(value)
                    if len(calls) == 2:
                        serving = value["health"]["serving"]
                        if name == "options": serving["model_options"]["verify_graph_rows"] = 32
                        elif name == "checkpoint": serving["checkpoint_identity"] = "f" * 64
                        elif name == "nonce": serving["startup_nonce"] = "b" * 32
                        elif name == "qualification": serving["model_options_qualified"] = True
                    return value

                client.identity = identity
                if name == "missing-header": client.responders.add((self.pin["checkpoint_identity"], None))
                elif name == "other-response": client.responders.add((self.pin["checkpoint_identity"], "b" * 32))
                with patch.object(determinism, "receipt") as publish, self.assertRaises(SystemExit):
                    with patch.object(determinism, "Client", return_value=client), \
                         patch.object(determinism, "answer_control", return_value={}):
                        determinism.record(args)
                publish.assert_not_called()

    def test_release_admission_refuses_marker_presence_even_when_health_claims_qualification(self):
        value = self.record(self.client(), self.arguments())
        for placement in ("receipt", "server", "both"):
            for health in ("original", "qualified", "removed"):
                for mechanics in (False, True):
                    with self.subTest(placement=placement, health=health, mechanics=mechanics):
                        bad = copy.deepcopy(value)
                        if placement == "receipt": bad["server"].pop(experimental.MARKER)
                        elif placement == "server": bad.pop(experimental.MARKER)
                        if health == "qualified":
                            bad["server"]["health"]["serving"]["model_options_qualified"] = True
                        elif health == "removed": bad["server"].pop("health")
                        with self.assertRaisesRegex(ValueError, "not release qualification"):
                            kinds.admit(bad, mechanics=mechanics)


if __name__ == "__main__":
    unittest.main()
