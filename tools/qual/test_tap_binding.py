"""tap_binding: a tap step counts only when its own metadata names the live
candidate process (startup nonce) and the fresh request (engine request
ID) that wrote it, and the candidate's whole serving identity is the
capture's. Synthetic servers and steps; needs numpy for the step files."""

import contextlib
import copy
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
try:
    import numpy as np
    from tools.qual.test_tapdiff import item, write_step
except ModuleNotFoundError:
    np = None
import common  # noqa: E402
import declaration  # noqa: E402
import tap_binding  # noqa: E402

PIN = declaration.load(declaration.path("qwen38"))["qualification"]["checkpoint"]
URL = "http://127.0.0.1:39001"
SERVING = {"bundle_id": "qwen38.flash.v2", "checkpoint_identity": PIN["checkpoint_identity"],
           "startup_nonce": "a" * 32, "package_identity": "1" * 64, "metadata_identity": "2" * 64,
           "source_commit": "3" * 40, "version": "0.1.0", "profile": {"name": None, "tensor_parallel_size": 1}}


def serving(model):
    """SERVING for `model`'s checkpoint pin."""
    pin = declaration.load(declaration.path(model))["qualification"]["checkpoint"]
    return SERVING | {"checkpoint_identity": pin["checkpoint_identity"]}


def server_log(path, serving=SERVING, address="127.0.0.1:39001"):
    path.write_text(
        f"startup direct_package: {json.dumps({'checkpoint_identity': serving['checkpoint_identity']})}\n"
        f"startup serving_identity: {json.dumps(serving)}\n"
        + json.dumps({"status": "listening", "address": address}, separators=(",", ":")) + "\n")
    return path


class FakeServer:
    """A bound diagnostic-taps server. Each request is engine request N (from
    `first_request`): it writes `steps` steps of `tensors` rows, each naming
    its process and request, and traces its X-Client-Request-Id to N in `log`.
    `during(taps)` runs inside the request, as another writer could. With
    `steps_of(engine, body)`, a request writes the (items, tensors, encoder)
    steps it returns instead."""

    def __init__(self, taps, log, tensors, tokens, serving=SERVING, cached=0, steps=1, first_request=1,
                 during=None, steps_of=None):
        self.url, self.taps, self.log, self.tensors, self.tokens = URL, taps, log, tensors, tokens
        self.serving, self.cached, self.steps, self.next, self.during = serving, cached, steps, first_request, during
        self.steps_of = steps_of or (lambda engine, body: [
            ([item(engine, 0, list(range(len(self.tokens))), self.tokens)], self.tensors, None)] * self.steps)
        self.responders = set()

    def identity(self, model=None):
        self.responders.add((self.serving["checkpoint_identity"], self.serving["startup_nonce"]))
        return {"url": self.url, "model": model, "models": [model], "health": {"serving": self.serving}}

    def request(self, path, body, headers=None):
        assert body["cache_salt"] and headers["X-Client-Request-Id"]
        self.responders.add((self.serving["checkpoint_identity"], self.serving["startup_nonce"]))
        engine, self.next = self.next, self.next + 1
        for items, tensors, encoder in self.steps_of(engine, body):
            write_step(self.taps / f"step{len(list(self.taps.glob('step*'))):04d}", items,
                       {name: [(0, rows, dtype)] for name, (rows, dtype) in tensors.items()}
                       if any(isinstance(value, tuple) for value in tensors.values())
                       else {name: [(0, rows, "f32")] for name, rows in tensors.items()},
                       encoder=encoder, producer=self.serving["startup_nonce"])
        if self.during:
            self.during(self.taps)
        with self.log.open("a") as log:
            log.write(json.dumps({"schema": tap_binding.TRACE, "request_id": headers["X-Client-Request-Id"],
                                  "engine_request_id": engine}) + "\n")
        return {"id": f"cmpl_{engine}", "usage": {"prompt_tokens": len(self.tokens),
                                                   "prompt_tokens_details": {"cached_tokens": self.cached}}}, 0.1


def serve(fake):
    """Patches common.Client to answer from `fake`."""
    return patch.object(common, "Client", lambda url, retries=0, timeout=None: fake)


@unittest.skipIf(np is None, "needs numpy (set CHOPINFER_REFRUN_PYTHON)")
class TapBindingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR"))
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = declaration.load(declaration.path("qwen38"))
        self.tokens = list(range(100, 110))
        self.tensors = {"L00.out": np.ones((10, 16), dtype=np.float32)}

    def capture(self, name="cap", **server):
        taps = self.root / f"taps-{name}"
        taps.mkdir()
        log = server_log(self.root / f"{name}.log", server.get("serving", SERVING))
        fake = FakeServer(taps, log, self.tensors, self.tokens, **server)
        with serve(fake):
            return tap_binding.capture(URL, log, taps, [("p", "/v1/completions", {"prompt": self.tokens})],
                                       self.root / name, self.config)

    def admit(self, directory, serving=SERVING):
        fake = FakeServer(self.root, self.root / "unused.log", self.tensors, self.tokens, serving=serving)
        with serve(fake):
            return tap_binding.admit(directory, URL, self.config)

    def test_a_capture_names_its_process_and_request_and_admits_against_it(self):
        manifest = self.capture()
        self.assertEqual(manifest["server"], SERVING)
        (record,) = manifest["requests"]
        self.assertEqual((record["engine_request_id"], record["steps"]), (1, ["step0000"]))
        self.assertEqual(self.admit(self.root / "cap")["requests"], manifest["requests"])

    def test_each_differing_live_identity_field_is_insufficient(self):
        self.capture()
        changes = {"bundle_id": "other", "package_identity": "4" * 64, "metadata_identity": "5" * 64,
                   "startup_nonce": "b" * 32, "source_commit": "6" * 40, "version": "0.2.0",
                   "profile": {"name": None, "tensor_parallel_size": 2}}
        self.assertEqual(set(changes), set(tap_binding.IDENTITY) - {"checkpoint_identity"})
        for key, value in changes.items():
            with self.subTest(key=key), self.assertRaisesRegex(tap_binding.Unbound, key):
                self.admit(self.root / "cap", SERVING | {key: value})
        with self.assertRaisesRegex(ValueError, "checkpoint"):
            self.admit(self.root / "cap", SERVING | {"checkpoint_identity": "7" * 64})

    def test_a_step_copied_in_during_a_genuine_request_is_refused(self):
        old = self.root / "old-step"
        write_step(old, [item(1, 0, list(range(10)), self.tokens)],
                   {name: [(0, rows, "f32")] for name, rows in self.tensors.items()}, producer="c" * 32)
        copied = lambda taps: shutil.copytree(old, taps / "step0999")  # noqa: E731
        with self.assertRaisesRegex(tap_binding.Unbound, "written by process " + "c" * 32):
            self.capture(during=copied)
        # The same process's step of an earlier request.
        earlier = self.root / "earlier-step"
        write_step(earlier, [item(7, 0, list(range(10)), self.tokens)],
                   {name: [(0, rows, "f32")] for name, rows in self.tensors.items()}, producer="a" * 32)
        with self.assertRaisesRegex(tap_binding.Unbound, r"engine requests \[7\]"):
            self.capture("again", during=lambda taps: shutil.copytree(earlier, taps / "step0999"))

    def test_a_relabeled_manifest_or_edited_step_is_refused(self):
        self.capture()
        manifest_path = self.root / "cap" / tap_binding.MANIFEST
        original = json.loads(manifest_path.read_text())
        candidate = SERVING | {"startup_nonce": "b" * 32, "source_commit": "6" * 40}
        relabeled = copy.deepcopy(original)
        relabeled["server"] |= {"source_commit": candidate["source_commit"]}
        manifest_path.write_text(json.dumps(relabeled))
        with self.assertRaisesRegex(tap_binding.Unbound, "startup_nonce"):
            self.admit(self.root / "cap", candidate)
        relabeled["server"] |= {"startup_nonce": candidate["startup_nonce"]}
        manifest_path.write_text(json.dumps(relabeled))
        with self.assertRaisesRegex(tap_binding.Unbound, "written by process " + "a" * 32):
            self.admit(self.root / "cap", candidate)
        manifest_path.write_text(json.dumps(original))
        capture = self.root / "cap" / tap_binding.STEPS / "step0000" / "capture.json"
        value = json.loads(capture.read_text())
        value["producer"]["startup_nonce"] = "b" * 32
        capture.write_text(json.dumps(value))
        with self.assertRaisesRegex(tap_binding.Unbound, "differs from the files"):
            self.admit(self.root / "cap")

    def test_cached_prompts_untraced_requests_and_steps_without_a_producer_are_refused(self):
        with self.assertRaisesRegex(tap_binding.Unbound, "cached"):
            self.capture("cached", cached=4)
        with self.assertRaisesRegex(tap_binding.Unbound, "written by process None"):
            self.capture("unnamed", during=lambda taps: [
                (step / "capture.json").write_text(json.dumps({"schema": "chopinfer.taps.v1"}))
                for step in taps.glob("step*")])
        untraced = FakeServer.request

        def request(fake, path, body, headers=None):
            # A server without CHOPINFER_REQUEST_TRACE=1 traces nothing.
            response = untraced(fake, path, body, headers)
            fake.log.write_text("\n".join(line for line in fake.log.read_text().splitlines()
                                          if tap_binding.TRACE not in line) + "\n")
            return response
        with patch.object(FakeServer, "request", request), patch.object(tap_binding, "TRACE_WAIT_SECONDS", 0.1), \
                self.assertRaisesRegex(tap_binding.Unbound, "CHOPINFER_REQUEST_TRACE"):
            self.capture("untraced")


if __name__ == "__main__":
    unittest.main()
