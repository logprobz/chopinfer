"""The MiMo gates accept only the pinned references and a server that
reported the pinned checkpoint, through their real entry points. The
teacher check binds the same way inside `chopinfer eval run`
(crates/runtime-eval/tests/teacher.rs)."""

from qual_test_config import setUpModule  # noqa: F401

import argparse
import contextlib
import copy
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest import mock

import common
import declaration
import multimodal
import prefix_exact
import release_gate
import verify_exact

# Audited to run once rather than under every gate config (ci/qualification.py).
QUALIFICATION_RUNS_ONCE = "reference and checkpoint binding against an in-process replay server"


CONFIG = declaration.load(declaration.path("mimo26"))
PIN = CONFIG["qualification"]["checkpoint"]
TEACHER = common.ROOT / CONFIG["qualification"]["references"]["teacher"]["path"]
MEDIA = common.ROOT / CONFIG["qualification"]["references"]["media"]["path"]
RL_IDENTITY = "0" * 64
# The startup nonces of the live process and of an earlier one at its address.
NONCE, STALE_NONCE = "a" * 32, "b" * 32


def server_log(directory, port, identity=PIN["checkpoint_identity"], nonce=NONCE):
    path = Path(directory, f"server-{port}-{identity[:8]}-{nonce[:4]}.log")
    record = json.dumps({"package_identity": "p", "checkpoint_identity": identity})
    serving = json.dumps({"checkpoint_identity": identity, "startup_nonce": nonce})
    path.write_text(f"startup direct_package: {record}\n"
                    f"startup serving_identity: {serving}\n"
                    f"startup direct_load: {json.dumps({'checkpoint_identity': identity})}\n"
                    + json.dumps({"status": "listening", "address": f"127.0.0.1:{port}"},
                                 separators=(",", ":")) + "\n")
    return path


def live(identity=PIN["checkpoint_identity"], nonce=NONCE):
    """A client factory for a live server with this /health identity."""
    return lambda url, timeout=None: ReplayServer(url, identity=identity, nonce=nonce)


class ReplayServer:
    """A live server's /health identity and the processes that answered
    each scoring request."""

    def __init__(self, url, timeout=None, model=PIN["served_model_name"], identity=PIN["checkpoint_identity"],
                 nonce=NONCE, replacement=None, interloper=None):
        self.url, self.model = url, model
        self.serving = {"checkpoint_identity": identity, "startup_nonce": nonce}
        # The /health identity of a process that takes the address after the first answer.
        self.replacement = replacement
        # The identity of a process that writes the first scored response only.
        self.interloper = interloper
        self.responders = set()

    def responded(self, serving):
        self.responders.add((serving.get("checkpoint_identity"), serving.get("startup_nonce")))

    def identity(self, model=None):
        serving = self.serving
        if self.replacement is not None:
            self.serving = self.replacement
        self.responded(serving)
        return {"url": self.url, "model": self.model, "models": [self.model],
                "health": {"status": "ready", "serving": serving}}

    def request(self, path, body):
        serving, self.interloper = self.serving if self.interloper is None else self.interloper, None
        self.responded(serving)
        assert path == SCORE, path
        return {}, 0.01


def run_main(module, argv, client, patches=()):
    """`module.main()`'s exit code (or refusal message) and its receipt."""
    with tempfile.TemporaryDirectory() as receipts, contextlib.ExitStack() as stack:
        stack.enter_context(mock.patch.dict(os.environ, {"CHOPINFER_QUAL_RECEIPTS": receipts}))
        stack.enter_context(mock.patch.object(module, "Client", client))
        stack.enter_context(mock.patch.object(sys, "argv", argv))
        stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
        for name, value in patches:
            stack.enter_context(mock.patch.object(module, name, value))
        code = 0
        try:
            module.main()
        except SystemExit as exited:
            code = exited.code
        written = list(Path(receipts).glob("*.json"))
        return code, json.loads(written[0].read_text()) if written else None


# The request a scoring step's stand-in sends, so the server writes a scored response.
SCORE = "/score"


def scoring(value):
    """A scoring step's stand-in: one request to its client, then `value`."""
    def score(client, *args, **kwargs):
        client.request(SCORE, {})
        return copy.deepcopy(value)
    return score


# Each gate entry point scoring the server at URL, with the scoring itself
# replaced: the binding happens before it and the confirmation after it.
URL = "http://127.0.0.1:8101"
ENTRY_POINTS = {
    "multimodal": (multimodal, ["multimodal.py", "--url", URL], (
        ("run", scoring({"passed": True, "rows": [], "errors": []})),
        ("run_mixed", scoring({"passed": True, "case_count": 4, "rows": [], "errors": []})),
        ("load_probes", mock.Mock(return_value=([], {}, "capture"))),
        ("probe_images", mock.Mock(return_value={})),
        ("probe_references", mock.Mock(return_value=({}, {}, {}))),
        ("run_probes", scoring({"passed": True})))),
    "verify": (verify_exact, ["verify_exact.py", "record", "--url", URL, "--arm", "ar"], (
        ("record_pass", scoring({"requests": []})),
        ("metrics_text", mock.Mock(return_value="")),
        ("prefill_capture", mock.Mock(return_value={})),
        ("detokenized", mock.Mock(return_value={})))),
    "prefix": (prefix_exact, ["prefix_exact.py", "record", "--url", URL, "--skip-media",
                              "--acceptance-prompts", "0"], (
        ("score_position", scoring({})),
        ("score_prompt_end", scoring({})),
        ("detokenized", mock.Mock(return_value={})),
        ("evaluate", mock.Mock(return_value={"passed": True})),
        ("skipped_checks", mock.Mock(return_value=[])))),
}


def run_entry(name, extra, client):
    module, argv, patches = ENTRY_POINTS[name]
    return run_main(module, [*argv, *extra], client, patches)


class LiveBindingCases:
    """Every gate entry point binds the live server's /health identity and
    confirms it after scoring. A test case names the gate's pin, the entry
    points it has and their extra arguments."""

    PIN = None
    ARGUMENTS = {}

    def client(self, identity=None, nonce=NONCE, replacement=None, interloper=None):
        identity = self.PIN["checkpoint_identity"] if identity is None else identity
        return lambda url, timeout=None: ReplayServer(url, model=self.PIN["served_model_name"], identity=identity,
                                                      nonce=nonce, replacement=replacement,
                                                      interloper=interloper)

    def score(self, name, log, client):
        return run_entry(name, [*self.ARGUMENTS.get(name, []), *(["--server-log", str(log)] if log else [])],
                         client)

    def test_the_live_pinned_server_binds_with_or_without_its_log(self):
        with tempfile.TemporaryDirectory() as directory:
            for name in self.ARGUMENTS:
                for log in (server_log(directory, 8101, self.PIN["checkpoint_identity"]), None):
                    with self.subTest(name, log=log):
                        code, value = self.score(name, log, self.client())
                        self.assertNotIn("refused", str(code))
                        bound = value["server"]["checkpoint"]
                        self.assertEqual((bound["checkpoint_identity"], bound["startup_nonce"]),
                                         (self.PIN["checkpoint_identity"], NONCE))
                        self.assertEqual(bound.get("server_log"), log and str(log))

    def test_a_stale_log_another_instance_or_no_identity_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            pinned = self.PIN["checkpoint_identity"]
            cases = {
                # An earlier process's log at the same address: same checkpoint, another nonce.
                "stale log at the same address": (server_log(directory, 8101, pinned, STALE_NONCE), self.client(),
                                                  "not the log of the live server"),
                # Another live instance at the address of a genuine pinned log.
                "another live instance": (server_log(directory, 8101, pinned),
                                          self.client(RL_IDENTITY, STALE_NONCE), "not the pinned"),
                "another live instance, no log": (None, self.client(RL_IDENTITY), "not the pinned"),
                "no live identity": (None, self.client(nonce=None), "no checkpoint identity"),
            }
            for name in self.ARGUMENTS:
                for label, (log, client, reason) in cases.items():
                    with self.subTest(name, case=label):
                        code, value = self.score(name, log, client)
                        self.assertIn(reason, str(code))
                        self.assertIsNone(value)

    def test_a_server_replaced_while_scoring_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            log = server_log(directory, 8101, self.PIN["checkpoint_identity"])
            replacements = {
                "same checkpoint, another process": {"checkpoint_identity": self.PIN["checkpoint_identity"],
                                                     "startup_nonce": STALE_NONCE},
                "another checkpoint": {"checkpoint_identity": RL_IDENTITY, "startup_nonce": STALE_NONCE},
                "no identity": {},
            }
            for name in self.ARGUMENTS:
                for label, replacement in replacements.items():
                    with self.subTest(name, case=label):
                        code, value = self.score(name, log, self.client(replacement=replacement))
                        self.assertIn("refused: the server at", str(code))
                        self.assertIsNone(value)


    def test_a_scored_response_from_another_process_is_refused(self):
        # /health reaches the bound process before and after; one scored
        # response comes from another (a proxy or shared listener).
        with tempfile.TemporaryDirectory() as directory:
            log = server_log(directory, 8101, self.PIN["checkpoint_identity"])
            interlopers = {
                "same checkpoint, another process": ({"checkpoint_identity": self.PIN["checkpoint_identity"],
                                                      "startup_nonce": STALE_NONCE}, "named another process"),
                "another checkpoint": ({"checkpoint_identity": RL_IDENTITY, "startup_nonce": STALE_NONCE},
                                       "named another process"),
                # A server built before the identity headers.
                "no identity headers": ({}, f"lack the {common.NONCE_HEADER} and {common.CHECKPOINT_HEADER} "
                                            "headers: the server predates per-response identity headers; "
                                            "rebuild it"),
            }
            for name in self.ARGUMENTS:
                for label, (interloper, reason) in interlopers.items():
                    with self.subTest(name, case=label):
                        code, value = self.score(name, log, self.client(interloper=interloper))
                        self.assertIn(reason, str(code))
                        self.assertIsNone(value)


class MimoLiveBindingTests(LiveBindingCases, unittest.TestCase):
    PIN = PIN
    ARGUMENTS = {"multimodal": [], "verify": [], "prefix": []}


class HelperTests(unittest.TestCase):
    def test_the_gate_pin_matches_the_repository_and_both_references(self):
        pin = common.checkpoint_pin()
        self.assertEqual(pin["pin"], json.loads((common.ROOT / pin["registry"]).read_text())["default"])
        common.pinned_reference("teacher", TEACHER)
        common.pinned_reference("media", MEDIA)
        for kind, path in (("teacher", TEACHER), ("media", MEDIA)):
            stated = json.loads(path.read_text())["provenance"]["checkpoint"]
            self.assertEqual({key: stated[key] for key in common.PINNED_CHECKPOINT_KEYS},
                             {key: pin[key] for key in common.PINNED_CHECKPOINT_KEYS})
        changed = copy.deepcopy(CONFIG)
        changed["qualification"]["checkpoint"]["revision"] = "0" * 40
        with self.assertRaisesRegex(ValueError, "differs from"):
            common.checkpoint_pin(changed)

    def test_a_log_must_name_one_checkpoint_for_the_server_at_the_url(self):
        with tempfile.TemporaryDirectory() as directory:
            log = server_log(directory, 8101)
            served = common.served_checkpoint(log, "http://127.0.0.1:8101")
            self.assertEqual((served["checkpoint_identity"], served["startup_nonce"]),
                             (PIN["checkpoint_identity"], NONCE))
            for name, text in (("no nonce", "".join(line for line in log.read_text().splitlines(True)
                                                    if "serving_identity" not in line)),
                               ("two nonces", log.read_text() + server_log(directory, 8101, nonce=STALE_NONCE)
                                .read_text())):
                with self.subTest(name), self.assertRaisesRegex(ValueError, "startup nonce"):
                    Path(directory, "nonce.log").write_text(text)
                    common.served_checkpoint(Path(directory, "nonce.log"), "http://127.0.0.1:8101")
            for nonce in ("", "E" * 32, "e" * 31):
                with self.subTest(nonce=nonce):
                    with self.assertRaisesRegex(ValueError, "startup nonce"):
                        common.served_checkpoint(server_log(directory, 8101, nonce=nonce), "http://127.0.0.1:8101")
                    with self.assertRaisesRegex(ValueError, "startup nonce"):
                        common.live_identity(ReplayServer("http://127.0.0.1:8101", nonce=nonce).identity())
            mixed = Path(directory, "mixed.log")
            mixed.write_text(log.read_text() + "startup weight_plan: "
                             + json.dumps({"checkpoint_identity": RL_IDENTITY}) + "\n")
            with self.assertRaisesRegex(ValueError, "single checkpoint"):
                common.served_checkpoint(mixed, "http://127.0.0.1:8101")


class ResponderTests(unittest.TestCase):
    def test_the_client_records_the_identity_headers_of_every_response(self):
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                status, nonce = {"/ok": (200, NONCE), "/refused": (400, STALE_NONCE)}.get(self.path, (200, None))
                self.send_response(status)
                if nonce:
                    self.send_header(common.NONCE_HEADER, nonce)
                    self.send_header(common.CHECKPOINT_HEADER, PIN["checkpoint_identity"])
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b"{}")

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            client = common.Client(f"http://127.0.0.1:{server.server_port}", timeout=5, retries=0)
            client.request("/ok")
            self.assertEqual(client.responders, {(PIN["checkpoint_identity"], NONCE)})
            with self.assertRaises(RuntimeError):
                client.request("/refused")
            client.request("/bare")
            self.assertEqual(client.responders, {(PIN["checkpoint_identity"], NONCE),
                                                 (PIN["checkpoint_identity"], STALE_NONCE), (None, None)})
        finally:
            server.shutdown()
            server.server_close()


class MultimodalBindingTests(unittest.TestCase):
    def run_main(self, log, client=live()):
        return run_entry("multimodal", [*(["--server-log", str(log)] if log else [])], client)

    def test_only_the_pinned_checkpoint_is_scored(self):
        with tempfile.TemporaryDirectory() as directory:
            code, value = self.run_main(server_log(directory, 8101))
            self.assertEqual(code, 0)
            check = next(check for check in CONFIG["checks"] if check["id"] == "multimodal")
            self.assertTrue(all(release_gate.judge(value, rule) for rule in check["rules"]))
            self.assertEqual(value["server"]["checkpoint"]["startup_nonce"], NONCE)
            code, value = self.run_main(None)
            self.assertEqual(code, 0)
            for log, client in ((server_log(directory, 8101, RL_IDENTITY), live()), (None, live(RL_IDENTITY))):
                code, value = self.run_main(log, client)
                self.assertIn("refused", str(code))
                self.assertIsNone(value)

class MixedApplicabilityTests(unittest.TestCase):
    def test_disabled_modalities_are_not_errors(self):
        for disabled in (("audio",), ("image", "video", "audio")):
            with self.subTest(disabled=disabled):
                rows = [{"id": kind, "kind": kind, "status": "not_applicable",
                         "reason": kind + " disabled", "passed": None} if kind in disabled else
                        {"id": kind, "kind": kind, "passed": True}
                        for kind in ("image", "video", "audio")]
                result = multimodal.applicable({}, rows)
                mixed = multimodal.applicable({"case_count": 5}, copy.deepcopy(rows))
                code, value = run_main(multimodal, ["multimodal.py", "--url", URL], live(),
                                      (("run", scoring(result)), ("run_mixed", scoring(mixed))))
                self.assertEqual(code, 0)
                self.assertEqual(value["errors"], [])
                self.assertEqual(value["not_applicable"], sorted(kind + " disabled" for kind in disabled))
                self.assertEqual(value["passed"], len(disabled) != 3)

    def test_mixed_not_applicable_reasons_are_merged(self):
        original = multimodal.applicable({}, [{"id": "image", "passed": True}])
        mixed = multimodal.applicable({}, [{"id": "image", "passed": True},
            {"id": "audio", "status": "not_applicable", "reason": "audio disabled", "passed": None}])
        code, value = run_main(multimodal, ["multimodal.py", "--url", URL], live(),
                              (("run", scoring(original)), ("run_mixed", scoring(mixed))))
        self.assertEqual(code, 0)
        self.assertEqual(value["errors"], [])
        self.assertEqual(value["not_applicable"], ["audio disabled"])


class VerifyBindingTests(unittest.TestCase):
    def test_gate_reports_need_records_of_the_pinned_checkpoint(self):
        digest = hashlib.sha256(TEACHER.read_bytes()).hexdigest()
        reference = json.loads(TEACHER.read_text())["texts"][0]
        start = reference["score_from"]
        positions = {str(start + i): {"logprob": logprob, "top": {"x": logprob}, "row": "decode"}
                     for i, logprob in enumerate(reference["exact"]["logprob"][:3])}
        record = {"arm": "ar", "k": 0, "concurrency": 1, "detokenized": {}, "prefill": {},
                  "reference_sha256": digest, "server": {"model": PIN["served_model_name"]},
                  "texts": [{"name": reference["name"], "passes": [{"positions": positions, "requests": [
                      {"start": start, "maximum": 3, "emitted": 3, "matched": 3}]}]}]}
        bound = {"checkpoint_identity": PIN["checkpoint_identity"], "startup_nonce": NONCE}
        servers = {
            "no checkpoint": {},
            "legacy, no nonce": {"checkpoint": {"checkpoint_identity": PIN["checkpoint_identity"]}},
            "rescored offline": {"checkpoint": bound, "rescored_from": "capture.json"},
            "bound live": {"checkpoint": bound},
        }
        with tempfile.TemporaryDirectory() as directory:
            args = argparse.Namespace(exact=None, ar=str(Path(directory, "ar.json")),
                                      verify=[str(Path(directory, "ar.json"))])
            class Scored(Exception):
                pass
            for label, server in servers.items():
                record["server"] = {"model": PIN["served_model_name"], **server}
                Path(directory, "ar.json").write_text(json.dumps(record))
                # A record that passes the binding goes on to be scored.
                with self.subTest(label), mock.patch.object(verify_exact, "compare", side_effect=Scored), \
                        self.assertRaises(SystemExit if label != "bound live" else Scored) as raised:
                    verify_exact.report(args)
                if label != "bound live":
                    self.assertIn("not bound live to the pinned checkpoint", str(raised.exception.code))


if __name__ == "__main__":
    unittest.main()
