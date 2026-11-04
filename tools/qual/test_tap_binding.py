"""tap_binding: a tap step counts only when its own metadata names the live
candidate process (startup nonce) and the fresh request (engine request
ID) that wrote it, and the candidate's whole serving identity is the
capture's. Synthetic servers and steps, written without numpy."""

import contextlib
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import common  # noqa: E402
import declaration  # noqa: E402
import experimental_determinism  # noqa: E402
import tap_binding  # noqa: E402

PIN = declaration.load(declaration.path("qwen38"))["qualification"]["checkpoint"]
URL = "http://127.0.0.1:39001"
SERVING = {"bundle_id": "qwen38.flash.v2", "checkpoint_identity": PIN["checkpoint_identity"],
           "startup_nonce": "a" * 32, "package_identity": "1" * 64, "metadata_identity": "2" * 64,
           "source_commit": "3" * 40, "version": "0.1.0", "profile": {"name": None, "tensor_parallel_size": 1}}
# A complete option map at its qualified defaults, as /health reports it.
OPTIONS = {"model_options": {"prefill_chunk_tokens": 8192, "fast_route": True}, "model_options_qualified": True}


def item(request, start, positions, tokens):
    """A step item of `request`'s rows, as test_tapdiff.item makes one."""
    return {"request": request, "start": start, "rows": len(positions), "prefix": positions[0],
            "positions": positions, "tokens": tokens}


def write_step(directory, items, tensors, encoder=None, producer=None):
    """A step directory as test_tapdiff.write_step writes one: `tensors` maps
    a stored name to its per-rank (first_row, rows, dtype) parts. Rows given
    as lists of F32 rows are written here, without numpy; numpy arrays (from
    the tap gates' tests, which need numpy) go to test_tapdiff's writer."""
    if any(not isinstance(rows, list) for parts in tensors.values() for _, rows, _ in parts):
        from tools.qual.test_tapdiff import write_step as write_arrays
        return write_arrays(directory, items, tensors, encoder=encoder, producer=producer)
    directory.mkdir(parents=True)
    lines = []
    for name, parts in tensors.items():
        for rank, (first, rows, dtype) in enumerate(parts):
            assert dtype == "f32", dtype
            (directory / f"{name}.r{rank}.bin").write_bytes(b"".join(struct.pack(f"<{len(row)}f", *row)
                                                                     for row in rows))
            line = {"name": name, "rank": rank, "first_row": first, "rows": len(rows), "cols": len(rows[0]),
                    "dtype": dtype}
            lines.append(line | {"axis": "encoder"} if name in (encoder or {}) else line)
    capture = {"schema": "chopinfer.taps.v1", "ranks": 1,
               "expected": [{"name": line["name"], "rank": line["rank"]} for line in lines]} \
        | ({"producer": {"startup_nonce": producer}} if producer else {})
    (directory / "capture.json").write_text(json.dumps(capture))
    (directory / "items.json").write_text(json.dumps({"items": items, **({"encoder": encoder} if encoder else {})}))
    (directory / "index.jsonl").write_text("".join(json.dumps(line) + "\n" for line in lines))
    (directory / "completion.json").write_text(json.dumps({"completion": "full", "reason": None}))


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
    steps it returns instead. Each request reports its token prompt's
    length (else len(tokens)) as its prompt tokens and `cached` cached
    prompt tokens (the next of a list per request), or with a `prefix_cache`
    granule G, the whole G-token granules its token prompt shares with an
    earlier prompt under its cache salt, as a server's prefix cache would."""

    def __init__(self, taps, log, tensors, tokens, serving=SERVING, cached=0, steps=1, first_request=1,
                 during=None, steps_of=None, prefix_cache=None):
        self.url, self.taps, self.log, self.tensors, self.tokens = URL, taps, log, tensors, tokens
        self.serving, self.cached, self.steps, self.next, self.during = serving, cached, steps, first_request, during
        self.steps_of = steps_of or (lambda engine, body: [
            ([item(engine, 0, list(range(len(self.tokens))), self.tokens)], self.tensors, None)] * self.steps)
        self.responders, self.prefix_cache, self.prompts, self.bodies = set(), prefix_cache, {}, []

    def usage(self, body):
        """(prompt tokens, cached prompt tokens) the request reports."""
        prompt = body["prompt"] if isinstance(body.get("prompt"), list) else self.tokens
        if self.prefix_cache is None:
            return len(prompt), self.cached.pop(0) if isinstance(self.cached, list) else self.cached
        shared = 0
        for earlier in self.prompts.get(body["cache_salt"], []):
            common_prefix = next((at for at, (a, b) in enumerate(zip(prompt, earlier)) if a != b),
                                 min(len(prompt), len(earlier)))
            shared = max(shared, min(common_prefix, len(prompt) - 1) // self.prefix_cache * self.prefix_cache)
        self.prompts.setdefault(body["cache_salt"], []).append(prompt)
        return len(prompt), shared

    def identity(self, model=None):
        self.responders.add((self.serving["checkpoint_identity"], self.serving["startup_nonce"]))
        return {"url": self.url, "model": model, "models": [model], "health": {"serving": self.serving}}

    def request(self, path, body, headers=None):
        assert body["cache_salt"] and headers["X-Client-Request-Id"]
        self.responders.add((self.serving["checkpoint_identity"], self.serving["startup_nonce"]))
        self.bodies.append(body)
        prompt_tokens, cached = self.usage(body)
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
        return {"id": f"cmpl_{engine}", "usage": {"prompt_tokens": prompt_tokens,
                                                   "prompt_tokens_details": {"cached_tokens": cached}}}, 0.1


def serve(fake):
    """Patches common.Client to answer from `fake`."""
    return patch.object(common, "Client", lambda url, retries=0, timeout=None: fake)


class TapBindingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR"))
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = declaration.load(declaration.path("qwen38"))
        self.tokens = list(range(100, 110))
        self.tensors = {"L00.out": [[1.0] * 16 for _ in self.tokens]}

    def capture(self, name="cap", experimental_pin=None, **server):
        taps = self.root / f"taps-{name}"
        taps.mkdir()
        log = server_log(self.root / f"{name}.log", server.get("serving", SERVING))
        fake = FakeServer(taps, log, self.tensors, self.tokens, **server)
        pinned = {"experimental_pin": experimental_pin} if experimental_pin else {}
        with serve(fake):
            return tap_binding.capture(URL, log, taps, [("p", "/v1/completions", {"prompt": self.tokens})],
                                       self.root / name, self.config, **pinned)

    def pair(self, name, requests, **server):
        """(fake server, manifest) of a capture of `requests`, each (name,
        prompt tokens, primer or None)."""
        taps = self.root / f"taps-{name}"
        taps.mkdir()
        log = server_log(self.root / f"{name}.log", server.get("serving", SERVING))
        fake = FakeServer(taps, log, self.tensors, self.tokens, **server)
        sends = [(request, "/v1/completions", {"prompt": prompt}, *([primer] if primer else []))
                 for request, prompt, primer in requests]
        with serve(fake):
            return fake, tap_binding.capture(URL, log, taps, sends, self.root / name, self.config)

    def pairs(self):
        """A primer of 8 tokens, a cold request and the primer's continuation."""
        return [("primer", self.tokens[:8], None), ("cold", self.tokens, None), ("hit", self.tokens, "primer")]

    def sealed(self, options, serving=SERVING):
        """A sealed experimental pin of `options` for the process `serving`
        names, as determinism.py --experimental-pin reads one."""
        path = self.root / "pin.json"
        path.write_text(json.dumps({"model_options": options, "checkpoint_identity": serving["checkpoint_identity"],
                                    "startup_nonce": serving["startup_nonce"]}))
        path.chmod(0o600)
        return experimental_determinism.Pin.load(path, hashlib.sha256(path.read_bytes()).hexdigest())

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

    def test_a_capture_keeps_the_complete_option_map_beside_its_identity(self):
        manifest = self.capture(serving=SERVING | OPTIONS)
        self.assertEqual(set(tap_binding.OPTIONS), set(OPTIONS))
        self.assertEqual(manifest["server"], SERVING | OPTIONS)
        self.assertNotIn(experimental_determinism.MARKER, manifest)
        # The tap gates admit it as before.
        self.assertEqual(self.admit(self.root / "cap", SERVING | OPTIONS)["server"], SERVING | OPTIONS)

    def test_non_default_options_need_a_sealed_pin_and_make_a_capture_no_gate_admits(self):
        options = dict(OPTIONS["model_options"], fast_route=False)
        serving = SERVING | {"model_options": options, "model_options_qualified": False}
        # Unpinned, non-default or incompletely reported options are refused.
        with self.assertRaisesRegex(ValueError, "non-default, unqualified"):
            self.capture("unpinned", serving=serving)
        with self.assertRaisesRegex(ValueError, "incompletely reported"):
            self.capture("partial", serving=SERVING | {"model_options": OPTIONS["model_options"]})
        pin = self.sealed(options)
        manifest = self.capture(serving=serving, experimental_pin=pin)
        self.assertEqual(manifest["server"], serving)
        self.assertEqual(manifest[experimental_determinism.MARKER], pin.marker())
        self.assertIs(manifest["qualified"], False)
        self.assertEqual(json.loads((self.root / "cap" / tap_binding.MANIFEST).read_text()), manifest)
        # No tap gate admits it, even against a candidate of its identity
        # that serves the qualified defaults.
        with self.assertRaisesRegex(tap_binding.Unbound, "experimental"):
            self.admit(self.root / "cap", SERVING | OPTIONS)
        # The pin names exactly the options, checkpoint and process served.
        with self.assertRaisesRegex(ValueError, "differ from the sealed pin"):
            self.capture("other", serving=serving, experimental_pin=self.sealed(OPTIONS["model_options"]))
        with self.assertRaisesRegex(ValueError, "differ from the sealed pin"):
            self.capture("restarted", serving=serving | {"startup_nonce": "b" * 32},
                         experimental_pin=self.sealed(options))
        with self.assertRaisesRegex(ValueError, "explicitly unqualified"):
            self.capture("qualified", serving=SERVING | OPTIONS,
                         experimental_pin=self.sealed(OPTIONS["model_options"]))

    def test_an_offline_reader_keeps_a_captures_experimental_status(self):
        self.assertIsNone(tap_binding.experimental_marker(self.capture("defaults", serving=SERVING | OPTIONS)))
        options = dict(OPTIONS["model_options"], fast_route=False)
        serving = SERVING | {"model_options": options, "model_options_qualified": False}
        manifest = self.capture(serving=serving, experimental_pin=self.sealed(options))
        self.assertEqual(tap_binding.experimental_marker(manifest), manifest[experimental_determinism.MARKER])
        marker = experimental_determinism.MARKER

        def changed(change):
            value = copy.deepcopy(manifest)
            change(value)
            return value
        for name, value, error in (
                # Without its pin and mark, the capture still reports unqualified options.
                ("unmarked", changed(lambda m: [m.pop(marker), m.pop("qualified")]),
                 "no experimental pin, and no qualified default options"),
                ("marked", changed(lambda m: m.pop(marker)), "no experimental pin, and no qualified default options"),
                ("legacy", changed(lambda m: [m.pop(marker), m.pop("qualified"),
                                              [m["server"].pop(key) for key in tap_binding.OPTIONS]]),
                 "no experimental pin, and no qualified default options"),
                ("no canonical digest", changed(lambda m: m[marker].pop("pin_sha256")),
                 "does not hold its canonical digest"),
                ("no file digest", changed(lambda m: m[marker].pop("file_sha256")), "records no sealed file digest"),
                ("short file digest", changed(lambda m: m[marker].update(file_sha256="0" * 63)),
                 "records no sealed file digest"),
                ("edited pin", changed(lambda m: m[marker]["pin"]["model_options"].update(fast_route=True)),
                 "does not hold its canonical digest"),
                ("invalid pin", changed(lambda m: m[marker]["pin"].pop("startup_nonce")),
                 "experimental pin needs the complete options, checkpoint and startup nonce"),
                ("qualified", changed(lambda m: m.update(qualified=True)),
                 "does not name its recorded options, checkpoint and process")):
            with self.subTest(name), self.assertRaisesRegex(tap_binding.Unbound, error):
                tap_binding.experimental_marker(value)

    def test_the_command_takes_a_sealed_pin_only_with_experimental(self):
        pin = self.sealed(OPTIONS["model_options"])
        body = self.root / "body.json"
        body.write_text(json.dumps({"prompt": self.tokens}))
        passed = []

        def capture(*args, experimental_pin=None, **kwargs):
            passed.append(experimental_pin)
            return {"requests": []}
        argv = ["tap_binding", "capture", "--url", URL, "--server-log", str(self.root / "server.log"),
                "--taps-dir", str(self.root), "--out", str(self.root / "out"),
                "--request", f"p=/v1/completions={body}"]
        digest = hashlib.sha256((self.root / "pin.json").read_bytes()).hexdigest()
        sealed = ["--experimental-pin", str(self.root / "pin.json"), "--experimental-pin-sha256", digest]
        with patch.object(tap_binding, "capture", capture), \
                patch.object(common, "qualification_config", lambda: self.config):
            with patch("sys.argv", argv + ["--experimental", *sealed]), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(tap_binding.main(), 0)
            self.assertEqual(passed.pop().marker(), pin.marker())
            with patch("sys.argv", argv), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(tap_binding.main(), 0)
            self.assertIsNone(passed.pop())
            with patch("sys.argv", argv + sealed), contextlib.redirect_stderr(io.StringIO()) as error:
                self.assertEqual(tap_binding.main(), 3)
            self.assertIn("require --experimental", error.getvalue())
            self.assertEqual(passed, [])

    def test_a_prefix_hit_pair_shares_one_fresh_salt_and_binds_each_step_to_its_own_request(self):
        # The fake caches by salt, so the continuation hits only under its
        # primer's salt.
        fake, manifest = self.pair("pair", self.pairs(), prefix_cache=4)
        records = manifest["requests"]
        self.assertEqual([(r["name"], r["primer"], r["prompt_tokens"], r["cached_tokens"]) for r in records],
                         [("primer", None, 8, 0), ("cold", None, 10, 0), ("hit", "primer", 10, 8)])
        self.assertEqual([body["cache_salt"] for body in fake.bodies], [r["cache_salt"] for r in records])
        self.assertEqual(records[2]["cache_salt"], records[0]["cache_salt"])
        self.assertNotEqual(records[1]["cache_salt"], records[0]["cache_salt"])
        self.assertEqual([(r["engine_request_id"], r["steps"]) for r in records],
                         [(1, ["step0000"]), (2, ["step0001"]), (3, ["step0002"])])
        tap_binding.check_requests(records)
        self.assertEqual(self.admit(self.root / "pair")["requests"], records)

    def test_each_request_records_the_digest_of_the_body_it_was_given(self):
        fake, manifest = self.pair("digests", self.pairs(), prefix_cache=4)

        def digest(prompt):
            return hashlib.sha256(json.dumps({"prompt": prompt}, sort_keys=True, separators=(",", ":"))
                                  .encode()).hexdigest()
        # Not the model and cache salt the capture adds.
        self.assertEqual([record["body_sha256"] for record in manifest["requests"]],
                         [digest(self.tokens[:8]), digest(self.tokens), digest(self.tokens)])
        self.assertEqual({tuple(sorted(body)) for body in fake.bodies}, {("cache_salt", "model", "prompt")})

    def test_a_pair_without_a_hit_within_its_primer_or_a_cold_request_with_a_hit_is_refused(self):
        for name, cached, error in (
                ("miss", [0, 0, 0], r"hit: a prefix hit requires reported 0 < cached_tokens <= 8 "
                                    r"\(the prompt tokens of its primer primer\), got 0"),
                ("beyond", [0, 0, 9], "got 9"),
                ("warm-primer", [4, 0, 8], "primer: the server reused 4 cached prompt tokens"),
                ("warm-cold", [0, 4, 8], "cold: the server reused 4 cached prompt tokens")):
            with self.subTest(name), self.assertRaisesRegex(tap_binding.Unbound, error):
                self.pair(name, self.pairs(), cached=cached)

    def test_a_continuation_reuses_its_primers_salt_and_names_one_earlier_cold_primer(self):
        _, manifest = self.pair("pair", self.pairs(), prefix_cache=4)
        for at, salt_of, error in ((2, 1, "hit: does not reuse the cache salt of its primer primer"),
                                   (2, None, "hit: records no cache salt"),
                                   (1, 0, "cold: shares its cache salt with an earlier request")):
            records = copy.deepcopy(manifest["requests"])
            records[at]["cache_salt"] = None if salt_of is None else records[salt_of]["cache_salt"]
            with self.subTest(error), self.assertRaisesRegex(tap_binding.Unbound, error):
                tap_binding.check_requests(records)
        primer, hit = self.pairs()[0], self.pairs()[2]
        for name, requests, error in (
                ("late", [hit, primer], "hit: names no single earlier cold request primer as its primer"),
                ("chained", [primer, hit, ("again", self.tokens, "hit")],
                 "again: names no single earlier cold request hit as its primer"),
                ("twice", [primer, hit, ("again", self.tokens, "primer")], "again: primer already primes hit")):
            with self.subTest(name), self.assertRaisesRegex(tap_binding.Unbound, error):
                self.pair(name, requests, prefix_cache=4)

    def test_each_step_of_a_continuation_must_name_its_own_request(self):
        def steps_of(engine, body):
            # The continuation's (engine request 2) step names its primer's.
            named = engine - 1 if engine == 2 else engine
            return [([item(named, 0, list(range(len(body["prompt"]))), body["prompt"])], self.tensors, None)]
        with self.assertRaisesRegex(tap_binding.Unbound, r"engine requests \[1\], not \[2\]"):
            self.pair("borrowed", [self.pairs()[0], self.pairs()[2]], prefix_cache=4, steps_of=steps_of)

    def test_engine_request_ids_are_integers_wherever_they_are_read(self):
        # true equals 1 in Python; it names no engine request.
        boolean = lambda engine, body: [([item(True, 0, list(range(len(self.tokens))), self.tokens)],  # noqa: E731
                                         self.tensors, None)]
        with self.assertRaisesRegex(tap_binding.Unbound, "names engine request True, not an integer ID"):
            self.capture("boolean-item", steps_of=boolean)
        traced = FakeServer.request

        def request(fake, path, body, headers=None):
            # The trace record names the request as true.
            response = traced(fake, path, body, headers)
            fake.log.write_text(fake.log.read_text().replace('"engine_request_id": 1}', '"engine_request_id": true}'))
            return response
        with patch.object(FakeServer, "request", request), \
                self.assertRaisesRegex(tap_binding.Unbound, "names engine request True, not an integer ID"):
            self.capture("boolean-trace")
        # A manifest that records true for its steps' request 1, read by a
        # gate or offline.
        self.capture(serving=SERVING | OPTIONS)
        path = self.root / "cap" / tap_binding.MANIFEST
        manifest = json.loads(path.read_text())
        manifest["requests"][0]["engine_request_id"] = True
        path.write_text(json.dumps(manifest))
        for read in (lambda: self.admit(self.root / "cap", SERVING | OPTIONS),
                     lambda: tap_binding.recorded(self.root / "cap")):
            with self.assertRaisesRegex(tap_binding.Unbound, "names engine request True, not an integer ID"):
                read()

    def test_the_command_sends_prefix_hits_in_order_after_their_primers(self):
        bodies = {}
        for name, prompt in (("short", self.tokens[:8]), ("long", self.tokens)):
            bodies[name] = self.root / f"{name}.json"
            bodies[name].write_text(json.dumps({"prompt": prompt}))
        sent = []

        def capture(url, log, taps, requests, out, config, experimental_pin=None):
            sent.extend(requests)
            return {"requests": []}
        argv = ["tap_binding", "capture", "--url", URL, "--server-log", str(self.root / "server.log"),
                "--taps-dir", str(self.root), "--out", str(self.root / "out"),
                "--request", f"primer=/v1/completions={bodies['short']}",
                "--prefix-hit", f"hit=primer=/v1/completions={bodies['long']}",
                "--request", f"cold=/v1/completions={bodies['long']}"]
        with patch.object(tap_binding, "capture", capture), \
                patch.object(common, "qualification_config", lambda: self.config), \
                patch("sys.argv", argv), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(tap_binding.main(), 0)
        self.assertEqual(sent, [("primer", "/v1/completions", {"prompt": self.tokens[:8]}),
                                ("hit", "/v1/completions", {"prompt": self.tokens}, "primer"),
                                ("cold", "/v1/completions", {"prompt": self.tokens})])


if __name__ == "__main__":
    unittest.main()
