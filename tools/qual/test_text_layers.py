"""CPU checks for the model-neutral text layer comparison (text_layers.py),
over each declaration that declares the check, run standalone: the check
reads its constants from the declaration itself, as the gate worker sets
them. It runs tapdiff and needs numpy (the runner's Python); it is skipped
without it."""

import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import text_layers as compare

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
try:
    import numpy as np
    from tools.qual.test_tapdiff import item, write_step
except ModuleNotFoundError:
    np = None
import test_tap_binding as bound_taps  # noqa: E402
from tools.refrun import keys, modes  # noqa: E402


def declarations():
    """Each declaration under models/ that declares `text_layers`."""
    return [path for path in sorted((compare.ROOT / "models").glob("*/qual*.json"))
            if "text_layers" in json.loads(path.read_text()).get("checks", {})]


def dtype(name):
    """A synthetic tap's dtype: U32 selected ids, else F32."""
    return "u32" if name.endswith(".ids") else "f32"


@unittest.skipIf(np is None, "needs numpy (set CHOPINFER_REFRUN_PYTHON)")
@unittest.skipIf(not declarations(), "no declaration declares text_layers")
class BoundComparisonTests(unittest.TestCase):
    """A dump is compared with the runner's taps only once `capture` bound it
    to a live server of the declaration's pinned checkpoint and both sides
    are reverified; each tap is held by its median row and as a whole, and
    the reference holds the declaration's model, each mode's adapter, lock
    and mode policies."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR"))
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        environment = patch.dict(os.environ, {"CHOPINFER_QUAL_CONFIG": str(declarations()[0])})
        environment.start()
        self.addCleanup(environment.stop)
        # The required taps, or three when the declaration requires none.
        self.taps = sorted(compare.required_taps()) or ["L00.attn.out", "L00.in.streams", "L00.out"]
        pin = compare.config()["qualification"]["checkpoint"]
        self.serving = bound_taps.SERVING | {"checkpoint_identity": pin["checkpoint_identity"]}
        self.tokens = list(range(100, 110))
        rng = np.random.default_rng(7)
        # Selected ids (sets of distinct ids) and dense rows.
        self.f32 = {name: (np.stack([rng.permutation(32)[:16] for _ in range(10)]).astype(np.uint32)
                           if name.endswith(".ids") else rng.standard_normal((10, 16)).astype(np.float32))
                    for name in self.taps}
        self.bf16 = {name: rows if name.endswith(".ids") else
                     (rows * (1 + 4e-3 * np.sin(np.arange(16)))).astype(np.float32)
                     for name, rows in self.f32.items()}
        self.run = self.reference()
        from qual_test_config import fixture_evidence
        pinned = patch.object(compare, "verify_input", side_effect=fixture_evidence)
        pinned.start()
        self.addCleanup(pinned.stop)
        self.log = bound_taps.server_log(self.root / "server.log", self.serving)

    def reference(self, lock=None, dequant=None, model=None, adapter=None, files=None, layout=None):
        """A runner reference: by default the declaration's layout, one run,
        or one run per mode with its own adapter. `layout` maps each mode to
        (its run directory under the reference, its adapter); `files` (the
        cache keys' adapter files) replaces every mode's, and `adapter`
        names another adapter in every run.json."""
        run = self.root / f"run-{len(list(self.root.glob('run-*')))}"
        layout = layout or {mode: (compare.mode_run(Path("."), mode).as_posix(), compare.adapter(mode))
                            for mode in ("f32", "bf16")}
        source_lock = lock or keys.checkpoint(keys.read_lock(compare.ROOT / compare.lock()), [])[
            "source_lock_sha256"]
        for mode, values in (("f32", self.f32), ("bf16", self.bf16)):
            directory, used = layout[mode]
            held = run / directory
            write_step(held / "taps" / mode / "step0000", [item("frozen0", 0, list(range(10)), self.tokens)],
                       {name: [(0, rows, dtype(name))] for name, rows in values.items()})
            policy = modes.policy(mode, (dequant or compare.declared("MODES"))[mode])
            key = {"model": model or compare.model(),
                   "checkpoint": {"source_lock_sha256": source_lock, "files_used": []},
                   "adapter": {"files": files if files is not None else
                               {path: keys.file_sha256(compare.ROOT / path) for path in {*keys.NUMERICS, used}}},
                   "mode": keys.mode(policy), "sequence": {"token_ids_sha256": keys.ids_sha256(self.tokens)}}
            (held / f"{mode}.json").write_text(json.dumps({"schema": "chopinfer.refrun.scores.v1", "policy": policy,
                                                          "entries": [{"name": "frozen0", "key": key}]}))
            (held / "run.json").write_text(json.dumps({"argv": ["run", "--adapter", adapter or used,
                                                                "--lock", compare.lock()]}))
        return run

    def adapters(self, value):
        """The declaration with ADAPTERS `value`, its other constants kept."""
        declared = compare.declared
        return patch.object(compare, "declared", lambda name: value if name == "ADAPTERS" else declared(name))

    def split(self):
        """ADAPTERS that give bf16 an adapter of its own: the declaration's
        where it declares them, else the runner's tiny adapter (any repo
        file; the fixtures compute nothing)."""
        declared = compare.declared("ADAPTERS")
        return declared if compare.mode_run(Path("."), "bf16") != Path(".") else {"bf16": "tools/refrun/tiny.py"}

    def capture(self, engine):
        taps = self.root / f"taps-{len(list(self.root.glob('taps-*')))}"
        taps.mkdir()
        out = self.root / f"dump-{len(list(self.root.glob('dump-*')))}"
        engine = {name: (rows, dtype(name)) for name, rows in engine.items()}
        fake = bound_taps.FakeServer(taps, self.log, engine, self.tokens, serving=self.serving)
        with bound_taps.serve(fake), \
             patch("sys.argv", ["text_layers", "capture", "--url", bound_taps.URL, "--server-log", str(self.log),
                                "--taps-dir", str(taps), "--reference", str(self.run), "--out", str(out)]), \
             contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()) as error:
            status = compare.main()
        return status, out, error.getvalue()

    def check(self, dump, reference=None):
        out = self.root / f"report-{len(list(self.root.glob('report-*')))}.json"
        candidate = bound_taps.FakeServer(self.root, self.log, {}, self.tokens, serving=self.serving)
        with bound_taps.serve(candidate), \
             patch("sys.argv", ["text_layers", "--dump", str(dump), "--reference", str(reference or self.run),
                                "--url", bound_taps.URL, "--json", str(out)]), \
             contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            status = compare.main()
        return status, json.loads(out.read_text())

    def bound(self, engine=None):
        status, dump, error = self.capture(engine or {name: rows if name.endswith(".ids") else
                                                      (rows * (1 + 1e-3)).astype(np.float32)
                                                      for name, rows in self.f32.items()})
        self.assertEqual(status, 0, error)
        return dump

    def test_a_bound_dump_within_its_floors_passes_tap_by_tap(self):
        status, report = self.check(self.bound())
        self.assertEqual((status, report["status"], report["first"]), (0, "match", None))
        self.assertEqual(sorted((row["tensor"], row["pass"]) for row in report["rows"]),
                         [(name, True) for name in self.taps])
        self.assertEqual(report["coverage"], {"compared": len(self.taps), "gated": len(self.taps)})
        self.assertEqual(report["binding"]["request"]["engine_request_id"], 1)

    def test_a_diverged_tap_is_named_first(self):
        engine = dict(self.f32)
        engine["L00.attn.out"] = -engine["L00.attn.out"]
        self.assertIn("L00.attn.out", self.taps)
        status, report = self.check(self.bound(engine))
        self.assertEqual((status, report["status"], report["first"]), (1, "diverged", "L00.attn.out"))

    def test_an_unbound_dump_counts_no_tap(self):
        dump = self.root / "loose"
        write_step(dump / "engine/step0000", [item(1, 0, list(range(10)), self.tokens)],
                   {name: [(0, rows, dtype(name))] for name, rows in self.f32.items()})
        status, report = self.check(dump)
        self.assertEqual((status, report["status"], report["rows"]), (3, "insufficient", []))

    def test_a_reference_of_another_model_lock_adapter_or_policy_counts_no_tap(self):
        dump = self.bound()
        other = {mode: {"f32": "exact" if dequant != "exact" else "rounded"}.get(mode, dequant)
                 for mode, dequant in compare.declared("MODES").items()}
        for reference in (self.reference(lock="0" * 64), self.reference(dequant=other),
                          self.reference(model="another"), self.reference(adapter="models/another/reference.py")):
            status, report = self.check(dump, reference)
            self.assertEqual((status, report["status"], report["rows"]), (3, "insufficient", []))

    def test_a_declared_router_has_tapdiff_judge_routes(self):
        dump = self.bound()
        declared = compare.declared

        def router(spec):
            return patch.object(compare, "declared", side_effect=lambda name: spec if name == "ROUTER"
                                else declared(name))
        with router({}):
            status, report = self.check(dump)
        self.assertEqual((status, report["routes"]), (0, None))
        with router({"score": "softmax", "top_k": 2}):
            status, report = self.check(dump)
        self.assertEqual((status, report["status"], report["routes"]["rule"]), (0, "match", "route-near-tie.v2"))
        # A bias input the declaration does not declare counts no tap.
        with router({"score": "sigmoid", "top_k": 2, "bias": "no_such_input"}):
            status, report = self.check(dump)
        self.assertEqual((status, report["status"], report["rows"]), (3, "insufficient", []))
        self.assertIn("does not declare", report["reason"])

    def test_standalone_runs_apply_the_declarations_constants(self):
        # The module keeps its defaults; the declaration's values apply.
        check = json.loads(declarations()[0].read_text())["checks"]["text_layers"]
        for name, entry in check.get("constants", {}).get("text_layers", {}).items():
            self.assertEqual(compare.declared(name), entry["value"], name)
        self.assertEqual(compare.MODES, {"f32": "exact", "bf16": "hf-bf16"})
        self.assertEqual(compare.ADAPTERS, {})
        self.assertEqual(compare.REQUIRED_TAPS, [])
        self.assertEqual(compare.ROUTER, {})
        # A reference of the module's default policy, where the declaration
        # declares another, is refused before capture.
        if compare.declared("MODES") != compare.MODES:
            self.run = self.reference(dequant=compare.MODES)
            status, _, error = self.capture(self.f32)
            self.assertEqual(status, 3)
            self.assertIn("policy", error)

    def test_a_reference_without_the_adapters_current_identity_counts_no_tap(self):
        dump = self.bound()
        files = {path: keys.file_sha256(compare.ROOT / path) for path in keys.NUMERICS}
        missing = self.reference(files=files)
        stale = self.reference(files={**files, compare.adapter(): "0" * 64})
        for reference in (missing, stale):
            status, report = self.check(dump, reference)
            self.assertEqual((status, report["status"], report["rows"]), (3, "insufficient", []))
            self.assertIn(compare.adapter(), report["reason"])
        # Capture refuses them too.
        for reference in (missing, stale):
            self.run = reference
            self.assertEqual(self.capture(self.f32)[0], 3)

    def test_one_adapter_for_every_mode_is_one_run(self):
        with self.adapters({}):
            self.assertEqual({compare.adapter(mode) for mode in ("f32", "bf16")},
                             {f"models/{compare.model()}/reference.py"})
            self.assertEqual(compare.mode_run(self.root, "bf16"), self.root)
            self.run = self.reference()
            status, report = self.check(self.bound())
            self.assertEqual((status, report["status"]), (0, "match"))
            # A per-mode reference is not the declared one-run layout.
            split = self.reference(layout={"f32": ("f32", compare.adapter("f32")),
                                           "bf16": ("bf16", compare.adapter("bf16"))})
            self.assertEqual(self.check(self.bound(), split)[1]["status"], "insufficient")

    def test_a_mode_with_its_own_adapter_is_bound_to_its_own_run(self):
        adapters = self.split()
        floor, plain = adapters["bf16"], f"models/{compare.model()}/reference.py"
        with self.adapters(adapters):
            self.assertEqual((compare.adapter("f32"), compare.adapter("bf16")), (plain, floor))
            self.assertEqual(compare.mode_run(self.root, "bf16"), self.root / "bf16")
            self.run = self.reference()
            # One run per mode; the f32 run holds no bf16 taps, so the
            # match below read the floor from the bf16 run.
            self.assertEqual({path.relative_to(self.run).as_posix() for path in self.run.glob("*/run.json")},
                             {"f32/run.json", "bf16/run.json"})
            self.assertEqual(json.loads((self.run / "bf16/run.json").read_text())["argv"][2], floor)
            status, report = self.check(self.bound())
            self.assertEqual((status, report["status"], report["first"]), (0, "match", None))
            files = {path: keys.file_sha256(compare.ROOT / path) for path in keys.NUMERICS}
            refused = {
                # One run of the plain adapter for both modes.
                "one run": self.reference(layout={"f32": (".", plain), "bf16": (".", plain)}),
                # The floor run without its adapter.
                "plain floor": self.reference(layout={"f32": ("f32", plain), "bf16": ("bf16", plain)}),
                # The adapters swapped.
                "swapped": self.reference(layout={"f32": ("f32", floor), "bf16": ("bf16", plain)}),
                # The floor's cache keys without its adapter's current digest.
                "stale floor": self.reference(layout={"f32": ("f32", plain), "bf16": ("bf16", floor)},
                                              files={**files, plain: keys.file_sha256(compare.ROOT / plain),
                                                     floor: "0" * 64}),
            }
            dump = self.bound()
            for label, reference in refused.items():
                status, report = self.check(dump, reference)
                self.assertEqual((status, report["status"], report["rows"]), (3, "insufficient", []), label)
            self.assertIn(floor, self.check(dump, refused["stale floor"])[1]["reason"])
            self.assertIn(floor, self.check(dump, refused["plain floor"])[1]["reason"])

    def test_adapters_of_an_undeclared_mode_count_no_tap(self):
        dump = self.bound()
        with self.adapters({"f64": f"models/{compare.model()}/reference.py"}):
            status, report = self.check(dump)
        self.assertEqual((status, report["status"], report["rows"]), (3, "insufficient", []))
        self.assertIn("f64", report["reason"])

    def test_a_capture_missing_a_required_tap_is_insufficient(self):
        if not compare.required_taps():
            self.skipTest("the declaration requires no taps")
        dense = [name for name in self.taps if not name.endswith((".ids", "router.weights"))]
        dropped = dense[len(dense) // 2]
        engine = {name: rows for name, rows in self.f32.items() if name != dropped}
        status, report = self.check(self.bound(engine))
        self.assertEqual((status, report["status"]), (3, "insufficient"))
        self.assertEqual(report["scope"]["missing"], [dropped])
        self.assertFalse(any(row["pass"] for row in report["rows"]))

    def test_without_required_taps_the_scope_is_the_intersection(self):
        engine = {name: rows for name, rows in self.f32.items() if name == "L00.attn.out"}
        dump = self.bound(engine)
        with patch.object(compare, "required_taps", lambda: set()):
            status, report = self.check(dump)
        self.assertEqual((status, report["status"], report["scope"]["kind"]), (0, "match", "intersection"))
        self.assertEqual(report["coverage"]["compared"], 1)

    def test_without_a_declaration_nothing_is_compared(self):
        dump = self.bound()
        with patch.dict(os.environ):
            del os.environ["CHOPINFER_QUAL_CONFIG"]
            status, report = self.check(dump)
        self.assertEqual((status, report["status"]), (3, "insufficient"))
        self.assertIn("CHOPINFER_QUAL_CONFIG", report["reason"])


class RunnerArmTests(unittest.TestCase):
    def test_the_layers_arm_selects_every_site_by_pattern(self):
        # CHOPINFER_TAPS_SITES takes comma separated site patterns
        # (crates/runtime-bundle-sdk/src/taps.rs); a bare word such as
        # "all" names no site, and the bundle refuses the server's start.
        self.assertEqual(compare.RUNNER_ENVIRONMENT["layers"]["CHOPINFER_TAPS_SITES"], "*")


if __name__ == "__main__":
    unittest.main()
