"""CPU checks for the Qwen3.8 activation dump comparison. It runs tapdiff
and needs numpy (the runner's Python); it is skipped without it."""

import array
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
import contextlib
import io
from unittest.mock import patch

import qwen38_layers as compare

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
try:
    import numpy as np
    from tools.qual.test_tapdiff import item, write_step
except ModuleNotFoundError:
    np = None
import declaration  # noqa: E402
import test_tap_binding as bound_taps  # noqa: E402
from tools.refrun import keys, modes  # noqa: E402


def bf16_bytes(values):
    wide = array.array("f", values).tobytes()
    return bytes(b for index in range(0, len(wide), 4) for b in wide[index + 2:index + 4])


def write_dump(directory, name, parts, cols, dtype="bf16"):
    with open(directory / "index.jsonl", "a") as index:
        for rank, values in enumerate(parts):
            raw = bf16_bytes(values) if dtype == "bf16" else array.array("f", values).tobytes()
            (directory / f"{name}.r{rank}.bin").write_bytes(raw)
            index.write(json.dumps({"name": name, "rank": rank, "rows": len(values) // cols,
                                    "cols": cols, "dtype": dtype}) + "\n")


TAPS = ("L00.in.streams", "L00.attn.out", "L00.out")


@unittest.skipIf(np is None, "needs numpy (set CHOPINFER_REFRUN_PYTHON)")
class BoundComparisonTests(unittest.TestCase):
    """A dump is compared with the runner's taps only once `capture` bound it
    to a live server of the pinned checkpoint and both sides are reverified;
    each tap is held by its median row and as a whole."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR"))
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.tokens = list(range(100, 110))
        rng = np.random.default_rng(7)
        self.f32 = {name: rng.standard_normal((10, 16)).astype(np.float32) for name in TAPS}
        self.bf16 = {name: (rows * (1 + 4e-3 * np.sin(np.arange(16)))).astype(np.float32)
                     for name, rows in self.f32.items()}
        self.run = self.reference()
        from qual_test_config import fixture_evidence
        pinned = patch.object(compare, "verify_input", side_effect=fixture_evidence)
        pinned.start()
        self.addCleanup(pinned.stop)
        self.log = bound_taps.server_log(self.root / "server.log")
        environment = patch.dict(os.environ, {"CHOPINFER_QUAL_CONFIG": str(declaration.named("qwen38"))})
        environment.start()
        self.addCleanup(environment.stop)

    def reference(self, lock=None, dequant="exact", tokens=None, bf16=None):
        run = self.root / f"run-{len(list(self.root.glob('run-*')))}"
        tokens = self.tokens if tokens is None else tokens
        for mode, values in (("f32", self.f32), ("bf16", self.bf16 if bf16 is None else bf16)):
            write_step(run / "taps" / mode / "step0000", [item("frozen0", 0, list(range(10)), tokens)],
                       {name: [(0, rows, "f32")] for name, rows in values.items()})
        source_lock = lock or keys.checkpoint(keys.read_lock(compare.ROOT / compare.LOCK), [])["source_lock_sha256"]
        adapter = {path: keys.file_sha256(compare.ROOT / path) for path in {*keys.NUMERICS, compare.ADAPTER}}
        for mode in ("f32", "bf16"):
            policy = modes.policy(mode, {"f32": dequant, "bf16": "hf-bf16"}[mode])
            key = {"model": "qwen38", "checkpoint": {"source_lock_sha256": source_lock, "files_used": []},
                   "adapter": {"files": adapter}, "mode": keys.mode(policy),
                   "sequence": {"token_ids_sha256": keys.ids_sha256(self.tokens)}}
            (run / f"{mode}.json").write_text(json.dumps({"schema": "chopinfer.refrun.scores.v1", "policy": policy,
                                                         "entries": [{"name": "frozen0", "key": key}]}))
        (run / "run.json").write_text(json.dumps({"argv": ["run", "--adapter", compare.ADAPTER,
                                                           "--lock", compare.LOCK]}))
        return run

    def capture(self, engine, **server):
        taps = self.root / f"taps-{len(list(self.root.glob('taps-*')))}"
        taps.mkdir()
        out = self.root / f"dump-{len(list(self.root.glob('dump-*')))}"
        fake = bound_taps.FakeServer(taps, self.log, engine, self.tokens, **server)
        with bound_taps.serve(fake), \
             patch("sys.argv", ["qwen38_layers", "capture", "--url", bound_taps.URL, "--server-log", str(self.log),
                                "--taps-dir", str(taps), "--reference", str(self.run), "--out", str(out)]), \
             contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()) as error:
            status = compare.main()
        return status, out, error.getvalue()

    def check(self, dump, reference=None, serving=bound_taps.SERVING):
        """The comparison against a live candidate with `serving` identity."""
        out = self.root / f"report-{len(list(self.root.glob('report-*')))}.json"
        candidate = bound_taps.FakeServer(self.root, self.log, {}, self.tokens, serving=serving)
        with bound_taps.serve(candidate), \
             patch("sys.argv", ["qwen38_layers", "--dump", str(dump), "--reference", str(reference or self.run),
                                "--url", bound_taps.URL, "--json", str(out)]), \
             contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            status = compare.main()
        return status, json.loads(out.read_text())

    def bound(self, engine=None):
        status, dump, error = self.capture(engine or {name: (rows * (1 + 1e-3)).astype(np.float32)
                                                      for name, rows in self.f32.items()})
        self.assertEqual(status, 0, error)
        return dump

    def test_a_bound_dump_within_its_floors_passes_tap_by_tap(self):
        status, report = self.check(self.bound())
        self.assertEqual((status, report["status"], report["first"]), (0, "match", None))
        self.assertEqual([(row["tensor"], row["pass"]) for row in report["rows"]], [(name, True) for name in TAPS])
        self.assertEqual(report["coverage"], {"compared": 3, "gated": 3})
        self.assertEqual(report["binding"]["server"]["startup_nonce"], bound_taps.SERVING["startup_nonce"])
        self.assertEqual(report["binding"]["request"]["engine_request_id"], 1)
        self.assertTrue(all(row["aggregate"]["passed"] for row in report["rows"]))

    def test_a_diverged_tap_is_named_first(self):
        engine = dict(self.f32)
        engine["L00.attn.out"] = -engine["L00.attn.out"]
        status, report = self.check(self.bound(engine))
        self.assertEqual((status, report["status"], report["first"]), (1, "diverged", "L00.attn.out"))

    def test_a_hundredfold_final_row_fails_its_tap_as_a_whole(self):
        # The median row is untouched; the tap's relative L2 is about 30.
        engine = {name: rows.copy() for name, rows in self.f32.items()}
        for rows in engine.values():
            rows[-1] *= 100
        status, report = self.check(self.bound(engine))
        self.assertEqual((status, report["status"], report["first"]), (1, "diverged", TAPS[0]))
        row = report["rows"][0]
        self.assertEqual(row["e"]["median"], 0.0)
        self.assertGreater(row["aggregate"]["e"], 25)
        self.assertFalse(any(row["pass"] for row in report["rows"]))

    def test_an_unbound_dump_counts_no_tap(self):
        dump = self.root / "loose"
        write_step(dump / "engine/step0000", [item(1, 0, list(range(10)), self.tokens)],
                   {name: [(0, rows, "f32")] for name, rows in self.f32.items()})
        status, report = self.check(dump)
        self.assertEqual((status, report["status"], report["rows"], report["coverage"]["gated"]),
                         (3, "insufficient", [], 0))
        # Without a live candidate a bound dump counts nothing either.
        out = self.root / "report-unlive.json"
        with patch("sys.argv", ["qwen38_layers", "--dump", str(self.bound()), "--reference", str(self.run),
                                "--json", str(out)]), \
             contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(compare.main(), 3)
        self.assertIn("live candidate", json.loads(out.read_text())["reason"])

    def test_a_candidate_other_than_the_dumps_process_counts_no_tap(self):
        dump = self.bound()
        for key, value in {"startup_nonce": "b" * 32, "package_identity": "4" * 64,
                           "metadata_identity": "5" * 64, "source_commit": "6" * 40,
                           "profile": {"name": None, "tensor_parallel_size": 2}}.items():
            with self.subTest(key=key):
                status, report = self.check(dump, serving=bound_taps.SERVING | {key: value})
                self.assertEqual((status, report["status"], report["coverage"]["gated"]), (3, "insufficient", 0))
                self.assertIn(key, report["reason"])

    def test_a_changed_or_linked_step_file_unbinds_the_dump(self):
        dump = self.bound()
        values = dump / "engine/step0000" / "L00.out.r0.bin"
        data = bytearray(values.read_bytes())
        data[0] ^= 1
        values.write_bytes(bytes(data))
        self.assertEqual(self.check(dump)[1]["status"], "insufficient")
        dump = self.bound()
        target = dump / "engine/step0000" / "L00.out.r0.bin"
        moved = self.root / "elsewhere.bin"
        target.rename(moved)
        target.symlink_to(moved)
        status, report = self.check(dump)
        self.assertEqual((status, report["status"]), (3, "insufficient"))
        self.assertIn("not a regular file", report["reason"])

    def test_a_reference_of_another_checkpoint_or_policy_counts_no_tap(self):
        dump = self.bound()
        for reference in (self.reference(lock="0" * 64), self.reference(dequant="rounded"),
                          self.reference(tokens=[1] * 10)):
            status, report = self.check(dump, reference)
            self.assertEqual((status, report["status"], report["rows"]), (3, "insufficient", []))
        # A run of the same pinned inputs with other taps is not the one the
        # dump was captured against.
        status, report = self.check(dump, self.reference(bf16={name: rows * 1.01 for name, rows in self.bf16.items()}))
        self.assertEqual(status, 3)
        self.assertIn("captured against", report["reason"])

    def test_capture_refuses_another_checkpoint_cached_rows_or_extra_steps(self):
        engine = self.f32
        other = bound_taps.SERVING | {"checkpoint_identity": "0" * 64}
        for server in ({"serving": other}, {"cached": 4}, {"steps": 2}):
            with self.subTest(server=server):
                status, out, error = self.capture(engine, **server)
                self.assertEqual(status, 3, server)
        # The two-step prefill's capture never qualifies.
        status, report = self.check(out)
        self.assertEqual((status, report["status"]), (3, "insufficient"))


if __name__ == "__main__":
    unittest.main()
