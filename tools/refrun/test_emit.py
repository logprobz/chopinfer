"""Fixture emission from runner output directories. Runs without torch."""

import json
from pathlib import Path
import tempfile
import unittest

from tools.model_sdk.diagnostic import canonical
from tools.refrun import cache, emit, run, workload
from tools.refrun.test_cache import ANSWER, TEXTS, Tree

LOCK = "models/demo/checkpoint-files.json"


def skeleton():
    return {"schema": "demo.teacher-reference.v1", "provenance": {"gate_variant": "exact"},
            "texts": [{"name": name, "category": "prose", "prompt_token_ids": ids, "score_from": 1}
                      for name, ids in TEXTS.items()]}


class EmitTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.base = Path(self.directory.name)
        self.tree = Tree(self.base / "repo")
        self.store = cache.Store(self.base / "cache")
        self.output = self.tree.root / "fixtures/demo/reference.json"
        self.output.parent.mkdir(parents=True)

    def tearDown(self):
        self.directory.cleanup()

    def write(self, directory, label, document):
        (self.base / directory).mkdir(exist_ok=True)
        (self.base / directory / f"{label}.json").write_bytes(canonical({"schema": run.SCHEMA, **document}))
        return self.base / directory

    def emit(self, directories, modes=("f32=exact", "bf16"), **kwargs):
        return emit.emit(skeleton(), directories, self.tree.workload(**kwargs), self.output, LOCK, "base",
                         list(modes), store=self.store)

    def test_a_mode_split_over_jobs_is_joined_in_workload_order(self):
        f32, bf16 = self.tree.run("f32"), self.tree.run("bf16", offset=0.1)
        first = self.write("a", "f32", f32)
        # The second job scored beta, then alpha again; both are joined once.
        second = self.write("b", "f32", {**f32, "entries": f32["entries"][::-1]})
        self.write("a", "bf16", {**bf16, "entries": bf16["entries"][:1]})
        self.write("b", "bf16", {**bf16, "entries": bf16["entries"][1:]})
        report = self.emit([first, second])
        value = json.loads(self.output.read_text())
        self.assertEqual(report["entries"], {"f32": 2, "bf16": 2})
        self.assertEqual(report["sha256"], cache.keys.sha256(self.output.read_bytes()))
        modes = value["binding"]["modes"]
        self.assertEqual((modes["f32"]["variant"], modes["bf16"]["variant"]), ("exact", "bf16"))
        self.assertEqual([name for name, _, _ in modes["f32"]["entries"]], list(TEXTS))
        alpha = value["texts"][0]
        self.assertEqual(alpha["exact"]["logprob"], f32["entries"][0]["scores"]["logprob"])
        self.assertEqual(set(alpha["exact"]), set(emit.FIELDS))
        self.assertEqual(alpha["bf16"]["logprob"], bf16["entries"][0]["scores"]["logprob"])
        canonical_workload = self.output.with_name("reference.workload.json")
        self.assertEqual(Path(report["workload"]), canonical_workload.resolve())
        self.assertIsNotNone(self.store.refset(report["refset_sha256"]))
        for document in (f32, bf16):
            for entry in document["entries"]:
                self.store.put(entry["key"], entry["scores"])
        result = cache.verify(value, self.tree.root, self.tree.checkpoint, self.store, canonical_workload)
        self.assertEqual(result["cache"]["missing"], 0)

    def test_top_k_rows_are_cut(self):
        f32 = self.tree.run("f32")
        directory = self.write("a", "f32", f32)
        emit.emit(skeleton(), [directory], self.tree.workload(), self.output, LOCK, "base", ["f32"], topk=1)
        rows = json.loads(self.output.read_text())["texts"][0]["f32"]["topk_ids"]
        self.assertEqual(rows, [row[:1] for row in f32["entries"][0]["scores"]["topk_ids"]])

    def test_a_generated_answer_is_stated_with_its_step_scores(self):
        greedy = {"answer": ANSWER}
        directory = self.write("a", "f32", self.tree.run("f32", greedy=greedy))
        self.write("a", "bf16", self.tree.run("bf16", greedy={**greedy, "entry": False}))
        emit.emit(skeleton(), [directory], self.tree.workload(greedy=True), self.output, LOCK, "base",
                  ["f32", "bf16"], greedy_fields=("logprob",), store=self.store)
        value = json.loads(self.output.read_text())
        beta = value["texts"][1]
        self.assertEqual(beta["answer_token_ids"], ANSWER)
        self.assertEqual(set(beta["greedy"]), {"answer", "logprob"})
        cache.verify(value, self.tree.root, workload=self.output.with_name("reference.workload.json"))

    def test_refusals(self):
        f32 = self.tree.run("f32")
        changed = json.loads(json.dumps(f32))
        changed["entries"][0]["scores"]["logprob"][0] = 0.0
        cases = {
            "no forced entry of \\['beta'\\]": ([self.write("short", "f32", {**f32, "entries": f32["entries"][:1]})],
                                               ["f32"]),
            "scored differently": ([self.write("a", "f32", f32), self.write("b", "f32", changed)], ["f32"]),
            "no run of \\['f64'\\]": ([self.write("c", "f32", f32)], ["f32", "f64"]),
            "names a variant twice": ([self.write("d", "f32", f32)], ["f32=x", "bf16=x"]),
        }
        for message, (directories, modes) in cases.items():
            with self.subTest(message), self.assertRaisesRegex(ValueError, message):
                emit.emit(skeleton(), directories, self.tree.workload(), self.output, LOCK, "base", modes)
        partial = skeleton()
        partial["texts"].pop()
        with self.assertRaisesRegex(ValueError, "beta is not an item of the skeleton"):
            emit.emit(partial, [self.write("e", "f32", f32)], self.tree.workload(), self.output, LOCK, "base", ["f32"])

    def media_workload(self, source):
        """A one-sequence workload at `source` with a pixel and a grid tensor under media/."""
        (source.parent / "media").mkdir(parents=True, exist_ok=True)
        (source.parent / "media/pixels.bin").write_bytes(bytes(8))
        (source.parent / "media/grid.i64").write_bytes(bytes(24))
        source.write_text(json.dumps({"schema": workload.SCHEMA, "model": "demo", "sequences": [
            {"name": "image", "category": "image", "token_ids": [1, 2, 3], "score_from": 1,
             "media": [{"kind": "image", "tensor": {"path": "media/pixels.bin", "dtype": "f32", "shape": [2]},
                        "grid": {"path": "media/grid.i64", "dtype": "i64", "shape": [1, 3]},
                        "processor": {"code_sha256": "a" * 64, "config_sha256": "b" * 64}}]}]}))
        return source

    def test_a_workload_beside_the_fixture_keeps_its_media_in_place(self):
        source = self.media_workload(self.output.parent / "source.json")
        target = emit.copy_workload(source, emit.canonical_workload(self.output))
        self.assertEqual(target.read_bytes(), source.read_bytes())
        self.assertEqual((target.parent / "media/pixels.bin").read_bytes(), bytes(8))
        self.assertEqual(cache.workload_sha256(target), cache.workload_sha256(source))

    def test_the_workload_is_copied_with_its_media(self):
        source = self.media_workload(self.base / "workloads/media.json")
        target = emit.copy_workload(source, emit.canonical_workload(self.output))
        self.assertEqual(target.name, "reference.workload.json")
        self.assertEqual((target.parent / "media/grid.i64").read_bytes(), bytes(24))
        self.assertEqual(cache.workload_sha256(target), cache.workload_sha256(source))


if __name__ == "__main__":
    unittest.main()
