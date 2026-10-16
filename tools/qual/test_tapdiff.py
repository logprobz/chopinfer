"""tapdiff on synthetic dumps: request matching across packed and chunked
prefill steps, rank reduction and the first tap outside the floor. Needs
numpy (the runner's Python); skipped without it."""

import ast
import base64
import contextlib
import io
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock

try:
    import numpy as np
    from tools.model_sdk import inventory, taps as tap_names
    from tools.qual import tapdiff
except ModuleNotFoundError:
    np = None

COLS = 16


def bf16(values):
    """Values rounded to BF16 (nearest even) as F32, and their raw bits."""
    bits = np.asarray(values, dtype=np.float32).view(np.uint32)
    bits = ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16).astype(np.uint16)
    return (bits.astype(np.uint32) << 16).view(np.float32), bits


def write_step(directory, items, tensors, schema="chopinfer.taps.v1", ranks=1, expected=(), completion="full",
               encoder=None, model=None, producer=None):
    """A step directory: `tensors` maps a stored name to its per-rank
    (first_row, array, dtype) parts. Today's Qwen dumps record the expected
    tensor count, runner taps the expected list. `encoder` maps encoder taps
    to their row segments."""
    directory.mkdir(parents=True)
    lines = []
    for name, parts in tensors.items():
        for rank, (first, values, dtype) in enumerate(parts):
            data = bf16(values)[1] if dtype == "bf16" else values.astype(tapdiff.DTYPES[dtype])
            data.tofile(directory / f"{name}.r{rank}.bin")
            line = {"name": name, "rank": rank, "first_row": first, "rows": values.shape[0],
                    "cols": values.shape[1], "dtype": dtype}
            lines.append(line | {"axis": "encoder"} if name in (encoder or {}) else line)
    if schema == "chopinfer.qwen38.capture.v1":
        expected = len(lines)
    else:
        # The runner's form: every stored tap and rank, unless given.
        expected = list(expected) or [{"name": line["name"], "rank": line["rank"]} for line in lines]
    capture = {"schema": schema, "ranks": ranks, "expected": expected} | ({"model": model} if model else {}) \
        | ({"producer": {"startup_nonce": producer}} if producer else {})
    (directory / "capture.json").write_text(json.dumps(capture))
    (directory / "items.json").write_text(json.dumps({"items": items, **({"encoder": encoder} if encoder else {})}))
    (directory / "index.jsonl").write_text("".join(json.dumps(line) + "\n" for line in lines))
    (directory / "completion.json").write_text(json.dumps({"completion": completion, "reason": None}))


def other(ids):
    """A top-k id row with a different set: its first id replaced."""
    out = np.array(ids).copy()
    out[0] = next(i for i in range(512) if i not in set(out.tolist()))
    return out


def item(request, start, positions, tokens):
    return {"request": request, "start": start, "rows": len(positions), "prefix": positions[0],
            "positions": positions, "tokens": tokens}


@unittest.skipIf(np is None, "needs numpy (set CHOPINFER_REFRUN_PYTHON)")
class TapDiffTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR"))
        self.root = Path(self.tmp.name)
        self.rng = np.random.default_rng(5)
        self.sequences = {"long": list(range(100, 110)), "short": [7, 8, 9, 10]}

        quiet = contextlib.redirect_stdout(io.StringIO())
        quiet.__enter__()
        self.addCleanup(quiet.__exit__, None, None, None)

    def tearDown(self):
        self.tmp.cleanup()

    def values(self, cols=COLS):
        return {name: self.rng.standard_normal((len(tokens), cols)).astype(np.float32)
                for name, tokens in self.sequences.items()}

    def reference(self, mode, taps, positions=None, dtype=None, completion="full"):
        """A runner tap directory: one step per sequence, canonical names."""
        for step, (name, tokens) in enumerate(self.sequences.items()):
            keep = positions or list(range(len(tokens)))
            write_step(self.root / mode / f"step{step:04d}", [item(name, 0, keep, [tokens[p] for p in keep])],
                       {tap: [(0, values[name][keep], dtype or mode)] for tap, values in taps.items()},
                       completion=completion)
        return self.root / mode

    def clear(self):
        for name in ("engine", "f32"):
            shutil.rmtree(self.root / name)

    def diff(self, engine, f32, *extra):
        """tapdiff's exit status and receipt."""
        receipt = self.root / f"receipt{len(list(self.root.glob('receipt*')))}.json"
        code = tapdiff.run(tapdiff.parse_args(["--engine", str(self.reference("engine", engine, dtype="f32")),
                                               "--f32", str(self.reference("f32", f32)), *extra,
                                               "--json", str(receipt)]))
        return code, json.loads(receipt.read_text())

    def noisy(self, values, scale):
        return {name: (v * (1 + scale * self.rng.standard_normal(v.shape))).astype(np.float32)
                for name, v in values.items()}

    def test_qwen_dump_packed_and_chunked_prefill_in_both_schemas(self):
        """The same engine tensors as an older Qwen dump (its own names and
        schema) and as a chopinfer.taps.v1 dump give the same result."""
        long, short = self.sequences["long"], self.sequences["short"]
        # Every tap has its declared width (models/qwen38/taps.json).
        f32 = {"L00.x.hc.attn": self.values(2560), "L00.router.ids": None, "L00.router.weights": self.values(10),
               "L00.attn.out": None, "L00.x.gdn.qkv": self.values(10240), "L00.x.moe.routed": self.values(25600),
               "L01.mlp.out": self.values(2560)}
        # Router ids: 10 per row, below 256 so that the bf16 run holds them
        # exactly; the engine lists every row in another order and changes
        # one row.
        f32["L00.router.ids"] = {name: np.stack([self.rng.permutation(256)[:10] for _ in tokens]).astype(np.float32)
                                 for name, tokens in self.sequences.items()}
        # Attention output: the rank-ordered F32 sum of four BF16 partials.
        partials = {name: [bf16(self.rng.standard_normal((len(tokens), 2560)))[0] for _ in range(4)]
                    for name, tokens in self.sequences.items()}
        f32["L00.attn.out"] = {}
        for name, parts in partials.items():
            total = parts[0].copy()
            for part in parts[1:]:
                total = total + part
            f32["L00.attn.out"][name] = total
        bf16_taps = {tap: self.noisy(values, 4e-3) if tap != "L00.router.ids" else values
                     for tap, values in f32.items()}
        bf16_taps["L00.router.ids"] = {name: v.copy() for name, v in f32["L00.router.ids"].items()}
        bf16_taps["L00.router.ids"]["long"][3] = other(f32["L00.router.ids"]["long"][3])
        engine = {tap: self.noisy(values, 1e-3) for tap, values in f32.items()}
        engine["L01.mlp.out"] = self.noisy(f32["L01.mlp.out"], 0.1)
        # The engine lists the top 10 in the other order.
        engine["L00.router.ids"] = {name: v[:, ::-1].copy() for name, v in f32["L00.router.ids"].items()}
        engine["L00.router.weights"] = {name: v[:, ::-1].copy() for name, v in f32["L00.router.weights"].items()}
        engine["L00.router.ids"]["short"][1] = other(f32["L00.router.ids"]["short"][1])

        def stored(name, rows):
            """Engine Qwen v1 tensors of the (sequence, position) rows of one step."""
            def take(tap):
                return np.concatenate([engine[tap][seq][[p]] for seq, p in rows])
            qkv = take("L00.x.gdn.qkv")
            shards = []
            for rank in range(4):
                blocks = [(rank * 512, 512), (2048 + rank * 512, 512),
                          (4096 + rank * 1536, 1536)]
                shards.append((0, np.concatenate([qkv[:, s:s + c] for s, c in blocks], axis=1), "bf16"))
            attn = [np.concatenate([partials[seq][r][[p]] for seq, p in rows]) for r in range(4)]
            return {"layer00.attn.in": [(0, take("L00.x.hc.attn"), "bf16")],
                    "layer00.attn.partial": [(0, part, "bf16") for part in attn],
                    "layer00.moe.ids": [(0, take("L00.router.ids"), "u32")],
                    "layer00.moe.weights": [(0, take("L00.router.weights"), "f32")],
                    "layer00.lin.qkv": shards,
                    "layer00.moe.routed": [(0, take("L00.x.moe.routed"), "bf16")] * 4,
                    "layer01.mlp.partial": [(0, take("L01.mlp.out"), "bf16")] +
                                           [(0, np.zeros((len(rows), 2560), np.float32), "bf16")] * 3,
                    "layer03.attn.selected": [(0, np.zeros((len(rows), 512)), "u32")] * 4}
        qwen = tap_names.v1_declaration("chopinfer.qwen38.capture.v1")

        def renamed(tensors):
            """The tensors under their chopinfer.taps.v1 names, and the
            expected entries the engine declares for them."""
            out, expected = {}, []
            for v1, parts in tensors.items():
                name, entry = qwen.canonical(v1)
                out[name] = parts
                expected += [{"name": name, "rank": rank, "sharding": entry["sharding"],
                              "compare": entry["compare"]} for rank in range(len(parts))]
            return out, expected

        reference = ["--f32", str(self.reference("f32", f32)), "--bf16", str(self.reference("bf16", bf16_taps))]
        reports = {}
        for schema in ("chopinfer.qwen38.capture.v1", "chopinfer.taps.v1"):
            engine_dir = self.root / schema

            def step(directory, items, tensors):
                if schema == "chopinfer.taps.v1":
                    tensors, expected = renamed(tensors)
                    write_step(engine_dir / directory, items, tensors, schema, 4, expected, model="qwen38")
                else:
                    write_step(engine_dir / directory, items, tensors, schema, 4)

            # A warmup pass of another request, then a packed prefill of the
            # long sequence's first 6 positions and the whole short one, then
            # the rest of the long one as a chunk with a prefix.
            warm = [(("long", 0), 1), (("long", 1), 2)]
            step("step0000", [item(3, 0, [0, 1], [1, 2])], stored("warm", [k for k, _ in warm]))
            rows = [("long", p) for p in range(6)] + [("short", p) for p in range(4)]
            step("step0001", [item(16, 0, list(range(6)), long[:6]), item(17, 6, list(range(4)), short)],
                 stored("packed", rows))
            rows = [("long", p) for p in range(6, 10)]
            step("step0002", [item(16, 0, list(range(6, 10)), long[6:])], stored("chunk", rows))
            receipt = self.root / f"{schema}.json"
            self.assertEqual(tapdiff.run(tapdiff.parse_args(
                ["--engine", str(engine_dir), *reference, "--json", str(receipt)])), 1)
            reports[schema] = json.loads(receipt.read_text())
        old, report = reports.values()
        self.assertIn("step0001/items.json", report["inputs"]["engine"]["files"])
        self.assertNotEqual(old["inputs"]["engine"], report["inputs"]["engine"])
        for value in (old, report):
            del value["inputs"]["engine"]
        self.assertEqual(old, report)
        self.assertEqual(report["requests"], {"16": "long", "17": "short"})
        self.assertEqual(report["ignored_requests"], ["3"])
        taps = {row["tap"]: row for row in report["taps"]}
        self.assertEqual([row["tap"] for row in report["taps"]],
                         ["L00.x.hc.attn", "L00.router.ids", "L00.router.weights", "L00.attn.out", "L00.x.gdn.qkv",
                          "L00.x.moe.routed", "L01.mlp.out"])
        # Expert slots are not reduced, so not compared.
        routed = taps["L00.x.moe.routed"]
        self.assertEqual(routed["status"], 'not compared: the engine holds "expert_partial" shards')
        self.assertTrue(all(row["rows"] == 14 for row in report["taps"] if row is not routed))
        # Weights follow their ids; the engine row routed elsewhere is a flip, not a weight error.
        weights = taps["L00.router.weights"]
        self.assertEqual((weights["compare"], weights["e"]["max"], weights["route_flips"], weights["bf16_rows"]),
                         ("paired", 0.0, 1, 12))
        # The engine's attention output is the reference's own F32 sum; the shards concatenate exactly
        # up to BF16 rounding.
        self.assertEqual(taps["L00.attn.out"]["e"]["max"], 0.0)
        self.assertLess(taps["L00.x.gdn.qkv"]["e"]["max"], 4e-3)
        self.assertEqual((taps["L00.router.ids"]["disagree"], taps["L00.router.ids"]["bf16_disagree"]),
                         (1 / 14, 1 / 14))
        self.assertEqual([taps[t]["status"] for t in taps if t != "L00.x.moe.routed"],
                         ["ok", "ok", "ok", "ok", "ok", "fail"])
        self.assertEqual(report["first"], "L01.mlp.out")
        self.assertEqual(report["first_exceeding"], {"L00.router.ids": 1, "L01.mlp.out": 13})
        self.assertEqual(report["engine_only"], ["L03.x.indexer.topk"])

    def test_declarations_are_required_and_checked(self):
        """A capture whose schema no declaration names, an undeclared tap, or
        a tap whose dtype, columns or ranks differ from its declaration is
        incompatible, as is an undeclared sharding in a multi-rank capture."""
        v1, shared = "chopinfer.qwen38.capture.v1", "chopinfer.taps.v1"
        f32_dir = str(self.reference("f32", {"L00.x.hc.attn": self.values(2560)}))
        selected = {name: np.stack([self.rng.permutation(1024)[:512] for _ in tokens]).astype(np.uint32)
                    for name, tokens in self.sequences.items()}

        def engine(label, name, schema, values=None, dtype="bf16", ranks=1):
            directory = self.root / label
            values = values or self.values(2560)
            for step, (sequence, tokens) in enumerate(self.sequences.items()):
                write_step(directory / f"step{step:04d}", [item(sequence, 0, list(range(len(tokens))), tokens)],
                           {name: [(0, values[sequence], dtype)] * ranks}, schema, 4)
            return str(directory)

        cases = [
            ("v1 without a declaration", engine("a", "layer00.attn.in", v1), True,
             r"chopinfer\.qwen38\.capture\.v1.*layer00\.attn\.in"),
            ("unknown schema", engine("b", "L00.x.hc.attn", "chopinfer.mystery.v1"), False,
             r"chopinfer\.mystery\.v1.*L00\.x\.hc\.attn"),
            ("undeclared tap", engine("c", "layer00.attn.renamed", v1), False,
             r"chopinfer\.qwen38\.capture\.v1.*layer00\.attn\.renamed"),
            ("dtype", engine("d", "layer00.attn.in", v1, dtype="f32"), False, r"layer00\.attn\.in.*dtype"),
            ("columns", engine("e", "layer00.attn.in", v1, values=self.values(16)), False,
             r"layer00\.attn\.in.*columns"),
            ("rank copies", engine("f", "layer03.attn.selected", v1, values=selected, dtype="u32"), False,
             r"layer03\.attn\.selected.*ranks"),
            ("sharding", engine("g", "L00.x.hc.attn", shared), False, r"L00\.x\.hc\.attn.*sharding"),
        ]
        for case, directory, hidden, pattern in cases:
            with self.subTest(case), contextlib.ExitStack() as stack:
                if hidden:
                    stack.enter_context(mock.patch.object(tap_names, "declarations", lambda root=None: []))
                args = tapdiff.parse_args(["--engine", directory, "--f32", f32_dir])
                with self.assertRaisesRegex(ValueError, pattern):
                    tapdiff.run(args)
                stderr = stack.enter_context(contextlib.redirect_stderr(io.StringIO()))
                self.assertEqual(tapdiff.main(["--engine", directory, "--f32", f32_dir]), 2)
                self.assertRegex(stderr.getvalue(), "incompatible: .*" + pattern)

    def test_rank_copies_must_agree_bitwise(self):
        """A tap every rank computes alike (attn.selected) is compared on
        rank 0 only when every rank's copy is bitwise equal."""
        ids = {name: np.stack([self.rng.permutation(1024)[:512] for _ in tokens]).astype(np.float32)
               for name, tokens in self.sequences.items()}
        reference = ["--f32", str(self.reference("f32", {"L03.x.indexer.topk": ids})),
                     "--bf16", str(self.reference("bf16", {"L03.x.indexer.topk": ids}))]
        for schema, name in [("chopinfer.taps.v1", "L03.x.indexer.topk"),
                             ("chopinfer.qwen38.capture.v1", "layer03.attn.selected")]:
            for differ in (False, True):
                directory = self.root / f"{schema}-{differ}"
                for step, (sequence, tokens) in enumerate(self.sequences.items()):
                    copies = [ids[sequence].astype(np.uint32) for _ in range(4)]
                    if differ and sequence == "long":
                        copies[2] = copies[2].copy()
                        copies[2][1, 0] += 1
                    expected = [{"name": name, "rank": rank, "sharding": "replicated", "compare": "set"}
                                for rank in range(4)] if schema == "chopinfer.taps.v1" else ()
                    write_step(directory / f"step{step:04d}", [item(sequence, 0, list(range(len(tokens))), tokens)],
                               {name: [(0, copy, "u32") for copy in copies]}, schema, 4, expected)
                receipt = self.root / f"{schema}-{differ}.json"
                code = tapdiff.run(tapdiff.parse_args(["--engine", str(directory), *reference,
                                                       "--json", str(receipt)]))
                report = json.loads(receipt.read_text())
                row = report["taps"][0]
                with self.subTest(schema=schema, differ=differ):
                    if differ:
                        self.assertEqual((code, report["status"], report["first"]),
                                         (1, "diverged", "L03.x.indexer.topk"))
                        self.assertEqual(row["status"], "fail: rank 2's copy differs from rank 0's")
                    else:
                        self.assertEqual((code, row["status"]), (0, "ok"))

    def test_canonical_captures_match_their_index_and_declaration(self):
        """A chopinfer.taps.v1 capture whose index holds a tap capture.json
        does not declare, whose declared dtype, columns or rows differ from
        the index, or whose model's declaration does not name a tap or
        declares it otherwise, is incompatible."""
        f32 = {"L00.out": self.values(1)}
        f32_dir = str(self.reference("f32", f32))

        def engine(label, tensors, expected, model=None, ranks=4):
            directory = self.root / label
            for step, (sequence, tokens) in enumerate(self.sequences.items()):
                rows = [{**entry, "rows": len(tokens)} if "rows" in entry else entry for entry in expected]
                write_step(directory / f"step{step:04d}", [item(sequence, 0, list(range(len(tokens))), tokens)],
                           {name: [(0, values[sequence], dtype)] for name, (values, dtype) in tensors.items()},
                           ranks=ranks, expected=rows, model=model)
            return str(directory)

        out = {"name": "L00.out", "rank": 0, "sharding": "replicated", "compare": "dense", "dtype": "f32",
               "cols": 1, "first_row": 0, "rows": None}
        hidden = {"L00.out": (self.values(10240), "bf16")}
        hidden_entry = {**out, "dtype": "bf16", "cols": 10240}
        cases = [
            ("extra index tap", engine("a", {"L00.out": (f32["L00.out"], "f32"),
                                             "L00.bogus": (f32["L00.out"], "f32")}, [out], ranks=1),
             r"L00\.bogus rank 0 is in the index.*not in capture\.json"),
            ("declared dtype and width", engine("b", {"L00.out": (f32["L00.out"], "f32")},
                                                [{**out, "dtype": "u32", "cols": 2}], ranks=1),
             r"L00\.out rank 0 has dtype f32, capture\.json declares u32"),
            ("undeclared by the model", engine("c", {"L00.x.bogus": (f32["L00.out"], "f32")},
                                               [{**out, "name": "L00.x.bogus"}], model="qwen38"),
             r"qwen38.*does not declare L00\.x\.bogus"),
            ("the model declares another width", engine("d", {"L00.out": (f32["L00.out"], "f32")}, [out],
                                                        model="qwen38"),
             r"L00\.out \(capture schema chopinfer\.taps\.v1\) has dtype \['f32'\], its declaration bf16"),
            ("no declaration of the model", engine("e", hidden, [hidden_entry], model="mystery"),
             r"no tap declaration.*model 'mystery'"),
        ]
        for case, directory, pattern in cases:
            with self.subTest(case):
                args = tapdiff.parse_args(["--engine", directory, "--f32", f32_dir])
                with self.assertRaisesRegex(ValueError, pattern):
                    tapdiff.run(args)
                with contextlib.redirect_stderr(io.StringIO()):
                    self.assertEqual(tapdiff.main(["--engine", directory, "--f32", f32_dir]), 2)
        # The model's own layout reads.
        good = engine("f", hidden, [hidden_entry], model="qwen38")
        reference = str(self.reference("f32-hidden", {"L00.out": hidden["L00.out"][0]}, dtype="f32"))
        self.assertIn(tapdiff.run(tapdiff.parse_args(["--engine", good, "--f32", reference])), (0, 1))

    def test_runner_format_sampled_rows_and_declared_shards(self):
        f32 = {"L00.in": self.values(), "L00.out": self.values()}
        bf16_taps = {tap: self.noisy(values, 4e-3) for tap, values in f32.items()}
        engine_dir = self.root / "engine"
        for step, (name, tokens) in enumerate(self.sequences.items()):
            n = len(tokens)
            halves = [f32["L00.in"][name][:, :8], f32["L00.in"][name][:, 8:]]
            write_step(engine_dir / f"step{step:04d}", [item(name, 0, list(range(n)), tokens)],
                       {"L00.in": [(0, halves[1], "f32"), (0, halves[0], "f32")],
                        "L00.out": [(0, f32["L00.out"][name], "f32")]}, ranks=2,
                       expected=[{"name": "L00.in", "rank": 0}, {"name": "L00.in", "rank": 1},
                                 {"name": "L00.out", "rank": 0, "sharding": "replicated"}])
        declare = self.root / "taps.json"
        declare.write_text(json.dumps({"taps": [{"name": "L*.in", "sharding": {"col_shards": [[[8, 8]], [[0, 8]]]}}]}))
        # The reference keeps positions 0, 2 and 3 only.
        common = ["--engine", str(engine_dir), "--f32", str(self.reference("f32", f32, [0, 2, 3])),
                  "--bf16", str(self.reference("bf16", bf16_taps, [0, 2, 3]))]
        with self.assertRaisesRegex(ValueError, "no declared sharding"):
            tapdiff.run(tapdiff.parse_args(common))
        args = tapdiff.parse_args(common + ["--declare", str(declare), "--json", str(self.root / "r.json")])
        self.assertEqual(tapdiff.run(args), 0)
        report = json.loads((self.root / "r.json").read_text())
        self.assertEqual([(row["tap"], row["rows"], row["e"]["max"]) for row in report["taps"]],
                         [("L00.in", 6, 0.0), ("L00.out", 6, 0.0)])
        self.assertEqual(report["status"], "match")

    def test_reductions(self):
        parts = [bf16(self.rng.standard_normal((3, 4)))[0] for _ in range(4)]
        expected = ((parts[0] + parts[1]) + parts[2]) + parts[3]
        self.assertTrue(np.array_equal(tapdiff.reduce(parts, "rank_partial", 4), expected))
        with self.assertRaisesRegex(ValueError, "ranks 3 of 4"):
            tapdiff.reduce(parts[:3], "rank_partial", 4)
        shards = {"col_shards": [[[0, 2]], [[2, 2]], [[4, 2]], [[4, 2]]]}
        with self.assertRaisesRegex(ValueError, "overlap"):
            tapdiff.reduce([p[:, :2] for p in parts], shards, 4)

    def router(self):
        """f32 router ids (top 2 of 8) and weights per sequence."""
        ids = {name: np.stack([self.rng.permutation(8)[:2] for _ in tokens]).astype(np.float32)
               for name, tokens in self.sequences.items()}
        weights = {name: np.sort(self.rng.random((len(tokens), 2)), axis=1)[:, ::-1].astype(np.float32)
                   for name, tokens in self.sequences.items()}
        return {"L00.router.ids": ids, "L00.router.weights": weights}

    def test_router_weights_follow_their_ids(self):
        f32 = self.router()
        bf16_taps = {"L00.router.ids": f32["L00.router.ids"],
                     "L00.router.weights": self.noisy(f32["L00.router.weights"], 1e-3)}
        bf16_dir = str(self.reference("bf16", bf16_taps))
        # Ids and weights permuted together: the same routing.
        together = {tap: {name: v[:, ::-1].copy() for name, v in values.items()} for tap, values in f32.items()}
        code, report = self.diff(together, f32, "--bf16", bf16_dir)
        self.assertEqual((code, report["status"]), (0, "match"))
        # Only the weights swapped: the same ids, the wrong weight on each expert.
        self.clear()
        swapped = {"L00.router.ids": f32["L00.router.ids"],
                   "L00.router.weights": {name: v[:, ::-1].copy() for name, v in f32["L00.router.weights"].items()}}
        code, report = self.diff(swapped, f32, "--bf16", bf16_dir)
        self.assertEqual((code, report["first"]), (1, "L00.router.weights"))

    def test_router_weights_need_their_ids(self):
        f32 = self.router()
        code, report = self.diff({"L00.router.weights": f32["L00.router.weights"]}, f32)
        self.assertEqual((code, report["status"]), (2, "incompatible"))
        self.assertIn("without L00.router.ids", report["taps"][1]["status"])

    def test_optional_arms_cannot_hide_a_divergence(self):
        f32 = {"L00.in": self.values(), "L00.out": self.values()}
        engine = {"L00.in": f32["L00.in"], "L00.out": {name: 2 * v for name, v in f32["L00.out"].items()}}
        code, report = self.diff(engine, f32)
        self.assertEqual((code, report["first"]), (1, "L00.out"))
        # An f64 arm with only L00.in, and a bf16 arm with only positions 0 and 1.
        f64 = str(self.reference("f64", {"L00.in": f32["L00.in"]}, dtype="f32"))
        bf16_dir = str(self.reference("bf16", {"L00.out": self.noisy(f32["L00.out"], 1e-3)}, [0, 1]))
        self.clear()
        code, report = self.diff(engine, f32, "--f64", f64, "--bf16", bf16_dir)
        self.assertEqual((code, report["first"]), (1, "L00.out"))
        out = report["taps"][1]
        self.assertEqual((out["rows"], out["bf16_rows"], out["unfloored_rows"], out.get("f64_rows")), (14, 4, 10, None))
        self.assertEqual((report["taps"][0]["f64_rows"], report["taps"][0]["num"]["max"]), (14, 0.0))

    def test_every_row_holds_each_encoder_row_to_its_floor(self):
        """One corrupt row of six passes the median rule; --every-row fails
        it. A row whose bf16 run equals f32 is held to the row floor, not to
        exactness."""
        f32 = self.rng.standard_normal((6, COLS)).astype(np.float32)
        bf16 = f32 * (1 + 4e-3 * self.rng.standard_normal(f32.shape)).astype(np.float32)
        bf16[2] = f32[2]
        engine = f32 * (1 + 4e-3 * self.rng.standard_normal(f32.shape)).astype(np.float32)
        engine[2] = f32[2] * (1 + 2e-3)
        long = self.sequences["long"]
        segments = [{"request": "long", "start": 0, "kind": "image", "item": 0, "index": 0, "rows": 6}]
        for corrupt, every_row, expected in ((False, True, (0, "match")), (True, False, (0, "match")),
                                             (True, True, (1, "diverged"))):
            base = self.root / f"case-{corrupt}-{every_row}"
            rows = engine.copy()
            if corrupt:
                rows[4] = -rows[4]
            for arm, values in (("engine", rows), ("f32", f32), ("bf16", bf16)):
                write_step(base / arm / "step0000", [item(16 if arm == "engine" else "long", 0, list(range(10)), long)],
                           {"enc.vision.out": [(0, values, "f32")]},
                           encoder={"enc.vision.out": segments if arm != "engine"
                                    else [dict(segments[0], request=16)]})
            extra = ["--every-row", "enc.*"] if every_row else []
            code = tapdiff.main(["--engine", str(base / "engine"), "--f32", str(base / "f32"), "--bf16",
                                 str(base / "bf16"), *extra, "--json", str(base / "receipt.json")])
            report = json.loads((base / "receipt.json").read_text())
            self.assertEqual((code, report["status"]), expected, base.name)
            vision = report["taps"][0]
            self.assertEqual(sorted(vision["e_over_floor"]), ["max", "median", "p90"])
            if every_row:
                self.assertEqual(vision["every_row"]["exceeding"], int(corrupt), base.name)
                self.assertEqual(report["encoder_rows"]["enc.vision.out"]["exceeding"], int(corrupt))
                self.assertEqual(report["row_floor"], 2.0 ** -8)

    def test_encoder_only_reference_sequences_take_their_tokens_from_the_workload(self):
        """The encoder reference (tools/refrun/encoder.py) stores encoder rows
        without positions; --workload names its sequences' tokens, so the
        engine's request matches."""
        f32 = self.rng.standard_normal((6, COLS)).astype(np.float32)
        long = self.sequences["long"]
        segments = [{"request": "long", "start": 0, "kind": "image", "item": 0, "index": 0, "rows": 6}]
        write_step(self.root / "f32" / "step0000",
                   [{"request": "long", "start": 0, "rows": 0, "prefix": 0, "positions": [], "tokens": []}],
                   {"enc.vision.out": [(0, f32, "f32")]}, encoder={"enc.vision.out": segments})
        write_step(self.root / "engine" / "step0000", [item(16, 0, list(range(10)), long)],
                   {"enc.vision.out": [(0, f32, "f32")]}, encoder={"enc.vision.out": [dict(segments[0], request=16)]})
        workload = self.root / "workload.json"
        workload.write_text(json.dumps({"sequences": [{"name": "long", "token_ids": long + [1, 2]},
                                                      {"name": "short", "token_ids": self.sequences["short"]}]}))
        arguments = ["--engine", str(self.root / "engine"), "--f32", str(self.root / "f32")]
        with contextlib.redirect_stderr(io.StringIO()) as error:
            self.assertEqual(tapdiff.main(arguments), 2)
        self.assertIn("no engine request matches", error.getvalue())
        receipt = self.root / "workload-receipt.json"
        self.assertEqual(tapdiff.main([*arguments, "--workload", str(workload), "--json", str(receipt)]), 0)
        report = json.loads(receipt.read_text())
        self.assertEqual((report["requests"], report["taps"][0]["rows"]), ({"16": "long"}, 6))

    def test_every_row_skips_rows_whose_codes_flipped(self):
        """With :CODES/GROUP an audio row is judged only where the engine's
        codes it merges equal f32's: a flipped code's row is left to the
        codes tap and counted; any other row still fails alone."""
        f32 = self.rng.standard_normal((3, COLS)).astype(np.float32)
        bf16 = f32 * (1 + 4e-3 * self.rng.standard_normal(f32.shape)).astype(np.float32)
        codes = self.rng.integers(0, 1024, (10, 4)).astype(np.float64)
        long = self.sequences["long"]

        def run(case, corrupt, flipped):
            base = self.root / case
            rows, engine_codes = f32.copy(), codes.copy()
            rows[corrupt] = -rows[corrupt]
            engine_codes[flipped, 2] += 1
            for arm, (values, arm_codes) in (("engine", (rows, engine_codes)), ("f32", (f32, codes)),
                                             ("bf16", (bf16, codes))):
                request = 16 if arm == "engine" else "long"
                segments = {tap: [{"request": request, "start": 0, "kind": "audio", "item": 0, "index": 0,
                                   "rows": count}] for tap, count in (("enc.audio.out", 3), ("enc.audio.x.codes", 10))}
                write_step(base / arm / "step0000", [item(request, 0, list(range(10)), long)],
                           {"enc.audio.x.codes": [(0, arm_codes, "u32")], "enc.audio.out": [(0, values, "f32")]},
                           encoder=segments)
            code = tapdiff.main(["--engine", str(base / "engine"), "--f32", str(base / "f32"), "--bf16",
                                 str(base / "bf16"), "--every-row", "enc.audio.out:enc.audio.x.codes/4",
                                 "--codes-margin", "0.5", "--json", str(base / "receipt.json")])
            report = json.loads((base / "receipt.json").read_text())
            return code, next(row for row in report["taps"] if row["tap"] == "enc.audio.out")
        # Code row 5 is merged by audio row 1, code row 9 by the short last row 2.
        code, row = run("flipped-row-corrupt", 1, [5])
        self.assertEqual((code, row["code_flipped_rows"], row["rows"], row["every_row"]["exceeding"]), (0, 1, 3, 0))
        code, row = run("matched-row-corrupt", 0, [9])
        self.assertEqual((code, row["code_flipped_rows"], row["every_row"]["exceeding"]), (1, 1, 1))
        with contextlib.redirect_stderr(io.StringIO()) as error:
            self.assertEqual(tapdiff.main(["--engine", str(self.root), "--f32", str(self.root),
                                           "--every-row", "enc.*:codes"]), 2)
        self.assertIn("expected GLOB:CODES_TAP/GROUP", error.getvalue())

    def test_direction_and_scale_of_encoder_rows(self):
        """--direction-scale judges each row's direction against the bf16
        run's, and each item's largest row scale error against the largest
        of the bf16 run's and the scale variants' rounded up to 0.005."""
        f32 = self.rng.standard_normal((6, COLS)).astype(np.float32) + 3
        bent = (f32 * (1 + 2e-3 * self.rng.standard_normal(f32.shape))).astype(np.float32)
        long = self.sequences["long"]
        segments = [{"request": "long", "start": 0, "kind": "image", "item": 0, "index": 0, "rows": 6}]

        def run(case, engine, variant_scale, corrupt=False):
            base = self.root / case
            engine = engine.copy()
            if corrupt:
                engine[2] = engine[2][::-1]
            arms = {"engine": engine, "f32": f32, "bf16": bent * 1.004, "policy": f32 * variant_scale}
            for arm, values in arms.items():
                request = 16 if arm == "engine" else "long"
                write_step(base / arm / "step0000", [item(request, 0, list(range(10)), long)],
                           {"enc.vision.out": [(0, values, "f32")]},
                           encoder={"enc.vision.out": [dict(segments[0], request=request)]})
            code = tapdiff.main(["--engine", str(base / "engine"), "--f32", str(base / "f32"), "--bf16",
                                 str(base / "bf16"), "--direction-scale", "enc.vision.out", "--scale-variant",
                                 f"policy={base / 'policy'}", "--json", str(base / "receipt.json")])
            return code, json.loads((base / "receipt.json").read_text())["taps"][0]
        # The engine is 1% large: the bf16 run's 0.4% bound rounds up to 0.5%,
        # a policy run 1.2% large raises it to 1.5%.
        code, tap = run("scaled", bent * 1.01, 1.0)
        scale = tap["scale"]["long/image/0"]
        self.assertEqual((code, scale["limit"], scale["passed"]), (1, 0.005, False))
        self.assertAlmostEqual(scale["engine"], 0.01, delta=0.002)
        code, tap = run("variant", bent * 1.01, 1.012)
        self.assertEqual((code, tap["scale"]["long/image/0"]["limit"], tap["status"]), (0, 0.015, "ok"))
        self.assertEqual(tap["direction"]["every_row"]["exceeding"], 0)
        code, tap = run("turned", bent * 1.01, 1.012, corrupt=True)
        self.assertEqual((code, tap["direction"]["every_row"]["exceeding"]), (1, 1))

    def test_codes_agree_within_the_bf16_runs_agreement(self):
        """Codes compare one by one: the engine may agree with f32 less often
        than bf16 does by at most the margin, over all codes and the first
        stage."""
        f32 = self.rng.integers(0, 1024, (40, 4)).astype(np.float64)
        bf16 = f32.copy()
        bf16[:8, 1:] += 1  # 24 of 160 codes, none of the first stage
        long = self.sequences["long"]
        segments = [{"request": "long", "start": 0, "kind": "audio", "item": 0, "index": 0, "rows": 40}]

        def run(engine, case, margin=None):
            base = self.root / case
            for arm, values in (("engine", engine), ("f32", f32), ("bf16", bf16)):
                write_step(base / arm / "step0000", [item(16 if arm == "engine" else "long", 0, list(range(10)), long)],
                           {"enc.audio.x.codes": [(0, values, "u32")]},
                           encoder={"enc.audio.x.codes": segments if arm != "engine"
                                    else [dict(segments[0], request=16)]})
            extra = ["--codes-margin", str(margin)] if margin is not None else []
            code = tapdiff.main(["--engine", str(base / "engine"), "--f32", str(base / "f32"), "--bf16",
                                 str(base / "bf16"), *extra, "--json", str(base / "receipt.json")])
            return code, json.loads((base / "receipt.json").read_text())["taps"][0]
        within = f32.copy()
        within[:9, 1:] += 1  # 27 of 160 codes: 0.83125 against bf16's 0.85
        code, tap = run(within, "within")
        self.assertEqual((code, tap["compare"], tap["agreement"], tap["bf16_agreement"]), (0, "codes", 0.83125, 0.85))
        stage0 = f32.copy()
        stage0[:2, 0] += 1  # 2 first-stage codes: 0.95 against 1.0
        code, tap = run(stage0, "stage0")
        self.assertEqual((code, tap["stage0_agreement"], tap["bf16_stage0_agreement"]), (1, 0.95, 1.0))
        code, tap = run(stage0, "stage0-wide", margin=0.05)
        self.assertEqual(code, 0)
        worse = f32.copy()
        worse[:12, 1:] += 1
        code, tap = run(worse, "worse")
        self.assertEqual((code, tap["agreement"], tap["status"]), (1, 0.775, "fail"))

    def test_zero_reference_rows(self):
        zero = {"L00.out": {name: np.zeros((len(tokens), COLS), np.float32)
                            for name, tokens in self.sequences.items()}}
        bf16_dir = str(self.reference("bf16", zero))
        code, report = self.diff(zero, zero, "--bf16", bf16_dir)
        self.assertEqual((code, report["taps"][0]["e"]["max"]), (0, 0.0))
        self.clear()
        ones = {"L00.out": {name: np.ones_like(v) for name, v in zero["L00.out"].items()}}
        code, report = self.diff(ones, zero, "--bf16", bf16_dir)
        self.assertEqual((code, report["first"], report["taps"][0]["e"]["median"]), (1, "L00.out", "inf"))

    def test_nonfinite_values(self):
        f32 = {"L00.out": self.values()}
        engine = {"L00.out": {name: v.copy() for name, v in f32["L00.out"].items()}}
        engine["L00.out"]["long"][2, 0] = np.nan
        engine["L00.out"]["long"][5, 3] = np.inf
        code, report = self.diff(engine, f32)
        row = report["taps"][0]
        self.assertEqual((code, row["status"], row["e"]["max"], row["e"]["median"], row["e"]["inf_rows"]),
                         (0, "ok", "inf", 0.0, 2))
        self.assertEqual(report["first_exceeding"], {"L00.out": 2})
        self.clear()
        broken = {"L00.out": {name: np.full_like(v, np.nan) for name, v in f32["L00.out"].items()}}
        code, report = self.diff(broken, f32)
        self.assertEqual((code, report["taps"][0]["e"]["p90"]), (1, "inf"))
        self.clear()
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            self.diff(f32, broken)

    def test_incomplete_captures_never_match(self):
        f32 = {"L00.in": self.values(), "L00.out": self.values()}
        self.reference("engine", {"L00.in": f32["L00.in"]}, dtype="f32", completion="incomplete")
        f32_dir = str(self.reference("f32", f32))
        args = ["--engine", str(self.root / "engine"), "--f32", f32_dir]
        self.assertEqual(tapdiff.run(tapdiff.parse_args(args)), 3)
        # A complete record whose declared taps the index lacks.
        write_step(self.root / "declared" / "step0000", [item("long", 0, [0, 1], self.sequences["long"][:2])],
                   {"L00.in": [(0, f32["L00.in"]["long"][:2], "f32")]},
                   expected=[{"name": "L00.in", "rank": 0}, {"name": "L00.out", "rank": 0}])
        receipt = self.root / "declared.json"
        code = tapdiff.run(tapdiff.parse_args(["--engine", str(self.root / "declared"), "--f32", f32_dir,
                                               "--json", str(receipt)]))
        report = json.loads(receipt.read_text())
        self.assertEqual((code, report["status"]), (3, "incomplete"))
        self.assertIn("L00.out", report["incomplete_steps"]["engine"]["step0000"])
        # A divergence in what was captured is still reported.
        shutil.rmtree(self.root / "engine")
        self.reference("engine", {"L00.in": {name: 2 * v for name, v in f32["L00.in"].items()}}, dtype="f32",
                       completion="incomplete")
        self.assertEqual(tapdiff.run(tapdiff.parse_args(args)), 1)

    def probe(self, tensors_of):
        """tapdiff over one four-row step per arm, every tap declared."""
        for arm in ("engine", "f32", "bf16"):
            tensors = tensors_of(arm)
            write_step(self.root / arm / "step0000", [item("seq", 0, [0, 1, 2, 3], [10, 11, 12, 13])], tensors,
                       expected=[{"name": name, "rank": 0, "sharding": "replicated"} for name in tensors])
        receipt = self.root / "probe.json"
        code = tapdiff.main(["--engine", str(self.root / "engine"), "--f32", str(self.root / "f32"),
                             "--bf16", str(self.root / "bf16"), "--json", str(receipt)])
        return code, json.loads(receipt.read_text())

    def test_a_tap_holding_fewer_rows_than_its_step_is_incomplete(self):
        # The engine's L00.out index and payload both hold position 0 only.
        code, report = self.probe(lambda arm: {
            "L00.in": [(0, np.ones((4, 2)), "f32")],
            "L00.out": [(0, np.ones((1 if arm == "engine" else 4, 2)), "f32")]})
        self.assertEqual((code, report["status"]), (3, "incomplete"))
        self.assertIn("L00.out rank 0 holds rows 0..1 of 0..4", report["incomplete_steps"]["engine"]["step0000"])

    def test_router_ids_must_index_every_weight(self):
        # One id for two weights: the second weight has no expert.
        code, report = self.probe(lambda arm: {
            "L00.router.ids": [(0, np.full((4, 1), 2), "u32")],
            "L00.router.weights": [(0, np.tile([0.9, 100 if arm == "engine" else 0.1], (4, 1)), "f32")]})
        self.assertEqual((code, report["status"]), (2, "incompatible"))
        self.assertIn("1 router ids for 2 weights", report["taps"][1]["status"])
        for name in ("engine", "f32", "bf16"):
            shutil.rmtree(self.root / name)
        self.root.joinpath("probe.json").unlink()
        code, report = self.probe(lambda arm: {
            "L00.router.ids": [(0, np.tile([2, 2 if arm == "engine" else 3], (4, 1)), "u32")],
            "L00.router.weights": [(0, np.tile([0.9, 0.1], (4, 1)), "f32")]})
        self.assertEqual(code, 2)
        self.assertIn("engine router ids repeat an expert", report["taps"][1]["status"])

    def test_encoder_rows_align_by_index_across_steps(self):
        """Encoder taps' rows are keyed by (request, encoder row index): the
        engine's two steps of a request's image rows line up with the
        reference's one step."""
        f32 = self.rng.standard_normal((6, COLS)).astype(np.float32)
        long = self.sequences["long"]
        write_step(self.root / "f32" / "step0000", [item("long", 0, list(range(10)), long)],
                   {"enc.vision.out": [(0, f32, "f32")], "L00.in": [(0, np.ones((10, COLS)), "f32")]},
                   encoder={"enc.vision.out": [{"request": "long", "start": 0, "kind": "image", "item": 0, "index": 0, "rows": 6}]})
        write_step(self.root / "engine" / "step0000", [item(16, 0, list(range(4)), long[:4])],
                   {"enc.vision.out": [(0, f32[:2], "f32")], "L00.in": [(0, np.ones((4, COLS)), "f32")]},
                   encoder={"enc.vision.out": [{"request": 16, "start": 0, "kind": "image", "item": 0, "index": 0, "rows": 2}]})
        wrong = f32[2:].copy()
        wrong[3] *= 2
        write_step(self.root / "engine" / "step0001", [item(16, 0, list(range(4, 10)), long[4:])],
                   {"enc.vision.out": [(0, wrong, "f32")], "L00.in": [(0, np.ones((6, COLS)), "f32")]},
                   encoder={"enc.vision.out": [{"request": 16, "start": 0, "kind": "image", "item": 0, "index": 2, "rows": 4}]})
        receipt = self.root / "encoder.json"
        code = tapdiff.main(["--engine", str(self.root / "engine"), "--f32", str(self.root / "f32"),
                             "--json", str(receipt)])
        report = json.loads(receipt.read_text())
        vision = report["taps"][0]
        self.assertEqual((code, vision["tap"], vision["rows"], vision["status"]), (0, "enc.vision.out", 6, "ok"))
        self.assertEqual(report["first_exceeding"], {})
        self.assertEqual(report["encoder_rows"], {"enc.vision.out": {"reference": 6, "exceeding": 1}})
        self.assertEqual(vision["e"]["max"], 1.0)

    def test_an_encoder_reference_without_positions_is_named_by_request(self):
        """tools.refrun.encoder's steps hold a sequence's encoder rows and no
        positions: --request names the sequence, and only a sequence the
        reference holds."""
        f32 = self.rng.standard_normal((6, COLS)).astype(np.float32)
        segment = {"start": 0, "kind": "image", "item": 0, "index": 0, "rows": 6}
        write_step(self.root / "f32" / "step0000", [{"request": "long", "start": 0, "rows": 0, "prefix": 0,
                                                     "positions": [], "tokens": []}],
                   {"enc.vision.out": [(0, f32, "f32")]},
                   encoder={"enc.vision.out": [{"request": "long", **segment}]})
        write_step(self.root / "engine" / "step0000", [item(16, 0, list(range(10)), self.sequences["long"])],
                   {"enc.vision.out": [(0, f32, "f32")]},
                   encoder={"enc.vision.out": [{"request": 16, **segment}]})
        common = ["--engine", str(self.root / "engine"), "--f32", str(self.root / "f32")]
        with self.assertRaisesRegex(ValueError, "no engine request matches"):
            tapdiff.run(tapdiff.parse_args(common))
        with self.assertRaisesRegex(ValueError, "does not hold"):
            tapdiff.run(tapdiff.parse_args(common + ["--request", "16=short"]))
        receipt = self.root / "named.json"
        code = tapdiff.run(tapdiff.parse_args(common + ["--request", "16=long", "--json", str(receipt)]))
        report = json.loads(receipt.read_text())
        self.assertEqual((code, report["status"], report["requests"]), (0, "match", {"16": "long"}))
        self.assertEqual(report["encoder_rows"], {"enc.vision.out": {"reference": 6, "exceeding": 0}})

    def test_encoder_rows_are_counted_apart_from_positions(self):
        """Every position and one encoder row exceed: the positions count
        only positions, the encoder tap its own rows."""
        f32 = self.rng.standard_normal((6, COLS)).astype(np.float32)
        long = self.sequences["long"]
        write_step(self.root / "f32" / "step0000", [item("long", 0, list(range(10)), long)],
                   {"enc.vision.out": [(0, f32, "f32")], "L00.in": [(0, np.ones((10, COLS)), "f32")]},
                   encoder={"enc.vision.out": [{"request": "long", "start": 0, "kind": "image", "item": 0, "index": 0, "rows": 6}]})
        wrong = f32.copy()
        wrong[3] *= 2
        write_step(self.root / "engine" / "step0000", [item(16, 0, list(range(10)), long)],
                   {"enc.vision.out": [(0, wrong, "f32")], "L00.in": [(0, np.full((10, COLS), 2.0), "f32")]},
                   encoder={"enc.vision.out": [{"request": 16, "start": 0, "kind": "image", "item": 0, "index": 0, "rows": 6}]})
        receipt = self.root / "counts.json"
        with contextlib.redirect_stdout(io.StringIO()) as printed:
            tapdiff.main(["--engine", str(self.root / "engine"), "--f32", str(self.root / "f32"),
                          "--json", str(receipt)])
        report = json.loads(receipt.read_text())
        self.assertEqual(report["positions"], {"reference": 10, "with_exceeding_tap": 10})
        self.assertEqual(report["first_exceeding"], {"L00.in": 10})
        self.assertEqual(report["encoder_rows"], {"enc.vision.out": {"reference": 6, "exceeding": 1}})
        self.assertIn("(10 of 10 positions exceed somewhere)", printed.getvalue())
        self.assertIn("enc.vision.out: 1 of 6 encoder rows exceed", printed.getvalue())

    def test_encoder_rows_align_by_kind_and_item(self):
        """The reference stores a request's images, then its video; the engine
        stores its image, video, image in prompt order. Rows pair by (kind,
        item, row), so the order never pairs other items' rows; mislabeled
        items diverge, and a missing kind or unlabeled segment is
        incomplete."""
        f32 = self.rng.standard_normal((7, COLS)).astype(np.float32)
        long = self.sequences["long"]
        rows = {"image 0": (0, 2), "image 1": (2, 4), "video 0": (4, 7)}
        reference = [{"request": "long", "start": a, "kind": label.split()[0], "item": int(label.split()[1]),
                      "index": 0, "rows": b - a} for label, (a, b) in rows.items()]

        def engine(order, labels=None):
            values, segments, start = [], [], 0
            for label in order:
                a, b = rows[label]
                kind, item = (labels or {}).get(label, label).split()
                values.append(f32[a:b])
                segments.append({"request": 16, "start": start, "kind": kind, "item": int(item),
                                 "index": 0, "rows": b - a})
                start += b - a
            return np.concatenate(values), segments
        prompt = ("image 0", "video 0", "image 1")
        cases = {"prompt order": (engine(prompt), (0, "match")),
                 "mislabeled": (engine(prompt, {"image 0": "image 1", "image 1": "image 0"}), (1, "diverged")),
                 "no video": (engine(("image 0", "image 1")), (3, "incomplete"))}
        unlabeled = engine(prompt)
        del unlabeled[1][1]["kind"]
        cases["unlabeled"] = (unlabeled, (3, "incomplete"))
        for case, ((values, segments), expected) in cases.items():
            base = self.root / case.replace(" ", "-")
            write_step(base / "f32" / "step0000", [item("long", 0, list(range(10)), long)],
                       {"enc.vision.out": [(0, f32, "f32")], "L00.in": [(0, np.ones((10, COLS)), "f32")]},
                       encoder={"enc.vision.out": reference})
            write_step(base / "engine" / "step0000", [item(16, 0, list(range(10)), long)],
                       {"enc.vision.out": [(0, values, "f32")], "L00.in": [(0, np.ones((10, COLS)), "f32")]},
                       encoder={"enc.vision.out": segments})
            with contextlib.redirect_stdout(io.StringIO()):
                code = tapdiff.main(["--engine", str(base / "engine"), "--f32", str(base / "f32"),
                                     "--json", str(base / "receipt.json")])
            report = json.loads((base / "receipt.json").read_text())
            self.assertEqual((code, report["status"]), expected, case)
            if case == "prompt order":
                self.assertEqual(report["taps"][0]["rows"], 7)
                self.assertEqual(report["taps"][0]["e"]["max"], 0.0)

    def test_encoder_taps_need_every_reference_row(self):
        """Negative controls: the engine's encoder rows must be the reference's,
        mapped one segment row per stored row."""
        f32 = np.arange(1, 25, dtype=np.float32).reshape(6, 4)
        whole = [{"request": "seq", "start": 0, "kind": "image", "item": 0, "index": 0, "rows": 6}]

        def engine(case):
            """The engine's vision rows and segments for one case."""
            values, segments = f32.copy(), [dict(segment) for segment in whole]
            if case == "missing suffix":
                values, segments[0]["rows"] = f32[:2], 2
            elif case == "gap":
                values = f32[[0, 1, 4, 5]]
                segments = [{"request": "seq", "start": 0, "kind": "image", "item": 0, "index": 0, "rows": 2},
                            {"request": "seq", "start": 2, "kind": "image", "item": 0, "index": 4, "rows": 2}]
            elif case == "disjoint indices":
                segments[0]["index"] = 6
            elif case == "reversed":
                values = f32[::-1].copy()
            elif case == "short payload":
                values = f32[:2]
            elif case == "physical holes":
                values[1:5] *= 100
                segments = [{"request": "seq", "start": 0, "kind": "image", "item": 0, "index": 0, "rows": 1},
                            {"request": "seq", "start": 5, "kind": "image", "item": 0, "index": 5, "rows": 1}]
            elif case == "overlap":
                segments = [{"request": "seq", "start": 0, "kind": "image", "item": 0, "index": 0, "rows": 4},
                            {"request": "seq", "start": 3, "kind": "image", "item": 0, "index": 3, "rows": 3}]
            return values, segments
        outcomes = {"full": (0, "match"), "missing suffix": (3, "incomplete"), "gap": (3, "incomplete"),
                    "disjoint indices": (2, "incompatible"), "reversed": (1, "diverged"),
                    "short payload": (3, "incomplete"), "physical holes": (3, "incomplete"),
                    "overlap": (3, "incomplete")}
        for case, expected in outcomes.items():
            base = self.root / case.replace(" ", "-")
            for arm in ("engine", "f32", "bf16"):
                values, segments = engine(case) if arm == "engine" else (f32, whole)
                tensors = {"enc.vision.out": [(0, values, "f32")], "L00.in": [(0, np.ones((4, 4)), "f32")]}
                write_step(base / arm / "step0000", [item("seq", 0, [0, 1, 2, 3], [10, 11, 12, 13])], tensors,
                           expected=[{"name": name, "rank": 0} for name in tensors],
                           encoder={"enc.vision.out": segments})
            code = tapdiff.main(["--engine", str(base / "engine"), "--f32", str(base / "f32"), "--bf16",
                                 str(base / "bf16"), "--json", str(base / "receipt.json")])
            report = json.loads((base / "receipt.json").read_text())
            self.assertEqual((code, report["status"]), expected, case)
            counts = {"missing suffix": (2, 4, 0), "gap": (4, 2, 0), "disjoint indices": (6, 6, 6)}.get(case)
            if counts:
                self.assertEqual(report["taps"][0]["encoder_rows"], {"expected": 6, **dict(zip(
                    ("observed", "missing", "unexpected"), counts))}, case)

    def test_qwen_names(self):
        qwen = tap_names.v1_declaration("chopinfer.qwen38.capture.v1")
        name = lambda v1: qwen.canonical(v1) and qwen.canonical(v1)[0]
        self.assertEqual(name("layer03.attn.partial"), "L03.attn.out")
        self.assertEqual(name("layer03.in.x"), "L03.in.streams")
        self.assertEqual(name("embed"), "embed.out")
        self.assertEqual(qwen.canonical("layer07.moe.weights")[1]["compare"], "paired")
        # The PLE layer's input is X before the PLE add; its in.x is after it.
        self.assertEqual(name("layer01.ple.in"), "L01.in.streams")
        self.assertEqual(name("layer01.ple.out"), "L01.x.ple.out")
        self.assertEqual(name("layer01.in.x"), "L01.x.hc.attn.streams")
        # Linear and full attention name their o_proj input alike.
        self.assertEqual((name("layer02.lin.normed"), name("layer03.attn.gated")), ("L02.attn.core", "L03.attn.core"))
        self.assertEqual(name("layer03.attn.selected"), "L03.x.indexer.topk")
        self.assertIsNone(qwen.canonical("layer03.attn.renamed"))
        self.assertIsNone(tap_names.v1_declaration("chopinfer.taps.v1"))

    def test_declared_names_follow_the_runner_grammar(self):
        """Every model's declared names are runner tap names."""
        try:
            from tools.refrun.taps import NAME
        except ModuleNotFoundError:
            self.skipTest("needs torch")
        declarations = tap_names.declarations()
        # A model whose reference adapter has a tap map (`taps(model)`, which
        # tools/refrun/run.py writes) declares its taps, released or not.
        tapped = set()
        for model in inventory.shipped():
            adapter = inventory.ROOT / "models" / model / "reference.py"
            if adapter.is_file() and any(isinstance(node, ast.FunctionDef) and node.name == "taps"
                                         for node in ast.parse(adapter.read_text()).body):
                tapped.add(model)
        self.assertTrue(tapped)
        self.assertLessEqual(tapped, {d.model for d in declarations})
        # Each declaration sits in its own model's directory.
        self.assertEqual([d.model for d in declarations], [d.path.parent.name for d in declarations])
        self.assertLessEqual({d.model for d in declarations}, set(inventory.shipped()))
        for declaration in declarations:
            for _, _, entry in declaration.entries:
                name = entry["name"].replace("{nn}", "03")
                self.assertIsNotNone(NAME.fullmatch(name), (declaration.model, name))

    def test_ambiguous_requests_need_a_name(self):
        self.sequences = {"a": [1, 2, 3], "b": [1, 2, 4]}
        f32 = {"L00.in": self.values()}
        engine_dir = self.root / "engine"
        write_step(engine_dir / "step0000", [item(5, 0, [0, 1], [1, 2])],
                   {"L00.in": [(0, f32["L00.in"]["a"][:2], "f32")]})
        common = ["--engine", str(engine_dir), "--f32", str(self.reference("f32", f32))]
        with self.assertRaisesRegex(ValueError, "use --request"):
            tapdiff.run(tapdiff.parse_args(common))
        self.assertEqual(tapdiff.run(tapdiff.parse_args(common + ["--request", "5=a"])), 0)
        with self.assertRaisesRegex(ValueError, "does not hold"):
            tapdiff.run(tapdiff.parse_args(common + ["--request", "5=c"]))
        write_step(self.root / "engine2" / "step0000", [item(6, 0, [0, 1, 2], [1, 2, 3])],
                   {"L00.in": [(0, f32["L00.in"]["a"], "f32")]})
        with self.assertRaisesRegex(ValueError, "disagrees"):
            tapdiff.run(tapdiff.parse_args(["--engine", str(self.root / "engine2"), "--f32", common[3],
                                            "--request", "6=b"]))


ROWS, EXPERTS, TOP_K, LAYERS = 8, 16, 2, ("L00", "L01")


@unittest.skipIf(np is None, "needs numpy (set CHOPINFER_REFRUN_PYTHON)")
class RouteTests(unittest.TestCase):
    """route-near-tie.v2 (--router): a tap that fails only its aggregate
    passes when the fewest rows behind a near-tie routing flip, each within
    the BF16 run's worst row times the ratio, explain it. Two MoE layers of
    16 experts, top 2, over one 8-token sequence; row 1 carries ten times the
    others' norm, so its error dominates an aggregate, and the BF16 run
    routes row 5 differently at L00 (a near tie of its own) and is far off
    there afterwards, as a BF16 run's own swapped rows are."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR"))
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        quiet = contextlib.redirect_stdout(io.StringIO())
        quiet.__enter__()
        self.addCleanup(quiet.__exit__, None, None, None)

    def build(self, flips=(), errors=(), ties=((0, 1), (0, 5)), score="sigmoid", bias_nan=False, bf16_far=(),
              replace=(), bf16_worst=0.1):
        """The engine, f32 and bf16 tap directories and the bias sidecar.
        `flips` {(layer, row): "near" | "far"}: the engine drops F32's 2nd
        expert for its 3rd (a near tie where `ties` made them 1e-5 apart) or
        its last. `errors` {(tap, row): relative error} of the engine's dense
        rows. `bf16_far` (layer, row): the BF16 run's boundary moved by 0.1
        there. `replace` {(tap, row): f(f32 row)}: the engine's row instead.
        `bf16_worst`: the BF16 run's error at row 5 of each mlp.out tap, its
        worst row there."""
        rng = np.random.default_rng(11)
        tokens = list(range(200, 200 + ROWS))
        flips, errors, replace = dict(flips), dict(errors), dict(replace)
        modes = {"engine": {}, "f32": {}, "bf16": {}}
        biases = {}
        for layer, label in enumerate(LAYERS):
            ranks = np.stack([rng.permutation(EXPERTS) for _ in range(ROWS)])
            target = np.empty((ROWS, EXPERTS))
            for row in range(ROWS):
                target[row, ranks[row]] = 0.9 - 0.04 * np.arange(EXPERTS)
                if (layer, row) in ties:
                    target[row, ranks[row, TOP_K]] = target[row, ranks[row, TOP_K - 1]] - 1e-5
            if score == "sigmoid":
                bias = rng.uniform(0.0, 0.2, EXPERTS)
                if bias_nan:
                    bias[3] = np.nan
                biases[int(label[1:])] = bias
                share = target - np.nan_to_num(bias)
                logits = np.log(share / (1 - share))
            else:
                bias, logits = None, np.log(target)
            noisy = logits + rng.normal(0.0, 2e-4, logits.shape)
            for row in range(ROWS):
                second, third = ranks[row, TOP_K - 1], ranks[row, TOP_K]
                if row == 5:
                    # The BF16 run reroutes row 5 at every layer: its 3rd
                    # expert's score just above its 2nd's.
                    above = tapdiff.selection_scores({"score": score}, noisy[row:row + 1], bias)[0, second] + 1e-5
                    noisy[row, third] = (np.log(above - bias[third]) - np.log(1 - above + bias[third])
                                         if score == "sigmoid" else noisy[row, second] + 1e-4)
                elif (layer, row) in ties:
                    noisy[row, [second, third]] = logits[row, [second, third]]
                if (layer, row) in bf16_far:
                    noisy[row, second] += 0.5
            choose = lambda values: np.argsort(-tapdiff.selection_scores({"score": score}, values, bias),
                                               kind="stable")[:, :TOP_K]
            ids = {"f32": choose(logits), "bf16": choose(noisy)}
            engine = ids["f32"].copy()
            for (where, row), kind in flips.items():
                if where == layer:
                    added = ranks[row, TOP_K] if kind == "near" else ranks[row, -1]
                    engine[row] = [added if expert == ranks[row, TOP_K - 1] else expert for expert in engine[row]]
            ids["engine"] = engine
            dense = {}
            for tap in (f"{label}.x.hc.mlp", f"{label}.mlp.out"):
                f32 = rng.standard_normal((ROWS, COLS))
                f32 /= np.linalg.norm(f32, axis=1, keepdims=True)
                f32[1] *= 10

                def off(scale, rows=f32):
                    noise = rng.standard_normal(rows.shape)
                    noise /= np.linalg.norm(noise, axis=1, keepdims=True)
                    return rows + scale * np.linalg.norm(rows, axis=1, keepdims=True) * noise
                bf16 = off(np.full((ROWS, 1), 4e-3))
                if tap.endswith("mlp.out") or layer:
                    bf16[5] = off(np.full((1, 1), bf16_worst if tap.endswith("mlp.out") else 0.1), f32[5:6])[0]
                scales = np.full((ROWS, 1), 1e-3)
                for (name, row), value in errors.items():
                    if name == tap:
                        scales[row] = value
                engine = off(scales)
                for (name, row), make in replace.items():
                    if name == tap:
                        engine[row] = make(f32[row])
                dense[tap] = {"f32": f32, "bf16": bf16, "engine": engine}
            for mode in modes:
                taps = modes[mode]
                taps[f"{label}.x.hc.mlp"] = dense[f"{label}.x.hc.mlp"][mode]
                taps[f"{label}.router.logits"] = noisy if mode == "bf16" else logits
                taps[f"{label}.router.ids"] = ids[mode]
                taps[f"{label}.mlp.out"] = dense[f"{label}.mlp.out"][mode]
        for mode, taps in modes.items():
            write_step(self.root / mode / "step0000", [item("s", 0, list(range(ROWS)), tokens)],
                       {name: [(0, values.astype(np.float32 if not name.endswith(".ids") else np.uint32),
                                "u32" if name.endswith(".ids") else "f32")] for name, values in taps.items()})
        sidecar = {"schema": "chopinfer.router-bias.v1", "model": "m", "experts": EXPERTS,
                   "layers": {str(layer): base64.b64encode(np.asarray(values, "<f4").tobytes()).decode()
                              for layer, values in biases.items()}}
        if biases:
            (self.root / "bias.json").write_text(json.dumps(sidecar))
        return score

    def diff(self, score="sigmoid", router=True, bias=True, spec=None):
        receipt = self.root / f"receipt{len(list(self.root.glob('receipt*')))}.json"
        argv = ["--engine", str(self.root / "engine"), "--f32", str(self.root / "f32"),
                "--bf16", str(self.root / "bf16"), "--aggregate", "*", "--json", str(receipt)]
        if router:
            spec = spec or {"score": score, "top_k": TOP_K, **({"bias": "bias"} if score == "sigmoid" else {})}
            argv += ["--router", json.dumps(spec)]
            if bias and score == "sigmoid":
                argv += ["--router-bias", str(self.root / "bias.json")]
        code = tapdiff.run(tapdiff.parse_args(argv))
        report = json.loads(receipt.read_text())
        return code, report, {row["tap"]: row for row in report["taps"]}

    def test_a_near_tie_flip_explains_the_aggregate_it_dominates(self):
        self.build(flips={(0, 1): "near"}, errors={("L00.mlp.out", 1): 0.05, ("L01.x.hc.mlp", 1): 0.05})
        # Without route judging, row 1 fails both aggregates.
        code, report, taps = self.diff(router=False)
        self.assertEqual((code, report["first"]), (1, "L00.mlp.out"))
        self.assertTrue(taps["L01.x.hc.mlp"]["fail"])
        code, report, taps = self.diff()
        self.assertEqual((code, report["status"], report["first"]), (0, "match", None))
        routes = report["routes"]
        self.assertEqual((routes["status"], routes["rule"]), ("judged", "route-near-tie.v2"))
        self.assertEqual(routes["first_flips"], {"near_tie": 1, "decisive": 0})
        self.assertEqual(routes["bf16_first_flips_beyond_margin"], [])
        flip = routes["flips"][0]
        self.assertEqual((flip["layer"], flip["position"], flip["near_tie"]), ("L00", 1, True))
        self.assertLess(flip["gap"], flip["margin"])
        for name in ("L00.mlp.out", "L01.x.hc.mlp"):
            explanation = taps[name]["route_explanation"]
            self.assertTrue(explanation["explained"])
            self.assertFalse(taps[name]["fail"])
            self.assertFalse(taps[name]["aggregate"]["passed"])
            self.assertEqual([row["position"] for row in explanation["removed"]], [1])
            self.assertLessEqual(explanation["removed"][0]["e"], explanation["row_bound"])
            self.assertTrue(explanation["aggregate"]["passed"])
        # The row before the router is not behind the flip.
        self.assertNotIn("route_explanation", taps["L00.x.hc.mlp"])

    def test_the_same_flip_at_a_large_gap_still_fails(self):
        self.build(flips={(0, 1): "far"}, errors={("L00.mlp.out", 1): 0.05})
        code, report, taps = self.diff()
        self.assertEqual((code, report["first"]), (1, "L00.mlp.out"))
        self.assertEqual(report["routes"]["first_flips"], {"near_tie": 0, "decisive": 1})
        self.assertGreater(report["routes"]["flips"][0]["gap"], report["routes"]["flips"][0]["margin"])
        self.assertNotIn("route_explanation", taps["L00.mlp.out"])

    def test_a_dense_excess_without_a_flip_still_fails(self):
        self.build(errors={("L00.mlp.out", 1): 0.05})
        code, report, taps = self.diff()
        self.assertEqual((code, report["first"]), (1, "L00.mlp.out"))
        self.assertEqual(report["routes"]["flips"], [])
        self.assertNotIn("route_explanation", taps["L00.mlp.out"])

    def test_a_near_tie_row_beyond_its_row_bound_fails(self):
        self.build(flips={(0, 1): "near"}, errors={("L00.mlp.out", 1): 0.5})
        code, report, taps = self.diff()
        self.assertEqual((code, report["first"]), (1, "L00.mlp.out"))
        explanation = taps["L00.mlp.out"]["route_explanation"]
        self.assertFalse(explanation["explained"])
        self.assertIn("per-row bound", explanation["reason"])
        self.assertGreater(explanation["removed"][0]["e"], explanation["row_bound"])

    def test_a_median_failure_is_never_rejudged(self):
        self.build(flips={(0, 1): "near"}, errors={("L00.mlp.out", row): 0.02 for row in range(ROWS)})
        code, report, taps = self.diff()
        self.assertEqual((code, report["first"]), (1, "L00.mlp.out"))
        self.assertGreater(taps["L00.mlp.out"]["excess"]["median"], 1)
        self.assertNotIn("route_explanation", taps["L00.mlp.out"])

    def test_without_biases_or_with_nonfinite_scores_every_flip_is_decisive(self):
        self.build(flips={(0, 1): "near"}, errors={("L00.mlp.out", 1): 0.05})
        code, report, _ = self.diff(bias=False)
        self.assertEqual((code, report["routes"]["status"]), (1, "unjudged"))
        self.assertIn("--router-bias", report["routes"]["reason"])
        code, report, _ = self.diff(spec={"score": "sigmoid", "top_k": TOP_K, "bias": "bias", "n_group": 8,
                                          "topk_group": 4})
        self.assertEqual((code, report["routes"]["status"]), (1, "unjudged"))
        self.assertIn("router.groups", report["routes"]["reason"])
        shutil.rmtree(self.root)
        self.root.mkdir()
        self.build(flips={(0, 1): "near"}, errors={("L00.mlp.out", 1): 0.05}, bias_nan=True)
        code, report, _ = self.diff()
        self.assertEqual((code, report["routes"]["status"]), (1, "unjudged"))
        self.assertIn("nonfinite", report["routes"]["reason"])

    def test_a_rows_first_flip_decides_it(self):
        # A near-tie first flip keeps its row behind it through a later
        # large-gap flip.
        self.build(flips={(0, 1): "near", (1, 1): "far"}, errors={("L01.mlp.out", 1): 0.05},
                   ties=((0, 1), (0, 5), (1, 1)))
        code, report, taps = self.diff()
        self.assertEqual(code, 0)
        later = [flip for flip in report["routes"]["flips"] if flip["layer"] == "L01"]
        self.assertEqual([(flip["first"], flip["near_tie"]) for flip in later], [(False, None)])
        self.assertTrue(taps["L01.mlp.out"]["route_explanation"]["explained"])
        # A decisive first flip keeps today's rules through a later near tie.
        shutil.rmtree(self.root)
        self.root.mkdir()
        self.build(flips={(0, 1): "far", (1, 1): "near"}, errors={("L01.mlp.out", 1): 0.05},
                   ties=((0, 1), (0, 5), (1, 1)))
        code, report, taps = self.diff()
        self.assertEqual((code, report["first"]), (1, "L01.mlp.out"))
        self.assertNotIn("route_explanation", taps["L01.mlp.out"])

    def test_a_softmax_router(self):
        self.build(flips={(0, 1): "near"}, errors={("L00.mlp.out", 1): 0.05}, score="softmax")
        code, report, taps = self.diff(score="softmax")
        self.assertEqual((code, report["routes"]["first_flips"]["near_tie"]), (0, 1))
        self.assertTrue(taps["L00.mlp.out"]["route_explanation"]["explained"])

    def test_the_spread_counts_only_rows_the_bf16_run_has_not_rerouted(self):
        # Row 5, rerouted by the BF16 run at L00, moves 0.1 at L01's
        # boundary: left out of the spread, which stays the rounding one.
        self.build(bf16_far=((1, 5),))
        _, report, _ = self.diff()
        layers = report["routes"]["layers"]
        self.assertEqual((layers["L00"]["rounding_only_rows"], layers["L01"]["rounding_only_rows"]), (ROWS, ROWS - 1))
        self.assertEqual(layers["L00"]["bf16_flips"], 1)
        self.assertLess(layers["L01"]["margin"], 1e-3)
        self.assertEqual(layers["L01"]["margin"], 3.0 * max(layers["L00"]["spread"], layers["L01"]["spread"]))


    def flagged(self, make, bf16_worst):
        """The verdict at L00.mlp.out when the engine's row 1, behind a
        near-tie flip at L00's router, is make(F32's row)."""
        shutil.rmtree(self.root)
        self.root.mkdir()
        self.build(flips={(0, 1): "near"}, replace={("L00.mlp.out", 1): make}, bf16_worst=bf16_worst)
        code, report, taps = self.diff()
        return code, report, taps["L00.mlp.out"]

    def test_a_destroyed_row_at_a_near_tie_flip_still_fails(self):
        # The negative control of a real layer dump: a zeroed row has e
        # 1.0, inside the row bound of three times the BF16 run's worst
        # row (a rerouted row at 0.3637), but no longer resembles the
        # reference row, so it is not excused.
        code, report, row = self.flagged(np.zeros_like, 0.3637)
        self.assertEqual((code, report["first"]), (1, "L00.mlp.out"))
        explanation = row["route_explanation"]
        self.assertFalse(explanation["explained"])
        self.assertIn("plausibility", explanation["reason"])
        removed = explanation["removed"][0]
        self.assertEqual((removed["position"], removed["norm_ratio"], removed["plausible"]), (1, 0.0, False))
        self.assertAlmostEqual(removed["e"], 1.0, places=6)
        self.assertLessEqual(removed["e"], explanation["row_bound"])
        self.assertAlmostEqual(explanation["row_bound"], 3 * 0.3637, places=3)

    def test_a_nonfinite_row_is_not_excused(self):
        def poisoned(values):
            values = values.copy()
            values[3] = np.nan
            return values
        shutil.rmtree(self.root)
        self.root.mkdir()
        self.build(flips={(0, 1): "near"}, replace={("L00.mlp.out", 1): poisoned}, bf16_worst=0.7)
        # The receipt's aggregate error is inf, which JSON cannot hold: the
        # verdict is read from the report before it is written.
        with mock.patch.object(tapdiff.diagnostic, "write_new") as write:
            argv = ["--engine", str(self.root / "engine"), "--f32", str(self.root / "f32"),
                    "--bf16", str(self.root / "bf16"), "--aggregate", "*", "--json", str(self.root / "receipt.json"),
                    "--router", json.dumps({"score": "sigmoid", "top_k": TOP_K, "bias": "bias"}),
                    "--router-bias", str(self.root / "bias.json")]
            code = tapdiff.run(tapdiff.parse_args(argv))
        report = write.call_args.args[1]
        row = next(row for row in report["taps"] if row["tap"] == "L00.mlp.out")
        self.assertEqual((code, report["first"]), (1, "L00.mlp.out"))
        explanation = row["route_explanation"]
        self.assertFalse(explanation["explained"])
        self.assertIn("plausibility", explanation["reason"])
        self.assertEqual((explanation["removed"][0]["norm_ratio"], explanation["removed"][0]["cosine"]), (None, None))

    def test_a_row_norm_must_be_within_the_ratio_of_the_reference_rows(self):
        # Inside D3's row bound (2.1) throughout: only the norm decides.
        for scale, explained in ((2.99, True), (3.01, False), (1 / 2.99, True), (1 / 3.01, False)):
            with self.subTest(scale=scale):
                code, _, row = self.flagged(lambda values: scale * values, 0.7)
                explanation = row["route_explanation"]
                removed = explanation["removed"][0]
                self.assertLessEqual(removed["e"], explanation["row_bound"])
                self.assertAlmostEqual(removed["norm_ratio"], scale, places=5)
                self.assertGreater(removed["cosine"], 0.999)
                self.assertEqual((code, explanation["explained"], removed["plausible"]),
                                 (0 if explained else 1, explained, explained))

    def test_a_row_must_correlate_positively_with_the_reference_row(self):
        code, _, row = self.flagged(np.negative, 0.7)
        explanation = row["route_explanation"]
        removed = explanation["removed"][0]
        self.assertLessEqual(removed["e"], explanation["row_bound"])
        self.assertEqual((code, explanation["explained"], removed["plausible"]), (1, False, False))
        self.assertAlmostEqual(removed["norm_ratio"], 1.0, places=5)
        self.assertAlmostEqual(removed["cosine"], -1.0, places=5)

    def test_the_plausibility_guard_at_its_boundaries(self):
        unit = np.eye(4)
        cases = {
            "the same row": (unit[0], unit[0], True, 1.0, 1.0),
            "three times the norm": (3 * unit[0], unit[0], True, 3.0, 1.0),
            "a third of the norm": (unit[0], 3 * unit[0], True, 1 / 3, 1.0),
            "just above three times": (np.nextafter(3.0, 4.0) * unit[0], unit[0], False, None, 1.0),
            "orthogonal": (unit[1], unit[0], False, 1.0, 0.0),
            "barely positive": (unit[1] + 1e-12 * unit[0], unit[0], True, None, None),
            "negative": (-unit[0], unit[0], False, 1.0, -1.0),
            "zero": (0 * unit[0], unit[0], False, 0.0, None),
            "infinite": (np.array([np.inf, 0, 0, 0]), unit[0], False, None, None),
            "a zero reference": (unit[0], 0 * unit[0], False, None, None),
        }
        for name, (values, reference, passed, ratio, cosine) in cases.items():
            with self.subTest(name):
                guard = tapdiff.plausible(values, reference, 3.0)
                self.assertIs(guard["plausible"], passed)
                if ratio is not None:
                    self.assertAlmostEqual(guard["norm_ratio"], ratio)
                if cosine is not None:
                    self.assertEqual(guard["cosine"], cosine)
        self.assertEqual(tapdiff.plausible(np.array([np.nan, 0.0]), np.ones(2), 3.0),
                         {"plausible": False, "norm_ratio": None, "cosine": None})

if __name__ == "__main__":
    unittest.main()
