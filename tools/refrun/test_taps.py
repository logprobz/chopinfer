"""Runner taps: names, row selection, the written format, and taps that
leave scores unchanged. The tiny model runs on CPU; skipped without torch."""

import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest

from tools.refrun.test_runner import keys_root, write_workload

try:
    import numpy as np
    import torch
    from tools.qual import tapdiff
    from tools.refrun import modes, run, taps, tiny
except ModuleNotFoundError:
    torch = None

SEQUENCES = [{"name": "text", "category": "prose", "token_ids": [1, 5, 9, 33, 2, 7, 40, 41], "score_from": 1},
             {"name": "answer", "category": "probe", "token_ids": [3, 4, 5], "score_from": 3,
              "answer": {"policy": "forced", "ids": [17, 18, 19]}}]


def setUpModule():
    if torch is not None:
        modes.prepare_environment(cuda=False)


@unittest.skipIf(torch is None, "needs torch (set CHOPINFER_REFRUN_PYTHON)")
class NameTests(unittest.TestCase):
    def test_grammar(self):
        for name, expected in (("L03.attn.out", "attn.out"), ("embed.out", "out"), ("final.norm", "norm"),
                               ("L05.x.hc.pre", "x.hc.pre"), ("enc.vision.B07.out", "out"),
                               ("mtp.L0.in.streams", "in.streams"), ("L20.x.indexer.topk", "x.indexer.topk")):
            self.assertEqual(taps.site(name), expected, name)
        for name in ("L3.attn.out", "layer03.out", "L03.attn", "L03.x", "L03.attn.out.y", "final"):
            with self.assertRaises(ValueError, msg=name):
                taps.site(name)

    def test_rows(self):
        self.assertEqual(taps.rows(5), [0, 1, 2, 3, 4])
        chosen = taps.rows(3000)
        self.assertEqual(chosen[:2048], list(range(2048)))
        self.assertEqual(chosen[2048:2051], [2048, 2056, 2064])
        self.assertEqual(chosen[-17:], [2976] + list(range(2984, 3000)))
        self.assertEqual(len(chosen), 2048 + (2976 - 2048) // 8 + 1 + 16)
        self.assertEqual(taps.rows(10, [7, 2, 7]), [2, 7])
        with self.assertRaisesRegex(ValueError, "outside"):
            taps.rows(10, [10])


@unittest.skipIf(torch is None, "needs torch (set CHOPINFER_REFRUN_PYTHON)")
class TapRunTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR"))
        root = Path(cls.directory.name)
        cls.checkpoint = tiny.write_checkpoint(root / "checkpoint")
        cls.workload = write_workload(root, SEQUENCES)

        def runner(name, *extra):
            run.run(run.parse_args([
                "--adapter", str(keys_root() / "tools/refrun/tiny.py"), "--checkpoint", str(cls.checkpoint),
                "--lock", str(cls.checkpoint / tiny.LOCK), "--workload", str(cls.workload),
                "--modes", "f32,bf16", "--devices", "cpu", "--output", str(root / name),
                "--cache", str(root / "cache"), *extra]))
            return root / name
        cls.runner = staticmethod(runner)
        cls.plain = runner("plain")
        cls.tapped = runner("tapped", "--taps", "--tap-sites", "all")

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def read(self, mode, step, name):
        directory = self.tapped / "taps" / mode / f"step{step:04d}"
        entry = next(e for e in map(json.loads, (directory / "index.jsonl").read_text().splitlines())
                     if e["name"] == name)
        return tapdiff.read(directory / f"{name}.r0.bin", entry)

    def test_taps_leave_scores_unchanged(self):
        for mode in ("f32", "bf16"):
            self.assertEqual((self.tapped / f"{mode}.json").read_bytes(), (self.plain / f"{mode}.json").read_bytes())
        self.assertFalse((self.plain / "taps").exists())
        # Taps are not cached: the tapped run computes every pass the plain run cached.
        for record in json.loads((self.tapped / "run.json").read_text())["modes"].values():
            self.assertEqual((record["computed"], record["cache_hits"]), (record["entries"], 0))

    def test_format_and_program_order(self):
        step = self.tapped / "taps" / "f32" / "step0001"
        index = [json.loads(line) for line in (step / "index.jsonl").read_text().splitlines()]
        names = [e["name"] for e in index]
        self.assertEqual(names[:9], ["embed.out", "L00.in", "L00.attn.in", "L00.attn.q", "L00.attn.core",
                                     "L00.attn.out", "L00.mlp.in", "L00.mlp.out", "L00.out"])
        self.assertEqual(names[-1], "final.norm")
        self.assertEqual(len(names), 2 + 8 * 4)
        self.assertEqual({(e["rank"], e["first_row"], e["rows"], e["dtype"]) for e in index}, {(0, 0, 6, "f32")})
        self.assertEqual(index[0]["cols"], 64)
        items = json.loads((step / "items.json").read_text())
        self.assertEqual(items["items"][0]["request"], "answer")
        self.assertEqual(items["items"][0]["tokens"], [3, 4, 5, 17, 18, 19])
        self.assertEqual(json.loads((step / "completion.json").read_text())["completion"], "full")
        self.assertEqual(json.loads((step / "capture.json").read_text())["schema"], taps.SCHEMA)
        bf16 = [json.loads(line) for line in
                (self.tapped / "taps" / "bf16" / "step0001" / "index.jsonl").read_text().splitlines()]
        self.assertEqual({e["dtype"] for e in bf16}, {"bf16"})
        self.assertEqual(sorted(p.name for p in (self.tapped / "taps" / "f32").iterdir()), ["step0000", "step0001"])

    def test_values_follow_the_residual_stream(self):
        """A layer's output is its input plus the attention and MLP outputs,
        in F32 as HF adds them, and the next layer's input."""
        for layer in range(4):
            x = self.read("f32", 0, f"L{layer:02d}.in")
            out = self.read("f32", 0, f"L{layer:02d}.out")
            attn, mlp = self.read("f32", 0, f"L{layer:02d}.attn.out"), self.read("f32", 0, f"L{layer:02d}.mlp.out")
            self.assertTrue(np.array_equal((x + attn) + mlp, out), layer)
            if layer < 3:
                self.assertTrue(np.array_equal(out, self.read("f32", 0, f"L{layer + 1:02d}.in")))
        self.assertTrue(np.array_equal(self.read("f32", 0, "embed.out"), self.read("f32", 0, "L00.in")))

    def test_row_selection_and_the_diff(self):
        output = self.runner("rows", "--taps", "--tap-rows", "0,2,5", "--tap-sites", "in,out")
        items = json.loads((output / "taps" / "f32" / "step0000" / "items.json").read_text())
        self.assertEqual(items["items"][0]["positions"], [0, 2, 5])
        self.assertEqual(items["items"][0]["tokens"], [1, 9, 7])
        full = self.read("f32", 0, "L02.out")
        entry = {"rows": 3, "cols": 64, "dtype": "f32"}
        self.assertTrue(np.array_equal(tapdiff.read(output / "taps/f32/step0000/L02.out.r0.bin", entry),
                                       full[[0, 2, 5]]))
        # The f32 taps against themselves match; bf16 taps scored against f32 with bf16 as the floor sit at ratio 1.
        receipt = Path(self.directory.name) / "self.json"
        args = ["--engine", str(self.tapped / "taps/bf16"), "--f32", str(self.tapped / "taps/f32"),
                "--bf16", str(self.tapped / "taps/bf16"), "--json", str(receipt)]
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(tapdiff.main(args), 0)
        report = json.loads(receipt.read_text())
        self.assertEqual(report["requests"], {"answer": "answer", "text": "text"})
        dense = [row for row in report["taps"] if row.get("e_over_floor")]
        self.assertTrue(dense and all(row["e_over_floor"]["max"] == 1.0 for row in dense))

    def test_budget_is_refused_and_recorded(self):
        with self.assertRaisesRegex(ValueError, "budget"):
            self.runner("budget", "--taps", "--tap-bytes", "4096")
        completion = Path(self.directory.name) / "budget/taps/f32/step0000/completion.json"
        self.assertEqual(json.loads(completion.read_text())["completion"], "incomplete")
        # The early taps the budget allowed match, but a truncated capture is never a match.
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(tapdiff.main(["--engine", str(completion.parent.parent),
                                           "--f32", str(self.tapped / "taps/f32")]), 3)


@unittest.skipIf(torch is None, "needs torch (set CHOPINFER_REFRUN_PYTHON)")
class EncoderTapTests(unittest.TestCase):
    def test_encoder_rows_join_the_calls_and_follow_the_media(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder, self.layer = torch.nn.Identity(), torch.nn.Identity()

            def forward(self, x, media):
                for item in media:
                    self.encoder(item)
                return self.layer(x)
        model = Model()
        x, image, video = torch.ones((1, 3, 4)), torch.full((2, 4), 2.0), torch.full((3, 4), 3.0)
        with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as directory:
            root = Path(directory)
            tapped = taps.Taps(model, {"enc.vision.out": taps.Hook("encoder", rows=("image", "video")),
                                       "L00.out": "layer"}, root, sites=None)
            with tapped.step("both", [1, 2, 3], media={"image", "video"}):
                model(x, [image, video])
            with tapped.step("text", [1, 2, 3]):
                model(x, [])
            with self.assertRaisesRegex(ValueError, "without image/video media"):
                with tapped.step("unexpected", [1, 2, 3], media={"audio"}):
                    model(x, [image])
            step = root / "step0000"
            index = [json.loads(line) for line in (step / "index.jsonl").read_text().splitlines()]
            self.assertEqual([(e["name"], e["rows"], e.get("axis")) for e in index],
                             [("enc.vision.out", 5, "encoder"), ("L00.out", 3, None)])
            self.assertTrue(np.array_equal(tapdiff.read(step / "enc.vision.out.r0.bin", index[0]),
                                           torch.cat([image, video]).numpy()))
            items = json.loads((step / "items.json").read_text())
            self.assertEqual(items["encoder"], {"enc.vision.out": [
                {"request": "both", "start": 0, "kind": "image", "item": 0, "index": 0, "rows": 2},
                {"request": "both", "start": 2, "kind": "video", "item": 0, "index": 0, "rows": 3}]})
            for name, completion in (("step0000", "full"), ("step0001", "full"), ("step0002", "incomplete")):
                self.assertEqual(json.loads((root / name / "completion.json").read_text())["completion"],
                                 completion, name)


    def test_each_kinds_call_splits_into_its_items(self):
        """An image, a video and an image: the images' call holds both images'
        rows in order, the video's call follows; each call splits into one
        item per row of the grid it receives, by their patches, and each row
        is recorded under its kind and item. A call without a grid is one
        item; shares that do not split a call's rows are refused."""
        class Encoder(torch.nn.Module):
            def forward(self, x, grid_thw=None):
                return x * 2

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.vision, self.audio, self.layer = Encoder(), Encoder(), torch.nn.Identity()

            def forward(self, x, calls, audio=None):
                for pixels, grid in calls:
                    self.vision(pixels, grid_thw=grid)
                if audio is not None:
                    self.audio(audio)
                return self.layer(x)
        model = Model()
        x = torch.ones((1, 3, 4))
        images, video = torch.arange(12.0).reshape(3, 4), torch.full((2, 4), 7.0)
        # Patches: 8 and 4 for the images (2 and 1 rows), 8 for the video.
        grids = torch.tensor([[1, 2, 4], [1, 2, 2]]), torch.tensor([[2, 2, 2]])
        with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as directory:
            root = Path(directory)
            tapped = taps.Taps(model, {"enc.vision.out": taps.Hook("vision", rows=("image", "video")),
                                       "enc.audio.out": taps.Hook("audio", rows=("audio",)),
                                       "L00.out": "layer"}, root, sites=None)
            with tapped.step("interleaved", [1, 2, 3], media={"image", "video", "audio"}):
                model(x, [(images, grids[0]), (video, grids[1])], audio=torch.ones((5, 4)))
            items = json.loads((root / "step0000" / "items.json").read_text())
            self.assertEqual(items["encoder"]["enc.vision.out"], [
                {"request": "interleaved", "start": 0, "kind": "image", "item": 0, "index": 0, "rows": 2},
                {"request": "interleaved", "start": 2, "kind": "image", "item": 1, "index": 0, "rows": 1},
                {"request": "interleaved", "start": 3, "kind": "video", "item": 0, "index": 0, "rows": 2}])
            self.assertEqual(items["encoder"]["enc.audio.out"], [
                {"request": "interleaved", "start": 0, "kind": "audio", "item": 0, "index": 0, "rows": 5}])
            with self.assertRaisesRegex(ValueError, r"image items of shares \[4, 4\] do not split 3 rows"):
                with tapped.step("short", [1, 2, 3], media={"image"}):
                    model(x, [(images, torch.tensor([[1, 2, 2], [1, 2, 2]]))])

    def test_calls_marked_as_items_are_recorded_per_item(self):
        """An encoder called once per item (tools/refrun/encoder.py) marks
        each call with `taps.item`: image, video, image become image 0,
        video 0 and image 1, a block tap firing twice in an item continues
        its rows, and an unmarked call beside marked ones is refused."""
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder, self.block, self.layer = (torch.nn.Identity() for _ in range(3))

            def forward(self, x, items):
                for kind, rows in items:
                    with taps.item(kind):
                        for _ in range(2):
                            self.block(torch.ones((rows, 4)))
                        self.encoder(torch.full((rows, 4), float(rows)))
                return self.layer(x)
        model = Model()
        x = torch.ones((1, 3, 4))
        with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as directory:
            root = Path(directory)
            tapped = taps.Taps(model, {"enc.vision.out": taps.Hook("encoder", rows=("image", "video")),
                                       "enc.vision.B00.out": taps.Hook("block", rows=("image", "video")),
                                       "L00.out": "layer"}, root, sites=None)
            with tapped.step("interleaved", [1, 2, 3], media={"image", "video"}):
                model(x, [("image", 2), ("video", 3), ("image", 1)])
            segments = json.loads((root / "step0000" / "items.json").read_text())["encoder"]
            self.assertEqual(segments["enc.vision.out"], [
                {"request": "interleaved", "start": 0, "kind": "image", "item": 0, "index": 0, "rows": 2},
                {"request": "interleaved", "start": 2, "kind": "video", "item": 0, "index": 0, "rows": 3},
                {"request": "interleaved", "start": 5, "kind": "image", "item": 1, "index": 0, "rows": 1}])
            self.assertEqual([(s["kind"], s["item"], s["index"], s["rows"]) for s in segments["enc.vision.B00.out"]],
                             [("image", 0, 0, 2), ("image", 0, 2, 2), ("video", 0, 0, 3), ("video", 0, 3, 3),
                              ("image", 1, 0, 1), ("image", 1, 1, 1)])
            with self.assertRaisesRegex(ValueError, "both inside and outside media items"):
                with tapped.step("mixed", [1, 2, 3], media={"image"}):
                    with taps.item("image"):
                        model.encoder(torch.ones((1, 4)))
                    model.encoder(torch.ones((1, 4)))
            with self.assertRaisesRegex(ValueError, "a video item in a sequence without video media"):
                with tapped.step("absent", [1, 2, 3], media={"image"}):
                    with taps.item("video"):
                        pass
            with taps.item("image"):
                model.encoder(torch.ones((1, 4)))

    def test_chunked_passes_join_rows_by_position(self):
        """A pass in chunks (an adapter's chunked prefill): rows of each chunk
        land at their absolute positions, encoder rows are written once, and
        a tap whose chunks are out of order or stop short fails its step."""
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder, self.layer = torch.nn.Identity(), torch.nn.Identity()

            def forward(self, x, chunks, image=None):
                if image is not None:
                    self.encoder(image)
                for start, stop in chunks:
                    with taps.chunk(start, stop):
                        self.layer(x[:, start:stop])
        model = Model()
        x = torch.arange(7.0).reshape(1, 7, 1).repeat(1, 1, 2)
        image = torch.full((3, 2), 9.0)
        with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as directory:
            root = Path(directory)
            tapped = taps.Taps(model, {"enc.vision.out": taps.Hook("encoder", rows=("image",)),
                                       "L00.out": "layer"}, root, sites=None)
            with tapped.step("chunks", list(range(7)), [0, 2, 3, 6], media={"image"}):
                model(x, [(0, 3), (3, 5), (5, 7)], image)
            with self.assertRaisesRegex(ValueError, "do not follow"):
                with tapped.step("order", list(range(7))):
                    model(x, [(3, 7), (0, 3)])
            with self.assertRaisesRegex(ValueError, "rows 0..5 of 7"):
                with tapped.step("short", list(range(7))):
                    model(x, [(0, 3), (3, 5)])
            # Sparse rows: the first and middle chunks select none.
            with tapped.step("sparse", list(range(7)), [6]):
                model(x, [(0, 3), (3, 5), (5, 7)])
            with taps.chunk(0, 3):
                model(x, [])
            step = root / "step0000"
            index = [json.loads(line) for line in (step / "index.jsonl").read_text().splitlines()]
            self.assertEqual([(e["name"], e["rows"]) for e in index], [("enc.vision.out", 3), ("L00.out", 4)])
            self.assertTrue(np.array_equal(tapdiff.read(step / "L00.out.r0.bin", index[1]),
                                           np.array([[0.0, 0.0], [2.0, 2.0], [3.0, 3.0], [6.0, 6.0]], np.float32)))
            self.assertEqual(json.loads((step / "items.json").read_text())["positions"], [0, 2, 3, 6])
            sparse = root / "step0003"
            entry = json.loads((sparse / "index.jsonl").read_text())
            self.assertEqual((entry["name"], entry["rows"], entry["cols"]), ("L00.out", 1, 2))
            self.assertTrue(np.array_equal(tapdiff.read(sparse / "L00.out.r0.bin", entry),
                                           np.array([[6.0, 6.0]], np.float32)))
            for name, completion in (("step0000", "full"), ("step0001", "incomplete"), ("step0002", "incomplete"),
                                     ("step0003", "full")):
                self.assertEqual(json.loads((root / name / "completion.json").read_text())["completion"],
                                 completion, name)
            self.assertEqual(taps.OPEN, [])

if __name__ == "__main__":
    unittest.main()
