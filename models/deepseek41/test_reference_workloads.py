"""The DeepSeek V4.1 Flash reference workloads, built on CPU from the
checkpoint's tokenizer and pinned encoding.py and the evals cache's
wikitext; skipped without them ($CHOPINFER_DEEPSEEK_CHECKPOINT,
$CHOPINFER_DATA/evals)."""

import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "models/deepseek41"))
CHECKPOINT = os.environ.get("CHOPINFER_DEEPSEEK_CHECKPOINT")
EVALS = Path(os.environ.get("CHOPINFER_DATA", "/nonexistent")) / "evals"

try:
    import tokenizers  # noqa: F401
except ModuleNotFoundError:
    tokenizers = None


@unittest.skipIf(tokenizers is None or not CHECKPOINT or not (EVALS / "wikitext.json").exists(),
                 "needs tokenizers, the checkpoint and the evals cache")
class Workloads(unittest.TestCase):
    def test_teacher_and_long_context(self):
        import reference_workloads as workloads
        from tools.refrun import workload
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "out"
            manifest = workloads.build(CHECKPOINT, EVALS, output, 65536)
            model, teacher = workload.load(output / "teacher.json")
            self.assertEqual(model, "deepseek41")
            _, long = workload.load(output / "long-context.json")
            self.assertEqual([s.name for s in long], ["long_4k", "long_64k"])
            self.assertEqual(len(long[1].token_ids), 65536)
            self.assertEqual(long[1].score_from, 65536 - 1024)
            bos = workloads.encode(workloads.tokenizer(CHECKPOINT), workloads.BOS)
            for sequence in teacher:
                self.assertLessEqual(len(sequence.token_ids), workloads.F64_TOKENS)
                self.assertEqual(list(sequence.token_ids[:len(bos)]), bos, sequence.name)
            chats = {s.name for s in teacher if s.category == "chat"}
            self.assertEqual(chats, set(workloads.CHATS))
            quote = manifest["long_quote_source_start"]["long_64k"]
            self.assertEqual(list(long[1].token_ids[-1024:]), list(long[1].token_ids[quote:quote + 1024]))
            _, probes = workload.load(output / "probes-greedy.json")
            for probe in probes:
                (image,) = probe.media
                rows, cols = (int(v) for v in memoryview(image.grid.data).cast("q"))
                self.assertEqual(tuple(image.tensor.shape), (rows * cols, 3, 14, 14))
                span = list(probe.token_ids).count(129264)
                self.assertEqual(span, -(-rows // 3) * (-(-cols // 3) + 1) + 2, probe.name)
                self.assertEqual(probe.answer.policy, "greedy")
            again = Path(directory) / "again"
            workloads.build(CHECKPOINT, EVALS, again, 65536)
            self.assertEqual(json.loads((output / "manifest.json").read_text()),
                             json.loads((again / "manifest.json").read_text()))


if __name__ == "__main__":
    unittest.main()
