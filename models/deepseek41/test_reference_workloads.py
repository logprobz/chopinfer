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


@unittest.skipIf(tokenizers is None, "needs tokenizers (set CHOPINFER_REFRUN_PYTHON)")
@unittest.skipUnless(CHECKPOINT, "needs the official checkpoint (CHOPINFER_DEEPSEEK_CHECKPOINT)")
class Workloads(unittest.TestCase):
    @unittest.skipUnless((EVALS / "wikitext.json").is_file(), "needs the wikitext cache (CHOPINFER_DATA/evals)")
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


    def test_chat(self):
        import reference_workloads as workloads
        from tools.refrun import workload
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "chat"
            workloads.build_chat(CHECKPOINT, output)
            model, chat = workload.load(output / "chat.json")
            self.assertEqual(model, "deepseek41")
            self.assertEqual([s.name for s in chat], [*workloads.CHATS, *workloads.CHATS_EXTENDED])
            bos = workloads.encode(workloads.tokenizer(CHECKPOINT), workloads.BOS)
            for sequence in chat:
                self.assertEqual(list(sequence.token_ids[:len(bos)]), bos, sequence.name)
                self.assertEqual(sequence.score_from, 1)
                self.assertLessEqual(len(sequence.token_ids), workloads.F64_TOKENS)


PROBES = Path(os.environ.get("CHOPINFER_DATA", "/nonexistent")) / "refrun/deepseek41/workloads-20261018-probes"
GREEDY = (Path(os.environ.get("CHOPINFER_DATA", "/nonexistent"))
          / "refrun/deepseek41/43-b1-ref-probes-2-20261018T015515/probes/f32.json")


@unittest.skipIf(not (PROBES / "probes.json").exists() or not GREEDY.exists(),
                 "needs the probe workload and its F32 greedy run under CHOPINFER_DATA")
class ServedAnswers(unittest.TestCase):
    def test_served_answers_drop_the_stop_token(self):
        import reference_workloads as workloads
        import shutil
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "probes"
            shutil.copytree(PROBES, output)
            forced = workloads.answered(output, GREEDY)
            served = workloads.answered(output, GREEDY, served=True)
            self.assertEqual(forced, json.loads((PROBES / "probes.json").read_text())["sequences"])
            for full, cut in zip(forced, served, strict=True):
                self.assertEqual(full["answer"]["ids"][-1:], list(workloads.STOP_IDS))
                self.assertEqual(cut, full | {"answer": {"policy": "forced", "ids": full["answer"]["ids"][:-1]}})
            # An answer that did not stop has no served form.
            run = json.loads(GREEDY.read_text())
            for entry in run["entries"]:
                if entry["key"]["sequence"]["answer"]["policy"] == "greedy":
                    entry["scores"]["answer"] = entry["scores"]["answer"][:-1]
            cut_run = Path(directory) / "cut.json"
            cut_run.write_text(json.dumps(run))
            with self.assertRaisesRegex(ValueError, "did not stop"):
                workloads.answered(output, cut_run, served=True)


class ExtendedChats(unittest.TestCase):
    def test_specs_take_the_encoding_form(self):
        import reference_workloads as workloads
        retry = workloads.encode_spec(workloads.CHATS_EXTENDED["chat_tool_retry"])
        self.assertEqual(retry["thinking_mode"], "thinking")
        self.assertEqual(retry["reasoning_effort"], "low")
        self.assertEqual(retry["messages"][0], {"role": "system", "content": "You can call tools.",
                                                "tools": [workloads.CALENDAR]})
        plan = workloads.encode_spec(workloads.CHATS_EXTENDED["chat_multi_turn_plan"])
        self.assertEqual(plan["thinking_mode"], "chat")
        self.assertNotIn("reasoning_effort", plan)
        self.assertFalse(any("reasoning_content" in m for m in plan["messages"]))
        support = workloads.encode_spec(workloads.CHATS_EXTENDED["chat_system_format"])
        self.assertTrue(support["messages"][0]["content"].startswith("You are a support assistant"))
        self.assertNotIn("tools", support["messages"][0])


if __name__ == "__main__":
    unittest.main()
