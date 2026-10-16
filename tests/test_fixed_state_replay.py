"""CPU checks for the fixed-state decode replay's traces, arms and comparison."""
import contextlib
import importlib.util
import io
import json
import math
import os
import struct
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location(
    "fixed_state", Path(__file__).resolve().parents[1] / "tools/dev/fixed-state-replay.py")
replay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay)

VOCABULARY = 2048


def reference():
    texts = [dict(name=f"text{i}", score_from=1,
                  prompt_token_ids=[(7 * i + 3 * j) % 1000 + 1 for j in range(12 + i)],
                  exact=dict(logprob=[-0.5 - 0.01 * j for j in range(11 + i)]))
             for i in range(5)]
    texts.append(dict(name="source", score_from=1, prompt_token_ids=list(range(1000, 1200)),
                      exact=dict(logprob=[-1.0] * 199)))
    return dict(texts=texts)


def arguments(root, **changes):
    path = root / "reference.json"
    path.write_text(json.dumps(reference()))
    values = dict(reference=path, rows=8, probes=4, decode_steps=3, shortest=16, longest=40,
                  holder_tokens=2, vocabulary=VOCABULARY, prefill_tokens=64, page_tokens=16)
    values.update(changes)
    return SimpleNamespace(**values)


def check_trace(trace):
    """The Rust version-5 rules: whole prefill once, then one forced token
    per decode item, histories extending by the inputs."""
    histories, prefilled, decoding = {}, set(), False
    for step in trace["steps"]:
        prefill = step["items"][0]["prefill"]
        assert not (prefill and decoding)
        decoding |= not prefill
        for item in step["items"]:
            assert item["prefill"] == prefill and item["drafts"] == 0
            assert (item["slot"] in prefilled) != prefill
            if prefill:
                prefilled.add(item["slot"])
            else:
                assert len(item["inputs"]) == 1
            histories[item["slot"]] = histories.get(item["slot"], []) + item["inputs"]
            assert histories[item["slot"]] == item["history"]
    assert decoding and prefilled == set(range(trace["concurrency"]))


class Traces(unittest.TestCase):
    def test_arms_share_probe_rows_and_change_only_their_variable(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            traces, pages, lengths = replay.make_traces(arguments(root), root)
            loaded = {name: json.loads(Path(t["path"]).read_text()) for name, t in traces.items()}
            for trace in loaded.values():
                check_trace(trace)
                self.assertEqual(trace["version"], replay.MIXED_DECODE)
                for step in trace["steps"]:
                    self.assertLessEqual(sum(len(i["inputs"]) for i in step["items"] if i["prefill"]), 64)
            a, b, holders = loaded["a"], loaded["b"], loaded["holders"]
            # Probes and fillers alternate; fillers keep their windows and
            # take the reversed lengths.
            decode_a, decode_b = a["steps"][-1]["items"], b["steps"][-1]["items"]
            self.assertEqual([len(i["history"]) for i in decode_a[1::2]],
                             [len(i["history"]) for i in decode_b[1::2]][::-1])
            self.assertEqual([i["history"] for i in decode_a[0::2]], [i["history"] for i in decode_b[0::2]])
            for step_a, step_b in zip(a["steps"][-3:], b["steps"][-3:]):
                self.assertEqual([i["inputs"] for i in step_a["items"]], [i["inputs"] for i in step_b["items"]])
            self.assertNotEqual(lengths["a"][0], lengths["b"][0])
            self.assertEqual(sorted(lengths["a"][0]), sorted(lengths["b"][0]))
            # Holders take the low slots in a prefill step of their own and
            # never decode; the bucket's steps are otherwise the same.
            self.assertEqual(holders["concurrency"], 16)
            self.assertEqual([i["slot"] for i in holders["steps"][0]["items"]], list(range(8)))
            self.assertTrue(all(i["slot"] >= 8 for s in holders["steps"][1:] for i in s["items"]))
            self.assertEqual([[i["history"] for i in s["items"]] for s in holders["steps"][1:]],
                             [[i["history"] for i in s["items"]] for s in a["steps"]])
            self.assertEqual(pages, 16 * math.ceil(traces["holders"]["final_tokens"] / 16) + 16)
            probes = json.loads((root / "probes.json").read_text())
            self.assertEqual(len(probes), 4 * 4)
            first = next(p for p in probes.values() if p["text"] == "text0" and p["step"] == 0)
            self.assertEqual((first["position"], first["exact"]), (8, -0.5 - 0.01 * 7))

    def test_traces_refuse_unchanged_composition_and_foreign_tokens(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(ValueError, "composition"):
                replay.make_traces(arguments(root, rows=5), root)
            with self.assertRaisesRegex(ValueError, "vocabulary"):
                replay.make_traces(arguments(root, vocabulary=1100), root)
            with self.assertRaisesRegex(ValueError, "texts that can force"):
                replay.make_traces(arguments(root, decode_steps=12), root)


def write_arm(root, rows):
    """A recorded arm: one frame per step of (history, f32 logits) rows."""
    root.mkdir(parents=True)
    with (root / "frames.jsonl").open("w") as frames, (root / "rows.bin").open("wb") as data:
        frames.write("{}\n")
        for step in rows:
            records = []
            for history, values in step:
                raw = struct.pack(f"<{len(values)}f", *values)
                data.write(raw)
                records.append(dict(element_type=1, width=len(values), byte_count=len(raw),
                                    metadata=dict(history=history), sample={}))
            frames.write(json.dumps(dict(rows=records)) + "\n")


class Selection(unittest.TestCase):
    def test_arm_subsets_keep_run_order_and_complete_comparisons(self):
        arms, comparisons = replay.select_arms("b1,a1-repeat,a1")
        self.assertEqual([arm for arm, _, _ in arms], ["a1", "a1-repeat", "b1"])
        self.assertEqual([(c[0], c[1], c[2]) for c in comparisons],
                         [("determinism", "a1", "a1-repeat"), ("composition", "a1", "b1")])
        self.assertEqual(replay.select_arms("")[0], replay.ARMS)
        for names in ("a1,a3", "a1,a1"):
            with self.assertRaisesRegex(ValueError, "--arms"):
                replay.select_arms(names)
        self.assertEqual(replay.fixed_batch.parse_overrides(["CHOPINFER_MIMIC=0"]), {"CHOPINFER_MIMIC": "0"})
        with self.assertRaises(ValueError):
            replay.fixed_batch.parse_overrides(["CHOPINFER_KERNEL_GPUS=0"])


PROBE, FILLER, OTHER_FILLER, HOLDER = [1, 2], [3, 4], [3, 5], [9]
BASE = [[(PROBE, [0.0, 1.0, 2.0]), (FILLER, [1.0, 1.0, 1.0])],
        [(PROBE + [2], [2.0, 0.0, 0.0]), (FILLER + [6], [0.0, 0.0, 0.0])]]
MOVED = [[(PROBE, [0.0, 1.0, 2.0]), (OTHER_FILLER, [5.0, 1.0, 1.0])],
         [(PROBE + [2], [0.0, 0.0, 2.0]), (OTHER_FILLER + [6], [0.0, 0.0, 0.0])]]
CAPACITY = [BASE[0], [BASE[1][0], (FILLER + [6], [0.0, 0.5, 0.0])]]


def compared(root, changes=None):
    """compare over recorded arms: the fixture's, with `changes` (arm to
    rows) applied."""
    run = root / "run"
    arms = {"a1": BASE, "a1-repeat": BASE, "a2": CAPACITY, "a2-repeat": CAPACITY,
            "holders2": [[(HOLDER, [0.0, 0.0, 0.0])]] + CAPACITY, "b1": MOVED, "b2": MOVED}
    arms.update(changes or {})
    for arm, rows in arms.items():
        write_arm(run / arm, rows)
    (run / "result.json").write_text(json.dumps(dict(passed=True)))
    traces = dict(a=dict(rows=4), b=dict(rows=4), holders=dict(rows=5))
    (root / "prepared.json").write_text(json.dumps(dict(
        arms=replay.ARMS, comparisons=replay.COMPARISONS, bucket=2, traces=traces)))
    probes = {replay.history_key(PROBE): dict(text="p", step=0, token=2, exact=-1.0),
              replay.history_key(PROBE + [2]): dict(text="p", step=1, token=0, exact=None)}
    (root / "probes.json").write_text(json.dumps(probes))
    with contextlib.redirect_stdout(io.StringIO()):
        return replay.compare(SimpleNamespace(run=run))


class Compare(unittest.TestCase):
    def test_compare_reports_bitwise_capacity_placement_and_composition(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / "run"
            report = compared(root)
            outcome = {(c["question"], c["other"]): c for c in report["comparisons"]}
            self.assertTrue(outcome["determinism", "a1-repeat"]["bitwise_equal"])
            capacity = outcome["capacity", "a2"]
            self.assertFalse(capacity["bitwise_equal"])
            self.assertEqual((capacity["rows"], capacity["identical"]), (4, 3))
            self.assertEqual(capacity["first_difference"], dict(frame=1, probe=None, element=1))
            self.assertTrue(outcome["placement", "holders2"]["bitwise_equal"])
            self.assertEqual((outcome["placement", "holders2"]["rows"],
                              outcome["placement", "holders2"]["unpaired_rows"]), (4, 1))
            composition = outcome["composition", "b1"]
            self.assertEqual((composition["rows"], composition["identical"]), (2, 1))
            step = composition["probe_steps"]["1"]
            self.assertEqual((step["rows"], step["top1_differs"], step["max_abs_logit"]), (1, 1, 2.0))
            # log_softmax([2,0,0])[0] - log_softmax([0,0,2])[0] = 2.
            self.assertAlmostEqual(step["dlogprob"], 2.0, places=6)
            expected = abs(2.0 - math.log(math.exp(0) + math.exp(1) + math.exp(2)) + 1.0)
            self.assertAlmostEqual(report["probe_error"]["a1"]["mean_abs_dlogprob"], expected, places=6)
            self.assertEqual(report["probe_error"]["a1"]["positions"], 1)
            self.assertTrue((run / "compare.json").is_file())

    def test_shortened_or_diverging_arms_are_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            # A repeat that stopped a step early would match on what it has.
            with self.assertRaisesRegex(ValueError, "a1-repeat: 2 rows recorded, its trace has 4"):
                compared(Path(temporary) / "short", {"a1-repeat": BASE[:1]})
            # Same count, other histories.
            other = [BASE[0], [BASE[1][0], (FILLER + [7], [0.0, 0.0, 0.0])]]
            with self.assertRaisesRegex(ValueError, "a1/a1-repeat: the arms recorded different histories"):
                compared(Path(temporary) / "diverged", {"a1-repeat": other})
            # Placement needs every row of the plain arm.
            holders = [[(HOLDER, [0.0, 0.0, 0.0])], [BASE[1][0], (FILLER + [7], [0.0, 0.0, 0.0])],
                       [(PROBE, [0.0, 1.0, 2.0]), (FILLER, [1.0, 1.0, 1.0])]]
            with self.assertRaisesRegex(ValueError, "rows of a2 are missing from holders2"):
                compared(Path(temporary) / "placement", {"holders2": holders})

    def test_signed_zero_rows_differ_bitwise_without_a_numeric_difference(self):
        with tempfile.TemporaryDirectory() as temporary:
            zero = [BASE[0], [BASE[1][0], (FILLER + [6], [0.0, -0.0, 0.0])]]
            report = compared(Path(temporary), {"a1-repeat": zero})
            determinism = report["comparisons"][0]
            self.assertEqual((determinism["question"], determinism["bitwise_equal"]), ("determinism", False))
            self.assertEqual((determinism["identical"], determinism["max_abs_logit"]), (3, 0.0))
            self.assertEqual(determinism["first_difference"], dict(frame=1, probe=None, element=1))

    def test_bf16_rows_widen_to_their_float_values(self):
        raw = struct.pack("<3H", 0x3F80, 0xC000, 0x0000)
        self.assertEqual(list(replay.logits(raw, 2)), [1.0, -2.0, 0.0])


class Arms(unittest.TestCase):
    def prepared(self, root):
        binary, bundle = root / "replay-tests", root / "bundle"
        binary.write_bytes(b"binary")
        bundle.mkdir()
        (bundle / "manifest.json").write_text("{}")
        traces = {}
        for name, requests in (("a", 4), ("b", 4), ("holders", 8)):
            path = root / f"trace-{name}.json"
            path.write_text(name)
            traces[name] = dict(path=str(path), sha256=replay.diagnostics.digest(path),
                                requests=requests, steps=5, rows=17)
        return dict(format=replay.FORMAT, source="fixture", binary=str(binary),
                    binary_sha256=replay.diagnostics.digest(binary), bundle=str(bundle),
                    manifest_sha256=replay.diagnostics.digest(bundle / "manifest.json"),
                    checkpoint=str(root / "checkpoint"), topology="fixture-tp4", ranks=4,
                    bucket=4, kv_pages=40, traces=traces, arms=replay.ARMS, comparisons=replay.COMPARISONS,
                    environment={"CHOPINFER_SWITCH": "0"})

    def test_run_replays_each_arm_at_its_capacity_with_shared_pages(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prepared = self.prepared(root)
            path = root / "prepared.json"
            path.write_text(json.dumps(prepared))
            seen = []

            def execute(command, log, env, timeout):
                config = json.loads(Path(env["CHOPINFER_BATCH_REPLAY_CONFIG"]).read_text())
                seen.append((config, env))
                output = Path(config["output"])
                output.mkdir()
                trace = next(t for t in prepared["traces"].values() if t["path"] == config["trace"])
                (output / "receipt.json").write_text(json.dumps(dict(
                    passed=True, steps=trace["steps"], rows=trace["rows"], mixed=True,
                    max_sequences=config["max_sequences"], kv_pages=config["kv_pages"],
                    rank_rows={}, memory_after_steps={})))

            with patch.object(replay.fixed_batch, "command", execute), \
                    patch.object(replay.fixed_batch, "gpu_state", lambda devices: "idle"), \
                    patch.dict(os.environ, CHOPINFER_MIMIC="1"), contextlib.redirect_stdout(io.StringIO()):
                replay.run(SimpleNamespace(prepared=path, devices="4,5,6,7", minutes=5, compare_minutes=0))
            self.assertEqual([(c["max_sequences"], c["trace"]) for c, _ in seen],
                             [(m * 4, prepared["traces"][t]["path"]) for _, t, m in replay.ARMS])
            for config, env in seen:
                self.assertTrue(config["mixed"] and config["trace_input"])
                self.assertEqual((config["kv_pages"], config["concurrency"] <= config["max_sequences"]), (40, True))
                self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "4,5,6,7")
                self.assertEqual(env["CHOPINFER_KERNEL_GPUS"], "0,1,2,3")
                self.assertNotIn("CHOPINFER_MIMIC", env)
                self.assertEqual(env["CHOPINFER_SWITCH"], "0")
                self.assertNotIn("CHOPINFER_BATCH_REPLAY_DRY_RUN", env)
            result = json.loads(next(root.glob("run-*/result.json")).read_text())
            self.assertTrue(result["passed"])

    def test_changed_inputs_and_dry_run_identity_are_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prepared = self.prepared(root)
            Path(prepared["traces"]["b"]["path"]).write_text("changed")
            with self.assertRaisesRegex(ValueError, "trace changed"):
                replay.check_prepared(prepared)

            def execute(command, log, env, timeout):
                self.assertEqual((env["CUDA_VISIBLE_DEVICES"], env["CHOPINFER_BATCH_REPLAY_DRY_RUN"]), ("", "1"))
                config = json.loads(Path(env["CHOPINFER_BATCH_REPLAY_CONFIG"]).read_text())
                trace = next(t for t in prepared["traces"].values() if t["path"] == config["trace"])
                identity = dict(format="fixed-batch-mixed-v1", max_sequences=config["max_sequences"],
                                kv_pages=41, trace_sha256=trace["sha256"])
                log.write_text("test replay ... " + replay.DRY_RUN + json.dumps(dict(identity=identity)) + "\n")

            with patch.object(replay.fixed_batch, "command", execute):
                with self.assertRaisesRegex(RuntimeError, "dry run identity"):
                    replay.dry_run(prepared, root)
                prepared["kv_pages"] = 41
                again = root / "again"
                again.mkdir()
                self.assertEqual(set(replay.dry_run(prepared, again)), {arm for arm, _, _ in replay.ARMS})


if __name__ == "__main__":
    unittest.main()
