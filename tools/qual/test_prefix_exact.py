"""Offline checks of prefix_exact.py's positions, protocol checks and gate."""

from qual_test_config import setUpModule

import base64
import copy
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest import mock
import zlib

import declaration
import prefix_exact
from prefix_exact import (MEDIA_PAIRS, PROMPT_END_ARMS, block_colors, cache_errors, evaluate,
                          first_difference, load_exact, png, positions, primer,
                          prompt_end_positions, summarize, summarize_acceptance)

# Audited to run once rather than under every gate config (ci/qualification.py).
QUALIFICATION_RUNS_ONCE = "prefix_exact positions, protocol checks and gate; other gates are selected explicitly"


def text(name, ids, logprob, top1, score_from=1):
    exact = {"logprob": logprob, "top1": top1}
    return {"name": name, "category": "prose", "prompt_token_ids": ids,
            "score_from": score_from, "exact": exact}


def row(logprob, top):
    return {"logprob": logprob, "top": top}


class PositionTests(unittest.TestCase):
    def test_positions_start_past_the_first_block_and_anchor_at_the_last_boundary(self):
        texts = [text("a", list(range(10)), [], [], score_from=1),
                 text("b", list(range(12)), [], [], score_from=9)]
        scored = [(entry["name"], p, anchor) for entry, p, anchor in positions(texts, block=4)]
        self.assertEqual(scored, [("a", 5, 4), ("a", 6, 4), ("a", 7, 4), ("a", 8, 4),
                                  ("a", 9, 8), ("b", 9, 8), ("b", 10, 8), ("b", 11, 8)])
        # Every hit prefills at most one block after its anchor.
        self.assertTrue(all(anchor < p <= anchor + 4 for _, p, anchor in scored))
        self.assertEqual([p for _, p, _ in positions(texts[:1], block=4, stride=2)], [5, 7, 9])

    def test_the_primer_shares_the_anchor_and_then_diverges_for_a_block(self):
        ids = [5, 5, 6, 5, 7]
        # Its prompt-end anchor, even moved down to a granule of up to a
        # block, lies past the divergence.
        self.assertEqual(primer(ids, 3, block=2), [5, 5, 6, 6, 6])
        self.assertEqual(primer(ids, 4, block=2), [5, 5, 6, 5, 5, 5])

    def test_a_primer_past_a_chunk_ends_its_final_chunk_after_it_diverges(self):
        ids = [5, 5, 6, 5, 7, 7, 7, 7, 7, 7]
        self.assertEqual(primer(ids, 2, chunk=4, block=2), [5, 5, 5, 5])
        # Its final chunk starts at 8, past the anchor at 5 where it diverges.
        self.assertEqual(primer(ids, 5, chunk=4, block=2), [5, 5, 6, 5, 7, 5, 5, 5, 5])
        self.assertEqual(primer(ids, 8, chunk=4, block=2), [5, 5, 6, 5, 7, 7, 7, 7] + [5] * 5)

    def test_expected_hits_follow_the_models_prefill_granule(self):
        main = {"position": 129, "anchor": 128, "cached": {"primer": 0, "chunked": 0, "hit": 128, "cold": 0}}
        prompt_end = {"anchor": 133, "cached": {"primer": 0, "hit": 133, "cold": 0}}
        # MiMo captures at the primer's end, Qwen at the granule below it.
        self.assertEqual(prefix_exact.prefix_policy()["prefill_granule"], 1)
        self.assertEqual(prefix_exact.expected_cached(prompt_end, 64, PROMPT_END_ARMS)["hit"], 133)
        self.assertEqual(prefix_exact.expected_cached(main, 64),
                         {"primer": 0, "chunked": 0, "hit": 128, "cold": 0})
        with mock.patch.dict(os.environ, {"CHOPINFER_QUAL_CONFIG": str(declaration.path("qwen38"))}):
            self.assertEqual(prefix_exact.prefix_policy()["prefill_granule"], 64)
            self.assertEqual(prefix_exact.expected_cached(prompt_end, 64, PROMPT_END_ARMS),
                             {"primer": 0, "hit": 128, "cold": 0})
            self.assertEqual(prefix_exact.expected_cached(main, 64)["hit"], 128)

    def test_cache_errors_name_every_unexpected_hit_or_miss(self):
        good = {"text": "a", "position": 5, "anchor": 4,
                "cached": {"primer": 0, "chunked": 0, "hit": 4, "cold": 0}}
        stale = {**good, "position": 6, "cached": {**good["cached"], "chunked": 4}}
        self.assertEqual(cache_errors([good], 4), [])
        self.assertEqual([error["position"] for error in cache_errors([good, stale], 4)], [6])

    def test_the_media_start_is_the_first_differing_token(self):
        self.assertEqual(first_difference([1, 2, 3], [1, 2, 4]), 2)
        self.assertEqual(first_difference([1, 2], [1, 2, 4]), 2)


def qwen_text(name, ids, logprob, top1):
    """A text of the Qwen3.8 reference (the GPU runner's layout): one exact
    variant, no TP4 partials."""
    return {"name": name, "category": "prose", "prompt_token_ids": ids, "score_from": 1,
            "exact": {"logprob": logprob, "top1": top1}}


class QwenReferenceTests(unittest.TestCase):
    def setUp(self):
        selected = mock.patch.dict(os.environ, {"CHOPINFER_QUAL_CONFIG": str(
            declaration.path("qwen38"))})
        selected.start()
        self.addCleanup(selected.stop)

    def test_the_qwen_reference_has_no_partials_variant(self):
        ids = list(range(10, 16))
        reference = {"schema": "chopinfer.qwen38.teacher-reference.v2",
                     "texts": [qwen_text("a", ids, [-1.0] * 5, ids[1:])]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "teacher-exact.json")
            path.write_text(json.dumps(reference))
            exact = load_exact(path)
        self.assertEqual(prefix_exact.prefix_policy()["family"], "qwen38")
        self.assertEqual(exact["texts"][0]["score_from"], 1)
        rows = [{"text": "a", "category": "prose", "position": p, "anchor": 3,
                 "cold": row(-1.0, {"x": -1.0}), "hit": row(-1.0, {"x": -1.0}),
                 "cached": {"primer": 0, "hit": 3, "cold": 0}} for p in range(4, 6)]
        scores = prefix_exact.report_scores(rows, exact, {}, PROMPT_END_ARMS)
        self.assertEqual(list(scores["by_anchor"]), ["anchor_dense"])
        self.assertNotIn("gate_chunking_floor", scores["all"])
        self.assertEqual(list(scores["all"]["distances"]), ["hit_vs_cold"])

    def test_prompt_end_anchors_are_never_block_boundaries(self):
        entry = text("a", list(range(300)), [], [])
        scored = prompt_end_positions([entry], block=64)
        anchors = [anchor for _, _, anchor in scored]
        self.assertTrue(all(64 <= anchor < p and anchor % 64 for _, p, anchor in scored))
        # Primers end at every offset within a block of 4.
        self.assertEqual({anchor % 4 for anchor in anchors}, {0, 1, 2, 3})
        self.assertTrue(all(p - anchor <= 6 for _, p, anchor in scored))

    def test_a_prompt_end_hit_resumes_the_granule_below_the_primer_and_joins_the_gate(self):
        ids = list(range(1000, 1200))
        server = FakeServer(64, ids, granule=64)
        entry = text("a", ids, [-0.5] * 199, ids[1:])
        scored = [prefix_exact.score_prompt_end(server, "m", entry, p, anchor, "run")
                  for _, p, anchor in prompt_end_positions([entry], block=64)]
        self.assertEqual(cache_errors(scored, 64, PROMPT_END_ARMS), [])
        self.assertTrue(all(row["cached"]["hit"] == row["anchor"] // 64 * 64 < row["anchor"]
                            for row in scored))
        result = evaluate({"family": prefix_exact.prefix_policy()["family"], "positions": [], "prompt_end": scored, "detokenized": {}},
                          {"schema": prefix_exact.prefix_policy()["reference_schema"], "texts": [entry]})
        self.assertTrue(result["checks"]["prompt_end_hit_equals_cold"])
        self.assertTrue(result["checks"]["prompt_end_hit_vs_cold"])
        self.assertEqual(result["prompt_end_scores"]["all"]["positions"], len(scored))
        # A prompt-end hit that missed its anchor is a cache error.
        scored[0]["cached"]["hit"] = 0
        self.assertEqual(len(cache_errors(scored, 64, PROMPT_END_ARMS)), 1)

    def test_a_hit_at_the_primers_end_is_not_what_qwen_captures(self):
        ids = list(range(1000, 1200))
        entry = text("a", ids, [-0.5] * 199, ids[1:])
        scored = [prefix_exact.score_prompt_end(FakeServer(64, ids), "m", entry, p, anchor, "run")
                  for _, p, anchor in prompt_end_positions([entry], block=64)]
        self.assertEqual(len(cache_errors(scored, 64, PROMPT_END_ARMS)), len(scored))


class LoaderTests(unittest.TestCase):
    def test_the_checked_in_mimo_reference_loads(self):
        exact = load_exact(Path(__file__).resolve().parents[2] / "fixtures/mimo26/teacher-exact-reference.json")
        self.assertEqual(prefix_exact.prefix_policy()["family"], "mimo26")
        self.assertTrue(all(text["score_from"] >= 1 for text in exact["texts"]))


def qwen_reference(ids, exact_logprob):
    return {"schema": "chopinfer.qwen38.teacher-reference.v2",
            "texts": [qwen_text(name, ids, [exact_logprob] * (len(ids) - 1), ids[1:]) for name in ("a", "b")]}


def aligned_rows(ids, cold, chunked, hit):
    """Main-arm rows at a 64-token anchor, which every granule keeps, with
    the cached tokens the anchor rules give them."""
    rows = []
    for name in ("a", "b"):
        for position in range(2, len(ids)):
            scored = {"text": name, "category": "prose", "position": position,
                      "anchor": 64, "cold": row(cold, {str(ids[position]): cold}),
                      "chunked": row(chunked, {str(ids[position]): chunked}),
                      "hit": row(hit, {str(ids[position]): hit})}
            rows.append({**scored, "cached": prefix_exact.expected_cached(scored, 64)})
    return rows


class QwenGateTests(unittest.TestCase):
    def setUp(self):
        selected = mock.patch.dict(os.environ, {"CHOPINFER_QUAL_CONFIG": str(
            declaration.path("qwen38"))})
        selected.start()
        self.addCleanup(selected.stop)

    ids = list(range(20, 30))

    def evaluate(self, rows):
        return evaluate({"family": prefix_exact.prefix_policy()["family"], "positions": rows, "prompt_end": [], "detokenized": {}},
                        qwen_reference(self.ids, -1.0))

    def test_rows_equal_to_cold_pass(self):
        result = self.evaluate(aligned_rows(self.ids, -1.10, -1.10, -1.10))
        self.assertTrue(result["passed"], result["checks"])

    def test_a_hit_within_the_accuracy_margin_but_not_equal_to_cold_fails(self):
        # Exact -1.00, cold and chunked -1.10, a faulty hit -1.05: closer to
        # exact, so the accuracy rule passes it, but it moved.
        result = self.evaluate(aligned_rows(self.ids, -1.10, -1.10, -1.05))
        self.assertTrue(result["checks"]["hit_vs_cold"])
        self.assertTrue(result["checks"]["chunked_equals_cold"])
        self.assertFalse(result["checks"]["hit_equals_cold"])
        self.assertFalse(result["passed"])

    def test_a_chunked_row_that_differs_from_cold_fails(self):
        result = self.evaluate(aligned_rows(self.ids, -1.10, -1.10000001, -1.10))
        self.assertTrue(result["checks"]["hit_equals_cold"])
        self.assertFalse(result["checks"]["chunked_equals_cold"])
        self.assertFalse(result["passed"])

    def test_mimo_requires_equal_rows_too(self):
        rows = aligned_rows(self.ids, -1.10, -1.10, -1.05)
        mimo = qwen_reference(self.ids, -1.0)
        mimo["schema"] = "chopinfer.mimo26.teacher-exact-reference.v3"
        with mock.patch.dict(os.environ, {"CHOPINFER_QUAL_CONFIG": str(
                declaration.path("mimo26"))}):
            result = evaluate({"family": prefix_exact.prefix_policy()["family"], "positions": rows, "prompt_end": [], "detokenized": {}}, mimo)
        self.assertFalse(result["checks"]["hit_equals_cold"])
        self.assertFalse(result["passed"])


def split_row(anchor, hit, control, position=None):
    """A split row at an anchor on Qwen's granule; `anchor` is the primer's end."""
    position = position or anchor + 1
    return {"text": "a", "position": position, "anchor": anchor,
            "hit": {"token_ids": [7] * prefix_exact.SPLIT_TOKENS,
                    "logprob": [hit] * prefix_exact.SPLIT_TOKENS,
                    "top": [{"7": hit}] * prefix_exact.SPLIT_TOKENS},
            "control": {"token_ids": [7] * prefix_exact.SPLIT_TOKENS,
                        "logprob": [control] * prefix_exact.SPLIT_TOKENS,
                        "top": [{"7": control}] * prefix_exact.SPLIT_TOKENS},
            "cached": {"primer": 0, "hit": anchor, "control": 0}}


class SplitTests(unittest.TestCase):
    def setUp(self):
        selected = mock.patch.dict(os.environ, {"CHOPINFER_QUAL_CONFIG": str(
            declaration.path("qwen38"))})
        selected.start()
        self.addCleanup(selected.stop)

    def test_split_positions_resume_at_multiples_of_the_chunk(self):
        scored = prefix_exact.split_positions([list(range(300))], 65, (1, 2, 65))
        self.assertEqual([(p, anchor) for _, p, anchor in scored][:4],
                         [(66, 65), (67, 65), (130, 65), (131, 130)])
        self.assertTrue(all(anchor % 65 == 0 and 0 < p - anchor <= 65 for _, p, anchor in scored))

    def test_a_split_record_passes_only_when_every_hit_equals_its_control(self):
        rows = [split_row(64, -0.5, -0.5), split_row(128, -0.25, -0.25)]
        self.assertTrue(prefix_exact.summarize_split(rows, 64)["passed"])
        rows.append(split_row(192, -0.5, -0.500001))
        summary = prefix_exact.summarize_split(rows, 64)
        self.assertFalse(summary["passed"])
        self.assertEqual(summary["differing"], [{"text": "a", "position": 193, "anchor": 192}])
        # A hit at the primer's end, off the granule, is not what Qwen captures.
        self.assertFalse(prefix_exact.summarize_split([split_row(100, -0.5, -0.5)], 64)["passed"])

    def test_a_split_hit_resumes_the_granule_below_its_primer_and_matches_cold(self):
        ids = list(range(1000, 1400))
        server = FakeServer(64, ids, granule=64)
        # The fake forces the next reference token, so stop one short.
        prompt = ids[:-1]
        rows = [prefix_exact.score_split(server, "m", "a", prompt, p, anchor, "run", tokens=1)
                for _, p, anchor in prefix_exact.split_positions([prompt], 100)]
        summary = prefix_exact.summarize_split(rows, 64, tokens=1)
        self.assertEqual(summary["cache_errors"], [])
        self.assertTrue(summary["passed"])
        self.assertEqual(summary["anchors"], [100, 200, 300])
        self.assertEqual({row["cached"]["hit"] for row in rows}, {64, 192, 256})

    def split_passes(self, mismatch=None):
        barrier = threading.Barrier(4)
        active = 0
        lock = threading.Lock()
        observed = []

        def score(client, model, name, ids, position, anchor, run_id, tokens):
            nonlocal active
            with lock:
                active += 1
                observed.append((tokens, active, run_id))
            try:
                if tokens == 1:
                    barrier.wait(timeout=5)
                result = split_row(anchor, -0.5, -0.5)
                for arm in ("hit", "control"):
                    result[arm] = {"token_ids": [7] * tokens,
                                   "logprob": [-0.5] * tokens,
                                   "top": [{"7": -0.5}] * tokens}
                # A sound server may change decode rows when requests overlap.
                if tokens > 1 and active > 1:
                    result["hit"]["logprob"][-1] -= 0.25
                if mismatch == "prefill" and tokens == 1:
                    result["hit"]["top"][0] = {"7": -0.75}
                if mismatch == "decode" and tokens > 1:
                    result["hit"]["logprob"][-1] -= 0.25
                return result
            finally:
                with lock:
                    active -= 1

        entries = [(0, anchor + 1, anchor) for anchor in (64, 128, 192, 256)]
        with mock.patch.object(prefix_exact, "score_split", side_effect=score):
            rows = prefix_exact.score_split_passes(None, "m", [("a", list(range(300)))],
                                                   entries, "run", 4)
        self.assertEqual(max(count for tokens, count, _ in observed if tokens == 1), 4)
        self.assertTrue(all(count == 1 for tokens, count, _ in observed if tokens > 1))
        self.assertEqual({run_id for tokens, _, run_id in observed if tokens == 1},
                         {"run/prefill"})
        self.assertEqual({run_id for tokens, _, run_id in observed if tokens > 1},
                         {"run/continuation"})
        return prefix_exact.summarize_split(rows, 64)

    def test_concurrent_prefill_and_controlled_decode_pass(self):
        self.assertTrue(self.split_passes()["passed"])

    def test_concurrent_prefill_mismatch_fails(self):
        self.assertFalse(self.split_passes("prefill")["passed"])

    def test_controlled_decode_mismatch_fails_and_reports_its_delta(self):
        summary = self.split_passes("decode")
        self.assertFalse(summary["passed"])
        self.assertEqual(summary["max_abs_dlogprob"], 0.25)

    def test_missing_or_misaligned_generated_rows_fail_in_both_passes(self):
        for phase, expected in (("continuation", prefix_exact.SPLIT_TOKENS), ("prefill", 1)):
            for lengths in ((0, 0, 0), (expected - 1,) * 3, (expected + 1,) * 3,
                            (expected, expected - 1, expected),
                            (expected, expected, expected - 1),
                            (expected - 1, expected, expected)):
                with self.subTest(phase=phase, lengths=lengths):
                    result = split_row(64, -0.5, -0.5)
                    first = split_row(64, -0.5, -0.5)
                    for arm in ("hit", "control"):
                        first[arm] = {key: values[:1] for key, values in first[arm].items()}
                    result["concurrent_prefill"] = first
                    target = first if phase == "prefill" else result
                    for arm in ("hit", "control"):
                        target[arm] = {key: [value] * count for key, value, count in zip(
                            ("token_ids", "logprob", "top"),
                            (7, -0.5, {"7": -0.5}), lengths)}
                    self.assertFalse(prefix_exact.summarize_split([result], 64)["passed"])

    def test_max_delta_includes_later_generated_rows(self):
        result = split_row(64, -0.5, -0.5)
        result["hit"]["logprob"][-1] = -0.75
        self.assertEqual(prefix_exact.summarize_split([result], 64)["max_abs_dlogprob"], 0.25)


class GateTests(unittest.TestCase):
    def setUp(self):
        ids = [10, 11, 12, 13, 14, 15]
        self.exact = {"texts": [text("a", ids, [-1.0] * 5, ids[1:]),
                                text("b", ids, [-1.0] * 5, ids[1:])]}
        self.strings = {}

    def scored(self, hit_shift, chunked_shift=0.0):
        rows = []
        for name in ("a", "b"):
            for position in range(1, 6):
                cold = row(-1.0, {"x": -1.0})
                rows.append({"text": name, "category": "prose", "position": position,
                             "anchor": 0, "cold": cold,
                             "chunked": row(-1.0 - chunked_shift, {"x": -1.0 - chunked_shift}),
                             "hit": row(-1.0 - hit_shift, {"x": -1.0 - hit_shift})})
        return rows

    def test_rows_equal_to_cold_pass_the_accuracy_gate(self):
        report = summarize(self.scored(0.0), self.exact, self.strings)
        self.assertEqual(report["positions"], 10)
        self.assertEqual(report["distances"]["hit_vs_cold"]["bitwise_equal_rows"], 10)
        self.assertEqual(report["distances"]["chunked_vs_cold"]["bitwise_equal_rows"], 10)
        self.assertTrue(report["gate_hit_vs_cold"]["passed"])
        self.assertNotIn("gate_chunking_floor", report)

    def test_the_accuracy_gate_bounds_how_far_a_hit_moved(self):
        report = summarize(self.scored(0.02, 0.02), self.exact, self.strings)
        self.assertEqual(report["distances"]["hit_vs_cold"]["bitwise_equal_rows"], 0)
        self.assertAlmostEqual(report["vs_exact"]["hit"]["mean_abs_dlogprob"], 0.02)
        self.assertAlmostEqual(report["paired_vs_exact"]["hit_minus_cold"]["mean"], 0.02)
        # 0.02 beyond cold exceeds the 0.005 row margin against exact.
        self.assertFalse(report["gate_hit_vs_cold"]["passed"])
        self.assertTrue(summarize(self.scored(0.004, 0.004), self.exact,
                                  self.strings)["gate_hit_vs_cold"]["passed"])

    def test_top1_flips_against_exact_count_toward_the_gate(self):
        rows = self.scored(0.0)
        for entry in rows:
            # The hit ranks another token above the reference token.
            entry["hit"] = row(-1.0, {"x": -1.0, "y": -0.5})
        report = summarize(rows, self.exact, self.strings)
        self.assertEqual(report["vs_exact"]["hit"]["top1_disagreement"], 1.0)
        self.assertEqual(report["gate_hit_vs_cold"]["top1_test_only"], 10)
        self.assertFalse(report["gate_hit_vs_cold"]["passed"])
        self.assertEqual(report["distances"]["hit_vs_cold"]["top1_differs"], 10)


def serving_server():
    serving = {
        "checkpoint_identity": "a" * 64, "package_identity": "b" * 64,
        "metadata_identity": "c" * 64, "startup_nonce": "d" * 32,
        "source_commit": "e" * 40, "bundle_id": "fixture.v1", "version": "1",
        "served_model_names": ["fixture"],
        "profile": {"name": "tp2", "tensor_parallel_size": 2, "kv_cache_dtype": "bf16",
                    "max_sequences": 32, "prefix_cache": True,
                    "speculation": {"mode": "off", "drafter": None, "tokens": 0}}}
    return {"model": "fixture", "health": {"serving": serving},
            "checkpoint": {key: serving[key] for key in (
                "checkpoint_identity", "package_identity", "metadata_identity", "startup_nonce")}}


class ReportRefusalTests(unittest.TestCase):
    def test_missing_main_prompt_end_and_split_anchors_are_clear_refusals(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            exact = root / "exact.json"
            ids = list(range(70))
            exact.write_text(json.dumps({"schema": prefix_exact.prefix_policy()["reference_schema"],
                                         "texts": [text("a", ids, [-1.0] * 69, ids[1:])]}))
            scored = {"text": "a", "category": "prose", "position": 65, "anchor": 64,
                      **{arm: row(-1.0, {"65": -1.0}) for arm in prefix_exact.ARMS},
                      "cached": {"primer": 0, "cold": 0, "chunked": 0, "hit": 64}}
            main = {"family": "mimo26", "server": serving_server(), "run_id": "test",
                    "reference_sha256": hashlib.sha256(exact.read_bytes()).hexdigest(),
                    "block_tokens": 64, "top_logprobs": 5, "positions": [scored],
                    "prompt_end": [], "detokenized": {"65": "65"}}
            for group in ("positions", "prompt_end", "rows"):
                with self.subTest(group=group):
                    recorded = copy.deepcopy(main)
                    split = {"family": "mimo26", "server": serving_server(),
                             "rows": [split_row(64, -1.0, -1.0)]}
                    if group == "prompt_end":
                        recorded[group] = [copy.deepcopy(scored)]
                        del recorded[group][0]["chunked"]
                        del recorded[group][0]["cached"]["chunked"]
                    target = split if group == "rows" else recorded
                    del target[group][0]["anchor"]
                    record_path, split_path = root / "record.json", root / "split.json"
                    record_path.write_text(json.dumps(recorded))
                    split_path.write_text(json.dumps(split))
                    result = subprocess.run(
                        [sys.executable, "-B", prefix_exact.__file__, "report", "--exact", str(exact),
                         "--record", str(record_path), "--split-record", str(split_path)],
                        capture_output=True, text=True, timeout=30)
                    self.assertEqual(result.returncode, 1, result.stderr)
                    self.assertNotIn("Traceback", result.stderr)
                    self.assertIn(f"{group}[0] is missing anchor", result.stderr)
                    self.assertIn(str(split_path if group == "rows" else record_path), result.stderr)


    def test_a_slice_report_takes_one_shard_record_alone(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            exact = root / "exact.json"
            ids = list(range(70))
            exact.write_text(json.dumps({"schema": prefix_exact.prefix_policy()["reference_schema"],
                                         "texts": [text("a", ids, [-1.0] * 69, ids[1:])]}))
            whole = {"family": "fixture", "server": serving_server(), "run_id": "test",
                     "reference_sha256": hashlib.sha256(exact.read_bytes()).hexdigest(),
                     "block_tokens": 64, "top_logprobs": 5, "positions": [], "prompt_end": [],
                     "detokenized": {}}
            path = root / "record.json"
            path.write_text(json.dumps(whole))
            result = subprocess.run([sys.executable, "-B", prefix_exact.__file__, "report", "--slice", "--exact",
                                     str(exact), "--record", str(path), "--record", str(path)],
                                    capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 1, result.stderr)
            self.assertIn("--slice reports one shard record", result.stderr)


class CoverageTests(unittest.TestCase):
    def setUp(self):
        ids = list(range(1000, 1140))
        self.exact = {"texts": [text("a", ids, [], []), text("b", ids[:100], [], []),
                                text("short", ids[:40], [], [])]}

    def part(self, names, stride=1, **checks):
        rows = [{"text": entry["name"], "position": position, "anchor": anchor}
                for entry, position, anchor in positions(
                    [entry for entry in self.exact["texts"] if entry["name"] in names],
                    stride=stride)]
        return {"family": "mimo26", "reference_sha256": "r", "block_tokens": 64, "top_logprobs": 5,
                "server": serving_server(), "run_id": "run", "positions": rows, "prompt_end": [], "detokenized": {"1": "x"},
                "media": checks.get("media"), "acceptance": checks.get("acceptance")}

    def test_records_over_disjoint_texts_cover_every_position_together(self):
        ran = {"media": {"passed": True}, "acceptance": {"passed": True}}
        whole = self.part({"a", "b"}, **ran)
        self.assertEqual(prefix_exact.skipped_checks(whole, self.exact), [])
        first, second = self.part({"a"}, **ran), self.part({"b"})
        self.assertEqual(prefix_exact.skipped_checks(first, self.exact), ["full position set"])
        merged = prefix_exact.merged_records([first, second])
        self.assertEqual(prefix_exact.skipped_checks(merged, self.exact), [])
        self.assertEqual(merged["media"], {"passed": True})
        # A text scored twice, a strided record or a check no record ran leaves
        # the gate unpassed.
        for parts, missing in (([first, self.part({"a", "b"})], ["full position set"]),
                               ([self.part({"a", "b"}, stride=2, **ran)], ["full position set"]),
                               ([self.part({"a"}), second], ["media", "acceptance"])):
            self.assertEqual(prefix_exact.skipped_checks(prefix_exact.merged_records(parts),
                                                         self.exact), missing)

    def shard(self, index, count, **checks):
        rows = [{"text": entry["name"], "position": position, "anchor": anchor}
                for entry, position, anchor in prefix_exact.shard_entries(positions(self.exact["texts"]), index, count)]
        return {**self.part(set(), **checks), "positions": rows, "shard": {"index": index, "count": count}}

    def test_shards_slice_every_position_once_balanced_by_prefill(self):
        entries = positions(self.exact["texts"])
        for count in (1, 2, 3, 5):
            with self.subTest(count=count):
                slices = [prefix_exact.shard_entries(entries, index, count) for index in range(1, count + 1)]
                # Contiguous and complete: the slices concatenate to the whole set.
                self.assertEqual([entry for part in slices for entry in part], entries)
                costs = [sum(position for _, position, _ in part) for part in slices]
                total = sum(costs)
                self.assertLessEqual(max(costs) - min(costs), max(position for _, position, _ in entries) + total * 0.05)
        for bad in ("0/2", "3/2", "1", "a/b", ""):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                prefix_exact.parse_shard(bad)

    def test_shard_records_cover_every_position_together(self):
        ran = {"media": {"passed": True}, "acceptance": {"passed": True}}
        shards = [self.shard(1, 3, **ran), self.shard(2, 3), self.shard(3, 3)]
        # Each shard is complete for its slice; shard 1 alone owes media and acceptance.
        self.assertEqual([prefix_exact.skipped_checks(value, self.exact) for value in shards], [[], [], []])
        self.assertEqual(prefix_exact.skipped_checks(self.shard(1, 3), self.exact), ["media", "acceptance"])
        merged = prefix_exact.merged_records(shards)
        self.assertIsNone(merged["shard"])
        self.assertEqual(prefix_exact.skipped_checks(merged, self.exact), [])
        # A missing, repeated or differently counted shard is refused or leaves the set incomplete.
        for parts in ([shards[0], shards[1]], [shards[0], shards[1], shards[1]],
                      [shards[0], shards[1], self.shard(3, 4)], [shards[0], shards[1], self.part({"a"})]):
            with self.subTest(parts=[value.get("shard") for value in parts]), self.assertRaises(SystemExit):
                prefix_exact.merged_records(parts)

    def test_a_larger_recorded_block_does_not_reduce_required_coverage(self):
        recorded = self.part({"a", "b"}, media={"passed": True}, acceptance={"passed": True})
        recorded["block_tokens"] = 2048
        recorded["positions"] = []
        self.assertIn("full position set", prefix_exact.skipped_checks(recorded, self.exact))

    def test_every_anchor_must_match_policy_even_with_a_matching_hit_count(self):
        for config in ("mimo26", "qwen38"):
            with self.subTest(config=config), mock.patch.dict(os.environ, {
                    "CHOPINFER_QUAL_CONFIG": str(declaration.path(config))}):
                recorded = self.part({"a", "b"}, media={"passed": True}, acceptance={"passed": True})
                recorded["prompt_end"] = [
                    {"text": entry["name"], "position": position, "anchor": anchor}
                    for entry, position, anchor in prompt_end_positions(self.exact["texts"])]
                self.assertEqual(prefix_exact.skipped_checks(recorded, self.exact), [])
                for arm in ("positions", "prompt_end"):
                    changed = copy.deepcopy(recorded)
                    changed[arm][-1]["anchor"] -= 1
                    changed[arm][-1]["cached"] = {"hit": prefix_exact.granule_floor(
                        changed[arm][-1]["anchor"])}
                    self.assertIn("full position set", prefix_exact.skipped_checks(changed, self.exact))

    def test_cache_arm_keys_cannot_define_the_expected_arm_set(self):
        scored = {"text": "a", "position": 70, "anchor": 64,
                  "cached": {"primer": 0, "hit": 64, "cold": 0}}
        self.assertTrue(cache_errors([scored], 64))

    def test_combining_records_rejects_each_identity_difference(self):
        changes = {"checkpoint_identity": "f" * 64, "package_identity": "f" * 64,
                   "metadata_identity": "f" * 64, "source_commit": "f" * 40,
                   "bundle_id": "other.v1", "version": "2", "served_model_names": ["other"]}
        for key, value in changes.items():
            with self.subTest(field=key):
                first, second = self.part({"a"}), self.part({"b"})
                second["server"]["health"]["serving"][key] = value
                if key in second["server"]["checkpoint"]:
                    second["server"]["checkpoint"][key] = value
                with self.assertRaises(SystemExit):
                    prefix_exact.merged_records([first, second])
        for key, value in {"name": "other", "tensor_parallel_size": 4, "kv_cache_dtype": "fp8",
                           "max_sequences": 16, "prefix_cache": False,
                           "speculation": {"mode": "adaptive", "drafter": "mtp", "tokens": 3}}.items():
            with self.subTest(profile=key):
                first, second = self.part({"a"}), self.part({"b"})
                second["server"]["health"]["serving"]["profile"][key] = value
                with self.assertRaises(SystemExit):
                    prefix_exact.merged_records([first, second])
        first, second = self.part({"a"}), self.part({"b"})
        second["server"]["model"] = "other"
        with self.assertRaises(SystemExit):
            prefix_exact.merged_records([first, second])

    def test_missing_identity_and_inconsistent_binding_are_rejected(self):
        for key in serving_server()["health"]["serving"]:
            with self.subTest(missing=key):
                part = self.part({"a"})
                del part["server"]["health"]["serving"][key]
                with self.assertRaises(SystemExit):
                    prefix_exact.merged_records([part])
        for key in serving_server()["checkpoint"]:
            with self.subTest(binding=key):
                part = self.part({"a"})
                part["server"]["checkpoint"][key] = "f" * len(part["server"]["checkpoint"][key])
                with self.assertRaises(SystemExit):
                    prefix_exact.merged_records([part])

    def test_restarts_require_equal_build_and_preserve_input_identities(self):
        first, second = self.part({"a"}), self.part({"b"})
        second["server"]["checkpoint"]["startup_nonce"] = "f" * 32
        second["server"]["health"]["serving"]["startup_nonce"] = "f" * 32
        merged = prefix_exact.merged_records([first, second])
        self.assertEqual(merged["serving_identity"]["source_commit"], "e" * 40)
        self.assertEqual([entry["server"] for entry in merged["inputs"]],
                         [first["server"], second["server"]])

    def test_a_check_run_by_several_records_must_pass_in_each(self):
        merged = prefix_exact.merged_records([
            self.part({"a"}, media={"passed": True}), self.part({"b"}, media={"passed": False})])
        self.assertFalse(merged["media"]["passed"])
        with self.assertRaises(SystemExit):
            prefix_exact.merged_records([self.part({"a"}), {**self.part({"b"}), "block_tokens": 32}])


class AcceptanceTests(unittest.TestCase):
    def run_row(self, hit, cold, anchor=64):
        def arm(accepted, cached):
            return {"cached_tokens": cached, "token_ids": [1, 2], "draft_tokens": 100,
                    "accepted_tokens": accepted}
        return {"prompt_tokens": 100, "anchor": anchor, "hit": arm(hit, anchor),
                "chunked": arm(hit, 0), "cold": arm(cold, 0)}

    def test_acceptance_after_hits_may_trail_cold_only_within_the_margin(self):
        close = summarize_acceptance([self.run_row(60, 61)] * 4, 64)
        self.assertTrue(close["passed"])
        self.assertEqual(close["hit_tokens_equal_chunked"], 4)
        far = summarize_acceptance([self.run_row(40, 70)] * 4, 64)
        self.assertFalse(far["passed"])

    def test_continuations_that_differ_from_cold_fail(self):
        for arm in ("hit", "chunked"):
            run = self.run_row(60, 60)
            run[arm]["token_ids"] = [1, 3]
            result = summarize_acceptance([run], 64)
            self.assertEqual(result["differing_tokens"], [0])
            self.assertFalse(result["passed"])
            for value in run.values():
                if isinstance(value, dict):
                    value["draft_tokens"] = 0
            self.assertFalse(summarize_acceptance([run], 64)["passed"])

    def test_a_missed_hit_fails_and_no_drafts_skip(self):
        missed = self.run_row(60, 60)
        missed["hit"]["cached_tokens"] = 0
        self.assertFalse(summarize_acceptance([missed], 64)["passed"])
        idle = self.run_row(0, 0)
        for arm in ("hit", "chunked", "cold"):
            idle[arm]["draft_tokens"] = 0
        result = summarize_acceptance([idle], 64)
        self.assertTrue(result["passed"])
        self.assertIn("skipped", result)


class FakeServer:
    """The anchor rules of the prefix-cache contract for token prompts, with
    forced outputs: runtime-scheduler `capture_points` for requests without
    followers (shared prefix, prompt end and final-chunk start, on
    `granule`). `block` is the scheduler's prefix block, one page on every
    rank of a context-parallel group; `chunk` the prefill chunk."""

    def __init__(self, block, reference, granule=1, chunk=prefix_exact.PREFILL_CHUNK_TOKENS):
        self.block, self.reference, self.granule, self.chunk = block, reference, granule, chunk
        self.anchors, self.seen, self.drafts = set(), set(), 0

    def request(self, path, body=None, stream=False):
        if path == "/detokenize":
            return {"prompt": f"t{body['tokens'][0]}"}, 0.0
        ids, salt = body["prompt"], body["cache_salt"]
        hit = max((b for b in range(1, len(ids)) if (salt, tuple(ids[:b])) in self.anchors),
                  default=0)
        seen = 0
        while (seen + 1) * self.block <= len(ids) and \
                (salt, tuple(ids[:(seen + 1) * self.block])) in self.seen:
            seen += 1

        def absent(tokens):
            return tokens >= self.block and (salt, tuple(ids[:tokens])) not in self.anchors

        points = []
        shared = min(seen * self.block, (len(ids) - 1) // self.block * self.block)
        shared = shared // self.granule * self.granule
        if shared > hit and not (hit > 0 and len(ids) - hit <= self.chunk) and absent(shared):
            points.append(shared)
        end = len(ids) // self.granule * self.granule
        if end > hit and end >= self.block:
            points.append(end)
        start = max([hit] + [tokens for tokens in points if tokens < len(ids)])
        repeat = (start + max(len(ids) - start - 1, 0) // self.chunk * self.chunk) // self.granule * self.granule
        if repeat > start and absent(repeat):
            points.append(repeat)
        self.anchors.update((salt, tuple(ids[:tokens])) for tokens in points)
        self.seen.update((salt, tuple(ids[:end]))
                         for end in range(self.block, len(ids) + 1, self.block))
        forced = self.reference[len(ids)] if ids == self.reference[:len(ids)] else 0
        tokens = [forced] * body["max_tokens"]
        self.drafts += body["max_tokens"]
        return {"choices": [{"token_ids": tokens, "logprobs": {
                    "token_logprobs": [-0.5] * len(tokens),
                    "top_logprobs": [{"t1": -0.5}] * len(tokens)}}],
                "usage": {"prompt_tokens_details": {"cached_tokens": hit}}}, 0.0

    def metrics(self, _client):
        return (f"chop_spec_verify_items_total {self.drafts}\n"
                f"chop_spec_draft_tokens_total {self.drafts}\n"
                f"chop_spec_accepted_tokens_total {self.drafts // 2}\n"
                f'chop_spec_drafted_position_total{{position="1"}} {self.drafts}\n'
                f'chop_spec_accepted_position_total{{position="1"}} {self.drafts // 2}\n')


class ProtocolTests(unittest.TestCase):
    def test_each_position_hits_only_in_the_hit_arm(self):
        ids = list(range(1000, 1200))
        entry = text("a", ids, [-0.5] * 199, ids[1:])
        for granule, config in ((1, "mimo26"), (64, "qwen38")):
            server = FakeServer(64, ids, granule)
            with self.subTest(config=config), mock.patch.dict(os.environ, {
                    "CHOPINFER_QUAL_CONFIG": str(declaration.path(config))}):
                scored = [prefix_exact.score_position(server, "m", entry, p, anchor, "run")
                          for _, p, anchor in positions([entry])]
                self.assertEqual(len(scored), 135)
                self.assertEqual(cache_errors(scored, 64), [])
                self.assertEqual({row["hit"]["logprob"] for row in scored}, {-0.5})
                self.assertTrue(prefix_exact.equal_rows(scored, "hit"))

    def test_a_primer_ending_one_past_the_anchor_turns_the_chunked_arm_into_a_hit(self):
        # Before the primer ran a block past the anchor, Qwen's prompt-end
        # anchor at the granule below the primer's end served the chunked arm.
        ids = list(range(1000, 1200))
        server = FakeServer(64, ids, granule=64)
        short = ids[:128] + [ids[0]]
        complete = prefix_exact.complete
        self.assertEqual(complete(server, "m", short, "s")["cached_tokens"], 0)
        self.assertEqual(complete(server, "m", ids[:150], "s")["cached_tokens"], 128)
        self.assertEqual(complete(server, "m", primer(ids, 128), "t")["cached_tokens"], 0)
        self.assertEqual(complete(server, "m", ids[:150], "t")["cached_tokens"], 0)

    def test_acceptance_runs_hit_at_the_last_block_boundary(self):
        server = FakeServer(64, [])
        prompts = [list(range(1000 + index, 1100 + index)) for index in range(2)]
        with mock.patch.object(prefix_exact, "acceptance_prompts", return_value=prompts), \
                mock.patch.object(prefix_exact, "metrics_text", server.metrics):
            result = prefix_exact.acceptance_check(server, "m", "run", 2, 8, 64)
        self.assertEqual(result["cache_errors"], [])
        self.assertEqual([run["hit"]["cached_tokens"] for run in result["runs"]], [64, 64])
        self.assertEqual(result["totals"]["hit"], {"draft_tokens": 16, "accepted_tokens": 8,
                                                   "rate": 0.5})
        self.assertTrue(result["passed"])


def grouped_server(page=64, size=None):
    """serving_server() on a host that reports its KV page tokens and, given
    `size`, a context-parallel size."""
    server = serving_server()
    server["health"]["serving"]["kv_page_tokens"] = page
    if size is not None:
        server["health"]["serving"]["profile"]["context_parallel_size"] = size
    return server


class ContextParallelReuseTests(unittest.TestCase):
    """At context-parallel size N the scheduler finds shared prefixes and
    captures prompt ends in stripes of N pages (GPU job QDCP2 09 at size 2:
    every hit below 128 tokens resumed nothing, and a prompt ending on a
    page boundary past an odd page resumed one page less)."""

    def setUp(self):
        self.granule(64)

    def granule(self, granule):
        """The selected policy with 64-token blocks and this prefill granule."""
        policy = {**prefix_exact.prefix_policy(), "block_tokens": 64, "prefill_granule": granule}
        selected = mock.patch.object(prefix_exact, "prefix_policy", return_value=policy)
        selected.start()
        self.addCleanup(selected.stop)

    def test_the_reuse_unit_is_the_servers_page_on_every_group_rank(self):
        reuse = prefix_exact.reuse_tokens
        self.assertEqual([reuse(grouped_server(64, size)) for size in (None, 1, 2, 4)], [64, 64, 128, 256])
        self.assertEqual(reuse(grouped_server(32, 2)), 64)
        # A host that predates kv_page_tokens served size 1 over the declared block.
        self.assertEqual(reuse(serving_server()), prefix_exact.prefix_policy()["block_tokens"])
        for server in (grouped_server(None, 2), grouped_server(0, 2), grouped_server(64, 0),
                       grouped_server("64", 2)):
            with self.assertRaisesRegex(SystemExit, "refused"):
                reuse(server)
        self.assertEqual(prefix_exact.record_reuse({"server": grouped_server(64, 2), "reuse_tokens": 128}), 128)
        with self.assertRaisesRegex(SystemExit, "scored with reuse_tokens 64"):
            prefix_exact.record_reuse({"server": grouped_server(64, 2), "reuse_tokens": 64})

    def test_without_context_parallelism_every_expectation_is_unchanged(self):
        entry = text("a", list(range(5000)), [], [])
        for granule in (1, 64):
            with self.subTest(granule=granule):
                self.granule(granule)
                for _, position, anchor in positions([entry]):
                    self.assertEqual(prefix_exact.resumed(position, anchor, 64),
                                     prefix_exact.granule_floor(anchor), position)
                anchors = [anchor for _, _, anchor in prompt_end_positions([entry])]
                anchors += [anchor for _, _, anchor in prefix_exact.split_positions(
                    [entry["prompt_token_ids"]], 2048)]
                for anchor in anchors:
                    self.assertEqual(prefix_exact.prompt_end_resumed(anchor, 64),
                                     prefix_exact.granule_floor(anchor), anchor)

    def test_hits_follow_the_servers_stripe_at_every_context_parallel_size(self):
        ids = list(range(1000, 1700))
        entry = text("a", ids, [-0.5] * (len(ids) - 1), ids[1:])
        expected = {1: {100: 64, 200: 192, 256: 192, 300: 256, 512: 448, 520: 512},
                    2: {100: 0, 200: 192, 256: 128, 300: 256, 512: 384, 520: 512},
                    4: {100: 0, 200: 0, 256: 0, 300: 256, 512: 256, 520: 512}}
        for size, hits in expected.items():
            reuse = 64 * size
            with self.subTest(size=size):
                server = FakeServer(reuse, ids, granule=64)
                scored = [prefix_exact.score_position(server, "m", entry, p, anchor, "run")
                          for _, p, anchor in positions([entry])]
                prompt_end = [prefix_exact.score_prompt_end(server, "m", entry, p, anchor, "run")
                              for _, p, anchor in prompt_end_positions([entry])]
                self.assertEqual(cache_errors(scored, reuse), [])
                self.assertEqual(cache_errors(prompt_end, reuse, PROMPT_END_ARMS), [])
                self.assertEqual({row["position"]: row["cached"]["hit"] for row in scored
                                  if row["position"] in hits}, hits)
                self.assertTrue(all(row["cached"]["hit"] == (row["anchor"] // 64 * 64 >= reuse) * row["anchor"]
                                    // 64 * 64 for row in prompt_end))
                # The expectations of a server without context parallelism fail
                # here, as on job 09's GPU server.
                self.assertEqual(not cache_errors(scored, 64), size == 1)
                self.assertEqual(not cache_errors(prompt_end, 64, PROMPT_END_ARMS), size == 1)
                recorded = {"family": prefix_exact.prefix_policy()["family"], "server": grouped_server(64, size),
                            "reuse_tokens": reuse,
                            "positions": scored, "prompt_end": prompt_end, "detokenized": {}}
                reference = {"schema": prefix_exact.prefix_policy()["reference_schema"], "texts": [entry]}
                result = evaluate(recorded, reference)
                self.assertTrue(result["checks"]["cache"])
                self.assertEqual(result["reuse_tokens"], reuse)
                if size > 1:
                    # The same rows from a server that reports another size fail.
                    with self.assertRaisesRegex(SystemExit, "scored with reuse_tokens"):
                        evaluate({**recorded, "server": grouped_server(64, 1)}, reference)
                    del recorded["reuse_tokens"]
                    recorded["server"] = grouped_server(64, 1)
                    self.assertFalse(evaluate(recorded, reference)["checks"]["cache"])

    def test_a_final_chunk_anchor_past_the_shared_units_is_resumed(self):
        """The chunked arm's final chunk starts a whole number of chunks
        after its shared-prefix capture; when that lies past the shared
        units, the hit resumes it (a 2026-10-07 review case: a 4,096-token unit, 2,048-token
        chunks)."""
        ids = list(range(1000, 1700))
        entry = text("a", ids, [-0.5] * (len(ids) - 1), ids[1:])
        # Chunks below a unit plus a page, where the final chunk can start
        # past the shared units (without it: 128, 256, 384 and 512; 256).
        for reuse, chunk, hits in ((128, 64, {256: 192, 384: 320, 512: 448, 640: 576}),
                                   (256, 128, {448: 384, 512: 384, 300: 256})):
            with self.subTest(reuse=reuse), mock.patch.object(prefix_exact, "PREFILL_CHUNK_TOKENS", chunk):
                server = FakeServer(reuse, ids, granule=64, chunk=chunk)
                scored = [prefix_exact.score_position(server, "m", entry, p, anchor, "run")
                          for _, p, anchor in positions([entry])]
                self.assertEqual(cache_errors(scored, reuse), [])
                self.assertEqual({row["position"]: row["cached"]["hit"] for row in scored
                                  if row["position"] in hits}, hits)
                prompt_end = [prefix_exact.score_prompt_end(server, "m", entry, p, anchor, "run")
                              for _, p, anchor in prompt_end_positions([entry])]
                self.assertEqual(cache_errors(prompt_end, reuse, PROMPT_END_ARMS), [])
        with self.subTest(reuse=4096):
            self.assertEqual(prefix_exact.resumed(6208, 6144, 4096), 6144)

    def test_acceptance_and_split_hits_follow_the_stripe(self):
        prompts = [list(range(1000, 1256)), list(range(2000, 2346))]
        for size, hits in ((1, [192, 320]), (2, [128, 320]), (4, [0, 320])):
            reuse = 64 * size
            with self.subTest(size=size):
                server = FakeServer(reuse, [], granule=64)
                with mock.patch.object(prefix_exact, "acceptance_prompts", return_value=prompts), \
                        mock.patch.object(prefix_exact, "metrics_text", server.metrics):
                    result = prefix_exact.acceptance_check(server, "m", "run", 2, 8, reuse)
                self.assertEqual([run["hit"]["cached_tokens"] for run in result["runs"]], hits)
                self.assertEqual(result["cache_errors"], [])
                self.assertEqual(summarize_acceptance(result["runs"], 64)["cache_errors"] == [], size == 1)
        split = split_row(2048, -0.5, -0.5)
        for reuse in (64, 128, 256):
            self.assertEqual(prefix_exact.summarize_split([split], reuse)["cache_errors"], [])


class MediaTests(unittest.TestCase):
    def merged_blocks(self, width, height):
        """Block colors in the order the patch grid emits merged 32x32 blocks."""
        return [block_colors(width, height, bw * 32, bh * 32)
                for bh in range(height // 32) for bw in range(width // 32)]

    def test_the_geometry_pair_has_identical_patch_rows_in_different_grids(self):
        _, _, first, second = dict((pair[0], pair) for pair in MEDIA_PAIRS)["geometry"]
        self.assertEqual(first[:2], (64, 128))
        self.assertEqual(second[:2], (128, 64))
        # Solid blocks, one color each, emitted in the same order: equal payloads.
        self.assertEqual(self.merged_blocks(64, 128), self.merged_blocks(128, 64))
        self.assertEqual(len(set(self.merged_blocks(64, 128))), 8)

    def test_every_pair_must_share_nothing_past_the_image(self):
        server = mock.Mock()

        def tokenize(_client, _model, messages):
            image = any(part["type"] == "image_url" for part in messages[0]["content"])
            return [1, 2] + [9] * image + [3] * 200

        def check(outcomes):
            iterator = iter(outcomes)
            with mock.patch.object(prefix_exact, "tokenize", side_effect=tokenize), \
                    mock.patch.object(prefix_exact, "metrics_text", return_value=""), \
                    mock.patch.object(prefix_exact, "chat", side_effect=lambda *_: next(iterator)):
                return prefix_exact.media_check(server, "m", "run", 64)

        def run(cached):
            return {"content": "x", "tokens": ["x"], "logprobs": [-0.1], "cached_tokens": cached}
        hit, miss, shared = run(64), run(0), run(64)
        # Per pair: three repeats, the other image, the other image cold.
        self.assertTrue(check([miss, miss, hit, miss, miss] * 2)["passed"])
        self.assertFalse(check([miss, miss, hit, miss, miss] + [miss, miss, hit, shared, miss])
                         ["passed"])
        # Without the positive control's hit the check proves nothing.
        self.assertFalse(check([miss, miss, miss, miss, miss] + [miss, miss, hit, miss, miss])
                         ["passed"])


class MediaChatServer:
    """A chat front end over FakeServer's anchor rules at `reuse`-token
    units. /tokenize renders a template around one token per two characters
    of text (at most `text_limit` of them) and one placeholder per 28-pixel
    cell of an image, whose id is the image's media key: a digest of its
    bytes, or one id for every image when `keyed` is false."""

    def __init__(self, reuse, keyed=True, text_limit=None):
        self.anchors = FakeServer(reuse, [], granule=64)
        self.keyed, self.text_limit = keyed, text_limit

    def ids(self, messages):
        ids = [1, 2]
        for part in messages[0]["content"]:
            if part["type"] == "image_url":
                data = base64.b64decode(part["image_url"]["url"].split(",", 1)[1])
                width, height = struct.unpack(">II", data[16:24])
                key = int(hashlib.sha256(data if self.keyed else b"").hexdigest()[:4], 16)
                ids += [3] + [100 + key % 9000] * (-(-width // 28) * -(-height // 28)) + [4]
            else:
                text = part["text"]
                ids += [10000 + ord(text[i]) * 128 + ord(text[i + 1:i + 2] or "\0")
                        for i in range(0, len(text), 2)][:self.text_limit]
        return ids + [5]

    def request(self, path, body=None, stream=False):
        if path == "/tokenize":
            return {"tokens": self.ids(body["messages"])}, 0.0
        response, _ = self.anchors.request("/v1/completions", {
            "prompt": self.ids(body["messages"]), "cache_salt": body["cache_salt"],
            "max_tokens": body["max_tokens"]})
        tokens = response["choices"][0]["token_ids"]
        return {"choices": [{"message": {"content": ""}, "logprobs": {"content": [
                    {"token": f"t{token}", "logprob": -0.5} for token in tokens]}}],
                "usage": response["usage"]}, 0.0


class MediaSizingTests(unittest.TestCase):
    """At context-parallel size 4 (a 256-token reuse unit) the geometry
    case's fixed 192-token text left its prompt short of one unit, so its
    positive control could never hit (a 2026-10-08 gate's dcp4-prefix check: cached tokens
    [0, 0, 0])."""

    def check(self, server, reuse, cases=None):
        with mock.patch.object(prefix_exact, "metrics_text",
                               return_value="chop_media_cache_hits_total 0\n"):
            return prefix_exact.media_check(server, "m", "run", reuse, cases)

    def test_each_prompt_spans_two_reuse_units_past_its_image_at_every_size(self):
        for reuse in (64, 128, 256):
            with self.subTest(reuse=reuse):
                result = self.check(MediaChatServer(reuse), reuse)
                self.assertTrue(result["passed"])
                self.assertEqual(result["reuse_tokens"], reuse)
                for case in result["cases"]:
                    self.assertEqual(case["reuse_tokens"], reuse)
                    for side in ("first", "second"):
                        self.assertGreaterEqual(case[side]["prompt_tokens"], case["media_start"] + 2 * reuse)
                    self.assertTrue(case["same_image_hit"])
                    self.assertGreater(case["first"]["cached_tokens"][2], case["media_start"])
                    self.assertLessEqual(case["second"]["cached_tokens"], case["media_start"])
                    self.assertFalse(case["shared_past_media"] or case["shared_encoder_rows"])
                # The first size keeps the old text, which already reached it.
                if reuse == 64:
                    self.assertEqual({case["text_tokens"] for case in result["cases"]},
                                     {prefix_exact.MEDIA_TEXT_TOKENS})

    def test_the_old_fixed_sizing_misses_the_positive_control_at_256(self):
        old = prefix_exact.media_cases(MediaChatServer(64), "m", "run", 64)
        self.assertEqual({case["text_tokens"] for case in old}, {prefix_exact.MEDIA_TEXT_TOKENS})
        result = self.check(MediaChatServer(256), 256, old)
        cases = {case["name"]: case for case in result["cases"]}
        self.assertFalse(result["passed"])
        self.assertLess(cases["geometry"]["first"]["prompt_tokens"], 256)
        self.assertEqual(cases["geometry"]["first"]["cached_tokens"], [0, 0, 0])
        self.assertFalse(cases["geometry"]["same_image_hit"])
        self.assertTrue(cases["color"]["passed"])

    def test_shared_anchors_past_the_image_still_fail_at_every_size(self):
        for reuse in (64, 128, 256):
            with self.subTest(reuse=reuse):
                result = self.check(MediaChatServer(reuse, keyed=False), reuse)
                self.assertFalse(result["passed"])
                self.assertTrue(all(case["shared_past_media"] and case["same_image_hit"]
                                    for case in result["cases"]))

    def test_a_case_that_cannot_reach_two_reuse_units_is_refused(self):
        server = MediaChatServer(256, text_limit=100)
        with self.assertRaisesRegex(SystemExit, "refused: media case 'color' spans .* short of its media "
                                    "start 2 plus two reuse units of 256 tokens"):
            prefix_exact.media_cases(server, "m", "run", 256)
        with self.assertRaisesRegex(SystemExit, "refused: media case 'color'"):
            self.check(server, 256)


class PngTests(unittest.TestCase):
    def test_png_is_a_valid_solid_image(self):
        data = png(3, 2, (1, 2, 3))
        self.assertEqual(data[:8], b"\x89PNG\r\n\x1a\n")
        width, height = struct.unpack(">II", data[16:24])
        self.assertEqual((width, height), (3, 2))
        offset, chunks = 8, {}
        while offset < len(data):
            length = struct.unpack(">I", data[offset:offset + 4])[0]
            kind = data[offset + 4:offset + 8]
            body = data[offset + 8:offset + 8 + length]
            crc = struct.unpack(">I", data[offset + 8 + length:offset + 12 + length])[0]
            self.assertEqual(crc, zlib.crc32(kind + body) & 0xFFFFFFFF)
            chunks[kind] = body
            offset += 12 + length
        self.assertEqual(zlib.decompress(chunks[b"IDAT"]), (b"\0" + b"\1\2\3" * 3) * 2)


if __name__ == "__main__":
    unittest.main()
