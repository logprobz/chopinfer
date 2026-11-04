"""CPU checks of the KV layout served-bits equality check."""
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from qual_test_config import setUpModule  # noqa: F401
import experimental_determinism
import kv_layout_equality
import release_preview as runner
import tap_binding
from test_determinism import recorded

QUAL = Path(__file__).resolve().parent
SIDES = kv_layout_equality.SIDES


def gate():
    """The first model declaration that pins its checkpoint, exact input
    and a long_context reference of one text in the repository, a text its
    exact reference does not name: forced records of both are recorded
    against it offline."""
    import declaration
    for path in sorted(declaration.ROOT.glob("models/*/qual.json")):
        value = json.loads(path.read_text())
        spec, exact = value.get("references", {}).get("long_context"), value["inputs"].get("exact")
        if spec is None or not isinstance(spec["path"], str) or not isinstance(exact, str):
            continue
        [*texts] = json.loads((declaration.ROOT / spec["path"]).read_text())["texts"]
        names = {text["name"] for text in json.loads((declaration.ROOT / exact).read_text())["texts"]}
        if len(texts) == 1 and texts[0]["name"] not in names:
            return path
    raise AssertionError("no model declaration pins a one-text long_context reference in the repository")


# The native layout's planes the record fixtures report and declare.
RECORD_PLANES = [{"name": "compressed", "code": "fp4_e2m1", "scale": {"format": "e4m3", "block": 16},
                  "bytes_per_token": 720},
                 {"name": "index", "code": "fp4_e2m1", "scale": {"format": "ue8m0", "block": 32},
                  "bytes_per_token": 170}]


def gated(test):
    """Selects gate()'s declaration and RECORD_PLANES for `test`."""
    for selected in (patch.dict(os.environ, {"CHOPINFER_QUAL_CONFIG": str(gate())}),
                     patch.object(kv_layout_equality, "NATIVE_PLANES", RECORD_PLANES)):
        selected.start()
        test.addCleanup(selected.stop)


def server(kv_cache_dtype, package="b", nonce="d"):
    """A server identity of one build bound to the gate's checkpoint pin,
    serving `kv_cache_dtype` (with RECORD_PLANES when it is mixed)."""
    import common
    pin = common.checkpoint_pin()
    profile = {"name": "tp4", "tensor_parallel_size": 4, "kv_cache_dtype": kv_cache_dtype,
               "max_sequences": 32, "prefix_cache": True,
               "speculation": {"mode": "adaptive", "drafter": "fixture", "tokens": 5}}
    if kv_cache_dtype == "mixed":
        profile["kv_cache_planes"] = copy.deepcopy(RECORD_PLANES)
    serving = {
        "checkpoint_identity": pin["checkpoint_identity"], "package_identity": package * 64,
        "metadata_identity": "c" * 64, "startup_nonce": nonce * 32,
        "source_commit": "e" * 40, "bundle_id": "fixture.v1", "version": "1",
        "served_model_names": [pin["served_model_name"]], "profile": profile, **copy.deepcopy(OPTIONS)}
    return {"model": pin["served_model_name"], "health": {"serving": serving},
            "checkpoint": {key: serving[key] for key in (
                "checkpoint_identity", "package_identity", "metadata_identity", "startup_nonce")}}


def records(native_dtype="mixed"):
    """A native and a comparison determinism record of the same answers in
    one run."""
    native = dict(recorded(image=False, arm="kv-native"), server=server(native_dtype))
    comparison = dict(recorded(image=False, arm="kv-comparison"), server=server("bf16", package="f"))
    return native, comparison


def forced(native_dtype="mixed"):
    """A native and a comparison forced record of the same rows at every
    position the gate's pinned references define (verify_exact.py record
    --k 0 --reference long_context)."""
    digest, references, expected, _ = kv_layout_equality.forced_expectation()
    texts, prefill = [], {}
    for name, positions, kinds, scored in expected:
        rows = {str(position): {"logprob": -1.0 - position / 4096, "top": {"x": -0.5, "y": -1.0 - position / 8192},
                                "row": kind} for position, kind in zip(positions, kinds)}
        texts.append({"name": name, "category": "c", "passes": [{"phase": 0, "requests": [], "positions": rows}]})
        prefill[name] = {str(position): {"logprob": -2.0 - position / 4096, "top1": position % 7} for position in scored}
    value = {"passed": True, "arm": "kv-native-forced", "k": 0, "pooled_decode": False, "concurrency": 1,
             "top_logprobs": 5, "reference_sha256": digest, "references": references, "texts": texts,
             "prefill": prefill, "server": server(native_dtype, nonce="1")}
    return value, dict(copy.deepcopy(value), arm="kv-comparison-forced", server=server("bf16", package="f", nonce="2"))


def compared(determinism=None, forced_records=None):
    return kv_layout_equality.compare(*(determinism or records()), *(forced_records or forced()))


class CompareTests(unittest.TestCase):
    def setUp(self):
        gated(self)

    def test_identical_answers_and_forced_rows_from_two_layouts_pass(self):
        result = compared()
        self.assertTrue(result["passed"], result["errors"])
        self.assertTrue(result["equal"])
        self.assertIsNone(result["first_difference"])
        self.assertEqual(result["kv_cache_dtypes"], ["mixed", "bf16"])
        # The package may follow the KV layout; both are reported.
        self.assertEqual(result["package_identities"], ["b" * 64, "f" * 64])
        self.assertEqual((result["requests"], result["tokens"]), (11, 29))
        forced_result = result["forced"]
        self.assertTrue(forced_result["equal"])
        self.assertEqual(forced_result["kv_cache_dtypes"], ["mixed", "bf16"])
        import declaration
        pins = declaration.load(gate())["qualification"]["references"]
        self.assertEqual(forced_result["references"], {"long_context": pins["long_context"]["sha256"]})
        # The exact reference's texts and then the long-context text, every
        # position of which, after a long forced prefill, is forced: one
        # request whose first output comes from the prompt's last prefill
        # row and every later one from a decode row.
        _, _, expected, _ = kv_layout_equality.forced_expectation()
        self.assertEqual(forced_result["texts"], len(expected))
        long, positions, kinds, scored = expected[-1]
        self.assertEqual(positions, scored)
        self.assertGreater(positions[0], 8192)
        self.assertEqual(kinds, ["prefill"] + ["decode"] * (len(positions) - 1))
        self.assertEqual(forced_result["forced_references"], {"long_context": [
            {"text": long, "requests": [{"prompt_tokens": positions[0], "forced_outputs": len(positions)}],
             "prefill_rows": 1, "decode_rows": len(positions) - 1}]})
        self.assertEqual(forced_result["positions"], sum(len(positions) for _, positions, _, _ in expected))

    def test_one_ulp_logprob_difference_fails_at_its_request_and_position(self):
        native, comparison = records()
        value = comparison["requests"][2]["logprobs"][1]
        comparison["requests"][2]["logprobs"][1] = math.nextafter(value, 0.0)
        result = compared((native, comparison))
        self.assertFalse(result["passed"])
        self.assertEqual({key: result["first_difference"][key] for key in ("request", "field", "position")},
                         {"request": "fibonacci", "field": "logprobs", "position": 1})
        self.assertIn("kv-native and kv-comparison differ at fibonacci logprobs position 1", result["errors"][0])

    def test_a_token_difference_fails_at_its_first_position(self):
        native, comparison = records()
        comparison["requests"][0]["token_ids"][2] += 1
        result = compared((native, comparison))
        self.assertFalse(result["passed"])
        self.assertEqual((result["first_difference"]["field"], result["first_difference"]["position"]),
                         ("token_ids", 2))

    def test_a_one_ulp_change_in_a_forced_row_fails_at_its_text_and_position(self):
        def ulp(record, text, position, kind, field):
            if kind == "forced":
                [text_entry] = [entry for entry in record["texts"] if entry["name"] == text]
                row = text_entry["passes"][0]["positions"][str(position)]
            else:
                row = record["prefill"][text][str(position)]
            if field == "top":
                row["top"]["y"] = math.nextafter(row["top"]["y"], 0.0)
            else:
                row[field] = math.nextafter(row[field], 0.0)
        _, _, expected, _ = kv_layout_equality.forced_expectation()
        (short, _, _, scored), (long, positions, _, _) = expected[1], expected[-1]
        for text, position, kind, field in ((long, positions[len(positions) // 2], "forced", "logprob"),
                                            (long, positions[0], "forced", "top"),
                                            (short, scored[-1], "prefill", "logprob")):
            with self.subTest(text=text, kind=kind, field=field):
                native, comparison = forced()
                ulp(comparison, text, position, kind, field)
                result = compared(forced_records=(native, comparison))
                self.assertFalse(result["passed"])
                self.assertEqual({key: result["forced"]["first_difference"][key]
                                  for key in ("text", "position", "row", "field")},
                                 {"text": text, "position": position, "row": kind, "field": field})
                self.assertEqual(result["forced"]["differing_positions"], 1)
                self.assertIn(f"the forced records differ at {text} position {position} ({kind} {field}:",
                              result["errors"][-1])
        # A changed prefill top-1 token differs too.
        native, comparison = forced()
        comparison["prefill"][long][str(positions[1])]["top1"] += 1
        self.assertEqual(compared(forced_records=(native, comparison))["forced"]["first_difference"]["field"], "top1")

    def test_forced_and_prefill_rows_must_be_complete_and_typed(self):
        # Records that keep every position and binding but no numbers (the
        # same in both, so they compared equal), and rows of other types or
        # widths: each is refused before any comparison.
        def rows(record, kind):
            if kind == "forced":
                return [row for text in record["texts"] for row in text["passes"][0]["positions"].values()]
            return [row for rows in record["prefill"].values() for row in rows.values()]
        both = ("forced", "prefill")
        for name, kinds, mutate, error in (
                ("every row empty", both, lambda row: row.clear(), "holds fields [], not"),
                ("null logprobs", both, lambda row: row.update(logprob=None), "has logprob None, not a finite float"),
                ("boolean logprobs", both, lambda row: row.update(logprob=False),
                 "has logprob False, not a finite float"),
                ("integer logprobs", both, lambda row: row.update(logprob=-1), "has logprob -1, not a finite float"),
                ("NaN logprobs", ("forced",), lambda row: row.update(logprob=math.nan),
                 "has logprob nan, not a finite float"),
                ("empty top-5", ("forced",), lambda row: row.update(top={}),
                 "has a top map of 0 entries, not 1 to 6 labels of finite floats"),
                ("too wide a top-5", ("forced",), lambda row: row.update(top={str(i): -1.0 for i in range(7)}),
                 "has a top map of 7 entries"),
                ("boolean top-5 logprob", ("forced",), lambda row: row["top"].update(x=True),
                 "has a top map of 2 entries, not 1 to 6 labels of finite floats"),
                ("integer top-5 label", ("forced",), lambda row: row["top"].update({7: -1.0}),
                 "has a top map of 3 entries, not 1 to 6 labels of finite floats"),
                ("unlabelled row", ("forced",), lambda row: row.pop("row"), "holds fields ['logprob', 'top'], not"),
                ("another row kind", ("forced",), lambda row: row.update(row="verify"),
                 "is labelled 'verify', not prefill or decode"),
                ("boolean top-1", ("prefill",), lambda row: row.update(top1=True), "has top-1 True, not a token ID"),
                ("negative top-1", ("prefill",), lambda row: row.update(top1=-1), "has top-1 -1, not a token ID"),
                ("extra prefill field", ("prefill",), lambda row: row.update(top={}),
                 "holds fields ['logprob', 'top', 'top1'], not")):
            with self.subTest(name):
                native, comparison = forced()
                for record in (native, comparison):
                    for kind in kinds:
                        for row in rows(record, kind):
                            mutate(row)
                result = compared(forced_records=(native, comparison))
                self.assertFalse(result["passed"])
                self.assertTrue(any(error in message for message in result["errors"]), result["errors"])
                self.assertTrue(any(message.startswith("the comparison forced record's") for message in result["errors"]))
                self.assertNotIn("equal", result["forced"])
        for width in (4, True, None):
            with self.subTest(top_logprobs=width):
                native, comparison = forced()
                comparison["top_logprobs"] = width
                result = compared(forced_records=(native, comparison))
                self.assertIn(f"the comparison forced record declares top_logprobs {width!r}, not 5", result["errors"])

    def test_records_must_name_one_build_profile_and_options_in_one_run_with_two_layouts(self):
        native, comparison = records()
        serving = lambda value: value["server"]["health"]["serving"]  # noqa: E731
        for mutate, error in (
                (lambda value: serving(value)["profile"].update(max_sequences=16),
                 "the records differ in profile max_sequences"),
                (lambda value: serving(value).update(source_commit="9" * 40),
                 "the records differ in serving identity source_commit"),
                (lambda value: serving(value)["model_options"].update(fast_route=False),
                 "the records differ in serving identity model_options"),
                (lambda value: serving(value)["model_options"].update(fast_route=1),
                 "the records differ in serving identity model_options"),
                (lambda value: serving(value).pop("model_options"),
                 "the comparison record reports no complete model_options and model_options_qualified"),
                (lambda value: value.update(run_id="another"), "the records come from different runs")):
            changed = copy.deepcopy(comparison)
            mutate(changed)
            result = compared((native, changed))
            self.assertFalse(result["passed"])
            self.assertIn(error, result["errors"])
            # A refused pair is never compared.
            self.assertNotIn("equal", result)
        # A record without a bound serving identity is refused.
        unbound = dict(comparison, server={"model": "fixture"})
        with self.assertRaises(SystemExit):
            compared((native, unbound))

    def test_layouts_follow_the_declared_planes_and_the_bf16_arm_reports_none(self):
        native, comparison = records()
        profile = lambda value: value["server"]["health"]["serving"]["profile"]  # noqa: E731
        plane = {"name": "window", "code": "fp8_e4m3", "scale": {"format": "ue8m0", "block": 32},
                 "bytes_per_token": 33}
        for name, side, mutate, error in (
                ("missing plane", 0, lambda planes: planes.pop(), "the native profile's kv_cache_planes differ"),
                ("extra plane", 0, lambda planes: planes.append(plane), "the native profile's kv_cache_planes differ"),
                ("mismatched plane", 0, lambda planes: planes[0].update(bytes_per_token=18),
                 "the native profile's kv_cache_planes differ"),
                ("reordered planes", 0, lambda planes: planes.reverse(), "the native profile's kv_cache_planes differ"),
                ("no planes", 0, lambda planes: planes.clear(), "the native profile's kv_cache_planes differ")):
            with self.subTest(name):
                changed = [copy.deepcopy(native), copy.deepcopy(comparison)]
                mutate(profile(changed[side])["kv_cache_planes"])
                result = compared(changed)
                self.assertFalse(result["passed"])
                self.assertTrue(any(message.startswith(error) for message in result["errors"]), result["errors"])
        for name, mutate, error in (
                ("mixed without planes", lambda value: profile(value[0]).pop("kv_cache_planes"),
                 "the native profile serves KV dtype mixed without kv_cache_planes"),
                ("bf16 arm reports planes", lambda value: profile(value[1]).update(
                    kv_cache_planes=copy.deepcopy(RECORD_PLANES)), "the comparison profile reports kv_cache_planes"),
                ("both bf16", lambda value: [profile(value[0]).update(kv_cache_dtype="bf16"),
                                             profile(value[0]).pop("kv_cache_planes")], "both serve KV dtype bf16"),
                ("unknown dtype", lambda value: profile(value[1]).update(kv_cache_dtype="auto"),
                 "the comparison profile reports KV dtype 'auto', not one of bf16, fp8, mixed")):
            with self.subTest(name):
                changed = [copy.deepcopy(native), copy.deepcopy(comparison)]
                mutate(changed)
                result = compared(changed)
                self.assertFalse(result["passed"])
                self.assertIn(error, result["errors"])
        # The same rule binds the forced records.
        forced_native, forced_comparison = forced()
        profile(forced_comparison)["kv_cache_planes"] = copy.deepcopy(RECORD_PLANES)
        result = compared(forced_records=(forced_native, forced_comparison))
        self.assertIn("the comparison profile reports kv_cache_planes", result["errors"])

    def test_unbound_or_foreign_forced_records_are_refused(self):
        serving = lambda value: value["server"]["health"]["serving"]  # noqa: E731

        def bound(key, value):
            return lambda record: [where.update({key: value}) for where in (serving(record),
                                                                              record["server"]["checkpoint"])]
        another_checkpoint = bound("checkpoint_identity", "0" * 64)
        for name, mutate, error in (
                ("another checkpoint", another_checkpoint,
                 "the comparison forced record is not bound live to the gate's checkpoint pin"),
                ("rescored", lambda value: value["server"].update(rescored_from="x"), None),
                ("unqualified options", lambda value: serving(value).update(model_options_qualified=False),
                 "the comparison forced record is not bound live to the gate's checkpoint pin"),
                ("another build", lambda value: serving(value).update(source_commit="9" * 40),
                 "the comparison forced record differs from its determinism record in source_commit"),
                ("another package", bound("package_identity", "9" * 64),
                 "the comparison forced record differs from its determinism record in package_identity"),
                ("failed", lambda value: value.update(passed=False),
                 "the comparison forced record is not a passing K=0 record"),
                ("speculative", lambda value: value.update(k=1),
                 "the comparison forced record is not a passing K=0 record"),
                ("another exact reference", lambda value: value.update(reference_sha256="0" * 64),
                 "the comparison forced record scores another exact reference than the gate pins"),
                ("no long context", lambda value: value.pop("references"),
                 "the comparison forced record forces references [], not the pinned ['long_context']"),
                ("another long context", lambda value: value["references"].update(long_context="0" * 64),
                 "the comparison forced record forces references ['long_context'], not the pinned ['long_context']")):
            with self.subTest(name):
                native, comparison = forced()
                mutate(comparison)
                if error is None:
                    with self.assertRaises(SystemExit):
                        compared(forced_records=(native, comparison))
                    continue
                result = compared(forced_records=(native, comparison))
                self.assertFalse(result["passed"])
                self.assertIn(error, result["errors"])
                self.assertNotIn("equal", result["forced"])

    def test_the_long_text_is_one_request_of_one_prefill_and_the_rest_decode_rows(self):
        # A record whose long-context text scores fewer decode rows, or
        # labels them otherwise, is refused on either side, even when both
        # sides agree.
        _, _, expected, _ = kv_layout_equality.forced_expectation()
        long, positions, _, _ = expected[-1]
        decode = len(positions) - 1

        def relabel(record, position, kind):
            record["texts"][-1]["passes"][0]["positions"][str(position)]["row"] = kind
        for name, sides, change, error in (
                ("a second prefill row", ("comparison",),
                 lambda record: relabel(record, positions[len(positions) // 2], "prefill"),
                 f"the comparison forced record scores {long} with 2 prefill and {decode - 1} decode rows, not the 1 "
                 f"and {decode} its references define"),
                ("two requests on both sides", ("native", "comparison"),
                 lambda record: relabel(record, positions[512], "prefill"),
                 f"the native forced record scores {long} with 2 prefill and {decode - 1} decode rows"),
                ("no prefill row", ("native",), lambda record: relabel(record, positions[0], "decode"),
                 f"the native forced record scores {long} with 0 prefill and {decode + 1} decode rows"),
                ("prefill row moved", ("comparison",),
                 lambda record: [relabel(record, positions[0], "decode"), relabel(record, positions[3], "prefill")],
                 f"the comparison forced record labels {long} position {positions[0]} decode, not prefill")):
            with self.subTest(name):
                records = dict(zip(SIDES, forced()))
                for side in sides:
                    change(records[side])
                result = compared(forced_records=(records["native"], records["comparison"]))
                self.assertFalse(result["passed"])
                self.assertTrue(any(message.startswith(error) for message in result["errors"]), result["errors"])
                self.assertNotIn("equal", result["forced"])

    def test_forced_position_sets_must_be_equal_and_the_ones_the_references_define(self):
        def drop(record, text, position, kind="forced"):
            if kind == "forced":
                [entry] = [entry for entry in record["texts"] if entry["name"] == text]
                del entry["passes"][0]["positions"][str(position)]
            else:
                del record["prefill"][text][str(position)]
        _, _, expected, _ = kv_layout_equality.forced_expectation()
        (first, *_), (short, _, _, scored), (long, positions, _, _) = expected[0], expected[1], expected[-1]
        native, comparison = forced()
        drop(comparison, long, positions[-1])
        result = compared(forced_records=(native, comparison))
        self.assertFalse(result["passed"])
        self.assertIn(f"the forced records hold other forced positions of {long}: the native record alone holds "
                      f"[{positions[-1]}], the comparison record []", result["errors"])
        native, comparison = forced()
        for record in (native, comparison):
            drop(record, long, positions[-1])
            drop(record, short, scored[0], "prefill")
        result = compared(forced_records=(native, comparison))
        self.assertIn(f"the forced records hold other forced positions of {long} than its references define: "
                      f"missing [{positions[-1]}], extra []", result["errors"])
        self.assertIn(f"the forced records hold other prefill positions of {short} than its references define: "
                      f"missing [{scored[0]}], extra []", result["errors"])
        # Without the long-context text (a record of the exact texts alone,
        # or one whose server could not hold the long history) nothing passes.
        native, comparison = forced()
        for record in (native, comparison):
            record["texts"].pop()
            record["prefill"].pop(long)
        result = compared(forced_records=(native, comparison))
        self.assertTrue(any(message.startswith(f"the forced records hold other forced positions of {long} than its "
                                               f"references define: missing {positions[:3]}")
                            for message in result["errors"]), result["errors"])
        native, comparison = forced()
        comparison["texts"].append(copy.deepcopy(comparison["texts"][0]))
        result = compared(forced_records=(native, comparison))
        self.assertIn(f"the comparison forced record does not hold one phase-0 pass of text {first}", result["errors"])


# Synthetic tap captures of two KV layouts in the representation a
# diagnostic-taps server writes (models' taps.json, "KV-state taps"): KV
# steps (capture.json program KvState) of each readback and row axis, whose
# items.json `kv` names the phase, axis, ratio, storage, state end and each
# row's position, holding every rank's copy of each plane. The native layout
# stores codes and block scales and writes their logical BF16 products and
# its raw stored bytes (`.codes`, `.scales`, u32 words of the stored bytes in
# memory order); the BF16 layout stores and writes the products alone. Layer
# 0 has a window only; layer 2 also has compressed latents and index keys at
# compress ratio 2. Widths are small.
RANKS = 2
WINDOW = 4
RATIO = 2
# Each plane's code, scale format, scale block and width, as kernel_core's
# trained KV quantizers store them: E4M3 codes with a UE8M0 scale per 32,
# E2M1 codes with a UE4M3 (unsigned E4M3) scale per 16, and E2M1 codes with
# a UE8M0 scale per 32.
PLANES = {"window": ("e4m3", "ue8m0", 32, 128), "compressed": ("e2m1", "ue4m3", 16, 64),
          "index": ("e2m1", "ue8m0", 32, 128)}
LAYERS = {0: ("window",), 2: ("window", "compressed", "index")}
TEXT_TAPS = (("L00.attn.out", "bf16", (0,)), ("L00.x.attn.partial", "f32", (0, 1)))
TEXT_COLS = 8
# A primer prefilled in two chunks and decoded twice, and its continuation,
# a prefix hit restoring 4 tokens, prefilling 5 and decoding once: (name,
# prompt tokens, cached tokens, primer, prefill chunks, decode steps).
REQUESTS = (("primer", 6, 0, None, ((0, 4), (4, 6)), 2), ("hit", 9, 4, "primer", ((4, 9),), 1))
# A longer pair, whose restore readback keeps rows from token 4 on.
LONGER = (("primer", 8, 0, None, ((0, 4), (4, 8)), 2), ("hit", 11, 6, "primer", ((6, 11),), 1))
NATIVE_PLANES = [{"name": "compressed", "code": "fp4_e2m1", "scale": {"format": "e4m3", "block": 16},
                  "bytes_per_token": 18},
                 {"name": "index", "code": "fp4_e2m1", "scale": {"format": "ue8m0", "block": 32},
                  "bytes_per_token": 34}]
OPTIONS = {"model_options": {"prefill_chunk_tokens": 4, "fast_route": True}, "model_options_qualified": True}


def e4m3(byte):
    sign = -1.0 if byte & 0x80 else 1.0
    exponent, mantissa = (byte >> 3) & 0xF, byte & 7
    return sign * (mantissa / 8 * 2.0 ** -6 if exponent == 0 else (1 + mantissa / 8) * 2.0 ** (exponent - 7))


def e2m1(nibble):
    sign = -1.0 if nibble & 8 else 1.0
    exponent, mantissa = (nibble >> 1) & 3, nibble & 1
    return sign * (mantissa / 2 if exponent == 0 else (1 + mantissa / 2) * 2.0 ** (exponent - 1))


def bf16(value):
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    assert bits & 0xFFFF == 0, value
    return struct.pack("<H", bits >> 16)


def stored(layer, plane, position, rank):
    """A native row's stored codes and scales, distinct per layer, plane,
    position and rank: E4M3 codes without NaNs, E2M1 codes two per byte,
    the even channel in the low nibble, UE8M0 scales 2^-4 .. 2^3 and UE4M3
    scales 0.5 .. 1.875."""
    code, scale, block, width = PLANES[plane]
    draw = random.Random(f"{layer}/{plane}/{position}/{rank}").randrange
    codes = bytes(draw(256) & 0xBF if code == "e4m3" else draw(256)
                  for _ in range(width if code == "e4m3" else width // 2))
    scales = bytes(123 + draw(8) if scale == "ue8m0" else 0x30 + draw(16) for _ in range(width // block))
    return codes, scales


def logical(plane, codes, scales):
    """The BF16 products of a native row's codes and scales."""
    code, scale, block, width = PLANES[plane]
    values = []
    for channel in range(width):
        value = e4m3(codes[channel]) if code == "e4m3" else e2m1(codes[channel // 2] >> 4 * (channel % 2) & 0xF)
        factor = 2.0 ** (scales[channel // block] - 127) if scale == "ue8m0" else e4m3(scales[channel // block])
        values.append(bf16(value * factor))
    return b"".join(values)


def readbacks(requests=REQUESTS):
    """(request, phase, state end) of every readback, in capture order."""
    found = []
    for name, prompt, cached, _, chunks, decodes in requests:
        found += [(name, "restore", cached)] if cached else []
        found += [(name, "prefill", end) for _, end in chunks]
        found += [(name, "decode", prompt + step + 1) for step in range(decodes)]
    return found


def axis_rows(plane, end):
    """The positions a readback ending at `end` holds of a plane: token
    positions in the window, compressed positions otherwise."""
    return list(range(max(0, end - WINDOW), end)) if plane == "window" else list(range(end // RATIO))


def capture_server(layout, nonce):
    """The serving identity a capture of `layout` records."""
    profile = {"name": "tp2", "tensor_parallel_size": RANKS,
               "kv_cache_dtype": "mixed" if layout == "native" else "bf16", "max_sequences": 4, "prefix_cache": True,
               "speculation": {"mode": "off", "drafter": None, "tokens": 0}}
    if layout == "native":
        profile["kv_cache_planes"] = copy.deepcopy(NATIVE_PLANES)
    return {"bundle_id": "fixture.v1", "package_identity": ("b" if layout == "native" else "f") * 64,
            "checkpoint_identity": "a" * 64, "metadata_identity": "c" * 64, "startup_nonce": nonce * 32,
            "profile": profile, "version": "1", "source_commit": "e" * 40, **copy.deepcopy(OPTIONS)}


def kept_rows(tokens, taps_from):
    """The rows a step keeps of rows whose first tokens are `tokens`
    (CHOPINFER_TAPS_FROM, runtime_bundle_sdk Taps::rows): the span from the
    first to the last at or past `taps_from`, or none."""
    kept = [row for row, token in enumerate(tokens) if taps_from is None or token >= taps_from]
    return (kept[0], kept[-1] + 1) if kept else (0, 0)


def write_step(directory, nonce, items, declared, tensors, kept):
    """A tap step of `items` declaring `declared` ((name, rank, cols, dtype)
    each, in the order the step writes them) that keeps rows `kept` of each:
    `tensors` maps (name, rank) to the kept rows' bytes. A step that keeps no
    row writes no tensor, as the producer leaves one."""
    directory.mkdir(parents=True)
    first, end = kept
    expected = [{"name": name, "rank": rank, "first_row": first, "rows": end - first, "cols": cols, "dtype": dtype}
                for name, rank, cols, dtype in declared]
    index = expected if end > first else []
    for entry in index:
        (directory / f"{entry['name']}.r{entry['rank']}.bin").write_bytes(tensors[entry["name"], entry["rank"]])
    capture = {"schema": "chopinfer.taps.v1", "model": "fixture", "ranks": RANKS,
               "capture_id": f"{nonce}-{directory.name}", "program": items["program"], "rows": items["rows"],
               "first_row": first, "captured_rows": end - first, "comparison_scope": None, "expected": expected,
               "tensor_bytes": sum(len(tensors[entry["name"], entry["rank"]]) for entry in index),
               "producer": {"startup_nonce": nonce}}
    completion = ({"completion": "full", "expected": len(expected), "observed": len(index), "reason": None} if index
                  else {"completion": "incomplete", "expected": len(expected), "observed": 0,
                        "reason": "no selected live tensors"})
    for name, value in (("capture.json", capture), ("items.json", dict(items, dump_rows=[first, end])),
                        ("completion.json", completion)):
        (directory / name).write_text(json.dumps(value))
    (directory / "index.jsonl").write_text("".join(json.dumps(line) + "\n" for line in index))


def write_capture(root, layout, corrupt=None, omit=(), first_request=1, experimental=False, requests=REQUESTS,
                  taps_from=None):
    """A tools/qual/tap_binding.py capture of `requests` from a server of
    `layout` (native or bf16). `corrupt` flips the lowest bit of the first
    stored byte of one row, ("kv", layer, plane, position, rank) or ("text",
    tap, position, rank), in every readback or step that holds it; `omit`
    leaves out the (tap, rank) entries named; `taps_from` keeps the rows
    whose first token is at or past it. An `experimental` capture records a
    sealed pin's diagnostic options."""
    nonce = "1" if layout == "native" else "2"
    server = capture_server(layout, nonce)
    root.mkdir(parents=True)
    steps, records, count = root / tap_binding.STEPS, [], 0

    def flipped(key, data):
        return bytes([data[0] ^ 1]) + data[1:] if key == corrupt else data

    def step(items, declared, tensors, kept):
        nonlocal count
        name = f"step{count:04d}"
        count += 1
        write_step(steps / name, server["startup_nonce"], dict(items, dump_from=taps_from),
                   [entry for entry in declared if entry[:2] not in omit], tensors, kept)
        return name

    def kv_steps(engine, slot, phase, end):
        names = []
        for axis, ratio, planes in (("window", 1, [(layer, "window") for layer in LAYERS]),
                                    ("compressed", RATIO, [(layer, plane) for layer, held in LAYERS.items()
                                                           for plane in held if plane != "window"])):
            positions = axis_rows(axis, end)
            first, last = kept_rows([position * ratio for position in positions], taps_from)
            if first == last:
                continue
            declared, tensors = [], {}
            for layer, plane in planes:
                rows = {}
                for rank in range(RANKS):
                    for position in positions[first:last]:
                        key = ("kv", layer, plane, position, rank)
                        codes, scales = stored(layer, plane, position, rank)
                        rows.setdefault(rank, []).append(
                            (flipped(key, codes), scales, logical(plane, flipped(key, codes), scales))
                            if layout == "native" else (None, None, flipped(key, logical(plane, codes, scales))))
                for part, at in (("codes", 0), ("scales", 1), (None, 2)) if layout == "native" else ((None, 2),):
                    tap = f"L{layer:02d}.x.kv.{plane}" + (f".{part}" if part else "")
                    for rank in range(RANKS):
                        tensors[tap, rank] = b"".join(row[at] for row in rows[rank])
                        declared.append((tap, rank, len(rows[rank][0][at]) // (4 if part else 2),
                                         "u32" if part else "bf16"))
            items = {"program": "KvState", "rows": len(positions),
                     "kv": {"phase": phase, "axis": axis, "ratio": ratio, "storage": layout, "end": end,
                            "positions": positions},
                     "items": [{"request": engine, "slot": slot, "start": 0, "rows": len(positions)}]}
            names.append(step(items, declared, tensors, (first, last)))
        return names

    def text_step(engine, slot, tokens, first, end):
        kept = kept_rows(range(first, end), taps_from)
        declared, tensors = [], {}
        for tap, dtype, ranks in TEXT_TAPS:
            for rank in ranks:
                rows = []
                for position in range(first + kept[0], first + kept[1]):
                    draw = random.Random(f"{tap}/{position}/{rank}").random
                    values = [bf16(round(draw() * 64 - 32) / 8) if dtype == "bf16" else struct.pack("<f", draw() - 0.5)
                              for _ in range(TEXT_COLS)]
                    rows.append(flipped(("text", tap, position, rank), b"".join(values)))
                tensors[tap, rank] = b"".join(rows)
                declared.append((tap, rank, TEXT_COLS, dtype))
        items = {"program": "Prefill", "rows": end - first,
                 "items": [{"request": engine, "slot": slot, "start": 0, "rows": end - first, "prefix": first,
                            "seq_len": end, "tokens": tokens[first:end], "positions": list(range(first, end))}]}
        return step(items, declared, tensors, kept)

    for at, (name, prompt, cached, primer, chunks, decodes) in enumerate(requests):
        engine, slot, tokens = first_request + at, 3 * at + (layout != "native"), list(range(100, 100 + prompt))
        named = kv_steps(engine, slot, "restore", cached) if cached else []
        for first, end in chunks:
            named += [text_step(engine, slot, tokens, first, end), *kv_steps(engine, slot, "prefill", end)]
        for decode in range(decodes):
            named += kv_steps(engine, slot, "decode", prompt + decode + 1)
        records.append({"name": name, "path": "/v1/completions",
                        "body_sha256": hashlib.sha256(json.dumps({"prompt": tokens}).encode()).hexdigest(),
                        "challenge": f"tap-capture-{name}-{layout}", "engine_request_id": engine,
                        "response_id": f"cmpl_{engine}", "prompt_tokens": prompt, "cached_tokens": cached,
                        "cache_salt": f"tap-capture/{primer or name}-{layout}", "primer": primer, "steps": named})
    manifest = {"schema": tap_binding.SCHEMA, "server": server, "server_log_sha256": "0" * 64,
                "url": "http://127.0.0.1:39001", "requests": records, "files": {}, "subject": None}
    if experimental:
        options = dict(OPTIONS["model_options"], fast_route=False)
        server.update(model_options=options, model_options_qualified=False)
        pin = {key: server[key] for key in ("model_options", "checkpoint_identity", "startup_nonce")}
        manifest |= {experimental_determinism.MARKER: sealed(pin), "qualified": False}
    (root / tap_binding.MANIFEST).write_text(json.dumps(manifest))
    return reseal(root)


def reseal(root):
    """Records the digests of the steps in `root` in its manifest, as the
    capture that wrote them would."""
    edit_manifest(root, lambda manifest: manifest.update(files={
        step.name: tap_binding.step_files(step) for step in sorted((root / tap_binding.STEPS).glob("step*"))}))
    return root


def sealed(pin):
    """The marker experimental_determinism.Pin.marker records for `pin`."""
    return {"pin": pin, "pin_sha256": hashlib.sha256(experimental_determinism.canonical(pin)).hexdigest(),
            "file_sha256": hashlib.sha256(json.dumps(pin, indent=2).encode()).hexdigest(), "qualified": False}


def edit_manifest(root, change):
    """Applies `change` to the capture manifest in `root`."""
    path = root / tap_binding.MANIFEST
    manifest = json.loads(path.read_text())
    change(manifest)
    path.write_text(json.dumps(manifest))


class TapsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR"))
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def capture(self, layout, name=None, **change):
        """A capture of `layout`: the clean one, written once, unless named."""
        root = self.root / (name or layout)
        return root if name is None and root.exists() else write_capture(root, layout, **change)

    def compare(self, native=None, comparison=None, planes=NATIVE_PLANES, experimental=False):
        return kv_layout_equality.compare_taps(native or self.capture("native"), comparison or self.capture("bf16"),
                                               planes, experimental)

    def keys(self, result):
        """The differing rows' keys: readback, layer, plane, position and rank."""
        return {(d["request"], d["phase"], d["end"], d["layer"], d["plane"], d["position"], d["rank"])
                for d in result["differences"]}

    def test_two_layouts_holding_the_same_logical_values_compare_equal_through_every_phase(self):
        result = self.compare()
        self.assertTrue(result["passed"], result["errors"])
        self.assertTrue(result["equal"])
        self.assertIsNone(result["first_difference"])
        self.assertEqual(result["kv_cache_dtypes"], ["mixed", "bf16"])
        self.assertEqual(result["package_identities"], ["b" * 64, "f" * 64])
        self.assertEqual(result["kv_readbacks"], {"prefill": 3, "decode": 3, "restore": 1})
        # Every rank's window, compressed and index rows of every readback.
        kv_rows = sum(RANKS * len(axis_rows(plane, end)) for _, _, end in readbacks()
                      for held in LAYERS.values() for plane in held)
        # Every native row's codes and scales decode to its logical values.
        self.assertEqual(result["compared"], {"requests": 2, "steps": 17, "kv_rows": kv_rows, "text_rows": 3 * 11,
                                              "native_rows_decoded": kv_rows})
        # Only the native layout writes its raw stored bytes, which are its own.
        self.assertEqual(result["raw_rows"], {"native": 2 * kv_rows, "comparison": 0})
        self.assertNotIn("qualified", result)

    def test_one_corrupted_value_of_each_plane_and_layout_fails_at_its_readbacks_layer_plane_position_and_rank(self):
        native, comparison = self.capture("native"), self.capture("bf16")
        for layout in ("native", "bf16"):
            for plane, position in (("window", 3), ("compressed", 1), ("index", 1)):
                key = ("kv", 2, plane, position, 1)
                with self.subTest(layout=layout, plane=plane):
                    corrupted = self.capture(layout, f"{layout}-{plane}", corrupt=key)
                    result = self.compare(*((corrupted, comparison) if layout == "native" else (native, corrupted)))
                    expected = {(request, phase, end, 2, plane, position, 1) for request, phase, end in readbacks()
                                if position in axis_rows(plane, end)}
                    # In the window, the restore, a prefill and a decode
                    # readback; compressed, all seven.
                    self.assertEqual({key[1] for key in expected}, {"restore", "prefill", "decode"})
                    self.assertFalse(result["passed"])
                    self.assertEqual(self.keys(result), expected)
                    self.assertEqual(result["differing_rows"], len(expected))
                    self.assertEqual(result["differing_taps"], [[f"L02.x.kv.{plane}", len(expected)]])
                    first = result["first_difference"]
                    self.assertEqual((first["request"], first["tap"], first["difference"]),
                                     ("primer", f"L02.x.kv.{plane}", "differs"))
                    self.assertIn(f"differ at L02.x.kv.{plane} rank 1", result["errors"][-1])

    def test_a_mutated_text_value_fails_at_its_tap_position_and_rank(self):
        # Position 2 is the primer's alone, 7 the continuation's.
        for tap, position, rank in (("L00.attn.out", 2, 0), ("L00.x.attn.partial", 7, 1)):
            with self.subTest(tap=tap):
                result = self.compare(comparison=self.capture("bf16", tap, corrupt=("text", tap, position, rank)))
                self.assertFalse(result["passed"])
                self.assertEqual(result["differing_taps"], [[tap, 1]])
                first = result["first_difference"]
                self.assertEqual({key: first[key] for key in ("request", "tap", "position", "rank", "difference")},
                                 {"request": "primer" if position < 4 else "hit", "tap": tap, "position": position,
                                  "rank": rank, "difference": "differs"})

    def test_a_missing_tap_fails_naming_it(self):
        for layout, omitted, rows in (("bf16", ("L02.x.kv.index", 1), sum(
                len(axis_rows("index", end)) for _, _, end in readbacks())), ("native", ("L00.x.attn.partial", 1), 11)):
            with self.subTest(omitted=omitted):
                lacking = self.capture(layout, f"lacking-{omitted[0]}", omit=(omitted,))
                result = self.compare(*((lacking, None) if layout == "native" else (None, lacking)))
                self.assertFalse(result["passed"])
                self.assertEqual(result["differing_taps"], [[omitted[0], rows]])
                first = result["first_difference"]
                lacked = "native" if layout == "native" else "comparison"
                self.assertEqual((first["tap"], first["rank"], first["difference"]),
                                 (*omitted, f"missing from the {lacked} capture"))

    def test_native_codes_and_scales_must_decode_exactly_to_its_logical_values(self):
        codes, scales = "L00.x.kv.window.codes", "L00.x.kv.window.scales"

        def rewrite(tap, value=None):
            """Sets (or flips the lowest bit of) the first stored byte of
            `tap`'s rank 0 in the primer's first window readback."""
            def change(step):
                path = step / f"{tap}.r0.bin"
                data = bytearray(path.read_bytes())
                data[0] = data[0] ^ 1 if value is None else value
                path.write_bytes(bytes(data))
            return change
        where = r"L00\.x\.kv\.window rank 0 \(prefill readback ending at 4, request primer position 0\)"
        for name, changes, error in (
                ("changed code", [rewrite(codes)],
                 f"the native capture's {where} holds values its codes and scales do not decode to"),
                # E4M3 448 times 2^127 overflows BF16.
                ("overflow", [rewrite(codes, 0x7E), rewrite(scales, 254)],
                 rf"insufficient evidence: the native capture's {where} decodes channel 0 to 7\.6\d*e\+40, not an "
                 "exact finite BF16"),
                # E4M3 2^-9 times 2^-127 lies between BF16 subnormals.
                ("subnormal", [rewrite(codes, 0x01), rewrite(scales, 0)],
                 rf"insufficient evidence: the native capture's {where} decodes channel 0 to 1\.14\d*e-41, not an "
                 "exact finite BF16")):
            with self.subTest(name):
                native = self.capture("native", f"native-{name}")
                for change in changes:
                    change(native / tap_binding.STEPS / "step0001")
                result = self.compare(native=reseal(native))
                self.assertFalse(result["passed"])
                self.assertTrue(any(re.fullmatch(error, message) for message in result["errors"]), result["errors"])
                # Refused, not a difference between the layouts.
                self.assertTrue(result["equal"])
        # The native capture must hold both raw taps of every plane it
        # writes, and the comparison capture none.
        raw = [(f"L{layer:02d}.x.kv.{plane}.{part}", rank) for layer, held in LAYERS.items() for plane in held
               for part in ("codes", "scales") for rank in range(RANKS)]
        for name, omitted in (("bare", raw), ("no scales", [tap for tap in raw if tap[0].endswith(".scales")])):
            with self.subTest(name), self.assertRaisesRegex(
                    tap_binding.Unbound, r"step0001: the native capture holds L00\.x\.kv\.window rank 0 without "
                                         "its stored codes and scales"):
                self.compare(native=self.capture("native", name, omit=omitted))
        relabeled = self.capture("native", "relabeled")
        edit_manifest(relabeled, lambda m: [m["server"]["profile"].update(kv_cache_dtype="bf16"),
                                            m["server"]["profile"].pop("kv_cache_planes")])
        with self.assertRaisesRegex(tap_binding.Unbound, r"step0001: the comparison capture holds raw stored tap "
                                                         r"L00\.x\.kv\.window\.codes rank 0"):
            self.compare(comparison=relabeled)

    def test_engine_request_ids_must_be_integers(self):
        # true equals 1 in Python; it names no engine request.
        def boolean(root, manifest_too):
            manifest = json.loads((root / tap_binding.MANIFEST).read_text())
            if manifest_too:
                edit_manifest(root, lambda m: m["requests"][0].update(engine_request_id=True))
            for name in manifest["requests"][0]["steps"]:
                items = root / tap_binding.STEPS / name / "items.json"
                value = json.loads(items.read_text())
                for entry in value["items"]:
                    entry["request"] = True
                items.write_text(json.dumps(value))
            return reseal(root)
        for name, manifest_too in (("boolean items", False), ("boolean manifest and items", True)):
            with self.subTest(name), self.assertRaisesRegex(tap_binding.Unbound,
                                                            "names engine request True, not an integer ID"):
                self.compare(comparison=boolean(self.capture("bf16", name), manifest_too))

    def test_index_entries_must_match_their_declared_descriptors(self):
        # The comparison's first window readback declares its first tap with
        # another descriptor field than its index holds, or none.
        for field, value in (("dtype", "u8"), ("cols", 999), ("first_row", 1), ("rows", 999), ("rows", 4.0),
                             ("cols", None)):
            with self.subTest(field=field, value=value):
                root = self.capture("bf16", f"descriptor-{field}-{value}")
                path = root / tap_binding.STEPS / "step0001" / "capture.json"
                capture = json.loads(path.read_text())
                declared = capture["expected"][0]
                if value is None:
                    declared.pop(field)
                else:
                    declared[field] = value
                path.write_text(json.dumps(capture))
                with self.assertRaisesRegex(tap_binding.Unbound,
                                            rf"step0001: {re.escape(declared['name'])} rank 0 has {field} "
                                            rf"\S+; capture\.json declares {re.escape(repr(value))}$"):
                    self.compare(comparison=reseal(root))

    def test_logical_kv_values_must_be_bf16_rows(self):
        # Both captures declare L00.x.kv.window rank 0 of the primer's first
        # window readback F32, each row its BF16 bytes followed by others.
        roots = [self.capture(layout, f"{layout}-f32") for layout in ("native", "bf16")]
        for root in roots:
            step = root / tap_binding.STEPS / "step0001"
            tap = step / "L00.x.kv.window.r0.bin"
            data, row = tap.read_bytes(), PLANES["window"][3] * 2
            tap.write_bytes(b"".join(data[at:at + row] + struct.pack("<f", 1.0) * (row // 4)
                                     for at in range(0, len(data), row)))
            for name in ("index.jsonl", "capture.json"):
                text = (step / name).read_text()
                entry = '"name": "L00.x.kv.window", "rank": 0, '
                at = text.index(entry)
                self.assertIn('"dtype": "bf16"', text[at:at + 120])
                (step / name).write_text(text[:at] + text[at:at + 120].replace('"bf16"', '"f32"') + text[at + 120:])
            reseal(root)
        with self.assertRaisesRegex(tap_binding.Unbound, r"step0001: L00\.x\.kv\.window rank 0 has dtype f32; a "
                                                         "KV-state tap's logical values are bf16"):
            self.compare(*roots)

    def test_rows_kept_from_a_position_compare_at_their_positions_and_steps_that_kept_none_pair(self):
        # From token 4 on: the primer's first chunk keeps no row and writes
        # no tensor, and its first readback no KV step.
        native, comparison = (self.capture(layout, f"{layout}-from", taps_from=4, requests=LONGER)
                              for layout in ("native", "bf16"))
        empty = json.loads((native / tap_binding.STEPS / "step0000" / "completion.json").read_text())
        self.assertEqual((empty["completion"], empty["reason"]), ("incomplete", "no selected live tensors"))
        result = self.compare(native, comparison)
        self.assertTrue(result["passed"], result["errors"])
        self.assertEqual(result["kv_readbacks"], {"prefill": 2, "decode": 3, "restore": 1})
        # A compressed position kept from its readbacks' fourth row on.
        corrupted = self.capture("bf16", "corrupted-from", taps_from=4, requests=LONGER,
                                 corrupt=("kv", 2, "compressed", 3, 0))
        expected = {(request, phase, end, 2, "compressed", 3, 0) for request, phase, end in readbacks(LONGER)
                    if 3 < end // RATIO}
        self.assertEqual(self.keys(self.compare(native, corrupted)), expected)
        # A step that kept a row but wrote no tensor is incomplete.
        broken = self.capture("bf16", "broken-from", taps_from=4, requests=LONGER)
        step = broken / tap_binding.STEPS / "step0001"
        for tap in step.glob("*.bin"):
            tap.unlink()
        (step / "index.jsonl").write_text("")
        (step / "completion.json").write_text(json.dumps({"completion": "incomplete", "expected": 3, "observed": 0,
                                                          "reason": "no selected live tensors"}))
        with self.assertRaisesRegex(tap_binding.Unbound, "step0001 is incomplete: no selected live tensors"):
            self.compare(native, reseal(broken))

    def test_a_comparison_without_rows_of_each_plane_and_phase_is_insufficient(self):
        nothing = ["no text tap rows were compared", "no KV-state rows were compared",
                   *(f"no KV-state rows of plane {plane} were compared" for plane in kv_layout_equality.KV_PLANES),
                   *(f"no KV-state rows of phase {phase} were compared" for phase in ("prefill", "decode"))]
        for name, change, errors in (
                ("no requests", {"requests": ()}, nothing),
                ("filtered", {"taps_from": 100}, [*nothing, "no KV-state rows of phase restore were compared"]),
                ("no index keys", {"omit": [(f"L02.x.kv.index{part}", rank) for part in ("", ".codes", ".scales")
                                            for rank in range(RANKS)]},
                 ["no KV-state rows of plane index were compared"]),
                # The default pair's restore holds tokens 0 .. 3 alone.
                ("no restore", {"taps_from": 5}, ["no KV-state rows of phase restore were compared"]),
                ("no decode", {"requests": tuple(request[:-1] + (0,) for request in REQUESTS)},
                 ["no KV-state rows of phase decode were compared"])):
            with self.subTest(name):
                native, comparison = (self.capture(layout, f"{layout}-{name}", **change) for layout in ("native", "bf16"))
                result = self.compare(native, comparison)
                self.assertFalse(result["passed"])
                self.assertEqual(result["errors"], [f"insufficient evidence: {error}" for error in errors])
        # Without a prefix-hit pair, no restore readback is owed.
        alone = {"requests": REQUESTS[:1]}
        result = self.compare(*(self.capture(layout, f"{layout}-alone", **alone) for layout in ("native", "bf16")))
        self.assertTrue(result["passed"], result["errors"])

    def test_rows_no_position_or_item_names_are_refused_not_dropped(self):
        def moved(name, step, change, corrupt=False):
            """Captures of both layouts with `change` applied to `step`'s
            files, and one value of the comparison's changed."""
            roots = [self.capture(layout, f"{layout}-{name}") for layout in ("native", "bf16")]
            for root in roots:
                change(root / tap_binding.STEPS / step)
            if corrupt:
                tap = roots[1] / tap_binding.STEPS / step / "L00.x.kv.window.r0.bin"
                tap.write_bytes(bytes([tap.read_bytes()[0] ^ 1]) + tap.read_bytes()[1:])
            return [reseal(root) for root in roots]

        def past_positions(directory):
            # The primer's first window readback holds positions 0 .. 3.
            for name in ("index.jsonl", "capture.json"):
                text = (directory / name).read_text()
                entry = '"name": "L00.x.kv.window", "rank": 0, "first_row": '
                self.assertIn(entry + "0", text)
                (directory / name).write_text(text.replace(entry + "0", entry + "4"))

        def unowned(directory):
            # The primer's first text step: its item no longer owns row 3.
            items = json.loads((directory / "items.json").read_text())
            items["items"][0]["positions"] = items["items"][0]["positions"][:3]
            (directory / "items.json").write_text(json.dumps(items))
        for name, step, change, corrupt, error in (
                ("past", "step0001", past_positions, True,
                 r"L00\.x\.kv\.window rank 0 holds rows 4 \.\. 7, past its rows 0 \.\. 3"),
                ("unowned", "step0000", unowned, False, r"L00\.attn\.out rank 0 holds row 3, which no item owns")):
            with self.subTest(name), self.assertRaisesRegex(tap_binding.Unbound, error):
                self.compare(*moved(name, step, change, corrupt))

    def test_captures_that_differ_in_identity_options_or_layout_are_refused(self):
        for change, error in (
                (lambda m: m["server"].update(source_commit="9" * 40),
                 "the captures differ in serving identity source_commit"),
                (lambda m: m["server"]["profile"].update(max_sequences=8),
                 "the captures differ in profile max_sequences"),
                (lambda m: m["server"]["model_options"].update(fast_route=False),
                 "the captures differ in model_options"),
                (lambda m: m["server"]["profile"].update(kv_cache_planes=NATIVE_PLANES),
                 "the comparison profile reports kv_cache_planes"),
                (lambda m: m["server"]["profile"].update(kv_cache_dtype="mixed"),
                 "the comparison profile serves KV dtype mixed without kv_cache_planes")):
            with self.subTest(error=error):
                comparison = self.capture("bf16", error)
                edit_manifest(comparison, change)
                result = self.compare(comparison=comparison)
                self.assertFalse(result["passed"])
                self.assertIn(error, result["errors"])
                self.assertNotIn("equal", result)
        for planes, error in ((NATIVE_PLANES[:1], "the native profile's kv_cache_planes differ from NATIVE_PLANES"),
                              (None, "no native planes are declared (NATIVE_PLANES)")):
            with self.subTest(error=error):
                self.assertIn(error, self.compare(planes=planes)["errors"])
        same = self.capture("native", "same")
        self.assertIn("both serve KV dtype mixed",
                      kv_layout_equality.compare_taps(self.capture("native"), same, NATIVE_PLANES)["errors"])
        # The package identity follows the layout and may differ: the clean pair passes.

    def test_provenance_fields_must_be_typed_valid_and_present(self):
        def both(name, change):
            roots = [self.capture(layout, f"{layout}-{name}") for layout in ("native", "bf16")]
            for root in roots:
                edit_manifest(root, lambda m: change(m, root.name.startswith("native")))
            return self.compare(*roots)

        def records(field, value=None):
            return lambda m, _: [record.pop(field) if value is None else record.update({field: value})
                                 for record in m["requests"]]
        for name, change, error in (
                ("no comparison dtype", lambda m, native: native or m["server"]["profile"].pop("kv_cache_dtype"),
                 "the comparison profile reports KV dtype None, not one of bf16, fp8, mixed"),
                ("auto dtype", lambda m, native: native or m["server"]["profile"].update(kv_cache_dtype="auto"),
                 "the comparison profile reports KV dtype 'auto', not one of bf16, fp8, mixed"),
                ("typed options", lambda m, native: native or m["server"]["model_options"].update(fast_route=1),
                 "the captures differ in model_options"),
                ("typed profile", lambda m, native: native or m["server"]["profile"].update(prefix_cache=1),
                 "the captures differ in profile prefix_cache"),
                ("no path", records("path"), "the native capture's request primer records no valid path"),
                ("no prompt tokens", records("prompt_tokens"),
                 "the native capture's request primer records no valid prompt_tokens"),
                ("no cached tokens", records("cached_tokens"),
                 "the native capture's request primer records no valid cached_tokens"),
                ("boolean prompt tokens", records("prompt_tokens", True),
                 "the native capture's request primer records no valid prompt_tokens"),
                ("upper-case digest", records("body_sha256", "A" * 64),
                 "the native capture's request primer records no valid body_sha256"),
                ("short digest", records("body_sha256", "0" * 63),
                 "the native capture's request primer records no valid body_sha256")):
            with self.subTest(name):
                result = both(name, change)
                self.assertFalse(result["passed"])
                self.assertIn(error, result["errors"])
                self.assertNotIn("equal", result)
        # Every recorded identity and option field is required, even when
        # both captures lack it, before anything is compared.
        for name, change, error in (
                *((f"no {key}", lambda m, _, key=key: m["server"].pop(key), f"the server reports no {key}")
                  for key in ("bundle_id", "version", "source_commit")),
                ("blank version", lambda m, _: m["server"].update(version=" "), "the server reports no version"),
                ("no options", lambda m, _: [m["server"].pop(key) for key in tap_binding.OPTIONS],
                 "the capture records no complete model_options and model_options_qualified"),
                ("empty options", lambda m, _: m["server"].update(model_options={}),
                 "the capture records no complete model_options and model_options_qualified")):
            with self.subTest(name), self.assertRaisesRegex(tap_binding.Unbound, error):
                both(name, change)
        # Paired steps' metadata is compared with its types too.
        roots = [self.capture(layout, f"{layout}-typed-step") for layout in ("native", "bf16")]
        items = roots[1] / tap_binding.STEPS / "step0002" / "items.json"
        items.write_text(items.read_text().replace('"ratio": 2', '"ratio": 2.0'))
        self.assertIn("request primer step 2 (step0002 and step0002) differs in items.kv.ratio",
                      self.compare(roots[0], reseal(roots[1]))["errors"])

    def test_captures_that_differ_in_prompts_or_steps_or_break_the_cache_rules_are_refused(self):
        def hit(change):
            return lambda m: change(next(record for record in m["requests"] if record["name"] == "hit"))
        for change, error in (
                (hit(lambda r: r.update(body_sha256="0" * 64)), "the captures differ in request hit body_sha256"),
                (hit(lambda r: r.update(prompt_tokens=10)), "the captures differ in request hit prompt_tokens"),
                (hit(lambda r: r.update(cache_salt="tap-capture/other")),
                 "the comparison capture: hit: does not reuse the cache salt of its primer primer"),
                (hit(lambda r: r.update(cached_tokens=0)),
                 "the comparison capture: hit: a prefix hit requires reported 0 < cached_tokens <= 6 (the prompt "
                 "tokens of its primer primer), got 0"),
                (lambda m: [record.pop("body_sha256") for record in m["requests"]],
                 "the comparison capture's request primer records no valid body_sha256")):
            with self.subTest(error=error):
                comparison = self.capture("bf16", error)
                edit_manifest(comparison, change)
                result = self.compare(comparison=comparison)
                self.assertFalse(result["passed"])
                self.assertIn(error, result["errors"])
                self.assertNotIn("equal", result)
        self.assertIn("the captures hold requests ['primer', 'hit'] and ['primer']",
                      self.compare(comparison=self.capture("bf16", "primer-only", requests=REQUESTS[:1]))["errors"])
        # A request with another step count, and a step that holds another readback.
        undecoded = self.capture("bf16", "undecoded", requests=(REQUESTS[0], REQUESTS[1][:-1] + (0,)))
        self.assertIn("request hit has 7 steps in the native capture and 5 in the comparison capture",
                      self.compare(comparison=undecoded)["errors"])
        moved = self.capture("bf16", "moved")
        edit_manifest(moved, hit(lambda r: r["steps"].reverse()))
        self.assertIn("request hit step 0 (step0010 and step0016) differs in capture.rows",
                      self.compare(comparison=moved)["errors"])
        # A step file edited after the capture is refused before any comparison.
        edited = self.capture("bf16", "edited")
        tap = edited / tap_binding.STEPS / "step0001" / "L00.x.kv.window.r0.bin"
        tap.write_bytes(bytes([tap.read_bytes()[0] ^ 1]) + tap.read_bytes()[1:])
        with self.assertRaisesRegex(tap_binding.Unbound, "differs from the files its manifest records"):
            self.compare(comparison=edited)

    def test_experimental_captures_compare_only_with_experimental_and_never_qualify(self):
        native = self.capture("native", "native-experimental", experimental=True)
        comparison = self.capture("bf16", "bf16-experimental", experimental=True)
        refused = self.compare(native, comparison)
        self.assertIn("an experimental capture compares only with --experimental, and never qualifies",
                      refused["errors"])
        self.assertNotIn("equal", refused)
        result = self.compare(native, comparison, experimental=True)
        self.assertTrue(result["passed"], result["errors"])
        self.assertIs(result["qualified"], False)
        self.assertEqual(set(result[experimental_determinism.MARKER]), {"native", "comparison"})
        self.assertIn("--experimental compares two experimental captures, each under its sealed pin",
                      self.compare(experimental=True)["errors"])
        # An experimental and a qualified capture differ in their options.
        self.assertIn("the captures differ in model_options_qualified", self.compare(native=native)["errors"])
        # A pin must name the capture's recorded options.
        renamed = self.capture("bf16", "bf16-renamed", experimental=True)
        edit_manifest(renamed, lambda m: m.update({experimental_determinism.MARKER: sealed(dict(
            m[experimental_determinism.MARKER]["pin"], model_options=OPTIONS["model_options"]))}))
        self.assertIn("the comparison capture: the capture's experimental pin does not name its recorded options, "
                      "checkpoint and process", self.compare(native, renamed, experimental=True)["errors"])
        # Without its pin and mark, an experimental capture still reports
        # unqualified options.
        unmarked = self.capture("bf16", "bf16-unmarked", experimental=True)
        edit_manifest(unmarked, lambda m: [m.pop(experimental_determinism.MARKER), m.pop("qualified")])
        self.assertIn("the comparison capture: the capture records no experimental pin, and no qualified default "
                      "options", self.compare(native, unmarked, experimental=True)["errors"])


class CommandTests(unittest.TestCase):
    def test_compare_writes_a_receipt_and_exits_with_the_result(self):
        # The command reads the module's NATIVE_PLANES, none: an FP8 native
        # layout reports no planes.
        with patch.dict(os.environ, {"CHOPINFER_QUAL_CONFIG": str(gate())}):
            (native, comparison), (native_forced, comparison_forced) = records("fp8"), forced("fp8")
            long, positions, _, _ = kv_layout_equality.forced_expectation()[2][-1]
        failing = copy.deepcopy(comparison_forced)
        failing["texts"][-1]["passes"][0]["positions"][str(positions[1])]["logprob"] += 1.0
        with tempfile.TemporaryDirectory() as directory:
            paths = {}
            for name, value in (("native", native), ("comparison", comparison), ("native_forced", native_forced),
                                ("comparison_forced", comparison_forced), ("failing", failing)):
                paths[name] = Path(directory) / f"{name}.json"
                paths[name].write_text(json.dumps(value))
            receipts = Path(directory) / "receipts"
            env = dict(os.environ, CHOPINFER_QUAL_RECEIPTS=str(receipts),
                       CHOPINFER_QUAL_CONFIG=str(gate()))
            command = [sys.executable, "-B", str(QUAL / "kv_layout_equality.py"), "compare", str(paths["native"]),
                       str(paths["comparison"]), "--native-forced", str(paths["native_forced"])]
            for other, code in ((paths["comparison_forced"], 0), (paths["failing"], 1)):
                result = subprocess.run([*command, "--comparison-forced", str(other)], env=env,
                                        capture_output=True, text=True)
                self.assertEqual(result.returncode, code, result.stderr)
                summary = json.loads(result.stdout.splitlines()[-1])
                self.assertEqual(summary["passed"], code == 0, summary["errors"])
            self.assertIn(f"the forced records differ at {long} position {positions[1]} (forced logprob:",
                          summary["errors"][0])
            written = sorted(receipts.glob("*-kv-layout-equality.json"))
            self.assertEqual(len(written), 2)
            value = json.loads(written[0].read_text())
            self.assertEqual((value["check"], value["passed"]), ("kv-layout-equality", True))
            self.assertEqual(value["server"]["forced_records"]["comparison"]["health"]["serving"]["startup_nonce"],
                             "2" * 32)
            refused = subprocess.run(command, env=env, capture_output=True, text=True)
            self.assertEqual(refused.returncode, 2)
            self.assertIn("compare needs --native-forced and --comparison-forced", refused.stderr)

    def test_taps_compares_two_captures_writes_a_receipt_and_exits_with_the_result(self):
        with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as directory:
            root = Path(directory)
            native, comparison = write_capture(root / "native", "native"), write_capture(root / "bf16", "bf16")
            failing = write_capture(root / "failing", "bf16", corrupt=("kv", 0, "window", 2, 0))
            planes = root / "planes.json"
            planes.write_text(json.dumps(NATIVE_PLANES))
            env = dict(os.environ, CHOPINFER_QUAL_RECEIPTS=str(root / "receipts"))
            command = [sys.executable, "-B", str(QUAL / "kv_layout_equality.py"), "taps", str(native)]
            for other, code in ((comparison, 0), (failing, 1)):
                result = subprocess.run([*command, str(other), "--native-planes", str(planes)], env=env,
                                        capture_output=True, text=True)
                self.assertEqual(result.returncode, code, result.stderr)
                summary = json.loads(result.stdout.splitlines()[-1])
                self.assertEqual(summary["passed"], code == 0)
            self.assertEqual((summary["first_difference"]["tap"], summary["first_difference"]["position"]),
                             ("L00.x.kv.window", 2))
            written = sorted((root / "receipts").glob("*-kv-layout-equality-taps.json"))
            self.assertEqual(len(written), 2)
            value = json.loads(written[0].read_text())
            self.assertEqual((value["check"], value["passed"]), ("kv-layout-equality-taps", True))
            self.assertEqual(value["server"]["captures"]["native"]["profile"]["kv_cache_planes"], NATIVE_PLANES)
            refused = subprocess.run([*command, str(comparison)], env=env, capture_output=True, text=True)
            self.assertEqual(refused.returncode, 2)
            self.assertIn("taps needs --native-planes", refused.stderr)


class RunnerTests(unittest.TestCase):
    FILES = ["kv_layout_native", "kv_layout_comparison", "kv_layout_native_forced", "kv_layout_comparison_forced"]

    def test_the_native_arms_replace_the_configuration_kv_dtype_and_the_check_runs_offline(self):
        # Any previewed model: the check replaces its suite and KV arms.
        facts = runner.model_facts(sorted(runner.declaration.previewed(runner.ROOT))[0])
        facts = dict(facts, full=["kv_layout_equality"],
                     checks={"kv_layout_equality": {"module": "kv_layout_equality", "files": self.FILES}})
        serve = ["--kv-cache-dtype", "bf16", "--speculative-tokens", "0"]
        with patch.object(runner, "module_records", lambda name, model=None: (
                kv_layout_equality.RUNNER_ARMS, kv_layout_equality.RUNNER_RECORDS)):
            arms = runner.plan(facts, "full", "default", {"serve": serve, "environment": {}, "features": []})
        by_name = {arm["arm"]: arm for arm in arms}
        self.assertEqual(by_name["kv-native"]["serve"], ["--speculative-tokens", "0", "--kv-cache-dtype", "auto"])
        self.assertEqual(by_name["kv-native-forced"]["serve"],
                         ["--speculative-tokens", "0", "--kv-cache-dtype", "auto", *runner.FORCED])
        # The comparison layout's forced record runs on the forced prefix arm.
        self.assertEqual(by_name["prefix"]["serve"], [*serve, *runner.FORCED])
        self.assertEqual(by_name["spec"]["serve"], serve)
        records = {step["input"]: (arm["arm"], step) for arm in arms for step in arm["steps"]
                   if step["kind"] == "record" and step["input"] in self.FILES}
        self.assertEqual({name: arm for name, (arm, _) in records.items()},
                         {"kv_layout_native": "kv-native", "kv_layout_comparison": "spec",
                          "kv_layout_native_forced": "kv-native-forced", "kv_layout_comparison_forced": "prefix"})
        # Forced records are verify_ar's K=0 records with row shapes, of the
        # exact reference and the long-context reference.
        verify_ar = runner.RECORDS["verify_ar"][2]
        for name, label in (("kv_layout_native_forced", "kv-native-forced"),
                            ("kv_layout_comparison_forced", "kv-comparison-forced")):
            step = records[name][1]
            self.assertEqual(step["script"], "verify_exact.py")
            arguments = step["arguments"]
            self.assertEqual(arguments[arguments.index("--arm") + 1], label)
            self.assertEqual(set(verify_ar) - {"ar"}, set(arguments) - {label, "--reference", "long_context"})
            self.assertEqual(arguments[arguments.index("--reference") + 1], "long_context")
        self.assertIn("kv_layout_equality", [step.get("check") for step in by_name["offline"]["steps"]])

    def test_only_a_configuration_running_the_check_adds_the_long_context_reference_to_its_forcing_file(self):
        # Each previewed model's configurations and suites write the forcing
        # file the runner wrote before (verify_exact.py references --out)
        # byte for byte, except a configuration whose suite runs this check
        # (a BF16 comparison configuration's full suite).
        import verify_exact
        exact = {"texts": [{"name": "t", "prompt_token_ids": [1, 2, 3]}]}
        long = {"texts": [{"name": "long", "prompt_token_ids": [1, 4, 5, 6]}]}

        def written(argv, out):
            out.unlink(missing_ok=True)
            with patch.object(sys, "argv", ["verify_exact.py", *argv]), \
                    patch.object(verify_exact, "exact_path", return_value=Path("exact.json")), \
                    patch.object(verify_exact, "load_exact", return_value=exact), \
                    patch.object(verify_exact, "named_references",
                                 lambda names: {name: ("0" * 64, long) for name in names or []}):
                verify_exact.main()
            return out.read_bytes()
        changed, running = [], []
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "reference-tokens.json"
            before = written(["references", "--out", str(out)], out)
            for model in sorted(runner.declaration.previewed(runner.ROOT)):
                facts = runner.model_facts(model)
                for name in runner.configurations(model):
                    selected = runner.configuration_facts(facts, name)
                    for suite in runner.SUITES:
                        command = runner.references_command(selected, suite, str(out))
                        self.assertEqual(command[:3], [sys.executable, "-B", str(runner.TOOLS / "verify_exact.py")])
                        after = written(command[3:], out)
                        if after != before:
                            changed.append((model, name, suite, json.loads(after)["sequences"]))
                        if any(selected["checks"][check]["module"] == "kv_layout_equality"
                               for check in selected["quick" if suite == "quick" else "full"]):
                            running.append((model, name, suite, [[1, 2, 3], [1, 4, 5, 6]]))
        self.assertTrue(running)
        self.assertEqual(changed, running)
        self.assertEqual({suite for _, _, suite, _ in running}, {"full"})


class DeclarationTests(unittest.TestCase):
    """Each model configuration that runs this check (a BF16 comparison
    configuration such as kv-bf16)."""

    @staticmethod
    def configurations():
        """The expanded declaration path of each configuration whose full
        suite runs kv_layout_equality."""
        import declaration
        found = []
        for model in sorted(runner.declaration.previewed(runner.ROOT)):
            facts = runner.model_facts(model)
            for name in runner.configurations(model):
                selected = runner.configuration_facts(facts, name)
                if any(selected["checks"][check]["module"] == "kv_layout_equality" for check in selected["full"]):
                    found.append(declaration.named(runner.declaration_name(model, name)))
        return found

    def test_a_configuration_declares_both_layouts_records_and_the_native_planes(self):
        import declaration
        configurations = self.configurations()
        self.assertTrue(configurations)
        for path in configurations:
            with self.subTest(configuration=path.name):
                value = declaration.load(path)
                [check] = [check for check in value["checks"] if check["module"] == "kv_layout_equality"]
                self.assertEqual(check["files"], RunnerTests.FILES)
                self.assertEqual(set(check["files"]), set(kv_layout_equality.RUNNER_RECORDS))
                self.assertEqual(check["args"], ["compare", "{kv_layout_native}", "{kv_layout_comparison}",
                                                 "--native-forced", "{kv_layout_native_forced}",
                                                 "--comparison-forced", "{kv_layout_comparison_forced}"])
                # The package's native planes, each a code its block scale and
                # bytes per token (docs/operations-observability.md).
                planes = check["constants"]["kv_layout_equality"]["NATIVE_PLANES"]["value"]
                self.assertTrue(planes)
                for plane in planes:
                    self.assertEqual(set(plane), {"name", "code", "scale", "bytes_per_token"})
                    self.assertIn(plane["name"], kv_layout_equality.KV_PLANES)
                    self.assertIn(plane["code"], ("bf16", "fp8_e4m3", "fp4_e2m1"))
                    self.assertIn(plane["scale"]["format"], ("e4m3", "ue8m0"))
                    self.assertTrue(type(plane["scale"]["block"]) is int and plane["scale"]["block"] > 0)
                    self.assertTrue(type(plane["bytes_per_token"]) is int and plane["bytes_per_token"] > 0)
                # Every record is bound to the served model and one topology.
                sizes = set()
                for name in check["files"]:
                    rules = {tuple(rule["path"]): rule["value"] for rule in check["input_rules"][name]}
                    self.assertEqual(rules[("server", "model")], "{model}", name)
                    sizes.add(rules[("server", "health", "serving", "profile", "tensor_parallel_size")])
                self.assertEqual(len(sizes), 1)
                dtypes = {tuple(rule["path"]): rule["value"] for rule in check["rules"]}
                self.assertEqual(dtypes[("forced", "kv_cache_dtypes")], dtypes[("kv_cache_dtypes",)])
                # The configuration keeps the forced references its forcing
                # file and forced records need.
                self.assertLessEqual(set(kv_layout_equality.RUNNER_FORCED_REFERENCES),
                                     set(value["qualification"]["references"]))

    def test_the_long_context_reference_moves_no_forceable_position_of_the_exact_texts(self):
        # The runner adds the long-context text to the configuration's forcing
        # file, which every forced arm of the configuration serves: no exact
        # text may gain or lose a forceable position, so its other forced
        # records score the positions they did.
        import common
        import declaration
        import verify_exact
        checked = 0
        for path in self.configurations():
            value = declaration.load(path)
            declared = [common.declared_path(value["inputs"]["exact"]),
                        *(common.declared_path(value["qualification"]["references"][name]["path"])
                          for name in kv_layout_equality.RUNNER_FORCED_REFERENCES)]
            if not all(file is not None and file.is_file() for file in declared):
                continue
            with patch.dict(os.environ, {"CHOPINFER_QUAL_CONFIG": str(path)}):
                exact = verify_exact.load_exact(verify_exact.exact_path(None))
                named = verify_exact.named_references(kv_layout_equality.RUNNER_FORCED_REFERENCES)
            forced = verify_exact.forced_texts(exact, named)
            for text in exact["texts"]:
                ids, start = text["prompt_token_ids"], text["score_from"]
                alone, beside = (verify_exact.agreed_continuations(texts, ids)[start:]
                                 for texts in (exact["texts"], forced))
                self.assertEqual([reach >= 1 for reach in alone], [reach >= 1 for reach in beside],
                                 (path.name, text["name"]))
            checked += 1
        if not checked:
            self.skipTest("needs a configuration's pinned exact and long-context references under $CHOPINFER_DATA")

if __name__ == "__main__":
    unittest.main()
