"""CPU checks of the KV layout served-bits equality check."""
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
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


def server(kv_cache_dtype, package="b"):
    """A bound server identity of one build, serving `kv_cache_dtype`."""
    serving = {
        "checkpoint_identity": "a" * 64, "package_identity": package * 64,
        "metadata_identity": "c" * 64, "startup_nonce": "d" * 32,
        "source_commit": "e" * 40, "bundle_id": "fixture.v1", "version": "1",
        "served_model_names": ["fixture"],
        "profile": {"name": "tp4", "tensor_parallel_size": 4, "kv_cache_dtype": kv_cache_dtype,
                    "max_sequences": 32, "prefix_cache": True,
                    "speculation": {"mode": "adaptive", "drafter": "fixture", "tokens": 5}}}
    return {"model": "fixture", "health": {"serving": serving},
            "checkpoint": {key: serving[key] for key in (
                "checkpoint_identity", "package_identity", "metadata_identity", "startup_nonce")}}


def records():
    """A native and a comparison record of the same answers in one run."""
    native = dict(recorded(image=False, arm="kv-native"), server=server("auto"))
    comparison = dict(recorded(image=False, arm="kv-comparison"), server=server("bf16", package="f"))
    return native, comparison


class CompareTests(unittest.TestCase):
    def test_identical_answers_from_two_layouts_pass(self):
        result = kv_layout_equality.compare(*records())
        self.assertTrue(result["passed"], result["errors"])
        self.assertTrue(result["equal"])
        self.assertIsNone(result["first_difference"])
        self.assertEqual(result["kv_cache_dtypes"], ["auto", "bf16"])
        # The package may follow the KV layout; both are reported.
        self.assertEqual(result["package_identities"], ["b" * 64, "f" * 64])
        self.assertEqual((result["requests"], result["tokens"]), (11, 29))

    def test_one_ulp_logprob_difference_fails_at_its_request_and_position(self):
        native, comparison = records()
        value = comparison["requests"][2]["logprobs"][1]
        comparison["requests"][2]["logprobs"][1] = math.nextafter(value, 0.0)
        result = kv_layout_equality.compare(native, comparison)
        self.assertFalse(result["passed"])
        self.assertEqual({key: result["first_difference"][key] for key in ("request", "field", "position")},
                         {"request": "fibonacci", "field": "logprobs", "position": 1})
        self.assertIn("kv-native and kv-comparison differ at fibonacci logprobs position 1", result["errors"][0])

    def test_a_token_difference_fails_at_its_first_position(self):
        native, comparison = records()
        comparison["requests"][0]["token_ids"][2] += 1
        result = kv_layout_equality.compare(native, comparison)
        self.assertFalse(result["passed"])
        self.assertEqual((result["first_difference"]["field"], result["first_difference"]["position"]),
                         ("token_ids", 2))

    def test_records_must_name_one_build_and_profile_in_one_run_with_two_layouts(self):
        native, comparison = records()
        for mutate, error in (
                (lambda value: value["server"]["health"]["serving"]["profile"].update(kv_cache_dtype="auto"),
                 "both records serve KV dtype auto"),
                (lambda value: value["server"]["health"]["serving"]["profile"].update(max_sequences=16),
                 "the records differ in profile max_sequences"),
                (lambda value: value["server"]["health"]["serving"].update(source_commit="9" * 40),
                 "the records differ in serving identity source_commit"),
                (lambda value: value.update(run_id="another"), "the records come from different runs")):
            changed = copy.deepcopy(comparison)
            mutate(changed)
            result = kv_layout_equality.compare(native, changed)
            self.assertFalse(result["passed"])
            self.assertIn(error, result["errors"])
            # A refused pair is never compared.
            self.assertNotIn("equal", result)
        # A record without a bound serving identity is refused.
        unbound = dict(comparison, server={"model": "fixture"})
        with self.assertRaises(SystemExit):
            kv_layout_equality.compare(native, unbound)


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
# Each plane's code, scale format, scale block and width.
PLANES = {"window": ("e4m3", "ue8m0", 32, 128), "compressed": ("e2m1", "e4m3", 16, 64),
          "index": ("e2m1", "ue8m0", 32, 128)}
LAYERS = {0: ("window",), 2: ("window", "compressed", "index")}
TEXT_TAPS = (("L00.attn.out", "bf16", (0,)), ("L00.x.attn.partial", "f32", (0, 1)))
TEXT_COLS = 8
# A primer prefilled in two chunks and decoded twice, and its continuation,
# a prefix hit restoring 4 tokens, prefilling 5 and decoding once: (name,
# prompt tokens, cached tokens, primer, prefill chunks, decode steps).
REQUESTS = (("primer", 6, 0, None, ((0, 4), (4, 6)), 2), ("hit", 9, 4, "primer", ((4, 9),), 1))
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
    the even channel in the low nibble, UE8M0 scales 2^-4 .. 2^3 and E4M3
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


def readbacks():
    """(request, phase, state end) of every readback, in capture order."""
    found = []
    for name, prompt, cached, _, chunks, decodes in REQUESTS:
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
        self.assertEqual(result["compared"], {"requests": 2, "steps": 17, "kv_rows": kv_rows, "text_rows": 3 * 11})
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
        # The native layout's raw stored bytes are not compared: a capture
        # without them still compares equal.
        bare = self.capture("native", "bare", omit=[(f"L{layer:02d}.x.kv.{plane}.{part}", rank)
                                                    for layer, held in LAYERS.items() for plane in held
                                                    for part in ("codes", "scales") for rank in range(RANKS)])
        self.assertTrue(self.compare(native=bare)["passed"])

    def test_rows_kept_from_a_position_compare_at_their_positions_and_steps_that_kept_none_pair(self):
        # From token 5 on: the primer's first chunk keeps no row and writes
        # no tensor, and its first readback no KV step.
        native, comparison = (self.capture(layout, f"{layout}-from", taps_from=5) for layout in ("native", "bf16"))
        empty = json.loads((native / tap_binding.STEPS / "step0000" / "completion.json").read_text())
        self.assertEqual((empty["completion"], empty["reason"]), ("incomplete", "no selected live tensors"))
        result = self.compare(native, comparison)
        self.assertTrue(result["passed"], result["errors"])
        self.assertEqual(result["kv_readbacks"], {"prefill": 2, "decode": 3})
        # A compressed position kept from its readbacks' fourth row on.
        corrupted = self.capture("bf16", "corrupted-from", taps_from=5, corrupt=("kv", 2, "compressed", 3, 0))
        expected = {(request, phase, end, 2, "compressed", 3, 0) for request, phase, end in readbacks()
                    if 3 < end // RATIO}
        self.assertEqual(self.keys(self.compare(native, corrupted)), expected)
        # A step that kept a row but wrote no tensor is incomplete.
        broken = self.capture("bf16", "broken-from", taps_from=5)
        step = broken / tap_binding.STEPS / "step0001"
        for tap in step.glob("*.bin"):
            tap.unlink()
        (step / "index.jsonl").write_text("")
        (step / "completion.json").write_text(json.dumps({"completion": "incomplete", "expected": 3, "observed": 0,
                                                          "reason": "no selected live tensors"}))
        with self.assertRaisesRegex(tap_binding.Unbound, "step0001 is incomplete: no selected live tensors"):
            self.compare(native, reseal(broken))

    def test_captures_that_differ_in_identity_options_or_layout_are_refused(self):
        for change, error in (
                (lambda m: m["server"].update(source_commit="9" * 40),
                 "the captures differ in serving identity source_commit"),
                (lambda m: m["server"]["profile"].update(max_sequences=8),
                 "the captures differ in profile max_sequences"),
                (lambda m: m["server"]["model_options"].update(fast_route=False),
                 "the captures differ in model_options"),
                (lambda m: [m["server"].pop(key) for key in tap_binding.OPTIONS],
                 "the comparison capture records no complete model_options and model_options_qualified"),
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
                 "the comparison capture records no request body digests")):
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
        native, comparison = records()
        failing = copy.deepcopy(comparison)
        failing["requests"][0]["tokens"][0] += "x"
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for name, value in (("native", native), ("comparison", comparison), ("failing", failing)):
                paths.append(Path(directory) / f"{name}.json")
                paths[-1].write_text(json.dumps(value))
            receipts = Path(directory) / "receipts"
            env = dict(os.environ, CHOPINFER_QUAL_RECEIPTS=str(receipts))
            for other, code in ((paths[1], 0), (paths[2], 1)):
                result = subprocess.run([sys.executable, "-B", str(QUAL / "kv_layout_equality.py"), "compare",
                                         str(paths[0]), str(other)], env=env, capture_output=True, text=True)
                self.assertEqual(result.returncode, code, result.stderr)
                self.assertEqual(json.loads(result.stdout.splitlines()[-1])["passed"], code == 0)
            written = sorted(receipts.glob("*-kv-layout-equality.json"))
            self.assertEqual(len(written), 2)
            value = json.loads(written[0].read_text())
            self.assertEqual(value["check"], "kv-layout-equality")

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
    def test_the_native_arm_replaces_the_configuration_kv_dtype_and_the_check_runs_offline(self):
        # Any previewed model: the check replaces its suite and KV arms.
        facts = runner.model_facts(sorted(runner.declaration.previewed(runner.ROOT))[0])
        facts = dict(facts, full=["kv_layout_equality"],
                     checks={"kv_layout_equality": {"module": "kv_layout_equality",
                                                    "files": ["kv_layout_native", "kv_layout_comparison"]}})
        with patch.object(runner, "module_records", lambda name, model=None: (
                kv_layout_equality.RUNNER_ARMS, kv_layout_equality.RUNNER_RECORDS)):
            arms = runner.plan(facts, "full", "default", {"serve": ["--kv-cache-dtype", "bf16"],
                                                          "environment": {}, "features": []})
        by_name = {arm["arm"]: arm for arm in arms}
        self.assertEqual(by_name["kv-native"]["serve"], ["--kv-cache-dtype", "auto"])
        self.assertEqual(by_name["spec"]["serve"], ["--kv-cache-dtype", "bf16"])
        self.assertEqual([step["input"] for step in by_name["kv-native"]["steps"]], ["kv_layout_native"])
        self.assertIn("kv_layout_comparison", [step.get("input") for step in by_name["spec"]["steps"]])
        self.assertIn("kv_layout_equality", [step.get("check") for step in by_name["offline"]["steps"]])


if __name__ == "__main__":
    unittest.main()
