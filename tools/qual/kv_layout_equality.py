#!/usr/bin/env python3
"""Served-bits equality of two KV storage layouts of the same values.

A model whose comparison KV layout stores the same values as its served one
(BF16 pages holding exactly what the native formats decode to) must serve
the same bits from both. The correctness runner records the determinism
request set (tools/qual/determinism.py: C1 greedy, 48 output tokens with
top-5 logprobs, a long prompt over two prefill chunks, token-ID sends and
prefix hits) on two servers of one configuration that differ only in
`--kv-cache-dtype`: the configuration's own server (`spec`, serving its
comparison layout) and a `kv-native` server, which serves `auto`, the
package's preferred layout. `compare` requires every request's text, finish
reason, token IDs, tokens and selected and top logprobs to be bit-identical
across the two records, and names the first differing request, field and
position.

Both records must come from one run and one build: the same checkpoint,
metadata, source commit, bundle, version and served model, and the same
effective profile in every field but `kv_cache_dtype`, which must differ.
The package identity may differ with the KV layout; both are reported.

`taps` compares the two layouts' diagnostic taps instead: two captures from
tools/qual/tap_binding.py capture, one from a diagnostic-taps server of each
layout, each read offline (tap_binding.recorded: every identity and option
field is recorded, and every step holds its recorded digests and names its
process and its own request). It refuses
captures that differ in prompts (each request's name, path, body digest,
primer and reported prompt and cached tokens, under tap_binding's cache
rules), in steps (each request's steps, their capture.json and items.json
with engine request IDs read as request names, apart from state slots, the
process's capture ID and producer, the declared taps and their bytes, and a
KV step's storage), or in serving identity: every identity and profile field
must be equal except the startup nonce, the package identity (reported; it
may differ) and the layout fields, which `layout_errors` validates against
the declared native planes (--native-planes, a JSON list: NATIVE_PLANES).
Both captures must record the same complete `model_options` and
`model_options_qualified`. An experimental capture (a sealed pin's
diagnostic options) compares only with --experimental, against another one,
and the result is never qualification.

Within each pair of steps every tap row is compared bit for bit, in logical
representation. A KV-state tap (`L<nn>.x.kv.<plane>` in a step whose
items.json has `kv`) is keyed by readback (request, phase and state end),
layer, plane, position and rank: its logical BF16 values are compared. A
layout's raw stored bytes (`<tap>.codes`, `<tap>.scales`) are its own
representation and are not compared across layouts, but the native capture
must hold both for every plane, rank and readback it writes, and they must
decode exactly to its logical values in the formats NATIVE_KV_FORMATS names
(window E4M3 codes with a UE8M0 scale per 32, compressed E2M1 codes with a
UE4M3 scale per 16, index E2M1 codes with a UE8M0 scale per 32, decoded as
kernel_core::codec does). A product that is not exactly a finite BF16 is
refused as insufficient evidence, never rounded, and the comparison capture
must hold no raw taps. A text tap is keyed by name, request, position (or
encoder row) and rank. A row one
capture holds and the other lacks is a difference. The result names the
first difference in the native capture's order (requests, steps, taps in
program order, positions), lists up to SHOWN differences and counts the
differing rows of each tap. A comparison is insufficient evidence, and
fails, unless it compared text tap rows and KV-state rows of every plane
in KV_PLANES and every phase in KV_PHASES, and of a restore when the
captures hold a prefix-hit pair. A step that wrote no tap (one that kept
no row, or a verify step) compares as empty.

    python3 tools/qual/kv_layout_equality.py compare NATIVE_RECORD COMPARISON_RECORD
    python3 tools/qual/kv_layout_equality.py taps NATIVE_CAPTURE COMPARISON_CAPTURE \\
        --native-planes PLANES.json [--experimental]
"""

import argparse
import contextlib
import copy
import functools
import json
import math
from pathlib import Path
import re
import struct
import sys

from common import receipt
import determinism
import experimental_determinism
from prefix_exact import serving_identity
import tap_binding
# tap_binding puts the repository root on the path.
from tools.model_sdk import diagnostic as wire

_RECORD = ["record", "--url", "{url}", "--model", "{name}", "--timeout", "120", "--run-id", "{run_id}",
           "--server-log", "{server_log}", "{image}"]
# The correctness runner's arms and records (tools/qual/README.md#kv-layout-equality):
# the kv-native server replaces the configuration's --kv-cache-dtype with
# auto; the configuration's own server records the comparison layout.
RUNNER_ARMS = {"kv-native": ["--kv-cache-dtype", "auto"]}
RUNNER_RECORDS = {"kv_layout_native": ("kv-native", "determinism.py", [*_RECORD, "--arm", "kv-native"]),
                  "kv_layout_comparison": ("spec", "determinism.py", [*_RECORD, "--arm", "kv-comparison"])}
RUNNER_CHECK_ARM = "offline"
# The serving identity fields that follow the KV layout.
LAYOUT_FIELDS = ("package_identity",)
# The effective profile fields of the KV layout, which `layout_errors`
# validates instead of requiring them equal.
LAYOUT_PROFILE = ("kv_cache_dtype", "kv_cache_planes")
SIDES = ("native", "comparison")
# The KV elements a serving profile resolves (`kv_cache_dtype`).
KV_DTYPES = ("bf16", "fp8", "mixed")
# A capture request's fields both captures must share, beside its name.
REQUEST_FIELDS = ("path", "body_sha256", "primer", "prompt_tokens", "cached_tokens")
# The capture.json fields of a step that name its process or follow its KV
# layout (the declared taps and their bytes).
PROCESS_STEP_FIELDS = ("capture_id", "producer", "expected", "tensor_bytes")
# The fields of a declared tap its index entry must hold, as tapdiff checks.
DESCRIPTOR_FIELDS = ("dtype", "cols", "first_row", "rows")
# A KV-state tap of a KV step: layer, plane, and a layout's raw stored part.
KV_TAP = r"L(\d+)\.x\.kv\.([a-z0-9_]+)(?:\.(codes|scales))?"
TAP_BYTES = {"bf16": 2, "f32": 4, "f64": 8, "u32": 4, "u8": 1}
# The differences a taps result lists.
SHOWN = 32
# The bytes of each tap file read at once.
BLOCK_BYTES = 16 << 20
# The KV-state planes and readback phases a taps comparison must compare
# rows of; a restore is owed only by captures holding a prefix-hit pair.
KV_PLANES = ("window", "compressed", "index")
KV_PHASES = ("prefill", "decode")
RESTORE_PHASE = "restore"
# Each native KV-state plane's stored element code, scale format and values
# per scale, as kernel_core's trained KV quantizers store them
# (precision::quantizer::BlockQuantization E4M3_CEIL32, E2M1E4M3Scale16 and
# E2M1Ceil32): E4M3 codes with a UE8M0 scale per 32, E2M1 codes (two a
# byte, the even channel in the low nibble) with a UE4M3 (unsigned E4M3)
# scale per 16, and E2M1 codes with a UE8M0 scale per 32.
NATIVE_KV_FORMATS = {"window": ["e4m3", "ue8m0", 32], "compressed": ["e2m1", "ue4m3", 16],
                     "index": ["e2m1", "ue8m0", 32]}
# The completion reason of a step that wrote no tensor (runtime_bundle_sdk
# taps): with CHOPINFER_TAPS_FROM past its rows, or no tap declared.
NOTHING_SELECTED = "no selected live tensors"


def identities(native, comparison):
    """The two records' serving identities, refused unless they name one
    build and profile that differ only in KV storage."""
    left, right = serving_identity(native), serving_identity(comparison)
    errors = []
    if native.get("run_id") != comparison.get("run_id") or not native.get("run_id"):
        errors.append("the records come from different runs")
    for key in sorted((left.keys() | right.keys()) - {"profile", *LAYOUT_FIELDS}):
        if left.get(key) != right.get(key):
            errors.append(f"the records differ in serving identity {key}")
    profiles = left["profile"], right["profile"]
    for key in sorted((profiles[0].keys() | profiles[1].keys()) - {"kv_cache_dtype"}):
        if profiles[0].get(key) != profiles[1].get(key):
            errors.append(f"the records differ in profile {key}")
    dtypes = [profile["kv_cache_dtype"] for profile in profiles]
    if dtypes[0] == dtypes[1]:
        errors.append(f"both records serve KV dtype {dtypes[0]}")
    return {"native": left, "comparison": right, "kv_cache_dtypes": dtypes}, errors


def same(left, right):
    """Whether two JSON values are equal with their types: `true` is not
    `1`, nor `2` `2.0`."""
    return wire.canonical(left) == wire.canonical(right)


def layout_errors(native, comparison, native_planes):
    """Why two effective profiles are not one native and one comparison KV
    layout: `kv_cache_planes` is present exactly when `kv_cache_dtype` is
    `mixed` (docs/operations-observability.md, "Serving identity"), the
    native planes equal `native_planes` (the declared NATIVE_PLANES), the
    comparison reports none, and the two serve different KV dtypes, each
    one of KV_DTYPES."""
    if not isinstance(native_planes, list):
        return ["no native planes are declared (NATIVE_PLANES)"]
    errors = []
    for side, profile in zip(SIDES, (native, comparison)):
        if not any(same(profile.get("kv_cache_dtype"), dtype) for dtype in KV_DTYPES):
            errors.append(f"the {side} profile reports KV dtype {profile.get('kv_cache_dtype')!r}, not one of "
                          f"{', '.join(KV_DTYPES)}")
        if profile.get("kv_cache_dtype") == "mixed" and "kv_cache_planes" not in profile:
            errors.append(f"the {side} profile serves KV dtype mixed without kv_cache_planes")
    if "kv_cache_planes" in native and native.get("kv_cache_dtype") != "mixed":
        errors.append(f"the native profile reports kv_cache_planes for KV dtype {native.get('kv_cache_dtype')}")
    if "kv_cache_planes" in comparison:
        errors.append("the comparison profile reports kv_cache_planes")
    if not same(native.get("kv_cache_planes", []), native_planes):
        errors.append("the native profile's kv_cache_planes differ from NATIVE_PLANES")
    if same(native.get("kv_cache_dtype"), comparison.get("kv_cache_dtype")):
        errors.append(f"both serve KV dtype {native.get('kv_cache_dtype')}")
    return errors


def compare(native, comparison):
    """The check's result for two determinism records."""
    served, errors = identities(native, comparison)
    result = {"kv_cache_dtypes": served["kv_cache_dtypes"],
              "package_identities": [served[side].get("package_identity") for side in ("native", "comparison")],
              "requests": len(native.get("requests") or []),
              "tokens": sum(len(row.get("token_ids") or []) for row in native.get("requests") or [])}
    if not errors:
        metrics = determinism.compare(native, comparison)
        result |= {"equal": metrics["equal"], "first_difference": metrics.get("first_difference"),
                   "max_abs_difference": metrics.get("max_abs_difference")}
        if not metrics["passed"]:
            errors.append(determinism.describe(metrics))
    return dict(result, passed=not errors, errors=errors)


def capture_identities(native, comparison, native_planes, experimental=False):
    """The two capture manifests' layouts and experimental pins, and the
    errors that refuse the pair: see the module docstring."""
    servers = native["server"], comparison["server"]
    errors = []
    excluded = {"profile", "startup_nonce", *LAYOUT_FIELDS, *tap_binding.OPTIONS}
    for key in sorted((servers[0].keys() | servers[1].keys()) - excluded):
        if not same(servers[0].get(key), servers[1].get(key)):
            errors.append(f"the captures differ in serving identity {key}")
    profiles = servers[0]["profile"], servers[1]["profile"]
    for key in sorted((profiles[0].keys() | profiles[1].keys()) - set(LAYOUT_PROFILE)):
        if not same(profiles[0].get(key), profiles[1].get(key)):
            errors.append(f"the captures differ in profile {key}")
    errors += layout_errors(*profiles, native_planes)
    for key in tap_binding.OPTIONS:
        if not same(servers[0].get(key), servers[1].get(key)):
            errors.append(f"the captures differ in {key}")
    markers = []
    for side, manifest in zip(SIDES, (native, comparison)):
        try:
            markers.append(tap_binding.experimental_marker(manifest))
        except tap_binding.Unbound as error:
            errors.append(f"the {side} capture: {error}")
            markers.append(None)
    pinned = [marker is not None for marker in markers]
    if experimental and not all(pinned):
        errors.append("--experimental compares two experimental captures, each under its sealed pin")
    if not experimental and any(pinned):
        errors.append("an experimental capture compares only with --experimental, and never qualifies")
    return {"kv_cache_dtypes": [profile.get("kv_cache_dtype") for profile in profiles],
            "package_identities": [server.get("package_identity") for server in servers],
            "markers": markers}, errors


def request_errors(native, comparison):
    """Why the two captures do not hold the same prompts under the same
    cache rules: every record names its path, reported prompt and cached
    tokens and the 64-hex-digit digest of its body, and both captures'
    records agree in their types."""
    errors = []
    valid = {"path": lambda value: isinstance(value, str) and value != "",
             "prompt_tokens": lambda value: type(value) is int and value > 0,
             "cached_tokens": lambda value: type(value) is int and value >= 0,
             "body_sha256": lambda value: isinstance(value, str) and re.fullmatch("[0-9a-f]{64}", value)}
    for side, manifest in zip(SIDES, (native, comparison)):
        try:
            tap_binding.check_requests(manifest["requests"])
        except tap_binding.Unbound as error:
            errors.append(f"the {side} capture: {error}")
        errors += [f"the {side} capture's request {record.get('name')} records no valid {field}"
                   for record in manifest["requests"] for field, check in valid.items()
                   if not check(record.get(field))]
    names = [[record["name"] for record in manifest["requests"]] for manifest in (native, comparison)]
    if not same(*names):
        errors.append(f"the captures hold requests {names[0]} and {names[1]}")
    elif len(set(names[0])) != len(names[0]):
        errors.append(f"the captures name two requests alike: {names[0]}")
    else:
        for left, right in zip(native["requests"], comparison["requests"]):
            errors += [f"the captures differ in request {left['name']} {key}"
                       for key in REQUEST_FIELDS if not same(left.get(key), right.get(key))]
    return errors


def differing_field(left, right, path=""):
    """The first field, as a dotted path, at which two JSON values differ,
    or None."""
    if isinstance(left, dict) and isinstance(right, dict):
        for key in [*left, *(key for key in right if key not in left)]:
            if key not in left or key not in right:
                return path + key
            if found := differing_field(left[key], right[key], f"{path}{key}."):
                return found
        return None
    if isinstance(left, list) and isinstance(right, list) and len(left) == len(right):
        return next((found for at, (a, b) in enumerate(zip(left, right))
                     if (found := differing_field(a, b, f"{path}{at}."))), None)
    return None if same(left, right) else path.rstrip(".") or "value"


class Step:
    """One step of a capture's request, its engine request ID read as the
    request's name: its index, what a paired step must share (`shape`) and
    each tensor row's label (`labels`)."""

    def __init__(self, directory, request, engine):
        self.dir, self.request = directory, request
        engine = tap_binding.engine_id(engine, f"the record of {directory}")
        capture = wire.read_json(directory / "capture.json")
        items = wire.read_json(directory / "items.json")
        completion = wire.read_json(directory / "completion.json")
        self.index = [wire.decode_json(line) for line in wire.read_bytes(directory / "index.jsonl").splitlines()]
        stored = [(entry["name"], entry["rank"]) for entry in self.index]
        declared = [(entry["name"], entry.get("rank", 0)) for entry in capture.get("expected") or []]
        # A step that kept no row, or declared no tap, wrote nothing.
        empty = (not self.index and completion.get("observed") == 0 and completion.get("reason") == NOTHING_SELECTED
                 and (capture.get("captured_rows") == 0 or not declared))
        if completion.get("completion") != "full" and not empty:
            raise tap_binding.Unbound(f"{directory} is incomplete: {completion.get('reason')}")
        if not empty and (len(set(stored)) != len(stored) or sorted(stored) != sorted(declared)):
            raise tap_binding.Unbound(f"{directory}: index.jsonl holds other taps than capture.json expects")
        descriptors = {(entry["name"], entry.get("rank", 0)): entry for entry in capture.get("expected") or []}
        for entry in self.index:
            declared = descriptors[(entry["name"], entry["rank"])]
            for field in DESCRIPTOR_FIELDS:
                if not same(entry.get(field), declared.get(field)):
                    raise tap_binding.Unbound(f"{directory}: {entry['name']} rank {entry['rank']} has {field} "
                                              f"{entry.get(field)!r}; capture.json declares {declared.get(field)!r}")
        self.kv = items.get("kv")
        named = copy.deepcopy(items)
        self.positions, self.encoder = {}, {}
        for item in named["items"]:
            item.pop("slot", None)
            if tap_binding.engine_id(item.get("request"), f"{directory}/items.json") == engine:
                item["request"] = request
                for offset, position in enumerate(item.get("positions") or []):
                    self.positions[item["start"] + offset] = position
        for tap, segments in (named.get("encoder") or {}).items():
            for segment in segments:
                if tap_binding.engine_id(segment.get("request"), f"{directory}/items.json {tap}") == engine:
                    segment["request"] = request
                    for offset in range(segment["rows"]):
                        self.encoder.setdefault(tap, {})[segment["start"] + offset] = [
                            "enc", segment.get("kind"), segment.get("item"), segment["index"] + offset]
        if self.kv is not None:
            named["kv"].pop("storage", None)
            positions = self.kv.get("positions") or []
            if len(set(positions)) != len(positions):
                raise tap_binding.Unbound(f"{directory}: a KV step holds a position twice")
            for entry in self.index:
                if not re.fullmatch(KV_TAP, entry["name"]):
                    raise tap_binding.Unbound(f"{directory}: KV step holds {entry['name']}, not a KV-state tap")
        for entry in self.index:
            self.check(entry, capture)
        self.shape = {"capture": {key: value for key, value in capture.items() if key not in PROCESS_STEP_FIELDS},
                      "items": named}

    def check(self, entry, capture):
        """Refuses an index entry with malformed descriptors, or a row the
        step does not name: a sequence tap's rows lie within the rows the
        step kept, a KV-state tap's within its positions, and a text tap's
        are each owned by an item; an encoder tap's are each mapped by a
        segment."""
        where = f"{self.dir}: {entry.get('name')} rank {entry.get('rank')}"
        if (not all(type(entry.get(key)) is int for key in ("rank", "first_row", "rows", "cols"))
                or entry["first_row"] < 0 or entry["rows"] < 1 or entry["cols"] < 1
                or entry.get("dtype") not in TAP_BYTES or entry.get("axis") not in (None, "encoder")):
            raise tap_binding.Unbound(f"{where} has malformed row descriptors")
        first, end = entry["first_row"], entry["first_row"] + entry["rows"]
        if entry.get("axis") == "encoder":
            if unmapped := [row for row in range(first, end) if row not in self.encoder.get(entry["name"], {})]:
                raise tap_binding.Unbound(f"{where} holds encoder row {unmapped[0]}, which no segment maps")
            return
        kept = capture.get("first_row"), capture.get("captured_rows")
        if not all(type(value) is int for value in kept):
            raise tap_binding.Unbound(f"{self.dir}: capture.json names no rows kept")
        if first < kept[0] or end > kept[0] + kept[1]:
            raise tap_binding.Unbound(f"{where} holds rows {first} .. {end - 1}, past its rows {kept[0]} .. "
                                      f"{kept[0] + kept[1] - 1}")
        if self.kv is not None:
            if end > len(self.kv.get("positions") or []):
                raise tap_binding.Unbound(f"{where} holds rows {first} .. {end - 1}, past its KV positions")
            part = self.kv_tap(entry["name"])[2]
            if entry["dtype"] != ("bf16" if part is None else "u32"):
                raise tap_binding.Unbound(f"{where} has dtype {entry['dtype']}; a KV-state tap's "
                                          + ("logical values are bf16" if part is None else "stored bytes are u32"))
        elif unowned := [row for row in range(first, end) if row not in self.positions]:
            raise tap_binding.Unbound(f"{where} holds row {unowned[0]}, which no item owns")

    def kv_tap(self, name):
        """(layer, plane, raw part or None) of a KV step's tap, else None."""
        found = re.fullmatch(KV_TAP, name) if self.kv is not None else None
        return (int(found[1]), found[2], found[3]) if found else None

    def labels(self, entry):
        """Each tensor row's position (a KV step's state or compressed
        position, a text tap's token position) or encoder row; `check`
        refused an entry with a row that has none."""
        rows = range(entry["first_row"], entry["first_row"] + entry["rows"])
        if self.kv is not None:
            return [self.kv["positions"][row] for row in rows]
        if entry.get("axis") == "encoder":
            return [self.encoder[entry["name"]][row] for row in rows]
        return [self.positions[row] for row in rows]

    def where(self, entry, label):
        """A row's key: its request, tap, rank and position, and a KV-state
        tap's readback (phase and state end), layer and plane."""
        key = {"request": self.request, "step": self.dir.name, "tap": entry["name"], "rank": entry["rank"],
               "position": label}
        if kv := self.kv_tap(entry["name"]):
            key |= {"phase": self.kv.get("phase"), "end": self.kv.get("end"), "layer": kv[0], "plane": kv[1]}
        return key


def decoded(fmt, code):
    """The exact value of `code` in `fmt` (e4m3, e2m1, ue4m3 or ue8m0), as
    kernel_core::codec decodes it: NaN for E4M3's all-ones magnitude, for
    UE4M3 codes from 0x80 and for UE8M0 0xFF."""
    if fmt == "ue8m0":
        return math.nan if code == 0xFF else math.ldexp(1.0, code - 127)
    if fmt == "ue4m3" and code & 0x80:
        return math.nan
    if fmt in ("e4m3", "ue4m3"):
        exponent, mantissa, sign = (code >> 3) & 0xF, code & 7, code & 0x80
        if exponent == 15 and mantissa == 7:
            return math.nan
        value = math.ldexp(mantissa, -9) if exponent == 0 else math.ldexp(8 + mantissa, exponent - 10)
    elif fmt == "e2m1":
        exponent, mantissa, sign = (code >> 1) & 3, code & 1, code & 8
        value = math.ldexp(mantissa, -1) if exponent == 0 else math.ldexp(2 + mantissa, exponent - 2)
    else:
        raise ValueError(f"no decoder for {fmt}")
    return -value if sign else value


def bf16_bits(value):
    """The BF16 bits of `value` when it is exactly a finite BF16, else None:
    nothing is rounded."""
    if not math.isfinite(value):
        return None
    try:
        packed = struct.pack("<f", value)
    except OverflowError:
        return None
    bits = struct.unpack("<I", packed)[0]
    return None if struct.unpack("<f", packed)[0] != value or bits & 0xFFFF else bits >> 16


@functools.lru_cache(maxsize=None)
def products(element, scale_format, scale):
    """For one scale code, per stored code byte: translate tables of each
    little-endian byte of its channels' BF16 products (one channel a byte,
    or two for E2M1, the even channel in the low nibble), and a table of 1
    for the bytes whose every product is exactly a finite BF16."""
    factor, pairs = decoded(scale_format, scale), element == "e2m1"
    tables, valid = [bytearray(256) for _ in range(4 if pairs else 2)], bytearray(256)
    for byte in range(256):
        bits = [bf16_bits(decoded(element, code) * factor) for code in ((byte & 0xF, byte >> 4) if pairs else (byte,))]
        if None in bits:
            continue
        valid[byte] = 1
        for at, value in enumerate(bits):
            tables[2 * at][byte], tables[2 * at + 1][byte] = value & 0xFF, value >> 8
    return [bytes(table) for table in tables], bytes(valid)


def native_row(plane, codes, scales, values):
    """None when a native row's stored codes and scales decode exactly to
    its logical BF16 `values`; ("inexact", channel, product) for the first
    channel whose product is not exactly a finite BF16; ("differs",) when
    the products are not every byte of `values`, or the row's codes, scales
    and values are not whole rows of one another."""
    element, scale_format, block = NATIVE_KV_FORMATS[plane]
    per_byte = 2 if element == "e2m1" else 1
    span = block // per_byte
    if len(codes) != span * len(scales) or len(values) != 2 * block * len(scales):
        return ("differs",)
    for at, scale in enumerate(scales):
        tables, valid = products(element, scale_format, scale)
        chunk = codes[at * span:(at + 1) * span]
        flags = chunk.translate(valid)
        if 0 in flags:
            byte = flags.index(0)
            for part, code in enumerate((chunk[byte] & 0xF, chunk[byte] >> 4) if per_byte == 2 else (chunk[byte],)):
                product = decoded(element, code) * decoded(scale_format, scale)
                if bf16_bits(product) is None:
                    return "inexact", at * block + byte * per_byte + part, product
        expected = values[at * block * 2:(at + 1) * block * 2]
        if any(chunk.translate(table) != expected[part::len(tables)] for part, table in enumerate(tables)):
            return ("differs",)
    return None


def native_storage(step):
    """(rows decoded, errors) of a KV step of the native capture. Each plane
    and rank must hold its logical values, codes and scales over the same
    rows, in the widths NATIVE_KV_FORMATS gives, or the capture is refused;
    an error names a row whose stored bytes do not decode exactly to its
    logical values: products differ, or one is not exactly a finite BF16
    (insufficient evidence, never rounded)."""
    groups = {}
    for entry in step.index:
        layer, plane, part = step.kv_tap(entry["name"])
        groups.setdefault((layer, plane, entry["rank"]), {})[part] = entry
    rows, errors = 0, []
    for (layer, plane, rank), parts in groups.items():
        values = parts.get(None)
        if values is None or set(parts) != {None, "codes", "scales"}:
            name = values["name"] if values else next(iter(parts.values()))["name"]
            raise tap_binding.Unbound(f"{step.dir}: the native capture holds {name} rank {rank} without its "
                                      + ("stored codes and scales" if values else "logical values"))
        if plane not in NATIVE_KV_FORMATS:
            raise tap_binding.Unbound(f"{step.dir}: no native format for plane {plane} (NATIVE_KV_FORMATS)")
        element, _, block = NATIVE_KV_FORMATS[plane]
        width = values["cols"]
        widths = {"codes": width // (2 if element == "e2m1" else 1), "scales": width // block}
        for part, entry in parts.items():
            if part is not None and ((entry["first_row"], entry["rows"]) != (values["first_row"], values["rows"])
                                     or entry["dtype"] != "u32" or entry["cols"] * 4 != widths[part]
                                     or width % block):
                raise tap_binding.Unbound(f"{step.dir}: the native capture's {entry['name']} rank {rank} does not "
                                          f"hold {widths[part]} stored bytes of each of {values['name']}'s rows")
        labels = step.labels(values)
        with contextlib.ExitStack() as files:
            streams = {}
            for part, entry in parts.items():
                stream, size = files.enter_context(wire.opened(step.dir / f"{entry['name']}.r{rank}.bin"))
                row_bytes = entry["cols"] * TAP_BYTES[entry["dtype"]]
                if size != entry["rows"] * row_bytes:
                    raise tap_binding.Unbound(f"{step.dir}: {entry['name']} rank {rank} does not hold the rows its "
                                              "index declares")
                streams[part] = stream, row_bytes
            for label in labels:
                read = {part: stream.read(row_bytes) for part, (stream, row_bytes) in streams.items()}
                found = native_row(plane, read["codes"], read["scales"], read[None])
                if found and len(errors) < SHOWN:
                    where = (f"the native capture's {values['name']} rank {rank} ({step.kv.get('phase')} readback "
                             f"ending at {step.kv.get('end')}, request {step.request} position {label})")
                    errors.append(f"{where} holds values its codes and scales do not decode to"
                                  if found[0] == "differs" else
                                  f"insufficient evidence: {where} decodes channel {found[1]} to {found[2]!r}, not "
                                  "an exact finite BF16")
            rows += len(labels)
    return rows, errors


def differing_rows(left, right, row_bytes, rows):
    """The rows whose bytes differ between two tap files of `rows` rows."""
    found, block = [], max(1, BLOCK_BYTES // row_bytes) * row_bytes
    with wire.opened(left) as (a, left_size), wire.opened(right) as (b, right_size):
        if left_size != rows * row_bytes or right_size != rows * row_bytes:
            raise tap_binding.Unbound(f"{left} or {right} does not hold the {rows} rows its index declares")
        for start in range(0, rows * row_bytes, block):
            x, y = a.read(block), b.read(block)
            if x != y:
                found += [start // row_bytes + row for row in range(len(x) // row_bytes)
                          if x[row * row_bytes:(row + 1) * row_bytes] != y[row * row_bytes:(row + 1) * row_bytes]]
    return found


def unique_rows(steps):
    """Refuses a capture holding a row key twice: a KV readback's tap and
    rank in two steps, or a text tap's position in two steps."""
    seen = set()
    for step in steps:
        for entry in step.index:
            kv = step.kv_tap(entry["name"])
            if kv is not None:
                keys = [(step.request, step.kv.get("phase"), step.kv.get("end"), entry["name"], entry["rank"])]
            else:
                keys = [(step.request, entry["name"], entry["rank"], json.dumps(label)) for label in step.labels(entry)]
            for key in keys:
                if key in seen:
                    raise tap_binding.Unbound(f"{step.dir}: {entry['name']} rank {entry['rank']} holds a row "
                                              "another step holds")
                seen.add(key)


def compare_steps(native_dir, native, comparison_dir, comparison):
    """(errors refusing the captures' steps, the bitwise comparison of their
    taps or None)."""
    errors, pairs = [], []
    for left, right in zip(native["requests"], comparison["requests"]):
        name = left["name"]
        if len(left["steps"]) != len(right["steps"]):
            errors.append(f"request {name} has {len(left['steps'])} steps in the native capture and "
                          f"{len(right['steps'])} in the comparison capture")
            continue
        for ordinal, (a, b) in enumerate(zip(left["steps"], right["steps"])):
            pair = (Step(native_dir / tap_binding.STEPS / a, name, left["engine_request_id"]),
                    Step(comparison_dir / tap_binding.STEPS / b, name, right["engine_request_id"]))
            if field := differing_field(*(step.shape for step in pair)):
                errors.append(f"request {name} step {ordinal} ({a} and {b}) differs in {field}")
            pairs.append(pair)
    if errors:
        return errors, None
    for side in (0, 1):
        unique_rows(pair[side] for pair in pairs)
    shown, taps, raw, storage = [], {}, {side: 0 for side in SIDES}, []
    compared = {"requests": len(native["requests"]), "steps": len(pairs), "kv_rows": 0, "text_rows": 0,
                "native_rows_decoded": 0}
    readbacks, planes, phases = {}, dict.fromkeys(KV_PLANES, 0), {}
    for pair in pairs:
        if pair[0].kv is not None:
            readbacks.setdefault(pair[0].kv.get("phase"), set()).add((pair[0].request, pair[0].kv.get("end")))
            decoded_rows, errors_of = native_storage(pair[0])
            compared["native_rows_decoded"] += decoded_rows
            storage += errors_of
        if pair[1].kv is not None:
            for entry in pair[1].index:
                if pair[1].kv_tap(entry["name"])[2] is not None:
                    raise tap_binding.Unbound(f"{pair[1].dir}: the comparison capture holds raw stored tap "
                                              f"{entry['name']} rank {entry['rank']}")
        entries = [{(entry["name"], entry["rank"]): entry for entry in step.index} for step in pair]
        for key in [*entries[0], *(key for key in entries[1] if key not in entries[0])]:
            found = [side.get(key) for side in entries]
            kv = pair[0].kv_tap(key[0])
            if kv is not None and kv[2] is not None:
                for side, entry in zip(SIDES, found):
                    raw[side] += entry["rows"] if entry else 0
                continue
            if None in found:
                lacking = SIDES[found.index(None)]
                present = 1 - found.index(None)
                rows = list(enumerate(pair[present].labels(found[present])))
                what = f"missing from the {lacking} capture"
            else:
                if any(found[0].get(field) != found[1].get(field)
                       for field in ("dtype", "cols", "first_row", "rows", "axis")):
                    errors.append(f"request {pair[0].request} {pair[0].dir.name} tap {key[0]} rank {key[1]} has "
                                  "another dtype, width or rows in the comparison capture")
                    continue
                entry, present = found[0], 0
                labels = pair[0].labels(entry)
                compared["kv_rows" if kv else "text_rows"] += len(labels)
                if kv:
                    planes[kv[1]] = planes.get(kv[1], 0) + len(labels)
                    phases[pair[0].kv.get("phase")] = phases.get(pair[0].kv.get("phase"), 0) + len(labels)
                paths = [step.dir / f"{key[0]}.r{key[1]}.bin" for step in pair]
                differ = differing_rows(*paths, entry["cols"] * TAP_BYTES[entry["dtype"]], entry["rows"])
                rows = [(row, labels[row]) for row in differ]
                what = "differs"
            if rows:
                taps[key[0]] = taps.get(key[0], 0) + len(rows)
                shown += [dict(pair[present].where(found[present], label), difference=what)
                          for _, label in rows[:max(0, SHOWN - len(shown))]]
    if errors:
        return errors, None
    total = sum(taps.values())
    owed = [*KV_PHASES, *([RESTORE_PHASE] if any(record.get("primer") for record in native["requests"]) else [])]
    insufficient = [what for what, rows in (("text tap rows", compared["text_rows"]),
                                            ("KV-state rows", compared["kv_rows"]),
                                            *((f"KV-state rows of plane {plane}", planes[plane])
                                              for plane in KV_PLANES),
                                            *((f"KV-state rows of phase {phase}", phases.get(phase, 0))
                                              for phase in owed)) if not rows]
    return [], {"equal": total == 0, "first_difference": shown[0] if shown else None, "differences": shown,
                "differing_rows": total, "differing_taps": [[tap, rows] for tap, rows in taps.items()],
                "compared": compared, "kv_readbacks": {phase: len(held) for phase, held in readbacks.items()},
                "kv_planes": planes, "kv_phases": phases, "raw_rows": raw, "storage_errors": storage,
                "insufficient": [f"insufficient evidence: no {what} were compared" for what in insufficient]}


def describe_taps(result):
    """One line naming a taps comparison's first difference."""
    d = result["first_difference"]
    where = f"{d['phase']} readback ending at {d['end']}, " if "phase" in d else ""
    return (f"the native and comparison captures differ at {d['tap']} rank {d['rank']} ({where}request "
            f"{d['request']} position {d['position']}: {d['difference']}); {result['differing_rows']} rows of "
            f"{len(result['differing_taps'])} taps differ")


def compare_taps(native, comparison, native_planes, experimental=False):
    """The result for two tap captures (directories): see the module
    docstring."""
    (native_dir, native), (comparison_dir, comparison) = (tap_binding.recorded(path) for path in (native, comparison))
    served, errors = capture_identities(native, comparison, native_planes, experimental)
    errors += request_errors(native, comparison)
    result = {"kv_cache_dtypes": served["kv_cache_dtypes"], "package_identities": served["package_identities"],
              "requests": [record["name"] for record in native["requests"]],
              "servers": {"native": native["server"], "comparison": comparison["server"]}}
    if any(served["markers"]):
        result |= {experimental_determinism.MARKER: dict(zip(SIDES, served["markers"])), "qualified": False}
    if not errors:
        step_errors, compared = compare_steps(native_dir, native, comparison_dir, comparison)
        errors += step_errors
        if compared is not None:
            result |= compared
            if not compared["equal"]:
                errors.append(describe_taps(compared))
            errors += compared["storage_errors"] + compared["insufficient"]
    return dict(result, passed=not errors, errors=errors)


def taps_command(args):
    try:
        result = compare_taps(args.native, args.comparison, json.loads(args.native_planes.read_text()),
                              args.experimental)
    except (OSError, ValueError, KeyError, SystemExit) as error:
        result = {"passed": False, "errors": [str(error)]}
    servers = result.pop("servers", None)
    # Compares saved captures. No timed work, so no host load samples: the
    # receipt's timing reads informational.
    receipt("kv-layout-equality-taps", {"captures": servers}, dict(result, native=str(args.native),
                                                                    comparison=str(args.comparison)))
    print(json.dumps({key: result.get(key) for key in ("passed", "errors", "first_difference")}))
    raise SystemExit(0 if result["passed"] else 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("command", choices=("compare", "taps"))
    parser.add_argument("native", type=Path,
                        help="the record (taps: the tap capture) of the package's preferred KV layout")
    parser.add_argument("comparison", type=Path, help="the record (taps: the tap capture) of the comparison KV layout")
    parser.add_argument("--native-planes", type=Path,
                        help="taps: a JSON file of the native layout's kv_cache_planes (NATIVE_PLANES)")
    parser.add_argument("--experimental", action="store_true",
                        help="taps: compare two experimental captures under sealed pins, never qualification")
    args = parser.parse_args()
    if args.command == "taps":
        if args.native_planes is None:
            parser.error("taps needs --native-planes")
        taps_command(args)
    if args.native_planes is not None or args.experimental:
        parser.error("--native-planes and --experimental apply to taps")
    records = []
    try:
        records = [json.loads(path.read_text()) for path in (args.native, args.comparison)]
        result = compare(*records)
    except (OSError, ValueError, KeyError, SystemExit) as error:
        result = {"passed": False, "errors": [str(error)]}
    servers = {side: (value.get("server") if isinstance(value, dict) else None)
               for side, value in zip(("native", "comparison"), records)}
    # Compares saved records. No timed work, so no host load samples: the
    # receipt's timing reads informational.
    receipt("kv-layout-equality", {"records": servers}, dict(result, native=str(args.native),
                                                             comparison=str(args.comparison)))
    print(json.dumps({key: result.get(key) for key in ("passed", "errors", "first_difference")}))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
