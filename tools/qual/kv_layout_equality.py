#!/usr/bin/env python3
"""Served-bits equality of two KV storage layouts of the same values.

A model whose comparison KV layout stores the same values as its served one
(BF16 pages holding exactly what the native formats decode to) must serve
the same bits from both. Bitwise equality of native against BF16 storage is
required in the model's BF16 comparison configuration (kv-bf16) on the
determinism records, on forced records (`compare`) and on the decoded
KV-state taps of a diagnostic-taps build (`taps`). The BF16 layout matches
the native layout's stored values and production prefill partials, so the
comparison isolates KV storage; every route only the native layout takes
(the kernels reading its packed codes and scales, and each fast route a
package admits at its native layout alone) is compared as served against
the BF16 layout's admitted paths. Production accuracy is qualified
separately, by the default configuration's teacher, chat, long_context,
multimodal, verify_exact pairs, greedy AR versus DSpark and determinism
gates.

The correctness runner records, on servers of one configuration that differ
only in `--kv-cache-dtype`, the determinism request set
(tools/qual/determinism.py: C1 greedy, 48 output tokens with top-5
logprobs, a long prompt over two prefill chunks, token-ID sends and prefix
hits) on the configuration's own server (`spec`, its comparison layout) and
a `kv-native` server, which serves `auto`, the package's preferred layout;
and a forced record of each layout (verify_exact.py record --k 0
--require-row-shapes --reference long_context, on the forced `prefix` arm
and a `kv-native-forced` arm), which teacher-forces the exact reference's
texts and the pinned long-context text: for a long_128k reference one
request of a 130,048-token prompt and 1,024 forced outputs, the first from
the prompt's last prefill row and the other 1,023 from decode rows, which
each record must hold exactly (row kinds as forced_expectation derives
them), and which no teacher capture can stand in for since it scores
prefill rows only. The
runner adds RUNNER_FORCED_REFERENCES to the configuration's forcing file.
`compare` requires every request's text, finish reason, token IDs, tokens
and selected and top logprobs to be bit-identical across the determinism
records, naming the first differing request, field and position; and the
forced records to hold the same positions, those the gate's pinned
references define, with bit-identical logprobs and top-5 at every forced
position and bit-identical prefill rows, naming the first differing text
and position. Every row must be complete and typed before any comparison:
a forced row holds a finite float logprob, its row kind and a top-5 map of
1 to 6 string labels of finite float logprobs, and a prefill row exactly a
finite float logprob and its top-1 token ID. The map holds 5 entries, or 6
when the forced token ranks below them, and fewer only when tokens share a
label: the completions API labels top logprobs by decoded text, and every
incomplete UTF-8 byte token decodes to "", so their entries merge.

Both pairs must name one checkpoint, metadata, source commit, bundle,
version, served model, effective profile and complete `model_options` map,
compared with their JSON types (the determinism records from one run). The
package identity may differ with the KV layout; both are reported. The
layout fields are validated by `layout_errors` against NATIVE_PLANES. Each
forced record must be live-bound to the gate's checkpoint pin and equal its
layout's determinism record in every identity, option and profile field,
its package included, but FORCED_ARM_FIELDS: its own process's startup
nonce (forcing reference tokens reports nothing in /health).
Presence is part of equality: a compared field one side holds, even as
null, and the other lacks is a difference.

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
must be present in both and equal except the startup nonce, the package
identity (reported; it may differ) and the layout fields, which
`layout_errors` validates against the declared native planes
(--native-planes, a JSON list: NATIVE_PLANES).
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

    python3 tools/qual/kv_layout_equality.py compare NATIVE_RECORD COMPARISON_RECORD \\
        --native-forced NATIVE_FORCED --comparison-forced COMPARISON_FORCED
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

import hashlib

from common import bound_live, receipt
import determinism
import experimental_determinism
from prefix_exact import serving_identity
import tap_binding
import verify_exact
# tap_binding puts the repository root on the path.
from tools.model_sdk import diagnostic as wire

_RECORD = ["record", "--url", "{url}", "--model", "{name}", "--timeout", "120", "--run-id", "{run_id}",
           "--server-log", "{server_log}", "{image}"]
# The pinned references whose texts the forced records force beside the
# exact reference's (verify_exact.py --reference), and the runner adds to
# the configuration's forcing file: the long-context reference, whose forced
# decode continues its long history.
RUNNER_FORCED_REFERENCES = ["long_context"]
# A forced record, as release_preview.py RECORDS["verify_ar"] takes one.
_FORCED = ["record", "--url", "{url}", "--model", "{name}", "--k", "0", "--require-row-shapes", "--server-log",
           "{server_log}", *(word for name in RUNNER_FORCED_REFERENCES for word in ("--reference", name))]
# The correctness runner's arms and records (docs/check-kinds.md#kv-layout-equality):
# the kv-native servers replace the configuration's --kv-cache-dtype with
# auto, kv-native-forced also forcing the reference tokens as
# release_preview.py FORCED does; the configuration's own server records
# the comparison layout's determinism, and its forced prefix arm the
# comparison layout's forced record.
RUNNER_ARMS = {"kv-native": ["--kv-cache-dtype", "auto"],
               "kv-native-forced": ["--kv-cache-dtype", "auto", "--debug-reference-tokens", "{reference_tokens}"]}
RUNNER_RECORDS = {"kv_layout_native": ("kv-native", "determinism.py", [*_RECORD, "--arm", "kv-native"]),
                  "kv_layout_comparison": ("spec", "determinism.py", [*_RECORD, "--arm", "kv-comparison"]),
                  "kv_layout_native_forced": ("kv-native-forced", "verify_exact.py",
                                              [*_FORCED, "--arm", "kv-native-forced"]),
                  "kv_layout_comparison_forced": ("prefix", "verify_exact.py",
                                                  [*_FORCED, "--arm", "kv-comparison-forced"])}
RUNNER_CHECK_ARM = "offline"
# The native layout's kv_cache_planes as its package declares them, which the
# native record must report: none for a single-code element. A model's
# declaration states its own (constants.kv_layout_equality.NATIVE_PLANES).
NATIVE_PLANES = []
# The serving identity fields that follow the KV layout.
LAYOUT_FIELDS = ("package_identity",)
# The serving identity fields in which a forced record may differ from the
# determinism record of its layout. Each forced arm serves that record's arm
# arguments with --debug-reference-tokens added, in another process, which
# draws its own startup nonce (each record is bound live to its own, and
# serving_identity leaves it out). /health reports no reference-tokens option
# or digest, so every other identity, option and profile field, the package
# identity included, must be equal.
FORCED_ARM_FIELDS = ("startup_nonce",)
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


def differing_keys(left, right, excluded=()):
    """The keys, sorted and `excluded` aside, at which two JSON objects
    differ, each with the index of the object that alone holds it, or None:
    values unequal with their JSON types differ, and so does a key one
    object holds, even as null, and the other lacks (presence is part of
    equality)."""
    return [(key, None if key in left and key in right else int(key in right))
            for key in sorted((left.keys() | right.keys()) - set(excluded))
            if key not in left or key not in right or not same(left[key], right[key])]


def picked(value, keys):
    """The fields among `keys` that the JSON object `value` holds."""
    return {key: value[key] for key in keys if key in value}


def presence(names, alone):
    """What a difference adds when one of the two holders `names` alone
    holds the field (`differing_keys`)."""
    return "" if alone is None else f": only the {names[alone]} holds it"


def served_errors(left, right, what):
    """Why two serving identities do not name one build, profile and option
    map in two KV layouts: every identity and profile field, `model_options`
    included, is present in both and equal with its JSON type but the
    package identity (which may follow the layout) and the layout fields,
    which `layout_errors` validates against NATIVE_PLANES; each reports a
    complete option map."""
    errors = []
    names = [f"{side} {what[:-1]}" for side in SIDES]
    for side, identity in zip(SIDES, (left, right)):
        if not (isinstance(identity.get("model_options"), dict) and identity["model_options"]
                and type(identity.get("model_options_qualified")) is bool):
            errors.append(f"the {side} {what[:-1]} reports no complete model_options and model_options_qualified")
    errors += [f"the {what} differ in serving identity {key}{presence(names, alone)}"
               for key, alone in differing_keys(left, right, {"profile", *LAYOUT_FIELDS})]
    profiles = left["profile"], right["profile"]
    errors += [f"the {what} differ in profile {key}{presence(names, alone)}"
               for key, alone in differing_keys(*profiles, LAYOUT_PROFILE)]
    return errors + layout_errors(*profiles, NATIVE_PLANES)


def identities(native, comparison):
    """The two records' serving identities, refused unless they come from
    one run and name one build, profile and option map that differ only in
    KV storage (`served_errors`)."""
    left, right = serving_identity(native), serving_identity(comparison)
    errors = []
    if native.get("run_id") != comparison.get("run_id") or not native.get("run_id"):
        errors.append("the records come from different runs")
    errors += served_errors(left, right, "records")
    dtypes = [identity["profile"]["kv_cache_dtype"] for identity in (left, right)]
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


def forced_expectation():
    """(exact reference digest, {forced reference: digest}, [(text, forced
    positions, their row kinds, prefill positions)], {forced reference:
    [each text's requests and row counts]}) the gate's pinned references
    define for a forced record (verify_exact.py record --k 0 --reference
    ...): each text of the exact reference and of RUNNER_FORCED_REFERENCES,
    every position forcing reaches from its score_from (agreement read over
    every forced text), one request per forced run, whose first output
    comes from its prompt's last prefill row and every later one from a
    decode row, and every scored prefill row. A long-context text of
    131,072 tokens scored from 130,048 is one request of a 130,048-token
    prompt and 1,024 forced outputs: 1 prefill and 1,023 decode rows."""
    path = verify_exact.exact_path(None)
    exact = verify_exact.load_exact(path)
    named = verify_exact.named_references(RUNNER_FORCED_REFERENCES)
    texts = verify_exact.forced_texts(exact, named)
    expected, requests = [], {}
    for text in texts:
        ids = text["prompt_token_ids"]
        agreed = verify_exact.agreed_continuations(texts, ids)
        runs = list(verify_exact.forced_runs(agreed, text["score_from"], len(ids)))
        forced = [start + offset for start, count in runs for offset in range(count)]
        kinds = [label["row"] for _, count in runs for label in verify_exact.row_labels(0, count)]
        expected.append((text["name"], forced, kinds, list(range(text["score_from"], len(ids)))))
        requests[text["name"]] = {"text": text["name"],
                                  "requests": [{"prompt_tokens": start, "forced_outputs": count} for start, count in runs],
                                  "prefill_rows": kinds.count("prefill"), "decode_rows": kinds.count("decode")}
    return (hashlib.sha256(path.read_bytes()).hexdigest(), {name: digest for name, (digest, _) in named.items()},
            expected, {name: [requests[text["name"]] for text in value["texts"]] for name, (_, value) in named.items()})


def finite(value):
    """Whether `value` is a finite float, never a bool or an int."""
    return type(value) is float and math.isfinite(value)


def forced_row_fault(row, width):
    """Why a K=0 forced record's row is not complete and typed, or None: it
    holds exactly the fields verify_exact.py writes, a finite logprob, its
    row kind and the completions API's top logprobs, a map of string labels
    to finite logprobs. That map holds `width` entries, or `width` + 1 when
    the forced token ranks below them (vLLM's order), and fewer when tokens
    share a label: every incomplete UTF-8 byte token decodes to "", so
    their entries merge, a limit of the completions API's labels."""
    if not isinstance(row, dict) or set(row) != {"logprob", "top", "row"}:
        return f"holds fields {sorted(row) if isinstance(row, dict) else type(row).__name__}, not logprob, top and row"
    if not finite(row["logprob"]):
        return f"has logprob {row['logprob']!r}, not a finite float"
    top = row["top"]
    if not (isinstance(top, dict) and 1 <= len(top) <= width + 1
            and all(isinstance(label, str) and finite(value) for label, value in top.items())):
        size = f"{len(top)} entries" if isinstance(top, dict) else type(top).__name__
        return f"has a top map of {size}, not 1 to {width + 1} labels of finite floats"
    if row["row"] not in ("prefill", "decode"):
        return f"is labelled {row['row']!r}, not prefill or decode"
    return None


def prefill_row_fault(row):
    """Why a forced record's prefill row (verify_exact.py prefill_capture) is
    not complete and typed, or None: exactly a finite logprob and the row's
    top-1 token ID."""
    if not isinstance(row, dict) or set(row) != {"logprob", "top1"}:
        return f"holds fields {sorted(row) if isinstance(row, dict) else type(row).__name__}, not logprob and top1"
    if not finite(row["logprob"]):
        return f"has logprob {row['logprob']!r}, not a finite float"
    if not (type(row["top1"]) is int and row["top1"] >= 0):
        return f"has top-1 {row['top1']!r}, not a token ID"
    return None


def forced_rows(record, side):
    """({text: {position: forced row}}, {text: {position: prefill row}},
    errors) of one forced record: a passing, live-bound K=0 record of one
    phase per text with top_logprobs verify_exact.TOP_LOGPROBS, whose every
    forced and prefill row is complete and typed (forced_row_fault,
    prefill_row_fault)."""
    errors = []
    if not (record.get("passed") is True and type(record.get("k")) is int and record["k"] == 0
            and not record.get("pooled_decode")):
        errors.append(f"the {side} forced record is not a passing K=0 record")
    if not bound_live(record.get("server") or {}):
        errors.append(f"the {side} forced record is not bound live to the gate's checkpoint pin")
    width = record.get("top_logprobs")
    if not (type(width) is int and width == verify_exact.TOP_LOGPROBS):
        errors.append(f"the {side} forced record declares top_logprobs {width!r}, not {verify_exact.TOP_LOGPROBS}")
        width = verify_exact.TOP_LOGPROBS
    forced, prefill = {}, {}
    for text in record.get("texts") or []:
        passes = text.get("passes") or []
        if len(passes) != 1 or not same(passes[0].get("phase"), 0) or text.get("name") in forced:
            errors.append(f"the {side} forced record does not hold one phase-0 pass of text {text.get('name')}")
            continue
        forced[text["name"]] = {int(position): row for position, row in passes[0]["positions"].items()}
    for name, rows in (record.get("prefill") or {}).items():
        prefill[name] = {int(position): row for position, row in rows.items()}
    for kind, held, fault in (("forced", forced, lambda row: forced_row_fault(row, width)),
                              ("prefill", prefill, prefill_row_fault)):
        faults = [(name, position, why) for name, rows in held.items() for position, row in sorted(rows.items())
                  if (why := fault(row)) is not None]
        if faults:
            name, position, why = faults[0]
            errors.append(f"the {side} forced record's {kind} row of {name} position {position} {why}; "
                          f"{len(faults)} {kind} rows are not complete")
    return forced, prefill, errors


def first_positions(left, right, limit=3):
    """Up to `limit` positions one set holds and the other lacks, each way."""
    return sorted(left - right)[:limit], sorted(right - left)[:limit]


def compare_forced(native, comparison, served):
    """The forced records' result and errors: both live-bound records of the
    gate's pinned references, forcing every position those define, each
    served as its layout's determinism record in `served` was (every field
    but FORCED_ARM_FIELDS), and bit-identical at every forced row's logprob
    and top-5 and every prefill row."""
    errors = []
    identities_of = [serving_identity(record) for record in (native, comparison)]
    errors += served_errors(*identities_of, "forced records")
    names = ("forced record", "determinism record")
    for side, identity in zip(SIDES, identities_of):
        fields = [(key, alone) for key, alone in differing_keys(identity, served[side],
                                                                {"profile", *FORCED_ARM_FIELDS})]
        fields += [(f"profile {key}", alone) for key, alone in differing_keys(identity["profile"],
                                                                              served[side]["profile"])]
        errors += [f"the {side} forced record differs from its determinism record in {key}{presence(names, alone)}"
                   for key, alone in fields]
    digest, references, expected, requests = forced_expectation()
    result = {"kv_cache_dtypes": [identity["profile"].get("kv_cache_dtype") for identity in identities_of],
              "references": references, "texts": len(expected), "forced_references": requests}
    rows = []
    for side, record in zip(SIDES, (native, comparison)):
        if record.get("reference_sha256") != digest:
            errors.append(f"the {side} forced record scores another exact reference than the gate pins")
        if not same(record.get("references", {}), references):
            errors.append(f"the {side} forced record forces references {sorted(record.get('references') or {})}, "
                          f"not the pinned {sorted(references)}")
        forced, prefill, refused = forced_rows(record, side)
        errors += refused
        rows.append((forced, prefill))
    if errors:
        return result, errors
    for name, positions, _, scored in expected:
        held = [({*forced.get(name, {})}, {*prefill.get(name, {})}) for forced, prefill in rows]
        if held[0] != held[1]:
            for kind, index in (("forced", 0), ("prefill", 1)):
                if held[0][index] != held[1][index]:
                    native_only, comparison_only = first_positions(held[0][index], held[1][index])
                    errors.append(f"the forced records hold other {kind} positions of {name}: the native record "
                                  f"alone holds {native_only}, the comparison record {comparison_only}")
            continue
        for kind, index, wanted in (("forced", 0, set(positions)), ("prefill", 1, set(scored))):
            if held[0][index] != wanted:
                missing, extra = first_positions(wanted, held[0][index])
                errors.append(f"the forced records hold other {kind} positions of {name} than its references "
                              f"define: missing {missing}, extra {extra}")
    named = {name for forced, _ in rows for name in forced} | {name for _, prefill in rows for name in prefill}
    if unexpected := sorted(named - {name for name, _, _, _ in expected}):
        errors.append(f"the forced records hold texts no pinned reference names: {unexpected}")
    if errors:
        return result, errors
    # Each forced position's row kind is the one its request gives it: a
    # record that scored a text in other requests, or fewer decode rows, is
    # refused.
    for name, positions, kinds, _ in expected:
        for side, (forced, _) in zip(SIDES, rows):
            held = [forced[name][position]["row"] for position in positions]
            if held.count("prefill") != kinds.count("prefill") or held.count("decode") != kinds.count("decode"):
                errors.append(f"the {side} forced record scores {name} with {held.count('prefill')} prefill and "
                              f"{held.count('decode')} decode rows, not the {kinds.count('prefill')} and "
                              f"{kinds.count('decode')} its references define")
            elif (at := next((index for index, kind in enumerate(kinds) if held[index] != kind), None)) is not None:
                errors.append(f"the {side} forced record labels {name} position {positions[at]} {held[at]}, not "
                              f"{kinds[at]}")
    if errors:
        return result, errors
    differences, first = 0, None
    for name, positions, _, scored in expected:
        cells = [(position, "forced", 0, field) for position in positions for field in ("logprob", "top")]
        cells += [(position, "prefill", 1, field) for position in scored for field in ("logprob", "top1")]
        differing = set()
        for position, kind, index, field in sorted(cells, key=lambda cell: (cell[0], cell[2])):
            values = [side[index][name][position].get(field) for side in rows]
            if not same(*values):
                differing.add(position)
                if first is None:
                    first = {"text": name, "position": position, "row": kind, "field": field,
                             "native": values[0], "comparison": values[1]}
        differences += len(differing)
    result |= {"positions": sum(len(positions) for _, positions, _, _ in expected),
               "prefill_rows": sum(len(scored) for _, _, _, scored in expected),
               "equal": first is None, "first_difference": first, "differing_positions": differences}
    if first is not None:
        errors.append(f"the forced records differ at {first['text']} position {first['position']} "
                      f"({first['row']} {first['field']}: {first['native']!r} vs {first['comparison']!r}); "
                      f"{differences} positions differ")
    return result, errors


def compare(native, comparison, native_forced, comparison_forced):
    """The check's result for two determinism records and two forced records
    of the native and comparison layouts."""
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
    result["forced"], forced_errors = compare_forced(native_forced, comparison_forced, served)
    return dict(result, passed=not errors and not forced_errors, errors=errors + forced_errors)


def capture_identities(native, comparison, native_planes, experimental=False):
    """The two capture manifests' layouts and experimental pins, and the
    errors that refuse the pair: see the module docstring."""
    servers = native["server"], comparison["server"]
    names = [f"{side} capture" for side in SIDES]
    excluded = {"profile", "startup_nonce", *LAYOUT_FIELDS, *tap_binding.OPTIONS}
    errors = [f"the captures differ in serving identity {key}{presence(names, alone)}"
              for key, alone in differing_keys(*servers, excluded)]
    profiles = servers[0]["profile"], servers[1]["profile"]
    errors += [f"the captures differ in profile {key}{presence(names, alone)}"
               for key, alone in differing_keys(*profiles, LAYOUT_PROFILE)]
    errors += layout_errors(*profiles, native_planes)
    errors += [f"the captures differ in {key}{presence(names, alone)}"
               for key, alone in differing_keys(*(picked(server, tap_binding.OPTIONS) for server in servers))]
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
        names = [f"{side} capture" for side in SIDES]
        for left, right in zip(native["requests"], comparison["requests"]):
            errors += [f"the captures differ in request {left['name']} {key}{presence(names, alone)}"
                       for key, alone in differing_keys(picked(left, REQUEST_FIELDS), picked(right, REQUEST_FIELDS))]
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
    parser.add_argument("--native-forced", type=Path,
                        help="compare: the native layout's forced record (verify_exact.py record)")
    parser.add_argument("--comparison-forced", type=Path, help="compare: the comparison layout's forced record")
    parser.add_argument("--native-planes", type=Path,
                        help="taps: a JSON file of the native layout's kv_cache_planes (NATIVE_PLANES)")
    parser.add_argument("--experimental", action="store_true",
                        help="taps: compare two experimental captures under sealed pins, never qualification")
    args = parser.parse_args()
    if args.command == "taps":
        if args.native_planes is None:
            parser.error("taps needs --native-planes")
        if args.native_forced is not None or args.comparison_forced is not None:
            parser.error("--native-forced and --comparison-forced apply to compare")
        taps_command(args)
    if args.native_planes is not None or args.experimental:
        parser.error("--native-planes and --experimental apply to taps")
    if args.native_forced is None or args.comparison_forced is None:
        parser.error("compare needs --native-forced and --comparison-forced")
    paths = (args.native, args.comparison, args.native_forced, args.comparison_forced)
    records = []
    try:
        records = [json.loads(path.read_text()) for path in paths]
        result = compare(*records)
    except (OSError, ValueError, KeyError, SystemExit) as error:
        result = {"passed": False, "errors": [str(error)]}
    servers = [value.get("server") if isinstance(value, dict) else None for value in records] + [None] * 4
    # Compares saved records. No timed work, so no host load samples: the
    # receipt's timing reads informational.
    receipt("kv-layout-equality", {"records": dict(zip(SIDES, servers[:2])),
                                   "forced_records": dict(zip(SIDES, servers[2:4]))},
            dict(result, native=str(args.native), comparison=str(args.comparison),
                 native_forced=str(args.native_forced), comparison_forced=str(args.comparison_forced)))
    print(json.dumps({key: result.get(key) for key in ("passed", "errors", "first_difference")}))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
