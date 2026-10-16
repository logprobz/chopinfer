"""The declaration-driven encoder parity gate.

`taps --run RUN` is the `encoder_parity` check (models/mimo26/qual.json and
models/qwen38/qual.json): the serving engine's encoder taps against the
generic encoder reference (tools/refrun/encoder.py) on the GPU runner,
judged by tools/qual/tapdiff.py. RUN holds `parity.json` (schema
chopinfer.encoder-parity-run.v1: `model`; `engine`, `engine`, the steps of
the tap capture RUN holds; `references`, one encoder reference output per
encoder declaration, each with taps in the exact, policy and
reference-bf16 modes; `workload`, the reference runner's workload;
`source`, the commit both ran) and a tools/qual/tap_binding.py capture
(tap-capture.json): each workload sequence sent once, under its name, to a
diagnostic-taps server. The capture is admitted against the live candidate
(--url, tap_binding.admit): the candidate must be the process that wrote
the steps, every identity field equal, and each step must name that
process and its request; its records name each engine request's
sequence. An unbound capture is insufficient evidence. The gate:

- vision rows (`enc.vision.out`), per item, by direction and scale. MiMo's
  block 27 writes a massive activation into one channel that every BF16
  implementation misses by a few ulps, and the merger's RMSNorm turns that
  into a whole-row scale error (docs/numerics.md, MiMo vision row scale).
  Direction: d = |engine / |engine| - exact / |exact|| per row against the
  same of reference-bf16; every row d <= PARITY_RATIO * max(floor,
  ROW_FLOOR), and the median d / (PARITY_RATIO * floor) at most 1. Scale,
  with s the item's largest row scale error | |engine| / |exact| - 1 |, by
  the precision policy the encoder declaration names (the engine runs it,
  and the policy run emulates it):
  - an F32 policy (F32_POLICIES): s at most the policy run's own largest
    row scale error over the item, at least 0.1% and rounded up to 0.1%
    (tapdiff's POLICY_SCALE_STEP). The vendor BF16 run's far larger scale
    errors are reported and set no limit.
  - a BF16 policy (BF16_POLICIES): s at most the largest of
    reference-bf16's and the policy run's, rounded up to 0.5%.
  The receipt lists each item's three scale errors and the rule.
- MiMo's audio rows (`enc.audio.out`): e = |engine - exact| / |exact| per
  row against floor = |reference-bf16 - exact| / |exact|, every row and the
  median as above. An audio row merges 4 codes of 20 stages and is judged
  only where all of the engine's codes it merges equal the exact run's: a
  flipped code moves its row far from any floor and is the codes rule's
  (below). The receipt counts the rows left out (`code_flipped_rows`); at
  BF16 nearly every row has a flip in some stage.
- MiMo's RVQ codes (`enc.audio.x.codes`), compared code by code with the
  exact run's: the engine's agreement over all codes and over the first
  stage must each be at least the vendor BF16 run's (reference-bf16, the
  vendor's BF16 quantizer distance) less CODES_MARGIN, measured on the same
  inputs in the same run. The receipt reports the engine's, the vendor BF16
  run's and MiMo's legacy policy run's agreement side by side.
- the receipt reports p50, p90 and p100 of e / floor (d / floor for
  vision).

Every one of the model's taps must hold compared rows; tapdiff must find
no tap outside its rule and no incomplete or incompatible capture.

"""

import argparse
import ast
from array import array
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from common import startup, ROOT, qualification_config, receipt
import declaration
import tap_binding
import encoder_declarations

NPY_TYPES = {"<f4": "f", "<i8": "q", "<i4": "i"}
# The encoder_parity check (`taps`). Rows: tapdiff's ratio over each row's
# BF16 floor, raised to ROW_FLOOR (twice the BF16 unit roundoff, the error
# of storing a row in BF16) where the reference-bf16 row nearly equals the
# exact one. Codes: MiMo's legacy BF16 policy run agrees with 71.2% of the
# exact codes and the vendor BF16 run with 73.2% (the GPU encoder
# reference of a speech clip since withdrawn, 2026-10-01); the margin leaves
# room for that difference.
PARITY_RATIO = 3.0
ROW_FLOOR = 2.0 ** -8
CODES_MARGIN = 0.03
PARITY_RUN = "chopinfer.encoder-parity-run.v1"
# Reference modes of tools/refrun/encoder.py: the gate reference, the
# vendor BF16 spread and the model's declared policy.
EXACT, VENDOR, POLICY = "exact", "reference-bf16", "policy"
# Encoder precision policies by the scale rule they take (`taps`).
F32_POLICIES = ("f32-pairs.v1",)
BF16_POLICIES = ("bf16-legacy.v1",)


# Models opt into the producer by pinning its reference inputs in their
# own configuration file. Existing externally produced runs keep their arm.
def RUNNER_ARMS(model):
    import encoder_capture
    return ({"encoders": ["--modalities", "all", "--prefix-cache", "off", "--speculative-tokens", "0"]}
            if model and encoder_capture.configured(model) else {})


def RUNNER_RECORDS(model):
    return ({"encoder_parity_run": ("encoders", "encoder_capture.py", ["record", "--url", "{url}",
             "--server-log", "{server_log}", "--taps-dir", "{taps_dir}", "--run", "{run_dir}"])}
            if RUNNER_ARMS(model) else {})


def RUNNER_CHECK_ARM(model):
    return "encoders" if RUNNER_ARMS(model) else "spec"


RUNNER_FEATURES = {"encoders": ["diagnostic-taps"]}
def RUNNER_ENVIRONMENT(model):
    if not RUNNER_ARMS(model):
        return {}
    rules = encoder_declarations.parity_rules(encoder_declarations.for_model(model))
    sites = [*rules["rows"], *([rules["codes"]] if rules["codes"] else [])]
    return {"encoders": {"CHOPINFER_TAPS_DIR": "{taps_dir}", "CHOPINFER_TAPS_SITES": ",".join(sites),
                         "CHOPINFER_REQUEST_TRACE": "1"}}


RUNNER_DIRECTORY_INPUTS = ["encoder_parity_run"]


def precision(reference):
    """The precision policy of an encoder reference output: the policy its
    run's declaration names, which must be the working tree's file and the
    one its policy mode ran, and the taps the output holds."""
    run = json.loads((reference / "run.json").read_text())
    declared = run["declaration"]
    path = ROOT / declared["path"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != declared["sha256"]:
        raise ValueError(f"{reference}: {declared['path']} differs from the declaration it ran")
    policy = json.loads(path.read_text())["precision"]
    ran = json.loads((reference / f"{POLICY}.json").read_text())["mode"]["precision"]
    if ran != policy:
        raise ValueError(f"{reference}: its policy mode ran {ran}, its declaration names {policy}")
    return policy, {tap for mode in run["modes"].values() for tap in mode.get("taps", ())}


def scale_rule(references, taps):
    """`f32-policy` or `bf16-spread`: the scale rule of the references that
    hold `taps`, which must name one kind of policy."""
    kinds = set()
    for reference in references:
        policy, held = precision(reference)
        if not held & set(taps):
            continue
        if policy in F32_POLICIES:
            kinds.add("f32-policy")
        elif policy in BF16_POLICIES:
            kinds.add("bf16-spread")
        else:
            raise ValueError(f"{reference}: precision policy {policy} has no scale rule")
    if len(kinds) != 1:
        raise ValueError(f"the references of {sorted(taps)} name {sorted(kinds) or 'no'} scale rules")
    return kinds.pop()


def read_npy(path):
    """Read a little-endian, C-order float32, int64 or int32 npy file."""
    data = Path(path).read_bytes()
    if data[:6] != b"\x93NUMPY":
        raise ValueError(f"{path}: not an npy file")
    major = data[6]
    if major == 1:
        length, start = int.from_bytes(data[8:10], "little"), 10
    elif major in (2, 3):
        length, start = int.from_bytes(data[8:12], "little"), 12
    else:
        raise ValueError(f"{path}: unsupported npy version {major}")
    header = ast.literal_eval(data[start:start + length].decode("latin1"))
    code = NPY_TYPES.get(header["descr"])
    if code is None or header["fortran_order"]:
        raise ValueError(f"{path}: unsupported dtype {header['descr']} or Fortran order")
    shape = tuple(header["shape"])
    values = array(code)
    if values.itemsize != int(header["descr"][2:]):
        raise ValueError(f"{path}: platform array type {code} has the wrong width")
    values.frombytes(data[start + length:])
    if len(values) != math.prod(shape):
        raise ValueError(f"{path}: {len(values)} values for shape {shape}")
    if sys.byteorder != "little":
        values.byteswap()
    return shape, values


def native(values):
    if sys.byteorder != "little":
        values.byteswap()
    return values


def read_bf16(path):
    halves = native(array("H", Path(path).read_bytes()))
    return native(array("f", native(array("I", (h << 16 for h in halves))).tobytes()))


def read_i32(path):
    return native(array("i", Path(path).read_bytes()))


def bf16_bits(values):
    """Round float32 values to BF16 bit patterns, nearest even, NaN kept quiet."""
    words = native(array("I", native(array("f", values)).tobytes()))
    out = array("H")
    for word in words:
        if word & 0x7FFFFFFF > 0x7F800000:
            out.append((word >> 16) | 0x40)
        else:
            out.append((word + 0x7FFF + ((word >> 16) & 1)) >> 16)
    return out


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 1.0 if na == nb else 0.0
    return dot / (na * nb)


def row_cosines(reference, candidate, rows, width):
    return [cosine(reference[r * width:(r + 1) * width], candidate[r * width:(r + 1) * width])
            for r in range(rows)]


def merged_taps(references, label, into):
    """One tap directory of `label`'s steps of every reference output, as
    links renumbered in order."""
    into.mkdir(parents=True)
    count = 0
    for reference in references:
        steps = sorted((reference / "taps" / label).glob("step*"))
        if not steps:
            raise ValueError(f"{reference}: no {label} taps")
        for step in steps:
            (into / f"step{count:04d}").symlink_to(step.resolve(), target_is_directory=True)
            count += 1
    return into


def tapdiff(arguments, receipt_path):
    """tapdiff's exit status and receipt, run with the reference runner's
    Python (numpy), or this one when CHOPINFER_REFRUN_PYTHON is unset."""
    python = os.environ.get("CHOPINFER_REFRUN_PYTHON") or sys.executable
    result = subprocess.run([python, "-B", "-m", "tools.qual.tapdiff", *map(str, arguments),
                             "--json", str(receipt_path)], cwd=ROOT, text=True, capture_output=True)
    print(result.stdout, end="")
    print(result.stderr, end="", file=sys.stderr)
    report = json.loads(receipt_path.read_text()) if receipt_path.exists() else None
    return result.returncode, report, result.stderr.strip()


def encoder_steps(engine):
    """The engine's steps holding encoder rows: a medium's later prefill
    chunks only read rows of its encode's step, and with only encoder taps
    selected they capture nothing."""
    return [step.name for step in sorted(Path(engine).glob("step*"))
            if json.loads((step / "items.json").read_text()).get("encoder")]


def parity_tap(row):
    """A tap row of tapdiff's receipt as the parity receipt reports it."""
    out = {key: row[key] for key in ("tap", "status", "rows", "encoder_rows", "scale") if key in row}
    if "direction" in row:
        quantiles = row["direction"]["e_over_floor"] or {}
        out["direction"] = {"e_over_floor": {"p50": quantiles.get("median"), "p90": quantiles.get("p90"),
                                             "p100": quantiles.get("max")},
                            "every_row": row["direction"]["every_row"]}
    elif "e_over_floor" in row:
        quantiles = row["e_over_floor"] or {}
        out["e_over_floor"] = {"p50": quantiles.get("median"), "p90": quantiles.get("p90"),
                               "p100": quantiles.get("max")}
        out["every_row"] = row.get("every_row")
    for key in ("code_flipped_rows", "agreement", "stage0_agreement", "bf16_agreement", "bf16_stage0_agreement"):
        if key in row:
            out[key] = row[key]
    return out


def common_arguments(modes, workload):
    """tapdiff's references and the gate's thresholds."""
    return ["--f32", modes[EXACT], "--bf16", modes[VENDOR], "--workload", workload,
            "--ratio", PARITY_RATIO, "--row-floor", ROW_FLOOR, "--codes-margin", CODES_MARGIN]


def gate_arguments(engine, modes, workload, requests, spec, rule):
    """tapdiff's arguments for the gate over `engine`'s encoder steps,
    `requests` mapping each engine request to its workload sequence."""
    named = [x for request, name in requests.items() for x in ("--request", f"{request}={name}")]
    return ["--engine", engine, "--steps", ",".join(encoder_steps(engine)),
            *common_arguments(modes, workload), *named,
            *[x for rule in spec["every_row"] for x in ("--every-row", rule)],
            *[x for tap in spec["direction_scale"] for x in ("--direction-scale", tap)],
            "--scale-variant", f"policy={modes[POLICY]}",
            *(["--scale-policy", "policy"] if rule == "f32-policy" else [])]


def lab(args):
    """The gate's tapdiff rules over a lab capture of the encoder alone
    (each step's request named as its workload sequence), against reference
    outputs that may split the modes between runs. A lab capture is bound
    to no serving process, so this is evidence for the encoder, not the
    encoder_parity check."""
    spec = encoder_declarations.parity_rules(encoder_declarations.for_model(args.model))
    references = [Path(reference) for reference in args.reference]
    engine, workload = Path(args.engine), Path(args.workload)
    names = [sequence["name"] for sequence in json.loads(workload.read_text())["sequences"]]
    with tempfile.TemporaryDirectory() as scratch:
        scratch = Path(scratch)
        modes = {}
        for label in (EXACT, VENDOR, POLICY):
            held = [reference for reference in references if (reference / "taps" / label).is_dir()]
            if not held:
                raise ValueError(f"no reference holds {label} taps")
            modes[label] = merged_taps(held, label, scratch / label)
        rule = scale_rule([reference for reference in references if (reference / f"{POLICY}.json").exists()],
                          spec["direction_scale"])
        # tapdiff writes this run's receipt in the scratch directory, so an
        # earlier receipt at --json can never stand in for it.
        status, report, error = tapdiff(
            gate_arguments(engine, modes, workload, {name: name for name in names}, spec, rule),
            scratch / "receipt.json")
        if report is not None:
            Path(args.json).write_text(json.dumps(report, indent=2) + "\n")
        else:
            Path(args.json).unlink(missing_ok=True)
    errors = [] if status == 0 and report else [f"tapdiff exit {status}: {error}"]
    taps = {row["tap"]: row for row in (report or {}).get("taps", [])}
    for name in spec["rows"]:
        if not taps.get(name, {}).get("rows"):
            errors.append(f"{name}: no compared rows")
    if report and report["status"] != "match":
        errors.append(f"tapdiff {report['status']}: first failing tap {report['first']}")
    for line in errors:
        print(f"encoder_parity lab: {line}", file=sys.stderr)
    print(f"encoder_parity lab: {'match' if not errors else 'FAILED'} ({rule}, receipt {args.json})")
    return 0 if not errors else 1


def parity(args):
    """The encoder_parity check over a parity run directory."""
    run = Path(args.run)
    manifest_path = run / "parity.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != PARITY_RUN:
        raise ValueError(f"{manifest_path}: schema {manifest.get('schema')!r}, not {PARITY_RUN}")
    model = manifest.get("model")
    config = (qualification_config() if os.environ.get("CHOPINFER_QUAL_CONFIG")
              else declaration.load(declaration.path(model)))
    import encoder_capture
    if encoder_capture.configured(model):
        encoder_capture.admit(run, manifest, config)
    encoders = encoder_declarations.for_model(model)
    if os.environ.get("CHOPINFER_QUAL_CONFIG") and qualification_config()["model"] != model:
        raise ValueError(f"{manifest_path}: a {model} run under the {qualification_config()['model']} gate")
    # An encoder of media the declaration serves but does not qualify has no
    # required rows; the receipt names it with the declared reason.
    withdrawn = encoder_declarations.unqualified(encoders, config.get("capabilities"))
    unqualified = {value["path"].relative_to(ROOT).as_posix(): reason for value, reason in withdrawn}
    spec = encoder_declarations.parity_rules([value for value in encoders
                                              if all(value is not other for other, _ in withdrawn)])
    engine = run / manifest["engine"]
    references = [run / reference for reference in manifest["references"]]
    workload = run / manifest["workload"]
    errors, rows, codes = [], {}, None
    rule = scale_rule(references, spec["direction_scale"])
    try:
        if manifest["engine"] != tap_binding.STEPS:
            raise tap_binding.Unbound(f"{manifest_path}: the engine taps are not the capture's steps")
        config = (qualification_config() if os.environ.get("CHOPINFER_QUAL_CONFIG")
                  else declaration.load(declaration.path(model)))
        capture = tap_binding.admit(run, args.url, config)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"encoder_parity: unbound: {error}", file=sys.stderr)
        receipt("encoder-parity", {"run": str(run), "model": model}, {
            "passed": False, "errors": [f"unbound: {error}"], "taps": {}, "codes": None,
            "rule": None, "tapdiff_status": None, "unqualified": unqualified})
        return 1
    requests = {str(record["engine_request_id"]): record["name"] for record in capture["requests"]}
    with tempfile.TemporaryDirectory() as scratch:
        scratch = Path(scratch)
        modes = {label: merged_taps(references, label, scratch / label) for label in (EXACT, VENDOR, POLICY)}
        common = common_arguments(modes, workload)
        status, report, error = tapdiff(gate_arguments(engine, modes, workload, requests, spec, rule),
                                        scratch / "engine.json")
        if report is None:
            errors.append(f"tapdiff exit {status}: {error}")
        else:
            taps = {row["tap"]: row for row in report["taps"]}
            for name in (*spec["rows"], *filter(None, [spec["codes"]])):
                row = taps.get(name)
                if row is None or not row.get("rows"):
                    errors.append(f"{name}: no compared rows ({row['status'] if row else 'absent'})")
                    continue
                rows[name] = parity_tap(row)
            if report["status"] != "match":
                errors.append(f"tapdiff {report['status']}: first failing tap {report['first']}")
        if spec["codes"] in rows:
            # The model's own policy run, scored like the engine, for the report.
            names = [sequence["name"] for sequence in json.loads(workload.read_text())["sequences"]]
            _, legacy, error = tapdiff(["--engine", modes[POLICY], *common,
                                        *[x for name in names for x in ("--request", f"{name}={name}")]],
                                       scratch / "policy.json")
            legacy_codes = next((row for row in (legacy or {}).get("taps", []) if row["tap"] == spec["codes"]),
                                None)
            engine_codes = rows[spec["codes"]]
            codes = {"engine": {"agreement": engine_codes["agreement"],
                                "stage0_agreement": engine_codes["stage0_agreement"]},
                     "vendor_bf16": {"agreement": engine_codes.get("bf16_agreement"),
                                     "stage0_agreement": engine_codes.get("bf16_stage0_agreement")},
                     "legacy_policy": legacy_codes and {"agreement": legacy_codes.get("agreement"),
                                                        "stage0_agreement": legacy_codes.get("stage0_agreement")},
                     "margin": CODES_MARGIN}
        identity = {"run": str(run), "model": model, "parity_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                    "source": manifest.get("source"), "server": capture["server"], "requests": requests,
                    "capture_sha256": hashlib.sha256((run / tap_binding.MANIFEST).read_bytes()).hexdigest(),
                    "tapdiff_inputs": report and report["inputs"]}
    passed = not errors
    for line in errors:
        print(f"encoder_parity: {line}", file=sys.stderr)
    receipt("encoder-parity", identity, {
        "passed": passed, "errors": errors, "rows": sum(row["rows"] for row in rows.values()),
        "taps": rows, "codes": codes, "unqualified": unqualified,
        "rule": {"ratio": PARITY_RATIO, "row_floor": ROW_FLOOR, "codes_margin": CODES_MARGIN, "scale": rule,
                 "reference": EXACT, "spread": VENDOR},
        "tapdiff_status": report and report["status"]})
    return 0 if passed else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    gate = commands.add_parser("taps", help="the encoder_parity check over a parity run")
    gate.add_argument("--run", required=True, help="directory holding parity.json")
    gate.add_argument("--url", help="the candidate: the diagnostic-taps process that wrote the capture")
    bench = commands.add_parser("lab", help="the gate's tapdiff rules over a lab capture of the encoder alone")
    bench.add_argument("--model", required=True, help="the model whose encoder declarations set the rules")
    bench.add_argument("--engine", required=True, help="the lab capture's tap directory")
    bench.add_argument("--reference", action="append", required=True,
                       help="an encoder reference output (tools/refrun/encoder.py); repeat for split modes")
    bench.add_argument("--workload", required=True, help="the workload both ran")
    bench.add_argument("--json", required=True, help="tapdiff's receipt")
    args = parser.parse_args()
    startup()
    return lab(args) if args.command == "lab" else parity(args)


if __name__ == "__main__":
    sys.exit(main())
