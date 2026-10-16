"""Compare a Qwen3.8 bundle activation dump with the GPU reference runner's taps.

The bundle writes taps when served with CHOPINFER_TAPS_DIR set
(models/qwen38/bundle/src/dump.rs, built with its diagnostic-taps feature):
one directory per phase step under the shared tap names
(models/qwen38/taps.json). The reference is a run of the GPU reference runner
over one prompt with taps (`python3 -m tools.refrun run --adapter
models/qwen38/reference.py --lock models/qwen38/checkpoint-files.json ...
--modes f32,bf16 --taps --tap-sites all`): its F32 taps (`RUN/taps/f32`)
are the reference and its BF16 taps (`RUN/taps/bf16`) the floor, the HF
modeling's own BF16 distance from F32.

Nothing is counted before both sides are bound (otherwise the status is
`insufficient` and no row passes):

- The reference run, a declared input the reference lock pins
  (refs.verify_input): run.json names the pinned adapter and checkpoint lock;
  each mode's scores file holds the mode policy modes.py gives (F32 with
  exact dequantization, BF16 with HF's) and cache keys naming the gate's
  checkpoint lock, the working tree's adapter and runner numerics files,
  the model and the tapped sequence's token IDs. Each mode's taps hold that
  one sequence.
- The dump: `capture` (below) sends the reference sequence once to a
  diagnostic-taps server through tools/qual/tap_binding.py, which binds the
  live server to the gate's checkpoint pin and its startup log and admits
  only the step its own metadata names as written by that process (startup
  nonce) for that request (engine request ID). The dump's manifest
  (tap-capture.json) records the serving identity, the request and the
  digests of the step's files, and, as its subject, the digests of the
  reference run's files. The comparison re-admits the dump against the live
  candidate at --url (tap_binding.admit): every identity field, startup
  nonce, package, metadata, checkpoint, profile and source commit, must be
  the candidate's, so the candidate is the process that wrote the step. It
  also requires a fresh prefill (one item, prefix 0, positions from 0) of
  the reference sequence at the candidate's tensor-parallel size, and
  refuses symbolic links and files that change while read.

tools/qual/tapdiff.py then aligns the rows by request and position and
judges every tap the two hold alike: each is one row of the report, passing
when its median row error is within FLOOR_RATIO times the BF16 floor
(RELATIVE_RMS_LIMIT where the floor is zero) and its relative L2 over all
compared rows within FLOOR_RATIO times the BF16 run's, raised to tapdiff's
ROW_FLOOR (2^-8), so a few badly wrong rows fail the tap. `first` names the
first failing tap in program order. tapdiff runs under
CHOPINFER_REFRUN_PYTHON (numpy). The runner's cache keys do not cover tap
payloads; the manifest's digests pin the reference taps the dump was
captured against.

Usage:
  python3 tools/qual/qwen38_layers.py capture --url URL --server-log LOG \
      --taps-dir DIR --reference RUN --out DUMP
  python3 tools/qual/qwen38_layers.py --dump DUMP --reference RUN --url URL [--json OUT]
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools/qual"))
import common  # noqa: E402
import declaration  # noqa: E402
from refs import verify_input  # noqa: E402
import router_bias  # noqa: E402
import tap_binding  # noqa: E402
from tools.model_sdk import diagnostic as wire  # noqa: E402
from tools.refrun import keys, modes  # noqa: E402

RELATIVE_RMS_LIMIT = 0.05
FLOOR_RATIO = 3.0
# The router, for judging routing flips (tapdiff --router,
# route-near-tie.v2; text_layers.ROUTER): empty, flips are not judged.
ROUTER = {}


ADAPTER = "models/qwen38/reference.py"
LOCK = "models/qwen38/checkpoint-files.json"
# The reference modes and the dequantization each must use.
MODES = {"f32": "exact", "bf16": "hf-bf16"}


# The correctness runner captures and judges the dump on the same live arm.
RUNNER_ARMS = {"layers": ["--prefix-cache", "off", "--speculative-tokens", "0"]}
RUNNER_ENVIRONMENT = {"layers": {"CHOPINFER_TAPS_DIR": "{taps_dir}", "CHOPINFER_TAPS_SITES": "*",
                                    "CHOPINFER_REQUEST_TRACE": "1"}}
RUNNER_FEATURES = {"layers": ["diagnostic-taps"]}
RUNNER_CHECK_ARM = "layers"
RUNNER_DIRECTORY_INPUTS = ["layer_dump"]
RUNNER_EXTERNAL_INPUTS = ["layer_reference"]
RUNNER_RECORDS = {"layer_dump": ("layers", Path(__file__).name,
                                 ["capture", "--url", "{url}", "--server-log", "{server_log}",
                                  "--taps-dir", "{taps_dir}", "--reference", "{layer_reference}",
                                  "--out", "{run_dir}"])}


Unbound = tap_binding.Unbound


def config():
    value = (common.qualification_config() if os.environ.get("CHOPINFER_QUAL_CONFIG")
             else declaration.load(declaration.path("qwen38")))
    if value.get("model") != "qwen38":
        raise Unbound(f"a {value.get('model')} gate config, not Qwen3.8's")
    return value


def step_scope(step):
    """The sequence a complete one-item prefill step holds from position 0:
    its request, tokens and rank count."""
    capture = wire.read_json(step / "capture.json")
    completion = wire.read_json(step / "completion.json")
    items = wire.read_json(step / "items.json")["items"]
    if completion.get("completion") != "full":
        raise Unbound(f"{step} is incomplete: {completion.get('reason')}")
    if len(items) != 1:
        raise Unbound(f"{step} holds {len(items)} items, not one sequence")
    item = items[0]
    if item.get("prefix") != 0 or item.get("positions") != list(range(len(item.get("tokens") or []))) \
            or not item["tokens"]:
        raise Unbound(f"{step} is not a fresh prefill from position 0")
    return {"request": item["request"], "tokens": item["tokens"], "ranks": capture.get("ranks")}


def reference_binding(run):
    """The bound reference run: its sequence, token IDs and the digests of
    its run, mode and tap files."""
    run = wire.external(Path(run).absolute())
    # The run's bytes are the ones the reference lock pins.
    verify_input(config(), "layer_reference", run)
    argv = wire.read_json(run / "run.json")["argv"]
    option = lambda name: argv[argv.index(name) + 1] if name in argv else None  # noqa: E731
    if option("--adapter") != ADAPTER or option("--lock") != LOCK:
        raise Unbound(f"{run} was not run with {ADAPTER} and {LOCK}")
    lock = keys.checkpoint(keys.read_lock(ROOT / LOCK), [])["source_lock_sha256"]
    if lock != common.checkpoint_pin(config())["source_lock_sha256"]:
        raise Unbound(f"{LOCK} differs from the gate's checkpoint pin")
    numerics = {}
    sequences, files = {}, {"run.json": wire.identity(run / "run.json")}
    for mode, dequant in MODES.items():
        scores = wire.read_json(run / f"{mode}.json")
        files[f"{mode}.json"] = wire.identity(run / f"{mode}.json")
        expected = modes.policy(mode, dequant)
        if scores.get("policy") != expected:
            raise Unbound(f"{run}/{mode}.json does not hold modes.py's {mode} policy with {dequant} weights")
        entries = {}
        for entry in scores["entries"]:
            key = entry["key"]
            if key.get("model") != "qwen38" or key["checkpoint"]["source_lock_sha256"] != lock:
                raise Unbound(f"{run}/{mode}.json entry {entry['name']} names another model or checkpoint")
            if key["mode"] != keys.mode(expected):
                raise Unbound(f"{run}/{mode}.json entry {entry['name']} names another mode policy")
            adapter = key["adapter"]["files"]
            for path in sorted(set(adapter) | set(keys.NUMERICS)):
                if path not in numerics:
                    numerics[path] = keys.file_sha256(ROOT / path) if (ROOT / path).exists() else None
                if adapter.get(path) != numerics[path]:
                    raise Unbound(f"{path} differs from the reference run's adapter and numerics files")
            entries[entry["name"]] = key["sequence"]["token_ids_sha256"]
        steps = sorted((run / "taps" / mode).glob("step*"))
        if len(steps) != 1:
            raise Unbound(f"{run}/taps/{mode} holds {len(steps)} steps, not one sequence's")
        scope = step_scope(steps[0])
        name = scope["request"]
        if entries.get(name) != keys.ids_sha256(scope["tokens"]):
            raise Unbound(f"{run}/taps/{mode} holds {name}, whose tokens no {mode} cache key names")
        sequences[mode] = (name, scope["tokens"])
        files.update({f"taps/{mode}/{steps[0].name}/{file}": value
                      for file, value in tap_binding.step_files(steps[0]).items()})
    if sequences["f32"] != sequences["bf16"]:
        raise Unbound(f"{run}'s f32 and bf16 taps hold different sequences")
    name, tokens = sequences["f32"]
    return {"run": str(run), "sequence": name, "tokens": tokens, "files": files}


def capture(args):
    """Sends the reference sequence once to the bound diagnostic-taps server
    and writes the dump: the capture of its one new prefill step."""
    reference = reference_binding(args.reference)
    body = {"prompt": reference["tokens"], "max_tokens": 1, "temperature": 0}
    manifest = tap_binding.capture(args.url, args.server_log, args.taps_dir,
                                   [("layers", "/v1/completions", body)], args.out, config(),
                                   subject={"reference": reference["files"], "sequence": reference["sequence"]})
    dump_step(Path(args.out), manifest, reference)
    print(json.dumps({"dump": str(Path(args.out).absolute()), "steps": manifest["requests"][0]["steps"],
                      "engine_request_id": manifest["requests"][0]["engine_request_id"]}))
    return 0


def dump_step(dump, manifest, reference):
    """The dump's one step, a fresh prefill of the reference sequence at the
    captured server's tensor-parallel size, captured against `reference`."""
    if (manifest.get("subject") or {}).get("reference") != reference["files"]:
        raise Unbound("the reference run differs from the one the dump was captured against")
    (record,) = manifest["requests"]
    if len(record["steps"]) != 1:
        raise Unbound(f"the reference prefill wrote {len(record['steps'])} steps, not one")
    step = Path(dump).absolute() / tap_binding.STEPS / record["steps"][0]
    scope = step_scope(step)
    if scope["tokens"] != reference["tokens"]:
        raise Unbound("the dump's step is not a prefill of the reference sequence")
    if scope["ranks"] != manifest["server"]["profile"]["tensor_parallel_size"]:
        raise Unbound("the dump's step and the server name different tensor-parallel sizes")
    return step


def tapdiff(step, reference, receipt, routing=()):
    """tapdiff's exit status, receipt and error text: the dump's step against
    the runner's F32 taps with its BF16 taps as the floor, each tap also held
    as a whole."""
    python = os.environ.get("CHOPINFER_REFRUN_PYTHON") or sys.executable
    result = subprocess.run([python, "-B", "-m", "tools.qual.tapdiff", "--engine", str(step.parent),
                             "--steps", step.name,
                             "--f32", str(reference / "taps/f32"), "--bf16", str(reference / "taps/bf16"),
                             "--ratio", str(FLOOR_RATIO), "--limit", str(RELATIVE_RMS_LIMIT),
                             "--aggregate", "*", *routing, "--json", str(receipt)],
                            cwd=ROOT, text=True, capture_output=True)
    return result.returncode, json.loads(receipt.read_text()) if receipt.exists() else None, result.stderr.strip()


def insufficient(dump, reference, reason):
    return {"dump": str(dump), "reference": str(reference), "status": "insufficient", "first": None,
            "rows": [], "reason": reason, "coverage": {"compared": 0, "gated": 0}}


def compare(dump, reference, url=None):
    """The report: one row per compared tap and the first failing one, or
    insufficient evidence when a binding is missing or differs."""
    try:
        bound = reference_binding(reference)
        manifest = tap_binding.admit(dump, url, config())
        step = dump_step(dump, manifest, bound)
    except (OSError, ValueError, KeyError, TypeError) as error:
        return insufficient(dump, reference, f"unbound: {error}")
    dump, reference = Path(dump).absolute(), Path(bound["run"])
    with tempfile.TemporaryDirectory() as scratch:
        try:
            routing = router_bias.tapdiff_args(ROUTER, config(), scratch)
        except (OSError, ValueError, KeyError, TypeError) as error:
            return insufficient(dump, reference, f"unbound: {error}")
        code, diff, error = tapdiff(step, reference, Path(scratch) / "tapdiff.json", routing)
    try:
        # The files tapdiff read are the ones bound before it ran.
        if reference_binding(reference)["files"] != bound["files"] or \
                tap_binding.step_files(step) != manifest["files"][step.name]:
            raise Unbound("the dump or reference changed while compared")
    except (OSError, ValueError, KeyError, TypeError) as error:
        return insufficient(dump, reference, f"unbound: {error}")
    if diff is None:
        return {"dump": str(dump), "reference": str(reference), "status": "incompatible", "first": None,
                "rows": [], "reason": error or f"tapdiff exit {code}", "coverage": {"compared": 0, "gated": 0}}
    status = {"match": "match", "diverged": "diverged", "incompatible": "incompatible"}.get(diff["status"],
                                                                                          "insufficient")
    judged = status in ("match", "diverged")
    rows = []
    for tap in diff["taps"]:
        if not tap.get("rows"):
            continue
        row = {"tensor": tap["tap"], "rows": tap["rows"], "status": tap["status"],
               "pass": (not tap.get("fail") and tap["status"] == "ok") if judged else None}
        for key in ("e", "floor", "e_over_floor", "excess", "aggregate", "disagree", "bf16_disagree",
                    "route_explanation"):
            if key in tap:
                row[key] = tap[key]
        rows.append(row)
    return {"dump": str(dump), "reference": str(reference), "status": status, "first": diff["first"],
            "rows": rows, "binding": {"server": manifest["server"], "request": manifest["requests"][0],
                                      "sequence": bound["sequence"],
                                      "manifest_sha256": wire.identity(dump / tap_binding.MANIFEST)["sha256"]},
            "routes": diff.get("routes"),
            "tapdiff": {"status": diff["status"], "incomplete_steps": diff["incomplete_steps"],
                        "requests": diff["requests"], "inputs": diff["inputs"]},
            "coverage": {"compared": len(rows), "gated": sum(row["pass"] is not None for row in rows)}}


def main():
    if sys.argv[1:2] == ["capture"]:
        parser = argparse.ArgumentParser(prog="qwen38_layers.py capture",
                                         description="capture a bound layer dump from a diagnostic-taps server")
        parser.add_argument("--url", required=True, help="the diagnostic-taps server")
        parser.add_argument("--server-log", required=True, help="its startup log")
        parser.add_argument("--taps-dir", required=True, help="its CHOPINFER_TAPS_DIR")
        parser.add_argument("--reference", required=True, type=Path, help="the reference runner run")
        parser.add_argument("--out", required=True, help="new dump directory")
        args = parser.parse_args(sys.argv[2:])
        try:
            return capture(args)
        except (OSError, ValueError, KeyError, RuntimeError) as error:
            print(f"capture refused: {error}", file=sys.stderr)
            return 3
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dump", required=True, type=Path, help="a dump directory `capture` wrote")
    parser.add_argument("--reference", required=True, type=Path,
                        help="reference runner run directory (taps/f32 and taps/bf16)")
    parser.add_argument("--url", help="the candidate: the diagnostic-taps process that wrote the dump")
    parser.add_argument("--json", help="write the report here")
    args = parser.parse_args()
    report = compare(args.dump, args.reference, args.url)
    for row in report["rows"]:
        mark = {True: "ok  ", False: "FAIL", None: "    "}[row["pass"]]
        print(f"{mark} {row['tensor']:28s} {row['status']}", file=sys.stderr)
    print(f"{report['status']}: first failing tap {report['first']}"
          + (f" ({report['reason']})" if report.get("reason") else ""), file=sys.stderr)
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=1, allow_nan=False) + "\n")
    print(json.dumps(report, allow_nan=False))
    return {"match": 0, "diverged": 1, "incompatible": 2, "insufficient": 3}[report["status"]]


if __name__ == "__main__":
    sys.exit(main())
