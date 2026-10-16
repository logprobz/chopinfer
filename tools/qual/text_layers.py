"""Compare a bundle's text-layer taps with the GPU reference runner's taps.

The model-neutral form of the layer comparison: the declaration
(CHOPINFER_QUAL_CONFIG) names the model, whose reference adapter is
`models/<model>/reference.py` and whose checkpoint lock is the gate's pin
(`qualification.checkpoint.lock`). The constants (MODES, the dequantization
each reference mode must use; ADAPTERS, a mode's adapter where it is not
the model's reference.py; REQUIRED_TAPS; FLOOR_RATIO and
RELATIVE_RMS_LIMIT) are the declaration's for its text_layers check where
it declares them (`declared`), so a standalone run applies the same policy
as the gate worker, else the module's.

A diagnostic-taps bundle writes taps when served with CHOPINFER_TAPS_DIR
set: one directory per phase step under the shared tap names
(models/<model>/taps.json). The reference is a run of the GPU reference
runner over one prompt with taps (`python3 -m tools.refrun run --adapter
models/<model>/reference.py --lock LOCK ... --modes f32,bf16 --taps`): its
F32 taps (`RUN/taps/f32`) are the reference and its BF16 taps
(`RUN/taps/bf16`) the floor, the HF modeling's own BF16 distance from F32.
Where ADAPTERS gives the modes different adapters (a floor that models a
served format the plain HF modeling lacks, say), one run cannot hold both:
the reference is then a directory of one run per mode, `RUN/<mode>/`, each
run with that mode's adapter, so the F32 taps are `RUN/f32/taps/f32` and the
floor `RUN/bf16/taps/bf16`.

Nothing is counted before both sides are bound (otherwise the status is
`insufficient` and no row passes):

- The reference run, a declared input the reference lock pins
  (refs.verify_input): each mode's run.json names that mode's adapter and
  the pinned checkpoint lock; each mode's scores file holds the mode policy
  modes.py gives (with MODES' dequantization) and cache keys naming the
  gate's checkpoint lock, the model, the tapped sequence's token IDs and
  the working tree's runner numerics files and the mode's adapter, each
  present with its current digest. Each mode's taps
  hold that one sequence.
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
ROW_FLOOR (2^-8), so a few badly wrong rows fail the tap. With ROUTER
declared, tapdiff also judges routing flips (route-near-tie.v2): a tap that
fails only its aggregate passes when the fewest rows behind a near-tie
routing flip, largest error first, each within FLOOR_RATIO times the BF16
run's worst row, explain it; the rows and flips are reported
(`route_explanation`, `routes`). The correction biases come from the
declared input ROUTER names, authenticated by the reference lock and bound
to the checkpoint pin (tools/qual/router_bias.py). `first` names the
first failing tap in program order. Coverage: tapdiff judges the taps both
sides hold. With REQUIRED_TAPS (declaration tap names, `{nn}` expanded over
the layers models/<model>/taps.json gives each), every required tap must be
compared, else the status is `insufficient`; without it the report says its
scope is the intersection, a diagnostic comparison. tapdiff runs under
CHOPINFER_REFRUN_PYTHON (numpy). The runner's cache keys do not cover tap
payloads; the manifest's digests pin the reference taps the dump was
captured against.

Usage (CHOPINFER_QUAL_CONFIG set to the model's declaration):
  python3 tools/qual/text_layers.py capture --url URL --server-log LOG \\
      --taps-dir DIR --reference RUN --out DUMP
  python3 tools/qual/text_layers.py --dump DUMP --reference RUN --url URL [--json OUT]
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
from refs import verify_input  # noqa: E402
import router_bias  # noqa: E402
import tap_binding  # noqa: E402
from tools.model_sdk import diagnostic as wire, taps as tap_names  # noqa: E402
from tools.refrun import keys, modes  # noqa: E402

RELATIVE_RMS_LIMIT = 0.05
FLOOR_RATIO = 3.0


# The reference modes and the dequantization each must use.
MODES = {"f32": "exact", "bf16": "hf-bf16"}
# The reference adapter of each mode whose adapter is not the model's
# reference.py. With one adapter for every mode the reference is one run,
# else one run per mode under `<mode>/`.
ADAPTERS = {}
# The taps a comparison must hold to pass (taps.json names); none: the taps
# both sides hold, reported as such.
REQUIRED_TAPS = []
# The model's router, for judging routing flips (tapdiff --router,
# route-near-tie.v2): {"score": "sigmoid" or "softmax", "top_k", "n_group",
# "topk_group", "bias": the declared input holding the correction biases
# (tools/qual/router_bias.py), or absent}. Empty: flips are not judged and
# every tap keeps the plain rules.
ROUTER = {}


# The correctness runner captures and judges the dump on the same live arm.
RUNNER_ARMS = {"layers": ["--prefix-cache", "off", "--speculative-tokens", "0"]}
RUNNER_ENVIRONMENT = {"layers": {"CHOPINFER_TAPS_DIR": "{taps_dir}", "CHOPINFER_TAPS_SITES": "*",
                                    "CHOPINFER_REQUEST_TRACE": "1"}}
RUNNER_FEATURES = {"layers": ["diagnostic-taps"]}
RUNNER_CHECK_ARM = "layers"
RUNNER_DIRECTORY_INPUTS = ["layer_dump"]
RUNNER_EXTERNAL_INPUTS = ["layer_reference"]
RUNNER_RECORDS = {"layer_dump": ("layers", "text_layers.py",
                                 ["capture", "--url", "{url}", "--server-log", "{server_log}",
                                  "--taps-dir", "{taps_dir}", "--reference", "{layer_reference}",
                                  "--out", "{run_dir}"])}


Unbound = tap_binding.Unbound


def config():
    if not os.environ.get("CHOPINFER_QUAL_CONFIG"):
        raise Unbound("set CHOPINFER_QUAL_CONFIG to the model's declaration")
    return common.qualification_config()


def model():
    """The declaration's model: the one its reference adapter and taps
    belong to (a checkpoint variant names its model)."""
    return config()["model"]


def declared(name):
    """The constant `name` of the declaration's text_layers check (the check
    whose module this is), else the module's own value."""
    checks = config().get("checks", [])
    checks = [check for check in (checks.values() if isinstance(checks, dict) else checks)
              if check.get("module") == "text_layers"]
    if len(checks) > 1:
        raise Unbound("the declaration declares text_layers more than once")
    constants = checks[0].get("constants", {}).get("text_layers", {}) if checks else {}
    return constants[name]["value"] if name in constants else globals()[name]


def required_taps():
    """The tap names REQUIRED_TAPS names: each declared `{nn}` name over the
    layers its taps.json entries give (every layer the declaration lists
    where an entry gives none), as the first matching entry applies."""
    patterns = declared("REQUIRED_TAPS")
    if not patterns:
        return set()
    declaration = tap_names.model_declaration(model(), ROOT)
    if declaration is None:
        raise Unbound(f"models/{model()}/taps.json declares no taps")
    every = sorted({layer for _, _, entry in declaration.entries for layer in entry.get("layers", [])})
    names = set()
    for pattern in patterns:
        entries = [entry for _, _, entry in declaration.entries if entry["name"] == pattern]
        if not entries:
            raise Unbound(f"REQUIRED_TAPS names {pattern}, which taps.json does not declare")
        for entry in entries:
            candidates = ([pattern.replace("{nn}", f"{layer:02d}") for layer in entry.get("layers", every)]
                          if "{nn}" in pattern else [pattern])
            names |= {name for name in candidates if declaration.entry(name) is entry}
    return names


def adapter(mode="f32"):
    """The reference adapter of `mode`: the declaration's ADAPTERS entry,
    else the model's reference.py."""
    declared_adapters = declared("ADAPTERS")
    unknown = sorted(set(declared_adapters) - set(declared("MODES")))
    if unknown:
        raise Unbound(f"ADAPTERS names modes MODES does not: {', '.join(unknown)}")
    return declared_adapters.get(mode, f"models/{model()}/reference.py")


def mode_run(run, mode):
    """The run directory that holds `mode`: the reference itself when every
    mode shares one adapter, else its `<mode>/` run."""
    shared = len({adapter(name) for name in declared("MODES")}) == 1
    return Path(run) if shared else Path(run) / mode


def lock():
    return config()["qualification"]["checkpoint"]["lock"]


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
    its run, mode and tap files (named relative to the reference)."""
    run = wire.external(Path(run).absolute())
    # The run's bytes are the ones the reference lock pins.
    verify_input(config(), "layer_reference", run)
    name, pinned = model(), lock()
    source = keys.checkpoint(keys.read_lock(ROOT / pinned), [])["source_lock_sha256"]
    if source != common.checkpoint_pin(config())["source_lock_sha256"]:
        raise Unbound(f"{pinned} differs from the gate's checkpoint pin")
    numerics = {}
    sequences, files = {}, {}
    for mode, dequant in declared("MODES").items():
        held, used_adapter = mode_run(run, mode), adapter(mode)
        prefix = held.relative_to(run).as_posix() + "/" if held != run else ""
        argv = wire.read_json(held / "run.json")["argv"]
        option = lambda name: argv[argv.index(name) + 1] if name in argv else None  # noqa: E731
        if option("--adapter") != used_adapter or option("--lock") != pinned:
            raise Unbound(f"{held} was not run with {used_adapter} and {pinned}")
        files[f"{prefix}run.json"] = wire.identity(held / "run.json")
        scores = wire.read_json(held / f"{mode}.json")
        files[f"{prefix}{mode}.json"] = wire.identity(held / f"{mode}.json")
        expected = modes.policy(mode, dequant)
        if scores.get("policy") != expected:
            raise Unbound(f"{held}/{mode}.json does not hold modes.py's {mode} policy with {dequant} weights")
        entries = {}
        for entry in scores["entries"]:
            key = entry["key"]
            if key.get("model") != name or key["checkpoint"]["source_lock_sha256"] != source:
                raise Unbound(f"{held}/{mode}.json entry {entry['name']} names another model or checkpoint")
            if key["mode"] != keys.mode(expected):
                raise Unbound(f"{held}/{mode}.json entry {entry['name']} names another mode policy")
            used = key["adapter"]["files"]
            # The mode's adapter and the runner numerics must each be named
            # with its current digest.
            for file in sorted(set(used) | set(keys.NUMERICS) | {used_adapter}):
                if file not in numerics:
                    numerics[file] = keys.file_sha256(ROOT / file) if (ROOT / file).exists() else None
                if used.get(file) != numerics[file]:
                    raise Unbound(f"{file} differs from the reference run's adapter and numerics files")
            entries[entry["name"]] = key["sequence"]["token_ids_sha256"]
        steps = sorted((held / "taps" / mode).glob("step*"))
        if len(steps) != 1:
            raise Unbound(f"{held}/taps/{mode} holds {len(steps)} steps, not one sequence's")
        scope = step_scope(steps[0])
        request = scope["request"]
        if entries.get(request) != keys.ids_sha256(scope["tokens"]):
            raise Unbound(f"{held}/taps/{mode} holds {request}, whose tokens no {mode} cache key names")
        sequences[mode] = (request, scope["tokens"])
        files.update({f"{prefix}taps/{mode}/{steps[0].name}/{file}": value
                      for file, value in tap_binding.step_files(steps[0]).items()})
    if sequences["f32"] != sequences["bf16"]:
        raise Unbound(f"{run}'s f32 and bf16 taps hold different sequences")
    request, tokens = sequences["f32"]
    return {"run": str(run), "sequence": request, "tokens": tokens, "files": files}


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
                             "--f32", str(mode_run(reference, "f32") / "taps/f32"),
                             "--bf16", str(mode_run(reference, "bf16") / "taps/bf16"),
                             "--ratio", str(declared("FLOOR_RATIO")),
                             "--limit", str(declared("RELATIVE_RMS_LIMIT")),
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
        required = required_taps()
    except (OSError, ValueError, KeyError, TypeError) as error:
        return insufficient(dump, reference, f"unbound: {error}")
    dump, reference = Path(dump).absolute(), Path(bound["run"])
    with tempfile.TemporaryDirectory() as scratch:
        try:
            routing = router_bias.tapdiff_args(declared("ROUTER"), config(), scratch)
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
    missing = sorted(required - {row["tensor"] for row in rows})
    scope = ({"kind": "required", "required": len(required), "missing": missing} if required else
             {"kind": "intersection", "note": "the taps both sides hold; no required coverage is declared, "
                                              "so this is a diagnostic comparison"})
    reason = None
    if missing and judged:
        status, reason = "insufficient", (f"{len(missing)} required taps not compared, first "
                                          f"{', '.join(missing[:5])}")
        for row in rows:
            row["pass"] = None
    return {"dump": str(dump), "reference": str(reference), "status": status, "first": diff["first"],
            **({"reason": reason} if reason else {}), "scope": scope,
            "rows": rows, "binding": {"server": manifest["server"], "request": manifest["requests"][0],
                                      "sequence": bound["sequence"],
                                      "manifest_sha256": wire.identity(dump / tap_binding.MANIFEST)["sha256"]},
            "routes": diff.get("routes"),
            "tapdiff": {"status": diff["status"], "incomplete_steps": diff["incomplete_steps"],
                        "requests": diff["requests"], "inputs": diff["inputs"]},
            "coverage": {"compared": len(rows), "gated": sum(row["pass"] is not None for row in rows)}}


def main():
    if sys.argv[1:2] == ["capture"]:
        parser = argparse.ArgumentParser(prog="text_layers.py capture",
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
                        help="reference runner run directory (taps/f32 and taps/bf16), or one run per mode "
                             "(f32/ and bf16/) where ADAPTERS gives the modes different adapters")
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
