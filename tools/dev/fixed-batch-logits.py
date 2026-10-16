#!/usr/bin/env python3
"""Prepare without devices; replay only on four idle GPUs."""
import argparse
import datetime
import hashlib
import json
import os
import re
from pathlib import Path
import shutil
import subprocess
import time
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import diagnostics

REPO = Path(__file__).resolve().parents[2]
TEST = "executor::fixed_batch::device::replay_installed_trace_gpu"
# Frame metadata defaults to the reader's 256 MiB ceiling: drafter traces
# carry one grammar mask per verify row, and the replay's bound for the
# widest comparison (32 requests, eight rows each, a 2^20-token vocabulary)
# stays below it (runtime-host `widest_comparison_trace_fits_the_metadata_cap`).
# Each replay checks its own trace's bound before running.
DEFAULT_LIMITS = dict(logits_bytes=4 << 30, step_bytes=256 << 20,
                      metadata_bytes=diagnostics.MAX_METADATA_LIMIT)
# Per rank count: the cases, and each package's model, drafter and extra
# redeploy arguments. TP2 covers Qwen only; its images need the host
# n-gram table.
TOPOLOGIES = {
    4: ([("qwen-ar", 1), ("qwen-ar", 16), ("qwen-ar", 32),
         ("qwen-mtp", 16), ("mimo-ar", 16), ("mimo-ar", 32)],
        {"qwen-ar": ("qwen38", "none", []), "qwen-mtp": ("qwen38", "mtp", []),
         "mimo-ar": ("mimo", "none", [])}),
    2: ([("qwen-ar", 16), ("qwen-ar", 32), ("qwen-mtp", 16)],
        {"qwen-ar": ("qwen38", "none", []), "qwen-mtp": ("qwen38", "mtp", [])}),
}
# Each model's package ID, and its topology at a rank count.
BUNDLES = {"qwen38": ("qwen38.flash.v2", "qwen38-flash-tp{}"),
           "mimo": ("mimo.v26.flash.v2", "mimo-v26-flash-tp{}")}
# The checkpoint each model's direct package loads.
CHECKPOINTS = {"qwen38": "CHOPINFER_QWEN38_CHECKPOINT", "mimo": "CHOPINFER_MIMO_CHECKPOINT"}
FORMAT = "fixed-batch-direct-v5"
# Bundles whose instances serving at most LIMIT sequences verify at most
# LIMIT rows per step unless SWITCH=1 opts into the wide geometry at process
# start (Qwen3.8's `narrow_verify_tokens`, models/qwen38/contract). The
# manifest does not record this runtime limit.
NARROW_VERIFY_ROWS = {"qwen38.": (32, "CHOPINFER_SPECULATIVE_WIDE_ROWS")}


def speculation_problems(manifest, environment, concurrencies, max_sequences, rows_per_item):
    """Why a package cannot replay the package-comparison trace, whose verify
    steps carry bundle-drafted rows, up to `rows_per_item` per item for every
    live request. Empty when it can."""
    speculation = manifest.get("speculation") or {}
    name = manifest.get("id", "package")
    if not speculation.get("bundle_drafter"):
        return [f"{name}: the comparison trace verifies bundle drafts, but the package has no "
                "bundle drafter; package it with its drafter (for example --drafter mtp or dflash)"]
    problems = []
    if speculation.get("maximum_draft_tokens", 0) < rows_per_item - 1:
        problems.append(f"{name}: drafts at most {speculation.get('maximum_draft_tokens', 0)} tokens, "
                        f"the trace {rows_per_item - 1}")
    if rows_per_item > max(speculation.get("row_buckets") or [0]):
        problems.append(f"{name}: verify row buckets {speculation.get('row_buckets')} "
                        f"do not cover {rows_per_item} rows per item")
    rows = max(concurrencies) * rows_per_item
    if rows > speculation.get("maximum_verify_tokens", 0):
        problems.append(f"{name}: verifies at most {speculation.get('maximum_verify_tokens', 0)} "
                        f"rows per step, the trace {rows}")
    for prefix, (limit, switch) in NARROW_VERIFY_ROWS.items():
        if (name.startswith(prefix) and max_sequences <= limit and rows > limit
                and environment.get(switch) != "1"):
            problems.append(f"{name}: instances serving at most {limit} sequences verify at most "
                            f"{limit} rows per step, the trace {rows}; set {switch}=1 with "
                            "--base-env and --tip-env")
    return problems


def write_new(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def output(args):
    return subprocess.check_output(args, cwd=REPO, text=True).strip()


def source():
    if output(["git", "status", "--porcelain"]):
        raise RuntimeError("commit the prepared source before building or running")
    return output(["git", "rev-parse", "HEAD"])


def command(args, log, env=None, timeout=None):
    print("run:", " ".join(map(str, args)), "log:", log, flush=True)
    with log.open("x") as stream:
        # subprocess.run kills and collects only its own child on timeout.
        subprocess.run(list(map(str, args)), cwd=REPO, env=env, stdout=stream,
                       stderr=subprocess.STDOUT, check=True, timeout=timeout)


def build_recorder(destination):
    build = destination / "build.jsonl"
    command(["cargo", "test", "-p", "runtime-host", "--lib", "--release", "--features",
             "linux-cuda-host", "--locked", "--offline", "--no-run", "--message-format=json"], build)
    artifacts = []
    for line in build.read_text().splitlines():
        if not line.startswith("{"):
            continue
        row = json.loads(line)
        if row.get("target", {}).get("name") == "runtime_host" and row.get("executable"):
            artifacts.append(row["executable"])
    if len(artifacts) != 1:
        raise RuntimeError("expected exactly one host test executable")
    binary = destination / "replay-tests"
    shutil.copy2(artifacts[0], binary)
    listing = output([str(binary), "--list", "--ignored", "--exact", TEST])
    if f"{TEST}: test" not in listing or "1 test," not in listing:
        raise RuntimeError("device replay test is absent")
    return binary


def prepare(root, ranks, hc_compare=False):
    revision = source()
    root.mkdir(parents=True, exist_ok=True)
    destination = root / ("prepared-" + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S"))
    destination.mkdir()
    binary = build_recorder(destination)
    packages = {}
    for name, (model, drafter, extra) in TOPOLOGIES[ranks][1].items():
        bundle_id, topology = BUNDLES[model][0], BUNDLES[model][1].format(ranks)
        checkpoint = os.environ.get(CHECKPOINTS[model])
        if not checkpoint:
            raise RuntimeError(f"set {CHECKPOINTS[model]} to the checkpoint the package loads")
        rank_args = [] if ranks == 4 else ["--ranks", ranks]
        log = destination / (name + "-prepare.log")
        command(["bash", f"tools/dev/{model}-redeploy.sh", "--prepare-only", "--drafter", drafter, *rank_args,
                 *extra, "--checkpoint", checkpoint, "--work-dir", root / "install",
                 "--model-cache-dir", root / "model-cache", "--logs-dir", destination / "package-logs"], log)
        prepared = [line.removeprefix("prepared ") for line in log.read_text().splitlines()
                    if line.startswith("prepared ")]
        if len(prepared) != 1:
            raise RuntimeError("package preparation did not name one direct package")
        bundle = Path(prepared[0])
        manifest = bundle / "manifest.json"
        if not manifest.is_file() or json.loads(manifest.read_text()).get("id") != bundle_id:
            raise RuntimeError(f"missing prepared {bundle_id} package at {bundle}")
        packages[name] = {"bundle": str(bundle), "topology": topology, "checkpoint": checkpoint}
    if source() != revision:
        raise RuntimeError("source changed during preparation")
    receipt = {"format": FORMAT, "source": revision, "ranks": ranks, "binary": str(binary),
               "binary_sha256": digest(binary), "packages": packages, "hc_compare": hc_compare}
    write_new(destination / "prepared.json", receipt)
    # This mutable pointer names an immutable completed preparation.
    (root / "latest-prepared").write_text(str(destination / "prepared.json") + "\n")
    print("prepared:", destination / "prepared.json", flush=True)


def rank_ordinals(devices, ranks):
    """One visible ordinal per rank, on distinct devices."""
    ordinals = devices.split(",")
    if (ranks not in TOPOLOGIES or len(ordinals) != ranks
            or len(set(ordinals)) != ranks or not all(x.isdigit() for x in ordinals)):
        raise ValueError("rank count requires one distinct GPU ordinal per rank")
    return ",".join(str(rank) for rank in range(ranks))


def arm_environment(parent, devices, model, owned, ranks=4):
    env = {key: value for key, value in parent.items()
           if not key.startswith(("CHOPINFER_MIMO_", "CHOPINFER_QWEN38_"))}
    env.pop("CHOPINFER_KERNEL_GPU", None)
    env.pop("CHOPINFER_KERNEL_SHARED_DEVICE", None)
    env.update(CUDA_VISIBLE_DEVICES=devices, CHOPINFER_KERNEL_GPUS=rank_ordinals(devices, ranks),
               CHOPINFER_SAMPLING_DEFER_VIEWS="1" if owned else "0")
    if model.startswith("qwen"):
        env["CHOPINFER_QWEN38_ROW_OWNED_LOGITS"] = "1" if owned else "0"
    elif not owned:
        env["CHOPINFER_MIMO_ROW_OWNED_LOGITS"] = "0"
        env["CHOPINFER_MIMO_ROW_OWNED_SPECULATION"] = "0"
    # The owned arm deliberately exercises the default for this bundle.
    return env


def gpu_state(devices):
    count = len(devices.split(","))
    processes = output(["nvidia-smi", "-i", devices, "--query-compute-apps=pid", "--format=csv,noheader"])
    rows = output(["nvidia-smi", "-i", devices, "--query-gpu=index,uuid,name,driver_version,memory.used",
                   "--format=csv,noheader,nounits"])
    if processes or any(int(row.rsplit(",", 1)[1].strip()) >= 1000 for row in rows.splitlines()):
        raise RuntimeError(f"GPUs are not empty: {processes}\n{rows}")
    if len(rows.splitlines()) != count:
        raise RuntimeError(f"expected {count} visible physical devices")
    return rows


def plan_arms(ranks=4, hc_compare=False):
    if hc_compare:
        # HC precompute runs Qwen's decode buckets 16 and 32: its C16 and C32
        # cases at this rank count.
        cases = [(model, concurrency) for model, concurrency in TOPOLOGIES[ranks][0]
                 if model.startswith("qwen-") and concurrency in (16, 32)]
        return [(model, concurrency, arm)
                for arm in ("gather", "control", "owned-off", "owned-on")
                for model, concurrency in cases]
    # Gate every case before starting any owned arm.
    return [(model, concurrency, arm) for arm in ("gather", "control", "owned")
            for model, concurrency in TOPOLOGIES[ranks][0]]


def run(root, devices, minutes, ranks, hc_compare=False):
    ordinals = devices.split(",")
    gpus = ranks
    if len(ordinals) != gpus or len(set(ordinals)) != gpus or not all(x.isdigit() for x in ordinals):
        raise RuntimeError(f"TP{ranks} needs {gpus} distinct GPU ordinals")
    prepared_path = Path((root / "latest-prepared").read_text().strip())
    prepared = json.loads(prepared_path.read_text())
    if prepared.get("ranks", 4) != ranks:
        raise RuntimeError("latest preparation is for another rank count")
    if hc_compare and prepared.get("hc_compare") is not True:
        raise RuntimeError("preparation does not identify an HC comparison build")
    revision = source()
    # Documentation-only and runner changes do not invalidate the built kernels.
    subprocess.run(["git", "diff", "--exit-code", prepared["source"], "--", "crates", "models", "Cargo.toml", "Cargo.lock", "kernels.lock"], cwd=REPO, check=True)
    binary = Path(prepared["binary"])
    if digest(binary) != prepared["binary_sha256"]:
        raise RuntimeError("prepared executable changed")
    directory = root / ("run-" + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S"))
    directory.mkdir()
    before = gpu_state(devices)
    write_new(directory / "start.json", {"prepared": prepared, "runner_source": revision,
              "devices": before, "minutes": minutes, "hc_compare": hc_compare})
    deadline = time.monotonic() + minutes * 60
    passed = False
    try:
        for model, concurrency, arm in plan_arms(ranks, hc_compare):
            owned = arm.startswith("owned")
            case = directory / f"{model}-c{concurrency}"
            case.mkdir(exist_ok=True)
            config = {**prepared["packages"][model], "trace": str(case / "trace.json"),
                      "output": str(case / arm), "reference": None if arm == "gather" else str(case / "gather"),
                      "control": str(case / "control") if owned else None,
                      "concurrency": concurrency, "verify": model == "qwen-mtp", "owned": owned,
                      "source": {"build": prepared["source"], "binary_sha256": prepared["binary_sha256"]}}
            if owned:
                control = json.loads((case / "control/receipt.json").read_text())
                if control.get("passed") is not True:
                    raise RuntimeError("gathered repeat did not pass")
            config_path = case / (arm + ".json")
            write_new(config_path, config)
            env = arm_environment(os.environ, devices, model, owned, ranks)
            if hc_compare:
                env["CHOPINFER_QWEN38_HC_PRECOMPUTE"] = "1" if arm == "owned-on" else "0"
            env["CHOPINFER_BATCH_REPLAY_CONFIG"] = str(config_path)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError("slot deadline reached")
            command([binary, "--ignored", "--exact", TEST, "--test-threads=1", "--nocapture"],
                    case / (arm + ".log"), env, remaining)
            if hc_compare:
                expected_switch = "on" if arm == "owned-on" else "off"
                if f"Qwen3.8 HC precompute: {expected_switch}" not in (case / (arm + ".log")).read_text():
                    raise RuntimeError("replay did not report requested HC switch")
            receipt = json.loads((case / arm / "receipt.json").read_text())
            if receipt.get("passed") is not True or receipt.get("steps") != 13 or receipt.get("rows", 0) <= 0:
                raise RuntimeError("missing complete trace receipt")
            gpu_state(devices)
        passed = True
    finally:
        after = gpu_state(devices)
        write_new(directory / "result.json", {"passed": passed, "devices_after": after,
                  "source": revision, "prepared": str(prepared_path), "hc_compare": hc_compare})
        print("result:", directory, "passed:", passed, "GPUs empty", flush=True)


def parse_overrides(values):
    overrides = {}
    for value in values:
        key, separator, setting = value.partition("=")
        # Explicit bundle dispatch switches only. Device selection and recorder
        # configuration belong to the runner and cannot be overridden here.
        if not separator or not re.fullmatch(r"CHOPINFER_[A-Z0-9_]+", key):
            raise ValueError("arm overrides require CHOPINFER_NAME=value")
        if key.startswith(("CHOPINFER_KERNEL_", "CHOPINFER_BATCH_", "CHOPINFER_BUNDLE_")):
            raise ValueError("device, recorder and package controls cannot be arm overrides")
        if key in overrides:
            raise ValueError(f"duplicate arm override: {key}")
        overrides[key] = setting
    return overrides


def comparison_environment(parent, devices, ranks, overrides):
    # No inherited runtime switches: each arm's complete switches are recorded.
    env = {key: value for key, value in parent.items()
           if not key.startswith("CHOPINFER_")}
    env.update(CUDA_VISIBLE_DEVICES=devices,
               CHOPINFER_KERNEL_GPUS=rank_ordinals(devices, ranks))
    env.update(overrides)
    return env


def comparison_arms(concurrencies):
    return [(concurrency, arm) for arm in ("base", "control", "tip")
            for concurrency in concurrencies]


def prepare_compare(args):
    concurrencies = args.concurrency or [16, 32]
    # Decode comparisons take C1 (one- and two-row steps) and the C16/C32 buckets.
    if not getattr(args, "prefill_only", False) and any(c not in (1, 16, 32) for c in concurrencies):
        raise ValueError("decode comparisons take C1, C16 or C32; other sizes require --prefill-only")
    if max(concurrencies) > args.max_sequences:
        raise ValueError("--max-sequences must cover every live concurrency")
    revision = source()
    packages = {}
    if args.checkpoint is None:
        raise ValueError("comparison requires the --checkpoint both packages load")
    for arm, path, settings in (("base", args.base_bundle, args.base_env),
                                ("tip", args.tip_bundle, args.tip_env)):
        if path is None:
            raise ValueError("comparison requires the direct packages --base-bundle and --tip-bundle")
        diagnostics.external(path)
        packages[arm] = {"bundle": str(path), "topology": args.topology,
                         "checkpoint": str(args.checkpoint),
                         "manifest_sha256": digest(path / "manifest.json"),
                         "environment": parse_overrides(settings)}
    if not args.topology:
        raise ValueError("comparison requires --topology")
    # Refuse here, not after both arms verify their images on the GPUs.
    if not getattr(args, "prefill_only", False):
        problems = [problem for package in packages.values()
                    for problem in speculation_problems(
                        json.loads((Path(package["bundle"]) / "manifest.json").read_text()),
                        package["environment"], concurrencies, args.max_sequences,
                        args.verify_rows_per_item)]
        if problems:
            raise ValueError("packages cannot replay the comparison trace:\n" + "\n".join(problems))
    base, tip = packages['base'], packages['tip']
    differences = diagnostics.changes('manifest',
        diagnostics.manifest_identity(json.loads((Path(base['bundle']) / 'manifest.json').read_text())),
        diagnostics.manifest_identity(json.loads((Path(tip['bundle']) / 'manifest.json').read_text())))
    differences += diagnostics.changes('environment', base['environment'], tip['environment'])
    fields = {d['field'] for d in differences}
    if fields != set(args.allow_difference):
        raise ValueError('declare each intended change with --allow-difference FIELD: ' + json.dumps(differences))
    args.root.mkdir(parents=True, exist_ok=True)
    destination = args.root / ("prepared-compare-" + datetime.datetime.now(
        datetime.timezone.utc).strftime("%Y%m%dT%H%M%S"))
    destination.mkdir()
    binary = build_recorder(destination)
    if source() != revision:
        raise RuntimeError("source changed during preparation")
    receipt = {"format": FORMAT, "source": revision,
               "ranks": args.ranks, "binary": str(binary), "binary_sha256": digest(binary),
               "packages": packages, "concurrencies": concurrencies, "max_sequences": args.max_sequences,
               "verify_rows_per_item": None if getattr(args, "prefill_only", False) else args.verify_rows_per_item,
               "prefill_only": getattr(args, "prefill_only", False),
               "declared_differences": differences, "limits": args.limits}
    write_new(destination / "prepared.json", receipt)
    (args.root / "latest-compare").write_text(str(destination / "prepared.json") + "\n")
    print("prepared:", destination / "prepared.json", flush=True)


def run_compare(root, devices, minutes, ranks):
    ordinals = devices.split(",")
    count = ranks
    if len(ordinals) != count or len(set(ordinals)) != count or not all(x.isdigit() for x in ordinals):
        raise RuntimeError(f"TP{ranks} requires {count} distinct GPU ordinals")
    prepared_path = Path((root / "latest-compare").read_text().strip())
    prepared = json.loads(prepared_path.read_text())
    if prepared.get("format") != FORMAT or prepared["ranks"] != ranks:
        raise RuntimeError("comparison preparation format or rank count differs")
    revision = source()
    subprocess.run(["git", "diff", "--exit-code", prepared["source"], "--", "crates",
                    "Cargo.toml", "Cargo.lock"], cwd=REPO, check=True)
    binary = Path(prepared["binary"])
    if digest(binary) != prepared["binary_sha256"]:
        raise RuntimeError("prepared executable changed")
    for package in prepared["packages"].values():
        if digest(Path(package["bundle"]) / "manifest.json") != package["manifest_sha256"]:
            raise RuntimeError("prepared bundle manifest changed")
    directory = root / ("compare-" + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S"))
    directory.mkdir()
    write_new(directory / "start.json", {"prepared": prepared, "runner_source": revision,
                                       "devices": gpu_state(devices), "minutes": minutes})
    deadline = time.monotonic() + minutes * 60
    passed = False
    try:
        if prepared.get("prefill_only", False):
            raise ValueError("prefill-only preparations require replay with an explicit trace")
        for concurrency, arm in comparison_arms(prepared["concurrencies"]):
            case = directory / f"c{concurrency}"
            case.mkdir(exist_ok=True)
            if arm == "tip":
                control = json.loads((case / "control/receipt.json").read_text())
                if control.get("passed") is not True:
                    raise RuntimeError("base repeat did not pass")
            package = prepared["packages"]["tip" if arm == "tip" else "base"]
            config = {"bundle": package["bundle"], "topology": package["topology"],
                      "checkpoint": package["checkpoint"],
                      "trace": str(case / "trace.json"), "output": str(case / arm),
                      "reference": None if arm == "base" else str(case / "base"),
                      "control": str(case / 'control') if arm == 'tip' else None,
                      "concurrency": concurrency, "verify": True, "owned": False,
                      "verify_rows_per_item": prepared["verify_rows_per_item"],
                      "max_sequences": prepared["max_sequences"],
                      "source": {"build": prepared["source"], "binary_sha256": prepared["binary_sha256"]}}
            config['diagnostic'] = diagnostic_options(prepared, package, arm,
                reproduction(prepared_path, config, concurrency, arm, devices, ranks, minutes))
            config_path = case / (arm + ".json")
            write_new(config_path, config)
            env = comparison_environment(os.environ, devices, ranks, package["environment"])
            env["CHOPINFER_BATCH_REPLAY_CONFIG"] = str(config_path)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError("slot deadline reached")
            command([binary, "--ignored", "--exact", TEST, "--test-threads=1", "--nocapture"],
                    case / (arm + ".log"), env, remaining)
            receipt = json.loads((case / arm / "receipt.json").read_text())
            expected_rows = concurrency * (1 + 6 * (2 * prepared["verify_rows_per_item"] - 1))
            if (receipt.get("passed") is not True or receipt.get("steps") != 13
                    or receipt.get("rows") != expected_rows
                    or receipt.get("max_sequences") != prepared["max_sequences"]
                    or receipt.get("maximum_verify_tokens") !=
                    prepared["max_sequences"] * prepared["verify_rows_per_item"]):
                raise RuntimeError("missing complete full and partial verify receipt")
            gpu_state(devices)
        passed = True
    finally:
        after = gpu_state(devices)
        write_new(directory / "result.json", {"passed": passed, "devices_after": after,
                                             "prepared": str(prepared_path), "source": revision})
        print("result:", directory, "passed:", passed, flush=True)


def diagnostic_options(prepared, package, arm, argv):
    return dict(environment=package['environment'],
                declared_differences=prepared.get('declared_differences', []) if arm == 'tip' else [],
                limits=prepared.get('limits', DEFAULT_LIMITS), reproduction=argv)


def reproduction(prepared_path, config, concurrency, arm, devices, ranks, minutes):
    args = ['bash', 'tools/dev/fixed-batch-logits.sh', 'replay', devices,
            '--prepared', str(prepared_path), '--trace', config['trace'],
            '--concurrency', str(concurrency), '--arm', arm, '--ranks', str(ranks),
            '--minutes', str(minutes), '--output', config['output'] + '-replay']
    for key in ('reference', 'control'):
        if config.get(key):
            args += ['--' + key, config[key]]
    return args


def replay_config(prepared, args):
    if prepared.get('format') != FORMAT or prepared['ranks'] != args.ranks:
        raise ValueError('replay requires a matching package-comparison preparation')
    if len(args.concurrency or []) != 1 or args.concurrency[0] not in prepared['concurrencies']:
        raise ValueError('replay requires one prepared concurrency')
    if not args.trace or not args.output:
        raise ValueError('replay requires --trace and a fresh --output')
    for path in (args.trace, args.output, args.reference, args.control):
        if path is not None:
            diagnostics.external(path)
    if args.output.exists():
        raise ValueError('replay output already exists')
    if args.arm != 'base' and args.reference is None:
        raise ValueError('control/tip replay requires --reference')
    if args.arm == 'tip' and args.control is None:
        raise ValueError('tip replay requires the successful baseline --control')
    if args.arm == 'base' and (args.reference or args.control):
        raise ValueError('baseline replay cannot name a reference or control')
    if args.arm == 'control' and args.control:
        raise ValueError('control replay cannot name another control')
    package = prepared['packages']['tip' if args.arm == 'tip' else 'base']
    config = dict(bundle=package['bundle'], topology=package['topology'],
                  checkpoint=package['checkpoint'], trace=str(args.trace),
                  trace_input=True, output=str(args.output),
                  reference=str(args.reference) if args.reference else None,
                  control=str(args.control) if args.control else None,
                  concurrency=args.concurrency[0], verify=True, owned=False,
                  verify_rows_per_item=prepared['verify_rows_per_item'], max_sequences=prepared['max_sequences'],
                  source=dict(build=prepared['source'], binary_sha256=prepared['binary_sha256']))
    if prepared.get("prefill_only", False):
        config.update(prefill_only=True, verify=False)
    config['diagnostic'] = diagnostic_options(prepared, package, args.arm,
        reproduction(args.prepared, config, args.concurrency[0], args.arm, args.devices, args.ranks, args.minutes))
    return config


def replay(args):
    diagnostics.external(args.prepared)
    prepared = json.loads(args.prepared.read_text())
    config = replay_config(prepared, args)
    binary = Path(prepared['binary'])
    if digest(binary) != prepared['binary_sha256']:
        raise ValueError('prepared executable changed')
    for package in prepared['packages'].values():
        if digest(Path(package['bundle']) / 'manifest.json') != package['manifest_sha256']:
            raise ValueError('prepared bundle manifest changed')
    source()
    subprocess.run(['git', 'diff', '--exit-code', prepared['source'], '--',
                    'crates', 'models', 'Cargo.toml', 'Cargo.lock'], cwd=REPO, check=True)
    count = args.ranks
    if (len(args.devices.split(',')) != count or len(set(args.devices.split(','))) != count
            or not all(x.isdigit() for x in args.devices.split(','))):
        raise ValueError('replay device count differs from prepared topology')
    gpu_state(args.devices)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    config_path = args.output.with_name(args.output.name + '.json')
    write_new(config_path, config)
    package = prepared['packages']['tip' if args.arm == 'tip' else 'base']
    env = comparison_environment(os.environ, args.devices, args.ranks, package['environment'])
    env['CHOPINFER_BATCH_REPLAY_CONFIG'] = str(config_path)
    try:
        command([binary, '--ignored', '--exact', TEST, '--test-threads=1', '--nocapture'],
                args.output.with_name(args.output.name + '.log'), env, args.minutes * 60)
    finally:
        if args.reference:
            print(json.dumps(diagnostics.compare(args.reference, args.output, 'fixed-schedule'), indent=2))
        gpu_state(args.devices)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "run", "prepare-compare", "run-compare", "replay"])
    parser.add_argument("devices", nargs="?")
    parser.add_argument("--root", type=Path)
    parser.add_argument("--hc-compare", action="store_true", help="Qwen gather repeat and row-owned HC off/on on identical traces (prepare records it)")
    parser.add_argument("--minutes", type=int, default=25)
    parser.add_argument("--ranks", type=int, choices=sorted(TOPOLOGIES), default=4,
                        help="TP4 (default) or TP2")
    parser.add_argument("--base-bundle", type=Path,
                        help="baseline direct package (a build's prepare --direct output) for prepare-compare")
    parser.add_argument("--tip-bundle", type=Path, help="candidate direct package for prepare-compare")
    parser.add_argument("--checkpoint", type=Path, help="checkpoint both compared packages load")
    parser.add_argument("--topology", help="topology shared by both packages")
    parser.add_argument("--base-env", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--tip-env", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument('--allow-difference', action='append', default=[], metavar='FIELD')
    parser.add_argument('--prepared', type=Path)
    parser.add_argument('--trace', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--control', type=Path)
    parser.add_argument('--arm', choices=['base', 'control', 'tip'], default='tip')
    for key, default in DEFAULT_LIMITS.items():
        parser.add_argument('--' + key.replace('_', '-'), type=int, default=default)
    parser.add_argument("--concurrency", action="append", type=int, choices=range(1, 33),
                        help="comparison cases, default: both C16 and C32")
    parser.add_argument("--max-sequences", type=int, choices=range(1, 33), default=32,
                        help="comparison instance capacity, independent of live trace requests")
    parser.add_argument("--prefill-only", action="store_true",
                        help="prepare bounded supplied one-step prefill traces for replay")
    parser.add_argument("--verify-rows-per-item", type=int, choices=range(2, 9), default=8,
                        help="full verify width; alternating partial steps use one fewer row")
    args = parser.parse_args()
    args.limits = {key: getattr(args, key) for key in DEFAULT_LIMITS}
    if any(v <= 0 for v in args.limits.values()):
        parser.error('capture limits must be positive')
    if args.limits['metadata_bytes'] > diagnostics.MAX_METADATA_LIMIT:
        parser.error('metadata limit must be at most 256 MiB')
    if args.action == 'replay':
        if not args.prepared or not args.devices or not 1 <= args.minutes <= 25:
            parser.error('replay requires --prepared, GPU ordinals and a 1..25 minute budget')
        replay(args)
        return
    if args.root is None:
        if not os.environ.get("CHOPINFER_DATA"):
            parser.error("set CHOPINFER_DATA in local.env or pass --root")
        suffix = "" if args.ranks == 4 else f"-tp{args.ranks}"
        args.root = Path(os.environ["CHOPINFER_DATA"]) / f"c16-host/fixed-batch{suffix}"
    try:
        diagnostics.external(args.root)
    except ValueError as error:
        parser.error(str(error))
    if args.action == "prepare":
        prepare(args.root, args.ranks, args.hc_compare)
    elif args.action == "prepare-compare":
        prepare_compare(args)
    else:
        if not args.devices or not 1 <= args.minutes <= 25:
            parser.error("run requires GPU ordinals and a 1..25 minute budget")
        if args.action == "run-compare":
            run_compare(args.root, args.devices, args.minutes, args.ranks)
        else:
            run(args.root, args.devices, args.minutes, args.ranks, args.hc_compare)


if __name__ == "__main__":
    main()
