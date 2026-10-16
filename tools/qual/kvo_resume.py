"""Compare preemption accuracy on reserved, idle GPUs.

CPU preflight: --check. A run needs --go with idle devices and an end time;
this tool never acquires locks or stops existing servers. Supply NUL-separated
TP4 serve arguments, such as the serve words a tools/dev/*-redeploy.sh
--dry-run prints: `serve --model CHECKPOINT ...`, or `serve --profile
FILE#NAME --bundle PACKAGE_DIR --checkpoint CHECKPOINT ...` for an explicit
package. For `--model` the family helper
installed beside --host (chopinfer-model-mimo26 or chopinfer-model-qwen38)
prepares the package the server will select, so the preflight reads its
manifest. Each model gets half the default one-hour budget. Only child PIDs
are stopped.
"""

import argparse
import hashlib
from decimal import Decimal
from datetime import datetime, timezone
import json
import os
import re
from pathlib import Path
import signal
import subprocess
import struct
import sys
import time
import urllib.request

from common import ROOT
from preempt_compare import compare_records, prompts


def replace_option(arguments, option, value=None):
    result = list(arguments)
    while option in result:
        index = result.index(option)
        del result[index:index + 2]
    if value is not None:
        result.extend((option, str(value)))
    return result


# The family helper, installed beside the host binary, for each args file.
HELPERS = {"mimo": "chopinfer-model-mimo26", "qwen38": "chopinfer-model-qwen38"}


def option(arguments, flag):
    return arguments[arguments.index(flag) + 1] if flag in arguments else None


def profile_file(reference):
    """The serving profile an explicit `--profile FILE#NAME` names."""
    path, _, name = reference.rpartition("#")
    if not path or not name:
        raise ValueError(f"--profile {reference}: expected FILE#NAME")
    path = Path(path) if Path(path).is_absolute() else ROOT / path
    return json.loads(path.read_text())["profiles"][name]


def require_tp4(args):
    """Refuses anything but TP4 on four distinct devices. `serve --model`
    serves TP4 unless --tensor-parallel-size says otherwise."""
    if (topology := option(args, "--topology")) is not None:
        tp4 = topology.endswith("-tp4")
    elif (reference := option(args, "--profile")) is not None:
        tp4 = profile_file(reference).get("tensor_parallel") == 4
    elif "--model" in args:
        tp4 = option(args, "--tensor-parallel-size") in (None, "4")
    else:
        raise ValueError("serve arguments name no --model, --profile or --topology")
    if not tp4 or ("--model" in args and option(args, "--tensor-parallel-size") not in (None, "4")):
        raise ValueError("the runner requires TP4 on four distinct devices")


def serve_arguments(path, port, optimistic, default_tokens=0):
    """`default_tokens` is the draft depth the serving profile supplies when
    the arguments carry no --speculative-tokens."""
    args = [word for word in path.read_bytes().decode().split("\0") if word]
    if not args or args[0] != "serve":
        raise ValueError(f"{path}: expected NUL-separated serve arguments")
    require_tp4(args)
    draft_tokens = int(option(args, "--speculative-tokens") or default_tokens)
    for key, value in (("--listen", f"127.0.0.1:{port}"), ("--prefix-cache", "on"),
                       ("--admission", "optimistic" if optimistic else "reserved"),
                       ("--debug-preempt-every", 16 if optimistic else None),
                       ("--max-preemptions", 3 if optimistic else None),
                       ("--speculation", "fixed" if draft_tokens > 0 else None)):
        args = replace_option(args, key, value)
    # CUDA_VISIBLE_DEVICES maps the selected devices to these four ordinals.
    return replace_option(args, "--devices", "0,1,2,3")


RESUME_COUNTERS = {
    "exact_resumes": "chop_prefix_restore_resumed_decoding_total",
    "capture_failures": "chop_prefix_cache_capture_failures_total",
    "restore_failures": "chop_prefix_cache_restore_failures_total",
}


def resume_metric_deltas(before, after):
    def read(text, name):
        samples = re.findall(r"^" + re.escape(name) + r" (\S+)$", text, re.MULTILINE)
        if len(samples) != 1:
            raise ValueError(f"missing or duplicate resume metric {name}")
        value = Decimal(samples[0])
        if not value.is_finite() or value < 0 or value != value.to_integral_value():
            raise ValueError(f"invalid resume counter {name}: {value}")
        return int(value)
    result = {key: read(after, name) - read(before, name) for key, name in RESUME_COUNTERS.items()}
    if any(value < 0 for value in result.values()):
        raise ValueError("resume counters reset during a cohort")
    return result


def prepare_direct(arguments, helper):
    """The direct package `serve --model` selects for these arguments, as the
    family helper's `prepare --direct` reports it (no GPU work)."""
    command = [str(helper), "prepare", "--direct", "--model", option(arguments, "--model")]
    for serve_flag, helper_flag in (("--tensor-parallel-size", "--tensor-parallel-size"),
                                    ("--model-cache-dir", "--cache-dir"),
                                    ("--activation-policy", "--activation-policy"),
                                    ("--modalities", "--modalities")):
        if (value := option(arguments, serve_flag)) is not None:
            command += [helper_flag, value]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=1800)
    if result.returncode:
        raise ValueError(f"{helper} prepare --direct failed: {result.stderr.strip()[-2000:]}")
    lines = [line for line in result.stdout.splitlines() if line.startswith("{")]
    if not lines:
        raise ValueError(f"{helper} prepare --direct printed no JSON result")
    record = json.loads(lines[-1])
    package = record.get("package", record)
    if not package.get("bundle"):
        raise ValueError(f"{helper} prepare --direct named no package directory")
    return package


def package_manifest(arguments, helper=None):
    """The manifest path of the package these serve arguments load, with the
    helper's prepared record for `serve --model` (None otherwise)."""
    if "--install-root" in arguments:
        raise ValueError("--install-root and installed bundle IDs are gone; use serve --model CHECKPOINT, "
                         "or --bundle PACKAGE_DIR --checkpoint CHECKPOINT")
    package = None
    if "--model" in arguments:
        if helper is None:
            raise ValueError("serve --model needs the family helper to find its package")
        package = prepare_direct(arguments, helper)
        bundle = Path(package["bundle"])
    elif "--bundle" in arguments:
        bundle = Path(arguments[arguments.index("--bundle") + 1])
        if not bundle.is_dir():
            raise ValueError(f"bundle {bundle} is not a package directory; a bundle ID needs --install-root, "
                             "which serving no longer takes: use serve --model CHECKPOINT, "
                             "or --bundle PACKAGE_DIR --checkpoint CHECKPOINT")
    else:
        raise ValueError("serve arguments name neither --model nor a --bundle package directory")
    return bundle / "manifest.json", package


def package_drafter(arguments, helper=None):
    manifest_path, package = package_manifest(arguments, helper)
    raw = manifest_path.read_bytes()
    manifest = json.loads(raw)
    speculation = manifest.get("speculation") or {}
    if package is not None:
        settings = (package.get("profile") or {}).get("profile") or {}
    elif (reference := option(arguments, "--profile")) is not None:
        settings = profile_file(reference)
    else:
        settings = {}
    result = {"kind": "none", "manifest": str(manifest_path.resolve()),
              "manifest_sha256": hashlib.sha256(raw).hexdigest(), "speculation": speculation,
              "profile_tokens": int((settings.get("speculation") or {}).get("tokens", 0))}
    if package is not None:
        result["package"] = {key: package.get(key) for key in
                             ("bundle", "package_identity", "checkpoint_identity", "metadata_identity")}
        result["profile"] = (package.get("profile") or {}).get("name")
    if speculation.get("bundle_drafter"):
        if not speculation.get("drafter_kind"):
            raise ValueError(f"{manifest_path}: the package declares a bundle drafter without speculation.drafter_kind")
        result["kind"] = speculation["drafter_kind"]
    return result


def source_revision():
    if (ROOT / "REVISION").is_file():
        return (ROOT / "REVISION").read_text().strip()
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def bitwise_c1(reserved, preempted):
    # Compare the API's parsed logprob bits, including signed zero, along
    # with tokens, alternatives, finish reasons and client-visible usage.
    def bits(value):
        if isinstance(value, float):
            return struct.pack("!d", value)
        if isinstance(value, list):
            return [bits(item) for item in value]
        if isinstance(value, dict):
            return {key: bits(item) for key, item in value.items()}
        return value
    fields = ("tokens", "logprobs", "top", "finish_reason", "usage")
    changed = [left["name"] for left, right in zip(reserved["requests"], preempted["requests"])
               if any(bits(left[field]) != bits(right[field]) for field in fields)]
    return {"identical": not changed, "different_requests": changed,
            "scope": "API token strings, logprob bits, top alternatives, finish reasons and usage"}


def remaining(deadline):
    seconds = deadline - time.monotonic()
    if seconds <= 0:
        raise TimeoutError("GPU slot budget exhausted")
    return seconds


def stop_child(child):
    if child is not None and child.poll() is None:
        child.terminate()
        try:
            child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait(timeout=10)


def gpu_state(devices):
    result = subprocess.run(
        ["nvidia-smi", "-i", devices, "--query-compute-apps=pid,used_gpu_memory",
         "--format=csv,noheader"], check=True, capture_output=True, text=True, timeout=10)
    return result.stdout.strip()


def run_command(command, deadline, log, env=None):
    with log.open("w") as output:
        subprocess.run(command, cwd=ROOT, env=env, stdout=output, stderr=subprocess.STDOUT,
                       timeout=remaining(deadline), check=True)


def wait_ready(server, url, deadline):
    while True:
        if server.poll() is not None:
            raise RuntimeError(f"server exited with {server.returncode}")
        try:
            for path in ("/health", "/v1/models"):
                with urllib.request.urlopen(url + path, timeout=min(2, remaining(deadline))) as response:
                    if response.status != 200:
                        raise RuntimeError(f"{path}: HTTP {response.status}")
            return
        except (OSError, TimeoutError):
            time.sleep(min(1, remaining(deadline)))


def assess(records):
    names = None
    for label, record in records.items():
        rows = record["requests"]
        current = [row["name"] for row in rows]
        if len(rows) != 36 or len(set(current)) != 36 or (names is not None and current != names):
            raise ValueError(f"{label}: missing, duplicate, or reordered requests")
        names = current
        if record["output_tokens"] != 96 or any("error" in row for row in rows):
            raise ValueError(f"{label}: incomplete accuracy workload")
        metrics = record.get("resume_metrics")
        if metrics is None or set(metrics) != set(RESUME_COUNTERS):
            raise ValueError(f"{label}: missing exact-resume evidence")
        if any(not isinstance(value, int) or value < 0 for value in metrics.values()):
            raise ValueError(f"{label}: invalid resume metric delta")
        if metrics["capture_failures"] != 0 or metrics["restore_failures"] != 0:
            raise ValueError(f"{label}: capture or restore failed: {metrics}")
        if label.startswith("optimistic") and metrics["exact_resumes"] <= 0:
            raise ValueError(f"{label}: exact resume path was not exercised")
        count = record.get("preemptions")
        if count is None or (count <= 0 if label.startswith("optimistic") else count != 0):
            raise ValueError(f"{label}: unexpected preemption count {count}")
    c1 = compare_records(records["reserved-c1"], records["optimistic-c1"], 16, 0.5)
    baseline = compare_records(records["reserved-c16"], records["repeat-c16"], 16, 0.5, True)
    c16 = compare_records(records["reserved-c16"], records["optimistic-c16"], 16, 0.5)
    # Paired baselines replace the literal C16 gate after even reserved pairs
    # diverged before token 16. Keep both raw reports for the required review.
    return {"c1": c1, "c1_bitwise": bitwise_c1(records["reserved-c1"], records["optimistic-c1"]),
            "resume_metrics": {label: record["resume_metrics"] for label, record in records.items()},
            "c16_reserved_pair": baseline, "c16_preempted_pair": c16,
            "c16_assessment": "requires comparison against the reserved pair",
            "gpu_bitwise_qualification": "pending: no serving-step GPU bitwise run"}


def run_model(model, args, directory, deadline):
    records = {}
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.devices)
    for arm, concurrencies in (("reserved", (1, 16)), ("repeat", (16,)), ("optimistic", (1, 16))):
        command = [str(args.host)] + serve_arguments(args.args_dir / f"{model}.args", args.port, arm == "optimistic",
                                                     args.profile_tokens[model])
        (directory / f"{arm}-command.json").write_text(json.dumps(command, indent=2) + "\n")
        child = None
        try:
            with (directory / f"{arm}-server.log").open("w") as output:
                child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=output, stderr=subprocess.STDOUT)
            wait_ready(child, f"http://127.0.0.1:{args.port}", deadline)
            for concurrency in concurrencies:
                label = f"{arm}-c{concurrency}"
                receipts = directory / label
                receipts.mkdir()
                metrics_url = f"http://127.0.0.1:{args.port}/metrics"
                with urllib.request.urlopen(metrics_url, timeout=min(10, remaining(deadline))) as response:
                    (directory / f"{label}-metrics-before.txt").write_bytes(response.read())
                run_command([sys.executable, "-B", str(ROOT / "tools/qual/preempt_compare.py"), "record",
                             "--url", f"http://127.0.0.1:{args.port}", "--arm", arm,
                             "--concurrency", str(concurrency), "--output-tokens", "96",
                             "--timeout", str(max(1, int(remaining(deadline))))], deadline,
                            directory / f"{label}.log", dict(env, CHOPINFER_QUAL_RECEIPTS=str(receipts)))
                with urllib.request.urlopen(metrics_url, timeout=min(10, remaining(deadline))) as response:
                    (directory / f"{label}-metrics-after.txt").write_bytes(response.read())
                paths = list(receipts.glob("*.json"))
                if len(paths) != 1:
                    raise RuntimeError(f"{label}: expected exactly one receipt, found {len(paths)}")
                records[label] = json.loads(paths[0].read_text())
                records[label]["resume_metrics"] = resume_metric_deltas(
                    (directory / f"{label}-metrics-before.txt").read_text(),
                    (directory / f"{label}-metrics-after.txt").read_text())
                print(f"{model} {label}: recorded", flush=True)
        finally:
            stop_child(child)
    report = assess(records)
    (directory / "assessment.json").write_text(json.dumps(report, indent=2) + "\n")
    return report["c1"]["passed"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", type=Path, required=True, help="host binary built from this revision")
    parser.add_argument("--args-dir", type=Path, required=True, help="prepared mimo.args and qwen38.args")
    parser.add_argument("--models", choices=("both", "mimo", "qwen38"), default="both")
    parser.add_argument("--devices", required=True, help="four idle devices")
    parser.add_argument("--seconds", type=int, default=3600)
    parser.add_argument("--port", type=int, default=8439)
    parser.add_argument("--out", type=Path, default=Path(os.environ.get("CHOPINFER_DATA", Path.home() / "chopinfer-data")) / "kvo-resume")
    parser.add_argument("--check", action="store_true", help="CPU only: check packages and frozen inputs")
    args = parser.parse_args()
    devices = args.devices.split(",")
    if len(devices) != 4 or len(set(devices)) != 4 or not all(device.isdecimal() for device in devices):
        parser.error("--devices must name four distinct GPU ordinals")
    if not 60 <= args.seconds <= 3600 or not 1 <= args.port <= 65535:
        parser.error("--seconds must be 60..3600 and --port 1..65535")
    args.host = args.host.resolve(strict=True)
    args.args_dir = args.args_dir.resolve(strict=True)
    args.out = args.out.resolve()
    if not args.out.is_relative_to("/data") or args.out.is_relative_to(ROOT):
        parser.error("--out must be under /data and outside the worktree")
    models = ("mimo", "qwen38") if args.models == "both" else (args.models,)
    assert len(prompts()) == 36
    drafters = {}
    for model in models:
        command = serve_arguments(args.args_dir / f"{model}.args", args.port, False)
        drafters[model] = package_drafter(command, args.host.parent / HELPERS[model])
    # The profile's draft depth applies when the arguments carry none; the
    # arms pin it with fixed speculation.
    args.profile_tokens = {model: drafters[model]["profile_tokens"] for model in models}
    if args.check:
        print("CPU preflight passed; GPU execution needs --go")
        return 0
    if busy := gpu_state(args.devices):
        raise RuntimeError(f"assigned GPUs are not empty: {busy}")
    directory = args.out / datetime.now(timezone.utc).strftime("gpu-%Y%m%dT%H%M%S%fZ")
    directory.mkdir(parents=True)
    # Leave cleanup time inside the requested time budget.
    deadline = time.monotonic() + args.seconds - 30
    (directory / "provenance.json").write_text(json.dumps({
        "host": str(args.host), "host_sha256": hashlib.sha256(args.host.read_bytes()).hexdigest(),
        "devices": args.devices, "seconds": args.seconds,
        "revision": source_revision(),
        "host_revision": (args.host.parent / "REVISION").read_text().strip(),
        "drafters": drafters,
    }, indent=2) + "\n")
    def interrupted(*_):
        raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM, interrupted)
    passed = True
    try:
        for model in models:
            model_dir = directory / model
            model_dir.mkdir()
            model_deadline = min(deadline, time.monotonic() + args.seconds / len(models))
            passed = run_model(model, args, model_dir, model_deadline) and passed
    finally:
        busy = gpu_state(args.devices)
        (directory / "gpu-after.txt").write_text(busy + "\n")
        if busy:
            raise RuntimeError(f"GPUs are not empty after cleanup: {busy}")
        print(f"GPUs {args.devices} are empty. Logs: {directory}", flush=True)
    print("Exact resume gates and C1 accuracy passed; inspect C1 bitwise and C16 assessments" if passed else "C1 gate failed")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
