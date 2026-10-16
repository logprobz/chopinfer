#!/usr/bin/env python3
"""Prepare CPU-built test executables, then run the KV offload lab on idle GPUs.

The caller frees the GPUs first; this script stops and starts no other
process. Paths and caches come from local.env.
"""
import argparse
import datetime
import json
import os
from pathlib import Path
import resource
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
TARGETS = ("kv_offload_copy", "kv_offload_interference")


def utc_deadline(text):
    value = datetime.datetime.fromisoformat(text.replace("Z", "+00:00"))
    if value.tzinfo is None:
        raise argparse.ArgumentTypeError("end time must include a timezone")
    return value.timestamp()


def run_process(command, env, log, deadline, low_memlock=False):
    remaining = deadline - time.time()
    if remaining <= 0:
        raise TimeoutError("hard end time reached before launch")

    def limit():
        _, hard = resource.getrlimit(resource.RLIMIT_MEMLOCK)
        resource.setrlimit(resource.RLIMIT_MEMLOCK, (min(1 << 20, hard) if hard != -1 else 1 << 20, hard))

    with log.open("xb") as output:
        child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=output,
                                 stderr=subprocess.STDOUT,
                                 preexec_fn=limit if low_memlock else None)
        try:
            code = child.wait(timeout=max(0.001, deadline - time.time()))
        except subprocess.TimeoutExpired:
            # Direct test executable, no shell or cargo child. Kill only its PID.
            child.kill()
            child.wait()
            raise TimeoutError(f"hard end time: killed PID {child.pid}; log {log}")
    if code:
        raise RuntimeError(f"exit {code}: {command}; log {log}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only-m3", action="store_true", help="rerun interference without repeating completed allocation and copy cases")
    parser.add_argument("--prepare", action="store_true", help="CPU only: compile and record executable paths")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--end-time", type=utc_deadline, help="absolute ISO-8601 hard end time")
    parser.add_argument("--authorized-slot", action="store_true", help="confirm exclusive use of the GPUs for this run")
    args = parser.parse_args()
    if args.prepare:
        command = ["cargo", "test", "-p", "kernel-lab", "--release", "--no-run", "--locked", "--offline", "--message-format=json"]
        for target in TARGETS:
            command.extend(["--test", target])
        result = subprocess.run(command, cwd=ROOT, text=True, stdout=subprocess.PIPE, check=True)
        executables = {}
        for line in result.stdout.splitlines():
            message = json.loads(line)
            if message.get("reason") == "compiler-artifact" and message.get("executable") and message["target"]["name"] in TARGETS:
                executables[message["target"]["name"]] = message["executable"]
        if set(executables) != set(TARGETS):
            raise RuntimeError("required test executable missing")
        executables["revision"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(executables, indent=2) + "\n")
        return
    if not args.authorized_slot or args.end_time is None or args.output is None:
        parser.error("GPU run requires --authorized-slot, --end-time and --output")
    if args.end_time <= time.time():
        parser.error("end time is already past")
    manifest = json.loads(args.manifest.read_text())
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if revision != manifest["revision"]:
        parser.error("revision changed since --prepare; prepare again on CPU")
    args.output.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    for name in ("CHOPINFER_KERNEL_GPU", "CHOPINFER_KERNEL_GPUS", "CHOPINFER_KERNEL_SHARED_DEVICE"):
        env.pop(name, None)
    env["CHOPINFER_KVO_LAB"] = "1"
    env["CHOPINFER_KVO_RUN_ID"] = args.output.name
    env["CHOPINFER_KERNEL_TIMEOUT_SECS"] = str(max(1, int(args.end_time - time.time())))
    metadata = {"revision": revision, "end_time_epoch": args.end_time, "memlock": resource.getrlimit(resource.RLIMIT_MEMLOCK), "pid": os.getpid()}
    (args.output / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
    # Report unavailable DIMM data explicitly; never infer it from GPU clocks.
    try:
        run_process(["dmidecode", "--type", "memory"], env, args.output / "dimms.log", min(args.end_time, time.time() + 10))
    except (OSError, RuntimeError) as error:
        (args.output / "dimms-unavailable.txt").write_text(str(error) + "\n")
    jobs = [("m1_pinned_allocation", "0", False), ("m1_memlock_probe", "0", True)]
    jobs += [("m2_copy_rates_single", str(gpu), False) for gpu in range(4)]
    jobs += [("m2_copy_rates_four", "0,1,2,3", False), ("m3_all_reduce_interference", "0,1,2,3", False)]
    if args.only_m3:
        jobs = [job for job in jobs if job[0].startswith("m3")]
    for index, (test, ordinals, low_memlock) in enumerate(jobs):
        selected = env.copy()
        selected["CHOPINFER_KERNEL_GPU" if "," not in ordinals else "CHOPINFER_KERNEL_GPUS"] = ordinals
        target = TARGETS[1] if test.startswith("m3") else TARGETS[0]
        command = [manifest[target], "--exact", test, "--nocapture", "--test-threads=1"]
        log = args.output / f"{index:02d}-{test}.log"
        run_process(command, selected, log, args.end_time, low_memlock)
        # libtest returns success for zero matches, which is not evidence.
        if "1 passed; 0 failed" not in log.read_text():
            raise RuntimeError(f"test did not execute: {log}")
    (args.output / "complete.json").write_text(json.dumps({"jobs": len(jobs), "tests": [job[0] for job in jobs], "finished_epoch": time.time()}) + "\n")


if __name__ == "__main__":
    main()
