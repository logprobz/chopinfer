"""Wait for an owned server's exit and NVIDIA compute accounting to clear.

No CUDA context is created. UUID enumeration is used only to translate CUDA
ordinals, which need not equal nvidia-smi indices.
"""

import csv
import ctypes
import io
import os
from pathlib import Path
import select
import subprocess
import sys
import time
import uuid


def cuda_uuids():
    cuda = ctypes.CDLL("libcuda.so.1")

    def check(result):
        if result:
            raise RuntimeError(f"CUDA device enumeration failed: {result}")

    check(cuda.cuInit(0))
    count = ctypes.c_int()
    check(cuda.cuDeviceGetCount(ctypes.byref(count)))
    result = []
    for ordinal in range(count.value):
        device = ctypes.c_int()
        check(cuda.cuDeviceGet(ctypes.byref(device), ordinal))
        value = (ctypes.c_ubyte * 16)()
        check(cuda.cuDeviceGetUuid(ctypes.byref(value), device))
        result.append("GPU-" + str(uuid.UUID(bytes=bytes(value))))
    return result


def query(kind, fields, deadline):
    result = subprocess.run(
        ["nvidia-smi", f"--query-{kind}={fields}", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True,
        timeout=max(0.001, deadline - time.monotonic()),
    )
    return [list(map(str.strip, row)) for row in csv.reader(io.StringIO(result.stdout)) if row]


def selected_uuids(gpus, serve_args, deadline):
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    # Full or unambiguous GPU UUIDs already specify the visible order. Other
    # forms go through CUDA itself, including numeric lists and device order.
    if visible and all(value.startswith("GPU-") for value in visible):
        ordered = []
        for value in visible:
            matches = [gpu for gpu in gpus if gpu.startswith(value)]
            if len(matches) != 1 or matches[0] in ordered:
                raise ValueError(f"invalid CUDA_VISIBLE_DEVICES UUID: {value}")
            ordered.append(matches[0])
    else:
        result = subprocess.run(
            [sys.executable, "-B", str(Path(__file__).resolve()), "--cuda-uuids"],
            check=True, capture_output=True, text=True,
            timeout=max(0.001, deadline - time.monotonic()),
        )
        ordered = result.stdout.splitlines()
    devices = None
    for index, argument in enumerate(serve_args):
        if argument == "--devices":
            devices = serve_args[index + 1]
        elif argument.startswith("--devices="):
            devices = argument.split("=", 1)[1]
    if devices is not None:
        ordinals = [int(value) for value in devices.split(",")]
        if not ordinals or len(set(ordinals)) != len(ordinals) or any(
            value < 0 or value >= len(ordered) for value in ordinals
        ):
            raise ValueError("--devices must name distinct visible CUDA ordinals")
        ordered = [ordered[value] for value in ordinals]
    if not ordered or any(value not in gpus for value in ordered):
        raise ValueError("visible CUDA GPUs could not be mapped to nvidia-smi UUIDs")
    return ordered


def wait_released(pid, pidfd, serve_args, deadline):
    poller = select.poll()
    if pidfd is not None:
        poller.register(pidfd, select.POLLIN)
    selected = None
    pending = f"PID {pid}: GPU accounting not yet queried"
    last_error = ""
    while True:
        if time.monotonic() >= deadline:
            raise TimeoutError(f"GPU release timeout: {pending}{last_error}")
        try:
            rows = query("gpu", "index,uuid,memory.used,memory.total", deadline)
            gpus = {row[1]: row for row in rows}
            if selected is None:
                selected = selected_uuids(gpus, serve_args, deadline)
            apps = query("compute-apps", "pid,used_memory,gpu_uuid", deadline)
            if any(len(row) != 3 or int(row[0]) <= 0 for row in apps):
                raise ValueError("invalid NVIDIA compute process row")
            last_error = ""
            pending_rows = []
            exited = pidfd is None or bool(poller.poll(0))
            for gpu in selected:
                index, _, used, total = gpus[gpu]
                holders = [f"PID {row[0]} ({row[1]} MiB)" for row in apps if row[2] == gpu]
                old_present = any(row[0] == str(pid) and row[2] == gpu for row in apps)
                if old_present or not exited:
                    pending_rows.append(
                        f"GPU {index} ({gpu}), used {used}/{total} MiB, "
                        f"compute processes: {', '.join(holders) or 'none'}"
                    )
            if exited and not pending_rows:
                return
            pending = f"old PID {pid} {'exited' if exited else 'still running'}; " + "; ".join(pending_rows)
        except (subprocess.SubprocessError, OSError, ValueError, KeyError, IndexError) as error:
            # Fail closed, including transient accounting failures. Keep the
            # last useful GPU/PID snapshot in the timeout diagnostic.
            last_error = f"; GPU query failed: {error}"
        time.sleep(min(0.5, max(0, deadline - time.monotonic())))


if __name__ == "__main__":
    if sys.argv[1:] != ["--cuda-uuids"]:
        raise SystemExit("internal helper: expected --cuda-uuids")
    print("\n".join(cuda_uuids()))
