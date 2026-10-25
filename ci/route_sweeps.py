"""Run each selected model's route envelope in its own bounded libtest process."""
import concurrent.futures
import fcntl
import hashlib
import json
import os
import shlex
import stat
import tempfile
from pathlib import Path
import re
import subprocess
import sys
import time


def libraries(selection, packages):
    selected = None if selection == ["--workspace"] else set(selection[1::2])
    return [p["name"] for p in packages
            if (selected is None or p["name"] in selected)
            and p.get("manifest_path")
            and Path(p["manifest_path"]).with_name("route-allowlist.toml").is_file()]


def rotation_seed(name, tier, env):
    if "CHOPINFER_ROUTE_SEED" in env:
        seed = int(env["CHOPINFER_ROUTE_SEED"])
        if not 0 <= seed < 2 ** 64:
            raise ValueError("CHOPINFER_ROUTE_SEED must fit an unsigned 64-bit integer")
        return seed
    if tier == "quick":
        return int(time.time()) // 86400
    if not env.get("CHOPINFER_DATA"):
        raise ValueError("CPU rotation needs CHOPINFER_DATA, or an explicit CHOPINFER_ROUTE_SEED")
    directory = Path(env["CHOPINFER_DATA"]) / "route-guard" / "rotation"
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    key = hashlib.sha256(name.encode()).hexdigest()
    state = directory / (key + ".json")
    flags = os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK
    lock_fd = os.open(directory / (key + ".lock"), os.O_RDWR | os.O_CREAT | flags, 0o600)
    with os.fdopen(lock_fd, "r+") as lock:
        if not stat.S_ISREG(os.fstat(lock.fileno()).st_mode):
            raise ValueError("rotation lock must be a regular file")
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            state_fd = os.open(state, os.O_RDONLY | flags)
        except FileNotFoundError:
            seed = 0
        else:
            with os.fdopen(state_fd) as source:
                if not stat.S_ISREG(os.fstat(source.fileno()).st_mode):
                    raise ValueError("rotation counter must be a regular file")
                value = json.loads(source.read(4096))
            seed = value.get("next_seed")
            if value.get("model") != name or type(seed) is not int or not 0 <= seed < 2 ** 64 - 1:
                raise ValueError("invalid route rotation counter")
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", dir=directory, prefix=key, delete=False) as output:
                temporary = Path(output.name)
                json.dump({"model": name, "next_seed": seed + 1}, output)
                output.write("\n")
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, state)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        return seed


def check(name, output, code, elapsed, tier):
    if code:
        raise RuntimeError(f"{name}: route sweep exited {code}")
    count = re.search(r"test result: ok\. ([1-9][0-9]*) passed; 0 failed; 0 ignored;", output)
    if count is None or "cases," not in output or "peak_rss_bytes=" not in output:
        raise RuntimeError(f"{name}: missing executed route cases or peak RSS evidence")
    if tier == "cpu" and ('"rotation_period":' not in output or '"selected_combinations":' not in output):
        raise RuntimeError(f"{name}: missing deterministic CPU rotation evidence")
    budget = 20 if tier == "quick" else 60
    if elapsed > budget:
        print(f"WARNING: ROUTE SWEEP BUDGET EXCEEDED: {name} {tier} took {elapsed:.3f}s "
              f"(budget {budget}s); retain coverage and investigate the regression", file=sys.stderr, flush=True)
    return int(count[1])


def run(binaries, tier, threads=8):
    if not binaries:
        return 0
    if tier not in ("quick", "cpu"):
        raise ValueError("full and exhaustive sweeps require tools/route_guard/sweep.py")
    started = time.monotonic()
    workers = min(4, len(binaries)) if tier == "quick" else 1

    def model(item):
        name, (binary, cwd, env) = item
        seed = rotation_seed(name, tier, env)
        reproduce = ["env", f"CHOPINFER_ROUTE_SWEEP={tier}", f"CHOPINFER_ROUTE_SEED={seed}",
                     "bash", "tools/dev/heavy.sh", "cargo", "test", "--locked", "--offline",
                     "-p", name, "--lib", "route_sweep::", "--", "--nocapture"]
        print(f"route seed: model={name} tier={tier} seed={seed} reproduce={shlex.join(reproduce)}", flush=True)
        before = time.monotonic()
        try:
            done = subprocess.run([binary, "route_sweep::", "--nocapture", "--test-threads", str(threads)],
                                  cwd=cwd, env={**env, "CHOPINFER_ROUTE_SWEEP": tier, "CHOPINFER_ROUTE_SEED": str(seed)},
                                  stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  text=True, timeout=600, check=False)
            elapsed = time.monotonic() - before
            return name, elapsed, done.stdout, done.returncode
        except subprocess.TimeoutExpired as error:
            output = error.stdout or b""
            if isinstance(output, bytes):
                output = output.decode(errors="replace")
            return name, time.monotonic() - before, output, 124

    failed = False
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for name, elapsed, output, code in pool.map(model, sorted(binaries.items())):
            print(f"==> {name} {tier} route envelope ({elapsed:.3f}s)\n{output}", flush=True)
            try:
                count = check(name, output, code, elapsed, tier)
                print(f"route budget: model={name} tier={tier} tests={count} elapsed_seconds={elapsed:.3f}", flush=True)
            except RuntimeError as error:
                print(error, file=sys.stderr, flush=True)
                failed = True
    print(f"route envelopes: tier={tier} models={len(binaries)} parallel_models={workers} "
          f"added_wall_seconds={time.monotonic() - started:.3f}", flush=True)
    return int(failed)
