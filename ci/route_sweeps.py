"""Run each selected model's route envelope in its own bounded libtest process."""
import concurrent.futures
import shlex
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


def dense_seed(env):
    if "CHOPINFER_ROUTE_SEED" in env:
        seed = int(env["CHOPINFER_ROUTE_SEED"])
        if not 0 <= seed < 2 ** 64:
            raise ValueError("CHOPINFER_ROUTE_SEED must fit an unsigned 64-bit integer")
        return seed
    return int(time.time()) // 86400


def check(name, output, code, elapsed, tier):
    if code:
        raise RuntimeError(f"{name}: route sweep exited {code}")
    count = re.search(r"test result: ok\. ([1-9][0-9]*) passed; 0 failed; 0 ignored;", output)
    if count is None:
        raise RuntimeError(f"{name}: missing executed route tests")
    if tier == "quick" and re.search(r"^\S+: quick tier represented by \S+$", output, re.MULTILINE):
        # The shared envelope explicitly delegates equivalent quick profiles.
        # run() still requires real case/RSS evidence from this model.
        return int(count[1])
    if tier != "quick" and re.search(r"^\S+: route sweep covers the quick tier only$", output, re.MULTILINE):
        # A candidate's own sweep may stop at the quick tier. run() still
        # requires real case/RSS evidence from another profile of the model.
        return int(count[1])
    if "cases," not in output or "peak_rss_bytes=" not in output:
        raise RuntimeError(f"{name}: missing executed route cases or peak RSS evidence")
    if tier == "cpu" and any(field not in output for field in (
            '"dense_rotation_period":', '"boundary_rows":', '"dense_rows":')):
        raise RuntimeError(f"{name}: missing deterministic boundaries or dense probe evidence")
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
        seed = dense_seed(env)
        child_env = {**env, "CHOPINFER_ROUTE_SWEEP": tier, "CHOPINFER_ROUTE_SEED": str(seed)}
        try:
            listing = subprocess.check_output([binary, "route_sweep::", "--list"],
                                              cwd=cwd, env=child_env, text=True, timeout=60)
            profiles = [line.removesuffix(": test") for line in listing.splitlines()
                        if line.endswith(": test") and "route_sweep::" in line]
            if not profiles or len(set(profiles)) != len(profiles):
                raise ValueError("missing or duplicate route profile tests")
        except (subprocess.SubprocessError, ValueError) as error:
            return [(name, 0, str(error), 1)]
        results = []
        # VmHWM is a process lifetime high-water mark. Isolate each profile so
        # one completed sweep cannot consume another profile's memory budget.
        for profile in profiles:
            reproduce = ["env", f"CHOPINFER_ROUTE_SWEEP={tier}", f"CHOPINFER_ROUTE_SEED={seed}",
                         "bash", "tools/dev/heavy.sh", "cargo", "test", "--locked", "--offline",
                         "-p", name, "--lib", profile, "--", "--exact", "--nocapture"]
            print(f"route seed: model={name} tier={tier} seed={seed} reproduce={shlex.join(reproduce)}", flush=True)
            before = time.monotonic()
            try:
                done = subprocess.run([binary, profile, "--exact", "--nocapture", "--test-threads", str(threads)],
                                      cwd=cwd, env=child_env, stdin=subprocess.DEVNULL,
                                      stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                      text=True, timeout=600, check=False)
                results.append((f"{name}/{profile}", time.monotonic() - before, done.stdout, done.returncode))
            except subprocess.TimeoutExpired as error:
                output = error.stdout or b""
                if isinstance(output, bytes):
                    output = output.decode(errors="replace")
                results.append((f"{name}/{profile}", time.monotonic() - before, output, 124))
        if not any(code == 0 and "cases," in output and "peak_rss_bytes=" in output
                   for _, _, output, code in results):
            results.append((name, 0, "model has no executed route cases or peak RSS evidence", 1))
        return results

    failed = False
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for results in pool.map(model, sorted(binaries.items())):
            package = results[0][0].partition("/")[0]
            model_failed, tests, total = False, 0, 0
            for name, elapsed, output, code in results:
                profile = name.partition("/")[2]
                print(f"==> {package} {tier} route envelope ({elapsed:.3f}s) profile={profile}\n{output}", flush=True)
                total += elapsed
                try:
                    count = check(name, output, code, elapsed, tier)
                    tests += count
                    print(f"route profile budget: model={package} profile={profile} tier={tier} tests={count} elapsed_seconds={elapsed:.3f}", flush=True)
                except RuntimeError as error:
                    print(error, file=sys.stderr, flush=True)
                    model_failed = failed = True
            # The inventory's package verdict covers every profile together.
            if not model_failed:
                print(f"route budget: model={package} tier={tier} tests={tests} elapsed_seconds={total:.3f}", flush=True)
    print(f"route envelopes: tier={tier} models={len(binaries)} parallel_models={workers} "
          f"added_wall_seconds={time.monotonic() - started:.3f}", flush=True)
    return int(failed)
