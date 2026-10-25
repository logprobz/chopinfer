"""Run Rust tests for the workspace or selected packages.

cargo-nextest runs each test in its own process and tests from different
binaries at once, within the groups in .config/nextest.toml. A package whose
Cargo.toml sets package.metadata.chopinfer.lib-tests-in-one-process keeps its
library tests in one libtest process, as cargo test runs them, beside nextest
and from the same build. CHOPINFER_CI_SERIAL_LIBRARIES=1 runs these libraries
one at a time before nextest to bound their combined memory. Nextest does not
run doctests, so cargo test --doc
runs them afterwards. Where nextest is missing, cargo test runs the same
tests one binary at a time, and the run says so."""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import tomllib
from xml.etree import ElementTree

import route_sweeps

PINNED = "0.9.146"
LOCKED = ["--locked", "--offline"]


def high_memory(selection):
    """Exact high-memory overrides, with their reason in the group name.

    Keep model-specific identities in nextest's configuration. Both ordinary
    runners and the exclusive pass derive their selection from this registry.
    """
    config = tomllib.loads((Path(__file__).resolve().parents[1] / ".config/nextest.toml").read_text())
    entries = []
    for override in config["profile"]["ci"].get("overrides", []):
        group = override.get("test-group", "")
        if not group.startswith("high-memory-"):
            continue
        match = re.fullmatch(r"\(binary_id\(=([\w:-]+)\) & test\(=([\w:]+)\)\)", override["filter"])
        if not match or config["test-groups"][group]["max-threads"] != 1:
            raise ValueError("high-memory overrides require an exact binary/test pair and a serial group")
        entries.append((match[1], match[2], group.removeprefix("high-memory-").replace("-", " ")))
    if not entries or len({entry[:2] for entry in entries}) != len(entries):
        raise ValueError("high-memory registry is empty or contains duplicate identities")
    return [entry for entry in entries
            if selection == ["--workspace"] or entry[0].split("::")[0] in selection[1::2]]


def memory_filter(entries):
    return " | ".join(f"(binary_id(={binary}) & test(={name}))" for binary, name, _ in entries)


def save_junit(name, since, expected=None):
    """Keep each pass before nextest overwrites its profile's report."""
    source = Path(__file__).resolve().parents[1] / "target/nextest/ci/junit.xml"
    if not source.is_file() or source.stat().st_mtime < since:
        if expected is not None or os.environ.get("CHOPINFER_CI_JUNIT_DIR"):
            raise ValueError("high-memory tests produced no current JUnit report")
        return
    if expected is not None:
        cases = ElementTree.parse(source).getroot().iter("testcase")
        actual = [(case.get("classname"), case.get("name")) for case in cases]
        if sorted(actual) != sorted((binary, test) for binary, test, _ in expected):
            raise ValueError("high-memory JUnit identities differ from the declared cases")
    if directory := os.environ.get("CHOPINFER_CI_JUNIT_DIR"):
        destination = Path(directory)
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination / (name + ".xml"))


def nextest_version():
    try:
        output = subprocess.run(["cargo", "nextest", "--version"], capture_output=True, text=True, check=False)
    except OSError:
        return None
    found = re.match(r"cargo-nextest (\d+\.\d+\.\d+)", output.stdout)
    return found.group(1) if output.returncode == 0 and found else None


def workspace_packages():
    return json.loads(subprocess.check_output(
        ["cargo", "metadata", *LOCKED, "--no-deps", "--format-version", "1"]))["packages"]


def library_packages(names, packages=None):
    """The selected packages that have a library target, which cargo test --doc needs."""
    libraries = {package["name"] for package in packages or workspace_packages()
                 if any(kind in ("lib", "rlib", "dylib", "proc-macro") for target in package["targets"]
                        for kind in target["kind"])}
    return [name for name in names if name in libraries]


def one_process_libraries(selection, packages=None):
    """Selected packages whose library tests run in one libtest process."""
    declared = [package["name"] for package in packages or workspace_packages()
                if ((package.get("metadata") or {}).get("chopinfer") or {}).get("lib-tests-in-one-process")]
    return declared if selection == ["--workspace"] else [name for name in declared if name in selection[1::2]]


def commands(selection, runner, threads, profile="ci", one_process=(), binaries_metadata=None, skip_routes=False,
             skip_high_memory=False, route_tests=()):
    """The commands that run `selection` (["--workspace"] or -p pairs).
    Library tests of the `one_process` packages are left out of nextest."""
    if runner == "cargo":
        skipped = (list(route_tests) if skip_high_memory else ["route_sweep::"] if skip_routes else []) + (
            [name for _, name, _ in high_memory(selection)] if skip_high_memory else [])
        return [["cargo", "test", *selection, *LOCKED,
                 *(["--", *(["--exact"] if skip_high_memory else []),
                    *[arg for name in skipped for arg in ("--skip", name)]] if skipped else [])]]
    # The metadata already names the built selection; nextest refuses Cargo
    # build options when reusing it. Doctests still use the original selection.
    build_selection = [] if binaries_metadata else selection
    build_options = ["--binaries-metadata", binaries_metadata] if binaries_metadata else LOCKED
    run = ["cargo", "nextest", "run", *build_selection, "--profile", profile, "--no-tests", "warn", *build_options]
    if threads:
        run[3:3] = ["--test-threads", str(threads)]
    excluded = [f"binary_id(={name})" for name in one_process]
    if skip_routes:
        excluded.append("test(/route_sweep::/)")
    if skip_high_memory and (entries := high_memory(selection)):
        excluded.append(memory_filter(entries))
    if excluded:
        run += ["-E", "not (" + " | ".join(excluded) + ")"]
    doc = selection if selection == ["--workspace"] else [
        argument for name in library_packages(selection[1::2]) for argument in ("-p", name)]
    return [run] + ([["cargo", "test", *doc, "--doc", *LOCKED]] if doc else [])


def choose(requested):
    """(runner, note) for CHOPINFER_RUST_TEST_RUNNER=auto|nextest|cargo."""
    if requested == "cargo":
        return "cargo", "rust tests: cargo test (requested); one test binary at a time"
    version = nextest_version()
    if version is None:
        if requested == "nextest":
            raise SystemExit("rust tests: cargo-nextest is required but not installed; see docs/contributing.md")
        return "cargo", ("rust tests: cargo-nextest is not installed, so cargo test runs one test binary at a "
                         f"time, which is slower; install cargo-nextest {PINNED} (docs/contributing.md)")
    note = f"rust tests: cargo-nextest {version}"
    if version != PINNED:
        note += f" (CI pins {PINNED})"
    return "nextest", note


def library_binaries(selection, names, packages, metadata_path):
    """{package: (binary path, working directory, environment)} for the
    library test binaries nextest builds for `names`, run as nextest would:
    from the package directory, with the build's library paths and build
    script variables."""
    listed = json.loads(subprocess.check_output(
        ["cargo", "nextest", "list", *selection, "--list-type", "binaries-only", "--message-format", "json",
         *LOCKED]))
    with open(metadata_path, "w", encoding="utf-8") as metadata:
        json.dump(listed, metadata)
    meta = listed["rust-build-meta"]
    target = meta["target-directory"]
    library_path = []
    for directory in meta.get("base-output-directories", []):
        library_path += [os.path.join(target, directory), os.path.join(target, directory, "deps")]
    library_path += [os.path.join(target, path) for path in meta.get("linked-paths", [])]
    directories = {package["name"]: os.path.dirname(package["manifest_path"]) for package in packages}
    binaries = {}
    for binary in listed["rust-binaries"].values():
        name = binary["binary-id"]
        if name in names and binary["kind"] == "lib":
            env = dict(os.environ, CARGO_MANIFEST_DIR=directories[name], CARGO_PKG_NAME=name)
            env.update(meta.get("build-script-info", {}).get(binary["package-id"], {}).get("envs", {}))
            env["LD_LIBRARY_PATH"] = os.pathsep.join(
                library_path + [path for path in [os.environ.get("LD_LIBRARY_PATH")] if path])
            binaries[name] = (binary["binary-path"], directories[name], env)
    missing = sorted(set(names) - set(binaries))
    if missing:
        raise SystemExit("rust tests: no library test binary for " + ", ".join(missing))
    return binaries


def run_nextest(selection, threads, skip_high_memory=False):
    """Nextest, one-process libraries beside or before it, then doctests."""
    packages = workspace_packages()
    one_process = one_process_libraries(selection, packages)
    serial_libraries = os.environ.get("CHOPINFER_CI_SERIAL_LIBRARIES") == "1"
    routes = route_sweeps.libraries(selection, packages)
    run, *doc = commands(selection, "nextest", threads, one_process=one_process, skip_routes=bool(routes),
                         skip_high_memory=skip_high_memory)
    children = []
    with tempfile.TemporaryDirectory(prefix="chopinfer-rust-tests-") as logs:
        try:
            route_status = 0
            if one_process or routes:
                # Reuse the listed build so Cargo cannot replace a running test's executable.
                metadata = os.environ.get("CHOPINFER_CI_BINARIES_METADATA") or os.path.join(logs, "binaries.json")
                binaries = library_binaries(selection, sorted(set(one_process) | set(routes)), packages, metadata)
                run = commands(selection, "nextest", threads, one_process=one_process,
                               binaries_metadata=metadata, skip_routes=bool(routes),
                               skip_high_memory=skip_high_memory)[0]
                route_status = route_sweeps.run({name: binaries[name] for name in routes},
                                               os.environ.get("CHOPINFER_ROUTE_SWEEP", "quick"))
                for name in one_process:
                    binary, cwd, env = binaries[name]
                    log = open(os.path.join(logs, name + ".log"), "w+b")
                    when = "before" if serial_libraries else "beside"
                    print(f"rust tests: {name} library tests in one process {when} nextest", file=sys.stderr, flush=True)
                    # --skip is a substring filter unless --exact is set:
                    # tp4_mtp alone would also remove tp4_mtp_k5 and DCP tests.
                    skipped = (route_test_names(binary, cwd, env) if routes and skip_high_memory
                               else ["route_sweep::"] if routes else []) + (
                        [test for owner, test, _ in high_memory(selection) if owner == name]
                        if skip_high_memory else [])
                    children.append((name, subprocess.Popen([binary, *(["--exact"] if skip_high_memory else []),
                                                            *[arg for test in skipped for arg in ("--skip", test)]], cwd=cwd, env=env, stdin=subprocess.DEVNULL,
                                                            stdout=log, stderr=subprocess.STDOUT), log))
                    if serial_libraries:
                        children[-1][1].wait()
            began = time.time()
            status = subprocess.run(run).returncode
            save_junit("portable-test", began)
            failed = [" ".join(run)] if status else []
            if route_status:
                failed.append("route envelopes")
            for name, child, log in children:
                code = child.wait()
                log.seek(0)
                sys.stdout.write(f"==> {name} library tests (one process)\n")
                sys.stdout.flush()
                sys.stdout.buffer.write(log.read())
                sys.stdout.flush()
                log.close()
                if code:
                    failed.append(f"{name} library tests")
        finally:
            for _, child, log in children:
                if child.poll() is None:
                    child.send_signal(signal.SIGTERM)
                    child.wait()
                log.close()
    for command in doc:
        if subprocess.run(command).returncode:
            failed.append(" ".join(command))
    for item in failed:
        print(f"rust tests: {item} failed", file=sys.stderr)
    return 1 if failed else 0


def cargo_route_binaries(names, packages):
    if not names:
        return {}
    selection = [item for name in names for item in ("-p", name)]
    output = subprocess.check_output(["cargo", "test", *selection, "--lib", "--no-run",
                                      "--message-format=json", *LOCKED], text=True)
    directories = {p["name"]: os.path.dirname(p["manifest_path"]) for p in packages}
    binaries = {}
    for line in output.splitlines():
        message = json.loads(line)
        if message.get("reason") != "compiler-artifact" or not message.get("executable"):
            continue
        target = message.get("target", {})
        for name in names:
            if target.get("name") == name.replace("-", "_") and message.get("profile", {}).get("test"):
                binary = message["executable"]
                env = dict(os.environ, CARGO_MANIFEST_DIR=directories[name], CARGO_PKG_NAME=name)
                env["LD_LIBRARY_PATH"] = os.pathsep.join([os.path.dirname(binary),
                    *([os.environ["LD_LIBRARY_PATH"]] if os.environ.get("LD_LIBRARY_PATH") else [])])
                binaries[name] = (binary, directories[name], env)
    if set(binaries) != set(names):
        raise ValueError("missing model library binaries for route sweeps")
    return binaries


def route_test_names(binary, cwd, env):
    output = subprocess.check_output([binary, "route_sweep::", "--list"], cwd=cwd, env=env, text=True)
    return [line.removesuffix(": test") for line in output.splitlines() if line.endswith(": test")]


def main(arguments=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", action="store_true")
    parser.add_argument("-p", "--package", action="append", default=[])
    parser.add_argument("--memory-pass", choices=("all", "ordinary", "high"),
                        default=os.environ.get("CHOPINFER_CI_MEMORY_PASS", "all"))
    parser.add_argument("--runner", choices=("auto", "nextest", "cargo"),
                        default=os.environ.get("CHOPINFER_RUST_TEST_RUNNER", "auto"))
    parser.add_argument("--test-threads", type=int, default=int(os.environ.get("CHOPINFER_CI_TEST_THREADS", "16")),
                        help="Concurrent tests under nextest (default 16, or CHOPINFER_CI_TEST_THREADS)")
    args = parser.parse_args(arguments)
    if args.workspace == bool(args.package):
        parser.error("select --workspace or packages with -p")
    selection = ["--workspace"] if args.workspace else [item for name in args.package for item in ("-p", name)]
    return run(selection, args.runner, args.test_threads, args.memory_pass)


def run_high_memory(selection, runner):
    entries = high_memory(selection)
    if not entries:
        return 0
    results = []
    for index, (binary, name, reason) in enumerate(entries):
        print(f"high-memory test: {binary} {name}: {reason}", flush=True)
        if runner == "nextest":
            metadata = os.environ.get("CHOPINFER_CI_BINARIES_METADATA")
            command = ["cargo", "nextest", "run", "--profile", "ci", "--test-threads", "1",
                       "--no-tests", "fail", "-E", memory_filter([(binary, name, reason)])]
            command += (["--binaries-metadata", metadata] if metadata and Path(metadata).is_file()
                        else [*selection, *LOCKED])
        else:
            package, _, target = binary.partition("::")
            command = ["cargo", "test", "-p", package, *(["--test", target] if target else ["--lib"]),
                       *LOCKED, name, "--", "--exact", "--test-threads=1"]
        began = time.time()
        started = time.monotonic()
        with tempfile.TemporaryFile(mode="w+") as log:
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            _, status, usage = os.wait4(child.pid, 0)
            child.returncode = os.waitstatus_to_exitcode(status)
            log.seek(0)
            output = log.read()
        print(output, end="", flush=True)
        code = child.returncode
        if runner == "nextest":
            save_junit(f"high-memory-{index}", began, expected=[(binary, name, reason)])
        elif "test result: ok. 1 passed; 0 failed; 0 ignored;" not in output:
            code = code or 1
        result = {"binary": binary, "name": name, "reason": reason, "exit": code,
                  "seconds": time.monotonic() - started, "max_rss_kib": usage.ru_maxrss,
                  "cpu_seconds": usage.ru_utime + usage.ru_stime}
        results.append(result)
        print("high-memory result: " + json.dumps(result), flush=True)
    if directory := os.environ.get("CHOPINFER_CI_JUNIT_DIR"):
        Path(directory).parent.joinpath("high-memory-results.json").write_text(json.dumps(results, indent=2) + "\n")
    return int(any(result["exit"] for result in results))


def run(selection, requested, threads, memory_pass="all"):
    runner, note = choose(requested)
    print(note, file=sys.stderr, flush=True)
    if memory_pass == "high":
        return run_high_memory(selection, runner)
    if runner == "nextest":
        status = run_nextest(selection, threads, skip_high_memory=True)
    else:
        packages = workspace_packages()
        routes = route_sweeps.libraries(selection, packages)
        binaries = cargo_route_binaries(routes, packages)
        route_status = route_sweeps.run(binaries,
                                       os.environ.get("CHOPINFER_ROUTE_SWEEP", "quick"))
        route_tests = [name for binary, cwd, env in binaries.values() for name in route_test_names(binary, cwd, env)]
        status = subprocess.run(commands(selection, runner, threads, skip_routes=bool(routes),
                                         skip_high_memory=True, route_tests=route_tests)[0]).returncode or route_status
        if status:
            print(f"rust tests: cargo test failed with exit {status}", file=sys.stderr)
    if memory_pass == "all":
        status = run_high_memory(selection, runner) or status
    return status


if __name__ == "__main__":
    sys.exit(main())
