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
import re
import signal
import subprocess
import sys
import tempfile

import route_sweeps

PINNED = "0.9.146"
LOCKED = ["--locked", "--offline"]


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


def commands(selection, runner, threads, profile="ci", one_process=(), binaries_metadata=None, skip_routes=False):
    """The commands that run `selection` (["--workspace"] or -p pairs).
    Library tests of the `one_process` packages are left out of nextest."""
    if runner == "cargo":
        return [["cargo", "test", *selection, *LOCKED, *(["--", "--skip", "route_sweep::"] if skip_routes else [])]]
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


def run_nextest(selection, threads):
    """Nextest, one-process libraries beside or before it, then doctests."""
    packages = workspace_packages()
    one_process = one_process_libraries(selection, packages)
    serial_libraries = os.environ.get("CHOPINFER_CI_SERIAL_LIBRARIES") == "1"
    routes = route_sweeps.libraries(selection, packages)
    run, *doc = commands(selection, "nextest", threads, one_process=one_process, skip_routes=bool(routes))
    children = []
    with tempfile.TemporaryDirectory(prefix="chopinfer-rust-tests-") as logs:
        try:
            route_status = 0
            if one_process or routes:
                # Reuse the listed build so Cargo cannot replace a running test's executable.
                metadata = os.path.join(logs, "binaries.json")
                binaries = library_binaries(selection, sorted(set(one_process) | set(routes)), packages, metadata)
                run = commands(selection, "nextest", threads, one_process=one_process,
                               binaries_metadata=metadata, skip_routes=bool(routes))[0]
                route_status = route_sweeps.run({name: binaries[name] for name in routes},
                                               os.environ.get("CHOPINFER_ROUTE_SWEEP", "quick"))
                for name in one_process:
                    binary, cwd, env = binaries[name]
                    log = open(os.path.join(logs, name + ".log"), "w+b")
                    when = "before" if serial_libraries else "beside"
                    print(f"rust tests: {name} library tests in one process {when} nextest", file=sys.stderr, flush=True)
                    children.append((name, subprocess.Popen([binary, *(["--skip", "route_sweep::"] if routes else [])], cwd=cwd, env=env, stdin=subprocess.DEVNULL,
                                                            stdout=log, stderr=subprocess.STDOUT), log))
                    if serial_libraries:
                        children[-1][1].wait()
            status = subprocess.run(run).returncode
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


def main(arguments=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", action="store_true")
    parser.add_argument("-p", "--package", action="append", default=[])
    parser.add_argument("--runner", choices=("auto", "nextest", "cargo"),
                        default=os.environ.get("CHOPINFER_RUST_TEST_RUNNER", "auto"))
    parser.add_argument("--test-threads", type=int, default=int(os.environ.get("CHOPINFER_CI_TEST_THREADS", "16")),
                        help="Concurrent tests under nextest (default 16, or CHOPINFER_CI_TEST_THREADS)")
    args = parser.parse_args(arguments)
    if args.workspace == bool(args.package):
        parser.error("select --workspace or packages with -p")
    selection = ["--workspace"] if args.workspace else [item for name in args.package for item in ("-p", name)]
    return run(selection, args.runner, args.test_threads)


def run(selection, requested, threads):
    runner, note = choose(requested)
    print(note, file=sys.stderr, flush=True)
    if runner == "nextest":
        return run_nextest(selection, threads)
    packages = workspace_packages()
    routes = route_sweeps.libraries(selection, packages)
    route_status = route_sweeps.run(cargo_route_binaries(routes, packages),
                                   os.environ.get("CHOPINFER_ROUTE_SWEEP", "quick"))
    status = subprocess.run(commands(selection, runner, threads, skip_routes=bool(routes))[0]).returncode
    status = status or route_status
    if status:
        print(f"rust tests: cargo test failed with exit {status}", file=sys.stderr)
    return status


if __name__ == "__main__":
    sys.exit(main())
