"""Run Rust tests for the workspace or selected packages.

cargo-nextest runs each test in its own process and tests from different
binaries at once, within the groups in .config/nextest.toml. A package whose
Cargo.toml sets package.metadata.chopinfer.lib-tests-in-one-process keeps its
library tests in one libtest process, as cargo test runs them, beside nextest
and from the same build. Nextest does not run doctests, so cargo test --doc
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


def commands(selection, runner, threads, profile="ci", one_process=()):
    """The commands that run `selection` (["--workspace"] or -p pairs).
    Library tests of the `one_process` packages are left out of nextest."""
    if runner == "cargo":
        return [["cargo", "test", *selection, *LOCKED]]
    run = ["cargo", "nextest", "run", *selection, "--profile", profile, "--no-tests", "warn", *LOCKED]
    if threads:
        run[3:3] = ["--test-threads", str(threads)]
    if one_process:
        run += ["-E", "not (" + " | ".join(f"binary_id(={name})" for name in one_process) + ")"]
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


def library_binaries(selection, names, packages):
    """{package: (binary path, working directory, environment)} for the
    library test binaries nextest builds for `names`, run as nextest would:
    from the package directory, with the build's library paths and build
    script variables."""
    listed = json.loads(subprocess.check_output(
        ["cargo", "nextest", "list", *selection, "--list-type", "binaries-only", "--message-format", "json",
         *LOCKED]))
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
    """Nextest for everything, one-process libraries beside it; then doctests."""
    packages = workspace_packages()
    one_process = one_process_libraries(selection, packages)
    run, *doc = commands(selection, "nextest", threads, one_process=one_process)
    children = []
    with tempfile.TemporaryDirectory(prefix="chopinfer-rust-tests-") as logs:
        try:
            if one_process:
                # Listing builds every selected test binary, so both runs use one build.
                for name, (binary, cwd, env) in library_binaries(selection, one_process, packages).items():
                    log = open(os.path.join(logs, name + ".log"), "w+b")
                    print(f"rust tests: {name} library tests in one process beside nextest", file=sys.stderr, flush=True)
                    children.append((name, subprocess.Popen([binary], cwd=cwd, env=env, stdin=subprocess.DEVNULL,
                                                            stdout=log, stderr=subprocess.STDOUT), log))
            status = subprocess.run(run).returncode
            failed = [" ".join(run)] if status else []
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
    status = subprocess.run(commands(selection, runner, threads)[0]).returncode
    if status:
        print(f"rust tests: cargo test failed with exit {status}", file=sys.stderr)
    return status


if __name__ == "__main__":
    sys.exit(main())
