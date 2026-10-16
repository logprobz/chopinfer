"""What the CPU gate needs from its environment, checked before its first step
so a run that would fail on it stops before it starts (ci/cpu.sh and
ci/quick.sh; ci/cpu.sh --preflight runs only this check)."""
import json
import os
import pathlib
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[1]
# Tests bind Unix sockets under TMPDIR, and a socket path must stay under 108
# bytes: runtime-media's is TMPDIR plus 37 bytes.
TMPDIR_BYTES = 64
# Runs that read the pinned references: the release gate's
# test_reference_binding cells and runtime-eval's shipped-reference test, one
# of the workspace tests.
REFERENCE_STEPS = ("portable", "release-gate")
REFERENCE_PACKAGES = ("runtime-eval",)
# The data root local.env sets. The CPU gate reads the files pinned under it
# only where they are present, except reference workloads (see
# reference_workloads).
DATA_ROOTS = ("CHOPINFER_DATA",)
# Steps that run the workspace tests under cargo-nextest: `portable` unless the
# run opts out with CHOPINFER_RUST_TEST_RUNNER, the exhaustive comparisons
# always.
NEXTEST_STEPS = ("portable", "exhaustive-comparisons")
# The gate's runner when CHOPINFER_RUST_TEST_RUNNER is unset (ci/cpu.py).
GATE_RUNNER = "nextest"
GIT_ROUTING = ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_OBJECT_DIRECTORY",
               "GIT_ALTERNATE_OBJECT_DIRECTORIES", "GIT_COMMON_DIR", "GIT_NAMESPACE", "GIT_CEILING_DIRECTORIES")


def temporary_directory(env):
    """Problems with TMPDIR as the steps see it."""
    value = env.get("TMPDIR")
    if not value:
        return ["TMPDIR is unset"]
    path = pathlib.Path(value)
    if not path.is_absolute():
        return [f"TMPDIR={value} is not an absolute path"]
    if not path.is_dir():
        return [f"TMPDIR={value} is not a directory: create it first"]
    problems = []
    if not os.access(path, os.W_OK | os.X_OK):
        problems.append(f"TMPDIR={value} is not writable")
    if len(os.fsencode(value)) > TMPDIR_BYTES:
        problems.append(f"TMPDIR={value} is {len(os.fsencode(value))} bytes, more than {TMPDIR_BYTES}: tests bind "
                        "Unix sockets under it and a socket path must stay under 108 bytes")
    return problems


def reference_variables(root=ROOT):
    """{variable: [entry, ...]} for the entries of
    models/*/qual-references.lock.json kept under a variable with no default,
    other than a data root."""
    found = {}
    for lock in sorted(pathlib.Path(root).glob("models/*/qual-references.lock.json")):
        for name, entry in json.loads(lock.read_text()).get("entries", {}).items():
            path = entry.get("path") if isinstance(entry, dict) else None
            if (isinstance(path, dict) and path.get("env") and path.get("default") is None
                    and path["env"] not in DATA_ROOTS):
                found.setdefault(path["env"], []).append(f"{lock.parent.name} {name}")
    return found


def references(env, root=ROOT):
    """Problems with the variables the declared references need."""
    problems = []
    for variable, needed in sorted(reference_variables(root).items()):
        value = env.get(variable)
        if not value:
            problems.append(f"{variable} is unset; it names the directory that holds the pinned {', '.join(needed)}")
        elif not pathlib.Path(value).is_dir():
            problems.append(f"{variable}={value} is not a directory; it holds the pinned {', '.join(needed)}")
    return problems


def reference_workloads(root=ROOT):
    """{(variable, path): [entry, ...]} for each reference workload a model
    declaration keeps under a data root (qual.json `references`, its
    `configurations` and any qual.VARIANT.json). runtime-eval's
    shipped-reference tests verify such a workload whenever its variable is
    set (crates/runtime-eval/tests/references.rs), so with the variable set
    it is a gate input."""
    found = {}
    root = pathlib.Path(root)
    for path in [*sorted(root.glob("models/*/qual.json")), *sorted(root.glob("models/*/qual.*.json"))]:
        document = json.loads(path.read_text())
        configurations = document.get("configurations")
        sections = [document.get("references"),
                    *(entry.get("references") for entry in (configurations or {}).values()
                      if isinstance(entry, dict))]
        for section in sections:
            for name, pin in (section or {}).items():
                workload = pin.get("workload") if isinstance(pin, dict) else None
                if isinstance(workload, dict) and workload.get("env") in DATA_ROOTS and workload.get("path"):
                    found.setdefault((workload["env"], workload["path"]), []).append(
                        f"{path.parent.name} {name}")
    return found


def workloads(env, root=ROOT):
    """Problems with the reference workloads under a set data root. With the
    variable unset the tests expect those references to be insufficient, so
    nothing is required."""
    problems = []
    for (variable, relative), needed in sorted(reference_workloads(root).items()):
        if variable not in env:
            continue
        path = pathlib.Path(env[variable]) / relative
        if not path.is_absolute():
            path = pathlib.Path(root) / path
        if not path.is_file():
            problems.append(f"{variable}={env[variable]} holds no {relative}: the workload of the pinned "
                            f"{', '.join(needed)} reference, which runtime-eval's reference tests verify "
                            f"whenever {variable} is set (docs/contributing.md)")
    return problems


def test_runner(steps, env):
    """Problems with the Rust test runner the selected gate steps need. The
    gate runs its workspace tests under cargo-nextest, so a missing nextest
    stops the run at once rather than falling back to cargo test, unless
    CHOPINFER_RUST_TEST_RUNNER asks for cargo or auto."""
    needed = set(steps) & set(NEXTEST_STEPS)
    if not needed:
        return []
    requested = env.get("CHOPINFER_RUST_TEST_RUNNER") or GATE_RUNNER
    if requested not in ("auto", "nextest", "cargo"):
        return [f"CHOPINFER_RUST_TEST_RUNNER={requested} is not auto, nextest or cargo"]
    if requested != "nextest" and needed == {"portable"}:
        return []
    import rust_tests
    if rust_tests.nextest_version() is not None:
        return []
    if "exhaustive-comparisons" in needed:
        return [f"cargo-nextest is not installed; the exhaustive comparisons run under cargo-nextest "
                f"{rust_tests.PINNED} (docs/contributing.md)"]
    return [f"cargo-nextest is not installed; the gate runs the workspace tests under cargo-nextest "
            f"{rust_tests.PINNED} (docs/contributing.md), or set CHOPINFER_RUST_TEST_RUNNER=cargo to run "
            "them with cargo test, one test binary at a time"]


def problems(steps=(), packages=(), env=None, root=ROOT):
    """Every problem a run of `steps` (ci/cpu.py step names) and of the Rust
    tests of `packages` ("--workspace" for all) would meet."""
    env = os.environ if env is None else env
    found = temporary_directory(env)
    if set(steps) & set(REFERENCE_STEPS) or set(packages) & {*REFERENCE_PACKAGES, "--workspace"}:
        found += references(env, root) + workloads(env, root)
    return found + test_runner(steps, env)


def head(root=ROOT):
    """The commit checked out at `root`, or None outside a Git checkout."""
    # The checkout at root, never one inherited through the environment, and
    # a poll that never takes the index lock a step's own Git command needs.
    env = {name: value for name, value in os.environ.items() if name not in GIT_ROUTING}
    env["GIT_OPTIONAL_LOCKS"] = "0"
    found = subprocess.run(["git", "-C", str(root), "rev-parse", "--verify", "-q", "HEAD"], capture_output=True,
                           text=True, env=env, stdin=subprocess.DEVNULL, check=False)
    return (found.stdout.strip() or None) if found.returncode == 0 else None
