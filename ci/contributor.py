"""Bounded contributor exercises. Validation obligations remain owned by validation.py."""
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import time
import tomllib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.model_sdk import inventory


def models(root=ROOT):
    declared = inventory.load(root)
    inventory.check_files(declared, root)
    return declared


def modes():
    return ("sdk", *(model.name for model in models() if model.contributor))



def recipe(mode):
    return next(model.contributor for model in models() if model.name == mode and model.contributor)


def roots(mode):
    if mode == "sdk":
        return ("model-compiler-sdk", "runtime-bundle", "runtime-kernel-contract")
    return (*recipe(mode).roots, "model-compiler-sdk")


def environment():
    env = dict(os.environ, CARGO_NET_OFFLINE="true", RUSTUP_AUTO_INSTALL="0")
    keys = {key for key in env if key.startswith("CHOPINFER_PACKAGE_")}
    for model in models():
        if model.contributor:
            keys.update(model.contributor.clear_env)
        if model.checkpoint_env:
            keys.update("CHOPINFER_" + model.checkpoint_env + suffix for suffix in
                        ("_CHECKPOINT", "_TOKENIZER_DIR"))
    for index in (ROOT / "models").glob("*/compiler/fixtures/checkpoints/index.json"):
        for checkpoint in json.loads(index.read_text()):
            keys.update((checkpoint["checkpoint_env"], checkpoint["tokenizer_env"]))
    for key in keys:
        env.pop(key, None)
    return env


def external_path(value, label, root=ROOT):
    if not value:
        raise ValueError(f"set {label} to an external directory, for example under CHOPINFER_DATA")
    path = Path(value).expanduser().resolve()
    if path.is_relative_to(root.resolve()) or any((parent / ".git").exists() for parent in (path, *path.parents)):
        raise ValueError(f"{label} must be outside every repository")
    parent = path
    while not parent.exists():
        parent = parent.parent
    if not parent.is_dir() or not os.access(parent, os.W_OK):
        raise ValueError(f"{label} needs a writable directory or parent")
    return path


def probe(command, remedy, *, root=ROOT, env=None):
    try:
        result = subprocess.run(command, cwd=root, env=env, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True)
    except (OSError, subprocess.CalledProcessError) as error:
        detail = getattr(error, "stdout", "") or str(error)
        raise ValueError(f"{remedy}\n{detail.strip()}") from error
    return result.stdout.strip()


def preflight(mode, *, root=ROOT, env=None):
    env = dict(environment() if env is None else env, RUSTUP_AUTO_INSTALL="0")
    # The shared Cargo metadata helpers inherit this process's environment.
    os.environ["RUSTUP_AUTO_INSTALL"] = "0"
    if sys.version_info < (3, 11):
        raise ValueError("Python 3.11 or newer is required; select it in PATH")
    for tool, remedy in (("git", "install Git and select it in PATH"),
                         ("cargo", "install Rust 1.89.0 and select it in PATH"),
                         ("rustc", "install Rust 1.89.0 and select it in PATH"),
                         ("cc", "install a C linker/toolchain exposing cc in PATH")):
        if shutil.which(tool, path=env.get("PATH")) is None:
            raise ValueError(f"missing {tool}; {remedy}")
    rust = probe(["rustc", "--version"], "select Rust 1.89.0", root=root, env=env)
    if not rust.startswith("rustc 1.89.0 "):
        raise ValueError(f"select Rust 1.89.0; observed {rust}")
    for component in ("fmt", "clippy"):
        probe(["cargo", component, "--version"],
              f"install the Rust 1.89.0 {'rustfmt' if component == 'fmt' else 'clippy'} component", root=root, env=env)
    target = external_path(env.get("CARGO_TARGET_DIR"), "CARGO_TARGET_DIR", root)
    scratch = external_path(env.get("CHOPINFER_SCRATCH") or env.get("TMPDIR"), "CHOPINFER_SCRATCH or TMPDIR", root)
    if not scratch.is_dir():
        raise ValueError(f"create the external scratch directory before running fixtures: {scratch}")
    if env.get("TMPDIR"):
        temporary = external_path(env["TMPDIR"], "TMPDIR", root)
        if not temporary.is_dir():
            raise ValueError(f"create the external TMPDIR before running fixtures: {temporary}")
    if env.get("TMPDIR") or not mode:
        # ci/cpu.sh and ci/quick.sh refuse to start without these.
        import preflight
        problems = preflight.temporary_directory(env)
        if problems:
            raise ValueError("; ".join(problems))
    assets = ["Cargo.lock", "crates/model-compiler-sdk/tests/emitter_contract.rs"]
    if mode and mode != "sdk":
        assets += list(recipe(mode).fixtures)
    else:
        assets += ["examples/external-model/fixtures/checkpoint.json"]
    missing = [name for name in assets if not (root / name).is_file()]
    if missing:
        raise ValueError("restore missing fixture inputs from this checkout: " + ", ".join(missing))
    # Keep the source audit's all-feature graph and selected build graph distinct.
    sys.path.insert(0, str(root))
    from tools.model_sdk.graph import metadata, selected_graph
    from tools.model_sdk.workflow import native_toolchain
    try:
        data = metadata(root / "Cargo.toml")
        graph, selected = selected_graph(data, root / "Cargo.toml", packages=roots(mode) if mode else ())
    except (subprocess.CalledProcessError, OSError) as error:
        raise ValueError("locked offline dependency resolution failed; provision the Cargo cache explicitly with "
                         "CARGO_NET_OFFLINE=false cargo fetch --locked, then retry") from error
    if mode:
        graph.require_cuda_free(roots(mode))
    native = native_toolchain(graph, selected, env)
    if not mode:
        # ci/cpu.sh's reference checks; contributor modes never run them.
        import model_checks
        import preflight
        native["reference_python"] = model_checks.reference_python(env)
        problems = preflight.references(env, root) + preflight.workloads(env, root)
        if problems:
            raise ValueError("; ".join(problems))
        native["reference_variables"] = sorted(preflight.reference_variables(root))
    return {"python":sys.version.split()[0], "rust":rust, "target":str(target), "scratch":str(scratch), **native}


def steps(mode):
    selected = [arg for name in roots(mode) for arg in ("-p", name)]
    result = [dict(name="format", phase="format", argv=["cargo", "fmt", "--all", "--check"], tests=False),
              dict(name="clippy", phase="check", argv=["cargo", "clippy", *selected, "--all-targets", "--locked", "--offline", "--", "-D", "warnings"], tests=False)]
    def cargo(name, phase, package, selection, case=None, compile=False):
        argv = ["cargo", "test", "-p", package, *selection, "--locked", "--offline"]
        if compile:
            argv += ["--no-run"]
        elif case:
            argv += [case, "--", "--exact"]
        result.append(dict(name=name, phase=phase, argv=argv, tests=not compile))
    cargo("compile SDK fixtures", "compile", "model-compiler-sdk", ["--lib", "--test", "emitter_contract", "--test", "preparation"], compile=True)
    if mode != "sdk":
        for selection in recipe(mode).tests:
            for row in selection.expanded():
                flags = (["--lib"] if row.lib else [])
                flags += [arg for name in row.bins for arg in ("--bin", name)]
                flags += [arg for name in row.suites for arg in ("--test", name)]
                cargo(row.name, row.phase, row.package, flags, row.case, row.compile)
    else:
        cargo("direct preparation and cache pruning", "preparation", "model-compiler-sdk", ["--test", "preparation"])
    cargo("canonical emission and tamper refusals", "package_and_verify_fixtures", "model-compiler-sdk", ["--test", "emitter_contract"])
    return result


def exhaustive(root=ROOT):
    """(package, test path) of each exhaustive comparison: tests #[ignore]d
    out of the gate on purpose, which the exhaustive profile of
    .config/nextest.toml names and `ci/cpu.sh --exhaustive` runs."""
    document = tomllib.loads((root / ".config/nextest.toml").read_text())
    expression = document["profile"]["exhaustive"]["default-filter"]
    return {(binary.split("::", 1)[0], test) for binary, test in
            re.findall(r"binary_id\(([^)]+)\)\s*&\s*test\(=([^)]+)\)", expression)}


def test_counts(output, package=None, declared=frozenset()):
    """Passed, failed and ignored counts of a fixture run. An ignored case
    passes only when it is a declared exhaustive comparison of `package`;
    any other ignored case, a failure or zero passing tests fails."""
    rows = re.findall(r"test result: ok\. (\d+) passed; (\d+) failed; (\d+) ignored", output)
    counts = {key:sum(int(row[i]) for row in rows) for i, key in enumerate(("passed", "failed", "ignored"))}
    ignored = re.findall(r"^test (\S+) \.\.\. ignored\b", output, re.MULTILINE)
    expected = sorted(name for name in ignored if (package, name) in declared)
    if not counts["passed"] or counts["failed"] or counts["ignored"] != len(expected):
        unexpected = sorted(set(ignored) - set(expected))
        raise ValueError("contributor fixture requires nonzero passing tests, with no failures or ignored cases"
                         + (f" other than declared exhaustive comparisons; unexpected: {', '.join(unexpected)}"
                            if unexpected else ""))
    counts["exhaustive"] = expected
    return counts


def execute(step, *, root=ROOT, env=None, runner=None):
    print(step["name"] + ": " + shlex.join(step["argv"]), flush=True)
    start = time.monotonic()
    if runner:
        stdout, stderr = runner(step["argv"])
        output = (stdout or "") + (stderr or "")
    else:
        result = subprocess.run(step["argv"], cwd=root, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        output = result.stdout
        if result.returncode:
            print(output, end="", file=sys.stderr)
            raise subprocess.CalledProcessError(result.returncode, step["argv"])
    row = dict(step, seconds=time.monotonic() - start)
    if step["tests"]:
        argv = step["argv"]
        package = argv[argv.index("-p") + 1] if "-p" in argv else None
        row.update(test_counts(output, package, exhaustive(root)))
        print(f"  {row['passed']} passed, {row['ignored']} ignored; {row['seconds']:.3f}s", flush=True)
        for name in row["exhaustive"]:
            print(f"  ignored as a declared exhaustive comparison (ci/cpu.sh --exhaustive runs it): {name}", flush=True)
    else:
        print(f"  completed in {row['seconds']:.3f}s", flush=True)
    return row


def outstanding(report):
    return [item for item in report["obligations"] if item["evidence"]["selection"] == "selected"
            and not item["evidence"].get("satisfied", False)]


def show_obligations(report):
    pending = outstanding(report)
    print("Outstanding validation from the current change report: " + (", ".join(item["id"] for item in pending) or "none selected"))
    for item in pending:
        if item["id"] == "shared:cpu":
            for command in item["commands"]:
                print("Full CPU check: " + shlex.join(command))
    print("Use bash ci/quick.sh --report for prerequisites and remaining numerical/serving evidence.")


def run(args, report):
    if args.doctor:
        if args.contributor:
            print("Setup ready. Next fixture check: bash ci/quick.sh --contributor " + args.contributor)
        else:
            print("Native prerequisites are available.")
    elif args.list:
        for step in steps(args.contributor):
            print(step["name"] + ": " + shlex.join(step["argv"]))
    else:
        for step in steps(args.contributor):
            execute(step, env=environment())
        print("Contributor fixtures passed. This bounded exercise is not a full CPU check or model qualification.")
    show_obligations(report)
    return 0
