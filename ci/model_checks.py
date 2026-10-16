"""Plan model-owned CPU checks from validated inventories; ci/cpu.py runs them."""
import argparse
from pathlib import Path
import json
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.model_sdk import inventory


# The reference checks import these in the reference runner's Python.
REFERENCE_MODULES = ("torch", "transformers", "numpy")
REFERENCE_REMEDY = ("set CHOPINFER_REFRUN_PYTHON in local.env to the reference runner's Python "
                    "(docs/reference-runner.md lists its packages); the checks run on CPU")
PROBE = ("import importlib.util, sys\n"
         "print(' '.join(name for name in sys.argv[1:] if importlib.util.find_spec(name) is None))")


class PrerequisiteError(ValueError):
    pass


def reference_python(env=None):
    """The reference runner's Python, once it is known to find the reference modules."""
    env = os.environ if env is None else env
    python = env.get("CHOPINFER_REFRUN_PYTHON")
    if not python:
        raise PrerequisiteError("the reference checks need CHOPINFER_REFRUN_PYTHON: " + REFERENCE_REMEDY)
    try:
        found = subprocess.run([python, "-c", PROBE, *REFERENCE_MODULES], capture_output=True, text=True,
                               stdin=subprocess.DEVNULL, timeout=300)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise PrerequisiteError(f"CHOPINFER_REFRUN_PYTHON={python} does not run ({error}): {REFERENCE_REMEDY}")
    missing = found.stdout.split()
    if found.returncode or missing:
        detail = "lacks " + ", ".join(missing) if missing else "failed: " + found.stderr.strip()[-500:]
        raise PrerequisiteError(f"CHOPINFER_REFRUN_PYTHON={python} {detail}; {REFERENCE_REMEDY}")
    return python


def plan(root=ROOT, env=None, models=None):
    env = os.environ if env is None else env
    models = inventory.load(root) if models is None else models
    inventory.check_files(models, root)
    python = env.get("CHOPINFER_REFRUN_PYTHON") or "python3"
    references = []
    kinds = dict.fromkeys(path for model in models for path in model.checks.reference_tests)
    for path in kinds:
        for model in models:
            if path not in model.checks.reference_tests:
                continue
            kind = Path(path).stem.removeprefix("test_").removesuffix("reference").rstrip("_")
            label = "refrun-" + (kind + "-" if kind else "") + model.name
            references.append((label, [python, "-B", "-m", "unittest", model.directory + "/" + path, "-v"]))
    if not references:
        raise inventory.InventoryError("CPU references plan is empty")
    targets = [target for model in models for target in model.checks.profile_targets]
    if not targets:
        raise inventory.InventoryError("CPU profile plan is empty")
    libraries = list(dict.fromkeys(target.package for target in targets if target.kind == "cdylib"))
    packages = list(dict.fromkeys(target.package for target in targets if target.kind == "bin"))
    bins = [target.name for target in targets if target.kind == "bin"]
    # Keep package entrypoints before profile commands, as in the original gate.
    entrypoints = sorted(name for name in bins if name.endswith("-package"))
    commands = sorted((name for name in bins if not name.endswith("-package")), reverse=True)
    build = ["cargo", "build", *[arg for package in libraries for arg in ("-p", package)]]
    if libraries:
        build.append("--lib")
    build += ["-p", "runtime-host", *[arg for package in packages for arg in ("-p", package)],
              "--bin", "chopinfer", "--bin", "chopinfer-host"]
    build += [arg for name in (*entrypoints, *commands) for arg in ("--bin", name)]
    build += ["--locked", "--offline"]
    files = sorted(path.relative_to(root).as_posix() for path in (root / "tools/refrun").glob("*.py"))
    files += [path for model in models for path in inventory.compile_files(root, model)]
    if not files:
        raise inventory.InventoryError("CPU compile plan is empty")
    return {"references": references, "profile": build, "compile": ["python3", "-m", "py_compile", *files]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("references", "profile", "compile", "prerequisites"))
    args = parser.parse_args()
    if args.kind == "prerequisites":
        print("reference Python: " + reference_python())
        return
    print(json.dumps(plan()[args.kind], indent=1))


if __name__ == "__main__":
    try:
        main()
    except (inventory.InventoryError, PrerequisiteError) as error:
        print(str(error), file=sys.stderr)
        sys.exit(1)
