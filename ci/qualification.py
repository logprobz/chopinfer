"""Run the qualification CPU tests, one process per module and gate config.

Every module in tools/qual runs under every gate config, except modules that
declare QUALIFICATION_RUNS_ONCE: such a module pins its own config in
setUpModule through qual_test_config, an audit found that nothing in it reads
the outer CHOPINFER_QUAL_CONFIG, and it runs once (see docs/testing.md). The
runner refuses a declaration in a module that no longer pins its config. A
cell passes only if its module ran at least one test and unittest reported OK.
"""
import argparse
import ast
import collections
import os
import pathlib
import re
import shutil
import sys
import tempfile
import json

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import runner

ROOT = pathlib.Path(__file__).resolve().parents[1]
MARKER = "QUALIFICATION_RUNS_ONCE"
TESTS = "tools/qual"


def configs(root):
    found = sorted((root / "models").glob("*/qual.json")) + [root / "tools/qual/gates/mechanics.json"]
    for path in found:
        if not path.is_file():
            raise SystemExit(f"Missing qualification configuration: {path.relative_to(root)}")
    return [(config_label(path, root), str(path.relative_to(root))) for path in found]


def config_label(path, root):
    relative = pathlib.Path(path).relative_to(root)
    return relative.parts[1] if relative.parts[0] == "models" else relative.stem


def modules(root):
    return sorted(path.stem for path in (root / TESTS).glob("test_*.py"))


def default_pin(root):
    """The model qual_test_config.configure selects without an argument."""
    tree = ast.parse((root / TESTS / "qual_test_config.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "configure":
            names = [argument.arg for argument in node.args.args]
            defaults = dict(zip(names[len(names) - len(node.args.defaults):], node.args.defaults))
            if isinstance(defaults.get("name"), ast.Constant):
                return defaults["name"].value
    raise ValueError("tools/qual/qual_test_config.py: configure(name=...) has no literal default")


def pin(source, default):
    """The gate config a module selects for all its tests through
    qual_test_config's setUpModule or configure(), else None."""
    for node in ast.parse(source).body:
        if (isinstance(node, ast.ImportFrom) and node.module == "qual_test_config"
                and any(alias.name == "setUpModule" and alias.asname is None for alias in node.names)):
            return default
        if isinstance(node, ast.FunctionDef) and node.name == "setUpModule":
            for statement in node.body:
                call = statement.value if isinstance(statement, ast.Expr) else None
                if (isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == "configure"
                        and not call.keywords):
                    if not call.args:
                        return default
                    if len(call.args) == 1 and isinstance(call.args[0], ast.Constant):
                        return call.args[0].value
    return None


def declaration(source):
    """A module's QUALIFICATION_RUNS_ONCE reason, or None."""
    for node in ast.parse(source).body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name) and node.targets[0].id == MARKER):
            value = node.value
            return value.value if isinstance(value, ast.Constant) and isinstance(value.value, str) else ""
    return None


def load_inventory(root):
    """The modules that run once, checked: each declares a reason and still
    pins its gate config at module level."""
    default = default_pin(root)
    run_once, errors = set(), []
    for name in modules(root):
        source = (root / TESTS / f"{name}.py").read_text()
        reason = declaration(source)
        if reason is None:
            continue
        if not reason.strip():
            errors.append(f"{TESTS}/{name}.py: {MARKER} needs the audited reason")
        elif pin(source, default) is None:
            errors.append(f"{TESTS}/{name}.py: declares {MARKER} but no longer pins its gate config in "
                          "setUpModule, so it must run under every config")
        else:
            run_once.add(name)
    if errors:
        raise ValueError("\n".join(errors))
    return run_once


def durations_path():
    cache = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    return pathlib.Path(cache) / "chopinfer/qualification-durations.json"


def load_durations():
    """Seconds per module from earlier runs; they order cells, never select them."""
    try:
        return {name: float(value) for name, value in json.loads(durations_path().read_text()).items()}
    except (OSError, ValueError, AttributeError, TypeError):
        return {}


def save_durations(durations):
    path = durations_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f"{path.name}.{os.getpid()}")
        temporary.write_text(json.dumps(durations, indent=1, sort_keys=True) + "\n")
        os.replace(temporary, path)
    except OSError:
        pass


def cells(matrix, selected, run_once, durations=None):
    """(config label, config path or None, module), the longest modules first."""
    durations = durations or {}
    order = sorted(selected, key=lambda name: (-durations.get(name, 0.0), name))
    result = []
    for module in order:
        for index, (label, path) in enumerate(matrix):
            if index and module in run_once:
                continue
            result.append((label, path, module))
    return result


SUMMARY = re.compile(r"^Ran (\d+) tests? in [0-9.]+s$", re.M)
VERDICT = re.compile(r"^(OK|FAILED)(?: \((.*)\))?$", re.M)
SKIP = re.compile(r" \.\.\. skipped (.*)$", re.M)


def outcome(text):
    """(tests, verdict, skip reasons) from a unittest -v log."""
    ran = SUMMARY.findall(text)
    verdicts = VERDICT.findall(text)
    return (int(ran[-1]) if ran else 0, verdicts[-1][0] if verdicts else None, SKIP.findall(text))


def main(arguments=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=int(os.environ.get("CHOPINFER_CI_JOBS", "8")))
    parser.add_argument("--matrix", choices=("reduced", "full"), default="reduced",
                        help="full also repeats the reviewed run-once modules under every config")
    parser.add_argument("--no-config", action="store_true", help="Run each module once without CHOPINFER_QUAL_CONFIG")
    parser.add_argument("--module", action="append", default=[], help="Run only this module (repeatable)")
    parser.add_argument("--discover", action="store_true",
                        help="Previous behavior for comparison: one unittest discover process per config")
    parser.add_argument("--receipts", choices=("isolate", "inherit"), default="isolate",
                        help="inherit keeps CHOPINFER_QUAL_RECEIPTS as given, so a check that no test writes "
                             "under $CHOPINFER_DATA still sees such writes")
    parser.add_argument("--log", type=pathlib.Path, help="Write every cell's unittest output to this file")
    parser.add_argument("--work", type=pathlib.Path, help="Keep cell logs and temporary files here")
    parser.add_argument("--root", type=pathlib.Path, default=ROOT)
    args = parser.parse_args(arguments)
    root = args.root.resolve()
    present = modules(root)
    selected = args.module or present
    missing = sorted(set(selected) - set(present))
    if missing:
        raise SystemExit("qualification: no module " + ", ".join(missing))
    matrix = [("none", None)] if args.no_config else configs(root)
    try:
        run_once = load_inventory(root)
    except (OSError, ValueError) as error:
        raise SystemExit(f"qualification inventory: {error}")
    if args.matrix == "full" or args.no_config or args.discover:
        run_once = set()
    work = args.work or pathlib.Path(tempfile.mkdtemp(prefix="chopinfer-qualification-",
                                                      dir=os.environ.get("TMPDIR")))
    try:
        if args.discover:
            planned = [(label, path, None) for label, path in matrix]
        else:
            planned = cells(matrix, selected, run_once, load_durations())
        steps = [step(root, work, label, path, module, args.receipts == "isolate")
                 for label, path, module in planned]
        try:
            results = runner.run(steps, max(args.jobs, 1), work / "logs", keep_going=True,
                                 out=sys.stdout, label="qualification")
        except runner.Interrupted as interruption:
            return 128 + interruption.signum
        if not args.discover:
            recorded = load_durations()
            for (_, _, module), result in zip(planned, results):
                if result.status == "ok":
                    recorded[module] = round(result.seconds, 1)
            save_durations(recorded)
        return report(root, planned, results, args.log, run_once, matrix, selected if not args.discover else None)
    finally:
        if not args.work:
            shutil.rmtree(work, ignore_errors=True)


def step(root, work, label, path, module, isolate_receipts=True):
    name = f"{label}/{module or 'discover'}"
    temporary = work / "tmp" / name.replace("/", "_")
    temporary.mkdir(parents=True, exist_ok=True)
    # Concurrent cells must not share temporary files or receipts.
    env = dict(os.environ, TMPDIR=str(temporary))
    if isolate_receipts:
        receipts = work / "receipts" / name.replace("/", "_")
        receipts.mkdir(parents=True, exist_ok=True)
        env["CHOPINFER_QUAL_RECEIPTS"] = str(receipts)
    env.pop("CHOPINFER_QUAL_CONFIG", None)
    if path is not None:
        env["CHOPINFER_QUAL_CONFIG"] = path
    pattern = f"{module}.py" if module else "test_*.py"
    argv = [sys.executable, "-B", "-m", "unittest", "discover", "-s", TESTS, "-p", pattern, "-v"]
    return runner.Step(name, argv, env=env, cwd=str(root), echo=False)


def report(root, planned, results, log, run_once, matrix, selected):
    by_name = {result.name: result for result in results}
    totals = collections.defaultdict(lambda: [0, 0, collections.Counter()])
    errors = []
    combined = []
    for label, path, module in planned:
        name = f"{label}/{module or 'discover'}"
        result = by_name[name]
        text = pathlib.Path(result.log).read_text(errors="replace") if result.log else ""
        combined.append(f"Qualification configuration: {path or 'none'}{' module ' + module if module else ''}\n{text}")
        tests, verdict, skips = outcome(text)
        totals[label][0] += 1
        totals[label][1] += tests
        totals[label][2].update(skips)
        if result.status != "ok" or verdict != "OK":
            errors.append(f"{name}: {result.status}, exit {result.exit}, verdict {verdict}")
        elif tests == 0:
            errors.append(f"{name}: ran no tests")
    if selected is not None:
        covered = {module for _, _, module in planned}
        absent = sorted(set(selected) - covered)
        if absent:
            errors.append("modules without a cell: " + ", ".join(absent))
    if log:
        log.write_text("".join(combined))
    for label, (count, tests, skips) in totals.items():
        print(f"qualification {label}: {count} cells, {tests} tests, {sum(skips.values())} skipped")
        for reason, number in sorted(skips.items()):
            print(f"qualification {label} skips: {number} {reason}")
    if run_once and len(matrix) > 1:
        print(f"qualification: {len(run_once)} modules that declare {MARKER} ran once under {matrix[0][1]}")
    for error in errors:
        print("qualification failed: " + error, file=sys.stderr)
    for label, path, module in planned:
        result = by_name[f"{label}/{module or 'discover'}"]
        if result.status != "ok" and result.log:
            sys.stderr.write(f"==> {result.name}\n{pathlib.Path(result.log).read_text(errors='replace')}\n")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
