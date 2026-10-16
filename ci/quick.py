"""Check changed workspace packages and their transitive workspace dependents,
and run ci/cpu.sh's Python and shell checks the change affects."""
import argparse
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import tomllib

import change_scope
import check_inputs
import preflight
import runner
import rust_tests


def affected(metadata, paths):
    return change_scope.Graph(metadata).select(paths)["packages"]


def tooling_checks(paths, work, jobs, scanner=None, everything=False):
    """(checks to run, gate-only checks, covered paths) for the changed `paths`.

    The checks are ci/cpu.py's independent steps whose inputs
    (ci/check_inputs.py) contain a changed path, or all of them with
    `everything`, and the qualification run limited to the tools/qual modules
    whose inputs do. The gate-only ones build Rust in fresh targets and stay
    with ci/cpu.sh. A selected check brings the setup steps it runs after
    (cpu.SETUP, such as the profile build whose copy qualification scores
    with) and runs after the selected steps it names. A path covered only by
    a compile step is not covered."""
    import cpu
    import qualification
    target = pathlib.Path(os.environ.get("CARGO_TARGET_DIR") or change_scope.ROOT / "target")
    shared, independent = cpu.steps(target, work, jobs, int(os.environ.get("CHOPINFER_CI_TEST_THREADS", "16")))
    scanner = scanner or check_inputs.Scanner(change_scope.ROOT)
    selected, gate_only, covered = [], [], set()
    for step in independent:
        inputs = scanner.inputs(step.argv)
        hits = {path for path in paths if inputs.covers(path)}
        if hits or everything:
            if step.name not in cpu.COMPILE_ONLY:
                covered |= hits
            (gate_only if step.name in cpu.FRESH_BUILDS else selected).append(step)
    modules = []
    for module in qualification.modules(change_scope.ROOT):
        inputs = scanner.inputs(["python3", "-B", f"{qualification.TESTS}/{module}.py"])
        hits = {path for path in paths if inputs.covers(path)}
        if hits or everything:
            covered |= hits
            modules.append(module)
    steps = {step.name: step for step in cpu.order(shared, independent)}
    if modules:
        run = steps[cpu.QUALIFICATION]
        if not everything:
            run.argv = [*run.argv, *(argument for module in modules for argument in ("--module", module))]
        selected.append(run)
    selected = cpu.select(list(steps.values()), [step.name for step in selected])
    for step in selected:
        step.echo = False
    return selected, [step.name for step in gate_only], covered


def run_checks(checks, work, jobs):
    results = runner.run(checks, jobs, work / "logs", keep_going=True, label="quick")
    failed = runner.failed(results)
    if failed:
        raise subprocess.CalledProcessError(1, ["ci/cpu.py", *[arg for name in failed for arg in ("--only", name)]])


def output(*command):
    # New files have no version at the comparison base.
    stderr = subprocess.DEVNULL if command[:2] == ("git", "show") else None
    return subprocess.check_output(command, stderr=stderr)


def main(arguments=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="HEAD", help="Compare working tree with this ref (default HEAD)")
    parser.add_argument("--list", action="store_true", help="Print selection without compiling")
    parser.add_argument("--report", action="store_true", help="Explain validation obligations without executing checks")
    parser.add_argument("--all", action="store_true", help="Conservative whole-workspace override")
    parser.add_argument("--format", choices=("text", "json"), default="text")
    parser.add_argument("--context", type=pathlib.Path, help="Explicit candidate identity expectations; never producer observations")
    parser.add_argument("--receipt", type=pathlib.Path, action="append", default=[])
    parser.add_argument("--kernel-reports", type=pathlib.Path, action="append", default=[],
                        help="Current producer report file or directory (repeatable)")
    parser.add_argument("--previous-kernel-reports", type=pathlib.Path, action="append", default=[],
                        help="Historical producer report file or directory (repeatable)")
    parser.add_argument("--output", type=pathlib.Path, help="New report file outside the repository")
    import contributor
    parser.add_argument("--contributor", choices=contributor.modes(), help="Run bounded fixtures without CUDA, weights or GPUs")
    parser.add_argument("--doctor", action="store_true", help="Check fixture prerequisites, or native prerequisites without --contributor")
    args = parser.parse_args(arguments)
    if not args.report and (args.context or args.receipt or args.output or args.format != "text"
                            or args.kernel_reports or args.previous_kernel_reports):
        parser.error("evidence inputs, --output and --format require --report")
    if args.contributor or args.doctor:
        import contributor
        if not args.list:
            setup = contributor.preflight(args.contributor)
            print("contributor setup: " + json.dumps(setup, sort_keys=True), file=sys.stderr)
    try:
        metadata = json.loads(output("cargo", "metadata", "--format-version", "1", "--all-features", "--locked", "--offline"))
    except (subprocess.CalledProcessError, OSError, ValueError):
        metadata = {"workspace_root": str(change_scope.ROOT)}
    base = output("git", "rev-parse", "--verify", args.base + "^{commit}").decode().strip()
    paths = output("git", "diff", "--name-only", "--no-renames", "-z", base).split(b"\0")
    paths += output("git", "ls-files", "--others", "--exclude-standard", "-z").split(b"\0")
    paths = sorted({p.decode() for p in paths if p})
    edges = change_scope.declarations()
    old_sources = {}
    old_declarations = {}
    for path in paths:
        declaration = path.startswith("models/") and pathlib.Path(path).name in ("model.toml", "validation.toml", "profiles.json")
        if path == change_scope.REGISTRY or path.endswith(".rs") or declaration:
            try:
                old = output("git", "show", base + ":" + path).decode()
            except subprocess.CalledProcessError:
                continue
            if path == change_scope.REGISTRY:
                edges += tomllib.loads(old)["edges"]
            elif declaration:
                old_declarations[path] = old
            else:
                old_sources[path] = old
    graph = change_scope.Graph(metadata, edges=edges, old_sources=old_sources)
    if not (args.report or args.contributor or args.doctor):
        return check(args, graph, paths)
    import validation
    catalog = validation.Catalog(change_scope.ROOT, metadata)
    catalog.retain_previous(old_declarations)
    # Changed ownership/profile declarations require broad obligations. Old
    # source edges above retain removed direct readers, independent of Cargo.
    report = validation.build_report(metadata, paths, graph=graph, catalog=catalog, all_checks=args.all,
              context=validation.read_json(args.context) if args.context else None, receipts=args.receipt,
              kernel_reports=args.kernel_reports, previous_kernel_reports=args.previous_kernel_reports,
              source_identity={"revision": output("git", "rev-parse", "--verify", "HEAD^{commit}").decode().strip(),
                               "base": base})
    report["base"] = base
    if args.report:
        validation.write_report(report, args.format, args.output)
        return 0
    return contributor.run(args, report)


def refused(problems):
    """Prints each problem and returns the exit status: 2 when there are any."""
    for problem in problems:
        print("quick preflight: " + problem, file=sys.stderr)
    if problems:
        print("quick preflight failed: nothing ran (see docs/toolchain.md)", file=sys.stderr)
    return 2 if problems else 0


def check(args, graph, paths):
    jobs = int(os.environ.get("CHOPINFER_CI_JOBS", os.cpu_count() or 8))
    if refused(preflight.temporary_directory(os.environ)):
        return 2
    work = pathlib.Path(tempfile.mkdtemp(prefix="chopinfer-quick.", dir=os.environ.get("TMPDIR")))
    try:
        checks, gate_only, covered = tooling_checks(paths, work, jobs, everything=args.all)
        selection = graph.select(paths, broad=args.all, checked=covered)
        names = selection["packages"]
        print("quick packages: " + (", ".join(names) or "none"), flush=True)
        print("quick checks: " + (", ".join(step.name for step in checks) or "none"), flush=True)
        if gate_only:
            print("left to ci/cpu.sh (fresh Rust builds): " + ", ".join(gate_only), flush=True)
        if args.list:
            return
        packages = (["--workspace"] if selection["workspace"] else
                    [argument for name in names for argument in ("-p", name)])
        if refused(preflight.problems([step.name for step in checks], ["--workspace"] if selection["workspace"]
                                      else names)):
            return 2
        subprocess.run(["cargo", "fmt", "--all", "--check"], check=True)
        if not names and not selection["workspace"] and not checks:
            print("No changed inputs. bash ci/cpu.sh runs every check.")
            return
        if checks:
            run_checks(checks, work, jobs)
        if not names and not selection["workspace"]:
            return
        subprocess.run(["cargo", "clippy", *packages, "--all-targets", "--locked", "--offline",
                        "--", "-D", "warnings"], check=True)
        status = rust_tests.run(packages, os.environ.get("CHOPINFER_RUST_TEST_RUNNER", "auto"),
                                int(os.environ.get("CHOPINFER_CI_TEST_THREADS", "16")))
        if status:
            raise subprocess.CalledProcessError(status, ["ci/rust_tests.py", *packages])
    finally:
        # The external SDK smoke writes read-only copies.
        subprocess.run(["chmod", "-R", "u+w", "--", str(work)], stderr=subprocess.DEVNULL, check=False)
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except subprocess.CalledProcessError as error:
        print(f"quick: check failed with exit {error.returncode}: {' '.join(error.cmd)}", file=sys.stderr)
        sys.exit(error.returncode)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print("quick: " + str(error), file=sys.stderr)
        sys.exit(2)
