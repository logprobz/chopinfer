"""Collect executed tests from a CPU gate's logs and nextest JUnit reports.

Older Python logs contain only unittest counts. Keep those counts explicitly,
never invent names from discovery. Route records identify the model, profile,
seed and executed case count; that seed determines the CPU case plan.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
from xml.etree import ElementTree

import runner

PORTABLE = {"portable", "portable-test", "high-memory-tests"}
RUST_CASE = re.compile(r"^test (.+?) \.\.\. (ok|FAILED|ignored)(?:$|,| )", re.M)
PYTHON_CASE = re.compile(r"^(test\w*|runTest|setUpClass|tearDownClass|setUpModule|tearDownModule) \(([^)]+)\)(.*)$")
PYTHON_RESULT = re.compile(r"(?:^| \.\.\. )(ok|FAIL|ERROR|skipped[^\n]*|expected failure|unexpected success)$")
ROUTE = re.compile(r"([\w.-]+): route sweep (\w+), seed=(\d+), (\d+) cases,")


def scope(step):
    return "portable" if step in PORTABLE else step


def outcome(value):
    if value in ("ok", "passed", "expected failure"):
        return "passed"
    if value == "ignored" or value.startswith("skipped"):
        return "skipped"
    return "failed"


def log_records(step, text):
    """Named libtest/doctest cases, route records, and Python names/counts."""
    tests, routes, named, fixtures = [], [], [], []
    python_name, python_suite = None, ""
    python_fixture = False
    passed_routes = set()
    qualification, cells = {}, []
    binary, kind, model = None, "rust", None
    for line in text.splitlines():
        if match := re.match(r"==> (\S+/test_\S+) \((?:ok|failed)\)", line):
            python_suite = match[1] + "/"
        if match := re.match(r"qualification (\S+/\S+): [\d.]+s \(exit (-?\d+)\)", line):
            cells.append({"name": match[1], "exit": int(match[2])})
        if match := re.match(r"qualification (\S+): (\d+) cells, (\d+) tests, (\d+) skipped", line):
            key, occurrence = match[1], 1
            while key in qualification:
                occurrence += 1
                key = f"{match[1]}#{occurrence}"
            qualification[key] = {"cells": int(match[2]), "ran": int(match[3]), "skipped": int(match[4])}
        if match := PYTHON_CASE.match(line):
            if python_name:
                (fixtures if python_fixture else named).append({"name": python_name, "outcome": "unconfirmed"})
            python_name = f"{python_suite}{match[2]}::{match[1]}"
            python_fixture = match[1] in ("setUpClass", "tearDownClass", "setUpModule", "tearDownModule")
            rest = match[3]
        else:
            rest = line
        if python_name and (match := PYTHON_RESULT.search(rest)):
            (fixtures if python_fixture else named).append({"name": python_name, "outcome": outcome(match[1])})
            python_name = None
        if match := re.match(r"==> (\S+) library tests \(one process\)", line):
            binary, kind = match[1], "rust"
        elif match := re.match(r"==> (\S+) (?:cpu|quick) route envelope", line):
            binary, model, kind = match[1], match[1], "rust"
        elif match := re.match(r"\s*Doc-tests (\S+)", line):
            binary, kind = match[1].replace("_", "-"), "doctest"
        elif match := re.match(r"\s*Running (.+?) \(([^)]+)\)", line):
            # Cargo's hashed executable names are stable identities without
            # the build hash or a machine's target-directory prefix.
            stem = re.sub(r"-[0-9a-f]{16}$", "", Path(match[2]).name)
            binary, kind = match[1] + " [" + stem + "]", "rust"
        if match := ROUTE.search(line):
            routes.append({"step": scope(step), "model": model, "profile": match[1],
                           "tier": match[2].lower(), "seed": int(match[3]), "cases": int(match[4])})
        if match := re.match(r"route budget: model=(\S+) ", line):
            passed_routes.add(match[1])
        if match := RUST_CASE.match(line):
            if binary is None:
                raise ValueError(f"{step}: libtest case has no binary context: {match[1]}")
            tests.append({"step": scope(step), "kind": kind, "binary": binary,
                          "name": match[1], "outcome": outcome(match[2])})
        if kind == "doctest" and (match := re.match(
                r"test result: (ok|FAILED)\. (\d+) passed; (\d+) failed; (\d+) ignored;", line)):
            tests.append({"step": scope(step), "kind": "doctest-suite", "binary": binary,
                          "name": "<suite>", "outcome": outcome(match[1]),
                          "passed": int(match[2]), "failed": int(match[3]), "ignored": int(match[4])})
    ran, verdicts, skips, skipped = runner.unittest_outcome(text)
    if python_name:
        (fixtures if python_fixture else named).append({"name": python_name, "outcome": "unconfirmed"})
    if qualification:
        total = sum(item["ran"] for item in qualification.values())
        if ran and ran != total:
            raise ValueError(f"{step}: qualification totals differ from unittest logs")
        ran = total
        skipped = sum(item["skipped"] for item in qualification.values())
        verdicts = ["OK" if cells and all(cell["exit"] == 0 for cell in cells) else "FAILED"]
    python = None
    if verdicts or ran:
        skipped = max(skipped, len(skips))
        inferred = 0
        # A diagnostic without a final newline can merge with unittest's
        # verdict ("diagnosticok"). Only an OK suite with every identity and
        # every skip accounted for proves those remaining cases passed.
        if (len(named) == ran and verdicts and all(v == "OK" for v in verdicts)
                and sum(case["outcome"] == "skipped" for case in named + fixtures) == skipped):
            for case in named:
                if case["outcome"] == "unconfirmed":
                    case["outcome"] = "passed"
                    inferred += 1
        python = {"step": step, "ran": ran, "skipped": max(skipped, len(skips)),
                  "verdicts": verdicts, "tests": named, "fixtures": fixtures, "passes_from_suite": inferred,
                  "complete_names": len(named) == ran and all(case["outcome"] != "unconfirmed"
                                                              for case in named + fixtures),
                  "configurations": qualification, "cells": sorted(cells, key=lambda c: c["name"])}
    for route in routes:
        route["outcome"] = "passed" if route["model"] in passed_routes else "unconfirmed"
    return tests, routes, python


def junit_records(path, step="portable"):
    tests = []
    for suite in ElementTree.parse(path).getroot().iter("testsuite"):
        for case in suite.findall("testcase"):
            result = ("failed" if case.find("failure") is not None or case.find("error") is not None
                      else "skipped" if case.find("skipped") is not None else "passed")
            tests.append({"step": step, "kind": "rust", "binary": case.get("classname") or suite.get("name"),
                          "name": case.get("name"), "outcome": result})
    return tests


def collect(work, summary, junit=()):
    work = Path(work)
    tests, routes, python, checks = [], [], [], []
    for step in summary["steps"]:
        name = step["name"]
        checks.append({"step": name, "status": step["status"], "exit": step.get("exit")})
        if not step.get("log"):
            continue
        path = work / "logs" / (name.replace("/", "_") + ".log")
        if not path.is_file():
            raise ValueError(f"missing gate log: {path}")
        found, swept, counted = log_records(name, path.read_text(errors="replace"))
        tests.extend(found)
        routes.extend(swept)
        if counted:
            python.append(counted)
    reports = list(junit) or [p for p in sorted((work / "junit").glob("*.xml"))
                             if p.stat().st_mtime >= summary.get("started", 0)]
    for path in reports:
        tests.extend(junit_records(path, "exhaustive-comparisons" if "exhaustive" in str(path) else "portable"))
    # Preserve multiplicity: the same test can run in several feature variants.
    # A duplicated portable case, however, means the pass partition is broken.
    identities = Counter((t["binary"], t["name"]) for t in tests
                         if t["step"] == "portable" and t["kind"] == "rust")
    duplicates = [list(key) for key, count in identities.items() if count > 1]
    return {"schema": "chopinfer-executed-tests-v1", "checks": checks,
            "tests": sorted(tests, key=lambda t: tuple(t.values())),
            "routes": sorted(routes, key=lambda r: (r["model"] or "", r["profile"], r["seed"])),
            "python": sorted(python, key=lambda p: p["step"]), "duplicate_portable_tests": duplicates,
            "junit_reports": len(reports)}


def counted(items):
    return Counter(json.dumps(item, sort_keys=True) for item in items)


def difference(left, right):
    return [{"record": json.loads(key), "count": count} for key, count in sorted((left - right).items())]


def compare(before, after, allow_added=()):
    """Compare every recorded identity/outcome and count without dropping repeats."""
    old, new = counted(before["tests"]), counted(after["tests"])
    problems = {"missing_tests": difference(old, new), "added_tests": difference(new, old),
                "missing_routes": difference(counted(before["routes"]), counted(after["routes"])),
                "added_routes": difference(counted(after["routes"]), counted(before["routes"])),
                "python": [], "checks": [], "duplicates": after["duplicate_portable_tests"]}
    additions = []
    previous = {item["step"]: item for item in before["python"]}
    current = {item["step"]: item for item in after["python"]}
    for step in sorted(previous.keys() | current.keys()):
        left, right = previous.get(step), current.get(step)
        if left is None or right is None:
            problems["python"].append({"step": step, "before": left, "after": right})
            continue
        fields = ("ran", "skipped", "verdicts")
        for field in ("configurations", "cells"):
            if left.get(field) != right.get(field):
                problems["python"].append({"step": step, "field": field,
                                           "before": left.get(field), "after": right.get(field)})
        if any(left[field] != right[field] for field in fields):
            change = {"step": step, "before": {f: left[f] for f in fields},
                      "after": {f: right[f] for f in fields}}
            if (step in allow_added and right["ran"] >= left["ran"] and left["skipped"] == right["skipped"]
                    and left["verdicts"] == right["verdicts"]):
                additions.append(change)
            else:
                problems["python"].append(change)
        if left["complete_names"] and right["complete_names"]:
            if counted(left.get("fixtures", [])) != counted(right.get("fixtures", [])):
                problems["python"].append({"step": step, "before_fixtures": left.get("fixtures", []),
                                           "after_fixtures": right.get("fixtures", [])})
            missing = difference(counted(left["tests"]), counted(right["tests"]))
            added = difference(counted(right["tests"]), counted(left["tests"]))
            if missing or (added and step not in allow_added):
                problems["python"].append({"step": step, "missing": missing, "added": added})
    def checks(inventory):
        portable = [c for c in inventory["checks"]
                    if c["step"] in PORTABLE | {"portable-build", "portable-lint"}]
        result = [c for c in inventory["checks"] if c not in portable]
        if portable:
            result.append({"step": "portable", "status": "ok" if all(c["status"] == "ok" for c in portable)
                           else "failed", "exit": 0 if all(c["exit"] == 0 for c in portable) else 1})
        return counted(result)
    problems["checks"] = difference(checks(before), checks(after)) + difference(checks(after), checks(before))
    return {"equal": not any(problems.values()), "differences": problems, "allowed_python_additions": additions,
            "python_compared_by_count": [s for s in sorted(previous.keys() & current.keys())
                                         if not (previous[s]["complete_names"] and current[s]["complete_names"])]}


def write(path, document):
    Path(path).write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")


def main(arguments=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--junit", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compare", type=Path, nargs=2, metavar=("BEFORE", "AFTER"))
    parser.add_argument("--allow-added-python-step", action="append", default=[])
    args = parser.parse_args(arguments)
    if args.compare:
        report = compare(*(json.loads(path.read_text()) for path in args.compare), args.allow_added_python_step)
        write(args.output, report)
        return 0 if report["equal"] else 1
    if not args.work:
        parser.error("--work is required unless comparing inventories")
    summary = json.loads((args.summary or args.work / "results.json").read_text())
    inventory = collect(args.work, summary, args.junit)
    write(args.output, inventory)
    return 1 if inventory["duplicate_portable_tests"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
