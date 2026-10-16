#!/usr/bin/env python3
"""Preserve the approved prefix of per-model projection schedule overrides."""
import importlib.util
import json
from pathlib import Path
import sys

spec = importlib.util.spec_from_file_location("encoder_adoptions", Path(__file__).with_name("encoder-adoptions.py"))
base_guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base_guard)


def check(text, base):
    try:
        records = json.loads(text)
        approved = json.loads(base) if base is not None else []
    except (ValueError, TypeError):
        return ["invalid JSON schedule ledger"]
    if not isinstance(records, list):
        return ["schedule ledger must be an array"]
    problems = []
    if json.dumps(records, indent=2) + "\n" != text:
        problems.append("schedule ledger is not canonical pretty JSON")
    if records[:len(approved)] != approved:
        problems.append("approved schedule entries were removed or rewritten")
    latest = {}
    for record in records:
        if (not isinstance(record, dict) or set(record) != {"inputs", "schedule"}
                or not isinstance(record["inputs"], dict) or not isinstance(record["schedule"], dict)
                or record["schedule"].get("kind") not in {"Typed", "Row", "Grouped"}):
            problems.append("invalid schedule override record")
            continue
        key = json.dumps(record["inputs"], sort_keys=True)
        if latest.get(key) == record["schedule"]:
            problems.append("schedule override repeats the current record")
        latest[key] = record["schedule"]
    return problems


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]
    revision = base_guard.base_revision(root)
    old = base_guard.git(root, "ls-tree", "-r", "--name-only", revision, "--", "models")
    if old.returncode:
        raise SystemExit("cannot enumerate approved schedule ledgers")
    paths = {p.relative_to(root).as_posix() for p in root.glob("models/*/bundle/projection-schedules.json")}
    paths.update(p for p in old.stdout.splitlines() if p.endswith("/bundle/projection-schedules.json"))
    failed = 0
    for relative in sorted(paths):
        path = root / relative
        baseline = base_guard.git(root, "show", f"{revision}:{relative}")
        problems = (["approved schedule ledger is missing or is a symlink"]
                    if not path.is_file() or path.is_symlink()
                    else check(path.read_text(), baseline.stdout if baseline.returncode == 0 else None))
        for problem in problems:
            print(f"{relative}: {problem}")
        failed += bool(problems)
    print(f"projection schedules: {len(paths)} ledgers, {failed} failures")
    return int(bool(failed))


if __name__ == "__main__":
    sys.exit(main())
