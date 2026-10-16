#!/usr/bin/env python3
"""Refuse removal or rewriting of a bundle's projection symbol adoptions.

A record with `replaced_by` retires its symbol: it has no placement, and
following `replaced_by` through the latest records must end at an active
record without a cycle."""
import importlib.util
import json
import re
import sys
from pathlib import Path

spec = importlib.util.spec_from_file_location("encoder_adoptions", Path(__file__).with_name("encoder-adoptions.py"))
base_guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base_guard)
SYMBOL = re.compile(r"[A-Za-z_][A-Za-z_0-9]*_v[1-9][0-9]*\Z")
DIGEST = re.compile(r"[0-9a-f]{64}\Z")
IDENTITY = {"symbol", "generated", "entry_sha256"}
FIELDS = (IDENTITY, IDENTITY | {"placement"}, IDENTITY | {"replaced_by"})


def check(text, base):
    problems = []
    try:
        ledger = json.loads(text)
        approved = json.loads(base) if base is not None else []
    except (ValueError, TypeError):
        return ["invalid JSON ledger"]
    if not isinstance(ledger, list):
        return ["ledger must be an array"]
    if json.dumps(ledger, indent=2) + "\n" != text:
        problems.append("ledger is not canonical pretty JSON")
    latest = {}
    for index, entry in enumerate(ledger):
        if not isinstance(entry, dict) or set(entry) not in FIELDS:
            problems.append(f"adoption {index} has invalid fields")
            continue
        if not all(isinstance(entry[key], str) for key in IDENTITY | ({"replaced_by"} & set(entry))):
            problems.append(f"adoption {index} has non-string fields")
            continue
        if (not SYMBOL.fullmatch(entry["symbol"])
                or not SYMBOL.fullmatch(entry["generated"])
                or not entry["generated"].startswith("chop_proj_")
                or not DIGEST.fullmatch(entry["entry_sha256"])
                or "replaced_by" in entry and not SYMBOL.fullmatch(entry["replaced_by"])):
            problems.append(f"adoption {index} has invalid identity")
        if "placement" in entry:
            place = entry["placement"]
            if (not isinstance(place, dict) or set(place) != {"cubin", "position"}
                    or not isinstance(place["cubin"], str) or not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", place["cubin"])
                    or type(place["position"]) is not int or place["position"] < 0):
                problems.append(f"adoption {index} has invalid cubin placement")
        if latest.get(entry["symbol"]) == entry:
            problems.append(f"adoption {index} repeats the current record without a change")
        latest[entry["symbol"]] = entry
    for symbol, entry in latest.items():
        seen = {symbol}
        while isinstance(entry.get("replaced_by"), str):
            successor = entry["replaced_by"]
            if successor in seen:
                problems.append(f"retired {symbol} is replaced in a cycle")
                break
            seen.add(successor)
            if successor not in latest:
                problems.append(f"retired {symbol} is replaced by missing {successor}")
                break
            entry = latest[successor]
    slots = set()
    for entry in latest.values():
        if "replaced_by" in entry:
            continue
        place = entry.get("placement")
        if isinstance(place, dict) and isinstance(place.get("cubin"), str) and type(place.get("position")) is int:
            slot = (place["cubin"], place["position"])
            if slot in slots:
                problems.append("active adoptions share a cubin slot")
            slots.add(slot)
    if ledger[:len(approved)] != approved:
        problems.append("approved entries were removed or rewritten")
    return problems


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]
    revision = base_guard.base_revision(root)
    old_paths = base_guard.git(root, "ls-tree", "-r", "--name-only", revision, "--", "models", "examples")
    if old_paths.returncode:
        raise SystemExit("projection adoptions: cannot enumerate approved ledgers")
    paths = {p.relative_to(root).as_posix() for owner in ("models", "examples")
             for p in root.glob(f"{owner}/*/bundle/projection-adoptions.json")}
    paths.update(p for p in old_paths.stdout.splitlines() if p.endswith("/bundle/projection-adoptions.json"))
    failures = 0
    for relative in sorted(paths):
        path = root / relative
        shown = base_guard.git(root, "show", f"{revision}:{relative}")
        if not path.is_file() or path.is_symlink():
            problems = ["approved ledger is missing or is a symlink"]
        else:
            problems = check(path.read_text(), shown.stdout if shown.returncode == 0 else None)
        for problem in problems:
            print(f"{relative}: {problem}; adoptions are append only")
        failures += bool(problems)
    print(f"projection adoptions: {len(paths)} ledgers, {failures} failures")
    return int(bool(failures))


if __name__ == "__main__":
    sys.exit(main())
