#!/usr/bin/env python3
"""Every bundle's encoder adoption ledger is append only.

Each `models/<m>/bundle/encoder-adoptions.json` (kernels-sm120
`encoder::stability`) must be in its canonical form (pretty JSON, two-space
indent, final newline), each adoption must name the digest of the one before
it (`previous`, the genesis digest for the first), and the ledger must start
with the entries of the approved base: the file on `main`, or at the merge
base with `origin/main` when there is no `main`. A base entry written before
adoptions were chained (no `previous`) is compared without it. A ledger new
on this branch has no base entries.

Usage: tests/encoder-adoptions.py [ROOT]. Exit 0 when every ledger passes.
"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

GENESIS = "0" * 64


def digest(adoption):
    """The digest an adoption's successor names (`stability::digest`)."""
    text = (f"previous {adoption['previous']}\n"
            f"declarations {' '.join(adoption['declarations'])}\n"
            f"nvcc {adoption['nvcc_sha256']}\n")
    for cubin in adoption["cubins"]:
        text += f"cubin {cubin['name']} {cubin['source_closure_sha256']} {cubin['cubin_sha256']}\n"
    return hashlib.sha256(text.encode()).hexdigest()


def git(root, *args):
    return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)


def base_revision(root):
    if git(root, "rev-parse", "--verify", "--quiet", "main^{commit}").returncode == 0:
        return "main"
    merge_base = git(root, "merge-base", "HEAD", "origin/main")
    if merge_base.returncode == 0:
        return merge_base.stdout.strip()
    raise SystemExit("encoder adoptions: no approved base, neither main nor origin/main exists")


def check(text, base):
    """The problems of ledger `text` against approved ledger `base` (None
    when the file is new)."""
    ledger = json.loads(text)
    problems = []
    if json.dumps(ledger, indent=2) + "\n" != text:
        problems.append("not in its canonical form (pretty JSON, two-space indent, final newline)")
    previous = GENESIS
    for index, adoption in enumerate(ledger):
        if adoption.get("previous") != previous:
            problems.append(f"adoption {index} breaks the chain: it names "
                            f"{adoption.get('previous')} but the adoptions before it digest to {previous}")
            break
        previous = digest(adoption)
    for index, approved in enumerate(json.loads(base) if base is not None else []):
        if index >= len(ledger):
            problems.append(f"adoption {index} of the approved base is missing")
            break
        entry = dict(ledger[index])
        if "previous" not in approved:
            entry.pop("previous", None)
        if entry != approved:
            problems.append(f"adoption {index} differs from the approved base")
            break
    return problems


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parents[1])
    revision = base_revision(root)
    ledgers = sorted(root.glob("models/*/bundle/encoder-adoptions.json"))
    failures = 0
    for path in ledgers:
        relative = path.relative_to(root).as_posix()
        shown = git(root, "show", f"{revision}:{relative}")
        base = shown.stdout if shown.returncode == 0 else None
        problems = check(path.read_text(), base)
        for problem in problems:
            print(f"{relative}: {problem}; adoptions are append only")
        failures += bool(problems)
    print(f"encoder adoptions: {len(ledgers)} ledgers against {revision}, "
          f"{'all append only' if not failures else f'{failures} rewritten'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
