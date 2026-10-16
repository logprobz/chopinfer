#!/usr/bin/env python3
"""Write the `text_edits` policy that turns one text journal body into another.

Usage: text_edits.py BASE.txt HEAD.txt [--out POLICY.json]

The output is a journal policy, {"text_edits": [...]}, for the spec's
`journals` entry of that journal: every removed base line and every added
head line, one-based, with its exact text. It is a starting point for a
reviewer, who reads each line; the proof never derives a policy itself.
"""
import argparse
import difflib
import json
from pathlib import Path
import sys


def edits(base, head):
    out = []
    matcher = difflib.SequenceMatcher(a=base, b=head, autojunk=False)
    for tag, a0, a1, b0, b1 in matcher.get_opcodes():
        if tag == 'equal':
            continue
        out.extend({'remove': {'line': i + 1, 'text': base[i]}} for i in range(a0, a1))
        out.extend({'add': {'line': i + 1, 'text': head[i]}} for i in range(b0, b1))
    return out


def lines(path):
    data = Path(path).read_bytes()
    return data.decode('utf-8').split('\n') if data else []


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('base')
    parser.add_argument('head')
    parser.add_argument('--out')
    args = parser.parse_args(argv)
    text = json.dumps({'text_edits': edits(lines(args.base), lines(args.head))}, indent=1) + '\n'
    if args.out:
        Path(args.out).write_text(text)
    else:
        sys.stdout.write(text)
    return 0


if __name__ == '__main__':
    sys.exit(main())
