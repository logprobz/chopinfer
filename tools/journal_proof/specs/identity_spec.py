#!/usr/bin/env python3
"""Write the identity spec of a sealed capture: every fixture the capture
binds, with the empty auxiliary rule and no policies.

Usage: identity_spec.py --repo REPO --model ID --revision REV --index INDEX [--out SPEC.json]

Proving a revision against another with this spec claims that no journal,
create or device history changed. A fixture's `create` is its journal named
*-create.txt when it has one, else its first journal by name.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pins import snapshot

EMPTY = {'added_resolves': [], 'added_modules': [], 'create_policy': {}}


def identity_spec(repo, model, revision, index):
    snap, _, _ = snapshot(repo, revision, model, index)
    groups = {}
    for name, pin in snap['pins'].items():
        groups.setdefault(pin['fixture'], []).append(name)
    fixtures = []
    for fixture, names in sorted(groups.items()):
        names = sorted(names)
        creates = [n for n in names if n.endswith('-create.txt')]
        fixtures.append(dict(name=fixture, create=creates[0] if len(creates) == 1 else names[0], journals=names))
    return dict(schema='chopinfer.journal-proof.spec.v1',
                description=f'Identity spec of {len(snap["pins"])} journals in {len(fixtures)} fixtures at {snap["commit"]}.',
                fixtures=fixtures, journals={}, auxiliary_rules={f['name']: dict(EMPTY) for f in fixtures})


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for option in ('repo', 'model', 'revision', 'index'):
        p.add_argument('--' + option, required=True)
    p.add_argument('--out')
    args = p.parse_args(argv)
    text = json.dumps(identity_spec(args.repo, args.model, args.revision, args.index), indent=1) + '\n'
    if args.out:
        Path(args.out).write_text(text)
    else:
        sys.stdout.write(text)
    return 0


if __name__ == '__main__':
    sys.exit(main())
