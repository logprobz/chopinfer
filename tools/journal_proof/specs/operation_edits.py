#!/usr/bin/env python3
"""Write exact operation edits for review from two sealed device captures.

The input is a local layout declaration containing the intended resources,
positional handles and module activations. Existing operations are replaced.
The output is the complete declaration, ready to review and commit before
running layout_spec.py. The prover never derives these edits.
"""
import argparse
import difflib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import auxproof  # noqa: E402
from common import need, unique  # noqa: E402
from engine import Unproven  # noqa: E402
import history  # noqa: E402
import layout  # noqa: E402
from pins import snapshot  # noqa: E402
from layout_spec import fixture_events, fixture_rule, fixtures_of  # noqa: E402


def edits(decl, modules, fixture, ae, be, rule, create, owners_a, owners_b):
    buffers, functions, streams, ids, ai, bi, witness = auxproof.derive(
        ae, be, rule, layout=(decl, modules), fixture=fixture)
    witness.check_stream_and_lifetime(bi, be, create)
    device = layout.Device(decl, witness, ae, be, ai, bi, create)
    mapper = history.Mapper(buffers, functions, streams, ids, closed=False,
                            removed_buffers={r[1] for i, r in ai['allocations'].items() if i not in ids},
                            removed_streams=set(ai['streams']) - set(ids), removed_symbols=witness.handles.removed_symbols)
    base = [row[1] for row in ae if row[0] == 'note']
    head = [row[1] for row in be if row[0] == 'note']
    old, new, positions_a, positions_b = [], [], [], []
    for i, line in enumerate(device.expected):
        if line is None:
            continue
        try:
            mapped = layout.resolve_placeholders(mapper(line))
        except Unproven:
            # A use of a removed handle must be an explicit removal. This
            # sentinel cannot equal a head string or authorize a retained use.
            mapped = ('requires removal', i)
        old.append((owners_a[i], mapped))
        positions_a.append(i)
    for i, line in enumerate(head):
        if i not in device.added:
            new.append((owners_b[i], line))
            positions_b.append(i)
    output = []
    for tag, a0, a1, b0, b1 in difflib.SequenceMatcher(a=old, b=new, autojunk=False).get_opcodes():
        if tag != 'equal':
            output.extend(dict(remove=dict(line=i + 1, text=base[i])) for i in positions_a[a0:a1])
            output.extend(dict(add=dict(line=i + 1, text=head[i])) for i in positions_b[b0:b1])
    return output


def generate(repo, model, base_rev, head_rev, base_index, head_index, data):
    raw = json.loads(data, object_pairs_hook=unique)
    layout.parse(data)
    raw.pop('operations', None)
    decl = layout.parse(json.dumps(raw))
    base, bp, bo = snapshot(repo, base_rev, model, base_index)
    head, hp, ho = snapshot(repo, head_rev, model, head_index)
    need(base['root'] == head['root'], 'the bundle directory moved between the revisions')
    modules = layout.authenticate(repo, base['commit'], head['commit'], base['root'], decl)
    ba, ha = auxproof.load(base, base_index, bp, bo), auxproof.load(head, head_index, hp, ho)
    fixtures = fixtures_of(base, head)
    names = {f['name'] for f in fixtures}
    need(not (set(decl['handles']) | set(decl['module_activations'])) - names, 'layout names unknown fixtures')
    operations = {}
    for fixture in fixtures:
        name, journals = fixture['name'], fixture['journals']
        kinds = {snap['pins'][n]['kind'] for snap in (base, head) for n in journals}
        if kinds == {'text'}:
            need(name not in decl['handles'] and name not in decl['module_activations'],
                 'position-bound declarations need a compared device fixture')
            continue
        need(kinds == {'device'}, f'{name}: not a device fixture')
        ae, be, create = fixture_events(fixture, ba, ha)
        rule = fixture_rule(decl, name, ae, be)
        owners = []
        for aux, events in ((ba, ae), (ha, be)):
            bindings = {n: aux['manifest']['bindings'][n] for n in journals}
            notes = [row[1] for row in events if row[0] == 'note']
            owners.append(history.owners(notes, bindings, next(iter(bindings.values()))['create']))
        proposed = edits(decl, modules, name, ae, be, rule, create, *owners)
        if proposed:
            operations[name] = proposed
        raw['operations'] = operations
        checked = layout.parse(json.dumps(raw))
        results, _, detail = auxproof.prove(bp, hp, fixture, {}, ba, ha, rule, layout=(checked, modules))
        need(all(r['verdict'] != 'UNPROVEN' for r in results.values()) and detail.get('history', {}).get('status') == 'COMPARED',
             f'{name}: generated operations do not prove: {detail.get("error", results)}')
    if operations:
        raw['operations'] = operations
    else:
        raw.pop('operations', None)
    return raw


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('repo', 'model', 'base', 'head', 'base-index', 'head-index', 'declaration', 'out'):
        parser.add_argument('--' + key, required=True)
    args = parser.parse_args(argv)
    result = generate(args.repo, args.model, args.base, args.head, args.base_index, args.head_index,
                      Path(args.declaration).read_bytes())
    Path(args.out).write_text(json.dumps(result, indent=1) + '\n')
    print(json.dumps(dict(fixtures=len(result.get('operations', {})),
                          operations=sum(map(len, result.get('operations', {}).values())))))
    return 0


if __name__ == '__main__':
    sys.exit(main())
