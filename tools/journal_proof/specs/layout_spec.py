#!/usr/bin/env python3
"""Write the spec of a declared layout change from two sealed captures.

Usage: layout_spec.py --repo REPO --model ID --base REV --head REV
       --base-index INDEX --head-index INDEX --declaration PATH
       [--declaration-revision REV] --out SPEC.json

The declaration (schema chopinfer.journal-proof.layout.v1, see
tools/journal_proof/layout.py) is read with git at the head commit, or at
--declaration-revision for a change declared after it landed; the spec names
its path and SHA-256. For every device fixture the writer finds a witness:
- the base modules the declaration removes and the head modules it adds,
  with the resolves they own;
- the head allocations of declared added resources, found by aligning the
  base and head issue sequences (every other ID pairs in order, keeping its
  role, and an allocation its rank, alignment and size or a declared resize);
- the resized allocations, each with its declared resource.

Text journals the declaration lists get their `text_edits`. Every other
fixture keeps the empty rule. The proof checks each witness entry against
the declaration and the traces; the writer cannot widen what is allowed.
"""
import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import git, need, resolve  # noqa: E402
import layout as declared  # noqa: E402
from text_edits import edits as text_edits  # noqa: E402


def fixtures_of(base, head):
    """The fixtures both captures bind, as build.py groups them."""
    need(set(base['pins']) == set(head['pins']), 'a layout change keeps the pinned journal membership')
    groups = collections.defaultdict(list)
    for name, pin in base['pins'].items():
        groups[pin['fixture']].append(name)
    for name, pin in head['pins'].items():
        need(sorted(groups[pin['fixture']]) == sorted(n for n, p in head['pins'].items() if p['fixture'] == pin['fixture']),
             f'{name}: fixture membership changed')
    return [dict(name=name, create=next((n for n in sorted(names) if n.endswith('-create.txt')), sorted(names)[0]),
                 journals=sorted(names)) for name, names in sorted(groups.items())]


def module_entries(inv, names):
    """Modules of `names` in an inventory and the resolves they own, as
    auxiliary rule entries."""
    groups = collections.defaultdict(list)
    for i, row in inv['modules'].items():
        if row[2] in names:
            groups[tuple(row[2:5])].append(i)
    modules = [dict(name=n, fnv1a=f, bytes=b, ids=ids, count=len(ids)) for (n, f, b), ids in groups.items()]
    owned = {i for ids in groups.values() for i in ids}
    symbols = collections.defaultdict(list)
    for i, row in inv['resolves'].items():
        if row[2] in owned:
            symbols[row[3]].append(i)
    for symbol, ids in symbols.items():
        every = [i for i, row in inv['resolves'].items() if row[3] == symbol]
        need(every == ids, f'{symbol} is also resolved from a module the declaration keeps')
    resolves = [dict(symbol=s, ids=ids, count=len(ids)) for s, ids in symbols.items()]
    return modules, resolves, owned | {i for ids in symbols.values() for i in ids}


def unique_resource(decl, kind, test, what):
    found = [r for r in decl['resources'].values() if r['kind'] == kind and test(r)]
    need(len(found) <= 1, f'{what} matches several declared {kind} resources: {[r["name"] for r in found]}')
    return found[0] if found else None


def align(decl, a, b, removed, added_records):
    """The added head allocations and resized pairs: source IDs (base minus
    removed) pair in order with head IDs (minus added modules and resolves)
    except for head allocations of declared added resources. Among the
    alignments, pairing is preferred to adding at each step."""
    source = [i for i in a['issued'] if i not in removed]
    head = [i for i in b['issued'] if i not in added_records]
    k = len(head) - len(source)
    need(k >= 0, f'the head issues {-k} fewer IDs than the base keeps')

    def addable(j):
        row = b['allocations'].get(head[j])
        if row is None:
            return None
        return unique_resource(decl, 'added', lambda r: (r['bytes'], r['alignment']) == (row[3], row[4]), f'head allocation {row[1]}')

    def resized(x, y):
        return unique_resource(decl, 'resized', lambda r: (r['from'], r['to'], r['alignment']) == (x[3], y[3], x[4]) and x[4] == y[4],
                               f'base allocation {x[1]}')

    def compatible(i, j):
        old, new = source[i], head[j]
        if a['typed'].get(old) != b['typed'].get(new):
            return False
        if a['typed'].get(old) != 'allocation':
            return True
        x, y = a['allocations'][old], b['allocations'][new]
        return (x[2] == y[2] and x[4] == y[4] and (x[2] is not None or x[5] == y[5])
                and (x[3] == y[3] or resized(x, y) is not None))
    n = len(source)
    # can[i][d]: from source position i with d additions made, the rest aligns.
    can = [[False] * (k + 1) for _ in range(n + 1)]
    can[n][k] = True
    for i in range(n, -1, -1):
        for d in range(k, -1, -1):
            if i == n and d == k:
                continue
            j = i + d
            ok = False
            if i < n and j < len(head) and compatible(i, j) and can[i + 1][d]:
                ok = True
            elif d < k and j < len(head) and addable(j) is not None and can[i][d + 1]:
                ok = True
            can[i][d] = ok
    need(can[0][0], 'no alignment pairs every kept ID: an issued resource changed beyond the declared additions and resizes')
    i = d = 0
    added, pairs = [], []
    while i < n or d < k:
        j = i + d
        if i < n and compatible(i, j) and can[i + 1][d]:
            pairs.append((source[i], head[j]))
            i += 1
        else:
            r = addable(j)
            row = b['allocations'][head[j]]
            added.append(dict(resource=r['name'], head_raw=row[1], rank=row[2]))
            d += 1
    resizes = []
    for old, new in pairs:
        if a['typed'].get(old) == 'allocation':
            x, y = a['allocations'][old], b['allocations'][new]
            if x[3] != y[3]:
                resizes.append(dict(resource=resized(x, y)['name'], base_raw=x[1], head_raw=y[1], rank=x[2]))
    return added, resizes


def generate(repo, model, base_rev, head_rev, base_index, head_index, path, revision=None):
    from auxproof import inventory, load
    from pins import snapshot
    from proof import SCHEMA as SPEC_SCHEMA
    base, bodies_a, origins_a = snapshot(repo, base_rev, model, base_index)
    head, bodies_b, origins_b = snapshot(repo, head_rev, model, head_index)
    at = resolve(repo, revision) if revision else head['commit']
    data = git(repo, 'show', f'{at}:{path}')
    sha = hashlib.sha256(data).hexdigest()
    decl = declared.parse(data)
    declared.authenticate(repo, base['commit'], head['commit'], base['root'], decl)
    ba, ha = load(base, base_index, bodies_a, origins_a), load(head, head_index, bodies_b, origins_b)
    fixtures = fixtures_of(base, head)
    spec = dict(schema=SPEC_SCHEMA, description=decl['description'], fixtures=fixtures, journals={}, auxiliary_rules={},
                layout=dict(path=path, sha256=sha))
    totals = collections.Counter()
    for fixture in fixtures:
        names = fixture['journals']
        rule = dict(added_resolves=[], added_modules=[], create_policy={})
        spec['auxiliary_rules'][fixture['name']] = rule
        kinds = {base['pins'][n]['kind'] for n in names} | {head['pins'][n]['kind'] for n in names}
        if kinds == {'text'}:
            for name in names:
                if bodies_a[name] != bodies_b[name]:
                    need(name in decl['text_journals'], f'{name}: a text journal changes and the declaration does not list it')
                    spec['journals'][name] = dict(text_edits=text_edits(bodies_a[name], bodies_b[name]))
                    totals['text_journals'] += 1
            continue
        need(kinds == {'device'}, f'{fixture["name"]}: journals of binding kinds {sorted(kinds)}')
        ab = [ba['manifest']['bindings'][n] for n in names]
        hb = [ha['manifest']['bindings'][n] for n in names]
        need(len({x['host'] for x in ab}) == len({x['host'] for x in hb}) == 1, f'{fixture["name"]}: not one device lifetime')
        ae = ba['hosts'][ab[0]['host']]['events'][:max(x['records'] for x in ab)]
        be = ha['hosts'][hb[0]['host']]['events'][:max(x['records'] for x in hb)]
        ai, bi = inventory(ae), inventory(be)
        removed_modules, removed_resolves, removed = module_entries(ai, set(decl['removed_modules']))
        added_modules, added_resolves, added_records = module_entries(bi, set(decl['added_modules']))
        added, resized = align(decl, ai, bi, removed, added_records)
        for key, value in (('removed_modules', removed_modules), ('removed_resolves', removed_resolves),
                           ('added_allocations', added), ('resized_allocations', resized)):
            if value:
                rule[key] = value
        rule['added_modules'], rule['added_resolves'] = added_modules, added_resolves
        totals['fixtures'] += 1
        totals['added_allocations'] += len(added)
        totals['resized_allocations'] += len(resized)
        totals['removed_modules'] += sum(d['count'] for d in removed_modules)
        totals['added_modules'] += sum(d['count'] for d in added_modules)
    return spec, dict(totals)


def main(argv=None):
    os.umask(0o022)
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for key in ('repo', 'model', 'base', 'head', 'base-index', 'head-index', 'declaration', 'out'):
        p.add_argument('--' + key, required=True)
    p.add_argument('--declaration-revision')
    a = p.parse_args(argv)
    spec, totals = generate(a.repo, a.model, a.base, a.head, a.base_index, a.head_index, a.declaration, a.declaration_revision)
    Path(a.out).write_text(json.dumps(spec, indent=1) + '\n')
    print(json.dumps(totals, sort_keys=True))
    return 0


if __name__ == '__main__':
    sys.exit(main())
