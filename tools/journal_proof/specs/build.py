#!/usr/bin/env python3
"""Write a change spec from two sealed captures and a change declaration.

Usage: build.py --repo REPO --model ID --base REV --head REV
       --base-index INDEX --head-index INDEX --declaration CHANGE.json --out SPEC.json

The declaration (schema chopinfer.journal-proof.change.v1) states a
labelled kernel change in a few fields, and every count, ID and position
in the spec comes from the captures:

    {"schema": "chopinfer.journal-proof.change.v1",
     "description": "...",
     "module": "LOCKED_CUBIN",
     "removed_symbols": ["chop_old_a_v1"],
     "added_module": {"name": "NEW_CUBIN", "symbols": ["chop_shared_a_v1"]},
     "launch_rules": [{"from_symbol": "chop_old_a_v1", "to_symbol": "chop_shared_a_v1",
                       "spans": 3, "append": ["ScalarU32(512)"], "flags": 1,
                       "expect": {"block_x": 256}}]}

- `module` drops exactly `removed_symbols` from its lock entries and
  resources (order kept, every retained record byte-identical, payload
  changed); every other locked module stays byte-identical. A locked module
  replacement with `removed_symbols` and locked resolve order follows, and
  each fixture's `removed_resolves` are the base resolves of those symbols,
  one per loaded `module`.
- `added_module` is the one new lock, holding exactly `symbols`; each rank
  that loads `module` loads it once and resolves each symbol once. Every
  load carries its captured payload, whose SHA-256 is the lock's
  `cubin_sha256`. Its module load and resolves are declared with their head
  IDs and positions, in the create, the pinned journals and the history.
- each launch rule turns every base launch of `from_symbol` (exactly
  `spans` buffer arguments, launch flags `flags`) into `to_symbol` with the
  same spans and `append`ed literals, every config field kept and `expect`
  asserted.

The builder refuses captures where a removed ID appears in a base record
other than its own resolve, where the head still names a removed symbol,
or where an added ID appears other than as its own module load or resolve.
Text fixtures and fixtures of only declared journals keep the empty rule;
nothing else is declared, so every other record stays exact.
"""
import argparse
import collections
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import git, need, unique  # noqa: E402
from engine import MODULE, RAW, RESOLVE, launch  # noqa: E402

SCHEMA = 'chopinfer.journal-proof.change.v1'
FIELDS = {'schema', 'description', 'module', 'removed_symbols', 'added_module', 'launch_rules'}
RULE_FIELDS = {'from_symbol', 'to_symbol', 'spans', 'append', 'flags', 'expect'}
# Record verbs whose following words may be issued IDs.
RECORDS = ('resolve', 'module', 'configure', 'release', 'stream', 'event', 'launch', 'upload', 'copy',
           'download', 'zero', 'synchronize', 'record', 'wait', 'capture', 'captured', 'replay')


def read_declaration(path):
    d = json.loads(Path(path).read_text(), object_pairs_hook=unique)
    need(isinstance(d, dict) and d.get('schema') == SCHEMA and set(d) <= FIELDS, f'not a {SCHEMA} declaration')
    need(isinstance(d.get('description'), str), 'a declaration needs a description')
    removed, added, rules = d.get('removed_symbols', []), d.get('added_module'), d.get('launch_rules', [])
    need(isinstance(removed, list) and len(set(removed)) == len(removed) and all(isinstance(s, str) for s in removed),
         'removed_symbols must be a list of distinct symbols')
    need(not (removed or added) or isinstance(d.get('module'), str), 'removed_symbols and added_module need the changed module')
    if added is not None:
        need(isinstance(added, dict) and set(added) == {'name', 'symbols'} and isinstance(added['name'], str)
             and isinstance(added['symbols'], list) and added['symbols'] and len(set(added['symbols'])) == len(added['symbols']),
             'added_module needs a name and its distinct symbols')
    need(isinstance(rules, list), 'launch_rules must be a list')
    for rule in rules:
        need(isinstance(rule, dict) and {'from_symbol', 'to_symbol', 'spans'} <= set(rule) <= RULE_FIELDS
             and type(rule['spans']) is int and rule['spans'] >= 0
             and all(isinstance(v, str) for v in rule.get('append', [])) and isinstance(rule.get('expect', {}), dict),
             'invalid launch rule declaration')
    need(len({r['from_symbol'] for r in rules}) == len(rules), 'one launch rule per old symbol')
    return d


def locks(repo, revision, root):
    """Every locked module of the bundle at `revision`: name -> (lock path, record)."""
    result = {}
    for path in git(repo, 'ls-tree', '-r', '--name-only', revision, root + 'kernels.lock.d').decode().splitlines():
        if not path.endswith('.lock'):
            continue
        for row in json.loads(git(repo, 'show', revision + ':' + path)):
            need(row['name'] not in result, 'duplicate locked cubin')
            result[row['name']] = (path, row)
    return result


def check_locks(old_locks, new_locks, declaration):
    """The locked modules (name -> (lock path, record)) change exactly as
    declared. Returns the changed module's base and head records, or None
    when the declaration names no module."""
    def same(a, b):
        return json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    module = declaration.get('module')
    removed = declaration.get('removed_symbols', [])
    added = declaration.get('added_module')
    expected_new = {added['name']} if added else set()
    need(set(old_locks) <= set(new_locks) and set(new_locks) - set(old_locks) == expected_new,
         f'locked modules added or removed beyond the declaration: {sorted(set(old_locks) ^ set(new_locks))}')
    for name, (path, row) in old_locks.items():
        if name != module:
            need(new_locks[name][0] == path and same(new_locks[name][1], row), f'{name}: locked record changed')
    if added:
        symbols = [e['symbol'] for e in new_locks[added['name']][1]['entries']]
        need(symbols == added['symbols'], f'{added["name"]} holds {symbols}, not the declared symbols')
    if module is None:
        return None
    need(module in old_locks, f'{module} is not a locked module')
    old_row, new_row = old_locks[module], new_locks[module]
    need(old_row[0] == new_row[0], f'{module}: lock file moved')
    old, new = old_row[1], new_row[1]
    need(old['cubin_sha256'] != new['cubin_sha256'], f'{module}: payload unchanged')
    def rest(r):
        return {k: v for k, v in r.items() if k not in ('entries', 'resources', 'cubin_sha256')}
    need(same(rest(old), rest(new)), f'{module}: record fields beyond entries, resources and payload changed')
    for key in ('entries', 'resources'):
        if key not in old and key not in new:
            continue
        symbols = [e['symbol'] for e in old[key]]
        need(all(s in symbols for s in removed), f'{module}: the base {key} lack a removed symbol')
        kept = [e for e in old[key] if e['symbol'] not in removed]
        need(len(kept) == len(new[key]) and all(same(a, b) for a, b in zip(kept, new[key])),
             f'{module}: the {key} change beyond the removed records')
    return old_row, new_row


def removed_resolves(ai, bi, fixture, declaration):
    """The fixture's `removed_resolves` from the base and head inventories
    (auxproof.inventory): each removed symbol resolved exactly once in each
    base module of the changed cubin and never in the head, the head loading
    as many of those modules."""
    module, removed = declaration.get('module'), declaration.get('removed_symbols', [])
    rows = [i for i, r in ai['modules'].items() if r[2] == module]
    head_rows = [i for i, r in bi['modules'].items() if r[2] == module]
    need(len(rows) == len(head_rows), f'{fixture}: {len(rows)} base {module} modules, {len(head_rows)} head')
    need(not any(r[3] in removed for r in bi['resolves'].values()), f'{fixture}: the head resolves a removed symbol')
    out = []
    for symbol in removed:
        ids = [i for i, r in ai['resolves'].items() if r[3] == symbol]
        need(sorted(ai['resolves'][i][2] for i in ids) == sorted(rows),
             f'{fixture}: {symbol} is not resolved once in each {module} module')
        if ids:
            out.append(dict(symbol=symbol, ids=ids, count=len(ids)))
    return out


def added_records(ai, bi, fixture, declaration, locked_sha256=None):
    """The fixture's added module loads and resolves: (auxiliary rule
    entries, added ID -> (verb, name)). Each rank that loads the changed
    module loads the added one once, which resolves each symbol once. Every
    load must carry its captured payload, whose sealed SHA-256 is the added
    module's `cubin_sha256` in the head lock (`locked_sha256`): the spec then
    declares only the committed bytes."""
    added = declaration.get('added_module')
    if not added:
        return dict(added_modules=[], added_resolves=[]), {}
    module = declaration.get('module')
    rows = [i for i, r in ai['modules'].items() if r[2] == module]
    modules = [i for i, r in bi['modules'].items() if r[2] == added['name']]
    need(not any(r[2] == added['name'] for r in ai['modules'].values()), f'{fixture}: the base loads {added["name"]}')
    need(not any(r[3] in added['symbols'] for r in ai['resolves'].values()), f'{fixture}: the base resolves an added symbol')
    need(len(rows) == len(modules), f'{fixture}: {len(rows)} {module} modules, {len(modules)} {added["name"]} modules')
    rule = dict(added_modules=[], added_resolves=[])
    ids = {i: ('module', added['name']) for i in modules}
    if modules:
        groups = collections.defaultdict(list)
        for i in modules:
            row = bi['modules'][i]
            need(len(row) == 6, f'{fixture}: added module {added["name"]} (ID {i}) was loaded without a captured payload')
            need(row[5] == locked_sha256,
                 f'{fixture}: added module {added["name"]} (ID {i}) payload {row[5]} is not the head lock\'s {locked_sha256}')
            groups[tuple(row[2:5])].append(i)
        for (name, fnv, size), group in groups.items():
            rule['added_modules'].append(dict(name=name, fnv1a=fnv, bytes=size, ids=group, count=len(group)))
        for symbol in added['symbols']:
            resolved = [i for i, r in bi['resolves'].items() if r[3] == symbol]
            need(len(resolved) == len(modules) and {bi['resolves'][i][2] for i in resolved} == set(modules),
                 f'{fixture}: {symbol} is not resolved once in each new module')
            rule['added_resolves'].append(dict(symbol=symbol, ids=resolved, count=len(resolved)))
            ids.update((i, ('resolve', symbol)) for i in resolved)
    return rule, ids


def additions(lines, added):
    """The added records of `lines` as policy entries with one-based
    positions: `module ID NAME` and `resolve ID SYMBOL` of an added ID under
    its own role (`added` maps ID -> (verb, name)). Any other appearance of
    an added ID is refused."""
    groups = collections.defaultdict(list)
    for position, line in enumerate(lines, 1):
        m = RESOLVE.fullmatch(line) or MODULE.fullmatch(line)
        if m and int(m[1]) in added:
            verb, name = added[int(m[1])]
            need(line.split()[0] == verb and m[2] == name, f'an added ID appears in a record the change does not allow: {line[:200]}')
            groups[('added_resolves' if verb == 'resolve' else 'added_modules', name)].append((position, int(m[1])))
            continue
        refuse_ids(line, added, 'an added ID')
    policy = {}
    for (kind, name), rows in sorted(groups.items()):
        policy.setdefault(kind, []).append(dict(symbol=name, ids=[i for _, i in rows], positions=[p for p, _ in rows], count=len(rows)))
    return policy


def refuse_ids(line, ids, what):
    words = line.split()
    named = [int(w) for w in words[1:3] if w.isdigit()] if words and words[0] in RECORDS else []
    if words[:2] == ['release', 'graph'] and len(words) > 2 and words[2].isdigit():
        named.append(int(words[2]))
    need(not any(i in ids for i in named), f'a record names {what}: {line[:200]}')
    need(not any(int(raw) & 0xffffffff in ids for raw in RAW.findall(line)), f'a buffer handle names {what}: {line[:200]}')


def removals_only(lines, removed):
    """Refuses a base record naming a removed ID other than its own
    `resolve ID SYMBOL` (`removed` maps ID -> symbol)."""
    for line in lines:
        m = RESOLVE.fullmatch(line)
        if m and int(m[1]) in removed:
            need(m[2] == removed[int(m[1])], f'a removed ID resolves another symbol: {line[:200]}')
            continue
        refuse_ids(line, removed, 'a removed ID')


def no_removed_symbols(lines, where, removed):
    for line in lines:
        need(not any(symbol in line for symbol in removed), f'{where}: a head record names a removed symbol: {line[:200]}')


def launch_rules(lines, declaration):
    """One exact rule per declared old symbol launched in `lines`."""
    rules = []
    for rule in declaration.get('launch_rules', []):
        items = [p for p in map(launch, lines) if p is not None and p['symbol'] == rule['from_symbol']]
        if not items:
            continue
        for p in items:
            need(len(p['args']) == rule['spans'] and all(a.startswith('DeviceBuffer(') for a in p['args']),
                 f'{rule["from_symbol"]}: a launch does not take exactly {rule["spans"]} buffer spans')
            need('flags' not in rule or p['fields']['flags'] == rule['flags'], f'{rule["from_symbol"]}: a launch has other flags')
        args = [dict(arg=i) for i in range(rule['spans'])] + [dict(literal=v) for v in rule.get('append', [])]
        rules.append(dict(from_symbol=rule['from_symbol'], to_symbol=rule['to_symbol'], count=len(items),
                          expect=dict(rule.get('expect', {})), args=args))
    return rules


def generate(repo, model, base_rev, head_rev, base_index, head_index, declaration):
    from auxproof import inventory, load
    from history import owners
    from pins import snapshot
    from proof import SCHEMA as SPEC_SCHEMA
    base, b, bo = snapshot(repo, base_rev, model, base_index)
    head, h, ho = snapshot(repo, head_rev, model, head_index)
    ba, ha = load(base, base_index, b, bo), load(head, head_index, h, ho)
    need(set(base['pins']) == set(head['pins']), 'the change keeps the pinned journal membership')
    groups = collections.defaultdict(list)
    for name, pin in base['pins'].items():
        groups[pin['fixture']].append(name)
    for name, pin in head['pins'].items():
        need(sorted(groups[pin['fixture']]) == sorted(n for n, p in head['pins'].items() if p['fixture'] == pin['fixture']),
             f'{name}: fixture membership changed')
    fixtures = [dict(name=name, create=next((n for n in sorted(names) if n.endswith('-create.txt')), sorted(names)[0]),
                     journals=sorted(names)) for name, names in sorted(groups.items())]
    head_locks = locks(repo, head['commit'], head['root'])
    changed = check_locks(locks(repo, base['commit'], base['root']), head_locks, declaration)
    added_module = declaration.get('added_module')
    added_sha256 = head_locks[added_module['name']][1]['cubin_sha256'] if added_module else None
    removed_symbols = declaration.get('removed_symbols', [])
    spec = dict(schema=SPEC_SCHEMA, fixtures=fixtures, journals={}, auxiliary_rules={})
    if changed is not None:
        old_row, new_row = changed
        spec['module_replacements'] = [dict(
            module=declaration['module'],
            base=dict(lock=old_row[0], sha256=old_row[1]['cubin_sha256']),
            head=dict(lock=new_row[0], sha256=new_row[1]['cubin_sha256']),
            renames={}, removed_symbols=list(removed_symbols), resolve_order='locked')]
    totals = collections.Counter()
    for fixture in fixtures:
        names = fixture['journals']
        rule = dict(added_resolves=[], added_modules=[], create_policy={})
        spec['auxiliary_rules'][fixture['name']] = rule
        kinds = {base['pins'][n]['kind'] for n in names} | {head['pins'][n]['kind'] for n in names}
        if kinds == {'text'}:
            # No device lifetime: the proof compares the bodies as they are.
            need(not any(launch_rules(b[n], declaration) for n in names), f'{fixture["name"]}: a text journal launches a changed symbol')
            totals['text_fixtures'] += 1
            continue
        need(kinds == {'device'}, f'{fixture["name"]}: journals of binding kinds {sorted(kinds)}')
        ab = [ba['manifest']['bindings'][n] for n in names]
        hb = [ha['manifest']['bindings'][n] for n in names]
        need(len({x['host'] for x in ab}) == len({x['host'] for x in hb}) == 1, f'{fixture["name"]}: not one device lifetime')
        ae = ba['hosts'][ab[0]['host']]['events'][:max(x['records'] for x in ab)]
        be = ha['hosts'][hb[0]['host']]['events'][:max(x['records'] for x in hb)]
        ai, bi = inventory(ae), inventory(be)
        added_rule, added = added_records(ai, bi, fixture['name'], declaration, added_sha256)
        rule.update(added_rule)
        if changed is not None:
            removed = removed_resolves(ai, bi, fixture['name'], declaration)
            if removed:
                rule['removed_resolves'] = removed
        removed_ids = {i: d['symbol'] for d in rule.get('removed_resolves', []) for i in d['ids']}
        notes_a = [e[1] for e in ae if e[0] == 'note']
        notes_b = [e[1] for e in be if e[0] == 'note']
        removals_only(notes_a, removed_ids)
        removals_only(ab[0]['create_lines'], removed_ids)
        no_removed_symbols(notes_b, fixture['name'], removed_symbols)
        no_removed_symbols(hb[0]['create_lines'], fixture['name'], removed_symbols)
        rule['create_policy'] = additions(hb[0]['create_lines'], added)
        for name in names:
            removals_only(b[name], removed_ids)
            no_removed_symbols(h[name], name, removed_symbols)
            policy = additions(h[name], added)
            rules = launch_rules(b[name], declaration)
            if rules:
                policy['launch_rules'] = rules
                totals['pinned_launches'] += sum(r['count'] for r in rules)
            if policy:
                spec['journals'][name] = policy
        owner_a = owners(notes_a, dict(zip(names, ab)), ab[0]['create'])
        owner_b = owners(notes_b, dict(zip(names, hb)), hb[0]['create'])
        old = [notes_a[i] for i, o in enumerate(owner_a) if o == 'history']
        new = [notes_b[i] for i, o in enumerate(owner_b) if o == 'history']
        history = additions(new, added)
        rules = launch_rules(old, declaration)
        if rules:
            history['launch_rules'] = rules
            totals['history_launches'] += sum(r['count'] for r in rules)
        if history:
            rule['history_policy'] = history
        totals['fixtures'] += 1
        totals['added_modules'] += sum(d['count'] for d in rule['added_modules'])
        totals['removed_resolves'] += len(removed_ids)
    spec['description'] = declaration['description']
    return spec, dict(totals)


def main(argv=None):
    os.umask(0o022)
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for key in ('repo', 'model', 'base', 'head', 'base-index', 'head-index', 'declaration', 'out'):
        p.add_argument('--' + key, required=True)
    a = p.parse_args(argv)
    spec, totals = generate(a.repo, a.model, a.base, a.head, a.base_index, a.head_index, read_declaration(a.declaration))
    Path(a.out).write_text(json.dumps(spec, indent=2) + '\n')
    print(json.dumps(totals))
    return 0


if __name__ == '__main__':
    sys.exit(main())
