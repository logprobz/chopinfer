"""Fixture-scoped load changes of modules whose lock entries stay exact."""
from common import need
from engine import require


def parse(value):
    need(isinstance(value, dict), 'module_activations must map fixtures to added/removed module names')
    for fixture, groups in value.items():
        need(isinstance(fixture, str) and fixture and isinstance(groups, dict) and groups
             and set(groups) <= {'added', 'removed'}, 'invalid module activation fixture')
        seen = set()
        for names in groups.values():
            need(isinstance(names, list) and names and all(isinstance(n, str) and n for n in names),
                 'module activations need nonempty module name lists')
            need(len(names) == len(set(names)) and not seen.intersection(names), 'repeated module activation')
            seen.update(names)


def check(decl, fixture, modules, witness, a, b):
    groups = decl.get('module_activations', {}).get(fixture, {})
    for side, inv, other, ids in (('added', b, a, witness.added), ('removed', a, b, witness.removed)):
        names = groups.get(side, [])
        loads = {i for i, row in inv['modules'].items() if row[2] in names}
        resolves = {i for i, row in inv['resolves'].items() if row[2] in loads}
        for name in names:
            require(any(inv['modules'][i][2] == name for i in loads), f'{name}: declared module activation is absent')
            require(not any(row[2] == name for row in other['modules'].values()),
                    f'{name}: module activation does not change this fixture')
        require(ids & set(inv['modules']) == loads, 'module activations need exactly their positional load witnesses')
        require(ids & set(inv['resolves']) == resolves, 'module activations need exactly their positional resolve witnesses')
        for i in loads:
            row = inv['modules'][i]
            locked = modules.get(row[2])
            need(locked is not None and locked[0] == 'unchanged', f'{row[2]}: module activation is not authenticated')
            require(len(row) == 6 and row[5] == locked[1]['sha256'],
                    f'{row[2]}: activated payload is not the locked SHA-256')
        outside = {r[3] for j, r in inv['resolves'].items() if j not in resolves}
        outside.update(r[3] for r in other['resolves'].values())
        for i in resolves:
            row = inv['resolves'][i]
            name = inv['modules'][row[2]][2]
            require(row[3] in modules[name][1]['symbols'], f'{row[3]} is not an entry of its locked module')
            require(row[3] not in outside,
                    f'{row[3]} is also resolved from a module the activation does not change')
    witness.removed_symbols = {a['resolves'][i][3] for i in witness.removed if i in a['resolves']}
    witness.removed_symbols.difference_update(r[3] for r in b['resolves'].values())
