"""Exact, position-bound resource witnesses in authenticated device traces.

Opaque IDs have no recorded resource role. They prove issuance only, never
an inferred host-service type. Positional modules and resolves require
fixture-scoped activations of unchanged locked modules.
"""
import re

from common import need
from engine import require
import history
from text import validate as validate_edits

KINDS = {'allocation', 'stream', 'event', 'graph', 'mapped-gather', 'opaque', 'module', 'resolve'}


def parse(handles, operations):
    need(isinstance(handles, dict), 'handles must map fixtures to added/removed lists')
    for fixture, groups in handles.items():
        need(isinstance(fixture, str) and fixture and isinstance(groups, dict)
             and groups and set(groups) <= {'added', 'removed'}, 'invalid handle fixture')
        names = set()
        for side, entries in groups.items():
            need(isinstance(entries, list), f'handles.{fixture}.{side} must be a list')
            for entry in entries:
                need(isinstance(entry, dict) and set(entry) == {'name', 'kind', 'at', 'records'}, 'invalid handle declaration')
                name = entry['name']
                need(isinstance(name, str) and name and name not in names, 'handle names must be distinct within a fixture')
                names.add(name)
                need(isinstance(entry['kind'], str) and entry['kind'] in KINDS, 'unknown declared handle kind')
                need(type(entry['at']) is int and entry['at'] > 0, 'handle at must be a one-based trace record ordinal')
                rows = entry['records']
                need(isinstance(rows, list) and rows and rows[0] == ['id', '@id'], 'handle records must start with [id, @id]')
                need(all(isinstance(row, list) and row and isinstance(row[0], str)
                         and all(v is None or type(v) in (str, int) for v in row) for row in rows), 'invalid handle records')
    need(isinstance(operations, dict), 'operations must map fixtures to exact line edits')
    for fixture, edits in operations.items():
        need(isinstance(fixture, str) and fixture, 'operations needs a fixture name')
        validate_edits(edits)
        for entry in edits:
            edit = next(iter(entry.values()))
            need(history.form(edit['text']) is not None, 'an operation edit must have a known operation form')
            need(not re.match(r'(?:allocate|stream|event|captured|module|resolve|configure|async mapped create) ', edit['text']),
                 'creator records use resource declarations, not operation edits')


def instantiate(value, ident):
    if isinstance(value, list):
        return [instantiate(v, ident) for v in value]
    if value == '@id':
        return ident
    if value == '@raw':
        return (1 << 32) | ident
    if isinstance(value, str):
        return value.replace('@raw', str((1 << 32) | ident)).replace('@id', str(ident))
    return value


def creators(events, inv):
    """Issue ordinal and creator-note positions, including event proxies.

    This supplements the inventory without changing legacy typed maps or
    reports. An opaque issue must have neither a typed row nor any creator
    note anywhere in the authenticated prefix.
    """
    found = {i: dict(at=k, kind=inv['typed'].get(i, 'opaque'), notes=[])
             for k, row in enumerate(events) if row[0] == 'id' for i in [row[1]]}
    note = 0
    for k, row in enumerate(events):
        if row[0] != 'note':
            continue
        line = row[1]
        module = re.match(r'(module|resolve|configure) (\d+) ', line)
        if module and int(module[2]) in found:
            ident = int(module[2])
            entry = found[ident]
            kind = 'resolve' if module[1] == 'configure' else module[1]
            require(entry['kind'] == kind and entry['at'] < k, 'module creator disagrees with its issuance')
            require(history.form(line) is not None, 'module creator has an unknown operation form')
            if module[1] == 'resolve':
                require(line == f'resolve {ident} {inv["resolves"][ident][3]}', 'resolve note differs from its metadata')
            if module[1] == 'module':
                require(line == f'module {ident} {inv["modules"][ident][2]}', 'module note differs from its metadata')
            entry['notes'].append((note, k))
            note += 1
            continue
        match = re.fullmatch(r'(event|stream) (\d+)', line)
        gather = re.fullmatch(r'async mapped create (\d+)', line)
        graph = re.fullmatch(r'captured \d+ (\d+) \d+', line)
        ident = int(match[2]) if match else int(gather[1]) if gather else int(graph[1]) if graph else None
        if ident is not None and ident in found:
            kind = match[1] if match else 'mapped-gather' if gather else 'graph'
            entry = found[ident]
            require(entry['at'] < k, 'creator note precedes its issued ID')
            require((not entry['notes'] and entry['kind'] in ('opaque', kind)) or
                    (kind == 'mapped-gather' and entry['kind'] == 'event' and len(entry['notes']) == 1),
                    'issued handle has duplicate or conflicting creators')
            entry['kind'] = kind
            entry['notes'].append((note, k))
        elif line.startswith('allocate ') and k + 1 < len(events) and events[k + 1][0] == 'allocation':
            ident = events[k + 1][1] & 0xffffffff
            if ident in found:
                found[ident]['notes'].append((note, k))
        note += 1
    return found


def witness_entries(decl, fixture, side, events, inv):
    """Generate IDs from declared positions only; never infer additions."""
    out = []
    for entry in decl.get('handles', {}).get(fixture, {}).get(side, []):
        at = entry['at'] - 1
        require(at < len(events) and events[at][0] == 'id', f'{entry["name"]}: no issuance at declared trace position')
        out.append(dict(handle=entry['name'], id=events[at][1]))
    return out


class Witness:
    def __init__(self, decl, fixture, rule, ae, be, a, b):
        self.added, self.removed = set(), set()
        self.added_notes, self.removed_notes = set(), set()
        for side, events, inv, ids, notes in (
                ('added', be, b, self.added, self.added_notes), ('removed', ae, a, self.removed, self.removed_notes)):
            declarations = {d['name']: d for d in decl.get('handles', {}).get(fixture, {}).get(side, [])}
            entries = rule.get(side + '_handles', [])
            seen = set()
            evidence = creators(events, inv) if declarations or entries else {}
            for entry in entries:
                need(isinstance(entry, dict) and set(entry) == {'handle', 'id'} and type(entry['id']) is int
                     and entry['id'] > 0, f'invalid {side} handle witness')
                name, ident = entry['handle'], entry['id']
                need(name in declarations and name not in seen and ident not in ids, 'unknown or repeated handle witness')
                seen.add(name)
                d = declarations[name]
                observed = evidence.get(ident)
                require(observed is not None and observed['kind'] == d['kind'], f'{name}: handle absent or wrong kind')
                at = d['at'] - 1
                require(observed['at'] == at, f'{name}: handle issuance is at the wrong position')
                rows = instantiate(d['records'], ident)
                require(events[at:at + len(rows)] == rows, f'{name}: creator trace records differ')
                require(sum(row[0] == 'id' for row in rows) == 1, f'{name}: records contain another issuance')
                expected_notes = {k for _, k in observed['notes']}
                require({at + k for k, row in enumerate(rows) if row[0] == 'note'} == expected_notes,
                        f'{name}: records must contain exactly this handle\'s creator notes')
                typed = inv['typed'].get(ident)
                expected_rows = {'allocation': inv['allocations'].get(ident), 'stream': inv['streams'].get(ident),
                                 'module': inv['modules'].get(ident), 'resolve': inv['resolves'].get(ident)}
                metadata = [row for row in rows[1:] if row[0] != 'note']
                require(metadata == ([expected_rows[typed]] if typed in expected_rows else []),
                        f'{name}: records must contain exactly this handle\'s resource metadata')
                require(d['kind'] in ('opaque', 'module', 'resolve') or expected_notes, f'{name}: no creating journal line')
                if d['kind'] == 'allocation':
                    require(len(rows) == 3 and rows[1][0] == 'note' and rows[2] == inv['allocations'][ident],
                            f'{name}: allocation creator must follow its allocate note')
                    m = re.fullmatch(r'allocate (\d+) (\d+)(?: -> raw: (\d+))?', rows[1][1])
                    require(m is not None and (int(m[1]), int(m[2])) == tuple(rows[2][3:5])
                            and (m[3] is None or int(m[3]) == rows[2][1]), f'{name}: allocation note differs from metadata')
                ids.add(ident)
                notes.update(i for i, _ in observed['notes'])
            require(seen == set(declarations), f'{side} handles missing declared witnesses: {sorted(set(declarations) - seen)}')


def apply_operations(device, edits, notes_a, notes_b):
    """Reuse text_edits' exact add/remove format, on whole-history ordinals.

    Retained lines still pass through the typed map and owner-order audit.
    No wildcard, creator rewrite, inferred occurrence or implicit use of an
    added handle is admitted.
    """
    for entry in edits:
        side, edit = next(iter(entry.items()))
        i = edit['line'] - 1
        lines = notes_a if side == 'remove' else notes_b
        require(i < len(lines) and lines[i] == edit['text'], f'{side} operation {i + 1} is not the declared line')
        if side == 'remove':
            require(device.expected[i] == lines[i] and not device.changes[i], 'operation overlaps another declaration')
            device.expected[i] = None
            device.changes[i]['removed_operations'] += 1
        else:
            require(i not in device.added, 'operation overlaps another declaration')
            device.added[i] = 'added_operations'
