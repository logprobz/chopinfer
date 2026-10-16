"""Declared served-layout changes (schema chopinfer.journal-proof.layout.v1).

A layout declaration states, in a few reviewable items, how a served layout
changes between two revisions:

- `resources.added`: new device allocations, each with its bytes,
  alignment, count per rank, lifetime, initialization and the base
  allocation (`alongside`) whose ranks must hold it;
- `resources.resized`: existing allocations whose size changes, optionally
  with a plane layout;
- `modules`: cubins the head no longer loads and cubins it newly loads,
  authenticated by both revisions' kernel locks; an added cubin is loaded
  exactly as often as a named base cubin, and its resolves carry exactly the
  declared configure values;
- `collectives`: a collective launch kind replaced by a new one whose peer
  arrays (or buffers) name added allocations, for a declared rank count;
- `kernels`: a named kernel family whose launches are renamed and regridded
  by exact cases, with argument literal maps, host-buffer span lengths and
  plane strides tied to a resized layout;
- `text_journals`: the text journals that may carry `text_edits`.

The proof applies the declaration to every device fixture, to its create
range, every pinned journal and the history alike. Every selection is
evaluated on the base: the head is never read to choose what a rule applies
to. A spec names the declaration by path and SHA-256; the proof reads it
with git at the head commit, or at an explicitly named revision for a change
declared after it landed. A spec's per-fixture witness (the head IDs of added
allocations, the module and resolve IDs removed or added, the resized
allocations) is checked against the declaration and the authenticated
traces; it chooses a witness, never what is allowed.
"""
import collections
import json
import re
from common import git, need, unique
from engine import ALLOC, Unproven, launch, render, require
import history

SCHEMA = 'chopinfer.journal-proof.layout.v1'
FIELDS = {'schema', 'description', 'modules', 'resources', 'collectives', 'kernels', 'text_journals'}
CONFIG_FIELDS = ('flags', 'grid_x', 'grid_y', 'grid_z', 'block_x', 'block_y', 'block_z', 'shared_memory_bytes')
SYMBOL = re.compile(r'chop_\w+')
LITERAL = re.compile(r'(?:Scalar)?(?:U32|U64|I32|F32)\([-+0-9.e]+\)|NullDevicePointer')
SPAN = re.compile(r'DeviceBufferSpanV1 \{ buffer: BundleDeviceBufferHandleV1 \{ raw: (\d+) \}, offset: (\d+), length: (\d+) \}')
LAYOUT_FIELDS = {'planes', 'from_stride', 'to_stride', 'from_tail', 'to_tail'}
# The witness fields a fixture's auxiliary rule may carry under a layout.
WITNESS_FIELDS = {'added_allocations', 'removed_modules', 'resized_allocations'}
# A placeholder for a head handle that has no base counterpart: the typed
# mapper (history.Mapper) rewrites only `raw: DIGITS`, so the head handle passes through.
PLACEHOLDER = 'raw: @'


def placeholder_span(raw, length):
    return f'DeviceBufferSpanV1 {{ buffer: BundleDeviceBufferHandleV1 {{ {PLACEHOLDER}{raw} }}, offset: 0, length: {length} }}'


def resolve_placeholders(line):
    return line.replace(PLACEHOLDER, 'raw: ')


def _names(value, what):
    need(isinstance(value, list) and all(isinstance(v, str) and v for v in value) and len(set(value)) == len(value),
         f'{what} must be a list of distinct names')
    return list(value)


def _positive(value, what, zero=False):
    need(type(value) is int and (value >= 0 if zero else value > 0), f'{what} must be a {"nonnegative" if zero else "positive"} integer')
    return value


def _index(key):
    need(isinstance(key, str) and key.isdigit() and str(int(key)) == key, f'argument index {key!r} is not a decimal index')
    return int(key)


def _selector(value, what, case=False):
    """A selection evaluated on a base launch: config fields, `rows`,
    `prefix`, argument literals and the previous launch on the same stream."""
    need(isinstance(value, dict), f'{what} must be an object')
    need(set(value) <= set(CONFIG_FIELDS) | {'rows', 'prefix', 'args', 'after'}, f'{what} has an unknown field')
    need(value or not case, f'{what} of a case must select something')
    for key in CONFIG_FIELDS:
        if key in value:
            _positive(value[key], f'{what}.{key}', zero=True)
    if 'rows' in value:
        need(value['rows'] is None or type(value['rows']) is int, f'{what}.rows must be an integer or null')
    if 'prefix' in value:
        need(isinstance(value['prefix'], str) and value['prefix'], f'{what}.prefix must be a string')
    args = value.get('args', {})
    need(isinstance(args, dict) and all(isinstance(v, str) and LITERAL.fullmatch(v) for v in args.values()),
         f'{what}.args must map argument indexes to scalar literals')
    for key in args:
        _index(key)
    if 'after' in value:
        after = value['after']
        need(isinstance(after, dict) and 'symbols' in after and set(after) <= {'symbols', *CONFIG_FIELDS},
             f'{what}.after needs symbols and may name config fields')
        need(_names(after['symbols'], f'{what}.after.symbols') and all(SYMBOL.fullmatch(s) for s in after['symbols']),
             f'{what}.after.symbols must be launch symbols')
        for key in CONFIG_FIELDS:
            if key in after:
                _positive(after[key], f'{what}.after.{key}', zero=True)
    return value


def _change(item, collective, resources):
    kind = 'collective' if collective else 'kernel family'
    allowed = {'name', 'from_symbol', 'to_symbol', 'when', 'args', 'lengths', 'strides', 'cases'}
    if collective:
        allowed |= {'ranks', 'peers', 'buffers'}
    need(isinstance(item, dict) and {'name', 'from_symbol', 'to_symbol'} <= set(item) <= allowed, f'invalid {kind} fields')
    name = item['name']
    need(isinstance(name, str) and name, f'a {kind} needs a name')
    for key in ('from_symbol', 'to_symbol'):
        need(isinstance(item[key], str) and SYMBOL.fullmatch(item[key]), f'{name}: {key} must be a launch symbol')
    _selector(item.get('when', {}), f'{name}.when')
    maps = item.get('args', {})
    need(isinstance(maps, dict), f'{name}.args must be an object')
    for key, table in maps.items():
        _index(key)
        need(isinstance(table, dict) and table and all(LITERAL.fullmatch(k) and isinstance(v, str) and LITERAL.fullmatch(v)
                                                      and k != v for k, v in table.items()),
             f'{name}.args[{key}] must map scalar literals to other scalar literals')
    lengths = item.get('lengths', {})
    need(isinstance(lengths, dict), f'{name}.lengths must be an object')
    for key, table in lengths.items():
        _index(key)
        need(isinstance(table, dict) and table and all(k.isdigit() and str(int(k)) == k and type(v) is int and v >= 0 and int(k) != v
                                                      for k, v in table.items()),
             f'{name}.lengths[{key}] must map base span lengths to other head lengths')
    strides = item.get('strides', {})
    need(isinstance(strides, dict) and all(type(v) is int and v >= 0 and _index(k) != v for k, v in strides.items()),
         f'{name}.strides must map a stride argument to the argument whose span it strides')
    edited = [_index(k) for k in maps] + [_index(k) for k in lengths] + [_index(k) for k in strides]
    need(len(edited) == len(set(edited)), f'{name}: an argument has two edits')
    cases = item.get('cases', [])
    need(isinstance(cases, list), f'{name}.cases must be a list')
    for i, case in enumerate(cases):
        where = f'{name}.cases[{i}]'
        need(isinstance(case, dict) and set(case) == {'when', 'then'}, f'{where} needs when and then')
        _selector(case['when'], f'{where}.when', case=True)
        then = case['then']
        need(isinstance(then, dict) and then and set(then) <= {'symbol', 'args', *CONFIG_FIELDS}, f'{where}.then is invalid')
        if 'symbol' in then:
            need(isinstance(then['symbol'], str) and SYMBOL.fullmatch(then['symbol']), f'{where}.then.symbol must be a symbol')
        for key in CONFIG_FIELDS:
            if key in then:
                _positive(then[key], f'{where}.then.{key}', zero=True)
        for key, edit in then.get('args', {}).items():
            _index(key)
            need((isinstance(edit, str) and LITERAL.fullmatch(edit)) or
                 (isinstance(edit, dict) and set(edit) == {'length'} and type(edit['length']) is int and edit['length'] >= 0),
                 f'{where}.then.args[{key}] must be a scalar literal or a host-buffer span length')
    if collective:
        _positive(item.get('ranks'), f'{name}.ranks')
        targets = {}
        for key in ('peers', 'buffers'):
            value = item.get(key, {})
            need(isinstance(value, dict), f'{name}.{key} must be an object')
            for index, resource in value.items():
                need(_index(index) not in targets, f'{name}: argument {index} is named twice')
                need(resource in resources and resources[resource]['kind'] == 'added',
                     f'{name}.{key}[{index}] names no added resource')
                targets[int(index)] = (key, resource)
        need(targets, f'{name}: a collective names the added buffers its launches take (peers or buffers)')
        need(not (set(targets) & set(edited)), f'{name}: an argument is both edited and replaced')
    else:
        need(item['from_symbol'] != item['to_symbol'] or cases or maps or lengths or strides, f'{name}: the family changes nothing')
    return item


def parse(data):
    """The validated declaration from its bytes: `resources` keyed by name
    with their kind, `changes` the collectives and kernel families in order."""
    d = json.loads(data, object_pairs_hook=unique)
    need(isinstance(d, dict) and d.get('schema') == SCHEMA and set(d) <= FIELDS, f'not a {SCHEMA} declaration')
    need(isinstance(d.get('description'), str) and d['description'].strip(), 'a layout declaration needs a description')
    modules = d.get('modules', {})
    need(isinstance(modules, dict) and set(modules) <= {'removed', 'added'}, 'modules has removed and added lists')
    removed = _names(modules.get('removed', []), 'modules.removed')
    rules = {}
    entries = modules.get('added', [])
    need(isinstance(entries, list), 'modules.added must be a list')
    for m in entries:
        need(isinstance(m, dict) and {'name', 'loads_like'} <= set(m) <= {'name', 'loads_like', 'configures'}
             and isinstance(m['name'], str) and m['name'] and isinstance(m['loads_like'], str) and m['loads_like']
             and m['name'] not in rules and m['loads_like'] != m['name'],
             'modules.added entries name a module, the base module it loads like, and optionally its configures')
        configures = m.get('configures', {})
        need(isinstance(configures, dict) and all(SYMBOL.fullmatch(k) and isinstance(v, list) and v and
                                                  all(type(x) is int and x >= 0 for x in v) for k, v in configures.items()),
             f'{m["name"]}.configures must map symbols to their configure values')
        rules[m['name']] = dict(loads_like=m['loads_like'], configures=configures)
    added = list(rules)
    need(not (set(removed) & set(added)), 'a module is both removed and added')
    groups = d.get('resources', {})
    need(isinstance(groups, dict) and set(groups) <= {'added', 'resized'}, 'resources has added and resized lists')
    resources = {}
    for kind in ('added', 'resized'):
        entries = groups.get(kind, [])
        need(isinstance(entries, list), f'resources.{kind} must be a list')
        for r in entries:
            fields = ({'name', 'bytes', 'alignment', 'per_rank', 'lifetime', 'init', 'alongside'} if kind == 'added'
                      else {'name', 'from', 'to', 'alignment', 'per_rank'})
            need(isinstance(r, dict) and fields <= set(r) <= fields | ({'layout'} if kind == 'resized' else set()),
                 f'invalid resources.{kind} entry fields')
            need(isinstance(r['name'], str) and r['name'] and r['name'] not in resources, 'resource names must be distinct')
            _positive(r['alignment'], f'{r["name"]}.alignment')
            _positive(r['per_rank'], f'{r["name"]}.per_rank')
            if kind == 'added':
                _positive(r['bytes'], f'{r["name"]}.bytes')
                need(r['lifetime'] == 'instance', f'{r["name"]}: the only supported lifetime is instance (allocated in create)')
                need(r['init'] in ('zero', 'none'), f'{r["name"]}: init is zero or none')
                anchor = r['alongside']
                need(isinstance(anchor, dict) and set(anchor) == {'bytes', 'alignment'} and
                     all(type(anchor[k]) is int and anchor[k] > 0 for k in anchor),
                     f'{r["name"]}: alongside names the base allocation (bytes, alignment) whose ranks hold it')
            else:
                _positive(r['from'], f'{r["name"]}.from')
                _positive(r['to'], f'{r["name"]}.to')
                need(r['from'] != r['to'], f'{r["name"]}: a resize changes the size')
                if 'layout' in r:
                    lay = r['layout']
                    need(isinstance(lay, dict) and set(lay) == LAYOUT_FIELDS and all(type(lay[k]) is int and lay[k] >= 0 for k in lay)
                         and lay['planes'] >= 1 and lay['from_stride'] > 0 and lay['to_stride'] > 0, f'{r["name"]}: invalid layout')
                    need(lay['planes'] * lay['from_stride'] + lay['from_tail'] == r['from'] and
                         lay['planes'] * lay['to_stride'] + lay['to_tail'] == r['to'], f'{r["name"]}: the layout does not tile from and to')
            resources[r['name']] = dict(r, kind=kind)
    changes = []
    for key, collective in (('collectives', True), ('kernels', False)):
        entries = d.get(key, [])
        need(isinstance(entries, list), f'{key} must be a list')
        changes.extend(dict(_change(item, collective, resources), collective=collective) for item in entries)
    need(len({c['name'] for c in changes}) == len(changes), 'collective and kernel family names must be distinct')
    texts = _names(d.get('text_journals', []), 'text_journals')
    return dict(description=d['description'], removed_modules=removed, added_modules=added, added_module_rules=rules,
                resources=resources,
                changes=changes, text_journals=texts)


def read(repo, revision, path, sha256):
    """The declaration committed at `revision`, refusing a digest that is
    not the spec's. Returns (parsed, raw bytes)."""
    import hashlib
    need(isinstance(path, str) and path and not path.startswith('/') and '..' not in path.split('/'),
         'the layout declaration path must be relative to the repository')
    need(isinstance(sha256, str) and re.fullmatch('[0-9a-f]{64}', sha256), 'the layout declaration needs a SHA-256')
    data = git(repo, 'show', f'{revision}:{path}')
    need(hashlib.sha256(data).hexdigest() == sha256, f'{path} at {revision} does not have the declared SHA-256')
    return parse(data), data


def locks(repo, revision, root):
    """Every locked module of the bundle at `revision`: name -> record."""
    result = {}
    listed = git(repo, 'ls-tree', '-r', '--name-only', revision, root + 'kernels.lock.d').decode().splitlines()
    for path in listed:
        if not path.endswith('.lock'):
            continue
        rows = json.loads(git(repo, 'show', revision + ':' + path))
        need(isinstance(rows, list), f'{path}: a lock holds a module list')
        for row in rows:
            need(isinstance(row, dict) and isinstance(row.get('name'), str) and row['name'] not in result, f'{path}: duplicate or unnamed module')
            sha = row.get('cubin_sha256')
            need(isinstance(sha, str) and re.fullmatch('[0-9a-f]{64}', sha), f'{row["name"]}: invalid locked cubin SHA-256')
            result[row['name']] = dict(path=path, sha256=sha, symbols=[e['symbol'] for e in row.get('entries', [])])
    return result


def authenticate(repo, base, head, root, decl):
    """Removed modules are locked in the base and not in the head; added
    modules the reverse. Returns name -> (side, locked record)."""
    a, b = locks(repo, base, root), locks(repo, head, root)
    out = {}
    for name in decl['removed_modules']:
        need(name in a and name not in b, f'{name}: a removed module must be locked in the base and not in the head')
        out[name] = ('removed', a[name])
    for name in decl['added_modules']:
        need(name in b and name not in a, f'{name}: an added module must be locked in the head and not in the base')
        out[name] = ('added', b[name])
        like = decl['added_module_rules'][name]['loads_like']
        need(like in a, f'{name}: it loads like {like}, which is not locked in the base')
        unknown = set(decl['added_module_rules'][name]['configures']) - set(b[name]['symbols'])
        need(not unknown, f'{name}: configures name symbols it does not lock: {sorted(unknown)}')
    return out


def rank_key(row):
    """The rank an allocation belongs to: its recorded rank, else its
    unregistered (synthetic) stream, which the typed mapping keeps fixed and
    compares exactly. A test that gives each rank its own synthetic stream
    is thereby keyed per rank; one that shares a stream across ranks counts
    their allocations together, so per-rank counts refuse it."""
    return row[2] if row[2] is not None else ('stream', row[5])


def rank_label(key):
    return f'stream {key[1]}' if isinstance(key, tuple) else f'rank {key}'


class Witness:
    """One fixture's declared additions and removals, checked against the
    declaration, the locks and both inventories (auxproof.inventory)."""

    def __init__(self, decl, modules, rule, a, b):
        self.decl = decl
        res = decl['resources']
        # Added allocations: head slot -> resource; each a head allocation of
        # the declared bytes and alignment. Every rank that holds an added
        # allocation holds exactly `per_rank` of each added resource.
        self.added = {}
        per = collections.Counter()
        for entry in rule.get('added_allocations', []):
            need(isinstance(entry, dict) and set(entry) == {'resource', 'head_raw', 'rank'} and type(entry['head_raw']) is int
                 and (entry['rank'] is None or type(entry['rank']) is int), 'invalid added allocation entry')
            r = res.get(entry['resource'])
            need(r is not None and r['kind'] == 'added', f'added allocation names no added resource: {entry["resource"]}')
            slot = entry['head_raw'] & 0xffffffff
            need(slot not in self.added, 'an added allocation is listed twice')
            row = b['allocations'].get(slot)
            require(row is not None and row[1] == entry['head_raw'], f'added allocation {entry["head_raw"]} is not a head allocation')
            require((row[2], row[3], row[4]) == (entry['rank'], r['bytes'], r['alignment']),
                    f'added allocation {entry["head_raw"]} is not a {r["name"]} ({r["bytes"]} bytes, alignment {r["alignment"]}) of rank {entry["rank"]}')
            self.added[slot] = r['name']
            per[(r['name'], rank_key(row))] += 1
        # The ranks that must hold an added resource are the base ranks that
        # hold its `alongside` allocation; every other rank holds none.
        for name in (n for n, r in res.items() if r['kind'] == 'added'):
            anchor = res[name]['alongside']
            expected = {rank_key(row) for row in a['allocations'].values() if (row[3], row[4]) == (anchor['bytes'], anchor['alignment'])}
            for rank in sorted(expected | {k for n, k in per if n == name}, key=str):
                count, want = per[(name, rank)], res[name]['per_rank'] if rank in expected else 0
                require(count == want, f'{rank_label(rank)} holds {count} {name}, the declaration {want}')
        # Added allocations by (resource, rank): the head handles a rank's
        # peer entry or buffer may become.
        self.by_rank = collections.defaultdict(list)
        for slot, name in sorted(self.added.items()):
            self.by_rank[(name, rank_key(b['allocations'][slot]))].append(b['allocations'][slot][1])
        # Module removals and additions with their resolves.
        self.removed_ids, self.added_ids = set(), set()
        for kind, side, inv, ids in (('removed_modules', 'removed', a, self.removed_ids), ('added_modules', 'added', b, self.added_ids)):
            for d in rule.get(kind, []):
                need(isinstance(d, dict) and set(d) == {'name', 'fnv1a', 'bytes', 'ids', 'count'} and isinstance(d['ids'], list)
                     and type(d['count']) is int and d['count'] == len(d['ids']) == len(set(d['ids'])) > 0, f'invalid {kind} entry')
                locked = modules.get(d['name'])
                need(locked is not None and locked[0] == side, f'{d["name"]} is not a declared {side} module')
                for i in d['ids']:
                    row = inv['modules'].get(i)
                    require(row is not None and row[2:5] == [d['name'], d['fnv1a'], d['bytes']], f'{kind}: module {i} identity differs')
                    need(len(row) == 6, f'{d["name"]} (ID {i}) was loaded without a captured payload')
                    require(row[5] == locked[1]['sha256'], f'{d["name"]} (ID {i}): captured payload is not the locked SHA-256')
                    ids.add(i)
        for side, inv, ids in (('removed', a, self.removed_ids), ('added', b, self.added_ids)):
            names = {n for n, (s, _) in modules.items() if s == side}
            every = {i for i, row in inv['modules'].items() if row[2] in names}
            require(every == ids, f'the {"base" if side == "removed" else "head"} loads a declared {side} module the witness does not list')
        # An added module loads exactly as often as its declared base
        # counterpart, and every load resolves the same symbols, each once.
        for name, r in decl['added_module_rules'].items():
            loads = [i for i, row in b['modules'].items() if row[2] == name]
            like = sum(row[2] == r['loads_like'] for row in a['modules'].values())
            require(len(loads) == like, f'the head loads {name} {len(loads)} times, the base {r["loads_like"]} {like}')
            lists = {tuple(row[3] for row in b['resolves'].values() if row[2] == i) for i in loads}
            require(len(lists) <= 1, f'the loads of {name} resolve different symbols')
            for symbols in lists:
                require(len(symbols) == len(set(symbols)), f'a load of {name} resolves a symbol twice')
        resolves = {'removed': rule.get('removed_resolves', []), 'added': rule.get('added_resolves', [])}
        self.removed_resolves = self._resolves(resolves['removed'], a, self.removed_ids, modules, 'removed')
        self.added_resolves = self._resolves(resolves['added'], b, self.added_ids, modules, 'added')
        # Resized allocations: base slot -> resource.
        self.resized = {}
        per = collections.Counter()
        for d in rule.get('resized_allocations', []):
            need(isinstance(d, dict) and set(d) == {'resource', 'base_raw', 'head_raw', 'rank'}, 'invalid resized allocation entry')
            r = res.get(d['resource'])
            need(r is not None and r['kind'] == 'resized', f'resized allocation names no resized resource: {d["resource"]}')
            old, new = d['base_raw'] & 0xffffffff, d['head_raw'] & 0xffffffff
            need(old not in self.resized, 'a resized allocation is listed twice')
            ar, br = a['allocations'].get(old), b['allocations'].get(new)
            require(ar is not None and br is not None and ar[1] == d['base_raw'] and br[1] == d['head_raw'],
                    f'resized allocation {d["base_raw"]} -> {d["head_raw"]} is not a base and head allocation')
            require((ar[2], br[2], ar[3], br[3], ar[4], br[4]) == (d['rank'], d['rank'], r['from'], r['to'], r['alignment'], r['alignment']),
                    f'resized allocation {d["base_raw"]} is not a {r["name"]} ({r["from"]} -> {r["to"]} bytes) of rank {d["rank"]}')
            self.resized[old] = dict(resource=r['name'], head_slot=new, base_raw=d['base_raw'], **{'from': r['from'], 'to': r['to']})
            per[(r['name'], rank_key(ar))] += 1
        # Every base rank that holds an allocation of a resize's `from` size
        # resizes exactly `per_rank` of them.
        for name in (n for n, r in res.items() if r['kind'] == 'resized'):
            expected = {rank_key(row) for row in a['allocations'].values() if (row[3], row[4]) == (res[name]['from'], res[name]['alignment'])}
            for rank in sorted(expected | {k for n, k in per if n == name}, key=str):
                count = per[(name, rank)]
                require(count == res[name]['per_rank'], f'{rank_label(rank)} resizes {count} {name}, the declaration {res[name]["per_rank"]}')

    @staticmethod
    def _resolves(entries, inv, module_ids, modules, side):
        ids = set()
        for d in entries:
            need(isinstance(d, dict) and set(d) == {'symbol', 'ids', 'count'} and isinstance(d['ids'], list)
                 and d['count'] == len(d['ids']) == len(set(d['ids'])) > 0, f'invalid {side} resolve entry')
            for i in d['ids']:
                row = inv['resolves'].get(i)
                require(row is not None and row[3] == d['symbol'] and row[2] in module_ids,
                        f'{side} resolve {i} of {d["symbol"]} is not owned by a {side} module')
                symbols = modules[inv['modules'][row[2]][2]][1]['symbols']
                require(d['symbol'] in symbols, f'{d["symbol"]} is not an entry of its locked module')
                ids.add(i)
        every = {i for i, row in inv['resolves'].items() if row[2] in module_ids}
        require(every == ids, f'the {side} modules own resolves the witness does not list')
        return ids

    def check_stream_and_lifetime(self, b, events, create):
        """Each added allocation is allocated inside the fixture's create
        range (lifetime instance), on a stream another allocation of its rank
        uses."""
        streams = collections.defaultdict(set)
        for slot, row in b['allocations'].items():
            if slot not in self.added:
                streams[rank_key(row)].add(row[5])
        index = {e[1] & 0xffffffff: k for k, e in enumerate(events) if e[0] == 'allocation'}
        for slot, name in self.added.items():
            row = b['allocations'][slot]
            require(row[5] in streams[rank_key(row)], f'added {name} {row[1]} is on a stream no other {rank_label(rank_key(row))} allocation uses')
            require(create['start_event'] <= index[slot] < create['end_event'],
                    f'added {name} {row[1]} is not allocated inside the create range (lifetime instance)')


def allocate_notes(events):
    """Each allocation's `allocate` note, the note FakeDevice records right
    before the allocation event: slot -> (note index, note)."""
    out = {}
    notes = 0
    for k, event in enumerate(events):
        if event[0] == 'allocation':
            require(k > 0 and events[k - 1][0] == 'note' and ALLOC.fullmatch(events[k - 1][1]),
                    f'allocation {event[1]} is not recorded right after its allocate note')
            out[event[1] & 0xffffffff] = (notes - 1, events[k - 1][1])
        elif event[0] == 'note':
            notes += 1
    return out


def stream_key(item):
    return item['prefix'].split(' ', 1)[0]


HANDLE = re.compile(r'BundleDeviceBufferHandleV1 \{ raw: (\d+) \}')


def initialization_order(notes, inits):
    """Each declared zero must initialize its buffer before anything uses it.

    `inits` maps a buffer's head handle to (resource, allocate note index,
    zero note index, zero stream). The zero must follow the allocation and
    run when issued (not inside a graph capture), no record that runs before
    it may name the buffer, and every record that runs after it and names the
    buffer must be ordered after it. The zero's stream is ordered after it; a
    `wait T E` makes stream T ordered when event E was recorded on an ordered
    stream after the zero; a `synchronize S` of an ordered stream orders
    everything the host issues afterwards. A record without a stream needs
    that host order. A record's stream is its operation form's
    (history.stream_of).

    Graph capture follows FakeDevice (crates/bundle-runtime/src/fake_host.rs):
    `capture S` starts a capture on S; an event recorded on a stream in a
    capture belongs to that capture, and a stream that waits on it joins the
    capture; `captured S G N` ends it as graph G. Launches issued on the lead
    or a joined stream run when G is replayed, not when issued, so each
    `replay R G` must itself run after the zero and be ordered on R. Record
    and wait on a captured stream are graph edges and order nothing outside
    the graph. FakeDevice puts in G every launch a member issued since the
    capture began, including those issued before it joined; such launches
    are checked both when issued and at each replay.

    Anything else is not modelled and fails: a stream that is already in an
    open capture starting or joining another, a captured zero, any other
    operation on a captured stream that names a checked buffer, and a graph
    replayed inside a capture when it uses a checked buffer."""
    state = {}
    for raw, (name, allocated, zeroed, stream) in inits.items():
        require(allocated < zeroed, f'added {name} {raw}: zeroed before it is allocated')
        state[raw] = dict(name=name, zero=zeroed, streams={stream}, events=set(), host=False)
    if not state:
        return
    starts, members, event_capture, graphs = {}, {}, {}, {}
    # Per open capture: the buffers each uncaptured stream named since it began.
    issued = {}

    def capture_of(stream):
        if stream in starts:
            return stream
        return next((lead for lead, joined in members.items() if stream in joined), None)

    def ordered(raw, stream, j, what):
        st = state[raw]
        require(j > st['zero'], f'added {st["name"]} {raw}: {what} (head note {j + 1}) runs before its zero (note {st["zero"] + 1})')
        require(st['host'] or (stream is not None and stream in st['streams']),
                f'added {st["name"]} {raw}: {what} (head note {j + 1}) is not ordered after its zero (note {st["zero"] + 1})')

    def join(capture, stream, j):
        current = capture_of(stream)
        require(current in (None, capture),
                f'head note {j + 1}: stream {stream} joins the capture on {capture} while in the capture on {current}, '
                'which is not modelled')
        if current is None:
            members[capture].add(stream)
            starts[capture] |= issued[capture].pop(stream, set())

    for j, line in enumerate(notes):
        stream = history.stream_of(line)
        words = line.split()
        named = {int(r) for r in HANDLE.findall(line)} & set(state)
        if words[0] == 'capture' and stream is not None:
            require(capture_of(stream) is None,
                    f'head note {j + 1}: stream {stream} begins a capture while in the capture on {capture_of(stream)}, '
                    'which is not modelled')
            starts[stream], members[stream], issued[stream] = set(), set(), collections.defaultdict(set)
            continue
        if words[0] == 'captured' and stream is not None and stream in starts:
            graphs[int(words[2])] = starts.pop(stream)
            members.pop(stream)
            issued.pop(stream)
            event_capture = {e: lead for e, lead in event_capture.items() if lead != stream}
            continue
        lead = capture_of(stream) if stream is not None else None
        if lead is not None:
            # On a captured stream: a graph node or edge, run at the replays.
            for raw in named:
                require(j != state[raw]['zero'], f'added {state[raw]["name"]} {raw}: its zero (head note {j + 1}) is captured into a graph')
                require(words[0] == 'launch', f'added {state[raw]["name"]} {raw}: head note {j + 1} names it in a {words[0]} '
                        'on a captured stream, which is not modelled')
            starts[lead] |= named
            if words[0] == 'replay':
                require(int(words[2]) in graphs, f'head note {j + 1}: graph {words[2]} is replayed but was never captured')
                require(not graphs[int(words[2])], f'head note {j + 1}: graph {words[2]}, which uses a checked buffer, '
                        'is replayed inside a capture, which is not modelled')
            elif words[0] == 'record' and len(words) == 3:
                event_capture[int(words[2])] = lead
                for st in state.values():
                    st['events'].discard(int(words[2]))
            elif words[0] == 'wait' and len(words) == 3:
                waited = event_capture.get(int(words[2]))
                if waited is not None and waited != stream:
                    join(waited, stream, j)
            continue
        for raw in named:
            if j != state[raw]['zero']:
                ordered(raw, stream, j, 'a use')
        for pending in issued.values():
            pending[stream] |= named - {raw for raw in named if j == state[raw]['zero']}
        if words[0] == 'replay':
            require(int(words[2]) in graphs, f'head note {j + 1}: graph {words[2]} is replayed but was never captured')
            for raw in graphs[int(words[2])]:
                ordered(raw, stream, j, f'graph {words[2]} replay')
        if words[0] == 'record' and len(words) == 3:
            event_capture.pop(int(words[2]), None)
        if words[0] == 'wait' and len(words) == 3:
            waited = event_capture.get(int(words[2]))
            if waited is not None and waited != stream:
                # Waiting on a capture's event joins the stream to it.
                join(waited, stream, j)
                continue
        for raw, st in state.items():
            if j <= st['zero']:
                continue
            if words[0] == 'synchronize' and stream in st['streams']:
                st['host'] = True
            elif words[0] == 'record' and len(words) == 3:
                (st['events'].add if st['host'] or stream in st['streams'] else st['events'].discard)(int(words[2]))
            elif words[0] == 'wait' and len(words) == 3 and int(words[2]) in st['events']:
                st['streams'].add(stream)


def reads_only(line, raw):
    """Whether `line` only reads buffer `raw`: as the source of a download or
    readback, or only in the source span of a copy or peer copy. Any other
    record naming it writes it or may, as far as the proof can tell."""
    found = history.form(line)
    if found is None:
        return False
    kind = found[0]
    spans = [int(r) for r in HANDLE.findall(line)]
    if kind == 'download':
        return True
    if kind == 'copy':
        return len(spans) == 2 and spans[0] == raw and spans[1] != raw
    return False


PLAN_RECORD = re.compile(r'launch r\d+ .+')


def plan_only(notes):
    """Whether a device journals only `allocate` notes and `launch rN` plan
    records, which lowering tests write with `device.note` to describe the
    launches they plan; nothing on such a device executes. Any other record
    (a stream, zero, upload, copy, synchronize, record, wait, readback or an
    executed `launch S` on a stream) makes it an executing device."""
    for line in notes:
        found = history.form(line)
        if found is None or not (found[0] == 'allocate' or (found[0] == 'launch' and PLAN_RECORD.fullmatch(line))):
            return False
    return True


def unzeroed_uses(notes_a, notes_b, zeroed, replaced, sites, plan=False):
    """A device whose create is only its allocation prefix never runs the
    instance create that zeroes a declared `init: zero` buffer. The head may
    then use such a buffer unzeroed only where it keeps the base's
    initialization: every head record naming it must come from a declared
    collective argument (`replaced` maps the added head handle to the base
    buffers it replaces there), and at each such argument (`sites` maps the
    added handle to (base buffer, base note index) pairs) no earlier base
    record may write the replaced buffer. Only reads_only records and the
    other replacement sites of that buffer, which the head runs on the added
    buffer instead, leave it unwritten; any other record naming it counts as
    initializing it. With `plan` (both revisions' devices are plan_only)
    plan records do not count as writes, because nothing executes. `zeroed`
    maps added head handles to resources."""
    replaced_at = collections.defaultdict(set)
    for added in sites.values():
        for base, i in added:
            replaced_at[base].add(i)
    for j, line in enumerate(notes_b):
        for raw in {int(r) for r in HANDLE.findall(line)} & set(zeroed):
            name = zeroed[raw]
            require(replaced.get(raw), f'added {name} {raw}: head note {j + 1} uses it on a device that never runs the instance '
                    'create, and no declared collective argument identifies the base buffer it replaces')
    for raw, name in zeroed.items():
        for base, i in sorted(sites.get(raw, ())):
            for k in range(i):
                if k in replaced_at[base] or base not in {int(r) for r in HANDLE.findall(notes_a[k])}:
                    continue
                if plan and PLAN_RECORD.fullmatch(notes_a[k]):
                    continue
                require(reads_only(notes_a[k], base),
                        f'added {name} {raw}: it is used unzeroed on a device that never runs the instance create, but base '
                        f'note {k + 1} writes {base}, which it replaces, before the collective at base note {i + 1} reads it')


def declared_record(line, ids, resolves, j, side):
    """The kind (module, resolve or configure) of `line` when it is the
    record of a declared added or removed ID in `ids`, else None. Such a line
    must have its operation form exactly, and a resolve must name the symbol
    its trace row resolves."""
    m = re.match(r'^(module|resolve|configure) (\d+) ', line)
    if not m or int(m[2]) not in ids:
        return None
    found = history.form(line)
    require(found is not None and found[0] == m[1], f'{side} note {j + 1}: the record of declared ID {m[2]} has no {m[1]} form')
    if m[1] == 'resolve':
        require(line == f'resolve {m[2]} {resolves[int(m[2])][3]}',
                f'{side} note {j + 1}: resolve {m[2]} does not name the symbol its trace resolves')
    return m[1]


def selects(sel, item, previous):
    """Whether selector `sel` picks base launch `item`, whose previous
    launch on the same stream is `previous`."""
    for key in CONFIG_FIELDS:
        if key in sel and item['fields'].get(key) != sel[key]:
            return False
    if 'rows' in sel and item['rows'] != sel['rows']:
        return False
    if 'prefix' in sel and item['prefix'] != sel['prefix']:
        return False
    for key, literal in sel.get('args', {}).items():
        i = int(key)
        if i >= len(item['args']) or item['args'][i] != literal:
            return False
    if 'after' in sel:
        after = sel['after']
        if previous is None or previous['symbol'] not in after['symbols']:
            return False
        if any(key in after and previous['fields'].get(key) != after[key] for key in CONFIG_FIELDS):
            return False
    return True


class Device:
    """The expected head line of every base note of one fixture device, and
    the head notes that are declared additions.

    `expected[i]` is None for a dropped base note (a record of a removed
    module or resolve); `changes[i]` names what rewrote it. `added[j]` names
    the declaration that adds head note j."""

    def __init__(self, decl, witness, ae, be, ai, bi, head_create):
        self.decl, self.w = decl, witness
        notes_a = [e[1] for e in ae if e[0] == 'note']
        notes_b = [e[1] for e in be if e[0] == 'note']
        alloc_a = allocate_notes(ae)
        alloc_b = allocate_notes(be)
        self.base_sizes = {row[1]: row for row in ai['allocations'].values()}
        # Added head handle -> the base buffers it replaces in declared
        # collective arguments (Device.peer).
        self.replaced = collections.defaultdict(set)
        # Added head handle -> (base buffer, base note index) of each
        # replacement, and the base note being transformed.
        self.sites = collections.defaultdict(set)
        self.note_index = None
        self.resized = {w['base_raw']: w for w in witness.resized.values()}
        resized_notes = {alloc_a[slot][0]: (slot, w) for slot, w in witness.resized.items()}
        self.expected, self.changes = [], []
        last = {}
        dropped = witness.removed_ids | witness.removed_resolves
        for i, line in enumerate(notes_a):
            self.note_index = i
            tags = collections.Counter()
            item = launch(line) if line.startswith('launch ') else None
            previous = last.get(stream_key(item)) if item else None
            if item:
                last[stream_key(item)] = item
            kind = declared_record(line, dropped, ai['resolves'], i, 'base')
            if kind:
                self.expected.append(None)
                tags['removed_' + kind + 's'] += 1
                self.changes.append(tags)
                continue
            if i in resized_notes:
                slot, w = resized_notes[i]
                n = ALLOC.fullmatch(line)
                require(n is not None and int(n[1]) == w['from'], f'base note {i + 1} is not the allocate note of resized {w["resource"]}')
                line = f'allocate {w["to"]} {n[2]}' + (f' -> raw: {n[3]}' if n[3] else '')
                tags['resized_allocations'] += 1
            if item:
                line = self.launch(item, previous, line, tags)
            line = self.spans(line, tags)
            self.expected.append(line)
            self.changes.append(tags)
        # Head notes that are declared additions.
        self.added = {}
        for slot, name in witness.added.items():
            self.added[alloc_b[slot][0]] = 'added_allocations'
            n = ALLOC.fullmatch(alloc_b[slot][1])
            require(n is not None and int(n[1]) == bi['allocations'][slot][3], f'the allocate note of added {name} differs')
        # An added resolve's configure records follow it directly and are
        # exactly its module's declared values for the symbol.
        configured = collections.defaultdict(list)
        added = witness.added_ids | witness.added_resolves
        for j, line in enumerate(notes_b):
            kind = declared_record(line, added, bi['resolves'], j, 'head')
            if kind:
                m = re.match(r'^(module|resolve|configure) (\d+) ', line)
                self.added[j] = 'added_' + kind + 's'
                if kind == 'configure':
                    previous = re.match(r'^(resolve|configure) (\d+)( |$)', notes_b[j - 1]) if j else None
                    require(previous is not None and previous[2] == m[2], f'head note {j + 1}: configure {m[2]} is detached from its resolve')
                    configured[int(m[2])].append(int(line.split()[2]))
        for ident in witness.added_resolves:
            row = bi['resolves'][ident]
            module = bi['modules'][row[2]][2]
            want = decl['added_module_rules'][module]['configures'].get(row[3], [])
            require(configured[ident] == want, f'added resolve {ident} of {row[3]} is configured {configured[ident]}, the declaration {want}')
        zeroed = {raw: name for slot, name in witness.added.items()
                  if decl['resources'][name]['init'] == 'zero' for raw in [bi['allocations'][slot][1]]}
        # A zero counts only on the stream the buffer was allocated on.
        streams = {row[1]: row[5] & 0xffffffff for row in bi['allocations'].values()}
        found, at = collections.Counter(), {}
        for j in range(head_create['start_note'], head_create['end_note']):
            m = re.fullmatch(r'zero (\d+) ' + SPAN.pattern, notes_b[j])
            if (m and int(m[2]) in zeroed and int(m[1]) == streams[int(m[2])] and int(m[3]) == 0
                    and int(m[4]) == decl['resources'][zeroed[int(m[2])]]['bytes']):
                found[int(m[2])] += 1
                at[int(m[2])] = j
                self.added[j] = 'added_zeros'
        # An instance create zeroes each such buffer once; a device whose
        # create is only its allocation prefix (a test that never ran the
        # instance create) zeroes none.
        expected = 0 if head_create['kind'] == 'direct-allocation-prefix' else 1
        for raw, name in zeroed.items():
            require(found[raw] == expected,
                    f'added {name} {raw}: {found[raw]} whole zeros in the create range ({head_create["kind"]}), the declaration {expected}')
        if not expected:
            unzeroed_uses(notes_a, notes_b, zeroed, self.replaced, self.sites, plan_only(notes_a) and plan_only(notes_b))
        # The zero is not compared (it is a declared addition), so its place
        # relative to the buffer's uses is checked here.
        initialization_order(notes_b, {raw: (name, alloc_b[raw & 0xffffffff][0], at[raw], streams[raw])
                                       for raw, name in zeroed.items() if raw in at})

    def launch(self, item, previous, line, tags):
        chosen = [c for c in self.decl['changes'] if c['from_symbol'] == item['symbol'] and selects(c.get('when', {}), item, previous)]
        require(len(chosen) <= 1, f'{item["symbol"]}: declared changes {[c["name"] for c in chosen]} all select one launch')
        if not chosen:
            return line
        change = chosen[0]
        cases = [c for c in change.get('cases', []) if selects(c['when'], item, previous)]
        require(len(cases) <= 1, f'{change["name"]}: {len(cases)} declared cases match one launch')
        then = cases[0]['then'] if cases else {}
        new = dict(item, fields=dict(item['fields']), args=list(item['args']))
        new['symbol'] = then.get('symbol', change['to_symbol'])
        regrid = False
        for key in CONFIG_FIELDS:
            if key in then and then[key] != item['fields'][key]:
                new['config'], n = re.subn(r'\b' + key + r': \d+\b', f'{key}: {then[key]}', new['config'])
                need(n == 1, f'ambiguous config field {key}')
                new['fields'][key] = then[key]
                regrid = True
        for key, table in change.get('args', {}).items():
            i = int(key)
            require(i < len(new['args']), f'{change["name"]}: argument {i} is out of range')
            if new['args'][i] in table:
                new['args'][i] = table[new['args'][i]]
                tags['argument_literals'] += 1
        for key, table in change.get('lengths', {}).items():
            i = int(key)
            m = self.host_span(new['args'], i, change)
            if m[3] in table:
                new['args'][i] = new['args'][i].replace(f'length: {m[3]} }}', f'length: {table[m[3]]} }}')
                tags['host_span_lengths'] += 1
        for key, source in change.get('strides', {}).items():
            i = int(key)
            require(max(i, source) < len(new['args']), f'{change["name"]}: argument {max(i, source)} is out of range')
            spans = [int(m[1]) for m in SPAN.finditer(item['args'][source])]
            require(len(spans) == 1, f'{change["name"]}: argument {source} is not one buffer span')
            if spans[0] not in self.resized:
                continue
            r = self.decl['resources'][self.resized[spans[0]]['resource']]
            require('layout' in r, f'{change["name"]}: argument {source} spans {r["name"]}, which declares no layout')
            m = re.fullmatch(r'Scalar(U32|U64)\((\d+)\)', new['args'][i])
            require(m is not None and int(m[2]) == r['layout']['from_stride'],
                    f'{change["name"]}: argument {i} is not the plane stride of {r["name"]}')
            new['args'][i] = f'Scalar{m[1]}({r["layout"]["to_stride"]})'
            tags['layout_strides'] += 1
        for key, edit in then.get('args', {}).items():
            i = int(key)
            require(i < len(new['args']), f'{change["name"]}: argument {i} is out of range')
            if isinstance(edit, str):
                require(LITERAL.fullmatch(new['args'][i]), f'{change["name"]}: a case replaces argument {i}, which is not a '
                        'scalar literal')
                if new['args'][i] != edit:
                    new['args'][i] = edit
                    tags['argument_literals'] += 1
                continue
            m = self.host_span(new['args'], i, change)
            new['args'][i] = new['args'][i].replace(f'length: {m[3]} }}', f'length: {edit["length"]} }}')
            tags['host_span_lengths'] += 1
        if change['collective']:
            for key, resource in change.get('peers', {}).items():
                i = int(key)
                require(i < len(new['args']), f'{change["name"]}: argument {i} is out of range')
                m = re.fullmatch(r'DevicePointerArray\(\[(.*)\]\)', new['args'][i])
                require(m is not None, f'{change["name"]}: argument {i} is not a peer array')
                entries = [tuple(map(int, s.groups())) for s in SPAN.finditer(m[1])]
                require(len(entries) == change['ranks'] and m[1].count('Some(') == len(entries) and 'None' not in m[1],
                        f'{change["name"]}: argument {i} holds {len(entries)} peers, the declaration {change["ranks"]} ranks')
                keys = [rank_key(self.base_sizes[raw]) if raw in self.base_sizes else None for raw, _, _ in entries]
                require(None not in keys and len(set(keys)) == len(keys),
                        f'{change["name"]}: argument {i} does not name {change["ranks"]} distinct ranks')
                new['args'][i] = 'DevicePointerArray([' + ', '.join(
                    f'Some({self.peer(raw, resource, change)})' for raw, _, _ in entries) + '])'
                tags['collective_peer_arrays'] += 1
            for key, resource in change.get('buffers', {}).items():
                i = int(key)
                require(i < len(new['args']), f'{change["name"]}: argument {i} is out of range')
                m = re.fullmatch(r'DeviceBuffer\(' + SPAN.pattern + r'\)', new['args'][i])
                require(m is not None, f'{change["name"]}: argument {i} is not a buffer span')
                new['args'][i] = f'DeviceBuffer({self.peer(int(m[1]), resource, change)})'
                tags['collective_buffers'] += 1
            tags['collective_launches'] += 1
        else:
            tags['family_launches'] += 1
        if new['symbol'] != item['symbol']:
            tags['renamed_launches'] += 1
        if regrid:
            tags['grid_changes'] += 1
        return render(new)

    def host_span(self, args, i, change):
        """Argument i as a span of a buffer the device never allocated (a
        host-provided buffer), whose length a declaration may set."""
        require(i < len(args), f'{change["name"]}: argument {i} is out of range')
        m = re.fullmatch(r'DeviceBuffer\(' + SPAN.pattern + r'\)', args[i])
        require(m is not None, f'{change["name"]}: argument {i} is not a buffer span')
        require(int(m[1]) not in self.base_sizes, f'{change["name"]}: argument {i} is a device allocation; declare a resized resource')
        return m

    def peer(self, raw, resource, change):
        """The same rank's added `resource`, whole, for a base buffer."""
        row = self.base_sizes.get(raw)
        require(row is not None, f'{change["name"]}: a replaced buffer {raw} is not a base allocation')
        head = self.w.by_rank.get((resource, rank_key(row)), [])
        require(len(head) == 1, f'{change["name"]}: {rank_label(rank_key(row))} holds {len(head)} added {resource}, a collective takes one')
        self.replaced[head[0]].add(raw)
        self.sites[head[0]].add((raw, self.note_index))
        return placeholder_span(head[0], self.decl['resources'][resource]['bytes'])

    def spans(self, line, tags):
        """Every span of a resized allocation, mapped by its form."""
        def image(m):
            raw, offset, length = map(int, m.groups())
            w = self.resized.get(raw)
            if w is None:
                return m[0]
            r = self.decl['resources'][w['resource']]
            if (offset, length) == (0, r['from']):
                new = (0, r['to'])
            else:
                require('layout' in r, f'span ({offset}, {length}) of resized {r["name"]} is not whole and the resource declares no layout')
                new = layout_image(r['layout'], r['from'], r['to'], offset, length, r['name'])
            tags['resized_spans'] += 1
            return f'DeviceBufferSpanV1 {{ buffer: BundleDeviceBufferHandleV1 {{ raw: {raw} }}, offset: {new[0]}, length: {new[1]} }}'
        return SPAN.sub(image, line)


def layout_image(lay, base_bytes, head_bytes, offset, length, name):
    """The head span of a base span inside a plane layout, by its form:
    a whole plane, inside one plane, one entry in every plane (read with the
    plane stride), or inside the tail."""
    planes, old, new = lay['planes'], lay['from_stride'], lay['to_stride']
    start = planes * old
    if offset >= start:
        require(offset + length <= base_bytes and offset - start + length <= lay['to_tail'],
                f'span ({offset}, {length}) of {name} leaves the tail')
        return planes * new + offset - start, length
    layer, within = divmod(offset, old)
    if within == 0 and length == old:
        return layer * new, new
    if within + length <= old:
        require(within + length <= new, f'span ({offset}, {length}) of {name} crosses a head plane boundary')
        return layer * new + within, length
    entry = length - (planes - 1) * old
    require(planes >= 2 and layer == 0 and 0 < entry and within + entry <= min(old, new),
            f'span ({offset}, {length}) of {name} is neither a plane, inside a plane, one entry of every plane nor in the tail')
    return offset, (planes - 1) * new + entry


def owner_lines(device, notes_b, base_positions, head_positions, base_lines):
    """One owner's expected lines (base handles, placeholders for added
    ones), its observed head lines, and the provenance the order check uses:
    `trace[k]` the owner-local base line expected line k derives from, `kept[k]`
    the owner-local head line observed line k is, `originals[k]` the base line
    expected line k replaces, and the counts of the owner's changes."""
    expected, trace, originals = [], [], []
    counts = collections.Counter()
    for k, i in enumerate(base_positions):
        counts.update(device.changes[i])
        if device.expected[i] is None:
            continue
        expected.append(device.expected[i])
        trace.append((k,))
        originals.append(base_lines[k])
    observed, kept = [], []
    for k, j in enumerate(head_positions):
        if j in device.added:
            counts[device.added[j]] += 1
            continue
        observed.append(notes_b[j])
        kept.append(k)
    return expected, trace, originals, observed, kept, counts
