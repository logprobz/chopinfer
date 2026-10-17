#!/usr/bin/env python3
"""Differential mutation harness for the declared layout rules (layout.py).

A layout declaration may replace a base buffer in a collective argument with
an added `init: zero` buffer. On a device whose create is only its allocation
prefix nothing zeroes the added buffer, so the head may use it only where the
base buffer it replaces is unwritten when the collective reads it
(layout.unzeroed_uses). On a device that runs the instance create, the
declared zero must initialize the added buffer before any use
(layout.initialization_order).

The harness starts from a proven base and head history of one device fixture:
a capture pair under its committed declaration (`pair`), or a plan view
synthesized from any capture (`overlay`). It mirrors one mutation into both
histories at the same place, or changes the head's declared zero, and proves
the pair again with auxproof.prove. An oracle written apart from layout.py
states each mutation's expected outcome from the journal alone:

- refuse: the mutated base may write the replaced buffer before a collective
  reads it; a launch names it in an argument the head does not replace, so
  the journal cannot tell whether it is read or written; or the head's
  declared zero no longer precedes the added buffer's uses;
- prove: a safe control, such as a read or a copy out of the replaced buffer,
  a write of an unrelated buffer, a plan record on a plan-only device, or the
  declared zero moved right after its allocation.

A refuse mutation that proves is an escape; a prove mutation that is refused
is a false refusal. Writes of the replaced buffer on a device that runs the
instance create are outside the rules (the declared zero initializes the
added buffer there) and are only reported (`scope`).

Each write class runs against a deliberately weakened rule (`--weaken`,
WEAKENED); the harness must then report escapes of that class.
"""
import argparse
import collections
import contextlib
import copy
import json
import os
import re
import sys
from pathlib import Path

import auxproof
import history
import layout
from common import Invalid, fnv, need
from engine import Unproven, render
from engine import launch as parse_launch

GEN = 1 << 32
_SPAN = r'DeviceBufferSpanV1 \{ buffer: BundleDeviceBufferHandleV1 \{ raw: (\d+) \}, offset: \d+, length: \d+ \}'
HANDLES = re.compile(r'BundleDeviceBufferHandleV1 \{ raw: (\d+) \}')
PEERS = re.compile(r'DevicePointerArray\(\[(.*)\]\)')
CONFIG = ('LaunchConfigV1 { flags: 1, grid_x: 1, grid_y: 1, grid_z: 1, block_x: 32, block_y: 1, block_z: 1, '
          'shared_memory_bytes: 0, cluster: LaunchClusterV1 { x: 1, y: 1, z: 1 } }')

# The oracle's own reading of the journal forms (history.FORMS holds the
# proof's): what reads a buffer, what is a plan record, what is an allocation.
READS = [re.compile(rf'download \d+ {_SPAN}'), re.compile(rf'readback \d+ {_SPAN} \d+')]
COPIES = [re.compile(rf'copy \d+ {_SPAN} {_SPAN}'), re.compile(rf'peer copy \d+ \d+ {_SPAN} \d+ {_SPAN}')]
PLAN = re.compile(r'launch r\d+ .+')
# A plan record's rank label no real rank uses, so an inserted plan record is
# never the previous launch a declared selector (`after`) reads on a rank.
PLAN_RANK = 9999
ALLOCATE = re.compile(r'allocate \d+ \d+(?: -> raw: \d+)?')

# Mutation classes. WRITES write the replaced buffer, or may as far as the
# journal tells; EXECUTED is an executed launch that names only an unrelated
# buffer, which makes a plan-only device executing; CONTROLS are reads and
# unrelated writes; PLAN_RECORD names the replaced buffer in a plan record.
WRITES = ('zero', 'upload', 'copy-into', 'peer-copy-into', 'launch-argument')
EXECUTED = 'executed-launch'
CONTROLS = ('download', 'readback', 'copy-out', 'peer-copy-out', 'unrelated-zero')
PLAN_RECORD = 'plan-record'
# A site launch (an earlier one, or the site itself) that also names the
# replaced buffer in another argument.
SITE_ARGUMENT = 'site-argument'
# One added buffer standing for two base regions; a graph captured and
# replayed on an executing device; a write on another stream after an
# executed site, before the host waits for it.
REGION, GRAPH_REPLAY, LATER_WRITE = 'region', 'graph-replay', 'later-write'
# Head-only changes of an added buffer's declared zero.
ZEROS = ('zero-deleted', 'zero-after-use', 'zero-other-stream', 'zero-earliest')


def named(line):
    return {int(r) for r in HANDLES.findall(line)}


def span(raw, length, offset=0):
    return f'DeviceBufferSpanV1 {{ buffer: BundleDeviceBufferHandleV1 {{ raw: {raw} }}, offset: {offset}, length: {length} }}'


def plan_device(notes):
    return all(ALLOCATE.fullmatch(line) or PLAN.fullmatch(line) for line in notes)


def reads(line, raw):
    """Whether `line` only reads `raw`: a download or readback of it, or a
    copy or peer copy whose source, and not its destination, is it."""
    if any(p.fullmatch(line) for p in READS):
        return True
    for p in COPIES:
        m = p.fullmatch(line)
        if m:
            return int(m[1]) == raw and int(m[2]) != raw
    return False


STREAM = re.compile(r'(?:launch|upload|copy|peer copy|download|readback|zero|synchronize|record|wait|capture|captured|replay) (\d+)\b')
GRAPH = re.compile(r'(?:capture|captured|replay) \d+.*')
SPAN_OF = re.compile(r'DeviceBufferSpanV1 \{ buffer: BundleDeviceBufferHandleV1 \{ raw: (\d+) \}, offset: (\d+), length: (\d+) \}')


def stream(line):
    m = STREAM.match(line)
    return int(m[1]) if m else None


def oracle(notes_a, notes_b, pairs, sites, buffers):
    """('refuse', why) or ('prove', None) for a base and head history of a
    device without instance create. `sites` lists (added head handle, base
    buffer, base note) for every declared collective argument that replaces a
    base buffer with an `init: zero` added one; `pairs` maps a kept base note
    to its head note; `buffers` maps base handles to head handles.

    The head keeps the base's initialization only if, at each site, nothing
    before it may have written the base buffer: a record naming it is a write
    unless it only reads it (`reads`), it is a plan record on a device where
    both revisions hold only allocations and plan records, or it is an earlier
    site where the same added buffer replaces every argument that names it
    (the head then makes that write to the added buffer). The journal cannot
    tell, and so the proof must refuse:
    - a site launch that also names the base buffer in an argument the head
      does not replace with the same added buffer (which argument the launch
      writes first);
    - an added buffer standing for more than one base region (buffer, offset,
      length) across its sites;
    - on an executing device, any graph capture or replay (a captured site
      runs at its replays);
    - on an executing device, a write of the base buffer on another stream
      after an executed site, before the host synchronizes the site's
      stream (which runs first)."""
    plan = plan_device(notes_a) and plan_device(notes_b)
    own = collections.defaultdict(set)
    for h, b, i in sites:
        own[(h, b)].add(i)

    def replaced(k, h, b):
        base, head = parse_launch(notes_a[k]), parse_launch(notes_b[pairs[k]])
        if base is None or head is None or len(base['args']) != len(head['args']):
            return None
        return [(x, y) for x, y in zip(base['args'], head['args']) if b in named(x)]

    def replaced_only(k, h, b):
        args = replaced(k, h, b)
        return args is not None and all(h in named(y) and buffers.get(b) not in named(y) for _, y in args)

    regions = collections.defaultdict(set)
    for h, b, i in sites:
        for x, y in replaced(i, h, b) or []:
            if h in named(y):
                regions[h] |= {(b, int(m[2]), int(m[3])) for m in SPAN_OF.finditer(x) if int(m[1]) == b}
    for h, found in sorted(regions.items()):
        if len(found) > 1:
            return 'refuse', f'{h} stands for base regions {sorted(found)}'
    if not plan and sites:
        k = next((k for k, line in enumerate(notes_a) if GRAPH.fullmatch(line)), None)
        if k is not None:
            return 'refuse', f'base note {k + 1} captures or replays a graph on an executing device'
    for h, b, i in sorted(sites, key=lambda s: s[2]):
        if not replaced_only(i, h, b):
            return 'refuse', f'the collective at base note {i + 1} names {b} in an argument the head does not replace with {h}'
        for k in range(i):
            line = notes_a[k]
            if b not in named(line) or reads(line, b) or (plan and PLAN.fullmatch(line)):
                continue
            if k in own[(h, b)] and replaced_only(k, h, b):
                continue
            return 'refuse', f'base note {k + 1} may write {b} before the collective at base note {i + 1} reads it'
        site = None if plan else stream(notes_a[i])
        for k in range(i + 1, len(notes_a) if site is not None else i + 1):
            line = notes_a[k]
            if line == f'synchronize {site}':
                break
            if (b in named(line) and not reads(line, b) and stream(line) not in (None, site)
                    and not (k in own[(h, b)] and replaced_only(k, h, b))):
                return 'refuse', f'base note {k + 1} may write {b} on another stream before the collective at base note {i + 1} reads it'
    return 'prove', None


class Case:
    """One device fixture's proof inputs in memory: `bodies` and `aux` are the
    (base, head) pinned bodies and loaded auxiliary captures (auxproof.load's
    shape), `rule` the fixture's auxiliary rule with its layout witness, and
    `layout` the parsed declaration with its authenticated modules."""

    def __init__(self, label, journals, bodies, aux, rule, layout_, declared=None):
        self.label, self.journals, self.bodies, self.aux = label, list(journals), tuple(bodies), tuple(aux)
        self.rule, self.layout = rule, layout_
        self.declared = declared or dict(base=[], head=[])

    def bindings(self, side):
        aux = self.aux[side]
        return [aux['manifest']['bindings'][n] for n in self.journals + self.declared['base' if side == 0 else 'head']]

    def host(self, side):
        return self.bindings(side)[0]['host']

    def prefix(self, side):
        """The device events through the fixture's last pinned checkpoint."""
        return self.aux[side]['hosts'][self.host(side)]['events'][:max(b['records'] for b in self.bindings(side))]


def prove(case):
    """(proven, first reason) of one fixture under its declaration."""
    fixture = dict(name=case.label, journals=case.journals, declared_journals=case.declared, piece_policy=None)
    try:
        results, _, _ = auxproof.prove(case.bodies[0], case.bodies[1], fixture, {}, case.aux[0], case.aux[1], case.rule, (),
                                       layout=case.layout)
    except (Unproven, Invalid) as error:
        return False, str(error)
    failed = sorted((n, v['reasons'][0]) for n, v in results.items() if v['verdict'] == 'UNPROVEN')
    return not failed, f'{failed[0][0]}: {failed[0][1]}' if failed else None


class Facts:
    """What the unmutated, proven case declares, read through the proof's own
    derivation (auxproof.derive, layout.Device): the replacement sites, the
    added buffers, the base-to-head note pairing and the typed mapping."""

    def __init__(self, case):
        ae, be = case.prefix(0), case.prefix(1)
        buffers, _, streams, _, ai, bi, witness = auxproof.derive(ae, be, case.rule, (), case.layout)
        create = case.bindings(1)[0]['create']
        device = layout.Device(case.layout[0], witness, ae, be, ai, bi, create)
        self.notes = ([e[1] for e in ae if e[0] == 'note'], [e[1] for e in be if e[0] == 'note'])
        self.kind = create['kind']
        self.create = (case.bindings(0)[0]['create'], create)
        self.buffers, self.streams = buffers, streams
        self.sizes = {row[1]: row[3] for row in ai['allocations'].values()}
        self.issued = set(ai['issued']) | set(bi['issued'])
        self.top = (max(ai['issued'], default=0), max(bi['issued'], default=0))
        self.resized = {w['base_raw'] for w in witness.resized.values()}
        decl = case.layout[0]
        self.added = {bi['allocations'][slot][1]: name for slot, name in witness.added.items()}
        self.zeroed = {h: name for h, name in self.added.items() if decl['resources'][name]['init'] == 'zero'}
        self.alloc_notes = {bi['allocations'][slot][1]: index for slot, (index, _) in layout.allocate_notes(be).items()
                            if bi['allocations'][slot][1] in self.added}
        self.alloc_streams = {row[1]: row[5] & 0xffffffff for row in bi['allocations'].values()}
        self.head_streams = sorted(bi['streams'])
        self.sites = sorted((h, b, i) for h, items in device.sites.items() if h in self.zeroed for b, i in items)
        kept_a = [i for i, line in enumerate(device.expected) if line is not None]
        kept_b = [j for j in range(len(self.notes[1])) if j not in device.added]
        need(len(kept_a) == len(kept_b), f'{case.label}: kept base and head notes do not pair')
        self.pairs = dict(zip(kept_a, kept_b))
        self.zeros = {}
        for j, what in device.added.items():
            if what == 'added_zeros':
                h = next(iter(named(self.notes[1][j]) & set(self.zeroed)))
                self.zeros[h] = j
        # Each site is a launch of a declared collective's base symbol that
        # names the base buffer, and its head launch names the added buffer.
        symbols = {c['from_symbol'] for c in decl['changes'] if c['collective']}
        for h, b, i in self.sites:
            item = parse_launch(self.notes[0][i])
            need(item is not None and item['symbol'] in symbols and b in named(self.notes[0][i])
                 and h in named(self.notes[1][self.pairs[i]]), f'{case.label}: base note {i + 1} is not a declared collective site')


def edit_notes(notes, edits):
    """`notes` after `edits` ((kind, original index, line)): `insert` before
    the original note, `delete` and `replace` it (an `id` edit issues an ID
    there and adds no note). Returns the new list and the new index of every
    kept original note."""
    tagged = [(k, line) for k, line in enumerate(notes)] + [(len(notes), None)]
    for kind, n, *line in sorted((e for e in edits if e[0] != 'id'), key=lambda e: e[1], reverse=True):
        at = next(p for p, (k, _) in enumerate(tagged) if k == n)
        if kind == 'insert':
            tagged.insert(at, (None, line[0]))
        elif kind == 'delete':
            del tagged[at]
        else:
            tagged[at] = (n, line[0])
    tagged.pop()
    return [line for _, line in tagged], {k: p for p, (k, _) in enumerate(tagged) if k is not None}


def edit_side(aux, bodies, host, edits):
    """The auxiliary capture and bodies of one revision after note `edits` on
    `host`: every binding's positions, record count, create range and lines,
    and every pinned body that holds an edited note."""
    events = aux['hosts'][host]['events']
    tagged = [(k, e) for k, e in enumerate(events)]
    note_event = [k for k, e in enumerate(events) if e[0] == 'note']
    for kind, n, *line in sorted(edits, key=lambda e: e[1], reverse=True):
        target = note_event[n] if n < len(note_event) else len(events)
        at = next((p for p, (k, _) in enumerate(tagged) if k == target), len(tagged))
        if kind == 'insert':
            tagged.insert(at, (None, ['note', line[0]]))
        elif kind == 'id':
            tagged.insert(at, (None, ['id', line[0]]))
        elif kind == 'delete':
            del tagged[at]
        else:
            tagged[at] = (target, ['note', line[0]])
    new_events = [e for _, e in tagged]
    event_map = {k: p for p, (k, _) in enumerate(tagged) if k is not None}
    new_notes = [e[1] for e in new_events if e[0] == 'note']
    note_of_event = {k: n for n, k in enumerate(note_event)}
    note_index = {}
    count = 0
    for k, e in tagged:
        if e[0] == 'note':
            if k is not None:
                note_index[note_of_event[k]] = count
            count += 1

    def last(old, start, end):
        kept = [old[k] for k in range(start, end) if k in old]
        return kept[-1] + 1 if kept else None

    bindings = dict(aux['manifest']['bindings'])
    bodies = dict(bodies)
    for name, b in aux['manifest']['bindings'].items():
        if b['host'] != host:
            continue
        b = copy.deepcopy(b)
        positions = [note_index[p] for p in b['positions'] if p in note_index]
        b['records'] = last(event_map, 0, b['records'])
        c = b['create']
        if c['end_note'] > c['start_note']:
            start = min(note_index[k] for k in range(c['start_note'], c['end_note']) if k in note_index)
            c['end_note'], c['start_note'] = last(note_index, c['start_note'], c['end_note']), start
            c['end_event'] = last(event_map, c['start_event'], c['end_event'])
            c['start_event'] = min(event_map[k] for k in range(c['start_event'], len(events)) if k in event_map)
        b['create_lines'] = new_notes[c['start_note']:c['end_note']]
        b['positions'] = positions
        bindings[name] = b
        if name in bodies:
            bodies[name] = [new_notes[p] for p in positions]
    hosts = dict(aux['hosts'])
    hosts[host] = dict(aux['hosts'][host], events=new_events, notes=new_notes)
    return dict(aux, manifest=dict(aux['manifest'], bindings=bindings), hosts=hosts), bodies


def mutated(case, base_edits, head_edits):
    sides = [edit_side(case.aux[s], case.bodies[s], case.host(s), edits) for s, edits in ((0, base_edits), (1, head_edits))]
    return Case(case.label, case.journals, [s[1] for s in sides], [s[0] for s in sides], case.rule, case.layout, case.declared)


def records(cls, stream, b, x, length_b, length_x):
    """The journal line of a mutation class on stream `stream`, naming the
    replaced buffer `b` and the unrelated buffer `x`."""
    n = min(length_b, length_x)
    whole_b, whole_x, part_b, part_x = span(b, length_b), span(x, length_x), span(b, n), span(x, n)
    return {
        'zero': f'zero {stream} {whole_b}',
        'upload': f'upload {stream} {whole_b} {fnv(b"mutation")}',
        'copy-into': f'copy {stream} {part_x} {part_b}',
        'peer-copy-into': f'peer copy {stream} 0 {part_x} 1 {part_b}',
        'launch-argument': f'launch {stream} chop_mutation_write_v1 {CONFIG} [DeviceBuffer({whole_b})]',
        'executed-launch': f'launch {stream} chop_mutation_other_v1 {CONFIG} [DeviceBuffer({whole_x})]',
        'download': f'download {stream} {whole_b}',
        'readback': f'readback {stream} {whole_b} {length_b}',
        'copy-out': f'copy {stream} {part_b} {part_x}',
        'peer-copy-out': f'peer copy {stream} 0 {part_b} 1 {part_x}',
        'unrelated-zero': f'zero {stream} {whole_x}',
        'plan-record': f'launch r{PLAN_RANK} Mutation chop_mutation_plan_v1 {CONFIG} [DeviceBuffer({whole_b})]',
    }[cls]


def with_argument(line, b, length):
    """`line`, a launch, with one more argument: the whole buffer `b`."""
    item = parse_launch(line)
    item = dict(item, args=item['args'] + [f'DeviceBuffer({span(b, length)})'])
    return render(item)


def narrow(line_a, line_b, b, h):
    """Base launch `line_a` with the span of `b` it replaces with `h` (in its
    head launch `line_b`) halved, or None."""
    base, head = parse_launch(line_a), parse_launch(line_b)
    if base is None or head is None or len(base['args']) != len(head['args']):
        return None
    args = list(base['args'])
    for k, (x, y) in enumerate(zip(base['args'], head['args'])):
        if b in named(x) and h in named(y):
            m = next((m for m in SPAN_OF.finditer(x) if int(m[1]) == b and int(m[3]) > 1), None)
            if m is None:
                return None
            args[k] = x[:m.start()] + span(b, int(m[3]) // 2, int(m[2])) + x[m.end():]
            return render(dict(base, args=args))
    return None


class Mutation:
    def __init__(self, label, cls, op, context, base, head, expected=None):
        self.label, self.cls, self.op, self.context = label, cls, op, context
        self.base, self.head, self.expected = base, head, expected
        self.why = None


def expectation(facts, m):
    """The oracle's verdict on mutation `m` over the facts' notes."""
    if m.expected is not None:
        return m.expected, m.why
    notes_a, map_a = edit_notes(facts.notes[0], m.base)
    notes_b, map_b = edit_notes(facts.notes[1], m.head)
    pairs = {map_a[i]: map_b[j] for i, j in facts.pairs.items() if i in map_a and j in map_b}
    sites = [(h, b, map_a[i]) for h, b, i in facts.sites if i in map_a]
    return oracle(notes_a, notes_b, pairs, sites, facts.buffers)


def unrelated(facts):
    """A base allocation that no site replaces and no declaration resizes."""
    replaced = {b for _, b, _ in facts.sites}
    for raw in sorted(facts.sizes):
        if raw not in replaced and raw not in facts.resized and raw in facts.buffers:
            return raw
    raise Invalid('no unrelated base allocation')


def mutations(facts, per_resource=1, positions=('before', 'start', 'after')):
    """The mutations of one proven case: for `per_resource` replaced buffers of
    each added resource, every class at each position (`before` its first
    site, `start` of the history, `after` its last site), moves and deletes of
    existing records that name it, and a second argument naming it in a site
    launch; on a device that runs the instance create, the head-only changes
    of its declared zero and the controls."""
    notes_a, notes_b = facts.notes
    out = []
    by_resource = collections.defaultdict(list)
    for h, b, i in facts.sites:
        by_resource[facts.added[h]].append((h, b, i))
    x = unrelated(facts)
    prefix = facts.kind == 'direct-allocation-prefix'
    if prefix:
        fixed = max(facts.issued) + 101
    for resource in sorted(by_resource):
        chosen = []
        for h, b, _ in by_resource[resource]:
            if (h, b) not in chosen and len(chosen) < per_resource:
                chosen.append((h, b))
        for h, b in chosen:
            own = sorted(i for hh, bb, i in by_resource[resource] if (hh, bb) == (h, b))
            first, final = own[0], own[-1]
            executed = history.stream_of(notes_a[first])
            if prefix and executed is None:
                stream = stream_b = fixed
            else:
                stream = history.stream_of(notes_a[first])
                stream_b = facts.streams.get(stream, stream)
            length_b, length_x = facts.sizes[b], facts.sizes[x]
            h_b, h_x = facts.buffers[b], facts.buffers[x]
            start = facts.create[0]['end_note']
            places = {'before': first, 'start': start if start < first else None,
                      'after': final + 1 if final + 1 < len(notes_a) and final + 1 in facts.pairs else None}
            places = {k: v for k, v in places.items() if k in positions and v is not None}
            tag = f'{resource} {b}'

            def mirrored(cls, at, context_edits=((), ())):
                line_a = records(cls, stream, b, x, length_b, length_x)
                line_b = records(cls, stream_b, h_b, h_x, length_b, length_x)
                return ([*context_edits[0], ('insert', at, line_a)], [*context_edits[1], ('insert', facts.pairs[at], line_b)])

            if prefix:
                contexts = {'as recorded': ((), ())}
                if plan_device(notes_a) and plan_device(notes_b):
                    launched = (records(EXECUTED, stream, b, x, length_b, length_x), records(EXECUTED, stream_b, h_b, h_x, length_b, length_x))
                    contexts['executing'] = ((('insert', start, launched[0]),), (('insert', facts.pairs[start], launched[1]),))
                for context, extra in contexts.items():
                    classes = [*WRITES, EXECUTED, *CONTROLS] + ([PLAN_RECORD] if context == 'as recorded' else [])
                    if context == 'executing':
                        classes = [c for c in classes if c != EXECUTED]
                    for where, at in places.items():
                        for cls in classes:
                            base, head = mirrored(cls, at, extra)
                            out.append(Mutation(f'{tag}: {cls} {where}', cls, 'insert', context, base, head))
                    # A second argument naming the buffer in an earlier site
                    # launch, and in the site launch itself.
                    for label, at in (('an earlier site', own[0] if len(own) > 1 else None), ('the site', own[-1])):
                        if at is None:
                            continue
                        base = [*extra[0], ('replace', at, with_argument(notes_a[at], b, length_b))]
                        head = [*extra[1], ('replace', facts.pairs[at], with_argument(notes_b[facts.pairs[at]], h_b, length_b))]
                        out.append(Mutation(f'{tag}: {label} also names it in another argument', SITE_ARGUMENT, 'argument', context, base, head))
                    # Move the first record naming it after its last site to
                    # before its first; delete the last one before its first.
                    later = next((k for k in range(final + 1, len(notes_a)) if b in named(notes_a[k]) and k in facts.pairs), None)
                    if later is not None:
                        base = [*extra[0], ('delete', later), ('insert', first, notes_a[later])]
                        head = [*extra[1], ('delete', facts.pairs[later]), ('insert', facts.pairs[first], notes_b[facts.pairs[later]])]
                        out.append(Mutation(f'{tag}: record moved before its first site', 'move', 'move', context, base, head))
                    earlier = next((k for k in range(first - 1, -1, -1) if b in named(notes_a[k]) and k not in own and k in facts.pairs), None)
                    if earlier is not None:
                        base = [*extra[0], ('delete', earlier)]
                        head = [*extra[1], ('delete', facts.pairs[earlier])]
                        out.append(Mutation(f'{tag}: record before its first site deleted', 'delete', 'delete', context, base, head))
                    # The first site replaces another region of the buffer
                    # than the others: half of it, the head still whole.
                    narrowed = narrow(notes_a[first], notes_b[facts.pairs[first]], b, h)
                    if len(own) > 1 and narrowed is not None:
                        out.append(Mutation(f'{tag}: the first site replaces another region of it', REGION, 'replace', context,
                                            [*extra[0], ('replace', first, narrowed)], list(extra[1])))
                    # An unrelated graph captured and replayed at the start.
                    graph = fixed + 1
                    lines = [None, None]
                    for side, top, at, xx in ((0, facts.top[0], start, x), (1, facts.top[1], facts.pairs[start], h_x)):
                        gid = top + 1
                        body = [f'capture {graph}', records(EXECUTED, graph, b, xx, length_b, length_x), f'captured {graph} {gid} 1',
                                f'replay {graph} {gid}']
                        lines[side] = [*extra[side], ('id', at, gid), *(('insert', at, line) for line in body)]
                    out.append(Mutation(f'{tag}: a graph captured and replayed', GRAPH_REPLAY, 'insert', context, lines[0], lines[1]))
                    # Writes on another stream after an executed site, with
                    # and without the host waiting for the site's stream.
                    if executed is not None and 'after' in places:
                        at = places['after']
                        for cls in WRITES:
                            other_a, other_b = records(cls, fixed, b, x, length_b, length_x), records(cls, fixed, h_b, h_x, length_b, length_x)
                            out.append(Mutation(f'{tag}: {cls} on another stream after its last site', LATER_WRITE, 'insert', context,
                                                [*extra[0], ('insert', at, other_a)], [*extra[1], ('insert', facts.pairs[at], other_b)]))
                            waits = sorted({history.stream_of(notes_a[i]) for i in own})
                            out.append(Mutation(f'{tag}: {cls} on another stream after the host waits for its sites', LATER_WRITE, 'insert',
                                                context, [*extra[0], *(('insert', at, f'synchronize {w}') for w in waits), ('insert', at, other_a)],
                                                [*extra[1], *(('insert', facts.pairs[at], f'synchronize {facts.streams.get(w, w)}') for w in waits),
                                                 ('insert', facts.pairs[at], other_b)]))
                continue
            # A device that runs the instance create: the base-side controls,
            # the base-side writes (outside the rules) and the declared zero.
            for where, at in places.items():
                for cls in (*CONTROLS, *WRITES):
                    base, head = mirrored(cls, at)
                    expected = 'prove' if cls in CONTROLS else 'scope'
                    out.append(Mutation(f'{tag}: {cls} {where}', cls, 'insert', 'instance create', base, head, expected))
            z = facts.zeros.get(h)
            if z is None:
                continue
            zero = notes_b[z]
            use = next((j for j in range(z + 1, len(notes_b)) if h in named(notes_b[j])), None)
            m = Mutation(f'{resource} {h}: declared zero deleted', 'zero-deleted', 'delete', 'instance create', [], [('delete', z)], 'refuse')
            m.why = 'the added buffer is used unzeroed'
            out.append(m)
            m = None if use is None else Mutation(f'{resource} {h}: declared zero after its first use', 'zero-after-use', 'move', 'instance create', [],
                         [('delete', z), ('insert', use + 1, zero)], 'refuse')
            if m is not None:
                m.why = 'the first use reads the added buffer before its zero'
                out.append(m)
            other = next((s for s in facts.head_streams if s != facts.alloc_streams[h]), None)
            if other is not None:
                moved = re.sub(r'^zero \d+ ', f'zero {other} ', zero)
                m = Mutation(f'{resource} {h}: declared zero on another stream', 'zero-other-stream', 'replace', 'instance create', [],
                             [('replace', z, moved)], 'refuse')
                m.why = 'nothing orders the zero before uses on its allocation stream'
                out.append(m)
            alloc = facts.alloc_notes[h]
            if alloc + 1 < z:
                out.append(Mutation(f'{resource} {h}: declared zero right after its allocation', 'zero-earliest', 'move', 'instance create',
                                    [], [('delete', z), ('insert', alloc + 1, zero)], 'prove'))
    return out


def weaken_reads(predicate):
    real = layout.reads_only

    def weak(line, raw):
        return real(line, raw) or predicate(line)
    return [('reads_only', weak)]


def weak_plan_only(notes):
    """plan_only that takes an executed launch for a plan record."""
    return all((history.form(line) or ('',))[0] in ('allocate', 'launch') for line in notes)


def accept(*_):
    return None


WEAKENED = {
    'zero': lambda: weaken_reads(lambda line: line.startswith('zero ')),
    'upload': lambda: weaken_reads(lambda line: line.startswith('upload ')),
    'copy-into': lambda: weaken_reads(lambda line: line.startswith('copy ')),
    'peer-copy-into': lambda: weaken_reads(lambda line: line.startswith('peer copy ')),
    'launch-argument': lambda: weaken_reads(lambda line: line.startswith('launch ')),
    'executed-launch': lambda: [('plan_only', weak_plan_only)],
    # A site launch's other arguments are never checked, and every earlier
    # replacement site of the buffer, of any added buffer, leaves it
    # unwritten.
    SITE_ARGUMENT: lambda: [('check_site_alias', accept),
                            ('unwritten_site', lambda k, base, raw, sites, shared: any((base, k) in own for own in sites.values()))],
    REGION: lambda: [('check_regions', accept)],
    GRAPH_REPLAY: lambda: [('check_graphs', accept)],
    LATER_WRITE: lambda: [('check_later_writes', accept)],
}


@contextlib.contextmanager
def weakened(cls):
    """layout with the rule for mutation class `cls` deliberately weakened."""
    if cls is None:
        yield
        return
    patches = WEAKENED[cls]()
    real = [(name, getattr(layout, name)) for name, _ in patches]
    for name, weak in patches:
        setattr(layout, name, weak)
    try:
        yield
    finally:
        for name, fn in real:
            setattr(layout, name, fn)


# The cases and mutations a run proves, read by forked workers (run, jobs).
_WORK = []


def _verdict(index):
    case, facts, m = _WORK[index]
    expected, why = expectation(facts, m)
    proven, reason = prove(mutated(case, m.base, m.head))
    return expected, why, proven, reason


def _orphan_guard(parent):
    """A forked worker's initializer: the kernel sends it SIGTERM when its
    parent dies (Linux PR_SET_PDEATHSIG), and it exits now if the parent
    is already gone. A killed run then leaves no idle worker behind holding
    the parent's inherited descriptors, such as a launcher slot lock."""
    if sys.platform.startswith('linux'):
        import ctypes
        import signal
        ctypes.CDLL(None, use_errno=True).prctl(1, signal.SIGTERM, 0, 0, 0)  # PR_SET_PDEATHSIG
    if os.getppid() != parent:
        os._exit(1)


def workers(jobs):
    """A pool of `jobs` forked workers that exit with their parent."""
    import concurrent.futures
    import multiprocessing
    return concurrent.futures.ProcessPoolExecutor(jobs, mp_context=multiprocessing.get_context('fork'),
                                                  initializer=_orphan_guard, initargs=(os.getpid(),))


def run(cases, weaken=None, per_resource=1, positions=('before', 'start', 'after'), log=None, jobs=1):
    """Prove every case and each of its mutations, with `jobs` forked
    workers. Returns the report."""
    report = dict(cases=[], mutations=[])
    _WORK.clear()
    for case in cases:
        proven, why = prove(case)
        entry = dict(case=case.label, baseline='PROVEN' if proven else 'REFUSED', reason=why)
        report['cases'].append(entry)
        if not proven:
            continue
        facts = Facts(case)
        entry.update(kind=facts.kind, sites=len(facts.sites), plan_only=plan_device(facts.notes[0]) and plan_device(facts.notes[1]))
        _WORK.extend((case, facts, m) for m in mutations(facts, per_resource, positions))
    with weakened(weaken):
        if jobs > 1:
            with workers(jobs) as pool:
                verdicts = pool.map(_verdict, range(len(_WORK)), chunksize=4)
                verdicts = list(_logged(verdicts, log))
        else:
            verdicts = list(_logged(map(_verdict, range(len(_WORK))), log))
    for (case, _, m), (expected, why, proven, reason) in zip(_WORK, verdicts):
        outcome = ('escape' if expected == 'refuse' and proven else
                   'false refusal' if expected == 'prove' and not proven else 'ok')
        report['mutations'].append(dict(case=case.label, mutation=m.label, cls=m.cls, op=m.op, context=m.context,
                                        expected=expected, why=why, proven=proven, reason=(reason or '')[:300], outcome=outcome))
    _WORK.clear()
    report['summary'] = summary(report, weaken)
    return report


def _logged(verdicts, log):
    for index, verdict in enumerate(verdicts):
        if log:
            case, _, m = _WORK[index]
            expected, _, proven, _ = verdict
            outcome = ('escape' if expected == 'refuse' and proven else
                       'false refusal' if expected == 'prove' and not proven else 'ok')
            print(f'{outcome:13} {expected:6} {"PROVEN" if proven else "refused":7} {case.label[:60]} :: {m.context} :: {m.label}',
                  file=log, flush=True)
        yield verdict


def summary(report, weaken=None):
    rows = report['mutations']
    by = collections.Counter((r['cls'], r['expected'], r['outcome']) for r in rows)
    refused = {r['cls'] for r in rows if r['expected'] == 'refuse' and not r['proven']}
    escapes = [r for r in rows if r['outcome'] == 'escape']
    false = [r for r in rows if r['outcome'] == 'false refusal']
    return dict(weakened=weaken, cases=len(report['cases']), refused_baselines=[c['case'] for c in report['cases'] if c['baseline'] != 'PROVEN'],
                mutations=len(rows), escapes=len(escapes), false_refusals=len(false),
                escaped_classes=sorted({r['cls'] for r in escapes}), refused_unsafe_classes=sorted(refused),
                exercised_classes=sorted({r['cls'] for r in rows if r['expected'] == 'refuse'}),
                counts={f'{c} {e} {o}': n for (c, e, o), n in sorted(by.items())})


# Cases from captures.

def pair_cases(repo, model, base_rev, head_rev, spec, base_index, head_index, declaration_revision=None):
    """Every device fixture of a capture pair under the spec's declaration."""
    import pins
    import proof
    base, bp, bo = pins.snapshot(repo, base_rev, model, base_index)
    head, hp, ho = pins.snapshot(repo, head_rev, model, head_index)
    decl, _ = proof.read_layout(repo, spec, base, head, declaration_revision)
    need(decl is not None, 'the spec names no layout declaration')
    ba, ha = auxproof.load(base, base_index, bp, bo), auxproof.load(head, head_index, hp, ho)
    b, h = set(base['pins']), set(head['pins'])
    cases = []
    for fixture in spec['fixtures']:
        names = [n for n in fixture['journals'] if n in b & h]
        if not names or proof.kind_of(base, names) != {'device'}:
            continue
        declared = dict(base=sorted(set(fixture['journals']) & (b - h)), head=sorted(set(fixture['journals']) & (h - b)))
        cases.append(Case(fixture['name'], names, (bp, hp), (ba, ha), spec['auxiliary_rules'][fixture['name']], decl, declared))
    return cases


def plan_view(events, records, journal, test, host, max_launches=None, executed=False):
    """A plan-only device from an executing device's first `records` events,
    in the form lowering tests journal: every allocation (with its `allocate`
    note, rank and stream) and every other issued ID as a bare ID, then each
    executed launch as a plan record `launch rRANK Plan ...` of its stream's
    rank. With `executed`, the launches stay executed, each on a stream
    numbered past every issued ID (no stream is created, so the device is
    an executing one without an instance create). Returns (ID and allocation events, plan note events, allocation
    count), or None for a device without launches."""
    prefix = events[:records]
    ranks = {e[1]: e[2] for e in prefix if e[0] == 'stream'}
    allocations, note = {}, None
    for e in prefix:
        if e[0] == 'note':
            note = e[1]
        elif e[0] == 'allocation':
            allocations[e[1] & 0xffffffff] = (note, list(e))
    plans = []
    top = max(e[1] for e in prefix if e[0] == 'id')
    for e in prefix:
        if e[0] != 'note':
            continue
        found = history.form(e[1])
        if found and found[0] == 'launch' and not PLAN.fullmatch(e[1]):
            stream = int(found[1][1])
            label = f'{top + 1000 + stream}' if executed else f'r{ranks.get(stream) or 0} Plan'
            plans.append(f'launch {label} {e[1].split(" ", 2)[2]}')
    if max_launches:
        plans = plans[:max_launches]
    if not plans or not allocations:
        return None
    out = []
    for k in range(1, top + 1):
        out.append(['id', k])
        if k in allocations:
            out += [['note', allocations[k][0]], allocations[k][1]]
    return out, [['note', line] for line in plans], len(allocations)


def collective_candidates(allocs, plans):
    """(symbol, argument index, ranks, (bytes, alignment), exclusive) of every
    launch symbol whose launches all hold, at that argument, a peer array over
    distinct ranks' allocations of one size, each always the same span, that
    no other argument of those launches names. `exclusive` when no other
    launch names those allocations."""
    rows = {row[1]: row for row in allocs.values()}
    by_symbol = collections.defaultdict(list)
    items = [parse_launch(line) for line in plans]
    for k, item in enumerate(items):
        by_symbol[item['symbol']].append(k)
    out = []
    for symbol, launches in sorted(by_symbol.items()):
        width = len(items[launches[0]]['args'])
        for a in range(width):
            entries, spans, ok = set(), collections.defaultdict(set), True
            for k in launches:
                args = items[k]['args']
                m = PEERS.fullmatch(args[a]) if a < len(args) else None
                raws = [int(r) for r in HANDLES.findall(m[1])] if m else []
                keys = [layout.rank_key(rows[r]) if r in rows else None for r in raws]
                if (not m or 'None' in m[1] or len(raws) < 2 or None in keys or len(set(keys)) != len(keys)
                        or m[1].count('Some(') != len(raws)):
                    ok = False
                    break
                entries |= set(raws)
                for m in SPAN_OF.finditer(args[a]):
                    spans[int(m[1])].add((int(m[2]), int(m[3])))
            # One region per buffer (an added buffer stands for one region),
            # and no other argument of the collective names its buffers.
            if (not ok or any(len(v) != 1 for v in spans.values())
                    or any(named(arg) & entries for k in launches for i, arg in enumerate(items[k]['args']) if i != a)):
                continue
            shapes = {(rows[r][3], rows[r][4]) for r in entries}
            counts = {len(PEERS.fullmatch(items[k]['args'][a])[1].split('Some(')) - 1 for k in launches}
            if len(shapes) != 1 or len(counts) != 1:
                continue
            exclusive = all(not (named(line) & entries) or items[k]['symbol'] == symbol for k, line in enumerate(plans))
            out.append((symbol, a, counts.pop(), shapes.pop(), exclusive))
    return out


def overlay(events, records, journal, test, host, label, choice='exclusive', max_launches=None, changes=None, executed=False):
    """A declared collective change synthesized on a plan view of a captured
    device: for each (symbol, argument, resource) of `changes`, a zeroed added
    buffer per rank (resource `resource`) replaces the base buffers of that
    collective's peer array. Without `changes`, one collective is chosen:
    `exclusive` picks buffers nothing else names, `shared` buffers that other
    records or arguments name too. Returns a Case or None."""
    view = plan_view(events, records, journal, test, host, max_launches, executed)
    if view is None:
        return None
    base_ids, plan_notes, count = view
    allocs = {e[1] & 0xffffffff: e for e in base_ids if e[0] == 'allocation'}
    plans = [e[1] for e in plan_notes]
    candidates = {(c[0], c[1]): c for c in collective_candidates(allocs, plans)}
    if changes is None:
        picks = [c for c in candidates.values() if c[4] == (choice == 'exclusive')]
        if not picks:
            return None
        changes = [(picks[0][0], picks[0][1], 'mutation.inbox')]
    need(all((symbol, argument) in candidates for symbol, argument, _ in changes), 'an overlay change is not a peer-array collective')
    rows = {row[1]: row for row in allocs.values()}
    head_ids = list(base_ids)
    slot = max(e[1] for e in base_ids if e[0] == 'id')
    added, witness, resources = {}, [], {}
    for symbol, argument, resource in changes:
        if resource in resources:
            continue
        size, alignment = candidates[(symbol, argument)][3]
        resources[resource] = dict(name=resource, bytes=size, alignment=alignment, per_rank=1, lifetime='instance', init='zero',
                                   alongside=dict(bytes=size, alignment=alignment))
        anchors = {}
        for row in allocs.values():
            if (row[3], row[4]) == (size, alignment):
                anchors.setdefault(layout.rank_key(row), row)
        for key in sorted(anchors, key=str):
            slot += 1
            row = anchors[key]
            head_ids += [['id', slot], ['note', f'allocate {size} {alignment}'], ['allocation', GEN | slot, row[2], size, alignment, row[5]]]
            added[(resource, key)] = GEN | slot
            witness.append(dict(resource=resource, head_raw=GEN | slot, rank=row[2]))
    head_plans = []
    for line in plans:
        item = parse_launch(line)
        args = list(item['args'])
        for symbol, argument, resource in changes:
            if item['symbol'] == symbol:
                raws = [int(r) for r in HANDLES.findall(PEERS.fullmatch(args[argument])[1])]
                size = resources[resource]['bytes']
                args[argument] = 'DevicePointerArray([' + ', '.join(
                    f'Some({span(added[(resource, layout.rank_key(rows[r]))], size)})' for r in raws) + '])'
        head_plans.append(render(dict(item, args=args)) if args != item['args'] else line)
    what = '; '.join(f'{symbol} argument {argument} by {resource}' for symbol, argument, resource in changes)
    decl = {'schema': layout.SCHEMA, 'description': f'Mutation harness overlay on {label}: {what}.',
            'resources': {'added': list(resources.values())},
            'collectives': [dict(name=f'mutation collective {k}', from_symbol=symbol, to_symbol=symbol,
                                 ranks=candidates[(symbol, argument)][2], peers={str(argument): resource})
                            for k, (symbol, argument, resource) in enumerate(changes)]}
    parsed = layout.parse(json.dumps(decl).encode())
    sides = []
    for ids, lines, allocated in ((base_ids, plans, count), (head_ids, head_plans, count + len(added))):
        events_ = ids + [['note', line] for line in lines]
        notes = [e[1] for e in events_ if e[0] == 'note']
        create = dict(kind='direct-allocation-prefix', start_event=0, end_event=len(ids), start_note=0, end_note=allocated,
                      path='', sha256='')
        binding = dict(host=host, test=test, records=len(events_), positions=list(range(allocated, len(notes))), create=create,
                       create_lines=notes[:allocated])
        aux = dict(manifest=dict(bindings={journal: binding}, unrecoverable={}, hosts={}, text={}),
                   hosts={host: dict(events=events_, notes=notes, test=test)})
        sides.append((aux, {journal: notes[allocated:]}))
    rule = dict(added_resolves=[], added_modules=[], create_policy={}, added_allocations=witness)
    return Case(f'{label} [overlay: {what}]', [journal], [s[1] for s in sides], [s[0] for s in sides],
                rule, (parsed, {}))


def capture_overlays(repo, model, revision, index, max_launches=None, limit=None):
    """Exclusive and shared overlays on every device fixture of one capture."""
    import pins
    snap, bodies, origins = pins.snapshot(repo, revision, model, index)
    aux = auxproof.load(snap, index, bodies, origins)
    hosts = collections.defaultdict(list)
    for name, b in aux['manifest']['bindings'].items():
        hosts[b['host']].append((name, b))
    cases = []
    for host, items in sorted(hosts.items()):
        records = max(b['records'] for _, b in items)
        journal = 'plan-view-' + sorted(n for n, _ in items)[0]
        for choice, executed in (('exclusive', False), ('shared', False), ('exclusive', True)):
            kind = 'executed' if executed else 'plan'
            case = overlay(aux['hosts'][host]['events'], records, journal, items[0][1]['test'], f'{kind}-view/' + host,
                           f'{host} ({choice}, {kind} view)', choice, max_launches, executed=executed)
            if case is not None:
                cases.append(case)
        if limit and len(cases) >= limit:
            break
    return cases[:limit] if limit else cases


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    sub = parser.add_subparsers(dest='mode', required=True)
    pair = sub.add_parser('pair', help='a capture pair under its committed layout declaration')
    pair.add_argument('--base', required=True)
    pair.add_argument('--head', required=True)
    pair.add_argument('--spec', required=True, type=Path)
    pair.add_argument('--base-capture-index', required=True, type=Path)
    pair.add_argument('--head-capture-index', required=True, type=Path)
    pair.add_argument('--declaration-revision')
    view = sub.add_parser('overlay', help='synthesized collective changes on plan views of one capture')
    view.add_argument('--revision', required=True)
    view.add_argument('--capture-index', required=True, type=Path)
    view.add_argument('--max-launches', type=int)
    for p in (pair, view):
        p.add_argument('--repo', required=True)
        p.add_argument('--model', required=True)
        p.add_argument('--report', required=True, type=Path)
        p.add_argument('--weaken', choices=sorted(WEAKENED))
        p.add_argument('--fixtures', type=int, help='at most this many cases')
        p.add_argument('--match', action='append', default=[], help='only cases whose label contains this text (repeatable)')
        p.add_argument('--per-resource', type=int, default=1)
        p.add_argument('--positions', default='before,start,after')
        p.add_argument('--jobs', type=int, default=1, help='forked workers proving mutations')
    args = parser.parse_args(argv)
    try:
        if args.mode == 'pair':
            spec = json.loads(args.spec.read_text())
            cases = pair_cases(args.repo, args.model, args.base, args.head, spec, args.base_capture_index, args.head_capture_index,
                               args.declaration_revision)
            cases = [c for c in cases if c.layout is not None][:args.fixtures]
        else:
            cases = capture_overlays(args.repo, args.model, args.revision, args.capture_index, args.max_launches, args.fixtures)
    except Invalid as error:
        print(f'invalid input: {error}', file=sys.stderr)
        return 2
    if args.match:
        cases = [c for c in cases if any(t in c.label for t in args.match)]
    if not cases:
        print('no case to mutate: the capture holds no matching device fixture', file=sys.stderr)
        return 1
    report = run(cases, args.weaken, args.per_resource, tuple(args.positions.split(',')), log=sys.stdout, jobs=args.jobs)
    args.report.write_text(json.dumps(report, indent=1) + '\n')
    s = report['summary']
    print(json.dumps({k: v for k, v in s.items() if k != 'counts'}, indent=1))
    return 0 if passed(s) else 1


def passed(s):
    """A run passes with every baseline proven, at least one mutation and no
    false refusal; without a weakened rule, with no escape; with one, when it
    exercised its class (a mutation of it the oracle refuses) and that class
    escaped."""
    if s['refused_baselines'] or not s['mutations'] or s['false_refusals']:
        return False
    if s['weakened'] is None:
        return not s['escapes']
    return s['weakened'] in s['exercised_classes'] and s['weakened'] in s['escaped_classes']


if __name__ == '__main__':
    sys.exit(main())
