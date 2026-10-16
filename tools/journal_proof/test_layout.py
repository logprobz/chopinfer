#!/usr/bin/env python3
"""Declared layout changes (layout.py, specs/layout_spec.py): added
allocations, new collectives and grid changes of named kernel families, with
the module, resize, stride and host-span changes they come with.

The toy change splits a two-rank MoE: the head drops the `moe_ep` cubin for
`moe_split` and `comm_f32`, grows each rank's route table from 32 to 64
bytes, adds a zeroed 256-byte F32 inbox per rank, regrids the MoE launch and
turns the all-reduce after it (not the one after attention) into an F32
all-reduce over the new inboxes. Each test proves base against head through
proof.py; red tests change one thing the declaration does not cover, or drop
one declared item, and must not be PROVEN."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent / 'specs'))
import layout_spec  # noqa: E402
from common import Invalid, fnv  # noqa: E402
import layout  # noqa: E402
from engine import Unproven  # noqa: E402
from proof import main  # noqa: E402
from provenance import init  # noqa: E402
from testkit import CONFIG, CREATE, GEN, MODEL, ROOT, STEP, GitFixture, checkpoint, seal_capture, span, write_device  # noqa: E402

LOCK = ROOT + 'kernels.lock.d/toy.lock'
DECL = 'models/toy/journal-changes/split.json'
TEXT = 'toy-text.txt'
HOST = GEN + 1_000_000
PAYLOADS = {'core': b'core cubin', 'moe_ep': b'moe ep cubin', 'moe_split': b'moe split cubin', 'comm_f32': b'comm f32 cubin'}
SYMBOLS = {'core': ['chop_allreduce_bf16_v1', 'chop_attn_v1', 'chop_copy_v1'], 'moe_ep': ['chop_moe_ep_v1'],
           'moe_split': ['chop_moe_split_v1'], 'comm_f32': ['chop_allreduce_f32_v1']}
EP, SPLIT, BF16, F32, ATTN = 'chop_moe_ep_v1', 'chop_moe_split_v1', 'chop_allreduce_bf16_v1', 'chop_allreduce_f32_v1', 'chop_attn_v1'


def bare(raw, length, offset=0):
    return span(raw, offset, length)[len('DeviceBuffer('):-1]


def peers(*buffers):
    return 'DevicePointerArray([' + ', '.join(f'Some({bare(raw, n)})' for raw, n in buffers) + '])'


def launch(stream, symbol, grid, args):
    return f'launch {stream} {symbol} {CONFIG.replace("{grid}", str(grid))} [' + ', '.join(args) + ']'


def declaration():
    return {
        'schema': layout.SCHEMA, 'description': 'Toy MoE split over two ranks with an F32 MoE all-reduce.',
        'modules': {'removed': ['moe_ep'], 'added': [dict(name='moe_split', loads_like='moe_ep'), dict(name='comm_f32', loads_like='core')]},
        'resources': {
            'added': [dict(name='comm.f32_inbox', bytes=256, alignment=256, per_rank=1, lifetime='instance', init='zero',
                           alongside=dict(bytes=128, alignment=256))],
            'resized': [{'name': 'moe.route', 'from': 32, 'to': 64, 'alignment': 256, 'per_rank': 1}]},
        'collectives': [dict(name='MoE reduce, F32', from_symbol=BF16, to_symbol=F32, ranks=2,
                             when=dict(after=dict(symbols=[EP])), peers={'1': 'comm.f32_inbox'})],
        'kernels': [dict(name='MoE, split', from_symbol=EP, to_symbol=SPLIT, args={'2': {'ScalarU32(1)': 'ScalarU32(0)'}},
                         cases=[dict(when=dict(grid_x=8), then=dict(grid_x=2))])]}


def base_world():
    create = [('stream', 1, 0), ('stream', 2, 1), ('module', 3, 'core'), ('resolve', 4, 3, BF16), ('configure', 4, 64),
              ('resolve', 5, 3, ATTN), ('module', 6, 'moe_ep'), ('resolve', 7, 6, EP)]
    for rank, (data, inbox, route) in enumerate(((8, 9, 10), (11, 12, 13))):
        create += [('alloc', data, rank, 64, rank + 1), ('alloc', inbox, rank, 128, rank + 1), ('alloc', route, rank, 32, rank + 1)]
    create += [('zero', 1, 9, 128), ('zero', 2, 12, 128), 'synchronize 1', 'synchronize 2']
    step = []
    for rank, (data, route) in enumerate(((8, 10), (11, 13))):
        stream = rank + 1
        step += [launch(stream, EP, 8, [span(GEN + data), span(GEN + route, 0, 32), f'ScalarU32({rank})']),
                 launch(stream, BF16, 4, [span(GEN + data), peers((GEN + 9, 128), (GEN + 12, 128)), 'ScalarU32(64)']),
                 launch(stream, ATTN, 4, [span(GEN + data)]),
                 launch(stream, BF16, 4, [span(GEN + data), peers((GEN + 9, 128), (GEN + 12, 128)), 'ScalarU32(64)'])]
    return dict(create=create, step=step)


def head_world():
    create = [('stream', 1, 0), ('stream', 2, 1), ('module', 3, 'core'), ('resolve', 4, 3, BF16), ('configure', 4, 64),
              ('resolve', 5, 3, ATTN), ('module', 6, 'moe_split'), ('resolve', 7, 6, SPLIT), ('module', 8, 'comm_f32'),
              ('resolve', 9, 8, F32)]
    for rank, (data, inbox, route, f32) in enumerate(((10, 11, 12, 13), (14, 15, 16, 17))):
        create += [('alloc', data, rank, 64, rank + 1), ('alloc', inbox, rank, 128, rank + 1), ('alloc', route, rank, 64, rank + 1),
                   ('alloc', f32, rank, 256, rank + 1)]
    create += [('zero', 1, 11, 128), ('zero', 2, 15, 128), ('zero', 1, 13, 256), ('zero', 2, 17, 256), 'synchronize 1', 'synchronize 2']
    step = []
    for rank, (data, route) in enumerate(((10, 12), (14, 16))):
        stream = rank + 1
        step += [launch(stream, SPLIT, 2, [span(GEN + data), span(GEN + route, 0, 64), 'ScalarU32(0)']),
                 launch(stream, F32, 4, [span(GEN + data), peers((GEN + 13, 256), (GEN + 17, 256)), 'ScalarU32(64)']),
                 launch(stream, ATTN, 4, [span(GEN + data)]),
                 launch(stream, BF16, 4, [span(GEN + data), peers((GEN + 11, 128), (GEN + 15, 128)), 'ScalarU32(64)'])]
    return dict(create=create, step=step)


def events_of(items):
    """FakeDevice's records for `items`, in its order: a stream, resolve and
    allocation issue their ID before their note; a module issues its ID and
    payload without a note."""
    events, notes = [], []

    def note(text):
        events.append(['note', text])
        notes.append(text)
    for item in items:
        if isinstance(item, str):
            note(item)
            continue
        kind = item[0]
        if kind == 'id':
            events.append(['id', item[1]])
        elif kind == 'event':
            # A raw trace event, for a record whose note and event disagree.
            events.append(list(item[1]))
        elif kind == 'stream':
            _, ident, rank = item
            events.append(['id', ident])
            note(f'stream {ident}')
            events.append(['stream', ident, rank])
        elif kind == 'module':
            _, ident, name = item
            payload = PAYLOADS[name]
            events += [['id', ident], ['module', ident, name, fnv(payload), len(payload), f'{name}.bin']]
        elif kind == 'resolve':
            _, ident, module, symbol = item
            events.append(['id', ident])
            note(f'resolve {ident} {symbol}')
            events.append(['resolve', ident, module, symbol])
        elif kind == 'configure':
            note(f'configure {item[1]} {item[2]}')
        elif kind == 'alloc':
            _, ident, rank, size, stream = item
            events.append(['id', ident])
            note(f'allocate {size} 256')
            events.append(['allocation', GEN + ident, rank, size, 256, GEN + stream])
        elif kind == 'zero':
            _, stream, ident, size = item
            note(f'zero {stream} {bare(GEN + ident, size)}')
    return events, notes


def lock(names, shas=None):
    shas = shas or {}
    return json.dumps([dict(name=n, cubin_sha256=shas.get(n, hashlib.sha256(PAYLOADS[n]).hexdigest()),
                            entries=[dict(symbol=s) for s in SYMBOLS[n]]) for n in names]).encode()


class LayoutTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='jproof-layout-test-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        p = patch('provenance.AUTHORITY', self.root / 'authority')
        p.start()
        self.addCleanup(p.stop)
        init()
        self.git = GitFixture(self.root / 'objects.git')
        self.base, self.head, self.decl = base_world(), head_world(), declaration()
        self.base_locks, self.head_locks = ['core', 'moe_ep'], ['core', 'moe_split', 'comm_f32']
        self.head_shas = {}
        self.text = None
        self.labels = 0

    def capture(self, world, locks, shas=None, declaration=None, text=None):
        self.labels += 1
        root = self.root / f'capture-{self.labels}'
        created, created_notes = events_of(world['create'])
        between, _ = events_of(world.get('history', []))
        stepped, step_notes = events_of(world['step'])
        # With `create_unpinned` no journal pins the create range, as for a
        # device whose tests check only its steps. With `no_instance_create`
        # the device also has no create marker: its create is its prefix of
        # allocations before the first launch (`direct-allocation-prefix`).
        prefix = world.get('no_instance_create', False)
        unpinned = prefix or world.get('create_unpinned', False)
        marker = [] if prefix else [['created', 0, 0, len(created), len(created_notes)]]
        events = created + marker + between + stepped
        bodies = {STEP: step_notes} if unpinned else {CREATE: created_notes, STEP: step_notes}
        if text is not None:
            bodies[TEXT] = text
        files = {LOCK: lock(locks, shas)}
        if declaration is not None:
            files[DECL] = declaration
        revision = self.git.commit(bodies, extra_files=files)
        write_device(root, 1, events)
        for name, payload in PAYLOADS.items():
            (root / 'aux' / f'{name}.bin').write_bytes(payload)
        if not unpinned:
            checkpoint(root, CREATE, [(1, events, len(created) + 1)])
        checkpoint(root, STEP, [(1, events, len(events))])
        if text is not None:
            checkpoint(root, TEXT, [], test='texts')
        return revision, seal_capture(root, revision, bodies)

    def data(self, decl=None):
        return (json.dumps(decl or self.decl, indent=1) + '\n').encode()

    def captures(self, decl_bytes=None, in_head=True):
        text = self.text
        b = self.capture(self.base, self.base_locks, text=text[0] if text else None)
        h = self.capture(self.head, self.head_locks, self.head_shas, (decl_bytes or self.data()) if in_head else None,
                         text=text[1] if text else None)
        return b, h

    def generate(self, b, h, revision=None):
        spec, _ = layout_spec.generate(self.git.repo, MODEL, b[0], h[0], b[1], h[1], DECL, revision)
        return spec

    def green_spec(self):
        """The witness spec of the declared change, written by the generator
        from the unchanged worlds and declaration."""
        saved = (self.base, self.head, self.decl, self.head_locks, self.head_shas, self.text)
        self.base, self.head, self.decl, self.head_locks, self.head_shas = (
            base_world(), head_world(), declaration(), ['core', 'moe_split', 'comm_f32'], {})
        try:
            b, h = self.captures()
            return self.generate(b, h)
        finally:
            self.base, self.head, self.decl, self.head_locks, self.head_shas, self.text = saved

    def prove(self, spec, b, h, *extra):
        path = self.root / f'spec-{self.labels}.json'
        path.write_text(json.dumps(spec))
        report = self.root / f'report-{self.labels}.json'
        with contextlib.redirect_stdout(io.StringIO()):
            code = main(['--repo', str(self.git.repo), '--model', MODEL, '--base', b[0], '--head', h[0], '--spec', str(path),
                         '--base-capture-index', str(b[1]), '--head-capture-index', str(h[1]), '--report', str(report), *extra])
        return code, json.loads(report.read_text())

    def run_case(self, expected, witness='generated', spec_edit=None):
        """Prove the test's base and head under its declaration. The witness
        is the generator's for this test's captures, or (`green`) the
        declared change's own witness, as a spec author would hold it."""
        data = self.data()
        b, h = self.captures(data)
        spec = self.generate(b, h) if witness == 'generated' else self.green_spec()
        spec['layout']['sha256'] = hashlib.sha256(data).hexdigest()
        if spec_edit:
            spec_edit(spec)
        code, report = self.prove(spec, b, h)
        self.assertEqual(code, expected, json.dumps({k: report.get(k) for k in ('status', 'problems', 'journals')}, indent=1)[:4000])
        return report

    def drop_witness(self, key, test):
        def edit(spec):
            for rule in spec['auxiliary_rules'].values():
                if key in rule:
                    rule[key] = [e for e in rule[key] if not test(e)]
        return edit

    # The declared change.

    def test_declared_split_is_proven(self):
        report = self.run_case(0)
        self.assertEqual({v['verdict'] for v in report['journals'].values()}, {'PROVEN'})
        self.assertTrue(report['coverage']['full_history_proof'])
        self.assertTrue(report['layout_declaration']['read_at_head'])
        create = report['auxiliary_create_counts']
        self.assertEqual((create['added_allocations'], create['added_zeros'], create['resized_allocations']), (2, 2, 2))
        step = report['counts']
        self.assertEqual((step['collective_launches'], step['family_launches'], step['grid_changes'], step['resized_spans']), (2, 2, 2, 2))

    def test_generator_writes_the_witness(self):
        b, h = self.captures()
        rule = next(iter(self.generate(b, h)['auxiliary_rules'].values()))
        self.assertEqual(rule['added_allocations'], [dict(resource='comm.f32_inbox', head_raw=GEN + 13, rank=0),
                                                     dict(resource='comm.f32_inbox', head_raw=GEN + 17, rank=1)])
        self.assertEqual(rule['resized_allocations'], [dict(resource='moe.route', base_raw=GEN + 10, head_raw=GEN + 12, rank=0),
                                                       dict(resource='moe.route', base_raw=GEN + 13, head_raw=GEN + 16, rank=1)])
        self.assertEqual([d['name'] for d in rule['removed_modules']], ['moe_ep'])
        self.assertEqual(sorted(d['name'] for d in rule['added_modules']), ['comm_f32', 'moe_split'])

    def test_generator_refuses_an_undeclared_allocation(self):
        self.decl['resources']['added'] = []
        self.decl['collectives'] = []
        b, h = self.captures()
        with self.assertRaisesRegex(Invalid, 'no alignment pairs every kept ID'):
            self.generate(b, h)

    # Added allocations.

    def test_undeclared_added_allocation_fails(self):
        self.decl['resources']['added'] = []
        self.decl['collectives'] = []
        report = self.run_case(1, 'green', self.drop_witness('added_allocations', lambda e: True))
        self.assertIn('issued ID count differs', report['journals'][STEP]['reasons'][0])

    def test_witness_naming_an_undeclared_resource_is_invalid(self):
        self.decl['resources']['added'] = []
        self.decl['collectives'] = []
        self.run_case(2, 'green')

    def test_added_allocation_of_another_size_fails(self):
        self.decl['resources']['added'][0]['bytes'] = 512
        report = self.run_case(1, 'green')
        self.assertIn('is not a comm.f32_inbox (512 bytes', report['journals'][STEP]['reasons'][0])

    def test_extra_added_allocation_on_a_rank_fails(self):
        self.head['create'].insert(-6, ('alloc', 18, 0, 256, 1))
        self.run_case(1, 'green')

    def test_extra_allocation_beyond_the_declared_count_fails_in_the_generator_witness(self):
        self.head['create'].insert(-6, ('alloc', 18, 0, 256, 1))
        report = self.run_case(1)
        self.assertIn('rank 0 holds 2 comm.f32_inbox, the declaration 1', report['journals'][STEP]['reasons'][0])

    def test_declared_buffer_missing_on_one_rank_fails(self):
        # A second added resource no launch names: rank 1 does not allocate it.
        self.decl['resources']['added'].append(dict(name='comm.f32_flags', bytes=64, alignment=256, per_rank=1,
                                                    lifetime='instance', init='none', alongside=dict(bytes=128, alignment=256)))
        self.head['create'].insert(-6, ('alloc', 18, 0, 64, 1))
        report = self.run_case(1)
        self.assertIn('rank 1 holds 0 comm.f32_flags, the declaration 1', report['journals'][STEP]['reasons'][0])
        self.head['create'].insert(-6, ('alloc', 19, 1, 64, 2))
        self.run_case(0)

    def test_added_allocation_outside_create_fails(self):
        # Rank 1's inbox is allocated after create, in the history.
        item = next(i for i in self.head['create'] if i[:2] == ('alloc', 17))
        self.head['create'].remove(item)
        self.head['create'].remove(('zero', 2, 17, 256))
        self.head['history'] = [item]
        self.decl['resources']['added'][0]['init'] = 'none'
        self.head['create'].remove(('zero', 1, 13, 256))
        report = self.run_case(1)
        self.assertIn('not allocated inside the create range', report['journals'][STEP]['reasons'][0])

    def test_missing_declared_zero_fails(self):
        self.head['create'].remove(('zero', 2, 17, 256))
        report = self.run_case(1)
        self.assertIn('0 whole zeros in the create range (instance-create), the declaration 1', report['journals'][STEP]['reasons'][0])

    def test_repeated_zero_fails(self):
        self.head['create'].append(('zero', 2, 17, 256))
        self.run_case(1)

    def test_zero_of_a_buffer_declared_uninitialized_fails(self):
        self.decl['resources']['added'][0]['init'] = 'none'
        report = self.run_case(1)
        self.assertIn('own fixture create differs', report['journals'][STEP]['reasons'][0])

    def test_added_buffer_in_an_undeclared_argument_fails(self):
        self.head['step'][2] = launch(1, ATTN, 4, [span(GEN + 13, 0, 256)])
        self.run_case(1)

    # New collectives.

    def test_undeclared_collective_fails(self):
        self.decl['collectives'] = []
        report = self.run_case(1)
        self.assertIn('unexplained line differences', report['journals'][STEP]['reasons'][0])

    def test_collective_with_the_wrong_peer_argument_fails(self):
        self.decl['collectives'][0]['peers'] = {'0': 'comm.f32_inbox'}
        self.run_case(1)

    def test_collective_rank_count_must_match(self):
        self.decl['collectives'][0]['ranks'] = 4
        report = self.run_case(1)
        self.assertIn('holds 2 peers, the declaration 4 ranks', report['journals'][STEP]['reasons'][0])

    def test_collective_converted_outside_its_selection_fails(self):
        # The all-reduce after attention becomes F32 too: the declaration
        # selects only the one after the MoE launch, evaluated on the base.
        self.head['step'][3] = launch(1, F32, 4, [span(GEN + 10), peers((GEN + 13, 256), (GEN + 17, 256)), 'ScalarU32(64)'])
        self.run_case(1)

    def test_collective_peer_order_is_kept(self):
        self.head['step'][1] = launch(1, F32, 4, [span(GEN + 10), peers((GEN + 17, 256), (GEN + 13, 256)), 'ScalarU32(64)'])
        self.run_case(1)

    def test_collective_over_an_old_buffer_fails(self):
        self.head['step'][1] = launch(1, F32, 4, [span(GEN + 10), peers((GEN + 11, 128), (GEN + 15, 128)), 'ScalarU32(64)'])
        self.run_case(1)

    def test_collective_names_only_added_resources(self):
        self.decl['collectives'][0]['peers'] = {'1': 'moe.route'}
        self.run_case(2, 'green')

    # Grid changes of a named kernel family.

    def test_undeclared_grid_change_fails(self):
        del self.decl['kernels'][0]['cases']
        report = self.run_case(1)
        self.assertIn('unexplained line differences', report['journals'][STEP]['reasons'][0])

    def test_grid_change_to_another_grid_fails(self):
        self.decl['kernels'][0]['cases'][0]['then']['grid_x'] = 4
        self.run_case(1)

    def test_grid_change_of_an_undeclared_family_fails(self):
        self.head['step'][2] = launch(1, ATTN, 2, [span(GEN + 10)])
        self.run_case(1)

    def test_ambiguous_cases_fail(self):
        self.decl['kernels'][0]['cases'].append(dict(when=dict(block_x=128), then=dict(grid_x=2)))
        report = self.run_case(1)
        self.assertIn('2 declared cases match one launch', report['journals'][STEP]['reasons'][0])

    def test_case_keyed_by_the_previous_launch(self):
        # One base grid becomes two head grids, told apart by the previous
        # launch on the stream: after attention 2, after an all-reduce 1.
        self.base['step'][:0] = [launch(1, ATTN, 4, [span(GEN + 8)])]
        self.head['step'][:0] = [launch(1, ATTN, 4, [span(GEN + 10)])]
        self.base['step'].insert(5, launch(1, EP, 8, [span(GEN + 8), span(GEN + 10, 0, 32), 'ScalarU32(0)']))
        self.head['step'].insert(5, launch(1, SPLIT, 1, [span(GEN + 10), span(GEN + 12, 0, 64), 'ScalarU32(0)']))
        self.decl['kernels'][0]['cases'] = [dict(when=dict(grid_x=8, after=dict(symbols=[ATTN])), then=dict(grid_x=2)),
                                            dict(when=dict(grid_x=8, after=dict(symbols=[BF16])), then=dict(grid_x=1)),
                                            dict(when=dict(grid_x=8), then=dict(grid_x=2))]
        report = self.run_case(1)
        self.assertIn('2 declared cases match one launch', json.dumps(report['journals']))
        self.decl['kernels'][0]['cases'].pop()
        # Rank 1's MoE launch has no predecessor: it matches no case and
        # keeps its grid, so the head must too.
        self.head['step'][6] = self.head['step'][6].replace('grid_x: 2,', 'grid_x: 8,')
        self.run_case(0)
        self.head['step'][5] = self.head['step'][5].replace('grid_x: 1,', 'grid_x: 2,')
        self.run_case(1)

    def test_unrenamed_family_symbol_fails(self):
        self.decl['kernels'][0]['to_symbol'] = EP
        self.run_case(1)

    def test_family_literal_map_is_needed(self):
        del self.decl['kernels'][0]['args']
        self.run_case(1)

    # Modules.

    def test_removed_module_still_locked_in_the_head_is_invalid(self):
        self.head_locks = ['core', 'moe_ep', 'moe_split', 'comm_f32']
        report = self.run_case(2, 'green')
        self.assertIn('moe_ep: a removed module must be locked in the base and not in the head', report['problems'][0])

    def test_added_module_payload_must_be_locked(self):
        self.head_shas = {'comm_f32': '0' * 64}
        report = self.run_case(1, 'green')
        self.assertIn('captured payload is not the locked SHA-256', report['journals'][STEP]['reasons'][0])

    def test_undeclared_module_removal_fails(self):
        self.decl['modules']['removed'] = []
        self.run_case(1, 'green', lambda spec: [rule.pop(k, None) for rule in spec['auxiliary_rules'].values()
                                                for k in ('removed_modules', 'removed_resolves')])

    # Resized allocations.

    def test_undeclared_resize_fails(self):
        self.decl['resources']['resized'] = []
        self.run_case(1, 'green', self.drop_witness('resized_allocations', lambda e: True))

    def test_partial_span_of_a_resize_without_layout_fails(self):
        self.base['step'][0] = launch(1, EP, 8, [span(GEN + 8), span(GEN + 10, 8, 16), 'ScalarU32(0)'])
        self.head['step'][0] = launch(1, SPLIT, 2, [span(GEN + 10), span(GEN + 12, 8, 16), 'ScalarU32(0)'])
        report = self.run_case(1)
        self.assertIn('is not whole and the resource declares no layout', json.dumps(report['journals']))

    def test_partial_span_follows_a_declared_layout(self):
        self.decl['resources']['resized'][0]['layout'] = dict(planes=2, from_stride=16, to_stride=32, from_tail=0, to_tail=0)
        self.base['step'][0] = launch(1, EP, 8, [span(GEN + 8), span(GEN + 10, 16, 16), 'ScalarU32(0)'])
        self.head['step'][0] = launch(1, SPLIT, 2, [span(GEN + 10), span(GEN + 12, 32, 32), 'ScalarU32(0)'])
        self.run_case(0)

    def test_stride_argument_follows_the_layout(self):
        self.decl['resources']['resized'][0]['layout'] = dict(planes=2, from_stride=16, to_stride=32, from_tail=0, to_tail=0)
        self.decl['kernels'].append(dict(name='copy, grown planes', from_symbol='chop_copy_v1', to_symbol='chop_copy_v1',
                                         strides={'1': 0}))
        self.base['step'].append(launch(1, 'chop_copy_v1', 1, [span(GEN + 10, 0, 32), 'ScalarU64(16)']))
        self.head['step'].append(launch(1, 'chop_copy_v1', 1, [span(GEN + 12, 0, 64), 'ScalarU64(32)']))
        self.run_case(0)
        self.head['step'][-1] = launch(1, 'chop_copy_v1', 1, [span(GEN + 12, 0, 64), 'ScalarU64(16)'])
        self.run_case(1)

    # Host-buffer spans.

    def test_host_span_length_is_declared_exactly(self):
        for k, (old, new) in ((2, (8, 10)), (6, (11, 14))):
            self.base['step'][k] = launch(k // 4 + 1, ATTN, 4, [span(GEN + old), span(HOST, 64, 72)])
            self.head['step'][k] = launch(k // 4 + 1, ATTN, 4, [span(GEN + new), span(HOST, 64, 288)])
        self.run_case(1)
        self.decl['kernels'].append(dict(name='attention scales', from_symbol=ATTN, to_symbol=ATTN, lengths={'1': {'72': 288}}))
        report = self.run_case(0)
        self.assertEqual(report['coverage']['unbounded_host_span_lengths'], 2)
        self.decl['kernels'][-1]['lengths'] = {'1': {'72': 144}}
        self.run_case(1)

    def test_host_span_lengths_cannot_resize_a_device_allocation(self):
        self.decl['kernels'].append(dict(name='attention', from_symbol=ATTN, to_symbol=ATTN, lengths={'0': {'64': 128}}))
        self.run_case(1)

    # Unranked devices: a rank per synthetic stream.

    def unranked(self, world, streams):
        """The world with no recorded streams and unranked allocations on
        synthetic streams, its launches labelled by rank."""
        world['create'] = [('id', i[1]) if i[0] == 'stream' else i for i in world['create']]
        world['create'] = [('alloc', i[1], None, i[3], streams[i[2]]) if i[0] == 'alloc' else i for i in world['create']]
        world['create'] = [('zero', streams[i[1] - 1], i[2], i[3]) if i[0] == 'zero' else i for i in world['create']]
        world['create'] = [f'synchronize {streams[int(i.split()[1]) - 1]}' if isinstance(i, str) and i.startswith('synchronize ') else i
                           for i in world['create']]
        world['step'] = [line.replace('launch 1 ', 'launch r0 ').replace('launch 2 ', 'launch r1 ') for line in world['step']]
        return world

    def test_unranked_device_keys_ranks_by_synthetic_stream(self):
        streams = {0: 3000, 1: 3001}
        self.base, self.head = self.unranked(self.base, streams), self.unranked(self.head, streams)
        self.run_case(0)

    def test_unranked_device_sharing_one_stream_fails_the_rank_count(self):
        streams = {0: 3000, 1: 3000}
        self.base, self.head = self.unranked(self.base, streams), self.unranked(self.head, streams)
        report = self.run_case(1, 'green', lambda spec: [e.update(rank=None) for rule in spec['auxiliary_rules'].values()
                                                         for e in rule.get('added_allocations', []) + rule.get('resized_allocations', [])])
        self.assertIn('holds 2 comm.f32_inbox, the declaration 1', json.dumps(report['journals']))

    # Text journals.

    def test_text_journal_edits_must_be_declared(self):
        self.text = (['asks a'], ['asks a', 'asks rounded'])
        self.decl['text_journals'] = [TEXT]
        self.run_case(0)
        self.decl['text_journals'] = []
        data = self.data()
        b, h = self.captures(data)
        with self.assertRaisesRegex(Invalid, 'declaration does not list it'):
            self.generate(b, h)
        self.decl['text_journals'] = [TEXT]
        b2, h2 = self.captures(self.data())
        spec = self.generate(b2, h2)
        spec['layout']['sha256'] = hashlib.sha256(data).hexdigest()
        code, _ = self.prove(spec, b, h)
        self.assertEqual(code, 2)
        del spec['journals'][TEXT]
        code, report = self.prove(spec, b, h)
        self.assertEqual(code, 1)
        self.assertIn('declares no text_edits', report['journals'][TEXT]['reasons'][0])

    # The declaration's binding.

    def test_declaration_digest_must_match(self):
        self.run_case(2, spec_edit=lambda spec: spec['layout'].update(sha256='0' * 64))

    def test_declaration_must_be_committed_with_the_head_unless_named(self):
        data = self.data()
        b, h = self.captures(data, in_head=False)
        later = self.git.commit({CREATE: [], STEP: []}, extra_files={DECL: data})
        spec = self.generate(b, h, later)
        code, report = self.prove(spec, b, h)
        self.assertEqual(code, 2, report)
        self.assertIn('git show failed', report['problems'][0])
        code, report = self.prove(spec, b, h, '--declaration-revision', later)
        self.assertEqual(code, 0, report['problems'])
        self.assertFalse(report['layout_declaration']['read_at_head'])
        self.assertIn('declared after it was made', ' '.join(report['claim']['not_compared']))

    def test_layout_spec_takes_no_other_device_policy(self):
        self.run_case(2, spec_edit=lambda spec: spec['journals'].update({STEP: dict(launch_rules=[])}))
        self.run_case(2, spec_edit=lambda spec: next(iter(spec['auxiliary_rules'].values()))['create_policy'].update(moves=[]))

    def test_witness_fields_need_a_layout(self):
        self.run_case(2, spec_edit=lambda spec: spec.pop('layout'))

    def test_declaration_revision_needs_a_layout_spec(self):
        b, h = self.captures()
        spec = self.generate(b, h)
        spec.pop('layout')
        for rule in spec['auxiliary_rules'].values():
            for key in layout.WITNESS_FIELDS | {'removed_resolves', 'added_resolves', 'added_modules'}:
                rule.pop(key, None)
            rule.update(added_resolves=[], added_modules=[])
        code, report = self.prove(spec, b, h, '--declaration-revision', h[0])
        self.assertEqual(code, 2)
        self.assertIn('needs a spec that names a layout declaration', report['problems'][0])

    def test_declarations_are_read_strictly(self):
        for edit in (lambda d: d.update(extra=1), lambda d: d.pop('description'),
                     lambda d: d['resources']['added'][0].update(lifetime='step'),
                     lambda d: d['resources']['resized'][0].update(to=32),
                     lambda d: d['resources']['resized'][0].update(layout=dict(planes=2, from_stride=8, to_stride=32, from_tail=0, to_tail=0)),
                     lambda d: d['kernels'][0].update(args={'2': {'ScalarU32(1)': 'ScalarU32(1)'}}),
                     lambda d: d['kernels'][0]['cases'][0].update(when={}),
                     lambda d: d['kernels'][0].update(peers={'1': 'comm.f32_inbox'}),
                     lambda d: d['collectives'][0].pop('peers'),
                     lambda d: d['modules'].update(added=[dict(name='moe_ep', loads_like='core')]),
                     lambda d: d['modules'].update(added=['moe_split', 'comm_f32']),
                     lambda d: d['modules']['added'][0].pop('loads_like'),
                     lambda d: d['modules']['added'][1].update(configures={F32: []}),
                     lambda d: d['resources']['added'][0].pop('alongside'),
                     lambda d: d['kernels'].append(dict(d['kernels'][0]))):
            decl = declaration()
            edit(decl)
            with self.assertRaises(Invalid):
                layout.parse(json.dumps(decl).encode())
        layout.parse(json.dumps(declaration()).encode())

    def test_committed_declarations_parse(self):
        # Each model's declarations (README, Layout declarations).
        for path in sorted(Path(__file__).resolve().parents[2].glob('models/*/journal-changes/*.json')):
            with self.subTest(path.name):
                decl = layout.parse(path.read_bytes())
                self.assertTrue(decl['changes'] or decl['resources'] or decl['removed_modules'] or decl['added_modules'])

    # Review jproof-1 probes: each was PROVEN at d6d677af3.

    def after_resolve(self, ident, item):
        """Insert `item` into the head create right after resolve `ident`."""
        at = next(k for k, i in enumerate(self.head['create']) if i[:2] == ('resolve', ident))
        self.head['create'].insert(at + 1, item)

    def test_extra_configures_of_an_added_resolve_fail(self):
        self.head['create'] += [('configure', 9, 99999), ('configure', 9, 12345)]
        report = self.run_case(1)
        self.assertIn('configure 9 is detached from its resolve', report['journals'][STEP]['reasons'][0])
        self.head = head_world()
        self.after_resolve(9, ('configure', 9, 99999))
        report = self.run_case(1)
        self.assertIn('is configured [99999], the declaration []', report['journals'][STEP]['reasons'][0])

    def test_declared_configures_are_exact(self):
        self.after_resolve(9, ('configure', 9, 4096))
        self.decl['modules']['added'][1]['configures'] = {F32: [4096]}
        self.run_case(0)
        self.decl['modules']['added'][1]['configures'] = {F32: [8192]}
        report = self.run_case(1)
        self.assertIn('is configured [4096], the declaration [8192]', report['journals'][STEP]['reasons'][0])

    def test_extra_load_of_an_added_module_fails(self):
        self.head['create'] += [('module', 18, 'comm_f32'), ('resolve', 19, 18, F32), ('configure', 19, 99999)]
        report = self.run_case(1)
        self.assertIn('the head loads comm_f32 2 times, the base core 1', report['journals'][STEP]['reasons'][0])

    def test_zero_on_another_ranks_stream_fails(self):
        self.head['create'].remove(('zero', 2, 17, 256))
        self.head['create'].append(('zero', 1, 17, 256))
        report = self.run_case(1)
        self.assertIn('0 whole zeros in the create range', report['journals'][STEP]['reasons'][0])

    def without_collectives(self):
        self.decl['collectives'] = []
        for i, data in ((1, 10), (5, 14)):
            self.head['step'][i] = launch(i // 4 + 1, BF16, 4, [span(GEN + data), peers((GEN + 11, 128), (GEN + 15, 128)),
                                                                 'ScalarU32(64)'])

    def test_rank_missing_every_added_buffer_fails(self):
        # No launch names the added buffers; rank 1 allocates none of them.
        self.without_collectives()
        self.run_case(0)
        self.head['create'].remove(('alloc', 17, 1, 256, 2))
        self.head['create'].remove(('zero', 2, 17, 256))
        report = self.run_case(1)
        self.assertIn('rank 1 holds 0 comm.f32_inbox, the declaration 1', report['journals'][STEP]['reasons'][0])

    def test_added_buffer_on_a_rank_without_its_anchor_fails(self):
        self.without_collectives()
        self.decl['resources']['added'][0]['alongside'] = dict(bytes=999, alignment=256)
        report = self.run_case(1)
        self.assertIn('rank 0 holds 1 comm.f32_inbox, the declaration 0', report['journals'][STEP]['reasons'][0])

    def test_rank_missing_its_resize_fails(self):
        i = self.head['create'].index(('alloc', 16, 1, 64, 2))
        self.head['create'][i] = ('alloc', 16, 1, 32, 2)
        self.head['step'][4] = self.head['step'][4].replace(span(GEN + 16, 0, 64), span(GEN + 16, 0, 32))
        report = self.run_case(1)
        self.assertIn('rank 1 resizes 0 moe.route, the declaration 1', report['journals'][STEP]['reasons'][0])

    def test_collective_over_a_repeated_rank_fails(self):
        for i in (1, 5):
            self.base['step'][i] = self.base['step'][i].replace(bare(GEN + 12, 128), bare(GEN + 9, 128))
            self.head['step'][i] = self.head['step'][i].replace(bare(GEN + 17, 256), bare(GEN + 13, 256))
        report = self.run_case(1)
        self.assertIn('does not name 2 distinct ranks', report['journals'][STEP]['reasons'][0])

    # Review jproof-1 r2: a declared zero must initialize its buffer before
    # any use, in stream order. Each red case was PROVEN at cb850d576.

    def launches_in_create(self):
        """Both worlds run their step launches inside create as well; the step
        copies get grid_z 2 so the two bind separately."""
        for world in (self.base, self.head):
            world['create'].extend(world['step'])
            world['step'] = [line.replace('grid_z: 1', 'grid_z: 2') for line in world['step']]

    def test_zero_before_its_uses_in_create_is_proven(self):
        self.launches_in_create()
        self.run_case(0)

    def test_zero_after_a_use_in_create_fails(self):
        self.launches_in_create()
        for zero in [('zero', 1, 13, 256), ('zero', 2, 17, 256)]:
            self.head['create'].remove(zero)
            self.head['create'].append(zero)
        report = self.run_case(1)
        self.assertIn('runs before its zero', json.dumps(report['journals']))

    def without_syncs(self, *worlds):
        for world in worlds:
            world['create'] = [i for i in world['create'] if not (isinstance(i, str) and i.startswith('synchronize '))]

    def test_cross_stream_use_without_ordering_fails(self):
        # Rank 0's F32 all-reduce names rank 1's inbox, which stream 2 zeroes.
        self.without_syncs(self.base, self.head)
        report = self.run_case(1)
        self.assertIn('comm.f32_inbox 4294967313: a use (head note', report['journals'][STEP]['reasons'][0])
        self.assertIn('is not ordered after its zero', report['journals'][STEP]['reasons'][0])

    def with_events(self, waits):
        """Order the ranks' streams with events instead of synchronizes: after
        create, each world records one event per (record stream, wait stream)
        pair."""
        self.without_syncs(self.base, self.head)
        for world, first in ((self.base, 14), (self.head, 18)):
            world['history'] = [item for k, (record, wait) in enumerate(waits)
                                for item in (('id', first + k), f'event {first + k}', f'record {record} {first + k}', f'wait {wait} {first + k}')]

    def test_cross_stream_use_after_record_and_wait_is_proven(self):
        self.with_events([(1, 2), (2, 1)])
        self.run_case(0)

    def test_record_and_wait_in_one_direction_only_fails(self):
        self.with_events([(1, 2)])
        report = self.run_case(1)
        self.assertIn('comm.f32_inbox 4294967313: a use', report['journals'][STEP]['reasons'][0])

    def test_initialization_order_follows_graph_replays(self):
        raw = GEN + 13
        use = launch(1, F32, 4, [span(raw)])
        inits = {raw: ('comm.f32_inbox', 0, 1, 2)}
        ordered = ['allocate 256 256', f'zero 2 {bare(raw, 256)}', 'capture 1', use, 'captured 1 40 1']
        layout.initialization_order(ordered + ['synchronize 2', 'replay 1 40'], inits)
        layout.initialization_order(ordered + ['record 2 41', 'wait 1 41', 'replay 1 40'], inits)
        with self.assertRaisesRegex(Unproven, 'graph 40 replay .* is not ordered after its zero'):
            layout.initialization_order(ordered + ['replay 1 40'], inits)
        with self.assertRaisesRegex(Unproven, 'graph 40 replay'):
            layout.initialization_order(ordered + ['record 1 41', 'wait 1 41', 'replay 1 40'], inits)
        # A graph captured before the zero runs at its replays: a replay after
        # the ordered zero is valid, one before it is not.
        early = ['allocate 256 256', 'capture 1', use, 'captured 1 40 1']
        layout.initialization_order(early + [f'zero 2 {bare(raw, 256)}', 'synchronize 2', 'replay 1 40'], {raw: ('comm.f32_inbox', 0, 4, 2)})
        with self.assertRaisesRegex(Unproven, 'graph 40 replay .* runs before its zero'):
            layout.initialization_order(early + ['replay 1 40', f'zero 2 {bare(raw, 256)}'], {raw: ('comm.f32_inbox', 0, 5, 2)})
        with self.assertRaisesRegex(Unproven, 'its zero .* is captured into a graph'):
            layout.initialization_order(['allocate 256 256', 'capture 2', f'zero 2 {bare(raw, 256)}', 'captured 2 40 0'],
                                        {raw: ('comm.f32_inbox', 0, 2, 2)})
        with self.assertRaisesRegex(Unproven, 'zeroed before it is allocated'):
            layout.initialization_order([f'zero 2 {bare(raw, 256)}', 'allocate 256 256'], {raw: ('comm.f32_inbox', 1, 0, 2)})

    # Review jproof-1 r3: graph capture follows FakeDevice's membership
    # (a stream that waits on a captured event joins the capture), and
    # create and pinned operations map event and graph identities as the
    # history does. Each red case was PROVEN at 0c2e3b693; the valid-member
    # green was refused there.

    def move_f32_zeros_to_the_end_of_create(self):
        for zero in [('zero', 1, 13, 256), ('zero', 2, 17, 256)]:
            self.head['create'].remove(zero)
            self.head['create'].append(zero)

    def forked_graph(self, lead_is_new):
        """History orders streams 1 and 2 with reciprocal events, then captures
        rank 0's four step launches into a graph and replays it. With
        `lead_is_new` the capture is on a new rank-0 stream and forks to
        stream 1, which issues the launches; otherwise it is on stream 1 and
        forks to the new stream, which issues them. The step copies get
        grid_z 2 so they bind separately."""
        for world, first in ((self.base, 14), (self.head, 18)):
            new, a, b, fork, join, graph = range(first, first + 6)
            lead, member = (new, 1) if lead_is_new else (1, new)
            captured = [line.replace('launch 1 ', f'launch {member} ') for line in world['step'][:4]]
            world['history'] = [
                ('stream', new, 0),
                ('id', a), f'event {a}', f'record 1 {a}', f'wait 2 {a}',
                ('id', b), f'event {b}', f'record 2 {b}', f'wait 1 {b}',
                ('id', fork), f'event {fork}', ('id', join), f'event {join}',
                f'capture {lead}', f'record {lead} {fork}', f'wait {member} {fork}',
                *captured, f'record {member} {join}', f'wait {lead} {join}',
                ('id', graph), f'captured {lead} {graph} 4', f'replay {lead} {graph}']
            world['step'] = [line.replace('grid_z: 1', 'grid_z: 2') for line in world['step']]

    def test_graph_forked_to_a_member_is_proven(self):
        self.forked_graph(lead_is_new=True)
        self.run_case(0)

    def test_member_work_replayed_on_an_unordered_lead_fails(self):
        # Streams 1 and 2 stay ordered after the moved zeros; the replay
        # stream, which ran the graph, does not.
        self.forked_graph(lead_is_new=True)
        self.move_f32_zeros_to_the_end_of_create()
        report = self.run_case(1)
        self.assertIn('graph 23 replay (head note', report['journals'][STEP]['reasons'][0])
        self.assertIn('is not ordered after its zero', report['journals'][STEP]['reasons'][0])

    def test_member_work_replayed_on_an_ordered_lead_is_proven(self):
        # The member's launches run at the replay on stream 1, which the
        # history's events order after both zeros.
        self.forked_graph(lead_is_new=False)
        self.run_case(0)
        self.move_f32_zeros_to_the_end_of_create()
        self.run_case(0)

    def create_event(self, base_event, head_event):
        """Eight events at the end of both creates; create records one of them
        after a launch on stream 1 that reads rank 0's old inbox, and the
        history waits on it from stream 2."""
        for world, first, inbox, event in ((self.base, 14, 9, base_event), (self.head, 18, 11, head_event)):
            world['create'] += [item for e in range(first, first + 8) for item in (('id', e), f'event {e}')]
            world['create'] += [launch(1, ATTN, 4, [span(GEN + inbox)]), f'record 1 {event}']
            world['history'] = [f'wait 2 {first + 6}']

    def test_create_record_of_the_mapped_event_is_proven(self):
        self.create_event(20, 24)
        self.run_case(0)

    def test_create_record_of_another_event_fails(self):
        # Head event 20 exists, but the base's event 20 maps to head event 24,
        # which the history waits on and nothing now records.
        self.create_event(20, 20)
        report = self.run_case(1)
        self.assertIn('own fixture create differs', report['journals'][STEP]['reasons'][0])
        self.assertIn('record 1 24', report['journals'][STEP]['reasons'][0])

    def test_member_of_a_forked_capture_is_checked_at_replay(self):
        raw = GEN + 13
        use = launch(1, F32, 4, [span(raw)])
        inits = {raw: ('comm.f32_inbox', 0, 1, 1)}
        forked = ['allocate 256 256', f'zero 1 {bare(raw, 256)}', 'capture 3', 'record 3 40', 'wait 1 40', use,
                  'record 1 41', 'wait 3 41', 'captured 3 42 1']
        with self.assertRaisesRegex(Unproven, 'graph 42 replay .* is not ordered after its zero'):
            layout.initialization_order(forked + ['replay 3 42'], inits)
        layout.initialization_order(forked + ['replay 1 42'], inits)
        layout.initialization_order(forked + ['record 1 43', 'wait 3 43', 'replay 3 42'], inits)
        # Captured record and wait order nothing outside the graph: stream 3
        # waited on 41 only inside the capture.
        with self.assertRaisesRegex(Unproven, 'a use .* is not ordered after its zero'):
            layout.initialization_order(forked + [launch(3, F32, 4, [span(raw)])], inits)

    # Review jproof-1 r4: what the capture model and the mapper do not model
    # is NOT PROVEN with a reason. Each red case was PROVEN at df4efe8e3; the
    # readback greens were refused there.

    def overlapping_captures(self, world, first):
        """Stream 1 joins two open captures (FakeDevice puts its launches in
        both graphs); only the second graph is replayed."""
        a, b, ordered_a, ordered_b, fork_a, fork_b, graph_a, graph_b = range(first, first + 8)
        world['history'] = [
            ('stream', a, 0), ('stream', b, 0),
            ('id', ordered_a), f'event {ordered_a}', f'record 1 {ordered_a}', f'wait 2 {ordered_a}',
            ('id', ordered_b), f'event {ordered_b}', f'record 2 {ordered_b}', f'wait 1 {ordered_b}',
            ('id', fork_a), f'event {fork_a}', ('id', fork_b), f'event {fork_b}',
            f'capture {a}', f'record {a} {fork_a}', f'wait 1 {fork_a}',
            f'capture {b}', f'record {b} {fork_b}', f'wait 1 {fork_b}',
            *world['step'][:4],
            ('id', graph_a), f'captured {a} {graph_a} 4',
            ('id', graph_b), f'captured {b} {graph_b} 4', f'replay {b} {graph_b}']
        world['step'] = [line.replace('grid_z: 1', 'grid_z: 2') for line in world['step']]

    def test_a_stream_in_two_open_captures_is_not_modelled(self):
        self.overlapping_captures(self.base, 14)
        self.overlapping_captures(self.head, 18)
        for moved in (False, True):
            if moved:
                self.move_f32_zeros_to_the_end_of_create()
            report = self.run_case(1)
            self.assertIn('stream 1 joins the capture on 19 while in the capture on 18, which is not modelled',
                          report['journals'][STEP]['reasons'][0])

    def fixed_stream_copy(self, owner, stream):
        """Created stream 14 (head 18) uploads into rank 0's old inbox, then a
        copy on fixed (unrecorded) host stream `stream` reads it."""
        for world, created, inbox, data in ((self.base, 14, 9, 8), (self.head, 18, 11, 10)):
            world['create'].append(('stream', created, 0))
            world[owner] += [f'upload {created} {bare(GEN + inbox, 64)} 0000000000000001',
                             f'copy {stream} {bare(GEN + inbox, 64)} {bare(GEN + data, 64)}']

    def test_a_fixed_host_stream_distinct_from_created_streams_is_proven(self):
        for owner in ('create', 'step'):
            with self.subTest(owner):
                self.base, self.head = base_world(), head_world()
                self.fixed_stream_copy(owner, 40)
                self.run_case(0)

    def test_a_fixed_host_stream_may_not_alias_a_created_stream(self):
        # The base runs the upload and the copy on streams 14 and 18; the head
        # runs both on 18, the image of 14.
        for owner in ('create', 'step'):
            with self.subTest(owner):
                self.base, self.head = base_world(), head_world()
                self.fixed_stream_copy(owner, 18)
                report = self.run_case(1)
                self.assertIn('stream 18 has no recorded creation and is the image of a created stream',
                              json.dumps(report['journals']))

    def readback(self, owner, head_stream):
        """Eight created rank-0 streams; a launch on the seventh writes rank
        0's old inbox and a readback on `head_stream` (head) reads it."""
        for world, first, inbox in ((self.base, 14, 9), (self.head, 18, 11)):
            world['create'] += [('stream', stream, 0) for stream in range(first, first + 8)]
            stream = first + 6 if world is self.base else head_stream
            world[owner] += [launch(first + 6, ATTN, 4, [span(GEN + inbox)]),
                             f'readback {stream} {bare(GEN + inbox, 64)} 64']

    def test_a_readback_on_the_mapped_stream_is_proven(self):
        for owner in ('create', 'step'):
            with self.subTest(owner):
                self.base, self.head = base_world(), head_world()
                self.readback(owner, 24)
                self.run_case(0)

    def test_a_readback_on_another_stream_fails(self):
        # Head stream 20 exists, but base stream 20 maps to head 24.
        for owner in ('create', 'step'):
            with self.subTest(owner):
                self.base, self.head = base_world(), head_world()
                self.readback(owner, 20)
                report = self.run_case(1)
                reason = report['journals'][CREATE if owner == 'create' else STEP]['reasons'][0]
                self.assertIn('"expected": "readback 24 ' if owner == 'create' else 'unexplained line differences', reason)

    def test_an_unmodelled_journal_line_fails(self):
        for owner, line in (('create', 'prefetch 1 4096'), ('step', 'step two {raw: 9}')):
            with self.subTest(owner):
                self.base, self.head = base_world(), head_world()
                for world in (self.base, self.head):
                    world[owner].append(line)
                report = self.run_case(1)
                self.assertIn('unrecognized journal line form', json.dumps(report['journals']))

    def test_unmodelled_capture_cases_fail(self):
        raw = GEN + 13
        use = launch(1, F32, 4, [span(raw)])
        inits = {raw: ('comm.f32_inbox', 0, 1, 1)}
        start = ['allocate 256 256', f'zero 1 {bare(raw, 256)}']
        cases = {
            'stream 1 begins a capture while in the capture on 3': ['capture 3', 'record 3 40', 'wait 1 40', 'capture 1'],
            'stream 3 joins the capture on 4 while in the capture on 3': ['capture 3', 'capture 4', 'record 4 40', 'wait 3 40'],
            'names it in a copy on a captured stream': ['capture 3', 'record 3 40', 'wait 1 40',
                                                        f'copy 1 {bare(raw, 256)} {bare(GEN + 10, 256)}'],
            'is replayed inside a capture': ['capture 1', use, 'captured 1 42 1', 'capture 3', 'replay 3 42'],
        }
        for reason, notes in cases.items():
            with self.subTest(reason), self.assertRaisesRegex(Unproven, reason):
                layout.initialization_order(start + notes, inits)
        # A lead waiting on its own event, and a member waiting again on its
        # capture's event, stay modelled.
        layout.initialization_order(start + ['capture 3', 'record 3 40', 'wait 3 40', 'wait 1 40', 'record 1 41', 'wait 1 40',
                                             use, 'captured 3 42 1', 'replay 1 42'], inits)

    def test_a_fixed_handle_in_create_may_not_alias_an_allocation_image(self):
        # Base raw GEN+14 is no base allocation (a host buffer), but base
        # allocation 11 maps onto head GEN+14. No journal pins the create.
        for world in (self.base, self.head):
            world['create'].append(launch(1, ATTN, 4, [span(GEN + 14)]))
            world['create_unpinned'] = True
        report = self.run_case(1)
        self.assertIn('allocation mapping redirects a fixed resource', report['journals'][STEP]['reasons'][0])

    def test_a_replay_of_a_graph_never_captured_fails(self):
        raw = GEN + 13
        with self.assertRaisesRegex(Unproven, 'graph 42 is replayed but was never captured'):
            layout.initialization_order(['allocate 256 256', f'zero 1 {bare(raw, 256)}', 'replay 1 42'],
                                        {raw: ('comm.f32_inbox', 0, 1, 1)})

    def test_a_case_may_not_replace_a_buffer_argument_with_a_literal(self):
        self.decl['kernels'][0]['cases'][0]['then']['args'] = {'0': 'ScalarU32(0)'}
        for i in (0, 4):
            self.head['step'][i] = self.head['step'][i].replace(span(GEN + (10 if i == 0 else 14)) + ',', 'ScalarU32(0),', 1)
        report = self.run_case(1)
        self.assertIn('a case replaces argument 0, which is not a scalar literal', report['journals'][STEP]['reasons'][0])

    def test_a_declared_record_must_have_its_exact_form(self):
        # Added resolve 9 resolves F32 in the trace, but its note names
        # another symbol.
        i = self.head['create'].index(('resolve', 9, 8, F32))
        self.head['create'][i:i + 1] = [('id', 9), 'resolve 9 chop_other_v1', ('event', ['resolve', 9, 8, F32])]
        report = self.run_case(1)
        self.assertIn('resolve 9 does not name the symbol its trace resolves', report['journals'][STEP]['reasons'][0])

    def test_a_declared_added_allocation_may_not_be_a_fixed_handle(self):
        # Base raw GEN+17 is a host handle; the head's added rank-1 F32 inbox
        # has that handle.
        for world in (self.base, self.head):
            world['create'].append(launch(1, ATTN, 4, [span(GEN + 17)]))
        report = self.run_case(1)
        self.assertIn('a declared added allocation has the handle of a fixed resource', report['journals'][STEP]['reasons'][0])

    # A device that never runs the instance create (its create is only its
    # allocation prefix) zeroes no declared buffer. The head may use an added
    # `init: zero` buffer there only where the base buffer it replaces, in the
    # same declared collective argument, is unwritten on the base device too.
    # On a plan-only device (only allocations and `launch rN` plan records)
    # nothing executes, so plan records are not writes.

    def without_instance_create(self, *worlds, plan=True):
        """Only allocations before the first launch: streams are bare IDs,
        resolves trace events without notes, and nothing is configured or
        zeroed; no create marker and no create journal. With `plan`, as on the
        real lowering devices, the step is plan records `launch rN LABEL ...`,
        both ranks' MoE reduces before either attention reduce, so on a device
        a test makes executing only its own records write the replaced inboxes
        before the MoE reduces read them. Without it the step stays executed
        launches in rank order."""
        labels = {EP: 'Moe', SPLIT: 'Moe', BF16: 'Reduce', F32: 'Reduce', ATTN: 'Attention'}
        for world in worlds:
            if plan:
                step = []
                for line in [world['step'][k] for k in (0, 1, 4, 5, 2, 3, 6, 7)]:
                    _, stream, symbol, rest = line.split(' ', 3)
                    step.append(f'launch r{int(stream) - 1} {labels[symbol]} {symbol} {rest}')
                world['step'] = step
            out = []
            for item in world['create']:
                if isinstance(item, str) or item[0] in ('configure', 'zero'):
                    continue
                if item[0] == 'stream':
                    out.append(('id', item[1]))
                elif item[0] == 'resolve':
                    out += [('id', item[1]), ('event', ['resolve', *item[1:]])]
                else:
                    out.append(item)
            world['create'] = out
            world['no_instance_create'] = True

    def run_without_instance_create(self, expected, plan):
        if not hasattr(layout, 'unzeroed_uses'):
            # An older tool, in a red run: its verdict alone.
            return self.run_case(expected)
        with patch('layout.unzeroed_uses', wraps=layout.unzeroed_uses) as rule:
            report = self.run_case(expected)
        self.assertTrue(rule.called, 'the device did not bind as an allocation prefix')
        if hasattr(layout, 'plan_only'):
            self.assertEqual(rule.call_args.args[5], plan, 'the device is not the plan-only or executing device the test builds')
        return report

    def zero_after_the_first_launch(self, world, raws):
        """Zeroes of `raws` (raw, length) on stream 1 after rank 0's first
        launch, which make the device executing."""
        world['step'][1:1] = [f'zero 1 {bare(GEN + raw, length)}' for raw, length in raws]

    def test_a_device_without_instance_create_keeps_the_base_initialization(self):
        # Neither revision zeroes anything, as the real lowering devices: a
        # plan-only device, whose plan records name the replaced inboxes.
        self.without_instance_create(self.base, self.head)
        self.run_without_instance_create(0, True)

    def test_executed_launches_writing_the_replaced_buffer_fail(self):
        # The same step executed in rank order: rank 0's attention reduce
        # writes both old inboxes before rank 1's MoE reduce reads them.
        self.without_instance_create(self.base, self.head, plan=False)
        report = self.run_without_instance_create(1, False)
        self.assertIn(f'base note 10 writes {GEN + 9}, which it replaces, before the collective at base note 12 reads it',
                      report['journals'][STEP]['reasons'][0])

    def test_one_executed_launch_on_a_plan_device_makes_it_executing(self):
        # Plan records but one executed launch writing rank 0's old inbox
        # (and its image) before the MoE reduce reads it.
        self.without_instance_create(self.base, self.head)
        for world, inbox in ((self.base, 9), (self.head, 11)):
            world['step'][1:1] = [launch(1, ATTN, 4, [span(GEN + inbox)])]
        report = self.run_without_instance_create(1, False)
        self.assertIn(f'base note 8 writes {GEN + 9}, which it replaces, before the collective at base note 9 reads it',
                      report['journals'][STEP]['reasons'][0])

    def test_zeroing_unrelated_buffers_keeps_the_replaced_buffers_status(self):
        # Both revisions zero rank 0's data buffer, which no collective
        # argument replaces; the replaced inboxes stay unzeroed on both.
        self.without_instance_create(self.base, self.head)
        self.zero_after_the_first_launch(self.base, [(8, 64)])
        self.zero_after_the_first_launch(self.head, [(10, 64)])
        self.run_without_instance_create(0, False)

    def test_an_unzeroed_added_buffer_whose_base_buffer_is_zeroed_fails(self):
        # The base zeroes rank 0's old inbox, which the F32 inbox replaces in
        # the collective's peer array; the head zeroes the old inbox's image
        # but not the F32 inbox.
        self.without_instance_create(self.base, self.head)
        self.zero_after_the_first_launch(self.base, [(9, 128)])
        self.zero_after_the_first_launch(self.head, [(11, 128)])
        report = self.run_without_instance_create(1, False)
        self.assertIn(f'is used unzeroed on a device that never runs the instance create, but base note 8 writes {GEN + 9}',
                      report['journals'][STEP]['reasons'][0])

    def test_a_device_that_zeroes_other_buffers_but_not_the_added_one_fails(self):
        # Both revisions zero their data, route and old inbox buffers; only
        # the F32 inboxes, which replace the old inboxes, stay unzeroed.
        self.without_instance_create(self.base, self.head)
        self.zero_after_the_first_launch(self.base, [(8, 64), (9, 128), (10, 32)])
        self.zero_after_the_first_launch(self.head, [(10, 64), (11, 128), (12, 64)])
        report = self.run_without_instance_create(1, False)
        self.assertIn(f'writes {GEN + 9}, which it replaces, before the collective', report['journals'][STEP]['reasons'][0])

    def test_an_unzeroed_use_whose_replaced_buffer_is_unknown_fails(self):
        raw = GEN + 13
        notes_b = ['allocate 256 256', launch(1, F32, 4, [span(raw)])]
        with self.assertRaisesRegex(Unproven, 'head note 2 uses it on a device that never runs the instance create, and no '
                                    'declared collective argument identifies the base buffer it replaces'):
            layout.unzeroed_uses([], notes_b, {raw: 'comm.f32_inbox'}, {}, {})
        layout.unzeroed_uses([], notes_b, {raw: 'comm.f32_inbox'}, {raw: {GEN + 9}}, {raw: {(GEN + 9, 0)}})

    # Review jproof-1 r5: any recorded write of the replaced base buffer
    # before the collective reads it initializes it, not only `zero`.

    def test_a_zero_upload_into_the_replaced_buffer_fails(self):
        self.without_instance_create(self.base, self.head)
        for world, raw in ((self.base, 9), (self.head, 11)):
            world['step'][1:1] = [f'upload 1 {bare(GEN + raw, 128)} {fnv(bytes(128))}']
        report = self.run_without_instance_create(1, False)
        self.assertIn(f'writes {GEN + 9}, which it replaces, before the collective', report['journals'][STEP]['reasons'][0])

    def test_a_copy_of_zeros_into_the_replaced_buffer_fails(self):
        self.without_instance_create(self.base, self.head)
        for world, scratch, inbox in ((self.base, 14, 9), (self.head, 18, 11)):
            world['create'].append(('alloc', scratch, 0, 128, 1))
            world['step'][1:1] = [f'zero 1 {bare(GEN + scratch, 128)}', f'copy 1 {bare(GEN + scratch, 128)} {bare(GEN + inbox, 128)}']
        report = self.run_without_instance_create(1, False)
        self.assertIn(f'writes {GEN + 9}, which it replaces, before the collective', report['journals'][STEP]['reasons'][0])

    def test_a_zero_upload_into_an_unrelated_buffer_keeps_the_exception(self):
        self.without_instance_create(self.base, self.head)
        for world, raw in ((self.base, 8), (self.head, 10)):
            world['step'][1:1] = [f'upload 1 {bare(GEN + raw, 64)} {fnv(bytes(64))}']
        self.run_without_instance_create(0, False)

    def test_only_reads_and_replacement_sites_leave_the_replaced_buffer_unwritten(self):
        raw, base = GEN + 13, GEN + 9
        use = launch(1, F32, 4, [span(raw)])
        other = bare(GEN + 8, 128)
        reads = [f'download 1 {bare(base, 128)}', f'readback 1 {bare(base, 128)} 128', f'copy 1 {bare(base, 128)} {other}',
                 f'peer copy 1 1 {bare(base, 128)} 2 {other}']
        collective = launch(1, BF16, 4, [span(base)])
        # Reads, and an earlier site replacing the same buffer, are not writes.
        notes_a = ['allocate 128 256', *reads, collective, collective]
        sites = {raw: {(base, 5), (base, 6)}}
        layout.unzeroed_uses(notes_a, [use], {raw: 'comm.f32_inbox'}, {raw: {base}}, sites)
        for write in (f'zero 1 {bare(base, 128)}', f'upload 1 {bare(base, 128)} {fnv(bytes(128))}',
                      f'copy 1 {other} {bare(base, 128)}', f'peer copy 1 1 {other} 2 {bare(base, 128)}',
                      launch(1, ATTN, 4, [span(base)]), f'step x view {bare(base, 128)}'):
            with self.subTest(write), self.assertRaisesRegex(Unproven, 'base note 2 writes'):
                layout.unzeroed_uses(['allocate 128 256', write, collective], [use], {raw: 'comm.f32_inbox'}, {raw: {base}},
                                     {raw: {(base, 2)}})

    def test_plan_records_are_writes_unless_the_device_is_plan_only(self):
        raw, base = GEN + 13, GEN + 9
        use = launch('r0 Reduce', F32, 4, [span(raw)])
        notes_a = ['allocate 128 256', launch('r0 Attention', ATTN, 4, [span(base)]), launch('r0 Reduce', BF16, 4, [span(base)])]
        args = [use], {raw: 'comm.f32_inbox'}, {raw: {base}}, {raw: {(base, 2)}}
        layout.unzeroed_uses(notes_a, *args, True)
        with self.assertRaisesRegex(Unproven, 'base note 2 writes'):
            layout.unzeroed_uses(notes_a, *args, False)

    def test_only_allocations_and_plan_records_make_a_plan_only_device(self):
        plan = ['allocate 128 256', launch('r1 Attention', ATTN, 4, [span(GEN + 9)])]
        self.assertTrue(layout.plan_only(plan))
        self.assertTrue(layout.plan_only([]))
        for record in ('stream 1', f'zero 1 {bare(GEN + 9, 128)}', f'upload 1 {bare(GEN + 9, 128)} {fnv(bytes(128))}',
                       f'copy 1 {bare(GEN + 8, 128)} {bare(GEN + 9, 128)}', 'synchronize 1', 'record 1 5', 'wait 2 5',
                       f'readback 1 {bare(GEN + 9, 128)} 128', launch(1, ATTN, 4, [span(GEN + 9)]), 'step x'):
            with self.subTest(record):
                self.assertFalse(layout.plan_only([*plan, record]))

    # Dropping any one declared item.

    def test_dropping_any_declared_item_fails(self):
        def items(decl):
            yield 'modules.removed', lambda d: d['modules'].update(removed=[]), \
                lambda s: [r.pop(k, None) for r in s['auxiliary_rules'].values() for k in ('removed_modules', 'removed_resolves')]
            for name in ('moe_split', 'comm_f32'):
                symbols = set(SYMBOLS[name])
                yield f'modules.added {name}', (lambda d, name=name: d['modules'].update(
                    added=[m for m in d['modules']['added'] if m['name'] != name])), \
                    (lambda s, name=name, symbols=symbols: [r.update(
                        added_modules=[e for e in r['added_modules'] if e['name'] != name],
                        added_resolves=[e for e in r['added_resolves'] if e['symbol'] not in symbols]) for r in s['auxiliary_rules'].values()])
            yield 'resources.added', lambda d: (d['resources'].update(added=[]), d.update(collectives=[])), \
                self.drop_witness('added_allocations', lambda e: True)
            yield 'resources.resized', lambda d: d['resources'].update(resized=[]), self.drop_witness('resized_allocations', lambda e: True)
            yield 'collective', lambda d: d.update(collectives=[]), None
            yield 'collective selection', lambda d: d['collectives'][0].pop('when'), None
            yield 'kernel family', lambda d: d.update(kernels=[]), None
            yield 'kernel case', lambda d: d['kernels'][0].pop('cases'), None
            yield 'kernel literal map', lambda d: d['kernels'][0].pop('args'), None
        for label, edit, witness in items(self.decl):
            with self.subTest(label):
                self.decl = declaration()
                edit(self.decl)
                data = self.data()
                b, h = self.captures(data)
                spec = self.green_spec()
                spec['layout']['sha256'] = hashlib.sha256(data).hexdigest()
                if witness:
                    witness(spec)
                code, report = self.prove(spec, b, h)
                self.assertNotEqual(code, 0, label)
                self.assertNotEqual(report['status'], 'PROVEN', label)


if __name__ == '__main__':
    unittest.main()
