#!/usr/bin/env python3
"""The spec writers' rules: specs/build.py (change declarations) and
specs/graph_nodes.py (graph node counts), without captures."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent / 'specs'))
import build  # noqa: E402
import graph_nodes  # noqa: E402
from common import Invalid  # noqa: E402
from testkit import CREATE, GEN, MODEL, STEP, launch as journal_launch  # noqa: E402

MODULE, NEW = 'row_toy', 'row_shared'
OLD_A, OLD_B = 'chop_row_toy_a_v1', 'chop_row_toy_b_v1'
SHARED_A, SHARED_B = 'chop_row_shared_a_v1', 'chop_row_shared_b_v1'
DECLARATION = dict(schema=build.SCHEMA, description='toy change', module=MODULE, removed_symbols=[OLD_A, OLD_B],
                   added_module=dict(name=NEW, symbols=[SHARED_A, SHARED_B]),
                   launch_rules=[dict(from_symbol=OLD_A, to_symbol=SHARED_A, spans=3, append=['ScalarU32(512)', 'ScalarU32(4)'],
                                      flags=1, expect=dict(block_x=256)),
                                 dict(from_symbol=OLD_B, to_symbol=SHARED_B, spans=3, append=['ScalarU32(9)'], flags=1)])
REMOVAL = dict(schema=build.SCHEMA, description='toy removal', module=MODULE, removed_symbols=[OLD_A, OLD_B])
ADDED = {300: ('module', NEW), 301: ('resolve', SHARED_A), 302: ('resolve', SHARED_B)}
SPAN = 'DeviceBuffer(DeviceBufferSpanV1 {{ buffer: BundleDeviceBufferHandleV1 {{ raw: {} }}, offset: 0, length: 64 }})'
CONFIG = ('LaunchConfigV1 { flags: 1, grid_x: 32, grid_y: 1, grid_z: 1, block_x: 256, block_y: 1, block_z: 1, '
          'shared_memory_bytes: 0, cluster: LaunchClusterV1 { x: 1, y: 1, z: 1 } }')


def launch(symbol, stream=2, args=None, flags=1):
    args = args if args is not None else [SPAN.format(4294967396 + i) for i in range(3)]
    return f'launch {stream} {symbol} {CONFIG.replace("flags: 1", f"flags: {flags}")} [' + ', '.join(args) + ']'


def lock(sha, symbols, flags=('-O3',), params=None, registers=None):
    params = params or {}
    registers = registers or {}
    return ('kernels.lock.d/x.lock', dict(
        name='x', flags=list(flags), cubin_sha256=sha,
        entries=[dict(symbol=s, params=params.get(s, [dict(kind='U32', name='n')]), max_threads=256) for s in symbols],
        resources=[dict(symbol=s, registers=registers.get(s, 32), shared_static_bytes=0) for s in symbols]))


LOCKED = 'a' * 64


def inventory(modules, resolves, payloads=None):
    """auxproof.inventory's shape: modules ID -> [kind, ID, name, fnv, bytes]
    plus the sealed payload SHA-256 for IDs in `payloads`, resolves ID ->
    [kind, ID, module, symbol]."""
    payloads = payloads or {}
    return dict(modules={i: ['module', i, name, f'{name}_hash', 100] + ([payloads[i]] if i in payloads else [])
                         for i, name in modules.items()},
                resolves={i: ['resolve', i, m, s] for i, (m, s) in resolves.items()})


class Declarations(unittest.TestCase):
    def test_a_declaration_is_read_strictly(self):
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / 'change.json'
            path.write_text(json.dumps(DECLARATION))
            self.assertEqual(build.read_declaration(path)['module'], MODULE)
            for broken in (dict(DECLARATION, extra=1), dict(DECLARATION, schema='x'), {k: v for k, v in DECLARATION.items() if k != 'module'},
                           dict(DECLARATION, launch_rules=[dict(from_symbol=OLD_A, to_symbol=SHARED_A)]),
                           dict(DECLARATION, removed_symbols=[OLD_A, OLD_A])):
                path.write_text(json.dumps(broken))
                with self.assertRaises(Invalid, msg=str(broken)[:80]):
                    build.read_declaration(path)


class Locks(unittest.TestCase):
    OLD = {MODULE: lock('a', ['x', OLD_A, OLD_B, 'y']), 'comm': lock('c', ['z'])}

    def test_the_module_loses_exactly_the_removed_entries(self):
        new = {MODULE: lock('b', ['x', 'y']), 'comm': lock('c', ['z'])}
        old_row, new_row = build.check_locks(self.OLD, new, REMOVAL)
        self.assertEqual((old_row[1]['cubin_sha256'], new_row[1]['cubin_sha256']), ('a', 'b'))

    def test_the_added_module_holds_exactly_its_symbols(self):
        new = {MODULE: lock('b', ['x', 'y']), 'comm': lock('c', ['z']), NEW: lock('n', [SHARED_A, SHARED_B])}
        build.check_locks(self.OLD, new, DECLARATION)
        with self.assertRaises(Invalid):
            build.check_locks(self.OLD, dict(new, **{NEW: lock('n', [SHARED_B, SHARED_A])}), DECLARATION)
        with self.assertRaises(Invalid):
            build.check_locks(self.OLD, {MODULE: lock('b', ['x', 'y']), 'comm': lock('c', ['z'])}, DECLARATION)

    def test_other_changes_are_refused(self):
        for new in (
            {MODULE: lock('b', ['x']), 'comm': lock('c', ['z'])},               # one more entry gone
            {MODULE: lock('b', ['y', 'x']), 'comm': lock('c', ['z'])},          # order changed
            {MODULE: lock('b', ['x', 'y']), 'comm': lock('d', ['z'])},          # another payload
            {MODULE: lock('b', ['x', 'y'])},                                    # a module removed
            {MODULE: lock('b', ['x', 'y']), 'comm': lock('c', ['z']), 'n': lock('e', [])},
            {MODULE: lock('a', ['x', 'y']), 'comm': lock('c', ['z'])},          # same payload
            {MODULE: lock('b', ['x', 'y'], params={'x': [dict(kind='U64', name='n')]}), 'comm': lock('c', ['z'])},
            {MODULE: lock('b', ['x', 'y'], registers={'y': 40}), 'comm': lock('c', ['z'])},
            {MODULE: lock('b', ['x', 'y'], flags=('-O2',)), 'comm': lock('c', ['z'])},
            {MODULE: lock('b', ['x', 'y']), 'comm': lock('c', ['z'], registers={'z': 64})},
        ):
            with self.assertRaises(Invalid, msg=str(new)[:120]):
                build.check_locks(self.OLD, new, REMOVAL)


class Resolves(unittest.TestCase):
    def test_each_module_loses_its_removed_resolves(self):
        a = inventory({10: MODULE, 20: MODULE}, {11: (10, OLD_A), 12: (10, OLD_B), 13: (10, 'y'), 21: (20, OLD_A), 22: (20, OLD_B)})
        b = inventory({10: MODULE, 18: MODULE}, {11: (10, 'y')})
        self.assertEqual(build.removed_resolves(a, b, 'f', REMOVAL), [dict(symbol=OLD_A, ids=[11, 21], count=2),
                                                                      dict(symbol=OLD_B, ids=[12, 22], count=2)])
        self.assertEqual(build.removed_resolves(inventory({}, {}), inventory({}, {}), 'f', REMOVAL), [])

    def test_a_missing_or_doubled_resolve_or_a_head_resolve_is_refused(self):
        rows = {10: MODULE, 20: MODULE}
        for a, b in (
            (inventory(rows, {11: (10, OLD_A), 12: (10, OLD_B), 22: (20, OLD_B)}), inventory(rows, {})),
            (inventory(rows, {11: (10, OLD_A), 12: (10, OLD_B), 13: (10, OLD_A), 21: (20, OLD_A), 22: (20, OLD_B)}), inventory(rows, {})),
            (inventory(rows, {11: (10, OLD_A), 12: (10, OLD_B), 21: (20, OLD_A), 22: (20, OLD_B)}), inventory(rows, {11: (10, OLD_B)})),
            (inventory(rows, {11: (10, OLD_A), 12: (10, OLD_B), 21: (20, OLD_A), 22: (20, OLD_B)}), inventory({10: MODULE}, {})),
        ):
            with self.assertRaises(Invalid):
                build.removed_resolves(a, b, 'f', REMOVAL)

    def test_one_added_module_per_changed_module_resolving_each_symbol_once(self):
        a = inventory({10: MODULE}, {11: (10, OLD_A)})
        b = inventory({10: MODULE, 30: NEW}, {31: (30, SHARED_A), 32: (30, SHARED_B)}, {30: LOCKED})
        rule, ids = build.added_records(a, b, 'f', DECLARATION, LOCKED)
        self.assertEqual(rule['added_modules'], [dict(name=NEW, fnv1a=f'{NEW}_hash', bytes=100, ids=[30], count=1)])
        self.assertEqual(ids, {30: ('module', NEW), 31: ('resolve', SHARED_A), 32: ('resolve', SHARED_B)})
        for b in (inventory({10: MODULE}, {}), inventory({10: MODULE, 30: NEW}, {31: (30, SHARED_A)}, {30: LOCKED}),
                  inventory({10: MODULE, 30: NEW, 40: NEW}, {31: (30, SHARED_A), 32: (30, SHARED_B)}, {30: LOCKED, 40: LOCKED})):
            with self.assertRaises(Invalid):
                build.added_records(a, b, 'f', DECLARATION, LOCKED)

    def test_every_added_module_load_carries_the_head_locked_payload(self):
        # Review C3 r1: the spec declared whatever bytes the capture loaded.
        a = inventory({10: MODULE, 20: MODULE}, {})
        resolves = {31: (30, SHARED_A), 32: (30, SHARED_B), 41: (40, SHARED_A), 42: (40, SHARED_B)}
        rule, _ = build.added_records(a, inventory({10: MODULE, 20: MODULE, 30: NEW, 40: NEW}, resolves,
                                                   {30: LOCKED, 40: LOCKED}), 'f', DECLARATION, LOCKED)
        self.assertEqual(rule['added_modules'][0]['ids'], [30, 40])
        for payloads, message in (({30: LOCKED, 40: 'f' * 64}, r'ID 40\) payload f+ is not the head lock'),
                                  ({30: LOCKED}, r'ID 40\) was loaded without a captured payload')):
            with self.assertRaisesRegex(Invalid, message):
                build.added_records(a, inventory({10: MODULE, 20: MODULE, 30: NEW, 40: NEW}, resolves, payloads),
                                    'f', DECLARATION, LOCKED)


class Records(unittest.TestCase):
    def test_the_added_module_and_resolves_are_declared_with_positions(self):
        lines = ['resolve 299 chop_other_v1', f'module 300 {NEW}', f'resolve 301 {SHARED_A}', f'resolve 302 {SHARED_B}',
                 'configure 299 4096']
        policy = build.additions(lines, ADDED)
        self.assertEqual(policy['added_modules'], [dict(symbol=NEW, ids=[300], positions=[2], count=1)])
        self.assertEqual(sorted((d['symbol'], d['ids'], d['positions']) for d in policy['added_resolves']),
                         [(SHARED_A, [301], [3]), (SHARED_B, [302], [4])])

    def test_other_records_naming_an_added_id_are_refused(self):
        for line in ('configure 301 12345', 'configure 300 12345', 'release 300', 'stream 302', 'event 301',
                     f'resolve 301 {SHARED_B}', f'module 301 {NEW}', f'module 300 {MODULE}',
                     launch('chop_other_v1', stream=301), launch('chop_other_v1', args=[SPAN.format((1 << 32) | 302)] * 3)):
            with self.assertRaises(Invalid, msg=line):
                build.additions([line], ADDED)
        self.assertEqual(build.additions(['configure 299 12345', 'release 5', launch('chop_other_v1')], ADDED), {})

    def test_removed_ids_appear_only_as_their_own_resolves(self):
        ids = {301: OLD_A, 302: OLD_B}
        build.removals_only([f'resolve 301 {OLD_A}', f'resolve 302 {OLD_B}', 'configure 300 4096', launch('chop_other_v1')], ids)
        for line in (f'resolve 301 {OLD_B}', 'configure 301 4096', 'release 302', 'stream 301', 'event 302',
                     launch('chop_other_v1', stream=301), launch('chop_other_v1', args=[SPAN.format((1 << 32) | 302)]),
                     'record 2 301', 'wait 2 302', 'captured 2 301 4', 'replay 2 302', 'release graph 301'):
            with self.assertRaises(Invalid, msg=line):
                build.removals_only([line], ids)

    def test_head_records_naming_a_removed_symbol_are_refused(self):
        for line in (f'resolve 9 {OLD_A}', launch(OLD_B)):
            with self.assertRaises(Invalid, msg=line):
                build.no_removed_symbols([line], 'j', [OLD_A, OLD_B])
        build.no_removed_symbols([f'resolve 9 {SHARED_A}', launch(SHARED_B)], 'j', [OLD_A, OLD_B])

    def test_each_old_symbol_gets_one_exact_launch_rule(self):
        lines = [launch(OLD_A), launch('chop_other_v1'), launch(OLD_A), launch(OLD_B)]
        rules = {r['from_symbol']: r for r in build.launch_rules(lines, DECLARATION)}
        self.assertEqual(rules[OLD_A]['count'], 2)
        self.assertEqual(rules[OLD_A]['to_symbol'], SHARED_A)
        self.assertEqual(rules[OLD_A]['args'][3:], [dict(literal='ScalarU32(512)'), dict(literal='ScalarU32(4)')])
        self.assertEqual(rules[OLD_A]['expect'], dict(block_x=256))
        self.assertEqual(len(rules[OLD_B]['args']), 4)

    def test_launches_outside_the_declared_form_are_refused(self):
        with self.assertRaises(Invalid):
            build.launch_rules([launch(OLD_A, args=[SPAN.format(4294967396)] * 3 + ['ScalarU32(1)'])], DECLARATION)
        with self.assertRaises(Invalid):
            build.launch_rules([launch(OLD_B, flags=0)], DECLARATION)


class GraphNodes(unittest.TestCase):
    NOTES = ['capture 1', launch('chop_a'), launch('chop_b'), 'captured 1 7 2', 'capture 1', launch('chop_c'), 'captured 1 8 1']

    def test_a_graph_whose_launches_change_is_counted(self):
        self.assertEqual(graph_nodes.changed_graphs(self.NOTES, {}), (0, 0))
        self.assertEqual(graph_nodes.changed_graphs(self.NOTES, {1: 0}), (1, 0))
        self.assertEqual(graph_nodes.changed_graphs(self.NOTES, {1: 1, 5: 2}), (1, 0))



class GenerateTests(unittest.TestCase):
    """build.generate end to end on sealed captures (review C3 r1): a change
    that drops a symbol from one locked module and loads a new locked module.
    The spec is written only when the captured added module is the head
    lock's payload."""

    def setUp(self):
        import hashlib
        from unittest.mock import patch
        from testkit import GitFixture
        from provenance import init
        self.tmp = tempfile.TemporaryDirectory(prefix='journal-spec-generate-test-')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        authority = patch('provenance.AUTHORITY', self.root / 'private-authority')
        authority.start()
        self.addCleanup(authority.stop)
        init()
        self.repo = GitFixture(self.root / 'fixture.git')
        self.sha = lambda data: hashlib.sha256(data).hexdigest()
        self.declaration = dict(schema=build.SCHEMA, description='Drop a symbol and load a locked shared module',
                                module='row_toy', removed_symbols=['chop_removed'],
                                added_module=dict(name='row_shared', symbols=['chop_new']))
        self.base = self.capture('base', {CREATE: ['stream 1', 'module 2 row_toy', 'resolve 3 chop_removed', 'resolve 4 chop_keep',
                                                   f'allocate 64 256 -> raw: {GEN + 5}'], STEP: [journal_launch('chop_keep', GEN + 5)]},
                                 [self.lock('row_toy', b'base row module', ['chop_removed', 'chop_keep'])],
                                 {'row_toy': b'base row module'})

    def lock(self, name, payload, symbols):
        return dict(name=name, cubin_sha256=self.sha(payload), entries=[dict(symbol=s) for s in symbols],
                    resources=[dict(symbol=s) for s in symbols])

    def capture(self, label, bodies, locks, payloads):
        from common import fnv
        from testkit import ROOT, checkpoint, create_events, seal_capture, write_device
        root = self.root / label
        (root / 'aux').mkdir(parents=True)
        revision = self.repo.commit(bodies, extra_files={ROOT + 'kernels.lock.d/toy.lock': json.dumps(locks).encode()})
        files = {}
        for name, data in payloads.items():
            (root / 'aux' / f'{name}.bin').write_bytes(data)
            files[name] = (fnv(data), len(data), f'{name}.bin')
        events = create_events(bodies[CREATE], files)
        end = len(events)
        events.extend(['note', line] for line in bodies[STEP])
        write_device(root, 1, events)
        checkpoint(root, CREATE, [(1, events, end)])
        checkpoint(root, STEP, [(1, events, len(events))])
        return revision, seal_capture(root, revision, bodies)

    def head(self, label, loaded):
        bodies = {CREATE: ['stream 1', 'module 2 row_toy', 'resolve 3 chop_keep', 'module 4 row_shared', 'resolve 5 chop_new',
                           f'allocate 64 256 -> raw: {GEN + 6}'], STEP: [journal_launch('chop_keep', GEN + 6)]}
        payloads = {'row_toy': b'head row module', **({'row_shared': loaded} if loaded is not None else {})}
        return self.capture(label, bodies, [self.lock('row_toy', b'head row module', ['chop_keep']),
                                            self.lock('row_shared', b'locked added module', ['chop_new'])], payloads)

    def generate(self, head):
        return build.generate(self.repo.repo, MODEL, self.base[0], head[0], self.base[1], head[1], self.declaration)

    def test_the_locked_added_payload_writes_a_spec_that_proves(self):
        import contextlib, io
        from proof import main
        head = self.head('matching', b'locked added module')
        spec, _ = self.generate(head)
        (self.root / 'spec.json').write_text(json.dumps(spec))
        report = self.root / 'report.json'
        with contextlib.redirect_stdout(io.StringIO()):
            code = main(['--repo', str(self.repo.repo), '--model', MODEL, '--base', self.base[0], '--head', head[0],
                         '--base-capture-index', str(self.base[1]), '--head-capture-index', str(head[1]),
                         '--spec', str(self.root / 'spec.json'), '--report', str(report)])
        value = json.loads(report.read_text())
        self.assertEqual(code, 0, value.get('problems') or value.get('journals'))
        self.assertTrue(value['coverage']['full_history_proof'])

    def test_a_different_added_payload_is_refused(self):
        with self.assertRaisesRegex(Invalid, "row_shared \\(ID 4\\) payload [0-9a-f]{64} is not the head lock's"):
            self.generate(self.head('mismatching', b'different added module'))

    def test_an_added_module_without_a_captured_payload_is_refused(self):
        with self.assertRaisesRegex(Invalid, r'row_shared \(ID 4\) was loaded without a captured payload'):
            self.generate(self.head('missing', None))

if __name__ == '__main__':
    unittest.main()
