#!/usr/bin/env python3
"""The differential mutation harness (mutations.py) over toy devices.

A toy two-rank device runs four collectives over per-rank inboxes, scratch
and flag buffers. Its head adds a zeroed F32 inbox per rank that replaces the
inboxes in the reduce's peer array. The harness runs on that instance-create
pair and on plan views of the device under three synthesized declarations:
an exclusive collective (only its peer array names its buffers), a shared one
(its buffers also appear in another argument) and two collectives whose
inboxes two different added resources replace. Every unsafe mutation must be
refused and every control proven; with any one write rule weakened, the
harness must report escapes of that class. The same harness runs on real
captures through `mutations.py pair` and `mutations.py overlay`."""
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import layout  # noqa: E402
import mutations  # noqa: E402
from mutations import CONFIG, GEN, Case, span  # noqa: E402

CREATE, STEP = 'toy-create.txt', 'toy-step.txt'
REDUCE, GATHER, PAIR, LATE, COMPUTE = 'chop_toy_reduce_v1', 'chop_toy_gather_v1', 'chop_toy_pair_v1', 'chop_toy_late_v1', 'chop_toy_compute_v1'
BUFFERS = (('data', 64), ('inbox', 128), ('scratch', 256), ('flags', 32))
REAL_READS_ONLY = layout.reads_only


def buffer(raw, length):
    return f'DeviceBuffer({span(raw, length)})'


def peers(raws, length):
    return 'DevicePointerArray([' + ', '.join(f'Some({span(r, length)})' for r in raws) + '])'


def launch(stream, symbol, args):
    return f'launch {stream} {symbol} {CONFIG} [' + ', '.join(args) + ']'


def toy_device(head=False, steps=2):
    """FakeDevice's records for the toy device: streams 1 and 2 of ranks 0 and
    1, each rank's buffers, the instance create zeroing the inboxes (and, in
    the head, the added F32 inboxes), then `steps` steps of five launches per
    rank. Returns (events, create note count)."""
    events, slots, ident = [], {}, 0

    def note(line):
        events.append(['note', line])

    def issue():
        nonlocal ident
        ident += 1
        events.append(['id', ident])
        return ident
    for rank in (0, 1):
        stream = issue()
        note(f'stream {stream}')
        events.append(['stream', stream, rank])
    sizes = list(BUFFERS) + ([('f32', 512)] if head else [])
    for name, size in sizes:
        for rank in (0, 1):
            slots[(name, rank)] = GEN + issue()
            note(f'allocate {size} 256')
            events.append(['allocation', slots[(name, rank)], rank, size, 256, GEN + rank + 1])
    for rank in (0, 1):
        note(f'zero {rank + 1} {span(slots[("inbox", rank)], 128)}')
    if head:
        for rank in (0, 1):
            note(f'zero {rank + 1} {span(slots[("f32", rank)], 512)}')
    note('synchronize 1')
    note('synchronize 2')
    created = sum(e[0] == 'note' for e in events)
    events.append(['created', 0, 0, len(events), created])

    def ranks(name):
        return [slots[(name, r)] for r in (0, 1)]
    inboxes = peers(ranks('f32'), 512) if head else peers(ranks('inbox'), 128)
    for _ in range(steps):
        for rank in (0, 1):
            stream, data = rank + 1, buffer(slots[('data', rank)], 64)
            note(launch(stream, COMPUTE, [data]))
            note(launch(stream, REDUCE, [data, inboxes, 'ScalarU32(64)']))
            note(launch(stream, GATHER, [buffer(slots[('scratch', rank)], 256), peers(ranks('scratch'), 256)]))
            note(launch(stream, PAIR, [data, peers(ranks('flags'), 32)]))
            note(launch(stream, LATE, [data, peers(ranks('inbox'), 128)]))
    return events, created


def side(events, created, host='toy/device-1.jsonl', test='toy::test'):
    """The auxiliary capture and bodies of one revision: a create journal of
    the instance create and a step journal of everything after it."""
    notes = [e[1] for e in events if e[0] == 'note']
    marker = next(e for e in events if e[0] == 'created')
    create = dict(kind='instance-create', start_event=0, end_event=marker[3], start_note=0, end_note=created, path='', sha256='')
    bindings = {CREATE: dict(host=host, test=test, records=marker[3] + 1, positions=list(range(created)), create=create,
                             create_lines=notes[:created]),
                STEP: dict(host=host, test=test, records=len(events), positions=list(range(created, len(notes))), create=create,
                           create_lines=notes[:created])}
    aux = dict(manifest=dict(bindings=bindings, unrecoverable={}, hosts={}, text={}), hosts={host: dict(events=events, notes=notes, test=test)})
    return aux, {CREATE: notes[:created], STEP: notes[created:]}


def instance_case():
    """The toy change: the reduce's inboxes become zeroed F32 inboxes."""
    decl = {'schema': layout.SCHEMA, 'description': 'Toy reduce over zeroed F32 inboxes.',
            'resources': {'added': [dict(name='comm.f32_inbox', bytes=512, alignment=256, per_rank=1, lifetime='instance', init='zero',
                                         alongside=dict(bytes=128, alignment=256))]},
            'collectives': [dict(name='reduce, F32 inboxes', from_symbol=REDUCE, to_symbol=REDUCE, ranks=2, peers={'1': 'comm.f32_inbox'})]}
    base, head = side(*toy_device()), side(*toy_device(head=True))
    head_events = head[0]['hosts']['toy/device-1.jsonl']['events']
    added = [e for e in head_events if e[0] == 'allocation' and e[3] == 512]
    rule = dict(added_resolves=[], added_modules=[], create_policy={},
                added_allocations=[dict(resource='comm.f32_inbox', head_raw=e[1], rank=e[2]) for e in added])
    return Case('toy instance create', [CREATE, STEP], [base[1], head[1]], [base[0], head[0]], rule,
                (layout.parse(json.dumps(decl).encode()), {}))


def overlay_cases():
    events, _ = toy_device()
    args = (events, len(events), 'toy-plan.txt', 'toy::test', 'plan-view/toy', 'toy')
    return [mutations.overlay(*args, changes=[(PAIR, 1, 'mutation.flags')]),
            mutations.overlay(*args, changes=[(GATHER, 1, 'mutation.scratch')]),
            mutations.overlay(*args, changes=[(REDUCE, 1, 'mutation.inbox'), (LATE, 1, 'mutation.late_inbox')])]


class MutationHarnessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = [instance_case(), *overlay_cases()]
        cls.report = mutations.run(cls.cases)

    def rows(self, report=None, **match):
        return [r for r in (report or self.report)['mutations'] if all(r[k] == v for k, v in match.items())]

    def test_the_toy_cases_prove_before_mutation(self):
        self.assertEqual([c['baseline'] for c in self.report['cases']], ['PROVEN'] * 4, self.report['cases'])
        kinds = [(c['kind'], c['plan_only']) for c in self.report['cases']]
        self.assertEqual(kinds, [('instance-create', False)] + [('direct-allocation-prefix', True)] * 3)

    def test_the_automatic_overlays_choose_exclusive_and_shared_collectives(self):
        events, _ = toy_device()
        args = (events, len(events), 'toy-plan.txt', 'toy::test', 'plan-view/toy', 'toy')
        self.assertIsNone(mutations.overlay(*args, choice='exclusive', max_launches=1))
        self.assertIn(f'{PAIR} argument 1', mutations.overlay(*args, choice='exclusive').label)
        self.assertIn(f'{GATHER} argument 1', mutations.overlay(*args, choice='shared').label)

    def test_every_unsafe_mutation_is_refused_and_every_control_proven(self):
        summary = self.report['summary']
        escapes = [(r['case'], r['context'], r['mutation'], r['why']) for r in self.rows(outcome='escape')]
        false = [(r['case'], r['context'], r['mutation'], r['reason']) for r in self.rows(outcome='false refusal')]
        self.assertEqual(escapes, [])
        self.assertEqual(false, [])
        refused = {r['cls'] for r in self.rows(expected='refuse', proven=False)}
        self.assertLessEqual({*mutations.WRITES, mutations.EXECUTED, mutations.SITE_ARGUMENT, 'zero-deleted', 'zero-after-use',
                              'zero-other-stream'}, refused)
        proven = {r['cls'] for r in self.rows(expected='prove', proven=True)}
        self.assertLessEqual({*mutations.CONTROLS, mutations.PLAN_RECORD, 'zero-earliest'}, proven)
        self.assertEqual(summary['escapes'], 0)

    def test_writes_before_the_collective_are_refused_and_after_it_proven(self):
        # On the exclusive plan view made executing, each write is refused
        # before the first site and proven after the last one.
        flags = [r for r in self.rows(context='executing') if r['mutation'].startswith('mutation.flags ')]
        for cls in mutations.WRITES:
            with self.subTest(cls):
                before = [r for r in flags if r['mutation'].endswith(f': {cls} before')]
                after = [r for r in flags if r['mutation'].endswith(f': {cls} after')]
                self.assertEqual([(r['expected'], r['proven']) for r in before], [('refuse', False)])
                self.assertEqual([(r['expected'], r['proven']) for r in after], [('prove', True)])

    def test_an_executed_launch_makes_a_plan_device_executing(self):
        # The plan records of the shared buffers precede the reduce: plan
        # records until one launch executes, then writes.
        outcomes = {}
        for r in self.rows(cls=mutations.EXECUTED, context='as recorded'):
            outcomes.setdefault(r['mutation'].split(' ')[0], set()).add((r['expected'], r['proven']))
        self.assertEqual(outcomes, {'mutation.flags': {('prove', True)}, 'mutation.scratch': {('refuse', False)},
                                    'mutation.inbox': {('refuse', False)}, 'mutation.late_inbox': {('refuse', False)}})

    def test_an_earlier_site_that_writes_the_buffer_elsewhere_is_refused(self):
        rows = self.rows(context='executing', op='argument')
        self.assertTrue(any(r['mutation'].endswith('an earlier site also names it in another argument') and r['expected'] == 'refuse'
                            for r in rows))
        self.assertEqual([r['mutation'] for r in rows if r['proven'] != (r['expected'] == 'prove')], [])

    def test_a_buffer_another_added_buffer_replaced_earlier_is_refused(self):
        # The late collective reads an inbox the reduce, now over another
        # added buffer, wrote: the late inbox does not keep its base state.
        case = next(c for c in self.report['cases'] if 'late_inbox' in c['case'])
        rows = [r for r in self.rows(context='executing') if r['case'] == case['case']]
        self.assertTrue(rows)
        self.assertEqual({(r['expected'], r['proven']) for r in rows}, {('refuse', False)})

    def test_each_weakened_rule_lets_its_class_escape(self):
        for cls in sorted(mutations.WEAKENED):
            with self.subTest(cls):
                report = mutations.run(self.cases, weaken=cls)
                self.assertIn(cls, report['summary']['escaped_classes'], report['summary'])
                self.assertIs(layout.reads_only, REAL_READS_ONLY)

    def test_the_oracle_reads_the_journal(self):
        raw, base, x = GEN + 20, GEN + 9, GEN + 8
        use = launch(1, REDUCE, [peers([raw], 128)])
        site = launch(1, REDUCE, [peers([base], 128)])
        for line, verdict in ((f'download 1 {span(base, 128)}', 'prove'), (f'readback 1 {span(base, 128)} 128', 'prove'),
                              (f'copy 1 {span(base, 64)} {span(x, 64)}', 'prove'), (f'copy 1 {span(x, 64)} {span(base, 64)}', 'refuse'),
                              (f'copy 1 {span(base, 64)} {span(base, 64, 64)}', 'refuse'),
                              (f'peer copy 1 0 {span(base, 64)} 1 {span(x, 64)}', 'prove'),
                              (f'peer copy 1 0 {span(x, 64)} 1 {span(base, 64)}', 'refuse'), (f'zero 1 {span(base, 128)}', 'refuse'),
                              (launch(1, COMPUTE, [buffer(base, 128)]), 'refuse'), (f'zero 1 {span(x, 128)}', 'prove'),
                              (f'step x view {span(base, 128)}', 'refuse')):
            with self.subTest(line[:40]):
                notes_a, notes_b = ['allocate 128 256', line, site], ['allocate 128 256', line, use]
                self.assertEqual(mutations.oracle(notes_a, notes_b, {1: 1, 2: 2}, [(raw, base, 2)], {base: base, x: x})[0], verdict)
        plan = launch('r0 Reduce', REDUCE, [peers([base], 128)])
        notes = ['allocate 128 256', launch('r1 Write', COMPUTE, [buffer(base, 128)]), plan]
        self.assertEqual(mutations.oracle(notes, notes[:2] + [plan.replace(str(base), str(raw))], {1: 1, 2: 2}, [(raw, base, 2)],
                                          {base: base})[0], 'prove')


if __name__ == '__main__':
    unittest.main()
