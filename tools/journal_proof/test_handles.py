"""Declared issued resources: exact creators, positions and retained lines."""
import copy
import json
import unittest

import auxproof
from common import Invalid
from engine import Unproven
import layout

GEN = 1 << 32
FIXTURE = dict(name='host', create='step.txt', journals=['step.txt'])


def creation(kind, ident=1):
    rows = [['id', ident]]
    if kind == 'allocation':
        rows += [['note', 'allocate 64 256'], ['allocation', GEN + ident, 0, 64, 256, GEN + 90]]
    elif kind == 'stream':
        rows += [['note', f'stream {ident}'], ['stream', ident, 0]]
    elif kind in ('event', 'mapped-gather'):
        rows += [['note', f'event {ident}']]
        if kind == 'mapped-gather':
            rows += [['note', f'async mapped create {ident}']]
    elif kind == 'graph':
        rows += [['note', f'captured 90 {ident} 0']]
    return rows


def case(kind, side='added'):
    small = [['id', 1], ['note', 'event 1'], ['note', 'record 90 1']]
    prefix = [['note', 'capture 90']] if kind == 'graph' else []
    big = prefix + creation(kind) + [['id', 2], ['note', 'event 2'], ['note', 'record 90 2']]
    template = creation(kind, '@id') if kind not in ('stream', 'allocation') else creation(kind)
    if kind == 'allocation':
        template = [['id', '@id'], ['note', 'allocate 64 256'], ['allocation', '@raw', 0, 64, 256, GEN + 90]]
    if kind == 'stream':
        template = [['id', '@id'], ['note', 'stream @id'], ['stream', '@id', 0]]
    d = dict(schema=layout.SCHEMA, description='One exact issued resource.',
             handles={'host': {side: [dict(name='resource', kind=kind, at=len(prefix) + 1, records=template)]}})
    if prefix:
        d['operations'] = {'host': [{('add' if side == 'added' else 'remove'): dict(line=1, text='capture 90')}]}
    rule = dict(auxproof.EMPTY_RULE, **{side + '_handles': [dict(handle='resource', id=1)]})
    return (small, big, d, rule) if side == 'added' else (big, small, d, rule)


def auxiliary(events):
    notes = [r[1] for r in events if r[0] == 'note']
    create = dict(path='create', start_note=0, end_note=len(notes)-1, start_event=0,
                  end_event=len(events)-1, kind='instance-create')
    binding = dict(host='device', test='test', records=len(events), create=create,
                   create_lines=notes[:-1], positions=[len(notes)-1])
    return dict(manifest=dict(bindings={'step.txt': binding}, unrecoverable={}),
                hosts={'device': dict(events=events)}), {'step.txt': notes[-1:]}


def prove(a, b, d, rule):
    aa, ab = auxiliary(a)
    ba, bb = auxiliary(b)
    return auxproof.prove(ab, bb, FIXTURE, {}, aa, ba, rule, layout=(layout.parse(json.dumps(d)), {}))


class HandleTests(unittest.TestCase):
    def green(self, kind, side='added'):
        args = case(kind, side)
        results, _, detail = prove(*args)
        self.assertEqual(results['step.txt']['verdict'], 'PROVEN', detail)
        self.assertEqual(detail['history']['status'], 'COMPARED')
        return args

    def refused(self, args):
        try:
            result, _, detail = prove(*args)
        except (Invalid, Unproven):
            return
        self.assertEqual(result['step.txt']['verdict'], 'UNPROVEN', detail)

    def test_each_handle_addition_and_removal(self):
        for kind in ('allocation', 'stream', 'event', 'graph', 'mapped-gather', 'opaque'):
            for side in ('added', 'removed'):
                with self.subTest(kind=kind, side=side):
                    self.green(kind, side)

    def test_each_handle_refuses_missing_and_false_declarations(self):
        for kind in ('allocation', 'stream', 'event', 'graph', 'mapped-gather', 'opaque'):
            for side in ('added', 'removed'):
                for mutation in ('undeclared', 'absent', 'wrong-kind', 'wrong-position', 'after-map', 'missing-witness'):
                    with self.subTest(kind=kind, side=side, mutation=mutation):
                        a, b, d, rule = self.green(kind, side)
                        declaration = d['handles']['host'][side][0]
                        if mutation == 'undeclared':
                            d['handles']['host'][side] = []
                            rule[side + '_handles'] = []
                        elif mutation == 'absent':
                            if side == 'added':
                                b = copy.deepcopy(a)
                            else:
                                a = copy.deepcopy(b)
                        elif mutation == 'wrong-kind':
                            declaration['kind'] = 'graph' if kind != 'graph' else 'event'
                        elif mutation == 'wrong-position':
                            declaration['at'] += 1
                        elif mutation == 'after-map':
                            b[-1][1] = b[-1][1].replace('90', '91')
                        else:
                            rule[side + '_handles'] = []
                        self.refused((a, b, d, rule))

    def test_operation_edits_are_exact_and_exhaustive(self):
        a, b, d, rule = self.green('event')
        # The head's changed record is declared in the same exact edit
        # format as text journals, at whole-device note ordinals.
        b[-1][1] = 'wait 90 2'
        d['operations'] = {'host': [dict(remove=dict(line=2, text='record 90 1')),
                                   dict(add=dict(line=3, text='wait 90 2'))]}
        self.assertEqual(prove(a, b, d, rule)[0]['step.txt']['verdict'], 'PROVEN')
        for mutation in ('undeclared', 'absent', 'extra', 'wrong-position'):
            with self.subTest(mutation=mutation):
                aa, bb, dd, rr = copy.deepcopy((a, b, d, rule))
                if mutation == 'undeclared':
                    dd.pop('operations')
                elif mutation == 'absent':
                    bb[-1][1] = 'record 90 2'
                elif mutation == 'extra':
                    bb.append(['note', 'wait 90 2'])
                else:
                    dd['operations']['host'][1]['add']['line'] = 2
                self.refused((aa, bb, dd, rr))

    def test_opaque_requires_actual_issuance(self):
        a, b, d, rule = self.green('opaque')
        b.pop(0)
        self.refused((a, b, d, rule))

    def test_mapped_gather_requires_both_proxy_and_service_creator(self):
        a, b, d, rule = self.green('mapped-gather')
        b.pop(1)
        self.refused((a, b, d, rule))

    def test_declared_creator_cannot_hide_in_operation_edits(self):
        a, b, d, rule = case('event')
        d['operations'] = {'host': [dict(add=dict(line=1, text='event 1'))]}
        self.refused((a, b, d, rule))


if __name__ == '__main__':
    unittest.main()
