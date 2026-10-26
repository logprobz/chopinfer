"""Compact operations name every ordinal and keep expanded-edit semantics."""
import copy
import json
import unittest

from common import Invalid
import handles
import layout
import test_layout as toy
import operation_edits


class CompactOperationsTests(unittest.TestCase):
    def setUp(self):
        self.h = h = toy.LayoutTests()
        h.setUp()
        self.addCleanup(h.doCleanups)
        self.fixture = h.green_spec()['fixtures'][0]['name']
        h.base_locks = h.head_locks = []
        h.base = dict(create=[('stream', 1, 0)], step=['synchronize 1', 'synchronize 1', 'synchronize 1'])
        h.head = dict(create=[('stream', 1, 0)], step=['synchronize 2', 'synchronize 1', 'synchronize 2'])
        h.decl = dict(schema=layout.SCHEMA, description='Replace two exactly named occurrences.',
                      operations={self.fixture: [dict(remove=dict(text='synchronize 1', lines=[2, 4])),
                                                 dict(add=dict(text='synchronize 2', lines=[2, 4]))]})

    def test_compact_and_expanded_have_identical_proof_results(self):
        compact = self.h.run_case(0)
        edits = self.h.decl['operations'][self.fixture]
        self.h.decl['operations'][self.fixture] = handles.expand_operations(edits)
        expanded = self.h.run_case(0)
        self.assertEqual(compact['counts'], expanded['counts'])
        self.assertEqual(compact['fixtures'], expanded['fixtures'])
        for name in compact['journals']:
            for field in ('verdict', 'counts', 'reasons'):
                self.assertEqual(compact['journals'][name][field], expanded['journals'][name][field])

    def test_each_missing_ordinal_or_wrong_text_is_refused(self):
        original = copy.deepcopy(self.h.decl)
        for side in ('add', 'remove'):
            for mutation in ('missing', 'position', 'text', 'absent'):
                with self.subTest(side=side, mutation=mutation):
                    self.h.decl = copy.deepcopy(original)
                    edit = next(e[side] for e in self.h.decl['operations'][self.fixture] if side in e)
                    if mutation == 'missing':
                        edit['lines'].pop()
                    elif mutation == 'position':
                        edit['lines'][0] = 1
                    elif mutation == 'text':
                        edit['text'] = 'synchronize 3'
                    else:
                        edit['lines'].append(99)
                    report = self.h.run_case(1)
                    self.assertIn('unexplained line differences' if mutation == 'missing' else 'is not the declared line',
                                  json.dumps(report))

    def test_duplicate_mixed_ordinals_and_invalid_lists_are_refused(self):
        original = copy.deepcopy(self.h.decl)
        for lines in ([], [2, 2], [0], [-1], [True], ['2'], 2, [None]):
            with self.subTest(lines=lines):
                self.h.decl = copy.deepcopy(original)
                self.h.decl['operations'][self.fixture][0]['remove']['lines'] = lines
                with self.assertRaises(Invalid):
                    layout.parse(self.h.data())
        self.h.decl = original
        self.h.decl['operations'][self.fixture].append(dict(remove=dict(line=2, text='synchronize 1')))
        with self.assertRaisesRegex(Invalid, 'repeats line 2'):
            layout.parse(self.h.data())
        self.h.decl['operations'][self.fixture].pop()
        self.h.decl['operations'][self.fixture][0]['remove']['line'] = 2
        with self.assertRaisesRegex(Invalid, 'compact operation'):
            layout.parse(self.h.data())

    def test_compact_creator_edits_are_still_refused(self):
        self.h.decl['operations'][self.fixture][0]['remove']['text'] = 'event 1'
        with self.assertRaisesRegex(Invalid, 'creator records use resource declarations'):
            layout.parse(self.h.data())

    def test_compact_automatic_zero_overlap_is_still_refused(self):
        h = toy.LayoutTests()
        h.setUp()
        self.addCleanup(h.doCleanups)
        fixture = h.green_spec()['fixtures'][0]['name']
        _, notes = toy.events_of(h.head['create'])
        zero = f'zero 1 {toy.bare(toy.GEN + 13, 256)}'
        h.decl['operations'] = {fixture: [dict(add=dict(text=zero, lines=[notes.index(zero) + 1]))]}
        report = h.run_case(1)
        self.assertIn('operation overlaps another declaration', json.dumps(report))

    def test_compaction_preserves_every_explicit_ordinal(self):
        expanded = [dict(remove=dict(line=i, text='release')) for i in (2, 7, 10)]
        expanded += [dict(add=dict(line=4, text='synchronize 1'))]
        compact = operation_edits.compact(expanded)
        self.assertEqual(compact, [dict(remove=dict(text='release', lines=[2, 7, 10])),
                                   dict(add=dict(text='synchronize 1', lines=[4]))])
        self.assertEqual(handles.expand_operations(compact), expanded)


if __name__ == '__main__':
    unittest.main()
