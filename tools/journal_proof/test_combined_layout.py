"""Retain a locked module while removing resolves and exact teardown lines."""
import copy
import hashlib
import json
import unittest
from unittest.mock import patch

import layout
import test_layout as toy
from proof import SCHEMA
from testkit import CREATE, STEP


class CombinedLayoutTests(unittest.TestCase):
    def setUp(self):
        self.h = h = toy.LayoutTests()
        h.setUp()
        self.addCleanup(h.doCleanups)
        fixture = h.green_spec()['fixtures'][0]['name']
        self.fixture = fixture
        h.base_locks = h.head_locks = ['core']
        h.base = dict(create=[('stream', 1, 0), ('module', 2, 'core'), ('resolve', 3, 2, toy.BF16),
                              ('configure', 3, 64), ('resolve', 4, 2, toy.ATTN), ('resolve', 5, 2, 'chop_copy_v1'),
                              ('configure', 5, 128), ('id', 6), 'event 6'],
                      step=[toy.launch(1, toy.BF16, 4, ['ScalarU32(64)']), 'record 1 6', 'release', 'release'])
        h.head = dict(create=[('stream', 1, 0), ('module', 2, 'core'), ('resolve', 3, 2, toy.BF16),
                              ('configure', 3, 64), ('resolve', 4, 2, toy.ATTN), ('id', 5), 'event 5'],
                      step=[toy.launch(1, toy.BF16, 4, ['ScalarU32(64)']), 'record 1 5', 'release'])
        _, notes = toy.events_of(h.base['create'])
        self.release_line = len(notes) + 4
        h.decl = dict(schema=layout.SCHEMA, description='Remove one locked symbol and its teardown release.',
                      operations={fixture: [dict(remove=dict(line=self.release_line, text='release'))]})
        self.new_payload = b'core without the unused copy symbol'
        self.new_symbols = [toy.BF16, toy.ATTN]
        self.spec = dict(schema=SCHEMA, fixtures=[dict(name=fixture, create=CREATE, journals=[CREATE, STEP])], journals={},
                         auxiliary_rules={fixture: dict(added_modules=[], added_resolves=[], create_policy={},
                                                        removed_resolves=[dict(symbol='chop_copy_v1', ids=[5], count=1)])},
                         module_replacements=[dict(module='core', renames={}, removed_symbols=['chop_copy_v1'], resolve_order='locked',
                             base=dict(lock=toy.LOCK, sha256=hashlib.sha256(toy.PAYLOADS['core']).hexdigest()),
                             head=dict(lock=toy.LOCK, sha256=hashlib.sha256(self.new_payload).hexdigest()))])

    def run_case(self, expected=0, reason=None):
        h = self.h
        b = h.capture(h.base, h.base_locks)
        with patch.dict(toy.PAYLOADS, core=self.new_payload), patch.dict(toy.SYMBOLS, core=self.new_symbols):
            head = h.capture(h.head, h.head_locks, declaration=h.data())
        spec = copy.deepcopy(self.spec)
        spec['layout'] = dict(path=toy.DECL, sha256=hashlib.sha256(h.data()).hexdigest())
        code, report = h.prove(spec, b, head)
        self.assertEqual(code, expected, json.dumps(report)[:2500])
        if reason:
            self.assertIn(reason, json.dumps(report))
        return report

    def test_retained_module_removal_and_teardown_operations_prove(self):
        report = self.run_case()
        self.assertEqual(report['auxiliary_history_status'][self.fixture], 'COMPARED')
        self.assertEqual(report['journals'][STEP]['counts']['removed_operations'], 1)

    def test_changed_release_position_is_refused(self):
        self.h.decl['operations'][self.fixture][0]['remove']['line'] = self.release_line - 2
        self.run_case(1, 'is not the declared line')

    def test_missing_or_extra_teardown_removal_is_refused(self):
        edits = self.h.decl['operations'][self.fixture]
        edit = edits.pop()
        self.run_case(1, 'unexplained line differences')
        edits.extend([edit, dict(remove=dict(line=self.release_line - 1, text='release'))])
        self.run_case(1, 'unexplained line differences')

    def test_changed_kept_launch_is_refused(self):
        self.h.head['step'][0] = self.h.head['step'][0].replace('ScalarU32(64)', 'ScalarU32(65)')
        self.run_case(1, 'unexplained line differences')

    def test_changed_retained_resolve_is_refused(self):
        self.h.head['create'][4] = ('resolve', 4, 2, 'chop_foreign_v1')
        self.run_case(1, 'resolved symbol is absent from its locked module')

    def test_removed_resolve_needs_its_exact_replacement_and_ids(self):
        removed = self.spec['auxiliary_rules'][self.fixture]['removed_resolves'][0]
        removed['ids'] = [4]
        self.run_case(1, 'removed resolve')
        removed['ids'] = [5]
        self.spec['module_replacements'] = []
        self.run_case(1, 'not owned by a removed module')

    def test_wrong_payload_still_fails_lock_authentication(self):
        self.spec['module_replacements'][0]['head']['sha256'] = '0' * 64
        self.run_case(2, 'declared payload digest does not match revision lock')

    def test_combination_keeps_graph_piece_and_resource_policies_closed(self):
        self.spec['graph_piece_policy'] = {}
        self.run_case(2, 'not a graph piece policy')
        self.spec.pop('graph_piece_policy')
        self.h.decl['handles'] = {self.fixture: {'added': []}}
        self.run_case(2, 'module replacements require an operations-only layout')


if __name__ == '__main__':
    unittest.main()
