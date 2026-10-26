"""Sealed proofs for fixture-scoped activation of unchanged locked modules."""
import copy
import hashlib
import json
import unittest

from common import Invalid
import layout
import test_layout as toy
from test_layout import PAYLOADS, F32, LOCK, events_of, lock
from testkit import ROOT


class ActivationTests(unittest.TestCase):
    def setUp(self):
        self.h = h = toy.LayoutTests()
        h.setUp()
        self.addCleanup(h.doCleanups)
        self.fixture = h.green_spec()['fixtures'][0]['name']
        h.base = dict(create=[('stream', 1, 0), ('id', 2), 'event 2'], step=['record 1 2'])
        h.head = dict(create=[('stream', 1, 0), ('module', 2, 'comm_f32'), ('resolve', 3, 2, F32),
                              ('configure', 3, 64), ('id', 4), 'event 4'], step=['record 1 4'])
        h.base_locks = h.head_locks = ['comm_f32']
        events, _ = events_of(h.head['create'])
        declarations = []
        for ident, kind in ((2, 'module'), (3, 'resolve')):
            at = events.index(['id', ident])
            end = next(i for i in range(at + 1, len(events)) if events[i][0] == 'id')
            records = copy.deepcopy(events[at:end])
            records[0][1] = '@id'
            for row in records[1:]:
                if row[0] == 'note':
                    row[1] = row[1].replace(f' {ident} ', ' @id ')
                else:
                    row[1] = '@id'
                if row[0] == 'module':
                    row[5] = hashlib.sha256(PAYLOADS['comm_f32']).hexdigest()
            declarations.append(dict(name=kind, kind=kind, at=at + 1, records=records))
        h.decl = dict(schema=layout.SCHEMA, description='Activate a locked module in one fixture.',
                      module_activations={self.fixture: {'added': ['comm_f32']}},
                      handles={self.fixture: {'added': declarations}})

    def run_case(self, code=0, reason=None, spec=None):
        h = self.h
        b, head = h.captures()
        spec = h.generate(b, head) if spec is None else copy.deepcopy(spec)
        spec['layout']['sha256'] = hashlib.sha256(h.data()).hexdigest()
        actual, report = h.prove(spec, b, head)
        self.assertEqual(actual, code, report)
        if reason:
            self.assertIn(reason, json.dumps(report))
        return spec, report

    def reverse(self):
        h = self.h
        h.base, h.head = h.head, h.base
        for key in ('module_activations', 'handles'):
            h.decl[key][self.fixture]['removed'] = h.decl[key][self.fixture].pop('added')

    def test_activation_and_deactivation_with_ordered_handle_shifts(self):
        for reverse in (False, True):
            if reverse:
                self.reverse()
            with self.subTest(reverse=reverse):
                spec, report = self.run_case()
                self.assertEqual(report['status'], 'PROVEN')
                b, h = self.h.captures()
                code, _ = self.h.prove(spec, b, h, '--require', 'identity')
                self.assertEqual(code, 1)

    def test_changed_lock_payload_is_refused(self):
        spec, _ = self.run_case()
        self.h.head_shas = {'comm_f32': '0' * 64}
        self.run_case(2, 'byte-identical lock entries', spec)

    def test_changed_lock_metadata_or_bytes_is_refused(self):
        h = self.h
        data = lock(['comm_f32'])
        base = h.git.commit({}, extra_files={LOCK: data})
        for changed in (data.replace(b'"entries":', b'"extra": 1, "entries":'),
                        data.replace(b'"entries":', b'"entries" :')):
            head = h.git.commit({}, extra_files={LOCK: changed})
            with self.assertRaisesRegex(Invalid, 'byte-identical lock entries'):
                layout.authenticate(h.git.repo, base, head, ROOT, layout.parse(h.data()))

    def test_wrong_position_is_refused_for_each_kind_and_side(self):
        for reverse in (False, True):
            if reverse:
                self.reverse()
            side = 'removed' if reverse else 'added'
            spec, _ = self.run_case()
            for entry in self.h.decl['handles'][self.fixture][side]:
                with self.subTest(side=side, kind=entry['kind']):
                    entry['at'] += 1
                    self.run_case(1, 'issuance is at the wrong position', spec)
                    entry['at'] -= 1

    def test_undeclared_activation_is_refused(self):
        spec, _ = self.run_case()
        self.h.decl.pop('module_activations')
        self.run_case(1, 'exactly their positional load witnesses', spec)

    def test_each_missing_positional_witness_is_refused(self):
        spec, _ = self.run_case()
        entries = self.h.decl['handles'][self.fixture]['added']
        for entry in list(entries):
            with self.subTest(kind=entry['kind']):
                entries.remove(entry)
                changed = copy.deepcopy(spec)
                changed['auxiliary_rules'][self.fixture]['added_handles'] = [
                    e for e in changed['auxiliary_rules'][self.fixture]['added_handles'] if e['handle'] != entry['name']]
                self.run_case(1, 'positional ' + ('load' if entry['kind'] == 'module' else 'resolve') + ' witnesses', changed)
                entries.append(entry)

    def test_declaration_in_an_unchanged_fixture_is_refused(self):
        self.h.base = copy.deepcopy(self.h.head)
        self.h.decl.pop('handles')
        self.run_case(1, 'module activation does not change this fixture')

    def test_declared_but_absent_module_is_refused(self):
        self.h.head = copy.deepcopy(self.h.base)
        self.h.decl.pop('handles')
        self.run_case(1, 'declared module activation is absent')

    def test_creator_metadata_is_exact(self):
        spec, _ = self.run_case()
        self.h.decl['handles'][self.fixture]['added'][1]['records'][2][2] = 1
        self.run_case(1, 'creator trace records differ', spec)

    def test_captured_payload_must_match_the_unchanged_lock(self):
        self.h.base_locks = self.h.head_locks = ['comm_f32']
        spec, _ = self.run_case()
        # Both lock entries change together; the captured payload does not.
        from unittest.mock import patch
        with patch('test_layout.lock', side_effect=lambda names, shas=None: lock(names, {'comm_f32': '0' * 64})):
            self.run_case(1, 'activated payload is not the locked SHA-256', spec)

    def test_retained_operation_mismatch_is_refused(self):
        spec, _ = self.run_case()
        self.h.head['step'] = ['record 2 4']
        self.run_case(1, 'unexplained line differences', spec)

    def test_extra_configure_is_not_implicitly_authorized(self):
        spec, _ = self.run_case()
        self.h.head['create'].insert(4, ('configure', 3, 128))
        self.run_case(1, "records must contain exactly this handle's creator notes", spec)

    def test_deactivated_symbol_cannot_keep_a_launch(self):
        self.reverse()
        line = toy.launch(1, F32, 4, ['ScalarU32(64)'])
        self.h.base['step'] = self.h.head['step'] = [line]
        self.run_case(1, 'launch uses a deactivated symbol')

    def test_deactivated_launch_requires_an_exact_removal_or_substitution(self):
        self.reverse()
        line = toy.launch(1, F32, 4, ['ScalarU32(64)'])
        self.h.base['step'].append(line)
        _, created = events_of(self.h.base['create'])
        self.h.decl['operations'] = {self.fixture: [dict(remove=dict(line=len(created) + 2, text=line))]}
        self.run_case()
        # An explicit substitution accounts for the formerly implicit use.
        self.h.head['step'].append('synchronize 1')
        _, created = events_of(self.h.head['create'])
        self.h.decl['operations'][self.fixture].append(dict(add=dict(line=len(created) + 2, text='synchronize 1')))
        self.run_case()

    def test_activation_cannot_swap_or_share_a_symbol_provider(self):
        from unittest.mock import patch
        for swap in (False, True):
            with self.subTest(swap=swap), patch.dict(toy.PAYLOADS, comm_alt=b'other cubin'), \
                    patch.dict(toy.SYMBOLS, comm_alt=[F32]):
                h = self.h
                h.base_locks = h.head_locks = ['comm_f32', 'comm_alt']
                h.base = dict(create=[('stream', 1, 0), ('module', 2, 'comm_alt'), ('resolve', 3, 2, F32),
                                      ('id', 4), 'event 4'], step=[toy.launch(1, F32, 4, [])])
                h.head = dict(create=[('stream', 1, 0), ('module', 2, 'comm_f32'), ('resolve', 3, 2, F32),
                                      ('configure', 3, 64)], step=list(h.base['step']))
                h.decl['module_activations'] = {self.fixture: {'added': ['comm_f32']}}
                h.decl['handles'][self.fixture].pop('removed', None)
                if swap:
                    h.decl['module_activations'][self.fixture]['removed'] = ['comm_alt']
                    h.decl['handles'][self.fixture]['removed'] = [
                        dict(name='old-load', kind='module', at=4, records=[['id', '@id'],
                             ['module', '@id', 'comm_alt', toy.fnv(toy.PAYLOADS['comm_alt']), len(toy.PAYLOADS['comm_alt']),
                              hashlib.sha256(toy.PAYLOADS['comm_alt']).hexdigest()]]),
                        dict(name='old-resolve', kind='resolve', at=6, records=[['id', '@id'],
                             ['note', f'resolve @id {F32}'], ['resolve', '@id', 2, F32]])]
                    h.head['create'] += [('id', 4), 'event 4']
                else:
                    h.head['create'] += [('module', 4, 'comm_alt'), ('resolve', 5, 4, F32), ('id', 6), 'event 6']
                self.run_case(1, 'also resolved from a module the activation does not change')

    def test_activated_symbol_must_belong_to_its_lock(self):
        spec, _ = self.run_case()
        self.h.head['create'][2] = ('resolve', 3, 2, 'chop_foreign_v1')
        records = self.h.decl['handles'][self.fixture]['added'][1]['records']
        records[1][1] = 'resolve @id chop_foreign_v1'
        records[2][3] = 'chop_foreign_v1'
        self.run_case(1, 'not an entry of its locked module', spec)

    def test_activation_fixture_admission_and_report(self):
        spec, report = self.run_case()
        self.assertEqual(report['layout_declaration']['module_activations'], self.h.decl['module_activations'])
        self.h.decl['module_activations']['unknown-fixture'] = {'added': ['comm_f32']}
        self.run_case(2, 'layout names unknown fixtures', spec)
        self.h.decl['module_activations'].pop('unknown-fixture')
        self.h.text = (['text'], ['text'])
        b, head = self.h.captures()
        spec = self.h.generate(b, head)
        fixture = next(f['name'] for f in spec['fixtures'] if toy.TEXT in f['journals'])
        self.h.decl['module_activations'][fixture] = {'added': ['comm_f32']}
        self.run_case(2, 'position-bound declarations need a compared device fixture', spec)


if __name__ == '__main__':
    unittest.main()
