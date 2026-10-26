"""Generate reviewable operations through authenticated ordered maps."""
import json
import unittest

from common import Invalid
from engine import Unproven
import test_activations
import test_layout
from test_layout import F32, launch
from testkit import MODEL
import operation_edits


class OperationEditsTests(unittest.TestCase):
    def setUp(self):
        self.activation = a = test_activations.ActivationTests()
        a.setUp()
        self.addCleanup(a.doCleanups)
        self.h = a.h

    def generate(self):
        h = self.h
        b, head = h.captures()
        return operation_edits.generate(h.git.repo, MODEL, b[0], head[0], b[1], head[1], h.data())

    def prove_generated(self, generated):
        self.h.decl = generated
        self.activation.run_case()

    def test_module_activation_maps_retained_lines_and_declares_only_new_uses(self):
        line = launch(1, F32, 4, ['ScalarU32(64)'])
        self.h.head['step'].append(line)
        generated = self.generate()
        _, notes = test_layout.events_of(self.h.head['create'])
        self.assertEqual(generated['operations'], {self.activation.fixture: [dict(add=dict(lines=[len(notes) + 2], text=line))]})
        self.prove_generated(generated)
        self.h.decl.pop('operations')
        self.activation.run_case(1, 'unexplained line differences')

    def test_removed_function_use_becomes_an_exact_removal(self):
        self.h.head['step'].append('configure 3 128')
        # The configure belongs in the exact resolve records, even outside
        # create. It cannot be hidden as an operation edit.
        with self.assertRaisesRegex(Unproven, 'creator notes'):
            self.generate()
        self.h.head['step'][-1] = launch(1, F32, 4, ['ScalarU32(64)'])
        self.activation.reverse()
        generated = self.generate()
        self.assertEqual(len(generated['operations'][self.activation.fixture]), 1)
        self.assertIn('remove', generated['operations'][self.activation.fixture][0])
        self.prove_generated(generated)

    def test_positional_event_map_is_derived_before_diffing(self):
        h = self.h
        fixture = self.activation.fixture
        h.base = dict(create=[('id', 1), 'event 1'], step=['record 90 1'])
        h.head = dict(create=[('id', 1), 'event 1', ('id', 2), 'event 2'], step=['wait 90 1', 'record 90 2'])
        h.decl = dict(schema=test_layout.layout.SCHEMA, description='One new event.',
                      handles={fixture: {'added': [dict(name='new', kind='event', at=1,
                                                       records=[['id', '@id'], ['note', 'event @id']])]}})
        generated = self.generate()
        self.assertEqual(generated['operations'], {fixture: [dict(add=dict(lines=[3], text='wait 90 1'))]})
        self.prove_generated(generated)

    def test_automatic_resource_edits_are_not_duplicated(self):
        h = test_layout.LayoutTests()
        h.setUp()
        self.addCleanup(h.doCleanups)
        self.h = h
        generated = self.generate()
        self.assertNotIn('operations', generated)
        h.decl = generated
        h.run_case(0)

    def test_regeneration_replaces_stale_operations(self):
        self.h.decl['operations'] = {self.activation.fixture: [dict(add=dict(line=999, text='synchronize 90'))]}
        generated = self.generate()
        self.assertNotIn('operations', generated)
        self.prove_generated(generated)

    def test_wrong_position_is_not_inferred(self):
        self.h.decl['handles'][self.activation.fixture]['added'][0]['at'] += 1
        with self.assertRaisesRegex(Unproven, 'no issuance at declared trace position'):
            self.generate()

    def test_unknown_fixture_is_refused(self):
        self.h.decl['module_activations']['absent'] = {'added': ['comm_f32']}
        with self.assertRaisesRegex(Invalid, 'layout names unknown fixtures'):
            self.generate()

    def test_cli_writes_a_complete_declaration_for_review(self):
        h = self.h
        line = 'synchronize 1'
        h.head['step'].extend([line, line])
        b, head = h.captures()
        source, target = h.root / 'input.json', h.root / 'output.json'
        source.write_bytes(h.data())
        import contextlib
        import io
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = operation_edits.main(['--repo', str(h.git.repo), '--model', MODEL, '--base', b[0], '--head', head[0],
                                         '--base-index', str(b[1]), '--head-index', str(head[1]),
                                         '--declaration', str(source), '--out', str(target)])
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(output.getvalue())['per_fixture'], {self.activation.fixture: 2})
        self.assertEqual(json.loads(output.getvalue())['operations'], 2)
        generated = json.loads(target.read_text())
        self.assertEqual(generated['module_activations'], h.decl['module_activations'])
        self.assertEqual(generated['operations'][self.activation.fixture][0]['add']['text'], line)
        self.assertEqual(len(generated['operations'][self.activation.fixture]), 1)
        self.assertEqual(len(generated['operations'][self.activation.fixture][0]['add']['lines']), 2)
        self.prove_generated(generated)


if __name__ == '__main__':
    unittest.main()
