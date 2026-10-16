#!/usr/bin/env python3
"""Text journals: pinned journals a test checks without a fake device."""
import contextlib
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from auxiliary import LEGACY_TEXT, normalize
from common import Invalid
from engine import Unproven
from proof import SCHEMA, main
from provenance import init
from testkit import CREATE, MODEL, STEP, GitFixture, checkpoint, create_events, fixtures, seal_capture, write_device
from text import compare

ASKS = 'toy-asks.txt'
GAPS = 'toy-gaps.txt'
LAUNCHES = 'toy-launches.txt'
EMPTY = dict(added_resolves=[], added_modules=[], create_policy={})


class CompareTests(unittest.TestCase):
    def test_identical_bodies(self):
        self.assertEqual(compare(['a', 'b'], ['a', 'b'], []), ('IDENTICAL', {}))

    def test_a_difference_without_edits_is_unproven(self):
        with self.assertRaisesRegex(Unproven, 'differs at line 2'):
            compare(['a', 'b'], ['a', 'c'], [])

    def test_declared_edits_prove_a_change(self):
        edits = [{'remove': {'line': 2, 'text': 'b'}}, {'add': {'line': 2, 'text': 'c'}}, {'add': {'line': 4, 'text': 'e'}}]
        self.assertEqual(compare(['a', 'b', 'd'], ['a', 'c', 'd', 'e'], edits),
                         ('PROVEN', {'text_removed_lines': 1, 'text_added_lines': 2}))

    def test_an_edit_must_name_its_exact_text(self):
        with self.assertRaisesRegex(Unproven, 'base line 2 is not the declared text'):
            compare(['a', 'b'], ['a'], [{'remove': {'line': 2, 'text': 'x'}}])

    def test_an_undeclared_difference_beside_edits_is_unproven(self):
        with self.assertRaisesRegex(Unproven, 'beyond its declared edits'):
            compare(['a', 'b', 'c'], ['a', 'x', 'd'], [{'remove': {'line': 2, 'text': 'b'}}, {'add': {'line': 2, 'text': 'x'}}])

    def test_malformed_edits_are_invalid(self):
        for edits in ({'remove': 1}, [{'remove': {'line': 0, 'text': 'a'}}], [{'drop': {'line': 1, 'text': 'a'}}],
                      [{'add': {'line': 1, 'text': 'a'}}, {'add': {'line': 1, 'text': 'b'}}]):
            with self.assertRaises(Invalid):
                compare(['a'], ['a'], edits)

    def test_written_edits_prove_the_change_they_describe(self):
        sys.path.insert(0, str(Path(__file__).resolve().parent / 'specs'))
        from text_edits import edits
        base, head = ['a', 'b', 'c', 'd'], ['a', 'x', 'c', 'd', 'e']
        self.assertEqual(compare(base, head, edits(base, head)),
                         ('PROVEN', {'text_removed_lines': 1, 'text_added_lines': 2}))
        self.assertEqual(edits(base, base), [])

    def test_legacy_manifest_lists_no_candidate_journals_as_text(self):
        manifest = dict(bindings={}, unrecoverable={
            ASKS: dict(reason=LEGACY_TEXT, candidate_devices=[], test='asks'),
            GAPS: dict(reason=LEGACY_TEXT, candidate_devices=[1, 2], test='gaps')})
        value = normalize(manifest)
        self.assertEqual(value['text'], {ASKS: dict(test='asks', devices=None)})
        self.assertEqual(set(value['unrecoverable']), {GAPS})


class TextProofTests(unittest.TestCase):
    """A device fixture (create and step on one device), a text journal of a
    test without devices (asks) and two text journals of a test whose device
    never recorded their lines (gaps, launches)."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='journal-proof-text-test-')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        authority = patch('provenance.AUTHORITY', self.root / 'private-authority')
        authority.start()
        self.addCleanup(authority.stop)
        init()
        self.git = GitFixture(self.root / 'objects.git')
        bodies, _, _ = fixtures()
        texts = {ASKS: ['plan a', 'ask x'], GAPS: ['gap 1'], LAUNCHES: ['launch-claim a', 'launch-claim b']}
        self.base = dict(bodies, **texts)
        self.head = copy.deepcopy(self.base)
        self.spec = dict(schema=SCHEMA, journals={},
                         fixtures=[dict(name='host', create=CREATE, journals=[CREATE, STEP]),
                                   dict(name='asks', create=ASKS, journals=[ASKS]),
                                   dict(name='invariance', create=GAPS, journals=[GAPS, LAUNCHES])],
                         auxiliary_rules={'host': EMPTY, 'asks': EMPTY, 'invariance': EMPTY})
        self.on_device = set()
        self.count = 0

    def capture(self, bodies, on_device=()):
        self.count += 1
        root = self.root / f'capture-{self.count}'
        revision = self.git.commit(bodies)
        device = create_events(bodies[CREATE])
        create_records = len(device)
        device.extend(['note', line] for line in bodies[STEP])
        write_device(root, 1, device)
        checkpoint(root, CREATE, [(1, device, create_records)])
        checkpoint(root, STEP, [(1, device, len(device))])
        checkpoint(root, ASKS, [], test='asks')
        # The invariance test's own device records unrelated work, or, for
        # journals in `on_device`, the journal's lines.
        other = [['id', 1], ['stream', 1, 0], ['note', 'stream 1'], ['created', 0, 0, 3, 1]]
        for name in (GAPS, LAUNCHES):
            if name in on_device:
                other.extend(['note', line] for line in bodies[name])
        write_device(root, 2, other)
        for name in (GAPS, LAUNCHES):
            checkpoint(root, name, [(2, other, len(other))], test='invariance')
        return revision, seal_capture(root, revision, bodies)

    def prove(self, expected, head_on_device=(), extra=()):
        base = self.capture(self.base)
        head = self.capture(self.head, head_on_device)
        (self.root / 'spec.json').write_text(json.dumps(self.spec))
        report = self.root / 'report.json'
        with contextlib.redirect_stdout(io.StringIO()):
            code = main(['--repo', str(self.git.repo), '--model', MODEL, '--base', base[0], '--head', head[0],
                         '--spec', str(self.root / 'spec.json'), '--base-capture-index', str(base[1]),
                         '--head-capture-index', str(head[1]), '--report', str(report), *extra])
        value = json.loads(report.read_text())
        self.assertEqual(code, expected, json.dumps(value.get('problems') or value.get('journals'), indent=1)[:3000])
        return value

    def test_unchanged_text_journals_are_identical_in_a_full_proof(self):
        report = self.prove(0, extra=('--require', 'identity'))
        self.assertEqual({v['verdict'] for v in report['journals'].values()}, {'IDENTICAL'})
        self.assertEqual(report['coverage']['text_fixtures'], ['asks', 'invariance'])
        self.assertEqual(report['coverage']['history_compared'], 1)
        self.assertTrue(report['coverage']['full_history_proof'])
        self.assertEqual(report['auxiliary_history_status'], {'asks': 'TEXT', 'host': 'COMPARED', 'invariance': 'TEXT'})
        self.assertIn('text_journals', report['claim']['compared'])
        self.assertTrue(all(report['identity_checklist'].values()))

    def test_a_changed_text_journal_without_edits_is_unproven(self):
        self.head[GAPS] = ['gap 2']
        report = self.prove(1)
        self.assertEqual(report['journals'][GAPS]['verdict'], 'UNPROVEN')
        self.assertEqual(report['journals'][LAUNCHES]['verdict'], 'IDENTICAL')
        self.assertFalse(report['coverage']['full_history_proof'])

    def test_declared_text_edits_prove_a_change(self):
        self.head[ASKS] = ['plan a', 'ask y']
        self.spec['journals'][ASKS] = dict(text_edits=[{'remove': {'line': 2, 'text': 'ask x'}}, {'add': {'line': 2, 'text': 'ask y'}}])
        report = self.prove(0)
        self.assertEqual(report['journals'][ASKS]['verdict'], 'PROVEN')
        self.assertEqual(report['counts'], {'text_removed_lines': 1, 'text_added_lines': 1})
        self.assertTrue(report['coverage']['full_history_proof'])

    def test_an_empty_text_journal_is_identical_to_itself(self):
        # A journal of zero lines (pinned (0, 0xcbf29ce484222325)) is a text
        # journal even though its test's device recorded other lines.
        self.base[GAPS] = self.head[GAPS] = []
        report = self.prove(0, extra=('--require', 'identity'))
        self.assertEqual(report['journals'][GAPS]['verdict'], 'IDENTICAL')
        self.assertEqual(report['journals'][GAPS]['head_pin']['lines'], 0)
        self.assertTrue(report['coverage']['full_history_proof'])

    def test_text_edits_prove_a_journal_emptied(self):
        # The guard "no gaps remain": every base line removed, the head empty.
        self.base[GAPS] = ['gap 1', 'gap 2', 'gap 3']
        self.head[GAPS] = []
        self.spec['journals'][GAPS] = dict(text_edits=[{'remove': {'line': i, 'text': f'gap {i}'}} for i in (1, 2, 3)])
        report = self.prove(0)
        self.assertEqual(report['journals'][GAPS]['verdict'], 'PROVEN')
        self.assertEqual(report['counts'], {'text_removed_lines': 3, 'text_added_lines': 0})
        self.assertTrue(report['coverage']['full_history_proof'])
        sys.path.insert(0, str(Path(__file__).resolve().parent / 'specs'))
        from text_edits import edits
        self.assertEqual(edits(self.base[GAPS], []), self.spec['journals'][GAPS]['text_edits'])

    def test_an_emptied_journal_without_edits_is_unproven(self):
        self.head[GAPS] = []
        self.assertEqual(self.prove(1)['journals'][GAPS]['verdict'], 'UNPROVEN')

    def test_text_edits_on_a_device_journal_are_invalid(self):
        self.spec['journals'][STEP] = dict(text_edits=[])
        self.prove(2)

    def test_device_policies_on_a_text_journal_are_invalid(self):
        self.spec['journals'][ASKS] = dict(removals=[dict(line=1, symbol='chop_old')])
        self.prove(2)

    def test_a_text_fixture_takes_only_the_empty_auxiliary_rule(self):
        self.spec['auxiliary_rules']['asks'] = dict(EMPTY, create_policy={'state_sizes': []})
        self.prove(2)

    def test_a_journal_that_moves_onto_a_device_is_unproven(self):
        self.spec['fixtures'][2] = dict(name='invariance', create=GAPS, journals=[GAPS, LAUNCHES])
        report = self.prove(1, head_on_device=(GAPS, LAUNCHES))
        self.assertIn('binding kind changed: base text, head device', report['journals'][GAPS]['reasons'][0])

    def test_a_fixture_cannot_join_text_and_device_journals(self):
        self.spec['fixtures'] = [dict(name='host', create=CREATE, journals=[CREATE, STEP, ASKS]),
                                 dict(name='invariance', create=GAPS, journals=[GAPS, LAUNCHES])]
        self.spec['auxiliary_rules'] = {'host': EMPTY, 'invariance': EMPTY}
        report = self.prove(1)
        self.assertTrue(any('combines independent source fixtures' in p for p in report['problems']))
        self.assertIn('fixture joins journals of binding kinds device/text', report['journals'][ASKS]['reasons'][0])


if __name__ == '__main__':
    unittest.main()
