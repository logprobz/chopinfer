"""Journal suffixes share pin, trace and identity authentication."""
import json
import sys
from unittest.mock import patch

import capture
from common import Invalid
from pins import JOURNAL
from test_capture import Harness
from testkit import CREATE, MODEL, STEP, write_dump

sys.path.insert(0, str(capture.ROOT / 'specs'))
from identity_spec import identity_spec
from layout_spec import fixtures_of


class SuffixTests(Harness):
    def renamed(self, suffix):
        create = 'toy-create' + suffix
        bodies = {create: self.base[CREATE], 'toy-before-step' + suffix: self.base[STEP],
                  'toy-empty' + suffix: []}
        return create, bodies

    def dump(self, out, bodies, create):
        with patch('testkit.CREATE', create):
            return write_dump(out, bodies)

    def test_names_accept_only_safe_explicit_suffixes(self):
        for name in ('a.txt', 'a.journal', 'toy-tp1.bf16.journal', 'toy_1-create.txt'):
            self.assertIsNotNone(JOURNAL.fullmatch(name), name)
        for name in ('../a.journal', '/a.journal', 'sub/a.journal', r'sub\a.journal',
                     '.hidden.journal', 'a.journal.pin', 'a.journal.bak', 'a.JOURNAL',
                     'a.json', 'a', 'a journal', 'a\n.journal'):
            self.assertIsNone(JOURNAL.fullmatch(name), name)

    def test_suffixes_capture_checkpoints_text_and_identity(self):
        for suffix in ('.txt', '.journal'):
            with self.subTest(suffix=suffix):
                create, bodies = self.renamed(suffix)
                revision, out, _ = self.capture(bodies, dump=lambda out: self.dump(out, bodies, create))
                spec = identity_spec(self.repo.repo, MODEL, revision, out / 'index.json')
                device = next(f for f in spec['fixtures'] if create in f['journals'])
                self.assertEqual(device['create'], create)
                manifest = json.loads((out / 'aux/manifest.json').read_text())
                self.assertEqual(set(manifest['text']), {'toy-empty' + suffix})
                self.assertEqual(set(manifest['bindings']), set(bodies) - set(manifest['text']))
                self.assertTrue((out / f'aux/{create}.checkpoint.json').is_file())
                self.assertTrue((out / f'{create}.pin').is_file())
                code, report = self.prove((revision, out), (revision, out), spec, '--require', 'identity')
                self.assertEqual(code, 0, report)

    def test_layout_spec_selects_create_with_either_suffix(self):
        for suffix in ('.txt', '.journal'):
            create, bodies = self.renamed(suffix)
            snap = dict(pins={n: dict(fixture='host') for n in bodies})
            self.assertEqual(fixtures_of(snap, snap)[0]['create'], create)

    def test_mixed_suffixes_are_all_authenticated(self):
        bodies = {CREATE: self.base[CREATE], 'toy-step.journal': self.base[STEP]}
        revision, out, _ = self.capture(bodies)
        index = json.loads((out / 'index.json').read_text())
        self.assertEqual(set(index['journals']), set(bodies))
        spec = identity_spec(self.repo.repo, MODEL, revision, out / 'index.json')
        code, report = self.prove((revision, out), (revision, out), spec, '--require', 'identity')
        self.assertEqual(code, 0, report)

    def test_journal_body_and_sidecars_refuse_symlinks_and_directories(self):
        create, bodies = self.renamed('.journal')
        for suffix in ('', '.pin', '.site', '.checkpoint.json'):
            for kind in ('symlink', 'directory'):
                with self.subTest(suffix=suffix, kind=kind):
                    def dump(out):
                        checks = self.dump(out, bodies, create)
                        path = (out / 'aux/tests/fixture' if suffix == '.checkpoint.json' else checks) / (create + suffix)
                        saved = out / 'saved'
                        path.rename(saved)
                        path.symlink_to(saved) if kind == 'symlink' else path.mkdir()
                    with self.assertRaises((Invalid, OSError)):
                        self.capture(bodies, dump=dump)
                    self.assertFalse((self.root / f'capture-{self.count}/index.json').exists())

    def test_missing_journal_pin_or_checkpoint_is_refused(self):
        create, bodies = self.renamed('.journal')
        for suffix in ('.pin', '.site', '.checkpoint.json'):
            with self.subTest(suffix=suffix):
                def dump(out):
                    checks = self.dump(out, bodies, create)
                    directory = out / 'aux/tests/fixture' if suffix == '.checkpoint.json' else checks
                    (directory / (create + suffix)).unlink()
                with self.assertRaises(Invalid):
                    self.capture(bodies, dump=dump)

    def test_modified_journal_body_is_refused(self):
        create, bodies = self.renamed('.journal')
        def dump(out):
            (self.dump(out, bodies, create) / create).write_text('tampered')
        with self.assertRaisesRegex(Invalid, 'the run checked'):
            self.capture(bodies, dump=dump)
