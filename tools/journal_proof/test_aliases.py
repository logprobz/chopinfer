"""Repeated table pins require explicit, byte-identical journal aliases."""
import collections
import json
import sys

import capture
from common import Invalid, fnv
from provenance import seal
from test_capture import FAKE_HOST, Harness, INSTANCE, TEST
from testkit import CREATE, MODEL, ROOT, STEP, fixture_sites, write_dump

sys.path.insert(0, str(capture.ROOT / 'specs'))
from identity_spec import identity_spec

ALIAS = 'toy-step-alias.journal'


class AliasTests(Harness):
    def table_revision(self, bodies=None, extra=''):
        bodies = bodies or self.base
        pin = f'({len(bodies[STEP])}, 0x{fnv(chr(10).join(bodies[STEP]).encode())})'
        source = (TEST + f'const ROWS: [(usize, u64); 1] = [{pin}];\n'
                  'fn table() { check_journal(&lines, expected, &file).unwrap(); }\n' + extra)
        return self.repo.commit({CREATE: bodies[CREATE]}, source_extra=source, extra_files={
            'crates/bundle-runtime/src/fake_host.rs': FAKE_HOST.encode(),
            ROOT + 'src/instance.rs': INSTANCE.encode(), 'ci/rust-profile.env': b'export CARGO_BUILD_JOBS=1\n'})

    def aliases(self):
        bodies = dict(self.base, **{ALIAS: self.base[STEP]})
        sites = {CREATE: [ROOT + 'src/fixture_tests.rs', 2],
                 STEP: [ROOT + 'src/fixture_tests.rs', 7], ALIAS: [ROOT + 'src/fixture_tests.rs', 7]}
        def dump(out):
            write_dump(out, self.base, {name: sites[name] for name in self.base})
            write_dump(out, {CREATE: bodies[CREATE], ALIAS: bodies[ALIAS]},
                       {name: sites[name] for name in (CREATE, ALIAS)}, test='alias')
        return self.capture(bodies, self.table_revision(), dump=dump)

    def test_table_alias_is_sealed_and_reported_in_identity_proof(self):
        revision, out, _ = self.aliases()
        index = json.loads((out / 'index.json').read_text())
        self.assertEqual(index['pin_aliases'], {ALIAS: STEP})
        self.assertEqual(set(index['journals']), {CREATE, STEP, ALIAS})
        spec = identity_spec(self.repo.repo, MODEL, revision, out / 'index.json')
        code, report = self.prove((revision, out), (revision, out), spec, '--require', 'identity')
        self.assertEqual(code, 0, report)
        self.assertEqual(report['coverage']['pin_aliases'], dict(base={ALIAS: STEP}, head={ALIAS: STEP}))
        self.assertIn(ALIAS, report['claim']['compared']['pin_aliases'])
        self.assertIn(STEP, report['claim']['compared']['pin_aliases'])

    def test_no_alias_keeps_index_and_report_fields_absent(self):
        revision, out, _ = self.capture(self.base)
        self.assertNotIn('pin_aliases', json.loads((out / 'index.json').read_text()))
        spec = identity_spec(self.repo.repo, MODEL, revision, out / 'index.json')
        code, report = self.prove((revision, out), (revision, out), spec, '--require', 'identity')
        self.assertEqual(code, 0, report)
        self.assertNotIn('pin_aliases', report['coverage'])
        self.assertNotIn('pin_aliases', report['claim']['compared'])

    def test_literal_site_cannot_check_an_extra_alias(self):
        revision = self.commit(self.base)
        bodies = dict(self.base, **{ALIAS: self.base[STEP]})
        sites = dict(fixture_sites(self.base))
        sites[ALIAS] = sites[STEP]
        with self.assertRaisesRegex(Invalid, 'captured journals do not match the source pins'):
            self.capture(bodies, revision, dump=lambda out: write_dump(out, bodies, sites))

    def test_alias_cannot_introduce_a_pin_absent_from_source(self):
        bodies = dict(self.base, **{ALIAS: ['unlisted body']})
        sites = {CREATE: [ROOT + 'src/fixture_tests.rs', 2],
                 STEP: [ROOT + 'src/fixture_tests.rs', 7], ALIAS: [ROOT + 'src/fixture_tests.rs', 7]}
        with self.assertRaisesRegex(Invalid, 'unpinned'):
            self.capture(bodies, self.table_revision(), dump=lambda out: write_dump(out, bodies, sites))

    def test_alias_does_not_hide_unreached_literal_call(self):
        pin = f'({len(self.base[STEP])}, 0x{fnv(chr(10).join(self.base[STEP]).encode())})'
        revision = self.table_revision(extra=f'fn ghost() {{ check_journal(&lines, {pin}, "ghost.journal"); }}\n')
        bodies = dict(self.base, **{ALIAS: self.base[STEP], 'another-alias.journal': self.base[STEP]})
        sites = {name: [ROOT + 'src/fixture_tests.rs', 2 if name == CREATE else 7] for name in bodies}
        with self.assertRaisesRegex(Invalid, 'never reached'):
            self.capture(bodies, revision, dump=lambda out: write_dump(out, bodies, sites))

    def test_proof_recomputes_sealed_aliases(self):
        revision, out, _ = self.aliases()
        spec = identity_spec(self.repo.repo, MODEL, revision, out / 'index.json')
        payload = json.loads((out / 'index.json').read_text())
        payload.pop('attestation')
        for bad in ({}, {STEP: ALIAS}, {ALIAS: CREATE}):
            with self.subTest(aliases=bad):
                (out / 'index.json').write_text(json.dumps(seal(dict(payload, pin_aliases=bad))))
                code, report = self.prove((revision, out), (revision, out), spec, '--require', 'identity')
                self.assertEqual(code, 2, report)
                self.assertIn('pin aliases', report['problems'][0])

    def test_equal_fnv_with_different_sha256_is_not_an_alias(self):
        from pins import pin_aliases
        pin = [1, '0123456789abcdef']
        sites = {('source.rs', 7): None}
        checked = {STEP: ('source.rs', 7), ALIAS: ('source.rs', 7)}
        with self.assertRaisesRegex(Invalid, 'unpinned'):
            pin_aliases({STEP: pin, ALIAS: pin}, collections.Counter({tuple(pin): 1}), sites, checked,
                        {STEP: 'a' * 64, ALIAS: 'b' * 64})

    def test_repeated_literal_check_cannot_be_hidden_by_a_table_check(self):
        from pins import pin_aliases
        pin = [1, '0123456789abcdef']
        sites = {('source.rs', 7): None, ('source.rs', 9): pin}
        checked = {STEP: ('source.rs', 9), ALIAS: ('source.rs', 7)}
        with self.assertRaisesRegex(Invalid, 'unpinned'):
            pin_aliases({STEP: pin, ALIAS: pin}, collections.Counter({tuple(pin): 1}), sites, checked,
                        {STEP: 'a' * 64, ALIAS: 'a' * 64}, {ALIAS: [['other-test', 'source.rs', 9]]})
