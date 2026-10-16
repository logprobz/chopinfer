#!/usr/bin/env python3
"""The capture helper: archive patching, the build command, result parsing,
dump authentication, call-site coverage and the sealed v3 index."""
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import capture
from common import Invalid, fnv
from instrument import patch as patch_archive, test_names
from pins import check_sites
from proof import SCHEMA, main
from provenance import init
from testkit import CREATE, MODEL, ROOT, STEP, GitFixture, fixture_sites, fixtures, write_dump

FAKE_HOST = '''pub struct FakeDevice<O = NoObserver> {
    next: Cell<usize>,
}
fn issue(&self) -> usize {
        self.next.set(next);
        next
}
    pub fn note(&self, entry: String) {
    }
fn allocate() {
        self.buffer_order.borrow_mut().push(id);
        BundleDeviceBufferHandleV1::from_parts(1, id)
}
fn stream() {
        self.stream_ranks.borrow_mut().insert(id, rank);
}
fn module() {
        BundleCudaModuleHandleV1::from_parts(1, id)
    }
fn resolve() {
        self.symbol_names.borrow_mut().insert(id, name);
}
pub fn check_journal(lines: &[String], expected: (usize, u64), file: &str) -> Result<(), String> {
    let observed = journal_digest(lines);
    if observed == expected {
        return Ok(());
    }
}
'''
INSTANCE = '''impl Instance {
    fn create(services: HostServicesV1, configuration: &[u8]) -> Result<Self, Failed<Self>> {
        create(services, configuration)
    }
}
'''
TEST = '#[test]\nfn fixture_test() {}\n'
EMPTY = dict(added_resolves=[], added_modules=[], create_policy={})


class Harness(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='journal-capture-test-')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        authority = patch('provenance.AUTHORITY', self.root / 'private-authority')
        authority.start()
        self.addCleanup(authority.stop)
        init()
        self.repo = GitFixture(self.root / 'fixture.git')
        self.base, self.head, self.spec = fixtures()
        self.local = self.root / 'local.env'
        self.local.write_text('export CHOPINFER_KERNEL_GPU=0\n')
        (self.root / 'targets').mkdir()
        self.count = 0

    def commit(self, bodies, source_extra=''):
        return self.repo.commit(bodies, source_extra=TEST + source_extra, extra_files={
            'crates/bundle-runtime/src/fake_host.rs': FAKE_HOST.encode(),
            ROOT + 'src/instance.rs': INSTANCE.encode(),
            'ci/rust-profile.env': b'export CARGO_BUILD_JOBS=1\n'})

    def capture(self, bodies, revision=None, dump=None, **options):
        """A capture of `bodies` whose test run is replaced by `dump`
        (default: testkit.write_dump of the bodies)."""
        self.count += 1
        out = self.root / f'capture-{self.count}'
        revision = revision or self.commit(bodies)

        def run(source, target, out, *args):
            self.archived = source
            (dump or (lambda out: write_dump(out, bodies)))(out)
            return 7, dict(build=dict(seconds=1.0, peak_rss_kib=10), tests=dict(seconds=2.0, peak_rss_kib=20))
        with patch('capture.run_build', side_effect=run):
            record = capture.capture(self.repo.repo, MODEL, revision, out, self.local, scratch=self.root,
                                     target_root=self.root / 'targets', minimum_free_gib=0, start_free_gib=0,
                                     require_clean=False, **options)
        return revision, out, record

    def prove(self, base, head, spec, *extra):
        (self.root / 'spec.json').write_text(json.dumps(spec))
        report = self.root / 'report.json'
        with contextlib.redirect_stdout(io.StringIO()):
            code = main(['--repo', str(self.repo.repo), '--model', MODEL, '--base', base[0], '--head', head[0],
                         '--spec', str(self.root / 'spec.json'), '--base-capture-index', str(base[1] / 'index.json'),
                         '--head-capture-index', str(head[1] / 'index.json'), '--report', str(report), *extra])
        return code, json.loads(report.read_text())


class CaptureTests(Harness):
    def test_sealed_v3_capture_is_proven(self):
        base = self.capture(self.base)
        head = self.capture(self.head)
        record = head[2]
        self.assertTrue(record['complete'] and record['archive_removed'] and record['target_removed'])
        self.assertEqual(record['tests_passed'], 7)
        index = json.loads((head[1] / 'index.json').read_text())
        self.assertEqual(index['schema'], 'chopinfer.journal-captures.v3')
        self.assertEqual(index['model'], MODEL)
        self.assertEqual(index['sites'], fixture_sites(self.head))
        self.assertEqual(index['journals'], {name: name for name in sorted(self.head)})
        self.assertNotIn(str(self.root), json.dumps(index))
        spec = dict(self.spec, auxiliary_rules={'host': dict(
            added_resolves=[dict(symbol='chop_new', ids=[5], count=1)],
            added_modules=[dict(name='cubin_new', fnv1a='cubin_new_hash', bytes=100, ids=[4], count=1)],
            create_policy=self.spec['journals'][CREATE])})
        code, report = self.prove(base, head, spec)
        self.assertEqual(code, 0, report.get('problems') or report.get('journals'))
        self.assertEqual(report['auxiliary_provenance']['head']['schema'], 'chopinfer.journal-captures.v3')
        self.assertIn('every call site', report['auxiliary_provenance']['head']['call_sites'])

    def test_runner_and_jobs_do_not_change_what_is_sealed(self):
        revision = self.commit(self.base)
        serial = self.capture(self.base, revision)
        parallel = self.capture(self.base, revision, runner='nextest', jobs=8)
        files = lambda root: {str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob('*'))
                              if p.is_file() and (p.parent != root or p.suffix in ('.txt', '.pin', '.site') or p.name == 'index.json')}
        self.assertEqual(files(serial[1]), files(parallel[1]))
        self.assertEqual((serial[2]['runner'], parallel[2]['runner']), ('libtest', 'nextest'))
        code, report = self.prove(serial[:2], parallel[:2], dict(
            schema=SCHEMA, fixtures=[dict(name='host', create=CREATE, journals=[CREATE, STEP])], auxiliary_rules={'host': EMPTY}),
            '--require', 'identity')
        self.assertEqual(code, 0, report.get('problems') or report.get('identity_checklist'))

    def test_the_archive_copy_is_patched_and_the_repository_is_not(self):
        revision = self.commit(self.base)
        _, out, record = self.capture(self.base, revision)
        self.assertEqual([p['path'] for p in record['patches']],
                         ['crates/bundle-runtime/src/fake_host.rs', 'crates/bundle-runtime/src/fake_host/jproof_trace.rs',
                          ROOT + 'src/instance.rs'])
        self.assertNotIn('track_caller', self.repo.run('show', f'{revision}:crates/bundle-runtime/src/fake_host.rs'))

    def test_a_table_call_that_is_never_reached_is_refused(self):
        ghost = 'fn ghost() { check_journal(&lines, expected, &file).unwrap(); }\n'
        with self.assertRaisesRegex(Invalid, 'never reached'):
            self.capture(self.base, self.commit(self.base, ghost))

    def test_the_proof_refuses_a_table_call_its_capture_never_reached(self):
        import shutil
        from provenance import seal
        base = self.capture(self.base)
        ghost = self.commit(self.base, 'fn ghost() { check_journal(&lines, expected, &file).unwrap(); }\n')
        head = self.root / 'resealed'
        shutil.copytree(base[1], head)
        payload = json.loads((head / 'index.json').read_text())
        payload.pop('attestation')
        payload['revision'] = payload['archive_commit'] = ghost
        manifest = json.loads((head / 'aux/manifest.json').read_text())
        manifest['revision'] = ghost
        (head / 'aux/manifest.json').write_text(json.dumps(manifest))
        digest = hashlib.sha256((head / 'aux/manifest.json').read_bytes()).hexdigest()
        payload['auxiliary']['sha256'] = payload['auxiliary_files']['aux/manifest.json'] = digest
        (head / 'index.json').write_text(json.dumps(seal(payload)))
        code, report = self.prove(base[:2], (ghost, head), dict(
            schema=SCHEMA, fixtures=[dict(name='host', create=CREATE, journals=[CREATE, STEP])], auxiliary_rules={'host': EMPTY}))
        self.assertEqual(code, 2, report)
        self.assertRegex(report['problems'][0], r'never reached.*fixture_tests\.rs:7')

    def test_a_reached_table_call_is_accepted(self):
        # The step's pin is a table row, and the step is checked through the
        # table read on its own line.
        body = self.base[STEP]
        table = (f'const ROWS: [((usize, u64), &str); 1] = [(({len(body)}, 0x{fnv(chr(10).join(body).encode())}), "{STEP}")];\n'
                 'fn table() { check_journal(&lines, expected, &file).unwrap(); }\n')
        revision = self.repo.commit({CREATE: self.base[CREATE]}, source_extra=TEST + table, extra_files={
            'crates/bundle-runtime/src/fake_host.rs': FAKE_HOST.encode(), ROOT + 'src/instance.rs': INSTANCE.encode(),
            'ci/rust-profile.env': b'export CARGO_BUILD_JOBS=1\n'})
        sites = {CREATE: [ROOT + 'src/fixture_tests.rs', 2], STEP: [ROOT + 'src/fixture_tests.rs', 7]}
        _, out, record = self.capture(self.base, revision, dump=lambda out: write_dump(out, self.base, sites))
        self.assertTrue(record['complete'])
        self.assertEqual(json.loads((out / 'index.json').read_text())['sites'], sites)

    def test_an_empty_journal_is_captured_as_a_text_journal(self):
        # A guard pinned as the empty body (0, 0xcbf29ce484222325): its test's
        # device recorded other lines, none of which can witness it.
        bodies = dict(self.base, **{'toy-empty.txt': []})
        revision, out, record = self.capture(bodies)
        index = json.loads((out / 'index.json').read_text())
        self.assertEqual(index['pins']['toy-empty.txt'], [0, 'cbf29ce484222325'])
        self.assertEqual((out / 'toy-empty.txt').read_bytes(), b'')
        manifest = json.loads((out / 'aux/manifest.json').read_text())
        self.assertIn('toy-empty.txt', manifest['text'])
        self.assertEqual(set(manifest['bindings']), {CREATE, STEP})

    def test_a_zero_line_pin_must_be_the_empty_body(self):
        extra = 'fn bad() { check_journal(&lines, (0, 0x1), "toy-bad.txt").unwrap(); }\n'
        with self.assertRaisesRegex(Invalid, r'invalid pin tuple \(0, 0x1\); a journal of 0 lines is pinned \(0xcbf29ce484222325\)'):
            self.capture(self.base, self.commit(self.base, extra))

    def test_a_journal_checked_at_no_call_site_is_refused(self):
        sites = dict(fixture_sites(self.base), **{STEP: [ROOT + 'src/fixture_tests.rs', 40]})
        with self.assertRaisesRegex(Invalid, 'checked at no check_journal call site'):
            self.capture(self.base, dump=lambda out: write_dump(out, self.base, sites))

    def test_a_missing_call_site_record_is_refused(self):
        def dump(out):
            (write_dump(out, self.base) / f'{STEP}.site').unlink()
        with self.assertRaisesRegex(Invalid, r'no \.site sidecar for its check'):
            self.capture(self.base, dump=dump)

    def test_a_dump_that_failed_its_pin_is_refused(self):
        def dump(out):
            (write_dump(out, self.base) / STEP).write_text('corrupted')
        with self.assertRaisesRegex(Invalid, 'the run checked'):
            self.capture(self.base, dump=dump)
        record = json.loads((self.root / f'capture-{self.count}/capture.json').read_text())
        self.assertFalse(record['complete'])
        self.assertTrue(record['archive_removed'] and record['target_removed'])
        self.assertFalse((self.root / f'capture-{self.count}/index.json').exists())

    def test_a_missing_dump_is_refused(self):
        def dump(out):
            checks = write_dump(out, self.base)
            for suffix in ('', '.pin', '.site'):
                (checks / f'{STEP}{suffix}').unlink()
            (out / 'aux/tests/fixture' / f'{STEP}.checkpoint.json').unlink()
        with self.assertRaisesRegex(Invalid, 'captured journals do not match the source pins'):
            self.capture(self.base, dump=dump)

    def test_changed_helper_is_not_sealed(self):
        with patch('capture.helper_identity', side_effect=[('before', {}), ('after', {})]):
            with self.assertRaisesRegex(Invalid, 'helper files changed'):
                self.capture(self.base)
        self.assertFalse((self.root / f'capture-{self.count}/index.json').exists())

    def test_low_disk_refuses_before_writes(self):
        out = self.root / 'low-disk'
        with patch('capture.shutil.disk_usage', return_value=SimpleNamespace(free=179 * 1024**3)):
            with self.assertRaisesRegex(Invalid, 'required free space'):
                capture.capture(self.repo.repo, MODEL, self.commit(self.base), out, self.local, scratch=self.root,
                                target_root=self.root / 'targets', require_clean=False)
        self.assertFalse(out.exists())

    def test_a_missing_key_stops_the_capture_before_any_build(self):
        (self.root / 'private-authority/key').unlink()
        with patch('capture.archive') as archive, patch('capture.run_build') as build:
            with self.assertRaisesRegex(Invalid, 'provenance.py init'):
                capture.capture(self.repo.repo, MODEL, self.commit(self.base), self.root / 'no-key', self.local,
                                scratch=self.root, target_root=self.root / 'targets', minimum_free_gib=0,
                                start_free_gib=0, require_clean=False)
        archive.assert_not_called()
        build.assert_not_called()
        self.assertFalse((self.root / 'no-key').exists())
        self.assertFalse((self.root / 'private-authority/key').exists())

    def test_allowed_ignored_list_is_sealed_without_its_path(self):
        allowed = self.root / 'allowed.txt'
        allowed.write_text('sweeps::every_row\n')
        _, out, record = self.capture(self.base, allowed_ignored=allowed)
        index = json.loads((out / 'index.json').read_text())
        self.assertEqual(index['allowed_ignored'], dict(sha256=hashlib.sha256(allowed.read_bytes()).hexdigest(),
                                                        tests=['sweeps::every_row']))
        self.assertEqual(record['allowed_ignored']['path'], str(allowed))


class IdentitySpecTests(Harness):
    def test_the_identity_spec_of_a_capture_proves_it_against_itself(self):
        sys.path.insert(0, str(capture.ROOT / 'specs'))
        from identity_spec import identity_spec
        revision, out, _ = self.capture(self.base)
        spec = identity_spec(self.repo.repo, MODEL, revision, out / 'index.json')
        self.assertEqual([f['journals'] for f in spec['fixtures']], [[CREATE, STEP]])
        self.assertEqual(spec['fixtures'][0]['create'], CREATE)
        code, report = self.prove((revision, out), (revision, out), spec, '--require', 'identity')
        self.assertEqual(code, 0, report.get('problems') or report.get('identity_checklist'))


class SiteTests(unittest.TestCase):
    sites = {('a.rs', 3): [1, '00'], ('a.rs', 9): None}

    def test_every_call_site_must_be_reached(self):
        with self.assertRaisesRegex(Invalid, r'never reached.*a\.rs:9'):
            check_sites(self.sites, {'x.txt': ('a.rs', 3)}, {'x.txt': [1, '00']})
        # Review C2 r2: a literal call too, even when its pin is covered.
        with self.assertRaisesRegex(Invalid, r'never reached.*a\.rs:3'):
            check_sites(self.sites, {'x.txt': ('a.rs', 9), 'y.txt': ('a.rs', 9)}, {'x.txt': [1, '00'], 'y.txt': [1, '00']})
        check_sites(self.sites, {'x.txt': ('a.rs', 3), 'y.txt': ('a.rs', 9)}, {'x.txt': [1, '00'], 'y.txt': [2, '01']})

    def test_a_literal_call_checks_its_own_pin(self):
        with self.assertRaisesRegex(Invalid, 'with another pin'):
            check_sites(self.sites, {'x.txt': ('a.rs', 3), 'y.txt': ('a.rs', 9)}, {'x.txt': [2, '01'], 'y.txt': [1, '00']})


class InstrumentTests(unittest.TestCase):
    """The archive patch applies to every model of this checkout."""

    def test_patch_applies_to_every_bundle(self):
        top = Path(subprocess.check_output(['git', '-C', str(capture.ROOT), 'rev-parse', '--show-toplevel'], text=True).strip())
        bundles = [p.parent / 'bundle' for p in sorted((top / 'models').glob('*/model.toml')) if (p.parent / 'bundle/src/instance.rs').is_file()]
        self.assertTrue(bundles)
        for bundle in bundles:
            relative = str(bundle.relative_to(top))
            with self.subTest(bundle=relative), tempfile.TemporaryDirectory(prefix='journal-patch-test-') as name:
                source = Path(name)
                data = subprocess.check_output(['git', '-C', str(top), 'archive', '--format=tar', 'HEAD',
                                                'crates/bundle-runtime/src', relative + '/src'])
                with tarfile.open(fileobj=io.BytesIO(data)) as archive:
                    archive.extractall(source, filter='data')
                records = patch_archive(source, relative)
                self.assertEqual(len(records), 3)
                shared = (source / 'crates/bundle-runtime/src/fake_host.rs').read_text()
                self.assertIn('#[track_caller]\npub fn check_journal(', shared)
                self.assertIn('jproof_trace::checkpoint(file);', shared)
                self.assertIn('jproof_trace::end_create', (source / relative / 'src/instance.rs').read_text())
                self.assertTrue(test_names(source, relative))


STUB = '''#!PYTHON
import json, os, sys
from pathlib import Path
args = sys.argv[1:]
log = Path(os.environ['CHOPINFER_JOURNAL_DUMP']) / 'stub-calls.jsonl'
with log.open('a') as out:
    out.write(json.dumps(dict(args=args, cwd=os.getcwd(), env={k: os.environ.get(k) for k in (
        'CUDA_VISIBLE_DEVICES', 'CHOPINFER_KERNEL_GPU', 'KERNELS_LOCK_UPDATE', 'RUST_TEST_THREADS', 'CARGO_TARGET_DIR',
        'CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_RUNNER')})) + '\\n')
if '--no-run' in args:
    sys.exit(0)
if args[0] == 'test':
    print(os.environ.get('STUB_LOG', 'running 3 tests\\ntest result: ok. 3 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.01s'))
    sys.exit(int(os.environ.get('STUB_EXIT', '0')))
if args[:2] == ['nextest', 'run']:
    report = Path('target/nextest/journal-capture/junit.xml')
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text('<testsuites><testsuite name="bundle-toy"><testcase name="a"/><testcase name="b"/></testsuite></testsuites>')
    sys.exit(0)
if args[:2] == ['nextest', 'list']:
    print(json.dumps({'rust-suites': {'bundle-toy': {'package-name': 'bundle-toy', 'testcases': {
        'a': {'ignored': False}, 'b': {'ignored': False}, 'sweep': {'ignored': True}}}}}))
    sys.exit(0)
sys.exit(3)
'''


class BuildTests(unittest.TestCase):
    """run_build with a stub cargo, as the capture helper runs it."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='journal-build-test-')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root / 'source'
        (self.source / 'ci').mkdir(parents=True)
        (self.source / 'ci/rust-profile.env').write_text('export CARGO_BUILD_JOBS=1\nexport RUST_TEST_THREADS=1\n')
        self.local = self.root / 'local.env'
        self.local.write_text('export CHOPINFER_KERNEL_GPU=0\nexport KERNELS_LOCK_UPDATE=1\n')
        self.out = self.root / 'out'
        self.out.mkdir()
        self.target = self.root / 'target'
        self.cargo = self.root / 'cargo'
        self.cargo.write_text(STUB.replace('PYTHON', sys.executable))
        self.cargo.chmod(0o755)

    def run_build(self, runner='libtest', jobs=1, timeout=60, heavy=None, allowed=None, env=None):
        with patch.dict(os.environ, env or {}):
            return capture.run_build(self.source, self.target, self.out, self.local, str(self.cargo), 'bundle-toy',
                                     runner, jobs, timeout, heavy, self.root, 0, allowed)

    def calls(self):
        return [json.loads(line) for line in (self.out / 'stub-calls.jsonl').read_text().splitlines()]

    def test_libtest_builds_then_runs_with_a_clean_environment(self):
        passed, timings = self.run_build(jobs=4)
        self.assertEqual(passed, 3)
        self.assertEqual(set(timings), {'build', 'tests'})
        build, run = self.calls()
        common = ['-p', 'bundle-toy', '--release', '--locked', '--offline', '--lib']
        self.assertEqual(build['args'], ['test', *common, '--no-run'])
        self.assertEqual(run['args'], ['test', *common, '--no-fail-fast', '--', '--test-threads=4'])
        self.assertEqual(run['cwd'], str(self.source))
        self.assertEqual(run['env']['CUDA_VISIBLE_DEVICES'], '')
        self.assertIsNone(run['env']['CHOPINFER_KERNEL_GPU'])
        self.assertIsNone(run['env']['KERNELS_LOCK_UPDATE'])
        self.assertIsNone(run['env']['RUST_TEST_THREADS'])
        self.assertEqual(run['env']['CARGO_TARGET_DIR'], str(self.target))
        self.assertIn('test-processes.tsv', run['env']['CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_RUNNER'])
        self.assertIsNone(build['env']['CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_RUNNER'])

    def test_nextest_runs_one_process_per_test_and_reads_its_reports(self):
        allowed = dict(path='x', sha256='0', tests=['sweep'])
        passed, _ = self.run_build(runner='nextest', jobs=8, allowed=allowed)
        self.assertEqual(passed, 2)
        build, run, listing = self.calls()
        self.assertEqual(run['args'][:2], ['nextest', 'run'])
        self.assertEqual(run['args'][-4:], ['--profile', 'journal-capture', '--test-threads', '8'])
        self.assertIn('--no-fail-fast', run['args'])
        self.assertEqual(listing['args'][:2], ['nextest', 'list'])
        self.assertTrue((self.out / 'junit.xml').is_file())
        with self.assertRaisesRegex(Invalid, 'no allowed list names them'):
            self.run_build(runner='nextest')

    def test_heavy_wrapper_runs_the_whole_build(self):
        heavy = self.root / 'heavy.sh'
        heavy.write_text('echo "$@" > "$(dirname "$0")/heavy-args"\nexec "$@"\n')
        self.run_build(heavy=heavy)
        self.assertTrue((self.root / 'heavy-args').read_text().startswith('bash -c'))
        self.assertEqual(json.loads((self.out / 'build-command.json').read_text())[:2], ['bash', str(heavy)])

    def test_failed_zero_and_skipped_runs_are_refused(self):
        for log, code, pattern in (('running 3 tests\ntest result: ok. 3 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out;', '101', 'exited'),
                                   ('running 0 tests\ntest result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out;', '0', 'nonzero passing'),
                                   ('running 3 tests\ntest result: ok. 3 passed; 0 failed; 0 ignored; 0 measured; 2 filtered out;', '0', 'nonzero passing')):
            with self.subTest(pattern=pattern):
                with self.assertRaisesRegex(Invalid, pattern):
                    self.run_build(env=dict(STUB_LOG=log, STUB_EXIT=code))

    def test_timeout_stops_the_owned_build(self):
        self.cargo.write_text(STUB.replace('PYTHON', sys.executable).replace("if '--no-run' in args:", "import time; time.sleep(60)\nif '--no-run' in args:"))
        with self.assertRaisesRegex(TimeoutError, 'build deadline'):
            self.run_build(timeout=1)


class ResultTests(unittest.TestCase):
    def log(self, ran=(), ignored=()):
        lines = [f'test {name} ... ok' for name in ran] + [f'test {name} ... ignored, sweep' for name in ignored]
        return '\n'.join([f'running {1 + len(ran) + len(ignored)} tests'] + lines + [f'test result: ok. {1 + len(ran)} passed; 0 failed; {len(ignored)} ignored; 0 measured; 0 filtered out;'])

    def allowed(self, *tests):
        return dict(path='x', sha256='0', tests=sorted(tests))

    def test_listed_ignored_tests_pass(self):
        self.assertEqual(capture.test_results(self.log(ignored=['s::a', 's::b']), self.allowed('s::a', 's::b')), 1)

    def test_unlisted_or_unrun_ignored_tests_fail(self):
        with self.assertRaisesRegex(Invalid, 'allowed list does not name'):
            capture.test_results(self.log(ignored=['s::a', 's::x']), self.allowed('s::a'))
        with self.assertRaisesRegex(Invalid, 'ran or no longer exist'):
            capture.test_results(self.log(ran=['s::b'], ignored=['s::a']), self.allowed('s::a', 's::b'))
        with self.assertRaisesRegex(Invalid, 'no allowed list names them'):
            capture.test_results(self.log(ignored=['s::a']))

    def test_allowed_lists_name_exact_tests_once(self):
        with tempfile.TemporaryDirectory() as name:
            for text in ('s::*\n', 's::a b\n', 'a\na\n', '\n'):
                path = Path(name) / 'list.txt'
                path.write_text(text)
                with self.assertRaises(Invalid):
                    capture.read_allowed_ignored(path)

    def test_a_failed_nextest_case_is_refused(self):
        junit = '<testsuites><testsuite><testcase name="a"/><testcase name="b"><failure/></testcase></testsuite></testsuites>'
        listing = json.dumps({'rust-suites': {'bundle-toy': {'package-name': 'bundle-toy', 'testcases': {
            'a': {'ignored': False}, 'b': {'ignored': False}}}}})
        with self.assertRaisesRegex(Invalid, r"did not all pass: \['b'\]"):
            capture.nextest_results(junit, listing, 'bundle-toy')
        with self.assertRaisesRegex(Invalid, 'lists 0 test binaries'):
            capture.nextest_results(junit, listing, 'bundle-other')



DUMP_PROGRAM = """mod jproof_trace {
    pub fn checkpoint(_: &str) {}
    pub fn check_directory() -> String {
        format!("checks/{}", std::thread::current().name().unwrap_or("main").replace("::", "."))
    }
}
fn journal_digest(lines: &[String]) -> (usize, u64) { (lines.len(), lines.iter().map(|l| l.len() as u64).sum()) }
#[track_caller]
pub fn check_journal(lines: &[String], expected: (usize, u64), file: &str) -> Result<(), String> {
DUMP
        return Ok(());
    }
    Err(format!("{file}: observed {:?}, expected {expected:?}", journal_digest(lines)))
}
fn main() {
    let good = vec!["good".to_string()];
    let bad = vec!["badd".to_string(), "x".to_string()];
    match std::env::args().nth(1).unwrap().as_str() {
        "once" => check_journal(&good, (1, 4), "toy.txt").unwrap(),
        "mismatch" => check_journal(&bad, (1, 4), "toy.txt").unwrap(),
        // A test that catches the refusal of its second check and passes.
        "caught" => {
            check_journal(&good, (1, 4), "toy.txt").unwrap();
            let _ = std::panic::catch_unwind(|| check_journal(&bad, (1, 4), "toy.txt"));
        }
        // A loop checking `bad` then `good` against one pin and file name.
        _ => for lines in [&bad, &good] { let _ = check_journal(lines, (1, 4), "toy.txt"); },
    }
}
// Two tests checking one journal, as a test rerunning another's checks.
#[test]
fn first_of_two() { check_journal(&["good".to_string()], (1, 4), "toy-two.txt").unwrap(); }
#[test]
fn second_of_two() { check_journal(&["good".to_string()], (1, 4), "toy-two.txt").unwrap(); }
#[test]
#[should_panic]
fn a_second_check_is_expected_to_panic() {
    check_journal(&["good".to_string()], (1, 4), "toy.txt").unwrap();
    check_journal(&["badd".to_string(), "x".to_string()], (1, 4), "toy.txt").unwrap();
}
"""
ABORTED = -6  # SIGABRT


class DumpTests(unittest.TestCase):
    """The instrumented checker, compiled: it writes each journal once, still
    fails a mismatch as the original does (review C2 r1), and a refused dump
    ends the process, so no test can catch it (review C2 r2)."""

    @classmethod
    def setUpClass(cls):
        from instrument import DUMP
        cls.tmp = tempfile.TemporaryDirectory(prefix='journal-dump-test-')
        cls.root = Path(cls.tmp.name)
        text = DUMP_PROGRAM.replace('DUMP', DUMP)
        (cls.root / 'dump.rs').write_text(text)
        cls.once_line = next(n for n, line in enumerate(text.splitlines(), 1) if '"once" =>' in line)
        for extra, name in (([], 'dump'), (['--test'], 'dump-tests')):
            subprocess.run(['rustc', '--edition', '2021', *extra, '-o', name, 'dump.rs'], check=True, capture_output=True,
                           cwd=cls.root)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def run_case(self, case, *args, binary='dump'):
        out = self.root / case
        out.mkdir()
        result = subprocess.run([str(self.root / binary), *args], env={**os.environ, 'CHOPINFER_JOURNAL_DUMP': str(out)},
                                capture_output=True, text=True)
        return result, out

    def test_a_good_check_writes_body_pin_and_call_site(self):
        result, out = self.run_case('once', 'once')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual((out / 'checks/main/toy.txt').read_text(), 'good')
        self.assertEqual((out / 'checks/main/toy.txt.site').read_text(), f'dump.rs:{self.once_line}\n')
        self.assertTrue((out / 'checks/main/toy.txt.pin').exists())

    def test_a_mismatch_still_fails_the_test(self):
        result, out = self.run_case('mismatch', 'mismatch')
        self.assertEqual(result.returncode, 101)
        self.assertIn('expected (1, 4)', result.stderr)

    def test_a_second_check_of_one_name_ends_the_process_and_keeps_the_first_body(self):
        result, out = self.run_case('loop', 'loop')
        self.assertEqual(result.returncode, ABORTED)
        self.assertIn('a test checks a journal twice; aborting the test process', result.stderr)
        self.assertEqual((out / 'checks/main/toy.txt').read_text(), 'badd\nx')

    def test_catch_unwind_cannot_catch_a_refused_dump(self):
        # Review C2 r2: the test caught the panic and passed, and the first
        # (good) body stood for the refused check.
        result, out = self.run_case('caught', 'caught')
        self.assertEqual(result.returncode, ABORTED, result.stderr)
        self.assertIn('a test checks a journal twice', result.stderr)
        self.assertEqual((out / 'checks/main/toy.txt').read_text(), 'good')

    def test_two_tests_may_check_one_journal_each_in_its_own_directory(self):
        # The first real captures aborted here: a test reran another's checks.
        result, out = self.run_case('two-tests', 'of_two', '--test-threads=2', binary='dump-tests')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('test result: ok. 2 passed', result.stdout)
        for test in ('first_of_two', 'second_of_two'):
            self.assertEqual((out / 'checks' / test / 'toy-two.txt').read_text(), 'good')

    def test_should_panic_cannot_pass_on_a_refused_dump(self):
        result, out = self.run_case('should-panic', 'a_second_check', '--test-threads=1', binary='dump-tests')
        self.assertEqual(result.returncode, ABORTED, result.stdout + result.stderr)
        self.assertNotIn('test result: ok', result.stdout)
        self.assertIn('a test checks a journal twice; aborting the test process', result.stderr)


class ReviewTests(Harness):
    def test_two_calls_on_one_line_are_refused(self):
        # Review C2 r1: a reached literal call could hide an unreached table
        # read on the same line, the call site being a line.
        extra = ('fn two() { check_journal(&lines, (1, 0x0123456789abcdef), "toy-x.txt").unwrap(); '
                 'check_journal(&lines, expected, &file).unwrap(); }\n')
        with self.assertRaisesRegex(Invalid, 'two check_journal calls on one line'):
            self.capture(self.base, self.commit(self.base, extra))

    def test_replace_refs_do_not_change_what_is_read(self):
        # Review C2 r1: a replace ref made git archive and git show read
        # another commit's tree under the original's name.
        from common import git, resolve
        original = self.commit(self.base)
        other = self.commit(self.head)
        self.repo.run('replace', original, other)
        self.assertEqual(self.repo.run('show', f'{original}:{ROOT}src/fixture_tests.rs'),
                         self.repo.run('show', f'{other}:{ROOT}src/fixture_tests.rs'))
        self.assertEqual(resolve(self.repo.repo, original), original)
        self.assertEqual(git(self.repo.repo, 'show', f'{original}:{ROOT}src/fixture_tests.rs').decode(),
                         subprocess.check_output(['git', '--no-replace-objects', '-C', str(self.repo.repo), 'show',
                                                  f'{original}:{ROOT}src/fixture_tests.rs'], text=True))
        destination = self.root / 'archive'
        destination.mkdir()
        capture.archive(self.repo.repo, original, destination)
        self.assertEqual((destination / ROOT / 'src/fixture_tests.rs').read_text().strip(),
                         subprocess.check_output(['git', '--no-replace-objects', '-C', str(self.repo.repo), 'show',
                                                  f'{original}:{ROOT}src/fixture_tests.rs'], text=True).strip())
        _, out, record = self.capture(self.base, original)
        self.assertEqual(json.loads((out / 'index.json').read_text())['pins'].keys(), self.base.keys())

    def test_an_unreached_literal_call_is_refused_when_a_table_call_covers_its_pin(self):
        # Review C2 r2: toy-a and toy-b, checked through one table read with
        # one pin, made the pin multiset whole while the literal call for
        # toy-ghost.txt, with that pin, was never reached.
        body = self.base[STEP]
        pin = f'({len(body)}, 0x{fnv(chr(10).join(body).encode())})'
        revision = self.repo.commit({CREATE: self.base[CREATE], 'toy-ghost.txt': body}, source_extra=TEST + (
            f'const ROWS: [((usize, u64), &str); 1] = [({pin}, "toy-a.txt")];\n'
            'fn table() { check_journal(&lines, expected, &file).unwrap(); }\n'), extra_files={
            'crates/bundle-runtime/src/fake_host.rs': FAKE_HOST.encode(), ROOT + 'src/instance.rs': INSTANCE.encode(),
            'ci/rust-profile.env': b'export CARGO_BUILD_JOBS=1\n'})
        source = self.repo.run('show', f'{revision}:{ROOT}src/fixture_tests.rs').splitlines()
        line = lambda text: next(n for n, item in enumerate(source, 1) if text in item)
        bodies = {CREATE: self.base[CREATE], 'toy-a.txt': body, 'toy-b.txt': body}
        sites = {CREATE: [ROOT + 'src/fixture_tests.rs', line(CREATE)],
                 'toy-a.txt': [ROOT + 'src/fixture_tests.rs', line('fn table()')],
                 'toy-b.txt': [ROOT + 'src/fixture_tests.rs', line('fn table()')]}
        with self.assertRaisesRegex(Invalid, rf'never reached.*fixture_tests\.rs:{line("toy-ghost.txt")}'):
            self.capture(bodies, revision, dump=lambda out: write_dump(out, bodies, sites))

    def test_the_archive_copy_holds_the_committed_bytes_whatever_the_attributes(self):
        # Review C2 r2: git archive applied export-subst from an untracked
        # info/attributes and changed a committed test's bytes; export-ignore
        # would drop a file.
        revision = self.commit(self.base, '// built from $Format:%H$\n')
        attributes = Path(self.repo.run('rev-parse', '--git-path', 'info/attributes').strip())
        attributes = attributes if attributes.is_absolute() else self.repo.repo / attributes
        attributes.parent.mkdir(parents=True, exist_ok=True)
        attributes.write_text('*.rs export-subst\nci/* export-ignore\n')
        tree = {line.split('\t')[1]: line.split()[2] for line in self.repo.run('ls-tree', '-r', revision).splitlines()}
        committed = {path: subprocess.check_output(['git', '-C', str(self.repo.repo), 'cat-file', 'blob', oid])
                     for path, oid in tree.items()}
        listed = subprocess.check_output(['git', '-C', str(self.repo.repo), 'archive', revision, ROOT + 'src/fixture_tests.rs'])
        with tarfile.open(fileobj=io.BytesIO(listed)) as exported:
            substituted = exported.extractfile(ROOT + 'src/fixture_tests.rs').read()
        self.assertNotEqual(substituted, committed[ROOT + 'src/fixture_tests.rs'])  # the attribute is live
        destination = self.root / 'archive'
        destination.mkdir()
        capture.archive(self.repo.repo, revision, destination)
        written = {str(p.relative_to(destination)): p.read_bytes() for p in destination.rglob('*') if p.is_file()}
        self.assertEqual(written, committed)
        seen = {}
        def dump(out):
            seen['test'] = (self.archived / ROOT / 'src/fixture_tests.rs').read_bytes()
            write_dump(out, self.base)
        self.capture(self.base, revision, dump=dump)
        self.assertEqual(seen['test'], committed[ROOT + 'src/fixture_tests.rs'])

    def test_the_archive_writer_closes_its_pipes_on_every_path(self):
        # Review C2 r3 nit: the blob reader's stderr pipe was never closed.
        import gc, warnings
        revision = self.commit(self.base)
        clash = self.root / 'clash'
        (clash / ROOT / 'src').mkdir(parents=True)
        (clash / ROOT / 'src/fixture_tests.rs').write_text('already here')
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            capture.archive(self.repo.repo, revision, self.root / 'clean')
            with self.assertRaises(FileExistsError):
                capture.archive(self.repo.repo, revision, clash)
            gc.collect()
        self.assertEqual([str(w.message) for w in caught if issubclass(w.category, ResourceWarning)], [])

    def test_helper_bytes_must_equal_their_committed_blobs(self):
        # Review C2 r1: assume-unchanged hid a modified helper from git status.
        import shutil
        from provenance import HELPER_FILES
        tools = self.root / 'checkout/tools/journal_proof'
        tools.mkdir(parents=True)
        for name in HELPER_FILES:
            shutil.copy(capture.ROOT / name, tools / name)
        top = tools.parents[1]
        env = {**os.environ, 'GIT_AUTHOR_NAME': 'f', 'GIT_COMMITTER_NAME': 'f',
               'GIT_AUTHOR_EMAIL': 'f@example.invalid', 'GIT_COMMITTER_EMAIL': 'f@example.invalid'}
        for args in (['init', '-q'], ['add', '.'], ['commit', '-q', '-m', 'helper']):
            subprocess.run(['git', '-C', str(top), *args], check=True, env=env, capture_output=True)
        with patch('capture.ROOT', tools):
            self.assertTrue(capture.helper_source()['clean'])
            (tools / 'instrument.py').write_text((tools / 'instrument.py').read_text() + '# changed\n')
            subprocess.run(['git', '-C', str(top), 'update-index', '--assume-unchanged', 'tools/journal_proof/instrument.py'], check=True)
            self.assertEqual(subprocess.check_output(['git', '-C', str(top), 'status', '--porcelain'], text=True), '')
            with self.assertRaisesRegex(Invalid, r"differ from their committed bytes.*instrument\.py"):
                capture.helper_source()
            self.assertFalse(capture.helper_source(require_clean=False)['clean'])


class RepeatTests(Harness):
    """A journal several tests check (real-run cases: a test that reruns
    another test's checks on real plans). Found by the first real captures,
    which refused it as a duplicate."""

    def two_tests(self, first, second, sites=None, second_sites=None):
        def dump(out):
            write_dump(out, first, sites, test='a::synthetic')
            write_dump(out, second, second_sites or sites, test='b::real')
        return dump

    def test_identical_checks_by_two_tests_bind_the_last_and_record_the_other(self):
        revision, out, record = self.capture(self.base, dump=self.two_tests(self.base, self.base))
        index = json.loads((out / 'index.json').read_text())
        sites = fixture_sites(self.base)
        self.assertEqual(index['repeated_checks'], {name: [['a::synthetic', *site]] for name, site in sorted(sites.items())})
        manifest = json.loads((out / 'aux/manifest.json').read_text())
        self.assertEqual({binding['test'] for binding in manifest['bindings'].values()}, {'b::real'})
        self.assertFalse((out / 'checks').exists())
        sys.path.insert(0, str(capture.ROOT / 'specs'))
        from identity_spec import identity_spec
        spec = identity_spec(self.repo.repo, MODEL, revision, out / 'index.json')
        code, report = self.prove((revision, out), (revision, out), spec, '--require', 'identity')
        self.assertEqual(code, 0, report.get('problems') or report.get('identity_checklist'))
        self.assertIn('also checked by other tests', report['auxiliary_provenance']['head']['call_sites'])

    def test_the_merge_does_not_depend_on_which_test_ran_first(self):
        revision = self.commit(self.base)
        _, one, _ = self.capture(self.base, revision, dump=self.two_tests(self.base, self.base))
        def reverse(out):
            write_dump(out, self.base, test='b::real')
            write_dump(out, self.base, test='a::synthetic')
        _, two, _ = self.capture(self.base, revision, dump=reverse)
        files = lambda root: {str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob('*'))
                              if p.is_file() and (p.parent != root or p.suffix in ('.txt', '.pin', '.site'))}
        self.assertEqual(files(one), files(two))
        index = lambda root: {k: v for k, v in json.loads((root / 'index.json').read_text()).items() if k != 'attestation'}
        self.assertEqual(index(one), index(two))

    def test_two_tests_checking_one_name_with_different_bodies_are_refused(self):
        # The second test's check failed and the test caught the error: its
        # dump stays, so the merge refuses instead of keeping the first.
        changed = dict(self.base, **{STEP: self.base[STEP][:-1] + ['launch 9 chop_other']})
        def dump(out):
            write_dump(out, self.base, test='a::synthetic')
            checks = write_dump(out, changed, test='b::real')
            pin = (checks / f'{STEP}.pin').read_text().split(' expected ')[0]
            first = (out / 'checks/a.synthetic' / f'{STEP}.pin').read_text().split(' expected ')[1]
            (checks / f'{STEP}.pin').write_text(f'{pin} expected {first}')
        with self.assertRaisesRegex(Invalid, f'{STEP}: a::synthetic and b::real checked different bodies or pins'):
            self.capture(self.base, dump=dump)

    def test_a_call_site_reached_only_by_a_repeated_check_counts_as_reached(self):
        # The step's pin is one table row read at two call sites; each test
        # reaches one of them.
        table = ('const ROWS: [((usize, u64), &str); 1] = '
                 f'[(({len(self.base[STEP])}, 0x{fnv(chr(10).join(self.base[STEP]).encode())}), "{STEP}")];\n'
                 'fn table_one() { check_journal(&lines, expected, &file).unwrap(); }\n'
                 'fn table_two() { check_journal(&lines, expected, &file).unwrap(); }\n')
        revision = self.repo.commit({CREATE: self.base[CREATE]}, source_extra=TEST + table, extra_files={
            'crates/bundle-runtime/src/fake_host.rs': FAKE_HOST.encode(), ROOT + 'src/instance.rs': INSTANCE.encode(),
            'ci/rust-profile.env': b'export CARGO_BUILD_JOBS=1\n'})
        source = self.repo.run('show', f'{revision}:{ROOT}src/fixture_tests.rs').splitlines()
        at = lambda text: [ROOT + 'src/fixture_tests.rs', next(n for n, item in enumerate(source, 1) if text in item)]
        one, two = {CREATE: at(CREATE), STEP: at('fn table_one()')}, {CREATE: at(CREATE), STEP: at('fn table_two()')}
        _, out, record = self.capture(self.base, revision, dump=self.two_tests(self.base, self.base, one, two))
        index = json.loads((out / 'index.json').read_text())
        self.assertEqual(index['sites'][STEP], two[STEP])
        self.assertEqual(index['repeated_checks'][STEP], [['a::synthetic', *one[STEP]]])
        # Without the repeated check the first table read is unreached.
        with self.assertRaisesRegex(Invalid, rf'never reached.*fixture_tests\.rs:{one[STEP][1]}'):
            self.capture(self.base, revision, dump=lambda out: write_dump(out, self.base, two, test='b::real'))


class ChildRunTests(unittest.TestCase):
    """A test that reruns the test binary for one test prints that child
    run's summary into the log (found in the first real captures)."""
    PARENT = 'running 3 tests\ntest a ... ok\ntest b ... '
    CHILD = '\nrunning 1 test\ntest b ... ok\n\ntest result: {status}. 1 passed; {failed} failed; 0 ignored; 0 measured; 2 filtered out; finished in 0.00s\n'
    END = 'ok\ntest c ... ok\n\ntest result: ok. 3 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 1.00s\n'

    def test_child_runs_do_not_count_as_the_run(self):
        self.assertEqual(capture.test_results(self.PARENT + self.CHILD.format(status='ok', failed=0) * 2 + self.END), 3)

    def test_a_failed_child_run_is_refused(self):
        with self.assertRaisesRegex(Invalid, 'a child test run failed'):
            capture.test_results(self.PARENT + self.CHILD.format(status='FAILED', failed=1) + self.END)

    def test_the_run_must_count_every_announced_test(self):
        with self.assertRaisesRegex(Invalid, 'nonzero passing'):
            capture.test_results(self.PARENT.replace('running 3', 'running 4') + self.END)


class InventoryTests(unittest.TestCase):
    LISTING = {'rust-suites': {'bundle-toy': {'package-name': 'bundle-toy', 'testcases': {
        'a': {'ignored': False}, 'b': {'ignored': False}, 'sweep': {'ignored': True}}}}}
    ALLOWED = dict(path='x', sha256='0', tests=['sweep'])

    def junit(self, *names):
        return '<testsuites><testsuite>' + ''.join(f'<testcase name="{n}"/>' for n in names) + '</testsuite></testsuites>'

    def test_the_run_must_hold_every_listed_test(self):
        listing = json.dumps(self.LISTING)
        self.assertEqual(capture.nextest_results(self.junit('a', 'b'), listing, 'bundle-toy', self.ALLOWED), 2)
        # Review C2 r1: a default filter ran one of two tests and passed.
        for names, pattern in ((('a',), r"not run \['b'\]"), (('a', 'b', 'b'), r"repeated \['b'\]"),
                               (('a', 'b', 'c'), r"unlisted or repeated \['c'\]")):
            with self.assertRaisesRegex(Invalid, pattern):
                capture.nextest_results(self.junit(*names), listing, 'bundle-toy', self.ALLOWED)

    def test_listing_and_run_use_one_profile(self):
        phases = capture.commands('bundle-toy', 'nextest', 8)
        profile = lambda args: args[args.index('--profile') + 1]
        self.assertEqual(profile(phases['test']), profile(phases['list']))
        self.assertIn('default-filter = "all()"', capture.NEXTEST_PROFILE)


if __name__ == '__main__':
    unittest.main()
