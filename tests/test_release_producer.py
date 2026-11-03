"""Controlled producer acceptance with real Rust and CUDA artifacts."""
import atexit
import json
import os
import subprocess
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools/release'))
import provenance  # noqa: E402
import source_context  # noqa: E402
import release_fixture  # noqa: E402

_FIXTURE = None


def fixture():
    global _FIXTURE
    if _FIXTURE is None:
        temporary = tempfile.TemporaryDirectory(prefix='release-provenance-')
        atexit.register(temporary.cleanup)
        directory = Path(temporary.name)
        root = release_fixture.create(directory)
        scratch = directory / 'builds'
        release_fixture.write(scratch, 'release/chopinfer', 'unproven old output\n')
        env = {**os.environ, 'CHOPINFER_BUILD_REVISION': 'a' * 40, 'CARGO_TARGET_DIR': str(scratch)}
        built = subprocess.run(['bash', root / 'tools/release/build.sh', directory / 'archive'],
                               env=env, text=True, capture_output=True)
        if built.returncode:
            raise RuntimeError(built.stdout + built.stderr)
        prebuilt = Path(next(line.removeprefix('Prebuilt: ') for line in built.stdout.splitlines()
                             if line.startswith('Prebuilt: ')))
        result = {'prebuilt': str(prebuilt), 'record': str(prebuilt / provenance.RECORD),
                  'target': str(prebuilt.parent / 'target')}
        _FIXTURE = dict(directory=directory, root=root, scratch=scratch, result=result, prebuilt=prebuilt,
                        record=json.loads((prebuilt / provenance.RECORD).read_text()))
    return _FIXTURE


class SourceContext(unittest.TestCase):
    def test_controlled_build_caps_jobs_and_honors_a_smaller_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(provenance, 'rust_tools', return_value=(root / 'rustc', root / 'cargo')):
                for requested, expected in [('64', '8'), ('2', '2')]:
                    with self.subTest(requested=requested), patch.dict(os.environ, {'CARGO_BUILD_JOBS': requested}):
                        env = provenance.controlled_environment(root, {'target': provenance.TARGET}, 'a' * 40,
                                                               root / 'target', root / 'stub')
                        self.assertEqual(env['CARGO_BUILD_JOBS'], expected)

    def test_record_binding_covers_artifact_bytes_and_roles(self):
        record = {'source': {}, 'configuration': {}, 'locks': {}, 'kernels': [],
                  'artifacts': [{'path': 'host', 'role': 'bin', 'size': 1, 'sha256': 'a' * 64}]}
        original = provenance.build_identity(record)
        record['artifacts'][0]['build_sha256'] = original
        self.assertEqual(provenance.build_identity(record), original)
        record['artifacts'][0]['sha256'] = 'b' * 64
        self.assertNotEqual(provenance.build_identity(record), original)

    def test_snapshot_binds_selected_inputs_and_ignores_unrelated_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ('Cargo.toml', 'Cargo.lock', 'rust-toolchain.toml', 'LICENSE', 'THIRD_PARTY_NOTICES'):
                release_fixture.write(root, name, 'source\n')
            release_fixture.write(root, '.git/HEAD', 'a' * 40)
            release_fixture.write(root, 'crates/selected/src/lib.rs', 'pub fn value() {}\n')
            roots = [root / 'crates/selected']
            with patch.dict(os.environ, {'CHOPINFER_BUILD_REVISION': 'a' * 40}):
                first = source_context.snapshot(root, roots)
                release_fixture.write(root, 'unrelated/local.json', '{"unrelated": true}\n')
                release_fixture.write(root, 'crates/unselected/src/lib.rs', 'unrelated\n')
                self.assertEqual(source_context.snapshot(root, roots), first)
                release_fixture.write(root, 'crates/selected/src/lib.rs', 'pub fn changed() {}\n')
                self.assertNotEqual(source_context.snapshot(root, roots)['sha256'], first['sha256'])
                with patch.dict(os.environ, {'CHOPINFER_BUILD_REVISION': 'b' * 40}):
                    with self.assertRaisesRegex(ValueError, 'differs from Git HEAD'):
                        source_context.snapshot(root, roots)

    def test_file_reads_refuse_symlinks_and_replacement_during_copy(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / 'artifact'
            path.write_bytes(b'artifact')
            linked = root / 'link'
            linked.symlink_to(path)
            with self.assertRaises(OSError):
                source_context.file_identity(linked)
            class Replace:
                def write(self, block):
                    path.unlink()
                    path.write_bytes(block)
            with self.assertRaisesRegex(ValueError, 'changed while being read'):
                source_context.file_identity(path, destination=Replace())



class CompiledProducer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        for name, value in fixture().items():
            setattr(cls, name, value)

    def test_real_artifacts_have_complete_portable_producer_evidence(self):
        record = self.record
        self.assertEqual(record['schema'], provenance.SCHEMA)
        self.assertEqual(record['source']['revision'], 'a' * 40)
        self.assertEqual(record['source']['context'], 'retained-refs')
        self.assertEqual(len(record['artifacts']), 6)
        for artifact in record['artifacts']:
            path = self.prebuilt / artifact['path']
            self.assertEqual(path.read_bytes()[:4], b'\x7fELF')
            self.assertEqual(source_context.file_identity(path, provenance.MAX_ARTIFACT_BYTES),
                             {key: artifact[key] for key in ('size', 'sha256')})
            if artifact['path'] in ('chopinfer', 'chopinfer-host'):
                env = {**os.environ, 'LD_LIBRARY_PATH': str(self.prebuilt.parent / 'cuda-link-stub')}
                version = subprocess.check_output([path, '--version'], env=env, text=True).strip()
                self.assertEqual(version, f"{artifact['path']} 0.1.0-alpha.1 ({'a' * 40})")
        names = {identity['name'] for evidence in record['kernels'] for identity in evidence['identities']}
        self.assertEqual(names, {'sampling_sm120', 'sampling_mapped_sm120', 'fixture_main', 'fixture_child'})
        self.assertNotIn(str(self.directory), json.dumps(record))
        self.assertNotIn('location', json.dumps(record))
        self.assertEqual((self.scratch / 'release/chopinfer').read_text(), 'unproven old output\n')

    def test_active_bypasses_and_incompatible_profiles_are_rejected(self):
        for name in provenance.BYPASSES:
            with self.subTest(name=name), patch.dict(os.environ, {name: '1'}):
                with self.assertRaisesRegex(ValueError, 'forbidden'):
                    provenance.configuration(self.root)
        for change in ({'CARGO_PROFILE_RELEASE_DEBUG': 'true'}, {'RUSTFLAGS': '-C opt-level=0'},
                       {'RUSTC_BOOTSTRAP': '1'}):
            with self.subTest(change=change), patch.dict(os.environ, change):
                with self.assertRaises(ValueError):
                    provenance.configuration(self.root)

    def test_docker_link_stub_flags_keep_the_same_release_configuration(self):
        stub = self.directory / 'docker-link-stub'
        stub.mkdir()
        cuda = Path(os.environ.get('CHOPINFER_CUDA_HOME', '/usr/local/cuda')).resolve()
        (stub / 'libcuda.so.1').symlink_to(cuda / 'targets/x86_64-linux/lib/stubs/libcuda.so')
        with patch.dict(os.environ, {'RUSTFLAGS': '-L native=' + str(stub)}):
            self.assertEqual(provenance.configuration(self.root), self.record['configuration'])

    def test_missing_host_and_child_kernel_reports_are_rejected(self):
        target = Path(self.result['target'])
        paths = list(target.rglob('kernel-build-report.json'))  # check-inputs: walks outside the repository
        for name in ('sampling_sm120', 'fixture_child'):
            matches = [path for path in paths if json.loads(path.read_text())['cubins'][0]['name'] == name]
            self.assertTrue(matches)
            moved = []
            try:
                for path in matches:
                    backup = path.with_suffix('.saved')
                    path.rename(backup)
                    moved.append((path, backup))
                with self.assertRaisesRegex(ValueError, 'missing kernel report coverage'):
                    provenance.verify_kernel_outputs(self.root, {'fixture': {target}}, self.record['locks'], target)
            finally:
                for path, backup in moved:
                    backup.rename(path)


if __name__ == '__main__':
    unittest.main()
