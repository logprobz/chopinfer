"""Verify staged release provenance through the production entry point."""
from copy import deepcopy
import json
import os
import subprocess
import shutil
import tarfile
import unittest
from unittest.mock import patch

from test_release_producer import CompiledProducer, SourceContext, fixture, provenance
import staging


class StagedProvenance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        for name, value in fixture().items():
            setattr(cls, name, value)

    def clone(self, label):
        path = self.directory / self._testMethodName / label
        shutil.copytree(self.prebuilt, path)
        return path

    def write_record(self, prebuilt, record):
        record['build_sha256'] = provenance.build_identity(record)
        for artifact in record['artifacts']:
            artifact['build_sha256'] = record['build_sha256']
        (prebuilt / provenance.RECORD).write_text(json.dumps(record))

    def test_verified_prebuilt_round_trip_uses_the_existing_entry_point(self):
        out = self.directory / 'restaged'
        with patch.dict(os.environ, {'CHOPINFER_BUILD_REVISION': 'a' * 40}):
            result = subprocess.run(['bash', self.root / 'tools/release/build.sh', '--stage-from', self.prebuilt, out],
                                    text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        def contents(directory):
            with tarfile.open(next(directory.glob('*.tar.gz'))) as stream:
                return {entry.name: (entry.mode, stream.extractfile(entry).read())
                        for entry in stream.getmembers() if entry.isfile()}
        self.assertEqual(contents(out), contents(self.directory / 'archive'))
        files = contents(out)
        provenance_name = next(name for name in files if name.endswith('/BUILD_PROVENANCE.json'))
        self.assertEqual(json.loads(files[provenance_name][1]), self.record)
        sums = next(body for name, (_, body) in files.items() if name.endswith('/SHA256SUMS'))
        self.assertIn(b'./BUILD_PROVENANCE.json', sums)

    def test_unrecorded_malformed_or_altered_records_are_rejected(self):
        for label, content in [('missing', None), ('duplicate', '{"schema":1,"schema":2}'),
                               ('altered', json.dumps({**self.record, 'build_sha256': '0' * 64}))]:
            with self.subTest(label=label):
                prebuilt = self.clone(label)
                path = prebuilt / provenance.RECORD
                path.unlink() if content is None else path.write_text(content)
                with self.assertRaises((ValueError, OSError)):
                    staging.verify(self.root, prebuilt)

    def test_every_artifact_is_checked_before_any_staged_code_runs(self):
        for artifact in self.record['artifacts']:
            with self.subTest(artifact=artifact['path']):
                prebuilt = self.clone(artifact['path'])
                path = prebuilt / artifact['path']
                path.write_bytes(path.read_bytes() + b'changed artifact')
                with self.assertRaisesRegex(ValueError, 'artifact bytes differ'):
                    staging.verify(self.root, prebuilt)
        prebuilt = self.clone('executable-control')
        marker = self.directory / 'unexpected-execution'
        import shlex
        (prebuilt / 'chopinfer').write_text('#!/bin/sh\ntouch ' + shlex.quote(str(marker)) + '\nexit 0\n')
        with self.assertRaisesRegex(ValueError, 'artifact bytes differ'):
            staging.stage(self.root, prebuilt, self.directory / 'refused-code')
        self.assertFalse(marker.exists())

    def test_consistently_rebound_incompatible_evidence_is_rejected(self):
        mutations = {
            'revision': lambda r: r['source'].update(revision='b' * 40),
            'profile': lambda r: r['configuration']['profile'].update({'opt-level': 0}),
            'target': lambda r: r['configuration'].update(target='aarch64-unknown-linux-gnu'),
            'artifact-profile': lambda r: r['artifacts'][0]['profile'].update(debug_assertions=True),
            'features': lambda r: r['artifacts'][0].update(features=['diagnostic-taps']),
            'missing-artifact': lambda r: r['artifacts'].pop(),
            'duplicate-artifact': lambda r: r['artifacts'].append(deepcopy(r['artifacts'][0])),
            'traversal': lambda r: r['artifacts'][0].update(path='../chopinfer'),
            'profiled-cubin': lambda r: r['kernels'][0]['identities'][0].update(line_info=True),
            'compiler-profile': lambda r: r['kernels'][0]['compiler']['inputs'].update({'nvcc.profile': '0' * 64}),
        }
        mutations.update({f'status-{status}': lambda r, status=status: r['kernels'][0].update(lock_status=status)
                          for status in ('Updated', 'Profiling', 'Skipped')})
        for name in ('sampling_sm120', 'sampling_mapped_sm120', 'fixture_child'):
            mutations['missing-' + name] = lambda r, name=name: r.update(kernels=[entry for entry in r['kernels']
                if not any(identity['name'] == name for identity in entry['identities'])])
        for label, mutate in mutations.items():
            with self.subTest(label=label):
                prebuilt = self.clone(label)
                record = deepcopy(self.record)
                mutate(record)
                self.write_record(prebuilt, record)
                with self.assertRaises((ValueError, OSError)):
                    staging.verify(self.root, prebuilt)

    def test_symlinks_and_changes_between_verification_and_copy_are_rejected(self):
        prebuilt = self.clone('symlink')
        path = prebuilt / 'libbundle_fixture.so'
        path.unlink()
        path.symlink_to(self.prebuilt / path.name)
        with self.assertRaises(OSError):
            staging.verify(self.root, prebuilt)
        prebuilt = self.clone('copy-race')
        original = staging.copy_artifacts
        def replace(source, destination, record):
            path = source / 'fixture-package'
            path.write_bytes(path.read_bytes() + b'replacement')
            original(source, destination, record)
        with patch.object(staging, 'copy_artifacts', replace), self.assertRaisesRegex(ValueError, 'changed during staging copy'):
            staging.stage(self.root, prebuilt, self.directory / 'refused-race')

    def test_changed_source_and_locks_cannot_reuse_a_producer_record(self):
        for relative in ('models/fixture/bundle/src/lib.rs', 'crates/runtime-kernels/kernels.lock'):
            path = self.root / relative
            original = path.read_bytes()
            try:
                path.write_bytes(original + b'\n ')
                with self.assertRaisesRegex(ValueError, 'source revision or input inventory differs'):
                    staging.verify(self.root, self.prebuilt)
            finally:
                path.write_bytes(original)


if __name__ == '__main__':
    unittest.main()
