"""Bounded IO tests and, when invoked by Qwen, actual full-size lifecycle exports."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools/dev'))
sys.path.insert(0, str(ROOT / 'tools/qual'))
import diagnostics
import lifecycle_diagnostics as lifecycle
import validation_evidence as evidence


def encode(value):
    return json.dumps(value, separators=(',', ':')).encode()


class BoundedReader(unittest.TestCase):
    def test_no_follow_including_parents_and_special_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'real').mkdir()
            (root / 'real/data').write_text('{}')
            (root / 'alias').symlink_to(root / 'real', target_is_directory=True)
            (root / 'link').symlink_to(root / 'real/data')
            os.mkfifo(root / 'fifo')
            for path in (root / 'alias/data', root / 'link', root / 'fifo'):
                with self.subTest(path=path.name), self.assertRaises((ValueError, OSError)):
                    lifecycle.read_json(path)
            self.assertEqual(lifecycle.read_json(root / 'real/data'), {})

    def test_metadata_bound_duplicate_fields_and_trailing_data(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'input.json'
            for raw in ('{"a":1,"a":2}', '{}{}', '{"a":NaN}'):
                path.write_text(raw)
                with self.assertRaises(ValueError):
                    lifecycle.read_json(path)
            path.write_bytes(b' ' * 1024)
            with patch.object(lifecycle, 'METADATA_LIMIT', 16), self.assertRaises(ValueError):
                lifecycle.read_json(path)

    def test_repository_paths_are_refused(self):
        with self.assertRaises(ValueError):
            lifecycle.read_json(ROOT / 'Cargo.toml')

    def test_stream_bound_and_unterminated_records(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'events'
            for data in (b'{}', b'{}\n'*20):
                path.write_bytes(data)
                with patch.object(lifecycle, 'DATA_LIMIT', 32), self.assertRaises(ValueError):
                    list(lifecycle.lines(path))


if 'CHOPINFER_LIFECYCLE_TEST_CAPTURE' in os.environ:
    class RecordedLifecycle(unittest.TestCase):
        @classmethod
        def setUpClass(cls):
            cls.root = Path(os.environ['CHOPINFER_LIFECYCLE_TEST_CAPTURE'])
            cls.baseline = cls.root / 'baseline'
            cls.cli = ROOT / 'tools/dev/diagnostics.py'

        def setUp(self):
            self.temporary = tempfile.TemporaryDirectory(prefix='lifecycle-negative-')
            self.work = Path(self.temporary.name)

        def tearDown(self):
            self.temporary.cleanup()

        def variant(self, source='matching'):
            target = self.work / source
            # Copy from this test's external captures. No shared inode is mutated.
            shutil.copytree(self.root / source, target)
            return target

        def meta(self, path, change):
            name = path / 'diagnostic.json'
            value = json.loads(name.read_bytes())
            change(value)
            name.write_bytes(encode(value))

        def artifact(self, path, name, change):
            file = path / name
            file.write_bytes(change(file.read_bytes()))
            self.meta(path, lambda v: v['artifacts'].__setitem__(name, lifecycle.digest(file, lifecycle.DATA_LIMIT)))

        def report(self, candidate, status, reference=None):
            output = subprocess.run([sys.executable, '-B', str(self.cli), 'compare', '--mode', 'journal',
                                     '--reference', str(reference or self.baseline), '--candidate', str(candidate)],
                                    capture_output=True, text=True)
            self.assertEqual(output.returncode, diagnostics.CODES[status], output.stdout + output.stderr)
            report = json.loads(output.stdout)
            self.assertEqual(report['status'], status)
            return report

        def test_baseline_repeat_full_journal_and_preserved_trigger(self):
            for name in ('control', 'matching'):
                report = self.report(self.root / name, 'match')
                self.assertEqual(report['completed_actions'], 9)
                self.assertEqual(report['device']['execution'], 'not_run')
            capture = lifecycle.Capture(self.baseline)
            self.assertGreater(capture.value['artifacts']['journal.txt']['bytes'], evidence.MAX_BYTES)
            self.assertEqual(capture.value['events'],
                             int(os.environ.get('CHOPINFER_LIFECYCLE_TEST_EVENTS', '19768')))
            self.assertEqual(capture.input['settings']['served_capacity'], 32)
            self.assertIs(capture.input['settings']['effective_hc'], True)
            self.assertEqual([len(s['rows']) for s in capture.value['workload']], [2, 2, 3])
            mixed = capture.value['workload'][2]['rows']
            self.assertEqual([(r['request'], r['slot'], r['context']) for r in mixed], [(22, 2, 0), (20, 0, 4), (21, 1, 2051)])
            self.assertEqual(mixed[2]['pages'], list(range(64, 97)))
            events = list(capture.events())
            result = [e for e in events if e['kind'] == 'result']
            self.assertEqual([e['pending'] for e in result], [False, True, False, True, False, True, False, False, None])
            self.assertEqual(result[6]['owners'], [None]*3)
            graph = [(e['action'], e['service']) for e in events if e['kind'] == 'journal' and e['service'] in ('capture', 'captured', 'replay')]
            self.assertTrue(graph)
            self.assertIn('replay', [s for _, s in graph])
            # Byte-for-byte journal equality also covers all setup and eager warmup calls.
            self.assertEqual(capture.value['artifacts']['journal.txt'], lifecycle.Capture(self.root / 'matching').value['artifacts']['journal.txt'])

        def test_executed_failure_and_repeat_first_divergence(self):
            reports = [self.report(self.root / name, 'diverged') for name in ('failure', 'repeat-failure')]
            self.assertEqual(reports[0]['first'], reports[1]['first'])
            first = reports[0]['first']
            self.assertEqual((first['action'], first['operation'], first['stage']), (6, 'cancel', 'copy_result'))
            self.assertEqual(first['expected']['status'], 'Success')
            self.assertEqual(first['observed']['status'], 'HostError')
            self.assertEqual(first['observed']['rank'], 1)
            self.assertEqual(first['expected']['source'], first['observed']['source'])
            self.assertEqual(first['expected']['destination'], first['observed']['destination'])
            self.assertEqual(reports[0]['teardown']['status'], 'Poisoned')
            self.assertEqual(reports[0]['completion'], 'closed-failure')

        def test_reported_reproduction_executes_and_checks_pins(self):
            report = self.report(self.root / 'failure', 'diverged')
            # The recorded command names a fresh output and the original immutable input.
            reproduced = self.root / 'failure-reproduced'
            self.assertFalse(reproduced.exists())
            output = subprocess.run(report['reproduction'], capture_output=True, text=True)
            self.assertEqual(output.returncode, 0, output.stdout + output.stderr)
            self.assertIn('1 passed', output.stdout)
            self.assertEqual(self.report(reproduced, 'diverged')['first'], report['first'])
            for key in ('INPUT_SHA256', 'BINARY_SHA256', 'REFERENCE_SHA256', 'CONTROL_SHA256'):
                command = [arg.split('=', 1)[0] + '=' + '0'*64
                           if arg.startswith('CHOPINFER_QWEN38_LIFECYCLE_' + key + '=') else arg
                           for arg in report['reproduction']]
                output = subprocess.run(command, capture_output=True, text=True)
                self.assertNotEqual(output.returncode, 0)
                self.assertIn('stale', output.stdout + output.stderr)
            output = subprocess.run(report['reproduction'], capture_output=True, text=True)
            self.assertNotEqual(output.returncode, 0)  # No overwriting captures.

        def test_incompatible_inputs_configuration_environment_and_producer(self):
            for field in ('configuration', 'environment', 'producer', 'tokens'):
                with self.subTest(field=field):
                    path = self.variant()
                    inp = json.loads((path / 'input.json').read_bytes())
                    if field == 'configuration':
                        inp['configuration'][0] ^= 1
                    elif field == 'environment':
                        inp['settings']['environment']['CHOPINFER_QWEN38_HC_PRECOMPUTE'] = '0'
                    elif field == 'producer':
                        inp['producer']['binary_sha256'] = '0'*64
                    else:
                        # One token byte inside the first V3 item, preserving shape and mapping.
                        inp['actions'][1]['wire'][50] ^= 1
                    self.artifact(path, 'input.json', lambda _: encode(inp))
                    def update(v):
                        v['identity'] = dict(input_sha256=v['artifacts']['input.json']['sha256'], settings=inp['settings'])
                        v['producer'] = inp['producer']
                    self.meta(path, update)
                    self.report(path, 'incompatible')
                    shutil.rmtree(path)

        def test_exporter_rejects_unsupported_incomplete_and_oversize_inputs(self):
            command = json.loads((self.root / 'failure/diagnostic.json').read_bytes())['reproduction']
            original = (self.root / 'input.json').read_bytes()
            for mutation in ('event', 'order', 'context', 'actions', 'trailing', 'limit', 'missing'):
                with self.subTest(mutation=mutation):
                    value = json.loads(original)
                    if mutation == 'event':
                        value['actions'][6]['kind'] = 'preempt'
                    elif mutation == 'order':
                        value['actions'].reverse()
                    elif mutation == 'context':
                        # The enclosing length remains valid; the SDK rejects changed shape.
                        value['actions'][5]['wire'][-8] ^= 1
                    elif mutation == 'actions':
                        value['actions'] *= 8
                    raw = encode(value)
                    if mutation == 'trailing':
                        raw += b'{}'
                    elif mutation == 'limit':
                        raw = b' ' * (lifecycle.METADATA_LIMIT + 1)
                    path = self.work / ('input-' + mutation)
                    if mutation != 'missing':
                        path.write_bytes(raw)
                    output_path = self.work / ('output-' + mutation)
                    replacements = {'INPUT': str(path), 'INPUT_SHA256': hashlib.sha256(raw).hexdigest(), 'OUTPUT': str(output_path)}
                    args = []
                    for arg in command:
                        for key, replacement in replacements.items():
                            prefix = 'CHOPINFER_QWEN38_LIFECYCLE_' + key + '='
                            if arg.startswith(prefix):
                                arg = prefix + replacement
                                break
                        args.append(arg)
                    result = subprocess.run(args, capture_output=True, text=True)
                    self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                    self.assertFalse(output_path.exists(), 'invalid inputs must be refused before model creation')

        def test_unsupported_reordered_and_over_budget_actions(self):
            for change in (lambda a: a[6].__setitem__('kind', 'restore_prefix'),
                           lambda a: a.reverse(), lambda a: a.extend(a*8)):
                path = self.variant()
                inp = json.loads((path / 'input.json').read_bytes())
                change(inp['actions'])
                self.artifact(path, 'input.json', lambda _: encode(inp))
                self.meta(path, lambda v: v['identity'].__setitem__('input_sha256', v['artifacts']['input.json']['sha256']))
                self.report(path, 'insufficient')
                shutil.rmtree(path)

        def test_missing_reordered_and_trailing_observations(self):
            for mutation in ('probe', 'result', 'order', 'trailing', 'copy_status'):
                path = self.variant('failure')
                def change(raw):
                    events = [json.loads(line) for line in raw.splitlines()]
                    index = next(i for i, e in enumerate(events) if e['kind'] == ('copy_result' if mutation == 'copy_status' else 'result'))
                    if mutation == 'probe':
                        del events[index]['owners']
                    elif mutation in ('result', 'copy_status'):
                        del events[index]
                    elif mutation == 'order':
                        events[0], events[1] = events[1], events[0]
                    else:
                        events.append(events[-1])
                    return b''.join(encode(e)+b'\n' for e in events)
                self.artifact(path, 'observations.jsonl', change)
                self.report(path, 'insufficient')
                shutil.rmtree(path)

        def test_incomplete_publication_hashes_and_byte_limits(self):
            for mutation in ('terminal', 'truncated', 'trailing', 'completion', 'budget', 'missing_input'):
                path = self.variant('failure')
                if mutation == 'terminal':
                    (path / 'diagnostic.json').unlink()
                elif mutation == 'missing_input':
                    (path / 'input.json').unlink()
                elif mutation == 'completion':
                    self.meta(path, lambda v: v.__setitem__('completion', 'interrupted'))
                elif mutation == 'budget':
                    self.meta(path, lambda v: v['artifacts']['journal.txt'].__setitem__('bytes', lifecycle.DATA_LIMIT))
                else:
                    file = path / 'journal.txt'
                    raw = file.read_bytes()
                    file.write_bytes(raw[:-1] if mutation == 'truncated' else raw+b'extra\n')
                self.report(path, 'insufficient')
                shutil.rmtree(path)
            path = self.variant()
            with patch.object(lifecycle, 'METADATA_LIMIT', 64):
                self.assertEqual(lifecycle.compare(self.baseline, path)['status'], 'insufficient')

        def test_stale_baseline_control_and_changed_repeat(self):
            for key in ('reference', 'control'):
                path = self.variant()
                self.meta(path, lambda v: v[key].__setitem__('sha256', '0'*64))
                self.report(path, 'insufficient')
                shutil.rmtree(path)
            control = self.variant('control')
            self.artifact(control, 'observations.jsonl', lambda raw: raw.replace(b'"owners":[20,21,null]', b'"owners":[99,21,null]', 1))
            path = self.variant()
            self.meta(path, lambda v: v.__setitem__('control', dict(path=str(control), sha256=lifecycle.digest(control / 'diagnostic.json')['sha256'])))
            self.report(path, 'insufficient')

        def test_validation_keeps_oversize_artifact_unverified_and_device_missing(self):
            records = evidence.load(self.baseline / 'diagnostic.json')
            self.assertEqual(len(records), 1)
            record = records[0]
            self.assertEqual(record.kind, 'fake_host')
            self.assertTrue(any('unverified' in issue and 'journal.txt' in issue for issue in record.issues), record.issues)
            self.assertTrue(any('idle-device' in issue for issue in record.issues), record.issues)
            self.assertNotIn('devices', record.observed['candidate'])
            self.assertEqual(evidence.MAX_BYTES, 8_000_000)
            self.assertFalse(evidence.assess(record, dict(id='qwen:lifecycle', kind='fake_host'), {})['satisfied'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
