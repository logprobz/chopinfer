"""CPU tests for shard coverage and evidence validation."""
import copy
import hashlib
import json
from pathlib import Path
import stat
import tempfile
import unittest
import projection_job as job


def cases():
    return {'typed_count': 3, 'operations': [
        {'capability': {'kind': kind, 'stages': [str(i)], 'usage': {'capacity': 33}},
         'oracle_rows': [0, 1, 17, 33], 'chunk_invariant': True}
        for i, kind in enumerate(['Dense', {'Grouped': {}}, 'Lookup', {'Grouped': {}}])],
        'copies': [{'symbol': 'copy', 'capacity': 33}]}


def receipt(index, entry):
    selector, case = entry
    if selector == job.COPY:
        return dict(case, case_index=index, exact=True)
    return dict(case_index=index, capability=case['capability'], oracle=True, replay=True, eager_vs_graph=True, wrong_output_refused=True)


class Selection(unittest.TestCase):
    def test_default_selects_the_largest_row_set_per_symbol(self):
        inventory = cases()
        inventory['operations'][1] = copy.deepcopy(inventory['operations'][0])
        inventory['operations'][1]['oracle_rows'].append(64)
        inventory['representatives'] = [1, 2, 3, 4]
        self.assertEqual(set(job.selected(inventory)), {1, 2, 3, 4})
        self.assertEqual(set(job.selected(inventory, all_cases=True)), set(range(5)))
        self.assertEqual(set(job.selected(inventory, symbols=['0'])), {1})
        with self.assertRaises(ValueError):
            job.selected(inventory, symbols=['missing'])
        with self.assertRaises(ValueError):
            job.selected(inventory, symbols=[])

    def test_dense_evidence_requires_graph_and_wrong_output_refusal(self):
        inventory = cases()
        inventory['operations'][0]['capability']['usage']['capacity'] = 64
        expected = {0: job.selected(inventory, all_cases=True)[0]}
        record = receipt(0, expected[0])
        for bad in [dict(record, eager_vs_graph=False), dict(record, wrong_output_refused=False)]:
            with self.assertRaises(ValueError):
                job.coverage(expected, [bad])

    def test_explicit_cases_keep_distinct_phase_classes(self):
        inventory = cases()
        inventory['operations'][1] = copy.deepcopy(inventory['operations'][0])
        inventory['operations'][0]['capability']['usage']['phase'] = 'Prefill'
        inventory['operations'][1]['capability']['usage']['phase'] = 'Verify'
        self.assertEqual(set(job.selected(inventory, case_indices=[0, 1])), {0, 1})
        for invalid in [[], [0, 0], [100], [True]]:
            with self.assertRaises(ValueError):
                job.selected(inventory, case_indices=invalid)

    def test_coverage_rejects_missing_duplicate_wrong_and_unassigned_evidence(self):
        expected = job.selected(cases(), all_cases=True)
        records = [receipt(i, entry) for i, entry in expected.items()]
        self.assertTrue(job.coverage(expected, records)['complete'])
        for invalid in [records[:-1], records + records[:1], records + [dict(records[0], case_index=100)], [dict(records[0], oracle=False)] + records[1:], [dict(records[0], capability={})] + records[1:]]:
            with self.assertRaises(ValueError):
                job.coverage(expected, invalid)


if __name__ == '__main__':
    unittest.main()


class DryRun(unittest.TestCase):
    """A dry run checks the prepared manifest as a GPU run does and launches
    nothing: a manifest whose test inventory differs is refused before any
    launch, and a valid one plans one command per needed selector."""

    def prepared(self, root, tests):
        inventory = dict(cases(), representatives=[0, 1, 2, 3, 4])
        (root / 'artifacts').mkdir()
        case_bytes = json.dumps(inventory).encode()
        (root / 'artifacts/cases.json').write_bytes(case_bytes)
        (root / 'target').mkdir()
        binaries = {}
        for name in {name for name, _ in job.SELECTORS}:
            binary = root / 'target' / name
            # Lists every selector; a launch would fail the test loudly.
            listing = ''.join(f'{selector}: test\\n' for n, selector in job.SELECTORS if n == name)
            binary.write_text(f'#!/bin/sh\n[ "$1" = --list ] && printf "{listing}" && exit 0\nexit 99\n')
            binary.chmod(binary.stat().st_mode | stat.S_IEXEC)
            binaries[name] = {'path': str(binary), 'sha256': hashlib.sha256(binary.read_bytes()).hexdigest()}
        (root / 'prepared.json').write_text(json.dumps({
            'revision': 'r', 'binaries': binaries, 'tests': tests,
            'cases_sha256': hashlib.sha256(case_bytes).hexdigest()}))

    def test_dry_run_checks_the_manifest_and_launches_nothing(self):
        projection = {}
        for name, selector in job.SELECTORS:
            projection.setdefault(name, []).append(selector)
        with tempfile.TemporaryDirectory() as temp:
            root, output = Path(temp) / 'prepared', Path(temp) / 'out'
            root.mkdir()
            output.mkdir()
            # Extra selectors beside the projection ones: the inventory differs.
            self.prepared(root, dict(projection, other=['other_test']))
            with self.assertRaisesRegex(ValueError, 'prepared test inventory differs'):
                job.run(root, output, symbols=['0', '1', '2'], dry_run=True)
        with tempfile.TemporaryDirectory() as temp:
            root, output = Path(temp) / 'prepared', Path(temp) / 'out'
            root.mkdir()
            output.mkdir()
            self.prepared(root, projection)
            planned = job.run(root, output, symbols=['0', '1', '2'], dry_run=True)
            self.assertEqual({p['test'] for p in planned}, {job.DENSE[1], job.GROUPED[1], job.LOOKUP[1]})
            self.assertTrue((output / 'dry-run.json').is_file())
            self.assertFalse((output / 'results.json').exists())
