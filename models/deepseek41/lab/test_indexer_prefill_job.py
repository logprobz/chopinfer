"""The pair must reject missing shards, failures and incomplete timing coverage."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
import indexer_prefill_job as job


def logs():
    result = []
    for shard in (0, 1):
        lines = []
        for i, shape in enumerate(job.shapes()):
            if i % 2 != shard:
                continue
            case = dict(shape, id=i, mismatches=0, negative_control_rejected=True,
                        candidate_planes=shape['layer'] == 20, score_bytes=256 << 20,
                        workspace_bytes=546848768, old_batches=128, new_batches=2)
            lines.append('test prefix ... PREFILL_BATCH_CASE ' + json.dumps(case))
            if not shape['case']:
                for stage in (('all', 'pack_score') if shape['layer'] == 2 else ('all', 'pack_score', 'pack_score_select')):
                    lines.append('test prefix ... PREFILL_BATCH_TIMING ' + json.dumps(dict(
                        id=i, stage=stage, exclusive=True, median_ms=[1., 0.5], paired_ratio=0.5)))
        lines.extend([f'prefix PREFILL_BATCH_COMPLETE shard={shard} shapes=27 total=54',
                      'test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 1 filtered out'])
        result.append('\n'.join(lines))
    return result


class Checks(unittest.TestCase):
    def test_markers_anywhere_including_zero_failed(self):
        self.assertEqual(job.results(logs())['shapes'], 54)

    def test_mutations_fail(self):
        source = logs()
        for before, after in [('shapes=27', 'shapes=26'), ('0 failed', '1 failed'),
                              ('0 ignored', '1 ignored'), ('"id": 0', '"id": 2'),
                              ('"mismatches": 0', '"mismatches": 1'),
                              ('"negative_control_rejected": true', '"negative_control_rejected": false'),
                              ('"paired_ratio": 0.5', '"paired_ratio": NaN'),
                              ('"stage": "all"', '"stage": "missing"')]:
            changed = [source[0].replace(before, after, 1), source[1]]
            self.assertNotEqual(changed, source)
            with self.assertRaises(AssertionError):
                job.results(changed)

    def test_receipts_require_every_sample_and_exclusive_devices(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for i, shape in enumerate(job.shapes()):
                records = []
                for measurement in range(0 if shape['case'] else (2 if shape['layer'] == 2 else 3)):
                    records.append(dict(kind='paired_summary', measurement=measurement, rounds=3))
                    records.extend(dict(measurement=measurement, round=r, arm=a, elapsed_ms=1.) for r in range(3) for a in (0, 1))
                path = Path(tmp) / f'{i}.json'
                path.write_text(json.dumps(dict(id=i, measurement_label='exclusive-device', mismatches=0, timing_rounds=records)))
                paths.append(path)
            job.receipts(paths)
            record = json.loads(paths[0].read_text())
            wrong = copy.deepcopy(record)
            wrong['timing_rounds'].pop()
            paths[0].write_text(json.dumps(wrong))
            with self.assertRaises(AssertionError):
                job.receipts(paths)
            record['measurement_label'] = 'shared-device'
            paths[0].write_text(json.dumps(record))
            with self.assertRaises(AssertionError):
                job.receipts(paths)


if __name__ == '__main__':
    unittest.main()
