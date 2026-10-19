"""Marker coverage and failure parsing without a GPU."""
import json
import unittest
import indexer_guard_job as job


def logs():
    out = []
    for shard in (0, 1):
        lines = ['running 3 tests', f'test prefix ... GUARD_MATRIX_COMPLETE shard={shard} shapes={47-shard} total=93',
                 f'test prefix ... GUARD_REAL_COMPLETE shard={shard} captures=15',
                 f'test prefix ... SERVING_MMA_COMPLETE shard={shard} comparisons=150']
        for i in range(shard, 123, 2):
            capture = i >= 93
            shape = dict(capture=i-93, context=(8192,32768,131072)[(i-93)//10], rows=32) if capture else dict(matrix=i, case=max(0, i%31-23))
            control = i in (24,55,86)
            case = dict(shape=shape, guard_mismatches=0, negative_control_rejected=control,
                        admitted=100, fallback_slots=int(control), fallback_rows=0, unguarded_score_mismatches=int(control))
            lines.append('test prefix ... GUARD_CASE ' + json.dumps(case))
            if capture or shape['case'] == 0:
                for pair in ([0,2],[1,2]):
                    lines.append('prefix GUARD_TIMING ' + json.dumps(dict(shape=shape, arms=pair,
                                median_ms=[1.0,1.1], exclusive=True)))
        lines.append('test result: ok. 3 passed; 0 failed; 0 ignored; 0 measured; 1 filtered out; finished in 1.0s')
        out.append('\n'.join(lines))
    return out


class Checks(unittest.TestCase):
    def test_markers_anywhere_and_shard_sums(self):
        result = job.results(logs())
        self.assertEqual(result['matrix_shapes'], 93)
        self.assertEqual(set(result['real_fallback_frequency']), {'8192','32768','131072'})

    def test_missing_shape_failing_test_and_missing_control_are_rejected(self):
        source = logs()
        for before, after in [('shapes=47','shapes=46'),('0 failed','1 failed'),
                              ('"matrix": 0','"matrix": 2'),('"guard_mismatches": 0','"guard_mismatches": 1'),
                              ('"negative_control_rejected": true','"negative_control_rejected": false')]:
            changed = [source[0].replace(before, after, 1), source[1]]
            self.assertNotEqual(changed, source)
            with self.assertRaises(AssertionError):
                job.results(changed)


if __name__ == '__main__':
    unittest.main()
