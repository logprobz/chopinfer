"""CPU tests for eviction receipt statistics."""
import unittest
import contextlib
import io
import json
from pathlib import Path
import tempfile

from eviction_table import main, summarize


class EvictionTableTests(unittest.TestCase):
    def samples(self):
        return [
            dict(measurement=7, operation="target", round=r, arm=arm, elapsed_ms=value,
                 gpu_state={'flags': ['sm_clock_drift'] if r == 0 else []})
            for r, pair in enumerate([(1, 50), (10, 2), (100, 30)])
            for arm, value in enumerate(pair)
        ]

    def test_pairs_preserve_rounds_and_count_flags(self):
        samples = self.samples()
        samples.append(dict(kind='paired_summary', measurement=7))
        self.assertEqual(summarize(samples), [(7, "target", 10000, 30000, .3, 2)])

    def test_calibration_keeps_own_label(self):
        samples = self.samples()
        for sample in samples:
            sample['operation'] = 'launch_floor'
        with tempfile.TemporaryDirectory() as root:
            Path(root, 'kernel.json').write_text(json.dumps({
                'symbol': 'kernel_entry', 'timing_rounds': samples,
            }))
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                main(root)
            self.assertEqual(output.getvalue().splitlines()[1].split('\t')[1], 'launch_floor')

    def test_incomplete_pairs_fail(self):
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            summarize(self.samples()[:-1])

    def test_unpaired_timings_are_excluded(self):
        self.assertEqual(summarize([dict(round=0, elapsed_ms=1)]), [])


if __name__ == '__main__':
    unittest.main()
