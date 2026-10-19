"""Negative controls for fold completeness, overlap attribution and routing."""
import copy
import unittest
import fold_bandwidth_report as report


def records(ratio=0.9, variant='n8k1'):
    return [dict(n=512, k=5120, groups=1, paired=False, rows=1, variant=variant,
                 device=device, bitwise=True, clock_bitwise=True,
                 **{mode: dict(paired_ratio=ratio, exclusive_paired_ratio=ratio)
                    for mode in report.MODES}) for device in (0, 1)]


class Verdict(unittest.TestCase):
    def test_libtest_requires_one_pass_and_no_skip(self):
        good = 'test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured;'
        report.check_log(good)
        for bad in (good.replace('1 passed', '0 passed'), good+'\nSkipped GPU',
                    good.replace('0 ignored', '1 ignored'), good+'\nFAILED', good+'\n'+good):
            with self.assertRaises(AssertionError):
                report.check_log(bad)

    def test_both_devices_and_modes_must_win(self):
        self.assertEqual(len(report.select(records())), 1)
        for field in ('paired_ratio', 'exclusive_paired_ratio'):
            for value in (0.98, 1.0, 1.0014, 1.03, 1.04):
                pair = records()
                pair[1]['prefetched'][field] = value
                self.assertEqual(report.select(pair), {})
                self.assertTrue(all(r['selected_variant'] == 'baseline' for r in pair))
        for bad in (records()[:1], [records()[0]]*2, records(float('nan'))):
            with self.assertRaises(AssertionError):
                report.select(bad)

    def test_pick_one_variant_and_reject_mismatch(self):
        rows = records(0.9) + records(0.8, 'n8k4')
        self.assertEqual(set(report.select(rows).values()), {'n8k4'})
        self.assertEqual(sum(r['selected'] for r in rows), 2)
        rows[0]['clock_bitwise'] = False
        with self.assertRaises(AssertionError):
            report.select(rows)

    def test_median_interleaved_ratios_and_round_completeness(self):
        rounds = [dict(measurement=2, arm=arm, round=i, elapsed_ms=value,
                       order=[0, 1] if i % 2 == 0 else [1, 0])
                  for i in range(5) for arm, value in enumerate((1.0, 1.0014))]
        _, ratio = report.paired_rounds(rounds)
        self.assertAlmostEqual(ratio, 1.0014)
        for bad in (rounds[:-1], rounds[:-1]+[rounds[0]]):
            with self.assertRaises(AssertionError):
                report.paired_rounds(bad)
        bad = copy.deepcopy(rounds)
        bad[0]['order'] = [1, 0]
        with self.assertRaises(AssertionError):
            report.paired_rounds(bad)

    def test_exclusive_nested_intervals(self):
        events = [(1, 10, 30, 'a'), (2, 20, 25, 'b'), (3, 27, 40, 'c'), (4, 50, 60, 'a')]
        self.assertEqual(report.exclusive(events), {'a': 30, 'b': 0, 'c': 10})

    def test_trace_requires_all_replays_and_right_symbols(self):
        metadata = dict(baseline=14, candidate=31, boundary=8, admission=9, warmup_per_mode=2, rounds_per_mode=5)
        row = dict(baseline_symbols=['old_mxfp8_fold'], candidate_symbols=['chop_mxfp8_direct_test'],
                   trace_replays=metadata, floor_us=2.0, weight_bytes=3_600_000,
                   cold={}, prefetched={})
        def trace(name, count, width):
            return dict(symbols=[name], replays=[[(1, i*10000, i*10000+width, name)] for i in range(count)])
        traces = [trace(row['baseline_symbols'][0], 14, 4000), trace(row['candidate_symbols'][0], 31, 3000)]
        report.add_exclusive([row], traces)
        self.assertEqual(row['cold']['exclusive_paired_ratio'], 0.75)
        self.assertEqual(row['prefetched']['exclusive_floor_ratio'], 1.5)
        for bad in (traces[:1], [traces[0], trace('chop_mxfp8_direct_wrong', 31, 3000)]):
            with self.assertRaises(AssertionError):
                report.add_exclusive([row], bad)


if __name__ == '__main__':
    unittest.main()
