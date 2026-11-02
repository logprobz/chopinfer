import unittest

from reply_history_report import analyze, OUTCOMES, ROW_KINDS, CONTEXTS, CALIBRATION


def snapshots(eligible=0, candidate=1000, unknown=0, turns=100):
    first = {'deployment': 'fixture', 'profile': 'tp4-dspark', 'captured_at': 100,
             'metrics': ''}
    last = dict(first, captured_at=200)
    def metrics(final):
        counters = dict.fromkeys(OUTCOMES, 0)
        if final:
            counters.update(submitted=turns, compared=turns, eligible_turn=turns if eligible else 0,
                            any_page=turns, inside_reply=turns, finish_stop=turns)
        rows = {key: [0]*5 for key in ROW_KINDS}
        if final:
            rows['candidate'][2] = candidate
            rows['eligible'][2] = eligible
            rows['prefilled'][2] = candidate + unknown
            rows['unobserved'][2] = unknown
            rows['repeated'][2] = candidate // 2
        values = {'chop_reply_history_enabled': 1, 'chop_reply_history_allocated_bytes': 4096,
                  'chop_reply_history_entries': 20, 'chop_process_uptime_seconds': 200 if final else 100,
                  'chop_prefix_cache_recomputed_generated_tokens_total': sum(rows['repeated'])}
        lines = [f'{name} {value}' for name, value in values.items()]
        lines += [f'chop_reply_history_observations_total{{outcome="{key}"}} {value}' for key, value in counters.items()]
        lines += [f'chop_reply_history_rows_total{{kind="{kind}",context="{bucket}"}} {value}'
                  for kind, row in rows.items() for bucket, value in zip(CONTEXTS, row)]
        return '\n'.join(lines)
    first['metrics'], last['metrics'] = metrics(False), metrics(True)
    return first, last


def change(snapshot, old, new):
    snapshot['metrics'] = snapshot['metrics'].replace(old, new)


class ReplyHistoryReportTest(unittest.TestCase):
    def test_counter_only_zero_opportunity_stops_without_gpu_timing(self):
        report = analyze(*snapshots())
        self.assertEqual(report['estimate']['window_busy_fraction_bounds'], (0, 0))
        self.assertEqual(report['estimate']['window_decision'], 'stop')
        self.assertEqual(report['estimate']['future_decision'], 'inconclusive')
        self.assertEqual(report['totals']['repeated'], 500)
        self.assertEqual(report['totals']['eligible'], 0)
        self.assertEqual(report['rates']['any_page_per_submission']['fraction'], 1)
        self.assertFalse(report['optional_traces']['provided'])

    def test_only_exact_rows_enter_the_estimate(self):
        report = analyze(*snapshots(eligible=100), transfer=[1,1], assumptions='Fixture equal-cost workload.')
        bounds = report['estimate']['window_busy_fraction_bounds']
        self.assertAlmostEqual(bounds[0], CALIBRATION * .1)
        self.assertEqual(bounds[0], bounds[1])
        self.assertEqual(report['estimate']['window_decision'], 'stop')
        self.assertEqual(report['totals']['candidate'], 1000)
        self.assertEqual(report['totals']['prefilled'], 1000)

    def test_cost_uncertainty_crosses_two_percent(self):
        report = analyze(*snapshots(eligible=500), cost_low=[1]*5, cost_high=[2]*5,
                         transfer=[.8,1.2], assumptions='Fixture 2:1 row-cost envelope and 20% transfer uncertainty.')
        low, high = report['estimate']['window_busy_fraction_bounds']
        self.assertLess(low, .02)
        self.assertGreater(high, .02)
        self.assertEqual(report['estimate']['window_decision'], 'inconclusive')
        with self.assertRaisesRegex(ValueError, 'assumptions'):
            analyze(*snapshots(eligible=10), transfer=[1,1])

    def test_absent_transfer_does_not_turn_equal_cost_into_a_bound(self):
        report = analyze(*snapshots(eligible=1))
        self.assertEqual(report['estimate']['equal_cost_share'], .001)
        self.assertEqual(report['estimate']['window_decision'], 'inconclusive')

    def test_missing_history_and_ambiguity_cannot_prove_zero(self):
        first, last = snapshots(unknown=5000)
        report = analyze(first, last, transfer=[1,1], assumptions='Fixture comparable workload.')
        self.assertFalse(report['coverage_complete'])
        self.assertEqual(report['estimate']['window_decision'], 'inconclusive')
        self.assertEqual(report['unknown_candidate_rows_upper'], 5000)
        for name in ('ambiguous', 'retention_loss_possible', 'drop_capacity'):
            first, last = snapshots()
            change(last, f'outcome="{name}"}} 0', f'outcome="{name}"}} 1')
            report = analyze(first, last, transfer=[1,1], assumptions='Fixture comparable workload.')
            self.assertEqual(report['estimate']['window_decision'], 'inconclusive')

    def test_disabled_resets_mixed_deployments_and_missing_metrics_are_rejected(self):
        for name in ('disabled', 'reset', 'identity', 'missing', 'uptime'):
            first, last = snapshots()
            if name == 'disabled': change(last, 'chop_reply_history_enabled 1', 'chop_reply_history_enabled 0')
            if name == 'reset': change(first, 'outcome="submitted"} 0', 'outcome="submitted"} 101')
            if name == 'identity': last['deployment'] = 'another'
            if name == 'missing': change(last, 'chop_reply_history_entries 20', '')
            if name == 'uptime': change(last, 'chop_process_uptime_seconds 200', 'chop_process_uptime_seconds 10')
            with self.subTest(name=name), self.assertRaises(ValueError): analyze(first, last)

    def test_counter_aliases_and_overlapping_rows_are_not_added(self):
        first, last = snapshots()
        change(last, 'chop_prefix_cache_recomputed_generated_tokens_total 500',
               'chop_prefix_cache_recomputed_generated_tokens_total 501')
        with self.assertRaisesRegex(ValueError, 'aliases'): analyze(first, last)
        first, last = snapshots(eligible=1001)
        with self.assertRaisesRegex(ValueError, 'inconsistent'): analyze(first, last)

    def test_context_weights_and_optional_trace_gaps(self):
        first, last = snapshots(eligible=100)
        report = analyze(first, last, cost_low=[1,1,3,1,1], cost_high=[1,1,3,1,1],
                         transfer=[1,1], assumptions='Fixture bucket cost is three.', optional_traces=[None])
        self.assertEqual(report['rows']['64_128k']['eligible'], 100)
        self.assertFalse(report['optional_traces']['complete'])
        self.assertAlmostEqual(report['estimate']['eligible_cost_share_bounds'][0], .1)

    def test_optional_native_traces_need_one_complete_record_per_submission(self):
        traces = [dict(schema='chopinfer.request.v1', engine_request_id=i, prompt_tokens=100,
                       generation_tokens=10, finish_reason='stop') for i in range(100)]
        report = analyze(*snapshots(), optional_traces=traces)
        self.assertTrue(report['optional_traces']['complete'])
        traces[-1] = traces[0]
        report = analyze(*snapshots(), optional_traces=traces)
        self.assertFalse(report['optional_traces']['complete'])

    def test_future_sampling_uncertainty_is_explicit_and_bounded(self):
        report = analyze(*snapshots(turns=10000), transfer=[1,1], assumptions='Fixture independent equal-cost turns.',
                         independent_turns=10000)
        self.assertEqual(report['estimate']['window_decision'], 'stop')
        self.assertGreater(report['estimate']['future_busy_fraction_bounds'][1], 0)
        with self.assertRaisesRegex(ValueError, 'independent'):
            analyze(*snapshots(), independent_turns=101)


if __name__ == '__main__':
    unittest.main()
