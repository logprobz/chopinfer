import copy
import unittest
import moe_prefill_job as job


def fixture():
    records = []
    for experts, top_k in [(384, 6), (128, 3)]:
        for rows in [512, 2048, 4096]:
            for routing in ['All', 'Seeded', 'Concentrated']:
                counts = ([rows] * top_k + [0] * (experts - top_k) if routing == 'Concentrated'
                          else [rows * top_k // experts] * experts)
                records.append({'case': f'e{experts}k{top_k}-Prefill-{rows}-{routing}',
                    'rows': rows, 'experts': experts, 'top_k': top_k, 'routing': routing,
                    'counts': counts, 'active_experts': sum(n > 0 for n in counts), 'ranks': 4,
                    'selection': 'old prefill', 'bitwise': True, 'negative_detected': True,
                    'compared_outputs': 15, 'live_replays': [rows, rows//2+1, 0, rows],
                    'calibration': {'bf16_tflops': 100., 'dram_gb_s': 1600., 'l2_gb_s': 8000., 'required_power_limit_w': 250},
                    'timing': {p: {'ms': [4., 2.], 'round_samples_ms': [[[4.] * 5, [2.] * 5] for _ in range(5)],
                                   'replays_per_sample': 2} for p in job.PARTS}})
    return records


class PrefillJobTests(unittest.TestCase):
    def test_complete_case_matrix_and_table(self):
        records = fixture()
        self.assertEqual(job.check_records(records), [])
        table = job.speed_table(records)
        for record in records:
            self.assertIn(record['case'], table)
        self.assertIn('input_quant=', table)
        self.assertIn('router_projection=', table)
        self.assertIn('combine=', table)

    def test_closed_form_flops_bytes_and_tile_utilization(self):
        record = fixture()[0]
        roof = job.roofline(record)
        self.assertEqual(roof['logical_flops']['gateup'], 4 * 512 * 6 * 5120 * 576)
        self.assertEqual(roof['unique_routed_weight_bytes'], 384 * 3 * 5120 * 576 * 17/32)
        self.assertEqual(roof['padded_rows'], 384 * 64)
        self.assertEqual(roof['grouped_issued_rows'], 512 * 6)
        self.assertEqual(roof['old_row_utilization'], 1/8)
        self.assertEqual(roof['tp4_logical_flops']['full'], 4 * roof['logical_flops']['full'])
        wide = next(r for r in fixture() if r['experts'] == 128 and r['rows'] == 4096 and r['routing'] == 'All')
        roof = job.roofline(wide)
        self.assertEqual(roof['padded_rows'], 128 * 128)
        self.assertEqual(roof['grouped_issued_rows'], 128 * 96)
        self.assertEqual(roof['issued_routed_bf16_flops']['grouped']['down'], 2 * 128 * 96 * 5120 * 640)

    def test_missing_parity_control_and_case_are_rejected(self):
        records = fixture()
        for field, value in [('negative_detected', False), ('bitwise', False), ('compared_outputs', 12),
                             ('selection', 'grouped'), ('live_replays', [512])]:
            bad = copy.deepcopy(records)
            bad[0][field] = value
            self.assertTrue(job.check_records(bad), field)
        self.assertTrue(job.check_records(records[:-1]))
        self.assertTrue(job.check_records(records + records[:1]))

    def test_timing_uses_round_medians_and_retains_a_noise_witness(self):
        records = fixture()
        timing = records[0]['timing']['full']
        timing['round_samples_ms'][0][0] = [4., 4., 4., 100., 100.]
        self.assertEqual(job.check_records(records), [])
        timing['ms'][0] = 100.
        self.assertTrue(job.check_records(records))

    def test_missing_stage_bad_calibration_and_route_count_are_rejected(self):
        records = fixture()
        for modify in [lambda r: r['timing'].pop('routing'),
                       lambda r: r['calibration'].update(bf16_tflops=float('nan')),
                       lambda r: r['counts'].__setitem__(0, 0),
                       lambda r: r['timing']['full']['round_samples_ms'].pop()]:
            bad = copy.deepcopy(records)
            modify(bad[0])
            self.assertTrue(job.check_records(bad))


if __name__ == '__main__':
    unittest.main()
