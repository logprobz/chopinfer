import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import qwen38_e5_diagnostics as d


class DiagnosticsTest(unittest.TestCase):
    def test_changed_or_ambiguous_source_anchors_are_refused(self):
        self.assertEqual(d.replace('abc', 'b', 'x'), 'axc')
        for source in ('abc', 'xx'):
            with self.assertRaisesRegex(ValueError, 'source anchor changed'):
                d.replace(source, 'x', 'y')

    def test_hc_generator_accepts_current_source_and_keeps_block_accumulation(self):
        source = d.hc_source(Path(__file__).resolve().parents[2])
        for variant in d.HC:
            self.assertIn('e5_hc_' + variant + '(', source)
        self.assertIn('bool Precomputed = false, bool Padded = false', source)
        self.assertIn('ptx::mma_bf16(block[pair * 2]', source)
        self.assertIn('acc[j].v[v] += block[j].v[v];', source)
        self.assertIn('block[j].v[v] = 0.0f;', source)
        self.assertIn('diagnostic_sink ^=', source)
        self.assertIn('run_hc_up_mix<32,0,true>', source)

    def test_resource_report_requires_every_diagnostic_entry(self):
        symbols = ['e5_hc_' + v for v in d.HC] + ['e5_delta_' + v for v in d.DELTA]
        raw = '\n'.join(f' Function {s}:\n  REG:76 STACK:0 SHARED:1024 LOCAL:0' for s in symbols)
        self.assertEqual(len(d.resource_report(raw)), 10)
        with self.assertRaisesRegex(ValueError, 'incomplete diagnostic'):
            d.resource_report(raw.replace('e5_hc_trace', 'unrelated'))

    def test_loaded_resources_separate_driver_reserved_shared_memory(self):
        resource = {'REG': 170, 'STACK': 0, 'SHARED': 1024, 'LOCAL': 0}
        prepared = {'path': '/data/diagnostic-test.cubin',
                    'resources': {'e5_delta_trace': resource}}
        driver = Mock()
        def device_attribute(out, attribute, device):
            out._obj.value = {39: 1536, 16: 188, 111: 1024}[attribute]
            return 0
        driver.cuDeviceGetAttribute.side_effect = device_attribute
        def function_attribute(out, attribute, function):
            out._obj.value = 170 if attribute == 4 else 0
            return 0
        driver.cuFuncGetAttribute.side_effect = function_attribute
        def blocks(out, *args):
            out._obj.value = 2
            return 0
        driver.cuOccupancyMaxActiveBlocksPerMultiprocessor.side_effect = blocks
        def loaded(out, *args):
            out._obj.value = 1
            return 0
        for name in ('cuDeviceGet', 'cuDevicePrimaryCtxRetain', 'cuModuleLoad', 'cuModuleGetFunction'):
            getattr(driver, name).side_effect = loaded
        for name in ('cuInit', 'cuCtxSetCurrent', 'cuModuleUnload', 'cuDevicePrimaryCtxRelease_v2'):
            getattr(driver, name).return_value = 0
        with patch.object(d.C, 'CDLL', return_value=driver):
            record = d.occupancy(prepared)['kernels']['e5_delta_trace']
        self.assertEqual(record['SHARED'], 0)
        self.assertEqual(record['cubin_shared_bytes'], 1024)
        self.assertEqual(record['driver_reserved_shared_bytes'], 1024)
        self.assertEqual(record['max_active_blocks_per_sm'], 2)
        driver.cuModuleUnload.assert_called_once()
        driver.cuDevicePrimaryCtxRelease_v2.assert_called_once()

    def test_unexplained_loaded_resource_differences_still_fail(self):
        resource = {'REG': 170, 'STACK': 0, 'SHARED': 1024, 'LOCAL': 0}
        self.assertEqual(d.loaded_resources('trace', resource, 170, 1024, 1024)['SHARED'], 1024)
        for registers, shared, reserved in ((169, 0, 1024), (170, 512, 1024), (170, 0, 0)):
            with self.subTest(registers=registers, shared=shared, reserved=reserved):
                with self.assertRaisesRegex(ValueError, 'loaded_REG=.*driver_reserved_SHARED='):
                    d.loaded_resources('trace', resource, registers, shared, reserved)

    def test_cpu_resource_preflight_rejects_stale_file_or_report(self):
        symbols = ['e5_hc_' + v for v in d.HC] + ['e5_delta_' + v for v in d.DELTA]
        raw = '\n'.join(f' Function {s}:\n  REG:76 STACK:0 SHARED:1024 LOCAL:0' for s in symbols)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'diagnostics.cubin'
            path.write_bytes(b'cubin fixture')
            prepared = {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                        'resources': d.resource_report(raw)}
            with patch.object(d.subprocess, 'check_output', return_value=raw), \
                    patch.object(d.C, 'CDLL', side_effect=AssertionError('CPU preflight touched CUDA')):
                d.verify_resources(prepared)
                prepared['resources']['e5_delta_trace']['REG'] += 1
                with self.assertRaisesRegex(ValueError, 'CPU resource report changed'):
                    d.verify_resources(prepared)
                path.write_bytes(b'different cubin')
                with self.assertRaisesRegex(ValueError, 'diagnostic cubin changed'):
                    d.verify_resources(prepared)

    def fixture(self, root):
        folder = root / 'receipts/qwen38-e5-diagnostic'
        folder.mkdir(parents=True)
        baseline = {}
        for shape, mode in [('16', 'mix'), ('16', 'chain'), ('32', 'mix'), ('32', 'chain'),
                            ('16x1', 'delta'), ('32x1', 'delta'), ('16x2', 'delta')]:
            family = 'delta' if mode == 'delta' else 'hc'
            variants = d.DELTA if family == 'delta' else d.HC
            baseline[(shape, mode)] = 10
            for variant in (*variants, 'control_after'):
                record = {'family': family, 'shape': shape, 'rows': shape, 'mode': mode,
                          'variant': variant, 'diagnostic_only': True,
                          'measurement_label': 'exclusive-device', 'matches_shipped_bits': True,
                          'median_us_per_call': 7 if variant in ('no_bottleneck', 'stream_rows') else 10}
                if variant == 'trace':
                    record['trace'] = {'cta_stamps_ns': [[1, 2]], 'intervals': []}
                (folder / f'{shape}-{mode}-{variant}.json').write_text(json.dumps(record))
        resources = {f'e5_hc_{v}': {'REG': 118} for v in d.HC}
        resources.update({f'e5_delta_{v}': {'REG': 76 if v == 'stream_rows' else 163} for v in d.DELTA})
        (root / 'occupancy.json').write_text(json.dumps({'sm_count': 188, 'max_threads_per_sm': 1536,
            'kernels': {symbol: dict(value, max_active_blocks_per_sm=3) for symbol, value in resources.items()}}))
        return folder, {'resources': resources}, baseline

    def test_decision_requires_all_shapes_controls_and_traces(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder, prepared, baseline = self.fixture(root)
            result = d.decide(root, prepared, baseline)
            self.assertEqual(result['diagnostic_receipts'], 42)
            self.assertFalse(result['serving_change_qualified'])
            self.assertTrue(all('bottleneck' in x['decision'] for x in result['cases'] if x['mode'] != 'delta'))
            path = folder / '16x2-delta-trace.json'
            record = json.loads(path.read_text())
            record.pop('trace')
            path.write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError, 'globaltimer'):
                d.decide(root, prepared, baseline)
            path.unlink()
            with self.assertRaisesRegex(ValueError, 'incomplete diagnostic variants'):
                d.decide(root, prepared, baseline)

    def test_nonexact_streaming_variant_cannot_be_recommended(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder, prepared, baseline = self.fixture(root)
            path = folder / '16x2-delta-stream_rows.json'
            record = json.loads(path.read_text())
            record['matches_shipped_bits'] = False
            path.write_text(json.dumps(record))
            case = next(x for x in d.decide(root, prepared, baseline)['cases'] if x['shape'] == '16x2')
            self.assertIn('Keep the current', case['decision'])

    def test_control_drift_refuses_a_performance_decision(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder, prepared, baseline = self.fixture(root)
            path = folder / '32-chain-control_after.json'
            record = json.loads(path.read_text())
            record['median_us_per_call'] = 14
            path.write_text(json.dumps(record))
            case = next(x for x in d.decide(root, prepared, baseline)['cases'] if x['shape'] == '32' and x['mode'] == 'chain')
            self.assertIn('Repeat the measurement', case['decision'])


if __name__ == '__main__':
    unittest.main()
