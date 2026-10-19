import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import prefill_attention_job as job
import prefill_attention_resources as resources


REQUEST = dict(symbol='aligned_dynamic', module='probe', registers=180,
               shared_static_bytes=1024, local_bytes=0, stack_bytes=0,
               max_threads=128, min_blocks=2, dynamic_shared_bytes=42240)
ATTRIBUTES = dict(max_threads=128, static_shared_bytes=0, local_bytes=0,
                  registers=180, max_dynamic_shared_bytes=42240)


class FakeDriver:
    """Replay the reported query values without loading a driver library."""
    def __init__(self, attributes=None, blocks=2):
        self.attributes = dict(ATTRIBUTES if attributes is None else attributes)
        self.blocks = blocks
        self.ordinals = []
        self.released = []

    def __getattr__(self, name):
        def call(*args):
            if name == 'cuDeviceGetCount':
                args[0]._obj.value = 2
            elif name == 'cuDeviceGet':
                self.ordinals.append(args[1])
                args[0]._obj.value = args[1]
            elif name in ('cuDevicePrimaryCtxRetain', 'cuModuleLoad', 'cuModuleGetFunction'):
                args[0]._obj.value = 123
            elif name == 'cuDevicePrimaryCtxRelease_v2':
                self.released.append(args[0].value)
            elif name == 'cuDeviceGetAttribute':
                args[0]._obj.value = {16:64, 38:16 << 20, 39:1536}[args[1]]
            elif name == 'cuFuncGetAttribute':
                field = {0:'max_threads', 1:'static_shared_bytes', 3:'local_bytes',
                         4:'registers', 8:'max_dynamic_shared_bytes'}[args[1]]
                args[0]._obj.value = self.attributes[field]
            elif name == 'cuOccupancyMaxActiveBlocksPerMultiprocessor':
                assert args[2:] == (128, 42240)
                args[0]._obj.value = self.blocks
            elif name not in ('cuInit', 'cuCtxSetCurrent', 'cuFuncSetAttribute', 'cuModuleUnload'):
                raise AssertionError(name)
            return 0
        return call


class ResourceAccounting(unittest.TestCase):
    def collect(self, driver):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stderr(io.StringIO()):
            root = Path(directory)
            (root / 'resource-requests.json').write_text(json.dumps([REQUEST]))
            return resources.collect(root, driver)

    def test_compiler_section_and_driver_static_bytes_are_distinct(self):
        driver = FakeDriver()
        result = self.collect(driver)
        for rank in result['ranks']:
            function = rank['functions'][0]
            self.assertEqual(function['shared_static_bytes'], 1024)
            self.assertEqual(function['driver']['static_shared_bytes'], 0)
            self.assertEqual(function['active_ctas_per_sm'], 2)
            self.assertAlmostEqual(function['theoretical_warp_occupancy_percent'], 100 / 6)
        self.assertEqual(driver.ordinals, [0,1])
        self.assertEqual(driver.released, [0,1])

    def test_wrong_registers_limits_spills_and_zero_residency_still_fail(self):
        for field, value in [('registers',181), ('max_threads',64), ('local_bytes',8),
                             ('max_dynamic_shared_bytes',40000), ('static_shared_bytes',-1)]:
            with self.subTest(field=field), self.assertRaises(AssertionError):
                self.collect(FakeDriver(dict(ATTRIBUTES, **{field:value})))
        with self.assertRaises(AssertionError):
            self.collect(FakeDriver(blocks=0))


class StepLogs(unittest.TestCase):
    def test_failure_keeps_driver_traceback_and_status(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            script = "import sys; print('rank=1'); raise RuntimeError('cuFuncGetAttribute returned 1')"
            with self.assertRaisesRegex(job.StepError, 'resources.stderr.log'):
                job.run_step(out, 'resources', [sys.executable, '-c', script])
            self.assertIn('rank=1', (out / 'resources.stdout.log').read_text())
            self.assertIn('cuFuncGetAttribute returned 1', (out / 'resources.stderr.log').read_text())
            self.assertEqual(json.loads((out / 'resources.step.json').read_text())['exit_code'], 1)

    def test_success_keeps_stderr_and_device_mask(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            for mask in ['0,1', '2,3']:
                script = "import os,sys; print(os.environ['CUDA_VISIBLE_DEVICES']); print('diagnostic', file=sys.stderr)"
                stdout, code = job.run_step(out, 'success', [sys.executable, '-c', script],
                                           env=dict(os.environ, CUDA_VISIBLE_DEVICES=mask))
                self.assertEqual((stdout.strip(), code), (mask, 0))
                self.assertEqual((out / 'success.stderr.log').read_text(), 'diagnostic\n')

    def test_timeout_keeps_partial_logs_and_reaps_child(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            script = "import sys,time; print('started', file=sys.stderr, flush=True); time.sleep(60)"
            with self.assertRaises(job.StepError):
                job.run_step(out, 'timeout', [sys.executable, '-c', script], timeout=1)
            record = json.loads((out / 'timeout.step.json').read_text())
            self.assertGreaterEqual(record['ended_unix_seconds'],record['started_unix_seconds'])
            self.assertIn('TimeoutExpired', record['error'])
            self.assertLess(record['exit_code'], 0)
            self.assertIn('started', (out / 'timeout.stderr.log').read_text())
            with self.assertRaises(ChildProcessError):
                os.waitpid(record['pid'], os.WNOHANG)

    def test_missing_executable_still_has_stderr_and_status(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            with self.assertRaises(job.StepError):
                job.run_step(out, 'missing', [str(out / 'absent')])
            self.assertIn('FileNotFoundError', (out / 'missing.stderr.log').read_text())
            self.assertIsNone(json.loads((out / 'missing.step.json').read_text())['exit_code'])

    def test_nonzero_journal_retains_output_without_hiding_status(self):
        with tempfile.TemporaryDirectory() as directory:
            stdout, code = job.run_step(Path(directory), 'journal',
                                       [sys.executable, '-c', "print('partial journal'); raise SystemExit(7)"],
                                       check=False)
            self.assertEqual((stdout.strip(), code), ('partial journal', 7))

    def test_existing_bitwise_and_timing_controls(self):
        with contextlib.redirect_stdout(io.StringIO()):
            job.controls()

    def test_all_layer_controls_require_both_exact_candidates(self):
        with contextlib.redirect_stdout(io.StringIO()):
            job.reuse_controls()

    def test_imports_do_not_initialize_cuda(self):
        script = "import ctypes; ctypes.CDLL=lambda *a,**k: (_ for _ in ()).throw(AssertionError('CUDA loaded')); import prefill_attention_resources"
        subprocess.run([sys.executable, '-B', '-c', script], cwd=Path(__file__).parent, check=True)


class DecisionJob(unittest.TestCase):
    def exercise(self, corrupt=False):
        calls = []
        def step(name, args, timeout, **kwargs):
            calls.append((name, timeout, kwargs['env']['CHOPINFER_KERNEL_TIMEOUT_SECS']))
            variant = args[-1]
            if args[1] == 'decision-check':
                return '\n'.join(f'prefix PASS prefill_attention_bitwise rank={rank} variant={variant} rows=8192 workspace=true output=true replay=true f64=true candidate=pair reuse_cases=true' for rank in range(1 if corrupt else 2))
            self.assertEqual(len([c for c in calls if c[0].startswith('check-')]), 3)
            return '\n'.join(f'prefix TIMING prefill_decision rank={rank} variant={variant} rows=8192 old_total_us=29 candidate_total_us=6 paired_ratio=0.21 candidate=pair warmup=1 samples=3' for rank in range(2))
        with tempfile.TemporaryDirectory() as directory:
            result = job.decision_steps(step, {}, Path(directory))
        return calls, result

    def test_three_bitwise_phases_precede_three_bounded_timing_phases(self):
        calls, result = self.exercise()
        self.assertEqual([c[0] for c in calls], ['check-window','check-mixed64','check-mixed32','time-window','time-mixed64','time-mixed32'])
        self.assertEqual([c[1:] for c in calls], [(95,'90')]*3 + [(50,'45')]*3)
        self.assertEqual(len(result['timings']), 6)
        self.assertEqual(result['bitwise_variants'], ['window','mixed64','mixed32'])

    def test_missing_bitwise_rank_prevents_timing(self):
        with self.assertRaises(AssertionError):
            self.exercise(corrupt=True)

    def test_timing_timeout_preserves_completed_bitwise_results(self):
        def step(name, args, timeout, **kwargs):
            if name.startswith('time-'):
                raise job.StepError('timing deadline')
            return '\n'.join(f'prefix PASS prefill_attention_bitwise rank={rank} variant={args[-1]} rows=8192 workspace=true output=true replay=true f64=true candidate=pair reuse_cases=true' for rank in range(2))
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            with self.assertRaises(job.StepError):
                job.decision_steps(step, {}, out)
            progress = json.loads((out/'decision-progress.json').read_text())
            self.assertEqual(progress, dict(bitwise_variants=['window','mixed64','mixed32'],timings=[]))

    def test_timing_parser_rejects_missing_duplicate_nan_and_wrong_samples(self):
        good = '\n'.join(f'prefix TIMING prefill_decision rank={rank} variant=window rows=8192 old_total_us=29 candidate_total_us=6 paired_ratio=0.21 candidate=pair warmup=1 samples=3' for rank in range(2))
        self.assertEqual(len(job.parse_decision_timing(good,'window')), 2)
        for bad in [good.splitlines()[0],good+'\n'+good,good.replace('candidate_total_us=6','candidate_total_us=nan'),good.replace('samples=3','samples=7'),good+'\nFAILED',good.replace('candidate=pair','candidate=vector')]:
            with self.assertRaises((AssertionError,ValueError)):
                job.parse_decision_timing(bad,'window')


if __name__ == '__main__':
    unittest.main()
