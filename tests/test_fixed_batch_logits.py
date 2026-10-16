"""CPU checks for the fixed-batch device runner's gates and arm settings."""
import importlib.util
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("fixed_batch", Path(__file__).resolve().parents[1] / "tools/dev/fixed-batch-logits.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class FixedBatch(unittest.TestCase):
    def test_comparison_cli_accepts_c1_and_larger_control(self):
        with tempfile.TemporaryDirectory() as temporary:
            argv = ['fixed-batch-logits.py', 'prepare-compare', '--root', temporary,
                    '--concurrency', '1', '--concurrency', '16', '--max-sequences', '16',
                    '--verify-rows-per-item', '2']
            with patch('sys.argv', argv), patch.object(runner, 'prepare_compare') as prepare:
                runner.main()
            args = prepare.call_args.args[0]
            self.assertEqual(args.concurrency, [1, 16])
            self.assertEqual((args.max_sequences, args.verify_rows_per_item), (16, 2))

    def test_decode_comparison_guard_accepts_c1_and_rejects_other_small_sizes(self):
        def prepare(concurrency):
            args = SimpleNamespace(concurrency=concurrency, prefill_only=False, max_sequences=16,
                                   checkpoint=None)
            with patch.object(runner, 'source', return_value='fixture'), \
                    self.assertRaises(ValueError) as caught:
                runner.prepare_compare(args)
            return str(caught.exception)
        # C1 with C16 passes the size guard and stops at the next required input.
        self.assertIn('--checkpoint', prepare([1, 16]))
        self.assertIn('other sizes require --prefill-only', prepare([2, 16]))

    def test_single_case_replay_preserves_explicit_trace_and_capacity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            trace = root / 'trace.json'
            trace.write_text('{"input_tokens":[1,2,3]}')
            prepared = dict(format=runner.FORMAT, ranks=4, source='fixture',
                            binary_sha256='binary', concurrencies=[16, 32], max_sequences=32,
                            verify_rows_per_item=8, declared_differences=[],
                            packages={arm: dict(bundle=str(root / arm), topology='fixture',
                                                checkpoint=str(root / 'checkpoint'), environment={})
                                      for arm in ('base', 'tip')})
            args = SimpleNamespace(concurrency=[16], ranks=4, trace=trace, output=root / 'replay',
                                   reference=root / 'baseline', control=root / 'control', arm='tip',
                                   prepared=root / 'prepared.json', devices='0,1,2,3', minutes=10)
            config = runner.replay_config(prepared, args)
            self.assertTrue(config['trace_input'])
            self.assertEqual(config['checkpoint'], str(root / 'checkpoint'))
            self.assertEqual((config['concurrency'], config['max_sequences'], config['verify_rows_per_item']), (16,32,8))
            self.assertEqual(config['trace'], str(trace))
            self.assertEqual(trace.read_text(), '{"input_tokens":[1,2,3]}')
            self.assertIn('--trace', config['diagnostic']['reproduction'])
            prepared.update(prefill_only=True, concurrencies=[2], verify_rows_per_item=None)
            args.concurrency = [2]
            config = runner.replay_config(prepared, args)
            self.assertTrue(config['prefill_only'])
            self.assertFalse(config['verify'])
            self.assertIsNone(config['verify_rows_per_item'])
            prepared.update(prefill_only=False, concurrencies=[16, 32], verify_rows_per_item=8)
            args.concurrency = [16]
            args.control = None
            with self.assertRaisesRegex(ValueError, '--control'):
                runner.replay_config(prepared, args)
            args.arm, args.reference = 'base', None
            self.assertIsNone(runner.replay_config(prepared, args)['reference'])
            args.output.mkdir()
            with self.assertRaisesRegex(ValueError, 'already exists'):
                runner.replay_config(prepared, args)

    def test_diagnostic_paths_reject_repository_and_symlink_escape(self):
        with self.assertRaises(ValueError):
            runner.diagnostics.external(runner.REPO / 'diagnostic-output')
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'link').symlink_to(runner.REPO, target_is_directory=True)
            with self.assertRaises(ValueError):
                runner.diagnostics.external(root / 'link' / 'output')
            self.assertEqual(runner.diagnostics.external(root / 'fresh'), root / 'fresh')

    def test_every_control_precedes_any_owned_arm(self):
        arms = runner.plan_arms()
        self.assertEqual(len(arms), 18)
        self.assertEqual({(model, c) for model, c, arm in arms if arm == "control"}, set(runner.TOPOLOGIES[4][0]))
        self.assertEqual([arm for _, _, arm in arms], ["gather"] * 6 + ["control"] * 6 + ["owned"] * 6)
        for ranks, cases in [(2, 3)]:
            arms = runner.plan_arms(ranks)
            self.assertEqual([arm for _, _, arm in arms], ["gather"] * cases + ["control"] * cases + ["owned"] * cases)
            self.assertTrue(all(model.startswith("qwen-") for model, _, _ in arms))

    def test_hc_mode_replays_both_owned_settings_against_one_trace(self):
        arms = runner.plan_arms(4, True)
        self.assertEqual(len(arms), 12)
        self.assertEqual([a for _, _, a in arms], ['gather']*3+['control']*3+['owned-off']*3+['owned-on']*3)
        # TP2 compares its own Qwen C16 and C32 cases.
        for ranks, cases in [(2, {('qwen-ar', 16), ('qwen-ar', 32), ('qwen-mtp', 16)})]:
            arms = runner.plan_arms(ranks, True)
            self.assertEqual({(m, c) for m, c, _ in arms}, cases)
            self.assertEqual([a for _, _, a in arms],
                             [a for a in ('gather', 'control', 'owned-off', 'owned-on') for _ in cases])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary = root/'replay-tests'
            binary.write_bytes(b'CPU fixture')
            prepared = {'source': 'same', 'ranks': 4, 'hc_compare': True,
                        'binary': str(binary), 'binary_sha256': runner.digest(binary),
                        'packages': {model: {'bundle': '/data/'+model, 'topology': 'tp4'}
                                     for model in ('qwen-ar', 'qwen-mtp')}}
            (root/'prepared.json').write_text(json.dumps(prepared))
            (root/'latest-prepared').write_text(str(root/'prepared.json'))
            seen = {}
            def execute(command, log, env, timeout):
                self.assertIn(runner.TEST, command)
                config = json.loads(Path(env['CHOPINFER_BATCH_REPLAY_CONFIG']).read_text())
                arm = Path(config['output']).name
                self.assertEqual(config['owned'], arm.startswith('owned'))
                self.assertEqual(env['CHOPINFER_QWEN38_ROW_OWNED_LOGITS'], '1' if config['owned'] else '0')
                self.assertEqual(env['CHOPINFER_QWEN38_HC_PRECOMPUTE'], '1' if arm == 'owned-on' else '0')
                if config['owned']:
                    self.assertTrue(Path(config['control'], 'receipt.json').is_file())
                    self.assertEqual(Path(config['reference']).name, 'gather')
                case = Path(config['trace']).parent.name
                if arm == 'owned-off':
                    seen[case] = {k: v for k, v in config.items() if k != 'output'}
                if arm == 'owned-on':
                    self.assertEqual(seen[case], {k: v for k, v in config.items() if k != 'output'})
                output = Path(config['output'])
                output.mkdir()
                (output/'receipt.json').write_text(json.dumps({'passed': True, 'steps': 13, 'rows': 208}))
                log.write_text('Qwen3.8 HC precompute: '+('on' if arm == 'owned-on' else 'off'))
            with patch.object(runner, 'source', return_value='same'), patch.object(runner.subprocess, 'run'), \
                 patch.object(runner, 'gpu_state', return_value='CPU fixture'), patch.object(runner, 'command', side_effect=execute):
                runner.run(root, '0,1,2,3', 25, 4, True)
            self.assertEqual(set(seen), {'qwen-ar-c16', 'qwen-ar-c32', 'qwen-mtp-c16'})
            self.assertTrue(json.loads(next(root.glob('run-*/result.json')).read_text())['passed'])

    def test_prepare_records_an_hc_comparison_build(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary = root / 'replay-tests'
            binary.write_bytes(b'CPU fixture')

            def package(args, log, *rest, **kw):
                ranks = args[args.index('--ranks') + 1] if '--ranks' in args else 4
                bundle = log.parent / f'package-{log.stem}'
                bundle.mkdir()
                bundle_id = runner.BUNDLES['qwen38'][0].format(ranks)
                (bundle / 'manifest.json').write_text(json.dumps({'id': bundle_id}))
                log.write_text(f'prepared {bundle}\n')

            env = {'CHOPINFER_QWEN38_CHECKPOINT': str(root / 'checkpoint')}
            for ranks, hc in [(2, True), (2, False)]:
                with patch.object(runner, 'source', return_value='same'), \
                     patch.object(runner, 'build_recorder', return_value=binary), \
                     patch.object(runner, 'command', side_effect=package), \
                     patch.dict(runner.os.environ, env):
                    runner.prepare(root / f'tp{ranks}-hc{hc}', ranks, hc)
                latest = Path((root / f'tp{ranks}-hc{hc}' / 'latest-prepared').read_text().strip())
                prepared = json.loads(latest.read_text())
                self.assertEqual((prepared['ranks'], prepared['hc_compare']), (ranks, hc))

    def test_rank_ordinals_require_one_distinct_device_per_rank(self):
        self.assertEqual(runner.rank_ordinals("4,5,6,7", 4), "0,1,2,3")
        self.assertEqual(runner.rank_ordinals("2,3", 2), "0,1")
        for devices, ranks in [("0,1,2,3", 8), ("0,1", 4), ("0,0", 2)]:
            with self.assertRaises(ValueError):
                runner.rank_ordinals(devices, ranks)
        env = runner.arm_environment({}, "0,1,2,3", "qwen-ar", False, 4)
        self.assertEqual(env["CHOPINFER_KERNEL_GPUS"], "0,1,2,3")
        self.assertEqual(env["CHOPINFER_QWEN38_ROW_OWNED_LOGITS"], "0")

    def test_default_owned_arm_removes_inherited_overrides(self):
        inherited = {"CHOPINFER_MIMO_ROW_OWNED_LOGITS": "0", "CHOPINFER_MIMO_ROW_OWNED_SPECULATION": "0",
                     "CHOPINFER_KERNEL_SHARED_DEVICE": "1", "CHOPINFER_KERNEL_GPU": "7", "PATH": "/bin"}
        env = runner.arm_environment(inherited, "4,5,6,7", "mimo-ar", True)
        self.assertFalse(any(key.startswith("CHOPINFER_MIMO_") for key in env))
        self.assertNotIn("CHOPINFER_KERNEL_SHARED_DEVICE", env)
        self.assertNotIn("CHOPINFER_KERNEL_GPU", env)
        self.assertEqual(env["CHOPINFER_KERNEL_GPUS"], "0,1,2,3")
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "4,5,6,7")
        self.assertEqual(env["CHOPINFER_SAMPLING_DEFER_VIEWS"], "1")
        gather = runner.arm_environment(inherited, "4,5,6,7", "mimo-ar", False)
        self.assertEqual(gather["CHOPINFER_MIMO_ROW_OWNED_LOGITS"], "0")
        self.assertEqual(gather["CHOPINFER_SAMPLING_DEFER_VIEWS"], "0")
        owned = runner.arm_environment(inherited, "4,5,6,7", "qwen-mtp", True)
        self.assertEqual(owned["CHOPINFER_QWEN38_ROW_OWNED_LOGITS"], "1")

    def test_package_comparison_controls_and_explicit_environment(self):
        self.assertEqual(runner.comparison_arms([16, 32]),
                         [(16, "base"), (32, "base"), (16, "control"),
                          (32, "control"), (16, "tip"), (32, "tip")])
        settings = runner.parse_overrides(["CHOPINFER_EXAMPLE_ROUTE=1"])
        env = runner.comparison_environment(
            {"PATH": "/bin", "CUDA_VISIBLE_DEVICES": "7",
             "CHOPINFER_EXAMPLE_ROUTE": "0", "CHOPINFER_KERNEL_SHARED_DEVICE": "1",
             "CHOPINFER_BATCH_REPLAY_CONFIG": "/stale"},
            "0,1,2,3", 4, settings)
        self.assertEqual(env, {"PATH": "/bin", "CUDA_VISIBLE_DEVICES": "0,1,2,3",
                               "CHOPINFER_KERNEL_GPUS": "0,1,2,3",
                               "CHOPINFER_EXAMPLE_ROUTE": "1"})
        for values in [["CUDA_VISIBLE_DEVICES=7"], ["CHOPINFER_KERNEL_GPUS=7"],
                       ["CHOPINFER_BATCH_REPLAY_CONFIG=bad"],
                       ["CHOPINFER_BUNDLE_VERIFY_CACHE_DIR=bad"],
                       ["CHOPINFER_EXAMPLE_ROUTE"], ["CHOPINFER_EXAMPLE_ROUTE=0",
                                                     "CHOPINFER_EXAMPLE_ROUTE=1"]]:
            with self.assertRaises(ValueError):
                runner.parse_overrides(values)

    def test_comparison_refuses_capacity_below_live_requests(self):
        with patch.object(runner, "source") as source:
            with self.assertRaisesRegex(ValueError, "cover every live concurrency"):
                runner.prepare_compare(SimpleNamespace(concurrency=[16, 32], max_sequences=16))
            source.assert_not_called()

    def test_comparison_packages_must_draft_and_verify_the_trace_rows(self):
        drafting = {"id": "mimo.v26.flash.v2", "speculation": {
            "bundle_drafter": True, "maximum_draft_tokens": 7,
            "maximum_verify_tokens": 256, "row_buckets": [2, 4, 8]}}
        self.assertEqual(runner.speculation_problems(drafting, {}, [16, 32], 32, 8), [])
        # An AR package cannot draft the trace's verify rows.
        ar = {"id": drafting["id"], "speculation": {**drafting["speculation"], "bundle_drafter": False}}
        self.assertRegex(runner.speculation_problems(ar, {}, [16], 32, 8)[0], "no bundle drafter")
        self.assertRegex(runner.speculation_problems({"id": "x"}, {}, [16], 32, 8)[0], "no bundle drafter")
        shallow = {"id": drafting["id"], "speculation": {**drafting["speculation"],
                   "maximum_draft_tokens": 3, "maximum_verify_tokens": 64, "row_buckets": [2, 4]}}
        self.assertEqual(len(runner.speculation_problems(shallow, {}, [16], 32, 8)), 3)
        # Qwen3.8 verifies 32 rows per step at 32 sequences unless wide.
        qwen = {**drafting, "id": "qwen38.flash.v2"}
        narrow = runner.speculation_problems(qwen, {}, [16], 32, 8)
        self.assertEqual(len(narrow), 1)
        self.assertIn("CHOPINFER_SPECULATIVE_WIDE_ROWS=1", narrow[0])
        wide = {"CHOPINFER_SPECULATIVE_WIDE_ROWS": "1"}
        self.assertEqual(runner.speculation_problems(qwen, wide, [16], 32, 8), [])
        self.assertEqual(runner.speculation_problems(qwen, {}, [16], 32, 2), [])

    def test_comparison_refuses_undrafting_packages_before_building(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for arm, drafter in (("base", False), ("tip", True)):
                (root / arm).mkdir()
                (root / arm / "manifest.json").write_text(json.dumps({
                    "id": "mimo.v26.flash.v2", "speculation": {
                        "bundle_drafter": drafter, "maximum_draft_tokens": 7,
                        "maximum_verify_tokens": 256, "row_buckets": [2, 4, 8]}}))
            args = SimpleNamespace(concurrency=[16], max_sequences=32, verify_rows_per_item=8,
                                   base_bundle=root / "base", tip_bundle=root / "tip",
                                   checkpoint=root / "checkpoint",
                                   base_env=[], tip_env=[], topology="mimo-v26-flash-tp4",
                                   allow_difference=[], root=root / "out", ranks=4, limits={})
            with patch.object(runner, "source", return_value="rev"), \
                    patch.object(runner, "build_recorder") as build:
                with self.assertRaisesRegex(ValueError, "base.*no bundle drafter|no bundle drafter"):
                    runner.prepare_compare(args)
                build.assert_not_called()
            self.assertFalse((root / "out").exists())

    def test_default_metadata_budget_is_the_reader_ceiling(self):
        # Drafter traces write one grammar mask per verify row; 16 MiB ended
        # C16 and C32 comparisons at their second grammar step.
        self.assertEqual(runner.DEFAULT_LIMITS["metadata_bytes"], runner.diagnostics.MAX_METADATA_LIMIT)
        self.assertEqual(runner.DEFAULT_LIMITS["metadata_bytes"], 256 << 20)

    def test_busy_devices_fail_before_execution(self):
        empty = "\n".join(f"{i}, uuid{i}, GPU, 580, 2" for i in range(4))
        for processes, rows, valid in [("", empty, True), ("123", empty, False),
                                       ("", empty.replace(", 2", ", 1000"), False),
                                       ("", empty.splitlines()[0], False)]:
            with patch.object(runner, "output", side_effect=[processes, rows]):
                if valid:
                    self.assertEqual(runner.gpu_state("0,1,2,3"), empty)
                else:
                    with self.assertRaises(RuntimeError):
                        runner.gpu_state("0,1,2,3")


if __name__ == "__main__":
    unittest.main()
