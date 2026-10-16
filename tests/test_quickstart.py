"""Quickstart builds, installs private binaries and serves `serve --model CHECKPOINT`."""
import contextlib
import importlib.util
import io
import json
import os
import subprocess
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('quickstart', ROOT / 'tools/quickstart.py')
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def quickstart(argv, env=None):
    """Run main() in dry-run or with fakes; returns (stdout, stderr, exit code or None)."""
    output, errors = io.StringIO(), io.StringIO()
    code = None
    with patch.object(sys, 'argv', ['quickstart', *map(str, argv)]), patch.dict(os.environ, env or {}), \
            contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors):
        try:
            MODULE.main()
        except SystemExit as exit:
            code = exit.code
    return output.getvalue(), errors.getvalue(), code


def fake_target(root, model):
    """Cargo-like release outputs, group writable as the shared targets are."""
    for _, name in MODULE.binaries(model):
        path = root / 'release' / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name)
        path.chmod(0o775)


class QuickstartTest(unittest.TestCase):
    def test_released_models_are_the_choices(self):
        models = MODULE.inventory.load(ROOT)
        variables = MODULE.checkpoint_variables()
        self.assertEqual(list(variables), list(MODULE.inventory.released(ROOT)))
        for model in models:
            with self.subTest(model=model.name):
                if model.release.enabled:
                    self.assertEqual(variables[model.name], f'CHOPINFER_{model.checkpoint_env}_CHECKPOINT')
                    _, errors, code = quickstart([model.name, 'missing-profile', '/data', '--dry-run'])
                    self.assertIn("unknown profile 'missing-profile'", errors)
                else:
                    _, errors, code = quickstart([model.name, 'tp4-ar', '/data', '--dry-run'])
                    self.assertIn('invalid choice', errors)
                self.assertEqual(code, 2)

    def test_dry_run_prints_build_install_and_serve_model(self):
        with tempfile.TemporaryDirectory() as temporary:
            data = Path(temporary) / 'absent'
            stdout, _, code = quickstart(['qwen38', 'tp4-mtp', data, '--checkpoint', '/checkpoint', '--dry-run'])
            self.assertIsNone(code)
            self.assertFalse(data.exists())
            self.assertIn('cargo build --release --locked --offline -p runtime-host -p model-qwen38 -p bundle-qwen38 '
                          '--lib --bin chopinfer --bin chopinfer-model-qwen38 --features runtime-host/linux-cuda-host',
                          stdout)
            # CI may name a shared target directory; quickstart builds there.
            target = Path(os.environ.get('CARGO_TARGET_DIR', data / 'target')).resolve()
            self.assertIn(f'install -m 0755 {target}/release/libbundle_qwen38.so '
                          f'{data}/releases/NEW-RELEASE/install/lib/libbundle_qwen38.so', stdout)
            self.assertIn(f'{data}/releases/NEW-RELEASE/install/bin/chopinfer serve --model /checkpoint '
                          f'--tensor-parallel-size 4 --model-cache-dir {data}/model-cache\n', stdout)
            self.assertNotIn(' verify ', stdout)
            for removed in ['materializ', 'bundle install', 'bundle verify', 'install-root', '--profile', 'rank']:
                self.assertNotIn(removed, stdout)

    def test_autoregressive_profiles_turn_speculation_off(self):
        for model, profile, tp in [('qwen38', 'tp4-ar', 4), ('qwen38', 'tp2-ar', 2), ('mimo26', 'tp4-ar', 4),
                                   ('mimo26', 'tp2-ar', 2)]:
            stdout, _, code = quickstart([model, profile, '/data', '--checkpoint', '/checkpoint', '--dry-run',
                                          '--skip-build', '--model-cache-dir', '/cache', '--listen', '0.0.0.0:9000',
                                          '--', '--max-sequences', '8'])
            self.assertIsNone(code)
            self.assertIn(f'serve --model /checkpoint --tensor-parallel-size {tp} --model-cache-dir /cache '
                          '--speculative-tokens 0 --listen 0.0.0.0:9000 --max-sequences 8', stdout)
            self.assertNotIn('cargo', stdout)
        # An explicit override is not doubled.
        stdout, _, _ = quickstart(['qwen38', 'tp4-ar', '/data', '--checkpoint', '/c', '--dry-run', '--skip-build',
                                   '--', '--speculative-tokens=0'])
        self.assertEqual(stdout.count('--speculative-tokens'), 1)

    def test_verify_uses_mimo_manifest_and_allows_unrecorded(self):
        default = json.loads((ROOT / 'models/mimo26/checkpoints.json').read_text())['default']
        stdout, _, code = quickstart(['mimo26', 'tp4-ar', '/data', '--checkpoint', '/checkpoint', '--dry-run',
                                      '--skip-build', '--skip-serve', '--verify'])
        self.assertIsNone(code)
        manifest = f"{ROOT}/models/mimo26/checkpoint-files.json#{default}"
        self.assertIn(f"chopinfer verify --model /checkpoint --manifest '{manifest}' --allow-unrecorded", stdout)
        self.assertNotIn(' serve ', stdout)
        stdout, _, _ = quickstart(['qwen38', 'tp4-mtp', '/data', '--checkpoint', '/checkpoint', '--dry-run',
                                   '--skip-build', '--verify'])
        self.assertIn('chopinfer verify --model /checkpoint\n', stdout)

    def test_removed_flags_name_their_replacement(self):
        base = ['mimo26', 'tp4-ar', '/data', '--checkpoint', '/c', '--dry-run']
        for flag, expected in [('--skip-materialize', 'no rank images'), ('--skip-package', 'model cache'),
                               ('--skip-install', 'nothing is installed'), ('--space-margin-gib=0', 'space preflight'),
                               ('--drafter-checkpoint', 'CHECKPOINT/dflash/')]:
            _, errors, code = quickstart([flag, *base] if flag == '--drafter-checkpoint' else [*base, flag])
            self.assertEqual(code, 2, flag)
            self.assertIn('was removed', errors)
            self.assertIn(expected, errors)
        for override, expected in [('--install-root=/x', 'serve --model'), ('--load', 'directly'),
                                   ('--model', '--checkpoint'), ('--tensor-parallel-size', 'choose a profile'),
                                   ('--bundle', 'serve --model')]:
            _, errors, code = quickstart([*base, '--', override, 'value'])
            self.assertEqual(code, 2, override)
            self.assertIn(expected, errors)
        _, errors, code = quickstart([*base, '--max-sequences', '8'])
        self.assertEqual(code, 2)
        self.assertIn('serve overrides must follow --', errors)

    def test_mimo_profile_must_match_the_checkpoint_layout(self):
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary)
            base = ['/data', '--checkpoint', checkpoint, '--dry-run', '--skip-build']
            _, errors, code = quickstart(['mimo26', 'tp4-dflash', *base])
            self.assertEqual(code, 2)
            self.assertIn(f'{checkpoint}/dflash/', errors)
            _, errors, code = quickstart(['mimo26', 'tp4-mtp', *base])
            self.assertIn('model_mtp.safetensors', errors)
            (checkpoint / 'model_mtp.safetensors').write_text('')
            self.assertIsNone(quickstart(['mimo26', 'tp4-mtp', *base])[2])
            (checkpoint / 'dflash').mkdir()
            _, errors, code = quickstart(['mimo26', 'tp4-mtp', *base])
            self.assertEqual(code, 2)
            self.assertIn('picks tp4-dflash', errors)
            self.assertIsNone(quickstart(['mimo26', 'tp4-dflash', *base])[2])
            self.assertIsNone(quickstart(['qwen38', 'tp4-mtp', *base])[2])

    def test_serve_installs_private_binaries_then_execs_serve_model(self):
        with tempfile.TemporaryDirectory() as temporary:
            data = Path(temporary)
            checkpoint = data / 'checkpoint'
            (checkpoint / 'dflash').mkdir(parents=True)
            fake_target(data / 'build', 'mimo26')
            calls, execs = [], []
            argv = ['mimo26', 'tp4-dflash', data, '--checkpoint', checkpoint, '--skip-build', '--verify',
                    '--release', data / 'releases/mine']
            env = {'CARGO_TARGET_DIR': str(data / 'build'), 'CUDA_VISIBLE_DEVICES': '0,1,2,3'}
            asked = []
            def selected(helper, checkpoint, ranks):
                asked.append((helper, checkpoint, ranks))
                return 'tp4-dflash'
            with patch.object(MODULE, 'run', lambda command, **_: calls.append(list(map(str, command)))), \
                    patch.object(MODULE.os, 'execvp', lambda name, command: execs.append(command)), \
                    patch.object(MODULE, 'selected_profile', selected):
                self.assertIsNone(quickstart(argv, env)[2])
                # Rerunning into the same release replaces its binaries.
                self.assertIsNone(quickstart(argv, env)[2])
            self.assertEqual(asked[0], (data / 'releases/mine/install/bin/chopinfer-model-mimo26', checkpoint, 4))
            install = data / 'releases/mine/install'
            for directory, name in MODULE.binaries('mimo26'):
                self.assertEqual((install / directory / name).read_text(), name)
            for path in [install, *install.rglob('*')]:
                self.assertEqual(path.stat().st_mode & 0o777, 0o755, path)
            host = str(install / 'bin/chopinfer')
            self.assertEqual([call[:2] for call in calls], [[host, 'profiles'], [host, 'verify']] * 2)
            self.assertEqual(execs[0], [host, 'serve', '--model', str(checkpoint), '--tensor-parallel-size', '4',
                                        '--model-cache-dir', str(data / 'model-cache')])
            self.assertFalse((data / 'model-cache').exists())

    def served(self, model, profile, selected, extra=()):
        """Runs quickstart through its serve step with the helper selecting
        `selected`; returns (stdout, stderr, exit code, calls, exec'd command)."""
        with tempfile.TemporaryDirectory() as temporary:
            data = Path(temporary)
            (data / 'checkpoint' / 'dflash').mkdir(parents=True)
            fake_target(data / 'build', model)
            calls, execs = [], []
            env = {'CARGO_TARGET_DIR': str(data / 'build'), 'CUDA_VISIBLE_DEVICES': '0,1,2,3'}
            with patch.object(MODULE, 'run', lambda command, **_: calls.append(list(map(str, command)))), \
                    patch.object(MODULE.os, 'execvp', lambda name, command: execs.append(command)), \
                    patch.object(MODULE, 'selected_profile', lambda *_: selected):
                stdout, errors, code = quickstart([model, profile, data, '--checkpoint', data / 'checkpoint',
                                                   '--skip-build', *extra], env)
            return stdout, errors, code, calls, execs[0] if execs else None

    def test_a_profile_the_checkpoint_does_not_serve_is_refused_before_its_qualification(self):
        # A mixed-precision Qwen checkpoint has no MTP: its helper selects
        # tp4-ar, so tp4-mtp's qualification must not be shown for it.
        profiles = json.loads((ROOT / 'models/qwen38/profiles.json').read_text())['profiles']
        stdout, errors, code, calls, served = self.served('qwen38', 'tp4-mtp', 'tp4-ar')
        self.assertEqual(code, 2)
        self.assertIn('this checkpoint serves tp4-ar at 4 ranks, not tp4-mtp; choose tp4-ar', errors)
        self.assertNotIn(profiles['tp4-mtp']['qualification'], stdout)
        self.assertEqual((calls, served), ([], None))
        # A speculative profile of another rank count or bundle is refused too.
        self.assertEqual(self.served('qwen38', 'tp2-ar', 'tp4-mtp')[2], 2)

    def test_an_autoregressive_request_runs_the_selected_profile_with_speculation_off(self):
        profiles = json.loads((ROOT / 'models/mimo26/profiles.json').read_text())['profiles']
        stdout, _, code, calls, served = self.served('mimo26', 'tp4-ar', 'tp4-dflash')
        self.assertIsNone(code)
        self.assertIn("Profile: tp4-ar runs this checkpoint's tp4-dflash with speculation off; tp4-dflash: "
                      f"{profiles['tp4-dflash']['status']}: {profiles['tp4-dflash']['qualification']}", stdout)
        self.assertEqual(calls[0][1:], ['profiles', 'show', str(ROOT / 'models/mimo26/profiles.json'), 'tp4-dflash'])
        self.assertEqual(served[-2:], ['--speculative-tokens', '0'])
        # The requested profile, when selected, is shown as it is.
        stdout, _, code, calls, _ = self.served('mimo26', 'tp4-dflash', 'tp4-dflash')
        self.assertIsNone(code)
        self.assertIn(f"Profile: tp4-dflash: {profiles['tp4-dflash']['status']}", stdout)

    def test_the_helper_is_asked_for_its_profile(self):
        with tempfile.TemporaryDirectory() as temporary:
            helper = Path(temporary) / 'chopinfer-model-qwen38'
            helper.write_text('#!/bin/sh\n[ "$1 $2 $3 $4 $5" = "profile --model /c --tensor-parallel-size 4" ] '
                              '|| exit 3\necho \'{"result":"direct-profile-selected","name":"tp4-ar","profile":{}}\'\n')
            helper.chmod(0o755)
            self.assertEqual(MODULE.selected_profile(helper, Path('/c'), 4), 'tp4-ar')
            with self.assertRaises(SystemExit) as refused:
                MODULE.selected_profile(helper, Path('/c'), 2)
            self.assertIn('names no serving profile for /c', str(refused.exception))
        stdout, _, code = quickstart(['qwen38', 'tp4-mtp', '/data', '--checkpoint', '/c', '--dry-run', '--skip-build'])
        self.assertIsNone(code)
        self.assertIn('chopinfer-model-qwen38 profile --model /c --tensor-parallel-size 4\n', stdout)

    def test_release_with_rank_images_is_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            data = Path(temporary)
            for name in ['ranks', 'package', 'installed', 'verify-cache']:
                (data / 'old' / name).mkdir(parents=True)
            _, errors, code = quickstart(['qwen38', 'tp4-mtp', data, '--checkpoint', '/c', '--dry-run',
                                          '--release', data / 'old'])
            self.assertEqual(code, 2)
            self.assertIn('use a fresh release directory', errors)

    def test_missing_build_and_unassigned_devices_have_actionable_errors(self):
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / 'checkpoint'
            checkpoint.mkdir()
            args = ['mimo26', 'tp4-ar', temporary, '--checkpoint', checkpoint, '--skip-build', '--skip-serve']
            _, errors, code = quickstart(args, {'CARGO_TARGET_DIR': temporary + '/build'})
            self.assertEqual(code, 2)
            self.assertIn('rerun without --skip-build', errors)
            args.remove('--skip-serve')
            _, errors, code = quickstart(args, {'CUDA_VISIBLE_DEVICES': ''})
            self.assertEqual(code, 2)
            self.assertIn('set CUDA_VISIBLE_DEVICES', errors)
            _, errors, code = quickstart(['mimo26', 'tp4-ar', temporary, '--checkpoint', f'{temporary}/absent',
                                          '--skip-serve'])
            self.assertIn('checkpoint directory not found', errors)

    def test_wrapper_resolves_paths_from_callers_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            caller = Path(temporary)
            env = dict(os.environ, CHOPINFER_LOCAL_ENV='/dev/null', CARGO_TARGET_DIR='build')
            result = subprocess.run([ROOT / 'tools/quickstart.sh', 'mimo26', 'tp4-dflash', 'data',
                '--checkpoint', 'checkpoint', '--model-cache-dir', 'cache',
                '--release', 'data/release', '--dry-run', '--skip-build'],
                cwd=caller, env=env, text=True, capture_output=True, check=True)
            for expected in ['data/release/install/bin/chopinfer serve', 'checkpoint', 'cache',
                             'build/release/chopinfer']:
                self.assertIn(str(caller / expected), result.stdout)
            self.assertFalse((caller / 'data').exists())


if __name__ == '__main__':
    unittest.main()
