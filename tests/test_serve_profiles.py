"""Validate every model-owned preset with the actual model-neutral CLI."""
import json
import os
import re
from pathlib import Path
import subprocess
import sys
import tomllib
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.model_sdk import inventory as model_inventory  # noqa: E402


def released(profiles_path):
    """Whether the model owning `profiles_path` (models/MODEL/profiles.json)
    is released (tools/model_sdk/inventory.py). A model that declares
    `[release] enabled = false` until it qualifies is exempt from the tp4-ar
    C64 qualification."""
    return profiles_path.parent.name in model_inventory.released(ROOT)

class ProfilesTest(unittest.TestCase):
    def test_all_model_profiles_enable_prefix_caching(self):
        paths = sorted((ROOT / 'models').glob('*/profiles.json'))
        self.assertTrue(paths)
        for path in paths:
            model = path.parent.name
            profiles = json.loads(path.read_text())['profiles']
            self.assertNotIn('tp4-mtp-prefix', profiles)
            for name, profile in profiles.items():
                with self.subTest(model=model, profile=name):
                    self.assertTrue(profile['prefix_cache'])

    def test_qwen_redeploy_prefix_default_and_escape_hatch(self):
        # The profile turns prefix caching on; the redeploy passes only an
        # explicit choice.
        env = dict(os.environ, CHOPINFER_LOCAL_ENV='/dev/null')
        for drafter in ('none', 'mtp'):
            for extra, expected in (([], None), (['--prefix-cache', 'off'], 'off')):
                result = subprocess.run([
                    'bash', str(ROOT / 'tools/dev/qwen38-redeploy.sh'), '--dry-run',
                    '--checkpoint', '/data/checkpoint', '--drafter', drafter, *extra,
                ], env=env, text=True, capture_output=True, timeout=10)
                self.assertEqual(result.returncode, 0, result.stderr)
                serve = [line for line in result.stdout.splitlines() if ' serve --model ' in line]
                self.assertEqual(len(serve), 1, result.stdout)
                if expected is None:
                    self.assertNotIn('--prefix-cache', serve[0])
                else:
                    self.assertIn('--prefix-cache ' + expected, serve[0])

    def test_glm5next_redeploy_serves_the_helper_profile_with_explicit_overrides(self):
        # The helper picks the profile from the checkpoint's layout; the
        # redeploy adds only explicit choices, and refuses a real run until
        # the model inventory lists the helper binary.
        env = dict({k: v for k, v in os.environ.items() if not k.startswith('CHOPINFER_')},
                   CHOPINFER_LOCAL_ENV='/dev/null')
        script = str(ROOT / 'models/glm5next/redeploy.sh')

        def run(*extra, dry=True):
            command = ['bash', script, *(['--dry-run'] if dry else []), '--checkpoint', '/data/checkpoint', *extra]
            return subprocess.run(command, env=env, text=True, capture_output=True, timeout=10)

        def serve(result):
            self.assertEqual(result.returncode, 0, result.stderr)
            lines = [line for line in result.stdout.splitlines() if ' serve --model ' in line]
            self.assertEqual(len(lines), 1, result.stdout)
            return lines[0]

        default = serve(run())
        self.assertIn('chopinfer-model-glm5next prepare --direct --model /data/checkpoint', run().stdout)
        self.assertIn('--tensor-parallel-size 4', default)
        for word in ('--speculative-tokens', '--context-parallel-size', '--profile', '--kv-cache-dtype'):
            self.assertNotIn(word, default)
        self.assertIn('--speculative-tokens 0', serve(run('--drafter', 'none')))
        self.assertIn('--context-parallel-size 2', serve(run('--context-parallel-size', '2')))
        # --modalities reaches the helper's prepare and serve alike, so a
        # text run serves the package it prepared.
        self.assertNotIn('--modalities', run().stdout)
        text = run('--modalities', 'text')
        self.assertIn('--modalities text', serve(text))
        prepare = [line for line in text.stdout.splitlines() if ' prepare --direct ' in line]
        self.assertEqual(len(prepare), 1, text.stdout)
        self.assertIn('--modalities text', prepare[0])
        chosen = serve(run('--serve-arg', '--speculative-tokens', '--serve-arg', '5', '--drafter', 'none'))
        self.assertIn('--speculative-tokens 5', chosen)
        self.assertNotIn('--speculative-tokens 0', chosen)
        for refused, message in ((('--ranks', '2'), 'TP4 only'), (('--context-parallel-size', '3'), '1, 2 or 4'),
                                 (('--drafter', 'dflash'), 'none or mtp'),
                                 (('--serve-arg', '--modalities', '--serve-arg', 'text'), 'pass --modalities')):
            result = run(*refused)
            self.assertEqual(result.returncode, 2, result.stdout)
            self.assertIn(message, result.stderr)
        inventory = tomllib.loads((ROOT / 'models/glm5next/model.toml').read_text())
        if 'chopinfer-model-glm5next' not in inventory['compiler'].get('bin', []):
            result = run(dry=False)
            self.assertEqual(result.returncode, 2, result.stdout)
            self.assertIn('no chopinfer-model-glm5next helper yet', result.stderr)

    def test_glm5next_redeploy_installs_the_helper_for_development(self):
        # Once the inventory lists the helper, a real redeploy installs it
        # and the bundle from the model's own entry, not the release plan;
        # --stop refuses a PID record it cannot verify.
        import shutil
        import tempfile
        with tempfile.TemporaryDirectory(prefix='glm5next-redeploy-') as temporary:
            root = Path(temporary)
            model = root / 'models/glm5next'
            for role in ('bundle', 'compiler', 'contract'):
                (model / role).mkdir(parents=True)
                (model / role / 'Cargo.toml').write_text('')
            inventory = (ROOT / 'models/glm5next/model.toml').read_text()
            if 'chopinfer-model-glm5next' not in tomllib.loads(inventory)['compiler'].get('bin', []):
                inventory = inventory.replace('package = "model-glm5next"\n',
                                              'package = "model-glm5next"\nbin = ["chopinfer-model-glm5next"]\n', 1)
            inventory = re.sub(r'\n\[checks\].*', '\n', inventory, flags=re.S)
            (model / 'model.toml').write_text(inventory)
            shutil.copy(ROOT / 'models/glm5next/redeploy.sh', model / 'redeploy.sh')
            (root / 'tools/dev').mkdir(parents=True)
            for name in ('redeploy-common.sh', 'redeploy-wait.py'):
                shutil.copy(ROOT / 'tools/dev' / name, root / 'tools/dev' / name)
            shutil.copytree(ROOT / 'tools/model_sdk', root / 'tools/model_sdk',
                            ignore=shutil.ignore_patterns('__pycache__'))
            release = root / 'target/release'
            release.mkdir(parents=True)
            (release / 'chopinfer').write_text('#!/bin/sh\nexit 3\n')
            (release / 'libbundle_glm5next.so').write_bytes(b'module')
            (release / 'chopinfer-model-glm5next').write_text("""#!/usr/bin/env python3
import json, pathlib, sys
options = dict(zip(sys.argv[3::2], sys.argv[4::2]))
assert sys.argv[1:3] == ['prepare', '--direct'], sys.argv
here = pathlib.Path(sys.argv[0]).resolve().parent
assert (here.parent / 'lib/libbundle_glm5next.so').is_file()
bundle = pathlib.Path(options['--cache-dir']) / 'p-glm5next'
bundle.mkdir(parents=True, exist_ok=True)
print(json.dumps({'result': 'prepared-direct', 'bundle': str(bundle)}))
""")
            for path in release.iterdir():
                path.chmod(0o755)
            (root / 'checkpoint').mkdir()
            (root / 'checkpoint/config.json').write_text('{}')
            env = dict({k: v for k, v in os.environ.items() if not k.startswith('CHOPINFER_')},
                       CHOPINFER_LOCAL_ENV='/dev/null', CARGO_TARGET_DIR=str(root / 'target'))
            common = ['--checkpoint', str(root / 'checkpoint'), '--model-cache-dir', str(root / 'cache'),
                      '--work-dir', str(root / 'work'), '--logs-dir', str(root / 'logs'), '--port', '8441']
            result = subprocess.run(['bash', '-c', 'cargo() { :; }; export -f cargo; exec "$@"', 'test',
                                     'bash', str(model / 'redeploy.sh'), '--prepare-only', *common],
                                    env=env, text=True, capture_output=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn(f'prepared {root / "cache/p-glm5next"}', result.stdout)
            installs = list((root / 'work/runs').glob('*/install'))
            self.assertEqual(len(installs), 1)
            for name in ('bin/chopinfer', 'bin/chopinfer-model-glm5next', 'lib/libbundle_glm5next.so'):
                self.assertTrue((installs[0] / name).is_file(), name)
            stop = ['bash', str(model / 'redeploy.sh'), '--stop', '--logs-dir', str(root / 'logs'), '--port', '8441']
            self.assertEqual(subprocess.run(stop, env=env, capture_output=True, timeout=30).returncode, 0)
            (root / 'logs/glm5next-redeploy-8441.pid').write_text('12345\n')
            refused = subprocess.run(stop, env=env, text=True, capture_output=True, timeout=30)
            self.assertNotEqual(refused.returncode, 0)
            self.assertIn('refusing to signal', refused.stderr)

    def test_glm5next_redeploy_stops_only_under_the_lock_and_its_owner(self):
        # A record --stop refuses to read shows whether it reached the stop.
        import fcntl
        import tempfile
        env = dict({k: v for k, v in os.environ.items() if not k.startswith('CHOPINFER_')},
                   CHOPINFER_LOCAL_ENV='/dev/null', CHOPINFER_DEV_RELEASE_TIMEOUT='1')
        with tempfile.TemporaryDirectory(prefix='glm5next-stop-') as temporary:
            logs = Path(temporary) / 'logs'
            record, mine = logs / 'glm5next-redeploy-8441.pid', Path(temporary) / 'owned.pid'

            def stop(*options):
                return subprocess.run(['bash', str(ROOT / 'models/glm5next/redeploy.sh'), '--stop', '--logs-dir',
                                       str(logs), '--port', '8441', *options],
                                      env=env, text=True, capture_output=True, timeout=30)

            mine.write_text('999 1 token\n')
            result = stop('--owner-record', str(mine))
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('nothing of this owner to stop', result.stdout)
            record.write_text('12345\n')
            # --dry-run reads, locks and signals nothing.
            result = stop('--dry-run')
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn(f'would stop the server named by {record}', result.stdout)
            self.assertEqual(record.read_text(), '12345\n')
            # Another start's record is left alone; the owner's is stopped.
            result = stop('--owner-record', str(mine))
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('records another start', result.stdout)
            mine.write_text('12345\n')
            result = stop('--owner-record', str(mine))
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('refusing to signal', result.stderr)
            # A start in progress holds the port's lock; a stop waits, then
            # gives up without touching the record.
            with open(logs / 'glm5next-redeploy-8441.lock', 'w') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                result = stop()
            self.assertEqual(result.returncode, 1, result.stderr)
            self.assertIn("holds port 8441's lock", result.stderr)
            self.assertEqual(record.read_text(), '12345\n')
            self.assertEqual(stop('--dry-run', '--owner-record', str(mine)).returncode, 0)
            result = subprocess.run(['bash', str(ROOT / 'models/glm5next/redeploy.sh'), '--dry-run',
                                     '--checkpoint', temporary, '--owner-record', str(mine)],
                                    env=env, text=True, capture_output=True, timeout=30)
            self.assertEqual(result.returncode, 2)
            self.assertIn('--owner-record applies to --stop', result.stderr)

    def test_redeploys_build_a_diagnostic_bundle_only_for_taps(self):
        env = dict({k: v for k, v in os.environ.items() if not k.startswith('CHOPINFER_')},
                   CHOPINFER_LOCAL_ENV='/dev/null')
        for model, taps, diagnostic in (('qwen38', {}, False), ('qwen38', {'CHOPINFER_TAPS_DIR': '/tmp/taps'}, True),
                                        ('qwen38', {'CHOPINFER_QWEN38_DUMP_DIR': '/tmp/taps'}, True),
                                        ('mimo', {}, False), ('mimo', {'CHOPINFER_TAPS_DIR': '/tmp/taps'}, True)):
            result = subprocess.run(['bash', str(ROOT / f'tools/dev/{model}-redeploy.sh'), '--dry-run',
                                     '--checkpoint', '/data/checkpoint'],
                                    env=dict(env, **taps), text=True, capture_output=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)
            build = [line for line in result.stdout.splitlines() if 'cargo build' in line]
            self.assertEqual(len(build), 1, result.stdout)
            bundle = 'qwen38' if model == 'qwen38' else 'mimo26'
            self.assertEqual(f'bundle-{bundle}/diagnostic-taps' in build[0], diagnostic, (model, taps, build))

    def test_glm5next_redeploy_builds_the_diagnostic_bundle_only_for_taps(self):
        env = dict({k: v for k, v in os.environ.items() if not k.startswith('CHOPINFER_')},
                   CHOPINFER_LOCAL_ENV='/dev/null')
        for taps, diagnostic in (({}, False), ({'CHOPINFER_TAPS_DIR': '/tmp/taps'}, True)):
            result = subprocess.run(['bash', str(ROOT / 'models/glm5next/redeploy.sh'), '--dry-run',
                                     '--checkpoint', '/data/checkpoint'],
                                    env=dict(env, **taps), text=True, capture_output=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)
            build = [line for line in result.stdout.splitlines() if 'cargo build' in line]
            self.assertEqual(len(build), 1, result.stdout)
            self.assertEqual('bundle-glm5next/diagnostic-taps' in build[0], diagnostic, (taps, build))

    def check_cap(self, profile, qualified_at_64):
        # Every profile serves 32 sequences by default (maintainer
        # decision, 2026-10-14); 64 stays qualified and opt-in where evidence exists.
        self.assertEqual(profile['max_sequences'], 32)
        if qualified_at_64:
            self.assertEqual(profile['qualified_max_sequences'], 64)
            self.assertEqual(profile['status'], 'qualified')

    @staticmethod
    def qualified_at_64(path, name, maximum=None):
        """A released model's tp4-ar stays qualified through min(64, its
        package maximum), so through 64 only when its package serves 64
        sequences; the package maximum is the declared
        preview.package_max_sequences."""
        if name != 'tp4-ar' or not released(path):
            return False
        if maximum is None:
            maximum = json.loads(path.with_name('qual.json').read_text())['preview']['package_max_sequences']
        return min(64, maximum) == 64

    def test_a_released_tp4_ar_profile_must_stay_qualified_at_64(self):
        path = ROOT / 'models/qwen38/profiles.json'
        self.assertTrue(released(path))
        self.assertTrue(self.qualified_at_64(path, 'tp4-ar'))
        profile = json.loads(path.read_text())['profiles']['tp4-ar']
        self.check_cap(profile, True)
        # Neither a lost qualification nor a 64 default passes.
        for changed in (dict(profile, status='experimental'), dict(profile, max_sequences=64)):
            with self.assertRaises(AssertionError):
                self.check_cap(changed, True)
        # The qualified count follows the package maximum, not the model.
        self.assertTrue(self.qualified_at_64(path, 'tp4-ar', 64))
        self.assertFalse(self.qualified_at_64(path, 'tp4-ar', 32))

    def test_shipped_profiles(self):
        host = Path(os.environ.get('CARGO_TARGET_DIR', ROOT / 'target')) / 'debug/chopinfer'
        paths = sorted((ROOT / 'models').glob('*/profiles.json'))
        self.assertTrue(paths, 'no shipped profiles found')
        for path in paths:
            data = json.loads(path.read_text())
            self.assertTrue(data['profiles'])
            for name, profile in data['profiles'].items():
                with self.subTest(path=path, name=name):
                    self.assertEqual(profile['kv_cache_dtype'], 'auto')
                    self.assertIsNone(profile['queue_depth'])
                    at_64 = self.qualified_at_64(path, name) or (data['model'], name) == ('mimo26', 'tp4-dflash')
                    self.check_cap(profile, at_64)
                    if (data['model'], name) == ('mimo26', 'tp4-dflash'):
                        # Its pooled speculative accuracy gate was waived by
                        # a maintainer decision, not passed.
                        self.assertIn('accuracy gate waived by a maintainer decision', profile['qualification'])
                    if (data['model'], name) == ('qwen38', 'tp4-mtp'):
                        # Qualified through 32 only, by throughput policy.
                        self.assertEqual(profile['qualified_max_sequences'], 32)
                        self.assertEqual(profile['status'], 'qualified')
                    elif data['model'] == 'glm5next' and name != 'tp4-mtp-k5' and not at_64:
                        # Qualified through 32 by its adopted final gate
                        # (GATES4); the five-draft benchmark arm is not.
                        self.assertEqual(profile['qualified_max_sequences'], 32)
                        self.assertEqual(profile['status'], 'qualified')
                        self.assertIn('GATES4', profile['qualification'])
                    elif not at_64:
                        self.assertIsNone(profile['qualified_max_sequences'])
                    if data['model'] == 'mimo26' and name in ('tp4-mtp', 'tp4-dflash'):
                        self.assertEqual(profile['prefill_budget'], 'tbt')
                        self.assertEqual(profile['decode_tbt_target_ms'], 150)
                    else:
                        self.assertEqual(profile.get('prefill_budget', 'fixed'), 'fixed')
                        self.assertNotIn('decode_tbt_target_ms', profile)
                    if name in ('tp4-mtp', 'tp4-dflash'):
                        spec = profile['speculation']
                        self.assertEqual(spec['mode'], 'adaptive')
                        self.assertEqual(spec['tokens'], 7 if name.endswith('dflash') else 3)
                        self.assertEqual(spec['draft_cost'], 'per-block' if name.endswith('dflash') else 'per-token')
                    for evidence in profile['evidence']:
                        file, _, anchor = evidence.partition('#')
                        self.assertTrue((ROOT / file).is_file(), evidence)
                        if anchor:
                            text = (ROOT / file).read_text()
                            headings = re.findall(r'^#{1,6} (.+)$', text, re.MULTILINE)
                            slugs = [re.sub(r'[^\w -]', '', heading.lower()).replace(' ', '-') for heading in headings]
                            # An explicit anchor keeps an old link working after a heading changes.
                            slugs += re.findall(r'^<a id="([^"]+)"></a>$', text, re.MULTILINE)
                            self.assertIn(anchor, slugs, evidence)
                    result = subprocess.run([host, 'profiles', 'show', path, name], text=True, capture_output=True)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    expected = dict(profile)
                    expected.setdefault('prefill_budget', 'fixed')
                    if expected['prefill_budget'] == 'tbt':
                        expected.setdefault('decode_tbt_target_ms', 100)
                    self.assertEqual(json.loads(result.stdout), expected)
            subprocess.run([host, 'profiles', 'list', path], check=True, capture_output=True)
        for name in ['chopinfer-host', 'mimo26-package', 'qwen38-package', 'chopinfer-model-mimo26', 'chopinfer-model-qwen38',
                     'chopinfer-model-glm5next']:
            result = subprocess.run([host.parent / name, '--help'], text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(result.stdout.strip())
        for args in [['--help'], ['serve', '--help'], ['verify', '--help']]:
            subprocess.run([host, *args], check=True, capture_output=True)

if __name__ == '__main__':
    unittest.main()
