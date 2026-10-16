"""CPU checks for deployment artifacts and documented commands."""
import fnmatch
import hashlib
import importlib.util
import json
import os
import re
import shlex
import shutil
import sys
from unittest.mock import patch
import subprocess
import tempfile
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.model_sdk import inventory
COMMON = ROOT / 'tools/dev/redeploy-common.sh'


class DeploymentArtifacts(unittest.TestCase):
    def test_template_instances_keep_devices_ports_and_checkpoints_separate(self):
        unit = (ROOT / 'deploy/systemd/chopinfer@.service').read_text()
        self.assertIn('EnvironmentFile=/etc/chopinfer/%i.env', unit)
        self.assertIn('ExecStart=/opt/chopinfer/bin/entrypoint.sh $CHOPINFER_SERVE_ARGS', unit)
        environments = []
        for model, devices, port in [('mimo26', '0,1', '8311'), ('qwen38', '2,3', '8431')]:
            text = (ROOT / f'deploy/systemd/{model}-tp2.env.example').read_text()
            env = dict(line.split('=', 1) for line in text.splitlines()
                       if line and not line.startswith('#'))
            self.assertEqual(env['CUDA_VISIBLE_DEVICES'], devices)
            self.assertEqual(env['CHOPINFER_LISTEN'], f'127.0.0.1:{port}')
            self.assertEqual(env['CHOPINFER_TENSOR_PARALLEL_SIZE'], '2')
            self.assertEqual(env['CHOPINFER_MODEL_CACHE'], '/var/lib/chopinfer/model-cache')
            environments.append(env)
        self.assertNotEqual(*(env['CHOPINFER_MODEL'] for env in environments))

    def test_systemd_examples_name_a_checkpoint_and_pass_validation(self):
        spec = importlib.util.spec_from_file_location('deployment', ROOT / 'deploy/docker/deployment.py')
        deployment = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(deployment)
        for path in sorted((ROOT / 'deploy/systemd').glob('*.env.example')):
            with self.subTest(path=path.name):
                text = path.read_text()
                env = dict(line.split('=', 1) for line in text.splitlines()
                           if line and not line.startswith('#'))
                self.assertTrue(env['CHOPINFER_MODEL'].startswith('/var/lib/chopinfer/models/'))
                self.assertEqual(env['CHOPINFER_MODEL_CACHE'], '/var/lib/chopinfer/model-cache')
                for gone in ('PACKAGE', 'INSTALL_ROOT', 'BUNDLE_VERIFY_CACHE_DIR',
                             'drafter images', 'text-only'):
                    self.assertNotIn(gone, text)
                with patch.dict(os.environ, env, clear=True):
                    deployment.validate()
        for model in inventory.released(ROOT):
            profiles = json.loads((ROOT / 'models' / model / 'profiles.json').read_text())['profiles']
            bundles = {profile['bundle_id'] for profile in profiles.values()}
            unit = (ROOT / f'deploy/systemd/chopinfer-{model}.service').read_text()
            expected = re.search(r'CHOPINFER_EXPECTED_BUNDLE_ID=(\S+) /opt/chopinfer/bin/entrypoint\.sh', unit)
            self.assertIn(expected and expected.group(1), bundles)
            self.assertIn('ReadWritePaths=/var/lib/chopinfer', unit)

    def test_image_ships_the_host_helpers_modules_and_profiles(self):
        dockerfile = (ROOT / 'deploy/docker/Dockerfile').read_text()
        runtime = dockerfile.split('FROM ${RUNTIME_IMAGE} AS runtime\n')[1]
        self.assertIn('bash tools/release/build.sh /opt/archive', dockerfile)
        self.assertIn('COPY LICENSE /usr/share/doc/chopinfer/LICENSE', runtime)
        self.assertIn('/usr/share/doc/chopinfer/THIRD_PARTY_NOTICES', runtime)
        self.assertIn('COPY --from=build /opt/release/ /opt/chopinfer/', runtime)
        self.assertIn('ENV CHOPINFER_MODEL=/data/model CHOPINFER_MODEL_CACHE=/data/model-cache', runtime)
        self.assertIn('USER chopinfer', runtime)

    def test_image_checks_selected_libraries_and_owns_no_model_policy(self):
        # Each package carries its model's deployment policy; the image keeps
        # no per-bundle file.
        dockerfile = (ROOT / 'deploy/docker/Dockerfile').read_text()
        self.assertIn('tools/release/plan.py --view libraries', dockerfile)
        self.assertIn('ldd "/opt/release/$library"', dockerfile)
        self.assertNotIn('ldd /opt/release/lib/libbundle_', dockerfile)
        self.assertNotIn('defaults', dockerfile)
        self.assertFalse((ROOT / 'deploy/docker/defaults').exists())

    def test_docker_selection_shell_checks_libraries_and_propagates_failures(self):
        from test_release_policy import inventory_fixture
        dockerfile = (ROOT / 'deploy/docker/Dockerfile').read_text()
        start = dockerfile.index('    python3 -B tools/release/plan.py --view libraries')
        block = dockerfile[start:dockerfile.index('\n\n', start)].replace('\\\n', '')
        for failure, status in ((None, 0), ('library', 23)):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                metadata = inventory_fixture(root, models=('first', 'second', 'unselected'))
                (root / 'tools/release').mkdir(parents=True)
                (root / 'tools/model_sdk').mkdir()
                shutil.copyfile(ROOT / 'tools/release/plan.py', root / 'tools/release/plan.py')
                shutil.copyfile(ROOT / 'tools/model_sdk/inventory.py', root / 'tools/model_sdk/inventory.py')
                release = root / 'release'
                (release / 'bin').mkdir(parents=True)
                (release / 'lib').mkdir()
                for relative in ('bin/chopinfer', 'lib/libbundle_first.so', 'lib/libbundle_second.so', 'lib/libbundle_unselected.so'):
                    (release / relative).write_bytes(b'fixture artifact')
                programs = root / 'programs'
                programs.mkdir()
                cargo = programs / 'cargo'
                cargo.write_text('#!/usr/bin/env python3\nimport json\n' + f'print(json.dumps({metadata!r}))\n')
                ldd = programs / 'ldd'
                ldd.write_text(r"""#!/usr/bin/env python3
import json, os, sys
from pathlib import Path
with Path(os.environ['LDD_CALLS']).open('a') as log:
    log.write(json.dumps(sys.argv[1:]) + '\n')
if Path(sys.argv[1]).name == os.environ.get('FAIL_LIBRARY'):
    raise SystemExit(23)
assert Path(sys.argv[1]).is_file()
""")
                for path in (cargo, ldd):
                    path.chmod(0o755)
                checked = root / 'checked.jsonl'
                env = dict(os.environ, PATH=str(programs) + ':' + os.environ['PATH'], LDD_CALLS=str(checked),
                           FAIL_LIBRARY='libbundle_first.so' if failure == 'library' else '')
                command = block
                for before, after in (('/opt/release', str(release)),
                                      ('/tmp/release-libraries', str(root / 'libraries.tsv'))):
                    command = command.replace(before, after)
                result = subprocess.run(['sh', '-c', 'umask 077; ' + command], cwd=root, env=env,
                                        text=True, capture_output=True, timeout=10)
                self.assertEqual(result.returncode, status, result.stderr)
                expected_libraries = ['bin/chopinfer', 'lib/libbundle_first.so']
                if failure != 'library':
                    expected_libraries.append('lib/libbundle_second.so')
                self.assertEqual([json.loads(line) for line in checked.read_text().splitlines()],
                                 [[str(release / name)] for name in expected_libraries])

    def test_redeploy_installs_declared_names_and_refuses_unselected_models(self):
        from test_release_policy import inventory_fixture
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata = inventory_fixture(root, models=('third', 'unselected'))
            path = root / 'models/third/model.toml'
            replacements = {'third-package': 'assemble-third', 'chopinfer-model-third': 'chopinfer-model-custom', 'bundle_third': 'native_custom'}
            text = path.read_text()
            for before, after in replacements.items():
                text = text.replace(before, after)
                for package in metadata['packages']:
                    for target in package['targets']:
                        if target['name'] == before:
                            target['name'] = after
            path.write_text(text)
            (root / 'tools/dev').mkdir(parents=True)
            shutil.copyfile(COMMON, root / 'tools/dev/redeploy-common.sh')
            (root / 'tools/release').mkdir()
            shutil.copyfile(ROOT / 'tools/release/plan.py', root / 'tools/release/plan.py')
            (root / 'tools/model_sdk').mkdir()
            shutil.copyfile(ROOT / 'tools/model_sdk/inventory.py', root / 'tools/model_sdk/inventory.py')
            release = root / 'release'
            release.mkdir()
            names = ('chopinfer', 'chopinfer-model-custom', 'libnative_custom.so', 'assemble-third')
            for name in names:
                (release / name).write_bytes(name.encode())
            programs = root / 'programs'
            programs.mkdir()
            cargo = programs / 'cargo'
            cargo.write_text('#!/usr/bin/env python3\nimport json\n'+f'print(json.dumps({metadata!r}))\n')
            cargo.chmod(0o755)
            env = dict(os.environ, PATH=str(programs) + ':' + os.environ['PATH'])
            command = ['bash', '-c', 'set -e; source "$1"; install_serving_binaries "$2" "$3" "$4"; install_redeploy_extras "$2" "$3" "$4"',
                       'test-install', str(root / 'tools/dev/redeploy-common.sh'), str(release), str(root / 'install'), 'third']
            result = subprocess.run(command, env=env, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            expected = ('bin/chopinfer', 'bin/chopinfer-model-custom', 'lib/libnative_custom.so', 'bin/assemble-third')
            self.assertEqual({path.relative_to(root / 'install').as_posix() for path in (root / 'install').rglob('*') if path.is_file()}, set(expected))
            for relative in expected:
                path = root / 'install' / relative
                self.assertEqual(path.read_bytes(), path.name.encode())
                self.assertEqual(path.stat().st_mode & 0o777, 0o755)
            result = subprocess.run([*command[:-1], 'unselected'], env=env, text=True, capture_output=True)
            self.assertEqual(result.returncode, 2)
            self.assertIn("model 'unselected' is not selected for release", result.stderr)
            (release / 'libnative_custom.so').unlink()
            result = subprocess.run(command, env=env, text=True, capture_output=True)
            self.assertEqual(result.returncode, 1)
            self.assertIn('missing build output', result.stderr)

    def test_local_files_are_excluded_from_context(self):
        patterns = (ROOT / '.dockerignore').read_text().splitlines()
        for name in ('local.env', '.env', '.env.production', 'local.env.private',
                     '.venv', 'venv', 'logs', 'startup.log', '.aws', '.ssh', '.netrc',
                     '.docker', '.cargo/credentials.toml', '.pypirc', '.npmrc', '.local-tool', '.gitignore', 'leak-denylist.local'):
            with self.subTest(name=name):
                self.assertTrue(any(fnmatch.fnmatch(name, p) for p in patterns), name)

    def test_healthcheck_follows_listener(self):
        dockerfile = (ROOT / 'deploy/docker/Dockerfile').read_text()
        command = dockerfile.split('    CMD ', 1)[1].splitlines()[0]
        command = command.replace('/opt/chopinfer/bin', str(ROOT / 'deploy/docker'))
        with tempfile.TemporaryDirectory() as directory:
            curl = Path(directory) / 'curl'
            curl.write_text('#!/usr/bin/env python3\nimport json, sys\nprint(json.dumps(sys.argv[1:]))\n')
            curl.chmod(0o755)
            for listen, url in [('0.0.0.0:8000', 'http://127.0.0.1:8000/health'),
                                ('0.0.0.0:9000', 'http://127.0.0.1:9000/health'),
                                ('192.0.2.4:8001', 'http://192.0.2.4:8001/health'),
                                ('[::]:9000', 'http://[::1]:9000/health'),
                                ('[::1]:8001', 'http://[::1]:8001/health')]:
                with self.subTest(listen=listen):
                    result = subprocess.run(['bash', '-c', command], text=True, capture_output=True,
                                            env=dict(os.environ, CHOPINFER_LISTEN=listen,
                                                     PATH=directory + ':' + os.environ['PATH']))
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn(url, json.loads(result.stdout))

    def test_fatal_alert_uses_independent_restart_history(self):
        text = (ROOT / 'docs/operations-observability.md').read_text()
        rule = text.split('- alert: ChopinferWorkerFailed\n')[1].split('- alert:', 1)[0]
        self.assertIn('increase(chopinfer_supervisor_restarts_total', rule)
        self.assertNotIn('for:', rule)
        self.assertIn('absent(up{job="chopinfer"})', text)
        self.assertIn('chopinfer_supervisor_start_time_seconds', text)
        self.assertIn('not metrics served by Chopinfer', text)

    def test_startup_deadlines_cover_cold_authentication(self):
        self.assertIn('--start-period=600s', (ROOT / 'deploy/docker/Dockerfile').read_text())
        for model in inventory.released(ROOT):
            self.assertIn('timeout 600s curl --retry 600',
                          (ROOT / f'docs/deploy-{model}.md').read_text())

    def test_documented_release_gate_commands_parse(self):
        document = (ROOT / 'docs/serving-mimo26.md').read_text()
        commands = re.findall(r'`(?:python3 )?tools/qual/release_gate.py ([^`]+)`', document)
        self.assertTrue(commands)
        sys.path.insert(0, str(ROOT / 'tools/qual'))
        import release_gate
        with tempfile.TemporaryDirectory() as directory:
            inputs = Path(directory) / 'release-inputs.json'
            inputs.write_text('{}')
            for arguments in commands:
                invocation = document[document.index('tools/qual/release_gate.py ' + arguments) - 8:]
                self.assertTrue(invocation.startswith('python3 '), invocation[:80])
                command = ['python3', 'tools/qual/release_gate.py', *shlex.split(arguments)]
                result = subprocess.run([*command, '--help'], cwd=ROOT, text=True, capture_output=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                argv = [str(inputs) if word == 'release-inputs.json' else word for word in command[1:]]
                with patch.object(sys, 'argv', argv), patch.object(release_gate, 'run', return_value=0) as run:
                    self.assertEqual(release_gate.main(), 0)
                    self.assertEqual(run.call_args.args[0]['model'], 'mimo26')
                    self.assertEqual(run.call_args.args[2], command[command.index('--profile') + 1])

    def test_revision_requires_a_full_identity_and_rejects_disagreement(self):
        spec = importlib.util.spec_from_file_location('deployment_build', ROOT / 'deploy/docker/build.py')
        build = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(build)
        revision = 'a' * 40
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            root = Path(directory)
            with patch.object(build, 'ROOT', root):
                with self.assertRaisesRegex(ValueError, 'full source revision required'):
                    build.revision()
                with patch.dict(os.environ, CHOPINFER_BUILD_REVISION=revision):
                    with self.assertRaisesRegex(ValueError, 'full source revision required'):
                        build.revision()
                git = root / '.git'
                (git / 'refs/heads').mkdir(parents=True)
                (git / 'HEAD').write_text('ref: refs/heads/main\n')
                (git / 'refs/heads/main').write_text(revision + '\n')
                self.assertEqual(build.revision(), revision)
                (git / 'refs/heads/main').unlink()
                (git / 'packed-refs').write_text(revision + ' refs/heads/main\n')
                self.assertEqual(build.revision(), revision)
                (git / 'HEAD').write_text(revision + '\n')
                self.assertEqual(build.revision(), revision)
                with patch.dict(os.environ, CHOPINFER_BUILD_REVISION='b' * 40):
                    with self.assertRaisesRegex(ValueError, 'differs from Git HEAD'):
                        build.revision()
                (git / 'HEAD').write_text('unknown\n')
                with self.assertRaisesRegex(ValueError, 'full source revision required'):
                    build.revision()

    def test_release_build_records_validated_inputs(self):
        spec = importlib.util.spec_from_file_location('deployment_build', ROOT / 'deploy/docker/build.py')
        build = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(build)
        env = {name: f'example/{name.lower()}@sha256:' + 'a' * 64
               for name in ('CUDA_DEVEL_IMAGE', 'CUDA_RUNTIME_IMAGE', 'BUILDKIT_SYNTAX')}
        with patch.dict(os.environ, env, clear=True), patch.object(sys, 'argv', ['build.py', '-t', 'test']), \
             patch.object(build.subprocess, 'run') as run:
            build.main()
            command = run.call_args.args[0]
            self.assertEqual(command[:4], ['docker', 'buildx', 'build', '--load'])
            self.assertIn('CARGO_LOCK_SHA256=' + hashlib.sha256((ROOT / 'Cargo.lock').read_bytes()).hexdigest(), command)
            self.assertIn('RUST_TOOLCHAIN=1.89.0', command)
            for name, value in env.items():
                self.assertIn(f'{name}={value}', command)
            checked = build.inputs()
        with patch.dict(os.environ, {**checked, 'CARGO_LOCK_SHA256': '0' * 64}, clear=True), \
             patch.object(sys, 'argv', ['build.py', '--check']):
            with self.assertRaisesRegex(ValueError, 'CARGO_LOCK_SHA256'):
                build.main()
        with patch.dict(os.environ, {**env, 'BUILD_IMAGE': 'mutable:tag'}, clear=True):
            with self.assertRaisesRegex(ValueError, 'BUILD_IMAGE'):
                build.inputs()
        for name in env:
            with patch.dict(os.environ, {**env, name: 'mutable:tag'}, clear=True):
                with self.assertRaisesRegex(ValueError, name):
                    build.inputs()
        dockerfile = (ROOT / 'deploy/docker/Dockerfile').read_text()
        for label in ('cuda-devel-image', 'cuda-runtime-image', 'build-image', 'runtime-image',
                      'frontend-image', 'cuda-compiler'):
            self.assertIn('org.chopinfer.build.' + label + '=', dockerfile)
        for label in ('rust-toolchain', 'cargo-lock-sha256'):
            self.assertNotIn('org.chopinfer.build.' + label + '=', dockerfile)


if __name__ == '__main__':
    unittest.main()
