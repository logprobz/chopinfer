"""CPU CI passes tokenizer assets while keeping checkpoint reads opt-in."""
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def isolated_workspace(root):
    """Copy the CPU CI entrypoint and its inventory inputs into root."""
    # Execute the actual entrypoint in an isolated workspace so a local.env
    # in the developer's checkout cannot affect the test.
    (root / 'ci').mkdir()
    for name in ('cpu.sh', 'cpu.py', 'runner.py', 'rust-profile.env', 'model_checks.py', 'preflight.py', 'rust_tests.py'):
        (root / 'ci' / name).write_bytes((ROOT / 'ci' / name).read_bytes())
    (root / 'local.env').write_text('')
    # The checkpoint variable stems come from the model inventories.
    (root / 'tests').mkdir()
    (root / 'tests/model_inventory.py').write_bytes((ROOT / 'tests/model_inventory.py').read_bytes())
    (root / 'tests/model-geometry-recipe.sh').write_bytes((ROOT / 'tests/model-geometry-recipe.sh').read_bytes())
    (root / 'tests/encoder-membership.sh').write_bytes((ROOT / 'tests/encoder-membership.sh').read_bytes())
    (root / 'tools/model_sdk').mkdir(parents=True)
    for module in (ROOT / 'tools/model_sdk').glob('*.py'):
        (root / 'tools/model_sdk' / module.name).write_bytes(module.read_bytes())
    for inventory in (ROOT / 'models').glob('*/model.toml'):
        model = root / inventory.relative_to(ROOT).parent
        model.mkdir(parents=True)
        (model / 'model.toml').write_bytes(inventory.read_bytes())
        for manifest in inventory.parent.glob('*/Cargo.toml'):
            (model / manifest.parent.name).mkdir()
            (model / manifest.parent.name / 'Cargo.toml').write_text('')
        sys.path.insert(0, str(ROOT / 'tools'))
        from model_sdk.inventory import file_inputs, load_model
        declaration = load_model(ROOT, inventory.parent)
        for relative in file_inputs(ROOT, declaration):
            copied = root / relative
            copied.parent.mkdir(parents=True, exist_ok=True)
            copied.write_bytes((ROOT / relative).read_bytes())
        for gate in inventory.parent.glob('contract/tests/encoder-membership.sh'):
            copied = root / gate.relative_to(ROOT)
            copied.parent.mkdir(parents=True, exist_ok=True)
            copied.write_bytes(gate.read_bytes())


class CheckpointEnvironment(unittest.TestCase):
    def test_both_models_follow_the_checkpoint_opt_in(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            isolated_workspace(root)
            cargo = root / 'cargo'
            cargo.write_text('''#!/usr/bin/env python3
import json, os
print(json.dumps({key: value for key, value in os.environ.items()
                  if key.endswith(('_CHECKPOINT', '_TOKENIZER_DIR', '_DFLASH_DIR'))}))
raise SystemExit(77)
''')
            cargo.chmod(0o755)
            for enabled in ('0', '1'):
                env = {key: value for key, value in os.environ.items()
                       if not key.startswith('CHOPINFER_')}
                env.update(PATH=str(root) + os.pathsep + env['PATH'],
                           CHOPINFER_MIMO_CHECKPOINT='/inputs/first',
                           CHOPINFER_QWEN38_CHECKPOINT='/inputs/second',
                           CHOPINFER_QWEN38_TOKENIZER_DIR='/inputs/explicit',
                           CHOPINFER_CI_CHECKPOINT_TESTS=enabled)
                # Inspect the first build's environment without requiring any artifacts.
                result = subprocess.run(['bash', str(root / 'ci/cpu.sh'), '--only', 'profile-cli-build'], env=env,
                                        capture_output=True, text=True, timeout=30)
                self.assertNotEqual(result.returncode, 0, result.stderr)
                self.assertIn('cpu-ci profile-cli-build:', result.stderr)
                self.assertIn('(exit 77)', result.stderr)
                values = json.loads(next(line for line in result.stderr.splitlines() if line.startswith('{')))
                self.assertEqual(values['CHOPINFER_MIMO_TOKENIZER_DIR'], '/inputs/first')
                self.assertEqual(values['CHOPINFER_QWEN38_TOKENIZER_DIR'], '/inputs/explicit')
                for model in ('MIMO', 'QWEN38'):
                    self.assertEqual(f'CHOPINFER_{model}_CHECKPOINT' in values, enabled == '1')


    def test_recipe_keeps_the_ci_tokenizer_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            isolated_workspace(root)
            (root / 'local.env').write_text('''if [[ ${CHOPINFER_ENV_READ:-0} == 1 ]]; then
    export CHOPINFER_QWEN38_CHECKPOINT=/inputs/restored
fi
export CHOPINFER_ENV_READ=1
''')
            (root / 'tests/model_geometry_recipe.py').write_text('''import json, os
print(json.dumps({key: value for key, value in os.environ.items()
                  if key.endswith(('_CHECKPOINT', '_TOKENIZER_DIR', '_DFLASH_DIR'))}))
raise SystemExit(77)
''')
            stubs = root / 'stubs'
            stubs.mkdir()
            (stubs / 'cargo').write_text('#!/bin/sh\nexit 0\n')
            (stubs / 'python3').write_text(f'''#!/bin/sh
case " $* " in
  *" tests/model_inventory.py "*|*" ci/cpu.py "*|*"/tests/model_geometry_recipe.py "*) exec {sys.executable} "$@" ;;
esac
exit 0
''')
            for stub in stubs.iterdir():
                stub.chmod(0o755)
            for enabled in ('0', '1'):
                env = {key: value for key, value in os.environ.items()
                       if not key.startswith('CHOPINFER_')}
                env.update(PATH=str(stubs) + os.pathsep + env['PATH'],
                           CHOPINFER_QWEN38_CHECKPOINT='/inputs/checkpoint',
                           CHOPINFER_QWEN38_TOKENIZER_DIR='/inputs/tokenizer',
                           CHOPINFER_MIMO_TOKENIZER_DIR='/inputs/mimo-tokenizer',
                           CHOPINFER_MIMO_DFLASH_DIR='/inputs/mimo-dflash',
                           CHOPINFER_CI_CHECKPOINT_TESTS=enabled)
                result = subprocess.run(['bash', str(root / 'ci/cpu.sh'), '--only', 'geometry-recipe'], env=env,
                                        capture_output=True, text=True, timeout=30)
                self.assertNotEqual(result.returncode, 0, result.stderr)
                self.assertIn('cpu-ci geometry-recipe:', result.stderr)
                self.assertIn('(exit 77)', result.stderr)
                values = json.loads([line for line in result.stderr.splitlines() if line.startswith('{')][-1])
                self.assertEqual(values['CHOPINFER_QWEN38_TOKENIZER_DIR'], '/inputs/tokenizer')
                self.assertEqual(values['CHOPINFER_MIMO_TOKENIZER_DIR'], '/inputs/mimo-tokenizer')
                self.assertEqual(values['CHOPINFER_MIMO_DFLASH_DIR'], '/inputs/mimo-dflash')
                self.assertEqual('CHOPINFER_QWEN38_CHECKPOINT' in values, enabled == '1')
                if enabled == '1':
                    self.assertEqual(values['CHOPINFER_QWEN38_CHECKPOINT'], '/inputs/checkpoint')


class ExternalSdkScratch(unittest.TestCase):
    def test_the_external_sdk_directory_is_removed_after_a_pass_or_a_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            isolated_workspace(root)
            stubs = root / 'stubs'
            stubs.mkdir()
            # python3 runs the inventory and the gate for real. The smoke step
            # fills its output (with a read-only source copy, as the real smoke
            # writes) and exits with SMOKE_STATUS; the report step that reads
            # it exits with REPORT_STATUS. Cargo emits placeholder build
            # artifacts so the selected setup runs the real profile copy.
            (stubs / 'cargo').write_text(f'''#!{sys.executable}
import os, pathlib, sys
args = sys.argv[1:]
debug = pathlib.Path(os.environ['CARGO_TARGET_DIR']) / 'debug'
for index, argument in enumerate(args):
    if argument in ('--bin', '--example'):
        name = ('examples/' if argument == '--example' else '') + args[index + 1]
    elif argument == '--lib':
        name = 'lib' + args[index - 1].replace('-', '_') + '.so'
    else:
        continue
    artifact = debug / name
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text('build fixture')
''')
            (stubs / 'python3').write_text(f'''#!/bin/sh
case " $* " in
  *" tests/model_inventory.py "*|*" ci/cpu.py "*|*" -c "*) exec {sys.executable} "$@" ;;
  *" tools.model_sdk.smoke "*)
    while [ $# -gt 0 ] && [ "$1" != --output ]; do shift; done
    mkdir -p "$2/target" "$2/model/sdk/crates/core" && echo built >"$2/target/artifact"
    echo source >"$2/model/sdk/crates/core/lib.rs"
    chmod a-w "$2/model/sdk/crates/core/lib.rs" "$2/model/sdk/crates/core"
    exit "$SMOKE_STATUS" ;;
  *" tests/test_kernel_report_integration.py "*) exit "$REPORT_STATUS" ;;
esac
exit 0
''')
            for stub in stubs.iterdir():
                stub.chmod(0o755)
            scratch = root / 'scratch'
            for smoke_status, report_status in (('0', '0'), ('3', '0'), ('0', '4')):
                scratch.mkdir()
                env = {key: value for key, value in os.environ.items()
                       if not key.startswith('CHOPINFER_')}
                env.update(PATH=str(stubs) + os.pathsep + env['PATH'], TMPDIR=str(scratch),
                           CARGO_TARGET_DIR=str(root / 'target'),
                           SMOKE_STATUS=smoke_status, REPORT_STATUS=report_status,
                           # The stubbed cargo runs the workspace tests the reader's reports need.
                           CHOPINFER_RUST_TEST_RUNNER='cargo')
                result = subprocess.run(['bash', str(root / 'ci/cpu.sh'), '--only', 'external-model-smoke',
                                         '--only', 'kernel-report-integration', '--with-producers'], env=env,
                                        capture_output=True, text=True, timeout=30)
                passed = smoke_status == report_status == '0'
                self.assertEqual(result.returncode == 0, passed, result.stderr)
                self.assertIn('==> profile-copy (ok)', result.stderr)
                self.assertIn(f'cpu-ci external-model-smoke: ', result.stderr)
                self.assertIn(f'(exit {smoke_status})', result.stderr)
                self.assertEqual(list(scratch.iterdir()), [])
                subprocess.run(['chmod', '-R', 'u+w', '--', str(scratch)], check=True)
                shutil.rmtree(scratch)


class InheritedGitEnvironment(unittest.TestCase):
    def test_shell_steps_that_run_git_drop_the_callers_git_state(self):
        """Each shell step that runs Git, itself or through its recipe, starts
        its first git or python3 child without the GIT_* state `git rebase -x`
        exports and with the caller's Git configuration ignored."""
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            stubs, scratch = base / 'stubs', base / 'scratch'
            stubs.mkdir()
            scratch.mkdir()
            record = base / 'child.json'
            for name in ('git', 'python3'):
                stub = stubs / name
                stub.write_text(f'''#!{sys.executable}
import json, os, sys
from pathlib import Path
Path({str(record)!r}).write_text(json.dumps({{
    'child': Path(sys.argv[0]).name,
    'git': {{key: value for key, value in os.environ.items() if key.startswith('GIT_')}}}}))
raise SystemExit(42)
''')
                stub.chmod(0o755)
            sentinel = base / 'sentinel/.git'
            env = dict(os.environ, PATH=str(stubs) + os.pathsep + os.environ['PATH'], TMPDIR=str(scratch),
                       GIT_DIR=str(sentinel), GIT_INDEX_FILE=str(sentinel / 'index'), GIT_WORK_TREE=str(base),
                       GIT_CONFIG_GLOBAL=str(base / 'inherited.gitconfig'))
            for script, arguments, child in (('tests/qual-clean.sh', [], 'git'),
                                             ('tests/project-layout.sh', [], 'git'),
                                             ('tests/model-geometry-recipe.sh', ['--contract'], 'python3'),
                                             ('tests/model-export-recipe.sh', ['--ci'], 'python3')):
                with self.subTest(script):
                    record.unlink(missing_ok=True)
                    result = subprocess.run(['bash', str(ROOT / script), *arguments], env=env,
                                            capture_output=True, text=True, timeout=30)
                    self.assertEqual(result.returncode, 42, result.stderr)
                    self.assertEqual(json.loads(record.read_text()), {
                        'child': child, 'git': {'GIT_CONFIG_GLOBAL': os.devnull, 'GIT_CONFIG_NOSYSTEM': '1'}})
            self.assertEqual(list(scratch.iterdir()), [])


class QualificationEnvironment(unittest.TestCase):
    def setUp(self):
        # Isolate fixture writes and Git calls made by the code under test.
        environment = {key: value for key, value in os.environ.items() if not key.startswith('GIT_')}
        environment.update(GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_NOSYSTEM='1')
        isolated = patch.dict(os.environ, environment, clear=True)
        isolated.start()
        self.addCleanup(isolated.stop)
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.root = self.directory / 'checkout'
        (self.root / 'tests').mkdir(parents=True)
        shutil.copyfile(ROOT / 'tests/qual-clean.sh', self.root / 'tests/qual-clean.sh')
        (self.root / 'ci').mkdir()
        shutil.copyfile(ROOT / 'ci/rust-profile.env', self.root / 'ci/rust-profile.env')
        (self.root / '.gitignore').write_text('/local.env\n/ignored-output\n')
        (self.root / 'probe.sh').write_text(
            'if [[ -f local.env ]]; then source local.env; fi\n'
            'printf "%s\\n" "${CHOPINFER_DATA:-unset}"\n')
        # Exercise the gate's real isolated Cargo build without runtime dependencies.
        (self.root / 'Cargo.toml').write_text(
            '[package]\nname = "runtime-host"\nversion = "0.0.0"\nedition = "2021"\n'
            '[[bin]]\nname = "chopinfer"\npath = "src/main.rs"\n')
        (self.root / 'Cargo.lock').write_text(
            'version = 4\n[[package]]\nname = "runtime-host"\nversion = "0.0.0"\n')
        (self.root / 'src').mkdir()
        (self.root / 'src/main.rs').write_text('fn main() {}\n')
        subprocess.run(['git', 'init', '-q', str(self.root)], check=True)
        subprocess.run(['git', '-C', str(self.root), 'add', '--',
                        'tests/qual-clean.sh', 'ci/rust-profile.env', '.gitignore', 'probe.sh',
                        'Cargo.toml', 'Cargo.lock', 'src/main.rs'], check=True)
        subprocess.run(['git', '-C', str(self.root), '-c', 'user.name=Fixture',
                        '-c', 'user.email=fixture@example.invalid', '-c', 'commit.gpgsign=false',
                        'commit', '-qm', 'Add clean qualification fixture'], check=True)
        (self.root / 'local.env').write_text('export CHOPINFER_DATA=checkout-data\n')
        (self.root / 'untracked-source').write_text('not a tracked input')
        self.binaries = self.directory / 'bin'
        self.binaries.mkdir()
        self.scratch = self.directory / 'scratch'
        self.scratch.mkdir()

    def run_clean(self, write_on='', output_name='untracked-output', build_fixture=False, extra_env=None):
        stub = self.binaries / 'python3'
        program = """import json, os, subprocess, sys
from pathlib import Path
print(json.dumps({'env': dict(os.environ), 'args': sys.argv[1:], 'cwd': os.getcwd(),
                  'scorer_built': (Path(os.environ['CARGO_TARGET_DIR']) / 'debug/chopinfer').is_file(),
                  'local_env': Path('local.env').exists(),
                  'untracked_source': Path('untracked-source').exists(),
                  'shell_data': subprocess.check_output(['bash', 'probe.sh'], text=True).strip()}))
label = 'empty' if not os.environ.get('CHOPINFER_DATA') else 'missing'
if label == WRITE_ON:
    Path(OUTPUT_NAME).write_text('unexpected checkout output')
# Like ci/qualification.py, write every module's unittest output to --log.
lines = []
if label == 'missing':
    lines += ["test_numpy (...) ... skipped 'needs numpy (set CHOPINFER_REFRUN_PYTHON)'"] * 20
    lines += ["test_image (...) ... skipped 'no Qwen3.8 image probe workload under $CHOPINFER_DATA'"] * 3
lines.append("test_prompts (...) ... skipped 'needs CHOPINFER_DATA with qwen38-refs/prompts/prompts.json'")
arguments = sys.argv[1:]
Path(arguments[arguments.index('--log') + 1]).write_text('\\n'.join(lines) + '\\n')
"""
        if build_fixture:
            program += "subprocess.run(['cargo', 'check', '--locked', '--offline'], check=True)\n"
        program = program.replace('WRITE_ON', repr(write_on)).replace('OUTPUT_NAME', repr(output_name))
        stub.write_text(f'#!{sys.executable}\n' + program)
        stub.chmod(0o755)
        env = dict(os.environ, PATH=str(self.binaries) + os.pathsep + os.environ['PATH'],
                   TMPDIR=str(self.scratch),
                   CARGO_HOME=os.environ.get('CARGO_HOME', str(Path.home() / '.cargo')),
                   RUSTUP_HOME=os.environ.get('RUSTUP_HOME', str(Path.home() / '.rustup')),
                   CHOPINFER_DATA='inherited-data',
                   CHOPINFER_MIMO_FIXTURES='inherited-fixtures', CHOPINFER_REFRUN_PYTHON='inherited-python',
                   CHOPINFER_OTHER_CHECKPOINT='inherited-checkpoint', CHOPINFER_QUAL_RECEIPTS='inherited-receipts',
                   CHOPINFER_LOCAL_ENV='inherited.env', CARGO_PROFILE_DEV_OPT_LEVEL='3',
                   CARGO_PROFILE_RELEASE_LTO='true')
        if build_fixture:
            env.update(CARGO_TARGET_DIR=str(self.root / 'target'))
        env.update(extra_env or {})
        result = subprocess.run(['bash', str(self.root / 'tests/qual-clean.sh')], env=env,
                                capture_output=True, text=True, timeout=60)
        rows = [json.loads(line) for line in result.stdout.splitlines() if line.startswith('{')]
        self.assertEqual(list(self.scratch.iterdir()), [])
        return result, rows, env

    def test_clean_qual_gate_excludes_local_inputs_and_preserves_build_tools(self):
        profile = dict(line.split('=', 1) for line in (ROOT / 'ci/rust-profile.env').read_text().splitlines()
                       if line and not line.startswith('#'))
        self.assertEqual(profile['CARGO_PROFILE_DEV_OPT_LEVEL'], '1')
        result, rows, env = self.run_clean()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(rows), 2)
        for row in rows:
            clean = row['env']
            self.assertEqual({key for key in clean if key.startswith('CHOPINFER_')},
                             {'CHOPINFER_DATA', 'CHOPINFER_LOCAL_ENV'})
            for key in ('PATH', 'CARGO_HOME', 'RUSTUP_HOME'):
                self.assertEqual(clean[key], env[key])
            self.assertEqual(clean['CARGO_NET_OFFLINE'], 'true')
            self.assertNotEqual(clean['HOME'], os.environ['HOME'])
            self.assertNotEqual(Path(row['cwd']), self.root)
            self.assertTrue(row['scorer_built'])
            self.assertNotEqual(Path(clean['CARGO_TARGET_DIR']), self.root / 'target')
            # Exactly the repository's CI Rust profile, not the caller's.
            self.assertEqual({key: value for key, value in clean.items() if key.startswith('CARGO_PROFILE_')},
                             profile)
            self.assertFalse(row['local_env'])
            self.assertFalse(row['untracked_source'])
        self.assertEqual(rows[0]['env']['CHOPINFER_DATA'], '')
        self.assertTrue(rows[1]['env']['CHOPINFER_DATA'].endswith('/missing-data'))

    def test_clean_qual_gate_does_not_load_checkout_env_in_shells(self):
        result, rows, _ = self.run_clean()
        self.assertEqual(result.returncode, 0, result.stderr)
        for row in rows:
            self.assertEqual(row['shell_data'], row['env']['CHOPINFER_DATA'] or 'unset')

    def test_clean_qual_gate_limits_empty_pass_to_receipt_modules(self):
        result, rows, _ = self.run_clean()
        self.assertEqual(result.returncode, 0, result.stderr)
        for row in rows:
            arguments = row['args']
            self.assertEqual(arguments[:5], ['-B', 'ci/qualification.py', '--no-config', '--receipts', 'inherit'])
            self.assertEqual(arguments[5], '--jobs')
            self.assertTrue(arguments[6].isdigit())
            self.assertEqual(arguments[7], '--log')
        modules = rows[0]['args'][9:]
        self.assertEqual(modules[::2], ['--module'] * 8)
        self.assertEqual(modules[1::2], [
            'test_bench', 'test_fill_batch', 'test_needle', 'test_logprobs_parity', 'test_prefix_cache',
            'test_release_gate', 'test_step_overlap_ab', 'test_review_config'])
        # The missing-data pass runs every module.
        self.assertEqual(rows[1]['args'][9:], [])

    def test_clean_qual_gate_refuses_checkout_writes_after_each_pass(self):
        for write_on in ('empty', 'missing'):
            for output_name in ('untracked-output', 'ignored-output'):
                with self.subTest(pass_name=write_on, output=output_name):
                    result, _, _ = self.run_clean(write_on, output_name)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn('unexpected checkout files', result.stderr)

    def test_clean_qual_gate_reports_skip_causes(self):
        result, _, _ = self.run_clean()
        self.assertEqual(result.returncode, 0, result.stderr)
        output = result.stdout + result.stderr
        self.assertIn("qual-clean empty skips: 1 'needs CHOPINFER_DATA with ", output)
        self.assertIn("qual-clean missing skips: 20 'needs numpy (set CHOPINFER_REFRUN_PYTHON)'", output)
        self.assertIn("qual-clean missing skips: 3 'no Qwen3.8 image probe workload under $CHOPINFER_DATA'", output)

    def test_clean_qual_gate_refuses_a_failed_scorer_build(self):
        (self.root / 'src/main.rs').write_text('compile_error!("scorer build refused");\n')
        result, rows, _ = self.run_clean()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('scorer build refused', result.stderr)
        self.assertEqual(rows, [])

    def test_clean_qual_gate_ignores_an_inherited_repository_and_git_configuration(self):
        """Started as `git rebase -x` starts it, with GIT_DIR and GIT_INDEX_FILE
        naming another repository and the caller's global configuration, the gate
        clones and reads its own checkout and leaves that repository alone."""
        sentinel = self.directory / 'sentinel'
        def git(*args):
            return subprocess.run(['git', '-C', str(sentinel), *args], check=True, capture_output=True).stdout
        sentinel.mkdir()
        git('init', '-q')
        (sentinel / 'kept').write_text('kept\n')
        git('add', 'kept')
        git('-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid', '-c', 'commit.gpgsign=false',
            'commit', '-qm', 'Sentinel')
        (sentinel / 'staged-only').write_text('staged\n')
        git('add', 'staged-only')
        def state():
            return ((sentinel / '.git/index').read_bytes(), (sentinel / '.git/config').read_bytes(),
                    git('log', '--all', '--format=%H %s'), git('worktree', 'list', '--porcelain'))
        before = state()
        global_config = self.directory / 'inherited.gitconfig'
        global_config.write_text('[clone]\n\tdefaultRemoteName = inherited\n')
        user_config = self.directory / 'config/git/config'
        user_config.parent.mkdir(parents=True)
        user_config.write_text('[clone]\n\tdefaultRemoteName = user\n')
        # Stop at the scorer build: record the clone's remotes and index.
        marker = self.directory / 'clone.json'
        cargo = self.binaries / 'cargo'
        cargo.write_text(f'''#!{sys.executable}
import json, subprocess
from pathlib import Path
git = lambda *args: subprocess.check_output(['git', *args], text=True).split()
Path({str(marker)!r}).write_text(json.dumps({{'remotes': git('remote'), 'index': git('ls-files', '--cached')}}))
raise SystemExit(42)
''')
        cargo.chmod(0o755)
        result, _, _ = self.run_clean(extra_env={
            'GIT_DIR': str(sentinel / '.git'), 'GIT_INDEX_FILE': str(sentinel / '.git/index'),
            'GIT_CONFIG_GLOBAL': str(global_config), 'XDG_CONFIG_HOME': str(self.directory / 'config')})
        self.assertEqual(state(), before)
        self.assertEqual(result.returncode, 42, result.stdout + result.stderr)
        clone = json.loads(marker.read_text())
        self.assertEqual(clone['remotes'], ['origin'])
        self.assertEqual(clone['index'], sorted([
            '.gitignore', 'Cargo.lock', 'Cargo.toml', 'ci/rust-profile.env', 'probe.sh', 'src/main.rs',
            'tests/qual-clean.sh']))

    def test_clean_build_does_not_reuse_a_deleted_source_root(self):
        (self.root / 'build.rs').write_text(
            'fn main() {\n'
            '    println!("cargo:rerun-if-env-changed=QUAL_BUILD_MODE");\n'
            '    std::fs::canonicalize(env!("CARGO_MANIFEST_DIR"))\n'
            '        .expect("resolve fixture source root");\n'
            '}\n')
        subprocess.run(['git', '-C', str(self.root), 'add', '--',
                        'build.rs'], check=True)
        subprocess.run(['git', '-C', str(self.root), '-c', 'user.name=Fixture',
                        '-c', 'user.email=fixture@example.invalid', '-c', 'commit.gpgsign=false',
                        'commit', '-qm', 'Add source-root build fixture'], check=True)
        result, rows, env = self.run_clean(build_fixture=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(len(rows), 2)
        # Rerun the cached build script after the clean copy has been removed.
        env.update(QUAL_BUILD_MODE='configured', CARGO_INCREMENTAL='0')
        configured = subprocess.run(['cargo', 'check', '--locked', '--offline'],
                                    cwd=self.root, env=env, capture_output=True, text=True, timeout=30)
        self.assertEqual(configured.returncode, 0, configured.stdout + configured.stderr)



class GraphTests(unittest.TestCase):
    def test_the_graph_names_every_step_and_its_prerequisites_and_runs_nothing(self):
        with tempfile.TemporaryDirectory() as work:
            result = subprocess.run([sys.executable, '-B', str(ROOT / 'ci/cpu.py'), '--graph', '--work', work],
                                    capture_output=True, text=True, timeout=120, cwd=ROOT)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(list(Path(work).iterdir()), [])
        graph = json.loads(result.stdout)
        self.assertEqual(graph['release-gate'], ['profile-copy'])
        self.assertEqual(graph['profile-copy'], ['preparation-fixture-build'])
        self.assertEqual(graph['release-gate-clean'], [])
        self.assertTrue(all(set(after) <= set(graph) for after in graph.values()))


class Preflight(unittest.TestCase):
    """What a run needs is checked before its first step, and HEAD may not
    move while it runs."""

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        isolated_workspace(self.root)
        stubs = self.root / 'stubs'
        stubs.mkdir()
        # Every step's cargo leaves a mark, and with MOVE commits there first.
        # The preflight's nextest probe leaves none; NO_NEXTEST hides nextest.
        (stubs / 'cargo').write_text('#!/bin/sh\n'
                                     'if [ "$1 $2" = "nextest --version" ]; then\n'
                                     '  [ -z "$NO_NEXTEST" ] || exit 101\n'
                                     '  echo "cargo-nextest 0.9.146"; exit 0\n'
                                     'fi\n'
                                     'touch "$MARK"\n'
                                     '[ -z "$MOVE" ] || git -C "$MOVE" commit -q --allow-empty -m moved\n')
        (stubs / 'cargo').chmod(0o755)
        self.mark = self.root / 'cargo-ran'
        self.scratch = self.root / 'scratch'
        self.scratch.mkdir()
        self.env = {key: value for key, value in os.environ.items() if not key.startswith(('CHOPINFER_', 'GIT_'))}
        self.env.update(PATH=str(stubs) + os.pathsep + self.env['PATH'], TMPDIR=str(self.scratch),
                        MARK=str(self.mark), CARGO_TARGET_DIR=str(self.root / 'target'),
                        GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_NOSYSTEM='1', GIT_AUTHOR_NAME='ci',
                        GIT_AUTHOR_EMAIL='ci@example.invalid', GIT_COMMITTER_NAME='ci',
                        GIT_COMMITTER_EMAIL='ci@example.invalid')

    def gate(self, *args, **variables):
        return subprocess.run(['bash', str(self.root / 'ci/cpu.sh'), *args], env={**self.env, **variables},
                              capture_output=True, text=True, timeout=60)

    def assertRefused(self, result, *reasons):
        self.assertEqual(result.returncode, 2, result.stderr)
        for reason in reasons:
            self.assertIn('cpu-ci preflight: ' + reason, result.stderr)
        self.assertIn('cpu-ci preflight failed: nothing ran', result.stderr)
        self.assertFalse(self.mark.exists())

    def test_an_unusable_tmpdir_stops_the_run_before_any_step(self):
        (self.root / 'file').write_text('')
        long = self.root / ('t' * (70 - len(str(self.root))))
        long.mkdir()
        cases = [('missing', str(self.root / 'missing'), 'is not a directory: create it first'),
                 ('file', str(self.root / 'file'), 'is not a directory'),
                 ('relative', 'scratch', 'is not an absolute path'),
                 ('long', str(long), f'is {len(str(long))} bytes, more than 64')]
        if os.geteuid() != 0:
            locked = self.root / 'locked'
            locked.mkdir(mode=0o500)
            cases.append(('read-only', str(locked), 'is not writable'))
        for label, value, reason in cases:
            with self.subTest(tmpdir=label):
                self.assertRefused(self.gate('--only', 'profile-cli-build', TMPDIR=value), f'TMPDIR={value} {reason}')
        self.assertEqual(self.gate('--only', 'profile-cli-build').returncode, 0)
        self.assertTrue(self.mark.exists())

    def test_a_pinned_reference_under_an_unset_variable_stops_the_steps_that_read_references(self):
        # Every pinned entry kept under a variable with no default is needed,
        # except under the data root.
        for model, entries in (('qwen38', {'reference:long': {'path': {'env': 'CHOPINFER_TEST_LONG', 'path': 'r.json'}},
                                           'workload:long': {'path': {'env': 'CHOPINFER_TEST_LONG', 'path': 'w.json'}},
                                           'gsm8k': {'path': {'env': 'CHOPINFER_TEST_FIXTURES', 'default': 'fixtures',
                                                              'path': 'gsm8k'}}}),
                               ('mimo26', {'layer_reference': {'path': {'env': 'CHOPINFER_TEST_LAYERS', 'path': '.'}},
                                           'needle': {'path': {'env': 'CHOPINFER_DATA', 'path': 'n.json'}}})):
            (self.root / 'models' / model / 'qual-references.lock.json').write_text(json.dumps({'entries': entries}))
        long = 'CHOPINFER_TEST_LONG is unset; it names the directory that holds the pinned qwen38 reference:long, qwen38 workload:long'
        layers = 'CHOPINFER_TEST_LAYERS is unset; it names the directory that holds the pinned mimo26 layer_reference'
        for step in ('portable', 'release-gate'):
            with self.subTest(step=step):
                self.assertRefused(self.gate('--preflight', '--only', step), layers, long)
                self.assertRefused(self.gate('--only', step), layers, long)
                self.assertRefused(self.gate('--only', step, CHOPINFER_TEST_LAYERS=str(self.scratch)), long)
        self.assertNotIn('CHOPINFER_DATA', self.gate('--preflight').stderr)
        self.assertNotIn('CHOPINFER_TEST_FIXTURES', self.gate('--preflight').stderr)
        file = self.root / 'file'
        self.assertRefused(self.gate('--preflight', CHOPINFER_TEST_LONG=str(file), CHOPINFER_TEST_LAYERS=str(self.scratch)),
                           f'CHOPINFER_TEST_LONG={file} is not a directory; it holds the pinned qwen38 reference:long')
        result = self.gate('--preflight', CHOPINFER_TEST_LONG=str(self.scratch), CHOPINFER_TEST_LAYERS=str(self.scratch))
        self.assertEqual((result.returncode, result.stderr.strip()), (0, 'cpu-ci preflight: ok'))
        # A selection that reads no reference does not need it.
        self.assertEqual(self.gate('--only', 'leak-guard', '--preflight').returncode, 0)

    def test_a_reference_workload_under_a_set_data_root_stops_the_steps_that_read_references(self):
        # runtime-eval's reference tests verify a workload kept under the data
        # root whenever the data root is set, so a set data root must hold it.
        for model in ('qwen38', 'mimo26', 'glm5next'):
            (self.root / 'models' / model).mkdir(parents=True, exist_ok=True)
            (self.root / 'models' / model / 'qual-references.lock.json').write_text(json.dumps({'entries': {}}))
            (self.root / 'models' / model / 'qual.json').write_text(json.dumps({'references': {}}))
        (self.root / 'models/qwen38/qual.json').write_text(json.dumps({'references': {
            'image_probes': {'workload': {'env': 'CHOPINFER_DATA', 'path': 'refs/probes.workload.json'}},
            'long': {'workload': {'env': 'CHOPINFER_TEST_LONG', 'path': 'w.json'}}}}))
        (self.root / 'models/mimo26/qual.json').write_text(json.dumps({'references': {}, 'configurations': {
            'tp2': {'references': {'media': {'workload': {'env': 'CHOPINFER_DATA', 'path': 'media.json'}}}}}}))
        data = self.root / 'data'
        data.mkdir()
        probes = (f'CHOPINFER_DATA={data} holds no refs/probes.workload.json: the workload of the pinned '
                  'qwen38 image_probes reference')
        media = f'CHOPINFER_DATA={data} holds no media.json: the workload of the pinned mimo26 media reference'
        for step in ('portable', 'release-gate'):
            with self.subTest(step=step):
                self.assertRefused(self.gate('--only', step, CHOPINFER_DATA=str(data)), probes, media)
        # Unset, the tests expect those references to be insufficient; a
        # selection that reads no reference does not need them.
        self.assertEqual(self.gate('--preflight').returncode, 0)
        self.assertEqual(self.gate('--only', 'leak-guard', '--preflight', CHOPINFER_DATA=str(data)).returncode, 0)
        (data / 'refs').mkdir()
        (data / 'refs/probes.workload.json').write_text('{}')
        (data / 'media.json').write_text('{}')
        result = self.gate('--preflight', CHOPINFER_DATA=str(data))
        self.assertEqual((result.returncode, result.stderr.strip()), (0, 'cpu-ci preflight: ok'))

    def test_a_missing_nextest_stops_the_workspace_tests_unless_the_run_opts_out(self):
        missing = 'cargo-nextest is not installed; the gate runs the workspace tests under cargo-nextest'
        self.assertRefused(self.gate('--only', 'portable', NO_NEXTEST='1'), missing)
        self.assertRefused(self.gate('--preflight', NO_NEXTEST='1'), missing)
        self.assertRefused(self.gate('--only', 'portable', CHOPINFER_RUST_TEST_RUNNER='nextest', NO_NEXTEST='1'),
                           missing)
        self.assertRefused(self.gate('--only', 'portable', CHOPINFER_RUST_TEST_RUNNER='fast'),
                           'CHOPINFER_RUST_TEST_RUNNER=fast is not auto, nextest or cargo')
        for runner in ('cargo', 'auto'):
            with self.subTest(runner=runner):
                result = self.gate('--preflight', '--only', 'portable', CHOPINFER_RUST_TEST_RUNNER=runner, NO_NEXTEST='1')
                self.assertEqual((result.returncode, result.stderr.strip()), (0, 'cpu-ci preflight: ok'))
        # A selection without the workspace tests does not need nextest.
        self.assertEqual(self.gate('--only', 'leak-guard', '--preflight', NO_NEXTEST='1').returncode, 0)

    def test_the_workspace_tests_run_under_nextest_unless_the_caller_chose(self):
        sys.path.insert(0, str(ROOT / 'ci'))
        import cpu
        for value, expected in ((None, 'nextest'), ('', 'nextest'), ('cargo', 'cargo'), ('auto', 'auto')):
            env = {key: item for key, item in os.environ.items() if key != 'CHOPINFER_RUST_TEST_RUNNER'}
            if value is not None:
                env['CHOPINFER_RUST_TEST_RUNNER'] = value
            with self.subTest(runner=value), patch.dict(os.environ, env, clear=True), \
                    tempfile.TemporaryDirectory() as directory:
                shared, independent = cpu.steps(Path(directory) / 'target', Path(directory), 4, 4)
                portable = next(step for step in shared + independent if step.name == 'portable')
                self.assertEqual(portable.env['CHOPINFER_RUST_TEST_RUNNER'], expected)

    def test_head_moving_during_the_run_fails_it(self):
        git = ['git', '-C', str(self.root)]
        for command in (['init', '-q'], ['add', '-A'], ['commit', '-q', '-m', 'base']):
            subprocess.run(git + command, env=self.env, check=True, capture_output=True)
        head = subprocess.run(git + ['rev-parse', 'HEAD'], env=self.env, check=True, capture_output=True,
                              text=True).stdout.strip()
        result = self.gate('--only', 'profile-cli-build', MOVE=str(self.root))
        moved = subprocess.run(git + ['rev-parse', 'HEAD'], env=self.env, check=True, capture_output=True,
                               text=True).stdout.strip()
        self.assertNotEqual(moved, head)
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn('==> profile-cli-build (ok)', result.stderr)
        self.assertIn(f'cpu-ci failed: HEAD moved during the run from {head} to {moved}', result.stderr)
        self.assertEqual(self.gate('--only', 'profile-cli-build').returncode, 0)

    def test_the_head_watch_stops_the_run_once_head_moves_and_never_after_it_is_stopped(self):
        sys.path.insert(0, str(ROOT / 'ci'))
        import cpu
        git = ['git', '-C', str(self.root)]
        for command in (['init', '-q'], ['commit', '-q', '--allow-empty', '-m', 'base']):
            subprocess.run(git + command, env=self.env, check=True, capture_output=True)
        received = threading.Event()
        previous = signal.signal(signal.SIGTERM, lambda *_: received.set())
        self.addCleanup(signal.signal, signal.SIGTERM, previous)
        with patch.dict(os.environ, self.env, clear=True):
            head = cpu.preflight.head(self.root)
            watch = cpu.HeadWatch(self.root, head, seconds=0.05)
            watch.start()
            time.sleep(0.3)
            self.assertFalse(received.is_set())
            subprocess.run(git + ['commit', '-q', '--allow-empty', '-m', 'moved'], check=True, capture_output=True)
            watch.join(timeout=30)
            self.assertTrue(received.wait(30))
            self.assertEqual(watch.moved, cpu.preflight.head(self.root))
            received.clear()
            stopped = cpu.HeadWatch(self.root, head, seconds=0.05)
            stopped.start()
            stopped.stop()
            stopped.join(timeout=30)
            self.assertFalse(stopped.is_alive())
            self.assertIsNone(stopped.moved)
            self.assertFalse(received.is_set())


if __name__ == '__main__':
    unittest.main()
