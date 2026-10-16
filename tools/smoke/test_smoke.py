#!/usr/bin/env python3
"""tools/smoke/smoke.py: legs from serve profiles, run files, job packing,
smoke-staleness paths, and `run` end to end on the serving runner's CPU
fixtures (tools/ab/serving/stubs.py). No GPU, no build."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
SERVING = HERE.parent / 'ab/serving'
sys.path.insert(0, str(HERE))
import smoke  # noqa: E402

sys.path.insert(1, str(SERVING))
from common import digest, read, subset, write  # noqa: E402
import stubs  # noqa: E402
import test_serving  # noqa: E402


def profile(tp, drafter='none', tokens=0, cp=None, status='qualified', sequences=32):
    return dict(tensor_parallel=tp, context_parallel_size=cp, max_sequences=sequences, status=status,
                speculation=dict(drafter=drafter, tokens=tokens, mode='adaptive'))


DECLARED = {'tp4-spec-fast': profile(4, 'drafter-b', 7, status='experimental'),
            'tp4-ar': profile(4, sequences=64), 'tp4-spec': profile(4, 'drafter-a', 3),
            'tp4-ar-cp2': profile(4, cp=2), 'tp2-ar': profile(2, status='experimental')}


def commit_source(source, message):
    files = [str(path.relative_to(source)) for path in source.rglob('*')
             if path.is_file() and '.git' not in path.relative_to(source).parts]
    subprocess.run(['git', '-C', str(source), 'add', '--', *files], env=smoke.git_environment(),
                   check=True, capture_output=True)
    subprocess.run(['git', '-C', str(source), '-c', 'user.name=fixture', '-c', 'user.email=fixture@example.invalid',
                    'commit', '--quiet', '-m', message], env=smoke.git_environment(), check=True, capture_output=True)
    return smoke.git('rev-parse', 'HEAD', repo=source)


def source_history(source):
    source.mkdir(exist_ok=True)
    subprocess.run(['git', 'init', '--quiet', str(source)], env=smoke.git_environment(), check=True, capture_output=True)
    (source / 'history.txt').write_text('base\n')
    base = commit_source(source, 'base')
    (source / 'history.txt').write_text('tip\n')
    return base, commit_source(source, 'tip')


class LegTests(unittest.TestCase):
    def test_default_legs_are_tp4_ar_tp4_speculative_and_tp2_ar_qualified_first(self):
        self.assertEqual(smoke.default_legs(DECLARED), ['tp4-ar', 'tp4-spec', 'tp2-ar'])
        self.assertEqual(smoke.default_legs({'tp4-ar': DECLARED['tp4-ar']}), ['tp4-ar'])

    def test_a_leg_serves_its_profile_with_fixed_speculation(self):
        spec = smoke.leg('tp4-spec', DECLARED['tp4-spec'])
        self.assertEqual(spec['variants'], dict(spec=dict(mode='fixed', tokens=3, drafter='drafter-a')))
        self.assertEqual((spec['serve_profile'], spec['serve_args']),
                         ('tp4-spec', ['--tensor-parallel-size', '4', '--max-sequences', '32']))
        self.assertEqual(spec['expect'], dict(profile=dict(tensor_parallel_size=4, max_sequences=32)))
        cp = smoke.leg('tp4-ar-cp2', DECLARED['tp4-ar-cp2'])
        self.assertEqual(cp['serve_args'][-2:], ['--context-parallel-size', '2'])
        self.assertEqual(cp['variants'], dict(ar=dict(mode='off', tokens=0)))
        self.assertEqual(cp['expect'], dict(profile=dict(tensor_parallel_size=4, max_sequences=32,
                                                         context_parallel_size=2)))

    def test_a_leg_keeps_every_numerical_option_of_its_profile(self):
        full = dict(profile(4, 'drafter-a', 3), kv_cache_dtype='bf16', prefix_cache=False, max_batch_tokens=1024,
                    prefill_chunk_tokens=256, queue_depth=8, kv_memory_fraction=0.85, prefill_budget='tbt',
                    decode_tbt_target_ms=150, model_options={'checkpoint.quantization_layout': 'fp4', 'topk': 64})
        full['speculation']['draft_cost'] = 'per-block'
        value = smoke.leg('full', full)
        self.assertEqual(value['serve_args'], [
            '--tensor-parallel-size', '4', '--max-sequences', '32', '--max-batch-tokens', '1024',
            '--prefill-chunk-tokens', '256', '--kv-memory-fraction', '0.85', '--kv-cache-dtype', 'bf16',
            '--prefix-cache', 'off', '--queue-depth', '8', '--prefill-budget', 'tbt', '--decode-tbt-target-ms', '150',
            '--draft-cost', 'per-block', '--model-option', 'checkpoint.quantization_layout=fp4',
            '--model-option', 'topk=64'])
        self.assertEqual(value['expect'], dict(
            profile=dict(tensor_parallel_size=4, max_sequences=32, prefix_cache=False, kv_cache_dtype='bf16'),
            model_options={'checkpoint.quantization_layout': 'fp4', 'topk': 64}))
        # Two profiles that differ only in a model option are different legs.
        other = dict(full, model_options={'checkpoint.quantization_layout': 'fp8', 'topk': 64})
        self.assertNotEqual(smoke.leg('full', other)['serve_args'], value['serve_args'])
        with self.assertRaisesRegex(smoke.Refusal, 'unknown_option'):
            smoke.leg('new', dict(full, unknown_option=1))

    def test_a_declared_context_parallel_size_is_expected_even_when_one(self):
        health = dict(profile=dict(tensor_parallel_size=2, max_sequences=32))
        explicit = smoke.leg('tp2-cp1', profile(2, cp=1))['expect']
        with self.assertRaisesRegex(smoke.Refusal, 'context_parallel_size: missing'):
            subset(explicit, health)
        with self.assertRaisesRegex(smoke.Refusal, 'got 2, expected 1'):
            subset(explicit, dict(profile=dict(health['profile'], context_parallel_size=2)))
        subset(explicit, dict(profile=dict(health['profile'], context_parallel_size=1)))
        # An unset size may be absent, as /health of some unsplit profiles omits it.
        subset(smoke.leg('tp2-ar', DECLARED['tp2-ar'])['expect'], health)

    def test_the_run_file_is_a_smoke_ab_pair_of_the_determinism_set_and_decode(self):
        with tempfile.TemporaryDirectory() as name:
            src = Path(name) / 'src'
            (src / 'models/stub').mkdir(parents=True)
            (src / 'models/stub/qual.json').write_text('{}')
            value = smoke.run_file('stub', smoke.leg('tp4-spec', DECLARED['tp4-spec']),
                                   dict(base=Path(name) / 'a', new=Path(name) / 'b'), '/ckpt', 'identity', src,
                                   commits=dict(base='a' * 40, new='b' * 40))
            added = smoke.run_file('stub', smoke.leg('tp4-spec', DECLARED['tp4-spec']),
                                   dict(base=Path(name) / 'a', new=Path(name) / 'b'), '/ckpt', 'identity', src,
                                   ['warm', 'decode-c8'])
        self.assertEqual((value['smoke'], value['order'], value['phases']),
                         (True, ['base', 'new'], ['determinism', 'warm', 'decode-c1']))
        # --phases adds to the smoke phases.
        self.assertEqual(added['phases'], ['determinism', 'warm', 'decode-c1', 'decode-c8'])
        self.assertEqual((value['serve_profile'], value['serve_args'], 'expect' in value), ('tp4-spec', [], False))
        self.assertEqual(value['installs']['new'], dict(path=f'{name}/b/install', cache=f'{name}/b/cache',
                                                        commit='b' * 40))
        self.assertEqual((value['models'], value['qual_tools']), (str(src / 'models'), str(src / 'tools/qual')))
        self.assertEqual(value['env'], dict(CHOPINFER_QUAL_CONFIG=str(src / 'models/stub/qual.json'),
                                           CHOPINFER_QUAL_MODEL_ROOT=str(src)))
        self.assertEqual(value['gpus'], [0, 1, 2, 3])

    def test_jobs_share_a_device_count_and_fit_the_queue_limit(self):
        legs = [smoke.leg(name, DECLARED[name]) for name in ['tp4-ar', 'tp4-spec', 'tp4-ar-cp2', 'tp2-ar']]
        self.assertEqual(smoke.pack(legs, 120), [dict(tp=4, legs=['tp4-ar', 'tp4-spec', 'tp4-ar-cp2']),
                                                  dict(tp=2, legs=['tp2-ar'])])
        self.assertEqual(smoke.pack(legs, 180), [dict(tp=4, legs=['tp4-ar', 'tp4-spec']), dict(tp=4, legs=['tp4-ar-cp2']),
                                                  dict(tp=2, legs=['tp2-ar'])])
        self.assertEqual(smoke.pack(legs, 400), [dict(tp=4, legs=['tp4-ar']), dict(tp=4, legs=['tp4-spec']),
                                                  dict(tp=4, legs=['tp4-ar-cp2']), dict(tp=2, legs=['tp2-ar'])])
        with self.assertRaisesRegex(smoke.Refusal, 'more than a job allows'):
            smoke.pack(legs, 900)

    def test_build_targets_come_from_the_model_inventory(self):
        inventory = dict(compiler=dict(package='model-stub'), bundle=dict(package='bundle-stub', cdylib=['bundle_stub']))
        self.assertEqual(smoke.cargo_args(inventory, 'stub')[:7],
                         ['build', '-p', 'runtime-host', '-p', 'model-stub', '-p', 'bundle-stub'])
        self.assertIn('chopinfer-model-stub', smoke.cargo_args(inventory, 'stub'))
        self.assertEqual(smoke.install_files(inventory, 'stub'),
                         {'bin/chopinfer': 'chopinfer', 'bin/chopinfer-model-stub': 'chopinfer-model-stub',
                          'lib/libbundle_stub.so': 'libbundle_stub.so'})
        self.assertEqual(smoke.roots(inventory, 'stub'), ['runtime-host', 'model-stub', 'bundle-stub'])


class PathTests(unittest.TestCase):
    def metadata(self, root):
        root = Path(root)
        def package(name, path):
            directory = root / path
            (directory / 'src').mkdir(parents=True, exist_ok=True)
            (directory / 'src/lib.rs').touch()
            targets = [dict(kind=['lib'], src_path=str(directory / 'src/lib.rs')),
                       dict(kind=['test'], src_path=str(directory / 'tests/t.rs'))]
            return dict(id=f'{name} 0.1', name=name, manifest_path=str(directory / 'Cargo.toml'), targets=targets)
        packages = [package('runtime-host', 'crates/runtime-host'), package('model-stub', 'models/stub/compiler'),
                    package('bundle-stub', 'models/stub/bundle'), package('kernels', 'crates/kernels'),
                    package('build-helper', 'crates/build-helper'), package('test-only', 'crates/test-only'),
                    package('unrelated', 'crates/unrelated'), dict(id='serde 1.0', name='serde', targets=[],
                                                                   manifest_path='/registry/serde/Cargo.toml')]
        def dep(name, kind=None):
            return dict(pkg=f'{name} 0.1' if name != 'serde' else 'serde 1.0', dep_kinds=[dict(kind=kind)])
        nodes = [dict(id='runtime-host 0.1', deps=[dep('kernels'), dep('serde'), dep('test-only', 'dev')]),
                 dict(id='model-stub 0.1', deps=[]), dict(id='bundle-stub 0.1', deps=[dep('build-helper', 'build')]),
                 dict(id='kernels 0.1', deps=[]), dict(id='build-helper 0.1', deps=[]),
                 dict(id='test-only 0.1', deps=[]), dict(id='unrelated 0.1', deps=[]), dict(id='serde 1.0', deps=[])]
        return dict(workspace_root=str(root), packages=packages, resolve=dict(nodes=nodes),
                    workspace_members=[p['id'] for p in packages if p['name'] != 'serde'])

    def test_paths_are_the_model_build_closure_and_its_configuration(self):
        with tempfile.TemporaryDirectory() as name:
            specs = smoke.paths(self.metadata(name), ['runtime-host', 'model-stub', 'bundle-stub'], 'stub')
        # The build configuration is listed whether or not it exists yet.
        self.assertEqual(specs, ['.cargo/', 'Cargo.lock', 'Cargo.toml', 'crates/build-helper/', 'crates/kernels/',
                                 'crates/runtime-host/', 'models/stub/', 'rust-toolchain', 'rust-toolchain.toml',
                                 'tools/nvcc_cache/'])

    def test_files_a_crate_embeds_from_outside_its_directory_are_listed(self):
        with tempfile.TemporaryDirectory() as name:
            metadata = self.metadata(name)
            source = Path(name) / 'crates/kernels/src'
            (source / 'nested').mkdir()
            (source / 'nested/policy.rs').write_text(
                'const A: &str = include_str!("../../../../tools/refrun/policy.json");\n'
                'const B: &[u8] = include_bytes!(\n    "../../../shared/table.bin");\n'
                'const C: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/../../fixtures/plan.json"));\n'
                'include!(concat!(env!("OUT_DIR"), "/generated.rs"));\n'
                'const D: &str = include_str!("inside.txt");\n')
            for file in ['tools/refrun/policy.json', 'crates/shared/table.bin', 'fixtures/plan.json', 'docs/x.md']:
                (Path(name) / file).parent.mkdir(parents=True, exist_ok=True); (Path(name) / file).touch()
            (Path(name) / 'crates/kernels/tests').mkdir()
            (Path(name) / 'crates/kernels/tests/t.rs').write_text('const T: &str = include_str!("../../../docs/x.md");')
            specs = smoke.paths(metadata, ['runtime-host'], 'stub')
        self.assertTrue({'tools/refrun/policy.json', 'crates/shared/table.bin', 'fixtures/plan.json'} <= set(specs), specs)
        self.assertFalse([x for x in specs if 'inside' in x or 'docs' in x or 'generated' in x], specs)

    def test_files_the_tip_build_read_are_listed(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            metadata = self.metadata(root / 'src')
            release = root / 'target/release'
            (release / 'deps').mkdir(parents=True)
            (release / 'deps/kernels-0123.d').write_text(
                f'{release}/deps/libkernels.rlib: {root}/src/crates/kernels/src/lib.rs {root}/src/data/a\\ b.json '
                f'/usr/include/cuda.h {release}/build/out.rs {root}/src/crates/removed/lib.rs\n\n'
                f'{root}/src/data/a\\ b.json:\n')
            (release / 'build/kernels-89abcdef').mkdir(parents=True)
            (release / 'build/kernels-89abcdef/output').write_text(
                'cargo:rerun-if-changed=../../headers/common.cuh\ncargo::rerun-if-changed=src/lib.rs\n'
                'cargo:rerun-if-env-changed=CUDA_HOME\n')
            for file in ['data/a b.json', 'headers/common.cuh']:
                (root / 'src' / file).parent.mkdir(parents=True, exist_ok=True); (root / 'src' / file).touch()
            inputs = smoke.built_inputs(root / 'target', metadata)
            specs = smoke.paths(metadata, ['runtime-host'], 'stub', inputs)
        self.assertTrue({'data/a b.json', 'headers/common.cuh'} <= set(specs), specs)
        self.assertFalse([x for x in specs if 'cuda.h' in x or 'out.rs' in x or 'CUDA_HOME' in x or 'removed' in x],
                         specs)

    def test_watched_directories_and_excluded_local_packages_are_listed(self):
        with tempfile.TemporaryDirectory() as name:
            metadata = self.metadata(Path(name) / 'src')
            # vendor/helper: a path dependency the workspace excludes.
            helper = Path(name) / 'src/vendor/helper'
            (helper / 'src').mkdir(parents=True)
            metadata['packages'].append(dict(id='helper 0.1', name='helper', targets=[],
                                             manifest_path=str(helper / 'Cargo.toml')))
            metadata['resolve']['nodes'].append(dict(id='helper 0.1', deps=[]))
            metadata['resolve']['nodes'][0]['deps'].append(dict(pkg='helper 0.1', dep_kinds=[dict(kind=None)]))
            (Path(name) / 'src/assets/policies').mkdir(parents=True)
            inputs = {Path(name) / 'src/assets/policies', Path(name) / 'src/gone'}
            specs = smoke.paths(metadata, ['runtime-host'], 'stub', inputs)
        self.assertTrue({'vendor/helper/', 'assets/policies/'} <= set(specs), specs)
        self.assertNotIn('gone', ' '.join(specs))

    @unittest.skipUnless(shutil.which('cargo'), 'cargo is not on PATH (source local.env)')
    def test_a_real_build_input_change_marks_the_smoke_stale(self):
        """A real Cargo workspace whose binary changes with a file in a watched
        directory and with an excluded path dependency's manifest: the
        staleness check's `git diff --name-only BASE TIP -- PATHS` sees both."""
        with tempfile.TemporaryDirectory() as name:
            source, target = Path(name) / 'src', Path(name) / 'target'
            host, helper = source / 'crates/runtime-host', source / 'vendor/helper'
            files = {
                'Cargo.toml': '[workspace]\nmembers = ["crates/runtime-host"]\nexclude = ["vendor/helper"]\nresolver = "2"\n',
                'crates/runtime-host/Cargo.toml': '[package]\nname = "runtime-host"\nversion = "0.1.0"\nedition = "2021"\n\n'
                                                  '[dependencies]\nhelper = { path = "../../vendor/helper" }\n\n'
                                                  '[features]\nlinux-cuda-host = []\n',
                'crates/runtime-host/build.rs': 'fn main() {\n'
                    '    let inputs = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../assets/policies");\n'
                    '    println!("cargo:rerun-if-changed={}", inputs.display());\n'
                    '    let value = std::fs::read_to_string(inputs.join("served.txt")).unwrap();\n'
                    '    let out = std::path::Path::new(&std::env::var("OUT_DIR").unwrap()).join("policy.rs");\n'
                    '    std::fs::write(out, format!("const POLICY: &str = {:?};", value.trim())).unwrap();\n}\n',
                'crates/runtime-host/src/main.rs': 'include!(concat!(env!("OUT_DIR"), "/policy.rs"));\n'
                                                   'fn main() { print!("{} {}", POLICY, helper::VALUE); }\n',
                'vendor/helper/Cargo.toml': '[package]\nname = "helper"\nversion = "0.1.0"\nedition = "2021"\n\n'
                                            '[features]\ndefault = []\nchanged = []\n',
                'vendor/helper/src/lib.rs': '#[cfg(feature = "changed")]\npub const VALUE: u32 = 2;\n'
                                            '#[cfg(not(feature = "changed"))]\npub const VALUE: u32 = 1;\n',
                'assets/policies/served.txt': 'base\n'}
            for relative, text in files.items():
                (source / relative).parent.mkdir(parents=True, exist_ok=True)
                (source / relative).write_text(text)
            env = {k: v for k, v in os.environ.items() if not k.startswith('GIT_')}
            env.update(CARGO_TARGET_DIR=str(target), CARGO_NET_OFFLINE='true', CARGO_INCREMENTAL='0')
            def run(*argv):
                return subprocess.run(argv, cwd=source, env=env, check=True, capture_output=True, text=True).stdout.strip()
            run('cargo', 'generate-lockfile', '--offline')
            run('git', 'init', '--quiet')
            run('git', 'add', '.')
            run('git', '-c', 'user.name=t', '-c', 'user.email=t@t.invalid', 'commit', '--quiet', '-m', 'base')
            base = run('git', 'rev-parse', 'HEAD')
            build = ['cargo', 'build', '--release', '--locked', '--offline', '--features', 'runtime-host/linux-cuda-host']
            run(*build)
            self.assertEqual(run(str(target / 'release/runtime-host')), 'base 1')
            metadata = json.loads(run('cargo', 'metadata', '--format-version', '1', '--locked', '--offline'))
            specs = smoke.paths(metadata, ['runtime-host'], 'stub', smoke.built_inputs(target, metadata))
            for relative, old, new in [('assets/policies/served.txt', 'base', 'tip'),
                                       ('vendor/helper/Cargo.toml', 'default = []', 'default = ["changed"]')]:
                with self.subTest(changed=relative):
                    (source / relative).write_text((source / relative).read_text().replace(old, new))
                    run('git', 'add', relative)
                    run('git', '-c', 'user.name=t', '-c', 'user.email=t@t.invalid', 'commit', '--quiet', '-m', relative)
                    tip = run('git', 'rev-parse', 'HEAD')
                    before = run(str(target / 'release/runtime-host'))
                    run(*build)
                    self.assertNotEqual(run(str(target / 'release/runtime-host')), before)
                    self.assertEqual(run('git', 'diff', '--name-only', base, tip, '--', *specs), relative)
                    base = tip

    @unittest.skipUnless(shutil.which('cargo'), 'cargo is not on PATH (source local.env)')
    def test_build_inputs_of_two_packages_of_one_name_are_both_listed(self):
        """Two local versions of `helper`, each build script watching its own
        ../policy.txt: a second build is clean (the builder's no-op check
        passes), and a change to either policy changes the binary and is
        selected by `git diff --name-only BASE TIP -- PATHS`."""
        with tempfile.TemporaryDirectory() as name:
            source, target = Path(name) / 'src', Path(name) / 'target'
            files = {
                'Cargo.toml': '[workspace]\nmembers = ["crates/runtime-host"]\n'
                              'exclude = ["vendor/old/helper", "vendor/new/helper"]\nresolver = "2"\n',
                'crates/runtime-host/Cargo.toml': '[package]\nname = "runtime-host"\nversion = "0.1.0"\nedition = "2021"\n\n'
                    '[dependencies]\nold_helper = { package = "helper", path = "../../vendor/old/helper" }\n'
                    'new_helper = { package = "helper", path = "../../vendor/new/helper" }\n\n'
                    '[features]\nlinux-cuda-host = []\n',
                'crates/runtime-host/src/main.rs': 'fn main() { print!("{} {}", old_helper::POLICY, new_helper::POLICY); }\n'}
            for kind, version in [('old', '0.1.0'), ('new', '0.2.0')]:
                files[f'vendor/{kind}/helper/Cargo.toml'] = (f'[package]\nname = "helper"\nversion = "{version}"\n'
                                                             'edition = "2021"\n')
                files[f'vendor/{kind}/helper/src/lib.rs'] = 'pub const POLICY: &str = env!("SERVED_POLICY");\n'
                files[f'vendor/{kind}/helper/build.rs'] = (
                    'fn main() {\n    println!("cargo:rerun-if-changed=../policy.txt");\n'
                    '    let policy = std::fs::read_to_string("../policy.txt").unwrap();\n'
                    '    println!("cargo:rustc-env=SERVED_POLICY={}", policy.trim());\n}\n')
                files[f'vendor/{kind}/policy.txt'] = kind + '\n'
            for relative, text in files.items():
                (source / relative).parent.mkdir(parents=True, exist_ok=True)
                (source / relative).write_text(text)
            env = {k: v for k, v in os.environ.items() if not k.startswith('GIT_')}
            env.update(CARGO_TARGET_DIR=str(target), CARGO_NET_OFFLINE='true', CARGO_INCREMENTAL='0')
            def run(*argv):
                result = subprocess.run(argv, cwd=source, env=env, check=True, capture_output=True, text=True)
                return result.stdout.strip(), result.stderr
            run('cargo', 'generate-lockfile', '--offline')
            run('git', 'init', '--quiet')
            run('git', 'add', '.')
            run('git', '-c', 'user.name=t', '-c', 'user.email=t@t.invalid', 'commit', '--quiet', '-m', 'base')
            base = run('git', 'rev-parse', 'HEAD')[0]
            build = ['cargo', 'build', '--release', '--locked', '--offline', '--features', 'runtime-host/linux-cuda-host']
            run(*build)
            log = '\n'.join(run(*build, '-v'))
            self.assertFalse([x for x in log.splitlines() if x.lstrip().startswith(('Compiling ', 'Running '))], log)
            binary = str(target / 'release/runtime-host')
            self.assertEqual(run(binary)[0], 'old new')
            metadata = json.loads(run('cargo', 'metadata', '--format-version', '1', '--locked', '--offline')[0])
            self.assertEqual(len([p for p in metadata['packages'] if p['name'] == 'helper']), 2)
            specs = smoke.paths(metadata, ['runtime-host'], 'stub', smoke.built_inputs(target, metadata))
            for kind, output in [('old', 'changed new'), ('new', 'changed changed')]:
                with self.subTest(changed=kind):
                    relative = f'vendor/{kind}/policy.txt'
                    (source / relative).write_text('changed\n')
                    run('git', 'add', relative)
                    run('git', '-c', 'user.name=t', '-c', 'user.email=t@t.invalid', 'commit', '--quiet', '-m', kind)
                    tip = run('git', 'rev-parse', 'HEAD')[0]
                    run(*build)
                    self.assertEqual(run(binary)[0], output)
                    self.assertEqual(run('git', 'diff', '--name-only', base, tip, '--', *specs)[0], relative)
                    base = tip

    def test_an_unknown_root_is_refused(self):
        with tempfile.TemporaryDirectory() as name:
            with self.assertRaisesRegex(smoke.Refusal, 'not in the workspace'):
                smoke.paths(self.metadata(name), ['runtime-host', 'gone'], 'stub')


class QualificationSourceTests(unittest.TestCase):
    def setUp(self):
        os.umask(0o022)
        self.tmp = tempfile.TemporaryDirectory(prefix='smoke-source-')
        self.addCleanup(self.tmp.cleanup)
        self.repo = Path(self.tmp.name) / 'repo'
        self.base, self.tip = source_history(self.repo)
        self.out = Path(self.tmp.name) / 'out'

    def prepare(self):
        with patch.object(smoke, 'REPO', self.repo):
            # git's default was bound when the module was imported.
            original = smoke.git
            def git(*args, repo=None):
                return original(*args, repo=repo or self.repo)
            with patch.object(smoke, 'git', side_effect=git):
                return smoke.qualification_source(self.out, self.tip)

    def test_a_pinned_source_has_ancestry_and_survives_parent_changes(self):
        record = self.prepare()
        source = Path(record['path'])
        self.assertEqual(smoke.git('merge-base', self.base, 'HEAD', repo=source), self.base)
        self.assertEqual((source / 'history.txt').read_text(), 'tip\n')
        (self.repo / 'history.txt').write_text('next tip\n')
        commit_source(self.repo, 'later parent commit')
        (self.repo / 'history.txt').write_text('uncommitted parent change\n')
        with patch.dict(os.environ, GIT_DIR=str(self.repo / '.git'), GIT_WORK_TREE=str(self.repo)):
            self.assertEqual(self.prepare(), record)
        self.assertEqual((source / 'history.txt').read_text(), 'tip\n')
        self.assertIn('locked', smoke.git('worktree', 'list', '--porcelain', repo=self.repo))

    def test_reuse_refuses_a_changed_or_attached_checkout_without_resetting_it(self):
        record = self.prepare()
        source = Path(record['path'])
        (source / 'history.txt').write_text('do not discard\n')
        with self.assertRaisesRegex(smoke.Refusal, 'not clean'):
            self.prepare()
        self.assertEqual((source / 'history.txt').read_text(), 'do not discard\n')
        (source / 'history.txt').write_text('tip\n')
        with self.assertRaisesRegex(smoke.Refusal, 'detached'):
            smoke.source_record(self.repo, self.tip)

    def test_shallow_history_and_wrong_tree_records_are_refused(self):
        record = self.prepare()
        (self.repo / '.git/shallow').write_text(self.tip + '\n')
        with self.assertRaisesRegex(smoke.Refusal, 'complete history'):
            self.prepare()
        (self.repo / '.git/shallow').unlink()
        pair = dict(tip=self.tip, qualification_source=dict(record, tree='0' * 40))
        with self.assertRaisesRegex(smoke.Refusal, 'identity differs'):
            smoke.verify_qualification(pair, {})

    def test_ready_builds_still_use_the_requested_tip_source(self):
        fixture = test_serving.fixture(Path(self.tmp.name) / 'fixture', ['warm', 'decode-c1'])
        shutil.copytree(fixture['models'], self.repo / 'models')
        model = self.repo / 'models/stub/model.toml'
        model.write_text('checkpoint_env = "STUB"\n' + ReadyTests.MODEL_TOML)
        (model.parent / 'qual.json').write_text('{}')
        self.tip = commit_source(self.repo, 'model at tip')
        # Both arms are already built. Their archived src is left as it was;
        # even a stale copy there must never select qualification inputs.
        archive = self.out / 'arms' / self.tip / 'src'
        archive.mkdir(parents=True)
        (archive / 'sentinel').write_text('keep this build export\n')
        args = SimpleNamespace(out=self.out, model='stub', base=self.base, tip=self.tip, legs='tp1-ar',
            checkpoint=fixture['checkpoint'], jobs=1, phases=None, start_seconds=120, job_prefix='stub')
        original = smoke.git
        def git(*args, repo=None):
            return original(*args, repo=repo or self.repo)
        with patch.object(smoke, 'git', side_effect=git), \
             patch.object(smoke, 'build_arm', return_value=dict(checkpoint_identity='stub-checkpoint')) as build, \
             patch.object(smoke, 'pin_runner', return_value=self.out / 'runner'), \
             patch.object(smoke, 'preflight_qualification') as preflight, \
             patch.object(smoke, 'dry_run'), patch.object(smoke, 'cargo_metadata'), \
             patch.object(smoke, 'built_inputs', return_value=[]), patch.object(smoke, 'paths', return_value=['models/stub/']):
            self.assertEqual(smoke.build(args), 0)
        self.assertEqual(build.call_count, 2)
        source = self.out / 'qualification' / self.tip
        value = read(self.out / 'legs/tp1-ar.json')
        self.assertEqual(value['models'], str(source / 'models'))
        self.assertEqual(value['env']['CHOPINFER_QUAL_CONFIG'], str(source / 'models/stub/qual.json'))
        self.assertEqual(preflight.call_args.args[1], value)
        self.assertEqual((archive / 'sentinel').read_text(), 'keep this build export\n')
        self.assertEqual(smoke.git('rev-parse', 'HEAD', repo=source), self.tip)


class ReadyTests(unittest.TestCase):
    """A READY arm is kept only when it is the requested build."""

    MODEL_TOML = ('[bundle]\npackage = "bundle-stub"\ncdylib = ["bundle_stub"]\n\n'
                  '[compiler]\npackage = "model-stub"\nbin = ["chopinfer-model-stub"]\n')

    def setUp(self):
        os.umask(0o022)
        self.tmp = tempfile.TemporaryDirectory(prefix='smoke-ready-')
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        fixture = test_serving.fixture(root / 'fixture', ['warm', 'decode-c1'])
        self.commit = 'a' * 40
        self.out, self.checkpoint = root / 'out', Path(fixture['checkpoint'])
        self.arm = self.out / 'arms' / self.commit
        shutil.copytree(fixture['installs']['base']['path'], self.arm / 'install')
        self.models = Path(fixture['models']) / 'stub'
        self.source()
        self.record = dict(commit=self.commit, model='stub', checkpoint=str(self.checkpoint.resolve()),
                           cargo=smoke.cargo_args(smoke.inventory(self.arm / 'src/models', 'stub'), 'stub'), tps=[1, 2],
                           install_sha256=digest(self.arm / 'install/INSTALL.json'), checkpoint_identity='stub-checkpoint')

    def source(self):
        """The arm's src/ as an export leaves it (a rebuild removes it)."""
        shutil.rmtree(self.arm / 'src', ignore_errors=True)
        shutil.copytree(self.models, self.arm / 'src/models/stub')
        (self.arm / 'src/models/stub/model.toml').write_text(self.MODEL_TOML)

    def build(self, record, commit=None, tps=(1,), checkpoint=None):
        self.source()
        write(self.arm / 'READY', record)
        class Rebuild(Exception):
            pass
        try:
            with patch.object(smoke.subprocess, 'Popen', side_effect=Rebuild):
                return smoke.build_arm(self.out, commit or self.commit, 'stub', list(tps), checkpoint or self.checkpoint, 1)
        except Rebuild:
            return 'rebuild'

    def test_the_requested_build_is_kept(self):
        self.assertEqual(self.build(self.record), self.record)

    def test_any_other_build_is_rebuilt(self):
        receipt = read(self.arm / 'install/INSTALL.json')
        other = Path(self.tmp.name) / 'other-checkpoint'; other.mkdir()
        for change in [dict(commit='d' * 40), dict(model='other'), dict(checkpoint=str(other)),
                       dict(cargo=['build']), dict(tps=[2]), dict(install_sha256='0' * 64)]:
            with self.subTest(change=change):
                self.assertEqual(self.build(dict(self.record, **change)), 'rebuild')
        # A READY record and receipt naming another revision under this arm.
        with self.subTest(receipt='other commit'):
            path = self.arm / 'install/INSTALL.json'; path.chmod(0o644)
            write(path, dict(receipt, commit='c' * 40))
            self.assertEqual(self.build(dict(self.record, commit='c' * 40, install_sha256=digest(path))), 'rebuild')
            self.assertEqual(self.build(dict(self.record, install_sha256=digest(path))), 'rebuild')
            write(path, receipt)
        with self.subTest(change='install file'):
            with (self.arm / 'install/bin/chopinfer').open('ab') as stream:
                stream.write(b'\n')
            self.assertEqual(self.build(self.record), 'rebuild')


class RunTests(unittest.TestCase):
    """`smoke.py run` on stub installs: two legs, each one AB pair."""

    def setUp(self):
        os.umask(0o022)
        self.tmp = tempfile.TemporaryDirectory(prefix='smoke-test-')
        self.addCleanup(self.tmp.cleanup)
        self.out = Path(self.tmp.name) / 'out'
        c = test_serving.fixture(Path(self.tmp.name) / 'fixture', ['determinism', 'warm', 'decode-c1'])
        source = Path(self.tmp.name) / 'source'
        shutil.copytree(c['models'], source / 'models')
        (source / 'models/stub/qual.json').write_text('{}')
        qual = source / 'tools/qual'
        qual.mkdir(parents=True)
        (qual / 'determinism.py').write_text(stubs.DETERMINISM)
        self.base, self.tip = source_history(source)
        self.source = self.out / 'qualification' / self.tip
        self.source.parent.mkdir(parents=True)
        smoke.git('worktree', 'add', '--detach', str(self.source), self.tip, repo=source)
        self.pair = dict(model='stub', base=self.base, tip=self.tip, legs=['ar', 'spec'],
                         qualification_source=dict(schema='chopinfer.smoke-qualification-source.v1',
                            path=str(self.source), commit=self.tip, tree=smoke.git('rev-parse', 'HEAD^{tree}', repo=self.source)))
        write(self.out / 'pair.json', self.pair)
        for arm, commit in [('base', self.base), ('new', self.tip)]:
            path = Path(c['installs'][arm]['path']) / 'INSTALL.json'
            write(path, dict(read(path), commit=commit, stub_declaration=arm == 'new'))
            c['installs'][arm]['commit'] = commit
        smoke.pin_runner(self.out)
        (self.out / 'legs').mkdir()
        self.c = c
        for name, variants in [('ar', dict(ar=dict(mode='off', tokens=0))),
                               ('spec', dict(spec=dict(mode='fixed', tokens=3)))]:
            leg = dict(c, smoke=True, order=['base', 'new'], models=str(self.source / 'models'),
                       qual_tools=str(self.source / 'tools/qual'), variants=variants)
            leg['env'] = dict(c['env'], CHOPINFER_QUAL_CONFIG=str(self.source / 'models/stub/qual.json'),
                              CHOPINFER_QUAL_MODEL_ROOT=str(self.source))
            (self.out / 'legs' / f'{name}.json').write_text(json.dumps(leg))

    def run_legs(self, *legs, **env):
        return subprocess.run([sys.executable, '-B', str(HERE / 'smoke.py'), 'run', str(self.out), *legs],
                              env={**os.environ, 'CUDA_VISIBLE_DEVICES': '0', **env}, capture_output=True,
                              text=True, timeout=1800)

    def test_every_leg_passes_as_one_ab_pair(self):
        result = self.run_legs('ar', 'spec')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(result.stdout.splitlines()[-2:], ['PASS ar', 'PASS spec'])
        written = sorted(p.name for p in (self.out / 'runs').glob('*.json'))
        self.assertEqual(len(written), 2)
        value = json.loads((self.out / 'runs' / written[0]).read_text())
        self.assertEqual(value['gpus'], ['0'])
        self.assertNotEqual(value['port'], self.c['port'])
        self.assertEqual(len(Path(self.c['env']['STUB_STARTS']).read_text().splitlines()), 4)

    def test_a_tip_answering_differently_fails_its_leg(self):
        leg = json.loads((self.out / 'legs/ar.json').read_text())
        leg['installs']['new']['env'] = dict(STUB_DETERMINISM_DIFFERENT='1')
        (self.out / 'legs/ar.json').write_text(json.dumps(leg))
        result = self.run_legs('ar')
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn('FAIL ar: Refusal: determinism records differ', result.stdout)

    def test_a_leg_needs_its_gpus(self):
        leg = json.loads((self.out / 'legs/ar.json').read_text())
        leg['gpus'] = [0, 1]
        (self.out / 'legs/ar.json').write_text(json.dumps(leg))
        result = self.run_legs('ar')
        self.assertEqual(result.returncode, 2)
        self.assertIn('needs 2 GPUs', result.stderr)

    def test_a_source_without_its_history_is_refused_before_start(self):
        (self.source / '.git').unlink()
        result = self.run_legs('ar')
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertFalse(Path(self.c['env']['STUB_STARTS']).exists())

    def test_a_dirty_or_wrong_source_is_refused_before_start(self):
        for change in ('tracked', 'untracked', 'ignored', 'revision'):
            with self.subTest(change=change):
                if change == 'revision':
                    smoke.git('checkout', '--detach', self.base, repo=self.source)
                else:
                    path = self.source / ('history.txt' if change == 'tracked' else 'unexpected')
                    if change == 'ignored':
                        smoke.git('config', 'core.excludesFile', str(Path(self.tmp.name) / 'ignore'), repo=self.source)
                        (Path(self.tmp.name) / 'ignore').write_text('unexpected\n')
                    path.write_text('changed\n')
                result = self.run_legs('ar')
                self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                self.assertFalse(Path(self.c['env']['STUB_STARTS']).exists())
                if change == 'tracked':
                    path.write_text('tip\n')
                elif change != 'revision':
                    path.unlink()

    def test_a_leg_cannot_read_qualification_from_another_source(self):
        for field in ('models', 'qual_tools', 'CHOPINFER_QUAL_CONFIG', 'CHOPINFER_QUAL_MODEL_ROOT'):
            with self.subTest(field=field):
                path = self.out / 'legs/ar.json'
                original = path.read_bytes()
                value = read(path)
                (value['env'] if field.startswith('CHOPINFER_') else value)[field] = str(Path(self.tmp.name))
                write(path, value)
                result = self.run_legs('ar')
                self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                self.assertFalse(Path(self.c['env']['STUB_STARTS']).exists())
                path.write_bytes(original)

    def test_a_source_changed_during_a_leg_cannot_pass(self):
        original = subprocess.run
        def change(command, **kwargs):
            if command[:2] == [sys.executable, '-B'] and command[2].endswith('/run.py'):
                (self.source / 'history.txt').write_text('changed during capture\n')
                return subprocess.CompletedProcess(command, 0)
            return original(command, **kwargs)
        with patch.object(smoke.subprocess, 'run', side_effect=change):
            with self.assertRaisesRegex(smoke.Refusal, 'qualification source'):
                smoke.run(SimpleNamespace(dir=self.out, legs=['ar']))

    def test_preparation_preserves_existing_builds_failures_and_old_setup(self):
        pair = dict(self.pair)
        pair.pop('qualification_source')
        write(self.out / 'pair.json', pair)
        archive = self.out / 'arms' / self.tip / 'src'
        archive.mkdir(parents=True)
        (archive / '.commit').write_text(self.tip + '\n')
        (archive.parent / 'READY').write_text('unchanged READY\n')
        (self.out / 'runs').mkdir()
        (self.out / 'runs/old-failure.log').write_text('saved failure\n')
        saved = {path: path.read_bytes() for path in [archive / '.commit', archive.parent / 'READY',
                 self.out / 'runs/old-failure.log', self.out / 'pair.json', self.out / 'legs/ar.json']}
        with patch.object(smoke, 'preflight_qualification') as preflight:
            self.assertEqual(smoke.prepare_qualification(self.out), 0)
        self.assertEqual(preflight.call_count, 2)
        self.assertEqual(read(self.out / 'pair.json')['qualification_source'], self.pair['qualification_source'])
        backup, = (self.out / 'qualification-updates').iterdir()
        self.assertEqual((backup / 'pair.before.json').read_bytes(), saved[self.out / 'pair.json'])
        self.assertEqual((backup / 'legs.before/ar.json').read_bytes(), saved[self.out / 'legs/ar.json'])
        for path in [archive / '.commit', archive.parent / 'READY', self.out / 'runs/old-failure.log']:
            self.assertEqual(path.read_bytes(), saved[path])

    def test_failed_cpu_preflight_does_not_replace_the_saved_setup(self):
        saved = {path: path.read_bytes() for path in [self.out / 'pair.json', self.out / 'legs/ar.json',
                 self.out / 'runner/tools/smoke/smoke.py']}
        with patch.object(smoke, 'preflight_qualification', side_effect=smoke.Refusal('invalid adoption')):
            with self.assertRaisesRegex(smoke.Refusal, 'invalid adoption'):
                smoke.prepare_qualification(self.out)
        for path, content in saved.items():
            self.assertEqual(path.read_bytes(), content)


if __name__ == '__main__':
    unittest.main()
