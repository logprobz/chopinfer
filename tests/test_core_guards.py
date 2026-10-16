"""Exercise guard failures with synthetic shared sources and model graphs."""
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.model_sdk import core_policy, neutral_sources
from tools.model_sdk.graph import Graph

spec = importlib.util.spec_from_file_location('leak_guard', ROOT / 'tests/leak_guard.py')
leak_guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(leak_guard)
spec = importlib.util.spec_from_file_location('sdk_test_support', ROOT / 'tests/sdk_test_support.py')
sdk_test_support = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sdk_test_support)


class CoreGuards(unittest.TestCase):
    def setUp(self):
        # Temporary repositories must not inherit the validation runner's Git context.
        env = {key: value for key, value in os.environ.items() if not key.startswith('GIT_')}
        env.update(GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_NOSYSTEM='1')
        context = patch.dict(os.environ, env, clear=True)
        context.start()
        self.addCleanup(context.stop)

    def test_shared_source_mutations_are_rejected(self):
        metadata = json.dumps({'packages': [], 'workspace_members': [], 'resolve': {'nodes': []}}).encode()
        check_output = subprocess.check_output
        def output(command, **kwargs):
            return metadata if command[1] == 'metadata' else check_output(command, **kwargs)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(['git', 'init', '-q', root], check=True)
            (root / 'tests').mkdir()
            (root / neutral_sources.ALLOWLIST).write_text('{}')
            old = Path.cwd()
            try:
                os.chdir(root)
                for name in ('crates/kernels-sm120/cuda/probe.cuh', 'crates/kernel-lab/cuda/probe.cu',
                             'crates/xtask/src/probe.rs', 'crates/kernel-core/tests/probe.rs',
                             'crates/encoder-spec/tests/probe.rs', 'tools/probe.py', 'ci/probe.sh',
                             'deploy/probe.c', 'crates/runtime-host/src/probe.h',
                             'crates/kernels-sm120/cuda/qwen38_probe.cuh'):
                    path = root / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text('generic_shape = 4096\n' if 'qwen38_probe' in name
                                    else 'CHOP_QWEN38_HIDDEN = 4096\n')
                    with self.subTest(name=name), patch('subprocess.check_output', side_effect=output):
                        with self.assertRaises(SystemExit):
                            core_policy.check_neutral()
                    path.unlink()
            finally:
                os.chdir(old)

    def test_declared_family_and_non_utf8_source_are_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(['git', 'init', '-q', root], check=True)
            (root / 'tests').mkdir()
            (root / neutral_sources.ALLOWLIST).write_text('{}')
            model = root / 'models/orchid/model.toml'
            model.parent.mkdir(parents=True)
            model.write_text('schema = "chopinfer-model-inventory-v1"')
            source = root / 'tools/probe.cu'
            source.parent.mkdir()
            source.write_bytes(b'\xff ORCHID_HIDDEN = 2048')
            with self.assertRaises(SystemExit):
                neutral_sources.check_sources(root)

    def test_short_model_names_match_only_whole_identifier_words(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / 'models/m3/model.toml'
            model.parent.mkdir(parents=True)
            model.write_text('schema = "chopinfer-model-inventory-v1"')
            pattern = neutral_sources.vocabulary(root)
            for text in ('fp8_e4m3', 'E4M3', 'dim3', 'm34', 'm3x'):
                with self.subTest(text=text):
                    self.assertEqual([m.group() for m in pattern.finditer(text)], [])
            for text in ('m3', 'M3_HIDDEN', 'load_m3', 'models/m3/x', 'ProbeM3Tests', 'M3Probe'):
                with self.subTest(text=text):
                    self.assertEqual([m.group().lower() for m in pattern.finditer(text)], ['m3'])

    def test_allowances_fail_when_removed_or_expanded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(['git', 'init', '-q', root], check=True)
            (root / 'tests').mkdir()
            (root / 'tools').mkdir()
            path = root / 'tools/probe.py'
            path.write_text('qwen38')
            found = neutral_sources.findings(root)
            (root / neutral_sources.ALLOWLIST).write_text(json.dumps({
                name: {'reason': 'Synthetic migration case', 'matches': hits} for name, hits in found.items()}))
            neutral_sources.check_sources(root)
            for text in ('', 'qwen38 qwen38', 'mimo'):
                path.write_text(text)
                with self.subTest(text=text), self.assertRaises(SystemExit):
                    neutral_sources.check_sources(root)
            path.unlink()
            with self.assertRaises(SystemExit):
                neutral_sources.check_sources(root)
            (root / neutral_sources.ALLOWLIST).write_text(json.dumps({
                'tools/probe.py': {'reason': 'Stale synthetic fixture', 'matches': None}}))
            with self.assertRaises(SystemExit):
                neutral_sources.check_sources(root)

    def test_third_model_contributors_reject_execution_dependencies(self):
        for role in ('compiler', 'contract'):
            model = SimpleNamespace(packages={
                'compiler': SimpleNamespace(name='third-compiler'),
                'contract': SimpleNamespace(name='third-contract'),
                'bundle': SimpleNamespace(name='third-native')})
            for dependency in ('runtime-host', 'runtime-server', 'runtime-cuda', 'bundle-other', 'third-native'):
                names = ('third-compiler', 'third-contract', dependency)
                graph = Graph({'packages': [{'id': name, 'name': name} for name in names],
                               'resolve': {'nodes': [
                                   {'id': name, 'deps': [{'pkg': dependency, 'dep_kinds': [{'kind': None}]}]
                                    if name == 'third-' + role else []} for name in names]}})
                with self.subTest(role=role, dependency=dependency), self.assertRaises(SystemExit):
                    core_policy.check_model_contributors(graph, [model])

    def test_declared_compilers_and_contracts_are_cuda_free_roots(self):
        from tools.model_sdk import inventory
        metadata = json.loads(subprocess.check_output([
            os.environ.get('CARGO', 'cargo'), 'metadata', '--locked', '--offline',
            '--format-version', '1', '--all-features'], cwd=ROOT))
        models = inventory.load(ROOT)
        roles = {role: inventory.Package('third', role, 'third-' + role, frozenset(), 'models/third')
                 for role in ('compiler', 'contract', 'bundle')}
        models.append(inventory.Model('third', None, roles))
        kernel_build = next(p['id'] for p in metadata['packages'] if p['name'] == 'kernel-build')
        for role, package in roles.items():
            metadata['packages'].append({'id': package.name, 'name': package.name, 'targets': []})
            metadata['workspace_members'].append(package.name)
            metadata['resolve']['nodes'].append({'id': package.name, 'deps': (
                [{'pkg': kernel_build, 'dep_kinds': [{'kind': 'build'}]}] if role == 'bundle' else [])})
        class SelectedRoots(Exception):
            pass
        with patch('subprocess.check_output', return_value=json.dumps(metadata).encode()), \
             patch.object(inventory, 'load', return_value=models), \
             patch.object(inventory, 'check_metadata'), \
             patch.object(core_policy, 'selected_graph', side_effect=SelectedRoots) as selected:
            with self.assertRaises(SelectedRoots):
                core_policy.check_bundle()
        roots = selected.call_args.kwargs['packages']
        for model in models:
            for role in ('compiler', 'contract'):
                if role in model.packages:
                    self.assertIn(model.packages[role].name, roots)

    def test_prefix_cache_policy_includes_new_profile_files(self):
        spec = importlib.util.spec_from_file_location('profiles', ROOT / 'tests/test_serve_profiles.py')
        profiles = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(profiles)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for model in ('mimo26', 'qwen38', 'third'):
                path = root / 'models' / model / 'profiles.json'
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps({'profiles': {'default': {'prefix_cache': model != 'third'}}}))
            with patch.object(profiles, 'ROOT', root), self.assertRaises(AssertionError):
                profiles.ProfilesTest('test_all_model_profiles_enable_prefix_caching').test_all_model_profiles_enable_prefix_caching()

    def test_internal_labels_are_rejected(self):
        # Split so this file does not carry the shapes it tests.
        samples = ['W' + '1.7', 'W' + '7.2-ENC-CATALOG', 'C' + '16-SMALL',
                   'plan ' + 'B12.3', 'GPU ' + 'lock', 'w17s-' + 'b-parity-mimo',
                   'operator' + ' decision 3', 'operator' + ' decisions',
                   'PERF-' + 'B12', 'PERF-' + 'B17-1 finding 1', 'REVIEW-' + 'A4', 'FIX-' + 'SC1',
                   'HICONC-' + 'TP4', 'GATES' + '4', '(GATES' + '2 dcp4-prefix)', 'CONC-' + 'QWEN',
                   'CONC-' + 'QWEN-STATE', 'MIMO-' + 'REQUAL batch', 'TOPO-' + 'MIMO', 'KVO-' + 'VALIDATE',
                   'JPROOF' + '2-1', 'the MM-' + '7 ABI', 'The operator' + ' adopted the results',
                   'an operator' + '-accepted row', "the operator's" + ' approval', '(operator' + ', 2026-10-11)',
                   'plan ' + 'step 39', 'adopted by the ' + 'operator on 2026-10-15',
                   'the rule was waived by the ' + 'operator', 'Review of ' + 'deadbeef1, finding 1', 'Review of ' + 'specref-2',
                   'R1' + ' finding 3', 'Consolidated GPU review' + ', finding 1']
        for sample in samples:
            with self.subTest(sample=sample):
                self.assertTrue(any(pattern.search(sample) for _, pattern, allowed in leak_guard.RULES
                                    if allowed is None))
        for sample in ('weight-plan-v1', 'fixture-graph-plan-v1', 'version 1.7',
                       'weight plan v1', 'execution plan v2', 'weight plan V1',
                       'HMAC-SHA256', 'b"SUM-BF16-V1"', 'SHA-256', 'UTF-8', 'GLM-5.3 Flash', 'GPT-2',
                       'MPEG-4', 'ISO-8601', 'YYYY-MM-DD', '--checkpoint NEW-DIR', 'SAFETY-REVIEWED',
                       "the operator's program", 'unless the operator says so',
                       'operator:2026-10-12:no-accepted-arm', 'the execution plan step',
                       'review of the plan', 'the review of 3 files', 'Regression: a request',
                       'a process started by the operator'):
            with self.subTest(sample=sample):
                self.assertFalse(any(pattern.search(sample) for _, pattern, allowed in leak_guard.RULES
                                     if allowed is None))

    def test_local_root_paths_are_rejected(self):
        for sample in ('/srv' + '/package', 'mount at /mnt' + '/models', 'cd /root' + '/chopinfer'):
            with self.subTest(sample=sample):
                self.assertTrue(any(pattern.search(sample) for label, pattern, allowed in leak_guard.RULES
                                    if allowed is None and 'path' in label))
        self.assertFalse(any(pattern.search('$CHOPINFER_DATA/srv/x') for label, pattern, _ in leak_guard.RULES
                             if 'path' in label))

    def test_pending_files_are_tracked_files(self):
        tracked = set(subprocess.run(['git', 'ls-files'], cwd=ROOT, check=True, capture_output=True,
                                     text=True).stdout.split())
        labels = {label for label, _, _ in leak_guard.RULES}
        for label, names in leak_guard.PENDING.items():
            self.assertIn(label, labels)
            for name in names:
                with self.subTest(name=name):
                    self.assertIn(name, tracked)

    def test_pending_files_still_need_their_rewrite(self):
        rules = {label: pattern for label, pattern, _ in leak_guard.RULES}
        for label, names in leak_guard.PENDING.items():
            for name in names:
                with self.subTest(label=label, name=name):
                    text = (ROOT / name).read_text(errors='replace')
                    self.assertTrue(any(rules[label].search(line) for line in text.splitlines()),
                                    'clean: remove it from PENDING')

    def test_only_the_quoted_licence_contact_address_is_allowed(self):
        email = next(pattern for label, pattern, _ in leak_guard.RULES if label == 'email address')
        quoted = 'model-business' + '@notice.qwencloud.com'
        self.assertIsNone(email.search(f'please contact {quoted}.'))
        for sample in ('jane.doe' + '@mail.org', 'x' + quoted, quoted + '.mail.org', quoted + 'munity'):
            with self.subTest(sample=sample):
                self.assertIsNotNone(email.search(sample))

    def test_machine_patterns_come_from_the_untracked_deny_list(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(['git', 'init', '-q', str(root)], check=True)
            (root / 'notes.md').write_text('Built on zz-bench' + '-host-7 last week.\n')
            subprocess.run(['git', '-C', str(root), 'add', 'notes.md'], check=True)
            denylist = root / 'denylist'
            with patch.object(leak_guard, 'ROOT', root), \
                    patch.object(leak_guard, 'DENYLIST_FILES', [denylist, root / 'leak-denylist.local']):
                # With neither file present the public-tree checks still
                # pass, and the output says the machine checks did not run.
                stderr = io.StringIO()
                with contextlib.redirect_stderr(stderr):
                    self.assertEqual(leak_guard.main(), 0)
                self.assertIn('no local deny-list', stderr.getvalue())
                self.assertIn(str(denylist), stderr.getvalue())
                self.assertIn('machine-specific patterns were not checked', stderr.getvalue())
                denylist.write_text('# one regular expression per line\n\nzz-bench-host-[0-9]+\n')
                rules = leak_guard.local_rules()
                self.assertEqual([label for label, _, _ in rules], ['local deny-list pattern (denylist:3)'])
                with patch('sys.stderr'):
                    self.assertEqual(leak_guard.main(), 1)
                denylist.unlink()
                (root / 'leak-denylist.local').write_text('ZZ-BENCH-HOST\n')
                with patch('sys.stderr'):
                    self.assertEqual(leak_guard.main(), 1)

    def test_the_shared_runtime_uses_kernels_sm120_only_in_tests(self):
        def check(source, test_source=None, manifest=None):
            with tempfile.TemporaryDirectory() as directory:
                src = Path(directory) / 'crates/bundle-runtime/src'
                src.mkdir(parents=True)
                (src / 'lib.rs').write_text(source)
                if manifest is not None:
                    (src.parent / 'Cargo.toml').write_text(manifest)
                if test_source is not None:
                    (src / 'tests.rs').write_text(test_source)
                core_policy.check_projection_sources(directory)

        # Production source names no item of the kernel set, however reached.
        for source in [
            'use kernels_sm120::precision::Compute;',
            'use kernels_sm120::{gemm::plan as supplied};',
            'fn plan() { kernels_sm120::gemm::plan(); }',
            'use kernels_sm120 as engine;',
            'extern crate kernels_sm120 as engine;',
            'pub use kernels_sm120::precision::*;',
            'use r#kernels_sm120::precision::KernelOffer;',
        ]:
            with self.subTest(source=source), self.assertRaisesRegex(SystemExit, 'only test code'):
                check(source)
        dev = '[dev-dependencies]\nrenamed = {package="kernels-sm120", path="../kernels-sm120"}\n'
        with self.assertRaisesRegex(SystemExit, 'renamed'):
            check('use renamed::precision::Compute;', manifest=dev)
        # Test code may, and text that only mentions it names nothing.
        check('#[cfg(test)] mod tests;', 'use kernels_sm120::precision::KernelOffer;')
        check('#[cfg(test)] mod tests { use renamed::precision::Compute; }', manifest=dev)
        check('#[cfg(test)] mod tests { use kernels_sm120::precision::KernelOffer; }\n'
              '// kernels_sm120::Profile is documentation, not an import.\n'
              'fn text() { let s = "kernels_sm120::Profile"; }\n')
        for source, test_source in [
            ('mod tests;', 'use kernels_sm120::precision::KernelOffer;'),
            ('#[cfg(any(test, feature = "enabled"))] mod tests { use kernels_sm120::Profile; }', None),
        ]:
            with self.subTest(source=source), self.assertRaises(SystemExit):
                check(source, test_source)
        # The manifest links it only for tests.
        check('', manifest='[dev-dependencies]\nkernels-sm120.workspace = true\n')
        for manifest in [
            '[dependencies]\nkernels-sm120 = {path="../kernels-sm120"}\n',
            '[build-dependencies]\nkernels-sm120 = {path="../kernels-sm120"}\n',
            '[dependencies]\nrenamed = {package="kernels-sm120", path="../kernels-sm120"}\n',
            '[target.\'cfg(unix)\'.dependencies]\nkernels-sm120 = {path="../kernels-sm120"}\n',
        ]:
            with self.subTest(manifest=manifest), self.assertRaisesRegex(SystemExit, 'dev-dependencies'):
                check('', manifest=manifest)

    def test_models_name_sm120_kernels_only_in_sm120_modules_or_frozen_files(self):
        import hashlib

        def check(files, listed=(), manifest=None, baseline=None):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                for name, text in files.items():
                    path = root / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(text)
                if manifest is not None:
                    (root / 'models/m/bundle/Cargo.toml').write_text(manifest)
                (root / 'models/m').mkdir(parents=True, exist_ok=True)
                (root / 'models/m' / core_policy.KERNEL_IMPORTS).write_text(
                    '# frozen\n' + ''.join(name + '\n' for name in listed))
                frozen = ['models/m/' + name for name in (listed if baseline is None else baseline)]
                (root / 'tools/model_sdk').mkdir(parents=True)
                (root / core_policy.KERNEL_IMPORT_BASELINE).write_text(
                    '# hashes\n' + ''.join(hashlib.sha256(name.encode()).hexdigest() + '\n' for name in frozen))
                core_policy.check_model_kernel_imports(root)

        use = 'use kernels_sm120::row::rmsnorm::RmsNorm;\n'
        # sm120 modules, as a directory or a file, may name the kernels.
        check({'models/m/bundle/src/sm120/mod.rs': use, 'models/m/contract/src/sm120.rs': use,
               'examples/e/bundle/src/sm120/catalog.rs': use})
        # Comments and strings name nothing.
        check({'models/m/bundle/src/lower.rs': '// kernels_sm120\nconst S: &str = "kernels_sm120";\n'})
        # A listed file keeps its import.
        check({'models/m/bundle/src/lower.rs': use}, listed=['bundle/src/lower.rs'])
        # A list may name fewer files than the baseline, never others.
        check({'models/m/bundle/src/lower.rs': use}, listed=['bundle/src/lower.rs'],
              baseline=['bundle/src/lower.rs', 'bundle/src/gone.rs'])
        with self.assertRaises(SystemExit):
            check({'models/m/bundle/src/lower.rs': use}, listed=['bundle/src/lower.rs'], baseline=[])
        failures = [
            # A new import elsewhere, also through a path or a renamed dependency.
            ({'models/m/bundle/src/lower.rs': use}, [], None),
            ({'examples/e/bundle/src/lib.rs': 'fn f() { kernels_sm120::comm::choose(); }\n'}, [], None),
            ({'models/m/bundle/src/lower.rs': 'use engine::comm;\n'}, [],
             '[dependencies]\nengine = { package = "kernels-sm120", path = "../k" }\n'),
            ({'models/m/bundle/src/lower.rs': 'use engine::comm;\n'}, [],
             '[target.\'cfg(unix)\'.dependencies]\nengine = { package = "kernels-sm120", path = "../k" }\n'),
            # A rename the workspace makes, inherited with name.workspace = true.
            ({'Cargo.toml': '[workspace]\n[workspace.dependencies]\nengine = { package = "kernels-sm120", path = "k" }\n',
              'models/m/bundle/src/lower.rs': 'use engine::comm;\n'}, [], '[dependencies]\nengine.workspace = true\n'),
            # A string continued over lines, or a quote character literal, ends where Rust ends it.
            ({'models/m/bundle/src/lower.rs':
              'const S: &str = "a \\\n b";\nuse kernels_sm120::comm;\nconst T: &str = "c";\n'}, [], None),
            ({'models/m/bundle/src/lower.rs':
              'const Q: char = \'"\';\nuse kernels_sm120::comm;\nconst T: &str = "c";\n'}, [], None),
            # A listed file that no longer imports, or that moved into an sm120 module.
            ({'models/m/bundle/src/lower.rs': 'fn f() {}\n'}, ['bundle/src/lower.rs'], None),
            ({'models/m/bundle/src/sm120/mod.rs': use}, ['bundle/src/sm120/mod.rs'], None),
            ({}, ['bundle/src/gone.rs'], None),
            # One model's list does not cover another model's files.
            ({'models/n/bundle/src/lower.rs': use}, ['bundle/src/lower.rs'], None),
        ]
        for files, listed, manifest in failures:
            with self.subTest(files=files, listed=listed), self.assertRaises(SystemExit):
                check(files, listed, manifest)

    def frozen(self, list_name, baseline_name, check, files, listed=(), manifest=None, baseline=None):
        import hashlib

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, text in files.items():
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text)
            if manifest is not None:
                (root / 'models/m/bundle/Cargo.toml').write_text(manifest)
            (root / 'models/m').mkdir(parents=True, exist_ok=True)
            (root / 'models/m' / list_name).write_text('# frozen\n' + ''.join(line + '\n' for line in listed))
            frozen = ['models/m/' + line for line in (listed if baseline is None else baseline)]
            (root / 'tools/model_sdk').mkdir(parents=True)
            (root / baseline_name).write_text(
                '# hashes\n' + ''.join(hashlib.sha256(line.encode()).hexdigest() + '\n' for line in frozen))
            check(root)

    def test_models_reexport_sm120_items_only_in_sm120_modules_tests_or_frozen_lines(self):
        def check(files, listed=(), manifest=None, baseline=None):
            self.frozen(core_policy.SM120_REEXPORTS, core_policy.SM120_REEXPORT_BASELINE,
                        core_policy.check_model_sm120_reexports, files, listed, manifest, baseline)

        glob = 'pub use crate::sm120::launch::*;\n'
        line = 'bundle/src/launch.rs crate::sm120::launch::*'
        passes = [
            # sm120 modules, as a directory or a file, may re-export.
            {'models/m/bundle/src/sm120/mod.rs': glob, 'models/m/contract/src/sm120.rs': glob,
             'examples/e/bundle/src/sm120/launch.rs': 'pub use kernels_sm120::moe::Phase;\n'},
            # Private imports, other modules and renames to sm120 re-export no SM120 item.
            {'models/m/bundle/src/launch.rs': 'use crate::sm120::launch::*;\npub(self) fn f() {}\n'
             'pub use crate::sm120_tables::Rows;\npub use crate::rows as sm120;\n'},
            # Comments and strings name nothing.
            {'models/m/bundle/src/launch.rs': '// pub use crate::sm120::launch::*;\n'
             'const S: &str = "pub use crate::sm120::launch::*;";\n'},
            # Test code: cfg(test) items, the files only they declare, and tests directories.
            {'models/m/bundle/src/launch.rs': '#[cfg(test)]\npub use crate::sm120::launch::*;\n'
             '#[cfg(test)]\nmod tests {\n    pub use crate::sm120::launch::*;\n}\n#[cfg(test)]\nmod fixtures;\n',
             'models/m/bundle/src/launch/fixtures.rs': glob, 'models/m/contract/Cargo.toml': '[package]\n',
             'models/m/contract/tests/sm120_rows.rs': glob,
             'models/m/contract/tests/support/mod.rs': glob},
            # Module paths and includes that stay outside sm120, private globs and globs of other modules.
            {'models/m/bundle/src/lib.rs': '#[path = "../../compiler/src/rows.rs"]\nmod rows;\n'
             'include!(concat!(env!("OUT_DIR"), "/embedded.rs"));\nuse super::*;\npub use crate::rows::*;\n'},
        ]
        for files in passes:
            with self.subTest(files=files):
                check(files)
        # A listed line keeps its re-export. The baseline holds the listed lines' hashes, no
        # others: a retired line's hash goes with it, so the line cannot come back unreviewed.
        check({'models/m/bundle/src/launch.rs': glob}, listed=[line])
        for baseline in ([], [line, 'bundle/src/gone.rs crate::sm120::comm']):
            with self.subTest(baseline=baseline), self.assertRaises(SystemExit):
                check({'models/m/bundle/src/launch.rs': glob}, listed=[line], baseline=baseline)
        renamed = '[dependencies]\nengine = { package = "kernels-sm120", path = "../k" }\n'
        workspace = '[workspace]\n[workspace.dependencies]\ngpu = { package = "kernels-sm120", path = "k" }\n'
        failures = [
            # Re-exports at any visibility, by glob, name or group, and through any sm120 path.
            ({'models/m/bundle/src/launch.rs': glob}, [], None),
            ({'models/m/bundle/src/launch.rs': 'pub(crate) use crate::sm120::comm;\n'}, [], None),
            ({'models/m/bundle/src/launch.rs': 'pub(super) use super::sm120::comm::Plan as Comm;\n'}, [], None),
            ({'models/m/bundle/src/launch.rs': 'pub(in crate::lower) use self::sm120::{comm, rows::Rows};\n'}, [], None),
            ({'models/m/bundle/src/launch.rs': 'pub use model_m_contract::{sm120::entries::dense, shapes};\n'}, [], None),
            ({'models/m/contract/src/lib.rs': 'pub mod sm120;\npub use sm120 as kernels;\n'}, [], None),
            ({'models/m/bundle/src/launch.rs': 'pub use ::r#kernels_sm120::moe::Phase;\n'}, [], None),
            ({'models/m/bundle/src/launch.rs': 'pub use engine::moe::Phase;\n'}, [], renamed),
            # A rename the workspace makes.
            ({'Cargo.toml': workspace, 'models/m/bundle/src/launch.rs': 'pub use gpu::moe::Phase;\n'}, [],
             '[dependencies]\ngpu.workspace = true\n'),
            # Globs of crate, super or self can carry an sm120 module along unnamed.
            ({'models/m/bundle/src/launch.rs': 'pub use super::*;\n'}, [], None),
            ({'models/m/bundle/src/launch.rs': 'pub(crate) use crate::{*};\n'}, [], None),
            ({'models/m/bundle/src/launch.rs': 'pub use self::super::*;\n'}, [], None),
            ({'examples/e/bundle/src/launch.rs': 'pub mod attention {\n    pub use crate::sm120::attention::*;\n}\n'},
             [], None),
            # A tests directory that is no crate's is production code.
            ({'models/m/bundle/src/tests/mod.rs': glob}, [], None),
            # A module whose path reaches into an sm120 module.
            ({'models/m/bundle/src/lib.rs': '#[path = "sm120/launch.rs"]\npub mod launch;\n'}, [], None),
            ({'models/m/bundle/src/lower.rs': '#[path = r"../sm120.rs"]\nmod kernels;\n'}, [], None),
            ({'models/m/bundle/src/lib.rs': '#[cfg_attr(unix, path = "sm120/launch.rs")]\nmod launch;\n'}, [], None),
            ({'models/m/bundle/src/lib.rs': '#[path = "sm\\x3120/launch.rs"]\nmod launch;\n'}, [], None),
            ({'models/m/bundle/src/lib.rs': '#[path = r#"sm120/launch.rs"#]\nmod launch;\n'}, [], None),
            # Including a file from an sm120 module.
            ({'models/m/bundle/src/lib.rs': 'mod launch {\n    include!("sm120/launch.rs");\n}\n'}, [], None),
            ({'models/m/bundle/src/lib.rs': 'const T: &str = include_str!(concat!("sm120/", "t.txt"));\n'}, [], None),
            # A listed line that no longer re-exports, or that sits in an sm120 module now.
            ({'models/m/bundle/src/launch.rs': 'use crate::sm120::launch::*;\n'}, [line], None),
            ({'models/m/bundle/src/sm120/launch.rs': glob}, ['bundle/src/sm120/launch.rs crate::sm120::launch::*'], None),
            # A listed file covers its listed paths only, and one model's list no other model.
            ({'models/m/bundle/src/launch.rs': glob + 'pub use crate::sm120::comm;\n'}, [line], None),
            ({'models/n/bundle/src/launch.rs': glob}, [line], None),
        ]
        for files, listed, manifest in failures:
            with self.subTest(files=files, listed=listed), self.assertRaises(SystemExit):
                check(files, listed, manifest)


    def test_models_name_sm120_modules_only_in_them_tests_or_frozen_lines(self):
        def check(files, listed=(), baseline=None):
            self.frozen(core_policy.SM120_USES, core_policy.SM120_USE_BASELINE,
                        core_policy.check_model_sm120_uses, files, listed, None, baseline)

        use = 'use crate::sm120::overlap;\n'
        line = 'bundle/src/lower.rs crate::sm120::overlap'
        passes = [
            # sm120 modules may name each other.
            {'models/m/bundle/src/sm120/mod.rs': use, 'models/m/contract/src/sm120.rs': use},
            # Declaring the module, kernels_sm120 (KERNEL_IMPORTS) and re-exports (SM120_REEXPORTS)
            # are other lists' business; comments, strings and other names name nothing.
            {'models/m/bundle/src/lib.rs': 'pub mod sm120;\nmod lower;\nuse kernels_sm120::moe;\n'
             'pub use crate::sm120::comm;\nfn sm120() {}\nconst S: &str = "crate::sm120::x";\n'
             '// crate::sm120::overlap\nuse crate::sm120_tables::Rows;\n'},
            # Test code: cfg(test) items, the files only they declare (also by #[path]), tests directories.
            {'models/m/bundle/src/lib.rs': '#[cfg(test)]\nmod fake;\n#[cfg(test)]\nmod t {\n    ' + use + '}\n',
             'models/m/bundle/src/fake.rs': '#[path = "journal_tests.rs"]\nmod journal;\n' + use,
             'models/m/bundle/src/journal_tests.rs': 'fn f() { crate::sm120::overlap::adjust(); }\n',
             'models/m/bundle/src/lower.rs': '#[cfg(test)]\n#[path = "lower_tests.rs"]\nmod tests;\n',
             'models/m/bundle/src/lower_tests.rs': use,
             'models/m/contract/Cargo.toml': '[package]\n', 'models/m/contract/tests/rows.rs': use},
        ]
        for files in passes:
            with self.subTest(files=files):
                check(files)
        # Test modules at any visibility and attribute order, by normalized paths, and the
        # out-of-line children of inline test modules.
        check({'models/m/bundle/src/lib.rs': '#[cfg(test)]\npub mod a;\n#[cfg(test)]\npub(crate) mod b;\n'
               '#[path = "c_tests.rs"]\n#[cfg(test)]\nmod c;\n#[cfg(test)]\n#[path = "x/../d.rs"]\nmod d;\n'
               '#[cfg(test)]\nmod e {\n    mod helpers;\n}\n',
               **{f'models/m/bundle/src/{name}': use for name in ('a.rs', 'b.rs', 'c_tests.rs', 'd.rs', 'e/helpers.rs')}})
        # A listed line keeps its use; the baseline holds the listed lines' hashes, no others.
        check({'models/m/bundle/src/lower.rs': use}, listed=[line])
        for baseline in ([], [line, 'bundle/src/gone.rs crate::sm120::x']):
            with self.subTest(baseline=baseline), self.assertRaises(SystemExit):
                check({'models/m/bundle/src/lower.rs': use}, listed=[line], baseline=baseline)
        failures = [
            # Private uses, through any sm120 path, and paths in code.
            ({'models/m/bundle/src/lower.rs': use}, []),
            ({'models/m/bundle/src/lower.rs': 'use super::{sm120::overlap::Facts, rows};\n'}, []),
            ({'models/m/bundle/src/lower.rs': 'pub(self) use model_m_contract::sm120::stacked;\n'}, []),
            ({'models/m/bundle/src/lower.rs': 'use crate::sm120;\n'}, []),
            ({'models/m/bundle/src/lower.rs': 'fn f() { let b = crate::sm120::backend(); }\n'}, []),
            ({'models/m/bundle/src/lower.rs': 'fn f(p: &model_m_contract::sm120::stacked::Part) {}\n'}, []),
            ({'examples/e/bundle/src/lower.rs': 'fn f() { sm120::comm::plan::<u8>(); }\n'}, []),
            # A file a production module declares is production, also by #[path].
            ({'models/m/bundle/src/lib.rs': '#[path = "fake.rs"]\nmod fake;\n',
              'models/m/bundle/src/fake.rs': use}, []),
            ({'models/m/bundle/src/lower.rs': '#[cfg(test)]\n#[path = "lower_tests.rs"]\nmod tests;\n',
              'models/m/bundle/src/lower_other.rs': use}, []),
            # A file that test code and production code both declare is production, with its children.
            ({'models/m/bundle/src/lib.rs': '#[cfg(test)]\n#[path = "shared.rs"]\nmod a;\n#[path = "shared.rs"]\nmod b;\n',
              'models/m/bundle/src/shared.rs': '#[path = "child.rs"]\nmod child;\n',
              'models/m/bundle/src/child.rs': use}, []),
            # A listed line that went, a new item in a listed file, another model's line.
            ({'models/m/bundle/src/lower.rs': 'use crate::rows;\n'}, [line]),
            ({'models/m/bundle/src/lower.rs': use + 'use crate::sm120::comm;\n'}, [line]),
            ({'models/n/bundle/src/lower.rs': use}, [line]),
        ]
        for files, listed in failures:
            with self.subTest(files=files, listed=listed), self.assertRaises(SystemExit):
                check(files, listed)

    def write_tree(self, root, files):
        for name, text in files.items():
            (root / name).parent.mkdir(parents=True, exist_ok=True)
            (root / name).write_text(text)

    def refusals(self, root, cases, allowed):
        """Each case changes files (and the allowlist); the refusals name
        exactly the `expected` file or files. The tree is clean again afterwards."""
        for expected, changes, extra in cases:
            with self.subTest(expected=expected, changes=sorted(changes)):
                originals = {name: (root / name).read_text() if (root / name).exists() else None for name in changes}
                self.write_tree(root, changes)
                try:
                    with patch.object(sdk_test_support, 'ALLOWED', allowed | extra, create=True):
                        found = sdk_test_support.violations(root)
                finally:
                    for name, original in originals.items():
                        if original is None:
                            (root / name).unlink()
                        else:
                            (root / name).write_text(original)
                expected = (expected,) if isinstance(expected, str) else expected
                self.assertEqual({line.split(':')[0] for line in found}, set(expected), found)
        with patch.object(sdk_test_support, 'ALLOWED', allowed, create=True):
            self.assertEqual(sdk_test_support.violations(root), [])

    def test_sdk_build_sources_never_name_its_test_support(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(['git', 'init', '-q', root], check=True)
            sdk = 'crates/model-compiler-sdk'
            self.write_tree(root, {f'{sdk}/Cargo.toml': '[package]\nname = "model-compiler-sdk"\n',
                                   f'{sdk}/src/lib.rs': 'pub mod nested;\n',
                                   f'{sdk}/src/nested/mod.rs': 'pub fn f() {}\n',
                                   f'{sdk}/test-support/checkpoint.rs': 'pub fn fixture() {}\n'})
            self.refusals(root, (
                (f'{sdk}/src/nested/mod.rs',
                 {f'{sdk}/src/nested/mod.rs': '#[path = "../../test-support/checkpoint.rs"]\nmod checkpoint;\n'}, {}),
                (f'{sdk}/src/lib.rs', {f'{sdk}/src/lib.rs': '#[cfg(feature = "test-support")]\npub mod test_support;\n'}, {}),
                (f'{sdk}/build.rs', {f'{sdk}/build.rs': 'fn main() { println!("cargo:rerun-if-changed=test-support"); }\n'}, {}),
                (f'{sdk}/Cargo.toml', {f'{sdk}/Cargo.toml': '[package]\nname = "model-compiler-sdk"\n[features]\ntest-support = []\n'}, {}),
            ), {})
        self.assertEqual(sdk_test_support.violations(ROOT), [])

    def test_only_allowlisted_test_files_name_the_sdk_test_support(self):
        native = '{}/crates/model-compiler-sdk/test-support/native.rs'
        host, model = 'crates/host', 'models/m/compiler'
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(['git', 'init', '-q', root], check=True)
            self.write_tree(root, {
                'Cargo.toml': '[workspace]\nmembers = ["crates/*", "models/m/compiler"]\n',
                'crates/model-compiler-sdk/Cargo.toml': '[package]\nname = "model-compiler-sdk"\n',
                'crates/model-compiler-sdk/src/lib.rs': 'pub fn f() {}\n',
                'crates/model-compiler-sdk/test-support/native.rs': 'pub fn compile() {}\n',
                f'{host}/Cargo.toml': '[package]\nname = "host"\n',
                f'{host}/src/main.rs': 'fn main() {}\n',
                # Another crate's own feature and module of that name are not the SDK's directory.
                'crates/frontend/Cargo.toml': '[package]\nname = "frontend"\n[features]\ntest-support = []\n',
                'crates/frontend/src/lib.rs': '#[cfg(feature = "test-support")]\npub mod test_support;\n',
                'crates/frontend/src/test_support.rs': 'pub fn cases() {}\n',
                # The allowlisted includers: a #[cfg(test)] module and a tests/ helper it and a test load.
                f'{model}/Cargo.toml': '[package]\nname = "model-m"\n',
                f'{model}/src/lib.rs': '#[cfg(test)]\nmod test_native;\npub mod plan;\n',
                f'{model}/src/plan.rs': 'pub fn plan() {}\n',
                f'{model}/src/test_native.rs': '#[path = "../tests/support/mod.rs"]\nmod support;\n'
                                               f'#[path = "{native.format("../../../..")}"]\nmod native;\n',
                f'{model}/tests/support/mod.rs': f'#[path = "{native.format("../../../../..")}"]\nmod native;\n',
                f'{model}/tests/headers.rs': '#[path = "support/mod.rs"]\nmod support;\n',
                'tests/guard.py': f'SUPPORT = "{native.format(".")}"\n',
            })
            allowed = {
                f'{model}/src/test_native.rs': ('cfg-test', 'fixture', (f'{model}/src/lib.rs',)),
                f'{model}/tests/support/mod.rs': ('tests', 'helper', (f'{model}/tests/headers.rs', f'{model}/src/test_native.rs')),
                'tests/guard.py': ('guard', 'guard', ()),
            }
            bridge = f'#[path = "{native.format("../../..")}"]\nmod native;\n'
            cases = (
                # Build scripts: build.rs, and a Cargo `build` key without a .rs suffix.
                (f'{host}/build.rs', {f'{host}/build.rs': f'const N: &str = include_str!("{native.format("../..")}");\nfn main() {{}}\n'}, {}),
                (f'{host}/gen/driver', {f'{host}/Cargo.toml': '[package]\nname = "host"\nbuild = "gen/driver"\n',
                                        f'{host}/gen/driver': f'{bridge}fn main() {{}}\n'}, {}),
                ('crates/model-compiler-sdk/build/build.rs',
                 {'crates/model-compiler-sdk/build/build.rs': 'fn main() { println!("cargo:rerun-if-changed=../test-support"); }\n'}, {}),
                # Served modules and roots, by any spelling and through any file.
                (f'{model}/src/plan.rs', {f'{model}/src/plan.rs': f'#[path = "{native.format("../../../..")}"]\nmod native;\n'}, {}),
                (f'{host}/src/main.rs', {f'{host}/src/main.rs': f'include!("{native.format("../../..")}");\nfn main() {{}}\n'}, {}),
                (f'{model}/src/plan.rs', {f'{model}/src/plan.rs':
                    'const N: &str = include_str!(concat!("../../../../crates/model-compiler-sdk", "/", "test-support", "/native.rs"));\n'}, {}),
                (f'{model}/src/plan.rs', {f'{model}/src/plan.rs':
                    'const N: &str = include_str!(r#"../../../../crates / model-compiler-sdk / test-support / native.rs"#);\n'}, {}),
                (f'{host}/src/bridge.inc', {f'{host}/src/lib.rs': 'include!("bridge.inc");\n',
                                            f'{host}/src/bridge.inc': f'pub const N: &str = include_str!("{native.format("../..")}");\n'}, {}),
                (f'{host}/src/.generated/bridge.rs', {f'{host}/src/lib.rs': '#[path = ".generated/bridge.rs"]\nmod bridge;\n',
                                                     f'{host}/src/.generated/bridge.rs': f'pub const N: &str = include_str!("{native.format("../../..")}");\n'}, {}),
                (f'{host}/tests/bridge.rs', {f'{host}/src/lib.rs': 'include ! ("../tests/bridge.rs");\n', f'{host}/tests/bridge.rs': bridge}, {}),
                (f'{host}/tests/bridge.rs', {f'{host}/src/lib.rs': 'include!(r"../tests/bridge.rs");\n', f'{host}/tests/bridge.rs': bridge}, {}),
                (f'{host}/tests/bridge.rs', {f'{host}/src/lib.rs': '#[path = r"../tests/bridge.rs"]\nmod bridge;\n', f'{host}/tests/bridge.rs': bridge}, {}),
                # A file a #[cfg(test)] module and a Cargo example share, even when allowlisted.
                (f'{host}/examples/demo.rs', {f'{host}/src/lib.rs': '#[cfg(test)]\n#[path = "../examples/demo.rs"]\nmod demo;\n',
                                              f'{host}/examples/demo.rs': f'{bridge}fn main() {{}}\n'}, {}),
                (f'{host}/examples/demo.rs', {f'{host}/src/lib.rs': '#[cfg(test)]\n#[path = "../examples/demo.rs"]\nmod demo;\n',
                                              f'{host}/examples/demo.rs': f'{bridge}fn main() {{}}\n'},
                 {f'{host}/examples/demo.rs': ('cfg-test', 'example', (f'{host}/src/lib.rs',))}),
                # A served module that loads an allowlisted file without naming the test support.
                (f'{model}/src/plan.rs', {f'{model}/src/plan.rs': 'include ! (r"../tests/support/mod.rs");\n'}, {}),
                (f'{model}/src/plan.rs', {f'{model}/src/plan.rs': '#[path = r"../tests/support/mod.rs"]\nmod support;\n'}, {}),
                # Allowlist entries that cannot be verified.
                ((f'{model}/src/test_native.rs', f'{model}/tests/support/mod.rs'),
             {f'{model}/src/lib.rs': 'mod test_native;\npub mod plan;\n'}, {}),
                (f'{model}/tests/support/mod.rs', {f'{model}/src/plan.rs': '#[path = "../tests/support/mod.rs"]\nmod support;\n'},
                 {f'{model}/tests/support/mod.rs':
                      ('tests', 'helper', (f'{model}/tests/headers.rs', f'{model}/src/test_native.rs', f'{model}/src/plan.rs'))}),
                (f'{model}/src/gone.rs', {}, {f'{model}/src/gone.rs': ('cfg-test', 'gone', (f'{model}/src/lib.rs',))}),
            )
            # A served alias of a #[cfg(test)] module, with cfg(test) in a comment, in a string or on
            # another item.
            cases += tuple(((f'{model}/src/test_native.rs', f'{model}/tests/support/mod.rs'),
                       {f'{model}/src/lib.rs': '#[cfg(test)]\nmod test_native;\npub mod plan;\n' + alias}, {}) for alias in (
                '#[path = "test_native.rs"]\nmod served_native; // #[cfg(test)]\n',
                '/* #[cfg(test)] */\n#[path = "test_native.rs"]\nmod served_native;\n',
                '#[doc = "#[cfg(test)]"]\n#[path = "test_native.rs"]\nmod served_native;\n',
                '#[cfg(test)]\nfn helper() {}\n#[path = "test_native.rs"]\nmod served_native;\n',
            ))
            # Literal paths that normalize to the test support or to an allowlisted file.
            cases += tuple(
                (expected, {path: text}, {}) for expected, path, text in (
                    (f'{host}/build.rs', f'{host}/build.rs',
                     'const N: &str = include_str!("../model-compiler-sdk/./test-support/./native.rs");\n'),
                    (f'{host}/build.rs', f'{host}/build.rs',
                     'const N: &str = include_str!("../model-compiler-sdk//test-support///native.rs");\n'),
                    (f'{host}/build.rs', f'{host}/build.rs',
                     'const N: &str = include_str!("../model-compiler-sdk/src/../test-support/native.rs");\n'),
                    (f'{model}/src/plan.rs', f'{model}/src/plan.rs', 'include!("../tests/./support//mod.rs");\n'),
                ))
            self.refusals(root, cases, allowed)

if __name__ == '__main__':
    unittest.main()
