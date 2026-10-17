"""Behavioral obligations and negative consumer cases, independent of installed assets."""
from copy import deepcopy
from dataclasses import replace
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import tomllib
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.model_sdk import inventory
# Every model a broad change selects.
MODELS = set(inventory.shipped(ROOT))
sys.path.insert(0, str(ROOT / "ci"))
sys.path.insert(0, str(ROOT / "tools/qual"))
import change_scope
import quick
import release_gate
import validation
import kernel_scope
import kernel_reports as kr
from kernel_report_fixture import HEADER, VERSION, make_report, to_directory, write_report


def metadata_fixture(root=ROOT):
    """Cargo-shaped all-feature fixture from manifests, not a second production resolver."""
    workspace = tomllib.loads((root / 'Cargo.toml').read_text())['workspace']
    packages, manifests = [], {}
    for pattern in workspace['members']:
        for path in sorted(root.glob(pattern + '/Cargo.toml')):
            data = tomllib.loads(path.read_text())
            name = data['package']['name']
            manifests[name] = data
            targets = [{'name': b['name'], 'kind': ['bin']} for b in data.get('bin', [])]
            library = data.get('lib', {})
            if library or (path.parent / 'src/lib.rs').is_file():
                targets.append({'name': library.get('name', name.replace('-', '_')),
                                'kind': library.get('crate-type', ['lib'])})
            targets += [{'name': test['name'], 'kind': ['test']} for test in data.get('test', [])]
            if data['package'].get('autotests', True):
                for test in (path.parent / 'tests').glob('*'):
                    if not (test.suffix == '.rs' or (test.is_dir() and (test / 'main.rs').is_file())):
                        continue
                    name_of_test = test.stem if test.is_file() else test.name
                    if not any(t['name'] == name_of_test and 'test' in t['kind'] for t in targets):
                        targets.append({'name': name_of_test, 'kind': ['test']})
            for binary in (path.parent / 'src/bin').glob('*'):
                if not (binary.suffix == '.rs' or (binary.is_dir() and (binary / 'main.rs').is_file())):
                    continue
                name_of_bin = binary.stem if binary.is_file() else binary.name
                if not any(t['name'] == name_of_bin for t in targets):
                    targets.append({'name': name_of_bin, 'kind': ['bin']})
            packages.append({'id': name, 'name': name, 'manifest_path': str(path), 'targets': targets})
    nodes = []
    for name, manifest in manifests.items():
        dependencies = []
        for target, tables in [(None, manifest), *manifest.get('target', {}).items()]:
            for table, kind in [('dependencies', None), ('build-dependencies', 'build'), ('dev-dependencies', 'dev')]:
                for dep, options in tables.get(table, {}).items():
                    options = options if isinstance(options, dict) else {}
                    dep = options.get('package', dep)
                    if dep in manifests:
                        dependencies.append({'pkg': dep, 'name': dep, 'dep_kinds': [{'kind': kind, 'target': target}]})
        nodes.append({'id': name, 'deps': dependencies, 'dependencies': [d['pkg'] for d in dependencies]})
    return {'workspace_root': str(root), 'workspace_members': list(manifests), 'packages': packages, 'resolve': {'nodes': nodes}}


class Reports(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.metadata = metadata_fixture()
        cls.graph = change_scope.Graph(cls.metadata)
        cls.catalog = validation.Catalog(ROOT, cls.metadata)
        if cls.catalog.issues or cls.graph.issues:
            raise AssertionError((cls.catalog.issues, cls.graph.issues))

    def report(self, *paths, **kwargs):
        return validation.build_report(self.metadata, paths, graph=self.graph, catalog=self.catalog, **kwargs)

    def ids(self, report):
        return {row['id'] for row in report['obligations'] if row['evidence']['selection'] == 'selected'}

    def models(self, report):
        return {row['model'] for row in report['profiles']}

    def item(self, report, identifier):
        return next(row for row in report['obligations'] if row['id'] == identifier)

    def kernel_fixture(self, directory, package, *, path=HEADER, source='chopinfer', selection='fixture.v1', content=None):
        path, data, bindings = make_report(directory, package, source=source, paths=(path,),
                                           content=(ROOT / path).read_bytes() if content is None else content)
        data['context']['selection'] = selection
        write_report(path, data)
        return path, data

    def consumers(self, report, selected=False):
        return [c for c in report['artifacts']['kernel_consumers'] if not selected or c['selected']]

    def test_shared_source_identity_joins_production_lab_and_exported_sdk(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = []
            for name in ['bundle-qwen38', 'bundle-mimo26', 'kernel-lab', 'runtime-host']:
                path, _ = self.kernel_fixture(root / name, name + '@' + VERSION,
                    path='crates/runtime-host/src/lib.rs' if name == 'runtime-host' else HEADER)
                paths.append(path)
            sdk, data = self.kernel_fixture(root / 'sdk', 'example-bundle@0.1.0', source='external-model', selection='example.kernels.v1')
            paths.append(sdk)
            report = self.report(HEADER, kernel_reports=paths)
            hits = [c for c in self.consumers(report) if c['affected_sources']]
            self.assertEqual({c['identity']['package'].split('@')[0] for c in hits},
                             {'bundle-qwen38', 'bundle-mimo26', 'kernel-lab', 'example-bundle'})
            for consumer in hits:
                self.assertTrue(consumer['current_evidence'])
                self.assertEqual(consumer['affected_sources'], [{'kind': 'Project', 'owner': 'chopinfer', 'path': HEADER}])
                self.assertEqual(consumer['affected_cubins'][0]['name'], 'fixture_v1')
                self.assertEqual(consumer['variants'][0]['lock']['status'], 'Verified')
                self.assertTrue(consumer['variants'][0]['cubins'][0]['identity']['cubin_sha256'])
            unrelated = next(c for c in self.consumers(report) if c['identity']['package'].startswith('runtime-host@'))
            self.assertFalse(unrelated['selected'])
            self.assertIn('no membership hit', unrelated['membership_decision'])
            self.assertNotIn('example-bundle', report['artifacts']['packages'])
            build = self.item(report, 'shared:kernel_build')
            self.assertIn(str(sdk.parent / 'kernels.lock'), [l['path'] for l in build['locks']])
            self.assertTrue(any('--feature' in c and 'cuda-probe' in c for c in build['commands']))
            self.assertFalse(report['qualified'])
            self.assertEqual(report, self.report(HEADER, kernel_reports=list(reversed(paths))))

    def test_digest_changes_previous_only_consumers_and_removed_membership(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            current, _ = self.kernel_fixture(root / 'now', f'bundle-qwen38@{VERSION}')
            old, old_data = self.kernel_fixture(root / 'old', f'bundle-qwen38@{VERSION}', content=b'old contents')
            gone, gone_data = self.kernel_fixture(root / 'gone', 'removed-bundle@0.0.1', content=b'removed consumer')
            for value in [old_data, gone_data]:
                for file in value['files'].values():
                    Path(file['location']).unlink()
            report = self.report(HEADER, kernel_reports=[current], previous_kernel_reports=[old, gone])
            qwen = next(c for c in self.consumers(report) if c['identity']['package'].startswith('bundle-qwen38@'))
            self.assertTrue(qwen['current_evidence'])
            self.assertEqual({r['kind'] for r in qwen['reasons']}, {'current membership', 'previous membership'})
            removed = next(c for c in self.consumers(report) if c['identity']['package'] == 'removed-bundle@0.0.1')
            self.assertTrue(removed['selected'])
            self.assertIsNone(removed['workspace_package'])
            self.assertNotIn('removed-bundle', str(report['iteration']['commands']))
            # Current closure no longer includes the changed header.
            replacement, _ = self.kernel_fixture(root / 'replacement', f'bundle-qwen38@{VERSION}', path='README.md')
            union = self.report(HEADER, kernel_reports=[replacement], previous_kernel_reports=[old])
            qwen = next(c for c in self.consumers(union) if c['identity']['package'].startswith('bundle-qwen38@'))
            self.assertTrue(qwen['selected'])
            self.assertEqual(qwen['reasons'][0]['kind'], 'previous membership')
            deleted = self.report('crates/kernels-sm120/cuda/removed.cuh', kernel_reports=[replacement])
            self.assertTrue(any('removed input without previous' in r['reason'] for c in self.consumers(deleted) for r in c['reasons']))

    def test_current_and_stale_variants_coexist_and_keep_their_reasons(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fresh, _ = self.kernel_fixture(root / 'fresh', f'kernel-lab@{VERSION}')
            stale, data = self.kernel_fixture(root / 'stale', f'kernel-lab@{VERSION}')
            Path(data['compiler_path']).write_bytes(b'changed compiler')
            report = self.report(HEADER, kernel_reports=[fresh, stale])
            lab = next(c for c in self.consumers(report) if c['identity']['package'].startswith('kernel-lab@'))
            self.assertTrue(lab['current_evidence'])
            self.assertEqual({v['status'] for v in lab['variants']}, {'current', 'stale'})
            self.assertTrue(any('stale compiler' in r['reason'] for r in lab['reasons']))
            self.assertEqual(report, self.report(HEADER, kernel_reports=[stale, fresh]))

    def test_accumulated_stale_variants_keep_current_consumer(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fresh, data = self.kernel_fixture(root / 'z-current', f'kernel-lab@{VERSION}')
            paths = []
            for index in range(48):
                old, _ = self.kernel_fixture(root / f'a-old-{index:02}', f'kernel-lab@{VERSION}',
                                             content=b'old contents')
                paths.append(old)
            size = sum(f['bytes'] for f in data['files'].values())
            index = kernel_scope.Index(self.metadata, self.graph, self.catalog, [*paths, fresh],
                                       limits=replace(kr.Limits(), closure_bytes=size))
            row = index.consumers[('chopinfer', f'kernel-lab@{VERSION}', 'fixture.v1')]
            current = [v for v in row['variants'] if v['status'] == 'current']
            self.assertEqual([v['path'] for v in current], [str(fresh)])
            self.assertEqual(sum(v['status'] == 'stale' for v in row['variants']), 48)
            self.assertEqual(index.reader.files.closure_bytes, size)
            self.assertFalse(index.issues)

    def test_conflicting_current_variants_retain_consumers(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            a, _ = self.kernel_fixture(root / 'a', f'kernel-lab@{VERSION}', path='README.md')
            b, data = self.kernel_fixture(root / 'b', f'kernel-lab@{VERSION}', path='docs/testing.md')
            report = self.report(HEADER, kernel_reports=[a, b])
            lab = next(c for c in self.consumers(report) if c['identity']['package'].startswith('kernel-lab@'))
            self.assertTrue(lab['current_evidence'])
            self.assertEqual(lab['conflicts'], ['fixture_v1'])
            self.assertTrue(any('conflicting current' in r['reason'] for r in lab['reasons']))

    def test_missing_evidence_respects_host_frontend_docs_and_local_domains(self):
        with tempfile.TemporaryDirectory() as temporary:
            stale, data = self.kernel_fixture(Path(temporary) / 'stale', f'kernel-lab@{VERSION}')
            Path(data['compiler_path']).unlink()
            for inputs in [[], [Path(temporary) / 'missing'], [stale]]:
                for path in ['crates/runtime-host/src/lib.rs', 'models/qwen38/compiler/src/frontend.rs', 'README.md']:
                    report = self.report(path, kernel_reports=inputs)
                    self.assertEqual(report['artifacts']['packages'], self.graph.select([path])['packages'])
                    self.assertEqual(self.consumers(report, selected=True), [])
            shared = self.report(HEADER)
            selected = {c['identity']['package'].split('@')[0] for c in self.consumers(shared, selected=True)}
            self.assertTrue({'bundle-qwen38', 'bundle-mimo26', 'kernel-lab', 'runtime-kernels', 'example-bundle'} <= selected)
            local = self.report('models/qwen38/bundle/cuda/qwen38_bundle.cuh')
            selected = {c['identity']['package'].split('@')[0] for c in self.consumers(local, selected=True)}
            self.assertIn('bundle-qwen38', selected)
            self.assertIn('kernel-lab', selected)
            self.assertNotIn('bundle-mimo26', selected)
            self.assertNotIn('example-bundle', selected)
            external = self.report('examples/external-model/bundle/src/sm120/selection.rs')
            self.assertTrue(any(c['identity']['source'] == 'external-model' for c in self.consumers(external, selected=True)))
            frontend = self.report('examples/external-model/compiler/src/lib.rs')
            self.assertEqual(self.consumers(frontend, selected=True), [])

    def test_external_sdk_follows_real_rust_edges_despite_no_cuda_hit(self):
        with tempfile.TemporaryDirectory() as temporary:
            sdk, data = self.kernel_fixture(Path(temporary), 'example-bundle@0.1.0', source='external-model', selection='example.kernels.v1')
            for path in ['crates/kernels-sm120/src/row/rmsnorm.rs', 'crates/kernel-core/src/lib.rs', 'crates/kernel-build/src/lib.rs']:
                report = self.report(path, kernel_reports=[sdk])
                external = next(c for c in self.consumers(report) if c['identity']['source'] == 'external-model')
                self.assertTrue(external['selected'])
                self.assertEqual(external['affected_sources'], [])
                self.assertTrue(any(r['kind'] == 'external SDK dependency' and r['chain'][0]['kind'] == 'owner' for r in external['reasons']))
            broad = self.report('README.md', kernel_reports=[sdk], all_checks=True)
            self.assertTrue(all(c['selected'] for c in self.consumers(broad)))

    def test_precision_selection_is_retained_and_reproduced(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            probe, _ = self.kernel_fixture(root / 'probe', 'example-bundle@0.1.0',
                source='external-model', selection='example.kernels.v1')
            report = self.report(HEADER, kernel_reports=[probe])
            external = {c['identity']['selection']: c for c in self.consumers(report)
                        if c['identity']['source'] == 'external-model'}
            self.assertIn('example.precision.v1', external)
            precision = external['example.precision.v1']
            self.assertTrue(precision['selected'])
            self.assertFalse(precision['current_evidence'])
            self.assertTrue(any('missing current report' in r['reason'] for r in precision['reasons']))
            commands = self.item(report, 'shared:kernel_build')['commands']
            self.assertTrue(any('--feature' in c and 'precision-exercise' in c for c in commands))
            mixed, _ = self.kernel_fixture(root / 'precision', 'example-bundle@0.1.0',
                source='external-model', selection='example.precision.v1')
            report = self.report('examples/external-model/contract/src/precision.rs',
                                 kernel_reports=[probe, mixed])
            external = [c for c in self.consumers(report) if c['identity']['source'] == 'external-model']
            self.assertEqual({c['identity']['selection'] for c in external},
                             {'example.kernels.v1', 'example.precision.v1'})
            self.assertTrue(all(c['selected'] and c['current_evidence'] for c in external))
            commands = self.item(report, 'shared:kernel_build')['commands']
            self.assertTrue(any('--feature' in c and 'cuda-probe' in c for c in commands))
            self.assertTrue(any('--feature' in c and 'precision-exercise' in c for c in commands))

    def test_unknown_cargo_kinds_retain_producers_and_versions_bind_exactly(self):
        metadata = deepcopy(self.metadata)
        for node in metadata['resolve']['nodes']:
            node.pop('deps')
        index = kernel_scope.Index(metadata, change_scope.Graph(metadata, scan=False), self.catalog)
        expected = {identity[1].split('@')[0] for identity in index.expected}
        self.assertTrue({'bundle-qwen38', 'bundle-mimo26', 'kernel-lab', 'runtime-kernels'} <= expected)
        with tempfile.TemporaryDirectory() as temporary:
            wrong, _ = self.kernel_fixture(Path(temporary), 'bundle-qwen38@99.0.0')
            report = self.report(HEADER, kernel_reports=[wrong])
            wrong_row = next(c for c in self.consumers(report) if c['identity']['package'].endswith('@99.0.0'))
            self.assertEqual(wrong_row['mapping'], 'unresolved')
            self.assertFalse(wrong_row['current_evidence'])
            self.assertTrue(any('missing current report' in r['reason'] for c in self.consumers(report)
                                if c['identity']['package'].startswith('bundle-qwen38@0.') for r in c['reasons']))

    def test_ambiguous_cargo_consumer_names_cannot_bind_reports(self):
        metadata = deepcopy(self.metadata)
        duplicate = deepcopy(next(p for p in metadata['packages'] if p['name'] == 'bundle-qwen38'))
        duplicate['id'] += '-duplicate'
        metadata['packages'].append(duplicate)
        metadata['workspace_members'].append(duplicate['id'])
        metadata['resolve']['nodes'].append({'id': duplicate['id'], 'dependencies': []})
        graph = change_scope.Graph(metadata, scan=False)
        with tempfile.TemporaryDirectory() as temporary:
            path, _ = self.kernel_fixture(Path(temporary), f'bundle-qwen38@{VERSION}')
            report = validation.build_report(metadata, [HEADER], graph=graph, catalog=self.catalog, kernel_reports=[path])
            row = next(c for c in self.consumers(report) if c['identity']['package'].startswith('bundle-qwen38@'))
            self.assertEqual(row['mapping'], 'unresolved')
            self.assertFalse(row['current_evidence'])
            self.assertTrue(row['selected'])
            self.assertTrue(report['artifacts']['workspace'])
            self.assertIn('--workspace', report['iteration']['commands'][0])

    def test_incomplete_and_invalid_reports_retain_expected_consumers(self):
        with tempfile.TemporaryDirectory() as temporary:
            path, original = self.kernel_fixture(Path(temporary), f'bundle-qwen38@{VERSION}', path='README.md')
            for field, value in [('schema', 'unsupported'), ('source_membership_complete', False),
                                 ('source_ownership_complete', False), ('lock_status', 'Skipped')]:
                data = deepcopy(original)
                data[field] = value
                write_report(path, data)
                report = self.report(HEADER, kernel_reports=[path])
                qwen = next(c for c in self.consumers(report) if c['identity']['package'].startswith('bundle-qwen38@'))
                self.assertTrue(qwen['selected'])
                self.assertFalse(qwen['current_evidence'])
                self.assertTrue(any(r['kind'] == 'kernel evidence fallback' for r in qwen['reasons']))
                if qwen['variants']:
                    self.assertEqual(qwen['variants'][0]['completeness']['source_membership_complete'], data['source_membership_complete'])

    def test_source_kinds_and_namespaces_do_not_alias(self):
        import kernel_reports as kr
        with tempfile.TemporaryDirectory() as temporary:
            path, original = self.kernel_fixture(Path(temporary), f'runtime-host@{VERSION}')
            for kind, owner in [('Project', 'another-publisher'), ('Toolkit', None), ('System', None)]:
                data = deepcopy(original)
                file_key = next(k for k, f in data['files'].items() if f['identity']['path'] == HEADER)
                file = data['files'].pop(file_key)
                file['identity'].update(kind=kind, owner=owner)
                new_key = kr.file_id(file)
                data['files'][new_key] = file
                for member in data['cubins'][0]['membership']:
                    if member['file'] == file_key:
                        member['file'] = new_key
                write_report(path, data)
                report = self.report(HEADER, kernel_reports=[path])
                unrelated = next(c for c in self.consumers(report) if c['identity']['package'].startswith('runtime-host@'))
                self.assertEqual(unrelated['affected_sources'], [])
                self.assertEqual(unrelated['affected_cubins'], [])

    def test_qwen_frontend_stays_local_and_selects_real_bundle_consumer(self):
        report = self.report('models/qwen38/compiler/src/frontend-multimodal.json')
        self.assertEqual(self.models(report), {'qwen38'})
        self.assertTrue({'qwen38:frontend', 'qwen38:package', 'qwen38:gate:teacher', 'qwen38:gate:api'} <= self.ids(report))
        self.assertIn('bundle-qwen38', report['artifacts']['packages'])
        self.assertEqual(report['release_plans'][0]['cohort'], 'full')
        self.assertIn('soak', report['release_plans'][0]['description']['selected_checks'])
        self.assertIn('qwen38:gate:qwen38_capture', self.ids(report))
        self.assertIn('qwen38:gate:media_preprocessing', self.ids(report))
        self.assertIn('TOKENIZER_DIR', ' '.join(self.item(report, 'qwen38:frontend')['prerequisites']))
        self.assertFalse(report['qualified'])

    def test_qwen_local_cuda_includes_lab_locks_and_real_oracle(self):
        report = self.report('models/qwen38/bundle/cuda/qwen38_bundle.cuh')
        self.assertEqual(self.models(report), {'qwen38'})
        self.assertIn('kernel-lab', report['artifacts']['packages'])
        check = self.item(report, 'qwen38:kernel_oracle')
        self.assertEqual({l['path'] for l in check['locks']}, {'models/qwen38/bundle/kernels.lock', 'crates/kernel-lab/kernels.lock'})
        self.assertTrue(all(l['identities'] and l['sha256'] for l in check['locks']))
        self.assertIn('qwen38_bundle_rows', str(check['commands']))
        self.assertEqual(check['kind'], 'gpu_correctness')
        self.assertIn('qwen38:fixed_batch', self.ids(report))
        self.assertIn('tolerances', check['required_identity']['dimensions'])
        self.assertNotIn('checkpoint', check['required_identity']['dimensions'])
        fixed = self.item(report, 'qwen38:fixed_batch')['required_identity']
        self.assertEqual(fixed['arms'], ['candidate', 'baseline'])
        self.assertIn('checkpoint', fixed['dimensions'])

    def test_mimo_program_and_ordinary_docs_keep_model_scope(self):
        alone = self.report('models/mimo26/bundle/src/program.rs')
        mixed = self.report('models/mimo26/bundle/src/program.rs', 'docs/serving-mimo26.md')
        self.assertEqual(self.models(alone), {'mimo26'})
        self.assertEqual(self.ids(alone), self.ids(mixed))
        self.assertEqual(alone['counts']['profiles'], mixed['counts']['profiles'])
        self.assertTrue({'mimo26:journal', 'mimo26:fixed_batch', 'mimo26:quality', 'mimo26:lifecycle'} <= self.ids(alone))
        self.assertNotIn('kernel_oracle', ' '.join(self.ids(alone)))

    def test_docs_only_and_empty_diff_keep_the_standing_cpu_check(self):
        for paths in [(), ('docs/serving-mimo26.md',), ('README.md',)]:
            report = self.report(*paths)
            self.assertEqual(self.ids(report), {'shared:cpu'})
            self.assertEqual(report['profiles'], [])
            self.assertEqual(report['obligations'][0]['evidence']['execution'], 'not_run')
        # A document no Rust source names keeps every package in scope; a root
        # file Rust sources name selects their packages and dependents
        # (docs/testing.md, Affected checks): runtime-host's checkpoint test
        # and MiMo's compiler catalog both write a README.md of their own.
        self.assertEqual(self.report('docs/serving-mimo26.md')['counts']['packages'], len(self.metadata['packages']))
        readme = self.report('README.md')['artifacts']['packages']
        self.assertTrue({'runtime-host', 'model-mimo26'} <= set(readme), readme)
        self.assertNotIn('kernels-sm120', readme)

    def test_shared_cuda_keeps_both_bundles_without_a_cargo_edge(self):
        metadata = deepcopy(self.metadata)
        for node in metadata['resolve']['nodes']:
            if node['id'] == 'bundle-qwen38':
                node['deps'] = []; node['dependencies'] = []
        report = validation.build_report(metadata, ['crates/kernels-sm120/cuda/common/ptx.cuh'])
        self.assertEqual(self.models(report), MODELS)
        self.assertTrue({'qwen38:kernel_oracle', 'mimo26:kernel_oracle'} <= self.ids(report))
        self.assertIn('bundle-qwen38', report['artifacts']['packages'])
        self.assertIn('membership', report['artifacts']['completeness']['cuda_membership'])

    def test_host_lifecycle_requires_observed_host_not_model_fake_hosts(self):
        report = self.report('crates/runtime-host/src/lib.rs')
        self.assertEqual(self.models(report), MODELS)
        self.assertTrue({'shared:host', 'shared:installed_bundle', 'qwen38:lifecycle', 'mimo26:fixed_batch'} <= self.ids(report))
        self.assertFalse(any(identifier.endswith(':journal') for identifier in self.ids(report)))
        installed = self.item(report, 'shared:installed_bundle')
        self.assertIn('crates/runtime-host/src/lib.rs', installed['observation']['inputs'])
        self.assertTrue(installed['artifacts'])
        self.assertIsNone(self.item(report, 'qwen38:prefix')['coverage_gap'])
        self.assertIsNone(self.item(report, 'qwen38:speculation')['coverage_gap'])
        self.assertTrue({'qwen38:gate:prefix_exact', 'qwen38:gate:spec_verify',
                         'qwen38:gate:verify_exact'} <= self.ids(report))

    def test_cuda_host_uses_feature_bearing_driver_commands(self):
        report = self.report('crates/runtime-cuda/src/lib.rs')
        check = self.item(report, 'shared:host_cuda_fake')
        self.assertEqual(check['commands'], [['cargo', 'test', '-p', 'runtime-host', '--features', 'cuda-fake-driver-tests', '--lib', '--locked', '--offline']])
        self.assertEqual(check['observation']['features'], ['cuda-fake-driver-tests'])
        self.assertIn('runtime-host', report['artifacts']['packages'])
        self.assertIn('--test-threads=1', self.item(report, 'shared:cuda_fake')['commands'][0])
        self.assertIn('shared:host', self.ids(report))
        self.assertTrue(any(reason['class'] == 'cargo_consumer' for reason in self.item(report, 'shared:host')['reasons']))

    def test_abi_and_bundle_sdk_journals_observe_the_contract(self):
        for path in ('include/chop_bundle_v1.h', 'crates/runtime-bundle-sdk/src/lib.rs', 'crates/bundle-runtime/src/lib.rs'):
            report = self.report(path)
            self.assertTrue({'shared:abi', 'shared:boundaries', 'shared:installed_bundle', 'qwen38:journal', 'mimo26:journal'} <= self.ids(report))
        self.assertIn('crates/runtime-bundle-sdk/src/lib.rs', self.item(self.report('crates/runtime-bundle-sdk/src/lib.rs'), 'qwen38:journal')['observation']['inputs'])

    def test_packing_recipe_and_shared_transforms_select_numerical_evidence(self):
        local = self.report('models/mimo26/compiler/src/materialize.rs')
        self.assertEqual(self.models(local), {'mimo26'})
        self.assertTrue({'mimo26:package', 'mimo26:fixed_batch', 'mimo26:quality'} <= self.ids(local))
        shared = self.report('crates/kernel-core/src/transform/fragment.rs')
        self.assertEqual(self.models(shared), MODELS)
        self.assertIn('qwen38:package', self.ids(shared))

    def test_encoder_execution_keeps_frontend_and_state_contracts(self):
        for model, source in [('qwen38', 'encoder/run.rs'), ('mimo26', 'encoders/arena.rs')]:
            with self.subTest(model=model):
                report = self.report(f'models/{model}/bundle/src/{source}')
                self.assertTrue({f'{model}:{check}' for check in
                                 ('frontend', 'package', 'journal', 'fixed_batch', 'quality', 'service', 'lifecycle')}
                                <= self.ids(report))

    def test_bundle_and_layout_inputs_retain_compiler_contracts(self):
        for model, sources in [('qwen38', ('contract.rs', 'views.rs', 'scratch.rs')),
                               ('mimo26', ('contract.rs', 'topology.rs', 'index.rs'))]:
            for source in sources:
                with self.subTest(model=model, source=source):
                    report = self.report(f'models/{model}/bundle/src/{source}')
                    self.assertTrue({f'{model}:package', f'{model}:journal', f'{model}:fixed_batch'} <= self.ids(report))
        self.assertIn('mimo26:package', self.ids(self.report('models/mimo26/contract/src/tp.rs')))

    def test_canonical_package_schema_and_digest_consumers(self):
        schema = self.report('crates/runtime-bundle/src/lib.rs')
        self.assertTrue({'shared:abi', 'shared:boundaries', 'shared:installed_bundle',
                         'qwen38:package', 'mimo26:package', 'qwen38:journal', 'mimo26:journal'} <= self.ids(schema))
        digest = self.report('crates/runtime-weights/src/lib.rs')
        self.assertTrue({'shared:host', 'shared:installed_bundle'} <= self.ids(digest))

    def test_cpu_contracts_cover_independently_derived_cargo_consumers(self):
        members = set(self.metadata['workspace_members'])
        packages = {p['id']: p for p in self.metadata['packages']}
        reverse = {key: set() for key in members}
        for node in self.metadata['resolve']['nodes']:
            for dependency in node['deps']:
                if dependency['pkg'] in reverse:
                    reverse[dependency['pkg']].add(node['id'])
        checks = {'shared:' + key: row for key, row in self.catalog.data['checks'].items()}
        for model, row in self.catalog.models.items():
            parameters = {role: package.name for role, package in row['inventory'].packages.items()}
            for key, check in self.catalog.data['model_checks'].items():
                checks[f'{model}:{key}'] = {**check, 'commands': [
                    [arg.format_map(parameters) if arg in ('{bundle}', '{compiler}', '{contract}') else arg for arg in command]
                    for command in check['commands']]}
        targets = {}
        for key, check in checks.items():
            if check['kind'] == 'cpu_contract':
                targets[key] = {command[i + 1] for command in check['commands'] if command[:2] == ['cargo', 'test']
                                for i, arg in enumerate(command[:-1]) if arg in ('-p', '--package')}
        for seed in sorted(members):
            with self.subTest(package=packages[seed]['name']):
                reached, pending = set(), [seed]
                while pending:
                    current = pending.pop()
                    if current not in reached:
                        reached.add(current)
                        pending.extend(reverse.get(current, ()))
                names = {packages[key]['name'] for key in reached & members}
                source = next(iter(sorted((Path(packages[seed]['manifest_path']).parent / 'src').rglob('*.rs'))))
                report = self.report(source.relative_to(ROOT).as_posix())
                expected = {key for key, package_targets in targets.items() if package_targets & names}
                self.assertFalse(expected - self.ids(report), sorted(expected - self.ids(report)))

    def test_every_broad_fallback_covers_the_declared_catalog(self):
        expected = {'shared:' + key for key in self.catalog.data['checks']}
        expected |= {f'{model}:{key}' for model in self.catalog.models for key in self.catalog.data['model_checks']}
        expected |= {f'{gate}:gate:{check["id"]}' for model, row in self.catalog.models.items()
                     for gate, declared in validation.gates(model, row) for check in declared['gate']['checks']}
        cases = [('all', (), {'all_checks': True})]
        cases += [(path, (path,), {}) for path in ('unknown-input', 'crates/removed/src/lib.rs', 'Cargo.lock',
                  'rust-toolchain.toml', 'models/qwen38/profiles.json', 'models/mimo26/profiles.json',
                  'tools/qual/validation_evidence.py', 'tools/qual/test_release_gate.py')]
        for broken in ('catalog', 'graph'):
            copy = deepcopy(getattr(self, broken))
            copy.issues.append('unavailable source or model metadata')
            cases.append((broken, ('models/qwen38/compiler/src/frontend.rs',), {broken: copy}))
        for label, paths, options in cases:
            with self.subTest(case=label):
                report = validation.build_report(self.metadata, paths, **{'graph': self.graph, 'catalog': self.catalog, **options})
                self.assertFalse(expected - self.ids(report), sorted(expected - self.ids(report)))

    def test_broad_class_completeness_is_validated_when_loading(self):
        for section in ('checks', 'model_checks'):
            with self.subTest(section=section), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                (root / 'ci').mkdir()
                (root / 'models').mkdir()
                # A newly declared check must not silently escape either fallback.
                addition = f'\n[{section}.new_contract]\nclaim="new contract"\nkind="cpu_contract"\ncommands=[]\nobserves=[]\ntargets=[]\nprerequisites=[]\nstanding=false\n'
                (root / 'ci/validation.toml').write_text((ROOT / 'ci/validation.toml').read_text() + addition)
                with patch.object(validation.validation_inventory, 'read', return_value=([], [])):
                    with self.assertRaisesRegex(ValueError, 'incomplete.*(unknown|validation_policy)'):
                        validation.Catalog(root, self.metadata)

    def test_requested_gate_check_outside_cohort_remains_unresolved(self):
        for absent in ('check', 'config'):
            with self.subTest(absent=absent):
                catalog = deepcopy(self.catalog)
                if absent == 'check':
                    catalog.models['qwen38']['gate']['profiles']['full']['checks'].remove('qwen38_layers')
                else:
                    catalog.models['qwen38']['gate'] = {}
                    catalog.issues.append('qwen38: missing gate configuration')
                report = validation.build_report(self.metadata, ['tools/qual/qwen38_layers.py'], graph=self.graph, catalog=catalog)
                self.assertIn('qwen38:gate:qwen38_layers', self.ids(report))
                item = self.item(report, 'qwen38:gate:qwen38_layers')
                self.assertTrue(item['coverage_gap'])
                self.assertEqual(item['commands'], [])
                self.assertEqual(item['evidence']['execution'], 'not_run')
                self.assertTrue(any('qwen38_layers' in gap and 'cohort' in gap for gap in report['gaps']))

    def test_cpu_consumers_accept_long_package_arguments(self):
        catalog = deepcopy(self.catalog)
        commands = catalog.data['checks']['host']['commands']
        commands[:] = [[('--package' if arg == '-p' else arg) for arg in command] for command in commands]
        report = validation.build_report(self.metadata, ['crates/runtime-weights/src/lib.rs'], graph=self.graph, catalog=catalog)
        self.assertIn('shared:host', self.ids(report))

    def test_kernel_lab_family_locks_select_the_shared_oracle_and_report_their_identities(self):
        for path in ('crates/kernel-lab/kernels.lock', 'crates/kernel-lab/kernels.lock.d/fixture.lock'):
            with self.subTest(path=path):
                oracle = self.item(self.report(path), 'shared:kernel_oracle')
                self.assertEqual(oracle['evidence']['selection'], 'selected')
                self.assertEqual(oracle['observation']['inputs'], [path])
                self.assertTrue({'crates/kernel-lab/kernels.lock', 'crates/kernel-lab/kernels.lock.d'}
                                <= {lock['path'] for lock in oracle['locks']})
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root / 'crates/kernel-lab/kernels.lock.d'
            directory.mkdir(parents=True)
            record = {'name': 'fixture_entry', 'cubin_sha256': '00' * 32}
            (directory / 'fixture.lock').write_text(json.dumps([record]))
            (directory / 'fixture.provenance.json').write_text('{}')
            locks = validation.lock_identities(root, self.catalog.data['checks']['kernel_oracle']['locks'])
            family = next(lock for lock in locks if lock['path'] == 'crates/kernel-lab/kernels.lock.d')
            self.assertEqual([(f['family'], f['path'], f['identities']) for f in family['families']],
                             [('fixture', 'crates/kernel-lab/kernels.lock.d/fixture.lock', [record])])

    def test_shared_sampling_and_external_coverage_remain_explicit(self):
        for path in ('crates/kernels-sm120/cuda/sampling/filtered.cuh', 'unknown-input'):
            with self.subTest(path=path):
                report = self.report(path)
                locks = {lock['path'] for item in report['obligations'] for lock in item.get('locks', [])}
                self.assertIn('crates/runtime-kernels/kernels.lock', locks)
                shared = self.item(report, 'shared:kernel_oracle')
                self.assertIn(['cargo', 'test', '-p', 'kernel-lab', '--locked', '--offline'], shared['commands'])
                self.assertEqual(shared['scope'], 'kernel_operation')
                self.assertTrue(shared['coverage_gap'])
                self.assertIn('required assets', ' '.join(shared['prerequisites']))
                self.assertTrue(any('external' in gap.lower() and 'admission' in gap for gap in report['gaps']))
        external = self.report('examples/external-model/model.toml')
        self.assertTrue(any('external' in gap.lower() and 'admission' in gap for gap in external['gaps']))

    def test_current_membership_preserves_shared_oracle_and_cargo_obligations(self):
        with tempfile.TemporaryDirectory() as temporary:
            path, data = self.kernel_fixture(Path(temporary), f'runtime-kernels@{VERSION}')
            lock = Path(temporary) / 'crates/runtime-kernels/kernels.lock'
            lock.parent.mkdir(parents=True)
            Path(data['lock_path']).rename(lock)
            data['lock_path'] = str(lock)
            write_report(path, data)
            for paths in [(HEADER,), (HEADER, 'docs/testing.md')]:
                report = self.report(*paths, kernel_reports=[path])
                oracle = self.item(report, 'shared:kernel_oracle')
                observed = next(row for row in oracle['locks'] if row['path'] == 'crates/runtime-kernels/kernels.lock')
                self.assertEqual(observed['membership'], 'current')
                self.assertTrue(observed['identities'])
                self.assertEqual(oracle['evidence']['execution'], 'not_run')
                self.assertTrue(oracle['coverage_gap'])
                self.assertTrue(any(reason['class'] == 'cargo_consumer'
                                    for item in report['obligations'] for reason in item['reasons']))
                self.assertFalse(report['qualified'])

    def test_directory_lock_reports_render_and_retain_consumers(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path, data = self.kernel_fixture(root / 'bundle', f'bundle-qwen38@{VERSION}')
            lock, provenance = to_directory(path, data)
            family = data['lock_families'][0]
            # Provenance-only drift, even with the report-only file absent, stays current.
            family['provenance_changed'] = ['fixture_v1']
            write_report(path, data)
            provenance.unlink()
            report = self.report(HEADER, kernel_reports=[path])
            qwen = next(c for c in self.consumers(report) if c['identity']['package'].startswith('bundle-qwen38@'))
            self.assertTrue(qwen['current_evidence'])
            self.assertTrue(qwen['selected'])
            described = qwen['variants'][0]['lock']
            self.assertIsNone(described['sha256'])
            self.assertEqual([(f['family'], f['path'], f['sha256'], f['provenance_changed']) for f in described['families']],
                             [('fixture', str(lock), family['lock_sha256'], ['fixture_v1'])])
            text = validation.render_text(report)
            self.assertIn('  Lock: ' + data['lock_path'] + ' directory; current\n', text)
            self.assertIn('    Family: fixture ' + str(lock) + ' ' + family['lock_sha256']
                          + '; provenance differs: fixture_v1\n', text)
            # Catalogs may name a lock directory or one family's lock file.
            locks = validation.lock_identities(root, ['bundle/kernels.lock.d', 'bundle/kernels.lock.d/fixture.lock',
                                                      'bundle/absent.lock.d'], self.consumers(report))
            record = {k: v for k, v in data['cubins'][0]['identity'].items()
                      if k in ('name', 'nvcc_sha256', 'cubin_sha256', 'flags', 'defines')}
            self.assertEqual(locks[0], {'path': 'bundle/kernels.lock.d', 'membership': 'current', 'families': [
                {'family': 'fixture', 'path': 'bundle/kernels.lock.d/fixture.lock',
                 'sha256': family['lock_sha256'], 'identities': [record]}]})
            self.assertEqual(locks[1], {'path': 'bundle/kernels.lock.d/fixture.lock', 'membership': 'current',
                                        'sha256': family['lock_sha256'], 'identities': [record]})
            self.assertTrue(locks[2]['missing'])
            # A missing or changed family keeps the consumer selected without current evidence.
            original = lock.read_bytes()
            for content in (None, b'[]'):
                lock.unlink()
                if content is not None:
                    lock.write_bytes(content)
                report = self.report(HEADER, kernel_reports=[path])
                qwen = next(c for c in self.consumers(report) if c['identity']['package'].startswith('bundle-qwen38@'))
                self.assertFalse(qwen['current_evidence'])
                self.assertTrue(qwen['selected'])
                self.assertIn('  Lock: ' + data['lock_path'] + ' directory; stale\n', validation.render_text(report))
                lock.unlink(missing_ok=True)
                lock.write_bytes(original)
            # Duplicate family coverage is an invalid report, kept as a gap.
            data['lock_families'].append(family | {'family': 'second'})
            write_report(path, data)
            report = self.report(HEADER, kernel_reports=[path])
            self.assertTrue(any('duplicate lock family coverage' in gap for gap in report['gaps']))
            self.assertIn('duplicate lock family coverage', validation.render_text(report))

    def runtime_host_compilers(self):
        """The models whose compiler links runtime-host, and through it the
        evaluation kinds every gate reader feeds."""
        nodes = {node['id']: node for node in self.metadata['resolve']['nodes']}
        return {model.name for model in inventory.load(ROOT)
                if 'runtime-host' in nodes[model.packages['compiler'].name]['dependencies']}

    def test_tool_reader_and_cross_model_fixture_consumers(self):
        tool = self.report('tools/qual/qwen38_layers.py')
        # The executable reader keeps its gate scope, while the package fallback
        # conservatively retains the CPU contracts of every compiler that
        # links its Cargo consumer.
        self.assertLessEqual({'glm5next', 'mimo26', 'qwen38'}, self.runtime_host_compilers())
        self.assertEqual(self.models(tool), self.runtime_host_compilers() | {'qwen38'})
        self.assertIn('mimo26:package', self.ids(tool))
        self.assertNotIn('mimo26:quality', self.ids(tool))
        self.assertIn('qwen38:gate:qwen38_layers', self.ids(tool))
        self.assertNotIn('qwen38:gate:api', self.ids(tool))
        # Each model's HF tokenizer reference selects its own frontend gate.
        for path, model, other in [('fixtures/mimo26/frontend/tokenizer-golden.json', 'mimo26', 'qwen38'),
                                   ('fixtures/qwen38/frontend/tokenizer-golden.json', 'qwen38', 'mimo26')]:
            with self.subTest(path=path):
                samples = self.ids(self.report(path))
                import declaration
                def frontend_check(name):
                    config = declaration.load(declaration.path(name))
                    return next(check['id'] for check in config['checks'] if check['module'] == 'frontend_goldens')
                self.assertIn(f'{model}:gate:{frontend_check(model)}', samples)
                self.assertNotIn(f'{other}:gate:{frontend_check(other)}', samples)
        # Qwen3.8's gate still reads MiMo's media cases, so both gates consume them.
        shared = self.report('fixtures/mimo26/capture/multimodal_cases.py')
        self.assertEqual(self.models(shared), MODELS)
        self.assertTrue({'mimo26:gate:api', 'qwen38:gate:api'} <= self.ids(shared))
        policy = self.report('docs/numerics.md')
        self.assertEqual(self.models(policy), MODELS)

    def test_hidden_inputs_reach_their_executing_packages(self):
        for path, packages in [
            ('fixtures/tokenizer-whitespace.json', {'model-qwen38', 'model-mimo26'}),
            ('crates/runtime-host/src/lib.rs', {'runtime-conformance'}),
            ('fixtures/mimo26/teacher-exact-reference.json', {'kernel-lab'}),
        ]:
            with self.subTest(path=path):
                report = self.report(path)
                self.assertTrue(packages <= set(report['artifacts']['packages']))
                for package in packages:
                    self.assertTrue(any(reason['input'] == path for reason in report['artifacts']['reasons'][package]))
        reference = self.report('models/mimo26/contract/src/tp.rs')
        reason = reference['artifacts']['reasons']['kernel-lab']
        self.assertTrue(any(r['conservative'] for r in reason))

    def test_generated_cuda_and_manifest_relative_inputs_have_observed_readers(self):
        self.assertIn(('crates/kernel-lab/build.rs', 'generated CUDA header'),
                      self.graph.reads['models/qwen38/bundle/cuda/qwen38_speculation.cuh'])
        self.assertIn(('crates/runtime-bundle-abi/tests/abi_contract.rs', 'manifest-relative input'),
                      self.graph.reads['include/chop_bundle_v1.h'])

    def test_unknown_deleted_and_missing_metadata_widen(self):
        for path in ['new-unowned-file', 'crates/removed/src/lib.rs']:
            report = self.report(path)
            self.assertEqual(self.models(report), MODELS)
            self.assertTrue(any(c['class'] == 'unknown' for c in report['changes']))
        broken = deepcopy(self.catalog)
        broken.models['third'] = {'inventory': None, 'data': {}, 'gate': {}, 'profiles': {}, 'readers': {}}
        broken.issues.append('third: missing inventory')
        report = validation.build_report(self.metadata, ['models/qwen38/compiler/src/frontend.rs'], graph=self.graph, catalog=broken)
        self.assertIn('third:unresolved', self.ids(report))
        self.assertTrue(report['gaps'])
        self.assertEqual(self.models(report), MODELS)

    def test_a_configuration_gate_is_selected_like_a_base_gate(self):
        """glm5next's k5 configuration (qual.json) is a gate of its own: its
        readers and change classes select its checks, under --model
        glm5next.k5. Every declared configuration is a gate."""
        raw = json.loads((ROOT / 'models/glm5next/qual.json').read_text())
        for name in ('glm5next', 'qwen38', 'mimo26'):
            declared = json.loads((ROOT / f'models/{name}/qual.json').read_text())['configurations']
            self.assertEqual(set(self.catalog.models[name]['variants']), set(declared))
        self.assertIn('k5', self.catalog.models['glm5next']['variants'])
        report = self.report('models/glm5next/compiler/src/frontend.rs')
        self.assertIn('glm5next.k5:gate:teacher', self.ids(report))
        plan = next(plan for plan in report['release_plans'] if plan['model'] == 'glm5next.k5')
        self.assertEqual(plan['command'][plan['command'].index('--model') + 1], 'glm5next.k5')
        self.assertIn('teacher', plan['selected_checks'])
        frontend = self.ids(report)
        _, classes = validation.configuration_gate(raw, 'k5')
        self.assertTrue(classes['frontend'])
        self.assertTrue({f'glm5next.k5:gate:{check}' for check in classes['frontend']} <= frontend)
        self.assertTrue({'glm5next:gate:teacher', 'glm5next:gate:api'} <= frontend)
        self.assertNotIn('glm5next.k5:gate:long_context', self.ids(self.report('unknown-input')))

    def test_an_external_variant_is_never_a_gate(self):
        """An external variant under CHOPINFER_QUAL_VARIANTS (or --variant-dir)
        qualifies another checkpoint outside the tree: the catalog never
        reads its root, so no change selects it."""
        sys.path.insert(0, str(ROOT / 'tools/qual'))
        from test_variant import synthetic_variant
        with tempfile.TemporaryDirectory() as temporary:
            model, _ = synthetic_variant(Path(temporary), adopt=False)
            with patch.dict(os.environ, {'CHOPINFER_QUAL_VARIANTS': temporary}):
                catalog = validation.Catalog(ROOT, self.metadata)
                report = validation.build_report(self.metadata, ['unknown-input'], graph=self.graph,
                                                 catalog=catalog)
        self.assertEqual(catalog.models[model]['variants'], self.catalog.models[model]['variants'])
        self.assertNotIn('requant', catalog.models[model]['variants'])
        self.assertFalse([item for item in self.ids(report) if item.startswith(f'{model}.requant:')])

    def test_a_variant_release_run_answers_the_variant_obligations(self):
        """A release run of glm5next.k5 (its summary names model glm5next
        and the variant in its stored policy) is matched to the
        glm5next.k5:gate obligations it ran or recorded as not run, never
        to the base gate's."""
        config, _ = validation.configuration_gate(json.loads((ROOT / 'models/glm5next/qual.json').read_text()), 'k5')
        with tempfile.TemporaryDirectory() as temporary:
            summary = Path(temporary) / 'summary.json'
            summary.write_text(json.dumps({
                'schema': 'chopinfer-release-receipt/v1', 'model': 'glm5next', 'profile': 'full',
                'config': config,
                'config_sha256': hashlib.sha256((json.dumps(config, indent=2) + '\n').encode()).hexdigest(),
                'checks': [{'id': 'teacher', 'passed': False}], 'not_run': ['chat']}))
            report = self.report('models/glm5next/compiler/src/frontend.rs', context={}, receipts=[summary])
        claims = {row['claim'] for row in report['supplied_receipts']}
        self.assertEqual(claims, {'glm5next.k5:gate:teacher', 'glm5next.k5:gate:chat'})
        teacher = self.item(report, 'glm5next.k5:gate:teacher')['evidence']
        self.assertEqual((teacher['execution'], teacher['result']), ('executed', 'failed'))
        self.assertFalse(teacher['satisfied'])
        self.assertEqual(self.item(report, 'glm5next.k5:gate:chat')['evidence']['execution'], 'not_run')
        self.assertNotEqual(self.item(report, 'glm5next:gate:teacher')['evidence'].get('result'), 'failed')

    def test_missing_change_class_mapping_keeps_every_gate_check(self):
        frontend = 'models/mimo26/compiler/src/frontend.rs'
        expected = {'mimo26:gate:teacher', 'mimo26:gate:api', 'mimo26:gate:structured'}
        self.assertTrue(expected <= self.ids(self.report(frontend)))
        # A model row without its change classes must widen, never drop, gate obligations.
        catalog = deepcopy(self.catalog)
        catalog.models['mimo26']['change_classes'] = {}
        report = validation.build_report(self.metadata, [frontend], graph=self.graph, catalog=catalog)
        self.assertTrue(expected <= self.ids(report))
        self.assertTrue(any('mimo26' in gap and 'change class' in gap for gap in report['gaps']))

    def test_declaration_without_change_classes_is_a_catalog_issue(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shutil.copytree(ROOT / 'models/mimo26', root / 'models/mimo26',
                            ignore=shutil.ignore_patterns('bundle', 'compiler', 'contract'))
            for file in ['ci/validation.toml', 'tests/model_inventory.py', 'Cargo.toml']:
                (root / file).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT / file, root / file)
            for role in ['bundle', 'compiler']:
                (root / 'models/mimo26' / role).mkdir()
                (root / 'models/mimo26' / role / 'Cargo.toml').write_text(f'[package]\nname="mimo26-{role}"\n')
            (root / 'Cargo.toml').write_text('[workspace]\nmembers=["models/*/bundle", "models/*/compiler"]\n')
            (root / 'models/mimo26/model.toml').write_text(
                'schema="chopinfer-model-inventory-v1"\n[bundle]\npackage="mimo26-bundle"\n[compiler]\npackage="mimo26-compiler"\n')
            metadata = metadata_fixture(root)
            self.assertEqual(validation.Catalog(root, metadata).issues, [])
            path = root / 'models/mimo26/qual.json'
            declared = json.loads(path.read_text())
            del declared['change_classes']
            path.write_text(json.dumps(declared))
            issues = validation.Catalog(root, metadata).issues
            self.assertEqual(len(issues), 1)
            self.assertIn('change_classes: is required', issues[0])

    def test_third_model_catalog_requires_no_selector_conditionals(self):
        catalog = deepcopy(self.catalog)
        third = deepcopy(catalog.models['qwen38'])
        third['profiles'] = {'single': next(iter(third['profiles'].values()))}
        catalog.models['third'] = third
        report = validation.build_report(self.metadata, ['models/third/bundle/src/program.rs'], graph=self.graph, catalog=catalog)
        # This synthetic owner is not in Cargo metadata, so CPU consumers widen.
        self.assertEqual(self.models(report), {'third', *MODELS})
        self.assertIn('third:journal', self.ids(report))
        self.assertEqual(self.item(report, 'third:journal')['profiles'], ['single'])

    def test_new_inventory_is_discovered_through_the_sdk_loader(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for file in ['ci/validation.toml', 'tests/model_inventory.py']:
                (root / file).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT / file, root / file)
            (root / 'Cargo.toml').write_text('[workspace]\nmembers=["models/*/bundle", "models/*/compiler"]\n')
            model = root / 'models/third'
            for role in ['bundle', 'compiler']:
                (model / role).mkdir(parents=True)
                (model / role / 'Cargo.toml').write_text(f'[package]\nname="third-{role}"\n')
            (model / 'model.toml').write_text('schema="chopinfer-model-inventory-v1"\n[bundle]\npackage="third-bundle"\n[compiler]\npackage="third-compiler"\n')
            source = (ROOT / 'models/qwen38/validation.toml').read_text().replace('qwen38', 'third')
            (model / 'validation.toml').write_text(source)
            profile = {'schema': 'chopinfer-serve-profiles-v2', 'model': 'third', 'profiles': {'single': {'status': 'experimental'}}}
            (model / 'profiles.json').write_text(json.dumps(profile))
            declared = json.loads((ROOT / 'models/qwen38/qual.json').read_text())
            declared['model'] = 'third'
            (model / 'qual.json').write_text(json.dumps(declared))
            metadata = metadata_fixture(root)
            catalog = validation.Catalog(root, metadata)
            self.assertEqual(catalog.issues, [])
            report = validation.build_report(metadata, ['models/third/bundle/src/program.rs'], root=root,
                                             graph=change_scope.Graph(metadata, edges=[], scan=False), catalog=catalog)
            self.assertEqual(self.models(report), {'third'})
            self.assertEqual(self.item(report, 'third:journal')['commands'][0][3], 'third-bundle')

    def test_removed_profiles_remain_visible_from_base_declarations(self):
        catalog = deepcopy(self.catalog)
        catalog.retain_previous({'models/qwen38/profiles.json': json.dumps({'profiles': {'old-profile': {'status': 'experimental'}}})})
        report = validation.build_report(self.metadata, ['models/qwen38/profiles.json'], graph=self.graph, catalog=catalog)
        previous = next(p for p in report['profiles'] if p['profile'] == 'old-profile')
        self.assertTrue(previous['removed'])
        self.assertIn('unresolved', previous['qualification'])
        self.assertTrue(any('old-profile' in gap and 'refusal' in gap for gap in report['gaps']))
        self.assertIn('old-profile', self.item(report, 'qwen38:journal')['profiles'])
        self.assertIn('old-profile', self.item(report, 'qwen38:gate:api')['profiles'])

    def test_unproven_source_membership_never_narrows_shared_selection(self):
        plain = self.report('crates/kernels-sm120/cuda/common/ptx.cuh')
        supplied = self.report('crates/kernels-sm120/cuda/common/ptx.cuh', context={'cuda_membership': {'complete': True, 'models': ['qwen38']}})
        self.assertEqual(self.ids(plain), self.ids(supplied))
        self.assertEqual(self.models(supplied), MODELS)
        self.assertTrue(any('membership' in gap for gap in supplied['gaps']))

    def test_missing_receipt_cannot_report_execution_or_success(self):
        with tempfile.TemporaryDirectory() as temporary:
            report = self.report('models/qwen38/compiler/src/frontend.rs', context={}, receipts=[Path(temporary) / 'absent.json'])
        self.assertTrue(report['supplied_receipts'][0]['issues'])
        self.assertTrue(all(o['evidence']['applicability'] == 'missing' for o in report['obligations']))
        self.assertFalse(report['qualified'])

    def test_all_is_a_conservative_superset_and_deterministic(self):
        local = self.report('models/mimo26/bundle/src/program.rs')
        broad = self.report('models/mimo26/bundle/src/program.rs', all_checks=True)
        self.assertTrue(self.ids(local) <= self.ids(broad))
        self.assertEqual(self.models(broad), MODELS)
        self.assertEqual(broad, self.report('models/mimo26/bundle/src/program.rs', all_checks=True))
        self.assertEqual(self.report('README.md', 'docs/testing.md'), self.report('docs/testing.md', 'README.md'))

    def test_contradictory_capabilities_cannot_exclude_profiles(self):
        catalog = deepcopy(self.catalog)
        catalog.models['qwen38']['data']['capabilities'] = []
        report = validation.build_report(self.metadata, ['crates/runtime-host/src/lib.rs'], graph=self.graph, catalog=catalog)
        self.assertIn('qwen38:speculation', self.ids(report))
        self.assertTrue(any('contradictory' in gap for gap in report['gaps']))

    def test_missing_or_unverified_capability_absence_never_excludes(self):
        catalog = deepcopy(self.catalog)
        row = catalog.models['qwen38']
        row['data']['capabilities'] = []
        for settings in row['profiles'].values():
            settings['prefix_cache'] = False
        args = dict(graph=self.graph, catalog=catalog)
        report = validation.build_report(self.metadata, ['crates/runtime-host/src/lib.rs'], **args)
        self.assertEqual(self.item(report, 'qwen38:prefix')['evidence']['selection'], 'selected')
        self.assertTrue(any('absence' in gap for gap in report['gaps']))
        row['data'].pop('capabilities')
        report = validation.build_report(self.metadata, ['crates/runtime-host/src/lib.rs'], **args)
        self.assertEqual(self.item(report, 'qwen38:prefix')['evidence']['selection'], 'selected')
        catalog.retain_previous({'models/qwen38/profiles.json': json.dumps({'profiles': {
            'removed-enabled': {'prefix_cache': True, 'speculation': {'tokens': 4}}}})})
        report = validation.build_report(self.metadata, ['crates/runtime-host/src/lib.rs'], **args)
        self.assertIn('removed-enabled', self.item(report, 'qwen38:prefix')['profiles'])
        self.assertEqual(self.item(report, 'qwen38:prefix')['evidence']['selection'], 'selected')
        self.assertTrue(any('contradictory' in gap for gap in report['gaps']))
        report = validation.build_report(self.metadata, ['crates/runtime-host/src/lib.rs'],
                    context={'package_capabilities': {'qwen38': ['prefix']}}, **args)
        self.assertEqual(self.item(report, 'qwen38:prefix')['evidence']['selection'], 'selected')

    def test_output_and_cli_are_read_only_and_external(self):
        inputs = [HEADER]
        temporary_reports = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_reports.cleanup)
        kernel_path, _ = self.kernel_fixture(Path(temporary_reports.name), f'bundle-qwen38@{VERSION}')
        def output(*cmd):
            if cmd[0] == 'cargo':
                self.assertEqual(cmd[1], 'metadata')
                return json.dumps(self.metadata).encode()
            if cmd[:2] == ('git', 'rev-parse'):
                return b'base\n'
            if cmd[:2] == ('git', 'diff'):
                return '\0'.join(inputs).encode()
            if cmd[:2] == ('git', 'show'):
                return b''
            if cmd[:2] == ('git', 'ls-files'):
                return b''
            raise AssertionError(cmd)
        outputs = {}
        with patch.object(quick, 'output', side_effect=output), \
             patch('subprocess.run', side_effect=AssertionError('executed command')), \
             patch.object(release_gate.Client, 'identity', side_effect=AssertionError('HTTP')), \
             patch.object(release_gate, 'run', side_effect=AssertionError('gate execution')):
            for format in ['json', 'text']:
                stream = io.StringIO()
                with contextlib.redirect_stdout(stream):
                    self.assertEqual(quick.main(['--report', '--format', format, '--kernel-reports', str(kernel_path),
                        '--kernel-reports', str(kernel_path.parent), '--previous-kernel-reports', str(kernel_path)]), 0)
                outputs[format] = stream.getvalue()
        report = json.loads(outputs['json'])
        for item in report['obligations']:
            self.assertIn(item['id'], outputs['text'])
        for consumer in self.consumers(report):
            self.assertIn(consumer['identity']['package'], outputs['text'])
            for variant in consumer['variants']:
                self.assertIn(variant['status'], outputs['text'])
        with self.assertRaises(ValueError):
            validation.write_report(report, output=ROOT / 'must-not-create.json')
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'report.json'
            validation.write_report(report, 'json', path)
            self.assertEqual(json.loads(path.read_text()), report)
            with self.assertRaises(FileExistsError):
                validation.write_report(report, 'json', path)
            (Path(temporary) / 'repo').symlink_to(ROOT, target_is_directory=True)
            with self.assertRaises(ValueError):
                validation.write_report(report, output=Path(temporary) / 'repo/blocked.json')


if __name__ == '__main__':
    unittest.main()
