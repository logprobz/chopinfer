"""External inventories cannot widen the SDK or host dependency boundary."""
import copy
import json
import os
import subprocess
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from model_sdk.graph import Graph, PolicyError, check_external, metadata, selected_graph
from model_sdk.inventory import InventoryError
from model_sdk import source, workflow, smoke


class ExternalPolicyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.sdk = self.root / 'sdk'
        self.data = {'workspace_root': str(self.root), 'workspace_members': [],
                     'packages': [], 'resolve': {'nodes': []}}
        self.add('bundle-demo', 'bundle', [('demo', 'cdylib')], member=True)
        self.add('compile-demo', 'compiler', [('compile-demo', 'bin')], member=True)
        (self.root / 'Cargo.toml').write_text('[workspace]\n')
        (self.root / 'model.toml').write_text('''schema = "chopinfer-model-inventory-v1"
[bundle]
package = "bundle-demo"
cdylib = ["demo"]
[compiler]
package = "compile-demo"
bin = ["compile-demo"]
''')

    def test_root_inventory_accepts_operational_sections(self):
        path = self.root / 'model.toml'
        (self.root / 'asset.json').write_text('{}')
        (self.root / 'bundle/kernels.lock').write_text('')
        (self.root / 'profiles.json').write_text('{}')
        with path.open('a') as output:
            output.write(f'\n[release]\nenabled = true\nbins = ["compile-demo"]\n[[release.assets]]\nsource = "asset.json"\ndestination = "models/{self.root.name}/asset.json"\n[checks]\ncompile_inputs = ["asset.json"]\n')
        self.check()

    def add(self, name, directory, targets=(), member=False, key=None):
        path = self.root / directory / 'Cargo.toml'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        key = name if key is None else key
        self.data['packages'].append({'id': key, 'name': name, 'manifest_path': str(path),
                                     'targets': [{'name': n, 'kind': [k]} for n, k in targets]})
        self.data['resolve']['nodes'].append({'id': key, 'deps': []})
        if member:
            self.data['workspace_members'].append(key)
        return key

    def edge(self, source, dest, kind=None, alias='renamed', target=None):
        node = next(n for n in self.data['resolve']['nodes'] if n['id'] == source)
        node['deps'].append({'name': alias, 'pkg': dest,
                             'dep_kinds': [{'kind': kind, 'target': target}]})

    def check(self):
        check_external(self.data, self.root / 'Cargo.toml', self.root / 'model.toml', self.sdk)

    def add_native_test_helper(self):
        self.add('native-helper', 'native-helper')
        self.add('kernel-build', 'sdk/crates/kernel-build')
        self.edge('compile-demo', 'native-helper', 'dev')
        self.edge('native-helper', 'kernel-build', 'build')

    def test_encoder_spec_is_part_of_every_role_surface(self):
        # External models declare their encoders with encoder-spec, directly
        # or through kernels-sm120's generated catalogs.
        self.add('encoder-spec', 'sdk/crates/encoder-spec')
        self.add('kernels-sm120', 'sdk/crates/kernels-sm120')
        self.edge('bundle-demo', 'kernels-sm120')
        self.edge('kernels-sm120', 'encoder-spec')
        self.edge('bundle-demo', 'encoder-spec')
        self.edge('compile-demo', 'encoder-spec')
        self.check()

    def test_projection_spec_is_part_of_every_role_surface(self):
        self.add('projection-spec', 'sdk/crates/projection-spec')
        self.edge('bundle-demo', 'projection-spec')
        self.edge('compile-demo', 'projection-spec')
        self.check()

    def test_a_test_crate_runs_kernels_on_the_harness_and_never_registers_in_kernel_lab(self):
        # A workspace member no role declares is a test crate: its dev
        # dependencies may enter the SDK through the kernel harness, the CUDA
        # driver and the host, and reach any SDK crate behind them but the
        # unsupported ones.
        self.add('demo-lab', 'lab', member=True)
        for name in ('kernel-harness', 'runtime-cuda', 'runtime-host', 'runtime-server', 'kernels-sm120'):
            self.add(name, f'sdk/crates/{name}')
        for name in ('kernel-harness', 'runtime-host', 'kernels-sm120'):
            self.edge('demo-lab', name, 'dev')
        self.edge('kernel-harness', 'runtime-cuda')
        self.edge('runtime-host', 'runtime-server')
        self.check()
        unlink = lambda source, dest: [node['deps'].remove(edge) for node in self.data['resolve']['nodes']
                                       if node['id'] == source for edge in list(node['deps']) if edge['pkg'] == dest]
        # The roles keep their own surfaces.
        self.edge('compile-demo', 'kernel-harness')
        with self.assertRaisesRegex(PolicyError, 'compile-demo links unsupported SDK package kernel-harness'):
            self.check()
        unlink('compile-demo', 'kernel-harness')
        # A test crate enters the SDK only through its surface.
        self.edge('demo-lab', 'runtime-server', 'dev')
        with self.assertRaisesRegex(PolicyError, 'tests entry point runtime-server'):
            self.check()
        unlink('demo-lab', 'runtime-server')
        self.check()
        # Never kernel-lab, directly or behind the harness.
        self.add('kernel-lab', 'sdk/crates/kernel-lab')
        self.edge('kernel-harness', 'kernel-lab')
        with self.assertRaisesRegex(PolicyError, 'demo-lab links unsupported SDK package kernel-lab'):
            self.check()

    def test_contributor_closure_checks_external_dev_build_owners(self):
        self.add_native_test_helper()
        self.check()  # The production surface check intentionally omits dev edges.
        graph = Graph(self.data)
        self.assertNotIn('native-helper', graph.closure('compile-demo'))
        with self.assertRaisesRegex(PolicyError, 'CUDA.*native-helper'):
            check_external(self.data, self.root/'Cargo.toml', self.root/'model.toml', self.sdk,
                           selected=graph, contributor_roots=('compile-demo',))

    def test_contributor_closure_checks_normal_and_transitive_build_owners(self):
        self.add('model-compiler-sdk', 'sdk/crates/model-compiler-sdk')
        self.add('runtime-kernels', 'sdk/crates/runtime-kernels')
        self.add('kernel-build', 'sdk/crates/kernel-build')
        self.edge('compile-demo', 'model-compiler-sdk')
        self.edge('model-compiler-sdk', 'runtime-kernels')
        self.edge('runtime-kernels', 'kernel-build', 'build')
        for root in ['compile-demo', 'model-compiler-sdk', 'runtime-kernels']:
            with self.subTest(root=root), self.assertRaisesRegex(PolicyError, 'CUDA.*runtime-kernels'):
                Graph(self.data).require_cuda_free((root,))

    def test_dependency_tests_are_not_part_of_the_compiler_test_closure(self):
        self.add_native_test_helper()
        self.add('helper', 'helper')
        self.data['resolve']['nodes'][1]['deps'] = []
        self.edge('compile-demo', 'helper')
        self.edge('helper', 'native-helper', 'dev')
        Graph(self.data).require_cuda_free(('compile-demo',))

    def test_source_build_and_smoke_enforce_external_contributor_closure(self):
        self.add_native_test_helper()
        graph = Graph(self.data)
        output = self.root.with_name(self.root.name + '-outputs')
        with patch.object(source, 'verify_source', return_value={}), \
             patch.object(source, 'metadata', return_value=self.data), \
             patch.object(source, 'validate_lock'), \
             patch.object(source, 'selected_graph', return_value=(graph, ['compile-demo'])), \
             patch.object(workflow, 'metadata', return_value=self.data), \
             patch.object(workflow, 'selected_graph', return_value=(graph, ['compile-demo'])), \
             patch.object(workflow, 'environment', return_value={}), \
             patch.object(workflow, 'run') as command:
            with self.assertRaisesRegex(PolicyError, 'CUDA.*native-helper'):
                source.check(self.root, contributor_roots=('compile-demo',))
            with self.assertRaisesRegex(PolicyError, 'CUDA.*native-helper'):
                workflow.build(self.root, output/'target', output/'logs', contributor_roots=('compile-demo',))
            with patch.object(smoke, 'export', return_value=self.root), \
                 patch.object(smoke, 'provision_lock'), \
                 patch.object(smoke, 'git', return_value=b''), \
                 patch.object(smoke, 'sha', return_value='unchanged'):
                with self.assertRaisesRegex(PolicyError, 'CUDA.*native-helper'):
                    smoke.smoke(self.root, output, output/'host', output/'conformance')
            command.assert_not_called()

    def test_selected_cargo_graph_respects_features_and_test_dependencies(self):
        (self.root/'Cargo.toml').write_text('[workspace]\nmembers=["bundle","compiler","helper"]\nexclude=["sdk"]\nresolver="3"\n')
        manifests = {
            'bundle': '[lib]\nname="demo"\ncrate-type=["cdylib","rlib"]\n',
            'compiler': '[dev-dependencies]\nhelper={path="../helper"}\n[features]\nnative=["helper/native"]\n',
            'helper': '[build-dependencies]\nrenamed={package="kernel-build",path="../sdk/crates/kernel-build",optional=true}\n[features]\nnative=["dep:renamed"]\n',
            'sdk/crates/kernel-build': '',
        }
        for folder, extra in manifests.items():
            name = {'bundle':'bundle-demo', 'compiler':'compile-demo'}.get(folder, Path(folder).name)
            directory = self.root/folder
            (directory/'src').mkdir(parents=True, exist_ok=True)
            (directory/'src'/('main.rs' if folder == 'compiler' else 'lib.rs')).write_text('fn main() {}\n')
            (directory/'Cargo.toml').write_text(f'[package]\nname="{name}"\nversion="0.1.0"\nedition="2024"\n'+extra)
        (self.sdk/'Cargo.toml').write_text('[workspace]\nmembers=["crates/kernel-build"]\nresolver="3"\n')
        # Bootstrap this registry-free fixture once, as the existing metadata test does.
        subprocess.run([os.environ.get('CARGO','cargo'), 'metadata', '--offline', '--format-version', '1'],
                       cwd=self.root, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True)
        data = metadata(self.root/'Cargo.toml')
        for features, native in [((), False), (('compile-demo/native',), True)]:
            with self.subTest(features=features):
                graph, roots = selected_graph(data, self.root/'Cargo.toml', packages=('compile-demo',), features=features)
                self.assertEqual(graph.cuda_build_owners(roots), ['helper'] if native else [])
                if native:
                    with self.assertRaisesRegex(PolicyError, 'CUDA.*helper'):
                        check_external(data, self.root/'Cargo.toml', self.root/'model.toml', self.sdk,
                                       selected=graph, contributor_roots=('compile-demo',))
                else:
                    check_external(data, self.root/'Cargo.toml', self.root/'model.toml', self.sdk,
                                   selected=graph, contributor_roots=('compile-demo',))

    def test_selected_roles_do_not_promote_dev_dependencies_to_build(self):
        self.sdk.mkdir()
        (self.root/'Cargo.toml').write_text('[workspace]\nmembers=["bundle"]\nexclude=["sdk"]\nresolver="3"\n')
        (self.sdk/'Cargo.toml').write_text('[workspace]\nmembers=["crates/kernel-build"]\nresolver="3"\n')
        for folder, name, extra in [
            ('bundle', 'bundle-demo', '\n'.join([
                '[build-dependencies]',
                'native={package="kernel-build",path="../sdk/crates/kernel-build",optional=true}',
                '[dev-dependencies]',
                'native={package="kernel-build",path="../sdk/crates/kernel-build"}',
                '[features]', 'cuda=["dep:native"]', '',
            ])),
            ('sdk/crates/kernel-build', 'kernel-build', ''),
        ]:
            directory = self.root/folder
            (directory/'src').mkdir(parents=True, exist_ok=True)
            (directory/'src/lib.rs').write_text('pub fn fixture() {}\n')
            (directory/'Cargo.toml').write_text(f'[package]\nname="{name}"\nversion="0.1.0"\nedition="2024"\n'+extra)
        subprocess.run([os.environ.get('CARGO','cargo'), 'metadata', '--offline', '--format-version', '1'],
                       cwd=self.root, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True)
        data = metadata(self.root/'Cargo.toml')
        for features, expected in [((), []), (('cuda',), ['bundle-demo'])]:
            with self.subTest(features=features):
                graph, roots = selected_graph(data, self.root/'Cargo.toml', features=features)
                self.assertEqual(graph.cuda_build_owners(roots), expected)
                if expected:
                    with self.assertRaisesRegex(PolicyError, 'CUDA.*bundle-demo'):
                        graph.require_cuda_free(('bundle-demo',))
                else:
                    graph.require_cuda_free(('bundle-demo',))

    def test_compilers_link_the_tokenizer_and_kernel_catalog(self):
        for name in ['runtime-tokenizer', 'kernels-sm120']:
            self.add(name, 'sdk/crates/' + name)
            self.edge('compile-demo', name)
        self.check()
        initial = copy.deepcopy(self.data)
        # The tokenizer stays off the bundle's surface, and media decoding
        # is no SDK crate for any role.
        for origin, name, message in [('bundle-demo', 'runtime-tokenizer', 'bundle entry point runtime-tokenizer'),
                                      ('compile-demo', 'runtime-media', 'unsupported SDK package runtime-media')]:
            with self.subTest(origin=origin, name=name):
                self.data = copy.deepcopy(initial)
                if name == 'runtime-media':
                    self.add(name, 'sdk/crates/' + name)
                self.edge(origin, name)
                with self.assertRaisesRegex(PolicyError, message):
                    self.check()

    def test_roles_and_internal_sdk_dependencies(self):
        for name in ['runtime-bundle-sdk', 'runtime-bundle-abi', 'model-compiler-sdk',
                     'runtime-bundle', 'runtime-kernels', 'kernel-build', 'runtime-frontend-contract',
                     'runtime-sampling', 'runtime-model-api']:
            self.add(name, 'sdk/crates/' + name)
        for a, b in [('bundle-demo', 'runtime-bundle-sdk'),
                     ('runtime-bundle-sdk', 'runtime-bundle-abi'),
                     ('compile-demo', 'model-compiler-sdk'),
                     ('model-compiler-sdk', 'runtime-bundle'),
                     ('runtime-bundle', 'runtime-kernels'),
                     ('runtime-kernels', 'kernel-build'),
                     ('compile-demo', 'runtime-frontend-contract'),
                     ('runtime-frontend-contract', 'runtime-sampling'),
                     ('runtime-sampling', 'runtime-model-api')]:
            self.edge(a, b)
        self.check()
        self.edge('bundle-demo', 'runtime-kernels')
        with self.assertRaisesRegex(PolicyError, 'entry point'):
            self.check()

    def test_bundles_link_the_declared_frontend_and_not_its_host(self):
        for name in ['frontend-format', 'runtime-grammar', 'runtime-frontend-contract',
                     'runtime-bundle-abi', 'runtime-frontend']:
            self.add(name, 'sdk/crates/' + name)
        for a, b in [('bundle-demo', 'frontend-format'),
                     ('frontend-format', 'runtime-bundle-abi'),
                     ('frontend-format', 'runtime-frontend-contract'),
                     ('frontend-format', 'runtime-grammar')]:
            self.edge(a, b)
        self.check()
        self.edge('bundle-demo', 'runtime-frontend')
        with self.assertRaisesRegex(PolicyError, 'unsupported SDK package runtime-frontend'):
            self.check()

    def test_forbidden_host_edges_and_helpers(self):
        initial = copy.deepcopy(self.data)
        for name in ['runtime-cuda', 'runtime-engine', 'runtime-host', 'runtime-server']:
            for kind in (None, 'build'):
                for helper in (False, True):
                    with self.subTest(name=name, kind=kind, helper=helper):
                        self.data = copy.deepcopy(initial)
                        self.add(name, 'sdk/crates/' + name)
                        origin = 'bundle-demo'
                        if helper:
                            self.add('helper', 'helper', member=True)
                            self.edge(origin, 'helper')
                            origin = 'helper'
                        self.edge(origin, name, kind, target='cfg(windows)')
                        with self.assertRaisesRegex(PolicyError, 'forbidden host'):
                            self.check()

    def test_core_models_fixtures_and_tools_are_not_sdk(self):
        initial = copy.deepcopy(self.data)
        for name, directory in [('bundle-qwen38', 'models/qwen38/bundle'),
                                ('model-mimo26', 'models/mimo26/compiler'),
                                ('bundle-tp4-fixture', 'crates/bundle-tp4-fixture'),
                                ('kernel-lab', 'crates/kernel-lab'),
                                ('runtime-conformance', 'crates/runtime-conformance')]:
            for root in ['bundle-demo', 'compile-demo']:
                with self.subTest(name=name, root=root):
                    self.data = copy.deepcopy(initial)
                    self.add(name, 'sdk/' + directory)
                    self.add('helper', 'helper', member=True)
                    self.edge(root, 'helper')
                    self.edge('helper', name, 'build')
                    with self.assertRaisesRegex(PolicyError, 'unsupported SDK'):
                        self.check()

    def add_contract(self):
        self.add('demo-contract', 'contract', member=True)
        with (self.root / 'model.toml').open('a') as output:
            output.write('[contract]\npackage = "demo-contract"\n')
        for root in ['bundle-demo', 'compile-demo']:
            self.edge(root, 'demo-contract')

    def test_a_contract_shared_by_both_roles_links_the_model_side_sdk(self):
        self.add_contract()
        for name in ['model-compiler-sdk', 'runtime-bundle', 'runtime-weights', 'bundle-runtime',
                     'kernel-build', 'kernels-sm120', 'kernel-core', 'frontend-format']:
            self.add(name, 'sdk/crates/' + name)
            self.edge('demo-contract', name)
        # The chat format a contract holds reaches the shared crates behind
        # frontend-format, never the host's frontend.
        for name in ['runtime-grammar', 'runtime-frontend-contract', 'runtime-bundle-abi']:
            self.add(name, 'sdk/crates/' + name)
            self.edge('frontend-format', name)
        self.check()
        # The bundle itself keeps its own surface.
        self.edge('bundle-demo', 'model-compiler-sdk')
        with self.assertRaisesRegex(PolicyError, 'bundle entry point model-compiler-sdk'):
            self.check()

    def test_a_contract_crossing_is_checked_against_the_contract_surface(self):
        self.add_contract()
        initial = copy.deepcopy(self.data)
        # runtime-frontend-contract is on the compiler's surface and
        # runtime-grammar on the bundle's, but neither on the contract's,
        # whichever role's closure reaches the contract.
        for name in ['runtime-frontend-contract', 'runtime-grammar']:
            with self.subTest(name=name):
                self.data = copy.deepcopy(initial)
                self.add(name, 'sdk/crates/' + name)
                self.edge('demo-contract', name)
                with self.assertRaisesRegex(PolicyError, f'contract entry point {name} '):
                    self.check()

    def test_a_contract_cannot_reach_a_host_crate(self):
        self.add_contract()
        initial = copy.deepcopy(self.data)
        for name in ['runtime-engine', 'runtime-cuda', 'runtime-host', 'runtime-server']:
            for via in (None, 'model-compiler-sdk'):
                with self.subTest(name=name, via=via):
                    self.data = copy.deepcopy(initial)
                    self.add(name, 'sdk/crates/' + name)
                    origin = 'demo-contract'
                    if via:
                        self.add(via, 'sdk/crates/' + via)
                        self.edge(origin, via)
                        origin = via
                    self.edge(origin, name)
                    with self.assertRaisesRegex(PolicyError, name):
                        self.check()
                    # Also when only the compiler's closure reaches the contract.
                    node = next(n for n in self.data['resolve']['nodes'] if n['id'] == 'bundle-demo')
                    node['deps'] = []
                    with self.assertRaisesRegex(PolicyError, f'unsupported SDK package {name}'):
                        self.check()

    def test_helper_shared_with_compiler_keeps_bundle_restrictions(self):
        self.add('model-compiler-sdk', 'sdk/crates/model-compiler-sdk')
        self.add('helper', 'helper', member=True)
        for root in ['bundle-demo', 'compile-demo']:
            self.edge(root, 'helper')
        self.edge('helper', 'model-compiler-sdk')
        with self.assertRaisesRegex(PolicyError, 'entry point'):
            self.check()

    def test_second_sdk_identity_is_refused(self):
        self.add('runtime-bundle-sdk', 'sdk/crates/runtime-bundle-sdk', key='one')
        self.add('runtime-bundle-sdk', 'second/crates/runtime-bundle-sdk', key='two')
        self.edge('bundle-demo', 'one')
        self.edge('bundle-demo', 'two')
        with self.assertRaisesRegex(PolicyError, 'outside the pinned SDK'):
            self.check()

    def test_inventory_cannot_grant_exemptions(self):
        path = self.root / 'model.toml'
        path.write_text(path.read_text() + 'allowed = ["runtime-host"]\n')
        with self.assertRaisesRegex(InventoryError, 'unknown keys'):
            self.check()

    def test_undeclared_cdylib_and_bin_are_refused(self):
        for kind in ['cdylib', 'bin']:
            data = copy.deepcopy(self.data)
            self.add('helper', 'helper', [('secret', kind)], member=True)
            with self.assertRaisesRegex(InventoryError, 'no role'):
                self.check()
            self.data = data

    def test_dev_dependency_does_not_hide_production_edge(self):
        self.add('runtime-host', 'sdk/crates/runtime-host')
        self.edge('bundle-demo', 'runtime-host', 'dev')
        self.check()
        self.edge('bundle-demo', 'runtime-host', 'build')
        with self.assertRaisesRegex(PolicyError, 'forbidden host'):
            self.check()

    def test_real_cargo_includes_optional_target_build_edges(self):
        (self.root / 'Cargo.toml').write_text('[workspace]\nmembers=["bundle","compiler","helper"]\nexclude=["sdk"]\nresolver="3"\n')
        manifests = {
            'bundle': '[dependencies]\nhelper={path="../helper"}\n[lib]\nname="demo"\ncrate-type=["cdylib","rlib"]\n',
            'compiler': '',
            'helper': '[target.\'cfg(windows)\'.build-dependencies]\nrenamed={package="runtime-host",path="../sdk/crates/runtime-host",optional=true}\n',
            'sdk/crates/runtime-host': '',
        }
        for folder, extra in manifests.items():
            name = {'bundle':'bundle-demo','compiler':'compile-demo'}.get(folder, Path(folder).name)
            directory = self.root / folder
            (directory/'src').mkdir(parents=True, exist_ok=True)
            (directory/'src'/('main.rs' if folder == 'compiler' else 'lib.rs')).write_text('fn main() {}\n')
            (directory/'Cargo.toml').write_text(f'[package]\nname="{name}"\nversion="0.1.0"\nedition="2024"\n' + extra)
        (self.sdk/'Cargo.toml').write_text('[workspace]\nmembers=["crates/runtime-host"]\nresolver="3"\n')
        # Provision this fresh, registry-free fixture lock once. Verification
        # below uses the production locked/offline all-feature metadata path.
        subprocess.run([os.environ.get('CARGO','cargo'),'metadata','--offline','--format-version','1'],
                       cwd=self.root, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True)
        self.data = metadata(self.root/'Cargo.toml')
        with self.assertRaisesRegex(PolicyError, 'forbidden host'):
            self.check()

    def test_metadata_selects_all_features_without_host_target_filter(self):
        with patch('model_sdk.graph.subprocess.check_output', return_value=json.dumps(self.data)) as call:
            metadata(self.root / 'Cargo.toml')
        args = call.call_args.args[0]
        for flag in ['--locked', '--offline', '--all-features']:
            self.assertIn(flag, args)
        self.assertNotIn('--filter-platform', args)


if __name__ == '__main__':
    unittest.main()
