"""Offline distribution license and installer integrity checks."""
import hashlib
import importlib.util
import io
import json
import tarfile
import tomllib
import os
import shutil
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('notices', ROOT / 'tools/release/notices.py')
notices = importlib.util.module_from_spec(spec)
spec.loader.exec_module(notices)


plan_spec = importlib.util.spec_from_file_location('release_plan', ROOT / 'tools/release/plan.py')
plan = importlib.util.module_from_spec(plan_spec)
plan_spec.loader.exec_module(plan)


def inventory_fixture(root, models=('mimo26', 'qwen38', 'unselected')):
    """Small metadata graph with selected and developer-only binary targets."""
    packages = []
    for model in models:
        bins = [f'chopinfer-model-{model}', f'{model}-package']
        if model == 'qwen38':
            bins.append('qwen38-oracle')
        selected = model != 'unselected'
        directory = root / 'models' / model
        (directory / 'bundle').mkdir(parents=True)
        (directory / 'compiler').mkdir()
        for role, targets in (('bundle', [(f'bundle_{model}', 'cdylib')]), ('compiler', [(name, 'bin') for name in bins])):
            manifest = directory / role / 'Cargo.toml'
            manifest.write_text('')
            name = ('bundle-' if role == 'bundle' else 'model-') + model
            packages.append(dict(id=name, name=name, manifest_path=str(manifest),
                                 targets=[dict(name=target, kind=[kind]) for target, kind in targets]))
        text = 'schema = "chopinfer-model-inventory-v1"\n'
        text += f'[bundle]\npackage = "bundle-{model}"\ncdylib = ["bundle_{model}"]\n'
        text += f'[compiler]\npackage = "model-{model}"\nbin = {json.dumps(bins)}\n'
        if selected:
            text += f'[release]\nenabled = true\nbins = ["{model}-package", "chopinfer-model-{model}"]\ncdylibs = ["bundle_{model}"]\n'
        (directory / 'model.toml').write_text(text)
        (directory / 'bundle/kernels.lock').write_text('{}')
        (directory / 'profiles.json').write_text('{}')
        (directory / 'checkpoint-files.json').write_text('{}')
    return dict(packages=packages, workspace_members=[package['id'] for package in packages])


class ReleasePolicy(unittest.TestCase):
    def package(self, root, license='MIT AND BSD-3-Clause', texts=None):
        if texts is None:
            texts = {'LICENSE': 'MIT permission and copyright\n',
                     'LICENSE.httprouter': 'BSD permission and copyright\n'}
        directory = root / 'registry/cache/example'
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / 'router-1.0.crate'
        with tarfile.open(path, 'w:gz') as archive:
            manifest = '[package]\n' + (f'license = "{license}"\n' if license else '')
            for name, text in {'Cargo.toml': manifest, **texts}.items():
                info = tarfile.TarInfo('router-1.0/' + name)
                data = text.encode()
                info.size = len(data)
                archive.addfile(info, io.BytesIO(data))
        return {'package': [{'name': 'router', 'version': '1.0',
                'source': 'registry+https://github.com/rust-lang/crates.io-index',
                'checksum': hashlib.sha256(path.read_bytes()).hexdigest()}]}

    def test_reproduce_all_notices_and_reject_modified_or_missing_text(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = self.package(root)
            rendered = notices.render(data, root)
            self.assertEqual(rendered, notices.render(data, root))
            self.assertIn(b'MIT permission and copyright', rendered)
            self.assertIn(b'BSD permission and copyright', rendered)
            next(root.glob('registry/cache/*/*.crate')).write_bytes(b'replacement')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                notices.render(data, root)
            data = self.package(root, texts={})
            with self.assertRaisesRegex(ValueError, 'no license texts'):
                notices.render(data, root)

    def test_nested_license_texts_keep_their_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = self.package(root, texts={'LICENSE': 'Root permission\n',
                                            'src/tables/LICENSE': 'Nested permission\n'})
            rendered = notices.render(data, root)
            self.assertIn(b'--- src/tables/LICENSE ---\nNested permission\n', rendered)
            self.assertEqual(rendered, notices.render(data, root))

    def test_unknown_and_copyleft_licenses_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for license in (None, 'GPL-3.0-only', 'MIT OR Unknown'):
                with self.subTest(license=license), self.assertRaisesRegex(ValueError, 'unreviewed'):
                    notices.render(self.package(root, license), root)

    def test_locked_graph_has_verified_notices(self):
        lock = tomllib.loads((ROOT / 'Cargo.lock').read_text())
        home = Path(os.environ.get('CARGO_HOME', Path.home() / '.cargo'))
        result = notices.render(lock, home)
        self.assertIn(b'matchit ', result)
        self.assertIn(b'LICENSE.httprouter', result)
        package = next(p for p in lock['package'] if p['name'] == 'regex-syntax')
        prefix = f"regex-syntax-{package['version']}"
        path = next((home / 'registry/cache').glob(f'*/{prefix}.crate'))
        with tarfile.open(path) as archive:
            unicode_license = archive.extractfile(prefix + '/src/unicode_tables/LICENSE-UNICODE').read()
        self.assertIn(b'--- src/unicode_tables/LICENSE-UNICODE ---\n' + unicode_license, result)
        self.assertEqual(result, notices.render(lock, home))

    def test_every_crate_declares_the_project_license(self):
        # LICENSE is Apache-2.0, and every tracked manifest says so: a package
        # states it or inherits it from its workspace, which states it.
        self.assertIn('Apache License\n                           Version 2.0, January 2004',
                      (ROOT / 'LICENSE').read_text())
        manifests = subprocess.run(['git', 'ls-files', '-z', '--', 'Cargo.toml', '*/Cargo.toml'], cwd=ROOT,
                                   check=True, capture_output=True).stdout.decode().split('\0')
        manifests = [ROOT / name for name in manifests if name]
        self.assertIn(ROOT / 'Cargo.toml', manifests)
        workspaces = {}
        for path in manifests:
            data = tomllib.loads(path.read_text())
            if 'workspace' in data:
                workspaces[path.parent] = data['workspace'].get('package', {}).get('license')
        for path in manifests:
            package = tomllib.loads(path.read_text()).get('package')
            if package is None:
                continue
            with self.subTest(manifest=path.relative_to(ROOT).as_posix()):
                license = package.get('license')
                if license == {'workspace': True}:
                    owner = next((parent for parent in path.parents if parent in workspaces), None)
                    self.assertIsNotNone(owner, 'license.workspace without a tracked workspace')
                    license = workspaces[owner]
                self.assertEqual(license, 'Apache-2.0')
        for owner, license in workspaces.items():
            with self.subTest(workspace=owner.relative_to(ROOT).as_posix()):
                self.assertEqual(license, 'Apache-2.0')

    def test_archive_contains_project_and_dependency_licenses(self):
        # Build outputs are synthetic; exercise the real staging and archive code.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scripts = root / 'tools/release'
            scripts.mkdir(parents=True)
            for name in ('build.sh', 'plan.py', 'notices.py', 'INSTALL.md'):
                shutil.copyfile(ROOT / 'tools/release' / name, scripts / name)
            (root / 'tools/model_sdk').mkdir()
            shutil.copyfile(ROOT / 'tools/model_sdk/inventory.py', root / 'tools/model_sdk/inventory.py')
            metadata = inventory_fixture(root)
            (root / 'LICENSE').write_text('Project license\n')
            (root / 'Cargo.toml').write_text('[workspace.package]\nversion = "0.1.0-alpha.1"\n')
            (root / 'rust-toolchain.toml').write_text('[toolchain]\nchannel = "1.89.0"\n')
            home = root / 'cargo-home'
            package = self.package(home)['package'][0]
            (root / 'Cargo.lock').write_text('version = 4\n[[package]]\n' + ''.join(
                f'{key} = "{value}"\n' for key, value in package.items()))
            binaries = root / 'target/release'
            binaries.mkdir(parents=True)
            revision = 'a' * 40
            for name in ('chopinfer', 'chopinfer-host', 'chopinfer-top', 'mimo26-package', 'qwen38-package',
                         'chopinfer-model-mimo26', 'chopinfer-model-qwen38'):
                # chopinfer-top reports its version without the revision.
                version = '0.1.0-alpha.1' + ('' if name == 'chopinfer-top' else f' ({revision})')
                (binaries / name).write_text(
                    '#!/bin/sh\ncase "$*" in\n'
                    f'  --version) echo "{name} {version}" ;;\n'
                    '  "eval files --model models/mimo26") printf "%s\\n" fixtures/mimo26/frontend/chat-renders.json models/mimo26/checkpoint-files.json ;;\n'
                    '  "eval files --model models/qwen38") echo fixtures/qwen38/frontend/chat-renders.json ;;\n'
                    '  "eval files --model models/qwen38 --variant demo") printf "%s\\n" models/qwen38/qual.demo.json fixtures/qwen38/demo/teacher.json fixtures/qwen38/frontend/chat-renders.json ;;\n'
                    '  *) exit 2 ;;\nesac\n')
            eval_files = ('fixtures/mimo26/frontend/chat-renders.json',
                          'fixtures/qwen38/frontend/chat-renders.json',
                          'models/qwen38/qual.demo.json', 'fixtures/qwen38/demo/teacher.json')
            for name in eval_files:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('synthetic eval input\n')
            for model in ('mimo26', 'qwen38'):
                (binaries / f'libbundle_{model}.so').write_bytes(b'synthetic library')
                model_root = root / 'models' / model
                (model_root / 'bundle').mkdir(parents=True, exist_ok=True)
                for name in ('bundle/kernels.lock', 'profiles.json', 'checkpoint-files.json'):
                    (model_root / name).write_text('{}')
                    (model_root / name).chmod(0o664)
            # Qwen's bundle also keeps a directory-form lock family.
            family = root / 'models/qwen38/bundle/kernels.lock.d'
            family.mkdir()
            for name in ('context_parallel.lock', 'context_parallel.provenance.json'):
                (family / name).write_text('[]')
            shutil.copyfile(ROOT / 'fixtures/GSM8K-LICENSE', root / 'fixtures/GSM8K-LICENSE')
            (root / 'THIRD_PARTY_NOTICES').write_text('Project third-party notices\n')
            programs = root / 'bin'
            programs.mkdir()
            cargo = programs / 'cargo'
            cargo.write_text('#!/usr/bin/env python3\nimport json, sys\nfrom pathlib import Path\n'
                             f'data = {metadata!r}\n'
                             'if sys.argv[1] == "metadata": print(json.dumps(data))\n'
                             'else:\n'
                             f'    with Path({str(root / "build-argv")!r}).open("a") as log: log.write(json.dumps(sys.argv[1:]) + "\\n")\n')
            cargo.chmod(0o755)
            for name, body in {'rustc': 'echo "rustc 1.89.0 (fixture)"'}.items():
                path = programs / name
                path.write_text('#!/bin/sh\n' + body + '\n')
                path.chmod(0o755)
            cuda = root / 'cuda'
            (cuda / 'bin').mkdir(parents=True)
            nvcc = cuda / 'bin/nvcc'
            nvcc.write_text('#!/bin/sh\necho V13.2.86\n')
            nvcc.chmod(0o755)
            env = dict(os.environ, CARGO_HOME=str(home), CARGO_TARGET_DIR=str(root / 'target'),
                       PATH=str(programs) + ':' + os.environ['PATH'], CHOPINFER_CUDA_HOME=str(cuda),
                       CHOPINFER_BUILD_REVISION=revision,
                       KERNELS_LOCK_UPDATE='0', CHOPINFER_KERNEL_LINEINFO='0')
            result = subprocess.run(['bash', scripts / 'build.sh', root / 'dist'],
                                    env=env, text=True, capture_output=True, preexec_fn=lambda: os.umask(0o002))
            self.assertEqual(result.returncode, 0, result.stderr)
            with tarfile.open(next((root / 'dist').glob('*.tar.gz'))) as archive:
                prefix = 'chopinfer-0.1.0-alpha.1-linux-x86_64-cuda13.2/'
                for name in eval_files:
                    self.assertEqual(archive.extractfile(prefix + name).read(), b'synthetic eval input\n')
                expected = {
                    'bin/chopinfer', 'bin/chopinfer-host', 'bin/chopinfer-top', 'bin/mimo26-package', 'bin/qwen38-package',
                    'bin/chopinfer-model-qwen38', 'bin/chopinfer-model-mimo26',
                    'lib/libbundle_mimo26.so', 'lib/libbundle_qwen38.so',
                    'models/mimo26/kernels.lock',
                    'models/mimo26/checkpoint-files.json', 'models/qwen38/kernels.lock',
                    'models/qwen38/kernels.lock.d/context_parallel.lock',
                    'models/qwen38/kernels.lock.d/context_parallel.provenance.json',
                    'fixtures/mimo26/frontend/chat-renders.json', 'fixtures/qwen38/frontend/chat-renders.json',
                    'models/qwen38/qual.demo.json', 'fixtures/qwen38/demo/teacher.json',
                    'fixtures/GSM8K-LICENSE', 'LICENSE', 'THIRD_PARTY_NOTICES', 'README.md', 'BUILD_INFO', 'SHA256SUMS',
                }
                self.assertEqual({member.name.removeprefix(prefix) for member in archive.getmembers() if member.isfile()}, expected)
                self.assertEqual(sum(member.name == prefix + 'models/mimo26/checkpoint-files.json' for member in archive.getmembers()), 1)
                for model in ('mimo26', 'qwen38'):
                    self.assertEqual(archive.getmember(prefix + f'models/{model}/kernels.lock').mode, 0o664)
                # Serving profiles ship inside each package, not beside the host.
                self.assertFalse(any(member.name.endswith('profiles.json') for member in archive.getmembers()))
                self.assertFalse(any('unselected' in member.name or 'qwen38-oracle' in member.name for member in archive.getmembers()))
                self.assertEqual(archive.extractfile(prefix + 'LICENSE').read(), b'Project license\n')
                self.assertEqual(archive.extractfile(prefix + 'fixtures/GSM8K-LICENSE').read(),
                                 (ROOT / 'fixtures/GSM8K-LICENSE').read_bytes())
                self.assertIn(b'fixtures/GSM8K-LICENSE', archive.extractfile(prefix + 'SHA256SUMS').read())
                self.assertIn(prefix + 'THIRD_PARTY_NOTICES', archive.getnames())
                rendered = archive.extractfile(prefix + 'THIRD_PARTY_NOTICES').read()
                self.assertTrue(rendered.startswith(b'Project third-party notices\n'))
                self.assertIn(b'BSD permission and copyright', rendered)
                self.assertIn(b'THIRD_PARTY_NOTICES', archive.extractfile(prefix + 'SHA256SUMS').read())


            self.assertEqual([json.loads(line) for line in (root / 'build-argv').read_text().splitlines()], [
                ['build', '--release', '--locked', '--offline', '-p', 'runtime-host', '--no-default-features', '--features', 'linux-cuda-host', '--bin', 'chopinfer', '--bin', 'chopinfer-host'],
                ['build', '--release', '--locked', '--offline', '-p', 'chopinfer-top', '--bin', 'chopinfer-top'],
                ['build', '--release', '--locked', '--offline', '-p', 'model-mimo26', '-p', 'model-qwen38', '--bin', 'mimo26-package', '--bin', 'qwen38-package', '--bin', 'chopinfer-model-qwen38', '--bin', 'chopinfer-model-mimo26'],
                ['build', '--release', '--locked', '--offline', '-p', 'bundle-mimo26', '-p', 'bundle-qwen38'],
            ])
            # A build that rewrites a lock family file fails, as for kernels.lock.
            cargo.write_text(cargo.read_text() + f'    Path({str(family / "context_parallel.lock")!r}).write_text("changed")\n')
            result = subprocess.run(['bash', scripts / 'build.sh', root / 'dist-changed'],
                                    env=env, text=True, capture_output=True, preexec_fn=lambda: os.umask(0o002))
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('context_parallel.lock: FAILED', result.stdout + result.stderr)

    def test_staging_plan_matches_the_previous_order_and_paths(self):
        # The released models' shape: a developer-only binary and an
        # unselected model stay out. Each model's own requirements
        # (models/<m>/test_inventory.py) pin its real release names.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = plan.read(root, inventory_fixture(root))
            rows = plan.stage(records, root)
        self.assertEqual(rows, (
            ('binary', 'chopinfer', 'bin/chopinfer', '755'),
            ('binary', 'chopinfer-host', 'bin/chopinfer-host', '755'),
            ('binary', 'chopinfer-top', 'bin/chopinfer-top', '755'),
            ('binary', 'mimo26-package', 'bin/mimo26-package', '755'),
            ('binary', 'qwen38-package', 'bin/qwen38-package', '755'),
            ('binary', 'chopinfer-model-qwen38', 'bin/chopinfer-model-qwen38', '755'),
            ('binary', 'chopinfer-model-mimo26', 'bin/chopinfer-model-mimo26', '755'),
            ('binary', 'libbundle_mimo26.so', 'lib/libbundle_mimo26.so', '755'),
            ('asset', 'models/mimo26/bundle/kernels.lock', 'models/mimo26/kernels.lock', 'copy'),
            ('eval', 'models/mimo26', '-', '-'),
            ('binary', 'libbundle_qwen38.so', 'lib/libbundle_qwen38.so', '755'),
            ('asset', 'models/qwen38/bundle/kernels.lock', 'models/qwen38/kernels.lock', 'copy'),
            ('eval', 'models/qwen38', '-', '-'),
        ))

    def test_staging_plan_ships_every_released_model_once(self):
        from model_sdk import inventory
        models = inventory.load(ROOT)
        released = [model for model in models if model.release.enabled]
        self.assertTrue(released)
        records = inventory.release_plan(models)
        rows = plan.stage(records, ROOT)
        self.assertEqual(len(rows), len(set(rows)))
        expected = {('binary', name, 'bin/' + name, '755') for name in ('chopinfer', 'chopinfer-host', 'chopinfer-top')}
        for model in released:
            expected.update(('binary', name, 'bin/' + name, '755') for name in model.release.bins)
            expected.update(('binary', f'lib{name}.so', f'lib/lib{name}.so', '755') for name in model.release.cdylibs)
            # Every kernel lock form the bundle keeps ships at the same relative path.
            expected.update(('asset', source, f'models/{model.name}/' + source.removeprefix(f'{model.directory}/bundle/'), 'copy')
                            for source in inventory.kernel_locks(model))
            expected.add(('eval', model.directory, '-', '-'))
            expected.update(('eval', model.directory, path.name.removeprefix('qual.').removesuffix('.json'), '-')
                            for path in (ROOT / model.directory).glob('qual.*.json'))
            expected.update(('asset', asset.source, asset.destination, 'copy') for asset in model.release.assets)
        self.assertEqual(set(rows), expected)

    def test_each_declared_variant_stages_beside_its_model(self):
        """A released model's checkpoint variants (qual.VARIANT.json) each
        get an eval row after the model's own, in name order, so an install
        holds each variant's declaration, locks and references; an
        unselected model's variants stay out."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = plan.read(root, inventory_fixture(root))
            for model, names in (('qwen38', ('qual.json', 'qual.b-2.json', 'qual.a.json', 'qual-references.a.lock.json')),
                                 ('unselected', ('qual.json', 'qual.c.json'))):
                for name in names:
                    (root / 'models' / model / name).write_text('{}')
            rows = [row for row in plan.stage(records, root) if row[0] == 'eval']
        self.assertEqual(rows, [('eval', 'models/mimo26', '-', '-'), ('eval', 'models/qwen38', '-', '-'),
                                ('eval', 'models/qwen38', 'a', '-'), ('eval', 'models/qwen38', 'b-2', '-')])

    def test_selected_third_model_uses_declared_artifact_names(self):
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
            records = plan.read(root, metadata)
            self.assertEqual(plan.build_args(records, 'bin'), ('-p', 'model-third', '--bin', 'assemble-third', '--bin', 'chopinfer-model-custom'))
            self.assertEqual(plan.build_args(records, 'cdylib'), ('-p', 'bundle-third'))
            self.assertEqual(plan.stage(records, root), (
                ('binary', 'chopinfer', 'bin/chopinfer', '755'),
                ('binary', 'chopinfer-host', 'bin/chopinfer-host', '755'),
                ('binary', 'chopinfer-top', 'bin/chopinfer-top', '755'),
                ('binary', 'assemble-third', 'bin/assemble-third', '755'),
                ('binary', 'chopinfer-model-custom', 'bin/chopinfer-model-custom', '755'),
                ('binary', 'libnative_custom.so', 'lib/libnative_custom.so', '755'),
                ('asset', 'models/third/bundle/kernels.lock', 'models/third/kernels.lock', 'copy'),
                ('eval', 'models/third', '-', '-'),
            ))

    def test_missing_release_asset_and_undeclared_binary_are_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata = inventory_fixture(root)
            (root / 'models/mimo26/bundle/kernels.lock').unlink()
            with self.assertRaisesRegex(plan.inventory.InventoryError, "release.kernels_lock: missing file"):
                plan.read(root, metadata)
            (root / 'models/mimo26/bundle/kernels.lock').write_text('{}')
            path = root / 'models/mimo26/model.toml'
            path.write_text(path.read_text().replace('"mimo26-package", "chopinfer-model-mimo26"', '"developer"'))
            with self.assertRaisesRegex(plan.inventory.InventoryError, r"release.bins\[0\]: target 'developer' needs exactly one owning role"):
                plan.read(root, metadata)

    def test_tampered_installer_never_executes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            marker = root / 'executed'
            curl = root / 'curl'
            curl.write_text('#!/bin/sh\nfor last; do :; done\nprintf "#!/bin/sh\\ntouch %s\\n" "$MARKER" > "$last"\n')
            curl.chmod(0o755)
            result = subprocess.run(['bash', 'tools/release/install-rust.sh', '1.89.0'], cwd=ROOT,
                                    env=dict(os.environ, PATH=str(root) + ':' + os.environ['PATH'], MARKER=str(marker)),
                                    text=True, capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('FAILED', result.stdout)
            self.assertFalse(marker.exists())


if __name__ == '__main__':
    unittest.main()
