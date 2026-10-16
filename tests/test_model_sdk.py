#!/usr/bin/env python3
"""Source identities, one-time lock provisioning, model relocks and runner safeguards."""
import contextlib
import io
import json
import os
import posixpath
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import tomllib
import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.model_sdk import source, workflow, diagnostic, model_export, relock, inventory
from tools.model_sdk.graph import Graph, PolicyError


ROOT = Path(__file__).resolve().parents[1]


def writable(root):
    for p in [root, *root.rglob('*')]:
        if not p.is_symlink():
            p.chmod(p.stat().st_mode | 0o700)


class DiagnosticMetadataTests(unittest.TestCase):
    def capture(self, root):
        import struct
        scope = dict(model='external-example', evidence='chained', contract='example.exact.v1',
                     inputs={'kind':'tensor', 'rows':[{'position':0}], 'state':{'kind':'fresh'}},
                     probes=[{'name':'z', 'shape':[1,2]}, {'name':'a', 'shape':[1,2]}])
        description = dict(capture_id='fixture', scope=scope, reproduction=['example', '--capture'],
                           limits={'metadata_bytes':1 << 20, 'tensor_bytes':1024})
        diagnostic.write_new(root / 'scope.json', description)
        tensors = []
        for name in ('z', 'a'):
            (root / (name + '.bin')).write_bytes(struct.pack('<2f', 1, 2))
            tensors.append(diagnostic.tensor_descriptor(name, [1,2], 'F32', name + '.bin'))
        return diagnostic.inventory(root, description, tensors, {'checkpoint':{'sha256':'a' * 64}})

    def test_sealed_inventory_is_ordered_and_immutable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            value = self.capture(root)
            self.assertEqual(diagnostic.load(root), value)
            self.assertEqual([t['name'] for t in value['tensors']], ['z', 'a'])
            with self.assertRaises(FileExistsError):
                diagnostic.seal(root)
            (root / 'a.bin').write_bytes(b'changed!')
            with self.assertRaisesRegex(ValueError, 'stale'):
                diagnostic.load(root)

    def test_partial_scope_and_symlink_payload_cannot_be_sealed_as_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.capture(root)
            producer = diagnostic.read_json(root / 'producer.json')
            producer['tensors'].pop()
            (root / 'producer.json').write_bytes(diagnostic.canonical(producer))
            (root / 'diagnostic.json').unlink()
            diagnostic.seal(root)
            with self.assertRaisesRegex(ValueError, 'required probe'):
                diagnostic.load(root)
            (root / 'a.bin').unlink()
            (root / 'a.bin').symlink_to(root / 'z.bin')
            with self.assertRaisesRegex(ValueError, 'symlink'):
                diagnostic.read_bytes(root / 'a.bin')

    def test_bad_metadata_and_tensor_spans_are_rejected_before_allocation(self):
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            diagnostic.decode_json('{"a":1,"a":2}')
        with self.assertRaisesRegex(ValueError, 'nonfinite'):
            diagnostic.decode_json('{"a":NaN}')
        with self.assertRaises(ValueError):
            diagnostic.tensor_descriptor('a', [1 << 62, 4], 'F64', 'a.bin')
        with self.assertRaises(ValueError):
            diagnostic.tensor_descriptor('a', [1], 'F32', '../a.bin')


class SourceCase(unittest.TestCase):
    def setUp(self):
        # Isolate fixture writes and Git calls made by the code under test.
        environment = {key: value for key, value in os.environ.items() if not key.startswith('GIT_')}
        environment.update(GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_NOSYSTEM='1')
        isolated = patch.dict(os.environ, environment, clear=True)
        isolated.start()
        self.addCleanup(isolated.stop)
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name)
        self.root = self.base / 'source'
        self.root.mkdir()
        self.template = self.root / 'examples/external-model'
        self.template.mkdir(parents=True)
        (self.template / 'Cargo.toml').write_text('[workspace]\n')
        (self.root / 'Cargo.lock').write_text('version = 4\n')
        (self.root / 'input.rs').write_text('original\n')
        subprocess.run(['git', 'init', '-q', str(self.root)], check=True)
        source.git(self.root, 'add', 'Cargo.lock', 'input.rs', 'examples/external-model/Cargo.toml')
        env = dict(os.environ, GIT_AUTHOR_NAME='test', GIT_AUTHOR_EMAIL='test',
                   GIT_COMMITTER_NAME='test', GIT_COMMITTER_EMAIL='test')
        subprocess.run(['git', '-C', str(self.root), '-c', 'commit.gpgsign=false', 'commit', '-qm', 'Initial'],
                       env=env, check=True)

    def tearDown(self):
        writable(self.base)
        self.temporary.cleanup()

    def export(self, name='model', **kwargs):
        return source.export(self.root, self.base / name, **kwargs)


class SourceTests(SourceCase):
    def test_release_is_relocatable_pinned_and_read_only(self):
        model = self.export()
        pin = source.verify_source(model, release=True)
        self.assertEqual(pin['revision'], source.git(self.root, 'rev-parse', 'HEAD').decode().strip())
        self.assertFalse((model / 'Cargo.lock').exists())
        self.assertFalse((model / 'sdk/.git').exists())
        self.assertEqual((model / 'sdk/input.rs').stat().st_mode & 0o222, 0)
        other = self.base / 'unrelated' / 'model'
        shutil.copytree(model, other)
        self.assertEqual(source.verify_source(other), pin)
        (other / 'sdk/input.rs').chmod(0o600)
        (other / 'sdk/input.rs').write_text('modified\n')
        with self.assertRaisesRegex(PolicyError, 'sources changed'):
            source.verify_source(other, release=True)

    def test_dirty_release_refused_and_development_captures_edits(self):
        before = source.verify_source(self.export('before'))
        (self.root / 'input.rs').write_text('edited\n')
        (self.root / 'new.rs').write_text('selected\n')
        with self.assertRaisesRegex(PolicyError, 'clean source'):
            self.export('release')
        after = self.export('development', development=True, includes=['new.rs'])
        pin = source.verify_source(after)
        self.assertNotEqual(pin['source_sha256'], before['source_sha256'])
        self.assertEqual((after / 'sdk/input.rs').read_text(), 'edited\n')
        self.assertEqual((after / 'sdk/new.rs').read_text(), 'selected\n')
        with self.assertRaisesRegex(PolicyError, 'development SDK'):
            source.verify_source(after, release=True)

    def test_an_export_leaves_out_the_workspace_lock_file(self):
        (self.template / workflow.SDK_LOCK).write_text('123')
        model = self.export(development=True, includes=[f'examples/external-model/{workflow.SDK_LOCK}'])
        self.assertFalse((model / workflow.SDK_LOCK).exists())
        self.assertIn(f'/{workflow.SDK_LOCK}', (ROOT / 'examples/external-model/.gitignore').read_text().split())

    def test_exported_workspaces_commit_their_lock_and_pin(self):
        # The docs say to commit Cargo.lock and sdk-pin.json; the SDK, build
        # outputs and the SDK's workspace lock file stay ignored.
        for name, ignore in (('template', (ROOT / 'examples/external-model/.gitignore').read_text()),
                             ('model export', model_export.IGNORED)):
            with self.subTest(name):
                repository = self.base / name.replace(' ', '-')
                repository.mkdir()
                subprocess.run(['git', 'init', '-q', str(repository)], check=True)
                (repository / '.gitignore').write_text(ignore)
                paths = ['Cargo.lock', 'sdk-pin.json', 'sdk/Cargo.toml', 'target/debug', workflow.SDK_LOCK]
                ignored = subprocess.run(['git', '-C', str(repository), 'check-ignore', '--no-index', *paths],
                                         stdout=subprocess.PIPE, text=True).stdout.split()
                self.assertEqual(ignored, ['sdk/Cargo.toml', 'target/debug', workflow.SDK_LOCK])

    def test_exports_and_source_inclusions_cannot_escape(self):
        with self.assertRaisesRegex(PolicyError, 'unrelated'):
            source.export(self.root, self.root / 'generated')
        with self.assertRaisesRegex(PolicyError, 'inside'):
            self.export(development=True, includes=['../secret'])
        (self.root / 'link.rs').symlink_to(self.root / 'input.rs')
        with self.assertRaisesRegex(PolicyError, 'symlink'):
            self.export('linked', development=True, includes=['link.rs'])

    def test_provision_is_explicit_once_then_checks_locked(self):
        model = self.export()
        with patch.object(source.subprocess, 'run') as cargo, patch.object(source, 'check', return_value={}) as check:
            source.provision_lock(model)
            command = cargo.call_args.args[0]
            self.assertIn('--offline', command)
            self.assertNotIn('--locked', command)
            check.assert_called_once_with(model)
            with self.assertRaisesRegex(PolicyError, 'cannot replace'):
                source.provision_lock(model)
            self.assertEqual(cargo.call_count, 1)

    def test_lock_update_resolves_from_the_sdk_pins_and_restores_on_refusal(self):
        def package(name, version, source=None, dependencies=()):
            text = f'[[package]]\nname = "{name}"\nversion = "{version}"\n'
            if source:
                text += f'source = "{source}"\nchecksum = "{name}"\n'
            if dependencies:
                text += 'dependencies = [\n' + ''.join(f' "{d}",\n' for d in dependencies) + ']\n'
            return text
        def lock(*packages):
            return '# @generated\nversion = 4\n' + ''.join('\n' + p for p in packages)
        registry = 'registry+https://github.com/rust-lang/crates.io-index'
        (self.root / 'Cargo.lock').write_text(lock(package('pinned', '1.0.0', registry),
                                                   package('sdk-only', '2.0.0', registry)))
        model = self.export(development=True)
        with self.assertRaisesRegex(PolicyError, 'run lock first'):
            source.update_lock(model)
        previous = lock(package('model-bundle', '0.1.0', dependencies=['old-dep', 'pinned']),
                        package('old-dep', '0.3.0', registry), package('own', '0.5.0', registry),
                        package('pinned', '1.0.0', registry))
        (model / 'Cargo.lock').write_text(previous)
        resolved = lock(package('model-bundle', '0.1.0', dependencies=['new-dep', 'own', 'pinned']),
                        package('new-dep', '1.2.0', registry), package('own', '0.5.0', registry),
                        package('pinned', '1.0.0', registry))
        seeds = []
        def cargo(command, **kwargs):
            self.assertIn('--offline', command)
            self.assertNotIn('--locked', command)
            seeds.append((model / 'Cargo.lock').read_text())
            (model / 'Cargo.lock').write_text(resolved)
        with patch.object(source.subprocess, 'run', side_effect=cargo), \
                patch.object(source, 'check', return_value={'mode': 'release'}) as check:
            result = source.update_lock(model)
        check.assert_called_once_with(model.resolve())
        # Cargo starts from the SDK's tested lock and the model's own registry
        # packages; the model's path packages are resolved again.
        _, seeded = source.lock_packages(seeds[0])
        self.assertEqual(sorted(seeded), [('old-dep', '0.3.0', registry), ('own', '0.5.0', registry),
                                          ('pinned', '1.0.0', registry), ('sdk-only', '2.0.0', registry)])
        self.assertEqual(result, {'mode': 'release', 'added': [f'new-dep 1.2.0 ({registry})'],
                                  'removed': [f'old-dep 0.3.0 ({registry})'], 'changed': ['model-bundle 0.1.0']})
        self.assertEqual((model / 'Cargo.lock').read_text(), resolved)
        # A resolution that moves an SDK pin fails check and keeps the old lock.
        drifted = resolved.replace('version = "1.0.0"', 'version = "1.0.1"')
        def drift(command, **kwargs):
            (model / 'Cargo.lock').write_text(drifted)
        refusal = PolicyError('SDK registry dependency differs from its tested lock: pinned 1.0.1')
        with patch.object(source.subprocess, 'run', side_effect=drift), patch.object(source, 'check', side_effect=refusal):
            with self.assertRaisesRegex(PolicyError, 'kept the previous Cargo.lock: SDK registry dependency differs'):
                source.update_lock(model)
        self.assertEqual((model / 'Cargo.lock').read_text(), resolved)

    def test_lock_drift_in_sdk_closure_is_rejected(self):
        model = self.export()
        sdk_lock = model / 'sdk/Cargo.lock'
        sdk_lock.chmod(0o600)
        lock = 'version = 4\n[[package]]\nname = "dep"\nversion = "1.0.0"\nsource = "registry+test"\nchecksum = "aaa"\n'
        sdk_lock.write_text(lock)
        (model / 'Cargo.lock').write_text(lock)
        data = {'packages':[
            {'id':'sdk', 'name':'runtime-bundle-sdk', 'manifest_path':str(model/'sdk/crates/runtime-bundle-sdk/Cargo.toml')},
            {'id':'dep', 'name':'dep', 'version':'1.0.0', 'source':'registry+test', 'manifest_path':str(self.base/'cache/Cargo.toml')}],
            'resolve':{'nodes':[{'id':'sdk','deps':[{'pkg':'dep','dep_kinds':[{'kind':None}]}]}, {'id':'dep','deps':[]}]}}
        source.validate_lock(model, data)
        for field, value in [('checksum','bbb'), ('version','1.1.0'), ('source','registry+other')]:
            with self.subTest(field=field):
                original = {'checksum':'aaa','version':'1.0.0','source':'registry+test'}[field]
                (model/'Cargo.lock').write_text(lock.replace(original, value))
                with self.assertRaisesRegex(PolicyError, 'tested lock'):
                    source.validate_lock(model, data)


MODEL_FILES = {
    'Cargo.toml': '[workspace]\nmembers = ["crates/shared", "models/*/bundle"]\nresolver = "3"\n'
                  '[workspace.package]\nversion = "0.1.0"\nedition = "2024"\n'
                  '[workspace.dependencies]\nshared = { path = "crates/shared" }\n'
                  'model-m3-contract = { path = "models/m3/contract" }\n'
                  'model-other = { path = "models/other/compiler" }\n'
                  'serde = { version = "=1.0.219", features = ["derive"] }\nlibc = "=0.2.189"\n'
                  '[workspace.lints.rust]\nunsafe_code = "deny"\n'
                  '[profile.dev.package.sha2]\nopt-level = 3\n',
    'rust-toolchain.toml': '[toolchain]\nchannel = "1.89.0"\n',
    'crates/shared/Cargo.toml': '[package]\nname = "shared"\n',
    'crates/shared/cuda/shared.cuh': '',
    'docs/guide.md': '',
    'fixtures/m3/geometries/small/config.json': '{"source": "fixtures/m3/geometries/small"}\n',
    'fixtures/other/cases.json': '{}\n',
    'fixtures/shared.txt': 'shared fixture\n',
    # A shared fixture with the name of one of the model's: a rewritten
    # reference must not be rewritten again onto it.
    'fixtures/local.txt': 'shared-data',
    'top.txt': 'checkout-data',
    'fixtures/m3/local.txt': 'model-data',
    'models/m3/data.rs': 'pub const DATA: &str = include_str!(r"../../fixtures/m3/local.txt");\n',
    # Same name as the shared fixture: a parent component must not land here.
    'models/m3/shared.txt': 'model file\n',
    'models/m3/model.toml': '[contributor]\nfixtures = ["models/m3/contract/fixtures/config.json"]\n',
    'models/m3/bundle/Cargo.toml': '[package]\nname = "bundle-m3"\n',
    'models/m3/bundle/cuda/m3.cuh': '',
    'models/m3/bundle/kernels.lock.d/rows.lock': '[{"source": "models/m3/bundle/cuda"}]\n',
    'models/m3/compiler/Cargo.toml': '[package]\nname = "model-m3"\n',
    # An include names fixtures/... from its own file: this resolves to the
    # crate's top.txt, not the checkout's.
    'models/m3/compiler/src/lib.rs': 'pub const DATA: &str = include_str!("fixtures/m3/../../top.txt");\n',
    'models/m3/compiler/src/top.txt': 'model-source',
    # A join to a base the text establishes is relocated; a build script's
    # climb inside the model from a receiver it does not establish is kept
    # and, naming nothing, warned about.
    'models/m3/compiler/build.rs':
        'fn main() {\n'
        '    let model = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap()).join("..");\n'
        '    let data = fs::read_to_string(model.join("../../fixtures/m3/local.txt")).unwrap();\n'
        '    let input = cache_dir().join("../absent/input.txt");\n'
        '}\n',
    'models/m3/contract/Cargo.toml': '[package]\nname = "model-m3-contract"\n',
    'models/m3/contract/fixtures/config.json': '{"path": "models/m3/contract/fixtures/config.json"}\n',
    'models/m3/contract/src/lib.rs':
        '//! Rows from `models/m3/bundle/cuda` and `crates/shared/cuda` (e4m3/models/m3x).\n'
        'fn spec() -> Spec {\n'
        '    Spec {\n'
        '        sources_root: PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../.."),\n'
        '        include_dirs: vec!["crates/shared/cuda".into(), "models/m3/bundle/cuda".into()],\n'
        '        config: include_bytes!("../fixtures/config.json"),\n'
        '        geometry: PathBuf::from(env!("CARGO_MANIFEST_DIR"))\n'
        '            .join("../../../fixtures/m3/geometries/small"),\n'
        '    }\n'
        '}\n',
    # A build script joins from its manifest directory through a variable.
    'models/m3/bundle/src/sm120/build.rs':
        'fn main() {\n'
        '    let manifest = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap());\n'
        '    let root = manifest.join("../../..");\n'
        '    let tables = include_str!("../../../contract/tables.json");\n'
        '    let raw = include_str!(r"../../../contract/tables.json");\n'
        '    let hashed = include_bytes!(r#"../../../../../fixtures/m3/geometries/small/config.json"#);\n'
        '    let raw_concat = concat!(env!("CARGO_MANIFEST_DIR"), r"/../../../crates/shared/cuda/shared.cuh");\n'
        '    let shared = include_str!(concat!(\n'
        '        env!("CARGO_MANIFEST_DIR"),\n'
        '        "/../../../crates/shared/cuda/shared.cuh"\n'
        '    ));\n'
        '    let own = concat!(env!("CARGO_MANIFEST_DIR"), "/../model.toml");\n'
        '}\n',
    'models/m3/README.md': 'See [the guide](../../docs/guide.md), fixtures/m3/geometries/ and '
                           '[contract](contract/src/lib.rs).\n',
    'models/m3/qual.json': '{"exact": "fixtures/m3/geometries/small/config.json", '
                           '"cases": "fixtures/other/cases.json", "lock": "models/m3/model.toml", '
                           '"missing": "fixtures/absent/x.json", "parent": "fixtures/probe/../shared.txt", '
                           '"own_parent": "fixtures/m3/../shared.txt", "sentence": "in fixtures/m3/geometries/.", '
                           '"top": "fixtures/m3/../../top.txt", "directory": "models/m3", '
                           '"model_parent": "models/m3/contract/../model.toml"}\n',
    'models/other/compiler/Cargo.toml': '[package]\nname = "model-other"\n',
}


class ModelExportTests(SourceCase):
    def setUp(self):
        super().setUp()
        for relative, text in MODEL_FILES.items():
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
        source.git(self.root, 'add', '.')
        env = dict(os.environ, GIT_AUTHOR_NAME='test', GIT_AUTHOR_EMAIL='test',
                   GIT_COMMITTER_NAME='test', GIT_COMMITTER_EMAIL='test')
        subprocess.run(['git', '-C', str(self.root), '-c', 'commit.gpgsign=false', 'commit', '-qm', 'Model'],
                       env=env, check=True)

    def test_a_model_directory_becomes_a_workspace_with_model_relative_fixtures(self):
        warnings = io.StringIO()
        with contextlib.redirect_stderr(warnings):
            workspace = self.export(model='models/m3')
        source.verify_source(workspace, release=True)
        self.assertEqual(source.git(self.root, 'status', '--porcelain', '--untracked-files=all'), b'')
        manifest = tomllib.loads((workspace / 'Cargo.toml').read_text())
        self.assertEqual(manifest['workspace']['members'], ['bundle', 'compiler', 'contract'])
        self.assertEqual(manifest['workspace']['exclude'], ['sdk'])
        self.assertEqual(manifest['workspace']['package'], {'version': '0.1.0', 'edition': '2024'})
        self.assertEqual(manifest['workspace']['dependencies'], {
            'shared': {'path': 'sdk/crates/shared'}, 'model-m3-contract': {'path': 'contract'},
            'serde': {'version': '=1.0.219', 'features': ['derive']}, 'libc': '=0.2.189'})
        self.assertEqual(manifest['workspace']['lints'], {'rust': {'unsafe_code': 'deny'}})
        self.assertEqual(manifest['profile'], {'dev': {'package': {'sha2': {'opt-level': 3}}}})
        self.assertEqual((workspace / 'rust-toolchain.toml').read_text(), MODEL_FILES['rust-toolchain.toml'])
        # The generated workspace ignores what the example ignores, the SDK's workspace lock included.
        self.assertEqual((workspace / '.gitignore').read_text(), (ROOT / 'examples/external-model/.gitignore').read_text())
        # Fixture data and locks are byte copies; the fixtures sit in the model root.
        for exported, original in (('fixtures/geometries/small/config.json', 'fixtures/m3/geometries/small/config.json'),
                                   ('contract/fixtures/config.json', 'models/m3/contract/fixtures/config.json'),
                                   ('bundle/kernels.lock.d/rows.lock', 'models/m3/bundle/kernels.lock.d/rows.lock')):
            self.assertEqual((workspace / exported).read_text(), MODEL_FILES[original])
        self.assertFalse((workspace / 'fixtures/m3').exists())
        library = (workspace / 'contract/src/lib.rs').read_text()
        self.assertIn('Rows from `bundle/cuda` and `sdk/crates/shared/cuda` (e4m3/models/m3x).', library)
        self.assertIn('.join(".."),', library)
        self.assertIn('vec!["sdk/crates/shared/cuda".into(), "bundle/cuda".into()]', library)
        self.assertIn('include_bytes!("../fixtures/config.json")', library)
        self.assertIn('.join("../fixtures/geometries/small")', library)
        # Rust reads a climb from its crate, an include from its file.
        build = (workspace / 'bundle/src/sm120/build.rs').read_text()
        self.assertIn('manifest.join("..");', build)
        self.assertIn('include_str!("../../../contract/tables.json")', build)
        # Raw literals read the same way as normal ones.
        self.assertIn('include_str!(r"../../../contract/tables.json")', build)
        self.assertIn('include_bytes!(r#"../../../fixtures/geometries/small/config.json"#)', build)
        self.assertIn('concat!(env!("CARGO_MANIFEST_DIR"), r"/../sdk/crates/shared/cuda/shared.cuh")', build)
        self.assertIn('        "/../sdk/crates/shared/cuda/shared.cuh"\n', build)
        self.assertIn('concat!(env!("CARGO_MANIFEST_DIR"), "/../model.toml")', build)
        self.assertEqual((workspace / 'README.md').read_text(),
                         'See [the guide](sdk/docs/guide.md), fixtures/geometries/ and [contract](contract/src/lib.rs).\n')
        self.assertEqual(json.loads((workspace / 'qual.json').read_text()), {
            'exact': 'fixtures/geometries/small/config.json', 'cases': 'sdk/fixtures/other/cases.json',
            'lock': 'model.toml', 'missing': 'fixtures/absent/x.json',
            # Parent components resolve in the checkout before relocating.
            'parent': 'sdk/fixtures/shared.txt', 'own_parent': 'sdk/fixtures/shared.txt',
            'sentence': 'in fixtures/geometries/.',
            # A parent path reaching a top-level checkout file, and the model
            # directory itself, which is the workspace root.
            'top': 'sdk/top.txt', 'directory': '.',
            'model_parent': 'model.toml'})
        self.assertIn('"contract/fixtures/config.json"', (workspace / 'model.toml').read_text())
        receipt = json.loads((workspace / 'model-export.json').read_text())
        # Kept references that name nothing in the checkout are listed and
        # printed as warnings, the run-time build input among them.
        build = (workspace / 'compiler/build.rs').read_text()
        self.assertIn('model.join("fixtures/local.txt")', build)
        self.assertIn('cache_dir().join("../absent/input.txt")', build)
        build_input = {'file': 'compiler/build.rs', 'line': 4, 'reference': '../absent/input.txt',
                       'read_from': 'compiler'}
        self.assertIn(build_input, receipt['unresolved'])
        self.assertIn({'file': 'qual.json', 'line': 1, 'reference': 'fixtures/absent/x.json',
                       'read_from': 'checkout root'}, receipt['unresolved'])
        self.assertEqual(len(warnings.getvalue().splitlines()), len(receipt['unresolved']))
        self.assertIn("warning: compiler/build.rs:4: kept '../absent/input.txt' (read from compiler)",
                      warnings.getvalue())
        self.assertEqual((receipt['model'], receipt['source'], receipt['fixtures']), ('m3', 'models/m3', 'fixtures/m3'))
        # Rewritten once: the emitted fixtures/local.txt is the model's, not sdk/fixtures/local.txt.
        self.assertEqual((workspace / 'data.rs').read_text(),
                         'pub const DATA: &str = include_str!(r"fixtures/local.txt");\n')
        self.assertEqual((workspace / 'fixtures/local.txt').read_text(), 'model-data')
        self.assertEqual((workspace / 'compiler/src/lib.rs').read_text(),
                         MODEL_FILES['models/m3/compiler/src/lib.rs'])
        self.assertEqual(sorted(receipt['relocated']), ['README.md', 'bundle/src/sm120/build.rs', 'compiler/build.rs',
                                                        'contract/src/lib.rs', 'data.rs', 'model.toml', 'qual.json'])
        for relative, digests in receipt['relocated'].items():
            self.assertEqual(digests['source_sha256'], source.sha(self.root / 'models/m3' / relative))
            self.assertEqual(digests['sha256'], source.sha(workspace / relative))

    def test_an_sdk_manifest_without_optional_tables_still_exports(self):
        sdk = self.base / 'minimal-sdk'
        sdk.mkdir()
        (sdk / 'Cargo.toml').write_text('[workspace]\nmembers = ["models/*/compiler"]\n')
        manifest = tomllib.loads(model_export.manifest(sdk, 'm3', ['compiler']))
        self.assertEqual(manifest, {'workspace': {'members': ['compiler'], 'exclude': ['sdk']}})
        (sdk / 'Cargo.toml').write_text('[package]\nname = "single"\n')
        with self.assertRaisesRegex(PolicyError, 'declares no \\[workspace\\]'):
            model_export.manifest(sdk, 'm3', ['compiler'])

    def test_a_rewrite_that_names_other_content_fails_the_export(self):
        rooted, climb = model_export.Relocation.rooted, model_export.Relocation.climb
        def first_components(self, match):
            # The relocation by the first two components that
            # fixtures/<m>/../shared.txt once received.
            if match.group() == 'fixtures/m3/../shared.txt':
                return 'fixtures/../shared.txt'
            return rooted(self, match)
        def one_climb_short(self, value, base):
            result = climb(self, value, base)
            return result[3:] if result != value and result.startswith('../') else result
        text = model_export.Relocation.text
        def second_pass(self, relative, original):
            # The emitted text matched again, as a step-by-step rewrite did.
            return model_export.ROOTED.sub(lambda match: rooted(self, match), text(self, relative, original))
        def kept_at_top(self, match):
            # A `..` path reaching a top-level file, once left as written.
            if posixpath.normpath(match.group()) == 'top.txt':
                return match.group()
            return rooted(self, match)
        names = r'the export rewrote .* which does not name'
        for name, attribute, wrong, message in (
                ('rooted', 'rooted', first_components, f'qual.json: {names}'),
                ('climb', 'climb', one_climb_short, rf'[a-z/0-9]+\.rs: {names}'),
                ('second pass', 'text', second_pass, r'[a-z/]+\.rs: the exported file does not hold the rewrite'),
                ('kept', 'rooted', kept_at_top, "qual.json: the export kept 'fixtures/m3/../../top.txt', which does not name")):
            with self.subTest(name), patch.object(model_export.Relocation, attribute, wrong):
                with self.assertRaisesRegex(PolicyError, message):
                    self.export(name.replace(' ', '-'), model='models/m3')

    def test_model_exports_refuse_other_paths_templates_and_layout_collisions(self):
        for name in ('m3', 'models/m3/bundle', 'models/../m3', 'models/absent', 'crates/shared'):
            with self.subTest(model=name), self.assertRaises(PolicyError):
                self.export(name.replace('/', '-'), model=name)
        with self.assertRaisesRegex(PolicyError, 'not both'):
            self.export('both', model='models/m3', template=self.template)
        (self.root / 'models/m3/Cargo.toml').write_text('[workspace]\n')
        source.git(self.root, 'add', 'models/m3/Cargo.toml')
        with self.assertRaisesRegex(PolicyError, 'holds Cargo.toml'):
            self.export('collision', model='models/m3', development=True)


# Every legal static argument form of an include macro and of #[path]
# (review probes): each names compiler/src/top.*, read from its own file, or
# climbs to the checkout's top.* and is relocated into sdk/.
INCLUDE_FORMS = {
    'normal': 'include_str!("fixtures/probe/../../top.txt")',
    'raw': 'include_str!(r"fixtures/probe/../../top.txt")',
    'hashed': 'include_str!(r##"fixtures/probe/../../top.txt"##)',
    'bytes': 'include_bytes!(r#"fixtures/probe/../../top.txt"#)',
    'braces': 'include_str! { "fixtures/probe/../../top.txt" }',
    'brackets': 'include_str!["fixtures/probe/../../top.txt"]',
    'space': 'include_str! ("fixtures/probe/../../top.txt")',
    'comment': 'include_str!(/* static input */ "fixtures/probe/../../top.txt" // trailing\n)',
    'concat': 'include_str!(concat!("fixtures/probe/../../top.txt"))',
    'qualified': 'core::include_str!("fixtures/probe/../../top.txt")',
    'climb': 'include_str! { "../../../../top.txt" }',
    'raw-identifier': 'r#include_str!("fixtures/probe/../../top.txt")',
    'raw-concat': 'include_str!(r#concat!("fixtures/probe/../../top.txt"))',
    # Joined, this climbs to the checkout's top.txt; its first piece alone
    # would name the model root's.
    'split-concat': 'include_str!(concat!("../../../../", "top.txt"))',
}
# Forms that read the checkout's top.*, relocated into sdk/.
CLIMBS = ('climb', 'split-concat')
PATH_FORMS = {
    'normal': '#[path = "fixtures/probe/../../top.rs"]',
    'raw': '#[path = r"fixtures/probe/../../top.rs"]',
    'hashed': '#[path = r##"fixtures/probe/../../top.rs"##]',
    'space': '# [ path = "fixtures/probe/../../top.rs" ]',
    'comment': '#[path /* static input */ = "fixtures/probe/../../top.rs"]',
    'cfg-attr': '#[cfg_attr(all(), path = "fixtures/probe/../../top.rs")]',
    'nested-cfg-attr': '#[cfg_attr(all(), cfg_attr(any(not(any()), all()), path = "fixtures/probe/../../top.rs"))]',
    'climb': '#[path = "../../../../top.rs"]',
    'raw-identifier': '#[r#path = "fixtures/probe/../../top.rs"]',
}


@unittest.skipUnless(shutil.which('cargo'), 'needs cargo')
class ExportPathFormTests(unittest.TestCase):
    """A model crate using every static include and #[path] form passes its
    test in the checkout and, exported, in the workspace."""
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name).resolve()

    def tearDown(self):
        writable(self.base)
        self.temporary.cleanup()

    def checkout(self, lib):
        checkout = self.base / 'checkout'
        files = {
            'Cargo.toml': '[workspace]\nmembers = ["models/probe/compiler"]\nresolver = "3"\n',
            'Cargo.lock': 'version = 4\n\n[[package]]\nname = "probe-forms"\nversion = "0.1.0"\n',
            'models/probe/model.toml': '[model]\nname = "probe"\n',
            'models/probe/compiler/Cargo.toml':
                '[package]\nname = "probe-forms"\nversion = "0.1.0"\nedition = "2024"\n[lib]\ndoctest = false\n',
            'models/probe/compiler/src/fixtures/probe/placeholder': '',
            'models/probe/compiler/src/top.txt': 'model-source',
            'models/probe/compiler/src/top.rs': 'pub const DATA: &str = "model-source";\n',
            'top.txt': 'checkout-data',
            'top.rs': 'pub const DATA: &str = "checkout-data";\n',
            # Files at the paths a wrong rewrite would name, read from the
            # crate or without the inline module's directory: a wrong export
            # would build and read them silently.
            'models/probe/compiler/src/sdk/top.txt': 'shadow-source',
            'models/probe/compiler/src/sdk/top.rs': 'pub const DATA: &str = "shadow-source";\n',
            'models/probe/compiler/sdk/models/top.rs': 'pub const DATA: &str = "shadow-source";\n',
            # The root of the workspace, where a wrongly kept climb lands.
            'models/probe/top.txt': 'model-root-data',
            'crates/shared/data.txt': 'shared-data',
            'tools/dev/check.py': '',
            'models/probe/compiler/inputs/placeholder': '',
            'models/probe/compiler/src/lib.rs': lib,
        }
        for relative, contents in files.items():
            (checkout / relative).parent.mkdir(parents=True, exist_ok=True)
            (checkout / relative).write_text(contents)
        shutil.copyfile(ROOT / 'rust-toolchain.toml', checkout / 'rust-toolchain.toml')
        return checkout

    def cargo_test(self, workspace):
        env = dict(os.environ, CARGO_TARGET_DIR=str(self.base / 'target' / workspace.name))
        result = subprocess.run(['cargo', 'test', '--locked', '--offline', '--manifest-path',
                                 str(workspace / 'Cargo.toml')], cwd=workspace, env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        self.assertEqual(result.returncode, 0, result.stdout)

    def test_every_static_include_and_path_form_reads_the_same_file_after_export(self):
        lib, checks = '', ''
        for index, (name, expression) in enumerate(INCLUDE_FORMS.items()):
            expected = 'checkout-data' if name in CLIMBS else 'model-source'
            value = f'core::str::from_utf8({expression}).unwrap()' if 'bytes' in expression else expression
            lib += f'// {name}\npub fn include_{index}() -> &\'static str {{ {value} }}\n'
            checks += f'    assert_eq!(include_{index}(), "{expected}");\n'
        for index, (name, attribute) in enumerate(PATH_FORMS.items()):
            expected = 'checkout-data' if name in CLIMBS else 'model-source'
            lib += f'// {name}\n{attribute}\npub mod path_{index};\n'
            checks += f'    assert_eq!(path_{index}::DATA, "{expected}");\n'
        # An include inside an inline module still reads from its file.
        lib += 'pub mod inline {\n    include!("fixtures/probe/../../top.rs");\n}\n'
        checks += '    assert_eq!(inline::DATA, "model-source");\n'
        # A checkout-rooted literal in a file that joins its crate up to the
        # checkout root reads from that root, before and after export.
        lib += ('pub fn shared() -> std::path::PathBuf {\n'
                '    let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../..");\n'
                '    // A starts_with on a value this function joins to the root.\n'
                '    let path = std::path::Path::new("crates/shared/data.txt");\n'
                '    assert!(path.starts_with("crates/shared/"));\n'
                '    root.join(path)\n}\n')
        checks += '    assert_eq!(std::fs::read_to_string(shared()).unwrap(), "shared-data");\n'
        # Joins whose base the text establishes, with each spelling the
        # export normalizes; every one names the checkout's top.txt.
        lib += ('fn read(path: std::path::PathBuf) -> String { std::fs::read_to_string(path).unwrap() }\n'
                'fn from_parameter(base: &std::path::Path) -> String { read(base.join("../../../top.txt")) }\n'
                'pub fn joins() -> Vec<String> {\n'
                '    let manifest = std::path::PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());\n'
                '    let inputs = manifest.join("inputs");\n'
                '    vec![\n'
                '        read(manifest.join("./../../../top.txt")),\n'
                '        read(manifest.join("inputs/../../../../top.txt")),\n'
                '        read(manifest.join("\\x2e\\x2e/\\u{2e}\\u{2e}/\\x2E\\x2e/top.txt")),\n'
                '        read(inputs.join("../../../../top.txt")),\n'
                '        from_parameter(&manifest),\n'
                '        include_str!("\\x2e\\x2e/\\x2e\\x2e/\\x2e\\x2e/\\x2e\\x2e/top.txt").to_owned(),\n'
                '    ]\n}\n')
        checks += '    assert_eq!(joins(), vec!["checkout-data"; 6]);\n'
        lib += '#[test]\nfn original_content() {\n' + checks + '}\n'
        checkout = self.checkout(lib)
        self.cargo_test(checkout)
        workspace = self.base / 'export'
        shutil.copytree(checkout, workspace / 'sdk')
        with contextlib.redirect_stderr(io.StringIO()):
            receipt = model_export.export_model(workspace / 'sdk', workspace, 'models/probe')
        shutil.copyfile(checkout / 'Cargo.lock', workspace / 'Cargo.lock')
        self.assertEqual(receipt['unresolved'], [])
        exported = (workspace / 'compiler/src/lib.rs').read_text()
        self.assertNotIn('"sdk/top.', exported)
        # The split concat! becomes one literal; with the escaped include and
        # the derived base's join, five literals name sdk/top.* from src/.
        self.assertEqual(exported.count('"../../sdk/top.'), 5)
        self.assertIn('include_str!("../../sdk/top.txt")', exported)
        # Every form but the climbs (one of them a #[path]), and the inline include.
        self.assertEqual(exported.count('fixtures/probe/../../top.'),
                         len(INCLUDE_FORMS) + len(PATH_FORMS) - len(CLIMBS) - 1 + 1)
        self.assertEqual(exported.count('"sdk/crates/shared/data.txt"'), 1)
        self.assertEqual(exported.count('"sdk/crates/shared/"'), 1)
        # Normalized and decoded paths become plain relocated literals.
        self.assertEqual(exported.count('manifest.join("../sdk/top.txt")'), 3)
        self.assertIn('inputs.join("../../sdk/top.txt")', exported)
        self.assertIn('base.join("../sdk/top.txt")', exported)
        self.assertIn('include_str!("../../sdk/top.txt").to_owned()', exported)
        self.cargo_test(workspace)

    def test_paths_outside_the_recognized_forms_fail_the_export(self):
        refusals = {
            # A checkout-rooted literal in a file that joins no path to the
            # checkout root: nothing says what it is read from.
            'pub fn data() -> String { std::fs::read_to_string("crates/shared/data.txt").unwrap() }\n':
                r'lib\.rs:1: "crates/shared/data\.txt" names crates/shared/data\.txt from the checkout root, but '
                'this file joins no path to the checkout root',
            # A climb out of the model directory in no recognized form.
            'pub fn top() -> String {\n    std::fs::read_to_string("../../../top.txt").unwrap()\n}\n':
                r'lib\.rs:2: "\.\./\.\./\.\./top\.txt" climbs to top\.txt in the checkout',
            # A concat! mixing literals with another macro.
            'pub const P: &str = concat!(env!("OUT_DIR"), "../../../top.txt");\n':
                r'lib\.rs:1: "\.\./\.\./\.\./top\.txt" climbs to top\.txt',
            # A decoded climb in no recognized form.
            'pub fn top() -> String {\n    std::fs::read_to_string("\\x2e./../../top.txt").unwrap()\n}\n':
                r'lib\.rs:2: .* climbs to top\.txt in the checkout',
            # A join on a receiver the text does not establish: a parameter
            # one caller passes another directory, a field, a call.
            'fn read(base: &std::path::Path) -> std::path::PathBuf { base.join("../../../top.txt") }\n'
            'pub fn top() -> std::path::PathBuf { read(std::path::Path::new("/tmp")) }\n':
                r'lib\.rs:1: "\.\./\.\./\.\./top\.txt": climbs, but the base of this call could not be '
                'established from the text',
            'pub struct Dirs { pub inputs: std::path::PathBuf }\n'
            'pub fn top(dirs: &Dirs) -> std::path::PathBuf { dirs.inputs.join("../../../../top.txt") }\n':
                r'lib\.rs:2: .*: climbs, but the base of this call could not be established',
            # A build script or test may climb inside the model from such a
            # receiver; other sources may not.
            'pub fn src(dirs: &Dirs) -> std::path::PathBuf { dirs.root.join("../src") }\n':
                r'lib\.rs:1: "\.\./src": climbs, but the base of this call could not be established',
            # A checkout-rooted literal joined to a base other than the root,
            # in a file that also joins its crate up to the root.
            'pub fn shared() -> std::path::PathBuf {\n'
            '    let manifest = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));\n'
            '    let _root = manifest.join("../../..");\n'
            '    manifest.join("crates/shared/data.txt")\n}\n':
                r'lib\.rs:4: "crates/shared/data\.txt": names crates/shared/data\.txt from the checkout root, but '
                'the receiver is not the checkout root',
            'pub fn shared(dirs: &Dirs) -> std::path::PathBuf { dirs.root.join("crates/shared/data.txt") }\n':
                r'lib\.rs:1: .*: names crates/shared/data\.txt from the checkout root, but the base of this call',
            # A starts_with whose receiver is neither the root nor joined to it.
            'pub fn shared(path: &std::path::Path) -> bool { path.starts_with("crates/shared/") }\n':
                r'lib\.rs:1: "crates/shared/": names crates/shared from the checkout root, but the base',
        }
        for index, (lib, message) in enumerate(refusals.items()):
            with self.subTest(lib=lib):
                checkout = self.checkout(lib)
                workspace = self.base / f'export-{index}'
                shutil.copytree(checkout, workspace / 'sdk')
                with self.assertRaisesRegex(PolicyError, message):
                    model_export.export_model(workspace / 'sdk', workspace, 'models/probe')
                shutil.rmtree(checkout)
        # The same climb inside the model from an unestablished receiver is
        # kept in a test of the crate.
        checkout = self.checkout('')
        test = checkout / 'models/probe/compiler/tests/inside.rs'
        test.parent.mkdir()
        test.write_text('pub fn src(dirs: &Dirs) -> std::path::PathBuf { dirs.root.join("../src") }\n')
        workspace = self.base / 'export-test'
        shutil.copytree(checkout, workspace / 'sdk')
        model_export.export_model(workspace / 'sdk', workspace, 'models/probe')
        self.assertEqual((workspace / 'compiler/tests/inside.rs').read_text(), test.read_text())
        shutil.rmtree(checkout)
        # A rooted literal naming in-tree tooling the SDK surface does not
        # include is kept and reported, as is a path naming nothing.
        checkout = self.checkout('pub const TOOL: &str = "tools/dev/check.py";\n'
                                 'pub const NOTHING: &str = "crates/absent/x.txt";\n')
        workspace = self.base / 'export-kept'
        shutil.copytree(checkout, workspace / 'sdk')
        warnings = io.StringIO()
        with contextlib.redirect_stderr(warnings):
            receipt = model_export.export_model(workspace / 'sdk', workspace, 'models/probe')
        self.assertEqual(receipt['unresolved'], [
            {'file': 'compiler/src/lib.rs', 'line': 2, 'reference': 'crates/absent/x.txt',
             'read_from': 'checkout root'},
            {'file': 'compiler/src/lib.rs', 'line': 1, 'reference': 'tools/dev/check.py',
             'read_from': 'checkout root', 'reason': model_export.TOOLING}])
        self.assertIn("compiler/src/lib.rs:1: kept 'tools/dev/check.py' (read from checkout root), which names "
                      'checkout tooling the export does not include', warnings.getvalue())
        self.assertEqual((workspace / 'compiler/src/lib.rs').read_text(), (checkout / 'models/probe/compiler/src/lib.rs').read_text())

    def test_a_path_attribute_inside_an_inline_module_fails_the_export(self):
        # Rust reads this from compiler/src/inner, so it names the model's
        # top.rs; a rewrite read without the module directory would name
        # the shadow file in compiler/sdk/models.
        checkout = self.checkout('pub mod inner {\n    #[path = "fixtures/probe/../../../../../top.rs"]\n'
                                 '    pub mod data;\n}\n'
                                 '#[test]\nfn original_content() { assert_eq!(inner::data::DATA, "model-source"); }\n')
        (checkout / 'models/probe/compiler/src/inner/fixtures/probe').mkdir(parents=True)
        (checkout / 'models/probe/top.rs').write_text('pub const DATA: &str = "model-source";\n')
        self.cargo_test(checkout)
        workspace = self.base / 'export'
        shutil.copytree(checkout, workspace / 'sdk')
        message = r"compiler/src/lib\.rs:2: a #\[path\] attribute inside an inline module reads from the module's"
        with self.assertRaisesRegex(PolicyError, message):
            model_export.export_model(workspace / 'sdk', workspace, 'models/probe')


class RunnerTests(unittest.TestCase):
    def native_graph(self, owner):
        return Graph({'packages': [{'id':owner, 'name':owner}, {'id':'kernel-build', 'name':'kernel-build'}],
                      'resolve': {'nodes': [
                          {'id':owner, 'deps':[{'pkg':'kernel-build', 'dep_kinds':[{'kind':'build'}]}]},
                          {'id':'kernel-build', 'deps':[]}]}})

    def test_portable_build_does_not_query_nvcc(self):
        graph = Graph({'packages':[{'id':'compiler','name':'compiler'}],
                       'resolve':{'nodes':[{'id':'compiler','deps':[]}]}})
        with patch.object(workflow, 'run') as command:
            self.assertEqual(workflow.native_toolchain(graph, ['compiler'], {}),
                             {'cuda_build_owners':[], 'nvcc':None})
            command.assert_not_called()

    def test_older_sdk_and_explicit_probe_record_native_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'bin').mkdir()
            (root/'bin/nvcc').write_bytes(b'fixture executable')
            for owner in ['runtime-kernels', 'example-bundle']:
                with self.subTest(owner=owner), patch.object(workflow, 'run', return_value='Cuda V13.2.86') as command:
                    result = workflow.native_toolchain(self.native_graph(owner), [owner], {'CHOPINFER_CUDA_HOME':directory})
                    self.assertEqual(result['cuda_build_owners'], [owner])
                    self.assertEqual(result['nvcc']['sha256'], source.sha(root/'bin/nvcc'))
                    command.assert_called_once_with([root/'bin/nvcc', '--version'])

    def test_native_build_reports_missing_or_wrong_toolchain(self):
        graph = self.native_graph('native-helper')
        with tempfile.TemporaryDirectory() as directory:
            env = {'CHOPINFER_CUDA_HOME':directory}
            with self.assertRaisesRegex(PolicyError, 'CUDA 13.2.86.*native-helper'):
                workflow.native_toolchain(graph, ['native-helper'], env)
            root = Path(directory)
            (root/'bin').mkdir()
            (root/'bin/nvcc').touch()
            with patch.object(workflow, 'run', return_value='Cuda V13.1.0'):
                with self.assertRaisesRegex(PolicyError, 'requires nvcc 13.2.86'):
                    workflow.native_toolchain(graph, ['native-helper'], env)

    def test_update_and_profiling_bypasses_are_refused(self):
        for name in workflow.BYPASSES:
            with self.subTest(name=name), patch.dict(os.environ, {name:'1'}):
                with self.assertRaisesRegex(PolicyError, name):
                    workflow.environment()

    def test_bad_lock_reports_environment_without_repair(self):
        failed = subprocess.CompletedProcess(['cargo'], 1, '', 'kernels.lock mismatch; rerun with KERNELS_LOCK_UPDATE=1 to accept')
        with patch.object(workflow.subprocess, 'run', return_value=failed) as invoked:
            with self.assertRaisesRegex(PolicyError, 'do not rewrite SDK locks') as failure:
                workflow.run(['cargo', 'build', '--locked', '--offline'])
            self.assertEqual(invoked.call_count, 1)
            self.assertNotIn('rerun with KERNELS_LOCK_UPDATE', str(failure.exception))
            self.assertIn(workflow.TOOLCHAIN_IMAGE, str(failure.exception))
            self.assertIn('model-owned lock change with `relock`', str(failure.exception))

    @contextmanager
    def sdk_build(self, root, cargo):
        """`workflow.build` with every step but cargo stubbed, for a model whose
        roles are a bundle cdylib and a compiler bin."""
        model = SimpleNamespace(packages={
            'bundle': inventory.Package('m', 'bundle', 'example-bundle', frozenset({('example_bundle', 'cdylib')}), '.'),
            'compiler': inventory.Package('m', 'compiler', 'example-compiler', frozenset({('example-compile', 'bin')}), '.'),
            'contract': inventory.Package('m', 'contract', 'example-contract', frozenset(), '.')})
        (root / 'sdk').mkdir(parents=True)
        (root / 'Cargo.lock').write_text('')
        with patch.object(workflow.subprocess, 'check_output', return_value='rustc 1.89.0 (fixture)'), \
             patch.object(workflow, 'metadata'), patch.object(workflow, 'selected_graph', return_value=(None, [])) as graph, \
             patch.object(workflow, 'check', return_value={}), patch.object(workflow, 'verify_source'), \
             patch.object(workflow.inventory, 'load_model', return_value=model), \
             patch.object(workflow, 'native_toolchain', return_value={}), patch.object(workflow, 'run', side_effect=cargo):
            yield graph

    def test_builds_force_every_bypass_empty_through_cli_config(self):
        # The model's `.cargo/config.toml` [env] could otherwise reach build scripts.
        self.assertEqual(forced_env(workflow.cargo_config()), {k: '' for k in workflow.BYPASSES})
        self.assertEqual(forced_env(workflow.cargo_config('/tmp/c "d"'))[workflow.RELOCK_ENV], '/tmp/c "d"')
        commands = []
        def cargo(command, **kwargs):
            commands.append(list(map(str, command)))
            raise PolicyError('stop after the first cargo command')
        with tempfile.TemporaryDirectory() as directory:
            with self.sdk_build(Path(directory) / 'model', cargo):
                with self.assertRaisesRegex(PolicyError, 'stop after'):
                    workflow.build(Path(directory) / 'model', Path(directory) / 'target', Path(directory) / 'logs')
        self.assertEqual(forced_env(commands[0]), {k: '' for k in workflow.BYPASSES})

    def test_a_build_selects_every_role_and_returns_only_artifacts_it_produced(self):
        # Workspace default members could otherwise drop the bundle, and a
        # bundle an earlier build left in the target would be returned.
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory).resolve()
            root, target = base / 'model', base / 'target'
            stale = target / 'debug/libexample_bundle.so'
            stale.parent.mkdir(parents=True)
            stale.write_bytes(b'stale bundle')
            compiler = target / 'debug/example-compile'
            compiler.write_bytes(b'compiler')
            def artifact(manifest, name, kind, **fields):
                return json.dumps({'reason': 'compiler-artifact', 'manifest_path': str(root / manifest),
                                   'target': {'name': name, 'kind': [kind]}, 'fresh': True,
                                   'filenames': [], 'executable': None, **fields})
            produced = [artifact('compiler/Cargo.toml', 'example-compile', 'bin', executable=str(compiler))]
            commands = []
            def cargo(command, **kwargs):
                commands.append(list(map(str, command)))
                # Another invocation on this workspace or target is refused while cargo runs.
                for held in (workflow.workspace_lock(root), workflow.target_lock(target)):
                    with self.assertRaisesRegex(PolicyError, f'pid {os.getpid()}'):
                        with held:
                            pass
                return '\n'.join(produced) if command[1] == 'build' else ''
            with self.sdk_build(root, cargo) as graph:
                with self.assertRaisesRegex(PolicyError, 'produced no bundle cdylib example_bundle'):
                    workflow.build(root, target, base / 'logs')
                roles = ['example-bundle', 'example-compiler', 'example-contract']
                self.assertEqual(graph.call_args.kwargs['packages'], roles)
                self.assertEqual([selected_packages(c) for c in commands], [roles, roles])
                # An SDK crate's target of the same name is not the role's.
                produced.append(artifact('sdk/crates/x/Cargo.toml', 'example_bundle', 'cdylib', filenames=[str(stale)]))
                with self.assertRaisesRegex(PolicyError, 'produced no bundle cdylib'):
                    workflow.build(root, target, base / 'logs')
                produced.append(artifact('bundle/Cargo.toml', 'example_bundle', 'cdylib',
                                         filenames=[str(target / 'debug/libexample_bundle.rlib'), str(stale)]))
                receipt = workflow.build(root, target, base / 'logs')
            self.assertEqual(receipt['artifacts']['example_bundle']['path'], str(stale))
            self.assertEqual(receipt['artifacts']['example-compile']['sha256'], source.sha(compiler))

    def test_a_build_fails_on_any_report_its_build_scripts_left_unverified(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory).resolve() / 'model'
            lock = workspace / 'bundle/precision.lock'
            lock.parent.mkdir(parents=True)
            lock.write_text('[]\n')
            out = Path(directory) / 'debug/build/model-bundle-0/out'
            report = out / 'precision' / workflow.KERNEL_BUILD_REPORT
            report.parent.mkdir(parents=True)
            def write(status='Verified', path=lock, digest=None):
                report.write_text(json.dumps({'lock_status': status, 'lock_path': str(path),
                                              'lock_sha256': digest or source.sha(lock)}))
            write('Updated')
            # Old timestamps do not matter: a cached build script still consumed it.
            os.utime(report, ns=(0, 0))
            messages = '\n'.join(['Compiling bundle', json.dumps({'reason': 'compiler-artifact'}),
                                   json.dumps({'reason': 'build-script-executed', 'out_dir': str(out)}), '{not json'])
            self.assertEqual(workflow.build_script_outputs(messages), [out])
            with self.assertRaisesRegex(PolicyError, 'Updated, not Verified'):
                workflow.require_verified([out], workspace)
            with patch.object(workflow, 'run', return_value=messages) as cargo:
                with self.assertRaisesRegex(PolicyError, 'Updated, not Verified'):
                    workflow.verified_cargo(['cargo', 'build'], workspace, env={}, log=None)
                self.assertEqual(cargo.call_args.args[0][-1], '--message-format=json-render-diagnostics')
            write()
            workflow.require_verified([out], workspace)
            # Another workspace's verified lock, or this lock changed since, is no evidence.
            other = Path(directory).resolve() / 'other/bundle/precision.lock'
            other.parent.mkdir(parents=True)
            other.write_text('[]\n')
            write(path=other)
            with self.assertRaisesRegex(PolicyError, 'outside this workspace'):
                workflow.require_verified([out], workspace)
            write(digest='0' * 64)
            with self.assertRaisesRegex(PolicyError, 'changed since it was verified'):
                workflow.require_verified([out], workspace)
            # A directory-form lock names each family's file and digest.
            family = workspace / 'bundle/kernels.lock.d/row.lock'
            family.parent.mkdir()
            family.write_text('[]\n')
            families = [{'lock_path': str(family), 'lock_sha256': source.sha(family)}]
            report.write_text(json.dumps({'lock_status': 'Verified', 'lock_path': str(family.parent),
                                          'lock_sha256': None, 'lock_families': families}))
            workflow.require_verified([out], workspace)
            family.write_text('[{}]\n')
            with self.assertRaisesRegex(PolicyError, 'row.lock changed'):
                workflow.require_verified([out], workspace)

    def test_racing_workspaces_never_both_claim_a_fresh_target(self):
        import threading
        with tempfile.TemporaryDirectory() as directory:
            for trial in range(200):
                target = Path(directory) / f'target-{trial}'
                barrier, outcomes = threading.Barrier(2), []
                def claim(workspace):
                    barrier.wait()
                    try:
                        workflow.claim_target(target, Path(directory) / workspace)
                        outcomes.append(workspace)
                    except PolicyError:
                        pass
                threads = [threading.Thread(target=claim, args=(w,)) for w in ('a', 'b')]
                for thread in threads:
                    thread.start()
                for thread in threads:
                    thread.join()
                self.assertEqual(len(outcomes), 1, f'trial {trial}: {outcomes} both claimed {target}')
                self.assertEqual((target / workflow.TARGET_OWNER).read_text(), str(Path(directory) / outcomes[0]))
                self.assertEqual(sorted(p.name for p in target.iterdir()), [workflow.TARGET_OWNER])

    def test_a_second_process_is_refused_while_the_first_holds_the_target(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)
            holder = subprocess.Popen(
                [sys.executable, '-B', '-c',
                 'import sys; from pathlib import Path; from tools.model_sdk import workflow\n'
                 'with workflow.target_lock(Path(sys.argv[1])):\n'
                 '    print("held", flush=True); sys.stdin.readline()', str(target)],
                cwd=Path(__file__).resolve().parents[1], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
            try:
                self.assertEqual(holder.stdout.readline().strip(), 'held')
                with self.assertRaisesRegex(PolicyError, f'in use by another SDK build or relock \\(pid {holder.pid}\\)'):
                    with workflow.target_lock(target):
                        pass
            finally:
                holder.communicate('\n', timeout=30)
            self.assertEqual(holder.returncode, 0)
            with workflow.target_lock(target):
                self.assertEqual((target / workflow.SDK_LOCK).read_text(), str(os.getpid()))

    def test_a_target_directory_serves_one_workspace(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'target'
            workflow.claim_target(target, Path(directory) / 'a')
            workflow.claim_target(target, Path(directory) / 'a')
            with self.assertRaisesRegex(PolicyError, 'one target directory per workspace'):
                workflow.claim_target(target, Path(directory) / 'b')

    def test_outputs_are_outside_sources(self):
        for output in ['/tmp/model', '/tmp/model/target', '/tmp']:
            with self.subTest(output=output), self.assertRaisesRegex(PolicyError, 'unrelated'):
                workflow.external_output('/tmp/model', output)
        self.assertEqual(workflow.external_output('/tmp/model','/tmp/artifacts'), Path('/tmp/artifacts'))


def selected_packages(command):
    return [value for flag, value in zip(command, command[1:]) if flag == '--package']


def forced_env(command):
    """The `[env]` values a command forces through Cargo CLI configuration."""
    config = {}
    for flag, value in zip(command, command[1:]):
        if flag == '--config':
            for key, table in tomllib.loads(value).get('env', {}).items():
                config.setdefault(key, {}).update(table)
    # A configured `relative = true` would survive a value and force override.
    return {key: table['value'] for key, table in config.items()
            if table.get('force') is True and table.get('relative') is False}


def apply(targets):
    written = [t for t, c in targets.items() if relock.read_lock(t) != c.read_bytes()]
    for target in written:
        relock.write_lock(target, targets[target].read_bytes())
    return written


def lock_record(name, cubin='c0', closure='s0', **fields):
    return dict(name=name, line_info=False, flags=['-O3'], cubin_sha256=cubin,
                source_closure_sha256=closure) | fields


class RelockTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name).resolve()
        self.root = self.base / 'model'
        (self.root / 'bundle').mkdir(parents=True)
        (self.root / 'sdk/crates/kernel-lab').mkdir(parents=True)
        self.candidates = self.base / 'candidates'
        self.candidates.mkdir()
        self.out = self.base / 'target/debug/build/bundle-0/out'

    def tearDown(self):
        writable(self.base)
        self.temporary.cleanup()

    def write(self, path, records):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(records, indent=2) + '\n')

    def candidate(self, target, records):
        path = self.candidates / target.relative_to('/')
        self.write(path, records)
        return path

    def test_report_names_records_and_whether_cubins_or_provenance_changed(self):
        kernels, unchanged = self.root / 'bundle/kernels.lock', self.root / 'bundle/precision.lock'
        family = self.root / 'bundle/kernels.lock.d/row.lock'
        provenance = family.with_name('row.provenance.json')
        created = self.root / 'bundle/new.lock'
        self.write(kernels, [lock_record('same'), lock_record('cubin'), lock_record('moved'),
                             lock_record('recipe'), lock_record('gone')])
        self.candidate(kernels, [lock_record('same'), lock_record('cubin', cubin='c1', closure='s1'),
                                 lock_record('moved', closure='s1'), lock_record('recipe', flags=['-O2']),
                                 lock_record('added')])
        self.write(unchanged, [lock_record('precision')])
        self.candidate(unchanged, [lock_record('precision')])
        gated = [{k: v for k, v in lock_record('row').items() if k != 'source_closure_sha256'}]
        self.write(family, gated)
        self.candidate(family, gated)
        self.write(provenance, [{'name': 'row', 'source_closure_sha256': 's0'}])
        self.candidate(provenance, [{'name': 'row', 'source_closure_sha256': 's1'}])
        self.candidate(created, [lock_record('first')])
        before = unchanged.stat().st_mtime_ns

        (self.candidates / relock.MARKER).write_bytes(b'')
        targets = relock.lock_targets(self.root, self.candidates)
        self.assertNotIn(self.candidates / relock.MARKER, targets.values())
        locks = {entry['lock']: entry for entry in relock.review(self.root, targets)}
        self.assertEqual(sorted(locks), ['bundle/kernels.lock', 'bundle/kernels.lock.d/row.lock',
                                         'bundle/new.lock', 'bundle/precision.lock'])
        main = locks['bundle/kernels.lock']
        self.assertEqual((main['status'], main['added'], main['removed']), ('updated', ['added'], ['gone']))
        changes = {c['name']: c for c in main['changed']}
        self.assertEqual(sorted(changes), ['cubin', 'moved', 'recipe'])
        self.assertEqual(changes['cubin']['kind'], 'cubin')
        self.assertEqual(changes['cubin']['cubin_sha256'], {'old': 'c0', 'new': 'c1'})
        self.assertEqual(changes['cubin']['fields'], ['cubin_sha256', 'source_closure_sha256'])
        self.assertEqual((changes['moved']['kind'], changes['moved']['cubin_changed']), ('provenance', False))
        self.assertEqual((changes['recipe']['kind'], changes['recipe']['fields']), ('recipe', ['flags']))
        self.assertFalse(main['provenance_only'])
        row = locks['bundle/kernels.lock.d/row.lock']
        self.assertEqual(row['files'], ['bundle/kernels.lock.d/row.lock', 'bundle/kernels.lock.d/row.provenance.json'])
        self.assertTrue(row['provenance_only'])
        self.assertEqual([c['kind'] for c in row['changed']], ['provenance'])
        self.assertEqual((locks['bundle/new.lock']['status'], locks['bundle/new.lock']['added']), ('created', ['first']))
        same = locks['bundle/precision.lock']
        self.assertEqual((same['status'], same['added'], same['removed'], same['changed']), ('unchanged', [], [], []))

        written = apply(targets)
        self.assertEqual(sorted(p.relative_to(self.root).as_posix() for p in written),
                         ['bundle/kernels.lock', 'bundle/kernels.lock.d/row.provenance.json', 'bundle/new.lock'])
        for target, candidate in targets.items():
            self.assertEqual(target.read_bytes(), candidate.read_bytes())
        self.assertEqual(unchanged.stat().st_mtime_ns, before)
        self.assertEqual([e['status'] for e in relock.review(self.root, targets)], ['unchanged'] * 4)

    def test_provenance_sidecar_cannot_mask_a_gating_change(self):
        family = self.root / 'bundle/kernels.lock.d/tiny.lock'
        provenance = family.with_name('tiny.provenance.json')
        gated = lambda cubin: [{k: v for k, v in lock_record('tiny', cubin=cubin).items()
                                if k != 'source_closure_sha256'}]
        self.write(family, gated('c0'))
        # A stale sidecar carrying the new cubin digest must not overwrite the locked one.
        self.write(provenance, [{'name': 'tiny', 'source_closure_sha256': 's0', 'cubin_sha256': 'c1'}])
        self.candidate(family, gated('c1'))
        self.candidate(provenance, [{'name': 'tiny', 'source_closure_sha256': 's1'}])
        [entry] = relock.review(self.root, relock.lock_targets(self.root, self.candidates))
        [change] = entry['changed']
        self.assertEqual((change['kind'], change['cubin_changed']), ('cubin', True))
        self.assertEqual(change['fields'], ['cubin_sha256', 'source_closure_sha256'])
        self.assertEqual(change['cubin_sha256'], {'old': 'c0', 'new': 'c1'})
        self.assertFalse(entry['provenance_only'])
        self.write(provenance, [{'name': 'tiny'}])
        with self.assertRaisesRegex(PolicyError, 'invalid provenance record'):
            relock.review(self.root, relock.lock_targets(self.root, self.candidates))

    def test_sdk_and_foreign_locks_are_refused_before_any_write(self):
        model = self.root / 'bundle/kernels.lock'
        self.write(model, [lock_record('a')])
        self.candidate(model, [lock_record('a', cubin='c1')])
        sdk = self.root / 'sdk/crates/kernel-lab/kernels.lock'
        self.write(sdk, [lock_record('shared')])
        for foreign in [sdk, self.base / 'elsewhere/kernels.lock', self.root / 'sdk/new/kernels.lock']:
            with self.subTest(lock=foreign):
                candidate = self.candidate(foreign, [lock_record('shared', cubin='c9')])
                with self.assertRaisesRegex(PolicyError, 'outside the model'):
                    relock.lock_targets(self.root, self.candidates)
                candidate.unlink()
        (self.root / 'linked').symlink_to(self.root / 'sdk/crates/kernel-lab')
        candidate = self.candidate(self.root / 'linked/kernels.lock', [lock_record('shared', cubin='c9')])
        with self.assertRaisesRegex(PolicyError, 'symlink'):
            relock.lock_targets(self.root, self.candidates)
        candidate.unlink()
        self.assertEqual(json.loads(sdk.read_text()), [lock_record('shared')])
        self.assertEqual(json.loads(model.read_text()), [lock_record('a')])
        self.assertEqual(list(relock.lock_targets(self.root, self.candidates)), [model])

    def test_a_parent_swapped_for_a_symlink_after_validation_is_refused(self):
        lock = self.root / 'bundle/kernels.lock'
        self.write(lock, [lock_record('a')])
        self.candidate(lock, [lock_record('a', cubin='c1')])
        targets = relock.lock_targets(self.root, self.candidates)
        outside = self.base / 'outside'
        outside.mkdir()
        (outside / 'kernels.lock').write_text('outside\n')
        (self.root / 'bundle').rename(self.base / 'moved-bundle')
        (self.root / 'bundle').symlink_to(outside)
        for operation in (lambda: apply(targets), lambda: relock.review(self.root, targets)):
            with self.assertRaisesRegex(PolicyError, 'symlink'):
                operation()
        self.assertEqual((outside / 'kernels.lock').read_text(), 'outside\n')
        self.assertEqual(sorted(p.name for p in outside.iterdir()), ['kernels.lock'])
        (self.root / 'bundle').unlink()
        (self.base / 'moved-bundle').rename(self.root / 'bundle')
        lock.unlink()
        lock.symlink_to(outside / 'kernels.lock')
        with self.assertRaisesRegex(PolicyError, 'symlink'):
            apply(targets)
        self.assertEqual((outside / 'kernels.lock').read_text(), 'outside\n')
        lock.unlink()
        previous = os.umask(0o022)
        try:
            self.assertEqual(apply(targets), [lock])
        finally:
            os.umask(previous)
        self.assertEqual(lock.stat().st_mode & 0o777, 0o644)
        self.assertEqual(sorted(p.name for p in lock.parent.iterdir()), ['kernels.lock'])

    def candidate_dir(self, command, env):
        """The candidate directory a relock build passed, or None for a verifying build."""
        return forced_env(command)[relock.RELOCK_ENV] or None

    def cargo(self, locks, calls, fail=None):
        """A fake cargo: a candidate build writes `locks`; `fail(call)` may raise.
        Each call reports the build script output `self.out` as cargo's JSON does."""
        def run(command, *, env, cwd, log):
            calls.append((list(map(str, command)), dict(env)))
            with self.assertRaisesRegex(PolicyError, f'pid {os.getpid()}'):
                with workflow.target_lock(self.base / 'target'):
                    pass
            if fail:
                fail(len(calls))
            directory = self.candidate_dir(calls[-1][0], env)
            if directory:
                # kernel_build::ModelLock refuses a directory without the marker.
                assert (Path(directory) / relock.MARKER).is_file()
                for lock, records in locks.items():
                    self.write(Path(directory) / lock.relative_to('/'), records)
            return json.dumps({'reason': 'build-script-executed', 'out_dir': str(self.out)}) + '\n'
        return run

    @contextmanager
    def sdk_workflow(self, cargo):
        pin = {'schema': source.PIN, 'revision': 'r', 'files': {}}
        for path in [*(self.root / 'sdk').rglob('*'), self.root / 'sdk']:
            path.chmod(path.stat().st_mode & ~0o222)
        with patch.object(workflow.subprocess, 'check_output', return_value='rustc 1.89.0 (fixture)'), \
             patch.object(relock, 'metadata'), patch.object(relock, 'selected_graph', return_value=(None, [])), \
             patch.object(relock, 'check', return_value=pin), patch.object(relock, 'verify_source'), \
             patch.object(relock, 'native_toolchain', return_value={'nvcc': None}), \
             patch.object(relock, 'run', side_effect=cargo), patch.object(workflow, 'run', side_effect=cargo), \
             patch.object(relock, 'role_packages', return_value=['example-bundle', 'example-compiler']):
            yield

    def test_relock_builds_candidates_once_then_verifies_without_bypasses(self):
        lock = self.root / 'bundle/kernels.lock'
        self.write(lock, [lock_record('a')])
        calls = []
        with self.sdk_workflow(self.cargo({lock: [lock_record('a', cubin='c1')]}, calls)):
            for name in workflow.BYPASSES:
                with self.subTest(name=name), patch.dict(os.environ, {name: '1'}):
                    with self.assertRaisesRegex(PolicyError, name):
                        relock.relock(self.root, self.base / 'target', self.base / 'logs')
            self.assertEqual(calls, [])
            report = relock.relock(self.root, self.base / 'target', self.base / 'logs', features=['cuda-probe'])
        self.assertEqual(len(calls), 2)
        self.assertTrue(self.candidate_dir(*calls[0]))
        self.assertIsNone(self.candidate_dir(*calls[1]))
        self.assertTrue(all('KERNELS_LOCK_UPDATE' not in env and relock.RELOCK_ENV not in env for _, env in calls))
        for command, _ in calls:
            self.assertEqual(selected_packages(command), ['example-bundle', 'example-compiler'])
            forced = forced_env(command)
            self.assertEqual(sorted(forced), sorted(workflow.BYPASSES))
            self.assertTrue(all(forced[k] == '' for k in workflow.BYPASSES if k != relock.RELOCK_ENV))
        self.assertEqual((report['status'], report['written']), ('applied', ['bundle/kernels.lock']))
        self.assertEqual(report['locks'][0]['changed'][0]['kind'], 'cubin')
        self.assertEqual(json.loads((self.base / 'logs' / relock.REPORT).read_text()), report)
        self.assertEqual(json.loads(lock.read_text()), [lock_record('a', cubin='c1')])
        self.assertEqual([p.name for p in (self.base / 'logs').iterdir()], [relock.REPORT])

    def failing_relock(self, fail, write=None):
        """Relock two locks, one new, with `fail` injected; returns the locks and logs."""
        kernels, created = self.root / 'bundle/kernels.lock', self.root / 'bundle/new/created.lock'
        self.write(kernels, [lock_record('a')])
        locks = {kernels: [lock_record('a', cubin='c1')], created: [lock_record('b')]}
        logs = self.base / 'logs'
        with self.sdk_workflow(self.cargo(locks, [], fail)), \
             patch.object(relock, 'write_lock', side_effect=write or relock.write_lock):
            with self.assertRaises(BaseException) as failure:
                relock.relock(self.root, self.base / 'target', logs)
        return kernels, created, logs, failure.exception

    def assert_restored(self, kernels, created, logs, error):
        self.assertIsInstance(error, PolicyError)
        self.assertIn('restored every lock', str(error))
        self.assertEqual(json.loads(kernels.read_text()), [lock_record('a')])
        self.assertFalse(created.exists())
        self.assertEqual([p.name for p in kernels.parent.rglob('*') if p.is_file()], ['kernels.lock'])
        report = json.loads((logs / relock.REPORT).read_text())
        self.assertEqual(report['status'], 'restored')
        self.assertEqual(report['written'], ['bundle/kernels.lock', 'bundle/new/created.lock'])
        kept = Path(report['candidates'])
        self.assertEqual(json.loads((kept / 'bundle/kernels.lock').read_text()), [lock_record('a', cubin='c1')])
        self.assertTrue((kept / 'bundle/new/created.lock').is_file())
        self.assertFalse((logs / relock.STATE).exists())

    def test_a_failed_replacement_restores_every_old_lock(self):
        original, writes = relock.write_lock, []
        def write(target, data):
            writes.append(target)
            if len(writes) == 2:
                raise OSError('injected second replacement failure')
            original(target, data)
        self.assert_restored(*self.failing_relock(None, write))

    def test_a_failed_verification_build_restores_every_old_lock(self):
        def fail(call):
            if call == 2:
                raise PolicyError('injected verification failure')
        kernels, created, logs, error = self.failing_relock(fail)
        self.assert_restored(kernels, created, logs, error)
        self.assertIn('injected verification failure', json.loads((logs / relock.REPORT).read_text())['error'])

    def test_a_verification_that_writes_no_verified_report_restores_every_old_lock(self):
        def fail(call):
            if call == 2:
                report = self.out / workflow.KERNEL_BUILD_REPORT
                report.parent.mkdir(parents=True, exist_ok=True)
                report.write_text(json.dumps({'lock_status': 'Updated'}))
        kernels, created, logs, error = self.failing_relock(fail)
        self.assert_restored(kernels, created, logs, error)
        self.assertIn('not Verified', json.loads((logs / relock.REPORT).read_text())['error'])

    def test_an_interrupted_verification_restores_then_reraises(self):
        def fail(call):
            if call == 2:
                raise KeyboardInterrupt
        kernels, created, logs, error = self.failing_relock(fail)
        self.assertIsInstance(error, KeyboardInterrupt)
        self.assertEqual(json.loads(kernels.read_text()), [lock_record('a')])
        self.assertFalse(created.exists())
        self.assertEqual(json.loads((logs / relock.REPORT).read_text())['status'], 'restored')

    def test_a_failure_writing_the_final_report_restores_every_old_lock(self):
        original = relock.publish_report
        def publish(logs, report):
            if report['status'] == 'applied':
                raise OSError('injected final report failure')
            original(logs, report)
        with patch.object(relock, 'publish_report', side_effect=publish):
            kernels, created, logs, error = self.failing_relock(None)
        self.assert_restored(kernels, created, logs, error)
        self.assertIn('injected final report failure', json.loads((logs / relock.REPORT).read_text())['error'])
        self.assertEqual(sorted(p.name for p in logs.iterdir() if p.name.startswith('.')), [])

    def interrupted_after_applying(self, point):
        """Relock with a KeyboardInterrupt at `point` of the cleanup after the
        applied report; then a second relock must finish it and run."""
        kernels = self.root / 'bundle/kernels.lock'
        self.write(kernels, [lock_record('a')])
        locks = {kernels: [lock_record('a', cubin='c1')]}
        logs = self.base / 'logs'
        interrupted = []
        def interrupt(*args, **kwargs):
            retiring = point == 'rename' or Path(args[0]).name.startswith(relock.RETIRED_PREFIX)
            if retiring and not interrupted:
                interrupted.append(args)
                raise KeyboardInterrupt
            return original(*args, **kwargs)
        original = relock.retire if point == 'rename' else shutil.rmtree
        target = (relock, 'retire') if point == 'rename' else (relock.shutil, 'rmtree')
        with self.sdk_workflow(self.cargo(locks, [])), patch.object(*target, side_effect=interrupt):
            with self.assertRaises(KeyboardInterrupt):
                relock.relock(self.root, self.base / 'target', logs)
        # The locks are applied and the report says so, with its cleanup unfinished.
        self.assertEqual(json.loads(kernels.read_text()), [lock_record('a', cubin='c1')])
        report = json.loads((logs / relock.REPORT).read_text())
        self.assertEqual((report['status'], report['written']), ('applied', ['bundle/kernels.lock']))
        leftovers = sorted(p.name for p in logs.iterdir() if p.name.startswith(('relock-state', 'relock-applied-')))
        self.assertEqual(len(leftovers), 1)
        with self.sdk_workflow(self.cargo(locks, [])):
            again = relock.relock(self.root, self.base / 'target', logs)
        self.assertEqual((again['status'], again['written']), ('applied', []))
        self.assertEqual(sorted(p.name for p in logs.iterdir() if p.is_dir()), [])
        return leftovers[0]

    def test_an_interrupt_after_the_final_report_leaves_state_the_next_run_retires(self):
        self.assertEqual(self.interrupted_after_applying('rename'), relock.STATE)

    def test_an_interrupt_during_deletion_leaves_a_retired_directory_the_next_run_removes(self):
        self.assertTrue(self.interrupted_after_applying('delete').startswith(relock.RETIRED_PREFIX))

    def interrupt_after_applied(self, *, restoring=None, partial_restore=False):
        """Relock two locks with a KeyboardInterrupt right after `applied` is
        published, then `restoring` handling and restoration as given."""
        kernels, created = self.root / 'bundle/kernels.lock', self.root / 'bundle/new/created.lock'
        self.write(kernels, [lock_record('a')])
        locks = {kernels: [lock_record('a', cubin='c1')], created: [lock_record('b')]}
        logs = self.base / 'logs'
        publish, restore = relock.publish_report, relock.restore
        def published(directory, report):
            if report['status'] == 'restoring' and restoring:
                raise restoring
            publish(directory, report)
            if report['status'] == 'applied':
                raise KeyboardInterrupt
        def interrupted_restore(root, manifest, state):
            # A second interrupt after the first lock is restored.
            restore(root, dict(sorted(manifest.items())[:1]), state)
            raise KeyboardInterrupt
        with self.sdk_workflow(self.cargo(locks, [])), \
             patch.object(relock, 'publish_report', side_effect=published), \
             patch.object(relock, 'restore', side_effect=interrupted_restore if partial_restore else restore):
            with self.assertRaises(BaseException) as failure:
                relock.relock(self.root, self.base / 'target', logs)
        return kernels, created, logs, failure.exception

    def test_a_second_interrupt_during_rollback_leaves_restoring_state_that_blocks(self):
        kernels, created, logs, error = self.interrupt_after_applied(partial_restore=True)
        self.assertIsInstance(error, KeyboardInterrupt)
        # A mixed lock set: the first lock restored, the created one still present.
        self.assertEqual(json.loads(kernels.read_text()), [lock_record('a')])
        self.assertTrue(created.is_file())
        self.assertEqual(json.loads((logs / relock.REPORT).read_text())['status'], 'restoring')
        with self.sdk_workflow(self.cargo({}, [])):
            with self.assertRaisesRegex(PolicyError, 'report status restoring'):
                relock.relock(self.root, self.base / 'target', logs)
        self.assertTrue((logs / relock.STATE / 'manifest.json').is_file())

    def test_a_failed_rollback_publication_changes_no_lock_and_keeps_the_state(self):
        kernels, created, logs, error = self.interrupt_after_applied(restoring=OSError('injected'))
        self.assertIsInstance(error, PolicyError)
        self.assertIn('could not record its rollback', str(error))
        # Every lock is still the applied candidate, as the applied report says.
        self.assertEqual(json.loads(kernels.read_text()), [lock_record('a', cubin='c1')])
        self.assertTrue(created.is_file())
        self.assertEqual(json.loads((logs / relock.REPORT).read_text())['status'], 'applied')
        self.assertTrue((logs / relock.STATE).is_dir())
        locks = {kernels: [lock_record('a', cubin='c1')], created: [lock_record('b')]}
        with self.sdk_workflow(self.cargo(locks, [])):
            again = relock.relock(self.root, self.base / 'target', logs)
        self.assertEqual((again['status'], again['written']), ('applied', []))
        self.assertFalse((logs / relock.STATE).exists())

    def test_a_second_relock_of_the_workspace_with_another_target_is_refused(self):
        lock = self.root / 'bundle/kernels.lock'
        self.write(lock, [lock_record('a')])
        other_target, other_logs, refusals = self.base / 'target-2', self.base / 'logs-2', []
        def second(call):
            # While the first relock builds its candidates.
            if call == 1:
                with self.assertRaises(PolicyError) as refusal:
                    relock.relock(self.root, other_target, other_logs)
                refusals.append(str(refusal.exception))
        with self.sdk_workflow(self.cargo({lock: [lock_record('a', cubin='c1')]}, [], second)):
            report = relock.relock(self.root, self.base / 'target', self.base / 'logs')
        [message] = refusals
        self.assertIn(f'{self.root} is in use by another SDK build or relock (pid {os.getpid()})', message)
        # Refused before it claimed a target or touched its logs.
        self.assertFalse(other_target.exists())
        self.assertFalse(other_logs.exists())
        self.assertEqual((report['status'], report['written']), ('applied', ['bundle/kernels.lock']))
        with workflow.workspace_lock(self.root):
            self.assertEqual((self.root / workflow.SDK_LOCK).read_text(), str(os.getpid()))

    def test_pending_state_of_another_run_still_blocks_relock(self):
        logs = self.base / 'logs'
        (logs / relock.STATE).mkdir(parents=True)
        (logs / relock.STATE / 'id').write_text('other')
        relock.publish_report(logs, {'status': 'applied', 'state_id': 'mine'})
        with self.sdk_workflow(self.cargo({}, [])):
            with self.assertRaisesRegex(PolicyError, 'interrupted relock'):
                relock.relock(self.root, self.base / 'target', logs)
        self.assertTrue((logs / relock.STATE).is_dir())

    def test_an_unrestorable_failure_and_an_interrupted_run_keep_recovery_state(self):
        original, writes = relock.write_lock, []
        def write(target, data):
            writes.append(target)
            if len(writes) >= 2:
                raise OSError('injected write failure')
            original(target, data)
        kernels, created, logs, error = self.failing_relock(None, write)
        self.assertIn('could not restore bundle/kernels.lock', str(error))
        report = json.loads((logs / relock.REPORT).read_text())
        self.assertEqual((report['status'], report['unrestored']), ('restore-failed', ['bundle/kernels.lock']))
        state = logs / relock.STATE
        manifest = json.loads((state / 'manifest.json').read_text())
        self.assertEqual(manifest['bundle/new/created.lock']['old'], None)
        self.assertEqual(json.loads((state / manifest['bundle/kernels.lock']['old']).read_text()),
                         [lock_record('a')])
        calls = []
        with self.sdk_workflow(self.cargo({}, calls)):
            with self.assertRaisesRegex(PolicyError, r'interrupted relock \(report status restore-failed\) left its recovery state'):
                relock.relock(self.root, self.base / 'target', logs)
        self.assertEqual(calls, [])
        self.assertTrue(state.is_dir())


@unittest.skipUnless(shutil.which('cargo'), 'needs cargo')
class CargoConfigTests(unittest.TestCase):
    """Real Cargo builds of fixture model crates whose build script reports
    its lock as kernel_build does: a hostile or mistaken `.cargo/config.toml`,
    and two exports sharing one target directory."""
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name).resolve()
        self.crate = self.model('model')
        (self.crate / '.cargo').mkdir()
        (self.crate / '.cargo/config.toml').write_text(
            f'[env]\n{workflow.RELOCK_ENV} = {{ value = "ignored", force = true, relative = true }}\n')
        self.env = {k: v for k, v in os.environ.items() if k not in workflow.BYPASSES}
        self.env['CARGO_TARGET_DIR'] = str(self.base / 'target')

    def tearDown(self):
        self.temporary.cleanup()

    def model(self, name, *, rerun=True):
        """A crate whose build script reports Updated whenever it sees a relock
        directory, like ModelLock, and otherwise Verified with its lock's path
        and digest; it reruns when that absolute lock path changes."""
        crate = self.base / name
        (crate / 'src').mkdir(parents=True)
        shutil.copyfile(Path(__file__).resolve().parents[1] / 'rust-toolchain.toml', crate / 'rust-toolchain.toml')
        (crate / 'Cargo.toml').write_text('[package]\nname = "model"\nversion = "0.1.0"\nedition = "2021"\n\n[workspace]\n')
        (crate / 'src/lib.rs').write_text('')
        (crate / 'kernels.lock').write_text('[]\n')
        self.build_script(crate, rerun=rerun)
        return crate

    def build_script(self, crate, *, rerun):
        directive = f'println!("cargo:rerun-if-env-changed={workflow.RELOCK_ENV}");' if rerun else ''
        (crate / 'build.rs').write_text(
            'fn main() {\n' + directive + '\n'
            '    let lock = std::fs::canonicalize(concat!(env!("CARGO_MANIFEST_DIR"), "/kernels.lock")).unwrap();\n'
            '    println!("cargo:rerun-if-changed={}", lock.display());\n'
            '    let digest = std::process::Command::new("sha256sum").arg(&lock).output().unwrap().stdout;\n'
            '    let digest = String::from_utf8(digest).unwrap()[..64].to_string();\n'
            f'    let relock = std::env::var("{workflow.RELOCK_ENV}").unwrap_or_default();\n'
            '    let status = if relock.is_empty() { "Verified" } else { "Updated" };\n'
            '    let out = std::env::var("OUT_DIR").unwrap();\n'
            f'    std::fs::write(format!("{{out}}/{workflow.KERNEL_BUILD_REPORT}"), format!(\n'
            '        "{{\\"lock_status\\":\\"{status}\\",\\"lock_path\\":\\"{}\\",\\"lock_sha256\\":\\"{digest}\\"}}",\n'
            '        lock.display())).unwrap();\n}\n')

    def sdk_build(self, crate=None):
        crate = crate or self.crate
        command = ['cargo', 'build', '--offline', '--manifest-path', crate / 'Cargo.toml', *workflow.cargo_config()]
        workflow.verified_cargo(command, crate, env=self.env, log=self.base / 'build.log')

    def reports(self, key='lock_status'):
        return [json.loads(p.read_text())[key] for p in (self.base / 'target').rglob(workflow.KERNEL_BUILD_REPORT)]

    def test_a_relative_config_cannot_turn_the_forced_empty_value_into_a_directory(self):
        for _ in range(2):
            self.sdk_build()
            self.assertEqual(self.reports(), ['Verified'])

    def test_a_cached_unverified_report_from_an_earlier_build_fails(self):
        # A raw build under the model configuration leaves an Updated report;
        # without a rerun directive the SDK build reuses it.
        self.build_script(self.crate, rerun=False)
        subprocess.run(['cargo', 'build', '--offline', '--manifest-path', str(self.crate / 'Cargo.toml')],
                       env=self.env, cwd=self.crate, check=True, capture_output=True)
        self.assertEqual(self.reports(), ['Updated'])
        for report in (self.base / 'target').rglob(workflow.KERNEL_BUILD_REPORT):
            os.utime(report, ns=(0, 0))
        for _ in range(2):
            with self.assertRaisesRegex(PolicyError, 'Updated, not Verified'):
                self.sdk_build()

    def test_another_export_sharing_the_target_cannot_lend_its_verified_report(self):
        first, second = self.model('a'), self.model('b')
        self.sdk_build(first)
        (second / 'kernels.lock').write_text('[{"cubin_sha256": "0"}]\n')
        # Cargo reuses the first export's build script output for the second.
        with self.assertRaisesRegex(PolicyError, 'outside this workspace'):
            self.sdk_build(second)
        self.assertEqual(self.reports('lock_path'), [str(first / 'kernels.lock')])


if __name__ == '__main__':
    unittest.main()
