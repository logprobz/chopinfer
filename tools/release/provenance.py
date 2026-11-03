"""Controlled release production and portable artifact provenance.

The record trusts this producer. It does not authenticate an arbitrary builder
and does not replace CPU or model qualification gates.
"""
import argparse
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import tomllib

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / 'tools'))
sys.path.insert(0, str(ROOT / 'ci'))
import plan  # noqa: E402
import source_context as source  # noqa: E402
import kernel_reports as reports  # noqa: E402
from model_sdk.workflow import BYPASSES, cargo_config, cargo_messages  # noqa: E402

SCHEMA = 'chopinfer-release-build.v1'
RECORD = 'release-build.json'
TARGET = 'x86_64-unknown-linux-gnu'
MAX_ARTIFACT_BYTES = 512 << 20
RELEASE_PROFILE = {'opt-level': 3, 'debug': False, 'debug-assertions': False, 'overflow-checks': False,
                   'lto': False, 'panic': 'unwind', 'codegen-units': 16, 'incremental': False, 'strip': 'none'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def rust_tools(root, environment):
    launcher = shutil.which('rustc', path=environment.get('PATH'))
    require(launcher, 'Rust compiler is unavailable')
    sysroot = Path(subprocess.check_output([launcher, '--print', 'sysroot'], cwd=root,
                                           env=environment, text=True).strip()).resolve()
    return sysroot / 'bin/rustc', sysroot / 'bin/cargo'


def configuration(root, environment=None):
    env = os.environ if environment is None else environment
    require(platform.system() == 'Linux' and platform.machine() == 'x86_64', 'Linux x86_64 required')
    for name in BYPASSES:
        require(env.get(name, '') in ('', '0'), f'{name} is forbidden in release builds')
    for name in ('RUSTC_BOOTSTRAP', 'CARGO_ENCODED_RUSTFLAGS'):
        require(not env.get(name), f'{name} is incompatible with a controlled release')
    for name, value in env.items():
        if name.startswith('CARGO_PROFILE_RELEASE_'):
            key = name[len('CARGO_PROFILE_RELEASE_'):].lower().replace('_', '-')
            wanted = RELEASE_PROFILE.get(key)
            require(wanted is not None and value == str(wanted).lower(), f'incompatible release profile: {name}')
    manifest = tomllib.loads((root / 'Cargo.toml').read_text())
    for name, value in manifest.get('profile', {}).get('release', {}).items():
        require(name in RELEASE_PROFILE and value == RELEASE_PROFILE[name], 'incompatible source release profile')
    cargo_home = Path(env.get('CARGO_HOME', Path.home() / '.cargo'))
    cargo_configs = {}
    for label, directory in [('cargo-home', cargo_home), *[(f'source-parent-{i}', parent / '.cargo')
                               for i, parent in enumerate((root, *root.parents))]]:
        for filename in ('config', 'config.toml'):
            path = directory / filename
            if path.is_file():
                data = tomllib.loads(path.read_text())
                for name, value in data.get('profile', {}).get('release', {}).items():
                    require(name in RELEASE_PROFILE and value == RELEASE_PROFILE[name], 'incompatible Cargo release profile')
                cargo_configs[f'{label}/{filename}'] = source.file_identity(path)
    toolchain = tomllib.loads((root / 'rust-toolchain.toml').read_text())['toolchain']['channel']
    rustc, cargo = rust_tools(root, env)
    require(rustc.is_file(), 'Rust compiler is unavailable')
    require(not env.get('RUSTC') or Path(env['RUSTC']).resolve() == rustc, 'RUSTC differs from the pinned compiler')
    rust_version = subprocess.check_output([rustc, '--version', '--verbose'], env=env, text=True).strip()
    require(rust_version.startswith(f'rustc {toolchain} '), 'pinned Rust toolchain required')
    require(f'host: {TARGET}' in rust_version, 'Rust host target differs from the release target')
    cargo_version = subprocess.check_output([cargo, '--version'], env=env, text=True).strip()
    require(cargo_version.startswith(f'cargo {toolchain} '), 'pinned Cargo toolchain required')
    cuda = Path(env.get('CHOPINFER_CUDA_HOME', '/usr/local/cuda')).resolve()
    nvcc_version = subprocess.check_output([cuda / 'bin/nvcc', '--version'], env=env, text=True).strip()
    require('V13.2.86' in nvcc_version, 'pinned CUDA compiler required')
    flags = env.get('RUSTFLAGS', '').split()
    if flags:
        require(len(flags) == 2 and flags[0] == '-L' and flags[1].startswith('native='),
                'release RUSTFLAGS support only the CUDA link-stub search path')
        requested = Path(flags[1][len('native='):]) / 'libcuda.so.1'
        require(requested.resolve() == (cuda / 'targets/x86_64-linux/lib/stubs/libcuda.so').resolve(),
                'RUSTFLAGS native directory is not the CUDA link stub')
    return {'target': TARGET, 'profile': RELEASE_PROFILE, 'rust': rust_version, 'cargo': cargo_version,
            'rustc_sha256': source.file_identity(rustc, MAX_ARTIFACT_BYTES)['sha256'],
            'cargo_sha256': source.file_identity(cargo, MAX_ARTIFACT_BYTES)['sha256'],
            'nvcc': nvcc_version, 'nvcc_sha256': source.file_identity(cuda / 'bin/nvcc', MAX_ARTIFACT_BYTES)['sha256'],
            'nvcc_profile_sha256': source.file_identity(cuda / 'bin/nvcc.profile')['sha256'],
            'rustflags': ['cuda-link-stub'], 'rustc_wrappers': False, 'cargo_configuration': cargo_configs,
            'cuda_link_stub_sha256': source.file_identity(cuda / 'targets/x86_64-linux/lib/stubs/libcuda.so')['sha256']}


def selected_artifacts(models, metadata):
    owners = {package['name']: package for package in metadata['packages']}
    targets = [('runtime-host', 'chopinfer', 'bin'), ('runtime-host', 'chopinfer-host', 'bin'),
               ('choptop', 'choptop', 'bin')]
    targets += [(target.package, target.name, target.kind) for model in models for target in model.targets]
    selected = {}
    for package, name, kind in targets:
        owner = owners[package]
        require(any(t['name'] == name and kind in t['kind'] for t in owner['targets']), 'missing release target')
        filename = f'lib{name}.so' if kind == 'cdylib' else name
        require(filename not in selected, 'duplicate release artifact filename')
        selected[filename] = {'package': package, 'package_id': owner['id'], 'target': name, 'kind': kind,
                              'manifest_path': owner['manifest_path'],
                              'destination': ('lib/' if kind == 'cdylib' else 'bin/') + filename}
    return selected


def lock_inventory(root, roots, models):
    """All production dependency locks, including host and child selections."""
    paths = set()
    for directory in roots:
        if (directory / 'Cargo.toml').is_file():
            if (directory / 'kernels.lock').is_file():
                paths.add(directory / 'kernels.lock')
            paths |= set(directory.glob('kernels.lock.d/*.lock'))
    paths |= {root / asset.source for model in models for asset in model.assets
              if Path(asset.source).name == 'kernels.lock' or
              Path(asset.source).suffix == '.lock' and Path(asset.source).parent.name == 'kernels.lock.d'}
    paths |= {root / 'crates/runtime-kernels/kernels.lock', root / 'crates/runtime-kernels/sampling-mapped.lock'}
    locks = {}
    for path in sorted(paths):
        identity = source.file_identity(path)
        with os.fdopen(source.regular(path), 'rb') as stream:
            records = reports.parse(stream.read(reports.Limits().report_bytes))
        require(isinstance(records, list) and records and all(isinstance(record, dict) and record.get('name') for record in records),
                'invalid kernel lock inventory')
        require(len({record['name'] for record in records}) == len(records), 'duplicate locked cubin')
        locks[str(path.relative_to(root))] = {**identity, 'records': records}
    return locks


def verify_kernel_outputs(root, outputs, locks, target):
    reader = reports.Reader()
    expected = {(path, record['name']): record for path, lock in locks.items() for record in lock['records']}
    covered, evidence = {}, []
    for package, directories in outputs.items():
        before_gaps = len(reader.gaps)
        paths = reader.enumerate(directories)
        require(all(gap.startswith('empty report directory:') or
                    gap.startswith('report input ') and 'No such file or directory' in gap
                    for gap in reader.gaps[before_gaps:]), 'unusable kernel report traversal')
        # Build scripts without kernels may have no report. Expected locks below
        # decide coverage; a nonempty malformed report is always a refusal.
        for path in paths:
            require(path.is_relative_to(target), 'kernel report escaped the private build target')
            require(path.name == 'kernel-build-report.json', 'profiled kernel report in release build')
            report = reader.files.json(path)
            reader.structure(report)
            reader.verify(report)
            for file in report['files'].values():
                if file['identity']['kind'] == 'Project':
                    require(Path(file['location']).is_relative_to(root), 'kernel source escaped the source context')
            families = report.get('lock_families') or [{'lock_path': report['lock_path'],
                                                       'cubins': [c['name'] for c in report['cubins']]}]
            names = {c['name']: c for c in report['cubins']}
            for family in families:
                relative = str(Path(family['lock_path']).relative_to(root))
                for name in family['cubins']:
                    cubin = names[name]
                    identity = cubin['identity']
                    key = (relative, name)
                    require(identity['line_info'] is False and key in expected, 'unexpected or profiled release cubin')
                    gating = reports.family_record(identity) if report.get('lock_families') else identity
                    require(gating == expected[key], 'kernel report does not cover the intended lock record')
                    previous = covered.setdefault(key, gating)
                    require(previous == gating, 'conflicting kernel report coverage')
            compiler = report['compiler_freshness']
            evidence.append({'producer': package, 'selection': (report.get('context') or {}).get('selection'),
                             'lock_status': report['lock_status'],
                             'locks': [str(Path(f['lock_path']).relative_to(root)) for f in families],
                             'identities': [c['identity'] for c in report['cubins']],
                             'compiler': {'sha256': compiler['sha256'], 'toolchain_version': compiler['toolchain_version'],
                                          'inputs': {name: value['sha256'] for name, value in compiler['files'].items()}}})
    require(set(covered) == set(expected), 'missing kernel report coverage: ' + ', '.join(
        f'{path}:{name}' for path, name in sorted(set(expected) - set(covered))))
    return evidence


def controlled_environment(root, config, revision, target, stub):
    env = dict(os.environ)
    for name in list(env):
        if name.startswith(('CARGO_PROFILE_', 'CARGO_TARGET_')) or name in ('RUSTFLAGS', 'CARGO_ENCODED_RUSTFLAGS'):
            env.pop(name)
    env.update({name: '' for name in BYPASSES})
    rustc, _ = rust_tools(root, env)
    env['PATH'] = str(rustc.parent) + ':' + env.get('PATH', '')
    env.update(CARGO_TARGET_DIR=str(target), CARGO_BUILD_TARGET=config['target'], CARGO_NET_OFFLINE='true',
               CARGO_INCREMENTAL='0', CHOPINFER_BUILD_REVISION=revision,
               CARGO_ENCODED_RUSTFLAGS='-L\x1fnative=' + str(stub), RUSTC_WRAPPER='', RUSTC_WORKSPACE_WRAPPER='',
               RUNTIME_KERNELS_CUBIN_DIR=str(target / 'host-cubins'), RUSTC=str(rustc))
    return env


def cargo_controls(target=None, revision=None, rustc=None):
    forced = cargo_config()
    for name, value in RELEASE_PROFILE.items():
        forced += ['--config', f'profile.release.{name}={json.dumps(value)}']
    for name in ('rustc-wrapper', 'rustc-workspace-wrapper'):
        forced += ['--config', f'build.{name}=""']
    for name in ('RUSTC_WRAPPER', 'RUSTC_WORKSPACE_WRAPPER', 'RUSTC_BOOTSTRAP'):
        for field in ('value=""', 'force=true', 'relative=false'):
            forced += ['--config', f'env.{name}.{field}']
    if target is not None:
        for field in (f'value={json.dumps(str(target / "host-cubins"))}', 'force=true', 'relative=false'):
            forced += ['--config', 'env.RUNTIME_KERNELS_CUBIN_DIR.' + field]
    for name, value in (('CHOPINFER_BUILD_REVISION', revision), ('RUSTC', rustc)):
        if value is not None:
            for field in (f'value={json.dumps(value)}', 'force=true', 'relative=false'):
                forced += ['--config', f'env.{name}.{field}']
    return forced


def produce(root, scratch):
    root = Path(root).resolve()
    config = configuration(root)
    scratch = Path(scratch).resolve()
    if scratch == root or root in scratch.parents:
        scratch = Path(tempfile.gettempdir()).resolve()
    scratch.mkdir(parents=True, exist_ok=True)
    private = Path(tempfile.mkdtemp(prefix='chopinfer-release-', dir=scratch))
    target, prebuilt = private / 'target', private / 'prebuilt'
    target.mkdir()
    prebuilt.mkdir()
    cuda = Path(os.environ.get('CHOPINFER_CUDA_HOME', '/usr/local/cuda')).resolve()
    stub = private / 'cuda-link-stub'
    stub.mkdir()
    (stub / 'libcuda.so.1').symlink_to(cuda / 'targets/x86_64-linux/lib/stubs/libcuda.so')
    revision = source.revision(root)
    env = controlled_environment(root, config, revision, target, stub)
    forced = cargo_controls(target, revision, env['RUSTC'])
    metadata = json.loads(subprocess.check_output(['cargo', 'metadata', '--format-version', '1', '--no-deps',
                                                  '--locked', '--offline', *forced], cwd=root, env=env))
    models = plan.read(root, metadata)
    selected = selected_artifacts(models, metadata)
    roots = source.source_roots(root, metadata, {item['package'] for item in selected.values()}, models)
    before = source.snapshot(root, roots)
    locks = lock_inventory(root, roots, models)
    commands = [
        ['-p', 'runtime-host', '--no-default-features', '--features', 'linux-cuda-host', '--bin', 'chopinfer', '--bin', 'chopinfer-host'],
        ['-p', 'choptop', '--bin', 'choptop'], list(plan.build_args(models, 'bin')), list(plan.build_args(models, 'cdylib'))]
    local = {package['id']: package for package in metadata['packages']}
    outputs, artifacts = {}, {}
    for index, arguments in enumerate(commands):
        if not arguments:
            continue
        log = private / f'cargo-{index}.jsonl'
        with log.open('w') as stream:
            subprocess.run(['cargo', 'build', '--release', '--locked', '--offline', '--target', TARGET,
                            '--message-format=json-render-diagnostics', *forced, *arguments],
                           cwd=root, env=env, stdout=stream, check=True)
        for message in cargo_messages(log.read_text()):
            if message.get('reason') == 'build-script-executed' and message['package_id'] in local:
                out = Path(message['out_dir'])
                require(out.is_relative_to(target), 'build-script output escaped the private target')
                package = local[message['package_id']]
                directories = outputs.setdefault(f"{package['name']}@{package['version']}", set())
                directories.add(out)
                if package['name'] == 'runtime-kernels':
                    directories.add(target / 'host-cubins')
            if message.get('reason') != 'compiler-artifact':
                continue
            if message['package_id'] in local and any(Path(path).is_relative_to(target / TARGET / 'release')
                                                      for path in message['filenames']):
                profile = message['profile']
                require(profile['opt_level'] == '3' and not profile['debug_assertions'] and not profile['overflow_checks']
                        and not profile['test'] and profile['debuginfo'] in (0, None), 'incompatible dependency release profile')
                require(not set(message['features']) & {'diagnostic-taps', 'test-support', 'fake-driver-tests',
                                                        'cuda-fake-driver-tests', 'fake-host-services'},
                        'diagnostic dependency features in release build')
            for filename, item in selected.items():
                if message['package_id'] != item['package_id'] or message['target']['name'] != item['target']:
                    continue
                if item['kind'] not in message['target']['kind']:
                    continue
                require(Path(message['manifest_path']) == Path(item['manifest_path']), 'artifact producer manifest differs')
                profile = message['profile']
                require(profile['opt_level'] == '3' and not profile['debug_assertions'] and not profile['overflow_checks']
                        and not profile['test'] and profile['debuginfo'] in (0, None), 'incompatible artifact release profile')
                features = sorted(message['features'])
                require(features == (['linux-cuda-host'] if item['package'] == 'runtime-host' else []),
                        'incompatible release artifact features')
                paths = [Path(path) for path in message['filenames'] if Path(path).name == filename]
                require(len(paths) == 1 and paths[0].is_relative_to(target), 'missing or escaping produced artifact')
                artifacts[filename] = {'path': paths[0], 'features': features, 'profile': profile}
    require(set(artifacts) == set(selected), 'this controlled build did not produce every selected artifact')
    kernels = verify_kernel_outputs(root, outputs, locks, target)
    sealed = []
    for filename, item in selected.items():
        artifact = artifacts[filename]
        with (prebuilt / filename).open('xb') as destination:
            identity = source.file_identity(artifact['path'], MAX_ARTIFACT_BYTES, destination)
        require(source.file_identity(prebuilt / filename, MAX_ARTIFACT_BYTES) == identity, 'copied artifact differs')
        (prebuilt / filename).chmod(0o755)
        sealed.append({'path': filename, 'role': item['kind'], 'package': item['package'], 'target': item['target'],
                       'destination': item['destination'], 'features': artifact['features'], 'profile': artifact['profile'], **identity})
    require(source.snapshot(root, roots) == before, 'source inputs or revision changed during the controlled build')
    require(configuration(root) == config and lock_inventory(root, roots, models) == locks, 'release inputs changed during the build')
    record = {'schema': SCHEMA, 'source': before, 'configuration': config, 'artifacts': sealed,
              'locks': locks, 'kernels': kernels}
    record['build_sha256'] = build_identity(record)
    for artifact in sealed:
        artifact['build_sha256'] = record['build_sha256']
    atomic_json(prebuilt / RECORD, record)
    return {'prebuilt': str(prebuilt), 'record': str(prebuilt / RECORD), 'target': str(target)}


def build_identity(record):
    material = {key: record[key] for key in ('source', 'configuration', 'locks', 'kernels')}
    material['artifacts'] = [{key: value for key, value in artifact.items() if key != 'build_sha256'}
                             for artifact in record['artifacts']]
    return source.digest(material)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--scratch', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(produce(args.root, args.scratch)))


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, KeyError, TypeError, subprocess.CalledProcessError) as error:
        raise SystemExit(f'release provenance: {error}')
