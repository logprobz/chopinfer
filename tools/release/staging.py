"""Admit producer-recorded artifacts before copying or executing staged code."""
import argparse
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import tomllib

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import archive  # noqa: E402
import provenance as producer  # noqa: E402

source = producer.source
require = producer.require


def comparable_configuration(value):
    # Global Cargo configuration is retained as producer evidence. The enforced
    # profile, flags and toolchain define compatibility across build hosts.
    return {key: item for key, item in value.items() if key != 'cargo_configuration'}


def kernel_evidence(record, metadata, root):
    expected = {(path, item['name']): item for path, lock in record['locks'].items() for item in lock['records']}
    packages = {f"{package['name']}@{package['version']}": Path(package['manifest_path']).parent
                for package in metadata['packages']}
    covered = set()
    for evidence in record['kernels']:
        require(evidence.get('lock_status') == 'Verified', 'kernel evidence is not Verified')
        require(evidence['producer'] in packages, 'unknown kernel producer')
        owner = packages[evidence['producer']]
        require(evidence['locks'] and len(set(evidence['locks'])) == len(evidence['locks']), 'duplicate or missing report locks')
        for path in evidence['locks']:
            archive.relative_path(path)
            require(path in record['locks'] and (root / path).is_relative_to(owner), 'kernel lock differs from its producer')
        compiler = evidence['compiler']
        producer.reports.hash_string(compiler['sha256'])
        require(compiler['toolchain_version'] == record['configuration']['nvcc'], 'compiler version differs')
        require(compiler['inputs'].get('nvcc') == record['configuration']['nvcc_sha256']
                and compiler['inputs'].get('nvcc.profile') == record['configuration']['nvcc_profile_sha256'],
                'compiler freshness differs from the release configuration')
        require(2 <= len(compiler['inputs']) <= 256, 'compiler input inventory bound')
        for name, sha in compiler['inputs'].items():
            require(Path(name).name == name, 'compiler input identity is not portable')
            producer.reports.hash_string(sha)
        names = set()
        for identity in evidence['identities']:
            require(identity['line_info'] is False and identity['name'] not in names, 'profiled or duplicate kernel identity')
            names.add(identity['name'])
            matches = [(path, identity['name']) for path in evidence['locks'] if (path, identity['name']) in expected]
            require(len(matches) == 1, 'kernel identity lacks one intended lock')
            key = matches[0]
            gating = identity if 'source_closure_sha256' in expected[key] else producer.reports.family_record(identity)
            require(gating == expected[key], 'kernel identity differs from intended lock')
            covered.add(key)
    require(covered == set(expected), 'missing host, model or child kernel evidence')


def verify(root, prebuilt):
    root = Path(root).resolve()
    prebuilt = Path(os.path.abspath(prebuilt))
    buffer = io.BytesIO()
    source.file_identity(prebuilt / producer.RECORD, producer.reports.Limits().report_bytes, buffer)
    record = producer.reports.parse(buffer.getvalue())
    require(record['schema'] == producer.SCHEMA, 'unsupported release producer record')
    producer.reports.hash_string(record['build_sha256'])
    require(record['build_sha256'] == producer.build_identity(record), 'release producer record binding differs')
    config = producer.configuration(root)
    require(comparable_configuration(record['configuration']) == comparable_configuration(config),
            'release configuration differs from the intended build')
    rustc, cargo = producer.rust_tools(root, os.environ)
    env = {**os.environ, 'RUSTC': str(rustc), 'RUSTC_WRAPPER': '', 'RUSTC_WORKSPACE_WRAPPER': ''}
    metadata = json.loads(subprocess.check_output([cargo, 'metadata', '--format-version', '1', '--no-deps',
                                                  '--locked', '--offline', *producer.cargo_controls()], cwd=root, env=env))
    models = producer.plan.read(root, metadata)
    selected = producer.selected_artifacts(models, metadata)
    roots = source.source_roots(root, metadata, {item['package'] for item in selected.values()}, models)
    current = source.snapshot(root, roots)
    require(record['source']['context'] in ('git', 'retained-refs'), 'unknown source context')
    require(all(record['source'][key] == current[key] for key in ('revision', 'inputs', 'sha256')),
            'source revision or input inventory differs from the intended build')
    require(record['locks'] == producer.lock_inventory(root, roots, models), 'kernel locks differ from the intended build')
    kernel_evidence(record, metadata, root)
    artifacts = {}
    for artifact in record['artifacts']:
        path = str(archive.relative_path(artifact['path']))
        require('/' not in path and path not in artifacts and path in selected, 'duplicate or unexpected artifact path')
        wanted = selected[path]
        require(artifact['role'] == wanted['kind'] and all(artifact[key] == wanted[key]
                for key in ('package', 'target', 'destination')), 'artifact role differs from release selection')
        require(artifact['build_sha256'] == record['build_sha256'], 'artifact belongs to another build')
        features = ['linux-cuda-host'] if artifact['package'] == 'runtime-host' else []
        require(artifact['features'] == features, 'artifact features differ')
        profile = artifact['profile']
        require(profile['opt_level'] == '3' and not profile['debug_assertions'] and not profile['overflow_checks']
                and not profile['test'] and profile['debuginfo'] in (0, None), 'artifact profile differs')
        require(type(artifact['size']) is int and 0 < artifact['size'] <= producer.MAX_ARTIFACT_BYTES, 'artifact size bound')
        producer.reports.hash_string(artifact['sha256'])
        require(source.file_identity(prebuilt / path, producer.MAX_ARTIFACT_BYTES) ==
                {key: artifact[key] for key in ('size', 'sha256')}, 'staged artifact bytes differ from producer record')
        artifacts[path] = artifact
    require(set(artifacts) == set(selected), 'missing selected release artifact')
    return record, models, roots


def copy_artifacts(prebuilt, package, record):
    for artifact in record['artifacts']:
        path = package / archive.relative_path(artifact['destination'])
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as stream:
            identity = source.file_identity(Path(prebuilt) / artifact['path'], producer.MAX_ARTIFACT_BYTES, stream)
            stream.flush()
            os.fsync(stream.fileno())
        require(identity == {key: artifact[key] for key in ('size', 'sha256')}, 'artifact changed during staging copy')
        require(source.file_identity(path, producer.MAX_ARTIFACT_BYTES) == identity, 'staging destination digest differs')
        path.chmod(0o755)


def stage(root, prebuilt, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    require(output != root and root not in output.parents, 'release output must be outside the source tree')
    record, models, roots = verify(root, prebuilt)
    version = tomllib.loads((root / 'Cargo.toml').read_text())['workspace']['package']['version']
    require(re.fullmatch(r'(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)-alpha\.(0|[1-9][0-9]*)', version),
            'workspace version must be a SemVer alpha release')
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.stage.', dir=output) as directory:
        temporary = Path(directory)
        package = temporary / f'chopinfer-{version}-linux-x86_64-cuda13.2'
        package.mkdir()
        copy_artifacts(prebuilt, package, record)
        stub = temporary / 'cuda-link-stub'
        stub.mkdir()
        cuda = Path(os.environ.get('CHOPINFER_CUDA_HOME', '/usr/local/cuda')).resolve()
        (stub / 'libcuda.so.1').symlink_to(cuda / 'targets/x86_64-linux/lib/stubs/libcuda.so')
        libraries = os.environ.get('LD_LIBRARY_PATH', '')
        env = {**os.environ, 'LD_LIBRARY_PATH': str(stub) + (':' + libraries if libraries else '')}
        revision = record['source']['revision']
        for name in ('chopinfer', 'chopinfer-host', 'choptop'):
            expected = f'{name} {version}' + ('' if name == 'choptop' else f' ({revision})')
            require(subprocess.check_output([package / 'bin' / name, '--version'], env=env, text=True).strip() == expected,
                    'verified executable version differs from the release identity')
        archive.populate(root, package, models, stub, record['source']['inputs'])
        for artifact in record['artifacts']:
            require(source.file_identity(package / artifact['destination'], producer.MAX_ARTIFACT_BYTES) ==
                    {key: artifact[key] for key in ('size', 'sha256')}, 'artifact changed while assembling the archive')
        rust = record['configuration']['rust'].split()[1]
        (package / 'BUILD_INFO').write_text(f'version={version}\nrevision={revision}\nrust={rust}\ncuda=13.2\nnvcc=13.2.86\n'
                                           f"source_sha256={record['source']['sha256']}\nbuild_sha256={record['build_sha256']}\n")
        producer.atomic_json(package / 'BUILD_PROVENANCE.json', record)
        current = source.snapshot(root, roots)
        require(all(current[key] == record['source'][key] for key in ('revision', 'inputs', 'sha256')),
                'source inputs changed during staging')
        return archive.finish(package, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=producer.ROOT)
    parser.add_argument('prebuilt', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    print('Artifact: ' + str(stage(args.root, args.prebuilt, args.output)))


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, KeyError, TypeError, subprocess.CalledProcessError) as error:
        raise SystemExit(f'release staging: {error}')
