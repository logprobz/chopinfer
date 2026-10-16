"""Build a release image with explicit inputs, or check those inputs in BuildKit."""
import hashlib
import os
from pathlib import Path
import re
import subprocess
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[2]


def inputs():
    defaults = dict(re.findall(r'^ARG (\w+)=(.+)$',
                               (ROOT / 'deploy/docker/Dockerfile').read_text(), re.M))
    values = {}
    for name in ('CUDA_DEVEL_IMAGE', 'CUDA_RUNTIME_IMAGE', 'BUILDKIT_SYNTAX'):
        value = os.environ.get(name) or defaults.get(name, '')
        if not re.fullmatch(r'[^\s@]+@sha256:[0-9a-f]{64}', value):
            raise ValueError(f'{name} must name an approved image by sha256 digest')
        values[name] = value
    for name, stage in [('BUILD_IMAGE', 'toolchain'), ('RUNTIME_IMAGE', 'runtime-tools')]:
        value = os.environ.get(name, stage)
        if value != stage and not re.fullmatch(r'[^\s@]+@sha256:[0-9a-f]{64}', value):
            raise ValueError(f'{name} must be {stage} or an approved prepared image by digest')
        values[name] = value
    values['RUST_TOOLCHAIN'] = tomllib.loads((ROOT / 'rust-toolchain.toml').read_text())['toolchain']['channel']
    values['CARGO_LOCK_SHA256'] = hashlib.sha256((ROOT / 'Cargo.lock').read_bytes()).hexdigest()
    values['CUDA_COMPILER_VERSION'] = '13.2.86-1'
    return values


def revision():
    supplied = os.environ.get('CHOPINFER_BUILD_REVISION', '')
    head = ROOT / '.git/HEAD'
    found = ''
    if head.is_file():
        found = head.read_text().strip()
        if found.startswith('ref: '):
            ref = found[5:]
            if not re.fullmatch(r'refs/[A-Za-z0-9_./-]+', ref) or '..' in ref:
                raise ValueError('invalid Git HEAD reference')
            loose = ROOT / '.git' / ref
            found = loose.read_text().strip() if loose.is_file() else ''
            packed = ROOT / '.git/packed-refs'
            if not found and packed.is_file():
                found = next((line.split()[0] for line in packed.read_text().splitlines()
                              if len(line.split()) == 2 and line.split()[1] == ref), '')
    if not re.fullmatch(r'[0-9a-f]{40}', found):
        raise ValueError('full source revision required from Git HEAD: use a standalone clone or --build-arg BUILDKIT_CONTEXT_KEEP_GIT_DIR=1')
    if supplied and supplied != found:
        raise ValueError('CHOPINFER_BUILD_REVISION differs from Git HEAD')
    return found


def main():
    if sys.argv[1:] == ['--revision']:
        print(revision())
        return
    values = inputs()
    if sys.argv[1:] == ['--check']:
        for name in ('RUST_TOOLCHAIN', 'CARGO_LOCK_SHA256', 'CUDA_COMPILER_VERSION'):
            if os.environ.get(name) and os.environ[name] != values[name]:
                raise ValueError(f'{name} differs from the source build input')
        rust = subprocess.check_output(['rustc', '--version'], text=True).split()[1]
        if rust != values['RUST_TOOLCHAIN']:
            raise ValueError('installed Rust compiler differs from RUST_TOOLCHAIN')
        nvcc = subprocess.check_output(['nvcc', '--version'], text=True)
        if 'V13.2.86' not in nvcc:
            raise ValueError('installed CUDA compiler differs from CUDA_COMPILER_VERSION')
        return
    source_revision = os.environ.get('CHOPINFER_BUILD_REVISION') or subprocess.check_output(
        ['git', 'rev-parse', '--verify', 'HEAD'], cwd=ROOT, text=True).strip()
    command = ['docker', 'buildx', 'build', '--load', *sys.argv[1:],
               '--build-arg', f'CHOPINFER_BUILD_REVISION={source_revision}']
    for name, value in values.items():
        command.extend(['--build-arg', f'{name}={value}'])
    command.extend(['-f', str(ROOT / 'deploy/docker/Dockerfile'), str(ROOT)])
    subprocess.run(command, check=True)


if __name__ == '__main__':
    try:
        main()
    except (ValueError, subprocess.CalledProcessError) as error:
        sys.exit(str(error))
