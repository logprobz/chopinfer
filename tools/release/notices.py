#!/usr/bin/env python3
"""Render the project's third-party notices and the locked registry license
texts offline, including build dependencies.

The output is a distribution artifact, never a source-tree cache. It starts
with the source tree's THIRD_PARTY_NOTICES (vendored and adapted files), then
every registry crate. All license expressions must be reviewed here; unknown
or copyleft expressions fail closed.
"""
import argparse
import hashlib
import os
from pathlib import Path
import tarfile
import tomllib

APPROVED = {
    'MIT', 'Apache-2.0', 'MIT OR Apache-2.0', 'Apache-2.0 OR MIT',
    'MIT/Apache-2.0', '0BSD OR MIT OR Apache-2.0', 'Unlicense OR MIT',
    'MIT AND BSD-3-Clause', 'MIT OR Zlib OR Apache-2.0',
    'Zlib OR Apache-2.0 OR MIT', 'MIT OR Apache-2.0 OR Zlib',
    'Apache-2.0 OR BSL-1.0', '(MIT OR Apache-2.0) AND Unicode-3.0',
    'Apache-2.0 WITH LLVM-exception OR Apache-2.0 OR MIT',
    # signal-hook's older spelling of MIT OR Apache-2.0.
    'Apache-2.0/MIT',
    # zlib-rs, locked through flate2's weak `zlib-rs?/std` feature (png's
    # encoder dependency) and never built.
    'Zlib',
}


def render(lock, cargo_home, project=b''):
    sections = ([project.decode('utf-8')] if project else []) + ['Third-party notices\n\nAll registry packages in Cargo.lock, including build and platform dependencies.\nLicense texts are reproduced without modification.\n']
    for package in sorted(lock['package'], key=lambda p: (p['name'], p['version'])):
        if 'source' not in package:
            continue
        identity = f"{package['name']} {package['version']}"
        if package['source'] != 'registry+https://github.com/rust-lang/crates.io-index':
            raise ValueError(f'{identity}: unreviewed source')
        prefix = f"{package['name']}-{package['version']}"
        archives = sorted((cargo_home / 'registry/cache').glob(f'*/{prefix}.crate'))
        archives = [path for path in archives
                    if hashlib.sha256(path.read_bytes()).hexdigest() == package['checksum']]
        if not archives:
            raise ValueError(f'{identity}: missing archive or checksum differs')
        with tarfile.open(archives[0], 'r:gz') as archive:
            manifest = tomllib.loads(archive.extractfile(f'{prefix}/Cargo.toml').read().decode())
            license = manifest['package'].get('license')
            if license not in APPROVED:
                raise ValueError(f'{identity}: unreviewed license {license!r}')
            files = sorted(member.name for member in archive.getmembers()
                           if member.isfile() and member.name.startswith(prefix + '/') and
                           Path(member.name).name.lower().startswith(('license', 'copying', 'copyright', 'notice')))
            if not files:
                raise ValueError(f'{identity}: no license texts')
            sections.append(f"\n{'=' * 72}\n{identity}\nLicense: {license}\n")
            for name in files:
                sections.append(f'\n--- {Path(name).relative_to(prefix).as_posix()} ---\n' + archive.extractfile(name).read().decode('utf-8'))
    return '\n'.join(sections).encode('utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    lock = tomllib.loads((root / 'Cargo.lock').read_text())
    cargo_home = Path(os.environ.get('CARGO_HOME', Path.home() / '.cargo'))
    args.output.write_bytes(render(lock, cargo_home, (root / 'THIRD_PARTY_NOTICES').read_bytes()))


if __name__ == '__main__':
    main()
