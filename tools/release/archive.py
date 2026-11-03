"""Archive layout after the caller has verified and copied native artifacts."""
import io
import os
from pathlib import Path, PurePosixPath
import stat
import subprocess
import tarfile
import tomllib

import notices
import plan
import source_context as source


def relative_path(value):
    path = PurePosixPath(value)
    if not value or path.is_absolute() or '..' in path.parts or str(path) != value or '\\' in value:
        raise ValueError('release path must be canonical and relative')
    return path


def copy_asset(root, package, origin, destination, inputs=None, mode=None):
    origin, destination = relative_path(origin), relative_path(destination)
    path, output = root / origin, package / destination
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('wb') as stream:
        identity = source.file_identity(path, destination=stream)
    if inputs is not None and inputs.get(str(origin)) != identity:
        raise ValueError('release asset differs from validated source inputs')
    output.chmod(mode if mode is not None else stat.S_IMODE(path.stat(follow_symlinks=False).st_mode) & 0o777)


def read_input(root, name, inputs):
    buffer = io.BytesIO()
    identity = source.file_identity(root / name, destination=buffer)
    if inputs is not None and inputs.get(name) != identity:
        raise ValueError('release notice input differs from validated source inputs')
    return buffer.getvalue()


def populate(root, package, models, stub, inputs=None):
    """Install source assets/licenses. This function does not admit binaries."""
    root, package = Path(root), Path(package)
    for kind, origin, destination, _ in plan.stage(models, root):
        if kind == 'asset':
            copy_asset(root, package, origin, destination, inputs)
        elif kind == 'eval':
            variant = [] if destination == '-' else ['--variant', destination]
            libraries = os.environ.get('LD_LIBRARY_PATH', '')
            env = {**os.environ, 'LD_LIBRARY_PATH': str(stub) + (':' + libraries if libraries else '')}
            files = subprocess.check_output([package / 'bin/chopinfer', 'eval', 'files', '--model', origin, *variant],
                                            cwd=root, env=env, text=True).splitlines()
            for path in files:
                copy_asset(root, package, path, path, inputs, 0o644)
    for origin, destination in [('LICENSE', 'LICENSE'), ('fixtures/GSM8K-LICENSE', 'fixtures/GSM8K-LICENSE'),
                                ('tools/release/INSTALL.md', 'README.md')]:
        copy_asset(root, package, origin, destination, inputs, 0o644)
    cargo_home = Path(os.environ.get('CARGO_HOME', Path.home() / '.cargo'))
    lock = tomllib.loads(read_input(root, 'Cargo.lock', inputs).decode())
    (package / 'THIRD_PARTY_NOTICES').write_bytes(notices.render(lock, cargo_home, read_input(root, 'THIRD_PARTY_NOTICES', inputs)))


def checksums(directory):
    entries = []
    for path in sorted(directory.rglob('*')):  # check-inputs: walks outside the repository
        if path.is_symlink():
            raise ValueError('symlink in release archive')
        if path.is_file() and path.name != 'SHA256SUMS':
            identity = source.file_identity(path, 512 << 20)
            entries.append(f"{identity['sha256']}  ./{path.relative_to(directory)}\n")
    (directory / 'SHA256SUMS').write_text(''.join(entries))


def finish(package, output):
    package, output = Path(package), Path(output)
    checksums(package)
    archive = output / (package.name + '.tar.gz')
    if archive.exists():
        raise ValueError('archive already exists')
    def portable(info):
        info.uid = info.gid = 0
        info.uname = info.gname = ''
        return info
    with tarfile.open(archive, 'w:gz') as stream:
        stream.add(package, arcname=package.name, recursive=True, filter=portable)
    identity = source.file_identity(archive, 8 << 30)
    (output / 'SHA256SUMS').write_text(f"{identity['sha256']}  {archive.name}\n")
    return archive
