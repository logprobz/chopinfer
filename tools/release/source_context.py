"""Source identities for native checkouts and retained-ref Docker contexts."""
import hashlib
import json
import os
from pathlib import Path
import re
import stat

MAX_SOURCE_BYTES = 64 << 20
MAX_INPUTS = 50000
# A clone with many branches packs its refs into one file far larger than a ref.
MAX_PACKED_REFS_BYTES = 16 << 20
IGNORED_DIRECTORIES = {'.git', '.agents', '__pycache__', 'target', 'node_modules'}


def regular(path):
    """Open a bounded input through no-follow directory descriptors."""
    path = Path(os.path.abspath(path))
    parent = os.open('/', os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in path.parts[1:-1]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
            os.close(parent)
            parent = child
        fd = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
    finally:
        os.close(parent)
    if not stat.S_ISREG(os.fstat(fd).st_mode):
        os.close(fd)
        raise ValueError('release input is not a regular file')
    return fd


def file_identity(path, limit=MAX_SOURCE_BYTES, destination=None):
    """Hash one stable regular file, optionally copying its opened bytes."""
    with os.fdopen(regular(path), 'rb') as stream:
        before = os.fstat(stream.fileno())
        if before.st_size > limit:
            raise ValueError('release input exceeds its byte bound')
        digest, size = hashlib.sha256(), 0
        for block in iter(lambda: stream.read(1 << 20), b''):
            size += len(block)
            if size > limit:
                raise ValueError('release input grew beyond its byte bound')
            digest.update(block)
            if destination is not None:
                destination.write(block)
        after = os.fstat(stream.fileno())
        current = os.stat(path, follow_symlinks=False)
        stamp = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
        if stamp(before) != stamp(after) or stamp(after) != stamp(current) or size != after.st_size:
            raise ValueError('release input changed while being read')
        return {'size': size, 'sha256': digest.hexdigest()}


def read_text(path, limit=64 << 10):
    with os.fdopen(regular(path), 'rb') as stream:
        value = stream.read(limit + 1)
    if len(value) > limit:
        raise ValueError('Git metadata exceeds its byte bound')
    return value.decode().strip()


def git_directory(root):
    directory = Path(root) / '.git'
    if directory.is_file():
        text = read_text(directory)
        if not text.startswith('gitdir: '):
            raise ValueError('invalid worktree Git directory')
        directory = Path(text[8:]) if Path(text[8:]).is_absolute() else Path(root) / text[8:]
    return directory.resolve()


def revision(root, supplied=None):
    directory = git_directory(root)
    common = directory
    if (directory / 'commondir').is_file():
        common = (directory / read_text(directory / 'commondir')).resolve()
    try:
        found = read_text(directory / 'HEAD')
        if found.startswith('ref: '):
            ref = found[5:]
            if not re.fullmatch(r'refs/[A-Za-z0-9_./-]+', ref) or '..' in ref:
                raise ValueError('invalid Git HEAD reference')
            loose = common / ref
            found = read_text(loose) if loose.is_file() else ''
            packed = common / 'packed-refs'
            if not found and packed.is_file():
                lines = read_text(packed, MAX_PACKED_REFS_BYTES).splitlines()
                found = next((line.split()[0] for line in lines
                              if len(line.split()) == 2 and line.split()[1] == ref), '')
    except FileNotFoundError:
        found = ''
    if not re.fullmatch('[0-9a-f]{40}', found):
        raise ValueError('full source revision required from Git HEAD')
    supplied = os.environ.get('CHOPINFER_BUILD_REVISION', '') if supplied is None else supplied
    if supplied and supplied != found:
        raise ValueError('CHOPINFER_BUILD_REVISION differs from Git HEAD')
    return found


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def source_roots(root, metadata, packages, models):
    """Local production/build dependencies and release-owned inputs."""
    root = Path(root).resolve()
    by_name = {package['name']: package for package in metadata['packages']}
    by_path = {Path(package['manifest_path']).parent.resolve(): package for package in metadata['packages']}
    selected, pending = set(), [by_name[name] for name in packages]
    while pending:
        package = pending.pop()
        directory = Path(package['manifest_path']).parent.resolve()
        if directory in selected:
            continue
        if not directory.is_relative_to(root):
            raise ValueError('local release dependency lies outside the source context')
        selected.add(directory)
        for dependency in package.get('dependencies', []):
            if dependency.get('path') and dependency.get('kind') != 'dev':
                pending.append(by_path[Path(dependency['path']).resolve()])
    selected |= {root / model.directory for model in models}
    selected |= {root / name for name in ('tools/release', 'tools/model_sdk', 'ci', 'include', 'fixtures', 'deploy/docker')}
    return sorted(selected)


def snapshot(root, roots):
    root = Path(root).resolve()
    # Every model's `chopinfer eval files` ships the shared qualification module metadata.
    files = {root / name for name in ('Cargo.toml', 'Cargo.lock', 'rust-toolchain.toml', 'LICENSE', 'THIRD_PARTY_NOTICES',
                                      'tools/qual/modules.json')}
    files |= set((root / 'models').glob('*/model.toml'))
    if (root / '.dockerignore').is_file():
        files.add(root / '.dockerignore')
    for name in ('config', 'config.toml'):
        if (root / '.cargo' / name).exists():
            files.add(root / '.cargo' / name)
    for selected in roots:
        if selected.exists():
            for directory, children, names in os.walk(selected, followlinks=False):  # check-inputs: walks crates, models, tools, ci, include, fixtures, deploy
                children[:] = sorted(name for name in children if name not in IGNORED_DIRECTORIES)
                if any((Path(directory) / child).is_symlink() for child in children):
                    raise ValueError('symlinked release source directory')
                files |= {Path(directory) / name for name in names if not name.endswith(('.pyc', '.log'))}
                if len(files) > MAX_INPUTS:
                    raise ValueError('release source inventory exceeds its file bound')
    inputs = {str(path.relative_to(root)): file_identity(path) for path in sorted(files)}
    return {'revision': revision(root), 'inputs': inputs, 'sha256': digest(inputs),
            'context': 'git' if (git_directory(root) / 'index').is_file() else 'retained-refs'}
