"""Export one SDK source identity and provision an external workspace lock."""
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tarfile
import tempfile
import tomllib

from .graph import Graph, PolicyError, check_external, metadata, selected_graph
from .model_export import export_model

PIN = 'chopinfer-sdk-source-v1'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def tree(root):
    files = {}
    for path in sorted(Path(root).rglob('*')):
        if '.git' in path.relative_to(root).parts:
            continue
        if path.is_symlink():
            raise PolicyError(f'SDK source symlink is unsupported: {path.relative_to(root)}')
        if path.is_file():
            files[path.relative_to(root).as_posix()] = {
                'sha256': sha(path), 'executable': bool(path.stat().st_mode & 0o111)}
    return files


def identity(files):
    return hashlib.sha256(json.dumps(files, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def git(root, *args):
    return subprocess.check_output(['git', '-C', str(root), *args])


def source_paths(root, includes):
    paths = git(root, 'ls-files', '-z').decode().split('\0')
    for relative in [*paths, *includes]:
        if not relative:
            continue
        path = Path(relative)
        if path.is_absolute() or '..' in path.parts or '.git' in path.parts:
            raise PolicyError('source inclusion must stay inside the checkout')
        source = root / path
        if not source.resolve().is_relative_to(root):
            raise PolicyError(f'SDK source symlink or path escapes checkout: {relative}')
        if source.is_symlink():
            raise PolicyError(f'SDK source symlink is unsupported: {relative}')
        if source.is_file():
            yield source, relative


def export(source, output, *, revision='HEAD', development=False, includes=(), template=None, model=None):
    """Export the SDK at REVISION of SOURCE into OUTPUT/sdk beside a model
    workspace: a copy of TEMPLATE (the external example by default), or the
    in-tree model directory MODEL (models/<m>) at that revision laid out as a
    standalone workspace (model_export)."""
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.is_relative_to(source) or source.is_relative_to(output):
        raise PolicyError('export must use an unrelated external directory')
    if output.exists():
        raise PolicyError('export destination already exists')
    if template and model:
        raise PolicyError('export takes a template or a model directory, not both')
    template = Path(template).resolve() if template else source / 'examples/external-model'
    if not development and git(source, 'status', '--porcelain', '--untracked-files=all').strip():
        raise PolicyError('release export requires a clean source checkout')
    revision = git(source, 'rev-parse', '--verify', revision + '^{commit}').decode().strip()
    if development and revision != git(source, 'rev-parse', 'HEAD').decode().strip():
        raise PolicyError('development export must use the current working tree')
    if model:
        output.mkdir(parents=True)
    else:
        # The template contains sources only. Local exports and build outputs stay out.
        ignore = shutil.ignore_patterns('sdk', 'target', 'Cargo.lock', 'sdk-pin.json', '.chopinfer-sdk-lock')
        shutil.copytree(template, output, ignore=ignore)
    sdk = output / 'sdk'
    sdk.mkdir()
    if development:
        for path, relative in source_paths(source, includes):
            destination = sdk / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
    else:
        with tempfile.TemporaryFile() as archive:
            subprocess.run(['git', '-C', str(source), 'archive', revision], stdout=archive, check=True)
            archive.seek(0)
            with tarfile.open(fileobj=archive) as tar:
                for member in tar.getmembers():
                    if not member.isfile() and not member.isdir():
                        raise PolicyError('SDK archive must contain regular source files')
                tar.extractall(sdk, filter='data')
    files = tree(sdk)
    pin = {'schema': PIN, 'mode': 'development' if development else 'release',
           'revision': revision, 'source_sha256': identity(files), 'files': files}
    (output / 'sdk-pin.json').write_text(json.dumps(pin, sort_keys=True, indent=2) + '\n')
    if model:
        export_model(sdk, output, model)
    # Read-only source permissions complement the read-only container mount and
    # post-build identity verification. Writable targets are always elsewhere.
    for path in sdk.rglob('*'):
        path.chmod(path.stat().st_mode & ~0o222)
    sdk.chmod(sdk.stat().st_mode & ~0o222)
    return output


def verify_source(root, *, release=False):
    root = Path(root).resolve()
    pin = json.loads((root / 'sdk-pin.json').read_text())
    if pin.get('schema') != PIN or pin.get('mode') not in ('development', 'release'):
        raise PolicyError('unsupported SDK source pin')
    if release and pin['mode'] != 'release':
        raise PolicyError('development SDK identities cannot qualify as release pins')
    files = tree(root / 'sdk')
    if files != pin['files'] or identity(files) != pin['source_sha256']:
        raise PolicyError('pinned SDK sources changed; export a new SDK revision')
    return pin


def validate_lock(root, data):
    root = Path(root)
    sdk = (root / 'sdk').resolve()
    locks = [tomllib.loads((p / 'Cargo.lock').read_text())['package'] for p in (sdk, root)]
    def registry(lock):
        return {(p['name'], p['version'], p.get('source')): p.get('checksum')
                for p in lock if p.get('source', '').startswith('registry+')}
    expected, actual = map(registry, locks)
    graph = Graph(data)
    reached = set()
    for key, package in graph.packages.items():
        if Path(package['manifest_path']).resolve().is_relative_to(sdk):
            reached |= graph.closure(key)
    for key in reached:
        p = graph.packages[key]
        if (p.get('source') or '').startswith('registry+'):
            entry = (p['name'], p['version'], p['source'])
            if entry not in expected or entry not in actual or actual[entry] != expected[entry]:
                raise PolicyError(f'SDK registry dependency differs from its tested lock: {entry[0]} {entry[1]}')


def check(root, *, release=False, features=(), contributor_roots=(), selected=None):
    root = Path(root).resolve()
    pin = verify_source(root, release=release)
    data = metadata(root / 'Cargo.toml')
    if contributor_roots and selected is None:
        selected, _ = selected_graph(data, root / 'Cargo.toml', features=features)
    check_external(data, root / 'Cargo.toml', root / 'model.toml', root / 'sdk',
                   selected=selected, contributor_roots=contributor_roots)
    validate_lock(root, data)
    return pin


def resolve_lock(root):
    """Let Cargo resolve ROOT's workspace offline from the Cargo.lock in place."""
    subprocess.run([os.environ.get('CARGO', 'cargo'), 'metadata', '--offline', '--all-features',
                    '--format-version', '1', '--manifest-path', str(root / 'Cargo.toml')],
                   cwd=root, stdout=subprocess.DEVNULL, check=True)


def provision_lock(root):
    root = Path(root).resolve()
    verify_source(root)
    if (root / 'Cargo.lock').exists():
        raise PolicyError('lock provisioning cannot replace an existing Cargo.lock; '
                          'use lock --update after changing the model\'s dependencies')
    shutil.copyfile(root / 'sdk/Cargo.lock', root / 'Cargo.lock')
    # Bootstrap only. Cargo prunes core-only packages and adds the model roles.
    # Subsequent metadata, builds and tests always use --locked --offline.
    resolve_lock(root)
    return check(root)


def lock_packages(text):
    """The [[package]] entries of Cargo.lock TEXT: its header and a map from
    each package's (name, version, source) to its text."""
    header, *blocks = re.split(r'\n(?=\[\[package\]\]\n)', text)
    packages = {}
    for block in blocks:
        data = tomllib.loads(block)
        if set(data) != {'package'} or len(data['package']) != 1:
            raise PolicyError('Cargo.lock holds a section other than [[package]] entries')
        entry = data['package'][0]
        packages[(entry['name'], entry['version'], entry.get('source'))] = block.rstrip('\n') + '\n'
    return header.rstrip('\n') + '\n', packages


def update_lock(root):
    """Resolve ROOT's existing Cargo.lock again after the model's own
    dependencies changed. Cargo starts from the SDK's tested lock plus the
    model's own registry packages, so the SDK's pins are kept; the update is
    refused, and the previous lock restored, when the result changes one or
    fails `check`. Returns the check result with the packages added to and
    removed from the lock and those whose dependencies changed."""
    from .workflow import workspace_lock

    root = Path(root).resolve()
    verify_source(root)
    lock = root / 'Cargo.lock'
    if not lock.is_file() or lock.is_symlink():
        raise PolicyError('lock --update needs an existing Cargo.lock; run lock first')
    with workspace_lock(root):
        previous = lock.read_bytes()
        header, sdk = lock_packages((root / 'sdk/Cargo.lock').read_text())
        _, current = lock_packages(previous.decode())
        own = [text for key, text in current.items()
               if key not in sdk and (key[2] or '').startswith('registry+')]
        try:
            lock.write_text(header + ''.join('\n' + text for text in [*sdk.values(), *own]))
            resolve_lock(root)
            result = check(root)
        except BaseException as error:
            lock.write_bytes(previous)
            if isinstance(error, (PolicyError, subprocess.CalledProcessError)):
                raise PolicyError(f'lock --update kept the previous Cargo.lock: {error}') from error
            raise
        _, updated = lock_packages(lock.read_text())
    def names(keys):
        keys = sorted(keys, key=lambda key: (key[0], key[1], key[2] or ''))
        return [f'{name} {version}' + (f' ({source})' if source else '') for name, version, source in keys]
    return {**result, 'added': names(updated.keys() - current.keys()),
            'removed': names(current.keys() - updated.keys()),
            'changed': names(key for key in current.keys() & updated.keys() if current[key] != updated[key])}
