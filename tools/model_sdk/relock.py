"""Rewrite a workspace's model-owned kernel locks and report every change.

A model build script that locks through `kernel_build::ModelLock` writes its
records to a candidate under the directory named by `RELOCK_ENV` instead of
its lock. Relock passes that directory only as Cargo CLI configuration, which
the model's own Cargo configuration cannot override, and builds once in that
mode. It refuses any candidate outside the model's own tree (every SDK lock
stays read-only and is never rewritten), writes a pending review report,
backs up the old locks, applies the changed locks and verifies the workspace
builds against them with every bypass forced empty. A failed write or
verification restores every old lock; an interrupted run leaves its recovery
state in the logs directory and blocks the next relock until it is resolved.
Shared crates never relock, and ordinary builds force `RELOCK_ENV` empty.
"""
from contextlib import contextmanager
import errno
import json
import os
from pathlib import Path
import secrets
import shutil
import stat
import tempfile

from .graph import PolicyError, metadata, selected_graph
from .source import check, verify_source
from .workflow import (RELOCK_ENV, cargo_config, claim_target, environment, external_output,
                       native_toolchain, package_selection, require, role_packages, run, target_lock,
                       verified_cargo, workspace_lock)

REPORT = 'relock-report.json'
# kernel_build::MODEL_RELOCK_MARKER: ModelLock writes candidates only into a
# directory holding it.
MARKER = '.chopinfer-relock-candidates'
# Recovery state of a relock that is replacing locks.
STATE = 'relock-state'
# Recovery state of an applied relock, renamed out of the way before deletion.
RETIRED_PREFIX = 'relock-applied-'
SCHEMA = 'chopinfer-sdk-relock-v1'
PROVENANCE_SUFFIX = '.provenance.json'
# Report-only fields: the source closure digest keys every source by path and
# content, so it changes with source paths or the SDK revision alone.
PROVENANCE_FIELDS = ('source_closure_sha256',)
CUBIN_FIELD = 'cubin_sha256'


def lock_targets(root, candidates):
    """Map each candidate file to the workspace lock it replaces.

    A candidate mirrors its lock's absolute path. Only files inside the model
    workspace and outside its `sdk/` tree are model-owned; any other target,
    or a path through a symlink, refuses the whole relock before a write."""
    root, candidates = Path(root).resolve(), Path(candidates)
    sdk = root / 'sdk'
    targets = {}
    for candidate in sorted(p for p in candidates.rglob('*') if not p.is_dir() and p != candidates / MARKER):
        require(candidate.is_file() and not candidate.is_symlink(),
                f'relock candidate is not a regular file: {candidate}')
        target = Path('/') / candidate.relative_to(candidates)
        require(target.is_relative_to(root) and not target.is_relative_to(sdk),
                f'relock refuses a lock outside the model\'s own tree (SDK locks are never rewritten): {target}')
        # A lock reached through a symlink could alias an SDK file.
        require(not target.is_symlink() and target.parent.resolve() == target.parent,
                f'relock refuses a lock reached through a symlink: {target}')
        targets[target] = candidate
    return targets


def lock_units(targets):
    """Group a directory-form family's lock and provenance files into one unit."""
    units = {}
    for target in targets:
        name = target.name
        key = target.with_name(name[:-len(PROVENANCE_SUFFIX)] + '.lock') if name.endswith(PROVENANCE_SUFFIX) else target
        units.setdefault(key, []).append(target)
    return {key: sorted(files) for key, files in sorted(units.items())}


def read_records(files):
    """Records by cubin name from `(path, bytes or None)` pairs, with a
    family's provenance merged in.

    The gating lock file is authoritative for every cubin and recipe field. A
    provenance sidecar contributes only its validated provenance fields, so it
    can never mask a gating change, whatever else it carries."""
    records, provenance = {}, {}
    for path, data in files:
        if data is None:
            continue
        entries = json.loads(data)
        require(isinstance(entries, list), f'{path}: a lock holds a list of records')
        sidecar = path.name.endswith(PROVENANCE_SUFFIX)
        for record in entries:
            valid = isinstance(record, dict) and isinstance(record.get('name'), str)
            if sidecar:
                valid = valid and all(isinstance(record.get(f), str) for f in PROVENANCE_FIELDS)
            require(valid, f'{path}: invalid {"provenance" if sidecar else "lock"} record')
            if sidecar:
                provenance[record['name']] = {f: record[f] for f in PROVENANCE_FIELDS}
            else:
                records[record['name']] = dict(record)
    for name, record in records.items():
        record.update(provenance.get(name, {}))
    return records


def compare_records(old, new):
    """Records added, removed and changed between two lock record maps.

    A changed record names its changed fields and whether its cubin bytes
    changed, another gating field changed, or only provenance changed."""
    changed = []
    for name in sorted(set(old) & set(new)):
        fields = sorted(k for k in set(old[name]) | set(new[name]) if old[name].get(k) != new[name].get(k))
        if not fields:
            continue
        gating = [f for f in fields if f not in PROVENANCE_FIELDS]
        change = {'name': name, 'fields': fields,
                  'cubin_changed': CUBIN_FIELD in fields,
                  'kind': 'cubin' if CUBIN_FIELD in fields else 'recipe' if gating else 'provenance'}
        if change['cubin_changed']:
            change['cubin_sha256'] = {'old': old[name][CUBIN_FIELD], 'new': new[name][CUBIN_FIELD]}
        changed.append(change)
    return {'added': sorted(set(new) - set(old)), 'removed': sorted(set(old) - set(new)), 'changed': changed}


def review(root, targets):
    """The report entry of every lock unit, comparing the workspace lock with its candidate."""
    root = Path(root).resolve()
    locks = []
    for key, files in lock_units(targets).items():
        old = [(f, read_lock(f)) for f in files]
        new = [(f, targets[f].read_bytes()) for f in files]
        same = old == new
        existed = any(data is not None for _, data in old)
        entry = {'lock': key.relative_to(root).as_posix(),
                 'files': [f.relative_to(root).as_posix() for f in files],
                 'status': 'unchanged' if same else 'updated' if existed else 'created'}
        entry.update(compare_records(read_records(old), read_records(new)))
        entry['provenance_only'] = (not same and not entry['added'] and not entry['removed']
                                    and all(c['kind'] == 'provenance' for c in entry['changed']))
        locks.append(entry)
    return locks


@contextmanager
def lock_directory(target, *, create=False):
    """The lock's parent, opened one component at a time without following
    symbolic links, so a directory swapped after validation is refused."""
    descriptor = os.open('/', os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        for part in Path(target).parent.parts[1:]:
            if create:
                try:
                    os.mkdir(part, dir_fd=descriptor)
                except FileExistsError:
                    pass
            try:
                child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                                dir_fd=descriptor)
            except OSError as error:
                require(error.errno not in (errno.ELOOP, errno.ENOTDIR),
                        f'relock refuses a lock reached through a symlink: {target}')
                raise
            os.close(descriptor)
            descriptor = child
        yield descriptor
    finally:
        os.close(descriptor)


def read_lock(target):
    """A lock's bytes, or None when it does not exist; never follows a link."""
    try:
        with lock_directory(target) as parent:
            descriptor = os.open(Path(target).name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC,
                                 dir_fd=parent)
    except FileNotFoundError:
        return None
    except OSError as error:
        require(error.errno != errno.ELOOP, f'relock refuses a lock reached through a symlink: {target}')
        raise
    with os.fdopen(descriptor, 'rb') as stream:
        require(stat.S_ISREG(os.fstat(stream.fileno()).st_mode), f'lock is not a regular file: {target}')
        return stream.read()


def write_lock(target, data):
    """Atomically replace one lock relative to its no-follow parent descriptor."""
    name, temporary = Path(target).name, f'.relock-{secrets.token_hex(8)}'
    with lock_directory(target, create=True) as parent:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
                             0o644, dir_fd=parent)
        try:
            with os.fdopen(descriptor, 'wb') as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, name, src_dir_fd=parent, dst_dir_fd=parent)
        except BaseException:
            try:
                os.unlink(temporary, dir_fd=parent)
            except FileNotFoundError:
                pass
            raise
        os.fsync(parent)


def remove_lock(target):
    """Remove a lock relative to its no-follow parent descriptor, if present."""
    try:
        with lock_directory(target) as parent:
            os.unlink(Path(target).name, dir_fd=parent)
    except FileNotFoundError:
        pass


def stage(root, targets, state):
    """Record the old and new bytes of every changed lock before any write.

    `state/manifest.json` maps each lock to its backup under `old/` (null when
    the lock did not exist) and its candidate under `new/`."""
    manifest = {}
    for target, candidate in targets.items():
        old, new = read_lock(target), candidate.read_bytes()
        if old == new:
            continue
        relative = target.relative_to(root).as_posix()
        entry = {'old': None, 'new': f'new/{relative}'}
        if old is not None:
            entry['old'] = f'old/{relative}'
        for key, data in (('old', old), ('new', new)):
            if entry[key]:
                (state / entry[key]).parent.mkdir(parents=True, exist_ok=True)
                (state / entry[key]).write_bytes(data)
        manifest[relative] = entry
    (state / 'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    return manifest


def restore(root, manifest, state):
    """Put back every staged lock's old bytes, or remove a lock relock created.
    Returns the locks that still differ from their old bytes."""
    failed = []
    for relative, entry in manifest.items():
        target = root / relative
        try:
            if entry['old'] is None:
                remove_lock(target)
            else:
                write_lock(target, (state / entry['old']).read_bytes())
            old = None if entry['old'] is None else (state / entry['old']).read_bytes()
            if read_lock(target) != old:
                failed.append(relative)
        except (OSError, PolicyError):
            failed.append(relative)
    return failed


def publish_report(logs, report):
    """Replace the report atomically, so a reader always sees one complete status."""
    temporary = logs / f'.{REPORT}.{secrets.token_hex(8)}'
    try:
        with temporary.open('x') as stream:
            stream.write(json.dumps(report, indent=2) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, logs / REPORT)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def retire(state):
    """Delete an applied relock's recovery state. The atomic rename means an
    interruption leaves either the state its `applied` report names or a
    retired directory, and the next relock finishes either."""
    retired = state.with_name(f'{RETIRED_PREFIX}{secrets.token_hex(8)}')
    os.replace(state, retired)
    shutil.rmtree(retired)


def finish_retirement(logs):
    """Finish the cleanup of a relock interrupted after it applied every lock."""
    for retired in logs.glob(f'{RETIRED_PREFIX}*'):
        shutil.rmtree(retired)
    state, report = logs / STATE, logs / REPORT
    if state.is_dir() and report.is_file() and (state / 'id').is_file():
        published = json.loads(report.read_text())
        if published.get('status') == 'applied' and published.get('state_id') == (state / 'id').read_text():
            retire(state)


def relock(workspace, target, logs, *, features=()):
    root = Path(workspace).resolve()
    target, logs = (external_output(root, p) for p in (target, logs))
    env = environment()
    # Held for the whole run, before the target: one run at a time per workspace.
    with workspace_lock(root):
        state = logs / STATE
        if logs.is_dir():
            finish_retirement(logs)
        if state.exists():
            published = json.loads((logs / REPORT).read_text()) if (logs / REPORT).is_file() else {}
            raise PolicyError(
                f'an interrupted relock (report status {published.get("status")}) left its recovery state in '
                f'{state}: manifest.json maps each lock to '
                'its old bytes under old/ (null: the lock did not exist) and its candidate under new/. '
                'Restore or keep each lock, then remove the directory')
        packages = role_packages(root)
        graph, roots = selected_graph(metadata(root / 'Cargo.toml'), root / 'Cargo.toml',
                                      packages=packages, features=features)
        pin = check(root, features=features, selected=graph)
        require(all(p.stat().st_mode & 0o222 == 0 for p in (root / 'sdk').rglob('*')),
                'SDK inputs must remain read-only; use the exported checkout')
        native = native_toolchain(graph, roots, env)
        logs.mkdir(parents=True, exist_ok=True)
        claim_target(target, root)
        # Held for the whole run, so concurrent invocations never share outputs.
        with target_lock(target):
            env['CARGO_TARGET_DIR'] = str(target)
            command = [env.get('CARGO', 'cargo'), 'build', '--locked', '--offline',
                       '--manifest-path', root / 'Cargo.toml', *package_selection(packages)]
            if features:
                command += ['--features', ','.join(features)]
            # The candidate directory travels only as CLI configuration, which the
            # model's Cargo configuration cannot override; verification forces it empty.
            relock_build = lambda directory: [*command, *cargo_config(directory)]
            report = {'schema': SCHEMA, 'sdk': {k: v for k, v in pin.items() if k != 'files'},
                      'features': list(features), 'nvcc': native['nvcc']}
            def write_report(status, **fields):
                report.update(status=status, **fields)
                publish_report(logs, report)
            # A fresh directory name reruns every model lock's build script.
            with tempfile.TemporaryDirectory(dir=logs, prefix='relock-candidates-') as candidates:
                (Path(candidates) / MARKER).write_bytes(b'')
                try:
                    run(relock_build(candidates), env=env, cwd=root, log=logs / 'relock-build.log')
                finally:
                    verify_source(root)
                targets = lock_targets(root, candidates)
                require(targets, 'no model-owned lock was built; select the features whose build scripts lock kernels')
                locks = review(root, targets)
                state.mkdir()
                state_id = secrets.token_hex(8)
                (state / 'id').write_text(state_id)
                manifest = stage(root, targets, state)
            # From here an interruption leaves the pending report and the recovery state.
            write_report('pending', changed=bool(manifest), written=sorted(manifest), locks=locks,
                         recovery=str(state), state_id=state_id)
            try:
                for relative, entry in manifest.items():
                    write_lock(root / relative, (state / entry['new']).read_bytes())
                verified_cargo(relock_build(''), root, env=env, log=logs / 'relock-verify.log')
                verify_source(root)
                # Published while the backups still exist, so a failure here restores them.
                write_report('applied', recovery=None)
            except BaseException as error:
                # No lock changes back before a durable non-applied status: an
                # interrupted rollback leaves `restoring`, which the next run never
                # cleans up, and a failed publication leaves every lock as it is.
                try:
                    write_report('restoring', error=str(error))
                except Exception as publication:
                    raise PolicyError(f'relock failed ({error}) and could not record its rollback ({publication}); '
                                      'every lock is as the failure left it and the recovery state is in '
                                      f'{state}') from error
                unrestored = restore(root, manifest, state)
                if unrestored:
                    write_report('restore-failed', error=str(error), unrestored=unrestored)
                    raise PolicyError(f'relock failed ({error}) and could not restore {", ".join(unrestored)}; '
                                      f'the old and candidate locks are in {state}') from error
                kept = Path(tempfile.mkdtemp(dir=logs, prefix='relock-failed-'))
                write_report('restored', error=str(error), recovery=None, candidates=str(kept / 'new'))
                os.replace(state, kept)
                if not isinstance(error, Exception):
                    raise
                raise PolicyError(f'relock failed and restored every lock to its old bytes: {error}; '
                                  f'the candidate locks are in {kept / "new"}') from error
            retire(state)
            return report
