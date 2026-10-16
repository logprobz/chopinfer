#!/usr/bin/env python3
"""Capture and seal every pinned journal of one revision from a disposable
copy of its tree, with no GPUs.

Usage: capture.py --repo REPO --model ID --revision REV --out NEW_DIR
       [--runner libtest|nextest] [--jobs N] [--allow-ignored FILE]

The archive's bundle lib tests run once, with the journal dump and the
fake-host trace patched into the archive copy only. Every dumped journal
must match its pin, and the pins must be exactly the bundle sources'
literal pin tuples. The bodies, the device traces (aux/) and a sealed
v3 index (index.json) form the capture. A capture's journals, aux/ and
index.json depend only on the revision and the helper, not on the runner
or the number of jobs. See README.md.
"""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from auxiliary import collect as collect_auxiliary
from common import GIT, GIT_ENV, git, need, resolve
from instrument import patch as patch_archive
from pins import JOURNAL, bundle_root, dumped, read_regular, read_site
from provenance import HELPER_FILES, digest, helper_identity, seal, sealing_key

ROOT = Path(__file__).resolve().parent
PRODUCER = 'tools/journal_proof/capture.py'
TIME = Path('/usr/bin/time')
NEXTEST_PROFILE = '''
# Added to the archive copy by tools/journal_proof/capture.py.
[profile.journal-capture]
default-filter = "all()"
fail-fast = false
retries = 0

[profile.journal-capture.junit]
path = "junit.xml"
'''


def helper_source(require_clean=True):
    """The commit of the checkout holding this helper. With `require_clean`,
    every helper file's bytes must equal its blob at that commit (git status
    alone can be told to look away, by assume-unchanged or skip-worktree)."""
    commit = git(ROOT, 'rev-parse', 'HEAD').decode().strip()
    prefix = git(ROOT, 'rev-parse', '--show-prefix').decode().strip()
    changed = []
    for name in HELPER_FILES:
        try:
            committed = git(ROOT, 'cat-file', 'blob', f'{commit}:{prefix}{name}')
        except ValueError:
            committed = None
        if committed != (ROOT / name).read_bytes():
            changed.append(name)
    need(not (require_clean and changed), f'helper files differ from their committed bytes at {commit[:12]}: {changed}')
    return dict(commit=commit, clean=not changed)


def no_worktree(path):
    existing = Path(path).resolve()
    while not existing.exists():
        existing = existing.parent
    probe = subprocess.run([*GIT, '-C', str(existing), 'rev-parse', '--is-inside-work-tree'], capture_output=True, text=True, timeout=10)
    need(probe.returncode != 0 or probe.stdout.strip() != 'true', f'path is inside a live worktree: {path}')


def archive(repo, commit, destination):
    """Write `commit`'s tree into `destination` from its blob objects. Unlike
    `git archive`, no attribute (export-subst, export-ignore, from the tree or
    from info/attributes) can change or drop a file: every file holds its
    committed bytes. Only regular files are supported."""
    entries = []
    for entry in git(repo, 'ls-tree', '-r', '-z', '--full-tree', commit).split(b'\0'):
        if not entry:
            continue
        meta, name = entry.split(b'\t', 1)
        mode, kind, oid = meta.decode().split()
        path = Path(name.decode())
        need(kind == 'blob' and mode in ('100644', '100755'), f'unsupported tree entry {mode} {kind}: {path}')
        need(not path.is_absolute() and not {'..', '.', '.git'} & set(path.parts), f'unsafe tree entry: {path}')
        entries.append((mode, oid, path))
    need(entries, f'{commit}: empty tree')
    process = subprocess.Popen([*GIT, '-C', str(repo), 'cat-file', '--batch'], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, env=GIT_ENV)

    def feed():
        try:
            process.stdin.write(b''.join(f'{oid}\n'.encode() for _, oid, _ in entries))
            process.stdin.close()
        except BrokenPipeError:
            pass
    writer = threading.Thread(target=feed, daemon=True)
    writer.start()
    try:
        for mode, oid, path in entries:
            header = process.stdout.readline().split()
            need(len(header) == 3 and header[0].decode() == oid and header[1] == b'blob', f'{path}: blob {oid} not read')
            data = process.stdout.read(int(header[2]))
            need(len(data) == int(header[2]) and process.stdout.read(1) == b'\n', f'{path}: blob {oid} truncated')
            target = destination / path
            target.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o755 if mode == '100755' else 0o644)
            with os.fdopen(fd, 'wb') as output:
                output.write(data)
    finally:
        process.stdout.close()
        try:
            process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        writer.join(timeout=60)
        # Every pipe is closed on every path, the writer's too when it
        # stopped on a broken pipe.
        try:
            process.stdin.close()
        except BrokenPipeError:
            pass
        errors = process.stderr.read()
        process.stderr.close()
    need(process.returncode == 0, 'git cat-file failed: ' + errors.decode(errors='replace')[:300])


def add_nextest_profile(source):
    path = source / '.config/nextest.toml'
    before = path.read_text() if path.exists() else None
    path.parent.mkdir(exist_ok=True)
    path.write_text((before or '') + NEXTEST_PROFILE)
    return dict(path='.config/nextest.toml', before_sha256=hashlib.sha256(before.encode()).hexdigest() if before is not None else None,
                after_sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def stop_owned(process):
    # The leader is kept unreaped until the group signals are sent.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        members = []
        for p in Path('/proc').glob('[0-9]*/stat'):
            try:
                fields = p.read_text().rsplit(')', 1)[1].split()
                if int(fields[2]) == process.pid and fields[0] not in ('Z', 'X'):
                    members.append(p)
            except (OSError, ValueError, IndexError):
                pass
        if not members:
            break
        time.sleep(.05)
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    return process.wait(timeout=10)


def commands(package, runner, jobs):
    """The build, test and (nextest) list commands, after `cargo`."""
    common = ['-p', package, '--release', '--locked', '--offline', '--lib']
    if runner == 'libtest':
        return dict(build=['test', *common, '--no-run'],
                    test=['test', *common, '--no-fail-fast', '--', f'--test-threads={jobs}'])
    return dict(build=['nextest', 'run', *common, '--no-run'],
                test=['nextest', 'run', *common, '--no-fail-fast', '--profile', 'journal-capture', '--test-threads', str(jobs)],
                list=['nextest', 'list', *common, '--profile', 'journal-capture', '--message-format', 'json'])


SCRIPT = '''set -euo pipefail
set -a
source "$1"
source "$2/ci/rust-profile.env"
set +a
cd "$2"
export CARGO_TARGET_DIR="$3" CHOPINFER_JOURNAL_DUMP="$4"
export CARGO_NET_OFFLINE=true CARGO_INCREMENTAL=0
export CARGO_BUILD_JOBS=8 OMP_NUM_THREADS=4
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=
unset CHOPINFER_KERNEL_GPU CHOPINFER_KERNEL_GPUS KERNELS_LOCK_UPDATE RUST_TEST_THREADS
export CHOPINFER_NVCC_WRAPPER="$2/tools/nvcc_cache/nvcc_cache.py"
cargo=$5 builds=$6 tests=$7 lists=$8
shift 8
build=("${@:1:$builds}")
test=("${@:$((builds + 1)):$tests}")
list=("${@:$((builds + tests + 1)):$lists}")
/usr/bin/time -o "$CHOPINFER_JOURNAL_DUMP/build.time" -f '%e %M' "$cargo" "${build[@]}"
export CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_RUNNER="/usr/bin/time -a -o $CHOPINFER_JOURNAL_DUMP/test-processes.tsv -f %e\\t%M\\t%C"
/usr/bin/time -o "$CHOPINFER_JOURNAL_DUMP/tests.time" -f '%e %M' "$cargo" "${test[@]}"
unset CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_RUNNER
if [ ${#list[@]} -gt 0 ]; then
  "$cargo" "${list[@]}" > "$CHOPINFER_JOURNAL_DUMP/nextest-list.json"
  cp target/nextest/journal-capture/junit.xml "$CHOPINFER_JOURNAL_DUMP/junit.xml"
fi
'''


def run_build(source, target, out, local_env, cargo, package, runner, jobs, timeout, heavy, target_root, minimum_free_gib,
              allowed_ignored=None):
    """Build and run the archive's bundle lib tests in one process group,
    through the heavy-command wrapper when one is configured. Returns the
    passing test count and the build and test timings."""
    profile = source / 'ci/rust-profile.env'
    need(local_env.is_file(), 'configured local.env is missing; pass --local-env')
    need(profile.is_file(), 'archive has no ci/rust-profile.env')
    need(TIME.is_file(), f'{TIME} is required for the build and test timings')
    phases = commands(package, runner, jobs)
    lists = phases.get('list', [])
    command = ['bash', '-c', SCRIPT, 'journal-proof-build', str(local_env), str(source), str(target), str(out), str(cargo),
               str(len(phases['build'])), str(len(phases['test'])), str(len(lists)), *phases['build'], *phases['test'], *lists]
    if heavy is not None:
        need(Path(heavy).is_file(), f'heavy command wrapper is missing: {heavy}')
        command = ['bash', str(heavy), *command]
    (out / 'build-command.json').write_text(json.dumps(command, indent=2) + '\n')
    with (out / 'bundle-tests.log').open('w') as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        (out / 'build-process.json').write_text(json.dumps(dict(pid=process.pid, pgid=process.pid)) + '\n')
        deadline = time.monotonic() + timeout
        next_disk_check = time.monotonic()
        try:
            while os.waitid(os.P_PID, process.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT) is None:
                if time.monotonic() >= deadline:
                    raise TimeoutError('archive capture build deadline')
                if time.monotonic() >= next_disk_check:
                    next_disk_check = time.monotonic() + 10
                    if shutil.disk_usage(target_root).free < minimum_free_gib * 1024**3:
                        raise RuntimeError(f'disk free space fell below {minimum_free_gib} GiB during build')
                time.sleep(.2)
        finally:
            previous = {s: signal.signal(s, signal.SIG_IGN) for s in (signal.SIGTERM, signal.SIGINT)}
            try:
                code = stop_owned(process)
            finally:
                for s, handler in previous.items():
                    signal.signal(s, handler)
    need(code == 0, f'bundle tests exited {code}; see bundle-tests.log')
    if runner == 'libtest':
        passed = test_results((out / 'bundle-tests.log').read_text(), allowed_ignored)
    else:
        passed = nextest_results((out / 'junit.xml').read_text(), (out / 'nextest-list.json').read_text(), package, allowed_ignored)
    timings = {}
    for phase in ('build', 'tests'):
        seconds, kib = (out / f'{phase}.time').read_text().split()[-2:]
        timings[phase] = dict(seconds=float(seconds), peak_rss_kib=int(kib))
    return passed, timings


def read_allowed_ignored(path):
    """The exact test paths a capture may report as ignored, one per line in
    `path`: no patterns, no duplicates, at least one. The list and its hash
    come from one read of the file, so a file replaced meanwhile cannot
    pair one list with another's hash."""
    data = Path(path).read_bytes()
    lines = data.decode().splitlines()
    tests = [line.strip() for line in lines if line.strip()]
    need(tests, f'allowed ignored list is empty: {path}')
    need(all(not any(c in test for c in ' \t*?[]{}!') for test in tests),
         f'allowed ignored list names patterns or spaces, not exact test paths: {path}')
    need(len(set(tests)) == len(tests), f'allowed ignored list repeats a test: {path}')
    return dict(path=str(Path(path).absolute()), sha256=hashlib.sha256(data).hexdigest(), tests=sorted(tests))


def check_ignored(ignored, allowed_ignored):
    allowed = None if allowed_ignored is None else allowed_ignored['tests']
    need(len(set(ignored)) == len(ignored), 'bundle tests reported an ignored test twice')
    if allowed is None:
        need(not ignored, f'bundle tests ignored tests and no allowed list names them: {sorted(ignored)[:5]}')
        return
    unlisted = sorted(set(ignored) - set(allowed))
    need(not unlisted, f'bundle tests ignored tests the allowed list does not name: {unlisted[:5]}')
    missing = sorted(set(allowed) - set(ignored))
    need(not missing, f'allowed ignored tests ran or no longer exist: {missing[:5]}')


def merge_checks(out):
    """Fold the run's checks into one dump per journal. Each test dumps its
    checks under checks/<test>/ (body, .pin, .site) and its trace checkpoint
    under aux/tests/<test>/. A journal several tests check must have the same
    body and pin in every check. Its dump and checkpoint come from the test
    last in name order, the check a serial run makes last, so the result never
    depends on scheduling. Returns the other checks, {journal: [[test, path,
    line], ...]}, whose call sites count as reached."""
    root = out / 'checks'
    checks = {}
    for directory in sorted(root.iterdir()) if root.is_dir() else []:
        need(directory.is_dir() and not directory.is_symlink(), f'unexpected check entry {directory.name}')
        for path in sorted(directory.iterdir()):
            name = path.name
            if name.endswith(('.pin', '.site')):
                continue
            need(JOURNAL.fullmatch(name) is not None, f'unsafe journal name {name}')
            checkpoint = out / 'aux/tests' / directory.name / f'{name}.checkpoint.json'
            need(checkpoint.is_file(), f'{name}: no trace checkpoint for its check in {directory.name}')
            test = json.loads(checkpoint.read_text())['test']
            need(test.replace('::', '.') == directory.name, f'{name}: checkpoint test {test} is not {directory.name}')
            for suffix in ('.pin', '.site'):
                need((directory / f'{name}{suffix}').is_file(), f'{name}: no {suffix} sidecar for its check in {test}')
            checks.setdefault(name, []).append((test, directory, checkpoint))
    repeated = {}
    for name, made in sorted(checks.items()):
        made.sort(key=lambda check: check[0])
        first = [read_regular(made[0][1] / f'{name}{suffix}') for suffix in ('', '.pin')]
        for test, directory, _ in made[1:]:
            need([read_regular(directory / f'{name}{suffix}') for suffix in ('', '.pin')] == first,
                 f'{name}: {made[0][0]} and {test} checked different bodies or pins')
        test, directory, checkpoint = made[-1]
        for suffix in ('', '.pin', '.site'):
            need(not (out / f'{name}{suffix}').exists(), f'{name}: dumped outside a test check directory')
            shutil.copyfile(directory / f'{name}{suffix}', out / f'{name}{suffix}')
        shutil.copyfile(checkpoint, out / 'aux' / f'{name}.checkpoint.json')
        if made[:-1]:
            repeated[name] = [[other, *read_site(path / f'{name}.site')] for other, path, _ in made[:-1]]
    if root.exists():
        shutil.rmtree(root)
    return repeated


def test_results(log, allowed_ignored=None):
    """The passing test count of a libtest log. The run's own summary is its
    last `test result` line, and it must count every test its first
    `running N tests` line announced: at least one passed, none failed,
    measured or filtered out. Earlier summaries come from child runs a test
    starts itself (one test, the rest filtered out); none of them may fail.
    No test may be ignored unless `allowed_ignored` (read_allowed_ignored)
    lists them, and then the tests the log reports ignored are exactly the
    listed ones."""
    import re
    results = re.findall(r'test result: (\w+)\. (\d+) passed; (\d+) failed; (\d+) ignored; (\d+) measured; (\d+) filtered out;', log)
    announced = re.search(r'^running (\d+) tests?$', log, re.MULTILINE)
    need(results and announced, 'bundle tests did not report a test run')
    status, passed, failed, ignored, measured, filtered = results[-1]
    need(status == 'ok' and int(passed) > 0 and (failed, measured, filtered) == ('0', '0', '0')
         and int(passed) + int(ignored) == int(announced[1]),
         'bundle tests did not report nonzero passing tests with no failures or filters')
    need(all(row[0] == 'ok' and row[2] == '0' for row in results[:-1]), 'a child test run failed')
    names = re.findall(r'^test (\S+) \.\.\. ignored(?:,.*)?$', log, re.MULTILINE)
    need(len(names) == int(ignored), 'bundle tests reported ignored tests that the log does not name')
    check_ignored(names, allowed_ignored)
    return int(passed)


def nextest_results(junit, listing, package, allowed_ignored=None):
    """The passing test count of a nextest run. `nextest list` (same profile)
    gives the package's whole lib test inventory. Its ignored tests follow
    the libtest rule, and the JUnit report must hold exactly every other
    test, once each, all passed: a test a filter or profile skipped, or one
    the report adds, is refused."""
    import xml.etree.ElementTree as ElementTree
    suites = [suite for suite in json.loads(listing)['rust-suites'].values() if suite['package-name'] == package]
    need(len(suites) == 1, f'nextest lists {len(suites)} test binaries of {package}, not its lib tests alone')
    cases = suites[0]['testcases']
    ignored = [name for name, case in cases.items() if case['ignored']]
    check_ignored(ignored, allowed_ignored)
    expected = sorted(name for name, case in cases.items() if not case['ignored'])
    ran, failed = [], []
    for case in ElementTree.fromstring(junit).iter('testcase'):
        ran.append(case.get('name'))
        if any(child.tag in ('failure', 'error', 'skipped') for child in case):
            failed.append(case.get('name'))
    need(expected and not failed, f'bundle tests did not all pass: {failed[:5]}')
    need(sorted(ran) == expected, 'the nextest run differs from the listed tests: '
         f'not run {sorted(set(expected) - set(ran))[:5]}, unlisted or repeated {sorted(set(r for r in ran if ran.count(r) > 1 or r not in expected))[:5]}')
    return len(ran)


def default_lock():
    data = os.environ.get('CHOPINFER_DATA')
    return Path(data) / 'journal-proof/capture.lock' if data else None


def capture(repo, model, revision, out, local_env, runner='libtest', jobs=1, cargo='cargo', scratch=None, target_root=None,
            heavy=None, timeout=7200, minimum_free_gib=150, start_free_gib=180, allowed_ignored=None, lock=None,
            require_clean=True):
    need(runner in ('libtest', 'nextest') and type(jobs) is int and jobs >= 1, 'invalid runner or jobs')
    sealing_key()  # refuse before the archive and build when there is no key to seal with
    source_identity = helper_source(require_clean)
    helper_at_start = helper_identity(ROOT)
    commit = resolve(repo, revision)
    bundle = bundle_root(repo, commit, model).rstrip('/')
    import tomllib
    package = tomllib.loads(git(repo, 'show', f'{commit}:{bundle}/Cargo.toml').decode())['package']['name']
    out = Path(out).absolute()
    target_root = Path(target_root or os.environ.get('CHOPINFER_SCRATCH') or tempfile.gettempdir())
    no_worktree(out)
    no_worktree(scratch or tempfile.gettempdir())
    no_worktree(target_root)
    need(target_root.is_dir(), f'target root is missing: {target_root}')
    free = shutil.disk_usage(target_root).free
    need(free >= max(start_free_gib, minimum_free_gib) * 1024**3,
         f'less than required free space on the target volume ({free / 1024**3:.0f} GiB, need {start_free_gib})')
    held = None
    if lock is not None:
        Path(lock).parent.mkdir(parents=True, exist_ok=True)
        held = open(lock, 'w')
        fcntl.flock(held, fcntl.LOCK_EX)
    try:
        out.mkdir(parents=True, exist_ok=False)
        allowed = None if allowed_ignored is None else read_allowed_ignored(allowed_ignored)
        record = dict(schema='chopinfer.journal-capture.v2', model=model, revision=commit, complete=False, runner=runner, jobs=jobs,
                      helper_source=source_identity, started=datetime.datetime.now(datetime.timezone.utc).isoformat())
        if allowed is not None:
            record['allowed_ignored'] = allowed
        try:
            with tempfile.TemporaryDirectory(prefix='journal-proof-archive-', dir=scratch) as source_name, \
                 tempfile.TemporaryDirectory(prefix='journal-proof-target-', dir=target_root) as target_name:
                source, target = Path(source_name), Path(target_name)
                record.update(archive=str(source), target=str(target))
                archive(repo, commit, source)
                record['patches'] = patch_archive(source, bundle)
                if runner == 'nextest':
                    record['runner_patch'] = add_nextest_profile(source)
                started = time.monotonic()
                record['tests_passed'], record['timings'] = run_build(
                    source, target, out, Path(local_env).absolute(), cargo, package, runner, jobs, timeout, heavy, target_root,
                    minimum_free_gib, allowed)
                record['timings']['wall_seconds'] = round(time.monotonic() - started, 1)
                repeated = merge_checks(out)
                pins, bodies, sites = dumped(out, repo, commit, bundle + '/', repeated)
                auxiliary = collect_auxiliary(out, dict(commit=commit, pins=set(pins)), bodies)
                need(helper_identity(ROOT) == helper_at_start, 'helper files changed during capture; no provenance seal issued')
                helper_digest, helper_files = helper_at_start
                payload = dict(schema='chopinfer.journal-captures.v3', model=model, revision=commit, archive_commit=commit,
                               producer=PRODUCER, real_build=cargo == 'cargo', trace_format=2,
                               helper_digest=helper_digest, helper_files=helper_files, helper_commit=source_identity['commit'],
                               journals={name: name for name in sorted(pins)},
                               journal_sha256={name: digest(out / name) for name in sorted(pins)},
                               pins={name: pins[name] for name in sorted(pins)},
                               sites={name: sites[name] for name in sorted(sites)},
                               repeated_checks={name: repeated[name] for name in sorted(repeated)},
                               auxiliary=dict(auxiliary, path='aux/manifest.json'),
                               auxiliary_files={str(p.relative_to(out)): digest(p)
                                                for p in sorted((out / 'aux').rglob('*')) if p.is_file()},
                               patches=record['patches'], tests_passed=record['tests_passed'])
                if allowed is not None:
                    payload['allowed_ignored'] = dict(sha256=allowed['sha256'], tests=allowed['tests'])
                (out / 'index.json').write_text(json.dumps(seal(payload), indent=2, sort_keys=True) + '\n')
                record.update(complete=True, count=len(pins), index=dict(path=str(out / 'index.json'), sha256=digest(out / 'index.json')))
            record['archive_removed'] = not Path(record['archive']).exists()
            record['target_removed'] = not Path(record['target']).exists()
        except BaseException as error:
            record['complete'] = False
            record['error'] = f'{type(error).__name__}: {error}'
            record['archive_removed'] = 'archive' not in record or not Path(record['archive']).exists()
            record['target_removed'] = 'target' not in record or not Path(record['target']).exists()
            raise
        finally:
            record['finished'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
            (out / 'capture.json').write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')
    finally:
        if held is not None:
            held.close()
    return record


def main():
    os.umask(0o022)
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--repo', required=True, type=Path)
    p.add_argument('--model', required=True)
    p.add_argument('--revision', required=True)
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--runner', choices=['libtest', 'nextest'], default='libtest',
                   help='libtest: one test process (--jobs test threads); nextest: one process per test')
    p.add_argument('--jobs', type=int, default=1, help='tests run at once (default 1)')
    p.add_argument('--allow-ignored', type=Path,
                   help='file of the exact test paths the run may report ignored, one per line; the run must ignore exactly these')
    p.add_argument('--local-env', type=Path, default=Path.home() / '.config/chopinfer/local.env')
    p.add_argument('--scratch', type=Path, help='archive scratch directory outside all live worktrees')
    p.add_argument('--target-root', type=Path, help='parent of the private Cargo target (default $CHOPINFER_SCRATCH)')
    p.add_argument('--heavy', type=Path, default=os.environ.get('CHOPINFER_HEAVY_WRAPPER'),
                   help='wrapper that admits heavy commands (default $CHOPINFER_HEAVY_WRAPPER; none runs directly)')
    p.add_argument('--lock', type=Path, default=default_lock(),
                   help='capture lock file, held for the whole capture (default $CHOPINFER_DATA/journal-proof/capture.lock)')
    p.add_argument('--timeout', type=int, default=7200)
    p.add_argument('--minimum-free-gib', type=int, default=150, help='stop the build below this much free space')
    p.add_argument('--start-free-gib', type=int, default=180, help='refuse to start below this much free space')
    args = p.parse_args()
    need(args.timeout > 0, 'timeout must be positive')

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'interrupted by signal {signum}')
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, interrupted)
    try:
        result = capture(args.repo, args.model, args.revision, args.out, args.local_env, args.runner, args.jobs,
                         scratch=args.scratch, target_root=args.target_root, heavy=args.heavy, timeout=args.timeout,
                         minimum_free_gib=args.minimum_free_gib, start_free_gib=args.start_free_gib,
                         allowed_ignored=args.allow_ignored, lock=args.lock)
    except (OSError, ValueError, TimeoutError, RuntimeError, KeyboardInterrupt) as error:
        print(f'CAPTURE FAILED: {error}', file=sys.stderr)
        return 2
    allowed = result.get('allowed_ignored')
    listed = f'; {len(allowed["tests"])} listed tests ignored ({allowed["sha256"][:16]})' if allowed else ''
    t = result['timings']
    print(f'CAPTURE AUTHENTICATED: {result["count"]} journals; {result["tests_passed"]} tests{listed}; '
          f'build {t["build"]["seconds"]:.0f} s, tests {t["tests"]["seconds"]:.0f} s at {t["tests"]["peak_rss_kib"] // 1024} MiB peak; '
          'archive and target removed')
    return 0


if __name__ == '__main__':
    sys.exit(main())
