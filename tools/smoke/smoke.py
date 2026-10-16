#!/usr/bin/env python3
"""Base-against-tip serving smoke of one model, through tools/ab/serving.

  smoke.py legs MODEL                     the model's serve profiles and the default legs
  smoke.py paths MODEL [--repo DIR] [--target DIR]
                                          pathspecs whose change can change the served model
  smoke.py build MODEL BASE TIP --out DIR [--legs PROFILE,...] [--checkpoint DIR] [--phases P,...]
                 [--job-prefix NAME] [--start-seconds S] [--jobs N]
  smoke.py run DIR LEG...                 inside a GPU job only (the job files call it)
  smoke.py prepare-qualification DIR      pin history and preflight an existing pair, on CPU

A leg is a serve profile of models/MODEL/profiles.json, served whole through
the runner's serve_profile (every numerical option, checked against /health
and the prepared package). The defaults are the TP4 autoregressive, TP4
speculative and TP2 autoregressive profiles the model declares (a qualified
one first). A speculative profile serves at its draft depth in fixed mode, so
the verify shapes do not depend on timings.

`build` (CPU only) exports each commit with `git archive | tar --touch`,
builds the server, the model's compiler and its bundle in the arm's own
target, installs them with an INSTALL.json receipt, prepares a direct package
per tensor-parallel size with the arm's own compiler, and requires a second
build to compile nothing. An arm already READY is kept only when it is the
requested commit, model, checkpoint and build and its install passes the
runner's check. It then pins the runner (tools/ab/serving and this file)
under DIR/runner, writes one smoke run file per leg (each install bound to
its commit), dry-runs each, writes DIR/paths.txt (the smoke-staleness
pathspecs of the tip: the local packages the build compiles, files and
directories they embed or the tip build read from elsewhere, the workspace
manifests, build configuration, compiler wrapper and the model's directory)
and splits the legs into GPU queue jobs of at most 1800 s, one device count
per job.

`run` serves each leg as one AB pair with the runner's smoke mode: the
determinism.py request set (with its image request when the server serves
images), a decode prefix warmup and single-stream decode, plus the phases
`--phases` adds (such as batched decode for a model that serves concurrent
streams reproducibly). A leg passes when base and tip answer bit for bit alike and, with speculation, the
tip's acceptance rate is not lower. Ports are chosen free at run time and
GPUs are the ones the queue exported (CUDA_VISIBLE_DEVICES).
"""
import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
import time
import tomllib

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
# The serving runner beside this file (in the repository and in a pinned
# copy): its profile expansion, smoke phases and install check.
sys.path.insert(0, str(HERE.parent / 'ab/serving'))
from common import Refusal, need  # noqa: E402
import evidence  # noqa: E402
import model as serving_model  # noqa: E402
from run import verify_install  # noqa: E402

# Every smoke runs the runner's smoke phases (single-stream requests);
# --phases adds others.
PHASES = evidence.SMOKE
JOB_TIMEOUT = 1800
JOB_BUDGET = 1740
# Seconds a leg needs beyond its two server starts: the phases, shutdown and
# release.
PHASE_SECONDS = 240
# Build inputs outside the crates' directories that no Cargo metadata names:
# the build configuration and the compiler wrapper the build selects.
BUILD_CONFIGURATION = ['.cargo/', 'rust-toolchain', 'rust-toolchain.toml', 'tools/nvcc_cache/']
INCLUDE = re.compile(r'include(?:_str|_bytes)?!\s*\(\s*(concat!\s*\(\s*env!\s*\(\s*"CARGO_MANIFEST_DIR"\s*\)\s*,\s*)?"([^"]+)"')
RERUN = re.compile(r'^cargo::?rerun-if-changed=(.+)$')


def inventory(models, model):
    with (Path(models) / model / 'model.toml').open('rb') as stream:
        return tomllib.load(stream)


def profiles(models, model):
    value = json.loads((Path(models) / model / 'profiles.json').read_text())
    need(value.get('model') == model and isinstance(value.get('profiles'), dict), f'{model}: no serve profiles')
    return value['profiles']


def shape(profile):
    """(tensor-parallel size, context-parallel size, drafter, draft tokens)."""
    speculation = profile.get('speculation', {})
    return (profile['tensor_parallel'], profile.get('context_parallel_size') or 1,
            speculation.get('drafter', 'none'), speculation.get('tokens', 0))


def default_legs(declared):
    """TP4 autoregressive, TP4 speculative and TP2 autoregressive profiles, a
    qualified one first, each when declared."""
    def pick(tp, speculative):
        found = [name for name, p in declared.items()
                 if shape(p)[:2] == (tp, 1) and (shape(p)[2] != 'none') == speculative]
        found.sort(key=lambda name: declared[name].get('status') != 'qualified')
        return found[:1]
    return pick(4, False) + pick(4, True) + pick(2, False)


def leg(name, profile):
    """The profile served whole (the runner's serve_profile: its options and
    the /health values they imply), at its draft depth in fixed mode."""
    tp, cp, drafter, tokens = shape(profile)
    args, expect = serving_model.serving(name, profile)
    if drafter == 'none':
        variant = dict(ar=dict(mode='off', tokens=0))
    else:
        variant = dict(spec=dict(mode='fixed', tokens=tokens, drafter=drafter))
    return dict(name=name, tp=tp, serve_profile=name, serve_args=args, variants=variant, expect=expect)


def smoke_phases(extra=()):
    """The smoke phases, then the added ones in order."""
    return PHASES + [name for name in extra if name not in PHASES]


def run_file(model, legspec, arms, checkpoint, identity, src, phases=(), commits=None):
    """A smoke run file of tools/ab/serving for one leg: the smoke phases plus
    `phases`, each install bound to its commit when `commits` names them;
    port, gpus and budget are set when the job runs."""
    value = dict(schema='chopinfer.ab-serving.v1', model=model, models=str(src / 'models'),
                 checkpoint=str(checkpoint), checkpoint_identity=identity, smoke=True,
                 installs={arm: dict(path=str(path / 'install'), cache=str(path / 'cache'))
                           for arm, path in arms.items()},
                 port=0, gpus=list(range(legspec['tp'])), serve_profile=legspec['serve_profile'], serve_args=[],
                 variants=legspec['variants'], order=['base', 'new'], phases=smoke_phases(phases),
                 qual_tools=str(src / 'tools/qual'), budget_seconds=JOB_BUDGET)
    for arm, commit in (commits or {}).items():
        value['installs'][arm]['commit'] = commit
    qual = src / 'models' / model / 'qual.json'
    if qual.is_file():
        # determinism.py's image request reads the frontend's thinking switch
        # from the model's gate config.
        value['env'] = dict(CHOPINFER_QUAL_CONFIG=str(qual), CHOPINFER_QUAL_MODEL_ROOT=str(src))
    return value


def pack(legs, start_seconds, job_seconds=JOB_BUDGET):
    """Jobs of legs that share a device count and fit `job_seconds`: each
    leg starts two servers."""
    cost = 2 * start_seconds + PHASE_SECONDS
    need(cost <= job_seconds, f'one leg needs {cost} s, more than a job allows')
    jobs = []
    for tp in sorted({x['tp'] for x in legs}, reverse=True):
        group = [x['name'] for x in legs if x['tp'] == tp]
        per = job_seconds // cost
        jobs += [dict(tp=tp, legs=group[i:i + per]) for i in range(0, len(group), per)]
    return jobs


def closure(metadata, roots):
    """The local packages `roots` (workspace package names) build from: their
    normal and build dependencies, transitively, whose directory is inside
    the workspace root, members or not (a path dependency the workspace
    excludes still compiles in)."""
    packages = {p['id']: p for p in metadata['packages']}
    members = set(metadata['workspace_members'])
    nodes = {n['id']: n for n in metadata['resolve']['nodes']}
    by_name = {packages[i]['name']: i for i in members}
    need(set(roots) <= set(by_name), f'packages not in the workspace: {sorted(set(roots) - set(by_name))}')
    seen, pending = set(), [by_name[name] for name in roots]
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        for dep in nodes[current]['deps']:
            if any(kind.get('kind') in (None, 'build') for kind in dep['dep_kinds']):
                pending.append(dep['pkg'])
    workspace = Path(metadata['workspace_root'])
    return [packages[i] for i in sorted(seen) if Path(packages[i]['manifest_path']).parent.is_relative_to(workspace)]


def embedded(package):
    """Files the package's library, binaries and build script embed with
    include!, include_str! or include_bytes! of a literal path."""
    directory = Path(package['manifest_path']).parent
    found = set()
    for target in package.get('targets', []):
        if set(target['kind']) & {'test', 'bench', 'example'}:
            continue
        source = Path(target['src_path'])
        files = [source] if source.parent == directory else sorted(source.parent.rglob('*.rs'))
        for file in files:
            for manifest_relative, literal in INCLUDE.findall(file.read_text(errors='replace')):
                base = directory if manifest_relative else file.parent
                found.add(Path(os.path.normpath(base / literal.lstrip('/') if manifest_relative else base / literal)))
    return found


def built_inputs(target, metadata):
    """Files a finished build read: every dep-info file's sources and every
    build script's rerun-if-changed paths, relative to its package. A build
    directory names its package, not its version, and one build may compile
    several packages of a name; a relative path is resolved against each of
    their directories (paths() keeps those that exist)."""
    directories = {}
    for package in metadata['packages']:
        directories.setdefault(package['name'], []).append(Path(package['manifest_path']).parent)
    found = set()
    release = Path(target) / 'release'
    for info in release.rglob('*.d'):
        for line in info.read_text(errors='replace').splitlines():
            if ': ' in line:
                found.update(Path(x.replace('\\ ', ' ')) for x in re.split(r'(?<!\\)\s+', line.split(': ', 1)[1]) if x)
    for output in (release / 'build').glob('*/output'):
        packages = directories.get(output.parent.name.rsplit('-', 1)[0], [])
        for line in output.read_text(errors='replace').splitlines():
            match = RERUN.match(line.strip())
            if match:
                found.update(Path(os.path.normpath(package / match.group(1))) for package in packages)
    return found


def paths(metadata, roots, model, inputs=()):
    """Pathspecs whose change can change what `roots` (package names) build:
    the local packages they depend on (closure), the files those embed from
    outside their directories, `inputs` (files and watched directories a
    build read, such as built_inputs) that the workspace holds, the
    workspace manifests and lock, the build
    configuration and compiler wrapper, and the model's directory."""
    workspace = Path(metadata['workspace_root'])
    packages = closure(metadata, roots)
    specs = {str(Path(p['manifest_path']).parent.relative_to(workspace)) + '/' for p in packages}
    specs |= {'Cargo.toml', 'Cargo.lock', f'models/{model}/', *BUILD_CONFIGURATION}
    files = set(inputs)
    for package in packages:
        files |= embedded(package)
    for file in files:
        if file.is_absolute() and file.is_relative_to(workspace):
            file = file.relative_to(workspace)
        # A reused target can hold dep-info of files the tip no longer has.
        # A watched directory covers everything under it.
        if not file.is_absolute() and '..' not in file.parts:
            if (workspace / file).is_file():
                specs.add(str(file))
            elif (workspace / file).is_dir():
                specs.add(str(file) + '/')
    return sorted(spec for spec in specs
                  if not any(other != spec and other.endswith('/') and spec.startswith(other) for other in specs))


def roots(model_inventory, model):
    return ['runtime-host', model_inventory['compiler']['package'], model_inventory['bundle']['package']]


def cargo_args(model_inventory, model):
    return ['build', '-p', 'runtime-host', '-p', model_inventory['compiler']['package'],
            '-p', model_inventory['bundle']['package'], '--release', '--lib', '--bin', 'chopinfer',
            '--bin', f'chopinfer-model-{model}', '--features', 'runtime-host/linux-cuda-host', '--locked', '--offline']


def install_files(model_inventory, model):
    cdylibs = model_inventory['bundle']['cdylib']
    need(len(cdylibs) == 1, f'{model}: expected one bundle cdylib')
    return {'bin/chopinfer': 'chopinfer', f'bin/chopinfer-model-{model}': f'chopinfer-model-{model}',
            f'lib/lib{cdylibs[0]}.so': f'lib{cdylibs[0]}.so'}


def sha256(path):
    with open(path, 'rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def note(text):
    print(datetime.datetime.now(datetime.timezone.utc).strftime('%H:%M:%S'), text, flush=True)


def git_environment():
    """Git resolves history in the selected checkout, not an inherited index or repository."""
    return {key: value for key, value in os.environ.items() if not key.startswith('GIT_')}


def git(*args, repo=REPO):
    return subprocess.run(['git', '-C', str(repo), *args], env=git_environment(),
                          check=True, capture_output=True, text=True).stdout.strip()


def source_record(source, commit):
    """Refuse anything except a clean, detached checkout of this exact tip with history."""
    source = Path(source)
    need(isinstance(commit, str) and re.fullmatch('[0-9a-f]{40}', commit), 'qualification source needs a full tip')
    need(source.is_absolute() and not source.is_symlink(), 'qualification source must be an absolute checkout')
    try:
        need(Path(git('rev-parse', '--show-toplevel', repo=source)).resolve() == source.resolve(),
             'qualification source is not its own Git checkout')
        need(git('rev-parse', 'HEAD', repo=source) == commit, 'qualification source is not the requested tip')
        need(git('rev-parse', '--abbrev-ref', 'HEAD', repo=source) == 'HEAD',
             'qualification source must have detached HEAD')
        need(git('rev-parse', '--is-shallow-repository', repo=source) == 'false',
             'qualification source needs complete history')
        need(not git('-c', 'core.fsmonitor=false', 'status', '--porcelain=v1', '--untracked-files=all', '--ignored',
                     repo=source), 'qualification source is not clean (including ignored files)')
        tree = git('rev-parse', 'HEAD^{tree}', repo=source)
    except subprocess.CalledProcessError as error:
        raise Refusal('qualification source history is unavailable: ' + error.stderr.strip()) from error
    return dict(schema='chopinfer.smoke-qualification-source.v1', path=str(source.resolve()), commit=commit, tree=tree)


def qualification_source(out, commit):
    """Keep qualification separate from READY build exports, which have no Git history."""
    need(isinstance(commit, str) and re.fullmatch('[0-9a-f]{40}', commit), 'qualification source needs a full tip')
    source = Path(out).resolve() / 'qualification' / commit
    if not source.exists():
        source.parent.mkdir(parents=True, exist_ok=True)
        git('worktree', 'add', '--detach', '--lock', '--reason', 'smoke qualification source', str(source), commit)
    return source_record(source, commit)


def bind_qualification(value, source):
    value['models'] = str(source / 'models')
    value['qual_tools'] = str(source / 'tools/qual')
    value['env'] = dict(value.get('env', {}), CHOPINFER_QUAL_MODEL_ROOT=str(source),
                        CHOPINFER_QUAL_CONFIG=str(source / 'models' / value['model'] / 'qual.json'))
    return value


def verify_qualification(pair, value):
    record = pair.get('qualification_source')
    need(isinstance(record, dict) and isinstance(record.get('path'), str),
         'missing qualification source: run smoke.py prepare-qualification DIR before serving')
    source = Path(record['path'])
    need(record == source_record(source, pair['tip']), 'qualification source identity differs from the pinned record')
    need(value['model'] == pair['model'] and value['installs']['new'].get('commit') == pair['tip'],
         'qualification source does not match the new install and model')
    for field, expected in [('models', source / 'models'), ('qual_tools', source / 'tools/qual')]:
        need(value.get(field) == str(expected), f'qualification source differs in {field}')
    expected = dict(CHOPINFER_QUAL_MODEL_ROOT=str(source),
                    CHOPINFER_QUAL_CONFIG=str(source / 'models' / pair['model'] / 'qual.json'))
    need(all(value.get('env', {}).get(key) == path for key, path in expected.items()),
         'qualification source differs in the model root or config')
    for env in [value.get('env', {}), *[arm.get('env', {}) for arm in value['installs'].values()],
                *[variant.get('env', {}) for variant in value['variants'].values()]]:
        need(not any(key.startswith('GIT_') for key in env), 'qualification source cannot override Git history')
        need(all(key not in env or env[key] == path for key, path in expected.items()),
             'qualification source is overridden by an arm or variant')
    return source


def preflight_qualification(pair, value, directory):
    """Exercise the real frontend and lock loaders with the matching new scorer, without serving."""
    source = verify_qualification(pair, value)
    installed = verify_install(value['installs']['new']['path'], serving_model.facts(pair['model'], source / 'models'))
    need(installed['commit'] == pair['tip'], 'qualification scorer is not the requested tip install')
    directory.mkdir(parents=True)
    env = {**git_environment(), **value['env'],
           'CHOPINFER_EVAL': str(Path(installed['path']) / 'bin/chopinfer'),
           'CHOPINFER_QUAL_RECEIPTS': str(directory / 'receipts')}
    script = '''import hashlib, json, sys
sys.path[:0] = [sys.argv[1], sys.argv[2]]
import common, refs
common.startup(True)
config = common.qualification_config()
lock = refs.load_lock(config)
assert isinstance(lock, dict), "qualification reference lock is missing"
print(json.dumps(dict(model=config["model"], objects=len(lock["objects"]),
                     frontend_sha256=hashlib.sha256(common.input_bytes("frontend")).hexdigest())))
'''
    result = subprocess.run([sys.executable, '-I', '-B', '-c', script, str(source / 'tools/qual'), str(source)],
                            cwd=source, env=env, capture_output=True, text=True)
    (directory / 'load.log').write_text(result.stdout + result.stderr)
    (directory / 'exit').write_text(str(result.returncode) + '\n')
    need(result.returncode == 0, f'qualification frontend/lock preflight failed: {directory / "load.log"}')
    report = json.loads(result.stdout)
    need(report.get('model') == pair['model'], 'qualification preflight loaded another model')
    verify_qualification(pair, value)
    (directory / 'result.json').write_text(json.dumps(dict(source=pair['qualification_source'],
        scorer_receipt_sha256=installed['receipt_sha256'], **report), indent=2) + '\n')


def prepare_qualification(out):
    """Migrate an existing pair only after its unchanged installs pass CPU source preflight."""
    out = Path(out).resolve()
    pair = json.loads((out / 'pair.json').read_text())
    pair['qualification_source'] = qualification_source(out, pair['tip'])
    source = Path(pair['qualification_source']['path'])
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    update = out / 'qualification-updates' / stamp
    proposed = update / 'proposed'
    proposed.mkdir(parents=True)
    for name in pair['legs']:
        value = bind_qualification(json.loads((out / 'legs' / f'{name}.json').read_text()), source)
        path = proposed / f'{name}.json'
        path.write_text(json.dumps(value, indent=1) + '\n')
        preflight_qualification(pair, value, update / name)
        dry_run(REPO, path)
    # Keep the old setup beside the preflight. Never change arms/, READY,
    # paths.txt or runs/, including failures captured before this migration.
    shutil.copyfile(out / 'pair.json', update / 'pair.before.json')
    shutil.copytree(out / 'legs', update / 'legs.before')
    if (out / 'runner').exists():
        shutil.copytree(out / 'runner', update / 'runner.before')
    pin_runner(out)
    for name in pair['legs']:
        shutil.copyfile(proposed / f'{name}.json', out / 'legs' / f'{name}.json')
    (out / 'pair.json').write_text(json.dumps(pair, indent=1) + '\n')
    note(f'qualification source pinned and preflighted: {source}')
    return 0


def reusable(arm, record, commit, model, tps, checkpoint):
    """Whether a READY arm is the requested build: READY and the install
    receipt name the requested commit, model, checkpoint and build, the
    install passes the runner's full check, and every TP is prepared."""
    try:
        receipt = verify_install(arm / 'install', serving_model.facts(model, arm / 'src/models'))
    except (Refusal, OSError, ValueError, KeyError) as error:
        note(f'{commit}: READY install refused ({error})')
        return False
    return (record.get('commit') == commit and receipt['commit'] == commit and record.get('model') == model
            and record.get('checkpoint') == str(Path(checkpoint).resolve())
            and record.get('cargo') == cargo_args(inventory(arm / 'src/models', model), model)
            and record.get('install_sha256') == receipt['receipt_sha256'] and set(record.get('tps', [])) >= set(tps))


def build_arm(out, commit, model, tps, checkpoint, jobs):
    """One arm: src, target, install with INSTALL.json, a prepared direct
    package per tensor-parallel size in cache/, READY last. A READY arm is
    kept only when it is the requested build (reusable)."""
    arm = out / 'arms' / commit
    ready = arm / 'READY'
    if ready.exists():
        record = json.loads(ready.read_text())
        if reusable(arm, record, commit, model, tps, checkpoint):
            note(f'{commit}: READY and unchanged, kept')
            return record
        note(f'{commit}: READY arm is not the requested build; rebuilding')
        ready.unlink()
    src, target, install, cache = arm / 'src', arm / 'target', arm / 'install', arm / 'cache'
    if not (src / '.commit').exists() or (src / '.commit').read_text().strip() != commit:
        shutil.rmtree(src, ignore_errors=True)
        src.mkdir(parents=True)
        # tar --touch: git archive stamps files with the commit time, older
        # than a reused target's outputs, so cargo would keep them.
        archive = subprocess.Popen(['git', '-C', str(REPO), 'archive', commit], stdout=subprocess.PIPE)
        extract = subprocess.run(['tar', '-x', '--touch', '-C', str(src)], stdin=archive.stdout)
        archive.stdout.close()
        need(archive.wait() == 0 and extract.returncode == 0, f'git archive {commit} failed')
        (src / '.commit').write_text(commit + '\n')
    model_inventory = inventory(src / 'models', model)
    env = dict(os.environ, CARGO_TARGET_DIR=str(target), CHOPINFER_BUILD_REVISION=commit, CARGO_INCREMENTAL='0',
               CARGO_NET_OFFLINE='true', CARGO_BUILD_JOBS=str(jobs))
    wrapper = src / 'tools/nvcc_cache/nvcc_cache.py'
    if env.get('CHOPINFER_NVCC_CACHE_DIR') and wrapper.exists():
        env['CHOPINFER_NVCC_WRAPPER'] = str(wrapper)
    note(f'{commit}: cargo build')
    started = time.monotonic()
    subprocess.run(['nice', '-n', '10', 'cargo', *cargo_args(model_inventory, model)], cwd=src, env=env, check=True)
    note(f'{commit}: built in {time.monotonic() - started:.0f} s')
    shutil.rmtree(install, ignore_errors=True)
    files = {}
    for relative, built in install_files(model_inventory, model).items():
        (install / relative).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(target / 'release' / built, install / relative)
        os.chmod(install / relative, 0o755)
        files[relative] = sha256(install / relative)
    for directory in (install, install / 'bin', install / 'lib'):
        os.chmod(directory, 0o755)
    (install / 'INSTALL.json').write_text(json.dumps(dict(commit=commit, files=files), indent=1) + '\n')
    os.chmod(install / 'INSTALL.json', 0o444)
    shutil.rmtree(cache, ignore_errors=True)
    identities = set()
    for tp in sorted(set(tps)):
        log = arm / f'prepare-tp{tp}.log'
        note(f'{commit}: prepare TP{tp}')
        with log.open('w') as stream:
            result = subprocess.run([str(install / f'bin/chopinfer-model-{model}'), 'prepare', '--direct', '--model',
                                     str(checkpoint), '--tensor-parallel-size', str(tp), '--cache-dir', str(cache)],
                                    stdout=stream, stderr=subprocess.STDOUT)
        need(result.returncode == 0, f'{commit}: prepare TP{tp} failed; see {log}')
        records = [json.loads(line) for line in log.read_text().splitlines() if line.startswith('{')]
        need(records and records[-1].get('result') == 'prepared-direct', f'{commit}: no direct preparation in {log}')
        identities.add(records[-1]['checkpoint_identity'])
    need(len(identities) == 1, f'{commit}: preparations name different checkpoint identities')
    check = subprocess.run(['cargo', *cargo_args(model_inventory, model), '-v'], cwd=src, env=env,
                           capture_output=True, text=True)
    (arm / 'rebuild-check.log').write_text(check.stdout + check.stderr)
    need(check.returncode == 0 and not any(line.lstrip().startswith(('Compiling ', 'Running '))
                                           for line in (check.stdout + check.stderr).splitlines()),
         f'{commit}: the second build compiled something')
    record = dict(commit=commit, model=model, checkpoint=str(Path(checkpoint).resolve()),
                  cargo=cargo_args(model_inventory, model), tps=sorted(set(tps)),
                  install_sha256=sha256(install / 'INSTALL.json'), checkpoint_identity=identities.pop())
    ready.write_text(json.dumps(record) + '\n')
    note(f'{commit}: READY')
    return record


def free_port():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


def pin_runner(out):
    """The runner and this file, copied into DIR/runner with their layout."""
    runner = out / 'runner'
    shutil.rmtree(runner, ignore_errors=True)
    for source in [*sorted((REPO / 'tools/ab/serving').glob('*.py')), Path(__file__)]:
        if source.name.startswith('test_') or source.name == 'stubs.py':
            continue
        target = runner / source.relative_to(REPO)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    return runner


def dry_run(runner, path):
    value = json.loads(path.read_text())
    value['port'] = free_port()
    probe = path.with_suffix('.dry.json')
    probe.write_text(json.dumps(value, indent=1))
    result = subprocess.run([sys.executable, '-B', str(runner / 'tools/ab/serving/run.py'), str(probe), '--dry-run'],
                            capture_output=True, text=True)
    path.with_suffix('.dry-run.log').write_text(result.stdout + result.stderr)
    need(result.returncode == 0, f'dry run of {path.name} failed: {result.stderr.strip()[-500:]}')


def cargo_metadata(repo):
    """Cargo metadata with the build's features."""
    return json.loads(subprocess.run(['cargo', 'metadata', '--format-version', '1', '--features',
                                      'runtime-host/linux-cuda-host', '--locked', '--offline'],
                                     cwd=repo, check=True, capture_output=True, text=True).stdout)


def build(args):
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    commits = dict(base=git('rev-parse', '--verify', args.base + '^{commit}'),
                   new=git('rev-parse', '--verify', args.tip + '^{commit}'))
    need(commits['base'] != commits['new'], 'BASE and TIP are the same commit')
    qualification = qualification_source(out, commits['new'])
    source = Path(qualification['path'])
    models = source / 'models'
    declared = profiles(models, args.model)
    names = args.legs.split(',') if args.legs else default_legs(declared)
    need(names and set(names) <= set(declared), f'unknown legs: {sorted(set(names) - set(declared))}')
    legs = [leg(name, declared[name]) for name in names]
    model_inventory = inventory(models, args.model)
    checkpoint = args.checkpoint or os.environ.get(f'CHOPINFER_{model_inventory["checkpoint_env"]}_CHECKPOINT')
    need(checkpoint and Path(checkpoint).is_dir(), 'no checkpoint: pass --checkpoint or set the model\'s variable')
    tps = [x['tp'] for x in legs]
    records = {arm: build_arm(out, commit, args.model, tps, checkpoint, args.jobs) for arm, commit in commits.items()}
    need(records['base']['checkpoint_identity'] == records['new']['checkpoint_identity'],
         'the arms prepared different checkpoint identities')
    tip_src = out / 'arms' / commits['new'] / 'src'
    runner = pin_runner(out)
    pair = dict(model=args.model, base=commits['base'], tip=commits['new'], qualification_source=qualification)
    (out / 'legs').mkdir(exist_ok=True)
    for spec in legs:
        value = run_file(args.model, spec, {arm: out / 'arms' / c for arm, c in commits.items()}, checkpoint,
                         records['base']['checkpoint_identity'], source,
                         args.phases.split(',') if args.phases else (), commits)
        path = out / 'legs' / f'{spec["name"]}.json'
        path.write_text(json.dumps(value, indent=1) + '\n')
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
        preflight_qualification(pair, value, out / 'qualification-preflight' / stamp / spec['name'])
        dry_run(runner, path)
        note(f'leg {spec["name"]}: dry run passed')
    metadata = cargo_metadata(tip_src)
    inputs = built_inputs(out / 'arms' / commits['new'] / 'target', metadata)
    (out / 'paths.txt').write_text('# Smoke-staleness pathspecs of the tip (tools/smoke/smoke.py paths)\n' +
                                   ''.join(spec + '\n' for spec in paths(metadata, roots(model_inventory, args.model),
                                                                          args.model, inputs)))
    jobs = pack(legs, args.start_seconds)
    (out / 'jobs').mkdir(exist_ok=True)
    for index, job in enumerate(jobs, 1):
        label = f'{args.job_prefix}-tp{job["tp"]}' + (f'-{index}' if len(jobs) > len({j["tp"] for j in jobs}) else '')
        (out / 'jobs' / f'{label}.job').write_text(
            f'# timeout {JOB_TIMEOUT}\n# gpus {job["tp"]}\n'
            f'# {label}: {args.model} serving smoke, base {commits["base"][:9]} tip {commits["new"][:9]}, '
            f'legs {", ".join(job["legs"])} (tools/smoke)\n'
            f'python3 -B {runner / "tools/smoke/smoke.py"} run {out} {" ".join(job["legs"])}\n')
    (out / 'pair.json').write_text(json.dumps(dict(pair, legs=names, jobs=jobs), indent=1) + '\n')
    note(f'done: {len(jobs)} job file(s) in {out / "jobs"}')
    return 0


def run(args):
    """Serve each named leg as one AB pair; exit 0 only when all pass."""
    out = Path(args.dir).absolute()
    pair = json.loads((out / 'pair.json').read_text())
    runner = out / 'runner/tools/ab/serving/run.py'
    deadline = time.monotonic() + JOB_BUDGET
    visible = [x for x in os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',') if x]
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ') + f'-{os.getpid()}'
    (out / 'runs').mkdir(exist_ok=True)
    verdicts = []
    for index, name in enumerate(args.legs):
        value = json.loads((out / 'legs' / f'{name}.json').read_text())
        verify_qualification(pair, value)
        tp = len(value['gpus'])
        gpus = visible[:tp] if visible else [str(i) for i in range(tp)]
        need(len(gpus) == tp, f'{name} needs {tp} GPUs; the job has {visible}')
        # The remaining time, shared equally by the remaining legs.
        budget = math.floor((deadline - time.monotonic()) / (len(args.legs) - index))
        if budget < 2 * 60 + PHASE_SECONDS:
            verdicts.append((name, 'FAIL', 'no time left in the job'))
            continue
        value.update(port=free_port(), gpus=gpus, budget_seconds=min(budget, JOB_BUDGET))
        label = f'{name}-{stamp}'
        path = out / 'runs' / f'{label}.json'
        path.write_text(json.dumps(value, indent=1) + '\n')
        note(f'{name}: serving base then tip on GPUs {",".join(gpus)} (budget {value["budget_seconds"]} s)')
        result = subprocess.run([sys.executable, '-B', str(runner), str(path), '--out', str(out / 'runs' / label)],
                                timeout=value['budget_seconds'] + 30, env=git_environment())
        (out / 'runs' / f'{label}.exit').write_text(f'{result.returncode}\n')
        verify_qualification(pair, value)
        try:
            report = json.loads((out / 'runs' / label / 'result.json').read_text())
        except (OSError, ValueError):
            report = dict(passed=False, errors=[f'runner exit {result.returncode} without a result'])
        passed = result.returncode == 0 and report.get('passed') is True
        verdicts.append((name, 'PASS' if passed else 'FAIL', '; '.join(report.get('errors') or [])))
    summary = ''.join(f'{verdict} {name}{": " + why if why else ""}\n' for name, verdict, why in verdicts)
    (out / 'runs' / f'{stamp}.summary').write_text(summary)
    print(summary, end='')
    return 0 if all(verdict == 'PASS' for _, verdict, _ in verdicts) else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)
    command = commands.add_parser('legs')
    command.add_argument('model')
    command = commands.add_parser('paths')
    command.add_argument('model')
    command.add_argument('--repo', type=Path, default=REPO)
    command.add_argument('--target', type=Path, help='a build of --repo whose dep-info adds the files it read')
    command = commands.add_parser('build')
    command.add_argument('model')
    command.add_argument('base')
    command.add_argument('tip')
    command.add_argument('--out', required=True)
    command.add_argument('--legs')
    command.add_argument('--checkpoint')
    command.add_argument('--phases', help='serving runner phases to add to ' + ','.join(PHASES))
    command.add_argument('--job-prefix', default='smoke')
    command.add_argument('--start-seconds', type=int, default=120,
                         help='seconds one server start may take (sizes the jobs)')
    command.add_argument('--jobs', type=int, default=12, help='cargo build jobs')
    command = commands.add_parser('run')
    command.add_argument('dir')
    command.add_argument('legs', nargs='+')
    command = commands.add_parser('prepare-qualification')
    command.add_argument('dir', type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == 'legs':
            declared = profiles(REPO / 'models', args.model)
            for name, profile in declared.items():
                print(name, *shape(profile), profile.get('status'))
            print('default:', ','.join(default_legs(declared)))
            return 0
        if args.command == 'paths':
            metadata = cargo_metadata(args.repo)
            inputs = built_inputs(args.target, metadata) if args.target else ()
            print('\n'.join(paths(metadata, roots(inventory(args.repo / 'models', args.model), args.model),
                                   args.model, inputs)))
            return 0
        if args.command == 'prepare-qualification':
            return prepare_qualification(args.dir)
        return build(args) if args.command == 'build' else run(args)
    except (Refusal, subprocess.CalledProcessError, OSError, KeyError, ValueError) as error:
        print(f'REFUSED: {error}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
