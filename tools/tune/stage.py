#!/usr/bin/env python3
"""Seal a depth tuner stage.

Usage: python3 -I tools/tune/stage.py --model models/<model>/depth-tune.json
           --install-dir ROUND_DIR [--install-name head] [--job NAME] STAGE_DIR
       python3 -I tools/tune/stage.py --model ... --template STAGE_DIR

ROUND_DIR holds a frozen install in the integration layout: install-NAME
(bin/ and lib/), cache-NAME with the prepared package, source-NAME (the
revision's source, with `.commit`) and logs/ (version-NAME.txt,
install-NAME.sha256, prepare-NAME.json). The stage builds the workload's
prompts from source-NAME, copies the harness into STAGE_DIR/harness, hashes
the harness, prompts, install binaries, prepared package and source-NAME's
tools/qual (the job imports spec_metrics from it), and writes
runtime-seal.json, the queue wrapper tune.job (with the seal's digest),
keeper-spec.json and staged.json (with an estimate of the job's minutes).
The wrapper's `# keeper use keeper-spec.json SHA256` header asks runner6 for
a weight keeper started with the first arm's serve options, from which every
arm's server borrows its weights.

--template seals without an install, prompts and tools/qual from this tree,
for CPU dry runs only: the job refuses it outside --dry-run.
--job NAME applies the model config's `jobs[NAME]` over its settings (for
example a low- and a high-concurrency half of the sweep, each in its own
budget).
STAGE_DIR must be outside the repository (stage under $CHOPINFER_DATA).
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
HARNESS = ('tune.py', 'runner.py', 'report.py', 'faults.py', 'stub.py', 'prompts.py', 'preflight.py', 'recorded.py',
           'tune.job.in')
sys.path.insert(0, str(HERE))
import prompts as builder  # noqa: E402


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 22), b''):
            h.update(block)
    return h.hexdigest()


def frozen(directory, name):
    """The install in `directory` under `name`, checked as built, and every
    file to seal."""
    root, cache, source, logs = (directory / f'install-{name}', directory / f'cache-{name}',
                                 directory / f'source-{name}', directory / 'logs')
    revision = (source / '.commit').read_text().strip()
    if not (logs / f'version-{name}.txt').read_text().strip().endswith(f'({revision})'):
        raise RuntimeError(f'{name}: the install does not report revision {revision}')
    for line in (logs / f'install-{name}.sha256').read_text().splitlines():
        want, path = line.split(None, 1)
        if digest(path) != want:
            raise RuntimeError(f'{name}: {path} changed after its build')
    prepared = json.loads((logs / f'prepare-{name}.json').read_text())
    if prepared['result'] != 'prepared-direct' or Path(prepared['bundle']).parent != cache:
        raise RuntimeError(f'{name}: unexpected preparation {prepared}')
    files = {logs / f'version-{name}.txt', logs / f'prepare-{name}.json', logs / f'install-{name}.sha256'}
    for tree in (root, Path(prepared['bundle'])):
        for path in [tree, *tree.rglob('*')]:
            if path.is_symlink() or path.stat().st_mode & 0o022:
                raise RuntimeError('unsafe frozen path: ' + str(path))
            if path.is_file():
                files.add(path)
    return {'install': str(root), 'cache': str(cache), 'source': str(source), 'revision': revision,
            'prepared': prepared}, files


def estimate(config):
    """Expected job seconds from the model config's planning figures: per arm
    a startup and fixed overhead, then every warm-up, reference (first arm)
    and timed window request at the planned request tok/s plus its settling
    time."""
    sys.path.insert(0, str(HERE))
    import tune
    model, plan = config['model'], config['model']['planning']
    tokens = lambda c: config['tokens'].get(str(c), config['tokens']['default'])
    total = 0.0
    for index, arm in enumerate(tune.visits(config)):
        depth = arm['depth']
        rates = plan['ar_tokens_per_second'] if depth == 0 else plan['tokens_per_second']
        total += plan['startup_seconds'] + plan['process_overhead_seconds']
        total += sum(c * config['warm_tokens'] / rates[str(c)] + plan['settle_seconds'] for c in config['warm'])
        windows = (tune.windows_for(config, depth, True) if index == 0 else []) \
            + [window for _, part in arm['passes'] for window in part]
        for _, c in windows:
            groups = model.get('groups', {}).get(str(c), 1)
            total += groups * (c * tokens(c) / rates[str(c)] + plan['settle_seconds'])
    return total


def keeper_spec(config, install, stage):
    """The weight keeper SPEC for runner6: the install and the first arm's
    serve options, which every arm shares but for its speculation options."""
    sys.path.insert(0, str(HERE))
    import runner
    import tune
    model = config['model']
    runner.MODEL, runner.GPUS = model['served_model_name'], model['tensor_parallel_size']
    runner.KV_PAGES, runner.SERVE = model['kv_pages'], model['serve_arguments']
    arm = tune.arms(config)[0]
    arguments = ['--speculation', model['speculation_mode'], '--speculation-depths', f"1:{arm['depth']}",
                 *model.get('extra_arguments', []), *arm['arguments']]
    checkpoint = os.environ.get(model['checkpoint_env'])
    if not checkpoint:
        raise SystemExit(f"set {model['checkpoint_env']} (source local.env) to stage the weight keeper")
    spec = {'install': install['install'], 'argv': runner.serve_options(install['cache'], arguments, checkpoint)}
    path = stage / 'keeper-spec.json'
    path.write_text(json.dumps(spec, indent=1) + '\n')
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage')
    ap.add_argument('--model', required=True, help='models/<model>/depth-tune.json')
    ap.add_argument('--install-dir', help='the integration round directory holding the frozen install')
    ap.add_argument('--install-name', default='head')
    ap.add_argument('--template', action='store_true', help='seal without an install, for dry runs only')
    ap.add_argument('--name', help='the job name (default: the model and install revision)')
    ap.add_argument('--job', help="a job of the model config's `jobs`, applied over its settings")
    args = ap.parse_args()
    stage = Path(args.stage).resolve()
    if stage == ROOT or ROOT in stage.parents:
        raise SystemExit('stage outside the repository, under $CHOPINFER_DATA')
    if bool(args.template) == bool(args.install_dir):
        raise SystemExit('give exactly one of --install-dir and --template')
    model = json.loads(Path(args.model).read_text())
    jobs = model.pop('jobs', {})
    if args.job:
        model = {**model, **jobs[args.job], 'job': args.job}
    workload = json.loads((ROOT / model['workload']).read_text())
    for c in model['concurrencies']:
        if c * model.get('groups', {}).get(str(c), 1) > workload['prompts']:
            raise SystemExit(f"C{c}: its prompt groups need more than the workload's {workload['prompts']} prompts")
    if args.template:
        install, files = {'install': str(stage / 'install'), 'cache': str(stage / 'cache'), 'source': str(ROOT),
                          'revision': 'template', 'prepared': {'bundle': ''}}, set()
    else:
        install, files = frozen(Path(args.install_dir).resolve(), args.install_name)
    source = Path(install['source'])
    stage.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(source / 'tools/qual'))
    import bench
    built = builder.build(workload, source, lambda i: bench.stream_prompt('diverse', i).strip())
    name = args.name or f"depth tune {model['model']}{' ' + args.job if args.job else ''} {install['revision'][:10]}"
    keys = ('tokens', 'warm', 'warm_tokens', 'kinds', 'sampling', 'selection', 'expected_power_w', 'work_seconds',
            'queue_timeout_seconds')
    config = {'name': name, 'model': model, 'workload': workload['name'], **{k: model[k] for k in keys}}
    copy = stage / 'harness'
    if copy.exists():
        shutil.rmtree(copy)
    copy.mkdir()
    for item in HARNESS:
        shutil.copy2(HERE / item, copy / item)
    files |= {copy / item for item in HARNESS}
    files |= {x for x in (source / 'tools/qual').rglob('*') if x.is_file() and '__pycache__' not in x.parts}
    keeper = None if args.template or not model.get('keeper', True) else keeper_spec(config, install, stage)
    if keeper:
        files.add(keeper)
    seal = {'schema': 'chopinfer-depth-tune-seal.v1', 'ready': not args.template, 'config': config, 'install': install,
            'harness_source': str(source), 'prompts': built['prompts'], 'prompt_sources': built['sources'],
            'sha256': {str(x): digest(x) for x in sorted(files)}}
    (stage / 'runtime-seal.json').write_text(json.dumps(seal, indent=2, sort_keys=True) + '\n')
    payload = digest(stage / 'runtime-seal.json')
    text = (copy / 'tune.job.in').read_text()
    keeper_line = f'# keeper use {keeper} {digest(keeper)}\n' if keeper else ''
    for key, value in (('@KEEPER@\n', keeper_line), ('@STAGE@', str(stage)), ('@PAYLOAD_SHA@', payload),
                       ('@NAME@', name),
                       ('@CHECKPOINT_ENV@', model['checkpoint_env']),
                       ('@DEVICES@', ','.join(map(str, range(model['tensor_parallel_size'])))),
                       ('@QUEUE_TIMEOUT@', str(config['queue_timeout_seconds'])),
                       ('@WORK_TIMEOUT@', str(config['work_seconds'] + 100)),
                       ('@WORK_SECONDS@', str(config['work_seconds']))):
        text = text.replace(key, value)
    job = stage / 'tune.job'
    job.write_text(text)
    job.chmod(0o755)
    seconds = estimate(config)
    receipt = {'payload_sha256': payload, 'revision': install['revision'], 'ready': seal['ready'], 'job': str(job),
               'keeper_spec_sha256': digest(keeper) if keeper else None,
               'sha256': digest(job), 'estimated_minutes': round(seconds / 60, 1),
               'work_budget_minutes': round(config['work_seconds'] / 60, 1), 'prompts': len(built['prompts'])}
    (stage / 'staged.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
