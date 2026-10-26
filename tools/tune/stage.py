#!/usr/bin/env python3
"""Seal a depth tuner stage.

Usage: python3 -I tools/tune/stage.py --model models/<model>/depth-tune.json
           --install-dir ROUND_DIR [--install-name NAME ...] [--job NAME] STAGE_DIR
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
arm's server borrows its weights. Repeat --install-name to seal multiple
builds (default: head); each arm's optional install selects one by name.
Multiple builds require matching proven rank content/layouts and non-speculation
options. Each build needs its own CPU content proof, supplied with
--content-proof NAME=FILE or logs/weight-content-NAME.json in ROUND_DIR.

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
import re
import shutil
import stat
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


def protect(stage):
    """The queue authenticates the keeper spec only when its owned inputs
    cannot be changed by other writers. Preserve executable file modes."""
    for path in [stage, *stage.rglob('*')]:
        if path.is_symlink():
            raise RuntimeError(f'stage contains a symlink: {path}')
        path.chmod(path.stat().st_mode & ~0o022)


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


def cold_budget(config):
    import math
    import tune
    model, plan = config['model'], config['model']['planning']
    budgets = {}
    for c in model['concurrencies']:
        rate = max(plan['tokens_per_second'][str(c)], plan['ar_tokens_per_second'][str(c)])
        tokens = config['tokens'].get(str(c), config['tokens']['default'])
        budgets[str(c)] = 2 * c + 2 * math.ceil(15 * rate / tokens)
    visits = tune.visits(config)
    needs = {visit['label']: sum(config['warm']) + sum(budgets[str(c)] for _, part in visit['passes'] for _, c in part)
             for visit in visits}
    arms = builder.arm_visits(visits)
    per_arm = max((needs[label] - 1) * len(members) + index + 1
                  for members in arms.values() for index, label in enumerate(members) if needs[label])
    return {'request_budgets': budgets, 'prompt_needs': needs, 'prompts_required': len(arms) * per_arm}


def estimate(config):
    """Plan priming, measurement, drain, cache warm-up and unmatched replay."""
    import tune
    model, plan = config['model'], config['model']['planning']
    tokens = lambda c: config['tokens'].get(str(c), config['tokens']['default'])
    rate_for = lambda arm: plan['ar_tokens_per_second'] if arm['depth'] == 0 else plan['tokens_per_second']
    startup = plan['startup_seconds'] + plan['process_overhead_seconds']

    def closed(c, rate):
        cohort = c * tokens(c) / rate
        return max(4, cohort) + 15 + cohort + plan['settle_seconds']

    total = 0.0
    for index, arm in enumerate(tune.visits(config)):
        rates = rate_for(arm)
        total += startup + sum(c * config['warm_tokens'] / rates[str(c)] + plan['settle_seconds'] for c in config['warm'])
        if model.get('mode') == 'ab':
            total += sum(closed(c, rates[str(c)]) for _, part in arm['passes'] for _, c in part)
            continue
        windows = (tune.windows_for(config, arm['depth'], True) if index == 0 else []) + [w for _, part in arm['passes'] for w in part]
        for _, c in windows:
            total += model.get('groups', {}).get(str(c), 1) * (c * tokens(c) / rates[str(c)] + plan['settle_seconds'])
    if model.get('mode') == 'ab':
        arms = {arm['label']: arm for arm in tune.arms(config)}
        if not config.get('cold_engram'):
            rates = rate_for(tune.arms(config)[0])
            total += startup + sum(closed(c, rates[str(c)]) for c, _ in tune.warmup_windows(config))
            total += sum(c * config['warm_tokens'] / rates[str(c)] + plan['settle_seconds'] for c in config['warm'])
        for baseline in dict.fromkeys(a for a, _ in model['pairs']):
            total += startup
            for c in model['concurrencies']:
                rates = rate_for(arms[baseline])
                repeats = model.get('concurrency_repeats', {}).get(str(c), model['repeats'])
                if config.get('cold_engram'):
                    requests = cold_budget(config)['request_budgets'][str(c)]
                    total += len(arms) * repeats * requests * tokens(c) / rates[str(c)]
                else:
                    # Matched requests compare directly. Budget a cohort plus
                    # 25% of measured output per pair for unmatched tails.
                    for a, b in model['pairs']:
                        if a == baseline:
                            throughput = max(rates[str(c)], rate_for(arms[b])[str(c)])
                            total += repeats * (c * tokens(c) + .25 * 15 * throughput) / rates[str(c)]
    return total


def check_budget(config, prompts):
    seconds = estimate(config)
    if seconds > config['work_seconds']:
        raise ValueError(f'estimated {seconds:.1f} seconds exceeds work budget {config["work_seconds"]}')
    if config.get('cold_engram'):
        import tune
        needed = cold_budget(config)
        sizes = builder.visit_pool_sizes(prompts, tune.visits(config))
        if any(sizes[label] < count for label, count in needed['prompt_needs'].items()):
            raise ValueError(f'balanced cold prompt pool has {prompts}, needs {needed["prompts_required"]}')
        needed['balanced_pool_sizes'] = sizes
        config['cold_engram'].update(needed)
    return seconds



def content_proof(path, binary, recipe, ranks):
    """Read a protected proof and bind it to this executable's fresh CPU plan."""
    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError('duplicate content proof key: ' + key)
            value[key] = item
        return value

    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except OSError as error:
        raise ValueError(f'cannot read weight content proof: {path}') from error
    with os.fdopen(fd, 'rb') as stream:
        before = os.fstat(stream.fileno())
        if (not stat.S_ISREG(before.st_mode) or before.st_uid != os.getuid()
                or before.st_mode & 0o022 or before.st_size > 1 << 20):
            raise ValueError('content proof must be a bounded owned regular file protected from other writers')
        body = stream.read((1 << 20) + 1)
        after = os.fstat(stream.fileno())
        stamps = lambda s: (s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
        if len(body) > 1 << 20 or stamps(before) != stamps(after):
            raise ValueError('content proof changed while being read')
    proof = json.loads(body, object_pairs_hook=unique)
    if (set(proof) != {'schema', 'producer_sha256', 'checkpoint', 'images'}
            or proof['schema'] != 'chopinfer.weight-content.v1'
            or proof['producer_sha256'] != digest(binary)):
        raise ValueError('content proof belongs to another executable or schema')
    images, seen = [], set()
    for image in proof['images']:
        key = (image.get('image_role'), image.get('rank'))
        identity = image.get('identity', {})
        if (set(image) != {'image_role', 'rank', 'byte_length', 'recipe_identity', 'identity'}
                or not isinstance(key[0], str) or not key[0] or type(key[1]) is not int
                or not 0 <= key[1] < ranks or key in seen
                or type(image['byte_length']) is not int or image['byte_length'] <= 0
                or set(identity) != {'content_sha256', 'layout_sha256'}
                or any(not isinstance(v, str) or not re.fullmatch('[0-9a-f]{64}', v)
                       for v in [image['recipe_identity'], *identity.values()])):
            raise ValueError('invalid rank content proof')
        seen.add(key)
        images.append({**image, 'identity': {**identity, 'content_sha256': ''}})
    order = lambda image: (image['image_role'], image['rank'])
    if (not seen or any({rank for name, rank in seen if name == role} != set(range(ranks))
                        for role, _ in seen)
            or proof['checkpoint'] != recipe['checkpoint']
            or sorted(images, key=order) != sorted(recipe['images'], key=order)):
        raise ValueError('keeper identity differs: content proof does not match fresh checkpoint, recipe or layout')
    proof['images'].sort(key=order)
    return body, proof


def content_key(proof, devices):
    """The lane staging key: checkpoint, rank content/layout/size and devices.
    Producer executables and recipes remain bound by each build's own proof."""
    value = {'checkpoint': proof['checkpoint'], 'devices': devices,
             'images': [{k: image[k] for k in ('image_role', 'rank', 'byte_length', 'identity')}
                        for image in proof['images']]}
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def keeper_compatibility(config, seal, stage, checkpoint, proof_paths):
    """Validate each build's proof against fresh recipes, then compare content."""
    import runner
    import tune
    speculation = {'--speculation', '--speculation-depths', '--speculation-widths',
                   '--speculation-graph-rows', '--speculation-confidence', '--speculation-costs',
                   '--speculative-rows', '--speculative-tokens'}
    reserved = {'--model', '--checkpoint', '--bundle', '--model-cache-dir',
                '--weight-keeper', '--weight-keeper-policy', '--weight-keeper-identity'}
    common, compatible, files = None, None, set()
    directory = stage / 'keeper-identities'
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir()
    for index, arm in enumerate(tune.arms(config)):
        name, install = tune.arm_install(seal, arm)
        arguments = tune.arm_arguments(config, arm)
        options = [*runner.SERVE, *arguments]
        if any(word.split('=', 1)[0] in reserved for word in options):
            raise ValueError('multi-install keeper options cannot override the checkpoint, cache or keeper')
        shared, iterator = [], iter(options)
        for word in iterator:
            if word in speculation:
                if next(iterator, None) is None:
                    raise ValueError(f'missing speculation value: {word}')
            else:
                shared.append(word)
        if common is not None and shared != common:
            raise ValueError('multi-install keeper requires identical serve options apart from speculation')
        common = shared
        path = directory / f'arm-{index}.json'
        binary = Path(install['install']) / 'bin/chopinfer'
        serve = runner.serve_options(install['cache'], arguments, checkpoint)
        argv = [str(binary), 'weights', 'identity', '--output', str(path), *serve]
        # The native command plans on CPU. Hiding devices also protects against
        # an accidental device-dependent implementation in either revision.
        runner.command(argv, path.with_suffix('.log'), 180,
                       env={**runner.server_environment(), 'CUDA_VISIBLE_DEVICES': '', 'PYTHONDONTWRITEBYTECODE': '1'})
        value = json.loads(path.read_text())
        if (value.get('schema') != 'chopinfer.weight-recipes.v1' or value.get('content_verified') is not False
                or not value.get('checkpoint', {}).get('stamps') or not value.get('images')):
            raise ValueError(f'{name}: missing keeper recipe identity')
        body, proof = content_proof(proof_paths[name], binary, value, runner.GPUS)
        devices = serve[serve.index('--devices') + 1] if '--devices' in serve else None
        key = content_key(proof, devices)
        if compatible is not None and key != compatible:
            raise ValueError(f'{name}: keeper content_key differs')
        compatible = key
        target = directory / f'content-{name}.json'
        if target.exists() and target.read_bytes() != body:
            raise ValueError(f'{name}: content proof changed between arms')
        target.write_bytes(body)
        target.chmod(0o600)
        install.update(weight_keeper_identity=str(target), content_key=key)
        files.add(target)
        files.add(path)
    return files


def keeper_spec(config, install, stage, seal, proof_paths):
    """The weight keeper SPEC for runner6: the install and the first arm's
    serve options, which every arm shares but for its speculation options."""
    sys.path.insert(0, str(HERE))
    import runner
    import tune
    model = config['model']
    runner.MODEL, runner.GPUS = model['served_model_name'], model['tensor_parallel_size']
    runner.KV_PAGES, runner.SERVE = model['kv_pages'], model['serve_arguments']
    arm = tune.arms(config)[0]
    checkpoint = os.environ.get(model['checkpoint_env'])
    if not checkpoint:
        raise SystemExit(f"set {model['checkpoint_env']} (source local.env) to stage the weight keeper")
    runner.CHECKPOINT_ENV = model['checkpoint_env']
    files = keeper_compatibility(config, seal, stage, checkpoint, proof_paths) if proof_paths else set()
    arguments = tune.arm_arguments(config, arm, install)
    spec = {'install': install['install'], 'argv': runner.serve_options(install['cache'], arguments, checkpoint)}
    if 'weight_keeper_identity' in install:
        spec['identity_sha256'] = digest(install['weight_keeper_identity'])
    path = stage / 'keeper-spec.json'
    path.write_text(json.dumps(spec, indent=1) + '\n')
    return path, files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage')
    ap.add_argument('--model', required=True, help='models/<model>/depth-tune.json')
    ap.add_argument('--install-dir', help='the integration round directory holding the frozen install')
    ap.add_argument('--install-name', action='append', help='repeat for multiple frozen installs (default: head)')
    ap.add_argument('--content-proof', action='append', default=[], metavar='NAME=FILE',
                    help='use a staged content proof (default for multiple installs: logs/weight-content-NAME.json)')
    ap.add_argument('--template', action='store_true', help='seal without an install, for dry runs only')
    ap.add_argument('--name', help='the job name (default: the model and install revision)')
    ap.add_argument('--job', help="a job of the model config's `jobs`, applied over its settings")
    args = ap.parse_args()
    stage = Path(args.stage).resolve()
    if stage == ROOT or ROOT in stage.parents:
        raise SystemExit('stage outside the repository, under $CHOPINFER_DATA')
    if bool(args.template) == bool(args.install_dir):
        raise SystemExit('give exactly one of --install-dir and --template')
    names = args.install_name or ['head']
    if len(names) != len(set(names)) or any(not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', n) for n in names):
        raise SystemExit('install names must be unique simple names')
    if args.template and names != ['head']:
        raise SystemExit('template stages use only the head install; use frozen fixtures for two-install dry runs')
    proof_paths = {}
    if len(names) > 1:
        proof_paths = {name: Path(args.install_dir).resolve() / 'logs' / f'weight-content-{name}.json' for name in names}
    selected = set()
    for item in args.content_proof:
        name, separator, path = item.partition('=')
        if not separator or not path or name not in proof_paths or name in selected:
            raise SystemExit('content-proof needs one NAME=FILE per selected multi-install build')
        selected.add(name)
        proof_paths[name] = Path(os.path.abspath(path))
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
        installs, files = {}, set()
        for name in names:
            installs[name], sealed = frozen(Path(args.install_dir).resolve(), name)
            files |= sealed
        install = installs[names[0]]
    source = Path(install['source'])
    stage.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(source / 'tools/qual'))
    import bench
    built = builder.build(workload, source, lambda i: bench.stream_prompt('diverse', i).strip())
    name = args.name or f"depth tune {model['model']}{' ' + args.job if args.job else ''} {install['revision'][:10]}"
    keys = ('tokens', 'warm', 'warm_tokens', 'kinds', 'sampling', 'selection', 'work_seconds',
            'queue_timeout_seconds')
    config = {'name': name, 'model': model, 'workload': workload['name'], **{k: model[k] for k in keys}}
    import tune
    selection = {'config': config, 'install': install}
    if len(names) > 1:
        selection.update(installs=installs, default_install=names[0])
    tune.check_installs(selection)
    if len(names) > 1 and not model.get('keeper', True):
        raise ValueError('multi-install jobs require keeper staging')
    if 'allow_fallbacks' in model:
        config['allow_fallbacks'] = model['allow_fallbacks']
    cold_files = []
    cold = model.get('cold_engram', workload.get('cold_engram'))
    if cold:
        if model.get('mode') != 'ab':
            raise SystemExit('cold_engram requires mode ab')
        config['cold_engram'] = cold
        for image in cold.get('images', []):
            path = Path(image['path'])
            import faults
            identity = faults.file_identity(path)
            if identity['size'] != image['bytes']:
                raise ValueError('cold image size differs from its declaration')
            cold_files.append(identity)
            image['identity'] = identity
            files.discard(path)
        if not args.template and not cold.get('images'):
            raise SystemExit('cold_engram needs explicit rank image sources and Engram view ranges')
    if model.get('mode') == 'ab' and model.get('speculation_mode') != 'off' and not model.get('keeper', True):
        raise SystemExit('A/B speculative visits require keeper staging')
    seconds = check_budget(config, len(built['prompts']))
    copy = stage / 'harness'
    if copy.exists():
        shutil.rmtree(copy)
    copy.mkdir()
    for item in HARNESS:
        shutil.copy2(HERE / item, copy / item)
    files |= {copy / item for item in HARNESS}
    files |= {x for x in (source / 'tools/qual').rglob('*') if x.is_file() and '__pycache__' not in x.parts}
    _, first = tune.arm_install(selection, tune.arms(config)[0])
    keeper = None
    if not args.template and model.get('keeper', True):
        keeper, proofs = keeper_spec(config, first, stage, selection, proof_paths)
        files |= proofs | {keeper}
    seal = {'schema': 'chopinfer-depth-tune-seal.v1', 'ready': not args.template, 'config': config, 'install': install,
            'harness_source': str(source), 'prompts': built['prompts'], 'prompt_sources': built['sources'],
            'cold_files': cold_files, 'sha256': {str(x): digest(x) for x in sorted(files)}}
    if len(names) > 1:
        seal.update(installs=installs, default_install=names[0])
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
    receipt = {'payload_sha256': payload, 'revision': install['revision'], 'ready': seal['ready'], 'job': str(job),
               'keeper_spec_sha256': digest(keeper) if keeper else None,
               'sha256': digest(job), 'estimated_minutes': round(seconds / 60, 1),
               'work_budget_minutes': round(config['work_seconds'] / 60, 1), 'prompts': len(built['prompts'])}
    (stage / 'staged.json').write_text(json.dumps(receipt, indent=2) + '\n')
    protect(stage)
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
