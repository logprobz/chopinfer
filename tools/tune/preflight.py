#!/usr/bin/env python3
"""Parse every arm with its frozen executable, without CUDA. Depth targets
retain serve --check settings; other jobs keep --list-model-options.

Usage: python3 -I STAGE/harness/preflight.py STAGE (after stage.py sealed it).
Writes STAGE/validation/argv-<ns>/preflight.json.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import runner as r  # noqa: E402
import tune  # noqa: E402
import depth_identity  # noqa: E402


def resolve(argv, environment, root):
    root.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(argv, env=dict(environment, CUDA_VISIBLE_DEVICES=''), stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True, timeout=180)
    (root / 'stdout.json').write_text(result.stdout)
    (root / 'stderr.log').write_text(result.stderr)
    if result.returncode != 0:
        raise ValueError(f"{root.name}: exit {result.returncode}: {result.stderr[-2000:]}")
    return json.loads(result.stdout)


def depth_checks(seal, root):
    """Resolve a sealed profile target and each arm with the frozen executable."""
    model = seal['config']['model']
    target = model['adoption_target']
    if (not isinstance(target, dict) or set(target) - {'profile', 'arguments', 'env'}
            or not isinstance(target.get('profile'), str) or not target['profile']
            or not isinstance(target.get('arguments', []), list)
            or any(not isinstance(word, str) for word in target.get('arguments', []))
            or any(word.split('=', 1)[0] == '--speculation-depths' for word in target.get('arguments', []))):
        raise ValueError('adoption target needs a profile and non-depth overrides')
    first = tune.arms(seal['config'])[0]
    _, install = tune.arm_install(seal, first)
    import os
    target_argv = [str(Path(install['install']) / 'bin/chopinfer'), 'serve',
            '--bundle', install['prepared']['bundle'], '--checkpoint', os.environ[model['checkpoint_env']],
            '--profile', target['profile'], *target.get('arguments', []), '--check']
    environment = tune.arm_environment(seal['config'], {'env': target.get('env', {})})
    target_check = resolve(target_argv, r.apply_environment(r.server_environment(), environment), root / 'target')
    depth_identity.normalized(target_check)
    rows = []
    for arm in tune.arms(seal['config']):
        _, install = tune.arm_install(seal, arm)
        server = r.Server(root / arm['label'], install['install'], install['cache'],
                          tune.arm_arguments(seal['config'], arm, install),
                          environment=tune.arm_environment(seal['config'], arm),
                          bundle=tune.arm_bundle(seal['config'], install))
        argv = [*server.server_command, '--check']
        check = resolve(argv, server.env, root / arm['label'])
        depth_identity.normalized(check)
        rows.append({'arm': arm['label'], 'depth': arm['depth'], 'argv': argv, 'resolved': check,
                     'env': arm.get('env', {}), **tune.arm_identity(seal, arm)})
    out = {'capture_sha256': depth_identity.digest(seal['capture_identity']), 'cpu_only': True,
           'target_declaration': target, 'target': {'argv': target_argv, 'resolved': target_check},
           'cases': rows}
    (root / 'preflight.json').write_text(json.dumps(out, indent=2) + '\n')
    return out


def main(stage):
    stage = Path(stage).resolve()
    seal = json.loads((stage / 'runtime-seal.json').read_text())
    if not seal['ready']:
        raise SystemExit('a template stage has no executable to parse with')
    for path, expected in seal['sha256'].items():
        if Path(path).is_symlink() or tune.digest(path) != expected:
            raise ValueError('sealed file changed before preflight: ' + path)
    if seal.get('capture_identity') != depth_identity.sealed_capture(seal):
        raise ValueError('capture identity differs from the runtime seal')
    model = seal['config']['model']
    tune.check_kv_config(seal['config'])
    tune.check_installs(seal, require_proofs=True)
    r.MODEL, r.CHECKPOINT_ENV, r.GPUS = model['served_model_name'], model['checkpoint_env'], model['tensor_parallel_size']
    r.KV_PAGES, r.KV_LINE, r.SERVE = model['kv_pages'], model['kv_pages_line'], model['serve_arguments']
    root = stage / 'validation' / f'argv-{time.time_ns()}'
    root.mkdir(parents=True)
    if model.get('adoption_target'):
        rows = depth_checks(seal, root)['cases']
        print(f'PASS: {len(rows)} frozen arm settings and adoption target, no CUDA: {root}')
    else:
        rows = []
        for arm in tune.arms(seal['config']):
            depth = arm['depth']
            _, install = tune.arm_install(seal, arm)
            arguments = tune.arm_arguments(seal['config'], arm, install)
            server = r.Server(root / arm['label'], install['install'], install['cache'], arguments,
                              environment=tune.arm_environment(seal['config'], arm), ar=model['speculation_mode'] == 'off',
                              bundle=tune.arm_bundle(seal['config'], install))
            argv = server.server_command + ['--list-model-options']
            if any('kv-memory-fraction' in x for x in argv):
                raise SystemExit('the tuner serves the test KV mode, never a KV memory fraction')
            result = subprocess.run(argv, env=dict(server.env, CUDA_VISIBLE_DEVICES=''), stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE, text=True, timeout=180)
            (root / arm['label'] / 'stdout.json').write_text(result.stdout)
            (root / arm['label'] / 'stderr.log').write_text(result.stderr)
            if result.returncode != 0:
                raise SystemExit(f"{arm['label']}: exit {result.returncode}: {result.stderr[-2000:]}")
            json.loads(result.stdout)
            rows.append({'arm': arm['label'], 'depth': depth, 'argv': argv, 'exit': 0, **tune.arm_identity(seal, arm),
                         **({'env': arm['env']} if arm.get('env') else {})})
        (root / 'preflight.json').write_text(json.dumps({'passed': True, 'cpu_only': True, 'cases': rows}, indent=2) + '\n')
        print(f'PASS: {len(rows)} frozen server parsers, no CUDA: {root}')
    keeper = stage / 'keeper-spec.json'
    if keeper.exists():
        # The keeper loads what the first arm's server would.
        spec = json.loads(keeper.read_text())
        _, install = tune.arm_install(seal, tune.arms(seal['config'])[0])
        if spec['argv'] != rows[0]['argv'][4:-1] or \
                spec['install'] != install['install']:
            raise SystemExit('keeper-spec.json differs from the first arm\'s serve options')
        if 'weight_keeper_identity' in install and spec.get('identity_sha256') != tune.digest(install['weight_keeper_identity']):
            raise SystemExit('keeper content proof differs from its spec digest')
        print('PASS: the keeper spec holds the first arm\'s serve options')


if __name__ == '__main__':
    main(sys.argv[1])
