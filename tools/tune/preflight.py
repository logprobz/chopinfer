#!/usr/bin/env python3
"""Parse every arm's server configuration with its selected frozen
executable, without CUDA (--list-model-options), before the job is queued.

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


def main(stage):
    stage = Path(stage).resolve()
    seal = json.loads((stage / 'runtime-seal.json').read_text())
    if not seal['ready']:
        raise SystemExit('a template stage has no executable to parse with')
    model = seal['config']['model']
    tune.check_installs(seal, require_proofs=True)
    r.MODEL, r.CHECKPOINT_ENV, r.GPUS = model['served_model_name'], model['checkpoint_env'], model['tensor_parallel_size']
    r.KV_PAGES, r.KV_LINE, r.SERVE = model['kv_pages'], model['kv_pages_line'], model['serve_arguments']
    root = stage / 'validation' / f'argv-{time.time_ns()}'
    root.mkdir(parents=True)
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
