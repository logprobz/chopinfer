#!/usr/bin/env python3
"""Parse every depth's server configuration with the stage's frozen
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


def main(stage):
    stage = Path(stage).resolve()
    seal = json.loads((stage / 'runtime-seal.json').read_text())
    if not seal['ready']:
        raise SystemExit('a template stage has no executable to parse with')
    model, install = seal['config']['model'], seal['install']
    r.MODEL, r.CHECKPOINT_ENV, r.GPUS = model['served_model_name'], model['checkpoint_env'], model['tensor_parallel_size']
    r.KV_PAGES, r.KV_LINE, r.SERVE = model['kv_pages'], model['kv_pages_line'], model['serve_arguments']
    root = stage / 'validation' / f'argv-{time.time_ns()}'
    root.mkdir(parents=True)
    rows = []
    for depth in model.get('depths') or range(model['max_depth'] + 1):
        arguments = ['--speculation', model['speculation_mode'], '--speculation-depths', f'1:{depth}',
                     *model.get('extra_arguments', [])]
        server = r.Server(root / f'd{depth}', install['install'], install['cache'], arguments)
        argv = server.server_command + ['--list-model-options']
        if any('kv-memory-fraction' in x for x in argv):
            raise SystemExit('the tuner serves the test KV mode, never a KV memory fraction')
        result = subprocess.run(argv, env=dict(server.env, CUDA_VISIBLE_DEVICES=''), stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True, timeout=180)
        (root / f'd{depth}' / 'stdout.json').write_text(result.stdout)
        (root / f'd{depth}' / 'stderr.log').write_text(result.stderr)
        if result.returncode != 0:
            raise SystemExit(f'depth {depth}: exit {result.returncode}: {result.stderr[-2000:]}')
        json.loads(result.stdout)
        rows.append({'depth': depth, 'argv': argv, 'exit': 0})
    (root / 'preflight.json').write_text(json.dumps({'passed': True, 'cpu_only': True, 'cases': rows}, indent=2) + '\n')
    print(f'PASS: {len(rows)} frozen server parsers, no CUDA: {root}')


if __name__ == '__main__':
    main(sys.argv[1])
