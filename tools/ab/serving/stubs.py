#!/usr/bin/env python3
"""CPU fixtures for the serving runner tests: a stub server, client, GPU query
and profiler. This module never calls a GPU tool or a real server."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

TOTALS = ['chop_spec_verify_items_total', 'chop_spec_draft_tokens_total',
          'chop_spec_accepted_tokens_total', 'chop_spec_emitted_tokens_total', 'chop_spec_rows_total']


def metrics(steps=0, speculative=True, accepted_per_step=3):
    lines = [f'chop_requests_completed_total {steps}', 'chop_requests_waiting 0']
    if steps and speculative:
        values = [steps, 3 * steps, accepted_per_step * steps, (1 + accepted_per_step) * steps, 4 * steps]
        lines += [f'{name} {value}' for name, value in zip(TOTALS, values)]
        for name in ['chop_spec_drafted_position_total', 'chop_spec_accepted_position_total']:
            maximum = 3 if name == 'chop_spec_drafted_position_total' else accepted_per_step
            lines += [f'{name}{{position="{p}"}} {steps if p <= maximum else 0}' for p in range(1, 17)]
    lines += [f'chop_speculation_k_total{{model_name="stub",engine="mtp",k="{k}"}} '
              f'{steps if k == (3 if speculative else 0) else 0}' for k in range(17)]
    return '\n'.join(lines) + '\n'


def choice(count=1, top=5, offset=0, value=-0.5):
    return dict(tokens=['a'] * count, token_logprobs=[None] * offset + [value] * (count - offset),
                top_logprobs=[None] * offset + [{chr(97 + k): value - k for k in range(top)}
                                              for _ in range(count - offset)])


def phase(name, catalog=None, value=-0.5):
    if catalog is None:
        from evidence import catalog as get_catalog
        catalog = get_catalog()
    d = catalog[name]
    p = dict(phase=name, kind=d['kind'], wall_ms=10.0, start_unix_ns=1, end_unix_ns=10000001)
    if d['kind'] == 'warm':
        from evidence import PREFIX_TOKENS
        return dict(p, requests=1, prompt_tokens=PREFIX_TOKENS, completion_tokens=1, choices=1,
                    outputs=[dict(text='a', token_ids=[1], finish_reason='length')])
    if d['kind'] == 'gsm8k':
        return dict(p, requests=8, concurrency=d['concurrency'], prompt_tokens=500,
                    completion_tokens=8, outputs=[dict(text='42', reasoning=None, token_ids=[1], completion_tokens=1,
                    finish_reason='stop') for _ in range(8)])
    s = d['spec']; prompt = s['prompt_tokens']['min']; n = s['max_tokens']
    output = dict(text='a', token_ids=[1] * n, finish_reason='length')
    if d['kind'] == 'decode':
        output.update(logprobs=[choice(n, 1, value=value)], done=True)
        p.update(ms_per_token=[1.0] * s['requests'], tokens_per_s=1000.0)
    elif d['kind'] == 'teacher':
        output['logprobs'] = choice(prompt + n, 5, 1, value)
        scores = {str(k): dict(decoded_token=chr(97 + k), rank=k + 1, logprob=value - k) for k in range(5)}
        output['prompt_logprobs'] = [None] + [dict(scores) for _ in range(prompt - 1)]
    else:
        output['logprobs'] = choice(n, 5, value=value)
        p['prompt_tokens_per_s'] = 1000.0
    return dict(p, requests=s['requests'], prompt_tokens=prompt, completion_tokens=s['requests'] * n,
                outputs=[json.loads(json.dumps(output)) for _ in range(s['requests'])])


# tools/qual/determinism.py for a run file's qual_tools: `record` writes a
# receipt naming its arguments and scorer; `compare` passes when the two
# receipts' requests are equal.
DETERMINISM = """import json, os, pathlib, sys
args = sys.argv[1:]
if args[0] == 'record':
    out = pathlib.Path(os.environ['CHOPINFER_QUAL_RECEIPTS']) / 'stub-determinism.json'
    out.write_text(json.dumps(dict(passed=True, requests=[dict(name='capital', text='Paris')], argv=args,
                                   scorer=os.environ.get('CHOPINFER_EVAL'))))
else:
    left, right = (json.load(open(path)) for path in args[1:3])
    same = left['requests'] == right['requests']
    print(json.dumps(dict(passed=same)))
    sys.exit(0 if same else 1)
"""


def option(name, default=None):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


def append(path, text):
    if path:
        with open(path, 'a') as stream:
            stream.write(text + '\n')


def server():
    mode = os.environ.get('STUB_SERVER', '')
    append(os.environ.get('STUB_STARTS'), str(os.getpid()))
    if mode == 'exit':
        return 7
    if mode == 'hang':
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        while True:
            time.sleep(1)
    cache = Path(option('--model-cache-dir'))
    packages = list((cache / 'records').glob('p-*.json'))
    package = packages[0].stem[2:]
    record = json.loads(packages[0].read_text()); record['value']['last_selected'] += 1
    record['checksum'] = hashlib.sha256(json.dumps(record['value'], sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    packages[0].write_text(json.dumps(record))
    install = Path(sys.argv[0]).resolve().parent.parent
    receipt = json.loads((install / 'INSTALL.json').read_text())
    tokens = int(option('--speculative-tokens', '0'))
    served = dict(bundle_id='stub.v1', checkpoint_identity='stub-checkpoint', source_commit=receipt['commit'],
        metadata_identity='e' * 64, package_identity=package, startup_nonce=uuid.uuid4().hex,
        model_options={'checkpoint.quantization_layout': os.environ.get('STUB_LAYOUT', 'stub-fp8')},
        profile=dict(tensor_parallel_size=1, context_parallel_size=1, kv_cache_dtype='fp8', prefix_cache=True,
                     max_sequences=32, speculation=dict(tokens=tokens, mode=option('--speculation', 'off'),
                                                        drafter='mtp' if tokens else None)))
    if mode == 'bad-identity':
        served['source_commit'] = 'f' * 40
    print('startup serving_identity: ' + json.dumps(served), flush=True)
    state = dict(steps=0)
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass
        def do_GET(self):
            if self.path == '/health':
                value = json.dumps(dict(status='ready', serving=dict(served, served_model_names=['stub-model'])))
                if mode == 'slow-health':
                    self.send_response(200); self.send_header('Content-Length', str(len(value))); self.end_headers()
                    try:
                        for char in value:
                            self.wfile.write(char.encode()); self.wfile.flush(); time.sleep(1)
                    except (BrokenPipeError, ConnectionResetError):
                        pass
                    return
            elif self.path == '/metrics':
                value = metrics(state['steps'], bool(tokens), int(os.environ.get('STUB_ACCEPT', '3')))
                if os.environ.get('STUB_BAD_METRICS'):
                    value = ''
            else:
                self.send_error(404); return
            data = value.encode(); self.send_response(200); self.send_header('Content-Length', str(len(data)))
            self.end_headers(); self.wfile.write(data)
        def do_POST(self):
            if self.path != '/step':
                self.send_error(404); return
            length = int(self.headers.get('Content-Length', '0'))
            self.rfile.read(length)
            state['steps'] += 1
            self.send_response(200); self.send_header('Content-Length', '0'); self.end_headers()
    host, port = option('--listen').rsplit(':', 1)
    ThreadingHTTPServer((host, int(port)), Handler).serve_forever()


def client():
    # The snapshotted client is this fixture; its evidence module is snapshotted too.
    url, output, kind, name = sys.argv[1:5]
    if os.environ.get('STUB_CLIENT') == 'fail':
        return 8
    data = json.loads(Path(output).read_text()) if Path(output).exists() else dict(model='stub-model', phases=[])
    value = -0.5
    if os.environ.get('STUB_DIFFERENT'):
        value = -0.6
    if kind == 'determinism':
        # The shared qualification tools need the new declaration operation.
        scorer = option('--scorer')
        subprocess.run([scorer, 'eval', 'declaration'], check=True)
        # A receipt as tools/qual/determinism.py writes one; its answer
        # differs when STUB_DETERMINISM_DIFFERENT is set.
        receipts = Path(output).parent / ('receipts-' + name)
        receipts.mkdir()
        path = receipts / 'stub-determinism.json'
        text = 'different' if os.environ.get('STUB_DETERMINISM_DIFFERENT') else 'Paris'
        path.write_text(json.dumps(dict(passed=True, requests=[dict(name='capital', text=text)], scorer=scorer)))
        p = dict(phase=name, kind=kind, wall_ms=10.0, start_unix_ns=1, end_unix_ns=2, requests=1, image=False,
                 receipt=str(path), receipt_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), passed=True)
    else:
        p = phase(name, value=value)
    if os.environ.get('STUB_CLIENT') == 'short':
        p['prompt_tokens'] = 1
    if os.environ.get('STUB_CLIENT') == 'empty-warm' and kind == 'warm':
        # A 200 response with no choice, recorded as is.
        p.update(choices=0, completion_tokens=0, outputs=[dict(text=None, token_ids=None, finish_reason=None)])
    data['phases'].append(p)
    Path(output).write_text(json.dumps(data))
    if kind in ('decode', 'gsm8k'):
        with urllib.request.urlopen(urllib.request.Request(url + '/step', data=b'x'), timeout=60):
            pass
    return 0


def query():
    append(os.environ.get('STUB_QUERIES'), 'query')
    mode = os.environ.get('STUB_QUERY', '')
    if mode == 'fail':
        return 3
    if mode in ('hang', 'late'):
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        time.sleep(30 if mode == 'hang' else .2)
    if mode == 'noise':
        print('driver failed', file=sys.stderr)
    if mode == 'malformed':
        print('not-a-pid'); return 0
    if mode == 'child-pipe':
        child = subprocess.Popen([sys.executable, __file__, 'child'], start_new_session=True)
        append(os.environ.get('STUB_CHILDREN'), str(child.pid))
    if any(x.startswith('--query-compute-apps=') for x in sys.argv):
        if mode == 'busy':
            print('999999999')
        elif mode in ('release-fail', 'release-query-fail') and Path(os.environ['STUB_STARTS']).exists():
            pid = int(Path(os.environ['STUB_STARTS']).read_text().splitlines()[-1])
            stat = Path('/proc') / str(pid) / 'stat'
            live = stat.exists() and stat.read_text().rsplit(')', 1)[1].split()[0] not in ('Z', 'X')
            if live:
                print(pid)
            elif mode == 'release-fail':
                print('999999999')
            else:
                return 3
    else:
        print('0, 1000, 2000, 150.0')
    return 0


def profile():
    if 'shutdown' in sys.argv:
        return 9 if os.environ.get('STUB_PROFILE_FAIL') else 0
    session = next(x.split('=', 1)[1] for x in sys.argv if x.startswith('--session-new='))
    index = next(i for i, x in enumerate(sys.argv) if x == sys.executable)
    child = subprocess.Popen(sys.argv[index:], start_new_session=True)
    agent_env = dict(os.environ); agent_env.pop('CHOPINFER_AB_OWNER', None)
    agent = subprocess.Popen([sys.executable, __file__, 'child', '--start-agent', '--session-name', session],
                             start_new_session=True, env=agent_env)
    append(os.environ.get('STUB_CHILDREN'), str(agent.pid)); append(os.environ.get('STUB_CHILDREN'), str(child.pid))
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    while True:
        time.sleep(1)


def main():
    if sys.argv[1:3] == ['eval', 'declaration']:
        install = Path(sys.argv[0]).resolve().parent.parent
        receipt = json.loads((install / 'INSTALL.json').read_text())
        if not receipt.get('stub_declaration', receipt['commit'] == 'b' * 40):
            print("eval: unsupported eval command 'declaration'", file=sys.stderr)
            return 2
        return 0
    if len(sys.argv) > 1 and sys.argv[1].startswith('http://'):
        return client()
    if len(sys.argv) > 1 and sys.argv[1] == 'query':
        return query()
    if len(sys.argv) > 1 and sys.argv[1] == 'profiler':
        return profile()
    if 'serve' in sys.argv:
        return server()
    if len(sys.argv) > 1 and sys.argv[1] == 'child':
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        while True:
            time.sleep(1)
    raise SystemExit('CPU fixture mode missing')


if __name__ == '__main__':
    sys.exit(main())
