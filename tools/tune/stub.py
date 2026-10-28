#!/usr/bin/env python3
"""CPU stand-in for one tuner server in dry runs.

It reports the arm's speculation mode on /health, prints the model's KV line,
memory after warmup and a weight upload, tokenizes raw prompts and chat
messages, serves batched completions (one response, or a stream for sampled
windows), and writes one engine step record per step with drafts limited by
`--speculation-depths` and the verify rows (`--speculative-rows`, default 32).
With `--weight-keeper` its weights are lent. Tokens follow the prompt, sampling
and seed only, so every depth emits the same tokens; acceptance by position
decays, so deeper drafts take fewer steps. /metrics carries the speculation
counters by position. TUNE_STUB_PERTURB names a label whose greedy output
differs at token 50; TUNE_STUB_PREEMPT one whose metrics report a preemption;
TUNE_STUB_IGNORE_DEPTHS one that drafts past its depth. TUNE_STUB_PERTURB_INSTALL
selects an install whose output bits differ, including sampled A/B requests.
"""
import argparse
import hashlib
import json
import os
import queue
import struct
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from tune import argument_value, confidence_chooses, confidence_floor, draft_option

def options(arguments, ar=False):
    if ar:
        if any(draft_option(word) for word in arguments):
            raise ValueError('--speculation requires positive --speculative-tokens; AR cannot pass draft flags')
        return 'off', [(1, 0)], 32
    mode, depth, rows = 'balanced', [(1, 5)], 32
    for flag, value in zip(arguments, arguments[1:]):
        if flag == '--speculation':
            mode = value
        elif flag == '--speculation-depths':
            depth = [tuple(int(x) for x in part.split(':')) for part in value.split(',')]
        elif flag == '--speculative-rows':
            rows = int(value)
    return mode, depth, rows


def stub(args):
    lock = threading.Lock()
    label = Path(args.directory).parent.name
    if args.kv_pages is None:
        args.kv_pages = int(os.environ.get('TUNE_STUB_DEFAULT_KV_PAGES', '4096'))
        if os.environ.get('TUNE_STUB_KV_CHANGE_LABEL') == label:
            args.kv_pages += 1
    perturb = (os.environ.get('TUNE_STUB_PERTURB') == label
               or os.environ.get('TUNE_STUB_PERTURB_INSTALL') == args.install_name)
    provenance = {'install': args.install_name, 'revision': args.revision, 'synthetic': True,
                  'model_controls': {key: value for key, value in os.environ.items() if key.startswith('CHOPINFER_')}}
    proof_path = argument_value(args.serve, '--weight-keeper-identity', None)
    if proof_path:
        body = Path(proof_path).read_bytes()
        proof = json.loads(body)
        if (proof.get('schema') != 'chopinfer.weight-content.v1'
                or proof['producer_sha256'] != hashlib.sha256(Path(args.binary).read_bytes()).hexdigest()):
            raise ValueError('stub content proof belongs to another executable')
        provenance['identity_sha256'] = hashlib.sha256(body).hexdigest()
    (Path(args.directory) / 'stub-install.json').write_text(json.dumps(provenance) + '\n')
    preempt = os.environ.get('TUNE_STUB_PREEMPT') == label
    ignore = os.environ.get('TUNE_STUB_IGNORE_DEPTHS') == label
    ignore_confidence = os.environ.get('TUNE_STUB_IGNORE_CONFIDENCE') == label
    claim_confidence = os.environ.get('TUNE_STUB_CLAIM_CONFIDENCE') == label
    mode, depth, verify_rows = options(args.serve, args.ar)
    confidence = argument_value(args.serve, '--speculation-confidence', 'off') != 'off'
    floor = confidence_floor(args.serve) or 0
    widths = [int(x) for x in argument_value(args.serve, '--speculation-widths', '6,8,16,32').split(',')]
    lent = '--weight-keeper' in args.serve
    state = {'batch': 0, 'clock_ns': time.time_ns(), 'drafted': {}, 'accepted': {}, 'verify_items': 0, 'live': 0}
    control = os.environ.get('CHOPINFER_MAPPED_ROW_CONTROL')
    if control:
        def observe_control():
            epoch = 0
            while True:
                try:
                    value = json.loads(Path(control).read_text())
                    if value['epoch'] > epoch:
                        epoch = value['epoch']
                        print(json.dumps({'schema': 'chopinfer.mapped-row-policy.v1', 'status': 'applied',
                                          'epoch': epoch, 'reset_cache': value['reset_cache'],
                                          'released_mapping_bytes': 4096, 'synthetic': True}), flush=True)
                except (OSError, ValueError):
                    pass
                time.sleep(.01)
        threading.Thread(target=observe_control, daemon=True).start()
    print(args.kv_line.replace('{pages}', str(args.kv_pages)), flush=True)
    print('startup speculation_confidence: ' + ('on, stub' if confidence or claim_confidence else 'off, stub'),
          flush=True)
    for rank in range(args.gpus):
        print(f'device memory after warmup rank {rank}: KV 3560 MiB, 294 MiB taken since sizing (graphs), '
              f'{6539 + rank} MiB free', flush=True)
    print('startup weight_upload: ' + json.dumps({'uploaded_bytes': 0 if lent else 1, 'lent_bytes': int(lent)}),
          flush=True)

    def token(key, index):
        value = int.from_bytes(hashlib.sha256(f'{key}/{index}'.encode()).digest()[:4], 'little')
        logprob = -((value % 1000) / 997.0)
        if perturb and index == 50:
            logprob = struct.unpack('>d', (struct.unpack('>Q', struct.pack('>d', logprob))[0] ^ 1).to_bytes(8, 'big'))[0]
        return value % 129280, logprob

    def step(keys, positions, prefill=0):
        """One step over `keys` (streams) at `positions`; returns each
        stream's emitted token count."""
        items = len(keys)
        scheduled = [d for i, d in depth if i <= items][-1]
        k = 0 if prefill else min(scheduled, (verify_rows - items) // items)
        if ignore and not prefill and items >= 16:
            k = 1
        rows = items * (k + 1)
        rows = max((w for w in widths if items <= w <= rows), default=rows)
        per_item = min(5, max(0, verify_rows // items - 1))
        reserved = items * (1 + per_item)
        allocated = (confidence and not ignore_confidence and not prefill and items < rows < reserved
                     and confidence_chooses(floor, items, rows, per_item))
        drafts = [0 if prefill else (rows - items) // items + (i < (rows - items) % items) for i in range(items)]
        verify_items = items if allocated else sum(k > 0 for k in drafts)
        emitted = []
        with lock:
            state['batch'] += 1
            for key, at, k in zip(keys, positions, drafts):
                accepted = 0
                for j in range(1, k + 1):
                    state['drafted'][j] = state['drafted'].get(j, 0) + 1
                for j in range(1, k + 1):
                    survive = 0.8 * 0.85 ** (j - 1)
                    roll = int.from_bytes(hashlib.sha256(f'{key}/{at}/{j}'.encode()).digest()[:2], 'little') / 65536
                    if roll >= survive:
                        break
                    accepted += 1
                    state['accepted'][j] = state['accepted'].get(j, 0) + 1
                emitted.append(1 + accepted)
            state['verify_items'] += verify_items
            micros = 12_000 + 600 * rows + 4000 * bool(verify_items)
            state['clock_ns'] = time.time_ns() if os.environ.get('TUNE_STUB_CLOSED_LOOP') else state['clock_ns'] + micros * 1000
            if os.environ.get('TUNE_STUB_CLOSED_LOOP'):
                micros = 1000
            print(json.dumps({
                'schema': 'chopinfer.step.v1', 'batch': state['batch'], 'prefill_tokens': prefill,
                'prefill_items': items if prefill else 0, 'decode_rows': 0 if prefill else items - verify_items,
                'verify_items': verify_items, 'verify_rows': sum(drafts) + verify_items,
                'drafted_tokens': sum(drafts),
                **({'reserved_rows': reserved} if allocated else {}),
                'accepted_tokens': sum(emitted) - items,
                'verify_emitted_tokens': sum(n for n, k in zip(emitted, drafts) if k or allocated),
                'execution_us': micros, 'completed_unix_ns': state['clock_ns'],
                'host': {'mapped_gather_us': 300 if prefill else 10},
                'route': {'fallback_launches': 0, 'families': {}},
                'speculation': {'mode': mode, 'items': items, 'reason': 'balanced'}}), flush=True)
        time.sleep(.0005)
        return [0] * items if prefill else emitted

    pending, active = [], []
    scheduled = threading.Condition()

    def batcher():
        while True:
            with scheduled:
                while not pending and not active:
                    scheduled.wait()
                new, pending[:] = list(pending), []
                active.extend(new)
            if new:
                time.sleep(float(os.environ.get('TUNE_STUB_PREFILL_DELAY_SECONDS', '0')))
                step([row['key'] for row in new], [0] * len(new),
                     prefill=sum(row['prompt_tokens'] for row in new))
            if active:
                emitted = step([row['key'] for row in active], [row['position'] for row in active])
                keep = []
                for row, n in zip(active, emitted):
                    start = row['position']
                    take = min(n, row['limit'] - start)
                    row['position'] += take
                    done = row['position'] == row['limit']
                    row['queue'].put((row['index'], row['key'], start, take, done))
                    if not done:
                        keep.append(row)
                active[:] = keep
            time.sleep(.002)

    if os.environ.get('TUNE_STUB_CLOSED_LOOP'):
        threading.Thread(target=batcher, daemon=True).start()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *unused):
            pass

        def respond(self, value, content='application/json'):
            data = (value if isinstance(value, str) else json.dumps(value)).encode()
            self.send_response(200)
            self.send_header('Content-Type', content)
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == '/health':
                self.respond({'status': 'ready', 'synthetic': True,
                              'serving': {'profile': {'speculation': {'mode': mode, 'tokens': 0 if args.ar else 5}}}})
            elif self.path == '/v1/models':
                self.respond({'data': [{'id': args.model}]})
            elif self.path == '/metrics':
                label = f'model_name="{args.model}"'
                with lock:
                    drafted, accepted = dict(state['drafted']), dict(state['accepted'])
                    items = state['verify_items']
                lines = [f'vllm:num_preemptions_total{{{label},engine="0"}} {1 if preempt else 0}',
                         'chop_prefix_cache_evictions_total{class="superseded"} 0',
                         'chop_prefix_cache_evictions_total{class="retained"} 0',
                         'chop_prefix_cache_evictions_total{class="maximal"} 0',
                         f'chop_spec_verify_items_total{{{label}}} {items}',
                         f'chop_spec_draft_tokens_total{{{label}}} {sum(drafted.values())}',
                         f'chop_spec_accepted_tokens_total{{{label}}} {sum(accepted.values())}']
                lines += [f'chop_spec_drafted_position_total{{{label},position="{j}"}} {n}' for j, n in drafted.items()]
                lines += [f'chop_spec_accepted_position_total{{{label},position="{j}"}} {accepted.get(j, 0)}'
                          for j in drafted]
                self.respond('\n'.join(lines) + '\n', 'text/plain')
            else:
                self.send_error(404)

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            if self.path == '/tokenize':
                text = json.dumps(body.get('messages')) if 'messages' in body else body['prompt']
                base = int.from_bytes(hashlib.sha256(text.encode()).digest()[:2], 'little')
                token_bytes = int(os.environ.get('TUNE_STUB_TOKEN_BYTES', '64'))
                length = 8 + len(text) // token_bytes
                if os.environ.get('TUNE_STUB_TOKEN_LENGTHS'):
                    lengths = [int(n) for n in os.environ['TUNE_STUB_TOKEN_LENGTHS'].split(',')]
                    length = lengths[base % len(lengths)]
                    base = int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], 'little')
                self.respond({'tokens': [base + i for i in range(length)]})
                return
            assert self.path == '/v1/completions' and body.get('ignore_eos') and body.get('logprobs') == 1
            prompts, out = body['prompt'], body['max_tokens']
            assert body.get('min_tokens') == out and all(isinstance(p, list) for p in prompts)
            sampled = body.get('temperature', 0) > 0
            keys = [f"{body.get('temperature')}/{body.get('seed')}/{p}" for p in prompts]
            usage = {'prompt_tokens': sum(map(len, prompts)), 'completion_tokens': out * len(prompts),
                     'prompt_tokens_details': {'cached_tokens': int(os.environ.get('TUNE_STUB_CACHED') == label)}}
            if os.environ.get('TUNE_STUB_CLOSED_LOOP'):
                events = queue.Queue()
                if sampled:
                    self.send_response(200)
                    self.send_header('Content-Type', 'text/event-stream')
                    self.end_headers()
                with scheduled:
                    pending.extend({'key': key, 'position': 0, 'limit': out, 'prompt_tokens': len(prompts[i]),
                                    'index': i, 'queue': events} for i, key in enumerate(keys))
                    scheduled.notify_all()
                finished = 0
                while finished < len(keys):
                    i, key, start, take, done = events.get(timeout=60)
                    finished += done
                    if sampled:
                        rows = [token(key, j) for j in range(start, start + take)]
                        event = {'choices': [{'index': i, 'token_ids': [x[0] for x in rows],
                                  'logprobs': {'token_logprobs': [x[1] for x in rows]},
                                  'finish_reason': 'length' if done else None}]}
                        self.wfile.write(('data: ' + json.dumps(event) + '\n\n').encode())
                if sampled:
                    self.wfile.write(('data: ' + json.dumps({'choices': [], 'usage': usage}) +
                                      '\n\ndata: [DONE]\n\n').encode())
                else:
                    choices = []
                    for i, key in enumerate(keys):
                        rows = [token(key, j) for j in range(out)]
                        choices.append({'index': i, 'token_ids': [x[0] for x in rows], 'finish_reason': 'length',
                                        'logprobs': {'token_logprobs': [x[1] for x in rows]}})
                    self.respond({'choices': choices, 'usage': usage})
                return
            with lock:
                state['live'] += len(keys)
            step(keys, [0] * len(keys), prefill=sum(map(len, prompts)))
            done = [0] * len(keys)
            if sampled:
                self.send_response(200)
                self.send_header('Content-Type', 'text/event-stream')
                self.end_headers()
            while True:
                live = [i for i in range(len(keys)) if done[i] < out]
                if not live:
                    break
                emitted = step([keys[i] for i in live], [done[i] for i in live])
                for i, n in zip(live, emitted):
                    take = min(n, out - done[i])
                    if sampled:
                        rows = [token(keys[i], j) for j in range(done[i], done[i] + take)]
                        event = {'choices': [{'index': i, 'token_ids': [x[0] for x in rows],
                                              'logprobs': {'token_logprobs': [x[1] for x in rows]},
                                              'finish_reason': 'length' if done[i] + take == out else None}]}
                        self.wfile.write(('data: ' + json.dumps(event) + '\n\n').encode())
                    done[i] += take
            if sampled:
                self.wfile.write(('data: ' + json.dumps({'choices': [], 'usage': usage}) + '\n\ndata: [DONE]\n\n').encode())
                with lock:
                    state['live'] -= len(keys)
                return
            choices = []
            for i, key in enumerate(keys):
                rows = [token(key, j) for j in range(out)]
                choices.append({'index': i, 'token_ids': [x[0] for x in rows], 'finish_reason': 'length',
                                'logprobs': {'token_logprobs': [x[1] for x in rows]}})
            self.respond({'choices': choices, 'usage': usage})
            with lock:
                state['live'] -= len(keys)

    class HTTPServer(ThreadingHTTPServer):
        request_queue_size = 128

    server = HTTPServer(('127.0.0.1', args.port), Handler)
    server.serve_forever()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, required=True)
    ap.add_argument('--directory', required=True)
    ap.add_argument('--kv-pages', type=int)
    ap.add_argument('--kv-line', required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--gpus', type=int, required=True)
    ap.add_argument('--install-name', default='head')
    ap.add_argument('--revision', default='template')
    ap.add_argument('--binary')
    ap.add_argument('--ar', action='store_true')
    ap.add_argument('serve', nargs=argparse.REMAINDER)
    parsed = ap.parse_args()
    parsed.serve = [x for x in parsed.serve if x != '--']
    stub(parsed)
