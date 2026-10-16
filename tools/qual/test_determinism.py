"""CPU checks of the served-output determinism record and its bit-exact comparison."""
import base64
import contextlib
import copy
import hashlib
import io
import json
import math
import os
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zlib

import determinism


LONG_IDS = [(7 * i + 3) % 50000 for i in range(2800)]


class FakeClient:
    """Answers every request greedily from its prompt. A token-ID prompt
    primes its salt; a later request under that salt reports a hit at the
    primer's last 64-token granule, as Qwen does. Others report `cached`."""

    def __init__(self, cached=0, hits=True):
        self.bodies = []
        self.cached = cached
        self.hits = hits
        self.primed = {}

    def identity(self, model=None):
        return {'url': 'http://fake', 'model': 'served', 'models': ['served'], 'health': {'status': 'ok'}}

    def request(self, path, body):
        self.bodies.append((path, body))
        if path == '/tokenize':
            return {'count': len(LONG_IDS), 'max_model_len': 262144, 'tokens': LONG_IDS, 'token_strs': None}, 0.01
        prompt = body.get('prompt') or json.dumps(body['messages'])
        text = prompt if isinstance(prompt, str) else json.dumps(prompt)
        seed = int(hashlib.sha256(text.encode()).hexdigest()[:6], 16)
        ids = [(seed + i) % 1000 for i in range(min(3, body['max_tokens']))]
        tokens = [f't{token}' for token in ids]
        scores = [-1.0 / (2 + token) for token in ids]
        primer = self.primed.get(body['cache_salt'])
        cached = primer // 64 * 64 if primer and self.hits else self.cached
        if isinstance(prompt, list):
            self.primed[body['cache_salt']] = len(prompt)
        usage = {'prompt_tokens': len(prompt) if isinstance(prompt, list) else 2800 if len(prompt) > 10000 else 9,
                 'completion_tokens': len(ids), 'total_tokens': 0, 'prompt_tokens_details': {'cached_tokens': cached}}
        if path == '/v1/chat/completions':
            choice = {'message': {'role': 'assistant', 'content': ''.join(tokens)}, 'token_ids': ids,
                      'finish_reason': 'length', 'logprobs': {'content': [
                          {'token': t, 'logprob': s, 'top_logprobs': [{'token': t, 'logprob': s},
                                                                      {'token': 'x', 'logprob': s - 1}]}
                          for t, s in zip(tokens, scores)]}}
        else:
            choice = {'text': ''.join(tokens), 'token_ids': ids, 'finish_reason': 'length', 'logprobs': {
                'tokens': tokens, 'token_logprobs': scores,
                'top_logprobs': [{t: s, 'x': s - 1} for t, s in zip(tokens, scores)]}}
        return {'choices': [choice], 'usage': usage}, 0.01


class BoundClient(FakeClient):
    """A fake server that reports a serving identity and names it on every response."""

    def __init__(self, identity, nonce='a' * 32, model=None, **kwargs):
        super().__init__(**kwargs)
        self.serving = {'checkpoint_identity': identity, 'startup_nonce': nonce}
        self.model = model
        self.responders = {(identity, nonce)}

    def identity(self, model=None):
        return {'url': 'http://127.0.0.1:39001', 'model': self.model, 'models': [self.model],
                'health': {'status': 'ok', 'serving': dict(self.serving)}}


def startup_log(directory, identity, nonce='a' * 32):
    """The startup log of the fake server at 127.0.0.1:39001."""
    path = Path(directory) / f'server-{identity[:8]}-{nonce[:4]}.log'
    serving = json.dumps({'checkpoint_identity': identity, 'startup_nonce': nonce})
    path.write_text(f"startup direct_package: {json.dumps({'checkpoint_identity': identity})}\n"
                    f"startup serving_identity: {serving}\n"
                    + json.dumps({'status': 'listening', 'address': '127.0.0.1:39001'}, separators=(',', ':'))
                    + '\n')
    return path


def recorded(image=True, arm='spec', client=None, server_log=None):
    """The receipt value `record` writes against a fake server."""
    client = client or FakeClient()
    written = []
    args = SimpleNamespace(url='http://127.0.0.1:39001' if server_log else 'http://fake', timeout=10, model=None,
                           arm=arm, run_id='run', image=image, image_reason=None, server_log=server_log)
    with patch.object(determinism, 'Client', return_value=client), \
         patch.object(determinism, 'answer_kwargs', return_value={'enable_thinking': False}), \
         patch.object(determinism, 'receipt', side_effect=lambda check, server, value, mechanics: written.append(
             dict(value, check=check, server=server, mechanics_only=mechanics))):
        determinism.record(args)
    return written[0]


class BoundRecordTests(unittest.TestCase):
    def test_a_record_with_a_server_log_binds_the_pinned_live_process(self):
        import common
        import declaration
        # Any shipped model declaration: its pin is the one the server must report.
        config = sorted(declaration.ROOT.glob('models/*/qual.json'))[0]
        with patch.dict(os.environ, {'CHOPINFER_QUAL_CONFIG': str(config)}):
            self.bound_records(common.checkpoint_pin())

    def bound_records(self, pin):
        with tempfile.TemporaryDirectory() as directory:
            log = startup_log(directory, pin['checkpoint_identity'])
            value = recorded(image=False, client=BoundClient(pin['checkpoint_identity'],
                                                             model=pin['served_model_name']), server_log=log)
            self.assertEqual(value['server']['checkpoint']['checkpoint_identity'], pin['checkpoint_identity'])
            self.assertEqual(value['server']['checkpoint']['startup_nonce'], 'a' * 32)
            self.assertEqual(value['server']['checkpoint']['server_log_sha256'],
                             hashlib.sha256(log.read_bytes()).hexdigest())
            # Another checkpoint, or another process answering, is refused.
            with self.assertRaisesRegex(ValueError, 'not the pinned'):
                recorded(image=False, client=BoundClient('1' * 64, model=pin['served_model_name']),
                         server_log=startup_log(directory, '1' * 64))
            interloped = BoundClient(pin['checkpoint_identity'], model=pin['served_model_name'])
            interloped.responders.add(('2' * 64, 'b' * 32))
            with self.assertRaises(SystemExit):
                recorded(image=False, client=interloped, server_log=log)


def row(value, name):
    return next(row for row in value['requests'] if row['name'] == name)


class RequestSetTests(unittest.TestCase):
    def test_request_set_is_fixed(self):
        self.assertEqual(determinism.request_set(True), determinism.request_set(True))
        self.assertEqual(determinism.request_digest(determinism.request_set(True)),
                         determinism.request_digest(determinism.request_set(True)))
        self.assertNotEqual(determinism.request_digest(determinism.request_set(True)),
                            determinism.request_digest(determinism.request_set(False)))
        names = [name for name, _, _ in determinism.request_set(True)]
        self.assertEqual(names, ['capital', 'haiku', 'fibonacci', 'arithmetic', 'long', 'long-cold-repeat', 'image'])
        self.assertEqual(names[:-1], [name for name, _, _ in determinism.request_set(False)])
        prompts = [body.get('prompt') for _, _, body in determinism.request_set(False)]
        self.assertEqual(prompts[:4], ['The capital of France is', 'Write a haiku about the ocean.',
                                       'def fibonacci(n):', 'Question: What is 17 multiplied by 23? Answer:'])
        self.assertEqual(prompts[4], prompts[5])
        for name, path, body in determinism.request_set(True, {'enable_thinking': False}):
            self.assertEqual((body['temperature'], body['max_tokens'], body['return_token_ids']), (0, 48, True))
            if name == 'image':
                self.assertEqual(path, '/v1/chat/completions')
                self.assertEqual((body['logprobs'], body['top_logprobs']), (True, 5))
                self.assertEqual(body['chat_template_kwargs'], {'enable_thinking': False})
            else:
                self.assertEqual((path, body['logprobs']), ('/v1/completions', 5))

    def test_long_prompt_is_pinned_and_ends_in_a_question(self):
        text = determinism.long_prompt()
        self.assertEqual(text, determinism.long_prompt())
        self.assertEqual(text.count('\nRule '), 92)
        self.assertTrue(text.endswith('Answer:'))
        # About 2,800 MiMo and 2,900 Qwen tokens; changing it changes every receipt's request set.
        self.assertEqual(len(text), 11090)
        self.assertEqual(hashlib.sha256(text.encode()).hexdigest(),
                         '02bd617350125a586f3bb71670a6dd1a8acc5e66ca6a0a2e23a9c76f971e9c2e')

    def test_image_is_a_valid_generated_png(self):
        data = determinism.image_png()
        self.assertEqual(data, determinism.image_png())
        self.assertTrue(data.startswith(b'\x89PNG\r\n\x1a\n'))
        offset, chunks = 8, {}
        while offset < len(data):
            length, = struct.unpack('>I', data[offset:offset + 4])
            kind, body = data[offset + 4:offset + 8], data[offset + 8:offset + 8 + length]
            crc, = struct.unpack('>I', data[offset + 8 + length:offset + 12 + length])
            self.assertEqual(crc, zlib.crc32(kind + body) & 0xFFFFFFFF)
            chunks[kind] = body
            offset += 12 + length
        self.assertEqual(struct.unpack('>II', chunks[b'IHDR'][:8]), (256, 256))
        pixels = zlib.decompress(chunks[b'IDAT'])
        self.assertEqual(len(pixels), 256 * (1 + 256 * 3))
        self.assertEqual(pixels[1:4], bytes((220, 30, 30)))
        url = determinism.request_set(True)[-1][2]['messages'][0]['content'][0]['image_url']['url']
        self.assertEqual(base64.b64decode(url.split(',', 1)[1]), data)


class RecordTests(unittest.TestCase):
    def test_record_sends_one_request_at_a_time_each_with_its_own_salt(self):
        client = FakeClient()
        value = recorded(client=client)
        determinism.check(value)
        self.assertEqual([body.get('prompt', 'chat') for _, body in client.bodies][:4],
                         list(determinism.SHORT_PROMPTS.values()))
        self.assertEqual([row['name'] for row in value['requests']], determinism.request_names(True))
        self.assertEqual(value['request_set']['names'], determinism.request_names(True))
        # Every cold request has its own salt; each hit reuses its primer's.
        sends = [body for path, body in client.bodies if path != '/tokenize']
        salts = [body['cache_salt'] for body in sends]
        self.assertEqual(len(set(salts)), len(salts) - len(determinism.HIT_ANCHORS))
        for anchor in determinism.HIT_ANCHORS:
            self.assertEqual(row(value, f'long-hit-{anchor}')['cache_salt'],
                             row(value, f'long-primer-{anchor}')['cache_salt'])
        self.assertTrue(all(body['model'] == 'served' for body in sends))
        self.assertEqual(value['request_set']['sha256'], determinism.request_digest(
            determinism.request_set(True, {'enable_thinking': False}) + determinism.hit_requests(LONG_IDS)))
        self.assertEqual(value['request_set']['long_prompt_tokens'], 2800)
        self.assertEqual(value['request_set']['long_prompt_token_ids_sha256'],
                         hashlib.sha256(json.dumps(LONG_IDS, separators=(',', ':')).encode()).hexdigest())
        self.assertTrue(value['cold_repeat']['supported'] and value['cold_repeat']['equal'])
        self.assertEqual(value['cold_repeat']['mechanism'], 'cache_salt')
        self.assertTrue(value['prefix_hit']['supported'] and value['prefix_hit']['equal'])
        self.assertEqual(value['prefix_hit']['anchors'], [1024, 2000])
        image = value['requests'][6]
        self.assertEqual(image['top_logprobs'][0][1][0], 'x')
        # Completion top logprobs keep the server's order as pairs.
        self.assertEqual([pair[0] for pair in value['requests'][0]['top_logprobs'][0]][1], 'x')
        self.assertEqual(value['requests'][0]['usage']['prompt_tokens_details']['cached_tokens'], 0)
        self.assertFalse(recorded(image=False)['image']['included'])

    def test_record_tokenizes_the_long_prompt_and_primes_each_anchor(self):
        client = FakeClient()
        value = recorded(image=False, client=client)
        paths = [path for path, _ in client.bodies]
        tokenize = paths.index('/tokenize')
        self.assertEqual(tokenize, len(determinism.request_set(False)))
        self.assertEqual(client.bodies[tokenize][1], {'model': 'served', 'prompt': determinism.long_prompt()})
        bodies = [body for _, body in client.bodies[tokenize + 1:]]
        self.assertEqual([body['prompt'] for body in bodies],
                         [LONG_IDS, LONG_IDS[:1024], LONG_IDS, LONG_IDS[:2000], LONG_IDS])
        self.assertEqual([body['max_tokens'] for body in bodies], [48, 1, 48, 1, 48])
        for body in bodies:
            self.assertEqual((body['temperature'], body['top_p'], body['logprobs'], body['return_token_ids']),
                             (0, 1, 5, True))
        cached = {name: row(value, name)['usage']['prompt_tokens_details']['cached_tokens']
                  for name in determinism.request_names(False)[-5:]}
        self.assertEqual(cached, {'long-ids': 0, 'long-primer-1024': 0, 'long-hit-1024': 1024,
                                  'long-primer-2000': 0, 'long-hit-2000': 1984})

    def test_record_refuses_a_cached_request(self):
        with self.assertRaisesRegex(ValueError, 'cached_tokens=0'):
            recorded(client=FakeClient(cached=16))

    def test_record_refuses_a_hit_that_prefilled_cold(self):
        with self.assertRaisesRegex(ValueError, 'long-hit-1024: a prefix hit requires reported 0 < cached_tokens <= 1024'):
            recorded(client=FakeClient(hits=False))


class CheckTests(unittest.TestCase):
    def test_incomplete_or_warm_records_fail(self):
        good = recorded()
        determinism.check(good)
        for mutation, message in (
                (lambda v: v['requests'][0]['usage']['prompt_tokens_details'].update(cached_tokens=5), 'not cold'),
                (lambda v: v['requests'].pop(), 'missing or reordered'),
                (lambda v: v['requests'][4]['usage'].update(prompt_tokens=2048), 'prefill chunk'),
                (lambda v: v['requests'][5]['usage'].update(prompt_tokens=2900), 'different prompts'),
                (lambda v: v['requests'][1].update(cache_salt=v['requests'][0]['cache_salt']), 'cache salt'),
                (lambda v: v['requests'][2]['logprobs'].__setitem__(0, float('nan')), 'selected logprobs'),
                (lambda v: v['requests'][2]['top_logprobs'].__setitem__(0, []), 'top logprobs'),
                (lambda v: v['requests'][3]['token_ids'].pop(), 'usage'),
                (lambda v: v['requests'][3]['tokens'].pop(), 'incomplete logprobs'),
                (lambda v: v['requests'][3].update(finish_reason='error'), 'finish'),
                (lambda v: v.update(sampling=dict(v['sampling'], max_tokens=64)), 'workload'),
                (lambda v: v.update(concurrency=4), 'workload'),
                (lambda v: v['image'].update(included=False), 'request set'),
                (lambda v: v.update(passed=False), 'passing'),
                (lambda v: v['cold_repeat'].update(supported=False), 'cold long repeat'),
                (lambda v: v['prefix_hit'].update(anchors=[1024]), 'missing prefix hits'),
                (lambda v: row(v, 'long-hit-1024')['usage']['prompt_tokens_details'].update(cached_tokens=0),
                 'long-hit-1024: not a prefix hit'),
                (lambda v: row(v, 'long-hit-1024')['usage']['prompt_tokens_details'].update(cached_tokens=1088),
                 'long-hit-1024: not a prefix hit'),
                (lambda v: row(v, 'long-primer-2000')['usage']['prompt_tokens_details'].update(cached_tokens=64),
                 'long-primer-2000: not cold'),
                (lambda v: row(v, 'long-ids')['usage']['prompt_tokens_details'].update(cached_tokens=1024),
                 'long-ids: not cold'),
                (lambda v: row(v, 'long-hit-2000').update(cache_salt='other'), 'cache salt'),
                (lambda v: (row(v, 'long-primer-1024')['token_ids'].append(1),
                            row(v, 'long-primer-1024')['usage'].update(completion_tokens=2)), 'usage'),
                (lambda v: row(v, 'long-primer-1024')['usage'].update(prompt_tokens=1025), 'number of tokens'),
                (lambda v: row(v, 'long-ids')['usage'].update(prompt_tokens=2801), 'number of tokens'),
                (lambda v: v['request_set'].update(long_prompt_tokens=None), 'long prompt token IDs')):
            broken = copy.deepcopy(good)
            mutation(broken)
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                determinism.check(broken)


class CompareTests(unittest.TestCase):
    def test_identical_records_pass(self):
        left, right = recorded(arm='spec'), recorded(arm='spec-repeat')
        result = determinism.compare(left, right)
        self.assertTrue(result['passed'])
        self.assertIsNone(result['first_difference'])
        self.assertEqual(result['max_abs_difference'], 0.0)
        self.assertEqual((result['requests'], result['tokens']), (12, 32))
        self.assertTrue(result['prefix_hit']['left']['equal'] and result['prefix_hit']['right']['equal'])

    def test_one_ulp_selected_logprob_difference_fails(self):
        left, right = recorded(arm='spec'), recorded(arm='ar')
        value = right['requests'][2]['logprobs'][1]
        right['requests'][2]['logprobs'][1] = math.nextafter(value, 0.0)
        result = determinism.compare(left, right)
        self.assertFalse(result['passed'])
        self.assertEqual({k: result['first_difference'][k] for k in ('request', 'field', 'position')},
                         {'request': 'fibonacci', 'field': 'logprobs', 'position': 1})
        self.assertEqual(result['max_abs_difference'], abs(math.nextafter(value, 0.0) - value))
        self.assertGreater(result['max_abs_difference'], 0.0)
        self.assertIn('spec and ar differ at fibonacci logprobs position 1', determinism.describe(result))

    def test_one_ulp_top_logprob_and_signed_zero_differences_fail(self):
        left = recorded()
        right = copy.deepcopy(left)
        entry = right['requests'][6]['top_logprobs'][2][1]
        entry[1] = math.nextafter(entry[1], -math.inf)
        result = determinism.compare(left, right)
        self.assertEqual((result['first_difference']['request'], result['first_difference']['field'],
                          result['first_difference']['position']), ('image', 'top_logprobs', 2))
        left['requests'][0]['logprobs'][0] = 0.0
        right = copy.deepcopy(left)
        right['requests'][0]['logprobs'][0] = -0.0
        result = determinism.compare(left, right)
        self.assertEqual((result['first_difference']['field'], result['max_abs_difference']), ('logprobs', 0.0))
        self.assertFalse(result['passed'])

    def test_token_text_and_finish_differences_fail_at_the_first_position(self):
        left = recorded()
        for mutation, field, position in (
                (lambda row: row['token_ids'].__setitem__(2, 7), 'token_ids', 2),
                (lambda row: row.update(text=row['text'] + ' '), 'text', None),
                (lambda row: row.update(finish_reason='stop'), 'finish_reason', None),
                (lambda row: (row['token_ids'].append(1), row['usage'].update(completion_tokens=4)), 'token_ids', 3)):
            right = copy.deepcopy(left)
            mutation(right['requests'][1])
            result = determinism.compare(left, right)
            with self.subTest(field=field):
                self.assertFalse(result['passed'])
                self.assertEqual(result['first_difference']['request'], 'haiku')
                self.assertEqual((result['first_difference']['field'], result['first_difference']['position']),
                                 (field, position))

    def test_cold_long_sends_must_agree_within_each_record(self):
        left = recorded()
        left['requests'][5]['logprobs'][0] = math.nextafter(left['requests'][5]['logprobs'][0], 0.0)
        right = copy.deepcopy(left)
        result = determinism.compare(left, right)
        self.assertIsNone(result['first_difference'])
        self.assertFalse(result['passed'])
        self.assertFalse(result['cold_repeat']['left']['equal'])
        self.assertEqual(result['cold_repeat']['left']['first_difference']['position'], 0)
        self.assertIn('the two cold long sends differ at logprobs position 0', determinism.describe(result))

    def test_prefix_hits_must_equal_the_cold_token_id_send_within_each_record(self):
        for name in ('long-hit-1024', 'long-hit-2000'):
            left = recorded()
            hit = row(left, name)
            hit['logprobs'][2] = math.nextafter(hit['logprobs'][2], 0.0)
            right = copy.deepcopy(left)
            result = determinism.compare(left, right)
            with self.subTest(name=name):
                self.assertIsNone(result['first_difference'])
                self.assertTrue(result['cold_repeat']['left']['equal'])
                self.assertFalse(result['passed'])
                self.assertFalse(result['prefix_hit']['right']['equal'])
                self.assertEqual({k: result['prefix_hit']['left']['first_difference'][k]
                                  for k in ('request', 'field', 'position')},
                                 {'request': name, 'field': 'logprobs', 'position': 2})
                self.assertIn(f'spec: {name} differs from the cold long-ids send at logprobs position 2',
                              determinism.describe(result))
        # A hit that differs on one side only fails the cross-record comparison too.
        left, right = recorded(), recorded()
        row(right, 'long-hit-1024')['token_ids'][1] = 7
        result = determinism.compare(left, right)
        self.assertFalse(result['passed'])
        self.assertEqual((result['first_difference']['request'], result['first_difference']['field']),
                         ('long-hit-1024', 'token_ids'))
        self.assertTrue(result['prefix_hit']['left']['equal'])
        self.assertFalse(result['prefix_hit']['right']['equal'])

    def test_different_request_sets_are_refused(self):
        with self.assertRaisesRegex(ValueError, 'different request sets'):
            determinism.compare(recorded(image=True), recorded(image=False))
        left, right = recorded(), recorded()
        right['request_set'] = dict(right['request_set'], sha256='other')
        with self.assertRaisesRegex(ValueError, 'different request sets'):
            determinism.compare(left, right)

    def test_cli_exit_status_follows_the_comparison(self):
        left = recorded()
        for change, code in ((None, 0), (1, 1)):
            right = copy.deepcopy(left)
            if change:
                right['requests'][0]['logprobs'][0] = math.nextafter(right['requests'][0]['logprobs'][0], 0.0)
            with tempfile.TemporaryDirectory() as tmp:
                paths = [Path(tmp) / name for name in ('left.json', 'right.json')]
                for path, value in zip(paths, (left, right)):
                    path.write_text(json.dumps(value))
                output = io.StringIO()
                with contextlib.redirect_stdout(output), contextlib.redirect_stderr(io.StringIO()), \
                        self.assertRaises(SystemExit) as stopped:
                    determinism.compare_command(SimpleNamespace(left=str(paths[0]), right=str(paths[1])))
                self.assertEqual(stopped.exception.code, code)
                self.assertEqual(json.loads(output.getvalue())['passed'], code == 0)


if __name__ == '__main__':
    unittest.main()
