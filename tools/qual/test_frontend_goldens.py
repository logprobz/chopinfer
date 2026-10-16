"""Declared frontend references and predicate-only worker exits."""

import contextlib
import copy
import http.server
import io
import json
import os
import threading
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import common
import declaration
import frontend_goldens
import kinds
from qual_test_config import fixture_lock


def declared():
    """A copy of a shipped declaration with a frontend_goldens check."""
    return copy.deepcopy(declaration.load(declaration.path('mimo26')))


class FakeServer(http.server.ThreadingHTTPServer):
    """/v1/models, /health and /tokenize: a chat request's IDs are its message
    count and the number of its tools and template arguments, a prompt's its
    character codes; `refuse` names prompts answered with HTTP 400."""

    def __init__(self, refuse=()):
        self.refuse, self.requests = set(refuse), []
        super().__init__(('127.0.0.1', 0), FakeHandler)

    @staticmethod
    def chat_ids(messages, tools, kwargs):
        return [len(messages), len(tools or []), len(kwargs or {})]


class FakeHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def reply(self, status, value):
        data = json.dumps(value).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self.reply(200, {'data': [{'id': 'served'}]} if self.path == '/v1/models' else {'status': 'ok'})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        self.server.requests.append(body)
        if 'prompt' in body:
            if body['prompt'] in self.server.refuse:
                return self.reply(400, {'error': {'message': 'refused'}})
            return self.reply(200, {'tokens': [ord(ch) for ch in body['prompt']]})
        self.reply(200, {'tokens': FakeServer.chat_ids(body['messages'], body.get('tools'),
                                                       body.get('chat_template_kwargs'))})


class ServedCapture(unittest.TestCase):
    """--url captures the served frontend for the declared rows and compares it."""

    def run_served(self, served, rows, texts, *, extra=()):
        config = declared()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference = root / 'reference'
            reference.mkdir()
            (reference / 'chat-renders.json').write_text(json.dumps(rows))
            (reference / 'api-samples.json').write_text('{}')
            golden = root / 'tokenizer-golden.json'
            golden.write_text(json.dumps(texts))
            config['inputs']['frontend_capture'] = str(reference)
            config['inputs']['tokenizer_goldens'] = str(golden)
            thread = threading.Thread(target=served.serve_forever, daemon=True)
            thread.start()
            url = f'http://127.0.0.1:{served.server_address[1]}'
            try:
                with fixture_lock(config), patch.object(frontend_goldens, 'qualification_config', return_value=config), \
                     patch.object(frontend_goldens, 'resolve_input',
                                  side_effect=lambda name, **kwargs: common.resolve_input(name, config=config)), \
                     patch('sys.argv', ['frontend', '--url', url, '--capture-out', str(root / 'capture'),
                                        '--required-file', 'chat-renders.json', '--required-file',
                                        'tokenizer-golden.json', *extra]), \
                     contextlib.redirect_stdout(io.StringIO()) as output:
                    code = frontend_goldens.main()
            finally:
                served.shutdown()
                served.server_close()
            captured = {name: json.loads((root / 'capture' / name).read_text())
                        for name in ('chat-renders.json', 'tokenizer-golden.json', 'capture-meta.json')}
            return code, json.loads(output.getvalue()), captured

    ROWS = [{'id': 'plain', 'messages': [{'role': 'user', 'content': 'hi'}], 'tools': [],
             'chat_template_kwargs': {}},
            {'id': 'tools', 'messages': [
                {'role': 'user', 'content': 'weather?'},
                {'role': 'assistant', 'content': '', 'tool_calls': [{'type': 'function', 'function': {
                    'name': 'get_weather', 'arguments': {'city': 'Paris', 'days': 3}}}]}],
             'tools': [{'type': 'function', 'function': {'name': 'get_weather'}}],
             'chat_template_kwargs': {'reasoning_effort': 'low'}}]

    def reference_rows(self):
        return [{**row, 'token_ids': FakeServer.chat_ids(row['messages'], row['tools'], row['chat_template_kwargs'])}
                for row in self.ROWS]

    def test_an_agreeing_server_matches_and_keeps_its_capture(self):
        served = FakeServer()
        texts = [{'text': 'ab', 'token_ids': [97, 98]}, {'text': '', 'token_ids': []}]
        code, report, captured = self.run_served(served, self.reference_rows(), texts)
        self.assertEqual((code, report['status']), (0, 'match'))
        self.assertEqual(report['coverage']['files'], {'chat-renders.json': 2, 'tokenizer-golden.json': 2})
        self.assertEqual([row['id'] for row in captured['chat-renders.json']], ['plain', 'tools'])
        self.assertEqual(captured['capture-meta.json']['server']['model'], 'served')
        chat = [body for body in served.requests if 'messages' in body]
        self.assertTrue(all(body['add_generation_prompt'] and body['model'] == 'served' for body in chat))
        self.assertNotIn('tools', chat[0])
        call = chat[1]['messages'][1]['tool_calls'][0]['function']
        self.assertEqual(json.loads(call['arguments']), {'city': 'Paris', 'days': 3})
        self.assertEqual(chat[1]['chat_template_kwargs'], {'reasoning_effort': 'low'})
        prompts = [body for body in served.requests if 'prompt' in body]
        self.assertTrue(all(body['add_special_tokens'] is False for body in prompts))

    def test_differing_or_refused_rows_diverge(self):
        rows = self.reference_rows()
        rows[1]['token_ids'] = [9]
        code, report, _ = self.run_served(FakeServer(), rows, [{'text': 'ab', 'token_ids': [97, 98]}])
        self.assertEqual((code, report['status']), (1, 'diverged'))
        self.assertEqual(report['chat-renders.json']['mismatches'], [1])
        code, report, captured = self.run_served(FakeServer(refuse={'ab'}), self.reference_rows(),
                                                 [{'text': 'ab', 'token_ids': [97, 98]}])
        self.assertEqual((code, report['status']), (1, 'diverged'))
        self.assertEqual(captured['tokenizer-golden.json'][0],
                         {'text': 'ab', 'status': 400, 'error': {'error': {'message': 'refused'}}})

    def test_a_capture_directory_and_url_are_exclusive(self):
        config = declared()
        for argv in (['frontend'], ['frontend', 'capture', '--url', 'http://127.0.0.1:1', '--capture-out', 'x'],
                     ['frontend', '--url', 'http://127.0.0.1:1']):
            with self.subTest(argv=argv), fixture_lock(config), \
                    patch.object(frontend_goldens, 'qualification_config', return_value=config), \
                    patch('sys.argv', argv), contextlib.redirect_stderr(io.StringIO()) as output:
                with self.assertRaises(SystemExit) as raised:
                    frontend_goldens.main()
                self.assertEqual(raised.exception.code, 2)
                self.assertIn('give either CAPTURE', output.getvalue())


class BoundCapture(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = declared()
        pin = common.checkpoint_pin(self.config)
        self.model = pin['served_model_name']
        self.live = {'checkpoint_identity': pin['checkpoint_identity'], 'startup_nonce': 'a' * 32,
                     'package_identity': 'b' * 64, 'metadata_identity': 'c' * 64, 'source_commit': 'd' * 40}
        self.server = {'url': 'http://127.0.0.1:12345', 'model': self.model, 'health': {'serving': self.live}}
        self.rows = [{'id': 'chat', 'messages': [{'role': 'user', 'content': 'hi'}], 'token_ids': [1]}]
        self.texts = [{'text': 'hello', 'token_ids': [2]}]
        self.references = {'chat-renders.json': 'e' * 64, 'tokenizer-golden.json': 'f' * 64}
        self.log = self.root / 'server.log'
        self.log.write_text('startup direct_package: ' + json.dumps(self.live) + '\nstartup serving_identity: '
                            + json.dumps(self.live) + '\n{"status":"listening","address":"127.0.0.1:12345"}\n')

    def capture(self, interloper=False, inline=False):
        server = self.server
        live = self.live
        class Client:
            def __init__(self, *args, **kwargs): self.responders = set()
            def identity(self, model=None): return copy.deepcopy(server)
            def request(self, endpoint, body):
                self.responders.add((live['checkpoint_identity'], '9' * 32 if interloper else live['startup_nonce']))
                return {'tokens': [1] if 'messages' in body else [2]}, 0.1
        out = self.root / 'capture'
        with patch.object(common, 'Client', Client), patch.dict(os.environ,
                {'CHOPINFER_FRONTEND_REQUIRE_BOUND': '1' if inline else '0'}):
            frontend_goldens.serve_capture(server['url'], self.model, self.rows, self.texts, out, 10,
                                           server_log=None if inline else self.log, config=self.config, references=self.references)
        return out

    def admit(self, out):
        return frontend_goldens.admit_capture(out, self.config, self.rows, self.texts, self.references)

    def test_full_declared_capture_keeps_all_four_files_and_request_proof(self):
        reference = self.root / 'reference';reference.mkdir()
        (reference / 'chat-renders.json').write_text(json.dumps(self.rows))
        golden = self.root / 'tokenizer-golden.json';golden.write_text(json.dumps(self.texts))
        (reference / 'api-shapes.json').write_text(json.dumps({'refusal': {'status': 400}}))
        self.config['inputs']['frontend_capture'] = str(reference)
        self.config['inputs']['tokenizer_goldens'] = str(golden)
        check = next(c for c in self.config['checks'] if c['module'] == 'frontend_goldens')
        check['rules'] = [{'path': [name, field], 'op': 'eq', 'value': True, 'source': 'test'}
                          for name, field in [('needles.json', 'passed'), ('api-samples.json', 'agrees'),
                                               ('chat-renders.json', 'agrees'), ('tokenizer-golden.json', 'agrees')]]
        plan = {'renders': self.rows, 'identity': {'fixture': 'a' * 64},
                'api': [{'name': 'refusal', 'endpoint': '/refusal', 'body': {'model': self.model}}],
                'needles': [{'name': 'needle', 'total_context': 8, 'expected': 'BLUE',
                             'endpoint': '/needle', 'body': {'model': self.model, 'prompt': [1, 2]}}]}
        server, live = self.server, self.live
        class Client:
            def __init__(self, *args, **kwargs): self.responders = set()
            def identity(self, model=None): return copy.deepcopy(server)
            def request(self, endpoint, body, **kwargs):
                self.responders.add((live['checkpoint_identity'], live['startup_nonce']))
                if endpoint == '/refusal': raise RuntimeError('HTTP 400: {"error":"unsupported"}')
                if endpoint == '/needle': return {'choices': [{'text': 'blue'}]}, 0.1
                return {'tokens': [1] if 'messages' in body else [2]}, 0.1
        out = self.root / 'capture'
        import refs
        scope = refs.scope
        with patch.object(refs, 'scope', side_effect=lambda config, key:
                          ['chat-renders.json', 'api-shapes.json'] if key == 'frontend_capture' else scope(config, key)), \
                fixture_lock(self.config), patch.object(frontend_goldens, 'qualification_config', return_value=self.config), \
                patch.object(frontend_goldens, 'resolve_input', side_effect=lambda name, **kwargs:
                             common.resolve_input(name, config=self.config)), \
                patch.object(frontend_goldens, 'extension', return_value=plan, create=True), \
                patch.object(common, 'Client', Client), \
                patch('sys.argv', ['frontend', '--url', server['url'], '--capture-out', str(out), '--require-bound']), \
                contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(frontend_goldens.main(), 0)
        report = json.loads(output.getvalue())
        self.assertEqual(set(report['coverage']['files']), set(frontend_goldens.FILES))
        metadata = json.loads((out / 'capture-meta.json').read_text())
        self.assertEqual(set(metadata['files']), set(frontend_goldens.FILES))
        references = metadata['references']
        proof = frontend_goldens.admit_capture(out, self.config, self.rows, self.texts, references, plan=plan)
        self.assertEqual(proof['files'], metadata['files'])
        for edit in (lambda m: m['extension']['requests']['api'][0].update(sha256='0' * 64),
                     lambda m: m['extension']['requests']['needles'].clear()):
            changed = copy.deepcopy(metadata);edit(changed)
            (out / 'capture-meta.json').write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, 'extra request workload'):
                frontend_goldens.admit_capture(out, self.config, self.rows, self.texts, references, plan=plan)
        (out / 'capture-meta.json').write_text(json.dumps(metadata))
        for name in ('api-samples.json', 'needles.json'):
            raw = (out / name).read_bytes();(out / name).write_text('{}')
            with self.assertRaisesRegex(ValueError, 'captured bytes changed'):
                frontend_goldens.admit_capture(out, self.config, self.rows, self.texts, references, plan=plan)
            (out / name).write_bytes(raw)

    def test_qualification_worker_cannot_turn_off_required_binding(self):
        kind = kinds.of('frontend_goldens')
        with patch.object(kinds.Kind, 'capture', return_value=0) as capture:
            self.assertEqual(kind.capture(['worker'], {'CHOPINFER_FRONTEND_REQUIRE_BOUND': '0'}, self.root / 'log', 1), 0)
        self.assertEqual(capture.call_args.args[1]['CHOPINFER_FRONTEND_REQUIRE_BOUND'], '1')

    def test_bound_capture_preserves_source_nonce_workload_and_file_hashes(self):
        out = self.capture()
        proof = self.admit(out)
        self.assertEqual(proof['server']['health']['serving'], self.live)
        self.assertTrue(proof['producer_source']['commit'])
        meta = json.loads((out / 'capture-meta.json').read_text())
        self.assertEqual([len(group) for group in meta['request_sha256']], [1, 1])
        self.assertEqual(meta['responders'], [[self.live['checkpoint_identity'], self.live['startup_nonce']]])
        self.assertEqual(set(proof['files']), {'chat-renders.json', 'tokenizer-golden.json'})

    def test_qualified_inline_capture_binds_live_identity_without_a_startup_log(self):
        proof = self.admit(self.capture(inline=True))
        self.assertIsNotNone(proof)
        self.assertEqual(proof['server']['checkpoint']['startup_nonce'], self.live['startup_nonce'])
        self.assertNotIn('server_log', proof['server']['checkpoint'])

    def test_qualified_inline_capture_refuses_a_foreign_responder(self):
        with self.assertRaisesRegex(SystemExit, 'another process'):
            self.capture(interloper=True, inline=True)
        self.assertFalse((self.root / 'capture').exists())

    def test_foreign_responder_refuses_before_publishing_a_capture(self):
        with self.assertRaisesRegex(SystemExit, 'another process'):
            self.capture(interloper=True)
        self.assertFalse((self.root / 'capture').exists())

    def test_changed_capture_request_reference_or_identity_refuses(self):
        out = self.capture(); path = out / 'capture-meta.json'; original = path.read_bytes()
        edits = [lambda m: m['request_sha256'][0].append('0' * 64),
                 lambda m: m.update(references={}),
                 lambda m: m.update(responders=[]),
                 lambda m: m['server']['checkpoint'].update(startup_nonce='0' * 32),
                 lambda m: m.update(producer_source={}),
                 lambda m: m.update(files={})]
        for edit in edits:
            value = json.loads(original); edit(value); path.write_text(json.dumps(value))
            with self.subTest(metadata=value), self.assertRaises(ValueError): self.admit(out)
        path.write_bytes(original)
        (out / 'chat-renders.json').write_text('[]')
        with self.assertRaisesRegex(ValueError, 'bytes changed'): self.admit(out)

    def test_qualification_refuses_a_capture_stripped_of_its_metadata(self):
        reference = self.root / 'reference'; reference.mkdir()
        (reference / 'chat-renders.json').write_text(json.dumps(self.rows))
        golden = self.root / 'tokenizer-golden.json'; golden.write_text(json.dumps(self.texts))
        self.config['inputs']['frontend_capture'] = str(reference)
        self.config['inputs']['tokenizer_goldens'] = str(golden)
        out = self.capture()
        (out / 'capture-meta.json').unlink()
        with fixture_lock(self.config), patch.object(frontend_goldens, 'qualification_config', return_value=self.config), \
                patch.object(frontend_goldens, 'resolve_input', side_effect=lambda name, **kwargs:
                             common.resolve_input(name, config=self.config)), \
                patch('sys.argv', ['frontend', str(out), '--require-bound']), \
                self.assertRaisesRegex(SystemExit, 'requires bound capture metadata'):
            frontend_goldens.main()

    def test_existing_output_is_refused_without_making_requests(self):
        self.capture()
        with patch.object(common, 'Client', side_effect=AssertionError('must not contact server')), \
                self.assertRaisesRegex(ValueError, 'already exists'):
            frontend_goldens.serve_capture(self.server['url'], self.model, self.rows, self.texts,
                                           self.root / 'capture', 10)


class FrontendContracts(unittest.TestCase):
    def test_seeded_choices_keep_the_hf_finish_reason_and_distinct_indexes(self):
        expected = {"completion-n2-seeded": {"status": 200, "finish_reason": "length",
                    "content": True, "reasoning": False, "tool_calls": []}}
        sample = {"completion-n2-seeded": {"status": 200, "body": {"choices": [
            {"index": 0, "text": "first", "finish_reason": "length"},
            {"index": 1, "text": "second", "finish_reason": "length"}]}}}
        self.assertTrue(frontend_goldens.api(sample, expected)["agrees"])
        choices = sample["completion-n2-seeded"]["body"]["choices"]
        choices[1]["finish_reason"] = "stop"
        self.assertFalse(frontend_goldens.api(sample, expected)["agrees"])
        choices[1]["finish_reason"] = "length"
        choices[1]["index"] = 0
        self.assertFalse(frontend_goldens.api(sample, expected)["agrees"])

    def test_declared_tokenizer_and_context_do_not_change_the_gate(self):
        config = declared()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            capture = root / 'capture'
            reference = root / 'reference'
            capture.mkdir()
            reference.mkdir()
            ids = [{'token_ids': [1]}]
            for directory in (capture, reference):
                (directory / 'chat-renders.json').write_text(json.dumps(ids))
                (directory / 'tokenizer-golden.json').write_text(json.dumps(ids))
                (directory / 'api-samples.json').write_text(json.dumps({'example': {'status': 400 if directory == reference else 200, 'body': {}}}))
            tokenizer = root / 'declared-tokenizer.json'
            tokenizer.write_text(json.dumps(ids))
            # The directory's tokenizer is incidental; use the separate declaration.
            (reference / 'tokenizer-golden.json').write_text(json.dumps([{'token_ids': [9]}]))
            config['inputs']['frontend_capture'] = str(reference)
            config['inputs']['tokenizer_goldens'] = str(tokenizer)
            with fixture_lock(config), patch.object(frontend_goldens, 'qualification_config', return_value=config), \
                 patch.object(frontend_goldens, 'resolve_input', side_effect=lambda name, **kwargs: common.resolve_input(name, config=config)), \
                 patch('sys.argv', ['frontend', str(capture)]), contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(frontend_goldens.main(), 0)
            report = json.loads(output.getvalue())
            check = next(check for check in config['checks'] if check['module'] == 'frontend_goldens')
            kind = kinds.of('frontend_goldens')
            with fixture_lock(config):
                pinned = kind.reference(check, config)
            self.assertEqual(kinds.rescore(kind, report, pinned, check)['status'], 'pass')
            self.assertEqual(report['reference']['files']['tokenizer-golden.json']['input'], 'tokenizer_goldens')
            tokenizer.write_text(json.dumps([{'token_ids': [2]}]))
            with fixture_lock(config):
                changed = kind.reference(check, config)
            self.assertEqual(kinds.rescore(kind, report, changed, check)['status'], 'insufficient')

    def test_environment_inputs_and_absolute_override_have_the_same_binding(self):
        config = declared()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ('chat-renders.json', 'tokenizer-golden.json'):
                (root / name).write_text(json.dumps([{'token_ids': [1]}, {'token_ids': [2]}]))
            (root / 'api-samples.json').write_text('{}')
            config['inputs']['frontend_capture'] = {'env': 'QUAL_TEST_GOLDENS', 'default': str(root)}
            config['inputs']['tokenizer_goldens'] = {'env': 'QUAL_TEST_GOLDENS', 'default': str(root),
                                                    'path': 'tokenizer-golden.json'}
            check = next(check for check in config['checks'] if check['module'] == 'frontend_goldens')
            kind = kinds.of('frontend_goldens')
            with fixture_lock(config):
                pinned = kind.reference(check, config)
            for override in ([], ['--reference', str(root.resolve())]):
                with fixture_lock(config), patch.object(frontend_goldens, 'qualification_config', return_value=config), \
                     patch.object(frontend_goldens, 'resolve_input', side_effect=lambda name, **kwargs: common.resolve_input(name, config=config)), \
                     patch('sys.argv', ['frontend', str(root), *override]), \
                     contextlib.redirect_stdout(io.StringIO()) as output:
                    self.assertEqual(frontend_goldens.main(), 0)
                report = json.loads(output.getvalue())
                self.assertEqual(report['reference'], pinned)
                self.assertEqual(kinds.rescore(kind, report, pinned, check)['status'], 'pass')
                report['chat-renders.json'].update(rows=1, reference_rows=1, agrees=True)
                self.assertEqual(kinds.rescore(kind, report, pinned, check)['status'], 'insufficient')

    def test_missing_frontend_check_is_a_usage_refusal(self):
        config = declared()
        config['checks'] = [check for check in config['checks'] if check['module'] != 'frontend_goldens']
        with fixture_lock(config), patch.object(frontend_goldens, 'qualification_config', return_value=config), \
             patch('sys.argv', ['frontend', 'unused']), contextlib.redirect_stderr(io.StringIO()) as output:
            with self.assertRaises(SystemExit) as raised:
                frontend_goldens.main()
        self.assertEqual(raised.exception.code, 2)
        self.assertIn('the declaration has no frontend_goldens check', output.getvalue())

    def test_lock_refusals_exit_without_a_traceback(self):
        import subprocess
        import sys
        from qual_test_config import capture_fixture
        with tempfile.TemporaryDirectory() as temporary:
            capture = Path(temporary)
            (capture / 'chat-renders.json').write_text(json.dumps([{'token_ids': [1]}]))
            (capture / 'tokenizer-golden.json').write_text(json.dumps([{'token_ids': [1]}]))
            for scenario in ('tampered', 'undeclared'):
                with self.subTest(scenario=scenario), capture_fixture(capture) as reference:
                    if scenario == 'tampered':
                        (reference / 'chat-renders.json').write_text('[]')
                        selected = reference
                    else:
                        selected = capture
                    result = subprocess.run([sys.executable, '-B', frontend_goldens.__file__, str(capture),
                                             '--reference', str(selected)], capture_output=True, text=True, check=False)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertTrue(result.stderr.startswith('refused: lock entry frontend_capture:'), result.stderr)
                    self.assertNotIn('Traceback', result.stderr)
                    self.assertNotIn('"status": "diverged"', result.stdout)


    def test_api_shape_lock_refusals_exit_without_a_traceback(self):
        import subprocess
        import sys
        import refs
        from qual_test_config import capture_fixture
        with tempfile.TemporaryDirectory() as temporary:
            capture = Path(temporary)
            for name in ("chat-renders.json", "tokenizer-golden.json"):
                (capture / name).write_text(json.dumps([{"token_ids": [1]}]))
            (capture / "api-samples.json").write_text(json.dumps({"sample": {"status": 400}}))
            with capture_fixture(capture) as reference:
                path = refs.lock_path("synthetic")
                original = json.loads(path.read_text())
                sha = refs.digest(reference / "api-shapes.json")
                mutations = {None: lambda lock: None,
                             "no object schema": lambda lock: lock["objects"][sha].pop("schema"),
                             "missing object": lambda lock: lock["objects"].pop(sha),
                             "unscoped member": lambda lock: lock["entries"]["frontend_capture"].update(
                                 scope=["api-shapes.json"])}
                for mutation, change in mutations.items():
                    with self.subTest(mutation=mutation):
                        lock = copy.deepcopy(original)
                        change(lock)
                        path.write_text(json.dumps(lock))
                        result = subprocess.run([sys.executable, "-B", frontend_goldens.__file__, str(capture)],
                                                capture_output=True, text=True, check=False)
                        if mutation is None:
                            self.assertEqual(result.returncode, 0, result.stderr)
                        else:
                            self.assertNotEqual(result.returncode, 0)
                            self.assertTrue(result.stderr.startswith("refused: "), result.stderr)
                            self.assertNotIn("Traceback", result.stderr)
                            self.assertNotIn('"status": "match"', result.stdout)


    def test_object_records_do_not_change_the_frontend_comparison_in_the_cli(self):
        # A lock pins the capture's bytes; what its object records say
        # produced them neither admits nor refuses the comparison.
        import subprocess
        import sys
        import refs
        from qual_test_config import capture_fixture
        with tempfile.TemporaryDirectory() as temporary:
            capture = Path(temporary)
            for name in ("chat-renders.json", "tokenizer-golden.json"):
                (capture / name).write_text(json.dumps([{"token_ids": [1]}]))
            (capture / "api-samples.json").write_text(json.dumps({"sample": {"status": 400}}))
            with capture_fixture(capture) as reference:
                path = refs.lock_path("synthetic")
                original = json.loads(path.read_text())
                sha = refs.digest(reference / "api-shapes.json")
                candidates = [None, {"producer": "legacy-unknown"}, {"producer": "runner-binding"},
                              {"producer": "runner-layer-reference"}]
                for production in candidates:
                    with self.subTest(production=production):
                        lock = copy.deepcopy(original)
                        if production is not None:
                            lock["objects"][sha]["production"] = production
                        path.write_text(json.dumps(lock))
                        result = subprocess.run([sys.executable, "-B", frontend_goldens.__file__, str(capture)],
                                                capture_output=True, text=True, check=False)
                        self.assertEqual(result.returncode, 0, result.stderr)
                (reference / "api-shapes.json").write_text("{}")
                result = subprocess.run([sys.executable, "-B", frontend_goldens.__file__, str(capture)],
                                        capture_output=True, text=True, check=False)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("frontend_capture: sha256 differs", result.stderr)
                self.assertNotIn("Traceback", result.stderr)
