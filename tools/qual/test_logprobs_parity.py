"""Offline checks for logprob capture attribution and comparison."""

from qual_test_config import setUpModule

import unittest
import copy
import contextlib
import io
import subprocess
import sys
import tempfile
from unittest.mock import patch

import release_gate as gate
import common
import declaration
import logprobs_parity
from logprobs_parity import cases
import json
from pathlib import Path

from logprobs_parity import compare_case, summarize

# Audited to run once rather than under every gate config (ci/qualification.py).
QUALIFICATION_RUNS_ONCE = "logprobs parity comparison on synthetic captures"
LOGPROB_MODELS = ("mimo26", "qwen38")


def chat_row(labels, streamed, tokens=None):
    content = [{"token": label, "logprob": -0.5, "top_logprobs": [{"token": label, "logprob": -0.5}]}
               for label in labels]
    chunks = [{"choices": [{"delta": {"content": "x"},
                            "logprobs": {"content": [{"token": label, "logprob": -0.5, "top_logprobs": []}
                                                     for label in group]}}]}
              for group in streamed]
    return {"endpoint": "chat", "stream": chunks, "stream_ids": chunks,
            "json": {"usage": {"completion_tokens": tokens or len(labels)},
                     "choices": [{"message": {"content": "x"},
                                  "logprobs": {"content": content}}]}}


def hidden_row():
    request = next(body for name, endpoint, body in cases() if body.get('include_reasoning') is False)
    choice = {'index': 0, 'message': {'role': 'assistant', 'content': 'answer'},
              'finish_reason': 'stop', 'logprobs': None}
    stream = [{'object': 'chat.completion.chunk', 'choices': [
        {'index': 0, 'delta': delta, 'finish_reason': finish, 'logprobs': None}]}
        for delta, finish in (({'role': 'assistant', 'content': ''}, None),
                              ({'content': 'answer'}, None), ({}, 'stop'))]
    ids = copy.deepcopy(stream); ids[0]['prompt_token_ids'] = [1, 2]
    return {'endpoint': 'chat', 'request': request, 'stream': stream, 'stream_ids': ids,
            'json': {'object': 'chat.completion', 'choices': [choice],
                     'usage': {'prompt_tokens': 2, 'completion_tokens': 1, 'total_tokens': 3}}}


def valid_capture():
    capture = {"arm": "fixture", "passed": True, "errors": [], "cases": []}
    for name, endpoint, request in cases():
        if request.get("echo"):
            path = Path(__file__).resolve().parents[2] / "fixtures/server/completion-echo-vllm.json"
            row = json.loads(path.read_text())["cases"][0]
        elif endpoint == 'chat' and request.get('include_reasoning') is False:
            row = hidden_row()
        elif endpoint == "chat":
            row = chat_row(["answer"], [["answer"]])
            top = [{"token": str(i), "logprob": -0.5 - i} for i in range(request["top_logprobs"])]
            row["json"]["choices"][0]["logprobs"]["content"][0]["top_logprobs"] = top
            for mode in ("stream", "stream_ids"):
                row[mode][0]["choices"][0]["logprobs"]["content"][0]["top_logprobs"] = top
        else:
            probs = {"tokens": ["answer"], "token_logprobs": [-0.5], "text_offset": [0],
                     "top_logprobs": [{"answer": -0.5, "other": -1.0}]}
            choice = {"text": "answer", "logprobs": probs}
            row = {"json": {"usage": {"completion_tokens": 1, "prompt_tokens": 2}, "choices": [choice]},
                   "stream": [{"choices": [choice]}], "stream_ids": [{"choices": [choice]}]}
        row.update(case=name, endpoint=endpoint, request=request)
        capture["cases"].append(row)
    return capture


class LogprobsParityTests(unittest.TestCase):
    def capture_fixture(self, *, headers=None, changed_final=None, initial=None, mechanics=False, write_receipt=False):
        pin = common.checkpoint_pin()
        live = {'checkpoint_identity': pin['checkpoint_identity'], 'startup_nonce': 'a' * 32}
        observed = {'requests': [], 'health': 0}
        rows = valid_capture()['cases']

        def response(request, timeout):
            serving = dict(live)
            identity_headers = {common.CHECKPOINT_HEADER: live['checkpoint_identity'],
                                common.NONCE_HEADER: live['startup_nonce']}
            if request.full_url.endswith('/v1/models'):
                value = {'data': [{'id': pin['served_model_name']}]}
            elif request.full_url.endswith('/health'):
                observed['health'] += 1
                serving.update((initial if observed['health'] == 1 else changed_final) or {})
                value = {'status': 'ready', 'serving': serving}
            else:
                index = len(observed['requests'])
                body = json.loads(request.data)
                mode = ('stream_ids' if body.get('return_token_ids') else 'stream') if body.get('stream') else 'json'
                observed['requests'].append((request.full_url, body, mode))
                if headers is not None:
                    headers(index, identity_headers)
                value = rows[index // 3][mode]
                if mode != 'json':
                    data = ''.join('data: ' + json.dumps(chunk) + '\n\n' for chunk in value)
                    value = (data + 'data: [DONE]\n\n').encode()
            result = io.BytesIO(value if isinstance(value, bytes) else json.dumps(value).encode())
            result.headers = identity_headers
            return result

        with patch.object(common.urllib.request, 'urlopen', side_effect=response), \
                patch.object(logprobs_parity, 'receipt', wraps=common.receipt if write_receipt else None) as receipt:
            argv = ['capture', '--arm', 'candidate', '--url', 'http://fixture.invalid:8101',
                    '--model', pin['served_model_name']]
            if mechanics:
                argv.append('--mechanics-only')
            try:
                code = logprobs_parity.main(argv)
            except (SystemExit, ValueError) as error:
                self.assertFalse(receipt.called, 'refused attribution must not emit a passing receipt')
                raise error
            return code, receipt.call_args, observed, live

    def test_capture_binds_all_thirty_responses_and_confirms_final_health(self):
        code, written, observed, live = self.capture_fixture()
        self.assertEqual(code, 0)
        self.assertEqual(observed['health'], 2)
        self.assertEqual(len(observed['requests']), 30)
        self.assertEqual(written.args[1]['checkpoint']['startup_nonce'], live['startup_nonce'])
        value = written.args[2]
        self.assertEqual(value['responders'], [(live['checkpoint_identity'], live['startup_nonce'])])
        self.assertTrue(value['passed'])
        self.assertEqual([row['case'] for row in value['cases']], [name for name, _, _ in cases()])
        for row in value['cases']:
            self.assertEqual(compare_case(summarize(row), summarize(row)), [])

    def test_capture_refuses_wrong_or_missing_response_identity_in_each_mode(self):
        for mode in range(3):
            for key in (common.CHECKPOINT_HEADER, common.NONCE_HEADER):
                for value in (None, 'b' * (64 if key == common.CHECKPOINT_HEADER else 32)):
                    def change(index, headers):
                        if index == mode:
                            if value is None:
                                headers.pop(key)
                            else:
                                headers[key] = value
                    with self.subTest(mode=mode, key=key, value=value), self.assertRaises(SystemExit):
                        self.capture_fixture(headers=change)

    def test_capture_refuses_changed_final_checkpoint_or_nonce(self):
        for key, value in (('checkpoint_identity', 'b' * 64), ('startup_nonce', 'b' * 32)):
            with self.subTest(key=key), self.assertRaisesRegex(SystemExit, 'changed while scoring'):
                self.capture_fixture(changed_final={key: value})

    def test_capture_requires_initial_pin_and_nonce(self):
        for initial in ({'checkpoint_identity': 'b' * 64}, {'startup_nonce': None}):
            with self.subTest(initial=initial), self.assertRaises((SystemExit, ValueError)):
                self.capture_fixture(initial=initial)

    def test_explicit_mechanics_capture_cannot_be_quality_input(self):
        code, written, observed, _ = self.capture_fixture(mechanics=True, initial={'startup_nonce': None})
        self.assertEqual(code, 0)
        self.assertEqual(observed['health'], 1)
        self.assertTrue(written.args[3])
        with self.assertRaisesRegex(ValueError, 'incomplete or failed quality evidence'):
            gate.judge_input('diagnostic', {**written.args[2], 'mechanics_only': written.args[3]}, [], {})

    def test_compare_preserves_diagnostics_through_cli_and_both_shipped_checks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            captures = []
            for diagnostic in (False, True):
                raw = root / ('diagnostic' if diagnostic else 'qualified')
                with patch.dict(common.os.environ, {'CHOPINFER_QUAL_RECEIPTS': str(raw)}), \
                        contextlib.redirect_stdout(io.StringIO()):
                    self.capture_fixture(mechanics=diagnostic, write_receipt=True,
                                         headers=(lambda _, headers: headers.clear()) if diagnostic else None)
                capture = next(raw.glob('*.json'))
                self.assertIs(json.loads(capture.read_text())['mechanics_only'], diagnostic)
                captures.append(capture)
            historical = json.loads(captures[0].read_text())
            historical.pop('mechanics_only')
            historical.pop('responders')
            historical['server'].pop('checkpoint')
            legacy = root / 'historical.json'; legacy.write_text(json.dumps(historical))
            pairs = [(captures[0], captures[0]), (legacy, legacy),
                     (captures[1], captures[0]), (captures[0], captures[1])]
            for index, pair in enumerate(pairs):
                diagnostic = index >= 2
                result = subprocess.run([sys.executable, '-B', str(gate.ROOT / 'tools/qual/logprobs_parity.py'),
                                         'compare', *map(str, pair)], capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                report = json.loads(result.stdout)
                self.assertTrue(report['passed'])
                with self.subTest(pair=index, boundary='CLI'):
                    self.assertIs(report.get('mechanics_only', False), diagnostic)
                for model in LOGPROB_MODELS:
                    path = declaration.path(model); config = declaration.load(path)
                    config['checks'] = [check for check in config['checks'] if check['id'] == 'logprobs_compare']
                    for profile in config['profiles'].values():
                        profile['checks'] = ['logprobs_compare']
                    output = root / f'{model}-{index}'
                    with patch.object(gate.Client, 'identity', return_value={'model': 'fixture'}), \
                            contextlib.redirect_stdout(io.StringIO()):
                        code = gate.run(config, path, 'full', 'http://unused', None,
                                        dict(zip(('stored_logprobs_candidate', 'logprobs_candidate'), map(str, pair))), output)
                    summary = json.loads(next(output.glob('*/summary.json')).read_text())
                    with self.subTest(pair=index, model=model, boundary='runner'):
                        self.assertEqual(len(summary['checks']), 1)
                        row = summary['checks'][0]
                        self.assertEqual(row['exit_code'], 0)
                        self.assertIs(json.loads(Path(row['receipt']).read_text())['mechanics_only'], diagnostic)
                        if diagnostic:
                            self.assertIn('mechanics-only', row['error'])
                        self.assertEqual(code, 1 if diagnostic else 0)
                        self.assertIs(summary['qualified'], not diagnostic)

    def test_hidden_reasoning_requires_explicit_null_scores_and_complete_observable_output(self):
        value = summarize(hidden_row())
        self.assertEqual(compare_case(value, value), [])
        self.assertEqual(value['hidden']['text'], 'answer')
        for key, replacement in [('include_reasoning', True), ('include_reasoning', None),
                                 ('include_reasoning', 0), ('include_reasoning', ''), ('logprobs', False)]:
            bad = hidden_row(); bad['request'][key] = replacement
            with self.subTest(key=key, value=replacement):
                self.assertTrue(compare_case(summarize(bad), summarize(bad)))
        bad = hidden_row(); del bad['request']['include_reasoning']
        self.assertTrue(compare_case(summarize(bad), summarize(bad)))
        bad = hidden_row(); bad['endpoint'] = 'completion'
        self.assertTrue(compare_case(summarize(bad), summarize(bad)))

    def test_hidden_reasoning_refuses_missing_or_exposed_evidence(self):
        changes = [lambda r: r.pop('json'), lambda r: r.update(json={}),
                   lambda r: r['json'].update(choices=[]),
                   lambda r: r['json']['choices'][0].update(logprobs={}),
                   lambda r: r['json']['choices'][0].pop('logprobs'),
                   lambda r: r['json']['choices'][0].update(finish_reason=None),
                   lambda r: r['json']['choices'][0]['message'].update(content=None),
                   lambda r: r['json']['usage'].update(completion_tokens=False),
                   lambda r: r['json']['usage'].pop('prompt_tokens'),
                   lambda r: r['json']['choices'][0]['message'].update(reasoning='hidden'),
                   lambda r: r['stream_ids'][0].pop('prompt_token_ids')]
        for mode in ('stream', 'stream_ids'):
            changes.extend([lambda r, mode=mode: r.pop(mode), lambda r, mode=mode: r.update({mode: []}),
                            lambda r, mode=mode: r[mode].pop(),
                            lambda r, mode=mode: r[mode][1].update(choices=[]),
                            lambda r, mode=mode: r[mode][1]['choices'][0].update(logprobs={'content': []}),
                            lambda r, mode=mode: r[mode][1]['choices'][0]['delta'].update(content='dropped')])
        for index, change in enumerate(changes):
            bad = hidden_row(); change(bad)
            with self.subTest(change=index):
                self.assertTrue(compare_case(summarize(bad), summarize(bad)))

    def test_hidden_reasoning_compares_visible_content_and_finish(self):
        first = hidden_row(); second = hidden_row()
        second['json']['choices'][0]['message']['content'] = 'another answer'
        for mode in ('stream', 'stream_ids'):
            second[mode][1]['choices'][0]['delta']['content'] = 'another answer'
        self.assertTrue(compare_case(summarize(first), summarize(second)))

    def test_cli_and_shipped_gate_fail_closed(self):
        good = valid_capture()
        variants = []
        for score in (None, float("nan"), float("inf")):
            bad = copy.deepcopy(good)
            bad["cases"][0]["json"]["choices"][0]["logprobs"]["content"][0]["logprob"] = score
            variants.append(bad)
        bad = copy.deepcopy(good)
        bad["cases"][0]["json"]["choices"][0]["logprobs"]["content"][0]["top_logprobs"][1]["token"] = "wrong"
        variants.append(bad)
        for mode in ("stream", "stream_ids"):
            bad = copy.deepcopy(good)
            bad["cases"][0][mode] = []
            variants.append(bad)
        bad = copy.deepcopy(good)
        bad["cases"].pop()
        variants.append(bad)
        bad = copy.deepcopy(good)
        bad["cases"][-1] = bad["cases"][0]
        variants.append(bad)
        bad = copy.deepcopy(good)
        bad["cases"][-1]["json"]["choices"][0]["logprobs"]["text_offset"] = []
        variants.append(bad)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = [root / "first.json", root / "second.json"]
            paths[0].write_text(json.dumps(good))
            for index, capture in enumerate([good, *variants]):
                paths[1].write_text(json.dumps(capture))
                result = subprocess.run([sys.executable, "-B", str(gate.ROOT / "tools/qual/logprobs_parity.py"),
                                         "compare", *map(str, paths)], capture_output=True, text=True)
                self.assertEqual(result.returncode, 0 if index == 0 else 1, result.stderr + result.stdout)
                report = json.loads(result.stdout)
                for model in LOGPROB_MODELS:
                    config = declaration.load(declaration.path(model))
                    check = next(c for c in config["checks"] if c["id"] == "logprobs_compare")
                    self.assertEqual(all(gate.judge(report, rule) for rule in check["rules"]), index == 0)
            # Exercise the real runner and worker with one valid and one invalid input.
            config["checks"] = [check]
            for profile in config["profiles"].values():
                profile["checks"] = ["logprobs_compare"]
            for valid in (True, False):
                paths[1].write_text(json.dumps(good if valid else variants[0]))
                output = root / str(valid)
                with patch.object(gate.Client, "identity", return_value={"model": "fixture"}), contextlib.redirect_stdout(io.StringIO()):
                    code = gate.run(config, declaration.path("qwen38"), "full", "http://unused", None,
                                    dict(zip(("stored_logprobs_candidate", "logprobs_candidate"), map(str, paths))), output)
                self.assertEqual(code, 0 if valid else 1)
                summary = json.loads(next(output.glob("*/summary.json")).read_text())
                self.assertEqual(summary["qualified"], valid)

    def test_frozen_echo_and_missing_rows(self):
        path = Path(__file__).resolve().parents[2] / "fixtures/server/completion-echo-vllm.json"
        row = json.loads(path.read_text())["cases"][0]
        summary = summarize(row)
        self.assertEqual(compare_case(summary, summary), [])
        for index in (0, -1):
            broken = copy.deepcopy(row)
            probs = broken["json"]["choices"][0]["logprobs"]
            for key in ("tokens", "token_logprobs", "top_logprobs", "text_offset"):
                probs[key].pop(index)
            self.assertTrue(compare_case(summary, summarize(broken)))
        for mode in ("stream", "stream_ids"):
            broken = copy.deepcopy(row)
            broken[mode] = []
            self.assertTrue(compare_case(summary, summarize(broken)))

    def test_missing_scores_top_tails_and_streams_fail(self):
        good = chat_row(["a"], [["a"]])
        for score in (None, float("nan"), float("inf")):
            bad = copy.deepcopy(good)
            bad["json"]["choices"][0]["logprobs"]["content"][0]["logprob"] = score
            self.assertTrue(compare_case(summarize(good), summarize(bad)))
        for mode in ("stream", "stream_ids"):
            bad = copy.deepcopy(good)
            bad[mode] = []
            self.assertTrue(compare_case(summarize(bad), summarize(bad)))
        a, b = summarize(good), summarize(good)
        a["top"], b["top"] = [["a", "b", "c"]], [["a", "x", "y"]]
        self.assertTrue(compare_case(a, b))
        bad = copy.deepcopy(good)
        bad["json"]["choices"][0]["logprobs"] = None
        self.assertTrue(compare_case(summarize(bad), summarize(bad)))

    def test_parallel_completion_arrays_must_match(self):
        path = Path(__file__).resolve().parents[2] / "fixtures/server/completion-echo-vllm.json"
        row = json.loads(path.read_text())["cases"][0]
        for key in ("token_logprobs", "top_logprobs", "text_offset"):
            bad = copy.deepcopy(row)
            bad["json"]["choices"][0]["logprobs"][key].pop()
            self.assertTrue(compare_case(summarize(bad), summarize(bad)))

    def test_identical_coverage_has_no_notes(self):
        row = chat_row(["<think>", "a", "</think>", "b", "<eos>"], [["a"], ["b"], ["<eos>"]])
        self.assertEqual(compare_case(summarize(row), summarize(row)), [])

    def test_content_only_coverage_is_reported(self):
        full = chat_row(["<think>", "a", "</think>", "b", "<eos>"], [["a"], ["b"], ["<eos>"]])
        partial = chat_row(["b"], [["b"]], tokens=5)
        notes = compare_case(summarize(full), summarize(partial))
        self.assertIn("second: 1 logprob rows for 5 completion tokens", notes)
        self.assertTrue(any("generated tokens differ from row 0" in note for note in notes))

    def test_streamed_drops_are_located(self):
        first = chat_row(["a", "b", "c"], [["a"], ["c"]])
        second = chat_row(["a", "b", "c"], [["a"], ["b"], ["c"]])
        notes = compare_case(summarize(first), summarize(second))
        self.assertTrue(any("first difference at streamed row 1" in note for note in notes))


if __name__ == "__main__":
    unittest.main()
