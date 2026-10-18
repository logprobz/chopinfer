"""Request transport regressions against each model's declared frontend."""

from qual_test_config import configure_receipts as setUpModule

from concurrent.futures import Future
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import color_probe
import common
import determinism
import evals
import frontend_goldens
import grounding
import gsm8k
import logprobs_parity
import multimodal
import prefix_cache
import prefix_exact
import soak
import spec_bench
import structured
from dev import encoder_parallel
from test_api import capture as capture_api
from test_gsm8k import examples


# SHA-256 of all 43 request bodies serialized by json.dumps before the
# transport change, including field order. The template frontends differ
# here only in whether they allow thinking to be disabled.
LEGACY_BODIES = {
    False: "cd45bf0aefdbfd9ebcef690704e7e884842b9f031103a7bda2fe5b9aa40e23f2",
    True: "054e8ae5c49e6b6003fc0e8a315b88bfc50307245b5896a72f18be0909303544",
}


@contextmanager
def declared(config):
    with patch.dict(os.environ, {"CHOPINFER_QUAL_CONFIG": str(config)}):
        yield common.frontend()


class Sent(Exception):
    """Stop a probe immediately after recording its request."""


def first_request(call):
    client = Mock()
    client.request.side_effect = Sent()
    try:
        call(client)
    except Sent:
        pass
    assert client.request.call_count == 1
    return client.request.call_args.args[1]


class InlinePool:
    """Run a bounded soak worker synchronously, with its real request builder."""

    def __init__(self, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def submit(self, function, *args):
        future = Future()
        future.set_result(function(*args))
        return future


def capture_soak():
    bodies = []

    def request(path, body):
        bodies.append(body)
        return {"choices": [{"message": {"content": "Blue."}, "finish_reason": "stop"}]}, 0.1

    client = Mock()
    client.identity.return_value = {"model": "m"}
    client.request.side_effect = request
    with patch("sys.argv", ["soak", "--remote-only", "--media-mix", "--structured-mix", "--duration", "1",
                            "--concurrency", "1", "--media-cases", "image-blue-square.png"]), \
            patch.object(soak, "Client", return_value=client), \
            patch.object(soak, "ThreadPoolExecutor", InlinePool), \
            patch.object(soak, "media_requests", return_value=[("image-blue-square.png", [])]), \
            patch.object(soak, "not_applicable", return_value=None), \
            patch.object(soak, "memory_policy", return_value=(0, {})), \
            patch.object(soak, "host_load_sample", return_value={}), \
            patch.object(soak.time, "monotonic", side_effect=lambda: 2 if len(bodies) >= 8 else 0), \
            patch.object(soak, "receipt"):
        try:
            soak.main()
        except SystemExit:
            pass  # The synthetic structured answer is intentionally not scored here.
    assert len(bodies) == 8
    return [bodies[0], bodies[4]]


def capture_gsm8k():
    client = Mock()
    client.identity.return_value = {"model": "m"}
    client.request.return_value = ({"choices": [{"message": {"content": "#### 1"}, "finish_reason": "stop"}]}, 0.1)
    with tempfile.TemporaryDirectory() as directory, \
            patch.object(gsm8k, "FIXTURE", examples(Path(directory))), \
            patch("sys.argv", ["gsm8k", "--mechanics-only", "--limit", "1"]), \
            patch.object(gsm8k, "screen_floor", return_value=1), \
            patch.object(gsm8k, "Client", return_value=client), \
            patch.object(gsm8k, "confirm_live"), patch.object(gsm8k, "receipt"):
        try:
            gsm8k.main()
        except SystemExit as error:
            assert error.code == 0
    assert client.request.call_count == 1
    return client.request.call_args.args[1]


def requests():
    """Every changed request builder, including both enabled and answer modes."""
    rows = {}
    for enabled in (False, True):
        rows[f"evals/{enabled}"] = evals.chat([], 32, enabled)
        with patch.object(multimodal, "media_policy", return_value={"thinking": enabled}), \
                patch.object(multimodal, "probe_messages", return_value=[]):
            rows[f"multimodal/probe/{enabled}"] = first_request(lambda client:
                multimodal.evaluate_probe(client, "m", {}, {}, {"answer": {"token_ids": []}}, 0.3))
    with patch.object(multimodal, "messages", return_value=[]):
        rows["multimodal/case"] = first_request(lambda client:
            multimodal.evaluate_case(client, "m", {}, {}, Path("."), {}))
    rows["color_probe"] = first_request(lambda client: color_probe.ask(client, "m", []))
    rows["grounding"] = grounding.Probe(None, "m").body([])
    rows["prefix_cache"] = first_request(lambda client: prefix_cache.stream_chat(client, "m", [], 8))
    rows["prefix_exact/chat"] = first_request(lambda client: prefix_exact.chat(client, "m", [], "salt"))
    rows["prefix_exact/tokenize"] = first_request(lambda client: prefix_exact.tokenize(client, "m", []))
    rows["spec_bench"] = spec_bench.workload("m", "run", "chat", 1, 0, "Question?", "Answer.", 32)
    rows["determinism"] = next(body for name, _, body in determinism.request_set(True, common.answer_control())
                               if name == determinism.IMAGE)
    for name, _, body in logprobs_parity.cases():
        rows[f"logprobs_parity/{name}"] = body
    for name, (_, body, _) in structured.cases(SimpleNamespace(reasoning_max_tokens=4096)).items():
        rows[f"structured/{name}"] = body
    for index, body in enumerate(capture_api()[0]):
        rows[f"api/{index}"] = body
    for index, body in enumerate(capture_soak()):
        rows[f"soak/{index}"] = body
    rows["gsm8k"] = capture_gsm8k()
    rows["encoder_parallel"] = encoder_parallel.chat_body("m", [])
    renders = [{"messages": [], "chat_template_kwargs": {"enable_thinking": enabled, "preserve_thinking": True}}
               for enabled in (False, True)]
    for index, body in enumerate(frontend_goldens.capture_bodies("m", renders, [])[0]):
        rows[f"frontend_goldens/{index}"] = body
    return rows


class ThinkingControls(unittest.TestCase):
    def test_legacy_helpers_and_unsupported_declarations(self):
        for switch in (None, "template-keywords", "thinking-object", "unsupported", 42):
            reasoning = {"thinking_keywords": ["ponder"]}
            if switch is not None:
                reasoning["switch"] = switch
            with self.subTest(switch=switch), patch.object(common, "frontend", return_value={"reasoning": reasoning}):
                self.assertEqual(common.thinking_kwargs(True), {"ponder": True})
                self.assertEqual(common.answer_kwargs(), {"ponder": False})
                if switch in ("unsupported", 42):
                    for call in (lambda: common.thinking_control(True), common.answer_control,
                                 lambda: common.template_control({"ponder": False})):
                        with self.assertRaisesRegex(ValueError, "unsupported thinking switch"):
                            call()
                    continue
                for enabled in (False, True):
                    expected = ({"thinking": {"type": "enabled" if enabled else "disabled"}}
                                if switch == "thinking-object" else {"chat_template_kwargs": {"ponder": enabled}})
                    self.assertEqual(common.thinking_control(enabled), expected)
                reasoning["allow_disabled"] = False
                self.assertEqual(common.answer_control(), {} if switch == "thinking-object" else {"chat_template_kwargs": {}})

    def test_reasoning_field_defaults_and_refusals(self):
        for response, expected in (({}, "reasoning"), ({"reasoning_field": "reasoning"}, "reasoning"),
                                   ({"reasoning_field": "reasoning_content"}, "reasoning_content")):
            with patch.object(common, "frontend", return_value={"response": response}):
                self.assertEqual(common.reasoning_field(), expected)
        for field in (None, "private", False):
            with patch.object(common, "frontend", return_value={"response": {"reasoning_field": field}}):
                with self.assertRaisesRegex(ValueError, "unsupported reasoning response field"):
                    common.reasoning_field()

    def test_declared_frontends_route_every_request_builder(self):
        for config in sorted((common.ROOT / "models").glob("*/qual.json")):
            with self.subTest(model=config.parent.name), declared(config) as frontend:
                rows = requests()
                if frontend["reasoning"].get("switch") != "thinking-object":
                    self.assertEqual(len(rows), 43)
                    digest = hashlib.sha256(json.dumps(rows).encode()).hexdigest()
                    self.assertEqual(digest, LEGACY_BODIES[frontend["reasoning"].get("allow_disabled", True)])
                    continue
                unchanged = {"logprobs_parity/chat-thinking", "logprobs_parity/chat-hidden-reasoning"}
                for name, body in rows.items():
                    if "messages" not in body or name in unchanged:
                        continue
                    self.assertIn("thinking", body, name)
                    self.assertFalse(set(frontend["reasoning"]["thinking_keywords"]) &
                                     body.get("chat_template_kwargs", {}).keys(), name)
                    enabled = (name.endswith("/True") or name in ("spec_bench", "gsm8k", "structured/reasoning_json_schema",
                                                                  "api/0", "api/1", "frontend_goldens/1"))
                    self.assertEqual(body["thinking"], {"type": "enabled" if enabled else "disabled"}, name)

    def test_captured_controls_preserve_options_and_promote_effort(self):
        frontend = {"reasoning": {"switch": "thinking-object", "thinking_keywords": ["ponder", "think"]}}
        with patch.object(common, "frontend", return_value=frontend):
            kwargs = {"ponder": False, "think": True, "reasoning_effort": "low", "preserve_thinking": True}
            self.assertEqual(common.template_control(kwargs), {
                "chat_template_kwargs": {"preserve_thinking": True}, "thinking": {"type": "disabled"},
                "reasoning_effort": "low"})
            self.assertEqual(len(kwargs), 4)

    def test_object_frontend_captured_rows_use_vendor_controls(self):
        for config in sorted((common.ROOT / "models").glob("*/qual.json")):
            with declared(config) as frontend:
                if frontend["reasoning"].get("switch") != "thinking-object":
                    continue
                renders = json.loads((common.ROOT / "fixtures" / config.parent.name / "frontend/chat-renders.json").read_text())
                bodies, _ = frontend_goldens.capture_bodies("m", renders, [])
                changed = 0
                for row, body in zip(renders, bodies):
                    kwargs = row.get("chat_template_kwargs", {})
                    if "enable_thinking" in kwargs:
                        changed += 1
                        self.assertEqual(body["thinking"], {"type": "enabled" if kwargs["enable_thinking"] else "disabled"})
                        self.assertNotIn("chat_template_kwargs", body)
                self.assertGreater(changed, 0)

    def test_stream_first_token_accepts_the_declared_reasoning_field(self):
        for field in ("reasoning", "reasoning_content"):
            with patch.object(common, "frontend", return_value={"reasoning": {"thinking_keywords": ["ponder"]},
                                                               "response": {"reasoning_field": field}}), \
                    patch.object(prefix_cache.time, "monotonic", return_value=10):
                client = Mock()
                client.request.return_value = ([(11, {"choices": [{"delta": {field: "step"}}]}),
                                                (12, {"choices": [], "usage": {"prompt_tokens": 3}})], 2)
                self.assertEqual(prefix_cache.stream_chat(client, "m", [], 8)["ttft_seconds"], 1)


if __name__ == "__main__":
    unittest.main()
