"""Answer-only probe requests follow the frontend's declared thinking policy."""

from qual_test_config import setUpModule

from contextlib import contextmanager
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import color_probe
import grounding
import logprobs_parity
import multimodal
import prefix_cache

# Audited to run once rather than under every gate config (ci/qualification.py).
QUALIFICATION_RUNS_ONCE = "probe request bodies against synthetic frontend declarations"

# Where thinking can be disabled, an answer-only request sends the switch off;
# a frontend that refuses that gets no switch and a reasoning budget.
POLICIES = ((True, {"ponder": False}), (False, {}))
# A declaration states its reasoning close in `format.parser`.
FORMAT = {"parser": {"reasoning_close": "</think>"}}


@contextmanager
def frontend(allowed):
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        declared = root / "frontend.json"
        reasoning = {"thinking_keywords": ["ponder"]} | ({} if allowed else {"allow_disabled": False})
        declared.write_text(json.dumps({"reasoning": reasoning, "format": FORMAT}))
        config = root / "gate.json"
        config.write_text(json.dumps({"receipts": str(root / "receipts"), "inputs": {"frontend": str(declared)}}))
        with patch.dict(os.environ, {"CHOPINFER_QUAL_CONFIG": str(config)}):
            yield


class Sent(Exception):
    """Stops a probe after its first request."""


class Recorder:
    def __init__(self, reply=None):
        self.reply, self.bodies = reply, []

    def request(self, path, body, stream=False):
        self.bodies.append(body)
        if self.reply is None:
            raise Sent()
        return self.reply, 0.1


def entries(tokens):
    return [{"token": token, "logprob": -0.5,
             "top_logprobs": [{"token": token, "logprob": -0.5}, {"token": "blue", "logprob": -2.0}]}
            for token in tokens]


class AnswerRequests(unittest.TestCase):
    def test_color_probe_scores_the_first_answer_token(self):
        for allowed, kwargs in POLICIES:
            with self.subTest(allowed=allowed), frontend(allowed):
                # A reasoning frontend's logprobs lead with the reasoning.
                tokens = ["Red"] if allowed else ["Think", " hard", "</think>", "\n\n", "Red"]
                client = Recorder({"choices": [{"message": {"content": "Red"},
                                                "logprobs": {"content": entries(tokens)}}],
                                   "usage": {"prompt_tokens": 3}})
                row = color_probe.ask(client, "m", [])
                self.assertEqual(client.bodies[0]["chat_template_kwargs"], kwargs)
                self.assertEqual(client.bodies[0]["max_tokens"], 8 if allowed else 4096)
                self.assertEqual(row["tokens"], ["Red"])
                self.assertEqual(row["top_color"], "red")
        with frontend(False):
            client = Recorder({"choices": [{"message": {"content": ""},
                                            "logprobs": {"content": entries(["Think"])}}],
                               "usage": {"prompt_tokens": 3}})
            with self.assertRaisesRegex(ValueError, "no answer after the reasoning"):
                color_probe.ask(client, "m", [])

    def test_grounding_answers_follow_the_policy(self):
        for allowed, kwargs in POLICIES:
            with self.subTest(allowed=allowed), frontend(allowed):
                client = Recorder({"choices": [{"message": {"content": "red"}}], "usage": {"prompt_tokens": 3}})
                probe = grounding.Probe(client, "m")
                probe.answer(grounding.user("?"), 16)
                self.assertEqual(client.bodies[0]["chat_template_kwargs"], kwargs)
                self.assertEqual(client.bodies[0]["max_tokens"], 16 if allowed else 4096)
                self.assertEqual(probe.body([])["chat_template_kwargs"], kwargs)

    def test_logprobs_parity_answer_cases_follow_the_policy(self):
        for allowed, kwargs in POLICIES:
            with self.subTest(allowed=allowed), frontend(allowed):
                cases = {name: body for name, endpoint, body in logprobs_parity.cases() if endpoint == "chat"}
                self.assertNotIn("chat_template_kwargs", cases.pop("chat-thinking"))
                self.assertNotIn("chat_template_kwargs", cases.pop("chat-hidden-reasoning"))
                for name, body in cases.items():
                    self.assertEqual(body["chat_template_kwargs"], kwargs, name)

    def test_prefix_cache_requests_follow_the_policy(self):
        events = [(0.1, {"choices": [{"delta": {"content": "a"},
                                      "logprobs": {"content": [{"token": "a", "logprob": -0.1}]}}]}),
                  (0.2, {"choices": [], "usage": {"prompt_tokens": 3}})]
        for allowed, kwargs in POLICIES:
            with self.subTest(allowed=allowed), frontend(allowed):
                client = Recorder(events)
                row = prefix_cache.stream_chat(client, "m", [], 4)
                self.assertEqual(client.bodies[0]["chat_template_kwargs"], kwargs)
                self.assertEqual(client.bodies[0]["max_tokens"], 4)
                self.assertEqual(row["tokens"], ["a"])

    def test_multimodal_cases_follow_the_policy(self):
        case = {"id": "image-x", "kind": "image", "file": "x.png", "prompt": "?"}
        for allowed, kwargs in POLICIES:
            with self.subTest(allowed=allowed), frontend(allowed), \
                    patch("multimodal.messages", return_value=[]), self.assertRaises(Sent):
                client = Recorder()
                try:
                    multimodal.evaluate_case(client, "m", case, {}, Path("/unused"), {})
                finally:
                    self.assertEqual(client.bodies[0]["chat_template_kwargs"], kwargs)
                    self.assertEqual(client.bodies[0]["max_tokens"], 256 if allowed else 4096)

    def test_multimodal_probes_follow_the_declared_media_thinking(self):
        for thinking in (False, True):
            for allowed, kwargs in POLICIES:
                with self.subTest(thinking=thinking, allowed=allowed), frontend(allowed), \
                        patch("multimodal.media_policy", return_value={"thinking": thinking, "expected": {}}), \
                        patch("multimodal.probe_messages", return_value=[]), self.assertRaises(Sent):
                    client = Recorder()
                    try:
                        multimodal.evaluate_probe(client, "m", {}, {}, {"answer": {"token_ids": []}}, 0.0)
                    finally:
                        body = client.bodies[0]
                        self.assertEqual(body["chat_template_kwargs"], {"ponder": True} if thinking else kwargs)
                        self.assertEqual(body["max_tokens"], 64 if thinking or allowed else 4096)


if __name__ == "__main__":
    unittest.main()
