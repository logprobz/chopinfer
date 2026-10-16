"""Uniform full-answer admission, before any engine scores are inspected."""
import unittest
from types import SimpleNamespace
from tools.refrun import mixed_answers as answers

class Tokens:
    def decode(self, ids, **kwargs):
        return " ".join(map(str, ids))
    def encode(self, text, **kwargs):
        return SimpleNamespace(ids=list(map(int, text.split())))

class AnswerTests(unittest.TestCase):
    def entry(self, ids):
        return {"scores": {"answer": ids}, "key_sha256": "a" * 64}

    def test_full_answer_keeps_every_text_position(self):
        ids = list(range(1, 97))
        got = answers.full_answer(self.entry([*ids, 999]), answers.protocol([999]), Tokens())
        self.assertEqual(got["answer_token_ids"], ids)
        self.assertEqual(got["terminal_stop_id"], 999)

    def test_short_or_truncated_answer_is_refused(self):
        for ids, message in (([1] * 31 + [999], "fewer than 32"), ([1] * 256, "truncated")):
            with self.subTest(ids=len(ids)), self.assertRaisesRegex(ValueError, message):
                answers.full_answer(self.entry(ids), answers.protocol([999]), Tokens())
