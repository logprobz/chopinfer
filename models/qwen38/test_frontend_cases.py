"""Qwen3.8's frontend cases module: how a history turn reads back.

    $CHOPINFER_REFRUN_PYTHON -B -m unittest models/qwen38/test_frontend_cases.py
"""
from pathlib import Path
import unittest

from tools.refrun.frontend_corpus import load_module

CASES = load_module(Path(__file__).resolve().parent / "frontend_cases.py")


class ReadBack(unittest.TestCase):
    def test_a_turn_after_a_chat_prompt_reads_back_without_reasoning(self):
        turn = {"role": "assistant", "content": "Done."}
        chat = {"prompt": "<|im_start|>assistant\n<think>\n\n</think>\n\n", "request": {}}
        self.assertEqual(CASES.expected_turn(turn, chat), {"reasoning": "", "content": "Done.", "tool_calls": []})
        thinking = {"prompt": "<|im_start|>assistant\n<think>\n", "request": {}}
        self.assertEqual(CASES.expected_turn(turn, thinking),
                         {"reasoning": "\n", "content": "\n\nDone.", "tool_calls": []})
        called = dict(turn, tool_calls=[{"function": {"name": "f", "arguments": {"n": 1}}}])
        self.assertEqual(CASES.expected_turn(called, chat)["content"], "Done.\n\n")


if __name__ == "__main__":
    unittest.main()
