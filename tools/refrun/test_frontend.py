"""The serving transport is translated into HF template inputs explicitly."""
import copy
from pathlib import Path
import unittest

from tools.refrun.frontend import messages_for_hf


class FrontendInputs(unittest.TestCase):
    def test_reasoning_alias_and_typed_tool_arguments_preserve_the_input(self):
        messages = [{"role": "assistant", "content": "", "reasoning": "kept", "reasoning_content": "fallback",
                     "tool_calls": [{"function": {"name": "f", "arguments": '{"count": 3, "enabled": true}'}}]}]
        previous = copy.deepcopy(messages)
        result = messages_for_hf(messages, tool_reasoning=True)
        self.assertEqual(result[0]["reasoning_content"], "kept")
        self.assertEqual(result[0]["tool_calls"][0]["function"]["arguments"], {"count": 3, "enabled": True})
        self.assertEqual(messages, previous)

    def test_reasoning_content_fallback_is_limited_to_tool_turns(self):
        ordinary = {"role": "assistant", "content": "answer", "reasoning_content": "r"}
        tool = {**ordinary, "tool_calls": [{"function": {"name": "f", "arguments": {}}}]}
        result = messages_for_hf([ordinary, tool], tool_reasoning=True)
        self.assertNotIn("reasoning_content", result[0])
        self.assertEqual(result[1]["reasoning_content"], "r")
        self.assertNotIn("reasoning_content", messages_for_hf([tool])[0])

    def test_images_must_resolve_to_the_pinned_bytes(self):
        messages = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "sha256:abc"}}]}]
        self.assertEqual(messages_for_hf(messages, images={"abc": Path("image.png")})[0]["content"],
                         [{"type": "image", "image": "image.png"}])
        with self.assertRaises(KeyError):
            messages_for_hf(messages, images={})


if __name__ == "__main__":
    unittest.main()
