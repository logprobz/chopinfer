"""Offline grounding checks with a synthetic chat server."""

from qual_test_config import setUpModule

import base64
import struct
import unittest
import zlib

from grounding import SOLID, Probe, png, run

# Audited to run once rather than under every gate config (ci/qualification.py).
QUALIFICATION_RUNS_ONCE = "grounding scoring against a synthetic chat server"


def row(scores):
    return {str(index): {"logprob": value, "decoded_token": word}
            for index, (word, value) in enumerate(scores.items(), start=100)}


class FakeClient:
    """Answers with the image's color when it sees one; otherwise says blue."""

    def __init__(self, sees_images=True):
        self.sees_images = sees_images

    def color(self, messages):
        content = messages[0]["content"]
        if isinstance(content, str) or not self.sees_images:
            return "blue"
        data = base64.b64decode(content[1]["image_url"]["url"].split(",", 1)[1])
        return next(color for color, rgb in SOLID.items()
                    if data in (png(rgb), png(rgb, square=161)))

    def request(self, path, body):
        word = self.color(body["messages"])
        if not body.get("continue_final_message"):
            return {"choices": [{"message": {"content": word.title()}}],
                    "usage": {"prompt_tokens": 3}}, 0.1
        scores = {"Red": -3.0, "Green": -3.0, "Blue": -3.0, word.title(): -0.1}
        return {"prompt_token_ids": [1, 2, 3, 4],
                "prompt_logprobs": [None, {}, {}, row(scores)]}, 0.1


class GroundingTests(unittest.TestCase):
    def test_png_is_a_solid_rgb_image(self):
        data = png((1, 2, 3), size=4)
        self.assertEqual(data[:8], b"\x89PNG\r\n\x1a\n")
        width, height, depth, kind = struct.unpack(">IIBB", data[16:26])
        self.assertEqual((width, height, depth, kind), (4, 4, 8, 2))
        length = struct.unpack(">I", data[33:37])[0]
        self.assertEqual(zlib.decompress(data[41:41 + length]), (b"\0" + bytes((1, 2, 3)) * 4) * 4)

    def test_square_is_centered_like_the_fixture(self):
        data = png((0, 0, 255), size=8, square=3)
        length = struct.unpack(">I", data[33:37])[0]
        rows = zlib.decompress(data[41:41 + length])
        white, blue = b"\xff\xff\xff", b"\0\0\xff"
        self.assertEqual(rows[:25], b"\0" + white * 8)
        self.assertEqual(rows[3 * 25:4 * 25], b"\0" + white * 3 + blue * 3 + white * 2)

    def test_squares_pass_only_when_the_image_conditions_the_answer(self):
        result = run(Probe(FakeClient(), "M"), None)
        self.assertTrue(result["passed"])
        self.assertEqual([row["id"] for row in result["rows"][2:]],
                         ["square-red", "square-green", "square-blue",
                          "solid-red", "solid-green", "solid-blue"])
        self.assertEqual(result["rows"][4]["gain_over_no_image"], 0.0)
        # A server that ignores images always says blue, so the red and green
        # squares fail. Solid images are recorded, not gated.
        blind = run(Probe(FakeClient(sees_images=False), "M"), None)
        self.assertFalse(blind["passed"])
        self.assertEqual([row["passed"] for row in blind["rows"]],
                         [True, True, False, False, True, True, True, True])
        self.assertEqual([row["correct"] for row in blind["rows"][5:]], [False, False, True])

    def test_a_server_without_image_input_is_not_applicable(self):
        from unittest.mock import patch
        import grounding
        server = {"model": "M", "health": {"status": "ready", "modalities": ["text", "audio"]}}
        with patch("sys.argv", ["grounding.py", "--url", "http://unused"]), \
             patch.object(grounding, "answer_kwargs", return_value={}), \
             patch.object(grounding.Client, "identity", return_value=server), \
             patch.object(grounding.Client, "request", side_effect=AssertionError("sent an image")), \
             patch.object(grounding, "receipt") as receipt:
            with self.assertRaises(SystemExit) as result:
                grounding.main()
        self.assertEqual(result.exception.code, 0)
        self.assertEqual(receipt.call_args.args[2], {
            "passed": False,
            "not_applicable": ["the server does not serve image input (--modalities text,audio)"]})


if __name__ == "__main__":
    unittest.main()
