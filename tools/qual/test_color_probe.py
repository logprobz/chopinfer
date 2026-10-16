"""Offline color probe checks with a synthetic chat server."""

from qual_test_config import setUpModule

import base64
import json
from pathlib import Path
import struct
import tempfile
import unittest
import zlib

from color_probe import compare, jpeg, png, probe_files, run

# Audited to run once rather than under every gate config (ci/qualification.py).
QUALIFICATION_RUNS_ONCE = "color probe scoring against a synthetic chat server"


class FakeClient:
    """Answers "Black" to everything, like MiMo on solid images."""

    def __init__(self):
        self.bodies = []

    def request(self, path, body):
        self.bodies.append(body)
        top = [{"token": word, "logprob": value}
               for word, value in (("Black", -0.2), ("White", -2.0), ("The", -3.0))]
        return {"choices": [{"message": {"content": "Black"},
                             "logprobs": {"content": [{"token": "Black", "logprob": -0.2,
                                                       "top_logprobs": top}]}}],
                "usage": {"prompt_tokens": 80}}, 0.1


class ColorProbeTests(unittest.TestCase):
    def test_png_holds_the_pixel_function(self):
        data = png(3, 2, lambda x, y: (x, y, 7))
        self.assertEqual(struct.unpack(">IIBB", data[16:26]), (3, 2, 8, 2))
        length = struct.unpack(">I", data[33:37])[0]
        self.assertEqual(zlib.decompress(data[41:41 + length]),
                         b"\0\0\0\7\1\0\7\2\0\7" + b"\0\0\1\7\1\1\7\2\1\7")

    def test_jpeg_is_baseline_420(self):
        data = jpeg(40, 24, lambda x, y: (255, 0, 0))
        self.assertEqual((data[:2], data[-2:]), (b"\xff\xd8", b"\xff\xd9"))
        sof = data.index(b"\xff\xc0")
        self.assertEqual(struct.unpack(">BHHB", data[sof + 4:sof + 10]), (8, 24, 40, 3))
        self.assertEqual(data[sof + 10:sof + 19], bytes((1, 0x22, 0, 2, 0x11, 1, 3, 0x11, 1)))

    def test_probe_set_covers_colors_formats_and_splits(self):
        files = [name for name, *_ in probe_files()]
        self.assertEqual(len(files), len(set(files)))
        self.assertIn("solid-yellow-2048x1536.png", files)
        self.assertIn("solid-red-256x256.jpg", files)
        self.assertIn("square-blue-256x256.jpg", files)
        self.assertIn("split-red-blue-left-right-320x192.png", files)

    def test_run_asks_both_orders_and_grades_solid_answers(self):
        client = FakeClient()
        result = run(client, "M")
        self.assertTrue(result["passed"])
        self.assertEqual(len(result["rows"]), 1 + 2 * len(list(probe_files())))
        first, second = (body["messages"][0]["content"] for body in client.bodies[1:3])
        self.assertEqual([part["type"] for part in first], ["image_url", "text"])
        self.assertEqual([part["type"] for part in second], ["text", "image_url"])
        self.assertEqual(first[0], second[1])
        url = first[0]["image_url"]["url"]
        self.assertTrue(base64.b64decode(url.split(",", 1)[1]).startswith(b"\x89PNG"))
        rows = {(row["id"], row["prompt"]): row for row in result["rows"]}
        self.assertTrue(rows[("solid-black-64x64.png", "text-first")]["correct"])
        self.assertFalse(rows[("solid-red-64x64.png", "image-first")]["correct"])
        self.assertIsNone(rows[("split-red-blue-top-bottom-256x256.png", "image-first")]["correct"])
        self.assertEqual(rows[("no-image", "text-only")]["top_color"], "black")
        self.assertEqual(result["summary"]["image-first"]["answers"], ["Black"])

    def test_compare_pairs_rows_of_two_receipts(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for name, answer, score in (("a", "Black", -0.2), ("b", "Red", -0.5)):
                path = Path(directory) / f"{name}.json"
                path.write_text(json.dumps({"server": {"model": name}, "rows": [
                    {"id": "solid-red", "prompt": "image-first", "expected": "red",
                     "answer": answer, "top_color": answer.casefold(),
                     "color_scores": {"red": score}}]}))
                paths.append(path)
            result = compare(*paths)
        row = result["rows"][0]
        self.assertEqual((result["same_answers"], result["total"]), (0, 1))
        self.assertEqual(row["answers"], ["Black", "Red"])
        self.assertAlmostEqual(row["max_abs_color_dlogprob"], 0.3)


if __name__ == "__main__":
    unittest.main()
