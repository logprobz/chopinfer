"""The committed image decode corpus against its producer.

The golden must name every file it lists by its bytes, and with the pinned
Pillow the producer must rebuild every file byte for byte, so the corpus
stays reproducible. Decoding is checked by runtime-media's
`tests/image_decode.rs`, not here.
"""

import hashlib
import json
import unittest

from tools.refrun import image_decode


class Corpus(unittest.TestCase):
    def golden(self):
        return json.loads(image_decode.GOLDEN.read_text())

    def test_golden_files_match_their_digests(self):
        cases = self.golden()["cases"]
        self.assertGreaterEqual(len(cases), 100)
        for case in cases:
            data = (image_decode.FIXTURES / case["file"]).read_bytes()
            self.assertEqual(hashlib.sha256(data).hexdigest(), case["file_sha256"], case["name"])
            self.assertEqual("divergence" in case,
                             any("runtime" in case[loader] for loader in ("load_image", "open_convert")),
                             case["name"])

    def test_producer_rebuilds_the_committed_files(self):
        try:
            import PIL
        except ImportError:
            self.skipTest("Pillow is not installed")
        if PIL.__version__ != image_decode.PILLOW:
            self.skipTest(f"the corpus is pinned to Pillow {image_decode.PILLOW}")
        rows = {case["name"]: case for case in self.golden()["cases"]}
        built = image_decode.cases()
        self.assertEqual([case.name for case in built], list(rows))
        for case in built:
            self.assertEqual(rows[case.name]["file"], f"image-decode/{case.name}.{case.extension}")
            self.assertEqual(hashlib.sha256(case.data).hexdigest(), rows[case.name]["file_sha256"], case.name)


if __name__ == "__main__":
    unittest.main()
