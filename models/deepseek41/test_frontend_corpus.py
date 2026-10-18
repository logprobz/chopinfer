"""DeepSeek V4.1 Flash's committed frontend corpus against the pinned encoder.

`fixtures/deepseek41/frontend/corpus.json` must still be what the checkpoint's
`encoding/encoding.py` renders, and what its tokenizer gives, at revision
1bae0e10 (`frontend_cases.py` pins it), and it must pass the coverage and
divergence gate with `models/deepseek41/frontend-divergences.json`. Needs
transformers and the pinned encoder and tokenizer files, from
`CHOPINFER_DEEPSEEK_CHECKPOINT`; skipped without them.

    CUDA_VISIBLE_DEVICES= $CHOPINFER_REFRUN_PYTHON -B -m unittest models/deepseek41/test_frontend_corpus.py
"""

from pathlib import Path
import hashlib
import os
import runpy
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.refrun.frontend_corpus import corpus_test_case  # noqa: E402

FrontendCorpus = corpus_test_case("deepseek41", ROOT)


class ToolImageOrder(unittest.TestCase):
    def test_vendor_collects_tool_images_after_sorting(self):
        from models.deepseek41.frontend_cases import PIN, tool_image_cases

        checkpoint = os.environ.get(PIN["checkpoint_env"])
        if not checkpoint:
            self.skipTest(f"{PIN['checkpoint_env']} is unset")
        source = Path(checkpoint) / PIN["source"]
        self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), PIN["source_sha256"])
        encoder = runpy.run_path(str(source))
        for name, request, order in tool_image_cases():
            with self.subTest(name=name):
                prompt, media = encoder["encode_messages"](
                    request["messages"], thinking_mode="chat", return_multi_modal_data=True)
                self.assertEqual([image["url"] for image in media["images"]], order)
                self.assertEqual(prompt.count("<｜deepseek_image｜>"), len(order))

if __name__ == "__main__":
    unittest.main()
