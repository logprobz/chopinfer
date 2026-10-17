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
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.refrun.frontend_corpus import corpus_test_case  # noqa: E402

FrontendCorpus = corpus_test_case("deepseek41", ROOT)

if __name__ == "__main__":
    unittest.main()
