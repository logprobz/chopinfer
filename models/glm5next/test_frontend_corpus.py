"""GLM 5.3 Flash's committed HF frontend corpus against the pinned template.

`fixtures/glm5next/frontend/corpus.json` must still be what transformers
renders and tokenizes at revision eb9eb208 (`frontend_cases.py` pins it), and
it must pass the coverage and divergence gate with
`models/glm5next/frontend-divergences.json`. Needs transformers and the
pinned template and tokenizer files, from `CHOPINFER_GLM5NEXT_TOKENIZER_DIR`
or `CHOPINFER_GLM5NEXT_CHECKPOINT`; skipped without them.

    CUDA_VISIBLE_DEVICES= $CHOPINFER_REFRUN_PYTHON -B -m unittest models/glm5next/test_frontend_corpus.py
"""

from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.refrun.frontend_corpus import corpus_test_case  # noqa: E402

FrontendCorpus = corpus_test_case("glm5next", ROOT)

if __name__ == "__main__":
    unittest.main()
