"""MiMo's committed HF frontend corpus passes the coverage and divergence gate.

`python -m tools.refrun.frontend_corpus check --model mimo26` re-renders the
committed corpus (`fixtures/mimo26/frontend/corpus*.json`) with HF at the
pinned template from `$CHOPINFER_MIMO_CHECKPOINT`, compares it byte for byte
and enforces the gate against `models/mimo26/frontend-divergences.json`.
Needs transformers and the checkpoint's template and tokenizer; skipped when
`CHOPINFER_MIMO_CHECKPOINT` is unset.
"""

import unittest

from tools.refrun.frontend_corpus import corpus_test_case

FrontendCorpus = corpus_test_case("mimo26")


if __name__ == "__main__":
    unittest.main()
