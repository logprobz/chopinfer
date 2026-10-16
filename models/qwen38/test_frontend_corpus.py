"""Qwen3.8's committed HF differential corpus against the pinned template.

Runs `frontend_corpus check` over `fixtures/qwen38/frontend/`: every case
renders as HF renders it at the pin, the coverage gate holds with
`models/qwen38/frontend-divergences.json`, and the committed cases keep
every feature tag of the full corpus. Skipped with a message when
`CHOPINFER_QWEN38_CHECKPOINT` is unset.

    $CHOPINFER_REFRUN_PYTHON -B models/qwen38/test_frontend_corpus.py
"""

from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.refrun.frontend_corpus import corpus_test_case  # noqa: E402

FrontendCorpus = corpus_test_case("qwen38", ROOT)

if __name__ == "__main__":
    unittest.main()
