# HF frontend references

These 61 chat cases and 52 tokenizer cases are generated with transformers
5.16.1 from the local MOPD checkpoint by `python -m tools.refrun.frontend`.
`provenance.json` pins the template, tokenizer and processor configuration.
The API's `reasoning` field maps to HF's `reasoning_content`, and string
function arguments are parsed as JSON before template rendering. A plain
API `reasoning_content` field remains ignored, as the serving contract states.
The generator reads only the cases' messages and text, then recomputes every
expected token ID and rendered string with HF. It loads no model weights.

`corpus.json`, `corpus-rounds.json` and `corpus-coverage.json` are the HF
differential corpus of the chat template, minimized from the full corpus by
`python -m tools.refrun.frontend_corpus minimize --model mimo26` (seed 0,
107 cases kept as 31, and its 12 round trips). The cases come from
`tools/refrun/frontend_cases.py` and `models/mimo26/frontend_cases.py`, and
`models/mimo26/frontend-divergences.json` lists every outcome that differs
from HF's on purpose. `python -m tools.refrun.frontend_corpus check --model
mimo26` (`models/mimo26/test_frontend_corpus.py`) re-renders them with HF and
enforces the coverage gate; `models/mimo26/compiler/tests/frontend_corpus.rs`
runs them through MiMo's export.
