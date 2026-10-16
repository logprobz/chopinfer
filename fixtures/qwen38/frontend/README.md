# HF frontend references

Chat renders, raw token IDs and image prompt IDs are generated on CPU with
transformers 5.16.1 from the local checkpoint. The generator is
`python -m tools.refrun.frontend`. `provenance.json` pins the template,
tokenizer, processor configuration and image bytes. The API's `reasoning`
field is mapped to HF's `reasoning_content`; tool-call argument strings are
parsed as JSON. This preserves the serving transport's declared semantics.

`api-requests.json` is the fixed request set. `api-prompts.json` records its
HF-rendered prompt IDs. `capture/frontend_api.py` prepares a workload for
`tools/refrun` with `models/qwen38/reference.py`, F32 greedy. The resulting
`api-shapes.json` records finish reason, nonempty content/reasoning and tool
names for nine successful non-stream cases. The other six cases exercise
error or stream transport contracts. All 15 cases remain required.

For `completion-n2-seeded`, the protocol is: shape from F32 greedy at the
same max_tokens; sampling parameters exercised by the server only. Its
server request keeps temperature 1.0, top_p 0.95, n=2, seed 7 and
max_tokens=16. The API check requires two choices with distinct indexes.
Both choices' finish reasons still equal the HF greedy shape exactly;
a sampled stop/length difference is a finding, not an allowed alternative.

`api-compatibility.json` contains fixed decoder input examples retained from
the previous API capture. Their text is test input for parsing reasoning,
tool calls and streaming deltas, not an expected model answer. The tool-result
request likewise preserves that historical conversation as fixed input.

Regenerate into an external directory first, compare every changed value,
and update digests only after reviewing the differences. F32 API answers run
through the GPU queue. Tokenizer and image-prompt regeneration loads no weights.

`corpus.json`, `corpus-rounds.json` and `corpus-coverage.json` are the
minimized HF differential corpus, written by `python -m
tools.refrun.frontend_corpus minimize --model qwen38` from a full corpus that
`generate` writes under `$CHOPINFER_DATA`. `frontend_corpus check` re-renders
them at the pin and enforces the coverage gate with
`models/qwen38/frontend-divergences.json`; `models/qwen38/compiler/tests/frontend_corpus.rs`
runs them through the served frontend.
