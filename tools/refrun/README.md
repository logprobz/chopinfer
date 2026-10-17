# Reference runner

`tools/refrun/` runs a model's upstream reference code (HF transformers or
the vendor's PyTorch modeling file) on GPUs and scores sequences under a
declared numerics policy. Qualification gates compare the engine against
these scores. The runner owns loading, placement, exact dequantization,
modes, scoring and cache keys; a per-model adapter,
`models/<model>/reference.py`, owns everything model-specific.

The [GPU reference runner guide](../../docs/reference-runner.md) is the
contributor page: environment setup, running workloads, numerics modes,
the cache and fixture bindings, taps and tapdiff, the encoder reference and
writing an adapter. This README is the module index.

## Commands

| Command | Use |
|---|---|
| `python3 -m tools.refrun run` | Score a workload through the cache (GPU, needs torch) |
| `python3 -m tools.refrun.dry_run run ...` | Check a `run` command line up to the weight load (CPU) |
| `python3 -m tools.refrun emit` | Assemble a bound gate fixture from run outputs |
| `python3 -m tools.refrun cache gc` | List or delete unreferenced cache entries |
| `python3 -m tools.refrun diff-entries A B` | Compare two cache entries or refsets |
| `python3 -m tools.refrun verify-binding FIXTURE...` | Check fixtures' reference bindings |
| `python3 -m tools.refrun.stamps` | Verify checkpoint files against their lock (CPU) |
| `python3 -m tools.refrun.encoder` | Evaluate an encoder declaration over a workload's media (GPU, needs torch) |
| `python3 -m tools.refrun.frontend_corpus generate\|minimize\|check\|rounds --model M` | The HF differential corpus of a model's chat frontend (CPU, needs transformers) |

Each takes `--help`. Run GPU commands with `CHOPINFER_REFRUN_PYTHON`, the
runner's CUDA environment ([setup](../../docs/reference-runner.md#prepare-and-run)).

## Modules

| File | Content |
|---|---|
| `workload.py` | Workload files: sequences with name, category, prompt token IDs, `score_from`, forced or greedy answer, media tensors with processor identity |
| `modes.py` | The `f32`, `f64` and `bf16` policies as canonical JSON, the process environment and the torch flags applied around every forward pass |
| `mode-policies.json` | The mode policy documents, for verifiers that cannot import `modes.py` (`chopinfer eval`) |
| `quant.py` | Exact decoders (E4M3, E2M1, E8M0), FP8 block, MXFP4, NVFP4 and BF16 dequantization, `QuantTensor`, `QuantLinear`, `QuantStack` |
| `place.py` | Safetensors headers, placement of units over devices by stored bytes, the bounded loader (`Store`), host row gathers (`HostRows`) |
| `ops.py` | Exact eager replacements: query-chunked attention and an HF attention-interface wrapper |
| `score.py` | Teacher forcing and greedy continuation |
| `margins.py` | Logit gaps between two tokens of a scored row, for near-tie decisions (no torch) |
| `keys.py` | Reference cache key documents (`keys.NUMERICS`, `keys.ENCODER_NUMERICS`) |
| `cache.py` | The reference cache: sealed entries, refsets, fixture bindings, gc and entry diffs |
| `stamps.py` | Checkpoint files checked against the lock by SHA-256, with the stamp cache |
| `taps.py` | Per-layer taps through forward hooks, in the engine dump format |
| `adapter.py` | The adapter interface and the default HF forward and decode calls |
| `run.py` | The CLI: `run`, `cache gc`, `diff-entries`, `verify-binding` |
| `emit.py` | `emit`: a bound gate fixture assembled from run outputs, with its canonical workload |
| `encoder.py` | The generic encoder reference |
| `encoder_ops.py` | One torch implementation per shared encoder component variant, with each mode's arithmetic |
| `tiny.py` | A synthetic adapter (tiny random Llama with FP8, MXFP4 and BF16 MLP weights) for tests and calibration |
| `calibrate.py` | GPU calibration: GEMM and attention rates, read rates, determinism |
| `requirements-cu130.txt` | The pinned distributions of the runner's environment |
| `frontend.py` | Frontend goldens of the legacy capture fixtures, rendered by HF |
| `template_coverage.py` | The chat-template lines one render runs, from the compiled template's line marks, and the function lines a pinned Python encoder runs |
| `frontend_cases.py` | Seeded, pairwise chat requests over model-neutral feature axes, each with its `axis:value` tags |
| `frontend_corpus.py` | The frontend corpus: HF renders through the model's transport translation, coverage, round-trip vectors, minimization and the coverage and divergence gate |

A model's frontend corpus reads `models/<model>/frontend_cases.py` (the
template pin, or for a checkpoint that ships a Python encoder instead of a
template, the encoder's `source` and `source_sha256` with `encoder_render`,
the mapping of the HF input onto its functions, and `INPUT_FIELDS`, the
request fields it renders beyond the HF input; the transport translation, generation endings, how a history
turn reads back, and the model's own axes and cases) and its divergence
table `models/<model>/frontend-divergences.json` (its optional `mapping`
section records each request-mapping rule a second authority governs, with
its source and the committed cases that show it). `generate` writes the full
corpus under `$CHOPINFER_DATA/frontend-corpus/<model>/`; `minimize` writes
the subset a model commits as `fixtures/<model>/frontend/corpus*.json`, and
`check` enforces the gate on it. `frontend_corpus.corpus_test_case(model)`
is the model's reference test, skipped without its checkpoint variable.

The HF input applies the transport's protocol to every model alike: a
top-level `reasoning_effort` replaces the nested one and, unless the client
sent `enable_thinking`, sets it to whether the effort differs from the
declared disabling value. A divergence table states only where a template
differs. `corpus-coverage.json` records the reference environment (Python,
transformers, jinja2 and tokenizers versions and the Unicode version), and
`check` warns when it runs another.

A round trip reads a history turn back through the served parser. Its
lossless domain is a turn that ends the request and carries no media,
whose history render less the generation prompt is the completion, which
the cases module reads back, whose every argument has the one type its
schema names as the parser resolves it (the declared property, else the
additional-property schema; a string where no schema types it or the type
is a string alias; a number for an integer), whose values the parser scans
as raw text spell no marker that ends them (a marker inside a JSON value's
string literal is content), and whose completion the tokenizer's
normalizer leaves as it is. A malformed or unoffered call is skipped. Any
other accepted turn outside the domain is a failed round trip of one class
(`prefix`, `stop`, `read-back`, `union`, `untyped`, `mismatch`,
`delimiter`, `tokenizer`), which needs exactly one `no-round-trip` entry
whose `when` the case's features plus `round-trip:<class>` and `mode:chat`
or `mode:thinking` (the mode the served frontend selects) satisfy; the
entry goes stale unless a committed case fails that way. A tokenizer loss
excuses only the bytes the normalizer rewrites: the case still yields a
round trip with the completion's ids that expects the normalized turn, and
the harness refuses it unless the normalized turn renders as the
normalized completion with the same ids.
A refused case yields no round trip, unless every refused entry it matches
sets `ending_only`: a format that refuses only the conversation's final
turn (a turn it would render closed, say) still reads that turn back after
the prefix it serves.

The model adapters are `models/mimo26/reference.py` and
`models/qwen38/reference.py`, with tiny CPU tests in
`models/<model>/test_reference.py`; model-specific encoder variants live in
`models/<model>/encoder_reference.py` (MiMo).

## Tests

```sh
python3 -B -m unittest discover -s tools/refrun -t .
```

Without torch the key, workload, mode and cache tests run and the rest skip.
With `CHOPINFER_REFRUN_PYTHON` set, `ci/cpu.sh` runs them all on CPU:
exhaustive decode of every E4M3, E2M1 and E8M0 code, block and chunked-grid
dequantization, golden keys, the tiny model scored by the runner and by a
plain forward with bitwise-equal results and replayed from the cache, and
the encoder reference against direct f64 definitions and each model's
upstream encoder code (`models/<model>/test_encoder_reference.py`).

A CPU-only process must not initialize the CUDA driver. transformers asks
`torch.cuda.is_current_stream_capturing()` on every forward, which does, so
`modes.prepare_environment(cuda=False)` answers that query itself.
