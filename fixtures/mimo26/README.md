# MiMo V2.6 Flash reference inputs

Recorded inputs for the MiMo V2.6 Flash frontend, kernel and model tests.
Frontend goldens come from local HF templates and tokenizers. The conversation
set is built by `capture/conversations.py`; see [frontend](frontend/README.md).

## Files

| File | Contents |
|---|---|
| `frontend/chat-renders.json` | 61 conversations (plain, system, multi-turn, thinking off, prior reasoning, tools, tool calls, tool results, GSM8K prompts) with HF-rendered prompt IDs and decoded text. |
| `frontend/tokenizer-golden.json` | 52 raw texts and their token IDs (HF tokenizer, no special tokens added). |
| `teacher-texts.json` | The teacher-forced texts for the exact reference: the 3 frozen texts plus further short texts. |
| `teacher-exact-reference.json` | Reference-code scores of the MOPD checkpoint over `teacher-texts.json` from the GPU reference runner, F32 with BF16 and f64 spread variants, with its cache binding; `teacher-exact-reference.workload.json` is the workload it scored (see [below](#reference-code-references)). |
| `image-layout-goldens.json` | Each rank's text, MTP and DFlash image layout at 2 and 4 ranks for Flash's synthetic served headers: arena length, view and piece counts, padding, first and last view, and the SHA-256 of canonical lines naming every view's arena offset, size and padding and every piece's arena offset, size and record. Produced by the pre-port planner (the retired rank slicer) on main of 2026-10-06 with the format in `models/mimo26/compiler/tests/weight_plan.rs`, which compares the derived planner's images with it; the fixture names the generating command. The encoder image is checked against an independent byte reference in the same test. |
| `routing-locality.json` | Distinct-expert counts per MoE layer for decode and verify batch shapes, measured from real hidden states. |
| `sampling/` | Real BF16 logits rows for the sampling kernel tests (see [sampling/README.md](sampling/README.md)). |
| `multimodal/` | HF processor inputs, F32 greedy answers and forced F32/BF16/F64 reference scores, with manifests and cache bindings. |
| `capture/` | The scripts that produce these files. |

Findings that bind the Chopinfer frontend:

- Prior assistant reasoning is read from the `reasoning` message field. A
  `reasoning_content` field is ignored and renders as `<think></think>`
  (`reasoning-content-field-ignored`).
- Responses carry reasoning in `message.reasoning` and tool calls with
  `finish_reason` `tool_calls`.
- Literal special-token text inside message content is parsed as the special
  token (`plain-15`).
- Tool-call arguments given as a JSON string are parsed and rendered as
  `<parameter=K>V</parameter>`, with non-string values in JSON (`toolcall-*`).

## Multimodal references

`capture/frontend_media.py` regenerates prompts and processor arrays on CPU.
The [processor README](multimodal/processor/README.md) documents the fixed
video RGB input and its unchanged digest. `capture/media_answers.py` turns
refrun F32 greedy answers into `reference.json` and a forced workload. The
forced reference uses `models/mimo26/reference.py` in F32, BF16 and F64.
All 11 media cases and their original limits remain required. The two
audio cases were withdrawn with their speech clip: MiMo's audio is served
but not qualified in-tree.

## Reference-code references

Accuracy is judged against the reference model code, not against vLLM. The
teacher and media gates score against runs of the vendor's
`modeling_mimo_v2.py` on the MOPD checkpoint (`models/mimo26/checkpoints.json`)
on the GPU reference runner ([tools/refrun](../../tools/refrun/README.md)),
with the adapter `models/mimo26/reference.py`. Each reference carries a
`binding` block: its refset, every cache entry's key and the sub-digests of
the checkpoint lock, upstream code, adapter and runner files, mode policies
and environment. The gates verify it against the working tree and the
workload beside it (`<stem>.workload.json`), and against the cache when
`$CHOPINFER_DATA/refcache/v1` is present (see [reference cache
bindings](../../tools/qual/README.md#reference-cache-bindings)).

Variants, in the runner's score layout (logprob, top-1, logit margin and the
top 5 IDs, logits and logprobs):

- `exact` (the gate variant): the runner's `f32` mode, each exact
  dequantized product rounded once to F32 (`--dequant f32=rounded`, since
  MiMo's FP8 block scales are F32), F32 log-softmax;
- `bf16`: upstream BF16 loading and BF16 activations and encoders, F32
  log-softmax;
- `f64`: exact dequantized products, f64 activations and log-softmax.

`bf16` and `f64`, scored against `exact` like a server, are the spread that
sets each gate's limits (`tools/qual/limits.py`).

`teacher-exact-reference.json` scores the 33 texts of `teacher-texts.json`
(4,108 positions). `multimodal/exact-reference.json` scores each frozen
answer of `multimodal/reference.json` forced after its prompt and the
case's processor output (one sequence per case; frames with equal pixels
share one payload), from `exact-reference.workload.json` and its `media/`
tensors. Regenerate both on the GPUs:

```sh
python3 fixtures/mimo26/capture/runner_references.py workloads --payloads DIR --out WORK
python3 fixtures/mimo26/capture/runner_references.py skeletons --out SKELETONS
# GPU job: python3 -m tools.refrun run --adapter models/mimo26/reference.py --lock-variant mopd \
#     --workload WORK/teacher.json --modes f32,bf16,f64 --dequant f32=rounded ... (and media.json)
python3 -m tools.refrun emit --skeleton SKELETONS/teacher.json --runs RUN [RUN ...] \
    --workload WORK/teacher.json --lock models/mimo26/checkpoint-files.json --lock-variant mopd \
    --modes f32=exact,bf16,f64 --output fixtures/mimo26/teacher-exact-reference.json
```

`DIR` holds the HF processor payloads (F32 pixel rows and mel frames). Emitting the media reference writes
`multimodal/exact-reference.json`; record its SHA256 in
`multimodal/manifest.json`. After regenerating either reference, update its
pins in `models/mimo26/qual.json` in the same change.

## Historical numerical inputs

Sampling logits and routing-locality observations remain fixed numerical
inputs. Sampling tests judge the sampling algorithm on those rows, not model
quality. Frontend and media correctness goldens use the HF reference.

## Third-party data

The GSM8K questions in the chat and tokenizer fixtures come from
[OpenAI's grade-school-math](https://github.com/openai/grade-school-math/tree/3101c7d5072418e28b9008a6636bde82a006892c),
revision `3101c7d5072418e28b9008a6636bde82a006892c`.
Copyright (c) 2021 OpenAI. The [MIT notice](../GSM8K-LICENSE) applies to those questions.
