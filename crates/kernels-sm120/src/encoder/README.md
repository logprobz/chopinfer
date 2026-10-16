# Multimodal encoder kernels

Bundles build their encoder cubins from their encoder declarations
(`crates/encoder-spec`). `encoder::catalog(&declaration, &registered)`
(`src/encoder/`) lists the entries one declaration runs, each once, and
`encoder::cubin_specs` builds a model's two encoder cubins from the union of
its declarations' entries: rows and GEMMs (`multimodal_rows_gemm_sm120`) and
contiguous attention (`multimodal_attention_sm120`). Each cubin includes
only the headers its entries' templates need: every row template has a
header of its own under `cuda/encoder/rows/` (with the frozen
`common.cuh`), so editing one variant's kernel moves only the locks of the
models whose entries call it.

- Symbols derive from (variant id, parameters):
  `chop_enc_<kind>_<name>[_<role>]_<template parameters>_<digest>_v<N>`, for
  example `chop_enc_linear_pairs_n3456_k1152_bias_f32_1a2b3c4d5e6f_v1`. The
  digest is the first 12 hex digits of the SHA-256 of the full variant id,
  the role and the entry's code, so ids that read alike (`foo-bar`,
  `foo.bar`) stay apart. The symbol without its digest is the entry's stem,
  which hand-written launch sequences use (`encoder::by_stem`). Equal stages
  of two models share a symbol and a body; two bodies never share a symbol.
- Runtime values are launch parameters, not code: stages that differ only in
  eps or capacity share one entry, which grids for the largest capacity
  (`encoder::union`).
- `src/encoder/lower.rs` lists what each shared variant runs. GEMMs inside a
  transformer, MLP or convolution run as the linear variant the precision
  policy picks. A model's registered (`<model>:`) variants supply their
  kernels through `encoder::Registered`, which the model implements beside
  its provider.
- Every entry plans for the declaration's `capacity_rows`, except the
  embedding override, which runs in the language model's prefill at
  `encoder::OVERRIDE_ROWS`.
- Each linear variant's GEMM epilogue is a functor in its own header
  (`cuda/encoder/linear_*.cuh`) that the shared engines take as a template
  parameter. A new epilogue is a new functor, never an engine edit.
- Each bundle's `encoder_catalog_stability` test builds its generated
  cubins on the CPU and compares them with its `kernels.lock` and with the
  adoption of its current declarations in `encoder-adoptions.json`
  (`encoder::stability`). That file is append only: a model's encoder
  cubins change only when its declarations change (adopting a new variant
  id), and then an adoption for the new declaration digests is appended; the
  failing test prints it. Each adoption names the digest of the one before
  it, so an edit before the last breaks the chain, and the ledger must start
  with the approved base's entries (the file on `main`), which
  `tests/encoder-adoptions.py` checks in `ci/cpu.sh` and the stability test
  checks when git is available. Refreshing the lock alone, or with a
  rewritten adoption, fails.

Generated conformance tests run each model's serving cubins directly. CPU
oracles can still read BF16-only encoder queries, but those cannot construct
GPU entries. `cuda/gemm/encoder_gemm.cuh` and `cuda/attention/varlen.cuh` are
frozen because the pair engines include them: removing unused bodies would
change serving source closures.

## Qwen3.8 F32 vision tower

The Qwen3.8 Flash vision declaration runs an F32 tower at capacity 65,536
rows (one 16,777,216-pixel image is one attention sequence) and 8192 for LM
injection. Rows between ops are F32. Every tensor-core operand that is an activation is a
BF16 pair, `hi = bf16(x)` and `lo = bf16(x - hi)` (`cuda/common/bf16_pairs.cuh`),
and both halves multiply the BF16 operand, so each product carries 16 bits
of the activation instead of 8:

- `cuda/gemm/encoder_gemm_pairs.cuh` (contract `GemmPairs`): the encoder GEMM's tiles
  and schedule over F32 activations. Two stages of 32 KiB F32 activations
  and 16 KiB weights fill the same 99 KiB; activation rows are 256 bytes
  with 32-byte column groups XOR-swizzled by row, and the math warps split
  each 8-byte fragment load into its pair and issue two MMAs. The output is
  F32, F32 added in place to the residual stream (`proj`, `fc2`), or BF16
  (the merger's last GEMM, the LM's rows).
- `cuda/encoder/rows/interpolated_position_add_f32.cuh`,
  `layernorm_f32.cuh` and `rope_qkv_pairs.cuh`: the F32 position add and
  LayerNorm, and `rope_qkv_pairs`, which rotates q and k of a packed F32 qkv row and
  rewrites the row in place as six BF16 planes (q hi, q lo, k hi, k lo, v
  hi, v lo).
- `cuda/attention/varlen_pairs.cuh` (contract `ContiguousAttentionPairs`): the
  varlen engine over those planes with F32 output. Scores take q hi k hi,
  q hi k lo and q lo k hi; every probability term multiplies both halves
  of v; probabilities keep at least two terms (three below 128 keys).
  Two stages of four key tiles need 90,112 bytes of dynamic shared memory.

The MLP is padded from 4304 to 4352 by the weight plan (zero fc1 rows and
bias, zero fc2 columns) to meet the encoder GEMM's N % 128 and K % 64.
Only cubins with these entries include their headers.

## Entry contracts

Other entries have capacity 8192 unless a dimension is explicitly in the query.
`live_rows`, `live_tokens` and `live_frames` are device `u32` pointers. The
planner must grid for the prepared capacity and read these counts on device.
Scalar `eps_bits` and `scale_bits` carry F32 bits. Rows use BF16 storage except
F32 logits, mel input, RVQ input and the F32 vision tower's rows. Outputs use
BF16 except RVQ codes (`i32`) and the F32 entries.

| Operation | Contract and layout |
|---|---|
| Contiguous attention | Q `[T,QH,D]`, K/V `[T,KH,D]`, output `[T,QH,D]`, `cu_seqlens[sequences+1]` i32 in token units. Each sequence starts at zero for causal and window comparisons. `window_left/right` are maximum index distances, inclusive. A 128-token causal window uses left distance 127. An optional sink adds one F32 bias per query head to the sequence's key 0 logit before softmax, only when that key is in the window. GQA groups contiguous Q heads per KV head. |
| LayerNorm | Input/output `[rows,width]`, affine weight and bias `[width]`. Population variance, `((x-mean)/sqrt(variance+eps))*weight+bias`. |
| GEMM bias and GELU | A `[M,K]`, B `[N,K]`, bias `[N]`. F32 accumulation; column bias precedes exact-erf or tanh GELU, SwiGLU, and residual. The typed `Epilogues` fields distinguish these identities. |
| Scratch RoPE | In-place Q `[T,QH,D]` and K `[T,KH,D]`, cos/sin `[T,rotary_dim/2]` F32. NeoX and GPT-J pairings are distinct. Unrotated columns pass through. The supplied cos/sin rows encode 1D or 2D positions. |
| Permute | `output[row] = input[indices[row]]`; indices form a bijection over live rows. Inverse permutation uses the inverse list. |
| 1D im2col | Input `[time,channels]`, output `[out_time,kernel*channels]` with tap-major K, zero padding and stride from the static query. F32 mel input is cast once to BF16. |
| RVQ nearest | F32 input `[rows,dim]`, BF16-decoded codebooks packed by stage, offsets `[stages+1]` in code units, output codes `[rows,stages]`. Each stage chooses minimum squared Euclidean distance in F64 reference arithmetic, lower code on ties, and subtracts that codeword before the next stage. |
| Multi-embedding sum | Tables packed `[tables,vocab,width]` BF16, codes `[rows,tables]`, output `[rows,width]` BF16. Sum in ascending table order in F32 on device and F64 in the oracle. |
| Interpolated position add (F32) | In-place-capable hidden `[rows,width]` F32, table `[table_rows,width]` BF16, indices `[rows,4]` i32, weights `[rows,4]` F32. `input + sum_j w[j] * table[idx[j]]` with each F32 product and sum rounded in tap order (no FMA). Taps outside the table add zero. |
| LayerNorm (F32) | As LayerNorm, F32 input and output. |
| QKV RoPE into BF16 pairs | In place over packed F32 rows `[T,(QH+2KH)*D]`: q and k rotate as scratch RoPE (each product rounded, then the add), then the row becomes the planes q hi, q lo, k hi, k lo, v hi, v lo (BF16, same bytes). |
| Pair attention | As contiguous attention, over the pair planes; F32 output. The oracle's `S` adds the logits' reduction term, `(D / K) sum_j p_j |v_j - o| scale sum_i |q_i k_ji|`, since no BF16 output rounding absorbs it; the kernel's omitted products (q lo k lo, and the probability past its terms) are within that bound (`pair_terms_fit_the_bound`). |
| Pair GEMM | A `[M,K]` F32 split into BF16 pairs, B `[N,K]` BF16, bias `[N]`; GELU, then an optional in-place F32 residual; F32 or BF16 output. The oracle multiplies the pairs' values. |
| Q/K RMSNorm (F32) | In place over packed F32 rows `[T,(QH+2KH)*D]`, before `QKV RoPE into BF16 pairs`: every q and k head becomes `(x*rstd)*weight` with BF16 weights `[D]` (one for q, one for k) and `rstd` the correctly rounded reciprocal square root of the head's F32 mean square plus eps; v is untouched. One warp per head, lanes summing their columns with FMA and a fixed butterfly (`rows/qk_rms_f32.cuh`). |
| Clamped SwiGLU (F32) | As SwiGLU over F32 rows, `silu(min(gate, limit)) * clamp(up, -limit, limit)` with `limit_bits` F32 bits; compares and selects keep a NaN a NaN (`rows/swiglu_clamped_f32.cuh`). |
| Zero-column pad (F32) | F32 rows `[T,input]` copied into rows `[T,output]`, zero past `input`, so the pair GEMM (K a multiple of 64) reads a payload of another width; zero columns add exact zeros (`rows/pad_columns_f32.cuh`). |
| GELU (F32) | Exact-erf GELU of F32 rows, in place or not, through the GEMMs' erf epilogue (`rows/gelu_f32.cuh`); the projector's GELU between its norm and its MLP. |
| Embedding override | In-place activation `[rows,width]` BF16, flat arena `[arena_rows,width]`, source `[capacity]` i64. `-1` keeps the gathered row; nonnegative values select an arena row. `arena_rows` is a runtime bound. |

The CPU oracles are in `kernel_core::multimodal_oracle`. Numerical IDs are
provided by `MultimodalQuery::numerical_contract_id`; GEMM uses the existing
`GemmQuery` identity with explicit bias and GELU suffixes. Encoder
declarations use the F32 RMS entries (`cuda/encoder/rows/rms_f32.cuh`).
