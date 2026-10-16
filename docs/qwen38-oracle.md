# Qwen3.8 Flash dump checks (`qwen38-oracle`)

Start with the [first-divergence workflow](debugging.md) for input identity,
fixed-schedule replay and journal comparisons before capturing layer tensors.

`qwen38-oracle` checks [bundle-qwen38](bundle-qwen38.md) tap dumps
(`CHOPINFER_TAPS_DIR`) operation by operation on the CPU. It holds no
whole-model reference: teacher-forced logprobs, image probes, long context and
layer taps come from the GPU [reference runner](reference-runner.md)
(`models/qwen38/reference.py`). Code: `models/qwen38/compiler/src/oracle.rs`
and `src/oracle/`.

Load the local environment first, then build in release:

```sh
cargo build -p model-qwen38 --release --locked --offline --bin qwen38-oracle
```

Common options: `--checkpoint DIR` (default `$CHOPINFER_QWEN38_CHECKPOINT`)
and `--prompts FILE` (default
`$CHOPINFER_DATA/qwen38-refs/prompts/prompts.json`, with `teacher` and `long`
lists of `{"name", "token_ids"}`), or `--completions FILE`, a JSON list of
`{"name", "text", "continuation"}` whose text is tokenized as a completions
request does and whose continuation tokens are appended. `--precision-component
FILE` selects the prepared weight-decoding contract for another compatible
checkpoint ([quantized checkpoints](quantized-checkpoints.md)). Outputs are
never overwritten.

## QSA selection policies

`qsa-capture` recomputes a full-attention layer's QSA key sets from the
dump's own BF16 `x.indexer.qk` rows under three policies
(`oracle/selection.rs`). Each keeps the 512 highest scores, ties to the lower
block:

- `engine`: the engine's BF16 queries and block keys, from an emulation of
  `indexer_query` and `indexer_key_write` in their F32 order and the engine's
  cos/sin table, scored in f64;
- `kernel`: `engine` scored in `block_scores`' F32 MMA order (an emulation,
  not the tensor core's bits);
- `hf-style`: an approximation of HF's BF16 indexer sequence with chosen F32
  reductions and the lower-block tie rule; not HF's bits.

## Commands

- `qwen38-oracle qsa-capture --dump DIR --prompt NAME [--occurrence K]
  [--stats-every N] [--rope-table hf|f64-angle] [--export DIR
  [--export-every N]] --out FILE` compares a dump's prefill
  `x.indexer.topk` sets with the three recomputed selections on every Nth
  sparse row (default 1) of each full-attention layer the dump holds.
  `--rope-table` names the engine's cos/sin table (`hf`, served, or
  `f64-angle`). Dump every row of the prompt with `CHOPINFER_TAPS_LAYERS` set
  to some of the full-attention layers
  (`3,7,11,15,19,23,27,31,35,39,43,47`) and
  `CHOPINFER_TAPS_SITES=x.indexer.qk,x.indexer.topk`. A layer without a sparse
  row (a prompt of 2,051 tokens or fewer) is refused.

  `--export DIR` (a new directory) writes, per layer, the operands of
  kernel-lab's `qsa_f32_selection_on_oracle_operands` test
  (`CHOPINFER_QSA_OPERANDS=DIR`, see
  [SPARSE.md](../crates/kernels-sm120/cuda/attention/SPARSE.md)):
  `layerNN.qk.bf16`, `layerNN.norm.bf16`, `layerNN.keys.bf16`,
  `layerNN.q.bf16` and `layerNN.json` (the exported rows with their exact
  top-k, the emulated kernel's set and the relative gap). Exported rows are
  those where the device's, `engine`'s and `kernel`'s sets differ, where
  `engine`'s relative gap is below 1e-5, and every row r with r % N == 0
  (`--export-every`, default 64). kernel-lab reads only the `hf` table.
- `qwen38-oracle batch-drift --prompt NAME --a DUMP [--a-occurrence K] --b
  DUMP [--b-occurrence K] [--reference TAPDIR] --out FILE` compares one
  prompt between two dumps, wherever it sat in a packed prefill: per tap the
  bit-identical rows and the first differing row, relative distances, and for
  `x.indexer.topk` the key-set differences and rank agreement. Dumps limited
  with `CHOPINFER_TAPS_FROM` compare only their kept rows.

  `--reference` names one mode's tap directory of a reference runner run
  (`$RUN/taps/f32`). The step holding the prompt must be the only one. Each
  tap the reference stores under the same name is compared with both dumps
  on the shared positions (relative L2 `||dump - ref|| / ||ref||`, and on the
  prompt's last row when stored). A `replicated` tap is compared on rank 0 and
  a `rank_partial` tap (`attn.out`, `mlp.out`, `moe.shared`) on its rank sum;
  column-sharded, expert-partial, set, paired and encoder taps are not
  compared. Add `x.hc.attn`, `x.hc.mlp` and `x.hc.mixer` (or `--tap-sites
  all`) to the runner's sites for the hyper-connection inputs.
- `qwen38-oracle prepared-capture --a STEP --precision-component FILE
  --layers A-B [--tail N] --out FILE` checks complete selected rows of TP4
  dense and expert projections on the exact BF16 inputs the running bundle
  captured. `--tail` selects the last N captured rows of the phase (default
  2). Capture every requested layer with `CHOPINFER_TAPS_LAYERS` and leave
  `CHOPINFER_TAPS_SITES` unset. Comparisons use the half-BF16-ulp plus
  `2^-22*K*S` bound and row cosine at least 0.9999; routing IDs must match
  exactly; missing operands, empty selections and incompatible shapes fail.
  This checks operation arithmetic, not whole-model quality.
- `qwen38-oracle measure-capture` takes the same arguments and reports the
  same statistics without failing on a numerical violation.

## Tests

`cargo test -p model-qwen38 --lib oracle` (in CI) checks, on synthetic data,
the selection policies against kernel-core's `block_scores` and `block_topk`,
the RoPE emulations, a capture's comparison and export, the drift and
reference tap readers, the tap name declaration and the prepared capture's
bound.
