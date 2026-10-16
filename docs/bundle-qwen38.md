# Qwen3.8 Flash bundle

`models/qwen38/bundle` (crate `bundle-qwen38`) is the native bundle for
Qwen3.8 Flash at 2 or 4 ranks (per-rank shapes in
[topology-qwen.md](topology-qwen.md)): text and image prefill, decode and MTP
speculation, up to 262,144 tokens. It holds the model program; the mechanics
(dispatch, graphs, input packing, upload, logits views, resources, the
instance table, snapshots, the encoder runtime) come from
[the shared bundle runtime](bundle-runtime.md). Geometry, layouts, views and
kernel catalogs live in the contract crate `model-qwen38-contract`
(`models/qwen38/contract`); the package and weight placement are in
[model-qwen38.md](model-qwen38.md). For supported profiles see
[deploy-qwen38.md](deploy-qwen38.md) and [serving profiles](serving-profiles.md).

## Modules

| Module | Holds |
|---|---|
| `contract` | Re-export of `model_qwen38_contract::contract`: per-rank geometry, page and state sizes, `BundleGeometry for Qwen` |
| `program` | The stage enum and the 48-layer order (`drafter_stages` for MTP) |
| `views` | Re-export of the contract's view declarations and `validate` |
| `launch` | The bundle's entry names as each geometry tags them, and the launch switches |
| `target` | What generic code asks of the GPU target: its backend and own kernels, modules, workspace, comm state and registered entries |
| `sm120` | The SM120 target: catalog entries (`sm120::catalog`), the backend's registrations, own kernels and the build script |
| `lower` | Binding each stage to views, phase inputs and rank buffers |
| `scratch` | Rank buffers sized from the geometry, and the cos/sin table |
| `side` | Per-phase side inputs: pool entries, row offsets, QSA piece counts, M-RoPE positions |
| `state` | Committed linear-state slots and the step checkpoint pool |
| `drafter` | MTP drafter constants over the shared `bundle_runtime::drafter` bookkeeping |
| `prefix`, `prefix_state` | Snapshot layout, capture and restore |
| `encoder` | Vision encoder image, acquisition and prefill injection |
| `precision` | Admission of the package's compiled operation inventory |
| `dump` | Diagnostic taps |
| `instance` | Acquisition, the step driver and the `entry::Bundle` implementation |
| `cuda/qwen38_bundle.cuh` | Stream expansion, the PLE residual, PLE staging, indexed state copies, the prefix copy kernel |

## The program

X is the four-stream residual, `[rows, 4, 2560]`, in two buffers: each
all-reduce and the PLE residual write the other one. Every sublayer runs:

1. `HcIn`: the down (320) and inject (4) projections of the normalized X.
2. `HcBottleneck`, `HcUp`, `HcMix`: `u = mean_s(sigmoid(up_s) * Xn_s)`.
   Decode runs all three as one launch
   (`chop_gemm_qwen38_hc_up_mix_decode_v1`).
3. The sublayer on `u`, ending in this rank's partial of its output.
4. `Reduce`: the hyper-connection all-reduce sums the partials, injects into
   X and writes the next grouped norm (the next sublayer's, PLE's before
   layer 1, or the final mixer's after layer 47). Decode graphs launch
   `chop_comm_qwen38_allreduce_hc_decode_r{N}_v3` at one or two prepared
   rows and `_v2` above; prefill launches `chop_comm_qwen38_allreduce_hc_v1`
   up to 32 rows and the phased `chop_comm_qwen38_reduce_scatter_v2` and
   `_allgather_hc_v2` pair above. All give the same bits; see the
   [collective reference](../crates/kernels-sm120/cuda/comm/README.md).

The prologue stages PLE state, gathers the embedding (then `MediaInject`
when the step injects image rows), expands it into four streams and
normalizes. Layer 1 first runs PLE (n-gram gathers, an all-gather of the
shards, projections, gate, grouped norm, per-item convolution). The
sublayers are linear attention (36 layers: convolution and gated delta rule),
full attention (12 layers: M-RoPE, paged KV, QSA top-512 block selection and
sparse attention, output gate) and MoE (48 layers, with a padded shared
expert). The epilogue runs the mixer and the vocabulary-shard head and gathers
the shards on rank 0; a prefill phase without logits rows skips them.

### Phases

Prefill runs eagerly, lowered stage by stage across ranks. Decode runs as
graphs per bucket (powers of two up to the served count rounded up, at most
256 items); the decode program binds only fixed buffers and device scalars
and plans QSA scoring for the whole context, so one graph serves every length.
A mixed step runs its prefill phase, then its decode phase.

The decode kernels hold at most 32 rows (`DECODE_ENTRY_ROWS`). Buckets of 64,
128 and 256 items run the token-indexed stages (dense and hyper-connection
GEMMs, routing and experts, head, all-reduce) on the prefill entries at the
bucket's rows; the per-item kernels (QSA, linear-attention verify entries,
PLE) run the whole bucket. Rows past the live count do no work or write rows
nothing reads.

## State

- **KV pages**, 64 tokens each: for each full layer K, V and the QSA block
  keys, layer after layer. Every package declares `qwen38.qsa-paged-bf16.v1`
  (835,584-byte pages) and `qwen38.qsa-paged-fp8.v1` (K and V as FP8 E4M3 at
  scale 1.0, block keys still BF16, so block selection is unchanged). With
  `--drafter mtp` they are `qwen38.qsa-paged-bf16-mtp.v1` and
  `qwen38.qsa-paged-fp8-mtp.v1`, whose drafter layer stays BF16. `serve
  --kv-cache-dtype fp8` selects FP8; `auto` (default) serves BF16.
- **QSA raw-key ring** (`qwen38.qsa-ring.v1`): per full layer, every slot's
  16 raw keys and their positions, plus a saved copy. Decode and verify rows
  never overwrite a committed open block; a prefill chunk can, so a step with
  a prefill saves the ring and cancel restores it. Its bytes are per state
  slot (`chop.state-per-item.v1`).
- **Linear state** (`chop.recurrent-state.v1`): per layer, a pool with one
  committed entry per served sequence of convolution history and F32
  delta-rule state. AR and prefill update the slot in place; verify items
  write a step checkpoint pool and commit copies the kept checkpoint. Because
  updates are in place, a cancel invalidates every touched slot and the host
  retires those requests (`chop.recurrent-state.in-place.v1`). A request
  whose sampling fails does not cancel the step: the step commits and only
  that request fails and retires.
- **PLE state**: one entry per slot (the 9-row convolution history and two
  n-gram ids). A step stages every item's entry, the kernels write a staged
  output, and commit copies it back.
- Retirement frees the slot; its next request starts fresh.

Both pools are allocated for every served slot during create, before the
serving KV pool (`chop.recurrent-state.preallocated.v1`), so the host does not
reserve them a second time. Per slot and rank at TP4 they are 552,960 bytes of
convolution, 28,311,552 of delta rule and 184,448 of PLE. The instance sizes
everything by the create record's served count (`chop.served-items.v1`, the
host's `--max-sequences`). See
[serving-recurrent-state.md](serving-recurrent-state.md) for the host side.

### Limits

- At most 256 sequences (`--max-sequences`), 128 with MTP
  (`MTP_QUALIFIED_SEQUENCES`), fewer where a rank cannot fit them.
- Verify steps: at most 8 rows per item (`MAX_VERIFY_ROWS`) and 32 drafting
  items (`DRAFT_ITEMS`). Instances serving at most 32 sequences verify at
  most 32 rows a step; wider ones up to 256
  (`chop.speculative-narrow-rows.v1`).
- At most 512 scored logits rows in a prefill phase (`SCORED_ROWS`, manifest
  `batch_limits.maximum_scored_rows`).
- A prefill chunk that stops short of the prompt end, and every public
  prefix-cache anchor, falls on a multiple of 64 tokens (`PREFILL_GRANULE`,
  the delta rule's chunk, manifest `batch_limits.prefill_granule`), so a cold
  prefill gives the same bits under any chunking. `--prefill-chunk-tokens`
  must be a multiple of 64.
- Images only through the vision encoder.

## Vision encoder

Image records run the 27-block vision tower and the 2 x 2 merger, and
prefill replaces each placeholder row's embedding with its merged row. The
encoder is replicated: every rank holds the whole encoder image, runs the
whole tower and keeps its own arena, because every rank gathers the whole
embedding. The request surface is in
[serving-request-surface.md](serving-request-surface.md).

### Create

The create configuration names the encoder image
(`qwen38.encoder-rank-images.v1`) and one index per rank
(`chop.bundle-metadata.qwen38.encoder-rank-index.r{rank}.v1`), or none of
them; a partial set is refused (`encoder::configured`). Without them the
bundle refuses any step with media records (`Unsupported`). With them,
`encoder::acquire` checks each rank's index against the contract's
`encoder::views::layout`, opens the image, matches the index's
`direct_identity` against the host's, and creates the shared encoder runtime
from `models/qwen38/contract/encoders/qwen38-vision.json`, whose buffer plan
sizes every region for one image of 65,536 patches (F32 activations). A
package with the encoder declares `chop.media-inputs.v1` and
`chop.position-axes.v1`.

### One image record

The shared `media::MediaArena` validates the step's media section and the
declaration's recipe admits each record: `[1, h, w, 0]` with even `h` and
`w`, `h * w` at most 65,536, `h * w / 4` output rows. A malformed section is
refused (`InvalidArgument`) before any device work. Each record then uploads
its F32 patch rows and host-computed side inputs
(`bundle_runtime::encoder::side`), runs the patch GEMM, 27 blocks and
the merger (generated `chop_enc_*` entries; every tensor-core operand a BF16
`hi + lo` pair of its F32 value, see [numerics](numerics.md)) and copies the
BF16 merged rows to the record's arena pages. Encoders run before the step's
transaction opens and before the prefill saves the QSA ring, so a failure
(`InternalError`) needs no restore.

Prefill runs `Stage::MediaInject` between `Embedding` and `ExpandStreams`: it
copies the arena row named by each live row's source into the embedding rows
(-1 keeps the token embedding). It launches only in steps with injects, so
text-only prefills and decode graphs are unchanged. Placeholder token IDs
still reach PLE and the n-gram hash.

### Encoder diagnostics

The shared encoder variables apply (see [bundle-runtime.md](bundle-runtime.md)).
Their captures record encoder execution, not request commit; the
`encoder_parity` gate reads the diagnostic-taps `enc.vision.out` tap instead
([encoder parity](../tools/qual/README.md#encoder-parity)).

## Prefix caching

Prefix caching is on by default in every Qwen profile; `--prefix-cache off`
disables it for diagnosis. The manifest requires `chop.prefix-cache.v1` and
`chop.prefix-cache.state-slot.v1`. Anchor placement is the host's
([serving KV pool](serving-kv-pool.md)). A hit is bitwise equal to a one-shot
cold prefill: the prefill kernels fix each prompt row's reduction order by
its absolute position, and chunks and anchors sit on multiples of 64. Tested
prefix-cache coverage is TP4 text prompts, AR and MTP; media and TP2
are outside that coverage. Separately, the `v0.1.0-alpha.2` validation
(2026-10-03) passed text/image checks at TP4 and TP2, AR and MTP; see the
[validation summary](../README.md#what-it-runs) and
[measured speed](serving-profiles.md#measured-speed).

### What an anchor holds

An anchor at boundary `B` shares the host pages of `[0, B)` and snapshots the
committed per-request state after token `B - 1`. Per rank at TP4:

| Piece | Where | Bytes |
|---|---|---|
| K, V and QSA block keys, 12 full layers (and the drafter layer) | host pages, shared | none |
| Linear convolution history, committed entry, 36 layers | snapshot | 552,960 |
| Gated delta-rule state (F32), committed entry, 36 layers | snapshot | 28,311,552 |
| PLE entry | snapshot | 184,448 |
| QSA raw-key ring, the slot's live copy | snapshot | 51,456 |
| With MTP: the drafter layer's ring slot | snapshot | 4,288 |
| With MTP: the slot's eight retained target X rows | snapshot | 163,840 |

That is 35 serving pages per anchor without the drafter and 33 of the larger
MTP pages with it. A 32-byte header in the last page's slack (magic
`QW38PFX2`) records the committed token count, with the drafter its frontier,
and the capturing request. Step checkpoints and the ring's saved copy are
never captured: a capture follows a commit.

### Capture and restore

`src/prefix.rs` lays the pieces end to end, and `chop_qwen38_prefix_copy_v1`
(in `cuda/qwen38_bundle.cuh`, which mirrors it; `kernel_pieces` refuses a
table that differs) copies them between a slot and the pages uploaded to
`prefix_pages`. `src/prefix_state.rs` drives it on every rank through the
shared `snapshot::copy_prefix`:

- **Capture** (`capture_prefix_state_v1`, after a prefill commit, or at a
  decode boundary before a preemption retires the request): the shared ledger
  finds the request's slot and refuses unless the committed state covers
  exactly `tokens`. The kernel copies the committed linear and PLE entries,
  the live ring and the retained drafter rows; rank 0 writes the header.
- **Anchors inside a prefill step.** A capturing prompt's end anchor is the
  last multiple of 64 at or below its end. Up to `ANCHOR_ITEMS` (2)
  prompt-ending items a step (`batch_limits.anchor_items`) keep their state
  at that anchor in spare entries beside the slots while the chunk runs past
  it, so a capturing prompt runs no extra step. A further such item ends a
  chunk at the anchor instead.
- **Drafter rows stay private.** A continuation first reruns drafter rows from
  the frontier (at least row `B - 1`), which rewrites the anchor's last
  drafter row. MTP packages declare `batch_limits.rewrites_last_page`, and the
  host copies the page holding row `B - 1` before an append. A
  decode-boundary capture whose frontier lags into an earlier page restores
  only for its own request (`prefix::may_restore`); others prefill cold.
- **Restore** needs the request's state slot, because the snapshot pages are
  free again when the callback returns. The host restores through
  `restore_prefix_state_in_slot_v1` with the slot it assigned at
  `admit_request_v1` (`restore_prefix_state_v1` alone returns
  `Unsupported`). Restore writes the slot's committed entry, reads the header
  back and refuses a token count or drafter presence that does not match,
  then records the owner and token count in the ledger and, with the
  drafter, its frontier and token history.
- Nothing is captured or restored while a step is open. Every launch is built
  before any rank submits and every rank that received work is drained; a
  failed copy leaves the slot free, so a restore failure falls back to cold
  prefill. A drain failure returns `InternalError`.

Image positions do not need snapshot state: the host sends every row's
M-RoPE positions over `chop.position-axes.v1`.

### Accuracy gate

The gate is bitwise equality with cold prefill ([numerics](numerics.md)). The
correctness runner's determinism records (the `tools/qual/determinism.py`
request set, taken by `chopinfer eval capture --kind determinism`) check hits
and chunk sizes on every run; `tools/qual/prefix_exact.py` checks every page
boundary of the exact reference.

With `CHOPINFER_QUAL_CONFIG=models/qwen38/qual.json`, `prefix_exact.py
record` (exact reference from the gate config, or `--exact
fixtures/qwen38/teacher-reference.json`) records cold, anchor-chunked and hit
rows at every scored position and requires hit and chunked rows to equal cold
bit for bit; expected `cached_tokens` follow from
`qualification.prefix.prefill_granule` (64). Run it with `--help` for the
prompt-end and `--split` arms. On the fake host, capture, restore into another
slot and capture again give byte-identical snapshots
(`fake_host_tests/snapshot_dump.rs`).

## MTP speculation

A package built with `qwen38-package --drafter mtp`
([model-qwen38.md](model-qwen38.md#packaging)), or prepared by `serve --model`
at a rank count with an MTP profile, declares the bundle as drafter.
`tools/dev/qwen38-redeploy.sh --drafter mtp` serves it with
`--speculative-tokens 3 --speculation adaptive`. Without the drafter the
package declares host-drafted speculation, enabled only with
`--speculative-tokens`. The bundle exports `commit_accepted_v1`.

### Verify rows

A decode phase with verify items runs as a graph bucket of rows and items
through the decode program; most stages are per row already. The stateful
ones:

- Linear attention: drafting items (at most 32; the host policy rotates
  eligible items within the bound, the rest decode as AR) write a step
  checkpoint pool, and `commit_accepted_v1` copies each kept checkpoint into
  its slot. Single-row AR siblings update their slots directly.
- PLE: `chop_qwen38_ngram_verify_v1` and `chop_qwen38_ple_conv_verify_v1` run
  every item in one launch each, and `chop_qwen38_ple_checkpoints_v1` writes
  the PLE entry after each row; commit copies the kept one, so rejected
  tokens never reach the n-gram history.
- The QSA ring needs no rollback: a verify step cannot overwrite committed
  keys, and a rejected row's entries are rewritten before any row reads them.

### The drafter

The drafter's layer runs as `drafter_stages` through the main lowering, then
the shared head on the draft rows and a sharded argmax. Its K, V and block
keys are a thirteenth full layer of each page and of the ring.

Drafter row `p` reads the target's final X of token `p` and token `p + 1`.
`drafter.rs` keeps each slot's frontier (rows that ran) and eight retained
target X rows:

- a drafting verify item runs rows `frontier..c` before the target, then one
  chained row per further draft;
- a continuing prefill runs the rows its earlier steps left, then every row
  its chunk completes;
- a plain decode item runs nothing until its retained rows would overflow,
  then catches up eight rows at once.

Runs of at most 32 rows replay graphs; larger runs, and every run with
`CHOPINFER_QWEN38_MTP_EAGER=1`, run eagerly. Drafts are read back from rank 0
and emitted before any logits. The drafter layer's ring is saved and restored
on cancel. The drafter's head stays BF16: its argmax only proposes drafts,
and verify scores them with the F32 head.

## Kernels

- **GEMMs.** `lower.rs` binds every GEMM to an op prepared at create: dense
  groups on Qwen's own kernels (`OwnOps::dense_group`), shared and expert
  projections on the target's backend.
  Prefill q, k and v run as one `attn_qkv_prefill` launch over the contiguous
  `q_proj`, `k_proj` and `v_proj` (`RankStage::packed_qkv`).
- **Logits.** Every logits row the host samples or scores is F32. The head
  runs `head_f32_m{M}` at the smallest decode bucket that holds its rows, and
  `head_f32_prefill` into `local_scored` for scored prompt rows.
- **Linear attention** runs `kernels_sm120::linear`, and the gated rows the
  `attention_qwen38_sm120` cubin, through the SM120 backend's prepared ops.
- **Encoder** kernels come from the vision declaration; see the
  [encoder kernel guide](../crates/kernels-sm120/src/encoder/README.md).

`CHOPINFER_QWEN38_HC_PRECOMPUTE` controls hyper-connection matrix
precomputation for decode buckets 16 and 32 (on by default; `0`, `false` or
`off` disables it for diagnostics).

## Diagnostic tools

- **Taps.** Export `CHOPINFER_TAPS_DIR=<dir>` (optionally
  `CHOPINFER_TAPS_LAYERS=0,1,3` or `all`) before `qwen38-redeploy.sh`, which
  then builds the bundle with its `diagnostic-taps` feature; other builds add
  `--features bundle-qwen38/diagnostic-taps`. A serving bundle refuses to
  create while a taps directory variable is set. Every phase then runs
  eagerly and writes `<dir>/step<NNNN>/` (`runtime_bundle_sdk::taps`,
  declared in `models/qwen38/taps.json`) under the reference runner's tap
  names (`L03.out`, `L03.x.indexer.topk`, `final.x.hc.mixer`, `enc.vision.out`).
  `CHOPINFER_TAPS_SITES` (comma-separated patterns, `*` matching any text)
  and `CHOPINFER_TAPS_FROM=P` limit taps and rows; invalid values refuse the
  create. The older `CHOPINFER_QWEN38_DUMP_*` names still apply.
- **Layer check against the reference runner.** Run the
  [GPU reference runner](reference-runner.md) over one prompt with
  `--modes f32,bf16 --taps --tap-sites all`, serve the diagnostic-taps build
  alone with prefix caching off and `CHOPINFER_REQUEST_TRACE=1`, then:

  ```sh
  python3 -B tools/qual/qwen38_layers.py capture --url URL --server-log LOG \
    --taps-dir DIR --reference RUN --out DUMP
  python3 -B tools/qual/qwen38_layers.py --dump DUMP --reference RUN --url URL
  ```

  The second command judges each tap against the runner's F32 taps with its
  BF16 taps as the floor and names the first divergence. `qwen38-oracle
  batch-drift` compares one prompt's rows between two dumps
  ([qwen38-oracle.md](qwen38-oracle.md)).
- **Teacher-forced log-probabilities:** `chopinfer eval run --model
  models/qwen38 --url http://127.0.0.1:<port> --check teacher` gates against
  `fixtures/qwen38/teacher-reference.json`.
- **Frontend, needles and API shapes:** compare served frontend observations
  with `python3 -B tools/qual/frontend_goldens.py <dir>`. The reference uses
  HF templates, tokenizers and F32 answer shapes; see
  [the fixture protocol](../fixtures/qwen38/frontend/README.md).
- **GSM8K:** `tools/qual/gsm8k.py` with the model declaration's settings.
