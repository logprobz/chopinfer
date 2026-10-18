# DeepSeek V4.1 Flash (`deepseek41`)

DeepSeek V4.1 Flash, from the official checkpoint
`deepseek-ai/DeepSeek-V4.1-Flash-1017` at revision
`1bae0e1088993b9b82de08f59de5be5a34e42170`, served as bundle
`deepseek41.v1` at topology `deepseek41-flash-tp4`. It is not released:
`model.toml` declares `[release] enabled = false` until it qualifies.

| Crate | Directory | Holds |
|---|---|---|
| `model-deepseek41-contract` | `contract/` | IDs, geometry, projection declarations and kernel contracts the compiler, bundle and lab share |
| `model-deepseek41` | `compiler/` | Checkpoint admission, the tokenizer contract and the package |
| `bundle-deepseek41` | `bundle/` | The program; SM120 registration in `src/sm120/` |
| `model-deepseek41-lab` | `lab/` | Kernel GPU correctness tests on `kernel-harness`; not a default workspace member |

Build and test the roles with Cargo's `--locked --offline` flags:

```sh
cargo test --locked --offline -p model-deepseek41-contract -p model-deepseek41 -p bundle-deepseek41
```

The lab runs only when named, and its GPU tests only when
`CHOPINFER_KERNEL_GPU` or `CHOPINFER_KERNEL_GPUS` names idle GPUs:

```sh
cargo test --locked --offline -p model-deepseek41-lab --test sm120
```

CPU checks that read the checkpoint take it from
`$CHOPINFER_DEEPSEEK_CHECKPOINT`; the tokenizer corpus test also reads the
HF corpus `tokenizer_goldens.py corpus` writes, named by
`$CHOPINFER_DEEPSEEK_TOKENIZER_CORPUS`.

## Checkpoint and geometry

`checkpoint-files.json` pins every file of the official revision, for
provenance and the qualification binding. Admission reads `config.json`
alone: aliases, semantic rules, the block-32 FP8 recipe, the CSA2, Engram
and DSpark relations (`contract/src/structure.rs`), then the closed match
against `geometries/flash.json`, each refusal naming its field. The
synthetic geometry (`fixtures/deepseek41/geometries/synthetic-small/`, 12
layers carrying every CSA2 role, 32 experts, one DSpark stage) admits at
TP1, TP2 and TP4 when `CHOPINFER_DEEPSEEK41_GEOMETRY_DIRS` names its
directory.

`contract/projections.json` declares every projection, stem, side tensor
and encoder tensor once; `checkpoint::bind` binds a checkpoint's
safetensors headers to it before any payload is read. The official
revision binds all 96,085 tensors, each claimed once
(`compiler/tests/checkpoint_binding.rs`).

## Reference

`reference.py` is the refrun adapter (`tools/refrun`). It executes the
checkpoint's own `inference/` code (sha256-pinned) with the tilelang kernels
replaced by `reference_kernel.py`, whose ledger records each replacement and
its result against the executed vendor kernel
(`reference_vendor_control.py`). `REPLACEMENTS` lists everything else it
changes: the row-blocked prefill of `reference_memory.py`, ties to the lower
index in the indexer and router top-k, the RoPE tables the vendor computes
on CUDA, Engram rows gathered on the host, and the vision tower on its own
device. The F32 mode rounds quantizer inputs to BF16 (F32-TQ, the gate
variant); F64 widens the vendor code except its RoPE angle lines.
`reference_pure_f32.py` is the same adapter without that rounding.
`test_reference.py` checks it bitwise against the vendor's
`convert.py`, `load_model` and `Transformer.forward` on a tiny checkpoint.

## Qualification

The model is not qualified. `qual.json` is a skeleton: its checks carry
only their pass and exactness rules, and every numerical limit is pinned from the spread
of the refrun references of the pinned vendor code before the first
candidate run, never moved after a result:

- teacher, chat and long_context (131,072 tokens): gated against the
  F32 run with BF16-rounded quantizer inputs (F32-TQ); mean |dlogprob|
  and top-1 limits from the BF16 versus F32-TQ and F32-TQ versus F64-TQ
  spreads, with complete scores for the same items;
- per-layer taps (`taps.json`) within the same spread;
- the `kv-bf16` comparison configuration (`serve --kv-cache-dtype bf16`):
  the BF16 KV layout holds the same values as the native trained formats,
  and native versus BF16 served bits must be equal. It runs prefix_exact
  and kv_layout_equality, which records the determinism request set on a
  native server and on the configuration's BF16 server in one run and
  requires every request's tokens and logprobs to be bit-identical, naming
  the first differing position; teacher, teacher_arms and chat join as
  they land. It is never a served default and not qualified for capacity,
  preemption under pressure or offload, since its pages are about 3.6
  times larger (3,200 bytes against 890 per token).

The needle check runs frozen token-ID prompts at total contexts of 8,192,
131,072, 262,144 and 1,048,576 tokens (7,168, 130,048, 261,120 and
1,047,552 prompt tokens, each with 1,024 tokens to answer) and requires the
three needle values in order. `fixtures/deepseek41/capture/needles.py`
builds them on CPU from MiMo's functional haystack, rendered by the
checkpoint's `encoding/encoding.py` in chat mode, with the needles near
10%, 50% and 90% of the haystack, into
`$CHOPINFER_DATA/deepseek41/needle/prompts.json`; the digests are in
`fixtures/deepseek41/needles.json` (`--check` rebuilds and compares).

`profiles.json` holds the experimental `tp4-ar` bring-up profile. The
release preview's full suite at the release revision is the
qualification evidence.
