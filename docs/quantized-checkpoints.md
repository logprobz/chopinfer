# Quantized checkpoints and operation precision

A family has one architecture implementation. Shared SDK importers interpret
checkpoint encodings, and each operation binds supplied kernels for its exact
geometry and precision. Qwen is the first consumer of mixed precision; MiMo
uses the same importer for its fixed encodings. GLM 5.3 Flash (`glm5next`)
imports its block-FP8 and NVFP4 exports through it too and lowers its dense
projections through the shared declaration adapter. This extends
[compatible checkpoint serving](compatible-checkpoints.md) and the
[external model SDK](external-models.md). Preparation and loading never compile
CUDA.

## Activation policy

The A16 standard runs every projection with BF16 activation operands on the
A16 engines. FP8- and FP4-activation MMA implementations are optional
performance variants that a model enables only after passing its own
accuracy gate.

A checkpoint trained with activation quantization must keep its quantization
points. The planned A16 implementation adds an exact rounding stage at those
points so the BF16 operands lie on the trained quantization grid. This stage
is not implemented; it is planned with the first model that needs it. The
current policies below describe the implemented paths and their refusals.

`--activation-policy auto|checkpoint|bf16` selects the policy when a Qwen
package is emitted, and defaults to `auto`. `serve --model` and the helper's
`prepare` take only the option. `qwen38-package` also reads
`CHOPINFER_ACTIVATION_POLICY`, which the option overrides. MiMo has one
precision and refuses the option.

| Policy | Resolution |
| --- | --- |
| `auto` | Prefer the declared recipe only among implementations enabled by the model's accuracy gate. Never select non-BF16 activations without that gate. If any required use is unavailable, select BF16 only through an implemented weight-reuse rule and report the fallback reason. |
| `checkpoint` | Require the declared activation recipe, calibration and transforms for every use, including the model's gate for non-BF16 activations. Refuse unavailable implementations without substituting BF16. |
| `bf16` | Require BF16 activation operands for projections. Refuse incompatible overrides and hidden FP8/FP4 activation conversion. |

The model publishes only gated activation variants in its supplied inventory;
`auto` and `checkpoint` do not grant that gate themselves.

One logical operation has one effective recipe across prefill, decode, verify
and every graph capacity; batch size cannot change activation precision.
Accumulators, reductions, routing scores, logits and recurrent state keep their
declared contracts, and KV cache precision is selected separately
(`--kv-cache-dtype`). The importer refuses an export's KV cache hint
(`kv_cache_scheme`, or a non-null `kv_cache_quant_algo`) unless the family
imports under the record-and-ignore KV policy, which records the static FP8
hint and imports nothing from it; every other hint form is still refused.
An emitted package serves the precision it records:
`serve --bundle` refuses `--activation-policy`.

Required pre-scales, rotations, AWQ/SVDQuant and other unimplemented transforms
cannot be bypassed with `bf16`, and calibration is never invented. Reusing W4A4
weights under BF16 activations (W4A16) changes execution semantics and does not
inherit the source recipe's quality or native W4A4 throughput.

## Qwen coverage

| Recipe | Text topology | `auto` and `bf16` | `checkpoint` |
| --- | --- | --- | --- |
| Compatible block-FP8 | TP2/4 | FP8 fragment weights and BF16 expert operands; `auto` reports plain-FP8 reuse | Refused: the dynamic-FP8 expert recipe is not supplied |
| NVIDIA mixed NVFP4 export | TP2/4 with MTP | NVFP4 routed experts, FP8 block-128 MTP experts, FP8 PLE, BF16 elsewhere | Refused: per-expert calibrated NVFP4 activations |
| local-inference-lab mixed NVFP4 export | TP2/4 with MTP | MXFP8 attention, linear-attention and shared-expert projections; NVFP4 routed experts and PLE; W4A16 NVFP4 MTP experts; F32 HC and PLE norm weights; BF16 activation operands | Refused: unavailable source-activation compositions |

Export names identify inspected examples, not allowlists; admission follows
tensor and operation contracts. A mixed checkpoint runs the MTP drafter when
its MTP layer has a composition for every format it stores: BF16 dense
projections, and experts the main layers' grouped compositions bind.
Otherwise the package serves autoregressively and `--drafter mtp` is
refused, naming the tensor and its format. The local-inference-lab export
stores its 100 HC and PLE norm weights as F32; they are bound by their view
dtype, and the communication all-reduce and RMSNorm epilogue that consume
them run in F32 forms. NVIDIA's BF16 vision tower is planned as the encoder
image; quantized vision is refused. FP8 packages keep their encoders and MTP
speculation.

```sh
chopinfer serve --model "$QWEN_MIXED_CHECKPOINT" --tensor-parallel-size 2 \
  --activation-policy auto
chopinfer serve --model "$QWEN_MIXED_CHECKPOINT" --tensor-parallel-size 4 \
  --activation-policy bf16
```

For an explicit package, pass `--activation-policy` to `qwen38-package` with
the checkpoint and `--module`; `--precision-component FILE[,FILE...]` reuses a
precision record resolved earlier, which packaging checks against the
checkpoint and refuses on conflict.

Measured on four RTX PRO 6000 Blackwell GPUs at TP4: packed weights take about
180 GiB (FP8), 128 GiB (NVIDIA NVFP4) and 105 GiB (local NVFP4/MXFP8) across
the four ranks (2026-09-28). With packed ring decode for NVFP4 experts, NVIDIA
NVFP4 decode was within about 0.5% of FP8 at C1 and faster at C4 to C32, and
both scored 195/200 on GSM8K (2026-09-30). Different weight quantizations are
not expected to produce identical logits.

Each mixed export is a configuration of `qwen38`: `models/qwen38/qual.json`
declares it under `configurations.<export>`, with its reference lock entries
in `models/qwen38/qual-references.lock.json`; its checkpoint lock and source
pin stay in `models/qwen38/exports/<export>/`, and its f64 refrun teacher
reference is in `fixtures/qwen38/exports/<export>/`.
Select it with `--model qwen38.<export>` in the Python qualification tools
(`tools/qual`) and with `--variant <export>` in `chopinfer eval`. Both exports
passed on a release build at TP2 and TP4 with MTP (2026-10-07):

| Check | NVIDIA | local-inference-lab | Gate |
| --- | --- | --- | --- |
| Teacher at TP2: mean abs logprob error, decisive top-one, near-tie top-one, top-one disagreement | 0.057, 0.0089, 0.0141, 0.023 | 0.043, 0.0038, 0.0117, 0.0156 | 0.1, 0.08, 0.08, 0.1 |
| verify_exact, TP2 and TP4 | K3 MTP rows equal AR top-one at all 4106 scored positions | same | exact |
| spec_verify, TP2 and TP4 | AR and MTP outputs equal on every copy and natural request | same | exact |
| Draft acceptance in spec_verify, copy/natural | TP2 0.747/0.703, TP4 0.729/0.698 | TP2 0.731/0.729, TP4 0.769/0.743 | none |
| GSM8K at TP2 with MTP | 197/200 | 195/200 | 194/200 |

The gates are in [numerics](numerics.md#serving-qualification-gates).

## MiMo V2.6 Flash coverage

MiMo's projection declaration (`models/mimo26/contract/projections.json`)
names its tensor roles; the compiler builds its importer roles from the
declaration's source names. The shared importer reads `config.json`'s
`quantization_config` and an optional `hf_quant_config.json`; packaging and
the bundle's weight plan run the same admission, and MiMo admits only what
its kernels read:

| Role | Served encoding | Released checkpoints declare it by |
| --- | --- | --- |
| Attention `qkv_proj`, layer 0 dense MLP, MTP `qkv_proj` and MLP | FP8 E4M3 128x128 blocks, F32 `weight_scale_inv` | Default block-FP8 recipe |
| Routed experts (layers 1-47) | MXFP4, UE8M0 block-32 `weight_scale` | `store_dtype: mxfp4` |
| Attention `o_proj` | BF16 | `ignored_layers` |
| Embedding, `lm_head`, routers, MTP `eh_proj` and `o_proj` | BF16 | Declared by the family |
| Norms, sinks, router bias | BF16, F32 bias | Side tensors |

Every projection runs BF16 activation operands, whatever activation recipe the
export declares. The DFlash drafter and the encoders are BF16 and checked by
their own tensor contracts. Other encodings, including NVFP4 or MXFP8 experts,
plain block-FP8 experts without `store_dtype`, and all-BF16 exports, are
refused before any weight is read, naming the tensor, its imported format and
the formats MiMo serves.

## Initial shared engine domain

| Operation | Weight formats | Activation implementations exposed |
| --- | --- | --- |
| Dense | BF16 | BF16 |
| Dense | FP8 per tensor, per channel, block 128 | BF16 and supported FP8 group-128 exact/power-of-two modes |
| Dense fragment | FP8 rows 64 | BF16 with complete fragment row groups |
| Dense | MXFP8, MXFP4 | BF16 and dynamic MXFP8 |
| Dense | NVFP4 | BF16, scalar-calibrated NVFP4, or explicitly declared dynamic-global NVFP4 |
| Grouped | FP8 block 128, exact F32 scales | BF16 and FP8 group-128 with explicit gather/quantizer stages |
| Grouped | MXFP8, MXFP4 | BF16 and dynamic MXFP8; output quantization must be declared |
| Grouped | NVFP4 | BF16 with per-expert/per-projection globals; NVFP4 activations only with bank-uniform globals and shared gate/up calibration |
| Lookup | BF16, per-tensor FP8, NVFP4 | Integer indices and BF16 output; FP8 scale dtype is explicit |

These are engine domains, not promises for every shape or fusion. Admission
uses the exact shape, phase and capacity, layout, scales, hardware and
dependencies of the supplied inventory. Activation-quantized dense SwiGLU,
arbitrary per-expert encoding mixtures and unlisted operation contracts are
refused.

The importer accepts the HF block-FP8 schema (including the dialect with
`fmt: e4m3`, `ignored_layers` and routed-expert `store_dtype: mxfp4` with
`mxfp4_block_size: 32`) and supported ModelOpt recipes for FP8, MXFP8, NVFP4
and weight-only NVFP4. It also reads ModelOpt's single-algorithm
`config.json` form: when the top-level `quant_algo` states no `group_size`,
the size comes from the config groups, which must all state it and declare
the top-level algorithm. A config group may target the `Linear` module class
instead of module paths. Its recipe then reaches only operands the family
marks as `Linear` projections (`LINEAR_CLASS_ROLE`; the shared header
adapter marks every dense and expert-bank projection, not tables), `ignore`
and the other exclusions win over it, and the top-level algorithm is not a
default for other modules. Any other class, or a class group beside groups
or layers that name module paths, is refused naming the group. Unknown
execution fields, conflicting targets, missing calibration and unsupported
transforms fail with source context. Persistent weights keep packed codes
and scales; A16 kernels decode tiles to BF16, with no
persistent full BF16 expansion.

MXFP4 and MXFP8 A16 use `ExactScaledOperand`: every code times
`2^(scale - 127)` must be finite and exactly representable in BF16. Admission
refuses any inexact operand and UE8M0 scale byte 255. Every E2M1 code is exact
at scale bytes 0 through 252; bytes 253 and 254 require checking the actual
codes. Dense and grouped MXFP4 paths use the same scaled conversion and trap
on invalid extreme-scale operands, including in cached images.

## Adding a family or exporter

Shared code owns encodings, codecs and export interpretation
(`kernel-core::precision`, `model_compiler_sdk::quantization`), one format
plug-in per weight format (`kernel_core::precision::format`), the projection
schema and its binding and derivations (`projection-spec`), offer generation
(`kernels_sm120::precision::generated`) and the shared launch
(`bundle_runtime::projection`). The family owns its projection
declaration, `contract/projections.json`, and registered variants for
arithmetic the shared schema cannot express; its program keeps operation
boundaries, routing and execution order. The serving runtime never
interprets checkpoint quantization. The
[projection-spec README](../crates/projection-spec/README.md) describes the
schema.

### A new family

1. Declare every quantized projection, fusion group, layer stem (MTP and
   drafter layers included), side tensor and use in `projections.json`.
   Shapes name geometry keys. Parse it with `projection_spec::parse` and the
   family's variant `Registry` (`Registry::default()` when it registers
   none), and pin its digest in a contract test.
2. Bind checkpoints from header metadata with `Declaration::bind`. The SDK's
   `HeaderCompanionRules` imports each source through `import_operand`;
   `quantization::declared::AdmittedRules` binds operands the family already
   imported, claiming their static input calibrations per source frame when
   the calibration tensor is shown in the `CheckpointMetadata` passed to
   binding (each calibration tensor whole, so fused operands and several
   sources can share one). To ignore an unused calibration, such as a static
   input scale under BF16 activations, leave its tensor out of that metadata;
   binding covers exactly the shown tensors.
   Binding refuses an undeclared tensor, a missing companion or an
   inadmissible shape or format, naming the tensor, before any payload is
   read.
3. Derive each rank and build the weight plan's records with
   `quantization::projection::records` and `side_records`.
   `quantization::declared::derive` composes binding over admitted operands,
   `declared::imported` and `projection_spec::derive::derive`. Call the last
   two directly to choose an import's `GeometryMode`, as the external example
   does to keep source shapes in its matrix operations. `declared::imported`
   reads separate per-expert sources, one stacked expert bank
   (`[experts, operands * rows, k]`, operands in declaration order) and
   assemblies over either, but not a stacked bank with parts or a dense
   projection whose fused operands share one tensor; build those imports
   explicitly for `derive::derive`. Grouped operands with distinct scalar
   weight globals pack one global per expert and operand, and
   `declared::admitted_activations` restores the checkpoint's activation
   recipes on the derived operations.
4. In the bundle's build, generate offers from the declaration with
   `generated::offer` (over `generated::domain` or derived operations) and
   publish audited capabilities in the native module. A bundle that keeps
   existing locked symbols for generated offers records them in
   `projection-adoptions.json`, and one that overrides default schedules
   records the overrides in `projection-schedules.json`, both beside
   `kernels.lock`; a family with neither needs no ledger.
5. Resolve precision through `resolve_operation` when the package is emitted
   (`Provider::emit_direct`) and emit the precision component. At create the
   bundle decodes it into bound handles (`projection::Store` or
   `projection::Authenticated`) and lowers each use with `projection::launch`;
   execution never searches for a different recipe.

### A new export or rank count

An export of an existing family can need no model code, but only when
everything it changes already fits the family's code:

- its tensor names and layouts match the declaration's source patterns, and
  the family's own admission (for example a catalog of expected tensors and
  side-tensor dtypes) accepts them;
- its side tensors use dtypes and views the declaration and their consumers
  already admit;
- the shared importer reads its export metadata and every recipe, and each
  operation lies inside the shared operation domain and a supplied kernel.

A new mix of the formats each port already offers is then data; the
[export recipe](testing.md#export-recipe) proves this for one synthesized
mix. A format a port does not offer is refused at preparation, naming the
operation and the missing composition, until the declaration offers it. The
change is data:

- checkpoint data: the export's identity and source pins, and metadata
  fixtures for its CPU admission tests;
- declaration data: support rules that admit the export's formats at the
  intended rank counts and phases, when the declaration currently refuses
  them; a format a port does not offer yet, added to its `offers` with the
  composition's recipe for it, when the shared kernels build that format
  (offering NVFP4 for a dense projection published only in BF16 and MXFP8,
  for example);
- generated lock records: new kernel instances compile at build time and
  append records to the bundle's `kernels.lock` (built with
  `KERNELS_LOCK_UPDATE=1`), each with kernel-lab oracle and invariance
  evidence before serving;
- adoption entries, for a family with an adoption ledger: its exporter
  appends discoveries and cubin placements to `projection-adoptions.json`
  (the in-tree models print them with their contract's
  `<m>_projection_adoptions` example). Existing records stay byte-identical.

Anything else is model adapter or catalog work: a source layout or name the
family's adapters do not map, a side-tensor dtype or view a consumer does not
admit (a new norm-weight dtype, for example, needs its consumers' kernels and
the family's admission), or an operation outside the supplied kernels. A rank
count the model does not serve yet is a topology in the model contract (its
topology list, resources and catalog profile); projection slices and offers
then follow from the declaration. Arithmetic the shared schema and plug-ins
cannot express is a namespaced registered variant (a source transform such as
a scale regroup, a stateful stage before a lookup, or an existing fused entry
kept for byte identity) with its `VariantDeriver` behavior. A new weight
format is a shared plug-in, not model code. Admission is never a
qualification: every new export needs its own numerical evidence.

Put exporter and codec tests in the shared modules, without family tensor
names. `quantization::identity::decode_component` reads a package's
per-operation precision record. The
[independent consumer](../examples/external-model/README.md) shows the same
workflow outside the tree: a declaration with no registered variants,
binding through `AdmittedRules`, the shared import and derivation of a
stacked expert bank, generated offers with an adoption ledger, and NVFP4
execution, BF16 override, fallback and refusals. Numerical contracts are in
[numerics](numerics.md).
