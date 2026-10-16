# kernels-sm120

SM12x CUDA engines, launch planners, instantiation catalogs and tuning
tables. The library overview is [docs/kernel-library.md](../../docs/kernel-library.md);
each engine family documents its ABI, schedules and numerics beside its
sources:

| Family | Reference |
|---|---|
| Dense GEMM | [cuda/gemm/README.md](cuda/gemm/README.md) |
| Grouped expert GEMM and routing | [cuda/moe/README.md](cuda/moe/README.md) |
| PCIe all-reduce, all-gather and logits exchange | [cuda/comm/README.md](cuda/comm/README.md) |
| Paged, verify and gated attention | [cuda/attention/README.md](cuda/attention/README.md) |
| Block-sparse attention with an indexer | [cuda/attention/SPARSE.md](cuda/attention/SPARSE.md) |
| Linear attention (gated delta rule) | [cuda/linear/README.md](cuda/linear/README.md) |
| Row ops (RMSNorm, RoPE, quantize, logits, DFlash, hyper-connection rows) | [cuda/row/README.md](cuda/row/README.md) |
| Sinkhorn-mixed hyper-connections (mix-collapse, expand, stream head) | [cuda/sinkhorn_hc/README.md](cuda/sinkhorn_hc/README.md) |
| Multimodal encoder kernels | [src/encoder/README.md](src/encoder/README.md) |

Tuning tables live in `tuning/` (`schedules.json` holds the shared
grouped-MoE rows); their format is described in
[kernel-build](../kernel-build/README.md#tuning-tables). Kernels are tested on
hardware by [kernel-lab](../kernel-lab/README.md).

## Common device layer

`cuda/common` contains device-only PTX wrappers for warp MMA, matrix loads
and stores, TMA, barriers, cluster control, packed conversions,
shared-memory swizzles and a staged TMA pipeline.
Sources target `sm_120f`. The `cuda/tests` kernels are compiled and
inspected by `tests/compile_common.rs`; they are never launched.

Cluster sizes above one, TMA multicast, DSMEM and gather4 compile but are
probe-gated: an engine that uses them validates them in kernel-lab first.
The pipeline uses one producer warp and a fixed number of consumer warps.
One elected lane per warp performs each barrier operation. The caller
synchronizes after barrier initialization and primes each empty stage before
the producer starts.

Packed FP4 tensor maps use `CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B` and a
`shared::cluster` TMA load, even for a single CTA. TMA tiles must respect
the 101376-byte per-block shared-memory limit. Descriptor prefetch uses
`prefetch.tensormap`. F32 to UE8M0 packing supports native `rz` and `rp`
rounding, and BF16x2 to UE8M0 uses native `cvt`. PTX has no direct UE8M0
conversion to or from FP16x2 on SM120, so only those helpers go through
scalar FP16 and FP32 conversions.

No CuTe or CUTLASS header is included.

## Model-owned construction

`row::rmsnorm::RmsNorm` constructs the BF16 RMSNorm engine from explicit
`kernel-core` requirements without a `Profile`. Its validated selection
supplies entry code, semantic declarations, launch plans and scalar
arguments. The row catalogs delegate their RMSNorm entries to it. Other
families keep their catalog construction paths. See
[docs/kernel-library.md](../../docs/kernel-library.md) (model-owned selections).
