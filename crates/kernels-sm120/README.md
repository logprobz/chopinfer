# kernels-sm120

The SM120 CUDA engines and typed launch planners. Start with
[kernel contribution rules](../../docs/kernel-library.md). Each engine's
local README owns its ABI and arithmetic requirements. Target limits are in
`src/device.rs`; catalogs and tuning belong to their model.

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
