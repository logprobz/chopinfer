#pragma once
#include <stdint.h>
#include "../../../kernels-sm120/cuda/common/ptx_control.cuh"
#include "../../../kernels-sm120/cuda/attention/tile.cuh"

// Lab-only PDL negative controls for the latent split merge chain
// (`families/pdl_misorder.rs`): the served sources of `latent/attention.cuh`
// (the split merge, `merge`) and `latent/fixed_splits.cuh` (the fixed-split
// attention, `attention_fixed_splits`), with the dependency wait of one stage
// made a no-op. The PTX helpers and the tile header are included first,
// unchanged; the macro then renames only the calls of the stage's own header.
// `CHOP_PDL_MISORDER_MERGE` 1 drops the merge's wait, 0 the split's. A cubin
// holding this probe holds nothing else, so no served entry sees the macro.
namespace chop::ptx {
__device__ __forceinline__ void griddep_wait_skipped() {}
}  // namespace chop::ptx

#if CHOP_PDL_MISORDER_MERGE
#define griddep_wait griddep_wait_skipped
#include "../../../kernels-sm120/cuda/latent/attention.cuh"
#undef griddep_wait
#include "../../../kernels-sm120/cuda/latent/fixed_splits.cuh"
#else
#include "../../../kernels-sm120/cuda/latent/attention.cuh"
#define griddep_wait griddep_wait_skipped
#include "../../../kernels-sm120/cuda/latent/fixed_splits.cuh"
#undef griddep_wait
#endif
