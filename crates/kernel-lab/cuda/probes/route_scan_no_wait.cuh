#pragma once
#include <stdint.h>
#include "../../../kernels-sm120/cuda/common/ptx_control.cuh"

// Lab-only control for the decode route's PDL wait: the served source of
// `moe/route_local_scan.cuh` (and the route headers it includes), compiled
// with every `ptx::griddep_wait()` call made a no-op. The PTX helpers are
// defined first, unchanged; the macro then renames only the calls in the
// route headers included below. A cubin holding this probe holds nothing
// else, so no served entry sees the macro.
namespace chop::ptx {
__device__ __forceinline__ void griddep_wait_skipped() {}
}  // namespace chop::ptx

#define griddep_wait griddep_wait_skipped
#include "../../../kernels-sm120/cuda/moe/route_local_scan.cuh"
#undef griddep_wait
