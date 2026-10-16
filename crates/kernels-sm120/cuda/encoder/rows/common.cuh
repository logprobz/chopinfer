#pragma once

// What every encoder row template shares: BF16 conversions and the
// programmatic-dependent-launch prologue. Every encoder row cubin includes
// this header, so it is frozen; a template's own helpers go in its header.

#include <stdint.h>
#include <math.h>
#include "../../row/row.cuh"

namespace chop::multimodal {

using chop::row::bf;
using chop::row::to_bf;

__device__ __forceinline__ void begin() {
  ptx::griddep_launch_dependents();
  ptx::griddep_wait();
}

} // namespace chop::multimodal
