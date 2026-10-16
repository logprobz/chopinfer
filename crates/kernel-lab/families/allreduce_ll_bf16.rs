//! The low-latency all-reduce of BF16 partials from F32 inputs
//! (`comm::ll_bf16::lab_cubin_specs`): the served entry and its mutants.
//! Its tests run behind the PDL producer of `allreduce_ll_f32` and compare
//! with that family's F32 entry.

use kernel_build::CubinSpec;

pub const FAMILY: &str = "allreduce_ll_bf16";

pub fn cubins() -> Vec<CubinSpec> {
    kernels_sm120::comm::ll_bf16::lab_cubin_specs()
}
