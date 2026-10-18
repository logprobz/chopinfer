//! The clamped SwiGLU with BF16 rounding and ceil E4M3/32 quantization's
//! lab cubin (`row::silu_mul_quantize::lab_cubin_specs`).

use kernel_build::CubinSpec;

pub const FAMILY: &str = "silu_mul_quantize";

pub fn cubins() -> Vec<CubinSpec> {
    kernels_sm120::row::silu_mul_quantize::lab_cubin_specs()
}
