//! Staged-scale TMA prefill tiles reading `mxfp8-fold-packed-8x128-v1`
//! weight codes, beside the row-major entries they must equal bit for bit.
//! The entry needs complete 128-K folds, so no shape has a final 64-K fold.

use crate::mxfp8_prefill_packed::entry;
use kernel_build::CubinSpec;
use kernel_core::contract::DType;
use kernels_sm120::gemm::mxfp8_prefill_tma_scales::{self as tma_scales, Entry};

pub const FAMILY: &str = "mxfp8_prefill_tma_scales_packed";
pub const CUBIN: &str = "chop_mxfp8_prefill_tma_scales_packed_lab";

/// One and several folds, partial and several column tiles, padded groups
/// and both output types.
pub const SHAPES: [(u32, u32, u32, u32, u32, DType); 4] = [
    (256, 200, 1280, 1, 0, DType::F32),
    (256, 128, 512, 2, 0, DType::F32),
    (256, 136, 512, 3, 128, DType::Bf16),
    (256, 64, 128, 1, 0, DType::F32),
];

/// Wide projections this entry serves at 4096 and 8192 rows.
pub const TIMING_SHAPES: [(u32, u32, u32, u32, u32, DType); 3] = [
    (8192, 5120, 2048, 1, 0, DType::Bf16),
    (8192, 8192, 1280, 1, 0, DType::Bf16),
    (8192, 1280, 5120, 1, 0, DType::Bf16),
];

pub fn entries() -> Vec<Entry> {
    SHAPES
        .into_iter()
        .map(|s| Entry {
            base: entry(FAMILY, s),
        })
        .collect()
}

pub fn timing_entries() -> Vec<Entry> {
    TIMING_SHAPES
        .into_iter()
        .map(|s| Entry {
            base: entry(FAMILY, s),
        })
        .collect()
}

pub fn cubins() -> Vec<CubinSpec> {
    let entries = [entries(), timing_entries()].concat();
    let mut spec = tma_scales::cubin_spec(CUBIN, &entries).unwrap();
    for e in &entries {
        spec.entries.push(e.packed_spec().unwrap());
    }
    crate::declared(&[CUBIN], vec![spec])
}
