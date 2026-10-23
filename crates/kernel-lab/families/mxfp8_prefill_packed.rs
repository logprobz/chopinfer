//! TMA prefill tiles reading `mxfp8-fold-packed-8x128-v1` weight codes,
//! beside the row-major entries they must equal bit for bit.

use kernel_build::CubinSpec;
use kernel_core::contract::DType;
use kernel_core::mxfp8_gemm::Mxfp8FoldGemmQuery;
use kernel_core::precision::execution::BlockScaledExecution;
use kernels_sm120::gemm::{mxfp8_fold::Entry, mxfp8_prefill};

pub const FAMILY: &str = "mxfp8_prefill_packed";
pub const CUBIN: &str = "chop_mxfp8_prefill_packed_lab";

/// One lab fold entry named for `family`. Activation groups are `k + pad`
/// apart and rows `groups * (k + pad) + pad`, so a nonzero pad puts padding
/// between groups and after the last.
pub fn entry(
    family: &str,
    (capacity, n, k, groups, pad, output): (u32, u32, u32, u32, u32, DType),
) -> Entry {
    let q = Mxfp8FoldGemmQuery {
        m_capacity: capacity,
        n,
        k,
        groups,
        activation_group_stride: k + pad,
        activation_row_stride: groups * (k + pad) + pad,
        output,
        execution: BlockScaledExecution::NativeBlockScaledMma,
    };
    let output = if output == DType::F32 { "f32" } else { "bf16" };
    Entry::new(
        q,
        format!("chop_{family}_lab_m{capacity}_n{n}_k{k}_g{groups}_p{pad}_{output}_v1"),
    )
    .unwrap()
}

/// Both step bodies (K below 1024 in complete folds, and the rest), a final
/// 64-K fold through the full-tile path (dense rows) and through the
/// boundary path only (padded rows), partial column tiles, several column
/// tiles, padded groups and both output types.
pub const SHAPES: [(u32, u32, u32, u32, u32, DType); 6] = [
    (256, 200, 576, 1, 0, DType::F32),
    (256, 128, 256, 2, 0, DType::Bf16),
    (256, 256, 2048, 1, 0, DType::Bf16),
    (256, 136, 512, 3, 128, DType::F32),
    (256, 64, 576, 2, 32, DType::Bf16),
    (256, 64, 192, 1, 0, DType::F32),
];

/// Wide prefill projections for the paired timing test.
pub const TIMING_SHAPES: [(u32, u32, u32, u32, u32, DType); 3] = [
    (8192, 5120, 2048, 1, 0, DType::Bf16),
    (8192, 8192, 1280, 1, 0, DType::Bf16),
    (8192, 5120, 576, 1, 0, DType::Bf16),
];

pub fn entries() -> Vec<Entry> {
    SHAPES.into_iter().map(|s| entry(FAMILY, s)).collect()
}

pub fn timing_entries() -> Vec<Entry> {
    TIMING_SHAPES
        .into_iter()
        .map(|s| entry(FAMILY, s))
        .collect()
}

pub fn cubins() -> Vec<CubinSpec> {
    let entries = [entries(), timing_entries()].concat();
    let mut spec = mxfp8_prefill::cubin_spec(CUBIN, &entries).unwrap();
    for e in &entries {
        spec.entries.push(mxfp8_prefill::packed_spec(e).unwrap());
    }
    crate::declared(&[CUBIN], vec![spec])
}
