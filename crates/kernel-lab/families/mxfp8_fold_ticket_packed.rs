//! Ticket folds on packed codes beside their row-major twins.
use kernel_build::CubinSpec;
use kernel_core::contract::DType;
use kernel_core::mxfp8_gemm::Mxfp8FoldGemmQuery;
use kernel_core::precision::execution::BlockScaledExecution;
use kernels_sm120::gemm::mxfp8_fold_ticket::{self, Entry};

pub const FAMILY: &str = "mxfp8_fold_ticket_packed";
pub const CUBIN: &str = "chop_mxfp8_fold_ticket_packed_lab";

pub fn entries() -> Vec<Entry> {
    [
        (40, 576, 2, DType::Bf16, false),
        (72, 1280, 1, DType::F32, false),
        (2304, 576, 2, DType::F32, true),
        (1152, 2048, 1, DType::Bf16, true),
        (5120, 2048, 1, DType::F32, false),
        (8192, 1280, 1, DType::F32, false),
        (5120, 576, 1, DType::F32, false),
    ]
    .into_iter()
    .enumerate()
    .map(|(i, (n, k, groups, output, paired))| {
        Entry::new(
            Mxfp8FoldGemmQuery {
                m_capacity: 64,
                n, k, groups, output,
                activation_group_stride: k + 32,
                activation_row_stride: groups * (k + 32) + 32,
                execution: BlockScaledExecution::NativeBlockScaledMma,
            },
            format!("{CUBIN}_{i}_v1"),
            paired,
        ).unwrap()
    })
    .collect()
}

pub fn cubins() -> Vec<CubinSpec> {
    let entries = entries();
    let mut spec = mxfp8_fold_ticket::cubin_spec(CUBIN, &entries).unwrap();
    for entry in &entries {
        spec.entries.push(entry.packed_spec().unwrap());
    }
    crate::declared(&[CUBIN], vec![spec])
}
