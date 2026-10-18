//! Native dense and grouped MXFP8 folds, plus failing numerical controls.

use kernel_build::CubinSpec;
use kernel_core::contract::DType;
use kernels_sm120::gemm::mxfp8_fold::{self, LAB_CUBIN};

pub const FAMILY: &str = "mxfp8_fold";

pub fn cubins() -> Vec<CubinSpec> {
    let entries = mxfp8_fold::lab_entries();
    let mut spec = mxfp8_fold::cubin_spec(LAB_CUBIN, &entries).unwrap();
    let witness = entries
        .iter()
        .find(|e| e.query.k == 65536 && e.query.output == DType::F32)
        .unwrap();
    spec.entries.push(witness.single_chain_control_spec());
    spec.entries.push(witness.wide_fold_control_spec());
    spec.entries.push(mxfp8_fold::group_offset_control().spec());
    let scales = entries
        .iter()
        .find(|e| e.query.n == 65 && e.query.k == 160 && e.query.output == DType::F32)
        .unwrap();
    spec.entries.push(scales.scale_row_control_spec());
    crate::declared(&[LAB_CUBIN], vec![spec])
}
