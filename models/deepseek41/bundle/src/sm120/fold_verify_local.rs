//! Two measured-size row tiles for a wide Flash TP4 query projection.
use kernel_build::CubinSpec;
use kernel_core::{
    contract::{DType, Refusal},
    mxfp8_gemm::Mxfp8FoldGemmQuery,
    precision::execution::BlockScaledExecution,
};
use kernels_sm120::gemm::{
    mxfp8_fold,
    mxfp8_fold_local_tiled::{self, Candidate},
    mxfp8_fold_local_wide,
};
use model_deepseek41_contract::geometry::{FLASH, Geometry};
pub const FAMILY: &str = "mxfp8_fold_verify_local64";
pub const CUBIN: &str = "mxfp8_fold_verify_local64_deepseek41_sm120";
fn tuning() -> Result<&'static kernel_build::tuning::Row, Refusal> {
    super::mxfp8_fold::status_tuning("dense_fold.status_wide.verify", 64, 64).ok_or(Refusal(
        "missing required tuning row: dense_fold.status_wide.verify mxfp8 ranks=4 bucket=64 row 64",
    ))
}

pub(crate) fn selected(op: &str, rows: u32) -> bool {
    // Table loading checks these rows against the required descriptor.
    super::mxfp8_fold::status_tuning(op, 64, rows).is_some()
}

pub fn entry() -> Result<Candidate, Refusal> {
    let tuning = tuning()?;
    let a = &FLASH.shape.attention;
    let n = a.heads / 4 * a.head_dim;
    let k = a.q_rank;
    let q = Mxfp8FoldGemmQuery {
        m_capacity: super::mxfp8_fold::tuning_count(tuning, "tile_rows"),
        n,
        k,
        groups: 1,
        activation_row_stride: k,
        activation_group_stride: k,
        output: DType::F32,
        execution: BlockScaledExecution::NativeBlockScaledMma,
    };
    Candidate::new(mxfp8_fold_local_wide::Entry {
        base: mxfp8_fold::Entry::new(
            q,
            format!("chop_{}_verify_mxfp8_n{n}_k{k}_g1_f32", FLASH.kernel_tag),
        )?,
        tile_n: super::mxfp8_fold::tuning_count(tuning, "tile_n"),
        stages: super::mxfp8_fold::tuning_count(tuning, "stages"),
    })
}
pub fn cubins(geometries: &[Geometry]) -> Result<Vec<(&'static str, CubinSpec)>, String> {
    if !geometries.iter().any(|g| g.id == FLASH.id) {
        return Ok(vec![]);
    }
    Ok(vec![(
        FAMILY,
        mxfp8_fold_local_tiled::cubin_spec(CUBIN, &[entry().map_err(|e| e.to_string())?])
            .map_err(|e| e.to_string())?,
    )])
}
