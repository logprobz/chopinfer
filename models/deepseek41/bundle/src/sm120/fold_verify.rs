//! Sixty-four-row cluster status tiles for Flash TP4 verify comparisons.
use kernel_build::CubinSpec;
use kernel_core::{
    contract::DType, mxfp8_gemm::Mxfp8FoldGemmQuery, precision::execution::BlockScaledExecution,
};
use kernels_sm120::gemm::{
    mxfp8_fold, mxfp8_fold_cluster,
    mxfp8_fold_cluster_fused::{self, Entry},
};
use model_deepseek41_contract::geometry::{FLASH, Geometry};

pub const FAMILY: &str = "mxfp8_fold_verify64";
pub const CUBIN: &str = "mxfp8_fold_verify64_deepseek41_sm120";
pub const IDS: [&str; 4] = [
    "attention.q-a",
    "attention.kv",
    "attention.o-a",
    "attention.o-b",
];

pub fn enabled(g: &Geometry, ranks: u32, verify_rows: u32) -> bool {
    g.id == FLASH.id && ranks == 4 && verify_rows > 32
}

/// The query projection and sliced output chain won on both devices at
/// every wide row count tested. The window-KV prefetch path remains faster.
pub fn served(id: &str, rows: u32) -> Option<Entry> {
    let op = format!("dense_fold.{}_status.verify", id.replace(['.', '-'], "_"));
    super::mxfp8_fold::status_tuning(&op, 64, rows)?;
    entry(id)
}

pub fn entry(id: &str) -> Option<Entry> {
    let (n, k, groups, pending, quantize) = match id {
        "attention.q-a" => (1280, 5120, 1, 0, false),
        "attention.kv" => (512, 5120, 1, 0, false),
        "attention.o-a" => (1024, 4096, 2, 0, true),
        "attention.o-b" => (5120, 2048, 1, 64, false),
        _ => return None,
    };
    let (tile_n, splits, stages, distributed) = if id == "attention.kv" {
        // This lab descriptor is not a served wide status route.
        (16, 8, 5, false)
    } else {
        let op = format!("dense_fold.{}_status.verify", id.replace(['.', '-'], "_"));
        let tuning = super::mxfp8_fold::status_tuning(&op, 64, 64)?;
        (
            super::mxfp8_fold::tuning_count(tuning, "tile_n"),
            super::mxfp8_fold::tuning_count(tuning, "splits"),
            super::mxfp8_fold::tuning_count(tuning, "stages"),
            super::fold_cluster::distributed(tuning),
        )
    };
    let q = Mxfp8FoldGemmQuery {
        n,
        k,
        groups,
        m_capacity: 64,
        activation_row_stride: k * groups,
        activation_group_stride: k,
        output: DType::F32,
        execution: BlockScaledExecution::NativeBlockScaledMma,
    };
    Some(Entry {
        cluster: mxfp8_fold_cluster::Entry {
            base: mxfp8_fold::Entry::new(
                q,
                format!(
                    "chop_{}_verify_mxfp8_n{n}_k{k}_g{groups}_f32",
                    FLASH.kernel_tag
                ),
            )
            .unwrap(),
            tile_n,
            tile_rows: 64,
            splits,
            stages,
            distributed,
        },
        pending,
        quantize,
    })
}

pub fn cubins(geometries: &[Geometry]) -> Result<Vec<(&'static str, CubinSpec)>, String> {
    if !geometries.iter().any(|g| g.id == FLASH.id) {
        return Ok(vec![]);
    }
    let entries = IDS.map(|id| entry(id).unwrap());
    let mut spec = mxfp8_fold_cluster_fused::cubin_spec(CUBIN, &entries, &[], true)
        .map_err(|e| e.to_string())?;
    spec.entries.extend(
        entries
            .iter()
            .map(Entry::packed_spec)
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| e.to_string())?,
    );
    Ok(vec![(FAMILY, spec)])
}
