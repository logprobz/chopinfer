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
    ((33..=64).contains(&rows) && id != "attention.kv")
        .then(|| entry(id))
        .flatten()
}

pub fn entry(id: &str) -> Option<Entry> {
    let (n, k, groups, tile_n, splits, stages, pending, quantize) = match id {
        "attention.q-a" => (1280, 5120, 1, 64, 8, 5, 0, false),
        "attention.kv" => (512, 5120, 1, 16, 8, 5, 0, false),
        "attention.o-a" => (1024, 4096, 2, 64, 8, 3, 0, true),
        "attention.o-b" => (5120, 2048, 1, 32, 2, 4, 64, false),
        _ => return None,
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
            distributed: false,
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
