//! Qualified HC finalization and cached input norm in an unchanged cubin.
use kernel_build::CubinSpec;
use kernels_sm120::sinkhorn_hc::finish_norm::Entry;
use model_deepseek41_contract::geometry::Geometry;
pub const FAMILY: &str = "sinkhorn_hc_finish_norm";
pub fn cubin_name(g: &Geometry) -> String {
    format!("sinkhorn_hc_{}_finish_norm_sm120", g.kernel_tag)
}
pub fn entry(g: &Geometry) -> Entry {
    Entry {
        mixer: super::hc_round_expand_mix::entry(g),
        norm: super::norm_quantize_cached::entry(g),
        prefix: format!("chop_{}_hc_finish_norm", g.kernel_tag),
    }
}
pub fn cubins(geometries: &[Geometry]) -> Result<Vec<(&'static str, CubinSpec)>, String> {
    geometries
        .iter()
        .map(|g| {
            Ok((
                FAMILY,
                entry(g)
                    .cubin_spec(&cubin_name(g))
                    .map_err(|e| e.to_string())?,
            ))
        })
        .collect()
}

/// The qualified pair, joined into one launch at the measured Flash
/// buckets (`COMPLETION_BUCKETS`), with its tail on two CTAs per row at
/// `TWIN_BUCKETS`.
pub fn register(
    backend: kernels_sm120::backend::Sm120,
    g: &Geometry,
) -> kernels_sm120::backend::Sm120 {
    let backend = register_pair(backend, g);
    if g.id == model_deepseek41_contract::geometry::FLASH.id {
        backend
            .with_sinkhorn_finish_norm_completion(
                super::hc_completion::served(),
                COMPLETION_BUCKETS.to_vec(),
            )
            .with_sinkhorn_finish_norm_twin(super::hc_finish_twin::served(), TWIN_BUCKETS.to_vec())
    } else {
        backend
    }
}

/// Measured (row bucket, warp count) pairs of the one-launch boundary: the
/// split layout with sixteen warps per chain CTA at C1 and eight at C8.
pub const COMPLETION_BUCKETS: [(u32, u32); 2] = [(1, 16), (8, 8)];

/// Measured (row bucket, warp count) pairs of the two-CTA tail: DSpark's
/// Verify-6 bucket (five drafts and the root).
pub const TWIN_BUCKETS: [(u32, u32); 1] = [(6, 32)];

/// Only measured Flash decode and verify buckets use the 1024-thread tail.
/// The lab compares one-launch boundaries against exactly this pair.
pub fn register_pair(
    backend: kernels_sm120::backend::Sm120,
    g: &Geometry,
) -> kernels_sm120::backend::Sm120 {
    if g.id == model_deepseek41_contract::geometry::FLASH.id {
        backend
            .with_sinkhorn_finish_norm(entry(g), vec![1, 4, 6, 8, 12, 16, 32])
            .with_sinkhorn_finish_norm_chain(
                super::hc_round_chain::entry(g),
                vec![(1, 8), (4, 8), (6, 4), (8, 4), (12, 4), (16, 4), (32, 4)],
            )
            .with_sinkhorn_finish_norm_rows(super::hc_round_rows::entry(g), vec![(32, 2)])
    } else {
        backend
    }
}
