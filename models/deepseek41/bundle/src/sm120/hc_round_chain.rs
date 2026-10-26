//! Exact rounded HC producers with the served partial workspace layout.
use kernel_build::CubinSpec;
use kernels_sm120::backend::Sm120;
use kernels_sm120::sinkhorn_hc::round_chain_partials::Entry;
use model_deepseek41_contract::geometry::Geometry;
pub const FAMILY: &str = "sinkhorn_hc_round_chain";
pub fn cubin_name(g: &Geometry) -> String {
    format!("sinkhorn_hc_{}_round_chain_sm120", g.kernel_tag)
}
pub fn entry(g: &Geometry) -> Entry {
    Entry {
        reference: super::hc_round_expand_mix::entry(g),
        prefix: format!("chop_{}_hc_round_chain", g.kernel_tag),
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

/// Flash TP4's small rows match the fallback bitwise, including poisoned
/// eager/captured replays and all four logical ranks. At 300 W with unlocked
/// clocks, paired lab gains at rows 2-7 were 21-30% on both devices; the
/// separate A/A controls differed by less than 0.4% (2026-10-25).
pub fn register(backend: Sm120, g: &Geometry, ranks: u32) -> Sm120 {
    if g.id == model_deepseek41_contract::geometry::FLASH.id {
        backend.with_sinkhorn_round_chain(entry(g), buckets(ranks))
    } else {
        backend
    }
}

pub(crate) fn buckets(ranks: u32) -> Vec<(u32, u32)> {
    use model_deepseek41_contract::{
        geometry::FLASH,
        tuning::{self, Family},
    };
    (1..=64)
        .filter_map(|rows| {
            let row = tuning::select(
                &FLASH,
                Family::Hc,
                "hc.round_chain.decode",
                "f32",
                &[
                    ("groups", FLASH.shape.hc.streams),
                    ("width", FLASH.shape.hidden),
                    ("ranks", ranks),
                ],
                64,
                rows,
            )?;
            Some((rows, row.knobs["warps"].count().expect("HC chain warps")))
        })
        .collect()
}
