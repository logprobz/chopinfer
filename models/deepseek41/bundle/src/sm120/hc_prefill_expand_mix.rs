//! Measured prefill rows of the expansion and the next lagged mixer in one
//! launch. Other row counts keep the expansion and the measured mixer.
use kernel_build::CubinSpec;
use kernel_core::sinkhorn_expand_mix::SinkhornExpandMixQuery;
use kernels_sm120::{
    backend::Sm120,
    sinkhorn_hc::prefill_expand_mix::{self, Entry, Mutation},
};
use model_deepseek41_contract::{
    geometry::{FLASH, Geometry},
    hc,
    serving::PREFILL_TOKENS,
};

pub const FAMILY: &str = "sinkhorn_hc_prefill_expand_mix";

/// (tile rows, chains) and the prefill rows each tile serves: the rows where
/// the paired lab chain (expansion, mixer, input norm) gained on two devices
/// at 250 W, cold and hot. 2496 rows tied and keeps the separate launches.
pub const BUCKETS: [((u32, u32), &[u32]); 2] = [((4, 4), &[64, 320]), ((8, 8), &[8192])];

pub fn qualified(g: &Geometry) -> bool {
    g == &FLASH
}

/// Whether a prefill of `rows` launches the expansion and mixer together.
pub fn selected(g: &Geometry, rows: u32) -> bool {
    qualified(g) && BUCKETS.iter().any(|(_, buckets)| buckets.contains(&rows))
}

pub fn cubin_name(g: &Geometry) -> String {
    format!("sinkhorn_hc_{}_prefill_expand_mix_sm120", g.kernel_tag)
}

pub fn entry(g: &Geometry, tile_rows: u32, chains: u32) -> Entry {
    Entry {
        query: SinkhornExpandMixQuery { mix: hc::mix(g) },
        capacity: PREFILL_TOKENS,
        tile_rows,
        chains,
        symbol: format!(
            "chop_{}_hc_prefill_expand_mix_r{tile_rows}_c{chains}_v1",
            g.kernel_tag
        ),
        mutation: Mutation::None,
    }
}

pub fn cubins(geometries: &[Geometry]) -> Result<Vec<(&'static str, CubinSpec)>, String> {
    geometries
        .iter()
        .filter(|g| qualified(g))
        .map(|g| {
            let entries: Vec<_> = BUCKETS
                .iter()
                .map(|&((rows, chains), _)| entry(g, rows, chains))
                .collect();
            Ok((
                FAMILY,
                prefill_expand_mix::cubin_spec(&cubin_name(g), &entries)
                    .map_err(|e| e.to_string())?,
            ))
        })
        .collect()
}

pub fn register(backend: Sm120, g: &Geometry) -> Sm120 {
    if !qualified(g) {
        return backend;
    }
    BUCKETS
        .iter()
        .fold(backend, |backend, &((rows, chains), buckets)| {
            backend.with_sinkhorn_prefill_expand_mix(entry(g, rows, chains), buckets.to_vec())
        })
}
