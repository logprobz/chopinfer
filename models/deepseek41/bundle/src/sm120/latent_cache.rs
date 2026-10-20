//! Qualified selected-key decoding for native Flash TP4 attention.
use kernel_build::CubinSpec;
use kernel_core::backend::{Backend, Need, Prepared, Refused};
use kernel_core::contract::AttentionPhase;
use kernel_core::latent_attention::LatentAttentionQuery;
use kernels_sm120::latent::sources_cached::{self, Buffers, Entry};
use model_deepseek41_contract::geometry::Geometry;
use model_deepseek41_contract::state::KvStorage;

pub const FAMILY: &str = "latent_decode_cache";
pub const CUBIN: &str = "latent_decode_cache_deepseek41_sm120";
pub const MAX_ROWS: u32 = 32;
pub const BYTES: u64 = MAX_ROWS as u64 * 640 * (512 * 2 + 4);

pub fn eligible(g: &Geometry, ranks: u32, storage: KvStorage) -> bool {
    super::latent_scheduled::qualified(g, ranks) && storage == KvStorage::Native
}

pub fn rows(phase: AttentionPhase) -> &'static [u32] {
    match phase {
        AttentionPhase::Decode => &[1, 8],
        AttentionPhase::Verify => &[4, 6, 32],
        _ => &[],
    }
}

pub fn selected(phase: AttentionPhase, count: u32) -> bool {
    #[cfg(test)]
    if REFERENCE.with(std::cell::Cell::get) {
        return false;
    }
    rows(phase).contains(&count)
}

/// Keep the qualified lab module byte-for-byte, including its unused controls.
pub fn entries(g: &Geometry) -> Result<Vec<Entry>, String> {
    Ok(super::latent_scheduled::entries(g, 4)?
        .into_iter()
        .filter(|s| !s.name.ends_with("_draft_block"))
        .flat_map(|source| {
            [false, true].map(move |profile| Entry {
                source: source.clone(),
                profile,
            })
        })
        .collect())
}

pub fn cubin(g: &Geometry) -> Result<CubinSpec, String> {
    sources_cached::cubin_spec(CUBIN, &entries(g)?).map_err(|e| e.to_string())
}

pub fn prepare(
    backend: &dyn Backend,
    g: &Geometry,
    query: &LatentAttentionQuery,
    need: Need,
) -> Result<Prepared<Buffers>, Refused> {
    let entry = entries(g)
        .map_err(Refused::Rule)?
        .into_iter()
        .find(|entry| !entry.profile && entry.source.query == *query)
        .ok_or_else(|| Refused::Rule("unqualified selected cache query".into()))?;
    let LatentAttentionQuery::Attention { phase, .. } = query else {
        unreachable!()
    };
    let rows: Vec<_> = rows(*phase)
        .iter()
        .copied()
        .filter(|&r| r <= need.capacity)
        .collect();
    let reference = backend.latent_attention_sources(query, need)?;
    sources_cached::prepare(reference, entry, need, &rows)
}

#[cfg(test)]
std::thread_local! {
    static REFERENCE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

#[cfg(test)]
pub(crate) fn with_reference<T>(run: impl FnOnce() -> T) -> T {
    struct Restore(bool);
    impl Drop for Restore {
        fn drop(&mut self) {
            REFERENCE.with(|flag| flag.set(self.0));
        }
    }
    let _restore = Restore(REFERENCE.with(|flag| flag.replace(true)));
    run()
}
