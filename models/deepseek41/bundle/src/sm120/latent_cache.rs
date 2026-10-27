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
pub const MAX_ROWS: u32 = model_deepseek41_contract::serving::VERIFY_DECODE_ROWS;

pub fn bytes(rows: u32) -> u64 {
    u64::from(rows.min(MAX_ROWS)) * 640 * (512 * 2 + 4)
}

pub fn eligible(g: &Geometry, ranks: u32, storage: KvStorage) -> bool {
    super::latent_scheduled::qualified(g, ranks) && storage == KvStorage::Native
}

pub fn rows(phase: AttentionPhase) -> Vec<u32> {
    let (op, bucket) = match phase {
        AttentionPhase::Decode => ("latent.cache.decode", 32),
        AttentionPhase::Verify => ("latent.cache.verify", 192),
        _ => return Vec::new(),
    };
    use model_deepseek41_contract::{
        geometry::FLASH,
        tuning::{Family, table},
    };
    table(&FLASH, Family::Latent)
        .unwrap()
        .rows()
        .iter()
        .filter(|row| row.key.op == op && row.key.bucket == bucket && row.key.shape["ranks"] == 4)
        .flat_map(|row| {
            let [lo, hi] = row.key.rows.unwrap_or([1, row.key.bucket]);
            lo..=hi
        })
        .collect()
}

pub fn selected(phase: AttentionPhase, count: u32) -> bool {
    #[cfg(test)]
    if REFERENCE.with(std::cell::Cell::get) {
        return false;
    }
    cache_row(phase, count).is_some()
}

fn cache_row(phase: AttentionPhase, rows: u32) -> Option<&'static kernel_build::tuning::Row> {
    let (op, bucket) = match phase {
        AttentionPhase::Decode => ("latent.cache.decode", 32),
        AttentionPhase::Verify => ("latent.cache.verify", 192),
        _ => return None,
    };
    use model_deepseek41_contract::{
        geometry::FLASH,
        tuning::{Family, select},
    };
    select(
        &FLASH,
        Family::Latent,
        op,
        "fp8",
        &[("ranks", 4)],
        bucket,
        rows,
    )
}

pub fn head_group(phase: AttentionPhase, rows: u32) -> u32 {
    cache_row(phase, rows)
        .and_then(|row| row.knobs["head_group"].count())
        .unwrap_or(0)
}

/// Serving cache descriptors; profiling controls stay in the model lab.
pub fn entries(g: &Geometry) -> Result<Vec<Entry>, String> {
    Ok(super::latent_scheduled::entries(g, 4)?
        .into_iter()
        .filter(|s| !s.name.ends_with("_draft_block"))
        .map(|source| Entry {
            source,
            profile: false,
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
    maximum_rows: u32,
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
        .filter(|&r| r <= need.capacity && r <= maximum_rows)
        .collect();
    let reference = backend.latent_attention_sources(query, need)?;
    // Complete captured attention graphs won at four rows with eight heads
    // sharing each key tile. Retain the measured routes at all other widths.
    let groups: Vec<_> = rows
        .iter()
        .filter_map(|&rows| {
            let group = head_group(*phase, rows);
            (group != 0).then_some((rows, group))
        })
        .collect();
    sources_cached::prepare_grouped(reference, entry, need, &rows, &groups)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_bounds_and_decode_rows() {
        assert_eq!(rows(AttentionPhase::Decode), [1, 2, 4, 8]);
        assert_eq!(bytes(32), 21_053_440);
        assert_eq!(bytes(64), 42_106_880);
        assert_eq!(bytes(192), bytes(64));
        for count in 1..=192 {
            assert_eq!(
                selected(AttentionPhase::Decode, count),
                matches!(count, 1 | 2 | 4 | 8)
            );
            assert_eq!(
                selected(AttentionPhase::Verify, count),
                matches!(count, 4 | 6 | 16 | 32..=64)
            );
            assert!(!selected(AttentionPhase::Prefill, count));
        }
    }

    #[test]
    fn verify_sixteen_uses_the_selected_cache() {
        assert!(rows(AttentionPhase::Verify).contains(&16));
        assert!(selected(AttentionPhase::Verify, 16));
        for rows in [8, 12, 15, 17, 24, 31] {
            assert!(!selected(AttentionPhase::Verify, rows));
        }
    }
}
