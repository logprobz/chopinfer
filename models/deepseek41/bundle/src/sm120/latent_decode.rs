//! Locked staging variants and measured native TP4 decode/verify row selections.
use kernel_build::CubinSpec;
use kernel_core::backend::{Need, Prepared, Refused, View};
use kernel_core::contract::AttentionPhase;
use kernel_core::latent_attention::{LatentAttentionQuery, LatentSourcesBuffers};
use kernels_sm120::latent::sources_decode::{self, Entry};
use model_deepseek41_contract::geometry::Geometry;
use model_deepseek41_contract::state::KvStorage;

pub const FAMILY: &str = "latent_decode_staging";
pub const CUBIN: &str = "latent_decode_staging_deepseek41_sm120";
pub const VARIANTS: &[(u32, u32, bool)] = &[
    (64, 0, false),
    (64, 0, true),
    (64, 2, true),
    (64, 4, true),
    (64, 8, true),
    (32, 0, true),
    (32, 2, true),
    (32, 4, true),
    (32, 8, true),
];
pub fn entries(g: &Geometry) -> Result<Vec<Entry>, String> {
    if !super::latent_scheduled::qualified(g, 4) {
        return Err("decode staging requires the qualified Flash TP4 geometry".into());
    }
    Ok(super::latent_scheduled::entries(g, 4)?
        .into_iter()
        .filter(|s| !s.name.ends_with("_draft_block"))
        .flat_map(|source| {
            VARIANTS
                .iter()
                .flat_map(move |&(stage_keys, head_group, packed)| {
                    let source = source.clone();
                    [false, true].map(move |profile| Entry {
                        source: source.clone(),
                        stage_keys,
                        head_group,
                        packed,
                        profile,
                    })
                })
        })
        .collect())
}
pub fn cubin(g: &Geometry) -> Result<CubinSpec, String> {
    let mut spec = sources_decode::cubin_spec(CUBIN, &entries(g)?).map_err(|e| e.to_string())?;
    spec.entries.extend([
        sources_decode::widen_spec(false),
        sources_decode::widen_spec(true),
    ]);
    Ok(spec)
}

pub fn loads(cubin: &str, g: &Geometry, ranks: u32, storage: KvStorage) -> bool {
    cubin != CUBIN || (super::latent_scheduled::qualified(g, ranks) && storage == KvStorage::Native)
}

fn selections(
    table: &kernel_build::tuning::Table,
    phase: AttentionPhase,
) -> Vec<(u32, u32, u32, bool)> {
    let (op, bucket) = match phase {
        AttentionPhase::Decode => ("latent.staging.decode", 32),
        AttentionPhase::Verify => ("latent.staging.verify", 192),
        _ => return vec![],
    };
    table
        .rows()
        .iter()
        .filter(|row| {
            row.key.op == op
                && row.key.weights == "fp8"
                && row.key.bucket == bucket
                && row.key.shape == [("ranks".into(), 4)].into()
        })
        .map(|row| {
            (
                row.key.rows.unwrap()[0],
                row.knobs["stage_keys"].count().unwrap(),
                row.knobs["head_group"].count().unwrap(),
                row.knobs["packed"].flag().unwrap(),
            )
        })
        .collect()
}

/// Only these exact phase/row pairs were timed. In particular, 16 rows and
/// eight verify rows have no timing evidence, and one-row decode was tied.
pub fn prepare(
    reference: Prepared<LatentSourcesBuffers<View>>,
    g: &Geometry,
    ranks: u32,
    query: &LatentAttentionQuery,
    need: Need,
) -> Result<Prepared<LatentSourcesBuffers<View>>, Refused> {
    use model_deepseek41_contract::tuning::{Family, table};
    prepare_with_table(reference, g, ranks, query, need, table(g, Family::Latent))
}

fn prepare_with_table(
    reference: Prepared<LatentSourcesBuffers<View>>,
    g: &Geometry,
    ranks: u32,
    query: &LatentAttentionQuery,
    need: Need,
    table: Option<&kernel_build::tuning::Table>,
) -> Result<Prepared<LatentSourcesBuffers<View>>, Refused> {
    #[cfg(test)]
    if REFERENCE.with(std::cell::Cell::get) {
        return Ok(reference);
    }
    if !super::latent_scheduled::qualified(g, ranks) {
        return Ok(reference);
    }
    let LatentAttentionQuery::Attention { phase, .. } = query else {
        return Ok(reference);
    };
    let Some(table) = table else {
        return Ok(reference);
    };
    let selected: Vec<_> = selections(table, *phase)
        .into_iter()
        .filter(|&(rows, ..)| rows <= need.capacity)
        .collect();
    if selected.is_empty() {
        return Ok(reference);
    }
    let Some(source) = super::latent_scheduled::serving_entries(g, ranks)
        .map_err(Refused::Rule)?
        .into_iter()
        .find(|entry| entry.query == *query && !entry.name.ends_with("_draft_block"))
    else {
        return Ok(reference);
    };
    let routes: Vec<_> = selected
        .into_iter()
        .map(|(rows, stage_keys, head_group, packed)| {
            (
                rows,
                Entry {
                    source: source.clone(),
                    stage_keys,
                    head_group,
                    packed,
                    profile: false,
                },
            )
        })
        .collect();
    sources_decode::prepare(reference, need, &routes)
}

#[cfg(test)]
std::thread_local! {
    static REFERENCE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Compare serving journals with the original source while retaining the
/// same loaded catalog. This switch exists only in CPU test binaries.
#[cfg(test)]
pub(crate) fn with_reference<T>(run: impl FnOnce() -> T) -> T {
    struct Restore(bool);
    impl Drop for Restore {
        fn drop(&mut self) {
            REFERENCE.with(|value| value.set(self.0));
        }
    }
    let _restore = Restore(REFERENCE.with(|value| value.replace(true)));
    run()
}

#[cfg(test)]
mod tests {
    use super::*;
    use kernel_core::backend::Backend;
    use kernels_sm120::backend::Sm120;
    use model_deepseek41_contract::{
        admission::{SUPPORTED_RANKS, check_ranks},
        geometry::GEOMETRIES,
    };

    #[test]
    fn prepares_every_admitted_staging_singleton() {
        use model_deepseek41_contract::{
            geometry::FLASH,
            tuning::{Family, table},
        };
        let mut table = table(&FLASH, Family::Latent).unwrap().clone();
        let mut added = table
            .rows()
            .iter()
            .find(|r| r.key.op == "latent.staging.decode")
            .unwrap()
            .clone();
        added.key.rows = Some([16, 16]);
        added
            .knobs
            .insert("stage_keys".into(), kernel_build::tuning::Knob::Count(32));
        added
            .knobs
            .insert("packed".into(), kernel_build::tuning::Knob::Flag(true));
        table.upsert(added).unwrap();
        let source = super::super::latent_scheduled::serving_entries(&FLASH, 4)
            .unwrap()
            .into_iter()
            .find(|e| {
                matches!(
                    e.query,
                    LatentAttentionQuery::Attention {
                        phase: AttentionPhase::Decode,
                        ..
                    }
                ) && !e.name.ends_with("_draft_block")
            })
            .unwrap();
        let backend =
            super::super::latent_scheduled::register_sources(Sm120::new(), &FLASH, 4).unwrap();
        for capacity in [8, 16, 32] {
            let need = Need::rows(capacity);
            let reference = backend
                .latent_attention_sources(&source.query, need)
                .unwrap();
            let routed =
                prepare_with_table(reference, &FLASH, 4, &source.query, need, Some(&table))
                    .unwrap();
            let symbols: Vec<_> = routed
                .info()
                .symbols
                .iter()
                .filter(|(s, _)| s.contains("_stage"))
                .collect();
            assert_eq!(symbols.len(), if capacity < 16 { 1 } else { 2 });
            assert!(
                symbols
                    .iter()
                    .any(|(s, _)| s.ends_with("_stage64_h0_b0_profile0_v1"))
            );
            assert_eq!(
                symbols
                    .iter()
                    .any(|(s, _)| s.ends_with("_stage32_h0_b1_profile0_v1")),
                capacity >= 16
            );
        }
    }

    #[test]
    fn selects_only_measured_native_target_queries() {
        for g in GEOMETRIES {
            for &ranks in SUPPORTED_RANKS {
                if check_ranks(g, ranks).is_err() {
                    continue;
                }
                let backend =
                    super::super::latent_scheduled::register_sources(Sm120::new(), g, ranks)
                        .unwrap();
                let all = super::super::latent_scheduled::serving_entries(g, ranks)
                    .unwrap()
                    .into_iter()
                    .chain(super::super::latent::bf16_entries(g, ranks).unwrap());
                for source in all {
                    let LatentAttentionQuery::Attention { phase, kv, .. } = source.query else {
                        unreachable!()
                    };
                    let eligible = super::super::latent_scheduled::qualified(g, ranks)
                        && kv != kernel_core::latent_attention::LatentKvFormat::Bf16
                        && !source.name.ends_with("_draft_block");
                    let measured = match phase {
                        AttentionPhase::Decode => 8,
                        AttentionPhase::Verify => 32,
                        _ => u32::MAX,
                    };
                    for capacity in [1, 7, 8, 16, 31, 32] {
                        let need = Need::rows(capacity);
                        let reference = backend
                            .latent_attention_sources(&source.query, need)
                            .unwrap();
                        let before = reference.info().clone();
                        let routed = prepare(reference, g, ranks, &source.query, need).unwrap();
                        let selected = eligible && measured <= capacity;
                        let candidates: Vec<_> = routed
                            .info()
                            .symbols
                            .iter()
                            .filter(|(s, _)| s.contains("_stage"))
                            .collect();
                        assert_eq!(
                            candidates.len(),
                            usize::from(selected),
                            "{} {capacity}",
                            source.symbol()
                        );
                        if selected {
                            assert!(candidates[0].0.ends_with("_stage64_h0_b0_profile0_v1"));
                            assert_eq!(candidates[0].1, 75008);
                        } else {
                            assert_eq!(*routed.info(), before);
                        }
                        assert_eq!(routed.info().bytes, before.bytes);
                        assert_eq!(routed.info().symbols.last(), before.symbols.last());
                    }
                }
                for storage in [KvStorage::Native, KvStorage::Bf16] {
                    assert_eq!(
                        loads(CUBIN, g, ranks, storage),
                        super::super::latent_scheduled::qualified(g, ranks)
                            && storage == KvStorage::Native
                    );
                    assert!(loads("unrelated", g, ranks, storage));
                }
            }
        }
    }
}
