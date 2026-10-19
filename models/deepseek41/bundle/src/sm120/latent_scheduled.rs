//! Additive attention candidates for paired lab checks before serving adoption.

use kernel_build::CubinSpec;
use kernel_core::contract::AttentionPhase;
use kernel_core::latent_attention::LatentAttentionQuery;
use kernels_sm120::backend::Sm120;
use kernels_sm120::latent::sources::{self, Entry, Schedule};
use model_deepseek41_contract::admission::{SUPPORTED_RANKS, check_ranks};
use model_deepseek41_contract::geometry::Geometry;

pub const FAMILY: &str = "latent_sources_scheduled";
pub const QUERY_FAMILY: &str = "latent_query_parallel";

/// The measured query winner covers decode and small verification batches.
pub const QUERY_ROWS: u32 = 48;

pub fn qualified(g: &Geometry, ranks: u32) -> bool {
    let a = &g.shape.attention;
    g.base
        && g.kernel_tag == "deepseek41"
        && ranks == 4
        && a.heads == 64
        && a.head_dim == 512
        && a.rope_dim == 64
        && a.window == 128
        && g.shape.indexer.top_k == 512
}

pub fn serving_entries(g: &Geometry, ranks: u32) -> Result<Vec<Entry>, String> {
    Ok(super::latent::entries(g, ranks)?
        .into_iter()
        .map(|mut entry| {
            if qualified(g, ranks)
                && !entry.name.ends_with("_draft_block")
                && matches!(
                    entry.query,
                    LatentAttentionQuery::Attention {
                        phase: AttentionPhase::Decode | AttentionPhase::Verify,
                        ..
                    }
                )
            {
                entry.schedule = Schedule::IndependentHeads;
            }
            entry
        })
        .collect())
}

pub fn register_sources(backend: Sm120, g: &Geometry, ranks: u32) -> Result<Sm120, String> {
    Ok(serving_entries(g, ranks)?
        .into_iter()
        .chain(super::latent::bf16_entries(g, ranks)?)
        .fold(backend, Sm120::with_latent_sources))
}

pub fn register_query(backend: Sm120, g: &Geometry, ranks: u32) -> Sm120 {
    if qualified(g, ranks) {
        let mut entry = query().binding();
        entry.capacity = QUERY_ROWS;
        backend.with_latent_glue(entry)
    } else {
        backend
    }
}

pub fn candidate_cubin(name: &str) -> bool {
    name.starts_with("latent_sources_scheduled_")
        || name == "latent_query_parallel_deepseek41_sm120"
}

pub fn query() -> kernels_sm120::latent::query_parallel::Entry {
    kernels_sm120::latent::query_parallel::Entry {
        capacity: model_deepseek41_contract::serving::PREFILL_TOKENS,
        symbol: "chop_deepseek41_latent_query_rope_heads_v1".into(),
    }
}

pub fn query_cubin() -> CubinSpec {
    kernels_sm120::latent::query_parallel::cubin_spec(
        "latent_query_parallel_deepseek41_sm120",
        &[query()],
    )
}

pub fn entries(g: &Geometry, ranks: u32) -> Result<Vec<Entry>, String> {
    Ok(super::latent::entries(g, ranks)?
        .into_iter()
        .filter(|e| {
            matches!(
                e.query,
                LatentAttentionQuery::Attention {
                    phase: AttentionPhase::Decode | AttentionPhase::Verify,
                    ..
                }
            )
        })
        .map(|mut e| {
            e.schedule = Schedule::IndependentHeads;
            e
        })
        .collect())
}

pub fn register(backend: Sm120, g: &Geometry, ranks: u32) -> Result<Sm120, String> {
    Ok(entries(g, ranks)?.into_iter().fold(
        backend.with_latent_glue(query().binding()),
        Sm120::with_latent_sources,
    ))
}

pub fn cubins(geometries: &[Geometry]) -> Result<Vec<CubinSpec>, String> {
    geometries
        .iter()
        .map(|g| {
            let mut all = Vec::new();
            for &ranks in SUPPORTED_RANKS {
                if check_ranks(g, ranks).is_ok() {
                    all.extend(entries(g, ranks)?);
                }
            }
            let mut spec =
                sources::cubin_spec(super::latent::profile(g), &all).map_err(|e| e.to_string())?;
            spec.name = format!("latent_sources_scheduled_{}_sm120", g.kernel_tag);
            // A wrong sink multiplicity must change the compared output.
            for e in &all {
                spec.entries
                    .push(e.sink_per_split_entry_spec().map_err(|e| e.to_string())?);
            }
            Ok(spec)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use kernel_core::backend::{Backend, Need};
    use model_deepseek41_contract::{geometry::GEOMETRIES, serving::PREFILL_TOKENS};

    #[test]
    fn serving_routes_only_qualified_source_shapes_and_query_rows() {
        for g in GEOMETRIES {
            for &ranks in SUPPORTED_RANKS {
                if check_ranks(g, ranks).is_err() {
                    continue;
                }
                let serving = serving_entries(g, ranks).unwrap();
                for entry in serving {
                    let eligible = qualified(g, ranks)
                        && !entry.name.ends_with("_draft_block")
                        && matches!(
                            entry.query,
                            LatentAttentionQuery::Attention {
                                phase: AttentionPhase::Decode | AttentionPhase::Verify,
                                ..
                            }
                        );
                    assert_eq!(entry.schedule == Schedule::IndependentHeads, eligible);
                    let backend = register_sources(Sm120::new(), g, ranks).unwrap();
                    let op = backend
                        .latent_attention_sources(&entry.query, Need::rows(1))
                        .unwrap();
                    assert_eq!(op.info().symbols[0].0, entry.symbol());
                    assert_eq!(
                        op.info().symbols[0].1,
                        entry.plan(1).unwrap().dynamic_shared_bytes
                    );
                }
                let reference = kernels_sm120::latent::glue::Entry {
                    kernel: kernels_sm120::latent::glue::Kernel::QueryRope,
                    capacity: PREFILL_TOKENS,
                    symbol: "chop_deepseek41_latent_query_rope_v1".into(),
                };
                let backend =
                    register_query(Sm120::new().with_latent_glue(reference.clone()), g, ranks);
                let query = model_deepseek41_contract::latent_glue::query_rope(g, ranks).unwrap();
                for rows in [1, 6, 8, 32, 48, 49, 192, PREFILL_TOKENS] {
                    let op = backend.latent_query_rope(&query, Need::rows(rows)).unwrap();
                    let parallel = qualified(g, ranks) && rows <= QUERY_ROWS;
                    assert_eq!(
                        op.info().symbols[0].0,
                        if parallel {
                            super::query().symbol
                        } else {
                            reference.symbol.clone()
                        }
                    );
                }
            }
        }
    }
}
