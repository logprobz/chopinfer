//! Paired-row attention for native Flash TP4 prefill rows.
use kernel_build::CubinSpec;
use kernel_core::backend::{
    Backend, Counts, Launch, Need, OpInfo, Prepared, PreparedOp, Refused, View,
};
use kernel_core::contract::{AttentionPhase, Refusal};
use kernel_core::latent_attention::{LatentAttentionQuery, LatentSourcesBuffers};
use kernels_sm120::latent::{sources, sources_prefill};
use model_deepseek41_contract::geometry::{FLASH, Geometry};
use model_deepseek41_contract::state::KvStorage;

pub const FAMILY: &str = "latent_prefill_reuse";
pub const CUBIN: &str = "latent_sources_prefill_reuse2_deepseek41_sm120";
pub const ROWS: u32 = 8192;

pub fn entries() -> Result<Vec<sources::Entry>, String> {
    let entries: Vec<_> = super::latent::entries(&FLASH, 4)?
        .into_iter()
        .filter(|entry| {
            matches!(
                entry.query,
                LatentAttentionQuery::Attention {
                    phase: AttentionPhase::Prefill,
                    ..
                }
            )
        })
        .collect();
    if entries.len() != 3 {
        return Err("expected window and two mixed prefill entries".into());
    }
    Ok(entries)
}

pub fn cubin() -> Result<CubinSpec, String> {
    sources_prefill::reuse_cubin_spec(super::latent::profile(&FLASH), &entries()?, 2)
        .map_err(|e| e.to_string())
}

/// Other layouts and topologies do not load an unused attention module.
pub fn loads(cubin: &str, geometry: &Geometry, ranks: u32, storage: KvStorage) -> bool {
    cubin != CUBIN || (geometry.id == FLASH.id && ranks == 4 && storage == KvStorage::Native)
}

pub fn prepare(
    backend: &dyn Backend,
    geometry: &Geometry,
    ranks: u32,
    query: &LatentAttentionQuery,
    need: Need,
) -> Result<Prepared<LatentSourcesBuffers<View>>, Refused> {
    let reference = backend.latent_attention_sources(query, need)?;
    let source = reference.info().symbols[0].0.clone();
    let reference = kernel_core::route::tagged(
        reference,
        vec![kernel_core::route::branch(
            &source,
            kernel_core::route::RouteClass::Reference,
            "latent_prefill",
            "reference",
            [1..=need.capacity],
        )],
    );
    let rows = super::latent_scheduled::paired_prefill_rows(geometry, ranks, need.capacity);
    if rows.is_empty() {
        return Ok(reference);
    }
    let Some(entry) = entries()
        .map_err(Refused::Rule)?
        .into_iter()
        .find(|e| e.query == *query)
    else {
        return Ok(reference);
    };
    let mut info = reference.info().clone();
    // Retain the reference symbols: the reference validates every launch and
    // keeps its merge. The numerical contract is unchanged; only input
    // staging is different.
    info.symbols.push((
        sources_prefill::reuse_symbol(&entry, 2),
        sources_prefill::reuse_plan(&entry, ROWS, 2)
            .map_err(|e| Refused::Rule(e.to_string()))?
            .dynamic_shared_bytes,
    ));
    need.check(&info)?;
    Ok(Box::new(PairedPrefill {
        reference,
        info,
        entry,
        rows,
    }))
}

struct PairedPrefill {
    reference: Prepared<LatentSourcesBuffers<View>>,
    info: OpInfo,
    entry: sources::Entry,
    rows: Vec<std::ops::RangeInclusive<u32>>,
}

impl PreparedOp<LatentSourcesBuffers<View>> for PairedPrefill {
    fn route(&self, counts: Counts, launch: &Launch) -> Option<kernel_core::route::RouteTag> {
        if launch.symbol == sources_prefill::reuse_symbol(&self.entry, 2) {
            Some(kernel_core::route::branch(
                &launch.symbol,
                kernel_core::route::RouteClass::Fast,
                "latent_prefill",
                "paired",
                self.rows.clone(),
            ))
        } else {
            self.reference.route(counts, launch)
        }
    }
    fn info(&self) -> &OpInfo {
        &self.info
    }

    fn launches(
        &self,
        buffers: &LatentSourcesBuffers<View>,
        workspace: View,
        counts: Counts,
    ) -> Result<Vec<Launch>, Refusal> {
        // Validate through the reference before changing the attention plan.
        // Device live counts still mask graph replays, including empty tails.
        // Every row count and packing of prompts pairs rows: each row reads
        // its own item's keys and keeps the reference arithmetic, and a CTA
        // stages one key tile for both rows only when their keys are the
        // same or shifted by one, otherwise one tile per row.
        let mut launches = self.reference.launches(buffers, workspace, counts)?;
        self.substitute(&mut launches, counts)?;
        Ok(launches)
    }
}

impl PairedPrefill {
    fn substitute(&self, launches: &mut [Launch], counts: Counts) -> Result<(), Refusal> {
        if !self.rows.iter().any(|range| range.contains(&counts.rows)) {
            return Ok(());
        }
        for launch in launches {
            if launch.symbol == self.entry.symbol() {
                launch.symbol = sources_prefill::reuse_symbol(&self.entry, 2);
                launch.plan = sources_prefill::reuse_plan(&self.entry, counts.rows, 2)?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use model_deepseek41_contract::tuning::{Family, table};

    #[test]
    fn paired_route_rows_keep_the_full_interval() {
        assert_eq!(
            super::super::latent_scheduled::paired_prefill_rows(&FLASH, 4, ROWS),
            [1..=8192]
        );
    }

    #[test]
    fn narrowed_paired_range_controls_substitution() {
        let mut table =
            kernel_build::tuning::Table::new(table(&FLASH, Family::Latent).unwrap().target);
        let mut row = model_deepseek41_contract::tuning::table(&FLASH, Family::Latent)
            .unwrap()
            .rows()
            .iter()
            .find(|r| r.key.op == "latent.paired.prefill")
            .unwrap()
            .clone();
        row.key.rows = Some([8, 16]);
        table.upsert(row).unwrap();
        let ranges =
            super::super::latent_scheduled::paired_prefill_rows_from(Some(&table), 4, ROWS);
        assert_eq!(ranges, [8..=16]);
        let entry = entries().unwrap().remove(0);
        let backend = kernels_sm120::backend::Sm120::new().with_latent_sources(entry.clone());
        let reference = backend
            .latent_attention_sources(&entry.query, Need::rows(ROWS))
            .unwrap();
        let paired = PairedPrefill {
            info: reference.info().clone(),
            reference,
            entry,
            rows: ranges,
        };
        for rows in [1, 7, 8, 12, 16, 17, ROWS] {
            let original = Launch {
                symbol: paired.entry.symbol(),
                plan: paired.entry.plan(rows).unwrap(),
                arguments: kernel_core::contract::ArgumentList(vec![]),
                rows,
                collective: false,
                persistent: false,
            };
            let mut launches = vec![original.clone()];
            paired
                .substitute(&mut launches, Counts::rows(rows))
                .unwrap();
            if (8..=16).contains(&rows) {
                assert_eq!(
                    launches[0].symbol,
                    sources_prefill::reuse_symbol(&paired.entry, 2)
                );
                assert_eq!(
                    launches[0].plan,
                    sources_prefill::reuse_plan(&paired.entry, rows, 2).unwrap()
                );
            } else {
                assert_eq!(launches[0].symbol, original.symbol);
                assert_eq!(launches[0].plan, original.plan);
            }
        }
    }
}
