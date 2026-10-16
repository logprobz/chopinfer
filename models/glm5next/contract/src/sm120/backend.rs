//! [`crate::backend::Kernels`] on SM120.

use kernel_core::contract::{Target, WeightFormat};
use kernel_core::precision::OperationUse;

use crate::backend::{Bound, GatherBytes, Kernels, OwnLaunch, StateBytes};
use crate::contract::Shape;
use crate::sm120::engines;
use crate::sm120::prepared::{self, Offer};

/// The SM120 kernel backend (`kernels_sm120`).
#[derive(Clone, Copy, Debug)]
pub struct Sm120;

impl Kernels for Sm120 {
    const TARGET: Target = Target::Sm120f;
    const ARCHITECTURE: &'static str = "sm_120f";

    type Entry = Offer;

    fn bind(id: &str, weight: WeightFormat, usage: OperationUse) -> Result<Bound<Offer>, String> {
        prepared::bind(id, weight, usage)
    }

    /// A use bound to its part of a stacked entry runs in the stacked
    /// projections' launch, with the entry's workspace and tickets at its
    /// bucket.
    fn own_launch(entry: &Offer) -> Result<Option<OwnLaunch>, String> {
        let Some(part) = &entry.stacked else {
            return Ok(None);
        };
        let plan = part
            .entry
            .plan(part.entry.capacity)
            .map_err(|r| format!("{}: {r}", part.entry.symbol))?;
        Ok(Some(OwnLaunch {
            workspace_bytes: plan.workspace_bytes,
            semaphore_bytes: part.entry.semaphore_bytes(),
        }))
    }

    const GATHER_ROWS_STATE: StateBytes = StateBytes {
        flags: kernels_sm120::comm::GATHER_ROWS_FLAG_BYTES,
        epoch: kernels_sm120::comm::GATHER_ROWS_EPOCH_BYTES,
    };

    fn sorted_rows(shape: Shape) -> Result<u32, String> {
        let served = engines::moe_shapes(shape)?;
        Ok(served
            .routes()
            .iter()
            .map(|route| route.sorted_rows_capacity())
            .max()
            .unwrap_or(0))
    }

    fn mix_scratch_bytes(shape: Shape, rows: u32) -> [u64; 2] {
        use kernels_sm120::sinkhorn_hc::split;
        [
            split::partials_bytes(engines::sinkhorn_geometry(shape).mix, rows),
            split::tickets_bytes(rows),
        ]
    }

    /// Selection partials depend on the bucket, so the indexer's take every
    /// power of two up to each entry's capacity.
    fn workspace_bytes(shape: Shape, step_rows: u32, prefill_rows: u32) -> u64 {
        let latent = engines::latent_entries(shape)
            .iter()
            .map(|entry| entry.workspace_bytes(step_rows.min(entry.capacity())))
            .max()
            .unwrap_or(0);
        let indexer = engines::pooled_indexer_entries(shape)
            .iter()
            .flat_map(|entry| {
                (0..=entry.capacity.ilog2()).map(move |power| entry.workspace_bytes(1 << power))
            })
            .max()
            .unwrap_or(0);
        let kda = engines::channel_decay_entries(shape)
            .iter()
            .map(|entry| entry.workspace_bytes(engines::kda_prefill_bucket(prefill_rows)))
            .max()
            .unwrap_or(0);
        latent.max(kda).max(indexer)
    }

    fn kda_checkpoint_rows(shape: Shape) -> Vec<u32> {
        use kernel_core::contract::RecurrentPhase;
        use kernel_core::per_channel_decay_delta_rule::PerChannelDecayDeltaRuleQuery;
        use kernels_sm120::linear::channel_decay::Query;
        engines::channel_decay_entries(shape)
            .iter()
            .filter_map(|entry| match entry.query {
                Query::Rule(
                    PerChannelDecayDeltaRuleQuery::ShortConv { phase, .. }
                    | PerChannelDecayDeltaRuleQuery::DeltaRule { phase, .. },
                ) => match phase {
                    RecurrentPhase::Verify { max_rows } => Some(max_rows),
                    _ => None,
                },
                Query::Norm(_) => None,
            })
            .collect()
    }

    fn moe_all_reduce_state_bytes(ranks: u32, hidden: u32) -> Result<u64, String> {
        let row = u64::from(hidden);
        let refused = |refusal: kernel_core::contract::Refusal| refusal.to_string();
        let ll = kernels_sm120::comm::ll_f32::state_bytes(
            ranks,
            u64::from(engines::ROUNDED_LOW_LATENCY_ROWS) * row,
        )
        .map_err(refused)?;
        let phased =
            kernels_sm120::comm::phased_f32::buffers(ranks, u64::from(engines::PHASED_ROWS) * row)
                .map_err(refused)?;
        Ok(ll.inbox + ll.epochs + phased.inbox_bytes + phased.flag_bytes + phased.epoch_bytes)
    }

    fn group_gather(size: u32, words: u64) -> Result<GatherBytes, String> {
        let buffers = kernels_sm120::context_parallel::group::gather_buffers(size, words)
            .map_err(|refusal| refusal.to_string())?;
        Ok(GatherBytes {
            output_bytes: buffers.output_bytes,
            flag_bytes: buffers.flag_bytes,
            epoch_bytes: buffers.epoch_bytes,
        })
    }
}
