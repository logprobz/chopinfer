//! Continuous domains of GLM's registered projection and decode entries.

use super::engines::Engines;
use kernel_core::contract::{AttentionPhase, RecurrentPhase};
use kernel_core::latent_attention::LatentAttentionQuery;
use kernel_core::route::{RouteClass, RouteTag};
use kernels_sm120::{latent, pooled_indexer, precision::Compute};

pub(super) fn entries(engines: &Engines) -> Vec<RouteTag> {
    let mut tags = Vec::new();
    for (id, usage, generated) in model_glm5next_contract::sm120::precision::offers(engines.shape)
        .expect("registered projection offers")
    {
        // lower::project chooses the smallest declared use holding the rows.
        // Wide verify uses can compile at prefill capacity, so use the declared
        // bucket here rather than the compiled GEMM capacity.
        let previous = model_glm5next_contract::precision::uses(id)
            .expect("registered projection uses")
            .into_iter()
            .map(|(other, _)| other)
            .filter(|other| {
                other.phase == usage.phase
                    && other.graph == usage.graph
                    && other.capacity < usage.capacity
            })
            .map(|other| other.capacity)
            .max()
            .unwrap_or(0);
        if let Compute::Dense(spec) = &generated.offer.handle().compute {
            tags.push(RouteTag::new(
                spec.symbol.clone(),
                RouteClass::Fast,
                "projection",
                [previous + 1..=usage.capacity],
            ));
        }
    }
    let decode_capacity = |entry: &latent::Entry| match entry.query {
        latent::Query::Latent(LatentAttentionQuery::Attention {
            phase: AttentionPhase::Decode,
            capacity,
            ..
        }) => Some(capacity),
        _ => None,
    };
    for entry in &engines.latent {
        if let Some(capacity) = decode_capacity(entry) {
            tags.push(RouteTag::new(
                entry.symbol(),
                RouteClass::Fast,
                "latent_decode",
                [1..=capacity],
            ));
        }
    }
    for entry in &engines.latent_merge {
        if let Some(capacity) = decode_capacity(&entry.attention) {
            tags.push(RouteTag::new(
                entry.symbol(),
                RouteClass::Fast,
                "latent_decode",
                [1..=capacity],
            ));
        }
    }
    for entry in &engines.indexer {
        if matches!(
            entry.kind,
            pooled_indexer::Kind::KeyWrite(RecurrentPhase::Decode)
        ) {
            tags.push(RouteTag::new(
                entry.symbol(),
                RouteClass::Fast,
                "indexer_decode",
                [1..=entry.capacity],
            ));
        }
    }
    // These symbols belong only to the admitted low-latency forms. Composed
    // collective fallbacks have different entries and receive no such tag.
    tags.push(RouteTag::new(
        kernels_sm120::comm::ll_bf16::SYMBOL,
        RouteClass::Fast,
        "bf16_all_reduce",
        [1..=kernels_sm120::backend::BF16_LOW_LATENCY_ROWS],
    ));
    tags.push(RouteTag::new(
        kernels_sm120::comm::ll_f32::ROUND_SYMBOL,
        RouteClass::Fast,
        "rounded_all_reduce",
        [1..=kernels_sm120::backend::ROUNDED_LOW_LATENCY_ROWS],
    ));
    tags
}
