//! Whether a phase's attention folds RoPE in, and MiMo's paged and
//! sliding-window attention builders, which the tests keep as the reference
//! the backend's attention ops match (`crate::ops`).

#[cfg(test)]
use std::collections::BTreeMap;

#[cfg(test)]
use bundle_runtime::geometry::Items;
#[cfg(test)]
use kernel_core::contract::{DType, Refusal, RuntimeCounts};
#[cfg(test)]
use runtime_bundle_abi::DeviceBufferSpanV1;

#[cfg(test)]
use crate::dispatch::BufferBindings;
#[cfg(test)]
use crate::pipeline::PreparedLaunch;
#[cfg(test)]
use crate::program::Stage;
#[cfg(test)]
use crate::program_lower::DeviceInputs;
#[cfg(test)]
use crate::scratch::RankScratch;
#[cfg(test)]
use crate::selection::MimoPhase as _;
use crate::selection::Phase;
#[cfg(test)]
use crate::sm120::selection::{Selected, select};
use crate::switches::Switches;
#[cfg(test)]
use crate::topology::Tp;

/// Set to `0` to run decode on the row engine's RoPE and KV write followed by
/// the unfused attention entries.
pub const FOLD_ROPE: &str = "CHOPINFER_MIMO_ATTENTION_FOLD";

/// Whether the phase's attention folds RoPE and the KV write in, so its RoPE
/// stages launch nothing. Decode does unless `FOLD_ROPE` is `0`
/// (`Switches::fold_rope`).
pub fn folds_rope(phase: Phase, switches: Switches) -> bool {
    matches!(phase, Phase::Decode(_)) && switches.fold_rope
}

/// Whether the environment leaves decode attention folding RoPE in.
pub(crate) fn fold_rope_enabled() -> bool {
    std::env::var(FOLD_ROPE).as_deref() != Ok("0")
}

/// Plans the partial and merge, folding RoPE in when the phase does; then
/// `buffers.q` binds the packed QKV rows. Exact verify rows run the decode
/// variant's rows entries over the bucket's token rows: the row-group
/// entries for paged layers unless `selection::EXACT_ATTENTION` is `rows`.
/// Wide verify buckets under `selection::VERIFY_ATTENTION` run the per-row
/// entries, since the row-group entries hold at most 16 rows.
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
fn plan(
    variant: crate::sm120::launch::attention::Variant,
    phase: Phase,
    switches: Switches,
    prepared_batch: u32,
    counts: RuntimeCounts,
    buffers: crate::sm120::launch::attention::Buffers,
    (theta, v_multiplier): (f32, f32),
    cos_sin: Option<u64>,
) -> Result<crate::sm120::launch::attention::PlannedAttention, Refusal> {
    if phase.exact_rows(switches) {
        if switches.exact_attention_grouped
            && variant.layout == kernel_core::contract::AttentionLayout::Paged
        {
            variant.plan_rows_shared(phase.capacity()?, counts, buffers)
        } else {
            variant.plan_rows(phase.capacity()?, counts, buffers)
        }
    } else if phase.rows_attention(switches) {
        variant.plan_rows(phase.capacity()?, counts, buffers)
    } else if folds_rope(phase, switches) {
        let rope = crate::sm120::launch::attention::RopeInputs {
            qkv: buffers.q,
            theta,
            v_multiplier,
        };
        if let Some(table) = cos_sin {
            variant.plan_folded_table(prepared_batch, counts, buffers, rope, table)
        } else {
            crate::sm120::launch::attention::plan_folded(
                variant,
                prepared_batch,
                counts,
                buffers,
                rope,
            )
        }
    } else {
        crate::sm120::launch::attention::plan_for_bucket(variant, prepared_batch, counts, buffers)
    }
}

/// The attention entry planned for an instance serving `items`: the
/// catalog's, whose tables hold 32 items, or above 32 the same entry
/// prepared for the largest bucket (`Variant::with_batch_capacity`).
#[cfg(test)]
fn served_variant(
    variant: crate::sm120::launch::attention::Variant,
    items: Items,
) -> Result<crate::sm120::launch::attention::Variant, Refusal> {
    if items.capacity() <= variant.query.max_batch_capacity {
        Ok(variant)
    } else {
        variant.with_batch_capacity(items.capacity())
    }
}

/// The items a prefill phase's attention is planned for: every served item
/// of the largest bucket, and at least the catalog's 32.
#[cfg(test)]
fn prefill_batch(items: Items) -> u32 {
    items.capacity().max(crate::contract::DECODE_ENTRY_ROWS)
}

#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub fn ring(
    stage: Stage,
    tp: Tp,
    ring: crate::ring::RingGeometry,
    items: Items,
    phase: Phase,
    switches: Switches,
    live_tokens: u32,
    live_items: u32,
    context: u32,
    inputs: DeviceInputs,
    scratch: &RankScratch,
    ring_state: DeviceBufferSpanV1,
    workspace: DeviceBufferSpanV1,
    views: &BTreeMap<String, DeviceBufferSpanV1>,
) -> Result<Option<PreparedLaunch>, Refusal> {
    let layer = match stage {
        Stage::AttentionPartial(layer) | Stage::AttentionMerge(layer)
            if !tp.geometry.is_global(layer) =>
        {
            layer
        }
        _ => return Err(Refusal("MiMo stage is not sliding-window attention")),
    };
    let local = layer - tp.geometry.global_ordinal(layer) as u32;
    let (k_ring, v_ring) = crate::ring::cache_views(ring_state, local, ring)
        .ok_or(Refusal("MiMo sliding-window cache planes are missing"))?;
    let Selected::Attention(variant) = select(stage, phase, DType::Bf16, tp, switches)? else {
        return Err(Refusal("MiMo sliding-window attention entry is missing"));
    };
    let variant = served_variant(variant, items)?;
    if variant.layout != kernel_core::contract::AttentionLayout::Ring {
        return Err(Refusal("MiMo sliding-window attention selected paged KV"));
    }
    let mut bindings = BufferBindings::default();
    let q = bindings.bind(if folds_rope(phase, switches) {
        scratch.qkv
    } else {
        scratch.q
    });
    let k_cache = bindings.bind(k_ring);
    let v_cache = bindings.bind(v_ring);
    let sink = bindings.bind(
        views
            .get(&format!(
                "model.layers.{layer}.self_attn.attention_sink_bias"
            ))
            .copied()
            .ok_or(Refusal("MiMo attention sink view is missing"))?,
    );
    let (k_map, v_map) = variant.cache_maps(ring.slots, k_cache, v_cache)?;
    let prepared_batch = match phase {
        Phase::Decode(live) => {
            crate::graphs::decode_bucket(live).ok_or(Refusal("invalid MiMo decode capacity"))?
        }
        Phase::Verify(_) => crate::graphs::decode_bucket(live_items)
            .ok_or(Refusal("invalid MiMo verify item capacity"))?,
        Phase::Prefill => prefill_batch(items),
    };
    let live_batch = match phase {
        Phase::Decode(live) => live,
        Phase::Verify(_) => live_items,
        Phase::Prefill => prefill_batch(items),
    };
    let counts = RuntimeCounts {
        m: live_tokens,
        batch: live_batch,
        context,
        // Unused by row/attention entries; kept for journal identity.
        routed_rows: live_tokens.saturating_mul(8),
        split_k: 1,
    };
    let buffers = crate::sm120::launch::attention::Buffers {
        q,
        k_cache,
        v_cache,
        k_page_stride_bytes: crate::sm120::launch::tight_page_stride(
            &variant,
            variant.query.qk_dim,
        ),
        v_page_stride_bytes: crate::sm120::launch::tight_page_stride(&variant, variant.query.v_dim),
        k_map: Some(k_map),
        v_map: Some(v_map),
        block_tables: None,
        slot_ids: Some(bindings.bind(inputs.slot_ids)),
        seq_lens: Some(bindings.bind(inputs.seq_lens)),
        q_start: Some(bindings.bind(inputs.q_start)),
        q_len: Some(bindings.bind(inputs.q_len)),
        prefix_len: Some(bindings.bind(inputs.prefix_len)),
        live_batch: bindings.bind(inputs.live_batch),
        k_new: Some(bindings.bind(scratch.k_new)),
        v_new: Some(bindings.bind(scratch.v_new)),
        sink: Some(sink),
        output: bindings.bind(scratch.attention),
        workspace: bindings.bind(workspace),
        k_scale: 1.0,
        v_scale: 1.0,
    };
    let planned = plan(
        variant,
        phase,
        switches,
        prepared_batch,
        counts,
        buffers,
        (
            tp.geometry.swa_rope_theta() as f32,
            tp.geometry.semantics.attention.value_scale as f32,
        ),
        scratch.rope_tables.map(|tables| bindings.bind(tables[1])),
    )?;
    if planned.partial.workspace_bytes > workspace.length {
        return Err(Refusal("MiMo attention workspace is too small"));
    }
    let partial = matches!(stage, Stage::AttentionPartial(_));
    Ok(crate::sm120::launch::attention::stage(
        planned,
        bindings,
        live_tokens,
        partial,
    ))
}

#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub fn paged(
    stage: Stage,
    tp: Tp,
    items: Items,
    phase: Phase,
    switches: Switches,
    ga_kv: DType,
    [k_scale, v_scale]: crate::row_plan::ScaleBits,
    live_tokens: u32,
    live_items: u32,
    context: u32,
    inputs: DeviceInputs,
    scratch: &RankScratch,
    pool: DeviceBufferSpanV1,
    workspace: DeviceBufferSpanV1,
) -> Result<Option<PreparedLaunch>, Refusal> {
    let layer = match stage {
        Stage::AttentionPartial(layer) | Stage::AttentionMerge(layer)
            if tp.geometry.is_global(layer) =>
        {
            layer
        }
        _ => return Err(Refusal("MiMo stage is not global attention")),
    };
    let pages =
        crate::ga_pool::layer_pages(pool, layer, ga_kv == DType::Fp8E4M3, tp).map_err(Refusal)?;
    let Selected::Attention(variant) = select(stage, phase, ga_kv, tp, switches)? else {
        return Err(Refusal("MiMo global attention entry is missing"));
    };
    let variant = served_variant(variant, items)?;
    if variant.layout != kernel_core::contract::AttentionLayout::Paged {
        return Err(Refusal("MiMo global attention selected ring KV"));
    }
    let mut bindings = BufferBindings::default();
    let q = bindings.bind(if folds_rope(phase, switches) {
        scratch.qkv
    } else {
        scratch.q
    });
    let k_cache = bindings.bind(pages.k);
    let v_cache = bindings.bind(pages.v);
    let (k_map, v_map) = variant.cache_maps_with_strides(
        u32::try_from(pages.page_capacity)
            .map_err(|_| Refusal("MiMo page capacity exceeds u32"))?,
        k_cache,
        v_cache,
        pages.k_page_stride_bytes,
        pages.v_page_stride_bytes,
    )?;
    let prepared_batch = match phase {
        Phase::Decode(live) => {
            crate::graphs::decode_bucket(live).ok_or(Refusal("invalid MiMo decode capacity"))?
        }
        Phase::Verify(_) => crate::graphs::decode_bucket(live_items)
            .ok_or(Refusal("invalid MiMo verify item capacity"))?,
        Phase::Prefill => prefill_batch(items),
    };
    let counts = RuntimeCounts {
        m: live_tokens,
        batch: match phase {
            Phase::Decode(live) => live,
            Phase::Verify(_) => live_items,
            Phase::Prefill => prefill_batch(items),
        },
        context,
        // Unused by row/attention entries; kept for journal identity.
        routed_rows: live_tokens.saturating_mul(8),
        split_k: 1,
    };
    let buffers = crate::sm120::launch::attention::Buffers {
        q,
        k_cache,
        v_cache,
        k_page_stride_bytes: pages.k_page_stride_bytes,
        v_page_stride_bytes: pages.v_page_stride_bytes,
        k_map: Some(k_map),
        v_map: Some(v_map),
        block_tables: Some(bindings.bind(inputs.block_tables)),
        slot_ids: None,
        seq_lens: Some(bindings.bind(inputs.seq_lens)),
        q_start: Some(bindings.bind(inputs.q_start)),
        q_len: Some(bindings.bind(inputs.q_len)),
        prefix_len: Some(bindings.bind(inputs.prefix_len)),
        live_batch: bindings.bind(inputs.live_batch),
        k_new: Some(bindings.bind(scratch.k_new)),
        v_new: Some(bindings.bind(scratch.v_new)),
        sink: None,
        output: bindings.bind(scratch.attention),
        workspace: bindings.bind(workspace),
        k_scale: f32::from_bits(k_scale),
        v_scale: f32::from_bits(v_scale),
    };
    let planned = plan(
        variant,
        phase,
        switches,
        prepared_batch,
        counts,
        buffers,
        (
            tp.geometry.rope_theta() as f32,
            tp.geometry.semantics.attention.value_scale as f32,
        ),
        scratch.rope_tables.map(|tables| bindings.bind(tables[0])),
    )?;
    if planned.partial.workspace_bytes > workspace.length {
        return Err(Refusal("MiMo attention workspace is too small"));
    }
    let partial = matches!(stage, Stage::AttentionPartial(_));
    Ok(crate::sm120::launch::attention::stage(
        planned,
        bindings,
        live_tokens,
        partial,
    ))
}
