//! Concrete row launch builders for the rank program's entry stages.

#[cfg(test)]
use model_mimo26_contract::geometry::V26_FLASH;

#[cfg(test)]
use kernel_core::contract::{Argument, Refusal, RuntimeCounts};
#[cfg(test)]
use runtime_bundle_abi::DeviceBufferSpanV1;

#[cfg(test)]
use crate::dispatch::BufferBindings;
#[cfg(test)]
use crate::pipeline::PreparedLaunch;
#[cfg(test)]
use crate::program::Stage;
#[cfg(test)]
use crate::selection::{MimoPhase as _, Phase};
#[cfg(test)]
use crate::sm120::selection::{Selected, select};
#[cfg(test)]
use crate::switches::Switches;
#[cfg(test)]
use crate::topology::Tp;

#[cfg(test)]
fn counts(live_tokens: u32) -> RuntimeCounts {
    RuntimeCounts {
        m: live_tokens,
        batch: 1,
        context: 1,
        // Unused by row/attention entries; kept for journal identity.
        routed_rows: 8,
        split_k: 1,
    }
}

#[cfg(test)]
fn row_entry(
    stage: Stage,
    phase: Phase,
    tp: Tp,
    switches: Switches,
) -> Result<crate::sm120::launch::row::RowInstantiation, Refusal> {
    match select(
        stage,
        phase,
        kernel_core::contract::DType::Bf16,
        tp,
        switches,
    )? {
        Selected::Row(entry) => Ok(entry),
        _ => Err(Refusal("rank entry stage did not select a row kernel")),
    }
}

#[cfg(test)]
/// Device inputs the ring backup reads for one phase.
pub struct RingBackupViews {
    pub ring_state: DeviceBufferSpanV1,
    pub slot_ids: DeviceBufferSpanV1,
    pub q_len: DeviceBufferSpanV1,
    pub prefix_len: DeviceBufferSpanV1,
    pub live_batch: DeviceBufferSpanV1,
}

#[cfg(test)]
/// One block per (item, recent position, SWA layer). Decode items touch one
/// position, so a decode bucket plans `bucket` block rows and the launch is
/// the same for every live count in it; prefill covers the last
/// `RING_TOKENS` positions of each of `items` items.
pub fn ring_backup(
    tp: Tp,
    ring: crate::ring::RingGeometry,
    phase: Phase,
    items: u32,
    views: RingBackupViews,
) -> Result<PreparedLaunch, Refusal> {
    let (rows, span) = match phase {
        Phase::Decode(live) => (
            crate::graphs::decode_bucket(live).ok_or(Refusal("invalid MiMo decode capacity"))?,
            1,
        ),
        // A verify item writes at most `MAX_VERIFY_ROWS` positions.
        Phase::Verify(_) => (
            crate::graphs::decode_bucket(items).ok_or(Refusal("invalid MiMo verify items"))?,
            crate::contract::MAX_VERIFY_ROWS,
        ),
        Phase::Prefill => (items, tp.geometry.window()),
    };
    if rows == 0 || rows > ring.slots {
        return Err(Refusal("MiMo ring backup item count"));
    }
    if views.ring_state.length < 2 * ring.working {
        return Err(Refusal("MiMo ring backup state is too small"));
    }
    let mut bindings = BufferBindings::default();
    let arguments = kernel_core::contract::ArgumentList(vec![
        Argument::DeviceBuffer(bindings.bind(views.ring_state)),
        Argument::DeviceBuffer(bindings.bind(views.slot_ids)),
        Argument::DeviceBuffer(bindings.bind(views.q_len)),
        Argument::DeviceBuffer(bindings.bind(views.prefix_len)),
        Argument::DeviceBuffer(bindings.bind(views.live_batch)),
        Argument::U32(span),
    ]);
    Ok(PreparedLaunch {
        symbol: crate::sm120::ring_entries::ring_backup_symbol(tp, ring.slots),
        rows: rows * span,
        plan: kernel_core::contract::LaunchPlan {
            grid: [rows * span, tp.geometry.sliding_layers() as u32, 1],
            block: [crate::topology::ring_threads(tp), 1, 1],
            dynamic_shared_bytes: 0,
            workspace_bytes: 0,
            cluster: [1, 1, 1],
            pdl: true,
        },
        arguments,
        bindings,
    })
}

#[cfg(test)]
/// The bundle-owned kernel that restores a verify step's rejected ring
/// tails (`chop_mimo_ring_restore_v1` at TP4 and 32 ring slots).
pub fn ring_restore_symbol(tp: Tp, slots: u32) -> String {
    crate::sm120::ring_entries::ring_symbol(tp, "ring_restore", slots)
}

#[cfg(test)]
/// Device inputs of one ring restore: per item, the slot, first rejected
/// position and rejected count, then the item count.
pub struct RingRestoreViews {
    pub ring_state: DeviceBufferSpanV1,
    pub slot_ids: DeviceBufferSpanV1,
    pub starts: DeviceBufferSpanV1,
    pub counts: DeviceBufferSpanV1,
    pub live_items: DeviceBufferSpanV1,
}

#[cfg(test)]
/// One block per (item, rejected position, SWA layer). `span` bounds each
/// item's rejected count; the launch covers `items` items.
pub fn ring_restore(
    tp: Tp,
    ring: crate::ring::RingGeometry,
    items: u32,
    span: u32,
    views: RingRestoreViews,
) -> Result<PreparedLaunch, Refusal> {
    if items == 0 || items > ring.slots || span == 0 {
        return Err(Refusal("MiMo ring restore geometry"));
    }
    if views.ring_state.length < 2 * ring.working {
        return Err(Refusal("MiMo ring restore state is too small"));
    }
    let mut bindings = BufferBindings::default();
    let arguments = kernel_core::contract::ArgumentList(vec![
        Argument::DeviceBuffer(bindings.bind(views.ring_state)),
        Argument::DeviceBuffer(bindings.bind(views.slot_ids)),
        Argument::DeviceBuffer(bindings.bind(views.starts)),
        Argument::DeviceBuffer(bindings.bind(views.counts)),
        Argument::DeviceBuffer(bindings.bind(views.live_items)),
        Argument::U32(span),
    ]);
    Ok(PreparedLaunch {
        symbol: ring_restore_symbol(tp, ring.slots),
        rows: items * span,
        plan: kernel_core::contract::LaunchPlan {
            grid: [items * span, tp.geometry.sliding_layers() as u32, 1],
            block: [crate::topology::ring_threads(tp), 1, 1],
            dynamic_shared_bytes: 0,
            workspace_bytes: 0,
            cluster: [1, 1, 1],
            pdl: true,
        },
        arguments,
        bindings,
    })
}

/// Copies `rows` rows of a 4096-wide table, chosen by device row ids, into
/// consecutive output rows; the embedding kernel with the table's own rows as
/// its vocabulary. Prefill gathered the rows that need logits this way.
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub fn gather_rows(
    tp: Tp,
    switches: Switches,
    rows: u32,
    table: DeviceBufferSpanV1,
    ids: DeviceBufferSpanV1,
    output: DeviceBufferSpanV1,
    live_rows: DeviceBufferSpanV1,
    table_rows: u32,
) -> Result<PreparedLaunch, Refusal> {
    let width = tp.geometry.hidden() as u64 * 2;
    if rows == 0
        || rows > table_rows
        || table.length < u64::from(table_rows) * width
        || output.length < u64::from(rows) * width
        || ids.length < u64::from(rows) * 4
    {
        return Err(Refusal("MiMo row gather geometry"));
    }
    let entry = row_entry(Stage::GatherHeadRows, Phase::Prefill, tp, switches)?;
    let mut bindings = BufferBindings::default();
    let table = bindings.bind(table);
    let ids = bindings.bind(ids);
    let output = bindings.bind(output);
    let live_rows = bindings.bind(live_rows);
    let (plan, arguments) = crate::sm120::launch::row::plan(
        &entry,
        Phase::Prefill.row_grid(rows, switches, crate::sm120::ROW_BUCKETS)?,
        counts(rows),
        vec![
            Argument::DeviceBuffer(table),
            Argument::DeviceBuffer(ids),
            Argument::DeviceBuffer(output),
            Argument::DeviceBuffer(live_rows),
            Argument::U32(0),
            Argument::U32(table_rows),
        ],
    )?;
    Ok(PreparedLaunch {
        symbol: entry.symbol,
        rows,
        plan,
        arguments,
        bindings,
    })
}

#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub fn embedding(
    tp: Tp,
    phase: Phase,
    switches: Switches,
    live_tokens: u32,
    table: DeviceBufferSpanV1,
    ids: DeviceBufferSpanV1,
    output: DeviceBufferSpanV1,
    live_count: DeviceBufferSpanV1,
) -> Result<PreparedLaunch, Refusal> {
    let capacity = phase.row_grid(live_tokens, switches, crate::sm120::ROW_BUCKETS)?;
    let Selected::Row(entry) = crate::sm120::selection::row_for(tp, "embedding_full")? else {
        return Err(Refusal("embedding is not a row kernel"));
    };
    embedding_entry(
        &entry,
        tp,
        capacity,
        live_tokens,
        EmbeddingViews {
            table,
            ids,
            output,
            live_count,
        },
    )
}

#[cfg(test)]
pub(crate) struct EmbeddingViews {
    pub table: DeviceBufferSpanV1,
    pub ids: DeviceBufferSpanV1,
    pub output: DeviceBufferSpanV1,
    pub live_count: DeviceBufferSpanV1,
}

#[cfg(test)]
pub(crate) fn embedding_entry(
    entry: &crate::sm120::launch::row::RowInstantiation,
    tp: Tp,
    capacity: u32,
    live_tokens: u32,
    views: EmbeddingViews,
) -> Result<PreparedLaunch, Refusal> {
    let EmbeddingViews {
        table,
        ids,
        output,
        live_count,
    } = views;
    let mut bindings = BufferBindings::default();
    let table = bindings.bind(table);
    let ids = bindings.bind(ids);
    let output = bindings.bind(output);
    let live_count = bindings.bind(live_count);
    let (plan, arguments) = crate::sm120::launch::row::plan(
        entry,
        capacity,
        counts(live_tokens),
        vec![
            Argument::DeviceBuffer(table),
            Argument::DeviceBuffer(ids),
            Argument::DeviceBuffer(output),
            Argument::DeviceBuffer(live_count),
            Argument::U32(0),
            Argument::U32(tp.geometry.vocab()),
        ],
    )?;
    Ok(PreparedLaunch {
        symbol: entry.symbol.clone(),
        rows: capacity,
        plan,
        arguments,
        bindings,
    })
}

/// The hidden norm as the row engine planned it before `ops` prepared it,
/// which tests compare the prepared op with.
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub fn initial_norm(
    tp: Tp,
    phase: Phase,
    switches: Switches,
    live_tokens: u32,
    input: DeviceBufferSpanV1,
    weight: DeviceBufferSpanV1,
    output: DeviceBufferSpanV1,
    live_count: DeviceBufferSpanV1,
) -> Result<PreparedLaunch, Refusal> {
    let capacity = phase.row_grid(live_tokens, switches, crate::sm120::ROW_BUCKETS)?;
    let Selected::Row(entry) = crate::sm120::selection::row_for(tp, "rms_hidden")? else {
        return Err(Refusal("MiMo hidden norm entry is missing"));
    };
    let mut bindings = BufferBindings::default();
    let input = bindings.bind(input);
    let weight = bindings.bind(weight);
    let output = bindings.bind(output);
    let live_count = bindings.bind(live_count);
    let (plan, arguments) = crate::sm120::launch::row::plan(
        &entry,
        capacity,
        counts(live_tokens),
        vec![
            Argument::DeviceBuffer(input),
            Argument::DeviceBuffer(weight),
            Argument::NullDevicePointer,
            Argument::NullDevicePointer,
            Argument::DeviceBuffer(output),
            Argument::DeviceBuffer(live_count),
            Argument::U32((tp.geometry.eps() as f32).to_bits()),
        ],
    )?;
    Ok(PreparedLaunch {
        symbol: entry.symbol,
        rows: capacity,
        plan,
        arguments,
        bindings,
    })
}

#[cfg(test)]
fn swa_layer(tp: Tp, layer: u32) -> Result<u32, Refusal> {
    if layer >= tp.geometry.layers() || tp.geometry.is_global(layer) {
        return Err(Refusal("MiMo sliding-window layer is invalid"));
    }
    Ok(layer - tp.geometry.global_ordinal(layer) as u32)
}

/// MiMo's RoPE base on sliding-window and global-attention layers, and the
/// multiplier applied to V with it.
#[cfg(test)]
pub const SWA_ROPE_THETA: f32 = V26_FLASH.swa_rope_theta() as f32;
#[cfg(test)]
pub const GA_ROPE_THETA: f32 = V26_FLASH.rope_theta() as f32;
#[cfg(test)]
pub const V_MULTIPLIER: f32 = V26_FLASH.semantics.attention.value_scale as f32;

#[cfg(test)]
pub struct SwaRopeViews {
    pub cos_sin: DeviceBufferSpanV1,
    pub qkv: DeviceBufferSpanV1,
    pub q: DeviceBufferSpanV1,
    pub k_new: DeviceBufferSpanV1,
    pub v_new: DeviceBufferSpanV1,
    pub ring_state: DeviceBufferSpanV1,
    /// The ring's layout: its slots and planes.
    pub ring: crate::ring::RingGeometry,
    pub slot_ids: DeviceBufferSpanV1,
    pub token_batch: DeviceBufferSpanV1,
    pub positions: DeviceBufferSpanV1,
    pub live_tokens: DeviceBufferSpanV1,
}

#[cfg(test)]
pub fn rope_swa(
    layer: u32,
    phase: Phase,
    switches: Switches,
    live_tokens: u32,
    views: SwaRopeViews,
    tp: Tp,
) -> Result<PreparedLaunch, Refusal> {
    let local = swa_layer(tp, layer)?;
    let (k_ring, v_ring) = crate::ring::cache_views(views.ring_state, local, views.ring)
        .ok_or(Refusal("MiMo SWA state has no cache planes"))?;
    let Selected::Row(entry) = select(
        Stage::RopeForSlidingWindow(layer),
        phase,
        kernel_core::contract::DType::Bf16,
        tp,
        switches,
    )?
    else {
        return Err(Refusal("MiMo SWA RoPE row entry is missing"));
    };
    let capacity = phase.row_grid(live_tokens, switches, crate::sm120::ROW_BUCKETS)?;
    let mut bindings = BufferBindings::default();
    let mut bind = |span| Argument::DeviceBuffer(bindings.bind(span));
    let args = vec![
        bind(views.qkv),
        bind(views.q),
        bind(views.k_new),
        bind(views.v_new),
        bind(k_ring),
        bind(v_ring),
        Argument::NullDevicePointer,
        bind(views.slot_ids),
        bind(views.token_batch),
        bind(views.positions),
        bind(views.cos_sin),
        bind(views.live_tokens),
        Argument::U32(tp.model_limits().max_pages()),
        Argument::U32(tp.geometry.window()),
        Argument::U64(0),
        Argument::U64(0),
        Argument::U32((tp.geometry.swa_rope_theta() as f32).to_bits()),
        Argument::U32((tp.geometry.semantics.attention.value_scale as f32).to_bits()),
        Argument::U32(1.0_f32.to_bits()),
        Argument::U32(1.0_f32.to_bits()),
    ];
    let (plan, arguments) =
        crate::sm120::launch::row::plan(&entry, capacity, counts(live_tokens), args)?;
    Ok(PreparedLaunch {
        symbol: entry.symbol,
        rows: capacity,
        plan,
        arguments,
        bindings,
    })
}

#[cfg(test)]
pub struct GaRopeViews {
    pub cos_sin: DeviceBufferSpanV1,
    pub qkv: DeviceBufferSpanV1,
    pub q: DeviceBufferSpanV1,
    pub k_new: DeviceBufferSpanV1,
    pub v_new: DeviceBufferSpanV1,
    pub pages: crate::ga_pool::LayerPages,
    pub block_tables: DeviceBufferSpanV1,
    pub token_batch: DeviceBufferSpanV1,
    pub positions: DeviceBufferSpanV1,
    pub live_tokens: DeviceBufferSpanV1,
}

/// The bits of a global layer's K and V cache scales.
#[cfg(test)]
pub type ScaleBits = [u32; 2];

#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub fn rope_ga(
    layer: u32,
    phase: Phase,
    switches: Switches,
    ga_kv: kernel_core::contract::DType,
    [k_scale, v_scale]: ScaleBits,
    live_tokens: u32,
    tp: Tp,
    views: GaRopeViews,
) -> Result<PreparedLaunch, Refusal> {
    if !tp.geometry.is_global(layer) || views.pages.page_capacity == 0 {
        return Err(Refusal("MiMo global-attention page geometry is invalid"));
    }
    let Selected::Row(entry) = select(Stage::RopeAndPagedWrite(layer), phase, ga_kv, tp, switches)?
    else {
        return Err(Refusal("MiMo paged RoPE row entry is missing"));
    };
    let capacity = phase.row_grid(live_tokens, switches, crate::sm120::ROW_BUCKETS)?;
    let mut bindings = BufferBindings::default();
    let mut bind = |span| Argument::DeviceBuffer(bindings.bind(span));
    let args = vec![
        bind(views.qkv),
        bind(views.q),
        bind(views.k_new),
        bind(views.v_new),
        bind(views.pages.k),
        bind(views.pages.v),
        bind(views.block_tables),
        Argument::NullDevicePointer,
        bind(views.token_batch),
        bind(views.positions),
        bind(views.cos_sin),
        bind(views.live_tokens),
        Argument::U32(tp.model_limits().max_pages()),
        Argument::U32(0),
        Argument::U64(views.pages.k_page_stride_bytes),
        Argument::U64(views.pages.v_page_stride_bytes),
        Argument::U32((tp.geometry.rope_theta() as f32).to_bits()),
        Argument::U32((tp.geometry.semantics.attention.value_scale as f32).to_bits()),
        Argument::U32(k_scale),
        Argument::U32(v_scale),
    ];
    let (plan, arguments) =
        crate::sm120::launch::row::plan(&entry, capacity, counts(live_tokens), args)?;
    Ok(PreparedLaunch {
        symbol: entry.symbol,
        rows: capacity,
        plan,
        arguments,
        bindings,
    })
}

#[cfg(test)]
pub fn ring_write_swa(
    layer: u32,
    phase: Phase,
    switches: Switches,
    live_tokens: u32,
    views: SwaRopeViews,
    last_positions: DeviceBufferSpanV1,
    tp: Tp,
) -> Result<PreparedLaunch, Refusal> {
    let local = swa_layer(tp, layer)?;
    let (k_ring, v_ring) = crate::ring::cache_views(views.ring_state, local, views.ring)
        .ok_or(Refusal("MiMo SWA state has no cache planes"))?;
    let Selected::Row(entry) = select(
        Stage::SlidingWindowWrite(layer),
        phase,
        kernel_core::contract::DType::Bf16,
        tp,
        switches,
    )?
    else {
        return Err(Refusal("MiMo SWA ring-write entry is missing"));
    };
    let capacity = phase.row_grid(live_tokens, switches, crate::sm120::ROW_BUCKETS)?;
    let mut bindings = BufferBindings::default();
    let mut bind = |span| Argument::DeviceBuffer(bindings.bind(span));
    let args = vec![
        bind(views.k_new),
        bind(views.v_new),
        bind(k_ring),
        bind(v_ring),
        bind(views.slot_ids),
        bind(views.token_batch),
        bind(views.positions),
        bind(last_positions),
        bind(views.live_tokens),
        Argument::U32(tp.geometry.window()),
        Argument::U32(1.0_f32.to_bits()),
        Argument::U32(1.0_f32.to_bits()),
    ];
    let (plan, arguments) =
        crate::sm120::launch::row::plan(&entry, capacity, counts(live_tokens), args)?;
    Ok(PreparedLaunch {
        symbol: entry.symbol,
        rows: capacity,
        plan,
        arguments,
        bindings,
    })
}

/// Routed-row tiles of 256 routes at prefill capacity, and the bytes of one
/// per-tile table of expert counts.
#[cfg(test)]
const ROUTE_TILES: u64 = crate::contract::PREFILL_TOKENS as u64 * V26_FLASH.top_k() as u64 / 256;
#[cfg(test)]
const ROUTE_TILE_TABLE_BYTES: u64 = ROUTE_TILES * V26_FLASH.experts() as u64 * 4;

#[cfg(test)]
#[derive(Clone, Copy)]
pub struct MoeRowBuffers {
    pub router_logits: DeviceBufferSpanV1,
    pub router_bias: DeviceBufferSpanV1,
    pub topk_ids: DeviceBufferSpanV1,
    pub topk_weights: DeviceBufferSpanV1,
    pub counts: DeviceBufferSpanV1,
    pub expert_offsets: DeviceBufferSpanV1,
    pub sorted_rows: DeviceBufferSpanV1,
    pub padded_rows: DeviceBufferSpanV1,
    pub activation: DeviceBufferSpanV1,
    pub gathered: DeviceBufferSpanV1,
    pub routed_output: DeviceBufferSpanV1,
    pub combined: DeviceBufferSpanV1,
    pub live_tokens: DeviceBufferSpanV1,
    /// Shared scratch; the prefill route keeps its per-tile expert counts
    /// and offsets here between its count, scan and scatter stages.
    pub workspace: DeviceBufferSpanV1,
}

/// Launches of one MoE row stage, as MiMo built them before `ops`
/// prepared its routes and combine, which tests compare the ops with. A
/// prefill route runs the 256-route tile kernels: count, then prefix and
/// scan, then scatter.
#[cfg(test)]
pub fn moe_row(
    stage: Stage,
    phase: Phase,
    switches: Switches,
    live_tokens: u32,
    tp: Tp,
    buffers: MoeRowBuffers,
) -> Result<Vec<PreparedLaunch>, Refusal> {
    let row = |name: &str| match crate::sm120::selection::row_for(tp, name)? {
        Selected::Row(entry) => Ok(entry),
        _ => Err(Refusal("MiMo MoE row stage is missing")),
    };
    // Prefill routes ran the 256-route tile kernels; the scan stage also
    // launched `permute_tile_prefix`. Other buckets ran the decode route.
    let tiled = phase.prefill_route();
    let entry = row(match stage {
        Stage::RouterTop8(_) => "router",
        Stage::PermuteCount(_) if tiled => "permute_tile_count_mimo_b64",
        Stage::PermuteScan(_) if tiled => "permute_tile_scan_mimo_b64",
        Stage::PermuteScatter(_) if tiled => "permute_tile_scatter_mimo_b64",
        Stage::CombineExperts(_) => "combine_hidden",
        _ => return Err(Refusal("not a MiMo MoE row stage")),
    })?;
    let capacity = phase.capacity()?;
    let [tile_counts, tile_offsets] = crate::ops::route_tile_tables(tp, buffers.workspace)?;
    let mut launches = Vec::new();
    if matches!(stage, Stage::PermuteScan(_)) {
        let prefix = row("permute_tile_prefix_mimo_b64")?;
        let mut bindings = BufferBindings::default();
        let mut bind = |span| Argument::DeviceBuffer(bindings.bind(span));
        let args = vec![
            bind(tile_counts),
            bind(buffers.counts),
            bind(tile_offsets),
            bind(buffers.live_tokens),
        ];
        let (plan, arguments) =
            crate::sm120::launch::row::plan(&prefix, capacity, counts(live_tokens), args)?;
        launches.push(PreparedLaunch {
            symbol: prefix.symbol,
            rows: capacity,
            plan,
            arguments,
            bindings,
        });
    }
    let mut bindings = BufferBindings::default();
    let mut bind = |span| Argument::DeviceBuffer(bindings.bind(span));
    let args = match stage {
        Stage::PermuteCount(_) => vec![
            bind(buffers.topk_ids),
            bind(tile_counts),
            bind(buffers.live_tokens),
        ],
        Stage::PermuteScan(_) => vec![
            bind(buffers.counts),
            bind(buffers.expert_offsets),
            bind(buffers.sorted_rows),
            bind(buffers.padded_rows),
        ],
        Stage::PermuteScatter(_) => vec![
            bind(buffers.topk_ids),
            bind(tile_offsets),
            bind(buffers.expert_offsets),
            bind(buffers.sorted_rows),
            bind(buffers.live_tokens),
        ],
        Stage::RouterTop8(_) => vec![
            bind(buffers.router_logits),
            bind(buffers.router_bias),
            bind(buffers.topk_ids),
            bind(buffers.topk_weights),
            bind(buffers.live_tokens),
            Argument::U32(1.0_f32.to_bits()),
        ],
        Stage::CombineExperts(_) => vec![
            bind(buffers.routed_output),
            bind(buffers.topk_weights),
            bind(buffers.combined),
            bind(buffers.live_tokens),
        ],
        _ => return Err(Refusal("not a MiMo MoE row stage")),
    };
    // The router top-8 and combine grids are one block per token; the
    // permute kernels keep the capacity their route tile tables assume.
    let grid = if matches!(stage, Stage::RouterTop8(_) | Stage::CombineExperts(_)) {
        phase.row_grid(live_tokens, switches, crate::sm120::ROW_BUCKETS)?
    } else {
        capacity
    };
    let (plan, arguments) =
        crate::sm120::launch::row::plan(&entry, grid, counts(live_tokens), args)?;
    launches.push(PreparedLaunch {
        symbol: entry.symbol,
        rows: grid,
        plan,
        arguments,
        bindings,
    });
    Ok(launches)
}

#[cfg(test)]
mod tests {
    use runtime_bundle_abi::{BundleDeviceBufferHandleV1, KernelArgumentV1};

    use super::*;

    #[test]
    fn ring_restore_covers_items_by_span_and_every_sliding_layer() {
        let buffer = BundleDeviceBufferHandleV1::from_parts(1, 1);
        let span = |bytes| DeviceBufferSpanV1::new(buffer, 0, bytes);
        let views = || RingRestoreViews {
            ring_state: span(crate::contract::RING_STATE_BYTES_PER_RANK),
            slot_ids: span(128),
            starts: span(128),
            counts: span(128),
            live_items: span(4),
        };
        let ring = |tp| crate::ring::RingGeometry::new(tp, 32);
        let launch = ring_restore(Tp::TP4, ring(Tp::TP4), 3, 8, views()).unwrap();
        assert_eq!(launch.symbol, "chop_mimo_ring_restore_v1");
        assert_eq!(launch.plan.block, [96, 1, 1]);
        assert_eq!(
            launch.plan.grid,
            [24, crate::contract::SWA_LAYERS as u32, 1]
        );
        assert!(
            crate::CUBINS
                .iter()
                .any(|cubin| cubin.symbols.contains(&launch.symbol.as_str()))
        );
        assert!(ring_restore(Tp::TP4, ring(Tp::TP4), 0, 8, views()).is_err());
        assert!(ring_restore(Tp::TP4, ring(Tp::TP4), 33, 8, views()).is_err());
        assert!(ring_restore(Tp::TP4, ring(Tp::TP4), 1, 0, views()).is_err());
        // TP2's ring is twice as wide; its kernels take more threads.
        let tp2 = RingRestoreViews {
            ring_state: span(crate::topology::ring_state_bytes_per_rank(Tp::TP2)),
            ..views()
        };
        let launch = ring_restore(Tp::TP2, ring(Tp::TP2), 3, 8, tp2).unwrap();
        assert_eq!(launch.symbol, "chop_mimo_tp2_ring_restore_v1");
        assert_eq!(launch.plan.block, [160, 1, 1]);
        assert!(ring_restore(Tp::TP2, ring(Tp::TP2), 3, 8, views()).is_err());
        // Another slot count runs its own entry over a state of its size.
        for (slots, served) in [(8, 5), (64, 40), (256, 255)] {
            let wide = crate::ring::RingGeometry::new(Tp::TP4, slots);
            let state = RingRestoreViews {
                ring_state: span(crate::topology::ring_state_bytes(Tp::TP4, slots)),
                ..views()
            };
            let launch = ring_restore(Tp::TP4, wide, served, 8, state).unwrap();
            assert_eq!(launch.symbol, format!("chop_mimo_ring_restore_s{slots}_v1"));
            assert_eq!(
                launch.plan.grid,
                [served * 8, crate::contract::SWA_LAYERS as u32, 1]
            );
            assert!(
                crate::CUBINS
                    .iter()
                    .any(|cubin| cubin.name == "mimo_ring_slots"
                        && cubin.symbols.contains(&launch.symbol.as_str()))
            );
            assert!(ring_restore(Tp::TP4, wide, slots + 1, 8, views()).is_err());
            if slots > 32 {
                // A 32-slot state is too small for a wider ring.
                assert!(ring_restore(Tp::TP4, wide, 1, 8, views()).is_err());
            }
        }
    }

    #[test]
    fn entry_rows_pack_full_embedding_and_hf_norm_for_each_phase() {
        let buffer = BundleDeviceBufferHandleV1::from_parts(1, 1);
        let span = |offset, length| DeviceBufferSpanV1::new(buffer, offset, length);
        for (phase, live) in [(Phase::Decode(3), 3), (Phase::Prefill, 500)] {
            let embedding = embedding(
                Tp::TP4,
                phase,
                Switches::configured(),
                live,
                span(0, 1_249_902_592),
                span(2_000_000_000, 32768),
                span(2_100_000_000, 67_108_864),
                span(2_200_000_000, 4),
            )
            .unwrap();
            assert!(embedding.plan.pdl);
            assert_eq!(embedding.symbol, "chop_row_embedding_full_v1");
            let args = crate::dispatch::arguments(&embedding.arguments, &embedding.bindings)
                .expect("embedding host ABI");
            assert_eq!(args[5], KernelArgumentV1::ScalarU32(152_576));

            let norm = initial_norm(
                Tp::TP4,
                phase,
                Switches::configured(),
                live,
                span(2_100_000_000, 67_108_864),
                span(2_300_000_000, 8192),
                span(2_400_000_000, 67_108_864),
                span(2_200_000_000, 4),
            )
            .unwrap();
            assert!(norm.plan.pdl);
            assert_eq!(norm.symbol, "chop_row_rms_mimo_hf4096_v1");
            let args =
                crate::dispatch::arguments(&norm.arguments, &norm.bindings).expect("norm host ABI");
            assert_eq!(args[2], KernelArgumentV1::NullDevicePointer);
            assert_eq!(args[3], KernelArgumentV1::NullDevicePointer);
            assert_eq!(args[6], KernelArgumentV1::ScalarU32(1e-6_f32.to_bits()));
        }
    }

    #[test]
    fn swa_rope_and_tail_write_bind_the_same_layer_planes() {
        let buffer = BundleDeviceBufferHandleV1::from_parts(1, 1);
        let span = |offset, length| DeviceBufferSpanV1::new(buffer, offset, length);
        let make_views = || SwaRopeViews {
            cos_sin: span(800, 256 * 128),
            qkv: span(0, 8192 * 3712 * 2),
            q: span(100, 8192 * 16 * 192 * 2),
            k_new: span(200, 8192 * 2 * 192 * 2),
            v_new: span(300, 8192 * 2 * 128 * 2),
            ring_state: span(4096, crate::contract::RING_STATE_BYTES_PER_RANK),
            ring: crate::ring::RingGeometry::new(Tp::TP4, 32),
            slot_ids: span(400, 32 * 4),
            token_batch: span(500, 8192 * 4),
            positions: span(600, 8192 * 4),
            live_tokens: span(700, 4),
        };
        for phase in [Phase::Decode(1), Phase::Prefill] {
            let rope = rope_swa(1, phase, Switches::configured(), 1, make_views(), Tp::TP4)
                .expect("SWA RoPE");
            assert!(rope.plan.pdl);
            let args = crate::dispatch::arguments(&rope.arguments, &rope.bindings).unwrap();
            assert_eq!(args.len(), 20);
            assert!(rope.symbol.contains("_table_"));
            assert_eq!(
                args[10],
                KernelArgumentV1::DeviceBuffer(span(800, 256 * 128))
            );
            assert_eq!(args[6], KernelArgumentV1::NullDevicePointer);
            assert_eq!(args[9], KernelArgumentV1::DeviceBuffer(span(600, 8192 * 4)));
            assert_eq!(args[14], KernelArgumentV1::ScalarU64(0));
            assert_eq!(args[15], KernelArgumentV1::ScalarU64(0));
            let write = ring_write_swa(
                1,
                phase,
                Switches::configured(),
                1,
                make_views(),
                span(800, 32 * 8),
                Tp::TP4,
            )
            .expect("SWA tail write");
            let write_args = crate::dispatch::arguments(&write.arguments, &write.bindings).unwrap();
            assert_eq!(write_args.len(), 12);
            assert_eq!(args[4], write_args[2]);
            assert_eq!(args[5], write_args[3]);
        }
        assert!(
            rope_swa(
                0,
                Phase::Decode(1),
                Switches::configured(),
                1,
                make_views(),
                Tp::TP4
            )
            .is_err()
        );
    }

    #[test]
    fn global_rope_uses_each_layers_interleaved_host_page_stride() {
        let buffer = BundleDeviceBufferHandleV1::from_parts(1, 1);
        let span = |offset, length| DeviceBufferSpanV1::new(buffer, offset, length);
        let pages = crate::ga_pool::layer_pages(
            span(4096, 128 * crate::contract::GA_PAGE_BYTES_BF16),
            5,
            false,
            Tp::TP4,
        )
        .unwrap();
        let launch = rope_ga(
            5,
            Phase::Decode(3),
            Switches::configured(),
            kernel_core::contract::DType::Bf16,
            [1.0_f32.to_bits(); 2],
            3,
            Tp::TP4,
            GaRopeViews {
                cos_sin: span(800, 256 * 128),
                qkv: span(0, 8192 * 3712 * 2),
                q: span(100, 8192 * 16 * 192 * 2),
                k_new: span(200, 8192 * 2 * 192 * 2),
                v_new: span(300, 8192 * 2 * 128 * 2),
                pages,
                block_tables: span(400, 32 * 16_384 * 4),
                token_batch: span(500, 8192 * 4),
                positions: span(600, 8192 * 4),
                live_tokens: span(700, 4),
            },
        )
        .unwrap();
        assert!(launch.plan.pdl);
        let args = crate::dispatch::arguments(&launch.arguments, &launch.bindings).unwrap();
        assert!(launch.symbol.contains("_table_"));
        assert_eq!(
            args[10],
            KernelArgumentV1::DeviceBuffer(span(800, 256 * 128))
        );
        assert_eq!(args[4], KernelArgumentV1::DeviceBuffer(pages.k));
        assert_eq!(args[5], KernelArgumentV1::DeviceBuffer(pages.v));
        assert_eq!(
            args[14],
            KernelArgumentV1::ScalarU64(crate::contract::GA_PAGE_BYTES_BF16)
        );
        assert_eq!(args[15], args[14]);
        assert_eq!(args.len(), 20);
    }

    #[test]
    fn moe_router_permute_gather_and_combine_pack_selected_abis() {
        let buffer = BundleDeviceBufferHandleV1::from_parts(1, 1);
        let span = |offset| DeviceBufferSpanV1::new(buffer, offset, 8192 * 4096 * 16);
        let buffers = MoeRowBuffers {
            router_logits: span(1),
            router_bias: span(2),
            topk_ids: span(3),
            topk_weights: span(4),
            counts: span(5),
            expert_offsets: span(6),
            sorted_rows: span(7),
            padded_rows: span(8),
            activation: span(9),
            gathered: span(10),
            routed_output: span(11),
            combined: span(12),
            live_tokens: span(13),
            workspace: span(14),
        };
        // Decode buckets of up to 16 rows ran the decode route instead.
        assert!(
            moe_row(
                Stage::PermuteCount(1),
                Phase::Decode(3),
                Switches::configured(),
                1,
                Tp::TP4,
                buffers
            )
            .is_err()
        );
        for phase in [Phase::Decode(32), Phase::Prefill] {
            for (stage, launches) in [
                (Stage::RouterTop8(1), vec![("router_mimo_sigmoid256", 6)]),
                (
                    Stage::PermuteCount(1),
                    vec![("permute_tile_count_mimo_b64", 3)],
                ),
                (
                    Stage::PermuteScan(1),
                    vec![
                        ("permute_tile_prefix_mimo_b64", 4),
                        ("permute_tile_scan_mimo_b64", 4),
                    ],
                ),
                (
                    Stage::PermuteScatter(1),
                    vec![("permute_tile_scatter_mimo_b64", 5)],
                ),
                (Stage::CombineExperts(1), vec![("combine_mimo4096", 4)]),
            ] {
                let planned = moe_row(stage, phase, Switches::configured(), 1, Tp::TP4, buffers)
                    .expect("MiMo MoE row plan");
                assert_eq!(planned.len(), launches.len());
                for (launch, (name, arguments)) in planned.iter().zip(launches) {
                    assert_eq!(launch.symbol, format!("chop_row_{name}_v1"));
                    assert!(launch.plan.pdl);
                    assert_eq!(
                        crate::dispatch::arguments(&launch.arguments, &launch.bindings)
                            .unwrap()
                            .len(),
                        arguments
                    );
                }
            }
        }
        // The tile tables sit at the start of the workspace and refuse a
        // workspace that cannot hold both.
        let small = MoeRowBuffers {
            workspace: DeviceBufferSpanV1::new(buffer, 14, 2 * ROUTE_TILE_TABLE_BYTES - 1),
            ..buffers
        };
        assert!(
            moe_row(
                Stage::PermuteCount(1),
                Phase::Prefill,
                Switches::configured(),
                1,
                Tp::TP4,
                small
            )
            .is_err()
        );
        let count = &moe_row(
            Stage::PermuteCount(1),
            Phase::Prefill,
            Switches::configured(),
            1,
            Tp::TP4,
            buffers,
        )
        .unwrap()[0];
        let args = crate::dispatch::arguments(&count.arguments, &count.bindings).unwrap();
        assert_eq!(
            args[1],
            KernelArgumentV1::DeviceBuffer(DeviceBufferSpanV1::new(
                buffer,
                14,
                ROUTE_TILE_TABLE_BYTES
            ))
        );
    }
}
