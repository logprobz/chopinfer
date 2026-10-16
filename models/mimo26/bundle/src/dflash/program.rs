//! DFlash drafter launches, lowered next to the target's rank program.
//!
//! Three pieces join the target program of every phase:
//!
//! - **Taps.** After the reductions of target layers 0, 11, 23, 35 and 47,
//!   the residual rows are copied into this rank's K shard of the
//!   concatenated five-layer context feature.
//! - **Context.** After the vocabulary head, `fc` (K-split, all-reduced with
//!   `hidden_norm`), the fused five-layer K/V projection, `k_norm` and partial
//!   RoPE write every row's context K and 0.612-scaled V into the per-slot
//!   circular cache at the row's position.
//! - **Block.** A verify phase first drafts: its rows become the draft block
//!   (the bonus token's embedding, then the mask embedding), five layers of
//!   bidirectional attention over the cache and the block run, and the
//!   vocabulary-sharded argmax of each later row is written into that row's
//!   input token, where the target's embedding reads it.
//!
//! All launches read live counts on the device and plan only from the
//! phase's token bucket, so they are captured with the decode and verify
//! graphs.

#[cfg(test)]
use model_mimo26_contract::geometry::V26_FLASH;

use std::collections::BTreeMap;

use kernel_core::contract::Refusal;
#[cfg(test)]
use kernel_core::contract::{Argument, ArgumentList, LaunchPlan};
use model_mimo26_contract::own_ops::dflash::{CacheWrite, CopyRows, Drafts, MaskRows};
use model_mimo26_contract::own_ops::retain::{GatherRows, History, RetainRows};
use runtime_bundle_abi::DeviceBufferSpanV1;

use crate::contract::PREFILL_TOKENS;

/// The context cache slots of the instance `bound` runs for: one per item
/// of its largest bucket.
fn slots(bound: &crate::program_lower::RankStage<'_>) -> u32 {
    bound.items.capacity()
}
#[cfg(test)]
use crate::dispatch::BufferBindings;
use crate::ops::HeadNorm;
#[cfg(test)]
use crate::pipeline::PreparedLaunch;
use crate::program::Stage;
use crate::program_lower::Lowered;
use crate::program_lower::RankStage;
use crate::selection::{MimoPhase as _, Phase};
use crate::topology::Tp;

#[cfg(test)]
pub const TAP_LAYERS: &[u32] = V26_FLASH.dflash().dflash_config.target_layer_ids;
pub const CACHE_TOKENS: u64 = 2048;
#[cfg(test)]
pub const V_SCALE: f32 = V26_FLASH.dflash().dflash_config.attention_value_scale as f32;
#[cfg(test)]
const HIDDEN: u64 = V26_FLASH.dflash().hidden_size as u64;

/// Per-rank K shard of the 5 x 4096 context feature (5120 at TP4).
pub fn fc_shard(tp: Tp) -> u64 {
    tp.dflash_fc_k()
}
/// Per-rank Q columns of one layer: 128 per head (16 heads at TP4).
fn q_columns(tp: Tp) -> u32 {
    (tp.dflash_q_heads() * tp.geometry.dflash().head_dim as u64) as u32
}
/// Per-rank K (or V) columns of one layer: 128 per head (2 heads at TP4).
fn kv_columns(tp: Tp) -> u32 {
    (tp.dflash_kv_heads() * tp.geometry.dflash().head_dim as u64) as u32
}
fn qkv_columns(tp: Tp) -> u32 {
    q_columns(tp) + 2 * kv_columns(tp)
}
/// Bytes of one cache plane of `slots` slots: [slot][position][head][128]
/// BF16. An instance holds one slot per item of its largest bucket.
fn plane_bytes(tp: Tp, slots: u32) -> u64 {
    u64::from(slots) * CACHE_TOKENS * u64::from(kv_columns(tp)) * 2
}
pub fn cache_bytes(tp: Tp, slots: u32) -> u64 {
    tp.geometry.dflash().num_hidden_layers as u64 * 2 * plane_bytes(tp, slots)
}
/// Threads of one block of the drafts entry, which is compiled for 32: one
/// block per 32 rows.
#[cfg(test)]
const DRAFT_THREADS: u32 = 32;
/// Like the target's hidden buffers, the taps keep slack rows so a
/// pipelined prefill half's span starting at its first row has full capacity.
pub fn tap_bytes(tp: Tp) -> u64 {
    (PREFILL_TOKENS as u64 + crate::scratch::HIDDEN_SLACK_ROWS) * fc_shard(tp) * 2
}
/// One (score, id) word pair per verify row: scores, then IDs.
pub const PAIR_BYTES: u64 = 2 * crate::contract::VERIFY_TOKENS as u64 * 4;
/// The all-gather's one-tile flags and epoch.
pub fn gather_flag_bytes(tp: Tp) -> u64 {
    2 * u64::from(tp.ranks()) * 4
}
pub const GATHER_EPOCH_BYTES: u64 = 4;

/// The bundle-owned drafter kernels of the topology's cubin
/// (`chop_mimo_dflash_copy_rows_v1` at TP4).
#[cfg(test)]
fn kernel(tp: Tp, name: &str) -> String {
    crate::topology::bundle_symbol(tp, &format!("dflash_{name}"))
}

/// One rank's drafter weights and device state.
#[derive(Clone)]
pub struct RankBuffers {
    /// Drafter views plus the target embedding and `lm_head`.
    pub views: BTreeMap<String, DeviceBufferSpanV1>,
    /// Per layer, the K plane then the V plane.
    pub cache: DeviceBufferSpanV1,
    /// [row][5120] BF16: this rank's columns of the context feature.
    pub taps: DeviceBufferSpanV1,
    pub pairs: DeviceBufferSpanV1,
    /// [rank][pairs] after the all-gather.
    pub gathered: DeviceBufferSpanV1,
    pub gather_flags: DeviceBufferSpanV1,
    pub gather_epoch: DeviceBufferSpanV1,
    /// A device u32 holding the vocabulary shard width.
    pub vocab_width: DeviceBufferSpanV1,
}

/// Every rank's all-gather outputs and flags, in rank order.
#[derive(Clone)]
pub struct Peers {
    pub gathered: Vec<DeviceBufferSpanV1>,
    pub gather_flags: Vec<DeviceBufferSpanV1>,
    pub gather_epochs: Vec<DeviceBufferSpanV1>,
}

fn slice(
    span: DeviceBufferSpanV1,
    offset: u64,
    length: u64,
) -> Result<DeviceBufferSpanV1, Refusal> {
    if offset
        .checked_add(length)
        .is_none_or(|end| end > span.length)
    {
        return Err(Refusal("DFlash device slice exceeds its buffer"));
    }
    Ok(DeviceBufferSpanV1::new(
        span.buffer,
        span.offset + offset,
        length,
    ))
}

/// The rest of a span from `offset` bytes.
fn tail(span: DeviceBufferSpanV1, offset: u64) -> Result<DeviceBufferSpanV1, Refusal> {
    slice(
        span,
        offset,
        span.length
            .checked_sub(offset)
            .ok_or(Refusal("DFlash span offset"))?,
    )
}

/// Layer `layer`'s K and V planes of the context cache.
pub fn cache_planes(
    cache: DeviceBufferSpanV1,
    layer: u32,
    tp: Tp,
    slots: u32,
) -> Result<(DeviceBufferSpanV1, DeviceBufferSpanV1), Refusal> {
    if layer >= tp.geometry.dflash().num_hidden_layers {
        return Err(Refusal("DFlash layer out of range"));
    }
    let plane = plane_bytes(tp, slots);
    let base = u64::from(layer) * 2 * plane;
    Ok((
        slice(cache, base, plane)?,
        slice(cache, base + plane, plane)?,
    ))
}

/// The columns of target tap `tap` that rank `rank`'s `fc` shard covers:
/// (first tap column, first shard column, width).
pub fn tap_columns(rank: u32, tap: usize, tp: Tp) -> Option<(u64, u64, u64)> {
    let width = fc_shard(tp);
    let shard = u64::from(rank) * width..u64::from(rank + 1) * width;
    let columns = tap as u64 * tp.geometry.dflash().target_hidden_size as u64
        ..(tap as u64 + 1) * tp.geometry.dflash().target_hidden_size as u64;
    let start = shard.start.max(columns.start);
    let end = shard.end.min(columns.end);
    (rank < tp.ranks()
        && tap < tp.geometry.dflash().dflash_config.target_layer_ids.len()
        && start < end)
        .then(|| (start - columns.start, start - shard.start, end - start))
}

/// A launch of `blocks` blocks over the step's `rows` token rows.
#[cfg(test)]
fn launch(
    symbol: &str,
    rows: u32,
    blocks: u32,
    threads: u32,
    arguments: Vec<Argument>,
    bindings: BufferBindings,
) -> PreparedLaunch {
    PreparedLaunch {
        symbol: symbol.into(),
        rows,
        plan: LaunchPlan {
            grid: [blocks, 1, 1],
            block: [threads, 1, 1],
            dynamic_shared_bytes: 0,
            workspace_bytes: 0,
            cluster: [1, 1, 1],
            pdl: true,
        },
        arguments: ArgumentList(arguments),
        bindings,
    }
}

/// Copies `width` columns of the live rows, optionally scaled; no input
/// writes zeros.
#[cfg(test)]
pub(crate) fn copy_rows(
    bound: &RankStage<'_>,
    input: Option<(DeviceBufferSpanV1, u32)>,
    output: DeviceBufferSpanV1,
    output_stride: u32,
    width: u32,
    scale: f32,
) -> Result<PreparedLaunch, Refusal> {
    let capacity = bound.phase.capacity()?;
    let rows = u64::from(capacity);
    let (input_stride, needed) = input.map_or((0, 0), |(_, stride)| {
        (stride, (rows - 1) * u64::from(stride) + u64::from(width))
    });
    if width == 0
        || !width.is_multiple_of(8)
        || !input_stride.is_multiple_of(8)
        || !output_stride.is_multiple_of(8)
        || input.is_some_and(|(span, _)| span.length < needed * 2 || span.offset % 16 != 0)
        || output.length < ((rows - 1) * u64::from(output_stride) + u64::from(width)) * 2
        || output.offset % 16 != 0
    {
        return Err(Refusal("DFlash row copy geometry"));
    }
    let mut bindings = BufferBindings::default();
    let input = input.map_or(Argument::NullDevicePointer, |(span, _)| {
        Argument::DeviceBuffer(bindings.bind(span))
    });
    let output = Argument::DeviceBuffer(bindings.bind(output));
    let live = Argument::DeviceBuffer(bindings.bind(bound.inputs.live_tokens));
    let grid =
        bound
            .phase
            .row_grid(bound.live_tokens, bound.switches, crate::sm120::ROW_BUCKETS)?;
    Ok(launch(
        &kernel(bound.tp, "copy_rows"),
        grid,
        grid,
        256,
        vec![
            input,
            output,
            live,
            Argument::U32(width),
            Argument::U32(input_stride),
            Argument::U32(output_stride),
            Argument::U32(scale.to_bits()),
        ],
        bindings,
    ))
}

/// Partial NeoX RoPE (64 of 128 dimensions) of compact Q or K heads, from
/// the sliding-window table.
fn rope(
    bound: &RankStage<'_>,
    q: bool,
    input: DeviceBufferSpanV1,
    output: DeviceBufferSpanV1,
) -> Result<Lowered, Refusal> {
    let table = bound
        .scratch
        .rope_tables
        .ok_or(Refusal("RoPE tables missing"))?[1];
    bound.ops.drafter_rope(
        q,
        bound.phase,
        bound.bucket,
        bound.live_tokens,
        input,
        output,
        bound.inputs.positions,
        table,
        bound.inputs.live_tokens,
    )
}

/// One declared drafter projection, through the authenticated store.
fn project(
    bound: &RankStage<'_>,
    id: &str,
    activation: DeviceBufferSpanV1,
    output: DeviceBufferSpanV1,
) -> Result<Lowered, Refusal> {
    let mut calls =
        bound.bound_projection(id, bound.phase, bound.live_tokens, activation, None, output)?;
    if calls.len() != 1 {
        return Err(Refusal("DFlash projection stage count differs"));
    }
    Ok(calls.remove(0))
}

/// All-reduces this rank's partial rows into the residual and normalizes
/// them with `norm` on every rank.
fn reduce(
    bound: &RankStage<'_>,
    input: DeviceBufferSpanV1,
    norm: DeviceBufferSpanV1,
) -> Result<Vec<Lowered>, Refusal> {
    bound.hidden_reduce(crate::ops::Route::Phase, input, norm, false)
}

fn view(buffers: &RankBuffers, name: &str) -> Result<DeviceBufferSpanV1, Refusal> {
    buffers
        .views
        .get(name)
        .copied()
        .ok_or(Refusal("DFlash view missing"))
}

/// Copies this rank's columns of tap `tap` for the rows the stage covers;
/// `row0` is the first row of a pipelined prefill half.
pub fn tap(
    bound: &RankStage<'_>,
    buffers: &RankBuffers,
    tap: usize,
    row0: u64,
) -> Result<Vec<Lowered>, Refusal> {
    let Some((source, target, width)) = tap_columns(bound.rank, tap, bound.tp) else {
        return Ok(Vec::new());
    };
    let shard = fc_shard(bound.tp);
    Ok(vec![bound.ops.dflash_copy_rows(
        &copy_tap(bound.tp, width as u32),
        bound.phase,
        bound.bucket,
        bound.switches,
        bound.live_tokens,
        Some(tail(bound.activation, source * 2)?),
        tail(buffers.taps, (row0 * shard + target) * 2)?,
        bound.inputs.live_tokens,
    )?])
}

/// Writes this rank's columns of tap `tap` for every row of a decode or
/// verify phase into the row's cell of its slot's retained `history`.
pub fn retain_tap(
    bound: &RankStage<'_>,
    tap: usize,
    history: DeviceBufferSpanV1,
) -> Result<Vec<Lowered>, Refusal> {
    let Some((source, target, width)) = tap_columns(bound.rank, tap, bound.tp) else {
        return Ok(Vec::new());
    };
    Ok(vec![bound.ops.retain_rows(
        &retain(bound.tp, bound.items.get(), width as u32),
        bound.phase,
        bound.bucket,
        tail(bound.activation, source * 2)?,
        tail(history, target * 2)?,
        &bound.inputs,
    )?])
}

/// The context-feature history of an instance serving `served` sequences:
/// one slot each, rows of the rank's fc shard.
fn history(tp: Tp, served: u32) -> History {
    History {
        stride: fc_shard(tp) as u32,
        capacity: super::state::history_capacity(tp) as u32,
        slots: served,
    }
}

/// The write of `width` columns of a tap's rows (the target's hidden rows)
/// into the context-feature history of `served` sequences.
pub fn retain(tp: Tp, served: u32, width: u32) -> RetainRows {
    RetainRows {
        width,
        input_stride: tp.geometry.dflash().target_hidden_size,
        history: history(tp, served),
    }
}

/// The copy of `width` columns of a tap's rows (the target's hidden rows)
/// into the rank's shard of the context feature, unscaled.
pub fn copy_tap(tp: Tp, width: u32) -> CopyRows {
    CopyRows {
        width,
        input_stride: Some(tp.geometry.dflash().target_hidden_size),
        output_stride: fc_shard(tp) as u32,
        scale_bits: 1.0_f32.to_bits(),
    }
}

/// The zeroed residual of `fc`'s all-reduce: whole drafter hidden rows.
pub fn zero_residual(tp: Tp) -> CopyRows {
    let hidden = tp.geometry.dflash().hidden_size;
    CopyRows {
        width: hidden,
        input_stride: None,
        output_stride: hidden,
        scale_bits: 1.0_f32.to_bits(),
    }
}

/// A draft block's V columns of its packed QKV rows into compact rows,
/// scaled by the drafter's value scale.
pub fn scale_v(tp: Tp) -> CopyRows {
    let kv = kv_columns(tp);
    CopyRows {
        width: kv,
        input_stride: Some(qkv_columns(tp)),
        output_stride: kv,
        scale_bits: (tp.geometry.dflash().dflash_config.attention_value_scale as f32).to_bits(),
    }
}

/// The context's K and V rows (K normalized and rotated into compact rows,
/// V read from the packed K and V rows of every layer) into a cache of
/// `slots` slots.
pub fn cache_write(tp: Tp, slots: u32) -> CacheWrite {
    let kv = kv_columns(tp);
    CacheWrite::of(
        tp,
        slots,
        kv,
        2 * kv * tp.geometry.dflash().num_hidden_layers,
    )
}

/// The mask embedding of a draft block's rows.
pub fn mask(tp: Tp) -> MaskRows {
    MaskRows {
        hidden: tp.geometry.dflash().hidden_size,
    }
}

/// The drafts of a block from every rank's gathered candidates.
pub fn drafts(tp: Tp) -> Drafts {
    Drafts { ranks: tp.ranks() }
}

/// The gather of every row's committed context feature from the history of
/// `served` sequences into the taps.
pub fn gather(tp: Tp, served: u32) -> GatherRows {
    let shard = fc_shard(tp) as u32;
    GatherRows {
        width: shard,
        output_stride: shard,
        history: history(tp, served),
    }
}

/// Projects every row's context feature in the taps into the five layers'
/// caches. With `history`, each row's committed feature is first gathered
/// from it into the taps. Rows may belong to several items.
pub fn context(
    bound: &RankStage<'_>,
    buffers: &RankBuffers,
    history: Option<DeviceBufferSpanV1>,
) -> Result<Vec<Lowered>, Refusal> {
    let scratch = bound.scratch;
    let kv = kv_columns(bound.tp);
    let mut result = history
        .map(|history| {
            bound.ops.gather_rows(
                &gather(bound.tp, bound.items.get()),
                bound.phase,
                bound.bucket,
                bound.live_tokens,
                history,
                buffers.taps,
                &bound.inputs,
            )
        })
        .transpose()?
        .into_iter()
        .collect::<Vec<_>>();
    result.extend([
        // fc has no residual: the all-reduce adds a zeroed one.
        bound.ops.dflash_copy_rows(
            &zero_residual(bound.tp),
            bound.phase,
            bound.bucket,
            bound.switches,
            bound.live_tokens,
            None,
            bound.activation,
            bound.inputs.live_tokens,
        )?,
        project(
            bound,
            "dflash.fc.bf16",
            buffers.taps,
            scratch.hidden_partial,
        )?,
    ]);
    result.extend(reduce(
        bound,
        scratch.hidden_partial,
        view(buffers, "hidden_norm.weight")?,
    )?);
    result.push(project(
        bound,
        "dflash.context-kv.bf16",
        bound.normalized,
        scratch.qkv,
    )?);
    for layer in 0..bound.tp.geometry.dflash().num_hidden_layers {
        let columns = u64::from(layer) * 2 * u64::from(kv);
        result.push(bound.ops.head_norm(
            HeadNorm::ContextK,
            bound.phase,
            bound.bucket,
            tail(scratch.qkv, columns * 2)?,
            view(buffers, &format!("layers.{layer}.self_attn.k_norm.weight"))?,
            scratch.q,
            bound.inputs.live_tokens,
        )?);
        result.push(rope(bound, false, scratch.q, scratch.k_new)?);
        let (k_cache, v_cache) = cache_planes(buffers.cache, layer, bound.tp, slots(bound))?;
        result.push(bound.ops.dflash_cache_write(
            &cache_write(bound.tp, slots(bound)),
            bound.phase,
            bound.bucket,
            bound.live_tokens,
            scratch.k_new,
            tail(scratch.qkv, (columns + u64::from(kv)) * 2)?,
            [k_cache, v_cache],
            &bound.inputs,
        )?);
    }
    Ok(result)
}

#[cfg(test)]
pub(crate) fn cache_write_builder(
    bound: &RankStage<'_>,
    buffers: &RankBuffers,
    layer: u32,
    k: (DeviceBufferSpanV1, u32),
    v: (DeviceBufferSpanV1, u32),
) -> Result<PreparedLaunch, Refusal> {
    let (k_cache, v_cache) = cache_planes(buffers.cache, layer, bound.tp, slots(bound))?;
    let mut bindings = BufferBindings::default();
    let mut bind = |span| Argument::DeviceBuffer(bindings.bind(span));
    let arguments = vec![
        bind(k.0),
        bind(v.0),
        bind(k_cache),
        bind(v_cache),
        bind(bound.inputs.slot_ids),
        bind(bound.inputs.token_batch),
        bind(bound.inputs.positions),
        bind(bound.inputs.last_positions),
        bind(bound.inputs.live_tokens),
        Argument::U32(k.1),
        Argument::U32(v.1),
        Argument::U32(
            (bound
                .tp
                .geometry
                .dflash()
                .dflash_config
                .attention_value_scale as f32)
                .to_bits(),
        ),
    ];
    let capacity = bound.phase.capacity()?;
    // A thread per 16 bytes of the K entry, then of the V entry.
    Ok(launch(
        &kernel(bound.tp, "cache_write"),
        capacity,
        capacity,
        kv_columns(bound.tp) / 4,
        arguments,
        bindings,
    ))
}

/// Block attention of one layer: queries of each item over its cached
/// context within the window and every row of its own block.
fn attention(
    bound: &RankStage<'_>,
    buffers: &RankBuffers,
    layer: u32,
) -> Result<Vec<Lowered>, Refusal> {
    let slots = slots(bound);
    let (k, v) = cache_planes(buffers.cache, layer, bound.tp, slots)?;
    // Each slot's circular cache follows the last, tightly packed.
    let stride = plane_bytes(bound.tp, 1);
    let cache = crate::ops::CacheViews {
        k,
        v,
        slots,
        strides: [stride, stride],
        scale_bits: [1.0_f32.to_bits(); 2],
    };
    let scratch = bound.scratch;
    let inputs = bound.inputs;
    let views = crate::ops::AttentionViews {
        q: scratch.q,
        k_new: scratch.k_new,
        v_new: scratch.v_new,
        slot_ids: Some(inputs.slot_ids),
        block_tables: None,
        seq_lens: inputs.seq_lens,
        q_start: inputs.q_start,
        q_len: inputs.q_len,
        prefix_len: inputs.prefix_len,
        live_batch: inputs.live_batch,
        sink: Some(view(
            buffers,
            &format!("layers.{layer}.self_attn.attention_sink_bias"),
        )?),
        cos_sin: None,
        output: scratch.attention,
        workspace: bound.workspace,
    };
    // The per-item arrays hold the served items; the op plans for every
    // drafting item of the largest bucket, and at least the catalog's 32,
    // and for the largest context, as the target verify attention is, so
    // the plan does not depend on the step.
    let items = bound.items.capacity();
    let context = bound.tp.geometry.context_tokens();
    let mut launches = Vec::new();
    for part in [
        crate::ops::AttentionPart::Partial,
        crate::ops::AttentionPart::Merge,
    ] {
        launches.extend(bound.ops.attention(
            crate::ops::AttentionLayer::Block,
            part,
            bound.phase,
            bound.bucket,
            bound.live_tokens,
            items,
            context,
            &views,
            &cache,
        )?);
    }
    Ok(launches)
}

/// Embeds the mask row in every draft row of the block (`mask_rows`).
#[cfg(test)]
pub(crate) fn mask_rows(
    bound: &RankStage<'_>,
    mask: DeviceBufferSpanV1,
) -> Result<PreparedLaunch, Refusal> {
    let capacity = bound.phase.capacity()?;
    let inputs = bound.inputs;
    let mut bindings = BufferBindings::default();
    let mut bind = |span| Argument::DeviceBuffer(bindings.bind(span));
    let arguments = vec![
        bind(bound.activation),
        bind(mask),
        bind(inputs.positions),
        bind(inputs.prefix_len),
        bind(inputs.token_batch),
        bind(inputs.live_tokens),
    ];
    Ok(launch(
        &kernel(bound.tp, "mask_rows"),
        capacity,
        capacity,
        256,
        arguments,
        bindings,
    ))
}

pub fn block(
    bound: &RankStage<'_>,
    buffers: &RankBuffers,
    peers: &Peers,
) -> Result<Vec<Lowered>, Refusal> {
    let Phase::Verify(_) = bound.phase else {
        return Err(Refusal("DFlash drafts only verify phases"));
    };
    let capacity = bound.phase.capacity()?;
    let scratch = bound.scratch;
    let inputs = bound.inputs;
    let mut result = vec![bound.ops.embedding(
        bound.phase,
        bound.switches,
        bound.live_tokens,
        &crate::ops::EmbeddingViews {
            table: view(buffers, "model.embed_tokens.weight")?,
            ids: inputs.token_ids,
            output: bound.activation,
            live_rows: inputs.live_tokens,
        },
    )?];
    result.push(bound.ops.dflash_mask_rows(
        &mask(bound.tp),
        bound.phase,
        bound.bucket,
        bound.activation,
        view(buffers, "mask_embedding.weight")?,
        &bound.inputs,
    )?);
    result.push(bound.ops.hidden_norm(
        bound.phase,
        bound.bucket,
        bound.switches,
        bound.live_tokens,
        bound.activation,
        view(buffers, "layers.0.input_layernorm.weight")?,
        bound.normalized,
        inputs.live_tokens,
    )?);
    let (q_cols, kv_cols) = (q_columns(bound.tp), kv_columns(bound.tp));
    for layer in 0..bound.tp.geometry.dflash().num_hidden_layers {
        let stem = format!("layers.{layer}");
        result.push(project(
            bound,
            &format!("dflash.qkv.{layer}.bf16"),
            bound.normalized,
            scratch.qkv,
        )?);
        // The attention output and the MLP buffer are free until attention
        // and gate/up, so they hold the normalized Q and K before RoPE.
        result.push(bound.ops.head_norm(
            HeadNorm::Q,
            bound.phase,
            bound.bucket,
            scratch.qkv,
            view(buffers, &format!("{stem}.self_attn.q_norm.weight"))?,
            scratch.attention,
            bound.inputs.live_tokens,
        )?);
        result.push(rope(bound, true, scratch.attention, scratch.q)?);
        result.push(bound.ops.head_norm(
            HeadNorm::K,
            bound.phase,
            bound.bucket,
            tail(scratch.qkv, u64::from(q_cols) * 2)?,
            view(buffers, &format!("{stem}.self_attn.k_norm.weight"))?,
            scratch.dense_gate,
            bound.inputs.live_tokens,
        )?);
        result.push(rope(bound, false, scratch.dense_gate, scratch.k_new)?);
        result.push(bound.ops.dflash_copy_rows(
            &scale_v(bound.tp),
            bound.phase,
            bound.bucket,
            bound.switches,
            bound.live_tokens,
            Some(tail(scratch.qkv, u64::from(q_cols + kv_cols) * 2)?),
            scratch.v_new,
            bound.inputs.live_tokens,
        )?);
        result.extend(attention(bound, buffers, layer)?);
        result.push(project(
            bound,
            &format!("dflash.output.{layer}.bf16"),
            scratch.attention,
            scratch.hidden_partial,
        )?);
        result.extend(reduce(
            bound,
            scratch.hidden_partial,
            view(buffers, &format!("{stem}.post_attention_layernorm.weight"))?,
        )?);
        result.push(project(
            bound,
            &format!("dflash.mlp.gate-up.{layer}.bf16"),
            bound.normalized,
            scratch.dense_gate,
        )?);
        result.push(project(
            bound,
            &format!("dflash.down.{layer}.bf16"),
            scratch.dense_gate,
            scratch.hidden_partial,
        )?);
        let next = if layer + 1 == bound.tp.geometry.dflash().num_hidden_layers {
            "norm.weight".to_owned()
        } else {
            format!("layers.{}.input_layernorm.weight", layer + 1)
        };
        result.extend(reduce(
            bound,
            scratch.hidden_partial,
            view(buffers, &next)?,
        )?);
    }
    // The target's BF16 head.
    result.push(project(
        bound,
        crate::projection::BF16_HEAD,
        bound.normalized,
        bound.local_logits,
    )?);
    result.extend(argmax(bound, buffers, peers, capacity)?);
    Ok(result)
}

/// Each rank's row maxima over its vocabulary shard, all-gathered, then the
/// best of every rank's written into the draft rows' input tokens.
fn argmax(
    bound: &RankStage<'_>,
    buffers: &RankBuffers,
    peers: &Peers,
    capacity: u32,
) -> Result<Vec<Lowered>, Refusal> {
    let words = 2 * u64::from(capacity);
    let ranks = bound.tp.ranks();
    let shard = crate::topology::vocab_per_rank(bound.tp);
    if buffers.pairs.length < words * 4
        || peers.gathered.len() != bound.tp.count()
        || peers.gather_flags.len() != bound.tp.count()
        || peers.gather_epochs.len() != bound.tp.count()
        || peers
            .gathered
            .iter()
            .any(|span| span.length < u64::from(ranks) * words * 4)
        || bound.local_logits.length < u64::from(capacity) * u64::from(shard) * 2
    {
        return Err(Refusal("DFlash argmax buffers"));
    }
    let scores = slice(buffers.pairs, 0, u64::from(capacity) * 4)?;
    let ids = slice(
        buffers.pairs,
        u64::from(capacity) * 4,
        u64::from(capacity) * 4,
    )?;
    let partial = bound.ops.dflash_argmax_partial(
        bound.rank,
        bound.phase,
        bound.bucket,
        &crate::ops::PartialViews {
            logits: bound.local_logits,
            live_width: buffers.vocab_width,
            scores,
            ids,
            live_rows: bound.inputs.live_tokens,
        },
    )?;
    // Every rank's pairs onto every rank.
    let gather = bound.ops.gather_candidates(
        bound.rank,
        bound.phase,
        bound.bucket,
        buffers.pairs,
        &peers.gathered,
        &peers.gather_flags,
        &peers.gather_epochs,
    )?;
    let mut result = vec![partial];
    result.extend(gather);
    result.push(bound.ops.dflash_drafts(
        &drafts(bound.tp),
        bound.phase,
        bound.bucket,
        buffers.gathered,
        &bound.inputs,
    )?);
    Ok(result)
}

/// Stores each draft row's best gathered (score, id) as its input token
/// (`drafts`).
#[cfg(test)]
pub(crate) fn drafts_builder(
    bound: &RankStage<'_>,
    gathered: DeviceBufferSpanV1,
    capacity: u32,
) -> PreparedLaunch {
    let mut bindings = BufferBindings::default();
    let mut bind = |span| Argument::DeviceBuffer(bindings.bind(span));
    let arguments = vec![
        bind(gathered),
        bind(bound.inputs.token_ids),
        bind(bound.inputs.positions),
        bind(bound.inputs.prefix_len),
        bind(bound.inputs.token_batch),
        bind(bound.inputs.live_tokens),
        Argument::U32(capacity),
    ];
    // The entry is compiled for 32 threads a block: one block per 32 rows.
    launch(
        &kernel(bound.tp, "drafts"),
        capacity,
        capacity.div_ceil(DRAFT_THREADS),
        DRAFT_THREADS,
        arguments,
        bindings,
    )
}

/// Adds the drafter's launches around one lowered target stage: the draft
/// block before a verify phase's embedding, a tap after each tapped layer's
/// reduction, and the context projection after the logits gather. Decode and
/// verify taps go straight into `history` when it is given.
pub fn extend(
    bound: &RankStage<'_>,
    buffers: &RankBuffers,
    peers: &Peers,
    stage: Stage,
    row0: u64,
    history: Option<DeviceBufferSpanV1>,
    launches: &mut Vec<Lowered>,
) -> Result<(), Refusal> {
    match stage {
        Stage::Embedding if matches!(bound.phase, Phase::Verify(_)) => {
            let mut drafted = block(bound, buffers, peers)?;
            drafted.append(launches);
            *launches = drafted;
        }
        Stage::MlpReduceNorm(_) | Stage::FinalReduceNorm => {
            let layer = match stage {
                Stage::MlpReduceNorm(layer) => layer,
                _ => bound.tp.geometry.layers() - 1,
            };
            if let Some(index) = bound
                .tp
                .geometry
                .dflash()
                .dflash_config
                .target_layer_ids
                .iter()
                .position(|&tapped| tapped == layer)
            {
                match history {
                    Some(history) if bound.phase.token_bucketed() => {
                        launches.extend(retain_tap(bound, index, history)?);
                    }
                    _ => launches.extend(tap(bound, buffers, index, row0)?),
                }
            }
        }
        Stage::GatherLogits => launches.extend(context(bound, buffers, history)?),
        _ => {}
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn each_rank_taps_its_contiguous_fc_shard() {
        let mut covered = vec![0_u32; 5 * HIDDEN as usize];
        let tp = Tp::TP4;
        for rank in 0..tp.ranks() {
            let mut next = 0;
            for tap in 0..TAP_LAYERS.len() {
                let Some((source, target, width)) = tap_columns(rank, tap, tp) else {
                    continue;
                };
                assert_eq!(target, next, "rank {rank} shard columns stay contiguous");
                next += width;
                for column in 0..width {
                    covered[(tap as u64 * HIDDEN + source + column) as usize] += 1;
                    assert_eq!(
                        tap as u64 * HIDDEN + source + column,
                        u64::from(rank) * fc_shard(tp) + target + column
                    );
                }
            }
            assert_eq!(next, fc_shard(tp));
        }
        assert!(covered.iter().all(|&count| count == 1));
        assert_eq!(tap_columns(0, 2, tp), None);
        assert_eq!(tap_columns(3, 4, tp), Some((0, 1024, 4096)));
        // TP2 rank 1 takes the second half of tap 2 and taps 3 and 4.
        assert_eq!(tap_columns(1, 2, Tp::TP2), Some((2048, 0, 2048)));
        assert_eq!(tap_columns(1, 4, Tp::TP2), Some((0, 6144, 4096)));
        assert_eq!(tap_columns(4, 0, tp), None);
    }

    #[test]
    fn cache_planes_split_five_layers_of_k_then_v() {
        let buffer = runtime_bundle_abi::BundleDeviceBufferHandleV1::from_parts(1, 1);
        let cache = DeviceBufferSpanV1::new(buffer, 4096, cache_bytes(Tp::TP4, 32));
        let (k, v) = cache_planes(cache, 4, Tp::TP4, 32).unwrap();
        assert_eq!(k.offset, 4096 + 8 * plane_bytes(Tp::TP4, 32));
        assert_eq!(v.offset, k.offset + plane_bytes(Tp::TP4, 32));
        assert_eq!(v.offset + v.length, cache.offset + cache.length);
        assert_eq!(plane_bytes(Tp::TP4, 32), 32 << 20);
        assert_eq!(plane_bytes(Tp::TP2, 32), 64 << 20);
        assert_eq!(plane_bytes(Tp::TP4, 256), 256 << 20);
        assert!(cache_planes(cache, 5, Tp::TP4, 32).is_err());
        // A 32-slot cache holds no 64-slot layer planes past the first.
        assert!(cache_planes(cache, 4, Tp::TP4, 64).is_err());
    }
}
