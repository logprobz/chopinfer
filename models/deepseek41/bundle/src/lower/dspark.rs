//! Drafter-only lowering, captured after committed-feature catch-up and
//! replayed before Engram staging.

pub mod heads;
mod ops;
pub mod rings;
use model_deepseek41_contract::state::KvStorage;
pub use ops::{MainViews, Prepared, Window, WindowViews};

use kernel_core::latent_visibility::WindowBlockVisibility;
use model_deepseek41_contract::geometry::Geometry;
use runtime_bundle_abi::DeviceBufferSpanV1;

use super::Lowering;
use crate::buffers::RankBuffers;
use crate::dspark::execution::DraftInput;
use crate::dspark::state::BLOCK_ROWS;
use crate::ops::Lowered;
use crate::program::{Phase, Stage, Sublayer};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DraftMode {
    Greedy,
    Sampled,
}

impl DraftMode {
    pub fn validate(self) -> Result<(), String> {
        match self {
            Self::Greedy => Ok(()),
            Self::Sampled => Err("DSpark currently supports greedy draft selection only".into()),
        }
    }
}

/// Host-generated metadata for all five rows of every drafting item. K
/// changes only the lifecycle's returned proposals, never these arrays.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Metadata {
    pub tokens: Vec<u32>,
    /// Drafts always take the text router bias, including after image rows.
    pub bias_select: Vec<u8>,
    pub positions: Vec<i32>,
    pub row_items: Vec<i32>,
    pub slots: Vec<i32>,
    /// Identity mapping into the per-item staged windows, not request slots.
    pub staged_slots: Vec<i32>,
    pub first: Vec<i32>,
    pub last: Vec<i32>,
    pub keys: Vec<i32>,
    /// At position p, gather these rows of the block's logits and scatter
    /// the selected IDs into these rows of its item-major proposal array.
    pub position_rows: [Vec<u32>; BLOCK_ROWS as usize],
}

pub fn metadata(g: &Geometry, inputs: &[DraftInput], mode: DraftMode) -> Result<Metadata, String> {
    mode.validate()?;
    if inputs.is_empty() || inputs.len() > model_deepseek41_contract::serving::DECODE_BATCH as usize
    {
        return Err("DSpark draft batch is outside the served capacity".into());
    }
    let visibility = WindowBlockVisibility {
        window_size: g.shape.attention.window,
        block_size: BLOCK_ROWS,
    };
    let stride = (visibility.window_size + BLOCK_ROWS) as usize;
    let mut out = Metadata {
        tokens: Vec::new(),
        bias_select: Vec::new(),
        positions: Vec::new(),
        row_items: Vec::new(),
        slots: Vec::new(),
        staged_slots: Vec::new(),
        first: Vec::new(),
        last: Vec::new(),
        keys: Vec::new(),
        position_rows: std::array::from_fn(|_| Vec::new()),
    };
    let mut requests = std::collections::BTreeSet::new();
    let mut slots = std::collections::BTreeSet::new();
    for (item, input) in inputs.iter().enumerate() {
        let block = input.block;
        if block.compute_rows != BLOCK_ROWS
            || !(1..=BLOCK_ROWS).contains(&block.emit_rows)
            || block.verify_rows != block.emit_rows + 1
            || input.bonus >= g.shape.vocab
            || !requests.insert(block.request)
            || !slots.insert(block.slot)
        {
            return Err("invalid DSpark block input or duplicate slot".into());
        }
        let first = i32::try_from(block.main_rows).map_err(|_| "DSpark position exceeds I32")?;
        let last = first
            .checked_add(BLOCK_ROWS as i32 - 1)
            .ok_or("DSpark block position overflows")?;
        out.first.push(first);
        out.last.push(last);
        out.slots
            .push(i32::try_from(block.slot).map_err(|_| "DSpark slot exceeds I32")?);
        out.staged_slots.push(item as i32);
        for position in 0..BLOCK_ROWS {
            out.bias_select.push(0);
            out.tokens.push(if position == 0 {
                input.bonus
            } else {
                g.shape.dspark.noise_token
            });
            out.positions.push(first + position as i32);
            out.row_items.push(item as i32);
            out.position_rows[position as usize].push(item as u32 * BLOCK_ROWS + position);
            let mut keys = visibility
                .key_indices(block.main_rows, position)
                .map_err(|e| e.to_string())?
                .collect::<Vec<_>>();
            keys.resize(stride, -1);
            out.keys.extend(keys);
        }
    }
    Ok(out)
}

/// Bind the stage's checkpoint namespace for the existing backbone math.
/// Scratch and collective state must belong to this draft invocation, or
/// be reused only after the previous target invocation has completed.
pub fn stage_views(buffers: &RankBuffers, g: &Geometry, stage: u32) -> Result<RankBuffers, String> {
    if stage >= g.dspark_stages() {
        return Err("DSpark stage is outside geometry".into());
    }
    let mut result = buffers.clone();
    let prefix = format!("mtp.{stage}.");
    let target = format!("layers.{}.", g.layers() + stage);
    for (name, &span) in &buffers.views {
        if let Some(suffix) = name.strip_prefix(&prefix) {
            result.views.insert(format!("{target}{suffix}"), span);
        }
    }
    Ok(result)
}

/// The two ordinary sublayers, with ratio-zero attention, on all five
/// block rows. Main-ring writes and draft writes are kept separate.
pub fn block_stages(g: &Geometry, stage: u32, phase: Phase) -> Result<Vec<Stage>, String> {
    if stage >= g.dspark_stages() || g.compress_ratio(g.layers() + stage) != 0 {
        return Err("DSpark requires a ratio-zero stage".into());
    }
    if !matches!(phase, Phase::Verify { rows, items } if items > 0 && items <= model_deepseek41_contract::serving::DECODE_BATCH && items.checked_mul(BLOCK_ROWS) == Some(rows))
    {
        return Err("DSpark always lowers a full five-position block".into());
    }
    let layer = g.layers() + stage;
    let mut result = Vec::new();
    for (sublayer, body) in [
        (
            Sublayer::Attention,
            crate::program::attention(g, layer, phase),
        ),
        (Sublayer::Mlp, crate::program::moe(layer)),
    ] {
        result.extend([
            Stage::HcMix(layer, sublayer),
            Stage::InputNorm(layer, sublayer),
            Stage::QuantizeInput(layer, sublayer),
        ]);
        result.extend(body);
        result.extend([
            Stage::Reduce(layer, sublayer),
            Stage::HcExpand(layer, sublayer),
        ]);
    }
    Ok(result)
}

/// One stage over caller-bound drafter scratch. `base.buffers` and peers
/// have `stage_views` aliases; `window` has the committed ring copied into
/// its first window cells, with all five draft cells reserved after them.
pub struct BlockViews {
    pub window: Window,
    /// Uploaded `Metadata::staged_slots`, one identity index per draft item.
    pub item_slots: DeviceBufferSpanV1,
    pub first: DeviceBufferSpanV1,
    pub last: DeviceBufferSpanV1,
    pub keys: DeviceBufferSpanV1,
}

/// Run once before the first draft stage, after uploading `Metadata`.
/// Causal target window-key generation is deliberately absent.
pub fn initialize(base: &Lowering<'_>, phase: Phase) -> Result<Vec<Lowered>, String> {
    block_stages(base.shape.geometry, 0, phase)?;
    let mut result = Vec::new();
    for stage in [
        Stage::Embedding,
        Stage::EmbeddingReduce,
        Stage::ExpandStreams,
        Stage::RopeRows,
    ] {
        result.extend(
            base.stage(phase, stage)?
                .ok_or_else(|| format!("DSpark initialization does not lower: {stage:?}"))?,
        );
    }
    Ok(result)
}

pub fn block(
    prepared: &Prepared,
    base: &Lowering<'_>,
    stage: u32,
    phase: Phase,
    views: BlockViews,
) -> Result<Vec<Lowered>, String> {
    if base.shape != &prepared.shape() || base.kv.storage != KvStorage::Native {
        return Err("DSpark block geometry or native KV binding differs".into());
    }
    let g = base.shape.geometry;
    if base.ops.expert_set.count != g.shape.dspark.experts
        || base.ops.expert_set.top_k != g.shape.dspark.top_k
    {
        return Err("DSpark expert set differs from the prepared operations".into());
    }
    let mut result = Vec::new();
    for operation in block_stages(base.shape.geometry, stage, phase)? {
        let launches = match operation {
            Stage::WindowKv(layer) => {
                prepared.block_keys(base, phase, layer, views.window, views.first, views.last)?
            }
            Stage::LatentAttention(layer) => prepared.attention(base, phase, layer, &views)?,
            Stage::HcMix(_, Sublayer::Attention) if stage == 0 => {
                // The first draft sublayer starts with the one-hot pre,
                // independent of the backbone's final mixer.
                let mut buffers = base.buffers.clone();
                buffers.scratch.pre_odd = buffers.scratch.first_pre;
                let first = Lowering {
                    buffers: &buffers,
                    ..*base
                };
                first
                    .stage(phase, operation)?
                    .ok_or("DSpark first mixer is unavailable")?
            }
            _ => base
                .stage(phase, operation)?
                .ok_or_else(|| format!("DSpark stage does not lower: {operation:?}"))?,
        };
        result.extend(launches);
    }
    Ok(result)
}

pub struct CatchUpViews {
    /// Obtain this range from `Features::projection_source` for the
    /// validated chunk. Provisional rows enter only during a long-prefill
    /// commit that explicitly projects from staging.
    pub raw_features: DeviceBufferSpanV1,
    pub feature_codes: DeviceBufferSpanV1,
    pub feature_scales: DeviceBufferSpanV1,
    pub ring: Window,
    pub row_slots: DeviceBufferSpanV1,
    pub last: DeviceBufferSpanV1,
}

/// Project a transfer chunk into one stage's committed ring. The caller
/// binds the chunk's positions and live count, and checks row status before
/// acknowledging it to the lifecycle's projection cursor.
pub fn catch_up(
    prepared: &Prepared,
    base: &Lowering<'_>,
    stage: u32,
    count: u32,
    v: &CatchUpViews,
) -> Result<Vec<Lowered>, String> {
    let g = base.shape.geometry;
    if base.shape != &prepared.shape()
        || stage >= g.dspark_stages()
        || base.kv.storage != KvStorage::Native
    {
        return Err("DSpark catch-up geometry or native KV binding differs".into());
    }
    let phase = Phase::Prefill { rows: count };
    let s = &base.buffers.scratch;
    let row = |span| crate::buffers::rows(span, count);
    let mut result = base
        .stage(phase, Stage::RopeRows)?
        .ok_or("DSpark catch-up RoPE is unavailable")?;
    result.extend(
        prepared
            .main(
                count,
                &MainViews {
                    stream: crate::ops::fold_stream::storage(
                        &base.buffers.views,
                        "mtp.0.main_proj.weight",
                        false,
                    ),
                    workspace: base.buffers.workspace,
                    raw_features: v.raw_features,
                    feature_codes: v.feature_codes,
                    feature_scales: v.feature_scales,
                    weight: base.buffers.view("mtp.0.main_proj.weight")?,
                    weight_scales: base.buffers.view("mtp.0.main_proj.scale")?,
                    projected: row(s.partial),
                    norm_weight: base.buffers.view("mtp.0.main_norm.weight")?,
                    normalized: row(s.normed_wide),
                    normalized_codes: row(s.normed_codes),
                    normalized_scales: row(s.normed_scales),
                    live_rows: base.inputs.live_tokens,
                    status: row(s.row_status),
                },
            )
            .map_err(|e| e.to_string())?,
    );
    let layer = g.layers() + stage;
    result.extend(
        base.stage(phase, Stage::Project(layer, "attention.kv"))?
            .ok_or("DSpark catch-up KV projection is unavailable")?,
    );
    result.extend(
        prepared
            .store_main(
                count,
                &WindowViews {
                    input: row(s.kv),
                    norm_weight: base
                        .buffers
                        .view(&super::layer_view(layer, "attn.kv_norm.weight"))?,
                    latent: row(s.window_latent),
                    cos_sin: row(s.rope_window),
                    positions: base.inputs.positions,
                    row_slots: v.row_slots,
                    first: None,
                    last: v.last,
                    window: v.ring,
                    codes: row(s.window_codes),
                    scales: row(s.window_scales),
                    live_rows: base.inputs.live_tokens,
                    status: row(s.row_status),
                    workspace: base.buffers.workspace,
                },
            )
            .map_err(|e| e.to_string())?,
    );
    Ok(result)
}

#[cfg(test)]
mod tests;
