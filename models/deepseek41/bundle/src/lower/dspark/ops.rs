use kernel_core::backend::{Backend, Counts, Need, Prepared as Op, Refused, View};
use kernel_core::contract::{AttentionPhase, DType, Refusal};
use kernel_core::latent_attention::{
    LatentSourcesBuffers, LatentWindowBuffers, LatentWindowStorage,
};
use kernel_core::latent_glue::{LatentWindowWriteBuffers, WindowAddressing};
use kernel_core::mxfp8_gemm::{Mxfp8FoldGemmBuffers, Mxfp8FoldGemmQuery};
use kernel_core::precision::execution::BlockScaledExecution;
use kernel_core::precision::quantizer::{BlockQuantizeBuffers, BlockQuantizeQuery};
use kernel_core::rmsnorm::RmsNormBuffers;
use kernel_core::row_copy::{GatherByIndex, GatherByIndexBuffers};
use kernel_core::sinkhorn_hyper_connection::{
    SinkhornHyperConnectionQuery, StreamHeadBuffers, StreamHeadMode,
};
use model_deepseek41_contract::serving::{DECODE_BATCH, PREFILL_TOKENS};
use model_deepseek41_contract::{latent, latent_glue, moe, rows as row_contract};
use runtime_bundle_abi::DeviceBufferSpanV1 as Span;

use super::{BLOCK_ROWS, DraftMode, Lowering};
use crate::buffers::rows;
use crate::ops::{Bind, Lowered};
use crate::program::Phase;
use crate::shape::Shape;

#[derive(Clone, Copy, Debug)]
pub struct Window {
    pub codes: Span,
    pub scales: Span,
    pub sequences: u32,
    pub cells: u32,
    pub code_stride: u64,
    pub scale_stride: u64,
}

pub(super) fn slice(span: Span, offset: u64, bytes: u64) -> Result<Span, String> {
    if offset
        .checked_add(bytes)
        .is_none_or(|end| end > span.length)
    {
        return Err("DSpark buffer slice exceeds its span".into());
    }
    let offset = span
        .offset
        .checked_add(offset)
        .ok_or("DSpark byte offset overflows")?;
    Ok(Span::new(span.buffer, offset, bytes))
}

impl Window {
    fn bind(self, bind: &mut Bind, keys: Option<(Span, u32)>) -> LatentWindowBuffers<View> {
        let (key_lists, budget) = match keys {
            Some((keys, budget)) => (bind.view(keys), budget),
            None => (View::EMPTY, 0),
        };
        LatentWindowBuffers {
            codes: bind.view(self.codes),
            scales: bind.view(self.scales),
            item_slots: View::EMPTY,
            items: 0,
            sequences: self.sequences,
            code_stride: self.code_stride,
            scale_stride: self.scale_stride,
            storage: LatentWindowStorage::Staged { slots: self.cells },
            key_lists,
            key_stride: budget,
            budget,
        }
    }

    /// The five provisional cells after each item's committed window.
    fn block(self, latent: u32, window: u32) -> Result<Self, String> {
        if self.cells != window + BLOCK_ROWS {
            return Err("DSpark staged window must reserve five block cells".into());
        }
        let code_offset = u64::from(window) * u64::from(latent);
        let scale_offset = code_offset / 32;
        Ok(Self {
            codes: slice(
                self.codes,
                code_offset,
                self.codes
                    .length
                    .checked_sub(code_offset)
                    .ok_or("short DSpark window")?,
            )?,
            scales: slice(
                self.scales,
                scale_offset,
                self.scales
                    .length
                    .checked_sub(scale_offset)
                    .ok_or("short DSpark window scales")?,
            )?,
            cells: BLOCK_ROWS,
            ..self
        })
    }
}

pub struct MainViews {
    pub workspace: Span,
    pub raw_features: Span,
    pub feature_codes: Span,
    pub feature_scales: Span,
    pub weight: Span,
    pub weight_scales: Span,
    pub projected: Span,
    pub norm_weight: Span,
    pub normalized: Span,
    pub normalized_codes: Span,
    pub normalized_scales: Span,
    pub live_rows: Span,
    pub status: Span,
}

pub struct WindowViews {
    pub input: Span,
    pub norm_weight: Span,
    pub latent: Span,
    pub cos_sin: Span,
    pub positions: Span,
    pub row_slots: Span,
    pub first: Option<Span>,
    pub last: Span,
    pub window: Window,
    pub codes: Span,
    pub scales: Span,
    pub live_rows: Span,
    pub status: Span,
    pub workspace: Span,
}

pub struct Prepared {
    shape: Shape,
    feature: Op<StreamHeadBuffers<View>>,
    feature_pack: Op<GatherByIndexBuffers<View>>,
    main_quantize: Op<BlockQuantizeBuffers<View>>,
    main_project: Op<Mxfp8FoldGemmBuffers<View>>,
    main_project_small: Op<Mxfp8FoldGemmBuffers<View>>,
    main_norm: Op<RmsNormBuffers<View>>,
    hidden_quantize: Op<BlockQuantizeBuffers<View>>,
    ring_write: Op<LatentWindowWriteBuffers<View>>,
    block_write: Op<LatentWindowWriteBuffers<View>>,
    attention: Op<LatentSourcesBuffers<View>>,
}

impl Prepared {
    pub fn shape(&self) -> Shape {
        self.shape
    }
    pub fn prepare(backend: &dyn Backend, shape: Shape, mode: DraftMode) -> Result<Self, Refused> {
        mode.validate().map_err(Refused::Rule)?;
        let g = shape.geometry;
        let need = Need::rows(PREFILL_TOKENS);
        let feature = backend.sinkhorn_head(
            &SinkhornHyperConnectionQuery::Head {
                streams: g.shape.hc.streams,
                width: g.shape.hidden,
                mode: StreamHeadMode::MeanOfStreams,
                capacity: PREFILL_TOKENS,
            },
            need,
        )?;
        let feature_pack = backend.gather_by_index(
            &GatherByIndex {
                dtype: DType::Bf16,
                width: g.shape.hidden,
                alignment_bytes: 16,
            },
            need,
        )?;
        let k = g
            .shape
            .hidden
            .checked_mul(g.shape.dspark.targets.len() as u32)
            .ok_or_else(|| Refused::Rule("DSpark feature width overflows".into()))?;
        let main_quantize = backend.block_quantize(&moe::quantizer(k), need)?;
        let query = Mxfp8FoldGemmQuery {
            m_capacity: PREFILL_TOKENS,
            n: g.shape.hidden,
            k,
            groups: 1,
            activation_row_stride: k,
            activation_group_stride: k,
            output: DType::F32,
            execution: BlockScaledExecution::NativeBlockScaledMma,
        };
        let main_project = backend.mxfp8_fold_gemm(&query, need)?;
        let capacity = crate::sm120::mxfp8_fold::SMALL_CAPACITY;
        let main_project_small = backend.mxfp8_fold_gemm(
            &Mxfp8FoldGemmQuery {
                m_capacity: capacity,
                ..query
            },
            Need::rows(capacity)
                .with_exact(kernel_core::properties::Invariance::any_rows(capacity)),
        )?;
        let main_norm = backend.rmsnorm(&row_contract::final_norm(g), need)?;
        let hidden_quantize = backend.block_quantize(
            &BlockQuantizeQuery {
                input: DType::F32,
                ..moe::quantizer(g.shape.hidden)
            },
            need,
        )?;
        let ring_write = backend.latent_window_write(
            &latent_glue::window_write(
                g,
                WindowAddressing::Circular {
                    slots: g.shape.attention.window,
                },
            ),
            need,
        )?;
        let block_write = backend.latent_window_write(
            &latent_glue::window_write(g, WindowAddressing::Staged { slots: BLOCK_ROWS }),
            need,
        )?;
        let query = latent::query(
            g,
            shape.ranks(),
            AttentionPhase::Verify,
            latent::Keys::DraftBlock,
        )
        .map_err(Refused::Rule)?;
        let attention =
            backend.latent_attention_sources(&query, Need::rows(DECODE_BATCH * BLOCK_ROWS))?;
        Ok(Self {
            shape,
            feature,
            feature_pack,
            main_quantize,
            main_project,
            main_project_small,
            main_norm,
            hidden_quantize,
            ring_write,
            block_write,
            attention,
        })
    }

    /// Mean the pre-attention streams into this target layer's dense
    /// staging plane. Pack these planes only after all target taps ran.
    pub fn feature(
        &self,
        base: &Lowering<'_>,
        layer: u32,
        count: u32,
        staging: Span,
    ) -> Result<Vec<Lowered>, String> {
        let g = self.shape.geometry;
        if base.shape != &self.shape || count == 0 || count > PREFILL_TOKENS {
            return Err("DSpark feature geometry or row count differs".into());
        }
        let target = g
            .shape
            .dspark
            .targets
            .iter()
            .position(|&id| id == layer)
            .ok_or("layer is not a DSpark target")?;
        let plane = u64::from(PREFILL_TOKENS) * u64::from(g.shape.hidden) * 2;
        let mut bind = Bind::default();
        let buffers = StreamHeadBuffers {
            x: bind.view(rows(
                super::super::streams(&base.buffers.scratch, 2 * layer),
                count,
            )),
            output: bind.view(slice(staging, target as u64 * plane, plane)?),
            live_rows: bind.view(base.inputs.live_tokens),
        };
        bind.launch(&self.feature, &buffers, Counts::rows(count))
            .map_err(|e| e.to_string())
    }

    /// Row indices for `[target, capacity, hidden]` to `[row, target, hidden]`.
    pub fn feature_indices(&self, rows: u32) -> Result<Vec<u32>, String> {
        if rows == 0 || rows > PREFILL_TOKENS {
            return Err("feature rows exceed prefill capacity".into());
        }
        Ok((0..rows)
            .flat_map(|row| {
                (0..self.shape.geometry.shape.dspark.targets.len() as u32)
                    .map(move |target| target * PREFILL_TOKENS + row)
            })
            .collect())
    }

    /// Device live counts for bounded gathers over the concatenated feature
    /// vectors. Capacity and live rows refer to target rows, not vectors.
    pub fn feature_pack_counts(&self, rows: u32, live: u32) -> Result<Vec<u32>, String> {
        if rows == 0 || rows > PREFILL_TOKENS || live > rows {
            return Err("DSpark feature pack rows exceed capacity".into());
        }
        let targets = self.shape.geometry.shape.dspark.targets.len() as u32;
        Ok((0..rows * targets)
            .step_by(PREFILL_TOKENS as usize)
            .map(|start| (live * targets).saturating_sub(start).min(PREFILL_TOKENS))
            .collect())
    }

    pub fn pack_features(
        &self,
        rows: u32,
        staging: Span,
        indices: Span,
        counts: &[Span],
        output: Span,
    ) -> Result<Vec<Lowered>, Refusal> {
        if rows == 0 || rows > PREFILL_TOKENS {
            return Err(Refusal("DSpark feature pack rows exceed capacity"));
        }
        let vectors = rows * self.shape.geometry.shape.dspark.targets.len() as u32;
        if counts.len() != vectors.div_ceil(PREFILL_TOKENS) as usize {
            return Err(Refusal(
                "DSpark feature pack needs one live count per chunk",
            ));
        }
        let width = u64::from(self.shape.geometry.shape.hidden) * 2;
        let mut result = Vec::new();
        for (start, &count) in (0..vectors).step_by(PREFILL_TOKENS as usize).zip(counts) {
            let capacity = (vectors - start).min(PREFILL_TOKENS);
            let mut bind = Bind::default();
            let buffers = GatherByIndexBuffers {
                source: bind.view(staging),
                row_indices: bind.view(
                    slice(indices, u64::from(start) * 4, u64::from(capacity) * 4)
                        .map_err(|_| Refusal("short DSpark feature pack indices"))?,
                ),
                count: bind.view(count),
                destination: bind.view(
                    slice(
                        output,
                        u64::from(start) * width,
                        u64::from(capacity) * width,
                    )
                    .map_err(|_| Refusal("short DSpark packed features"))?,
                ),
            };
            result.extend(bind.launch(&self.feature_pack, &buffers, Counts::rows(capacity))?);
        }
        Ok(result)
    }

    /// The entries the drafter's ops launch, each with the dynamic shared
    /// memory its op states, for loading their modules.
    pub(crate) fn symbols(&self) -> impl Iterator<Item = &(String, u32)> {
        [
            self.feature.info(),
            self.feature_pack.info(),
            self.main_quantize.info(),
            self.main_project.info(),
            self.main_project_small.info(),
            self.main_norm.info(),
            self.hidden_quantize.info(),
            self.ring_write.info(),
            self.block_write.info(),
            self.attention.info(),
        ]
        .into_iter()
        .flat_map(|info| &info.symbols)
    }

    pub fn workspace(&self) -> u64 {
        [
            self.main_project_small.info().workspace,
            self.ring_write.info().workspace,
            self.block_write.info().workspace,
            self.attention.info().workspace,
            // The attention partials are a named role, beyond backend scratch.
            self.attention.info().bytes_of("workspace").unwrap_or(0),
        ]
        .into_iter()
        .max()
        .unwrap_or(0)
    }

    /// Raw committed features through main_proj and main_norm, then the
    /// trained input quantizer for this stage's KV projection. The F32 norm
    /// is never rounded before that quantizer's prescribed input rounding.
    pub fn main(&self, count: u32, v: &MainViews) -> Result<Vec<Lowered>, Refusal> {
        let quantize = |op: &Op<BlockQuantizeBuffers<View>>, input, output, scales| {
            let mut bind = Bind::default();
            let buffers = BlockQuantizeBuffers {
                input: bind.view(input),
                output: bind.view(output),
                scales: Some(bind.view(scales)),
                live_rows: bind.view(v.live_rows),
                status: bind.view(v.status),
                status_row_offset: 0,
            };
            bind.launch(op, &buffers, Counts::rows(count))
        };
        let mut result = quantize(
            &self.main_quantize,
            v.raw_features,
            v.feature_codes,
            v.feature_scales,
        )?;
        let mut bind = Bind::default();
        let buffers = Mxfp8FoldGemmBuffers {
            activations: bind.view(v.feature_codes),
            activation_scales: bind.view(v.feature_scales),
            weights: bind.view(v.weight),
            weight_scales: bind.view(v.weight_scales),
            output: bind.view(v.projected),
            live_rows: bind.view(v.live_rows),
            status: bind.view(v.status),
            status_row_offset: 0,
        };
        let projection = if count <= crate::sm120::mxfp8_fold::SMALL_CAPACITY {
            &self.main_project_small
        } else {
            &self.main_project
        };
        result.extend(bind.launch_within(
            projection,
            &buffers,
            v.workspace,
            Counts::rows(count),
        )?);
        let mut bind = Bind::default();
        let buffers = RmsNormBuffers {
            x: bind.view(v.projected),
            weight: bind.view(v.norm_weight),
            residual: None,
            y: bind.view(v.normalized),
            live_rows: bind.view(v.live_rows),
            status: Some(bind.view(v.status)),
            status_row_offset: 0,
        };
        result.extend(bind.launch(&self.main_norm, &buffers, Counts::rows(count))?);
        result.extend(quantize(
            &self.hidden_quantize,
            v.normalized,
            v.normalized_codes,
            v.normalized_scales,
        )?);
        Ok(result)
    }

    fn write(&self, count: u32, v: &WindowViews) -> Result<Vec<Lowered>, Refusal> {
        let mut bind = Bind::default();
        let window = v.window.bind(&mut bind, None);
        let buffers = LatentWindowWriteBuffers {
            input: bind.view(v.input),
            weight: bind.view(v.norm_weight),
            latent: bind.view(v.latent),
            cos_sin: bind.view(v.cos_sin),
            positions: bind.view(v.positions),
            row_batch: bind.view(v.row_slots),
            first_positions: v.first.map(|s| bind.view(s)),
            last_positions: bind.view(v.last),
            window,
            quantized_codes: bind.view(v.codes),
            quantized_scales: bind.view(v.scales),
            live_rows: bind.view(v.live_rows),
            status: bind.view(v.status),
            status_row_offset: 0,
        };
        let op = if v.first.is_some() {
            &self.block_write
        } else {
            &self.ring_write
        };
        bind.launch_within(op, &buffers, v.workspace, Counts::rows(count))
    }

    /// Catch-up writes only committed rows. Split each pass at ring and
    /// history wraps with `dspark::transfers` before calling this method.
    pub fn store_main(&self, count: u32, v: &WindowViews) -> Result<Vec<Lowered>, Refusal> {
        if v.first.is_some() || v.window.cells != self.shape.geometry.shape.attention.window {
            return Err(Refusal(
                "DSpark committed KV requires its exact circular window",
            ));
        }
        self.write(count, v)
    }

    pub(super) fn block_keys(
        &self,
        base: &Lowering<'_>,
        phase: Phase,
        layer: u32,
        window: Window,
        first: Span,
        last: Span,
    ) -> Result<Vec<Lowered>, String> {
        let g = self.shape.geometry;
        let s = &base.buffers.scratch;
        let window = window.block(g.shape.attention.head_dim, g.shape.attention.window)?;
        self.write(
            phase.rows(),
            &WindowViews {
                input: rows(s.kv, phase.rows()),
                norm_weight: base
                    .buffers
                    .view(&super::super::layer_view(layer, "attn.kv_norm.weight"))?,
                latent: rows(s.window_latent, phase.rows()),
                cos_sin: rows(s.rope_window, phase.rows()),
                positions: base.inputs.positions,
                row_slots: base.inputs.token_batch,
                first: Some(first),
                last,
                window,
                codes: rows(s.window_codes, phase.rows()),
                scales: rows(s.window_scales, phase.rows()),
                live_rows: base.inputs.live_tokens,
                status: rows(s.row_status, phase.rows()),
                workspace: base.buffers.workspace,
            },
        )
        .map_err(|e| e.to_string())
    }

    pub(super) fn attention(
        &self,
        base: &Lowering<'_>,
        phase: Phase,
        layer: u32,
        views: &super::BlockViews,
    ) -> Result<Vec<Lowered>, String> {
        let s = &base.buffers.scratch;
        let mut bind = Bind::default();
        let mut window = views.window.bind(
            &mut bind,
            Some((
                views.keys,
                self.shape.geometry.shape.attention.window + BLOCK_ROWS,
            )),
        );
        let Phase::Verify { items, .. } = phase else {
            return Err("DSpark attention requires a verify block".into());
        };
        if items != views.window.sequences {
            return Err("DSpark staged windows must match the drafting items".into());
        }
        window.item_slots = bind.view(views.item_slots);
        window.items = items;
        let buffers = LatentSourcesBuffers {
            q: bind.view(rows(s.native_q, phase.rows())),
            row_batch: bind.view(base.inputs.token_batch),
            window,
            compressed: None,
            sinks: Some(
                bind.view(
                    base.buffers
                        .view(&super::super::layer_view(layer, "attn.attn_sink"))?,
                ),
            ),
            cos_sin: Some(bind.view(rows(s.rope_window, phase.rows()))),
            output: Some(bind.view(rows(s.attention, phase.rows()))),
            partial: None,
            workspace: bind.view(base.buffers.workspace),
            live_rows: bind.view(base.inputs.live_tokens),
        };
        bind.launch_within(
            &self.attention,
            &buffers,
            base.buffers.workspace,
            Counts::rows(phase.rows()),
        )
        .map_err(|e| e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_budget_workspace_covers_draft_attention() {
        for shape in crate::shape::topologies() {
            let kernels = crate::Kernels::new(&shape).unwrap();
            let prepared =
                Prepared::prepare(kernels.backend.as_ref(), shape, DraftMode::Greedy).unwrap();
            let required = prepared.attention.info().bytes_of("workspace").unwrap();
            assert!(
                prepared.workspace() >= required,
                "{}: allocated {}, draft attention requires {required}",
                shape.topology(),
                prepared.workspace(),
            );
        }
    }
}
