//! Lowering of the engines' stages into prepared launches: the latent KV
//! write, the absorbed head GEMMs and the latent attention with its split
//! merge (`lower/latent.rs`), the KDA convolution, delta rule and gated norm
//! (`lower/kda.rs`), the pooled indexer, the Sinkhorn mixer and the chained
//! MoE (`lower/indexer.rs`, `lower/sinkhorn.rs` and `lower/moe.rs`), over the
//! instance's prepared ops (`crate::ops`). Each launch binds the op's roles
//! to the rank's buffers, views, state and KV pool. Other stages stay
//! launch records (`crate::launch`).

mod context_parallel;
mod draft;
pub(crate) mod formats;
mod indexer;
mod kda;
mod latent;
mod logits;
mod moe;
mod project;
mod reduce;
mod rows;
mod sinkhorn;

pub(crate) use project::eh_uses;
#[cfg(feature = "diagnostic-taps")]
pub(crate) use sinkhorn::streams;

use bundle_runtime::kv::KvFormat;
use bundle_runtime::pipeline::PreparedLaunch;
use bundle_runtime::upload::DeviceInputs;
use kernel_core::latent_attention::HeadWeightLayout;
use model_glm5next_contract::contract::Shape;
use runtime_bundle_abi::DeviceBufferSpanV1;

use crate::buffers::{Buffer, RankBuffers, SCRATCH_ROWS};
use crate::ops::Lowered;
use crate::program::{Phase, Stage};

const BF16: u64 = 2;

/// What a launch binds to one entry parameter.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Value {
    Span(DeviceBufferSpanV1),
}

/// The widest prefill-path drafter pass that runs the 64 fixed splits
/// (`Lowering::drafter_splits`): kernel-lab timed them against one split
/// per row over FP8 pages at 3K and 32K contexts (PERF-B14b lab run
/// 20261009T075837Z-665316) at 0.11 of its time at 1 row, 0.58 at 36 rows,
/// 0.89 to 0.95 at 64 rows and 1.77 to 1.79 at 192 rows.
pub(crate) const DRAFT_SPLIT_ROWS: u32 = 64;

/// One rank's lowering of the merged engines.
pub struct Lowering<'a> {
    pub shape: Shape,
    pub format: &'a KvFormat,
    pub buffers: &'a RankBuffers,
    /// The phase's spans in the shared input arena.
    pub inputs: &'a DeviceInputs,
    /// The ops the instance prepared once on its backend (`crate::ops`).
    pub ops: &'a crate::ops::Ops,
    /// The rank within the topology, a scalar of the local routes.
    pub rank: u32,
    /// The resolved `text_config.index_topk`: the key list budget is these
    /// tokens plus the incomplete pool tail.
    pub index_topk: u32,
    /// An eager prefill's indexer pieces (`crate::step::pieces`); empty
    /// runs the indexer over the whole phase.
    pub pieces: &'a [Piece],
    /// The first row the indexer's row buffers bind from: a piece's.
    pub row_offset: u32,
    /// Every rank's comm state, in rank order.
    pub peers: &'a [crate::buffers::CommState],
    /// Rows the head scores: the phase's rows in decode and verify, an
    /// eager prefill's logits rows.
    pub head_rows: u32,
    /// The context-parallel groups above size 1 (`lower/context_parallel.rs`).
    pub context_parallel: Option<&'a crate::context_parallel::ContextParallel>,
    /// A drafter run's pass; `None` for the target's phases.
    pub draft: Option<crate::program::DraftPass>,
    /// The target's verify input tokens, which a drafter run's argmax fills
    /// with its drafts.
    pub verify_tokens: Option<DeviceBufferSpanV1>,
    /// The rank's projection store and scratch when the package carries the
    /// precision component; without it the projection stages stay records.
    pub projections: Option<crate::projections::Rank<'a>>,
}

/// One indexer piece of an eager prefill: its first row in the step, its
/// rows and its own inputs in the shared input arena.
#[derive(Clone, Copy, Debug)]
pub struct Piece {
    pub first: u32,
    pub rows: u32,
    pub inputs: DeviceInputs,
}

fn view_name(layer: u32, name: &str) -> String {
    format!("model.language_model.layers.{layer}.self_attn.{name}")
}

/// The name of layer `layer`'s view `name` (below `layers.{layer}.`).
pub(crate) fn layer_view(layer: u32, name: &str) -> String {
    format!("model.language_model.layers.{layer}.{name}")
}

impl Lowering<'_> {
    /// [`Self::lower`], with each sublayer boundary through the Sinkhorn
    /// mHC comm epilogue when `comm_epilogue` (`lower/reduce.rs`, the
    /// `runtime.sinkhorn_comm_epilogue` model option).
    pub fn lower_with(
        &self,
        phase: Phase,
        stage: Stage,
        comm_epilogue: bool,
    ) -> Result<Option<Vec<PreparedLaunch>>, String> {
        Ok(self
            .lower_facts(phase, stage, comm_epilogue)?
            .map(crate::ops::launches))
    }

    /// [`Self::lower_with`]'s launches, each with whether it waits for its
    /// peer ranks as the op that made it states.
    pub fn lower_facts(
        &self,
        phase: Phase,
        stage: Stage,
        comm_epilogue: bool,
    ) -> Result<Option<Vec<Lowered>>, String> {
        if comm_epilogue && let Some(lowered) = self.comm_epilogue_stage(phase, stage)? {
            return Ok(Some(lowered));
        }
        self.stage(phase, stage)
    }

    /// The launches of `stage` in `phase`, or `None` for a projection stage
    /// of an instance without the precision component.
    pub fn lower(&self, phase: Phase, stage: Stage) -> Result<Option<Vec<PreparedLaunch>>, String> {
        Ok(self.stage(phase, stage)?.map(crate::ops::launches))
    }

    /// [`Self::lower`]'s launches with the facts their ops state.
    fn stage(&self, phase: Phase, stage: Stage) -> Result<Option<Vec<Lowered>>, String> {
        if let Some(launches) = self.context_parallel_stage(phase, stage)? {
            return Ok(Some(launches));
        }
        Ok(Some(match stage {
            Stage::LatentKvWrite(layer) => vec![self.kv_write(phase, layer)?],
            Stage::QueryAbsorb(layer) => {
                vec![self.head_gemm(phase, layer, HeadWeightLayout::KByN)?]
            }
            Stage::ValueUp(layer) => vec![self.head_gemm(phase, layer, HeadWeightLayout::NByK)?],
            Stage::LatentAttention(layer) => self.attention(phase, layer)?,
            Stage::KdaConv(layer) => vec![self.kda_conv(phase, layer)?],
            Stage::KdaDeltaRule(layer) => self.kda_delta(phase, layer)?,
            Stage::KdaGatedNorm(layer) => vec![self.kda_norm(phase, layer)?],
            Stage::IndexKeyWrite(layer) => self.in_pieces(phase, |lowering, phase| {
                Ok(vec![lowering.index_key_write(phase, layer)?])
            })?,
            Stage::IndexSelect(layer) => {
                self.in_pieces(phase, |lowering, phase| lowering.index_select(phase, layer))?
            }
            // The decode and verify graphs fuse each main layer's input norm
            // into its mixer.
            Stage::HcMix(layer, sublayer) if self.fuses_input_norm(phase, layer) => {
                vec![self.hc_mix_norm(phase, layer, sublayer)?]
            }
            Stage::InputNorm(layer, _) if self.fuses_input_norm(phase, layer) => Vec::new(),
            Stage::HcMix(layer, sublayer) => vec![self.hc_mix(phase, layer, sublayer)?],
            Stage::HcExpand(layer, sublayer) => vec![self.hc_expand(phase, layer, sublayer)?],
            Stage::HeadCollapse => vec![self.hc_head(phase)?],
            Stage::Embedding => vec![self.embedding(phase)?],
            Stage::InputNorm(..)
            | Stage::FinalNorm
            | Stage::QueryNorm(_)
            | Stage::LatentNorm(_)
            | Stage::DraftEmbedNorm
            | Stage::DraftHiddenNorm
            | Stage::DraftHeadNorm => vec![self.norm(phase, stage)?],
            Stage::DraftResidual(_) => vec![self.residual_add(phase)?],
            Stage::DraftGather => vec![self.draft_gather(phase)?],
            Stage::DraftEhInput => self.eh_input(phase)?,
            Stage::DraftShareSelection => vec![self.share_selection()?],
            Stage::DraftHeadRows => vec![self.draft_head_rows(phase)?],
            Stage::DraftArgmax => self.draft_argmax(phase)?,
            Stage::EmbeddingReduce => self.embedding_reduce(phase)?,
            Stage::Reduce(layer, sublayer) => self.sublayer_reduce(phase, layer, sublayer)?,
            Stage::ExpandStreams => vec![self.expand_streams(phase)?],
            Stage::GatherHeadRows => vec![self.gather_head_rows(phase)?],
            Stage::GatherLogits => self.gather_logits(phase)?,
            // The decode and verify graphs run a KDA layer's projections of
            // its normed input as one stacked launch, at the first of them.
            Stage::Project(layer, id) if self.stacked_ops(phase, id).is_some() => {
                if self.ops.own.stacked_projections().first() == Some(&id) {
                    vec![self.stacked_kda_inputs(phase, layer)?]
                } else {
                    Vec::new()
                }
            }
            // Through the projection API when the instance holds its store;
            // the projections it does not lower yet stay records.
            Stage::Project(layer, id) => match self.project(phase, layer, id)? {
                Some(launches) => launches,
                None => return Ok(None),
            },
            Stage::LmHead => match self.head(phase)? {
                Some(launches) => launches,
                None => return Ok(None),
            },
            Stage::Route(layer) => self.route(phase, layer, false)?,
            Stage::LocalRoutes(layer) => self.route(phase, layer, true)?,
            Stage::ExpertsGateUp(layer) => self.experts(phase, layer, true)?,
            Stage::ExpertsDown(layer) => self.experts(phase, layer, false)?,
            Stage::RoutedSum(_) => self.routed_sum(phase)?,
            Stage::DenseGateUp(layer) => self.dense_gate_up(phase, layer, false)?,
            Stage::SharedGateUp(layer) => self.dense_gate_up(phase, layer, true)?,
        }))
    }

    /// The launches of an indexer stage: over each piece of an eager
    /// prefill in order, binding the piece's inputs and its rows of the
    /// indexer's row buffers, else over the whole phase.
    fn in_pieces<T>(
        &self,
        phase: Phase,
        stage: impl Fn(&Lowering<'_>, Phase) -> Result<Vec<T>, String>,
    ) -> Result<Vec<T>, String> {
        if !matches!(phase, Phase::Prefill { .. }) || self.pieces.is_empty() {
            if matches!(phase, Phase::Prefill { rows } if rows > crate::step::PIECE_ROWS) {
                return Err(format!(
                    "a prefill of {} rows runs its indexer in pieces",
                    phase.rows()
                ));
            }
            return stage(self, phase);
        }
        let mut launches = Vec::new();
        for piece in self.pieces {
            let lowering = Lowering {
                inputs: &piece.inputs,
                pieces: &[],
                row_offset: piece.first,
                ..*self
            };
            launches.extend(stage(&lowering, Phase::Prefill { rows: piece.rows })?);
        }
        Ok(launches)
    }

    /// The scratch span of `buffer`: the indexer's row buffers from the
    /// piece's first row, every other buffer whole.
    pub(crate) fn scratch(&self, buffer: Buffer) -> DeviceBufferSpanV1 {
        let span = self.buffers.get(buffer);
        let offset = match buffer {
            Buffer::IndexQuery
            | Buffer::IndexKey
            | Buffer::IndexWeights
            | Buffer::IndexGate
            | Buffer::KeyLists => {
                u64::from(self.row_offset) * (span.length / u64::from(SCRATCH_ROWS))
            }
            _ => 0,
        };
        if offset == 0 {
            return span;
        }
        DeviceBufferSpanV1::new(span.buffer, span.offset + offset, span.length - offset)
    }

    /// The step-input and scratch bindings every engine names alike. The
    /// shared arena carries each row's item (`token_batch`) and position,
    /// the live row and item counts, each item's slot and committed length
    /// (its first row's position) and the block tables.
    pub(crate) fn common(&self, name: &str) -> Option<Value> {
        let inputs = self.inputs;
        let span = match name {
            "row_batch" => inputs.token_batch,
            "positions" => inputs.positions,
            "live_rows" => inputs.live_tokens,
            "live_sequences" => inputs.live_batch,
            "block_tables" => inputs.block_tables,
            "slot_ids" => inputs.slot_ids,
            "start_positions" => inputs.prefix_len,
            _ => self.scratch(match name {
                "cu_seqlens" => Buffer::CuSeqlens,
                "initial" => Buffer::Initial,
                "anchor_ids" => Buffer::AnchorIds,
                "cu_rows" => Buffer::CuRows,
                "checkpoint_ids" => Buffer::CheckpointIds,
                "workspace" => Buffer::Workspace,
                // A draft step reads the extend pass's compacted lists.
                "key_lists" if self.draft == Some(crate::program::DraftPass::Step) => {
                    Buffer::SharedKeyLists
                }
                "key_lists" => Buffer::KeyLists,
                _ => return None,
            }),
        };
        Some(Value::Span(span))
    }
}
