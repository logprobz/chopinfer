//! Bind one rank-program stage to the rank image, the phase inputs and the
//! rank's fixed buffers.

use std::collections::BTreeMap;

use kernel_core::attention::{FoldInputs, KvCache};
use kernel_core::backend::{Prepared, View};
use kernel_core::collective::{
    GatherBuffers, HcAllReduceBuffers, RankState, RankViews, RowGatherBuffers,
};
use kernel_core::contract::{DType, Refusal};
use kernel_core::gated_mlp::GatedMlpProjection;
use kernel_core::gated_sparse::{
    BlockKeys, BlockScoresBuffers, BlockTopKBuffers, GatedKvWriteBuffers, IndexQueryBuffers,
    KeyRings, OutputGateBuffers, SparseAttentionBuffers, SparseKeyWriteBuffers,
};
use kernel_core::moe::{RouteBuffers, RouteMetadata, SharedCombineBuffers};
use kernel_core::moe_experts::{GatedMlpBuffers, GatedMlpScratch};
use kernel_core::precision::buffers::{ProjectionRouting, QuantizedWeights};
use kernel_core::rmsnorm::FusedResidual;
use kernel_core::row_copy::RowCopyBuffers;
use model_qwen38_contract::own_ops::dense::{DenseGroup, DenseGroupBuffers, Projection};
use model_qwen38_contract::own_ops::hc::HcUpMixBuffers;
use model_qwen38_contract::own_ops::mtp::{
    DraftArgmaxFinalBuffers, DraftArgmaxPartialBuffers, MtpCombineBuffers, MtpGatherBuffers,
    MtpNormsBuffers,
};
use model_qwen38_contract::own_ops::ple::{
    PleAnchorBuffers, PleCheckpointsBuffers, PleConvBuffers, PleResidualBuffers, PleStageBuffers,
};
use runtime_bundle_abi::DeviceBufferSpanV1;

use crate::contract::*;
use crate::ops::{
    AttentionOps, Bind, CollectiveOps, DenseOp, Key, Lowered, MoeOps, OwnStageOps, StageOps,
    launches,
};
use crate::pipeline::PreparedLaunch;
use crate::program::{Phase, Stage, Sublayer};
use crate::scratch::{RankScratch, logits_gather};
use crate::side::{PIECES, SideInputs};
use crate::upload::DeviceInputs;
use crate::views::MODEL;

mod context_parallel;
pub(crate) mod projection;

/// One item's rows in a phase: its first token and token count.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ItemRows {
    pub start: u32,
    pub rows: u32,
}

pub struct RankStage<'a> {
    pub precision: Option<&'a crate::precision::Prepared>,
    /// Whether each norm's weights are F32, resolved at create.
    pub(crate) norm_dtypes: &'a NormDtypes,
    /// The topology's per-rank sizes and kernel tag.
    pub shape: Shape,
    pub rank: u32,
    pub phase: Phase,
    /// Prepared token rows: the live tokens of a prefill, the bucket of a
    /// decode.
    pub rows: u32,
    /// Items a prefill runs, or one row per bucket item for decode.
    pub items: &'a [ItemRows],
    /// Rows the head projects: a prefill's rows that need logits, or the
    /// decode bucket.
    pub head_rows: u32,
    /// A prefill that scores prompt rows: F32 logits of every head row.
    pub scored: bool,
    /// A prefill with an item that keeps its state at an anchor
    /// (`ItemEntries::anchor`); without one the PLE anchor stage is empty.
    pub anchored: bool,
    /// The longest visible length: a prefill's, or the whole context for a
    /// decode graph, which replays at any length.
    pub context: u32,
    pub views: &'a BTreeMap<String, DeviceBufferSpanV1>,
    pub inputs: DeviceInputs,
    pub side: SideInputs,
    /// Every rank's buffers; this rank's are `peers[rank]`.
    pub peers: &'a [RankScratch],
    pub kv_pool: DeviceBufferSpanV1,
    /// The pool's layout: its page bytes, and the element and scales of the
    /// main full layers (the MTP layer's K and V are always BF16).
    pub kv: bundle_runtime::kv::KvFormat,
    /// The live half of the QSA raw-key ring, and its layout.
    pub ring: DeviceBufferSpanV1,
    pub ring_layout: Ring,
    /// Entries of each linear-state pool plane.
    pub pool_entries: u64,
    /// Decode graphs above 32 rows may take the wide all-reduce
    /// (`launch::WIDE_COMM`).
    pub wide_comm: bool,
    pub hc_precompute: bool,
    pub hc_padded_mix: bool,
    /// Set when this is a drafter program.
    pub drafter: Option<DrafterInputs>,
    /// Target logits exchange state, absent for the drafter and gather mode.
    pub exchange: Option<&'a [crate::logits::ExchangeRowsState]>,
    /// The encoder buffers of a prefill that injects media rows.
    pub media: Option<&'a crate::encoder::Rank>,
    /// The instance's context-parallel groups, above size 1 only.
    pub context_parallel: Option<&'a crate::context_parallel::ContextParallel>,
    /// The ops the instance prepared, absent before create prepares them.
    pub(crate) ops: Option<&'a crate::ops::Ops>,
    /// The GPU target the instance runs on.
    pub(crate) target: &'a dyn crate::target::Target,
}

/// The inputs of a drafter run beside its phase arena (`crate::drafter`).
#[derive(Clone, Copy, Debug)]
pub struct DrafterInputs {
    /// Per row: the target X row it reads (see `mtp_gather`) and where its
    /// token comes from.
    pub sources: DeviceBufferSpanV1,
    pub token_sources: DeviceBufferSpanV1,
    /// Per draft row: the drafting item's slot in `drafts`, and the verify
    /// row the draft fills (-1 for none).
    pub slots: DeviceBufferSpanV1,
    pub targets: DeviceBufferSpanV1,
    /// This rank's verify token ids, which the drafts fill.
    pub verify_tokens: DeviceBufferSpanV1,
}

/// `span` from row `row` of `row_bytes`-byte rows on.
fn at(span: DeviceBufferSpanV1, row: u64, row_bytes: u64) -> DeviceBufferSpanV1 {
    let skip = (row * row_bytes).min(span.length);
    DeviceBufferSpanV1::new(span.buffer, span.offset + skip, span.length - skip)
}

/// One 4-byte element of an item array.
fn element(span: DeviceBufferSpanV1, index: u32) -> DeviceBufferSpanV1 {
    DeviceBufferSpanV1::new(span.buffer, span.offset + 4 * u64::from(index), 4)
}

/// `in_proj_b` sits after the `in_proj_a` columns of each row (twelve at
/// TP4).
fn in_proj_b(s: &RankScratch, shape: Shape) -> DeviceBufferSpanV1 {
    at(s.lin_ab, 1, u64::from(shape.value_heads) * BF16_BYTES)
}

fn hc_stem(shape: Shape, layer: u32, sublayer: Sublayer) -> String {
    let name = match sublayer {
        Sublayer::Attention => "attn_hyper_connection",
        Sublayer::Mlp => "mlp_hyper_connection",
    };
    format!(
        "{}.{name}",
        model_qwen38_contract::views::layer_stem_for(layer, shape)
    )
}

/// The drafter's final hyper-connection mixer.
const DRAFTER_MIXER: &str = "mtp.hyper_connection_mixer";

/// The grouped norm weight the all-reduce after `(layer, sublayer)` applies:
/// the next sublayer's, PLE's query norm before layer 1, or the mixer's.
pub fn next_norm_for(shape: Shape, layer: u32, sublayer: Sublayer) -> String {
    match sublayer {
        Sublayer::Attention => format!("{}.hc_norm.weight", hc_stem(shape, layer, Sublayer::Mlp)),
        Sublayer::Mlp if layer == shape.mtp_layer() => format!("{DRAFTER_MIXER}.hc_norm.weight"),
        Sublayer::Mlp if layer + 1 == shape.ple_layer() => {
            format!("{MODEL}.layers.{}.ple.norm_query.weight", shape.ple_layer())
        }
        Sublayer::Mlp if layer + 1 == shape.layers() => {
            format!("{MODEL}.hyper_connection_mixer.hc_norm.weight")
        }
        Sublayer::Mlp => format!(
            "{}.hc_norm.weight",
            hc_stem(shape, layer + 1, Sublayer::Attention)
        ),
    }
}

/// A grouped norm the program applies: before the first layer (the
/// target's, or the drafter's at its layer), PLE's key, convolution and
/// output norms, or the next norm an all-reduce after `(layer, sublayer)`
/// fuses.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum NormSite {
    First,
    DrafterFirst,
    PleKey,
    PleConv,
    Ple,
    Reduce(u32, Sublayer),
}

impl NormSite {
    /// The site's norm weight view.
    pub(crate) fn weight(self, shape: Shape) -> String {
        let hc_norm = |layer| {
            format!(
                "{}.hc_norm.weight",
                hc_stem(shape, layer, Sublayer::Attention)
            )
        };
        let ple = |name| format!("{MODEL}.layers.{}.ple.{name}.weight", shape.ple_layer());
        match self {
            NormSite::First => hc_norm(0),
            NormSite::DrafterFirst => hc_norm(shape.mtp_layer()),
            NormSite::PleKey => ple("norm_key"),
            NormSite::PleConv => ple("norm_conv"),
            NormSite::Ple => hc_norm(shape.ple_layer()),
            NormSite::Reduce(layer, sublayer) => next_norm_for(shape, layer, sublayer),
        }
    }
}

/// Whether each norm site's weights are F32, resolved once at create from
/// the norm weights the image stores, so a step picks each consumer's op
/// without a lookup.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct NormDtypes {
    /// `First`, `DrafterFirst`, `PleKey`, `PleConv` and `Ple`.
    fixed: [bool; 5],
    /// The all-reduce after each layer's attention and MLP, through the
    /// drafter's layer.
    reduce: Vec<[bool; 2]>,
}

impl NormDtypes {
    /// The sites of `shape` over `norms`, or every site BF16 without them.
    pub(crate) fn new(
        shape: Shape,
        norms: Option<&model_qwen38_contract::norms::NormWeights>,
    ) -> Self {
        let f32 = |site: NormSite| {
            norms.is_some_and(|norms| {
                norms.dtype(&site.weight(shape)) == kernel_core::contract::DType::F32
            })
        };
        Self {
            fixed: [
                NormSite::First,
                NormSite::DrafterFirst,
                NormSite::PleKey,
                NormSite::PleConv,
                NormSite::Ple,
            ]
            .map(f32),
            reduce: (0..=shape.mtp_layer())
                .map(|layer| {
                    [Sublayer::Attention, Sublayer::Mlp]
                        .map(|sublayer| f32(NormSite::Reduce(layer, sublayer)))
                })
                .collect(),
        }
    }

    pub(crate) fn f32(&self, site: NormSite) -> bool {
        match site {
            NormSite::First => self.fixed[0],
            NormSite::DrafterFirst => self.fixed[1],
            NormSite::PleKey => self.fixed[2],
            NormSite::PleConv => self.fixed[3],
            NormSite::Ple => self.fixed[4],
            NormSite::Reduce(layer, sublayer) => self
                .reduce
                .get(layer as usize)
                .is_some_and(|sites| sites[usize::from(sublayer == Sublayer::Mlp)]),
        }
    }
}

/// Which residual buffer holds X before a stage that rewrites it: each
/// all-reduce and the PLE residual write the other one.
pub const fn residual_index_for(shape: Shape, layer: u32, sublayer: Sublayer) -> usize {
    let mlp = matches!(sublayer, Sublayer::Mlp) as u32;
    let flips = 2 * layer + mlp + (layer >= shape.ple_layer()) as u32;
    (flips % 2) as usize
}

/// The residual buffer holding the final X of the last program that ran:
/// the target's after layer 47, or the drafter's after its layer. The
/// drafter's input rows read it.
#[cfg(test)]
pub const TARGET_X: usize = 1 - residual_index(LAYERS - 1, Sublayer::Mlp);
#[cfg(test)]
const _: () = assert!(TARGET_X == 1 - residual_index(MTP_LAYER, Sublayer::Mlp));

const PLE_RESIDUAL_INDEX: usize = 0;

pub const fn target_x(shape: Shape) -> usize {
    1 - residual_index_for(shape, shape.layers() - 1, Sublayer::Mlp)
}

#[cfg(test)]
pub fn next_norm(layer: u32, sublayer: Sublayer) -> String {
    next_norm_for(Shape::TP4, layer, sublayer)
}

#[cfg(test)]
pub const fn residual_index(layer: u32, sublayer: Sublayer) -> usize {
    residual_index_for(Shape::TP4, layer, sublayer)
}

impl RankStage<'_> {
    fn scratch(&self) -> &RankScratch {
        &self.peers[self.rank as usize]
    }

    /// The rank-independent ops of this step's phase.
    fn stage_ops(&self) -> Result<&StageOps, Refusal> {
        self.ops
            .ok_or(Refusal("Qwen3.8 ops are not prepared"))?
            .stage(Key::of(self.phase))
    }

    /// The gated and sparse attention ops of this step's phase.
    fn attention_ops(&self) -> Result<&AttentionOps, Refusal> {
        self.ops
            .ok_or(Refusal("Qwen3.8 ops are not prepared"))?
            .attention(Key::of(self.phase))
    }

    /// Qwen's own ops of this step's phase.
    fn own_ops(&self) -> Result<&OwnStageOps, Refusal> {
        self.ops
            .ok_or(Refusal("Qwen3.8 ops are not prepared"))?
            .own(Key::of(self.phase))
    }

    /// The step's rows as a launch's counts.
    fn counts(&self) -> kernel_core::backend::Counts {
        kernel_core::backend::Counts::rows(self.rows)
    }

    /// `rows` rows over the step's items as a launch's counts.
    fn item_counts(&self, rows: u32) -> kernel_core::backend::Counts {
        kernel_core::backend::Counts::rows(rows).with_items(self.items.len() as u32, 0)
    }

    /// The step's items: one per row for decode, the verify bucket's
    /// items, or the prefill's items.
    fn sequences(&self) -> u32 {
        match self.phase {
            Phase::Decode(rows) => rows,
            Phase::Verify { items, .. } => items,
            Phase::Prefill => self.items.len() as u32,
        }
    }

    /// An attention launch's counts over `rows` rows: the step's items and
    /// context, and every row of the RoPE table, below which the side
    /// inputs keep every rotary position (`side::valid_rope_position`).
    /// A graph bucket's items past the served items are never live, and
    /// the block tables hold a row per served item, so the items are at
    /// most the tables' rows.
    fn attention_counts(&self, rows: u32) -> kernel_core::backend::Counts {
        let table = u64::from(self.shape.max_pages()) * 4;
        let served = u32::try_from(self.inputs.block_tables.length / table).unwrap_or(u32::MAX);
        kernel_core::backend::Counts::rows(rows)
            .with_items(self.sequences().min(served), self.context)
            .with_positions(self.shape.context_tokens() as u32)
    }

    /// Full layer `layer`'s paged K and V cache, bound.
    fn cache(&self, bind: &mut Bind, layer: u32) -> Result<KvCache<View>, Refusal> {
        let pages = self.pages(layer)?;
        let (_, [k_scale, v_scale]) = self.kv_storage(layer);
        Ok(KvCache {
            k: bind.view(pages.planes[0]),
            v: bind.view(pages.planes[1]),
            slots: page_slots(&pages)?,
            k_slot_stride: pages.page_stride_bytes,
            v_slot_stride: pages.page_stride_bytes,
            k_scale: f32::from_bits(k_scale),
            v_scale: f32::from_bits(v_scale),
        })
    }

    /// Full layer `layer`'s block-key plane, bound.
    fn block_keys(&self, bind: &mut Bind, layer: u32) -> Result<BlockKeys<View>, Refusal> {
        let pages = self.pages(layer)?;
        Ok(BlockKeys {
            keys: bind.view(pages.planes[2]),
            slots: page_slots(&pages)?,
            page_stride: pages.page_stride_bytes,
        })
    }

    fn view(&self, name: &str) -> Result<DeviceBufferSpanV1, Refusal> {
        self.views
            .get(name)
            .copied()
            .ok_or(Refusal("Qwen3.8 rank image has no required view"))
    }

    /// A graph phase: decode, verify or a drafter bucket.
    fn decode(&self) -> bool {
        self.phase.bucket().is_some()
    }

    /// The decode-kernel bucket of a graph phase that the decode entries
    /// hold (at most `DECODE_ENTRY_ROWS` rows). Prefill, and graph buckets
    /// above it, run the prefill entries at their prepared rows, which the
    /// device's live counts bound (as SPEC-HICONC's wide MiMo verify).
    fn decode_entries(&self) -> Option<u32> {
        self.phase
            .bucket()
            .filter(|&bucket| bucket <= DECODE_ENTRY_ROWS)
    }

    fn verify(&self) -> bool {
        matches!(self.phase, Phase::Verify { .. })
    }

    fn drafter_scratch(&self) -> Result<&crate::scratch::DrafterScratch, Refusal> {
        self.scratch()
            .drafter
            .as_ref()
            .ok_or(Refusal("Qwen3.8 instance has no drafter buffers"))
    }

    fn drafter_inputs(&self) -> Result<DrafterInputs, Refusal> {
        self.drafter
            .ok_or(Refusal("Qwen3.8 drafter stage outside a drafter program"))
    }

    /// The final mixer's weights: the target's or the drafter's.
    fn mixer_stem(&self) -> String {
        if self.drafter.is_some() {
            DRAFTER_MIXER.into()
        } else {
            format!("{MODEL}.hyper_connection_mixer")
        }
    }

    fn live_tokens(&self) -> DeviceBufferSpanV1 {
        self.inputs.live_tokens
    }

    /// The stored dtype of norm weight `name`, which selects each
    /// consumer's form: F32 only where the prepared rank views store it.
    /// The grouped norm of `site` over `input` into `output`, through the
    /// op of the weights' stored dtype.
    fn norm(
        &self,
        input: DeviceBufferSpanV1,
        site: NormSite,
        output: DeviceBufferSpanV1,
    ) -> Result<Vec<Lowered>, Refusal> {
        let weight = &site.weight(self.shape);
        let ops = self.stage_ops()?;
        let op = if self.norm_dtypes.f32(site) {
            ops.norm_f32
                .as_ref()
                .ok_or(Refusal("Qwen3.8 has no F32 grouped norm prepared"))?
        } else {
            &ops.norm
        };
        let mut bind = Bind::default();
        let buffers = kernel_core::rows::GroupedNormBuffers {
            x: bind.view(input),
            weight: bind.view(self.view(weight)?),
            output: bind.view(output),
            live_rows: bind.view(self.live_tokens()),
        };
        bind.lower(op, &buffers, self.counts())
    }

    /// This key's dense group of `projection` over `rows` rows of
    /// `activation` and the named weight views, into `outputs` (one per
    /// output of the group).
    fn dense(
        &self,
        projection: Projection,
        rows: u32,
        weights: &[&str],
        activation: DeviceBufferSpanV1,
        outputs: &[DeviceBufferSpanV1],
    ) -> Result<Vec<Lowered>, Refusal> {
        let weights = weights
            .iter()
            .map(|weight| self.view(weight))
            .collect::<Result<Vec<_>, _>>()?;
        let op = self.own_ops()?.dense.group(projection)?;
        self.dense_group(
            op, projection, rows, &weights, activation, outputs, None, true,
        )
    }

    /// `op`'s launch of `projection` over `rows` rows. A prepared
    /// component's legacy dense port of the same entry launches through its
    /// registered adapter (`registered`), which launches that entry too.
    #[allow(clippy::too_many_arguments)]
    fn dense_group(
        &self,
        op: &DenseOp,
        projection: Projection,
        rows: u32,
        weights: &[DeviceBufferSpanV1],
        activation: DeviceBufferSpanV1,
        outputs: &[DeviceBufferSpanV1],
        bottleneck: Option<DeviceBufferSpanV1>,
        registered: bool,
    ) -> Result<Vec<Lowered>, Refusal> {
        let symbol = &op.info().symbols[0].0;
        if registered && let Some(id) = self.precision.and_then(|p| p.dense_symbols.get(symbol)) {
            let strides = DenseGroup::of(self.shape, projection).strides;
            let outputs = outputs.iter().copied().zip(strides).collect::<Vec<_>>();
            return Ok(self.built(vec![
                self.registered_dense(id, rows, weights, activation, &outputs)?,
            ]));
        }
        let s = self.scratch();
        let mut bind = Bind::default();
        let buffers = DenseGroupBuffers {
            weights: weights.iter().map(|&weight| bind.view(weight)).collect(),
            activation: bind.view(activation),
            outputs: outputs.iter().map(|&output| bind.view(output)).collect(),
            bottleneck: bottleneck.map(|span| bind.view(span)),
            partial: bind.view(s.workspace),
            semaphores: bind.view(s.gemm_semaphores),
        };
        bind.lower(op, &buffers, kernel_core::backend::Counts::rows(rows))
    }

    /// Launches Qwen built without an op, with the facts of their symbols.
    fn built(&self, launches: Vec<PreparedLaunch>) -> Vec<Lowered> {
        launches
            .into_iter()
            .map(|launch| Lowered {
                facts: self.target.built(&launch),
                launch,
            })
            .collect()
    }

    /// The vocabulary head of `output` over the step's head rows into
    /// `logits`: the least decode bucket's that holds them, or the F32
    /// prefill head (`prefill`).
    fn head(
        &self,
        output: DType,
        prefill: bool,
        activation: DeviceBufferSpanV1,
        logits: DeviceBufferSpanV1,
    ) -> Result<Vec<Lowered>, Refusal> {
        let op = self
            .own_ops()?
            .dense
            .head(output, self.head_rows, prefill)?;
        self.dense_group(
            op,
            Projection::Head(output),
            self.head_rows,
            &[self.view("lm_head.weight")?],
            activation,
            &[logits],
            None,
            true,
        )
    }

    pub fn lower(&self, stage: Stage) -> Result<Vec<PreparedLaunch>, Refusal> {
        self.lower_part(stage, 0)
    }

    /// Lower with a media source-map offset for a contiguous prefill part.
    pub fn lower_part(&self, stage: Stage, first_row: u32) -> Result<Vec<PreparedLaunch>, Refusal> {
        Ok(launches(self.lower_with_facts(stage, first_row)?))
    }

    /// [`Self::lower_part`], with each launch's facts for the pipelined
    /// prefill.
    pub(crate) fn lower_with_facts(
        &self,
        stage: Stage,
        first_row: u32,
    ) -> Result<Vec<Lowered>, Refusal> {
        let ple_layer = self.shape.ple_layer();
        if let Some(launches) = self.prepared_stage(stage)? {
            return Ok(launches);
        }
        if let Some(launches) = self.context_parallel_stage(stage)? {
            return Ok(launches);
        }
        let s = self.scratch();
        let rows = self.rows;
        // A prefill whose items all continue in a later chunk needs no head.
        if self.head_rows == 0
            && matches!(
                stage,
                Stage::MixerIn
                    | Stage::MixerBottleneck
                    | Stage::MixerUp
                    | Stage::MixerMix
                    | Stage::GatherHeadRows
                    | Stage::LmHead
                    | Stage::GatherLogits
            )
        {
            return Ok(Vec::new());
        }
        match stage {
            Stage::PleStage => {
                let staging = s.ple_staging[usize::from(self.decode())];
                let mut bind = Bind::default();
                let buffers = PleStageBuffers {
                    pool: bind.view(s.ple_pool),
                    staging: bind.view(staging.input),
                    slot_ids: bind.view(self.side.ple_working),
                    prefix_len: bind.view(self.inputs.prefix_len),
                    live_items: bind.view(self.inputs.live_batch),
                };
                // A per-item state copy, staged for the step's rows.
                bind.lower(&self.own_ops()?.ple_stage, &buffers, self.item_counts(rows))
            }
            // Prefill and decode use the slot itself. Continuing prefill
            // already has its initial state there; fresh prefill initializes it.
            Stage::LinearStateCopy => Ok(Vec::new()),
            Stage::MediaInject => match self.media {
                Some(media) => Ok(self.built(vec![media.inject_part(
                    s.embed,
                    self.live_tokens(),
                    first_row,
                    rows,
                )?])),
                None => Ok(Vec::new()),
            },
            Stage::ExpandStreams => {
                let mut bind = Bind::default();
                let buffers = RowCopyBuffers {
                    source: bind.view(s.embed),
                    destination: bind.view(s.x[0]),
                    live_rows: bind.view(self.live_tokens()),
                };
                bind.lower(&self.stage_ops()?.expand_streams, &buffers, self.counts())
            }
            Stage::FirstNorm if self.drafter.is_some() => self.norm(
                s.x[residual_index_for(self.shape, self.shape.mtp_layer(), Sublayer::Attention)],
                NormSite::DrafterFirst,
                s.xn,
            ),
            Stage::FirstNorm => self.norm(s.x[0], NormSite::First, s.xn),
            // Lowered by their projection adapters before this match.
            Stage::Embedding | Stage::NgramGather => {
                Err(Refusal("projection stage outside its adapter"))
            }
            Stage::NgramAllGather => {
                let ops = self.collective_ops()?;
                let mut launches = Vec::new();
                for target in 0..self.shape.ranks() {
                    launches.extend(self.gather(
                        ops.ngram(self.rank, target)?,
                        s.ngram_local,
                        |peer| peer.ngram,
                        target as usize,
                        rows,
                    )?);
                }
                Ok(launches)
            }
            Stage::PleProject => {
                let ple = format!("{MODEL}.layers.{ple_layer}.ple");
                let mut launches = self.dense(
                    Projection::PleKey,
                    rows,
                    &[&format!("{ple}.key_proj.weight")],
                    s.ngram,
                    &[s.hc_up],
                )?;
                launches.extend(self.dense(
                    Projection::PleValue,
                    rows,
                    &[&format!("{ple}.value_proj.weight")],
                    s.ngram,
                    &[s.ple_value],
                )?);
                Ok(launches)
            }
            Stage::PleKeyNorm => self.norm(s.hc_up, NormSite::PleKey, s.ple_key_normed),
            Stage::PleGate => {
                // The all-reduce before layer 1 left norm_query(X) in xn.
                let mut bind = Bind::default();
                let buffers = kernel_core::rows::PleGateBuffers {
                    key: bind.view(s.ple_key_normed),
                    query: bind.view(s.xn),
                    value: bind.view(s.ple_value),
                    output: bind.view(s.ple_gated),
                    live_rows: bind.view(self.live_tokens()),
                };
                bind.lower(&self.stage_ops()?.ple_gate, &buffers, self.counts())
            }
            Stage::PleConvNorm => self.norm(s.ple_gated, NormSite::PleConv, s.hc_up),
            Stage::PleConv => self.ple_conv(),
            Stage::PleCheckpoints => {
                let mut bind = Bind::default();
                let buffers = PleCheckpointsBuffers {
                    input: bind.view(s.hc_up),
                    tokens: bind.view(self.inputs.token_ids),
                    staged: bind.view(s.ple_staging[1].input),
                    pool: bind.view(s.ple_pool),
                    cu_rows: bind.view(self.side.cu_rows),
                    q_len: bind.view(self.inputs.q_len),
                    live_items: bind.view(self.inputs.live_batch),
                    checkpoint_ids: bind.view(self.side.ple_checkpoints),
                };
                bind.lower(
                    self.own_ops()?.ple_checkpoints()?,
                    &buffers,
                    self.item_counts(rows),
                )
            }
            Stage::PleAnchor if !self.anchored => Ok(Vec::new()),
            Stage::PleAnchor => {
                let mut bind = Bind::default();
                let buffers = PleAnchorBuffers {
                    input: bind.view(s.hc_up),
                    tokens: bind.view(self.inputs.token_ids),
                    staged: bind.view(s.ple_staging[0].input),
                    anchors: bind.view(s.anchor_ple),
                    cu_rows: bind.view(self.side.cu_rows),
                    prefix_len: bind.view(self.inputs.prefix_len),
                    anchor_slots: bind.view(self.side.anchor_slot),
                    anchor_ends: bind.view(self.side.anchor_end),
                    live_items: bind.view(self.inputs.live_batch),
                };
                // A per-item state copy at the step's anchors, for its rows.
                bind.lower(
                    self.own_ops()?.ple_anchor()?,
                    &buffers,
                    self.item_counts(rows),
                )
            }
            Stage::MtpGather => {
                let drafter = self.drafter_inputs()?;
                let scratch = self.drafter_scratch()?;
                let last =
                    s.x[1 - residual_index_for(self.shape, self.shape.layers() - 1, Sublayer::Mlp)];
                let mut bind = Bind::default();
                let buffers = MtpGatherBuffers {
                    last: bind.view(last),
                    retained: bind.view(scratch.retained),
                    sources: bind.view(drafter.sources),
                    token_sources: bind.view(drafter.token_sources),
                    drafts: bind.view(scratch.drafts),
                    x: bind.view(s.ple_gated),
                    token_ids: bind.view(self.inputs.token_ids),
                    live_rows: bind.view(self.live_tokens()),
                };
                bind.lower(&self.own_ops()?.mtp_gather, &buffers, self.counts())
            }
            Stage::MtpNorms => {
                let mut bind = Bind::default();
                let buffers = MtpNormsBuffers {
                    x: bind.view(s.ple_gated),
                    hidden_weight: bind.view(self.view("mtp.pre_fc_norm_hidden.weight")?),
                    embed: bind.view(s.embed),
                    embed_weight: bind.view(self.view("mtp.pre_fc_norm_embedding.weight")?),
                    x_normed: bind.view(s.hc_up),
                    embed_normed: bind.view(s.ple_value),
                    live_rows: bind.view(self.live_tokens()),
                };
                bind.lower(&self.own_ops()?.mtp_norms, &buffers, self.counts())
            }
            Stage::MtpFc => self.drafter_fc(),
            Stage::MtpCombine => {
                let x = s.x
                    [residual_index_for(self.shape, self.shape.mtp_layer(), Sublayer::Attention)];
                let mut bind = Bind::default();
                let buffers = MtpCombineBuffers {
                    hidden: bind.view(s.ple_key_normed),
                    token: bind.view(s.ngram),
                    x: bind.view(x),
                    live_rows: bind.view(self.live_tokens()),
                };
                bind.lower(&self.own_ops()?.mtp_combine, &buffers, self.counts())
            }
            Stage::DraftBest => {
                let scratch = self.drafter_scratch()?;
                let mut bind = Bind::default();
                let buffers = DraftArgmaxPartialBuffers {
                    logits: bind.view(s.local_logits),
                    best: bind.view(scratch.best),
                    live_rows: bind.view(self.inputs.logit_rows),
                };
                bind.lower(
                    self.own_ops()?.draft_partial(self.rank)?,
                    &buffers,
                    kernel_core::backend::Counts::rows(self.head_rows),
                )
            }
            // Each rank's best of a row, four words, reaches every rank.
            Stage::DraftExchange => {
                let best = self.drafter_scratch()?.best;
                let ops = self.collective_ops()?;
                let mut launches = Vec::new();
                for target in 0..self.shape.ranks() {
                    launches.extend(self.gather(
                        ops.drafts(self.rank, target)?,
                        best,
                        |peer| peer.drafter.map_or(peer.unused, |drafter| drafter.bests),
                        target as usize,
                        self.head_rows,
                    )?);
                }
                Ok(launches)
            }
            Stage::DraftFinal => {
                let drafter = self.drafter_inputs()?;
                let scratch = self.drafter_scratch()?;
                let mut bind = Bind::default();
                let buffers = DraftArgmaxFinalBuffers {
                    bests: bind.view(scratch.bests),
                    slots: bind.view(drafter.slots),
                    targets: bind.view(drafter.targets),
                    drafts: bind.view(scratch.drafts),
                    token_ids: bind.view(drafter.verify_tokens),
                    live_rows: bind.view(self.inputs.logit_rows),
                };
                bind.lower(
                    &self.own_ops()?.draft_final,
                    &buffers,
                    kernel_core::backend::Counts::rows(self.head_rows),
                )
            }
            Stage::PleResidual => {
                let mut bind = Bind::default();
                let buffers = PleResidualBuffers {
                    x: bind.view(s.x[PLE_RESIDUAL_INDEX]),
                    gated: bind.view(s.ple_gated),
                    conv: bind.view(s.ple_conv),
                    out: bind.view(s.x[1 - PLE_RESIDUAL_INDEX]),
                    live_rows: bind.view(self.live_tokens()),
                };
                bind.lower(&self.own_ops()?.ple_residual, &buffers, self.counts())
            }
            Stage::PleNorm => self.norm(s.x[1 - PLE_RESIDUAL_INDEX], NormSite::Ple, s.xn),
            Stage::HcIn(layer, sublayer) => self.hc_in(&hc_stem(self.shape, layer, sublayer), true),
            Stage::MixerIn => self.hc_in(&self.mixer_stem(), false),
            // Decode buckets run the bottleneck and the up projection inside
            // the fused mix; wider ones run the three stages as prefill does.
            Stage::HcBottleneck(..) | Stage::MixerBottleneck | Stage::HcUp(..) | Stage::MixerUp
                if self.decode_entries().is_some() =>
            {
                Ok(Vec::new())
            }
            Stage::HcMix(layer, sublayer) if self.decode_entries().is_some() => {
                self.hc_up_mix(&hc_stem(self.shape, layer, sublayer))
            }
            Stage::MixerMix if self.decode_entries().is_some() => {
                self.hc_up_mix(&self.mixer_stem())
            }
            Stage::HcBottleneck(..) | Stage::MixerBottleneck => {
                let mut bind = Bind::default();
                let buffers = kernel_core::rows::BottleneckBuffers {
                    down: bind.view(s.hc_down),
                    output: bind.view(s.hc_bottleneck),
                    live_rows: bind.view(self.live_tokens()),
                };
                bind.lower(&self.stage_ops()?.bottleneck, &buffers, self.counts())
            }
            Stage::HcUp(layer, sublayer) => self.hc_up(&hc_stem(self.shape, layer, sublayer)),
            Stage::MixerUp => self.hc_up(&self.mixer_stem()),
            Stage::HcMix(..) | Stage::MixerMix => {
                let mut bind = Bind::default();
                let buffers = kernel_core::rows::MixBuffers {
                    up: bind.view(s.hc_up),
                    normed: bind.view(s.xn),
                    output: bind.view(s.u),
                    live_rows: bind.view(self.live_tokens()),
                };
                bind.lower(&self.stage_ops()?.mix, &buffers, self.counts())
            }
            Stage::LinearInProj(layer) => self.linear_in(layer),
            Stage::LinearConv(layer) => self.linear_conv(layer),
            Stage::LinearDelta(layer) => self.linear_delta(layer),
            Stage::LinearNorm(layer) => {
                let mut bind = Bind::default();
                let buffers = kernel_core::recurrent::GatedNormBuffers {
                    input: bind.view(s.lin_core),
                    gate: bind.view(s.lin_z),
                    gate_stride: self.shape.linear_value,
                    weight: bind.view(
                        self.view(&format!("{MODEL}.layers.{layer}.linear_attn.norm.weight"))?,
                    ),
                    output: bind.view(s.lin_normed),
                    live_rows: bind.view(self.live_tokens()),
                };
                bind.lower(&self.stage_ops()?.gated_norm, &buffers, self.counts())
            }
            Stage::LinearOut(layer) => self.dense(
                Projection::LinearOutput,
                rows,
                &[&format!(
                    "{MODEL}.layers.{layer}.linear_attn.out_proj.weight"
                )],
                s.lin_normed,
                &[s.partial],
            ),
            Stage::AttnInProj(layer) => self.attention_in(layer),
            Stage::AttnQkNormRopeKv(layer) => {
                let attn = format!(
                    "{}.self_attn",
                    model_qwen38_contract::views::layer_stem_for(layer, self.shape)
                );
                let (fp8, _) = self.kv_storage(layer);
                let mut bind = Bind::default();
                let buffers = GatedKvWriteBuffers {
                    qkv: bind.view(s.attn_qkv),
                    q: bind.view(s.attn_q),
                    cache: self.cache(&mut bind, layer)?,
                    block_tables: bind.view(self.inputs.block_tables),
                    token_batch: bind.view(self.inputs.token_batch),
                    positions: bind.view(self.inputs.positions),
                    live_rows: bind.view(self.live_tokens()),
                    rope: FoldInputs {
                        cos_sin: Some(bind.view(s.cos_sin)),
                        q_norm: Some(bind.view(self.view(&format!("{attn}.q_norm.weight"))?)),
                        k_norm: Some(bind.view(self.view(&format!("{attn}.k_norm.weight"))?)),
                        rope_positions: Some(bind.view(self.side.rope_positions)),
                    },
                };
                bind.lower(
                    self.attention_ops()?.kv_write(fp8),
                    &buffers,
                    self.attention_counts(rows),
                )
            }
            Stage::IndexQuery(layer) => {
                let mut bind = Bind::default();
                let buffers = IndexQueryBuffers {
                    projection: bind.view(s.attn_index),
                    weight: bind.view(self.view(&format!(
                        "{}.self_attn.indexer.q_layernorm.weight",
                        model_qwen38_contract::views::layer_stem_for(layer, self.shape)
                    ))?),
                    rope_positions: bind.view(self.side.rope_positions),
                    cos_sin: bind.view(s.cos_sin),
                    live_rows: bind.view(self.live_tokens()),
                    q: bind.view(s.index_q),
                };
                bind.lower(
                    &self.attention_ops()?.index_query,
                    &buffers,
                    self.attention_counts(rows),
                )
            }
            Stage::IndexKeyWrite(layer) => {
                let (keys, positions) = self.ring_layer(layer);
                let (anchor_keys, anchor_positions) = self.anchor_ring_layer(layer);
                let mut bind = Bind::default();
                let buffers = SparseKeyWriteBuffers {
                    projection: bind.view(s.attn_index),
                    weight: bind.view(self.view(&format!(
                        "{}.self_attn.indexer.k_layernorm.weight",
                        model_qwen38_contract::views::layer_stem_for(layer, self.shape)
                    ))?),
                    rope_positions: bind.view(self.side.rope_positions),
                    cos_sin: bind.view(s.cos_sin),
                    token_batch: bind.view(self.inputs.token_batch),
                    positions: bind.view(self.inputs.positions),
                    live_rows: bind.view(self.live_tokens()),
                    slot_ids: bind.view(self.inputs.slot_ids),
                    rings: KeyRings {
                        keys: bind.view(keys),
                        positions: bind.view(positions),
                        slots: self.ring_layout.slots as u32,
                    },
                    anchor_slots: bind.view(self.side.anchor_slot),
                    anchor_ends: bind.view(self.side.anchor_end),
                    anchors: KeyRings {
                        keys: bind.view(anchor_keys),
                        positions: bind.view(anchor_positions),
                        slots: ANCHOR_ITEMS,
                    },
                    block_tables: bind.view(self.inputs.block_tables),
                    block_keys: self.block_keys(&mut bind, layer)?,
                };
                bind.lower(
                    &self.attention_ops()?.key_write,
                    &buffers,
                    self.attention_counts(rows),
                )
            }
            Stage::BlockSelect(layer) => self.block_select(layer),
            Stage::SparseAttention(layer) => self.sparse_attention(layer),
            Stage::AttnGate(_) => {
                let mut bind = Bind::default();
                let buffers = OutputGateBuffers {
                    attention: bind.view(s.attn_out),
                    qkv: bind.view(s.attn_qkv),
                    output: bind.view(s.attn_gated),
                    live_rows: bind.view(self.live_tokens()),
                };
                bind.lower(&self.attention_ops()?.output_gate, &buffers, self.counts())
            }
            Stage::AttnOut(layer) => self.dense(
                Projection::AttentionOutput,
                rows,
                &[&format!(
                    "{}.self_attn.o_proj.weight",
                    model_qwen38_contract::views::layer_stem_for(layer, self.shape)
                )],
                s.attn_gated,
                &[s.partial],
            ),
            Stage::Router(layer) => self.dense(
                Projection::Router,
                rows,
                &[&format!(
                    "{}.mlp.gate.weight",
                    model_qwen38_contract::views::layer_stem_for(layer, self.shape)
                )],
                s.u,
                &[s.router_logits],
            ),
            Stage::Route(_) => self.route(),
            Stage::ExpertsGateUp(layer) => self.experts(layer, GatedMlpProjection::GateUp),
            Stage::ExpertsDown(layer) => self.experts(layer, GatedMlpProjection::Down),
            Stage::SharedGateUp(layer) => self.dense(
                Projection::SharedGateUp,
                rows,
                &[&format!(
                    "{}.mlp.shared_expert.gate_up_proj.weight",
                    model_qwen38_contract::views::layer_stem_for(layer, self.shape)
                )],
                s.u,
                &[s.shared_intermediate],
            ),
            Stage::SharedDown(layer) => self.dense(
                Projection::SharedDown,
                rows,
                &[&format!(
                    "{}.mlp.shared_expert.down_proj.weight",
                    model_qwen38_contract::views::layer_stem_for(layer, self.shape)
                )],
                s.shared_intermediate,
                &[s.shared_out],
            ),
            Stage::Combine(layer) => {
                let mut bind = Bind::default();
                let buffers = SharedCombineBuffers {
                    routed: bind.view(s.moe_routed),
                    ids: bind.view(s.topk_ids),
                    shared: bind.view(s.shared_out),
                    x: bind.view(s.u),
                    gate_weight: bind.view(self.view(&format!(
                        "{}.mlp.shared_expert_gate.weight",
                        model_qwen38_contract::views::layer_stem_for(layer, self.shape)
                    ))?),
                    output: bind.view(s.partial),
                    live_rows: bind.view(self.live_tokens()),
                };
                bind.lower(self.moe_ops()?.combine(self.rank)?, &buffers, self.counts())
            }
            Stage::Reduce(layer, sublayer) => self.reduce(layer, sublayer),
            Stage::GatherHeadRows => {
                // The step's rows are the table the head's rows come from.
                let mut bind = Bind::default();
                let buffers = kernel_core::rows::EmbeddingBuffers {
                    table: bind.view(s.u),
                    ids: bind.view(self.inputs.row_indices),
                    output: bind.view(s.head_rows),
                    live_rows: bind.view(self.inputs.logit_rows),
                    table_rows: rows,
                };
                bind.lower(
                    &self.stage_ops()?.head_rows,
                    &buffers,
                    kernel_core::backend::Counts::rows(self.head_rows),
                )
            }
            Stage::LmHead if self.scored => {
                self.head(DType::F32, true, s.head_rows, s.local_scored)
            }
            Stage::LmHead => {
                // Decode reads every bucket row in place; prefill and the
                // drafter gather their head rows first.
                let activation = if self.decode() && self.drafter.is_none() {
                    s.u
                } else {
                    s.head_rows
                };
                // At most one head row per item: the decode entries serve
                // prefill too, reading only the live rows of `head_rows`.
                // More head rows than their largest bucket take the F32
                // prefill head, whose output rows are the scored prompt
                // rows' buffer. Its one K segment is the decode entries' K
                // chain, so a row's logits never depend on which head the
                // other items of its step select.
                if self.head_rows > DECODE_ENTRY_ROWS {
                    if self.drafter.is_some() {
                        return Err(Refusal(
                            "Qwen3.8 drafter head rows exceed the decode buckets",
                        ));
                    }
                    return self.head(DType::F32, true, activation, s.local_scored);
                }
                // Every target logits row is F32, as MiMo's default; the
                // drafter's argmax reads BF16. Prefill rows share the scored
                // prompt rows' F32 buffers.
                let (output, logits) = match (self.drafter.is_some(), self.decode()) {
                    (true, _) => (DType::Bf16, s.local_logits),
                    (false, true) => (DType::F32, s.local_logits),
                    (false, false) => (DType::F32, s.local_scored),
                };
                self.head(output, false, activation, logits)
            }
            Stage::GatherLogits if self.decode() && self.exchange.is_some() => {
                let states = self.exchange.expect("exchange state");
                if self.drafter.is_some() || states.len() != self.peers.len() {
                    return Err(Refusal("Qwen3.8 target logits exchange state"));
                }
                let mut bind = Bind::default();
                let local = if self.head_rows > DECODE_ENTRY_ROWS {
                    s.local_scored
                } else {
                    s.local_logits
                };
                let buffers = GatherBuffers {
                    input: bind.view(local),
                    ranks: self
                        .peers
                        .iter()
                        .zip(states)
                        .map(|(peer, state)| RankViews {
                            output: bind.view(peer.gathered_decode),
                            state: vec![bind.view(state.flags), bind.view(state.epoch)],
                        })
                        .collect(),
                };
                bind.lower(
                    self.collective_ops()?.exchange(self.rank)?,
                    &buffers,
                    kernel_core::backend::Counts::rows(self.head_rows),
                )
            }
            // F32 rows move as pairs of BF16 values. Decode rows past the
            // decode heads' buckets are in the prefill head's output.
            Stage::GatherLogits if self.decode() => self.gather(
                self.collective_ops()?.logits(self.rank)?,
                if self.head_rows > DECODE_ENTRY_ROWS {
                    s.local_scored
                } else {
                    s.local_logits
                },
                |peer| peer.gathered_decode,
                logits_gather(self.shape),
                self.head_rows,
            ),
            Stage::GatherLogits => self.gather(
                self.collective_ops()?.logits(self.rank)?,
                s.local_scored,
                |peer| peer.gathered_scored,
                logits_gather(self.shape),
                self.head_rows,
            ),
        }
    }

    /// Per item: the PLE convolution over its rows from the staged history;
    /// decode takes every bucket item at once.
    fn ple_conv(&self) -> Result<Vec<Lowered>, Refusal> {
        let ple_layer = self.shape.ple_layer();
        let s = self.scratch();
        let staging = s.ple_staging[usize::from(self.decode())];
        let weight = self.view(&format!("{MODEL}.layers.{ple_layer}.ple.conv1d.weight"))?;
        // The key's form: verify items, decode items or one prefill item.
        let op = &self.own_ops()?.ple_conv;
        if self.decode() {
            let verify = self.verify();
            let mut bind = Bind::default();
            let buffers = PleConvBuffers {
                input: bind.view(s.hc_up),
                weight: bind.view(weight),
                state: bind.view(staging.input),
                output: bind.view(s.ple_conv),
                state_out: bind.view(staging.output),
                cu_rows: verify.then(|| bind.view(self.side.cu_rows)),
                q_len: bind.view(self.inputs.q_len),
                live_items: verify.then(|| bind.view(self.inputs.live_batch)),
            };
            let counts = if verify {
                self.item_counts(self.rows)
            } else {
                self.counts()
            };
            return bind.lower(op, &buffers, counts);
        }
        let (wide, entry) = (
            u64::from(self.shape.wide()) * BF16_BYTES,
            self.shape.ple_entry_bytes(),
        );
        let mut launches = Vec::new();
        for (item, rows) in self.items.iter().enumerate() {
            let start = u64::from(rows.start);
            let mut bind = Bind::default();
            let buffers = PleConvBuffers {
                input: bind.view(at(s.hc_up, start, wide)),
                weight: bind.view(weight),
                state: bind.view(at(staging.input, item as u64, entry)),
                output: bind.view(at(s.ple_conv, start, wide)),
                state_out: bind.view(at(staging.output, item as u64, entry)),
                cu_rows: None,
                q_len: bind.view(element(self.inputs.q_len, item as u32)),
                live_items: None,
            };
            launches.extend(bind.lower(
                op,
                &buffers,
                kernel_core::backend::Counts::rows(rows.rows),
            )?);
        }
        Ok(launches)
    }

    /// The drafter's `fc_hidden` over each stream of every row and its
    /// `fc_embedding` of the token. A graph bucket runs the cluster entries
    /// while their rows fit, like the target's decode GEMMs.
    ///
    /// No recorded operation owns these GEMMs, so they launch directly even
    /// when a recorded operation's legacy entry has the same symbol: the
    /// drafter's FC shares its entry with the TP4 linear QKV projection
    /// (both 2560 by 2560 BF16), but runs one row per stream, more rows
    /// than the drafter phase's bound use admits.
    fn drafter_fc(&self) -> Result<Vec<Lowered>, Refusal> {
        let s = self.scratch();
        let dense = &self.own_ops()?.dense;
        // Each stream is one GEMM row; a prefill's streams run in pieces of
        // the prefill entry's rows.
        let row_bytes = u64::from(self.shape.hidden()) * BF16_BYTES;
        let mut launches = Vec::new();
        let streams = self.rows * self.shape.streams();
        for first in (0..streams).step_by(PREFILL_TOKENS as usize) {
            let rows = (streams - first).min(PREFILL_TOKENS);
            launches.extend(self.dense_group(
                &dense.mtp_streams,
                Projection::MtpFc,
                rows,
                &[self.view("mtp.fc_hidden.weight")?],
                at(s.hc_up, u64::from(first), row_bytes),
                &[at(s.ple_key_normed, u64::from(first), row_bytes)],
                None,
                false,
            )?);
        }
        launches.extend(self.dense_group(
            &dense.mtp_rows,
            Projection::MtpFc,
            self.rows,
            &[self.view("mtp.fc_embedding.weight")?],
            s.ple_value,
            &[s.ngram],
            None,
            false,
        )?);
        Ok(launches)
    }

    /// One target's row gather of `input` into every rank's `output`.
    /// `op`'s gather of `rows` rows of `input` onto the outputs `output`
    /// picks of each rank, on gather state `set`.
    fn gather(
        &self,
        op: &Prepared<RowGatherBuffers<View>>,
        input: DeviceBufferSpanV1,
        output: impl Fn(&RankScratch) -> DeviceBufferSpanV1,
        set: usize,
        rows: u32,
    ) -> Result<Vec<Lowered>, Refusal> {
        let mut bind = Bind::default();
        let buffers = RowGatherBuffers {
            input: bind.view(input),
            indices: None,
            ranks: self
                .peers
                .iter()
                .map(|peer| {
                    let state = peer
                        .gathers
                        .get(set)
                        .ok_or(Refusal("Qwen3.8 has no such gather state"))?;
                    Ok(RankViews {
                        output: bind.view(output(peer)),
                        state: vec![bind.view(state.flags), bind.view(state.epoch)],
                    })
                })
                .collect::<Result<_, Refusal>>()?,
        };
        bind.lower(op, &buffers, kernel_core::backend::Counts::rows(rows))
    }

    fn precompute_hc(&self) -> bool {
        self.hc_precompute && matches!(self.decode_entries(), Some(16 | 32))
    }

    fn hc_in(&self, stem: &str, inject: bool) -> Result<Vec<Lowered>, Refusal> {
        let s = self.scratch();
        let down = format!("{stem}.input_mix_weight_down.weight");
        let block = format!("{stem}.block_inject_weight.weight");
        if self.decode_entries().is_none() {
            let mut launches =
                self.dense(Projection::HcDown, self.rows, &[&down], s.xn, &[s.hc_down])?;
            if inject {
                launches.extend(self.dense(
                    Projection::HcInject,
                    self.rows,
                    &[&block],
                    s.xn,
                    &[s.hc_inject],
                )?);
            }
            return Ok(launches);
        }
        let (projection, weights, outputs) = if inject {
            (
                Projection::HcIn,
                vec![self.view(&down)?, self.view(&block)?],
                vec![s.hc_down, s.hc_inject],
            )
        } else {
            (Projection::HcDown, vec![self.view(&down)?], vec![s.hc_down])
        };
        // Where the instance precomputes it, the producer also writes the
        // bottleneck the fused up mix reads.
        self.dense_group(
            self.own_ops()?.dense.group(projection)?,
            projection,
            self.rows,
            &weights,
            s.xn,
            &outputs,
            self.precompute_hc().then_some(s.hc_bottleneck),
            true,
        )
    }

    /// The up projection of prefill and of buckets wider than the decode
    /// entries'; the others fuse it into the mix.
    fn hc_up(&self, stem: &str) -> Result<Vec<Lowered>, Refusal> {
        let s = self.scratch();
        self.dense(
            Projection::HcUp,
            self.rows,
            &[&format!("{stem}.input_mix_weight_up.weight")],
            s.hc_bottleneck,
            &[s.hc_up],
        )
    }

    /// The fused up mix of a bucket the decode entries hold, over the
    /// bottleneck where the instance precomputes it, else the down
    /// projection (the prepared op's input).
    fn hc_up_mix(&self, stem: &str) -> Result<Vec<Lowered>, Refusal> {
        let s = self.scratch();
        let mut bind = Bind::default();
        let buffers = HcUpMixBuffers {
            weight: bind.view(self.view(&format!("{stem}.input_mix_weight_up.weight"))?),
            input: bind.view(if self.precompute_hc() {
                s.hc_bottleneck
            } else {
                s.hc_down
            }),
            normed: bind.view(s.xn),
            output: bind.view(s.u),
            live_rows: bind.view(self.live_tokens()),
        };
        bind.lower(self.own_ops()?.up_mix()?, &buffers, self.counts())
    }

    fn linear_in(&self, layer: u32) -> Result<Vec<Lowered>, Refusal> {
        let s = self.scratch();
        let linear = format!("{MODEL}.layers.{layer}.linear_attn");
        let weight = |name: &str| format!("{linear}.{name}.weight");
        let b = in_proj_b(s, self.shape);
        if self.decode_entries().is_some() {
            return self.dense(
                Projection::LinearIn,
                self.rows,
                &[
                    &weight("in_proj_qkv"),
                    &weight("in_proj_z"),
                    &weight("in_proj_a"),
                    &weight("in_proj_b"),
                ],
                s.u,
                &[s.lin_qkv, s.lin_z, s.lin_ab, b],
            );
        }
        let mut launches = self.dense(
            Projection::LinearQkv,
            self.rows,
            &[&weight("in_proj_qkv")],
            s.u,
            &[s.lin_qkv],
        )?;
        launches.extend(self.dense(
            Projection::LinearZ,
            self.rows,
            &[&weight("in_proj_z")],
            s.u,
            &[s.lin_z],
        )?);
        launches.extend(self.dense(
            Projection::LinearAb,
            self.rows,
            &[&weight("in_proj_a"), &weight("in_proj_b")],
            s.u,
            &[s.lin_ab],
        )?);
        Ok(launches)
    }

    /// Layer `layer`'s plane of a linear-state pool.
    fn pool_layer(
        &self,
        pool: DeviceBufferSpanV1,
        layer: u32,
        entry_bytes: u64,
    ) -> DeviceBufferSpanV1 {
        let bytes = self.pool_entries * entry_bytes;
        DeviceBufferSpanV1::new(
            pool.buffer,
            pool.offset + self.shape.ordinal(layer) * bytes,
            bytes,
        )
    }

    fn linear_conv(&self, layer: u32) -> Result<Vec<Lowered>, Refusal> {
        let s = self.scratch();
        let mut bind = Bind::default();
        let buffers = kernel_core::recurrent::ConvBuffers {
            input: bind.view(s.lin_qkv),
            input_stride: self.shape.linear_qkv,
            weights: vec![
                bind.view(self.view(&format!("{MODEL}.layers.{layer}.linear_attn.conv1d.weight"))?),
            ],
            state: bind.view(self.pool_layer(s.conv_pool, layer, self.shape.conv_entry_bytes)),
            output: bind.view(s.lin_conv),
            slots: self.linear_slots(&mut bind),
        };
        bind.lower(&self.stage_ops()?.conv, &buffers, self.linear_counts())
    }

    /// The per-sequence operands of a linear-attention launch: graph steps
    /// continue each sequence's committed entry and write its verify
    /// checkpoints; prefill reads its working, initial and anchor entries.
    fn linear_slots(&self, bind: &mut Bind) -> kernel_core::recurrent::RecurrentSlots<View> {
        let side = self.side;
        let decode = self.decode();
        kernel_core::recurrent::RecurrentSlots {
            row_offsets: Some(bind.view(side.cu_rows)),
            start_positions: None,
            slot_ids: bind.view(if decode { side.committed } else { side.working }),
            initial: (!decode).then(|| bind.view(side.initial)),
            anchor_ids: (!decode).then(|| bind.view(side.anchor_entry)),
            checkpoint_ids: decode.then(|| bind.view(side.checkpoints)),
            live_sequences: bind.view(self.inputs.live_batch),
        }
    }

    /// A linear-attention launch's rows and sequences: one row per item for
    /// decode, the verify bucket's items, or the prefill's rows and items.
    fn linear_counts(&self) -> kernel_core::backend::Counts {
        self.counts().with_items(self.sequences(), 0)
    }

    fn linear_delta(&self, layer: u32) -> Result<Vec<Lowered>, Refusal> {
        let s = self.scratch();
        let linear = format!("{MODEL}.layers.{layer}.linear_attn");
        let stride = self.shape.linear_ab_stride;
        let mut bind = Bind::default();
        let buffers = kernel_core::recurrent::DeltaRuleBuffers {
            qkv: bind.view(s.lin_conv),
            qkv_stride: self.shape.linear_qkv,
            decay: bind.view(s.lin_ab),
            decay_stride: stride,
            beta: bind.view(in_proj_b(s, self.shape)),
            beta_stride: stride,
            a_log: bind.view(self.view(&format!("{linear}.A_log"))?),
            dt_bias: bind.view(self.view(&format!("{linear}.dt_bias"))?),
            state: bind.view(self.pool_layer(s.delta_pool, layer, self.shape.delta_entry_bytes)),
            output: bind.view(s.lin_core),
            // Prefill chunks its rows through the planner workspace.
            workspace: (!self.decode()).then(|| bind.view(s.workspace)),
            slots: self.linear_slots(&mut bind),
        };
        bind.lower(&self.stage_ops()?.delta, &buffers, self.linear_counts())
    }

    /// Full layer `layer`'s q_proj, k_proj and v_proj views as one span.
    fn packed_qkv(&self, layer: u32) -> Result<DeviceBufferSpanV1, Refusal> {
        let attn = format!(
            "{}.self_attn",
            model_qwen38_contract::views::layer_stem_for(layer, self.shape)
        );
        let [q, k, v] =
            ["q_proj", "k_proj", "v_proj"].map(|name| self.view(&format!("{attn}.{name}.weight")));
        let (q, k, v) = (q?, k?, v?);
        if q.buffer != v.buffer
            || q.offset + q.length != k.offset
            || k.offset + k.length != v.offset
        {
            return Err(Refusal("Qwen3.8 q, k and v projections are not packed"));
        }
        Ok(DeviceBufferSpanV1::new(
            q.buffer,
            q.offset,
            v.offset + v.length - q.offset,
        ))
    }

    fn attention_in(&self, layer: u32) -> Result<Vec<Lowered>, Refusal> {
        let s = self.scratch();
        let attn = format!(
            "{}.self_attn",
            model_qwen38_contract::views::layer_stem_for(layer, self.shape)
        );
        let weight = |name: &str| format!("{attn}.{name}.weight");
        // Q+gate, K and V land in one packed row the QK norm kernel reads.
        let column = |columns: u32| {
            DeviceBufferSpanV1::new(
                s.attn_qkv.buffer,
                s.attn_qkv.offset + u64::from(columns) * BF16_BYTES,
                s.attn_qkv.length - u64::from(columns) * BF16_BYTES,
            )
        };
        let [q, v_start] = packed_qkv_offsets(self.shape);
        if self.decode_entries().is_some() {
            return self.dense(
                Projection::AttentionIn,
                self.rows,
                &[
                    &weight("q_proj"),
                    &weight("k_proj"),
                    &weight("v_proj"),
                    &weight("indexer.index_qk_proj"),
                ],
                s.u,
                &[s.attn_qkv, column(q), column(v_start), s.attn_index],
            );
        }
        // The placement stores q_proj, k_proj and v_proj back to back: one
        // [attn_row, HIDDEN] fragment view writes the packed row.
        let mut launches = self.dense_group(
            self.own_ops()?.dense.group(Projection::AttentionQkv)?,
            Projection::AttentionQkv,
            self.rows,
            &[self.packed_qkv(layer)?],
            s.u,
            &[s.attn_qkv],
            None,
            true,
        )?;
        launches.extend(self.dense(
            Projection::IndexQk,
            self.rows,
            &[&weight("indexer.index_qk_proj")],
            s.u,
            &[s.attn_index],
        )?);
        Ok(launches)
    }

    fn pages(&self, layer: u32) -> Result<bundle_runtime::pages::LayerPlanes, Refusal> {
        let (offset, planes, _) = self.shape.page_layer(&self.kv, layer);
        bundle_runtime::pages::planes_at(self.kv_pool, self.kv.page_bytes, offset, &planes)
            .map_err(Refusal)
    }

    /// Whether full layer `layer` stores FP8 K and V, and its K and V scale
    /// bits.
    fn kv_storage(&self, layer: u32) -> (bool, [u32; 2]) {
        kv_storage_for(self.shape, &self.kv, layer)
    }

    /// The raw-key ring of full layer `layer`: keys, then RoPE positions.
    fn ring_layer(&self, layer: u32) -> (DeviceBufferSpanV1, DeviceBufferSpanV1) {
        let ring = self.ring_layout;
        let base = self.ring.offset + self.shape.ordinal(layer) * ring.layer_bytes();
        (
            DeviceBufferSpanV1::new(self.ring.buffer, base, ring.key_bytes()),
            DeviceBufferSpanV1::new(
                self.ring.buffer,
                base + ring.key_bytes(),
                ring.position_bytes(),
            ),
        )
    }

    /// Full layer `layer` of the anchor items' ring (`RankScratch::anchor_ring`).
    fn anchor_ring_layer(&self, layer: u32) -> (DeviceBufferSpanV1, DeviceBufferSpanV1) {
        let ring = Ring {
            shape: self.shape,
            layers: self.ring_layout.layers,
            slots: u64::from(ANCHOR_ITEMS),
        };
        let spare = self.scratch().anchor_ring;
        let base = spare.offset + self.shape.ordinal(layer) * ring.layer_bytes();
        (
            DeviceBufferSpanV1::new(spare.buffer, base, ring.key_bytes()),
            DeviceBufferSpanV1::new(spare.buffer, base + ring.key_bytes(), ring.position_bytes()),
        )
    }

    fn block_select(&self, layer: u32) -> Result<Vec<Lowered>, Refusal> {
        let s = self.scratch();
        let ops = self.attention_ops()?;
        let index_row = u64::from(self.shape.index_heads() * self.shape.index_dim()) * BF16_BYTES;
        let pieces = if self.decode() {
            vec![(0, self.rows, self.live_tokens())]
        } else {
            (0..PIECES)
                .map(|piece| piece * SCORE_ROWS)
                .take_while(|&first| first < self.rows)
                .map(|first| {
                    (
                        first,
                        (self.rows - first).min(SCORE_ROWS),
                        self.side.piece_live(first / SCORE_ROWS),
                    )
                })
                .collect()
        };
        let mut launches = Vec::new();
        for (first, rows, live) in pieces {
            let first = u64::from(first);
            let counts = self.attention_counts(rows);
            let mut bind = Bind::default();
            let scores = BlockScoresBuffers {
                q: bind.view(at(s.index_q, first, index_row)),
                block_keys: self.block_keys(&mut bind, layer)?,
                block_tables: bind.view(self.inputs.block_tables),
                token_batch: bind.view(at(self.inputs.token_batch, first, 4)),
                positions: bind.view(at(self.inputs.positions, first, 4)),
                live_rows: bind.view(live),
                scores: bind.view(s.scores),
            };
            launches.extend(bind.lower(&ops.scores, &scores, counts)?);
            let mut bind = Bind::default();
            let top_k = BlockTopKBuffers {
                scores: bind.view(s.scores),
                positions: bind.view(at(self.inputs.positions, first, 4)),
                live_rows: bind.view(live),
                selected: bind.view(at(
                    s.selected,
                    first,
                    u64::from(self.shape.top_blocks()) * 4,
                )),
            };
            launches.extend(bind.lower(&ops.top_k, &top_k, counts)?);
        }
        Ok(launches)
    }

    fn sparse_attention(&self, layer: u32) -> Result<Vec<Lowered>, Refusal> {
        let s = self.scratch();
        // Verify rows run the decode attention, one row each, as in BF16.
        let (fp8, _) = self.kv_storage(layer);
        let mut bind = Bind::default();
        let buffers = SparseAttentionBuffers {
            q: bind.view(s.attn_q),
            cache: self.cache(&mut bind, layer)?,
            block_tables: bind.view(self.inputs.block_tables),
            token_batch: bind.view(self.inputs.token_batch),
            positions: bind.view(self.inputs.positions),
            live_rows: bind.view(self.live_tokens()),
            selected: bind.view(s.selected),
            workspace: bind.view(s.workspace),
            output: bind.view(s.attn_out),
        };
        bind.lower(
            self.attention_ops()?.attention(fp8),
            &buffers,
            self.attention_counts(self.rows),
        )
    }

    fn moe_ops(&self) -> Result<&MoeOps, Refusal> {
        self.ops
            .ok_or(Refusal("Qwen3.8 ops are not prepared"))?
            .moe(Key::of(self.phase))
    }

    /// The rank's route metadata, bound.
    fn route_metadata(&self, bind: &mut Bind) -> RouteMetadata<View> {
        let s = self.scratch();
        RouteMetadata {
            counts: Some(bind.view(s.expert_counts)),
            offsets: bind.view(s.expert_offsets),
            sorted_rows: bind.view(s.sorted_rows),
            padded_rows: bind.view(s.padded_rows),
            active_experts: Some(bind.view(s.active_experts)),
        }
    }

    fn route(&self) -> Result<Vec<Lowered>, Refusal> {
        let s = self.scratch();
        // The route tile tables: counts, then offsets.
        let half = s.route_tiles.length / 2;
        let tile_counts = DeviceBufferSpanV1::new(s.route_tiles.buffer, s.route_tiles.offset, half);
        let tile_offsets =
            DeviceBufferSpanV1::new(s.route_tiles.buffer, s.route_tiles.offset + half, half);
        let mut bind = Bind::default();
        let buffers = RouteBuffers {
            logits: bind.view(s.router_logits),
            bias: None,
            ids: bind.view(s.topk_ids),
            weights: bind.view(s.topk_weights),
            routes: self.route_metadata(&mut bind),
            tile_counts: Some(bind.view(tile_counts)),
            tile_offsets: Some(bind.view(tile_offsets)),
            ticket: Some(bind.view(s.route_ticket)),
            live_rows: bind.view(self.live_tokens()),
        };
        bind.lower(self.moe_ops()?.route(self.rank)?, &buffers, self.counts())
    }

    fn experts(&self, layer: u32, projection: GatedMlpProjection) -> Result<Vec<Lowered>, Refusal> {
        let s = self.scratch();
        let ops = self.moe_ops()?;
        let (name, activation, output, op) = match projection {
            GatedMlpProjection::GateUp => ("gate_up_proj", s.u, s.moe_intermediate, &ops.gate_up),
            GatedMlpProjection::Down => ("down_proj", s.moe_intermediate, s.moe_routed, &ops.down),
        };
        let experts = format!(
            "{}.mlp.experts.{name}",
            model_qwen38_contract::views::layer_stem_for(layer, self.shape)
        );
        let mut bind = Bind::default();
        let buffers = GatedMlpBuffers {
            activation: bind.view(activation),
            activation_scales: None,
            activation_global: None,
            live_tokens: bind.view(self.live_tokens()),
            output: bind.view(output),
            weights: QuantizedWeights {
                codes: bind.view(self.view(&format!("{experts}.weight"))?),
                local_scales: Some(bind.view(self.view(&format!("{experts}.weight_scale_inv"))?)),
                global_scales: None,
            },
            routing: Some(ProjectionRouting {
                weights: bind.view(s.topk_weights),
                routes: self.route_metadata(&mut bind),
            }),
            scratch: GatedMlpScratch {
                quantizer_status: None,
                quantizer_status_row_offset: 0,
                partial: None,
                output_codes: None,
                output_scales: None,
                unused: bind.view(s.unused),
            },
        };
        bind.lower_experts(op.as_ref(), &buffers, self.counts())
    }

    fn collective_ops(&self) -> Result<&CollectiveOps, Refusal> {
        self.ops
            .ok_or(Refusal("Qwen3.8 ops are not prepared"))?
            .collectives(Key::of(self.phase))
    }

    /// The all-reduce after `(layer, sublayer)` into the next norm, on
    /// the state its key's form uses (`CollectiveOps::decode_state`).
    fn reduce(&self, layer: u32, sublayer: Sublayer) -> Result<Vec<Lowered>, Refusal> {
        let ops = self.collective_ops()?;
        let decode_state = ops.decode_state;
        let s = self.scratch();
        let from = residual_index_for(self.shape, layer, sublayer);
        let site = NormSite::Reduce(layer, sublayer);
        let mut bind = Bind::default();
        let ranks = self
            .peers
            .iter()
            .map(|peer| RankState {
                state: if decode_state {
                    [
                        peer.hc_decode_inbox,
                        peer.hc_decode_flags,
                        peer.hc_decode_epochs,
                    ]
                } else {
                    [peer.hc_inbox, peer.hc_flags, peer.hc_epoch]
                }
                .map(|span| bind.view(span))
                .to_vec(),
            })
            .collect();
        let buffers = HcAllReduceBuffers {
            input: bind.view(s.partial),
            residual: FusedResidual {
                input: bind.view(s.x[from]),
                output: bind.view(s.x[1 - from]),
            },
            logits: bind.view(s.hc_inject),
            weight: bind.view(self.view(&site.weight(self.shape))?),
            normed: bind.view(s.xn),
            live_rows: bind.view(self.live_tokens()),
            ranks,
        };
        bind.lower(
            ops.reduce(self.rank, self.norm_dtypes.f32(site))?,
            &buffers,
            self.counts(),
        )
    }
}

/// The pages of a layer's planes.
fn page_slots(pages: &bundle_runtime::pages::LayerPlanes) -> Result<u32, Refusal> {
    u32::try_from(pages.page_capacity).map_err(|_| Refusal("Qwen3.8 KV pool has too many pages"))
}

fn packed_qkv_offsets(shape: Shape) -> [u32; 2] {
    let k = shape.q_heads * 2 * shape.attn_head();
    [k, k + shape.kv_heads() * shape.attn_head()]
}

fn kv_storage_for(shape: Shape, kv: &bundle_runtime::kv::KvFormat, layer: u32) -> (bool, [u32; 2]) {
    let (_, _, fp8) = shape.page_layer(kv, layer);
    let scales = if fp8 {
        kv.scale_bits(shape.ordinal(layer))
    } else {
        [1.0_f32.to_bits(); 2]
    };
    (fp8, scales)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packed_v_starts_after_every_local_kv_head() {
        static OTHER: model_qwen38_contract::geometry::Geometry = {
            let mut g = model_qwen38_contract::geometry::FLASH;
            g.shape.attention.kv_heads = 8;
            g
        };
        for ranks in [Ranks::TP2, Ranks::TP4] {
            let shape = Shape::of_geometry(&OTHER, ranks).unwrap();
            let [k, v] = packed_qkv_offsets(shape);
            assert_eq!(k, shape.q_heads * 2 * shape.attn_head());
            assert_eq!(v - k, (8 / ranks.get()) * shape.attn_head());
            assert_eq!(shape.attn_row - v, (8 / ranks.get()) * shape.attn_head());
        }
    }

    #[test]
    fn fp8_scales_follow_the_geometry_full_layer_ordinals() {
        use model_qwen38_contract::geometry::{FLASH, Geometry, LayerKind};
        static OTHER: Geometry = {
            let mut g = FLASH;
            g.shape.layers = &[LayerKind::Full, LayerKind::Linear, LayerKind::Full];
            g
        };
        let shape = Shape::new(&OTHER, Ranks::TP4);
        let mut kv = shape.kv_formats()[2];
        kv.scales = &[[0.25, 0.5], [0.125, 0.0625]];
        for (layer, scales) in [(0, kv.scales[0]), (2, kv.scales[1])] {
            assert_eq!(
                kv_storage_for(shape, &kv, layer),
                (true, scales.map(f32::to_bits))
            );
        }
        let mtp = shape.kv_formats()[3];
        assert_eq!(
            kv_storage_for(shape, &mtp, shape.layers()),
            (false, [1.0_f32.to_bits(); 2])
        );
    }

    #[test]
    fn residual_buffers_alternate_through_reduces_and_ple() {
        assert_eq!(residual_index(0, Sublayer::Attention), 0);
        assert_eq!(residual_index(0, Sublayer::Mlp), 1);
        // Layer 0 ends back in x0; PLE writes x1, so layer 1 starts there.
        assert_eq!(PLE_RESIDUAL_INDEX, 0);
        assert_eq!(residual_index(1, Sublayer::Attention), 1);
        assert_eq!(residual_index(1, Sublayer::Mlp), 0);
        assert_eq!(residual_index(47, Sublayer::Mlp), 0);
        assert_eq!(
            next_norm(0, Sublayer::Mlp),
            "model.language_model.layers.1.ple.norm_query.weight"
        );
        assert_eq!(
            next_norm(47, Sublayer::Mlp),
            "model.language_model.hyper_connection_mixer.hc_norm.weight"
        );
        assert_eq!(
            next_norm(3, Sublayer::Attention),
            "model.language_model.layers.3.mlp_hyper_connection.hc_norm.weight"
        );
    }
}
