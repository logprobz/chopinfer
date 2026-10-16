//! The ops an instance prepares once through `kernel_core::backend`, and
//! their launches over a step's buffers by role.
//!
//! Qwen3.8 states each op as a kernel-core contract and the exactness it
//! needs (`Need`); the backend of the instance's target (`crate::target`)
//! prepares it when the instance is created. A step then binds its spans to
//! the op's roles and calls `PreparedOp::launches`: no catalog search or
//! entry choice happens per step.

use kernel_core::backend::{Backend, Counts, Launch, Need, Prepared, PreparedOp, Refused, View};
use kernel_core::collective::{GatherBuffers, HcAllReduceBuffers, RowGatherBuffers, ShardedRows};
use kernel_core::context_parallel::{
    ContextParallelQuery, GroupTransfer, GroupTransferBuffers, GroupTransferOp,
    LocalizeKeysBuffers, OwnedKeyWriteOp, OwnedKvWriteOp, PartialMergeBuffers,
};
use kernel_core::contract::{
    AttentionPhase, ConvActivation, DType, GateActivation, GatedAttentionQuery, MoeQuery,
    NgramQuery, RecurrentPhase, RecurrentQuery, Refusal, RouterKind, RowQuery,
    SigmoidMeanHyperConnectionQuery as Hc, SparseAttentionQuery,
};
use kernel_core::gated_mlp::GatedMlpProjection;
use kernel_core::gated_sparse::{
    BlockScoresBuffers, BlockScoresOp, BlockTopKBuffers, GatedKvWriteBuffers, GatedKvWriteOp,
    IndexQueryBuffers, OutputGateBuffers, OutputGateOp, SelectedPartialsBuffers,
    SelectedPartialsOp, SparseAttentionBuffers, SparseAttentionOp, SparseKeyWriteBuffers,
    SparseKeyWriteOp,
};
use kernel_core::moe::{RouteBuffers, SharedCombineBuffers};
use kernel_core::moe_experts::{GatedMlpBuffers, PreparedExperts};
use kernel_core::pooled_indexer::{
    IndexerMergeBuffers, IndexerSelectBuffers, IndexerSelectionMode, PooledIndexerQuery,
};
use kernel_core::precision::{ExecutionPhase, OperationUse};
use kernel_core::properties::{Anchors, Chunk, Invariance, Rows};
use kernel_core::recurrent::{ConvBuffers, DeltaRuleBuffers, GatedNormBuffers};
use kernel_core::row_copy::{RowCopyBuffers, StreamBroadcast};
use kernel_core::rows::{
    BottleneckBuffers, EmbeddingBuffers, GroupedNormBuffers, MixBuffers, PleGateBuffers,
};
use model_qwen38_contract::own_ops::copies::{
    CopyEntries, CopyEntriesBuffers, PrefixCopy, PrefixCopyBuffers,
};
use model_qwen38_contract::own_ops::dense::{DenseGroup, DenseGroupBuffers, Projection};
use model_qwen38_contract::own_ops::hc::{HcUpMix, HcUpMixBuffers, UpMixInput};
use model_qwen38_contract::own_ops::mtp::{
    DraftArgmaxFinal, DraftArgmaxFinalBuffers, DraftArgmaxPartial, DraftArgmaxPartialBuffers,
    MtpCombine, MtpCombineBuffers, MtpGather, MtpGatherBuffers, MtpNorms, MtpNormsBuffers,
};
use model_qwen38_contract::own_ops::ple::{
    PleAnchor, PleAnchorBuffers, PleCheckpoints, PleCheckpointsBuffers, PleConv, PleConvBuffers,
    PleLayout, PleResidual, PleResidualBuffers, PleStage, PleStageBuffers,
};
use model_qwen38_contract::own_ops::{ItemForm, OwnOps};
use runtime_bundle_abi::DeviceBufferSpanV1;

use crate::contract::*;
use crate::dispatch::BufferBindings;
use crate::pipeline::PreparedLaunch;
use crate::program::Phase;
use crate::scratch::{CommBytes, Served};
use crate::target::Target;
use bundle_runtime::lanes::LaunchFacts as Facts;

/// Where a step's rows run: the prefill op, or a graph bucket's op of
/// decode or verify rows.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Key {
    Prefill,
    Decode(u32),
    Verify(u32),
}

impl Key {
    pub(crate) fn of(phase: Phase) -> Key {
        match phase {
            Phase::Prefill => Key::Prefill,
            Phase::Decode(rows) => Key::Decode(rows),
            Phase::Verify { rows, .. } => Key::Verify(rows),
        }
    }

    /// The key of a step of use `usage`.
    pub(crate) fn of_use(usage: OperationUse) -> Key {
        match usage.phase {
            ExecutionPhase::Prefill => Key::Prefill,
            ExecutionPhase::Decode => Key::Decode(usage.capacity),
            ExecutionPhase::Verify => Key::Verify(usage.capacity),
        }
    }

    /// The most rows a step of this key runs.
    pub(crate) fn capacity(self) -> u32 {
        match self {
            Key::Prefill => PREFILL_TOKENS,
            Key::Decode(rows) | Key::Verify(rows) => rows,
        }
    }
}

/// What a step of `key` needs of an op whose launches record `rows` rows:
/// prefill rows keep their bits in any chunk split at the manifest's
/// granule, and an op that writes per-slot state there also keeps it at
/// in-step anchors (`anchors`, the kind the program relies on); the rows
/// of an exact verify step (`EXACT_VERIFY`) have their one-row bits. Other
/// steps need nothing. This is what `kernels_sm120::invariance::Program`
/// checks of every launch.
pub(crate) fn exact(key: Key, rows: u32, anchors: Anchors) -> Invariance {
    match key {
        Key::Prefill => Invariance {
            chunk: Chunk::Granule(PREFILL_GRANULE),
            anchors,
            ..Invariance::NONE
        },
        Key::Verify(bucket) if bucket <= EXACT_VERIFY.rows => Invariance {
            rows: Rows::OneRow { max: rows },
            ..Invariance::NONE
        },
        Key::Decode(_) | Key::Verify(_) => Invariance::NONE,
    }
}

/// The anchors a key write asks for: its spare ring holds only for anchors
/// past a ring of rows, and the program's anchors sit on multiples of the
/// prefill granule in launches that start on one, so they lie at least the
/// granule's rows after their launch's first row.
pub(crate) const ANCHORED_RING: Anchors = Anchors::SpareAfter {
    rows: PREFILL_GRANULE,
};

/// The need of a row op of `key` that records the step's rows.
pub(crate) fn rows_need(key: Key) -> Need {
    Need::rows(key.capacity()).with_exact(exact(key, key.capacity(), Anchors::NotApplicable))
}

/// The need of an op that serves the uses of both `a` and `b` (a decode
/// and a verify use of one bucket share one projection): the one that
/// provides the other's exactness. Needs of different capacities, or with
/// claims neither provides, have none.
pub(crate) fn strongest(a: Need, b: Need) -> Result<Need, Refused> {
    if a.capacity == b.capacity && a.exact.provides(b.exact) {
        Ok(a)
    } else if a.capacity == b.capacity && b.exact.provides(a.exact) {
        Ok(b)
    } else {
        Err(Refused::Rule(format!("no one need serves {a:?} and {b:?}")))
    }
}

/// The graph buckets an instance of `served` prepares ops for: every power
/// of two up to its widest graph, which holds the drafter's buckets too.
pub(crate) fn buckets(served: Served) -> impl Iterator<Item = u32> {
    let widest = served.rows.max(DRAFT_ITEMS).next_power_of_two();
    (0..=widest.trailing_zeros()).map(|shift| 1 << shift)
}

/// One op (or set of ops) per key: prefill, and each decode and verify
/// graph bucket, indexed by the bucket's power of two.
pub(crate) struct ByKey<T> {
    prefill: T,
    decode: Vec<T>,
    verify: Vec<T>,
}

impl<T> ByKey<T> {
    fn prepare(
        served: Served,
        mut prepare: impl FnMut(Key) -> Result<T, Refused>,
    ) -> Result<Self, Refused> {
        let decode = buckets(served)
            .map(|bucket| prepare(Key::Decode(bucket)))
            .collect::<Result<Vec<_>, _>>()?;
        let verify = buckets(served)
            .map(|bucket| prepare(Key::Verify(bucket)))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            prefill: prepare(Key::Prefill)?,
            decode,
            verify,
        })
    }

    pub(crate) fn get(&self, key: Key) -> Result<&T, Refusal> {
        fn bucket<T>(ops: &[T], rows: u32) -> Option<&T> {
            rows.is_power_of_two()
                .then(|| ops.get(rows.trailing_zeros() as usize))
                .flatten()
        }
        match key {
            Key::Prefill => Some(&self.prefill),
            Key::Decode(rows) => bucket(&self.decode, rows),
            Key::Verify(rows) => bucket(&self.verify, rows),
        }
        .ok_or(Refusal("Qwen3.8 has no prepared op for the phase"))
    }
}

/// A lowered launch and the facts the pipelined prefill adjusts its grid
/// by (`Target::adjust`): those of the op that
/// made it (`LaunchFacts::of_op`), or, for a launch Qwen built, those of
/// its symbol (`Target::built`).
pub(crate) struct Lowered {
    pub launch: PreparedLaunch,
    pub facts: Facts,
}

/// The launches of `lowered`, without their facts.
pub(crate) fn launches(lowered: Vec<Lowered>) -> Vec<PreparedLaunch> {
    lowered.into_iter().map(|lowered| lowered.launch).collect()
}

/// The program of `lowered`, with which of its launches are collectives.
pub(crate) fn program(lowered: Vec<Lowered>) -> crate::pipeline::Program {
    crate::pipeline::Program::of(
        lowered
            .into_iter()
            .map(|lowered| (lowered.launch, lowered.facts)),
    )
}

/// The launches of `lowered` and their facts.
pub(crate) fn unzip(lowered: Vec<Lowered>) -> (Vec<PreparedLaunch>, Vec<Facts>) {
    lowered
        .into_iter()
        .map(|lowered| (lowered.launch, lowered.facts))
        .unzip()
}

/// [`unzip`]'s inverse.
pub(crate) fn zip(launches: Vec<PreparedLaunch>, facts: Vec<Facts>) -> Vec<Lowered> {
    launches
        .into_iter()
        .zip(facts)
        .map(|(launch, facts)| Lowered { launch, facts })
        .collect()
}

/// Spans bound for one stage's launches, as role views.
#[derive(Default)]
pub(crate) struct Bind(BufferBindings);

impl Bind {
    pub(crate) fn view(&mut self, span: DeviceBufferSpanV1) -> View {
        View {
            handle: self.0.bind(span),
            bytes: span.length,
        }
    }

    /// `op`'s launches over `buffers` as the pipeline's launches, every one
    /// with these bindings.
    pub(crate) fn launch<B>(
        self,
        op: &Prepared<B>,
        buffers: &B,
        counts: Counts,
    ) -> Result<Vec<PreparedLaunch>, Refusal> {
        Ok(launches(self.lower(op, buffers, counts)?))
    }

    /// [`Bind::launch`], each launch with the facts of the op.
    pub(crate) fn lower<B>(
        self,
        op: &Prepared<B>,
        buffers: &B,
        counts: Counts,
    ) -> Result<Vec<Lowered>, Refusal> {
        self.lower_op(op.as_ref(), buffers, counts)
    }

    /// An expert projection's launches, as [`Bind::launch`]'s.
    pub(crate) fn launch_experts(
        self,
        op: &dyn PreparedExperts,
        buffers: &GatedMlpBuffers<View>,
        counts: Counts,
    ) -> Result<Vec<PreparedLaunch>, Refusal> {
        Ok(launches(self.lower_experts(op, buffers, counts)?))
    }

    /// An expert projection's launches, as [`Bind::lower`]'s.
    pub(crate) fn lower_experts(
        self,
        op: &dyn PreparedExperts,
        buffers: &GatedMlpBuffers<View>,
        counts: Counts,
    ) -> Result<Vec<Lowered>, Refusal> {
        self.lower_op::<GatedMlpBuffers<View>>(op, buffers, counts)
    }

    fn lower_op<B>(
        self,
        op: &dyn PreparedOp<B>,
        buffers: &B,
        counts: Counts,
    ) -> Result<Vec<Lowered>, Refusal> {
        if op.info().workspace != 0 {
            return Err(Refusal("Qwen3.8 binds no backend workspace"));
        }
        let launches = op.launches(buffers, View::EMPTY, counts)?;
        let facts = Facts::of_op(&launches);
        Ok(launches
            .into_iter()
            .zip(facts)
            .map(|(launch, facts)| Lowered {
                launch: self.prepared(launch),
                facts,
            })
            .collect())
    }

    fn prepared(&self, launch: Launch) -> PreparedLaunch {
        PreparedLaunch {
            symbol: launch.symbol,
            rows: launch.rows,
            plan: launch.plan,
            arguments: launch.arguments,
            bindings: self.0.clone(),
        }
    }
}

/// The hyper-connection queries of `shape`: one launch serves up to a
/// prefill chunk of rows.
fn hc(shape: Shape) -> [Hc; 3] {
    let (streams, width, capacity) = (shape.streams(), shape.hidden(), PREFILL_TOKENS);
    [
        Hc::GroupedNorm {
            streams,
            width,
            eps_bits: shape.eps().to_bits(),
            capacity,
        },
        Hc::Bottleneck {
            streams,
            rank: shape.hc_rank(),
            capacity,
        },
        Hc::Mix {
            streams,
            width,
            capacity,
        },
    ]
}

/// The linear attention's queries of `phase` over `shape`'s rank heads, of
/// up to `capacity` rows: its convolution and its gated delta rule.
fn linear(shape: Shape, phase: RecurrentPhase, capacity: u32) -> [RecurrentQuery; 2] {
    [
        RecurrentQuery::CausalConv1d {
            channels: shape.linear_qkv,
            kernel: shape.conv_taps(),
            dilation: 1,
            activation: ConvActivation::Silu,
            phase,
            capacity,
        },
        RecurrentQuery::GatedDeltaRule {
            key_heads: shape.key_heads,
            value_heads: shape.value_heads,
            key_dim: shape.linear_head(),
            value_dim: shape.linear_head(),
            phase,
            capacity,
        },
    ]
}

/// The rank-independent ops of one key.
pub(crate) struct StageOps {
    /// The grouped norm over BF16 weights, and over F32 weights where the
    /// image stores them.
    pub norm: Prepared<GroupedNormBuffers<View>>,
    pub norm_f32: Option<Prepared<GroupedNormBuffers<View>>>,
    pub bottleneck: Prepared<BottleneckBuffers<View>>,
    pub mix: Prepared<MixBuffers<View>>,
    pub ple_gate: Prepared<PleGateBuffers<View>>,
    /// The embedding row, gathering the rows the head projects.
    pub head_rows: Prepared<EmbeddingBuffers<View>>,
    pub conv: Prepared<ConvBuffers<View>>,
    pub delta: Prepared<DeltaRuleBuffers<View>>,
    pub gated_norm: Prepared<GatedNormBuffers<View>>,
    /// Each embedding row copied to every stream of X.
    pub expand_streams: Prepared<RowCopyBuffers<View>>,
}

impl StageOps {
    fn prepare(
        backend: &dyn Backend,
        shape: Shape,
        served: Served,
        f32_norms: bool,
        key: Key,
    ) -> Result<StageOps, Refused> {
        let need = rows_need(key);
        let [norm, bottleneck, mix] = hc(shape);
        let streams = shape.streams();
        // Prefill keeps state at anchors in a spare entry; graph steps run
        // the verify entries, whose rows are one-row exact up to the
        // verify rows of an item.
        let (phase, capacity, items) = match key {
            Key::Prefill => (
                RecurrentPhase::Prefill,
                PREFILL_TOKENS,
                crate::launch::prefill_sequences(served),
            ),
            Key::Decode(_) | Key::Verify(_) => (
                RecurrentPhase::Verify {
                    max_rows: MAX_VERIFY_ROWS,
                },
                DECODE_BATCH * MAX_VERIFY_ROWS,
                key.capacity(),
            ),
        };
        let state_need =
            Need::rows(key.capacity()).with_exact(exact(key, key.capacity(), Anchors::Spare));
        let [conv, delta] = linear(shape, phase, capacity);
        Ok(StageOps {
            expand_streams: backend.broadcast_streams(
                &StreamBroadcast {
                    dtype: DType::Bf16,
                    width: shape.hidden(),
                    streams,
                    alignment_bytes: 16,
                },
                need,
            )?,
            norm: backend.hc_grouped_norm(&norm, kernel_core::contract::DType::Bf16, need)?,
            // The F32 forms declare no exactness, so they are asked for none.
            norm_f32: f32_norms
                .then(|| {
                    backend.hc_grouped_norm(
                        &norm,
                        kernel_core::contract::DType::F32,
                        Need::rows(key.capacity()),
                    )
                })
                .transpose()?,
            bottleneck: backend.hc_bottleneck(&bottleneck, need)?,
            mix: backend.hc_mix(&mix, need)?,
            ple_gate: backend.ple_gate(
                &NgramQuery::PleGate {
                    streams,
                    width: shape.hidden(),
                    capacity: PREFILL_TOKENS,
                },
                need,
            )?,
            head_rows: backend.embedding_gather(
                &RowQuery::EmbeddingGather {
                    vocab: shape.geometry.vocab(),
                    width: shape.hidden(),
                },
                0,
                need,
            )?,
            conv: backend.causal_conv1d(&conv, items, state_need)?,
            delta: backend.gated_delta_rule(&delta, items, state_need)?,
            gated_norm: backend.gated_rmsnorm(
                &RecurrentQuery::GatedRmsNorm {
                    width: shape.linear_value,
                    head_dim: shape.linear_head(),
                    eps_bits: shape.eps().to_bits(),
                    gate: GateActivation::Sigmoid,
                    capacity: PREFILL_TOKENS,
                },
                need,
            )?,
        })
    }
}

/// The gated and sparse attention of one key: the QK norm, M-RoPE and KV
/// write, the indexer's query, key write, block scores and top-k, the
/// sparse attention and the output gate. The KV write and the attention
/// come for BF16 and FP8 caches, since an FP8 instance keeps the MTP
/// layer's cache in BF16.
pub(crate) struct AttentionOps {
    kv_write: [Prepared<GatedKvWriteBuffers<View>>; 2],
    pub index_query: Prepared<IndexQueryBuffers<View>>,
    pub key_write: Prepared<SparseKeyWriteBuffers<View>>,
    /// Graph steps score every row at once; prefill scores pieces of up to
    /// `SCORE_ROWS` rows, so its scores and top-k hold a piece.
    pub scores: Prepared<BlockScoresBuffers<View>>,
    pub top_k: Prepared<BlockTopKBuffers<View>>,
    attention: [Prepared<SparseAttentionBuffers<View>>; 2],
    pub output_gate: Prepared<OutputGateBuffers<View>>,
}

impl AttentionOps {
    fn prepare(backend: &dyn Backend, shape: Shape, key: Key) -> Result<AttentionOps, Refused> {
        let need = rows_need(key);
        let capacity = key.capacity();
        let max_context = shape.context_tokens() as u32;
        let (theta_bits, mrope, eps_bits) = (
            shape.rope_theta().to_bits(),
            shape.geometry.mrope(),
            shape.eps().to_bits(),
        );
        let (block, page_size) = (shape.qsa_block(), PAGE_TOKENS as u32);
        let max_blocks = max_context / block;
        // Graph steps (decode and verify rows) run as decode rows.
        let phase = match key {
            Key::Prefill => AttentionPhase::Prefill,
            Key::Decode(_) | Key::Verify(_) => AttentionPhase::Decode,
        };
        let pieces = match key {
            Key::Prefill => SCORE_ROWS,
            Key::Decode(_) | Key::Verify(_) => capacity,
        };
        let piece_need = Need::rows(pieces).with_exact(exact(key, pieces, Anchors::NotApplicable));
        let kv_write = |kv_dtype| {
            backend.gated_kv_write(
                &GatedKvWriteOp {
                    query: GatedAttentionQuery::QkNormRopeKvWrite {
                        q_heads: shape.q_heads,
                        kv_heads: shape.kv_heads(),
                        head_dim: shape.attn_head(),
                        rotary_dim: shape.rotary(),
                        theta_bits,
                        mrope,
                        eps_bits,
                        page_size,
                        kv_dtype,
                        capacity,
                    },
                    max_context,
                },
                need,
            )
        };
        let attention = |kv_dtype| {
            backend.sparse_attention(
                &SparseAttentionOp {
                    query: SparseAttentionQuery::SparseAttention {
                        q_heads: shape.q_heads,
                        kv_heads: shape.kv_heads(),
                        head_dim: shape.attn_head(),
                        block,
                        top_blocks: shape.top_blocks(),
                        softmax_scale_bits: shape.softmax_scale().to_bits(),
                        page_size,
                        kv_dtype,
                        phase,
                        capacity,
                    },
                    max_context,
                },
                need,
            )
        };
        Ok(AttentionOps {
            kv_write: [kv_write(DType::Bf16)?, kv_write(DType::Fp8E4M3)?],
            index_query: backend.sparse_index_query(
                &SparseAttentionQuery::IndexerQueryGemmaNormMrope {
                    heads: shape.index_heads(),
                    head_dim: shape.index_dim(),
                    rotary_dim: shape.rotary(),
                    theta_bits,
                    mrope,
                    eps_bits,
                    capacity,
                },
                need,
            )?,
            // Prefill keeps each item's ring at its anchor in a spare slot.
            key_write: backend.sparse_key_write(
                &SparseKeyWriteOp {
                    query: SparseAttentionQuery::IndexerKeyWrite {
                        head_dim: shape.index_dim(),
                        rotary_dim: shape.rotary(),
                        theta_bits,
                        mrope,
                        eps_bits,
                        block,
                        ring: shape.qsa_ring(),
                        capacity,
                    },
                    index_heads: shape.index_heads(),
                    page_size,
                    max_context,
                },
                // Its spare ring is taken from the launch's rows, so it
                // holds only past a ring of rows: anchors sit on multiples
                // of the prefill granule in launches that start on one.
                Need::rows(capacity).with_exact(exact(key, capacity, ANCHORED_RING)),
            )?,
            scores: backend.sparse_block_scores(
                &BlockScoresOp {
                    query: SparseAttentionQuery::BlockScores {
                        heads: shape.index_heads(),
                        head_dim: shape.index_dim(),
                        max_blocks,
                        capacity: pieces,
                    },
                    block,
                    page_size,
                    max_context,
                },
                OperationUse {
                    capacity: pieces,
                    ..usage(key)
                },
                piece_need,
            )?,
            top_k: backend.sparse_block_top_k(
                &SparseAttentionQuery::BlockTopK {
                    block,
                    top_blocks: shape.top_blocks(),
                    max_blocks,
                    capacity: pieces,
                },
                piece_need,
            )?,
            attention: [attention(DType::Bf16)?, attention(DType::Fp8E4M3)?],
            output_gate: backend.output_gate(
                &OutputGateOp {
                    query: GatedAttentionQuery::OutputGate {
                        heads: shape.q_heads,
                        head_dim: shape.attn_head(),
                        capacity,
                    },
                    kv_heads: shape.kv_heads(),
                },
                need,
            )?,
        })
    }

    /// The KV write into a BF16 or an FP8 cache.
    pub(crate) fn kv_write(&self, fp8: bool) -> &Prepared<GatedKvWriteBuffers<View>> {
        &self.kv_write[usize::from(fp8)]
    }

    /// The attention over a BF16 or an FP8 cache.
    pub(crate) fn attention(&self, fp8: bool) -> &Prepared<SparseAttentionBuffers<View>> {
        &self.attention[usize::from(fp8)]
    }
}

/// How a step of `key` runs its rows, for ops whose backend may choose by
/// it.
pub(crate) fn usage(key: Key) -> OperationUse {
    let phase = match key {
        Key::Prefill => ExecutionPhase::Prefill,
        Key::Decode(_) => ExecutionPhase::Decode,
        Key::Verify(_) => ExecutionPhase::Verify,
    };
    OperationUse {
        phase,
        capacity: key.capacity(),
        graph: key != Key::Prefill,
    }
}

/// The MoE ops of one key: the routed experts' gate/up and down
/// projections, which every rank runs over its own experts' weights, and
/// each rank's route and shared-expert combine, indexed by rank.
pub(crate) struct MoeOps {
    pub gate_up: Box<dyn PreparedExperts>,
    pub down: Box<dyn PreparedExperts>,
    routes: Vec<Prepared<RouteBuffers<View>>>,
    combines: Vec<Prepared<SharedCombineBuffers<View>>>,
}

impl MoeOps {
    fn prepare(
        backend: &dyn Backend,
        target: &dyn Target,
        shape: Shape,
        key: Key,
    ) -> Result<MoeOps, Refused> {
        let need = rows_need(key);
        let capacity = key.capacity();
        let experts = |projection| {
            backend.gated_mlp_projection(
                &target.expert_contract(projection, key)?,
                usage(key),
                need,
            )
        };
        let gate_up = experts(GatedMlpProjection::GateUp)?;
        let down = experts(GatedMlpProjection::Down)?;
        // One route serves both projections, so they read routes in one
        // tile.
        let tile = gate_up.route_tile();
        if down.route_tile() != tile {
            return Err(Refused::Rule(
                "Qwen3.8's expert projections read routes in different tiles".into(),
            ));
        }
        let router = RowQuery::Router {
            experts: shape.experts(),
            top_k: shape.top_k(),
            kind: RouterKind::SoftmaxTopK { renormalize: true },
        };
        let routes = MoeQuery::LocalRoutes {
            experts: shape.experts(),
            local_experts: shape.experts() / shape.ranks(),
            top_k: shape.top_k(),
            capacity,
        };
        // A route's metadata depends on every route of the launch: its rows
        // keep their bits as the routed operation's, at every key.
        let route_need = Need::rows(capacity)
            .with_exact(exact(key, capacity, Anchors::NotApplicable).of_routed_operation());
        let combine = MoeQuery::SharedExpertGateCombine {
            width: shape.hidden(),
            capacity,
        };
        Ok(MoeOps {
            routes: (0..shape.ranks())
                .map(|rank| backend.moe_route(&router, &routes, rank, tile, usage(key), route_need))
                .collect::<Result<_, _>>()?,
            combines: (0..shape.ranks())
                .map(|rank| backend.moe_shared_expert_combine(&combine, &routes, rank, need))
                .collect::<Result<_, _>>()?,
            gate_up,
            down,
        })
    }

    /// Rank `rank`'s route.
    pub(crate) fn route(&self, rank: u32) -> Result<&Prepared<RouteBuffers<View>>, Refusal> {
        self.routes
            .get(rank as usize)
            .ok_or(Refusal("Qwen3.8 has no route for the rank"))
    }

    /// Rank `rank`'s shared-expert combine.
    pub(crate) fn combine(
        &self,
        rank: u32,
    ) -> Result<&Prepared<SharedCombineBuffers<View>>, Refusal> {
        self.combines
            .get(rank as usize)
            .ok_or(Refusal("Qwen3.8 has no combine for the rank"))
    }
}

/// Element `index` of per-rank ops.
fn of_rank<T>(ops: &[T], index: u32) -> Result<&T, Refusal> {
    ops.get(index as usize)
        .ok_or(Refusal("Qwen3.8 has no collective for the rank"))
}

/// The collectives of one key, each rank's in rank order: the
/// hyper-connection all-reduce into the next norm over BF16 weights (and
/// over F32 weights where the image stores them), the row gathers of the
/// n-gram rows and of the drafts onto every rank and of the logits onto
/// rank 0, and the logits row exchange.
pub(crate) struct CollectiveOps {
    /// The all-reduces run on the decode state: the prepared ops state other
    /// regions than the general state's (the backend's decode or wide
    /// entries), which the decode state holds ([`state_binding`]).
    pub decode_state: bool,
    reduce: Vec<Prepared<HcAllReduceBuffers<View>>>,
    reduce_f32: Vec<Prepared<HcAllReduceBuffers<View>>>,
    /// By rank, then target.
    ngram: Vec<Vec<Prepared<RowGatherBuffers<View>>>>,
    drafts: Vec<Vec<Prepared<RowGatherBuffers<View>>>>,
    logits: Vec<Prepared<RowGatherBuffers<View>>>,
    exchange: Vec<Prepared<GatherBuffers<View>>>,
}

/// The hyper-connection all-reduce into the next norm. One general state
/// holds a prefill chunk of rows.
pub(crate) fn hc_reduce(shape: Shape) -> Hc {
    Hc::AllReduceInjectNorm {
        ranks: shape.ranks(),
        streams: shape.streams(),
        width: shape.hidden(),
        eps_bits: shape.eps().to_bits(),
        capacity: PREFILL_TOKENS,
    }
}

/// Whether the prepared all-reduces `ops` run on the decode state of `comm`:
/// ops that state the general state's regions run on it, any others on the
/// decode state, which must hold their regions. Every op runs on one of them.
pub(crate) fn state_binding<'a>(
    ops: impl IntoIterator<Item = &'a Prepared<HcAllReduceBuffers<View>>>,
    comm: &CommBytes,
) -> Result<bool, Refused> {
    let general = [comm.hc_inbox, comm.hc_flags, comm.hc_epoch];
    let decode = [comm.decode_inbox, comm.decode_flags, comm.decode_epochs];
    let mut binding = None;
    for op in ops {
        let state = op.info().state.as_slice();
        let decode_state = if state == general {
            false
        } else if state.len() == decode.len() && state.iter().zip(decode).all(|(&s, d)| s <= d) {
            true
        } else {
            return Err(Refused::Rule(format!(
                "Qwen3.8 keeps no all-reduce state of {state:?} bytes"
            )));
        };
        if binding.is_some_and(|held| held != decode_state) {
            return Err(Refused::Rule(
                "Qwen3.8's all-reduces of one key run on different states".into(),
            ));
        }
        binding = Some(decode_state);
    }
    binding.ok_or_else(|| Refused::Rule("Qwen3.8 prepared no all-reduce".into()))
}

impl CollectiveOps {
    fn prepare(
        backend: &dyn Backend,
        comm: &CommBytes,
        shape: Shape,
        f32_norms: bool,
        key: Key,
    ) -> Result<CollectiveOps, Refused> {
        let (ranks, capacity) = (shape.ranks(), key.capacity());
        let reduce = hc_reduce(shape);
        let reduces = |weight, need| {
            (0..ranks)
                .map(|rank| {
                    backend.hc_all_reduce(&reduce, DType::Bf16, weight, usage(key), rank, need)
                })
                .collect::<Result<Vec<_>, _>>()
        };
        // Row moves copy bits, so they need nothing.
        let gathers = |shard, targets: std::ops::Range<u32>| {
            let op = ShardedRows {
                ranks,
                shard,
                dtype: DType::Bf16,
            };
            (0..ranks)
                .map(|rank| {
                    targets
                        .clone()
                        .map(|target| backend.gather_rows(&op, rank, target, Need::rows(capacity)))
                        .collect::<Result<Vec<_>, _>>()
                })
                .collect::<Result<Vec<_>, _>>()
        };
        let vocab = u64::from(shape.vocab_per_rank);
        let exchange = ShardedRows {
            ranks,
            shard: vocab,
            dtype: DType::F32,
        };
        let reduce = reduces(DType::Bf16, rows_need(key))?;
        // The F32 forms declare no exactness, so they are asked for none.
        let reduce_f32 = if f32_norms {
            reduces(DType::F32, Need::rows(capacity))?
        } else {
            Vec::new()
        };
        Ok(CollectiveOps {
            decode_state: state_binding(reduce.iter().chain(&reduce_f32), comm)?,
            reduce,
            reduce_f32,
            ngram: gathers(u64::from(shape.ngram_local), 0..ranks)?,
            // Each draft row's argmax pairs.
            drafts: gathers(8, 0..ranks)?,
            // F32 logits move as pairs of BF16 values.
            logits: gathers(2 * vocab, 0..1)?
                .into_iter()
                .map(|mut targets| targets.remove(0))
                .collect(),
            exchange: (0..ranks)
                .map(|rank| backend.exchange_rows(&exchange, rank, Need::rows(capacity)))
                .collect::<Result<Vec<_>, _>>()?,
        })
    }

    /// Rank `rank`'s all-reduce into a norm whose weights are F32 or BF16.
    pub(crate) fn reduce(
        &self,
        rank: u32,
        f32: bool,
    ) -> Result<&Prepared<HcAllReduceBuffers<View>>, Refusal> {
        of_rank(if f32 { &self.reduce_f32 } else { &self.reduce }, rank)
    }

    /// Rank `rank`'s gather of its n-gram rows onto `target`.
    pub(crate) fn ngram(
        &self,
        rank: u32,
        target: u32,
    ) -> Result<&Prepared<RowGatherBuffers<View>>, Refusal> {
        of_rank(of_rank(&self.ngram, rank)?, target)
    }

    /// Rank `rank`'s gather of its drafts onto `target`.
    pub(crate) fn drafts(
        &self,
        rank: u32,
        target: u32,
    ) -> Result<&Prepared<RowGatherBuffers<View>>, Refusal> {
        of_rank(of_rank(&self.drafts, rank)?, target)
    }

    /// Rank `rank`'s gather of its logits onto rank 0.
    pub(crate) fn logits(&self, rank: u32) -> Result<&Prepared<RowGatherBuffers<View>>, Refusal> {
        of_rank(&self.logits, rank)
    }

    /// Rank `rank`'s logits row exchange.
    pub(crate) fn exchange(&self, rank: u32) -> Result<&Prepared<GatherBuffers<View>>, Refusal> {
        of_rank(&self.exchange, rank)
    }
}

/// Qwen's own ops of one key (`model_qwen38_contract::own_ops`): the PLE
/// stage, residual and convolution, the stream expansion, the drafter's
/// rows and draft argmax (each rank's partial, in rank order), the PLE
/// anchor of a prefill, the PLE checkpoints of a verify step, and the
/// fused up mix of a bucket the decode entries hold.
pub(crate) struct OwnStageOps {
    pub ple_stage: Prepared<PleStageBuffers<View>>,
    pub ple_residual: Prepared<PleResidualBuffers<View>>,
    /// Prefill's per item, decode's over the bucket items, or verify's.
    pub ple_conv: Prepared<PleConvBuffers<View>>,
    pub mtp_gather: Prepared<MtpGatherBuffers<View>>,
    pub mtp_norms: Prepared<MtpNormsBuffers<View>>,
    pub mtp_combine: Prepared<MtpCombineBuffers<View>>,
    draft_partials: Vec<Prepared<DraftArgmaxPartialBuffers<View>>>,
    pub draft_final: Prepared<DraftArgmaxFinalBuffers<View>>,
    ple_anchor: Option<Prepared<PleAnchorBuffers<View>>>,
    ple_checkpoints: Option<Prepared<PleCheckpointsBuffers<View>>>,
    up_mix: Option<Prepared<HcUpMixBuffers<View>>>,
    pub dense: DenseOps,
}

impl OwnStageOps {
    /// `hc_precompute`: the decode buckets of 16 and 32 rows mix the
    /// precomputed bottleneck (`RankStage::precompute_hc`).
    fn prepare(
        own: &dyn OwnOps,
        shape: Shape,
        hc_precompute: bool,
        key: Key,
    ) -> Result<OwnStageOps, Refused> {
        let need = rows_need(key);
        let capacity = key.capacity();
        let layout = PleLayout::of(shape);
        // A verify step's own ops record its verify rows.
        let verify_need =
            Need::rows(capacity).with_exact(exact(key, MAX_VERIFY_ROWS, Anchors::NotApplicable));
        // The drafter's rows are drafts the target's verify step checks, so
        // its ops are asked for nothing.
        let drafter = Need::rows(capacity);
        let (conv, conv_need) = match key {
            Key::Prefill => (
                ItemForm::One,
                Need::rows(capacity).with_exact(exact(key, capacity, Anchors::Companion)),
            ),
            Key::Decode(_) => (ItemForm::Decode, need),
            Key::Verify(_) => (
                ItemForm::Verify {
                    rows: MAX_VERIFY_ROWS,
                },
                verify_need,
            ),
        };
        let up_mix = match key {
            Key::Decode(rows) | Key::Verify(rows) if rows <= DECODE_ENTRY_ROWS => {
                let input = if hc_precompute && matches!(rows, 16 | 32) {
                    UpMixInput::Bottleneck
                } else {
                    UpMixInput::Down
                };
                Some(own.hc_up_mix(&HcUpMix::of(shape, input), need)?)
            }
            _ => None,
        };
        Ok(OwnStageOps {
            ple_stage: own.ple_stage(&PleStage { layout }, need)?,
            ple_residual: own.ple_residual(&PleResidual::of(shape), need)?,
            ple_conv: own.ple_conv(&PleConv::of(shape, conv), conv_need)?,
            mtp_gather: own.mtp_gather(&MtpGather::of(shape), drafter)?,
            mtp_norms: own.mtp_norms(&MtpNorms::of(shape), drafter)?,
            mtp_combine: own.mtp_combine(&MtpCombine::of(shape), drafter)?,
            draft_partials: (0..shape.ranks())
                .map(|rank| own.draft_argmax_partial(&DraftArgmaxPartial::of(shape, rank), drafter))
                .collect::<Result<_, _>>()?,
            draft_final: own.draft_argmax_final(&DraftArgmaxFinal::of(shape), drafter)?,
            ple_anchor: (key == Key::Prefill)
                .then(|| {
                    own.ple_anchor(
                        &PleAnchor { layout },
                        Need::rows(capacity).with_exact(exact(key, capacity, Anchors::Spare)),
                    )
                })
                .transpose()?,
            ple_checkpoints: matches!(key, Key::Verify(_))
                .then(|| own.ple_checkpoints(&PleCheckpoints::of(shape), verify_need))
                .transpose()?,
            up_mix,
            dense: DenseOps::prepare(own, shape, hc_precompute, key)?,
        })
    }

    /// Rank `rank`'s draft argmax over its vocabulary shard.
    pub(crate) fn draft_partial(
        &self,
        rank: u32,
    ) -> Result<&Prepared<DraftArgmaxPartialBuffers<View>>, Refusal> {
        self.draft_partials
            .get(rank as usize)
            .ok_or(Refusal("Qwen3.8 has no draft argmax for the rank"))
    }

    /// The PLE anchor, prefill only.
    pub(crate) fn ple_anchor(&self) -> Result<&Prepared<PleAnchorBuffers<View>>, Refusal> {
        self.ple_anchor
            .as_ref()
            .ok_or(Refusal("Qwen3.8 anchors PLE state in prefill only"))
    }

    /// The PLE checkpoints, verify steps only.
    pub(crate) fn ple_checkpoints(
        &self,
    ) -> Result<&Prepared<PleCheckpointsBuffers<View>>, Refusal> {
        self.ple_checkpoints.as_ref().ok_or(Refusal(
            "Qwen3.8 checkpoints PLE state in verify steps only",
        ))
    }

    /// The fused up mix, at buckets the decode entries hold.
    pub(crate) fn up_mix(&self) -> Result<&Prepared<HcUpMixBuffers<View>>, Refusal> {
        self.up_mix.as_ref().ok_or(Refusal(
            "Qwen3.8 fuses the up mix at decode-entry buckets only",
        ))
    }
}

/// A dense group prepared for a key.
pub(crate) type DenseOp = Prepared<DenseGroupBuffers<View>>;

/// The dense groups of a key: each projection the program launches at the
/// key's rows, the heads of every decode-entry bucket up to them (and the
/// F32 prefill head past them), and the MTP drafter's fc over its stream
/// rows and its token rows.
pub(crate) struct DenseOps {
    groups: Vec<(Projection, DenseOp)>,
    /// By output and the most head rows.
    heads: Vec<(DType, u32, DenseOp)>,
    pub mtp_streams: DenseOp,
    pub mtp_rows: DenseOp,
}

impl DenseOps {
    /// `hc_precompute`: the hyper-connection producers of the 16 and
    /// 32-row buckets also write the bottleneck (`RankStage::precompute_hc`).
    fn prepare(
        own: &dyn OwnOps,
        shape: Shape,
        hc_precompute: bool,
        key: Key,
    ) -> Result<DenseOps, Refused> {
        use Projection::*;
        let capacity = key.capacity();
        let usage = usage(key);
        // A graph bucket the decode entries hold.
        let decode = key != Key::Prefill && capacity <= DECODE_ENTRY_ROWS;
        let projections: &[Projection] = if decode {
            &[
                LinearIn,
                AttentionIn,
                HcIn,
                HcDown,
                LinearOutput,
                AttentionOutput,
                Router,
                SharedGateUp,
                SharedDown,
                PleKey,
                PleValue,
            ]
        } else {
            &[
                LinearQkv,
                LinearZ,
                LinearAb,
                AttentionQkv,
                IndexQk,
                HcDown,
                HcInject,
                HcUp,
                LinearOutput,
                AttentionOutput,
                Router,
                SharedGateUp,
                SharedDown,
                PleKey,
                PleValue,
            ]
        };
        let bottleneck = hc_precompute && decode && matches!(capacity, 16 | 32);
        let groups = projections
            .iter()
            .map(|&projection| {
                let mut op = DenseGroup::of(shape, projection);
                if bottleneck && matches!(projection, HcIn | HcDown) {
                    op = op.with_bottleneck(shape.streams());
                }
                Ok((projection, own.dense_group(&op, usage, rows_need(key))?))
            })
            .collect::<Result<_, Refused>>()?;
        // A step's head rows pick the head of the least bucket that holds
        // them; past the buckets, logits take the F32 prefill head. The
        // drafter's BF16 heads are asked for nothing: drafts are checked by
        // the target's verify step.
        let mut heads = Vec::new();
        for bucket in DECODE_ENTRY_BUCKETS
            .into_iter()
            .filter(|&bucket| bucket <= capacity)
        {
            let head = |output| DenseGroup::of(shape, Head(output));
            let need = Need::rows(bucket).with_exact(exact(key, bucket, Anchors::NotApplicable));
            heads.push((
                DType::F32,
                bucket,
                own.dense_group(&head(DType::F32), usage, need)?,
            ));
            heads.push((
                DType::Bf16,
                bucket,
                own.dense_group(&head(DType::Bf16), usage, Need::rows(bucket))?,
            ));
        }
        if capacity > DECODE_ENTRY_ROWS {
            heads.push((
                DType::F32,
                PREFILL_TOKENS,
                own.dense_group(
                    &DenseGroup::of(shape, Head(DType::F32)),
                    usage,
                    Need::rows(PREFILL_TOKENS).with_exact(exact(
                        key,
                        PREFILL_TOKENS,
                        Anchors::NotApplicable,
                    )),
                )?,
            ));
        }
        let fc = DenseGroup::of(shape, MtpFc);
        let streams = (capacity * shape.streams()).min(PREFILL_TOKENS);
        Ok(DenseOps {
            groups,
            heads,
            mtp_streams: own.dense_group(&fc, usage, Need::rows(streams))?,
            mtp_rows: own.dense_group(&fc, usage, Need::rows(capacity))?,
        })
    }

    /// The group of `projection`.
    pub(crate) fn group(&self, projection: Projection) -> Result<&DenseOp, Refusal> {
        self.groups
            .iter()
            .find(|(prepared, _)| *prepared == projection)
            .map(|(_, op)| op)
            .ok_or(Refusal("Qwen3.8 launches no such dense group at this key"))
    }

    /// The head of `output` over `rows` head rows: the least bucket's that
    /// holds them, or the F32 prefill head (`prefill`: always it).
    pub(crate) fn head(
        &self,
        output: DType,
        rows: u32,
        prefill: bool,
    ) -> Result<&DenseOp, Refusal> {
        self.heads
            .iter()
            .filter(|(dtype, _, _)| *dtype == output)
            .find(|(_, most, _)| {
                if prefill {
                    *most == PREFILL_TOKENS
                } else {
                    *most >= rows
                }
            })
            .map(|(_, _, op)| op)
            .ok_or(Refusal("Qwen3.8 has no head for these rows at this key"))
    }
}

/// Qwen's state copies outside lowering, prepared once at create: each
/// staging's PLE entries committed to the pool over the served items, the
/// drafter's X rows over up to an index list of items, and the prefix
/// snapshot copies of a slot's and of an anchor item's state, each way.
/// Copies keep bits, so they are asked for nothing.
pub(crate) struct Copies {
    pub ple_commit: Prepared<CopyEntriesBuffers<View>>,
    pub rows: Prepared<CopyEntriesBuffers<View>>,
    /// By holder (slot, anchor), then direction (restore, capture).
    prefix: Option<[[PrefixOp; 2]; 2]>,
}

/// A prepared prefix snapshot copy and the copy it was prepared for.
type PrefixOp = (PrefixCopy, Prepared<PrefixCopyBuffers<View>>);

impl Copies {
    /// `listed`: the most items an index list names; `prefix`: the snapshot
    /// copies by holder and direction, where the instance has snapshots.
    pub(crate) fn prepare(
        own: &dyn OwnOps,
        shape: Shape,
        served: Served,
        listed: u32,
        prefix: Option<[[PrefixCopy; 2]; 2]>,
    ) -> Result<Copies, Refused> {
        let entries = |entry_bytes, items| {
            own.copy_entries(
                &CopyEntries {
                    entry_bytes,
                    planes: 1,
                    plane_stride: 0,
                },
                Need::rows(items),
            )
        };
        let snapshot = |op: PrefixCopy| {
            own.prefix_copy(&op, Need::rows(1))
                .map(|prepared| (op, prepared))
        };
        Ok(Copies {
            ple_commit: entries(shape.ple_entry_bytes(), served.items)?,
            rows: entries(crate::scratch::wide_row_bytes(shape), listed)?,
            prefix: prefix
                .map(|holders| -> Result<_, Refused> {
                    let [slot, anchor] = holders;
                    let [slot_restore, slot_capture] = slot;
                    let [anchor_restore, anchor_capture] = anchor;
                    Ok([
                        [snapshot(slot_restore)?, snapshot(slot_capture)?],
                        [snapshot(anchor_restore)?, snapshot(anchor_capture)?],
                    ])
                })
                .transpose()?,
        })
    }

    /// The snapshot copy of `op`, which the instance prepared for its
    /// holder (an anchor item's when `anchor`) and direction.
    pub(crate) fn prefix(
        &self,
        anchor: bool,
        op: &PrefixCopy,
    ) -> Result<&Prepared<PrefixCopyBuffers<View>>, Refusal> {
        let capture = op.direction == model_qwen38_contract::own_ops::copies::Direction::Capture;
        let (prepared, copy) = self
            .prefix
            .as_ref()
            .map(|ops| &ops[usize::from(anchor)][usize::from(capture)])
            .ok_or(Refusal("Qwen3.8 has no prefix snapshot copies"))?;
        if prepared != op {
            return Err(Refusal("Qwen3.8 prepared another prefix snapshot copy"));
        }
        Ok(copy)
    }
}

/// The full-attention layers' ops of one key in context-parallel groups
/// of `size` ranks (`crate::context_parallel`), each group rank's in group
/// rank order: the localization of appended positions to their owners,
/// the owner-routed KV write (BF16 and FP8 caches) and key write, the
/// per-rank selection and the merge of the group's selections, the
/// selected partials over the rank's own pages (BF16 and FP8 caches) and
/// their merge. Each is asked what the program relies on (`exact`): one-row
/// bits at exact verify steps and chunks at the prefill granule, the key
/// write's spare ring past the granule's rows (`ANCHORED_RING`); the model
/// declares these claims (`context_parallel::declared`), and the head-sum
/// selection gives its split merge's.
pub(crate) struct ContextParallelOps {
    localize: Vec<Prepared<LocalizeKeysBuffers<View>>>,
    kv_write: [Prepared<GatedKvWriteBuffers<View>>; 2],
    key_write: Vec<Prepared<SparseKeyWriteBuffers<View>>>,
    select: Vec<Prepared<IndexerSelectBuffers<View>>>,
    pub merge: Prepared<IndexerMergeBuffers<View>>,
    partials: Vec<[Prepared<SelectedPartialsBuffers<View>>; 2]>,
    pub partial_merge: Prepared<PartialMergeBuffers<View>>,
}

impl ContextParallelOps {
    fn prepare(
        backend: &dyn Backend,
        target: &dyn Target,
        shape: Shape,
        size: u32,
        key: Key,
    ) -> Result<ContextParallelOps, Refused> {
        let capacity = key.capacity();
        // Prefill selects and attends in pieces of up to `SCORE_ROWS` rows.
        let pieces = match key {
            Key::Prefill => SCORE_ROWS,
            Key::Decode(_) | Key::Verify(_) => capacity,
        };
        let need = |rows, anchors| Need::rows(rows).with_exact(exact(key, rows, anchors));
        let max_context = shape.context_tokens() as u32;
        let page = PAGE_TOKENS as u32;
        let (theta_bits, mrope, eps_bits) = (
            shape.rope_theta().to_bits(),
            shape.geometry.mrope(),
            shape.eps().to_bits(),
        );
        let block = shape.qsa_block();
        let max_top_k = shape.top_blocks() * block;
        let kv_write = |kv_dtype| {
            backend.owned_kv_write(
                &OwnedKvWriteOp {
                    query: ContextParallelQuery::OwnedQkNormRopeKvWrite {
                        write: GatedAttentionQuery::QkNormRopeKvWrite {
                            q_heads: shape.q_heads,
                            kv_heads: shape.kv_heads(),
                            head_dim: shape.attn_head(),
                            rotary_dim: shape.rotary(),
                            theta_bits,
                            mrope,
                            eps_bits,
                            page_size: page,
                            kv_dtype,
                            capacity,
                        },
                        ranks: size,
                    },
                    max_context,
                },
                need(capacity, Anchors::NotApplicable),
            )
        };
        let key_write = OwnedKeyWriteOp {
            query: ContextParallelQuery::OwnedIndexerKeyWrite {
                write: SparseAttentionQuery::IndexerKeyWrite {
                    head_dim: shape.index_dim(),
                    rotary_dim: shape.rotary(),
                    theta_bits,
                    mrope,
                    eps_bits,
                    block,
                    ring: shape.qsa_ring(),
                    capacity,
                },
                page,
                ranks: size,
            },
            index_heads: shape.index_heads(),
            max_context,
        };
        let select = PooledIndexerQuery::HeadSumSelect {
            heads: shape.index_heads(),
            head_dim: shape.index_dim(),
            group: block,
            max_top_k,
            always_tail: true,
            mode: IndexerSelectionMode::PerRank { ranks: size },
            page_size: page,
            capacity: pieces,
        };
        let phase = target.partials_phase(key);
        let partials = |rank, kv| {
            backend.selected_attention_partials(
                &SelectedPartialsOp {
                    query: ContextParallelQuery::SelectedAttentionPartials {
                        q_heads: shape.q_heads,
                        kv_heads: shape.kv_heads(),
                        head_dim: shape.attn_head(),
                        page,
                        block,
                        top_blocks: shape.top_blocks(),
                        kv,
                        phase,
                        capacity: pieces,
                    },
                    softmax_scale: shape.softmax_scale(),
                    max_context,
                },
                rank,
                size,
                need(pieces, Anchors::NotApplicable),
            )
        };
        Ok(ContextParallelOps {
            localize: (0..size)
                .map(|rank| {
                    backend.localize_keys(
                        &ContextParallelQuery::LocalizeKeys {
                            ranks: size,
                            page,
                            width: 1,
                            capacity,
                        },
                        rank,
                        need(capacity, Anchors::NotApplicable),
                    )
                })
                .collect::<Result<_, _>>()?,
            kv_write: [kv_write(DType::Bf16)?, kv_write(DType::Fp8E4M3)?],
            key_write: (0..size)
                .map(|rank| {
                    backend.owned_key_write(&key_write, rank, need(capacity, ANCHORED_RING))
                })
                .collect::<Result<_, _>>()?,
            select: (0..size)
                .map(|rank| {
                    backend.indexer_select(
                        &select,
                        max_top_k,
                        rank,
                        need(pieces, Anchors::NotApplicable),
                    )
                })
                .collect::<Result<_, _>>()?,
            merge: backend.indexer_merge(
                &ContextParallelQuery::TopKMergeBlocks {
                    ranks: size,
                    group: block,
                    max_top_k,
                    capacity: pieces,
                },
                max_top_k,
                need(pieces, Anchors::NotApplicable),
            )?,
            partials: (0..size)
                .map(|rank| {
                    Ok([
                        partials(rank, DType::Bf16)?,
                        partials(rank, DType::Fp8E4M3)?,
                    ])
                })
                .collect::<Result<_, Refused>>()?,
            partial_merge: backend.partial_merge(
                &ContextParallelQuery::PartialMerge {
                    ranks: size,
                    heads: shape.q_heads * shape.kv_heads(),
                    width: shape.attn_head(),
                    capacity: pieces,
                },
                need(pieces, Anchors::NotApplicable),
            )?,
        })
    }

    /// Group rank `rank`'s localization of appended positions.
    pub(crate) fn localize(
        &self,
        rank: u32,
    ) -> Result<&Prepared<LocalizeKeysBuffers<View>>, Refusal> {
        of_rank(&self.localize, rank)
    }

    /// The owner-routed KV write into a BF16 or an FP8 cache.
    pub(crate) fn kv_write(&self, fp8: bool) -> &Prepared<GatedKvWriteBuffers<View>> {
        &self.kv_write[usize::from(fp8)]
    }

    /// Group rank `rank`'s owner-routed key write.
    pub(crate) fn key_write(
        &self,
        rank: u32,
    ) -> Result<&Prepared<SparseKeyWriteBuffers<View>>, Refusal> {
        of_rank(&self.key_write, rank)
    }

    /// Group rank `rank`'s selection over its own block keys.
    pub(crate) fn select(
        &self,
        rank: u32,
    ) -> Result<&Prepared<IndexerSelectBuffers<View>>, Refusal> {
        of_rank(&self.select, rank)
    }

    /// Group rank `rank`'s selected partials over a BF16 or an FP8 cache.
    pub(crate) fn partials(
        &self,
        rank: u32,
        fp8: bool,
    ) -> Result<&Prepared<SelectedPartialsBuffers<View>>, Refusal> {
        Ok(&of_rank(&self.partials, rank)?[usize::from(fp8)])
    }
}

/// What the group transfers and the barrier are asked: each serves every
/// key of its bucket, so the union of their needs, prefill chunks at the
/// granule and exact verify rows. The shared all-gather is declared a move
/// of up to 8192 rows (`comm::declared_entries`).
fn transfer_need() -> Need {
    Need::rows(crate::context_parallel::MAX_BUCKET).with_exact(Invariance {
        chunk: Chunk::Granule(PREFILL_GRANULE),
        rows: Rows::OneRow {
            max: EXACT_VERIFY.rows,
        },
        ..Invariance::NONE
    })
}

/// The group transfers of one bucket of rows (`context_parallel::bucket`),
/// each group rank's in group rank order: the gathers of the selections'
/// pairs and of the queries, and the gathers of each owner's partial
/// blocks.
pub(crate) struct GroupTransfers {
    pairs: Vec<Prepared<GroupTransferBuffers<View>>>,
    queries: Vec<Prepared<GroupTransferBuffers<View>>>,
    /// By group rank, then owner.
    owners: Vec<Vec<Prepared<GroupTransferBuffers<View>>>>,
}

impl GroupTransfers {
    fn prepare(
        backend: &dyn Backend,
        shape: Shape,
        size: u32,
        bucket: u32,
    ) -> Result<GroupTransfers, Refused> {
        let dims = crate::context_parallel::dims(shape);
        let need = transfer_need();
        let transfers = |transfer, words| {
            let op = GroupTransferOp {
                transfer,
                size,
                words,
            };
            (0..size)
                .map(|rank| backend.group_transfer(&op, rank, need))
                .collect::<Result<Vec<_>, _>>()
        };
        Ok(GroupTransfers {
            pairs: transfers(GroupTransfer::Gather, dims.pair_words(bucket))?,
            queries: transfers(GroupTransfer::Gather, dims.query_words(bucket))?,
            owners: {
                let mut owners: Vec<Vec<_>> = (0..size).map(|_| Vec::new()).collect();
                for owner in 0..size {
                    let block = dims.block_floats(bucket);
                    for (rank, op) in transfers(GroupTransfer::OwnerGather { owner }, block)?
                        .into_iter()
                        .enumerate()
                    {
                        owners[rank].push(op);
                    }
                }
                owners
            },
        })
    }

    /// Group rank `rank`'s gather of its selection's pairs.
    pub(crate) fn pairs(
        &self,
        rank: u32,
    ) -> Result<&Prepared<GroupTransferBuffers<View>>, Refusal> {
        of_rank(&self.pairs, rank)
    }

    /// Group rank `rank`'s gather of its queries.
    pub(crate) fn queries(
        &self,
        rank: u32,
    ) -> Result<&Prepared<GroupTransferBuffers<View>>, Refusal> {
        of_rank(&self.queries, rank)
    }

    /// Group rank `rank`'s gather of its partial blocks onto `owner`.
    pub(crate) fn owner(
        &self,
        rank: u32,
        owner: u32,
    ) -> Result<&Prepared<GroupTransferBuffers<View>>, Refusal> {
        of_rank(of_rank(&self.owners, rank)?, owner)
    }
}

/// The context-parallel ops of an instance whose groups hold more than one
/// rank: each key's attention, each bucket's transfers and the group
/// barrier, a gather of one word whose output no op reads.
pub(crate) struct ContextParallelSet {
    keys: ByKey<ContextParallelOps>,
    buckets: Vec<GroupTransfers>,
    barrier: Vec<Prepared<GroupTransferBuffers<View>>>,
}

impl ContextParallelSet {
    pub(crate) fn key(&self, key: Key) -> Result<&ContextParallelOps, Refusal> {
        self.keys.get(key)
    }

    /// The transfers of `rows` rows' bucket.
    pub(crate) fn bucket(&self, rows: u32) -> Result<&GroupTransfers, Refusal> {
        let index = crate::context_parallel::bucket_index(rows)
            .map_err(|_| Refusal("no Qwen3.8 group bucket"))?;
        self.buckets
            .get(index)
            .ok_or(Refusal("no Qwen3.8 group bucket"))
    }

    /// Group rank `rank`'s barrier.
    pub(crate) fn barrier(
        &self,
        rank: u32,
    ) -> Result<&Prepared<GroupTransferBuffers<View>>, Refusal> {
        of_rank(&self.barrier, rank)
    }
}

/// Every op an instance launches through the backend.
pub(crate) struct Ops {
    stages: ByKey<StageOps>,
    attention: ByKey<AttentionOps>,
    moe: ByKey<MoeOps>,
    collectives: ByKey<CollectiveOps>,
    own: ByKey<OwnStageOps>,
    /// Absent at context-parallel group size 1.
    context_parallel: Option<ContextParallelSet>,
    /// The token embedding and the FP8 n-gram lookup with its fused hash
    /// of a package without a prepared precision component.
    lookups: ByKey<Lookups>,
}

/// The registered lookups a package without a prepared precision component
/// launches (`crate::projections::legacy_lookup`).
pub(crate) struct Lookups {
    pub embedding: std::sync::Arc<dyn bundle_runtime::projection::Launcher>,
    pub ngram: std::sync::Arc<dyn bundle_runtime::projection::Launcher>,
}

impl Ops {
    /// Prepares every op of an instance of `shape` serving `served` on
    /// `target`'s backend and Qwen's own ops on its own kernels;
    /// `f32_norms` when its image stores F32 norm weights, and
    /// `context_parallel` its group size; graphs take the wide all-reduce
    /// when `wide_comm`, and `hc_precompute` is the instance's switch.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn prepare(
        target: &dyn Target,
        shape: Shape,
        served: Served,
        f32_norms: bool,
        context_parallel: u32,
        wide_comm: bool,
        hc_precompute: bool,
    ) -> Result<Ops, Refused> {
        let backend = target.backend(wide_comm);
        let (backend, own) = (backend.as_ref(), target.own());
        let comm = target.comm_bytes(served);
        Ok(Ops {
            stages: ByKey::prepare(served, |key| {
                StageOps::prepare(backend, shape, served, f32_norms, key)
            })?,
            attention: ByKey::prepare(served, |key| AttentionOps::prepare(backend, shape, key))?,
            moe: ByKey::prepare(served, |key| MoeOps::prepare(backend, target, shape, key))?,
            collectives: ByKey::prepare(served, |key| {
                CollectiveOps::prepare(backend, &comm, shape, f32_norms, key)
            })?,
            own: ByKey::prepare(served, |key| {
                OwnStageOps::prepare(own, shape, hc_precompute, key)
            })?,
            context_parallel: (context_parallel > 1)
                .then(|| -> Result<_, Refused> {
                    let size = context_parallel;
                    Ok(ContextParallelSet {
                        keys: ByKey::prepare(served, |key| {
                            ContextParallelOps::prepare(backend, target, shape, size, key)
                        })?,
                        buckets: crate::context_parallel::buckets()
                            .map(|bucket| GroupTransfers::prepare(backend, shape, size, bucket))
                            .collect::<Result<_, _>>()?,
                        barrier: {
                            let op = GroupTransferOp {
                                transfer: GroupTransfer::Barrier,
                                size,
                                words: model_qwen38_contract::context_parallel::BARRIER_WORDS,
                            };
                            (0..size)
                                .map(|rank| backend.group_transfer(&op, rank, transfer_need()))
                                .collect::<Result<_, _>>()?
                        },
                    })
                })
                .transpose()?,
            lookups: ByKey::prepare(served, |key| {
                let lookup = |ngram| {
                    crate::projections::legacy_lookup(shape, backend, own, ngram, usage(key))
                };
                Ok(Lookups {
                    embedding: lookup(false)?,
                    ngram: lookup(true)?,
                })
            })?,
        })
    }

    pub(crate) fn stage(&self, key: Key) -> Result<&StageOps, Refusal> {
        self.stages.get(key)
    }

    pub(crate) fn attention(&self, key: Key) -> Result<&AttentionOps, Refusal> {
        self.attention.get(key)
    }

    pub(crate) fn moe(&self, key: Key) -> Result<&MoeOps, Refusal> {
        self.moe.get(key)
    }

    pub(crate) fn collectives(&self, key: Key) -> Result<&CollectiveOps, Refusal> {
        self.collectives.get(key)
    }

    pub(crate) fn own(&self, key: Key) -> Result<&OwnStageOps, Refusal> {
        self.own.get(key)
    }

    pub(crate) fn lookups(&self, key: Key) -> Result<&Lookups, Refusal> {
        self.lookups.get(key)
    }

    pub(crate) fn context_parallel(&self) -> Result<&ContextParallelSet, Refusal> {
        self.context_parallel
            .as_ref()
            .ok_or(Refusal("Qwen3.8 has no context-parallel ops"))
    }
}

#[cfg(test)]
#[path = "ops_tests.rs"]
mod tests;
