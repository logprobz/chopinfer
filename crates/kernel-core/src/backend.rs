//! The interface between model math and the kernels that compute it.
//!
//! A model states each op as a kernel-core contract (a requirement or query
//! with its F64 oracle) and the exactness it relies on ([`Need`]). A
//! [`Backend`] prepares the op once, when an instance is constructed, or
//! refuses it. A prepared op then only plans launches from buffers passed
//! by role ([`PreparedOp::launches`]): no search, catalog lookup or
//! environment read happens per step.
//!
//! A backend owns what a contract leaves open: numerics choices (reduction
//! orders, K splits, prefill segments), hardware limits, tuning and weight
//! layouts. It reports its choices in [`OpInfo`] and is judged against the
//! contract's oracle. A contract pins numerics only where a model requires
//! them, and the exactness a model relies on is a [`Need`], which a backend
//! that cannot guarantee it refuses at prepare time.
//!
//! Each op of a contract family has one [`Backend`] method (an op kind
//! whose buffers differ has its own), grouped by family. Every method
//! refuses by default, so adding a family or an op breaks no backend.

use std::fmt;

use crate::attention::{AttentionBuffers, AttentionOp};
use crate::collective::{
    AllReduceBuffers, FusedAllReduceBuffers, GatherBuffers, HcAllReduceBuffers, ReduceInput,
    RowGatherBuffers, ShardedRows,
};
use crate::compressed_kv::{CompressedKvWriteBuffers, CompressedKvWriteQuery};
use crate::context_parallel::{
    ContextParallelQuery, GroupTransferBuffers, GroupTransferOp, LocalizeKeysBuffers,
    OwnedKeyWriteOp, OwnedKvWriteOp, PagedPartialsBuffers, PagedPartialsOp, PartialMergeBuffers,
};
use crate::contract::{
    ArgumentList, CommQuery, DType, GemmQuery, LaunchPlan, MoeQuery, MultimodalQuery, NgramQuery,
    RecurrentQuery, Refusal, RowQuery, SigmoidMeanHyperConnectionQuery, SparseAttentionQuery,
};
use crate::gated_mlp::GatedMlpProjectionQuery;
use crate::gated_pooling::{GatedPoolingBuffers, GatedPoolingQuery};
use crate::gated_sparse::{
    BlockScoresBuffers, BlockScoresOp, BlockTopKBuffers, GatedKvWriteBuffers, GatedKvWriteOp,
    IndexQueryBuffers, OutputGateBuffers, OutputGateOp, SelectedPartialsBuffers,
    SelectedPartialsOp, SparseAttentionBuffers, SparseAttentionOp, SparseKeyWriteBuffers,
    SparseKeyWriteOp,
};
use crate::latent_attention::{
    HeadBatchedGemmQuery, HeadGemmBuffers, LatentAttentionBuffers, LatentAttentionQuery,
    LatentKvWriteBuffers, LatentSourcesBuffers,
};
use crate::moe::{
    CombineBuffers, ExpertCombineOp, ProjectedRouteBuffers, RouteBuffers, RoutedSumBuffers,
    RouterBuffers, SharedCombineBuffers,
};
use crate::moe_chained::ChainedMlpProjectionQuery;
use crate::moe_experts::PreparedExperts;
use crate::multimodal::{
    EmbeddingOverrideBuffers, EncoderAttentionBuffers, EncoderGemmBuffers, EncoderHeadBuffers,
    EncoderRowBuffers,
};
use crate::mxfp8_rows::{Mxfp8Rows, Mxfp8RowsBuffers};
use crate::per_channel_decay_delta_rule::PerChannelDecayOp;
use crate::pooled_indexer::{
    CandidateBlocksBuffers, CandidateBlocksQuery, IndexerKeyFormat, IndexerKeyWriteBuffers,
    IndexerMergeBuffers, IndexerSelectBuffers, PooledIndexerQuery, PositionIndexerQuery,
    PositionKeyWriteBuffers, PositionQueryBuffers, PositionSelectBuffers, SelectionCompactBuffers,
};
use crate::position_blocks::{
    CandidateBlockIdsBuffers, PositionBlockIndexerQuery, PositionBlockSelectBuffers,
};
use crate::precision::buffers::ProjectionBuffers;
use crate::precision::execution::Binding;
use crate::precision::{OperationSpec, OperationUse, WeightLayout};
use crate::properties::Invariance;
use crate::recurrent::{ConvBuffers, DeltaRuleBuffers, GatedNormBuffers};
use crate::rmsnorm::{RmsNormBuffers, RmsNormRequirement};
use crate::rope_kv::{RingWriteBuffers, RingWriteOp, RopeBuffers, RopeKvBuffers, RopeKvOp, RopeOp};
use crate::router::{RowBiasRouteBuffers, RowBiasRouterBuffers, RowBiasRouterQuery};
use crate::row_copy::{
    ColumnSlice, GatherByIndex, GatherByIndexBuffers, RowCopyBuffers, StreamBroadcast,
};
use crate::rows::{
    ArgmaxFinalBuffers, ArgmaxPartialBuffers, BottleneckBuffers, EmbeddingBuffers,
    GroupedNormBuffers, InjectBuffers, LogitsBuffers, MixBuffers, NgramBuffers, PleGateBuffers,
};
use crate::sinkhorn_hyper_connection::{
    LaggedMixCollapseBuffers, MixCollapseBuffers, MixCollapseNormBuffers,
    ReduceExpandMixNormBuffers, SinkhornHyperConnectionQuery, StreamExpandBuffers,
    StreamHeadBuffers, WeightedStreamHeadBuffers,
};

/// What a model needs of a prepared op beyond its contract.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Need {
    /// The most rows one launch processes.
    pub capacity: u32,
    /// The exactness the model relies on, such as verify rows equal to
    /// decode rows or a prefill chunk granule.
    pub exact: Invariance,
}

impl Need {
    /// Launches of up to `capacity` rows, with no exactness needed.
    pub const fn rows(capacity: u32) -> Need {
        Need {
            capacity,
            exact: Invariance::NONE,
        }
    }

    pub const fn with_exact(self, exact: Invariance) -> Need {
        Need { exact, ..self }
    }

    /// Refuses an op whose guarantees do not cover the needed exactness.
    /// Every backend checks this before it returns a prepared op.
    pub fn check(self, info: &OpInfo) -> Result<(), Refused> {
        if info.guarantees.provides(self.exact) {
            Ok(())
        } else {
            Err(Refused::Inexact {
                needed: self.exact,
                guaranteed: info.guarantees,
            })
        }
    }
}

/// The least alignment, in bytes, of the [`OpInfo::workspace`] view a host
/// passes: the alignment of the hosts' device buffer allocations. It holds
/// for this dedicated view only; a host that carves the workspace out of a
/// larger arena must place it at an offset that keeps this alignment. Role
/// views carry the alignment their contracts state, not this one.
pub const WORKSPACE_ALIGNMENT: u64 = 256;

/// The role path of the [`OpInfo::workspace`] view in reports and
/// conformance runs. It is reserved: no family's role record names a role
/// by it, so it cannot collide with a family's own `workspace` role.
pub const WORKSPACE_ROLE: &str = "backend_workspace";

/// What a backend chose for a prepared op.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OpInfo {
    /// The versioned implementation, such as
    /// `sm120.row.rmsnorm.tree256.v1`. It names the backend's numerics
    /// choices: ops with equal implementations and contracts compute equal
    /// bits.
    pub implementation: String,
    /// The exactness every launch keeps.
    pub guarantees: Invariance,
    /// The least bytes of each role the op sizes itself, by role path
    /// (`scratch.partial`): workspace roles its family names, and inputs
    /// or outputs whose size follows from the backend's choices. The host
    /// checks every listed role's view before it binds the view. A role
    /// record's named workspace binds where today's hosts lay it out; any
    /// other workspace is [`OpInfo::workspace`].
    pub bytes: Vec<(String, u64)>,
    /// The bytes of workspace the op needs beyond its roles, zero for
    /// none: a backend's own scratch, which no family's record names. The
    /// host passes it to every [`PreparedOp::launches`] as one view whose
    /// address is a multiple of [`WORKSPACE_ALIGNMENT`]; a backend that
    /// needs more alignment rounds up inside the view and asks for the
    /// extra bytes. Reports and conformance runs name it by the reserved
    /// role path [`WORKSPACE_ROLE`]. Its contents are uninitialized before
    /// the first launch and not preserved between steps, and the op owns
    /// it while its launches run: no other op or stream touches it until
    /// they finish.
    pub workspace: u64,
    /// The bytes of each persistent state region a collective keeps on
    /// every rank, in role order (`ranks[r].state[i]`). The host allocates
    /// them on every rank, zeroes them once before the first launch and
    /// leaves them to the op.
    pub state: Vec<u64>,
    /// The layout of each persistent weight role the op reads whose layout
    /// the backend chooses, by role path (`weights`, a record's whole
    /// codes and scales), which the model's weight preparation writes. A
    /// role whose record states a fixed layout (a row-major matrix, a norm
    /// weight vector, a layout the query names) is not listed.
    pub weight_layouts: Vec<(String, WeightLayout)>,
    /// Every symbol the op may launch, in launch order, with the dynamic
    /// shared memory the host configures it for at load.
    pub symbols: Vec<(String, u32)>,
}

impl OpInfo {
    /// An op that launches `symbols` and sizes no buffer itself.
    pub fn new(
        implementation: impl Into<String>,
        guarantees: Invariance,
        symbols: Vec<(String, u32)>,
    ) -> OpInfo {
        OpInfo {
            implementation: implementation.into(),
            guarantees,
            bytes: Vec::new(),
            workspace: 0,
            state: Vec::new(),
            weight_layouts: Vec::new(),
            symbols,
        }
    }

    /// The least bytes of role `path`, if the op sizes it.
    pub fn bytes_of(&self, path: &str) -> Option<u64> {
        self.bytes
            .iter()
            .find(|(role, _)| role == path)
            .map(|&(_, bytes)| bytes)
    }

    /// The layout of weight role `path`, if the backend chooses it.
    pub fn layout_of(&self, path: &str) -> Option<&WeightLayout> {
        self.weight_layouts
            .iter()
            .find(|(role, _)| role == path)
            .map(|(_, layout)| layout)
    }
}

/// A device view the host binds for a launch: the opaque handle the launch
/// arguments carry, and the bytes the view holds, which the op checks.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
pub struct View {
    pub handle: u64,
    pub bytes: u64,
}

impl View {
    /// A view of no bytes: the workspace of an op that needs none.
    pub const EMPTY: View = View {
        handle: 0,
        bytes: 0,
    };

    /// Refuses a view that holds fewer than `bytes` bytes.
    pub fn covers(self, bytes: u64) -> Result<(), Refusal> {
        if self.bytes < bytes {
            Err(Refusal(
                "a buffer does not cover the bytes the op reads or writes",
            ))
        } else {
            Ok(())
        }
    }
}

/// Host-known counts of one launch.
///
/// Every op reads `rows`. The other fields are zero unless the op's
/// [`Backend`] method names them; an op ignores the fields its method does
/// not name. A method also says which indexed device tables (block tables,
/// slot ids, positions into a rotary table) its op reads and what bounds
/// their entries: the host keeps every entry within those bounds, since an
/// op checks a table's size, never its contents.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Counts {
    /// The rows the step schedules for this op, from one to the prepared
    /// capacity: the live rows of an eager stage, the bucket of a captured
    /// graph stage, or the fixed capacity of a stage that always processes
    /// it. An op whose method documents it may also take zero, with its
    /// grid then given by another count (a gather that grids
    /// `padded_rows` and gathers none). Device-live counts below it travel
    /// in buffer roles. Launches
    /// report it as [`Launch::rows`] unless the op documents a per-launch
    /// count of its own.
    pub rows: u32,
    /// Rows after padding, zero when unused: for ops over routed rows, the
    /// routed rows after per-expert padding; for a row op that grids a
    /// padded bucket while its step schedules fewer rows (RMSNorm), that
    /// bucket.
    pub padded_rows: u32,
    /// The items (sequences) of the step, for ops over items: the rows of
    /// its per-item arrays and block tables. Zero otherwise.
    pub items: u32,
    /// The most keys any item of the step attends to (one more than its
    /// largest cache position), for ops that plan from it; zero otherwise.
    pub context: u32,
    /// The rows of a position-indexed rotary table the step reads, for ops
    /// that rotate by positions from such a table; zero otherwise: one more
    /// than the largest rotary position any of its launches reads, whether
    /// a row's own (every M-RoPE section's) or one an earlier step stored
    /// in state the launch reads (a key ring's saved positions). It
    /// differs from `context` where rotary positions are not cache
    /// positions, as with M-RoPE.
    pub positions: u32,
}

impl Counts {
    pub const fn rows(rows: u32) -> Counts {
        Counts {
            rows,
            padded_rows: 0,
            items: 0,
            context: 0,
            positions: 0,
        }
    }

    /// These counts over `items` items of up to `context` keys.
    pub const fn with_items(self, items: u32, context: u32) -> Counts {
        Counts {
            items,
            context,
            ..self
        }
    }

    /// These counts over a rotary table of `positions` rows.
    pub const fn with_positions(self, positions: u32) -> Counts {
        Counts { positions, ..self }
    }
}

/// One kernel launch of a prepared op.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Launch {
    pub symbol: String,
    /// The rows this launch records, at least one: the step's scheduled
    /// [`Counts::rows`], or the per-launch count the op documents (a stage
    /// that processes its bucket or fixed capacity records that).
    pub rows: u32,
    pub plan: LaunchPlan,
    /// The symbol's arguments in parameter order; device buffers are the
    /// handles of the views the caller passed.
    pub arguments: ArgumentList,
    /// The launch waits for its peer ranks: every rank launches it, and no
    /// rank may hold it behind work that waits for a peer.
    pub collective: bool,
    /// The launch keeps every SM busy until it finishes, so nothing can run
    /// beside it.
    pub persistent: bool,
}

/// A producer whose only remaining epilogue is one F32-to-BF16 conversion.
/// The consumer must retain this exact rounding point and write `output`
/// for live rows, leaving inactive rows untouched. Run `launches` first;
/// `input` stays valid only until the producer's state is reused.
pub struct DeferredBf16Round {
    pub launches: Vec<Launch>,
    pub input: View,
    pub output: View,
    pub live_rows: View,
    pub width: u32,
}

/// An op a backend prepared, which plans launches over buffers of role
/// record `B` (of [`View`]s).
pub trait PreparedOp<B>: Send + Sync {
    fn info(&self) -> &OpInfo;

    /// Expose an exact final BF16 round for a consumer to perform. `None`
    /// keeps the ordinary launches. Success preserves every preceding
    /// launch, allocation, state transition and refusal of `launches`.
    fn defer_bf16_round(
        &self,
        _buffers: &B,
        _workspace: View,
        _counts: Counts,
    ) -> Result<Option<DeferredBf16Round>, Refusal> {
        Ok(None)
    }

    /// Initialize persistent metadata from immutable weights before the
    /// first step. Repeat after replacing weights, never during graph
    /// capture. The default needs no initialization.
    fn initialize(&self, _buffers: &B) -> Result<Vec<Launch>, Refusal> {
        Ok(Vec::new())
    }

    /// The launches of one step over `buffers` and the op's `workspace`
    /// (at least [`OpInfo::workspace`] bytes; [`View::EMPTY`] when that is
    /// zero), in order. Refuses counts past the prepared capacity and
    /// buffers that do not match the op.
    fn launches(
        &self,
        buffers: &B,
        workspace: View,
        counts: Counts,
    ) -> Result<Vec<Launch>, Refusal>;
}

pub type Prepared<B> = Box<dyn PreparedOp<B>>;

/// Why a backend did not prepare an op.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Refused {
    /// The backend implements no op of this contract family.
    Unsupported,
    /// The backend cannot guarantee the exactness the model needs.
    Inexact {
        needed: Invariance,
        guaranteed: Invariance,
    },
    /// The op breaks a rule of its contract or a limit of the backend.
    Rule(String),
}

impl From<Refusal> for Refused {
    fn from(refusal: Refusal) -> Refused {
        Refused::Rule(refusal.0.into())
    }
}

impl fmt::Display for Refused {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Refused::Unsupported => f.write_str("the backend does not implement this op family"),
            Refused::Inexact { needed, guaranteed } => write!(
                f,
                "the backend guarantees {guaranteed:?}, which does not cover the needed {needed:?}"
            ),
            Refused::Rule(rule) => f.write_str(rule),
        }
    }
}

impl std::error::Error for Refused {}

/// A set of kernels that computes kernel-core contracts. Each method
/// prepares one op of its family or refuses it; a backend implements the
/// families it supports. Methods are grouped by contract family, one
/// section each.
pub trait Backend: Send + Sync {
    /// The backend's name in refusals and reports, such as `sm120`.
    fn name(&self) -> &str;

    // Rows: norms, embeddings, argmax, hyper-connection, n-gram and PLE rows.

    /// An RMSNorm over up to `need.capacity` rows. A requirement that
    /// leaves its numerics to the backend gets the backend's arithmetic,
    /// which the op names in `OpInfo::implementation`; a backend refuses
    /// pinned numerics it does not compute.
    ///
    /// Launches read `Counts::rows` and `Counts::padded_rows`, the padded
    /// bucket of a step that schedules fewer rows (zero for none). Both are at
    /// most the prepared capacity, and a backend refuses either past it; a
    /// backend may grid the padded rows, masking those past the device live
    /// count.
    fn rmsnorm(
        &self,
        op: &RmsNormRequirement,
        need: Need,
    ) -> Result<Prepared<RmsNormBuffers<View>>, Refused> {
        let _ = (op, need);
        Err(Refused::Unsupported)
    }

    /// An embedding gather (`RowQuery::EmbeddingGather`) from a table whose
    /// first row is vocabulary row `row_offset`: zero for a whole table, the
    /// rank's first row for a vocabulary shard.
    ///
    /// Launches read `Counts::rows`, the rows the ids and output hold, and
    /// `Counts::padded_rows`, the bucket a step grids them to (zero for
    /// none), which launches report as their rows. Ids need no bound: an id
    /// outside the table writes zeros.
    fn embedding_gather(
        &self,
        op: &RowQuery,
        row_offset: u32,
        need: Need,
    ) -> Result<Prepared<EmbeddingBuffers<View>>, Refused> {
        let _ = (op, row_offset, need);
        Err(Refused::Unsupported)
    }

    /// Gather immutable vocabulary shards into replicated rows, preserving
    /// the local gather, widening, ordered sum and final BF16 round.
    fn replicated_embedding(
        &self,
        op: &crate::replicated_embedding::ReplicatedEmbedding,
        rank: u32,
        need: Need,
    ) -> Result<Prepared<crate::replicated_embedding::ReplicatedEmbeddingBuffers<View>>, Refused>
    {
        let _ = (op, rank, need);
        Err(Refused::Unsupported)
    }

    /// The ids and log-probabilities of each row's largest logits:
    /// `RowQuery::LogitsArgmax` (one) or `RowQuery::LogitsTopK` (`k`).
    ///
    /// Launches read `Counts::rows` alone.
    fn logits_top_k(
        &self,
        op: &RowQuery,
        need: Need,
    ) -> Result<Prepared<LogitsBuffers<View>>, Refused> {
        let _ = (op, need);
        Err(Refused::Unsupported)
    }

    /// The argmax of rank `rank`'s vocabulary shard
    /// (`RowQuery::ShardedArgmaxPartial`), which holds vocabulary rows from
    /// `rank * vocab_per_rank`, over logits of the query's dtype.
    ///
    /// Launches read `Counts::rows` alone.
    fn sharded_argmax_partial(
        &self,
        op: &RowQuery,
        rank: u32,
        need: Need,
    ) -> Result<Prepared<ArgmaxPartialBuffers<View>>, Refused> {
        let _ = (op, rank, need);
        Err(Refused::Unsupported)
    }

    /// The argmax over the ranks' gathered partials
    /// (`RowQuery::ShardedArgmaxFinal`), laid out as the query states, into
    /// the buffers' dense output and scattered outputs. A rank-major layout
    /// holds at most its stride of rows (`need.capacity`).
    ///
    /// Launches read `Counts::rows` alone.
    fn sharded_argmax_final(
        &self,
        op: &RowQuery,
        need: Need,
    ) -> Result<Prepared<ArgmaxFinalBuffers<View>>, Refused> {
        let _ = (op, need);
        Err(Refused::Unsupported)
    }

    /// A hyper-connection grouped norm (`GroupedNorm`) over norm weights of
    /// dtype `weight`, BF16 or F32, as the model stores them.
    ///
    /// Launches read `Counts::rows` alone.
    fn hc_grouped_norm(
        &self,
        op: &SigmoidMeanHyperConnectionQuery,
        weight: DType,
        need: Need,
    ) -> Result<Prepared<GroupedNormBuffers<View>>, Refused> {
        let _ = (op, weight, need);
        Err(Refused::Unsupported)
    }

    /// A hyper-connection bottleneck (`Bottleneck`).
    ///
    /// Launches read `Counts::rows` alone.
    fn hc_bottleneck(
        &self,
        op: &SigmoidMeanHyperConnectionQuery,
        need: Need,
    ) -> Result<Prepared<BottleneckBuffers<View>>, Refused> {
        let _ = (op, need);
        Err(Refused::Unsupported)
    }

    /// A hyper-connection mix (`Mix`).
    ///
    /// Launches read `Counts::rows` alone.
    fn hc_mix(
        &self,
        op: &SigmoidMeanHyperConnectionQuery,
        need: Need,
    ) -> Result<Prepared<MixBuffers<View>>, Refused> {
        let _ = (op, need);
        Err(Refused::Unsupported)
    }

    /// A hyper-connection inject (`Inject`).
    ///
    /// Launches read `Counts::rows` alone.
    fn hc_inject(
        &self,
        op: &SigmoidMeanHyperConnectionQuery,
        need: Need,
    ) -> Result<Prepared<InjectBuffers<View>>, Refused> {
        let _ = (op, need);
        Err(Refused::Unsupported)
    }

    /// The n-gram hash gather of rank `rank`'s heads (`HashGather`) over
    /// one sequence chunk. `eos` is the end-of-document token at which a
    /// sequence's history resets (`ngram_oracle::ngram_rows`).
    ///
    /// Launches read `Counts::rows` alone. They are the chunk's rows.
    fn ngram_hash_gather(
        &self,
        op: &NgramQuery,
        eos: u32,
        rank: u32,
        need: Need,
    ) -> Result<Prepared<NgramBuffers<View>>, Refused> {
        let _ = (op, eos, rank, need);
        Err(Refused::Unsupported)
    }

    /// A PLE gate (`NgramQuery::PleGate`).
    ///
    /// Launches read `Counts::rows` alone.
    fn ple_gate(
        &self,
        op: &NgramQuery,
        need: Need,
    ) -> Result<Prepared<PleGateBuffers<View>>, Refused> {
        let _ = (op, need);
        Err(Refused::Unsupported)
    }

    // Sinkhorn hyper-connection.

    /// A mixer's mix-collapse before a sublayer (`MixCollapse` with the
    /// per-sublayer placement).
    ///
    /// Launches read `Counts::rows` alone.
    fn sinkhorn_mix_collapse(
        &self,
        op: &SinkhornHyperConnectionQuery,
        need: Need,
    ) -> Result<Prepared<MixCollapseBuffers<View>>, Refused> {
        let _ = (op, need);
        Err(Refused::Unsupported)
    }

    /// Expand BF16 streams with F32 weights, then run the next lagged mixer.
    /// Only post/post_in and comb/comb_in may alias; backend workspace is
    /// separate from every named role. Counts::rows bounds the device live count.
    fn sinkhorn_expand_mix(
        &self,
        op: &crate::sinkhorn_expand_mix::SinkhornExpandMixQuery,
        need: Need,
    ) -> Result<Prepared<crate::sinkhorn_expand_mix::SinkhornExpandMixBuffers<View>>, Refused> {
        let _ = (op, need);
        Err(Refused::Unsupported)
    }

    /// Round an F32 sublayer sum once to BF16, then expand and mix.
    /// The rounded intermediate is observable and is still written.
    /// Other arithmetic and alias rules match `sinkhorn_expand_mix`.
    fn sinkhorn_round_expand_mix(
        &self,
        op: &crate::sinkhorn_expand_mix::SinkhornExpandMixQuery,
        need: Need,
    ) -> Result<Prepared<crate::sinkhorn_expand_mix::SinkhornRoundExpandMixBuffers<View>>, Refused>
    {
        let _ = (op, need);
        Err(Refused::Unsupported)
    }

    /// Rounded expansion and lagged mixing followed by norm/ceil quantization.
    /// The BF16 collapse is observable; the norm shares its input/live views
    /// with the mixer and retains whole-row quantizer failure handling.
    fn sinkhorn_round_expand_mix_norm(
        &self,
        op: &crate::sinkhorn_expand_mix::SinkhornRoundNormQuery,
        need: Need,
    ) -> Result<Prepared<crate::sinkhorn_expand_mix::SinkhornRoundNormBuffers<View>>, Refused> {
        let _ = (op, need);
        Err(Refused::Unsupported)
    }

    /// A mixer's lagged mix-collapse before a sublayer (`MixCollapse` with
    /// the lagged single-pass placement), which collapses with the previous
    /// sublayer's `pre` and stores its own.
    ///
    /// Launches read `Counts::rows` alone.
    fn sinkhorn_lagged_mix_collapse(
        &self,
        op: &SinkhornHyperConnectionQuery,
        need: Need,
    ) -> Result<Prepared<LaggedMixCollapseBuffers<View>>, Refused> {
        let _ = (op, need);
        Err(Refused::Unsupported)
    }

    /// Prefill-only lagged mixing. Backends without a measured prefill
    /// selection retain their ordinary mixer and its exactness contract.
    fn sinkhorn_lagged_prefill(
        &self,
        op: &SinkhornHyperConnectionQuery,
        need: Need,
    ) -> Result<Prepared<LaggedMixCollapseBuffers<View>>, Refused> {
        self.sinkhorn_lagged_mix_collapse(op, need)
    }

    /// A mixer's mix-collapse and the sublayer's input norm of the
    /// collapsed row (`MixCollapseNorm`).
    ///
    /// Launches read `Counts::rows` alone.
    fn sinkhorn_mix_collapse_norm(
        &self,
        op: &SinkhornHyperConnectionQuery,
        need: Need,
    ) -> Result<Prepared<MixCollapseNormBuffers<View>>, Refused> {
        let _ = (op, need);
        Err(Refused::Unsupported)
    }

    /// The expand of a sublayer's output into the streams (`Expand` or
    /// `ExpandF32`).
    ///
    /// Launches read `Counts::rows` alone.
    fn sinkhorn_expand(
        &self,
        op: &SinkhornHyperConnectionQuery,
        need: Need,
    ) -> Result<Prepared<StreamExpandBuffers<View>>, Refused> {
        let _ = (op, need);
        Err(Refused::Unsupported)
    }

    /// The stream collapse before the final norm (`Head` with
    /// `MeanOfStreams`).
    ///
    /// Launches read `Counts::rows` alone.
    fn sinkhorn_head(
        &self,
        op: &SinkhornHyperConnectionQuery,
        need: Need,
    ) -> Result<Prepared<StreamHeadBuffers<View>>, Refused> {
        let _ = (op, need);
        Err(Refused::Unsupported)
    }

    /// The weighted stream collapse before the final norm (`Head` with
    /// `SigmoidWeighted` or `SigmoidWeightedF32`), by the last lagged
    /// mix-collapse's `pre`. The mode determines the output dtype.
    ///
    /// Launches read `Counts::rows` alone.
    fn sinkhorn_weighted_head(
        &self,
        op: &SinkhornHyperConnectionQuery,
        need: Need,
    ) -> Result<Prepared<WeightedStreamHeadBuffers<View>>, Refused> {
        let _ = (op, need);
        Err(Refused::Unsupported)
    }

    /// The fused comm epilogue that rank `rank` launches
    /// (`AllReduceExpandMixNorm`): the all-reduce of a sublayer's F32
    /// partials, then the expand, the next mix-collapse and its input norm.
    ///
    /// Launches read `Counts::rows` alone.
    fn sinkhorn_reduce_expand_mix_norm(
        &self,
        op: &SinkhornHyperConnectionQuery,
        rank: u32,
        need: Need,
    ) -> Result<Prepared<ReduceExpandMixNormBuffers<View>>, Refused> {
        let _ = (op, rank, need);
        Err(Refused::Unsupported)
    }

    // Projections.

    /// Dense or uniform grouped MXFP8 with independent group inputs and
    /// 128-K F32 folds. Launches read `Counts::rows` and the device live count.
    fn mxfp8_fold_gemm(
        &self,
        op: &crate::mxfp8_gemm::Mxfp8FoldGemmQuery,
        need: Need,
    ) -> Result<Prepared<crate::mxfp8_gemm::Mxfp8FoldGemmBuffers<View>>, Refused> {
        let _ = (op, need);
        Err(Refused::Unsupported)
    }

    /// Explicit staged prefill with the native fold128 contract.
    fn mxfp8_prefill_gemm(
        &self,
        op: &crate::mxfp8_gemm::Mxfp8FoldGemmQuery,
        need: Need,
    ) -> Result<Prepared<crate::mxfp8_gemm::Mxfp8FoldGemmBuffers<View>>, Refused> {
        let _ = (op, need);
        Err(Refused::Unsupported)
    }

    /// Two equal-shaped MXFP8 projections sharing activation and status
    /// views, with distinct weights and outputs. Reads `Counts::rows` and
    /// the device live count. Each output keeps its own 128-K fold chain.
    fn mxfp8_fold_gemm_pair(
        &self,
        op: &crate::mxfp8_gemm::Mxfp8FoldGemmQuery,
        need: Need,
    ) -> Result<Prepared<[crate::mxfp8_gemm::Mxfp8FoldGemmBuffers<View>; 2]>, Refused> {
        let _ = (op, need);
        Err(Refused::Unsupported)
    }

    /// Exact MXFP8 folds with operation-owned persistent state and packed
    /// immutable weights. Initialize once before capture, then keep those
    /// regions exclusive to this op across graph replays. Backend workspace
    /// remains disposable. Each row is admitted once inside the fold launch.
    fn mxfp8_fold_gemm_stream(
        &self,
        op: &crate::mxfp8_gemm::Mxfp8FoldGemmQuery,
        paired: bool,
        need: Need,
    ) -> Result<Prepared<crate::mxfp8_gemm::Mxfp8FoldStreamBuffers<View>>, Refused> {
        let _ = (op, paired, need);
        Err(Refused::Unsupported)
    }

    /// Round inputs to BF16, then quantize each row independently. The
    /// fake-quant form stores the exact decoded values in BF16.
    fn block_quantize(
        &self,
        op: &crate::precision::quantizer::BlockQuantizeQuery,
        need: Need,
    ) -> Result<Prepared<crate::precision::quantizer::BlockQuantizeBuffers<View>>, Refused> {
        let _ = (op, need);
        Err(Refused::Unsupported)
    }

    /// Normalize rows, optionally widen a BF16 output, then quantize it.
    /// Reads `Counts::rows`, `Counts::padded_rows` and the device live count.
    /// Preserves the standalone norm output and both stages' status semantics.
    fn rmsnorm_quantize(
        &self,
        op: &crate::rmsnorm_quantize::RmsNormQuantizeQuery,
        need: Need,
    ) -> Result<Prepared<crate::rmsnorm_quantize::RmsNormQuantizeBuffers<View>>, Refused> {
        let _ = (op, need);
        Err(Refused::Unsupported)
    }

    /// Clamp and evaluate SwiGLU in F32, round once to BF16, then quantize
    /// E4M3/32 with a ceil UE8M0 scale. Each live row is independent.
    fn silu_mul_quantize(
        &self,
        op: &crate::gated_mlp::quantized::SiluMulQuantizeQuery,
        need: Need,
    ) -> Result<Prepared<crate::gated_mlp::quantized::SiluMulQuantizeBuffers<View>>, Refused> {
        let _ = (op, need);
        Err(Refused::Unsupported)
    }

    /// A projection at one use, with the implementation identity its
    /// precision document authenticates. `need.capacity`, from one row to
    /// the use's capacity, is the most rows a launch takes: a model that
    /// runs a use at fewer rows than it binds asks for those. Launches read
    /// `Counts::rows`, the token rows; a grouped projection
    /// also reads `Counts::padded_rows`, its routed rows after padding,
    /// which its gather and quantizer stages refuse past their capacity.
    /// A sharded lookup's ids need no bound: ids outside the rank's shard
    /// (`row_offset`, `shard_rows`) write zeros.
    fn projection(
        &self,
        op: &OperationSpec,
        usage: OperationUse,
        need: Need,
    ) -> Result<Binding<Prepared<ProjectionBuffers<View>>>, Refused> {
        let _ = (op, usage, need);
        Err(Refused::Unsupported)
    }

    // Attention: paged, ring and row attention, folded RoPE.

    /// An attention step of up to `need.capacity` query rows (in decode,
    /// one per item). Its launches take `Counts::rows`, the step's query
    /// rows (a verify step's token bucket), with its items and the most
    /// keys any item attends to. Verify rows equal to decode rows are
    /// `Rows::OneRow` in `need.exact`, which claims no batch independence.
    /// An op that runs each row on its own (such verify rows) grids
    /// `Counts::padded_rows`, the bucket a step pads fewer query rows to,
    /// when that is larger (zero for none); launches report the rows.
    ///
    /// `Counts::items`, at most the prepared items, are the rows of the
    /// per-item arrays and block tables, which launches check at that
    /// count. The op plans for those items, or for its prepared items
    /// when `op.plan_at_capacity`. `Counts::context`, at most the
    /// query's `max_context_capacity`, bounds every item's keys, and the op
    /// plans its splits from it. A fold reads `Counts::positions` rows of
    /// its rotary table: the decode tokens' cache positions for a NeoX
    /// fold, the M-RoPE positions (`rope_positions`) for a gated one. The
    /// host keeps every block table entry below the cache's pages and every
    /// slot id below its rings.
    fn attention(
        &self,
        op: &AttentionOp,
        need: Need,
    ) -> Result<Prepared<AttentionBuffers<View>>, Refused> {
        let _ = (op, need);
        Err(Refused::Unsupported)
    }

    // RoPE and KV writes.

    /// A RoPE of each row's heads on its own, with no cache write.
    ///
    /// Launches read `Counts::rows`, the rows the buffers hold,
    /// `Counts::padded_rows`, the bucket a step grids them to (zero for
    /// none), which launches report as their rows, and a table-backed
    /// rotation `Counts::positions`, the rows of its table (at least one):
    /// one more than the largest row position.
    fn rope(&self, op: &RopeOp, need: Need) -> Result<Prepared<RopeBuffers<View>>, Refused> {
        let _ = (op, need);
        Err(Refused::Unsupported)
    }

    /// A RoPE of packed QKV rows fused with the write of their K and V
    /// into a paged cache or a window ring.
    ///
    /// Launches read `Counts::rows`, the rows the buffers hold,
    /// `Counts::padded_rows`, the bucket a step grids them to (zero for
    /// none), which launches report as their rows, and a table-backed
    /// rotation `Counts::positions`, the rows of its table (at least one):
    /// one more than the largest row position. The host keeps every row's item
    /// (`token_batch`) below the rows of the block tables or slot ids it binds,
    /// every block table entry below the cache's pages, every slot id below its
    /// rings, and a paged write's positions below its `max_context`.
    fn rope_kv_write(
        &self,
        op: &RopeKvOp,
        need: Need,
    ) -> Result<Prepared<RopeKvBuffers<View>>, Refused> {
        let _ = (op, need);
        Err(Refused::Unsupported)
    }

    /// The write of K and V rows into window rings.
    ///
    /// Launches read `Counts::rows`, the rows the buffers hold, and
    /// `Counts::padded_rows`, the bucket a step grids them to (zero for
    /// none), which launches report as their rows. The host keeps every row's item
    /// (`token_batch`) below the slot ids' rows and every slot id below the
    /// cache's rings.
    fn ring_write(
        &self,
        op: &RingWriteOp,
        need: Need,
    ) -> Result<Prepared<RingWriteBuffers<View>>, Refused> {
        let _ = (op, need);
        Err(Refused::Unsupported)
    }

    // Gated and sparse attention: the gated Q/K norm, M-RoPE and KV write
    // and output gate, and block-sparse attention with its indexer
    // (`crate::gated_sparse`).

    /// The gated Q/K norm, M-RoPE and paged KV write.
    ///
    /// Launches read `Counts::rows`, `Counts::items` (the rows of the block
    /// tables, at least one) and `Counts::positions`, the rotary table's rows:
    /// one more than the largest rotary position any launch reads, its rows'
    /// `rope_positions`. The host keeps every row's item below `Counts::items`,
    /// every cache position below the op's `max_context` and every block table
    /// entry below the cache's pages.
    fn gated_kv_write(
        &self,
        op: &GatedKvWriteOp,
        need: Need,
    ) -> Result<Prepared<GatedKvWriteBuffers<View>>, Refused> {
        let _ = (op, need);
        Err(Refused::Unsupported)
    }

    /// The sigmoid output gate of gated attention.
    ///
    /// Launches read `Counts::rows` alone.
    fn output_gate(
        &self,
        op: &OutputGateOp,
        need: Need,
    ) -> Result<Prepared<OutputGateBuffers<View>>, Refused> {
        let _ = (op, need);
        Err(Refused::Unsupported)
    }

    /// The sparse indexer's query norm and M-RoPE
    /// (`SparseAttentionQuery::IndexerQueryGemmaNormMrope`).
    ///
    /// Launches read `Counts::rows`, `Counts::context` (the most tokens any row
    /// sees, from one to the op's `max_context`; zero is refused) and
    /// `Counts::positions`, the rotary table's rows: one more than the largest
    /// rotary position any launch reads, its rows' `rope_positions`.
    fn sparse_index_query(
        &self,
        op: &SparseAttentionQuery,
        need: Need,
    ) -> Result<Prepared<IndexQueryBuffers<View>>, Refused> {
        let _ = (op, need);
        Err(Refused::Unsupported)
    }

    /// The sparse indexer's raw-key ring and block-key write.
    ///
    /// Launches read `Counts::rows`, `Counts::items` (the rows of the block
    /// tables, slot ids and anchors, at least one), `Counts::context` (from one
    /// to the op's `max_context`) and `Counts::positions`, the rotary table's
    /// rows: one more than the largest rotary position any launch reads, its
    /// rows' `rope_positions` and the positions an earlier step stored in the
    /// key rings for a block a row completes. The host keeps every row's item
    /// below `Counts::items`, every position below `max_context`, every block
    /// table entry below the cache's pages and every slot id below its rings.
    fn sparse_key_write(
        &self,
        op: &SparseKeyWriteOp,
        need: Need,
    ) -> Result<Prepared<SparseKeyWriteBuffers<View>>, Refused> {
        let _ = (op, need);
        Err(Refused::Unsupported)
    }

    /// The sparse indexer's block scores at use `usage`, whose phase says
    /// how the step runs its rows (a backend may score a graph step's rows
    /// and a prompt chunk's rows differently); `need.capacity` is the
    /// use's.
    ///
    /// Launches read `Counts::rows`, `Counts::items` (the rows of the block
    /// tables, at least one) and `Counts::context`, from one to the op's
    /// `max_context`: the most tokens any row sees, from which a backend may
    /// grid its key chunks, so no row may see more.
    fn sparse_block_scores(
        &self,
        op: &BlockScoresOp,
        usage: OperationUse,
        need: Need,
    ) -> Result<Prepared<BlockScoresBuffers<View>>, Refused> {
        let _ = (op, usage, need);
        Err(Refused::Unsupported)
    }

    /// The top-k block selection (`SparseAttentionQuery::BlockTopK`) over
    /// the scores of this backend's block scores.
    ///
    /// Launches read `Counts::rows` and `Counts::context`, from one to the op's
    /// `max_context`.
    fn sparse_block_top_k(
        &self,
        op: &SparseAttentionQuery,
        need: Need,
    ) -> Result<Prepared<BlockTopKBuffers<View>>, Refused> {
        let _ = (op, need);
        Err(Refused::Unsupported)
    }

    /// Attention over each row's selected blocks and incomplete tail.
    ///
    /// Launches read `Counts::rows`, `Counts::items` (the rows of the block
    /// tables, at least one) and `Counts::context`, from one to the op's
    /// `max_context`. The host keeps every row's item below `Counts::items`,
    /// every position below `max_context` and every block table entry below the
    /// cache's pages.
    fn sparse_attention(
        &self,
        op: &SparseAttentionOp,
        need: Need,
    ) -> Result<Prepared<SparseAttentionBuffers<View>>, Refused> {
        let _ = (op, need);
        Err(Refused::Unsupported)
    }

    /// The partials of sparse attention over the keys context-parallel group
    /// rank `group_rank` of `group_size` owns.
    ///
    /// Launches read `Counts::rows` and `Counts::items`, the rows of the block
    /// tables, at least one.
    fn selected_attention_partials(
        &self,
        op: &SelectedPartialsOp,
        group_rank: u32,
        group_size: u32,
        need: Need,
    ) -> Result<Prepared<SelectedPartialsBuffers<View>>, Refused> {
        let _ = (op, group_rank, group_size, need);
        Err(Refused::Unsupported)
    }

    // Latent attention.

    /// In-place F32 query-tail RoPE and its separate post-RoPE BF16 Q output.
    /// Launches read `Counts::rows`; the device live count selects written rows.
    fn latent_query_rope(
        &self,
        op: &crate::latent_glue::LatentQueryRope,
        need: Need,
    ) -> Result<Prepared<crate::latent_glue::LatentQueryRopeBuffers<View>>, Refused> {
        let _ = (op, need);
        Err(Refused::Unsupported)
    }

    /// Lists for staged prefixes with holes in the retained ring snapshot.
    fn latent_staged_window_keys(
        &self,
        op: &crate::latent_staged_window_keys::LatentStagedWindowKeys,
        need: Need,
    ) -> Result<
        Prepared<crate::latent_staged_window_keys::LatentStagedWindowKeyBuffers<View>>,
        Refused,
    > {
        let _ = (op, need);
        Err(Refused::Unsupported)
    }

    /// Copy each item's retained ring prefix into its prefill stage.
    fn latent_window_stage(
        &self,
        op: &crate::latent_window_stage::LatentWindowStage,
        need: Need,
    ) -> Result<Prepared<crate::latent_window_stage::LatentWindowStageBuffers<View>>, Refused> {
        let _ = (op, need);
        Err(Refused::Unsupported)
    }

    /// F32 normalization and tail rotation, then shared trained quantization
    /// and separate native window code and scale plane writes.
    fn latent_window_write(
        &self,
        op: &crate::latent_glue::LatentWindowWrite,
        need: Need,
    ) -> Result<Prepared<crate::latent_glue::LatentWindowWriteBuffers<View>>, Refused> {
        let _ = (op, need);
        Err(Refused::Unsupported)
    }

    /// The same window-write contract with a single producer/epilogue launch.
    /// Reads `Counts::rows` and the device live count. An unavailable fusion
    /// refuses rather than silently substituting a multi-launch chain.
    fn latent_window_write_fused(
        &self,
        op: &crate::latent_glue::LatentWindowWrite,
        need: Need,
    ) -> Result<Prepared<crate::latent_glue::LatentWindowWriteBuffers<View>>, Refused> {
        let _ = (op, need);
        Err(Refused::Unsupported)
    }

    /// Causal physical window slots in increasing absolute-position order.
    /// Launches read `Counts::rows`; the device live count selects written rows.
    fn latent_window_keys(
        &self,
        op: &crate::latent_glue::LatentWindowKeys,
        need: Need,
    ) -> Result<Prepared<crate::latent_glue::LatentWindowKeyBuffers<View>>, Refused> {
        let _ = (op, need);
        Err(Refused::Unsupported)
    }

    /// A latent KV write (`LatentAttentionQuery::KvWrite`).
    ///
    /// Launches read `Counts::rows` alone. The block tables state their own
    /// sequences and pages (`BlockTables`): the host keeps every row's sequence
    /// (`row_batch`) below them, every position below `max_pages * page_size`
    /// (or -1, which writes nothing) and every table entry below the cache's
    /// pages.
    fn latent_kv_write(
        &self,
        op: &LatentAttentionQuery,
        need: Need,
    ) -> Result<Prepared<LatentKvWriteBuffers<View>>, Refused> {
        let _ = (op, need);
        Err(Refused::Unsupported)
    }

    /// A latent attention (`LatentAttentionQuery::Attention`) over each
    /// row's key list, with whatever split merge the backend's partition
    /// needs.
    ///
    /// Launches read `Counts::rows` alone. The host keeps every row's sequence
    /// (`row_batch`) below the block tables' sequences, every key below
    /// `max_pages * page_size` and every table entry below the cache's pages.
    fn latent_attention(
        &self,
        op: &LatentAttentionQuery,
        need: Need,
    ) -> Result<Prepared<LatentAttentionBuffers<View>>, Refused> {
        let _ = (op, need);
        Err(Refused::Unsupported)
    }

    /// Native latent attention over a per-layer window and an optional
    /// paged compressed source. Lists specify visibility; -1 is a hole.
    /// The caller keeps sequence, page and key indices within their views.
    /// Source code/scale products must be exact finite BF16 operands.
    fn latent_attention_sources(
        &self,
        op: &LatentAttentionQuery,
        need: Need,
    ) -> Result<Prepared<LatentSourcesBuffers<View>>, Refused> {
        let _ = (op, need);
        Err(Refused::Unsupported)
    }

    /// A head-batched GEMM, an absorbed latent projection.
    ///
    /// Launches read `Counts::rows` alone.
    fn head_batched_gemm(
        &self,
        op: &HeadBatchedGemmQuery,
        need: Need,
    ) -> Result<Prepared<HeadGemmBuffers<View>>, Refused> {
        let _ = (op, need);
        Err(Refused::Unsupported)
    }

    /// F32 gated pooling followed by F32 RMSNorm, with exact F32 tail state.
    ///
    /// Launches read `Counts::rows` and `Counts::items`. The host bounds
    /// device live rows by `rows`, item IDs by `items`, and every source or
    /// checkpoint ID by `tail.entries`. Each item has one consecutive run
    /// of rows, with consecutive non-hole positions. Distinct items have
    /// distinct source slots; checkpoints alias neither sources nor each
    /// other. Verify runs hold at most `max_rows` rows per item.
    fn gated_pooling(
        &self,
        op: &GatedPoolingQuery,
        need: Need,
    ) -> Result<Prepared<GatedPoolingBuffers<View>>, Refused> {
        let _ = (op, need);
        Err(Refused::Unsupported)
    }

    /// Normalize F32 projected index keys, rotate their tail in F32 at each
    /// group's first token, then round once at the trained MXFP4 quantizer
    /// input and write its codes and scales.
    /// Launches read `Counts::rows` and `Counts::positions`. The host bounds
    /// item IDs by the page tables' sequences, compressed positions by
    /// `max_pages * page_size`, and their product with the compression ratio
    /// by `Counts::positions`. Page IDs are below `keys.pages`; distinct
    /// live rows address distinct slots. Negative positions/page IDs skip.
    fn position_indexer_key_write(
        &self,
        op: &PositionIndexerQuery,
        need: Need,
    ) -> Result<Prepared<PositionKeyWriteBuffers<View>>, Refused> {
        let _ = (op, need);
        Err(Refused::Unsupported)
    }

    /// The same index-key preparation with explicit physical storage.
    /// Fp4 writes the native code and scale planes. Bf16 writes each exact
    /// native E2M1 times UE8M0 product in channel order to `keys.codes`,
    /// with two bytes per value, `keys.scales = View::EMPTY` and scale
    /// stride zero. Both modes preserve the native finite-product status
    /// flags and hole rules; BF16 storage never bypasses quantization.
    fn position_indexer_key_write_with_storage(
        &self,
        op: &PositionIndexerQuery,
        storage: crate::pooled_indexer::IndexerKeyFormat,
        need: Need,
    ) -> Result<Prepared<PositionKeyWriteBuffers<View>>, Refused> {
        let _ = (op, storage, need);
        Err(Refused::Unsupported)
    }

    /// Rotate normalized F32 latent tails, round the quantizer input to
    /// BF16 once, and write trained E2M1 values
    /// with E4M3-per-16 scales, or their exact BF16 decoded representation.
    /// Counts and table bounds are those of `position_indexer_key_write`,
    /// with `cache.pages`. The input latent stays unmodified.
    fn compressed_kv_write(
        &self,
        op: &CompressedKvWriteQuery,
        need: Need,
    ) -> Result<Prepared<CompressedKvWriteBuffers<View>>, Refused> {
        let _ = (op, need);
        Err(Refused::Unsupported)
    }

    // Pooled indexer: the key write with its tail state, the selection, the
    // exact merge of per-rank selections and the selection compaction
    // (`crate::pooled_indexer`).

    /// A key LayerNorm and gated pooling write
    /// (`PooledIndexerQuery::KeyWrite`) at context-parallel group rank
    /// `group_rank` of `group_size`, which stores the pooled keys of the
    /// pages it owns (`group_size` 1 owns every page).
    ///
    /// Launches read `Counts::rows` alone. The host keeps every row's item
    /// (`row_batch`) below the rows of the page tables and tail slot ids, every
    /// position below `max_pages * page_size * group_size`, every page table
    /// entry below the pooled pages and every tail slot id below the tail
    /// entries.
    fn indexer_key_write(
        &self,
        op: &PooledIndexerQuery,
        group_rank: u32,
        group_size: u32,
        need: Need,
    ) -> Result<Prepared<IndexerKeyWriteBuffers<View>>, Refused> {
        let _ = (op, group_rank, group_size, need);
        Err(Refused::Unsupported)
    }

    /// A selection (`PooledIndexerQuery::Select` or `HeadSumSelect`) that
    /// keeps the instance's runtime `top_k` tokens. A `Global` selection
    /// takes rank 0; a `PerRank` one scores group rank `rank`'s pools.
    ///
    /// Launches read `Counts::rows` alone. The host keeps every row's item
    /// (`row_batch`) below the page tables' sequences, every position below
    /// `max_pages * page_size` and every page table entry below the pooled
    /// pages.
    fn indexer_select(
        &self,
        op: &PooledIndexerQuery,
        top_k: u32,
        rank: u32,
        need: Need,
    ) -> Result<Prepared<IndexerSelectBuffers<View>>, Refused> {
        let _ = (op, top_k, rank, need);
        Err(Refused::Unsupported)
    }

    /// The exact merge of per-rank selections of the runtime `top_k` into
    /// a key list (`ContextParallelQuery::TopKMerge`) or block ids
    /// (`TopKMergeBlocks`).
    ///
    /// Launches read `Counts::rows` alone.
    fn indexer_merge(
        &self,
        op: &ContextParallelQuery,
        top_k: u32,
        need: Need,
    ) -> Result<Prepared<IndexerMergeBuffers<View>>, Refused> {
        let _ = (op, top_k, need);
        Err(Refused::Unsupported)
    }

    /// A selection compaction (`PooledIndexerQuery::CompactSelection`).
    ///
    /// Launches read `Counts::rows` alone. The host keeps every row's source
    /// below the lists' rows.
    fn indexer_compact(
        &self,
        op: &PooledIndexerQuery,
        need: Need,
    ) -> Result<Prepared<SelectionCompactBuffers<View>>, Refused> {
        let _ = (op, need);
        Err(Refused::Unsupported)
    }

    /// F32 tail rotation followed by shared trained MXFP4 fake quantization.
    /// The only BF16 rounding is at the quantizer input; output BF16 stores
    /// the decoded values without loss. Launches read `Counts::rows` and
    /// `Counts::positions`, the number of table rows. Live query positions
    /// outside that table extent set POSITION_QUERY_INVALID_POSITION on the
    /// sticky preparation plane and produce zero rows. Inputs, output and
    /// backend workspace occupy distinct storage. The caller checks this
    /// plane before accepting selection; score/select status stays separate.
    fn position_indexer_query(
        &self,
        op: &PositionIndexerQuery,
        need: Need,
    ) -> Result<Prepared<PositionQueryBuffers<View>>, Refused> {
        let _ = (op, need);
        Err(Refused::Unsupported)
    }

    /// Position scores and exact top-k on losslessly stored BF16 queries,
    /// F32 projected head weights and MXFP4 key pages. Both dot operands
    /// decode exactly to F32. `position_capacity` is the score/mask row width; the
    /// instance's `top_k` is at most the query's maximum.
    ///
    /// Launches read `Counts::rows` alone. The host bounds each row's item
    /// by the page tables' sequences, token positions by the score width
    /// times the compression ratio, and page ids by the key pool. Negative
    /// positions and page-table entries are holes, excluded from selection.
    /// Each live row writes status 0, or status 1 and an empty list on
    /// non-finite admitted arithmetic. Failed scores become -inf so later
    /// graph stages can run safely. The caller must reject failed steps
    /// before accepting their results; see `PositionIndexerQuery`.
    fn position_indexer(
        &self,
        op: &PositionIndexerQuery,
        top_k: u32,
        position_capacity: u32,
        need: Need,
    ) -> Result<Prepared<PositionSelectBuffers<View>>, Refused> {
        let _ = (op, top_k, position_capacity, need);
        Err(Refused::Unsupported)
    }

    /// Position v3 with an explicit key-storage layout. The native method
    /// remains the default; BF16 requires a registered comparison reader.
    fn position_indexer_with_storage(
        &self,
        op: &PositionIndexerQuery,
        storage: IndexerKeyFormat,
        top_k: u32,
        position_capacity: u32,
        need: Need,
    ) -> Result<Prepared<PositionSelectBuffers<View>>, Refused> {
        match storage {
            IndexerKeyFormat::Fp4 => self.position_indexer(op, top_k, position_capacity, need),
            IndexerKeyFormat::Bf16 => Err(Refused::Unsupported),
        }
    }

    /// Candidate block pre-filter over a row's F32 scores. Launches read
    /// `Counts::rows` alone. The host bounds each visible compressed count
    /// by `position_capacity`; zero visible positions emit an all-zero mask.
    fn indexer_candidate_blocks(
        &self,
        op: &CandidateBlocksQuery,
        position_capacity: u32,
        need: Need,
    ) -> Result<Prepared<CandidateBlocksBuffers<View>>, Refused> {
        let _ = (op, position_capacity, need);
        Err(Refused::Unsupported)
    }

    /// Compact candidate producer. Same maxima, pinning and ties as the
    /// dense filter; writes sorted IDs plus row provenance and source status.
    /// Counts::rows bounds rows. Inputs and outputs occupy distinct storage.
    fn indexer_candidate_block_ids(
        &self,
        op: &CandidateBlocksQuery,
        position_capacity: u32,
        need: Need,
    ) -> Result<Prepared<CandidateBlockIdsBuffers<View>>, Refused> {
        let _ = (op, position_capacity, need);
        Err(Refused::Unsupported)
    }

    /// Keep `top_k_blocks` candidates in the fixed `op.top_k_blocks` ID
    /// stride, padding unused slots with -1. The count must be in
    /// 1..=op.top_k_blocks. Other producer semantics are unchanged.
    fn indexer_candidate_block_ids_runtime(
        &self,
        op: &CandidateBlocksQuery,
        top_k_blocks: u32,
        position_capacity: u32,
        need: Need,
    ) -> Result<Prepared<CandidateBlockIdsBuffers<View>>, Refused> {
        let _ = (op, top_k_blocks, position_capacity, need);
        Err(Refused::Unsupported)
    }

    /// Direct compact Position v3 scoring and selection. Position capacity
    /// bounds the original key coordinates, not the compact score slots.
    /// Uses Counts::rows only; page-table and row bounds match position_indexer.
    /// All input planes are read-only and distinct from scores/status/output.
    /// See position_blocks for provenance and malformed-list failure rules.
    fn position_block_indexer(
        &self,
        op: &PositionBlockIndexerQuery,
        top_k: u32,
        position_capacity: u32,
        need: Need,
    ) -> Result<Prepared<PositionBlockSelectBuffers<View>>, Refused> {
        let _ = (op, top_k, position_capacity, need);
        Err(Refused::Unsupported)
    }

    /// Compact Position v3 with the same explicit key-storage choice.
    fn position_block_indexer_with_storage(
        &self,
        op: &PositionBlockIndexerQuery,
        storage: IndexerKeyFormat,
        top_k: u32,
        position_capacity: u32,
        need: Need,
    ) -> Result<Prepared<PositionBlockSelectBuffers<View>>, Refused> {
        match storage {
            IndexerKeyFormat::Fp4 => {
                self.position_block_indexer(op, top_k, position_capacity, need)
            }
            IndexerKeyFormat::Bf16 => Err(Refused::Unsupported),
        }
    }

    // Recurrent layers: convolution, gated delta rule, per-channel decay,
    // gated norm (`crate::recurrent`).

    /// A depthwise causal convolution (`RecurrentQuery::CausalConv1d`).
    ///
    /// A recurrent op reads and writes per-sequence state in the host's pool by
    /// entry id (`crate::recurrent`, "State"); a prefill's chunk granule and
    /// spare anchors, and verify rows equal to decode rows, are guarantees a
    /// model needs through `need.exact`. `need.capacity` is at most the query's
    /// capacity, and `items` bounds the sequences one launch carries, which
    /// sizes the op's workspace. Launches read `Counts::rows` and
    /// `Counts::items`, the launch's sequences, at most `items`. The host keeps
    /// every slot id below the pool's entries and every row offset at most
    /// `Counts::rows`.
    fn causal_conv1d(
        &self,
        op: &RecurrentQuery,
        items: u32,
        need: Need,
    ) -> Result<Prepared<ConvBuffers<View>>, Refused> {
        let _ = (op, items, need);
        Err(Refused::Unsupported)
    }

    /// A gated delta rule (`RecurrentQuery::GatedDeltaRule`).
    ///
    /// A recurrent op reads and writes per-sequence state in the host's pool by
    /// entry id (`crate::recurrent`, "State"); a prefill's chunk granule and
    /// spare anchors, and verify rows equal to decode rows, are guarantees a
    /// model needs through `need.exact`. `need.capacity` is at most the query's
    /// capacity, and `items` bounds the sequences one launch carries, which
    /// sizes the op's workspace. Launches read `Counts::rows` and
    /// `Counts::items`, the launch's sequences, at most `items`. The host keeps
    /// every slot id below the pool's entries and every row offset at most
    /// `Counts::rows`.
    fn gated_delta_rule(
        &self,
        op: &RecurrentQuery,
        items: u32,
        need: Need,
    ) -> Result<Prepared<DeltaRuleBuffers<View>>, Refused> {
        let _ = (op, items, need);
        Err(Refused::Unsupported)
    }

    /// A per-head gated RMSNorm of a delta rule's output
    /// (`RecurrentQuery::GatedRmsNorm`).
    ///
    /// Launches read `Counts::rows` alone. The row-local gated norm takes each
    /// row as a sequence and reads no items.
    fn gated_rmsnorm(
        &self,
        op: &RecurrentQuery,
        need: Need,
    ) -> Result<Prepared<GatedNormBuffers<View>>, Refused> {
        let _ = (op, need);
        Err(Refused::Unsupported)
    }

    /// The q, k and v short convolution of a per-channel-decay delta rule
    /// (`PerChannelDecayDeltaRuleQuery::ShortConv`).
    ///
    /// A recurrent op reads and writes per-sequence state in the host's pool by
    /// entry id (`crate::recurrent`, "State"); a prefill's chunk granule and
    /// spare anchors, and verify rows equal to decode rows, are guarantees a
    /// model needs through `need.exact`. `need.capacity` is at most the query's
    /// capacity, and `items` bounds the sequences one launch carries, which
    /// sizes the op's workspace. Launches read `Counts::rows` and
    /// `Counts::items`, the launch's sequences, at most `items`; an op with
    /// `op.plan_at_capacity` plans every launch for `items` and checks its
    /// per-sequence arrays for `Counts::items` sequences. The host keeps
    /// every slot id below the pool's entries, every row offset at most
    /// `Counts::rows` and the device's live sequences at most `Counts::items`.
    fn short_conv(
        &self,
        op: &PerChannelDecayOp,
        items: u32,
        need: Need,
    ) -> Result<Prepared<ConvBuffers<View>>, Refused> {
        let _ = (op, items, need);
        Err(Refused::Unsupported)
    }

    /// A per-channel-decay delta rule
    /// (`PerChannelDecayDeltaRuleQuery::DeltaRule`).
    ///
    /// A recurrent op reads and writes per-sequence state in the host's pool by
    /// entry id (`crate::recurrent`, "State"); a prefill's chunk granule and
    /// spare anchors, and verify rows equal to decode rows, are guarantees a
    /// model needs through `need.exact`. `need.capacity` is at most the query's
    /// capacity, and `items` bounds the sequences one launch carries, which
    /// sizes the op's workspace. Launches read `Counts::rows` and
    /// `Counts::items`, the launch's sequences, at most `items`; an op with
    /// `op.plan_at_capacity` plans every launch for `items` and checks its
    /// per-sequence arrays for `Counts::items` sequences. The host keeps
    /// every slot id below the pool's entries, every row offset at most
    /// `Counts::rows` and the device's live sequences at most `Counts::items`.
    fn channel_decay_delta_rule(
        &self,
        op: &PerChannelDecayOp,
        items: u32,
        need: Need,
    ) -> Result<Prepared<DeltaRuleBuffers<View>>, Refused> {
        let _ = (op, items, need);
        Err(Refused::Unsupported)
    }

    // MoE: router, routes, permute, combine and grouped experts.

    /// Rank `rank`'s routes: each token's selection by `router`
    /// (`RowQuery::Router`), then the route metadata of the rank's experts
    /// (`routes`, `MoeQuery::LocalRoutes`, over the router's experts and
    /// top-k), each expert's routes padded to `route_tile`: the row tile of
    /// the expert op that reads them (`moe`) at the use's capacity. Where
    /// that expert op runs fewer rows at another tile
    /// (`moe_experts::PreparedExperts::route_tile`), the route op pads a
    /// launch of those rows to that tile. The op serves the step's use
    /// (`usage`: phase, the rows it holds, graph), whose capacity is
    /// `need.capacity`; the backend chooses for it whether one launch does
    /// both. Claims in `need.exact` about the route metadata are about the
    /// routed operation (`Scope::RoutedOperation`).
    ///
    /// Launches read `Counts::rows`, the token rows, alone.
    fn moe_route(
        &self,
        router: &RowQuery,
        routes: &MoeQuery,
        rank: u32,
        route_tile: u32,
        usage: OperationUse,
        need: Need,
    ) -> Result<Prepared<RouteBuffers<View>>, Refused> {
        let _ = (router, routes, rank, route_tile, usage, need);
        Err(Refused::Unsupported)
    }

    /// Rank `rank`'s routes of each token's router logits, which the op
    /// projects itself: the logits are the hidden rows times the router
    /// weight (`projection`, a dense GEMM of `experts` F32 columns, no
    /// epilogue and no grouping), then the selection by `router` and the
    /// rank's route metadata, as [`Backend::moe_route`] states them
    /// (`routes`, `route_tile`, `usage`, `need`), whose logits the op also
    /// writes. `projection.m_capacity` bounds the rows and is no math: it
    /// holds at least `need.capacity`. How the projection splits K and how
    /// its partials reach the route are the backend's, which names them in
    /// its implementation and reports the weight's layout
    /// (`OpInfo::weight_layouts`). The op writes route metadata, which
    /// depends on every route of the launch, so its claims, and the claims
    /// in `need.exact`, are about the routed operation
    /// (`Scope::RoutedOperation`): one-row bits keep each token's logits,
    /// expert ids and weights equal to those of the token launched alone,
    /// while the metadata is the launch's own.
    ///
    /// Launches read `Counts::rows`, the token rows, alone.
    #[allow(clippy::too_many_arguments)]
    fn moe_projected_route(
        &self,
        projection: &GemmQuery,
        router: &RowQuery,
        routes: &MoeQuery,
        rank: u32,
        route_tile: u32,
        usage: OperationUse,
        need: Need,
    ) -> Result<Prepared<ProjectedRouteBuffers<View>>, Refused> {
        let _ = (projection, router, routes, rank, route_tile, usage, need);
        Err(Refused::Unsupported)
    }

    /// Each token's selection by a router (`RowQuery::Router`) alone.
    ///
    /// Launches read `Counts::rows` alone.
    fn moe_router(
        &self,
        op: &RowQuery,
        need: Need,
    ) -> Result<Prepared<RouterBuffers<View>>, Refused> {
        let _ = (op, need);
        Err(Refused::Unsupported)
    }

    /// Sqrt-softplus selection with one of two biases per token, including
    /// the F32 scores used to select. Launches read `Counts::rows` alone;
    /// every live U8 bias selector is 0 or 1. Logits and biases are finite.
    fn moe_row_bias_router(
        &self,
        op: &RowBiasRouterQuery,
        need: Need,
    ) -> Result<Prepared<RowBiasRouterBuffers<View>>, Refused> {
        let _ = (op, need);
        Err(Refused::Unsupported)
    }

    /// Row-bias selection and rank-local route metadata, with the same
    /// ordering, padding, live-count and routed-operation claims as
    /// `moe_route`. Launches read `Counts::rows` alone; live selectors
    /// are 0 or 1 and logits and biases are finite.
    fn moe_row_bias_route(
        &self,
        router: &RowBiasRouterQuery,
        routes: &MoeQuery,
        rank: u32,
        route_tile: u32,
        usage: OperationUse,
        need: Need,
    ) -> Result<Prepared<RowBiasRouteBuffers<View>>, Refused> {
        let _ = (router, routes, rank, route_tile, usage, need);
        Err(Refused::Unsupported)
    }

    /// Rank `rank`'s shared-expert gate combine
    /// (`MoeQuery::SharedExpertGateCombine`) over the routes of `routes`
    /// (`MoeQuery::LocalRoutes`).
    ///
    /// Launches read `Counts::rows`, the token rows, alone; routed views hold
    /// `top_k` rows per token.
    fn moe_shared_expert_combine(
        &self,
        op: &MoeQuery,
        routes: &MoeQuery,
        rank: u32,
        need: Need,
    ) -> Result<Prepared<SharedCombineBuffers<View>>, Refused> {
        let _ = (op, routes, rank, need);
        Err(Refused::Unsupported)
    }

    /// The combine of each token's routed rows (`ExpertCombineOp`).
    ///
    /// Launches read `Counts::rows`, the token rows, alone; routed views hold
    /// `top_k` rows per token.
    fn moe_expert_combine(
        &self,
        op: &ExpertCombineOp,
        need: Need,
    ) -> Result<Prepared<CombineBuffers<View>>, Refused> {
        let _ = (op, need);
        Err(Refused::Unsupported)
    }

    /// Rank `rank`'s routed sum (`MoeQuery::LocalRoutedSum`).
    ///
    /// Launches read `Counts::rows`, the token rows, alone; routed views hold
    /// `top_k` rows per token.
    fn moe_routed_sum(
        &self,
        op: &MoeQuery,
        rank: u32,
        need: Need,
    ) -> Result<Prepared<RoutedSumBuffers<View>>, Refused> {
        let _ = (op, rank, need);
        Err(Refused::Unsupported)
    }

    /// One projection of a SwiGLU MLP: the routed experts' gate/up or down
    /// over their routed rows (`op.gemm.grouped`), with the SwiGLU or
    /// clamped SwiGLU epilogue and the route weight where `op` places it,
    /// or a dense MLP's on the same engines. `need.capacity` is the most
    /// token rows a launch serves; a grouped launch reads `top_k` routed
    /// rows per token, in the op's route tile
    /// (`PreparedExperts::route_tile`). A grouped `op` states its routed
    /// rows without padding (`GatedMlpProjectionQuery::logical`), which a
    /// backend refuses otherwise; the prepared op reports the padded rows
    /// its routed views hold (`PreparedExperts::padded_rows`). `usage` is
    /// the step's use: a backend whose entry choice depends on how the step
    /// runs reads it, such as entries that sum in one-token order for
    /// verify steps.
    ///
    /// Launches read `Counts::rows`, the token rows, alone.
    fn gated_mlp_projection(
        &self,
        op: &GatedMlpProjectionQuery,
        usage: OperationUse,
        need: Need,
    ) -> Result<Box<dyn PreparedExperts>, Refused> {
        let _ = (op, usage, need);
        Err(Refused::Unsupported)
    }

    /// Quantized routed experts with separate stored and loop K and route
    /// weights before down. Gate/up produces route-indexed BF16
    /// rows, then the shared ceil quantizer produces the output codes and
    /// scales. Gate/up input codes are token-indexed; down inputs are
    /// route-indexed. Payloads satisfy the contract's exact-BF16 admission.
    /// Launches read `Counts::rows`, the token count, alone. Sorted route
    /// entries are -1 or less than the live token count times top-k.
    /// Gate/up requires metadata covering every live top-k slot: its
    /// quantizer consumes all such rows. Expert-parallel metadata that
    /// omits non-local routes is unsupported by the row-bias route API.
    fn chained_mlp_projection(
        &self,
        op: &ChainedMlpProjectionQuery,
        usage: OperationUse,
        need: Need,
    ) -> Result<Box<dyn PreparedExperts>, Refused> {
        let _ = (op, usage, need);
        Err(Refused::Unsupported)
    }

    // Collectives: all-reduce, all-gather, row exchange.

    /// An all-reduce that rank `rank` launches over up to `need.capacity`
    /// rows; `op` describes one row (`elements` is the row's).
    ///
    /// Launches read `Counts::rows` alone.
    fn all_reduce(
        &self,
        op: &CommQuery,
        rank: u32,
        need: Need,
    ) -> Result<Prepared<AllReduceBuffers<View>>, Refused> {
        let _ = (op, rank, need);
        Err(Refused::Unsupported)
    }

    /// A fused all-reduce (`CommQuery::AllReduce` with `fused_residual` and
    /// `fused_rmsnorm`) that rank `rank` launches at one use: `op`
    /// describes one row, `input` how the partials are laid out. The use
    /// (its phase, capacity and whether it runs in a graph) chooses the
    /// backend's entries, as for projections; `need.capacity` is the use's
    /// and `need.exact` the exactness its rows need. A model prepares one
    /// op per use it launches.
    ///
    /// Launches read `Counts::rows` alone.
    fn fused_all_reduce(
        &self,
        op: &CommQuery,
        input: ReduceInput,
        usage: OperationUse,
        rank: u32,
        need: Need,
    ) -> Result<Prepared<FusedAllReduceBuffers<View>>, Refused> {
        let _ = (op, input, usage, rank, need);
        Err(Refused::Unsupported)
    }

    /// A hyper-connection all-reduce (`AllReduceInjectNorm`) that rank `rank`
    /// launches at one use, of `input` partials (BF16 or F32) and grouped
    /// norm weights of dtype `weight`, as the model stores them. The use
    /// chooses the backend's entries as for `fused_all_reduce`. Every rank
    /// computes the same bits; whether a row's bits also hold whatever rows
    /// share the launch is the op's guarantees (`OpInfo::guarantees`).
    ///
    /// In a graph use a launch may run a larger bucket of prepared rows
    /// than `Counts::rows`, within the capacity, as a graph captured at that
    /// bucket does: `Launch::rows` names the bucket, whose rows every view
    /// holds, and the device live count (`live_rows`), at most the step's
    /// rows, bounds the rows it writes.
    ///
    /// Launches read `Counts::rows` alone.
    fn hc_all_reduce(
        &self,
        op: &SigmoidMeanHyperConnectionQuery,
        input: DType,
        weight: DType,
        usage: OperationUse,
        rank: u32,
        need: Need,
    ) -> Result<Prepared<HcAllReduceBuffers<View>>, Refused> {
        let _ = (op, input, weight, usage, rank, need);
        Err(Refused::Unsupported)
    }

    /// An all-gather (`CommQuery::AllGather`) that rank `rank` launches in
    /// steps of up to `need.capacity` rows. Every launch moves each rank's
    /// whole block of `elements_per_rank`, whatever its rows, and the blocks
    /// land rank-major, rank `q`'s at element `q * elements_per_rank` of a
    /// receiving rank's output. `target` names the one rank that receives;
    /// `None` gathers onto every rank. A context-parallel group's gather is
    /// an op of the context-parallel family, not this one.
    ///
    /// Launches read `Counts::rows` alone.
    fn all_gather(
        &self,
        op: &CommQuery,
        rank: u32,
        target: Option<u32>,
        need: Need,
    ) -> Result<Prepared<GatherBuffers<View>>, Refused> {
        let _ = (op, rank, target, need);
        Err(Refused::Unsupported)
    }

    /// A row gather of sharded rows onto rank `target`, which rank `rank`
    /// launches over up to `need.capacity` rows (`ShardedRows`). With
    /// indices, every index is below the capacity.
    ///
    /// Launches read `Counts::rows` alone.
    fn gather_rows(
        &self,
        op: &ShardedRows,
        rank: u32,
        target: u32,
        need: Need,
    ) -> Result<Prepared<RowGatherBuffers<View>>, Refused> {
        let _ = (op, rank, target, need);
        Err(Refused::Unsupported)
    }

    /// A row exchange of sharded rows onto their owners
    /// (`collective::owned_rows`), which rank `rank` launches over up to
    /// `need.capacity` rows (`ShardedRows`). Each rank's output holds every
    /// row's offset; only its owned rows are written.
    ///
    /// Launches read `Counts::rows` alone.
    fn exchange_rows(
        &self,
        op: &ShardedRows,
        rank: u32,
        need: Need,
    ) -> Result<Prepared<GatherBuffers<View>>, Refused> {
        let _ = (op, rank, need);
        Err(Refused::Unsupported)
    }

    // Context parallelism: the merge of a group's attention partials, key
    // localization, one group rank's paged partials, the owner-routed KV and
    // indexer key writes, and the transfers between a group's members
    // (`crate::context_parallel`). A group's selected partials and top-k
    // merges are the gated sparse and pooled indexer families' ops.

    /// The rank-ordered merge of a group's attention partials
    /// (`ContextParallelQuery::PartialMerge`).
    ///
    /// Launches read `Counts::rows` alone. The buffers' segment layout
    /// (`rank_rows`, `input_heads`, `head_offset`) must hold every row's
    /// heads in each of the `ranks` segments, and `rank_rows` the step's
    /// rows.
    fn partial_merge(
        &self,
        op: &ContextParallelQuery,
        need: Need,
    ) -> Result<Prepared<PartialMergeBuffers<View>>, Refused> {
        let _ = (op, need);
        Err(Refused::Unsupported)
    }

    /// The localization of key lists (`ContextParallelQuery::LocalizeKeys`)
    /// to group rank `group_rank`, below the query's `ranks`.
    ///
    /// Launches read `Counts::rows` alone.
    fn localize_keys(
        &self,
        op: &ContextParallelQuery,
        group_rank: u32,
        need: Need,
    ) -> Result<Prepared<LocalizeKeysBuffers<View>>, Refused> {
        let _ = (op, group_rank, need);
        Err(Refused::Unsupported)
    }

    /// The partials of paged attention over the keys context-parallel group
    /// rank `group_rank` of `group_size` owns (`group_size` 1 owns every
    /// key).
    ///
    /// Launches read `Counts::rows` and `Counts::items`, the rows of the block
    /// tables, at least one. The host keeps every row's item below
    /// `Counts::items`, every position below the op's `max_context` and every
    /// block table entry below the cache's pages.
    fn paged_attention_partials(
        &self,
        op: &PagedPartialsOp,
        group_rank: u32,
        group_size: u32,
        need: Need,
    ) -> Result<Prepared<PagedPartialsBuffers<View>>, Refused> {
        let _ = (op, group_rank, group_size, need);
        Err(Refused::Unsupported)
    }

    /// The gated Q/K norm, M-RoPE and KV write with each row's K and V on
    /// the group rank that owns its page.
    ///
    /// Launches read `Counts::rows`, `Counts::items` (the rows of the block
    /// tables, at least one) and `Counts::positions`, the rotary table's
    /// rows: one more than the largest rotary position any launch reads, its
    /// rows' `rope_positions`. The host keeps every row's item below
    /// `Counts::items`, every local position below the op's `max_context` and
    /// every block table entry below the cache's pages.
    fn owned_kv_write(
        &self,
        op: &OwnedKvWriteOp,
        need: Need,
    ) -> Result<Prepared<GatedKvWriteBuffers<View>>, Refused> {
        let _ = (op, need);
        Err(Refused::Unsupported)
    }

    /// The sparse indexer's raw-key ring and block-key write at group rank
    /// `group_rank` of the query's `ranks`, which stores only the block keys
    /// of the pages it owns.
    ///
    /// Launches read `Counts::rows`, `Counts::items` (the rows of the block
    /// tables, slot ids and anchors, at least one) and `Counts::positions`,
    /// the rotary table's rows: one more than the largest rotary position any
    /// launch reads, its rows' `rope_positions` and the positions an earlier
    /// step stored in the key rings for a block a row completes. The host
    /// keeps every row's item below `Counts::items`, every position below the
    /// op's `max_context`, every block table entry below the rank's local
    /// pages and every slot id below its rings.
    fn owned_key_write(
        &self,
        op: &OwnedKeyWriteOp,
        group_rank: u32,
        need: Need,
    ) -> Result<Prepared<SparseKeyWriteBuffers<View>>, Refused> {
        let _ = (op, group_rank, need);
        Err(Refused::Unsupported)
    }

    /// A transfer between a group's members that group rank `group_rank`
    /// launches in steps of up to `need.capacity` rows; every launch moves
    /// the op's whole block.
    ///
    /// Launches read `Counts::rows` alone.
    fn group_transfer(
        &self,
        op: &GroupTransferOp,
        group_rank: u32,
        need: Need,
    ) -> Result<Prepared<GroupTransferBuffers<View>>, Refused> {
        let _ = (op, group_rank, need);
        Err(Refused::Unsupported)
    }

    // Multimodal encoder stages (`crate::multimodal`).

    /// An encoder row pass: a norm, the position add, a permute, SwiGLU or
    /// GELU rows, a residual add, an im2col unfold or column padding.
    ///
    /// Launches read `Counts::rows` alone. The method takes the variants whose
    /// operands fill its record (`multimodal::EncoderStage::of`; the module
    /// lists each variant's roles) and refuses the others; the query's
    /// `capacity` is the backend's, `need.capacity` the model's. The host keeps
    /// a position add's indices below the table's rows and a permute's below
    /// the input rows.
    fn encoder_rows(
        &self,
        op: &MultimodalQuery,
        need: Need,
    ) -> Result<Prepared<EncoderRowBuffers<View>>, Refused> {
        let _ = (op, need);
        Err(Refused::Unsupported)
    }

    /// An in-place pass over packed q, k, v rows: the rotary pair rewrite
    /// or the q/k RMSNorm.
    ///
    /// Launches read `Counts::rows` alone. The method takes the variants whose
    /// operands fill its record (`multimodal::EncoderStage::of`; the module
    /// lists each variant's roles) and refuses the others; the query's
    /// `capacity` is the backend's, `need.capacity` the model's.
    fn encoder_heads(
        &self,
        op: &MultimodalQuery,
        need: Need,
    ) -> Result<Prepared<EncoderHeadBuffers<View>>, Refused> {
        let _ = (op, need);
        Err(Refused::Unsupported)
    }

    /// Contiguous attention whose logits scale by `scale`.
    ///
    /// Launches read `Counts::rows` alone. The method takes the variants whose
    /// operands fill its record (`multimodal::EncoderStage::of`; the module
    /// lists each variant's roles) and refuses the others; the query's
    /// `capacity` is the backend's, `need.capacity` the model's. The sequences
    /// come from the `live_sequences` role, a host count, not from
    /// `Counts::items`; the host keeps every sequence offset (`cu_seqlens`) at
    /// most `Counts::rows`.
    fn encoder_attention(
        &self,
        op: &MultimodalQuery,
        scale: f32,
        need: Need,
    ) -> Result<Prepared<EncoderAttentionBuffers<View>>, Refused> {
        let _ = (op, scale, need);
        Err(Refused::Unsupported)
    }

    /// An encoder GEMM.
    ///
    /// Launches read `Counts::rows` alone. The method takes the variants whose
    /// operands fill its record (`multimodal::EncoderStage::of`; the module
    /// lists each variant's roles) and refuses the others; the query's
    /// `capacity` is the backend's, `need.capacity` the model's.
    fn encoder_gemm(
        &self,
        op: &MultimodalQuery,
        need: Need,
    ) -> Result<Prepared<EncoderGemmBuffers<View>>, Refused> {
        let _ = (op, need);
        Err(Refused::Unsupported)
    }

    /// The prefill embedding rows the encoders' arena overrides.
    ///
    /// Launches read `Counts::rows` alone. The method takes the variants whose
    /// operands fill its record (`multimodal::EncoderStage::of`; the module
    /// lists each variant's roles) and refuses the others; the query's
    /// `capacity` is the backend's, `need.capacity` the model's. A source
    /// outside the arena keeps its row, so sources need no bound.
    fn embedding_override(
        &self,
        op: &MultimodalQuery,
        need: Need,
    ) -> Result<Prepared<EmbeddingOverrideBuffers<View>>, Refused> {
        let _ = (op, need);
        Err(Refused::Unsupported)
    }

    /// Decode dense MXFP8 rows to BF16 with one RNE rounding.
    ///
    /// Launches read `Counts::rows` and grid `Counts::padded_rows` when
    /// nonzero, otherwise `Counts::rows`. The host keeps the device
    /// `live_rows` at most `Counts::rows`, which is at most `Need::capacity`,
    /// and binds nonoverlapping planes at their natural alignments.
    fn decode_mxfp8_rows(
        &self,
        op: &Mxfp8Rows,
        need: Need,
    ) -> Result<Prepared<Mxfp8RowsBuffers<View>>, Refused> {
        let _ = (op, need);
        Err(Refused::Unsupported)
    }

    // Row copies: the stream broadcast, the column slice and the gather by
    // index (`crate::row_copy`), which move element bytes and compute
    // nothing.

    /// Each live row copied into every stream.
    ///
    /// Launches read `Counts::rows` alone; the host keeps the device live
    /// count at most `Counts::rows` and binds every view at the op's
    /// `alignment_bytes`.
    fn broadcast_streams(
        &self,
        op: &StreamBroadcast,
        need: Need,
    ) -> Result<Prepared<RowCopyBuffers<View>>, Refused> {
        let _ = (op, need);
        Err(Refused::Unsupported)
    }

    /// The columns of each live row from the slice's offset.
    ///
    /// Launches read `Counts::rows` alone; the host keeps the device live
    /// count at most `Counts::rows` and binds every view at the op's
    /// `alignment_bytes`.
    fn column_slice(
        &self,
        op: &ColumnSlice,
        need: Need,
    ) -> Result<Prepared<RowCopyBuffers<View>>, Refused> {
        let _ = (op, need);
        Err(Refused::Unsupported)
    }

    /// The source row each row index names, for the first `count` rows.
    ///
    /// Launches gather at most `Counts::rows` rows and grid
    /// `Counts::padded_rows` rows when it is nonzero (a step that pads its
    /// grid to a row bucket), else `Counts::rows`; the host keeps the device
    /// count at most `Counts::rows`, every index of a gathered row below the
    /// source's rows, and binds every view at the op's `alignment_bytes`.
    fn gather_by_index(
        &self,
        op: &GatherByIndex,
        need: Need,
    ) -> Result<Prepared<GatherByIndexBuffers<View>>, Refused> {
        let _ = (op, need);
        Err(Refused::Unsupported)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context_parallel::GroupTransfer;
    use crate::contract::{
        ActivationFormat, ArgmaxLayout, AttentionPhase, ConvActivation, DType, Epilogues,
        GatedAttentionQuery, GemmQuery, RecurrentPhase, RmsNormKind, RouterKind, Target,
        WeightFormat,
    };
    use crate::gated_mlp::GatedMlpProjection;
    use crate::latent_attention::{HeadWeightLayout, LatentAttentionOutput, LatentKvFormat};
    use crate::per_channel_decay_delta_rule::PerChannelDecayDeltaRuleQuery;
    use crate::precision::{ArithmeticSpec, DecodeOrder, ExecutionPhase, OperationKind, Rounding};
    use crate::rmsnorm::RmsNormNumerics;

    struct Empty;

    impl Backend for Empty {
        fn name(&self) -> &str {
            "empty"
        }
    }

    fn decode(capacity: u32) -> OperationUse {
        OperationUse {
            phase: ExecutionPhase::Decode,
            capacity,
            graph: true,
        }
    }

    fn norm() -> RmsNormRequirement {
        RmsNormRequirement::bf16(
            64,
            1e-6,
            RmsNormKind::Plain,
            false,
            RmsNormNumerics::TREE_256,
        )
    }

    fn hc() -> SigmoidMeanHyperConnectionQuery {
        SigmoidMeanHyperConnectionQuery::Mix {
            streams: 4,
            width: 256,
            capacity: 4,
        }
    }

    fn sinkhorn() -> SinkhornHyperConnectionQuery {
        SinkhornHyperConnectionQuery::Expand {
            streams: 4,
            width: 256,
            capacity: 4,
        }
    }

    fn expand_mix() -> crate::sinkhorn_expand_mix::SinkhornExpandMixQuery {
        crate::sinkhorn_expand_mix::SinkhornExpandMixQuery {
            mix: crate::sinkhorn_hyper_connection::SinkhornMix {
                streams: 4,
                width: 256,
                iterations: 20,
                eps_bits: 1e-6_f32.to_bits(),
                norm_eps_bits: 1e-5_f32.to_bits(),
                order: crate::sinkhorn_hyper_connection::SinkhornOrder::RowSoftmaxThenColumnFirst,
                placement:
                    crate::sinkhorn_hyper_connection::HyperConnectionPlacement::LaggedSinglePass,
                fn_dtype: DType::F32,
            },
        }
    }

    fn ngram() -> NgramQuery {
        NgramQuery::PleGate {
            streams: 4,
            width: 256,
            capacity: 4,
        }
    }

    fn spec() -> OperationSpec {
        OperationSpec {
            id: "projection".to_owned(),
            kind: OperationKind::Dense,
            target: Target::Sm120f,
            weights: Vec::new(),
            geometry: None,
            activations: Vec::new(),
            arithmetic: ArithmeticSpec {
                weight_operand: DType::Bf16,
                accumulator: DType::F32,
                output: DType::Bf16,
                decode_order: DecodeOrder::ScaleThenCode,
                rounding: Rounding::NearestEven,
                numerical_contract: "projection".to_owned(),
            },
            epilogues: Epilogues::default(),
            uses: vec![decode(4)],
        }
    }

    fn sparse(capacity: u32) -> SparseAttentionQuery {
        SparseAttentionQuery::BlockScores {
            heads: 16,
            head_dim: 128,
            max_blocks: 1024,
            capacity,
        }
    }

    fn recurrent() -> RecurrentQuery {
        RecurrentQuery::CausalConv1d {
            channels: 256,
            kernel: 4,
            dilation: 1,
            activation: ConvActivation::Silu,
            phase: RecurrentPhase::Decode,
            capacity: 4,
        }
    }

    fn channel_decay() -> PerChannelDecayDeltaRuleQuery {
        PerChannelDecayDeltaRuleQuery::DeltaRule {
            heads: 2,
            head_dim: 128,
            gate: crate::per_channel_decay_delta_rule::DecayGate::Softplus,
            phase: RecurrentPhase::Decode,
            capacity: 4,
        }
    }

    fn latent() -> LatentAttentionQuery {
        LatentAttentionQuery::Attention {
            heads: 16,
            latent: 512,
            max_keys: 64,
            softmax_scale_bits: 0.0625_f32.to_bits(),
            page_size: 64,
            kv: LatentKvFormat::Bf16,
            phase: AttentionPhase::Decode,
            output: LatentAttentionOutput::Normalized,
            sink: false,
            tail_rope_dims: 0,
            second_fp4_keys: false,
            capacity: 8,
        }
    }

    fn indexer() -> PooledIndexerQuery {
        PooledIndexerQuery::CompactSelection {
            width: 2051,
            capacity: 32,
        }
    }

    fn router() -> RowQuery {
        RowQuery::Router {
            experts: 64,
            top_k: 4,
            kind: RouterKind::SoftmaxTopK { renormalize: true },
        }
    }

    fn routes() -> MoeQuery {
        MoeQuery::LocalRoutes {
            experts: 64,
            local_experts: 16,
            top_k: 4,
            capacity: 8,
        }
    }

    fn gated_mlp() -> GatedMlpProjectionQuery {
        GatedMlpProjectionQuery {
            gemm: GemmQuery {
                m_capacity: 8,
                n: 512,
                k: 256,
                weight: WeightFormat::Bf16,
                activation: ActivationFormat::Bf16,
                output: DType::Bf16,
                epilogues: Epilogues::default(),
                grouped: None,
            },
            projection: GatedMlpProjection::Down,
            swiglu_clamp: None,
            route_weight: None,
        }
    }

    fn reduce() -> CommQuery {
        CommQuery::AllReduce {
            ranks: 2,
            elements: 64,
            dtype: DType::F32,
            fused_residual: false,
            fused_rmsnorm: None,
        }
    }

    fn sharded() -> ShardedRows {
        ShardedRows {
            ranks: 2,
            shard: 64,
            dtype: DType::F32,
        }
    }

    fn encoder() -> MultimodalQuery {
        MultimodalQuery::GeluRowsF32 {
            width: 64,
            capacity: 4,
        }
    }

    /// One test per [`Backend`] method, named after it, that the default
    /// refuses with [`Refused::Unsupported`], and the list of the methods
    /// tested.
    macro_rules! refuse_by_default {
        ($($method:ident: $call:expr;)*) => {
            mod refuses_by_default {
                use super::*;

                $(
                    #[test]
                    fn $method() {
                        assert_eq!($call.err(), Some(Refused::Unsupported));
                    }
                )*
            }

            const TESTED: &[&str] = &[$(stringify!($method)),*];
        };
    }

    refuse_by_default! {
        rmsnorm: Empty.rmsnorm(&norm(), Need::rows(4));
        embedding_gather: Empty.embedding_gather(
            &RowQuery::EmbeddingGather { vocab: 64, width: 32 },
            0,
            Need::rows(4),
        );
        replicated_embedding: Empty.replicated_embedding(
            &crate::replicated_embedding::ReplicatedEmbedding {
                vocab: 64, width: 32, ranks: 2, streams: 4,
            },
            0,
            Need::rows(4),
        );
        logits_top_k: Empty.logits_top_k(&RowQuery::LogitsArgmax { vocab: 64 }, Need::rows(4));
        sharded_argmax_partial: Empty.sharded_argmax_partial(
            &RowQuery::ShardedArgmaxPartial {
                vocab_per_rank: 64,
                ranks: 2,
                logits: DType::F32,
            },
            1,
            Need::rows(4),
        );
        sharded_argmax_final: Empty.sharded_argmax_final(
            &RowQuery::ShardedArgmaxFinal {
                ranks: 2,
                pairs: ArgmaxLayout::RankMajor { stride: 4 },
            },
            Need::rows(4),
        );
        hc_grouped_norm: Empty.hc_grouped_norm(&hc(), DType::Bf16, Need::rows(4));
        hc_bottleneck: Empty.hc_bottleneck(&hc(), Need::rows(4));
        hc_mix: Empty.hc_mix(&hc(), Need::rows(4));
        hc_inject: Empty.hc_inject(&hc(), Need::rows(4));
        ngram_hash_gather: Empty.ngram_hash_gather(&ngram(), 2, 0, Need::rows(4));
        ple_gate: Empty.ple_gate(&ngram(), Need::rows(4));
        sinkhorn_mix_collapse: Empty.sinkhorn_mix_collapse(&sinkhorn(), Need::rows(4));
        sinkhorn_expand_mix: Empty.sinkhorn_expand_mix(&expand_mix(), Need::rows(4));
        sinkhorn_round_expand_mix: Empty.sinkhorn_round_expand_mix(&expand_mix(), Need::rows(4));
        sinkhorn_round_expand_mix_norm: Empty.sinkhorn_round_expand_mix_norm(
            &crate::sinkhorn_expand_mix::SinkhornRoundNormQuery {
                mixer: expand_mix(),
                norm: crate::rmsnorm_quantize::RmsNormQuantizeQuery {
                    norm: RmsNormRequirement::bf16(
                        256, 1e-6, RmsNormKind::Plain, false, RmsNormNumerics::TREE_256,
                    ),
                    amax_floor_bits: 1e-4_f32.to_bits(),
                },
            },
            Need::rows(4),
        );
        sinkhorn_lagged_mix_collapse: Empty.sinkhorn_lagged_mix_collapse(&sinkhorn(), Need::rows(4));
        sinkhorn_lagged_prefill: Empty.sinkhorn_lagged_prefill(&sinkhorn(), Need::rows(4));
        sinkhorn_mix_collapse_norm: Empty.sinkhorn_mix_collapse_norm(&sinkhorn(), Need::rows(4));
        sinkhorn_expand: Empty.sinkhorn_expand(&sinkhorn(), Need::rows(4));
        sinkhorn_head: Empty.sinkhorn_head(&sinkhorn(), Need::rows(4));
        sinkhorn_weighted_head: Empty.sinkhorn_weighted_head(&sinkhorn(), Need::rows(4));
        sinkhorn_reduce_expand_mix_norm: Empty.sinkhorn_reduce_expand_mix_norm(
            &sinkhorn(),
            0,
            Need::rows(4),
        );
        mxfp8_fold_gemm: Empty.mxfp8_fold_gemm(
            &crate::mxfp8_gemm::Mxfp8FoldGemmQuery {
                m_capacity: 4, n: 64, k: 160, groups: 2,
                activation_row_stride: 320, activation_group_stride: 160,
                output: DType::F32,
                execution: crate::precision::execution::BlockScaledExecution::ExactDecodedBf16Mma,
            }, Need::rows(4),
        );
        mxfp8_prefill_gemm: Empty.mxfp8_prefill_gemm(
            &crate::mxfp8_gemm::Mxfp8FoldGemmQuery {
                m_capacity: 4, n: 64, k: 160, groups: 1,
                activation_row_stride: 160, activation_group_stride: 160,
                output: DType::F32,
                execution: crate::precision::execution::BlockScaledExecution::NativeBlockScaledMma,
            }, Need::rows(4),
        );
        mxfp8_fold_gemm_pair: Empty.mxfp8_fold_gemm_pair(
            &crate::mxfp8_gemm::Mxfp8FoldGemmQuery {
                m_capacity: 4, n: 64, k: 160, groups: 1,
                activation_row_stride: 160, activation_group_stride: 160,
                output: DType::F32,
                execution: crate::precision::execution::BlockScaledExecution::NativeBlockScaledMma,
            }, Need::rows(4),
        );
        mxfp8_fold_gemm_stream: Empty.mxfp8_fold_gemm_stream(
            &crate::mxfp8_gemm::Mxfp8FoldGemmQuery {
                m_capacity: 4, n: 64, k: 160, groups: 1,
                activation_row_stride: 160, activation_group_stride: 160,
                output: DType::F32,
                execution: crate::precision::execution::BlockScaledExecution::ExactDecodedBf16Mma,
            }, false, Need::rows(4),
        );
        block_quantize: Empty.block_quantize(
            &crate::precision::quantizer::BlockQuantizeQuery {
                width: 64, input: DType::Bf16,
                format: crate::precision::quantizer::BlockQuantization::E4M3_CEIL32,
                output: crate::precision::quantizer::QuantizeOutput::Codes,
            }, Need::rows(4),
        );
        rmsnorm_quantize: Empty.rmsnorm_quantize(
            &crate::rmsnorm_quantize::RmsNormQuantizeQuery {
                norm: norm(), amax_floor_bits: 1e-4_f32.to_bits(),
            }, Need::rows(4),
        );
        silu_mul_quantize: Empty.silu_mul_quantize(
            &crate::gated_mlp::quantized::SiluMulQuantizeQuery {
                width: 64, clamp: crate::gated_mlp::SwigluClamp::symmetric(10.0),
                amax_floor_bits: 1e-4_f32.to_bits(),
            }, Need::rows(4),
        );
        projection: Empty.projection(&spec(), decode(4), Need::rows(4));
        attention: Empty.attention(&crate::attention::tests::decode(), Need::rows(4));
        rope: Empty.rope(
            &RopeOp {
                rope: crate::rope_kv::tests::rope(false),
                heads: 4,
                positions: DType::I32,
            },
            Need::rows(4),
        );
        rope_kv_write: Empty.rope_kv_write(&crate::rope_kv::tests::paged(), Need::rows(4));
        ring_write: Empty.ring_write(
            &RingWriteOp {
                query: RowQuery::RingWrite {
                    qk_dim: 128,
                    v_dim: 128,
                    kv_heads: 2,
                    ring_size: 128,
                    kv_dtype: DType::Bf16,
                },
                positions: DType::I32,
                single_sequence: false,
            },
            Need::rows(4),
        );
        gated_kv_write: Empty.gated_kv_write(&crate::gated_sparse::tests::kv_write(), Need::rows(4));
        output_gate: Empty.output_gate(
            &OutputGateOp {
                query: GatedAttentionQuery::OutputGate {
                    heads: 8,
                    head_dim: 256,
                    capacity: 4,
                },
                kv_heads: 1,
            },
            Need::rows(4),
        );
        sparse_index_query: Empty.sparse_index_query(&sparse(4), Need::rows(4));
        sparse_key_write: Empty.sparse_key_write(
            &crate::gated_sparse::tests::key_write(),
            Need::rows(4),
        );
        sparse_block_scores: Empty.sparse_block_scores(
            &BlockScoresOp {
                query: sparse(4),
                block: 4,
                page_size: 64,
                max_context: 4096,
            },
            decode(4),
            Need::rows(4),
        );
        sparse_block_top_k: Empty.sparse_block_top_k(&sparse(4), Need::rows(4));
        sparse_attention: Empty.sparse_attention(
            &SparseAttentionOp {
                query: sparse(4),
                max_context: 4096,
            },
            Need::rows(4),
        );
        selected_attention_partials: Empty.selected_attention_partials(
            &SelectedPartialsOp {
                query: ContextParallelQuery::SelectedAttentionPartials {
                    q_heads: 8,
                    kv_heads: 1,
                    head_dim: 128,
                    page: 64,
                    block: 4,
                    top_blocks: 64,
                    kv: DType::Bf16,
                    phase: AttentionPhase::Decode,
                    capacity: 4,
                },
                softmax_scale: 0.125,
                max_context: 4096,
            },
            0,
            2,
            Need::rows(4),
        );
        latent_query_rope: Empty.latent_query_rope(
            &crate::latent_glue::LatentQueryRope { heads: 2, latent: 512, rotary: 64 }, Need::rows(32)
        );
        latent_staged_window_keys: Empty.latent_staged_window_keys(
            &crate::latent_staged_window_keys::LatentStagedWindowKeys { window: 128, ring_slots: 136, staged_slots: 256 }, Need::rows(32)
        );
        latent_window_stage: Empty.latent_window_stage(
            &crate::latent_window_stage::LatentWindowStage { kv: crate::latent_attention::LatentKvFormat::Fp8E4M3Ue8M0Block32, latent: 512, window: 128, ring_slots: 136, staged_slots: 512 }, Need::rows(4)
        );
        latent_window_write: Empty.latent_window_write(
            &crate::latent_glue::LatentWindowWrite { kv: crate::latent_attention::LatentKvFormat::Fp8E4M3Ue8M0Block32, latent: 512, rotary: 64, epsilon_bits: 1e-20_f32.to_bits(), addressing: crate::latent_glue::WindowAddressing::Circular { slots: 136 } }, Need::rows(32)
        );
        latent_window_write_fused: Empty.latent_window_write_fused(
            &crate::latent_glue::LatentWindowWrite { kv: crate::latent_attention::LatentKvFormat::Fp8E4M3Ue8M0Block32, latent: 512, rotary: 64, epsilon_bits: 1e-20_f32.to_bits(), addressing: crate::latent_glue::WindowAddressing::Circular { slots: 136 } }, Need::rows(32)
        );
        latent_window_keys: Empty.latent_window_keys(
            &crate::latent_glue::LatentWindowKeys { window: 128, addressing: crate::latent_glue::WindowAddressing::Circular { slots: 136 } }, Need::rows(32)
        );
        latent_kv_write: Empty.latent_kv_write(
            &LatentAttentionQuery::KvWrite {
                latent: 512,
                page_size: 64,
                kv: LatentKvFormat::Bf16,
                capacity: 8,
            },
            Need::rows(4),
        );
        latent_attention: Empty.latent_attention(&latent(), Need::rows(4));
        latent_attention_sources: Empty.latent_attention_sources(&latent(), Need::rows(4));
        head_batched_gemm: Empty.head_batched_gemm(
            &HeadBatchedGemmQuery {
                heads: 16,
                n: 512,
                k: 256,
                layout: HeadWeightLayout::KByN,
                output: DType::Bf16,
                m_capacity: 8,
            },
            Need::rows(4),
        );
        gated_pooling: Empty.gated_pooling(
            &GatedPoolingQuery {
                group: 2, width: 512, norm_eps_bits: 1e-20f32.to_bits(),
                phase: RecurrentPhase::Prefill, capacity: 4,
            },
            Need::rows(4),
        );
        position_indexer_key_write: Empty.position_indexer_key_write(
            &PositionIndexerQuery { heads: 32, head_dim: 128, compression_ratio: 2,
                tail_rope_dims: 64, key_norm_eps_bits: 1e-20f32.to_bits(), max_top_k: 512,
                page_size: 64, candidates: crate::pooled_indexer::IndexerCandidates::All, capacity: 4 },
            Need::rows(4),
        );
        position_indexer_key_write_with_storage: Empty.position_indexer_key_write_with_storage(
            &PositionIndexerQuery { heads: 32, head_dim: 128, compression_ratio: 2,
                tail_rope_dims: 64, key_norm_eps_bits: 1e-20f32.to_bits(), max_top_k: 512,
                page_size: 64, candidates: crate::pooled_indexer::IndexerCandidates::All, capacity: 4 },
            crate::pooled_indexer::IndexerKeyFormat::Bf16,
            Need::rows(4),
        );
        compressed_kv_write: Empty.compressed_kv_write(
            &CompressedKvWriteQuery { width: 512, tail_rope_dims: 64, compression_ratio: 2,
                page_size: 64, storage: crate::compressed_kv::CompressedKvStorage::Packed, capacity: 4 },
            Need::rows(4),
        );
        indexer_key_write: Empty.indexer_key_write(&indexer(), 0, 1, Need::rows(4));
        indexer_select: Empty.indexer_select(&indexer(), 2048, 0, Need::rows(4));
        indexer_merge: Empty.indexer_merge(
            &ContextParallelQuery::TopKMergeBlocks {
                ranks: 2,
                group: 4,
                max_top_k: 2048,
                capacity: 2048,
            },
            2048,
            Need::rows(4),
        );
        indexer_compact: Empty.indexer_compact(&indexer(), Need::rows(4));
        position_indexer_query: Empty.position_indexer_query(&PositionIndexerQuery {
            heads: 32, head_dim: 128, compression_ratio: 2, tail_rope_dims: 64,
            key_norm_eps_bits: 1e-20_f32.to_bits(), max_top_k: 2048, page_size: 64,
            candidates: crate::pooled_indexer::IndexerCandidates::All, capacity: 4,
        }, Need::rows(4));
        position_indexer: Empty.position_indexer(&PositionIndexerQuery {
            heads: 32, head_dim: 128, compression_ratio: 2, tail_rope_dims: 64,
            key_norm_eps_bits: 1e-20_f32.to_bits(), max_top_k: 2048, page_size: 64,
            candidates: crate::pooled_indexer::IndexerCandidates::All, capacity: 4,
        }, 2048, 16392, Need::rows(4));
        position_indexer_with_storage: Empty.position_indexer_with_storage(&PositionIndexerQuery {
            heads: 32, head_dim: 128, compression_ratio: 2, tail_rope_dims: 64,
            key_norm_eps_bits: 1e-20_f32.to_bits(), max_top_k: 2048, page_size: 64,
            candidates: crate::pooled_indexer::IndexerCandidates::All, capacity: 4,
        }, IndexerKeyFormat::Bf16, 2048, 16392, Need::rows(4));
        indexer_candidate_blocks: Empty.indexer_candidate_blocks(&CandidateBlocksQuery {
            block_size: 8, top_k_blocks: 2048, capacity: 4,
        }, 16392, Need::rows(4));
        indexer_candidate_block_ids: Empty.indexer_candidate_block_ids(&CandidateBlocksQuery {
            block_size: 8, top_k_blocks: 2048, capacity: 4,
        }, 1 << 20, Need::rows(4));
        indexer_candidate_block_ids_runtime: Empty.indexer_candidate_block_ids_runtime(&CandidateBlocksQuery {
            block_size: 8, top_k_blocks: 2048, capacity: 4,
        }, 1024, 1 << 20, Need::rows(4));
        position_block_indexer: Empty.position_block_indexer(&PositionBlockIndexerQuery {
            indexer: PositionIndexerQuery {
                heads: 3, head_dim: 160, compression_ratio: 2, tail_rope_dims: 64,
                key_norm_eps_bits: 1e-20_f32.to_bits(), max_top_k: 7, page_size: 64,
                candidates: crate::pooled_indexer::IndexerCandidates::All, capacity: 4,
            }, block_size: 8, top_k_blocks: 2048,
        }, 7, 1 << 20, Need::rows(4));
        position_block_indexer_with_storage: Empty.position_block_indexer_with_storage(&PositionBlockIndexerQuery {
            indexer: PositionIndexerQuery {
                heads: 3, head_dim: 160, compression_ratio: 2, tail_rope_dims: 64,
                key_norm_eps_bits: 1e-20_f32.to_bits(), max_top_k: 7, page_size: 64,
                candidates: crate::pooled_indexer::IndexerCandidates::All, capacity: 4,
            }, block_size: 8, top_k_blocks: 2048,
        }, IndexerKeyFormat::Bf16, 7, 1 << 20, Need::rows(4));
        causal_conv1d: Empty.causal_conv1d(&recurrent(), 4, Need::rows(4));
        gated_delta_rule: Empty.gated_delta_rule(&recurrent(), 4, Need::rows(4));
        gated_rmsnorm: Empty.gated_rmsnorm(&recurrent(), Need::rows(4));
        short_conv: Empty.short_conv(&channel_decay().into(), 4, Need::rows(4));
        channel_decay_delta_rule: Empty.channel_decay_delta_rule(
            &channel_decay().into(),
            4,
            Need::rows(4),
        );
        moe_route: Empty.moe_route(&router(), &routes(), 0, 16, decode(8), Need::rows(8));
        moe_projected_route: Empty.moe_projected_route(
            &GemmQuery {
                n: 64,
                output: DType::F32,
                ..gated_mlp().gemm
            },
            &router(),
            &routes(),
            0,
            16,
            decode(8),
            Need::rows(8),
        );
        moe_router: Empty.moe_router(&router(), Need::rows(8));
        moe_row_bias_router: Empty.moe_row_bias_router(
            &RowBiasRouterQuery { experts: 128, top_k: 3, scaling_factor: 1.5 }, Need::rows(8)
        );
        moe_row_bias_route: Empty.moe_row_bias_route(
            &RowBiasRouterQuery { experts: 128, top_k: 3, scaling_factor: 1.5 },
            &routes(), 0, 16, decode(8), Need::rows(8)
        );
        moe_shared_expert_combine: Empty.moe_shared_expert_combine(
            &MoeQuery::SharedExpertGateCombine {
                width: 256,
                capacity: 8,
            },
            &routes(),
            0,
            Need::rows(8),
        );
        moe_expert_combine: Empty.moe_expert_combine(
            &ExpertCombineOp {
                experts: 64,
                top_k: 4,
                width: 256,
                route_weight: None,
            },
            Need::rows(8),
        );
        moe_routed_sum: Empty.moe_routed_sum(
            &MoeQuery::LocalRoutedSum {
                experts: 64,
                local_experts: 16,
                top_k: 4,
                width: 256,
                output: DType::F32,
                capacity: 8,
            },
            0,
            Need::rows(8),
        );
        gated_mlp_projection: Empty.gated_mlp_projection(&gated_mlp(), decode(8), Need::rows(8));
        chained_mlp_projection: Empty.chained_mlp_projection(
            &ChainedMlpProjectionQuery {
                projection: gated_mlp(),
                k: crate::moe_chained::ChainedK::new(gated_mlp().gemm.k).unwrap(),
            }, decode(8), Need::rows(8)
        );
        all_reduce: Empty.all_reduce(&reduce(), 0, Need::rows(4));
        fused_all_reduce: Empty.fused_all_reduce(
            &reduce(),
            ReduceInput::Rows,
            decode(4),
            0,
            Need::rows(4),
        );
        hc_all_reduce: Empty.hc_all_reduce(
            &hc(),
            DType::Bf16,
            DType::Bf16,
            decode(4),
            0,
            Need::rows(4),
        );
        all_gather: Empty.all_gather(
            &CommQuery::AllGather {
                ranks: 2,
                elements_per_rank: 64,
                dtype: DType::F32,
            },
            0,
            None,
            Need::rows(4),
        );
        gather_rows: Empty.gather_rows(&sharded(), 1, 0, Need::rows(4));
        exchange_rows: Empty.exchange_rows(&sharded(), 1, Need::rows(4));
        partial_merge: Empty.partial_merge(
            &ContextParallelQuery::PartialMerge {
                ranks: 2,
                heads: 8,
                width: 128,
                capacity: 4,
            },
            Need::rows(4),
        );
        localize_keys: Empty.localize_keys(
            &ContextParallelQuery::LocalizeKeys {
                ranks: 2,
                page: 64,
                width: 1,
                capacity: 4,
            },
            1,
            Need::rows(4),
        );
        paged_attention_partials: Empty.paged_attention_partials(
            &PagedPartialsOp {
                query: ContextParallelQuery::PagedAttentionPartials {
                    q_heads: 8,
                    kv_heads: 1,
                    head_dim: 128,
                    page: 64,
                    kv: DType::Bf16,
                    phase: AttentionPhase::Decode,
                    capacity: 4,
                },
                softmax_scale: 0.125,
                max_context: 4096,
            },
            0,
            2,
            Need::rows(4),
        );
        owned_kv_write: Empty.owned_kv_write(
            &OwnedKvWriteOp {
                query: ContextParallelQuery::OwnedQkNormRopeKvWrite {
                    write: crate::gated_sparse::tests::kv_write().query,
                    ranks: 2,
                },
                max_context: 4096,
            },
            Need::rows(4),
        );
        owned_key_write: Empty.owned_key_write(
            &OwnedKeyWriteOp {
                query: ContextParallelQuery::OwnedIndexerKeyWrite {
                    write: crate::gated_sparse::tests::key_write().query,
                    page: 64,
                    ranks: 2,
                },
                index_heads: 4,
                max_context: 4096,
            },
            0,
            Need::rows(4),
        );
        group_transfer: Empty.group_transfer(
            &GroupTransferOp {
                transfer: GroupTransfer::Gather,
                size: 2,
                words: 1024,
            },
            0,
            Need::rows(4),
        );
        encoder_rows: Empty.encoder_rows(&encoder(), Need::rows(4));
        encoder_heads: Empty.encoder_heads(&encoder(), Need::rows(4));
        encoder_attention: Empty.encoder_attention(&encoder(), 0.125, Need::rows(4));
        encoder_gemm: Empty.encoder_gemm(&encoder(), Need::rows(4));
        embedding_override: Empty.embedding_override(&encoder(), Need::rows(4));
        decode_mxfp8_rows: Empty.decode_mxfp8_rows(
            &Mxfp8Rows { width: 128 },
            Need::rows(4),
        );
        broadcast_streams: Empty.broadcast_streams(
            &StreamBroadcast {
                dtype: DType::Bf16,
                width: 256,
                streams: 4,
                alignment_bytes: 16,
            },
            Need::rows(4),
        );
        column_slice: Empty.column_slice(
            &ColumnSlice {
                dtype: DType::F32,
                width: 288,
                stride: 320,
                offset: 0,
                alignment_bytes: 16,
            },
            Need::rows(4),
        );
        gather_by_index: Empty.gather_by_index(
            &GatherByIndex {
                dtype: DType::Bf16,
                width: 256,
                alignment_bytes: 16,
            },
            Need::rows(4),
        );
    }

    #[test]
    fn every_method_has_a_default_refusal_test() {
        let source = include_str!("backend.rs");
        let start = source.find("pub trait Backend").unwrap();
        let end = start + source[start..].find("\n}\n").unwrap();
        let methods: Vec<&str> = source[start..end]
            .lines()
            .filter_map(|line| line.strip_prefix("    fn "))
            .map(|line| line.split(['(', '<']).next().unwrap())
            .filter(|&method| method != "name")
            .collect();
        assert_eq!(methods, TESTED);
    }

    #[test]
    fn a_need_refuses_guarantees_that_do_not_cover_it() {
        let info = OpInfo::new("test.v1", Invariance::one_row(4), Vec::new());
        assert_eq!(Need::rows(4).check(&info), Ok(()));
        assert_eq!(
            Need::rows(4)
                .with_exact(Invariance::one_row(4))
                .check(&info),
            Ok(())
        );
        let needed = Invariance::one_row(8);
        assert_eq!(
            Need::rows(8).with_exact(needed).check(&info),
            Err(Refused::Inexact {
                needed,
                guaranteed: Invariance::one_row(4)
            })
        );
    }
}
