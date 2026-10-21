//! One rank's device buffers the program binds, sized from the contract:
//! the activations between stages, the step inputs the engines read, the
//! engines' shared workspace and the request state pools. The weights are
//! the rank's placed views by checkpoint name, and the KV pool is the
//! host's serving pool.
//!
//! Every activation holds the scratch capacity's rows (the prefill
//! capacity, which covers every graph bucket); each phase binds its first
//! rows. The step inputs the shared input arena carries
//! (`bundle_runtime::upload::DeviceInputs`) bind from there; the KDA
//! engine's own per-sequence inputs are side buffers here.
//!
//! The scratch and the KDA state pools of the declared
//! `recurrent_state_layouts`, with their verify checkpoints and anchor
//! spares, are allocated before the serving KV pool, which the host sizes
//! from what they leave (`chop.recurrent-state.preallocated.v1`); the
//! indexer tail lives in the host's per-item state
//! (`chop.state-per-item.v1`).

use std::collections::BTreeMap;

use bundle_runtime::geometry::Items;
use bundle_runtime::graphs::verify_tokens;
use bundle_runtime::kv::KvFormat;
use bundle_runtime::state::{Anchors, Extent, RequestState};
use model_glm5next_contract::backend::{Kernels, Selected};
use model_glm5next_contract::contract::{
    DECODE_BATCH, Glm, MAX_VERIFY_ROWS, PREFILL_TOKENS, Shape, has_mtp_layer, latent_format,
};
use model_glm5next_contract::draft_argmax;
use model_glm5next_contract::state::request_state;
use runtime_bundle_abi::DeviceBufferSpanV1;

const BF16: u64 = 2;
const I32: u64 = 4;
const F32: u64 = 4;

/// The buffers a lowered launch binds besides weights and KV pages.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub enum Buffer {
    /// The KDA q, k and v projections side by side, `[rows, 3 * width]`.
    KdaQkv,
    /// Their convolution, the same layout.
    KdaConv,
    /// The forget gate projection, `[rows, width]`.
    KdaGate,
    /// The forget gate's and the output gate's low-rank rows
    /// `[rows, head_dim]`, the down projections' (`linear.forget-a`,
    /// `linear.gate-a`) outputs their up projections read.
    KdaForgetDown,
    KdaGateDown,
    /// The beta logits, `[rows, heads]`.
    KdaBeta,
    KdaOut,
    /// The output gate projection, `[rows, width]`.
    KdaOutputGate,
    KdaNorm,
    /// The query up projection, `[rows, heads, nope]`.
    QueryUp,
    /// The normalized latent, `[rows, latent]`.
    Latent,
    /// The absorbed queries, `[rows, heads, latent]`.
    QueryAbsorbed,
    /// The attention output over the latent, `[rows, heads, latent]`.
    AttentionOut,
    /// The value projection of the attention output, `[rows, heads, v]`.
    ValueOut,
    /// The indexer's key lists, `[rows, max_keys]` i32.
    KeyLists,
    /// The engines' shared split and chunk workspace.
    Workspace,
    /// The residual streams `[rows, streams, hidden]`, two buffers the
    /// sublayers' expansions alternate between.
    Streams0,
    Streams1,
    /// The mixer's collapsed row, `post` `[rows, streams]` and `comb`
    /// `[rows, streams, streams]`, BF16; the head's mean of the streams
    /// reuses the collapsed row.
    Collapsed,
    Post,
    Comb,
    /// The sublayer input after its RMSNorm, and its all-reduced output.
    Normed,
    Reduced,
    /// The indexer's queries `[rows, heads, dim]`, raw keys `[rows, dim]`,
    /// head weights `[rows, heads]` and pooling gate logits `[rows, dim]`.
    IndexQuery,
    IndexKey,
    IndexWeights,
    IndexGate,
    /// The router's F32 logits `[rows, experts]` and the routes: ids and
    /// weights `[rows, top_k]`, per-expert counts and offsets, the padded
    /// route rows, their count and the active experts.
    RouterLogits,
    /// The router GEMM's padded F32 scores `[rows, 320]`, compacted into
    /// `RouterLogits` (`copies::router_scores`, the shared column slice).
    RouterScores,
    RouteIds,
    RouteWeights,
    RouteCounts,
    RouteOffsets,
    SortedRows,
    PaddedRows,
    ActiveExperts,
    /// The experts' SwiGLU rows `[rows * top_k, intermediate]` BF16, their
    /// weighted down rows `[rows * top_k, hidden]` F32, and `[rows, hidden]`
    /// F32 of the MLP's output before it meets the other half: in a prefill
    /// the routed sum, the residual the shared expert's down projection
    /// adds into the partial; in the decode and verify graphs the shared
    /// expert's output, which the routed sum adds into the partial.
    ExpertIntermediate,
    Routed,
    MlpOut,
    /// The dense MLP's or shared expert's SwiGLU rows.
    DenseIntermediate,
    /// The KDA engine's per-sequence inputs the shared arena does not
    /// carry: row offsets with their end (`[items + 1]`), whether a prefill
    /// starts from zero state, its anchor spare, and verify checkpoints.
    CuSeqlens,
    CuRows,
    Initial,
    AnchorIds,
    /// The KDA checkpoint entries of each verify item, item `i`'s row `r`
    /// at `i * kda_checkpoint_stride + r` (`checkpoint_table`).
    CheckpointIds,
    /// The indexer tail's checkpoint entries of each verify item: its
    /// bucket index times `MAX_VERIFY_ROWS` plus the row, past the slots.
    TailCheckpointIds,
    /// The query-down projection's output `[rows, q_rank]` and its norm,
    /// which the query-up and indexer query projections read.
    QueryDown,
    QueryNormed,
    /// The latent-down projection's output `[rows, kv_rank]`, before its
    /// norm into `Latent`.
    LatentRaw,
    /// The sublayer's F32 rank partial `[rows, hidden]` the all-reduce
    /// sums: the output and down projections write it (the embedding
    /// widens into it). The sum is rounded once into `Reduced`.
    Partial,
    /// The prefill rows that need logits, gathered from the final norm's
    /// output `[rows, hidden]`.
    HeadRows,
    /// The head's F32 logits of the rank's vocabulary shard, `[logits_rows(),
    /// vocab_rows]`.
    Logits,
    /// The MTP drafter's single residual stream `[rows, hidden]`, and its
    /// input projection's input `[rows, 2 * hidden]`: its normalized
    /// embedding and target hidden rows side by side.
    DraftHidden,
    DraftEhInput,
    /// The drafter rows' gathered hidden inputs `h_p` `[rows, hidden]`.
    DraftSource,
    /// Each drafter row's source codes `[rows, 2]` i32: its hidden row and
    /// its token (`crate::drafter::Run::codes`).
    DraftCodes,
    /// The run's draft rows `[3, DRAFT_ITEMS]` i32: their rows in the run,
    /// their drafting items and their verify rows; and their count (u32).
    DraftRows,
    DraftCount,
    /// Each drafting item's latest draft `[DRAFT_ITEMS]` u32.
    Drafts,
    /// The draft rows' argmax pairs on the rank's vocabulary shard: their
    /// F32 scores `[DRAFT_ITEMS]`, then their I32 global ids
    /// `[DRAFT_ITEMS]` (`model_glm5next_contract::draft_argmax`).
    DraftPairs,
    /// A draft step's key lists `[DRAFT_ITEMS, max_keys]` i32: the extend
    /// pass's, compacted to each item's last row by `DraftShareRows`.
    SharedKeyLists,
    DraftShareRows,
    /// The split mix-collapse's F32 chain sums `[rows, mixes, chains]`
    /// (`Kernels::mix_scratch_bytes`), rewritten by every mixer
    /// launch before it reads them, and its tickets `[rows]` u32, zeroed at
    /// create and left zeroed by every launch. The split comm epilogue uses
    /// them too.
    MixPartials,
    MixTickets,
    /// F32 zeros `[widest_step_rows(), hidden]`, zeroed at create and never
    /// written: the residual the shared expert's down projection adds in
    /// the decode and verify graphs, where it runs beside the routed
    /// experts, so its output is its sum (`x + 0` is `x` but for `x = -0`,
    /// and the routed sum it then meets, from `+0`, is never `-0`).
    ZeroRows,
}

impl Buffer {
    pub const ALL: [Self; 65] = [
        Self::KdaQkv,
        Self::KdaConv,
        Self::KdaGate,
        Self::KdaForgetDown,
        Self::KdaGateDown,
        Self::KdaBeta,
        Self::KdaOut,
        Self::KdaOutputGate,
        Self::KdaNorm,
        Self::QueryUp,
        Self::Latent,
        Self::QueryAbsorbed,
        Self::AttentionOut,
        Self::ValueOut,
        Self::KeyLists,
        Self::Workspace,
        Self::Streams0,
        Self::Streams1,
        Self::Collapsed,
        Self::Post,
        Self::Comb,
        Self::Normed,
        Self::Reduced,
        Self::IndexQuery,
        Self::IndexKey,
        Self::IndexWeights,
        Self::IndexGate,
        Self::RouterLogits,
        Self::RouterScores,
        Self::RouteIds,
        Self::RouteWeights,
        Self::RouteCounts,
        Self::RouteOffsets,
        Self::SortedRows,
        Self::PaddedRows,
        Self::ActiveExperts,
        Self::ExpertIntermediate,
        Self::Routed,
        Self::MlpOut,
        Self::DenseIntermediate,
        Self::CuSeqlens,
        Self::CuRows,
        Self::Initial,
        Self::AnchorIds,
        Self::CheckpointIds,
        Self::TailCheckpointIds,
        Self::QueryDown,
        Self::QueryNormed,
        Self::LatentRaw,
        Self::Partial,
        Self::HeadRows,
        Self::Logits,
        Self::DraftHidden,
        Self::DraftEhInput,
        Self::DraftSource,
        Self::DraftCodes,
        Self::DraftRows,
        Self::DraftCount,
        Self::Drafts,
        Self::DraftPairs,
        Self::SharedKeyLists,
        Self::DraftShareRows,
        Self::MixPartials,
        Self::MixTickets,
        Self::ZeroRows,
    ];

    /// Whether only the MTP drafter's runs use the buffer.
    pub const fn drafter(self) -> bool {
        matches!(
            self,
            Self::DraftHidden
                | Self::DraftEhInput
                | Self::DraftSource
                | Self::DraftCodes
                | Self::DraftRows
                | Self::DraftCount
                | Self::Drafts
                | Self::DraftPairs
                | Self::SharedKeyLists
                | Self::DraftShareRows
        )
    }

    /// Bytes of the buffer on one rank.
    pub fn bytes(self, shape: Shape) -> Result<u64, String> {
        let p = shape.profile;
        let rows = u64::from(SCRATCH_ROWS);
        let items = u64::from(DECODE_BATCH);
        let drafts = u64::from(model_glm5next_contract::contract::DRAFT_ITEMS);
        let width = u64::from(p.linear_heads * p.linear_head_dim);
        let latent_heads = u64::from(p.latent_heads);
        Ok(match self {
            Self::KdaQkv | Self::KdaConv => rows * 3 * width * BF16,
            Self::KdaGate | Self::KdaOut | Self::KdaOutputGate | Self::KdaNorm => {
                rows * width * BF16
            }
            Self::KdaBeta => rows * u64::from(p.linear_heads) * BF16,
            Self::KdaForgetDown | Self::KdaGateDown => rows * u64::from(p.linear_head_dim) * BF16,
            Self::QueryUp => rows * latent_heads * u64::from(p.qk_nope) * BF16,
            Self::Latent | Self::LatentRaw => rows * u64::from(p.kv_rank) * BF16,
            Self::QueryDown | Self::QueryNormed => rows * u64::from(p.q_rank) * BF16,
            Self::Partial | Self::MlpOut => rows * u64::from(p.hidden) * F32,
            Self::ZeroRows => u64::from(widest_step_rows()) * u64::from(p.hidden) * F32,
            Self::HeadRows | Self::DraftHidden | Self::DraftSource => {
                rows * u64::from(p.hidden) * BF16
            }
            Self::DraftEhInput => rows * 2 * u64::from(p.hidden) * BF16,
            Self::Logits => u64::from(logits_rows()) * u64::from(p.vocab_rows) * F32,
            Self::QueryAbsorbed | Self::AttentionOut => {
                rows * latent_heads * u64::from(p.kv_rank) * BF16
            }
            Self::ValueOut => rows * latent_heads * u64::from(p.v_head) * BF16,
            Self::KeyLists => rows * u64::from(shape.latent_keys()) * I32,
            Self::DraftCodes => rows * 2 * I32,
            Self::DraftRows => 3 * drafts * I32,
            Self::DraftCount => I32,
            Self::Drafts | Self::DraftShareRows => drafts * I32,
            Self::DraftPairs => 2 * drafts * F32,
            Self::SharedKeyLists => drafts * u64::from(shape.latent_keys()) * I32,
            Self::Workspace => Selected::workspace_bytes(shape, widest_step_rows(), SCRATCH_ROWS),
            Self::MixPartials => Selected::mix_scratch_bytes(shape, SCRATCH_ROWS)[0],
            Self::MixTickets => Selected::mix_scratch_bytes(shape, SCRATCH_ROWS)[1],
            Self::Streams0 | Self::Streams1 => rows * u64::from(p.hc_width) * BF16,
            Self::Collapsed | Self::Normed | Self::Reduced => rows * u64::from(p.hidden) * BF16,
            Self::Post => rows * u64::from(p.hc_streams) * BF16,
            Self::Comb => rows * u64::from(p.hc_streams * p.hc_streams) * BF16,
            Self::IndexQuery => rows * u64::from(p.indexer_heads * p.indexer_dim) * BF16,
            Self::IndexKey | Self::IndexGate => rows * u64::from(p.indexer_dim) * BF16,
            Self::IndexWeights => rows * u64::from(p.indexer_heads) * BF16,
            Self::RouterLogits => rows * u64::from(shape.geometry.shape.moe.experts) * F32,
            Self::RouterScores => {
                rows * u64::from(model_glm5next_contract::precision::router_columns(shape).1) * F32
            }
            Self::RouteIds | Self::RouteWeights => rows * u64::from(p.top_k) * I32,
            Self::RouteCounts | Self::RouteOffsets | Self::ActiveExperts => {
                (u64::from(p.local_experts) + 1) * I32
            }
            Self::SortedRows => u64::from(Selected::sorted_rows(shape)?) * I32,
            Self::PaddedRows => I32,
            Self::ExpertIntermediate => rows * u64::from(p.top_k * p.expert_intermediate) * BF16,
            Self::Routed => rows * u64::from(p.top_k * p.hidden) * F32,
            Self::DenseIntermediate => {
                rows * u64::from(p.dense_intermediate.max(p.shared_intermediate)) * BF16
            }
            Self::CuSeqlens | Self::CuRows => (items + 1) * I32,
            Self::Initial | Self::AnchorIds => items * I32,
            Self::CheckpointIds => items * u64::from(kda_checkpoint_stride(shape)?) * I32,
            Self::TailCheckpointIds => items * u64::from(MAX_VERIFY_ROWS) * I32,
        })
    }
}

/// Checkpoint IDs per item in the KDA verify entries' table: their
/// compiled `max_rows`, the stride both the convolution and the delta rule
/// index it by (`checkpoint_ids[seq * max_rows + row]`). The package's
/// `MAX_VERIFY_ROWS` live rows fill the first entries of each item.
pub fn kda_checkpoint_stride(shape: Shape) -> Result<u32, String> {
    let strides = Selected::kda_checkpoint_rows(shape);
    match strides[..] {
        [stride, ..] if strides.iter().all(|&other| other == stride) => {
            if stride < MAX_VERIFY_ROWS {
                return Err(format!(
                    "the KDA verify entries take {stride} rows, the package {MAX_VERIFY_ROWS}"
                ));
            }
            Ok(stride)
        }
        [] => Err("no KDA verify entry".into()),
        _ => Err(format!(
            "the KDA verify entries differ in stride: {strides:?}"
        )),
    }
}

/// A verify step's checkpoint table at `stride` entries per item: item
/// `i`'s checkpoint after row `r` at `i * stride + r`. The entries past an
/// item's rows are never read; they hold -1.
pub fn checkpoint_table(stride: u32, items: &[Vec<i32>]) -> Vec<i32> {
    let stride = stride as usize;
    let mut table = vec![-1; items.len() * stride];
    for (item, ids) in items.iter().enumerate() {
        assert!(
            ids.len() <= stride,
            "{} checkpoints past the stride",
            ids.len()
        );
        table[item * stride..][..ids.len()].copy_from_slice(ids);
    }
    table
}

pub use model_glm5next_contract::contract::{SCORED_ROWS, head_rows, logits_rows};

/// Rows every activation holds: the prefill capacity, which covers every
/// decode and verify bucket.
pub const SCRATCH_ROWS: u32 = PREFILL_TOKENS;

/// Rows of the widest decode or verify launch: a decode bucket of every
/// served item, or a verify step of `verify_tokens` rows (wider steps are
/// refused when their phase is chosen).
pub fn widest_step_rows() -> u32 {
    let items = Items::new(DECODE_BATCH).expect("the package batch is an item count");
    verify_tokens::<Glm>(items).max(DECODE_BATCH)
}

/// The request state buffers of one instance and the extent they serve.
pub struct StatePools {
    pub state: RequestState,
    pub extent: Extent,
    /// One span per state buffer, in declaration order.
    pub spans: Vec<DeviceBufferSpanV1>,
    /// Per state buffer, its separate anchor spare buffer, if it has one.
    pub spares: Vec<Option<DeviceBufferSpanV1>>,
}

impl StatePools {
    /// Each slot's retained target hidden rows for the MTP drafter
    /// (`mtp.retained`): `RETAINED_ROWS` rows per slot entry, by position
    /// (`crate::drafter`), on an instance that runs the drafter.
    pub fn retained(&self) -> Result<DeviceBufferSpanV1, String> {
        self.layer("mtp.retained", 0)
    }

    /// The span of buffer `name`'s layer `layer` (its ordinal among the
    /// layers that use it).
    pub fn layer(&self, name: &str, layer: u64) -> Result<DeviceBufferSpanV1, String> {
        let (index, buffer) = self
            .state
            .buffers
            .iter()
            .enumerate()
            .find(|(_, buffer)| buffer.name == name)
            .ok_or_else(|| format!("no state buffer {name}"))?;
        if layer >= buffer.layers {
            return Err(format!("{name} has no layer {layer}"));
        }
        let stride = buffer.layer_stride(buffer.entries(self.extent));
        let span = self.spans[index];
        Ok(DeviceBufferSpanV1::new(
            span.buffer,
            span.offset + layer * stride,
            stride,
        ))
    }
}

/// One all-reduce protocol's state on one rank: the inbox and flags its
/// peers write, and its epoch.
#[derive(Clone, Copy, Debug)]
pub struct CommProtocol {
    pub inbox: DeviceBufferSpanV1,
    pub flags: DeviceBufferSpanV1,
    pub epoch: DeviceBufferSpanV1,
}

/// One gather of the drafter argmax's pairs on one rank: every rank's
/// gathered words `[ranks, draft_argmax::CAPACITY]`, which every peer
/// writes, the flags peers set and its epoch, the flags and epoch zeroed at
/// create.
#[derive(Clone, Copy, Debug)]
pub struct PairGather {
    pub output: DeviceBufferSpanV1,
    pub flags: DeviceBufferSpanV1,
    pub epoch: DeviceBufferSpanV1,
}

/// One rank's peer-visible comm state: the composed all-reduce's F32 pair
/// state and its F32 sum `[rows, hidden]`, which the pair's all-gather
/// writes on every peer, and the drafter argmax's two pair gathers; the
/// logits moves' flags and epochs with their full-vocabulary F32 rows,
/// decode and verify rows (`[head_rows(), vocab]`) exchanged to their
/// owners and prefill rows (`[SCORED_ROWS, vocab]`) gathered to rank 0;
/// then the low-latency all-reduce's one state; the phased all-reduce's one
/// state, which all its entries and row counts share; last, the MoE
/// sublayers' rounded all-reduce's own low-latency and phased states (a
/// phased state serves one payload format; its composed form shares the
/// pair and the sum). The all-reduces' regions take the bytes their ops
/// keep (`ops::reduce::StateRegions`). Allocated after peer access is granted; the flags and epochs, and
/// the low-latency inbox, start zeroed. The low-latency and the phased
/// all-reduce serve only while `low_latency` and `phased` hold, which
/// create clears on every rank when some device cannot vouch for their
/// grids' residency.
#[derive(Clone, Debug)]
pub struct CommState {
    /// The drafter argmax's gathers of its pairs' scores and ids, in that
    /// order, on an instance that runs the drafter.
    pub draft_gathers: Option<[PairGather; 2]>,
    pub two_shot: CommProtocol,
    pub summed: DeviceBufferSpanV1,
    pub exchange: bundle_runtime::logits::ExchangeRowsState,
    pub gather_flags: DeviceBufferSpanV1,
    pub gather_epoch: DeviceBufferSpanV1,
    pub decode_logits: DeviceBufferSpanV1,
    pub prefill_logits: DeviceBufferSpanV1,
    pub ll_inbox: DeviceBufferSpanV1,
    pub ll_epochs: DeviceBufferSpanV1,
    /// Whether the row counts SM120's table gives the low-latency entry
    /// (`engines::bf16_tuned`) reduce through it (else through the phased
    /// pair).
    pub low_latency: bool,
    pub phased_state: CommProtocol,
    /// Whether the row counts SM120's table gives a phased entry reduce
    /// through it (else through the phased pair).
    pub phased: bool,
    /// The rank's `Buffer::Reduced` scratch rows, which a phased rounded
    /// entry writes on every rank (not allocated here).
    pub reduced: DeviceBufferSpanV1,
    /// The MoE sublayers' low-latency round entry's inbox and epochs, which
    /// serve while `low_latency` holds.
    pub rounded_ll_inbox: DeviceBufferSpanV1,
    pub rounded_ll_epochs: DeviceBufferSpanV1,
    /// The MoE sublayers' phased F32 state, which serves while `phased`
    /// holds.
    pub rounded_phased_state: CommProtocol,
}

impl CommState {
    /// Allocates one rank's state through `allocate`; `reduced` is the
    /// rank's `Buffer::Reduced` scratch span. The all-reduces' regions take
    /// the bytes their ops keep (`regions`, from `OpInfo::state`), as do
    /// the drafter argmax's pair gathers' flags and epoch (`draft`, on an
    /// instance that runs the drafter).
    pub fn allocate<E>(
        shape: Shape,
        draft: Option<[u64; 2]>,
        reduced: DeviceBufferSpanV1,
        regions: crate::ops::reduce::StateRegions,
        mut allocate: impl FnMut(u64) -> Result<DeviceBufferSpanV1, E>,
    ) -> Result<Result<Self, String>, E> {
        let mut protocol = |[inbox, flags, epoch]: [u64; 3]| -> Result<CommProtocol, E> {
            Ok(CommProtocol {
                inbox: allocate(inbox)?,
                flags: allocate(flags)?,
                epoch: allocate(epoch)?,
            })
        };
        let two_shot = protocol(regions.pair)?;
        let draft_gathers = match draft {
            Some([flags, epoch]) => {
                let pairs = u64::from(shape.ranks()) * u64::from(draft_argmax::CAPACITY) * 4;
                let mut gather = || -> Result<PairGather, E> {
                    Ok(PairGather {
                        output: allocate(pairs)?,
                        flags: allocate(flags)?,
                        epoch: allocate(epoch)?,
                    })
                };
                Some([gather()?, gather()?])
            }
            None => None,
        };
        let summed = allocate(regions.sum)?;
        let exchange = bundle_runtime::logits::ExchangeRowsState {
            flags: allocate(kernel_core::row_exchange::EXCHANGE_ROWS_FLAG_BYTES)?,
            epoch: allocate(kernel_core::row_exchange::EXCHANGE_ROWS_EPOCH_BYTES)?,
        };
        let gather_flags = allocate(Selected::GATHER_ROWS_STATE.flags)?;
        let gather_epoch = allocate(Selected::GATHER_ROWS_STATE.epoch)?;
        let vocab = u64::from(shape.geometry.vocab()) * F32;
        let decode_logits = allocate(u64::from(head_rows()) * vocab)?;
        let prefill_logits = allocate(u64::from(SCORED_ROWS) * vocab)?;
        let ll_inbox = allocate(regions.low_latency[0])?;
        let ll_epochs = allocate(regions.low_latency[1])?;
        let [inbox, flags, epoch] = regions.phased;
        let phased_state = CommProtocol {
            inbox: allocate(inbox)?,
            flags: allocate(flags)?,
            epoch: allocate(epoch)?,
        };
        let rounded_ll_inbox = allocate(regions.rounded_low_latency[0])?;
        let rounded_ll_epochs = allocate(regions.rounded_low_latency[1])?;
        let [inbox, flags, epoch] = regions.rounded_phased;
        let rounded_phased_state = CommProtocol {
            inbox: allocate(inbox)?,
            flags: allocate(flags)?,
            epoch: allocate(epoch)?,
        };
        Ok(Ok(Self {
            draft_gathers,
            two_shot,
            summed,
            exchange,
            gather_flags,
            gather_epoch,
            decode_logits,
            prefill_logits,
            ll_inbox,
            ll_epochs,
            low_latency: true,
            phased_state,
            phased: true,
            reduced,
            rounded_ll_inbox,
            rounded_ll_epochs,
            rounded_phased_state,
        }))
    }

    /// The comm state regions of a BF16-partials all-reduce under
    /// `admission`, in the op's role order (`OpInfo::state`): the
    /// low-latency inbox and epochs where it is admitted, the phased state
    /// where it is admitted, else the F32 pair's, then the F32 sum. An op
    /// lists `regions` of them: the sum only where it composes the pair or
    /// feeds the comm epilogue.
    pub fn all_reduce_regions(
        &self,
        admission: crate::ops::reduce::Admission,
        regions: usize,
    ) -> Result<Vec<DeviceBufferSpanV1>, String> {
        let mut spans = Vec::new();
        if admission.low_latency {
            spans.extend([self.ll_inbox, self.ll_epochs]);
        }
        let shared = match admission.phased {
            true => self.phased_state,
            false => self.two_shot,
        };
        spans.extend([shared.inbox, shared.flags, shared.epoch, self.summed]);
        if regions + 1 < spans.len() || regions > spans.len() {
            return Err(format!(
                "an all-reduce of {admission:?} lists {regions} state regions, not those of \
                 its {} entries",
                spans.len()
            ));
        }
        spans.truncate(regions);
        Ok(spans)
    }

    /// The comm state regions of the MoE sublayers' rounded all-reduce
    /// under `admission`, in the op's role order (`OpInfo::state`): its
    /// low-latency inbox and epochs where admitted, its phased state where
    /// admitted, else the F32 pair's, then the F32 sum. The op lists
    /// `regions` of them: the sum only where it composes the pair.
    pub fn rounded_regions(
        &self,
        admission: crate::ops::reduce::Admission,
        regions: usize,
    ) -> Result<Vec<DeviceBufferSpanV1>, String> {
        let mut spans = Vec::new();
        if admission.low_latency {
            spans.extend([self.rounded_ll_inbox, self.rounded_ll_epochs]);
        }
        let shared = match admission.phased {
            true => self.rounded_phased_state,
            false => self.two_shot,
        };
        spans.extend([shared.inbox, shared.flags, shared.epoch, self.summed]);
        if regions + 1 < spans.len() || regions > spans.len() {
            return Err(format!(
                "a rounded all-reduce of {admission:?} lists {regions} state regions, not those \
                 of its {} entries",
                spans.len()
            ));
        }
        spans.truncate(regions);
        Ok(spans)
    }

    /// The spans that must start at zero.
    pub fn zeroed(&self) -> Vec<DeviceBufferSpanV1> {
        let gathers = self
            .draft_gathers
            .iter()
            .flatten()
            .flat_map(|gather| [gather.flags, gather.epoch]);
        gathers
            .chain([
                self.two_shot.flags,
                self.two_shot.epoch,
                self.exchange.flags,
                self.exchange.epoch,
                self.gather_flags,
                self.gather_epoch,
                self.ll_inbox,
                self.ll_epochs,
                self.phased_state.flags,
                self.phased_state.epoch,
                self.rounded_ll_inbox,
                self.rounded_ll_epochs,
                self.rounded_phased_state.flags,
                self.rounded_phased_state.epoch,
            ])
            .collect()
    }

    /// Every span, for bounds checks.
    pub fn spans(&self) -> Vec<DeviceBufferSpanV1> {
        let gathers = self
            .draft_gathers
            .iter()
            .flatten()
            .flat_map(|gather| [gather.output, gather.flags, gather.epoch]);
        gathers
            .chain([
                self.two_shot.inbox,
                self.two_shot.flags,
                self.two_shot.epoch,
                self.summed,
                self.exchange.flags,
                self.exchange.epoch,
                self.gather_flags,
                self.gather_epoch,
                self.decode_logits,
                self.prefill_logits,
                self.ll_inbox,
                self.ll_epochs,
                self.phased_state.inbox,
                self.phased_state.flags,
                self.phased_state.epoch,
                self.rounded_ll_inbox,
                self.rounded_ll_epochs,
                self.rounded_phased_state.inbox,
                self.rounded_phased_state.flags,
                self.rounded_phased_state.epoch,
            ])
            .collect()
    }
}

/// One rank's hidden rows a step stages for the MTP drafter's commit to
/// retain: each prefill item's last row `[DRAFT_ITEMS, hidden]` and every
/// decode or verify row `[head_rows(), hidden]`. The retained rows
/// themselves are request state (`StatePools::retained`), which prefix
/// snapshots carry.
#[derive(Clone, Copy, Debug)]
pub struct DrafterBuffers {
    pub staged: [DeviceBufferSpanV1; 2],
    /// The `u32` columns of the rank's logits shard, which create writes
    /// once: the live width the drafter's argmax partial reads.
    pub vocab_width: DeviceBufferSpanV1,
}

impl DrafterBuffers {
    pub fn allocate<E>(
        shape: Shape,
        mut allocate: impl FnMut(u64) -> Result<DeviceBufferSpanV1, E>,
    ) -> Result<Self, E> {
        let row = u64::from(shape.profile.hidden) * BF16;
        let items = u64::from(model_glm5next_contract::contract::DRAFT_ITEMS);
        Ok(Self {
            staged: [
                allocate(items * row)?,
                allocate(u64::from(head_rows()) * row)?,
            ],
            vocab_width: allocate(I32)?,
        })
    }

    /// Row `row` of `span`, `[rows, hidden]` BF16.
    pub fn row(shape: Shape, span: DeviceBufferSpanV1, row: u32) -> DeviceBufferSpanV1 {
        let bytes = u64::from(shape.profile.hidden) * BF16;
        DeviceBufferSpanV1::new(span.buffer, span.offset + u64::from(row) * bytes, bytes)
    }
}

/// One rank's buffers.
pub struct RankBuffers {
    pub scratch: BTreeMap<Buffer, DeviceBufferSpanV1>,
    pub comm: CommState,
    pub state: StatePools,
    /// The rank's placed weight views, by checkpoint name.
    pub views: BTreeMap<String, DeviceBufferSpanV1>,
    /// The rank's serving KV pool.
    pub kv_pool: DeviceBufferSpanV1,
    /// The MTP drafter's state, when the instance runs it.
    pub drafter: Option<DrafterBuffers>,
}

/// Allocates one rank's scratch through `allocate`, in [`Buffer::ALL`]
/// order: the error of `allocate`, or a sizing refusal. The drafter's
/// buffers (256 MiB per rank at the prefill capacity) only when the
/// instance runs the MTP drafter: the host sizes the KV pool from the
/// memory left after create's allocations.
pub fn allocate_scratch<E>(
    shape: Shape,
    drafter: bool,
    mut allocate: impl FnMut(u64) -> Result<DeviceBufferSpanV1, E>,
) -> Result<Result<BTreeMap<Buffer, DeviceBufferSpanV1>, String>, E> {
    allocate_scratch_named(shape, drafter, |_, bytes| allocate(bytes))
}

/// Named traversal lets a transport give peer-visible planes private owners.
pub(crate) fn allocate_scratch_named<E>(
    shape: Shape,
    drafter: bool,
    mut allocate: impl FnMut(Buffer, u64) -> Result<DeviceBufferSpanV1, E>,
) -> Result<Result<BTreeMap<Buffer, DeviceBufferSpanV1>, String>, E> {
    let mut scratch = BTreeMap::new();
    for buffer in Buffer::ALL {
        if buffer.drafter() && !drafter {
            continue;
        }
        let bytes = match buffer.bytes(shape) {
            Ok(bytes) => bytes,
            Err(error) => return Ok(Err(error)),
        };
        scratch.insert(buffer, allocate(buffer, bytes)?);
    }
    Ok(Ok(scratch))
}

impl StatePools {
    /// The span of entry `entry` of buffer `index`'s layer `layer`: each
    /// layer holds the buffer's entries back to back.
    pub fn entry(&self, index: usize, layer: u64, entry: u64) -> DeviceBufferSpanV1 {
        let buffer = &self.state.buffers[index];
        let bytes = buffer.bytes_per_layer_rank();
        let span = self.spans[index];
        DeviceBufferSpanV1::new(
            span.buffer,
            span.offset + layer * buffer.layer_stride(buffer.entries(self.extent)) + entry * bytes,
            bytes,
        )
    }

    /// Entry `entry` of buffer `index` in its first layer, and the bytes
    /// from one layer's entry to the next's (`entry`).
    pub fn entry_layers(&self, index: usize, entry: u64) -> (DeviceBufferSpanV1, u64) {
        let buffer = &self.state.buffers[index];
        let stride = buffer.layer_stride(buffer.entries(self.extent));
        (self.entry(index, 0, entry), stride)
    }

    /// Bytes of the drafter's retained rows (`mtp.retained`) on an
    /// instance of `extent` that runs it, which create allocates before
    /// it sizes the KV pool.
    pub fn retained_bytes(shape: Shape, format: &KvFormat, extent: Extent) -> Option<u64> {
        request_state(shape, format, has_mtp_layer(format))
            .buffers
            .iter()
            .find(|buffer| buffer.name == "mtp.retained")
            .map(|buffer| buffer.bytes(extent))
    }

    /// One rank's request state of an instance of `extent`: the KDA pools
    /// allocated through `allocate` (before the serving KV pool), the
    /// indexer tail in the rank's view of the host's per-item state and,
    /// on an instance that runs the drafter, its retained rows in
    /// `retained` (`retained_bytes`, allocated before the KV pool).
    pub fn new<E>(
        shape: Shape,
        format: &KvFormat,
        extent: Extent,
        retained: Option<DeviceBufferSpanV1>,
        tail: DeviceBufferSpanV1,
        mut allocate: impl FnMut(u64) -> Result<DeviceBufferSpanV1, E>,
    ) -> Result<Result<Self, String>, E> {
        let drafter = retained.is_some() && has_mtp_layer(format);
        let state = request_state(shape, format, drafter);
        // The separate anchor spares follow the last pool create allocates,
        // in its allocation.
        let spare_bytes = state
            .buffers
            .iter()
            .map(|buffer| buffer.spare_bytes())
            .collect::<Vec<_>>();
        let last_pool = state
            .buffers
            .iter()
            .rposition(|buffer| !matches!(buffer.name, "indexer.tail" | "mtp.retained"));
        let mut spare_room = None;
        let mut spans = Vec::new();
        for (index, buffer) in state.buffers.iter().enumerate() {
            let bytes = buffer.bytes(extent);
            if Some(index) == last_pool {
                let total = bytes + spare_bytes.iter().sum::<u64>();
                let span = allocate(total)?;
                spare_room = Some(DeviceBufferSpanV1::new(
                    span.buffer,
                    span.offset + bytes,
                    total - bytes,
                ));
                spans.push(DeviceBufferSpanV1::new(span.buffer, span.offset, bytes));
                continue;
            }
            spans.push(match (buffer.name, retained) {
                ("indexer.tail", _) if bytes > tail.length => {
                    return Ok(Err(format!(
                        "the indexer tail takes {bytes} bytes, the host's state {}",
                        tail.length
                    )));
                }
                ("indexer.tail", _) => DeviceBufferSpanV1::new(tail.buffer, tail.offset, bytes),
                ("mtp.retained", Some(span)) if span.length == bytes => span,
                ("mtp.retained", _) => {
                    return Ok(Err(format!("the retained rows take {bytes} bytes")));
                }
                _ => allocate(bytes)?,
            });
        }
        let mut spares = Vec::with_capacity(state.buffers.len());
        let mut at = 0;
        for &bytes in &spare_bytes {
            spares.push(match (bytes, spare_room) {
                (0, _) => None,
                (bytes, Some(room)) => {
                    at += bytes;
                    Some(DeviceBufferSpanV1::new(
                        room.buffer,
                        room.offset + at - bytes,
                        bytes,
                    ))
                }
                (_, None) => return Ok(Err("anchor spares without a state pool".to_owned())),
            });
        }
        Ok(Ok(Self {
            state,
            extent,
            spans,
            spares,
        }))
    }

    /// The entry anchor item `anchor`'s spares take in buffer `name`, whose
    /// spares follow its own entries and checkpoints.
    pub fn anchor_entry(&self, name: &str, anchor: u32) -> Result<u64, String> {
        let buffer = self
            .state
            .buffers
            .iter()
            .find(|buffer| buffer.name == name)
            .ok_or_else(|| format!("no state buffer {name}"))?;
        match buffer.anchors {
            Anchors::Spare {
                per_step,
                pooled: true,
            } if anchor < per_step => {
                Ok(buffer.entries(self.extent) - u64::from(per_step - anchor))
            }
            _ => Err(format!("{name} keeps no anchor spare {anchor}")),
        }
    }
}

impl RankBuffers {
    /// An allocated scratch buffer: the drafter's exist only on an instance
    /// that runs it, whose drafter programs alone use them.
    pub fn get(&self, buffer: Buffer) -> DeviceBufferSpanV1 {
        self.scratch[&buffer]
    }

    pub fn view(&self, name: &str) -> Result<DeviceBufferSpanV1, String> {
        self.views
            .get(name)
            .copied()
            .ok_or_else(|| format!("no placed view {name}"))
    }

    /// The KV pool from sparse layer `ordinal`'s planes in page 0, for a
    /// kernel that steps pages by the format's page bytes.
    pub fn kv_layer(&self, shape: Shape, format: &KvFormat, ordinal: u64) -> DeviceBufferSpanV1 {
        let offset = ordinal * shape.page_layer_bytes(latent_format(format));
        DeviceBufferSpanV1::new(
            self.kv_pool.buffer,
            self.kv_pool.offset + offset,
            self.kv_pool.length - offset,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn runtime_abi_handle(slot: u32) -> runtime_bundle_abi::BundleDeviceBufferHandleV1 {
        runtime_bundle_abi::BundleDeviceBufferHandleV1::from_parts(1, slot)
    }

    #[test]
    fn scratch_sizes_follow_the_contract() {
        let shape = Shape::TP4;
        let bytes = |buffer: Buffer| buffer.bytes(shape).unwrap();
        assert_eq!(bytes(Buffer::KdaQkv), 8192 * 6144 * 2);
        assert_eq!(bytes(Buffer::QueryAbsorbed), 8192 * 16 * 512 * 2);
        assert_eq!(bytes(Buffer::KeyLists), 8192 * 2051 * 4);
        assert_eq!(bytes(Buffer::CheckpointIds), 32 * 8 * 4);
        assert_eq!(bytes(Buffer::TailCheckpointIds), 32 * 6 * 4);
        // The latent attention's split partials of the widest step, 32 rows
        // of 64 splits, fit the workspace.
        assert_eq!(widest_step_rows(), 32);
        assert!(bytes(Buffer::Workspace) >= 32 * 64 * 16 * 514 * 4);
    }

    /// The tail's declared slot pieces, the span its commit copies
    /// (`StatePools::entry`) and the entry the key write kernel addresses
    /// (`tail_state + slot * state_elements` in each layer's span) agree:
    /// one contiguous 1,536-byte entry per layer.
    #[test]
    fn the_tail_declaration_matches_the_kernel_entry() {
        use bundle_runtime::state::Holder;
        let shape = Shape::TP4;
        let format = shape.kv_formats()[0];
        let extent = Extent {
            served: DECODE_BATCH,
            bucket: DECODE_BATCH,
            verify_rows: 4,
        };
        let handle = |slot| runtime_abi_handle(slot);
        let mut next = 0;
        let tail = DeviceBufferSpanV1::new(handle(999), 0, 1 << 30);
        let pools = StatePools::new(shape, &format, extent, None, tail, |bytes| {
            next += 1;
            Ok::<_, ()>(DeviceBufferSpanV1::new(handle(next), 0, bytes))
        })
        .unwrap()
        .unwrap();
        let index = pools
            .state
            .buffers
            .iter()
            .position(|buffer| buffer.name == "indexer.tail")
            .unwrap();
        let entry = shape.indexer_tail_bytes();
        assert_eq!(entry, 1536);
        assert_eq!(pools.state.buffers[index].bytes_per_layer_rank(), entry);
        for slot in [0, 1, 31] {
            let declared = pools
                .state
                .pieces(extent, Holder::Slot(slot))
                .unwrap()
                .into_iter()
                .filter(|piece| piece.buffer == index)
                .map(|piece| (piece.offset, piece.length))
                .collect::<Vec<_>>();
            let layers = pools.state.buffers[index].layers;
            let copied = (0..layers)
                .map(|layer| {
                    let span = pools.entry(index, layer, u64::from(slot));
                    (span.offset, span.length)
                })
                .collect::<Vec<_>>();
            let kernel = (0..layers)
                .map(|layer| {
                    let span = pools.layer("indexer.tail", layer).unwrap();
                    (span.offset + u64::from(slot) * entry, entry)
                })
                .collect::<Vec<_>>();
            assert_eq!(declared, copied, "slot {slot}");
            assert_eq!(copied, kernel, "slot {slot}");
        }
    }

    /// Every checkpoint ID the KDA verify kernels read at the served items
    /// and the package's verify rows lies inside the table.
    #[test]
    fn checkpoint_tables_hold_every_row_the_kernels_read() {
        let shape = Shape::TP4;
        let stride = kda_checkpoint_stride(shape).unwrap();
        assert_eq!(stride, 8);
        let last =
            (u64::from(DECODE_BATCH) - 1) * u64::from(stride) + u64::from(MAX_VERIFY_ROWS) - 1;
        assert!((last + 1) * I32 <= Buffer::CheckpointIds.bytes(shape).unwrap());
        let tail =
            model_glm5next_contract::sm120::engines::pooled_indexer_geometry(shape).verify_rows;
        let last = (u64::from(DECODE_BATCH) - 1) * u64::from(tail) + u64::from(MAX_VERIFY_ROWS) - 1;
        assert!((last + 1) * I32 <= Buffer::TailCheckpointIds.bytes(shape).unwrap());
    }

    /// The second item's checkpoints start one stride in, past the first
    /// item's unread entries.
    #[test]
    fn checkpoint_tables_lay_items_out_at_the_entry_stride() {
        let table = checkpoint_table(8, &[vec![40, 41, 42], vec![50, 51, 52, 53, 54, 55]]);
        assert_eq!(table.len(), 16);
        assert_eq!(table[..4], [40, 41, 42, -1]);
        assert_eq!(table[8..14], [50, 51, 52, 53, 54, 55]);
        assert_eq!(table[14..], [-1, -1]);
    }
}
