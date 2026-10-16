//! The ops an instance prepares once through `kernel_core::backend`, and
//! their launches over a step's buffers by role.
//!
//! MiMo states each op as a kernel-core contract and the exactness it needs
//! (`Need`); the backend of the instance's target (`Target::backend`)
//! prepares it when the instance is created. A step then only binds its spans to the op's roles
//! and calls `PreparedOp::launches`: no catalog search, entry choice or
//! environment read happens per step.

mod attention;
mod experts;
mod gather;
mod head_norm;
mod moe;
mod own;
mod rope;
mod rows;

pub use attention::{AttentionViews, Layer as AttentionLayer, Part as AttentionPart};

pub use experts::{ExpertViews, Experts};
pub use gather::{LogitsBuffers, Regions as GatherRegions, State as GatherState};
pub use head_norm::HeadNorm;
pub use moe::{RoutePart, RouteViews, route_tile_tables};
pub use own::{OwnPlan, RingBackupViews, RingRestoreViews};
pub use rope::{CacheViews, RopeViews, Target as RopeTarget};
pub use rows::{EmbeddingViews, Embeddings, PartialViews};

use kernel_core::backend::{Backend, Counts, Launch, Need, Prepared, Refused, View};
use kernel_core::collective::{FusedAllReduceBuffers, RankViews, ReduceInput};
use kernel_core::contract::{CommQuery, DType, Refusal, RmsNormEpilogue, RmsNormKind};
use kernel_core::precision::{ExecutionPhase, OperationUse};
use kernel_core::properties::Invariance;
use kernel_core::rmsnorm::{FusedResidual, RmsNormBuffers, RmsNormNumerics, RmsNormRequirement};
use kernel_core::row_copy::RowCopyBuffers;
use runtime_bundle_abi::DeviceBufferSpanV1;

use crate::dispatch::BufferBindings;
use crate::pipeline::PreparedLaunch;
use crate::program_lower::Lowered;
use crate::scratch::COMM_BUCKETS;
use crate::selection::{MimoPhase as _, Phase, RowBuckets};
use crate::switches::Switches;
use crate::target::Target;
use crate::topology::Tp;

/// The hidden state's RMSNorm, in HF order as the reference computes it.
pub fn hidden_norm(tp: Tp) -> RmsNormRequirement {
    RmsNormRequirement::bf16(
        tp.geometry.hidden(),
        tp.geometry.eps() as f32,
        RmsNormKind::HfBf16Intermediate,
        false,
        RmsNormNumerics::HF_ORDERED,
    )
}

/// One row of the hidden all-reduce: the rank partials' sum is added to the
/// residual and normalized as `hidden_norm` does.
fn hidden_reduce(tp: Tp) -> CommQuery {
    CommQuery::AllReduce {
        ranks: tp.ranks(),
        elements: u64::from(tp.geometry.hidden()),
        dtype: DType::Bf16,
        fused_residual: true,
        fused_rmsnorm: Some(RmsNormEpilogue {
            kind: RmsNormKind::HfBf16Intermediate,
            width: tp.geometry.hidden(),
            eps: tp.geometry.eps() as f32,
        }),
    }
}

/// The use of a `phase` step's ops: graphs for decode and verify buckets.
fn usage(phase: Phase) -> Result<OperationUse, Refusal> {
    Ok(OperationUse {
        phase: match phase {
            Phase::Prefill => ExecutionPhase::Prefill,
            Phase::Decode(_) => ExecutionPhase::Decode,
            Phase::Verify(_) => ExecutionPhase::Verify,
        },
        capacity: phase.capacity()?,
        graph: phase.token_bucketed(),
    })
}

/// What a `phase` step needs of a row op: the phase's rows, a row's bits
/// in any chunk for prefill, and for exact verify rows each row's one-row
/// bits.
fn need(phase: Phase, switches: Switches) -> Result<Need, Refusal> {
    let capacity = phase.capacity()?;
    let rows = Need::rows(capacity);
    Ok(match phase {
        Phase::Prefill => rows.with_exact(Invariance::PREFILL),
        Phase::Verify(_) if phase.exact_rows(switches) => {
            rows.with_exact(Invariance::one_row(capacity))
        }
        Phase::Decode(_) | Phase::Verify(_) => rows,
    })
}

/// The phase whose steps run a projection bound at `usage`.
pub(crate) fn phase_of(usage: OperationUse) -> Result<Phase, Refusal> {
    let phase = match usage.phase {
        ExecutionPhase::Prefill => Phase::Prefill,
        ExecutionPhase::Decode => Phase::Decode(usage.capacity),
        ExecutionPhase::Verify => Phase::Verify(usage.capacity),
    };
    if phase.capacity()? != usage.capacity {
        return Err(Refusal("a projection use's capacity is not a phase's"));
    }
    Ok(phase)
}

/// What a projection bound at `usage` needs: [`need`] of the phase whose
/// steps run it.
pub(crate) fn projection_need(usage: OperationUse, switches: Switches) -> Result<Need, Refusal> {
    need(phase_of(usage)?, switches)
}

/// One op per phase that prepares its own: prefill, and each decode and
/// verify bucket, indexed as `COMM_BUCKETS`.
struct ByPhase<T> {
    prefill: Option<T>,
    decode: Vec<Option<T>>,
    verify: Vec<Option<T>>,
}

impl<T> Default for ByPhase<T> {
    fn default() -> Self {
        Self {
            prefill: None,
            decode: Vec::new(),
            verify: Vec::new(),
        }
    }
}

impl<T> ByPhase<T> {
    /// `prepare`'s op, if any, for prefill and every decode and verify
    /// bucket.
    fn prepare(
        mut prepare: impl FnMut(Phase) -> Result<Option<T>, Refused>,
    ) -> Result<Self, Refused> {
        let mut bucketed = |phase: fn(u32) -> Phase| {
            COMM_BUCKETS
                .iter()
                .map(|&bucket| {
                    let phase = phase(bucket);
                    if phase.capacity() == Ok(bucket) {
                        prepare(phase)
                    } else {
                        Ok(None)
                    }
                })
                .collect::<Result<Vec<_>, _>>()
        };
        let decode = bucketed(Phase::Decode)?;
        let verify = bucketed(Phase::Verify)?;
        Ok(Self {
            prefill: prepare(Phase::Prefill)?,
            decode,
            verify,
        })
    }

    /// The op of `phase`, whose capacity is `COMM_BUCKETS[bucket]`.
    fn get(&self, phase: Phase, bucket: usize) -> Result<&T, Refusal> {
        match phase {
            Phase::Prefill => self.prefill.as_ref(),
            Phase::Decode(_) => self.decode.get(bucket).and_then(Option::as_ref),
            Phase::Verify(_) => self.verify.get(bucket).and_then(Option::as_ref),
        }
        .ok_or(Refusal("MiMo has no prepared op for this phase"))
    }
}

type Reduce = Prepared<FusedAllReduceBuffers<View>>;

/// How a hidden all-reduce runs.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Route {
    /// Prefill, or a decode or verify graph.
    Phase,
    /// An exact verify bucket's rows (`MimoPhase::exact_rows`), each with
    /// the bits it has reduced alone.
    Exact,
}

/// One rank's hidden all-reduces (residual add and norm fused), by route
/// and input.
#[derive(Default)]
struct Reduces {
    plain: ByPhase<Reduce>,
    exact: ByPhase<Reduce>,
    /// Decode and verify buckets that fold the MoE combine into the MLP
    /// all-reduce, whose input is each row's routed rows.
    combine: ByPhase<Reduce>,
    combine_exact: ByPhase<Reduce>,
}

/// The device views of one hidden all-reduce on a rank.
pub struct ReduceViews<'a> {
    pub input: DeviceBufferSpanV1,
    /// The residual, which the rounded sum replaces.
    pub residual: DeviceBufferSpanV1,
    pub norm_weight: DeviceBufferSpanV1,
    pub output: DeviceBufferSpanV1,
    /// The live row count of a routed input.
    pub live_rows: Option<DeviceBufferSpanV1>,
    /// Every rank's normalized rows and comm state (inbox, flags, epoch) of
    /// the phase's bucket, in rank order.
    pub outputs: &'a [DeviceBufferSpanV1],
    pub inboxes: &'a [DeviceBufferSpanV1],
    pub flags: &'a [DeviceBufferSpanV1],
    pub epochs: &'a [DeviceBufferSpanV1],
}

/// Every op an instance launches through its backend.
#[derive(Default)]
pub struct Ops {
    /// The initial norm and the drafters' input norms (`hidden_norm`).
    norms: ByPhase<Prepared<RmsNormBuffers<View>>>,
    /// The target's row buckets, which eager prefill row ops grid.
    row_buckets: RowBuckets,
    /// Each rank's hidden all-reduces.
    reduces: Vec<Reduces>,
    /// Per decode and verify bucket, whether its graphs fold the MoE
    /// combine into the MLP all-reduce.
    folds: ByPhase<bool>,
    /// Each rank's logits gathers and DFlash candidate all-gathers.
    gathers: Vec<gather::Gathers>,
    /// The RoPE and KV writes, which every rank launches alike.
    ropes: rope::Ropes,
    /// The attentions, which every rank launches alike.
    attentions: attention::Attentions,
    /// DFlash's embedding and prefill's head-row gather, which every rank
    /// launches alike.
    embeddings: Embeddings,
    /// The drafters' sharded argmax.
    argmax: rows::Argmax,
    /// DFlash's Q and K head norms, which every rank launches alike.
    head_norms: head_norm::HeadNorms,
    /// The MoE routes and combines, which every rank launches alike.
    moe: moe::Moe,
    /// MiMo's own kernels, which every rank launches alike.
    own: own::Own,
    /// An MTP instance's slice of each rank's EH shard (`Tp::mtp_eh_slice`).
    eh_slices: Vec<ByPhase<Prepared<RowCopyBuffers<View>>>>,
}

/// What an instance prepares its ops for.
pub struct Plan<'a> {
    pub tp: Tp,
    pub switches: Switches,
    /// The global layers' KV page format.
    pub kv: DType,
    /// The items the instance serves.
    pub items: u32,
    /// The target the backend's ops run on.
    pub target: &'a dyn Target,
    /// The bytes of a rank's comm state regions of `COMM_BUCKETS[bucket]`
    /// (inbox, flags, epoch), which every all-reduce of the bucket must fit.
    pub state: &'a dyn Fn(usize) -> [u64; 3],
    /// The gather regions of `COMM_BUCKETS[bucket]`.
    pub gathers: &'a dyn Fn(usize) -> GatherRegions,
    /// A DFlash instance's candidate all-gather flag and epoch bytes.
    pub dflash: Option<GatherState>,
    /// What the instance prepares its own kernels' ops for.
    pub own: OwnPlan<'a>,
}

impl Ops {
    /// Prepares every op of `plan` on `backend`.
    pub fn prepare(backend: &dyn Backend, plan: &Plan<'_>) -> Result<Self, Refused> {
        let Plan {
            tp,
            switches,
            kv,
            items,
            target,
            state,
            gathers,
            dflash,
            own,
        } = *plan;
        let requirement = hidden_norm(tp);
        let norms = ByPhase::prepare(|phase| {
            let need = need(phase, switches)?;
            no_workspace(backend.rmsnorm(&requirement, need)?).map(Some)
        })?;
        let folds = ByPhase::prepare(|phase| {
            Ok(Some(
                phase.token_bucketed() && target.folds_combine(phase.capacity()?),
            ))
        })?;
        let query = hidden_reduce(tp);
        let routed = ReduceInput::Routed {
            routes: tp.geometry.top_k(),
        };
        let reduces = (0..tp.ranks())
            .map(|rank| {
                let prepare = |input: ReduceInput, route: Route, phase: Phase| {
                    let capacity = phase.capacity()?;
                    let mut need = Need::rows(capacity);
                    if route == Route::Exact {
                        // Each row of an exact verify bucket keeps the bits it
                        // has reduced alone: the decode entries at 1 and 2
                        // rows and the exact entries above declare it against
                        // one reference (D-AR-EXACT).
                        need = need.with_exact(Invariance::one_row(capacity));
                    }
                    let op = no_workspace(backend.fused_all_reduce(
                        &query,
                        input,
                        usage(phase)?,
                        rank,
                        need,
                    )?)?;
                    let bucket = COMM_BUCKETS
                        .iter()
                        .position(|&rows| rows == capacity)
                        .ok_or(Refusal("MiMo has no comm state for this phase"))?;
                    if op.info().state.len() != 3
                        || op
                            .info()
                            .state
                            .iter()
                            .zip(state(bucket))
                            .any(|(&op, held)| op > held)
                    {
                        return Err(Refused::Rule(
                            "a MiMo all-reduce needs more comm state than its bucket holds".into(),
                        ));
                    }
                    Ok(Some(op))
                };
                let exact =
                    |phase: Phase| matches!(phase, Phase::Verify(_)) && phase.exact_rows(switches);
                let folded = |phase: Phase| Ok::<_, Refused>(*folds_of(&folds, phase)?);
                Ok(Reduces {
                    plain: ByPhase::prepare(|phase| {
                        prepare(ReduceInput::Rows, Route::Phase, phase)
                    })?,
                    exact: ByPhase::prepare(|phase| {
                        if exact(phase) {
                            prepare(ReduceInput::Rows, Route::Exact, phase)
                        } else {
                            Ok(None)
                        }
                    })?,
                    combine: ByPhase::prepare(|phase| {
                        if folded(phase)? {
                            prepare(routed, Route::Phase, phase)
                        } else {
                            Ok(None)
                        }
                    })?,
                    combine_exact: ByPhase::prepare(|phase| {
                        if folded(phase)? && exact(phase) {
                            prepare(routed, Route::Exact, phase)
                        } else {
                            Ok(None)
                        }
                    })?,
                })
            })
            .collect::<Result<_, Refused>>()?;
        let gathers = (0..tp.ranks())
            .map(|rank| gather::Gathers::prepare(backend, tp, rank, switches, gathers, dflash))
            .collect::<Result<_, Refused>>()?;
        let moe = moe::Moe::prepare(backend, target, tp, switches, &|phase| {
            Ok(*folds_of(&folds, phase)?)
        })?;
        // Every rank slices its EH shard at every phase.
        let eh_slices = (0..if own.mtp { tp.ranks() } else { 0 })
            .map(|rank| {
                let slice = tp.mtp_eh_slice(rank as usize);
                ByPhase::prepare(|phase| {
                    let need = Need::rows(phase.capacity()?);
                    no_workspace(backend.column_slice(&slice, need)?).map(Some)
                })
            })
            .collect::<Result<_, Refused>>()?;
        let row_buckets = target.row_buckets();
        Ok(Self {
            norms,
            row_buckets,
            reduces,
            folds,
            gathers,
            ropes: rope::Ropes::prepare(backend, tp, switches, kv, dflash.is_some())?,
            attentions: attention::Attentions::prepare(
                backend,
                target,
                tp,
                switches,
                kv,
                items,
                dflash.is_some(),
            )?,
            embeddings: Embeddings::prepare(backend, tp, switches, row_buckets)?,
            argmax: rows::Argmax::prepare(backend, tp, dflash.is_some())?,
            head_norms: head_norm::HeadNorms::prepare(backend, tp, dflash.is_some())?,
            moe,
            own: own::Own::prepare(&own, tp, switches, items, dflash.is_some())?,
            eh_slices,
        })
    }

    /// Whether a `phase` step folds the MoE combine into its MLP all-reduce.
    pub fn folds_combine(&self, phase: Phase, bucket: usize) -> bool {
        self.folds.get(phase, bucket).is_ok_and(|&folds| folds)
    }

    /// Whether a `phase` step folds the MoE combine into its MLP all-reduce,
    /// at the phase's own comm bucket.
    pub fn folds_combine_in(&self, phase: Phase) -> bool {
        bucket_of(phase).is_ok_and(|bucket| self.folds_combine(phase, bucket))
    }

    /// Rank `rank`'s hidden all-reduce of `rows` rows of a `phase` step
    /// (bucket `bucket` of `COMM_BUCKETS`) by `route`, of routed rows when
    /// `views.live_rows` is set.
    pub fn hidden_reduce(
        &self,
        rank: u32,
        phase: Phase,
        bucket: usize,
        route: Route,
        rows: u32,
        views: ReduceViews<'_>,
    ) -> Result<Vec<Lowered>, Refusal> {
        let reduces = self
            .reduces
            .get(rank as usize)
            .ok_or(Refusal("MiMo has no all-reduces for this rank"))?;
        let ops = match (route, views.live_rows.is_some()) {
            (Route::Phase, false) => &reduces.plain,
            (Route::Exact, false) => &reduces.exact,
            (Route::Phase, true) => &reduces.combine,
            (Route::Exact, true) => &reduces.combine_exact,
        };
        let op = ops.get(phase, bucket)?;
        let ranks = views.outputs.len();
        if [views.inboxes.len(), views.flags.len(), views.epochs.len()] != [ranks; 3] {
            return Err(Refusal("MiMo peer buffers do not cover the ranks"));
        }
        let mut bind = Bind::default();
        let buffers = FusedAllReduceBuffers {
            input: bind.view(views.input),
            residual: FusedResidual {
                input: bind.view(views.residual),
                output: bind.view(views.residual),
            },
            weight: bind.view(views.norm_weight),
            output: bind.view(views.output),
            live_rows: views.live_rows.map(|span| bind.view(span)),
            ranks: (0..ranks)
                .map(|peer| RankViews {
                    output: bind.view(views.outputs[peer]),
                    state: vec![
                        bind.view(views.inboxes[peer]),
                        bind.view(views.flags[peer]),
                        bind.view(views.epochs[peer]),
                    ],
                })
                .collect(),
        };
        Ok(bind.launches(op.launches(&buffers, View::EMPTY, Counts::rows(rows))?))
    }

    /// The hidden norm of `live_tokens` rows of a `phase` step (bucket
    /// `bucket` of `COMM_BUCKETS`). It grids the phase's row bucket
    /// (`MimoPhase::row_grid`), which it records as its rows.
    #[allow(clippy::too_many_arguments)]
    pub fn hidden_norm(
        &self,
        phase: Phase,
        bucket: usize,
        switches: Switches,
        live_tokens: u32,
        input: DeviceBufferSpanV1,
        weight: DeviceBufferSpanV1,
        output: DeviceBufferSpanV1,
        live_count: DeviceBufferSpanV1,
    ) -> Result<Lowered, Refusal> {
        let op = self.norms.get(phase, bucket)?;
        let mut bind = Bind::default();
        let buffers = RmsNormBuffers {
            x: bind.view(input),
            weight: bind.view(weight),
            residual: None,
            y: bind.view(output),
            live_rows: bind.view(live_count),
        };
        let counts = Counts::rows(phase.row_grid(live_tokens, switches, self.row_buckets)?);
        one(bind.launches(op.launches(&buffers, View::EMPTY, counts)?))
    }

    /// Rank `rank`'s EH shard of a `phase` step's normalized `input` rows
    /// (bucket `bucket` of `COMM_BUCKETS`) into `output`, over the phase's
    /// capacity; the device count `live_rows` bounds the rows it copies.
    pub fn mtp_eh_slice(
        &self,
        rank: u32,
        phase: Phase,
        bucket: usize,
        input: DeviceBufferSpanV1,
        output: DeviceBufferSpanV1,
        live_rows: DeviceBufferSpanV1,
    ) -> Result<Lowered, Refusal> {
        let op = self
            .eh_slices
            .get(rank as usize)
            .ok_or(Refusal("MiMo has no EH slice for this rank"))?
            .get(phase, bucket)?;
        let mut bind = Bind::default();
        let buffers = RowCopyBuffers {
            source: bind.aligned(input, 16)?,
            destination: bind.aligned(output, 16)?,
            live_rows: bind.view(live_rows),
        };
        let counts = Counts::rows(phase.capacity()?);
        one(bind.launches(op.launches(&buffers, View::EMPTY, counts)?))
    }
}

fn folds_of(folds: &ByPhase<bool>, phase: Phase) -> Result<&bool, Refusal> {
    folds.get(phase, bucket_of(phase)?)
}

/// The index of `phase`'s comm bucket among `COMM_BUCKETS`.
fn bucket_of(phase: Phase) -> Result<usize, Refusal> {
    let capacity = phase.capacity()?;
    COMM_BUCKETS
        .iter()
        .position(|&rows| rows == capacity)
        .ok_or(Refusal("MiMo has no comm bucket for this phase"))
}

/// An op that asks for no backend workspace: MiMo's arena binds none.
fn no_workspace<B>(op: Prepared<B>) -> Result<Prepared<B>, Refused> {
    if op.info().workspace > 0 {
        return Err(Refused::Rule(
            "MiMo binds no backend workspace for its ops".into(),
        ));
    }
    Ok(op)
}

fn one(mut launches: Vec<Lowered>) -> Result<Lowered, Refusal> {
    if launches.len() != 1 {
        return Err(Refusal("a MiMo op launched other than one kernel"));
    }
    Ok(launches.remove(0))
}

/// The spans one lowering binds: each role's view carries its local
/// binding id as its handle.
#[derive(Default)]
struct Bind(BufferBindings);

impl Bind {
    fn view(&mut self, span: DeviceBufferSpanV1) -> View {
        View {
            handle: self.0.bind(span),
            bytes: span.length,
        }
    }

    /// The view of `span`, refused unless it starts at `alignment` bytes:
    /// views carry no offset, so a role's alignment is the host's duty.
    fn aligned(&mut self, span: DeviceBufferSpanV1, alignment: u64) -> Result<View, Refusal> {
        if !span.offset.is_multiple_of(alignment) {
            return Err(Refusal("a MiMo own op view is off its alignment"));
        }
        Ok(self.view(span))
    }

    /// `launches` over these bindings, with their op's facts.
    fn launches(self, launches: Vec<Launch>) -> Vec<Lowered> {
        let facts = crate::overlap::Facts::of_op(&launches);
        launches
            .into_iter()
            .zip(facts)
            .map(|(launch, facts)| Lowered {
                launch: PreparedLaunch {
                    symbol: launch.symbol,
                    rows: launch.rows,
                    plan: launch.plan,
                    arguments: launch.arguments,
                    bindings: self.0.clone(),
                },
                facts,
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use runtime_bundle_abi::BundleDeviceBufferHandleV1;

    use super::*;

    fn span(offset: u64, length: u64) -> DeviceBufferSpanV1 {
        DeviceBufferSpanV1::new(BundleDeviceBufferHandleV1::from_parts(1, 1), offset, length)
    }

    /// Gather regions that hold any gather.
    pub(super) fn any_gathers(_bucket: usize) -> GatherRegions {
        GatherRegions {
            rows: [u64::MAX; 2],
            exchange: Some([u64::MAX; 2]),
        }
    }

    /// `tp`'s ops on `backend` with BF16 pages and DFlash, whose regions
    /// are `state`, `gathers` and `dflash`.
    pub(super) fn prepare_on(
        backend: &dyn Backend,
        tp: Tp,
        switches: Switches,
        state: &dyn Fn(usize) -> [u64; 3],
        gathers: &dyn Fn(usize) -> GatherRegions,
        dflash: Option<GatherState>,
    ) -> Result<Ops, Refused> {
        Ops::prepare(
            backend,
            &Plan {
                tp,
                switches,
                kv: DType::Bf16,
                items: crate::contract::DECODE_ENTRY_ROWS,
                target: &crate::sm120::own_ops::Sm120Own::new(tp),
                state,
                gathers,
                dflash,
                own: OwnPlan::new(
                    &crate::sm120::own_ops::Sm120Own::new(tp),
                    crate::contract::DECODE_ENTRY_ROWS,
                ),
            },
        )
    }

    /// `tp`'s ops on its SM120 backend, with comm state that holds any op.
    fn prepared(tp: Tp, switches: Switches, wide: bool) -> Result<Ops, Refused> {
        prepare_on(
            &*crate::sm120::backend(tp, switches, wide)?,
            tp,
            switches,
            &|_| [u64::MAX; 3],
            &any_gathers,
            Some([u64::MAX; 2]),
        )
    }

    /// Every setting of what the ops read: the switches `ops` reads (exact
    /// verify and the prefill row grids; the attention and dense switches
    /// reach no op here) and whether graphs may take the wide all-reduces.
    fn settings() -> Vec<(Switches, bool)> {
        let mut settings = Vec::new();
        for exact_verify in [false, true] {
            for capacity_row_grids in [false, true] {
                for wide in [false, true] {
                    let switches = Switches {
                        exact_verify,
                        capacity_row_grids,
                        ..Switches::configured()
                    };
                    settings.push((switches, wide));
                }
            }
        }
        settings
    }

    /// The ops of one topology and setting.
    struct Fixture {
        tp: Tp,
        switches: Switches,
        wide: bool,
        ops: Ops,
    }

    /// Every topology's ops in every setting, prepared once for the tests
    /// that compare them.
    fn fixtures() -> &'static [Fixture] {
        static FIXTURES: std::sync::OnceLock<Vec<Fixture>> = std::sync::OnceLock::new();
        FIXTURES.get_or_init(|| {
            Tp::SUPPORTED
                .into_iter()
                .flat_map(|tp| settings().into_iter().map(move |setting| (tp, setting)))
                .map(|(tp, (switches, wide))| Fixture {
                    tp,
                    switches,
                    wide,
                    ops: prepared(tp, switches, wide).unwrap(),
                })
                .collect()
        })
    }

    /// Every phase with ops and its bucket: prefill, and each decode and
    /// verify bucket.
    pub(super) fn phases() -> Vec<(Phase, usize)> {
        COMM_BUCKETS
            .iter()
            .enumerate()
            .flat_map(|(bucket, &rows)| {
                [Phase::Decode(rows), Phase::Verify(rows), Phase::Prefill]
                    .into_iter()
                    .filter(move |phase| phase.capacity() == Ok(rows))
                    .map(move |phase| (phase, bucket))
            })
            .collect()
    }

    pub(super) type Resolved = Vec<(
        String,
        u32,
        kernel_core::contract::LaunchPlan,
        Vec<runtime_bundle_abi::KernelArgumentV1>,
    )>;

    pub(super) fn resolved(launches: &[PreparedLaunch]) -> Resolved {
        launches
            .iter()
            .map(|launch| {
                (
                    launch.symbol.clone(),
                    launch.rows,
                    launch.plan,
                    crate::dispatch::arguments(&launch.arguments, &launch.bindings).unwrap(),
                )
            })
            .collect()
    }

    /// The prepared hidden norm launches what the row engine planned for
    /// the stage before (`row_plan::initial_norm`): symbol, rows, plan and
    /// resolved arguments, in every setting, at every phase and bucket, at
    /// TP2 and TP4. Both depend on the live rows only through the phase's
    /// row grid, so each distinct grid is compared at the least and the
    /// most live rows that take it.
    #[test]
    fn hidden_norms_launch_what_the_row_engine_planned() {
        let bytes = u64::from(crate::contract::PREFILL_TOKENS) * 4096 * 2;
        let (input, weight, output, live_count) = (
            span(0, bytes),
            span(bytes, 8192),
            span(2 * bytes, bytes),
            span(3 * bytes, 4),
        );
        for fixture in fixtures() {
            let (tp, switches) = (fixture.tp, fixture.switches);
            for (phase, bucket) in phases() {
                let mut grids = std::collections::BTreeMap::<u32, (u32, u32)>::new();
                for live in 1..=phase.capacity().unwrap() {
                    let grid = phase
                        .row_grid(live, switches, crate::sm120::ROW_BUCKETS)
                        .unwrap();
                    let range = grids.entry(grid).or_insert((live, live));
                    range.1 = live;
                }
                for live in grids.values().flat_map(|&(least, most)| [least, most]) {
                    let prepared = fixture
                        .ops
                        .hidden_norm(
                            phase, bucket, switches, live, input, weight, output, live_count,
                        )
                        .unwrap();
                    let legacy = crate::row_plan::initial_norm(
                        tp, phase, switches, live, input, weight, output, live_count,
                    )
                    .unwrap();
                    assert_eq!(
                        resolved(&[prepared.launch]),
                        resolved(&[legacy]),
                        "{tp:?} {switches:?} {phase:?} {live}"
                    );
                }
            }
        }
    }

    /// Prefill norms keep a row's bits in any chunk and exact verify rows
    /// their one-row bits; the backend refuses a norm that cannot.
    #[test]
    fn hidden_norms_claim_what_their_phases_need() {
        let tp = Tp::TP4;
        let switches = Switches {
            exact_verify: true,
            ..Switches::configured()
        };
        assert_eq!(
            need(Phase::Verify(8), switches).unwrap().exact,
            Invariance::one_row(8)
        );
        assert_eq!(
            need(Phase::Prefill, switches).unwrap().exact,
            Invariance::PREFILL
        );
        assert_eq!(
            need(Phase::Verify(16), switches).unwrap().exact,
            Invariance::NONE
        );
        assert!(prepared(tp, switches, true).is_ok());
        // A backend that implements no family refuses MiMo's ops.
        struct Unsupported;
        impl Backend for Unsupported {
            fn name(&self) -> &str {
                "unsupported"
            }
        }
        assert!(
            prepare_on(
                &Unsupported,
                tp,
                switches,
                &|_| [u64::MAX; 3],
                &any_gathers,
                None
            )
            .is_err()
        );
    }

    /// The prepared hidden all-reduces launch what MiMo's comm builders
    /// planned before (`comm_plan::fused_hidden_for`): symbols, rows, plans
    /// and resolved arguments, in every setting, for every rank, every
    /// route a phase has (the phase's, which DFlash takes at every bucket,
    /// and at exact verify buckets the exact one), both input forms where
    /// the bucket folds the combine, and every row count up to the phase's
    /// capacity (lowering passes a graph its bucket and prefill its live
    /// rows), at TP2 and TP4. The legacy launches depend on the setting only
    /// through the route, so each is built once and compared with every
    /// setting's prepared launches that take it.
    #[test]
    fn hidden_reduces_launch_what_the_comm_builders_planned() {
        use crate::sm120::comm_plan::{HiddenBuffers, HiddenRoute};
        let big = u64::from(crate::contract::PREFILL_TOKENS) * 4096 * 2 * 8;
        let (input, residual, weight, output, live) = (
            span(0, big),
            span(big, big),
            span(2 * big, 8192),
            span(3 * big, big),
            span(4 * big, 4),
        );
        for tp in Tp::SUPPORTED {
            let ranks = tp.ranks() as usize;
            let peers = |first: u64| {
                (0..ranks as u64)
                    .map(|rank| span((first + rank) * big, big))
                    .collect::<Vec<_>>()
            };
            let (outputs, inboxes, flags, epochs) = (peers(10), peers(20), peers(30), peers(40));
            let fixtures = fixtures().iter().filter(|f| f.tp == tp).collect::<Vec<_>>();
            let mut exact_compared = vec![false; fixtures.len()];
            for (phase, bucket) in phases() {
                for rank in 0..ranks as u32 {
                    for combine in [false, true] {
                        for rows in 1..=phase.capacity().unwrap() {
                            let views = || ReduceViews {
                                input,
                                residual,
                                norm_weight: weight,
                                output,
                                live_rows: combine.then_some(live),
                                outputs: &outputs,
                                inboxes: &inboxes,
                                flags: &flags,
                                epochs: &epochs,
                            };
                            let mut legacy = std::collections::BTreeMap::new();
                            for (index, fixture) in fixtures.iter().enumerate() {
                                if combine && !fixture.ops.folds_combine(phase, bucket) {
                                    continue;
                                }
                                let exact = phase.exact_rows(fixture.switches);
                                let routes: &[Route] = if exact {
                                    &[Route::Phase, Route::Exact]
                                } else {
                                    &[Route::Phase]
                                };
                                for &route in routes {
                                    exact_compared[index] |= route == Route::Exact;
                                    let legacy_route = match (phase, route) {
                                        (Phase::Prefill, _) => HiddenRoute::Prefill,
                                        (_, Route::Exact) => HiddenRoute::ExactRows,
                                        _ => HiddenRoute::Graph { wide: fixture.wide },
                                    };
                                    let label = || {
                                        format!(
                                            "{tp:?} {:?} wide={} {phase:?} {route:?} \
                                             combine={combine} rank={rank} rows={rows}",
                                            fixture.switches, fixture.wide
                                        )
                                    };
                                    let expected = legacy
                                        .entry(format!("{legacy_route:?}"))
                                        .or_insert_with(|| {
                                            crate::sm120::comm_plan::fused_hidden_for(
                                                tp,
                                                rows,
                                                rank,
                                                legacy_route,
                                                HiddenBuffers {
                                                    input,
                                                    output,
                                                    inboxes: inboxes.clone(),
                                                    flags: flags.clone(),
                                                    epoch: epochs[rank as usize],
                                                    gathered_outputs: outputs.clone(),
                                                    residual,
                                                    residual_output: residual,
                                                    norm_weight: weight,
                                                    combine_live_rows: combine.then_some(live),
                                                },
                                            )
                                            .map(|launches| resolved(&launches))
                                        });
                                    let prepared = fixture
                                        .ops
                                        .hidden_reduce(rank, phase, bucket, route, rows, views())
                                        .map(|launches| {
                                            resolved(&crate::program_lower::launches(launches))
                                        });
                                    match (prepared, expected) {
                                        (Ok(prepared), Ok(expected)) => {
                                            assert_eq!(&prepared, expected, "{}", label())
                                        }
                                        (prepared, expected) => panic!(
                                            "{}: prepared {:?}, legacy {:?}",
                                            label(),
                                            prepared.err(),
                                            expected.as_ref().err()
                                        ),
                                    }
                                }
                            }
                        }
                    }
                }
            }
            for (fixture, compared) in fixtures.iter().zip(exact_compared) {
                assert_eq!(compared, fixture.switches.exact_verify, "{tp:?}");
            }
        }
    }

    /// Every symbol a prepared op launches is in a cubin a `tp` instance
    /// loads.
    #[test]
    fn prepared_ops_launch_only_loaded_entries() {
        for fixture in fixtures() {
            let loaded = crate::CUBINS
                .iter()
                .filter(|cubin| crate::topology::cubin_serves(cubin.name, fixture.tp))
                .flat_map(|cubin| cubin.symbols.iter().copied())
                .collect::<std::collections::BTreeSet<_>>();
            let ops = &fixture.ops;
            let mut infos = Vec::new();
            for (phase, bucket) in phases() {
                infos.extend(ops.norms.get(phase, bucket).map(|op| op.info()));
                for reduces in &ops.reduces {
                    for set in [
                        &reduces.plain,
                        &reduces.exact,
                        &reduces.combine,
                        &reduces.combine_exact,
                    ] {
                        infos.extend(set.get(phase, bucket).map(|op| op.info()));
                    }
                }
            }
            assert!(!infos.is_empty());
            for info in infos {
                for (symbol, _) in &info.symbols {
                    assert!(
                        loaded.contains(symbol.as_str()),
                        "{symbol} at {:?}",
                        fixture.tp
                    );
                }
            }
        }
    }

    /// An instance whose comm state of some bucket is smaller than an
    /// all-reduce of the bucket needs refuses the op when it is created.
    #[test]
    fn undersized_comm_state_refuses_at_create() {
        let tp = Tp::TP4;
        let switches = Switches::configured();
        let backend = crate::sm120::backend(tp, switches, true).unwrap();
        let fits = prepare_on(
            &*backend,
            tp,
            switches,
            &|_| [u64::MAX; 3],
            &any_gathers,
            Some([u64::MAX; 2]),
        );
        assert!(fits.is_ok());
        for region in 0..3 {
            let short = prepare_on(
                &*backend,
                tp,
                switches,
                &|bucket| {
                    let mut held = [u64::MAX; 3];
                    if COMM_BUCKETS[bucket] == 8 {
                        held[region] = 1;
                    }
                    held
                },
                &any_gathers,
                Some([u64::MAX; 2]),
            );
            assert!(
                matches!(&short, Err(Refused::Rule(rule)) if rule.contains("comm state")),
                "region {region}: {:?}",
                short.err()
            );
        }
    }

    /// Exact verify buckets take the exact entries above the low-latency
    /// rows, and their ops claim every row's one-row bits at the bucket's
    /// rows, at TP2 and TP4.
    #[test]
    fn exact_reduces_claim_their_buckets_one_row_bits() {
        let switches = Switches {
            exact_verify: true,
            ..Switches::configured()
        };
        for tp in Tp::SUPPORTED {
            let ops = prepared(tp, switches, true).unwrap();
            for (phase, bucket) in phases() {
                let Phase::Verify(rows) = phase else {
                    continue;
                };
                if !phase.exact_rows(switches) {
                    continue;
                }
                for reduces in &ops.reduces {
                    for set in [&reduces.exact, &reduces.combine_exact] {
                        if let Ok(op) = set.get(phase, bucket) {
                            assert!(
                                op.info().guarantees.provides(Invariance::one_row(rows)),
                                "{tp:?} verify {rows}"
                            );
                        }
                    }
                }
            }
        }
        let ops = prepared(Tp::TP4, switches, true).unwrap();
        let bucket = COMM_BUCKETS.iter().position(|&rows| rows == 8).unwrap();
        let exact = ops.reduces[0].exact.get(Phase::Verify(8), bucket).unwrap();
        assert!(
            exact
                .info()
                .symbols
                .iter()
                .any(|(symbol, _)| symbol == kernels_verify_symbol())
        );
        // Without exact verify no op takes the exact entries.
        let ops = prepared(
            Tp::TP4,
            Switches {
                exact_verify: false,
                ..switches
            },
            true,
        )
        .unwrap();
        assert!(ops.reduces[0].exact.get(Phase::Verify(8), bucket).is_err());
    }
    fn kernels_verify_symbol() -> &'static str {
        crate::sm120::comm::exact_entries(Tp::TP4).unwrap().0
    }
}
