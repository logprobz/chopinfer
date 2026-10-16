//! The all-reduces: each sublayer's all-reduce of BF16 partials, rounded
//! into the rows the Sinkhorn expansion reads, the comm epilogue that
//! composes it with the expansion and the next mixer and input norm, and
//! the MoE sublayers' rounded all-reduce of F32 partials.
//!
//! The low-latency and phased entries spin on their counterparts on the
//! peer ranks, so create admits them against every rank's residency. Each
//! admission outcome registers its own entries (`Admission`), so the ops
//! are prepared for every outcome on its backend (`CommBackends`), and a
//! step takes the outcome create left.

use std::sync::Arc;

use kernel_core::backend::{Backend, Counts, Prepared, Refused, View};
use kernel_core::collective::{AllReduceBuffers, RankViews};
use kernel_core::contract::{CommQuery, DType, Refusal};
use kernel_core::sinkhorn_hyper_connection::{
    HyperConnectionPlacement, MixScratch, MixerWeights, ReduceExpandMixNormBuffers,
    SinkhornHyperConnectionQuery, SinkhornMix, SinkhornOrder,
};
use model_glm5next_contract::contract::{PREFILL_TOKENS, Shape};
use runtime_bundle_abi::DeviceBufferSpanV1;

use super::{Asks, Bind, Lowered, exact};

/// Which spinning all-reduce entries create admitted on every rank: the
/// low-latency entry and the phased entries.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Admission {
    pub low_latency: bool,
    pub phased: bool,
}

impl Admission {
    /// Every outcome create can leave.
    pub const ALL: [Admission; 4] = [
        Admission {
            low_latency: true,
            phased: true,
        },
        Admission {
            low_latency: false,
            phased: true,
        },
        Admission {
            low_latency: true,
            phased: false,
        },
        Admission {
            low_latency: false,
            phased: false,
        },
    ];
}

/// The backend of each admission outcome, registered with the entries it
/// admits (`crate::sm120::backend`).
#[derive(Clone, Default)]
pub struct CommBackends(pub Vec<(Admission, Arc<dyn Backend>)>);

impl std::fmt::Debug for CommBackends {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_list()
            .entries(self.0.iter().map(|(admission, _)| admission))
            .finish()
    }
}

/// One rank's ops of one admission outcome.
struct RankReduces {
    /// The sublayer all-reduce, rounded into the expansion's rows.
    reduce: Prepared<AllReduceBuffers<View>>,
    /// The MoE sublayers' all-reduce of F32 partials, rounded once into the
    /// expansion's rows.
    rounded: Prepared<AllReduceBuffers<View>>,
    /// The comm epilogue (`runtime.sinkhorn_comm_epilogue`).
    epilogue: Prepared<ReduceExpandMixNormBuffers<View>>,
}

/// The all-reduces of an instance: each admission outcome's per rank, with
/// the comm state regions they keep.
pub(crate) struct ReduceOps {
    by_admission: Vec<(Admission, Vec<RankReduces>)>,
    regions: StateRegions,
}

/// The bytes of each comm state region the all-reduces keep on every rank,
/// as their ops list them (`OpInfo::state`): the low-latency entry's inbox
/// and epochs, the phased entries' inbox, flags and epoch, the F32 pair's
/// inbox, flags and epoch and its F32 sum, then the rounded all-reduce's own
/// low-latency and phased states (its composed form shares the pair and
/// the sum).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct StateRegions {
    pub low_latency: [u64; 2],
    pub phased: [u64; 3],
    pub pair: [u64; 3],
    pub sum: u64,
    pub rounded_low_latency: [u64; 2],
    pub rounded_phased: [u64; 3],
}

impl StateRegions {
    /// The regions an all-reduce of BF16 partials under `admission` keeps,
    /// in its role order: the low-latency entry's where it is admitted, the
    /// phased state where it is admitted, else the F32 pair's, then the sum
    /// (`CommState::all_reduce_regions` binds them in this order).
    fn of(&self, admission: Admission) -> Vec<u64> {
        let mut regions = Vec::new();
        if admission.low_latency {
            regions.extend(self.low_latency);
        }
        regions.extend(match admission.phased {
            true => self.phased,
            false => self.pair,
        });
        regions.push(self.sum);
        regions
    }

    /// The regions the rounded all-reduce under `admission` keeps, in its
    /// role order: its low-latency state where admitted, its phased state
    /// where admitted, else the F32 pair's, then the sum
    /// (`CommState::rounded_regions` binds them in this order).
    fn rounded_of(&self, admission: Admission) -> Vec<u64> {
        let mut regions = Vec::new();
        if admission.low_latency {
            regions.extend(self.rounded_low_latency);
        }
        regions.extend(match admission.phased {
            true => self.rounded_phased,
            false => self.pair,
        });
        regions.push(self.sum);
        regions
    }

    /// The regions of every op in `ops`, refusing any that lists other
    /// regions than its admission's: `ops[0]` (both entries admitted)
    /// gives the low-latency and phased regions, `ops[3]` (neither) the
    /// pair's and the sum.
    fn read(ops: &[(Admission, Vec<RankReduces>)]) -> Result<Self, Refused> {
        let refused = |what: &str| {
            Refused::Rule(format!(
                "GLM 5.3 Flash's {what} keep other comm state regions than it binds"
            ))
        };
        let first = |at: usize| -> Result<&RankReduces, Refused> {
            let (_, ranks) = ops.get(at).ok_or_else(|| refused("all-reduces"))?;
            ranks.first().ok_or_else(|| refused("all-reduces"))
        };
        let (both, neither) = (first(0)?, first(3)?);
        let (
            &[ll_inbox, ll_epochs, inbox, flags, epoch],
            &[pair_inbox, pair_flags, pair_epoch, sum],
            &[
                rounded_inbox,
                rounded_epochs,
                rounded_phased_inbox,
                rounded_flags,
                rounded_epoch,
                ..,
            ],
        ) = (
            &both.reduce.info().state[..],
            &neither.reduce.info().state[..],
            &both.rounded.info().state[..],
        )
        else {
            return Err(refused("all-reduces"));
        };
        let regions = Self {
            low_latency: [ll_inbox, ll_epochs],
            phased: [inbox, flags, epoch],
            pair: [pair_inbox, pair_flags, pair_epoch],
            sum,
            rounded_low_latency: [rounded_inbox, rounded_epochs],
            rounded_phased: [rounded_phased_inbox, rounded_flags, rounded_epoch],
        };
        let fits = |state: &[u64], all: &[u64]| {
            let n = state.len();
            n + 1 >= all.len() && n <= all.len() && state[..] == all[..n]
        };
        for (admission, ranks) in ops {
            let (all, rounded) = (regions.of(*admission), regions.rounded_of(*admission));
            for rank in ranks {
                if ![&rank.reduce.info().state, &rank.epilogue.info().state]
                    .iter()
                    .all(|state| fits(state, &all))
                    || !fits(&rank.rounded.info().state, &rounded)
                {
                    return Err(refused("all-reduces"));
                }
            }
        }
        Ok(regions)
    }
}

/// One hidden row of a MoE sublayer's all-reduce: every rank's F32
/// partial summed in F32 in rank order from zero and rounded once to BF16.
pub(crate) fn rounded_query(shape: Shape) -> CommQuery {
    CommQuery::AllReduceRounded {
        ranks: shape.ranks(),
        elements: u64::from(shape.profile.hidden),
    }
}

/// One hidden row of a sublayer's all-reduce: every rank's F32 partial
/// rounded once to BF16, summed in rank order from F32 zero and rounded
/// once.
pub(crate) fn reduce_query(shape: Shape) -> CommQuery {
    CommQuery::AllReduce {
        ranks: shape.ranks(),
        elements: u64::from(shape.profile.hidden),
        dtype: DType::Bf16,
        fused_residual: false,
        fused_rmsnorm: None,
    }
}

/// The Sinkhorn mHC mixer over the residual streams.
fn mix(shape: Shape) -> SinkhornMix {
    let g = shape.geometry;
    SinkhornMix {
        streams: shape.profile.hc_streams,
        width: shape.profile.hidden,
        iterations: g.shape.hc.sinkhorn_iterations,
        eps_bits: (g.shape.hc.eps as f32).to_bits(),
        norm_eps_bits: g.eps().to_bits(),
        order: SinkhornOrder::RowSoftmaxThenColumnFirst,
        placement: HyperConnectionPlacement::PerSublayer,
        fn_dtype: DType::Bf16,
    }
}

/// The comm epilogue: the all-reduce of BF16 partials, then the expansion,
/// the next sublayer's mix-collapse and its input norm, whose squares sum
/// in the order every hidden-width norm of the model sums them (the
/// 256-lane tree, [`Shape::rmsnorm_requirement`]).
fn epilogue_query(shape: Shape) -> SinkhornHyperConnectionQuery {
    let hidden = shape.profile.hidden;
    let norm = shape
        .rmsnorm_requirement(hidden)
        .numerics
        .expect("the hidden-width norm pins its numerics")
        .reduction;
    SinkhornHyperConnectionQuery::AllReduceExpandMixNorm {
        ranks: shape.ranks(),
        dtype: DType::Bf16,
        mix: mix(shape),
        rmsnorm_eps_bits: shape.geometry.eps().to_bits(),
        norm,
        capacity: PREFILL_TOKENS,
    }
}

impl ReduceOps {
    /// The all-reduces of an instance of `shape` on the backend of each
    /// admission outcome.
    pub(crate) fn prepare(comm: &CommBackends, shape: Shape, asks: &Asks) -> Result<Self, Refused> {
        let ranks = shape.ranks();
        let by_admission = Admission::ALL
            .iter()
            .map(|&admission| {
                // Each phase reduces through the outcome create left, and
                // every outcome asks what every phase needs: with the
                // phased entries admitted every row count launches an entry
                // that declares it, without them some take the composed
                // form, which claims what GLM's rounding rows and the F32
                // pair declare together.
                let need = exact::need(PREFILL_TOKENS, exact::every_phase(exact::verify_rows()));
                let backend = comm
                    .0
                    .iter()
                    .find(|(of, _)| *of == admission)
                    .map(|(_, backend)| backend)
                    .ok_or(Refused::Rule(format!(
                        "GLM 5.3 Flash has no backend for {admission:?}"
                    )))?;
                let ops = (0..ranks)
                    .map(|rank| {
                        let what = |op| format!("reduce.{op} {admission:?} rank {rank}");
                        Ok(RankReduces {
                            reduce: asks.ask(what("all_reduce"), need, |need| {
                                backend.all_reduce(&reduce_query(shape), rank, need)
                            })?,
                            rounded: asks.ask(what("rounded"), need, |need| {
                                backend.all_reduce(&rounded_query(shape), rank, need)
                            })?,
                            // The comm epilogue claims what its
                            // all-reduce, widen row and split entry declare
                            // together.
                            epilogue: asks.ask(what("epilogue"), need, |need| {
                                backend.sinkhorn_reduce_expand_mix_norm(
                                    &epilogue_query(shape),
                                    rank,
                                    need,
                                )
                            })?,
                        })
                    })
                    .collect::<Result<Vec<_>, Refused>>()?;
                Ok((admission, ops))
            })
            .collect::<Result<Vec<_>, Refused>>()?;
        let regions = StateRegions::read(&by_admission)?;
        Ok(Self {
            by_admission,
            regions,
        })
    }

    /// The comm state regions every rank allocates for these ops.
    pub(crate) fn regions(&self) -> StateRegions {
        self.regions
    }

    fn rank(&self, admission: Admission, rank: u32) -> Result<&RankReduces, Refusal> {
        self.by_admission
            .iter()
            .find(|(of, _)| *of == admission)
            .and_then(|(_, ops)| ops.get(rank as usize))
            .ok_or(Refusal("GLM 5.3 Flash has no all-reduce for the rank"))
    }

    /// The state regions of rank `rank`'s sublayer all-reduce under
    /// `admission`, in role order (`OpInfo::state`).
    pub(crate) fn reduce_state(&self, admission: Admission, rank: u32) -> Result<&[u64], Refusal> {
        Ok(&self.rank(admission, rank)?.reduce.info().state)
    }

    /// The state regions of rank `rank`'s MoE sublayer all-reduce under
    /// `admission`, in role order (`OpInfo::state`).
    pub(crate) fn rounded_state(&self, admission: Admission, rank: u32) -> Result<&[u64], Refusal> {
        Ok(&self.rank(admission, rank)?.rounded.info().state)
    }

    /// The state regions of rank `rank`'s comm epilogue under `admission`.
    pub(crate) fn epilogue_state(
        &self,
        admission: Admission,
        rank: u32,
    ) -> Result<&[u64], Refusal> {
        Ok(&self.rank(admission, rank)?.epilogue.info().state)
    }

    /// The launches of rank `rank`'s sublayer all-reduce of `rows` rows
    /// under `admission`.
    pub(crate) fn reduce(
        &self,
        admission: Admission,
        rank: u32,
        rows: u32,
        views: &ReduceViews,
    ) -> Result<Vec<Lowered>, Refusal> {
        let op = &self.rank(admission, rank)?.reduce;
        let mut bind = Bind::default();
        let buffers = views.bind(&mut bind);
        bind.launch(op, &buffers, Counts::rows(rows))
    }

    /// The launches of rank `rank`'s MoE sublayer all-reduce of `rows` rows
    /// under `admission`.
    pub(crate) fn rounded(
        &self,
        admission: Admission,
        rank: u32,
        rows: u32,
        views: &ReduceViews,
    ) -> Result<Vec<Lowered>, Refusal> {
        let op = &self.rank(admission, rank)?.rounded;
        let mut bind = Bind::default();
        let buffers = views.bind(&mut bind);
        bind.launch(op, &buffers, Counts::rows(rows))
    }

    /// The launches of rank `rank`'s comm epilogue of `rows` rows under
    /// `admission`.
    pub(crate) fn epilogue(
        &self,
        admission: Admission,
        rank: u32,
        rows: u32,
        views: &EpilogueViews,
    ) -> Result<Vec<Lowered>, Refusal> {
        let op = &self.rank(admission, rank)?.epilogue;
        let mut bind = Bind::default();
        let reduce = views.reduce.bind(&mut bind);
        let mut view = |span| bind.view(span);
        let buffers = ReduceExpandMixNormBuffers {
            reduce,
            streams: view(views.streams),
            reduced: view(views.reduced),
            expanded: view(views.expanded),
            mixer: MixerWeights {
                fn_: view(views.mixer[0]),
                base: view(views.mixer[1]),
                scale: view(views.mixer[2]),
            },
            post: view(views.post),
            comb: view(views.comb),
            collapsed: view(views.collapsed),
            norm_weight: view(views.norm_weight),
            normed: view(views.normed),
            scratch: MixScratch {
                partials: Some(view(views.partials)),
                tickets: Some(view(views.tickets)),
            },
            live_rows: view(views.live_rows),
        };
        bind.launch(op, &buffers, Counts::rows(rows))
    }
}

/// One peer's spans of an all-reduce: its output rows and its state
/// regions in the op's role order.
pub(crate) struct PeerViews {
    pub output: DeviceBufferSpanV1,
    pub state: Vec<DeviceBufferSpanV1>,
}

/// A step's spans of an all-reduce: this rank's input and output rows, the
/// live row count and every rank's peer views in rank order.
pub(crate) struct ReduceViews {
    pub input: DeviceBufferSpanV1,
    pub output: DeviceBufferSpanV1,
    pub live_rows: Option<DeviceBufferSpanV1>,
    pub peers: Vec<PeerViews>,
}

impl ReduceViews {
    fn bind(&self, bind: &mut Bind) -> AllReduceBuffers<View> {
        AllReduceBuffers {
            input: bind.view(self.input),
            output: bind.view(self.output),
            live_rows: self.live_rows.map(|span| bind.view(span)),
            ranks: self
                .peers
                .iter()
                .map(|peer| RankViews {
                    output: bind.view(peer.output),
                    state: peer.state.iter().map(|&span| bind.view(span)).collect(),
                })
                .collect(),
        }
    }
}

/// A step's spans of a comm epilogue: its all-reduce's, then the streams it
/// expands from and into, the next mixer's weights (`fn`, base, scale) and
/// rows, the next input norm's weight and output, the split's partials and
/// tickets and the live row count.
pub(crate) struct EpilogueViews {
    pub reduce: ReduceViews,
    pub streams: DeviceBufferSpanV1,
    pub reduced: DeviceBufferSpanV1,
    pub expanded: DeviceBufferSpanV1,
    pub mixer: [DeviceBufferSpanV1; 3],
    pub post: DeviceBufferSpanV1,
    pub comb: DeviceBufferSpanV1,
    pub collapsed: DeviceBufferSpanV1,
    pub norm_weight: DeviceBufferSpanV1,
    pub normed: DeviceBufferSpanV1,
    pub partials: DeviceBufferSpanV1,
    pub tickets: DeviceBufferSpanV1,
    pub live_rows: DeviceBufferSpanV1,
}
