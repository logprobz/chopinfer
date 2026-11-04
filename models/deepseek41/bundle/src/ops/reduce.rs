//! The all-reduces: each sublayer's F32 rank partials summed in F32 in rank
//! order and rounded once to BF16 (`CommQuery::AllReduceRounded`, the
//! vendor's F32 all-reduce before its BF16 cast in `RowParallelLinear`,
//! `model.py:271-278`), and the embedding's BF16 shard rows widened into F32
//! partials (an own op) and reduced the same way, which gives back each
//! row's bits as one rank holds it.
//! Flash TP4 prefills at either KV storage default to the separate BF16
//! AllReduce contract, rounding each completed local partial before
//! transport.
//! Decode, verification and other shapes retain F32 partials.
//!
//! Each rank's op keeps comm state regions (`OpInfo::state`) that its
//! peers write; every rank allocates the same regions, zeroed at create.
//! With the `phased_decode_reduce` candidate option, decode and verify
//! steps' complete all-reduces (those whose round no HC boundary defers)
//! take the phased F32 rounding entries over state regions of their own,
//! which give the composed form's bits.

use kernel_core::backend::{Backend, Counts, Prepared, Refused, View};
use kernel_core::collective::{AllReduceBuffers, RankViews};
use kernel_core::contract::{CommQuery, DType, Refusal};
use model_deepseek41_contract::own_ops::OwnOps;
use model_deepseek41_contract::own_ops::widen::{WidenRows, WidenRowsBuffers};
use model_deepseek41_contract::serving::PREFILL_TOKENS;
use model_deepseek41_contract::state::KvStorage;
use runtime_bundle_abi::DeviceBufferSpanV1;

use super::{Asks, Bind, Lowered, exact};
use crate::program::Phase;
use crate::shape::Shape;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Partials {
    Off,
    Prefill,
}

/// Both KV storages take the same prefill partials, so BF16 storage
/// differs from native in its stored KV alone.
fn partials_mode(shape: &Shape, _storage: KvStorage) -> Partials {
    if shape.ranks() == 4 && shape.geometry == &model_deepseek41_contract::geometry::FLASH {
        Partials::Prefill
    } else {
        Partials::Off
    }
}

pub(crate) fn bf16_partials_requested(shape: &Shape, storage: KvStorage) -> bool {
    partials_mode(shape, storage) == Partials::Prefill
}

pub(crate) fn prefill_only_partials(shape: &Shape, storage: KvStorage) -> bool {
    bf16_partials_requested(shape, storage)
}

pub(crate) fn requested(shape: &Shape) -> bool {
    shape.geometry == &model_deepseek41_contract::geometry::FLASH
}

/// Whether the instance takes the phased decode reduction: the
/// `phased_decode_reduce` option on Flash TP4, whose rows are whole
/// 1024-element granules. An execute issues at most one decode or verify
/// phase of fewer than 256 layers (`RESET_RUNS` in `instance/prefill_reduce.rs`).
pub(crate) fn phased_decode_requested(shape: &Shape) -> bool {
    shape.phased_decode_reduce
        && requested(shape)
        && shape.ranks() == 4
        && shape.dims().hidden.is_multiple_of(1024)
        && shape.geometry.layers() < 256
}

/// The element format a prefill DMA reduction copies between ranks.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum DmaPayload {
    F32,
    Bf16,
}

impl DmaPayload {
    pub(crate) fn bytes(self) -> u64 {
        match self {
            Self::F32 => 4,
            Self::Bf16 => 2,
        }
    }
}

/// A row-local launch that may run on part of a DMA call's interval: which
/// arguments hold rows and how many bytes each row takes there. Every other
/// argument is the same for every row.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct RowLocal {
    pub(crate) kind: RowLocalKind,
    pub(crate) rows: Vec<(usize, u64)>,
    /// The argument bound to the call's own rows, if any: the reduced rows a
    /// consumer reads, or the partial rows a producer writes.
    pub(crate) call_rows: Option<usize>,
    /// The device cell holding the live row count.
    pub(crate) live: usize,
    /// A U64 argument that addresses rows from an explicit first row.
    pub(crate) row_offset: Option<usize>,
    /// Row arguments that may instead be null.
    pub(crate) optional: Vec<(usize, u64)>,
    /// Whether row views other than the call's rows may be longer than the
    /// launch's rows: capacity views the launch reads from their first row.
    pub(crate) capacity_views: bool,
    /// The grid dimension counting rows, and the rows each step covers.
    pub(crate) grid: (usize, u32),
}

/// The consumers that follow a hidden reduction, in their served order, and
/// the producer of its partial rows.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum RowLocalKind {
    Expand,
    Mix,
    Norm,
    /// The MoE route-row sum that writes the call's F32 partial rows.
    Produce,
}

/// Interval consumers remain subject to their row and operation gates.
pub(crate) fn dma_consume_requested() -> bool {
    true
}

/// Interval producers remain subject to their BF16 payload and row gates.
pub(crate) fn dma_produce_requested() -> bool {
    true
}

/// Source-stream copies require the host's successful capability probe.
pub(crate) fn dma_push_requested() -> bool {
    true
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum PrefillDma {
    Off,
    Skip,
}

impl PrefillDma {
    /// DMA serves negotiated upper-skip chunks with a prepared BF16 gather.
    pub(crate) fn admit(
        self,
        prefill: model_deepseek41_contract::prefill_profile::Profile,
        bf16_gather: bool,
    ) -> Self {
        match self {
            Self::Skip if !(prefill.skip && prefill.chunk > 0 && bf16_gather) => Self::Off,
            policy => policy,
        }
    }

    pub(crate) fn enabled(self) -> bool {
        self != Self::Off
    }
}

pub(crate) fn dma_requested() -> PrefillDma {
    PrefillDma::Skip
}

/// An op set beside the main one, over its own comm state regions, which
/// create must admit on every rank before use.
struct Alternate {
    reduces: Vec<Prepared<AllReduceBuffers<View>>>,
    /// Its regions in `ReduceOps::state`.
    state: std::ops::Range<usize>,
    admitted: bool,
}

/// The all-reduces of an instance, each prepared for a prefill's rows.
pub(crate) struct ReduceOps {
    bf16_partials: Partials,
    /// Each rank's rounded all-reduce of hidden rows, in rank order.
    reduces: Vec<Prepared<AllReduceBuffers<View>>>,
    /// The embedding's widening into F32 partials.
    widen: Prepared<WidenRowsBuffers<View>>,
    /// The bytes of each comm state region every rank keeps, in the ops'
    /// role order: the main ops', then each alternate's.
    state: Vec<u64>,
    /// The main ops' regions: the first of `state`.
    main_state: usize,
    width: u32,
    prefill: Option<Alternate>,
    /// The `phased_decode_reduce` candidate's phased F32 rounding all-reduce
    /// of decode and verify steps.
    phased_decode: Option<Alternate>,
}

/// One hidden row of the all-reduce: every rank's F32 partial summed in F32
/// in rank order from zero and rounded once to BF16.
pub(crate) fn query(shape: &Shape, _storage: KvStorage) -> CommQuery {
    rounded_query(shape)
}

fn rounded_query(shape: &Shape) -> CommQuery {
    CommQuery::AllReduceRounded {
        ranks: shape.ranks(),
        elements: u64::from(shape.dims().hidden),
    }
}

fn partial_query(shape: &Shape) -> CommQuery {
    CommQuery::AllReduce {
        ranks: shape.ranks(),
        elements: u64::from(shape.dims().hidden),
        dtype: DType::Bf16,
        fused_residual: false,
        fused_rmsnorm: None,
    }
}

/// One peer's spans of an all-reduce: its output rows and its comm state
/// regions in role order.
pub(crate) struct PeerViews {
    pub output: DeviceBufferSpanV1,
    pub state: Vec<DeviceBufferSpanV1>,
}

/// A step's spans of an all-reduce: this rank's F32 partials and BF16
/// output rows, the live row count and every rank's peer views in rank
/// order.
pub(crate) struct ReduceViews {
    pub input: DeviceBufferSpanV1,
    pub output: DeviceBufferSpanV1,
    pub live_rows: DeviceBufferSpanV1,
    pub peers: Vec<PeerViews>,
}

/// A step's spans of the embedding's widening: the BF16 shard rows, the
/// F32 partials and the live row count.
pub(crate) struct WidenViews {
    pub input: DeviceBufferSpanV1,
    pub output: DeviceBufferSpanV1,
    pub live_rows: DeviceBufferSpanV1,
}

impl ReduceOps {
    /// Every rank's all-reduce of `shape` and the embedding's widening.
    pub(crate) fn prepare(
        backend: &dyn Backend,
        own: &dyn OwnOps,
        shape: &Shape,
        storage: KvStorage,
        asks: &Asks,
    ) -> Result<ReduceOps, Refused> {
        // Every phase reduces its sublayers.
        let need = exact::need(PREFILL_TOKENS, exact::every_phase(shape.verify_rows));
        let query = query(shape, storage);
        let reduces = (0..shape.ranks())
            .map(|rank| {
                asks.ask(format!("reduce.all_reduce rank {rank}"), need, |need| {
                    backend.all_reduce(&query, rank, need)
                })
            })
            .collect::<Result<Vec<_>, Refused>>()?;
        let state = reduces
            .first()
            .map(|op| op.info().state.clone())
            .unwrap_or_default();
        if reduces.iter().any(|op| op.info().state != state) {
            return Err(Refused::Rule(format!(
                "{}'s ranks keep different all-reduce state regions",
                super::NAME
            )));
        }
        let widen = asks.ask("reduce.widen", need, |need| {
            own.widen_rows(
                &WidenRows {
                    width: shape.dims().hidden,
                },
                need,
            )
        })?;
        Ok(ReduceOps {
            bf16_partials: partials_mode(shape, storage),
            reduces,
            widen,
            main_state: state.len(),
            state,
            width: shape.dims().hidden,
            prefill: None,
            phased_decode: None,
        })
    }

    /// The prefill's BF16-partial all-reduces: a separate operation and
    /// state range keep decode and verify on their original transport.
    /// Create must admit every rank before use.
    pub(crate) fn prepare_prefill(
        &mut self,
        backend: &dyn Backend,
        shape: &Shape,
        asks: &Asks,
    ) -> Result<(), Refused> {
        let need = exact::need(PREFILL_TOKENS, exact::every_phase(shape.verify_rows));
        let query = partial_query(shape);
        let reduces = (0..shape.ranks())
            .map(|rank| {
                asks.ask(format!("reduce.prefill_bf16 rank {rank}"), need, |need| {
                    backend.all_reduce(&query, rank, need)
                })
            })
            .collect::<Result<Vec<_>, Refused>>()?;
        let state = &reduces[0].info().state;
        if reduces.iter().any(|op| op.info().state != *state) {
            return Err(Refused::Rule(
                "prefill reduction ranks disagree on state".into(),
            ));
        }
        let first_state = self.state.len();
        self.state.extend_from_slice(state);
        self.prefill = Some(Alternate {
            reduces,
            state: first_state..self.state.len(),
            admitted: false,
        });
        Ok(())
    }

    /// The `phased_decode_reduce` candidate: every rank's rounded
    /// all-reduce on `backend`'s phased F32 entries for decode and verify
    /// steps of up to the instance's verify rows, over state regions of its
    /// own. Create must admit every rank before use.
    pub(crate) fn prepare_phased_decode(
        &mut self,
        backend: &dyn Backend,
        shape: &Shape,
        asks: &Asks,
    ) -> Result<(), Refused> {
        let need = exact::need(shape.verify_rows, exact::verify(shape.verify_rows));
        let query = rounded_query(shape);
        let reduces = (0..shape.ranks())
            .map(|rank| {
                asks.ask(format!("reduce.phased_decode rank {rank}"), need, |need| {
                    backend.all_reduce(&query, rank, need)
                })
            })
            .collect::<Result<Vec<_>, Refused>>()?;
        let state = &reduces[0].info().state;
        if reduces.iter().any(|op| op.info().state != *state) {
            return Err(Refused::Rule(
                "phased decode reduction ranks disagree on state".into(),
            ));
        }
        let first_state = self.state.len();
        self.state.extend_from_slice(state);
        self.phased_decode = Some(Alternate {
            reduces,
            state: first_state..self.state.len(),
            admitted: false,
        });
        Ok(())
    }

    pub(crate) fn phased_decode_prepared(&self) -> bool {
        self.phased_decode.is_some()
    }

    pub(crate) fn admit_phased_decode(&mut self, admitted: bool) {
        if let Some(phased) = &mut self.phased_decode {
            phased.admitted = admitted;
        }
    }

    /// The admitted phased decode reduction's state regions.
    pub(crate) fn phased_decode_state(&self) -> Option<std::ops::Range<usize>> {
        self.phased_decode
            .as_ref()
            .filter(|p| p.admitted)
            .map(|p| p.state.clone())
    }

    pub(crate) fn prefill_prepared(&self) -> bool {
        self.prefill.is_some()
    }

    pub(crate) fn bf16_partials(&self) -> bool {
        self.bf16_partials != Partials::Off
    }

    pub(crate) fn admit_prefill(&mut self, admitted: bool) {
        if let Some(prefill) = &mut self.prefill {
            prefill.admitted = admitted;
        }
    }

    pub(crate) fn prefill_state(&self) -> Option<std::ops::Range<usize>> {
        self.prefill
            .as_ref()
            .filter(|p| p.admitted)
            .map(|p| p.state.clone())
    }

    /// The bytes of each comm state region every rank allocates.
    pub(crate) fn state(&self) -> &[u64] {
        &self.state
    }

    /// The launches of rank `rank`'s all-reduce of `rows` rows.
    pub(crate) fn reduce(
        &self,
        rank: u32,
        phase: Phase,
        views: &ReduceViews,
    ) -> Result<Vec<Lowered>, Refusal> {
        let rows = phase.rows();
        let (ops, state) = match (&self.prefill, &self.phased_decode) {
            (Some(p), _) if p.admitted && phase.prefill() => (&p.reduces, p.state.clone()),
            (_, Some(p)) if p.admitted && !phase.prefill() => (&p.reduces, p.state.clone()),
            _ => (&self.reduces, 0..self.main_state),
        };
        let op = ops
            .get(rank as usize)
            .ok_or(Refusal("DeepSeek V4.1 has no all-reduce for this rank"))?;
        let mut bind = Bind::default();
        let buffers = AllReduceBuffers {
            input: bind.view(views.input),
            output: bind.view(views.output),
            live_rows: Some(bind.view(views.live_rows)),
            ranks: views
                .peers
                .iter()
                .map(|peer| {
                    Ok(RankViews {
                        output: bind.view(peer.output),
                        state: peer
                            .state
                            .get(state.clone())
                            .ok_or(Refusal("prefill reduction state is missing"))?
                            .iter()
                            .map(|&span| bind.view(span))
                            .collect(),
                    })
                })
                .collect::<Result<_, Refusal>>()?,
        };
        bind.launch(op, &buffers, Counts::rows(rows))
    }

    /// Keep the exact collective prefix while a following op writes the
    /// BF16 round. The producer owns the F32 sum's state layout.
    pub(crate) fn defer_round(
        &self,
        rank: u32,
        rows: u32,
        views: &ReduceViews,
    ) -> Result<Option<(Vec<Lowered>, DeviceBufferSpanV1)>, Refusal> {
        let op = self
            .reduces
            .get(rank as usize)
            .ok_or(Refusal("no all-reduce for this rank"))?;
        // Decode keeps the original transport's state, ahead of any
        // alternate's range.
        let state = 0..self.main_state;
        let mut bind = Bind::default();
        let buffers = AllReduceBuffers {
            input: bind.view(views.input),
            output: bind.view(views.output),
            live_rows: Some(bind.view(views.live_rows)),
            ranks: views
                .peers
                .iter()
                .map(|peer| {
                    Ok(RankViews {
                        output: bind.view(peer.output),
                        state: peer
                            .state
                            .get(state.clone())
                            .ok_or(Refusal("reduction state is missing"))?
                            .iter()
                            .map(|&span| bind.view(span))
                            .collect(),
                    })
                })
                .collect::<Result<_, Refusal>>()?,
        };
        let Some(round) = op.defer_bf16_round(&buffers, View::EMPTY, Counts::rows(rows))? else {
            return Ok(None);
        };
        if round.width != self.width
            || round.output != buffers.output
            || Some(round.live_rows) != buffers.live_rows
        {
            return Err(Refusal("deferred round differs from the reduction output"));
        }
        round
            .input
            .covers(u64::from(rows) * u64::from(self.width) * 4)?;
        let sum = bind
            .0
            .get(round.input.handle)
            .map_err(|_| Refusal("deferred round source is not bound"))?;
        Ok(Some((
            round
                .launches
                .into_iter()
                .map(|l| bind.lowered(l))
                .collect(),
            sum,
        )))
    }

    /// The launches of the embedding's widening of `rows` rows.
    pub(crate) fn widen(&self, rows: u32, views: &WidenViews) -> Result<Vec<Lowered>, Refusal> {
        let mut bind = Bind::default();
        let buffers = WidenRowsBuffers {
            input: bind.view(views.input),
            output: bind.view(views.output),
            live_rows: bind.view(views.live_rows),
        };
        bind.launch(&self.widen, &buffers, Counts::rows(rows))
    }
}

#[cfg(test)]
mod selection_tests {
    use super::*;
    use bundle_runtime::geometry::Ranks;
    use model_deepseek41_contract::geometry::{FLASH, GEOMETRIES};

    #[test]
    fn partials_and_gather_keep_their_geometry_gates_at_both_storages() {
        for geometry in GEOMETRIES {
            for ranks in [Ranks::TP1, Ranks::TP2, Ranks::TP4] {
                let Ok(shape) = Shape::new(geometry, ranks) else {
                    continue;
                };
                assert_eq!(requested(&shape), geometry == &FLASH);
                for storage in [KvStorage::Native, KvStorage::Bf16] {
                    assert_eq!(
                        bf16_partials_requested(&shape, storage),
                        geometry == &FLASH && ranks == Ranks::TP4,
                    );
                    assert!(matches!(
                        query(&shape, storage),
                        CommQuery::AllReduceRounded { .. }
                    ));
                }
            }
        }
    }

    #[test]
    fn dma_requires_negotiated_skip_and_a_prepared_gather() {
        use model_deepseek41_contract::prefill_profile::Profile;
        for (chunk, skip, gather) in [
            (0, false, true),
            (8192, false, true),
            (8192, true, false),
            (8192, true, true),
        ] {
            let policy = dma_requested().admit(Profile { chunk, skip }, gather);
            assert_eq!(policy.enabled(), chunk > 0 && skip && gather);
        }
    }
}
