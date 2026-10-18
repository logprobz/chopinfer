//! The all-reduces: each sublayer's F32 rank partials summed in F32 in rank
//! order and rounded once to BF16 (`CommQuery::AllReduceRounded`, the
//! vendor's F32 all-reduce before its BF16 cast in `RowParallelLinear`,
//! `model.py:271-278`), and the embedding's BF16 shard rows widened into F32
//! partials (an own op) and reduced the same way, which gives back each
//! row's bits as one rank holds it.
//!
//! Each rank's op keeps comm state regions (`OpInfo::state`) that its
//! peers write; every rank allocates the same regions, zeroed at create.

use kernel_core::backend::{Backend, Counts, Prepared, Refused, View};
use kernel_core::collective::{AllReduceBuffers, RankViews};
use kernel_core::contract::{CommQuery, Refusal};
use model_deepseek41_contract::own_ops::OwnOps;
use model_deepseek41_contract::own_ops::widen::{WidenRows, WidenRowsBuffers};
use model_deepseek41_contract::serving::PREFILL_TOKENS;
use runtime_bundle_abi::DeviceBufferSpanV1;

use super::{Asks, Bind, Lowered, exact};
use crate::shape::Shape;

/// The all-reduces of an instance, each prepared for a prefill's rows.
pub(crate) struct ReduceOps {
    /// Each rank's rounded all-reduce of hidden rows, in rank order.
    reduces: Vec<Prepared<AllReduceBuffers<View>>>,
    /// The embedding's widening into F32 partials.
    widen: Prepared<WidenRowsBuffers<View>>,
    /// The bytes of each comm state region every rank keeps, in the ops'
    /// role order.
    state: Vec<u64>,
}

/// One hidden row of the all-reduce: every rank's F32 partial summed in F32
/// in rank order from zero and rounded once to BF16.
pub(crate) fn query(shape: &Shape) -> CommQuery {
    CommQuery::AllReduceRounded {
        ranks: shape.ranks(),
        elements: u64::from(shape.dims().hidden),
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
        asks: &Asks,
    ) -> Result<ReduceOps, Refused> {
        // Every phase reduces its sublayers.
        let need = exact::need(PREFILL_TOKENS, exact::every_phase(super::verify_rows()));
        let query = query(shape);
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
            reduces,
            widen,
            state,
        })
    }

    /// The bytes of each comm state region every rank allocates.
    pub(crate) fn state(&self) -> &[u64] {
        &self.state
    }

    /// The launches of rank `rank`'s all-reduce of `rows` rows.
    pub(crate) fn reduce(
        &self,
        rank: u32,
        rows: u32,
        views: &ReduceViews,
    ) -> Result<Vec<Lowered>, Refusal> {
        let op = self
            .reduces
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
                .map(|peer| RankViews {
                    output: bind.view(peer.output),
                    state: peer.state.iter().map(|&span| bind.view(span)).collect(),
                })
                .collect(),
        };
        bind.launch(op, &buffers, Counts::rows(rows))
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
