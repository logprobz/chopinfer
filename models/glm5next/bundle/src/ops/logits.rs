//! The logits moves between ranks: the head's F32 logits shards exchanged
//! to the rank that owns each row (decode and verify) or gathered to rank 0
//! (prefill).

use kernel_core::backend::{Backend, Counts, Prepared, Refused, View};
use kernel_core::collective::{GatherBuffers, RankViews, RowGatherBuffers, ShardedRows};
use kernel_core::contract::{DType, Refusal};
use model_glm5next_contract::contract::{SCORED_ROWS, Shape, head_rows};
use runtime_bundle_abi::DeviceBufferSpanV1;

use super::{Asks, Bind, Lowered, exact};

/// Each rank's logits moves, in rank order.
pub(crate) struct LogitsOps {
    /// The exchange of a decode or verify phase's rows, at most
    /// `head_rows()`.
    exchanges: Vec<Prepared<GatherBuffers<View>>>,
    /// The gather of a prefill's logits rows to rank 0, at most
    /// `SCORED_ROWS`.
    gathers: Vec<Prepared<RowGatherBuffers<View>>>,
}

/// One rank's views that a logits move reaches on every rank: the full
/// rows it receives, and its flags and epoch.
#[derive(Clone, Copy)]
pub(crate) struct PeerViews {
    pub output: DeviceBufferSpanV1,
    pub flags: DeviceBufferSpanV1,
    pub epoch: DeviceBufferSpanV1,
}

impl LogitsOps {
    /// Every rank's moves of `shape`'s vocabulary shards.
    pub(crate) fn prepare(
        backend: &dyn Backend,
        shape: Shape,
        asks: &Asks,
    ) -> Result<LogitsOps, Refused> {
        let ranks = shape.ranks();
        let rows = ShardedRows {
            ranks,
            shard: u64::from(shape.profile.vocab_rows),
            dtype: DType::F32,
        };
        let mut ops = LogitsOps {
            exchanges: Vec::new(),
            gathers: Vec::new(),
        };
        for rank in 0..ranks {
            let exchange = asks.ask(
                format!("logits.exchange rank {rank}"),
                exact::need(head_rows(), exact::verify(head_rows())),
                |need| backend.exchange_rows(&rows, rank, need),
            )?;
            ops.exchanges.push(exchange);
            let gather = asks.ask(
                format!("logits.gather rank {rank}"),
                exact::need(SCORED_ROWS, exact::PREFILL),
                |need| backend.gather_rows(&rows, rank, 0, need),
            )?;
            ops.gathers.push(gather);
        }
        Ok(ops)
    }

    /// Rank `rank`'s exchange of `rows` rows of its `local` shards to their
    /// owners' outputs.
    pub(crate) fn exchange(
        &self,
        rank: u32,
        rows: u32,
        local: DeviceBufferSpanV1,
        peers: &[PeerViews],
    ) -> Result<Vec<Lowered>, Refusal> {
        let op = of(&self.exchanges, rank)?;
        let mut bind = Bind::default();
        let input = bind.view(local);
        let ranks = peer_views(&mut bind, rank, peers)?;
        bind.launch(op, &GatherBuffers { input, ranks }, Counts::rows(rows))
    }

    /// Rank `rank`'s gather of `rows` rows of its `local` shards to rank
    /// 0's output.
    pub(crate) fn gather(
        &self,
        rank: u32,
        rows: u32,
        local: DeviceBufferSpanV1,
        peers: &[PeerViews],
    ) -> Result<Vec<Lowered>, Refusal> {
        let op = of(&self.gathers, rank)?;
        let mut bind = Bind::default();
        let input = bind.view(local);
        let ranks = peer_views(&mut bind, rank, peers)?;
        let buffers = RowGatherBuffers {
            input,
            indices: None,
            ranks,
        };
        bind.launch(op, &buffers, Counts::rows(rows))
    }
}

fn of<T>(ops: &[T], rank: u32) -> Result<&T, Refusal> {
    ops.get(rank as usize)
        .ok_or(Refusal("GLM 5.3 Flash has no logits move for this rank"))
}

/// Every rank's views, bound in the order the moves pass them: the outputs,
/// the flags, then `rank`'s epoch (the one its launch takes) before the
/// others'.
fn peer_views(
    bind: &mut Bind,
    rank: u32,
    peers: &[PeerViews],
) -> Result<Vec<RankViews<View>>, Refusal> {
    let own = peers.get(rank as usize).ok_or(Refusal(
        "GLM 5.3 Flash's logits move lacks this rank's views",
    ))?;
    let outputs: Vec<View> = peers.iter().map(|peer| bind.view(peer.output)).collect();
    let flags: Vec<View> = peers.iter().map(|peer| bind.view(peer.flags)).collect();
    let own_epoch = bind.view(own.epoch);
    Ok(peers
        .iter()
        .enumerate()
        .map(|(peer, views)| RankViews {
            output: outputs[peer],
            state: vec![
                flags[peer],
                match peer == rank as usize {
                    true => own_epoch,
                    false => bind.view(views.epoch),
                },
            ],
        })
        .collect())
}
