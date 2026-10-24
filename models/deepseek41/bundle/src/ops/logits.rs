//! The ops around the head, on the shared block
//! (`bundle_runtime::logits::shards`): the gather by index of the prefill
//! rows that need logits, and the head's F32 logits shards exchanged to
//! the rank that owns each row (decode and verify) or gathered to rank 0
//! (prefill).

use bundle_runtime::logits::shards::{Asker, Bound, Needs, ShardedHead};
use kernel_core::backend::{Need, OpInfo, Refused};
use kernel_core::contract::Refusal;
use kernel_core::row_copy::GatherByIndexBuffers;
use kernel_core::row_exchange::{EXCHANGE_ROWS_EPOCH_BYTES, EXCHANGE_ROWS_FLAG_BYTES};
use model_deepseek41_contract::serving::{PREFILL_TOKENS, SCORED_ROWS};
use runtime_bundle_abi::DeviceBufferSpanV1;

use super::{Asks, Bind, Lowered, exact};
use crate::buffers::head_rows;
use crate::shape::Shape;

pub(crate) use bundle_runtime::logits::shards::PeerSpans as PeerViews;

/// The head's ops of an instance, and the bytes of the comm state their
/// moves write on every rank: the exchange's flags and epoch, then the
/// gather's.
pub(crate) struct LogitsOps {
    head: ShardedHead,
    state: [u64; 4],
}

impl Asker for Asks {
    fn need(&self, need: Need) -> Need {
        need
    }

    fn record(&self, what: String, need: Need, info: &OpInfo) {
        Asks::record(self, what, need, info);
    }
}

/// `bound`'s launches through a binder of its own.
fn launch<B>(bind: Bind, bound: Bound<'_, B>) -> Result<Vec<Lowered>, Refusal> {
    bind.launch(bound.op, &bound.buffers, bound.counts)
}

impl LogitsOps {
    /// The head's ops of `shape`'s vocabulary shards: the head rows'
    /// gather asked for a prefill's chunk split, each exchange one-row bits
    /// up to `head_rows()`, each gather to rank 0 a prefill's chunk split.
    /// `gather_state` is the bytes of the kernels' gather flags and epoch.
    pub(crate) fn prepare(
        backend: &dyn kernel_core::backend::Backend,
        gather_state: [u64; 2],
        shape: &Shape,
        asks: &Asks,
    ) -> Result<LogitsOps, Refused> {
        let needs = Needs {
            head_rows: exact::need(PREFILL_TOKENS, exact::PREFILL),
            exchange: exact::need(head_rows(shape), exact::verify(head_rows(shape))),
            gather: exact::need(SCORED_ROWS, exact::PREFILL),
        };
        let head = ShardedHead::prepare(
            backend,
            &model_deepseek41_contract::rows::head_rows(shape.geometry),
            shape.ranks(),
            u64::from(shape.vocab_rows()),
            needs,
            asks,
        )?;
        Ok(LogitsOps {
            head,
            state: [
                EXCHANGE_ROWS_FLAG_BYTES,
                EXCHANGE_ROWS_EPOCH_BYTES,
                gather_state[0],
                gather_state[1],
            ],
        })
    }

    /// The bytes of each comm state region every rank allocates, zeroed.
    pub(crate) fn state(&self) -> [u64; 4] {
        self.state
    }

    /// Gathers `rows` head rows by index, gridded over the prefill's
    /// `grid` rows.
    pub(crate) fn head_rows(
        &self,
        rows: u32,
        grid: u32,
        spans: GatherByIndexBuffers<DeviceBufferSpanV1>,
    ) -> Result<Vec<Lowered>, Refusal> {
        let mut bind = Bind::default();
        let bound = self
            .head
            .head_rows(rows, grid, spans, &mut |span| bind.view(span));
        launch(bind, bound)
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
        let mut bind = Bind::default();
        let bound = self
            .head
            .exchange(rank, rows, local, peers, &mut |span| bind.view(span))?;
        launch(bind, bound)
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
        let mut bind = Bind::default();
        let bound = self
            .head
            .gather(rank, rows, local, peers, &mut |span| bind.view(span))?;
        launch(bind, bound)
    }
}
