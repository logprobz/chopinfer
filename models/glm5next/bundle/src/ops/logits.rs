//! The ops around the head, on the shared block
//! (`bundle_runtime::logits::shards`): the gather by index of the rows the
//! head scores (a prefill's rows that need logits, the drafter's draft
//! rows), and the head's F32 logits shards exchanged to the rank that owns
//! each row (decode and verify) or gathered to rank 0 (prefill).

use bundle_runtime::logits::shards::{Asker, Bound, Needs, ShardedHead};
use kernel_core::backend::{Need, OpInfo, Refused};
use kernel_core::contract::Refusal;
use kernel_core::row_copy::GatherByIndexBuffers;
use model_glm5next_contract::contract::{PREFILL_TOKENS, SCORED_ROWS, Shape, head_rows};
use runtime_bundle_abi::DeviceBufferSpanV1;

use super::{Asks, Bind, Lowered, exact};

pub(crate) use bundle_runtime::logits::shards::PeerSpans as PeerViews;

/// The head's ops of an instance.
pub(crate) struct LogitsOps(ShardedHead);

impl Asker for Asks {
    fn need(&self, need: Need) -> Need {
        Asks::asked_need(need)
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
    pub(crate) fn prepare(
        backend: &dyn kernel_core::backend::Backend,
        shape: Shape,
        asks: &Asks,
    ) -> Result<LogitsOps, Refused> {
        let needs = Needs {
            head_rows: exact::need(PREFILL_TOKENS, exact::PREFILL),
            exchange: exact::need(head_rows(), exact::verify(head_rows())),
            gather: exact::need(SCORED_ROWS, exact::PREFILL),
        };
        ShardedHead::prepare(
            backend,
            &model_glm5next_contract::copies::head_rows(shape),
            shape.ranks(),
            u64::from(shape.profile.vocab_rows),
            needs,
            asks,
        )
        .map(LogitsOps)
    }

    /// Gathers at most `rows` head rows by index; with `grid` nonzero, over
    /// the row bucket holding `grid` rows (a prefill's gather of its scored
    /// rows).
    pub(crate) fn head_rows(
        &self,
        rows: u32,
        grid: u32,
        spans: GatherByIndexBuffers<DeviceBufferSpanV1>,
    ) -> Result<Vec<Lowered>, Refusal> {
        let mut bind = Bind::default();
        let bound = self
            .0
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
            .0
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
            .0
            .gather(rank, rows, local, peers, &mut |span| bind.view(span))?;
        launch(bind, bound)
    }
}
