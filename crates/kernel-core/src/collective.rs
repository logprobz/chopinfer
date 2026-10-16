//! Contracts and buffer roles of collectives, whose launches wait for their
//! peer ranks.
//!
//! Every collective takes the views of every rank in rank order
//! ([`RankViews`]): each rank's output and its persistent state regions,
//! which the op lists in its info (`OpInfo::state`). A rank reads its own
//! state from its entry. Rows are dense: row `r` of a view of `n`-element
//! rows starts at element `r * n`.
//!
//! The all-reduce sums `CommQuery::AllReduce` rows over ranks, the fused
//! all-reduce also adds a residual and normalizes the rows, the
//! hyper-connection all-reduce injects the sum into residual streams and
//! normalizes them (`SigmoidMeanHyperConnectionQuery::AllReduceInjectNorm`,
//! oracle `hyper_connection_oracle::all_reduce_inject_norm`), and the
//! all-gather (`CommQuery::AllGather`) concatenates every rank's elements
//! rank-major. The row moves ([`ShardedRows`]) assemble whole rows from
//! their column shards without arithmetic.

use std::ops::Range;

use crate::contract::{DType, Refusal, Validate};
use crate::rmsnorm::FusedResidual;

crate::scratch_record! {
    /// An all-reduce's buffers on one rank: its `input` rows, the `output`
    /// rows it writes, the optional U32 device count `live_rows` of the
    /// rows that need their sums, and every rank's views in rank order,
    /// this rank's included. The op lists each rank's state regions in its
    /// info (`OpInfo::state`).
    ///
    /// Every rank passes the same live count, at most the launch's rows.
    /// Output rows below it hold their sums; output rows at or past it are
    /// unspecified after the call: an entry may leave them as they were or
    /// write them. Without a count every row of the launch holds its sum.
    /// An op whose launches read the count refuses them without one.
    #[derive(Clone, Debug, Eq, PartialEq)]
    pub struct AllReduceBuffers {
        pub input: value,
        pub output: value,
        pub live_rows: option,
        pub ranks: list(RankViews),
    }
}

crate::scratch_record! {
    /// One rank's views that a collective reads or writes from any rank:
    /// its output rows and its persistent state regions.
    #[derive(Clone, Debug, Eq, PartialEq)]
    pub struct RankViews {
        pub output: value,
        pub state: values,
    }
}

crate::scratch_record! {
    /// An all-gather's or a row exchange's buffers on one rank: its `input`
    /// and every rank's views in rank order, this rank's included. The
    /// receiving ranks' outputs are written.
    #[derive(Clone, Debug, Eq, PartialEq)]
    pub struct GatherBuffers {
        pub input: value,
        pub ranks: list(RankViews),
    }
}

crate::scratch_record! {
    /// A row gather's buffers on one rank: its `input` shards, the optional
    /// U32 `indices` of the rows to move (rows `0..rows` without it), and
    /// every rank's views in rank order.
    #[derive(Clone, Debug, Eq, PartialEq)]
    pub struct RowGatherBuffers {
        pub input: value,
        pub indices: option,
        pub ranks: list(RankViews),
    }
}

crate::scratch_record! {
    /// A fused all-reduce's buffers on one rank (`CommQuery::AllReduce` with
    /// `fused_residual` and `fused_rmsnorm`): its `input` partials (as its
    /// [`ReduceInput`] lays them out), the fused `residual`: the BF16 rows
    /// the sum is added to and the rows that keep the rounded pre-norm sum
    /// for the next layer (the output may alias the input, never `output`),
    /// the norm `weight` (one row), the normalized `output` rows, the
    /// device count `live_rows` of a routed input, and every rank's views
    /// in rank order, whose outputs are the ranks' normalized rows.
    #[derive(Clone, Debug, Eq, PartialEq)]
    pub struct FusedAllReduceBuffers {
        pub input: value,
        pub residual: record(FusedResidual),
        pub weight: value,
        pub output: value,
        pub live_rows: option,
        pub ranks: list(RankViews),
    }
}

/// How a fused all-reduce's input holds each rank's partial rows.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReduceInput {
    /// The partial rows themselves, one per row.
    Rows,
    /// Each row's `routes` preweighted routed rows (`[rows][routes][width]`),
    /// summed in route order and rounded to BF16 as the MoE row combine
    /// does before the sum over ranks. Rows at or past the device live
    /// count contribute zero.
    Routed { routes: u32 },
}

crate::scratch_record! {
    /// One rank's persistent state regions that a collective reads or
    /// writes from any rank.
    #[derive(Clone, Debug, Eq, PartialEq)]
    pub struct RankState {
        pub state: values,
    }
}

crate::scratch_record! {
    /// A hyper-connection all-reduce's buffers on one rank
    /// (`SigmoidMeanHyperConnectionQuery::AllReduceInjectNorm`): its
    /// `input` partial rows of `width`, sums over ranks of which are
    /// injected into every stream; the fused `residual`: the BF16 streams
    /// (`[rows][streams][width]`) the sums are injected into and the
    /// updated streams it writes; the BF16 inject `logits`
    /// (`[rows][streams]`); the grouped norm `weight` (`[streams][width]`);
    /// the `normed` rows of the streams' layout; the device count
    /// `live_rows`; and every rank's state in rank order. Rows at or past
    /// the live count are not written.
    #[derive(Clone, Debug, Eq, PartialEq)]
    pub struct HcAllReduceBuffers {
        pub input: value,
        pub residual: record(FusedResidual),
        pub logits: value,
        pub weight: value,
        pub normed: value,
        pub live_rows: value,
        pub ranks: list(RankState),
    }
}

/// Rows whose columns are split across ranks in rank order: rank `q` holds
/// columns `q * shard..(q + 1) * shard` of every row as dense rows of
/// `shard` elements of `dtype`, and a whole row is `ranks * shard`
/// elements. Logits rows of a vocabulary-sharded head are the common case.
///
/// A row move writes rank `q`'s shard of row `r` at element
/// `(r * ranks + q) * shard` of a receiving rank's output, so whole rows
/// land at their row offsets; it moves bits and computes nothing.
/// - A row gather (`Backend::gather_rows`) moves rows to one target rank:
///   rows `0..rows`, or the rows `indices` names, each at its own offset.
/// - A row exchange (`Backend::exchange_rows`) moves each row to the rank
///   that owns it ([`owned_rows`]).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ShardedRows {
    pub ranks: u32,
    /// Elements of one rank's shard of a row.
    pub shard: u64,
    pub dtype: DType,
}

impl Validate for ShardedRows {
    fn validate(&self) -> Result<(), Refusal> {
        if self.ranks == 0 || self.shard == 0 {
            return Err(Refusal("sharded rows need ranks and a shard"));
        }
        Ok(())
    }
}

/// The rows rank `rank` owns among `rows` in a row exchange: contiguous
/// blocks of `rows.div_ceil(ranks)` in rank order, so low ranks own full
/// blocks and trailing ranks may own none.
pub const fn owned_rows(rows: u32, ranks: u32, rank: u32) -> Range<u32> {
    let per = rows.div_ceil(if ranks == 0 { 1 } else { ranks });
    let first = if rank * per < rows { rank * per } else { rows };
    let last = if first + per < rows {
        first + per
    } else {
        rows
    };
    first..last
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn owned_rows_split_into_rank_ordered_blocks() {
        assert_eq!(owned_rows(10, 4, 0), 0..3);
        assert_eq!(owned_rows(10, 4, 3), 9..10);
        assert_eq!(owned_rows(2, 4, 2), 2..2);
        for (rows, ranks) in [(1, 2), (7, 2), (33, 8), (64, 4)] {
            let mut next = 0;
            for rank in 0..ranks {
                let owned = owned_rows(rows, ranks, rank);
                assert_eq!(owned.start, next);
                next = owned.end;
            }
            assert_eq!(next, rows);
        }
    }

    #[test]
    fn sharded_rows_need_ranks_and_a_shard() {
        let rows = ShardedRows {
            ranks: 16,
            shard: 4096,
            dtype: DType::F32,
        };
        assert_eq!(rows.validate(), Ok(()));
        assert!(ShardedRows { shard: 0, ..rows }.validate().is_err());
        assert!(ShardedRows { ranks: 0, ..rows }.validate().is_err());
    }
}
