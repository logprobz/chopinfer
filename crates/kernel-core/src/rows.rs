//! Buffer roles of the row-local ops other than RMSNorm (`rmsnorm`): the
//! embedding gather, the logits argmax and top-k, the sharded argmax, the
//! sigmoid-mean hyper-connection rows, the n-gram hash gather and the PLE
//! gate. Their contracts are `RowQuery`, `SigmoidMeanHyperConnectionQuery`
//! and `NgramQuery` in `contract`.
//!
//! Every op reads `live_rows`, the device count of live rows: it writes no
//! row at or past that count, and each row's output depends on that row's
//! inputs alone, except the n-gram hash gather's, which also depends on
//! the sequence's earlier tokens (the chunk's earlier rows and its
//! `history`). Rows are dense: row `r` of a view of `n`-element rows
//! starts at element `r * n`. BF16 values are 2 bytes, F32 values, ids and
//! counts 4.

crate::scratch_record! {
    /// An embedding gather's buffers (`RowQuery::EmbeddingGather`): the
    /// BF16 `table` of `table_rows` rows of `width`, the I32 token `ids`,
    /// and the BF16 `output` rows. Row `r` of `output` is table row
    /// `ids[r] - row_offset` (the op's prepared table offset), or zeros when
    /// that row is outside the table, so the shards of a sharded table sum
    /// to the whole gather.
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct EmbeddingBuffers {
        pub table: value,
        pub ids: value,
        pub output: value,
        pub live_rows: value,
        pub table_rows: scalar(u32),
    }
}

crate::scratch_record! {
    /// A logits argmax or top-k's buffers (`RowQuery::LogitsArgmax`,
    /// `RowQuery::LogitsTopK`): F32 `logits` rows of the contract's `vocab`,
    /// of which the device count `live_width` are live (the rest pad), the
    /// I32 `ids` of each row's `k` largest live logits in descending order
    /// (ties to the lower id), and optionally their F32 log-probabilities
    /// over the live logits, `k` per row.
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct LogitsBuffers {
        pub logits: value,
        pub live_width: value,
        pub ids: value,
        pub log_probs: option,
        pub live_rows: value,
    }
}

crate::scratch_record! {
    /// A sharded argmax partial's buffers (`RowQuery::ShardedArgmaxPartial`)
    /// on one rank: the rank's `logits` rows of `vocab_per_rank`, of the
    /// query's dtype, of which the device count `live_width` (at least
    /// one) are live, and the `pairs` it writes, one per row: the largest
    /// live logit's F32 bits and its global vocabulary id.
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct ArgmaxPartialBuffers {
        pub logits: value,
        pub live_width: value,
        pub pairs: record(ArgmaxPairs),
        pub live_rows: value,
    }
}

crate::scratch_record! {
    /// A sharded argmax final's buffers (`RowQuery::ShardedArgmaxFinal`):
    /// every rank's `pairs`, laid out as the query states, and where each
    /// row's I32 global argmax goes: row `r` of `output` when present, and
    /// each of `scatter`. No output may overlap another output or an
    /// input; a backend refuses views it can tell share memory.
    #[derive(Clone, Debug, Eq, PartialEq)]
    pub struct ArgmaxFinalBuffers {
        pub pairs: record(ArgmaxPairs),
        pub output: option,
        pub scatter: list(ArgmaxScatter),
        pub live_rows: value,
    }
}

crate::scratch_record! {
    /// Argmax pairs: F32 `scores` and I32 `ids` at equal indices. With
    /// `peer` set the views are one rank's memory that every rank's launch
    /// reaches through a peer lease, which the host validates before the
    /// launch; otherwise they are the launching rank's own.
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct ArgmaxPairs {
        pub scores: value,
        pub ids: value,
        pub peer: scalar(bool),
    }
}

crate::scratch_record! {
    /// A scattered argmax output: row `r`'s I32 id goes to element
    /// `rows[r]` of `output`, I32 `rows` one per row. A row index outside
    /// `output` (negative, or at or past its elements) writes nothing.
    /// Precondition: the live rows' indexes inside `output` are distinct.
    /// Two live rows with one destination leave that element unspecified,
    /// so no exactness claim covers such a launch; the indexes are device
    /// data, which the host cannot check.
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct ArgmaxScatter {
        pub rows: value,
        pub output: value,
    }
}

crate::scratch_record! {
    /// A hyper-connection grouped norm's buffers
    /// (`SigmoidMeanHyperConnectionQuery::GroupedNorm`): BF16 `x` rows of
    /// `streams * width`, the per-stream `weight` (`streams * width` values
    /// of the prepared weight dtype, BF16 or F32) and the BF16 normalized
    /// `output` rows of `x`'s shape.
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct GroupedNormBuffers {
        pub x: value,
        pub weight: value,
        pub output: value,
        pub live_rows: value,
    }
}

crate::scratch_record! {
    /// A hyper-connection bottleneck's buffers
    /// (`SigmoidMeanHyperConnectionQuery::Bottleneck`): the BF16 `down`
    /// projection rows of `rank` and the BF16 `output` rows of `rank`.
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct BottleneckBuffers {
        pub down: value,
        pub output: value,
        pub live_rows: value,
    }
}

crate::scratch_record! {
    /// A hyper-connection mix's buffers
    /// (`SigmoidMeanHyperConnectionQuery::Mix`): the BF16 `up` projection
    /// and grouped-norm (`normed`) rows of `streams * width`, and the BF16
    /// sublayer input `output` rows of `width`.
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct MixBuffers {
        pub up: value,
        pub normed: value,
        pub output: value,
        pub live_rows: value,
    }
}

crate::scratch_record! {
    /// A hyper-connection inject's buffers
    /// (`SigmoidMeanHyperConnectionQuery::Inject`): the BF16 residual `x`
    /// rows of `streams * width`, the BF16 sublayer output `y` rows of
    /// `width`, the BF16 `inject` projection rows of `streams`, and the BF16
    /// updated residual `output` rows of `x`'s shape.
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct InjectBuffers {
        pub x: value,
        pub y: value,
        pub inject: value,
        pub output: value,
        pub live_rows: value,
    }
}

crate::scratch_record! {
    /// An n-gram hash gather's buffers on one rank
    /// (`NgramQuery::HashGather`) for one sequence chunk of `live_rows`
    /// rows: the U32 `tokens`, the sequence's U32 `history` before the
    /// chunk (its last `orders - 1` tokens, oldest first), the runtime hash
    /// tables (U64 `multipliers` per n-gram position, U64 `sizes` and
    /// `offsets` per head), the rank's E4M3 `table` rows of `head_width`,
    /// which hold global rows `[table_start, table_start + table_rows)`,
    /// and the BF16 dequantization `scale`. It writes the BF16 `output`
    /// rows of `local_heads * head_width`, the U64 global table row of each
    /// row's local heads (`row_ids`) and the sequence's history after the
    /// chunk (`history_out`, which may not alias `history`).
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct NgramBuffers {
        pub tokens: value,
        pub history: value,
        pub multipliers: value,
        pub sizes: value,
        pub offsets: value,
        pub table: value,
        pub scale: value,
        pub output: value,
        pub row_ids: value,
        pub history_out: value,
        pub live_rows: value,
        pub table_start: scalar(u64),
        pub table_rows: scalar(u64),
    }
}

crate::scratch_record! {
    /// A PLE gate's buffers (`NgramQuery::PleGate`): the BF16 grouped-norm
    /// `key` and `query` rows of `streams * width`, the BF16 `value` rows of
    /// `width`, and the BF16 gated `output` rows of `streams * width`.
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct PleGateBuffers {
        pub key: value,
        pub query: value,
        pub value: value,
        pub output: value,
        pub live_rows: value,
    }
}
