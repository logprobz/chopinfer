//! The row ops: the RMSNorms of the sublayer inputs, the query-down output,
//! the window key and the final row (F32 arithmetic, the weight applied
//! before the one BF16 rounding, eps 1e-20, `model.py:281-293`), each
//! rank's embedding gather from its vocabulary shard, and the embedding
//! copied into every residual stream.

use kernel_core::backend::{Backend, Counts, Prepared, Refused, View};
use kernel_core::contract::Refusal;
use kernel_core::rmsnorm::RmsNormBuffers;
use kernel_core::row_copy::RowCopyBuffers;
use kernel_core::rows::EmbeddingBuffers;
use model_deepseek41_contract::rows;
use model_deepseek41_contract::serving::{DeepSeek, PREFILL_TOKENS};
use runtime_bundle_abi::DeviceBufferSpanV1;

use super::{Asks, Bind, Lowered, exact};
use crate::shape::Shape;

/// The most rows a decode or verify launch carries: the widest verify
/// bucket, which holds every decode bucket.
pub(crate) fn verify_rows() -> u32 {
    bundle_runtime::graphs::verify_tokens::<DeepSeek>(bundle_runtime::geometry::Items::package::<
        DeepSeek,
    >())
}

/// The row ops of an instance, each prepared for a prefill's rows, the
/// most any phase runs.
pub(crate) struct RowOps {
    /// The RMSNorm of each width the program normalizes.
    norms: Vec<(u32, Prepared<RmsNormBuffers<View>>)>,
    /// Each rank's gather from its vocabulary shard, in rank order.
    embeddings: Vec<Prepared<EmbeddingBuffers<View>>>,
    /// The vocabulary rows of one rank's shard.
    shard: u32,
    expand_streams: Prepared<RowCopyBuffers<View>>,
}

/// The views of an RMSNorm: `rows` of `x` normalized by `weight` into `y`,
/// of which the device count `live_rows` are live.
pub(crate) struct NormViews {
    pub x: DeviceBufferSpanV1,
    pub weight: DeviceBufferSpanV1,
    pub y: DeviceBufferSpanV1,
    pub live_rows: DeviceBufferSpanV1,
}

/// The views of an embedding gather: `output` row `r` is row
/// `ids[r] - rank * shard` of the rank's shard `table`, zero for ids of
/// other shards.
pub(crate) struct EmbeddingViews {
    pub table: DeviceBufferSpanV1,
    pub ids: DeviceBufferSpanV1,
    pub output: DeviceBufferSpanV1,
    pub live_rows: DeviceBufferSpanV1,
}

/// The views of the stream expansion: the device count `live_rows` of
/// `source`'s rows copied into every stream of `destination`.
pub(crate) struct StreamViews {
    pub source: DeviceBufferSpanV1,
    pub destination: DeviceBufferSpanV1,
    pub live_rows: DeviceBufferSpanV1,
}

impl RowOps {
    /// The norms of `shape`'s widths, each rank's embedding gather and the
    /// stream expansion.
    pub(crate) fn prepare(
        backend: &dyn Backend,
        shape: &Shape,
        asks: &Asks,
    ) -> Result<RowOps, Refused> {
        // Every phase normalizes, embeds and expands its rows.
        let need = exact::need(PREFILL_TOKENS, exact::every_phase(verify_rows()));
        let g = shape.geometry;
        let mut norms = Vec::new();
        for width in rows::norm_widths(g) {
            if norms.iter().any(|&(w, _)| w == width) {
                continue;
            }
            let requirement = rows::norm(g, width);
            let op = asks.ask(format!("rows.norm {width}"), need, |need| {
                backend.rmsnorm(&requirement, need)
            })?;
            norms.push((width, op));
        }
        let shard = shape.vocab_rows();
        let query = rows::embedding(g, shape.ranks());
        let embeddings = (0..shape.ranks())
            .map(|rank| {
                asks.ask(format!("rows.embedding rank {rank}"), need, |need| {
                    backend.embedding_gather(&query, rank * shard, need)
                })
            })
            .collect::<Result<_, _>>()?;
        Ok(RowOps {
            norms,
            embeddings,
            shard,
            expand_streams: asks.ask("rows.expand_streams", need, |need| {
                backend.broadcast_streams(&rows::expand_streams(g), need)
            })?,
        })
    }

    /// The RMSNorm of `rows` rows of `width`, gridded over the row bucket
    /// `bucket` that holds them.
    pub(crate) fn norm(
        &self,
        width: u32,
        rows: u32,
        bucket: u32,
        views: &NormViews,
    ) -> Result<Vec<Lowered>, Refusal> {
        let (_, op) = self
            .norms
            .iter()
            .find(|&&(w, _)| w == width)
            .ok_or(Refusal("DeepSeek V4.1 normalizes no rows of this width"))?;
        let mut bind = Bind::default();
        let buffers = RmsNormBuffers {
            status: None,
            status_row_offset: 0,
            x: bind.view(views.x),
            weight: bind.view(views.weight),
            residual: None,
            y: bind.view(views.y),
            live_rows: bind.view(views.live_rows),
        };
        let counts = Counts {
            padded_rows: bucket,
            ..Counts::rows(rows)
        };
        bind.launch(op, &buffers, counts)
    }

    /// Rank `rank`'s embedding of `rows` token ids from its vocabulary
    /// shard.
    pub(crate) fn embedding(
        &self,
        rank: u32,
        rows: u32,
        views: &EmbeddingViews,
    ) -> Result<Vec<Lowered>, Refusal> {
        let op = self
            .embeddings
            .get(rank as usize)
            .ok_or(Refusal("DeepSeek V4.1 has no embedding for this rank"))?;
        let mut bind = Bind::default();
        let buffers = EmbeddingBuffers {
            table: bind.view(views.table),
            ids: bind.view(views.ids),
            output: bind.view(views.output),
            live_rows: bind.view(views.live_rows),
            table_rows: self.shard,
        };
        bind.launch(op, &buffers, Counts::rows(rows))
    }

    /// The live rows of `views` copied into every residual stream, over the
    /// row bucket `bucket` that holds them.
    pub(crate) fn expand_streams(
        &self,
        bucket: u32,
        views: &StreamViews,
    ) -> Result<Vec<Lowered>, Refusal> {
        let mut bind = Bind::default();
        let buffers = RowCopyBuffers {
            source: bind.view(views.source),
            destination: bind.view(views.destination),
            live_rows: bind.view(views.live_rows),
        };
        bind.launch(&self.expand_streams, &buffers, Counts::rows(bucket))
    }
}
