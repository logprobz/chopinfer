//! The row ops: the RMSNorms of the sublayer inputs, the query-down output,
//! the latent and the final and drafter rows, the embedding gather of each
//! rank's vocabulary shard, and the shared row copies: the embedding into
//! every residual stream and the router's scores of every expert
//! (`model_glm5next_contract::copies`).

use kernel_core::backend::{Backend, Counts, Prepared, Refused, View};
use kernel_core::contract::{Refusal, RowQuery};
use kernel_core::rmsnorm::RmsNormBuffers;
use kernel_core::row_copy::RowCopyBuffers;
use kernel_core::rows::EmbeddingBuffers;
use model_glm5next_contract::contract::{PREFILL_TOKENS, Shape};
use model_glm5next_contract::copies;
use runtime_bundle_abi::DeviceBufferSpanV1;

use super::{Asks, Bind, Lowered, exact};

/// The row ops of an instance, each prepared for a prefill's rows, the
/// most any phase runs.
pub(crate) struct RowOps {
    /// The RMSNorm of each width the program normalizes.
    norms: Vec<(u32, Prepared<RmsNormBuffers<View>>)>,
    /// Each rank's gather from its vocabulary shard, in rank order.
    embeddings: Vec<Prepared<EmbeddingBuffers<View>>>,
    /// The vocabulary rows of one rank's shard.
    shard: u32,
    /// Each hidden row copied into every residual stream.
    expand_streams: Prepared<RowCopyBuffers<View>>,
    /// The router's scores of every expert from its GEMM's padded rows.
    router_scores: Prepared<RowCopyBuffers<View>>,
}

/// The views of a row copy: the device count `live_rows` of `source`'s rows
/// copied into `destination`.
pub(crate) struct CopyViews {
    pub source: DeviceBufferSpanV1,
    pub destination: DeviceBufferSpanV1,
    pub live_rows: DeviceBufferSpanV1,
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

impl RowOps {
    /// The norms of `shape`'s widths at their requirements, each rank's
    /// embedding gather, and the two row copies.
    pub(crate) fn prepare(
        backend: &dyn Backend,
        shape: Shape,
        asks: &Asks,
    ) -> Result<RowOps, Refused> {
        // Every phase normalizes, embeds, expands and compacts its rows.
        let need = exact::need(PREFILL_TOKENS, exact::every_phase(exact::verify_rows()));
        let mut norms = Vec::new();
        for width in shape.rmsnorm_widths() {
            if norms.iter().any(|&(w, _)| w == width) {
                continue;
            }
            let requirement = shape.rmsnorm_requirement(width);
            let norm = asks.ask(format!("rows.norm {width}"), need, |need| {
                backend.rmsnorm(&requirement, need)
            })?;
            norms.push((width, norm));
        }
        let p = shape.profile;
        let query = RowQuery::EmbeddingGather {
            vocab: p.vocab_rows,
            width: p.hidden,
        };
        let embeddings = (0..shape.ranks())
            .map(|rank| {
                asks.ask(format!("rows.embedding rank {rank}"), need, |need| {
                    backend.embedding_gather(&query, rank * p.vocab_rows, need)
                })
            })
            .collect::<Result<_, _>>()?;
        Ok(RowOps {
            norms,
            embeddings,
            shard: p.vocab_rows,
            expand_streams: asks.ask("rows.expand_streams", need, |need| {
                backend.broadcast_streams(&copies::expand_streams(shape), need)
            })?,
            router_scores: asks.ask("rows.router_scores", need, |need| {
                backend.column_slice(&copies::router_scores(shape), need)
            })?,
        })
    }

    /// The live rows of `views` copied into every residual stream, over the
    /// row bucket `bucket` that holds them.
    pub(crate) fn expand_streams(
        &self,
        bucket: u32,
        views: &CopyViews,
    ) -> Result<Vec<Lowered>, Refusal> {
        copy(&self.expand_streams, bucket, views)
    }

    /// The router's scores of every expert of the live rows of `views`,
    /// over the row bucket `bucket` that holds them.
    pub(crate) fn router_scores(
        &self,
        bucket: u32,
        views: &CopyViews,
    ) -> Result<Vec<Lowered>, Refusal> {
        copy(&self.router_scores, bucket, views)
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
            .ok_or(Refusal("GLM 5.3 Flash normalizes no rows of this width"))?;
        let mut bind = Bind::default();
        let buffers = RmsNormBuffers {
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
    /// shard. It grids the row bucket that holds them and records the rows.
    pub(crate) fn embedding(
        &self,
        rank: u32,
        rows: u32,
        views: &EmbeddingViews,
    ) -> Result<Vec<Lowered>, Refusal> {
        let op = self
            .embeddings
            .get(rank as usize)
            .ok_or(Refusal("GLM 5.3 Flash has no embedding for this rank"))?;
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
}

/// A row copy's launches over the `bucket` rows of `views`: one CTA per
/// row, the rows past the device live count returning.
fn copy(
    op: &Prepared<RowCopyBuffers<View>>,
    bucket: u32,
    views: &CopyViews,
) -> Result<Vec<Lowered>, Refusal> {
    let mut bind = Bind::default();
    let buffers = RowCopyBuffers {
        source: bind.view(views.source),
        destination: bind.view(views.destination),
        live_rows: bind.view(views.live_rows),
    };
    bind.launch(op, &buffers, Counts::rows(bucket))
}
