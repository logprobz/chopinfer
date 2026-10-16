//! Lowering of the row stages through the instance's row ops
//! (`crate::ops`): the RMSNorms of the sublayer inputs, the query-down
//! output, the latent and the final row, and the embedding gather of the
//! rank's vocabulary shard.

use super::{Lowering, layer_view};
use crate::buffers::Buffer;
use crate::ops::{EmbeddingViews, Lowered, NormViews};
use crate::program::{Phase, Stage, Sublayer};

/// The one launch of a row op.
fn one<T>(
    launches: Result<Vec<T>, kernel_core::contract::Refusal>,
    what: impl Fn() -> String,
) -> Result<T, String> {
    let mut launches = launches.map_err(|refusal| format!("{}: {refusal}", what()))?;
    match (launches.pop(), launches.is_empty()) {
        (Some(launch), true) => Ok(launch),
        _ => Err(format!("{} launches other than once", what())),
    }
}

/// The input norm weight of a sublayer, below its layer's views.
pub(super) fn input_norm_weight(sublayer: Sublayer) -> &'static str {
    match sublayer {
        Sublayer::Attention => "input_layernorm.weight",
        Sublayer::Mlp => "post_attention_layernorm.weight",
    }
}

impl Lowering<'_> {
    /// The row capacity bucket a phase's row launches run at
    /// (`Target::row_bucket`).
    pub(super) fn row_bucket(&self, phase: Phase) -> Result<u32, String> {
        self.ops
            .target
            .row_bucket(phase.rows())
            .ok_or_else(|| format!("no row bucket holds {} rows", phase.rows()))
    }

    /// `stage`'s RMSNorm: the collapsed row into the sublayer input or the
    /// head's row, the query-down output into the query the up projection
    /// reads, the raw latent into the latent the KV write stores. It grids
    /// the phase's row bucket and records the phase's rows.
    pub(crate) fn norm(&self, phase: Phase, stage: Stage) -> Result<Lowered, String> {
        let (width, views) = self.norm_views(stage)?;
        let launches = self
            .ops
            .rows
            .norm(width, phase.rows(), self.row_bucket(phase)?, &views);
        one(launches, || format!("{stage:?} RMSNorm of {width}"))
    }

    /// The width of `stage`'s RMSNorm and its views.
    fn norm_views(&self, stage: Stage) -> Result<(u32, NormViews), String> {
        let p = self.shape.profile;
        let drafter = self.shape.geometry.layers();
        let (width, input, weight, output) = match stage {
            Stage::InputNorm(layer, sublayer) => {
                let name = input_norm_weight(sublayer);
                // The drafter's layer reads its single stream.
                let input = if layer >= drafter {
                    Buffer::DraftHidden
                } else {
                    Buffer::Collapsed
                };
                (p.hidden, input, layer_view(layer, name), Buffer::Normed)
            }
            // The drafter's two norms write rows the input copy
            // (`DraftEhInput`) moves side by side before anything else
            // writes them: the sublayer input the drafter's first norm
            // writes next, and the collapsed row the embedding's reduce
            // has read.
            Stage::DraftEmbedNorm => (
                p.hidden,
                Buffer::Reduced,
                layer_view(drafter, "enorm.weight"),
                Buffer::Normed,
            ),
            Stage::DraftHiddenNorm => (
                p.hidden,
                Buffer::DraftSource,
                layer_view(drafter, "hnorm.weight"),
                Buffer::Collapsed,
            ),
            Stage::DraftHeadNorm => (
                p.hidden,
                Buffer::DraftHidden,
                layer_view(drafter, "shared_head.norm.weight"),
                Buffer::Normed,
            ),
            Stage::FinalNorm => (
                p.hidden,
                Buffer::Collapsed,
                "model.language_model.norm.weight".into(),
                Buffer::Normed,
            ),
            Stage::QueryNorm(layer) => (
                p.q_rank,
                Buffer::QueryDown,
                layer_view(layer, "self_attn.q_a_layernorm.weight"),
                Buffer::QueryNormed,
            ),
            Stage::LatentNorm(layer) => (
                p.kv_rank,
                Buffer::LatentRaw,
                layer_view(layer, "self_attn.kv_a_layernorm.weight"),
                Buffer::Latent,
            ),
            _ => return Err(format!("{stage:?} is not a norm")),
        };
        let views = NormViews {
            x: self.scratch(input),
            weight: self.buffers.view(&weight)?,
            y: self.scratch(output),
            live_rows: self.inputs.live_tokens,
        };
        Ok((width, views))
    }

    /// Each row's embedding from the rank's vocabulary shard, zero for ids
    /// of other shards: the rank's partial of the embedding the all-reduce
    /// assembles, in the collapsed row. It grids the phase's row bucket and
    /// records the phase's rows.
    pub(crate) fn embedding(&self, phase: Phase) -> Result<Lowered, String> {
        let launches = self
            .ops
            .rows
            .embedding(self.rank, phase.rows(), &self.embedding_views()?);
        one(launches, || format!("rank {}'s embedding", self.rank))
    }

    fn embedding_views(&self) -> Result<EmbeddingViews, String> {
        Ok(EmbeddingViews {
            table: self
                .buffers
                .view("model.language_model.embed_tokens.weight")?,
            ids: self.inputs.token_ids,
            output: self.scratch(Buffer::Collapsed),
            live_rows: self.inputs.live_tokens,
        })
    }
}
