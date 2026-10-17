//! One rank's lowering of a phase's stages into launches through the
//! prepared ops (`crate::ops`), over the step's inputs and the rank's
//! buffers.
//!
//! A stage whose family the instance lacks, or whose lowering is not
//! written yet, lowers to `None`: execute refuses such a step unless the
//! instance runs the lowered stages alone (the fake-host tests).

use bundle_runtime::upload::DeviceInputs;

use crate::buffers::{RankBuffers, rows};
use crate::ops::{EmbeddingViews, Lowered, NormViews, Ops, StreamViews};
use crate::program::{Phase, Stage, Sublayer};
use crate::shape::Shape;

/// What one rank lowers a phase with.
pub struct Lowering<'a> {
    pub shape: &'a Shape,
    pub ops: &'a Ops,
    pub buffers: &'a RankBuffers,
    pub inputs: &'a DeviceInputs,
    pub rank: u32,
}

/// The checkpoint name of layer `layer`'s tensor `name`.
fn layer_view(layer: u32, name: &str) -> String {
    format!("layers.{layer}.{name}")
}

impl Lowering<'_> {
    /// `stage`'s launches in `phase`, or `None` while it is not lowered.
    pub fn stage(&self, phase: Phase, stage: Stage) -> Result<Option<Vec<Lowered>>, String> {
        let error = |refusal: kernel_core::contract::Refusal| format!("{stage:?}: {refusal}");
        let Some(row_ops) = &self.ops.rows else {
            return Ok(None);
        };
        let scratch = &self.buffers.scratch;
        let rows_of = |span| rows(span, phase.rows());
        let live_rows = self.inputs.live_tokens;
        let norm = |weight: String| -> Result<Option<Vec<Lowered>>, String> {
            let views = NormViews {
                x: rows_of(scratch.collapsed),
                weight: self.buffers.view(&weight)?,
                y: rows_of(scratch.normed),
                live_rows,
            };
            row_ops
                .norm(self.shape.dims().hidden, phase.rows(), phase.rows(), &views)
                .map(Some)
                .map_err(error)
        };
        match stage {
            Stage::Embedding => {
                let views = EmbeddingViews {
                    table: self.buffers.view("embed.weight")?,
                    ids: self.inputs.token_ids,
                    output: rows_of(scratch.embedding),
                    live_rows,
                };
                row_ops
                    .embedding(self.rank, phase.rows(), &views)
                    .map(Some)
                    .map_err(error)
            }
            Stage::ExpandStreams => {
                let views = StreamViews {
                    source: rows_of(scratch.embedded),
                    destination: rows_of(scratch.streams),
                    live_rows,
                };
                row_ops
                    .expand_streams(phase.rows(), &views)
                    .map(Some)
                    .map_err(error)
            }
            Stage::InputNorm(layer, Sublayer::Attention) => {
                norm(layer_view(layer, "attn_norm.weight"))
            }
            Stage::InputNorm(layer, Sublayer::Mlp) => norm(layer_view(layer, "ffn_norm.weight")),
            Stage::FinalNorm => norm("norm.weight".into()),
            _ => Ok(None),
        }
    }
}
