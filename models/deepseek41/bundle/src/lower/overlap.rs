//! Disjoint row views for the final attention projection and its reduction.
use super::{Lowering, layer_view};
use crate::buffers::row_range;
use crate::ops::{GemmViews, Lowered, PeerViews, ReduceViews};
use runtime_bundle_abi::DeviceBufferSpanV1;

impl Lowering<'_> {
    pub(crate) fn output_b_part(
        &self,
        layer: u32,
        first: u32,
        rows: u32,
        live: DeviceBufferSpanV1,
    ) -> Result<(Vec<Lowered>, Vec<Lowered>), String> {
        let dense = self.ops.dense.as_ref().ok_or("no dense projection")?;
        let reduce = self.ops.reduce.as_ref().ok_or("no rounded reduction")?;
        let scratch = &self.buffers.scratch;
        let range = |span| row_range(span, first, rows);
        // The original full-chunk quantizer runs before either projection.
        // Its row-local codes, scales and sticky status are then read only.
        let project = dense
            .project_independent(
                "attention.o-b",
                rows,
                &GemmViews {
                    // Packed fold storage serves decode only.
                    stream: None,
                    activations: range(scratch.output_a_codes)?,
                    activation_scales: range(scratch.output_a_scales)?,
                    weights: self.buffers.view(&layer_view(layer, "attn.wo_b.weight"))?,
                    weight_scales: self.buffers.view(&layer_view(layer, "attn.wo_b.scale"))?,
                    output: range(scratch.partial)?,
                    live_rows: live,
                    status: range(scratch.row_status)?,
                },
            )
            .map_err(|e| e.to_string())?;
        let peers = self
            .peers
            .iter()
            .map(|peer| {
                Ok(PeerViews {
                    output: range(peer.scratch.reduced)?,
                    state: peer.comm.clone(),
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        let reduced = reduce
            .reduce(
                self.rank,
                crate::program::Phase::Prefill { rows },
                &ReduceViews {
                    input: range(scratch.partial)?,
                    output: range(scratch.reduced)?,
                    live_rows: live,
                    peers,
                },
            )
            .map_err(|e| e.to_string())?;
        Ok((project, reduced))
    }
}
