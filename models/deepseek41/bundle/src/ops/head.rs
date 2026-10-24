//! The LM head: the vendor's `F.linear(x.float(), head.weight.float())`
//! (`model.py`'s `ParallelHead`) over the rank's vocabulary shard. The
//! final norm's F32 rows (a prefill's gathered head rows) are the exact F32
//! projection's activation (`ActivationFormat::F32`,
//! `DecodeOrder::ExactF32Operands`), which writes the rank's F32 logits,
//! which the logits moves (`ops::logits`) assemble.

use kernel_core::backend::{Backend, Prepared, Refused, View};
use kernel_core::contract::{GemmQuery, Refusal};
use kernel_core::precision::buffers::ProjectionBuffers;
use runtime_bundle_abi::DeviceBufferSpanV1;

use super::exact_f32::{operation, project};
use super::{Asks, Lowered, exact};
use crate::buffers::logits_rows;
use crate::shape::Shape;

pub(crate) use super::exact_f32::USE;

/// The head's projection of `shape`: the rank's vocabulary rows of the
/// hidden width, BF16 weights over F32 activations into F32 logits.
pub(crate) fn query(shape: &Shape) -> GemmQuery {
    super::exact_f32::query(shape.vocab_rows(), shape.dims().hidden)
}

/// The head's op of an instance.
pub(crate) struct HeadOps {
    pub(crate) project: Prepared<ProjectionBuffers<View>>,
    wide: Option<super::exact_f32::cached::Wide>,
}

/// A step's spans of the head: its F32 rows, the rank's head weight shard,
/// its F32 logits and the live row count.
pub(crate) struct HeadViews {
    pub input: DeviceBufferSpanV1,
    pub weight: DeviceBufferSpanV1,
    pub logits: DeviceBufferSpanV1,
    pub live_rows: DeviceBufferSpanV1,
    pub state: super::exact_f32::ProjectionState,
}

impl HeadOps {
    /// The head of `shape`, asked for the most rows any phase scores.
    pub(crate) fn prepare(
        backend: &dyn Backend,
        shape: &Shape,
        asks: &Asks,
        target: bool,
    ) -> Result<HeadOps, Refused> {
        let need = exact::need(logits_rows(shape), exact::every_phase(shape.verify_rows));
        let project = backend
            .projection(&operation("head", "head.weight", query(shape))?, USE, need)?
            .handle;
        asks.record("head.project", need, project.info());
        let wide = super::exact_f32::cached::prepare(
            shape,
            asks,
            "head.project wide verify",
            shape.vocab_rows(),
            shape.dims().hidden,
            target,
        )?;
        Ok(HeadOps { project, wide })
    }

    /// The codes layout the head reads its weight view in.
    pub(crate) fn weight_layout(&self) -> Result<String, String> {
        super::exact_f32::codes_layout(self.project.info())
    }

    /// The head's projection of `rows` rows.
    pub(crate) fn head(
        &self,
        phase: crate::program::Phase,
        rows: u32,
        views: &HeadViews,
    ) -> Result<Vec<Lowered>, Refusal> {
        if let Some(wide) = self.wide.as_ref().filter(|wide| wide.for_phase(phase)) {
            return wide
                .project(
                    rows,
                    views.input,
                    views.weight,
                    views.logits,
                    views.live_rows,
                    views.state,
                )
                .map(super::exact_f32::cached::CachedProjectionStages::into_launches);
        }
        project(
            &self.project,
            rows,
            views.input,
            views.weight,
            views.logits,
            views.live_rows,
            views.state,
        )
    }
}
