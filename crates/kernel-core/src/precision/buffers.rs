//! A projection's buffers by role, for any backend that prepares it
//! (`crate::backend::Backend::projection`).

use super::{Calibration, DynamicScale, OperationSpec};
use crate::contract::ActivationFormat;
use crate::moe::RouteMetadata;

/// Where a projection's NVFP4 activation global comes from: none, a
/// supplied calibration in the persistent weights
/// (`ProjectionBuffers::activation_global`), or the tensor amax its
/// quantizer writes to scratch (`ProjectionScratch::activation_global`).
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum ActivationGlobal {
    #[default]
    Absent,
    Supplied,
    TensorAmax,
}

impl ActivationGlobal {
    /// The one activation global recipe of `operation`'s activations.
    pub fn of(operation: &OperationSpec) -> Result<ActivationGlobal, String> {
        let mut selected = None;
        for activation in &operation.activations {
            let recipe = if activation.format == ActivationFormat::Nvfp4 {
                match activation.quantizer.as_ref().map(|q| &q.calibration) {
                    Some(Calibration::Static(_)) => ActivationGlobal::Supplied,
                    Some(Calibration::Dynamic(DynamicScale::Nvfp4TensorAmax)) => {
                        ActivationGlobal::TensorAmax
                    }
                    _ => return Err("missing activation global recipe".into()),
                }
            } else {
                ActivationGlobal::Absent
            };
            if selected.is_some_and(|previous| previous != recipe) {
                return Err("one projection requires one activation global recipe".into());
            }
            selected = Some(recipe);
        }
        Ok(selected.unwrap_or_default())
    }
}

crate::scratch_record! {
    /// A projection's buffers on one rank: its `activation` rows and the
    /// device count of live rows, its `output` rows, a `residual` the
    /// output adds, its persistent weights, a supplied activation
    /// calibration global (`ActivationGlobal::Supplied`), a grouped
    /// projection's routing, and scratch. A sharded lookup also takes this
    /// rank's first table row and its row count (zero for other
    /// projections).
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct ProjectionBuffers {
        pub activation: value,
        pub live_rows: value,
        pub output: value,
        pub residual: option,
        pub weights: record(QuantizedWeights),
        pub activation_global: option,
        pub routing: optional(ProjectionRouting),
        pub scratch: record(ProjectionScratch),
        pub row_offset: scalar(u64),
        pub shard_rows: scalar(u64),
    }
}

crate::scratch_record! {
    /// A quantized weight's persistent views, in the layout its op reports:
    /// the `codes`, the format's block scales (`local_scales`: F32 for FP8
    /// block, row and channel scales, E8M0 or E4M3 bytes for MX and NVFP4
    /// blocks) and NVFP4's F32 `global_scales`, per tensor or per expert.
    /// Projections and gated MLP projections share it.
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct QuantizedWeights {
        pub codes: value,
        pub local_scales: option,
        pub global_scales: option,
    }
}

crate::scratch_record! {
    /// A grouped projection's routing, which a route op wrote: each routed
    /// row's F32 route `weights` (the router's, `top_k` per token in
    /// selection order) and the rank's route metadata (`routes`, as
    /// `moe::RouteBuffers` holds them). An op that reads the active experts
    /// needs `routes.active_experts`.
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct ProjectionRouting {
        pub weights: value,
        pub routes: record(RouteMetadata),
    }
}

crate::scratch_record! {
    /// Scratch regions of the caller's arena; no device allocation or upload
    /// happens while launches are planned or captured. An op lists the
    /// regions it uses and their least bytes in its info: a `partial`
    /// workspace, `semaphores` zeroed before the first launch, a `temporary`
    /// output before a layout copy, `gathered` rows, quantized activation
    /// `codes` and `scales`, a dynamic `activation_global` with per-row
    /// `row_amax`, and quantized `output_codes` and `output_scales`.
    /// `unused` is any live view, for arguments an op never reads; it is
    /// never written.
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct ProjectionScratch {
        pub partial: value,
        pub semaphores: value,
        pub unused: value,
        pub temporary: option,
        pub gathered: option,
        pub codes: option,
        pub scales: option,
        pub activation_global: option,
        pub row_amax: option,
        pub output_codes: option,
        pub output_scales: option,
        /// Optional caller-owned U32 sticky status words for a trained
        /// block quantizer, one per row. Clear before a launch sequence and
        /// check before using its outputs; legacy quantizers do not use it.
        pub quantizer_status: option,
        pub quantizer_status_row_offset: scalar(u64),
    }
}
