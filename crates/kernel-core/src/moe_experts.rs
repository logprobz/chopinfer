//! Buffer roles of a gated MLP projection
//! (`backend::Backend::gated_mlp_projection`): one projection of the routed
//! experts, grouped over their routed rows, or of a dense SwiGLU MLP. Its
//! contract is `gated_mlp::GatedMlpProjectionQuery` over a `GemmQuery`, and
//! its oracles are `moe_oracle::expert_mlp` and `gated_mlp_oracle`.
//!
//! The contract's math is the projection (gate/up or down), N, K, the
//! weight, activation and output formats, the epilogues (SwiGLU column
//! order, the weighted combine, an MXFP8 output quantization), the SwiGLU
//! clamp, the route weight point, and a grouped projection's experts and
//! top-k. `m_capacity` is a grouped projection's routed rows (at least its
//! tokens times top-k), and a model requests them as such: the routed
//! capacity is
//! `RoutedCapacity::TotalRows(m_capacity)`, with no padding
//! (`GatedMlpProjectionQuery::logical`). Everything else is the backend's
//! choice: tile shapes, K splits, rings, persistent schedules, the routes'
//! padding and the weight layout it reports in `OpInfo::weight_layouts`.
//!
//! Routed rows reach a grouped projection sorted by expert, each expert's
//! range starting at a multiple of the op's route tile and padded to it
//! (`PreparedExperts::route_tile`), which the backend chooses. A model
//! passes the prepared op's route tile to the route op that permutes its
//! rows, and the prepared op reports the padded routed rows its routed
//! views hold (`PreparedExperts::padded_rows`). A backend may run fewer rows
//! than the capacity at a smaller tile; its route op then pads a launch of
//! those rows to that tile, so both agree at every row count.
//!
//! Rows are dense: row `r` of a view of `n`-element rows starts at element
//! `r * n`. A routed row is `token * top_k + slot`. BF16 values are 2
//! bytes, F32 values, ids and counts 4.

use crate::backend::{PreparedOp, View};
use crate::precision::buffers::{ProjectionRouting, QuantizedWeights};

crate::scratch_record! {
    /// A gated MLP projection's buffers on one rank.
    ///
    /// `activation` holds BF16 rows: the token rows for a dense projection
    /// or a grouped gate/up, the routed rows for a grouped down. With
    /// quantized activations it holds the routed rows already gathered in
    /// expert order and quantized, with their block `activation_scales` and,
    /// for NVFP4, the F32 `activation_global`. `output` receives a row per
    /// token (dense) or per routed row (grouped): the SwiGLU intermediate of
    /// a gate/up, or the down projection, multiplied by its route weight
    /// when the route weight applies after the down projection. A grouped
    /// projection reads `routing` (the route ops' outputs) and writes no row
    /// of a token at or past the device count `live_tokens`.
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct GatedMlpBuffers {
        pub activation: value,
        pub activation_scales: option,
        pub activation_global: option,
        pub live_tokens: value,
        pub output: value,
        pub weights: record(QuantizedWeights),
        pub routing: optional(ProjectionRouting),
        pub scratch: record(GatedMlpScratch),
    }
}

crate::scratch_record! {
    /// Scratch regions: the K-split `partial` workspace the op sizes, the
    /// MXFP8 `output_codes` and `output_scales` of a quantized output, and
    /// `unused`, any live view for arguments an op never reads, which is
    /// never written.
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct GatedMlpScratch {
        pub partial: option,
        pub output_codes: option,
        pub output_scales: option,
        pub unused: value,
    }
}

/// A prepared gated MLP projection.
pub trait PreparedExperts: PreparedOp<GatedMlpBuffers<View>> {
    /// The routed-row tile the op's kernels read routes in at its capacity:
    /// each expert's sorted routed rows start at a multiple of it and are
    /// padded to it. The route op that permutes the op's rows takes it. One
    /// for a dense projection, which reads no routes.
    fn route_tile(&self) -> u32;

    /// The routed rows with padding the op's sorted routes and routed
    /// views hold at any row count up to its capacity: every route plus
    /// each expert's padding to the route tile. Zero for a dense
    /// projection.
    fn padded_rows(&self) -> u32;
}

#[cfg(test)]
mod tests {
    use crate::backend::{Backend, Need, Refused};
    use crate::contract::{
        ActivationFormat, DType, Epilogues, GemmQuery, GroupedGemm, RoutedCapacity, SwigluColumns,
        WeightFormat,
    };
    use crate::gated_mlp::{GatedMlpProjection, GatedMlpProjectionQuery, RouteWeightPoint};
    use crate::precision::{ExecutionPhase, OperationUse};

    struct Empty;

    impl Backend for Empty {
        fn name(&self) -> &str {
            "empty"
        }
    }

    #[test]
    fn gated_mlp_projections_refuse_by_default() {
        let op = GatedMlpProjectionQuery {
            gemm: GemmQuery {
                m_capacity: 64,
                n: 512,
                k: 256,
                weight: WeightFormat::Mxfp8,
                activation: ActivationFormat::Bf16,
                output: DType::Bf16,
                epilogues: Epilogues {
                    swiglu: Some(SwigluColumns::ConcatenatedHalves),
                    ..Epilogues::default()
                },
                grouped: Some(GroupedGemm {
                    experts: 16,
                    top_k: 4,
                    capacity: RoutedCapacity::TotalRows(64),
                }),
            },
            projection: GatedMlpProjection::GateUp,
            swiglu_clamp: None,
            route_weight: Some(RouteWeightPoint::AfterDown),
        };
        let usage = OperationUse {
            phase: ExecutionPhase::Decode,
            capacity: 16,
            graph: false,
        };
        assert_eq!(
            Empty.gated_mlp_projection(&op, usage, Need::rows(16)).err(),
            Some(Refused::Unsupported)
        );
    }
}
