//! Contracts for SwiGLU MLP projections whose epilogue carries what
//! `contract::Epilogues` does not: clamped SwiGLU inputs and the point where a
//! routed expert's weight scales its output.
//!
//! `GemmQuery` already describes the projection itself, dense or grouped:
//! 72 or 96 local experts, FP8 128x128, NVFP4 or MXFP8 weights, BF16
//! activations (A16), and `Epilogues::weighted_moe_combine`. Routing is
//! `RowQuery::Router` (for example 288 experts, top 8,
//! `SigmoidBiasSelectNormalized { scaling_factor: 2.5 }`) and rank filtering
//! is `MoeQuery::LocalRoutes`. A projection that needs neither a clamp nor a
//! declared route weight point stays a plain `GemmQuery`.

use crate::contract::{
    GemmQuery, NumericalContract, NumericalContractId, Refusal, RoutedCapacity, Validate,
};

/// Clamps of the SwiGLU inputs: `gate = min(gate, gate_max)` and
/// `up = clamp(up, -up_limit, up_limit)`, applied to the F32 GEMM
/// accumulators where the unclamped SwiGLU epilogue reads them, before
/// `silu(gate) * up`. A NaN input stays NaN (IEEE `fminf`/`fmaxf` would
/// drop it, so a kernel uses compare-and-select). Both limits are F32 bits;
/// a limit of 10.0 is `10.0_f32.to_bits()`.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct SwigluClamp {
    pub gate_max_bits: u32,
    pub up_limit_bits: u32,
}

impl SwigluClamp {
    /// The same limit for both halves.
    pub fn symmetric(limit: f32) -> Self {
        Self {
            gate_max_bits: limit.to_bits(),
            up_limit_bits: limit.to_bits(),
        }
    }

    pub fn gate_max(self) -> f32 {
        f32::from_bits(self.gate_max_bits)
    }

    pub fn up_limit(self) -> f32 {
        f32::from_bits(self.up_limit_bits)
    }
}

/// Where a routed expert's route weight multiplies, which decides output
/// bits.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum RouteWeightPoint {
    /// The down projection's F32 output is multiplied by the route weight in
    /// its weighted combine epilogue, then summed into the token in route
    /// order.
    AfterDown,
    /// The gate-up epilogue multiplies each route's F32 `silu(gate) * up` by
    /// the route weight before the intermediate rounding; the down
    /// projection then sums routes with weight 1. Declared for the contract
    /// and the oracle (`moe_oracle::RouteWeightAt::BeforeDown`); a grouped
    /// engine that lacks it refuses by name.
    BeforeDown,
}

/// Which projection of a SwiGLU MLP an entry is.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum GatedMlpProjection {
    /// Gate and up halves (`Epilogues::swiglu` gives their column order),
    /// with the SwiGLU epilogue producing the intermediate.
    GateUp,
    /// The intermediate back to the hidden width.
    Down,
}

/// One projection of a dense or routed SwiGLU MLP. Dense entries (no
/// `gemm.grouped`) declare no route weight point; grouped entries must.
/// Under `AfterDown` only the down projection sets
/// `Epilogues::weighted_moe_combine`; under `BeforeDown` the gate-up entry
/// never sets it. The clamp belongs to the gate-up projection.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct GatedMlpProjectionQuery {
    pub gemm: GemmQuery,
    pub projection: GatedMlpProjection,
    pub swiglu_clamp: Option<SwigluClamp>,
    pub route_weight: Option<RouteWeightPoint>,
}

impl GatedMlpProjectionQuery {
    /// The projection as a model requests it: a grouped projection's
    /// routed capacity is its logical routed rows,
    /// `RoutedCapacity::TotalRows(gemm.m_capacity)`, every route without
    /// padding. How a backend pads each expert's routes is its own choice,
    /// which the prepared op reports (`moe_experts::PreparedExperts`); a
    /// backend's entries may hold a padded capacity.
    pub fn logical(self) -> GatedMlpProjectionQuery {
        let mut op = self;
        if let Some(grouped) = &mut op.gemm.grouped {
            grouped.capacity = RoutedCapacity::TotalRows(op.gemm.m_capacity);
        }
        op
    }
}

impl Validate for GatedMlpProjectionQuery {
    fn validate(&self) -> Result<(), Refusal> {
        self.gemm.validate()?;
        let epilogues = self.gemm.epilogues;
        match self.projection {
            GatedMlpProjection::GateUp if epilogues.swiglu.is_none() => {
                return Err(Refusal("a gate-up projection needs the SwiGLU epilogue"));
            }
            GatedMlpProjection::Down if epilogues.swiglu.is_some() => {
                return Err(Refusal("a down projection has no SwiGLU epilogue"));
            }
            GatedMlpProjection::Down if self.swiglu_clamp.is_some() => {
                return Err(Refusal(
                    "the SwiGLU clamp belongs to the gate-up projection",
                ));
            }
            _ => {}
        }
        if let Some(clamp) = self.swiglu_clamp {
            let valid = |v: f32| v.is_finite() && v > 0.0;
            if !valid(clamp.gate_max()) || !valid(clamp.up_limit()) {
                return Err(Refusal("SwiGLU clamp limits must be finite and positive"));
            }
        }
        match (self.gemm.grouped, self.route_weight) {
            (None, Some(_)) => Err(Refusal("a dense MLP projection has no route weight")),
            (Some(_), None) => Err(Refusal(
                "a routed expert projection declares its route weight point",
            )),
            (Some(_), Some(RouteWeightPoint::AfterDown))
                if epilogues.weighted_moe_combine
                    != (self.projection == GatedMlpProjection::Down) =>
            {
                Err(Refusal(
                    "route weight after down needs the weighted combine on the down projection only",
                ))
            }
            (Some(_), Some(RouteWeightPoint::BeforeDown))
                if self.projection == GatedMlpProjection::GateUp
                    && epilogues.weighted_moe_combine =>
            {
                Err(Refusal(
                    "route weight before down leaves the gate-up projection uncombined",
                ))
            }
            _ => Ok(()),
        }
    }
}

impl NumericalContract for GatedMlpProjectionQuery {
    /// The GEMM's ID with the projection, clamp and route weight point.
    fn numerical_contract_id(&self) -> Result<NumericalContractId, Refusal> {
        self.validate()?;
        let gemm = self.gemm.numerical_contract_id()?;
        let projection = match self.projection {
            GatedMlpProjection::GateUp => "gate-up",
            GatedMlpProjection::Down => "down",
        };
        let clamp = if self.swiglu_clamp.is_some() {
            ".swiglu-clamp"
        } else {
            ""
        };
        let route = match self.route_weight {
            None => "",
            Some(RouteWeightPoint::AfterDown) => ".route-after-down",
            Some(RouteWeightPoint::BeforeDown) => ".route-before-down",
        };
        Ok(NumericalContractId::new(format!(
            "{gemm}.gated-mlp-{projection}{clamp}{route}"
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contract::{
        ActivationFormat, DType, Epilogues, GroupedGemm, MoeQuery, RoutedCapacity, RouterKind,
        RowQuery, ScaleMode, SwigluColumns, WeightFormat,
    };

    fn expert(
        weight: WeightFormat,
        experts: u32,
        projection: GatedMlpProjection,
    ) -> GatedMlpProjectionQuery {
        let (n, k) = match projection {
            GatedMlpProjection::GateUp => (4096, 4096),
            GatedMlpProjection::Down => (4096, 2048),
        };
        GatedMlpProjectionQuery {
            gemm: GemmQuery {
                m_capacity: 2048,
                n,
                k,
                weight,
                activation: ActivationFormat::Bf16,
                output: DType::Bf16,
                epilogues: Epilogues {
                    swiglu: (projection == GatedMlpProjection::GateUp)
                        .then_some(SwigluColumns::ConcatenatedHalves),
                    weighted_moe_combine: projection == GatedMlpProjection::Down,
                    ..Epilogues::default()
                },
                grouped: Some(GroupedGemm {
                    experts,
                    top_k: 8,
                    capacity: RoutedCapacity::TotalRows(2048),
                }),
            },
            projection,
            swiglu_clamp: (projection == GatedMlpProjection::GateUp)
                .then(|| SwigluClamp::symmetric(10.0)),
            route_weight: Some(RouteWeightPoint::AfterDown),
        }
    }

    const FORMATS: [WeightFormat; 3] = [
        WeightFormat::Fp8Block128x128(ScaleMode::ExactF32),
        WeightFormat::Nvfp4,
        WeightFormat::Mxfp8,
    ];

    #[test]
    fn routed_experts_at_72_and_96_local_validate_in_every_weight_format() {
        for experts in [72, 96] {
            for weight in FORMATS {
                for projection in [GatedMlpProjection::GateUp, GatedMlpProjection::Down] {
                    let query = expert(weight, experts, projection);
                    query.validate().unwrap();
                    query.numerical_contract_id().unwrap();
                }
            }
        }
        let routes = |local| MoeQuery::LocalRoutes {
            experts: local * 4,
            local_experts: local,
            top_k: 8,
            capacity: 2048,
        };
        routes(72).validate().unwrap();
        routes(96).validate().unwrap();
        RowQuery::Router {
            experts: 288,
            top_k: 8,
            kind: RouterKind::SigmoidBiasSelectNormalized {
                scaling_factor: 2.5,
            },
        }
        .validate()
        .unwrap();
    }

    #[test]
    fn clamp_and_route_point_name_distinct_contracts() {
        let clamped = expert(FORMATS[0], 72, GatedMlpProjection::GateUp);
        let plain = GatedMlpProjectionQuery {
            swiglu_clamp: None,
            ..clamped
        };
        assert_eq!(
            clamped.numerical_contract_id().unwrap().as_str(),
            "gemm.w-fp8-block128-f32.a-bf16.acc-f32.v1.gated-mlp-gate-up.swiglu-clamp.route-after-down"
        );
        assert_ne!(
            clamped.numerical_contract_id().unwrap(),
            plain.numerical_contract_id().unwrap()
        );
        let before = GatedMlpProjectionQuery {
            route_weight: Some(RouteWeightPoint::BeforeDown),
            ..clamped
        };
        assert!(
            before
                .numerical_contract_id()
                .unwrap()
                .as_str()
                .ends_with(".route-before-down")
        );
        let clamp = clamped.swiglu_clamp.unwrap();
        assert_eq!((clamp.gate_max(), clamp.up_limit()), (10.0, 10.0));
    }

    #[test]
    fn a_logical_request_holds_the_unpadded_routes_with_the_same_identity() {
        let mut padded = expert(FORMATS[1], 72, GatedMlpProjection::Down);
        padded.gemm.grouped.as_mut().unwrap().capacity = RoutedCapacity::TotalRows(2048 + 72 * 15);
        let logical = padded.logical();
        assert_eq!(
            logical.gemm.grouped.unwrap().capacity,
            RoutedCapacity::TotalRows(2048)
        );
        assert_eq!(logical.logical(), logical);
        assert_eq!(
            logical.numerical_contract_id(),
            padded.numerical_contract_id()
        );
        let mut dense = padded;
        dense.gemm.grouped = None;
        assert_eq!(dense.logical(), dense);
    }

    #[test]
    fn dense_clamped_projection_declares_no_route_weight() {
        let mut dense = expert(FORMATS[0], 72, GatedMlpProjection::GateUp);
        dense.gemm.grouped = None;
        dense.gemm.n = 2 * 3072;
        assert_eq!(
            dense.validate(),
            Err(Refusal("a dense MLP projection has no route weight"))
        );
        dense.route_weight = None;
        dense.validate().unwrap();
        assert_eq!(
            dense.numerical_contract_id().unwrap().as_str(),
            "gemm.w-fp8-block128-f32.a-bf16.acc-f32.v1.gated-mlp-gate-up.swiglu-clamp"
        );
    }

    #[test]
    fn inconsistent_declarations_refuse_by_name() {
        let gate_up = expert(FORMATS[1], 72, GatedMlpProjection::GateUp);
        let down = expert(FORMATS[1], 72, GatedMlpProjection::Down);
        let cases = [
            (
                GatedMlpProjectionQuery {
                    route_weight: None,
                    ..down
                },
                "a routed expert projection declares its route weight point",
            ),
            (
                GatedMlpProjectionQuery {
                    swiglu_clamp: Some(SwigluClamp::symmetric(10.0)),
                    ..down
                },
                "the SwiGLU clamp belongs to the gate-up projection",
            ),
            (
                GatedMlpProjectionQuery {
                    swiglu_clamp: Some(SwigluClamp::symmetric(f32::INFINITY)),
                    ..gate_up
                },
                "SwiGLU clamp limits must be finite and positive",
            ),
            (
                GatedMlpProjectionQuery {
                    projection: GatedMlpProjection::Down,
                    swiglu_clamp: None,
                    ..gate_up
                },
                "a down projection has no SwiGLU epilogue",
            ),
            (
                GatedMlpProjectionQuery {
                    projection: GatedMlpProjection::GateUp,
                    ..down
                },
                "a gate-up projection needs the SwiGLU epilogue",
            ),
        ];
        for (query, message) in cases {
            assert_eq!(query.validate(), Err(Refusal(message)));
        }
        let mut combined_gate_up = gate_up;
        combined_gate_up.gemm.epilogues.weighted_moe_combine = true;
        assert_eq!(
            combined_gate_up.validate(),
            Err(Refusal(
                "route weight after down needs the weighted combine on the down projection only"
            ))
        );
        combined_gate_up.route_weight = Some(RouteWeightPoint::BeforeDown);
        assert_eq!(
            combined_gate_up.validate(),
            Err(Refusal(
                "route weight before down leaves the gate-up projection uncombined"
            ))
        );
    }
}
