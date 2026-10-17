//! Routers whose selection bias is chosen independently for each row.

use crate::contract::{
    NumericalContract, NumericalContractId, Refusal, RouterKind, RowQuery, Validate,
};
use crate::moe::{RouteBuffers, RouterBuffers};

/// Sqrt-softplus routing over F32 logits, with two F32 bias vectors of
/// `experts` elements. A U8 selector per live token chooses vector 0 or 1.
/// The bias changes selection only; weights normalize the unbiased scores
/// with epsilon 1e-20 and apply `scaling_factor` (for example 1.5).
/// Equal F32 selection keys keep the lower expert id. The selection gate
/// uses the implementation's own F32 scores, independently of the f64
/// score-accuracy gate.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RowBiasRouterQuery {
    pub experts: u32,
    pub top_k: u32,
    pub scaling_factor: f32,
}

impl RowBiasRouterQuery {
    pub fn query(self) -> RowQuery {
        RowQuery::Router {
            experts: self.experts,
            top_k: self.top_k,
            kind: RouterKind::SqrtSoftplusBiasSelectNormalized {
                scaling_factor: self.scaling_factor,
            },
        }
    }
}

impl Validate for RowBiasRouterQuery {
    fn validate(&self) -> Result<(), Refusal> {
        self.query().validate()
    }
}

impl NumericalContract for RowBiasRouterQuery {
    fn numerical_contract_id(&self) -> Result<NumericalContractId, Refusal> {
        Ok(NumericalContractId::new(format!(
            "{}.row-bias-select",
            self.query().numerical_contract_id()?
        )))
    }
}

crate::scratch_record! {
    /// A row-bias router: `router.bias` is required and holds vector 0;
    /// `alternate_bias` holds vector 1. `bias_select` holds one U8 per
    /// token, each 0 or 1. All logits, scores, biases and weights are F32.
    /// `scores` receives the unbiased sqrt-softplus scores [tokens, experts]
    /// used for selection, so exact selection can be checked independently
    /// of score error. Rows at or past `router.live_rows` are not accessed.
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct RowBiasRouterBuffers {
        pub router: record(RouterBuffers),
        pub alternate_bias: value,
        pub bias_select: value,
        pub scores: value,
    }
}

crate::scratch_record! {
    /// Row-bias selection followed by local route metadata. `route.bias`
    /// is required and holds vector 0; the other roles have the same
    /// meaning as in `RowBiasRouterBuffers`. Route metadata retains its
    /// existing token then top-k ordering within each expert.
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct RowBiasRouteBuffers {
        pub route: record(RouteBuffers),
        pub alternate_bias: value,
        pub bias_select: value,
        pub scores: value,
    }
}
