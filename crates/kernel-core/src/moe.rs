//! Buffer roles of the MoE ops around the experts: the router
//! (`RowQuery::Router`), a rank's local routes (`MoeQuery::LocalRoutes`),
//! the routes of a router projection's logits (`ProjectedRouteBuffers`),
//! the combines of routed rows (`ExpertCombineOp`,
//! `MoeQuery::LocalRoutedSum`) and the shared-expert gate combine
//! (`MoeQuery::SharedExpertGateCombine`). Their contracts are in `contract`
//! and their F64 references in `oracle` (`route`, `router`) and
//! `moe_oracle`.
//!
//! The math a model relies on is the contract's: the score function, the
//! bias, the tie rule (lower expert id first), the normalization and scale
//! of the selected weights, the order of a token's routes (descending
//! selection order), the order of an expert's routes (token, then top-k
//! order) and the order of a combine's adds (top-k order). How a backend
//! splits routing into launches, the padding of each expert's routes and
//! the per-launch geometry are its choices.
//!
//! Padding couples a route op to the expert op that reads its sorted rows:
//! each expert's routes are padded to the row tile the expert op reads them
//! in. The expert op states that tile; the model passes it to the route op
//! (`Backend::moe_route`'s `route_tile`), which refuses a tile it cannot
//! pad to. A backend also names the padding in its implementation identity
//! (`OpInfo::implementation`) and sizes the sorted rows itself
//! (`OpInfo::bytes`). The expert op's request names the routed rows
//! without padding (`GatedMlpProjectionQuery::logical`), and the prepared
//! op reports the padded rows it reads
//! (`moe_experts::PreparedExperts::padded_rows`).
//!
//! Every op reads `live_rows`, the device count of live token rows, and
//! writes nothing for rows at or past it. Rows are dense: row `r` of a view
//! of `n`-element rows starts at element `r * n`. Ids, counts and offsets
//! are I32 and route weights F32. A routed row is indexed by its route:
//! token times top-k plus slot.

use crate::contract::{Refusal, RowQuery, Validate};
use crate::gated_mlp::RouteWeightPoint;

crate::scratch_record! {
    /// A rank's route metadata over its `local_experts` experts, renumbered
    /// from zero (`MoeQuery::LocalRoutes`), as a route op writes it and an
    /// expert op reads it (`precision::buffers::ProjectionRouting`): the
    /// routes per expert (`counts`, which a route op that writes them needs
    /// and no expert op reads), the first sorted row of each expert and the
    /// end (`offsets`, `local_experts + 1`), the route of each sorted row in
    /// token then top-k order within an expert and -1 in an expert's
    /// padding (`sorted_rows`), and the sorted rows with padding
    /// (`padded_rows`, one value), each expert's sorted rows padded to the
    /// op's route tile. The op sizes `routes.sorted_rows`
    /// (`OpInfo::bytes`). A backend whose expert GEMMs skip experts with no
    /// routes also writes the count of experts with routes followed by
    /// their ids in ascending order (`active_experts`, `local_experts + 1`);
    /// the op lists `routes.active_experts` in `OpInfo::bytes` when it
    /// writes it, and otherwise leaves it unread and unwritten.
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct RouteMetadata {
        pub counts: option,
        pub offsets: value,
        pub sorted_rows: value,
        pub padded_rows: value,
        pub active_experts: option,
    }
}

crate::scratch_record! {
    /// A rank's routes (`RowQuery::Router` then `MoeQuery::LocalRoutes`):
    /// the router's F32 `logits` rows of `experts`, the F32 score
    /// correction `bias` per expert (sigmoid-bias routers), each token's
    /// `top_k` expert `ids` and F32 route `weights` in selection order, and
    /// the rank's route metadata (`routes`). The backend sizes the
    /// workspace roles it uses (`OpInfo::bytes`): the route tile tables
    /// (`tile_counts`, `tile_offsets`) and an arrival `ticket`, which the
    /// host zeroes once at creation and binds to no other op or stream.
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct RouteBuffers {
        pub logits: value,
        pub bias: option,
        pub ids: value,
        pub weights: value,
        pub routes: record(RouteMetadata),
        pub tile_counts: option,
        pub tile_offsets: option,
        pub ticket: option,
        pub live_rows: value,
    }
}

crate::scratch_record! {
    /// A rank's routes of projected logits
    /// (`Backend::moe_projected_route`): the BF16 hidden rows `x` of the
    /// projection's K, the router projection's BF16 `weight` of `experts`
    /// rows of K in the layout the op reports (`OpInfo::weight_layouts`),
    /// and the rank's routes (`route`), whose F32 `logits` the op writes
    /// before it selects. The projection's split `partials` and arrival
    /// `semaphores` are workspace the op sizes (`OpInfo::bytes`); the host
    /// zeroes the semaphores once at creation, every launch leaves them
    /// zero, and binds them to no other op or stream.
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct ProjectedRouteBuffers {
        pub x: value,
        pub weight: value,
        pub route: record(RouteBuffers),
        pub partials: value,
        pub semaphores: value,
    }
}

crate::scratch_record! {
    /// A router's buffers (`RowQuery::Router`): the F32 `logits` rows of
    /// `experts`, the F32 score correction `bias` per expert (sigmoid-bias
    /// routers), and each token's `top_k` expert `ids` and F32 route
    /// `weights` in selection order.
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct RouterBuffers {
        pub logits: value,
        pub bias: option,
        pub ids: value,
        pub weights: value,
        pub live_rows: value,
    }
}

crate::scratch_record! {
    /// A rank's shared-expert gate combine
    /// (`MoeQuery::SharedExpertGateCombine`) of `width`: the BF16 `routed`
    /// rows, one per route and already weighted, of which the routes to the
    /// rank's experts (by the tokens' expert `ids`) are summed in top-k
    /// order; the BF16 shared expert output rows (`shared`); the BF16 layer
    /// input rows `x` and the BF16 shared expert gate weight row
    /// (`gate_weight`), whose dot product is each token's gate logit; and
    /// the BF16 `output` rows.
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct SharedCombineBuffers {
        pub routed: value,
        pub ids: value,
        pub shared: value,
        pub x: value,
        pub gate_weight: value,
        pub output: value,
        pub live_rows: value,
    }
}

/// A combine of each token's routed rows: the op whose catalog record is
/// `RowQuery::ExpertCombine { experts, top_k }`, with the math that record
/// leaves out. Each token's `top_k` routed BF16 rows of `width` (one per
/// route) are added in top-k order in F32, then rounded to BF16 once.
/// `route_weight` says where each route's weight was applied: at that
/// point of the expert MLP before the combine, so the rows are added as
/// they are, or with `None` by the combine, which multiplies each row by its
/// F32 route weight. Every field is math.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct ExpertCombineOp {
    pub experts: u32,
    pub top_k: u32,
    pub width: u32,
    pub route_weight: Option<RouteWeightPoint>,
}

impl ExpertCombineOp {
    /// The catalog record of the op.
    pub fn query(&self) -> RowQuery {
        RowQuery::ExpertCombine {
            experts: self.experts,
            top_k: self.top_k,
        }
    }
}

impl Validate for ExpertCombineOp {
    fn validate(&self) -> Result<(), Refusal> {
        self.query().validate()?;
        if self.width == 0 {
            return Err(Refusal("an expert combine has a zero width"));
        }
        Ok(())
    }
}

crate::scratch_record! {
    /// A combine of each token's routed rows (`ExpertCombineOp`) of
    /// `width`: the BF16 `routed` rows, one per route, the F32 route
    /// `weights` (read only when the combine applies them; a caller may
    /// leave them out otherwise) and the BF16 `output` rows.
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct CombineBuffers {
        pub routed: value,
        pub weights: option,
        pub output: value,
        pub live_rows: value,
    }
}

crate::scratch_record! {
    /// A rank's routed sum (`MoeQuery::LocalRoutedSum`) of `width`: the F32
    /// `routed` rows, one per route and already weighted, the tokens'
    /// global expert `ids`, `top_k` per token, and the `output` rows of the
    /// query's dtype. An F32 sum may add an F32 `addend` row per token after
    /// its routes (the query's contract states the add): a step that binds
    /// one launches the sum that adds it, a step that leaves it out the sum
    /// alone.
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct RoutedSumBuffers {
        pub routed: value,
        pub ids: value,
        pub addend: option,
        pub output: value,
        pub live_rows: value,
    }
}
