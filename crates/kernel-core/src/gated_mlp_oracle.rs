//! F64 reference arithmetic for the clamped SwiGLU MLP contracts in
//! `gated_mlp`: the clamped gated unit, dense (and shared-expert) clamped
//! MLPs, sigmoid-bias routing filtered to a rank's local experts with the
//! padded route metadata a grouped engine reads, and the forms in which a
//! kernel or a framework sums a token's weighted routes.
//!
//! Routed experts reuse `moe_oracle::expert_mlp`; this module names the
//! rounding points of the grouped engine (`ENGINE_ROUNDING`) and of a
//! framework's grouped GEMM path (`GROUPED_MM_ROUNDING`), and sums the
//! weighted routes either in F32 with one BF16 cast (`RouteSum::F32OneCast`),
//! in F32 with no cast for a later F32 add (`RouteSum::F32`), or by rounding
//! each route and adding it into a BF16 token row in ascending expert order
//! (`RouteSum::Bf16ByExpert`).

use crate::contract::{DType, MoeQuery, Refusal, RouterKind, Validate};
use crate::gated_mlp::{RouteWeightPoint, SwigluClamp};
use crate::moe_oracle::{
    ExpertMlp, ExpertMlpOutput, ExpertMlpRounding, ExpertWeights, LocalRoutes, RouteWeightAt,
    expert_mlp, local_expert_routes,
};
use crate::oracle::{
    OracleOutput, RouterSelection, SiluGlu, bf16_round, compensated_sum_with_magnitude, router,
};

/// The gated unit a clamp declares: `gate <= gate_max` (no lower bound) and
/// `-up_limit <= up <= up_limit`, before `silu(gate) * up`. NaN inputs stay
/// NaN. `silu_output` rounds `silu(gate)` as a framework that materializes
/// the activation does (BF16 only).
pub fn clamped_silu_glu(clamp: Option<SwigluClamp>, silu_output: Option<DType>) -> SiluGlu {
    let (gate_limit, up_limit) = match clamp {
        None => (None, None),
        Some(clamp) => {
            let (gate, up) = (f64::from(clamp.gate_max()), f64::from(clamp.up_limit()));
            (Some((f64::NEG_INFINITY, gate)), Some((-up, up)))
        }
    };
    SiluGlu {
        gate_limit,
        up_limit,
        silu_output,
    }
}

/// The oracle's route weight point for a contract's.
pub fn route_weight_at(point: RouteWeightPoint) -> RouteWeightAt {
    match point {
        RouteWeightPoint::AfterDown => RouteWeightAt::AfterDown,
        RouteWeightPoint::BeforeDown => RouteWeightAt::BeforeDown,
    }
}

/// The grouped engine: gate and up sums stay F32 through the clamp and
/// SwiGLU, the intermediate rounds to BF16 once, and the down sums and the
/// route weight stay F32 until the combine's one BF16 cast.
/// Like every rounding point here, F32 accumulation is modelled as exact:
/// the gate bounds the engine's F32 sums by `2^-22 K S`, which does not
/// cover sums whose products all fall below F32's normal range, and no
/// rounding point claims bit identity there.
pub const ENGINE_ROUNDING: ExpertMlpRounding = ExpertMlpRounding {
    input: None,
    gate_up_output: None,
    intermediate: Some(DType::Bf16),
    down_input: None,
    down_output: None,
    weighted_output: None,
};

/// A framework's grouped GEMM path: BF16 gate/up and down outputs and a BF16
/// product of the gated unit (pair it with `silu_output: Some(Bf16)`); the
/// F32 route weight promotes the weighted rows to F32.
pub const GROUPED_MM_ROUNDING: ExpertMlpRounding = ExpertMlpRounding {
    input: None,
    gate_up_output: Some(DType::Bf16),
    intermediate: Some(DType::Bf16),
    down_input: None,
    down_output: Some(DType::Bf16),
    weighted_output: None,
};

/// How a token's weighted routes become one row.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RouteSum {
    /// Sum the weighted routes in route order (f64 here, F32 in a kernel),
    /// then cast to BF16 once.
    F32OneCast,
    /// Sum the weighted routes in route order in F32 from +0: each route
    /// value rounds to F32 and each add rounds to nearest even, with no BF16
    /// cast (the F32 partial a later F32 add takes). Over F32 route values
    /// this is a kernel's in-order `__fadd_rn` sum bit for bit wherever no
    /// add consumes or produces NaN, and its BF16 cast is `F32OneCast`'s
    /// kernel order. A NaN sum is NaN on both sides, but PTX leaves F32 NaN
    /// encodings unspecified, so only NaN-ness compares, never its bits.
    F32,
    /// Round each weighted route to BF16 and add it into a BF16 row that
    /// starts at zero, in ascending expert order, rounding after each add
    /// (an eager per-expert loop with a BF16 `index_add`).
    Bf16ByExpert,
}

/// The weighted routes [routes, width] (route order of `routes`) summed per
/// token, [tokens, width]. S sums the routes' S and K is the longest route
/// K; the declared roundings are not covered by S.
pub fn routed_sum(
    weighted: &OracleOutput,
    routes: &LocalRoutes,
    tokens: usize,
    width: usize,
    sum: RouteSum,
) -> Result<OracleOutput, Refusal> {
    let count = routes.tokens.len();
    if tokens == 0
        || width == 0
        || routes.experts.len() != count
        || weighted.values.len() != count * width
        || weighted.sum_abs_products.len() != count * width
        || weighted.reduction_lengths.len() != count * width
        || routes.tokens.iter().any(|&t| t >= tokens)
    {
        return Err(Refusal("routed sum shape mismatch"));
    }
    let mut order = (0..count).collect::<Vec<_>>();
    if sum == RouteSum::Bf16ByExpert {
        order.sort_by_key(|&route| (routes.experts[route], route));
    }
    let mut values = vec![0.0; tokens * width];
    let mut s = vec![0.0; tokens * width];
    let mut k = vec![1; tokens * width];
    for route in order {
        let token = routes.tokens[route];
        for col in 0..width {
            let (from, to) = (route * width + col, token * width + col);
            values[to] = match sum {
                RouteSum::F32OneCast => values[to] + weighted.values[from],
                RouteSum::F32 => f64::from(values[to] as f32 + weighted.values[from] as f32),
                RouteSum::Bf16ByExpert => {
                    bf16_round(values[to] + bf16_round(weighted.values[from]))
                }
            };
            s[to] += weighted.sum_abs_products[from];
            k[to] = k[to].max(weighted.reduction_lengths[from]);
        }
    }
    if sum == RouteSum::F32OneCast {
        for value in &mut values {
            *value = bf16_round(*value);
        }
    }
    Ok(OracleOutput {
        values,
        sum_abs_products: s,
        reduction_lengths: k,
    })
}

/// A dense or shared-expert SwiGLU MLP over x [tokens, H] with `weights`
/// of one expert (gate and up rows concatenated, [2I, H], and down,
/// [H, I]): every token routes to it with weight 1, so `combined` is the
/// down projection after `weighted_output`.
pub fn dense_gated_mlp(
    x: &[f64],
    tokens: usize,
    weights: ExpertWeights<'_>,
    activation: SiluGlu,
    rounding: ExpertMlpRounding,
) -> Result<ExpertMlpOutput, Refusal> {
    if weights.experts != 1 {
        return Err(Refusal("a dense gated MLP has one expert"));
    }
    let routes = LocalRoutes {
        tokens: (0..tokens).collect(),
        experts: vec![0; tokens],
        routes: (0..tokens).collect(),
    };
    expert_mlp(
        x,
        tokens,
        &routes,
        &vec![1.0; tokens],
        weights,
        ExpertMlp {
            activation,
            route_weight_at: RouteWeightAt::AfterDown,
            rounding,
        },
    )
}

/// One rank's routing: `router` over F32 logits [tokens, experts] with the
/// selection bias, then the routes of the rank's experts with their weights,
/// in token then top-k order.
#[derive(Clone, Debug, PartialEq)]
pub struct RankRoutes {
    pub selections: Vec<RouterSelection>,
    pub local: LocalRoutes,
    pub weights: Vec<f64>,
}

pub fn rank_routes(
    logits: &[f64],
    bias: Option<&[f64]>,
    kind: RouterKind,
    query: MoeQuery,
    rank: usize,
) -> Result<RankRoutes, Refusal> {
    query.validate()?;
    let MoeQuery::LocalRoutes { experts, top_k, .. } = query else {
        return Err(Refusal("expected local expert route query"));
    };
    let (experts, top_k) = (experts as usize, top_k as usize);
    if logits.is_empty() || !logits.len().is_multiple_of(experts) {
        return Err(Refusal("router logits shape mismatch"));
    }
    let selections = logits
        .chunks_exact(experts)
        .map(|row| router(row, bias, top_k, kind))
        .collect::<Result<Vec<_>, _>>()?;
    let local = local_expert_routes(query, &selections, rank)?;
    let weights = local
        .routes
        .iter()
        .map(|&route| selections[route / top_k].weights[route % top_k])
        .collect();
    Ok(RankRoutes {
        selections,
        local,
        weights,
    })
}

/// The padded route metadata a grouped engine reads (the row engine's
/// layout over local experts): routes of each local expert in ascending
/// global route index `token * top_k + k`, each expert's range padded with
/// -1 to a multiple of `pad`, and the active experts in ascending order.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RouteMetadata {
    pub counts: Vec<i32>,
    /// `[local_experts + 1]` padded offsets.
    pub offsets: Vec<i32>,
    pub sorted_rows: Vec<i32>,
    pub num_padded_rows: i32,
    /// The active count, then the active local experts.
    pub active_experts: Vec<i32>,
}

/// Route metadata for global expert IDs [tokens, top_k] on `rank`.
pub fn local_route_metadata(
    ids: &[usize],
    top_k: usize,
    local_experts: usize,
    rank: usize,
    pad: usize,
) -> Result<RouteMetadata, Refusal> {
    if top_k == 0 || local_experts == 0 || pad == 0 || !ids.len().is_multiple_of(top_k) {
        return Err(Refusal("route metadata shape invalid"));
    }
    let mut rows = vec![Vec::new(); local_experts];
    for (route, &id) in ids.iter().enumerate() {
        if id / local_experts == rank {
            rows[id % local_experts].push(route as i32);
        }
    }
    let mut metadata = RouteMetadata {
        counts: Vec::with_capacity(local_experts),
        offsets: vec![0],
        sorted_rows: Vec::new(),
        num_padded_rows: 0,
        active_experts: vec![0],
    };
    for (expert, routes) in rows.iter().enumerate() {
        metadata.counts.push(routes.len() as i32);
        metadata.sorted_rows.extend(routes);
        let padded = routes.len().div_ceil(pad) * pad;
        metadata
            .sorted_rows
            .extend(std::iter::repeat_n(-1, padded - routes.len()));
        metadata.offsets.push(metadata.sorted_rows.len() as i32);
        if !routes.is_empty() {
            metadata.active_experts.push(expert as i32);
        }
    }
    metadata.num_padded_rows = metadata.sorted_rows.len() as i32;
    metadata.active_experts[0] = metadata.active_experts.len() as i32 - 1;
    Ok(metadata)
}

/// Neumaier-compensated f64 dot product and its sum of absolute products,
/// for long-K references (for example the 128-K accumulation gate).
pub fn compensated_dot(a: &[f64], b: &[f64]) -> Result<(f64, f64), Refusal> {
    if a.len() != b.len() {
        return Err(Refusal("compensated dot lengths differ"));
    }
    Ok(compensated_sum_with_magnitude(
        a.iter().zip(b).map(|(x, y)| x * y),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::oracle::silu_glu;

    fn silu(x: f64) -> f64 {
        x / (1.0 + (-x).exp())
    }

    #[test]
    fn clamp_bounds_gate_above_and_up_on_both_sides_before_silu() {
        let unit = clamped_silu_glu(Some(SwigluClamp::symmetric(10.0)), None);
        assert_eq!(unit.gate_limit, Some((f64::NEG_INFINITY, 10.0)));
        assert_eq!(unit.up_limit, Some((-10.0, 10.0)));
        let gate = [12.0, -30.0, 3.0, 10.0, f64::NAN, 1.0];
        let up = [2.0, 4.0, -15.0, 11.0, 1.0, f64::NAN];
        let out = unit.apply(&gate, &up).unwrap();
        assert_eq!(out[0], silu(10.0) * 2.0);
        // No lower gate bound: a very negative gate passes unchanged.
        assert_eq!(out[1], silu(-30.0) * 4.0);
        assert_eq!(out[2], silu(3.0) * -10.0);
        assert_eq!(out[3], silu(10.0) * 10.0);
        assert!(out[4].is_nan() && out[5].is_nan());
        // Unequal limits keep their halves apart.
        let clamp = SwigluClamp {
            gate_max_bits: 2.0_f32.to_bits(),
            up_limit_bits: 0.5_f32.to_bits(),
        };
        let out = clamped_silu_glu(Some(clamp), None)
            .apply(&[5.0, 5.0], &[-3.0, 0.25])
            .unwrap();
        assert_eq!(out, [silu(2.0) * -0.5, silu(2.0) * 0.25]);
        // No clamp is the unclamped unit.
        assert_eq!(clamped_silu_glu(None, None), SiluGlu::UNCLAMPED);
    }

    #[test]
    fn clamped_gemm_unit_keeps_a_valid_error_bound() {
        // A gate sum of 12 (S = 12) clamps to 10; the clamp is 1-Lipschitz,
        // so S of the unit is S_gate |up| + S_up |silu|.
        let gemm = OracleOutput::new(vec![12.0, -3.0], vec![12.0, 3.0], 4).unwrap();
        let out = silu_glu(
            &gemm,
            2,
            crate::contract::SwigluColumns::ConcatenatedHalves,
            clamped_silu_glu(Some(SwigluClamp::symmetric(10.0)), None),
        )
        .unwrap();
        assert_eq!(out.values, [silu(10.0) * -3.0]);
        assert_eq!(out.sum_abs_products, [12.0 * 3.0 + 3.0 * silu(10.0)]);
        assert_eq!(out.reduction_lengths, [4]);
    }

    #[test]
    fn sigmoid_bias_routing_selects_exactly_with_lower_index_ties() {
        let kind = RouterKind::SigmoidBiasSelectNormalized {
            scaling_factor: 2.5,
        };
        let experts = 288;
        let mut logits = vec![-4.0; experts];
        let mut bias = vec![0.0; experts];
        // Experts 7 and 250 tie on sigmoid + bias; 7 wins the tie. Expert 100
        // has the largest logit but a bias that drops it below both.
        logits[250] = 1.0;
        logits[7] = 1.0;
        logits[100] = 4.0;
        bias[100] = -1.0;
        // Six more clear winners with distinct keys.
        for (i, expert) in [20, 40, 60, 80, 120, 140].into_iter().enumerate() {
            logits[expert] = 3.0 - 0.25 * i as f64;
        }
        let query = MoeQuery::LocalRoutes {
            experts: experts as u32,
            local_experts: 72,
            top_k: 8,
            capacity: 4,
        };
        let routes = rank_routes(&logits, Some(&bias), kind, query, 0).unwrap();
        let ids = &routes.selections[0].expert_ids;
        assert_eq!(ids, &[20, 40, 60, 80, 120, 140, 7, 250]);
        let sigmoid = |v: f64| 1.0 / (1.0 + (-v).exp());
        let scores = ids.iter().map(|&e| sigmoid(logits[e])).collect::<Vec<_>>();
        let total = scores.iter().sum::<f64>() + 1e-20;
        for (weight, score) in routes.selections[0].weights.iter().zip(&scores) {
            assert_eq!(*weight, score / total * 2.5);
        }
        // Rank 0 owns experts 0..72: routes of 20, 40, 60 and 7, in top-k order.
        assert_eq!(routes.local.experts, [20, 40, 60, 7]);
        assert_eq!(routes.local.routes, [0, 1, 2, 6]);
        assert_eq!(
            routes.weights,
            [0, 1, 2, 6].map(|k| routes.selections[0].weights[k])
        );
        // Rank 3 owns 216..288: expert 250 renumbers to 34.
        let last = rank_routes(&logits, Some(&bias), kind, query, 3).unwrap();
        assert_eq!(last.local.experts, [34]);
        assert_eq!(last.local.routes, [7]);
        // The bias only ranks: without it expert 100 is selected first.
        let unbiased = rank_routes(&logits, None, kind, query, 1).unwrap();
        assert_eq!(unbiased.selections[0].expert_ids[0], 100);
    }

    #[test]
    fn route_metadata_pads_each_local_expert_in_route_order() {
        // Two tokens, top 3, 12 experts on 3 ranks of 4.
        let ids = [5, 1, 6, 6, 4, 11];
        let rank1 = local_route_metadata(&ids, 3, 4, 1, 4).unwrap();
        assert_eq!(rank1.counts, [1, 1, 2, 0]);
        assert_eq!(rank1.offsets, [0, 4, 8, 12, 12]);
        assert_eq!(
            rank1.sorted_rows,
            [4, -1, -1, -1, 0, -1, -1, -1, 2, 3, -1, -1]
        );
        assert_eq!(rank1.num_padded_rows, 12);
        assert_eq!(rank1.active_experts, [3, 0, 1, 2]);
        let rank0 = local_route_metadata(&ids, 3, 4, 0, 16).unwrap();
        assert_eq!(rank0.offsets, [0, 0, 16, 16, 16]);
        assert_eq!(rank0.active_experts, [1, 1]);
        let empty = local_route_metadata(&[], 3, 4, 0, 16).unwrap();
        assert_eq!(empty.num_padded_rows, 0);
        assert_eq!(empty.active_experts, [0]);
    }

    fn routes_of(tokens: &[usize], experts: &[usize]) -> LocalRoutes {
        LocalRoutes {
            tokens: tokens.to_vec(),
            experts: experts.to_vec(),
            routes: (0..tokens.len()).collect(),
        }
    }

    #[test]
    fn f32_route_sum_casts_once_where_bf16_adds_round_each_route() {
        // BF16 has 8 significant bits: 1 + 3 * 2^-9 rounds up to 1 + 2^-7,
        // but each 2^-9 added alone to 1 rounds away.
        let small = 2_f64.powi(-9);
        let weighted = OracleOutput::new(vec![1.0, small, small, small], vec![0.0; 4], 1).unwrap();
        let routes = routes_of(&[0, 0, 0, 0], &[3, 1, 2, 0]);
        let one_cast = routed_sum(&weighted, &routes, 1, 1, RouteSum::F32OneCast).unwrap();
        assert_eq!(one_cast.values, [1.0 + 2_f64.powi(-7)]);
        // In ascending expert order the three small routes come first and sum
        // exactly to 3 * 2^-9, then adding 1 rounds to 1 + 2^-7 as well.
        let by_expert = routed_sum(&weighted, &routes, 1, 1, RouteSum::Bf16ByExpert).unwrap();
        assert_eq!(by_expert.values, [1.0 + 2_f64.powi(-7)]);
        // With the large route on the lowest expert, every later add rounds away.
        let routes = routes_of(&[0, 0, 0, 0], &[0, 1, 2, 3]);
        let by_expert = routed_sum(&weighted, &routes, 1, 1, RouteSum::Bf16ByExpert).unwrap();
        assert_eq!(by_expert.values, [1.0]);
        let one_cast = routed_sum(&weighted, &routes, 1, 1, RouteSum::F32OneCast).unwrap();
        assert_eq!(one_cast.values, [1.0 + 2_f64.powi(-7)]);
        // A route value that is not BF16 is rounded before its add.
        let weighted = OracleOutput::new(vec![1.0 + small], vec![0.0], 1).unwrap();
        let single = routes_of(&[1], &[0]);
        let out = routed_sum(&weighted, &single, 2, 1, RouteSum::Bf16ByExpert).unwrap();
        assert_eq!(out.values, [0.0, 1.0]);
    }

    #[test]
    fn f32_route_sum_rounds_each_add_in_route_order_without_a_cast() {
        // F32 has 24 significant bits: 1 + 2^-24 is a tie that rounds to 1,
        // so after the large route each small one rounds away.
        let tiny = 2_f64.powi(-24);
        let weighted = OracleOutput::new(vec![1.0, tiny, tiny, tiny], vec![0.0; 4], 1).unwrap();
        let large_first = routes_of(&[0, 0, 0, 0], &[3, 1, 2, 0]);
        let out = routed_sum(&weighted, &large_first, 1, 1, RouteSum::F32).unwrap();
        assert_eq!(out.values, [1.0]);
        // Route order, not expert order: with the small routes first they sum
        // exactly to 3 * 2^-24, and 1 + 1.5 ulp ties to the even 1 + 2^-22.
        let weighted = OracleOutput::new(vec![tiny, tiny, tiny, 1.0], vec![0.0; 4], 1).unwrap();
        let out = routed_sum(&weighted, &large_first, 1, 1, RouteSum::F32).unwrap();
        assert_eq!(out.values, [1.0 + 2_f64.powi(-22)]);
        // No BF16 cast; F32OneCast keeps only BF16's 8 bits of the same sum.
        let one_cast = routed_sum(&weighted, &large_first, 1, 1, RouteSum::F32OneCast).unwrap();
        assert_eq!(one_cast.values, [1.0]);
        // A route value that is not F32 rounds to F32 before its add; a token
        // without routes, or whose only route is -0, stays +0; S and K sum
        // and take the maximum as in the other forms.
        let weighted = OracleOutput {
            values: vec![1.0 + 2_f64.powi(-30), -0.0],
            sum_abs_products: vec![3.0, 0.5],
            reduction_lengths: vec![7, 9],
        };
        let routes = routes_of(&[1, 2], &[0, 1]);
        let out = routed_sum(&weighted, &routes, 3, 1, RouteSum::F32).unwrap();
        let bits = out.values.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits, [0.0_f64, 1.0, 0.0].map(f64::to_bits));
        assert_eq!(out.sum_abs_products, [0.0, 3.0, 0.5]);
        assert_eq!(out.reduction_lengths, [1, 7, 9]);
        // A NaN route makes the sum NaN; only that, not its bits, is promised.
        let weighted = OracleOutput::new(vec![1.0, f64::NAN], vec![0.0; 2], 1).unwrap();
        let out = routed_sum(&weighted, &routes_of(&[0, 0], &[0, 1]), 1, 1, RouteSum::F32).unwrap();
        assert!(out.values[0].is_nan());
    }

    #[test]
    fn dense_clamped_mlp_matches_hand_values_at_both_rounding_points() {
        // H = 2, I = 1: gate = 3 + 8 = 11 clamps to 10, up = -5 - 12 = -17
        // clamps to -10; down multiplies by 2.
        let x = [1.0, 2.0];
        let gate_up = [3.0, 4.0, -5.0, -6.0];
        let down = [2.0, 0.5];
        let weights = ExpertWeights {
            gate_up: &gate_up,
            down: &down,
            experts: 1,
            hidden: 2,
            intermediate: 1,
        };
        let unit = clamped_silu_glu(Some(SwigluClamp::symmetric(10.0)), None);
        let exact = dense_gated_mlp(&x, 1, weights, unit, ExpertMlpRounding::default()).unwrap();
        let h = silu(10.0) * -10.0;
        assert_eq!(exact.intermediate.values, [h]);
        assert_eq!(exact.combined.values, [2.0 * h, 0.5 * h]);
        let engine = dense_gated_mlp(&x, 1, weights, unit, ENGINE_ROUNDING).unwrap();
        assert_eq!(
            engine.combined.values,
            [2.0 * bf16_round(h), 0.5 * bf16_round(h)]
        );
        // The grouped GEMM rounding also rounds silu(10) before the product.
        let rounded_unit = clamped_silu_glu(Some(SwigluClamp::symmetric(10.0)), Some(DType::Bf16));
        let hf = dense_gated_mlp(&x, 1, weights, rounded_unit, GROUPED_MM_ROUNDING).unwrap();
        let hf_h = bf16_round(bf16_round(silu(10.0)) * -10.0);
        assert_eq!(
            hf.combined.values,
            [bf16_round(2.0 * hf_h), bf16_round(0.5 * hf_h)]
        );
        assert_ne!(bf16_round(silu(10.0)), silu(10.0));
    }

    fn pseudo(seed: &mut u64) -> f64 {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*seed >> 11) as f64 / (1_u64 << 53) as f64) * 2.0 - 1.0
    }

    /// Sum over ranks of each rank's routed partial equals the computation
    /// with every expert local, for `local` experts per rank.
    fn rank_partials_cover_every_route(local: usize, ranks: usize) {
        let (tokens, hidden, intermediate, top_k) = (3, 4, 2, 8);
        let experts = local * ranks;
        let mut seed = local as u64;
        let logits = (0..tokens * experts)
            .map(|_| 4.0 * pseudo(&mut seed))
            .collect::<Vec<_>>();
        let bias = (0..experts)
            .map(|_| 0.1 * pseudo(&mut seed))
            .collect::<Vec<_>>();
        let x = (0..tokens * hidden)
            .map(|_| pseudo(&mut seed))
            .collect::<Vec<_>>();
        let gate_up = (0..experts * 2 * intermediate * hidden)
            .map(|_| 3.0 * pseudo(&mut seed))
            .collect::<Vec<_>>();
        let down = (0..experts * hidden * intermediate)
            .map(|_| pseudo(&mut seed))
            .collect::<Vec<_>>();
        let kind = RouterKind::SigmoidBiasSelectNormalized {
            scaling_factor: 2.5,
        };
        let mlp = ExpertMlp {
            activation: clamped_silu_glu(Some(SwigluClamp::symmetric(1.0)), None),
            route_weight_at: RouteWeightAt::AfterDown,
            rounding: ExpertMlpRounding::default(),
        };
        let run = |local_experts: usize, rank: usize, gate_up: &[f64], down: &[f64]| {
            let query = MoeQuery::LocalRoutes {
                experts: experts as u32,
                local_experts: local_experts as u32,
                top_k: top_k as u32,
                capacity: tokens as u32,
            };
            let routes = rank_routes(&logits, Some(&bias), kind, query, rank).unwrap();
            if routes.local.tokens.is_empty() {
                return (0, vec![0.0; tokens * hidden]);
            }
            let weights = ExpertWeights {
                gate_up,
                down,
                experts: local_experts,
                hidden,
                intermediate,
            };
            let out = expert_mlp(&x, tokens, &routes.local, &routes.weights, weights, mlp).unwrap();
            (routes.local.tokens.len(), out.combined.values)
        };
        let (all_routes, all) = run(experts, 0, &gate_up, &down);
        assert_eq!(all_routes, tokens * top_k);
        let mut summed = vec![0.0; tokens * hidden];
        let mut routes = 0;
        for rank in 0..ranks {
            let gate_up_shard = &gate_up[rank * local * 2 * intermediate * hidden..]
                [..local * 2 * intermediate * hidden];
            let down_shard =
                &down[rank * local * hidden * intermediate..][..local * hidden * intermediate];
            let (count, partial) = run(local, rank, gate_up_shard, down_shard);
            routes += count;
            for (s, p) in summed.iter_mut().zip(partial) {
                *s += p;
            }
        }
        assert_eq!(routes, tokens * top_k);
        for (s, a) in summed.iter().zip(&all) {
            assert!((s - a).abs() <= 1e-12 * a.abs().max(1.0), "{s} != {a}");
        }
    }

    #[test]
    fn routed_partials_cover_every_route_at_72_and_96_local_experts() {
        rank_partials_cover_every_route(72, 4);
        rank_partials_cover_every_route(96, 4);
    }

    #[test]
    fn compensated_dot_recovers_cancelled_terms() {
        let a = [1e16, 1.0, -1e16, 1.0];
        let b = [1.0; 4];
        assert_eq!(a.iter().sum::<f64>(), 1.0);
        assert_eq!(compensated_dot(&a, &b).unwrap(), (2.0, 2e16 + 2.0));
        assert!(compensated_dot(&a, &b[..3]).is_err());
    }

    #[test]
    fn contract_route_points_map_to_the_oracle() {
        assert_eq!(
            route_weight_at(RouteWeightPoint::AfterDown),
            RouteWeightAt::AfterDown
        );
        assert_eq!(
            route_weight_at(RouteWeightPoint::BeforeDown),
            RouteWeightAt::BeforeDown
        );
    }
}
