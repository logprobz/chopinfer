//! F64 reference arithmetic for the expert-sharded MoE:
//! checkpoint FP8 block weights, local route filtering, the routed expert MLP
//! and the shared expert's sigmoid-gated combine.

use crate::codec::{SmallFloat, bf16_to_f32};
use crate::contract::{DType, MoeQuery, Refusal, SwigluColumns, Validate};
use crate::oracle::{
    ExpertRowRange, GemmEpilogue, GemmShape, OracleOutput, RouterSelection, SiluGlu, bf16_round,
    expert_combine, gemm, silu_glu,
};
use crate::quantize::{self, QuantFormat, Quantized, Quantizer, ScaleFormat, Scales};
use crate::recurrent_oracle::sigmoid;

/// A checkpoint FP8 E4M3 weight [rows, k] with BF16 `weight_scale_inv` per
/// 128 x 128 block, [ceil(rows / 128), ceil(k / 128)]. Both references
/// multiply the code by the scale. BF16 scales convert to F32 exactly.
pub fn fp8_block_checkpoint_weight(
    codes: &[u8],
    rows: usize,
    k: usize,
    scale_inv: &[u16],
) -> Result<Quantized, Refusal> {
    if rows == 0
        || k == 0
        || codes.len() != rows * k
        || scale_inv.len() != rows.div_ceil(128) * k.div_ceil(128)
    {
        return Err(Refusal("FP8 block checkpoint weight shape mismatch"));
    }
    let scales = scale_inv.iter().map(|&bits| bf16_to_f32(bits)).collect();
    let format = QuantFormat {
        element: SmallFloat::E4M3,
        block: (128, 128),
        scale: ScaleFormat::Bf16,
        chunk_rows: None,
    };
    Quantized::from_parts(rows, k, format, codes.to_vec(), Scales::F32(scales), 1.0)
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LocalRoutes {
    pub tokens: Vec<usize>,
    /// Local expert IDs, zero-based on this rank.
    pub experts: Vec<usize>,
    /// Index of each kept route in the token-major global route list.
    pub routes: Vec<usize>,
}

/// Filter each token's selection to this rank's experts.
pub fn local_expert_routes(
    query: MoeQuery,
    selections: &[RouterSelection],
    rank: usize,
) -> Result<LocalRoutes, Refusal> {
    query.validate()?;
    let MoeQuery::LocalRoutes {
        experts,
        local_experts,
        top_k,
        capacity,
    } = query
    else {
        return Err(Refusal("expected local expert route query"));
    };
    let (local, top_k) = (local_experts as usize, top_k as usize);
    if selections.is_empty()
        || selections.len() > capacity as usize
        || rank >= (experts / local_experts) as usize
        || selections.iter().any(|s| {
            s.expert_ids.len() != top_k || s.expert_ids.iter().any(|&e| e >= experts as usize)
        })
    {
        return Err(Refusal(
            "local expert routes need full selections and a valid rank",
        ));
    }
    let mut routes = LocalRoutes {
        tokens: Vec::new(),
        experts: Vec::new(),
        routes: Vec::new(),
    };
    for (token, selection) in selections.iter().enumerate() {
        for (slot, &expert) in selection.expert_ids.iter().enumerate() {
            if expert / local == rank {
                routes.tokens.push(token);
                routes.experts.push(expert % local);
                routes.routes.push(token * top_k + slot);
            }
        }
    }
    Ok(routes)
}

/// Decoded expert weights: gate and up rows concatenated, [experts, 2I, H],
/// and down, [experts, H, I].
#[derive(Clone, Copy, Debug)]
pub struct ExpertWeights<'a> {
    pub gate_up: &'a [f64],
    pub down: &'a [f64],
    pub experts: usize,
    pub hidden: usize,
    pub intermediate: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExpertMlpOutput {
    /// SwiGLU output per route, over the decoded GEMM input, [routes, I].
    pub intermediate: OracleOutput,
    /// Down projection per route, over the decoded intermediate, [routes, H].
    pub routed: OracleOutput,
    /// Each route's down projection carrying its route weight, after
    /// `weighted_output`, [routes, H]. Under `BeforeDown` this is `routed`.
    pub weighted: OracleOutput,
    /// Sum of the weighted routes per token, in route order, [tokens, H].
    pub combined: OracleOutput,
}

/// Where the route weight multiplies: the down projection's output, or its
/// input (the gated intermediate).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RouteWeightAt {
    AfterDown,
    BeforeDown,
}

/// Rounding points of an expert MLP, in data order. `input` and
/// `down_input` emulate the quantization of each GEMM's input after an F32
/// cast; they must use one-row blocks with no chunk grid. `gate_up_output`
/// rounds the gate and up projections before any clamp or activation;
/// `intermediate` rounds the down projection's input (after the route weight
/// under `BeforeDown`) before `down_input`; `down_output` rounds each routed
/// down projection; `weighted_output` rounds it after the route weight
/// (under `AfterDown`) and before the combine. Those four round through F32
/// and accept only BF16. Projections, activation, route weighting and the
/// combine are otherwise f64, and S does not cover the declared roundings.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ExpertMlpRounding {
    pub input: Option<Quantizer>,
    pub gate_up_output: Option<DType>,
    pub intermediate: Option<DType>,
    pub down_input: Option<Quantizer>,
    pub down_output: Option<DType>,
    pub weighted_output: Option<DType>,
}

/// Every numerically relevant choice of a routed expert MLP.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ExpertMlp {
    pub activation: SiluGlu,
    pub route_weight_at: RouteWeightAt,
    pub rounding: ExpertMlpRounding,
}

/// Whether a value rounding point is BF16; other formats are refused.
fn bf16_point(point: Option<DType>) -> Result<bool, Refusal> {
    match point {
        None => Ok(false),
        Some(DType::Bf16) => Ok(true),
        Some(_) => Err(Refusal(
            "expert MLP value rounding points support BF16 only",
        )),
    }
}

fn round_bf16_if(round: bool, values: &mut [f64]) {
    if round {
        for value in values {
            *value = bf16_round(*value);
        }
    }
}

fn rounded_input(
    values: &[f64],
    rows: usize,
    k: usize,
    quantizer: Option<Quantizer>,
) -> Result<Vec<f64>, Refusal> {
    let Some(quantizer) = quantizer else {
        return Ok(values.to_vec());
    };
    if quantizer.format.block.0 != 1 || quantizer.format.chunk_rows.is_some() {
        return Err(Refusal(
            "expert MLP input quantization needs one-row blocks",
        ));
    }
    let input = values.iter().map(|&v| v as f32).collect::<Vec<_>>();
    Ok(quantize::quantize(&input, rows, k, quantizer)?.decoded())
}

/// Routed SwiGLU experts for `tokens` rows of x [tokens, H]. Each route runs
/// gate_up (gate and up as concatenated halves), the gated unit and down;
/// routes combine into their tokens in route order. Under `BeforeDown` the
/// routed outputs already carry their weights and combine with weight 1.
/// The intermediate output is the gated unit before the route weight and
/// the intermediate rounding; routed outputs follow `down_output`.
pub fn expert_mlp(
    x: &[f64],
    tokens: usize,
    routes: &LocalRoutes,
    route_weights: &[f64],
    weights: ExpertWeights<'_>,
    mlp: ExpertMlp,
) -> Result<ExpertMlpOutput, Refusal> {
    let ExpertWeights {
        gate_up,
        down,
        experts,
        hidden,
        intermediate,
    } = weights;
    let count = routes.tokens.len();
    if tokens == 0
        || x.len() != tokens * hidden
        || count == 0
        || routes.experts.len() != count
        || route_weights.len() != count
        || routes.tokens.iter().any(|&t| t >= tokens)
        || routes.experts.iter().any(|&e| e >= experts)
        || gate_up.len() != experts * 2 * intermediate * hidden
        || down.len() != experts * hidden * intermediate
    {
        return Err(Refusal("routed expert MLP operand shape mismatch"));
    }
    let round_gate_up = bf16_point(mlp.rounding.gate_up_output)?;
    let round_intermediate = bf16_point(mlp.rounding.intermediate)?;
    let round_down = bf16_point(mlp.rounding.down_output)?;
    let round_weighted = bf16_point(mlp.rounding.weighted_output)?;
    let x = rounded_input(x, tokens, hidden, mlp.rounding.input)?;
    let mut order = (0..count).collect::<Vec<_>>();
    order.sort_by_key(|&route| routes.experts[route]);
    let rows = order
        .iter()
        .flat_map(|&route| {
            let token = routes.tokens[route];
            x[token * hidden..(token + 1) * hidden].to_vec()
        })
        .collect::<Vec<_>>();
    let mut ranges = Vec::<ExpertRowRange>::new();
    for (row, &route) in order.iter().enumerate() {
        let expert = routes.experts[route];
        match ranges.last_mut() {
            Some(range) if range.expert == expert => range.end = row + 1,
            _ => ranges.push(ExpertRowRange {
                expert,
                start: row,
                end: row + 1,
            }),
        }
    }
    let gate_shape = GemmShape {
        m: count,
        n: 2 * intermediate,
        k: hidden,
        experts,
    };
    let no_epilogue = GemmEpilogue {
        residual: None,
        swiglu: None,
    };
    let mut gate_up_out = gemm(&rows, gate_up, gate_shape, &ranges, no_epilogue)?;
    round_bf16_if(round_gate_up, &mut gate_up_out.values);
    let h = silu_glu(
        &gate_up_out,
        2 * intermediate,
        SwigluColumns::ConcatenatedHalves,
        mlp.activation,
    )?;
    let mut down_input = h.values.clone();
    if mlp.route_weight_at == RouteWeightAt::BeforeDown {
        for (row, &route) in order.iter().enumerate() {
            for value in &mut down_input[row * intermediate..(row + 1) * intermediate] {
                *value *= route_weights[route];
            }
        }
    }
    round_bf16_if(round_intermediate, &mut down_input);
    let h_input = rounded_input(&down_input, count, intermediate, mlp.rounding.down_input)?;
    let down_shape = GemmShape {
        m: count,
        n: hidden,
        k: intermediate,
        experts,
    };
    let mut y = gemm(&h_input, down, down_shape, &ranges, no_epilogue)?;
    round_bf16_if(round_down, &mut y.values);
    // Undo the expert grouping so outputs follow route order.
    let mut position = vec![0; count];
    for (row, &route) in order.iter().enumerate() {
        position[route] = row;
    }
    let reorder = |output: &OracleOutput, width: usize| OracleOutput {
        values: (0..count)
            .flat_map(|r| output.values[position[r] * width..(position[r] + 1) * width].to_vec())
            .collect(),
        sum_abs_products: (0..count)
            .flat_map(|r| {
                output.sum_abs_products[position[r] * width..(position[r] + 1) * width].to_vec()
            })
            .collect(),
        reduction_lengths: (0..count)
            .flat_map(|r| {
                output.reduction_lengths[position[r] * width..(position[r] + 1) * width].to_vec()
            })
            .collect(),
    };
    let intermediate_out = reorder(&h, intermediate);
    let routed = reorder(&y, hidden);
    let mut weighted = routed.clone();
    if mlp.route_weight_at == RouteWeightAt::AfterDown {
        for (route, &weight) in route_weights.iter().enumerate() {
            let row = route * hidden..(route + 1) * hidden;
            for value in &mut weighted.values[row.clone()] {
                *value *= weight;
            }
            for bound in &mut weighted.sum_abs_products[row] {
                *bound *= weight.abs();
            }
        }
    }
    round_bf16_if(round_weighted, &mut weighted.values);
    let unit = vec![1.0; count];
    let combined = expert_combine(&weighted.values, &routes.tokens, &unit, tokens, hidden)?;
    Ok(ExpertMlpOutput {
        intermediate: intermediate_out,
        routed,
        weighted,
        combined,
    })
}

/// `routed + sigmoid(gate_logit) * shared`; routed and shared are
/// `[tokens, width]` and gate logits are `[tokens]`.
pub fn shared_expert_gate_combine(
    query: MoeQuery,
    routed: &[f64],
    shared: &[f64],
    gate_logits: &[f64],
) -> Result<OracleOutput, Refusal> {
    query.validate()?;
    let MoeQuery::SharedExpertGateCombine { width, capacity } = query else {
        return Err(Refusal("expected shared expert combine query"));
    };
    let width = width as usize;
    let tokens = gate_logits.len();
    if tokens == 0
        || tokens > capacity as usize
        || routed.len() != tokens * width
        || shared.len() != routed.len()
    {
        return Err(Refusal("shared expert combine shape mismatch"));
    }
    let mut values = Vec::with_capacity(routed.len());
    let mut bounds = Vec::with_capacity(routed.len());
    for (token, &logit) in gate_logits.iter().enumerate() {
        let gate = sigmoid(logit);
        for col in 0..width {
            let index = token * width + col;
            let product = gate * shared[index];
            values.push(routed[index] + product);
            bounds.push(routed[index].abs() + product.abs());
        }
    }
    OracleOutput::new(values, bounds, 2)
}
