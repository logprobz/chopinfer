//! F64 sqrt-softplus scores and exact selection over supplied F32 scores.

use crate::contract::{Refusal, Validate};
use crate::oracle::{RouterSelection, router};
use crate::router::RowBiasRouterQuery;
use crate::select::{NanKeys, Ties, topk};

/// Stable f64 scores over finite F32 logits. A router computes these in
/// F32; its score accuracy is checked separately from selection.
pub fn sqrt_softplus_scores(logits: &[f32]) -> Result<Vec<f64>, Refusal> {
    if logits.is_empty() || logits.iter().any(|x| !x.is_finite()) {
        return Err(Refusal(
            "sqrt-softplus router logits must be finite and nonempty",
        ));
    }
    Ok(logits
        .iter()
        .map(|&v| {
            let v = f64::from(v);
            (v.max(0.0) + (-v.abs()).exp().ln_1p()).sqrt()
        })
        .collect())
}

fn shape(
    op: RowBiasRouterQuery,
    values: &[f32],
    biases: [&[f32]; 2],
    bias_select: &[u8],
) -> Result<(), Refusal> {
    op.validate()?;
    let experts = op.experts as usize;
    if bias_select.is_empty()
        || bias_select.len().checked_mul(experts) != Some(values.len())
        || bias_select.iter().any(|&b| b > 1)
        || biases
            .iter()
            .any(|b| b.len() != experts || b.iter().any(|v| !v.is_finite()))
        || values.iter().any(|v| !v.is_finite())
    {
        return Err(Refusal(
            "row-bias router shape, selector or finite input invalid",
        ));
    }
    Ok(())
}

/// F64 mathematical routing, including each row's selected bias. It is an
/// accuracy oracle, not the exact-selection gate for nearly equal scores.
pub fn row_bias_router(
    op: RowBiasRouterQuery,
    logits: &[f32],
    biases: [&[f32]; 2],
    bias_select: &[u8],
) -> Result<Vec<RouterSelection>, Refusal> {
    shape(op, logits, biases, bias_select)?;
    let kind = crate::contract::RouterKind::SqrtSoftplusBiasSelectNormalized {
        scaling_factor: op.scaling_factor,
    };
    logits
        .chunks_exact(op.experts as usize)
        .zip(bias_select)
        .map(|(row, &select)| {
            let row = row.iter().copied().map(f64::from).collect::<Vec<_>>();
            let bias = biases[select as usize]
                .iter()
                .copied()
                .map(f64::from)
                .collect::<Vec<_>>();
            router(&row, Some(&bias), op.top_k as usize, kind)
        })
        .collect()
}

/// Exact ids over a kernel's own nonnegative finite F32 scores. Bias
/// addition rounds to F32 before ranking; this matters even when all
/// inputs are exactly representable. Weights use the selected unbiased
/// scores, with F32 adds in selection order, epsilon, division, then scale.
pub fn select_f32_scores(
    op: RowBiasRouterQuery,
    scores: &[f32],
    biases: [&[f32]; 2],
    bias_select: &[u8],
) -> Result<Vec<RouterSelection>, Refusal> {
    shape(op, scores, biases, bias_select)?;
    if scores.iter().any(|&v| v < 0.0) {
        return Err(Refusal("sqrt-softplus router scores must be nonnegative"));
    }
    scores
        .chunks_exact(op.experts as usize)
        .zip(bias_select)
        .map(|(row, &select)| {
            let keys = row
                .iter()
                .zip(biases[select as usize])
                .map(|(&s, &b)| f64::from(s + b))
                .collect::<Vec<_>>();
            if keys.iter().any(|v| !v.is_finite()) {
                return Err(Refusal("router F32 selection keys overflow"));
            }
            let expert_ids = topk(
                &keys,
                op.top_k as usize,
                Ties::LowerIndex,
                NanKeys::Refuse,
                None,
            )?;
            let sum = expert_ids.iter().fold(0.0_f32, |sum, &id| sum + row[id]) + 1e-20_f32;
            if !sum.is_finite() {
                return Err(Refusal("router F32 score sum overflow"));
            }
            let weights = expert_ids
                .iter()
                .map(|&id| f64::from((row[id] / sum) * op.scaling_factor))
                .collect::<Vec<_>>();
            if weights.iter().any(|v| !v.is_finite()) {
                return Err(Refusal("router F32 route weight overflow"));
            }
            Ok(RouterSelection {
                expert_ids,
                weights,
            })
        })
        .collect()
}
