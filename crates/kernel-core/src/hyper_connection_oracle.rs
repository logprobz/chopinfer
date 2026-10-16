//! F64 reference arithmetic for the hyper-connection residual.
//! X is [tokens, streams, H]; every row op works per token.

use crate::contract::{Refusal, SigmoidMeanHyperConnectionQuery, Validate};
use crate::oracle::{GemmEpilogue, GemmShape, NormOffset, OracleOutput, RmsNorm, all_reduce, gemm};
use crate::recurrent_oracle::{sigmoid, silu};

fn tokens_of(len: usize, width: usize, capacity: u32) -> Result<usize, Refusal> {
    if len == 0 || !len.is_multiple_of(width) || len / width > capacity as usize {
        return Err(Refusal("hyper-connection operand shape mismatch"));
    }
    Ok(len / width)
}

/// Gemma RMSNorm of each stream's H values, with a separate weight per stream.
/// The input is first added to +0.0, which turns -0.0 into +0.0, as this
/// reference always has.
fn grouped_norm_values(
    x: &[f64],
    weight: &[f64],
    width: usize,
    eps: f64,
) -> Result<OracleOutput, Refusal> {
    let x = x.iter().map(|&v| v + 0.0).collect::<Vec<_>>();
    RmsNorm::f64(eps, NormOffset::OnePlusWeight).apply(&x, weight, width)
}

/// x is [tokens, streams, H] and weight is [streams, H].
pub fn grouped_norm(
    query: SigmoidMeanHyperConnectionQuery,
    x: &[f64],
    weight: &[f64],
) -> Result<OracleOutput, Refusal> {
    query.validate()?;
    let SigmoidMeanHyperConnectionQuery::GroupedNorm {
        streams,
        width,
        eps_bits,
        capacity,
    } = query
    else {
        return Err(Refusal("expected hyper-connection grouped norm query"));
    };
    let (s, h) = (streams as usize, width as usize);
    tokens_of(x.len(), s * h, capacity)?;
    if weight.len() != s * h {
        return Err(Refusal("hyper-connection norm weight shape mismatch"));
    }
    grouped_norm_values(x, weight, h, f64::from(f32::from_bits(eps_bits)))
}

/// down is [tokens, rank]. SiLU is at most 1.1-Lipschitz.
pub fn bottleneck(
    query: SigmoidMeanHyperConnectionQuery,
    down: &[f64],
) -> Result<OracleOutput, Refusal> {
    query.validate()?;
    let SigmoidMeanHyperConnectionQuery::Bottleneck {
        streams,
        rank,
        capacity,
    } = query
    else {
        return Err(Refusal("expected hyper-connection bottleneck query"));
    };
    tokens_of(down.len(), rank as usize, capacity)?;
    let streams = f64::from(streams);
    let values = down.iter().map(|&d| silu(d / streams)).collect::<Vec<_>>();
    let bounds = down.iter().map(|d| 1.1 * d.abs() / streams).collect();
    OracleOutput::new(values, bounds, 1)
}

/// up and normed are [tokens, streams, H]; the output is [tokens, H].
pub fn mix(
    query: SigmoidMeanHyperConnectionQuery,
    up: &[f64],
    normed: &[f64],
) -> Result<OracleOutput, Refusal> {
    query.validate()?;
    let SigmoidMeanHyperConnectionQuery::Mix {
        streams,
        width,
        capacity,
    } = query
    else {
        return Err(Refusal("expected hyper-connection mix query"));
    };
    let (s, h) = (streams as usize, width as usize);
    let tokens = tokens_of(up.len(), s * h, capacity)?;
    if normed.len() != up.len() {
        return Err(Refusal("hyper-connection mix shape mismatch"));
    }
    let mut values = vec![0.0; tokens * h];
    let mut bounds = vec![0.0; tokens * h];
    for token in 0..tokens {
        for stream in 0..s {
            for col in 0..h {
                let index = (token * s + stream) * h + col;
                let product = sigmoid(up[index]) * normed[index] / s as f64;
                values[token * h + col] += product;
                bounds[token * h + col] += product.abs();
            }
        }
    }
    OracleOutput::new(values, bounds, s)
}

/// X [tokens, streams, H] gains `2 * sigmoid(inject_s / streams) * y` per
/// stream; inject logits are [tokens, streams] and y is [tokens, H].
pub fn inject(
    query: SigmoidMeanHyperConnectionQuery,
    x: &[f64],
    y: &[f64],
    inject_logits: &[f64],
) -> Result<OracleOutput, Refusal> {
    query.validate()?;
    let SigmoidMeanHyperConnectionQuery::Inject {
        streams,
        width,
        capacity,
    } = query
    else {
        return Err(Refusal("expected hyper-connection inject query"));
    };
    inject_values(
        x,
        y,
        inject_logits,
        streams as usize,
        width as usize,
        capacity,
    )
}

fn inject_values(
    x: &[f64],
    y: &[f64],
    inject_logits: &[f64],
    streams: usize,
    width: usize,
    capacity: u32,
) -> Result<OracleOutput, Refusal> {
    let tokens = tokens_of(x.len(), streams * width, capacity)?;
    if y.len() != tokens * width || inject_logits.len() != tokens * streams {
        return Err(Refusal("hyper-connection inject shape mismatch"));
    }
    let mut values = Vec::with_capacity(x.len());
    let mut bounds = Vec::with_capacity(x.len());
    for token in 0..tokens {
        for stream in 0..streams {
            let scale = 2.0 * sigmoid(inject_logits[token * streams + stream] / streams as f64);
            for col in 0..width {
                let residual = x[(token * streams + stream) * width + col];
                let product = scale * y[token * width + col];
                values.push(residual + product);
                bounds.push(residual.abs() + product.abs());
            }
        }
    }
    OracleOutput::new(values, bounds, 2)
}

#[derive(Clone, Debug, PartialEq)]
pub struct InjectNormOutput {
    /// Updated residual X, [tokens, streams, H].
    pub residual: OracleOutput,
    /// Grouped norm of the updated X, the next sublayer's `Xn`.
    pub normed: OracleOutput,
}

/// The fused step after a TP sublayer: y is the sum of the ranks' partials
/// (each [tokens, H]), injected into X, then normalized for the next wrapper.
pub fn all_reduce_inject_norm(
    query: SigmoidMeanHyperConnectionQuery,
    partials: &[Vec<f64>],
    x: &[f64],
    inject_logits: &[f64],
    next_norm_weight: &[f64],
) -> Result<InjectNormOutput, Refusal> {
    query.validate()?;
    let SigmoidMeanHyperConnectionQuery::AllReduceInjectNorm {
        ranks,
        streams,
        width,
        eps_bits,
        capacity,
    } = query
    else {
        return Err(Refusal("expected hyper-connection all-reduce query"));
    };
    let (s, h) = (streams as usize, width as usize);
    if partials.len() != ranks as usize || next_norm_weight.len() != s * h {
        return Err(Refusal(
            "hyper-connection all-reduce rank or weight mismatch",
        ));
    }
    let y = all_reduce(partials, None, None)?;
    let residual = inject_values(x, &y.values, inject_logits, s, h, capacity)?;
    let normed = grouped_norm_values(
        &residual.values,
        next_norm_weight,
        h,
        f64::from(f32::from_bits(eps_bits)),
    )?;
    Ok(InjectNormOutput { residual, normed })
}

/// Weights of one wrapper: norm [streams, H], down [rank, streams * H], up
/// [streams * H, rank] and inject [streams, streams * H] (absent for the
/// final mixer).
#[derive(Clone, Copy, Debug)]
pub struct HyperConnectionWeights<'a> {
    pub norm: &'a [f64],
    pub down: &'a [f64],
    pub up: &'a [f64],
    pub inject: Option<&'a [f64]>,
    pub streams: usize,
    pub width: usize,
    pub rank: usize,
    pub eps: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct HyperConnectionInput {
    /// The sublayer input, [tokens, H].
    pub mixed: Vec<f64>,
    /// Grouped norm of X, [tokens, streams, H].
    pub normed: Vec<f64>,
    /// Inject logits, [tokens, streams]; empty for the final mixer.
    pub inject_logits: Vec<f64>,
}

/// The whole wrapper before a sublayer (or the final mixer), composed from
/// the row ops and dense GEMMs above. Composite layer oracle: compare with
/// per-token cosine similarity.
pub fn hyper_connection_input(
    x: &[f64],
    weights: HyperConnectionWeights<'_>,
) -> Result<HyperConnectionInput, Refusal> {
    let HyperConnectionWeights {
        norm,
        down,
        up,
        inject,
        streams,
        width,
        rank,
        eps,
    } = weights;
    let wide = streams * width;
    if streams < 2
        || width == 0
        || rank == 0
        || x.is_empty()
        || !x.len().is_multiple_of(wide)
        || norm.len() != wide
        || down.len() != rank * wide
        || up.len() != wide * rank
        || inject.is_some_and(|w| w.len() != streams * wide)
    {
        return Err(Refusal("hyper-connection wrapper weight shape mismatch"));
    }
    let tokens = x.len() / wide;
    let normed = grouped_norm_values(x, norm, width, eps)?.values;
    let plain = GemmEpilogue {
        residual: None,
        swiglu: None,
    };
    let shape = |n, k| GemmShape {
        m: tokens,
        n,
        k,
        experts: 1,
    };
    let down_out = gemm(&normed, down, shape(rank, wide), &[], plain)?;
    let hidden = down_out
        .values
        .iter()
        .map(|&d| silu(d / streams as f64))
        .collect::<Vec<_>>();
    let up_out = gemm(&hidden, up, shape(wide, rank), &[], plain)?;
    let mut mixed = vec![0.0; tokens * width];
    for token in 0..tokens {
        for stream in 0..streams {
            for col in 0..width {
                let index = (token * streams + stream) * width + col;
                mixed[token * width + col] +=
                    sigmoid(up_out.values[index]) * normed[index] / streams as f64;
            }
        }
    }
    let inject_logits = match inject {
        Some(w) => gemm(&normed, w, shape(streams, wide), &[], plain)?.values,
        None => Vec::new(),
    };
    Ok(HyperConnectionInput {
        mixed,
        normed,
        inject_logits,
    })
}
