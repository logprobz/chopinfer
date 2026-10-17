//! F64 partial-chain and epilogue oracles for quantized routed experts.

use crate::codec::{SmallFloat, bf16_to_f32, f32_to_bf16};
use crate::contract::{Refusal, Validate};
use crate::gated_mlp::SwigluClamp;
use crate::moe_chained::ChainedK;
use crate::oracle::{OracleOutput, compensated_sum_with_magnitude};
use crate::quantize::{AmaxRatio, Quantized, Quantizer, ScaleRule, quantize};

/// One f64 value and S per 128-K chain. The last chain includes exact
/// zeros through loop K. Operands contain exactly stored K decoded values,
/// so there are no readable neighboring rows or scales in the oracle.
/// F32 MMA results are bounded against these values; f64 does not claim
/// to emulate a GPU MMA instruction's internal reduction order.
pub fn partial_chains(x: &[f64], weight: &[f64], k: ChainedK) -> Result<OracleOutput, Refusal> {
    k.validate()?;
    if x.len() != k.stored as usize
        || weight.len() != x.len()
        || x.iter().chain(weight).any(|v| !v.is_finite())
    {
        return Err(Refusal(
            "partial chains need finite operands of exactly stored K",
        ));
    }
    let mut values = Vec::new();
    let mut magnitudes = Vec::new();
    for start in (0..k.loop_k as usize).step_by(128) {
        let (value, magnitude) = compensated_sum_with_magnitude((start..start + 128).map(|i| {
            if i < k.stored as usize {
                x[i] * weight[i]
            } else {
                0.0
            }
        }));
        if !value.is_finite() || !magnitude.is_finite() {
            return Err(Refusal("partial chain arithmetic left the finite range"));
        }
        values.push(value);
        magnitudes.push(magnitude);
    }
    OracleOutput::new(values, magnitudes, 128)
}

/// Fold supplied F32 chain results, starting at +0 and adding in K order.
/// This checks the fold bitwise independently of the MMA accuracy gate.
pub fn fold_f32_chains(partials: &[f32]) -> Result<f32, Refusal> {
    if partials.is_empty() || partials.iter().any(|v| !v.is_finite()) {
        return Err(Refusal("chain fold needs finite nonempty partials"));
    }
    let value = partials.iter().fold(0.0_f32, |sum, &v| sum + v);
    if !value.is_finite() {
        return Err(Refusal("chain fold overflows F32"));
    }
    Ok(value)
}

/// F64 continuous epilogue over the exact inputs, before BF16 rounding.
/// It preserves the upper-only gate clamp and symmetric up clamp.
pub fn before_down_f64(
    gate: f64,
    up: f64,
    route_weight: f64,
    clamp: Option<SwigluClamp>,
) -> Result<f64, Refusal> {
    if [gate, up, route_weight].iter().any(|v| !v.is_finite()) {
        return Err(Refusal("before-down epilogue needs finite inputs"));
    }
    let (gate, up) = match clamp {
        Some(c) => {
            if !c.gate_max().is_finite()
                || c.gate_max() <= 0.0
                || !c.up_limit().is_finite()
                || c.up_limit() <= 0.0
            {
                return Err(Refusal("SwiGLU clamp limits must be finite and positive"));
            }
            (
                gate.min(f64::from(c.gate_max())),
                up.clamp(-f64::from(c.up_limit()), f64::from(c.up_limit())),
            )
        }
        None => (gate, up),
    };
    let silu = gate / (1.0 + (-gate).exp());
    let value = route_weight * (silu * up);
    if !value.is_finite() {
        return Err(Refusal("before-down epilogue left the finite range"));
    }
    Ok(value)
}

/// The two F32 products after a kernel's clamped SiLU, in the required
/// order. Keeping SiLU as an input separates transcendental error from the
/// exact association and subsequent quantization gates.
pub fn weight_before_down(silu: f32, up: f32, route_weight: f32) -> f32 {
    route_weight * (silu * up)
}

/// Round every epilogue value to BF16 before amax, then apply E4M3/32
/// with ceil UE8M0 scales. Exact scale/code oracle over supplied F32
/// epilogue outputs; no continuous-math error is included here.
pub fn quantize_before_down(
    values: &[f32],
    rows: usize,
    width: usize,
) -> Result<Quantized, Refusal> {
    if !width.is_multiple_of(32) {
        return Err(Refusal(
            "chained intermediate width needs whole groups of 32",
        ));
    }
    let values = values
        .iter()
        .map(|&v| bf16_to_f32(f32_to_bf16(v)))
        .collect::<Vec<_>>();
    quantize(
        &values,
        rows,
        width,
        Quantizer {
            rule: ScaleRule::CeilPowerOfTwo(AmaxRatio::MultiplyReciprocal),
            amax_floor: Some(1e-4),
            ..Quantizer::mx(SmallFloat::E4M3)
        },
    )
}
