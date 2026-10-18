//! Status and f64 value reference for plain F32 RMSNorm.
use crate::codec::bf16_to_f32;
use crate::contract::{DType, Refusal, RmsNormKind, Validate};
use crate::oracle::{OracleOutput, rmsnorm};
use crate::precision::quantizer::{NONFINITE_INPUT, UNREPRESENTABLE_OUTPUT};
use crate::rmsnorm::RmsNormRequirement;

pub fn oracle(
    q: RmsNormRequirement,
    x: &[f32],
    weight: &[u16],
    prior: &[u32],
) -> Result<(OracleOutput, Vec<u32>), Refusal> {
    q.validate()?;
    let width = q.width as usize;
    if q.input != DType::F32
        || q.output != DType::F32
        || q.kind != RmsNormKind::Plain
        || q.fused_residual
        || x.len() != prior.len() * width
        || weight.len() != width
    {
        return Err(Refusal("F32 norm status operand mismatch"));
    }
    let weight = weight
        .iter()
        .map(|&v| f64::from(bf16_to_f32(v)))
        .collect::<Vec<_>>();
    let mut status = prior.to_vec();
    let mut values = Vec::new();
    let mut magnitudes = Vec::new();
    for (row, input) in x.chunks_exact(width).enumerate() {
        if input.iter().any(|v| !v.is_finite()) || weight.iter().any(|v| !v.is_finite()) {
            status[row] |= NONFINITE_INPUT;
        }
        // Check the finite F32 reduction domain before asking the f64 oracle
        // for its value and numerical bound.
        let square = input
            .iter()
            .map(|&v| f64::from(v) * f64::from(v))
            .sum::<f64>();
        if status[row] & NONFINITE_INPUT == 0 && square > f64::from(f32::MAX) {
            status[row] |= UNREPRESENTABLE_OUTPUT;
        }
        if status[row] != 0 {
            values.extend(vec![0.0; width]);
            magnitudes.extend(vec![0.0; width]);
            continue;
        }
        let result = rmsnorm(
            &input.iter().map(|&v| f64::from(v)).collect::<Vec<_>>(),
            width,
            &weight,
            f64::from(f32::from_bits(q.epsilon_bits)),
            q.kind,
            None,
        )?;
        if result.values.iter().any(|&v| !(v as f32).is_finite()) {
            status[row] |= UNREPRESENTABLE_OUTPUT;
            values.extend(vec![0.0; width]);
            magnitudes.extend(vec![0.0; width]);
        } else {
            values.extend(result.values);
            magnitudes.extend(result.sum_abs_products);
        }
    }
    Ok((OracleOutput::new(values, magnitudes, width)?, status))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn nonfinite_norm_inputs_keep_prior_flags_and_zero_the_failed_rows() {
        let q = RmsNormRequirement {
            input: DType::F32,
            output: DType::F32,
            ..RmsNormRequirement::bf16_backend_chosen(512, 1e-20, RmsNormKind::Plain, false)
        };
        let mut x = vec![1.0; 4 * 512];
        x[0] = f32::NAN;
        x[512] = f32::MAX;
        let (out, flags) = oracle(q, &x, &vec![0x3f80; 512], &[0x80, 0, 2, 0]).unwrap();
        assert_eq!(flags, [0x81, 4, 2, 0]);
        assert!(out.values[..3 * 512].iter().all(|v| v.to_bits() == 0));
        assert!(out.values[3 * 512..].iter().all(|&v| v == 1.0));
    }
}
