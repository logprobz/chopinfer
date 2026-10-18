//! Status and f64 value reference for plain F32 RMSNorm.
use crate::codec::bf16_to_f32;
use crate::contract::{DType, Refusal, RmsNormKind, Validate};
use crate::oracle::{OracleOutput, rmsnorm};
use crate::precision::quantizer::{NONFINITE_INPUT, UNREPRESENTABLE_OUTPUT};
use crate::rmsnorm::RmsNormRequirement;

// Each lane owns eight adjacent values, then advances by 2048 columns.
// FMA accumulation and both warp-down trees match the F32 norm kernel.
fn square_sum(input: &[f32]) -> f32 {
    let mut lanes = [0.0_f32; 256];
    for (column, &value) in input.iter().enumerate() {
        let lane = &mut lanes[(column / 8) % 256];
        *lane = value.mul_add(value, *lane);
    }
    let reduce = |values: &mut [f32]| {
        for distance in [16, 8, 4, 2, 1] {
            for lane in 0..distance {
                values[lane] += values[lane + distance];
            }
        }
        values[0]
    };
    let mut warps = [0.0_f32; 32];
    for (warp, values) in lanes.chunks_exact_mut(32).enumerate() {
        warps[warp] = reduce(values);
    }
    reduce(&mut warps)
}

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
        // Current input errors take precedence over arithmetic errors in this
        // kernel. Prior status is combined only after all current checks.
        let mut current = 0;
        let result =
            if input.iter().any(|v| !v.is_finite()) || weight.iter().any(|v| !v.is_finite()) {
                current = NONFINITE_INPUT;
                None
            } else if !square_sum(input).is_finite() {
                current = UNREPRESENTABLE_OUTPUT;
                None
            } else {
                let result = rmsnorm(
                    &input.iter().map(|&v| f64::from(v)).collect::<Vec<_>>(),
                    width,
                    &weight,
                    f64::from(f32::from_bits(q.epsilon_bits)),
                    q.kind,
                    None,
                )?;
                if result.values.iter().any(|&v| !(v as f32).is_finite()) {
                    current = UNREPRESENTABLE_OUTPUT;
                }
                Some(result)
            };
        status[row] = prior[row] | current;
        if status[row] != 0 {
            values.extend(vec![0.0; width]);
            magnitudes.extend(vec![0.0; width]);
        } else {
            let result = result.expect("a successful row has its f64 reference");
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

#[cfg(test)]
mod status_boundary_tests {
    use super::*;
    fn query() -> RmsNormRequirement {
        RmsNormRequirement {
            input: DType::F32,
            output: DType::F32,
            ..RmsNormRequirement::bf16_backend_chosen(512, 1e-20, RmsNormKind::Plain, false)
        }
    }
    #[test]
    fn status_boundary_keeps_a_finite_fused_reduction() {
        let mut input = vec![0.0; 512];
        input[0] = f32::from_bits(0x5f7ffff3);
        input[1] = f32::from_bits(0x5a9ffffe);
        assert_eq!(
            input[1].mul_add(input[1], input[0] * input[0]).to_bits(),
            0x7f7fffff
        );
        assert_eq!(square_sum(&input).to_bits(), 0x7f7fffff);
        assert!(
            input
                .iter()
                .map(|&x| f64::from(x) * f64::from(x))
                .sum::<f64>()
                > f64::from(f32::MAX)
        );
        let (output, status) = oracle(query(), &input, &vec![0x3f80; 512], &[0]).unwrap();
        assert_eq!(status, [0]);
        assert!(output.values[0] > 20.0);
    }
    #[test]
    fn status_boundary_prior_input_flag_does_not_hide_current_reduction_overflow() {
        let mut input = vec![0.0; 512];
        input[0] = f32::MAX;
        let (output, status) = oracle(query(), &input, &vec![0x3f80; 512], &[1]).unwrap();
        assert_eq!(status, [5]);
        assert!(output.values.iter().all(|v| v.to_bits() == 0));
    }
    #[test]
    fn status_boundary_prior_flag_does_not_hide_current_output_overflow() {
        let mut input = vec![0.0; 512];
        input[0] = 1.0;
        let (output, status) = oracle(query(), &input, &vec![0x7f7f; 512], &[1]).unwrap();
        assert_eq!(status, [5]);
        assert!(output.values.iter().all(|v| v.to_bits() == 0));
    }
}
