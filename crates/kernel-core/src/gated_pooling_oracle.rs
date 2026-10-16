//! F64 reference arithmetic for gated group pooling (`gated_pooling`).

use crate::contract::{Refusal, Validate};
use crate::gated_pooling::GatedGroupPooling;
use crate::oracle::OracleOutput;

/// Pools one complete group. `keys` and `gates` are `[group][width]`, the
/// decoded BF16 values of the group's normalized keys and gate logits, and
/// `position` the decoded `[group][width]` position embedding. Values are
/// the f64 pooled key before its single BF16 rounding; S of channel d is
/// `sum_t |w_t * key_t[d]|` and K is the group.
pub fn pool(
    pooling: GatedGroupPooling,
    keys: &[f64],
    gates: &[f64],
    position: &[f64],
) -> Result<OracleOutput, Refusal> {
    pooling.validate()?;
    let (group, width) = (pooling.group as usize, pooling.width as usize);
    if [keys, gates, position]
        .iter()
        .any(|operand| operand.len() != group * width)
    {
        return Err(Refusal("gated pooling operand shape mismatch"));
    }
    let mut values = Vec::with_capacity(width);
    let mut s = Vec::with_capacity(width);
    for d in 0..width {
        let logit = |t: usize| gates[t * width + d] + position[t * width + d];
        let max = (0..group).map(logit).fold(f64::NEG_INFINITY, f64::max);
        let e = (0..group)
            .map(|t| (logit(t) - max).exp())
            .collect::<Vec<_>>();
        let total = e.iter().sum::<f64>();
        let products = (0..group)
            .map(|t| e[t] / total * keys[t * width + d])
            .collect::<Vec<_>>();
        values.push(products.iter().sum());
        s.push(products.iter().map(|p| p.abs()).sum());
    }
    OracleOutput::new(values, s, group)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::{bf16_to_f32, f32_to_bf16};
    use crate::contract::DType;
    use crate::oracle::bf16_round;

    const POOLING: GatedGroupPooling = GatedGroupPooling {
        group: 4,
        width: 8,
        position_dtype: DType::Bf16,
    };

    fn splat(rows: [f64; 4]) -> Vec<f64> {
        rows.iter().flat_map(|&v| [v; 8]).collect()
    }

    #[test]
    fn equal_logits_average_the_group() {
        let keys = splat([1.0, 2.0, 3.0, 6.0]);
        let out = pool(POOLING, &keys, &splat([0.5; 4]), &splat([0.0; 4])).unwrap();
        assert!(out.values.iter().all(|&v| v == 3.0));
        assert!(out.sum_abs_products.iter().all(|&s| s == 3.0));
        assert_eq!(out.reduction_lengths, vec![4; 8]);
    }

    #[test]
    fn position_embedding_joins_the_gate_logits() {
        // Logits ln 1, ln 3, -inf-like, -inf-like: weights 1/4 and 3/4.
        let gates = splat([0.0, 3_f64.ln(), -1000.0, 0.0]);
        let position = splat([0.0, 0.0, 0.0, -1000.0]);
        let keys = splat([4.0, 8.0, 100.0, 100.0]);
        let out = pool(POOLING, &keys, &gates, &position).unwrap();
        for v in out.values {
            assert!((v - 7.0).abs() < 1e-12, "{v}");
        }
    }

    /// The pooled key rounds once. Rounding the weights and each product
    /// to BF16 first (the HF sequence) lands on another BF16 value.
    #[test]
    fn pooled_key_rounds_once_unlike_bf16_weights_and_products() {
        // Weights 1/3, 1/3, 1/3 and 0 (the fourth logit far below).
        let gates = splat([0.0, 0.0, 0.0, -1000.0]);
        let keys = [1.0, 1.0, 3.0, 1.0];
        let out = pool(POOLING, &splat(keys), &gates, &splat([0.0; 4])).unwrap();
        assert!((out.values[0] - 5.0 / 3.0).abs() < 1e-15);
        assert_eq!(bf16_round(out.values[0]), 1.6640625);
        // HF: bf16(1/3) = 0.333984375, each product rounds to BF16, then
        // the F32 sum rounds once more.
        let weight = bf16_round(1.0 / 3.0);
        assert_eq!(weight, 0.333984375);
        let product = |key: f64| f64::from(bf16_to_f32(f32_to_bf16((weight * key) as f32)));
        let hf = bf16_round(keys[..3].iter().map(|&k| product(k)).sum::<f64>());
        assert_eq!(hf, 1.671875);
    }

    #[test]
    fn refuses_mismatched_operands() {
        let short = vec![0.0; 31];
        let full = vec![0.0; 32];
        assert_eq!(
            pool(POOLING, &short, &full, &full),
            Err(Refusal("gated pooling operand shape mismatch"))
        );
    }
}
