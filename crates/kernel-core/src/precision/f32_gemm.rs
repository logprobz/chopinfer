//! Unquantized F32 projections with at most 128 K in each F32 chain.
//!
//! BF16 weights widen exactly. A tensor implementation decomposes each F32
//! activation into three exact BF16 terms, keeps a separate chain per term,
//! and merges hi, mid, lo in that order before the outer sum in increasing K.
//! Finite values outside the tensor domain use SIMT F32 FMA, never BF16 or
//! TF32 rounding. Raw F32 weights use that SIMT path. Outputs remain F32.

use crate::codec::{bf16_to_f32, f32_to_bf16};
use crate::contract::{ActivationFormat, DType, GemmQuery, Refusal, Validate, WeightFormat};
use crate::oracle::OracleOutput;

pub const FOLD_K: usize = 128;

/// The numerical gate uses S from the original F32 and exactly widened
/// weight operands, not from rounded inputs or the number of component MMAs.
pub fn bound(sum_abs_products: f64) -> f64 {
    crate::oracle::accumulation_bound(FOLD_K, sum_abs_products)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DotPath {
    Bf16Expansion3,
    SimtF32,
}

/// Truncate toward zero to produce components with the input's sign. Each
/// subtraction is exact. Nonzero components must be normal BF16 and their
/// sum must reconstruct the input exactly, including signed zero. This
/// excludes values whose low bits need BF16 subnormals or smaller numbers.
/// `None` is a SIMT requirement, not permission to discard the residual.
pub fn decompose(value: f32) -> Result<Option<[u16; 3]>, Refusal> {
    if !value.is_finite() {
        return Err(Refusal("F32 GEMM requires finite activations"));
    }
    let mut remainder = value;
    let mut terms = [0; 3];
    for term in &mut terms {
        if remainder == 0.0 {
            *term = (value.to_bits() >> 16) as u16 & 0x8000;
            continue;
        }
        *term = (remainder.to_bits() >> 16) as u16;
        let part = bf16_to_f32(*term);
        if part != 0.0 && !part.is_normal() {
            return Ok(None);
        }
        remainder -= part;
    }
    let parts = terms.map(|term| f64::from(bf16_to_f32(term)));
    if remainder != 0.0
        || ((parts[0] + parts[1]) + parts[2]).to_bits() != f64::from(value).to_bits()
    {
        return Ok(None);
    }
    Ok(Some(terms))
}

fn operands(a: &[f32], b: &[f32], weight: DType) -> Result<(), Refusal> {
    if a.is_empty()
        || a.len() != b.len()
        || !matches!(weight, DType::Bf16 | DType::F32)
        || a.iter().chain(b).any(|v| !v.is_finite())
    {
        return Err(Refusal(
            "F32 dot requires equal nonempty finite F32 inputs and BF16 or F32 weights",
        ));
    }
    if weight == DType::Bf16
        && b.iter()
            .any(|&v| bf16_to_f32(f32_to_bf16(v)).to_bits() != v.to_bits())
    {
        return Err(Refusal(
            "BF16 weights must widen exactly to their supplied F32 values",
        ));
    }
    Ok(())
}

/// A BF16 tensor operand and every component product must be zero or normal
/// and finite, with the product exactly representable in F32. This guard
/// excludes operand flushing, underflow, overflow and lost product bits.
/// An engine may choose SIMT for more inputs; it may not admit products
/// that fail these conditions while claiming exact component products.
pub fn dot_path(a: &[f32], b: &[f32], weight: DType) -> Result<DotPath, Refusal> {
    operands(a, b, weight)?;
    if weight == DType::F32 {
        return Ok(DotPath::SimtF32);
    }
    for (&value, &weight) in a.iter().zip(b) {
        let Some(terms) = decompose(value)? else {
            return Ok(DotPath::SimtF32);
        };
        if weight != 0.0 && !weight.is_normal() {
            return Ok(DotPath::SimtF32);
        }
        for term in terms {
            let product = f64::from(bf16_to_f32(term)) * f64::from(weight);
            let rounded = product as f32;
            if !rounded.is_finite()
                || (product != 0.0 && !rounded.is_normal())
                || f64::from(rounded).to_bits() != product.to_bits()
            {
                return Ok(DotPath::SimtF32);
            }
        }
    }
    Ok(DotPath::Bf16Expansion3)
}

/// CPU witness of the required fold and component merge order. MMA's
/// internal reduction order may differ and must pass the same f64 bound.
pub fn folded_dot(a: &[f32], b: &[f32], weight: DType) -> Result<f32, Refusal> {
    let path = dot_path(a, b, weight)?;
    let mut total = 0.0_f32;
    for (a, b) in a.chunks(FOLD_K).zip(b.chunks(FOLD_K)) {
        let partial = match path {
            DotPath::SimtF32 => a
                .iter()
                .zip(b)
                .fold(0.0_f32, |sum, (&a, &b)| a.mul_add(b, sum)),
            DotPath::Bf16Expansion3 => {
                let mut partials = [0.0_f32; 3];
                for (&a, &b) in a.iter().zip(b) {
                    let terms = decompose(a)?.expect("admitted exact BF16 expansion");
                    for (sum, term) in partials.iter_mut().zip(terms) {
                        *sum = bf16_to_f32(term).mul_add(b, *sum);
                    }
                }
                (partials[0] + partials[1]) + partials[2]
            }
        };
        total += partial;
    }
    if !total.is_finite() {
        return Err(Refusal("F32 GEMM accumulation is not finite"));
    }
    Ok(total)
}

/// F64 compensated dots over original operands. The numerical reference's
/// reduction length is the 128-K chain limit, so `check_numeric` enforces
/// exactly 2^-22 * 128 * S with no BF16 output-rounding allowance.
pub fn oracle(
    query: GemmQuery,
    rows: usize,
    a: &[f32],
    b: &[f32],
) -> Result<OracleOutput, Refusal> {
    query.validate()?;
    let (n, k) = (query.n as usize, query.k as usize);
    if query.activation != ActivationFormat::F32
        || rows == 0
        || rows > query.m_capacity as usize
        || rows.checked_mul(k) != Some(a.len())
        || n.checked_mul(k) != Some(b.len())
    {
        return Err(Refusal("F32 GEMM oracle operands disagree with the query"));
    }
    let weight = if query.weight == WeightFormat::Bf16 {
        DType::Bf16
    } else {
        DType::F32
    };
    if a.iter().chain(b).any(|v| !v.is_finite()) {
        return Err(Refusal("F32 GEMM oracle requires finite inputs"));
    }
    if weight == DType::Bf16
        && b.iter()
            .any(|&v| bf16_to_f32(f32_to_bf16(v)).to_bits() != v.to_bits())
    {
        return Err(Refusal(
            "BF16 weights must widen exactly to their supplied F32 values",
        ));
    }
    let a: Vec<_> = a.iter().copied().map(f64::from).collect();
    let b: Vec<_> = b.iter().copied().map(f64::from).collect();
    let mut values = Vec::new();
    let mut magnitudes = Vec::new();
    for row in a.chunks_exact(k) {
        for col in b.chunks_exact(k) {
            let (value, magnitude) = crate::gated_mlp_oracle::compensated_dot(row, col)?;
            values.push(value);
            magnitudes.push(magnitude);
        }
    }
    OracleOutput::new(values, magnitudes, FOLD_K)
}
