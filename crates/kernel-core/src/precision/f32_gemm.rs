//! Unquantized F32 projections with at most 128 K in each F32 chain.
//!
//! BF16 weights widen exactly. A tensor implementation decomposes each F32
//! activation into three exact BF16 terms, keeps a separate chain per term,
//! and merges hi, mid, lo in that order before the outer sum in increasing K.
//! Finite values outside the tensor domain use SIMT F32 FMA, never BF16 or
//! TF32 rounding. Raw F32 weights use that SIMT path. Outputs remain F32.

use crate::codec::{bf16_to_f32, f32_to_bf16};
use crate::contract::{ActivationFormat, DType, GemmQuery, Refusal, Validate, WeightFormat};
use crate::oracle::CheckFailure;

pub const FOLD_K: usize = 128;

/// F32 round-to-nearest unit roundoff and half the least subnormal. The
/// contract requires gradual underflow: no FTZ flag or instruction modifier.
pub const UNIT_ROUNDOFF: f64 = 1.0 / 16_777_216.0;
pub const UNDERFLOW_ETA: f64 = f64::from_bits((1023 - 150) << 52);

// An upward-rounded (1+u)^n, valid even when n*u >= 1 and gamma_n has no
// finite bound. All multiplication operands are positive and at least one.
fn growth(mut n: usize) -> f64 {
    let mut result = 1.0;
    let mut factor = 1.0 + UNIT_ROUNDOFF;
    while n > 0 {
        if n & 1 != 0 {
            result = (result * factor).next_up();
        }
        n >>= 1;
        if n > 0 {
            factor = (factor * factor).next_up();
        }
    }
    result
}

/// Let t=min(K,128), B=ceil(K/128), u=2^-24 and eta=2^-150. Each rounding
/// obeys fl(z)=z*(1+delta)+epsilon, |delta|<=u, |epsilon|<=eta, provided no
/// intermediate overflows and subnormals are retained.
///
/// The tensor path has at most (t-1)+2+(B-1)=t+B roundings on a product's
/// path: a component sum (its first product is exact), two component merges,
/// then the outer sum (its first partial is exact). Same-sign decomposition
/// preserves S. Its at most 3*(K-B)+2*B+(B-1)=3*K-1 rounding sites each
/// contribute an absolute error amplified by at most the same growth factor.
/// SIMT has t+(B-1) relative roundings and K+(B-1) absolute sites, both
/// bounded by those counts. Thus G=(1+u)^(t+B) gives the common allowance
/// (G-1)*S + (3*K-1)*eta*G. This is the product-of-roundings derivation;
/// replacing each (1+u)^n by 1+gamma_n is also valid when n*u<1.
///
/// The arithmetic computing this allowance rounds upward. Invalid K or S
/// yields NaN, which the dedicated gate refuses. The generic GEMM gate is
/// not this contract, because it has no absolute underflow term.
pub fn bound(k: usize, sum_abs_products: f64) -> f64 {
    if k == 0 || k > u32::MAX as usize || !sum_abs_products.is_finite() || sum_abs_products < 0.0 {
        return f64::NAN;
    }
    let blocks = k.div_ceil(FOLD_K);
    let g = growth(k.min(FOLD_K) + blocks);
    let relative = ((g - 1.0).next_up() * sum_abs_products).next_up();
    let absolute = (((3.0 * k as f64 - 1.0) * UNDERFLOW_ETA).next_up() * g).next_up();
    (relative + absolute).next_up()
}

/// A dedicated result type prevents using the relative-only generic GEMM
/// gate for this contract. K is the full logical reduction length.
#[derive(Clone, Debug, PartialEq)]
pub struct Oracle {
    pub values: Vec<f64>,
    pub sum_abs_products: Vec<f64>,
    pub k: usize,
}

pub fn check_numeric(actual: &[f64], reference: &Oracle) -> Result<(), CheckFailure> {
    if actual.len() != reference.values.len()
        || reference.values.len() != reference.sum_abs_products.len()
        || reference.values.is_empty()
    {
        return Err(CheckFailure {
            index: actual.len().min(reference.values.len()),
            actual: f64::NAN,
            reference: f64::NAN,
            bound: 0.0,
            error: f64::INFINITY,
        });
    }
    let mut worst: Option<CheckFailure> = None;
    for (index, (&actual, (&expected, &s))) in actual
        .iter()
        .zip(reference.values.iter().zip(&reference.sum_abs_products))
        .enumerate()
    {
        let allowance = bound(reference.k, s);
        let error = (actual - expected).abs();
        if !actual.is_finite()
            || !expected.is_finite()
            || !allowance.is_finite()
            || error > allowance
        {
            let failure = CheckFailure {
                index,
                actual,
                reference: expected,
                bound: allowance,
                error: if error.is_finite() {
                    error
                } else {
                    f64::INFINITY
                },
            };
            if worst.as_ref().is_none_or(|w| {
                !allowance.is_finite() || failure.error - allowance > w.error - w.bound
            }) {
                worst = Some(failure);
            }
        }
    }
    worst.map_or(Ok(()), Err)
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

/// F64 compensated dots over original operands, retaining logical K for the
/// outer-reduction and underflow allowances in this module's `check_numeric`.
pub fn oracle(query: GemmQuery, rows: usize, a: &[f32], b: &[f32]) -> Result<Oracle, Refusal> {
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
    Ok(Oracle {
        values,
        sum_abs_products: magnitudes,
        k,
    })
}
