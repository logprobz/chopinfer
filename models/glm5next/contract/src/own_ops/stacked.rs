//! A KDA layer's input projections of its normed sublayer input in one
//! launch (`crate::sm120::stacked`, design row 8a): each part is a BF16 GEMM
//! of the same `[rows, k]` activation rows with its own `[rows, k]` weight
//! (padding rows included), and writes its first `width` output columns
//! into rows `row_stride` values apart, where its consumer reads them (q, k
//! and v their thirds of the convolution's input, beta its rank's columns
//! of its padded rows). The host binds each part's output view from its
//! first column.
//!
//! The reference is F64 over the BF16 operands; each output element may
//! differ from it by the numerical contract's bound (half a BF16 ulp plus
//! the F32 accumulation allowance, `oracle::numeric_bound`). A backend's
//! per-part bits are its business: SM120 keeps each part's own strip-chain
//! arithmetic, so its elements are bitwise the part's own entry's (shown by
//! kernel-lab's stacked family, not by this oracle).

use kernel_core::codec::bf16_to_f32;
use kernel_core::contract::{DType, Refusal, Validate};
use kernel_core::oracle::{
    CheckFailure, GemmEpilogue, GemmShape, OracleOutput, check_numeric, gemm,
};

/// One stacked GEMM: its weight rows and where its output columns go.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct StackedPart {
    /// Weight rows, padding included: the part's GEMM's N.
    pub rows: u32,
    /// Leading output columns written, at most `rows`.
    pub width: u32,
    /// Output values per token row, at least `width`.
    pub row_stride: u32,
}

impl StackedPart {
    /// A part whose contiguous `[tokens, rows]` output is written as is.
    pub fn contiguous(rows: u32) -> StackedPart {
        StackedPart {
            rows,
            width: rows,
            row_stride: rows,
        }
    }

    /// The output element of token row `row` and column `column`.
    pub fn at(&self, row: usize, column: usize) -> usize {
        row * self.row_stride as usize + column
    }

    /// The output values `rows` token rows span, from the first column of
    /// the first row through the last column of the last.
    pub fn span(&self, rows: u32) -> u64 {
        match rows {
            0 => 0,
            rows => u64::from(rows - 1) * u64::from(self.row_stride) + u64::from(self.width),
        }
    }
}

/// The stacked projections: `parts` GEMMs over one `k`-wide activation, in
/// part order. All math.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StackedProjection {
    pub k: u32,
    pub parts: Vec<StackedPart>,
}

impl Validate for StackedProjection {
    fn validate(&self) -> Result<(), Refusal> {
        if self.k == 0 || self.parts.is_empty() {
            return Err(Refusal("stacked projections need parts over a K"));
        }
        if self
            .parts
            .iter()
            .any(|p| p.width == 0 || p.width > p.rows || p.row_stride < p.width)
        {
            return Err(Refusal(
                "a stacked part writes 1 to its rows' columns within its row stride",
            ));
        }
        Ok(())
    }
}

kernel_core::scratch_record! {
    /// Stacked projections' buffers: the BF16 `[rows, k]` activation rows,
    /// each part's BF16 `[rows, k]` weight and output (from its first
    /// column), in part order, and the zeroed `u32` tickets of a split
    /// schedule. The activation and weights start at 128-byte alignment
    /// (tensor-map bases), the outputs at 2, the tickets at 4.
    #[derive(Clone, Debug, Eq, PartialEq)]
    pub struct StackedProjectionBuffers {
        pub x: value,
        pub weights: values,
        pub outputs: values,
        pub tickets: value,
    }
}

/// Each part's F64 reference over `rows` token rows, `[rows, width]` in row
/// order: the BF16 activation `x` times the part's weight rows (BF16, the
/// first `width` of `weights[p]`'s `rows`).
pub fn stacked_projection(
    op: &StackedProjection,
    x: &[u16],
    weights: &[&[u16]],
    rows: u32,
) -> Result<Vec<OracleOutput>, Refusal> {
    op.validate()?;
    if weights.len() != op.parts.len() {
        return Err(Refusal("stacked projections take one weight per part"));
    }
    if rows == 0 {
        return Err(Refusal("stacked projections need rows"));
    }
    let (k, m) = (op.k as usize, rows as usize);
    let widen = |bits: &[u16]| {
        bits.iter()
            .map(|&value| f64::from(bf16_to_f32(value)))
            .collect::<Vec<_>>()
    };
    let a = widen(
        x.get(..m * k)
            .ok_or(Refusal("the stacked activation is too small"))?,
    );
    op.parts
        .iter()
        .zip(weights)
        .map(|(part, weight)| {
            let n = part.width as usize;
            if weight.len() < part.rows as usize * k {
                return Err(Refusal("a stacked weight is too small"));
            }
            gemm(
                &a,
                &widen(&weight[..n * k]),
                GemmShape {
                    m,
                    n,
                    k,
                    experts: 1,
                },
                &[],
                GemmEpilogue {
                    residual: None,
                    swiglu: None,
                },
            )
        })
        .collect()
}

/// Checks part `part`'s BF16 `output` (from its first column) against its
/// reference within the numerical contract's bound.
pub fn check_part(
    part: &StackedPart,
    reference: &OracleOutput,
    output: &[u16],
    rows: u32,
) -> Result<(), CheckFailure> {
    let width = part.width as usize;
    let actual = (0..rows as usize * width)
        .map(|index| {
            output
                .get(part.at(index / width, index % width))
                .map_or(f64::NAN, |&bits| f64::from(bf16_to_f32(bits)))
        })
        .collect::<Vec<_>>();
    check_numeric(&actual, reference, DType::Bf16)
}

#[cfg(test)]
mod tests {
    use super::*;
    use kernel_core::codec::f32_to_bf16;

    /// Each part's reference is its own GEMM of its leading weight rows,
    /// checked where its row stride places it.
    #[test]
    fn parts_project_the_shared_rows_into_their_places() {
        let op = StackedProjection {
            k: 2,
            parts: vec![
                StackedPart::contiguous(2),
                StackedPart {
                    rows: 4,
                    width: 1,
                    row_stride: 3,
                },
            ],
        };
        let bf = |values: &[f32]| values.iter().map(|&v| f32_to_bf16(v)).collect::<Vec<_>>();
        let x = bf(&[1.0, 2.0, 3.0, 4.0]);
        let first = bf(&[1.0, 0.0, 0.0, 1.0]);
        let second = bf(&[1.0, 1.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0]);
        let references = stacked_projection(&op, &x, &[&first, &second], 2).unwrap();
        assert_eq!(references[0].values, [1.0, 2.0, 3.0, 4.0]);
        assert_eq!(references[1].values, [3.0, 7.0]);
        assert!(check_part(&op.parts[0], &references[0], &bf(&[1.0, 2.0, 3.0, 4.0]), 2).is_ok());
        let placed = bf(&[3.0, -1.0, -1.0, 7.0]);
        assert!(check_part(&op.parts[1], &references[1], &placed, 2).is_ok());
        assert!(check_part(&op.parts[1], &references[1], &placed[..3], 2).is_err());
        assert!(check_part(&op.parts[1], &references[1], &bf(&[3.0, 7.0, 0.0, 0.0]), 2).is_err());
        assert_eq!(op.parts[1].span(2), 4);
        // A short weight or a missing part is refused.
        assert!(stacked_projection(&op, &x, &[&first, &second[..7]], 2).is_err());
        assert!(stacked_projection(&op, &x, &[&first], 2).is_err());
        assert!(
            StackedProjection {
                k: 2,
                parts: vec![StackedPart {
                    rows: 2,
                    width: 3,
                    row_stride: 3
                }]
            }
            .validate()
            .is_err()
        );
    }
}
