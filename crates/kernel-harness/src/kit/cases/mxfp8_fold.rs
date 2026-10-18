//! Native MXFP8 folds over row-major codes and UE8M0 scales. Weight rows
//! reuse eight code patterns with non-periodic row-scale shifts. Exact power
//! of two scaling expands their f64 dots to every production output row.

use crate::LabResult;
use crate::conformance::{Expected, Images, Rows, Write};
use crate::kit::{OpCase, Required, Spec};
use kernel_core::backend::{Need, Prepared, View};
use kernel_core::codec::SmallFloat;
use kernel_core::contract::DType;
use kernel_core::mxfp8_gemm::{self, Mxfp8FoldGemmBuffers, Mxfp8FoldGemmQuery};
use kernel_core::oracle::OracleOutput;
use kernel_core::properties::Invariance;
use kernel_core::quantize::{QuantFormat, Quantized, ScaleFormat, Scales};
use kernels_sm120::backend::Sm120;
use kernels_sm120::gemm::mxfp8_fold;

fn matrix(rows: usize, k: usize, codes: Vec<u8>, scales: Vec<u8>) -> Quantized {
    Quantized {
        rows,
        k,
        codes,
        scales: Scales::Ue8M0(scales),
        global_scale: 1.0,
        format: QuantFormat {
            element: SmallFloat::E4M3,
            block: (1, 32),
            scale: ScaleFormat::Ue8M0,
            chunk_rows: None,
        },
    }
}

fn witness(q: Mxfp8FoldGemmQuery) -> bool {
    q.k == 65536 && q.n == 1 && q.groups == 1
}

fn scale_witness(q: Mxfp8FoldGemmQuery) -> bool {
    q.n == 65 && q.k == 160 && q.groups == 1
}

fn row_scale_shift(q: Mxfp8FoldGemmQuery, row: usize) -> i32 {
    if witness(q) {
        return 0;
    }
    if scale_witness(q) && row == 64 {
        return 1;
    }
    let band = row / 8;
    ((band ^ (band >> 2) ^ (row / 64)) % 5) as i32
}

fn activation(q: Mxfp8FoldGemmQuery, rows: u32) -> Quantized {
    let width = q.activation_row_stride as usize;
    let codes = (0..rows as usize * width)
        .map(|i| {
            let row = i / width;
            let col = i % width;
            if scale_witness(q) {
                0x38
            } else if witness(q) {
                if col == 0 {
                    1
                } else if col >= 128 && col % 32 == 0 {
                    0x38
                } else {
                    0
                }
            } else {
                ((row * 13 + col * 7) % 127) as u8 | if (row + col) % 3 == 0 { 128 } else { 0 }
            }
        })
        .collect();
    let scales = (0..rows as usize * width / 32)
        .map(|i| {
            if scale_witness(q) {
                127
            } else if witness(q) {
                if i % (width / 32) == 0 { 161 } else { 127 }
            } else {
                120 + ((i / (width / 32) * 7 + i % (width / 32) * 3) % 11) as u8
            }
        })
        .collect();
    matrix(rows as usize, width, codes, scales)
}

fn weight(q: Mxfp8FoldGemmQuery, compact: bool) -> Quantized {
    let n = if compact { q.n.min(8) } else { q.n } as usize;
    let rows = q.groups as usize * n;
    let k = q.k as usize;
    let pattern = |row: usize| ((row / n) * 3 + row % n) % 8;
    let codes = (0..rows * k)
        .map(|i| {
            if witness(q) || scale_witness(q) {
                0x38
            } else {
                let row = pattern(i / k);
                let col = i % k;
                ((row * 17 + col * 7) % 127) as u8 | if (row + col) % 3 == 0 { 128 } else { 0 }
            }
        })
        .collect();
    let scales = (0..rows * (k / 32))
        .map(|i| {
            let base = if witness(q) || scale_witness(q) {
                127
            } else {
                121 + ((pattern(i / (k / 32)) * 3 + (i % (k / 32)) * 5) % 13) as u8
            };
            base + if compact {
                0
            } else {
                row_scale_shift(q, i / (k / 32)) as u8
            }
        })
        .collect();
    matrix(rows, k, codes, scales)
}

fn reference(q: Mxfp8FoldGemmQuery, images: &Images, w: &Quantized) -> LabResult<OracleOutput> {
    let width = q.activation_row_stride as usize;
    let codes = images.bytes("activations")?.to_vec();
    let rows = codes.len() / width;
    if rows == 0 {
        return OracleOutput::new(vec![], vec![], q.k as usize).map_err(|e| e.to_string());
    }
    let a = matrix(
        rows,
        width,
        codes,
        images.bytes("activation_scales")?.to_vec(),
    );
    let small = q.n.min(8) as usize;
    let narrow = Mxfp8FoldGemmQuery {
        n: small as u32,
        ..q
    };
    let compact = mxfp8_gemm::oracle(narrow, rows, &a, w).map_err(|e| e.to_string())?;
    let mut values = Vec::with_capacity(rows * q.groups as usize * q.n as usize);
    let mut magnitudes = Vec::with_capacity(values.capacity());
    for row in 0..rows * q.groups as usize {
        for col in 0..q.n as usize {
            let weight_row = (row % q.groups as usize) * q.n as usize + col;
            let factor = 2.0_f64.powi(row_scale_shift(q, weight_row));
            values.push(compact.values[row * small + col % small] * factor);
            magnitudes.push(compact.sum_abs_products[row * small + col % small] * factor);
        }
    }
    OracleOutput::new(values, magnitudes, q.k as usize).map_err(|e| e.to_string())
}

fn expected(q: Mxfp8FoldGemmQuery, reference: OracleOutput) -> Expected {
    if witness(q) || scale_witness(q) {
        // Both constructions have exactly representable products and 128-K
        // partial/outer sums. An extra rounding error is never admissible.
        Expected::Bits(
            reference
                .values
                .iter()
                .flat_map(|&v| {
                    if q.output == DType::F32 {
                        (v as f32).to_ne_bytes().to_vec()
                    } else {
                        kernel_core::codec::f32_to_bf16(v as f32)
                            .to_ne_bytes()
                            .to_vec()
                    }
                })
                .collect(),
        )
    } else {
        Expected::Values(reference)
    }
}

pub fn case<'a>(
    query: Mxfp8FoldGemmQuery,
    capacity: u32,
    op: Prepared<Mxfp8FoldGemmBuffers<View>>,
) -> LabResult<OpCase<'a>> {
    let a = activation(query, capacity);
    let w = weight(query, false);
    let compact = weight(query, true);
    let Scales::Ue8M0(a_scales) = a.scales else {
        unreachable!()
    };
    let Scales::Ue8M0(w_scales) = w.scales else {
        unreachable!()
    };
    let roles = Mxfp8FoldGemmBuffers {
        activations: Spec::input(
            DType::U8,
            (Rows::Live, query.activation_row_stride as usize),
            a.codes,
        ),
        activation_scales: Spec::input(
            DType::U8,
            (Rows::Live, query.activation_row_stride as usize / 32),
            a_scales,
        ),
        weights: Spec::weight(w.codes),
        weight_scales: Spec::weight(w_scales),
        output: Spec::output(
            query.output,
            (Rows::Live, (query.groups * query.n) as usize),
        ),
        live_rows: Spec::live(),
        status: Spec::state_updating(
            DType::I32,
            (Rows::Live, 1),
            vec![0; capacity as usize * 4],
            Write::Map(Box::new(|_, _| {
                crate::poison::WriteMap::of(DType::I32).none()
            })),
            |_, live| {
                Ok(if live == 0 {
                    vec![]
                } else {
                    std::iter::once(0..live as usize * 4).collect()
                })
            },
        ),
        status_row_offset: 0,
    };
    let mut counts = vec![0, 1, capacity];
    if capacity > 1 {
        counts.push(capacity - 1);
    }
    if capacity > 3 {
        counts.push(3);
    }
    counts.sort_unstable();
    counts.dedup();
    Ok(OpCase::new(op, capacity, roles)?
        .oracle(move |images, _| {
            Ok(vec![
                (
                    "output".into(),
                    expected(query, reference(query, images, &compact)?),
                ),
                (
                    "status".into(),
                    Expected::Bits(images.bytes("status")?.to_vec()),
                ),
            ])
        })
        .live_counts(counts))
}

pub fn sm120() -> Sm120 {
    mxfp8_fold::lab_entries()
        .into_iter()
        .fold(Sm120::new(), Sm120::with_mxfp8_fold)
}

/// Row, batch and chunk comparisons require at least two rows. A one-row
/// case still requires numerical correctness, live counts and poison replays.
pub fn need(capacity: u32) -> Need {
    Need::rows(capacity).with_exact(if capacity > 1 {
        Invariance::any_rows(capacity)
    } else {
        Invariance::NONE
    })
}

pub fn required<'a>() -> Vec<Required<'a>> {
    mxfp8_fold::lab_entries()
        .into_iter()
        .map(|entry| {
            let q = entry.query;
            Required::new(
                format!("MXFP8 folds {q:?}"),
                need(q.m_capacity),
                move |backend, need| backend.mxfp8_fold_gemm(&q, need),
                move |op| case(q, q.m_capacity, op),
            )
            .required()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn requested_properties_have_a_distinct_row_and_split() {
        let (mut single, mut multi) = (0, 0);
        for case in required() {
            if case.need.capacity == 1 {
                single += 1;
                assert_eq!(case.need.exact, Invariance::NONE, "{}", case.name);
            } else {
                multi += 1;
                assert_eq!(case.need.exact, Invariance::any_rows(case.need.capacity));
            }
        }
        assert!(single > 0 && multi > 0);
    }

    #[test]
    fn compressed_reference_equals_full_f64_dots_and_catches_single_chain() {
        for entry in mxfp8_fold::lab_entries() {
            let q = entry.query;
            let a = activation(q, 1);
            let w = weight(q, false);
            let Scales::Ue8M0(scales) = &a.scales else {
                unreachable!()
            };
            let images = Images::default()
                .with("activations", a.codes.clone())
                .with("activation_scales", scales.clone());
            let compact = reference(q, &images, &weight(q, true)).unwrap();
            let full = mxfp8_gemm::oracle(q, 1, &a, &w).unwrap();
            assert_eq!(compact.values, full.values);
            assert_eq!(compact.sum_abs_products, full.sum_abs_products);
            if witness(q) {
                let a = mxfp8_gemm::exact_bf16_values(&a).unwrap();
                let b = mxfp8_gemm::exact_bf16_values(&w).unwrap();
                let one = a
                    .iter()
                    .zip(&b)
                    .fold(0.0_f32, |sum, (&a, &b)| (a as f32).mul_add(b as f32, sum));
                assert_ne!(f64::from(one), compact.values[0]);
                let folded = mxfp8_gemm::folded_dot(
                    &a.iter().map(|&v| v as f32).collect::<Vec<_>>(),
                    &b.iter().map(|&v| v as f32).collect::<Vec<_>>(),
                )
                .unwrap();
                assert_eq!(f64::from(folded), compact.values[0]);
                if q.output == DType::F32 {
                    let wide = a
                        .chunks(256)
                        .zip(b.chunks(256))
                        .fold(0.0_f32, |total, (a, b)| {
                            total
                                + a.iter().zip(b).fold(0.0_f32, |sum, (&a, &b)| {
                                    (a as f32).mul_add(b as f32, sum)
                                })
                        });
                    assert_eq!(folded, 33556476.0);
                    assert_eq!(wide, 33556472.0);
                    let old = OracleOutput::new(
                        compact.values.clone(),
                        compact.sum_abs_products.clone(),
                        128,
                    )
                    .unwrap();
                    kernel_core::oracle::check_numeric(&[f64::from(wide)], &old, DType::F32)
                        .unwrap();
                    let Expected::Bits(bits) = expected(q, compact) else {
                        panic!("fold witness must require exact bits")
                    };
                    assert_eq!(bits, folded.to_ne_bytes());
                    assert_ne!(bits, wide.to_ne_bytes());
                    assert_ne!(bits, one.to_ne_bytes());
                }
            } else if scale_witness(q) {
                assert_eq!(full.values[0], 160.0);
                assert_eq!(full.values[64], 320.0);
                let mut wrong = w.clone();
                let Scales::Ue8M0(scales) = &mut wrong.scales else {
                    unreachable!()
                };
                for row in 8..q.n as usize {
                    for block in 0..q.k as usize / 32 {
                        scales[row * q.k as usize / 32 + block] =
                            scales[(row % 8) * q.k as usize / 32 + block];
                    }
                }
                let wrong = mxfp8_gemm::oracle(q, 1, &a, &wrong).unwrap();
                assert_eq!(wrong.values[64], 160.0);
                assert!(matches!(expected(q, compact), Expected::Bits(_)));
            }
        }
    }
}
