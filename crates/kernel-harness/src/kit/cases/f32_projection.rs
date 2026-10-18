//! Original F32 operand oracles and exact witnesses for projection chains.
use crate::{
    LabResult,
    conformance::{Expected, Images, Rows},
    kit::{OpCase, Required, Spec},
};
use kernel_core::backend::{Need, Prepared, View};
use kernel_core::codec::{bf16_to_f32, f32_to_bf16};
use kernel_core::contract::{DType, WeightFormat};
use kernel_core::precision::{
    self,
    buffers::{ProjectionBuffers, ProjectionScratch, QuantizedWeights},
    f32_gemm,
};
use kernel_core::properties::Invariance;
use kernels_sm120::f32_projection::Entry;
use std::cell::RefCell;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Witness {
    General,
    LowComponent,
    WeightBits,
    SeparateChains,
    FoldBoundary,
    Fallback,
}

pub fn operation(entry: &Entry) -> precision::OperationSpec {
    let q = entry.query();
    let f32_weight = q.weight == WeightFormat::F32;
    precision::OperationSpec {
        id: entry.symbol().into(),
        kind: precision::OperationKind::Dense,
        target: kernel_core::contract::Target::Sm120f,
        weights: vec![precision::WeightSpec {
            codes: precision::TensorSpan {
                tensor: "weight".into(),
                dtype: if f32_weight { DType::F32 } else { DType::Bf16 },
                shape: vec![q.n.into(), q.k.into()],
                offset: 0,
                byte_length: entry.weight_bytes(),
            },
            logical_shape: vec![q.n.into(), q.k.into()],
            format: q.weight,
            encoding: if f32_weight {
                precision::CodeEncoding::F32LittleEndian
            } else {
                precision::CodeEncoding::Bf16LittleEndian
            },
            layout: precision::WeightLayout::row_major(),
            local_scale: None,
            global_scale: None,
            transforms: vec![],
        }],
        geometry: None,
        activations: vec![precision::ActivationSpec::f32()],
        arithmetic: entry.capability().arithmetic,
        epilogues: q.epilogues,
        uses: vec![entry.usage()],
    }
}

pub fn family<'a>(entries: &[Entry]) -> Vec<Required<'a>> {
    entries
        .iter()
        .map(|entry| required(entry.clone(), Witness::General))
        .collect()
}

pub fn required<'a>(entry: Entry, witness: Witness) -> Required<'a> {
    let spec = operation(&entry);
    let usage = entry.usage();
    let capacity = entry.query().m_capacity;
    Required::new(
        format!("{} {witness:?}", entry.symbol()),
        Need::rows(capacity).with_exact(Invariance::any_rows(capacity)),
        move |backend, need| backend.projection(&spec, usage, need).map(|b| b.handle),
        move |op| case(&entry, witness, op),
    )
    .required()
}

fn operands(entry: &Entry, witness: Witness) -> (Vec<f32>, Vec<f32>) {
    let q = entry.query();
    let (m, n, k) = (q.m_capacity as usize, q.n as usize, q.k as usize);
    let mut a = vec![0.; m * k];
    let mut b = vec![0.; n * k];
    for r in 0..m {
        for c in 0..k {
            a[r * k + c] = match witness {
                Witness::General => ((r * 127 + c * 13) % 997) as f32 / 251.13 - 1.73,
                Witness::WeightBits => {
                    if c == 0 {
                        1.
                    } else {
                        0.
                    }
                }
                Witness::LowComponent => {
                    if c == 0 {
                        1.0 + 2_f32.powi(-8) + 2_f32.powi(-23)
                    } else {
                        0.
                    }
                }
                Witness::SeparateChains => match c {
                    0 | 16 => 1.0 + 2_f32.powi(-8) + 2_f32.powi(-23),
                    32 => -2.,
                    _ => 0.,
                },
                Witness::FoldBoundary => match c {
                    0 => 1.,
                    128..=255 => 2_f32.powi(-24),
                    256 => -1.,
                    _ => 0.,
                },
                Witness::Fallback => {
                    if c == 0 {
                        match r % 4 {
                            0 => f32::from_bits(0x10001),
                            1 => 2_f32.powi(-100),
                            2 => 1.000123,
                            _ => -0.,
                        }
                    } else {
                        0.
                    }
                }
            };
        }
    }
    for r in 0..n {
        for c in 0..k {
            let v = match witness {
                Witness::General => ((r * 31 + c * 17) % 251) as f32 / 131.7 - 0.94,
                Witness::Fallback => {
                    if c == 0 {
                        2_f32.powi(100)
                    } else {
                        0.
                    }
                }
                Witness::WeightBits => {
                    if c == 0 {
                        1. + 2_f32.powi(-8) + 2_f32.powi(-23)
                    } else {
                        0.
                    }
                }
                Witness::LowComponent => {
                    if c == 0 {
                        1.
                    } else {
                        0.
                    }
                }
                _ => 1.,
            };
            b[r * k + c] = if q.weight == WeightFormat::Bf16 {
                bf16_to_f32(f32_to_bf16(v))
            } else {
                v
            };
        }
    }
    (a, b)
}

pub fn case<'a>(
    entry: &Entry,
    witness: Witness,
    op: Prepared<ProjectionBuffers<View>>,
) -> LabResult<OpCase<'a>> {
    let q = entry.query();
    let (a, b) = operands(entry, witness);
    let weights = if q.weight == WeightFormat::Bf16 {
        b.iter()
            .flat_map(|&v| f32_to_bf16(v).to_le_bytes())
            .collect()
    } else {
        b.iter().flat_map(|v| v.to_le_bytes()).collect()
    };
    let roles = ProjectionBuffers {
        activation: Spec::input(
            DType::F32,
            (Rows::Live, q.k as usize),
            a.iter().flat_map(|v| v.to_le_bytes()).collect(),
        ),
        live_rows: Spec::live(),
        output: Spec::output(DType::F32, (Rows::Live, q.n as usize)),
        residual: None,
        weights: QuantizedWeights {
            codes: Spec::weight(weights),
            local_scales: None,
            global_scales: None,
        },
        activation_global: None,
        routing: None,
        scratch: ProjectionScratch {
            partial: Spec::untouched(16),
            semaphores: Spec::untouched(16),
            unused: Spec::untouched(16),
            temporary: None,
            gathered: None,
            codes: None,
            scales: None,
            activation_global: None,
            row_amax: None,
            output_codes: None,
            output_scales: None,
        },
        row_offset: 0,
        shard_rows: 0,
    };
    let cached =
        RefCell::new(None::<(Vec<u8>, kernel_core::oracle::OracleOutput, Option<Vec<u8>>)>);
    Ok(OpCase::new(op, q.m_capacity, roles)?
        .oracle(move |images: &Images, live| {
            let input = images.bytes("activation")?;
            if cached
                .borrow()
                .as_ref()
                .is_none_or(|(key, _, _)| key != input)
            {
                let a = images.f32s("activation")?;
                let rows = a.len() / q.k as usize;
                // The contract oracle owns the K-dependent bound and underflow
                // term. Exact witnesses supplement it, never relax its bound.
                let oracle = f32_gemm::oracle(q, rows, &a, &b).map_err(|e| e.to_string())?;
                let exact = if witness == Witness::General {
                    None
                } else {
                    let dtype = if q.weight == WeightFormat::Bf16 {
                        DType::Bf16
                    } else {
                        DType::F32
                    };
                    let mut values = Vec::new();
                    for a in a.chunks_exact(q.k as usize) {
                        for b in b.chunks_exact(q.k as usize) {
                            values.push(
                                f32_gemm::folded_dot(a, b, dtype).map_err(|e| e.to_string())?,
                            );
                        }
                    }
                    let bytes = values
                        .iter()
                        .flat_map(|v| v.to_le_bytes())
                        .collect::<Vec<_>>();
                    judge(&oracle, &bytes, values.len())?;
                    Some(bytes)
                };
                *cached.borrow_mut() = Some((input.to_vec(), oracle, exact));
            }
            let cache = cached.borrow();
            let (_, oracle, exact) = cache.as_ref().expect("computed");
            let expected = if let Some(bits) = exact {
                Expected::Bits(bits.clone())
            } else {
                let oracle = oracle.clone();
                let count = live as usize * q.n as usize;
                Expected::Judge(Box::new(move |bytes| judge(&oracle, bytes, count)))
            };
            Ok(vec![("output".into(), expected)])
        })
        .live_counts(
            [0, 1, 3, 7, 8, 9, q.m_capacity - 1, q.m_capacity]
                .into_iter()
                .filter(|&r| r <= q.m_capacity)
                .collect(),
        ))
}

fn judge(
    reference: &kernel_core::oracle::OracleOutput,
    bytes: &[u8],
    count: usize,
) -> LabResult<()> {
    if bytes.len() < count * 4 || reference.values.len() < count {
        return Err("F32 oracle output length differs".into());
    }
    for (i, bytes) in bytes.chunks_exact(4).take(count).enumerate() {
        let value = f32::from_le_bytes(bytes.try_into().expect("four bytes")) as f64;
        let bound = f32_gemm::bound(reference.sum_abs_products[i]);
        let error = (value - reference.values[i]).abs();
        if !value.is_finite() || error > bound {
            return Err(format!(
                "output[{i}] differs from F64: error={error}, bound={bound}"
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use kernel_core::contract::{ActivationFormat, Epilogues, GemmQuery};
    use precision::{ExecutionPhase, OperationUse};

    #[test]
    fn exact_witnesses_distinguish_component_loss_and_merged_chains() {
        let entry = Entry::new(
            GemmQuery {
                m_capacity: 9,
                n: 17,
                k: 257,
                weight: WeightFormat::Bf16,
                activation: ActivationFormat::F32,
                output: DType::F32,
                epilogues: Epilogues::default(),
                grouped: None,
            },
            OperationUse {
                phase: ExecutionPhase::Prefill,
                capacity: 9,
                graph: false,
            },
            "witness",
        )
        .unwrap();
        for (witness, expected) in [
            (Witness::LowComponent, 1. + 2_f32.powi(-8) + 2_f32.powi(-23)),
            (Witness::SeparateChains, 2_f32.powi(-7) + 2_f32.powi(-22)),
            (Witness::FoldBoundary, 2_f32.powi(-17)),
        ] {
            let (a, b) = operands(&entry, witness);
            let correct = f32_gemm::folded_dot(&a[..257], &b[..257], DType::Bf16).unwrap();
            assert_eq!(correct.to_bits(), expected.to_bits(), "{witness:?}");
            if witness == Witness::FoldBoundary {
                continue;
            }
            let mut merged = 0_f32;
            let mut no_lo = 0_f32;
            for (&x, &w) in a[..257].iter().zip(&b[..257]) {
                let terms = f32_gemm::decompose(x).unwrap().unwrap();
                for &term in &terms {
                    merged = bf16_to_f32(term).mul_add(w, merged);
                }
                no_lo += (bf16_to_f32(terms[0]) + bf16_to_f32(terms[1])) * w;
            }
            assert_ne!(correct.to_bits(), no_lo.to_bits());
            if witness == Witness::SeparateChains {
                assert_ne!(correct.to_bits(), merged.to_bits());
            }
        }
    }
}
