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
use kernel_core::properties::{Invariance, Property};

pub const SINGLE_ROW_PROPERTIES: &str =
    "one-row capacity has no row permutation, neighbour or chunk split";
use kernels_sm120::f32_projection::Entry;
use std::cell::RefCell;

/// The original kernel's conservative whole-row/tile admission predicate.
pub fn tensor_operand(value: f32) -> bool {
    let bits = value.to_bits() & 0x7fff_ffff;
    bits == 0 || ((97 << 23)..(158 << 23)).contains(&bits)
}

/// Independent host classification for comparison with create-time flags.
pub fn weight_domains(weights: &[f32], k: usize, format: WeightFormat) -> Vec<u32> {
    weights
        .chunks(16 * k)
        .map(|tile| {
            u32::from(format == WeightFormat::Bf16 && tile.iter().all(|&w| tensor_operand(w)))
        })
        .collect()
}

/// F32 chain witnesses. General MMA results use the f64 oracle bound;
/// these serial chains also test split storage and ordered reduction on CPU.
pub fn chain_values(a: &[f32], b: &[f32], tensor: bool) -> Vec<f32> {
    a.chunks(128)
        .zip(b.chunks(128))
        .map(|(a, b)| {
            if !tensor {
                return a
                    .iter()
                    .zip(b)
                    .fold(0_f32, |sum, (&a, &b)| a.mul_add(b, sum));
            }
            let mut sums = [0_f32; 3];
            for (&a, &b) in a.iter().zip(b) {
                for (sum, code) in sums
                    .iter_mut()
                    .zip(f32_gemm::decompose(a).unwrap().unwrap())
                {
                    *sum = bf16_to_f32(code).mul_add(b, *sum);
                }
            }
            (sums[0] + sums[1]) + sums[2]
        })
        .collect()
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Witness {
    General,
    LowComponent,
    WeightBits,
    SeparateChains,
    MergeOrder,
    FoldBoundary,
    Fallback,
    Underflow,
    NonfiniteInput,
    NonfiniteWeight,
    Overflow,
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
        Need::rows(capacity).with_exact(if capacity > 1 {
            Invariance::any_rows(capacity)
        } else {
            Invariance::NONE
        }),
        move |backend, need| backend.projection(&spec, usage, need).map(|b| b.handle),
        move |op| case(&entry, witness, op),
    )
    .required()
}

pub fn operands(entry: &Entry, witness: Witness) -> (Vec<f32>, Vec<f32>) {
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
                Witness::MergeOrder => match c {
                    0 => 1.0 + 2_f32.powi(-8) + 2_f32.powi(-23),
                    16 => 65536.,
                    _ => 0.,
                },
                Witness::FoldBoundary => match c {
                    0 => 1.,
                    128..=255 => 2_f32.powi(-24),
                    256 => -1.,
                    _ => 0.,
                },
                Witness::Underflow => {
                    if c == 0 {
                        f32::from_bits(1 + (r % 4) as u32)
                    } else {
                        0.
                    }
                }
                Witness::NonfiniteInput => {
                    if c == 0 {
                        [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 1.][r % 4]
                    } else {
                        0.
                    }
                }
                Witness::NonfiniteWeight => {
                    if c == 0 {
                        1.
                    } else {
                        0.
                    }
                }
                Witness::Overflow => {
                    if c == 0 && r % 4 == 0 {
                        f32::MAX
                    } else if c == 0 {
                        1.
                    } else {
                        0.
                    }
                }
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
                Witness::Underflow => {
                    if c == 0 {
                        0.5
                    } else {
                        0.
                    }
                }
                Witness::NonfiniteInput => {
                    if c == 0 {
                        1.
                    } else {
                        0.
                    }
                }
                Witness::NonfiniteWeight => {
                    if c == 0 {
                        [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 1.][r % 4]
                    } else {
                        0.
                    }
                }
                Witness::Overflow => {
                    if c == 0 {
                        2.
                    } else {
                        0.
                    }
                }
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
            weight_domain: op.info().bytes_of("scratch.weight_domain").map(|_| {
                Spec::weight(
                    weight_domains(&b, q.k as usize, q.weight)
                        .into_iter()
                        .flat_map(u32::to_le_bytes)
                        .collect(),
                )
            }),
            quantizer_status: None,
            quantizer_status_row_offset: 0,
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
    let cached = RefCell::new(None::<(Vec<u8>, f32_gemm::Oracle, Option<Vec<u8>>)>);
    let case = OpCase::new(op, q.m_capacity, roles)?
        .oracle(move |images: &Images, live| {
            if matches!(
                witness,
                Witness::NonfiniteInput | Witness::NonfiniteWeight | Witness::Overflow
            ) {
                // Exceptional operands are outside the finite numerical contract.
                // Require IEEE propagation and isolation, never a device trap.
                let a = images.f32s("activation")?;
                let mut expected = Vec::new();
                for row in a.chunks_exact(q.k as usize).take(live as usize) {
                    for weight in b.chunks_exact(q.k as usize) {
                        let mut total = 0_f32;
                        for (a, b) in row.chunks(128).zip(weight.chunks(128)) {
                            let partial =
                                a.iter().zip(b).fold(0_f32, |v, (&a, &b)| a.mul_add(b, v));
                            total += partial;
                        }
                        expected.push(total);
                    }
                }
                return Ok(vec![(
                    "output".into(),
                    Expected::Judge(Box::new(move |bytes| {
                        if bytes.len() < expected.len() * 4 {
                            return Err("output extent differs".into());
                        }
                        for (i, (&expected, raw)) in
                            expected.iter().zip(bytes.chunks_exact(4)).enumerate()
                        {
                            let actual = f32::from_le_bytes(raw.try_into().unwrap());
                            if !(expected.is_nan() && actual.is_nan())
                                && actual.to_bits() != expected.to_bits()
                            {
                                return Err(format!(
                                    "output[{i}] exceptional propagation or row isolation differs"
                                ));
                            }
                        }
                        Ok(())
                    })),
                )]);
            }
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
        );
    Ok(if q.m_capacity == 1 {
        case.without_properties(
            &[Property::Rows, Property::Chunk, Property::Batch],
            SINGLE_ROW_PROPERTIES,
        )
    } else {
        case
    })
}

fn judge(reference: &f32_gemm::Oracle, bytes: &[u8], count: usize) -> LabResult<()> {
    if bytes.len() < count * 4 || reference.values.len() < count {
        return Err("F32 oracle output length differs".into());
    }
    for (i, bytes) in bytes.chunks_exact(4).take(count).enumerate() {
        let value = f32::from_le_bytes(bytes.try_into().expect("four bytes")) as f64;
        let bound = f32_gemm::bound(reference.k, reference.sum_abs_products[i]);
        let error = (value - reference.values[i]).abs();
        if !value.is_finite() || !bound.is_finite() || error > bound {
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
    fn cached_domain_covers_every_bf16_code_and_tile_tail() {
        for code in 0..=u16::MAX {
            let value = bf16_to_f32(code);
            let magnitude = code & 0x7fff;
            let expected = magnitude == 0 || (97 * 128..158 * 128).contains(&magnitude);
            assert_eq!(tensor_operand(value), expected, "{code:04x}");
            let mut weights = vec![1.; 17 * 129];
            weights[15 * 129 + 128] = value;
            assert_eq!(
                weight_domains(&weights, 129, WeightFormat::Bf16),
                [u32::from(expected), 1]
            );
            assert_eq!(weight_domains(&weights, 129, WeightFormat::F32), [0, 0]);
        }
    }

    #[test]
    fn split_chain_storage_matches_original_oracle_and_rejects_grouped_sums() {
        let mut seed = 0x3287_5101_u32;
        let mut random = || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            f32::from_bits((seed & 0x807f_ffff) | (127 << 23))
        };
        for k in [
            1, 63, 64, 65, 127, 128, 129, 255, 256, 257, 511, 512, 513, 5120,
        ] {
            for _ in 0..16 {
                let a: Vec<_> = (0..k).map(|_| random()).collect();
                let b: Vec<_> = (0..k).map(|_| bf16_to_f32(f32_to_bf16(random()))).collect();
                let chains = chain_values(&a, &b, true);
                // Completion order never determines a plane's K index.
                let mut stored = vec![0.; chains.len()];
                for index in (0..chains.len()).rev() {
                    stored[index] = chains[index];
                }
                let actual = stored.into_iter().fold(0_f32, |a, b| a + b);
                assert_eq!(
                    actual.to_bits(),
                    f32_gemm::folded_dot(&a, &b, DType::Bf16).unwrap().to_bits()
                );
                let query = GemmQuery {
                    m_capacity: 1,
                    n: 1,
                    k: k as u32,
                    weight: WeightFormat::Bf16,
                    activation: ActivationFormat::F32,
                    output: DType::F32,
                    epilogues: Epilogues::default(),
                    grouped: None,
                };
                let oracle = f32_gemm::oracle(query, 1, &a, &b).unwrap();
                f32_gemm::check_numeric(&[f64::from(actual)], &oracle).unwrap();
            }
        }
        let chains = [1e20_f32, 0., 0., 0., -1e20, 1., 0., 0.];
        let correct = chains.into_iter().fold(0_f32, |a, b| a + b);
        let grouped = chains
            .chunks(4)
            .map(|c| c.iter().copied().fold(0_f32, |a, b| a + b))
            .fold(0_f32, |a, b| a + b);
        assert_eq!(correct, 1.);
        assert_ne!(correct.to_bits(), grouped.to_bits());
    }

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
            (Witness::MergeOrder, 65537.),
        ] {
            let (a, b) = operands(&entry, witness);
            let correct = f32_gemm::folded_dot(&a[..257], &b[..257], DType::Bf16).unwrap();
            assert_eq!(correct.to_bits(), expected.to_bits(), "{witness:?}");
            if witness == Witness::MergeOrder {
                let mut components = [0_f32; 3];
                for &x in &a[..128] {
                    for (partial, term) in components
                        .iter_mut()
                        .zip(f32_gemm::decompose(x).unwrap().unwrap())
                    {
                        *partial += bf16_to_f32(term);
                    }
                }
                let wrong = components[0] + (components[1] + components[2]);
                assert_eq!(wrong.to_bits(), (65537_f32 + 2_f32.powi(-7)).to_bits());
                assert_ne!(correct.to_bits(), wrong.to_bits());
                let oracle = f32_gemm::oracle(entry.query(), 1, &a[..257], &b).unwrap();
                assert!(
                    (f64::from(wrong) - oracle.values[0]).abs()
                        <= f32_gemm::bound(257, oracle.sum_abs_products[0])
                );
                continue;
            }
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
