//! CPU fixtures for generated dense cases. Caches retain only the current shape.
use super::*;

type WeightKey = (usize, usize, String, bool);
type ActivationKey = (usize, usize);
thread_local! {
    static WEIGHT: RefCell<Option<(WeightKey, Rc<Encoded>)>> = const { RefCell::new(None) };
    static ACTIVATION: RefCell<Option<(ActivationKey, Encoded)>> = const { RefCell::new(None) };
}

pub(super) fn weight_fixture(variant: &GemmVariant) -> LabResult<Rc<Encoded>> {
    let q = query(variant);
    let fragment = gemm::fragment_weight_enabled(&variant.key);
    let key = (
        q.n as usize,
        q.k as usize,
        format!("{:?}", q.weight),
        fragment,
    );
    WEIGHT.with(|cache| {
        let mut cache = cache.borrow_mut();
        if let Some((old, encoded)) = &*cache
            && old == &key
        {
            return Ok(Rc::clone(encoded));
        }
        // Release a large previous matrix before encoding the next shape.
        *cache = None;
        let mut encoded = encode_seeded_weight(key.0, key.1, q.weight, true)?;
        if fragment {
            encoded.data = FragmentWeightLayout {
                rows: key.0,
                k: key.1,
                format: if q.weight == WeightFormat::Bf16 {
                    FragmentWeightFormat::Bf16
                } else {
                    FragmentWeightFormat::Fp8E4M3
                },
            }
            .pack(&encoded.data)
            .map_err(|e| e.to_string())?
            .bytes;
        }
        let encoded = Rc::new(encoded);
        *cache = Some((key, Rc::clone(&encoded)));
        Ok(encoded)
    })
}

pub(super) fn activation_fixture(
    capacity: usize,
    k: usize,
    m: usize,
    format: ActivationFormat,
) -> LabResult<Encoded> {
    if format != ActivationFormat::Bf16 {
        // Calibration depends on the live prefix, so preserve that encoding.
        let mut values = vec![0.0; capacity * k];
        values[..m * k].copy_from_slice(&seeded_values(m * k, 0x4143_5449_5641_5445));
        return activation(&values, capacity, k, format, true);
    }
    ACTIVATION.with(|cache| {
        let mut cache = cache.borrow_mut();
        if cache.as_ref().is_none_or(|(key, _)| *key != (capacity, k)) {
            *cache = None;
            *cache = Some((
                (capacity, k),
                encoded_bf16(&seeded_values(capacity * k, 0x4143_5449_5641_5445), true),
            ));
        }
        let mut encoded = cache.as_ref().expect("initialized fixture").1.clone();
        encoded.data[m * k * 2..].fill(0);
        encoded.decoded[m * k..].fill(0.0);
        Ok(encoded)
    })
}

/// Split output columns across at most eight workers. Each dot product keeps
/// the serial oracle's K order, and SwiGLU runs after restoring row order.
pub(super) fn parallel_oracle(
    a: &[f64],
    b: &[f64],
    m: usize,
    n: usize,
    k: usize,
    swiglu: Option<SwigluColumns>,
) -> LabResult<OracleOutput> {
    let workers = std::thread::available_parallelism()
        .map_or(1, usize::from)
        .min(8)
        .min(n);
    let columns = n.div_ceil(workers);
    let pieces = std::thread::scope(|scope| {
        let jobs = b
            .chunks(columns * k)
            .map(|b| {
                scope.spawn(move || {
                    oracle::gemm(
                        a,
                        b,
                        GemmShape {
                            m,
                            n: b.len() / k,
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
            })
            .collect::<Vec<_>>();
        jobs.into_iter()
            .map(|job| {
                job.join()
                    .map_err(|_| "oracle worker panicked".to_string())?
                    .map_err(|e| e.to_string())
            })
            .collect::<LabResult<Vec<_>>>()
    })?;
    let mut output =
        OracleOutput::new(vec![0.0; m * n], vec![0.0; m * n], k).map_err(|e| e.to_string())?;
    let mut first = 0;
    for piece in pieces {
        let width = piece.values.len() / m;
        for row in 0..m {
            output.values[row * n + first..row * n + first + width]
                .copy_from_slice(&piece.values[row * width..(row + 1) * width]);
            output.sum_abs_products[row * n + first..row * n + first + width]
                .copy_from_slice(&piece.sum_abs_products[row * width..(row + 1) * width]);
        }
        first += width;
    }
    match swiglu {
        Some(columns) => oracle::silu_glu(&output, n, columns, oracle::SiluGlu::UNCLAMPED)
            .map_err(|e| e.to_string()),
        None => Ok(output),
    }
}

#[test]
fn parallel_oracle_preserves_every_reference_bit() {
    for m in [1, 3] {
        for n in [2, 18, 514] {
            for swiglu in [
                None,
                Some(SwigluColumns::ConcatenatedHalves),
                Some(SwigluColumns::Interleaved),
            ] {
                let k = 65;
                let a = seeded_values(m * k, 271)
                    .into_iter()
                    .map(f64::from)
                    .collect::<Vec<_>>();
                let b = seeded_values(n * k, 981)
                    .into_iter()
                    .map(f64::from)
                    .collect::<Vec<_>>();
                let expected = oracle::gemm(
                    &a,
                    &b,
                    GemmShape {
                        m,
                        n,
                        k,
                        experts: 1,
                    },
                    &[],
                    GemmEpilogue {
                        residual: None,
                        swiglu,
                    },
                )
                .unwrap();
                let actual = parallel_oracle(&a, &b, m, n, k, swiglu).unwrap();
                assert_eq!(
                    expected
                        .values
                        .iter()
                        .map(|v| v.to_bits())
                        .collect::<Vec<_>>(),
                    actual
                        .values
                        .iter()
                        .map(|v| v.to_bits())
                        .collect::<Vec<_>>()
                );
                assert_eq!(
                    expected
                        .sum_abs_products
                        .iter()
                        .map(|v| v.to_bits())
                        .collect::<Vec<_>>(),
                    actual
                        .sum_abs_products
                        .iter()
                        .map(|v| v.to_bits())
                        .collect::<Vec<_>>()
                );
                assert_eq!(expected.reduction_lengths, actual.reduction_lengths);
            }
        }
    }
}

#[test]
fn cached_activation_preserves_live_prefix_and_zero_tail() {
    for m in [0, 1, 15, 16, 17, 32] {
        let mut values = vec![0.0; 32 * 128];
        values[..m * 128].copy_from_slice(&seeded_values(m * 128, 0x4143_5449_5641_5445));
        let expected = encoded_bf16(&values, true);
        let actual = activation_fixture(32, 128, m, ActivationFormat::Bf16).unwrap();
        assert_eq!(expected.data, actual.data);
        assert_eq!(expected.decoded, actual.decoded);
    }
}

/// Round only recipes that form a BF16 scaled weight before MMA.
pub(super) fn oracle_weight(value: f64, order: kernel_core::precision::DecodeOrder) -> f64 {
    if order == kernel_core::precision::DecodeOrder::ScaleThenCode {
        f64::from(bf16_to_f32(kernel_core::codec::f64_to_bf16(value)))
    } else {
        value
    }
}

#[test]
fn oracle_rounding_follows_the_declared_recipe() {
    use kernel_core::precision::DecodeOrder::*;
    let value = 0.137_f64 * 1.5;
    let rounded = f64::from(bf16_to_f32(kernel_core::codec::f64_to_bf16(value)));
    assert_ne!(rounded, value);
    assert_eq!(oracle_weight(value, ScaleThenCode), rounded);
    for order in [
        BlockScaleAfterMma,
        ExactScaledOperand,
        ExactLocalThenGlobal,
        NativeBlockScaledMma,
    ] {
        assert_eq!(oracle_weight(value, order), value);
    }
}
