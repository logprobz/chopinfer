//! GPU-gated dense GEMM correctness and timing cases.

#[path = "generated_host.rs"]
mod generated_host;

use kernel_core::codec::{bf16_to_f32, f32_to_bf16};
use kernel_core::contract::{
    ActivationFormat, DType, GemmQuery, RuntimeCounts, ScaleMode, StaticQuery, SwigluColumns,
    TensorMap, TensorMapElement, TensorMapInterleave, TensorMapL2Promotion, TensorMapOobFill,
    TensorMapSwizzle, WeightFormat,
};
use kernel_core::oracle::{self, GemmEpilogue, GemmShape, OracleOutput};
use kernel_core::quantize::{self, Quantized, Scales};
use kernel_core::transform::Transform;
use kernel_core::transform::fragment::{FragmentWeightFormat, FragmentWeightLayout};
use kernel_lab::launch::{self, Bindings};
use kernel_lab::poison::{self, Evidence, Fill, POISON, Replay, WriteMap};
use kernel_lab::{LabResult, Session, check_output, gpu_test, time};
use kernels_sm120::Profile;
use kernels_sm120::gemm::{self, GemmBuffers, GemmVariant};

use runtime_cuda::{DeviceBuffer, Function, KernelArgument, Stream};
use serde_json::{Value, json};
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::fs;
use std::rc::Rc;
use std::sync::mpsc::{self, RecvTimeoutError};
use std::time::Duration;

fn watched_gpu_test<T>(
    need_bytes: u64,
    timeout_seconds: u64,
    run: impl FnOnce(&Session) -> LabResult<T>,
) -> LabResult<Option<T>> {
    let (done, signal) = mpsc::channel::<()>();
    let watchdog = std::thread::spawn(move || {
        if matches!(
            signal.recv_timeout(Duration::from_secs(timeout_seconds)),
            Err(RecvTimeoutError::Timeout)
        ) {
            eprintln!("GEMM GPU test exceeded {timeout_seconds}s; terminating hung CUDA context");
            std::process::exit(124);
        }
    });
    let result = gpu_test(need_bytes, run);
    drop(done);
    watchdog.join().expect("GEMM watchdog thread");
    result
}

#[path = "gemm_mid.rs"]
mod mid_cases;

#[path = "gemm_chunks.rs"]
mod chunk_cases;

#[path = "gemm_invariance.rs"]
pub(crate) mod invariance;

const WEIGHT: u64 = 1;
const ACTIVATION: u64 = 2;
const OUTPUT: u64 = 3;
const PARTIAL: u64 = 4;
const SEMAPHORES: u64 = 5;
const WEIGHT_SCALES: u64 = 6;
const ACTIVATION_SCALES: u64 = 7;
const WEIGHT_GLOBAL: u64 = 8;
const ACTIVATION_GLOBAL: u64 = 9;
const RESIDUAL: u64 = 10;
const RESIDUAL_SEED: u64 = 0x5245_5349_4455_414c;

fn query(variant: &GemmVariant) -> GemmQuery {
    let StaticQuery::Gemm(query) = variant.key.query else {
        unreachable!("GEMM catalog contains only GEMM keys")
    };
    query
}

fn seeded_values(len: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let centered = ((state >> 32) % 513) as i32 - 256;
            centered as f32 / 4096.0
        })
        .collect()
}

#[derive(Clone)]
struct Encoded {
    data: Vec<u8>,
    scales: Option<Vec<u8>>,
    global: Option<f32>,
    decoded: Vec<f64>,
    element: TensorMapElement,
}

fn encoded_quantized(quantized: Quantized, oracle: bool) -> Encoded {
    let decoded = if oracle {
        quantized.decoded()
    } else {
        Vec::new()
    };
    let global = matches!(&quantized.scales, Scales::Ue4M3(_)).then_some(quantized.global_scale);
    let scales = match quantized.scales {
        Scales::F32(values) => values.iter().flat_map(|v| v.to_ne_bytes()).collect(),
        Scales::Ue8M0(values) | Scales::Ue4M3(values) => values,
    };
    let element = if quantized.format.element == kernel_core::codec::SmallFloat::E2M1 {
        TensorMapElement::U4Align8B
    } else {
        TensorMapElement::Fp8E4M3
    };
    Encoded {
        data: quantized.codes,
        scales: Some(scales),
        global,
        decoded,
        element,
    }
}

fn encoded_bf16(values: &[f32], oracle: bool) -> Encoded {
    let bits = values
        .iter()
        .map(|&value| f32_to_bf16(value))
        .collect::<Vec<_>>();
    Encoded {
        data: bits.iter().flat_map(|v| v.to_ne_bytes()).collect(),
        scales: None,
        global: None,
        decoded: if oracle {
            bits.iter()
                .map(|&bits| f64::from(bf16_to_f32(bits)))
                .collect()
        } else {
            Vec::new()
        },
        element: TensorMapElement::Bf16,
    }
}

thread_local! {
    /// Encoded seeded weights by shape. FP8 encoding takes seconds per
    /// matrix on the host, and the cases reuse a few shapes many times.
    static WEIGHTS: RefCell<BTreeMap<(usize, usize, String, bool), Encoded>> =
        const { RefCell::new(BTreeMap::new()) };
}

/// The seeded weight for a `rows` x `k` matrix in `format`, encoded once per
/// thread. FP8 formats get a distinct magnitude per 64-row group.
fn seeded_weight(rows: usize, k: usize, format: WeightFormat, oracle: bool) -> LabResult<Encoded> {
    let key = (rows, k, format!("{format:?}"), oracle);
    if let Some(encoded) = WEIGHTS.with(|cache| cache.borrow().get(&key).cloned()) {
        return Ok(encoded);
    }
    let encoded = encode_seeded_weight(rows, k, format, oracle)?;
    WEIGHTS.with(|cache| {
        let mut cache = cache.borrow_mut();
        // A TP2 correctness run reuses 14 shapes at every row count;
        // a smaller cache would encode them again for each.
        if cache.len() >= 32 {
            cache.clear();
        }
        cache.insert(key, encoded.clone());
    });
    Ok(encoded)
}

fn encode_seeded_weight(
    rows: usize,
    k: usize,
    format: WeightFormat,
    oracle: bool,
) -> LabResult<Encoded> {
    let mut values = seeded_values(rows * k, WEIGHT_SEED);
    // Uniform draws give every FP8 block the same amax and so the same
    // scale, which would hide a kernel reading the wrong scale row. Scale
    // each 64-row group by its own factor so neighbouring groups and blocks
    // differ.
    if matches!(
        format,
        WeightFormat::Fp8Block128x128(_) | WeightFormat::Fp8Rows64x128(_)
    ) {
        for (row, values) in values.chunks_exact_mut(k).enumerate() {
            let factor = 1.0 + 0.37 * ((row / 64) % 5) as f32;
            values.iter_mut().for_each(|value| *value *= factor);
        }
    }
    weight(&values, rows, k, format, oracle)
}

fn sampled_reference(
    variant: &GemmVariant,
    m: u32,
    w: &Encoded,
    a: &Encoded,
    parallel: bool,
) -> LabResult<(OracleOutput, Vec<usize>)> {
    let q = query(variant);
    let k = q.k as usize;
    let n = q.n as usize;
    let output_n = if q.epilogues.swiglu.is_some() {
        n / 2
    } else {
        n
    };
    let rows = kernel_lab::projection_cases::numerical_rows(
        m,
        &[
            u32::from(variant.key.knobs.tile_m),
            u32::from(variant.key.knobs.tile_n),
        ],
    );
    let mut columns = sampled_columns(output_n)
        .into_iter()
        .collect::<std::collections::BTreeSet<_>>();
    for boundary in (64..output_n).step_by(64) {
        columns.extend([boundary - 1, boundary]);
    }
    let columns = columns.into_iter().collect::<Vec<_>>();
    let weight_rows: Vec<_> = match q.epilogues.swiglu {
        None => columns.clone(),
        Some(SwigluColumns::ConcatenatedHalves) => columns
            .iter()
            .copied()
            .chain(columns.iter().map(|c| c + output_n))
            .collect(),
        Some(SwigluColumns::Interleaved) => {
            columns.iter().flat_map(|c| [2 * c, 2 * c + 1]).collect()
        }
    };
    let input = rows
        .iter()
        .flat_map(|r| a.decoded[r * k..(r + 1) * k].iter().copied())
        .collect::<Vec<_>>();
    let order = kernels_sm120::precision::dense_decode_order(
        &q,
        gemm::fragment_weight_enabled(&variant.key),
    );
    let weights = weight_rows
        .iter()
        .flat_map(|r| {
            w.decoded[r * k..(r + 1) * k]
                .iter()
                .map(|v| generated_host::oracle_weight(*v, order))
        })
        .collect::<Vec<_>>();
    let picks = rows
        .iter()
        .flat_map(|r| columns.iter().map(move |c| r * output_n + c))
        .collect::<Vec<_>>();
    let reference = if parallel {
        generated_host::parallel_oracle(
            &input,
            &weights,
            rows.len(),
            weight_rows.len(),
            k,
            q.epilogues.swiglu,
        )?
    } else {
        oracle::gemm(
            &input,
            &weights,
            GemmShape {
                m: rows.len(),
                n: weight_rows.len(),
                k,
                experts: 1,
            },
            &[],
            GemmEpilogue {
                residual: None,
                swiglu: q.epilogues.swiglu,
            },
        )
        .map_err(|e| e.to_string())?
    };
    Ok((reference, picks))
}

fn weight(
    values: &[f32],
    rows: usize,
    k: usize,
    format: WeightFormat,
    oracle: bool,
) -> LabResult<Encoded> {
    let quantized = match format {
        WeightFormat::Bf16 => return Ok(encoded_bf16(values, oracle)),
        WeightFormat::Fp8PerTensor => {
            quantize::fp8_weight_per_tensor(values, rows, k, ScaleMode::ExactF32)
        }
        WeightFormat::Fp8PerChannel => {
            quantize::fp8_weight_per_channel(values, rows, k, ScaleMode::ExactF32)
        }
        WeightFormat::Fp8Block128x128(mode) => quantize::fp8_weight_128x128(values, rows, k, mode),
        WeightFormat::Fp8Rows64x128(mode) => quantize::fp8_blocks(values, rows, k, 64, 128, mode),
        WeightFormat::Mxfp8 => quantize::mxfp8(values, rows, k),
        WeightFormat::Mxfp4 => quantize::mxfp4(values, rows, k),
        WeightFormat::Nvfp4 => quantize::nvfp4(values, rows, k),
    }
    .map_err(|e| e.to_string())?;
    Ok(encoded_quantized(quantized, oracle))
}

fn activation(
    values: &[f32],
    rows: usize,
    k: usize,
    format: ActivationFormat,
    oracle: bool,
) -> LabResult<Encoded> {
    let quantized = match format {
        ActivationFormat::Bf16 => return Ok(encoded_bf16(values, oracle)),
        ActivationFormat::Fp8Group128(mode) => {
            quantize::fp8_activation_1x128(values, rows, k, mode)
        }
        ActivationFormat::Mxfp8 => quantize::mxfp8(values, rows, k),
        ActivationFormat::Nvfp4 => quantize::nvfp4(values, rows, k),
    }
    .map_err(|e| e.to_string())?;
    Ok(encoded_quantized(quantized, oracle))
}

fn map(
    handle: u64,
    rows: u32,
    k: u32,
    tile_rows: u16,
    tile_k: u16,
    element: TensorMapElement,
    row_stride: u32,
) -> TensorMap {
    let byte_width = match element {
        TensorMapElement::Bf16 => 2_u64,
        TensorMapElement::U4Align8B => 1,
        _ => 1,
    };
    let row_bytes = if element == TensorMapElement::U4Align8B {
        u64::from(k) / 2
    } else {
        u64::from(k) * byte_width
    };
    TensorMap {
        base_buffer: handle,
        dims: vec![u64::from(k), u64::from(rows)],
        strides_bytes: vec![byte_width, row_bytes],
        box_dims: vec![u32::from(tile_k), u32::from(tile_rows) * row_stride],
        element_strides: vec![1, row_stride],
        swizzle: if (element == TensorMapElement::Bf16 && tile_k > 64)
            || (element == TensorMapElement::Fp8E4M3 && tile_k > 128)
        {
            TensorMapSwizzle::None
        } else {
            TensorMapSwizzle::B128
        },
        interleave: TensorMapInterleave::None,
        l2_promotion: TensorMapL2Promotion::B128,
        oob_fill: TensorMapOobFill::Zero,
        element,
    }
}

/// A case's GEMM arguments: its fixtures' scales and global scales, and
/// the residual its epilogue adds.
fn case_arguments(
    variant: &GemmVariant,
    counts: RuntimeCounts,
    w: &Encoded,
    a: &Encoded,
) -> LabResult<kernel_core::contract::ArgumentList> {
    let maps = gemm::tensor_maps(&variant.key, WEIGHT, ACTIVATION).map_err(|e| e.to_string())?;
    gemm::arguments(
        &variant.key,
        counts,
        maps,
        GemmBuffers {
            output: OUTPUT,
            partial: PARTIAL,
            semaphores: SEMAPHORES,
            weight_scales: w.scales.as_ref().map(|_| WEIGHT_SCALES),
            activation_scales: a.scales.as_ref().map(|_| ACTIVATION_SCALES),
            residual: residual_binding(&query(variant)),
            weight_global_scale: w.global.map(|_| WEIGHT_GLOBAL),
            activation_global_scale: a.global.map(|_| ACTIVATION_GLOBAL),
        },
    )
    .map_err(|e| e.to_string())
}

/// The residual buffer a GEMM of `q` reads: one when its epilogue adds it.
fn residual_binding(q: &GemmQuery) -> Option<u64> {
    q.epilogues.residual_add.then_some(RESIDUAL)
}

/// Seeded F32 values in [-1/16, 1/16) with full 24-bit significands, so
/// most need more precision than BF16 holds: a kernel that rounded an F32
/// residual to BF16 before adding it would miss the reference.
fn residual_values(len: usize, seed: u64) -> Vec<f32> {
    let mut state = seed | 1;
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let unit = (state >> 40) as f32 / (1_u64 << 24) as f32;
            (unit - 0.5) / 8.0
        })
        .collect()
}

/// A residual of `len` output elements in the output's dtype (each row's
/// output columns, as the epilogue reads it): its bytes and the values the
/// GEMM adds. `zero` gives zeros, which every row adds exactly.
fn residual_fixture(len: usize, output: DType, zero: bool) -> (Vec<u8>, Vec<f64>) {
    let values = match zero {
        true => vec![0.0; len],
        false => residual_values(len, RESIDUAL_SEED),
    };
    match output {
        DType::F32 => (
            values.iter().flat_map(|v| v.to_ne_bytes()).collect(),
            values.into_iter().map(f64::from).collect(),
        ),
        _ => {
            let bits = values.into_iter().map(f32_to_bf16).collect::<Vec<_>>();
            (
                bits.iter().flat_map(|v| v.to_ne_bytes()).collect(),
                bits.into_iter()
                    .map(|v| f64::from(bf16_to_f32(v)))
                    .collect(),
            )
        }
    }
}

/// Adds the residual to a reference of the output elements `picks` (every
/// element when none): the epilogue adds it to the F32 sum before the
/// output rounding, one more term, so its magnitude joins the bound's sum
/// of absolute products.
fn add_residual(
    reference: &mut OracleOutput,
    picks: Option<&[usize]>,
    residual: &[f64],
) -> LabResult<()> {
    for index in 0..reference.values.len() {
        let element = picks.map_or(index, |picks| picks[index]);
        let value = *residual
            .get(element)
            .ok_or("a reference element past the residual")?;
        reference.values[index] += value;
        reference.sum_abs_products[index] += value.abs();
    }
    Ok(())
}

fn upload(session: &Session, data: &[u8]) -> LabResult<DeviceBuffer> {
    let buffer = session.allocate(data.len())?;
    buffer
        .copy_from_bytes(data, &session.stream)
        .map_err(|e| e.to_string())?;
    Ok(buffer)
}

pub(crate) struct Case {
    variant: GemmVariant,
    counts: RuntimeCounts,
    plan: kernel_core::contract::LaunchPlan,
    args: kernel_core::contract::ArgumentList,
    buffers: BTreeMap<u64, DeviceBuffer>,
    reference: Option<OracleOutput>,
    output_len: usize,
    /// Lean Qwen3.8 runs and large MiMo TP2 cases: the output
    /// elements the sampled oracle covers, in reference order.
    picks: Option<Vec<usize>>,
    prepared: bool,
}

const WEIGHT_SEED: u64 = 0x0047_5745_4947_4854;

/// A BF16 weight of `Case::prepare`'s seed: its bits and the bytes the GEMM
/// reads (fragment-major if `fragment`).
struct LeanWeight {
    key: (usize, usize, bool),
    bits: Vec<u16>,
    data: Vec<u8>,
}

thread_local! {
    /// The last lean weight: the Qwen3.8 head's buckets and prefills share
    /// one [vocab shard, 2560] weight, 318 M elements at TP2.
    static LEAN_WEIGHT: std::cell::RefCell<Option<Rc<LeanWeight>>> =
        const { std::cell::RefCell::new(None) };
}

fn lean_weight(n: usize, k: usize, fragment: bool) -> LabResult<Rc<LeanWeight>> {
    let key = (n, k, fragment);
    if let Some(cached) = LEAN_WEIGHT.with_borrow(|c| c.clone().filter(|w| w.key == key)) {
        return Ok(cached);
    }
    LEAN_WEIGHT.with_borrow_mut(|c| *c = None);
    let bits = seeded_values(n * k, WEIGHT_SEED)
        .into_iter()
        .map(f32_to_bf16)
        .collect::<Vec<_>>();
    let mut data = bits
        .iter()
        .flat_map(|v| v.to_ne_bytes())
        .collect::<Vec<_>>();
    if fragment {
        data = FragmentWeightLayout {
            rows: n,
            k,
            format: FragmentWeightFormat::Bf16,
        }
        .pack(&data)
        .map_err(|error| error.to_string())?
        .bytes;
    }
    let weight = Rc::new(LeanWeight { key, bits, data });
    LEAN_WEIGHT.with_borrow_mut(|c| *c = Some(weight.clone()));
    Ok(weight)
}

/// Lean output columns of a fragment GEMM: every 61st, the first and last
/// (a vocabulary shard's first and last row; a SwiGLU output's first gate
/// and up rows and its last), and both sides of every 64-row boundary (the
/// 64-row decode and 128-row prefill tiles, four and eight 16-row strips).
/// Outputs of at most 1024 columns (the SwiGLU shared expert) all run.
fn lean_columns(columns: usize) -> Vec<usize> {
    if columns <= 1024 {
        return (0..columns).collect();
    }
    kernel_lab::sample_indices(columns, 61, &(64..columns).step_by(64).collect::<Vec<_>>())
}

/// Output columns for a sampled oracle: about 256 of `n`, on a diagonal so
/// that every residue modulo the stride appears, plus the last column.
fn sampled_columns(n: usize) -> Vec<usize> {
    let stride = n.div_ceil(256).max(1);
    let mut columns = (0..n)
        .filter(|&c| c % stride == (c / stride) % stride)
        .collect::<Vec<_>>();
    if columns.last() != Some(&(n - 1)) {
        columns.push(n - 1);
    }
    columns
}

struct GraphLayer {
    weight: DeviceBuffer,
    output: DeviceBuffer,
    partial: DeviceBuffer,
    semaphores: DeviceBuffer,
}

impl Case {
    pub(crate) fn prepare(
        session: &Session,
        symbol: &str,
        m: u32,
        correctness: bool,
    ) -> LabResult<Self> {
        Self::prepare_with(session, symbol, m, correctness, false)
    }

    /// A checked case: at a lean Qwen3.8 topology (`kernel_lab::qwen38_lean`)
    /// every row's oracle covers `lean_columns` only; otherwise as `prepare`.
    fn prepare_checked(session: &Session, symbol: &str, m: u32) -> LabResult<Self> {
        let qwen = symbol.starts_with("chop_gemm_qwen38_");
        Self::prepare_with(session, symbol, m, true, qwen && kernel_lab::qwen38_lean()?)
    }

    fn prepare_with(
        session: &Session,
        symbol: &str,
        m: u32,
        correctness: bool,
        lean: bool,
    ) -> LabResult<Self> {
        let variant = gemm::catalog(kernel_lab::catalogs())
            .into_iter()
            .find(|item| item.symbol == symbol)
            .ok_or_else(|| format!("missing GEMM catalog symbol {symbol}"))?;
        Self::prepare_variant(session, variant, m, correctness, lean)
    }

    /// A case of `variant`: a catalog entry or a lab-only variant of one
    /// (`gemm::lab_split_catalog`), launched from the entry's cubin.
    fn prepare_variant(
        session: &Session,
        variant: GemmVariant,
        m: u32,
        correctness: bool,
        lean: bool,
    ) -> LabResult<Self> {
        Self::prepare_variant_sampled(session, variant, m, correctness, lean, false)
    }

    fn prepare_variant_sampled(
        session: &Session,
        variant: GemmVariant,
        m: u32,
        correctness: bool,
        lean: bool,
        sampled: bool,
    ) -> LabResult<Self> {
        let symbol = variant.symbol.clone();
        let q = query(&variant);
        let n = q.n as usize;
        let k = q.k as usize;
        let capacity = q.m_capacity as usize;
        if m > q.m_capacity {
            return Err(format!("M={m} exceeds {} capacity", q.m_capacity));
        }
        let lean_weight = if lean {
            if q.weight != WeightFormat::Bf16 || q.activation != ActivationFormat::Bf16 {
                return Err(format!("{symbol}: lean cases are BF16"));
            }
            Some(lean_weight(
                n,
                k,
                gemm::fragment_weight_enabled(&variant.key),
            )?)
        } else {
            None
        };
        let (w, a, raw_activation) = if sampled {
            let w = generated_host::weight_fixture(&variant)?;
            let a = generated_host::activation_fixture(capacity, k, m as usize, q.activation)?;
            (w, a, Vec::new())
        } else {
            let mut w = match lean_weight {
                Some(_) => encoded_bf16(&[], false),
                None => seeded_weight(n, k, q.weight, correctness)?,
            };
            let mut raw_activation = vec![0.0_f32; capacity * k];
            raw_activation[..m as usize * k]
                .copy_from_slice(&seeded_values(m as usize * k, 0x4143_5449_5641_5445));
            let a = activation(
                &raw_activation,
                capacity,
                k,
                q.activation,
                correctness && !lean,
            )?;
            if correctness {
                let order = kernels_sm120::precision::dense_decode_order(
                    &q,
                    gemm::fragment_weight_enabled(&variant.key),
                );
                for value in &mut w.decoded {
                    *value = generated_host::oracle_weight(*value, order);
                }
            }
            if lean_weight.is_none() && gemm::fragment_weight_enabled(&variant.key) {
                let format = if q.weight == WeightFormat::Bf16 {
                    FragmentWeightFormat::Bf16
                } else {
                    FragmentWeightFormat::Fp8E4M3
                };
                w.data = FragmentWeightLayout { rows: n, k, format }
                    .pack(&w.data)
                    .map_err(|error| error.to_string())?
                    .bytes;
            }
            (Rc::new(w), a, raw_activation)
        };
        let splits = gemm::choose_split_k(&variant.key).map_err(|e| e.to_string())?;
        let counts = RuntimeCounts {
            m,
            batch: m,
            context: 0,
            routed_rows: 0,
            split_k: splits,
        };
        let plan = gemm::plan(&variant.key, counts).map_err(|e| e.to_string())?;
        let output_n = if q.epilogues.swiglu.is_some() {
            n / 2
        } else {
            n
        };
        let output_len = m as usize * output_n;
        let mut buffers = BTreeMap::new();
        buffers.insert(
            WEIGHT,
            upload(session, lean_weight.as_ref().map_or(&w.data, |c| &c.data))?,
        );
        buffers.insert(ACTIVATION, upload(session, &a.data)?);
        let output = session.allocate(output_len * if q.output == DType::F32 { 4 } else { 2 })?;
        let partial = session.allocate(plan.workspace_bytes as usize)?;
        poison::poison(session, &[&output, &partial])?;
        buffers.insert(OUTPUT, output);
        buffers.insert(PARTIAL, partial);
        let residual = match residual_binding(&q) {
            Some(_) if q.epilogues.swiglu.is_some() => {
                return Err(format!("{symbol}: the lab has no residual SwiGLU oracle"));
            }
            Some(_) => {
                let (data, values) = residual_fixture(output_len, q.output, false);
                buffers.insert(RESIDUAL, upload(session, &data)?);
                Some(values)
            }
            None => None,
        };
        // Prefill clock64 profiles store samples after the first 8 KiB.
        buffers.insert(
            SEMAPHORES,
            upload(
                session,
                &vec![0; (plan.workspace_bytes as usize).max(1 << 16)],
            )?,
        );
        if let Some(scales) = &w.scales {
            buffers.insert(WEIGHT_SCALES, upload(session, scales)?);
        }
        if let Some(scales) = &a.scales {
            buffers.insert(ACTIVATION_SCALES, upload(session, scales)?);
        }
        if let Some(global) = w.global {
            buffers.insert(WEIGHT_GLOBAL, upload(session, &global.to_ne_bytes())?);
        }
        if let Some(global) = a.global {
            buffers.insert(ACTIVATION_GLOBAL, upload(session, &global.to_ne_bytes())?);
        }
        let args = case_arguments(&variant, counts, &w, &a)?;
        let mut picks = None;
        let reference = if sampled && correctness {
            let (reference, indices) = sampled_reference(&variant, m, &w, &a, true)?;
            picks = Some(indices);
            Some(reference)
        } else if let Some(cached) = lean_weight.filter(|_| correctness) {
            // Every row at the sampled columns: their weight rows (a SwiGLU
            // output's gate row `c` and up row `c + output_n`) as a smaller
            // GEMM.
            let columns = lean_columns(output_n);
            let weight_rows = match q.epilogues.swiglu {
                Some(SwigluColumns::ConcatenatedHalves) => columns
                    .iter()
                    .copied()
                    .chain(columns.iter().map(|c| c + output_n))
                    .collect(),
                Some(_) => return Err(format!("{symbol}: lean SwiGLU halves only")),
                None => columns.clone(),
            };
            let b = weight_rows
                .iter()
                .flat_map(|&r| &cached.bits[r * k..(r + 1) * k])
                .map(|&x| f64::from(bf16_to_f32(x)))
                .collect::<Vec<_>>();
            let a = raw_activation[..m as usize * k]
                .iter()
                .map(|&v| f64::from(bf16_to_f32(f32_to_bf16(v))))
                .collect::<Vec<_>>();
            picks = Some(
                (0..m as usize)
                    .flat_map(|t| columns.iter().map(move |&c| t * output_n + c))
                    .collect(),
            );
            Some(
                oracle::gemm(
                    &a,
                    &b,
                    GemmShape {
                        m: m as usize,
                        n: weight_rows.len(),
                        k,
                        experts: 1,
                    },
                    &[],
                    GemmEpilogue {
                        residual: None,
                        swiglu: q.epilogues.swiglu,
                    },
                )
                .map_err(|e| e.to_string())?,
            )
        } else if let Some(columns) = (correctness
            && variant
                .profile
                .ranks_of(model_mimo26_contract::sm120::config::MODEL)
                .is_some_and(|ranks| ranks != 4)
            && m as usize * n * k > 1 << 30)
            .then(|| sampled_columns(output_n))
        {
            // MiMo's TP2 shapes take seconds to minutes in the f64
            // oracle; sample their output columns instead. A SwiGLU output
            // column takes its gate and up weight rows.
            let rows: Vec<usize> = match q.epilogues.swiglu {
                None => columns.clone(),
                Some(SwigluColumns::ConcatenatedHalves) => columns
                    .iter()
                    .copied()
                    .chain(columns.iter().map(|&c| c + output_n))
                    .collect(),
                Some(SwigluColumns::Interleaved) => {
                    columns.iter().flat_map(|&c| [2 * c, 2 * c + 1]).collect()
                }
            };
            let b = rows
                .iter()
                .flat_map(|&row| w.decoded[row * k..(row + 1) * k].iter().copied())
                .collect::<Vec<_>>();
            picks = Some(
                (0..m as usize)
                    .flat_map(|t| columns.iter().map(move |&c| t * output_n + c))
                    .collect(),
            );
            Some(
                oracle::gemm(
                    &a.decoded[..m as usize * k],
                    &b,
                    GemmShape {
                        m: m as usize,
                        n: rows.len(),
                        k,
                        experts: 1,
                    },
                    &[],
                    GemmEpilogue {
                        residual: None,
                        swiglu: q.epilogues.swiglu,
                    },
                )
                .map_err(|e| e.to_string())?,
            )
        } else if correctness {
            Some(
                oracle::gemm(
                    &a.decoded[..m as usize * k],
                    &w.decoded,
                    GemmShape {
                        m: m as usize,
                        n,
                        k,
                        experts: 1,
                    },
                    &[],
                    GemmEpilogue {
                        residual: None,
                        swiglu: q.epilogues.swiglu,
                    },
                )
                .map_err(|e| e.to_string())?,
            )
        } else {
            None
        };
        let mut reference = reference;
        if let (Some(reference), Some(residual)) = (reference.as_mut(), &residual) {
            add_residual(reference, picks.as_deref(), residual)?;
        }
        Ok(Self {
            variant,
            counts,
            plan,
            args,
            buffers,
            reference,
            output_len,
            picks,
            prepared: false,
        })
    }

    /// The TP4 QKV entry `block_symbol` at `m` rows, run as its 64-row scale
    /// probe: the same weights and activations, each 128-row block scale
    /// repeated for its two 64-row groups.
    fn rows64_twin(session: &Session, block_symbol: &str, m: u32) -> LabResult<Self> {
        let probe_symbol = block_symbol.replace("_v1", "_rows64_v1");
        let probe = model_mimo26_contract::sm120::lab::rows64_probe_catalog()
            .into_iter()
            .find(|item| item.symbol == probe_symbol)
            .ok_or_else(|| format!("missing probe {probe_symbol}"))?;
        let mut case = Case::prepare(session, block_symbol, m, false)?;
        let q = query(&case.variant);
        let (n, k_blocks) = (q.n as usize, q.k as usize / 128);
        let blocks =
            session.download_f32(&case.buffers[&WEIGHT_SCALES], n.div_ceil(128) * k_blocks)?;
        let rows64 = (0..n / 64)
            .flat_map(|group| blocks[group / 2 * k_blocks..(group / 2 + 1) * k_blocks].to_vec())
            .collect::<Vec<_>>();
        case.buffers
            .insert(WEIGHT_SCALES, session.upload_f32(&rows64)?);
        case.plan = gemm::plan(&probe.key, case.counts).map_err(|e| e.to_string())?;
        let maps = gemm::tensor_maps(&probe.key, WEIGHT, ACTIVATION).map_err(|e| e.to_string())?;
        case.args = gemm::arguments(
            &probe.key,
            case.counts,
            maps,
            GemmBuffers {
                output: OUTPUT,
                partial: PARTIAL,
                semaphores: SEMAPHORES,
                weight_scales: Some(WEIGHT_SCALES),
                activation_scales: None,
                residual: None,
                weight_global_scale: None,
                activation_global_scale: None,
            },
        )
        .map_err(|e| e.to_string())?;
        case.variant = probe;
        Ok(case)
    }

    pub(crate) fn function(&self, session: &Session) -> LabResult<Function> {
        if self.prepared {
            return kernel_lab::projection_function(
                session,
                &self.variant.symbol,
                self.plan.dynamic_shared_bytes,
            )?
            .ok_or("missing prepared dense function".into());
        }
        let module = session.module(&match self.variant.profile {
            _ if self.variant.symbol.ends_with("_rows64_v1") => "gemm_rows64_probe_sm120".into(),
            profile => format!("gemm_{}_sm120", profile.tag()),
        })?;
        let function = module
            .function(&self.variant.symbol)
            .map_err(|e| e.to_string())?;
        if self.plan.dynamic_shared_bytes >= 48 * 1024 {
            function
                .set_max_dynamic_shared_memory(self.plan.dynamic_shared_bytes)
                .map_err(|e| e.to_string())?;
        }
        Ok(function)
    }

    pub(crate) fn launch(
        &self,
        session: &Session,
        function: &Function,
        stream: &Stream,
    ) -> LabResult<()> {
        self.launch_layer(session, function, stream, None)
    }

    fn launch_layer(
        &self,
        session: &Session,
        function: &Function,
        stream: &Stream,
        layer: Option<&GraphLayer>,
    ) -> LabResult<()> {
        let mut buffers = self
            .buffers
            .iter()
            .map(|(&id, buffer)| (id, buffer))
            .collect::<BTreeMap<_, _>>();
        if let Some(layer) = layer {
            buffers.insert(WEIGHT, &layer.weight);
            buffers.insert(OUTPUT, &layer.output);
            buffers.insert(PARTIAL, &layer.partial);
            buffers.insert(SEMAPHORES, &layer.semaphores);
        }
        launch::launch(
            &session.context,
            function,
            stream,
            self.plan,
            &self.args,
            &Bindings { buffers },
        )
    }

    /// Bytes per output element.
    fn output_bytes(&self) -> usize {
        if query(&self.variant).output == DType::F32 {
            4
        } else {
            2
        }
    }

    fn output_dtype(&self) -> DType {
        if query(&self.variant).output == DType::F32 {
            DType::F32
        } else {
            DType::Bf16
        }
    }

    /// Poisoned checks: every output element is written, and the F32 split
    /// partials that the last or owning CTA sums in the same launch are
    /// scratch. The self-resetting semaphores are never touched.
    fn replay<'a>(&'a self, session: &'a Session) -> LabResult<Replay<'a>> {
        let output = &self.buffers[&OUTPUT];
        Ok(Replay::new(session)
            .output(
                "gemm output",
                output,
                WriteMap::of(self.output_dtype()).byte_prefix(output.byte_len())?,
            )
            .scratch("gemm partials", &self.buffers[&PARTIAL]))
    }

    /// Poisons `buffers` (outputs and split partials) before a launch.
    fn poison(&self, session: &Session, buffers: &[&DeviceBuffer]) -> LabResult<()> {
        poison::poison(session, buffers)
    }

    /// The values of an output launched on poison: every element must be
    /// written.
    fn written(
        &self,
        session: &Session,
        buffer: &DeviceBuffer,
        label: &str,
    ) -> LabResult<Vec<f32>> {
        let bytes = poison::read(session, buffer)?;
        poison::check(
            &format!("{} {label}", self.variant.symbol),
            &bytes,
            &WriteMap::of(self.output_dtype()).byte_prefix(bytes.len())?,
            Fill::Pattern(&[POISON]),
        )?;
        Ok(if self.output_bytes() == 4 {
            bytes
                .chunks_exact(4)
                .map(|value| f32::from_ne_bytes(value.try_into().expect("four bytes")))
                .collect()
        } else {
            bytes
                .chunks_exact(2)
                .map(|value| bf16_to_f32(u16::from_ne_bytes([value[0], value[1]])))
                .collect()
        })
    }

    fn output(&self, session: &Session) -> LabResult<Vec<f32>> {
        let buffer = &self.buffers[&OUTPUT];
        if query(&self.variant).output == DType::F32 {
            session.download_f32(buffer, self.output_len)
        } else {
            Ok(session
                .download_bf16(buffer, self.output_len)?
                .into_iter()
                .map(bf16_to_f32)
                .collect())
        }
    }

    fn receipt(&self, session: &Session, actual: &[f32], label: &str) -> LabResult<()> {
        let mut receipt = BTreeMap::new();
        receipt.insert("case".into(), json!(label));
        receipt.insert("symbol".into(), json!(self.variant.symbol));
        receipt.insert("profile".into(), json!(self.variant.profile.as_str()));
        receipt.insert("m".into(), json!(self.counts.m));
        receipt.insert("split_k".into(), json!(self.counts.split_k));
        record_qwen38_ranks(self.variant.profile, &mut receipt)?;
        let sampled;
        let actual = match &self.picks {
            Some(picks) => {
                receipt.insert("sampled_outputs".into(), json!(picks.len()));
                sampled = picks.iter().map(|&i| actual[i]).collect::<Vec<_>>();
                &sampled[..]
            }
            None => actual,
        };
        let gate = check_output(
            actual,
            self.reference.as_ref().ok_or("missing GEMM oracle")?,
            query(&self.variant).output,
            &mut receipt,
        );
        session.receipt("gemm", receipt.clone())?;
        gate.map_err(|error| {
            format!(
                "{}: {error}; worst={}",
                self.variant.symbol, receipt["worst_element"]
            )
        })
    }

    fn bitwise_receipt(
        &self,
        session: &Session,
        expected: &[f32],
        actual: &[f32],
        label: &str,
        evidence: &Evidence,
    ) -> LabResult<()> {
        let worst = expected
            .iter()
            .zip(actual)
            .enumerate()
            .filter(|(_, (left, right))| left.to_bits() != right.to_bits())
            .max_by(|(_, (left_a, right_a)), (_, (left_b, right_b))| {
                (**left_a - **right_a)
                    .abs()
                    .total_cmp(&(**left_b - **right_b).abs())
            });
        let mut receipt = BTreeMap::new();
        receipt.insert("case".into(), json!(label));
        receipt.insert("symbol".into(), json!(self.variant.symbol));
        receipt.insert("correct".into(), json!(worst.is_none()));
        record_qwen38_ranks(self.variant.profile, &mut receipt)?;
        evidence.record(&mut receipt);
        if let Some((index, (&left, &right))) = worst {
            receipt.insert(
                "worst_element".into(),
                json!({"index": index, "eager_bits": left.to_bits(),
                    "replay_bits": right.to_bits(), "eager": left, "replay": right}),
            );
        }
        session.receipt("gemm", receipt.clone())?;
        if worst.is_some() {
            return Err(format!(
                "{label}: bitwise mismatch; worst={}",
                receipt["worst_element"]
            ));
        }
        Ok(())
    }
}

fn run_correctness(session: &Session, profile: Profile) -> LabResult<()> {
    for m in [1_u32, 8, 32, 512, 2048] {
        for symbol in correctness_symbols(profile, m) {
            let case = Case::prepare_checked(session, &symbol, m)?;
            let function = case.function(session)?;
            case.launch(session, &function, &session.stream)?;
            let actual = case.output(session)?;
            case.receipt(session, &actual, "dense_gemm_correctness")?;
        }
    }
    Ok(())
}

/// Records the Qwen3.8 topology (`CHOPINFER_QWEN38_RANKS`) in a Qwen
/// receipt when it is not TP4's, and that the run is lean (its oracles on
/// sampled columns).
fn record_qwen38_ranks(
    profile: Profile,
    receipt: &mut BTreeMap<String, serde_json::Value>,
) -> LabResult<()> {
    if profile
        .ranks_of(model_qwen38_contract::sm120::config::MODEL)
        .is_some()
    {
        let ranks = kernel_lab::qwen38_ranks()?;
        if ranks != 4 {
            receipt.insert("qwen38_ranks".into(), json!(ranks));
        }
        if kernel_lab::qwen38_lean()? {
            receipt.insert("lean".into(), json!(true));
        }
    }
    Ok(())
}

fn correctness_symbols(profile: Profile, m: u32) -> Vec<String> {
    match (profile, m <= 32) {
        (model_mimo26_contract::sm120::config::PROFILE, true) => {
            vec![format!("chop_gemm_mimo_qkv_ga_m{m}_v1")]
        }
        (model_mimo26_contract::sm120::config::PROFILE, false) => {
            vec!["chop_gemm_mimo_router_prefill_v1".into()]
        }
        // Every TP2 shape the bundle selects, at decode and prefill.
        (p, decode)
            if p.ranks_of(model_mimo26_contract::sm120::config::MODEL)
                .is_some_and(|ranks| ranks != 4) =>
        {
            let tag = profile.tag();
            model_mimo26_contract::sm120::gemm::rank_shapes(
                profile
                    .ranks_of(model_mimo26_contract::sm120::config::MODEL)
                    .expect("MiMo profile"),
            )
            .into_iter()
            .map(|(name, ..)| {
                if decode {
                    format!("chop_gemm_{tag}_{name}_m{m}_v1")
                } else {
                    format!("chop_gemm_{tag}_{name}_prefill_v1")
                }
            })
            .collect()
        }
        (Profile::StdDenseMoe, true) => vec![format!("chop_gemm_std_dense_fp8_m{m}_v1")],
        (Profile::StdDenseMoe, false) => vec!["chop_gemm_std_router_prefill_v1".into()],
        (model_qwen38_contract::sm120::config::PROFILE, true) => vec![
            format!("chop_gemm_qwen38_shared_gate_up_m{m}_v1"),
            format!("chop_gemm_qwen38_shared_down_m{m}_v1"),
            format!("chop_gemm_qwen38_head_m{m}_v1"),
            format!("chop_gemm_qwen38_head_f32_m{m}_v1"),
        ],
        (model_qwen38_contract::sm120::config::PROFILE, false) => vec![
            "chop_gemm_qwen38_shared_gate_up_prefill_v1".into(),
            "chop_gemm_qwen38_shared_down_prefill_v1".into(),
            "chop_gemm_qwen38_router_prefill_v1".into(),
        ],
        // The router is TP4's at every rank count; the head is per rank.
        (p, true)
            if p.ranks_of(model_qwen38_contract::sm120::config::MODEL)
                .is_some_and(|ranks| ranks != 4) =>
        {
            let tag = profile.tag();
            vec![
                format!("chop_gemm_{tag}_shared_gate_up_m{m}_v1"),
                format!("chop_gemm_{tag}_shared_down_m{m}_v1"),
                format!("chop_gemm_{tag}_head_m{m}_v1"),
            ]
        }
        // The head prefill at 2048 rows is 651 G f64 multiply-adds at TP2,
        // past this test's watchdog; `gemm_qwen38_shapes` runs it and every
        // other per-rank prefill at fewer rows. `out` is the cheapest
        // prefill whose K changes with the rank count.
        (p, false)
            if p.ranks_of(model_qwen38_contract::sm120::config::MODEL)
                .is_some_and(|ranks| ranks != 4) =>
        {
            let tag = profile.tag();
            vec![
                format!("chop_gemm_{tag}_shared_gate_up_prefill_v1"),
                format!("chop_gemm_{tag}_shared_down_prefill_v1"),
                format!("chop_gemm_{tag}_out_prefill_v1"),
            ]
        }
        (p, _) => panic!("no correctness symbols for {}", p.as_str()),
    }
}

#[test]
fn gemm_catalog_case_coverage() {
    let all = gemm::catalog(kernel_lab::catalogs());
    let specs = gemm::cubin_specs(kernel_lab::catalogs(), None);
    // One per served profile, and std.
    assert_eq!(specs.len(), 5);
    for profile in kernel_lab::catalogs()
        .profiles()
        .into_iter()
        .filter(|&profile| all.iter().any(|item| item.profile == profile))
    {
        let selected = gemm::cubin_specs(kernel_lab::catalogs(), Some(profile));
        assert_eq!(selected.len(), 1);
        let symbols = selected[0]
            .entries
            .iter()
            .map(|entry| entry.symbol.as_str())
            .collect::<std::collections::HashSet<_>>();
        let expected = all.iter().filter(|item| item.profile == profile).count();
        assert_eq!(symbols.len(), expected);
        for m in [1_u32, 8, 32, 512, 2048] {
            for symbol in correctness_symbols(profile, m) {
                assert!(symbols.contains(symbol.as_str()), "missing {symbol}");
            }
        }
    }
}

/// Neighbouring 64-row groups and 128-row blocks of the seeded FP8 weights
/// carry different scales, so a kernel that reads the wrong scale row fails
/// the GEMM checks.
#[test]
fn gemm_seeded_fp8_scales_differ_by_row_group() {
    let scales = |format| -> Vec<f32> {
        let encoded = seeded_weight(640, 256, format, false).unwrap();
        encoded
            .scales
            .unwrap()
            .chunks_exact(4)
            .map(|b| f32::from_ne_bytes(b.try_into().unwrap()))
            .collect()
    };
    for (format, rows_per_scale) in [
        (WeightFormat::Fp8Rows64x128(ScaleMode::ExactF32), 64),
        (WeightFormat::Fp8Block128x128(ScaleMode::ExactF32), 128),
    ] {
        let scales = scales(format);
        let first_column = scales.iter().step_by(2).collect::<Vec<_>>();
        assert_eq!(first_column.len(), 640 / rows_per_scale);
        assert!(
            first_column.windows(2).all(|pair| pair[0] != pair[1]),
            "{format:?}: {first_column:?}"
        );
    }
}

#[test]
fn gemm_sampled_columns_cover_every_residue() {
    for n in [1856_usize, 4096, 6784, 7424] {
        let columns = sampled_columns(n);
        let stride = n.div_ceil(256);
        assert!(columns.len() <= 257 + stride, "{n}");
        assert_eq!(columns.last(), Some(&(n - 1)));
        assert!(columns.windows(2).all(|pair| pair[0] < pair[1]));
        for residue in 0..stride {
            assert!(
                columns.iter().any(|c| c % stride == residue),
                "{n} {residue}"
            );
        }
    }
}

#[test]
fn gemm_compact_fp4_map_matches_reference_bytes() {
    let values = seeded_values(256 * 256, 0x0046_5034_4d41_5053);
    for quantized in [
        quantize::mxfp4(&values, 256, 256).unwrap(),
        quantize::nvfp4(&values, 256, 256).unwrap(),
    ] {
        let encoded = encoded_quantized(quantized, true);
        let spec = map(WEIGHT, 256, 256, 64, 256, encoded.element, 1);
        let descriptor = launch::descriptor(&spec).unwrap();
        assert_eq!(encoded.data.len(), 256 * 256 / 2);
        assert_eq!(
            descriptor.data_type,
            runtime_cuda::TensorMapDataType::U4Align8B
        );
        assert_eq!(descriptor.global_dims[..2], [256, 256]);
        assert_eq!(descriptor.global_strides_bytes[0], 128);
        assert_eq!(descriptor.box_dims[..2], [256, 64]);
    }
}

#[test]
fn gemm_correctness_mimo_v26_flash() {
    watched_gpu_test(1024 * 1024 * 1024, 240, |session| {
        run_correctness(session, model_mimo26_contract::sm120::config::PROFILE)
    })
    .expect("MiMo dense GEMM correctness");
}

#[test]
fn gemm_correctness_mimo_v26_flash_tp2() {
    watched_gpu_test(1024 * 1024 * 1024, 600, |session| {
        run_correctness(session, model_mimo26_contract::sm120::config::MODEL.at(2))
    })
    .expect("MiMo TP2 dense GEMM correctness");
}

/// 64-row FP8 scales that repeat each 128-row block scale reproduce the
/// block-scaled TP4 QKV entries bit for bit, in decode and prefill.
#[test]
fn gemm_rows64_scales_reproduce_block128_bitwise() {
    watched_gpu_test(1024 * 1024 * 1024, 600, |session| {
        for probe in model_mimo26_contract::sm120::lab::rows64_probe_catalog() {
            let block_symbol = probe.symbol.replace("_rows64_v1", "_v1");
            let capacity = query(&probe).m_capacity;
            let rows: &[u32] = if capacity <= 32 {
                &[capacity]
            } else {
                &[1, 200, 2048]
            };
            for &m in rows {
                let mut outputs = Vec::new();
                for case in [
                    Case::prepare(session, &block_symbol, m, false)?,
                    Case::rows64_twin(session, &block_symbol, m)?,
                ] {
                    let function = case.function(session)?;
                    case.poison(session, &[&case.buffers[&OUTPUT], &case.buffers[&PARTIAL]])?;
                    case.launch(session, &function, &session.stream)?;
                    outputs.push(case.written(session, &case.buffers[&OUTPUT], "rows64 probe")?);
                }
                let bits = |values: &[f32]| values.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
                if bits(&outputs[0]) != bits(&outputs[1]) {
                    return Err(format!(
                        "{} M={m} differs from {block_symbol}",
                        probe.symbol
                    ));
                }
            }
        }
        Ok(())
    })
    .expect("64-row scale probe bitwise equality");
}

#[test]
fn gemm_correctness_std_dense_moe() {
    watched_gpu_test(1024 * 1024 * 1024, 240, |session| {
        run_correctness(session, Profile::StdDenseMoe)
    })
    .expect("standard dense GEMM correctness");
}

#[test]
fn gemm_correctness_qwen38_flash() {
    watched_gpu_test(1024 * 1024 * 1024, 240, |session| {
        run_correctness(session, kernel_lab::qwen38_profile()?)
    })
    .expect("Qwen3.8 dense GEMM correctness");
}

#[test]
fn gemm_swiglu_layouts() {
    watched_gpu_test(1024 * 1024 * 1024, 240, |session| {
        for (symbol, m) in [
            ("chop_gemm_mimo_mlp_gate_up_m1_v1", 1),
            ("chop_gemm_mimo_mlp_gate_up_prefill_v1", 128),
            ("chop_gemm_format_swiglu_interleaved_decode_v1", 1),
            ("chop_gemm_format_swiglu_interleaved_prefill_v1", 128),
        ] {
            let case = Case::prepare(session, symbol, m, true)?;
            let function = case.function(session)?;
            case.launch(session, &function, &session.stream)?;
            let actual = case.output(session)?;
            case.receipt(session, &actual, "dense_gemm_swiglu")?;
        }
        Ok(())
    })
    .expect("fused SwiGLU column layouts");
}

#[test]
fn gemm_fragment_prefill_formats() {
    watched_gpu_test(1024 * 1024 * 1024, 480, |session| {
        // Partial token tiles, one-token prefill, stream-K only schedules
        // (fewer tiles than SMs) and whole waves plus a stream-K tail
        // (o at M=770 and head at M=128).
        for (symbol, m) in [
            ("chop_gemm_mimo_qkv_ga_prefill_v1", 128),
            ("chop_gemm_mimo_qkv_ga_prefill_v1", 200),
            ("chop_gemm_mimo_qkv_swa_prefill_v1", 1),
            ("chop_gemm_mimo_o_prefill_v1", 128),
            ("chop_gemm_mimo_o_prefill_v1", 770),
            ("chop_gemm_mimo_router_prefill_v1", 333),
            ("chop_gemm_mimo_router_prefill_v1", 1),
            ("chop_gemm_mimo_router_prefill_v1", 33),
            ("chop_gemm_mimo_router_prefill_v1", 128),
            ("chop_gemm_mimo_router_prefill_v1", 129),
            ("chop_gemm_mimo_head_prefill_v1", 128),
            ("chop_gemm_mimo_mlp_gate_up_prefill_v1", 200),
            ("chop_gemm_mimo_mlp_down_prefill_v1", 130),
        ] {
            let case = Case::prepare(session, symbol, m, true)?;
            let function = case.function(session)?;
            case.launch(session, &function, &session.stream)?;
            let actual = case.output(session)?;
            case.receipt(session, &actual, "dense_gemm_fragment_prefill")?;
        }
        Ok(())
    })
    .expect("fragment-major BF16 and FP8 prefill");
}

#[test]
fn gemm_fragment_decode_shapes() {
    watched_gpu_test(1024 * 1024 * 1024, 240, |session| {
        for (symbol, m) in [
            ("chop_gemm_mimo_qkv_ga_m2_v1", 2),
            ("chop_gemm_mimo_qkv_ga_m4_v1", 3),
            ("chop_gemm_mimo_qkv_ga_m8_v1", 5),
            ("chop_gemm_mimo_o_m1_v1", 1),
            ("chop_gemm_mimo_o_m8_v1", 8),
            ("chop_gemm_mimo_mlp_gate_up_m1_v1", 1),
            ("chop_gemm_mimo_mlp_gate_up_m8_v1", 7),
            ("chop_gemm_mimo_mlp_down_m1_v1", 1),
            ("chop_gemm_mimo_o_m16_v1", 16),
            ("chop_gemm_mimo_o_m32_v1", 32),
            ("chop_gemm_mimo_router_m1_v1", 1),
            ("chop_gemm_mimo_router_m8_v1", 8),
            ("chop_gemm_mimo_router_m32_v1", 32),
            ("chop_gemm_mimo_head_m1_v1", 1),
            ("chop_gemm_mimo_qkv_ga_m16_v1", 16),
            ("chop_gemm_mimo_qkv_swa_m16_v1", 16),
            ("chop_gemm_mimo_qkv_swa_m32_v1", 32),
            ("chop_gemm_mimo_mlp_gate_up_m16_v1", 16),
            ("chop_gemm_mimo_mlp_gate_up_m32_v1", 32),
            ("chop_gemm_mimo_mlp_down_m16_v1", 16),
            ("chop_gemm_mimo_mlp_down_m32_v1", 32),
        ] {
            let case = Case::prepare(session, symbol, m, true)?;
            let function = case.function(session)?;
            case.launch(session, &function, &session.stream)?;
            let actual = case.output(session)?;
            case.receipt(session, &actual, "dense_gemm_fragment_decode")?;
        }
        Ok(())
    })
    .expect("fragment-major MiMo decode shapes");
}

/// The lab-only split variants: MiMo and Qwen3.8 fragment decode entries at
/// the multi-split reductions kernel-lab ran before they declared their
/// served single split, against the CPU oracle. Their splits do not depend
/// on live M, so one row per bucket bounds the oracles of the wide heads.
#[test]
fn gemm_lab_split_variants() {
    watched_gpu_test(1024 * 1024 * 1024, 600, |session| {
        let lean = kernel_lab::qwen38_lean()?;
        for variant in gemm::lab_split_catalog(kernel_lab::catalogs()) {
            let qwen = variant.symbol.starts_with("chop_gemm_qwen38_");
            let case = Case::prepare_variant(session, variant, 1, true, qwen && lean)?;
            let function = case.function(session)?;
            case.launch(session, &function, &session.stream)?;
            let actual = case.output(session)?;
            case.receipt(session, &actual, "dense_gemm_lab_splits")?;
        }
        Ok(())
    })
    .expect("lab split variants match the CPU oracle");
}

#[test]
fn gemm_dflash_bf16_shapes() {
    watched_gpu_test(1024 * 1024 * 1024, 120, |session| {
        for (symbol, m) in [
            ("chop_gemm_mimo_dflash_q_m2_v1", 2),
            ("chop_gemm_mimo_dflash_qkv_m2_v1", 2),
            ("chop_gemm_mimo_dflash_context_kv_m2_v1", 2),
            ("chop_gemm_mimo_dflash_o_m2_v1", 2),
            ("chop_gemm_mimo_dflash_gate_up_m2_v1", 2),
            ("chop_gemm_mimo_dflash_down_m2_v1", 2),
            ("chop_gemm_mimo_dflash_fc_m2_v1", 2),
            ("chop_gemm_mimo_dflash_qkv_prefill_v1", 1),
            ("chop_gemm_mimo_dflash_gate_up_prefill_v1", 1),
            ("chop_gemm_mimo_dflash_fc_prefill_v1", 1),
        ] {
            let case = Case::prepare(session, symbol, m, true)?;
            let function = case.function(session)?;
            case.launch(session, &function, &session.stream)?;
            let actual = case.output(session)?;
            case.receipt(session, &actual, "dflash_bf16")?;
        }
        Ok(())
    })
    .expect("DFlash BF16 GEMM instances match the CPU oracle");
}

/// MiMo verify buckets of 64, 128 and 256 rows run the prefill GEMMs with
/// the bucket as M, captured in the verify graph. Every target and DFlash
/// projection at each bucket, then graph replay and PDL at the widest.
#[test]
fn gemm_wide_verify_buckets() {
    watched_gpu_test(2 * 1024 * 1024 * 1024, 600, |session| {
        let names = [
            "qkv_ga",
            "qkv_swa",
            "o",
            "router",
            "head",
            "head_f32",
            "mlp_gate_up",
            "mlp_down",
            "dflash_q",
            "dflash_qkv",
            "dflash_o",
            "dflash_gate_up",
            "dflash_down",
        ];
        for m in [64_u32, 128, 256] {
            for name in names {
                let symbol = format!("chop_gemm_mimo_{name}_prefill_v1");
                let case = Case::prepare(session, &symbol, m, true)?;
                let function = case.function(session)?;
                case.launch(session, &function, &session.stream)?;
                let actual = case.output(session)?;
                case.receipt(session, &actual, &format!("dense_gemm_wide_verify_m{m}"))?;
            }
        }
        replay_cases(
            session,
            &[
                ("chop_gemm_mimo_qkv_swa_prefill_v1", 256),
                ("chop_gemm_mimo_head_f32_prefill_v1", 256),
                ("chop_gemm_mimo_dflash_gate_up_prefill_v1", 256),
            ],
            "dense_gemm_wide_verify",
        )
    })
    .expect("MiMo wide verify GEMMs");
}

#[test]
fn gemm_format_paths() {
    watched_gpu_test(256 * 1024 * 1024, 240, |session| {
        for (symbol, m) in [
            ("chop_gemm_format_fp8_tensor_a16_v1", 1),
            ("chop_gemm_format_fp8_channel_a16_v1", 1),
            ("chop_gemm_format_fp8_block_pow2_a16_v1", 1),
            ("chop_gemm_format_fp8_block_pow2_a8_v1", 1),
            ("chop_gemm_format_fp8_tensor_a8_v1", 1),
            ("chop_gemm_format_fp8_channel_a8_v1", 1),
            ("chop_gemm_format_fp8_group_pow2_a8_v1", 1),
            ("chop_gemm_format_mxfp8_decode_v1", 1),
            ("chop_gemm_format_mxfp8_a16_decode_v1", 1),
            ("chop_gemm_format_mxfp8_prefill_v1", 128),
            ("chop_gemm_format_mxfp8_a16_prefill_v1", 128),
            ("chop_gemm_format_mxfp4_a16_decode_v1", 1),
            ("chop_gemm_format_mxfp4_a16_prefill_v1", 128),
            ("chop_gemm_format_nvfp4_a16_decode_v1", 1),
            ("chop_gemm_format_nvfp4_a16_prefill_v1", 128),
            ("chop_gemm_format_mxfp4_a8_decode_v1", 1),
            ("chop_gemm_format_mxfp4_a8_prefill_v1", 128),
            ("chop_gemm_format_nvfp4_a4_decode_v1", 1),
            ("chop_gemm_format_nvfp4_a4_prefill_v1", 128),
        ] {
            let case = Case::prepare(session, symbol, m, true)?;
            let function = case.function(session)?;
            case.launch(session, &function, &session.stream)?;
            let actual = case.output(session)?;
            case.receipt(session, &actual, "dense_gemm_format")?;
        }
        Ok(())
    })
    .expect("dense GEMM FP8, MXFP8, MXFP4, and NVFP4 formats");
}

#[test]
fn gemm_graph_replay() {
    watched_gpu_test(256 * 1024 * 1024, 240, |session| {
        let case = Case::prepare(session, "chop_gemm_mimo_router_m1_v1", 1, true)?;
        let function = case.function(session)?;
        let mut checks = case.replay(session)?;
        checks.eager(|| case.launch(session, &function, &session.stream))?;
        let eager = case.output(session)?;
        case.receipt(session, &eager, "dense_gemm_graph_eager")?;
        let graph = session.capture(|stream| case.launch(session, &function, stream))?;
        checks.graph(&graph)?;
        let replay = case.output(session)?;
        case.receipt(session, &replay, "dense_gemm_graph_replay")?;
        case.bitwise_receipt(
            session,
            &eager,
            &replay,
            "dense_gemm_graph_bitwise",
            checks.evidence(),
        )
    })
    .expect("dense GEMM CUDA graph replay");
}

#[test]
fn gemm_determinism() {
    watched_gpu_test(256 * 1024 * 1024, 240, |session| {
        let case = Case::prepare(session, "chop_gemm_mimo_router_m1_v1", 1, true)?;
        let function = case.function(session)?;
        let mut checks = case.replay(session)?;
        checks.eager(|| case.launch(session, &function, &session.stream))?;
        let first = case.output(session)?;
        case.receipt(session, &first, "dense_gemm_determinism_first")?;
        checks.relaunch(|| case.launch(session, &function, &session.stream))?;
        let second = case.output(session)?;
        case.receipt(session, &second, "dense_gemm_determinism")?;
        case.bitwise_receipt(
            session,
            &first,
            &second,
            "dense_gemm_determinism_bitwise",
            checks.evidence(),
        )
    })
    .expect("dense GEMM split-K determinism");
}

#[test]
fn gemm_pdl_dependency() {
    watched_gpu_test(512 * 1024 * 1024, 240, |session| {
        let case = Case::prepare(session, "chop_gemm_mimo_router_m1_v1", 1, true)?;
        let function = case.function(session)?;
        let mut checks = case.replay(session)?;
        checks.eager(|| case.launch(session, &function, &session.stream))?;
        let eager = case.output(session)?;
        case.receipt(session, &eager, "dense_gemm_pdl_eager")?;
        let producer = FlushProducer::new(session)?;
        checks.pdl(
            || producer.launch(session),
            || case.launch(session, &function, &session.stream),
            case.plan.pdl,
        )?;
        let dependent = case.output(session)?;
        case.receipt(session, &dependent, "dense_gemm_pdl_dependency")?;
        case.bitwise_receipt(
            session,
            &eager,
            &dependent,
            "dense_gemm_pdl_bitwise",
            checks.evidence(),
        )
    })
    .expect("dense GEMM PDL dependency");
}

/// Eager runs, CUDA graph replay and a PDL launch behind a producer must
/// give identical bits.
fn replay_cases(session: &Session, cases: &[(&str, u32)], label: &str) -> LabResult<()> {
    for &(symbol, m) in cases {
        let case = Case::prepare_checked(session, symbol, m)?;
        let function = case.function(session)?;
        let labelled = |error: String| format!("{symbol} M={m}: {error}");
        let mut checks = case.replay(session)?;
        checks
            .eager(|| case.launch(session, &function, &session.stream))
            .map_err(labelled)?;
        let eager = case.output(session)?;
        case.receipt(session, &eager, &format!("{label}_eager"))?;
        checks
            .relaunch(|| case.launch(session, &function, &session.stream))
            .map_err(labelled)?;
        let second = case.output(session)?;
        case.bitwise_receipt(
            session,
            &eager,
            &second,
            &format!("{label}_determinism"),
            checks.evidence(),
        )?;
        // Two graph launches, each on poisoned buffers.
        let graph = session.capture(|stream| case.launch(session, &function, stream))?;
        checks.graph(&graph).map_err(labelled)?;
        checks.graph(&graph).map_err(labelled)?;
        let replay = case.output(session)?;
        case.bitwise_receipt(
            session,
            &eager,
            &replay,
            &format!("{label}_graph_bitwise"),
            checks.evidence(),
        )?;
        let producer = FlushProducer::new(session)?;
        checks
            .pdl(
                || producer.launch(session),
                || case.launch(session, &function, &session.stream),
                case.plan.pdl,
            )
            .map_err(labelled)?;
        let dependent = case.output(session)?;
        case.bitwise_receipt(
            session,
            &eager,
            &dependent,
            &format!("{label}_pdl_bitwise"),
            checks.evidence(),
        )?;
    }
    Ok(())
}

/// `chop_probe_flush_v1`: releases its dependents, then waits and writes
/// its own scratch, so a PDL dependent's pre-wait code overlaps it.
struct FlushProducer {
    function: Function,
    scratch: DeviceBuffer,
}

impl FlushProducer {
    fn new(session: &Session) -> LabResult<Self> {
        let function = session
            .module("probe_sm120")?
            .function("chop_probe_flush_v1")
            .map_err(|e| e.to_string())?;
        Ok(Self {
            function,
            scratch: session.allocate(4096)?,
        })
    }
    fn launch(&self, session: &Session) -> LabResult<()> {
        self.function
            .launch(
                &session.stream,
                launch::simple_config((1, 1, 1), (256, 1, 1), 0, (1, 1, 1), false)?,
                &[
                    KernelArgument::Buffer(&self.scratch),
                    KernelArgument::U64(1024),
                ],
            )
            .map_err(|e| e.to_string())
    }
}

#[test]
fn gemm_fragment_prefill_replay() {
    watched_gpu_test(1024 * 1024 * 1024, 240, |session| {
        // Stream-K fixups.
        replay_cases(
            session,
            &[
                ("chop_gemm_mimo_o_prefill_v1", 770),
                ("chop_gemm_mimo_qkv_ga_prefill_v1", 200),
                ("chop_gemm_mimo_router_prefill_v1", 100),
            ],
            "dense_gemm_prefill",
        )
    })
    .expect("dense GEMM prefill determinism, graph replay and PDL");
}

#[test]
fn gemm_fragment_decode_replay() {
    watched_gpu_test(1024 * 1024 * 1024, 240, |session| {
        // Cluster split-K exchanges, including a partial last row tile and
        // fewer live tokens than the bucket.
        replay_cases(
            session,
            &[
                ("chop_gemm_mimo_qkv_ga_m32_v1", 32),
                ("chop_gemm_mimo_qkv_ga_m32_v1", 27),
                ("chop_gemm_mimo_mlp_gate_up_m1_v1", 1),
                ("chop_gemm_mimo_o_m16_v1", 11),
            ],
            "dense_gemm_decode",
        )
    })
    .expect("dense GEMM decode determinism, graph replay and PDL");
}

fn peak(session: &Session, format: &str, memory_bound: bool) -> (f64, String) {
    let root = kernel_lab::receipt_root().join("probes");
    if let Ok(paths) = fs::read_dir(root) {
        let mut paths = paths
            .flatten()
            .map(|entry| entry.path())
            .collect::<Vec<_>>();
        paths.sort();
        for path in paths.into_iter().rev() {
            let Ok(bytes) = fs::read(&path) else { continue };
            let Ok(probe) = serde_json::from_slice::<Value>(&bytes) else {
                continue;
            };
            if probe["uuid"] != session.metadata["uuid"] {
                continue;
            }
            let metric = if memory_bound {
                probe["dram_read"]["gb_per_second"].as_f64()
            } else {
                probe["mma_tflops"][format]["tflops"].as_f64()
            };
            if let Some(value) = metric.filter(|v| *v > 0.0) {
                let filename = path
                    .file_name()
                    .expect("probe receipt filename")
                    .to_string_lossy();
                return (value, format!("probe:probes/{filename}"));
            }
        }
    }
    // NVIDIA RTX PRO 6000 Blackwell Max-Q, dense F32-accumulate tensor peaks.
    let fallback = if memory_bound {
        1800.0
    } else {
        match format {
            "bf16" => 438.9,
            "e4m3" | "mxf8_e4m3" | "mxf8_e2m1" => 877.9,
            _ => 1755.7,
        }
    };
    (
        fallback,
        "fallback:NVIDIA RTX Blackwell GPU Architecture Appendix A, 1.8 TB/s DRAM".into(),
    )
}

fn bench_case(session: &Session, symbol: &str, m: u32, launch_floor_us: f64) -> LabResult<()> {
    eprintln!("benchmarking {symbol} at M={m}");
    let case = Case::prepare(session, symbol, m, false)?;
    let function = case.function(session)?;
    let timing = kernel_lab::time_named(session, &format!("{symbol} M={m}"), 2, 3, |stream| {
        case.launch(session, &function, stream)
    })?;
    let q = query(&case.variant);
    let memory_bound = m <= 32;
    let format = match (q.activation, q.weight) {
        (ActivationFormat::Mxfp8, WeightFormat::Mxfp4) => "mxf8_e2m1",
        (ActivationFormat::Mxfp8, _) => "mxf8_e4m3",
        (ActivationFormat::Nvfp4, _) => "nvfp4_4x",
        (ActivationFormat::Fp8Group128(_), _) => "e4m3",
        _ => "bf16",
    };
    let (peak_value, peak_source) = peak(session, format, memory_bound);
    let weight_bytes = case.buffers[&WEIGHT].byte_len() as f64;
    let activation_bytes = (m as f64)
        * f64::from(q.k)
        * if q.activation == ActivationFormat::Bf16 {
            2.0
        } else {
            1.0
        };
    let output_bytes = case.buffers[&OUTPUT].byte_len() as f64;
    let logical_bytes = weight_bytes + activation_bytes + output_bytes;
    let logical_flops = 2.0 * f64::from(m) * f64::from(q.n) * f64::from(q.k);
    let knobs = case.variant.key.knobs;
    let output_n = if q.epilogues.swiglu.is_some() {
        q.n / 2
    } else {
        q.n
    };
    let row_tiles = if memory_bound {
        output_n.div_ceil(u32::from(knobs.tile_m))
    } else {
        m.div_ceil(u32::from(knobs.tile_m))
    };
    let column_tiles = if memory_bound {
        q.m_capacity.div_ceil(u32::from(knobs.tile_n))
    } else {
        output_n.div_ceil(u32::from(knobs.tile_n))
    };
    let useful_ctas =
        (row_tiles * column_tiles * u32::from(case.counts.split_k)).min(case.plan.grid[0]);
    let subtiles = u32::from(knobs.tile_m / 16) * u32::from(knobs.tile_n / 8);
    let per_warp = subtiles.div_ceil(8).min(4);
    let mma_waves = subtiles.div_ceil(8 * per_warp);
    let elapsed_seconds = f64::from(timing.median_ms) / 1000.0;
    let achieved = if memory_bound {
        logical_bytes / elapsed_seconds / 1e9
    } else {
        logical_flops / elapsed_seconds / 1e12
    };
    let peak_fraction = achieved / peak_value;
    let limiting_factor = if peak_fraction >= 0.7 {
        "roofline target met"
    } else if useful_ctas < 188 {
        "useful CTA count below SM count"
    } else if mma_waves > 1 {
        "serial MMA waves repeat K tile TMA transfers"
    } else if memory_bound && m >= 32 {
        "larger token tile increases serial MMA work"
    } else if q.weight != WeightFormat::Bf16 {
        "register weight dequantization and scale loads"
    } else {
        "short per-CTA K pipeline and launch latency"
    };
    let mut receipt = BTreeMap::new();
    receipt.insert("case".into(), json!("dense_gemm_benchmark"));
    receipt.insert("symbol".into(), json!(symbol));
    receipt.insert("profile".into(), json!(case.variant.profile.as_str()));
    receipt.insert("m".into(), json!(m));
    receipt.insert("n".into(), json!(q.n));
    receipt.insert("k".into(), json!(q.k));
    receipt.insert("split_k".into(), json!(case.counts.split_k));
    receipt.insert("useful_ctas".into(), json!(useful_ctas));
    receipt.insert("sm_coverage".into(), json!(f64::from(useful_ctas) / 188.0));
    receipt.insert("mma_waves_per_tile".into(), json!(mma_waves));
    receipt.insert(
        "k_chunks_per_split".into(),
        json!(
            q.k.div_ceil(u32::from(knobs.tile_k))
                .div_ceil(u32::from(case.counts.split_k))
        ),
    );
    receipt.insert("launch_floor_us".into(), json!(launch_floor_us));
    receipt.insert("limiting_factor_inference".into(), json!(limiting_factor));
    receipt.insert("median_us".into(), json!(timing.median_ms * 1000.0));
    receipt.insert("min_us".into(), json!(timing.min_ms * 1000.0));
    receipt.insert("rounds".into(), json!(timing.rounds));
    receipt.insert("logical_bytes".into(), json!(logical_bytes));
    receipt.insert("logical_flops".into(), json!(logical_flops));
    receipt.insert(
        "metric".into(),
        json!(if memory_bound { "GB/s" } else { "TFLOP/s" }),
    );
    receipt.insert("achieved".into(), json!(achieved));
    receipt.insert("peak".into(), json!(peak_value));
    receipt.insert("peak_source".into(), json!(peak_source));
    receipt.insert(
        "roofline_us".into(),
        json!(if memory_bound {
            logical_bytes / peak_value / 1000.0
        } else {
            logical_flops / peak_value / 1e6
        }),
    );
    receipt.insert("percent_of_peak".into(), json!(100.0 * peak_fraction));
    session.receipt("bench", receipt)?;
    Ok(())
}

#[test]
fn gemm_bench_profiles() {
    if std::env::var("CHOPINFER_KERNEL_BENCH").as_deref() != Ok("1") {
        kernel_lab::skip("GEMM benchmarks: set CHOPINFER_KERNEL_BENCH=1 on idle GPUs").unwrap();
        return;
    }
    let filter = std::env::var("CHOPINFER_KERNEL_BENCH_FILTER").ok();
    let profile = match std::env::var("CHOPINFER_KERNEL_BENCH_PROFILE").as_deref() {
        Ok("qwen38-flash") => model_qwen38_contract::sm120::config::PROFILE,
        _ => model_mimo26_contract::sm120::config::PROFILE,
    };
    watched_gpu_test(2 * 1024 * 1024 * 1024, 540, |session| {
        let probe = session.module("probe_sm120")?;
        let probe_function = probe
            .function("chop_probe_sm120_v1")
            .map_err(|error| error.to_string())?;
        let stamp = session.allocate(4)?;
        let launch_floor = kernel_lab::time_named(session, "launch_floor", 2, 3, |stream| {
            probe_function
                .launch(
                    stream,
                    launch::simple_config((1, 1, 1), (32, 1, 1), 0, (1, 1, 1), true)?,
                    &[runtime_cuda::KernelArgument::Buffer(&stamp)],
                )
                .map_err(|error| error.to_string())
        })?;
        let launch_floor_us = f64::from(launch_floor.median_ms) * 1000.0;
        let mut ran = 0;
        for item in gemm::catalog(kernel_lab::catalogs()) {
            if item.profile != profile
                || filter
                    .as_ref()
                    .is_some_and(|value| !item.symbol.contains(value))
            {
                continue;
            }
            let q = query(&item);
            if q.m_capacity <= 32 {
                bench_case(session, &item.symbol, q.m_capacity, launch_floor_us)?;
                ran += 1;
            } else {
                let prefill_m = std::env::var("CHOPINFER_KERNEL_BENCH_M")
                    .ok()
                    .map(|value| {
                        value
                            .split(',')
                            .map(|m| m.parse::<u32>().expect("bench M must be u32"))
                            .collect::<Vec<_>>()
                    })
                    .unwrap_or_else(|| vec![512, 2048, 8192]);
                for m in prefill_m {
                    bench_case(session, &item.symbol, m, launch_floor_us)?;
                    ran += 1;
                }
            }
        }
        for (symbol, m) in [
            ("chop_gemm_std_dense_m1_v1", 1),
            ("chop_gemm_std_dense_fp8_m1_v1", 1),
            ("chop_gemm_std_dense_m8_v1", 8),
            ("chop_gemm_std_dense_fp8_m8_v1", 8),
            ("chop_gemm_std_dense_m32_v1", 32),
            ("chop_gemm_std_dense_fp8_m32_v1", 32),
            ("chop_gemm_std_dense_prefill_v1", 512),
            ("chop_gemm_std_dense_fp8_prefill_v1", 512),
            ("chop_gemm_std_dense_prefill_v1", 2048),
            ("chop_gemm_std_dense_fp8_prefill_v1", 2048),
        ] {
            if filter.as_ref().is_some_and(|value| !symbol.contains(value)) {
                continue;
            }
            bench_case(session, symbol, m, launch_floor_us)?;
            ran += 1;
        }
        if ran == 0 {
            return Err(format!("no GEMM benchmark matched {filter:?}"));
        }
        Ok(())
    })
    .expect("dense GEMM profile benchmarks");
}

#[test]
fn gemm_graph_bench_48_layers() {
    if std::env::var("CHOPINFER_KERNEL_GRAPH_BENCH").as_deref() != Ok("1") {
        kernel_lab::skip("GEMM graph benchmarks: set CHOPINFER_KERNEL_GRAPH_BENCH=1").unwrap();
        return;
    }
    let filter = std::env::var("CHOPINFER_KERNEL_GRAPH_SYMBOL").ok();
    let filter_m = std::env::var("CHOPINFER_KERNEL_GRAPH_M")
        .ok()
        .map(|value| value.parse::<u32>().expect("graph M must be u32"));
    watched_gpu_test(8 * 1024 * 1024 * 1024, 1400, |session| {
        let module = session.module("probe_sm120")?;
        let producer = module
            .function("chop_probe_pdl_first_v1")
            .map_err(|error| error.to_string())?;
        let producer_config = launch::simple_config((1, 1, 1), (32, 1, 1), 0, (1, 1, 1), false)?;
        let stamps = (0..48)
            .map(|_| session.allocate(24))
            .collect::<LabResult<Vec<_>>>()?;
        let launch_producer = |stream: &Stream, layer: usize| {
            producer
                .launch(
                    stream,
                    producer_config,
                    &[
                        KernelArgument::Buffer(&stamps[layer]),
                        KernelArgument::U64(1000),
                    ],
                )
                .map_err(|error| error.to_string())
        };
        let producer_graph = session.capture(|stream| {
            for layer in 0..48 {
                launch_producer(stream, layer)?;
            }
            Ok(())
        })?;
        let producer_timing = time(session, 2, 5, |stream| {
            producer_graph.launch(stream).map_err(|error| error.to_string())
        })?;
        let mut ran = 0;
        // Decode buckets, then each prefill entry at the M values listed in
        // CHOPINFER_KERNEL_GRAPH_PREFILL_M (lm_head included, 12 layers).
        let prefill_m = std::env::var("CHOPINFER_KERNEL_GRAPH_PREFILL_M")
            .ok()
            .map(|value| {
                value
                    .split(',')
                    .map(|m| m.parse::<u32>().expect("prefill M must be u32"))
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        let mut cases = Vec::new();
        for name in ["qkv_ga", "qkv_swa", "o", "router", "mlp_gate_up", "mlp_down"] {
            for m in [1, 2, 4, 8, 16, 32] {
                cases.push((format!("chop_gemm_mimo_{name}_m{m}_v1"), m));
            }
        }
        for name in [
            "qkv_ga",
            "qkv_swa",
            "o",
            "router",
            "mlp_gate_up",
            "mlp_down",
            "head",
        ] {
            for &m in &prefill_m {
                cases.push((format!("chop_gemm_mimo_{name}_prefill_v1"), m));
            }
        }
        for (symbol, m) in cases {
            {
                if filter_m.is_some_and(|selected| selected != m) {
                    continue;
                }
                if filter.as_ref().is_some_and(|value| value != &symbol) {
                    continue;
                }
                let layer_count = if symbol.contains("_head_") { 12 } else { 48 };
                eprintln!("graph benchmarking {symbol}");
                let case = Case::prepare(session, &symbol, m, false)?;
                let function = case.function(session)?;
                let mut layers = Vec::with_capacity(layer_count - 1);
                for _ in 1..layer_count {
                    let weight = session.allocate(case.buffers[&WEIGHT].byte_len())?;
                    weight
                        .copy_range_from_async(
                            0,
                            &case.buffers[&WEIGHT],
                            0,
                            weight.byte_len(),
                            &session.stream,
                        )
                        .map_err(|error| error.to_string())?;
                    let output = session.allocate(case.buffers[&OUTPUT].byte_len())?;
                    let partial = session.allocate(case.buffers[&PARTIAL].byte_len())?;
                    let semaphores = upload(
                        session,
                        &vec![0; case.buffers[&SEMAPHORES].byte_len()],
                    )?;
                    layers.push(GraphLayer {
                        weight,
                        output,
                        partial,
                        semaphores,
                    });
                }
                session.stream.synchronize().map_err(|error| error.to_string())?;
                let mut checks = case.replay(session)?;
                checks.eager(|| case.launch(session, &function, &session.stream))?;
                let eager = case.output(session)?;
                let graph = session.capture(|stream| {
                    for layer in 0_usize..layer_count {
                        launch_producer(stream, layer)?;
                        case.launch_layer(
                            session,
                            &function,
                            stream,
                            layer.checked_sub(1).map(|index| &layers[index]),
                        )?;
                    }
                    Ok(())
                })?;
                for layer in &layers {
                    case.poison(session, &[&layer.output, &layer.partial])?;
                }
                case.poison(session, &[&case.buffers[&OUTPUT], &case.buffers[&PARTIAL]])?;
                graph.launch(&session.stream).map_err(|error| error.to_string())?;
                let replay = case.written(session, &layers[layer_count - 2].output, "last layer")?;
                let mut evidence = checks.evidence().clone();
                evidence.passed(poison::Phase::Graph, case.plan.pdl);
                case.bitwise_receipt(
                    session,
                    &eager,
                    &replay,
                    "dense_gemm_48_layer_graph",
                    &evidence,
                )?;
                let timing = time(session, 2, 5, |stream| {
                    graph.launch(stream).map_err(|error| error.to_string())
                })?;
                let gemm_graph = session.capture(|stream| {
                    for layer in 0_usize..layer_count {
                        case.launch_layer(
                            session,
                            &function,
                            stream,
                            layer.checked_sub(1).map(|index| &layers[index]),
                        )?;
                    }
                    Ok(())
                })?;
                let gemm_timing = time(session, 2, 5, |stream| {
                    gemm_graph.launch(stream).map_err(|error| error.to_string())
                })?;
                let pair_us = f64::from(timing.median_ms) * 1000.0 / layer_count as f64;
                let producer_us = f64::from(producer_timing.median_ms) * 1000.0 / 48.0;
                let gemm_us = f64::from(gemm_timing.median_ms) * 1000.0 / layer_count as f64;
                let incremental_us = (pair_us - producer_us).max(0.0);
                let q = query(&case.variant);
                let logical_bytes = case.buffers[&WEIGHT].byte_len() as f64
                    + f64::from(m * q.k * 2)
                    + case.buffers[&OUTPUT].byte_len() as f64;
                let (peak_gbps, peak_source) = peak(session, "bf16", true);
                let mut receipt = BTreeMap::new();
                receipt.insert("case".into(), json!("dense_gemm_48_layer_graph_benchmark"));
                receipt.insert("symbol".into(), json!(symbol));
                receipt.insert("m".into(), json!(m));
                receipt.insert("layers".into(), json!(layer_count));
                receipt.insert("distinct_weight_buffers".into(), json!(true));
                receipt.insert("distinct_output_and_workspace_buffers".into(), json!(true));
                receipt.insert("producer_pdl".into(), json!(true));
                receipt.insert("producer_us_per_layer".into(), json!(producer_us));
                receipt.insert("pair_us_per_layer".into(), json!(pair_us));
                receipt.insert("gemm_only_us_per_layer".into(), json!(gemm_us));
                receipt.insert("incremental_us_per_layer".into(), json!(incremental_us));
                receipt.insert("logical_bytes_per_layer".into(), json!(logical_bytes));
                receipt.insert("peak_gbps".into(), json!(peak_gbps));
                receipt.insert("peak_source".into(), json!(peak_source));
                receipt.insert(
                    "logical_dram_peak_percent".into(),
                    json!(100.0 * logical_bytes / (gemm_us * 1000.0 * peak_gbps)),
                );
                session.receipt("bench", receipt)?;
                eprintln!(
                    "{symbol}: pair {pair_us:.2} us, GEMM-only {gemm_us:.2} us, producer {producer_us:.2} us per layer"
                );
                ran += 1;
            }
        }
        // Verify steps above 32 rows as M / 32 launches of the 32-row
        // cluster decode entry on the same weights (their rows are bitwise
        // the decode rows), against the prefill entry at M above.
        let sliced_m = std::env::var("CHOPINFER_KERNEL_GRAPH_SLICED_M")
            .ok()
            .map(|value| {
                value
                    .split(',')
                    .map(|m| m.parse::<u32>().expect("sliced M must be u32"))
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        for name in ["qkv_ga", "qkv_swa", "o", "mlp_gate_up", "mlp_down"] {
            for &m in &sliced_m {
                let symbol = format!("chop_gemm_mimo_{name}_m32_v1");
                if filter.as_ref().is_some_and(|value| value != &symbol) {
                    continue;
                }
                let slices = m.div_ceil(32) as usize;
                let case = Case::prepare(session, &symbol, 32, false)?;
                let function = case.function(session)?;
                let copy = |id: u64| -> LabResult<DeviceBuffer> {
                    let source = &case.buffers[&id];
                    let buffer = session.allocate(source.byte_len())?;
                    buffer
                        .copy_range_from_async(0, source, 0, buffer.byte_len(), &session.stream)
                        .map_err(|error| error.to_string())?;
                    Ok(buffer)
                };
                let activations = (0..slices)
                    .map(|_| copy(ACTIVATION))
                    .collect::<LabResult<Vec<_>>>()?;
                // Per layer one weight; per layer and slice an output and a
                // workspace, since PDL overlaps consecutive slices.
                let weights = (0..48)
                    .map(|_| copy(WEIGHT))
                    .collect::<LabResult<Vec<_>>>()?;
                let slice_buffers = (0..48 * slices)
                    .map(|_| Ok((copy(OUTPUT)?, copy(PARTIAL)?)))
                    .collect::<LabResult<Vec<_>>>()?;
                session.stream.synchronize().map_err(|error| error.to_string())?;
                let launch_slices = |stream: &Stream| -> LabResult<()> {
                    for layer in 0..48 {
                        for slice in 0..slices {
                            let (output, partial) = &slice_buffers[layer * slices + slice];
                            let mut buffers = case
                                .buffers
                                .iter()
                                .map(|(&id, buffer)| (id, buffer))
                                .collect::<BTreeMap<_, _>>();
                            buffers.insert(WEIGHT, &weights[layer]);
                            buffers.insert(ACTIVATION, &activations[slice]);
                            buffers.insert(OUTPUT, output);
                            buffers.insert(PARTIAL, partial);
                            launch::launch(
                                &session.context,
                                &function,
                                stream,
                                case.plan,
                                &case.args,
                                &Bindings { buffers },
                            )?;
                        }
                    }
                    Ok(())
                };
                let graph = session.capture(launch_slices)?;
                let timing = time(session, 2, 5, |stream| {
                    graph.launch(stream).map_err(|error| error.to_string())
                })?;
                let per_layer = f64::from(timing.median_ms) * 1000.0 / 48.0;
                let mut receipt = BTreeMap::new();
                receipt.insert("case".into(), json!("dense_gemm_48_layer_sliced_decode"));
                receipt.insert("symbol".into(), json!(symbol));
                receipt.insert("m".into(), json!(m));
                receipt.insert("slices".into(), json!(slices));
                receipt.insert("gemm_only_us_per_layer".into(), json!(per_layer));
                session.receipt("bench", receipt)?;
                eprintln!("{symbol} x{slices} for M={m}: GEMM-only {per_layer:.2} us per layer");
                ran += 1;
            }
        }
        if ran == 0 {
            return Err("GEMM graph filter matched no catalog symbol".into());
        }
        Ok(())
    })
    .expect("48-layer dense GEMM graph benchmarks");
}

/// Every Qwen3.8 fragment-family entry of the selected topology
/// (`CHOPINFER_QWEN38_RANKS`, default TP4): decode buckets full and
/// partial, prefill at one token, a partial token tile, stream-K tails and
/// the F32 head. The router is TP4's at every rank count. Most decode
/// buckets run on `gemm::fused::catalog(kernel_lab::catalogs())` instead (kernel-lab test
/// `qwen38_gemm`).
///
/// TP2 is lean (`lean_qwen38_cases`): every per-rank entry once or
/// twice at few rows, the oracle on `lean_columns`.
#[test]
fn gemm_qwen38_shapes() {
    watched_gpu_test(2 * 1024 * 1024 * 1024, 540, |session| {
        let profile = kernel_lab::qwen38_profile()?;
        let tag = profile.tag();
        if kernel_lab::qwen38_lean()? {
            for (symbol, m) in lean_qwen38_cases(profile) {
                let case = Case::prepare_checked(session, &symbol, m)?;
                let function = case.function(session)?;
                case.launch(session, &function, &session.stream)?;
                let actual = case.output(session)?;
                case.receipt(session, &actual, "qwen38_dense_gemm")?;
            }
            return Ok(());
        }
        let mut cases = Vec::new();
        for item in gemm::catalog(kernel_lab::catalogs()) {
            let q = query(&item);
            if item.profile != profile || item.symbol.contains("_shared_") {
                continue;
            }
            if q.m_capacity <= 32 {
                cases.push((item.symbol.clone(), q.m_capacity));
                if q.m_capacity > 1 {
                    cases.push((item.symbol.clone(), q.m_capacity / 2 + 1));
                }
            } else if !item.symbol.contains("head_f32") {
                cases.push((item.symbol.clone(), 1));
                cases.push((
                    item.symbol.clone(),
                    if item.symbol.contains("head") {
                        128
                    } else {
                        200
                    },
                ));
            }
        }
        cases.push((format!("chop_gemm_{tag}_out_prefill_v1"), 770));
        cases.push(("chop_gemm_qwen38_router_prefill_v1".into(), 333));
        cases.push((format!("chop_gemm_{tag}_head_f32_prefill_v1"), 128));
        for (symbol, m) in cases {
            let case = Case::prepare(session, &symbol, m, true)?;
            let function = case.function(session)?;
            case.launch(session, &function, &session.stream)?;
            let actual = case.output(session)?;
            case.receipt(session, &actual, "qwen38_dense_gemm")?;
        }
        Ok(())
    })
    .expect("Qwen3.8 fragment-family GEMM shapes match the f64 oracle");
}

/// The lean cases of a TP2 profile, grouped by weight so the head's
/// weight is made once per group: every decode bucket full (the head's and
/// the shared expert's), partial at 8, 16 and 32 tokens; every prefill at one
/// row and at 100 (a partial 128-row tile), `out` at 770 as well (stream-K
/// tails), the F32 head at 100. The router is TP4's at every rank count.
fn lean_qwen38_cases(profile: Profile) -> Vec<(String, u32)> {
    let mut cases = Vec::new();
    for item in gemm::catalog(kernel_lab::catalogs())
        .into_iter()
        .filter(|i| i.profile == profile)
    {
        let capacity = query(&item).m_capacity;
        let symbol = item.symbol.clone();
        if capacity <= 32 {
            cases.push((symbol.clone(), capacity));
            if capacity >= 8 {
                cases.push((symbol, capacity / 2 + 1));
            }
        } else if symbol.contains("head_f32") {
            cases.push((symbol, 100));
        } else {
            cases.push((symbol.clone(), 1));
            cases.push((symbol.clone(), 100));
            if symbol.ends_with("_out_prefill_v1") {
                cases.push((symbol, 770));
            }
        }
    }
    let weight = |symbol: &str| {
        ["head", "shared_gate_up", "shared_down"]
            .into_iter()
            .position(|name| symbol.contains(&format!("_{name}_")))
    };
    cases.sort_by_key(|(symbol, _)| weight(symbol));
    cases
}

#[test]
fn gemm_qwen38_replay() {
    watched_gpu_test(1024 * 1024 * 1024, 240, |session| {
        // The selected topology's entries (`CHOPINFER_QWEN38_RANKS`). Lean
        // runs keep one decode and one prefill case (the partial bucket and
        // the stream-K tails).
        let tag = kernel_lab::qwen38_profile()?.tag();
        let mut cases = vec![
            (format!("chop_gemm_{tag}_head_m1_v1"), 1),
            (format!("chop_gemm_{tag}_head_m32_v1"), 27),
            (format!("chop_gemm_{tag}_head_f32_m32_v1"), 27),
            (format!("chop_gemm_{tag}_out_prefill_v1"), 770),
            (format!("chop_gemm_{tag}_lin_qkv_prefill_v1"), 200),
        ];
        if kernel_lab::qwen38_lean()? {
            cases = cases[1..3].to_vec();
        }
        let cases = cases
            .iter()
            .map(|(symbol, m)| (symbol.as_str(), *m))
            .collect::<Vec<_>>();
        replay_cases(session, &cases, "qwen38_dense_gemm")
    })
    .expect("Qwen3.8 GEMM determinism, graph replay and PDL");
}

#[path = "gemm_verify_drift.rs"]
mod verify_drift;

#[path = "gemm_accumulation.rs"]
mod accumulation;

/// The MiMo decode head launched over column chunks (whole 128-column tiles,
/// the same split-K) writes the full launch's values bitwise, so a chunked
/// head can overlap each chunk's logits gather with the next chunk.
#[test]
fn gemm_head_column_chunks() {
    gpu_test(2 * 1024 * 1024 * 1024, |session| {
        let bench = std::env::var("CHOPINFER_KERNEL_BENCH").as_deref() == Ok("1");
        for m in [1_u32, 8, 32] {
            let symbol = format!("chop_gemm_mimo_head_m{m}_v1");
            let case = Case::prepare(session, &symbol, m, false)?;
            let function = case.function(session)?;
            let q = query(&case.variant);
            let n = q.n as usize;
            let tile = usize::from(case.variant.key.knobs.tile_m);
            let element = if q.output == DType::F32 { 4 } else { 2 };
            case.poison(session, &[&case.buffers[&OUTPUT], &case.buffers[&PARTIAL]])?;
            case.launch(session, &function, &session.stream)?;
            case.written(session, &case.buffers[&OUTPUT], "full head")?;
            let mut full = vec![0_u8; m as usize * n * element];
            case.buffers[&OUTPUT]
                .copy_to_bytes(&mut full, &session.stream)
                .map_err(|e| e.to_string())?;
            let mut weights = vec![0_u8; case.buffers[&WEIGHT].byte_len()];
            case.buffers[&WEIGHT]
                .copy_to_bytes(&mut weights, &session.stream)
                .map_err(|e| e.to_string())?;
            // Fragment-major weights hold each 16-column strip contiguously.
            let strip_bytes = weights.len() / (n / 16);
            let tiles = n / tile;
            // CHOPINFER_KERNEL_HEAD_CHUNKS sets the chunk count (default 4).
            let count = std::env::var("CHOPINFER_KERNEL_HEAD_CHUNKS")
                .ok()
                .and_then(|value| value.parse::<usize>().ok())
                .unwrap_or(4);
            let chunk_tiles: Vec<usize> = (0..count)
                .map(|chunk| tiles * (chunk + 1) / count - tiles * chunk / count)
                .collect();
            let mut chunks = Vec::new();
            let mut first = 0;
            for count in chunk_tiles {
                let (begin, width) = (first * tile, count * tile);
                first += count;
                let weight = upload(
                    session,
                    &weights[begin / 16 * strip_bytes..(begin + width) / 16 * strip_bytes],
                )?;
                let output = session.allocate(m as usize * width * element)?;
                let partial = session.allocate(case.plan.workspace_bytes as usize)?;
                let semaphores = upload(session, &vec![0; 1 << 16])?;
                let mut args = case.args.clone();
                for arg in &mut args.0 {
                    if *arg == kernel_core::contract::Argument::U32(n as u32) {
                        *arg = kernel_core::contract::Argument::U32(width as u32);
                    }
                }
                let mut plan = case.plan;
                plan.grid[0] = plan.grid[0].min((count * usize::from(case.counts.split_k)) as u32);
                chunks.push((
                    begin, width, weight, output, partial, semaphores, args, plan,
                ));
            }
            let launch_chunks = |stream: &Stream| -> LabResult<()> {
                for (_, _, weight, output, partial, semaphores, args, plan) in &chunks {
                    let mut buffers = case
                        .buffers
                        .iter()
                        .map(|(&id, buffer)| (id, buffer))
                        .collect::<BTreeMap<_, _>>();
                    buffers.insert(WEIGHT, weight);
                    buffers.insert(OUTPUT, output);
                    buffers.insert(PARTIAL, partial);
                    buffers.insert(SEMAPHORES, semaphores);
                    launch::launch(
                        &session.context,
                        &function,
                        stream,
                        *plan,
                        args,
                        &Bindings { buffers },
                    )?;
                }
                Ok(())
            };
            for (_, _, _, output, partial, ..) in &chunks {
                case.poison(session, &[output, partial])?;
            }
            launch_chunks(&session.stream)?;
            for (begin, _, _, output, ..) in &chunks {
                case.written(session, output, &format!("chunk at column {begin}"))?;
            }
            let mut mismatches = 0_usize;
            for (begin, width, _, output, ..) in &chunks {
                let mut bytes = vec![0_u8; m as usize * width * element];
                output
                    .copy_to_bytes(&mut bytes, &session.stream)
                    .map_err(|e| e.to_string())?;
                for row in 0..m as usize {
                    let got = &bytes[row * width * element..(row + 1) * width * element];
                    let want =
                        &full[(row * n + begin) * element..(row * n + begin + width) * element];
                    mismatches += got
                        .chunks_exact(element)
                        .zip(want.chunks_exact(element))
                        .filter(|(left, right)| left != right)
                        .count();
                }
            }
            let mut receipt = BTreeMap::new();
            receipt.insert("case".into(), json!("head_column_chunks"));
            receipt.insert("symbol".into(), json!(symbol));
            receipt.insert("m".into(), json!(m));
            receipt.insert("split_k".into(), json!(case.counts.split_k));
            receipt.insert(
                "chunk_columns".into(),
                json!(chunks.iter().map(|chunk| chunk.1).collect::<Vec<_>>()),
            );
            receipt.insert("bitwise_mismatches".into(), json!(mismatches));
            if bench {
                let whole = session.capture(|stream| case.launch(session, &function, stream))?;
                let split = session.capture(launch_chunks)?;
                let whole_timing = time(session, 3, 21, |stream| {
                    whole.launch(stream).map_err(|e| e.to_string())
                })?;
                let split_timing = time(session, 3, 21, |stream| {
                    split.launch(stream).map_err(|e| e.to_string())
                })?;
                receipt.insert(
                    "whole_median_us".into(),
                    json!(whole_timing.median_ms * 1000.0),
                );
                receipt.insert(
                    "chunks_median_us".into(),
                    json!(split_timing.median_ms * 1000.0),
                );
            }
            eprintln!(
                "{}",
                serde_json::to_string(&receipt).map_err(|e| e.to_string())?
            );
            session.receipt("gemm", receipt)?;
            if mismatches != 0 {
                return Err(format!("{symbol}: {mismatches} chunked head values differ"));
            }
        }
        Ok(())
    })
    .expect("chunked MiMo head matches the whole launch");
}

#[path = "generated_dense.rs"]
mod generated_dense;
