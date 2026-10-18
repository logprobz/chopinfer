//! Cases of the pilot families (RMSNorm and projections), the SM120
//! selection the lab's catalogs give, and the pilot's required op set.
//!
//! Each case generator declares a prepared op's roles from its contract op
//! and the op's info, with inputs from [`super::bf16_values`], and the
//! contract's oracle over the role-named images. The same cases run on
//! every backend.

pub mod argmax;
pub mod attention;
pub mod block_quantize;
pub mod collectives;
pub mod context_parallel;
pub mod f32_projection;
pub mod fp4_key_write;
pub mod gated_pooling;
pub mod gated_sparse;
pub mod latent;
pub mod latent_glue;
pub mod latent_sources;
pub mod moe;
pub mod moe_chained_quantized;
pub mod moe_routed;
pub mod multimodal;
pub mod mxfp8_fold;
pub mod pooled_indexer;
pub mod position_blocks;
pub mod position_indexer;
pub mod position_query;
pub mod recurrent;
pub mod rope_kv;
#[cfg(feature = "host-sampling")]
pub mod sampling;
pub mod silu_mul_quantize;
pub mod sinkhorn;
pub mod sinkhorn_lagged;

use super::{OpCase, Required, Spec, bf16_bytes, bf16_values};
use crate::LabResult;
use crate::conformance::{Expected, Holes, Images, Rows};
use kernel_core::backend::{Need, Prepared, View};
use kernel_core::codec::{bf16_to_f32, f32_to_bf16};
use kernel_core::contract::{
    DType, Epilogues, Refusal, RmsNormKind, RowQuery, ScaleMode, StaticQuery, Target, WeightFormat,
};
use kernel_core::oracle::{self, GemmEpilogue, GemmShape, OracleOutput};
use kernel_core::precision::buffers::{ProjectionBuffers, ProjectionScratch, QuantizedWeights};
use kernel_core::precision::{
    ActivationSpec, ArithmeticSpec, CodeEncoding, DecodeOrder, ExecutionPhase, OperationKind,
    OperationSpec, OperationUse, Rounding, TensorSpan, WeightLayout, WeightSpec, dtype_bits,
};
use kernel_core::properties::Invariance;
use kernel_core::quantize::{self, Quantized, Scales};
use kernel_core::rmsnorm::{
    FusedResidual, RmsNormBuffers, RmsNormLayout, RmsNormNumerics, RmsNormRequirement,
};
use kernel_core::rows::{EmbeddingBuffers, LogitsBuffers};
use kernel_core::transform::fragment::{FragmentWeightFormat, FragmentWeightLayout};
use kernels_sm120::Catalogs;
use kernels_sm120::backend::{DenseForm, OfferList, ProjectionOffer, Sm120};
use kernels_sm120::precision::{KernelOffer, capabilities};
use std::cell::RefCell;

/// The rows every pilot row op serves: a capacity bucket of the SM120 row
/// kernels.
pub const ROW_CAPACITY: u32 = 16;

/// The BF16 values of `bytes`, widened.
fn widen(bits: &[u16]) -> Vec<f64> {
    bits.iter().map(|&b| f64::from(bf16_to_f32(b))).collect()
}

/// The case of an RMSNorm over `capacity` rows: BF16 rows and weight from
/// the kit's generators, a fused residual when the requirement has one, and
/// the contract's oracle. The residual's rounded sum is exact bits (an F32
/// add rounded to BF16, as the contract states); the output is exact bits
/// for the HF kind, whose oracle is the F32 arithmetic the contract pins,
/// and within the F64 oracle's tolerance for every other kind.
pub fn rmsnorm_case<'a>(
    requirement: RmsNormRequirement,
    capacity: u32,
    op: Prepared<RmsNormBuffers<View>>,
) -> LabResult<OpCase<'a>> {
    if [requirement.input, requirement.weight] != [DType::Bf16; 2]
        || !matches!(requirement.output, DType::Bf16 | DType::F32)
        || requirement.output_quantization.is_some()
        || !matches!(requirement.layout, RmsNormLayout::ContiguousRows { .. })
    {
        return Err("the kit's RMSNorm case covers contiguous unquantized BF16 inputs and BF16 or F32 output".into());
    }
    let width = requirement.width as usize;
    let elements = capacity as usize * width;
    let rows = |image: Vec<u16>| Spec::input(DType::Bf16, (Rows::Live, width), bf16_bytes(&image));
    let mut input = bf16_values(1, elements);
    if requirement.output == DType::F32 {
        // Scaling inputs leaves normalized values unchanged at this eps,
        // while making the existing f64 bound distinguish BF16 rounding.
        for v in &mut input {
            *v = f32_to_bf16(bf16_to_f32(*v) / 256.0);
        }
    }
    let roles = RmsNormBuffers {
        x: rows(input),
        weight: Spec::input(
            DType::Bf16,
            (Rows::Fixed(1), width),
            bf16_bytes(&bf16_values(2, width)),
        ),
        residual: requirement.fused_residual.then(|| FusedResidual {
            input: rows(bf16_values(3, elements)),
            output: Spec::output(DType::Bf16, (Rows::Live, width)),
        }),
        y: Spec::output(requirement.output, (Rows::Live, width)),
        live_rows: Spec::live(),
    };
    Ok(OpCase::new(op, capacity, roles)?
        .oracle(move |images, _| rmsnorm_oracle(&requirement, images)))
}

fn rmsnorm_oracle(
    requirement: &RmsNormRequirement,
    images: &Images,
) -> LabResult<Vec<(String, Expected)>> {
    let x = images.u16s("x")?;
    let mut results = Vec::new();
    let summed = if requirement.fused_residual {
        let residual = images.u16s("residual.input")?;
        let summed = x
            .iter()
            .zip(&residual)
            .map(|(&x, &r)| f32_to_bf16(bf16_to_f32(x) + bf16_to_f32(r)))
            .collect::<Vec<_>>();
        results.push((
            "residual.output".to_owned(),
            Expected::Bits(bf16_bytes(&summed)),
        ));
        summed
    } else {
        x
    };
    // The HF kind's exact reference sums in the order its numerics name.
    let reference = oracle::rmsnorm_numerics(
        &widen(&summed),
        requirement.width as usize,
        &widen(&images.u16s("weight")?),
        f64::from(f32::from_bits(requirement.epsilon_bits)),
        requirement.kind,
        None,
        requirement.numerics,
    )
    .map_err(|e| e.0.to_string())?;
    let y = match requirement.kind {
        RmsNormKind::HfBf16Intermediate => {
            let bits = reference
                .values
                .iter()
                .map(|&value| f32_to_bf16(value as f32))
                .collect::<Vec<_>>();
            Expected::Bits(bf16_bytes(&bits))
        }
        _ => Expected::Values(reference),
    };
    results.push(("y".to_owned(), y));
    Ok(results)
}

/// A projection weight in its format: codes in the row-major layout the
/// format's quantizer writes, its block scales and NVFP4 global, and the
/// values the codes decode to, row-major `[N, K]`.
struct EncodedWeight {
    codes: Vec<u8>,
    scales: Option<Vec<u8>>,
    global: Option<f32>,
    decoded: Vec<f64>,
}

/// The most weight elements the kit quantizes with kernel-core's
/// quantizers: they search each code, which takes minutes for a large
/// matrix in the unoptimized kernel-core the tests build. Larger weights
/// get codes and scales drawn directly in the quantizer's format.
const QUANTIZE_LIMIT: usize = 1 << 20;

/// An `[n, k]` weight in `format`. Up to [`QUANTIZE_LIMIT`] elements,
/// kernel-core's quantizers quantize the kit's generated values, as
/// today's GEMM lab cases quantize theirs; each 64-row group and each scale
/// block along K is scaled by its own factor, so neighbouring blocks have
/// different scales and a kernel that reads the wrong scale changes
/// outputs. Larger weights draw codes and per-block scales at random in the
/// format the quantizer gives a small sample ([`drawn_weight`]). Either way
/// the oracle's values are kernel-core's decode of the codes and scales.
fn encode_weight(n: usize, k: usize, format: WeightFormat) -> LabResult<EncodedWeight> {
    let bits = bf16_values(5, n * k);
    if format == WeightFormat::Bf16 {
        return Ok(EncodedWeight {
            codes: bf16_bytes(&bits),
            scales: None,
            global: None,
            decoded: widen(&bits),
        });
    }
    let quantized = if n * k > QUANTIZE_LIMIT {
        drawn_weight(n, k, format)?
    } else {
        quantized_weight(&bits, n, k, format)?
    };
    let decoded = quantized.decoded();
    let global = matches!(quantized.scales, Scales::Ue4M3(_)).then_some(quantized.global_scale);
    let scales = match quantized.scales {
        Scales::F32(values) => values.iter().flat_map(|v| v.to_le_bytes()).collect(),
        Scales::Ue8M0(bytes) | Scales::Ue4M3(bytes) => bytes,
    };
    Ok(EncodedWeight {
        codes: quantized.codes,
        scales: Some(scales),
        global,
        decoded,
    })
}

/// The kit's generated BF16 `bits`, each 64-row group and scale block
/// along K scaled by its own factor, quantized to `format`.
fn quantized_weight(
    bits: &[u16],
    n: usize,
    k: usize,
    format: WeightFormat,
) -> LabResult<Quantized> {
    let block = match format {
        WeightFormat::Fp8Block128x128(_) | WeightFormat::Fp8Rows64x128(_) => 128,
        _ => 32,
    };
    let values = bits
        .iter()
        .enumerate()
        .map(|(at, &bits)| {
            let group = at / k / 64 + at % k / block;
            bf16_to_f32(bits) * (1.0 + 0.37 * (group % 5) as f32)
        })
        .collect::<Vec<_>>();
    match format {
        WeightFormat::Fp8PerTensor => {
            quantize::fp8_weight_per_tensor(&values, n, k, ScaleMode::ExactF32)
        }
        WeightFormat::Fp8PerChannel => {
            quantize::fp8_weight_per_channel(&values, n, k, ScaleMode::ExactF32)
        }
        WeightFormat::Fp8Block128x128(mode) => quantize::fp8_weight_128x128(&values, n, k, mode),
        WeightFormat::Fp8Rows64x128(mode) => quantize::fp8_blocks(&values, n, k, 64, 128, mode),
        WeightFormat::Mxfp8 => quantize::mxfp8(&values, n, k),
        WeightFormat::Mxfp4 => quantize::mxfp4(&values, n, k),
        WeightFormat::Nvfp4 => quantize::nvfp4(&values, n, k),
        WeightFormat::Fp8Block32x32 => Err(Refusal(
            "32x32 UE8M0 fixtures require explicit codes and scales",
        )),
        WeightFormat::Bf16 => Err(Refusal("BF16 weights are not quantized")),
        WeightFormat::F32 => Err(Refusal("F32 weights are not quantized")),
    }
    .map_err(|e| e.0.to_string())
}

/// An `[n, k]` weight in `format` with codes and per-block scales drawn at
/// random: the format, scale storage and NVFP4 global are those
/// kernel-core's quantizer gives a 128 x 128 sample, whose blocks do not
/// depend on the matrix's size for the block-scaled formats. FP8 codes skip
/// the NaN codes; F32 scales are BF16-representable, powers of two when the
/// format's scales are.
fn drawn_weight(n: usize, k: usize, format: WeightFormat) -> LabResult<Quantized> {
    let per_matrix = matches!(
        format,
        WeightFormat::Fp8PerTensor | WeightFormat::Fp8PerChannel
    );
    if per_matrix {
        return Err(format!(
            "{format:?} scales per tensor or channel, so a {n} x {k} weight cannot reuse a \
             sample's format; it is too large for the kit to quantize"
        ));
    }
    let sample = quantized_weight(&bf16_values(8, 128 * 128), 128, 128, format)?;
    let layout = sample.format;
    let mut state = 0x6b69_745f_7765_6967_u64;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let codes = (0..layout.code_bytes(n * k))
        .map(|_| {
            let code = (next() >> 24) as u8;
            // E4M3's NaN codes; packed E2M1 has none.
            if layout.element == kernel_core::codec::SmallFloat::E4M3 && code & 0x7f == 0x7f {
                code - 1
            } else {
                code
            }
        })
        .collect::<Vec<_>>();
    let count = layout.scale_rows(n) * layout.scale_cols(k);
    let pow2 = matches!(
        format,
        WeightFormat::Fp8Block128x128(ScaleMode::PowerOfTwoF32)
            | WeightFormat::Fp8Rows64x128(ScaleMode::PowerOfTwoF32)
    );
    let scales = match sample.scales {
        Scales::F32(_) => Scales::F32(
            (0..count)
                .map(|_| {
                    let exponent = (next() >> 40) % 6;
                    let power = 1.0 / (1u32 << (4 + exponent)) as f32;
                    if pow2 {
                        power
                    } else {
                        bf16_to_f32(f32_to_bf16(power * (1.0 + (next() >> 56) as f32 / 256.0)))
                    }
                })
                .collect(),
        ),
        Scales::Ue8M0(_) => {
            Scales::Ue8M0((0..count).map(|_| 117 + (next() >> 40) as u8 % 7).collect())
        }
        Scales::Ue4M3(_) => Scales::Ue4M3(
            (0..count)
                .map(|_| 0x20 + (next() >> 40) as u8 % 0x28)
                .collect(),
        ),
    };
    Quantized::from_parts(n, k, layout, codes, scales, sample.global_scale)
        .map_err(|e| e.0.to_string())
}

/// The case of a dense projection of one weight over `capacity` rows of
/// BF16 activations: the activations and an `[N, K]` weight from the kit's
/// generators, the weight in its format (BF16, FP8 per tensor, channel or
/// block, MXFP8, MXFP4, NVFP4) with its scales and global from kernel-core's
/// quantizers and its codes in the layout the op reports (row-major, or
/// SM120's BF16 or FP8 fragments), the scratch the op sizes (semaphores
/// zeroed, which the launches must leave zero; roles the op does not size
/// must stay untouched), and the F64 GEMM oracle over the decoded weight,
/// rounded to BF16 first where the contract's decode order scales before
/// the code (`ScaleThenCode`), within its tolerance.
pub fn projection_case<'a>(
    spec: &OperationSpec,
    capacity: u32,
    op: Prepared<ProjectionBuffers<View>>,
) -> LabResult<OpCase<'a>> {
    let unsupported = || -> LabResult<OpCase<'a>> {
        Err(
            "the kit's projection case covers dense projections of one weight with BF16 \
             activations, no transforms and no epilogue"
                .into(),
        )
    };
    let [weight] = spec.weights.as_slice() else {
        return unsupported();
    };
    if spec.kind != OperationKind::Dense
        || !weight.transforms.is_empty()
        || spec.activations.len() != 1
        || !spec.activations[0].is_bf16()
        || spec.epilogues != Epilogues::default()
    {
        return unsupported();
    }
    let &[n, k] = weight.logical_shape.as_slice() else {
        return unsupported();
    };
    let (n, k) = (n as usize, k as usize);
    let output = spec.arithmetic.output;
    let info = op.info().clone();
    let encoded = encode_weight(n, k, weight.format)?;
    let fragments = |format| {
        kernel_core::weight_transform::fragment::pack(
            FragmentWeightLayout { rows: n, k, format },
            &encoded.codes,
        )
        .map_err(|e| format!("fragment packing: {e:?}"))
    };
    let codes = match info.layout_of("weights") {
        None => encoded.codes.clone(),
        Some(layout) if *layout == WeightLayout::row_major() => encoded.codes.clone(),
        Some(layout) if layout.id == "bf16-fragment" && layout.version == 1 => {
            fragments(FragmentWeightFormat::Bf16)?
        }
        Some(layout) if layout.id == "fp8-fragment" && layout.version == 1 => {
            fragments(FragmentWeightFormat::Fp8E4M3)?
        }
        Some(layout) => {
            return Err(format!(
                "the kit has no packer for weight layout {} v{}",
                layout.id, layout.version
            ));
        }
    };
    let sized = ["output", "scratch.partial", "scratch.semaphores"];
    if let Some((path, _)) = info
        .bytes
        .iter()
        .find(|(path, _)| !sized.contains(&path.as_str()))
    {
        return Err(format!("the kit's projection case binds no {path}"));
    }
    let bytes = |path| info.bytes_of(path).map(|bytes| bytes as usize);
    let roles = ProjectionBuffers {
        activation: Spec::input(
            DType::Bf16,
            (Rows::Live, k),
            bf16_bytes(&bf16_values(4, capacity as usize * k)),
        ),
        live_rows: Spec::live(),
        output: Spec::output(output, (Rows::Live, n)),
        residual: None,
        weights: QuantizedWeights {
            codes: Spec::weight(codes),
            local_scales: encoded.scales.clone().map(Spec::weight),
            global_scales: encoded
                .global
                .map(|global| Spec::weight(global.to_le_bytes().to_vec())),
        },
        activation_global: None,
        routing: None,
        scratch: ProjectionScratch {
            quantizer_status_row_offset: 0,
            quantizer_status: None,
            partial: bytes("scratch.partial").map_or_else(|| Spec::untouched(16), Spec::scratch),
            semaphores: bytes("scratch.semaphores")
                .map_or_else(|| Spec::untouched(16), Spec::zeroed),
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
    // The images never change between runs of a case without ID inputs,
    // so the oracle runs once.
    let round = spec.arithmetic.decode_order == DecodeOrder::ScaleThenCode;
    let weights = encoded
        .decoded
        .iter()
        .map(|&value| {
            if round {
                f64::from(bf16_to_f32(kernel_core::codec::f64_to_bf16(value)))
            } else {
                value
            }
        })
        .collect::<Vec<_>>();
    let reference = RefCell::new(None::<(Vec<u8>, OracleOutput)>);
    let oracle = move |images: &Images, _: u32| -> LabResult<Vec<(String, Expected)>> {
        let activation = images.bytes("activation")?;
        let mut cached = reference.borrow_mut();
        if cached.as_ref().is_none_or(|(key, _)| key != activation) {
            let x = widen(&images.u16s("activation")?);
            let shape = GemmShape {
                m: capacity as usize,
                n,
                k,
                experts: 1,
            };
            let epilogue = GemmEpilogue {
                residual: None,
                swiglu: None,
            };
            let result =
                oracle::gemm(&x, &weights, shape, &[], epilogue).map_err(|e| e.0.to_string())?;
            *cached = Some((activation.to_vec(), result));
        }
        let result = cached.as_ref().expect("computed above").1.clone();
        Ok(vec![("output".to_owned(), Expected::Values(result))])
    };
    Ok(OpCase::new(op, capacity, roles)?.oracle(oracle))
}

/// The rows of the table an embedding case binds: a vocabulary shard
/// small enough to upload, which the op reads through its `table_rows`.
pub const EMBEDDING_TABLE_ROWS: u32 = 64;

/// The case of an embedding gather over `capacity` rows from a table of
/// [`EMBEDDING_TABLE_ROWS`] whose first row is vocabulary row `row_offset`:
/// distinct table rows by ID, the last live row a hole (-1, which lies
/// outside every table and gathers zeros), and the exact gather as bits.
/// IDs index the table's rows when the offset is zero, so the run also
/// reruns them moved to the top of the table; a shard's IDs are vocabulary
/// rows, which the rerun does not move.
pub fn embedding_case<'a>(
    op: &RowQuery,
    row_offset: u32,
    capacity: u32,
    prepared: Prepared<EmbeddingBuffers<View>>,
) -> LabResult<OpCase<'a>> {
    let RowQuery::EmbeddingGather { vocab, width } = *op else {
        return Err("an embedding case needs an embedding gather".into());
    };
    let table_rows = EMBEDDING_TABLE_ROWS.min(vocab.saturating_sub(row_offset));
    if table_rows == 0 || capacity > table_rows {
        return Err(format!(
            "an embedding case gathers distinct rows: {capacity} rows from {table_rows}"
        ));
    }
    let (rows, width) = (table_rows as usize, width as usize);
    let ids = (0..capacity as usize)
        .map(|row| ((row * 7 + 3) % rows) as i32 + row_offset as i32)
        .collect::<Vec<_>>();
    let stride = width * 2;
    let pools: &[(&str, usize)] = if row_offset == 0 {
        &[("table", stride)]
    } else {
        &[]
    };
    let roles = EmbeddingBuffers {
        table: Spec::input(
            DType::Bf16,
            (Rows::Fixed(rows), width),
            bf16_bytes(&bf16_values(6, rows * width)),
        ),
        ids: Spec::ids((Rows::Live, 1), ids, pools, Holes::LastLive),
        output: Spec::output(DType::Bf16, (Rows::Live, width)),
        live_rows: Spec::live(),
        table_rows,
    };
    let oracle = move |images: &Images, _: u32| -> LabResult<Vec<(String, Expected)>> {
        let table = images.bytes("table")?;
        let mut output = vec![0; capacity as usize * stride];
        for (row, id) in images.i32s("ids")?.into_iter().enumerate() {
            let local = i64::from(id) - i64::from(row_offset);
            if (0..i64::from(table_rows)).contains(&local) {
                let local = local as usize;
                output[row * stride..][..stride]
                    .copy_from_slice(&table[local * stride..][..stride]);
            }
        }
        Ok(vec![("output".to_owned(), Expected::Bits(output))])
    };
    Ok(OpCase::new(prepared, capacity, roles)?.oracle(oracle))
}

/// The required embedding gather `op` from a table starting at vocabulary
/// row `row_offset`.
pub fn embedding<'a>(name: String, op: RowQuery, row_offset: u32, need: Need) -> Required<'a> {
    Required::new(
        name,
        need,
        move |backend, need| backend.embedding_gather(&op, row_offset, need),
        move |prepared| embedding_case(&op, row_offset, need.capacity, prepared),
    )
}

/// The logits past the live width in a logits case: above every live
/// logit, so a launch that reads them picks them.
const PAD_LOGIT: f32 = 4.0;

/// The case of a logits argmax or top-k over `capacity` rows: F32 logits
/// from the kit's BF16 generator (few distinct values, so most rows tie
/// and the lower-index rule decides), the last few vocabulary columns
/// padding past the device live width, and the contract's top-k as exact
/// ID bits. The case binds no log-probabilities.
pub fn logits_case<'a>(
    op: &RowQuery,
    capacity: u32,
    prepared: Prepared<LogitsBuffers<View>>,
) -> LabResult<OpCase<'a>> {
    let (vocab, k) = match *op {
        RowQuery::LogitsArgmax { vocab } => (vocab as usize, 1),
        RowQuery::LogitsTopK { vocab, k } => (vocab as usize, k as usize),
        _ => return Err("a logits case needs a logits argmax or top-k".into()),
    };
    let live = vocab.saturating_sub(5).max(k);
    let rows = capacity as usize;
    let mut logits = bf16_values(7, rows * vocab)
        .into_iter()
        .map(bf16_to_f32)
        .collect::<Vec<_>>();
    for row in logits.chunks_exact_mut(vocab) {
        row[live..].fill(PAD_LOGIT);
    }
    let roles = LogitsBuffers {
        logits: Spec::input(
            DType::F32,
            (Rows::Live, vocab),
            logits.iter().flat_map(|v| v.to_le_bytes()).collect(),
        ),
        live_width: Spec::input(
            DType::I32,
            (Rows::Fixed(1), 1),
            (live as u32).to_le_bytes().to_vec(),
        ),
        ids: Spec::output(DType::I32, (Rows::Live, k)),
        log_probs: None,
        live_rows: Spec::live(),
    };
    let reference = RefCell::new(None::<(Vec<u8>, Vec<u8>)>);
    let oracle = move |images: &Images, _: u32| -> LabResult<Vec<(String, Expected)>> {
        let key = images.bytes("logits")?;
        let mut cached = reference.borrow_mut();
        if cached.as_ref().is_none_or(|(old, _)| old != key) {
            let mut ids = Vec::with_capacity(rows * k * 4);
            for row in images.f32s("logits")?.chunks_exact(vocab) {
                let row = row.iter().map(|&v| f64::from(v)).collect::<Vec<_>>();
                let top =
                    oracle::logits_topk_log_probs(&row, live, k).map_err(|e| e.0.to_string())?;
                ids.extend(top.ids.iter().flat_map(|&id| (id as i32).to_le_bytes()));
            }
            *cached = Some((key.to_vec(), ids));
        }
        let ids = cached.as_ref().expect("computed above").1.clone();
        Ok(vec![("ids".to_owned(), Expected::Bits(ids))])
    };
    Ok(OpCase::new(prepared, capacity, roles)?.oracle(oracle))
}

/// The required logits argmax or top-k `op`.
pub fn logits<'a>(name: String, op: RowQuery, need: Need) -> Required<'a> {
    Required::new(
        name,
        need,
        move |backend, need| backend.logits_top_k(&op, need),
        move |prepared| logits_case(&op, need.capacity, prepared),
    )
}

/// The required RMSNorm `requirement` over `need`.
pub fn rmsnorm<'a>(name: String, requirement: RmsNormRequirement, need: Need) -> Required<'a> {
    Required::new(
        name,
        need,
        move |backend, need| backend.rmsnorm(&requirement, need),
        move |op| rmsnorm_case(requirement, need.capacity, op),
    )
}

/// The required projection `spec` at `usage` over `need`.
pub fn projection<'a>(
    name: String,
    spec: OperationSpec,
    usage: OperationUse,
    need: Need,
) -> Required<'a> {
    let case = spec.clone();
    Required::new(
        name,
        need,
        move |backend, need| {
            backend
                .projection(&spec, usage, need)
                .map(|bound| bound.handle)
        },
        move |op| projection_case(&case, usage.capacity, op),
    )
}

/// A dense BF16 projection of K = `k` to N = `n` columns, row-major
/// weights, with `output` and one use at `usage`.
pub fn bf16_projection(n: u64, k: u64, output: DType, usage: OperationUse) -> OperationSpec {
    let codes = TensorSpan {
        tensor: "matrix".into(),
        dtype: DType::Bf16,
        shape: vec![n, k],
        offset: 0,
        byte_length: n * k * dtype_bits(DType::Bf16) / 8,
    };
    OperationSpec {
        id: "dense_projection".into(),
        kind: OperationKind::Dense,
        target: Target::Sm120f,
        weights: vec![WeightSpec {
            codes,
            logical_shape: vec![n, k],
            format: WeightFormat::Bf16,
            encoding: CodeEncoding::Bf16LittleEndian,
            layout: WeightLayout::row_major(),
            local_scale: None,
            global_scale: None,
            transforms: vec![],
        }],
        geometry: None,
        activations: vec![ActivationSpec::bf16()],
        arithmetic: ArithmeticSpec {
            weight_operand: DType::Bf16,
            accumulator: DType::F32,
            output,
            decode_order: DecodeOrder::ScaleThenCode,
            rounding: Rounding::NearestEven,
            numerical_contract: "dense_bf16".into(),
        },
        epilogues: Epilogues::default(),
        uses: vec![usage],
    }
}

/// The RMSNorm entries of `catalogs` as requirements, the first entry of
/// each requirement: SM120's selection for the kit.
pub fn catalog_rmsnorms(catalogs: &Catalogs) -> Vec<(RmsNormRequirement, String)> {
    let mut entries: Vec<(RmsNormRequirement, String)> = Vec::new();
    for row in kernels_sm120::row::catalog(catalogs) {
        let RowQuery::RmsNorm {
            width,
            eps,
            kind,
            fused_residual,
            output_quantization: None,
        } = row.query
        else {
            continue;
        };
        let numerics = match kind {
            RmsNormKind::HfBf16Intermediate => RmsNormNumerics::HF_ORDERED,
            _ => RmsNormNumerics::TREE_256,
        };
        let requirement = RmsNormRequirement::bf16(width, eps, kind, fused_residual, numerics);
        if !entries.iter().any(|(other, _)| *other == requirement) {
            entries.push((requirement, row.symbol));
        }
    }
    entries
}

/// The dense entries of `catalogs` with BF16 activations, no epilogue and
/// a BF16 or F32 output, as SM120 offers at their full capacity: the
/// `count` smallest BF16 ones by `M * N * K`, and the smallest of every
/// other weight format and fragment layout the catalogs serve.
pub fn catalog_projections(catalogs: &Catalogs, count: usize) -> LabResult<Vec<ProjectionOffer>> {
    let mut offers = Vec::new();
    for variant in kernels_sm120::gemm::catalog(catalogs) {
        let StaticQuery::Gemm(query) = variant.key.query else {
            continue;
        };
        if query.grouped.is_some()
            || query.activation != kernel_core::contract::ActivationFormat::Bf16
            || query.epilogues != Epilogues::default()
            || !matches!(query.output, DType::Bf16 | DType::F32)
        {
            continue;
        }
        let usage = OperationUse {
            phase: if query.m_capacity <= 32 {
                ExecutionPhase::Decode
            } else {
                ExecutionPhase::Prefill
            },
            capacity: query.m_capacity,
            graph: false,
        };
        let size = u64::from(query.m_capacity) * u64::from(query.n) * u64::from(query.k);
        let offer = KernelOffer::catalog_dense(variant.instantiation.clone(), usage)
            .map_err(|e| format!("{}: {e}", variant.symbol))?;
        let form = format!(
            "{:?} {}",
            query.weight,
            kernels_sm120::gemm::fragment_weight_enabled(&variant.key)
        );
        offers.push((
            query.weight == WeightFormat::Bf16,
            form,
            size,
            variant.symbol.clone(),
            offer,
        ));
    }
    offers.sort_by(|left, right| (&left.1, left.2, &left.3).cmp(&(&right.1, right.2, &right.3)));
    let mut chosen = Vec::<(String, usize)>::new();
    let mut taken = Vec::new();
    for (bf16, form, _, _, offer) in offers {
        let limit = if bf16 { count } else { 1 };
        match chosen.iter_mut().find(|(other, _)| *other == form) {
            Some((_, held)) if *held >= limit => continue,
            Some((_, held)) => *held += 1,
            None => chosen.push((form, 1)),
        }
        taken.push(ProjectionOffer {
            offer,
            copy: None,
            dense: DenseForm::Tensor,
            live_output: false,
        });
    }
    Ok(taken)
}

/// SM120 over `catalogs`: every catalog RMSNorm, embedding gather and
/// logits argmax or top-k entry, the RoPE and KV write entries
/// ([`rope_kv::sm120_entries`]), the attention variants
/// ([`attention::variants`]), the recurrent entries
/// ([`recurrent::sm120_entries`]), and the given latent attention entries
/// (as [`latent::sm120_entries`] selects them) and projection offers. The
/// kit authenticates no package, so the offers' native dependency digest
/// is a placeholder.
pub fn sm120(
    catalogs: &Catalogs,
    latent: &[kernels_sm120::latent::Entry],
    offers: &[ProjectionOffer],
) -> LabResult<Sm120> {
    let mut backend = Sm120::new();
    for (requirement, symbol) in catalog_rmsnorms(catalogs) {
        backend = backend.with_rmsnorm(requirement, symbol);
    }
    // One catalog entry per gather (and form) and logits head, since equal
    // queries under two symbols are an SM120 tie: the one declaring the
    // most exactness, so the kit checks the most properties, the first of
    // equals.
    let mut chosen: Vec<((RowQuery, bool), kernels_sm120::row::RowInstantiation)> = Vec::new();
    for row in kernels_sm120::row::catalog(catalogs) {
        if !matches!(
            row.query,
            RowQuery::EmbeddingGather { .. }
                | RowQuery::LogitsArgmax { .. }
                | RowQuery::LogitsTopK { .. }
        ) {
            continue;
        }
        // A sharded entry instantiates the template with `true`.
        let key = (row.query, row.body.contains(",true>"));
        match chosen.iter_mut().find(|(other, _)| *other == key) {
            Some((_, held)) if row.invariance.claims().len() > held.invariance.claims().len() => {
                *held = row;
            }
            Some(_) => {}
            None => chosen.push((key, row)),
        }
    }
    for ((_, sharded), row) in chosen {
        backend = if sharded {
            backend.with_sharded_embedding(row)
        } else {
            backend.with_row(row)
        };
    }
    backend = backend
        .with_rope_kv(rope_kv::sm120_entries(catalogs))
        .with_attention(attention::sm120_entries(attention::variants(catalogs)));
    for entry in latent {
        backend = backend.with_latent(entry.clone());
    }
    backend = backend.with_linear(recurrent::sm120_entries(catalogs));
    if !offers.is_empty() {
        let kernels = offers.iter().map(|o| o.offer.clone()).collect::<Vec<_>>();
        let available = capabilities(&kernels)
            .map_err(|e| e.0.to_string())?
            .entries();
        let list = OfferList::new(offers.to_vec(), available, "0".repeat(64))
            .map_err(|e| e.0.to_string())?;
        backend = backend.with_projections(Box::new(list));
    }
    Ok(backend)
}

/// The pilot's required ops: every catalog RMSNorm requirement and the two
/// numerics on a narrow row, with and without a residual; every catalog
/// embedding gather over a whole table and logits argmax or top-k; BF16 row-major projections with
/// BF16 and F32 outputs; and the projections of `offers`.
/// No op is marked required: each backend's run states which it must
/// prepare.
pub fn pilot<'a>(catalogs: &Catalogs, offers: &[ProjectionOffer]) -> LabResult<Vec<Required<'a>>> {
    let mut requirements = catalog_rmsnorms(catalogs)
        .into_iter()
        .map(|(requirement, _)| requirement)
        .collect::<Vec<_>>();
    for fused in [false, true] {
        for (kind, numerics) in [
            (RmsNormKind::HfBf16Intermediate, RmsNormNumerics::HF_ORDERED),
            (RmsNormKind::Plain, RmsNormNumerics::TREE_256),
        ] {
            let requirement = RmsNormRequirement::bf16(512, 1e-6, kind, fused, numerics);
            if !requirements.contains(&requirement) {
                requirements.push(requirement);
            }
        }
    }
    let mut ops = requirements
        .into_iter()
        .map(|requirement| {
            let name = format!(
                "rmsnorm {:?} width {} eps {:e}{}",
                requirement.kind,
                requirement.width,
                f32::from_bits(requirement.epsilon_bits),
                if requirement.fused_residual {
                    " residual"
                } else {
                    ""
                }
            );
            // Verify rows rely on a row's bits at any row count.
            let need = Need::rows(ROW_CAPACITY).with_exact(Invariance::one_row(ROW_CAPACITY));
            rmsnorm(name, requirement, need)
        })
        .collect::<Vec<_>>();
    let mut gathers = Vec::new();
    for row in kernels_sm120::row::catalog(catalogs) {
        if let RowQuery::EmbeddingGather { vocab, width } = row.query
            && !gathers.contains(&row.query)
        {
            gathers.push(row.query);
            let name = format!("embedding gather {vocab}x{width}");
            ops.push(embedding(name, row.query, 0, Need::rows(ROW_CAPACITY)));
        }
    }
    let mut heads = Vec::new();
    for row in kernels_sm120::row::catalog(catalogs) {
        let name = match row.query {
            RowQuery::LogitsArgmax { vocab } => format!("logits argmax {vocab}"),
            RowQuery::LogitsTopK { vocab, k } => format!("logits top-{k} {vocab}"),
            _ => continue,
        };
        if !heads.contains(&row.query) {
            heads.push(row.query);
            ops.push(logits(name, row.query, Need::rows(ROW_CAPACITY)));
        }
    }
    let decode = OperationUse {
        phase: ExecutionPhase::Decode,
        capacity: 16,
        graph: true,
    };
    for output in [DType::Bf16, DType::F32] {
        ops.push(projection(
            format!("projection bf16 row-major 512 to 256 {output:?}"),
            bf16_projection(256, 512, output, decode),
            decode,
            Need::rows(decode.capacity),
        ));
    }
    for offer in offers {
        let capability = offer.offer.capability().map_err(|e| e.0.to_string())?;
        let spec = kernels_sm120::precision::generated::comparison::operation(&capability)?;
        let name = format!(
            "projection {:?} {} {}x{} capacity {}",
            capability.weight,
            capability.stages.join("+"),
            capability.geometry.rows,
            capability.geometry.k,
            capability.usage.capacity
        );
        let need = Need::rows(capability.usage.capacity);
        ops.push(projection(name, spec, capability.usage, need));
    }
    Ok(ops)
}

/// Test support for case authors and the families' `checks`: a family
/// op's case on a backend, and the images and write maps the property
/// checks see when they move rows. Each panics on a failure.
pub mod testing {
    use crate::conformance::{Images, Rows};
    use crate::kit::{OpCase, Required, Tag};
    use kernel_core::backend::Backend;
    use kernel_core::codec::bf16_to_f32;
    use kernel_core::contract::DType;
    use std::collections::BTreeSet;

    /// The largest gap, relative to each value's magnitude (at least 1),
    /// between the verify step's oracle results and each decode launch's
    /// where the launch compares them, at the case's capacity and claim
    /// `max`: a CPU witness that the case's decode mapping gives each
    /// verify row the decode launch of its own row. With `shift`, launch
    /// `k` runs the images of launch `k + 1`, which a mapping that tells
    /// rows apart gives large gaps. With `fault`, the verify step's oracle
    /// runs over `fault` of its images, as a faulty verify launch would.
    pub fn decode_gaps(
        case: &OpCase<'_>,
        max: u32,
        shift: bool,
        fault: Option<&dyn Fn(&Images) -> Images>,
    ) -> Vec<f64> {
        let Some(Ok(decoded)) = &case.decoded else {
            panic!("the case has no decode mapping");
        };
        let live = case.capacity;
        let bytes = |held: std::collections::BTreeMap<String, (DType, Vec<u8>)>| {
            held.into_iter()
                .map(|(name, (_, bytes))| (name, bytes))
                .collect()
        };
        let execute = |images: &Images| {
            let images = fault.map_or_else(|| images.clone(), |fault| fault(images));
            case.case.oracle_bytes(&images, live).map(bytes)
        };
        let (images, results, launches) = decoded
            .plan(&case.case.images(), live, max, &execute)
            .unwrap();
        let dtypes = case.case.oracle_bytes(&images, live).unwrap();
        let values = |dtype: DType, bytes: &[u8]| -> Vec<f64> {
            match dtype {
                DType::Bf16 => bytes
                    .chunks_exact(2)
                    .map(|half| f64::from(bf16_to_f32(u16::from_le_bytes([half[0], half[1]]))))
                    .collect(),
                DType::F32 => bytes
                    .chunks_exact(4)
                    .map(|word| f64::from(f32::from_le_bytes(word.try_into().unwrap())))
                    .collect(),
                other => panic!("no values of {other:?}"),
            }
        };
        let count = launches.len();
        (0..count)
            .map(|at| {
                let run = &launches[if shift { (at + 1) % count } else { at }];
                let decode = decoded.case.case.oracle_bytes(&run.images, 1).unwrap();
                let mut gap = 0.0_f64;
                for compared in &launches[at].compared {
                    let (dtype, _) = &dtypes[&compared.verify.0];
                    let left = &results[&compared.verify.0];
                    let (_, right) = &decode[&compared.decode.0];
                    let left = values(*dtype, &left[compared.verify.1.clone()]);
                    let right = values(*dtype, &right[compared.decode.1.clone()]);
                    for (a, b) in left.iter().zip(&right) {
                        gap = gap.max((a - b).abs() / a.abs().max(1.0));
                    }
                }
                gap
            })
            .collect()
    }

    /// `required`'s case as `backend` prepares it.
    pub fn prepared<'a>(backend: &dyn Backend, required: &Required<'a>) -> OpCase<'a> {
        (required.prepare)(backend, required.need)
            .unwrap_or_else(|refused| panic!("{}: {refused}", required.name))
            .unwrap_or_else(|error| panic!("{}: {error}", required.name))
    }

    /// The case's images with each live-row input's row `row` moved to the
    /// front, as the property checks move one row.
    pub fn front(case: &OpCase<'_>, images: &Images, row: usize) -> Images {
        let mut moved = images.clone();
        for leaf in &case.leaves {
            if leaf.tag != Tag::Input || leaf.rows != Rows::Live {
                continue;
            }
            let mut bytes = images.bytes(&leaf.path).unwrap().to_vec();
            let source = bytes[row * leaf.row_bytes..][..leaf.row_bytes].to_vec();
            bytes[..leaf.row_bytes].copy_from_slice(&source);
            moved = moved.with(&leaf.path, bytes);
        }
        moved
    }

    /// The bytes of state `name` a launch at `live` over `images` writes.
    pub fn writes(case: &OpCase<'_>, name: &str, images: &Images, live: u32) -> BTreeSet<usize> {
        case.case
            .state_writes(name, images, live)
            .unwrap()
            .into_iter()
            .flatten()
            .collect()
    }

    /// The state bytes row `row` alone writes: those at `row + 1` live rows
    /// that `row` live rows do not write.
    pub fn row_writes(case: &OpCase<'_>, name: &str, images: &Images, row: u32) -> BTreeSet<usize> {
        let before = writes(case, name, images, row);
        writes(case, name, images, row + 1)
            .difference(&before)
            .copied()
            .collect()
    }

    /// A moved row's state writes are the row's own: row `row` moved to the
    /// front and launched alone writes where it wrote among the others.
    pub fn assert_maps_follow_rows(case: &OpCase<'_>, name: &str, row: usize) {
        let images = case.case.images();
        let own = row_writes(case, name, &images, row as u32);
        assert!(!own.is_empty(), "row {row} writes no {name}");
        let moved = front(case, &images, row);
        let alone = writes(case, name, &moved, 1);
        assert!(
            alone == own,
            "row {row} moved to the front writes {name} from byte {:?}, not {:?}",
            alone.first(),
            own.first()
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The RMSNorm case's HF reference sums in the order the requirement's
    /// numerics name: on a row whose tree and serial sums of squares differ
    /// (1.0078125 then 4095 values of 2^-13), a tree requirement expects
    /// the tree order's bits and refuses the serial ones at column 1, and
    /// the other way round.
    #[test]
    fn the_rmsnorm_case_expects_the_bits_of_its_numerics_order() {
        let mut x = vec![f32_to_bf16(2.0_f32.powi(-13)); 4096];
        x[0] = 0x3f81;
        let weight = vec![f32_to_bf16(1.0); 4096];
        let images = Images::default()
            .with("x", bf16_bytes(&x))
            .with("weight", bf16_bytes(&weight));
        let expected = |numerics| {
            let requirement = RmsNormRequirement::bf16(
                4096,
                1e-6,
                RmsNormKind::HfBf16Intermediate,
                false,
                numerics,
            );
            let results = rmsnorm_oracle(&requirement, &images).unwrap();
            let [(name, Expected::Bits(bits))] = results.as_slice() else {
                panic!("one exact output");
            };
            assert_eq!(name, "y");
            let reference = requirement
                .oracle(&widen(&x), &widen(&weight), 1)
                .unwrap()
                .values
                .iter()
                .map(|&v| f32_to_bf16(v as f32))
                .collect::<Vec<_>>();
            assert_eq!(*bits, bf16_bytes(&reference));
            reference
        };
        let tree = expected(RmsNormNumerics::HF_TREE_256);
        let ordered = expected(RmsNormNumerics::HF_ORDERED);
        assert_eq!(oracle::check_exact(&ordered, &tree), Err(1));
        assert_eq!(oracle::check_exact(&tree, &ordered), Err(1));
    }

    /// Drawn weights fit their quantizer's format: kernel-core assembles
    /// and decodes them, and their scales differ from block to block.
    #[test]
    fn drawn_weights_decode_in_every_block_scaled_format() {
        for format in [
            WeightFormat::Fp8Block128x128(ScaleMode::ExactF32),
            WeightFormat::Fp8Block128x128(ScaleMode::PowerOfTwoF32),
            WeightFormat::Fp8Rows64x128(ScaleMode::ExactF32),
            WeightFormat::Mxfp8,
            WeightFormat::Mxfp4,
            WeightFormat::Nvfp4,
        ] {
            let weight = drawn_weight(256, 384, format).expect("a drawn weight");
            let decoded = weight.decoded();
            assert_eq!(decoded.len(), 256 * 384, "{format:?}");
            assert!(decoded.iter().all(|v| v.is_finite()), "{format:?}");
            let distinct = match &weight.scales {
                Scales::F32(values) => values
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<std::collections::BTreeSet<_>>()
                    .len(),
                Scales::Ue8M0(bytes) | Scales::Ue4M3(bytes) => bytes
                    .iter()
                    .collect::<std::collections::BTreeSet<_>>()
                    .len(),
            };
            assert!(distinct > 1, "{format:?}");
        }
        assert!(drawn_weight(256, 384, WeightFormat::Fp8PerTensor).is_err());
    }
}
