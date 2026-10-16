//! GPU-gated grouped expert GEMM checks. CPU CI compiles these cases only.

use kernel_core::codec::{SmallFloat, bf16_to_f32, decode_ue8m0_f64, f32_to_bf16};
use kernel_core::contract::{
    Argument, DType, LaunchPlan, OutputQuantization, RouterKind, RuntimeCounts, ScaleMode,
    SwigluColumns, Validate, WeightFormat,
};
use kernel_core::oracle::{self, ExpertRowRange, GemmEpilogue, GemmShape, OracleOutput};
use kernel_core::quantize::{self, QuantFormat, Quantized, ScaleFormat, Scales};
use kernel_core::transform::{Mxfp4MoeFragmentLayout, Transform};
use kernel_lab::launch::{self, Bindings};
use kernel_lab::poison::{self, Phase as ReplayPhase, WriteMap};
use kernel_lab::{LabResult, Session, check_output, gpu_test, time};
use kernels_sm120::Profile;
use kernels_sm120::gemm::{self, GemmBuffers};
use kernels_sm120::moe::{self, InputPath, Phase, Projection, Variant};

use kernel_lab::poison as replay;
#[path = "moe_replay.rs"]
mod replay_maps;

use runtime_cuda::{DeviceBuffer, Function, Module};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

const TOP_K: usize = 8;
const HIDDEN: usize = 4096;
const HANDLES: [u64; 15] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
const WEIGHT_HANDLE: u64 = 16;
/// UE8M0 scale of the sparse graph fixtures. Each layer feeds the next, so
/// the scale keeps activations decaying rather than overflowing to NaN,
/// which would route every token to an invalid expert.
const SPARSE_SCALE: u8 = 112;

fn err(error: impl std::fmt::Display) -> String {
    error.to_string()
}

fn upload_bytes(session: &Session, bytes: &[u8]) -> LabResult<DeviceBuffer> {
    let buffer = session.allocate(bytes.len().max(1))?;
    if !bytes.is_empty() {
        buffer
            .copy_from_bytes(bytes, &session.stream)
            .map_err(err)?;
    }
    Ok(buffer)
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect()
}

fn i32_bytes(values: &[i32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect()
}

fn download_i32(session: &Session, buffer: &DeviceBuffer, len: usize) -> LabResult<Vec<i32>> {
    let mut bytes = vec![0; len * 4];
    buffer
        .copy_to_bytes(&mut bytes, &session.stream)
        .map_err(err)?;
    Ok(bytes
        .chunks_exact(4)
        .map(|v| i32::from_ne_bytes(v.try_into().unwrap()))
        .collect())
}

fn download_f32(session: &Session, buffer: &DeviceBuffer, len: usize) -> LabResult<Vec<f32>> {
    session.download_f32(buffer, len)
}

fn sample(seed: u64) -> f32 {
    let mut x = seed.wrapping_add(0x9e3779b97f4a7c15);
    x ^= x >> 30;
    x = x.wrapping_mul(0xbf58476d1ce4e5b9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94d049bb133111eb);
    x ^= x >> 31;
    ((x >> 40) as i32 as f32 / 8_388_608.0) - 1.0
}

fn variant(
    profile: Profile,
    phase: Phase,
    projection: Projection,
    path: InputPath,
    weight: WeightFormat,
    tokens: usize,
) -> LabResult<Variant> {
    let bucket = if phase == Phase::Prefill {
        8192
    } else {
        tokens as u32
    };
    moe::catalog(kernel_lab::catalogs())
        .into_iter()
        .find(|v| {
            v.profile == profile
                && v.phase == phase
                && v.projection == projection
                && v.path == path
                && v.query.weight == weight
                && v.tokens_capacity == bucket
                && v.query.epilogues.swiglu != Some(SwigluColumns::Interleaved)
                && v.query.epilogues.output_quantization.is_none()
        })
        .ok_or_else(|| {
            format!(
                "missing MoE variant for {} tokens={tokens}",
                profile.as_str()
            )
        })
}

fn quantize_weight(
    values: &[f32],
    rows: usize,
    k: usize,
    format: WeightFormat,
) -> LabResult<Quantized> {
    match format {
        WeightFormat::Mxfp4 => quantize::mxfp4(values, rows, k),
        WeightFormat::Mxfp8 => quantize::mxfp8(values, rows, k),
        WeightFormat::Nvfp4 => quantize::nvfp4(values, rows, k),
        WeightFormat::Fp8Block128x128(ScaleMode::ExactF32) => {
            quantize::fp8_weight_128x128(values, rows, k, ScaleMode::ExactF32)
        }
        _ => return Err("unsupported MoE test format".into()),
    }
    .map_err(err)
}

/// MXFP4 rows (even K in the low nibble) with row-major UE8M0 scales per 32
/// values, decoded.
fn mxfp4_decoded(rows: usize, k: usize, codes: &[u8], scales: &[u8]) -> LabResult<Vec<f64>> {
    let format = QuantFormat {
        element: SmallFloat::E2M1,
        block: (1, 32),
        scale: ScaleFormat::Ue8M0,
        chunk_rows: None,
    };
    let scales = Scales::Ue8M0(scales.to_vec());
    let weight = Quantized::from_parts(rows, k, format, codes.to_vec(), scales, 1.0);
    Ok(weight.map_err(err)?.decoded())
}

/// SwiGLU of one expert's MXFP4 gate/up projection [1024, HIDDEN] (gate rows,
/// then up rows) over `tokens` BF16 rows, in f64: [tokens, 512].
fn mxfp4_gate_up(input: &[u16], tokens: usize, codes: &[u8], scales: &[u8]) -> LabResult<Vec<f64>> {
    let x = input[..tokens * HIDDEN]
        .iter()
        .map(|&v| f64::from(bf16_to_f32(v)))
        .collect::<Vec<_>>();
    let shape = GemmShape {
        m: tokens,
        n: 1024,
        k: HIDDEN,
        experts: 1,
    };
    let epilogue = GemmEpilogue {
        residual: None,
        swiglu: Some(SwigluColumns::ConcatenatedHalves),
    };
    let weight = mxfp4_decoded(1024, HIDDEN, codes, scales)?;
    Ok(oracle::gemm(&x, &weight, shape, &[], epilogue)
        .map_err(err)?
        .values)
}

struct WeightBank {
    codes: Vec<u8>,
    scales: Vec<u8>,
    f32_scales: Vec<f32>,
    decoded: [Vec<f64>; 2],
    logical_bytes: usize,
}

impl WeightBank {
    fn sparse_fragment_bench(v: &Variant, layer: usize, stored_experts: usize) -> LabResult<Self> {
        if v.query.weight != WeightFormat::Mxfp4 || v.weight_fragment_layout().is_none() {
            return Err("sparse MoE benchmark requires MiMo fragment weights".into());
        }
        let experts = v.query.grouped.ok_or("missing expert count")?.experts as usize;
        if stored_experts < TOP_K || stored_experts > experts {
            return Err("sparse MoE expert count is outside the supported range".into());
        }
        let n = v.query.n as usize;
        let k = v.query.k as usize;
        let mut codes = vec![0_u8; stored_experts * n * k / 2];
        for expert in 0..stored_experts {
            let code = ((layer + expert * 3) % 7 + 1) as u8;
            let pattern = code | (code << 4);
            codes[expert * n * k / 2..(expert + 1) * n * k / 2].fill(pattern);
        }
        let scales = vec![SPARSE_SCALE; stored_experts * n * k / 32];
        Ok(Self {
            logical_bytes: codes.len() + scales.len(),
            codes,
            scales,
            f32_scales: vec![1.0],
            decoded: [Vec::new(), Vec::new()],
        })
    }

    fn new(v: &Variant, seed: u64) -> LabResult<Self> {
        let experts = v.query.grouped.ok_or("missing grouped shape")?.experts as usize;
        let n = v.query.n as usize;
        let k = v.query.k as usize;
        let format = v.query.weight;
        let template_rows = if matches!(format, WeightFormat::Fp8Block128x128(_)) {
            128
        } else {
            16
        };
        let template_k = match format {
            WeightFormat::Nvfp4 => 16,
            WeightFormat::Fp8Block128x128(_) => 128,
            _ => 32,
        };
        let base = (0..template_rows * template_k)
            .map(|index| sample(seed ^ index as u64) * 0.035)
            .collect::<Vec<_>>();
        let templates = [
            quantize_weight(&base, template_rows, template_k, format)?,
            quantize_weight(
                &base.iter().map(|x| -*x).collect::<Vec<_>>(),
                template_rows,
                template_k,
                format,
            )?,
        ];
        let packed = matches!(format, WeightFormat::Mxfp4 | WeightFormat::Nvfp4);
        let bytes_per_row = if packed { k / 2 } else { k };
        let template_bytes_per_row = if packed { template_k / 2 } else { template_k };
        let mut codes = vec![0_u8; experts * n * bytes_per_row];
        for expert in 0..TOP_K {
            let template = &templates[expert & 1];
            for row in 0..n {
                let source = &template.codes[(row % template_rows) * template_bytes_per_row
                    ..(row % template_rows + 1) * template_bytes_per_row];
                let destination = (expert * n + row) * bytes_per_row;
                for chunk in 0..k / template_k {
                    let at = destination + chunk * template_bytes_per_row;
                    codes[at..at + template_bytes_per_row].copy_from_slice(source);
                }
            }
        }
        let mut scales = vec![0_u8; 1];
        let mut f32_scales = vec![1.0_f32];
        if let Some(layout) = v.weight_scale_layout() {
            let mut source = vec![0_u8; layout.rows * layout.k_blocks];
            for expert in 0..TOP_K {
                let template_scales = match &templates[expert & 1].scales {
                    Scales::Ue8M0(v) | Scales::Ue4M3(v) => v,
                    Scales::F32(_) => return Err("expected MX or NV scale codes".into()),
                };
                for row in 0..n {
                    for block in 0..layout.k_blocks {
                        source[(expert * n + row) * layout.k_blocks + block] =
                            template_scales[row % template_rows];
                    }
                }
            }
            scales = if let Some(fragment) = v.weight_fragment_layout() {
                fragment.pack_scales(&source).map_err(err)?.bytes
            } else {
                layout.pack(&source).map_err(err)?.bytes
            };
            if format == WeightFormat::Nvfp4 {
                f32_scales = vec![templates[0].global_scale];
            }
        } else {
            let blocks_n = n / 128;
            let blocks_k = k / 128;
            f32_scales = vec![1.0; experts * blocks_n * blocks_k];
            for expert in 0..TOP_K {
                let value = match &templates[expert & 1].scales {
                    Scales::F32(v) => v[0],
                    _ => return Err("expected FP8 block scales".into()),
                };
                for bn in 0..blocks_n {
                    for bk in 0..blocks_k {
                        f32_scales[(expert * blocks_n + bn) * blocks_k + bk] = value;
                    }
                }
            }
        }
        let decoded = std::array::from_fn(|pattern| {
            let block = templates[pattern].decoded();
            let block = &block;
            (0..n)
                .flat_map(|row| {
                    (0..k).map(move |col| {
                        block[(row % template_rows) * template_k + col % template_k]
                    })
                })
                .collect::<Vec<_>>()
        });
        let active_codes = TOP_K * n * bytes_per_row;
        let active_scales = match format {
            WeightFormat::Fp8Block128x128(_) => TOP_K * (n / 128) * (k / 128) * 4,
            WeightFormat::Nvfp4 => TOP_K * n * (k / 16) + 4,
            _ => TOP_K * n * (k / 32),
        };
        let logical_bytes = active_codes + active_scales;
        let codes = if let Some(fragment) = v.weight_fragment_layout() {
            fragment.pack_codes(&codes).map_err(err)?.bytes
        } else {
            codes
        };
        Ok(Self {
            codes,
            scales,
            f32_scales,
            decoded,
            logical_bytes,
        })
    }

    fn upload(&self, session: &Session) -> LabResult<GpuWeights> {
        Ok(GpuWeights {
            codes: upload_bytes(session, &self.codes)?,
            scales: upload_bytes(session, &self.scales)?,
            f32_scales: session.upload_f32(&self.f32_scales)?,
        })
    }

    fn upload_with_code_capacity(
        &self,
        session: &Session,
        code_capacity: usize,
    ) -> LabResult<GpuWeights> {
        let codes = session.allocate(code_capacity)?;
        codes
            .copy_from_bytes(&self.codes, &session.stream)
            .map_err(err)?;
        Ok(GpuWeights {
            codes,
            scales: upload_bytes(session, &self.scales)?,
            f32_scales: session.upload_f32(&self.f32_scales)?,
        })
    }
}

struct GpuWeights {
    codes: DeviceBuffer,
    scales: DeviceBuffer,
    f32_scales: DeviceBuffer,
}

fn load_cubin(session: &Session, name: &str) -> LabResult<Module> {
    if let Some(module) = prefill_diagnostics::candidate_cubin(session, name)? {
        return Ok(module);
    }
    // A frozen artifact set lets one executable compare original and candidate
    // kernels within the same assigned GPU window.
    let root = std::env::var_os("CHOPINFER_MOE_CUBIN_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("OUT_DIR")));
    let path = root.join(format!("{name}.cubin"));
    let bytes = fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    session.context.load_module(&bytes).map_err(err)
}

struct MoeFunctions {
    compute: Function,
    reduce: Option<Function>,
}

fn load_moe(session: &Session, v: &Variant, counts: RuntimeCounts) -> LabResult<MoeFunctions> {
    let cubin = moe::cubin_specs(kernel_lab::catalogs(), Some(v.profile))
        .into_iter()
        .find(|spec| spec.entries.iter().any(|entry| entry.symbol == v.symbol))
        .ok_or_else(|| format!("missing cubin for {}", v.symbol))?;
    let module = load_cubin(session, &cubin.name)?;
    let compute = module.function(&v.symbol).map_err(err)?;
    let plan = v.plan(counts).map_err(err)?;
    if plan.dynamic_shared_bytes > 48 * 1024 {
        compute
            .set_max_dynamic_shared_memory(plan.dynamic_shared_bytes)
            .map_err(err)?;
    }
    let reduce = v
        .reduction_symbol()
        .map(|name| module.function(&name).map_err(err))
        .transpose()?;
    Ok(MoeFunctions { compute, reduce })
}

struct MoeOperands<'a> {
    activation: &'a DeviceBuffer,
    activation_scales: &'a DeviceBuffer,
    activation_f32_scales: &'a DeviceBuffer,
    weights: &'a GpuWeights,
    output: &'a DeviceBuffer,
    topk_weights: &'a DeviceBuffer,
    offsets: &'a DeviceBuffer,
    sorted: &'a DeviceBuffer,
    padded: &'a DeviceBuffer,
    live: &'a DeviceBuffer,
    active: &'a DeviceBuffer,
    quantized_output: &'a DeviceBuffer,
    quantized_scales: &'a DeviceBuffer,
    partials: &'a DeviceBuffer,
}

fn launch_moe(
    session: &Session,
    v: &Variant,
    functions: &MoeFunctions,
    counts: RuntimeCounts,
    operands: &MoeOperands<'_>,
    pdl: bool,
) -> LabResult<()> {
    launch_moe_grid(session, v, functions, counts, operands, pdl, None)
}

/// Lab-only equivalent of the serving overlap planner's persistent grid trim.
fn launch_moe_grid(
    session: &Session,
    v: &Variant,
    functions: &MoeFunctions,
    counts: RuntimeCounts,
    operands: &MoeOperands<'_>,
    pdl: bool,
    grid: Option<u32>,
) -> LabResult<()> {
    let mut buffers = BTreeMap::new();
    for (handle, buffer) in [
        operands.activation,
        operands.activation_scales,
        operands.activation_f32_scales,
        &operands.weights.scales,
        &operands.weights.f32_scales,
        operands.output,
        operands.topk_weights,
        operands.offsets,
        operands.sorted,
        operands.padded,
        operands.live,
        operands.active,
        operands.quantized_output,
        operands.quantized_scales,
        operands.partials,
        &operands.weights.codes,
    ]
    .into_iter()
    .enumerate()
    {
        buffers.insert(handle as u64 + 1, buffer);
    }
    let bindings = Bindings { buffers };
    let mut weight_map = v.weight_tensor_map(WEIGHT_HANDLE).map_err(err)?;
    if v.fragment_decode() {
        // Sparse graph fixtures store only their active low-numbered experts.
        // Fragment decode reads the code pointer directly; the unused TMA
        // descriptor describes exactly the allocation bound to that pointer.
        weight_map.dims[1] = (operands.weights.codes.byte_len() / (v.query.k as usize / 2)) as u64;
        weight_map.validate().map_err(err)?;
    }
    let args = v.arguments(HANDLES, weight_map).map_err(err)?;
    let mut plan = v.plan(counts).map_err(err)?;
    plan.pdl = pdl;
    if let Some(ctas) = grid {
        if !v.fragment_prefill() || !(1..=moe::SM_COUNT).contains(&ctas) {
            return Err("grid override requires a valid persistent prefill grid".into());
        }
        plan.grid = [ctas, 1, 1];
    }
    launch::launch(
        &session.context,
        &functions.compute,
        &session.stream,
        plan,
        &args,
        &bindings,
    )?;
    kernel_lab::launched(&v.symbol);
    if let Some(reduce) = &functions.reduce {
        let mut reduce_plan = v
            .reduction_plan(counts)
            .map_err(err)?
            .ok_or("missing reduction plan")?;
        reduce_plan.pdl = pdl;
        let reduce_args = v
            .reduction_arguments(HANDLES)
            .map_err(err)?
            .ok_or("missing reduction ABI")?;
        launch::launch(
            &session.context,
            reduce,
            &session.stream,
            reduce_plan,
            &reduce_args,
            &bindings,
        )?;
        if let Some(symbol) = v.reduction_symbol() {
            kernel_lab::launched(&symbol);
        }
    }
    Ok(())
}

fn row_entry(symbol: &str) -> LabResult<kernels_sm120::row::RowInstantiation> {
    kernels_sm120::row::catalog(kernel_lab::catalogs())
        .into_iter()
        .find(|item| item.symbol == symbol)
        .ok_or_else(|| format!("missing row entry {symbol}"))
}

fn launch_row(
    session: &Session,
    module: &Module,
    symbol: &str,
    tokens: u32,
    values: Vec<Argument>,
    buffers: &[(u64, &DeviceBuffer)],
    pdl: bool,
) -> LabResult<()> {
    let entry = row_entry(symbol)?;
    let function = module.function(symbol).map_err(err)?;
    let counts = RuntimeCounts {
        m: tokens,
        batch: 0,
        context: 0,
        routed_rows: tokens * 8,
        split_k: 1,
    };
    let mut plan = entry.plan(tokens, counts).map_err(err)?;
    plan.pdl = pdl;
    let args = entry.pack_arguments(values).map_err(err)?;
    let bindings = Bindings {
        buffers: buffers.iter().copied().collect(),
    };
    launch::launch(
        &session.context,
        &function,
        &session.stream,
        plan,
        &args,
        &bindings,
    )?;
    kernel_lab::launched(symbol);
    Ok(())
}

/// The expert phase of a step of `tokens` rows: decode buckets above 16
/// rows (`serving::DECODE_EXPERT_ROWS`) route as prefill does and run the
/// prefill entries, as served.
fn expert_phase(tokens: usize) -> Phase {
    if tokens > 16 {
        Phase::Prefill
    } else {
        Phase::Decode
    }
}

fn row_names(profile: Profile, phase: Phase) -> (&'static str, &'static str) {
    match (profile, phase) {
        (Profile::StdDenseMoe, Phase::Decode) => ("std_softmax128", "std_b16"),
        (Profile::StdDenseMoe, Phase::Prefill) => ("std_softmax128", "std"),
        // MiMo's router and permutation rows do not depend on the rank count.
        (p, Phase::Decode)
            if p.ranks_of(model_mimo26_contract::sm120::config::MODEL)
                .is_some() =>
        {
            ("mimo_sigmoid256", "mimo_b16")
        }
        (p, Phase::Prefill)
            if p.ranks_of(model_mimo26_contract::sm120::config::MODEL)
                .is_some() =>
        {
            ("mimo_sigmoid256", "mimo_b64")
        }
        _ => unreachable!("Qwen has a separate MoE catalog"),
    }
}

fn combine_entry(profile: Profile) -> LabResult<String> {
    let experts = if profile
        .ranks_of(model_mimo26_contract::sm120::config::MODEL)
        .is_some()
    {
        256
    } else {
        128
    };
    let mut entries = kernels_sm120::row::catalog(kernel_lab::catalogs())
        .into_iter()
        .filter(|item| {
            item.width == HIDDEN as u32
                && matches!(item.query, kernel_core::contract::RowQuery::ExpertCombine {
                experts: e, top_k: 8
            } if e == experts)
        })
        .collect::<Vec<_>>();
    entries.sort_by_key(|item| !item.body.contains("<4096,8,true>"));
    entries
        .into_iter()
        .next()
        .map(|item| item.symbol)
        .ok_or_else(|| "row engine needs a 4096-wide combine entry for MoE down".into())
}

fn bf16_f32(values: &[u16]) -> Vec<f32> {
    values.iter().map(|&v| bf16_to_f32(v)).collect()
}

fn bf16_operand(values: &mut [f64]) {
    for value in values {
        *value = f64::from(bf16_to_f32(f32_to_bf16(*value as f32)));
    }
}

/// The BF16 weights an A16 entry multiplies, from exact decoded weights:
/// element times block scale rounded to BF16 (moe.cuh `weight_value`). An
/// NVFP4 `global` scale multiplies the F32 sums afterwards, so it stays out
/// of the rounding; dividing it out of an exact decode is exact.
fn a16_weight_operand(values: &mut [f64], format: WeightFormat, global: f32) {
    let global = if format == WeightFormat::Nvfp4 {
        f64::from(global)
    } else {
        1.0
    };
    for value in values {
        *value = f64::from(bf16_to_f32(f32_to_bf16((*value / global) as f32))) * global;
    }
}

fn check_bf16(
    session: &Session,
    buffer: &DeviceBuffer,
    width: usize,
    selected_rows: &[usize],
    reference: &OracleOutput,
    case: &str,
    stage: &str,
) -> LabResult<Vec<u16>> {
    let all = session.download_bf16(buffer, buffer.byte_len() / 2)?;
    let actual = selected_rows
        .iter()
        .flat_map(|&row| bf16_f32(&all[row * width..(row + 1) * width]))
        .collect::<Vec<_>>();
    let mut receipt = BTreeMap::new();
    receipt.insert("case".into(), json!(case));
    receipt.insert("stage".into(), json!(stage));
    receipt.insert("checked_rows".into(), json!(selected_rows));
    // The checked rows' bits, so receipts of two builds compare bitwise.
    let mut digest = Sha256::new();
    for &row in selected_rows {
        for value in &all[row * width..(row + 1) * width] {
            digest.update(value.to_le_bytes());
        }
    }
    receipt.insert(
        "checked_rows_sha256".into(),
        json!(format!("{:x}", digest.finalize())),
    );
    let gate = check_output(&actual, reference, DType::Bf16, &mut receipt);
    let worst = receipt.get("worst_element").cloned();
    session.receipt("moe", receipt)?;
    gate.map_err(|error| format!("{case} {stage}: {error}; worst element {worst:?}"))?;
    Ok(all)
}

struct RouterGemm {
    function: Function,
    fused_route: Option<Function>,
    plan: LaunchPlan,
    args: kernel_core::contract::ArgumentList,
    weights: DeviceBuffer,
    partials: DeviceBuffer,
    semaphores: DeviceBuffer,
}

impl RouterGemm {
    fn new(session: &Session, tokens: usize) -> LabResult<Self> {
        let symbol = format!("chop_gemm_mimo_router_m{tokens}_v1");
        let variant = gemm::catalog(kernel_lab::catalogs())
            .into_iter()
            .find(|item| item.symbol == symbol)
            .ok_or_else(|| format!("missing dense router {symbol}"))?;
        let q = match variant.key.query {
            kernel_core::contract::StaticQuery::Gemm(q) => q,
            _ => return Err("router catalog entry is not GEMM".into()),
        };
        let counts = RuntimeCounts {
            m: tokens as u32,
            batch: tokens as u32,
            context: 0,
            routed_rows: 0,
            split_k: gemm::choose_split_k(&variant.key).map_err(err)?,
        };
        let plan = gemm::plan(&variant.key, counts).map_err(err)?;
        let weights = upload_bytes(session, &vec![0; (q.n * q.k * 2) as usize])?;
        let partials = session.allocate(plan.workspace_bytes.max(1) as usize)?;
        let semaphores = upload_bytes(session, &vec![0; plan.workspace_bytes.max(1) as usize])?;
        let args = gemm::arguments(
            &variant.key,
            counts,
            gemm::tensor_maps(&variant.key, 1, 2).map_err(err)?,
            GemmBuffers {
                output: 3,
                partial: 4,
                semaphores: 5,
                weight_scales: None,
                activation_scales: None,
                residual: None,
                weight_global_scale: None,
                activation_global_scale: None,
            },
        )
        .map_err(err)?;
        let module = session.module("gemm_mimo_sm120")?;
        let function = module.function(&symbol).map_err(err)?;
        let fused_route = if tokens <= 32 {
            Some(
                load_cubin(
                    session,
                    model_mimo26_contract::sm120::moe::FUSED_ROUTER_PREFIX,
                )?
                .function(&format!(
                    "{}_t{tokens}_v1",
                    model_mimo26_contract::sm120::moe::FUSED_ROUTER_PREFIX
                ))
                .map_err(err)?,
            )
        } else {
            None
        };
        if plan.dynamic_shared_bytes > 48 * 1024 {
            function
                .set_max_dynamic_shared_memory(plan.dynamic_shared_bytes)
                .map_err(err)?;
        }
        Ok(Self {
            function,
            fused_route,
            plan,
            args,
            weights,
            partials,
            semaphores,
        })
    }

    fn launch(
        &self,
        session: &Session,
        input: &DeviceBuffer,
        logits: &DeviceBuffer,
        pdl: bool,
    ) -> LabResult<()> {
        let buffers = BTreeMap::from([
            (1, &self.weights),
            (2, input),
            (3, logits),
            (4, &self.partials),
            (5, &self.semaphores),
        ]);
        let mut plan = self.plan;
        plan.pdl = pdl;
        launch::launch(
            &session.context,
            &self.function,
            &session.stream,
            plan,
            &self.args,
            &Bindings { buffers },
        )
        .map_err(err)
    }
}

struct FullCase {
    profile: Profile,
    phase: Phase,
    tokens: usize,
    /// Allocation and row-launch bucket, independent of the device live count.
    prepared_tokens: usize,
    experts: usize,
    gate: Variant,
    down: Variant,
    gate_functions: MoeFunctions,
    down_functions: MoeFunctions,
    row_module: Module,
    route_module: Option<Module>,
    /// The decode route entry of `route_module`.
    route_symbol: String,
    fused_down_module: Option<Module>,
    router_gemm: Option<RouterGemm>,
    combine_symbol: String,
    gate_weights: GpuWeights,
    down_weights: GpuWeights,
    gate_decoded: [Vec<f64>; 2],
    down_decoded: [Vec<f64>; 2],
    weight_bytes: usize,
    input_host: Vec<u16>,
    logits_host: Vec<f32>,
    bias_host: Vec<f32>,
    input: DeviceBuffer,
    logits: DeviceBuffer,
    bias: DeviceBuffer,
    ids: DeviceBuffer,
    topk: DeviceBuffer,
    counts: DeviceBuffer,
    offsets: DeviceBuffer,
    sorted: DeviceBuffer,
    padded: DeviceBuffer,
    live: DeviceBuffer,
    active: DeviceBuffer,
    gate_output: DeviceBuffer,
    gate_partials: DeviceBuffer,
    down_output: DeviceBuffer,
    down_partials: DeviceBuffer,
    final_output: DeviceBuffer,
    combine_weights: DeviceBuffer,
    dummy: DeviceBuffer,
    one: DeviceBuffer,
    /// Per 256-route tile expert counts and offsets for the tiled route.
    tile_counts: DeviceBuffer,
    tile_offsets: DeviceBuffer,
}

impl FullCase {
    fn use_fused_router(&self) -> bool {
        let available = self
            .router_gemm
            .as_ref()
            .is_some_and(|gemm| gemm.fused_route.is_some());
        available
            && match std::env::var("CHOPINFER_MOE_FUSED_ROUTER").as_deref() {
                Ok("0") => false,
                Ok("1") => true,
                _ => model_mimo26_contract::sm120::moe::prefer_fused_router(self.tokens as u32),
            }
    }

    fn launch_fused_router_route(
        &self,
        session: &Session,
        input: &DeviceBuffer,
        pdl: bool,
    ) -> LabResult<()> {
        let gemm = self.router_gemm.as_ref().ok_or("router GEMV is absent")?;
        let function = gemm
            .fused_route
            .as_ref()
            .ok_or("fused router entry is absent")?;
        let mut plan = model_mimo26_contract::sm120::moe::fused_router_plan(
            self.tokens as u32,
            self.tokens as u32,
        )
        .map_err(err)?;
        plan.pdl = pdl;
        function
            .set_max_dynamic_shared_memory(plan.dynamic_shared_bytes)
            .map_err(err)?;
        let buffers = [
            &gemm.weights,
            input,
            &self.logits,
            &gemm.partials,
            &gemm.semaphores,
            &self.live,
            &self.bias,
            &self.ids,
            &self.topk,
            &self.counts,
            &self.offsets,
            &self.sorted,
            &self.padded,
            &self.active,
        ];
        let bindings = Bindings {
            buffers: buffers
                .iter()
                .enumerate()
                .map(|(i, buffer)| (i as u64 + 1, *buffer))
                .collect(),
        };
        let args =
            moe::fused_router_arguments([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], 1.0)
                .map_err(err)?;
        launch::launch(
            &session.context,
            function,
            &session.stream,
            plan,
            &args,
            &bindings,
        )
    }

    fn new(session: &Session, profile: Profile, phase: Phase, tokens: usize) -> LabResult<Self> {
        Self::build(session, profile, phase, tokens, None, false)
    }

    fn sparse_graph_layer(session: &Session, tokens: usize, layer: usize) -> LabResult<Self> {
        Self::build(
            session,
            model_mimo26_contract::sm120::config::PROFILE,
            expert_phase(tokens),
            tokens,
            Some(layer),
            false,
        )
    }

    fn random_route_graph_layer(
        session: &Session,
        tokens: usize,
        layer: usize,
        distinct_experts: usize,
    ) -> LabResult<Self> {
        let case = Self::build(
            session,
            model_mimo26_contract::sm120::config::PROFILE,
            expert_phase(tokens),
            tokens,
            Some(layer),
            true,
        )?;
        case.install_random_routes(session, layer, distinct_experts)?;
        Ok(case)
    }

    /// A broad-route layer whose decode route reproduces captured top-8
    /// expert IDs (token-major, eight per token): chosen experts get logit
    /// +8 and the rest -8 with a zero bias, so the route kernel picks exactly
    /// them with equal weights. The router GEMM is left out.
    fn real_route_graph_layer(
        session: &Session,
        tokens: usize,
        layer: usize,
        ids: &[u8],
    ) -> LabResult<Self> {
        let mut case = Self::build(
            session,
            model_mimo26_contract::sm120::config::PROFILE,
            expert_phase(tokens),
            tokens,
            Some(layer),
            true,
        )?;
        case.router_gemm = None;
        case.bias_host = vec![0.0; case.experts];
        case.bias
            .copy_from_bytes(&f32_bytes(&case.bias_host), &session.stream)
            .map_err(err)?;
        case.install_real_routes(session, ids)?;
        Ok(case)
    }

    fn install_real_routes(&mut self, session: &Session, ids: &[u8]) -> LabResult<()> {
        if ids.len() != self.tokens * TOP_K {
            return Err("captured route count does not match the token count".into());
        }
        let mut logits = vec![-8.0_f32; self.tokens * self.experts];
        for (row, &expert) in ids.iter().enumerate() {
            logits[row / TOP_K * self.experts + usize::from(expert)] = 8.0;
        }
        self.logits
            .copy_from_bytes(&f32_bytes(&logits), &session.stream)
            .map_err(err)?;
        self.logits_host = logits;
        Ok(())
    }

    fn install_random_routes(
        &self,
        session: &Session,
        layer: usize,
        distinct_experts: usize,
    ) -> LabResult<()> {
        let pool = (0..distinct_experts)
            .map(|index| ((index * 73 + layer * 19) % self.experts) as i32)
            .collect::<Vec<_>>();
        let ids = (0..self.tokens * TOP_K)
            .map(|row| pool[row % distinct_experts])
            .collect::<Vec<_>>();
        let permutation = oracle::expert_permute(
            &ids.iter().map(|&id| id as usize).collect::<Vec<_>>(),
            self.experts,
            self.gate.knobs.tile_m as usize,
        )
        .map_err(err)?;
        let mut counts = vec![0_i32; self.experts];
        for &id in &ids {
            counts[id as usize] += 1;
        }
        let offsets = permutation
            .expert_offsets
            .iter()
            .map(|&offset| offset as i32)
            .collect::<Vec<_>>();
        let sorted = permutation
            .forward
            .iter()
            .map(|row| row.map_or(-1, |row| row as i32))
            .collect::<Vec<_>>();
        let active_ids = counts
            .iter()
            .enumerate()
            .filter(|(_, count)| **count > 0)
            .map(|(id, _)| id as i32)
            .collect::<Vec<_>>();
        if active_ids.len() != distinct_experts {
            return Err("random route construction lost an expert".into());
        }
        let active = std::iter::once(distinct_experts as i32)
            .chain(active_ids)
            .collect::<Vec<_>>();
        let weights = (0..ids.len())
            .flat_map(|_| (1.0_f32 / TOP_K as f32).to_ne_bytes())
            .collect::<Vec<_>>();
        for (buffer, bytes) in [
            (&self.ids, i32_bytes(&ids)),
            (&self.topk, weights),
            (&self.counts, i32_bytes(&counts)),
            (&self.offsets, i32_bytes(&offsets)),
            (&self.sorted, i32_bytes(&sorted)),
            (&self.padded, i32_bytes(&[sorted.len() as i32])),
            (&self.active, i32_bytes(&active)),
        ] {
            buffer
                .copy_from_bytes(&bytes, &session.stream)
                .map_err(err)?;
        }
        Ok(())
    }

    fn build(
        session: &Session,
        profile: Profile,
        phase: Phase,
        tokens: usize,
        graph_layer: Option<usize>,
        random_routing: bool,
    ) -> LabResult<Self> {
        let format = if profile
            .ranks_of(model_mimo26_contract::sm120::config::MODEL)
            .is_some()
        {
            WeightFormat::Mxfp4
        } else {
            WeightFormat::Fp8Block128x128(ScaleMode::ExactF32)
        };
        let gate = variant(
            profile,
            phase,
            Projection::GateUp,
            InputPath::A16Gather,
            format,
            tokens,
        )?;
        let down = variant(
            profile,
            phase,
            Projection::Down,
            InputPath::A16Gather,
            format,
            tokens,
        )?;
        assert!(down.query.epilogues.weighted_moe_combine);
        let experts = gate.query.grouped.ok_or("missing expert count")?.experts as usize;
        let intermediate = gate.query.n as usize / 2;
        let gate_bank = if let Some(layer) = graph_layer {
            WeightBank::sparse_fragment_bench(
                &gate,
                layer,
                if random_routing { experts } else { TOP_K },
            )?
        } else {
            WeightBank::new(&gate, 0x3450)?
        };
        let down_bank = if let Some(layer) = graph_layer {
            WeightBank::sparse_fragment_bench(
                &down,
                layer + 47,
                if random_routing { experts } else { TOP_K },
            )?
        } else {
            WeightBank::new(&down, 0x7650)?
        };
        let weight_bytes = gate_bank.logical_bytes + down_bank.logical_bytes;
        let gate_weights = if graph_layer.is_some() && random_routing {
            gate_bank.upload_with_code_capacity(
                session,
                experts * gate.query.n as usize * gate.query.k as usize / 2,
            )?
        } else {
            gate_bank.upload(session)?
        };
        let down_weights = if graph_layer.is_some() && random_routing {
            down_bank.upload_with_code_capacity(
                session,
                experts * down.query.n as usize * down.query.k as usize / 2,
            )?
        } else {
            down_bank.upload(session)?
        };
        let mut gate_decoded = gate_bank.decoded;
        let mut down_decoded = down_bank.decoded;
        for pattern in 0..2 {
            bf16_operand(&mut gate_decoded[pattern]);
            bf16_operand(&mut down_decoded[pattern]);
        }
        let input_host = (0..tokens * HIDDEN)
            .map(|index| f32_to_bf16(sample(0x9200 ^ index as u64) * 0.12))
            .collect::<Vec<_>>();
        let decode_router_gemm = profile
            .ranks_of(model_mimo26_contract::sm120::config::MODEL)
            .is_some()
            && phase == Phase::Decode;
        let logits_host = (0..tokens * experts)
            .map(|index| {
                if decode_router_gemm {
                    return 0.0;
                }
                let expert = index % experts;
                if expert < TOP_K {
                    1.5 + sample(0xa300 ^ index as u64) * 0.3
                } else {
                    -7.0 + sample(0xb400 ^ index as u64) * 0.1
                }
            })
            .collect::<Vec<_>>();
        let bias_host = (0..experts)
            .map(|expert| {
                if decode_router_gemm {
                    if expert < TOP_K {
                        1.5 + sample(0xc500 ^ expert as u64) * 0.01
                    } else {
                        -7.0
                    }
                } else {
                    sample(0xc500 ^ expert as u64) * 0.01
                }
            })
            .collect::<Vec<_>>();
        let input = session.upload_bf16(&input_host)?;
        let logits = session.upload_f32(&logits_host)?;
        let bias = session.upload_f32(&bias_host)?;
        let router_gemm = decode_router_gemm
            .then(|| RouterGemm::new(session, tokens))
            .transpose()?;
        let routes = tokens * TOP_K;
        let ids = session.allocate(routes * 4)?;
        let topk = session.allocate(routes * 4)?;
        let counts = session.allocate(experts * 4)?;
        let offsets = session.allocate((experts + 1) * 4)?;
        let padded_capacity = match gate.query.grouped.ok_or("missing row capacity")?.capacity {
            kernel_core::contract::RoutedCapacity::TotalRows(rows) => rows as usize,
            _ => return Err("MoE test needs total routed-row capacity".into()),
        };
        let sorted = session.allocate(padded_capacity * 4)?;
        let padded = session.allocate(4)?;
        let live = upload_bytes(session, &(tokens as i32).to_ne_bytes())?;
        let active = upload_bytes(
            session,
            &i32_bytes(
                &(0..=experts)
                    .map(|i| if i == 0 { TOP_K as i32 } else { (i - 1) as i32 })
                    .collect::<Vec<_>>(),
            ),
        )?;
        let gate_output = session.allocate(routes * intermediate * 2)?;
        let down_output = session.allocate(routes * HIDDEN * 2)?;
        let gate_counts = RuntimeCounts {
            m: tokens as u32,
            batch: 0,
            context: 0,
            routed_rows: routes as u32,
            split_k: gate.knobs.split_k_capacity,
        };
        let down_counts = RuntimeCounts {
            split_k: down.knobs.split_k_capacity,
            ..gate_counts
        };
        let gate_partials = session
            .allocate(gate.plan(gate_counts).map_err(err)?.workspace_bytes.max(1) as usize)?;
        let down_partials = session
            .allocate(down.plan(down_counts).map_err(err)?.workspace_bytes.max(1) as usize)?;
        let gate_functions = load_moe(session, &gate, gate_counts)?;
        let down_functions = load_moe(session, &down, down_counts)?;
        let row_module = load_cubin(session, "row_engine")?;
        let route_module = decode_router_gemm
            .then(|| load_cubin(session, "chop_moe_mimo_decode_route_b16"))
            .transpose()?;
        let fused_down_module = decode_router_gemm
            .then(|| {
                load_cubin(
                    session,
                    model_mimo26_contract::sm120::moe::DOWN_COMBINE_PREFIX,
                )
            })
            .transpose()?;
        let combine_symbol = combine_entry(profile)?;
        Ok(Self {
            profile,
            phase,
            tokens,
            prepared_tokens: tokens,
            experts,
            gate,
            down,
            gate_functions,
            down_functions,
            row_module,
            route_module,
            route_symbol: model_mimo26_contract::sm120::moe::DECODE_ROUTE_SYMBOL.into(),
            fused_down_module,
            router_gemm,
            combine_symbol,
            gate_weights,
            down_weights,
            gate_decoded,
            down_decoded,
            weight_bytes,
            input_host,
            logits_host,
            bias_host,
            input,
            logits,
            bias,
            ids,
            topk,
            counts,
            offsets,
            sorted,
            padded,
            live,
            active,
            gate_output,
            gate_partials,
            down_output,
            down_partials,
            final_output: session.allocate(tokens * HIDDEN * 2)?,
            combine_weights: session.upload_f32(&vec![1.0; routes])?,
            dummy: upload_bytes(session, &[0])?,
            one: session.upload_f32(&[1.0])?,
            tile_counts: session.allocate(routes.div_ceil(256) * 256 * 4)?,
            tile_offsets: session.allocate(routes.div_ceil(256) * 256 * 4)?,
        })
    }

    fn counts(&self, split_k: u8) -> RuntimeCounts {
        RuntimeCounts {
            m: self.tokens as u32,
            batch: 0,
            context: 0,
            routed_rows: (self.tokens * TOP_K) as u32,
            split_k,
        }
    }

    fn launch_projection(&self, session: &Session, projection: Projection) -> LabResult<()> {
        self.launch_projection_grid(session, projection, None)
    }

    fn launch_projection_grid(
        &self,
        session: &Session,
        projection: Projection,
        grid: Option<u32>,
    ) -> LabResult<()> {
        let gate = projection == Projection::GateUp;
        let variant = if gate { &self.gate } else { &self.down };
        let functions = if gate {
            &self.gate_functions
        } else {
            &self.down_functions
        };
        let operands = MoeOperands {
            activation: if gate { &self.input } else { &self.gate_output },
            activation_scales: &self.dummy,
            activation_f32_scales: &self.one,
            weights: if gate {
                &self.gate_weights
            } else {
                &self.down_weights
            },
            output: if gate {
                &self.gate_output
            } else {
                &self.down_output
            },
            topk_weights: &self.topk,
            offsets: &self.offsets,
            sorted: &self.sorted,
            padded: &self.padded,
            live: &self.live,
            active: &self.active,
            quantized_output: &self.dummy,
            quantized_scales: &self.dummy,
            partials: if gate {
                &self.gate_partials
            } else {
                &self.down_partials
            },
        };
        launch_moe_grid(
            session,
            variant,
            functions,
            self.counts(variant.knobs.split_k_capacity),
            &operands,
            true,
            grid,
        )
    }

    fn launch_decode_route(&self, session: &Session, pdl: bool) -> LabResult<()> {
        let module = self
            .route_module
            .as_ref()
            .ok_or("decode route module is absent")?;
        let function = module.function(&self.route_symbol).map_err(err)?;
        let mut plan =
            moe::decode_route_plan(self.tokens as u32, self.tokens as u32).map_err(err)?;
        plan.pdl = pdl;
        let buffers = [
            &self.logits,
            &self.bias,
            &self.ids,
            &self.topk,
            &self.counts,
            &self.offsets,
            &self.sorted,
            &self.padded,
            &self.live,
            &self.active,
        ];
        let bindings = Bindings {
            buffers: buffers
                .iter()
                .enumerate()
                .map(|(i, buffer)| (i as u64 + 1, *buffer))
                .collect(),
        };
        let args =
            moe::decode_route_arguments([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 1.0).map_err(err)?;
        launch::launch(
            &session.context,
            &function,
            &session.stream,
            plan,
            &args,
            &bindings,
        )
    }

    fn launch_fused_down_combine(&self, session: &Session, pdl: bool) -> LabResult<()> {
        let module = self
            .fused_down_module
            .as_ref()
            .ok_or("fused down combine module is absent")?;
        let function = module
            .function(&format!(
                "{}_t{}_v1",
                model_mimo26_contract::sm120::moe::DOWN_COMBINE_PREFIX,
                self.tokens
            ))
            .map_err(err)?;
        let mut plan =
            moe::fused_down_combine_plan(self.tokens as u32, self.tokens as u32).map_err(err)?;
        plan.pdl = pdl;
        let buffers = [
            &self.gate_output,
            &self.down_weights.codes,
            &self.down_weights.scales,
            &self.ids,
            &self.topk,
            &self.final_output,
            &self.live,
        ];
        let bindings = Bindings {
            buffers: buffers
                .iter()
                .enumerate()
                .map(|(i, buffer)| (i as u64 + 1, *buffer))
                .collect(),
        };
        let args = moe::fused_down_combine_arguments([1, 2, 3, 4, 5, 6, 7]).map_err(err)?;
        launch::launch(
            &session.context,
            &function,
            &session.stream,
            plan,
            &args,
            &bindings,
        )
    }

    fn launch_row_stage(&self, session: &Session, stage: &str) -> LabResult<()> {
        if stage == "router_gemm" {
            return self
                .router_gemm
                .as_ref()
                .ok_or("router GEMM is absent")?
                .launch(session, &self.input, &self.logits, true);
        }
        if stage == "router_route" {
            return self.launch_fused_router_route(session, &self.input, true);
        }
        if stage == "route" {
            // Prefill-phase cases route with the router GEMM, the router row
            // and the permutation.
            return if self.phase == Phase::Prefill {
                self.launch_route_stages(session, &self.input, true)
            } else {
                self.launch_decode_route(session, true)
            };
        }
        let capacity = self.tokens as u32;
        let (router, permute) = row_names(self.profile, self.phase);
        let (symbol, args, buffers) = match stage {
            "router" => (
                format!("chop_row_router_{router}_v1"),
                vec![
                    Argument::DeviceBuffer(1),
                    Argument::DeviceBuffer(2),
                    Argument::DeviceBuffer(3),
                    Argument::DeviceBuffer(4),
                    Argument::DeviceBuffer(5),
                    Argument::U32(1.0_f32.to_bits()),
                ],
                vec![
                    (1, &self.logits),
                    (2, &self.bias),
                    (3, &self.ids),
                    (4, &self.topk),
                    (5, &self.live),
                ],
            ),
            "count" => (
                format!("chop_row_permute_count_{permute}_v1"),
                vec![
                    Argument::DeviceBuffer(1),
                    Argument::DeviceBuffer(2),
                    Argument::DeviceBuffer(3),
                ],
                vec![(1, &self.ids), (2, &self.counts), (3, &self.live)],
            ),
            "scan" => (
                format!("chop_row_permute_scan_{permute}_v1"),
                vec![
                    Argument::DeviceBuffer(1),
                    Argument::DeviceBuffer(2),
                    Argument::DeviceBuffer(3),
                ],
                vec![(1, &self.counts), (2, &self.offsets), (3, &self.padded)],
            ),
            "scatter" => (
                format!("chop_row_permute_scatter_{permute}_v1"),
                vec![
                    Argument::DeviceBuffer(1),
                    Argument::DeviceBuffer(2),
                    Argument::DeviceBuffer(3),
                    Argument::DeviceBuffer(4),
                ],
                vec![
                    (1, &self.ids),
                    (2, &self.offsets),
                    (3, &self.sorted),
                    (4, &self.live),
                ],
            ),
            "combine" => (
                self.combine_symbol.clone(),
                vec![
                    Argument::DeviceBuffer(1),
                    Argument::DeviceBuffer(2),
                    Argument::DeviceBuffer(3),
                    Argument::DeviceBuffer(4),
                ],
                vec![
                    (1, &self.down_output),
                    (2, &self.combine_weights),
                    (3, &self.final_output),
                    (4, &self.live),
                ],
            ),
            _ => return Err(format!("unknown MoE row stage {stage}")),
        };
        launch_row(
            session,
            &self.row_module,
            &symbol,
            capacity,
            args,
            &buffers,
            true,
        )
    }

    fn launch(&self, session: &Session, pdl: bool) -> LabResult<()> {
        self.launch_with_input(session, &self.input, pdl)
    }

    fn launch_with_input(
        &self,
        session: &Session,
        input: &DeviceBuffer,
        pdl: bool,
    ) -> LabResult<()> {
        self.launch_with_input_variant(session, input, pdl, false)
    }

    /// Prefill routes use the row engine's 256-route tile kernels when
    /// `CHOPINFER_MOE_TILED_ROUTE=1`; otherwise the per-expert kernels the
    /// bundle selects.
    fn tiled_route(&self) -> bool {
        self.phase == Phase::Prefill
            && self
                .profile
                .ranks_of(model_mimo26_contract::sm120::config::MODEL)
                .is_some()
            && std::env::var("CHOPINFER_MOE_TILED_ROUTE").as_deref() == Ok("1")
    }

    fn launch_tiled_route(&self, session: &Session, capacity: u32, pdl: bool) -> LabResult<()> {
        let stages: [(&str, &[(u64, &DeviceBuffer)]); 4] = [
            (
                "chop_row_permute_tile_count_mimo_b64_v1",
                &[(1, &self.ids), (2, &self.tile_counts), (3, &self.live)],
            ),
            (
                "chop_row_permute_tile_prefix_mimo_b64_v1",
                &[
                    (1, &self.tile_counts),
                    (2, &self.counts),
                    (3, &self.tile_offsets),
                    (4, &self.live),
                ],
            ),
            (
                "chop_row_permute_tile_scan_mimo_b64_v1",
                &[
                    (1, &self.counts),
                    (2, &self.offsets),
                    (3, &self.sorted),
                    (4, &self.padded),
                ],
            ),
            (
                "chop_row_permute_tile_scatter_mimo_b64_v1",
                &[
                    (1, &self.ids),
                    (2, &self.tile_offsets),
                    (3, &self.offsets),
                    (4, &self.sorted),
                    (5, &self.live),
                ],
            ),
        ];
        for (symbol, buffers) in stages {
            let values = (1..=buffers.len() as u64)
                .map(Argument::DeviceBuffer)
                .collect();
            launch_row(
                session,
                &self.row_module,
                symbol,
                capacity,
                values,
                buffers,
                pdl,
            )?;
        }
        Ok(())
    }

    /// Router and permute stages only (the expert GEMMs' route metadata).
    fn launch_route_stages(
        &self,
        session: &Session,
        input: &DeviceBuffer,
        pdl: bool,
    ) -> LabResult<()> {
        self.launch_with_input_stages(session, input, pdl, false, 1)
    }

    /// The fixed-order combine only.
    fn launch_combine_stage(&self, session: &Session, pdl: bool) -> LabResult<()> {
        self.launch_with_input_stages(session, &self.input, pdl, false, 2)
    }

    fn launch_with_input_variant(
        &self,
        session: &Session,
        input: &DeviceBuffer,
        pdl: bool,
        fused_down_combine: bool,
    ) -> LabResult<()> {
        self.launch_with_input_stages(session, input, pdl, fused_down_combine, 0)
    }

    /// `only`: 0 runs every stage, 1 stops after routing, 2 runs the combine.
    fn launch_with_input_stages(
        &self,
        session: &Session,
        input: &DeviceBuffer,
        pdl: bool,
        fused_down_combine: bool,
        only: u8,
    ) -> LabResult<()> {
        self.launch_with_input_stages_grid(session, input, pdl, fused_down_combine, only, None)
    }

    fn launch_with_input_stages_grid(
        &self,
        session: &Session,
        input: &DeviceBuffer,
        pdl: bool,
        fused_down_combine: bool,
        only: u8,
        grid: Option<u32>,
    ) -> LabResult<()> {
        let capacity = self.prepared_tokens as u32;
        if only == 2 {
            return self.launch_combine_row(session, capacity, pdl);
        }
        let (router, permute) = row_names(self.profile, self.phase);
        if self.use_fused_router() {
            self.launch_fused_router_route(session, input, pdl)?;
        } else if let Some(gemm) = &self.router_gemm {
            gemm.launch(session, input, &self.logits, pdl)?;
        }
        if self.route_module.is_some() {
            if !self.use_fused_router() {
                self.launch_decode_route(session, pdl)?;
            }
        } else {
            launch_row(
                session,
                &self.row_module,
                &format!("chop_row_router_{router}_v1"),
                capacity,
                vec![
                    Argument::DeviceBuffer(1),
                    Argument::DeviceBuffer(2),
                    Argument::DeviceBuffer(3),
                    Argument::DeviceBuffer(4),
                    Argument::DeviceBuffer(5),
                    Argument::U32(1.0_f32.to_bits()),
                ],
                &[
                    (1, &self.logits),
                    (2, &self.bias),
                    (3, &self.ids),
                    (4, &self.topk),
                    (5, &self.live),
                ],
                pdl,
            )?;
            if self.tiled_route() {
                self.launch_tiled_route(session, capacity, pdl)?;
            } else {
                launch_row(
                    session,
                    &self.row_module,
                    &format!("chop_row_permute_count_{permute}_v1"),
                    capacity,
                    vec![
                        Argument::DeviceBuffer(1),
                        Argument::DeviceBuffer(2),
                        Argument::DeviceBuffer(3),
                    ],
                    &[(1, &self.ids), (2, &self.counts), (3, &self.live)],
                    pdl,
                )?;
                launch_row(
                    session,
                    &self.row_module,
                    &format!("chop_row_permute_scan_{permute}_v1"),
                    capacity,
                    vec![
                        Argument::DeviceBuffer(1),
                        Argument::DeviceBuffer(2),
                        Argument::DeviceBuffer(3),
                    ],
                    &[(1, &self.counts), (2, &self.offsets), (3, &self.padded)],
                    pdl,
                )?;
                launch_row(
                    session,
                    &self.row_module,
                    &format!("chop_row_permute_scatter_{permute}_v1"),
                    capacity,
                    vec![
                        Argument::DeviceBuffer(1),
                        Argument::DeviceBuffer(2),
                        Argument::DeviceBuffer(3),
                        Argument::DeviceBuffer(4),
                    ],
                    &[
                        (1, &self.ids),
                        (2, &self.offsets),
                        (3, &self.sorted),
                        (4, &self.live),
                    ],
                    pdl,
                )?;
            }
        }
        if only == 1 {
            return Ok(());
        }
        let gate_operands = MoeOperands {
            activation: input,
            activation_scales: &self.dummy,
            activation_f32_scales: &self.one,
            weights: &self.gate_weights,
            output: &self.gate_output,
            topk_weights: &self.topk,
            offsets: &self.offsets,
            sorted: &self.sorted,
            padded: &self.padded,
            live: &self.live,
            active: &self.active,
            quantized_output: &self.dummy,
            quantized_scales: &self.dummy,
            partials: &self.gate_partials,
        };
        launch_moe_grid(
            session,
            &self.gate,
            &self.gate_functions,
            self.counts(self.gate.knobs.split_k_capacity),
            &gate_operands,
            pdl,
            grid,
        )?;
        if fused_down_combine {
            return self.launch_fused_down_combine(session, pdl);
        }
        let down_operands = MoeOperands {
            activation: &self.gate_output,
            activation_scales: &self.dummy,
            activation_f32_scales: &self.one,
            weights: &self.down_weights,
            output: &self.down_output,
            topk_weights: &self.topk,
            offsets: &self.offsets,
            sorted: &self.sorted,
            padded: &self.padded,
            live: &self.live,
            active: &self.active,
            quantized_output: &self.dummy,
            quantized_scales: &self.dummy,
            partials: &self.down_partials,
        };
        launch_moe_grid(
            session,
            &self.down,
            &self.down_functions,
            self.counts(self.down.knobs.split_k_capacity),
            &down_operands,
            pdl,
            grid,
        )?;
        self.launch_combine_row(session, capacity, pdl)
    }

    fn launch_combine_row(&self, session: &Session, capacity: u32, pdl: bool) -> LabResult<()> {
        launch_row(
            session,
            &self.row_module,
            &self.combine_symbol,
            capacity,
            vec![
                Argument::DeviceBuffer(1),
                Argument::DeviceBuffer(2),
                Argument::DeviceBuffer(3),
                Argument::DeviceBuffer(4),
            ],
            &[
                (1, &self.down_output),
                (
                    2,
                    if self
                        .profile
                        .ranks_of(model_mimo26_contract::sm120::config::MODEL)
                        .is_some()
                    {
                        &self.topk
                    } else {
                        &self.combine_weights
                    },
                ),
                (3, &self.final_output),
                (4, &self.live),
            ],
            pdl,
        )
    }

    fn check(&self, session: &Session) -> LabResult<()> {
        let case = format!(
            "{}_{}_t{}",
            self.profile.as_str(),
            if self.phase == Phase::Decode {
                "decode"
            } else {
                "prefill"
            },
            self.tokens
        );
        let ids = download_i32(session, &self.ids, self.tokens * TOP_K)?;
        let topk = download_f32(session, &self.topk, self.tokens * TOP_K)?;
        let kind = if self
            .profile
            .ranks_of(model_mimo26_contract::sm120::config::MODEL)
            .is_some()
        {
            RouterKind::SigmoidBiasSelectNormalized {
                scaling_factor: 1.0,
            }
        } else {
            RouterKind::SoftmaxTopK { renormalize: true }
        };
        for token in 0..self.tokens {
            let scores = self.logits_host[token * self.experts..(token + 1) * self.experts]
                .iter()
                .map(|&v| f64::from(v))
                .collect::<Vec<_>>();
            let bias = if self
                .profile
                .ranks_of(model_mimo26_contract::sm120::config::MODEL)
                .is_some()
            {
                Some(
                    self.bias_host
                        .iter()
                        .map(|&v| f64::from(v))
                        .collect::<Vec<_>>(),
                )
            } else {
                None
            };
            let reference = oracle::router(&scores, bias.as_deref(), TOP_K, kind).map_err(err)?;
            for slot in 0..TOP_K {
                let route = token * TOP_K + slot;
                if ids[route] != reference.expert_ids[slot] as i32 {
                    return Err(format!(
                        "{case} router route {route}: id {} != {}",
                        ids[route], reference.expert_ids[slot]
                    ));
                }
                if (f64::from(topk[route]) - reference.weights[slot]).abs() > 2e-6 {
                    return Err(format!(
                        "{case} router route {route}: weight {} != {}",
                        topk[route], reference.weights[slot]
                    ));
                }
            }
        }
        let mut route_receipt = BTreeMap::new();
        route_receipt.insert("case".into(), json!(case));
        route_receipt.insert("stage".into(), json!("router"));
        route_receipt.insert("correct".into(), json!(true));
        session.receipt("moe", route_receipt)?;
        let permutation = oracle::expert_permute(
            &ids.iter().map(|&id| id as usize).collect::<Vec<_>>(),
            self.experts,
            self.gate.knobs.tile_m as usize,
        )
        .map_err(err)?;
        let offsets = download_i32(session, &self.offsets, self.experts + 1)?;
        let padded_live = download_i32(session, &self.padded, 1)?[0];
        if padded_live < 0 || padded_live as usize * 4 > self.sorted.byte_len() {
            return Err(format!("{case} invalid padded count {padded_live}"));
        }
        let padded = padded_live as usize;
        let sorted = download_i32(session, &self.sorted, padded)?;
        let expected_offsets = permutation
            .expert_offsets
            .iter()
            .map(|&x| x as i32)
            .collect::<Vec<_>>();
        let expected_sorted = permutation
            .forward
            .iter()
            .map(|x| x.map_or(-1, |r| r as i32))
            .collect::<Vec<_>>();
        if offsets != expected_offsets
            || sorted != expected_sorted
            || padded != expected_sorted.len()
        {
            let first_sorted_difference = sorted
                .iter()
                .zip(&expected_sorted)
                .position(|(a, b)| a != b);
            return Err(format!(
                "{case} expert permutation differs at offset {:?} or sorted row {:?} (actual {:?}, expected {:?})",
                offsets
                    .iter()
                    .zip(&expected_offsets)
                    .position(|(a, b)| a != b),
                first_sorted_difference,
                first_sorted_difference.map(|i| sorted[i]),
                first_sorted_difference.map(|i| expected_sorted[i])
            ));
        }
        let mut permutation_receipt = BTreeMap::new();
        permutation_receipt.insert("case".into(), json!(case));
        permutation_receipt.insert("stage".into(), json!("expert_permute"));
        permutation_receipt.insert("correct".into(), json!(true));
        permutation_receipt.insert("padded_rows".into(), json!(padded));
        session.receipt("moe", permutation_receipt)?;
        let selected_tokens = if self.tokens <= 8 {
            (0..self.tokens).collect::<Vec<_>>()
        } else {
            vec![0, self.tokens / 2, self.tokens - 1]
        };
        let selected_routes = selected_tokens
            .iter()
            .flat_map(|&token| (0..TOP_K).map(move |slot| token * TOP_K + slot))
            .collect::<Vec<_>>();
        let ranges = selected_routes
            .iter()
            .enumerate()
            .map(|(row, &route)| ExpertRowRange {
                expert: ids[route] as usize & 1,
                start: row,
                end: row + 1,
            })
            .collect::<Vec<_>>();
        let gate_input = selected_routes
            .iter()
            .flat_map(|&route| {
                self.input_host[(route / TOP_K) * HIDDEN..(route / TOP_K + 1) * HIDDEN]
                    .iter()
                    .map(|&v| f64::from(bf16_to_f32(v)))
            })
            .collect::<Vec<_>>();
        let gate_weight = self.gate_decoded.concat();
        let gate_reference = oracle::gemm(
            &gate_input,
            &gate_weight,
            GemmShape {
                m: selected_routes.len(),
                n: self.gate.query.n as usize,
                k: HIDDEN,
                experts: 2,
            },
            &ranges,
            GemmEpilogue {
                residual: None,
                swiglu: Some(SwigluColumns::ConcatenatedHalves),
            },
        )
        .map_err(err)?;
        let intermediate = self.gate.query.n as usize / 2;
        let gate_actual = check_bf16(
            session,
            &self.gate_output,
            intermediate,
            &selected_routes,
            &gate_reference,
            &case,
            "gate_up",
        )?;
        let down_input = selected_routes
            .iter()
            .flat_map(|&route| {
                gate_actual[route * intermediate..(route + 1) * intermediate]
                    .iter()
                    .map(|&v| f64::from(bf16_to_f32(v)))
            })
            .collect::<Vec<_>>();
        let down_weight = self.down_decoded.concat();
        let down_reference = oracle::gemm(
            &down_input,
            &down_weight,
            GemmShape {
                m: selected_routes.len(),
                n: HIDDEN,
                k: intermediate,
                experts: 2,
            },
            &ranges,
            GemmEpilogue {
                residual: None,
                swiglu: None,
            },
        )
        .map_err(err)?;
        let mut weighted = down_reference.clone();
        for (row, &route) in selected_routes.iter().enumerate() {
            for col in 0..HIDDEN {
                let at = row * HIDDEN + col;
                weighted.values[at] *= f64::from(topk[route]);
                weighted.sum_abs_products[at] *= f64::from(topk[route]).abs();
            }
        }
        let down_actual = check_bf16(
            session,
            &self.down_output,
            HIDDEN,
            &selected_routes,
            &weighted,
            &case,
            "weighted_down",
        )?;
        let combine_input = selected_routes
            .iter()
            .flat_map(|&route| {
                down_actual[route * HIDDEN..(route + 1) * HIDDEN]
                    .iter()
                    .map(|&v| f64::from(bf16_to_f32(v)))
            })
            .collect::<Vec<_>>();
        let combine_reference = oracle::expert_combine(
            &combine_input,
            &(0..selected_tokens.len())
                .flat_map(|token| std::iter::repeat_n(token, TOP_K))
                .collect::<Vec<_>>(),
            &vec![1.0; selected_routes.len()],
            selected_tokens.len(),
            HIDDEN,
        )
        .map_err(err)?;
        let final_actual = check_bf16(
            session,
            &self.final_output,
            HIDDEN,
            &selected_tokens,
            &combine_reference,
            &case,
            "preweighted_combine",
        )?;
        let rounded_gate = gate_reference
            .values
            .iter()
            .map(|&value| f64::from(bf16_to_f32(f32_to_bf16(value as f32))))
            .collect::<Vec<_>>();
        let mut full_down = oracle::gemm(
            &rounded_gate,
            &down_weight,
            GemmShape {
                m: selected_routes.len(),
                n: HIDDEN,
                k: intermediate,
                experts: 2,
            },
            &ranges,
            GemmEpilogue {
                residual: None,
                swiglu: None,
            },
        )
        .map_err(err)?;
        for (row, &route) in selected_routes.iter().enumerate() {
            for col in 0..HIDDEN {
                let at = row * HIDDEN + col;
                full_down.values[at] = f64::from(bf16_to_f32(f32_to_bf16(
                    (full_down.values[at] * f64::from(topk[route])) as f32,
                )));
            }
        }
        let full_reference = oracle::expert_combine(
            &full_down.values,
            &(0..selected_tokens.len())
                .flat_map(|token| std::iter::repeat_n(token, TOP_K))
                .collect::<Vec<_>>(),
            &vec![1.0; selected_routes.len()],
            selected_tokens.len(),
            HIDDEN,
        )
        .map_err(err)?;
        let actual = selected_tokens
            .iter()
            .flat_map(|&token| {
                final_actual[token * HIDDEN..(token + 1) * HIDDEN]
                    .iter()
                    .map(|&value| f64::from(bf16_to_f32(value)))
            })
            .collect::<Vec<_>>();
        let layer_gate = oracle::check_layer_cosine(&actual, &full_reference.values, HIDDEN);
        let mut layer_receipt = BTreeMap::new();
        layer_receipt.insert("case".into(), json!(case));
        layer_receipt.insert("stage".into(), json!("end_to_end_layer_cosine"));
        layer_receipt.insert("checked_tokens".into(), json!(selected_tokens));
        layer_receipt.insert("correct".into(), json!(layer_gate.is_ok()));
        if let Err(worst) = &layer_gate {
            layer_receipt.insert(
                "worst_token".into(),
                json!({"index": worst.index,
                "cosine": worst.actual, "bound": worst.bound}),
            );
        }
        session.receipt("moe", layer_receipt)?;
        layer_gate.map_err(|worst| {
            format!(
                "{case} end-to-end token {} cosine {} < {}",
                worst.index, worst.actual, worst.bound
            )
        })?;
        Ok(())
    }
}

fn exact_receipt(
    session: &Session,
    case: &str,
    stage: &str,
    actual: &[u16],
    expected: &[u16],
) -> LabResult<()> {
    let mismatch = actual.iter().zip(expected).position(|(a, b)| a != b);
    let mut receipt = BTreeMap::new();
    receipt.insert("case".into(), json!(case));
    receipt.insert("stage".into(), json!(stage));
    receipt.insert(
        "correct".into(),
        json!(mismatch.is_none() && actual.len() == expected.len()),
    );
    if let Some(index) = mismatch {
        receipt.insert(
            "worst_element".into(),
            json!({"index": index,
            "actual_bits": actual[index], "reference_bits": expected[index]}),
        );
    }
    session.receipt("moe", receipt)?;
    if mismatch.is_some() || actual.len() != expected.len() {
        return Err(format!(
            "{case} {stage}: bitwise output mismatch at {mismatch:?}"
        ));
    }
    Ok(())
}

fn synthetic_gemm(
    session: &Session,
    selection: (Phase, Profile),
    format: WeightFormat,
    path: InputPath,
    projection: Projection,
    interleaved: bool,
    quantized: bool,
) -> LabResult<()> {
    let (phase, profile) = selection;
    let tokens = 1;
    let v = moe::catalog(kernel_lab::catalogs())
        .into_iter()
        .find(|v| {
            v.profile == profile
                && v.phase == phase
                && v.projection == projection
                && v.path == path
                && v.query.weight == format
                && (phase == Phase::Prefill || v.tokens_capacity == tokens as u32)
                && (v.query.epilogues.swiglu == Some(SwigluColumns::Interleaved)) == interleaved
                && (v.query.epilogues.output_quantization == Some(OutputQuantization::Mxfp8))
                    == quantized
        })
        .ok_or("missing synthetic MoE epilogue variant")?;
    synthetic_variant(session, v, false)
}

pub(crate) fn generated_catalog_oracle(session: &Session, v: Variant) -> LabResult<()> {
    synthetic_variant(session, v, true)
}

fn synthetic_variant(session: &Session, v: Variant, sampled: bool) -> LabResult<()> {
    let (phase, profile, format, path, projection) =
        (v.phase, v.profile, v.query.weight, v.path, v.projection);
    let interleaved = v.query.epilogues.swiglu == Some(SwigluColumns::Interleaved);
    let quantized = v.query.epilogues.output_quantization.is_some();
    let tokens = 1;
    let routes = tokens * TOP_K;
    let n = v.query.n as usize;
    let k = v.query.k as usize;
    let output_n = if projection == Projection::GateUp {
        n / 2
    } else {
        n
    };
    let experts = v.query.grouped.ok_or("missing grouped shape")?.experts as usize;
    let ids = (0..routes).collect::<Vec<_>>();
    let permutation =
        oracle::expert_permute(&ids, experts, v.knobs.tile_m as usize).map_err(err)?;
    let offsets = upload_bytes(
        session,
        &i32_bytes(
            &permutation
                .expert_offsets
                .iter()
                .map(|&x| x as i32)
                .collect::<Vec<_>>(),
        ),
    )?;
    let sorted = upload_bytes(
        session,
        &i32_bytes(
            &permutation
                .forward
                .iter()
                .map(|x| x.map_or(-1, |r| r as i32))
                .collect::<Vec<_>>(),
        ),
    )?;
    let padded = upload_bytes(session, &(permutation.forward.len() as i32).to_ne_bytes())?;
    let live = upload_bytes(session, &(tokens as i32).to_ne_bytes())?;
    let active = upload_bytes(session, &i32_bytes(&[TOP_K as i32, 0, 1, 2, 3, 4, 5, 6, 7]))?;
    let weights = (0..routes)
        .map(|route| 0.07 + route as f32 * 0.0125)
        .collect::<Vec<_>>();
    let topk_weights = session.upload_f32(&weights)?;
    let input_rows = if projection == Projection::GateUp {
        tokens
    } else {
        routes
    };
    let bf16_input = (0..input_rows * k)
        .map(|index| f32_to_bf16(sample(0xe190 ^ index as u64) * 0.1))
        .collect::<Vec<_>>();
    let mut input_decoded = Vec::with_capacity(routes * k);
    let (activation, activation_scales, activation_f32_scales) = if path == InputPath::A16Gather {
        for route in 0..routes {
            let source = if projection == Projection::GateUp {
                route / TOP_K
            } else {
                route
            };
            input_decoded.extend(
                bf16_input[source * k..(source + 1) * k]
                    .iter()
                    .map(|&value| f64::from(bf16_to_f32(value))),
            );
        }
        (
            session.upload_bf16(&bf16_input)?,
            upload_bytes(session, &[127])?,
            session.upload_f32(&[1.0])?,
        )
    } else {
        let mut gathered = vec![0.0_f32; permutation.forward.len() * k];
        for (slot, route) in permutation.forward.iter().enumerate() {
            if let Some(route) = route {
                let source = if projection == Projection::GateUp {
                    route / TOP_K
                } else {
                    *route
                };
                for col in 0..k {
                    gathered[slot * k + col] = bf16_to_f32(bf16_input[source * k + col]);
                }
            }
        }
        let q = match format {
            WeightFormat::Mxfp4 | WeightFormat::Mxfp8 => {
                quantize::mxfp8(&gathered, permutation.forward.len(), k)
            }
            WeightFormat::Nvfp4 => quantize::nvfp4(&gathered, permutation.forward.len(), k),
            WeightFormat::Fp8Block128x128(mode) => {
                quantize::fp8_activation_1x128(&gathered, permutation.forward.len(), k, mode)
            }
            _ => return Err("unsupported synthetic A8 activation".into()),
        }
        .map_err(err)?;
        for route in 0..routes {
            let slot = permutation.inverse[route];
            input_decoded.extend((0..k).map(|col| q.decode(slot, col)));
        }
        let (scale_bytes, f32_scales) = match &q.scales {
            Scales::Ue8M0(bytes) | Scales::Ue4M3(bytes) => (bytes.clone(), vec![q.global_scale]),
            Scales::F32(values) => (vec![127], values.clone()),
        };
        (
            upload_bytes(session, &q.codes)?,
            upload_bytes(session, &scale_bytes)?,
            session.upload_f32(&f32_scales)?,
        )
    };
    let bank = WeightBank::new(&v, 0xf280)?;
    let mut decoded = bank.decoded.concat();
    if path == InputPath::A16Gather {
        a16_weight_operand(&mut decoded, format, bank.f32_scales[0]);
    }
    let gpu_weights = bank.upload(session)?;
    let output = session.allocate(routes * output_n * 2)?;
    let quantized_codes = session.allocate(if quantized { routes * output_n } else { 1 })?;
    let quantized_scales = session.allocate(if quantized { routes * output_n / 32 } else { 1 })?;
    let counts = RuntimeCounts {
        m: tokens as u32,
        batch: 0,
        context: 0,
        routed_rows: routes as u32,
        split_k: v.knobs.split_k_capacity,
    };
    let partials = session.allocate(v.plan(counts).map_err(err)?.workspace_bytes.max(1) as usize)?;
    let functions = load_moe(session, &v, counts)?;
    let operands = MoeOperands {
        activation: &activation,
        activation_scales: &activation_scales,
        activation_f32_scales: &activation_f32_scales,
        weights: &gpu_weights,
        output: &output,
        topk_weights: &topk_weights,
        offsets: &offsets,
        sorted: &sorted,
        padded: &padded,
        live: &live,
        active: &active,
        quantized_output: &quantized_codes,
        quantized_scales: &quantized_scales,
        partials: &partials,
    };
    let mut previous = None;
    for _ in 0..if sampled { 3 } else { 1 } {
        if sampled {
            poison::poison(session, &[&output])?;
        }
        launch_moe(session, &v, &functions, counts, &operands, true)?;
        let actual = session.download_bf16(&output, routes * output_n)?;
        if previous.as_ref().is_some_and(|old| old != &actual) {
            return Err("generated grouped replay differs".into());
        }
        previous = Some(actual);
    }
    let ranges = ids
        .iter()
        .enumerate()
        .map(|(row, &expert)| ExpertRowRange {
            expert: expert & 1,
            start: row,
            end: row + 1,
        })
        .collect::<Vec<_>>();
    let mut reference = oracle::gemm(
        &input_decoded,
        &decoded,
        GemmShape {
            m: routes,
            n,
            k,
            experts: 2,
        },
        &ranges,
        GemmEpilogue {
            residual: None,
            swiglu: (projection == Projection::GateUp).then_some(if interleaved {
                SwigluColumns::Interleaved
            } else {
                SwigluColumns::ConcatenatedHalves
            }),
        },
    )
    .map_err(err)?;
    if projection == Projection::Down {
        for (route, &weight) in weights.iter().enumerate() {
            for col in 0..output_n {
                let at = route * output_n + col;
                reference.values[at] *= f64::from(weight);
                reference.sum_abs_products[at] *= f64::from(weight).abs();
            }
        }
    }
    let case = format!(
        "{}_{phase:?}_synthetic_{format:?}_{path:?}_{projection:?}_interleaved{interleaved}_quantized{quantized}",
        profile.as_str(),
    );
    check_bf16(
        session,
        &output,
        output_n,
        &ids,
        &reference,
        &case,
        "grouped_gemm",
    )?;
    if quantized {
        let expected = oracle::output_quantize(
            &reference.values,
            routes,
            output_n,
            OutputQuantization::Mxfp8,
        )
        .map_err(err)?;
        let mut codes = vec![0; routes * output_n];
        let mut scales = vec![0; routes * output_n / 32];
        quantized_codes
            .copy_to_bytes(&mut codes, &session.stream)
            .map_err(err)?;
        quantized_scales
            .copy_to_bytes(&mut scales, &session.stream)
            .map_err(err)?;
        let expected_scales = match expected.scales {
            Scales::Ue8M0(scales) => scales,
            _ => return Err("MXFP8 oracle produced unexpected scale type".into()),
        };
        let wrong_code = codes.iter().zip(&expected.codes).position(|(a, b)| a != b);
        let wrong_scale = scales
            .iter()
            .zip(&expected_scales)
            .position(|(a, b)| a != b);
        let mut receipt = BTreeMap::new();
        receipt.insert("case".into(), json!(case));
        receipt.insert("stage".into(), json!("mxfp8_output"));
        receipt.insert(
            "correct".into(),
            json!(wrong_code.is_none() && wrong_scale.is_none()),
        );
        receipt.insert("wrong_code".into(), json!(wrong_code));
        receipt.insert("wrong_scale".into(), json!(wrong_scale));
        session.receipt("moe", receipt)?;
        if wrong_code.is_some() || wrong_scale.is_some() {
            return Err(format!(
                "{case}: MXFP8 output differs at code {wrong_code:?}, scale {wrong_scale:?}"
            ));
        }
    }
    Ok(())
}

#[test]
fn grouped_moe_synthetic_formats() {
    gpu_test(6 * 1024 * 1024 * 1024, |session| {
        for (profile, formats) in [
            (
                model_mimo26_contract::sm120::config::PROFILE,
                &[WeightFormat::Mxfp4][..],
            ),
            (
                Profile::StdDenseMoe,
                &[
                    WeightFormat::Fp8Block128x128(ScaleMode::ExactF32),
                    WeightFormat::Nvfp4,
                    WeightFormat::Mxfp8,
                ][..],
            ),
        ] {
            for &format in formats {
                for path in [InputPath::A16Gather, InputPath::A8Pregather] {
                    for projection in [Projection::GateUp, Projection::Down] {
                        synthetic_gemm(
                            session,
                            (Phase::Decode, profile),
                            format,
                            path,
                            projection,
                            false,
                            false,
                        )?;
                        if profile == Profile::StdDenseMoe && projection == Projection::GateUp {
                            synthetic_gemm(
                                session,
                                (Phase::Decode, profile),
                                format,
                                path,
                                projection,
                                true,
                                false,
                            )?;
                        }
                        if profile == model_mimo26_contract::sm120::config::PROFILE
                            && path == InputPath::A8Pregather
                            && projection == Projection::GateUp
                        {
                            synthetic_gemm(
                                session,
                                (Phase::Decode, profile),
                                format,
                                path,
                                projection,
                                false,
                                true,
                            )?;
                        }
                    }
                }
            }
        }
        Ok(())
    })
    .expect("synthetic grouped MoE GEMM formats");
}

#[test]
fn grouped_moe_fragment_prefill_formats() {
    gpu_test(6 * 1024 * 1024 * 1024, |session| {
        for path in [InputPath::A16Gather, InputPath::A8Pregather] {
            for projection in [Projection::GateUp, Projection::Down] {
                synthetic_gemm(
                    session,
                    (
                        Phase::Prefill,
                        model_mimo26_contract::sm120::config::PROFILE,
                    ),
                    WeightFormat::Mxfp4,
                    path,
                    projection,
                    false,
                    false,
                )?;
                if path == InputPath::A8Pregather && projection == Projection::GateUp {
                    synthetic_gemm(
                        session,
                        (
                            Phase::Prefill,
                            model_mimo26_contract::sm120::config::PROFILE,
                        ),
                        WeightFormat::Mxfp4,
                        path,
                        projection,
                        false,
                        true,
                    )?;
                }
            }
        }
        Ok(())
    })
    .expect("MiMo fragment prefill formats");
}

#[test]
fn grouped_moe_mimo_full_path() {
    gpu_test(6 * 1024 * 1024 * 1024, |session| {
        let mut failures = replay::Failures::default();
        for (phase, tokens) in [
            (Phase::Decode, 1),
            (Phase::Decode, 8),
            (Phase::Decode, 16),
            // A 32-row step, as served above 16 rows.
            (Phase::Prefill, 32),
            (Phase::Prefill, 512),
            (Phase::Prefill, 2048),
        ] {
            let result = (|| {
                if std::env::var("CHOPINFER_MOE_FULL_TOKENS")
                    .ok()
                    .is_some_and(|selected| selected != tokens.to_string())
                {
                    return Ok(());
                }
                let case = FullCase::new(
                    session,
                    model_mimo26_contract::sm120::config::PROFILE,
                    phase,
                    tokens,
                )?;
                let mut checks = case.replay_checks(session, true, true, true, true)?;
                checks.phase(ReplayPhase::Eager, false, || case.launch(session, true))?;
                case.check(session)?;
                if phase == Phase::Decode {
                    let expected = session.download_bf16(&case.final_output, tokens * HIDDEN)?;
                    let mut fused_checks =
                        case.replay_checks(session, false, false, false, true)?;
                    fused_checks.phase(ReplayPhase::Eager, false, || {
                        case.launch_fused_down_combine(session, true)
                    })?;
                    let actual = session.download_bf16(&case.final_output, tokens * HIDDEN)?;
                    if actual != expected {
                        return Err(format!(
                            "fused down combine t{tokens} differs at element {:?}",
                            actual.iter().zip(&expected).position(|(a, b)| a != b)
                        ));
                    }
                }

                Ok(())
            })();
            failures.check(&format!("full path {phase:?} t{tokens}"), result);
        }
        failures.finish()
    })
    .expect("MiMo grouped MoE full path");
}

/// The expert path at TP2's per-rank intermediate (1024).
#[test]
fn grouped_moe_mimo_rank_full_paths() {
    gpu_test(6 * 1024 * 1024 * 1024, |session| {
        let mut failures = replay::Failures::default();
        for profile in model_mimo26_contract::sm120::PROFILES.into_iter().skip(1) {
            for (phase, tokens) in [
                (Phase::Decode, 1),
                (Phase::Decode, 8),
                (Phase::Prefill, 32),
                (Phase::Prefill, 512),
            ] {
                let result = (|| {
                    let case = FullCase::new(session, profile, phase, tokens)?;
                    case.launch(session, true)?;
                    case.check(session)?;

                    Ok(())
                })();
                failures.check(&format!("full path {phase:?} t{tokens}"), result);
            }
        }
        failures.finish()
    })
    .expect("MiMo TP2 grouped MoE full path");
}

#[test]
fn grouped_moe_mimo_decode_graph_replay() {
    gpu_test(6 * 1024 * 1024 * 1024, |session| {
        let mut failures = replay::Failures::default();
        for tokens in [1, 8] {
            let result = (|| {
                if std::env::var("CHOPINFER_MOE_FULL_TOKENS")
                    .ok()
                    .is_some_and(|selected| selected != tokens.to_string())
                {
                    return Ok(());
                }
                let case = FullCase::new(
                    session,
                    model_mimo26_contract::sm120::config::PROFILE,
                    Phase::Decode,
                    tokens,
                )?;
                let mut checks = case.replay_checks(session, true, true, true, true)?;
                checks.phase(ReplayPhase::Eager, false, || case.launch(session, true))?;
                let graph = session.capture(|_| case.launch(session, true))?;
                checks.phase(ReplayPhase::Graph, false, || {
                    graph.launch(&session.stream).map_err(err)
                })?;
                let first = (
                    session.download_bf16(&case.final_output, tokens * HIDDEN)?,
                    download_i32(session, &case.ids, tokens * TOP_K)?,
                    download_i32(session, &case.offsets, case.experts + 1)?,
                    download_i32(session, &case.padded, 1)?,
                );
                checks.phase(ReplayPhase::Graph, false, || {
                    graph.launch(&session.stream).map_err(err)
                })?;
                let second = (
                    session.download_bf16(&case.final_output, tokens * HIDDEN)?,
                    download_i32(session, &case.ids, tokens * TOP_K)?,
                    download_i32(session, &case.offsets, case.experts + 1)?,
                    download_i32(session, &case.padded, 1)?,
                );
                if first != second {
                    return Err(format!(
                        "fused router t{tokens} graph replay differs bitwise"
                    ));
                }
                case.check(session)?;
                Ok(())
            })();
            failures.check(&format!("decode graph t{tokens}"), result);
        }
        failures.finish()
    })
    .expect("MiMo decode graph replay");
}

fn read_mimo_rank0_gateup_checkpoint(root: &Path) -> LabResult<(Vec<u8>, Vec<u8>)> {
    let config = model_mimo26::config::MiMoConfig::load(root).map_err(err)?;
    let catalog = model_mimo26::catalog::Catalog::load(root, &config).map_err(err)?;
    let read_tensor = |projection: &str, suffix: &str, width: usize| -> LabResult<Vec<u8>> {
        let name = format!("model.layers.1.mlp.experts.0.{projection}.{suffix}");
        let tensor = catalog.tensor(&name).map_err(err)?;
        if tensor.dtype.safetensors_name() != "U8" || tensor.shape != [2048, width as u64] {
            return Err(format!("unexpected checkpoint dtype or shape for {name}"));
        }
        let source = catalog.source(&tensor.file).map_err(err)?;
        let mut file = source.try_clone().map_err(err)?;
        file.seek(SeekFrom::Start(tensor.file_start)).map_err(err)?;
        let mut data = vec![0; 512 * width];
        file.read_exact(&mut data).map_err(err)?;
        source.verify().map_err(err)?;
        Ok(data)
    };
    let mut codes = read_tensor("gate_proj", "weight", 2048)?;
    codes.extend(read_tensor("up_proj", "weight", 2048)?);
    let mut scales = read_tensor("gate_proj", "weight_scale", 128)?;
    scales.extend(read_tensor("up_proj", "weight_scale", 128)?);
    Ok((codes, scales))
}

#[test]
fn grouped_moe_mimo_real_layer1_gate_up() {
    gpu_test(1024 * 1024 * 1024, |session| {
        let Some(root) = std::env::var_os("CHOPINFER_MIMO_CHECKPOINT").map(PathBuf::from) else {
            kernel_lab::skip(
                "grouped_moe_mimo_real_layer1_gate_up: set CHOPINFER_MIMO_CHECKPOINT",
            )?;
            return Ok(());
        };
        let (codes, scales) = read_mimo_rank0_gateup_checkpoint(&root)?;
        let fragment = Mxfp4MoeFragmentLayout {
            experts: 1,
            rows_per_expert: 1024,
            k: HIDDEN,
        };
        let packed_codes = fragment.pack_codes(&codes).map_err(err)?.bytes;
        let packed_scales = fragment.pack_scales(&scales).map_err(err)?.bytes;
        let v = variant(
            model_mimo26_contract::sm120::config::PROFILE,
            Phase::Decode,
            Projection::GateUp,
            InputPath::A16Gather,
            WeightFormat::Mxfp4,
            2,
        )?;
        let input = (0..2 * HIDDEN)
            .map(|i| f32_to_bf16(sample(0xd710 ^ i as u64) * 0.3))
            .collect::<Vec<_>>();
        let mut offsets = vec![16_i32; 257];
        offsets[0] = 0;
        let mut sorted = vec![-1_i32; 16];
        sorted[0] = 0;
        sorted[1] = 8;
        let full_codes = session.allocate(256 * codes.len())?;
        full_codes
            .copy_from_bytes(&packed_codes, &session.stream)
            .map_err(err)?;
        let gpu_weights = GpuWeights {
            codes: full_codes,
            scales: upload_bytes(session, &packed_scales)?,
            f32_scales: session.upload_f32(&[1.0])?,
        };
        let output = session.allocate(16 * 512 * 2)?;
        let counts = RuntimeCounts {
            m: 2,
            batch: 0,
            context: 0,
            routed_rows: 16,
            split_k: v.knobs.split_k_capacity,
        };
        let operands = MoeOperands {
            activation: &session.upload_bf16(&input)?,
            activation_scales: &upload_bytes(session, &[127])?,
            activation_f32_scales: &session.upload_f32(&[1.0])?,
            weights: &gpu_weights,
            output: &output,
            topk_weights: &session.upload_f32(&[1.0; 16])?,
            offsets: &upload_bytes(session, &i32_bytes(&offsets))?,
            sorted: &upload_bytes(session, &i32_bytes(&sorted))?,
            padded: &upload_bytes(session, &i32_bytes(&[16]))?,
            live: &upload_bytes(session, &i32_bytes(&[2]))?,
            active: &upload_bytes(session, &i32_bytes(&[1, 0]))?,
            quantized_output: &upload_bytes(session, &[0])?,
            quantized_scales: &upload_bytes(session, &[0])?,
            partials: &session.allocate(v.plan(counts).map_err(err)?.workspace_bytes as usize)?,
        };
        let functions = load_moe(session, &v, counts)?;
        launch_moe(session, &v, &functions, counts, &operands, true)?;
        let actual = session.download_bf16(&output, 16 * 512)?;
        let reference = mxfp4_gate_up(&input, 2, &codes, &scales)?;
        let mut dot = 0.0_f64;
        let mut actual_norm = 0.0_f64;
        let mut expected_norm = 0.0_f64;
        let mut error_norm = 0.0_f64;
        for token in 0..2 {
            for feature in 0..512 {
                let expected = reference[token * 512 + feature];
                let got = f64::from(bf16_to_f32(actual[token * TOP_K * 512 + feature]));
                dot += expected * got;
                actual_norm += got * got;
                expected_norm += expected * expected;
                error_norm += (got - expected).powi(2);
            }
        }
        let cosine = dot / (actual_norm * expected_norm).sqrt();
        let relative_l2 = (error_norm / expected_norm).sqrt();
        session.receipt(
            "moe",
            BTreeMap::from([
                ("case".into(), json!("mimo_layer1_expert0_gateup_t2")),
                ("cosine".into(), json!(cosine)),
                ("relative_l2".into(), json!(relative_l2)),
                ("rank".into(), json!(0)),
            ]),
        )?;
        if !cosine.is_finite() || cosine < 0.995 || relative_l2 > 0.05 {
            return Err(format!(
                "real-weight gate/up differs: cosine={cosine:.6} relative_l2={relative_l2:.6}"
            ));
        }
        Ok(())
    })
    .expect("MiMo layer 1 real-weight gate/up");
}

#[test]
fn grouped_moe_mimo_fused_route_matches_row_engine() {
    gpu_test(6 * 1024 * 1024 * 1024, |session| {
        let mut failures = replay::Failures::default();
        for tokens in [1, 16] {
            let result = (|| {
                let mut case = FullCase::new(
                    session,
                    model_mimo26_contract::sm120::config::PROFILE,
                    Phase::Decode,
                    tokens,
                )?;
                let logits = (0..tokens * 256)
                    .map(|i| sample(0xc530 ^ i as u64) * 4.0)
                    .collect::<Vec<_>>();
                let bias = (0..256)
                    .map(|i| sample(0xa880 ^ i as u64) * 0.1)
                    .collect::<Vec<_>>();
                case.logits = session.upload_f32(&logits)?;
                case.bias = session.upload_f32(&bias)?;
                let mut row_checks =
                    case.route_only_checks(session, case.gate.knobs.tile_m as usize, false)?;
                row_checks.phase(ReplayPhase::Eager, false, || {
                    for stage in ["router", "count", "scan", "scatter"] {
                        case.launch_row_stage(session, stage)?;
                    }
                    Ok(())
                })?;
                let routes = tokens * TOP_K;
                let expected_ids = download_i32(session, &case.ids, routes)?;
                let expected_weights = download_f32(session, &case.topk, routes)?;
                let expected_offsets = download_i32(session, &case.offsets, 257)?;
                let padded = download_i32(session, &case.padded, 1)?[0] as usize;
                let expected_sorted = download_i32(session, &case.sorted, padded)?;
                let mut route_checks = case.route_only_checks(session, 16, true)?;
                route_checks.phase(ReplayPhase::Eager, false, || {
                    case.launch_decode_route(session, true)
                })?;
                let ids = download_i32(session, &case.ids, routes)?;
                let weights = download_f32(session, &case.topk, routes)?;
                let offsets = download_i32(session, &case.offsets, 257)?;
                let sorted = download_i32(session, &case.sorted, padded)?;
                let active = download_i32(session, &case.active, 257)?;
                let expected_active = (0..256)
                    .filter(|&expert| expected_offsets[expert + 1] > expected_offsets[expert])
                    .map(|expert| expert as i32)
                    .collect::<Vec<_>>();
                let max_weight_error = weights
                    .iter()
                    .zip(&expected_weights)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0_f32, f32::max);
                let correct = ids == expected_ids
                    && offsets == expected_offsets
                    && sorted == expected_sorted
                    && download_i32(session, &case.padded, 1)?[0] as usize == padded
                    && active[0] == expected_active.len() as i32
                    && active[1..=expected_active.len()] == expected_active
                    && max_weight_error <= 2e-6;
                session.receipt(
                    "moe",
                    BTreeMap::from([
                        ("case".into(), json!(format!("mimo_fused_route_t{tokens}"))),
                        ("correct".into(), json!(correct)),
                        ("active_experts".into(), json!(expected_active.len())),
                        ("max_weight_error".into(), json!(max_weight_error)),
                    ]),
                )?;
                if !correct {
                    return Err(format!(
                        "fused route differs from stable row engine at {tokens} tokens"
                    ));
                }

                Ok(())
            })();
            failures.check(&format!("route reference t{tokens}"), result);
        }
        failures.finish()
    })
    .expect("fused MiMo route matches row engine");
}

#[test]
fn grouped_moe_mimo_fused_router_matches_decode_route() {
    gpu_test(6 * 1024 * 1024 * 1024, |session| {
        let mut failures = replay::Failures::default();
        for tokens in [1, 2, 4, 8, 16] {
            let result = (|| {
                let case = FullCase::new(
                    session,
                    model_mimo26_contract::sm120::config::PROFILE,
                    Phase::Decode,
                    tokens,
                )?;
                if case
                    .router_gemm
                    .as_ref()
                    .is_none_or(|gemm| gemm.fused_route.is_none())
                {
                    return Err(format!("fused router entry is absent at {tokens} tokens"));
                }
                // Nonzero weights exercise the GEMM arithmetic as well as routing.
                let layout = kernel_core::transform::fragment::FragmentWeightLayout {
                    rows: 256,
                    k: HIDDEN,
                    format: kernel_core::transform::fragment::FragmentWeightFormat::Bf16,
                };
                let raw = (0..256 * HIDDEN)
                    .flat_map(|i| {
                        f32_to_bf16(
                            sample(0xfeed ^ (i / (2 * HIDDEN) * HIDDEN + i % HIDDEN) as u64) * 0.1,
                        )
                        .to_le_bytes()
                    })
                    .collect::<Vec<_>>();
                let packed = layout.pack(&raw).map_err(err)?;
                let gemm = case.router_gemm.as_ref().unwrap();
                gemm.weights
                    .copy_from_bytes(&packed.bytes, &session.stream)
                    .map_err(err)?;
                case.bias
                    .copy_from_bytes(
                        &f32_bytes(
                            &(0..256)
                                .map(|expert| sample(0xfed0 ^ (expert / 2) as u64) * 0.0001)
                                .collect::<Vec<_>>(),
                        ),
                        &session.stream,
                    )
                    .map_err(err)?;
                let routes = tokens * TOP_K;
                let snapshot = |session: &Session| -> LabResult<Vec<Vec<u8>>> {
                    let padded = download_i32(session, &case.padded, 1)?[0] as usize;
                    Ok(vec![
                        bytemuck_i32(&download_i32(session, &case.ids, routes)?),
                        download_f32(session, &case.topk, routes)?
                            .iter()
                            .flat_map(|v| v.to_bits().to_ne_bytes())
                            .collect(),
                        bytemuck_i32(&download_i32(session, &case.counts, 256)?),
                        bytemuck_i32(&download_i32(session, &case.offsets, 257)?),
                        bytemuck_i32(&download_i32(session, &case.sorted, padded)?),
                        bytemuck_i32(&download_i32(session, &case.active, 257)?),
                        bytemuck_i32(&[padded as i32]),
                    ])
                };
                let mut fused_checks =
                    case.replay_maps(session, [true, false, false, false], true)?;
                fused_checks.phase(ReplayPhase::Eager, false, || {
                    case.launch_fused_router_route(session, &case.input, true)
                })?;
                let fused = snapshot(session)?;
                let fused_logits = download_f32(session, &case.logits, tokens * 256)?;
                let mut plain_checks =
                    case.replay_maps(session, [true, false, false, false], false)?;
                plain_checks.phase(ReplayPhase::Eager, false, || {
                    gemm.launch(session, &case.input, &case.logits, true)?;
                    case.launch_decode_route(session, true)
                })?;
                let plain_logits = download_f32(session, &case.logits, tokens * 256)?;
                if fused_logits
                    .iter()
                    .map(|v| v.to_bits())
                    .ne(plain_logits.iter().map(|v| v.to_bits()))
                {
                    return Err(format!("fused router logits differ at {tokens} tokens"));
                }
                let reference = snapshot(session)?;
                let graph = session
                    .capture(|_| case.launch_fused_router_route(session, &case.input, true))?;
                for _ in 0..3 {
                    fused_checks.phase(ReplayPhase::Graph, false, || {
                        graph.launch(&session.stream).map_err(err)
                    })?;
                    if snapshot(session)? != fused {
                        return Err(format!("fused router replay differs at {tokens} tokens"));
                    }
                }
                if std::env::var("CHOPINFER_KERNEL_BENCH").as_deref() == Ok("1") {
                    let old = session.capture(|_| {
                        for _ in 0..47 {
                            gemm.launch(session, &case.input, &case.logits, true)?;
                            case.launch_decode_route(session, true)?;
                        }
                        Ok(())
                    })?;
                    let repeated = session.capture(|_| {
                        for _ in 0..47 {
                            case.launch_fused_router_route(session, &case.input, true)?;
                        }
                        Ok(())
                    })?;
                    let before = time(session, 5, 15, |stream| old.launch(stream).map_err(err))?;
                    let after = time(session, 5, 15, |stream| {
                        repeated.launch(stream).map_err(err)
                    })?;
                    session.receipt(
                        "moe",
                        BTreeMap::from([
                            ("case".into(), json!("route_samp_router_pair")),
                            ("tokens".into(), json!(tokens)),
                            ("before".into(), json!(before.median_ms * 1000.0 / 47.0)),
                            ("after".into(), json!(after.median_ms * 1000.0 / 47.0)),
                        ]),
                    )?;
                }
                let names = [
                    "ids",
                    "weights",
                    "counts",
                    "offsets",
                    "sorted_rows",
                    "active",
                    "padded",
                ];
                let mismatch = names
                    .iter()
                    .zip(fused.iter().zip(&reference))
                    .find(|(_, (a, b))| a != b)
                    .map(|(name, _)| *name);
                session.receipt(
                    "moe",
                    BTreeMap::from([
                        (
                            "case".into(),
                            json!(format!("mimo_fused_router_route_t{tokens}")),
                        ),
                        ("correct".into(), json!(mismatch.is_none())),
                        ("first_mismatch".into(), json!(mismatch)),
                    ]),
                )?;
                if let Some(name) = mismatch {
                    return Err(format!(
                        "fused router {name} differs from the decode route at {tokens} tokens"
                    ));
                }

                Ok(())
            })();
            failures.check(&format!("fused router t{tokens}"), result);
        }
        failures.finish()
    })
    .expect("fused MiMo router route matches the decode route");
}

/// Oracle top-8 IDs and weights of each token's F32 logits and bias.
fn route_oracle(logits: &[f32], bias: &[f32]) -> LabResult<(Vec<i32>, Vec<f64>)> {
    let bias = bias.iter().map(|&b| f64::from(b)).collect::<Vec<_>>();
    let mut ids = Vec::new();
    let mut weights = Vec::new();
    for row in logits.chunks_exact(256) {
        let selection = oracle::router(
            &row.iter().map(|&v| f64::from(v)).collect::<Vec<_>>(),
            Some(&bias),
            TOP_K,
            RouterKind::SigmoidBiasSelectNormalized {
                scaling_factor: 1.0,
            },
        )
        .map_err(err)?;
        ids.extend(selection.expert_ids.iter().map(|&id| id as i32));
        weights.extend(selection.weights);
    }
    Ok((ids, weights))
}

/// IDs, in route order, must equal the oracle's; the largest relative weight
/// error is returned.
fn route_weight_error(
    ids: &[i32],
    weights: &[f32],
    expected: &(Vec<i32>, Vec<f64>),
) -> LabResult<f64> {
    if ids != expected.0 {
        return Err("route IDs or their order differ from the oracle".into());
    }
    Ok(weights
        .iter()
        .zip(&expected.1)
        .map(|(&actual, &oracle)| (f64::from(actual) - oracle).abs() / oracle.abs().max(1e-30))
        .fold(0.0, f64::max))
}

/// Router logits well below zero with a positive correction bias, uniform
/// (-20 and 1, whose biased F32 score is exactly 1) and nonuniform with
/// MiMo-like biases near 11. The bias only ranks experts; the weights are
/// the unbiased sigmoids. The nonuniform biases are 1/256 apart, far more
/// than any sigmoid here, so F32 and f64 scores rank experts alike.
fn negative_route_cases(tokens: usize) -> [(&'static str, Vec<f32>, Vec<f32>); 2] {
    [
        ("uniform", vec![-20.0; tokens * 256], vec![1.0; 256]),
        (
            "nonuniform",
            (0..tokens * 256)
                .map(|i| -14.0 + 6.0 * sample(0x10c1 ^ i as u64))
                .collect(),
            (0..256)
                .map(|e| 10.5 + ((e * 97) % 256) as f32 / 256.0)
                .collect(),
        ),
    ]
}

#[test]
fn grouped_moe_mimo_decode_route_weights_match_oracle() {
    gpu_test(6 * 1024 * 1024 * 1024, |session| {
        let mut failures = replay::Failures::default();
        for tokens in [1, 2, 4, 8, 16] {
            let mut case = FullCase::new(
                session,
                model_mimo26_contract::sm120::config::PROFILE,
                Phase::Decode,
                tokens,
            )?;
            let routes = tokens * TOP_K;
            for (name, logits, bias) in negative_route_cases(tokens) {
                let result = (|| {
                    case.logits = session.upload_f32(&logits)?;
                    case.bias = session.upload_f32(&bias)?;
                    let expected = route_oracle(&logits, &bias)?;
                    let mut row_checks = replay::Checks::new(session)
                        .output(
                            "route IDs",
                            &case.ids,
                            WriteMap::of(DType::I32).prefix(routes)?,
                        )
                        .output(
                            "route weights",
                            &case.topk,
                            WriteMap::of(DType::F32).prefix(routes)?,
                        );
                    row_checks.phase(ReplayPhase::Eager, false, || {
                        case.launch_row_stage(session, "router")
                    })?;
                    let row_ids = download_i32(session, &case.ids, routes)?;
                    let row_weights = download_f32(session, &case.topk, routes)?;
                    let mut route_checks = case.route_only_checks(session, 16, true)?;
                    route_checks.phase(ReplayPhase::Eager, false, || {
                        case.launch_decode_route(session, true)
                    })?;
                    let ids = download_i32(session, &case.ids, routes)?;
                    let weights = download_f32(session, &case.topk, routes)?;
                    let error = route_weight_error(&ids, &weights, &expected)
                        .map_err(|e| format!("decode route {name} t{tokens}: {e}"))?;
                    let row_error = route_weight_error(&row_ids, &row_weights, &expected)
                        .map_err(|e| format!("row router {name} t{tokens}: {e}"))?;
                    let matches_row = ids == row_ids
                        && weights
                            .iter()
                            .map(|v| v.to_bits())
                            .eq(row_weights.iter().map(|v| v.to_bits()));
                    let correct = error <= 1e-5 && row_error <= 1e-5 && matches_row;
                    session.receipt(
                        "moe",
                        BTreeMap::from([
                            (
                                "case".into(),
                                json!(format!("mimo_decode_route_negative_{name}_t{tokens}")),
                            ),
                            ("correct".into(), json!(correct)),
                            ("max_relative_weight_error".into(), json!(error)),
                            ("row_max_relative_weight_error".into(), json!(row_error)),
                            ("bitwise_row_engine".into(), json!(matches_row)),
                        ]),
                    )?;
                    if !correct {
                        return Err(format!(
                            "decode route {name} weights at {tokens} tokens: relative error \
                         {error:e} (row engine {row_error:e}), bitwise row match {matches_row}"
                        ));
                    }

                    Ok(())
                })();
                failures.check(&format!("decode weights t{tokens} {name}"), result);
            }
        }
        failures.finish()
    })
    .expect("MiMo decode route weights match the oracle");
}

#[test]
fn grouped_moe_mimo_fused_router_weights_match_oracle() {
    gpu_test(6 * 1024 * 1024 * 1024, |session| {
        let mut failures = replay::Failures::default();
        for tokens in [1, 2, 4, 8, 16] {
            let mut case = FullCase::new(
                session,
                model_mimo26_contract::sm120::config::PROFILE,
                Phase::Decode,
                tokens,
            )?;
            let routes = tokens * TOP_K;
            // Token t's input is 1 at columns 0 and 1 + t, so its logit for
            // expert r is W[r][0] + W[r][1 + t]: the case's logits, rounded
            // to BF16 weights. The oracle reads the logits the GEMM wrote.
            let mut input = vec![0.0_f32; tokens * HIDDEN];
            for token in 0..tokens {
                input[token * HIDDEN] = 1.0;
                input[token * HIDDEN + 1 + token] = 1.0;
            }
            let input =
                session.upload_bf16(&input.iter().map(|&v| f32_to_bf16(v)).collect::<Vec<_>>())?;
            let layout = kernel_core::transform::fragment::FragmentWeightLayout {
                rows: 256,
                k: HIDDEN,
                format: kernel_core::transform::fragment::FragmentWeightFormat::Bf16,
            };
            for (name, logits, bias) in negative_route_cases(tokens) {
                let result = (|| {
                    let mut raw = vec![0.0_f32; 256 * HIDDEN];
                    for expert in 0..256 {
                        let base = logits[expert];
                        raw[expert * HIDDEN] = base;
                        for token in 0..tokens {
                            raw[expert * HIDDEN + 1 + token] = logits[token * 256 + expert] - base;
                        }
                    }
                    let raw = raw
                        .iter()
                        .flat_map(|&v| f32_to_bf16(v).to_le_bytes())
                        .collect::<Vec<_>>();
                    let packed = layout.pack(&raw).map_err(err)?;
                    let gemm = case.router_gemm.as_ref().ok_or("router GEMV is absent")?;
                    gemm.weights
                        .copy_from_bytes(&packed.bytes, &session.stream)
                        .map_err(err)?;
                    case.bias = session.upload_f32(&bias)?;
                    let mut fused_checks =
                        case.replay_maps(session, [true, false, false, false], true)?;
                    fused_checks.phase(ReplayPhase::Eager, false, || {
                        case.launch_fused_router_route(session, &input, true)
                    })?;
                    let ids = download_i32(session, &case.ids, routes)?;
                    let weights = download_f32(session, &case.topk, routes)?;
                    let written = download_f32(session, &case.logits, tokens * 256)?;
                    let expected = route_oracle(&written, &bias)?;
                    let error = route_weight_error(&ids, &weights, &expected)
                        .map_err(|e| format!("fused router {name} t{tokens}: {e}"))?;
                    let mut route_checks = case.route_only_checks(session, 16, true)?;
                    route_checks.phase(ReplayPhase::Eager, false, || {
                        case.launch_decode_route(session, true)
                    })?;
                    let matches_route = download_i32(session, &case.ids, routes)? == ids
                        && download_f32(session, &case.topk, routes)?
                            .iter()
                            .map(|v| v.to_bits())
                            .eq(weights.iter().map(|v| v.to_bits()));
                    let correct = error <= 1e-5 && matches_route;
                    session.receipt(
                        "moe",
                        BTreeMap::from([
                            (
                                "case".into(),
                                json!(format!("mimo_fused_router_negative_{name}_t{tokens}")),
                            ),
                            ("correct".into(), json!(correct)),
                            ("max_relative_weight_error".into(), json!(error)),
                            ("bitwise_decode_route".into(), json!(matches_route)),
                        ]),
                    )?;
                    if !correct {
                        return Err(format!(
                            "fused router {name} weights at {tokens} tokens: relative error \
                         {error:e}, bitwise decode route match {matches_route}"
                        ));
                    }

                    Ok(())
                })();
                failures.check(&format!("fused weights t{tokens} {name}"), result);
            }
        }
        failures.finish()
    })
    .expect("MiMo fused router weights match the oracle");
}

fn bytemuck_i32(values: &[i32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_ne_bytes()).collect()
}

#[test]
fn grouped_moe_mimo_weighted_down_row_combine() {
    gpu_test(6 * 1024 * 1024 * 1024, |session| {
        let case = FullCase::new(
            session,
            model_mimo26_contract::sm120::config::PROFILE,
            Phase::Decode,
            1,
        )?;
        let mut checks = case.replay_checks(session, true, true, true, true)?;
        checks.phase(ReplayPhase::Eager, false, || case.launch(session, true))?;
        let weights = download_f32(session, &case.topk, TOP_K)?;
        if weights.iter().any(|w| !w.is_finite()) || weights.iter().all(|&w| w == 1.0) {
            return Err("MoE combine test needs nontrivial router weights".into());
        }
        let routed = session.download_bf16(&case.down_output, TOP_K * HIDDEN)?;
        let expected = (0..HIDDEN)
            .map(|column| {
                let mut sum = 0.0_f32;
                for route in 0..TOP_K {
                    sum += bf16_to_f32(routed[route * HIDDEN + column]);
                }
                f32_to_bf16(sum)
            })
            .collect::<Vec<_>>();
        let actual = session.download_bf16(&case.final_output, HIDDEN)?;
        exact_receipt(
            session,
            "mimo_weighted_down_row_combine",
            "combine_once",
            &actual,
            &expected,
        )
    })
    .expect("MiMo weighted down output feeds preweighted row combine");
}

#[test]
fn grouped_moe_standard_full_path() {
    gpu_test(6 * 1024 * 1024 * 1024, |session| {
        let mut failures = replay::Failures::default();
        for (phase, tokens) in [
            (Phase::Decode, 1),
            (Phase::Decode, 8),
            (Phase::Decode, 32),
            (Phase::Prefill, 512),
            (Phase::Prefill, 2048),
        ] {
            let result = (|| {
                let case = FullCase::new(session, Profile::StdDenseMoe, phase, tokens)?;
                let mut checks = case.replay_checks(session, true, true, true, true)?;
                checks.phase(ReplayPhase::Eager, false, || case.launch(session, true))?;
                case.check(session)?;

                Ok(())
            })();
            failures.check(&format!("full path {phase:?} t{tokens}"), result);
        }
        failures.finish()
    })
    .expect("standard grouped MoE full path");
}

#[test]
fn grouped_moe_graph_determinism_pdl() {
    gpu_test(6 * 1024 * 1024 * 1024, |session| {
        // Bucket 1 and the ring buckets up to eight tokens.
        let mut failures = replay::Failures::default();
        for tokens in [1, 2, 4, 8] {
            let result = (|| {
                let name = format!("mimo_decode_t{tokens}");
                let case = FullCase::new(
                    session,
                    model_mimo26_contract::sm120::config::PROFILE,
                    Phase::Decode,
                    tokens,
                )?;
                let mut checks = case.replay_checks(session, true, true, true, true)?;
                checks.phase(ReplayPhase::Eager, false, || case.launch(session, true))?;
                let eager = session.download_bf16(&case.final_output, tokens * HIDDEN)?;
                checks.phase(ReplayPhase::Relaunch, true, || case.launch(session, true))?;
                let repeated = session.download_bf16(&case.final_output, tokens * HIDDEN)?;
                exact_receipt(session, &name, "determinism", &repeated, &eager)?;
                let graph = session.capture(|_| case.launch(session, true))?;
                checks.phase(ReplayPhase::Graph, false, || {
                    graph.launch(&session.stream).map_err(err)
                })?;
                let replay = session.download_bf16(&case.final_output, tokens * HIDDEN)?;
                exact_receipt(session, &name, "graph_replay", &replay, &eager)?;
                checks.phase(ReplayPhase::Relaunch, false, || case.launch(session, false))?;
                let non_pdl = session.download_bf16(&case.final_output, tokens * HIDDEN)?;
                exact_receipt(
                    session,
                    &name,
                    "pdl_after_router_permute",
                    &replay,
                    &non_pdl,
                )?;
                checks.phase(ReplayPhase::Pdl, true, || case.launch(session, true))?;
                let mut receipt = BTreeMap::from([("case".into(), json!(name))]);
                checks.record(&mut receipt);
                session.receipt("moe_replay", receipt)?;
                Ok(())
            })();
            failures.check(&format!("graph determinism t{tokens}"), result);
        }
        failures.finish()
    })
    .expect("MoE graph, determinism and PDL");
}

fn peak_rates(session: &Session) -> (f64, f64, f64, String) {
    let root = kernel_lab::receipt_root().join("probes");
    if let Ok(entries) = fs::read_dir(root) {
        let mut paths = entries
            .flatten()
            .map(|entry| entry.path())
            .collect::<Vec<_>>();
        paths.sort();
        for path in paths.into_iter().rev() {
            let Ok(bytes) = fs::read(&path) else { continue };
            let Ok(value) = serde_json::from_slice::<Value>(&bytes) else {
                continue;
            };
            if value["uuid"] != session.metadata["uuid"] {
                continue;
            }
            if let (Some(dram), Some(mma), Some(mxf8)) = (
                value["dram_read"]["gb_per_second"].as_f64(),
                value["mma_tflops"]["bf16"]["tflops"].as_f64(),
                value["mma_tflops"]["mxf8_e2m1"]["tflops"].as_f64(),
            ) && dram > 0.0
                && mma > 0.0
                && mxf8 > 0.0
            {
                return (dram, mma, mxf8, path.display().to_string());
            }
        }
    }
    // Until the device probe runs, these are explicit planning ceilings only.
    (
        1800.0,
        400.0,
        800.0,
        "fallback: 1.8 TB/s DRAM, 400 TFLOP/s BF16, 800 TFLOP/s MXFP8 planning ceilings".into(),
    )
}

#[test]
fn grouped_moe_mimo_47_layer_graph() {
    if std::env::var("CHOPINFER_MOE_47_LAYER_BENCH").as_deref() != Ok("1") {
        kernel_lab::skip("47-layer MoE graph: set CHOPINFER_MOE_47_LAYER_BENCH=1 on GPU 1")
            .unwrap();
        return;
    }
    let mut failures = replay::Failures::default();
    for tokens in [1, 8, 16, 32] {
        if std::env::var("CHOPINFER_MOE_47_TOKENS")
            .ok()
            .is_some_and(|selected| selected != tokens.to_string())
        {
            continue;
        }
        let result = gpu_test(6 * 1024 * 1024 * 1024, |session| {
            let layers = (0..47)
                .map(|layer| FullCase::sparse_graph_layer(session, tokens, layer))
                .collect::<LabResult<Vec<_>>>()?;
            let active_weight_bytes = layers.iter().map(|layer| layer.weight_bytes).sum::<usize>();
            let fused = match std::env::var("CHOPINFER_MOE_47_FUSED").as_deref() {
                Ok("1") => true,
                Ok("0") => false,
                _ => model_mimo26_contract::sm120::moe::prefer_fused_down_combine(tokens as u32),
            };
            let reference_graph = session.capture(|_| {
                let mut input = &layers[0].input;
                for layer in &layers {
                    layer.launch_with_input(session, input, true)?;
                    input = &layer.final_output;
                }
                Ok(())
            })?;
            let mut reference_checks = layers.iter().map(|layer| layer.replay_checks(session, true, true, true, true)).collect::<LabResult<Vec<_>>>()?;
            replay::phase_all(&mut reference_checks, ReplayPhase::Eager, false, || reference_graph.launch(&session.stream).map_err(err))?;
            let reference = session.download_bf16(&layers[46].final_output, tokens * HIDDEN)?;
            let graph = if fused {
                session.capture(|_| {
                    let mut input = &layers[0].input;
                    for layer in &layers {
                        layer.launch_with_input_variant(session, input, true, true)?;
                        input = &layer.final_output;
                    }
                    Ok(())
                })?
            } else {
                reference_graph
            };
            let mut checks = layers.iter().map(|layer| layer.replay_checks(session, true, true, !fused, true)).collect::<LabResult<Vec<_>>>()?;
            replay::phase_all(&mut checks, ReplayPhase::Eager, false, || graph.launch(&session.stream).map_err(err))?;
            let first = session.download_bf16(&layers[46].final_output, tokens * HIDDEN)?;
            if fused && first != reference {
                return Err(format!(
                    "47-layer fused down combine differs at token {tokens}, element {:?}: actual {:?}, reference {:?}",
                    first.iter().zip(&reference).position(|(a, b)| a != b),
                    first.iter().zip(&reference).find(|(a, b)| a != b).map(|(a, _)| a),
                    first.iter().zip(&reference).find(|(a, b)| a != b).map(|(_, b)| b),
                ));
            }
            replay::phase_all(&mut checks, ReplayPhase::Graph, false, || graph.launch(&session.stream).map_err(err))?;
            let second = session.download_bf16(&layers[46].final_output, tokens * HIDDEN)?;
            if first != second {
                return Err("47-layer MoE graph replay differs bitwise".into());
            }
            let timing = time(session, 1, 3, |stream| graph.launch(stream).map_err(err))?;
            let mut stage_us = BTreeMap::new();
            let stages: &[&str] = if layers[0].phase == Phase::Prefill {
                &["route", "gate_up", "down", "combine"]
            } else if fused && layers[0].use_fused_router() {
                &["router_route", "gate_up", "down_combine"]
            } else if fused {
                &["router_gemm", "route", "gate_up", "down_combine"]
            } else if layers[0].use_fused_router() {
                &["router_route", "gate_up", "down", "combine"]
            } else {
                &["router_gemm", "route", "gate_up", "down", "combine"]
            };
            for &stage in stages {
                let stage_graph = session.capture(|_| {
                    for layer in &layers {
                        match stage {
                            "gate_up" => layer.launch_projection(session, Projection::GateUp)?,
                            "down" => layer.launch_projection(session, Projection::Down)?,
                            "down_combine" => layer.launch_fused_down_combine(session, true)?,
                            _ => layer.launch_row_stage(session, stage)?,
                        }
                    }
                    Ok(())
                })?;
                let stage_timing = time(session, 1, 3, |stream| {
                    stage_graph.launch(stream).map_err(err)
                })?;
                stage_us.insert(
                    stage.to_string(),
                    stage_timing.median_ms * 1000.0 / layers.len() as f32,
                );
            }
            session.receipt(
                "bench",
                BTreeMap::from([
                    ("engine".into(), json!("grouped_moe")),
                    ("profile".into(), json!("mimo-v26-flash")),
                    ("path".into(), json!(if fused { "47_layer_pdl_graph_fused_down_combine" } else { "47_layer_pdl_graph" })),
                    ("tokens".into(), json!(tokens)),
                    ("layers".into(), json!(layers.len())),
                    ("active_experts_per_layer".into(), json!(TOP_K)),
                    ("distinct_weight_bytes".into(), json!(active_weight_bytes)),
                    ("median_us".into(), json!(timing.median_ms * 1000.0)),
                    (
                        "per_layer_us".into(),
                        json!(timing.median_ms * 1000.0 / layers.len() as f32),
                    ),
                    ("bitwise_replay".into(), json!(true)),
                    ("stage_graph_us_per_layer".into(), json!(stage_us)),
                    ("rounds".into(), json!(timing.rounds)),
                ]),
            )?;
            Ok(())
        })
        .map(|_| ());
        failures.check(&format!("layered graph t{tokens}"), result);
    }
    failures.finish().expect("47-layer MiMo MoE PDL graph");
}

/// Layers in the broad-route decode graph. Every layer stores all 256
/// experts (855 MB); `CHOPINFER_MOE_GRAPH_LAYERS` shrinks the graph when the
/// device is shared.
fn graph_layers() -> usize {
    std::env::var("CHOPINFER_MOE_GRAPH_LAYERS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(47)
}

fn check_random_route_oracle(session: &Session, case: &FullCase, layer: usize) -> LabResult<()> {
    let rows = case.tokens * TOP_K;
    let ids = download_i32(session, &case.ids, rows)?;
    let gate = session.download_bf16(&case.gate_output, rows * 512)?;
    let down = session.download_bf16(&case.down_output, rows * HIDDEN)?;
    let combined = session.download_bf16(&case.final_output, case.tokens * HIDDEN)?;
    let scale = decode_ue8m0_f64(SPARSE_SCALE);
    let mut high_experts = 0;
    for token in 0..case.tokens {
        let input_sum = case.input_host[token * HIDDEN..(token + 1) * HIDDEN]
            .iter()
            .map(|&value| f64::from(bf16_to_f32(value)))
            .sum::<f64>();
        for route in 0..TOP_K {
            let row = token * TOP_K + route;
            let expert = ids[row] as usize;
            high_experts += usize::from(expert >= TOP_K);
            let gate_code = ((layer + expert * 3) % 7 + 1) as u8;
            let gate_weight = SmallFloat::E2M1.decode_f64(gate_code) * scale;
            let product = input_sum * gate_weight;
            let expected_gate = product / (1.0 + (-product).exp()) * product;
            let expected_gate_bits = f32_to_bf16(expected_gate as f32);
            for column in [0, 511] {
                let actual = gate[row * 512 + column];
                if actual.abs_diff(expected_gate_bits) > 8 {
                    return Err(format!(
                        "random route gate oracle row {row} expert {expert} column {column}: actual {actual:#06x}, expected {expected_gate_bits:#06x}"
                    ));
                }
            }
            let down_code = ((layer + 47 + expert * 3) % 7 + 1) as u8;
            let down_weight = SmallFloat::E2M1.decode_f64(down_code) * scale;
            let gate_sum = gate[row * 512..(row + 1) * 512]
                .iter()
                .map(|&value| f64::from(bf16_to_f32(value)))
                .sum::<f64>();
            let expected_down_bits = f32_to_bf16((gate_sum * down_weight / 8.0) as f32);
            for column in [0, HIDDEN - 1] {
                let actual = down[row * HIDDEN + column];
                if actual.abs_diff(expected_down_bits) > 8 {
                    return Err(format!(
                        "random route down oracle row {row} expert {expert} column {column}: actual {actual:#06x}, expected {expected_down_bits:#06x}"
                    ));
                }
            }
        }
        for column in [0, HIDDEN - 1] {
            let mut sum = 0.0_f32;
            for route in 0..TOP_K {
                sum += bf16_to_f32(down[(token * TOP_K + route) * HIDDEN + column]);
            }
            let expected = f32_to_bf16(sum);
            let actual = combined[token * HIDDEN + column];
            if actual != expected {
                return Err(format!(
                    "random route combine oracle token {token} column {column}: actual {actual:#06x}, expected {expected:#06x}"
                ));
            }
        }
    }
    if high_experts == 0 {
        return Err("random route oracle did not exercise high expert IDs".into());
    }
    Ok(())
}

#[test]
fn grouped_moe_mimo_random_route_oracle() {
    gpu_test(6 * 1024 * 1024 * 1024, |session| {
        for (tokens, distinct_experts) in [(1, 8), (2, 16), (4, 31), (8, 56), (16, 100), (32, 160)]
        {
            let case = FullCase::random_route_graph_layer(session, tokens, 0, distinct_experts)?;
            case.launch_projection(session, Projection::GateUp)?;
            case.launch_projection(session, Projection::Down)?;
            case.launch_row_stage(session, "combine")?;
            check_random_route_oracle(session, &case, 0)?;
        }
        Ok(())
    })
    .expect("MiMo decode random-route high-expert oracle");
}

#[test]
fn grouped_moe_mimo_random_route_47_layer_graph() {
    if std::env::var("CHOPINFER_MOE_RANDOM_ROUTE_BENCH").as_deref() != Ok("1") {
        kernel_lab::skip("random-route MoE graph: set CHOPINFER_MOE_RANDOM_ROUTE_BENCH=1").unwrap();
        return;
    }
    let mut failures = replay::Failures::default();
    for (tokens, distinct_experts) in [(2, 16), (4, 31), (8, 56), (16, 100), (32, 160)] {
        if std::env::var("CHOPINFER_MOE_47_TOKENS")
            .ok()
            .is_some_and(|selected| selected != tokens.to_string())
        {
            continue;
        }
        let result = gpu_test(6 * 1024 * 1024 * 1024, |session| {
            let layers = (0..graph_layers())
                .map(|layer| {
                    FullCase::random_route_graph_layer(session, tokens, layer, distinct_experts)
                })
                .collect::<LabResult<Vec<_>>>()?;
            let graph = session.capture(|_| {
                for layer in &layers {
                    layer.launch_projection(session, Projection::GateUp)?;
                    layer.launch_projection(session, Projection::Down)?;
                    layer.launch_row_stage(session, "combine")?;
                }
                Ok(())
            })?;
            let mut checks = layers
                .iter()
                .map(|layer| layer.replay_checks(session, false, true, true, true))
                .collect::<LabResult<Vec<_>>>()?;
            replay::phase_all(&mut checks, ReplayPhase::Eager, false, || {
                graph.launch(&session.stream).map_err(err)
            })?;
            check_random_route_oracle(session, &layers[0], 0)?;
            let first =
                session.download_bf16(&layers[layers.len() - 1].final_output, tokens * HIDDEN)?;
            replay::phase_all(&mut checks, ReplayPhase::Graph, false, || {
                graph.launch(&session.stream).map_err(err)
            })?;
            let second =
                session.download_bf16(&layers[layers.len() - 1].final_output, tokens * HIDDEN)?;
            if first != second {
                return Err("random-route MoE graph replay differs bitwise".into());
            }
            let timing = time(session, 1, 3, |stream| graph.launch(stream).map_err(err))?;
            let mut stage_us = BTreeMap::new();
            for stage in ["gate_up", "down", "combine"] {
                let stage_graph = session.capture(|_| {
                    for layer in &layers {
                        match stage {
                            "gate_up" => layer.launch_projection(session, Projection::GateUp)?,
                            "down" => layer.launch_projection(session, Projection::Down)?,
                            _ => layer.launch_row_stage(session, stage)?,
                        }
                    }
                    Ok(())
                })?;
                let stage_timing = time(session, 1, 3, |stream| {
                    stage_graph.launch(stream).map_err(err)
                })?;
                stage_us.insert(
                    stage.to_string(),
                    stage_timing.median_ms * 1000.0 / layers.len() as f32,
                );
            }
            let gate_bytes = distinct_experts
                * (layers[0].gate.query.n * layers[0].gate.query.k * 17 / 32) as usize;
            let down_bytes = distinct_experts
                * (layers[0].down.query.n * layers[0].down.query.k * 17 / 32) as usize;
            let per_layer_us = timing.median_ms * 1000.0 / layers.len() as f32;
            let dram_peak_bytes_per_us = 1_330_000.0_f32;
            session.receipt(
                "bench",
                BTreeMap::from([
                    ("engine".into(), json!("grouped_moe")),
                    ("profile".into(), json!("mimo-v26-flash")),
                    ("path".into(), json!("random_route_47_layer_pdl_graph")),
                    ("tokens".into(), json!(tokens)),
                    ("layers".into(), json!(layers.len())),
                    ("distinct_experts_per_layer".into(), json!(distinct_experts)),
                    ("gate_weight_bytes_per_layer".into(), json!(gate_bytes)),
                    ("down_weight_bytes_per_layer".into(), json!(down_bytes)),
                    ("per_layer_us".into(), json!(per_layer_us)),
                    ("stage_graph_us_per_layer".into(), json!(stage_us)),
                    (
                        "dram_roofline_fraction".into(),
                        json!(
                            (gate_bytes + down_bytes) as f32
                                / per_layer_us
                                / dram_peak_bytes_per_us
                        ),
                    ),
                    ("bitwise_replay".into(), json!(true)),
                ]),
            )?;
            Ok(())
        })
        .map(|_| ());
        failures.check(&format!("layered graph t{tokens}"), result);
    }
    failures.finish().expect("random-route 47-layer MoE graph");
}

/// Replays captured MiMo routes (`fixtures/mimo26/routing-locality.json`,
/// written by `tools/moe_routing_locality.py`) through the decode MoE path:
/// each graph layer runs the route kernel on logits that select that layer's
/// captured experts, then gate/up, down and combine, with distinct weights
/// for all 256 experts per layer. Bytes are the FP4 codes and compact scales
/// of each layer's distinct experts.
#[test]
fn grouped_moe_mimo_real_route_47_layer_graph() {
    if std::env::var("CHOPINFER_MOE_REAL_ROUTE_BENCH").as_deref() != Ok("1") {
        kernel_lab::skip("real-route MoE graph: set CHOPINFER_MOE_REAL_ROUTE_BENCH=1").unwrap();
        return;
    }
    let fixture: Value = serde_json::from_slice(
        &fs::read(
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../../fixtures/mimo26/routing-locality.json"),
        )
        .expect("routing locality fixture"),
    )
    .expect("routing locality JSON");
    let shapes = fixture["shapes"].as_array().expect("shapes");
    let samples_limit = std::env::var("CHOPINFER_MOE_REAL_ROUTE_SAMPLES")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(usize::MAX);
    let mut failures = replay::Failures::default();
    for tokens in [8, 16, 32, 64, 128, 256] {
        if std::env::var("CHOPINFER_MOE_47_TOKENS")
            .ok()
            .is_some_and(|selected| selected != tokens.to_string())
        {
            continue;
        }
        let batches = shapes
            .iter()
            .filter(|shape| shape["tokens"].as_u64() == Some(tokens as u64))
            .flat_map(|shape| {
                let label = format!(
                    "{}x{}",
                    shape["sequences"].as_u64().unwrap_or(0),
                    shape["rows_per_sequence"].as_u64().unwrap_or(0)
                );
                shape["bench_samples"]
                    .as_array()
                    .into_iter()
                    .flatten()
                    .take(samples_limit)
                    .enumerate()
                    .map(move |(index, sample)| (label.clone(), index, sample.clone()))
            })
            .collect::<Vec<_>>();
        if batches.is_empty() {
            continue;
        }
        let result = gpu_test(6 * 1024 * 1024 * 1024, |session| {
            let (dram_peak, _, _, source) = peak_rates(session);
            let layer_count = graph_layers();
            let routes_of = |sample: &Value, layer: usize| -> LabResult<Vec<u8>> {
                let hex = sample["layers"][(layer + 1).to_string()]
                    .as_str()
                    .ok_or("missing captured layer")?;
                (0..hex.len())
                    .step_by(2)
                    .map(|at| u8::from_str_radix(&hex[at..at + 2], 16).map_err(|e| e.to_string()))
                    .collect()
            };
            let (_, _, first) = &batches[0];
            let mut layers = (0..layer_count)
                .map(|layer| {
                    FullCase::real_route_graph_layer(session, tokens, layer, &routes_of(first, layer)?)
                })
                .collect::<LabResult<Vec<_>>>()?;
            let expert_bytes = |case: &FullCase| {
                (case.gate.query.n * case.gate.query.k * 17 / 32
                    + case.down.query.n * case.down.query.k * 17 / 32) as usize
            };
            for (label, index, sample) in &batches {
                let mut distinct = Vec::new();
                let mut busiest = Vec::new();
                for (layer, case) in layers.iter_mut().enumerate() {
                    let ids = routes_of(sample, layer)?;
                    case.install_real_routes(session, &ids)?;
                    let mut counts = BTreeMap::new();
                    for &id in &ids {
                        *counts.entry(id).or_insert(0_usize) += 1;
                    }
                    distinct.push(counts.len());
                    busiest.push(counts.values().copied().max().unwrap_or(0));
                }
                let graph = session.capture(|_| {
                    for layer in &layers {
                        layer.launch(session, true)?;
                    }
                    Ok(())
                })?;
                let mut checks = layers.iter().map(|layer| layer.replay_checks(session, true, true, true, true)).collect::<LabResult<Vec<_>>>()?;
            replay::phase_all(&mut checks, ReplayPhase::Eager, false, || graph.launch(&session.stream).map_err(err))?;
                check_random_route_oracle(session, &layers[0], 0)?;
                let first =
                    session.download_bf16(&layers[layer_count - 1].final_output, tokens * HIDDEN)?;
                replay::phase_all(&mut checks, ReplayPhase::Graph, false, || graph.launch(&session.stream).map_err(err))?;
                if first
                    != session.download_bf16(&layers[layer_count - 1].final_output, tokens * HIDDEN)?
                {
                    return Err("real-route MoE graph replay differs bitwise".into());
                }
                let timing = time(session, 1, 3, |stream| graph.launch(stream).map_err(err))?;
                let mut stage_us = BTreeMap::new();
                for stage in ["route", "gate_up", "down", "combine", "without_route"] {
                    let stage_graph = session.capture(|_| {
                        for layer in &layers {
                            match stage {
                                // The chain after routing, on the routes the
                                // full graph left in place.
                                "without_route" => {
                                    layer.launch_projection(session, Projection::GateUp)?;
                                    layer.launch_projection(session, Projection::Down)?;
                                    layer.launch_row_stage(session, "combine")?;
                                }
                                "gate_up" => layer.launch_projection(session, Projection::GateUp)?,
                                "down" => layer.launch_projection(session, Projection::Down)?,
                                _ => layer.launch_row_stage(session, stage)?,
                            }
                        }
                        Ok(())
                    })?;
                    let stage_timing =
                        time(session, 1, 3, |stream| stage_graph.launch(stream).map_err(err))?;
                    stage_us.insert(
                        stage.to_string(),
                        stage_timing.median_ms * 1000.0 / layer_count as f32,
                    );
                }
                let bytes_per_layer = distinct.iter().sum::<usize>() as f64 * expert_bytes(&layers[0]) as f64
                    / layer_count as f64;
                let per_layer_us = timing.median_ms * 1000.0 / layer_count as f32;
                let moe_us = stage_us["gate_up"] + stage_us["down"];
                let roofline = |us: f32| bytes_per_layer / f64::from(us) / 1e3 / dram_peak;
                let mean = |values: &[usize]| values.iter().sum::<usize>() as f64 / values.len() as f64;
                let path = session.receipt(
                    "bench",
                    BTreeMap::from([
                        ("engine".into(), json!("grouped_moe")),
                        ("profile".into(), json!("mimo-v26-flash")),
                        ("path".into(), json!("real_route_47_layer_pdl_graph")),
                        ("tokens".into(), json!(tokens)),
                        ("shape".into(), json!(label)),
                        ("sample".into(), json!(index)),
                        ("gate_up_entry".into(), json!(layers[0].gate.symbol)),
                        ("down_entry".into(), json!(layers[0].down.symbol)),
                        ("streams".into(), sample["streams"].clone()),
                        ("layers".into(), json!(layer_count)),
                        ("distinct_experts_per_layer".into(), json!(distinct)),
                        ("distinct_experts_mean".into(), json!(mean(&distinct))),
                        ("busiest_expert_rows_mean".into(), json!(mean(&busiest))),
                        ("weight_bytes_per_layer".into(), json!(bytes_per_layer)),
                        ("per_layer_us".into(), json!(per_layer_us)),
                        ("stage_graph_us_per_layer".into(), json!(stage_us)),
                        ("full_dram_roofline_fraction".into(), json!(roofline(per_layer_us))),
                        ("gemm_dram_roofline_fraction".into(), json!(roofline(moe_us))),
                        ("peak_dram_read_gb_per_second".into(), json!(dram_peak)),
                        ("peak_source".into(), json!(source)),
                        ("bitwise_replay".into(), json!(true)),
                    ]),
                )?;
                eprintln!(
                    "real route t{tokens} {label} #{index} {}: distinct {:.1}, busiest {:.1}, full {per_layer_us:.2} us ({:.1}%), route {:.2}, gate/up {:.2}, down {:.2}, combine {:.2}, without route {:.2} ({:.1}% gemm), {}",
                    layers[0].gate.symbol,
                    mean(&distinct),
                    mean(&busiest),
                    roofline(per_layer_us) * 100.0,
                    stage_us["route"],
                    stage_us["gate_up"],
                    stage_us["down"],
                    stage_us["combine"],
                    stage_us["without_route"],
                    roofline(moe_us) * 100.0,
                    path.display()
                );
            }
            Ok(())
        })
        .map(|_| ());
        failures.check(&format!("layered graph t{tokens}"), result);
    }
    failures.finish().expect("real-route 47-layer MoE graph");
}

#[test]
fn grouped_moe_benchmarks() {
    if std::env::var("CHOPINFER_KERNEL_BENCH").as_deref() != Ok("1") {
        kernel_lab::skip("MoE benchmarks: set CHOPINFER_KERNEL_BENCH=1 on idle GPUs").unwrap();
        return;
    }
    gpu_test(6 * 1024 * 1024 * 1024, |session| {
        let (dram_peak, mma_peak, _, source) = peak_rates(session);
        for (profile, phase, tokens) in [
            (model_mimo26_contract::sm120::config::PROFILE, Phase::Decode, 1),
            (model_mimo26_contract::sm120::config::PROFILE, Phase::Decode, 2),
            (model_mimo26_contract::sm120::config::PROFILE, Phase::Decode, 4),
            (model_mimo26_contract::sm120::config::PROFILE, Phase::Decode, 8),
            (model_mimo26_contract::sm120::config::PROFILE, Phase::Decode, 16),
            (model_mimo26_contract::sm120::config::PROFILE, Phase::Prefill, 32),
            (model_mimo26_contract::sm120::config::PROFILE, Phase::Prefill, 512),
            (model_mimo26_contract::sm120::config::PROFILE, Phase::Prefill, 2048),
            (model_mimo26_contract::sm120::config::PROFILE, Phase::Prefill, 8192),
            (Profile::StdDenseMoe, Phase::Decode, 1),
            (Profile::StdDenseMoe, Phase::Decode, 32),
            (Profile::StdDenseMoe, Phase::Prefill, 512),
        ] {
            let case_name = format!(
                "{}_{}_t{}",
                profile.as_str(),
                if phase == Phase::Decode { "decode" } else { "prefill" },
                tokens
            );
            if std::env::var("CHOPINFER_MOE_BENCH_FILTER")
                .is_ok_and(|filter| !case_name.contains(&filter))
            {
                continue;
            }
            let case = FullCase::new(session, profile, phase, tokens)?;
            let graph = session.capture(|_| case.launch(session, true))?;
            let timing = kernel_lab::time_named(session, &format!("{case_name}::full"), 2, 3, |stream| graph.launch(stream).map_err(err))?;
            case.launch(session, true)?;
            session.stream.synchronize().map_err(err)?;
            let gate_graph = session.capture(|_| case.launch_projection(session, Projection::GateUp))?;
            let down_graph = session.capture(|_| case.launch_projection(session, Projection::Down))?;
            let gate_timing = kernel_lab::time_named(session, &format!("{case_name}::gate_up"), 2, 3, |stream| gate_graph.launch(stream).map_err(err))?;
            let down_timing = kernel_lab::time_named(session, &format!("{case_name}::down"), 2, 3, |stream| down_graph.launch(stream).map_err(err))?;
            let mut row_stage_us = BTreeMap::new();
            if phase == Phase::Decode {
                let stages: &[&str] = if case.router_gemm.is_some() {
                    &["router_gemm", "route", "combine"]
                } else {
                    &["router", "count", "scan", "scatter", "combine"]
                };
                for &stage in stages {
                    let stage_graph = session.capture(|_| case.launch_row_stage(session, stage))?;
                    let stage_timing = kernel_lab::time_named(session, &format!("{case_name}::{stage}"), 2, 3, |stream| stage_graph.launch(stream).map_err(err))?;
                    row_stage_us.insert(stage.to_string(), stage_timing.median_ms * 1000.0);
                }
            }
            let gb_s = case.weight_bytes as f64 / f64::from(timing.median_ms) / 1_000_000.0;
            let flops = 2_u64
                * (tokens * TOP_K) as u64
                * (u64::from(case.gate.query.n) * u64::from(case.gate.query.k)
                    + u64::from(case.down.query.n) * u64::from(case.down.query.k));
            let tflops = flops as f64 / f64::from(timing.median_ms) / 1_000_000_000.0;
            let memory_bound = phase == Phase::Decode;
            let mut receipt = BTreeMap::new();
            receipt.insert("engine".into(), json!("grouped_moe"));
            receipt.insert("profile".into(), json!(profile.as_str()));
            receipt.insert(
                "phase".into(),
                json!(if phase == Phase::Decode {
                    "decode"
                } else {
                    "prefill"
                }),
            );
            receipt.insert("tokens".into(), json!(tokens));
            receipt.insert(
                "path".into(),
                json!(if case.router_gemm.is_some() {
                    "bf16_router_gemm_fused_route_a16_gateup_down_combine"
                } else {
                    "top8_permute_a16_gateup_down_combine"
                }),
            );
            receipt.insert("median_us".into(), json!(timing.median_ms * 1000.0));
            receipt.insert("gate_up_us".into(), json!(gate_timing.median_ms * 1000.0));
            receipt.insert("down_us".into(), json!(down_timing.median_ms * 1000.0));
            receipt.insert("row_stage_us".into(), json!(row_stage_us));
            receipt.insert(
                "gemm_fraction_of_total".into(),
                json!((gate_timing.median_ms + down_timing.median_ms) / timing.median_ms),
            );
            receipt.insert(
                "bottleneck_inference".into(),
                json!(if case.router_gemm.is_some() {
                    "fused decode route builds a compact expert list; expert GEMM streams fragment-major MXFP4 weights"
                } else if memory_bound {
                    "isolated stage timings identify the decode bottleneck"
                } else {
                    "prefill stages bulk-copy fragment-major MiMo weights or TMA-load standard weights"
                }),
            );
            receipt.insert("min_us".into(), json!(timing.min_ms * 1000.0));
            receipt.insert("rounds".into(), json!(timing.rounds));
            receipt.insert("active_weight_bytes".into(), json!(case.weight_bytes));
            receipt.insert("achieved_gb_per_second".into(), json!(gb_s));
            receipt.insert("logical_gemm_flops".into(), json!(flops));
            receipt.insert("achieved_tflops".into(), json!(tflops));
            receipt.insert("peak_gb_per_second".into(), json!(dram_peak));
            receipt.insert("peak_bf16_tflops".into(), json!(mma_peak));
            receipt.insert(
                "primary_bound".into(),
                json!(if memory_bound { "dram" } else { "bf16_mma" }),
            );
            receipt.insert(
                "percent_of_measured_peak".into(),
                json!(
                    if memory_bound {
                        gb_s / dram_peak
                    } else {
                        tflops / mma_peak
                    } * 100.0
                ),
            );
            receipt.insert("peak_source".into(), json!(source));
            session.receipt("bench", receipt)?;
        }
        Ok(())
    })
    .expect("MoE benchmark suite");
}

#[test]
fn grouped_moe_a8_prefill_benchmarks() {
    if std::env::var("CHOPINFER_KERNEL_BENCH").as_deref() != Ok("1") {
        kernel_lab::skip("MoE A8 benchmarks: set CHOPINFER_KERNEL_BENCH=1 on idle GPUs").unwrap();
        return;
    }
    gpu_test(6 * 1024 * 1024 * 1024, |session| {
        let (_, _, peak_mxf8, source) = peak_rates(session);
        for tokens in [2048_usize, 8192] {
            for projection in [Projection::GateUp, Projection::Down] {
                let v = variant(
                    model_mimo26_contract::sm120::config::PROFILE,
                    Phase::Prefill,
                    projection,
                    InputPath::A8Pregather,
                    WeightFormat::Mxfp4,
                    tokens,
                )?;
                let routes = tokens * TOP_K;
                let experts = v.query.grouped.ok_or("missing grouped shape")?.experts as usize;
                let k = v.query.k as usize;
                let output_n = if projection == Projection::GateUp {
                    v.query.n as usize / 2
                } else {
                    v.query.n as usize
                };
                let bank = WeightBank::new(&v, 0xf280)?;
                let weights = bank.upload(session)?;
                let activation = session.allocate(routes * k)?;
                activation.zero_at_async(0, routes * k, &session.stream).map_err(err)?;
                let activation_scales = upload_bytes(session, &vec![127; routes * (k / 32)])?;
                let one = session.upload_f32(&[1.0])?;
                let dummy = upload_bytes(session, &[0])?;
                let offsets = upload_bytes(
                    session,
                    &i32_bytes(&(0..=experts).map(|e| (e.min(TOP_K) * tokens) as i32).collect::<Vec<_>>()),
                )?;
                let sorted = upload_bytes(
                    session,
                    &i32_bytes(&(0..TOP_K).flat_map(|e| (0..tokens).map(move |t| (t * TOP_K + e) as i32)).collect::<Vec<_>>()),
                )?;
                let padded = upload_bytes(session, &(routes as i32).to_ne_bytes())?;
                let live = upload_bytes(session, &(tokens as i32).to_ne_bytes())?;
                let active = upload_bytes(
                    session,
                    &i32_bytes(&[TOP_K as i32, 0, 1, 2, 3, 4, 5, 6, 7]),
                )?;
                let topk_weights = session.upload_f32(&vec![0.125; routes])?;
                let output = session.allocate(routes * output_n * 2)?;
                let counts = RuntimeCounts {
                    m: tokens as u32,
                    batch: 0,
                    context: 0,
                    routed_rows: routes as u32,
                    split_k: v.knobs.split_k_capacity,
                };
                let partials = session.allocate(v.plan(counts).map_err(err)?.workspace_bytes.max(1) as usize)?;
                let functions = load_moe(session, &v, counts)?;
                let operands = MoeOperands {
                    activation: &activation,
                    activation_scales: &activation_scales,
                    activation_f32_scales: &one,
                    weights: &weights,
                    output: &output,
                    topk_weights: &topk_weights,
                    offsets: &offsets,
                    sorted: &sorted,
                    padded: &padded,
                    live: &live,
                    active: &active,
                    quantized_output: &dummy,
                    quantized_scales: &dummy,
                    partials: &partials,
                };
                let graph = session.capture(|_| launch_moe(session, &v, &functions, counts, &operands, true))?;
                let timing = time(session, 2, 3, |stream| graph.launch(stream).map_err(err))?;
                let flops = 2_u64 * routes as u64 * u64::from(v.query.n) * k as u64;
                let tflops = flops as f64 / f64::from(timing.median_ms) / 1_000_000_000.0;
                let mut receipt = BTreeMap::new();
                receipt.insert("engine".into(), json!("grouped_moe"));
                receipt.insert("profile".into(), json!(v.profile.as_str()));
                receipt.insert("phase".into(), json!("prefill"));
                receipt.insert("tokens".into(), json!(tokens));
                receipt.insert("path".into(), json!("synthetic_a8_gemm"));
                receipt.insert("projection".into(), json!(format!("{projection:?}")));
                receipt.insert("median_us".into(), json!(timing.median_ms * 1000.0));
                receipt.insert("min_us".into(), json!(timing.min_ms * 1000.0));
                receipt.insert("rounds".into(), json!(timing.rounds));
                receipt.insert("active_weight_bytes".into(), json!(bank.logical_bytes));
                receipt.insert("logical_gemm_flops".into(), json!(flops));
                receipt.insert(
                    "weight_tma_barrier_stages".into(),
                    json!((routes / 64) * (output_n / 16) * (k / 32)),
                );
                receipt.insert(
                    "bottleneck_inference".into(),
                    json!("isolated GEMM graph timing remains far below MXFP8 peak; each CTA serializes K/32 weight TMA and barrier stages before warp MMA"),
                );
                receipt.insert("achieved_tflops".into(), json!(tflops));
                receipt.insert("peak_mxf8_tflops".into(), json!(peak_mxf8));
                receipt.insert("percent_of_measured_peak".into(), json!(100.0 * tflops / peak_mxf8));
                receipt.insert("peak_source".into(), json!(source));
                session.receipt("bench", receipt)?;
            }
        }
        Ok(())
    })
    .expect("MoE A8 prefill benchmark suite");
}

#[path = "moe_prefill.rs"]
mod prefill;
#[path = "prefill_diagnostics.rs"]
mod prefill_diagnostics;
#[path = "moe_qwen.rs"]
mod qwen38;

#[path = "moe_verify_drift.rs"]
mod verify_drift;

#[path = "moe_exact_rows.rs"]
mod exact_rows;

#[path = "moe_invariance.rs"]
pub(crate) mod invariance;

#[path = "moe_split_routes.rs"]
mod split_routes;

#[path = "prefill_boundaries.rs"]
mod prefill_boundaries;
