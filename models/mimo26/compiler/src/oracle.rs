//! MiMo text layer reference over checkpoint bytes.
//!
//! Matrix reductions accumulate in f64. HF stores the input, norm output,
//! projections, rotated Q/K, scaled V, attention output, residuals, expert
//! intermediates and final layer output in BF16. Those boundaries are rounded
//! explicitly; router logits and route weights remain F32.
//!
//! The rows of one call are processed together: projections run in parallel
//! over weight rows and each row attends to the rows before it, so a chunk
//! gives the same values as one row at a time. `run_with_partials` optionally
//! rounds each tensor-parallel rank's partial sums as Chopinfer does.

use model_mimo26_contract::geometry::V26_FLASH as GEOMETRY;

use std::collections::BTreeMap;
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use kernel_core::codec::{SmallFloat, bf16_to_f32, decode_ue8m0_f64, f32_to_bf16};
use kernel_core::contract::{DType, RouterKind};
pub use kernel_core::kv_fp8::round_e4m3;
use kernel_core::oracle::{
    CheckFailure, NormArithmetic, NormOffset, NormScale, RmsNorm, RouterSelection,
    bf16_round as rounded, check_layer_cosine, dot, e4m3_table, router,
};
use kernel_core::rotary::RotaryArithmetic;
use kernel_core::transform::ChunkedScaleGrid;

use crate::MimoResult;
use crate::catalog::{Catalog, open_source};
use crate::config::MiMoConfig;

pub mod dflash;
pub mod diagnostic;
pub mod mtp;

const HIDDEN: usize = model_mimo26_contract::tp::HIDDEN as usize;
const Q_HEADS: usize = model_mimo26_contract::tp::Q_HEADS as usize;
const QK: usize = model_mimo26_contract::tp::QK_DIM as usize;
const V: usize = model_mimo26_contract::tp::V_DIM as usize;
const ROTARY: usize = GEOMETRY.rotary() as usize;
const EXPERTS: usize = GEOMETRY.experts() as usize;
const TOP_K: usize = GEOMETRY.top_k() as usize;
/// Checkpoint data: the QKV projections use the export's TP chunks.
const CHECKPOINT_TP: usize = model_mimo26_contract::tp::QKV_CHUNKS as usize;

/// HF's BF16 RMSNorm with eps 1e-6: F32 arithmetic, the normalized value
/// and the output each rounded to BF16.
const HF_NORM: RmsNorm = RmsNorm {
    eps: GEOMETRY.eps(),
    offset: NormOffset::Plain,
    arithmetic: NormArithmetic::F32,
    scale: NormScale::MultiplyReciprocal,
    before_weight: Some(DType::Bf16),
    output: Some(DType::Bf16),
};

/// K after RoPE as `chop::rotary::rotate` computes it in F32 from BF16
/// inputs: `fma(x, cos, -(y * sin))` for the first of a pair and
/// `fma(x, cos, y * sin)` for the second, with HF table values. The fused
/// multiply-add is exact in f64 before its one F32 rounding.
fn serving_rope(input: &[f64], heads: usize, position: usize, theta: f64) -> Vec<f64> {
    let rotate = |x: f32, y: f32, (cos, sin): (f64, f64), second| {
        RotaryArithmetic::F32Fma.element(f64::from(x), f64::from(y), cos, sin, second) as f32
    };
    let mut output = input.to_vec();
    for pair in 0..ROTARY / 2 {
        let cs = kernel_core::hf_rope::cos_sin(position, pair, theta);
        for head in 0..heads {
            let base = head * QK;
            let (x, y) = (
                input[base + pair] as f32,
                input[base + pair + ROTARY / 2] as f32,
            );
            output[base + pair] = rotate(x, y, cs, false) as f64;
            output[base + pair + ROTARY / 2] = rotate(y, x, cs, true) as f64;
        }
    }
    output
}

fn e2m1_table() -> &'static [f64; 16] {
    static TABLE: OnceLock<[f64; 16]> = OnceLock::new();
    TABLE.get_or_init(|| std::array::from_fn(|code| SmallFloat::E2M1.decode_f64(code as u8)))
}

/// The checkpoint stores one rank-local Q|K|V projection per TP chunk.
fn split_qkv_chunks(
    projected: &[f64],
    chunks: usize,
    q_width: usize,
    k_width: usize,
    v_width: usize,
) -> MimoResult<(Vec<f64>, Vec<f64>, Vec<f64>)> {
    let chunk_width = q_width + k_width + v_width;
    if projected.len() != chunks * chunk_width {
        return Err("interleaved QKV projection has invalid shape".into());
    }
    let mut q = Vec::with_capacity(chunks * q_width);
    let mut k = Vec::with_capacity(chunks * k_width);
    let mut v = Vec::with_capacity(chunks * v_width);
    for chunk in projected.chunks_exact(chunk_width) {
        q.extend_from_slice(&chunk[..q_width]);
        k.extend_from_slice(&chunk[q_width..q_width + k_width]);
        v.extend_from_slice(&chunk[q_width + k_width..]);
    }
    Ok((q, k, v))
}

fn rounded_rows(rows: Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    rows.into_iter()
        .map(|row| row.into_iter().map(rounded).collect())
        .collect()
}

fn bf16_rows(bits: &[u16]) -> Vec<Vec<f64>> {
    bits.chunks_exact(HIDDEN)
        .map(|row| row.iter().map(|&v| f64::from(bf16_to_f32(v))).collect())
        .collect()
}

fn verified_read(
    root: &Path,
    catalog: &Catalog,
    verifier: Option<&std::sync::Mutex<crate::checkpoint_provenance::Verifier>>,
    name: &str,
) -> MimoResult<Vec<u8>> {
    match verifier {
        Some(verifier) => verifier
            .lock()
            .map_err(|_| "checkpoint verifier poisoned")?
            .read_tensor(name),
        None => read_tensor(root, catalog, name),
    }
}

fn read_tensor(root: &Path, catalog: &Catalog, name: &str) -> MimoResult<Vec<u8>> {
    let record = catalog.tensor(name)?;
    let file = open_source(&root.join(&record.file))?;
    let mut bytes = vec![0; record.bytes as usize];
    runtime_weights::read_exact_at(&file, &mut bytes, record.file_start).map_err(|error| {
        format!(
            "read tensor {name} from {}: {error}",
            root.join(&record.file).display()
        )
    })?;
    Ok(bytes)
}

fn u16s(bytes: &[u8]) -> Vec<u16> {
    bytes
        .chunks_exact(2)
        .map(|v| u16::from_le_bytes([v[0], v[1]]))
        .collect()
}

fn f32s(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|v| f32::from_le_bytes(v.try_into().expect("four bytes")))
        .collect()
}

fn worker_count() -> usize {
    std::thread::available_parallelism().map_or(1, usize::from)
}

/// Split `0..count` into one contiguous range per worker and concatenate the
/// results in order.
fn par_ranges<T: Send>(count: usize, work: impl Fn(Range<usize>) -> Vec<T> + Sync) -> Vec<T> {
    let workers = worker_count().min(count).max(1);
    if workers == 1 {
        return work(0..count);
    }
    let work = &work;
    std::thread::scope(|scope| {
        let handles = (0..workers)
            .map(|w| scope.spawn(move || work(count * w / workers..count * (w + 1) / workers)))
            .collect::<Vec<_>>();
        handles
            .into_iter()
            .flat_map(|handle| handle.join().expect("oracle worker"))
            .collect()
    })
}

/// Decoded weight rows for the batched projections.
trait Rows: Sync {
    fn rows(&self) -> usize;
    fn cols(&self) -> usize;
    fn decode_row(&self, row: usize, out: &mut [f64]);
}

/// Exact f64 products of every weight row with every input, as [input][row].
fn project(matrix: &impl Rows, inputs: &[Vec<f64>], parallel: bool) -> Vec<Vec<f64>> {
    let cols = matrix.cols();
    let work = |range: Range<usize>| {
        let mut weights = vec![0.0; cols];
        range
            .map(|row| {
                matrix.decode_row(row, &mut weights);
                inputs.iter().map(|x| dot(&weights, x)).collect::<Vec<_>>()
            })
            .collect::<Vec<_>>()
    };
    let by_row = if parallel {
        par_ranges(matrix.rows(), work)
    } else {
        work(0..matrix.rows())
    };
    (0..inputs.len())
        .map(|input| by_row.iter().map(|row| row[input]).collect())
        .collect()
}

/// Like `project`, with each dot product split into `parts` equal column
/// slices, as [input][row][part].
fn project_parts(
    matrix: &impl Rows,
    inputs: &[Vec<f64>],
    parallel: bool,
    parts: usize,
) -> Vec<Vec<Vec<f64>>> {
    let cols = matrix.cols();
    let width = cols / parts;
    let work = |range: Range<usize>| {
        let mut weights = vec![0.0; cols];
        range
            .map(|row| {
                matrix.decode_row(row, &mut weights);
                inputs
                    .iter()
                    .map(|x| {
                        (0..parts)
                            .map(|part| {
                                let slice = part * width..(part + 1) * width;
                                dot(&weights[slice.clone()], &x[slice])
                            })
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>()
    };
    let by_row = if parallel {
        par_ranges(matrix.rows(), work)
    } else {
        work(0..matrix.rows())
    };
    (0..inputs.len())
        .map(|input| by_row.iter().map(|row| row[input].clone()).collect())
        .collect()
}

/// A row-parallel projection's contribution to the residual: HF rounds the
/// whole product once; each of `ranks` ranks rounds its partial over an equal
/// share of the reduction, and the partials are summed in F32.
fn row_parallel(matrix: &impl Rows, inputs: &[Vec<f64>], ranks: Option<usize>) -> Vec<Vec<f64>> {
    if let Some(ranks) = ranks {
        project_parts(matrix, inputs, true, ranks)
            .into_iter()
            .map(|row| {
                row.into_iter()
                    .map(|parts| parts.into_iter().map(rounded).sum())
                    .collect()
            })
            .collect()
    } else {
        rounded_rows(project(matrix, inputs, true))
    }
}

struct Bf16Rows<'a> {
    bits: &'a [u16],
    cols: usize,
}

impl Rows for Bf16Rows<'_> {
    fn rows(&self) -> usize {
        self.bits.len() / self.cols
    }

    fn cols(&self) -> usize {
        self.cols
    }

    fn decode_row(&self, row: usize, out: &mut [f64]) {
        let source = &self.bits[row * self.cols..(row + 1) * self.cols];
        for (dst, &bits) in out.iter_mut().zip(source) {
            *dst = f64::from(bf16_to_f32(bits));
        }
    }
}

#[derive(Clone)]
pub struct Fp8Matrix {
    pub rows: usize,
    pub cols: usize,
    pub codes: Vec<u8>,
    pub scales: Vec<f32>,
    pub chunk_rows: usize,
}

impl Fp8Matrix {
    fn grid(&self) -> ChunkedScaleGrid {
        ChunkedScaleGrid {
            chunks: self.rows.div_ceil(self.chunk_rows),
            rows_per_chunk: self.chunk_rows,
            k_blocks: self.cols.div_ceil(128),
            block_rows: 128,
        }
    }

    pub fn dot(&self, row: usize, x: &[f64]) -> f64 {
        let scale_row = self.grid().scale_row(row).expect("valid FP8 weight row");
        let mut sum = 0.0;
        for (k, &value) in x.iter().enumerate() {
            let code = self.codes[row * self.cols + k];
            sum += value
                * SmallFloat::E4M3.decode_f64(code)
                * f64::from(self.scales[scale_row * self.cols.div_ceil(128) + k / 128]);
        }
        sum
    }
}

impl Rows for Fp8Matrix {
    fn rows(&self) -> usize {
        self.rows
    }

    fn cols(&self) -> usize {
        self.cols
    }

    fn decode_row(&self, row: usize, out: &mut [f64]) {
        let table = e4m3_table();
        let blocks = self.cols.div_ceil(128);
        let scale_row = self.grid().scale_row(row).expect("valid FP8 weight row");
        let codes = &self.codes[row * self.cols..(row + 1) * self.cols];
        for (k, (dst, &code)) in out.iter_mut().zip(codes).enumerate() {
            *dst = table[code as usize] * f64::from(self.scales[scale_row * blocks + k / 128]);
        }
    }
}

#[derive(Clone)]
pub struct Mxfp4Matrix {
    pub rows: usize,
    pub cols: usize,
    pub codes: Vec<u8>,
    pub scales: Vec<u8>,
}

impl Mxfp4Matrix {
    pub fn dot(&self, row: usize, x: &[f64]) -> f64 {
        let blocks = self.cols / 32;
        let mut sum = 0.0;
        for (k, &value) in x.iter().enumerate() {
            let packed = self.codes[row * self.cols / 2 + k / 2];
            let code = if k & 1 == 0 {
                packed & 0xf
            } else {
                packed >> 4
            };
            let weight = SmallFloat::E2M1.decode_f64(code)
                * decode_ue8m0_f64(self.scales[row * blocks + k / 32]);
            sum += value * weight;
        }
        sum
    }
}

impl Rows for Mxfp4Matrix {
    fn rows(&self) -> usize {
        self.rows
    }

    fn cols(&self) -> usize {
        self.cols
    }

    fn decode_row(&self, row: usize, out: &mut [f64]) {
        let table = e2m1_table();
        let blocks = self.cols / 32;
        for (k, pair) in out.chunks_exact_mut(2).enumerate() {
            let packed = self.codes[row * self.cols / 2 + k];
            let scale = decode_ue8m0_f64(self.scales[row * blocks + 2 * k / 32]);
            pair[0] = table[(packed & 0xf) as usize] * scale;
            pair[1] = table[(packed >> 4) as usize] * scale;
        }
    }
}

/// Validate a GPU top-k against the exact BF16 dot products. F32 accumulation
/// has error at most gamma_K * sum |x*w|, and sigmoid is 1/4-Lipschitz.
fn router_with_gpu_choices(
    exact_logits: &[f64],
    weights: &[u16],
    bias: &[f32],
    x: &[f64],
    gpu_ids: &[usize],
) -> MimoResult<(RouterSelection, usize)> {
    if exact_logits.len() != EXPERTS
        || weights.len() != EXPERTS * x.len()
        || bias.len() != EXPERTS
        || gpu_ids.len() != TOP_K
        || gpu_ids.iter().any(|&id| id >= EXPERTS)
        || gpu_ids
            .iter()
            .enumerate()
            .any(|(i, id)| gpu_ids[..i].contains(id))
    {
        return Err("invalid GPU router choices".into());
    }
    let bias64 = bias.iter().map(|&v| f64::from(v)).collect::<Vec<_>>();
    let ideal = router(
        exact_logits,
        Some(&bias64),
        TOP_K,
        RouterKind::SigmoidBiasSelectNormalized {
            scaling_factor: 1.0,
        },
    )?;
    let ideal_only = ideal
        .expert_ids
        .iter()
        .copied()
        .filter(|id| !gpu_ids.contains(id))
        .collect::<Vec<_>>();
    let gpu_only = gpu_ids
        .iter()
        .copied()
        .filter(|id| !ideal.expert_ids.contains(id))
        .collect::<Vec<_>>();
    if ideal_only.len() != gpu_only.len() {
        return Err("GPU router set mismatch".into());
    }
    let k = x.len() as f64;
    let u = 2.0_f64.powi(-24);
    let gamma = (k * u) / (1.0 - k * u);
    let score = |id: usize| 1.0 / (1.0 + (-exact_logits[id]).exp()) + bias64[id];
    let bound = |id: usize| {
        let abs_sum = x
            .iter()
            .enumerate()
            .map(|(col, &value)| {
                value.abs() * f64::from(bf16_to_f32(weights[id * x.len() + col])).abs()
            })
            .sum::<f64>();
        gamma * abs_sum * 0.25 + 4.0 * f64::from(f32::EPSILON)
    };
    // Every displaced winner must be close to a newly selected candidate.
    for &lost in &ideal_only {
        if !gpu_only
            .iter()
            .any(|&gained| score(lost) - score(gained) <= bound(lost) + bound(gained))
        {
            return Err(format!(
                "router top-k gap exceeds F32 GEMM bound: lost {lost}, gap {:.8}",
                gpu_only
                    .iter()
                    .map(|&gained| score(lost) - score(gained))
                    .fold(f64::INFINITY, f64::min)
            )
            .into());
        }
    }
    for &gained in &gpu_only {
        if !ideal_only
            .iter()
            .any(|&lost| score(lost) - score(gained) <= bound(lost) + bound(gained))
        {
            return Err(format!("router top-k gap exceeds F32 GEMM bound: gained {gained}").into());
        }
    }
    let unbiased = gpu_ids
        .iter()
        .map(|&id| 1.0 / (1.0 + (-exact_logits[id]).exp()))
        .collect::<Vec<_>>();
    let total = unbiased.iter().sum::<f64>() + 1e-20;
    Ok((
        RouterSelection {
            expert_ids: gpu_ids.to_vec(),
            weights: unbiased.iter().map(|v| v / total).collect(),
        },
        gpu_only.len(),
    ))
}

fn hf_rope(input: &[f64], heads: usize, position: usize, theta: f64) -> Vec<f64> {
    let mut output = serving_rope(input, heads, position, theta);
    for head in output.chunks_exact_mut(QK) {
        for value in &mut head[..ROTARY] {
            *value = rounded(*value);
        }
    }
    output
}

#[derive(Clone, Copy)]
struct AttentionGeometry {
    q_heads: usize,
    kv_heads: usize,
    qk_dim: usize,
    v_dim: usize,
    window_left: Option<usize>,
}

/// Attention of the query at `position` over history entries up to it.
fn attention_token(
    q: &[f64],
    history: &KvHistory,
    position: usize,
    geometry: AttentionGeometry,
    sink: Option<&[f64]>,
) -> Vec<f64> {
    let AttentionGeometry {
        q_heads,
        kv_heads,
        qk_dim,
        v_dim,
        window_left,
    } = geometry;
    let first = window_left.map_or(0, |left| position.saturating_sub(left));
    let mut output = vec![0.0; q_heads * v_dim];
    for head in 0..q_heads {
        let kv_head = head / (q_heads / kv_heads);
        let mut logits = Vec::with_capacity(position - first + 1);
        for token in first..=position {
            let dot = (0..qk_dim)
                .map(|d| q[head * qk_dim + d] * history.keys[token][kv_head * qk_dim + d])
                .sum::<f64>();
            let scale = f64::from(1.0_f32 / (qk_dim as f32).sqrt());
            logits.push(rounded(rounded(dot) * scale));
        }
        let sink_logit = sink.map(|v| v[head]);
        let maximum = logits
            .iter()
            .copied()
            .chain(sink_logit)
            .fold(f64::NEG_INFINITY, f64::max);
        let scores: Vec<_> = logits.iter().map(|v| (v - maximum).exp()).collect();
        let denominator =
            scores.iter().sum::<f64>() + sink_logit.map_or(0.0, |v| (v - maximum).exp());
        for d in 0..v_dim {
            let value = scores
                .iter()
                .enumerate()
                .map(|(i, &p)| {
                    rounded(p / denominator) * history.values[first + i][kv_head * v_dim + d]
                })
                .sum::<f64>();
            output[head * v_dim + d] = rounded(value);
        }
    }
    output
}

#[derive(Clone)]
struct Expert {
    gate: Mxfp4Matrix,
    up: Mxfp4Matrix,
    down: Mxfp4Matrix,
}

enum FeedForward {
    Dense {
        gate: Fp8Matrix,
        up: Fp8Matrix,
        down: Fp8Matrix,
    },
    Moe {
        router_weight: Vec<u16>,
        router_bias: Vec<f32>,
        experts: BTreeMap<usize, Expert>,
    },
}

#[derive(Default, Clone)]
pub struct KvHistory {
    pub keys: Vec<Vec<f64>>,
    pub values: Vec<Vec<f64>>,
}

/// BF16-rounded values at HF's layer boundaries, represented as f64 for checks.
pub struct LayerTrace {
    pub input_norm: Vec<f64>,
    pub qkv: Vec<f64>,
    pub q: Vec<f64>,
    pub k: Vec<f64>,
    pub v: Vec<f64>,
    pub attention: Vec<f64>,
    pub attention_residual: Vec<f64>,
    pub post_norm: Vec<f64>,
    pub router_logits: Option<Vec<f64>>,
    pub router_ids: Option<Vec<usize>>,
    pub router_weights: Option<Vec<f64>>,
    pub router_flips: usize,
    pub gate_up_routes: Option<Vec<Vec<f64>>>,
    pub moe: Vec<f64>,
    pub final_output: Vec<f64>,
}

/// Per-row feed-forward results before they are folded into traces.
struct FeedForwardRows {
    output: Vec<Vec<f64>>,
    router_logits: Option<Vec<Vec<f64>>>,
    selections: Option<Vec<(RouterSelection, usize)>>,
    gate_up_routes: Option<Vec<Vec<Vec<f64>>>>,
}

pub struct LayerOracle {
    root: PathBuf,
    catalog: Catalog,
    pub layer: usize,
    pub global: bool,
    input_norm: Vec<u16>,
    post_norm: Vec<u16>,
    qkv: Fp8Matrix,
    o: Vec<u16>,
    sink: Option<Vec<f64>>,
    mlp: FeedForward,
    verifier: Option<std::sync::Mutex<crate::checkpoint_provenance::Verifier>>,
}

impl LayerOracle {
    pub fn load(root: &Path, layer: usize) -> MimoResult<Self> {
        Self::load_inner(root, layer, None)
    }

    pub fn load_verified(
        root: &Path,
        layer: usize,
        identity: &crate::checkpoint::Identity,
    ) -> MimoResult<Self> {
        Self::load_inner(
            root,
            layer,
            Some(std::sync::Mutex::new(
                crate::checkpoint_provenance::Verifier::new(root, identity)?,
            )),
        )
    }

    fn load_inner(
        root: &Path,
        layer: usize,
        verifier: Option<std::sync::Mutex<crate::checkpoint_provenance::Verifier>>,
    ) -> MimoResult<Self> {
        let config = match &verifier {
            Some(verifier) => verifier
                .lock()
                .map_err(|_| "verifier lock poisoned")?
                .config()?,
            None => MiMoConfig::load(root)?,
        };
        if layer >= config.num_hidden_layers {
            return Err("layer index outside checkpoint".into());
        }
        let catalog = Catalog::load(root, &config)?;
        let prefix = format!("model.layers.{layer}");
        let read = |suffix: &str| {
            verified_read(
                root,
                &catalog,
                verifier.as_ref(),
                &format!("{prefix}.{suffix}"),
            )
        };
        let global = config.is_global(layer);
        let qkv_rows = model_mimo26_contract::tp::checkpoint_qkv_rows(global) as usize;
        let chunk_rows = model_mimo26_contract::tp::checkpoint_qkv_chunk_rows(global) as usize;
        let qkv = Fp8Matrix {
            rows: qkv_rows,
            cols: HIDDEN,
            codes: read("self_attn.qkv_proj.weight")?,
            scales: f32s(&read("self_attn.qkv_proj.weight_scale_inv")?),
            chunk_rows,
        };
        let mlp = if !GEOMETRY.is_moe(layer as u32) {
            let fp8 = |projection: &str, rows: usize, cols: usize| -> MimoResult<Fp8Matrix> {
                Ok(Fp8Matrix {
                    rows,
                    cols,
                    codes: read(&format!("mlp.{projection}.weight"))?,
                    scales: f32s(&read(&format!("mlp.{projection}.weight_scale_inv"))?),
                    chunk_rows: rows,
                })
            };
            FeedForward::Dense {
                gate: fp8(
                    "gate_proj",
                    model_mimo26_contract::tp::DENSE_INTERMEDIATE as usize,
                    HIDDEN,
                )?,
                up: fp8(
                    "up_proj",
                    model_mimo26_contract::tp::DENSE_INTERMEDIATE as usize,
                    HIDDEN,
                )?,
                down: fp8(
                    "down_proj",
                    HIDDEN,
                    model_mimo26_contract::tp::DENSE_INTERMEDIATE as usize,
                )?,
            }
        } else {
            FeedForward::Moe {
                router_weight: u16s(&read("mlp.gate.weight")?),
                router_bias: f32s(&read("mlp.gate.e_score_correction_bias")?),
                experts: BTreeMap::new(),
            }
        };
        let input_norm = u16s(&read("input_layernorm.weight")?);
        let post_norm = u16s(&read("post_attention_layernorm.weight")?);
        let o = u16s(&read("self_attn.o_proj.weight")?);
        let sink = (!global)
            .then(|| {
                read("self_attn.attention_sink_bias").map(|v| {
                    u16s(&v)
                        .iter()
                        .map(|&x| f64::from(bf16_to_f32(x)))
                        .collect()
                })
            })
            .transpose()?;
        Ok(Self {
            root: root.to_path_buf(),
            catalog,
            layer,
            global,
            input_norm,
            post_norm,
            qkv,
            o,
            sink,
            mlp,
            verifier,
        })
    }

    fn load_expert(
        root: &Path,
        catalog: &Catalog,
        verifier: Option<&std::sync::Mutex<crate::checkpoint_provenance::Verifier>>,
        layer: usize,
        id: usize,
    ) -> MimoResult<Expert> {
        let prefix = format!("model.layers.{layer}.mlp.experts.{id}");
        let matrix = |name: &str, rows: usize, cols: usize| -> MimoResult<Mxfp4Matrix> {
            Ok(Mxfp4Matrix {
                rows,
                cols,
                codes: verified_read(root, catalog, verifier, &format!("{prefix}.{name}.weight"))?,
                scales: verified_read(
                    root,
                    catalog,
                    verifier,
                    &format!("{prefix}.{name}.weight_scale"),
                )?,
            })
        };
        Ok(Expert {
            gate: matrix(
                "gate_proj",
                model_mimo26_contract::tp::EXPERT_INTERMEDIATE as usize,
                HIDDEN,
            )?,
            up: matrix(
                "up_proj",
                model_mimo26_contract::tp::EXPERT_INTERMEDIATE as usize,
                HIDDEN,
            )?,
            down: matrix(
                "down_proj",
                HIDDEN,
                model_mimo26_contract::tp::EXPERT_INTERMEDIATE as usize,
            )?,
        })
    }

    /// Drops the routed experts loaded so far; later rows load what they use.
    pub fn release_experts(&mut self) {
        if let FeedForward::Moe { experts, .. } = &mut self.mlp {
            experts.clear();
        }
    }

    /// Run a prefill chunk or later decode tokens. Each call appends to history.
    pub fn run(&mut self, input: &[u16], history: &mut KvHistory) -> MimoResult<Vec<u16>> {
        Ok(self.run_traced(input, history)?.0)
    }

    pub fn run_traced(
        &mut self,
        input: &[u16],
        history: &mut KvHistory,
    ) -> MimoResult<(Vec<u16>, Vec<LayerTrace>)> {
        self.run_traced_with_router_ids(input, history, None)
    }

    /// Check GPU routes against the f64 router and follow accepted near ties.
    pub fn run_traced_with_router_ids(
        &mut self,
        input: &[u16],
        history: &mut KvHistory,
        gpu_ids: Option<&[Vec<usize>]>,
    ) -> MimoResult<(Vec<u16>, Vec<LayerTrace>)> {
        self.run_with_partials(input, history, gpu_ids, None)
    }

    /// Direct f64 FP8 matmul for one checkpoint TP chunk, before BF16 rounding.
    pub fn qkv_rank_direct(&self, input_norm: &[f64], rank: usize) -> MimoResult<Vec<f64>> {
        if rank >= CHECKPOINT_TP || input_norm.len() != HIDDEN {
            return Err("QKV rank or normalized input has invalid shape".into());
        }
        let first = rank * self.qkv.chunk_rows;
        Ok((first..first + self.qkv.chunk_rows)
            .map(|row| self.qkv.dot(row, input_norm))
            .collect())
    }

    /// Post-attention norm and MLP of BF16 residual rows.
    fn feed_forward_rows(
        &mut self,
        residual: &[u16],
        gpu_ids: Option<&[Vec<usize>]>,
        tp_partials: Option<usize>,
    ) -> MimoResult<(Vec<Vec<f64>>, FeedForwardRows)> {
        let tokens = residual.len() / HIDDEN;
        let post = bf16_rows(&HF_NORM.apply_bf16(residual, &self.post_norm, HIDDEN)?);
        let feed_forward = match &mut self.mlp {
            FeedForward::Dense { gate, up, down } => {
                let gate = rounded_rows(project(gate, &post, true));
                let up = rounded_rows(project(up, &post, true));
                let activated = gate
                    .iter()
                    .zip(&up)
                    .map(|(gate, up)| {
                        gate.iter()
                            .zip(up)
                            .map(|(&g, &u)| rounded(rounded(g / (1.0 + (-g).exp())) * u))
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>();
                FeedForwardRows {
                    output: row_parallel(down, &activated, tp_partials),
                    router_logits: None,
                    selections: None,
                    gate_up_routes: None,
                }
            }
            FeedForward::Moe {
                router_weight,
                router_bias,
                experts,
            } => {
                let logits = project(
                    &Bf16Rows {
                        bits: router_weight,
                        cols: HIDDEN,
                    },
                    &post,
                    true,
                );
                let bias: Vec<_> = router_bias.iter().map(|&v| f64::from(v)).collect();
                let selections = logits
                    .iter()
                    .enumerate()
                    .map(|(index, logits)| match gpu_ids {
                        Some(ids) => router_with_gpu_choices(
                            logits,
                            router_weight,
                            router_bias,
                            &post[index],
                            &ids[index],
                        ),
                        None => Ok((
                            router(
                                logits,
                                Some(&bias),
                                TOP_K,
                                RouterKind::SigmoidBiasSelectNormalized {
                                    scaling_factor: 1.0,
                                },
                            )?,
                            0,
                        )),
                    })
                    .collect::<MimoResult<Vec<_>>>()?;
                let mut routes = BTreeMap::<usize, Vec<(usize, usize)>>::new();
                for (index, (selection, _)) in selections.iter().enumerate() {
                    for (slot, &id) in selection.expert_ids.iter().enumerate() {
                        routes.entry(id).or_default().push((index, slot));
                    }
                }
                let missing = routes
                    .keys()
                    .copied()
                    .filter(|id| !experts.contains_key(id))
                    .collect::<Vec<_>>();
                let loaded = par_ranges(missing.len(), |range| {
                    range
                        .map(|index| {
                            Self::load_expert(
                                &self.root,
                                &self.catalog,
                                self.verifier.as_ref(),
                                self.layer,
                                missing[index],
                            )
                            .map(|expert| (missing[index], expert))
                        })
                        .collect()
                });
                for result in loaded {
                    let (id, expert) = result?;
                    experts.insert(id, expert);
                }
                // Experts are independent; each computes the rows routed to it.
                let routes = routes.into_iter().collect::<Vec<_>>();
                let experts = &*experts;
                let post = &post;
                let computed = par_ranges(routes.len(), |range| {
                    range
                        .flat_map(|index| {
                            let (id, rows) = &routes[index];
                            let expert = &experts[id];
                            let inputs = rows
                                .iter()
                                .map(|&(row, _)| post[row].clone())
                                .collect::<Vec<_>>();
                            let gate = rounded_rows(project(&expert.gate, &inputs, false));
                            let up = rounded_rows(project(&expert.up, &inputs, false));
                            let activated = gate
                                .iter()
                                .zip(&up)
                                .map(|(gate, up)| {
                                    gate.iter()
                                        .zip(up)
                                        .map(|(&g, &u)| {
                                            rounded(rounded(g / (1.0 + (-g).exp())) * u)
                                        })
                                        .collect::<Vec<_>>()
                                })
                                .collect::<Vec<_>>();
                            // Each rank's routed row is its BF16 partial over its
                            // share of the expert intermediate.
                            let parts = tp_partials.unwrap_or(1);
                            let down = project_parts(&expert.down, &activated, false, parts)
                                .into_iter()
                                .map(|row| {
                                    row.into_iter()
                                        .map(|parts| {
                                            parts.into_iter().map(rounded).collect::<Vec<_>>()
                                        })
                                        .collect::<Vec<_>>()
                                })
                                .collect::<Vec<_>>();
                            rows.iter()
                                .copied()
                                .zip(activated)
                                .zip(down)
                                .map(|(((row, slot), activated), down)| {
                                    (row, slot, activated, down)
                                })
                                .collect::<Vec<_>>()
                        })
                        .collect()
                });
                let mut slots = vec![vec![(Vec::new(), Vec::new()); TOP_K]; tokens];
                for (row, slot, activated, down) in computed {
                    slots[row][slot] = (activated, down);
                }
                // Sum each row's routes in route order, like the single-row
                // reference; with TP4 partials each rank combines and rounds.
                let output = slots
                    .iter()
                    .zip(&selections)
                    .map(|(routes, (selection, _))| {
                        (0..HIDDEN)
                            .map(|column| {
                                let parts = routes[0].1[column].len();
                                (0..parts)
                                    .map(|part| {
                                        rounded(
                                            routes
                                                .iter()
                                                .zip(&selection.weights)
                                                .map(|((_, down), &weight)| {
                                                    down[column][part] * weight
                                                })
                                                .sum(),
                                        )
                                    })
                                    .sum()
                            })
                            .collect()
                    })
                    .collect();
                FeedForwardRows {
                    output,
                    router_logits: Some(logits),
                    selections: Some(selections),
                    gate_up_routes: Some(
                        slots
                            .into_iter()
                            .map(|routes| routes.into_iter().map(|(a, _)| a).collect())
                            .collect(),
                    ),
                }
            }
        };
        Ok((post, feed_forward))
    }

    /// Run consecutive rows. With `Some(ranks)`, round each rank's partial O
    /// and MLP output to BF16 and add the `ranks` partials to the residual
    /// with one rounding, as Chopinfer's row-parallel projections, per-rank
    /// MoE combine and fused all-reduce do, instead of HF's one rounded
    /// projection. GPU router choices, when given, are checked and followed
    /// as in `run_traced_with_router_ids`.
    pub fn run_with_partials(
        &mut self,
        input: &[u16],
        history: &mut KvHistory,
        gpu_ids: Option<&[Vec<usize>]>,
        tp_partials: Option<usize>,
    ) -> MimoResult<(Vec<u16>, Vec<LayerTrace>)> {
        let kv_heads = if self.global {
            model_mimo26_contract::tp::GA_KV_HEADS as usize
        } else {
            model_mimo26_contract::tp::SWA_KV_HEADS as usize
        };
        if input.is_empty()
            || !input.len().is_multiple_of(HIDDEN)
            || history.keys.len() != history.values.len()
            || history.keys.iter().any(|key| key.len() != kv_heads * QK)
            || history
                .values
                .iter()
                .any(|value| value.len() != kv_heads * V)
        {
            return Err("layer input or KV history has invalid shape".into());
        }
        let tokens = input.len() / HIDDEN;
        if gpu_ids.is_some_and(|ids| ids.len() != tokens) {
            return Err("GPU router row count mismatch".into());
        }
        let norm = HF_NORM.apply_bf16(input, &self.input_norm, HIDDEN)?;
        let normalized = bf16_rows(&norm);
        let projected = rounded_rows(project(&self.qkv, &normalized, true));
        let start = history.keys.len();
        let theta = if self.global {
            GEOMETRY.rope_theta()
        } else {
            GEOMETRY.swa_rope_theta()
        };
        let mut rotated = Vec::with_capacity(tokens);
        for (index, row) in projected.iter().enumerate() {
            let (q_projected, k_projected, v_projected) = split_qkv_chunks(
                row,
                CHECKPOINT_TP,
                Q_HEADS / CHECKPOINT_TP * QK,
                kv_heads / CHECKPOINT_TP * QK,
                kv_heads / CHECKPOINT_TP * V,
            )?;
            let q = hf_rope(&q_projected, Q_HEADS, start + index, theta);
            let k = hf_rope(&k_projected, kv_heads, start + index, theta);
            let v: Vec<_> = v_projected
                .iter()
                .map(|&value| rounded(value * GEOMETRY.semantics.attention.value_scale))
                .collect();
            history.keys.push(k.clone());
            history.values.push(v.clone());
            rotated.push((q, k, v));
        }
        let geometry = AttentionGeometry {
            q_heads: Q_HEADS,
            kv_heads,
            qk_dim: QK,
            v_dim: V,
            window_left: (!self.global).then_some(127),
        };
        let attention = {
            let history = &*history;
            let sink = self.sink.as_deref();
            let rotated = &rotated;
            par_ranges(tokens, |range| {
                range
                    .map(|index| {
                        attention_token(&rotated[index].0, history, start + index, geometry, sink)
                    })
                    .collect()
            })
        };
        let o = row_parallel(
            &Bf16Rows {
                bits: &self.o,
                cols: Q_HEADS * V,
            },
            &attention,
            tp_partials,
        );
        let mut residual = Vec::with_capacity(input.len());
        for (row, old) in o.iter().zip(input.chunks_exact(HIDDEN)) {
            for (&projected, &old) in row.iter().zip(old) {
                residual.push(f32_to_bf16(
                    (projected + f64::from(bf16_to_f32(old))) as f32,
                ));
            }
        }
        let (post, feed_forward) = self.feed_forward_rows(&residual, gpu_ids, tp_partials)?;
        let mut output = Vec::with_capacity(input.len());
        let mut traces = Vec::with_capacity(tokens);
        let FeedForwardRows {
            output: mlp,
            mut router_logits,
            selections,
            mut gate_up_routes,
        } = feed_forward;
        for (index, ((((normalized, qkv), (q, k, v)), attention), mlp)) in normalized
            .into_iter()
            .zip(projected)
            .zip(rotated)
            .zip(attention)
            .zip(mlp)
            .enumerate()
        {
            let residual = &residual[index * HIDDEN..(index + 1) * HIDDEN];
            let final_output = residual
                .iter()
                .zip(&mlp)
                .map(|(&r, m)| f32_to_bf16((f64::from(bf16_to_f32(r)) + m) as f32))
                .collect::<Vec<_>>();
            let selection = selections.as_ref().map(|all| &all[index]);
            output.extend_from_slice(&final_output);
            traces.push(LayerTrace {
                input_norm: normalized,
                qkv,
                q,
                k,
                v,
                attention,
                attention_residual: residual
                    .iter()
                    .map(|&v| f64::from(bf16_to_f32(v)))
                    .collect(),
                post_norm: post[index].clone(),
                router_logits: router_logits
                    .as_mut()
                    .map(|all| std::mem::take(&mut all[index])),
                router_ids: selection.map(|(s, _)| s.expert_ids.clone()),
                router_weights: selection.map(|(s, _)| s.weights.clone()),
                router_flips: selection.map_or(0, |(_, flips)| *flips),
                gate_up_routes: gate_up_routes
                    .as_mut()
                    .map(|all| std::mem::take(&mut all[index])),
                moe: mlp,
                final_output: final_output
                    .iter()
                    .map(|&v| f64::from(bf16_to_f32(v)))
                    .collect(),
            });
        }
        Ok((output, traces))
    }
}

/// Embedding table, final norm and vocabulary head of the text model.
pub struct ModelEnds {
    embed: Vec<u16>,
    norm: Vec<u16>,
    head: Vec<u16>,
}

impl ModelEnds {
    pub fn load(root: &Path) -> MimoResult<Self> {
        let config = MiMoConfig::load(root)?;
        let catalog = Catalog::load(root, &config)?;
        let read = |name: &str| read_tensor(root, &catalog, name).map(|bytes| u16s(&bytes));
        Ok(Self {
            embed: read("model.embed_tokens.weight")?,
            norm: read("model.norm.weight")?,
            head: read("lm_head.weight")?,
        })
    }

    pub fn vocab(&self) -> usize {
        self.head.len() / HIDDEN
    }

    /// BF16 embedding rows of `ids`.
    pub fn embed(&self, ids: &[u32]) -> MimoResult<Vec<u16>> {
        ids.iter()
            .map(|&id| {
                let id = id as usize;
                self.embed
                    .get(id * HIDDEN..(id + 1) * HIDDEN)
                    .ok_or_else(|| format!("token ID {id} outside the vocabulary").into())
            })
            .collect::<MimoResult<Vec<_>>>()
            .map(|rows| rows.concat())
    }

    /// HF final RMSNorm with BF16 output.
    pub fn final_norm(&self, hidden: &[u16]) -> MimoResult<Vec<u16>> {
        Ok(HF_NORM.apply_bf16(hidden, &self.norm, HIDDEN)?)
    }

    /// Exact f64 vocabulary logits of normalized rows, as [row][vocab].
    pub fn logits(&self, normalized: &[u16]) -> Vec<Vec<f64>> {
        project(
            &Bf16Rows {
                bits: &self.head,
                cols: HIDDEN,
            },
            &bf16_rows(normalized),
            true,
        )
    }
}

/// Apply the numerical contract's per-token layer gate and return the maximum absolute
/// element error for the receipt.
pub fn compare_layer_tokens(actual: &[u16], reference: &[f64]) -> Result<f64, CheckFailure> {
    let actual: Vec<_> = actual
        .iter()
        .map(|&bits| f64::from(bf16_to_f32(bits)))
        .collect();
    check_layer_cosine(&actual, reference, HIDDEN)?;
    Ok(actual
        .iter()
        .zip(reference)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max))
}

#[cfg(test)]
mod tests {
    use super::*;
    use kernel_core::contract::RotaryStyle;
    use kernel_core::oracle::{GemmEpilogue, GemmShape, gemm};

    /// Each of `ranks` ranks rounds its partial over an equal, contiguous
    /// share of the reduction to BF16; HF rounds the whole product once.
    #[test]
    fn row_parallel_rounds_one_partial_per_rank() {
        let (rows, cols) = (3, 64);
        let bits: Vec<u16> = (0..rows * cols)
            .map(|i| f32_to_bf16(((i * 37 % 101) as f32 - 50.0) / 7.0))
            .collect();
        let input: Vec<f64> = (0..cols)
            .map(|i| rounded(((i * 13 % 29) as f64 - 14.0) / 3.0))
            .collect();
        let matrix = Bf16Rows { bits: &bits, cols };
        let weight = |row: usize, col: usize| f64::from(bf16_to_f32(bits[row * cols + col]));
        let dot = |row: usize, range: Range<usize>| -> f64 {
            range.map(|col| weight(row, col) * input[col]).sum()
        };
        let hf = row_parallel(&matrix, std::slice::from_ref(&input), None);
        assert_eq!(
            hf[0],
            (0..rows)
                .map(|row| rounded(dot(row, 0..cols)))
                .collect::<Vec<_>>()
        );
        for ranks in [2, 4] {
            let width = cols / ranks;
            let expected: Vec<f64> = (0..rows)
                .map(|row| {
                    (0..ranks)
                        .map(|rank| rounded(dot(row, rank * width..(rank + 1) * width)))
                        .sum()
                })
                .collect();
            let partials = row_parallel(&matrix, std::slice::from_ref(&input), Some(ranks));
            assert_eq!(partials[0], expected, "{ranks}");
        }
    }

    /// Past the rotary pairs, K passes through; at position 0 the RoPE is
    /// the identity; elsewhere it exactly follows the shared HF-table F32
    /// rotation.
    #[test]
    fn serving_rope_follows_the_hf_table_f32_rotation() {
        let k: Vec<f64> = (0..2 * QK)
            .map(|i| f64::from(bf16_to_f32(f32_to_bf16((i as f32 - 150.0) / 37.0))))
            .collect();
        assert_eq!(serving_rope(&k, 2, 0, 1e7), k);
        let rotated = serving_rope(&k, 2, 5000, 1e7);
        for head in 0..2 {
            for pair in 0..32 {
                let (c, s) = kernel_core::hf_rope::cos_sin(5000, pair, 1e7);
                let (x, y) = (k[head * QK + pair], k[head * QK + pair + 32]);
                let exact = ((x * c - y * s) as f32) as f64;
                assert_eq!(rotated[head * QK + pair], exact);
            }
        }
        assert_eq!(rotated[64..QK], k[64..QK]);
    }

    #[test]
    fn checkpoint_qkv_chunks_form_global_head_order() {
        let source = [
            0.0, 1.0, 2.0, 3.0, 10.0, 11.0, 12.0, 13.0, 20.0, 21.0, 22.0, 23.0, 30.0, 31.0, 32.0,
            33.0,
        ];
        let (q, k, v) = split_qkv_chunks(&source, 4, 2, 1, 1).unwrap();
        assert_eq!(q, [0.0, 1.0, 10.0, 11.0, 20.0, 21.0, 30.0, 31.0]);
        assert_eq!(k, [2.0, 12.0, 22.0, 32.0]);
        assert_eq!(v, [3.0, 13.0, 23.0, 33.0]);
    }

    #[test]
    fn chunked_fp8_scales_restart_at_rank_boundary() {
        let matrix = Fp8Matrix {
            rows: 8,
            cols: 1,
            codes: vec![0x38; 8],
            scales: vec![1.0, 2.0, 3.0, 4.0],
            chunk_rows: 2,
        };
        assert_eq!(matrix.dot(2, &[1.0]), 2.0);
        assert_eq!(matrix.dot(7, &[1.0]), 4.0);
        let decoded = (0..8)
            .map(|row| f64::from(matrix.scales[row / 2]))
            .collect::<Vec<_>>();
        let reference = gemm(
            &[1.0],
            &decoded,
            GemmShape {
                m: 1,
                n: 8,
                k: 1,
                experts: 1,
            },
            &[],
            GemmEpilogue {
                residual: None,
                swiglu: None,
            },
        )
        .unwrap();
        assert_eq!(
            (0..8)
                .map(|row| matrix.dot(row, &[1.0]))
                .collect::<Vec<_>>(),
            reference.values
        );
    }

    #[test]
    fn mxfp4_nibbles_and_scales_match_codec() {
        let matrix = Mxfp4Matrix {
            rows: 1,
            cols: 32,
            codes: vec![0x21; 16],
            scales: vec![127],
        };
        let expected = 16.0 * (SmallFloat::E2M1.decode_f64(1) + SmallFloat::E2M1.decode_f64(2));
        assert_eq!(matrix.dot(0, &[1.0; 32]), expected);
        let decoded = (0..32)
            .map(|k| SmallFloat::E2M1.decode_f64(if k % 2 == 0 { 1 } else { 2 }))
            .collect::<Vec<_>>();
        let reference = gemm(
            &[1.0; 32],
            &decoded,
            GemmShape {
                m: 1,
                n: 1,
                k: 32,
                experts: 1,
            },
            &[],
            GemmEpilogue {
                residual: None,
                swiglu: None,
            },
        )
        .unwrap();
        assert_eq!(matrix.dot(0, &[1.0; 32]), reference.values[0]);
    }

    #[test]
    fn tiny_attention_history_causal_window_and_sink() {
        let history = KvHistory {
            keys: vec![vec![1.0], vec![1.0], vec![1.0]],
            values: vec![vec![8.0], vec![4.0], vec![2.0]],
        };
        let geometry = AttentionGeometry {
            q_heads: 1,
            kv_heads: 1,
            qk_dim: 1,
            v_dim: 1,
            window_left: None,
        };
        let global = attention_token(&[1.0], &history, 2, geometry, None);
        let window = attention_token(
            &[1.0],
            &history,
            2,
            AttentionGeometry {
                window_left: Some(1),
                ..geometry
            },
            None,
        );
        let sink = attention_token(
            &[1.0],
            &history,
            2,
            AttentionGeometry {
                window_left: Some(1),
                ..geometry
            },
            Some(&[1.0]),
        );
        assert_eq!(global, vec![rounded(rounded(1.0 / 3.0) * 14.0)]);
        assert_eq!(window, vec![3.0]);
        assert_eq!(sink, vec![rounded(rounded(1.0 / 3.0) * 6.0)]);
    }

    #[test]
    fn partial_neox_rope_agrees_with_core_at_position_zero() {
        let input = (0..QK)
            .map(|i| rounded((i as f64 - 96.0) / 128.0))
            .collect::<Vec<_>>();
        let core =
            kernel_core::oracle::rope(&input, &[0], QK, 64, 1e4, RotaryStyle::Neox, None).unwrap();
        assert_eq!(hf_rope(&input, 1, 0, 1e4), core);
        assert_eq!(&hf_rope(&input, 1, 12, 1e4)[64..], &input[64..]);
    }

    #[test]
    fn real_weight_layer_loads_when_requested() {
        let Ok(path) = std::env::var("CHOPINFER_MIMO_CHECKPOINT") else {
            return;
        };
        let layer = LayerOracle::load(Path::new(&path), 5).unwrap();
        assert!(layer.global);
        assert_eq!(layer.qkv.scales.len(), 108 * 32);
        assert_eq!(layer.qkv.codes.len(), 13568 * HIDDEN);
    }

    #[test]
    fn real_weight_chunk_matches_single_rows_when_requested() {
        let (Ok(path), Ok(_)) = (
            std::env::var("CHOPINFER_MIMO_CHECKPOINT"),
            std::env::var("CHOPINFER_MIMO_ORACLE_RUN"),
        ) else {
            return;
        };
        let mut layer = LayerOracle::load(Path::new(&path), 1).unwrap();
        let input = (0..3 * HIDDEN)
            .map(|i| f32_to_bf16(((i % 29) as f32 - 14.0) / 16.0))
            .collect::<Vec<_>>();
        let mut chunk_history = KvHistory::default();
        let chunk = layer.run(&input, &mut chunk_history).unwrap();
        let mut row_history = KvHistory::default();
        let rows = input
            .chunks_exact(HIDDEN)
            .flat_map(|row| layer.run(row, &mut row_history).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(chunk, rows);
    }

    #[test]
    fn router_near_tie_follows_gpu_choice_with_bounded_gap() {
        let x = vec![1.0; HIDDEN];
        let weight = vec![f32_to_bf16(1.0 / HIDDEN as f32); 256 * HIDDEN];
        let bias = vec![0.0; 256];
        let mut logits = vec![-1.0; 256];
        logits[..7].fill(1.0);
        logits[7] = 1e-6;
        logits[8] = 0.0;
        let ids = [0, 1, 2, 3, 4, 5, 6, 8];
        let (selected, flips) = router_with_gpu_choices(&logits, &weight, &bias, &x, &ids).unwrap();
        assert_eq!(selected.expert_ids, ids);
        assert_eq!(flips, 1);
        logits[7] = 0.1;
        assert!(router_with_gpu_choices(&logits, &weight, &bias, &x, &ids).is_err());
        logits[6] = 0.001;
        logits[7] = 0.000001;
        let two_swaps = [0, 1, 2, 3, 4, 5, 8, 9];
        assert!(router_with_gpu_choices(&logits, &weight, &bias, &x, &two_swaps).is_err());
    }

    #[test]
    fn real_weight_single_token_runs_when_requested() {
        let (Ok(path), Ok(_)) = (
            std::env::var("CHOPINFER_MIMO_CHECKPOINT"),
            std::env::var("CHOPINFER_MIMO_ORACLE_RUN"),
        ) else {
            return;
        };
        let mut layer = LayerOracle::load(Path::new(&path), 5).unwrap();
        let input = (0..HIDDEN)
            .map(|i| f32_to_bf16(((i % 31) as f32 - 15.0) / 32.0))
            .collect::<Vec<_>>();
        let mut history = KvHistory::default();
        let output = layer.run(&input, &mut history).unwrap();
        assert_eq!(output.len(), HIDDEN);
        assert_eq!(history.keys.len(), 1);
        assert!(output.iter().all(|&bits| bf16_to_f32(bits).is_finite()));
    }
}
