//! Shared parts of the BF16 accumulation studies (`tests/mma_accumulation.rs`,
//! `tests/ops/gemm_accumulation.rs` and `tests/qwen38_gemm.rs`): a compensated
//! f64 reference, error statistics against it, the probe cubin's launches, and
//! readers for checkpoint weights and dumped activations.
//!
//! Real inputs come from `CHOPINFER_QWEN38_CHECKPOINT`,
//! `CHOPINFER_MIMO_CHECKPOINT`, `CHOPINFER_MIMO_AUDIT_DIR` (the MiMo numerics
//! audit), `CHOPINFER_QWEN38_LAYER_DUMPS` (the GPU reference runner's BF16
//! taps, one directory of `step*` taps per prompt) and
//! `CHOPINFER_QWEN38_STEP_DUMPS` (bundle `step*` dumps).
#![allow(dead_code)]

use kernel_core::codec::{bf16_to_f32, f32_to_bf16};
use kernel_core::contract::{DType, SwigluColumns};
use kernel_core::oracle::{NormArithmetic, NormOffset, NormScale, RmsNorm};
use kernel_core::oracle::{
    OracleOutput, SiluGlu, accumulation_bound, compensated_sum_with_magnitude, numeric_bound,
    silu_glu, swiglu,
};
use kernel_lab::{LabResult, Session, launch};
use runtime_cuda::{DeviceBuffer, KernelArgument, LaunchConfig, Module};
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::fs::File;
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};

/// HF's BF16 RMSNorm with eps 1e-6: F32 arithmetic, the normalized value
/// and the output each rounded to BF16.
pub const NORM: RmsNorm = RmsNorm {
    eps: 1e-6,
    offset: NormOffset::Plain,
    arithmetic: NormArithmetic::F32,
    scale: NormScale::MultiplyReciprocal,
    before_weight: Some(DType::Bf16),
    output: Some(DType::Bf16),
};

pub const PROBE: &str = "probe_mma_accumulation_sm120";

pub fn bf16(bits: u16) -> f64 {
    f64::from(bf16_to_f32(bits))
}

/// Exact references for Y[token][row] = sum_k W[row][k] X[token][k] over
/// row-major BF16 weights [n][k] and activations [m][k]: Neumaier-compensated
/// f64 sums of the exact products in K order, with S per output.
pub fn reference(weight: &[u16], activation: &[u16], n: usize, m: usize, k: usize) -> OracleOutput {
    let w = weight.iter().map(|&b| bf16(b)).collect::<Vec<_>>();
    let x = activation.iter().map(|&b| bf16(b)).collect::<Vec<_>>();
    let mut values = vec![0.0; n * m];
    let mut sums = vec![0.0; n * m];
    let threads = std::thread::available_parallelism().map_or(8, |v| v.get());
    let tokens_per = m.div_ceil(threads).max(1);
    std::thread::scope(|scope| {
        for (chunk, (values, sums)) in values
            .chunks_mut(tokens_per * n)
            .zip(sums.chunks_mut(tokens_per * n))
            .enumerate()
        {
            let (w, x) = (&w, &x);
            scope.spawn(move || {
                for (i, (value, sum)) in values.iter_mut().zip(sums.iter_mut()).enumerate() {
                    let token = chunk * tokens_per + i / n;
                    let row = i % n;
                    (*value, *sum) = compensated_sum_with_magnitude(
                        w[row * k..][..k]
                            .iter()
                            .zip(&x[token * k..][..k])
                            .map(|(w, x)| w * x),
                    );
                }
            });
        }
    });
    OracleOutput::new(values, sums, k).expect("reference shape")
}

/// One F32 ulp of `value` (2^-149 below the normal range).
pub fn ulp32(value: f64) -> f64 {
    let magnitude = value.abs();
    if magnitude < 2_f64.powi(-126) {
        return 2_f64.powi(-149);
    }
    let exponent = ((magnitude.to_bits() >> 52) & 0x7ff) as i32 - 1023;
    2_f64.powi(exponent - 23)
}

/// Correctly rounded (nearest, ties to even) BF16 of an f64 value.
pub fn bf16_nearest(value: f64) -> u16 {
    let guess = f32_to_bf16(value as f32);
    let sign = guess & 0x8000;
    let mut best = guess;
    let mut best_error = (bf16(guess) - value).abs();
    let magnitude = guess & 0x7fff;
    for candidate in [magnitude.wrapping_sub(1), magnitude.wrapping_add(1)] {
        if candidate >= 0x7f80 {
            continue;
        }
        let bits = sign | candidate;
        let error = (bf16(bits) - value).abs();
        if error < best_error || (error == best_error && bits & 1 == 0 && best & 1 == 1) {
            best = bits;
            best_error = error;
        }
    }
    best
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Quantiles {
    pub max: f64,
    pub p9999: f64,
}

impl Quantiles {
    fn of(mut values: Vec<f64>) -> Self {
        if values.is_empty() {
            return Self::default();
        }
        let rank = ((values.len() as f64 * 0.9999).ceil() as usize).clamp(1, values.len()) - 1;
        let (_, p9999, _) = values.select_nth_unstable_by(rank, f64::total_cmp);
        let p9999 = *p9999;
        let max = values.iter().copied().fold(0.0, f64::max);
        Self { max, p9999 }
    }

    pub fn json(&self) -> Value {
        json!({"max": self.max, "p99.99": self.p9999})
    }
}

/// Errors of one output set against its reference.
#[derive(Clone, Debug, Default)]
pub struct Summary {
    pub outputs: usize,
    /// |y - ref| in F32 ulps of the reference.
    pub ulps: Quantiles,
    /// |y - ref| / (2^-22 K S): the accumulation term alone.
    pub accumulation: Quantiles,
    /// |y - ref| / numeric_bound for the output dtype.
    pub contract: Quantiles,
    /// Mean of sign(ref) (y - ref) / (2^-22 K S); negative means toward zero.
    pub bias: f64,
    /// Fraction whose round-to-nearest BF16 differs from the correctly
    /// rounded reference.
    pub misrounded: f64,
    /// Outputs over the contract bound.
    pub violations: usize,
}

impl Summary {
    pub fn json(&self) -> Value {
        json!({
            "outputs": self.outputs,
            "ulps": self.ulps.json(),
            "accumulation_ratio": self.accumulation.json(),
            "contract_ratio": self.contract.json(),
            "bias": self.bias,
            "bf16_misrounded": self.misrounded,
            "violations": self.violations,
        })
    }
}

/// Statistics of `actual` (the device outputs in reference order) against
/// `reference`, with the contract bound for `output`.
pub fn summarize(actual: &[f32], reference: &OracleOutput, output: DType) -> Summary {
    assert_eq!(
        actual.len(),
        reference.values.len(),
        "output and reference lengths"
    );
    let mut ulps = Vec::with_capacity(actual.len());
    let mut accumulation = Vec::with_capacity(actual.len());
    let mut contract = Vec::with_capacity(actual.len());
    let (mut bias, mut biased, mut misrounded, mut violations) = (0.0, 0_usize, 0_usize, 0_usize);
    for (index, &value) in actual.iter().enumerate() {
        let exact = reference.values[index];
        let error = (f64::from(value) - exact).abs();
        let scale = accumulation_bound(
            reference.reduction_lengths[index],
            reference.sum_abs_products[index],
        );
        let bound = numeric_bound(reference, index, output);
        ulps.push(error / ulp32(exact));
        accumulation.push(if scale > 0.0 { error / scale } else { 0.0 });
        let ratio = if bound > 0.0 {
            error / bound
        } else if error > 0.0 {
            f64::INFINITY
        } else {
            0.0
        };
        violations += usize::from(ratio > 1.0);
        contract.push(ratio);
        if scale > 0.0 {
            let sign = if exact == 0.0 { 0.0 } else { exact.signum() };
            bias += sign * (f64::from(value) - exact) / scale;
            biased += 1;
        }
        misrounded += usize::from(bf16(f32_to_bf16(value)) != bf16(bf16_nearest(exact)));
    }
    Summary {
        outputs: actual.len(),
        ulps: Quantiles::of(ulps),
        accumulation: Quantiles::of(accumulation),
        contract: Quantiles::of(contract),
        bias: if biased > 0 {
            bias / biased as f64
        } else {
            0.0
        },
        misrounded: misrounded as f64 / actual.len().max(1) as f64,
        violations,
    }
}

/// A run of the probe: one MMA chain over all of K (`Chain(None)`), runs of a
/// given K length summed in F32 (`Chain(Some(length))`), Kahan-compensated
/// runs, or sequential F32 FMA.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Scheme {
    Chain(Option<u32>),
    Kahan(u32),
    Fma,
}

impl Scheme {
    pub fn label(self) -> String {
        match self {
            Scheme::Chain(None) => "chain_k".into(),
            Scheme::Chain(Some(length)) => format!("chain_{length}"),
            Scheme::Kahan(length) => format!("kahan_{length}"),
            Scheme::Fma => "fma".into(),
        }
    }
}

/// Every scheme of the study: the single chain (a), 128-K runs (b), FMA (c),
/// the other run lengths (1024 is b12x's long-K flush) and Kahan at 16 (b12x's
/// prefill) and 128.
pub const SCHEMES: [Scheme; 11] = [
    Scheme::Chain(None),
    Scheme::Chain(Some(128)),
    Scheme::Fma,
    Scheme::Chain(Some(16)),
    Scheme::Chain(Some(32)),
    Scheme::Chain(Some(64)),
    Scheme::Chain(Some(256)),
    Scheme::Chain(Some(512)),
    Scheme::Chain(Some(1024)),
    Scheme::Kahan(16),
    Scheme::Kahan(128),
];

/// Device copies of one probe problem: weights [n][k], activations [m][k].
pub struct Problem {
    pub n: usize,
    pub m: usize,
    pub k: usize,
    weight: DeviceBuffer,
    activation: DeviceBuffer,
    output: DeviceBuffer,
}

impl Problem {
    pub fn upload(
        session: &Session,
        weight: &[u16],
        activation: &[u16],
        n: usize,
        m: usize,
        k: usize,
    ) -> LabResult<Self> {
        if n % 16 != 0
            || m % 32 != 0
            || k % 16 != 0
            || weight.len() != n * k
            || activation.len() != m * k
        {
            return Err(format!(
                "probe shape n={n} m={m} k={k} is not a multiple of 16x32x16"
            ));
        }
        Ok(Self {
            n,
            m,
            k,
            weight: session.upload_bf16(weight)?,
            activation: session.upload_bf16(activation)?,
            output: session.allocate(n * m * 4)?,
        })
    }

    /// Runs `scheme` and returns the F32 outputs [m][n].
    pub fn run(&self, session: &Session, module: &Module, scheme: Scheme) -> LabResult<Vec<f32>> {
        let mut arguments = vec![
            KernelArgument::Buffer(&self.weight),
            KernelArgument::Buffer(&self.activation),
            KernelArgument::Buffer(&self.output),
            KernelArgument::U32(self.n as u32),
            KernelArgument::U32(self.m as u32),
            KernelArgument::U32(self.k as u32),
        ];
        let (symbol, grid, block) = match scheme {
            Scheme::Fma => {
                let threads = (self.n * self.m) as u32;
                (
                    "chop_probe_fma_chain_v1",
                    (threads.div_ceil(256), 1, 1),
                    (256, 1, 1),
                )
            }
            Scheme::Chain(_) | Scheme::Kahan(_) => {
                let (chain, compensated) = match scheme {
                    Scheme::Chain(Some(length)) => (length, 0),
                    Scheme::Kahan(length) => (length, 1),
                    _ => (self.k as u32, 0),
                };
                arguments.push(KernelArgument::U32(chain));
                arguments.push(KernelArgument::U32(compensated));
                let strips = (self.n / 16) as u32;
                (
                    "chop_probe_mma_chain_v1",
                    (strips.div_ceil(4), (self.m / 32) as u32, 1),
                    (128, 1, 1),
                )
            }
        };
        module
            .function(symbol)
            .map_err(|e| e.to_string())?
            .launch(
                &session.stream,
                launch::simple_config(grid, block, 0, LaunchConfig::NO_CLUSTER, false)?,
                &arguments,
            )
            .map_err(|e| e.to_string())?;
        session.download_f32(&self.output, self.n * self.m)
    }
}

/// Sequential F32 fused multiply-add from zero in K order, as the probe's FMA
/// scheme computes output `index` of [m][n].
pub fn fma_emulation(weight: &[u16], activation: &[u16], n: usize, k: usize, index: usize) -> f32 {
    let row = &weight[(index % n) * k..][..k];
    let token = &activation[(index / n) * k..][..k];
    row.iter().zip(token).fold(0.0_f32, |total, (&w, &x)| {
        bf16_to_f32(w).mul_add(bf16_to_f32(x), total)
    })
}

pub fn env_path(name: &str) -> Option<PathBuf> {
    std::env::var_os(name)
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
}

#[derive(Clone, Debug)]
pub struct TensorInfo {
    pub dtype: String,
    pub shape: Vec<usize>,
    begin: u64,
    end: u64,
}

/// One safetensors file, read positionally.
pub struct SafeTensors {
    path: PathBuf,
    file: File,
    data: u64,
    pub tensors: BTreeMap<String, TensorInfo>,
}

impl SafeTensors {
    pub fn open(path: &Path) -> LabResult<Self> {
        let error = |e: &dyn std::fmt::Display| format!("{}: {e}", path.display());
        let file = File::open(path).map_err(|e| error(&e))?;
        let mut length = [0_u8; 8];
        file.read_exact_at(&mut length, 0).map_err(|e| error(&e))?;
        let length = u64::from_le_bytes(length);
        if length > 1 << 30 {
            return Err(error(&"safetensors header is too long"));
        }
        let mut header = vec![0_u8; length as usize];
        file.read_exact_at(&mut header, 8).map_err(|e| error(&e))?;
        let header: BTreeMap<String, Value> =
            serde_json::from_slice(&header).map_err(|e| error(&e))?;
        let mut tensors = BTreeMap::new();
        for (name, value) in header {
            if name == "__metadata__" {
                continue;
            }
            let field = |key: &str| {
                value
                    .get(key)
                    .ok_or_else(|| error(&format!("{name}: missing {key}")))
            };
            let dtype = field("dtype")?
                .as_str()
                .ok_or_else(|| error(&"dtype"))?
                .to_string();
            let shape = field("shape")?
                .as_array()
                .ok_or_else(|| error(&"shape"))?
                .iter()
                .map(|d| {
                    d.as_u64()
                        .map(|d| d as usize)
                        .ok_or_else(|| error(&"shape"))
                })
                .collect::<LabResult<Vec<_>>>()?;
            let offsets = field("data_offsets")?
                .as_array()
                .ok_or_else(|| error(&"offsets"))?;
            let offset = |i: usize| {
                offsets
                    .get(i)
                    .and_then(Value::as_u64)
                    .ok_or_else(|| error(&"offsets"))
            };
            tensors.insert(
                name,
                TensorInfo {
                    dtype,
                    shape,
                    begin: offset(0)?,
                    end: offset(1)?,
                },
            );
        }
        Ok(Self {
            path: path.into(),
            file,
            data: 8 + length,
            tensors,
        })
    }

    pub fn info(&self, name: &str) -> LabResult<&TensorInfo> {
        self.tensors
            .get(name)
            .ok_or_else(|| format!("{}: missing tensor {name}", self.path.display()))
    }

    fn bytes(&self, info: &TensorInfo, begin: u64, len: u64) -> LabResult<Vec<u8>> {
        if begin + len > info.end - info.begin {
            return Err(format!("{}: read past a tensor", self.path.display()));
        }
        let mut bytes = vec![0_u8; len as usize];
        self.file
            .read_exact_at(&mut bytes, self.data + info.begin + begin)
            .map_err(|e| format!("{}: {e}", self.path.display()))?;
        Ok(bytes)
    }

    /// Rows `[first, first + count)` of a BF16 tensor, viewed as [rows][cols]
    /// over its leading dimension.
    pub fn bf16_rows(&self, name: &str, first: usize, count: usize) -> LabResult<Vec<u16>> {
        let info = self.info(name)?;
        if info.dtype != "BF16" || info.shape.is_empty() {
            return Err(format!(
                "{name}: {} {:?} is not a BF16 matrix",
                info.dtype, info.shape
            ));
        }
        let cols = info.shape[1..].iter().product::<usize>();
        if first + count > info.shape[0] {
            return Err(format!(
                "{name}: rows {first}+{count} past {}",
                info.shape[0]
            ));
        }
        let bytes = self.bytes(info, (first * cols * 2) as u64, (count * cols * 2) as u64)?;
        Ok(bytes
            .chunks_exact(2)
            .map(|b| u16::from_le_bytes([b[0], b[1]]))
            .collect())
    }

    pub fn bf16(&self, name: &str) -> LabResult<(Vec<usize>, Vec<u16>)> {
        let shape = self.info(name)?.shape.clone();
        let data = self.bf16_rows(name, 0, shape[0])?;
        Ok((shape, data))
    }
}

/// A checkpoint directory with `model.safetensors.index.json`.
pub struct Checkpoint {
    root: PathBuf,
    weight_map: BTreeMap<String, String>,
    files: std::cell::RefCell<BTreeMap<String, std::rc::Rc<SafeTensors>>>,
}

impl Checkpoint {
    pub fn open(root: &Path) -> LabResult<Self> {
        let path = root.join("model.safetensors.index.json");
        let index: Value = serde_json::from_slice(
            &std::fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))?,
        )
        .map_err(|e| format!("{}: {e}", path.display()))?;
        let weight_map = index["weight_map"]
            .as_object()
            .ok_or_else(|| format!("{}: missing weight_map", path.display()))?
            .iter()
            .map(|(name, file)| (name.clone(), file.as_str().unwrap_or_default().to_string()))
            .collect();
        Ok(Self {
            root: root.into(),
            weight_map,
            files: Default::default(),
        })
    }

    pub fn file(&self, name: &str) -> LabResult<std::rc::Rc<SafeTensors>> {
        let file = self
            .weight_map
            .get(name)
            .ok_or_else(|| format!("{}: missing tensor {name}", self.root.display()))?;
        if let Some(open) = self.files.borrow().get(file) {
            return Ok(open.clone());
        }
        let open = std::rc::Rc::new(SafeTensors::open(&self.root.join(file))?);
        self.files.borrow_mut().insert(file.clone(), open.clone());
        Ok(open)
    }

    pub fn shape(&self, name: &str) -> LabResult<Vec<usize>> {
        Ok(self.file(name)?.info(name)?.shape.clone())
    }

    pub fn bf16_rows(&self, name: &str, first: usize, count: usize) -> LabResult<Vec<u16>> {
        self.file(name)?.bf16_rows(name, first, count)
    }

    pub fn bf16(&self, name: &str) -> LabResult<(Vec<usize>, Vec<u16>)> {
        self.file(name)?.bf16(name)
    }
}

/// Raw little-endian BF16 rows of `width` values.
pub fn raw_bf16(path: &Path, width: usize) -> LabResult<Vec<u16>> {
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    if bytes.len() % (2 * width) != 0 {
        return Err(format!(
            "{}: {} bytes are not rows of {width}",
            path.display(),
            bytes.len()
        ));
    }
    Ok(bytes
        .chunks_exact(2)
        .map(|b| u16::from_le_bytes([b[0], b[1]]))
        .collect())
}

/// The MiMo numerics audit's stream names, from its manifest.
pub fn audit_streams(dir: &Path) -> LabResult<Vec<String>> {
    let path = dir.join("manifest.json");
    let manifest: Value = serde_json::from_slice(
        &std::fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))?,
    )
    .map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(manifest["streams"]
        .as_array()
        .ok_or_else(|| format!("{}: missing streams", path.display()))?
        .iter()
        .filter_map(|stream| stream["name"].as_str().map(str::to_string))
        .collect())
}

/// Rank 0's rows of tensor `name` over every `step*` directory of a Qwen
/// bundle dump, in step order, as (cols, rows).
pub fn step_rows(dir: &Path, name: &str) -> LabResult<(usize, Vec<u16>)> {
    let mut steps = std::fs::read_dir(dir)
        .map_err(|e| format!("{}: {e}", dir.display()))?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| {
            path.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.starts_with("step"))
        })
        .collect::<Vec<_>>();
    steps.sort();
    let (mut cols, mut rows) = (0, Vec::new());
    for step in steps {
        let index = std::fs::read_to_string(step.join("index.jsonl")).unwrap_or_default();
        for line in index.lines() {
            let entry: Value =
                serde_json::from_str(line).map_err(|e| format!("{}: {e}", step.display()))?;
            if entry["name"] != name || entry["rank"] != 0 || entry["dtype"] != "bf16" {
                continue;
            }
            let width = entry["cols"].as_u64().ok_or("dump cols")? as usize;
            if cols != 0 && cols != width {
                return Err(format!("{name}: width changes between steps"));
            }
            cols = width;
            rows.extend(raw_bf16(&step.join(format!("{name}.r0.bin")), width)?);
        }
    }
    if cols == 0 {
        return Err(format!("{}: no rank-0 BF16 rows of {name}", dir.display()));
    }
    Ok((cols, rows))
}

/// `count` evenly spaced row indices of `rows`.
pub fn spread(rows: usize, count: usize) -> Vec<usize> {
    let count = count.min(rows);
    (0..count).map(|i| i * rows / count).collect()
}

/// Columns `[first, first + width)` of row-major [rows][cols] data.
pub fn columns(data: &[u16], cols: usize, first: usize, width: usize) -> Vec<u16> {
    data.chunks_exact(cols)
        .flat_map(|row| row[first..first + width].iter().copied())
        .collect()
}

/// Rows `rows` of row-major [..][cols] data, columns `[0, width)` of each.
pub fn block(data: &[u16], cols: usize, rows: std::ops::Range<usize>, width: usize) -> Vec<u16> {
    data[rows.start * cols..rows.end * cols]
        .chunks_exact(cols)
        .flat_map(|row| row[..width].iter().copied())
        .collect()
}

/// `count` rows of [..][cols] data, cycling through its rows, each widened
/// with zeros (or cut) to `width` columns.
pub fn rows_of(data: &[u16], cols: usize, count: usize, width: usize) -> Vec<u16> {
    let available = data.len() / cols;
    let mut result = Vec::with_capacity(count * width);
    for row in 0..count {
        let source = &data[(row % available.max(1)) * cols..][..cols];
        result.extend(source.iter().take(width).copied());
        result.extend(std::iter::repeat_n(0, width.saturating_sub(cols)));
    }
    result
}

/// `count` rows of wide-range BF16 values: random signs, significands in
/// [1, 2) and 13 binades.
pub fn wide_rows(count: usize, width: usize, seed: u64) -> Vec<u16> {
    let mut state = seed | 1;
    (0..count * width)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let binade = ((state >> 8) % 13) as i32;
            let unit = (state >> 40) as f64 / (1_u64 << 24) as f64;
            let sign = if state & 1 == 0 { 1.0 } else { -1.0 };
            f32_to_bf16((sign * (1.0 + unit) * 2_f64.powi(-binade)) as f32)
        })
        .collect()
}

/// SiLU(gate) * up of two [rows][width] BF16 projections, in f64, rounded to
/// BF16: the shared expert's down projection input.
pub fn swiglu_rows(gate: &[f64], up: &[f64]) -> Vec<u16> {
    swiglu(
        &[gate, up].concat(),
        gate.len(),
        SwigluColumns::ConcatenatedHalves,
    )
    .expect("SwiGLU rows")
    .values
    .iter()
    .map(|&v| f32_to_bf16(v as f32))
    .collect()
}

/// f64 products [rows][n] of activations [rows][k] and weights [n][k].
pub fn project(activation: &[u16], weight: &[u16], k: usize) -> Vec<f64> {
    let n = weight.len() / k;
    reference(weight, activation, n, activation.len() / k, k).values
}

/// Where a study's activations came from.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Origin {
    /// The GEMM's own input, from a dump (or derived exactly from one).
    Dump,
    /// A dumped tensor of the same kind standing in for the GEMM's input.
    Proxy,
    Synthetic,
}

impl Origin {
    pub fn label(self) -> &'static str {
        match self {
            Origin::Dump => "dump",
            Origin::Proxy => "proxy",
            Origin::Synthetic => "synthetic",
        }
    }
}

/// Cached (cols, rows) of a tensor.
type Rows = std::rc::Rc<(usize, Vec<u16>)>;

/// Qwen3.8 checkpoint weights and dumped activations: the reference runner's
/// BF16 taps of one prompt (`CHOPINFER_QWEN38_LAYER_DUMPS/<prompt>`, the
/// `taps/bf16` steps of a `tools/refrun` run) and the bundle step dumps
/// (`CHOPINFER_QWEN38_STEP_DUMPS`).
pub struct QwenSources {
    pub checkpoint: Checkpoint,
    layers: PathBuf,
    steps: Option<PathBuf>,
    cache: std::cell::RefCell<BTreeMap<String, Rows>>,
}

impl QwenSources {
    pub fn from_env(prompt: &str) -> LabResult<Option<Self>> {
        let (Some(checkpoint), Some(layers)) = (
            env_path("CHOPINFER_QWEN38_CHECKPOINT"),
            env_path("CHOPINFER_QWEN38_LAYER_DUMPS"),
        ) else {
            return Ok(None);
        };
        Ok(Some(Self {
            checkpoint: Checkpoint::open(&checkpoint)?,
            layers: layers.join(prompt),
            steps: env_path("CHOPINFER_QWEN38_STEP_DUMPS"),
            cache: Default::default(),
        }))
    }

    pub fn name(layer: usize, tail: &str) -> String {
        format!("model.language_model.layers.{layer}.{tail}")
    }

    /// A checkpoint tensor as (cols, rows).
    pub fn weight(&self, name: &str) -> LabResult<Rows> {
        if let Some(cached) = self.cache.borrow().get(name) {
            return Ok(cached.clone());
        }
        let (shape, data) = self.checkpoint.bf16(name)?;
        let value = std::rc::Rc::new((shape[1..].iter().product(), data));
        self.cache.borrow_mut().insert(name.into(), value.clone());
        Ok(value)
    }

    /// The runner's rows of `site` at `layer` (tap `L<nn>.<site>`) as
    /// (cols, rows), one row per token.
    pub fn rows(&self, layer: usize, site: &str) -> LabResult<Rows> {
        let name = format!("L{layer:02}.{site}");
        if let Some(cached) = self.cache.borrow().get(&name) {
            return Ok(cached.clone());
        }
        let value = std::rc::Rc::new(step_rows(&self.layers, &name)?);
        self.cache.borrow_mut().insert(name, value.clone());
        Ok(value)
    }

    /// The head input `final.x.hc.mixer` of every bundle step dump (an
    /// older dump's `final.u`), as (cols, rows).
    pub fn final_rows(&self) -> LabResult<Option<(usize, Vec<u16>)>> {
        self.steps
            .as_deref()
            .map(|dir| step_rows(dir, "final.x.hc.mixer").or_else(|_| step_rows(dir, "final.u")))
            .transpose()
    }
}

/// MiMo checkpoint (and DFlash drafter) weights and the numerics audit's
/// dumped rows (`CHOPINFER_MIMO_AUDIT_DIR`).
pub struct MimoSources {
    pub checkpoint: Checkpoint,
    pub drafter: Checkpoint,
    audit: PathBuf,
    cache: std::cell::RefCell<BTreeMap<String, std::rc::Rc<Vec<u16>>>>,
}

pub const MIMO_HIDDEN: usize = 4096;

impl MimoSources {
    pub fn from_env() -> LabResult<Option<Self>> {
        let (Some(checkpoint), Some(audit)) = (
            env_path("CHOPINFER_MIMO_CHECKPOINT"),
            env_path("CHOPINFER_MIMO_AUDIT_DIR"),
        ) else {
            return Ok(None);
        };
        Ok(Some(Self {
            checkpoint: Checkpoint::open(&checkpoint)?,
            drafter: Checkpoint::open(&checkpoint.join("dflash"))?,
            audit,
            cache: Default::default(),
        }))
    }

    fn cached(
        &self,
        key: &str,
        load: impl FnOnce() -> LabResult<Vec<u16>>,
    ) -> LabResult<std::rc::Rc<Vec<u16>>> {
        if let Some(cached) = self.cache.borrow().get(key) {
            return Ok(cached.clone());
        }
        let value = std::rc::Rc::new(load()?);
        self.cache.borrow_mut().insert(key.into(), value.clone());
        Ok(value)
    }

    /// One audit file over every stream, concatenated.
    pub fn audit(&self, file: &str, width: usize) -> LabResult<std::rc::Rc<Vec<u16>>> {
        self.cached(&format!("audit:{file}"), || {
            let mut rows = Vec::new();
            for stream in audit_streams(&self.audit)? {
                rows.extend(raw_bf16(&self.audit.join(&stream).join(file), width)?);
            }
            Ok(rows)
        })
    }

    /// HF RMSNorm (epsilon 1e-6, BF16 output) of hidden rows with a norm
    /// weight of the checkpoint.
    pub fn normed(&self, rows: &[u16], norm: &str) -> LabResult<Vec<u16>> {
        let (_, weight) = self.checkpoint.bf16(norm)?;
        NORM.apply_bf16(rows, &weight, MIMO_HIDDEN)
            .map_err(|e| e.0.to_string())
    }

    /// The router input of `layer`: the post-attention norm of its residual.
    pub fn router_rows(&self, layer: usize) -> LabResult<std::rc::Rc<Vec<u16>>> {
        self.cached(&format!("router:{layer}"), || {
            let residual = self.audit(&format!("layer{layer:02}.resid.bf16"), MIMO_HIDDEN)?;
            self.normed(
                &residual,
                &format!("model.layers.{layer}.post_attention_layernorm.weight"),
            )
        })
    }

    /// The final norm of `layer`'s output (the head input after layer 47).
    pub fn final_rows(&self, layer: usize) -> LabResult<std::rc::Rc<Vec<u16>>> {
        self.cached(&format!("final:{layer}"), || {
            let hidden = self.audit(&format!("layer{layer:02}.bf16"), MIMO_HIDDEN)?;
            self.normed(&hidden, "model.norm.weight")
        })
    }

    /// The DFlash fc input: the fused all-reduce's normalized output after
    /// target layers 0, 11, 23 and 35 (next layer's input norm) and the final
    /// norm after layer 47, concatenated per token.
    pub fn drafter_context(&self) -> LabResult<std::rc::Rc<Vec<u16>>> {
        self.cached("context", || {
            let mut taps = Vec::new();
            for layer in [0, 11, 23, 35] {
                let hidden = self.audit(&format!("layer{layer:02}.bf16"), MIMO_HIDDEN)?;
                taps.push(self.normed(
                    &hidden,
                    &format!("model.layers.{}.input_layernorm.weight", layer + 1),
                )?);
            }
            taps.push(self.final_rows(47)?.to_vec());
            let tokens = taps[0].len() / MIMO_HIDDEN;
            Ok((0..tokens)
                .flat_map(|t| {
                    taps.iter()
                        .flat_map(move |tap| tap[t * MIMO_HIDDEN..][..MIMO_HIDDEN].iter().copied())
                })
                .collect())
        })
    }
}

/// Output columns to score: all of them when the case is small, otherwise
/// enough evenly spread columns (at least 256) for 2^17 outputs, plus the
/// `extra` columns a caller wants covered.
pub fn study_columns(out_n: usize, m: usize, extra: &[usize]) -> Vec<usize> {
    if out_n * m <= 1 << 18 {
        return (0..out_n).collect();
    }
    let want = (1_usize << 17).div_ceil(m).max(256).min(out_n);
    let mut columns = spread(out_n, want);
    columns.extend(extra.iter().copied().filter(|&c| c < out_n));
    columns.sort_unstable();
    columns.dedup();
    columns
}

/// Pooled batches of one served entry at one M: its outputs as served and with
/// F32 output at the scored columns, the f64 reference, and the probe's
/// chain-128 scheme on the same weights, activations and outputs.
#[derive(Default)]
pub struct Study {
    served: Vec<f32>,
    f32: Vec<f32>,
    chain: Vec<f32>,
    values: Vec<f64>,
    sums: Vec<f64>,
    k: usize,
}

impl Study {
    /// Adds one batch. `weight` is the dense [n][k] weight, `rows` the batch's
    /// [m][k] activations, `served` and `f32` the entry's [m][out_n] outputs;
    /// `half` is the SwiGLU half width (gate rows first, then up rows).
    #[allow(clippy::too_many_arguments)]
    pub fn add(
        &mut self,
        session: &Session,
        probe: &Module,
        (weight, k): (&[u16], usize),
        half: Option<usize>,
        (rows, m): (&[u16], usize),
        columns: &[usize],
        served: &[f32],
        f32: &[f32],
    ) -> LabResult<()> {
        let out_n = served.len() / m;
        let weight_rows = match half {
            Some(half) => columns
                .iter()
                .copied()
                .chain(columns.iter().map(|c| c + half))
                .collect::<Vec<_>>(),
            None => columns.to_vec(),
        };
        let sub_n = weight_rows.len();
        let padded_n = sub_n.div_ceil(16) * 16;
        let padded_m = m.div_ceil(32) * 32;
        let mut sub = vec![0_u16; padded_n * k];
        for (i, &row) in weight_rows.iter().enumerate() {
            sub[i * k..(i + 1) * k].copy_from_slice(&weight[row * k..(row + 1) * k]);
        }
        let mut tokens = vec![0_u16; padded_m * k];
        tokens[..m * k].copy_from_slice(&rows[..m * k]);
        let exact = reference(&sub[..sub_n * k], &tokens[..m * k], sub_n, m, k);
        let gated = half
            .map(|_| {
                silu_glu(
                    &exact,
                    sub_n,
                    SwigluColumns::ConcatenatedHalves,
                    SiluGlu::UNCLAMPED,
                )
            })
            .transpose()
            .map_err(|e| e.to_string())?;
        let problem = Problem::upload(session, &sub, &tokens, padded_n, padded_m, k)?;
        let chain = problem.run(session, probe, Scheme::Chain(Some(128)))?;
        for t in 0..m {
            for (j, &c) in columns.iter().enumerate() {
                self.served.push(served[t * out_n + c]);
                self.f32.push(f32[t * out_n + c]);
                match &gated {
                    None => {
                        self.values.push(exact.values[t * sub_n + j]);
                        self.sums.push(exact.sum_abs_products[t * sub_n + j]);
                        self.chain.push(chain[t * padded_n + j]);
                    }
                    Some(gated) => {
                        let up_j = columns.len() + j;
                        self.values.push(gated.values[t * columns.len() + j]);
                        self.sums
                            .push(gated.sum_abs_products[t * columns.len() + j]);
                        // As the kernels' epilogue: (g / (1 + exp(-g))) * up in F32.
                        let (g, u) = (chain[t * padded_n + j], chain[t * padded_n + up_j]);
                        self.chain.push((g / (1.0 + (-g).exp())) * u);
                    }
                }
            }
        }
        self.k = k;
        Ok(())
    }

    pub fn outputs(&self) -> usize {
        self.values.len()
    }

    /// Summaries as served (contract bound of `output`), with F32 output and
    /// of the chain-128 counterpart (both against the F32 bound).
    pub fn finish(self, output: DType) -> LabResult<(Summary, Summary, Summary)> {
        let oracle = OracleOutput::new(self.values, self.sums, self.k.max(1))
            .map_err(|e| e.0.to_string())?;
        Ok((
            summarize(&self.served, &oracle, output),
            summarize(&self.f32, &oracle, DType::F32),
            summarize(&self.chain, &oracle, DType::F32),
        ))
    }
}

/// One printed and recorded row of a served-entry study.
#[allow(clippy::too_many_arguments)]
pub fn study_row(
    symbol: &str,
    m: usize,
    k: usize,
    origin: Origin,
    note: &str,
    served: &Summary,
    f32: &Summary,
    chain: &Summary,
) -> Value {
    let gain = if chain.accumulation.p9999 > 0.0 {
        f32.accumulation.p9999 / chain.accumulation.p9999
    } else {
        1.0
    };
    println!(
        "{symbol:46} M={m:5} K={k:6} {:9} n={:8} served contract {:8.2e}/{:8.2e}  f32 {:8.2e}/{:8.2e}  chain128 {:8.2e}/{:8.2e}  gain {gain:6.2}",
        origin.label(),
        served.outputs,
        served.contract.max,
        served.contract.p9999,
        f32.accumulation.max,
        f32.accumulation.p9999,
        chain.accumulation.max,
        chain.accumulation.p9999,
    );
    json!({
        "symbol": symbol,
        "m": m,
        "k": k,
        "origin": origin.label(),
        "note": note,
        "served": served.json(),
        "f32": f32.json(),
        "chain_128": chain.json(),
        "gain_p99.99": gain,
    })
}
