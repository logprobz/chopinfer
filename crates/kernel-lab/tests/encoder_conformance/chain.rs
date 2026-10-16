//! One Qwen3.8 vision block plus the merger against the exact f64 chain.
//! Per-component checks are generated in `encoder_conformance` from the
//! declarations and run the bundles' serving cubins. This chain retains its
//! per-token relative error bound of 1e-4 over synthetic weights.
//! Needs `CHOPINFER_KERNEL_GPU`; run in release.

use std::ops::Range;

use kernel_build::EntryParam;
use kernel_core::codec::{bf16_to_f32, f32_to_bf16};
use kernel_core::contract::{Argument, GeluMode, MultimodalQuery};
use kernel_core::multimodal_oracle as oracle;
use kernel_core::oracle::{GemmShape, OracleOutput};
use kernel_core::rotary::{AngleRule, Axes, ComputedTable};
use kernel_lab::launch::{self, Bindings};
use kernel_lab::{LabResult, Session, gpu_test};
use kernels_sm120::encoder::{self, Entry};
use runtime_cuda::{DeviceBuffer, Module};

const WIDTH: usize = 1152;
const HEADS: usize = 16;
const HEAD_DIM: usize = 72;
const MLP: usize = 4304;
const MLP_PADDED: usize = 4352;
const MERGED: usize = 4 * WIDTH;
const LM_WIDTH: usize = 2560;
const EPS: f32 = 1e-6;

fn entry(name: &str) -> Entry {
    let (stage, part) = match name {
        "gemm_vit_patch_v2" => (0, "linear"),
        "layernorm1152_v2" => (2, "norm"),
        "gemm_vit_qkv_v2" => (2, "qkv"),
        "rope_qkv_pairs_vit_v1" => (2, "rotary"),
        "attention_vit_full_v2" => (2, "attention"),
        "gemm_vit_proj_v2" => (2, "proj"),
        "gemm_vit_fc1_v2" => (2, "up"),
        "gemm_vit_fc2_v2" => (2, "down"),
        "gemm_merger_fc1_v2" => (5, "linear"),
        "gemm_merger_fc2_v2" => (6, "linear"),
        _ => panic!("unknown chain entry {name}"),
    };
    let declaration = &model_qwen38_contract::encoder::declarations().unwrap()[0];
    let entries = encoder::slots(declaration, &encoder::Shared).unwrap();
    let (_, entry) = entries
        .iter()
        .find(|(slot, _)| slot.stage == stage && slot.part == part)
        .unwrap();
    model_qwen38_contract::encoder::kernels()
        .unwrap()
        .iter()
        .find(|e| e.symbol == entry.symbol)
        .unwrap()
        .clone()
}

struct Tower<'a> {
    session: &'a Session,
    rows: Module,
    attention: Module,
}

impl<'a> Tower<'a> {
    fn new(session: &'a Session) -> LabResult<Self> {
        let load = |bytes: &[u8]| {
            session
                .context
                .load_module(bytes)
                .map_err(|e| e.to_string())
        };
        Ok(Self {
            session,
            rows: load(bundle_qwen38::embedded_cubin(encoder::ROWS_GEMM_CUBIN).unwrap())?,
            attention: load(bundle_qwen38::embedded_cubin(encoder::ATTENTION_CUBIN).unwrap())?,
        })
    }

    /// Launches one entry with buffers bound by parameter name (one buffer
    /// may fill two names for in-place use) and scalars by name. Unbound
    /// pointers are null.
    fn run(
        &self,
        name: &str,
        live: u32,
        buffers: &[(&str, &DeviceBuffer)],
        scalars: &[(&str, u64)],
    ) -> LabResult<()> {
        let entry = entry(name);
        let module = if entry.is_attention() {
            &self.attention
        } else {
            &self.rows
        };
        let function = module.function(&entry.symbol).map_err(|e| e.to_string())?;
        let plan = entry.plan(live).map_err(|e| e.to_string())?;
        if plan.dynamic_shared_bytes >= 48 * 1024 {
            function
                .set_max_dynamic_shared_memory(plan.dynamic_shared_bytes)
                .map_err(|e| e.to_string())?;
        }
        let scalar = |name: &str| {
            scalars
                .iter()
                .find(|(key, _)| *key == name)
                .map(|&(_, value)| value)
                .ok_or_else(|| format!("{}: missing scalar {name}", entry.symbol))
        };
        let arguments = entry
            .params
            .iter()
            .map(|param| match param {
                EntryParam::DevicePointer { name, .. } => Ok(buffers
                    .iter()
                    .position(|(key, _)| key == name)
                    .map_or(Argument::NullDevicePointer, |index| {
                        Argument::DeviceBuffer(index as u64 + 1)
                    })),
                EntryParam::U32 { name } => Ok(Argument::U32(scalar(name)? as u32)),
                EntryParam::U64 { name } => Ok(Argument::U64(scalar(name)?)),
                _ => Err(format!("{}: unexpected ABI kind", entry.symbol)),
            })
            .collect::<LabResult<Vec<_>>>()?;
        let arguments = entry.pack_arguments(arguments).map_err(|e| e.to_string())?;
        let bindings = Bindings {
            buffers: buffers
                .iter()
                .enumerate()
                .map(|(index, (_, buffer))| (index as u64 + 1, *buffer))
                .collect(),
        };
        launch::launch(
            &self.session.context,
            &function,
            &self.session.stream,
            plan,
            &arguments,
            &bindings,
        )
    }

    fn bytes(&self, bytes: &[u8]) -> LabResult<DeviceBuffer> {
        let buffer = self.session.allocate(bytes.len().max(1))?;
        buffer
            .copy_from_bytes(bytes, &self.session.stream)
            .map_err(|e| e.to_string())?;
        Ok(buffer)
    }
    fn bf16(&self, values: &[u16]) -> LabResult<DeviceBuffer> {
        self.session.upload_bf16(values)
    }
    fn f32(&self, values: &[f32]) -> LabResult<DeviceBuffer> {
        self.session.upload_f32(values)
    }
    fn i32(&self, values: &[i32]) -> LabResult<DeviceBuffer> {
        self.bytes(
            &values
                .iter()
                .flat_map(|x| x.to_ne_bytes())
                .collect::<Vec<_>>(),
        )
    }
    fn count(&self, value: usize) -> LabResult<DeviceBuffer> {
        self.bytes(&(value as u32).to_ne_bytes())
    }
    /// BF16 output storage poisoned with NaN, so an unwritten row fails.
    fn output(&self, len: usize) -> LabResult<DeviceBuffer> {
        self.bytes(&vec![0xff; len * 2])
    }
    /// F32 output storage poisoned with NaN.
    fn output_f32(&self, len: usize) -> LabResult<DeviceBuffer> {
        self.bytes(&vec![0xff; len * 4])
    }
    fn read(&self, buffer: &DeviceBuffer, len: usize) -> LabResult<Vec<u16>> {
        self.session.download_bf16(buffer, len)
    }
    fn read_f32(&self, buffer: &DeviceBuffer, len: usize) -> LabResult<Vec<f32>> {
        self.session.download_f32(buffer, len)
    }
}

/// BF16 values in [-scale, scale] from a fixed xorshift stream.
fn seeded(len: usize, seed: u64, scale: f32) -> Vec<u16> {
    let mut state = seed | 1;
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            f32_to_bf16((((state >> 40) % 513) as f32 / 256.0 - 1.0) * scale)
        })
        .collect()
}
/// F32 values in [-scale, scale] with all 24 bits populated, so the low
/// half of each BF16 pair is exercised.
fn seeded_f32(len: usize, seed: u64, scale: f32) -> Vec<f32> {
    let mut state = seed | 1;
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state >> 40) as f32 / (1u64 << 23) as f32 - 1.0) * scale
        })
        .collect()
}
fn decoded(values: &[u16]) -> Vec<f64> {
    values.iter().map(|&x| f64::from(bf16_to_f32(x))).collect()
}
fn widen(values: &[f32]) -> Vec<f64> {
    values.iter().map(|&x| f64::from(x)).collect()
}
/// The pair values (hi + lo) of tensor `tensor` (0 q, 1 k, 2 v) in packed
/// pair rows.
fn plane_values(rows: &[u16], tensor: usize) -> Vec<f64> {
    rows.chunks_exact(6 * WIDTH)
        .flat_map(|row| {
            let hi = &row[2 * tensor * WIDTH..][..WIDTH];
            let lo = &row[(2 * tensor + 1) * WIDTH..][..WIDTH];
            hi.iter()
                .zip(lo)
                .map(|(&hi, &lo)| f64::from(bf16_to_f32(hi)) + f64::from(bf16_to_f32(lo)))
                .collect::<Vec<_>>()
        })
        .collect()
}
fn rows_of<T: Copy>(values: &[T], width: usize, rows: &[usize]) -> Vec<T> {
    rows.iter()
        .flat_map(|&row| values[row * width..(row + 1) * width].iter().copied())
        .collect()
}
/// The largest per-token relative L2 error, which must stay within `limit`.
fn relative(
    name: &str,
    actual: &[f64],
    reference: &[f64],
    width: usize,
    limit: f64,
) -> LabResult<()> {
    let mut worst = (0, 0.0_f64);
    for (token, (a, r)) in actual
        .chunks_exact(width)
        .zip(reference.chunks_exact(width))
        .enumerate()
    {
        let error = a
            .iter()
            .zip(r)
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f64>()
            .sqrt();
        let norm = r.iter().map(|y| y * y).sum::<f64>().sqrt();
        let ratio = if norm == 0.0 { error } else { error / norm };
        if !ratio.is_finite() || ratio > worst.1 || ratio.is_nan() {
            worst = (token, ratio);
        }
    }
    eprintln!(
        "{name}: worst token relative error {:.3e} (token {})",
        worst.1, worst.0
    );
    if !worst.1.is_finite() || worst.1 > limit {
        return Err(format!(
            "{name}: token {} relative error {} above {limit}",
            worst.0, worst.1
        ));
    }
    Ok(())
}

/// Runs `work` over `count` items split into contiguous ranges, one per
/// available CPU, and concatenates the results in order.
fn parallel<T: Send>(count: usize, work: impl Fn(Range<usize>) -> Vec<T> + Sync) -> Vec<T> {
    let threads = std::thread::available_parallelism().map_or(8, |n| n.get());
    let chunk = count.div_ceil(threads).max(1);
    std::thread::scope(|scope| {
        let workers = (0..count)
            .step_by(chunk)
            .map(|start| {
                let work = &work;
                scope.spawn(move || work(start..(start + chunk).min(count)))
            })
            .collect::<Vec<_>>();
        workers
            .into_iter()
            .flat_map(|worker| worker.join().unwrap())
            .collect()
    })
}

fn concat(parts: Vec<OracleOutput>) -> OracleOutput {
    let mut out = OracleOutput {
        values: Vec::new(),
        sum_abs_products: Vec::new(),
        reduction_lengths: Vec::new(),
    };
    for part in parts {
        out.values.extend(part.values);
        out.sum_abs_products.extend(part.sum_abs_products);
        out.reduction_lengths.extend(part.reduction_lengths);
    }
    out
}

/// The f64 encoder GEMM oracle over rows of `a`, in parallel row blocks,
/// with an optional residual of the same rows.
fn gemm_rows(
    a: &[f64],
    b: &[f64],
    (n, k): (usize, usize),
    bias: &[f64],
    gelu: Option<GeluMode>,
    residual: Option<&[f64]>,
) -> OracleOutput {
    let m = a.len() / k;
    let parts = parallel(m, |rows| {
        vec![
            oracle::gemm_bias_epilogue(
                &a[rows.start * k..rows.end * k],
                b,
                GemmShape {
                    m: rows.len(),
                    n,
                    k,
                    experts: 1,
                },
                Some(bias),
                gelu,
                None,
                residual.map(|x| &x[rows.start * n..rows.end * n]),
            )
            .expect("GEMM oracle"),
        ]
    });
    concat(parts)
}

/// The attention oracle for the query tokens in `rows`, in parallel.
fn attention_rows(
    query: MultimodalQuery,
    rows: &[usize],
    q: &[f64],
    k: &[f64],
    v: &[f64],
    cu_seqlens: &[i32],
    scale: f64,
) -> OracleOutput {
    let width = HEADS * HEAD_DIM;
    let parts = parallel(rows.len(), |range| {
        let selected = &rows[range];
        let q_rows = rows_of(q, width, selected);
        vec![
            oracle::contiguous_attention_rows(
                query, selected, &q_rows, k, v, cu_seqlens, scale, None,
            )
            .expect("attention oracle"),
        ]
    });
    concat(parts)
}

/// 2D rotary rows [T,36] for images in raster order: 18 height and 18 width
/// frequencies of the 36-dim rotary embedding (theta 10000), as HF builds
/// them before duplicating the halves for NeoX pairing.
fn rotary(grids: &[(usize, usize)]) -> (Vec<f32>, Vec<f32>) {
    let table = ComputedTable {
        theta: 1e4,
        rule: AngleRule::MulRecipPowF32,
        axes: Axes::Axial2d,
    };
    let (mut cos, mut sin) = (Vec::new(), Vec::new());
    for &(height, width) in grids {
        for row in 0..height {
            for col in 0..width {
                for (c, s) in table.row([row, col, 0], 36).expect("axial F32 table") {
                    cos.push(c as f32);
                    sin.push(s as f32);
                }
            }
        }
    }
    (cos, sin)
}

fn vit_attention_query() -> MultimodalQuery {
    entry("attention_vit_full_v2").query
}

/// One tower block and the merger in f64 from the same F32 input and BF16
/// weights, never rounding. Returns every checked stage.
struct Chain {
    ln1: Vec<f64>,
    qkv: Vec<f64>,
    q: Vec<f64>,
    k: Vec<f64>,
    attention: Vec<f64>,
    residual1: Vec<f64>,
    ln2: Vec<f64>,
    fc1: Vec<f64>,
    block: Vec<f64>,
    merger_fc1: Vec<f64>,
    merged: Vec<f64>,
}

struct Weights {
    ln1: [Vec<u16>; 2],
    qkv: [Vec<u16>; 2],
    proj: [Vec<u16>; 2],
    ln2: [Vec<u16>; 2],
    /// Unpadded [4304,1152] and [1152,4304] as in the checkpoint.
    fc1: [Vec<u16>; 2],
    fc2: [Vec<u16>; 2],
    merger_ln: [Vec<u16>; 2],
    merger_fc1: [Vec<u16>; 2],
    merger_fc2: [Vec<u16>; 2],
}

fn norm(seed: u64, width: usize) -> [Vec<u16>; 2] {
    [
        seeded(width, seed, 1.0)
            .iter()
            .map(|&v| f32_to_bf16(1.0 + bf16_to_f32(v) / 4.0))
            .collect(),
        seeded(width, seed + 1, 0.1),
    ]
}
fn linear(seed: u64, n: usize, k: usize) -> [Vec<u16>; 2] {
    [
        seeded(n * k, seed, 1.7 / (k as f32).sqrt()),
        seeded(n, seed + 1, 0.1),
    ]
}

impl Weights {
    fn new() -> Self {
        Self {
            ln1: norm(0x5157_5010, WIDTH),
            qkv: linear(0x5157_5020, 3 * WIDTH, WIDTH),
            proj: linear(0x5157_5030, WIDTH, WIDTH),
            ln2: norm(0x5157_5040, WIDTH),
            fc1: linear(0x5157_5050, MLP, WIDTH),
            fc2: linear(0x5157_5060, WIDTH, MLP),
            merger_ln: norm(0x5157_5070, WIDTH),
            merger_fc1: linear(0x5157_5080, MERGED, MERGED),
            merger_fc2: linear(0x5157_5090, LM_WIDTH, MERGED),
        }
    }

    /// The weight plan's padding: fc1 gains zero rows and zero bias to
    /// 4352, fc2 gains zero columns to 4352.
    fn padded_mlp(&self) -> [Vec<u16>; 3] {
        let mut fc1 = self.fc1[0].clone();
        fc1.resize(MLP_PADDED * WIDTH, 0);
        let mut bias = self.fc1[1].clone();
        bias.resize(MLP_PADDED, 0);
        let fc2 = self.fc2[0]
            .chunks_exact(MLP)
            .flat_map(|row| {
                row.iter()
                    .copied()
                    .chain(std::iter::repeat_n(0, MLP_PADDED - MLP))
            })
            .collect();
        [fc1, bias, fc2]
    }
}

fn f64_chain(
    weights: &Weights,
    hidden: &[f64],
    cos: &[f64],
    sin: &[f64],
    cu_seqlens: &[i32],
) -> Chain {
    let d = |values: &[u16]| decoded(values);
    let ln = |x: &[f64], w: &[Vec<u16>; 2]| {
        oracle::layernorm_affine(x, WIDTH, &d(&w[0]), &d(&w[1]), f64::from(EPS))
            .expect("LayerNorm oracle")
            .values
    };
    let dense = |x: &[f64], w: &[Vec<u16>; 2], n: usize, k: usize, gelu: Option<GeluMode>| {
        gemm_rows(x, &d(&w[0]), (n, k), &d(&w[1]), gelu, None).values
    };
    let add = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(x, y)| x + y).collect::<Vec<_>>();
    let tokens = hidden.len() / WIDTH;
    let ln1 = ln(hidden, &weights.ln1);
    let qkv = dense(&ln1, &weights.qkv, 3 * WIDTH, WIDTH, None);
    let [q, k, v] = oracle::rope_qkv_pairs(entry("rope_qkv_pairs_vit_v1").query, &qkv, cos, sin)
        .expect("rotary oracle")
        .map(|output| output.values);
    let rows = (0..tokens).collect::<Vec<_>>();
    let attention = attention_rows(
        vit_attention_query(),
        &rows,
        &q,
        &k,
        &v,
        cu_seqlens,
        f64::from((HEAD_DIM as f32).powf(-0.5)),
    )
    .values;
    let residual1 = add(
        hidden,
        &dense(&attention, &weights.proj, WIDTH, WIDTH, None),
    );
    let ln2 = ln(&residual1, &weights.ln2);
    let fc1 = dense(&ln2, &weights.fc1, MLP, WIDTH, Some(GeluMode::Tanh));
    let block = add(&residual1, &dense(&fc1, &weights.fc2, WIDTH, MLP, None));
    let merger_ln = ln(&block, &weights.merger_ln);
    let merger_fc1 = dense(
        &merger_ln,
        &weights.merger_fc1,
        MERGED,
        MERGED,
        Some(GeluMode::Erf),
    );
    let merged = dense(&merger_fc1, &weights.merger_fc2, LM_WIDTH, MERGED, None);
    Chain {
        ln1,
        qkv,
        q,
        k,
        attention,
        residual1,
        ln2,
        fc1,
        block,
        merger_fc1,
        merged,
    }
}

#[test]
fn qwen38_vit_block_chain_gpu() {
    // Two images (16 x 16 and 12 x 12 patches, 400 rows) through one block
    // (LN, qkv, rotary into pairs, attention, proj into the residual, LN,
    // fc1 with tanh GELU, fc2 into the residual) and the merger (LN, 4 rows
    // viewed as one 4608 row, fc1 with erf GELU, fc2 to BF16),
    // device-resident, against the exact f64 chain.
    gpu_test(1 << 30, |session| {
        let tower = Tower::new(session)?;
        let weights = Weights::new();
        let grids = [(16, 16), (12, 12)];
        let tokens: usize = grids.iter().map(|(h, w)| h * w).sum();
        let cu_seqlens = [0, 256, tokens as i32];
        let hidden = seeded_f32(tokens * WIDTH, 0x5157_5001, 2.0);
        let (cos, sin) = rotary(&grids);

        let up = |values: &[u16]| tower.bf16(values);
        let pair =
            |w: &[Vec<u16>; 2]| -> LabResult<[DeviceBuffer; 2]> { Ok([up(&w[0])?, up(&w[1])?]) };
        let [fc1_padded, fc1_bias, fc2_padded] = weights.padded_mlp();
        let (ln1, qkv_w, proj, ln2) = (
            pair(&weights.ln1)?,
            pair(&weights.qkv)?,
            pair(&weights.proj)?,
            pair(&weights.ln2)?,
        );
        let (fc1_w, fc1_b, fc2_w, fc2_b) = (
            up(&fc1_padded)?,
            up(&fc1_bias)?,
            up(&fc2_padded)?,
            up(&weights.fc2[1])?,
        );
        let (merger_ln, merger_fc1, merger_fc2) = (
            pair(&weights.merger_ln)?,
            pair(&weights.merger_fc1)?,
            pair(&weights.merger_fc2)?,
        );
        let h = tower.f32(&hidden)?;
        let (cos_buffer, sin_buffer) = (tower.f32(&cos)?, tower.f32(&sin)?);
        let cu_buffer = tower.i32(&cu_seqlens)?;
        let live = tower.count(tokens)?;
        let merged_live = tower.count(tokens / 4)?;
        let rows = tokens as u32;
        let x = tower.output_f32(tokens * WIDTH)?;
        let qkv = tower.output_f32(tokens * 3 * WIDTH)?;
        let attn = tower.output_f32(tokens * WIDTH)?;
        let a1 = tower.output_f32(tokens * MLP_PADDED)?;
        let z = tower.output_f32(tokens / 4 * MERGED)?;
        let merged = tower.output(tokens / 4 * LM_WIDTH)?;
        let eps = [("eps_bits", u64::from(EPS.to_bits()))];
        let gemm = |name: &str,
                    live_rows: u32,
                    count: &DeviceBuffer,
                    x: &DeviceBuffer,
                    w: &DeviceBuffer,
                    b: &DeviceBuffer,
                    out: &DeviceBuffer| {
            tower.run(
                name,
                live_rows,
                &[
                    ("activation", x),
                    ("weight", w),
                    ("bias", b),
                    ("output", out),
                    ("live_rows", count),
                ],
                &[],
            )
        };
        let layernorm = |input: &DeviceBuffer, w: &[DeviceBuffer; 2]| {
            tower.run(
                "layernorm1152_v2",
                rows,
                &[
                    ("input", input),
                    ("weight", &w[0]),
                    ("bias", &w[1]),
                    ("output", &x),
                    ("live_rows", &live),
                ],
                &eps,
            )
        };
        let read = |buffer: &DeviceBuffer, len: usize| -> LabResult<Vec<f64>> {
            Ok(widen(&tower.read_f32(buffer, len)?))
        };
        layernorm(&h, &ln1)?;
        let gpu_ln1 = read(&x, tokens * WIDTH)?;
        gemm(
            "gemm_vit_qkv_v2",
            rows,
            &live,
            &x,
            &qkv_w[0],
            &qkv_w[1],
            &qkv,
        )?;
        let gpu_qkv = read(&qkv, tokens * 3 * WIDTH)?;
        tower.run(
            "rope_qkv_pairs_vit_v1",
            rows,
            &[
                ("rows", &qkv),
                ("cos", &cos_buffer),
                ("sin", &sin_buffer),
                ("live_rows", &live),
            ],
            &[],
        )?;
        let pairs = tower.read(&qkv, tokens * 6 * WIDTH)?;
        tower.run(
            "attention_vit_full_v2",
            rows,
            &[
                ("qkv", &qkv),
                ("cu_seqlens", &cu_buffer),
                ("output", &attn),
                ("live_tokens", &live),
            ],
            &[
                ("live_sequences", 2),
                (
                    "scale_bits",
                    u64::from((HEAD_DIM as f32).powf(-0.5).to_bits()),
                ),
            ],
        )?;
        gemm(
            "gemm_vit_proj_v2",
            rows,
            &live,
            &attn,
            &proj[0],
            &proj[1],
            &h,
        )?;
        let gpu_residual1 = read(&h, tokens * WIDTH)?;
        layernorm(&h, &ln2)?;
        let gpu_ln2 = read(&x, tokens * WIDTH)?;
        gemm("gemm_vit_fc1_v2", rows, &live, &x, &fc1_w, &fc1_b, &a1)?;
        gemm("gemm_vit_fc2_v2", rows, &live, &a1, &fc2_w, &fc2_b, &h)?;
        let gpu_block = read(&h, tokens * WIDTH)?;
        // The merger: LayerNorm per patch row, then [T/4, 4608] is a view.
        layernorm(&h, &merger_ln)?;
        let quarter = rows / 4;
        gemm(
            "gemm_merger_fc1_v2",
            quarter,
            &merged_live,
            &x,
            &merger_fc1[0],
            &merger_fc1[1],
            &z,
        )?;
        gemm(
            "gemm_merger_fc2_v2",
            quarter,
            &merged_live,
            &z,
            &merger_fc2[0],
            &merger_fc2[1],
            &merged,
        )?;

        let started = std::time::Instant::now();
        let chain = f64_chain(
            &weights,
            &widen(&hidden),
            &widen(&cos),
            &widen(&sin),
            &cu_seqlens,
        );
        eprintln!("f64 chain: {:.1}s", started.elapsed().as_secs_f64());

        // Each stage within 1e-4 of the exact chain per token; the BF16
        // tower this replaces was ~1e-2.
        let limit = 1e-4;
        relative("layernorm1", &gpu_ln1, &chain.ln1, WIDTH, limit)?;
        relative("qkv", &gpu_qkv, &chain.qkv, 3 * WIDTH, limit)?;
        relative("rotary q", &plane_values(&pairs, 0), &chain.q, WIDTH, limit)?;
        relative("rotary k", &plane_values(&pairs, 1), &chain.k, WIDTH, limit)?;
        relative(
            "attention",
            &read(&attn, tokens * WIDTH)?,
            &chain.attention,
            WIDTH,
            limit,
        )?;
        relative("residual1", &gpu_residual1, &chain.residual1, WIDTH, limit)?;
        relative("layernorm2", &gpu_ln2, &chain.ln2, WIDTH, limit)?;
        let fc1 = tower.read_f32(&a1, tokens * MLP_PADDED)?;
        if fc1
            .chunks_exact(MLP_PADDED)
            .any(|row| row[MLP..].iter().any(|&x| x != 0.0))
        {
            return Err("fc1 padding columns are not exactly zero".into());
        }
        let fc1 = fc1
            .chunks_exact(MLP_PADDED)
            .flat_map(|row| widen(&row[..MLP]))
            .collect::<Vec<_>>();
        relative("fc1", &fc1, &chain.fc1, MLP, limit)?;
        relative("block", &gpu_block, &chain.block, WIDTH, limit)?;
        relative(
            "merger fc1",
            &read(&z, tokens / 4 * MERGED)?,
            &chain.merger_fc1,
            MERGED,
            limit,
        )?;
        // The merged rows are BF16: half a unit is 2^-9 relative.
        relative(
            "merged",
            &decoded(&tower.read(&merged, tokens / 4 * LM_WIDTH)?),
            &chain.merged,
            LM_WIDTH,
            2_f64.powi(-8),
        )?;
        Ok(())
    })
    .expect("Qwen3.8 vision block chain");
}
