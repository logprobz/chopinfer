//! QSA block selections of a bundle dump against the selections recomputed
//! from the dump's own BF16 projection rows, to separate why a device's key
//! set differs.
//!
//! A policy fixes the indexer operands and the score arithmetic:
//!
//! - `engine`: the engine's operands. `norm_rotate` and `write_block` of
//!   `sparse_indexer.cuh` emulated in their F32 order (per-lane F32 FMA
//!   square sums, the butterfly reduction, `fma(sum, 1/128, eps)`, IEEE
//!   sqrt and reciprocal, `(x * inverse) * (1 + w)`, the table rotation of
//!   `RotaryArithmetic::F32Fma`, BF16 storage), on the engine's F32 cos/sin
//!   table, scored in f64. This is what an exact-arithmetic top-k kernel
//!   selects from the engine's operands.
//! - `kernel`: `engine` scored as `block_scores` does: BF16 MMA k16 steps
//!   added to two F32 accumulators (even and odd steps), their F32 sum, the
//!   F32 ReLU sum `(h0 + h1) + (h2 + h3)` and the F32 scale. Each k16 step is
//!   rounded once into its accumulator; the tensor core's internal alignment
//!   is not modeled, so this is an emulation of the kernel, not its bits.
//! - `hf-style`: an approximation of HF Transformers 5.16.1's
//!   `Qwen4ExpTextQSAIndexer` on BF16 tensors. It keeps HF's BF16 storage
//!   points (the norm output, BF16 RoPE with HF's BF16 cos and sin, the
//!   pooled mean) and scores in F32, but its reductions are chosen
//!   emulations, not Torch's: the norm's mean sums F32 squares in f64 and
//!   rounds once, the reciprocal square root is f64 then F32, the dot
//!   products are sequential F32 FMA, and ties break to the lower block,
//!   which `torch.topk` does not promise. Its sets are not HF's bits.
//!
//! Every policy keeps the 512 highest scores, ties to the lower block.

use serde::Serialize;

use super::LAYERS;
use super::trace::Tensor;
use super::weights::Checkpoint;
use crate::Result;
use kernel_core::codec::f32_to_bf16;
use kernel_core::hf_rope;
use kernel_core::oracle::bf16_round;
use kernel_core::rotary::RotaryArithmetic;
use model_qwen38_contract::contract;

/// Tokens per QSA block.
pub const BLOCK: usize = contract::QSA_BLOCK as usize;
/// Blocks a sparse row keeps.
pub const TOP_BLOCKS: usize = contract::TOP_BLOCKS as usize;
const INDEX_HEADS: usize = contract::INDEX_HEADS as usize;
const INDEX_DIM: usize = contract::INDEX_DIM as usize;
const ROTARY_DIM: usize = contract::ROTARY as usize;
const THETA: f32 = contract::ROPE_THETA;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Policy {
    #[default]
    Engine,
    Kernel,
    #[serde(rename = "hf-style")]
    HfStyle,
}

impl Policy {
    pub fn name(self) -> &'static str {
        match self {
            Policy::Engine => "engine",
            Policy::Kernel => "kernel",
            Policy::HfStyle => "hf-style",
        }
    }
}

/// Exported rows include every row whose `engine` relative gap is below this.
const NEAR_TIE: f64 = 1e-5;
/// Differing rows kept as examples per pair and layer.
const EXAMPLES: usize = 8;

#[derive(Clone, Debug, Default, Serialize)]
pub struct PairStats {
    /// None for a device's selection.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub candidate: Option<Policy>,
    pub reference: Policy,
    /// Rows whose kept sets differ.
    pub rows_differ: usize,
    /// Blocks kept by the reference and dropped by the candidate, summed.
    pub swaps: usize,
    pub max_swaps: usize,
    /// The reference's relative gap (512th - 513th) / 512th of each
    /// differing row: smallest, median and largest.
    pub gap_min: f64,
    pub gap_median: f64,
    pub gap_max: f64,
    /// (row, swaps, reference relative gap) of the first differing rows.
    pub examples: Vec<(usize, usize, f64)>,
    #[serde(skip)]
    gaps: Vec<f64>,
}

/// Writes the `engine` operands and key sets of some compared rows for
/// kernel-lab's `qsa_f32_selection_on_oracle_operands`: every row where the
/// device's set, `engine`'s and the emulated kernel's are not all equal,
/// every row whose `engine` relative gap is below `NEAR_TIE`, and every
/// compared sparse row r with r % every == 0. Per full-attention layer NN,
/// under `dir`:
///
/// - `layerNN.qk.bf16`: every row's BF16 projection [rows, 5 * 128], the
///   input of `indexer_query` and `indexer_key_write`;
/// - `layerNN.norm.bf16`: the query and key norm weights, [2, 128];
/// - `layerNN.keys.bf16`: the emulated block keys [blocks, 128], BF16 bits,
///   little endian;
/// - `layerNN.q.bf16`: the exported rows' emulated queries [rows, 4 * 128];
/// - `layerNN.json`: the table, `rows`, and per row the kept blocks of the
///   f64 scores (`exact`) and of the emulated kernel (`kernel`) of those
///   operands and the f64 relative gap (`gap`).
#[derive(Debug)]
pub struct Export {
    pub dir: std::path::PathBuf,
    pub every: usize,
}

/// The F32 cos/sin table the engine's indexer reads.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum EngineTable {
    /// HF's BF16 values (`hf_rope::cos_sin`), served since the HF RoPE
    /// adoption.
    #[default]
    Hf,
    /// F32 cos and sin of f64 angles `position * theta^(-2j / 64)`, served
    /// before it.
    F64Angle,
}

impl EngineTable {
    fn cos_sin(self, position: usize) -> Vec<(f64, f64)> {
        let theta = f64::from(THETA);
        (0..ROTARY_DIM / 2)
            .map(|pair| match self {
                EngineTable::Hf => hf_rope::cos_sin(position, pair, theta),
                EngineTable::F64Angle => {
                    let angle =
                        position as f64 * theta.powf(-2.0 * pair as f64 / ROTARY_DIM as f64);
                    (f32r(angle.cos()), f32r(angle.sin()))
                }
            })
            .collect()
    }
}

/// `norm_rotate` of `sparse_indexer.cuh` on one 128-wide head of BF16
/// values, in the device's F32 order (lane l holds dimensions 4l to 4l + 3),
/// then BF16 storage.
fn engine_norm_rotate(x: &[f64], weight: &[f64], table: &[(f64, f64)]) -> Vec<f64> {
    let x = x.iter().map(|&v| v as f32).collect::<Vec<_>>();
    let mut lanes = [0.0_f32; 32];
    for (lane, v) in lanes.iter_mut().zip(x.chunks_exact(4)) {
        let mut square = v[0] * v[0];
        for &e in &v[1..] {
            square = e.mul_add(e, square);
        }
        *lane = square;
    }
    for offset in [16, 8, 4, 2, 1] {
        let before = lanes;
        for (lane, value) in lanes.iter_mut().enumerate() {
            *value = before[lane] + before[lane ^ offset];
        }
    }
    let eps = f32::from_bits(super::EPS_BITS);
    let inverse = 1.0_f32 / lanes[0].mul_add(1.0 / INDEX_DIM as f32, eps).sqrt();
    let y = x
        .iter()
        .zip(weight)
        .map(|(&v, &w)| (v * inverse) * (1.0 + w as f32))
        .collect::<Vec<_>>();
    let rotate = |x: f32, y: f32, (cos, sin): (f64, f64), second| {
        RotaryArithmetic::F32Fma.element(f64::from(x), f64::from(y), cos, sin, second) as f32
    };
    let half = ROTARY_DIM / 2;
    let mut out = y.clone();
    for pair in 0..half {
        out[pair] = rotate(y[pair], y[pair + half], table[pair], false);
        out[pair + half] = rotate(y[pair + half], y[pair], table[pair], true);
    }
    out.into_iter().map(|v| bf16_round(f64::from(v))).collect()
}

/// The engine's BF16 queries and block keys of BF16 projection rows:
/// `indexer_query`, and `write_block`'s pooled mean (the f64 sum in
/// position order times 1/4, rounded to F32 and then BF16) through the same
/// norm and rotation at each block's first token.
fn engine_operands(
    qk: &[f64],
    q_norm: &[f64],
    k_norm: &[f64],
    table: EngineTable,
) -> (Vec<f64>, Vec<f64>) {
    let rows = qk.len() / (WIDTH + INDEX_DIM);
    let q = super::linalg::par_map(rows, |row| {
        let values = &qk[row * (WIDTH + INDEX_DIM)..][..WIDTH];
        let cos_sin = table.cos_sin(row);
        values
            .chunks_exact(INDEX_DIM)
            .flat_map(|head| engine_norm_rotate(head, q_norm, &cos_sin))
            .collect::<Vec<_>>()
    })
    .concat();
    let keys = super::linalg::par_map(rows / BLOCK, |block| {
        let raw = |r: usize| &qk[(block * BLOCK + r) * (WIDTH + INDEX_DIM) + WIDTH..][..INDEX_DIM];
        let mean = (0..INDEX_DIM)
            .map(|d| {
                let sum = (0..BLOCK).fold(0.0, |s, r| s + raw(r)[d]);
                bf16_round(sum * (1.0 / BLOCK as f64))
            })
            .collect::<Vec<_>>();
        engine_norm_rotate(&mean, k_norm, &table.cos_sin(block * BLOCK))
    })
    .concat();
    (q, keys)
}
const WIDTH: usize = INDEX_HEADS * INDEX_DIM;
/// `sum_h relu(q_h . k) / sqrt(128)` in f64, bit for bit as
/// kernel_core's `block_scores`.
fn exact_scores(q: &[f64], keys: &[f64]) -> Vec<f64> {
    let scale = (INDEX_DIM as f64).sqrt().recip();
    keys.chunks_exact(INDEX_DIM)
        .map(|key| {
            let mut score = 0.0;
            for head in q.chunks_exact(INDEX_DIM) {
                let dot = head.iter().zip(key).map(|(a, b)| a * b).sum::<f64>();
                score += dot.max(0.0);
            }
            score * scale
        })
        .collect()
}

fn f32r(x: f64) -> f64 {
    f64::from(x as f32)
}

/// `block_scores`' F32 arithmetic on BF16 operands (see the module docs).
fn kernel_scores(q: &[f64], keys: &[f64]) -> Vec<f64> {
    let scale = f32r((INDEX_DIM as f64).sqrt().recip());
    keys.chunks_exact(INDEX_DIM)
        .map(|key| {
            let mut relu = [0.0; INDEX_HEADS];
            for (head, r) in q.chunks_exact(INDEX_DIM).zip(&mut relu) {
                let mut acc = [0.0; 2];
                for (step, (a, b)) in head.chunks_exact(16).zip(key.chunks_exact(16)).enumerate() {
                    let products = a.iter().zip(b).map(|(x, y)| x * y).sum::<f64>();
                    acc[step % 2] = f32r(acc[step % 2] + products);
                }
                let x = f32r(acc[0] + acc[1]);
                *r = if x > 0.0 { x } else { 0.0 };
            }
            let sum = f32r(f32r(relu[0] + relu[1]) + f32r(relu[2] + relu[3]));
            f32r(sum * scale)
        })
        .collect()
}

/// F32 scores as HF computes them: `matmul(q.float(), k.float().T)` as
/// sequential FMA, ReLU, the head sum and the division by sqrt(128).
fn fma_scores(q: &[f64], keys: &[f64]) -> Vec<f64> {
    let divisor = f32r((INDEX_DIM as f64).sqrt());
    keys.chunks_exact(INDEX_DIM)
        .map(|key| {
            let mut sum = 0.0;
            for head in q.chunks_exact(INDEX_DIM) {
                let mut dot = 0.0;
                for (a, b) in head.iter().zip(key) {
                    // F32 products are exact in f64: one rounding, as an FMA.
                    dot = f32r(dot + a * b);
                }
                sum = f32r(sum + dot.max(0.0));
            }
            f32r(sum / divisor)
        })
        .collect()
}

/// An approximation of HF's `Qwen4ExpTextRMSNorm` of one BF16 head:
/// `x * rsqrt(mean + eps) * (1 + w)` with F32 roundings at HF's operation
/// boundaries and BF16 output. The mean's sum (f64 of F32 squares, rounded
/// once) and the reciprocal square root (f64, then F32) are not Torch's
/// reductions.
fn hf_norm(x: &[f64], weight: &[f64]) -> Vec<f64> {
    let squares = x.iter().map(|v| f32r(v * v)).sum::<f64>();
    let mean = f32r(f32r(squares) / x.len() as f64);
    let eps = super::EPS;
    let inverse = f32r(1.0 / f32r(mean + eps).sqrt());
    x.iter()
        .zip(weight)
        .map(|(v, w)| bf16_round(f32r(f32r(v * inverse) * f32r(1.0 + w))))
        .collect()
}

/// HF's BF16 cos and sin of position p for the 32 rotary pairs
/// (kernel_core's `hf_rope`: captured F32 frequencies, the F32 angle, then
/// BF16).
fn hf_cos_sin(position: usize) -> Vec<(f64, f64)> {
    (0..ROTARY_DIM / 2)
        .map(|pair| hf_rope::cos_sin(position, pair, f64::from(THETA)))
        .collect()
}

/// HF's `apply_rotary_pos_emb` on one BF16 head: `q * cos + rotate_half(q) *
/// sin` over the first 64 dimensions, each product and the sum rounded to
/// BF16 (`hf_rope::rotate_pure_hf`).
fn hf_rotate(x: &mut [f64], table: &[(f64, f64)]) {
    let half = ROTARY_DIM / 2;
    let original = x[..ROTARY_DIM].to_vec();
    for i in 0..ROTARY_DIM {
        let (pair, second) = (i % half, i >= half);
        let mate = original[if second { i - half } else { i + half }];
        x[i] = hf_rope::rotate_pure_hf(original[i], mate, table[pair], second);
    }
}

/// HF's queries and block keys of BF16 projection rows.
fn hf_operands(qk: &[f64], q_norm: &[f64], k_norm: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let rows = qk.len() / (WIDTH + INDEX_DIM);
    let mut q = Vec::with_capacity(rows * WIDTH);
    for (row, values) in qk.chunks_exact(WIDTH + INDEX_DIM).enumerate() {
        let table = hf_cos_sin(row);
        for head in values[..WIDTH].chunks_exact(INDEX_DIM) {
            let mut head = hf_norm(head, q_norm);
            hf_rotate(&mut head, &table);
            q.extend(head);
        }
    }
    let mut keys = Vec::with_capacity(rows / BLOCK * INDEX_DIM);
    for block in 0..rows / BLOCK {
        let raw = |r: usize| &qk[(block * BLOCK + r) * (WIDTH + INDEX_DIM) + WIDTH..][..INDEX_DIM];
        let mean = (0..INDEX_DIM)
            .map(|d| {
                let sum = (0..BLOCK).fold(0.0, |s, r| f32r(s + raw(r)[d]));
                bf16_round(f32r(sum / BLOCK as f64))
            })
            .collect::<Vec<_>>();
        let mut key = hf_norm(&mean, k_norm);
        hf_rotate(&mut key, &hf_cos_sin(block * BLOCK));
        keys.extend(key);
    }
    (q, keys)
}

fn scores(policy: Policy, q: &[f64], keys: &[f64]) -> Vec<f64> {
    match policy {
        Policy::Engine => exact_scores(q, keys),
        Policy::Kernel => kernel_scores(q, keys),
        Policy::HfStyle => fma_scores(q, keys),
    }
}

/// The kept blocks of a row with more than `TOP_BLOCKS` complete blocks, in
/// ascending order: the highest scores, ties to the lower block (the order
/// of kernel_core's `block_topk`), with the scores and indices of the last
/// kept and the first dropped block.
pub fn top_blocks(scores: &[f64]) -> Result<(Vec<usize>, [f64; 2], [usize; 2])> {
    if scores.iter().any(|s| s.is_nan()) {
        return Err("block top-k refuses NaN scores".into());
    }
    let order = |&a: &usize, &b: &usize| scores[b].total_cmp(&scores[a]).then(a.cmp(&b));
    let mut blocks = (0..scores.len()).collect::<Vec<_>>();
    blocks.select_nth_unstable_by(TOP_BLOCKS - 1, order);
    let kept = blocks[TOP_BLOCKS - 1];
    let dropped = *blocks[TOP_BLOCKS..]
        .iter()
        .min_by(|a, b| order(a, b))
        .expect("a dropped block");
    blocks.truncate(TOP_BLOCKS);
    blocks.sort_unstable();
    Ok((blocks, [scores[kept], scores[dropped]], [kept, dropped]))
}

/// (kept - dropped) / kept; 0 when both are 0.
fn relative_gap(boundary: [f64; 2]) -> f64 {
    if boundary[0] > 0.0 {
        (boundary[0] - boundary[1]) / boundary[0]
    } else {
        0.0
    }
}

/// Blocks in `reference` missing from `candidate`, both ascending.
fn swaps(candidate: &[usize], reference: &[usize]) -> usize {
    let (mut i, mut missing) = (0, 0);
    for &block in reference {
        while i < candidate.len() && candidate[i] < block {
            i += 1;
        }
        if i == candidate.len() || candidate[i] != block {
            missing += 1;
        }
    }
    missing
}
/// The rows of one layer an export holds, with per row the kept blocks of
/// the f64 scores of the `engine` operands, those of the emulated kernel and
/// the f64 relative gap.
#[derive(Debug, Default)]
struct ExportRows {
    rows: Vec<usize>,
    exact: Vec<Vec<usize>>,
    kernel: Vec<Vec<usize>>,
    gaps: Vec<f64>,
}

impl ExportRows {
    fn push(&mut self, row: usize, exact: &[usize], kernel: &[usize], gap: f64) {
        self.rows.push(row);
        self.exact.push(exact.to_vec());
        self.kernel.push(kernel.to_vec());
        self.gaps.push(gap);
    }
}

/// Writes layer `layer` of an `Export` under `dir` from the projection rows
/// `qk` [rows, 640], the norm weights, the table, and every row's `engine`
/// queries `q` [rows, 512] and block keys. A layer without rows is refused,
/// as kernel-lab's reader refuses it.
fn write_layer(
    dir: &std::path::Path,
    layer: usize,
    (qk, q_norm, k_norm, table): (&[f64], &[f64], &[f64], EngineTable),
    (q, keys): (&[f64], &[f64]),
    rows: &ExportRows,
) -> Result<()> {
    if rows.rows.is_empty() {
        return Err(format!("layer {layer}: no sparse row to export").into());
    }
    let bits = |values: &[f64]| {
        values
            .iter()
            .flat_map(|&v| f32_to_bf16(v as f32).to_le_bytes())
            .collect::<Vec<_>>()
    };
    let queries = rows
        .rows
        .iter()
        .flat_map(|&row| &q[row * WIDTH..(row + 1) * WIDTH])
        .copied()
        .collect::<Vec<_>>();
    std::fs::create_dir_all(dir)?;
    let write = |name: String, bytes: &[u8]| -> Result<()> {
        use std::io::Write;
        std::fs::File::create_new(dir.join(name))?.write_all(bytes)?;
        Ok(())
    };
    write(format!("layer{layer:02}.keys.bf16"), &bits(keys))?;
    write(format!("layer{layer:02}.q.bf16"), &bits(&queries))?;
    write(format!("layer{layer:02}.qk.bf16"), &bits(qk))?;
    write(
        format!("layer{layer:02}.norm.bf16"),
        &bits(&[q_norm, k_norm].concat()),
    )?;
    let manifest = serde_json::json!({
        "layer": layer,
        "blocks": keys.len() / INDEX_DIM,
        "table": table,
        "rows": rows.rows,
        "exact": rows.exact,
        "kernel": rows.kernel,
        "gap": rows.gaps,
    });
    write(
        format!("layer{layer:02}.json"),
        &serde_json::to_vec(&manifest)?,
    )
}
/// The selections of one full-attention layer of a bundle dump against the
/// policies recomputed from the dump's own BF16 projection rows.
#[derive(Clone, Debug, Default, Serialize)]
pub struct CaptureLayer {
    pub layer: usize,
    /// Sparse rows compared.
    pub rows: usize,
    /// Rows whose four ranks' selections disagree.
    pub rank_disagreements: usize,
    /// Per policy: the device's set against it.
    pub device: Vec<PairStats>,
    /// The emulated kernel against `engine` on the same operands.
    pub kernel_vs_engine: PairStats,
}

fn record(pair: &mut PairStats, row: usize, candidate: &[usize], reference: &[usize], gap: f64) {
    let n = swaps(candidate, reference);
    if n > 0 {
        pair.rows_differ += 1;
        pair.swaps += n;
        pair.max_swaps = pair.max_swaps.max(n);
        pair.gaps.push(gap);
        if pair.examples.len() < EXAMPLES {
            pair.examples.push((row, n, gap));
        }
    }
}

fn finish(pair: &mut PairStats) {
    pair.gaps.sort_by(f64::total_cmp);
    if let (Some(&min), Some(&max)) = (pair.gaps.first(), pair.gaps.last()) {
        pair.gap_min = min;
        pair.gap_max = max;
        pair.gap_median = pair.gaps[pair.gaps.len() / 2];
    }
}

/// The indexer's query and key norm weights of a full-attention layer.
pub fn indexer_norms(checkpoint: &Checkpoint, layer: usize) -> Result<(Vec<f64>, Vec<f64>)> {
    let prefix = format!("model.language_model.layers.{layer}.self_attn.indexer");
    Ok((
        checkpoint.bf16(&format!("{prefix}.q_layernorm.weight"))?,
        checkpoint.bf16(&format!("{prefix}.k_layernorm.weight"))?,
    ))
}

/// Compares the prefill selections (`x.indexer.topk`) of occurrence
/// `occurrence` of `tokens` in a bundle dump with the `engine`, `kernel` and
/// `hf-style` selections of the dump's own projection rows (`x.indexer.qk`)
/// under the engine's cos/sin `table`, on sparse rows r with r % every == 0,
/// in each full-attention layer the dump holds (`norms(layer)` gives the
/// layer's indexer norm weights). The dump must hold every row of the
/// prompt. With `export`, the layers' engine operands and key sets are
/// written for kernel-lab (`Export`).
#[allow(clippy::too_many_arguments)]
pub fn capture_selection(
    norms: &dyn Fn(usize) -> Result<(Vec<f64>, Vec<f64>)>,
    root: &std::path::Path,
    tokens: &[u32],
    occurrence: usize,
    every: usize,
    table: EngineTable,
    export: Option<&Export>,
    mut log: impl FnMut(String),
) -> Result<Vec<CaptureLayer>> {
    use super::drift::{gather, gather_ranks, locate};
    let pieces = locate(root, tokens, occurrence)?;
    let mut layers = Vec::new();
    for layer in (0..LAYERS).filter(|&layer| model_qwen38_contract::contract::is_full(layer as u32))
    {
        let name = |tap: &str| format!("L{layer:02}.{tap}");
        let qk = gather(root, &pieces, &name("x.indexer.qk"), 0)?;
        let selected = gather_ranks(root, &pieces, &name("x.indexer.topk"))?;
        let ((qk, first), (selected, selected_first)) = match (qk, selected) {
            (Some(qk), Some(selected)) => (qk, selected),
            (None, None) => continue,
            _ => {
                return Err(format!(
                    "{}: layer {layer} holds only one of x.indexer.qk and x.indexer.topk",
                    root.display()
                )
                .into());
            }
        };
        if first != 0
            || selected_first != 0
            || qk.shape != [tokens.len(), WIDTH + INDEX_DIM]
            || selected[0].shape != [tokens.len(), TOP_BLOCKS]
        {
            return Err(format!("layer {layer}: the dump does not hold every prompt row").into());
        }
        let (q_norm, k_norm) = norms(layer)?;
        let norms = (&q_norm[..], &k_norm[..], table);
        let report = compare_capture(layer, &qk.data, &selected, norms, every, export)?;
        log(format!(
            "layer {layer}: {} rows; device differs from {}",
            report.rows,
            report
                .device
                .iter()
                .map(|p| format!("{} in {}", p.reference.name(), p.rows_differ))
                .collect::<Vec<_>>()
                .join(", ")
        ));
        layers.push(report);
    }
    if layers.is_empty() {
        return Err(format!(
            "{}: no full-attention layer holds x.indexer.qk and x.indexer.topk",
            root.display()
        )
        .into());
    }
    Ok(layers)
}

/// One layer of `capture_selection`: the device's sets `selected` (per rank,
/// [rows, 512]) against the `engine`, `kernel` and `hf-style` policies on
/// projection rows `qk` [rows, 640], with the engine's table, and the
/// layer's export.
fn compare_capture(
    layer: usize,
    qk: &[f64],
    selected: &[Tensor],
    (q_norm, k_norm, table): (&[f64], &[f64], EngineTable),
    every: usize,
    export: Option<&Export>,
) -> Result<CaptureLayer> {
    let policies = [Policy::Engine, Policy::Kernel, Policy::HfStyle];
    let rounded = qk.iter().map(|&v| bf16_round(v)).collect::<Vec<_>>();
    let engine = engine_operands(&rounded, q_norm, k_norm, table);
    let hf = hf_operands(&rounded, q_norm, k_norm);
    let all = [&engine, &engine, &hf];
    let rows = (0..qk.len() / (WIDTH + INDEX_DIM))
        .filter(|&r| (r + 1) / BLOCK > TOP_BLOCKS && r % every.max(1) == 0)
        .collect::<Vec<_>>();
    let per_row = super::linalg::par_map(rows.len(), |i| {
        let row = rows[i];
        let complete = (row + 1) / BLOCK;
        policies
            .iter()
            .zip(all)
            .map(|(&p, (q, keys))| -> Result<(Vec<usize>, f64)> {
                let s = scores(
                    p,
                    &q[row * WIDTH..(row + 1) * WIDTH],
                    &keys[..complete * INDEX_DIM],
                );
                let (kept, boundary, _) = top_blocks(&s)?;
                Ok((kept, relative_gap(boundary)))
            })
            .collect::<Result<Vec<_>>>()
    })
    .into_iter()
    .collect::<Result<Vec<_>>>()?;
    let mut report = CaptureLayer {
        layer,
        rows: rows.len(),
        device: policies
            .iter()
            .map(|&reference| PairStats {
                reference,
                ..PairStats::default()
            })
            .collect(),
        kernel_vs_engine: PairStats {
            candidate: Some(Policy::Kernel),
            reference: Policy::Engine,
            ..PairStats::default()
        },
        ..CaptureLayer::default()
    };
    let mut exported = ExportRows::default();
    for (&row, sets) in rows.iter().zip(&per_row) {
        let device = |rank: usize| {
            let mut set = selected[rank].data[row * TOP_BLOCKS..(row + 1) * TOP_BLOCKS]
                .iter()
                .map(|&b| b as usize)
                .collect::<Vec<_>>();
            set.sort_unstable();
            set
        };
        let ours = device(0);
        if (1..selected.len()).any(|rank| device(rank) != ours) {
            report.rank_disagreements += 1;
        }
        for (pair, (set, gap)) in report.device.iter_mut().zip(sets) {
            record(pair, row, &ours, set, *gap);
        }
        let (engine, kernel) = (&sets[0], &sets[1]);
        record(
            &mut report.kernel_vs_engine,
            row,
            &kernel.0,
            &engine.0,
            engine.1,
        );
        if let Some(export) = export
            && (engine.0 != kernel.0
                || ours != engine.0
                || ours != kernel.0
                || engine.1 < NEAR_TIE
                || row % export.every.max(1) == 0)
        {
            exported.push(row, &engine.0, &kernel.0, engine.1);
        }
    }
    report.device.iter_mut().for_each(finish);
    finish(&mut report.kernel_vs_engine);
    if let Some(export) = export {
        let operands = (qk, q_norm, k_norm, table);
        write_layer(
            &export.dir,
            layer,
            operands,
            (&engine.0, &engine.1),
            &exported,
        )?;
    }
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use kernel_core::contract::SparseAttentionQuery;
    use kernel_core::sparse_attention_oracle::{block_scores, block_topk};

    /// Deterministic values in [-scale, scale).
    fn synthetic(count: usize, seed: u64, scale: f64) -> Vec<f64> {
        let mut state = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
        (0..count)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                ((state >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0) * scale
            })
            .collect()
    }

    /// Each sparse row's `engine` set as ascending blocks [rows, 512] (zeros
    /// on dense rows), and the last row's first dropped block.
    fn engine_sets(q: &[f64], keys: &[f64], rows: usize) -> (Vec<f64>, usize) {
        let mut sets = vec![0.0; rows * TOP_BLOCKS];
        let mut dropped = 0;
        for row in (0..rows).filter(|&r| (r + 1) / BLOCK > TOP_BLOCKS) {
            let complete = (row + 1) / BLOCK;
            let scores = exact_scores(
                &q[row * WIDTH..(row + 1) * WIDTH],
                &keys[..complete * INDEX_DIM],
            );
            let (kept, _, [_, first_dropped]) = top_blocks(&scores).unwrap();
            sets[row * TOP_BLOCKS..(row + 1) * TOP_BLOCKS]
                .iter_mut()
                .zip(kept)
                .for_each(|(d, b)| *d = b as f64);
            dropped = first_dropped;
        }
        (sets, dropped)
    }

    #[test]
    fn exact_scores_and_top_blocks_match_kernel_core() {
        let blocks = 1500;
        let q = synthetic(WIDTH, 51, 1.0);
        let mut keys = synthetic(blocks * INDEX_DIM, 52, 1.0);
        // Repeated keys give exact score ties across the boundary.
        let (head, rest) = keys.split_at_mut(INDEX_DIM);
        for key in rest.chunks_exact_mut(INDEX_DIM).step_by(7) {
            key.copy_from_slice(head);
        }
        let ours = exact_scores(&q, &keys);
        let reference = block_scores(
            SparseAttentionQuery::BlockScores {
                heads: INDEX_HEADS as u32,
                head_dim: INDEX_DIM as u32,
                max_blocks: blocks as u32,
                capacity: 1,
            },
            &q,
            &keys,
        )
        .unwrap()
        .values;
        assert_eq!(ours, reference);
        for scores in [
            ours,
            vec![0.0; blocks],
            (0..blocks).map(|b| (b % 5) as f64).collect(),
        ] {
            let visible = blocks * BLOCK + 2;
            let expected = block_topk(
                SparseAttentionQuery::BlockTopK {
                    block: BLOCK as u32,
                    top_blocks: TOP_BLOCKS as u32,
                    max_blocks: blocks as u32,
                    capacity: 1,
                },
                &scores,
                visible,
            )
            .unwrap();
            let (kept, boundary, [last, first_dropped]) = top_blocks(&scores).unwrap();
            let mut tokens = kept
                .iter()
                .flat_map(|&b| b * BLOCK..(b + 1) * BLOCK)
                .collect::<Vec<_>>();
            tokens.extend(blocks * BLOCK..visible);
            assert_eq!(tokens, expected);
            assert!(kept.contains(&last) && !kept.contains(&first_dropped));
            assert_eq!(boundary, [scores[last], scores[first_dropped]]);
        }
    }

    #[test]
    fn top_blocks_refuse_nan_scores_as_kernel_core_does() {
        let mut scores = vec![0.0; TOP_BLOCKS + 1];
        scores[TOP_BLOCKS] = f64::NAN;
        assert!(top_blocks(&scores).is_err());
        let visible = scores.len() * BLOCK;
        let query = SparseAttentionQuery::BlockTopK {
            block: BLOCK as u32,
            top_blocks: TOP_BLOCKS as u32,
            max_blocks: scores.len() as u32,
            capacity: 1,
        };
        assert!(block_topk(query, &scores, visible).is_err());
        scores[TOP_BLOCKS] = f64::INFINITY;
        assert_eq!(
            top_blocks(&scores).unwrap().2,
            [TOP_BLOCKS - 2, TOP_BLOCKS - 1]
        );
    }

    #[test]
    fn kernel_scores_round_like_f32_and_stay_close_to_exact() {
        let blocks = 64;
        let q = synthetic(WIDTH, 53, 1.0)
            .into_iter()
            .map(bf16_round)
            .collect::<Vec<_>>();
        let keys = synthetic(blocks * INDEX_DIM, 54, 1.0)
            .into_iter()
            .map(bf16_round)
            .collect::<Vec<_>>();
        let exact = exact_scores(&q, &keys);
        for scores in [kernel_scores(&q, &keys), fma_scores(&q, &keys)] {
            for (s, e) in scores.iter().zip(&exact) {
                assert_eq!(*s, f32r(*s));
                assert!((s - e).abs() <= 1e-5 * e.abs().max(1.0), "{s} against {e}");
            }
        }
    }

    #[test]
    fn hf_rope_is_neox_with_bf16_steps() {
        let mut x = vec![0.0; INDEX_DIM];
        x[0] = 1.0;
        x[32] = 0.5;
        x[100] = 3.0;
        let table = hf_cos_sin(7);
        hf_rotate(&mut x, &table);
        let (cos, sin) = table[0];
        assert_eq!(x[0], bf16_round(bf16_round(cos) + bf16_round(-0.5 * sin)));
        assert_eq!(x[32], bf16_round(bf16_round(0.5 * cos) + bf16_round(sin)));
        assert_eq!(x[100], 3.0);
        assert_eq!(hf_cos_sin(0)[5], (1.0, 0.0));
    }

    #[test]
    fn engine_norm_rotate_follows_the_f32_order() {
        // Ones with a zero weight: the square sum is 128, the inverse
        // 1 / sqrt(fma(128, 1/128, eps)) in F32, and position 0 rotates by
        // the identity.
        let ones = vec![1.0; INDEX_DIM];
        let out = engine_norm_rotate(&ones, &[0.0; INDEX_DIM], &EngineTable::Hf.cos_sin(0));
        let eps = f32::from_bits(super::super::EPS_BITS);
        let inverse = 1.0_f32 / 128_f32.mul_add(1.0 / 128.0, eps).sqrt();
        assert!(out.iter().all(|&v| v == bf16_round(f64::from(inverse))));
        // The two tables agree at position 0 and differ later.
        assert_eq!(EngineTable::Hf.cos_sin(0), EngineTable::F64Angle.cos_sin(0));
        assert_ne!(
            EngineTable::Hf.cos_sin(40_000),
            EngineTable::F64Angle.cos_sin(40_000)
        );
    }

    #[test]
    fn device_selections_compare_with_the_policies_of_their_own_projection() {
        let rows = TOP_BLOCKS * BLOCK + 3 * BLOCK;
        let qk = synthetic(rows * (WIDTH + INDEX_DIM), 55, 1.0)
            .into_iter()
            .map(bf16_round)
            .collect::<Vec<_>>();
        let (q_norm, k_norm) = (synthetic(INDEX_DIM, 56, 0.1), synthetic(INDEX_DIM, 57, 0.1));
        let (q, keys) = engine_operands(&qk, &q_norm, &k_norm, EngineTable::Hf);
        let (device, dropped) = engine_sets(&q, &keys, rows);
        let last = rows - 1;
        let tensor = |data: Vec<f64>| Tensor {
            shape: vec![rows, TOP_BLOCKS],
            data,
        };
        let mut other = device.clone();
        // Rank 1 drops a kept block of the last row for the first dropped one.
        other[last * TOP_BLOCKS] = dropped as f64;
        let ranks = [tensor(device.clone()), tensor(other)];
        let report =
            compare_capture(3, &qk, &ranks, (&q_norm, &k_norm, EngineTable::Hf), 1, None).unwrap();
        assert_eq!(report.rows, rows - (TOP_BLOCKS + 1) * BLOCK + 1);
        assert_eq!(report.rank_disagreements, 1);
        let stored_pair = &report.device[0];
        assert_eq!(stored_pair.reference, Policy::Engine);
        assert_eq!(stored_pair.rows_differ, 0);
        let ranks = [tensor(ranks[1].data.clone())];
        let report =
            compare_capture(3, &qk, &ranks, (&q_norm, &k_norm, EngineTable::Hf), 1, None).unwrap();
        assert_eq!(report.device[0].rows_differ, 1);
        assert_eq!(report.device[0].examples[0].0, last);
    }

    /// One layer of an export read back under the rules of kernel-lab's
    /// `ExportedLayer::read` (crates/kernel-lab/tests/ops/sparse_attention.rs),
    /// which must stay in step with it: even byte counts, the HF table, at
    /// least one row, shapes that agree, sparse rows only, and every set
    /// `TOP_BLOCKS` ascending distinct blocks of its row's complete blocks.
    struct ReadBack {
        qk: Vec<u16>,
        norms: Vec<u16>,
        keys: Vec<u16>,
        q: Vec<u16>,
        rows: Vec<usize>,
        exact: Vec<Vec<usize>>,
        kernel: Vec<Vec<usize>>,
    }

    fn read_export(dir: &std::path::Path, layer: usize) -> std::result::Result<ReadBack, String> {
        // kernel-lab's WIDTH: the projection row, index heads plus the key.
        let width = WIDTH + INDEX_DIM;
        let bits = |name: &str| -> std::result::Result<Vec<u16>, String> {
            let path = dir.join(format!("layer{layer:02}.{name}"));
            let bytes = std::fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))?;
            if bytes.len() % 2 != 0 {
                return Err(format!("{}: odd byte count", path.display()));
            }
            Ok(bytes
                .chunks_exact(2)
                .map(|b| u16::from_le_bytes([b[0], b[1]]))
                .collect())
        };
        let manifest: serde_json::Value = serde_json::from_slice(
            &std::fs::read(dir.join(format!("layer{layer:02}.json"))).map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?;
        if manifest["table"] != "hf" {
            return Err(format!("layer {layer}: table {}", manifest["table"]));
        }
        let field = |name: &str| manifest[name].clone();
        let read = ReadBack {
            qk: bits("qk.bf16")?,
            norms: bits("norm.bf16")?,
            keys: bits("keys.bf16")?,
            q: bits("q.bf16")?,
            rows: serde_json::from_value(field("rows")).map_err(|e| e.to_string())?,
            exact: serde_json::from_value(field("exact")).map_err(|e| e.to_string())?,
            kernel: serde_json::from_value(field("kernel")).map_err(|e| e.to_string())?,
        };
        let tokens = read.qk.len() / width;
        let n = read.rows.len();
        let sparse = |&row: &usize| row < tokens && (row + 1) / BLOCK > TOP_BLOCKS;
        let canonical = |set: &Vec<usize>, row: usize| {
            set.len() == TOP_BLOCKS
                && set.windows(2).all(|w| w[0] < w[1])
                && set.last().is_some_and(|&b| b < (row + 1) / BLOCK)
        };
        if n == 0
            || read.qk.len() != tokens * width
            || read.norms.len() != 2 * INDEX_DIM
            || read.keys.len() != tokens / BLOCK * INDEX_DIM
            || read.q.len() != n * WIDTH
            || read.exact.len() != n
            || read.kernel.len() != n
            || !read.rows.iter().all(sparse)
        {
            return Err(format!("layer {layer}: no sparse rows or shapes disagree"));
        }
        for (index, &row) in read.rows.iter().enumerate() {
            if !canonical(&read.exact[index], row) || !canonical(&read.kernel[index], row) {
                return Err(format!("layer {layer} row {row}: a set is not canonical"));
            }
        }
        Ok(read)
    }

    /// A one-step TP2 bundle dump of a prompt with layer 3's `x.indexer.qk`
    /// (BF16 values) and `x.indexer.topk` on both ranks.
    fn write_dump(root: &std::path::Path, tokens: &[u32], qk: &[f64], topk: &[u32]) {
        let dir = root.join("step0000");
        std::fs::create_dir_all(&dir).unwrap();
        let rows = tokens.len();
        let items = serde_json::json!({"program": "Prefill", "rows": rows, "items": [
            {"start": 0, "rows": rows, "prefix": 0, "tokens": tokens}]});
        std::fs::write(dir.join("items.json"), items.to_string()).unwrap();
        let capture = serde_json::json!({"schema": "chopinfer.taps.v1", "ranks": 2});
        std::fs::write(dir.join("capture.json"), capture.to_string()).unwrap();
        let qk_bytes = qk
            .iter()
            .flat_map(|&v| f32_to_bf16(v as f32).to_le_bytes())
            .collect::<Vec<_>>();
        let topk_bytes = topk
            .iter()
            .flat_map(|b| b.to_le_bytes())
            .collect::<Vec<_>>();
        let mut index = String::new();
        for rank in 0..2 {
            for (name, cols, dtype, bytes) in [
                ("L03.x.indexer.qk", WIDTH + INDEX_DIM, "bf16", &qk_bytes),
                ("L03.x.indexer.topk", TOP_BLOCKS, "u32", &topk_bytes),
            ] {
                std::fs::write(dir.join(format!("{name}.r{rank}.bin")), bytes).unwrap();
                index += &serde_json::json!({"name": name, "rank": rank, "rows": rows,
                    "cols": cols, "dtype": dtype})
                .to_string();
                index += "\n";
            }
        }
        std::fs::write(dir.join("index.jsonl"), index).unwrap();
    }

    #[test]
    fn capture_exports_the_dump_operands_kernel_lab_reads() {
        let root = std::env::temp_dir().join(format!("qwen38-qsa-export-{}", std::process::id()));
        let rows = TOP_BLOCKS * BLOCK + 3 * BLOCK;
        let tokens = (0..rows as u32).collect::<Vec<_>>();
        let qk = synthetic(rows * (WIDTH + INDEX_DIM), 61, 1.0)
            .into_iter()
            .map(bf16_round)
            .collect::<Vec<_>>();
        let norm = |seed| {
            synthetic(INDEX_DIM, seed, 0.1)
                .into_iter()
                .map(bf16_round)
                .collect::<Vec<_>>()
        };
        let (q_norm, k_norm) = (norm(62), norm(63));
        let (q, keys) = engine_operands(&qk, &q_norm, &k_norm, EngineTable::Hf);
        // The device keeps the engine's sets, except that the last row drops
        // a kept block for the first dropped one.
        let (sets, dropped) = engine_sets(&q, &keys, rows);
        let mut topk = sets.into_iter().map(|b| b as u32).collect::<Vec<_>>();
        let last = rows - 1;
        topk[last * TOP_BLOCKS] = dropped as u32;
        let dump = root.join("dump");
        write_dump(&dump, &tokens, &qk, &topk);
        let norms =
            |_: usize| -> Result<(Vec<f64>, Vec<f64>)> { Ok((q_norm.clone(), k_norm.clone())) };
        let capture = |table, dir: &str, every| {
            let export = Export {
                dir: root.join(dir),
                every,
            };
            capture_selection(&norms, &dump, &tokens, 0, 1, table, Some(&export), |_| {})
        };
        let layers = capture(EngineTable::Hf, "hf", 2).unwrap();
        assert_eq!((layers.len(), layers[0].layer), (1, 3));
        assert_eq!(layers[0].device[0].rows_differ, 1);

        let read = read_export(&root.join("hf"), 3).unwrap();
        let sparse = (0..rows).filter(|&r| (r + 1) / BLOCK > TOP_BLOCKS);
        for row in sparse.filter(|r| r % 2 == 0).chain([last]) {
            assert!(read.rows.contains(&row), "row {row} is exported");
        }
        let bf16_bits = |values: &[f64]| {
            values
                .iter()
                .map(|&v| f32_to_bf16(v as f32))
                .collect::<Vec<_>>()
        };
        assert_eq!(read.qk, bf16_bits(&qk));
        assert_eq!(read.norms, bf16_bits(&[&q_norm[..], &k_norm[..]].concat()));
        assert_eq!(read.keys, bf16_bits(&keys));
        for (index, &row) in read.rows.iter().enumerate() {
            let q_row = &q[row * WIDTH..(row + 1) * WIDTH];
            assert_eq!(
                &read.q[index * WIDTH..(index + 1) * WIDTH],
                bf16_bits(q_row)
            );
            let complete = (row + 1) / BLOCK;
            let scored = |policy| scores(policy, q_row, &keys[..complete * INDEX_DIM]);
            assert_eq!(
                read.exact[index],
                top_blocks(&scored(Policy::Engine)).unwrap().0
            );
            assert_eq!(
                read.kernel[index],
                top_blocks(&scored(Policy::Kernel)).unwrap().0
            );
        }

        // Another table is written but refused by the reader; a prompt
        // without sparse rows exports nothing.
        capture(EngineTable::F64Angle, "f64-angle", 2).unwrap();
        assert!(read_export(&root.join("f64-angle"), 3).is_err());
        let short = root.join("short");
        let dense = TOP_BLOCKS * BLOCK;
        write_dump(
            &short,
            &tokens[..dense],
            &qk[..dense * (WIDTH + INDEX_DIM)],
            &topk[..dense * TOP_BLOCKS],
        );
        let export = Export {
            dir: root.join("none"),
            every: 1,
        };
        let refused = capture_selection(
            &norms,
            &short,
            &tokens[..dense],
            0,
            1,
            EngineTable::Hf,
            Some(&export),
            |_| {},
        );
        assert!(refused.is_err());
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn swaps_count_reference_blocks_the_candidate_drops() {
        assert_eq!(swaps(&[1, 2, 3], &[1, 2, 3]), 0);
        assert_eq!(swaps(&[1, 2, 4], &[1, 2, 3]), 1);
        assert_eq!(swaps(&[5, 6, 7], &[1, 2, 3]), 3);
    }
}
