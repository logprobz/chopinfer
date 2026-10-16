//! DFlash drafter reference with the deployed vLLM fork's semantics.
//!
//! The context feature of each committed position concatenates the target
//! residual after layers 0, 11, 23, 35 and 47. `fc`, then `hidden_norm`, then
//! every layer's K/V projection give the context cache: K through `k_norm`
//! and partial RoPE, V scaled by 0.612. A draft block embeds the bonus token
//! and then the mask embedding at the following positions and runs five
//! Qwen3 layers whose queries attend, without a causal mask, to the cached
//! context within 1023 positions of the query and to the whole block, with a
//! per-head sink in the softmax denominator. Reductions run in f64, and
//! values are rounded to BF16 where the fork stores BF16 tensors; `fc`, `o`
//! and `down` add one BF16 partial per rank as the bundle's reductions do,
//! four unless `with_ranks` says otherwise.

use model_mimo26_contract::geometry::V26_FLASH as GEOMETRY;

use kernel_core::contract::RotaryStyle;
use kernel_core::rotary::{
    AngleRule, Axes, ComputedTable, FreqTable, RotaryPositions, RotaryRounding, RotarySlice,
    Rotation,
};

use super::*;
use crate::catalog::DrafterKind;

const HIDDEN: usize = GEOMETRY.dflash().hidden_size as usize;
const DFLASH_NORM: RmsNorm = RmsNorm {
    eps: GEOMETRY.dflash().rms_norm_eps,
    ..HF_NORM
};
const LAYERS: usize = GEOMETRY.dflash().num_hidden_layers as usize;
const HEADS: usize = model_mimo26_contract::tp::DFLASH_Q_HEADS as usize;
const KV_HEADS: usize = model_mimo26_contract::tp::DFLASH_KV_HEADS as usize;
const HEAD_DIM: usize = model_mimo26_contract::tp::DFLASH_HEAD_DIM as usize;
const INTERMEDIATE: usize = model_mimo26_contract::tp::DFLASH_INTERMEDIATE as usize;
/// Keys a query may see behind it: the fork's sliding window of 1024.
pub const WINDOW_LEFT: usize = (GEOMETRY.dflash().sliding_window - 1) as usize;
pub const V_SCALE: f64 = GEOMETRY.dflash().dflash_config.attention_value_scale;

/// Q/K normalization: one BF16 weight vector for every head, F32 arithmetic
/// and one BF16 rounding of the output.
const QK_NORM: RmsNorm = RmsNorm {
    before_weight: None,
    ..DFLASH_NORM
};

/// NeoX RoPE on the first 64 dimensions of each head with HF's BF16 cos and
/// sin, one F32 mate product and one F32 FMA, then BF16 storage.
const QK_ROPE: Rotation = Rotation {
    slice: RotarySlice {
        offset: 0,
        len: GEOMETRY.dflash().rotary_dim as usize,
    },
    pairing: RotaryStyle::Neox,
    inverse: false,
    rounding: RotaryRounding {
        table: Some(DType::Bf16),
        arithmetic: RotaryArithmetic::F32Fma,
        output: Some(DType::Bf16),
    },
};

struct Layer {
    input_norm: Vec<u16>,
    post_norm: Vec<u16>,
    q: Vec<u16>,
    k: Vec<u16>,
    v: Vec<u16>,
    q_norm: Vec<u16>,
    k_norm: Vec<u16>,
    o: Vec<u16>,
    sink: Vec<f64>,
    gate: Vec<u16>,
    up: Vec<u16>,
    down: Vec<u16>,
}

/// One layer's cached context: K after norm and RoPE, and scaled V, both as
/// BF16 values, [position][kv head][128].
#[derive(Clone, Debug, Default)]
pub struct ContextLayer {
    pub keys: Vec<f64>,
    pub values: Vec<f64>,
}

pub struct DflashOracle {
    fc: Vec<u16>,
    hidden_norm: Vec<u16>,
    norm: Vec<u16>,
    mask: Vec<u16>,
    layers: Vec<Layer>,
    /// Ranks whose BF16 partials the row-parallel projections add.
    ranks: usize,
}

fn rows(bits: &[u16], cols: usize) -> Bf16Rows<'_> {
    Bf16Rows { bits, cols }
}

fn bits_of(rows: &[Vec<f64>]) -> Vec<u16> {
    rows.iter()
        .flatten()
        .map(|&v| f32_to_bf16(v as f32))
        .collect()
}

/// Per-head RMSNorm, then NeoX RoPE on the first 64 dimensions at each row's
/// position; each result is rounded to BF16.
fn norm_rope(
    heads: &[f64],
    head_count: usize,
    weight: &[u16],
    positions: &[usize],
) -> MimoResult<Vec<f64>> {
    let bits = heads
        .iter()
        .map(|&v| f32_to_bf16(v as f32))
        .collect::<Vec<_>>();
    let normalized = QK_NORM
        .apply_bf16(&bits, weight, HEAD_DIM)?
        .into_iter()
        .map(|v| f64::from(bf16_to_f32(v)))
        .collect::<Vec<_>>();
    let expanded = positions
        .iter()
        .flat_map(|&position| std::iter::repeat_n(position, head_count))
        .collect::<Vec<_>>();
    Ok(QK_ROPE
        .apply(
            &normalized,
            HEAD_DIM,
            RotaryPositions::Linear(&expanded),
            &FreqTable::Computed(ComputedTable {
                theta: GEOMETRY.dflash().rope_theta,
                rule: AngleRule::HfF32AngleBf16CosSin,
                axes: Axes::Linear,
            }),
        )?
        .into_iter()
        .map(rounded)
        .collect())
}

impl DflashOracle {
    /// `root` is the MiMo checkpoint with its `dflash/` directory.
    pub fn load(root: &Path) -> MimoResult<Self> {
        let catalog = Catalog::load_drafter(root, DrafterKind::Dflash)?;
        let read = |name: &str| read_tensor(root, &catalog, name).map(|bytes| u16s(&bytes));
        let mut archive = Vec::new();
        std::io::Read::read_to_end(
            &mut crate::catalog::open_source(&root.join("dflash/mask_embedding.pt"))?,
            &mut archive,
        )?;
        let mask = u16s(&model_mimo26_contract::weight_plan::mask_embedding(
            &archive,
        )?);
        let layers = (0..LAYERS)
            .map(|layer| {
                let read = |suffix: &str| read(&format!("layers.{layer}.{suffix}"));
                Ok(Layer {
                    input_norm: read("input_layernorm.weight")?,
                    post_norm: read("post_attention_layernorm.weight")?,
                    q: read("self_attn.q_proj.weight")?,
                    k: read("self_attn.k_proj.weight")?,
                    v: read("self_attn.v_proj.weight")?,
                    q_norm: read("self_attn.q_norm.weight")?,
                    k_norm: read("self_attn.k_norm.weight")?,
                    o: read("self_attn.o_proj.weight")?,
                    sink: read("self_attn.attention_sink_bias")?
                        .into_iter()
                        .map(|v| f64::from(bf16_to_f32(v)))
                        .collect(),
                    gate: read("mlp.gate_proj.weight")?,
                    up: read("mlp.up_proj.weight")?,
                    down: read("mlp.down_proj.weight")?,
                })
            })
            .collect::<MimoResult<Vec<_>>>()?;
        Ok(Self {
            fc: read("fc.weight")?,
            hidden_norm: read("hidden_norm.weight")?,
            norm: read("norm.weight")?,
            mask,
            layers,
            ranks: 4,
        })
    }

    /// The oracle of a package at `ranks` ranks: 2 or 4.
    pub fn with_ranks(mut self, ranks: usize) -> Self {
        self.ranks = ranks;
        self
    }

    pub fn mask_embedding(&self) -> &[u16] {
        &self.mask
    }

    /// Projects context features, [row][5 x 4096] BF16 target residuals, at
    /// `positions` into each layer's cache entries.
    pub fn context(&self, features: &[u16], positions: &[usize]) -> MimoResult<Vec<ContextLayer>> {
        let width = GEOMETRY.dflash().dflash_config.target_layer_ids.len()
            * GEOMETRY.dflash().target_hidden_size as usize;
        if features.is_empty() || features.len() != positions.len() * width {
            return Err("DFlash context features and positions disagree".into());
        }
        let inputs = features
            .chunks_exact(width)
            .map(|row| row.iter().map(|&v| f64::from(bf16_to_f32(v))).collect())
            .collect::<Vec<Vec<f64>>>();
        let projected = rounded_rows(row_parallel(
            &rows(&self.fc, width),
            &inputs,
            Some(self.ranks),
        ));
        let hidden =
            bf16_rows(&DFLASH_NORM.apply_bf16(&bits_of(&projected), &self.hidden_norm, HIDDEN)?);
        self.layers
            .iter()
            .map(|layer| {
                let keys = project(&rows(&layer.k, HIDDEN), &hidden, true)
                    .into_iter()
                    .flatten()
                    .map(rounded)
                    .collect::<Vec<_>>();
                let values = project(&rows(&layer.v, HIDDEN), &hidden, true)
                    .into_iter()
                    .flatten()
                    .map(|v| rounded(rounded(v) * V_SCALE))
                    .collect();
                Ok(ContextLayer {
                    keys: norm_rope(&keys, KV_HEADS, &layer.k_norm, positions)?,
                    values,
                })
            })
            .collect()
    }

    /// Runs a draft block of `embeddings` rows (BF16, the bonus token first)
    /// at positions `start..`, attending to `context`, whose entry i sits at
    /// position i. Returns the final-norm rows the vocabulary head reads.
    pub fn block(
        &self,
        embeddings: &[u16],
        start: usize,
        context: &[ContextLayer],
    ) -> MimoResult<Vec<u16>> {
        self.block_emulated(embeddings, start, context, false)
    }

    /// `block`, optionally reading every cached and block K and V through an
    /// FP8 E4M3 cache with scale 1.0, as the deployed vLLM drafter's
    /// `kv_cache_dtype` of `fp8_e4m3` stores them.
    ///
    /// The FP8 path emulates that deployment for the acceptance receipts
    /// compared with vLLM (`tests/dflash_acceptance.rs`). It is not the
    /// oracle of a MiMo kernel: MiMo's DFlash context cache holds BF16 K and
    /// V, which `block` reproduces, and no MiMo kernel stores them as E4M3.
    pub fn block_emulated(
        &self,
        embeddings: &[u16],
        start: usize,
        context: &[ContextLayer],
        fp8_kv: bool,
    ) -> MimoResult<Vec<u16>> {
        let stored = |v: f64| if fp8_kv { round_e4m3(v) } else { v };
        let count = embeddings.len() / HIDDEN;
        if count == 0
            || embeddings.len() != count * HIDDEN
            || context.len() != LAYERS
            || context
                .iter()
                .any(|layer| layer.keys.len() != start * KV_HEADS * HEAD_DIM)
        {
            return Err("DFlash block rows or context mismatch".into());
        }
        let positions = (start..start + count).collect::<Vec<_>>();
        let mut residual = embeddings.to_vec();
        for (layer, cache) in self.layers.iter().zip(context) {
            let normalized =
                bf16_rows(&DFLASH_NORM.apply_bf16(&residual, &layer.input_norm, HIDDEN)?);
            let q = norm_rope(
                &project(&rows(&layer.q, HIDDEN), &normalized, true)
                    .into_iter()
                    .flatten()
                    .map(rounded)
                    .collect::<Vec<_>>(),
                HEADS,
                &layer.q_norm,
                &positions,
            )?;
            let k = norm_rope(
                &project(&rows(&layer.k, HIDDEN), &normalized, true)
                    .into_iter()
                    .flatten()
                    .map(rounded)
                    .collect::<Vec<_>>(),
                KV_HEADS,
                &layer.k_norm,
                &positions,
            )?;
            let v = project(&rows(&layer.v, HIDDEN), &normalized, true)
                .into_iter()
                .flatten()
                .map(|v| rounded(rounded(v) * V_SCALE))
                .collect::<Vec<_>>();
            let attention = (0..count)
                .map(|row| {
                    let first = (start + row).saturating_sub(WINDOW_LEFT);
                    (0..HEADS)
                        .flat_map(|head| {
                            let kv = head / (HEADS / KV_HEADS);
                            let query = &q[(row * HEADS + head) * HEAD_DIM..][..HEAD_DIM];
                            let keys = (first..start)
                                .map(|p| (&cache.keys, &cache.values, p))
                                .chain((0..count).map(|j| (&k, &v, j)))
                                .collect::<Vec<_>>();
                            let scores = keys
                                .iter()
                                .map(|(keys, _, p)| {
                                    let key = keys[(p * KV_HEADS + kv) * HEAD_DIM..][..HEAD_DIM]
                                        .iter()
                                        .map(|&v| stored(v))
                                        .collect::<Vec<_>>();
                                    dot(query, &key) / (HEAD_DIM as f64).sqrt()
                                })
                                .collect::<Vec<_>>();
                            let sink = layer.sink[head];
                            let maximum = scores.iter().copied().fold(sink, f64::max);
                            let weights = scores
                                .iter()
                                .map(|s| (s - maximum).exp())
                                .collect::<Vec<_>>();
                            let total = weights.iter().sum::<f64>() + (sink - maximum).exp();
                            (0..HEAD_DIM)
                                .map(|d| {
                                    let sum = keys
                                        .iter()
                                        .zip(&weights)
                                        .map(|((_, values, p), w)| {
                                            w * stored(values[(p * KV_HEADS + kv) * HEAD_DIM + d])
                                        })
                                        .sum::<f64>();
                                    rounded(sum / total)
                                })
                                .collect::<Vec<_>>()
                        })
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            let output = row_parallel(
                &rows(&layer.o, HEADS * HEAD_DIM),
                &attention,
                Some(self.ranks),
            );
            residual = add_rounded(&residual, &output);
            let post = bf16_rows(&DFLASH_NORM.apply_bf16(&residual, &layer.post_norm, HIDDEN)?);
            let gate = rounded_rows(project(&rows(&layer.gate, HIDDEN), &post, true));
            let up = rounded_rows(project(&rows(&layer.up, HIDDEN), &post, true));
            let activated = gate
                .iter()
                .zip(&up)
                .map(|(gate, up)| {
                    gate.iter()
                        .zip(up)
                        .map(|(&g, &u)| rounded(g / (1.0 + (-g).exp()) * u))
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            let output = row_parallel(
                &rows(&layer.down, INTERMEDIATE),
                &activated,
                Some(self.ranks),
            );
            residual = add_rounded(&residual, &output);
        }
        Ok(DFLASH_NORM.apply_bf16(&residual, &self.norm, HIDDEN)?)
    }
}

/// The residual plus a projection's rows, rounded once to BF16.
fn add_rounded(residual: &[u16], output: &[Vec<f64>]) -> Vec<u16> {
    residual
        .chunks_exact(HIDDEN)
        .zip(output)
        .flat_map(|(residual, output)| {
            residual
                .iter()
                .zip(output)
                .map(|(&r, o)| f32_to_bf16((f64::from(bf16_to_f32(r)) + o) as f32))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rope_follows_the_per_head_norm_and_rounds_each_value() {
        let weight = vec![f32_to_bf16(1.0); HEAD_DIM];
        let heads = (0..2 * HEAD_DIM)
            .map(|i| (i % 7) as f64 - 3.0)
            .collect::<Vec<_>>();
        let at_zero = norm_rope(&heads, 2, &weight, &[0]).unwrap();
        let bits = heads
            .iter()
            .map(|&v| f32_to_bf16(v as f32))
            .collect::<Vec<_>>();
        let normalized = QK_NORM
            .apply_bf16(&bits, &weight, HEAD_DIM)
            .unwrap()
            .into_iter()
            .map(|v| f64::from(bf16_to_f32(v)))
            .collect::<Vec<_>>();
        // Position zero rotates nothing.
        assert_eq!(at_zero, normalized);
        let moved = norm_rope(&heads, 2, &weight, &[5]).unwrap();
        assert_ne!(moved[..64], normalized[..64]);
        // The upper 64 dimensions are not rotated.
        assert_eq!(moved[64..128], normalized[64..128]);
        assert!(moved.iter().all(|&v| v == rounded(v)));
    }

    #[test]
    #[ignore = "requires CHOPINFER_MIMO_CHECKPOINT"]
    fn checkpoint_block_is_finite_and_depends_on_its_context() {
        let root = std::env::var_os("CHOPINFER_MIMO_CHECKPOINT").expect("checkpoint path");
        let oracle = DflashOracle::load(Path::new(&root)).unwrap();
        let features = (0..3 * 5 * HIDDEN)
            .map(|i| f32_to_bf16(((i % 127) as f32 - 63.0) / 64.0))
            .collect::<Vec<_>>();
        let context = oracle.context(&features, &[0, 1, 2]).unwrap();
        let mut block = vec![f32_to_bf16(0.25); HIDDEN];
        for _ in 0..3 {
            block.extend_from_slice(oracle.mask_embedding());
        }
        let first = oracle.block(&block, 3, &context).unwrap();
        assert!(first.iter().all(|&v| bf16_to_f32(v).is_finite()));
        let mut shifted = context.clone();
        shifted[0].values.iter_mut().for_each(|v| *v = -*v);
        assert_ne!(oracle.block(&block, 3, &shifted).unwrap(), first);
    }
}
