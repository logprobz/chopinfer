//! Header-only catalog and lossless placement/repack recipes for 1, 2 or
//! 4 ranks. Serving admission is defined by the model contract.
use crate::{Result, precision_source::SourcePrecision};
use kernel_core::contract::WeightFormat;
use kernel_core::topology::TensorParallel;
use runtime_weights::DType;
use std::collections::BTreeMap;
#[cfg(test)]
use std::collections::BTreeSet;
use std::ops::Range;
use std::path::Path;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Placement {
    Replicated,
    /// The axis holds `parts` equal parts: one per rank, or the KV heads,
    /// which ranks share in contiguous groups when there are more ranks.
    Axis {
        axis: usize,
        parts: usize,
    },
    /// Three independent q, k and v row slices, each split over the ranks.
    LinearQkv,
    Expert {
        rank: usize,
        local_expert: usize,
    },
    /// Global table rows stored in this checkpoint shard.
    Ngram {
        rows: Range<u64>,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ScaleLayout {
    None,
    Block128 {
        scale: String,
    },
    PerTensor {
        scale: String,
    },
    BlockInverseScale,
    TensorScale,
    QuantizedWeight {
        format: WeightFormat,
        local_scale: String,
        global_scale: Option<String>,
    },
    QuantizedBlockScale {
        format: WeightFormat,
    },
    GlobalScale,
    ActivationScale,
}

#[derive(Clone, Debug)]
pub struct TensorRecord {
    pub name: String,
    pub dtype: DType,
    pub shape: Vec<u64>,
    pub file: String,
    pub file_start: u64,
    pub bytes: u64,
    pub placement: Placement,
    pub scale: ScaleLayout,
}

#[derive(Debug)]
pub struct Catalog {
    pub geometry: &'static model_qwen38_contract::geometry::Geometry,
    pub tensors: BTreeMap<String, TensorRecord>,
    /// The rank count the placements are resolved for.
    pub ranks: usize,
    /// Source interpretation remains separate from implementation selection.
    /// Small injected packing fixtures do not need an export document.
    pub precision: Option<SourcePrecision>,
}

/// The TP4 catalog of the served package, the default.
pub const DEFAULT_RANKS: usize = 4;
#[cfg(test)]
use model_qwen38_contract::contract::{
    ATTN_KV_HEADS as CONTRACT_KV_HEADS, NGRAM_HEADS as CONTRACT_NGRAM_HEADS,
};
#[cfg(test)]
const NGRAM_HEADS: usize = CONTRACT_NGRAM_HEADS as usize;
#[cfg(test)]
const KV_HEADS: usize = CONTRACT_KV_HEADS as usize;
/// Shared expert columns per rank are padded to this tile (zero-padded to a multiple of 128).
pub const SHARED_TILE: usize = 128;

/// The rank count as a validated topology: 1, 2 or 4.
pub fn topology(ranks: usize) -> Result<TensorParallel> {
    if ranks > 4 {
        return Err("Qwen3.8 places 1, 2 or 4 ranks".into());
    }
    TensorParallel::new(ranks).map_err(|e| e.0.into())
}

/// The per-rank shape placement shards with: `kernel_core`'s derivation
/// from Qwen3.8's configuration values, at 1, 2 or 4 ranks.
pub fn rank_shape(ranks: usize) -> Result<kernel_core::rank_shape::RankShape> {
    topology(ranks)?;
    Ok(model_qwen38_contract::contract::rank_shape(ranks as u32).map_err(|e| e.0)?)
}

/// Experts each rank owns (expert parallelism).
pub fn local_experts(ranks: usize) -> Result<usize> {
    Ok(rank_shape(ranks)?
        .experts(model_qwen38_contract::contract::MOE)
        .experts as usize)
}

/// The closed tensor set for the served geometry.
#[cfg(test)]
pub fn expected_tensors() -> BTreeMap<String, (String, Vec<u64>)> {
    expected_tensors_for(&model_qwen38_contract::geometry::FLASH)
        .expect("validated served geometry")
}

/// Expand tensor names and shapes from execution geometry.
pub fn expected_tensors_for(
    geometry: &model_qwen38_contract::geometry::Geometry,
) -> Result<BTreeMap<String, (String, Vec<u64>)>> {
    let config = serde_json::from_str(include_str!("../fixtures/config.json"))?;
    expected_tensors_with_config(geometry, &config)
}

/// Expand the catalog with the export's FP8 block geometry.
pub fn expected_tensors_with_config(
    geometry: &model_qwen38_contract::geometry::Geometry,
    config: &serde_json::Value,
) -> Result<BTreeMap<String, (String, Vec<u64>)>> {
    validate_catalog_geometry(geometry)?;
    let g = geometry;
    let linear_v = (g.linear_value_heads() as usize) * (g.linear_value_head() as usize);
    let linear_qkv = 2 * (g.linear_key_heads() as usize) * (g.linear_head() as usize) + linear_v;
    let ngram_heads = ((g.ngram_orders() as usize) - 1) * (g.ngram_heads_per_order() as usize);
    let ngram_total = ngram_table(geometry, 0)?.rows as usize;
    let ngram_rows = ngram_total / (g.ngram_parts() as usize);
    // Mixed exports omit this field. Keep the reference companion geometry;
    // precision admission derives physical companions from each encoding.
    let reference_block = serde_json::json!([128, 128]);
    let block = config["quantization_config"]
        .get("weight_block_size")
        .unwrap_or(&reference_block)
        .as_array()
        .ok_or("missing FP8 weight block size")?;
    let [block_rows, block_columns] = block.as_slice() else {
        return Err("FP8 weight block size needs two dimensions".into());
    };
    let block_rows = block_rows
        .as_u64()
        .filter(|&n| n > 0)
        .ok_or("invalid FP8 row block")? as usize;
    let block_columns = block_columns
        .as_u64()
        .filter(|&n| n > 0)
        .ok_or("invalid FP8 column block")? as usize;
    let families = [
        (
            "model.language_model.hyper_connection_mixer.hc_norm.weight",
            "BF16",
            vec![(g.hidden() as usize) * (g.streams() as usize)],
        ),
        (
            "model.language_model.hyper_connection_mixer.input_mix_weight_down.weight",
            "BF16",
            vec![
                (g.hc_rank() as usize),
                (g.hidden() as usize) * (g.streams() as usize),
            ],
        ),
        (
            "model.language_model.hyper_connection_mixer.input_mix_weight_up.weight",
            "BF16",
            vec![
                (g.hidden() as usize) * (g.streams() as usize),
                (g.hc_rank() as usize),
            ],
        ),
        (
            "model.language_model.layers.{}.attn_hyper_connection.block_inject_weight.weight",
            "BF16",
            vec![
                (g.streams() as usize),
                (g.hidden() as usize) * (g.streams() as usize),
            ],
        ),
        (
            "model.language_model.layers.{}.attn_hyper_connection.hc_norm.weight",
            "BF16",
            vec![(g.hidden() as usize) * (g.streams() as usize)],
        ),
        (
            "model.language_model.layers.{}.attn_hyper_connection.input_mix_weight_down.weight",
            "BF16",
            vec![
                (g.hc_rank() as usize),
                (g.hidden() as usize) * (g.streams() as usize),
            ],
        ),
        (
            "model.language_model.layers.{}.attn_hyper_connection.input_mix_weight_up.weight",
            "BF16",
            vec![
                (g.hidden() as usize) * (g.streams() as usize),
                (g.hc_rank() as usize),
            ],
        ),
        (
            "model.language_model.layers.{}.linear_attn.A_log",
            "BF16",
            vec![(g.linear_value_heads() as usize)],
        ),
        (
            "model.language_model.layers.{}.linear_attn.conv1d.weight",
            "BF16",
            vec![linear_qkv, 1_usize, (g.conv_taps() as usize)],
        ),
        (
            "model.language_model.layers.{}.linear_attn.dt_bias",
            "BF16",
            vec![(g.linear_value_heads() as usize)],
        ),
        (
            "model.language_model.layers.{}.linear_attn.in_proj_a.weight",
            "BF16",
            vec![(g.linear_value_heads() as usize), (g.hidden() as usize)],
        ),
        (
            "model.language_model.layers.{}.linear_attn.in_proj_b.weight",
            "BF16",
            vec![(g.linear_value_heads() as usize), (g.hidden() as usize)],
        ),
        (
            "model.language_model.layers.{}.linear_attn.in_proj_qkv.weight",
            "BF16",
            vec![linear_qkv, (g.hidden() as usize)],
        ),
        (
            "model.language_model.layers.{}.linear_attn.in_proj_z.weight",
            "BF16",
            vec![linear_v, (g.hidden() as usize)],
        ),
        (
            "model.language_model.layers.{}.linear_attn.norm.weight",
            "BF16",
            vec![(g.linear_value_head() as usize)],
        ),
        (
            "model.language_model.layers.{}.linear_attn.out_proj.weight",
            "BF16",
            vec![(g.hidden() as usize), linear_v],
        ),
        (
            "model.visual.blocks.{}.attn.proj.bias",
            "BF16",
            vec![(g.vision_hidden() as usize)],
        ),
        (
            "model.visual.blocks.{}.attn.proj.weight",
            "BF16",
            vec![(g.vision_hidden() as usize), (g.vision_hidden() as usize)],
        ),
        (
            "model.visual.blocks.{}.attn.qkv.bias",
            "BF16",
            vec![3 * (g.vision_hidden() as usize)],
        ),
        (
            "model.visual.blocks.{}.attn.qkv.weight",
            "BF16",
            vec![
                3 * (g.vision_hidden() as usize),
                (g.vision_hidden() as usize),
            ],
        ),
        (
            "model.visual.blocks.{}.mlp.linear_fc1.bias",
            "BF16",
            vec![(g.vision_intermediate() as usize)],
        ),
        (
            "model.visual.blocks.{}.mlp.linear_fc1.weight",
            "BF16",
            vec![
                (g.vision_intermediate() as usize),
                (g.vision_hidden() as usize),
            ],
        ),
        (
            "model.visual.blocks.{}.mlp.linear_fc2.bias",
            "BF16",
            vec![(g.vision_hidden() as usize)],
        ),
        (
            "model.visual.blocks.{}.mlp.linear_fc2.weight",
            "BF16",
            vec![
                (g.vision_hidden() as usize),
                (g.vision_intermediate() as usize),
            ],
        ),
        (
            "model.visual.blocks.{}.norm1.bias",
            "BF16",
            vec![(g.vision_hidden() as usize)],
        ),
        (
            "model.visual.blocks.{}.norm1.weight",
            "BF16",
            vec![(g.vision_hidden() as usize)],
        ),
        (
            "model.visual.blocks.{}.norm2.bias",
            "BF16",
            vec![(g.vision_hidden() as usize)],
        ),
        (
            "model.visual.blocks.{}.norm2.weight",
            "BF16",
            vec![(g.vision_hidden() as usize)],
        ),
        (
            "model.visual.merger.linear_fc1.bias",
            "BF16",
            vec![(g.vision_hidden() as usize) * (g.vision_merge() as usize).pow(2)],
        ),
        (
            "model.visual.merger.linear_fc1.weight",
            "BF16",
            vec![
                ((g.vision_hidden() as usize) * (g.vision_merge() as usize).pow(2)),
                ((g.vision_hidden() as usize) * (g.vision_merge() as usize).pow(2)),
            ],
        ),
        (
            "model.visual.merger.linear_fc2.bias",
            "BF16",
            vec![(g.vision_out() as usize)],
        ),
        (
            "model.visual.merger.linear_fc2.weight",
            "BF16",
            vec![
                (g.vision_out() as usize),
                ((g.vision_hidden() as usize) * (g.vision_merge() as usize).pow(2)),
            ],
        ),
        (
            "model.visual.merger.norm.bias",
            "BF16",
            vec![(g.vision_hidden() as usize)],
        ),
        (
            "model.visual.merger.norm.weight",
            "BF16",
            vec![(g.vision_hidden() as usize)],
        ),
        (
            "model.visual.patch_embed.proj.bias",
            "BF16",
            vec![(g.vision_hidden() as usize)],
        ),
        (
            "model.visual.patch_embed.proj.weight",
            "BF16",
            vec![
                (g.vision_hidden() as usize),
                (g.vision_channels() as usize),
                (g.vision_temporal_patch() as usize),
                (g.vision_patch() as usize),
                (g.vision_patch() as usize),
            ],
        ),
        (
            "model.visual.pos_embed.weight",
            "BF16",
            vec![
                (g.vision_positions() as usize),
                (g.vision_hidden() as usize),
            ],
        ),
        (
            "model.language_model.layers.{}.mlp.experts.{}.gate_proj.weight_scale_inv",
            "BF16",
            vec![
                (g.expert_intermediate() as usize).div_ceil(block_rows),
                (g.hidden() as usize).div_ceil(block_columns),
            ],
        ),
        (
            "model.language_model.layers.{}.mlp.experts.{}.up_proj.weight_scale_inv",
            "BF16",
            vec![
                (g.expert_intermediate() as usize).div_ceil(block_rows),
                (g.hidden() as usize).div_ceil(block_columns),
            ],
        ),
        (
            "model.language_model.layers.{}.mlp.experts.{}.gate_proj.weight",
            "F8_E4M3",
            vec![(g.expert_intermediate() as usize), (g.hidden() as usize)],
        ),
        (
            "model.language_model.layers.{}.mlp.experts.{}.up_proj.weight",
            "F8_E4M3",
            vec![(g.expert_intermediate() as usize), (g.hidden() as usize)],
        ),
        (
            "model.language_model.layers.{}.mlp.experts.{}.down_proj.weight_scale_inv",
            "BF16",
            vec![
                (g.hidden() as usize).div_ceil(block_rows),
                (g.expert_intermediate() as usize).div_ceil(block_columns),
            ],
        ),
        (
            "model.language_model.layers.{}.mlp.gate.weight",
            "BF16",
            vec![(g.experts() as usize), (g.hidden() as usize)],
        ),
        (
            "model.language_model.layers.{}.mlp.shared_expert.down_proj.weight",
            "BF16",
            vec![
                (g.hidden() as usize),
                (g.shared_expert_intermediate() as usize),
            ],
        ),
        (
            "model.language_model.layers.{}.mlp.shared_expert.gate_proj.weight",
            "BF16",
            vec![
                (g.shared_expert_intermediate() as usize),
                (g.hidden() as usize),
            ],
        ),
        (
            "model.language_model.layers.{}.mlp.shared_expert.up_proj.weight",
            "BF16",
            vec![
                (g.shared_expert_intermediate() as usize),
                (g.hidden() as usize),
            ],
        ),
        (
            "model.language_model.layers.{}.mlp.shared_expert_gate.weight",
            "BF16",
            vec![1_usize, (g.hidden() as usize)],
        ),
        (
            "model.language_model.layers.{}.mlp.experts.{}.down_proj.weight",
            "F8_E4M3",
            vec![(g.hidden() as usize), (g.expert_intermediate() as usize)],
        ),
        (
            "model.language_model.layers.{}.ple.ple_embedding.layer_multipliers",
            "I64",
            vec![(g.ngram_orders() as usize)],
        ),
        (
            "model.language_model.layers.{}.ple.key_proj.weight",
            "BF16",
            vec![
                (g.hidden() as usize) * (g.streams() as usize),
                (g.ple_embed_dim() as usize),
            ],
        ),
        (
            "model.language_model.layers.{}.ple.norm_conv.weight",
            "BF16",
            vec![(g.hidden() as usize) * (g.streams() as usize)],
        ),
        (
            "model.language_model.layers.{}.ple.norm_key.weight",
            "BF16",
            vec![(g.hidden() as usize) * (g.streams() as usize)],
        ),
        (
            "model.language_model.layers.{}.ple.norm_query.weight",
            "BF16",
            vec![(g.hidden() as usize) * (g.streams() as usize)],
        ),
        (
            "model.language_model.layers.{}.ple.ple_embedding.ngram_embedding.weight_scale",
            "BF16",
            vec![1_usize],
        ),
        (
            "model.language_model.layers.{}.ple.ple_embedding.ngram_embedding.shard_{}.weight",
            "F8_E4M3",
            vec![ngram_rows, (g.ple_embed_dim() as usize) / ngram_heads],
        ),
        (
            "model.language_model.layers.{}.ple.ple_embedding.ngram_heads_offsets",
            "I64",
            vec![ngram_heads],
        ),
        (
            "model.language_model.layers.{}.ple.ple_embedding.ngram_heads_vocab_sizes",
            "I64",
            vec![ngram_heads],
        ),
        (
            "model.language_model.layers.{}.ple.conv1d.weight",
            "BF16",
            vec![
                (g.hidden() as usize) * (g.streams() as usize),
                1_usize,
                (g.ple_conv_taps() as usize),
            ],
        ),
        (
            "model.language_model.layers.{}.ple.value_proj.weight",
            "BF16",
            vec![(g.hidden() as usize), (g.ple_embed_dim() as usize)],
        ),
        (
            "model.language_model.layers.{}.self_attn.indexer.index_qk_proj.weight",
            "BF16",
            vec![
                ((g.index_heads() as usize) + (g.index_kv_heads() as usize))
                    * (g.index_dim() as usize),
                (g.hidden() as usize),
            ],
        ),
        (
            "model.language_model.layers.{}.self_attn.indexer.k_layernorm.weight",
            "BF16",
            vec![(g.index_dim() as usize)],
        ),
        (
            "model.language_model.layers.{}.self_attn.indexer.q_layernorm.weight",
            "BF16",
            vec![(g.index_dim() as usize)],
        ),
        (
            "model.language_model.layers.{}.self_attn.k_norm.weight",
            "BF16",
            vec![(g.head_dim() as usize)],
        ),
        (
            "model.language_model.layers.{}.self_attn.k_proj.weight",
            "BF16",
            vec![
                (g.attention_kv_heads() as usize) * (g.head_dim() as usize),
                (g.hidden() as usize),
            ],
        ),
        (
            "model.language_model.layers.{}.self_attn.o_proj.weight",
            "BF16",
            vec![
                (g.hidden() as usize),
                (g.attention_q_heads() as usize) * (g.head_dim() as usize),
            ],
        ),
        (
            "model.language_model.layers.{}.self_attn.q_norm.weight",
            "BF16",
            vec![(g.head_dim() as usize)],
        ),
        (
            "model.language_model.layers.{}.self_attn.q_proj.weight",
            "BF16",
            vec![
                2 * (g.attention_q_heads() as usize) * (g.head_dim() as usize),
                (g.hidden() as usize),
            ],
        ),
        (
            "model.language_model.layers.{}.self_attn.v_proj.weight",
            "BF16",
            vec![
                (g.attention_kv_heads() as usize) * (g.head_dim() as usize),
                (g.hidden() as usize),
            ],
        ),
        (
            "mtp.pre_fc_norm_embedding.weight",
            "BF16",
            vec![(g.hidden() as usize)],
        ),
        (
            "mtp.pre_fc_norm_hidden.weight",
            "BF16",
            vec![(g.hidden() as usize) * (g.streams() as usize)],
        ),
        (
            "mtp.fc_embedding.weight",
            "BF16",
            vec![(g.hidden() as usize), (g.hidden() as usize)],
        ),
        (
            "mtp.fc_hidden.weight",
            "BF16",
            vec![(g.hidden() as usize), (g.hidden() as usize)],
        ),
        (
            "model.language_model.embed_tokens.weight",
            "BF16",
            vec![(g.vocab() as usize), (g.hidden() as usize)],
        ),
        (
            "lm_head.weight",
            "BF16",
            vec![(g.vocab() as usize), (g.hidden() as usize)],
        ),
    ];
    let mut templates = Vec::new();
    for (name, dtype, shape) in families {
        let mut names = vec![name.to_owned()];
        if name.contains(".attn_hyper_connection.") {
            names.push(name.replace(".attn_hyper_connection.", ".mlp_hyper_connection."));
        }
        for name in names {
            if (name.starts_with("model.language_model.layers.")
                && !name.contains(".linear_attn.")
                && !name.contains(".ple."))
                || name.starts_with("model.language_model.hyper_connection_mixer.")
            {
                templates.push((
                    name.replace("model.language_model.", "mtp."),
                    dtype,
                    shape.clone(),
                ));
            }
            templates.push((name, dtype, shape.clone()));
        }
    }
    let mut expected = BTreeMap::new();
    for (template, dtype, tensor_shape) in templates {
        let layers: Vec<usize> = if template.starts_with("model.language_model.layers.") {
            if template.contains(".ple.") {
                g.ple_layer_ids()
                    .iter()
                    .map(|layer| (*layer - 1) as usize)
                    .collect()
            } else {
                (0..(g.layers() as usize))
                    .filter(|&layer| layer_family(&template, g.layer_types()[layer]))
                    .collect()
            }
        } else if template.starts_with("mtp.layers.") {
            (0..(g.mtp_layers() as usize))
                .filter(|&layer| layer_family(&template, g.mtp_layer_types()[layer]))
                .collect()
        } else if template.starts_with("model.visual.blocks.") {
            (0..(g.vision_blocks() as usize)).collect()
        } else {
            vec![0]
        };
        if (g.language_model_only() && template.starts_with("model.visual."))
            || ((g.mtp_layers() as usize) == 0 && template.starts_with("mtp."))
        {
            continue;
        }
        for layer in layers {
            let name = template.replacen("{}", &layer.to_string(), 1);
            let count = if name.contains(".experts.") {
                g.experts() as usize
            } else if name.contains(".shard_") {
                g.ngram_parts() as usize
            } else {
                1
            };
            for index in 0..count {
                let name = name.replacen("{}", &index.to_string(), 1);
                assert!(
                    expected
                        .insert(
                            name,
                            (
                                dtype.into(),
                                tensor_shape.iter().map(|&n| n as u64).collect()
                            )
                        )
                        .is_none()
                );
            }
        }
    }
    Ok(expected)
}

fn layer_family(name: &str, kind: model_qwen38_contract::geometry::LayerKind) -> bool {
    if name.contains(".linear_attn.") {
        kind == model_qwen38_contract::geometry::LayerKind::Linear
    } else if name.contains(".self_attn.") {
        kind == model_qwen38_contract::geometry::LayerKind::Full
    } else {
        true
    }
}

fn validate_catalog_geometry(geometry: &model_qwen38_contract::geometry::Geometry) -> Result<()> {
    let g = geometry;
    if g.layers() == 0 || g.mtp_layers() > 1 {
        return Err("catalog requires decoder layers and at most one MTP layer".into());
    }
    if g.mtp_layer_types()
        .iter()
        .any(|&kind| kind != model_qwen38_contract::geometry::LayerKind::Full)
    {
        return Err("unsupported catalog attention layer type".into());
    }
    if g.ple_layer_ids().len() != 1 {
        return Err("catalog supports exactly one PLE layer".into());
    }
    for &layer in g.ple_layer_ids() {
        if layer == 0
            || layer > g.layers()
            || g.layer_types()[(layer - 1) as usize]
                != model_qwen38_contract::geometry::LayerKind::Linear
        {
            return Err("PLE layer must name a linear attention layer in range".into());
        }
    }
    let table = ngram_table(geometry, 0)?;
    if (g.ngram_parts() as usize) == 0
        || !table.rows.is_multiple_of((g.ngram_parts() as usize) as u64)
    {
        return Err("n-gram rows do not divide into checkpoint shards".into());
    }
    if !(g.ple_embed_dim() as usize).is_multiple_of(table.sizes.len()) {
        return Err("PLE embedding width does not divide over n-gram heads".into());
    }
    Ok(())
}

fn ngram_table(
    geometry: &model_qwen38_contract::geometry::Geometry,
    ple_index: usize,
) -> Result<std::sync::Arc<kernel_core::ngram_oracle::NgramTables>> {
    use std::sync::{Arc, Mutex, OnceLock};
    type Tables = BTreeMap<[usize; 7], Arc<kernel_core::ngram_oracle::NgramTables>>;
    static TABLES: OnceLock<Mutex<Tables>> = OnceLock::new();
    let g = geometry;
    let key = [
        (g.vocab() as usize),
        (g.ngram_orders() as usize),
        (g.ngram_heads_per_order() as usize),
        (g.ngram_vocab_base() as usize),
        0,
        ple_index,
        (g.ngram_alignment() as usize),
    ];
    let mut tables = TABLES
        .get_or_init(|| Mutex::new(BTreeMap::new()))
        .lock()
        .map_err(|_| "n-gram table cache poisoned")?;
    if let Some(table) = tables.get(&key) {
        return Ok(Arc::clone(table));
    }
    let table = Arc::new(
        kernel_core::ngram_oracle::ngram_tables(
            key[0] as u64,
            key[1],
            key[2],
            key[3] as u64,
            key[4] as u64,
            key[5] as u64,
            key[6] as u64,
        )
        .map_err(|e| e.0)?,
    );
    tables.insert(key, Arc::clone(&table));
    Ok(table)
}

fn ple_index(geometry: &model_qwen38_contract::geometry::Geometry, name: &str) -> Result<usize> {
    let layer: usize = name
        .strip_prefix("model.language_model.layers.")
        .and_then(|tail| tail.split('.').next())
        .ok_or("missing PLE layer")?
        .parse()?;
    geometry
        .ple_layer_ids()
        .iter()
        .position(|&id| id.checked_sub(1).map(|id| id as usize) == Some(layer))
        .ok_or_else(|| "tensor names an unconfigured PLE layer".into())
}

pub fn linear_qkv_ranges(geometry: &model_qwen38_contract::geometry::Geometry) -> [Range<u64>; 3] {
    let g = geometry;
    let k = ((g.linear_key_heads() as usize) * (g.linear_head() as usize)) as u64;
    let v = ((g.linear_value_heads() as usize) * (g.linear_value_head() as usize)) as u64;
    [0..k, k..2 * k, 2 * k..2 * k + v]
}

pub(crate) fn precision_roles(
    source: &SourcePrecision,
    ranks: usize,
    geometry: &model_qwen38_contract::geometry::Geometry,
) -> Result<BTreeMap<String, (Placement, ScaleLayout)>> {
    use kernel_core::precision::{Calibration, ScaleScope};
    let mut roles = BTreeMap::new();
    let mut insert = |name: &str, placement: Placement, scale: ScaleLayout| -> Result<()> {
        let value = (placement, scale);
        if let Some(previous) = roles.insert(name.into(), value.clone())
            && previous != value
        {
            return Err(format!("tensor {name}: conflicting companion placement contracts").into());
        }
        Ok(())
    };
    for (name, operand) in &source.operands {
        let weight = &operand.weight;
        let placement = classify_for(geometry, name, ranks)?.0;
        let scale = match weight.format {
            WeightFormat::Bf16 => ScaleLayout::None,
            WeightFormat::Fp8Block128x128(_) => ScaleLayout::Block128 {
                scale: weight.local_scale.as_ref().unwrap().tensor.tensor.clone(),
            },
            WeightFormat::Fp8PerTensor => ScaleLayout::PerTensor {
                scale: weight.local_scale.as_ref().unwrap().tensor.tensor.clone(),
            },
            format => ScaleLayout::QuantizedWeight {
                format,
                local_scale: weight.local_scale.as_ref().unwrap().tensor.tensor.clone(),
                global_scale: weight
                    .global_scale
                    .as_ref()
                    .map(|s| s.tensor.tensor.clone()),
            },
        };
        insert(name, placement.clone(), scale)?;
        if let Some(local) = &weight.local_scale {
            let (placement, scale) =
                if matches!(local.scope, ScaleScope::Tensor | ScaleScope::Table) {
                    (Placement::Replicated, ScaleLayout::TensorScale)
                } else {
                    (
                        placement.clone(),
                        if matches!(weight.format, WeightFormat::Fp8Block128x128(_)) {
                            ScaleLayout::BlockInverseScale
                        } else {
                            ScaleLayout::QuantizedBlockScale {
                                format: weight.format,
                            }
                        },
                    )
                };
            insert(&local.tensor.tensor, placement, scale)?;
        }
        let scalar_placement = if matches!(placement, Placement::Expert { .. }) {
            placement
        } else {
            Placement::Replicated
        };
        if let Some(global) = &weight.global_scale {
            insert(
                &global.tensor.tensor,
                scalar_placement.clone(),
                ScaleLayout::GlobalScale,
            )?;
        }
        if let Some(q) = operand
            .activation
            .as_ref()
            .and_then(|a| a.quantizer.as_ref())
            && let Calibration::Static(scale) = &q.calibration
        {
            insert(
                &scale.tensor.tensor,
                scalar_placement,
                ScaleLayout::ActivationScale,
            )?;
        }
    }
    Ok(roles)
}

impl Catalog {
    /// The TP4 catalog.
    pub fn load(root: &Path) -> Result<Self> {
        Self::load_ranks(root, DEFAULT_RANKS)
    }

    /// The catalog with placements for `ranks`.
    pub fn load_ranks(root: &Path, ranks: usize) -> Result<Self> {
        topology(ranks)?;
        Self::from_captured(&capture(root)?, ranks)
    }

    pub fn from_captured(
        files: &model_compiler_sdk::source::SourceFiles,
        ranks: usize,
    ) -> Result<Self> {
        topology(ranks)?;
        Self::from_imported(files, SourcePrecision::from_captured(files)?, ranks)
    }

    /// [`Self::from_captured`] with the precision already imported from
    /// `files`, so one import serves the catalogs of several rank counts.
    pub fn from_imported(
        files: &model_compiler_sdk::source::SourceFiles,
        precision: SourcePrecision,
        ranks: usize,
    ) -> Result<Self> {
        topology(ranks)?;
        let config = crate::config::Qwen38Config::from_bytes(
            files.inputs.get("config.json").ok_or("missing config")?,
        )?;
        let geometry = config.geometry()?.geometry;
        let roles = precision_roles(&precision, ranks, geometry)?;
        let mut tensors = BTreeMap::new();
        for (file_name, header) in &files.headers {
            for (name, info) in header.tensors() {
                let (placement, scale) = match roles.get(name) {
                    Some(role) => role.clone(),
                    None => classify_for(geometry, name, ranks)?,
                };
                tensors.insert(
                    name.clone(),
                    TensorRecord {
                        name: name.clone(),
                        dtype: info.dtype(),
                        shape: if info.shape().is_empty() {
                            vec![1]
                        } else {
                            info.shape().to_vec()
                        },
                        file: file_name.clone(),
                        file_start: header
                            .data_offset()
                            .checked_add(info.data_offsets().0)
                            .ok_or("offset overflow")?,
                        bytes: info.byte_len(),
                        placement,
                        scale,
                    },
                );
            }
        }
        Ok(Self {
            geometry,
            tensors,
            ranks,
            precision: Some(precision),
        })
    }

    pub fn rank_bytes(&self) -> Vec<u64> {
        (0..self.ranks)
            .map(|rank| {
                self.tensors
                    .values()
                    .map(|t| t.rank_bytes_for(self.geometry, rank, self.ranks))
                    .sum()
            })
            .collect()
    }

    /// Each plan concatenates gate then up along N, including scale rows.
    /// Down and its scales retain their checkpoint layout. No requantization.
    pub fn expert_repacks(&self) -> Vec<ExpertRepack> {
        self.tensors
            .keys()
            .filter_map(|name| {
                let prefix = name.strip_suffix(".gate_proj.weight")?;
                if !prefix.contains(".experts.") {
                    return None;
                }
                let Placement::Expert { rank, local_expert } = self.tensors[name].placement else {
                    return None;
                };
                Some(ExpertRepack {
                    rank,
                    local_expert,
                    gate_up: [
                        format!("{prefix}.gate_proj.weight"),
                        format!("{prefix}.up_proj.weight"),
                    ],
                    gate_up_scales: [
                        format!("{prefix}.gate_proj.weight_scale_inv"),
                        format!("{prefix}.up_proj.weight_scale_inv"),
                    ],
                    down: format!("{prefix}.down_proj.weight"),
                    down_scales: format!("{prefix}.down_proj.weight_scale_inv"),
                })
            })
            .collect()
    }
}

#[derive(Debug)]
pub struct ExpertRepack {
    pub rank: usize,
    pub local_expert: usize,
    pub gate_up: [String; 2],
    pub gate_up_scales: [String; 2],
    pub down: String,
    pub down_scales: String,
}

impl TensorRecord {
    #[cfg(test)]
    pub fn rank_bytes(&self, rank: usize, ranks: usize) -> u64 {
        self.rank_bytes_for(&model_qwen38_contract::geometry::FLASH, rank, ranks)
    }

    pub fn rank_bytes_for(
        &self,
        geometry: &model_qwen38_contract::geometry::Geometry,
        rank: usize,
        ranks: usize,
    ) -> u64 {
        match &self.placement {
            Placement::Replicated => self.bytes,
            Placement::Expert { rank: owner, .. } => {
                assert!(rank < ranks);
                if rank == *owner { self.bytes } else { 0 }
            }
            Placement::Ngram { .. } => {
                self.rank_slices_for(geometry, rank, ranks)
                    .iter()
                    .map(|(_, r)| r.end - r.start)
                    .sum::<u64>()
                    * (self.bytes / self.shape[0])
            }
            Placement::Axis { .. } | Placement::LinearQkv => {
                let slices = self.rank_slices_for(geometry, rank, ranks);
                let axis = slices[0].0;
                let rows: u64 = slices.iter().map(|(_, r)| r.end - r.start).sum();
                self.bytes / self.shape[axis] * rows
            }
        }
    }

    /// Slices are (axis, range); QKV returns separate q, k and v channel ranges.
    /// KV heads are shared by contiguous rank groups (ranks 0/1 and 2/3 at
    /// TP4). N-gram ranges are shard-local.
    #[cfg(test)]
    pub fn rank_slices(&self, rank: usize, ranks: usize) -> Vec<(usize, Range<u64>)> {
        self.rank_slices_for(&model_qwen38_contract::geometry::FLASH, rank, ranks)
    }

    pub fn rank_slices_for(
        &self,
        geometry: &model_qwen38_contract::geometry::Geometry,
        rank: usize,
        ranks: usize,
    ) -> Vec<(usize, Range<u64>)> {
        let tp = topology(ranks).expect("valid rank count");
        assert!(rank < ranks);
        let part = |total: u64, parts: usize| {
            if parts == ranks {
                assert!(
                    total.is_multiple_of(ranks as u64),
                    "{} is not divisible",
                    self.name
                );
                let size = total / ranks as u64;
                return rank as u64 * size..(rank as u64 + 1) * size;
            }
            // KV heads: contiguous rank groups share each head, which
            // `RankShape::attention` counts as one copied head per rank.
            let size = total / parts as u64;
            assert_eq!(size * parts as u64, total, "{} is not divisible", self.name);
            let heads = tp
                .kv_heads(parts)
                .and_then(|split| split.rank(rank))
                .expect("valid split")
                .live;
            heads.start as u64 * size..heads.end as u64 * size
        };
        match &self.placement {
            Placement::Axis { axis, parts } => vec![(*axis, part(self.shape[*axis], *parts))],
            Placement::LinearQkv => linear_qkv_ranges(geometry)
                .into_iter()
                .map(|r| {
                    let local = part(r.end - r.start, ranks);
                    (0, r.start + local.start..r.start + local.end)
                })
                .collect(),
            Placement::Ngram { rows } => {
                let r = ngram_layer_rank_rows(
                    geometry,
                    ple_index(geometry, &self.name).expect("configured PLE tensor"),
                    rank,
                    ranks,
                )
                .expect("validated n-gram topology");
                let start = rows.start.max(r.start);
                let end = rows.end.min(r.end);
                if start >= end {
                    vec![]
                } else {
                    vec![(0, start - rows.start..end - rows.start)]
                }
            }
            Placement::Expert { rank: owner, .. } if *owner != rank => vec![],
            _ => vec![(0, 0..self.shape[0])],
        }
    }
}

/// Each rank owns consecutive hash heads. The final rank retains table padding.
pub fn ngram_rank_rows_flash(rank: usize, ranks: usize) -> Range<u64> {
    ngram_rank_rows_for(&model_qwen38_contract::geometry::FLASH, rank, ranks)
}

pub fn ngram_rank_rows_for(
    geometry: &model_qwen38_contract::geometry::Geometry,
    rank: usize,
    ranks: usize,
) -> Range<u64> {
    ngram_layer_rank_rows(geometry, 0, rank, ranks).expect("validated n-gram topology")
}

fn ngram_layer_rank_rows(
    geometry: &model_qwen38_contract::geometry::Geometry,
    ple_index: usize,
    rank: usize,
    ranks: usize,
) -> Result<Range<u64>> {
    let table = ngram_table(geometry, ple_index)?;
    if ranks == 0 || rank >= ranks || !table.sizes.len().is_multiple_of(ranks) {
        return Err("n-gram heads do not divide over the topology".into());
    }
    let local = table.sizes.len() / ranks;
    let start = table.offsets[rank * local];
    let end = if rank + 1 == ranks {
        table.rows
    } else {
        table.offsets[(rank + 1) * local]
    };
    Ok(start..end)
}

/// The MTP drafter's decoder layer `mtp.layers.0` is placed exactly like a
/// main full-attention layer, so it runs on the same kernels and lowering.
/// Its fusion inputs (`mtp.fc_*`, `mtp.pre_fc_norm_*`) and the final
/// hyper-connection mixer match none of the rules below and stay replicated.
#[cfg(test)]
pub(crate) fn classify(name: &str, ranks: usize) -> Result<(Placement, ScaleLayout)> {
    classify_for(&model_qwen38_contract::geometry::FLASH, name, ranks)
}

pub fn classify_for(
    geometry: &model_qwen38_contract::geometry::Geometry,
    name: &str,
    ranks: usize,
) -> Result<(Placement, ScaleLayout)> {
    topology(ranks)?;
    let g = geometry;
    if !(g.experts() as usize).is_multiple_of(ranks) {
        return Err("experts do not divide over the ranks".into());
    }
    let local = (g.experts() as usize) / ranks;
    let scale = if name.ends_with("weight_scale_inv") {
        ScaleLayout::BlockInverseScale
    } else if name.ends_with("weight_scale") {
        ScaleLayout::TensorScale
    } else if name.contains(".experts.") {
        ScaleLayout::Block128 {
            scale: format!("{}_scale_inv", name),
        }
    } else if let Some((prefix, _)) = name.split_once(".shard_") {
        ScaleLayout::PerTensor {
            scale: format!("{prefix}.weight_scale"),
        }
    } else {
        ScaleLayout::None
    };
    let placement = if let Some((_, suffix)) = name.split_once(".experts.") {
        let expert: usize = suffix.split('.').next().ok_or("missing expert")?.parse()?;
        Placement::Expert {
            rank: expert / local,
            local_expert: expert % local,
        }
    } else if let Some((_, suffix)) = name.split_once(".shard_") {
        let shard: u64 = suffix.split('.').next().ok_or("missing shard")?.parse()?;
        Placement::Ngram {
            rows: {
                let table = ngram_table(geometry, ple_index(geometry, name)?)?;
                if (g.ngram_parts() as usize) == 0
                    || !table.rows.is_multiple_of((g.ngram_parts() as usize) as u64)
                    || shard >= (g.ngram_parts() as usize) as u64
                {
                    return Err("invalid n-gram checkpoint shard".into());
                }
                let rows = table.rows / (g.ngram_parts() as usize) as u64;
                shard * rows..(shard + 1) * rows
            },
        }
    } else if name == "lm_head.weight" {
        Placement::Axis {
            axis: 0,
            parts: ranks,
        }
    } else if name.contains(".shared_expert.") {
        Placement::Axis {
            axis: usize::from(name.ends_with("down_proj.weight")),
            parts: ranks,
        }
    } else if name.contains(".linear_attn.") && !name.ends_with("norm.weight") {
        if name.ends_with("in_proj_qkv.weight") || name.ends_with("conv1d.weight") {
            Placement::LinearQkv
        } else {
            Placement::Axis {
                axis: usize::from(name.ends_with("out_proj.weight")),
                parts: ranks,
            }
        }
    } else if name.contains(".self_attn.") && !name.contains(".indexer.") {
        if name.ends_with("q_proj.weight") {
            Placement::Axis {
                axis: 0,
                parts: ranks,
            }
        } else if name.ends_with("o_proj.weight") {
            Placement::Axis {
                axis: 1,
                parts: ranks,
            }
        } else if name.ends_with("k_proj.weight") || name.ends_with("v_proj.weight") {
            Placement::Axis {
                axis: 0,
                parts: (g.attention_kv_heads() as usize),
            }
        } else {
            Placement::Replicated
        }
    } else {
        Placement::Replicated
    };
    Ok((placement, scale))
}

/// The documents and headers a catalog reads: the configuration, the
/// quantization rules and the shard headers the index names. Nothing else in
/// the checkpoint (an exporter's manifest, say) selects an encoding.
pub fn capture(root: &Path) -> Result<model_compiler_sdk::source::SourceFiles> {
    let files = model_compiler_sdk::source::SourceFiles::capture(
        root,
        &[
            ("config.json", 1024 * 1024),
            ("model.safetensors.index.json", 64 * 1024 * 1024),
        ],
        &[("hf_quant_config.json", 16 * 1024 * 1024)],
        "model.safetensors.index.json",
    )?;
    Ok(files)
}

/// The complete checkpoint's catalog from its pinned shapes, placed for
/// `ranks`, without file offsets.
#[cfg(test)]
pub(crate) fn synthetic(ranks: usize) -> Catalog {
    Catalog {
        geometry: &model_qwen38_contract::geometry::FLASH,
        tensors: expected_tensors()
            .into_iter()
            .map(|(name, (dtype, shape))| {
                let (dtype, width) = match dtype.as_str() {
                    "BF16" => (DType::Bf16, 2),
                    "I64" => (DType::I64, 8),
                    "F8_E4M3" => (DType::F8E4M3, 1),
                    _ => panic!("unexpected dtype"),
                };
                let (placement, scale) = classify(&name, ranks).unwrap();
                let record = TensorRecord {
                    name: name.clone(),
                    dtype,
                    bytes: shape.iter().product::<u64>() * width,
                    shape,
                    file: String::new(),
                    file_start: 0,
                    placement,
                    scale,
                };
                (name, record)
            })
            .collect(),
        ranks,
        precision: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[allow(dead_code)]
    mod checkpoint_index {
        include!("checkpoint_index.rs");
    }
    #[derive(serde::Deserialize)]
    struct Family {
        name: String,
        dtype: String,
        shape: Vec<u64>,
    }

    /// The complete closed tensor set, captured from pinned headers and expanded by geometry.
    fn golden_tensors() -> BTreeMap<String, (String, Vec<u64>)> {
        let families: Vec<Family> =
            serde_json::from_str(include_str!("../fixtures/tensor-families.json"))
                .expect("pinned families");
        let mut expected = BTreeMap::new();
        for f in families {
            let layers: Vec<usize> = if f.name.starts_with("model.language_model.layers.") {
                if f.name.contains(".ple.") {
                    vec![1]
                } else {
                    (0..48)
                        .filter(|i| {
                            if f.name.contains(".linear_attn.") {
                                i % 4 != 3
                            } else if f.name.contains(".self_attn.") {
                                i % 4 == 3
                            } else {
                                true
                            }
                        })
                        .collect()
                }
            } else if f.name.starts_with("model.visual.blocks.") {
                (0..27).collect()
            } else {
                vec![0]
            };
            for layer in layers {
                let name = f.name.replacen("{}", &layer.to_string(), 1);
                let count = if name.contains(".experts.") {
                    512
                } else if name.contains(".shard_") {
                    128
                } else {
                    1
                };
                for index in 0..count {
                    let name = name.replacen("{}", &index.to_string(), 1);
                    assert!(
                        expected
                            .insert(name, (f.dtype.clone(), f.shape.clone()))
                            .is_none()
                    );
                }
            }
        }
        expected
    }

    fn golden_classify(name: &str, ranks: usize) -> Result<(Placement, ScaleLayout)> {
        let local = 512 / ranks;
        let scale = if name.ends_with("weight_scale_inv") {
            ScaleLayout::BlockInverseScale
        } else if name.ends_with("weight_scale") {
            ScaleLayout::TensorScale
        } else if name.contains(".experts.") {
            ScaleLayout::Block128 {
                scale: format!("{}_scale_inv", name),
            }
        } else if let Some((prefix, _)) = name.split_once(".shard_") {
            ScaleLayout::PerTensor {
                scale: format!("{prefix}.weight_scale"),
            }
        } else {
            ScaleLayout::None
        };
        let placement = if let Some((_, suffix)) = name.split_once(".experts.") {
            let expert: usize = suffix.split('.').next().ok_or("missing expert")?.parse()?;
            Placement::Expert {
                rank: expert / local,
                local_expert: expert % local,
            }
        } else if let Some((_, suffix)) = name.split_once(".shard_") {
            let shard: u64 = suffix.split('.').next().ok_or("missing shard")?.parse()?;
            Placement::Ngram {
                rows: shard * 2_500_012..(shard + 1) * 2_500_012,
            }
        } else if name == "lm_head.weight" {
            Placement::Axis {
                axis: 0,
                parts: ranks,
            }
        } else if name.contains(".shared_expert.") {
            Placement::Axis {
                axis: usize::from(name.ends_with("down_proj.weight")),
                parts: ranks,
            }
        } else if name.contains(".linear_attn.") && !name.ends_with("norm.weight") {
            if name.ends_with("in_proj_qkv.weight") || name.ends_with("conv1d.weight") {
                Placement::LinearQkv
            } else {
                Placement::Axis {
                    axis: usize::from(name.ends_with("out_proj.weight")),
                    parts: ranks,
                }
            }
        } else if name.contains(".self_attn.") && !name.contains(".indexer.") {
            if name.ends_with("q_proj.weight") {
                Placement::Axis {
                    axis: 0,
                    parts: ranks,
                }
            } else if name.ends_with("o_proj.weight") {
                Placement::Axis {
                    axis: 1,
                    parts: ranks,
                }
            } else if name.ends_with("k_proj.weight") || name.ends_with("v_proj.weight") {
                Placement::Axis { axis: 0, parts: 2 }
            } else {
                Placement::Replicated
            }
        } else {
            Placement::Replicated
        };
        Ok((placement, scale))
    }

    #[test]
    fn flash_catalog_matches_captured_families_and_roles() {
        let generated = expected_tensors();
        let golden = golden_tensors();
        assert_eq!(generated, golden);
        for ranks in [1, 2, 4] {
            for name in golden.keys() {
                assert_eq!(
                    classify(name, ranks).unwrap(),
                    golden_classify(name, ranks).unwrap(),
                    "{name} at TP{ranks}"
                );
            }
        }
    }

    #[test]
    fn absent_fp8_block_size_preserves_reference_companions() {
        let mut config: serde_json::Value =
            serde_json::from_str(include_str!("../fixtures/config.json")).unwrap();
        config["quantization_config"]
            .as_object_mut()
            .unwrap()
            .remove("weight_block_size");
        assert_eq!(
            expected_tensors_with_config(&model_qwen38_contract::geometry::FLASH, &config).unwrap(),
            golden_tensors()
        );
        for invalid in [
            serde_json::Value::Null,
            serde_json::json!("128"),
            serde_json::json!([128]),
            serde_json::json!([0, 128]),
            serde_json::json!([128, 0]),
        ] {
            config["quantization_config"]["weight_block_size"] = invalid;
            assert!(
                expected_tensors_with_config(&model_qwen38_contract::geometry::FLASH, &config)
                    .is_err()
            );
        }
    }

    #[test]
    fn catalog_uses_geometry_layers_dimensions_and_shards() {
        use model_qwen38_contract::geometry::FLASH;
        let mut geometry = FLASH;
        geometry.shape.layers = &[
            model_qwen38_contract::geometry::LayerKind::Full,
            model_qwen38_contract::geometry::LayerKind::Linear,
        ];
        geometry.shape.hidden = 512;
        geometry.shape.moe.experts = 4;
        geometry.shape.vision.depth = 1;
        geometry.shape.ple.layer_ids = &[2];
        geometry.shape.ngram.vocab_base = 10;
        geometry.shape.ngram.heads_per_order = 2;
        geometry.shape.ngram.parts = 2;
        geometry.shape.ngram.alignment = 8;
        geometry.shape.linear.key_heads = 4;
        geometry.shape.linear.key_dim = 2;
        geometry.shape.linear.value_heads = 6;
        geometry.shape.linear.value_dim = 4;
        let tensors = expected_tensors_for(&geometry).unwrap();
        assert_eq!(
            tensors["model.language_model.layers.1.ple.value_proj.weight"].1,
            [512, 2560]
        );
        assert_eq!(linear_qkv_ranges(&geometry), [0..8, 8..16, 16..40]);

        assert_eq!(
            tensors["model.language_model.layers.0.self_attn.k_proj.weight"].1,
            [512, 512]
        );
        assert!(tensors.contains_key("model.language_model.layers.1.linear_attn.A_log"));
        assert!(!tensors.contains_key("model.language_model.layers.1.self_attn.k_proj.weight"));
        assert!(!tensors.contains_key("model.language_model.layers.2.mlp.gate.weight"));
        assert!(!tensors.contains_key("model.visual.blocks.1.attn.proj.weight"));
        assert!(
            !tensors.contains_key("model.language_model.layers.0.mlp.experts.4.down_proj.weight")
        );
        assert_eq!(ngram_table(&geometry, 0).unwrap().sizes, [11, 13, 17, 19]);
        assert_eq!(ngram_rank_rows_for(&geometry, 1, 2), 24..64);
        let shard =
            "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_1.weight";
        assert_eq!(tensors[shard].1, [32, 640]);
        assert_eq!(
            classify_for(
                &geometry,
                "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_1.weight",
                2
            )
            .unwrap()
            .0,
            Placement::Ngram { rows: 32..64 }
        );
    }

    #[test]
    fn catalog_refuses_invalid_geometry() {
        use model_qwen38_contract::geometry::FLASH;
        let invalid = [
            ("layer count", {
                let mut g = FLASH;
                g.shape.layers = &[];
                g
            }),
            ("MTP count", {
                let mut g = FLASH;
                g.shape.mtp.layers = &[model_qwen38_contract::geometry::LayerKind::Full; 2];
                g
            }),
            ("MTP attention", {
                let mut g = FLASH;
                g.shape.mtp.layers = &[model_qwen38_contract::geometry::LayerKind::Linear];
                g
            }),
            ("PLE count", {
                let mut g = FLASH;
                g.shape.ple.layer_ids = &[2, 3];
                g
            }),
            ("PLE zero", {
                let mut g = FLASH;
                g.shape.ple.layer_ids = &[0];
                g
            }),
            ("PLE range", {
                let mut g = FLASH;
                g.shape.ple.layer_ids = &[49];
                g
            }),
            ("PLE attention", {
                let mut g = FLASH;
                g.shape.ple.layer_ids = &[4];
                g
            }),
            ("shard zero", {
                let mut g = FLASH;
                g.shape.ngram.parts = 0;
                g
            }),
            ("shard divisor", {
                let mut g = FLASH;
                g.shape.ngram.parts = 3;
                g
            }),
            ("ngram orders", {
                let mut g = FLASH;
                g.shape.ngram.orders = 1;
                g
            }),
            ("head width", {
                let mut g = FLASH;
                g.shape.ple.embed_dim = 1;
                g
            }),
        ];
        for (case, geometry) in invalid {
            assert!(expected_tensors_for(&geometry).is_err(), "{case}");
        }
        let mut config: serde_json::Value =
            serde_json::from_str(include_str!("../fixtures/config.json")).unwrap();
        config["quantization_config"]["weight_block_size"] = serde_json::json!([0, 128]);
        assert!(expected_tensors_with_config(&FLASH, &config).is_err());
    }

    #[test]
    fn ngram_tables_follow_ple_layer_order_and_reuse_cached_tables() {
        use model_qwen38_contract::geometry::FLASH;
        let first = ngram_table(&FLASH, 0).unwrap();
        let second = ngram_table(&FLASH, 1).unwrap();
        assert_eq!(first.rows, 320_001_536);
        assert_eq!(second.rows, 320_005_120);
        assert_eq!(second.rows / FLASH.shape.ngram.parts as u64, 2_500_040);
        assert!(std::sync::Arc::ptr_eq(
            &first,
            &ngram_table(&FLASH, 0).unwrap()
        ));
        for index in [0, 1] {
            let table = ngram_table(&FLASH, index).unwrap();
            let rows: Vec<_> = (0..4)
                .map(|rank| ngram_layer_rank_rows(&FLASH, index, rank, 4).unwrap())
                .collect();
            assert_eq!(rows[0].start, 0);
            assert_eq!(rows[3].end, table.rows);
            for pair in rows.windows(2) {
                assert_eq!(pair[0].end, pair[1].start);
            }
        }
    }

    #[test]
    fn catalog_distinguishes_every_family_dimension() {
        let mut geometry = model_qwen38_contract::geometry::FLASH;
        let g = &mut geometry.shape;
        g.layers = &[
            model_qwen38_contract::geometry::LayerKind::Linear,
            model_qwen38_contract::geometry::LayerKind::Full,
        ];
        g.ple.layer_ids = &[1];
        g.hidden = 31;
        g.moe.intermediate = 37;
        g.moe.shared_intermediate = 41;
        g.ple.embed_dim = 44;
        g.hc.streams = 3;
        g.hc.rank = 5;
        g.linear.key_heads = 2;
        g.linear.key_dim = 7;
        g.linear.value_heads = 6;
        g.linear.value_dim = 11;
        g.linear.conv_taps = 13;
        g.attention.head_dim = 17;
        g.attention.heads = 19;
        g.attention.kv_heads = 23;
        g.indexer.heads = 29;
        g.indexer.kv_heads = 43;
        g.indexer.dim = 47;
        g.ple.conv_taps = 53;
        g.ngram.orders = 3;
        g.ngram.heads_per_order = 2;
        g.ngram.vocab_base = 10;
        g.ngram.alignment = 8;
        g.ngram.parts = 2;
        g.moe.experts = 4;
        g.vocab = 59;
        g.vision.depth = 1;
        g.vision.hidden = 61;
        g.vision.out = 67;
        g.vision.intermediate = 71;
        g.vision.channels = 73;
        g.vision.patch = 79;
        g.vision.temporal_patch = 83;
        g.vision.merge = 2;
        g.vision.positions = 89;
        let mut config: serde_json::Value =
            serde_json::from_str(include_str!("../fixtures/config.json")).unwrap();
        config["quantization_config"]["weight_block_size"] = serde_json::json!([8, 16]);
        let tensors = expected_tensors_with_config(&geometry, &config).unwrap();
        let check = |name: &str, shape: &[u64]| {
            assert_eq!(
                tensors
                    .get(name)
                    .unwrap_or_else(|| panic!("missing {name}"))
                    .1,
                shape,
                "{name}"
            );
        };
        for prefix in ["model.language_model.layers.0", "mtp.layers.0"] {
            for branch in ["attn", "mlp"] {
                for (suffix, dims) in [
                    ("block_inject_weight.weight", vec![3, 93]),
                    ("hc_norm.weight", vec![93]),
                    ("input_mix_weight_down.weight", vec![5, 93]),
                    ("input_mix_weight_up.weight", vec![93, 5]),
                ] {
                    check(
                        &format!("{prefix}.{branch}_hyper_connection.{suffix}"),
                        &dims,
                    );
                }
            }
            for (suffix, dims) in [
                ("gate.weight", vec![4, 31]),
                ("shared_expert_gate.weight", vec![1, 31]),
                ("shared_expert.gate_proj.weight", vec![41, 31]),
                ("shared_expert.up_proj.weight", vec![41, 31]),
                ("shared_expert.down_proj.weight", vec![31, 41]),
                ("experts.0.gate_proj.weight", vec![37, 31]),
                ("experts.0.up_proj.weight", vec![37, 31]),
                ("experts.0.down_proj.weight", vec![31, 37]),
                ("experts.0.gate_proj.weight_scale_inv", vec![5, 2]),
                ("experts.0.up_proj.weight_scale_inv", vec![5, 2]),
                ("experts.0.down_proj.weight_scale_inv", vec![4, 3]),
            ] {
                check(&format!("{prefix}.mlp.{suffix}"), &dims);
            }
        }
        for prefix in ["model.language_model.layers.1", "mtp.layers.0"] {
            for (suffix, dims) in [
                ("q_proj.weight", vec![646, 31]),
                ("k_proj.weight", vec![391, 31]),
                ("v_proj.weight", vec![391, 31]),
                ("o_proj.weight", vec![31, 323]),
                ("q_norm.weight", vec![17]),
                ("k_norm.weight", vec![17]),
                ("indexer.index_qk_proj.weight", vec![3384, 31]),
                ("indexer.q_layernorm.weight", vec![47]),
                ("indexer.k_layernorm.weight", vec![47]),
            ] {
                check(&format!("{prefix}.self_attn.{suffix}"), &dims);
            }
        }
        for (suffix, dims) in [
            ("A_log", vec![6]),
            ("dt_bias", vec![6]),
            ("conv1d.weight", vec![94, 1, 13]),
            ("in_proj_qkv.weight", vec![94, 31]),
            ("in_proj_z.weight", vec![66, 31]),
            ("in_proj_b.weight", vec![6, 31]),
            ("in_proj_a.weight", vec![6, 31]),
            ("norm.weight", vec![11]),
            ("out_proj.weight", vec![31, 66]),
        ] {
            check(
                &format!("model.language_model.layers.0.linear_attn.{suffix}"),
                &dims,
            );
        }
        for (suffix, dims) in [
            ("key_proj.weight", vec![93, 44]),
            ("value_proj.weight", vec![31, 44]),
            ("norm_key.weight", vec![93]),
            ("norm_query.weight", vec![93]),
            ("norm_conv.weight", vec![93]),
            ("conv1d.weight", vec![93, 1, 53]),
            ("ple_embedding.layer_multipliers", vec![3]),
            ("ple_embedding.ngram_heads_offsets", vec![4]),
            ("ple_embedding.ngram_heads_vocab_sizes", vec![4]),
            ("ple_embedding.ngram_embedding.weight_scale", vec![1]),
            ("ple_embedding.ngram_embedding.shard_0.weight", vec![32, 11]),
        ] {
            check(
                &format!("model.language_model.layers.0.ple.{suffix}"),
                &dims,
            );
        }
        for (suffix, dims) in [
            ("blocks.0.attn.qkv.weight", vec![183, 61]),
            ("blocks.0.attn.qkv.bias", vec![183]),
            ("blocks.0.attn.proj.weight", vec![61, 61]),
            ("blocks.0.attn.proj.bias", vec![61]),
            ("blocks.0.mlp.linear_fc1.weight", vec![71, 61]),
            ("blocks.0.mlp.linear_fc1.bias", vec![71]),
            ("blocks.0.mlp.linear_fc2.weight", vec![61, 71]),
            ("blocks.0.mlp.linear_fc2.bias", vec![61]),
            ("blocks.0.norm1.weight", vec![61]),
            ("blocks.0.norm1.bias", vec![61]),
            ("blocks.0.norm2.weight", vec![61]),
            ("blocks.0.norm2.bias", vec![61]),
            ("patch_embed.proj.weight", vec![61, 73, 83, 79, 79]),
            ("patch_embed.proj.bias", vec![61]),
            ("pos_embed.weight", vec![89, 61]),
            ("merger.norm.weight", vec![61]),
            ("merger.norm.bias", vec![61]),
            ("merger.linear_fc1.weight", vec![244, 244]),
            ("merger.linear_fc1.bias", vec![244]),
            ("merger.linear_fc2.weight", vec![67, 244]),
            ("merger.linear_fc2.bias", vec![67]),
        ] {
            check(&format!("model.visual.{suffix}"), &dims);
        }
        for prefix in ["model.language_model", "mtp"] {
            check(
                &format!("{prefix}.hyper_connection_mixer.hc_norm.weight"),
                &[93],
            );
            check(
                &format!("{prefix}.hyper_connection_mixer.input_mix_weight_down.weight"),
                &[5, 93],
            );
            check(
                &format!("{prefix}.hyper_connection_mixer.input_mix_weight_up.weight"),
                &[93, 5],
            );
        }
        for name in ["mtp.fc_hidden.weight", "mtp.fc_embedding.weight"] {
            check(name, &[31, 31]);
        }
        check("mtp.pre_fc_norm_hidden.weight", &[93]);
        check("mtp.pre_fc_norm_embedding.weight", &[31]);
        for name in ["lm_head.weight", "model.language_model.embed_tokens.weight"] {
            check(name, &[59, 31]);
        }
        assert_eq!(
            crate::placement::ngram_layout(&geometry),
            "fp8_e4m3_ngram_head_rows_11_v1"
        );
    }

    /// Real production shapes, with at most one expert matrix in memory.
    fn contributor_inventory(
        value: &serde_json::Value,
        weight_map: &BTreeMap<String, String>,
    ) -> Result<model_compiler_sdk::quantization::source::TensorInventory> {
        struct Temp(std::path::PathBuf);
        impl Drop for Temp {
            fn drop(&mut self) {
                std::fs::remove_dir_all(&self.0).unwrap();
            }
        }
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let unique = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let base =
            std::env::var_os("CHOPINFER_SCRATCH").map_or_else(std::env::temp_dir, Into::into);
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_nanos();
        let root = base.join(format!(
            "qwen38-contributor-{}-{stamp}-{unique}",
            std::process::id()
        ));
        std::fs::create_dir(&root)?;
        let _temp = Temp(root.clone());
        let mut header = serde_json::Map::new();
        let mut payload = Vec::new();
        for tensor in value.as_array().unwrap() {
            let elements: u64 = tensor["shape"]
                .as_array()
                .unwrap()
                .iter()
                .map(|n| n.as_u64().unwrap())
                .product();
            let width = match tensor["dtype"].as_str().unwrap() {
                "F8_E4M3" => 1,
                "F32" => 4,
                "I64" => 8,
                _ => 2,
            };
            let start = payload.len();
            payload.resize(start + (elements * width) as usize, 0x38);
            header.insert(tensor["name"].as_str().unwrap().into(), serde_json::json!({
                "dtype":tensor["dtype"], "shape":tensor["shape"], "data_offsets":[start,payload.len()]
            }));
        }
        assert!(payload.iter().any(|byte| *byte != 0));
        let header = serde_json::to_vec(&header)?;
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend(header);
        bytes.extend(payload);
        std::fs::write(root.join("tiny.safetensors"), bytes)?;
        const INDEX: &str = "model.safetensors.index.json";
        std::fs::write(
            root.join(INDEX),
            serde_json::to_vec(&serde_json::json!({"weight_map":weight_map}))?,
        )?;
        let files = model_compiler_sdk::source::SourceFiles::capture(
            &root,
            &[(INDEX, 1024 * 1024)],
            &[],
            INDEX,
        )?;
        Ok(
            model_compiler_sdk::quantization::source::TensorInventory::from_captured(
                &files, INDEX,
            )?,
        )
    }

    #[test]
    fn contributor_tensor_names_shapes_scales_and_pairs() {
        let fixture: serde_json::Value =
            serde_json::from_slice(include_bytes!("../fixtures/contributor-tensors.json")).unwrap();
        let reference = expected_tensors();
        let expected: BTreeMap<_, _> = fixture
            .as_array()
            .unwrap()
            .iter()
            .map(|tensor| {
                let name = tensor["name"].as_str().unwrap();
                (name.to_string(), reference[name].clone())
            })
            .collect();
        let map: BTreeMap<_, _> = expected
            .keys()
            .map(|name| (name.clone(), "tiny.safetensors".to_string()))
            .collect();
        let config =
            crate::config::Qwen38Config::from_bytes(include_bytes!("../fixtures/config.json"))
                .unwrap();
        let mut metadata = model_compiler_sdk::quantization::import::ExportMetadata::default();
        metadata
            .add_document(
                "config.json/quantization_config",
                &serde_json::to_vec(&config.quantization_config).unwrap(),
            )
            .unwrap();
        let admit = |value: &serde_json::Value,
                     map: &BTreeMap<String, String>|
         -> Result<SourcePrecision> {
            SourcePrecision::admit(
                &config,
                metadata.clone(),
                contributor_inventory(value, map)?,
                expected.clone(),
            )
        };
        let precision = admit(&fixture, &map).unwrap();
        let roles =
            precision_roles(&precision, 4, &model_qwen38_contract::geometry::FLASH).unwrap();
        let norm = fixture[0]["name"].as_str().unwrap();
        let weight = fixture[1]["name"].as_str().unwrap();
        let scale = fixture[2]["name"].as_str().unwrap();
        assert_eq!(
            roles[weight].0,
            Placement::Expert {
                rank: 2,
                local_expert: 1
            }
        );
        assert_eq!(roles[scale].0, roles[weight].0);
        assert!(matches!(roles[scale].1, ScaleLayout::BlockInverseScale));
        assert_eq!(classify(norm, 4).unwrap().0, Placement::Replicated);
        // Shared importers also admit valid FP32 block scales.
        let mut fp32 = fixture.clone();
        fp32[2]["dtype"] = "F32".into();
        let fp32 = admit(&fp32, &map).unwrap();
        assert_eq!(
            fp32.operands[weight]
                .weight
                .local_scale
                .as_ref()
                .unwrap()
                .tensor
                .dtype,
            kernel_core::contract::DType::F32
        );
        for label in ["name", "shape", "scale", "pair", "index"] {
            let mut bad = fixture.clone();
            let mut map = map.clone();
            match label {
                "name" => bad[0]["name"] = "unknown.weight".into(),
                "shape" => bad[0]["shape"][0] = 127.into(),
                "scale" => bad[2]["dtype"] = "I64".into(),
                "pair" => {
                    bad.as_array_mut().unwrap().remove(2);
                    map.remove(scale);
                }
                "index" => {
                    map.remove(scale);
                }
                _ => unreachable!(),
            }
            let error = admit(&bad, &map).unwrap_err().to_string();
            let tensor = match label {
                "name" => "unknown.weight",
                "shape" => norm,
                "pair" => weight,
                _ => scale,
            };
            assert!(error.contains(tensor), "{label}: {error}");
            let reason = match label {
                "name" | "index" => "mismatched shard assignment",
                "shape" => "Qwen side calculation/state requires BF16",
                "scale" => "scale",
                "pair" => "missing companion tensor",
                _ => unreachable!(),
            };
            assert!(error.contains(reason), "{label}: {error}");
        }
    }

    #[test]
    fn complete_shapes_placements_and_repack_coverage() {
        let catalog = synthetic(4);
        assert_eq!(catalog.tensors.len(), 152_089);
        assert_eq!(
            catalog.tensors.values().map(|t| t.bytes).sum::<u64>(),
            185_502_232_570
        );
        let plans = catalog.expert_repacks();
        assert_eq!(plans.len(), 49 * 512);
        let mut inputs = BTreeSet::new();
        for p in plans {
            for name in p
                .gate_up
                .iter()
                .chain(p.gate_up_scales.iter())
                .chain([&p.down, &p.down_scales])
            {
                assert!(inputs.insert(name.clone()));
                assert!(catalog.tensors.contains_key(name));
            }
        }
        assert_eq!(inputs.len(), 49 * 512 * 6);
        let ranks = catalog.rank_bytes();
        eprintln!("TP4 bytes: {ranks:?}");
        for (rank, bytes) in ranks.iter().enumerate() {
            assert!((49_000_000_000..50_000_000_000).contains(bytes));
            let qkv =
                &catalog.tensors["model.language_model.layers.0.linear_attn.in_proj_qkv.weight"];
            assert_eq!(
                qkv.rank_slices(rank, 4)
                    .iter()
                    .map(|(_, r)| r.end - r.start)
                    .sum::<u64>(),
                2560
            );
        }
        for t in catalog.tensors.values() {
            if matches!(
                t.placement,
                Placement::Ngram { .. } | Placement::Expert { .. }
            ) {
                assert_eq!((0..4).map(|r| t.rank_bytes(r, 4)).sum::<u64>(), t.bytes);
            }
        }
        assert_eq!(ngram_rank_rows_flash(0, 4).start, 0);
        assert_eq!(ngram_rank_rows_flash(3, 4).end, 320_001_536);
        for r in 0..3 {
            assert_eq!(
                ngram_rank_rows_flash(r, 4).end,
                ngram_rank_rows_flash(r + 1, 4).start
            );
        }
    }

    /// The TP4 placement as it was before the rank count became a parameter.
    fn tp4_slices(t: &TensorRecord, rank: usize) -> Vec<(usize, Range<u64>)> {
        match &t.placement {
            Placement::Axis { axis, parts } => {
                let size = t.shape[*axis] / *parts as u64;
                let part = rank * parts / 4;
                vec![(*axis, part as u64 * size..(part as u64 + 1) * size)]
            }
            Placement::LinearQkv => [0..2048, 2048..4096, 4096..10240]
                .into_iter()
                .map(|r: Range<u64>| {
                    let size = (r.end - r.start) / 4;
                    (
                        0,
                        r.start + rank as u64 * size..r.start + (rank as u64 + 1) * size,
                    )
                })
                .collect(),
            _ => t.rank_slices(rank, 4),
        }
    }

    #[test]
    fn tp4_placement_is_unchanged() {
        let catalog = synthetic(4);
        for t in catalog.tensors.values() {
            for rank in 0..4 {
                assert_eq!(t.rank_slices(rank, 4), tp4_slices(t, rank), "{}", t.name);
            }
            if let Some((_, suffix)) = t.name.split_once(".experts.") {
                let expert: usize = suffix.split('.').next().unwrap().parse().unwrap();
                assert_eq!(
                    t.placement,
                    Placement::Expert {
                        rank: expert / 128,
                        local_expert: expert % 128
                    }
                );
            }
        }
        let heads = [0, 4, 8, 12];
        for (rank, head) in heads.into_iter().enumerate() {
            let rows = ngram_rank_rows_flash(rank, 4);
            let table =
                kernel_core::ngram_oracle::ngram_tables(248320, 3, 8, 20_000_000, 0, 0, 128)
                    .unwrap();
            assert_eq!(rows.start, table.offsets[head]);
        }
    }

    /// At 1, 2 and 8 ranks every sharded source row is placed exactly once,
    /// except the KV heads, which each of their rank group holds.
    #[test]
    fn every_rank_count_places_each_source_once() {
        for ranks in [1, 2, 4] {
            let catalog = synthetic(ranks);
            for t in catalog.tensors.values() {
                if matches!(t.placement, Placement::Ngram { .. }) {
                    let bytes = (0..ranks).map(|r| t.rank_bytes(r, ranks)).sum::<u64>();
                    assert_eq!(bytes, t.bytes, "{}", t.name);
                    continue;
                }
                let copies = match &t.placement {
                    Placement::Replicated => ranks,
                    Placement::Axis { parts, .. } if *parts == KV_HEADS => {
                        (ranks / KV_HEADS).max(1)
                    }
                    _ => 1,
                };
                let mut ranges: Vec<_> = (0..ranks).flat_map(|r| t.rank_slices(r, ranks)).collect();
                ranges.sort_by_key(|(_, r)| (r.start, r.end));
                let axis = ranges[0].0;
                let mut next = 0;
                for group in ranges.chunks(copies) {
                    assert!(
                        group.iter().all(|g| *g == group[0]),
                        "{} at {ranks}",
                        t.name
                    );
                    assert_eq!(group[0].1.start, next, "{} at {ranks}", t.name);
                    next = group[0].1.end;
                }
                assert_eq!(next, t.shape[axis], "{} at {ranks}", t.name);
                assert_eq!(
                    (0..ranks).map(|r| t.rank_bytes(r, ranks)).sum::<u64>(),
                    t.bytes * copies as u64,
                    "{}",
                    t.name
                );
            }
            let heads = NGRAM_HEADS / ranks;
            let table =
                kernel_core::ngram_oracle::ngram_tables(248320, 3, 8, 20_000_000, 0, 0, 128)
                    .unwrap();
            for rank in 0..ranks {
                assert_eq!(
                    ngram_rank_rows_flash(rank, ranks).start,
                    table.offsets[rank * heads]
                );
            }
            assert_eq!(ngram_rank_rows_flash(ranks - 1, ranks).end, 320_001_536);
            let k = &catalog.tensors["model.language_model.layers.3.self_attn.k_proj.weight"];
            let heads: Vec<_> = (0..ranks)
                .map(|r| k.rank_slices(r, ranks)[0].1.clone())
                .collect();
            match ranks {
                1 => assert_eq!(heads[0], 0..512),
                2 => assert_eq!(heads, [0..256, 256..512]),
                8 => assert_eq!(heads[3], 0..256),
                _ => {}
            }
        }
        assert!(topology(16).is_err());
        assert!(topology(3).is_err());
    }

    /// Placement's sizes are the rank shape's, including the shared expert
    /// padding `placement::shared_padded` computes.
    #[test]
    fn placement_sizes_are_the_rank_shapes() {
        for ranks in [1, 2, 4] {
            let shape = rank_shape(ranks).unwrap();
            assert_eq!(
                local_experts(ranks).unwrap(),
                shape.experts(model_qwen38_contract::contract::MOE).experts as usize
            );
            assert_eq!(
                crate::placement::shared_padded(640, ranks).unwrap(),
                u64::from(shape.experts(model_qwen38_contract::contract::MOE).shared)
            );
            let catalog = synthetic(ranks);
            let head = &catalog.tensors["lm_head.weight"];
            let rows: u64 =
                head.rank_slices(0, ranks)[0].1.end - head.rank_slices(0, ranks)[0].1.start;
            assert_eq!(rows, u64::from(shape.vocab_shard()));
            let q = &catalog.tensors["model.language_model.layers.3.self_attn.q_proj.weight"];
            let q_rows = q.rank_slices(0, ranks)[0].1.end;
            let attention = shape.attention(kernels_heads());
            assert_eq!(q_rows, u64::from(attention.q_heads * 2 * 256));
        }
        assert!(rank_shape(3).is_err());
    }

    fn kernels_heads() -> kernel_core::rank_shape::AttentionConfig {
        kernel_core::rank_shape::AttentionConfig {
            q_heads: 24,
            kv_heads: 2,
            qk_head: 256,
            v_head: 256,
        }
    }

    #[test]
    fn checkpoint_headers_cover_every_tensor_once() {
        for checkpoint in checkpoint_index::checkpoints() {
            let Some(root) = checkpoint.checkpoint_root() else {
                eprintln!("SKIP: {} checkpoint assets are unset", checkpoint.name);
                continue;
            };
            let geometry = checkpoint.execution_geometry();
            let config: serde_json::Value =
                serde_json::from_slice(&std::fs::read(root.join("config.json")).unwrap()).unwrap();
            let expected = expected_tensors_with_config(geometry, &config).unwrap();
            let contract: serde_json::Value = checkpoint.fixture("catalog-contract.json");
            assert_eq!(
                expected.len() as u64,
                contract["tensor_count"].as_u64().unwrap()
            );
            for ranks in [2, 4] {
                let catalog = Catalog::load_ranks(&root, ranks).unwrap();
                assert_eq!(catalog.tensors.len(), expected.len());
                let mut bytes = vec![0; ranks];
                for (name, (dtype, shape)) in &expected {
                    let actual = &catalog.tensors[name];
                    assert_eq!(&actual.shape, shape, "{name}");
                    let (placement, _) = classify_for(geometry, name, ranks).unwrap();
                    assert_eq!(actual.placement, placement, "{name}");
                    let width = if dtype == "F8_E4M3" {
                        1
                    } else if dtype == "I64" {
                        8
                    } else {
                        2
                    };
                    let mut record = actual.clone();
                    record.bytes = shape.iter().product::<u64>() * width;
                    for (rank, total) in bytes.iter_mut().enumerate() {
                        *total += record.rank_bytes_for(geometry, rank, ranks);
                    }
                }
                assert_eq!(catalog.rank_bytes(), bytes);
            }
        }
    }
}
