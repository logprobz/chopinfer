//! The rank views the text program binds: each checkpoint tensor's share on
//! one rank (`crate::weight_plan`), and what binds it. Projection weights
//! bind through the projection API by projection ID; side tensors bind to
//! the engine that reads them.
//!
//! A projection's scales bind with its weight: block-FP8 `weight_scale_inv`,
//! the NVFP4 `weight_scale` and `weight_scale_2` and the MXFP8
//! `weight_scale`. An NVFP4 export's `input_scale` is held by no rank.
//!
//! The projection IDs are those of `projections.json`; the MTP layer binds
//! its own `mtp.*` declarations. `tests/projections.rs` checks every bound
//! ID against the declaration.
use crate::catalog::{DType, TensorSpec, checkpoint_catalog};
use crate::contract::Shape;
use crate::geometry::{Geometry, LayerKind};
use crate::quantization::{FP8_BLOCK128, Layout};
use crate::weight_plan::rank_tensor;

/// The engine that reads a side tensor.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Engine {
    /// Per-channel-decay delta rule: A_log, dt_bias, the q, k and v
    /// convolutions and the gated output norm.
    Kda,
    /// Latent attention: the latent and query norms and the absorbed up
    /// projection (`kv_b_proj`, W_UK and W_UV).
    Latent,
    /// Pooled indexer: the key LayerNorm and the pooling position embedding.
    Indexer,
    /// Sinkhorn mHC: `fn`, `base` and `scale` per sublayer.
    Sinkhorn,
    /// The router's F32 score correction bias.
    Router,
    /// Row norms: input, post-attention, final and the drafter's norms.
    Norm,
}

/// What binds a tensor.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Binding {
    /// A projection weight (or its block scales) of this projection ID.
    Projection(&'static str),
    Side(Engine),
    /// The vocabulary-split embedding rows.
    Embedding,
}

/// One tensor of one rank's view.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RankView {
    pub name: String,
    pub dtype: DType,
    pub shape: Vec<u64>,
    /// First index of the rank's slice along its split dimension.
    pub start: u64,
    pub binding: Binding,
}

impl RankView {
    pub fn bytes(&self) -> u64 {
        self.shape.iter().product::<u64>() * self.dtype.bytes()
    }
}

const TEXT: &str = "model.language_model.";

/// What binds the text tensor `name` of geometry `g`, or `None` for a
/// tensor the program does not read.
pub fn binding(g: &Geometry, name: &str) -> Option<Binding> {
    use Binding::{Projection, Side};
    if name == "lm_head.weight" {
        return Some(Projection("head"));
    }
    let local = name.strip_prefix(TEXT)?;
    if local == "embed_tokens.weight" {
        return Some(Binding::Embedding);
    }
    if local == "norm.weight" {
        return Some(Side(Engine::Norm));
    }
    // layers.<n>.<rest>; the layers after the main ones are the drafter's.
    let (layer, rest) = local.strip_prefix("layers.")?.split_once('.')?;
    let layer: u32 = layer.parse().ok()?;
    if layer >= g.layers() + g.mtp_layers() {
        return None;
    }
    let mtp = layer >= g.layers();
    let weight = rest
        .strip_suffix(".weight_scale_inv")
        .or_else(|| rest.strip_suffix(".weight_scale_2"))
        .or_else(|| rest.strip_suffix(".weight_scale"))
        .or_else(|| rest.strip_suffix(".weight"))
        .or_else(|| rest.strip_suffix(".bias"))
        .unwrap_or(rest);
    let weight = match weight.strip_prefix("mlp.experts.") {
        Some(expert) => &expert[expert.find('.')? + 1..],
        None => weight,
    };
    let projection = match weight {
        "self_attn.q_proj" => "linear.q",
        "self_attn.k_proj" => "linear.k",
        "self_attn.v_proj" => "linear.v",
        "self_attn.b_proj" => "linear.beta",
        "self_attn.f_a_proj" => "linear.forget-a",
        "self_attn.f_b_proj" => "linear.forget-b",
        "self_attn.g_a_proj" => "linear.gate-a",
        "self_attn.g_b_proj" => "linear.gate-b",
        "self_attn.q_a_proj" => "latent.q-a",
        "self_attn.q_b_proj" => "latent.q-b",
        "self_attn.kv_a_proj_with_mqa" => "latent.kv-a",
        "self_attn.indexer.wq_b" => "latent.indexer.q",
        "self_attn.indexer.wk" => "latent.indexer.k",
        "self_attn.indexer.weights_proj" => "latent.indexer.weights",
        // Declared as a side tensor until the shared importer reads a
        // projection source without a `.weight` suffix.
        "self_attn.indexer.index_kpool_compress_gate" => "latent.indexer.pool-gate",
        // KDA and latent attention both name their output `o_proj`.
        "self_attn.o_proj" if !mtp && g.layer_kind(layer) == LayerKind::Linear => "linear.output",
        "self_attn.o_proj" => "latent.output",
        "mlp.gate_proj" | "mlp.up_proj" => "dense.gate-up",
        "mlp.down_proj" => "dense.down",
        "mlp.gate" => "moe.router",
        "gate_proj" | "up_proj" => "moe.experts.gate-up",
        "down_proj" => "moe.experts.down",
        "mlp.shared_experts.gate_proj" | "mlp.shared_experts.up_proj" => "moe.shared.gate-up",
        "mlp.shared_experts.down_proj" => "moe.shared.down",
        "eh_proj" => "mtp.eh",
        _ => "",
    };
    if !projection.is_empty() {
        return Some(Projection(if mtp {
            drafter(projection)?
        } else {
            projection
        }));
    }
    let engine = match weight {
        "self_attn.A_log" | "self_attn.dt_bias" | "self_attn.q_conv1d" | "self_attn.k_conv1d"
        | "self_attn.v_conv1d" | "self_attn.o_norm" => Engine::Kda,
        "self_attn.q_a_layernorm" | "self_attn.kv_a_layernorm" | "self_attn.kv_b_proj" => {
            Engine::Latent
        }
        "self_attn.indexer.k_norm" | "self_attn.indexer.index_kpool_compress_ape" => {
            Engine::Indexer
        }
        "mlp.gate.e_score_correction_bias" => Engine::Router,
        "input_layernorm" | "post_attention_layernorm" | "enorm" | "hnorm" | "shared_head.norm" => {
            Engine::Norm
        }
        w if w.starts_with("hc_attn_") || w.starts_with("hc_ffn_") => Engine::Sinkhorn,
        _ => return None,
    };
    Some(Side(engine))
}

/// The MTP layer's declaration of a main-layer projection ID.
fn drafter(id: &'static str) -> Option<&'static str> {
    Some(match id {
        "latent.q-a" => "mtp.latent.q-a",
        "latent.q-b" => "mtp.latent.q-b",
        "latent.kv-a" => "mtp.latent.kv-a",
        "latent.output" => "mtp.latent.output",
        "latent.indexer.q" => "mtp.latent.indexer.q",
        "latent.indexer.k" => "mtp.latent.indexer.k",
        "latent.indexer.weights" => "mtp.latent.indexer.weights",
        "latent.indexer.pool-gate" => "mtp.latent.indexer.pool-gate",
        "moe.router" => "mtp.moe.router",
        "moe.experts.gate-up" => "mtp.moe.experts.gate-up",
        "moe.experts.down" => "mtp.moe.experts.down",
        "moe.shared.gate-up" => "mtp.moe.shared.gate-up",
        "moe.shared.down" => "mtp.moe.shared.down",
        "mtp.eh" => "mtp.eh",
        _ => return None,
    })
}

fn view(g: &Geometry, spec: &TensorSpec, shape: Vec<u64>, start: u64) -> Result<RankView, String> {
    let binding = binding(g, &spec.name)
        .ok_or_else(|| format!("{}: no binding in the text program", spec.name))?;
    Ok(RankView {
        name: spec.name.clone(),
        dtype: spec.dtype,
        shape,
        start,
        binding,
    })
}

/// Every text tensor `rank` holds of the block-FP8 checkpoint.
pub fn rank_views(shape: Shape, rank: u32) -> Result<Vec<RankView>, String> {
    checkpoint_rank_views(shape, &FP8_BLOCK128, rank)
}

/// Every text tensor `rank` holds of a checkpoint stored in `layout`, in
/// catalog order: the main layers, the MTP layer, the embedding, final norm
/// and head. The vision tower binds through the encoder runtime (step 33).
pub fn checkpoint_rank_views(
    shape: Shape,
    layout: &Layout,
    rank: u32,
) -> Result<Vec<RankView>, String> {
    let g = shape.geometry;
    checkpoint_catalog(g, layout)
        .iter()
        .filter(|spec| spec.name == "lm_head.weight" || spec.name.starts_with(TEXT))
        .filter_map(|spec| rank_tensor(g, spec, shape.ranks(), rank).transpose())
        .map(|part| {
            let part = part?;
            view(g, part.spec, part.shape, part.start)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn find<'a>(views: &'a [RankView], name: &str) -> &'a RankView {
        let full = format!("{TEXT}{name}");
        views
            .iter()
            .find(|view| view.name == full || view.name == name)
            .unwrap_or_else(|| panic!("{name}"))
    }

    #[test]
    fn every_text_tensor_binds() {
        let shape = Shape::TP4;
        for rank in 0..4 {
            let views = rank_views(shape, rank).unwrap();
            assert!(!views.is_empty());
            let experts = views
                .iter()
                .filter(|v| v.name.ends_with(".mlp.experts.0.gate_proj.weight"))
                .count();
            // Every rank holds a slice of expert 0 of each of the 42 main
            // and the MTP MoE layers.
            assert_eq!(experts, 43);
        }
    }

    #[test]
    fn heads_split_and_the_latent_indexer_and_router_replicate() {
        let views = rank_views(Shape::TP4, 1).unwrap();
        let shape = |name| find(&views, name).shape.clone();
        // KDA: 16 heads of 128 per rank.
        assert_eq!(shape("layers.0.self_attn.q_proj.weight"), [2048, 4096]);
        assert_eq!(shape("layers.0.self_attn.A_log"), [16]);
        assert_eq!(shape("layers.0.self_attn.o_proj.weight"), [4096, 2048]);
        assert_eq!(find(&views, "layers.0.self_attn.q_proj.weight").start, 2048);
        // Latent attention: 16 heads of 256 per rank, latent and q_a
        // replicated.
        assert_eq!(shape("layers.3.self_attn.q_b_proj.weight"), [4096, 1536]);
        assert_eq!(shape("layers.3.self_attn.kv_b_proj.weight"), [8192, 512]);
        assert_eq!(shape("layers.3.self_attn.q_a_proj.weight"), [1536, 4096]);
        assert_eq!(
            shape("layers.3.self_attn.kv_a_proj_with_mqa.weight"),
            [512, 4096]
        );
        assert_eq!(shape("layers.3.self_attn.o_proj.weight"), [4096, 4096]);
        assert_eq!(
            shape("layers.3.self_attn.indexer.wq_b.weight"),
            [4096, 1536]
        );
        assert_eq!(shape("layers.3.self_attn.indexer.wk.weight"), [128, 4096]);
        assert_eq!(shape("layers.5.mlp.gate.weight"), [288, 4096]);
        // Dense and shared MLPs split by column; embedding and head by row.
        assert_eq!(shape("layers.0.mlp.gate_proj.weight"), [3072, 4096]);
        assert_eq!(
            shape("layers.5.mlp.shared_experts.gate_proj.weight"),
            [512, 4096]
        );
        assert_eq!(
            shape("layers.5.mlp.shared_experts.down_proj.weight"),
            [4096, 512]
        );
        assert_eq!(shape("embed_tokens.weight"), [38720, 4096]);
        assert_eq!(shape("lm_head.weight"), [38720, 4096]);
        // A quarter of every expert per rank: rank 1 holds rows 512..1024 of
        // each gate and up projection and those columns of each down.
        let expert = find(&views, "layers.5.mlp.experts.287.gate_proj.weight");
        assert_eq!((expert.shape.clone(), expert.start), (vec![512, 4096], 512));
        let expert = find(&views, "layers.5.mlp.experts.0.down_proj.weight");
        assert_eq!((expert.shape.clone(), expert.start), (vec![4096, 512], 512));
        let bindings = |name| find(&views, name).binding;
        assert_eq!(
            bindings("layers.3.self_attn.o_proj.weight"),
            Binding::Projection("latent.output")
        );
        assert_eq!(
            bindings("layers.0.self_attn.o_proj.weight"),
            Binding::Projection("linear.output")
        );
        assert_eq!(
            bindings("layers.3.self_attn.kv_b_proj.weight"),
            Binding::Side(Engine::Latent)
        );
        assert_eq!(
            bindings("layers.0.hc_attn_fn"),
            Binding::Side(Engine::Sinkhorn)
        );
    }
}
