//! The projection declaration, `contract/projections.json`: every
//! projection, layer stem, side tensor and encoder tensor of the checkpoint,
//! declared once for the architecture with shapes named by geometry keys.
//! Checkpoints bind to it from their safetensors headers
//! (`Declaration::bind`); nothing here names a fixed per-checkpoint layout.
use crate::geometry::Geometry;
use kernel_core::contract::WeightFormat;
use kernel_core::precision::format::FormatId;
use model_compiler_sdk::quantization::import::{ActivationRecipe, EncodingRecipe};
use model_compiler_sdk::quantization::projection::metadata::{CompanionNames, HeaderRole};
use projection_spec::registry::Registry;
use projection_spec::{Declaration, ProjectionKind};
use std::collections::BTreeMap;

pub const DECLARATION: &str = include_str!("../projections.json");

/// The geometry keys the declaration's shapes and stems name.
pub fn geometry(g: &Geometry) -> projection_spec::Geometry {
    let s = &g.shape;
    let a = &s.attention;
    let mut keys = BTreeMap::new();
    let mut set = |key: &str, value: u64| {
        keys.insert(key.to_owned(), value);
    };
    set("hidden", s.hidden.into());
    set("vocab", s.vocab.into());
    set("layers", s.layers.into());
    set("heads", a.heads.into());
    set("head_dim", a.head_dim.into());
    set("q_rank", a.q_rank.into());
    set("o_rank", a.o_rank.into());
    set("attn_q_rows", u64::from(a.heads) * u64::from(a.head_dim));
    set("wo_a_rows", u64::from(a.o_groups) * u64::from(a.o_rank));
    set("o_group_k", g.o_group_width().into());
    set("index_heads", s.indexer.heads.into());
    set("index_dim", s.indexer.dim.into());
    set(
        "index_q_rows",
        u64::from(s.indexer.heads) * u64::from(s.indexer.dim),
    );
    set("experts", s.moe.experts.into());
    set("top_k", s.moe.top_k.into());
    set("moe_inter", s.moe.intermediate.into());
    set("hc_streams", s.hc.streams.into());
    set("hc_mix", g.hc_mix().into());
    set("hc_mix_k", g.hc_width().into());
    for (prefix, ids) in [
        ("kv_source", s.csa.kv_sources.to_vec()),
        (
            "reindex",
            s.csa
                .index_sources
                .iter()
                .copied()
                .filter(|id| !s.csa.kv_sources.contains(id))
                .collect(),
        ),
        ("engram_layer", s.engram.layers.to_vec()),
        ("engram_rows", s.engram.rows.to_vec()),
    ] {
        for (i, id) in ids.into_iter().enumerate() {
            set(&format!("{prefix}_{i}"), id.into());
        }
    }
    let e = &s.engram;
    set("engram_dim", e.head_dim.into());
    set(
        "engram_k",
        u64::from(e.max_ngram - 1) * u64::from(e.heads) * u64::from(e.head_dim),
    );
    set(
        "engram_kv_rows",
        u64::from(s.hc.streams + 1) * u64::from(s.hidden),
    );
    let d = &s.dspark;
    set("dspark_stages", d.stages.into());
    set("dspark_last", (d.stages - 1).into());
    set("dspark_experts", d.experts.into());
    set("dspark_top_k", d.top_k.into());
    set("markov_rank", d.markov_rank.into());
    set(
        "dspark_proj_k",
        d.targets.len() as u64 * u64::from(s.hidden),
    );
    set(
        "confidence_k",
        u64::from(s.hidden) + u64::from(d.markov_rank),
    );
    let v = &s.vision;
    set("vision_layers", v.depth.into());
    set("vision_hidden", v.hidden.into());
    set("vision_inter", v.intermediate.into());
    set("vision_qkv_rows", 3 * u64::from(v.hidden));
    set("vision_up_rows", 2 * u64::from(v.intermediate));
    set(
        "vision_patch_k",
        3 * u64::from(v.patch) * u64::from(v.patch),
    );
    set(
        "aligner_k",
        u64::from(v.downsample) * u64::from(v.downsample) * u64::from(v.hidden),
    );
    keys
}

/// The declaration, parsed and validated at `g`'s geometry. It registers no
/// variants.
pub fn declaration(g: &Geometry) -> Result<Declaration, projection_spec::Error> {
    projection_spec::parse(DECLARATION, &geometry(g), &Registry::default())
}

/// How the checkpoint's export names and encodes each declared projection,
/// read from the declaration itself: every quantized weight keeps its
/// scale in `{stem}.scale`, and a projection whose pinned format is not the
/// export's default (block-32 FP8, or MXFP4 for routed experts) states its
/// recipe, since the export lists no unconverted modules.
pub fn header_roles(declaration: &Declaration) -> BTreeMap<String, HeaderRole> {
    declaration
        .projections
        .iter()
        .filter(|p| p.source.is_some())
        .map(|p| {
            let lookup = matches!(p.kind, ProjectionKind::LookupTable {});
            let declared = match p.format {
                Some(FormatId::Bf16) => Some(EncodingRecipe {
                    weight: WeightFormat::Bf16,
                    activation: (!lookup).then_some(ActivationRecipe::Bf16),
                }),
                Some(FormatId::Mxfp8) => Some(EncodingRecipe {
                    weight: WeightFormat::Mxfp8,
                    activation: None,
                }),
                _ => None,
            };
            let quantized = p.format != Some(FormatId::Bf16);
            let role = HeaderRole {
                declared,
                companions: CompanionNames {
                    local: quantized.then(|| "{stem}.scale".to_owned()),
                    global: None,
                },
                ..Default::default()
            };
            (p.id.clone(), role)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_declaration_validates_at_every_compiled_geometry() {
        for g in crate::geometry::GEOMETRIES {
            let declaration = declaration(g).unwrap_or_else(|e| panic!("{}: {}", g.id, e.0));
            assert_eq!(declaration.namespace, "deepseek41");
        }
    }

    #[test]
    fn the_declaration_digest_is_pinned() {
        let declaration = declaration(&crate::geometry::FLASH).unwrap();
        assert_eq!(
            declaration.digest(),
            "93b04b3f9c6d0456f695636de99740737bc35f1a88e3c56b476a197a294ccf18"
        );
    }
}
