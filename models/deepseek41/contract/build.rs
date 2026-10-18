//! Generate the supported geometries from `models/deepseek41/geometries`
//! (and the directories listed in `CHOPINFER_DEEPSEEK41_GEOMETRY_DIRS`)
//! through the shared generator in `model_compiler_sdk::geometry_build`.
use model_compiler_sdk::geometry_build::{
    Catalog, Kind, Schema, merged_aliases, render_aliases, validate_name,
};
#[path = "src/structure.rs"]
mod structure;
use std::{collections::BTreeSet, env, fs, path::PathBuf};

/// (generated slot, config path, kind). The semantic rules of
/// `supported.json` cover the remaining execution fields.
const FIELDS: &[(&str, &str, Kind)] = &[
    ("shape.tokens.bos", "bos_token_id", Kind::U32),
    ("shape.tokens.eos", "eos_token_id", Kind::U32),
    ("shape.tokens.pad", "pad_token_id", Kind::U32),
    ("shape.tokens.image", "image_token_id", Kind::U32),
    ("shape.vocab", "text_config.vocab_size", Kind::U32),
    ("shape.hidden", "text_config.hidden_size", Kind::U32),
    ("shape.layers", "text_config.num_hidden_layers", Kind::U32),
    (
        "shape.context",
        "text_config.max_position_embeddings",
        Kind::U32,
    ),
    ("shape.eps", "text_config.rms_norm_eps", Kind::F64),
    ("shape.swiglu_limit", "text_config.swiglu_limit", Kind::F64),
    (
        "shape.compress_ratios",
        "text_config.compress_ratios",
        Kind::U32List,
    ),
    (
        "shape.attention.heads",
        "text_config.num_attention_heads",
        Kind::U32,
    ),
    (
        "shape.attention.head_dim",
        "text_config.head_dim",
        Kind::U32,
    ),
    (
        "shape.attention.rope_dim",
        "text_config.qk_rope_head_dim",
        Kind::U32,
    ),
    (
        "shape.attention.q_rank",
        "text_config.q_lora_rank",
        Kind::U32,
    ),
    (
        "shape.attention.o_rank",
        "text_config.o_lora_rank",
        Kind::U32,
    ),
    (
        "shape.attention.o_groups",
        "text_config.o_groups",
        Kind::U32,
    ),
    (
        "shape.attention.window",
        "text_config.sliding_window",
        Kind::U32,
    ),
    ("shape.rope.theta", "text_config.rope_theta", Kind::F64),
    (
        "shape.rope.compress_theta",
        "text_config.compress_rope_theta",
        Kind::F64,
    ),
    (
        "shape.rope.factor",
        "text_config.rope_scaling.factor",
        Kind::F64,
    ),
    (
        "shape.rope.beta_fast",
        "text_config.rope_scaling.beta_fast",
        Kind::F64,
    ),
    (
        "shape.rope.beta_slow",
        "text_config.rope_scaling.beta_slow",
        Kind::F64,
    ),
    (
        "shape.rope.original_context",
        "text_config.rope_scaling.original_max_position_embeddings",
        Kind::U32,
    ),
    (
        "shape.csa.kv_sources",
        "text_config.kv_source_layer_ids",
        Kind::U32List,
    ),
    (
        "shape.csa.index_sources",
        "text_config.index_source_layer_ids",
        Kind::U32List,
    ),
    (
        "shape.indexer.heads",
        "text_config.index_n_heads",
        Kind::U32,
    ),
    ("shape.indexer.dim", "text_config.index_head_dim", Kind::U32),
    ("shape.indexer.top_k", "text_config.index_topk", Kind::U32),
    (
        "shape.candidates.source",
        "text_config.candidate_source_layer_id",
        Kind::U32,
    ),
    (
        "shape.candidates.top_blocks",
        "text_config.candidate_topk_blocks",
        Kind::U32,
    ),
    (
        "shape.candidates.block",
        "text_config.candidate_block_size",
        Kind::U32,
    ),
    ("shape.hc.streams", "text_config.hc_mult", Kind::U32),
    (
        "shape.hc.sinkhorn_iterations",
        "text_config.hc_sinkhorn_iters",
        Kind::U32,
    ),
    ("shape.hc.eps", "text_config.hc_eps", Kind::F64),
    (
        "shape.moe.intermediate",
        "text_config.moe_intermediate_size",
        Kind::U32,
    ),
    (
        "shape.moe.experts",
        "text_config.n_routed_experts",
        Kind::U32,
    ),
    (
        "shape.moe.top_k",
        "text_config.num_experts_per_tok",
        Kind::U32,
    ),
    (
        "shape.moe.routed_scale",
        "text_config.routed_scaling_factor",
        Kind::F64,
    ),
    (
        "shape.engram.layers",
        "text_config.engram_layer_ids",
        Kind::U32List,
    ),
    (
        "shape.engram.rows",
        "text_config.engram_num_embeddings",
        Kind::U32List,
    ),
    (
        "shape.engram.max_ngram",
        "text_config.engram_max_ngram_size",
        Kind::U32,
    ),
    (
        "shape.engram.vocab",
        "text_config.engram_vocab_size",
        Kind::U32,
    ),
    (
        "shape.engram.heads",
        "text_config.engram_n_heads",
        Kind::U32,
    ),
    (
        "shape.engram.head_dim",
        "text_config.engram_head_dim",
        Kind::U32,
    ),
    (
        "shape.engram.pad_token",
        "text_config.engram_pad_token_id",
        Kind::U32,
    ),
    (
        "shape.engram.compressed_vocab",
        "text_config.engram_compressed_vocab_size",
        Kind::U32,
    ),
    (
        "shape.dspark.stages",
        "text_config.num_nextn_predict_layers",
        Kind::U32,
    ),
    (
        "shape.dspark.block",
        "text_config.dspark_block_size",
        Kind::U32,
    ),
    (
        "shape.dspark.noise_token",
        "text_config.dspark_noise_token_id",
        Kind::U32,
    ),
    (
        "shape.dspark.targets",
        "text_config.dspark_target_layer_ids",
        Kind::U32List,
    ),
    (
        "shape.dspark.markov_rank",
        "text_config.dspark_markov_rank",
        Kind::U32,
    ),
    (
        "shape.dspark.experts",
        "text_config.dspark_n_routed_experts",
        Kind::U32,
    ),
    (
        "shape.dspark.top_k",
        "text_config.dspark_num_experts_per_tok",
        Kind::U32,
    ),
    (
        "shape.vision.depth",
        "vision_config.num_hidden_layers",
        Kind::U32,
    ),
    (
        "shape.vision.hidden",
        "vision_config.hidden_size",
        Kind::U32,
    ),
    (
        "shape.vision.heads",
        "vision_config.num_attention_heads",
        Kind::U32,
    ),
    (
        "shape.vision.intermediate",
        "vision_config.intermediate_size",
        Kind::U32,
    ),
    ("shape.vision.patch", "vision_config.patch_size", Kind::U32),
    (
        "shape.vision.rope_theta",
        "vision_config.rope_theta",
        Kind::F64,
    ),
    (
        "shape.vision.downsample",
        "vision_config.downsample_ratio",
        Kind::U32,
    ),
    (
        "shape.vision.max_tokens",
        "vision_config.max_image_tokens",
        Kind::U32,
    ),
    (
        "shape.vision.min_pixels",
        "vision_config.min_pixels",
        Kind::U32,
    ),
];

const GROUPS: &[(&str, &str)] = &[
    ("shape", "Shape"),
    ("shape.tokens", "Tokens"),
    ("shape.attention", "Attention"),
    ("shape.rope", "Rope"),
    ("shape.csa", "Csa"),
    ("shape.indexer", "Indexer"),
    ("shape.candidates", "Candidates"),
    ("shape.hc", "Hc"),
    ("shape.moe", "Moe"),
    ("shape.engram", "Engram"),
    ("shape.dspark", "Dspark"),
    ("shape.vision", "Vision"),
];

const SCHEMA: Schema<Kind> = Schema {
    fields: FIELDS,
    groups: GROUPS,
};

fn main() {
    let directories = Catalog {
        variable: "CHOPINFER_DEEPSEEK41_GEOMETRY_DIRS",
        directory_keys: &[
            "aliases",
            "geometries",
            "runtime_fields",
            "semantic_rules",
            "quantization_recipe_classes",
        ],
        geometry_keys: &["id", "file", "kernel_tag"],
    }
    .read();
    let mut output = SCHEMA.declarations();
    // Runtime selection counts do not choose a different compiled geometry.
    let compile_fields = FIELDS
        .iter()
        .copied()
        .filter(|(_, path, _)| *path != "text_config.index_topk")
        .collect::<Vec<_>>();
    output += &Schema {
        fields: &compile_fields,
        groups: GROUPS,
    }
    .compile_fields(&directories[0].supported);
    let mut values = Vec::new();
    let mut tags = BTreeSet::new();
    let mut shapes = BTreeSet::new();
    for (directory_index, directory) in directories.iter().enumerate() {
        for geometry in &directory.geometries {
            let (id, path, config) = (geometry.id.as_str(), &geometry.path, &geometry.config);
            validate_name(id, "geometry id");
            let tag = geometry.declaration["kernel_tag"]
                .as_str()
                .map(str::to_owned)
                .unwrap_or_else(|| format!("deepseek41_{id}"));
            validate_name(&tag, "kernel tag");
            assert!(tags.insert(tag.clone()), "duplicate kernel tag {tag}");
            let fields = SCHEMA.literals(config, path);
            structure::check(config).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
            SCHEMA.check_leaves(config, &geometry.rules, path);
            let fields = fields.into_iter().collect::<Vec<_>>();
            let shape = SCHEMA.grouped_value("shape", &fields);
            assert!(
                shapes.insert(shape.clone()),
                "duplicate geometry shape {id}"
            );
            let base = directory_index == 0;
            let rules = geometry.rules.to_string();
            let document = &geometry.document;
            let geometry = format!(
                "Geometry {{ base: {base}, id: {id:?}, kernel_tag: {tag:?}, shape: {shape}, rules: {rules:?}, document: {document:?} }}"
            );
            if id == "flash" {
                output += &format!("pub const FLASH: Geometry = {geometry};\n");
            }
            values.push(geometry);
        }
    }
    assert!(
        directories[0].geometries.iter().any(|g| g.id == "flash"),
        "missing served geometry flash"
    );
    output += &render_aliases(&merged_aliases(&directories));
    output += &format!(
        "pub(crate) const ALL: [Geometry; {}] = [{}];\npub static GEOMETRIES: &[Geometry] = &ALL;\n",
        values.len(),
        values.join(",")
    );
    let out = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    fs::write(out.join("geometries.rs"), output).unwrap();
}
