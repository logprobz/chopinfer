//! Generate the supported geometries and their kernel profiles from
//! `models/glm5next/geometries` (and the directories listed in
//! `CHOPINFER_GLM5NEXT_GEOMETRY_DIRS`) through the shared generator in
//! `model_compiler_sdk::geometry_build`.
use model_compiler_sdk::geometry_build::{
    Catalog, Kind, RuntimeFieldSpec, Schema, TuningTables, merged_aliases, render_aliases,
    runtime_fields, runtime_limits, validate_name,
};
#[path = "src/structure.rs"]
mod structure;
use std::{collections::BTreeSet, env, fs, path::PathBuf};

const LAYERS: Kind = Kind::Names(
    "LayerKind",
    &[
        ("linear_attention", "Linear"),
        ("deepseek_sparse_attention", "Sparse"),
    ],
);
const MLP: Kind = Kind::Names("MlpKind", &[("dense", "Dense"), ("sparse", "Moe")]);

/// (generated slot, config path, kind). An empty slot is matched but not
/// generated; the build checks it against the slots it derives.
const FIELDS: &[(&str, &str, Kind)] = &[
    ("shape.tokens.image", "image_token_id", Kind::U32),
    ("shape.tokens.image_end", "image_end_token_id", Kind::U32),
    (
        "shape.tokens.image_start",
        "image_start_token_id",
        Kind::U32,
    ),
    (
        "semantics.tie_word_embeddings",
        "tie_word_embeddings",
        Kind::Bool,
    ),
    ("shape.tokens.video", "video_token_id", Kind::U32),
    ("shape.tokens.video_end", "video_end_token_id", Kind::U32),
    (
        "shape.tokens.video_start",
        "video_start_token_id",
        Kind::U32,
    ),
    (
        "semantics.attention_bias",
        "text_config.attention_bias",
        Kind::Bool,
    ),
    (
        "shape.tokens.eos",
        "text_config.eos_token_id",
        Kind::U32List,
    ),
    (
        "shape.dense_layers",
        "text_config.first_k_dense_replace",
        Kind::U32,
    ),
    ("shape.hc.eps", "text_config.hc_eps", Kind::F64),
    ("shape.hc.streams", "text_config.hc_mult", Kind::U32),
    (
        "shape.hc.sinkhorn_iterations",
        "text_config.hc_sinkhorn_iters",
        Kind::U32,
    ),
    ("", "text_config.head_dim", Kind::U32),
    ("shape.hidden", "text_config.hidden_size", Kind::U32),
    ("shape.indexer.dim", "text_config.index_head_dim", Kind::U32),
    ("shape.indexer.pool", "text_config.index_kpool", Kind::U32),
    (
        "semantics.index_pool_tail",
        "text_config.index_kpool_always_select_tail",
        Kind::Bool,
    ),
    (
        "semantics.index_pool_compress",
        "text_config.index_kpool_compress",
        Kind::Bool,
    ),
    (
        "shape.indexer.heads",
        "text_config.index_n_heads",
        Kind::U32,
    ),
    (
        "semantics.index_shared_across_drafts",
        "text_config.index_share_for_mtp_iteration",
        Kind::Bool,
    ),
    (
        "semantics.indexer_rope_interleave",
        "text_config.indexer_rope_interleave",
        Kind::Bool,
    ),
    (
        "shape.dense.intermediate",
        "text_config.intermediate_size",
        Kind::U32,
    ),
    (
        "shape.latent.kv_rank",
        "text_config.kv_lora_rank",
        Kind::U32,
    ),
    ("shape.layers", "text_config.layer_types", LAYERS),
    (
        "shape.linear.layers",
        "text_config.linear_attn_config.kda_layers",
        Kind::U32List,
    ),
    (
        "shape.linear.sparse_layers",
        "text_config.linear_attn_config.full_attn_layers",
        Kind::U32List,
    ),
    (
        "shape.linear.gate_lower_bound",
        "text_config.linear_attn_config.gate_lower_bound",
        Kind::F64,
    ),
    (
        "shape.linear.head_dim",
        "text_config.linear_attn_config.head_dim",
        Kind::U32,
    ),
    (
        "shape.linear.heads",
        "text_config.linear_attn_config.num_heads",
        Kind::U32,
    ),
    (
        "shape.linear.conv_taps",
        "text_config.linear_attn_config.short_conv_kernel_size",
        Kind::U32,
    ),
    (
        "shape.context",
        "text_config.max_position_embeddings",
        Kind::U32,
    ),
    ("semantics.hyper_connections", "text_config.mhc", Kind::Bool),
    (
        "semantics.latent_nope",
        "text_config.mla_use_nope",
        Kind::Bool,
    ),
    ("shape.mlp_layers", "text_config.mlp_layer_types", MLP),
    (
        "shape.moe.intermediate",
        "text_config.moe_intermediate_size",
        Kind::U32,
    ),
    ("shape.moe.groups", "text_config.n_group", Kind::U32),
    (
        "shape.moe.experts",
        "text_config.n_routed_experts",
        Kind::U32,
    ),
    (
        "shape.moe.shared",
        "text_config.n_shared_experts",
        Kind::U32,
    ),
    (
        "semantics.norm_topk_prob",
        "text_config.norm_topk_prob",
        Kind::Bool,
    ),
    (
        "shape.latent.heads",
        "text_config.num_attention_heads",
        Kind::U32,
    ),
    (
        "shape.moe.top_k",
        "text_config.num_experts_per_tok",
        Kind::U32,
    ),
    ("", "text_config.num_hidden_layers", Kind::U32),
    (
        "shape.latent.kv_heads",
        "text_config.num_key_value_heads",
        Kind::U32,
    ),
    (
        "shape.mtp_layers",
        "text_config.num_nextn_predict_layers",
        Kind::U32,
    ),
    (
        "semantics.output_router_logits",
        "text_config.output_router_logits",
        Kind::Bool,
    ),
    ("shape.tokens.pad", "text_config.pad_token_id", Kind::U32),
    ("shape.latent.q_rank", "text_config.q_lora_rank", Kind::U32),
    ("", "text_config.qk_head_dim", Kind::U32),
    (
        "shape.latent.qk_nope",
        "text_config.qk_nope_head_dim",
        Kind::U32,
    ),
    (
        "shape.latent.qk_rope",
        "text_config.qk_rope_head_dim",
        Kind::U32,
    ),
    ("shape.eps", "text_config.rms_norm_eps", Kind::F64),
    (
        "shape.moe.routed_scale",
        "text_config.routed_scaling_factor",
        Kind::F64,
    ),
    ("shape.swiglu_limit", "text_config.swiglu_limit", Kind::F64),
    ("", "text_config.tie_word_embeddings", Kind::Bool),
    ("shape.moe.top_groups", "text_config.topk_group", Kind::U32),
    ("semantics.use_cache", "text_config.use_cache", Kind::Bool),
    ("shape.latent.v_head", "text_config.v_head_dim", Kind::U32),
    ("shape.vocab", "text_config.vocab_size", Kind::U32),
    (
        "semantics.vision_attention_bias",
        "vision_config.attention_bias",
        Kind::Bool,
    ),
    ("shape.vision.depth", "vision_config.depth", Kind::U32),
    (
        "shape.vision.hidden",
        "vision_config.hidden_size",
        Kind::U32,
    ),
    (
        "shape.vision.image_size",
        "vision_config.image_size",
        Kind::U32,
    ),
    (
        "shape.vision.channels",
        "vision_config.in_channels",
        Kind::U32,
    ),
    (
        "shape.vision.intermediate",
        "vision_config.intermediate_size",
        Kind::U32,
    ),
    ("shape.vision.heads", "vision_config.num_heads", Kind::U32),
    (
        "shape.vision.out",
        "vision_config.out_hidden_size",
        Kind::U32,
    ),
    ("shape.vision.patch", "vision_config.patch_size", Kind::U32),
    (
        "shape.vision.projection_intermediate",
        "vision_config.projection_intermediate_size",
        Kind::U32,
    ),
    ("shape.vision.eps", "vision_config.rms_norm_eps", Kind::F64),
    (
        "shape.vision.merge",
        "vision_config.spatial_merge_size",
        Kind::U32,
    ),
    (
        "shape.vision.swiglu_limit",
        "vision_config.swiglu_limit",
        Kind::F64,
    ),
    (
        "shape.vision.temporal_patch",
        "vision_config.temporal_patch_size",
        Kind::U32,
    ),
];

const GROUPS: &[(&str, &str)] = &[
    ("shape", "Shape"),
    ("shape.hc", "Hc"),
    ("shape.linear", "Linear"),
    ("shape.latent", "Latent"),
    ("shape.indexer", "Indexer"),
    ("shape.dense", "Dense"),
    ("shape.moe", "Moe"),
    ("shape.tokens", "Tokens"),
    ("shape.vision", "Vision"),
    ("semantics", "Semantics"),
];

/// Rank counts each geometry compiles kernel profiles for.
const SERVED_RANKS: &[u32] = &[4];

/// Supported runtime fields: (config path, `RuntimeValues` member, the
/// config path of the step every value is a multiple of).
const RUNTIME_FIELDS: &[RuntimeFieldSpec] = &[(
    "text_config.index_topk",
    "index_topk",
    "text_config.index_kpool",
)];

const SCHEMA: Schema<Kind> = Schema {
    fields: FIELDS,
    groups: GROUPS,
};

fn main() {
    let directories = Catalog {
        variable: "CHOPINFER_GLM5NEXT_GEOMETRY_DIRS",
        directory_keys: &[
            "aliases",
            "geometries",
            "runtime_fields",
            "semantic_rules",
            "quantization_recipe_classes",
        ],
        geometry_keys: &[
            "id",
            "file",
            "kernel_tag",
            "limits",
            "semantic_rules",
            "quantization_recipe_classes",
        ],
    }
    .read();
    let mut output = SCHEMA.declarations();
    output += &SCHEMA.compile_fields(&directories[0].supported);
    let mut values = Vec::new();
    let mut profiles = Vec::new();
    let mut tags = BTreeSet::new();
    let mut shapes = BTreeSet::new();
    let mut tuning = TuningTables::default();
    for (directory_index, directory) in directories.iter().enumerate() {
        for geometry in &directory.geometries {
            let (id, path, config) = (geometry.id.as_str(), &geometry.path, &geometry.config);
            validate_name(id, "geometry id");
            let tag = geometry.declaration["kernel_tag"]
                .as_str()
                .map(str::to_owned)
                .unwrap_or_else(|| format!("glm5next_{id}"));
            validate_name(&tag, "kernel tag");
            assert!(tags.insert(tag.clone()), "duplicate kernel tag {tag}");
            let fields = SCHEMA.literals(config, path);
            structure::check(config).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
            SCHEMA.check_leaves(config, &geometry.rules, path);
            let fields = fields.into_iter().collect::<Vec<_>>();
            let shape = SCHEMA.grouped_value("shape", &fields);
            let semantics = SCHEMA.grouped_value("semantics", &fields);
            assert!(
                shapes.insert(shape.clone()),
                "duplicate geometry shape {id}"
            );
            let runtime = runtime_fields(RUNTIME_FIELDS, geometry, &geometry.rules);
            let limits = runtime_limits(RUNTIME_FIELDS, &geometry.declaration);
            let base_geometry = directory_index == 0;
            tuning.add(
                id,
                &if base_geometry {
                    directory.path.join("../contract/tuning")
                } else {
                    directory.path.join("tuning")
                },
            );
            let rules = geometry.rules.to_string();
            let document = &geometry.document;
            let geometry = format!(
                "Geometry {{ base: {base_geometry}, id: {id:?}, kernel_tag: {tag:?}, shape: {shape}, semantics: {semantics}, limits: RuntimeLimits {{ {limits} }}, runtime_fields: &[{runtime}], rules: {rules:?}, document: {document:?} }}"
            );
            if id == "flash" {
                output += &format!("pub const FLASH: Geometry = {geometry};\n");
            }
            for ranks in SERVED_RANKS {
                profiles.push(format!(
                    "KernelProfile::new(&crate::geometry::ALL[{}], {ranks})",
                    values.len()
                ));
            }
            values.push(geometry);
        }
    }
    output += &render_aliases(&merged_aliases(&directories));
    assert!(directories.iter().any(|directory| {
        directory
            .geometries
            .iter()
            .any(|geometry| geometry.id == "flash")
    }));
    output += &format!(
        "pub(crate) const ALL: [Geometry; {}] = [{}];\npub static GEOMETRIES: &[Geometry] = &ALL;\n",
        values.len(),
        values.join(",")
    );
    let out = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    fs::write(
        out.join("kernel_profiles.rs"),
        format!(
            "pub const SERVED_RANKS: &[u32] = &{SERVED_RANKS:?};\npub static PROFILES: [KernelProfile; {}] = [{}];\n",
            profiles.len(),
            profiles.join(",")
        ),
    )
    .unwrap();
    fs::write(out.join("geometries.rs"), output).unwrap();
    fs::write(out.join("tuning.rs"), tuning.render()).unwrap();
}
