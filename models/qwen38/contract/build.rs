// Generate the supported geometries and their kernel profiles from
// `models/qwen38/geometries` (and the directories listed in
// `CHOPINFER_QWEN38_GEOMETRY_DIRS`) through the shared generator in
// `model_compiler_sdk::geometry_build`.
use model_compiler_sdk::geometry_build::{Catalog, Kind, Schema, field, merged_aliases};
use serde_json::Value;
#[path = "src/geometry_names.rs"]
mod geometry_names;
use std::{collections::BTreeSet, env, fs, path::PathBuf};

const LAYERS: Kind = Kind::Names(
    "LayerKind",
    &[("full_attention", "Full"), ("linear_attention", "Linear")],
);

const FIELDS: &[(&str, &str, Kind)] = &[
    ("shape.tokens.image", "image_token_id", Kind::U32),
    (
        "semantics.language_model_only",
        "language_model_only",
        Kind::Bool,
    ),
    (
        "semantics.attention_bias",
        "text_config.attention_bias",
        Kind::Bool,
    ),
    ("shape.tokens.bos", "text_config.bos_token_id", Kind::U32),
    ("semantics.dtype", "text_config.dtype", Kind::Str),
    ("shape.tokens.eos", "text_config.eos_token_id", Kind::U32),
    ("", "text_config.full_attention_interval", Kind::U32),
    ("shape.hc.streams", "text_config.hc_count", Kind::U32),
    ("shape.hc.rank", "text_config.hc_lowrank", Kind::U32),
    (
        "shape.attention.head_dim",
        "text_config.head_dim",
        Kind::U32,
    ),
    (
        "shape.ngram.heads_per_order",
        "text_config.heads_per_ngram",
        Kind::U32,
    ),
    ("shape.hidden", "text_config.hidden_size", Kind::U32),
    (
        "shape.indexer.compress_ratio",
        "text_config.indexer_compress_ratio",
        Kind::U32,
    ),
    (
        "shape.indexer.dim",
        "text_config.indexer_head_dim",
        Kind::U32,
    ),
    (
        "shape.indexer.kv_heads",
        "text_config.indexer_kv_heads",
        Kind::U32,
    ),
    (
        "shape.indexer.heads",
        "text_config.indexer_n_heads",
        Kind::U32,
    ),
    (
        "shape.linear.conv_taps",
        "text_config.linear_conv_kernel_dim",
        Kind::U32,
    ),
    (
        "shape.linear.key_dim",
        "text_config.linear_key_head_dim",
        Kind::U32,
    ),
    (
        "shape.linear.key_heads",
        "text_config.linear_num_key_heads",
        Kind::U32,
    ),
    (
        "shape.linear.value_heads",
        "text_config.linear_num_value_heads",
        Kind::U32,
    ),
    (
        "shape.linear.value_dim",
        "text_config.linear_value_head_dim",
        Kind::U32,
    ),
    (
        "shape.ngram.alignment",
        "text_config.make_ngram_vocab_size_divisible_by",
        Kind::U32,
    ),
    (
        "shape.context",
        "text_config.max_position_embeddings",
        Kind::U32,
    ),
    (
        "shape.moe.intermediate",
        "text_config.moe_intermediate_size",
        Kind::U32,
    ),
    ("semantics.mtp_hybrid", "text_config.mtp.hybrid", Kind::Bool),
    (
        "semantics.mtp_hidden_state_layer",
        "text_config.mtp.mtp_use_hidden_state_from_layer",
        Kind::OptionalU32,
    ),
    ("", "text_config.mtp.num_hidden_layers", Kind::U32),
    ("shape.mtp.theta", "text_config.mtp.rope_theta", Kind::F64),
    ("", "text_config.mtp_num_hidden_layers", Kind::U32),
    (
        "semantics.mtp_use_dedicated_embeddings",
        "text_config.mtp_use_dedicated_embeddings",
        Kind::Bool,
    ),
    ("shape.ngram.orders", "text_config.ngram_size", Kind::U32),
    (
        "shape.ngram.vocab_base",
        "text_config.ngram_vocab_size_base",
        Kind::U32,
    ),
    (
        "shape.attention.heads",
        "text_config.num_attention_heads",
        Kind::U32,
    ),
    ("shape.moe.experts", "text_config.num_experts", Kind::U32),
    (
        "shape.moe.top_k",
        "text_config.num_experts_per_tok",
        Kind::U32,
    ),
    ("", "text_config.num_hidden_layers", Kind::U32),
    (
        "shape.attention.kv_heads",
        "text_config.num_key_value_heads",
        Kind::U32,
    ),
    (
        "semantics.output_router_logits",
        "text_config.output_router_logits",
        Kind::Bool,
    ),
    (
        "shape.tokens.pad",
        "text_config.pad_token_id",
        Kind::OptionalU32,
    ),
    ("", "text_config.partial_rotary_factor", Kind::F64),
    (
        "shape.ple.conv_taps",
        "text_config.ple_conv_kernel_size",
        Kind::U32,
    ),
    (
        "shape.ple.embed_dim",
        "text_config.ple_embed_dim",
        Kind::U32,
    ),
    (
        "shape.ple.layer_ids",
        "text_config.ple_layer_ids",
        Kind::U32List,
    ),
    ("shape.attention.eps", "text_config.rms_norm_eps", Kind::F64),
    (
        "shape.attention.mrope",
        "text_config.rope_parameters.mrope_section",
        Kind::U32Array(3),
    ),
    (
        "",
        "text_config.rope_parameters.partial_rotary_factor",
        Kind::F64,
    ),
    (
        "shape.attention.theta",
        "text_config.rope_parameters.rope_theta",
        Kind::F64,
    ),
    (
        "shape.moe.shared_intermediate",
        "text_config.shared_expert_intermediate_size",
        Kind::U32,
    ),
    (
        "shape.ngram.parts",
        "text_config.split_ngram_parts",
        Kind::U32,
    ),
    ("", "text_config.tie_word_embeddings", Kind::Bool),
    ("semantics.use_cache", "text_config.use_cache", Kind::Bool),
    ("shape.vocab", "text_config.vocab_size", Kind::U32),
    (
        "semantics.tie_word_embeddings",
        "tie_word_embeddings",
        Kind::Bool,
    ),
    ("shape.tokens.video", "video_token_id", Kind::U32),
    (
        "shape.vision.deepstack",
        "vision_config.deepstack_visual_indexes",
        Kind::U32List,
    ),
    ("shape.vision.depth", "vision_config.depth", Kind::U32),
    (
        "shape.vision.hidden",
        "vision_config.hidden_size",
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
        "shape.vision.positions",
        "vision_config.num_position_embeddings",
        Kind::U32,
    ),
    (
        "shape.vision.out",
        "vision_config.out_hidden_size",
        Kind::U32,
    ),
    ("shape.vision.patch", "vision_config.patch_size", Kind::U32),
    (
        "shape.vision.merge",
        "vision_config.spatial_merge_size",
        Kind::U32,
    ),
    (
        "shape.vision.temporal_patch",
        "vision_config.temporal_patch_size",
        Kind::U32,
    ),
    ("shape.tokens.vision_end", "vision_end_token_id", Kind::U32),
    (
        "shape.tokens.vision_start",
        "vision_start_token_id",
        Kind::U32,
    ),
    ("shape.layers", "text_config.layer_types", LAYERS),
    ("shape.mtp.layers", "text_config.mtp.layer_types", LAYERS),
    (
        "semantics.norm_topk_prob",
        "text_config.norm_topk_prob",
        Kind::Bool,
    ),
    ("shape.attention.rotary", "", Kind::U32),
];

const GROUPS: &[(&str, &str)] = &[
    ("shape", "Shape"),
    ("shape.hc", "Hc"),
    ("shape.linear", "Linear"),
    ("shape.attention", "Attention"),
    ("shape.indexer", "Indexer"),
    ("shape.moe", "Moe"),
    ("shape.ngram", "Ngram"),
    ("shape.ple", "Ple"),
    ("shape.mtp", "Mtp"),
    ("shape.tokens", "Tokens"),
    ("shape.vision", "Vision"),
    ("semantics", "Semantics"),
];

const SCHEMA: Schema<Kind> = Schema {
    fields: FIELDS,
    groups: GROUPS,
};

fn declarations() -> String {
    SCHEMA.declarations()
}

fn runtime_fields(rules: &Value, limit: u64, budget: u64) -> String {
    rules["runtime_fields"]
    .as_object()
    .expect("runtime field declarations")
    .iter()
    .map(|(name, declaration)| {
        assert_eq!(name, "text_config.indexer_budget", "unsupported runtime field");
        let overridable = declaration["overridable"].as_bool().expect("overridable flag");
        assert!(!overridable,
            "runtime field {name} is overridable but the bundle does not read chop.model-options.v1");
        let minimum = declaration["minimum"].as_u64().expect("runtime field minimum");
        assert!(minimum > 0 && minimum <= budget && minimum <= limit && u32::try_from(minimum).is_ok(),
            "invalid runtime field minimum for {name}");
        format!("RuntimeField {{ name: {name:?}, overridable: {overridable}, minimum: {minimum}, maximum: {limit}, qualified: {budget} }}")
    })
    .collect::<Vec<_>>()
    .join(",")
}

fn main() {
    let directories = Catalog {
        variable: "CHOPINFER_QWEN38_GEOMETRY_DIRS",
        directory_keys: &[
            "aliases",
            "geometries",
            "runtime_fields",
            "semantic_rules",
            "engine_domain",
            "quantization_recipe_classes",
            "quant_method_default",
        ],
        geometry_keys: &[
            "id",
            "file",
            "kernel_tag",
            "pinned_names",
            "tuning",
            "limits",
            "semantic_rules",
            "quantization_recipe_classes",
        ],
    }
    .read();
    let mut output = declarations();
    output += &SCHEMA.compile_fields(&directories[0].supported);
    let mut values = Vec::new();
    let mut kernel_profiles = Vec::new();
    let mut model_configs = Vec::new();
    let mut tags = std::collections::BTreeMap::new();
    let mut documents = BTreeSet::new();
    for (directory_index, directory) in directories.iter().enumerate() {
        for geometry in &directory.geometries {
            let (id, path, config) = (geometry.id.as_str(), &geometry.path, &geometry.config);
            let (entry, rules) = (&geometry.declaration, &geometry.rules);
            geometry_names::validate(id, "geometry id").unwrap_or_else(|e| panic!("{e}"));
            let tag = entry["kernel_tag"]
                .as_str()
                .map(str::to_owned)
                .unwrap_or_else(|| format!("qwen38_{id}"));
            geometry_names::insert(&mut tags, id, &tag).unwrap_or_else(|e| panic!("{e}"));
            let pinned_names = entry["pinned_names"].as_bool().unwrap_or(false);
            let base_geometry = directory_index == 0;
            let tuning = if entry["tuning"].as_bool().unwrap_or(false) {
                let tuning_dir = if base_geometry {
                    PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap())
                        .join("tuning")
                        .join(id)
                } else {
                    directory.path.join("tuning").join(id)
                };
                let mut files = Vec::new();
                for name in ["dense", "moe", "schedules"] {
                    let path = tuning_dir.join(format!("{name}.json"));
                    println!("cargo:rerun-if-changed={}", path.display());
                    let content = fs::read_to_string(&path)
                        .unwrap_or_else(|e| panic!("{}: {e}", path.display()));
                    serde_json::from_str::<Value>(&content).expect("valid tuning JSON");
                    files.push(format!("{name}: {content:?}"));
                }
                format!("Some(TuningData {{ {} }})", files.join(","))
            } else {
                "None".into()
            };
            let mut fields = SCHEMA.literals(config, path);
            assert_eq!(
                field(config, "text_config.layer_types", path)
                    .as_array()
                    .unwrap()
                    .len() as u64,
                field(config, "text_config.num_hidden_layers", path)
                    .as_u64()
                    .unwrap(),
                "{}: layer count differs",
                path.display()
            );
            assert_eq!(
                field(config, "text_config.mtp.layer_types", path)
                    .as_array()
                    .unwrap()
                    .len() as u64,
                field(config, "text_config.mtp.num_hidden_layers", path)
                    .as_u64()
                    .unwrap(),
                "{}: MTP layer count differs",
                path.display()
            );
            let layers = field(config, "text_config.num_hidden_layers", path)
                .as_u64()
                .unwrap();
            let ple = field(config, "text_config.ple_layer_ids", path)
                .as_array()
                .unwrap();
            assert!(
                ple.len() == 1
                    && ple.iter().all(|layer| layer
                        .as_u64()
                        .is_some_and(|layer| layer > 0 && layer <= layers)),
                "{}: PLE requires one valid 1-based layer",
                path.display()
            );
            assert_eq!(
                field(config, "text_config.mtp.rope_theta", path).as_f64(),
                field(config, "text_config.rope_parameters.rope_theta", path).as_f64(),
                "{}: MTP theta must equal main theta",
                path.display()
            );
            SCHEMA.check_leaves(config, rules, path);
            let rotary = field(config, "text_config.head_dim", path)
                .as_u64()
                .unwrap() as f64
                * field(
                    config,
                    "text_config.rope_parameters.partial_rotary_factor",
                    path,
                )
                .as_f64()
                .unwrap();
            assert!(
                rotary.is_finite()
                    && rotary.fract() == 0.0
                    && rotary >= 0.0
                    && rotary <= u32::MAX as f64,
                "{}: invalid rotary dimension",
                path.display()
            );
            fields.insert(
                "shape.attention.rotary".into(),
                format!("{}", rotary as u32),
            );
            let fields = fields.into_iter().collect::<Vec<_>>();
            let shape = SCHEMA.grouped_value("shape", &fields);
            let semantics = SCHEMA.grouped_value("semantics", &fields);
            assert!(
                documents.insert(shape.clone()),
                "duplicate geometry shape {id}"
            );
            let limit = entry["limits"]["text_config.indexer_budget"]
                .as_u64()
                .unwrap();
            let budget = field(config, "text_config.indexer_budget", path)
                .as_u64()
                .unwrap();
            assert!(
                budget > 0 && budget == limit && u32::try_from(limit).is_ok(),
                "{id}: indexer_budget must equal compiled limit"
            );
            let runtime_fields = runtime_fields(rules, limit, budget);
            let rules = rules.to_string();
            let document = &geometry.document;
            let geometry = format!(
                "Geometry {{ base: {base_geometry}, pinned_names: {pinned_names}, tuning: {tuning}, id: {id:?}, kernel_tag: {tag:?}, shape: {shape}, semantics: {semantics}, limits: RuntimeLimits {{ indexer_budget: {limit} }}, runtime_fields: &[{runtime_fields}], rules: {rules:?}, document: {document:?} }}"
            );
            if id == "flash" {
                output += &format!("pub const FLASH: Geometry = {geometry};\n");
            }
            kernel_profiles.push(format!(
                "geometry_model!({id:?}, {tag:?}, {geometry}, {})",
                model_configs.len()
            ));
            model_configs.push(format!("geometry_config!({geometry})"));
            values.push(geometry);
        }
    }
    for (name, list) in merged_aliases(&directories) {
        output += &format!(
            "pub const {}_ALIASES: &[&str] = &[{}];\n",
            name.to_uppercase(),
            list.as_array()
                .unwrap()
                .iter()
                .map(Value::to_string)
                .collect::<Vec<_>>()
                .join(",")
        );
    }
    assert!(tags.values().any(|id| id == "flash"));
    output += &format!(
        "pub static GEOMETRIES: &[Geometry] = &[{}];\n",
        values.join(",")
    );
    output += &format!(
        "pub const MODEL_CONFIGS: [kernel_core::rank_shape::ModelConfig; {}] = [{}];\n",
        model_configs.len(),
        model_configs.join(",")
    );
    fs::write(
        PathBuf::from(env::var_os("OUT_DIR").unwrap()).join("kernel_profiles.rs"),
        format!("pub const MODELS: [GeometryModel; {}] = [{}];\npub const PROFILES: [Profile; {}] = profiles();\n", kernel_profiles.len(), kernel_profiles.join(","), kernel_profiles.len() * 2),
    ).unwrap();
    fs::write(
        PathBuf::from(env::var_os("OUT_DIR").unwrap()).join("geometries.rs"),
        output,
    )
    .unwrap();
}
