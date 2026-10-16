#[path = "src/kv_calibration.rs"]
mod kv_calibration;
use model_compiler_sdk::geometry_build::{Catalog, Schema, field, leaf_paths, read_json};
use serde_json::Value;
use std::{
    collections::BTreeSet,
    env, fs,
    path::{Path, PathBuf},
};
#[path = "src/geometry_checks.rs"]
mod geometry_checks;
#[path = "src/supported_ranks.rs"]
mod supported_ranks;
fn literal(v: &Value, ty: &str) -> String {
    if ty.starts_with("Option") {
        return if v.is_null() {
            "None".into()
        } else {
            format!(
                "Some({})",
                literal(
                    v,
                    ty.strip_prefix("Option<")
                        .unwrap()
                        .strip_suffix('>')
                        .unwrap()
                )
            )
        };
    }
    if ty == "f64" {
        return format!("{:?}", v.as_f64().expect("float field"));
    }
    if ty == "u32" {
        return u32::try_from(v.as_u64().expect("unsigned dimension"))
            .expect("u32 dimension")
            .to_string();
    }
    if ty.contains("[VisionLayerKind]") {
        return format!(
            "&[{}]",
            v.as_array()
                .unwrap()
                .iter()
                .map(|v| match v.as_i64() {
                    Some(-1) => "VisionLayerKind::Full",
                    Some(0) => "VisionLayerKind::Window",
                    Some(1) => "VisionLayerKind::MergeGroupWindow",
                    _ => panic!("unknown vision layer kind"),
                })
                .collect::<Vec<_>>()
                .join(",")
        );
    }
    if ty.contains("[LayerKind]") {
        return format!(
            "&[{}]",
            v.as_array()
                .unwrap()
                .iter()
                .map(|v| match v.as_u64() {
                    Some(0) => "LayerKind::Global",
                    Some(1) => "LayerKind::Sliding",
                    _ if v.as_str() == Some("sliding_attention") => "LayerKind::Sliding",
                    _ if v.as_str() == Some("full_attention") => "LayerKind::Global",
                    _ => panic!("unknown layer kind"),
                })
                .collect::<Vec<_>>()
                .join(",")
        );
    }
    if ty.starts_with("&'static [") {
        return format!(
            "&[{}]",
            v.as_array()
                .unwrap()
                .iter()
                .map(|v| literal(v, if ty.contains("u32") { "u32" } else { "other" }))
                .collect::<Vec<_>>()
                .join(",")
        );
    }
    v.to_string()
}
fn valid_literal(value: &Value, ty: &str) -> bool {
    if let Some(inner) = ty.strip_prefix("Option<").and_then(|s| s.strip_suffix('>')) {
        return value.is_null() || valid_literal(value, inner);
    }
    if ty.starts_with("&'static [") {
        let element = if ty.contains("VisionLayerKind") {
            "vision_layer"
        } else if ty.contains("LayerKind") {
            "layer"
        } else if ty.contains("u32") {
            "u32"
        } else if ty.contains("i32") {
            "i32"
        } else {
            "&'static str"
        };
        return value
            .as_array()
            .is_some_and(|a| a.iter().all(|v| valid_literal(v, element)));
    }
    match ty {
        "u32" => value.as_u64().is_some_and(|n| u32::try_from(n).is_ok()),
        "i32" => value.as_i64().is_some_and(|n| i32::try_from(n).is_ok()),
        "layer" => {
            value.as_u64().is_some_and(|n| n <= 1)
                || matches!(value.as_str(), Some("sliding_attention" | "full_attention"))
        }
        "vision_layer" => value.as_i64().is_some_and(|n| (-1..=1).contains(&n)),
        "f64" => value.as_f64().is_some_and(f64::is_finite),
        "bool" => value.is_boolean(),
        "&'static str" => value.is_string(),
        _ => false,
    }
}

fn literal_field(config: &Value, key: &str, ty: &str, path: &Path) -> String {
    let raw = field(config, key, path);
    let parsed;
    let value = if matches!(
        key,
        "audio_config.speech_vocab_size" | "audio_config.speech_zeroemb_idx"
    ) {
        parsed = Value::from(
            raw.as_str()
                .unwrap()
                .parse::<u32>()
                .expect("speech dimension"),
        );
        &parsed
    } else {
        raw
    };
    assert!(
        valid_literal(value, ty),
        "{}: {key}: expected {ty}, received {value}",
        path.display()
    );
    literal(value, ty)
}

const GROUPS: &[(&str, &str)] = &[
    ("shape", "Shape"),
    ("shape.text", "TextShape"),
    ("shape.global", "AttentionShape"),
    ("shape.sliding", "SlidingShape"),
    ("shape.moe", "MoeShape"),
    ("shape.vision", "VisionShape"),
    ("shape.audio", "AudioShape"),
    ("shape.audio_tokenizer", "AudioTokenizerShape"),
    ("shape.dflash", "DraftShape"),
    ("tokens", "TokenIds"),
    ("semantics", "Semantics"),
    ("semantics.attention", "AttentionSemantics"),
    ("semantics.dflash", "DraftSemantics"),
    ("limits", "RuntimeLimits"),
];

// Rust slot, configuration path (or derived field), Rust type.
const FIELDS: &[(&str, &str, &str)] = &[
    ("shape.text.hidden", "hidden_size", "u32"),
    ("shape.text.vocab", "vocab_size", "u32"),
    ("limits.context", "max_position_embeddings", "u32"),
    ("shape.text.mtp_layers", "num_nextn_predict_layers", "u32"),
    ("shape.global.heads", "num_attention_heads", "u32"),
    ("shape.global.kv_heads", "num_key_value_heads", "u32"),
    ("shape.global.head_dim", "head_dim", "u32"),
    ("shape.global.value_dim", "v_head_dim", "u32"),
    ("shape.global.theta", "rope_theta", "f64"),
    ("shape.global.rotary", "@rotary", "u32"),
    ("shape.sliding.heads", "swa_num_attention_heads", "u32"),
    ("shape.sliding.kv_heads", "swa_num_key_value_heads", "u32"),
    ("shape.sliding.head_dim", "swa_head_dim", "u32"),
    ("shape.sliding.value_dim", "swa_v_head_dim", "u32"),
    ("shape.sliding.theta", "swa_rope_theta", "f64"),
    ("shape.sliding.window", "sliding_window", "u32"),
    ("shape.moe.experts", "n_routed_experts", "u32"),
    ("shape.moe.top_k", "num_experts_per_tok", "u32"),
    ("shape.moe.intermediate", "moe_intermediate_size", "u32"),
    ("shape.text.dense_intermediate", "intermediate_size", "u32"),
    ("semantics.eps", "layernorm_epsilon", "f64"),
    (
        "semantics.attention.value_scale",
        "attention_value_scale",
        "f64",
    ),
    (
        "semantics.attention.global_sink",
        "add_full_attention_sink_bias",
        "bool",
    ),
    (
        "semantics.attention.sliding_sink",
        "add_swa_attention_sink_bias",
        "bool",
    ),
    ("shape.text.layers", "@layers", "&'static [Layer]"),
    ("tokens.eos", "eos_token_id", "u32"),
    ("tokens.image", "processor_config.image_token_id", "u32"),
    ("tokens.video", "processor_config.video_token_id", "u32"),
    ("tokens.audio", "processor_config.audio_token_id", "u32"),
    (
        "shape.vision.hidden_size",
        "vision_config.hidden_size",
        "u32",
    ),
    ("shape.vision.in_chans", "vision_config.in_chans", "u32"),
    (
        "shape.vision.intermediate_size",
        "vision_config.intermediate_size",
        "u32",
    ),
    ("shape.vision.num_heads", "vision_config.num_heads", "u32"),
    (
        "shape.vision.num_key_value_heads",
        "vision_config.num_key_value_heads",
        "u32",
    ),
    (
        "shape.vision.num_query_groups",
        "vision_config.num_query_groups",
        "u32",
    ),
    ("shape.vision.patch_size", "vision_config.patch_size", "u32"),
    (
        "shape.vision.spatial_merge_size",
        "vision_config.spatial_merge_size",
        "u32",
    ),
    (
        "shape.vision.spatial_patch_size",
        "vision_config.spatial_patch_size",
        "u32",
    ),
    (
        "shape.vision.temporal_patch_size",
        "vision_config.temporal_patch_size",
        "u32",
    ),
    (
        "shape.vision.tokens_per_second",
        "vision_config.tokens_per_second",
        "u32",
    ),
    (
        "shape.vision.visual_token_window_size",
        "vision_config.visual_token_window_size",
        "u32",
    ),
    (
        "shape.vision.vit_window_attn_types",
        "vision_config.vit_window_attn_types",
        "&'static [VisionLayerKind]",
    ),
    (
        "shape.vision.window_size",
        "vision_config.window_size",
        "u32",
    ),
    (
        "shape.audio.audio_channels",
        "audio_config.audio_channels",
        "u32",
    ),
    (
        "shape.audio.audio_segment_size",
        "audio_config.audio_segment_size",
        "u32",
    ),
    ("shape.audio.group_size", "audio_config.group_size", "u32"),
    (
        "shape.audio.input_local_attn_heads",
        "audio_config.input_local_attn_heads",
        "u32",
    ),
    (
        "shape.audio.input_local_dim",
        "audio_config.input_local_dim",
        "u32",
    ),
    (
        "shape.audio.input_local_head_dim",
        "audio_config.input_local_head_dim",
        "u32",
    ),
    (
        "shape.audio.input_local_intermediate_size",
        "audio_config.input_local_intermediate_size",
        "u32",
    ),
    (
        "shape.audio.input_local_layers",
        "audio_config.input_local_layers",
        "u32",
    ),
    (
        "shape.audio.projection_layers",
        "audio_config.projection_layers",
        "u32",
    ),
    ("shape.audio.rope_theta", "audio_config.rope_theta", "f64"),
    (
        "shape.audio.speech_vocab_size",
        "audio_config.speech_vocab_size",
        "u32",
    ),
    (
        "shape.audio.speech_zeroemb_idx",
        "audio_config.speech_zeroemb_idx",
        "u32",
    ),
    (
        "shape.audio_tokenizer.avg_pooler",
        "audio_tokenizer_config.avg_pooler",
        "u32",
    ),
    (
        "shape.audio_tokenizer.codebook_size",
        "audio_tokenizer_config.codebook_size",
        "&'static [u32]",
    ),
    (
        "shape.audio_tokenizer.d_model",
        "audio_tokenizer_config.d_model",
        "u32",
    ),
    (
        "shape.audio_tokenizer.encoder_attention_heads",
        "audio_tokenizer_config.encoder_attention_heads",
        "u32",
    ),
    (
        "shape.audio_tokenizer.encoder_attn_window_size",
        "audio_tokenizer_config.encoder_attn_window_size",
        "&'static [u32]",
    ),
    (
        "shape.audio_tokenizer.encoder_ffn_dim",
        "audio_tokenizer_config.encoder_ffn_dim",
        "u32",
    ),
    (
        "shape.audio_tokenizer.encoder_layers",
        "audio_tokenizer_config.encoder_layers",
        "u32",
    ),
    (
        "shape.audio_tokenizer.encoder_skip_layer_id",
        "audio_tokenizer_config.encoder_skip_layer_id",
        "u32",
    ),
    (
        "shape.audio_tokenizer.fmax",
        "audio_tokenizer_config.fmax",
        "Option<u32>",
    ),
    (
        "shape.audio_tokenizer.fmin",
        "audio_tokenizer_config.fmin",
        "u32",
    ),
    (
        "shape.audio_tokenizer.hop_length",
        "audio_tokenizer_config.hop_length",
        "u32",
    ),
    (
        "shape.audio_tokenizer.hybrid_block_size",
        "audio_tokenizer_config.hybrid_block_size",
        "u32",
    ),
    (
        "shape.audio_tokenizer.kernel_size",
        "audio_tokenizer_config.kernel_size",
        "u32",
    ),
    (
        "limits.max_audio_seconds",
        "audio_tokenizer_config.max_audio_seconds",
        "u32",
    ),
    (
        "shape.audio_tokenizer.n_mels",
        "audio_tokenizer_config.n_mels",
        "u32",
    ),
    (
        "shape.audio_tokenizer.nfft",
        "audio_tokenizer_config.nfft",
        "u32",
    ),
    (
        "shape.audio_tokenizer.num_quantizers",
        "audio_tokenizer_config.num_quantizers",
        "u32",
    ),
    (
        "shape.audio_tokenizer.rope_theta",
        "audio_tokenizer_config.rope_theta",
        "f64",
    ),
    (
        "shape.audio_tokenizer.sampling_rate",
        "audio_tokenizer_config.sampling_rate",
        "u32",
    ),
    (
        "shape.audio_tokenizer.stride_size",
        "audio_tokenizer_config.stride_size",
        "u32",
    ),
    (
        "shape.audio_tokenizer.swa_per_block",
        "audio_tokenizer_config.swa_per_block",
        "u32",
    ),
    (
        "shape.audio_tokenizer.window_size",
        "audio_tokenizer_config.window_size",
        "u32",
    ),
    (
        "shape.dflash.intermediate_size",
        "dflash.intermediate_size",
        "u32",
    ),
    (
        "shape.dflash.num_attention_heads",
        "dflash.num_attention_heads",
        "u32",
    ),
    (
        "shape.dflash.num_key_value_heads",
        "dflash.num_key_value_heads",
        "u32",
    ),
    ("shape.dflash.head_dim", "dflash.head_dim", "u32"),
    ("shape.dflash.v_head_dim", "dflash.v_head_dim", "u32"),
    ("shape.dflash.block_size", "dflash.block_size", "u32"),
    (
        "shape.dflash.sliding_window",
        "dflash.sliding_window",
        "u32",
    ),
    ("semantics.dflash.is_causal", "dflash.is_causal", "bool"),
    ("shape.dflash.rope_theta", "dflash.rope_theta", "f64"),
    (
        "semantics.dflash.rms_norm_eps",
        "dflash.rms_norm_eps",
        "f64",
    ),
    ("shape.dflash.rotary_dim", "@draft_rotary", "u32"),
    (
        "shape.dflash.layers",
        "dflash.layer_types",
        "&'static [LayerKind]",
    ),
    (
        "shape.dflash.target_layer_ids",
        "dflash.dflash_config.target_layer_ids",
        "&'static [u32]",
    ),
    (
        "tokens.dflash_mask",
        "dflash.dflash_config.mask_token_id",
        "u32",
    ),
    (
        "semantics.dflash.attention_value_scale",
        "dflash.dflash_config.attention_value_scale",
        "f64",
    ),
    (
        "semantics.dflash.attention_sink_bias",
        "dflash.dflash_config.attention_sink_bias",
        "bool",
    ),
];

const SCHEMA: Schema<&str> = Schema {
    fields: FIELDS,
    groups: GROUPS,
};

/// Every slot's literal, in table order.
fn values(config: &Value, path: &Path) -> Vec<(String, String)> {
    FIELDS
        .iter()
        .map(|&(slot, key, rust_type)| {
            let value = match key {
                "@rotary" => rotary(config, "", path).to_string(),
                "@draft_rotary" => rotary(config, "dflash.", path).to_string(),
                "@layers" => {
                    let attention = field(config, "hybrid_layer_pattern", path)
                        .as_array()
                        .unwrap();
                    let feed_forward = field(config, "moe_layer_freq", path).as_array().unwrap();
                    assert_eq!(attention.len(), feed_forward.len());
                    let layers = attention.iter().zip(feed_forward).map(|(a, f)| {
                        let attention = if a.as_u64() == Some(0) { "Global" } else { "Sliding" };
                        let feed_forward = if f.as_u64() == Some(0) { "Dense" } else { "Moe" };
                        format!("Layer {{ attention: LayerKind::{attention}, feed_forward: FeedForwardKind::{feed_forward} }}")
                    }).collect::<Vec<_>>().join(",");
                    format!("&[{layers}]")
                }
                _ => literal_field(config, key, rust_type, path),
            };
            (slot.to_owned(), value)
        })
        .collect()
}

fn rotary(config: &Value, prefix: &str, path: &Path) -> u32 {
    let value = (field(config, &format!("{prefix}head_dim"), path)
        .as_f64()
        .unwrap()
        * field(config, &format!("{prefix}partial_rotary_factor"), path)
            .as_f64()
            .unwrap())
    .floor();
    assert!(
        value.is_finite() && value >= 0.0 && value <= u32::MAX as f64,
        "invalid rotary dimension"
    );
    value as u32
}

fn main() {
    let checkpoint_path =
        PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap()).join("../checkpoints.json");
    let (_, checkpoints) = read_json(&checkpoint_path);
    println!("cargo:rerun-if-changed={}", checkpoint_path.display());
    let directories = Catalog {
        variable: "CHOPINFER_MIMO26_GEOMETRY_DIRS",
        directory_keys: &[
            "aliases",
            "geometries",
            "semantic_rules",
            "engine_domain",
            "quantization_recipe_classes",
        ],
        geometry_keys: &[
            "id",
            "file",
            "kernel_tag",
            "pinned_names",
            "encoders",
            "semantic_rules",
            "quantization_recipe_classes",
        ],
    }
    .read();
    let rules = &directories[0].supported;
    let mut output = SCHEMA.declarations();
    output += &format!("pub const SUPPORTED: &str = {:?};\n", directories[0].text);
    let mut tags = BTreeSet::new();
    let mut documents: Vec<Value> = Vec::new();
    let mut names = Vec::new();
    let mut calibration_geometries = std::collections::BTreeMap::new();
    let mut rule_documents = Vec::new();
    let mut base_fields = None;
    let mut compile_fields = Vec::new();
    let ranks = supported_ranks::SUPPORTED_RANKS;
    for (directory_index, directory) in directories.iter().enumerate() {
        let path = directory.path.join("supported.json");
        // The shared catalog walk checks engine_domain the same way.
        if let Some(value) = directory.supported.get("aliases") {
            assert_eq!(
                value,
                &rules["aliases"],
                "{}: differing aliases",
                path.display()
            );
        }
        for geometry in &directory.geometries {
            let (id, path, config) = (geometry.id.as_str(), &geometry.path, &geometry.config);
            if documents.is_empty() {
                assert_eq!(id, "v26-flash", "v26-flash must be the first geometry");
            }
            assert!(
                id.as_bytes().first().is_some_and(u8::is_ascii_lowercase)
                    && id
                        .bytes()
                        .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-')
                    && !id.ends_with('-')
                    && !id.contains("--")
                    && !id.rsplit_once("-tp").is_some_and(
                        |(_, n)| !n.is_empty() && n.bytes().all(|b| b.is_ascii_digit())
                    )
                    && ![
                        "supported",
                        "geometries",
                        "compile-fields",
                        "geometry-rules"
                    ]
                    .contains(&id),
                "{}: invalid geometry id {id}",
                path.display()
            );
            let mut leaves = BTreeSet::new();
            leaf_paths(config, "", &mut leaves);
            if let Some(expected) = &base_fields {
                for key in expected {
                    assert!(leaves.contains(key), "{}: missing {key}", path.display());
                }
                assert_eq!(
                    &leaves,
                    expected,
                    "{}: execution field set differs",
                    path.display()
                );
            } else {
                // The built-in schema is independent of supplied geometry documents.
                let expected: Value =
                    serde_json::from_str(include_str!("../geometries/v26-flash.json")).unwrap();
                let mut expected_fields = BTreeSet::new();
                leaf_paths(&expected, "", &mut expected_fields);
                for key in &expected_fields {
                    assert!(leaves.contains(key), "{}: missing {key}", path.display());
                }
                assert_eq!(
                    leaves,
                    expected_fields,
                    "{}: execution field set differs",
                    path.display()
                );
                let compile: Vec<_> = leaves
                    .iter()
                    .filter(|p| {
                        !p.starts_with("vision_config.")
                            && !p.starts_with("audio_config.")
                            && !p.starts_with("processor_config.")
                            && !p.starts_with("audio_tokenizer_config.")
                            && p.as_str() != "model_type"
                            && p.as_str() != "architectures"
                    })
                    .cloned()
                    .chain(
                        [
                            "vision_config",
                            "audio_config",
                            "processor_config",
                            "audio_tokenizer_config",
                        ]
                        .map(String::from),
                    )
                    .collect();
                output += &format!("pub const COMPILE_FIELDS: &[&str] = &{compile:?};\n");
                compile_fields = compile;
                base_fields = Some(leaves.clone());
            }
            let rules = &geometry.rules;
            geometry_checks::validate(config, rules, &ranks, geometry_checks::Component::All)
                .unwrap_or_else(|e| panic!("{}: {e}", path.display()));
            let refs: Vec<_> = compile_fields.iter().map(String::as_str).collect();
            let previous: Vec<_> = documents
                .iter()
                .map(|doc| ("previous", doc.clone()))
                .collect();
            assert!(
                model_compiler_sdk::geometry::match_geometry(config, &previous, &refs).is_err(),
                "duplicate execution document {id}"
            );
            let global_layers = config["hybrid_layer_pattern"]
                .as_array()
                .unwrap()
                .iter()
                .filter(|v| v.as_u64() == Some(0) || v.as_str() == Some("full_attention"))
                .count();
            calibration_geometries.insert(id.to_owned(), global_layers);
            documents.push(config.clone());
            let name = id.replace('-', "_").to_ascii_uppercase();
            let entry = &geometry.declaration;
            let tag = entry["kernel_tag"].as_str().expect("kernel tag");
            assert!(
                tag.as_bytes().first().is_some_and(u8::is_ascii_lowercase)
                    && tag
                        .bytes()
                        .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_'),
                "invalid kernel tag {tag}"
            );
            assert!(tags.insert(tag.to_owned()), "duplicate kernel tag {tag}");
            let encoders = entry["encoders"].as_bool().unwrap_or(false);
            let base_geometry = directory_index == 0;
            let pinned_names = entry["pinned_names"].as_bool().unwrap_or(false);
            let values = values(config, path);
            let document = &geometry.document;
            output += &format!(
                "pub const {name}: Geometry = Geometry {{ base: {base_geometry}, pinned_names: {pinned_names}, encoders: {encoders}, id: {id:?}, kernel_tag: {tag:?}, shape: {}, semantics: {}, tokens: {}, limits: {}, document: {document:?} }};\n",
                SCHEMA.grouped_value("shape", &values),
                SCHEMA.grouped_value("semantics", &values),
                SCHEMA.grouped_value("tokens", &values),
                SCHEMA.grouped_value("limits", &values)
            );
            names.push(name);
            rule_documents.push(rules.to_string());
        }
    }
    assert!(
        calibration_geometries.contains_key("v26-flash"),
        "missing served geometry v26-flash"
    );
    output += &format!(
        "pub const GEOMETRIES: &[Geometry] = &[{}];\n",
        names.join(",")
    );
    output += &format!("pub const GEOMETRY_RULES: &[&str] = &{rule_documents:?};\n");
    let tables = kv_calibration::tables(&checkpoints, &calibration_geometries)
        .expect("checkpoint KV calibration");
    let mut calibration = String::from("pub const COMPILED_FP8_KV: &[Fp8KvCalibration] = &[\n");
    for table in tables {
        calibration += &format!(
            "Fp8KvCalibration {{ geometry: {:?}, layout_id: b{:?}, scales: &{:?} }},\n",
            table.geometry, table.layout_id, table.scales
        );
    }
    calibration += "];\n";
    fs::write(
        PathBuf::from(env::var_os("OUT_DIR").unwrap()).join("kv-calibration.rs"),
        calibration,
    )
    .expect("write checkpoint KV calibration");
    let path = PathBuf::from(env::var_os("OUT_DIR").unwrap()).join("geometries.rs");
    fs::write(&path, output).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn optional_literals_preserve_the_declared_numeric_type() {
        assert_eq!(literal(&json!(2), "Option<f64>"), "Some(2.0)");
        assert_eq!(literal(&json!(2), "Option<u32>"), "Some(2)");
        assert_eq!(literal(&Value::Null, "Option<f64>"), "None");
        assert!(std::panic::catch_unwind(|| literal(&json!("2"), "u32")).is_err());
    }

    #[test]
    fn rotary_width_refuses_non_finite_and_out_of_range_values() {
        for factor in [-1.0, 1e20] {
            assert!(
                std::panic::catch_unwind(|| rotary(
                    &json!({"head_dim":192,"partial_rotary_factor":factor}),
                    "",
                    Path::new("shape.json")
                ))
                .is_err()
            );
        }
        assert_eq!(
            rotary(
                &json!({"head_dim":192,"partial_rotary_factor":0.334}),
                "",
                Path::new("shape.json")
            ),
            64
        );
    }
}
