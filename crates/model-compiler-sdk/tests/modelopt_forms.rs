//! ModelOpt's single-algorithm `config.json` form (config groups that target
//! the `Linear` module class, the group size stated only in the groups) and
//! export KV cache hints, on hand-written configs.
use kernel_core::contract::WeightFormat;
use model_compiler_sdk::quantization::import::{
    ActivationRecipe, EncodingRecipe, ExportMetadata, ImportOptions, KvHint, KvHintPolicy,
    LINEAR_CLASS_ROLE, ROUTED_EXPERT_ROLE,
};
use serde_json::{Value, json};

fn nvfp4(dynamic: bool) -> Value {
    json!({"dynamic": dynamic, "num_bits": 4, "type": "float", "group_size": 16})
}

fn fp8_kv() -> Value {
    json!({"dynamic": false, "num_bits": 8, "type": "float"})
}

/// A class-form export: every `Linear` except the attention of block 0 and
/// the head stores NVFP4 weights with static NVFP4 inputs.
fn class_config() -> Value {
    json!({
        "producer": {"name": "modelopt", "version": "fixture"},
        "quant_method": "modelopt",
        "quant_algo": "NVFP4",
        "config_groups": {"group_0": {
            "weights": nvfp4(false),
            "input_activations": nvfp4(false),
            "targets": ["Linear"],
        }},
        "ignore": ["head", "blocks.0.attn*"],
    })
}

fn import(config: &Value) -> Result<ExportMetadata, String> {
    ExportMetadata::from_documents(Some(config), None).map_err(|e| e.to_string())
}

fn import_with(config: &Value, kv_hint: KvHintPolicy) -> Result<ExportMetadata, String> {
    ExportMetadata::from_documents_with(Some(config), None, &ImportOptions { kv_hint })
        .map_err(|e| e.to_string())
}

fn recipe(metadata: &ExportMetadata, targets: &[&str]) -> EncodingRecipe {
    let targets: Vec<String> = targets.iter().map(|t| t.to_string()).collect();
    metadata.recipe(&targets, None).unwrap()
}

const STATIC_NVFP4: EncodingRecipe = EncodingRecipe {
    weight: WeightFormat::Nvfp4,
    activation: Some(ActivationRecipe::Nvfp4Static),
};

fn bf16() -> EncodingRecipe {
    EncodingRecipe::bf16()
}

#[test]
fn a_linear_class_group_reaches_marked_roles_only() {
    let metadata = import(&class_config()).unwrap();
    // The top-level algorithm restates the group; it is no default.
    assert_eq!(metadata.default, None);
    assert!(metadata.rules.is_empty());
    let up = "blocks.1.mlp.up";
    assert_eq!(recipe(&metadata, &[up, LINEAR_CLASS_ROLE]), STATIC_NVFP4);
    assert_eq!(
        recipe(&metadata, &[up, LINEAR_CLASS_ROLE, ROUTED_EXPERT_ROLE]),
        STATIC_NVFP4
    );
    // An unmarked role (a norm, a table) is not a Linear.
    assert_eq!(recipe(&metadata, &[up]), bf16());
    assert_eq!(recipe(&metadata, &["blocks.1.norm"]), bf16());
    // An exclusion wins over the class rule and is no conflict.
    for excluded in ["blocks.0.attn.q", "head"] {
        assert_eq!(recipe(&metadata, &[excluded, LINEAR_CLASS_ROLE]), bf16());
    }
    // A family's declared recipe must agree with the class rule.
    let targets = vec![up.to_string(), LINEAR_CLASS_ROLE.to_string()];
    assert!(metadata.recipe(&targets, Some(bf16())).is_err());
    // The class rule does not reach a side tensor the family reads raw.
    metadata
        .require_unconverted("blocks.1.norm.weight")
        .unwrap();
    metadata
        .require_unconverted("blocks.1.mlp.up.weight")
        .unwrap();
    // A marker alone is no source target.
    assert!(
        metadata
            .recipe(&[LINEAR_CLASS_ROLE.to_string()], None)
            .is_err()
    );
}

#[test]
fn a_weight_only_class_group_declares_bf16_inputs() {
    let mut config = class_config();
    config["quant_algo"] = "W4A16_NVFP4".into();
    config["config_groups"]["group_0"]
        .as_object_mut()
        .unwrap()
        .remove("input_activations");
    let metadata = import(&config).unwrap();
    assert_eq!(
        recipe(&metadata, &["blocks.1.mlp.up", LINEAR_CLASS_ROLE]),
        EncodingRecipe {
            weight: WeightFormat::Nvfp4,
            activation: Some(ActivationRecipe::Bf16),
        }
    );
    assert_eq!(recipe(&metadata, &["blocks.1.mlp.up"]), bf16());
}

#[test]
fn the_group_size_may_be_stated_only_in_the_groups() {
    // Path-targeted groups: the top-level algorithm stays the default.
    let config = json!({
        "quant_method": "modelopt",
        "quant_algo": "NVFP4",
        "config_groups": {
            "experts": {"weights": nvfp4(false), "input_activations": nvfp4(false),
                        "targets": ["blocks.*.experts"]},
            "mlp": {"weights": nvfp4(false), "input_activations": nvfp4(false),
                    "targets": ["blocks.0.mlp"]},
        },
    });
    let metadata = import(&config).unwrap();
    assert_eq!(metadata.default, Some(STATIC_NVFP4));
    assert_eq!(recipe(&metadata, &["blocks.3.experts.0.up"]), STATIC_NVFP4);
    // A group stating another size is refused, naming it.
    let mut config = config;
    config["config_groups"]["mlp"] = json!({
        "weights": {"dynamic": false, "num_bits": 8, "type": "float", "group_size": 32},
        "targets": ["blocks.0.mlp"],
    });
    let error = import(&config).unwrap_err();
    assert!(
        error.contains("config_groups/mlp") && error.contains("group_size"),
        "{error}"
    );
}

#[test]
fn a_single_group_whose_size_disagrees_with_the_algorithm_is_refused() {
    let mxfp8 = json!({"dynamic": true, "num_bits": 8, "type": "float", "group_size": 32});
    let mut config = json!({
        "quant_method": "modelopt",
        "quant_algo": "NVFP4",
        "config_groups": {"wrong_size": {"weights": {"dynamic": false, "num_bits": 8,
            "type": "float", "group_size": 32}, "input_activations": mxfp8,
            "targets": ["blocks.0.mlp"]}},
    });
    let error = import(&config).unwrap_err();
    assert!(
        error.contains("config_groups/wrong_size") && error.contains("group_size"),
        "{error}"
    );
    // A group that states no size where the algorithm requires one.
    config["config_groups"]["wrong_size"]["weights"]
        .as_object_mut()
        .unwrap()
        .remove("group_size");
    let error = import(&config).unwrap_err();
    assert!(
        error.contains("config_groups/wrong_size") && error.contains("group_size"),
        "{error}"
    );
    // The same in the class form.
    let mut config = class_config();
    config["config_groups"]["group_0"]["weights"]["group_size"] = 32.into();
    let error = import(&config).unwrap_err();
    assert!(
        error.contains("config_groups/group_0") && error.contains("group_size"),
        "{error}"
    );
}

#[test]
fn groups_that_disagree_with_the_top_level_algorithm_are_refused() {
    // Weight-only at the top level, static inputs in the group.
    let mut config = class_config();
    config["quant_algo"] = "W4A16_NVFP4".into();
    let error = import(&config).unwrap_err();
    assert!(
        error.contains("config_groups/group_0") && error.contains("disagrees with quant_algo"),
        "{error}"
    );
    // Static inputs at the top level, dynamic inputs in the group.
    let mut config = class_config();
    config["config_groups"]["group_0"]["input_activations"] = nvfp4(true);
    let error = import(&config).unwrap_err();
    assert!(error.contains("config_groups/group_0"), "{error}");
    // The same holds for path-targeted groups whose size the top level
    // takes from them.
    let config = json!({
        "quant_method": "modelopt",
        "quant_algo": "W4A16_NVFP4",
        "config_groups": {"mlp": {"weights": nvfp4(false), "input_activations": nvfp4(false),
                                  "targets": ["blocks.0.mlp"]}},
    });
    let error = import(&config).unwrap_err();
    assert!(error.contains("config_groups/mlp"), "{error}");
}

#[test]
fn a_class_group_beside_path_targets_is_refused() {
    let mut config = class_config();
    config["config_groups"]["group_1"] = json!({
        "weights": nvfp4(false), "input_activations": nvfp4(false),
        "targets": ["blocks.0.mlp"],
    });
    let error = import(&config).unwrap_err();
    assert!(
        error.contains("config_groups/group_0") && error.contains("module paths"),
        "{error}"
    );
    // One group naming both.
    let mut config = class_config();
    config["config_groups"]["group_0"]["targets"] = json!(["Linear", "blocks.0.mlp"]);
    let error = import(&config).unwrap_err();
    assert!(
        error.contains("config_groups/group_0") && error.contains("module paths"),
        "{error}"
    );
    // Per-layer declarations beside a class group.
    let mut config = class_config();
    config["quant_algo"] = "MIXED_PRECISION".into();
    config["quantized_layers"] = json!({"blocks.0.mlp": {"quant_algo": "NVFP4", "group_size": 16}});
    let error = import(&config).unwrap_err();
    assert!(error.contains("config_groups/group_0"), "{error}");
    // A default from a second document.
    let hf = br#"{"producer": {"name": "modelopt", "version": "fixture"},
        "quantization": {"quant_algo": "NVFP4", "group_size": 16}}"#;
    let error = ExportMetadata::from_documents(Some(&class_config()), Some(hf))
        .unwrap_err()
        .to_string();
    assert!(error.contains("module class Linear"), "{error}");
}

#[test]
fn a_group_targeting_another_module_class_is_refused() {
    let mut config = class_config();
    config["config_groups"]["group_0"]["targets"] = json!(["Embedding"]);
    let error = import(&config).unwrap_err();
    assert!(
        error.contains("config_groups/group_0") && error.contains("Embedding"),
        "{error}"
    );
    // In a mixed-precision document too.
    let config = json!({
        "quant_method": "modelopt",
        "quant_algo": "MIXED_PRECISION",
        "config_groups": {"tables": {"weights": nvfp4(false), "targets": ["Embedding"]}},
    });
    let error = import(&config).unwrap_err();
    assert!(error.contains("config_groups/tables"), "{error}");
}

#[test]
fn explicit_rules_on_side_tensors_are_refused() {
    let config = json!({
        "quant_method": "modelopt",
        "quant_algo": "MIXED_PRECISION",
        "quantized_layers": {
            "blocks.0.mlp": {"quant_algo": "NVFP4", "group_size": 16},
            "blocks.0.norm": {"quant_algo": "NVFP4", "group_size": 16},
        },
        "ignore": ["blocks.0.mlp.down"],
    });
    let metadata = import(&config).unwrap();
    assert!(
        metadata
            .require_unconverted("blocks.0.norm.weight")
            .unwrap_err()
            .to_string()
            .contains("blocks.0.norm")
    );
    // A rule and an exclusion on the same side tensor conflict.
    assert!(
        metadata
            .require_unconverted("blocks.0.mlp.down.weight")
            .unwrap_err()
            .to_string()
            .contains("both quantized and excluded")
    );
    metadata
        .require_unconverted("blocks.1.norm.weight")
        .unwrap();
}

#[test]
fn a_kv_hint_is_refused_without_the_policy() {
    let mut config = class_config();
    config["kv_cache_scheme"] = fp8_kv();
    let error = import(&config).unwrap_err();
    assert!(error.contains("kv_cache_scheme"), "{error}");
    let hf = br#"{"producer": {"name": "modelopt", "version": "fixture"},
        "quantization": {"quant_algo": "NVFP4", "group_size": 16, "kv_cache_quant_algo": "FP8"}}"#;
    let error = ExportMetadata::from_documents(None, Some(hf))
        .unwrap_err()
        .to_string();
    assert!(error.contains("independent supported KV policy"), "{error}");
    // A null algorithm is no hint.
    let hf = br#"{"quantization": {"quant_algo": "NVFP4", "group_size": 16,
        "kv_cache_quant_algo": null}}"#;
    let metadata = ExportMetadata::from_documents(None, Some(hf)).unwrap();
    assert_eq!(metadata.kv_hint, None);
}

#[test]
fn the_static_fp8_kv_hint_is_recorded_and_ignored_under_the_policy() {
    let mut config = class_config();
    config["kv_cache_scheme"] = fp8_kv();
    let recorded = import_with(&config, KvHintPolicy::RecordAndIgnore).unwrap();
    assert_eq!(recorded.kv_hint, Some(KvHint::Fp8Static));
    let plain = import(&class_config()).unwrap();
    assert_eq!(
        (
            &recorded.rules,
            &recorded.class_rules,
            &recorded.default,
            &recorded.exclusions
        ),
        (
            &plain.rules,
            &plain.class_rules,
            &plain.default,
            &plain.exclusions
        )
    );
    let hf = br#"{"producer": {"name": "modelopt", "version": "fixture"},
        "quantization": {"quant_algo": "NVFP4", "group_size": 16, "kv_cache_quant_algo": "FP8"}}"#;
    let options = ImportOptions {
        kv_hint: KvHintPolicy::RecordAndIgnore,
    };
    let metadata = ExportMetadata::from_documents_with(None, Some(hf), &options).unwrap();
    assert_eq!(metadata.kv_hint, Some(KvHint::Fp8Static));
    // Without a hint nothing is recorded, and nothing new is serialized.
    let metadata = import_with(&class_config(), KvHintPolicy::RecordAndIgnore).unwrap();
    assert_eq!(metadata.kv_hint, None);
    let hf = br#"{"quantization": {"quant_algo": "NVFP4", "group_size": 16}}"#;
    let metadata = ExportMetadata::from_documents_with(None, Some(hf), &options).unwrap();
    let serialized = serde_json::to_value(&metadata).unwrap();
    assert!(
        serialized.get("kv_hint").is_none() && serialized.get("class_rules").is_none(),
        "{serialized}"
    );
    let restored: ExportMetadata = serde_json::from_value(serialized).unwrap();
    assert_eq!(restored, metadata);
}

#[test]
fn other_kv_hint_forms_are_refused_under_the_policy() {
    for hint in [
        Value::Null,
        false.into(),
        json!({"dynamic": true, "num_bits": 8, "type": "float"}),
        json!({"dynamic": false, "num_bits": 4, "type": "float"}),
        json!({"dynamic": false, "num_bits": 8, "type": "int"}),
        json!({"dynamic": false, "num_bits": 8, "type": "float", "strategy": "tensor"}),
    ] {
        let mut config = class_config();
        config["kv_cache_scheme"] = hint.clone();
        let error = import_with(&config, KvHintPolicy::RecordAndIgnore).unwrap_err();
        assert!(error.contains("kv_cache_scheme"), "{hint}: {error}");
    }
    // An unsupported algorithm, or a value that is not a string, is
    // refused naming the field under either policy.
    for algorithm in [json!("NVFP4"), json!(false), json!({"algo": "FP8"})] {
        let hf = serde_json::to_vec(&json!({"quantization": {"quant_algo": "NVFP4",
            "group_size": 16, "kv_cache_quant_algo": algorithm}}))
        .unwrap();
        for kv_hint in [KvHintPolicy::RecordAndIgnore, KvHintPolicy::Refuse] {
            let error =
                ExportMetadata::from_documents_with(None, Some(&hf), &ImportOptions { kv_hint })
                    .unwrap_err()
                    .to_string();
            if algorithm.is_string() && kv_hint == KvHintPolicy::Refuse {
                assert!(error.contains("independent supported KV policy"), "{error}");
            } else {
                assert!(
                    error.contains("hf_quant_config.json/kv_cache_quant_algo:")
                        && !error.contains("invalid type"),
                    "{algorithm}: {error}"
                );
            }
        }
    }
    // A null algorithm is no hint under the policy too.
    let hf = br#"{"quantization": {"quant_algo": "NVFP4", "group_size": 16,
        "kv_cache_quant_algo": null}}"#;
    let options = ImportOptions {
        kv_hint: KvHintPolicy::RecordAndIgnore,
    };
    let metadata = ExportMetadata::from_documents_with(None, Some(hf), &options).unwrap();
    assert_eq!(metadata.kv_hint, None);
}
