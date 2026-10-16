include!("../build.rs");
use model_compiler_sdk::geometry_build::{geometry_rules, leaf_paths};

#[test]
fn require_named_fields_and_validate_types() {
    let _generate: fn() = main;
    let file = std::path::Path::new("geometry.json");
    let missing = std::panic::catch_unwind(|| {
        field(&serde_json::json!({}), "text.hidden", file);
    });
    let error = missing.unwrap_err();
    assert_eq!(
        error.downcast_ref::<String>().unwrap(),
        "geometry.json: missing text.hidden"
    );
    assert_eq!(Kind::OptionalU32.literal(&Value::Null, "pad", file), "None");
    assert_eq!(Kind::F64.literal(&serde_json::json!(1), "eps", file), "1.0");
    assert!(std::panic::catch_unwind(|| Kind::U32.literal(&Value::Null, "hidden", file)).is_err());
    assert!(
        std::panic::catch_unwind(|| Kind::U32List.literal(
            &serde_json::json!([1, -1]),
            "layers",
            file
        ))
        .is_err()
    );
}

#[test]
fn field_table_covers_shape_and_document() {
    let structs = declarations();
    assert!(structs.contains("pub struct Shape {"));
    assert!(structs.contains("pub hidden: u32,"));
    assert!(structs.contains("pub layers: &'static [LayerKind],"));
    assert!(structs.contains("pub theta: f64,"));
    assert!(!structs.contains("usize"));
    assert_eq!(
        FIELDS
            .iter()
            .filter(|(slot, _, _)| !slot.is_empty())
            .count(),
        FIELDS
            .iter()
            .filter(|(slot, _, _)| !slot.is_empty())
            .map(|(slot, _, _)| slot)
            .collect::<BTreeSet<_>>()
            .len()
    );
    let mut leaves = BTreeSet::new();
    leaf_paths(
        &serde_json::from_str(include_str!("../../geometries/flash.json")).unwrap(),
        "",
        &mut leaves,
    );
    let supported: Value =
        serde_json::from_str(include_str!("../../geometries/supported.json")).unwrap();
    assert_eq!(
        leaves,
        FIELDS
            .iter()
            .filter(|(_, path, _)| !path.is_empty())
            .map(|(_, path, _)| (*path).to_owned())
            .chain(
                supported["semantic_rules"]
                    .as_object()
                    .unwrap()
                    .keys()
                    .cloned()
            )
            .chain(
                supported["runtime_fields"]
                    .as_object()
                    .unwrap()
                    .keys()
                    .cloned()
            )
            .collect()
    );
}

#[test]
fn inherit_and_retain_geometry_rules() {
    let base: Value =
        serde_json::from_str(include_str!("../../geometries/supported.json")).unwrap();
    let file = std::path::Path::new("supported.json");
    let declaration = serde_json::json!({
        "semantic_rules": {"text_config.layer_types": ["full_attention"]},
        "quantization_recipe_classes": ["fp8"]
    });
    let rules = geometry_rules(&base, &declaration, file);
    assert_eq!(
        rules["semantic_rules"]["text_config.layer_types"],
        serde_json::json!(["full_attention"])
    );
    assert_eq!(
        rules["semantic_rules"]["text_config.hidden_act"],
        base["semantic_rules"]["text_config.hidden_act"]
    );
    assert_eq!(
        rules["quantization_recipe_classes"],
        serde_json::json!(["fp8"])
    );
    assert!(
        std::panic::catch_unwind(|| geometry_rules(
            &base,
            &serde_json::json!({"semantic_rules":{"unknown":[true]}}),
            file
        ))
        .is_err()
    );
    assert!(std::panic::catch_unwind(|| geometry_rules(&base, &serde_json::json!({"semantic_rules":{"text_config.mtp.layer_types":["linear_attention"]}}), file)).is_err());
}

#[test]
fn reject_truncating_sizes_and_accept_float_theta() {
    let file = std::path::Path::new("geometry.json");
    assert!(
        std::panic::catch_unwind(|| Kind::U32.literal(
            &serde_json::json!(4294967296_u64),
            "hidden",
            file
        ))
        .is_err()
    );
    assert!(
        std::panic::catch_unwind(|| Kind::U32List.literal(
            &serde_json::json!([4294967296_u64]),
            "layers",
            file
        ))
        .is_err()
    );
    let scientific: Value = serde_json::from_str("1e7").unwrap();
    assert_eq!(Kind::F64.literal(&scientific, "theta", file), "10000000.0");
    assert_eq!(
        LAYERS.literal(
            &serde_json::json!(["linear_attention", "full_attention"]),
            "layers",
            file
        ),
        "&[LayerKind::Linear,LayerKind::Full]"
    );
}

#[test]
fn reject_runtime_overrides_without_bundle_consumption() {
    let rules = serde_json::json!({"runtime_fields": {
        "text_config.indexer_budget": {"overridable": true, "minimum": 1}
    }});
    let error = std::panic::catch_unwind(|| runtime_fields(&rules, 2048, 2048)).unwrap_err();
    assert_eq!(
        error.downcast_ref::<String>().unwrap(),
        "runtime field text_config.indexer_budget is overridable but the bundle does not read chop.model-options.v1"
    );
}

#[test]
fn runtime_minimum_comes_from_data() {
    let rules = serde_json::json!({"runtime_fields": {
        "text_config.indexer_budget": {"overridable": false, "minimum": 2}
    }});
    assert!(runtime_fields(&rules, 2048, 2048).contains("minimum: 2"));
    let mut invalid = rules;
    invalid["runtime_fields"]["text_config.indexer_budget"]["minimum"] = 0.into();
    assert!(std::panic::catch_unwind(|| runtime_fields(&invalid, 2048, 2048)).is_err());
}
