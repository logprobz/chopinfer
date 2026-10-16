use model_mimo26_contract::geometry::{V26_FLASH, match_dflash, match_main};
use serde_json::{Value, json};

fn document() -> Value {
    serde_json::from_str(V26_FLASH.document).unwrap()
}

#[test]
fn report_main_rules_before_closed_shape_differences() {
    for (pointer, value, expected) in [
        (
            "/attention_projection_layout",
            json!("separate"),
            "unsupported attention_projection_layout: expected [\"fused_qkv\"], received \"separate\"",
        ),
        (
            "/dtype",
            json!("float16"),
            "unsupported dtype: expected [\"bfloat16\"], received \"float16\"",
        ),
        (
            "/attention_bias",
            json!(true),
            "unsupported attention_bias: expected [false], received true",
        ),
        (
            "/n_shared_experts",
            json!(1),
            "unsupported n_shared_experts: expected [null], received 1",
        ),
        (
            "/n_group",
            json!(2),
            "unsupported n_group: expected [1], received 2",
        ),
        (
            "/hybrid_layer_pattern/0",
            json!(2),
            "unsupported hybrid_layer_pattern[0]: expected [0,1], received 2",
        ),
        (
            "/num_attention_heads",
            json!(128),
            "unsupported num_attention_heads: expected GQA group at most 8 or exactly 16 (attention engine), KV heads 4, received 128",
        ),
        (
            "/num_attention_heads",
            json!(6),
            "unsupported num_attention_heads: expected positive multiple of num_key_value_heads (4), received 6",
        ),
        (
            "/num_key_value_heads",
            json!(2),
            "unsupported num_attention_heads: expected GQA group at most 8 or exactly 16 (attention engine), KV heads 2, received 64",
        ),
        (
            "/num_experts_per_tok",
            json!(257),
            "unsupported num_experts_per_tok: expected 1..=256, received 257",
        ),
        (
            "/vision_config/depth",
            json!(28.0),
            "unsupported vision_config.depth: expected positive integer, received 28.0",
        ),
        (
            "/vision_config/fullatt_block_indexes",
            json!([]),
            "unsupported vision_config.fullatt_block_indexes: expected [0,9,18,27], received []",
        ),
        (
            "/model_type",
            json!("other"),
            "unsupported model_type: expected [\"mimo_v2\"], received \"other\"",
        ),
        (
            "/architectures",
            json!([]),
            "unsupported architectures: expected [\"MiMoV2ForCausalLM\"], received []",
        ),
        (
            "/moe_router_dtype",
            json!("float32"),
            "unsupported moe_router_dtype: expected [\"bfloat16\"], received \"float32\"",
        ),
        (
            "/image_token_id",
            json!(1),
            "unsupported image_token_id: expected 151655, received 1",
        ),
        (
            "/quantization_config",
            json!({"quant_method":"awq"}),
            "unsupported quantization_config.quant_method: expected [\"fp8\",\"modelopt\"], received \"awq\"",
        ),
        (
            "/hidden_size",
            json!(2048),
            "unsupported vision_config.out_hidden_size: expected 2048, received 4096",
        ),
        (
            "/max_position_embeddings",
            json!(4194304),
            "unsupported max_position_embeddings: expected 1..=1048576 (attention prepared capacity exceeds catalog limits), received 4194304",
        ),
    ] {
        let mut config = document();
        if pointer == "/quantization_config" {
            config["quantization_config"] = value;
        } else {
            *config.pointer_mut(pointer).unwrap() = value;
        }
        assert_eq!(match_main(&config).unwrap_err(), expected, "{pointer}");
    }
    let mut config = document();
    config.as_object_mut().unwrap().remove("architectures");
    assert_eq!(
        match_main(&config).unwrap_err(),
        "unsupported architectures: expected [\"MiMoV2ForCausalLM\"], received null"
    );
    config = document();
    config["num_attention_heads"] = json!(6);
    config["num_key_value_heads"] = json!(6);
    assert_eq!(
        match_main(&config).unwrap_err(),
        "unsupported num_attention_heads: expected divisible by supported TP4, received 6"
    );
}

#[test]
fn report_drafter_rules_before_closed_shape_differences() {
    for (pointer, value, expected) in [
        (
            "/layer_types/0",
            json!("full_attention"),
            "unsupported dflash.layer_types[0]: expected [\"sliding_attention\"], received \"full_attention\"",
        ),
        (
            "/model_type",
            json!("llama"),
            "unsupported dflash.model_type: expected [\"qwen3\"], received \"llama\"",
        ),
        (
            "/architectures",
            json!(["Other"]),
            "unsupported dflash.architectures[0]: expected [\"DFlashDraftModel\"], received \"Other\"",
        ),
        (
            "/dflash_config/target_layer_ids/0",
            json!(48),
            "unsupported dflash.dflash_config.target_layer_ids: expected increasing layer ids below 48, received [48,11,23,35,47]",
        ),
        (
            "/sliding_window",
            json!(2049),
            "unsupported dflash.sliding_window: expected 1..=2048 (ring verify window must fit 2048 slots), received 2049",
        ),
    ] {
        let mut draft = document()["dflash"].clone();
        *draft.pointer_mut(pointer).unwrap() = value;
        assert_eq!(match_dflash(&draft).unwrap_err(), expected, "{pointer}");
    }
}

#[test]
fn numeric_spelling_is_intentionally_equal_for_compile_and_encoder_pins() {
    let mut config = document();
    config["intermediate_size"] = json!(16384.0);
    config["vision_config"]["patch_size"] = json!(16.0);
    assert_eq!(match_main(&config).unwrap().id, V26_FLASH.id);
    let mut draft = document()["dflash"].clone();
    draft["hidden_size"] = json!(4096.0);
    assert_eq!(match_dflash(&draft).unwrap().id, V26_FLASH.id);
}

#[test]
fn components_do_not_inherit_missing_fields_and_must_share_geometry() {
    let mut config = document();
    config.as_object_mut().unwrap().remove("dflash");
    config
        .as_object_mut()
        .unwrap()
        .remove("audio_tokenizer_config");
    assert_eq!(match_main(&config).unwrap().id, V26_FLASH.id);
    let mut draft = document()["dflash"].clone();
    draft.as_object_mut().unwrap().remove("hidden_size");
    let error = match_dflash(&draft).unwrap_err();
    assert!(
        error.starts_with(
            "unsupported geometry: v26-flash: dflash.hidden_size expected 4096, received absent;"
        ),
        "{error}"
    );
    let ids = model_mimo26_contract::geometry::GEOMETRIES
        .iter()
        .map(|g| g.id)
        .collect::<Vec<_>>();
    assert!(
        error.ends_with(&format!("supported ids: {}", ids.join(", "))),
        "{error}"
    );
    for id in ids.iter().skip(1) {
        assert!(error.contains(&format!("; {id}: ")), "{error}");
    }
    let mut other = V26_FLASH;
    other.id = "other";
    assert_eq!(
        model_mimo26_contract::geometry::require_same(&V26_FLASH, &other).unwrap_err(),
        "unsupported drafter geometry: expected v26-flash, received other"
    );
}

#[test]
fn engine_domain_cites_shared_engine_source_text() {
    let rules: Value = serde_json::from_str(model_mimo26_contract::geometry::SUPPORTED).unwrap();
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../..");
    for (field, domain) in rules["engine_domain"].as_object().unwrap() {
        let source = domain["source"].as_str().unwrap();
        let path = source;
        assert!(
            path.starts_with("crates/kernels-sm120/"),
            "{field}: {source}"
        );
        let excerpt = std::fs::read_to_string(root.join(path)).unwrap();
        assert!(
            excerpt.contains(domain["source_text"].as_str().unwrap()),
            "{field}: {source}: {excerpt}"
        );
    }
}

#[test]
fn recipe_classes_match_the_shared_importer_dialects() {
    let rules: Value = serde_json::from_str(model_mimo26_contract::geometry::SUPPORTED).unwrap();
    let cases = [
        (
            "fp8",
            json!({"quant_method":"fp8", "activation_scheme":"dynamic", "weight_block_size":[128,128]}),
        ),
        (
            "modelopt",
            json!({"quant_method":"modelopt", "quant_algo":"FP8"}),
        ),
    ];
    assert_eq!(
        rules["quantization_recipe_classes"],
        json!(["fp8", "modelopt"])
    );
    for (class, document) in cases {
        model_compiler_sdk::quantization::import::ExportMetadata::from_documents(
            Some(&document),
            None,
        )
        .unwrap_or_else(|e| panic!("{class}: {e}"));
    }
}

#[test]
fn enforce_selected_shared_engine_shapes() {
    let mut config = document();
    config["hidden_size"] = json!(2048);
    config["vision_config"]["out_hidden_size"] = json!(2048);
    config["audio_config"]["out_hidden_size"] = json!(2048);
    assert_eq!(
        match_main(&config).unwrap_err(),
        "unsupported hidden_size: expected [4096] (specialized fragment hidden width, fused comm FUSED_EPILOGUE width 4096 and RoutedCombine<8>), received 2048"
    );
    config = document();
    config["moe_intermediate_size"] = json!(768);
    assert_eq!(
        match_main(&config).unwrap_err(),
        "unsupported moe_intermediate_size: expected [2048] (specialized fragment intermediate width at every supported rank count), received 768"
    );
    config = document();
    config["swa_num_attention_heads"] = json!(32);
    assert_eq!(
        match_main(&config).unwrap_err(),
        "unsupported swa_num_attention_heads: expected eight Q heads per KV head (windowed decode), KV heads 8, received 32"
    );
    let mut draft = document()["dflash"].clone();
    draft["block_size"] = json!(10);
    draft["dflash_config"]["block_size"] = json!(10);
    assert_eq!(
        match_dflash(&draft).unwrap_err(),
        "unsupported dflash.block_size: expected 1..=8 (verify items hold at most 8 tokens), received 10"
    );
    draft["block_size"] = json!(3);
    draft["dflash_config"]["block_size"] = json!(3);
    assert_eq!(
        match_dflash(&draft).unwrap_err(),
        "unsupported dflash.block_size: expected block_size * GQA group divisible by 16 (verify m16 tiles), received 3"
    );
}

#[test]
fn dense_layers_must_form_a_prefix() {
    let mut config = document();
    config["moe_layer_freq"][2] = json!(0);
    assert!(
        match_main(&config)
            .unwrap_err()
            .contains("dense layers forming a prefix")
    );
}
