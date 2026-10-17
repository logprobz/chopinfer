//! Admit a checkpoint's `config.json` before any weight is read: aliases,
//! semantic rules and the quantization recipe, the cross-field relations,
//! then the closed geometry match through the shared matcher, which names
//! the first differing field. Every refusal names the field, the expected
//! rule and the received value.
use crate::geometry::{
    ARCHITECTURE_ALIASES, COMPILE_FIELDS, GEOMETRIES, Geometry, MODEL_TYPE_ALIASES,
    TEXT_TYPE_ALIASES, VISION_TYPE_ALIASES,
};
use model_compiler_sdk::geometry::{field, match_geometry};
use serde_json::{Value, json};

/// Rank counts the code supports. A geometry that does not fit a rank count
/// is refused by its capacity, not by this list.
pub const SUPPORTED_RANKS: &[u32] = &[1, 2, 4];

fn display(value: Option<&Value>) -> String {
    value.map_or_else(|| "absent".into(), Value::to_string)
}

fn alias(config: &Value, path: &str, allowed: &[&str]) -> Result<(), String> {
    let value = field(config, path);
    if value
        .and_then(Value::as_str)
        .is_some_and(|name| allowed.contains(&name))
    {
        Ok(())
    } else {
        Err(format!(
            "unsupported {path}: expected one of {allowed:?}, received {}",
            display(value)
        ))
    }
}

fn aliases(config: &Value) -> Result<(), String> {
    let architectures = config.get("architectures");
    let listed = architectures.and_then(Value::as_array).is_some_and(|list| {
        !list.is_empty()
            && list.iter().all(|name| {
                name.as_str()
                    .is_some_and(|name| ARCHITECTURE_ALIASES.contains(&name))
            })
    });
    if !listed {
        return Err(format!(
            "unsupported architectures: expected some of {ARCHITECTURE_ALIASES:?}, received {}",
            display(architectures)
        ));
    }
    alias(config, "model_type", MODEL_TYPE_ALIASES)?;
    alias(config, "text_config.model_type", TEXT_TYPE_ALIASES)?;
    alias(config, "vision_config.model_type", VISION_TYPE_ALIASES)
}

/// The quantization recipe the checkpoint was trained and released in: FP8
/// E4M3 weights with UE8M0 scales per 32x32 block, dynamic activations, and
/// MXFP4 routed experts.
fn recipe() -> [(&'static str, Value); 4] {
    [
        ("activation_scheme", json!("dynamic")),
        ("weight_block_size", json!([32, 32])),
        ("scale_fmt", json!("ue8m0")),
        ("expert_dtype", json!("fp4")),
    ]
}

/// The geometry's semantic rules and recipe class, then each field of the
/// recipe.
fn semantics(config: &Value, rules: &Value) -> Result<(), String> {
    let quantization = config.get("quantization_config");
    model_compiler_sdk::geometry::admit(config, quantization, rules).map_err(|e| e.to_string())?;
    for (key, expected) in recipe() {
        let received = quantization.and_then(|q| q.get(key));
        if received != Some(&expected) {
            return Err(format!(
                "unsupported quantization_config.{key}: expected {expected}, received {}",
                display(received)
            ));
        }
    }
    Ok(())
}

/// The compiled geometry `config` (a parsed `config.json`) selects, or the
/// field that refuses it.
pub fn admit(config: &Value) -> Result<&'static Geometry, String> {
    aliases(config)?;
    let rules = GEOMETRIES
        .iter()
        .map(|geometry| serde_json::from_str::<Value>(geometry.rules).map_err(|e| e.to_string()))
        .collect::<Result<Vec<_>, _>>()?;
    // Report a semantic refusal before the closed shape comparison when no
    // geometry's rules accept the config.
    let checked = rules
        .iter()
        .map(|rules| semantics(config, rules))
        .collect::<Vec<_>>();
    if let Some(Err(error)) = checked.first()
        && checked.iter().all(Result::is_err)
    {
        return Err(error.clone());
    }
    crate::structure::check(config)?;
    let documents = GEOMETRIES
        .iter()
        .map(|geometry| {
            serde_json::from_str(geometry.document)
                .map(|document| (geometry.id, document))
                .map_err(|e| e.to_string())
        })
        .collect::<Result<Vec<(&str, Value)>, _>>()?;
    let index = match_geometry(config, &documents, COMPILE_FIELDS)?;
    checked[index].clone()?;
    Ok(&GEOMETRIES[index])
}

/// Whether `geometry` divides evenly over `ranks`: the vocabulary-split
/// embedding and head, the attention heads and output groups, the routed
/// experts' intermediate width in whole MXFP4 scale groups, and the
/// shared expert's. Capacity is checked separately.
pub fn check_ranks(geometry: &Geometry, ranks: u32) -> Result<(), String> {
    if !SUPPORTED_RANKS.contains(&ranks) {
        return Err(format!(
            "unsupported tensor-parallel size: expected one of {SUPPORTED_RANKS:?}, received {ranks}"
        ));
    }
    let s = &geometry.shape;
    let divided = [
        ("text_config.vocab_size", s.vocab, 1),
        ("text_config.num_attention_heads", s.attention.heads, 1),
        ("text_config.o_groups", s.attention.o_groups, 1),
        ("text_config.moe_intermediate_size", s.moe.intermediate, 32),
    ];
    for (path, value, unit) in divided {
        if !value.is_multiple_of(ranks * unit) {
            return Err(format!(
                "unsupported {path} at {ranks} ranks: expected a multiple of {}, received {value}",
                ranks * unit
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flash_config() -> Value {
        // The checkpoint's config.json is its geometry plus the fields the
        // geometry leaves out.
        let mut config: Value = serde_json::from_str(crate::geometry::FLASH.document).unwrap();
        config["architectures"] = json!(["DeepseekV41ForCausalLM"]);
        config["model_type"] = json!("deepseek_v41");
        config["transformers_version"] = json!("5.6.0");
        config["text_config"]["model_type"] = json!("deepseek_v41_text");
        config["text_config"]["initializer_range"] = json!(0.02);
        config["text_config"]["use_cache"] = json!(true);
        config["text_config"]["attention_dropout"] = json!(0.0);
        config["vision_config"]["model_type"] = json!("deepseek_v41_vision");
        config["quantization_config"] = json!({
            "quant_method": "fp8",
            "activation_scheme": "dynamic",
            "weight_block_size": [32, 32],
            "scale_fmt": "ue8m0",
            "expert_dtype": "fp4"
        });
        config
    }

    #[test]
    fn the_flash_config_selects_the_flash_geometry() {
        assert_eq!(admit(&flash_config()).unwrap().id, "flash");
    }

    #[test]
    fn the_checkpoint_config_selects_the_flash_geometry() {
        // CHOPINFER_DEEPSEEK_CHECKPOINT names the official checkpoint.
        let Some(root) = std::env::var_os("CHOPINFER_DEEPSEEK_CHECKPOINT") else {
            eprintln!("CHOPINFER_DEEPSEEK_CHECKPOINT unset: checkpoint config not read");
            return;
        };
        let text =
            std::fs::read_to_string(std::path::Path::new(&root).join("config.json")).unwrap();
        let config: Value = serde_json::from_str(&text).unwrap();
        assert_eq!(config, flash_config());
        assert_eq!(admit(&config).unwrap().id, "flash");
    }

    #[test]
    fn unsupported_configs_are_refused_by_field() {
        let cases: &[(&str, Value, &str)] = &[
            (
                "architectures",
                json!(["DeepseekV4ForCausalLM"]),
                "unsupported architectures: expected some of [\"DeepseekV41ForCausalLM\"], received [\"DeepseekV4ForCausalLM\"]",
            ),
            (
                "text_config.model_type",
                json!("deepseek_v4_text"),
                "unsupported text_config.model_type: expected one of [\"deepseek_v41_text\"], received \"deepseek_v4_text\"",
            ),
            (
                "text_config.scoring_func",
                json!("sigmoid"),
                "unsupported text_config.scoring_func",
            ),
            (
                "text_config.num_key_value_heads",
                json!(8),
                "unsupported text_config.num_key_value_heads",
            ),
            (
                "text_config.rope_scaling.rope_type",
                json!("default"),
                "unsupported text_config.rope_scaling.rope_type",
            ),
            (
                "quantization_config.weight_block_size",
                json!([128, 128]),
                "unsupported quantization_config.weight_block_size: expected [32,32], received [128,128]",
            ),
            (
                "quantization_config.scale_fmt",
                json!("e8m0"),
                "unsupported quantization_config.scale_fmt: expected \"ue8m0\", received \"e8m0\"",
            ),
            (
                "quantization_config.expert_dtype",
                Value::Null,
                "unsupported quantization_config.expert_dtype: expected \"fp4\", received null",
            ),
            (
                "text_config.compress_ratios",
                json!([0, 2]),
                "unsupported text_config.compress_ratios: expected 43 ratios",
            ),
            (
                "text_config.kv_source_layer_ids",
                json!([2, 8, 14]),
                "unsupported text_config.kv_source_layer_ids: expected a source at or before layer 20 compressing at its ratio 1",
            ),
            (
                "text_config.index_source_layer_ids",
                json!([2, 8, 20, 24, 28, 32, 36]),
                "unsupported text_config.index_source_layer_ids: expected every KV source layer, received 14",
            ),
            (
                "text_config.candidate_source_layer_id",
                json!(24),
                "unsupported text_config.candidate_source_layer_id: expected a KV source layer, received 24",
            ),
            (
                "text_config.engram_num_embeddings",
                json!([384006168, 384016683]),
                "unsupported text_config.engram_num_embeddings: expected [384006168, 384016682]",
            ),
            (
                "text_config.o_groups",
                json!(7),
                "unsupported text_config.o_groups: expected a divisor of 64 heads, received 7",
            ),
            (
                "text_config.dspark_target_layer_ids",
                json!([39, 40]),
                "unsupported text_config.dspark_target_layer_ids: expected increasing backbone layer ids below 40",
            ),
            (
                "text_config.hidden_size",
                json!(4096),
                "unsupported geometry: flash: text_config.hidden_size expected 5120, received 4096",
            ),
            (
                "text_config.index_topk",
                json!(1024),
                "unsupported geometry: flash: text_config.index_topk expected 512, received 1024",
            ),
        ];
        for (path, value, expected) in cases {
            let mut config = flash_config();
            let mut slot = &mut config;
            let keys = path.split('.').collect::<Vec<_>>();
            for key in &keys[..keys.len() - 1] {
                slot = &mut slot[*key];
            }
            slot[*keys.last().unwrap()] = value.clone();
            let error = admit(&config).unwrap_err();
            assert!(error.starts_with(expected), "{path}: {error}");
        }
        let mut config = flash_config();
        config["text_config"]
            .as_object_mut()
            .unwrap()
            .remove("sliding_window");
        assert!(admit(&config).unwrap_err().starts_with(
            "unsupported geometry: flash: text_config.sliding_window expected 128, received absent"
        ),);
    }

    #[test]
    fn flash_divides_over_every_supported_rank_count() {
        for ranks in SUPPORTED_RANKS {
            check_ranks(&crate::geometry::FLASH, *ranks).unwrap();
        }
        assert_eq!(
            check_ranks(&crate::geometry::FLASH, 8).unwrap_err(),
            "unsupported tensor-parallel size: expected one of [1, 2, 4], received 8"
        );
    }
}
