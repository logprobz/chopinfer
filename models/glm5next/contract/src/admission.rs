//! Admit a checkpoint's `config.json` before any weight is read: aliases,
//! semantic rules, the quantization recipe, then the closed geometry match
//! through the shared matcher, which names the first differing field, and
//! the quantization layout.
use crate::geometry::{
    ARCHITECTURE_ALIASES, COMPILE_FIELDS, GEOMETRIES, Geometry, MODEL_TYPE_ALIASES, RuntimeValues,
    TEXT_TYPE_ALIASES, VISION_TYPE_ALIASES,
};
use crate::kernels::{FP8_BLOCK, SERVED_RANKS};
use crate::quantization::Quantization;
use model_compiler_sdk::geometry::{field, match_geometry};
use serde_json::Value;

/// The compiled geometry a config selects, its runtime values and its
/// quantization layout.
#[derive(Clone, Debug)]
pub struct Matched {
    pub geometry: &'static Geometry,
    pub runtime: RuntimeValues,
    pub quantization: Quantization,
}

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

/// The geometry's semantic rules (lists must hold allowed values only) and
/// its quantization recipe class; glm5next declares no default class.
fn semantics(config: &Value, rules: &Value) -> Result<(), String> {
    let quantization = config.get("quantization_config");
    let class = model_compiler_sdk::geometry::admit(config, quantization, rules)
        .map_err(|e| e.to_string())?;
    if class == "fp8" {
        let block = quantization.and_then(|q| q.get("weight_block_size"));
        let expected = serde_json::json!([FP8_BLOCK, FP8_BLOCK]);
        if block != Some(&expected) {
            return Err(format!(
                "unsupported quantization_config.weight_block_size: expected {expected}, received {}",
                display(block)
            ));
        }
    }
    crate::quantization::export_metadata(quantization)?;
    Ok(())
}

/// Admit `config` (a parsed `config.json`) or name the field that refuses it.
pub fn admit(config: &Value) -> Result<Matched, String> {
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
    let geometry = &GEOMETRIES[index];
    let runtime = geometry.check_runtime(config)?;
    let quantization = crate::quantization::read(geometry, config)?;
    Ok(Matched {
        geometry,
        runtime,
        quantization,
    })
}

/// The largest `config.json` admitted (the pinned one is about 70 KB).
pub const CONFIG_BYTES_MAX: usize = 1 << 20;

/// The checkpoint document `name` as JSON, refused above `maximum` bytes or
/// when an object repeats a key (`model_compiler_sdk::metadata::parse`), so
/// no field is chosen by its position. The values are serde's own, which
/// admission compares with the geometry documents unchanged.
pub fn document(name: &str, bytes: &[u8], maximum: usize) -> Result<Value, String> {
    model_compiler_sdk::metadata::parse(bytes, maximum).map_err(|e| format!("{name}: {e}"))?;
    serde_json::from_slice(bytes).map_err(|e| format!("{name}: {e}"))
}

/// The parsed `config.json` bytes ([`document`]).
pub fn config(bytes: &[u8]) -> Result<Value, String> {
    document("config.json", bytes, CONFIG_BYTES_MAX)
}

/// Admit `config.json` bytes.
pub fn admit_bytes(bytes: &[u8]) -> Result<Matched, String> {
    admit(&config(bytes)?)
}

/// Refuse a rank count no kernel profile is compiled for.
pub fn check_ranks(ranks: u32) -> Result<(), String> {
    if SERVED_RANKS.contains(&ranks) {
        Ok(())
    } else {
        Err(format!(
            "unsupported tensor parallel size {ranks}: glm5next serves {SERVED_RANKS:?}"
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CONFIG: &[u8] = include_bytes!("../fixtures/config.json");

    fn config() -> Value {
        serde_json::from_slice(CONFIG).unwrap()
    }

    /// A key repeated at any depth, or a config past its bound, is refused
    /// by name rather than admitted with whichever value came last.
    #[test]
    fn a_repeated_field_or_an_oversized_config_is_refused() {
        let text = std::str::from_utf8(CONFIG).unwrap();
        for (repeated, field) in [
            (
                text.replacen('{', r#"{"architectures": ["Other"],"#, 1),
                "architectures",
            ),
            (
                text.replacen(
                    r#""text_config": {"#,
                    r#""text_config": {"hidden_size": 1,"#,
                    1,
                ),
                "hidden_size",
            ),
        ] {
            let error = admit_bytes(repeated.as_bytes()).unwrap_err();
            assert!(
                error.contains(&format!("duplicate metadata field {field}")),
                "{error}"
            );
        }
        let mut padded = CONFIG.to_vec();
        padded.resize(CONFIG_BYTES_MAX + 1, b' ');
        let error = admit_bytes(&padded).unwrap_err();
        assert!(error.contains("byte bound"), "{error}");
        admit_bytes(&padded[..CONFIG_BYTES_MAX]).unwrap();
    }

    #[test]
    fn the_pinned_config_selects_flash() {
        let matched = admit_bytes(CONFIG).unwrap();
        assert_eq!(matched.geometry.id, "flash");
        assert_eq!(matched.runtime.index_topk, 2048);
    }

    #[test]
    fn the_config_fixture_is_the_locked_checkpoint_file() {
        use sha2::{Digest, Sha256};
        let lock: Value =
            serde_json::from_str(include_str!("../../checkpoint-files.json")).unwrap();
        let digest = Sha256::digest(CONFIG)
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect::<String>();
        assert_eq!(lock["config.json"]["sha256"], digest.as_str());
        assert_eq!(lock["config.json"]["byte_length"], CONFIG.len());
    }

    #[test]
    fn a_changed_compile_time_field_is_refused_by_name() {
        let mut changed = config();
        changed["text_config"]["linear_attn_config"]["num_heads"] = 32.into();
        // A geometry added as data (the same-architecture recipe) adds its
        // own difference and id after flash's.
        let error = admit(&changed).unwrap_err();
        let ids = GEOMETRIES.iter().map(|g| g.id).collect::<Vec<_>>();
        assert!(
            error.starts_with(
                "unsupported geometry: flash: text_config.linear_attn_config.num_heads expected 64, received 32; "
            ) && error.ends_with(&format!("supported ids: {}", ids.join(", "))),
            "{error}"
        );
        let mut changed = config();
        changed["text_config"]["n_routed_experts"] = 256.into();
        let error = admit(&changed).unwrap_err();
        assert!(error.contains("text_config.n_routed_experts"), "{error}");
    }

    #[test]
    fn aliases_semantics_and_recipes_are_refused_by_name() {
        let mut changed = config();
        changed["model_type"] = "glm5".into();
        assert!(
            admit(&changed)
                .unwrap_err()
                .starts_with("unsupported model_type:")
        );
        let mut changed = config();
        changed["text_config"]["scoring_func"] = "softmax".into();
        assert_eq!(
            admit(&changed).unwrap_err(),
            "unsupported text_config.scoring_func: expected one of [\"sigmoid\"], received \"softmax\""
        );
        let mut changed = config();
        changed["quantization_config"]["weight_block_size"] = serde_json::json!([64, 64]);
        assert!(
            admit(&changed)
                .unwrap_err()
                .starts_with("unsupported quantization_config.weight_block_size:")
        );
    }

    #[test]
    fn a_recipe_without_a_quant_method_is_refused() {
        // glm5next's supported.json declares no default recipe class.
        let rules: Value = serde_json::from_str(GEOMETRIES[0].rules).unwrap();
        assert!(rules.get("quant_method_default").is_none());
        for method in [Value::Null, 5.into()] {
            let mut changed = config();
            let quantization = changed["quantization_config"].as_object_mut().unwrap();
            if method.is_null() {
                quantization.remove("quant_method");
            } else {
                quantization.insert("quant_method".into(), method.clone());
            }
            let received = if method.is_null() {
                "absent".into()
            } else {
                method.to_string()
            };
            assert_eq!(
                admit(&changed).unwrap_err(),
                format!(
                    "unsupported quantization_config.quant_method: expected one of [\"fp8\",\"modelopt\"], received {received}"
                )
            );
        }
    }

    #[test]
    fn index_topk_admits_whole_pools_only() {
        for top_k in [4, 2048] {
            let mut changed = config();
            changed["text_config"]["index_topk"] = top_k.into();
            assert_eq!(admit(&changed).unwrap().runtime.index_topk, top_k);
        }
        for top_k in [5, 2047] {
            let mut changed = config();
            changed["text_config"]["index_topk"] = top_k.into();
            assert_eq!(
                admit(&changed).unwrap_err(),
                "unsupported text_config.index_topk: must be a positive multiple of text_config.index_kpool"
            );
        }
    }

    #[test]
    fn a_config_without_an_indexer_per_layer_is_refused() {
        let mut changed = config();
        changed["text_config"]["indexer_types"] = serde_json::json!([]);
        assert!(
            admit(&changed)
                .unwrap_err()
                .starts_with("unsupported text_config.indexer_types:")
        );
    }

    #[test]
    fn only_compiled_rank_counts_are_served() {
        check_ranks(4).unwrap();
        assert_eq!(
            check_ranks(2).unwrap_err(),
            "unsupported tensor parallel size 2: glm5next serves [4]"
        );
    }
}
