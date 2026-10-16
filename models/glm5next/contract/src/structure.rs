//! Structural invariants of a geometry config that the generated shape
//! cannot express. `build.rs` refuses a geometry that breaks one, and
//! admission checks a checkpoint config with the same rules.
use serde_json::Value;

fn get<'a>(config: &'a Value, path: &str) -> Result<&'a Value, String> {
    path.split('.')
        .try_fold(config, |value, key| value.get(key))
        .ok_or_else(|| format!("unsupported {path}: absent"))
}

fn number(config: &Value, path: &str) -> Result<u64, String> {
    let value = get(config, path)?;
    value.as_u64().ok_or_else(|| {
        format!("unsupported {path}: expected an unsigned integer, received {value}")
    })
}

fn flag(config: &Value, path: &str) -> Result<bool, String> {
    let value = get(config, path)?;
    value
        .as_bool()
        .ok_or_else(|| format!("unsupported {path}: expected a boolean, received {value}"))
}

/// A per-layer list of `layers` entries.
fn layer_list<'a>(config: &'a Value, path: &str, layers: u64) -> Result<Vec<&'a str>, String> {
    let value = get(config, path)?;
    let list = value
        .as_array()
        .and_then(|list| list.iter().map(Value::as_str).collect::<Option<Vec<_>>>())
        .ok_or_else(|| {
            format!("unsupported {path}: expected a list of strings, received {value}")
        })?;
    if list.len() as u64 != layers {
        return Err(format!(
            "unsupported {path}: expected {layers} entries (text_config.num_hidden_layers), received {}",
            list.len()
        ));
    }
    Ok(list)
}

fn indices(config: &Value, path: &str) -> Result<Vec<u64>, String> {
    let value = get(config, path)?;
    value
        .as_array()
        .and_then(|list| list.iter().map(Value::as_u64).collect())
        .ok_or_else(|| {
            format!("unsupported {path}: expected a list of layer indices, received {value}")
        })
}

fn require(condition: bool, path: &str, rule: &str) -> Result<(), String> {
    if condition {
        Ok(())
    } else {
        Err(format!("unsupported {path}: {rule}"))
    }
}

pub fn check(config: &Value) -> Result<(), String> {
    let layers = number(config, "text_config.num_hidden_layers")?;
    let attention = layer_list(config, "text_config.layer_types", layers)?;
    let mlp = layer_list(config, "text_config.mlp_layer_types", layers)?;
    let indexers = layer_list(config, "text_config.indexer_types", layers)?;
    require(
        indexers.iter().all(|kind| *kind == "full"),
        "text_config.indexer_types",
        "every layer runs its own indexer (\"full\")",
    )?;
    let of_kind = |kind: &str| -> Vec<u64> {
        (0..layers)
            .filter(|&i| attention[i as usize] == kind)
            .collect()
    };
    require(
        indices(config, "text_config.linear_attn_config.kda_layers")?
            == of_kind("linear_attention"),
        "text_config.linear_attn_config.kda_layers",
        "must list the linear_attention layers of text_config.layer_types",
    )?;
    require(
        indices(config, "text_config.linear_attn_config.full_attn_layers")?
            == of_kind("deepseek_sparse_attention"),
        "text_config.linear_attn_config.full_attn_layers",
        "must list the sparse attention layers of text_config.layer_types",
    )?;
    let dense = number(config, "text_config.first_k_dense_replace")?;
    require(
        mlp.iter().enumerate().all(|(i, kind)| {
            *kind
                == if (i as u64) < dense {
                    "dense"
                } else {
                    "sparse"
                }
        }),
        "text_config.mlp_layer_types",
        "must be first_k_dense_replace dense layers, then MoE",
    )?;
    let pool = number(config, "text_config.index_kpool")?;
    require(
        (2..=16).contains(&pool) && pool.is_power_of_two(),
        "text_config.index_kpool",
        "indexer pools hold a power of two from 2 to 16 tokens",
    )?;
    let top_k = number(config, "text_config.index_topk")?;
    require(
        top_k > 0 && top_k.is_multiple_of(pool),
        "text_config.index_topk",
        "must be a positive multiple of text_config.index_kpool",
    )?;
    let nope = number(config, "text_config.qk_nope_head_dim")?;
    let rope = number(config, "text_config.qk_rope_head_dim")?;
    require(
        nope.checked_add(rope) == Some(number(config, "text_config.qk_head_dim")?),
        "text_config.qk_head_dim",
        "must equal qk_nope_head_dim + qk_rope_head_dim",
    )?;
    require(
        number(config, "text_config.num_key_value_heads")?
            == number(config, "text_config.num_attention_heads")?,
        "text_config.num_key_value_heads",
        "latent attention expands one key and value per query head",
    )?;
    require(
        !flag(config, "text_config.tie_word_embeddings")? && !flag(config, "tie_word_embeddings")?,
        "tie_word_embeddings",
        "the head is untied",
    )?;
    require(
        number(config, "text_config.num_nextn_predict_layers")? > 0,
        "text_config.num_nextn_predict_layers",
        "at least one MTP layer",
    )?;
    let heads = number(config, "vision_config.num_heads")?;
    require(
        heads > 0 && number(config, "vision_config.hidden_size")?.is_multiple_of(heads),
        "vision_config.num_heads",
        "must divide vision_config.hidden_size",
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> Value {
        serde_json::from_str(crate::geometry::FLASH.document).unwrap()
    }

    #[test]
    fn the_flash_geometry_is_well_formed() {
        check(&config()).unwrap();
    }

    #[test]
    fn indexer_types_must_cover_every_layer_with_a_full_indexer() {
        let mut empty = config();
        empty["text_config"]["indexer_types"] = serde_json::json!([]);
        assert_eq!(
            check(&empty).unwrap_err(),
            "unsupported text_config.indexer_types: expected 45 entries (text_config.num_hidden_layers), received 0"
        );
        let mut shared = config();
        shared["text_config"]["indexer_types"][7] = "shared".into();
        assert!(
            check(&shared)
                .unwrap_err()
                .starts_with("unsupported text_config.indexer_types:")
        );
    }

    #[test]
    fn overflowing_head_dimensions_are_refused_not_panicked() {
        let mut changed = config();
        changed["text_config"]["qk_nope_head_dim"] = u64::MAX.into();
        changed["text_config"]["qk_rope_head_dim"] = 1.into();
        assert_eq!(
            check(&changed).unwrap_err(),
            "unsupported text_config.qk_head_dim: must equal qk_nope_head_dim + qk_rope_head_dim"
        );
        assert!(
            crate::admission::admit(&changed)
                .unwrap_err()
                .starts_with("unsupported ")
        );
    }

    #[test]
    fn index_topk_must_hold_whole_pools() {
        for top_k in [5, 2047] {
            let mut changed = config();
            changed["text_config"]["index_topk"] = top_k.into();
            assert_eq!(
                check(&changed).unwrap_err(),
                "unsupported text_config.index_topk: must be a positive multiple of text_config.index_kpool"
            );
        }
        let mut pool = config();
        pool["text_config"]["index_kpool"] = 3.into();
        assert!(
            check(&pool)
                .unwrap_err()
                .starts_with("unsupported text_config.index_kpool:")
        );
    }
}
