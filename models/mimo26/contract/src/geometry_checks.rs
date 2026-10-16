//! Semantic and engine constraints shared by generation and admission.
use serde_json::Value;

fn field<'a>(value: &'a Value, path: &str) -> &'a Value {
    path.split('.').fold(value, |value, key| &value[key])
}

fn refuse(path: &str, expected: impl std::fmt::Display, actual: &Value) -> String {
    format!("unsupported {path}: expected {expected}, received {actual}")
}

#[derive(Clone, Copy, PartialEq)]
pub enum Component {
    All,
    Main,
    Dflash,
}

pub fn validate(
    config: &Value,
    supported: &Value,
    ranks: &[u32],
    component: Component,
) -> Result<(), String> {
    let main = component != Component::Dflash;
    let draft = component != Component::Main;
    if main {
        if !supported["aliases"]["model_type"]
            .as_array()
            .unwrap()
            .contains(&config["model_type"])
        {
            return Err(refuse(
                "model_type",
                &supported["aliases"]["model_type"],
                &config["model_type"],
            ));
        }
        let architectures = &config["architectures"];
        if architectures.as_array().is_none_or(|a| {
            a.is_empty()
                || a.iter().any(|v| {
                    !supported["aliases"]["architecture"]
                        .as_array()
                        .unwrap()
                        .contains(v)
                })
        }) {
            return Err(refuse(
                "architectures",
                &supported["aliases"]["architecture"],
                architectures,
            ));
        }
        if let Some(quant) = config.get("quantization_config").filter(|q| !q.is_null())
            && quant.get("quant_method").is_some()
            && !supported["quantization_recipe_classes"]
                .as_array()
                .unwrap()
                .contains(&quant["quant_method"])
        {
            return Err(refuse(
                "quantization_config.quant_method",
                &supported["quantization_recipe_classes"],
                &quant["quant_method"],
            ));
        }
    }
    for (path, allowed) in supported["semantic_rules"].as_object().unwrap() {
        if path.starts_with("dflash.") && !draft || !path.starts_with("dflash.") && !main {
            continue;
        }
        let actual = field(config, path);
        let values = actual
            .as_array()
            .map(Vec::as_slice)
            .unwrap_or_else(|| std::slice::from_ref(actual));
        if values
            .iter()
            .any(|v| !allowed.as_array().unwrap().contains(v))
        {
            if let Some(values) = actual.as_array() {
                let index = values
                    .iter()
                    .position(|v| !allowed.as_array().unwrap().contains(v))
                    .unwrap();
                return Err(refuse(&format!("{path}[{index}]"), allowed, &values[index]));
            }
            return Err(refuse(path, allowed, actual));
        }
    }
    if main {
        let n = field(config, "num_hidden_layers").as_u64().ok_or_else(|| {
            refuse(
                "num_hidden_layers",
                "positive u32",
                &config["num_hidden_layers"],
            )
        })?;
        if n == 0 || n > u32::MAX as u64 {
            return Err(refuse(
                "num_hidden_layers",
                "positive u32",
                &config["num_hidden_layers"],
            ));
        }
        for path in ["hybrid_layer_pattern", "moe_layer_freq"] {
            if field(config, path).as_array().map(Vec::len) != Some(n as usize) {
                return Err(refuse(path, format!("{n} entries"), field(config, path)));
            }
        }
        let mut seen_moe = false;
        for kind in field(config, "moe_layer_freq").as_array().unwrap() {
            if kind.as_u64() == Some(1) {
                seen_moe = true;
            } else if seen_moe {
                return Err(refuse(
                    "moe_layer_freq",
                    "dense layers forming a prefix",
                    field(config, "moe_layer_freq"),
                ));
            }
        }
        if field(config, "head_dim") != field(config, "swa_head_dim") {
            return Err(refuse(
                "swa_head_dim",
                "global head width for shared rotary width",
                field(config, "swa_head_dim"),
            ));
        }
        for (left, right) in [
            ("sliding_window", "attention_chunk_size"),
            ("sliding_window", "sliding_window_size"),
            (
                "partial_rotary_factor",
                "rope_parameters.partial_rotary_factor",
            ),
            ("rope_theta", "rope_parameters.rope_theta"),
            ("hidden_size", "dflash.target_hidden_size"),
            ("hidden_size", "dflash.hidden_size"),
            ("num_hidden_layers", "dflash.num_target_layers"),
            ("vocab_size", "dflash.vocab_size"),
            ("hidden_size", "vision_config.out_hidden_size"),
            ("hidden_size", "audio_config.out_hidden_size"),
            ("dflash.block_size", "dflash.dflash_config.block_size"),
        ] {
            if right.starts_with("dflash.") && component != Component::All {
                continue;
            }
            if field(config, left).as_f64() != field(config, right).as_f64() {
                return Err(refuse(right, field(config, left), field(config, right)));
            }
        }
    }
    for (q, kv) in [
        ("num_attention_heads", "num_key_value_heads"),
        ("swa_num_attention_heads", "swa_num_key_value_heads"),
        ("dflash.num_attention_heads", "dflash.num_key_value_heads"),
    ] {
        if q.starts_with("dflash.") && !draft || !q.starts_with("dflash.") && !main {
            continue;
        }
        let a = field(config, q).as_u64().unwrap_or(0);
        let b = field(config, kv).as_u64().unwrap_or(0);
        if a == 0 || b == 0 || a % b != 0 {
            return Err(refuse(
                q,
                format!("positive multiple of {kv} ({b})"),
                field(config, q),
            ));
        }
        let group = a / b;
        if group > 8 && group != 16 {
            return Err(refuse(
                q,
                format!("GQA group at most 8 or exactly 16 (attention engine), KV heads {b}"),
                field(config, q),
            ));
        }
        if q.starts_with("swa_") && group != 8 {
            return Err(refuse(
                q,
                format!("eight Q heads per KV head (windowed decode), KV heads {b}"),
                field(config, q),
            ));
        }
        if q.starts_with("dflash.")
            && config["dflash"]["block_size"]
                .as_u64()
                .unwrap_or(0)
                .checked_mul(group)
                .is_none_or(|n| n % 16 != 0)
        {
            return Err(refuse(
                "dflash.block_size",
                "block_size * GQA group divisible by 16 (verify m16 tiles)",
                &config["dflash"]["block_size"],
            ));
        }
        for &ranks in ranks {
            if a % u64::from(ranks) != 0 || b % u64::from(ranks) != 0 {
                return Err(refuse(
                    if a % u64::from(ranks) != 0 { q } else { kv },
                    format!("divisible by supported TP{ranks}"),
                    field(config, if a % u64::from(ranks) != 0 { q } else { kv }),
                ));
            }
        }
    }
    if main {
        let experts = config["n_routed_experts"].as_u64().unwrap_or(0);
        let top = config["num_experts_per_tok"].as_u64().unwrap_or(0);
        if top == 0 || top > experts {
            return Err(refuse(
                "num_experts_per_tok",
                format!("1..={experts}"),
                &config["num_experts_per_tok"],
            ));
        }
    }
    if draft {
        let draft = &config["dflash"];
        if draft["block_size"].as_f64() != draft["dflash_config"]["block_size"].as_f64() {
            return Err(refuse(
                "dflash.dflash_config.block_size",
                &draft["block_size"],
                &draft["dflash_config"]["block_size"],
            ));
        }
        if draft["layer_types"].as_array().map(Vec::len)
            != draft["num_hidden_layers"].as_u64().map(|n| n as usize)
        {
            return Err(refuse(
                "dflash.layer_types",
                &draft["num_hidden_layers"],
                &draft["layer_types"],
            ));
        }
        let taps = draft["dflash_config"]["target_layer_ids"]
            .as_array()
            .ok_or_else(|| {
                refuse(
                    "dflash.dflash_config.target_layer_ids",
                    "target layers",
                    &draft["dflash_config"]["target_layer_ids"],
                )
            })?;
        let n = if component == Component::All {
            config["num_hidden_layers"].as_u64()
        } else {
            draft["num_target_layers"].as_u64()
        }
        .unwrap_or(0);
        if taps.is_empty()
            || taps.iter().any(|v| v.as_u64().is_none_or(|i| i >= n))
            || taps.windows(2).any(|w| w[0].as_u64() >= w[1].as_u64())
        {
            return Err(refuse(
                "dflash.dflash_config.target_layer_ids",
                format!("increasing layer ids below {n}"),
                &draft["dflash_config"]["target_layer_ids"],
            ));
        }
    }
    if main {
        let vision = &config["vision_config"];
        let depth = vision["depth"]
            .as_u64()
            .ok_or_else(|| refuse("vision_config.depth", "positive integer", &vision["depth"]))?;
        if vision["vit_window_attn_types"].as_array().map(Vec::len)
            != vision["depth"].as_u64().map(|n| n as usize)
        {
            return Err(refuse(
                "vision_config.vit_window_attn_types",
                format!("{depth} entries"),
                &vision["vit_window_attn_types"],
            ));
        }
        let types = vision["vit_window_attn_types"].as_array().unwrap();
        let full: Vec<_> = types
            .iter()
            .enumerate()
            .filter_map(|(i, kind)| (kind.as_i64() == Some(-1)).then_some(serde_json::json!(i)))
            .collect();
        if vision["fullatt_block_indexes"] != serde_json::json!(full) {
            return Err(refuse(
                "vision_config.fullatt_block_indexes",
                serde_json::json!(full),
                &vision["fullatt_block_indexes"],
            ));
        }
        for (key, expected) in config["processor_config"].as_object().unwrap() {
            if key.ends_with("token_id")
                && [
                    "image_token_id",
                    "video_token_id",
                    "audio_token_id",
                    "audio_start_token_id",
                    "audio_end_token_id",
                    "vision_start_token_id",
                    "vision_end_token_id",
                ]
                .contains(&key.as_str())
                && config[key] != *expected
            {
                return Err(refuse(key, expected, &config[key]));
            }
        }
    }
    for (path, domain) in supported["engine_domain"].as_object().unwrap() {
        if path.starts_with("dflash.") && !draft || !path.starts_with("dflash.") && !main {
            continue;
        }
        if domain.get("relation").is_some() {
            continue;
        }
        let derived;
        let actual = if let Some(prefix) = path.strip_suffix("rotary_dim") {
            let head = field(config, &format!("{prefix}head_dim"))
                .as_f64()
                .unwrap_or(0.0);
            let factor = field(config, &format!("{prefix}partial_rotary_factor"))
                .as_f64()
                .unwrap_or(0.0);
            derived = serde_json::json!((head * factor).floor());
            &derived
        } else {
            field(config, path)
        };
        if let Err(refusal) = model_compiler_sdk::geometry::check_engine_limit(path, domain, actual)
        {
            let expected = domain
                .get("max")
                .map_or_else(|| domain["values"].to_string(), |max| format!("1..={max}"));
            return Err(refuse(
                path,
                format!("{expected} ({})", domain["reason"].as_str().unwrap()),
                &refusal.received,
            ));
        }
    }
    if main {
        for path in [
            "num_attention_heads",
            "swa_num_attention_heads",
            "num_key_value_heads",
            "swa_num_key_value_heads",
        ] {
            if field(config, path)
                .as_u64()
                .is_none_or(|heads| heads % 4 != 0)
            {
                return Err(refuse(
                    path,
                    "divisible by 4 checkpoint QKV chunks",
                    field(config, path),
                ));
            }
        }
    }
    Ok(())
}
