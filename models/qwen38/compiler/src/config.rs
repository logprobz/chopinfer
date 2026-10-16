//! Family architecture requirements, independent of checkpoint precision.
use crate::Result;
use model_qwen38_contract::backend::engine_refusal;
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct MtpConfig {
    pub hybrid: bool,
    pub layer_types: Vec<String>,
    pub mtp_use_hidden_state_from_layer: Option<usize>,
    pub num_hidden_layers: usize,
    pub rope_theta: f64,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct RopeParameters {
    pub mrope_interleaved: bool,
    pub mrope_section: Vec<usize>,
    pub partial_rotary_factor: f64,
    pub rope_theta: f64,
    pub rope_type: String,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct TextConfig {
    pub attention_bias: bool,
    pub attention_dropout: f64,
    pub bos_token_id: usize,
    pub dtype: String,
    pub eos_token_id: usize,
    pub full_attention_interval: usize,
    pub hc_count: usize,
    pub hc_lowrank: usize,
    pub head_dim: usize,
    pub heads_per_ngram: usize,
    pub hidden_act: String,
    pub hidden_size: usize,
    pub indexer_budget: usize,
    pub indexer_compress_ratio: usize,
    pub indexer_head_dim: usize,
    pub indexer_kv_heads: usize,
    pub indexer_n_heads: usize,
    pub initializer_range: f64,
    pub layer_types: Vec<String>,
    pub linear_conv_kernel_dim: usize,
    pub linear_key_head_dim: usize,
    pub linear_num_key_heads: usize,
    pub linear_num_value_heads: usize,
    pub linear_value_head_dim: usize,
    pub make_ngram_vocab_size_divisible_by: usize,
    pub mamba_ssm_dtype: String,
    pub max_position_embeddings: usize,
    pub model_type: String,
    pub moe_intermediate_size: usize,
    pub mtp: MtpConfig,
    pub mtp_num_hidden_layers: usize,
    pub mtp_use_dedicated_embeddings: bool,
    pub ngram_size: usize,
    pub ngram_vocab_size_base: usize,
    pub num_attention_heads: usize,
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
    pub num_hidden_layers: usize,
    pub num_key_value_heads: usize,
    pub output_gate_type: String,
    pub output_router_logits: bool,
    pub pad_token_id: Option<usize>,
    pub partial_rotary_factor: f64,
    pub ple_conv_kernel_size: usize,
    pub ple_embed_dim: usize,
    pub ple_layer_ids: Vec<usize>,
    pub rms_norm_eps: f64,
    pub rope_parameters: RopeParameters,
    pub router_aux_loss_coef: f64,
    pub shared_expert_intermediate_size: usize,
    pub split_ngram_parts: usize,
    pub tie_word_embeddings: bool,
    pub use_cache: bool,
    pub vocab_size: usize,
    #[serde(default = "default_true")]
    pub norm_topk_prob: bool,
    #[serde(default = "default_seed")]
    pub seed: usize,
    /// This family names its table encoding outside quantization_config. The
    /// role mapper passes it to the shared importer; it is not architecture.
    #[serde(default)]
    pub ple_embedding_dtype: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct VisionConfig {
    pub deepstack_visual_indexes: Vec<usize>,
    pub depth: usize,
    pub hidden_act: String,
    pub hidden_size: usize,
    pub in_channels: usize,
    pub initializer_range: f64,
    pub intermediate_size: usize,
    pub model_type: String,
    pub num_heads: usize,
    pub num_position_embeddings: usize,
    pub out_hidden_size: usize,
    pub patch_size: usize,
    pub spatial_merge_size: usize,
    pub temporal_patch_size: usize,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Qwen38Config {
    pub architectures: Vec<String>,
    #[serde(default, rename = "_name_or_path")]
    pub name_or_path: Option<String>,
    #[serde(default)]
    pub finetuning_task: Option<String>,
    pub image_token_id: usize,
    pub language_model_only: bool,
    pub model_type: String,
    pub text_config: TextConfig,
    pub tie_word_embeddings: bool,
    pub transformers_version: String,
    pub video_token_id: usize,
    pub vision_config: VisionConfig,
    pub vision_end_token_id: usize,
    pub vision_start_token_id: usize,
    pub quantization_config: serde_json::Value,
}

pub use model_qwen38_contract::geometry::{
    ARCHITECTURE_ALIASES, MODEL_TYPE_ALIASES, TEXT_TYPE_ALIASES, VISION_TYPE_ALIASES,
};
use model_qwen38_contract::geometry::{COMPILE_FIELDS, GEOMETRIES, Geometry};

pub use model_qwen38_contract::geometry::RuntimeValues;

#[derive(Clone, Copy, Debug)]
pub struct Matched {
    pub geometry: &'static Geometry,
    pub runtime: RuntimeValues,
}

#[cfg(test)]
fn engine_domain(config: &Qwen38Config, value: &serde_json::Value) -> Result<()> {
    let rules: serde_json::Value = serde_json::from_str(GEOMETRIES[0].rules).unwrap();
    engine_domain_for(config, value, &rules, None)
}

/// Admit the checkpoint's sizes to the selected backend's engines, which
/// check the base rules' engine domain (supported.json's `engine_domain`,
/// the shared engines' table limits) at each of their phases.
fn engine_domain_for(
    config: &Qwen38Config,
    value: &serde_json::Value,
    rules: &serde_json::Value,
    geometry: Option<&'static Geometry>,
) -> Result<()> {
    use model_compiler_sdk::geometry::{ENGINE_DOMAIN, check_engine_phase};
    use model_qwen38_contract::backend::{Kernels, Selected};
    let mut limits = |phase: &str| {
        check_engine_phase(
            &rules[ENGINE_DOMAIN],
            &Selected::ENGINE_PHASES,
            phase,
            value,
        )
        .map_err(|refusal| engine_refusal(refusal.path, &refusal.restriction(), refusal.received))
    };
    Selected::admit_engine(&engine_config(config), geometry, &mut limits).map_err(Into::into)
}

/// The sizes an engine admission reads.
fn engine_config(config: &Qwen38Config) -> model_qwen38_contract::backend::EngineConfig {
    use model_qwen38_contract::backend::{EngineConfig, EngineText, EngineVision};
    let (vision, text) = (&config.vision_config, &config.text_config);
    EngineConfig {
        vision: EngineVision {
            num_position_embeddings: vision.num_position_embeddings,
            num_heads: vision.num_heads,
            hidden_size: vision.hidden_size,
        },
        text: EngineText {
            hc_count: text.hc_count,
            head_dim: text.head_dim,
            heads_per_ngram: text.heads_per_ngram,
            hidden_size: text.hidden_size,
            indexer_budget: text.indexer_budget,
            indexer_compress_ratio: text.indexer_compress_ratio,
            linear_num_key_heads: text.linear_num_key_heads,
            linear_num_value_heads: text.linear_num_value_heads,
            linear_value_head_dim: text.linear_value_head_dim,
            max_position_embeddings: text.max_position_embeddings,
            ngram_size: text.ngram_size,
            num_attention_heads: text.num_attention_heads,
            num_experts: text.num_experts,
            num_key_value_heads: text.num_key_value_heads,
            ple_embed_dim: text.ple_embed_dim,
            shared_expert_intermediate_size: text.shared_expert_intermediate_size,
            rms_norm_eps: text.rms_norm_eps,
            rope_partial_rotary_factor: text.rope_parameters.partial_rotary_factor,
            rope_theta: text.rope_parameters.rope_theta,
            mrope_section: text.rope_parameters.mrope_section.clone(),
            mtp_rope_theta: text.mtp.rope_theta,
        },
    }
}

fn structural_domain(config: &Qwen38Config) -> Result<()> {
    let text = &config.text_config;
    let refusal = |path: &str, rule: &str| -> Box<dyn std::error::Error + Send + Sync> {
        format!("unsupported structure {path}: {rule}").into()
    };
    if text.layer_types.len() != text.num_hidden_layers {
        return Err(refusal(
            "text_config.layer_types",
            "length must equal num_hidden_layers",
        ));
    }
    if text.full_attention_interval == 0
        || text.layer_types.iter().enumerate().any(|(layer, kind)| {
            (kind == "full_attention") != ((layer + 1) % text.full_attention_interval == 0)
        })
    {
        return Err(refusal(
            "text_config.full_attention_interval",
            "must agree with layer_types",
        ));
    }
    if text.mtp.num_hidden_layers != text.mtp_num_hidden_layers
        || text.mtp.num_hidden_layers != text.mtp.layer_types.len()
        || text.mtp.num_hidden_layers > 1
    {
        return Err(refusal(
            "text_config.mtp.num_hidden_layers",
            "must equal mtp_num_hidden_layers and layer_types length, at most 1",
        ));
    }
    if text.partial_rotary_factor != text.rope_parameters.partial_rotary_factor {
        return Err(refusal(
            "text_config.partial_rotary_factor",
            "must equal rope_parameters.partial_rotary_factor",
        ));
    }
    if text.mtp.rope_theta != text.rope_parameters.rope_theta {
        return Err(refusal(
            "text_config.mtp.rope_theta",
            "must equal rope_parameters.rope_theta",
        ));
    }
    let sections = &text.rope_parameters.mrope_section;
    let sum = sections
        .iter()
        .try_fold(0usize, |sum, section| sum.checked_add(*section));
    let rotary = text.head_dim as f64 * text.rope_parameters.partial_rotary_factor;
    if sections.len() != 3 || sum.is_none_or(|sum| sum as f64 * 2.0 != rotary) {
        return Err(refusal(
            "text_config.rope_parameters.mrope_section",
            "three sections must sum to rotary dim / 2",
        ));
    }
    if text.ple_layer_ids.is_empty() {
        return Err(refusal(
            "text_config.ple_layer_ids",
            "served geometry requires one PLE layer",
        ));
    }
    if text
        .ple_layer_ids
        .iter()
        .any(|&layer| layer == 0 || layer > text.num_hidden_layers)
    {
        return Err(refusal(
            "text_config.ple_layer_ids",
            "every 1-based layer must be in 1..=num_hidden_layers",
        ));
    }
    Ok(())
}

fn semantic_domain(
    config: &Qwen38Config,
    value: &serde_json::Value,
    supported: &serde_json::Value,
) -> Result<()> {
    use model_compiler_sdk::geometry::{Refusal, admit};
    let quant = &config.quantization_config;
    // A ModelOpt export nests its recipe under `quantization`; supported.json
    // declares the class of a recipe without `quant_method`.
    let recipe = quant.get("quantization").unwrap_or(quant);
    admit(value, Some(recipe), supported).map_err(|refusal| match refusal {
        Refusal::Semantic {
            path,
            received: None,
            ..
        } => format!("unsupported semantic {path}: absent"),
        Refusal::Semantic {
            path,
            allowed,
            received: Some(actual),
        } => format!(
            "unsupported semantic {path}: expected one of {}, received {actual}",
            serde_json::Value::Array(allowed.to_vec())
        ),
        Refusal::RecipeClass { method, class, .. } => format!(
            "unsupported semantic quantization_config: recipe class {} is not supported",
            class.map_or_else(
                || method.map_or_else(|| "absent".into(), serde_json::Value::to_string),
                str::to_owned
            )
        ),
    })?;
    model_compiler_sdk::quantization::import::ExportMetadata::from_documents(Some(quant), None)
        .map_err(|error| format!("unsupported semantic quantization_config: {error}"))?;
    Ok(())
}

fn default_seed() -> usize {
    1234
}

fn default_true() -> bool {
    true
}

/// Exporters serialize additional HF output defaults and training metadata.
/// Normalize only these named fields before the closed execution schema is
/// decoded. A non-default execution option or unknown field still fails.
fn normalize(value: &mut serde_json::Value) -> Result<()> {
    use serde_json::json;
    for section in ["", "text_config", "vision_config"] {
        let object = if section.is_empty() {
            value.as_object_mut()
        } else {
            value[section].as_object_mut()
        }
        .ok_or_else(|| format!("config {section} must be an object"))?;
        for (key, expected) in [
            ("chunk_size_feed_forward", json!(0)),
            ("is_encoder_decoder", json!(false)),
            ("output_attentions", json!(false)),
            ("output_hidden_states", json!(false)),
            ("return_dict", json!(true)),
        ] {
            if let Some(actual) = object.remove(key)
                && actual != expected
            {
                return Err(format!(
                    "unsupported config {section}.{key}: expected {expected}, received {actual}"
                )
                .into());
            }
        }
        // Classification labels and provenance have no use in this family's
        // conditional-generation mathematics or tokenizer contract.
        for key in ["id2label", "label2id", "problem_type"] {
            object.remove(key);
        }
        if !section.is_empty() {
            object.remove("_name_or_path");
        }
        if section != "text_config"
            && let Some(dtype) = object.remove("dtype")
            && dtype != "bfloat16"
        {
            return Err(format!("unsupported config {section}.dtype: {dtype}").into());
        }
    }
    if let Some(multimodal) = value.as_object_mut().unwrap().remove("is_multimodal")
        && multimodal
            != !value["language_model_only"]
                .as_bool()
                .ok_or("invalid language_model_only")?
    {
        return Err("config is_multimodal conflicts with language_model_only".into());
    }
    let text = value["text_config"].as_object_mut().unwrap();
    text.entry("full_attention_interval").or_insert(json!(4));
    let rotary = text
        .get("rope_parameters")
        .map(|v| v["partial_rotary_factor"].clone())
        .unwrap_or(serde_json::Value::Null);
    text.entry("partial_rotary_factor").or_insert(rotary);
    if let Some(states) = text.remove("number_of_conv_states")
        && (states.as_u64().is_none()
            || states.as_u64()
                != text
                    .get("linear_conv_kernel_dim")
                    .and_then(|k| k.as_u64())
                    .and_then(|k| k.checked_sub(1)))
    {
        return Err(
            "unsupported text_config.number_of_conv_states for the convolution state".into(),
        );
    }
    // Every layer in this architecture uses moe_intermediate_size. The HF
    // dense-MLP default does not describe an executed projection.
    if let Some(width) = text.remove("intermediate_size")
        && width.as_u64().is_none_or(|w| w == 0)
    {
        return Err("invalid text_config.intermediate_size metadata".into());
    }
    if let Some(architectures) = value["vision_config"]
        .as_object_mut()
        .unwrap()
        .remove("architectures")
        && !architectures.is_null()
    {
        return Err("unsupported vision_config.architectures override".into());
    }
    Ok(())
}

impl Qwen38Config {
    pub fn load(root: &Path) -> Result<Self> {
        Self::from_value(model_compiler_sdk::metadata::read(
            root,
            "config.json",
            1024 * 1024,
        )?)
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        Self::from_value(model_compiler_sdk::metadata::parse(bytes, 1024 * 1024)?)
    }

    fn from_value(mut value: serde_json::Value) -> Result<Self> {
        normalize(&mut value)?;
        let config: Self = serde_json::from_value(value)?;
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<()> {
        self.geometry().map(|_| ())
    }

    pub fn geometry(&self) -> Result<Matched> {
        if self.architectures.is_empty()
            || self.architectures.len() > ARCHITECTURE_ALIASES.len()
            || self
                .architectures
                .iter()
                .any(|name| !ARCHITECTURE_ALIASES.contains(&name.as_str()))
        {
            return Err("unsupported architectures for the Qwen family".into());
        }
        if !MODEL_TYPE_ALIASES.contains(&self.model_type.as_str()) {
            return Err("unsupported model_type for the Qwen family".into());
        }
        if !TEXT_TYPE_ALIASES.contains(&self.text_config.model_type.as_str()) {
            return Err("unsupported text_config.model_type".into());
        }
        if !VISION_TYPE_ALIASES.contains(&self.vision_config.model_type.as_str()) {
            return Err("unsupported vision_config.model_type".into());
        }
        let value = serde_json::to_value(self)?;
        let rules = GEOMETRIES
            .iter()
            .map(|geometry| serde_json::from_str(geometry.rules))
            .collect::<std::result::Result<Vec<serde_json::Value>, _>>()?;
        // Report semantic refusals before the closed shape comparison.
        let semantics = rules
            .iter()
            .map(|rules| semantic_domain(self, &value, rules))
            .collect::<Vec<_>>();
        if semantics.iter().all(Result::is_err) {
            return Err(semantics.into_iter().next().unwrap().unwrap_err());
        }
        structural_domain(self)?;
        let documents = GEOMETRIES
            .iter()
            .map(|geometry| Ok((geometry.id, serde_json::from_str(geometry.document)?)))
            .collect::<Result<Vec<(&str, serde_json::Value)>>>()?;
        let matched =
            model_compiler_sdk::geometry::match_geometry(&value, &documents, COMPILE_FIELDS);
        // Every geometry declares the base directory's engine domain.
        engine_domain_for(
            self,
            &value,
            &rules[0],
            matched.as_ref().ok().map(|&index| &GEOMETRIES[index]),
        )?;
        let index = matched?;
        let geometry = &GEOMETRIES[index];
        semantic_domain(self, &value, &rules[index])?;
        geometry.check_runtime(&value)?;
        Ok(Matched {
            geometry,
            runtime: RuntimeValues {
                indexer_budget: self.text_config.indexer_budget,
            },
        })
    }
}

/// The engine domain against the SM120 kernels' qualification and sources.
#[cfg(test)]
#[path = "config/sm120.rs"]
mod sm120;

#[cfg(test)]
mod tests {
    use super::*;
    fn assert_refusal(error: String, expected: &str, path: &str) {
        if expected.starts_with("unsupported geometry:") && GEOMETRIES.len() > 1 {
            let flash = expected.strip_suffix("; supported ids: flash").unwrap();
            assert!(error.starts_with(&format!("{flash};")), "{path}: {error}");
            for geometry in &GEOMETRIES[1..] {
                assert!(
                    error.contains(&format!("; {}: ", geometry.id)),
                    "{path}: {error}"
                );
            }
            let ids = GEOMETRIES
                .iter()
                .map(|g| g.id)
                .collect::<Vec<_>>()
                .join(", ");
            assert!(
                error.ends_with(&format!("; supported ids: {ids}")),
                "{path}: {error}"
            );
        } else {
            assert_eq!(error, expected, "{path}");
        }
    }

    #[test]
    fn qsa_verify_rows_fit_the_kernel_ring() {
        let mut config =
            Qwen38Config::from_bytes(include_bytes!("../fixtures/config.json")).unwrap();
        config.text_config.indexer_compress_ratio = 16;
        let value = serde_json::to_value(&config).unwrap();
        let error = engine_domain(&config, &value).unwrap_err().to_string();
        assert!(
            error.contains("text_config.indexer_compress_ratio"),
            "{error}"
        );
        assert!(error.contains("QSA_RING"), "{error}");
        config.text_config.max_position_embeddings = 131_072;
        for block in [2, 4, 8] {
            config.text_config.indexer_compress_ratio = block;
            let result = engine_domain(&config, &serde_json::to_value(&config).unwrap());
            if block == 4 {
                result.unwrap();
            } else {
                assert!(
                    result
                        .unwrap_err()
                        .to_string()
                        .contains("qualified QSA block is 4")
                );
            }
        }
    }

    #[test]
    fn unknown_relation_phase_returns_a_refusal() {
        // A relation needs no reason; refusing its unknown phase must not
        // assume one (review W2-GEO3 S1).
        let config: Qwen38Config =
            serde_json::from_slice(include_bytes!("../fixtures/config.json")).unwrap();
        let value = serde_json::to_value(&config).unwrap();
        let mut rules: serde_json::Value = serde_json::from_str(GEOMETRIES[0].rules).unwrap();
        rules["engine_domain"]["text_config.hc_count"] =
            serde_json::json!({"relation": "x > y", "phase": "after_row_shape"});
        model_compiler_sdk::geometry_build::engine_domain(
            &rules["engine_domain"],
            Path::new("supported.json"),
        );
        assert_eq!(
            engine_domain_for(&config, &value, &rules, None)
                .unwrap_err()
                .to_string(),
            "unsupported engine shape text_config.hc_count: engine_domain phase \"after_row_shape\" is not checked, received 4"
        );
        // A limit with a reason keeps its wording.
        rules["engine_domain"]["text_config.hc_count"] =
            serde_json::json!({"values": [4], "phase": "after_row_shape", "reason": "r"});
        assert_eq!(
            engine_domain_for(&config, &value, &rules, None)
                .unwrap_err()
                .to_string(),
            "unsupported engine shape text_config.hc_count: r, received 4"
        );
    }

    #[test]
    fn every_engine_domain_entry_form_refuses_without_panicking() {
        use serde_json::{Value, json};
        let config: Qwen38Config =
            serde_json::from_slice(include_bytes!("../fixtures/config.json")).unwrap();
        let value = serde_json::to_value(&config).unwrap();
        let base: Value = serde_json::from_str(GEOMETRIES[0].rules).unwrap();
        let kinds = [
            json!({}),
            json!({"values": [4]}),
            json!({"values": [5]}),
            json!({"max": 2}),
            json!({"max_length": 1}),
            json!({"relation": "x > y"}),
            json!({"relation": 5}),
            json!({"relation": ""}),
            json!({"relation": "x > y", "values": [4]}),
            json!({"values": [4, null]}),
        ];
        let optional = |key: &str, values: &[Value]| {
            std::iter::once(None)
                .chain(values.iter().cloned().map(Some))
                .map(|value| (key.to_owned(), value))
                .collect::<Vec<_>>()
        };
        let reasons = optional("reason", &[json!("r"), json!(""), json!(5)]);
        let sources = optional("source", &[json!("a.rs"), json!(1)]);
        let texts = optional("source_text", &[json!("n"), json!(1)]);
        let phases = optional(
            "phase",
            &[
                json!(model_compiler_sdk::geometry::FIRST_ENGINE_PHASE),
                json!("after_row_widths"),
                json!("after_row_shape"),
                json!(""),
                json!(5),
                Value::Null,
            ],
        );
        let mut entries = vec![json!(5), Value::Null, json!("r"), json!([])];
        for kind in &kinds {
            for reason in &reasons {
                for source in &sources {
                    for text in &texts {
                        for phase in &phases {
                            let mut entry = kind.clone();
                            for (key, field) in [reason, source, text, phase] {
                                if let Some(field) = field {
                                    entry[key.as_str()] = field.clone();
                                }
                            }
                            entries.push(entry);
                        }
                    }
                }
            }
        }
        let mut panicked = Vec::new();
        for entry in entries {
            for listed in [entry.clone(), json!([entry])] {
                let mut rules = base.clone();
                rules["engine_domain"]["text_config.hc_count"] = listed.clone();
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    engine_domain_for(&config, &value, &rules, None).map_err(|e| e.to_string())
                }));
                if result.is_err() {
                    panicked.push(listed);
                }
            }
        }
        assert!(
            panicked.is_empty(),
            "{} panicked: {panicked:?}",
            panicked.len()
        );
    }

    #[test]
    fn vision_position_table_is_square() {
        let mut config =
            Qwen38Config::from_bytes(include_bytes!("../fixtures/config.json")).unwrap();
        config.vision_config.num_position_embeddings = 2305;
        let value = serde_json::to_value(&config).unwrap();
        let error = engine_domain(&config, &value).unwrap_err().to_string();
        assert!(
            error.contains("vision_config.num_position_embeddings"),
            "{error}"
        );
        assert!(error.contains("square"), "{error}");
    }

    #[test]
    fn a_recipe_without_a_quant_method_admits_as_the_declared_default() {
        let rules: serde_json::Value = serde_json::from_str(GEOMETRIES[0].rules).unwrap();
        assert_eq!(rules["quant_method_default"], "modelopt");
        let mut config =
            Qwen38Config::from_bytes(include_bytes!("../fixtures/config.json")).unwrap();
        // ModelOpt NVFP4 exports, flat and nested, without `quant_method`.
        let recipe = serde_json::json!({"quant_algo": "NVFP4", "group_size": 16});
        for quant in [recipe.clone(), serde_json::json!({"quantization": recipe})] {
            config.quantization_config = quant;
            let value = serde_json::to_value(&config).unwrap();
            semantic_domain(&config, &value, &rules).unwrap();
        }
        let mut fp8_only = rules;
        fp8_only["quantization_recipe_classes"] = serde_json::json!(["fp8"]);
        let value = serde_json::to_value(&config).unwrap();
        assert_eq!(
            semantic_domain(&config, &value, &fp8_only)
                .unwrap_err()
                .to_string(),
            "unsupported semantic quantization_config: recipe class modelopt is not supported"
        );
    }

    #[test]
    fn geometry_and_precise_refusals() {
        let config = Qwen38Config::from_bytes(include_bytes!("../fixtures/config.json")).unwrap();
        assert_eq!(config.geometry().unwrap().geometry.id, "flash");
        let cases = [
            (
                "text_config.max_position_embeddings",
                serde_json::json!(131072),
                "unsupported geometry: flash: text_config.max_position_embeddings expected 262144, received 131072; supported ids: flash",
            ),
            (
                "text_config.hc_lowrank",
                serde_json::json!(336),
                "unsupported engine shape text_config.hc_lowrank: hyper-connection rows require rank 320, received 336",
            ),
            (
                "text_config.layer_types",
                serde_json::json!(["unknown"]),
                "unsupported semantic text_config.layer_types: expected one of [\"linear_attention\",\"full_attention\"], received [\"unknown\"]",
            ),
            (
                "text_config.hidden_act",
                serde_json::json!("relu"),
                "unsupported semantic text_config.hidden_act: expected one of [\"silu\"], received \"relu\"",
            ),
            (
                "text_config.rope_parameters.rope_type",
                serde_json::json!("unknown"),
                "unsupported semantic text_config.rope_parameters.rope_type: expected one of [\"default\"], received \"unknown\"",
            ),
            (
                "quantization_config",
                serde_json::json!({"quant_method":"unknown"}),
                "unsupported semantic quantization_config: recipe class unknown is not supported",
            ),
            (
                "text_config.indexer_n_heads",
                serde_json::json!(8),
                "unsupported geometry: flash: text_config.indexer_n_heads expected 4, received 8; supported ids: flash",
            ),
            (
                "text_config.indexer_budget",
                serde_json::json!(2049),
                "unsupported runtime text_config.indexer_budget: expected 1..=2048, received 2049",
            ),
            (
                "text_config.indexer_budget",
                serde_json::json!(0),
                "unsupported runtime text_config.indexer_budget: expected 1..=2048, received 0",
            ),
            (
                "text_config.output_gate_type",
                serde_json::json!("relu"),
                "unsupported semantic text_config.output_gate_type: expected one of [\"sigmoid\"], received \"relu\"",
            ),
            (
                "text_config.mamba_ssm_dtype",
                serde_json::json!("float16"),
                "unsupported semantic text_config.mamba_ssm_dtype: expected one of [\"float32\"], received \"float16\"",
            ),
            (
                "vision_config.hidden_act",
                serde_json::json!("relu"),
                "unsupported semantic vision_config.hidden_act: expected one of [\"gelu_pytorch_tanh\"], received \"relu\"",
            ),
            (
                "text_config.rope_parameters.mrope_interleaved",
                serde_json::json!(false),
                "unsupported semantic text_config.rope_parameters.mrope_interleaved: expected one of [true], received false",
            ),
            (
                "text_config.mtp.layer_types",
                serde_json::json!(["linear_attention"]),
                "unsupported semantic text_config.mtp.layer_types: expected one of [\"full_attention\"], received [\"linear_attention\"]",
            ),
        ];
        for (path, replacement, expected) in cases {
            let mut value = serde_json::to_value(&config).unwrap();
            let mut target = &mut value;
            for key in path.split('.') {
                target = &mut target[key];
            }
            *target = replacement;
            let changed: Qwen38Config = serde_json::from_value(value).unwrap();
            assert_refusal(changed.geometry().unwrap_err().to_string(), expected, path);
        }
        let mut lower = config;
        lower.text_config.indexer_budget = 1024;
        assert_eq!(
            lower.geometry().unwrap_err().to_string(),
            "unsupported runtime text_config.indexer_budget: expected compiled limit 2048, received 1024"
        );
    }

    #[test]
    fn listed_geometries_pass_all_admission_checks() {
        fn overlay(target: &mut serde_json::Value, source: &serde_json::Value) {
            if let Some(object) = source.as_object() {
                for (key, value) in object {
                    overlay(&mut target[key], value);
                }
            } else {
                *target = source.clone();
            }
        }
        for geometry in GEOMETRIES {
            let mut value: serde_json::Value =
                serde_json::from_slice(include_bytes!("../fixtures/config.json")).unwrap();
            overlay(
                &mut value,
                &serde_json::from_str(geometry.document).unwrap(),
            );
            let config: Qwen38Config = serde_json::from_value(value).unwrap();
            let matched = config
                .geometry()
                .unwrap_or_else(|error| panic!("{}: {error}", geometry.id));
            assert_eq!(matched.geometry.id, geometry.id);
            assert_eq!(
                matched.runtime.indexer_budget,
                config.text_config.indexer_budget
            );
        }
    }

    #[test]
    fn real_checkpoint_selects_flash() {
        let Some(root) = std::env::var_os("CHOPINFER_QWEN38_CHECKPOINT") else {
            return;
        };
        let config = Qwen38Config::from_bytes(
            &std::fs::read(std::path::Path::new(&root).join("config.json")).unwrap(),
        )
        .unwrap();
        let matched = config.geometry().unwrap();
        assert_eq!(matched.geometry.id, "flash");
        assert_eq!(
            matched.runtime.indexer_budget,
            config.text_config.indexer_budget
        );
    }

    #[test]
    fn structural_refusals() {
        let config = Qwen38Config::from_bytes(include_bytes!("../fixtures/config.json")).unwrap();
        for (path, replacement, rule) in [
            (
                "text_config.num_hidden_layers",
                serde_json::json!(47),
                "unsupported structure text_config.layer_types: length must equal num_hidden_layers",
            ),
            (
                "text_config.full_attention_interval",
                serde_json::json!(3),
                "unsupported structure text_config.full_attention_interval: must agree with layer_types",
            ),
            (
                "text_config.mtp_num_hidden_layers",
                serde_json::json!(0),
                "unsupported structure text_config.mtp.num_hidden_layers: must equal mtp_num_hidden_layers and layer_types length, at most 1",
            ),
            (
                "text_config.partial_rotary_factor",
                serde_json::json!(0.5),
                "unsupported structure text_config.partial_rotary_factor: must equal rope_parameters.partial_rotary_factor",
            ),
            (
                "text_config.rope_parameters.mrope_section",
                serde_json::json!([1, 2, 3]),
                "unsupported structure text_config.rope_parameters.mrope_section: three sections must sum to rotary dim / 2",
            ),
            (
                "text_config.ple_layer_ids",
                serde_json::json!([49]),
                "unsupported structure text_config.ple_layer_ids: every 1-based layer must be in 1..=num_hidden_layers",
            ),
        ] {
            let mut value = serde_json::to_value(&config).unwrap();
            let mut target = &mut value;
            for key in path.split('.') {
                target = &mut target[key];
            }
            *target = replacement;
            let changed: Qwen38Config = serde_json::from_value(value).unwrap();
            assert_refusal(changed.geometry().unwrap_err().to_string(), rule, path);
        }
    }

    #[test]
    fn widened_attention_shapes_pass_the_engine_domain() {
        let mut config =
            Qwen38Config::from_bytes(include_bytes!("../fixtures/config.json")).unwrap();
        for dim in [128, 256] {
            for rotary in [32, 64] {
                for heads in [4, 8, 16] {
                    for theta in [1.0, 1e4, 5e6, 1e7] {
                        let text = &mut config.text_config;
                        text.head_dim = dim;
                        text.indexer_n_heads = heads;
                        text.partial_rotary_factor = rotary as f64 / dim as f64;
                        text.rope_parameters.partial_rotary_factor = text.partial_rotary_factor;
                        text.rope_parameters.mrope_section = if rotary == 32 {
                            vec![6, 5, 5]
                        } else {
                            vec![11, 11, 10]
                        };
                        text.rope_parameters.rope_theta = theta;
                        text.mtp.rope_theta = theta;
                        structural_domain(&config).unwrap();
                        engine_domain(&config, &serde_json::to_value(&config).unwrap()).unwrap();
                    }
                }
            }
        }
    }

    #[test]
    fn refuse_unqualified_attention_instances() {
        let base = Qwen38Config::from_bytes(include_bytes!("../fixtures/config.json")).unwrap();
        for (path, replacement, message) in [
            (
                "indexer_compress_ratio",
                serde_json::json!(2),
                "qualified QSA block is 4",
            ),
            (
                "indexer_compress_ratio",
                serde_json::json!(8),
                "qualified QSA block is 4",
            ),
            (
                "indexer_budget",
                serde_json::json!(4096),
                "qualified kept-block counts are [256, 512]",
            ),
            (
                "num_attention_heads",
                serde_json::json!(32),
                "TP4 group 8 at head dim 256 has no qualified dense prefill/verify instance",
            ),
        ] {
            let mut value = serde_json::to_value(&base).unwrap();
            value["text_config"]["max_position_embeddings"] = 131072.into();
            value["text_config"][path] = replacement;
            let config = serde_json::from_value(value.clone()).unwrap();
            let error = engine_domain(&config, &value).unwrap_err().to_string();
            assert!(error.contains(message), "{error}");
        }
        let mut config = base;
        config.text_config.rope_parameters.mrope_section = vec![16, 8, 8];
        let error = engine_domain(&config, &serde_json::to_value(&config).unwrap())
            .unwrap_err()
            .to_string();
        assert!(error.contains("qualified M-RoPE sections"), "{error}");
    }

    #[test]
    fn engine_refusals() {
        let config = Qwen38Config::from_bytes(include_bytes!("../fixtures/config.json")).unwrap();
        for (path, replacement, expected) in [
            (
                "text_config.indexer_n_heads",
                12,
                "sparse indexer requires heads in {4,8,16}",
            ),
            (
                "text_config.head_dim",
                192,
                "sparse and gated attention require head dim in {128,256}",
            ),
            (
                "text_config.indexer_head_dim",
                256,
                "sparse indexer requires head dim 128",
            ),
            (
                "text_config.linear_key_head_dim",
                256,
                "linear attention requires head dim 128: dim 256 prefill scan needs 157480 shared bytes, exceeding 101376",
            ),
            (
                "text_config.linear_value_head_dim",
                256,
                "linear attention instantiates one key/value dim and requires 128",
            ),
            (
                "text_config.indexer_kv_heads",
                2,
                "sparse indexer requires one raw key head",
            ),
            (
                "text_config.num_key_value_heads",
                0,
                "sparse attention requires an integral GQA group at most 16",
            ),
            (
                "text_config.linear_num_value_heads",
                64,
                "linear value heads must be a multiple of key heads with group at most 3",
            ),
            (
                "text_config.linear_conv_kernel_dim",
                1,
                "causal convolution requires taps in 2..=8",
            ),
            (
                "text_config.ple_conv_kernel_size",
                66,
                "causal convolution requires taps in 2..=8",
            ),
            (
                "text_config.hidden_size",
                2559,
                "must be positive and divisible by served rank counts 2 and 4",
            ),
            (
                "text_config.num_experts",
                64,
                "native expert prefill requires power-of-two local experts in 32..=256 at every served rank count",
            ),
            (
                "text_config.heads_per_ngram",
                1,
                "total ngram heads must be positive and divisible by served rank counts 2 and 4",
            ),
            (
                "text_config.hc_lowrank",
                513,
                "fused GEMM requires K in 16..=512 and divisible by 16",
            ),
            (
                "text_config.num_experts_per_tok",
                33,
                "parallel route requires top-k in 1..=32",
            ),
            (
                "text_config.indexer_compress_ratio",
                3,
                "QSA block must be at least 2 and divide the ring and page sizes",
            ),
            (
                "text_config.max_position_embeddings",
                262148,
                "QSA context must be positive, divisible by block, and at most 65536 blocks",
            ),
            (
                "text_config.mtp.rope_theta",
                0,
                "RoPE theta must be finite and positive in f32",
            ),
        ] {
            let mut value = serde_json::to_value(&config).unwrap();
            let mut target = &mut value;
            for key in path.split('.') {
                target = &mut target[key];
            }
            *target = replacement.into();
            let changed: Qwen38Config = serde_json::from_value(value.clone()).unwrap();
            assert_eq!(
                engine_domain(&changed, &value).unwrap_err().to_string(),
                engine_refusal(path, expected, replacement)
            );
        }
    }

    #[test]
    fn refuse_unserved_kv_and_ple_shapes() {
        let config = Qwen38Config::from_bytes(include_bytes!("../fixtures/config.json")).unwrap();
        let mut changed = config.clone();
        changed.text_config.num_key_value_heads = 4;
        assert_eq!(
            changed.geometry().unwrap_err().to_string(),
            "unsupported engine shape text_config.num_key_value_heads: attention layout requires one KV head per rank at every served rank count, received 4"
        );
        let mut changed = config.clone();
        changed.text_config.ple_layer_ids = vec![2, 3];
        assert_eq!(
            changed.geometry().unwrap_err().to_string(),
            "unsupported engine shape text_config.ple_layer_ids: bundle lowering supports at most one PLE layer, received 2"
        );
        let mut changed = config;
        changed.text_config.mtp.rope_theta *= 2.0;
        assert_eq!(
            changed.geometry().unwrap_err().to_string(),
            "unsupported structure text_config.mtp.rope_theta: must equal rope_parameters.rope_theta"
        );
    }

    #[test]
    fn context_must_align_to_kv_page() {
        let mut config =
            Qwen38Config::from_bytes(include_bytes!("../fixtures/config.json")).unwrap();
        config.text_config.max_position_embeddings = 131076;
        assert_eq!(
            engine_domain(&config, &serde_json::to_value(&config).unwrap())
                .unwrap_err()
                .to_string(),
            "unsupported engine shape text_config.max_position_embeddings: context must be a multiple of the 64-token KV page, received 131076"
        );
        config.text_config.max_position_embeddings = 131072;
        engine_domain(&config, &serde_json::to_value(&config).unwrap()).unwrap();
    }

    #[test]
    fn dense_fp8_prefill_refuses_spilling_groups_at_each_served_rank() {
        let mut config =
            Qwen38Config::from_bytes(include_bytes!("../fixtures/config.json")).unwrap();
        for kv in [1, 2] {
            config.text_config.num_key_value_heads = kv;
            config.text_config.num_attention_heads = 24;
            engine_domain(&config, &serde_json::to_value(&config).unwrap()).unwrap();
            for (heads, group) in [(8, 2), (16, 4)] {
                config.text_config.num_attention_heads = heads;
                let error = engine_domain(&config, &serde_json::to_value(&config).unwrap())
                    .unwrap_err()
                    .to_string();
                assert_eq!(
                    error,
                    engine_refusal(
                        "text_config.num_attention_heads",
                        "dense FP8 prefill at head dim 256 requires local GQA group in {1,3,5,6,7,8,16}: groups 2 and 4 spill and fail the zero-stack audit",
                        format!("TP4 group {group} from {heads} Q heads / {kv} KV heads")
                    )
                );
            }
            for group in 9..16 {
                config.text_config.num_attention_heads = 4 * group;
                assert_eq!(
                    engine_domain(&config, &serde_json::to_value(&config).unwrap())
                        .unwrap_err()
                        .to_string(),
                    engine_refusal(
                        "text_config.num_attention_heads",
                        "dense prefill requires padded local GQA group in {1,2,4,8,16} to fill m16 tiles; local groups 9..15 are not padded",
                        format!("TP4 group {group} padded to {group}")
                    )
                );
            }
            // TP4 group 16 fills its tiles, but TP2 group 32 still fails.
            config.text_config.num_attention_heads = 64;
            assert_eq!(
                engine_domain(&config, &serde_json::to_value(&config).unwrap())
                    .unwrap_err()
                    .to_string(),
                engine_refusal(
                    "text_config.num_key_value_heads",
                    "sparse attention requires an integral GQA group at most 16",
                    kv
                )
            );
        }
    }

    #[test]
    fn dense_head_128_refuses_group_16_fold_scratch() {
        let mut config =
            Qwen38Config::from_bytes(include_bytes!("../fixtures/config.json")).unwrap();
        config.text_config.num_key_value_heads = 2;
        config.text_config.head_dim = 128;
        config.text_config.partial_rotary_factor = 0.5;
        config.text_config.rope_parameters.partial_rotary_factor = 0.5;
        for heads in [8, 16, 24, 32, 64] {
            config.text_config.num_attention_heads = heads;
            let result = engine_domain(&config, &serde_json::to_value(&config).unwrap());
            if heads == 64 {
                assert_eq!(
                    result.unwrap_err().to_string(),
                    engine_refusal(
                        "text_config.num_attention_heads",
                        "dense paged decode at head dim 128 requires local GQA group at most 8: group 16 exceeds fold scratch",
                        "TP4 group 16 from 64 Q heads / 2 KV heads",
                    )
                );
            } else {
                result.unwrap();
            }
        }
    }

    #[test]
    fn dense_verify_refuses_odd_groups_before_catalog_build() {
        let mut config =
            Qwen38Config::from_bytes(include_bytes!("../fixtures/config.json")).unwrap();
        for dim in [128, 256] {
            config.text_config.head_dim = dim;
            config.text_config.rope_parameters.partial_rotary_factor = 64.0 / dim as f64;
            for group in [1, 3, 5, 7] {
                config.text_config.num_attention_heads = group * 4;
                let result = engine_domain(&config, &serde_json::to_value(&config).unwrap());
                assert_eq!(
                    result.unwrap_err().to_string(),
                    engine_refusal(
                        "text_config.num_attention_heads",
                        "dense verify requires an even local GQA group: eight query rows must fill m16 tiles",
                        format!("TP4 group {group} from {} Q heads", group * 4),
                    )
                );
            }
        }
    }

    #[test]
    fn theta_must_stay_positive_in_f32() {
        let mut config =
            Qwen38Config::from_bytes(include_bytes!("../fixtures/config.json")).unwrap();
        config.text_config.rope_parameters.rope_theta = 1e-100;
        config.text_config.mtp.rope_theta = 1e-100;
        assert_eq!(
            engine_domain(&config, &serde_json::to_value(&config).unwrap())
                .unwrap_err()
                .to_string(),
            engine_refusal(
                "text_config.rope_parameters.rope_theta",
                "RoPE theta must be finite and positive in f32",
                1e-100
            )
        );
    }

    #[test]
    fn float_theta_and_one_based_ple_bounds() {
        let mut value: serde_json::Value =
            serde_json::from_slice(include_bytes!("../fixtures/config.json")).unwrap();
        value["text_config"]["rope_parameters"]["rope_theta"] = serde_json::json!(1e7);
        value["text_config"]["mtp"]["rope_theta"] = serde_json::json!(1e7);
        let mut config = Qwen38Config::from_bytes(&serde_json::to_vec(&value).unwrap()).unwrap();
        assert_eq!(config.geometry().unwrap().geometry.id, "flash");
        config.text_config.ple_layer_ids = vec![config.text_config.num_hidden_layers];
        structural_domain(&config).unwrap();
        config.text_config.ple_layer_ids = vec![0];
        assert_eq!(
            structural_domain(&config).unwrap_err().to_string(),
            "unsupported structure text_config.ple_layer_ids: every 1-based layer must be in 1..=num_hidden_layers"
        );
    }

    #[test]
    fn selected_catalog_engine_refusals() {
        let config = Qwen38Config::from_bytes(include_bytes!("../fixtures/config.json")).unwrap();
        for (path, replacement, restriction) in [
            (
                "text_config.hc_count",
                8,
                "hyper-connection rows require 4 streams",
            ),
            (
                "text_config.hidden_size",
                5124,
                "hyper-connection rows require a positive width divisible by 256",
            ),
            (
                "text_config.hc_lowrank",
                336,
                "hyper-connection rows require rank 320",
            ),
            ("text_config.ngram_size", 5, "ngram rows require order 3"),
            (
                "text_config.heads_per_ngram",
                16,
                "ngram rows require 8 heads per order",
            ),
            (
                "text_config.ple_conv_kernel_size",
                5,
                "PLE rows require 4 convolution taps",
            ),
            (
                "text_config.ple_embed_dim",
                5124,
                "ngram rows require positive head widths divisible by 16",
            ),
            (
                "text_config.num_experts",
                1024,
                "native expert prefill requires power-of-two local experts in 32..=256 at every served rank count",
            ),
        ] {
            let mut value = serde_json::to_value(&config).unwrap();
            let mut target = &mut value;
            for key in path.split('.') {
                target = &mut target[key];
            }
            *target = replacement.into();
            let changed: Qwen38Config = serde_json::from_value(value.clone()).unwrap();
            assert_eq!(
                engine_domain(&changed, &value).unwrap_err().to_string(),
                engine_refusal(path, restriction, replacement)
            );
        }
        let mut changed = config.clone();
        changed.text_config.linear_num_key_heads = 32;
        changed.text_config.linear_num_value_heads = 96;
        assert_eq!(
            engine_domain(&changed, &serde_json::to_value(&changed).unwrap())
                .unwrap_err()
                .to_string(),
            "unsupported engine shape text_config.linear_num_value_heads: skinny linear projection requires local value heads in 1..=32 at every served rank count, received 96"
        );
        let mut changed = config;
        changed.text_config.num_attention_heads = 48;
        changed.text_config.num_key_value_heads = 2;
        assert_eq!(
            engine_domain(&changed, &serde_json::to_value(&changed).unwrap())
                .unwrap_err()
                .to_string(),
            "unsupported engine shape text_config.num_attention_heads: dense prefill requires padded local GQA group in {1,2,4,8,16} to fill m16 tiles; local groups 9..15 are not padded, received TP4 group 12 padded to 12"
        );
    }

    #[test]
    fn replicated_kv_and_padded_shared_width_pass_engine_domain() {
        let mut config =
            Qwen38Config::from_bytes(include_bytes!("../fixtures/config.json")).unwrap();
        config.text_config.num_key_value_heads = 1;
        config.text_config.shared_expert_intermediate_size += 1;
        engine_domain(&config, &serde_json::to_value(&config).unwrap()).unwrap();
    }

    #[test]
    fn rotary_and_epsilon_engine_refusals() {
        let mut config =
            Qwen38Config::from_bytes(include_bytes!("../fixtures/config.json")).unwrap();
        config.vision_config.num_heads = 8;
        let value = serde_json::to_value(&config).unwrap();
        assert_eq!(
            engine_domain(&config, &value).unwrap_err().to_string(),
            "unsupported engine shape vision_config.hidden_size: vision attention requires an integral head dimension of 64 or 72, received 1152 / 8 heads"
        );
        config.vision_config.num_heads = 16;
        config.text_config.rope_parameters.partial_rotary_factor = 0.5;
        let value = serde_json::to_value(&config).unwrap();
        assert_eq!(
            engine_domain(&config, &value).unwrap_err().to_string(),
            "unsupported engine shape text_config.rope_parameters.partial_rotary_factor: gated attention and sparse indexer require rotary dim in [32, 64], received 0.5"
        );
        config.text_config.rope_parameters.partial_rotary_factor = 0.25;
        config.text_config.rms_norm_eps = 0.0;
        let value = serde_json::to_value(&config).unwrap();
        assert_eq!(
            engine_domain(&config, &value).unwrap_err().to_string(),
            "unsupported engine shape text_config.rms_norm_eps: epsilon must be finite and positive in f32, received 0"
        );
    }

    #[test]
    fn compatible_checkpoint_metadata() {
        let mut value: serde_json::Value =
            serde_json::from_slice(include_bytes!("../fixtures/config.json")).unwrap();
        let edits: serde_json::Value =
            serde_json::from_slice(include_bytes!("../fixtures/contributor-config.json")).unwrap();
        value["transformers_version"] = edits["transformers_version"].clone();
        for key in ["initializer_range", "router_aux_loss_coef", "seed"] {
            value["text_config"][key] = edits[key].clone();
        }
        Qwen38Config::from_bytes(&serde_json::to_vec(&value).unwrap()).unwrap();
    }

    #[test]
    fn pinned_geometry_and_rejected_changes() {
        let config = Qwen38Config::from_bytes(include_bytes!("../fixtures/config.json")).unwrap();
        assert_eq!(config.text_config.layer_types.len(), 48);
        for (i, kind) in config.text_config.layer_types.iter().enumerate() {
            assert_eq!(
                kind,
                if i % 4 == 3 {
                    "full_attention"
                } else {
                    "linear_attention"
                }
            );
        }
        assert!(config.text_config.norm_topk_prob);
        let mut changed = config.clone();
        changed.text_config.hc_lowrank = 321;
        assert!(
            changed
                .validate()
                .unwrap_err()
                .to_string()
                .contains("text_config.hc_lowrank")
        );
        let mut changed = config.clone();
        changed.text_config.norm_topk_prob = false;
        assert!(changed.validate().is_err());
        let mut changed = config;
        changed.quantization_config =
            serde_json::json!({"quant_method":"modelopt","quant_algo":"NVFP4","group_size":16});
        // Validation admits the recipe class; the importer checks tensor encodings.
        changed.validate().unwrap();
    }
    #[test]
    fn compatible_aliases_still_require_execution_contract() {
        let mut config =
            Qwen38Config::from_bytes(include_bytes!("../fixtures/config.json")).unwrap();
        config.architectures = vec![ARCHITECTURE_ALIASES[1].into()];
        config.model_type = MODEL_TYPE_ALIASES[1].into();
        config.validate().unwrap();
        config.text_config.rms_norm_eps *= 2.0;
        assert!(
            config
                .validate()
                .unwrap_err()
                .to_string()
                .contains("text_config.rms_norm_eps")
        );
        let mut value: serde_json::Value =
            serde_json::from_slice(include_bytes!("../fixtures/config.json")).unwrap();
        value["text_config"]["unknown_execution_rule"] = true.into();
        assert!(Qwen38Config::from_bytes(&serde_json::to_vec(&value).unwrap()).is_err());
    }

    fn exported_config() -> serde_json::Value {
        let mut value: serde_json::Value =
            serde_json::from_slice(include_bytes!("../fixtures/config.json")).unwrap();
        value["architectures"] = serde_json::json!([
            "Qwen4ExpForConditionalGeneration",
            "Qwen3_8FlashNextForConditionalGeneration"
        ]);
        value["dtype"] = "bfloat16".into();
        value["is_multimodal"] = true.into();
        value["quantization_config"] = serde_json::json!({
            "quant_method":"modelopt", "quant_algo":"MIXED_PRECISION",
            "quantized_layers":{"model.language_model.layers.*.mlp.experts":{
                "quant_algo":"NVFP4","group_size":16}}
        });
        value["vision_config"]["model_type"] = "qwen3_5_vision".into();
        value["vision_config"]["architectures"] = serde_json::Value::Null;
        for section in ["", "text_config", "vision_config"] {
            let map = if section.is_empty() {
                value.as_object_mut().unwrap()
            } else {
                value[section].as_object_mut().unwrap()
            };
            for (field, default) in [
                ("return_dict", serde_json::json!(true)),
                ("is_encoder_decoder", serde_json::json!(false)),
                ("output_attentions", serde_json::json!(false)),
                ("output_hidden_states", serde_json::json!(false)),
                ("chunk_size_feed_forward", serde_json::json!(0)),
                ("id2label", serde_json::json!({"0":"LABEL_0"})),
            ] {
                map.insert(field.into(), default);
            }
        }
        let text = value["text_config"].as_object_mut().unwrap();
        text.remove("full_attention_interval");
        text.remove("partial_rotary_factor");
        text.insert("ple_embedding_dtype".into(), "nvfp4".into());
        text.insert("intermediate_size".into(), 12288.into());
        text.insert("number_of_conv_states".into(), 3.into());
        value
    }

    #[test]
    fn exporter_defaults_and_precision_are_independent_of_architecture() {
        let value = exported_config();
        let config = Qwen38Config::from_bytes(&serde_json::to_vec(&value).unwrap()).unwrap();
        assert_eq!(config.text_config.full_attention_interval, 4);
        assert_eq!(config.text_config.partial_rotary_factor, 0.25);
        assert_eq!(
            config.text_config.ple_embedding_dtype.as_deref(),
            Some("nvfp4")
        );
        for (section, key, wrong) in [
            ("text_config", "number_of_conv_states", serde_json::json!(4)),
            ("text_config", "output_attentions", serde_json::json!(true)),
            ("vision_config", "patch_size", serde_json::json!(32)),
            ("vision_config", "model_type", serde_json::json!("unknown")),
            ("text_config", "new_execution_rule", serde_json::json!(true)),
        ] {
            let mut changed = value.clone();
            changed[section][key] = wrong;
            let error =
                Qwen38Config::from_bytes(&serde_json::to_vec(&changed).unwrap()).unwrap_err();
            assert!(error.to_string().contains(key), "{key}: {error}");
        }
        for missing in ["rope_parameters", "linear_conv_kernel_dim"] {
            let mut changed = value.clone();
            changed["text_config"]
                .as_object_mut()
                .unwrap()
                .remove(missing);
            assert!(Qwen38Config::from_bytes(&serde_json::to_vec(&changed).unwrap()).is_err());
        }
        assert!(
            Qwen38Config::from_bytes(br#"{"model_type":"a","model_type":"b"}"#)
                .unwrap_err()
                .to_string()
                .contains("duplicate metadata field")
        );
    }

    #[test]
    fn checkpoint_config() {
        let Ok(root) = std::env::var("CHOPINFER_QWEN38_CHECKPOINT") else {
            eprintln!("SKIP: set CHOPINFER_QWEN38_CHECKPOINT for checkpoint config");
            return;
        };
        Qwen38Config::load(Path::new(&root)).unwrap();
    }
}
