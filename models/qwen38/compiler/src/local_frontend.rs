//! Resolve frontend assets without assigning the reference release's identity.
use crate::Result;
use runtime_frontend_contract::FrontendDescriptorV1;
use runtime_tokenizer::Tokenizer;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;
use std::path::Path;

/// The lowercase hex SHA-256 of `bytes`.
fn digest(bytes: &[u8]) -> String {
    use sha2::Digest;
    format!("{:x}", sha2::Sha256::digest(bytes))
}

pub const REQUIRED: &[&str] = &[
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "model.safetensors.index.json",
];
pub const OPTIONAL: &[&str] = &["chat_template.jinja", "generation_config.json"];

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LocalFrontend {
    pub definition: Vec<u8>,
    pub configuration: Vec<u8>,
    pub component: Vec<u8>,
}

impl LocalFrontend {
    pub fn load(root: &Path, source_identity: &str) -> Result<Self> {
        let mut inputs = BTreeMap::new();
        for name in ["tokenizer.json", "tokenizer_config.json"] {
            inputs.insert(
                name.into(),
                crate::source::read(root, name, 32 * 1024 * 1024)?,
            );
        }
        for name in OPTIONAL {
            match std::fs::symlink_metadata(root.join(name)) {
                Ok(_) => {
                    inputs.insert((*name).into(), crate::source::read(root, name, 64 * 1024)?);
                }
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                Err(e) => return Err(e.into()),
            }
        }
        Self::resolve(&inputs, source_identity)
    }

    pub fn resolve(inputs: &BTreeMap<String, Vec<u8>>, source_identity: &str) -> Result<Self> {
        if source_identity.len() != 64 || !source_identity.bytes().all(|b| b.is_ascii_hexdigit()) {
            return Err("local frontend needs a measured SHA-256 source identity".into());
        }
        let definition = inputs
            .get("tokenizer.json")
            .ok_or("missing tokenizer.json")?
            .clone();
        let configuration = inputs
            .get("tokenizer_config.json")
            .ok_or("missing tokenizer_config.json")?
            .clone();
        let config: Value = serde_json::from_slice(&configuration)?;
        validate_semantics(&config, inputs)?;
        let mut frontend: Value = serde_json::from_slice(crate::frontend::COMPONENT)?;
        frontend["models"][0]["id"] = format!("local-{source_identity}").into();
        frontend["models"][0]["owned_by"] = "local".into();
        let mut identities = BTreeMap::new();
        for name in [
            "tokenizer.json",
            "tokenizer_config.json",
            "chat_template.jinja",
            "generation_config.json",
        ] {
            identities.insert(name, inputs.get(name).map(|bytes| digest(bytes)));
        }
        frontend["tokenizer"]["repository"] = "local".into();
        frontend["tokenizer"]["revision"] = digest(&serde_json::to_vec(&identities)?).into();
        for (key, bytes) in [
            ("definition", &definition),
            ("configuration", &configuration),
        ] {
            frontend["tokenizer"][key]["byte_length"] = (bytes.len() as u64).into();
            frontend["tokenizer"][key]["sha256"] = digest(bytes).into();
        }
        let tokenizer: Value = serde_json::from_slice(&definition)?;
        frontend["tokenizer"]["merge_count"] = (tokenizer["model"]["merges"]
            .as_array()
            .ok_or("missing BPE merges")?
            .len() as u64)
            .into();
        if let Some(bytes) = inputs.get("generation_config.json") {
            let generation: Value = serde_json::from_slice(bytes)?;
            validate_generation(&generation)?;
            for key in ["temperature", "top_p", "top_k"] {
                if let Some(value) = generation.get(key) {
                    frontend["generation_config"][key] = value.clone();
                }
            }
        }
        let component = serde_json::to_vec(&frontend)?;
        let resolved = Self {
            definition,
            configuration,
            component,
        };
        resolved.tokenizer()?;
        Ok(resolved)
    }

    pub fn tokenizer(&self) -> Result<Tokenizer> {
        let descriptor = FrontendDescriptorV1::from_component_bytes(&self.component)?;
        Ok(runtime_frontend_contract::tokenizer_from_components_v1(
            &descriptor,
            &self.definition,
            &self.configuration,
        )?)
    }
}

fn validate_semantics(config: &Value, inputs: &BTreeMap<String, Vec<u8>>) -> Result<()> {
    let reference: Value =
        serde_json::from_slice(include_bytes!("../fixtures/tokenizer-semantics.json"))?;
    let template = config["chat_template"]
        .as_str()
        .ok_or("tokenizer_config.chat_template must be a supported string")?;
    let expected = reference["chat_template_sha256"]
        .as_str()
        .ok_or("invalid template fixture")?;
    if digest(template.as_bytes()) != expected {
        return Err(
            "unsupported chat template semantics; this family supports its declared text renderer"
                .into(),
        );
    }
    if let Some(bytes) = inputs.get("chat_template.jinja")
        && digest(bytes) != expected
    {
        return Err("chat_template.jinja disagrees with the supported renderer".into());
    }
    let expected_config = reference["configuration"]
        .as_object()
        .ok_or("invalid configuration fixture")?;
    for (key, value) in config
        .as_object()
        .ok_or("tokenizer configuration must be an object")?
    {
        if [
            "chat_template",
            "name_or_path",
            "_name_or_path",
            "transformers_version",
        ]
        .contains(&key.as_str())
        {
            continue;
        }
        if expected_config.get(key) != Some(value) {
            return Err(format!("unsupported tokenizer_config.{key}").into());
        }
    }
    for (key, value) in expected_config {
        if config.get(key) != Some(value) {
            return Err(format!("missing or unsupported tokenizer_config.{key}").into());
        }
    }
    Ok(())
}

fn validate_generation(value: &Value) -> Result<()> {
    let expected = serde_json::json!({"bos_token_id":248044,"pad_token_id":248044,"eos_token_id":[248046,248044],"do_sample":true});
    for (key, setting) in value
        .as_object()
        .ok_or("generation config must be an object")?
    {
        // Some exporters serialize every optional GenerationConfig field as
        // null. These known unset fields request no serving behavior. A value
        // still needs an implementation; unknown fields remain a refusal.
        if setting.is_null() && UNSET_GENERATION_FIELDS.contains(&key.as_str()) {
            continue;
        }
        if [
            "temperature",
            "top_k",
            "top_p",
            "transformers_version",
            "_from_model_config",
        ]
        .contains(&key.as_str())
        {
            continue;
        }
        if expected.get(key) != Some(setting) {
            return Err(format!("unsupported generation_config.{key}").into());
        }
    }
    Ok(())
}

const UNSET_GENERATION_FIELDS: &[&str] = &[
    "max_length",
    "max_new_tokens",
    "min_length",
    "min_new_tokens",
    "early_stopping",
    "max_time",
    "stop_strings",
    "num_beams",
    "use_cache",
    "cache_implementation",
    "cache_config",
    "min_p",
    "top_h",
    "typical_p",
    "epsilon_cutoff",
    "eta_cutoff",
    "repetition_penalty",
    "encoder_repetition_penalty",
    "length_penalty",
    "no_repeat_ngram_size",
    "bad_words_ids",
    "renormalize_logits",
    "forced_bos_token_id",
    "forced_eos_token_id",
    "remove_invalid_values",
    "exponential_decay_length_penalty",
    "suppress_tokens",
    "begin_suppress_tokens",
    "sequence_bias",
    "token_healing",
    "guidance_scale",
    "watermarking_config",
    "num_return_sequences",
    "output_attentions",
    "output_hidden_states",
    "output_scores",
    "output_logits",
    "return_dict_in_generate",
    "encoder_no_repeat_ngram_size",
    "decoder_start_token_id",
    "is_assistant",
    "num_assistant_tokens",
    "num_assistant_tokens_schedule",
    "assistant_confidence_threshold",
    "prompt_lookup_num_tokens",
    "max_matching_ngram_size",
    "assistant_early_exit",
    "assistant_lookbehind",
    "target_lookbehind",
    "compile_config",
    "disable_compile",
    "continuous_batching_config",
    "low_memory",
    "penalty_alpha",
    "dola_layers",
    "diversity_penalty",
    "num_beam_groups",
    "constraints",
    "force_words_ids",
    "prefill_chunk_size",
];

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn unsupported_templates_and_generation_rules_are_refused() {
        let mut config: Value =
            serde_json::from_slice(include_bytes!("../fixtures/tokenizer-semantics.json")).unwrap();
        config["configuration"]["chat_template"] = "different renderer".into();
        assert!(validate_semantics(&config["configuration"], &BTreeMap::new()).is_err());
        assert!(validate_generation(&serde_json::json!({"eos_token_id":[1]})).is_err());
        assert!(validate_generation(&serde_json::json!({"temperature":0.5,"top_k":12})).is_ok());
    }

    #[test]
    fn known_unset_generation_fields_do_not_enable_unsupported_behavior() {
        let fields: serde_json::Map<_, _> = UNSET_GENERATION_FIELDS
            .iter()
            .map(|k| ((*k).into(), Value::Null))
            .collect();
        assert!(validate_generation(&Value::Object(fields)).is_ok());
        for key in ["num_beams", "max_length", "compile_config", "stop_strings"] {
            let value = serde_json::json!({key: 2});
            assert!(
                validate_generation(&value)
                    .unwrap_err()
                    .to_string()
                    .contains(key)
            );
        }
        assert!(validate_generation(&serde_json::json!({"unrecognized_execution":null})).is_err());
    }

    #[test]
    fn local_identity_does_not_come_from_reference_tokenizers() {
        for checkpoint in crate::checkpoint_index::checkpoints() {
            let Some(root) = checkpoint.root() else {
                continue;
            };
            let first = LocalFrontend::load(&root, &"1".repeat(64)).unwrap();
            let second = LocalFrontend::load(&root, &"2".repeat(64)).unwrap();
            let a: Value = serde_json::from_slice(&first.component).unwrap();
            let b: Value = serde_json::from_slice(&second.component).unwrap();
            assert_ne!(a["models"], b["models"]);
            assert_eq!(a["tokenizer"], b["tokenizer"]);
            assert_eq!(a["tokenizer"]["repository"], "local");
            assert_eq!(a["tokenizer"]["revision"].as_str().unwrap().len(), 64);
            assert_ne!(a["tokenizer"]["revision"], crate::REVISION);
        }
    }
    #[test]
    fn compatible_tokenizer_serialization_and_ordinary_vocabulary_are_local() {
        for checkpoint in crate::checkpoint_index::checkpoints() {
            let Some(root) = checkpoint.root() else {
                continue;
            };
            let baseline = LocalFrontend::load(&root, &"1".repeat(64)).unwrap();
            let mut definition: Value = serde_json::from_slice(&baseline.definition).unwrap();
            let vocabulary = definition["model"]["vocab"].as_object_mut().unwrap();
            let first = vocabulary
                .iter()
                .find(|(_, v)| v.as_u64() == Some(0))
                .unwrap()
                .0
                .clone();
            let second = vocabulary
                .iter()
                .find(|(_, v)| v.as_u64() == Some(1))
                .unwrap()
                .0
                .clone();
            vocabulary.insert(first, 1.into());
            vocabulary.insert(second, 0.into());
            let mut config: Value = serde_json::from_slice(&baseline.configuration).unwrap();
            config["name_or_path"] = "local-fine-tune".into();
            let inputs = BTreeMap::from([
                (
                    "tokenizer.json".into(),
                    serde_json::to_vec(&definition).unwrap(),
                ),
                (
                    "tokenizer_config.json".into(),
                    serde_json::to_vec_pretty(&config).unwrap(),
                ),
            ]);
            let changed = LocalFrontend::resolve(&inputs, &"2".repeat(64)).unwrap();
            let component: Value = serde_json::from_slice(&changed.component).unwrap();
            assert_eq!(
                component["tokenizer"]["definition"]["sha256"],
                digest(&changed.definition)
            );
            assert_ne!(changed.definition, baseline.definition);
            let mut bad = inputs.clone();
            config["model_max_length"] = 42.into();
            bad.insert(
                "tokenizer_config.json".into(),
                serde_json::to_vec(&config).unwrap(),
            );
            assert!(LocalFrontend::resolve(&bad, &"2".repeat(64)).is_err());
        }
    }
}
