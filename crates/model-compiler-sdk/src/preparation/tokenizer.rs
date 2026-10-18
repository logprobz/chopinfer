//! A checkpoint's own tokenizer files, admitted by what they mean rather
//! than by their bytes. A family's frontend renders one chat format and
//! tokenizes with one configuration, so a checkpoint whose tokenizer files
//! state the same template and configuration is served with its own files,
//! named by their content instead of the official release.
use crate::files::Result;
use serde::Deserialize;
use serde_json::{Map, Value};
use std::collections::BTreeMap;

/// `tokenizer_config.json` keys that record where a configuration came
/// from or the template it carries, not how it tokenizes.
pub const PROVENANCE_KEYS: &[&str] = &[
    "chat_template",
    "name_or_path",
    "_name_or_path",
    "transformers_version",
    "is_local",
];

/// The tokenizer files a local tokenizer revision names, in order.
pub const TOKENIZER_FILES: &[&str] = &[
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "generation_config.json",
];

fn digest(bytes: &[u8]) -> String {
    runtime_weights::sha256_bytes(bytes)
}

/// The tokenizer configuration a family serves, an excerpt of its official
/// release (`fixtures/tokenizer-semantics.json`): the SHA-256 of its chat
/// template and every other key of its `tokenizer_config.json`.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TokenizerSemantics {
    pub chat_template_sha256: String,
    pub configuration: Map<String, Value>,
}

impl TokenizerSemantics {
    pub fn parse(bytes: &[u8]) -> Result<Self> {
        let semantics: Self = serde_json::from_slice(bytes)?;
        if semantics
            .configuration
            .keys()
            .any(|key| PROVENANCE_KEYS.contains(&key.as_str()))
        {
            return Err("tokenizer semantics name a provenance key".into());
        }
        Ok(semantics)
    }

    /// Refuses a checkpoint's `tokenizer_config.json` (`configuration`) and
    /// optional `chat_template.jinja` unless they state this chat template
    /// and every other key as this excerpt does. The template is the
    /// configuration's `chat_template` when it has one, which a
    /// `chat_template.jinja` beside it must equal, else that file's.
    /// Provenance keys are ignored.
    pub fn check(&self, configuration: &Value, chat_template: Option<&[u8]>) -> Result<()> {
        let config = configuration
            .as_object()
            .ok_or("tokenizer configuration must be an object")?;
        let template = match (config.get("chat_template"), chat_template) {
            (Some(template), _) => template
                .as_str()
                .ok_or("tokenizer_config.chat_template must be a supported string")?
                .as_bytes(),
            (None, Some(file)) => file,
            (None, None) => {
                return Err(
                    "the checkpoint has no chat template (tokenizer_config.chat_template or chat_template.jinja)"
                        .into(),
                );
            }
        };
        if digest(template) != self.chat_template_sha256 {
            return Err(
                "unsupported chat template semantics; this family supports its declared text renderer"
                    .into(),
            );
        }
        if let Some(file) = chat_template
            && digest(file) != self.chat_template_sha256
        {
            return Err("chat_template.jinja disagrees with the supported renderer".into());
        }
        for (key, value) in config {
            if PROVENANCE_KEYS.contains(&key.as_str()) {
                continue;
            }
            if self.configuration.get(key) != Some(value) {
                return Err(format!("unsupported tokenizer_config.{key}").into());
            }
        }
        for (key, value) in &self.configuration {
            if config.get(key) != Some(value) {
                return Err(format!("missing or unsupported tokenizer_config.{key}").into());
            }
        }
        Ok(())
    }
}

/// The content revision of a local tokenizer: the SHA-256 of the digest of
/// each of [`TOKENIZER_FILES`] the checkpoint has (`files`, by name), null
/// for one it lacks.
pub fn content_revision(files: &BTreeMap<String, Vec<u8>>) -> Result<String> {
    let identities: BTreeMap<&str, Option<String>> = TOKENIZER_FILES
        .iter()
        .map(|name| (*name, files.get(*name).map(|bytes| digest(bytes))))
        .collect();
    Ok(digest(&serde_json::to_vec(&identities)?))
}

/// Points `frontend`'s tokenizer at `definition` (`tokenizer.json`) and
/// `configuration` (`tokenizer_config.json`): their lengths and digests,
/// the definition's BPE merge count, and `repository` at `revision`.
pub fn rewrite_tokenizer(
    frontend: &mut Value,
    definition: &[u8],
    configuration: &[u8],
    repository: &str,
    revision: &str,
) -> Result<()> {
    let tokenizer = frontend
        .get_mut("tokenizer")
        .and_then(Value::as_object_mut)
        .ok_or("frontend has no tokenizer")?;
    tokenizer.insert("repository".into(), repository.into());
    tokenizer.insert("revision".into(), revision.into());
    for (key, bytes) in [("definition", definition), ("configuration", configuration)] {
        let file = tokenizer
            .get_mut(key)
            .and_then(Value::as_object_mut)
            .ok_or_else(|| format!("frontend tokenizer has no {key}"))?;
        file.insert("byte_length".into(), (bytes.len() as u64).into());
        file.insert("sha256".into(), digest(bytes).into());
    }
    let parsed: Value = serde_json::from_slice(definition)?;
    let merges = parsed["model"]["merges"]
        .as_array()
        .ok_or("missing BPE merges")?
        .len() as u64;
    tokenizer.insert("merge_count".into(), merges.into());
    Ok(())
}

/// Names `frontend`'s one served model `id`, owned by `owner`.
pub fn rename_model(frontend: &mut Value, id: &str, owner: &str) -> Result<()> {
    let [model] = frontend
        .get_mut("models")
        .and_then(Value::as_array_mut)
        .map(Vec::as_mut_slice)
        .ok_or("frontend has no models")?
    else {
        return Err("frontend must serve one model".into());
    };
    model["id"] = id.into();
    model["owned_by"] = owner.into();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn semantics() -> TokenizerSemantics {
        TokenizerSemantics::parse(
            &serde_json::to_vec(&json!({
                "chat_template_sha256": digest(b"{{ messages }}"),
                "configuration": {"eos_token": "<|end|>", "model_max_length": 8},
            }))
            .unwrap(),
        )
        .unwrap()
    }

    #[test]
    fn the_template_and_every_other_key_must_match() {
        let semantics = semantics();
        let official = json!({
            "chat_template": "{{ messages }}", "eos_token": "<|end|>",
            "model_max_length": 8, "name_or_path": "anywhere", "is_local": true,
        });
        semantics.check(&official, None).unwrap();
        semantics
            .check(&official, Some(b"{{ messages }}".as_slice()))
            .unwrap();
        // A template only in chat_template.jinja.
        let mut separate = official.clone();
        separate.as_object_mut().unwrap().remove("chat_template");
        semantics
            .check(&separate, Some(b"{{ messages }}".as_slice()))
            .unwrap();
        let refused = |config: &Value, file: Option<&[u8]>| {
            semantics.check(config, file).unwrap_err().to_string()
        };
        assert!(refused(&separate, None).contains("no chat template"));
        assert!(refused(&official, Some(b"{{ other }}".as_slice())).contains("disagrees"));
        let mut edited = official.clone();
        edited["chat_template"] = "{{ other }}".into();
        assert!(refused(&edited, None).contains("chat template semantics"));
        let mut longer = official.clone();
        longer["model_max_length"] = 9.into();
        assert!(refused(&longer, None).contains("tokenizer_config.model_max_length"));
        let mut added = official.clone();
        added["padding_side"] = "left".into();
        assert!(refused(&added, None).contains("tokenizer_config.padding_side"));
        let mut missing = official;
        missing.as_object_mut().unwrap().remove("eos_token");
        assert!(
            refused(&missing, None).contains("missing or unsupported tokenizer_config.eos_token")
        );
    }

    #[test]
    fn an_excerpt_may_not_name_a_provenance_key() {
        let bytes = serde_json::to_vec(&json!({
            "chat_template_sha256": "0".repeat(64),
            "configuration": {"name_or_path": "x"},
        }))
        .unwrap();
        assert!(TokenizerSemantics::parse(&bytes).is_err());
    }

    #[test]
    fn the_rewrite_names_the_files_and_their_content() {
        let mut frontend = json!({
            "models": [{"id": "Official", "owned_by": "author"}],
            "tokenizer": {
                "repository": "author/model", "revision": "1".repeat(40),
                "definition": {"name": "tokenizer.json", "byte_length": 1, "sha256": "0".repeat(64)},
                "configuration": {"name": "tokenizer_config.json", "byte_length": 1, "sha256": "0".repeat(64)},
                "merge_count": 7,
            },
        });
        let definition = br#"{"model":{"merges":["a b","c d"]}}"#;
        let configuration = b"{}";
        let files = BTreeMap::from([
            ("tokenizer.json".to_owned(), definition.to_vec()),
            ("tokenizer_config.json".to_owned(), configuration.to_vec()),
            ("unrelated.json".to_owned(), b"x".to_vec()),
        ]);
        let revision = content_revision(&files).unwrap();
        // Only the tokenizer files name the revision.
        let mut fewer = files.clone();
        fewer.remove("unrelated.json");
        assert_eq!(content_revision(&fewer).unwrap(), revision);
        fewer.insert("chat_template.jinja".into(), b"t".to_vec());
        assert_ne!(content_revision(&fewer).unwrap(), revision);
        rewrite_tokenizer(&mut frontend, definition, configuration, "local", &revision).unwrap();
        rename_model(&mut frontend, "local-x", "local").unwrap();
        let tokenizer = &frontend["tokenizer"];
        assert_eq!(tokenizer["repository"], "local");
        assert_eq!(tokenizer["revision"], revision.as_str());
        assert_eq!(tokenizer["merge_count"], 2);
        assert_eq!(tokenizer["definition"]["name"], "tokenizer.json");
        assert_eq!(
            tokenizer["definition"]["byte_length"],
            definition.len() as u64
        );
        assert_eq!(tokenizer["configuration"]["sha256"], digest(configuration));
        assert_eq!(
            frontend["models"],
            json!([{"id": "local-x", "owned_by": "local"}])
        );
        assert!(rewrite_tokenizer(&mut frontend, b"{}", b"{}", "local", &revision).is_err());
    }
}
