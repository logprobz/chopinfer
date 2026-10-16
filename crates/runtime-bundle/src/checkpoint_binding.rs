//! Portable compiler evidence binding a checkpoint identity to packaged files.
//! This is checked consistency within a trusted package, not a package signature.
use super::*;

pub const ROLE: &str = "chop.checkpoint-binding.v1";
pub const SCHEMA: &str = "chopinfer.checkpoint-binding.v1";

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FileDigest {
    pub byte_length: u64,
    pub sha256: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CheckpointBinding {
    pub schema: String,
    pub repository: String,
    pub revision: String,
    pub served_model_name: String,
    pub source_lock_sha256: String,
    /// Every physical package file except the binding itself and manifest.
    pub files: BTreeMap<String, FileDigest>,
}

fn bytes(file: &VerifiedComponentFileV1, limit: u64) -> Result<&[u8], BundleError> {
    if file.byte_length() > limit {
        return Err(BundleError::validation(
            "checkpoint binding metadata exceeds limit",
        ));
    }
    Ok(file.bytes())
}

pub(super) fn verify(
    manifest: &BundleManifestV1,
    components: &BTreeMap<BundleOpaqueId, VerifiedComponentFileV1>,
) -> Result<(), BundleError> {
    let declared: Vec<_> = manifest
        .components()
        .iter()
        .filter(|c| c.role().as_str() == ROLE)
        .collect();
    // Bound rank metadata keeps the requirement even if a binding declaration is removed.
    let mut required = false;
    for component in manifest.components().iter().filter(|c| {
        c.role().as_str().starts_with("chop.bundle-metadata.")
            && c.format().as_str().ends_with(".json.v1")
    }) {
        if let Some(file) = components.get(component.id()) {
            let value = parse_strict_json(bytes(file, 64 << 20)?)?;
            if let Some(binding) = value.get("checkpoint_binding") {
                if binding.as_str() != Some(ROLE) {
                    return Err(BundleError::validation(
                        "unsupported checkpoint binding reference",
                    ));
                }
                required = true;
            }
        }
    }
    if declared.is_empty() && !required {
        return Ok(());
    }
    if declared.len() != 1 {
        return Err(BundleError::validation(
            "package requires one checkpoint binding",
        ));
    }
    let declaration = declared[0];
    let file = components
        .get(declaration.id())
        .ok_or_else(|| BundleError::validation("checkpoint binding has no verified bytes"))?;
    let proof: CheckpointBinding =
        serde_json::from_value(parse_strict_json(bytes(file, 4 << 20)?)?)
            .map_err(|e| BundleError::validation(format!("checkpoint binding: {e}")))?;
    if proof.schema != SCHEMA
        || proof.repository.is_empty()
        || proof.served_model_name.is_empty()
        || proof.revision.len() != 40
        || !proof.revision.bytes().all(|b| b.is_ascii_hexdigit())
    {
        return Err(BundleError::validation(
            "invalid checkpoint binding identity",
        ));
    }
    validate_sha256(&proof.source_lock_sha256)?;
    let mut expected = BTreeMap::new();
    for component in manifest
        .components()
        .iter()
        .filter(|c| c.id() != declaration.id())
    {
        expected.insert(
            component.path().as_str().to_owned(),
            FileDigest {
                byte_length: component.byte_length(),
                sha256: component.sha256().to_owned(),
            },
        );
    }
    if let Some(native) = manifest.native_module() {
        expected.insert(
            native.path().as_str().to_owned(),
            FileDigest {
                byte_length: native.byte_length(),
                sha256: native.sha256().to_owned(),
            },
        );
    }
    if expected != proof.files {
        return Err(BundleError::validation(
            "checkpoint binding differs from packaged file identities",
        ));
    }
    let frontends: Vec<_> = manifest
        .components()
        .iter()
        .filter(|c| c.role().as_str() == "chop.frontend.openai.v1")
        .collect();
    if frontends.len() != 1 {
        return Err(BundleError::validation(
            "checkpoint binding needs one frontend",
        ));
    }
    let frontend = components
        .get(frontends[0].id())
        .ok_or_else(|| BundleError::validation("bound frontend has no verified bytes"))?;
    let value = parse_strict_json(bytes(frontend, 4 << 20)?)?;
    if value["tokenizer"]["repository"] != proof.repository
        || value["tokenizer"]["revision"] != proof.revision
        || value["models"]
            .as_array()
            .is_none_or(|models| models.len() != 1 || models[0]["id"] != proof.served_model_name)
    {
        return Err(BundleError::validation(
            "checkpoint binding differs from frontend identity",
        ));
    }
    Ok(())
}
