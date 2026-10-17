//! The checkpoint as binding sees it: every safetensors header, the weight
//! map and `config.json`'s `quantization_config`, read without any tensor
//! payload. Files open as direct preparation opens them
//! (`runtime_weights::checkpoint::CheckpointFile`).
use model_compiler_sdk::quantization::projection::metadata::HeaderCompanionRules;
use model_deepseek41_contract::{admission, projections};
use projection_spec::bind::{BoundDeclaration, CheckpointMetadata};
use projection_spec::registry::Registry;
use runtime_weights::checkpoint::CheckpointFile;
use serde_json::Value;
use std::collections::BTreeMap;
use std::os::unix::fs::FileExt;
use std::path::Path;

/// The largest safetensors header read, as the host bounds it.
const MAX_HEADER_BYTES: u64 = 256 << 20;

/// The largest document read.
const MAX_DOCUMENT_BYTES: u64 = 64 << 20;

fn open(root: &Path, name: &str) -> Result<CheckpointFile, String> {
    CheckpointFile::open(root, name).map_err(|e| format!("{name}: {e}"))
}

pub fn document(root: &Path, name: &str) -> Result<Value, String> {
    let bytes = open(root, name)?
        .read_bounded(MAX_DOCUMENT_BYTES)
        .map_err(|e| format!("{name}: {e}"))?;
    serde_json::from_slice(&bytes).map_err(|e| format!("{name}: {e}"))
}

/// The JSON header of the safetensors file `name`.
fn header(root: &Path, name: &str) -> Result<Value, String> {
    let file = open(root, name)?;
    let length = file.metadata().map_err(|e| format!("{name}: {e}"))?.len();
    let mut prefix = [0_u8; 8];
    file.read_exact_at(&mut prefix, 0)
        .map_err(|e| format!("{name}: {e}"))?;
    let header_length = u64::from_le_bytes(prefix);
    if header_length == 0 || header_length > MAX_HEADER_BYTES || 8 + header_length > length {
        return Err(format!(
            "{name}: safetensors header length is outside the file"
        ));
    }
    let mut bytes = vec![0; header_length as usize];
    file.read_exact_at(&mut bytes, 8)
        .map_err(|e| format!("{name}: {e}"))?;
    serde_json::from_slice(&bytes).map_err(|e| format!("{name}: {e}"))
}

/// The checkpoint's binding metadata: every tensor of every file the weight
/// map names, checked against that map.
pub fn metadata(root: &Path) -> Result<CheckpointMetadata, String> {
    let config = document(root, "config.json")?;
    let index = document(root, "model.safetensors.index.json")?;
    let files = index["weight_map"]
        .as_object()
        .ok_or("model.safetensors.index.json: no weight_map object")?
        .values()
        .map(|file| {
            file.as_str()
                .map(str::to_owned)
                .ok_or("weight_map value is not a file name")
        })
        .collect::<Result<std::collections::BTreeSet<_>, _>>()?;
    let headers = files
        .iter()
        .map(|file| Ok((file.clone(), header(root, file)?)))
        .collect::<Result<BTreeMap<_, _>, String>>()?;
    let quantization = config
        .get("quantization_config")
        .cloned()
        .unwrap_or(Value::Null);
    let metadata =
        CheckpointMetadata::from_headers(&headers, Some(&index), quantization).map_err(|e| e.0)?;
    if metadata.tensors.len() != index["weight_map"].as_object().map_or(0, |map| map.len()) {
        return Err("the headers and the weight map name different tensors".into());
    }
    Ok(metadata)
}

/// Admit the checkpoint's config, then bind every tensor its headers name
/// to the projection declaration ([`bind_metadata`]). `encoder` holds the
/// vision and aligner tensors the encoder declaration validated; the
/// declaration names them as that owner's (`external_tensors`).
pub fn bind(
    root: &Path,
    encoder: std::collections::BTreeSet<String>,
) -> Result<BoundDeclaration, String> {
    let config = document(root, "config.json")?;
    let mut metadata = metadata(root)?;
    metadata
        .validated_owners
        .insert("encoder-spec".into(), encoder);
    bind_metadata(&config, &metadata)
}

/// Bind `metadata` at the geometry `config` admits: an unknown tensor, a
/// missing one, a missing companion or an inadmissible dtype, shape or
/// format is refused by name before any payload is read.
pub fn bind_metadata(
    config: &Value,
    metadata: &CheckpointMetadata,
) -> Result<BoundDeclaration, String> {
    let geometry = admission::admit(config)?;
    let declaration = projections::declaration(geometry).map_err(|e| e.0)?;
    let mut rules = HeaderCompanionRules::new(metadata, None).map_err(|e| e.0)?;
    rules.roles = projections::header_roles(&declaration);
    declaration
        .bind(
            &projections::geometry(geometry),
            &Registry::default(),
            metadata,
            &rules,
        )
        .map_err(|e| e.0)
}
