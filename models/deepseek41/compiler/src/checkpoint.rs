//! The checkpoint as binding sees it: every safetensors header, the weight
//! map and `config.json`'s `quantization_config`, read without any tensor
//! payload. Files open as direct preparation opens them
//! (`runtime_weights::checkpoint::CheckpointFile`).
use model_compiler_sdk::quantization::projection::metadata::HeaderCompanionRules;
use model_deepseek41_contract::serving::RANK_IMAGE_ROLE;
use model_deepseek41_contract::{admission, projections};
use projection_spec::bind::{BoundDeclaration, CheckpointMetadata};
use projection_spec::registry::Registry;
use runtime_bundle_abi::weight_plan::{CheckpointFileV1, NamedBytesV1, WeightPlanRequestV1};
use runtime_weights::checkpoint::CheckpointFile;
use serde_json::Value;
use std::collections::BTreeMap;
use std::os::unix::fs::FileExt;
use std::path::Path;

/// The largest safetensors header read, as the host bounds it.
const MAX_HEADER_BYTES: u64 = 256 << 20;

/// The largest document read.
const MAX_DOCUMENT_BYTES: u64 = 64 << 20;

const CONFIG: &str = "config.json";
const INDEX: &str = "model.safetensors.index.json";

fn open(root: &Path, name: &str) -> Result<CheckpointFile, String> {
    CheckpointFile::open(root, name).map_err(|e| format!("{name}: {e}"))
}

pub fn document(root: &Path, name: &str) -> Result<Value, String> {
    let bytes = open(root, name)?
        .read_bounded(MAX_DOCUMENT_BYTES)
        .map_err(|e| format!("{name}: {e}"))?;
    serde_json::from_slice(&bytes).map_err(|e| format!("{name}: {e}"))
}

/// The safetensors file `name` as the host's checkpoint opener checks it
/// (`runtime_weights::parse_safetensors_header` over the file's length):
/// its length and its JSON header bytes.
fn header_bytes(root: &Path, name: &str) -> Result<(u64, Vec<u8>), String> {
    let at = |e: std::io::Error| format!("{name}: {e}");
    let file = open(root, name)?;
    let length = file.metadata().map_err(at)?.len();
    let mut prefix = [0_u8; 8];
    file.read_exact_at(&mut prefix, 0).map_err(at)?;
    let header_length = u64::from_le_bytes(prefix);
    if header_length == 0 || header_length > MAX_HEADER_BYTES || 8 + header_length > length {
        return Err(format!(
            "{name}: safetensors header length is outside the file"
        ));
    }
    let mut bytes = vec![0; 8 + header_length as usize];
    file.read_exact_at(&mut bytes, 0).map_err(at)?;
    runtime_weights::parse_safetensors_header(&mut bytes.as_slice(), length)
        .map_err(|e| format!("{name}: {e}"))?;
    bytes.drain(..8);
    Ok((length, bytes))
}

/// The JSON header of the safetensors file `name`.
fn header(root: &Path, name: &str) -> Result<Value, String> {
    let (_, bytes) = header_bytes(root, name)?;
    serde_json::from_slice(&bytes).map_err(|e| format!("{name}: {e}"))
}

/// The files the weight map names, in name order.
fn weight_files(index: &Value) -> Result<std::collections::BTreeSet<String>, String> {
    index["weight_map"]
        .as_object()
        .ok_or("model.safetensors.index.json: no weight_map object")?
        .values()
        .map(|file| {
            file.as_str()
                .map(str::to_owned)
                .ok_or_else(|| "weight_map value is not a file name".to_owned())
        })
        .collect()
}

/// The plan request of the text image for rank `rank` of `ranks` of the
/// checkpoint at `root`: every file the weight map names with its header,
/// checked as the host checks it, and the config and weight map the plan
/// reads.
pub fn request(root: &Path, ranks: u32, rank: u32) -> Result<WeightPlanRequestV1, String> {
    let read = |name: &str| {
        open(root, name)?
            .read_bounded(MAX_DOCUMENT_BYTES)
            .map_err(|e| format!("{name}: {e}"))
    };
    let index = read(INDEX)?;
    let parsed: Value = serde_json::from_slice(&index).map_err(|e| format!("{INDEX}: {e}"))?;
    let files = weight_files(&parsed)?
        .into_iter()
        .map(|name| {
            let (byte_length, header) = header_bytes(root, &name)?;
            Ok(CheckpointFileV1 {
                data_offset: 8 + header.len() as u64,
                name,
                byte_length,
                header,
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
    let checkpoint_identity = runtime_weights::checkpoint::header_identity(
        files
            .iter()
            .map(|file| (file.name.as_str(), file.byte_length, file.header.as_slice())),
    );
    Ok(WeightPlanRequestV1 {
        image_role: RANK_IMAGE_ROLE.into(),
        ranks,
        rank,
        checkpoint_identity,
        files,
        documents: vec![
            NamedBytesV1 {
                name: CONFIG.into(),
                bytes: read(CONFIG)?,
            },
            NamedBytesV1 {
                name: INDEX.into(),
                bytes: index,
            },
        ],
        ..Default::default()
    })
}

/// The checkpoint's binding metadata: every tensor of every file the weight
/// map names, checked against that map.
pub fn metadata(root: &Path) -> Result<CheckpointMetadata, String> {
    let config = document(root, CONFIG)?;
    let index = document(root, INDEX)?;
    let files = weight_files(&index)?;
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
