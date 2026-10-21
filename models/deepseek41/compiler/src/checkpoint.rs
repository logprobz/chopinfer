//! The checkpoint as binding sees it: every safetensors header, the weight
//! map and `config.json`'s `quantization_config`, read without any tensor
//! payload. Files open as direct preparation opens them
//! (`runtime_weights::checkpoint::CheckpointFile`).
use model_compiler_sdk::quantization::projection::metadata::HeaderCompanionRules;
use model_deepseek41_contract::serving::RANK_IMAGE_ROLE;
use model_deepseek41_contract::{admission, encoder_plan, projections};
use projection_spec::bind::{BoundDeclaration, CheckpointMetadata};
use projection_spec::registry::Registry;
use runtime_bundle_abi::weight_plan::{CheckpointFileV1, NamedBytesV1, WeightPlanRequestV1};
use runtime_weights::checkpoint::CheckpointFile;
use serde_json::Value;
use std::collections::BTreeMap;
use std::os::unix::fs::FileExt;
use std::path::Path;

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

/// The files the weight map names, in name order.
fn weight_files(index: &Value) -> Result<std::collections::BTreeSet<String>, String> {
    index["weight_map"]
        .as_object()
        .ok_or("model.safetensors.index.json: no weight_map object")?
        .values()
        .map(|file| {
            file.as_str()
                .filter(|name| {
                    !name.contains(['/', '\\'])
                        && Path::new(name).components().count() == 1
                        && matches!(
                            Path::new(name).components().next(),
                            Some(std::path::Component::Normal(_))
                        )
                })
                .map(str::to_owned)
                .ok_or_else(|| "weight_map value is not one normal file name".to_owned())
        })
        .collect()
}

/// One preparation operation's checked headers and documents. Descriptors
/// keep their original identities until publication; no tensor payload is read.
/// Rank requests reuse the same allocation rather than reopening each shard.
pub struct Snapshot {
    request: WeightPlanRequestV1,
    inputs: Inputs,
    config: Value,
    metadata: CheckpointMetadata,
}

enum Inputs {
    Files(Vec<CheckpointFile>),
    Captured(std::sync::Arc<model_compiler_sdk::preparation::direct::CheckpointSnapshot>),
}

impl Snapshot {
    /// Standalone planning reads only its config, index and indexed shards.
    /// Unrelated directory entries do not become planning requirements.
    pub fn open(root: &Path) -> Result<Self, String> {
        let mut inputs = Vec::new();
        let mut document = |name| {
            let file = open(root, name)?;
            let bytes = file
                .read_bounded(MAX_DOCUMENT_BYTES)
                .map_err(|e| e.to_string())?;
            inputs.push(file);
            Ok::<_, String>(bytes)
        };
        let config = document(CONFIG)?;
        let index_bytes = document(INDEX)?;
        let index: Value = serde_json::from_slice(&index_bytes).map_err(|e| e.to_string())?;
        let mut files = Vec::new();
        for name in weight_files(&index)? {
            let file = open(root, &name)?;
            let byte_length = file.metadata().map_err(|e| e.to_string())?.len();
            let mut prefix = [0; 8];
            file.read_exact_at(&mut prefix, 0)
                .map_err(|e| e.to_string())?;
            let length = u64::from_le_bytes(prefix);
            if length == 0 || length > 256 << 20 || length > byte_length.saturating_sub(8) {
                return Err(format!(
                    "{name}: safetensors header length is outside the file"
                ));
            }
            let mut header = vec![0; length as usize];
            file.read_exact_at(&mut header, 8)
                .map_err(|e| e.to_string())?;
            file.verify().map_err(|e| e.to_string())?;
            inputs.push(file);
            files.push(CheckpointFileV1 {
                name,
                byte_length,
                data_offset: 8 + length,
                header,
            });
        }
        Self::from_parts(config, index_bytes, files, Inputs::Files(inputs))
    }

    /// Build rank requests from the SDK's already captured bytes. The
    /// provider and the cache key observe exactly the same snapshot.
    pub fn from_checkpoint(
        captured: std::sync::Arc<model_compiler_sdk::preparation::direct::CheckpointSnapshot>,
    ) -> Result<Self, String> {
        let document = |name| {
            captured
                .document(name)
                .map(Vec::from)
                .map_err(|e| e.to_string())
        };
        let config = document(CONFIG)?;
        let index_bytes = document(INDEX)?;
        let index: Value = serde_json::from_slice(&index_bytes).map_err(|e| e.to_string())?;
        let mut files = Vec::new();
        for name in weight_files(&index)? {
            let (byte_length, header) = captured
                .headers()
                .get(&name)
                .ok_or_else(|| format!("missing checkpoint header {name}"))?;
            files.push(CheckpointFileV1 {
                name,
                byte_length: *byte_length,
                data_offset: 8 + header.len() as u64,
                header: header.clone(),
            });
        }
        Self::from_parts(config, index_bytes, files, Inputs::Captured(captured))
    }

    fn from_parts(
        config_bytes: Vec<u8>,
        index_bytes: Vec<u8>,
        files: Vec<CheckpointFileV1>,
        inputs: Inputs,
    ) -> Result<Self, String> {
        let config: Value = serde_json::from_slice(&config_bytes).map_err(|e| e.to_string())?;
        let index: Value = serde_json::from_slice(&index_bytes).map_err(|e| e.to_string())?;
        let mut headers = BTreeMap::new();
        for file in &files {
            let mut framed = (file.header.len() as u64).to_le_bytes().to_vec();
            framed.extend_from_slice(&file.header);
            runtime_weights::parse_safetensors_header(&mut framed.as_slice(), file.byte_length)
                .map_err(|e| e.to_string())?;
            headers.insert(
                file.name.clone(),
                serde_json::from_slice(&file.header).map_err(|e| e.to_string())?,
            );
        }
        let metadata = CheckpointMetadata::from_headers(
            &headers,
            Some(&index),
            config
                .get("quantization_config")
                .cloned()
                .unwrap_or(Value::Null),
        )
        .map_err(|e| e.0)?;
        if metadata.tensors.len() != index["weight_map"].as_object().map_or(0, |map| map.len()) {
            return Err("the headers and the weight map name different tensors".into());
        }
        let checkpoint_identity = runtime_weights::checkpoint::header_identity(
            files
                .iter()
                .map(|file| (file.name.as_str(), file.byte_length, file.header.as_slice())),
        );
        let snapshot = Self {
            request: WeightPlanRequestV1 {
                image_role: RANK_IMAGE_ROLE.into(),
                ranks: 1,
                checkpoint_identity,
                files,
                documents: vec![
                    NamedBytesV1 {
                        name: CONFIG.into(),
                        bytes: config_bytes,
                    },
                    NamedBytesV1 {
                        name: INDEX.into(),
                        bytes: index_bytes,
                    },
                ],
                ..Default::default()
            },
            inputs,
            config,
            metadata,
        };
        snapshot.verify()?;
        Ok(snapshot)
    }

    pub fn config(&self) -> &Value {
        &self.config
    }

    pub fn metadata(&self) -> &CheckpointMetadata {
        &self.metadata
    }

    /// Retain additional metadata identities across a standalone operation,
    /// or require them from the provider's already captured snapshot.
    pub fn track_documents(&mut self, root: &Path, names: &[&str]) -> Result<(), String> {
        for name in names {
            match &mut self.inputs {
                Inputs::Files(files) => files.push(open(root, name)?),
                Inputs::Captured(captured) => {
                    captured.document(name).map_err(|e| e.to_string())?;
                }
            }
        }
        Ok(())
    }

    /// All rank counts use the same owned header and document storage.
    pub fn request(&mut self, ranks: u32, rank: u32) -> &WeightPlanRequestV1 {
        self.request.ranks = ranks;
        self.request.rank = rank;
        &self.request
    }

    pub fn verify(&self) -> Result<(), String> {
        match &self.inputs {
            Inputs::Files(files) => files
                .iter()
                .try_for_each(|file| file.verify().map_err(|e| e.to_string())),
            Inputs::Captured(captured) => captured.verify().map_err(|e| e.to_string()),
        }
    }
}

/// The text image's rank request, checked as the host checks its headers.
pub fn request(root: &Path, ranks: u32, rank: u32) -> Result<WeightPlanRequestV1, String> {
    let mut snapshot = Snapshot::open(root)?;
    snapshot.request(ranks, rank);
    Ok(snapshot.request)
}

/// The binding metadata of every tensor, checked against the weight map.
pub fn metadata(root: &Path) -> Result<CheckpointMetadata, String> {
    Ok(Snapshot::open(root)?.metadata)
}

/// Admit the checkpoint's config, then bind every tensor its headers name
/// to the projection declaration ([`bind_metadata`]). `encoder` holds the
/// vision and aligner tensors the encoder declaration validated
/// (`encoder_plan::owned`); the declaration names them as that owner's
/// (`external_tensors`).
pub fn bind(
    root: &Path,
    encoder: std::collections::BTreeSet<String>,
) -> Result<BoundDeclaration, String> {
    let mut snapshot = Snapshot::open(root)?;
    snapshot
        .metadata
        .validated_owners
        .insert(encoder_plan::OWNER.into(), encoder);
    let bound = bind_metadata(snapshot.config(), snapshot.metadata())?;
    snapshot.verify()?;
    Ok(bound)
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
