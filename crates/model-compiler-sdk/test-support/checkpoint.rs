//! Checkpoint header facts for projection tests: JSON header fixtures, and
//! the headers of a checkpoint on disk. No tensor payload is read.
//!
//! Test support source, outside model-compiler-sdk's module tree: a model's
//! tests include it by `#[path]` and supply its dependencies, so no build
//! of the SDK, a host, a bundle or a package reads it. Each including test
//! uses part of it.
#![allow(dead_code)]

use std::collections::BTreeMap;
use std::path::Path;

use projection_spec::bind::{CheckpointMetadata, TensorInfo};
use serde::Deserialize;
use serde_json::Value;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Fixture {
    families: Vec<Family>,
    quantization_config: Value,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Family {
    tensor: String,
    dtype: String,
    shape: Vec<u64>,
    axes: BTreeMap<String, Axis>,
}
#[derive(Deserialize)]
#[serde(untagged)]
enum Axis {
    Range { range: [u64; 2] },
    List { indices: Vec<u64> },
}

/// The header facts of the JSON fixture `text`: each family's tensor name
/// with every axis token replaced by each of its indices (a `range` or an
/// `indices` list), its dtype and shape, and the fixture's
/// quantization_config, with no validated owners. Panics on a malformed
/// fixture or a name two families give.
pub fn fixture(text: &str) -> CheckpointMetadata {
    let f: Fixture = serde_json::from_str(text).unwrap();
    let mut tensors = BTreeMap::new();
    for family in f.families {
        let mut names = vec![family.tensor];
        for (token, axis) in family.axes {
            let indices: Vec<_> = match axis {
                Axis::Range {
                    range: [start, end],
                } => (start..end).collect(),
                Axis::List { indices } => indices,
            };
            let token = token.as_str();
            names = names
                .into_iter()
                .flat_map(|name| {
                    indices
                        .iter()
                        .map(move |i| name.replace(token, &i.to_string()))
                })
                .collect();
        }
        for name in names {
            assert!(
                tensors
                    .insert(
                        name,
                        TensorInfo {
                            dtype: family.dtype.clone(),
                            shape: family.shape.clone(),
                            file: None
                        }
                    )
                    .is_none()
            );
        }
    }
    CheckpointMetadata {
        tensors,
        quantization_config: f.quantization_config,
        validated_owners: BTreeMap::new(),
    }
}

/// The safetensors header JSON of the file `name` under `root`.
pub fn read_header(root: &Path, name: &str) -> Value {
    // The confined checkpoint reader enforces the existing no-follow policy.
    let file = runtime_weights::checkpoint::CheckpointFile::open(root, name).unwrap();
    let header = runtime_weights::read_safetensors_header(&file).unwrap();
    let mut bytes = vec![0; header.header_bytes() as usize];
    runtime_weights::read_exact_at(&file, &mut bytes, 8).unwrap();
    serde_json::from_slice(&bytes).unwrap()
}

/// That `actual` holds the same tensors as `expected`, each with the same
/// dtype and shape, and the same quantization_config.
pub fn assert_header_facts(actual: &CheckpointMetadata, expected: &CheckpointMetadata) {
    assert_eq!(actual.tensors.len(), expected.tensors.len());
    for (name, source) in &actual.tensors {
        let saved = &expected.tensors[name];
        assert_eq!(source.dtype, saved.dtype, "{name}");
        assert_eq!(source.shape, saved.shape, "{name}");
    }
    assert_eq!(actual.quantization_config, expected.quantization_config);
}

/// The header facts of the checkpoint at `root`: every file its
/// `model.safetensors.index.json` names, and config.json's
/// quantization_config (with the optional hf_quant_config.json captured).
pub fn checkpoint_headers(root: &Path) -> CheckpointMetadata {
    let captured = model_compiler_sdk::source::SourceFiles::capture(
        root,
        &[
            ("config.json", 1024 * 1024),
            ("model.safetensors.index.json", 64 * 1024 * 1024),
        ],
        &[("hf_quant_config.json", 16 * 1024 * 1024)],
        "model.safetensors.index.json",
    )
    .unwrap();
    let config: Value = serde_json::from_slice(&captured.inputs["config.json"]).unwrap();
    let index: Value =
        serde_json::from_slice(&captured.inputs["model.safetensors.index.json"]).unwrap();
    let headers = captured
        .headers
        .keys()
        .map(|name| (name.clone(), read_header(root, name)))
        .collect();
    CheckpointMetadata::from_headers(
        &headers,
        Some(&index),
        config["quantization_config"].clone(),
    )
    .unwrap()
}
