//! The NVFP4 exports as checkpoints of the flash geometry: local-inference-lab's
//! (`checkpoint-files-nvfp4.json`) and NVIDIA's
//! (`checkpoint-files-nvidia-nvfp4.json`). Each config admits its layout, the
//! layout's catalog binds the projection declaration, and with the export's
//! checkpoint variable set every shard header matches the catalog and binds.
//! Only headers are read, never tensor data.
use model_compiler_sdk::quantization::import::{EncodingRecipe, ImportOptions, KvHintPolicy};
use model_compiler_sdk::quantization::projection::metadata::{HeaderCompanionRules, HeaderRole};
use model_glm5next_contract::catalog::{self, Group, TensorSpec};
use model_glm5next_contract::geometry::FLASH;
use model_glm5next_contract::quantization::{Layout, NVFP4_DENSE_ROUTED, NVFP4_ROUTED};
use model_glm5next_contract::{admission, projections, weight_plan};
use projection_spec::bind::{BoundDeclaration, CheckpointMetadata, TensorInfo};
use projection_spec::{Declaration, Error, Geometry, ProjectionKind, TensorSource};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::io::Read;
use std::path::Path;

use kernel_core::precision::format::FormatId;

const MAX_HEADER: u64 = 64 << 20;

/// One export and what its headers hold.
struct Export {
    layout: &'static Layout,
    /// The `config.json` the always-on tests admit: the export's own bytes,
    /// or a synthetic config of the same layout (`synthetic`).
    config: &'static [u8],
    /// Whether `config` is synthetic rather than the export's file.
    synthetic: bool,
    lock: &'static str,
    variable: &'static str,
    shards: usize,
    tensors: usize,
    bytes: u64,
    rank_bytes: u64,
    /// The bound format of a projection ID.
    format: fn(&str) -> FormatId,
}

/// The tree holds no copy of the export's `config.json`: the fixture is the FP8 checkpoint's config (less
/// its `transformers_version`) with the NVFP4-routed layout's ModelOpt
/// `quantization_config` (NVFP4 routed experts in layers 3 to 44, MXFP8 MTP
/// experts in layer 45). With the checkpoint present, `check_headers` holds
/// every field of it to the export's config and that config to the pin.
const LOCAL_INFERENCE_LAB: Export = Export {
    layout: &NVFP4_ROUTED,
    config: include_bytes!("../fixtures/nvfp4-config.json"),
    synthetic: true,
    lock: include_str!("../../checkpoint-files-nvfp4.json"),
    variable: "CHOPINFER_GLM5NEXT_NVFP4_CHECKPOINT",
    shards: 44,
    tensors: 148_498,
    bytes: 198_042_331_512,
    rank_bytes: 50_875_242_744,
    format: |id| {
        if id.starts_with("mtp.moe.experts.") {
            FormatId::Mxfp8
        } else if id.starts_with("moe.experts.") {
            FormatId::Nvfp4
        } else {
            FormatId::Bf16
        }
    },
};

const NVIDIA: Export = Export {
    layout: &NVFP4_DENSE_ROUTED,
    config: include_bytes!("../fixtures/nvidia-nvfp4-config.json"),
    synthetic: false,
    lock: include_str!("../../checkpoint-files-nvidia-nvfp4.json"),
    variable: "CHOPINFER_GLM5NEXT_NVIDIA_NVFP4_CHECKPOINT",
    shards: 33,
    tensors: 147_661,
    bytes: 204_419_110_596,
    rank_bytes: 52_469_433_888,
    format: |id| {
        if id.starts_with("moe.experts.") || id.starts_with("dense.") {
            FormatId::Nvfp4
        } else {
            FormatId::Bf16
        }
    },
};

impl Export {
    fn tensors(&self) -> Vec<TensorSpec> {
        catalog::checkpoint_catalog(&FLASH, self.layout)
    }

    /// The config's `quantization_config`.
    fn quantization_config(&self) -> Value {
        let config: Value = serde_json::from_slice(self.config).unwrap();
        config["quantization_config"].clone()
    }

    /// Vision tensors belong to the encoder declaration; the catalog's
    /// vision group stands in for its attestation.
    fn attest_vision(&self, m: &mut CheckpointMetadata) {
        let vision = self
            .tensors()
            .into_iter()
            .filter(|t| t.group == Group::Vision && m.tensors.contains_key(&t.name))
            .map(|t| t.name)
            .collect();
        m.validated_owners.insert("encoder-spec".into(), vision);
    }

    fn assert_formats(&self, b: &BoundDeclaration) {
        let mut formats = BTreeMap::<&str, BTreeSet<FormatId>>::new();
        for o in &b.operands {
            formats.entry(&o.projection).or_default().insert(o.format);
        }
        assert_eq!(formats.len(), 38);
        for (id, format) in formats {
            assert_eq!(format, BTreeSet::from([(self.format)(id)]), "{id}");
        }
    }
}

/// `bytes`' SHA-256 and length equal the pinned `config.json` entry.
fn assert_pinned_config(lock: &str, bytes: &[u8]) {
    use sha2::{Digest, Sha256};
    let lock: Value = serde_json::from_str(lock).unwrap();
    let digest = Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect::<String>();
    assert_eq!(lock["config.json"]["sha256"], digest.as_str());
    assert_eq!(lock["config.json"]["byte_length"], bytes.len());
}

/// Every field `synthetic` carries, at any depth, is present in `real` with
/// the same value, so the synthetic fixture cannot drift from the export.
fn assert_fields_match(synthetic: &Value, real: &Value, path: &str) {
    match synthetic {
        Value::Object(fields) => {
            let real = real
                .as_object()
                .unwrap_or_else(|| panic!("{path}: the export holds no object"));
            for (key, value) in fields {
                let at = format!("{path}/{key}");
                let other = real
                    .get(key)
                    .unwrap_or_else(|| panic!("{at}: absent from the export's config.json"));
                assert_fields_match(value, other, &at);
            }
        }
        _ => assert_eq!(
            synthetic, real,
            "{path}: differs from the export's config.json"
        ),
    }
}

#[test]
fn the_config_fixtures_are_the_locked_export_files_or_synthetic() {
    for export in [LOCAL_INFERENCE_LAB, NVIDIA] {
        let lock: Value = serde_json::from_str(export.lock).unwrap();
        if export.synthetic {
            // The pin names the export's own file, which is not in the tree.
            assert!(lock["config.json"]["sha256"].is_string());
        } else {
            assert_pinned_config(export.lock, export.config);
        }
        let shards = lock
            .as_object()
            .unwrap()
            .keys()
            .filter(|name| name.ends_with(".safetensors"))
            .count();
        assert_eq!(shards, export.shards);
        let matched = admission::admit_bytes(export.config).unwrap();
        assert_eq!(matched.quantization.layout, export.layout);
    }
}

fn expand(d: &Declaration, g: &Geometry, source: &TensorSource) -> Vec<String> {
    let Some(id) = &source.stem else {
        return vec![source.tensor.clone()];
    };
    let stem = d.layer_stems.iter().find(|s| s.id == *id).unwrap();
    stem.indices
        .resolve(g)
        .unwrap()
        .into_iter()
        .map(|i| {
            let prefix = stem.template.replace("{i}", &i.to_string());
            source.tensor.replace("{stem}", &prefix)
        })
        .collect()
}

/// A projection whose every weight is BF16 is an explicit BF16 role: the
/// exports' exclusions do not name every BF16 projection.
fn rules(d: &Declaration, g: &Geometry, m: &CheckpointMetadata) -> HeaderCompanionRules {
    // glm5next records and ignores the export's KV cache hint.
    let options = ImportOptions {
        kv_hint: KvHintPolicy::RecordAndIgnore,
    };
    let mut rules = HeaderCompanionRules::new_with(m, None, &options).unwrap();
    for p in &d.projections {
        let source = p.source.as_ref().unwrap();
        let mut names = expand(d, g, source);
        if !p.operands.is_empty() {
            names = names
                .into_iter()
                .flat_map(|n| p.operands.iter().map(move |o| n.replace("{operand}", o)))
                .collect();
        }
        if let ProjectionKind::ExpertBank { .. } = p.kind {
            // Expert banks share one encoding; expert 0 stands for the bank.
            names = names
                .into_iter()
                .map(|n| n.replace("{expert}", "0"))
                .collect();
        }
        if names
            .iter()
            .all(|n| m.tensors.get(n).is_some_and(|t| t.dtype == "BF16"))
        {
            rules.roles.insert(
                p.id.clone(),
                HeaderRole {
                    declared: Some(EncodingRecipe::bf16()),
                    ..Default::default()
                },
            );
        }
    }
    rules
}

fn bind(m: &CheckpointMetadata) -> Result<BoundDeclaration, Error> {
    let d = projections::declaration(4)?;
    let g = projections::geometry(4)?;
    d.bind(&g, &projections::registry()?, m, &rules(&d, &g, m))
}

#[test]
fn each_nvfp4_catalog_binds_the_projection_declaration() {
    for export in [LOCAL_INFERENCE_LAB, NVIDIA] {
        let mut m = CheckpointMetadata {
            tensors: export
                .tensors()
                .into_iter()
                .map(|t| {
                    let info = TensorInfo {
                        dtype: t.dtype.name().into(),
                        shape: t.shape,
                        file: None,
                    };
                    (t.name, info)
                })
                .collect(),
            quantization_config: export.quantization_config(),
            validated_owners: BTreeMap::new(),
        };
        export.attest_vision(&mut m);
        export
            .assert_formats(&bind(&m).unwrap_or_else(|e| panic!("{}: {}", export.layout.id, e.0)));
    }
}

fn header(path: &Path) -> serde_json::Map<String, Value> {
    let mut file = std::fs::File::open(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let mut length = [0; 8];
    file.read_exact(&mut length).unwrap();
    let length = u64::from_le_bytes(length);
    assert!(
        length <= MAX_HEADER,
        "{}: header of {length} bytes",
        path.display()
    );
    let mut bytes = vec![0; length as usize];
    file.read_exact(&mut bytes).unwrap();
    let Value::Object(header) = serde_json::from_slice(&bytes).unwrap() else {
        panic!("{}: header is not an object", path.display());
    };
    header
}

fn check_headers(export: &Export) {
    let Some(root) = std::env::var_os(export.variable) else {
        eprintln!("skipped: {} is not set", export.variable);
        return;
    };
    let root = Path::new(&root);
    let config = std::fs::read(root.join("config.json")).unwrap();
    assert_pinned_config(export.lock, &config);
    if export.synthetic {
        let synthetic: Value = serde_json::from_slice(export.config).unwrap();
        let real: Value = serde_json::from_slice(&config).unwrap();
        assert_fields_match(&synthetic, &real, "config.json");
        let fixture = admission::admit_bytes(export.config).unwrap();
        let matched = admission::admit_bytes(&config).unwrap();
        assert_eq!(fixture.geometry.id, matched.geometry.id);
        assert_eq!(fixture.quantization.layout, matched.quantization.layout);
    } else {
        assert_eq!(
            config, export.config,
            "config.json differs from the fixture"
        );
    }
    let matched = admission::admit_bytes(&config).unwrap();
    assert_eq!(matched.quantization.layout, export.layout);
    let index: Value =
        serde_json::from_slice(&std::fs::read(root.join("model.safetensors.index.json")).unwrap())
            .unwrap();
    let weight_map = index["weight_map"].as_object().unwrap();
    let shards = weight_map
        .values()
        .map(|file| file.as_str().unwrap().to_owned())
        .collect::<BTreeSet<_>>();
    assert_eq!(shards.len(), export.shards);
    let mut headers = BTreeMap::new();
    let mut observed = BTreeMap::new();
    for shard in shards {
        let mut header = header(&root.join(&shard));
        header.remove("__metadata__");
        for (name, entry) in &header {
            assert_eq!(
                weight_map[name],
                shard.as_str(),
                "{name}: indexed in another shard"
            );
            let offsets = entry["data_offsets"].as_array().unwrap();
            let bytes = offsets[1].as_u64().unwrap() - offsets[0].as_u64().unwrap();
            let shape = entry["shape"]
                .as_array()
                .unwrap()
                .iter()
                .map(|d| d.as_u64().unwrap())
                .collect::<Vec<_>>();
            let dtype = entry["dtype"].as_str().unwrap().to_owned();
            assert!(
                observed
                    .insert(name.clone(), (dtype, shape, bytes))
                    .is_none()
            );
        }
        headers.insert(shard, Value::Object(header));
    }
    assert_eq!(observed.len(), weight_map.len());
    let tensors = catalog::checkpoint_catalog(matched.geometry, matched.quantization.layout);
    let expected = tensors
        .iter()
        .map(|t| {
            (
                t.name.clone(),
                (t.dtype.name().to_owned(), t.shape.clone(), t.bytes()),
            )
        })
        .collect::<BTreeMap<_, _>>();
    let missing = expected
        .keys()
        .filter(|name| !observed.contains_key(*name))
        .collect::<Vec<_>>();
    let extra = observed
        .keys()
        .filter(|name| !expected.contains_key(*name))
        .collect::<Vec<_>>();
    assert!(
        missing.is_empty() && extra.is_empty(),
        "missing {missing:?}; extra {extra:?}"
    );
    for (name, entry) in &expected {
        assert_eq!(&observed[name], entry, "{name}");
    }
    let total = observed.values().map(|(_, _, bytes)| bytes).sum::<u64>();
    assert_eq!((observed.len(), total), (export.tensors, export.bytes));
    assert_eq!(index["metadata"]["total_size"], total);
    let ranks = weight_plan::plan(matched.geometry, &tensors, 4).unwrap();
    assert!(ranks.iter().all(|r| r.bytes.total() == export.rank_bytes));

    let mut m =
        CheckpointMetadata::from_headers(&headers, Some(&index), export.quantization_config())
            .unwrap();
    assert_eq!(m.tensors.len(), tensors.len());
    export.attest_vision(&mut m);
    export.assert_formats(&bind(&m).unwrap());
}

#[test]
fn the_local_inference_lab_headers_match_the_catalog_and_bind() {
    check_headers(&LOCAL_INFERENCE_LAB);
}

#[test]
fn the_nvidia_headers_match_the_catalog_and_bind() {
    check_headers(&NVIDIA);
}
