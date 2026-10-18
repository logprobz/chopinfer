//! Mixed-precision checkpoints of the flash geometry. Each mixed
//! representative layout's synthetic config admits its classes and its
//! catalog binds the projection declaration in the layout's formats. With
//! `CHOPINFER_GLM5NEXT_EXTRA_CHECKPOINTS` naming checkpoints (`:`
//! separated), every shard header of each matches the catalog of the
//! layout its config and headers resolve to, and binds. Expectations come
//! from that layout, not from any export's pinned files. Only headers are
//! read, never tensor data.
use model_compiler_sdk::quantization::import::{EncodingRecipe, ImportOptions, KvHintPolicy};
use model_compiler_sdk::quantization::projection::metadata::{HeaderCompanionRules, HeaderRole};
use model_glm5next_contract::catalog::{self, Group, TensorSpec};
use model_glm5next_contract::geometry::FLASH;
use model_glm5next_contract::quantization::{
    self, EXTRA_CHECKPOINTS, Encoding, FP8_NVFP4_ROUTED, Layout, NVFP4_DENSE_ROUTED, NVFP4_ROUTED,
    synthetic,
};
use model_glm5next_contract::{admission, projections, weight_plan};
use projection_spec::bind::{BoundDeclaration, CheckpointMetadata, TensorInfo};
use projection_spec::profiles::UseProfileRef;
use projection_spec::{Declaration, Error, Geometry, ProjectionKind, TensorSource};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::io::Read;
use std::path::Path;

use kernel_core::precision::format::FormatId;

const MAX_HEADER: u64 = 64 << 20;

/// The mixed representative layouts.
const MIXES: [&Layout; 3] = [&NVFP4_ROUTED, &NVFP4_DENSE_ROUTED, &FP8_NVFP4_ROUTED];

/// The bound format of projection `id` in `layout`: its class's encoding
/// for a projection of a class's use profiles, else BF16.
fn format(declaration: &Declaration, layout: &Layout, id: &str) -> FormatId {
    let projection = declaration
        .projections
        .iter()
        .find(|p| p.id == id)
        .unwrap_or_else(|| panic!("no projection {id}"));
    let profile = match &projection.use_profile {
        Some(UseProfileRef::Id(id)) => Some(id.as_str()),
        Some(UseProfileRef::Overrides(overrides)) => Some(overrides.profile.as_str()),
        None => None,
    };
    let encoding = match profile {
        Some("fp8" | "fp8-f32") => layout.projections,
        Some("dense-mlp" | "dense-mlp-f32") => layout.dense,
        Some("experts") => layout.routed,
        Some("mtp-experts") => layout.mtp_routed,
        _ => Encoding::Bf16,
    };
    match encoding {
        Encoding::Fp8Block128 => FormatId::Fp8Block128,
        Encoding::Bf16 => FormatId::Bf16,
        Encoding::Nvfp4 => FormatId::Nvfp4,
        Encoding::Mxfp8 => FormatId::Mxfp8,
    }
}

/// Vision tensors belong to the encoder declaration; the catalog's vision
/// group stands in for its attestation.
fn attest_vision(layout: &Layout, m: &mut CheckpointMetadata) {
    let vision = catalog::checkpoint_catalog(&FLASH, layout)
        .into_iter()
        .filter(|t| t.group == Group::Vision && m.tensors.contains_key(&t.name))
        .map(|t| t.name)
        .collect();
    m.validated_owners.insert("encoder-spec".into(), vision);
}

/// Every projection binds in its class's format in `layout`.
fn assert_formats(layout: &Layout, b: &BoundDeclaration) {
    let declaration = projections::declaration(4).unwrap();
    let mut formats = BTreeMap::<&str, BTreeSet<FormatId>>::new();
    for o in &b.operands {
        formats.entry(&o.projection).or_default().insert(o.format);
    }
    assert_eq!(formats.len(), 38);
    for (id, bound) in formats {
        assert_eq!(
            bound,
            BTreeSet::from([format(&declaration, layout, id)]),
            "{} {id}",
            layout.id
        );
    }
}

/// The `quantization_config` of `config`.
fn quantization_config(config: &[u8]) -> Value {
    let config: Value = serde_json::from_slice(config).unwrap();
    config["quantization_config"].clone()
}

#[test]
fn each_mixed_config_admits_its_classes() {
    for layout in MIXES {
        let matched = admission::admit_bytes(&synthetic::config_of(layout)).unwrap();
        assert_eq!(matched.geometry.id, "flash");
        assert_eq!(
            matched.quantization.classes,
            layout.classes(),
            "{}",
            layout.id
        );
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

/// A projection whose every weight is BF16 is an explicit BF16 role: a
/// config's exclusions need not name every BF16 projection.
fn rules(d: &Declaration, g: &Geometry, m: &CheckpointMetadata) -> HeaderCompanionRules {
    // glm5next records and ignores a static FP8 KV cache hint.
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
fn each_mixed_catalog_binds_the_projection_declaration() {
    for layout in MIXES {
        let mut m = CheckpointMetadata {
            tensors: catalog::checkpoint_catalog(&FLASH, layout)
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
            quantization_config: quantization_config(&synthetic::config_of(layout)),
            validated_owners: BTreeMap::new(),
        };
        attest_vision(layout, &mut m);
        assert_formats(
            layout,
            &bind(&m).unwrap_or_else(|e| panic!("{}: {}", layout.id, e.0)),
        );
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

/// The checkpoint at `root`: every header tensor is its resolved layout's
/// catalog entry (dtype, shape and bytes), the index states their total,
/// every rank plans the same bytes, and the headers bind the declaration in
/// the layout's formats.
fn check_headers(root: &Path) {
    let at = root.display();
    let config = std::fs::read(root.join("config.json")).unwrap();
    let matched = admission::admit_bytes(&config).unwrap_or_else(|e| panic!("{at}: {e}"));
    let index: Value =
        serde_json::from_slice(&std::fs::read(root.join("model.safetensors.index.json")).unwrap())
            .unwrap();
    let weight_map = index["weight_map"].as_object().unwrap();
    let shards = weight_map
        .values()
        .map(|file| file.as_str().unwrap().to_owned())
        .collect::<BTreeSet<_>>();
    let mut headers = BTreeMap::new();
    let mut observed = BTreeMap::new();
    for shard in shards {
        let mut header = header(&root.join(&shard));
        header.remove("__metadata__");
        for (name, entry) in &header {
            assert_eq!(
                weight_map[name],
                shard.as_str(),
                "{at}: {name} indexed in another shard"
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
    assert_eq!(observed.len(), weight_map.len(), "{at}");
    let layout = quantization::resolve(matched.geometry, &matched.quantization, |tensor| {
        observed
            .get(tensor)
            .map(|(dtype, _, _)| dtype.clone())
            .ok_or_else(|| format!("{tensor} is not in the headers"))
    })
    .unwrap_or_else(|e| panic!("{at}: {e}"));
    eprintln!("{at}: layout {}", layout.id);
    let tensors = catalog::checkpoint_catalog(matched.geometry, layout);
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
        "{at}: missing {missing:?}; extra {extra:?}"
    );
    for (name, entry) in &expected {
        assert_eq!(&observed[name], entry, "{at}: {name}");
    }
    let total = observed.values().map(|(_, _, bytes)| bytes).sum::<u64>();
    let catalog_total = tensors.iter().map(TensorSpec::bytes).sum::<u64>();
    assert_eq!(total, catalog_total, "{at}");
    assert_eq!(index["metadata"]["total_size"], total, "{at}");
    let ranks = weight_plan::plan(matched.geometry, &tensors, 4).unwrap();
    let rank_bytes = ranks[0].bytes.total();
    assert!(ranks.iter().all(|r| r.bytes.total() == rank_bytes), "{at}");

    let mut m =
        CheckpointMetadata::from_headers(&headers, Some(&index), quantization_config(&config))
            .unwrap();
    assert_eq!(m.tensors.len(), tensors.len(), "{at}");
    attest_vision(layout, &mut m);
    assert_formats(
        layout,
        &bind(&m).unwrap_or_else(|e| panic!("{at}: {}", e.0)),
    );
}

#[test]
fn extra_checkpoint_headers_match_their_layout_catalog_and_bind() {
    let Some(paths) = std::env::var_os(EXTRA_CHECKPOINTS) else {
        eprintln!("skipped: {EXTRA_CHECKPOINTS} is not set");
        return;
    };
    for root in std::env::split_paths(&paths).filter(|p| !p.as_os_str().is_empty()) {
        check_headers(&root);
    }
}
