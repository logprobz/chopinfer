//! The package helper (`model_deepseek41::preparation`) and its draft
//! (`model_deepseek41::package`): routing by `config.json`, the serving
//! profile of each geometry and rank count, the plan request the helper
//! reads checked against the host's, the manifest of every topology, and
//! the capacity that leaves a rank count out.
//!
//! `the_helper_emits_the_synthetic_package` emits the package of the
//! synthetic checkpoint that `CHOPINFER_DEEPSEEK41_SYNTHETIC_CHECKPOINT`
//! names (its tokenizer is the pinned one; skipped when unset).
use std::path::Path;

use model_compiler_sdk::preparation::helper::Provider;
use model_compiler_sdk::{StagedContentV1, package_draft};
use model_deepseek41::package::{
    CAPABILITIES, RANK_IMAGE_COMPONENT, device_bytes, draft, profiles, slot_state_bytes,
    topology_id,
};
use model_deepseek41::preparation::{DeepSeek41Family, emit, select_profile, served_topologies};
use model_deepseek41_contract::geometry::{FLASH, GEOMETRIES, Geometry};
use model_deepseek41_contract::serving::{
    DECODE_BATCH, KV_LAYOUT_BF16, KV_LAYOUT_NATIVE, PREFILL_GRANULE, RANK_IMAGE_ROLE,
    REQUEST_STATE_ID, SCORED_ROWS,
};
use model_deepseek41_contract::state::{KvStorage, request_state};
use runtime_bundle::BundleKvDtypeV1;
use runtime_bundle::preparation::PrepareDirectRequest;
use runtime_host::direct::{DirectCheckpointV1, implementation_v1};

mod support;
use support::{config, sparse};

// The native fixture compile step every model shares, from the SDK's test
// support sources.
#[path = "../../../../crates/model-compiler-sdk/test-support/native.rs"]
mod native;

/// A native module that exports the bundle API symbol and runs nothing.
fn native_fixture() -> std::path::PathBuf {
    native::compile_native_fixture(
        "deepseek41-package-module",
        "deepseek41_package_fixture",
        "libbundle_deepseek41.so",
        r#"
#[unsafe(no_mangle)]
pub extern "C" fn chop_bundle_get_api_v1() -> *const core::ffi::c_void {
    core::ptr::null()
}
"#,
    )
}

/// A compiled geometry other than flash: the synthetic ones
/// `CHOPINFER_DEEPSEEK41_GEOMETRY_DIRS` adds.
fn synthetic() -> Option<&'static Geometry> {
    GEOMETRIES.iter().find(|geometry| geometry.id != FLASH.id)
}

/// A directory holding only `config`.
fn config_only(config: &serde_json::Value) -> std::path::PathBuf {
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let root = std::env::temp_dir().join(format!(
        "deepseek41-probe-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(
        root.join("config.json"),
        serde_json::to_vec(config).unwrap(),
    )
    .unwrap();
    root
}

#[test]
fn the_probe_routes_by_config_alone() {
    for geometry in GEOMETRIES {
        let root = config_only(&config(geometry));
        for ranks in [1, 2, 4] {
            let routed = DeepSeek41Family.probe(&root, ranks);
            let divides = model_deepseek41_contract::admission::check_ranks(geometry, ranks);
            assert_eq!(routed.is_ok(), divides.is_ok(), "{} tp{ranks}", geometry.id);
        }
        let error = DeepSeek41Family.probe(&root, 3).unwrap_err().to_string();
        assert!(error.contains("tensor-parallel size"), "{error}");
        std::fs::remove_dir_all(&root).unwrap();
    }
    let mut other = config(&FLASH);
    other["architectures"] = serde_json::json!(["Glm5NextForConditionalGeneration"]);
    let root = config_only(&other);
    let error = DeepSeek41Family.probe(&root, 4).unwrap_err().to_string();
    assert!(error.contains("unsupported architectures"), "{error}");
    std::fs::remove_dir_all(&root).unwrap();
}

/// The host's admission of a serving profile document: `profiles show`
/// validates every entry (evidence, devices, speculation) and parses it as
/// serve options. `document` is written to a temporary file.
fn host_admits(document: &str, name: &str) -> Result<(), String> {
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let path = std::env::temp_dir().join(format!(
        "deepseek41-profiles-{}-{}.json",
        std::process::id(),
        NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    ));
    std::fs::write(&path, document).unwrap();
    let arguments = ["profiles", "show", path.to_str().unwrap(), name].map(str::to_owned);
    let result = runtime_host::parse_host_command_v1(&arguments)
        .map(|_| ())
        .map_err(|e| e.to_string());
    std::fs::remove_file(&path).unwrap();
    result
}

/// Every geometry's package profiles pass the host's admission, and the
/// helper selects the package's own entry for the requested rank count.
#[test]
fn the_profile_follows_the_geometry_and_rank_count() {
    let flash = select_profile(&FLASH, 4).unwrap();
    assert_eq!(flash.name, "tp4-ar");
    assert_eq!(flash.profile["topology"], topology_id(&FLASH, 4));
    assert!(select_profile(&FLASH, 2).is_err());
    let document = profiles(&FLASH, &[4]).unwrap();
    let file: serde_json::Value = serde_json::from_str(&document).unwrap();
    assert_eq!(file["profiles"]["tp4-ar"], flash.profile);
    host_admits(&document, "tp4-ar").unwrap();
    assert!(profiles(&FLASH, &[2]).is_err());
    let Some(geometry) = synthetic() else {
        eprintln!("no synthetic geometry is compiled: flash only");
        return;
    };
    let document = profiles(geometry, &[1, 2, 4]).unwrap();
    let file: serde_json::Value = serde_json::from_str(&document).unwrap();
    let names: Vec<_> = file["profiles"]
        .as_object()
        .unwrap()
        .keys()
        .cloned()
        .collect();
    assert_eq!(names.len(), 3);
    for ranks in [1_u32, 2, 4] {
        let selected = select_profile(geometry, ranks).unwrap();
        assert_eq!(selected.name, format!("{}-tp{ranks}-ar", geometry.id));
        assert_eq!(file["profiles"][&selected.name], selected.profile);
        let p = &selected.profile;
        assert_eq!(p["topology"], topology_id(geometry, ranks));
        assert_eq!(p["tensor_parallel"], ranks);
        assert_eq!(p["devices_required"], ranks);
        assert_eq!(
            p["devices"],
            serde_json::json!((0..ranks).collect::<Vec<_>>())
        );
        assert_eq!(p["status"], "experimental");
        assert_eq!(p["qualified_max_sequences"], serde_json::Value::Null);
        assert_eq!(p["speculation"]["drafter"], "none");
        assert_eq!(
            p["evidence"],
            serde_json::json!(["models/deepseek41/README.md#checkpoint-and-geometry"])
        );
        host_admits(&document, &selected.name).unwrap();
    }
    // The admission route refuses what the host refuses: a profile with no
    // evidence.
    let mut empty = file.clone();
    let name = format!("{}-tp2-ar", geometry.id);
    empty["profiles"][&name]["evidence"] = serde_json::json!([]);
    let error = host_admits(&empty.to_string(), &name).unwrap_err();
    assert!(error.contains("qualification evidence"), "{error}");
}

/// The helper plans from the request the host later sends: the same files,
/// headers, identity and plan documents.
#[test]
fn the_plan_request_is_the_host_s() {
    let root = sparse(&FLASH);
    let host = DirectCheckpointV1::open(&root).unwrap();
    for (ranks, rank) in [(1, 0), (4, 3)] {
        let helper = model_deepseek41::checkpoint::request(&root, ranks, rank).unwrap();
        let host = host.request(
            RANK_IMAGE_ROLE,
            ranks,
            rank,
            implementation_v1([0; 32]),
            vec![],
            vec![],
        );
        assert_eq!(helper.image_role, host.image_role);
        assert_eq!((helper.ranks, helper.rank), (host.ranks, host.rank));
        assert_eq!(helper.checkpoint_identity, host.checkpoint_identity);
        assert_eq!(helper.files, host.files);
        for document in &helper.documents {
            assert!(
                host.documents.contains(document),
                "the host's request differs in {}",
                document.name
            );
        }
    }
    std::fs::remove_dir_all(&root).unwrap();
}

/// Each topology declares the native mixed-plane pages (890 bytes a token
/// at flash, 56,960 a page) with the native per-item state, and the BF16
/// pages with the BF16 state; the manifest requires the mixed-plane
/// capability.
#[test]
fn the_manifest_declares_each_topology_s_resources() {
    let geometry = &FLASH;
    let topologies = [(1, 3 << 30), (2, 2 << 30), (4, 1 << 30)];
    let components = vec![
        package_draft::tokenizer_definition(StagedContentV1::bytes(b"{}".to_vec())).unwrap(),
        package_draft::tokenizer_configuration(StagedContentV1::bytes(b"{}".to_vec())).unwrap(),
    ];
    let draft = draft(
        geometry,
        &topologies,
        components,
        StagedContentV1::bytes(b"module".to_vec()),
    )
    .unwrap();
    let manifest = model_compiler_sdk::plan_bundle_manifest_v1(&draft).unwrap();
    assert_eq!(manifest.id().as_str(), model_deepseek41::BUNDLE_ID);
    let declared: Vec<_> = manifest
        .target()
        .topologies()
        .iter()
        .map(|topology| (topology.id().as_str().to_owned(), topology.rank_count()))
        .collect();
    assert_eq!(
        declared,
        topologies
            .iter()
            .map(|&(ranks, _)| (topology_id(geometry, ranks), ranks))
            .collect::<Vec<_>>()
    );
    let planned = manifest
        .planned_components()
        .iter()
        .find(|component| component.id().as_str() == RANK_IMAGE_COMPONENT)
        .unwrap();
    assert_eq!(planned.role().as_str(), RANK_IMAGE_ROLE);
    let page_bytes = request_state(geometry, KvStorage::Native, false).page_bytes();
    for (ranks, weight_bytes) in topologies {
        let topology = manifest
            .for_topology(&topology_id(geometry, ranks))
            .unwrap();
        let requirements = topology.resource_requirements();
        assert_eq!(
            requirements.minimum_device_memory_bytes(),
            device_bytes(weight_bytes)
        );
        let capabilities: Vec<_> = requirements
            .required_capabilities()
            .iter()
            .map(|capability| capability.as_str())
            .collect();
        for capability in CAPABILITIES {
            assert!(capabilities.contains(&capability), "{capability}");
        }
        let kv = topology.kv_layout();
        assert_eq!(kv.id().as_str().as_bytes(), KV_LAYOUT_NATIVE);
        assert_eq!(kv.page_bytes(), page_bytes);
        assert_eq!(kv.maximum_sequence_tokens(), geometry.context_tokens());
        assert_eq!(kv.page_bytes(), 56_960);
        assert_eq!(kv.alignment_bytes(), 128);
        assert!(capabilities.contains(&"chop.kv-cache.mixed-planes.v1"));
        let state = topology.state_layout();
        assert_eq!(state.id().as_str(), REQUEST_STATE_ID);
        assert_eq!(state.bytes(), slot_state_bytes(geometry, KvStorage::Native));
        assert_eq!(state.bytes(), 2_970_624);
        let formats = topology.kv_formats();
        assert_eq!(
            formats.iter().map(|f| f.dtype()).collect::<Vec<_>>(),
            [BundleKvDtypeV1::Bf16, BundleKvDtypeV1::MixedPlanes]
        );
        assert_eq!(formats[1].layout(), kv);
        assert_eq!(formats[1].state_layout(), None);
        let bf16 = formats[0].layout();
        assert_eq!(bf16.id().as_str().as_bytes(), KV_LAYOUT_BF16);
        assert_eq!(
            bf16.page_bytes(),
            request_state(geometry, KvStorage::Bf16, false).page_bytes()
        );
        let bf16_state = formats[0].state_layout().unwrap();
        assert_eq!(bf16_state.id().as_str(), REQUEST_STATE_ID);
        assert_eq!(
            bf16_state.bytes(),
            slot_state_bytes(geometry, KvStorage::Bf16)
        );
        assert_eq!(bf16_state.bytes(), 5_668_864);
        let limits = topology.batch_limits().unwrap();
        assert_eq!(limits.maximum_sequences(), DECODE_BATCH);
        assert_eq!(limits.prefill_granule(), PREFILL_GRANULE);
        assert_eq!(limits.maximum_scored_rows(), SCORED_ROWS);
    }
}

/// Flash's weights fit a device at four ranks only: the package of its
/// checkpoint serves TP4, and a TP2 request is refused by capacity.
#[test]
fn flash_fits_at_four_ranks_alone() {
    let root = sparse(&FLASH);
    let served = served_topologies(&root, &FLASH, 4).unwrap();
    assert_eq!(
        served.iter().map(|&(ranks, _)| ranks).collect::<Vec<_>>(),
        [4]
    );
    eprintln!("flash tp4: {} weight bytes a rank", served[0].1);
    let error = served_topologies(&root, &FLASH, 2).unwrap_err().to_string();
    assert!(error.contains("does not fit at 2 ranks"), "{error}");
    std::fs::remove_dir_all(&root).unwrap();
    let Some(geometry) = synthetic() else {
        return;
    };
    let small = sparse(geometry);
    let served = served_topologies(&small, geometry, 2).unwrap();
    assert_eq!(
        served.iter().map(|&(ranks, _)| ranks).collect::<Vec<_>>(),
        [1, 2, 4]
    );
    std::fs::remove_dir_all(&small).unwrap();
}

#[test]
fn the_helper_emits_the_synthetic_package() {
    let Some(source) = std::env::var_os("CHOPINFER_DEEPSEEK41_SYNTHETIC_CHECKPOINT") else {
        eprintln!("CHOPINFER_DEEPSEEK41_SYNTHETIC_CHECKPOINT is unset: skipped");
        return;
    };
    let source = Path::new(&source);
    let config: serde_json::Value =
        serde_json::from_slice(&std::fs::read(source.join("config.json")).unwrap()).unwrap();
    let geometry = model_deepseek41_contract::admission::admit(&config).unwrap();
    let out = std::env::temp_dir().join(format!("deepseek41-emit-{}", std::process::id()));
    std::fs::create_dir_all(&out).unwrap();
    let module = native_fixture();
    let request = PrepareDirectRequest {
        path: source.to_path_buf(),
        cache_root: out.join("cache"),
        tensor_parallel_size: 2,
        activation_policy: None,
        modalities: Default::default(),
    };
    let emitted = emit(source, &request, &module, &out.join("package")).unwrap();
    assert_eq!(emitted.profile.name, format!("{}-tp2-ar", geometry.id));
    // The package carries the profiles of its own topologies, the selected
    // one among them, and the host admits it.
    let carried = std::fs::read_to_string(
        emitted
            .package
            .root()
            .join(runtime_bundle::deployment::SERVE_PROFILES_PATH_V1),
    )
    .unwrap();
    assert_eq!(carried, profiles(geometry, &[1, 2, 4]).unwrap());
    let file: serde_json::Value = serde_json::from_str(&carried).unwrap();
    assert_eq!(
        file["profiles"][&emitted.profile.name],
        emitted.profile.profile
    );
    host_admits(&carried, &emitted.profile.name).unwrap();
    let manifest = emitted.package.manifest();
    let declared: Vec<_> = manifest
        .target()
        .topologies()
        .iter()
        .map(|topology| topology.id().as_str().to_owned())
        .collect();
    assert_eq!(
        declared,
        [1, 2, 4].map(|ranks| topology_id(geometry, ranks))
    );
    for role in [
        "chop.tokenizer.definition.v1",
        "chop.tokenizer.configuration.v1",
    ] {
        assert!(
            manifest
                .components()
                .iter()
                .any(|component| component.role().as_str() == role),
            "{role}"
        );
    }
    assert!(
        manifest
            .planned_components()
            .iter()
            .any(|component| component.role().as_str() == RANK_IMAGE_ROLE)
    );
    let mut refused = request.clone();
    refused.activation_policy = Some("bf16".into());
    let Err(error) = emit(source, &refused, &module, &out.join("refused")) else {
        panic!("an explicit activation policy is refused");
    };
    let error = error.to_string();
    assert!(error.contains("--activation-policy"), "{error}");
    std::fs::remove_dir_all(&out).unwrap();
}
