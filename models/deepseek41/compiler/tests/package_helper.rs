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
use model_deepseek41_contract::weight_plan;
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
    let document = profiles(&FLASH, &[4]).unwrap();
    let file: serde_json::Value = serde_json::from_str(&document).unwrap();
    assert_eq!(
        file,
        serde_json::from_str::<serde_json::Value>(model_deepseek41::package::PROFILES).unwrap()
    );
    for name in ["tp4-ar", "tp4-dspark"] {
        let profile = &file["profiles"][name];
        assert_eq!(profile["prefill_chunk_tokens"], 8192);
        assert_eq!(profile["max_batch_tokens"], 8192);
        assert_eq!(profile["prefix_cache"], true);
    }
    let name = if file["profiles"].get("tp4-dspark").is_some() {
        "tp4-dspark"
    } else {
        "tp4-ar"
    };
    let flash = select_profile(&FLASH, 4).unwrap();
    assert_eq!(flash.name, name);
    assert_eq!(flash.profile["topology"], topology_id(&FLASH, 4));
    assert_eq!(flash.profile["prefix_cache"], true);
    assert!(select_profile(&FLASH, 2).is_err());
    assert_eq!(file["profiles"][name], flash.profile);
    host_admits(&document, name).unwrap();
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

/// Flash's topologies in the manifest tests: each rank count and the most
/// weight bytes one of its ranks holds.
const FLASH_TOPOLOGIES: [(u32, u64); 3] = [(1, 3 << 30), (2, 2 << 30), (4, 1 << 30)];

/// The manifest of a flash package of [`FLASH_TOPOLOGIES`].
fn flash_manifest() -> runtime_bundle::BundleManifestV1 {
    let components = vec![
        package_draft::tokenizer_definition(StagedContentV1::bytes(b"{}".to_vec())).unwrap(),
        package_draft::tokenizer_configuration(StagedContentV1::bytes(b"{}".to_vec())).unwrap(),
    ];
    let draft = draft(
        &FLASH,
        FLASH.runtime_defaults(),
        &FLASH_TOPOLOGIES,
        components,
        StagedContentV1::bytes(b"module".to_vec()),
    )
    .unwrap();
    model_compiler_sdk::plan_bundle_manifest_v1(&draft).unwrap()
}

#[test]
fn serve_lists_flash_checkpoint_options_without_loading_weights() {
    let module = native_fixture();
    let root = config_only(&config(&FLASH));
    for (top_k, blocks) in [(512, 2048), (256, 2048), (256, 1024)] {
        let components = vec![
            package_draft::tokenizer_definition(StagedContentV1::bytes(b"{}".to_vec())).unwrap(),
            package_draft::tokenizer_configuration(StagedContentV1::bytes(b"{}".to_vec())).unwrap(),
        ];
        let mut checkpoint = config(&FLASH);
        checkpoint["text_config"]["index_topk"] = top_k.into();
        checkpoint["text_config"]["candidate_topk_blocks"] = blocks.into();
        let runtime = FLASH.check_runtime(&checkpoint).unwrap();
        let draft = draft(
            &FLASH,
            runtime,
            &FLASH_TOPOLOGIES,
            components,
            StagedContentV1::file(&module),
        )
        .unwrap();
        let package = model_compiler_sdk::emit_bundle_directory_v1(
            &draft,
            &root.join(format!("package-{top_k}-{blocks}")),
        )
        .unwrap();
        let args = [
            "serve",
            "--bundle",
            package.root().to_str().unwrap(),
            "--topology",
            "deepseek41-flash-tp4",
            "--list-model-options",
            "--fake-host-services",
        ]
        .map(str::to_owned);
        let command = runtime_host::parse_host_command_v1(&args).unwrap();
        let output = runtime_host::run_host_command_v1(&command).unwrap();
        let options: serde_json::Value = serde_json::from_str(&output).unwrap();
        assert_eq!(
            options,
            serde_json::json!({
                "schema": "chop.model-options.v1",
                "options": [{"name": "text_config.candidate_topk_blocks", "default_is_qualified": blocks == 2048,
                         "definition": {"type": "integer", "default": blocks, "minimum": 1, "maximum": 2048}},
                        {"name": "text_config.index_topk", "default_is_qualified": top_k == 512,
                             "definition": {"type": "integer", "default": top_k, "minimum": 1, "maximum": 512}},
                        {"name": "text_config.indexer_mma", "default_is_qualified": true,
                             "definition": {"type": "boolean", "default": false}}]
            })
        );
        // The serving host resolves the complete declaration before encoding
        // create, including checkpoint defaults absent from the overrides.
        for override_top_k in [None, Some(128)] {
            let overrides = override_top_k
                .map(|k| ("text_config.index_topk".to_owned(), k.to_string()))
                .into_iter()
                .collect();
            let resolved = package
                .manifest()
                .resolve_model_options(&overrides)
                .unwrap();
            let expected = serde_json::json!({
                "text_config.index_topk": override_top_k.unwrap_or(top_k),
                "text_config.candidate_topk_blocks": blocks,
                "text_config.indexer_mma": false,
            });
            assert_eq!(serde_json::to_value(&resolved).unwrap(), expected);
            let host_config = runtime_host::HostInstanceConfigV1::new(
                runtime_host::HostTopologyV1::new(vec![0, 1, 2, 3]).unwrap(),
            )
            .with_model_options(resolved);
            let bytes = host_config.encode_v1(b"deepseek41-flash-tp4");
            let tag = runtime_bundle_abi::MODEL_OPTIONS_TAG_V1;
            let at = bytes.windows(tag.len()).position(|w| w == tag).unwrap() + tag.len();
            let length = u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap()) as usize;
            let tail: serde_json::Value = serde_json::from_slice(&bytes[at + 4..]).unwrap();
            assert_eq!(bytes.len(), at + 4 + length);
            assert_eq!(tail, expected);
        }
        for (ranks, _) in FLASH_TOPOLOGIES {
            let topology = package
                .manifest()
                .for_topology(&topology_id(&FLASH, ranks))
                .unwrap();
            assert!(
                topology
                    .resource_requirements()
                    .required_capabilities()
                    .iter()
                    .any(|id| id.as_str() == "chop.model-options.v1")
            );
        }
    }
    std::fs::remove_dir_all(root).unwrap();
}

/// Each topology declares the native mixed-plane pages (890 bytes a token
/// at flash, 56,960 a page) with the native per-item state, and the BF16
/// pages with the BF16 state; the manifest requires the mixed-plane
/// capability.
#[test]
fn the_manifest_declares_each_topology_s_resources() {
    let geometry = &FLASH;
    let topologies = FLASH_TOPOLOGIES;
    let manifest = flash_manifest();
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
    assert!(
        manifest
            .components()
            .iter()
            .any(|component| component.role().as_str() == "chop.frontend.openai.v1")
    );
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

/// Each topology declares its per-item state as its one recurrent state
/// layout, sized for the largest native or BF16 state: the host then gives
/// every request a state slot, names it in each batch and restores prefix
/// snapshots into it. Create allocates every slot's state before the
/// serving KV pool, so the host's slot pool reserves nothing beside it, and
/// the state the host creates the bundle with in either KV format fits the
/// declared slots. The bundle reserves drafter state only when enabled.
#[test]
fn each_topology_keeps_its_request_state_in_host_slots() {
    use runtime_bundle_abi::{
        PREFIX_CACHE_CAPABILITY_V1, PREFIX_CACHE_STATE_SLOT_CAPABILITY_V1,
        RECURRENT_STATE_CAPABILITY_V1, RECURRENT_STATE_IN_PLACE_CAPABILITY_V1,
        RECURRENT_STATE_PREALLOCATED_CAPABILITY_V1,
    };
    let manifest = flash_manifest();
    for (ranks, _) in FLASH_TOPOLOGIES {
        let topology = manifest.for_topology(&topology_id(&FLASH, ranks)).unwrap();
        let layouts = topology.recurrent_state_layouts();
        assert_eq!(layouts.len(), 1);
        let layout = &layouts[0];
        assert_eq!(layout.id().as_str(), REQUEST_STATE_ID);
        assert_eq!(
            layout.lifetime(),
            runtime_bundle::BundleStateLifetimeV1::Request
        );
        assert_eq!(layout.bytes(), slot_state_bytes(&FLASH, KvStorage::Bf16));
        assert_eq!(layout.bytes(), 5_668_864);
        assert_eq!(
            topology.recurrent_state_bytes_per_slot().unwrap(),
            layout.bytes()
        );
        // The host's state-slot restore needs the prefix cache and request
        // state slots beside its own capability; a cancelled step's slots
        // are invalid (`Cancel::Invalidate`), so the host retires its items.
        let requires = |capability: &str| {
            topology
                .resource_requirements()
                .required_capabilities()
                .iter()
                .any(|required| required.as_str() == capability)
        };
        for capability in [
            PREFIX_CACHE_CAPABILITY_V1,
            PREFIX_CACHE_STATE_SLOT_CAPABILITY_V1,
            RECURRENT_STATE_CAPABILITY_V1,
            RECURRENT_STATE_PREALLOCATED_CAPABILITY_V1,
            RECURRENT_STATE_IN_PLACE_CAPABILITY_V1,
        ] {
            assert!(requires(capability), "{capability}");
        }
        // With or without verify rows: the state rolls back by position.
        for verify_rows in [1, 7] {
            let pool = runtime_host::recurrent::StateSlotPoolV1::new(
                &topology,
                DECODE_BATCH as usize,
                verify_rows,
            )
            .unwrap();
            assert_eq!(pool.reserved_bytes_per_rank(), 0);
        }
        let bf16 = topology
            .kv_formats()
            .iter()
            .find(|format| format.dtype() == BundleKvDtypeV1::Bf16)
            .unwrap()
            .layout();
        for served in [1, 3, DECODE_BATCH] {
            let slots = runtime_bundle::state_slots_v1(served);
            for format in topology.kv_formats() {
                let created = topology
                    .state_bytes_for_v1(format.layout(), Some(served))
                    .unwrap();
                assert!(created <= layout.bytes() * slots, "{:?}", format.dtype());
            }
            assert_eq!(
                topology.state_bytes_for_v1(bf16, Some(served)).unwrap(),
                slot_state_bytes(&FLASH, KvStorage::Bf16) * slots
            );
        }
    }
}

/// Flash's weights fit a device at four ranks only: the package of its
/// checkpoint serves TP4, and a TP2 request is refused by capacity. A
/// rank's weights are its arena less the mapped Engram planes, which stay
/// in the checkpoint.
#[test]
fn flash_fits_at_four_ranks_alone() {
    let root = sparse(&FLASH);
    let served = served_topologies(&root, &FLASH, 4, 0).unwrap();
    assert_eq!(
        served.iter().map(|&(ranks, _)| ranks).collect::<Vec<_>>(),
        [4]
    );
    let plans: Vec<_> = (0..4)
        .map(|rank| {
            let request = model_deepseek41::checkpoint::request(&root, 4, rank).unwrap();
            weight_plan::plan(&request).unwrap().unwrap()
        })
        .collect();
    let device = plans.iter().map(weight_plan::device_bytes).max().unwrap();
    assert_eq!(served[0].1, device);
    for plan in &plans {
        let mapped: u64 = plan.views.iter().map(|view| view.byte_length).sum();
        assert!(mapped > 200_000_000_000, "the Engram planes are mapped");
        assert!(weight_plan::device_bytes(plan) + mapped <= plan.arena_bytes);
    }
    eprintln!("flash tp4: {} weight bytes a rank", served[0].1);
    let error = served_topologies(&root, &FLASH, 2, 0)
        .unwrap_err()
        .to_string();
    assert!(error.contains("does not fit at 2 ranks"), "{error}");
    std::fs::remove_dir_all(&root).unwrap();
    let Some(geometry) = synthetic() else {
        return;
    };
    let small = sparse(geometry);
    let served = served_topologies(&small, geometry, 2, 0).unwrap();
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
        "chop.frontend.openai.v1",
    ] {
        assert!(
            manifest
                .components()
                .iter()
                .any(|component| component.role().as_str() == role),
            "{role}"
        );
    }
    // The frontend states the geometry's logits width and context.
    assert_eq!(
        std::fs::read(emitted.package.root().join("components/frontend.json")).unwrap(),
        model_deepseek41::frontend::component_for(geometry).unwrap()
    );
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

/// Flash's checkpoint stores the vision tower the declaration pins, so it
/// provides images, which `auto` and `all` serve and a text list leaves
/// out; a synthetic geometry's smaller tower provides text alone.
#[test]
fn a_checkpoint_with_the_tower_provides_images() {
    use model_deepseek41::preparation::provided_modalities;
    use runtime_bundle::preparation::{Modality, ModalitySelection};
    use std::collections::BTreeSet;
    let root = sparse(&FLASH);
    assert_eq!(
        provided_modalities(&root, &FLASH).unwrap(),
        BTreeSet::from([Modality::Text, Modality::Image])
    );
    let request = |modalities| PrepareDirectRequest {
        path: root.clone(),
        cache_root: root.join("cache"),
        tensor_parallel_size: 4,
        activation_policy: None,
        modalities,
    };
    let served = |modalities| {
        DeepSeek41Family
            .direct_modalities(&root, &request(modalities))
            .map(Option::unwrap)
    };
    let both = BTreeSet::from([Modality::Text, Modality::Image]);
    assert_eq!(served(ModalitySelection::Auto).unwrap(), both);
    assert_eq!(served(ModalitySelection::All).unwrap(), both);
    let expected = select_profile(&FLASH, 4).unwrap();
    for modalities in [ModalitySelection::Auto, ModalitySelection::All] {
        let selected = DeepSeek41Family
            .direct_profile(&root, &request(modalities))
            .unwrap();
        assert_eq!(selected.name, expected.name);
        assert_eq!(selected.profile, expected.profile);
        assert_eq!(selected.profile["prefix_cache"], true);
    }
    let text = BTreeSet::from([Modality::Text]);
    assert_eq!(served(ModalitySelection::Only(text.clone())).unwrap(), text);
    let images = BTreeSet::from([Modality::Image]);
    assert!(served(ModalitySelection::Only(images)).is_err());
    std::fs::remove_dir_all(&root).unwrap();
    let Some(geometry) = synthetic() else {
        return;
    };
    let small = sparse(geometry);
    assert_eq!(provided_modalities(&small, geometry).unwrap(), text);
    std::fs::remove_dir_all(&small).unwrap();
}

/// A package with the planned encoder image serves images: the encoder
/// image is planned beside the text image, every topology requires the
/// media capability and holds the encoder's image and buffers, and the
/// frontend is the image one. The text package has none of them.
#[test]
fn an_image_package_plans_the_encoder_and_requires_media_inputs() {
    use model_deepseek41::encoder_image::{IMAGE_COMPONENT, IMAGE_FORMAT, planned};
    use model_deepseek41::package::media_bytes;
    use model_deepseek41_contract::encoder::{MEDIA_CAPABILITY, views::IMAGE_ROLE};
    use sha2::{Digest, Sha256};
    let root = sparse(&FLASH);
    let manifest = |media: bool| {
        let mut components = vec![
            package_draft::tokenizer_definition(StagedContentV1::bytes(b"{}".to_vec())).unwrap(),
        ];
        if media {
            components.push(planned(&root, 4).unwrap());
        }
        let draft = draft(
            &FLASH,
            FLASH.runtime_defaults(),
            &FLASH_TOPOLOGIES,
            components,
            StagedContentV1::bytes(b"module".to_vec()),
        )
        .unwrap();
        model_compiler_sdk::plan_bundle_manifest_v1(&draft).unwrap()
    };
    let hex = |bytes: &[u8]| -> String {
        Sha256::digest(bytes)
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect()
    };
    let encoder_bytes = media_bytes(true).unwrap();
    for media in [false, true] {
        let manifest = manifest(media);
        let encoder = manifest
            .planned_components()
            .iter()
            .find(|component| component.id().as_str() == IMAGE_COMPONENT);
        assert_eq!(encoder.is_some(), media);
        if let Some(encoder) = encoder {
            assert_eq!(encoder.role().as_str().as_bytes(), IMAGE_ROLE);
            assert_eq!(encoder.format().as_str(), IMAGE_FORMAT);
        }
        assert!(
            manifest
                .planned_components()
                .iter()
                .any(|component| component.id().as_str() == RANK_IMAGE_COMPONENT)
        );
        assert_eq!(
            runtime_host::video::media_inputs_required_v1(&manifest),
            media
        );
        let frontend = manifest
            .components()
            .iter()
            .find(|component| component.role().as_str() == "chop.frontend.openai.v1")
            .unwrap();
        let expected = if media {
            model_deepseek41::frontend::multimodal_component_for(&FLASH).unwrap()
        } else {
            model_deepseek41::frontend::COMPONENT.to_vec()
        };
        assert_eq!(frontend.sha256(), hex(&expected));
        for (ranks, weight_bytes) in FLASH_TOPOLOGIES {
            let requirements = manifest
                .for_topology(&topology_id(&FLASH, ranks))
                .unwrap()
                .resource_requirements()
                .clone();
            let capabilities: Vec<_> = requirements
                .required_capabilities()
                .iter()
                .map(|capability| capability.as_str())
                .collect();
            assert_eq!(capabilities.contains(&MEDIA_CAPABILITY), media);
            let extra = if media { encoder_bytes } else { 0 };
            assert_eq!(
                requirements.minimum_device_memory_bytes(),
                device_bytes(weight_bytes + extra)
            );
        }
    }
    // Flash still fits at four ranks with the encoder's bytes.
    let served = served_topologies(&root, &FLASH, 4, encoder_bytes).unwrap();
    assert_eq!(
        served.iter().map(|&(ranks, _)| ranks).collect::<Vec<_>>(),
        [4]
    );
    eprintln!("encoder: {encoder_bytes} bytes a rank");
    std::fs::remove_dir_all(&root).unwrap();
}

/// The existing rank image contains every drafter tensor. Its authenticated
/// component binds the advertised drafter to the package, at every topology.
#[test]
fn native_packages_advertise_full_block_drafting_and_bounded_verify_rows() {
    let manifest = flash_manifest();
    for (ranks, _) in FLASH_TOPOLOGIES {
        let topology = manifest.for_topology(&topology_id(&FLASH, ranks)).unwrap();
        let speculation = topology.speculation().unwrap();
        assert!(speculation.bundle_drafter());
        assert_eq!(speculation.drafter_kind(), Some("dspark"));
        assert_eq!(
            speculation.drafter_component().unwrap().as_str(),
            RANK_IMAGE_COMPONENT
        );
        assert_eq!(speculation.maximum_draft_tokens(), 5);
        assert_eq!(speculation.maximum_verify_tokens(), 32);
        assert_eq!(speculation.row_buckets(), &[2, 3, 4, 5, 6]);
        let slot = bundle_runtime::state::Extent {
            served: 1,
            bucket: 1,
            verify_rows: 0,
        };
        let target = request_state(&FLASH, KvStorage::Native, false);
        let declared = target
            .buffers
            .iter()
            .map(|buffer| buffer.bytes(slot))
            .sum::<u64>()
            .next_multiple_of(256);
        assert_eq!(topology.state_layout().bytes(), declared);
    }
}
