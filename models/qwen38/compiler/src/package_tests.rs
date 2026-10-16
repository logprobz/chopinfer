use super::*;
use model_qwen38_contract::contract::{
    BUNDLE_ID, KV_LAYOUT_BF16, KV_LAYOUT_BF16_MTP, PAGE_BYTES_BF16_MTP, index_role,
};

const RANKS: u32 = Shape::TP4.ranks();

fn native_module() -> StagedContentV1 {
    StagedContentV1::file(crate::test_native::module())
}

fn scratch(name: &str) -> std::path::PathBuf {
    let base = std::env::var_os("CHOPINFER_SCRATCH").map_or_else(std::env::temp_dir, Into::into);
    base.join(format!("qwen38-package-{name}-{}", std::process::id()))
}

fn has_capability(manifest: &runtime_bundle::BundleManifestV1, capability: &str) -> bool {
    manifest
        .resource_requirements()
        .required_capabilities()
        .iter()
        .any(|c| c.as_str() == capability)
}

/// The assets of a package without rank images for `shape` and `drafter`:
/// placeholder tokenizer files and the test native module.
fn image_free(shape: Shape, drafter: Drafter) -> PackageAssets {
    PackageAssets {
        shape,
        runtime: crate::config::RuntimeValues {
            indexer_budget: shape.geometry.limits.indexer_budget as usize,
        },
        drafter,
        tokenizer_definition: StagedContentV1::bytes(b"{}".to_vec()),
        tokenizer_configuration: StagedContentV1::bytes(b"{}".to_vec()),
        native_module: native_module(),
        encoders: None,
    }
}

/// Emits the package without rank images of `assets` at `output`, creating
/// its parent directory.
fn emit(assets: PackageAssets, output: &std::path::Path) -> model_compiler_sdk::EmittedBundleV1 {
    std::fs::create_dir_all(output.parent().unwrap()).unwrap();
    let draft = bundle_draft_direct(assets, None, None).expect("image-free draft");
    model_compiler_sdk::emit_bundle_directory_v1(&draft, output).expect("package emission")
}

/// A text package without media, served without speculation, still keeps
/// its prompt-end anchors inside a step: its anchor items select the V3
/// batch wire, which alone carries the kept-anchor flag.
#[test]
fn a_text_package_with_anchor_items_negotiates_a_v3_wire() {
    for drafter in [Drafter::None, Drafter::Mtp] {
        let root = scratch(&format!("anchor-wire-{drafter:?}"));
        let output = root.join("bundle");
        let emitted = emit(image_free(Shape::TP4, drafter), &output);
        let manifest = emitted.manifest();
        assert!(!has_capability(manifest, "chop.media-inputs.v1"));
        assert_eq!(manifest.batch_limits().unwrap().anchor_items(), 2);
        assert!(matches!(
            runtime_host::batch_wire_capability_v1(manifest),
            runtime_host::BatchWireCapabilityV1::V3Speculative
        ));
        std::fs::remove_dir_all(&root).unwrap();
    }
}

#[test]
fn an_image_free_text_package_declares_the_serving_contract() {
    let root = scratch("text");
    let output = root.join("bundle");
    let emitted = emit(image_free(Shape::TP4, Drafter::None), &output);
    let manifest = emitted.manifest();
    assert_eq!(manifest.id().as_str(), BUNDLE_ID);
    assert_eq!(manifest.target().topologies()[0].rank_count(), RANKS);
    let kv = manifest.kv_layout();
    assert_eq!(kv.id().as_str().as_bytes(), KV_LAYOUT_BF16);
    assert_eq!((kv.page_bytes(), kv.page_tokens()), (835_584, 64));
    assert!(!has_capability(
        manifest,
        runtime_bundle_abi::FP8_KV_CAPABILITY_V1
    ));
    // The ring is declared per state slot; 32 served sequences hold the
    // 3,293,184 bytes the package declared before the served count.
    let ring = manifest.state_layout();
    assert_eq!(ring.id().as_str().as_bytes(), RING_STATE_ID);
    assert_eq!(ring.bytes(), RING_STATE_SLOT_BYTES);
    assert_eq!(manifest.state_bytes_v1(Some(32)).unwrap(), 3_293_184);
    assert_eq!(
        manifest.state_bytes_v1(Some(100)).unwrap(),
        128 * RING_STATE_SLOT_BYTES
    );
    let recurrent: Vec<_> = manifest
        .recurrent_state_layouts()
        .iter()
        .map(|layout| (layout.id().as_str(), layout.bytes()))
        .collect();
    assert_eq!(recurrent, Shape::TP4.recurrent_state_layouts());
    assert_eq!(
        manifest.recurrent_state_bytes_per_slot().unwrap(),
        552_960 + 28_311_552 + 184_448
    );
    for capability in [
        SERVING_KV_CAPABILITY_V1,
        RECURRENT_STATE_CAPABILITY_V1,
        RECURRENT_STATE_PREALLOCATED_CAPABILITY_V1,
        REQUEST_RETIREMENT_CAPABILITY_V1,
        "chop.prompt-rows.v2",
        "chop.speculative.v1",
        "chop.prefix-cache.v1",
        "chop.kv-offload.v1",
        "chop.prefix-cache.state-slot.v1",
        runtime_bundle_abi::SERVED_ITEMS_CAPABILITY_V1,
        runtime_bundle_abi::STATE_PER_ITEM_CAPABILITY_V1,
        runtime_bundle_abi::weight_plan::WEIGHT_PLAN_CAPABILITY_V1,
        runtime_bundle_abi::frontend::FRONTEND_CAPABILITY_V1,
    ] {
        assert!(has_capability(manifest, capability), "{capability}");
    }
    let limits = manifest.batch_limits().expect("declared batch limits");
    assert_eq!(
        (limits.maximum_sequences(), limits.maximum_prefill_tokens()),
        (256, PREFILL_TOKENS)
    );
    // Chunks and anchors on the delta rule's 64-row chunks.
    assert_eq!(limits.prefill_granule(), 64);
    assert!(!limits.rewrites_last_page());
    assert_eq!(limits.anchor_items(), 2);
    // Without the drafter the host drafts.
    let speculation = manifest.speculation().expect("host-drafted speculation");
    assert!(!speculation.bundle_drafter());
    assert!(speculation.drafter_component().is_none());
    assert_eq!(speculation.maximum_draft_tokens(), 7);
    assert_eq!(speculation.row_buckets(), [2, 4, 8]);
    for capability in [MEDIA_CAPABILITY, POSITION_AXES_CAPABILITY] {
        assert!(!has_capability(manifest, capability), "{capability}");
    }
    // The weight plan supplies the rank images and their indexes.
    let roles: Vec<_> = manifest
        .components()
        .iter()
        .map(|c| c.role().as_str().to_owned())
        .collect();
    assert!(!roles.iter().any(|r| r.as_bytes() == RANK_IMAGE_ROLE));
    for rank in 0..RANKS {
        assert!(!roles.contains(&index_role(rank)));
    }
    let buckets: Vec<_> = manifest
        .graph_buckets()
        .iter()
        .map(|b| b.id().as_str())
        .collect();
    assert_eq!(
        buckets,
        [
            "qwen38-decode-b1",
            "qwen38-decode-b128",
            "qwen38-decode-b16",
            "qwen38-decode-b2",
            "qwen38-decode-b256",
            "qwen38-decode-b32",
            "qwen38-decode-b4",
            "qwen38-decode-b64",
            "qwen38-decode-b8",
            "qwen38-prefill"
        ]
    );
    assert_eq!(
        std::fs::read(output.join("components/frontend.json")).unwrap(),
        crate::frontend::COMPONENT
    );
    // The package carries the model's profiles and deployment policy byte
    // for byte.
    for (path, file) in [
        (
            runtime_bundle::deployment::SERVE_PROFILES_PATH_V1,
            "profiles.json",
        ),
        (
            runtime_bundle::deployment::DEPLOYMENT_POLICY_PATH_V1,
            "deploy/policy.json",
        ),
    ] {
        let model = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
        assert_eq!(
            std::fs::read(output.join(path)).unwrap(),
            std::fs::read(model.join(file)).unwrap()
        );
    }
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn the_mtp_drafter_declares_speculation_and_its_layer_state() {
    let root = scratch("mtp");
    let emitted = emit(image_free(Shape::TP4, Drafter::Mtp), &root.join("bundle"));
    let manifest = emitted.manifest();
    let kv = manifest.kv_layout();
    assert_eq!(kv.id().as_str().as_bytes(), KV_LAYOUT_BF16_MTP);
    assert_eq!(
        (kv.page_bytes(), kv.page_tokens()),
        (PAGE_BYTES_BF16_MTP, 64)
    );
    let ring = manifest.state_layout();
    assert_eq!(ring.id().as_str().as_bytes(), RING_STATE_ID);
    assert_eq!(ring.bytes(), RING_STATE_SLOT_BYTES_MTP);
    assert!(manifest.state_per_item());
    assert_eq!(
        manifest.state_bytes_v1(Some(8)).unwrap(),
        8 * RING_STATE_SLOT_BYTES_MTP
    );
    assert!(has_capability(manifest, SPECULATIVE_CAPABILITY_V1));
    assert!(has_capability(
        manifest,
        RECURRENT_STATE_PREALLOCATED_CAPABILITY_V1
    ));
    // Snapshots carry the drafter's ring layer and retained rows.
    assert!(has_capability(manifest, "chop.prefix-cache.state-slot.v1"));
    // MTP serves the sequences its accuracy is qualified for, 128, with at
    // most 32 of them drafting in a step (the checkpoint pool).
    let limits = manifest.batch_limits().expect("declared batch limits");
    assert_eq!(
        (limits.maximum_sequences(), limits.maximum_prefill_tokens()),
        (128, PREFILL_TOKENS)
    );
    // A continuation's first drafter row rewrites the anchor's last row.
    assert_eq!(limits.prefill_granule(), 64);
    assert!(limits.rewrites_last_page());
    assert_eq!(limits.anchor_items(), 2);
    assert_eq!(manifest.graph_buckets().len(), 8 + 1);
    let speculation = manifest.speculation().expect("MTP speculation");
    assert_eq!(speculation.maximum_draft_items(), Some(32));
    assert_eq!(speculation.maximum_draft_tokens(), 7);
    assert!(speculation.bundle_drafter());
    assert_eq!(speculation.maximum_verify_tokens(), 256);
    assert_eq!(speculation.row_buckets(), [2, 4, 8]);
    let drafter = speculation.drafter_component().unwrap();
    assert_eq!(drafter.as_str(), RANK_IMAGE_COMPONENT);
    let component = manifest
        .planned_components()
        .iter()
        .find(|c| c.id() == drafter)
        .unwrap();
    assert_eq!(component.role().as_str().as_bytes(), RANK_IMAGE_ROLE);
    std::fs::remove_dir_all(root).unwrap();
}

/// Each package declares BF16 and FP8 KV and prefers BF16, with and without
/// the drafter, whose layer stays BF16 in the FP8 layout.
#[test]
fn packages_declare_bf16_and_fp8_kv_and_prefer_bf16() {
    use model_qwen38_contract::contract::{
        KV_LAYOUT_FP8, KV_LAYOUT_FP8_MTP, PAGE_BYTES_BF16, PAGE_BYTES_FP8, PAGE_BYTES_FP8_MTP,
    };
    for (drafter, bf16, fp8) in [
        (
            Drafter::None,
            (KV_LAYOUT_BF16, PAGE_BYTES_BF16),
            (KV_LAYOUT_FP8, PAGE_BYTES_FP8),
        ),
        (
            Drafter::Mtp,
            (KV_LAYOUT_BF16_MTP, PAGE_BYTES_BF16_MTP),
            (KV_LAYOUT_FP8_MTP, PAGE_BYTES_FP8_MTP),
        ),
    ] {
        let root = scratch(&format!("kv-formats-{drafter:?}"));
        let emitted = emit(image_free(Shape::TP4, drafter), &root.join("bundle"));
        let manifest = emitted.manifest();
        let formats: Vec<_> = manifest
            .kv_formats()
            .iter()
            .map(|format| {
                (
                    format.dtype(),
                    format.layout().id().as_str().as_bytes(),
                    format.layout().page_bytes(),
                )
            })
            .collect();
        assert_eq!(
            formats,
            [
                (BundleKvDtypeV1::Bf16, bf16.0, bf16.1),
                (BundleKvDtypeV1::Fp8E4m3, fp8.0, fp8.1),
            ]
        );
        assert_eq!(manifest.kv_layout(), manifest.kv_formats()[0].layout());
        // The host serves BF16 for auto and bf16 and the FP8 layout for fp8.
        for (requested, dtype, layout) in [
            (
                runtime_host::KvCacheDtypeV1::Auto,
                BundleKvDtypeV1::Bf16,
                bf16.0,
            ),
            (
                runtime_host::KvCacheDtypeV1::Bf16,
                BundleKvDtypeV1::Bf16,
                bf16.0,
            ),
            (
                runtime_host::KvCacheDtypeV1::Fp8,
                BundleKvDtypeV1::Fp8E4m3,
                fp8.0,
            ),
        ] {
            let choice =
                runtime_host::resolve_kv_cache_dtype_v1(manifest, requested).expect("declared");
            assert_eq!(
                (choice.dtype, choice.automatic),
                (dtype, requested == runtime_host::KvCacheDtypeV1::Auto)
            );
            assert_eq!(
                runtime_host::kv_layout_of_v1(manifest, dtype)
                    .map(|layout| layout.id().as_str().as_bytes()),
                Some(layout)
            );
        }
        assert!(!has_capability(
            manifest,
            runtime_bundle_abi::FP8_KV_CAPABILITY_V1
        ));
        std::fs::remove_dir_all(root).unwrap();
    }
}

/// A TP2 rank image leaves room for KV only with the n-gram table in host
/// memory, where the weight plan places it.
#[test]
fn tp2_needs_the_ngram_table_in_host_memory() {
    let error = check_fits(Shape::TP2, 93_813_366_784, NgramPlacement::Gpu)
        .unwrap_err()
        .to_string();
    assert!(error.contains("87.4 GiB"), "{error}");
    check_fits(Shape::TP2, 93_813_366_784, NgramPlacement::Host).unwrap();
    check_fits(Shape::TP4, 48_412_143_616, NgramPlacement::Gpu).unwrap();
}

#[test]
fn wide_mtp_declaration_keeps_k2_and_default_buckets_stay_unchanged() {
    assert_eq!(crate::package::verify_row_buckets(true, true), [2, 3, 4, 8]);
    for (drafter, wide) in [(true, false), (false, false), (false, true)] {
        assert_eq!(crate::package::verify_row_buckets(drafter, wide), [2, 4, 8]);
    }
}

#[test]
fn an_image_free_package_plans_its_rank_images_and_leaves_out_the_indexes() {
    let root = scratch("image-free");
    for shape in Shape::ALL {
        let output = root.join(format!("tp{}", shape.ranks()));
        let emitted = emit(image_free(shape, Drafter::None), &output);
        assert_eq!(
            emitted.manifest().target().topologies()[0].rank_count(),
            shape.ranks()
        );
        let manifest = emitted.manifest();
        let roles: Vec<_> = manifest
            .components()
            .iter()
            .map(|c| c.role().as_str().to_owned())
            .collect();
        assert!(!roles.iter().any(|r| r.as_bytes() == RANK_IMAGE_ROLE));
        let planned: Vec<_> = manifest
            .planned_components()
            .iter()
            .map(|c| (c.id().as_str(), c.role().as_str().as_bytes()))
            .collect();
        assert_eq!(planned, [(RANK_IMAGE_COMPONENT, RANK_IMAGE_ROLE)]);
        assert!(manifest.graph_buckets().iter().all(|b| {
            b.components()
                .iter()
                .any(|c| c.as_str() == RANK_IMAGE_COMPONENT)
        }));
        for rank in 0..shape.ranks() {
            assert!(!roles.contains(&index_role(rank)));
        }
        assert!(!output.join("images").exists());
        assert!(has_capability(
            manifest,
            runtime_bundle_abi::weight_plan::WEIGHT_PLAN_CAPABILITY_V1
        ));
        // The plan places the n-gram table in host memory where the device
        // capacity rule does: at TP2 only.
        assert_eq!(
            has_capability(manifest, HOST_RESIDENT_VIEWS_CAPABILITY_V1),
            model_qwen38_contract::package::ngram_in_host_memory(shape)
        );
        assert_eq!(
            model_qwen38_contract::package::ngram_in_host_memory(shape),
            shape == Shape::TP2
        );
    }
    // The MTP drafter names the planned rank images, whose views it reads.
    let emitted = emit(image_free(Shape::TP4, Drafter::Mtp), &root.join("tp4-mtp"));
    let speculation = emitted.manifest().speculation().unwrap();
    assert!(speculation.bundle_drafter());
    assert_eq!(
        speculation.drafter_component().map(|c| c.as_str()),
        Some(RANK_IMAGE_COMPONENT)
    );
    std::fs::remove_dir_all(root).unwrap();
}

/// Planned encoders advertise image inputs only after the checkpoint's
/// vision tower was checked: encoder assets that were not made from a
/// checked catalog are refused.
#[test]
fn an_image_free_package_refuses_encoders_without_a_checked_vision_tower() {
    let mut assets = image_free(Shape::TP4, Drafter::None);
    assets.encoders = Some(crate::encoders::EncoderAssets {
        positions: crate::encoders::ImagePositions::Grid3Axis,
        checked_tower: None,
    });
    let Err(error) = bundle_draft_direct(assets, None, None) else {
        panic!("unchecked planned encoders");
    };
    let error = error.to_string();
    assert!(error.contains("vision tower"), "{error}");
    // A text-only checkpoint yields no planned encoders to package.
    let mut text_only = vision_catalog(4);
    text_only
        .tensors
        .retain(|name, _| !name.starts_with("model.visual."));
    let error = crate::encoders::EncoderAssets::planned(
        &text_only,
        crate::encoders::ImagePositions::Grid3Axis,
    )
    .map(drop)
    .expect_err("a text-only checkpoint")
    .to_string();
    assert!(error.contains("checkpoint lacks model.visual."), "{error}");
}

/// A catalog holding exactly the vision tower the encoder layout copies.
fn vision_catalog(ranks: usize) -> crate::catalog::Catalog {
    let tensors = model_qwen38_contract::encoder::views::layout_flash()
        .into_iter()
        .map(|view| {
            let bytes = 2 * view.source_shape.iter().product::<u64>();
            let record = crate::catalog::TensorRecord {
                name: view.name.clone(),
                dtype: runtime_weights::DType::Bf16,
                shape: view.source_shape,
                file: "model-00001.safetensors".into(),
                file_start: 0,
                bytes,
                placement: crate::catalog::Placement::Replicated,
                scale: crate::catalog::ScaleLayout::None,
            };
            (view.name, record)
        })
        .collect();
    crate::catalog::Catalog {
        geometry: &model_qwen38_contract::geometry::FLASH,
        tensors,
        ranks,
        precision: None,
    }
}

#[test]
fn an_image_free_package_plans_its_encoder_images_with_image_inputs() {
    let root = scratch("image-free-encoders");
    let encoder_role =
        std::str::from_utf8(model_qwen38_contract::encoder::views::IMAGE_ROLE).unwrap();
    for shape in Shape::ALL {
        let mut assets = image_free(shape, Drafter::None);
        assets.encoders = Some(
            crate::encoders::EncoderAssets::planned(
                &vision_catalog(shape.ranks() as usize),
                crate::encoders::ImagePositions::Grid3Axis,
            )
            .unwrap(),
        );
        let output = root.join(format!("tp{}", shape.ranks()));
        let emitted = emit(assets, &output);
        let manifest = emitted.manifest();
        let planned: Vec<_> = manifest
            .planned_components()
            .iter()
            .map(|c| (c.id().as_str(), c.role().as_str()))
            .collect();
        // By component id.
        assert_eq!(
            planned,
            [
                (crate::encoders::IMAGE_COMPONENT, encoder_role),
                (RANK_IMAGE_COMPONENT, utf8(RANK_IMAGE_ROLE)),
            ]
        );
        assert!(manifest.graph_buckets().iter().all(|b| {
            b.components()
                .iter()
                .any(|c| c.as_str() == crate::encoders::IMAGE_COMPONENT)
        }));
        // The plan supplies each rank's encoder index.
        let roles: Vec<_> = manifest
            .components()
            .iter()
            .map(|c| c.role().as_str().to_owned())
            .collect();
        for rank in 0..shape.ranks() {
            assert!(!roles.contains(&model_qwen38_contract::encoder::views::index_role(rank)));
        }
        assert!(!output.join("encoders").exists());
        for capability in [
            MEDIA_CAPABILITY,
            POSITION_AXES_CAPABILITY,
            runtime_bundle_abi::weight_plan::WEIGHT_PLAN_CAPABILITY_V1,
        ] {
            assert!(has_capability(manifest, capability), "{capability}");
        }
        let frontend = std::fs::read(output.join("components/frontend.json")).unwrap();
        assert_eq!(frontend, crate::frontend::multimodal_component());
        // Each rank holds the planned encoder image, its buffers and arena.
        assert_eq!(
            manifest
                .resource_requirements()
                .minimum_device_memory_bytes(),
            crate::package::minimum_device_memory(shape, false, Encoders::VISION)
        );
    }
    // Sequential image positions drop the rule and its capability.
    let mut assets = image_free(Shape::TP4, Drafter::None);
    assets.encoders = Some(
        crate::encoders::EncoderAssets::planned(
            &vision_catalog(4),
            crate::encoders::ImagePositions::Sequential,
        )
        .unwrap(),
    );
    let output = root.join("sequential");
    let emitted = emit(assets, &output);
    assert!(has_capability(emitted.manifest(), MEDIA_CAPABILITY));
    assert!(!has_capability(
        emitted.manifest(),
        POSITION_AXES_CAPABILITY
    ));
    let frontend = std::fs::read(output.join("components/frontend.json")).unwrap();
    let descriptor =
        runtime_frontend::FrontendDescriptorV1::from_component_bytes(&frontend).unwrap();
    assert_eq!(descriptor.multimodal.unwrap().modalities[0].positions, None);
    std::fs::remove_dir_all(root).unwrap();
}

/// Each served set of modalities is its own package: serving text alone
/// leaves out the vision encoder (its planned image, the media capability
/// and the frontend's media) and, at TP2, serves more sequences
/// from the device bytes it frees. A modality the checkpoint lacks is
/// refused by name.
#[test]
fn each_served_set_of_modalities_is_its_own_package() {
    use runtime_bundle::preparation::{Modality, ModalitySelection, PrepareDirectRequest};
    use std::collections::BTreeSet;
    let request = |modalities: Option<&[Modality]>| PrepareDirectRequest {
        path: "checkpoint".into(),
        cache_root: "cache".into(),
        tensor_parallel_size: 4,
        activation_policy: None,
        modalities: modalities.map_or(ModalitySelection::Auto, |names| {
            ModalitySelection::Only(names.iter().copied().collect::<BTreeSet<_>>())
        }),
    };
    let served = |modalities: Option<&[Modality]>, catalog: &crate::catalog::Catalog| {
        crate::preparation::served_encoders(&request(modalities), catalog)
            .map_err(|error| error.to_string())
    };
    let vision = vision_catalog(4);
    let text_only = crate::catalog::Catalog {
        geometry: &model_qwen38_contract::geometry::FLASH,
        tensors: Default::default(),
        ranks: 4,
        precision: None,
    };
    let (text, image) = (Modality::Text, Modality::Image);
    assert!(served(None, &vision).unwrap().is_some());
    assert!(served(Some(&[text, image]), &vision).unwrap().is_some());
    assert_eq!(served(Some(&[text]), &vision), Ok(None));
    assert_eq!(served(None, &text_only), Ok(None));
    // Qwen provides text and image only, so `all` serves what auto does.
    let all = PrepareDirectRequest {
        modalities: ModalitySelection::All,
        ..request(None)
    };
    assert_eq!(
        crate::preparation::served_modalities(&all, &vision).unwrap(),
        crate::preparation::served_modalities(&request(None), &vision).unwrap()
    );
    for (modalities, catalog, missing) in [
        (&[text, image][..], &text_only, "image"),
        (&[text, Modality::Video], &vision, "video"),
        (&[text, image, Modality::Audio], &vision, "audio"),
    ] {
        let refusal = served(Some(modalities), catalog).unwrap_err();
        assert!(
            refusal.contains(&format!("--modalities names {missing}")),
            "{refusal}"
        );
    }
    let root = scratch("modality-sets");
    for shape in Shape::ALL {
        let package = |modalities: Option<&[Modality]>| {
            let mut assets = image_free(shape, Drafter::None);
            let catalog = vision_catalog(shape.ranks() as usize);
            assets.encoders = served(modalities, &catalog)
                .unwrap()
                .map(|positions| crate::encoders::EncoderAssets::planned(&catalog, positions))
                .transpose()
                .unwrap();
            let output = root.join(format!("tp{}-{}", shape.ranks(), modalities.is_none()));
            let emitted = emit(assets, &output);
            let frontend = std::fs::read(output.join("components/frontend.json")).unwrap();
            let frontend =
                runtime_frontend::FrontendDescriptorV1::from_component_bytes(&frontend).unwrap();
            (emitted, frontend)
        };
        let (full, full_frontend) = package(None);
        let (text_package, text_frontend) = package(Some(&[text]));
        let (full, text_package) = (full.manifest(), text_package.manifest());
        assert_ne!(
            full.to_canonical_json().unwrap(),
            text_package.to_canonical_json().unwrap()
        );
        assert!(full_frontend.multimodal.is_some() && text_frontend.multimodal.is_none());
        assert!(has_capability(full, MEDIA_CAPABILITY));
        assert!(!has_capability(text_package, MEDIA_CAPABILITY));
        assert!(
            text_package
                .planned_components()
                .iter()
                .all(|c| c.id().as_str() != crate::encoders::IMAGE_COMPONENT)
        );
        let sequences = |manifest: &runtime_bundle::BundleManifestV1| {
            manifest.batch_limits().unwrap().maximum_sequences()
        };
        let floor = |manifest: &runtime_bundle::BundleManifestV1| {
            manifest
                .resource_requirements()
                .minimum_device_memory_bytes()
        };
        assert_eq!(
            (sequences(full), sequences(text_package)),
            match shape.ranks() {
                2 => (32, 64),
                4 => (256, 256),
                _ => (128, 256),
            }
        );
        assert_eq!(
            floor(text_package),
            crate::package::minimum_device_memory(shape, false, Encoders::NONE)
        );
        assert_eq!(
            floor(full),
            crate::package::minimum_device_memory(shape, false, Encoders::VISION)
        );
    }
    std::fs::remove_dir_all(root).unwrap();
}

/// One package serves every topology: selecting each gives the package of
/// that topology alone, with its n-gram placement, recurrent state, sequence
/// maximum and device floor, under the model's one ID.
#[test]
fn one_package_serves_every_topology() {
    use model_compiler_sdk::plan_bundle_manifest_v1;
    for drafter in [Drafter::None, Drafter::Mtp] {
        let drafts = Shape::ALL
            .map(|shape| bundle_draft_direct(image_free(shape, drafter), None, None).unwrap());
        let package = plan_bundle_manifest_v1(
            &model_compiler_sdk::BundleDraftV1::for_topologies(drafts.to_vec()).unwrap(),
        )
        .unwrap();
        assert_eq!(package.id().as_str(), BUNDLE_ID);
        for (shape, draft) in Shape::ALL.into_iter().zip(&drafts) {
            let selected = package.for_topology(shape.topology().id).unwrap();
            assert_eq!(selected, plan_bundle_manifest_v1(draft).unwrap());
            assert_eq!(
                has_capability(&selected, HOST_RESIDENT_VIEWS_CAPABILITY_V1),
                shape == Shape::TP2
            );
        }
    }
}

/// The package the direct preparation emits for the shipped geometry (the
/// pinned checkpoint's, `flash`) declares context-parallel size 2 on its TP4
/// topology, so the host serves sizes 1 and 2 there, and nothing above 1 at
/// TP2: the sizes `context_parallel::sizes` lists, which create
/// admits.
#[test]
fn the_flash_package_serves_context_parallel_sizes_one_and_two_at_tp4() {
    use model_compiler_sdk::plan_bundle_manifest_v1;
    for drafter in [Drafter::None, Drafter::Mtp] {
        let drafts = Shape::ALL
            .map(|shape| bundle_draft_direct(image_free(shape, drafter), None, None).unwrap());
        let package = plan_bundle_manifest_v1(
            &model_compiler_sdk::BundleDraftV1::for_topologies(drafts.to_vec()).unwrap(),
        )
        .unwrap();
        for shape in Shape::ALL {
            assert_eq!(shape.geometry.id, "flash");
            let topology = package
                .target()
                .topologies()
                .iter()
                .find(|topology| topology.id().as_str() == shape.topology().id)
                .unwrap();
            let served = (1..=4)
                .filter(|&size| topology.serves_context_parallel(size))
                .collect::<Vec<_>>();
            let expected: &[u32] = if shape == Shape::TP4 { &[1, 2] } else { &[1] };
            assert_eq!(served, expected, "{} {drafter:?}", shape.topology().id);
            assert_eq!(
                topology.context_parallel_sizes(),
                model_qwen38_contract::context_parallel::sizes(shape)
            );
        }
    }
}

#[test]
fn package_frontend_limits_come_from_geometry() {
    static GEOMETRY: model_qwen38_contract::geometry::Geometry = {
        let mut geometry = model_qwen38_contract::geometry::FLASH;
        geometry.shape.vocab = 248576;
        geometry.shape.context = 131072;
        geometry
    };
    for shape in Shape::ALL {
        for local in [None, Some(crate::frontend::COMPONENT.to_vec())] {
            let draft = bundle_draft_direct(
                image_free(Shape::new(&GEOMETRY, shape.ranks), Drafter::None),
                local,
                None,
            )
            .unwrap();
            let root = scratch(&format!(
                "frontend-geometry-tp{}-{}",
                shape.ranks(),
                draft.components().len()
            ));
            let output = root.join("bundle");
            std::fs::create_dir_all(&root).unwrap();
            model_compiler_sdk::emit_bundle_directory_v1(&draft, &output).unwrap();
            let bytes = std::fs::read(output.join("components/frontend.json")).unwrap();
            std::fs::remove_dir_all(&root).unwrap();
            let frontend =
                runtime_frontend_contract::FrontendDescriptorV1::from_component_bytes(&bytes)
                    .unwrap();
            assert_eq!(frontend.vocabulary.size, GEOMETRY.vocab());
            assert_eq!(
                frontend.context.maximum_total_tokens as u64,
                GEOMETRY.context_tokens()
            );
            assert_eq!(
                frontend.limits.maximum_generated_tokens as u64,
                GEOMETRY.context_tokens()
            );
            assert_eq!(frontend.tokenizer.model_max_length, 262144);
            assert_eq!(frontend.tokenizer.vocabulary_size, Some(248077));
        }
    }
}

/// Enabling a runtime declaration exercises the package hook, without
/// claiming that the shipped kernels support this hypothetical override.
#[test]
fn package_model_options_come_from_geometry() {
    use model_qwen38_contract::geometry::{FLASH, Geometry, RuntimeField};
    use runtime_bundle::model_options::{CAPABILITY, OptionDefinitionV1};
    let fields = Box::leak(Box::new([RuntimeField {
        overridable: true,
        ..FLASH.runtime_fields[0]
    }]));
    let geometry = Box::leak(Box::new(Geometry {
        runtime_fields: fields,
        ..FLASH
    }));
    for shape in Shape::ALL {
        let root = scratch(&format!("model-options-tp{}", shape.ranks()));
        let baseline = emit(image_free(shape, Drafter::None), &root.join("baseline"));
        assert!(baseline.manifest().model_options().is_none());
        assert!(!has_capability(baseline.manifest(), CAPABILITY));
        let legacy_geometry = Box::leak(Box::new(Geometry {
            runtime_fields: &[],
            ..FLASH
        }));
        let legacy = emit(
            image_free(Shape::new(legacy_geometry, shape.ranks), Drafter::None),
            &root.join("legacy"),
        );
        assert_eq!(
            std::fs::read(root.join("baseline/manifest.json")).unwrap(),
            std::fs::read(root.join("legacy/manifest.json")).unwrap()
        );
        assert_eq!(
            baseline.manifest().to_canonical_json().unwrap(),
            legacy.manifest().to_canonical_json().unwrap()
        );
        let mut assets = image_free(Shape::new(geometry, shape.ranks), Drafter::None);
        assets.runtime.indexer_budget = 1024;
        let draft = bundle_draft_direct(assets, None, None).unwrap();
        let declaration = model_qwen38_contract::package::without_rank_images(
            model_qwen38_contract::package::declarations(
                Shape::new(geometry, shape.ranks),
                model_qwen38_contract::package::ngram_in_host_memory(Shape::new(
                    geometry,
                    shape.ranks,
                )),
                crate::config::RuntimeValues {
                    indexer_budget: 1024,
                },
            ),
        );
        assert_eq!(draft.native_declarations(), declaration);
        let enabled =
            model_compiler_sdk::emit_bundle_directory_v1(&draft, &root.join("enabled")).unwrap();
        let options = enabled
            .manifest()
            .model_options()
            .expect("geometry declaration");
        assert!(has_capability(enabled.manifest(), CAPABILITY));
        assert_eq!(
            options.options,
            geometry.model_options(crate::config::RuntimeValues {
                indexer_budget: 1024
            })
        );
        assert_eq!(
            options.options[0].definition,
            OptionDefinitionV1::Integer {
                default: 1024,
                minimum: 1,
                maximum: 2048,
            }
        );
        assert!(!options.options[0].default_is_qualified);
        std::fs::remove_dir_all(root).unwrap();
    }
}
