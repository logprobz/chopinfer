//! The `deepseek41.v1` package draft: a direct-loading package whose text
//! rank image is planned from the checkpoint (`chop.weight-plan.v1`,
//! `model_deepseek41_contract::weight_plan`).
//!
//! One package serves every rank count of `admission::SUPPORTED_RANKS` its
//! geometry divides over and its weights fit, each as a topology with its
//! own resources: the rank's planned weight bytes and the host reserve, and
//! its two KV formats (`crate::kv`): the native mixed-plane pages, preferred,
//! with the manifest's per-item state, and the BF16 comparison pages with
//! the larger state of BF16 window rings. The package serves
//! autoregressively; the DSpark drafter's state joins with the drafter.
//!
//! The per-item state is also the topology's recurrent state layout, so the
//! host gives every request a state slot, names it in each batch and
//! restores prefix snapshots into it ([`recurrent_state_layout`]).
//!
//! With the planned encoder image (`crate::encoder_image`) the package also
//! takes image inputs: its frontend declares them, it requires the media
//! capability, and each rank's device memory also holds the encoder's image
//! and buffers.

use model_compiler_sdk::{BundleDraftV1, ComponentDraftV1, StagedContentV1, package_draft};
use model_deepseek41_contract::geometry::{Geometry, RuntimeValues};
use model_deepseek41_contract::serving::{
    DECODE_BATCH, KV_LAYOUT_BF16, KV_LAYOUT_NATIVE, PAGE_TOKENS, PREFILL_GRANULE, PREFILL_TOKENS,
    RANK_IMAGE_ROLE, REQUEST_STATE_ID, SCORED_ROWS,
};
use model_deepseek41_contract::state::{KvStorage, request_state};
use model_deepseek41_contract::{BUNDLE_ID, admission};
use runtime_bundle::{
    BundleBatchLimitsV1, BundleError, BundleKvLayoutV1, BundleOpaqueId, BundleRankPlacementV1,
    BundleResourceRequirementsV1, BundleSpeculationV1, BundleStateLayoutV1, BundleStateLifetimeV1,
    BundleTargetV1, BundleTopologyV1,
};

/// The device architecture the native module's kernels are built for.
pub const ARCHITECTURE: &str = "sm_120f";

/// The planned text image's component id and format.
pub const RANK_IMAGE_COMPONENT: &str = "rank-images";
pub const RANK_IMAGE_FORMAT: &str = "deepseek41.rank-image.v1";

/// The model's serving profiles: the flash geometry's, from which a
/// package's own are generated ([`profiles`]).
pub const PROFILES: &str = include_str!("../../profiles.json");

/// Where the synthetic geometries are documented, the evidence an
/// experimental profile of one cites.
const SYNTHETIC_EVIDENCE: &str = "models/deepseek41/README.md#checkpoint-and-geometry";

/// The serving profiles a package of `geometry` serving `ranks` carries, as
/// a `chopinfer-serve-profiles-v2` document. Each entry depends on the
/// geometry and its rank count alone, so the helper selects from the same
/// entries the package holds:
/// - flash: the model's profiles whose topology the package serves;
/// - another geometry: one autoregressive profile per rank count,
///   `<geometry>-tp<ranks>-ar`, the model's autoregressive profile moved to
///   that topology, experimental and unqualified.
pub fn profiles(geometry: &Geometry, ranks: &[u32]) -> Result<String, BundleError> {
    let invalid = |e: serde_json::Error| BundleError::validation(format!("serving profiles: {e}"));
    let mut file: serde_json::Value = serde_json::from_str(PROFILES).map_err(invalid)?;
    let entries = file["profiles"]
        .as_object()
        .cloned()
        .ok_or_else(|| BundleError::validation("serving profiles have no entries"))?;
    let served: Vec<String> = ranks.iter().map(|&r| topology_id(geometry, r)).collect();
    let mut chosen = serde_json::Map::new();
    if geometry.id == model_deepseek41_contract::geometry::FLASH.id {
        for (name, entry) in entries {
            if served.iter().any(|t| entry["topology"] == t.as_str()) {
                chosen.insert(name, entry);
            }
        }
    } else {
        let base = entries
            .values()
            .find(|entry| entry["speculation"]["drafter"] == "none")
            .ok_or_else(|| {
                BundleError::validation("deepseek41 declares no autoregressive profile")
            })?;
        for (&ranks, topology) in ranks.iter().zip(&served) {
            let mut entry = base.clone();
            let fields = entry
                .as_object_mut()
                .ok_or_else(|| BundleError::validation("invalid serving profile"))?;
            for (field, value) in [
                ("topology", topology.as_str().into()),
                ("tensor_parallel", ranks.into()),
                ("devices_required", ranks.into()),
                ("devices", (0..ranks).collect::<Vec<_>>().into()),
                (
                    "qualification",
                    format!(
                        "Not qualified: the synthetic test geometry {} at TP{ranks}, \
                         autoregressive, for bring-up on a synthetic checkpoint \
                         ({SYNTHETIC_EVIDENCE}).",
                        geometry.id
                    )
                    .into(),
                ),
                ("qualified_max_sequences", serde_json::Value::Null),
                ("status", "experimental".into()),
                ("evidence", serde_json::json!([SYNTHETIC_EVIDENCE])),
            ] {
                fields.insert(field.into(), value);
            }
            chosen.insert(format!("{}-tp{ranks}-ar", geometry.id), entry);
        }
    }
    if chosen.is_empty() {
        return Err(BundleError::validation(format!(
            "deepseek41 declares no serving profile for {}",
            served.join(", ")
        )));
    }
    file["profiles"] = chosen.into();
    serde_json::to_string_pretty(&file).map_err(invalid)
}

/// What the package requires of the host, from what the module exports:
/// paged serving KV with per-item state, request state in host slots that
/// create allocates before the serving KV pool and a cancelled step leaves
/// invalid (the state advances in place, so the host retires every item of
/// a step it cancels), request retirement, scored prompt rows, prefix
/// snapshots restored into the state slot, the planned text image and the
/// rows it gathers from the views it maps (the Engram tables), and the chat
/// frontend.
pub const CAPABILITIES: [&str; 17] = [
    runtime_bundle_abi::SERVING_KV_CAPABILITY_V1,
    runtime_bundle_abi::SPECULATIVE_CAPABILITY_V1,
    runtime_bundle_abi::SPECULATIVE_EXACT_GRAPHS_CAPABILITY_V1,
    runtime_bundle_abi::device_results::DEVICE_RESULTS_CAPABILITY_V1,
    runtime_bundle_abi::RECURRENT_STATE_CAPABILITY_V1,
    runtime_bundle_abi::RECURRENT_STATE_PREALLOCATED_CAPABILITY_V1,
    runtime_bundle_abi::RECURRENT_STATE_IN_PLACE_CAPABILITY_V1,
    runtime_bundle_abi::REQUEST_RETIREMENT_CAPABILITY_V1,
    runtime_bundle_abi::PROMPT_ROWS_CAPABILITY_V2,
    runtime_bundle_abi::PREFIX_CACHE_CAPABILITY_V1,
    runtime_bundle_abi::PREFIX_CACHE_STATE_SLOT_CAPABILITY_V1,
    runtime_bundle_abi::SERVED_ITEMS_CAPABILITY_V1,
    runtime_bundle_abi::SERVED_CONTEXT_CAPABILITY_V1,
    runtime_bundle_abi::STATE_PER_ITEM_CAPABILITY_V1,
    runtime_bundle_abi::weight_plan::WEIGHT_PLAN_CAPABILITY_V1,
    runtime_bundle_abi::HOST_MAPPED_GATHER_CAPABILITY_V1,
    runtime_bundle_abi::frontend::FRONTEND_CAPABILITY_V1,
];

fn id(value: impl Into<String>) -> Result<BundleOpaqueId, BundleError> {
    BundleOpaqueId::new(value)
}

/// The topology ID of `geometry` at `ranks`: `deepseek41-<geometry>-tp<ranks>`,
/// the bundle's `shape::topology_id`.
pub fn topology_id(geometry: &Geometry, ranks: u32) -> String {
    format!("deepseek41-{}-tp{ranks}", geometry.id)
}

/// The per-item state bytes of one state slot of `geometry` in `storage`
/// in whole 256-byte blocks. The bundle allocates drafter rings and
/// retained features separately when speculation is enabled.
pub fn slot_state_bytes(geometry: &Geometry, storage: KvStorage) -> u64 {
    let slot = bundle_runtime::state::Extent {
        served: 1,
        bucket: 1,
        verify_rows: 0,
    };
    request_state(geometry, storage, false)
        .buffers
        .iter()
        .map(|buffer| buffer.bytes(slot))
        .sum::<u64>()
        .next_multiple_of(256)
}

/// The device bytes a rank needs before its KV pool: its planned weights
/// and the host reserve.
pub fn device_bytes(weight_bytes: u64) -> u64 {
    weight_bytes + model_compiler_sdk::device::HOST_RESERVE_BYTES
}

/// The device bytes a rank of a package serving images, or not, holds
/// beyond its text weights: the encoder's image and buffers
/// (`encoder_image::device_bytes`), or none.
pub fn media_bytes(media: bool) -> Result<u64, BundleError> {
    if !media {
        return Ok(0);
    }
    crate::encoder_image::device_bytes().map_err(BundleError::validation)
}

/// Refuse `ranks` when the most weight bytes one of its ranks holds,
/// `weight_bytes`, leave no room on the device.
pub fn check_fit(ranks: u32, weight_bytes: u64) -> Result<(), String> {
    let needed = device_bytes(weight_bytes);
    if needed > model_compiler_sdk::device::DEVICE_BYTES {
        return Err(format!(
            "the checkpoint does not fit at {ranks} ranks: a rank needs {needed} bytes of \
             weights and host reserve, and a device has {}",
            model_compiler_sdk::device::DEVICE_BYTES
        ));
    }
    Ok(())
}

/// The draft of the package of `geometry` at each of `topologies` (a rank
/// count and the most text weight bytes one of its ranks holds), with its
/// tokenizer `components`, the geometry's frontend and its native module.
/// Components holding the planned encoder image make it a package of image
/// inputs: the frontend declares images, it requires the media capability,
/// and each rank's device memory also holds the encoder
/// ([`media_bytes`]).
pub fn draft(
    geometry: &'static Geometry,
    runtime: RuntimeValues,
    topologies: &[(u32, u64)],
    components: Vec<ComponentDraftV1>,
    native_module: StagedContentV1,
) -> Result<BundleDraftV1, BundleError> {
    let mut components = components;
    let media = components
        .iter()
        .any(|component| component.id().as_str() == crate::encoder_image::IMAGE_COMPONENT);
    let frontend = if media {
        crate::frontend::multimodal_component_for(geometry)
    } else {
        crate::frontend::component_for(geometry)
    }
    .map_err(BundleError::validation)?;
    components.push(package_draft::frontend(StagedContentV1::bytes(frontend))?);
    components.push(ComponentDraftV1::planned(
        id(RANK_IMAGE_COMPONENT)?,
        id(RANK_IMAGE_ROLE)?,
        id(RANK_IMAGE_FORMAT)?,
    )?);
    let ranks: Vec<u32> = topologies.iter().map(|&(ranks, _)| ranks).collect();
    components.push(package_draft::serve_profiles(
        profiles(geometry, &ranks)?.as_bytes(),
    )?);
    components.sort_by(|a, b| a.id().cmp(b.id()));
    let drafts = topologies
        .iter()
        .map(|&(ranks, weight_bytes)| {
            topology_draft(
                geometry,
                ranks,
                weight_bytes,
                media,
                components.clone(),
                native_module.clone(),
            )?
            .with_model_options(runtime_bundle::model_options::ModelOptionsV1 {
                schema: runtime_bundle::model_options::CAPABILITY.into(),
                options: geometry.model_options(runtime),
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    BundleDraftV1::for_topologies(drafts).map_err(|e| BundleError::validation(e.to_string()))
}

/// The per-item state layout of `geometry` in `storage`.
fn state_layout(
    geometry: &Geometry,
    storage: KvStorage,
) -> Result<BundleStateLayoutV1, BundleError> {
    BundleStateLayoutV1::new(
        id(REQUEST_STATE_ID)?,
        slot_state_bytes(geometry, storage),
        256,
        BundleStateLifetimeV1::Request,
    )
}

/// The topology's recurrent state layout: the per-item state, sized for
/// the most a slot holds in either KV format. The host
/// names a slot of it for every request in each batch, which the bundle
/// places the request's state in (`Placement::Host`), and restores prefix
/// snapshots into it. Create allocates every slot's state before the
/// serving KV pool (`chop.recurrent-state.preallocated.v1`), so the host
/// reserves nothing more for it: the pool is sized from what the state
/// leaves. The state rolls back by position, so a verify row needs no
/// copy of it.
pub fn recurrent_state_layout(geometry: &Geometry) -> Result<BundleStateLayoutV1, BundleError> {
    BundleStateLayoutV1::new(
        id(REQUEST_STATE_ID)?,
        [KvStorage::Native, KvStorage::Bf16]
            .map(|storage| slot_state_bytes(geometry, storage))
            .into_iter()
            .max()
            .unwrap_or_default(),
        256,
        BundleStateLifetimeV1::Request,
    )
}

/// The native mixed-plane page layout of `geometry`: its pages as the state
/// contract lays them out (890 bytes a token at flash, 56,960 a page),
/// aligned by the largest power of two up to 256 that divides a page.
pub fn native_layout(geometry: &Geometry) -> Result<BundleKvLayoutV1, BundleError> {
    let page_bytes = request_state(geometry, KvStorage::Native, false).page_bytes();
    let alignment = (page_bytes & page_bytes.wrapping_neg()).min(256);
    BundleKvLayoutV1::new(
        id(std::str::from_utf8(KV_LAYOUT_NATIVE).expect("ASCII layout identifier"))?,
        page_bytes,
        alignment,
        PAGE_TOKENS as u32,
        page_bytes / PAGE_TOKENS,
        1,
        geometry.context_tokens(),
        BundleRankPlacementV1::RankLocal,
        BundleStateLifetimeV1::Transaction,
    )
}

fn topology_draft(
    geometry: &'static Geometry,
    ranks: u32,
    weight_bytes: u64,
    media: bool,
    components: Vec<ComponentDraftV1>,
    native_module: StagedContentV1,
) -> Result<BundleDraftV1, BundleError> {
    admission::check_ranks(geometry, ranks).map_err(BundleError::validation)?;
    let weight_bytes = weight_bytes + media_bytes(media)?;
    check_fit(ranks, weight_bytes).map_err(BundleError::validation)?;
    let native = native_layout(geometry)?;
    let bf16 = package_draft::paged_kv_layout(
        std::str::from_utf8(KV_LAYOUT_BF16).expect("ASCII layout identifier"),
        request_state(geometry, KvStorage::Bf16, false).page_bytes(),
        PAGE_TOKENS,
        geometry.context_tokens(),
    )?;
    let formats = crate::kv::kv_formats(
        geometry,
        native.clone(),
        bf16,
        state_layout(geometry, KvStorage::Bf16)?,
    )?;
    let mut capabilities = CAPABILITIES
        .iter()
        .map(|capability| id(*capability))
        .collect::<Result<Vec<_>, _>>()?;
    if model_deepseek41_contract::serving::default_device_results(geometry, ranks) {
        capabilities.push(id(
            runtime_bundle_abi::device_results::DEVICE_RESULTS_DEFAULT_CAPABILITY_V1,
        )?);
    }
    if media {
        capabilities.push(id(model_deepseek41_contract::encoder::MEDIA_CAPABILITY)?);
    }
    let buckets = package_draft::graph_buckets(
        &components,
        DECODE_BATCH,
        |batch| format!("deepseek41-decode-b{batch}"),
        "deepseek41-prefill",
    )?;
    Ok(BundleDraftV1::new(
        id(BUNDLE_ID)?,
        BundleTargetV1::new(
            vec![id(ARCHITECTURE)?],
            vec![BundleTopologyV1::new(
                id(topology_id(geometry, ranks))?,
                ranks,
            )?],
        )?,
        BundleResourceRequirementsV1::new(device_bytes(weight_bytes), capabilities)?,
        native,
        state_layout(geometry, KvStorage::Native)?,
    )
    .with_kv_formats(formats)
    .with_speculation(
        BundleSpeculationV1::new(
            model_deepseek41_contract::serving::MAX_DRAFT_TOKENS,
            true,
            <model_deepseek41_contract::serving::DeepSeek as bundle_runtime::BundleGeometry>::VERIFY_TOKENS,
            (2..=model_deepseek41_contract::serving::MAX_VERIFY_ROWS).collect(),
            Some(id(RANK_IMAGE_COMPONENT)?),
        )?
        .with_drafter_kind(Some(id("dspark")?))?,
    )
    .with_recurrent_state_layouts(vec![recurrent_state_layout(geometry)?])
    .with_batch_limits(
        BundleBatchLimitsV1::new(DECODE_BATCH, PREFILL_TOKENS)?
            .with_prefill_granule(PREFILL_GRANULE)?
            .with_maximum_scored_rows(SCORED_ROWS)?,
    )
    .with_components(components)
    .with_graph_buckets(buckets)
    .with_native_module(package_draft::native_module(native_module)?))
}
