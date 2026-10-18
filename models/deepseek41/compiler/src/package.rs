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

use model_compiler_sdk::{BundleDraftV1, ComponentDraftV1, StagedContentV1, package_draft};
use model_deepseek41_contract::geometry::Geometry;
use model_deepseek41_contract::serving::{
    DECODE_BATCH, KV_LAYOUT_BF16, KV_LAYOUT_NATIVE, PAGE_TOKENS, PREFILL_GRANULE, PREFILL_TOKENS,
    RANK_IMAGE_ROLE, REQUEST_STATE_ID, SCORED_ROWS,
};
use model_deepseek41_contract::state::{KvStorage, request_state};
use model_deepseek41_contract::{BUNDLE_ID, admission};
use runtime_bundle::{
    BundleBatchLimitsV1, BundleError, BundleKvLayoutV1, BundleOpaqueId, BundleRankPlacementV1,
    BundleResourceRequirementsV1, BundleStateLayoutV1, BundleStateLifetimeV1, BundleTargetV1,
    BundleTopologyV1,
};

/// The device architecture the native module's kernels are built for.
pub const ARCHITECTURE: &str = "sm_120f";

/// The planned text image's component id and format.
pub const RANK_IMAGE_COMPONENT: &str = "rank-images";
pub const RANK_IMAGE_FORMAT: &str = "deepseek41.rank-image.v1";

/// The serving profiles, which every package carries.
pub const PROFILES: &str = include_str!("../../profiles.json");

/// What the package requires of the host, from what the module exports:
/// paged serving KV with per-item state, request retirement, scored prompt
/// rows, prefix snapshots restored into the state slot, and the planned
/// text image.
pub const CAPABILITIES: [&str; 9] = [
    runtime_bundle_abi::SERVING_KV_CAPABILITY_V1,
    runtime_bundle_abi::REQUEST_RETIREMENT_CAPABILITY_V1,
    runtime_bundle_abi::PROMPT_ROWS_CAPABILITY_V2,
    runtime_bundle_abi::PREFIX_CACHE_CAPABILITY_V1,
    runtime_bundle_abi::PREFIX_CACHE_STATE_SLOT_CAPABILITY_V1,
    runtime_bundle_abi::SERVED_ITEMS_CAPABILITY_V1,
    runtime_bundle_abi::SERVED_CONTEXT_CAPABILITY_V1,
    runtime_bundle_abi::STATE_PER_ITEM_CAPABILITY_V1,
    runtime_bundle_abi::weight_plan::WEIGHT_PLAN_CAPABILITY_V1,
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
/// without the drafter, in whole 256-byte blocks: what create checks
/// against the served item count.
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
/// count and the most weight bytes one of its ranks holds), with its
/// tokenizer `components` and its native module.
pub fn draft(
    geometry: &'static Geometry,
    topologies: &[(u32, u64)],
    components: Vec<ComponentDraftV1>,
    native_module: StagedContentV1,
) -> Result<BundleDraftV1, BundleError> {
    let mut components = components;
    components.push(ComponentDraftV1::planned(
        id(RANK_IMAGE_COMPONENT)?,
        id(RANK_IMAGE_ROLE)?,
        id(RANK_IMAGE_FORMAT)?,
    )?);
    components.push(package_draft::serve_profiles(PROFILES.as_bytes())?);
    components.sort_by(|a, b| a.id().cmp(b.id()));
    let drafts = topologies
        .iter()
        .map(|&(ranks, weight_bytes)| {
            topology_draft(
                geometry,
                ranks,
                weight_bytes,
                components.clone(),
                native_module.clone(),
            )
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
    components: Vec<ComponentDraftV1>,
    native_module: StagedContentV1,
) -> Result<BundleDraftV1, BundleError> {
    admission::check_ranks(geometry, ranks).map_err(BundleError::validation)?;
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
    let capabilities = CAPABILITIES
        .iter()
        .map(|capability| id(*capability))
        .collect::<Result<Vec<_>, _>>()?;
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
    .with_batch_limits(
        BundleBatchLimitsV1::new(DECODE_BATCH, PREFILL_TOKENS)?
            .with_prefill_granule(PREFILL_GRANULE)?
            .with_maximum_scored_rows(SCORED_ROWS)?,
    )
    .with_components(components)
    .with_graph_buckets(buckets)
    .with_native_module(package_draft::native_module(native_module)?))
}
