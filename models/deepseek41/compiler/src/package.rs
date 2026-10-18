//! The `deepseek41.v1` package draft: a direct-loading package whose text
//! rank image is planned from the checkpoint (`chop.weight-plan.v1`,
//! `model_deepseek41_contract::weight_plan`).
//!
//! One package serves every rank count of `admission::SUPPORTED_RANKS` its
//! geometry divides over and its weights fit, each as a topology with its
//! own resources: the rank's planned weight bytes and the host reserve, the
//! native KV layout's pages and the per-item state of that layout. The
//! package serves autoregressively; the BF16 KV layout and the DSpark
//! drafter need their own state bytes, which one state layout per topology
//! cannot declare beside the native one.

use model_compiler_sdk::{BundleDraftV1, ComponentDraftV1, StagedContentV1, package_draft};
use model_deepseek41_contract::geometry::Geometry;
use model_deepseek41_contract::serving::{
    DECODE_BATCH, KV_LAYOUT_NATIVE, PAGE_TOKENS, PREFILL_GRANULE, PREFILL_TOKENS, RANK_IMAGE_ROLE,
    REQUEST_STATE_ID, SCORED_ROWS,
};
use model_deepseek41_contract::state::{KvStorage, request_state};
use model_deepseek41_contract::{BUNDLE_ID, admission};
use runtime_bundle::{
    BundleBatchLimitsV1, BundleError, BundleOpaqueId, BundleResourceRequirementsV1,
    BundleStateLayoutV1, BundleStateLifetimeV1, BundleTargetV1, BundleTopologyV1,
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

/// The KV storage every topology declares.
const STORAGE: KvStorage = KvStorage::Native;

fn id(value: impl Into<String>) -> Result<BundleOpaqueId, BundleError> {
    BundleOpaqueId::new(value)
}

/// The topology ID of `geometry` at `ranks`: `deepseek41-<geometry>-tp<ranks>`,
/// the bundle's `shape::topology_id`.
pub fn topology_id(geometry: &Geometry, ranks: u32) -> String {
    format!("deepseek41-{}-tp{ranks}", geometry.id)
}

/// The per-item state bytes of one state slot of `geometry` in the native
/// storage without the drafter, in whole 256-byte blocks: what create
/// checks against the served item count.
pub fn slot_state_bytes(geometry: &Geometry) -> u64 {
    let slot = bundle_runtime::state::Extent {
        served: 1,
        bucket: 1,
        verify_rows: 0,
    };
    request_state(geometry, STORAGE, false)
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

fn topology_draft(
    geometry: &'static Geometry,
    ranks: u32,
    weight_bytes: u64,
    components: Vec<ComponentDraftV1>,
    native_module: StagedContentV1,
) -> Result<BundleDraftV1, BundleError> {
    admission::check_ranks(geometry, ranks).map_err(BundleError::validation)?;
    check_fit(ranks, weight_bytes).map_err(BundleError::validation)?;
    let page_bytes = request_state(geometry, STORAGE, false).page_bytes();
    let kv = package_draft::paged_kv_layout(
        std::str::from_utf8(KV_LAYOUT_NATIVE).expect("ASCII layout identifier"),
        page_bytes,
        PAGE_TOKENS,
        geometry.context_tokens(),
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
        kv,
        BundleStateLayoutV1::new(
            id(REQUEST_STATE_ID)?,
            slot_state_bytes(geometry),
            256,
            BundleStateLifetimeV1::Request,
        )?,
    )
    .with_batch_limits(
        BundleBatchLimitsV1::new(DECODE_BATCH, PREFILL_TOKENS)?
            .with_prefill_granule(PREFILL_GRANULE)?
            .with_maximum_scored_rows(SCORED_ROWS)?,
    )
    .with_components(components)
    .with_graph_buckets(buckets)
    .with_native_module(package_draft::native_module(native_module)?))
}
