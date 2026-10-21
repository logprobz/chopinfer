//! Installable Qwen3.8 Flash bundle package of 2 or 4 ranks without rank
//! images, from the identifiers and state sizes in
//! `model_qwen38_contract::contract`: the weight plan loads its images from
//! the checkpoint.
use crate::Result;
use model_compiler_sdk::package_draft::{self, whole_component};
use model_compiler_sdk::{BundleDraftV1, StagedContentV1};
use model_qwen38_contract::backend::{Kernels, Selected};
use model_qwen38_contract::contract::{
    DRAFT_ITEMS, MAX_DRAFT_TOKENS, MAX_VERIFY_TOKENS, PAGE_TOKENS, PREFILL_TOKENS, RANK_IMAGE_ROLE,
    RING_STATE_ID, Shape, VERIFY_ROW_BUCKETS,
};
#[cfg(test)]
use model_qwen38_contract::contract::{RING_STATE_SLOT_BYTES, RING_STATE_SLOT_BYTES_MTP};
use model_qwen38_contract::encoder::{MEDIA_CAPABILITY, POSITION_AXES_CAPABILITY};
use runtime_bundle::{
    BundleBatchLimitsV1, BundleError, BundleOpaqueId, BundleResourceRequirementsV1,
    BundleSpeculationV1, BundleStateLayoutV1, BundleStateLifetimeV1, BundleTargetV1,
    BundleTopologyV1, HOST_RESIDENT_VIEWS_CAPABILITY_V1, SPECULATIVE_CAPABILITY_V1,
};

// The tests name the KV dtypes.
#[cfg(test)]
use runtime_bundle::BundleKvDtypeV1;
use runtime_bundle_abi::{
    PREFIX_CACHE_CAPABILITY_V1, PREFIX_CACHE_STATE_SLOT_CAPABILITY_V1,
    RECURRENT_STATE_CAPABILITY_V1, RECURRENT_STATE_PREALLOCATED_CAPABILITY_V1,
    REQUEST_RETIREMENT_CAPABILITY_V1, SERVED_ITEMS_CAPABILITY_V1, SERVING_KV_CAPABILITY_V1,
    STATE_PER_ITEM_CAPABILITY_V1,
};

pub use model_qwen38_contract::package::{Encoders, minimum_device_memory};

/// Refuses the n-gram table on the GPU where the rank image with it does not
/// leave room for KV on a 96 GB device (TP2,
/// `model_qwen38_contract::package::ngram_in_host_memory`).
pub fn check_fits(shape: Shape, image_bytes: u64, ngram: NgramPlacement) -> Result<()> {
    use model_qwen38_contract::package::{DEVICE_BYTES, ngram_in_host_memory};
    let gib = |bytes: u64| bytes as f64 / f64::from(1 << 30);
    if ngram == NgramPlacement::Gpu && ngram_in_host_memory(shape) {
        return Err(format!(
            "TP{} needs {:.1} GiB of weights per rank on a {:.0} GiB device, which leaves \
             no room for KV and scratch with the n-gram table on the GPU; the weight plan \
             keeps it in host memory where the device capacity rule places it",
            shape.ranks(),
            gib(image_bytes),
            gib(DEVICE_BYTES)
        )
        .into());
    }
    Ok(())
}
pub use model_qwen38_contract::package::{
    RANK_IMAGE_ALIGNMENT, RANK_IMAGE_FORMAT, RANK_INDEX_FORMAT,
};
/// The host's prompt-logprob rows (`runtime-host`'s `PROMPT_ROWS_CAPABILITY_V1`).
pub use runtime_bundle_abi::PROMPT_ROWS_CAPABILITY_V2;

/// The drafter a package declares. The MTP drafter's weights are views of
/// the main rank images, so it adds no component.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Drafter {
    #[default]
    None,
    Mtp,
}

impl std::str::FromStr for Drafter {
    type Err = String;
    fn from_str(s: &str) -> std::result::Result<Self, String> {
        match s {
            "none" => Ok(Self::None),
            "mtp" => Ok(Self::Mtp),
            _ => Err(format!("drafter must be none or mtp, not {s}")),
        }
    }
}

/// Where the package places each rank's n-gram table (11.92 GiB per rank
/// at TP4). The GPU is the default. `host` keeps it in page-locked,
/// device-mapped host memory, which frees that device memory for KV and
/// needs as much pinned host RAM (docs/developer-preview.md).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum NgramPlacement {
    #[default]
    Gpu,
    Host,
}

impl std::str::FromStr for NgramPlacement {
    type Err = String;
    fn from_str(s: &str) -> std::result::Result<Self, String> {
        match s {
            "gpu" => Ok(Self::Gpu),
            "host" => Ok(Self::Host),
            _ => Err(format!("n-gram placement must be gpu or host, not {s}")),
        }
    }
}

/// The rank-image component, which also holds the MTP drafter's weights.
pub const RANK_IMAGE_COMPONENT: &str = "rank-images";

/// What a package carries besides its planned images: the weight plan
/// supplies the rank images and indexes (and the encoder images and indexes)
/// from the checkpoint.
pub struct PackageAssets {
    /// The topology the package serves.
    pub shape: Shape,
    pub runtime: crate::config::RuntimeValues,
    pub drafter: Drafter,
    pub tokenizer_definition: StagedContentV1,
    pub tokenizer_configuration: StagedContentV1,
    pub native_module: StagedContentV1,
    /// The planned vision encoders, which add image inputs.
    pub encoders: Option<crate::encoders::EncoderAssets>,
}

pub struct PreparedAssets {
    pub assets: PackageAssets,
    pub frontend: Vec<u8>,
    pub precision: kernel_core::precision::execution::PrecisionDocument,
}

/// The serving profiles and deployment policy every package carries.
pub(crate) const PROFILES: &str = include_str!("../../profiles.json");
pub(crate) const DEPLOYMENT_POLICY: &str = include_str!("../../deploy/policy.json");

fn id(value: &str) -> std::result::Result<BundleOpaqueId, BundleError> {
    BundleOpaqueId::new(value)
}

fn utf8(value: &[u8]) -> &str {
    std::str::from_utf8(value).expect("contract identifiers are ASCII")
}

pub use model_qwen38_contract::package::MTP_QUALIFIED_SEQUENCES;

/// The most sequences a package serves
/// (`model_qwen38_contract::package::maximum_sequences`).
pub fn maximum_sequences(shape: Shape, drafter: Drafter, encoders: Encoders) -> u32 {
    model_qwen38_contract::package::maximum_sequences(shape, drafter == Drafter::Mtp, encoders)
}

/// A package without rank images, which serves by loading the checkpoint
/// directly: the bundle's weight plan supplies the rank images, declared as
/// a planned component, and the rank indexes, with or without the MTP
/// drafter, whose views the rank images carry. With encoders, which only
/// `EncoderAssets::planned` makes after checking the checkpoint's vision
/// tower, the plan also supplies the encoder images, another planned
/// component, and their indexes.
///
/// Text serving: serving KV, per-request linear and PLE state with
/// retirement, the QSA ring as process state sized per served sequence,
/// decode graph buckets 1 to the package maximum of 256, scored prompt rows
/// and prefix caching (snapshots restore into the request's state slot).
/// Without a drafter it serves autoregressively; with MTP, pages and the
/// ring also hold the drafter's layer and the bundle drafts and verifies
/// speculation. With the encoders, also image inputs with 3-axis positions.
pub fn bundle_draft_direct(
    assets: PackageAssets,
    local_frontend: Option<Vec<u8>>,
    precision: Option<&kernel_core::precision::execution::PrecisionDocument>,
) -> std::result::Result<BundleDraftV1, BundleError> {
    if assets
        .encoders
        .as_ref()
        .is_some_and(|encoders| encoders.checked_tower.is_none())
    {
        return Err(BundleError::validation(
            "a package without rank images plans its encoders only for a checkpoint whose \
             vision tower was checked (EncoderAssets::planned)",
        ));
    }
    let shape = assets.shape;
    let ranks = shape.ranks();
    let topology = shape.topology().id;
    let media = assets.encoders.is_some();
    let encoders = Encoders { vision: media };
    // The plan places the n-gram table in host memory where the capacity
    // rule does.
    let host_ngram = model_qwen38_contract::package::ngram_in_host_memory(shape);
    let positions = assets.encoders.as_ref().map(|encoders| encoders.positions);
    let frontend = match (local_frontend, positions) {
        (Some(bytes), Some(positions)) => {
            let mut local: serde_json::Value = serde_json::from_slice(&bytes)
                .map_err(|e| BundleError::validation(e.to_string()))?;
            let media: serde_json::Value = serde_json::from_slice(
                &crate::frontend::multimodal_component_for(shape.geometry, positions),
            )
            .map_err(|e| BundleError::validation(e.to_string()))?;
            local["multimodal"] = media["multimodal"].clone();
            serde_json::to_vec(&local).map_err(|e| BundleError::validation(e.to_string()))?
        }
        (Some(bytes), None) => bytes,
        (None, Some(positions)) => {
            crate::frontend::multimodal_component_for(shape.geometry, positions)
        }
        (None, None) => crate::frontend::COMPONENT.to_vec(),
    };
    let frontend = crate::frontend::component_for_geometry(&frontend, shape.geometry)
        .map_err(|error| BundleError::validation(error.to_string()))?;
    // The weight plan supplies the rank images: the component is declared
    // by role and format, with no file, so the host asks the plan for it and
    // the MTP drafter can name it.
    let mut components = vec![model_compiler_sdk::ComponentDraftV1::planned(
        id(RANK_IMAGE_COMPONENT)?,
        id(utf8(RANK_IMAGE_ROLE))?,
        id(RANK_IMAGE_FORMAT)?,
    )?];
    components.extend([
        package_draft::frontend(StagedContentV1::bytes(frontend))?,
        package_draft::tokenizer_definition(assets.tokenizer_definition)?,
        package_draft::tokenizer_configuration(assets.tokenizer_configuration)?,
        package_draft::serve_profiles(PROFILES.as_bytes())?,
        package_draft::deployment_policy(DEPLOYMENT_POLICY.as_bytes())?,
    ]);
    if let Some(precision) = precision {
        if assets.drafter != Drafter::None
            && let Err(error) = model_qwen38_contract::projections::support::mtp(shape, precision)
        {
            return Err(BundleError::validation(format!(
                "prepared mixed precision does not run the MTP drafter: {error}"
            )));
        }
        let bytes = model_compiler_sdk::quantization::identity::encode_component(precision)
            .map_err(|e| BundleError::validation(e.to_string()))?;
        // Each topology's operations are its own: its component serves it
        // alone.
        components.push(
            whole_component(
                &model_qwen38_contract::prepared::component_id(shape),
                model_qwen38_contract::prepared::COMPONENT_ROLE,
                model_qwen38_contract::prepared::COMPONENT_FORMAT,
                &model_qwen38_contract::prepared::component_path(shape),
                StagedContentV1::bytes(bytes),
            )?
            .with_topologies(vec![id(topology)?]),
        );
    }
    let drafter = assets.drafter;
    if media {
        components.push(crate::encoders::component()?);
    }
    let sequences = maximum_sequences(shape, drafter, encoders);
    let buckets = package_draft::graph_buckets(
        &components,
        sequences,
        |batch| format!("qwen38-decode-b{batch}"),
        "qwen38-prefill",
    )?;
    let recurrent = shape
        .recurrent_state_layouts()
        .iter()
        .map(|(layout, bytes)| {
            // 128 divides every layout: the PLE entry is 1441 blocks of 128 bytes.
            BundleStateLayoutV1::new(id(layout)?, *bytes, 128, BundleStateLifetimeV1::Request)
        })
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let mut capabilities = vec![
        id(SERVING_KV_CAPABILITY_V1)?,
        id(RECURRENT_STATE_CAPABILITY_V1)?,
        // The linear-attention and PLE pools are allocated for every slot
        // during create, before the serving KV pool.
        id(RECURRENT_STATE_PREALLOCATED_CAPABILITY_V1)?,
        id(REQUEST_RETIREMENT_CAPABILITY_V1)?,
        id(PROMPT_ROWS_CAPABILITY_V2)?,
        id(SPECULATIVE_CAPABILITY_V1)?,
        // Restore writes the snapshot into the request's state slot.
        id(PREFIX_CACHE_CAPABILITY_V1)?,
        id(runtime_bundle_abi::KV_OFFLOAD_CAPABILITY_V1)?,
        id(PREFIX_CACHE_STATE_SLOT_CAPABILITY_V1)?,
        // The module reads the served item count from the create record.
        id(SERVED_ITEMS_CAPABILITY_V1)?,
    ];
    // Both layouts declare aligned ring bytes per served state slot.
    capabilities.push(id(STATE_PER_ITEM_CAPABILITY_V1)?);
    capabilities.push(id(
        runtime_bundle_abi::RECURRENT_STATE_IN_PLACE_CAPABILITY_V1,
    )?);
    capabilities.push(id(
        runtime_bundle_abi::SPECULATIVE_NARROW_ROWS_CAPABILITY_V1,
    )?);
    if media {
        capabilities.push(id(MEDIA_CAPABILITY)?);
    }
    if positions == Some(crate::encoders::ImagePositions::Grid3Axis) {
        capabilities.push(id(POSITION_AXES_CAPABILITY)?);
    }
    if host_ngram {
        capabilities.push(id(HOST_RESIDENT_VIEWS_CAPABILITY_V1)?);
    }
    capabilities.push(id(
        runtime_bundle_abi::weight_plan::WEIGHT_PLAN_CAPABILITY_V1,
    )?);
    capabilities.push(id(runtime_bundle_abi::frontend::FRONTEND_CAPABILITY_V1)?);
    // Per state slot (`chop.state-per-item.v1`): the host allocates it for
    // the served count rounded up to a power of two, with aligned bytes
    // for the main layers and, when enabled, the drafter layer.
    let ring_bytes = shape.ring_state_slot_bytes(drafter == Drafter::Mtp);
    // Every KV layout of this drafter: BF16, the preferred `kv_layout`, and
    // FP8 E4M3 main full layers with the MTP layer BF16; the host picks one
    // at startup with `--kv-cache-dtype`.
    let kv_layout = |format: usize| {
        let formats = shape.kv_formats();
        let format = &formats[usize::from(drafter == Drafter::Mtp) + 2 * format];
        package_draft::paged_kv_layout(
            utf8(format.layout_id),
            format.page_bytes,
            PAGE_TOKENS,
            shape.context_tokens(),
        )
    };
    let kv_formats = package_draft::bf16_and_fp8_kv(kv_layout(0)?, kv_layout(1)?);
    let draft = BundleDraftV1::new(
        id(model_qwen38_contract::contract::BUNDLE_ID)?,
        BundleTargetV1::new(
            vec![id(Selected::ARCHITECTURE)?],
            // The context-parallel sizes above 1 the bundle serves at this
            // shape (`context_parallel::sizes`, which the
            // contract's declaration, the cubin build and create read).
            vec![
                BundleTopologyV1::new(id(topology)?, ranks)?.with_context_parallel_sizes(
                    model_qwen38_contract::context_parallel::sizes(shape).to_vec(),
                )?,
            ],
        )?,
        BundleResourceRequirementsV1::new(
            // With media each rank also holds the encoder image, buffers
            // and arena.
            minimum_device_memory(shape, drafter == Drafter::Mtp, encoders),
            capabilities,
        )?,
        kv_layout(0)?,
        BundleStateLayoutV1::new(
            id(utf8(RING_STATE_ID))?,
            ring_bytes,
            256,
            BundleStateLifetimeV1::Process,
        )?,
    )
    .with_kv_formats(kv_formats)
    .with_recurrent_state_layouts(recurrent)
    .with_batch_limits(
        BundleBatchLimitsV1::new(sequences, PREFILL_TOKENS)?
            .with_maximum_scored_rows(model_qwen38_contract::contract::SCORED_ROWS)?
            .with_prefill_granule(model_qwen38_contract::contract::PREFILL_GRANULE)?
            // A continuation's first MTP drafter row reads its next token, so
            // it rewrites the anchor's last row (prefix_state.rs).
            .with_last_page_rewrites(drafter == Drafter::Mtp)
            .with_anchor_items(model_qwen38_contract::contract::ANCHOR_ITEMS)?,
    )
    .with_components(components)
    .with_graph_buckets(buckets)
    .with_native_module(package_draft::native_module(assets.native_module)?);
    // Without the drafter the host drafts; serving enables speculation only
    // with --speculative-tokens.
    let bundle_drafter = drafter == Drafter::Mtp;
    // The per-step checkpoint pool holds at most `DRAFT_ITEMS` drafting items.
    let draft = draft.with_speculation(
        BundleSpeculationV1::new(
            MAX_DRAFT_TOKENS,
            bundle_drafter,
            MAX_VERIFY_TOKENS,
            verify_row_buckets(
                bundle_drafter,
                std::env::var(WIDE_ROWS_ENV).as_deref() == Ok("1"),
            ),
            bundle_drafter
                .then(|| id(RANK_IMAGE_COMPONENT))
                .transpose()?,
        )?
        .with_drafter_kind(bundle_drafter.then(|| id("mtp")).transpose()?)?
        .with_maximum_draft_items(DRAFT_ITEMS)?,
    );
    let draft = with_model_options(draft, shape.geometry, assets.runtime)?;
    if !media && drafter == Drafter::None {
        let declarations =
            model_qwen38_contract::package::without_rank_images(if precision.is_some() {
                model_qwen38_contract::package::prepared_declarations(
                    shape,
                    host_ngram,
                    assets.runtime,
                )
            } else {
                model_qwen38_contract::package::declarations(shape, host_ngram, assets.runtime)
            });
        let draft = draft
            .with_native_declarations(&declarations)
            .map_err(|e| BundleError::validation(e.to_string()))?;
        return Ok(draft);
    }
    Ok(draft)
}

fn with_model_options(
    draft: BundleDraftV1,
    geometry: &model_qwen38_contract::geometry::Geometry,
    runtime: crate::config::RuntimeValues,
) -> std::result::Result<BundleDraftV1, BundleError> {
    let options = geometry.model_options(runtime);
    if options.is_empty() {
        Ok(draft)
    } else {
        draft.with_model_options(runtime_bundle::model_options::ModelOptionsV1 {
            schema: runtime_bundle::model_options::CAPABILITY.into(),
            options,
        })
    }
}

/// Set to `1`, packages with the MTP drafter declare the wide experiment's
/// verify row buckets ([`verify_row_buckets`]).
pub const WIDE_ROWS_ENV: &str = "CHOPINFER_SPECULATIVE_WIDE_ROWS";

/// Sequential MTP can emit two drafts without padding the item to K3.
/// Declare that depth only in packages prepared for the wide experiment.
pub(crate) fn verify_row_buckets(bundle_drafter: bool, wide: bool) -> Vec<u32> {
    if bundle_drafter && wide {
        vec![2, 3, 4, 8]
    } else {
        VERIFY_ROW_BUCKETS.to_vec()
    }
}

#[cfg(test)]
#[path = "package_tests.rs"]
mod tests;
