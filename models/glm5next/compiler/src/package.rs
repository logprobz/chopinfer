//! The `glm5next.v1` package draft: a direct-loading package whose text
//! rank image is planned from the checkpoint (`chop.weight-plan.v1`).
//!
//! Its KV formats are the latent pages in FP8 (preferred) and BF16. The
//! KDA convolution and state are the recurrent state layouts, which create
//! allocates before the KV pool with their verify checkpoints and anchor
//! spares (`chop.recurrent-state.preallocated.v1`); the indexer tail with
//! its verify checkpoints is the per-item state layout
//! (`chop.state-per-item.v1`).
//! The topology declares its context-parallel sizes and the package its
//! model options (the checkpoint's quantization layout, with that one
//! value, and `text_config.index_topk`) and its speculation: the MTP
//! drafter over the KV layouts with its sparse layer, by default declaring
//! the contract's `MAX_DRAFT_TOKENS` drafts, or host drafts. The served
//! `--speculative-tokens` picks how many of them a server drafts (three for
//! `tp4-mtp`, five for the experimental `tp4-mtp-k5`), and the bundle
//! allocates state and the host captures verify graphs for that served
//! count, not the package's maximum. With the planned encoder image
//! (`crate::encoder_image`) it also takes image inputs.

use model_compiler_sdk::{BundleDraftV1, ComponentDraftV1, StagedContentV1, package_draft};
use model_glm5next_contract::backend::{Kernels, Selected};
use model_glm5next_contract::contract::{
    ANCHOR_ITEMS, BUNDLE_ID, CAPABILITIES, DECODE_BATCH, DRAFT_ITEMS, INDEXER_TAIL_STATE_ID,
    MAX_DRAFT_TOKENS, PAGE_TOKENS, PREFILL_GRANULE, PREFILL_TOKENS, Shape, bundle_topology,
};
use model_glm5next_contract::direct::RANK_IMAGE_ROLE;
use model_glm5next_contract::geometry::RuntimeValues;
use model_glm5next_contract::quantization::Layout;
use runtime_bundle::{
    BundleBatchLimitsV1, BundleError, BundleOpaqueId, BundleResourceRequirementsV1,
    BundleSpeculationV1, BundleStateLayoutV1, BundleStateLifetimeV1, BundleTargetV1,
};

/// The planned text image's component id and format.
pub const RANK_IMAGE_COMPONENT: &str = "rank-images";
pub const RANK_IMAGE_FORMAT: &str = "glm5next.rank-image.v1";

/// The serving profiles of every checkpoint layout and the deployment
/// policy, which every package carries.
pub const PROFILES: &str = include_str!("../../profiles.json");
pub const DEPLOYMENT_POLICY: &str = include_str!("../../deploy/policy.json");

/// The precision component's id and path: the lowered projections'
/// document (`model_glm5next_contract::prepared`), which create
/// authenticates and binds every rank's projection store from.
pub const PRECISION_COMPONENT: &str = "precision";
pub const PRECISION_PATH: &str = "components/precision.bin";

/// The precision component of `bytes`
/// (`model_glm5next_contract::prepared::checkpoint_component`) for a
/// checkpoint stored in `layout`, refused unless it decodes to a document
/// serving every lowered projection from that layout's tensors.
pub fn precision(
    bytes: Vec<u8>,
    layout: &model_glm5next_contract::quantization::Layout,
) -> Result<ComponentDraftV1, BundleError> {
    model_glm5next_contract::prepared::decode_for(&bytes, layout)
        .map_err(BundleError::validation)?;
    package_draft::whole_component(
        PRECISION_COMPONENT,
        model_glm5next_contract::prepared::COMPONENT_ROLE,
        model_glm5next_contract::prepared::COMPONENT_FORMAT,
        PRECISION_PATH,
        StagedContentV1::bytes(bytes),
    )
}

/// The drafter a package declares. The MTP drafter's weights are views of
/// the rank images (layer 45), so it adds no component; its latent KV is a
/// twelfth sparse layer of the pages.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Drafter {
    /// Host drafts of up to `MAX_DRAFT_TOKENS` tokens.
    None,
    /// The MTP drafter making up to `drafts` drafts per verify item.
    Mtp { drafts: u32 },
}

/// The MTP drafter at the contract's maximum, so one package serves every
/// profile's draft count.
impl Default for Drafter {
    fn default() -> Self {
        Self::Mtp {
            drafts: MAX_DRAFT_TOKENS,
        }
    }
}

impl Drafter {
    /// Drafts per verify item the package declares.
    fn drafts(self) -> u32 {
        match self {
            Self::None => MAX_DRAFT_TOKENS,
            Self::Mtp { drafts } => drafts,
        }
    }
}

fn id(value: impl Into<String>) -> Result<BundleOpaqueId, BundleError> {
    BundleOpaqueId::new(value)
}

fn utf8(bytes: &[u8]) -> &str {
    std::str::from_utf8(bytes).expect("ASCII contract identifier")
}

/// The package draft of `shape` with its checkpoint's quantization layout
/// (which sets the device memory the weights need) and runtime values, its
/// frontend and tokenizer components, its native module and its drafter.
/// Every package carries the serving profiles and deployment policy.
/// Components holding the planned encoder image make it a package of image
/// inputs: it requires the media capability, and each rank's device memory
/// also holds the encoder's buffers (its image is the vision tower's share
/// of the checkpoint weights, counted already).
pub fn draft(
    shape: Shape,
    quantization: &Layout,
    runtime: RuntimeValues,
    mut components: Vec<ComponentDraftV1>,
    native_module: StagedContentV1,
    drafter: Drafter,
) -> Result<BundleDraftV1, BundleError> {
    let mtp = matches!(drafter, Drafter::Mtp { .. });
    let media = components
        .iter()
        .any(|component| component.id().as_str() == crate::encoder_image::IMAGE_COMPONENT);
    if !(1..=MAX_DRAFT_TOKENS).contains(&drafter.drafts()) {
        return Err(BundleError::validation(format!(
            "the MTP drafter makes 1 to {MAX_DRAFT_TOKENS} drafts, not {}",
            drafter.drafts()
        )));
    }
    components.push(ComponentDraftV1::planned(
        id(RANK_IMAGE_COMPONENT)?,
        id(RANK_IMAGE_ROLE)?,
        id(RANK_IMAGE_FORMAT)?,
    )?);
    components.push(package_draft::serve_profiles(PROFILES.as_bytes())?);
    components.push(package_draft::deployment_policy(
        DEPLOYMENT_POLICY.as_bytes(),
    )?);
    components.sort_by(|a, b| a.id().cmp(b.id()));
    let formats = shape.kv_formats();
    let layout = |index: usize| {
        let format = &formats[index];
        package_draft::paged_kv_layout(
            utf8(format.layout_id),
            format.page_bytes,
            PAGE_TOKENS,
            shape.geometry.context_tokens(),
        )
    };
    // FP8 is the preferred layout; the host picks one with
    // `--kv-cache-dtype`. With the MTP drafter each page also holds its
    // sparse layer.
    let (fp8, bf16) = if mtp { (1, 3) } else { (0, 2) };
    let kv_formats = package_draft::bf16_and_fp8_kv(layout(bf16)?, layout(fp8)?);
    let recurrent = shape
        .recurrent_state_layouts()
        .iter()
        .map(|(layout, bytes)| {
            BundleStateLayoutV1::new(id(*layout)?, *bytes, 128, BundleStateLifetimeV1::Request)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut capabilities = CAPABILITIES
        .iter()
        .map(|capability| id(*capability))
        .collect::<Result<Vec<_>, _>>()?;
    if media {
        capabilities.push(id(model_glm5next_contract::encoder::MEDIA_CAPABILITY)?);
    }
    let minimum_device_memory = model_glm5next_contract::fit::checkpoint_weight_bytes(
        shape.geometry,
        quantization,
        shape.ranks(),
    )
    .map_err(BundleError::validation)?
        + model_glm5next_contract::fit::HOST_RESERVE_BYTES
        + if media {
            model_glm5next_contract::encoder::buffer_bytes()
        } else {
            0
        };
    let buckets = package_draft::graph_buckets(
        &components,
        DECODE_BATCH,
        |batch| format!("glm5next-decode-b{batch}"),
        "glm5next-prefill",
    )?;
    BundleDraftV1::new(
        id(BUNDLE_ID)?,
        BundleTargetV1::new(vec![id(Selected::ARCHITECTURE)?], vec![bundle_topology()?])?,
        BundleResourceRequirementsV1::new(minimum_device_memory, capabilities)?,
        layout(fp8)?,
        BundleStateLayoutV1::new(
            id(INDEXER_TAIL_STATE_ID)?,
            shape.indexer_tail_slot_bytes(mtp),
            256,
            BundleStateLifetimeV1::Process,
        )?,
    )
    .with_kv_formats(kv_formats)
    .with_recurrent_state_layouts(recurrent)
    .with_batch_limits(
        BundleBatchLimitsV1::new(DECODE_BATCH, PREFILL_TOKENS)?
            .with_prefill_granule(PREFILL_GRANULE)?
            .with_anchor_items(ANCHOR_ITEMS)?
            .with_maximum_scored_rows(model_glm5next_contract::contract::SCORED_ROWS)?
            // A continuation's first drafter row reads its next token, so it
            // rewrites the anchor's last row.
            .with_last_page_rewrites(mtp),
    )
    .with_components(components)
    .with_graph_buckets(buckets)
    .with_native_module(package_draft::native_module(native_module)?)
    // The MTP drafter drafts in the bundle, which verifies the drafts and
    // commits the accepted rows; without it the host drafts. Serving enables
    // speculation with --speculative-tokens.
    .with_speculation(
        BundleSpeculationV1::new(
            drafter.drafts(),
            mtp,
            model_glm5next_contract::contract::head_rows(),
            (2..=drafter.drafts() + 1).collect(),
            mtp.then(|| id(RANK_IMAGE_COMPONENT)).transpose()?,
        )?
        .with_drafter_kind(mtp.then(|| id("mtp")).transpose()?)?
        .with_maximum_draft_items(DRAFT_ITEMS)?,
    )
    .with_model_options(runtime_bundle::model_options::ModelOptionsV1 {
        schema: runtime_bundle::model_options::CAPABILITY.into(),
        options: options(shape, quantization, runtime),
    })
}

/// The package's model options in name order: the layout (one allowed
/// value), the Sinkhorn mHC comm epilogue switch (off by default) and the
/// overridable runtime fields.
fn options(
    shape: Shape,
    quantization: &Layout,
    runtime: RuntimeValues,
) -> Vec<runtime_bundle::model_options::ModelOptionV1> {
    let mut options = vec![
        quantization.model_option(),
        model_glm5next_contract::options::comm_epilogue_option(),
    ];
    options.extend(shape.geometry.model_options(runtime));
    options.sort_by(|a, b| a.name.cmp(&b.name));
    options
}

#[cfg(test)]
mod tests {
    use super::*;
    use model_glm5next_contract::quantization::{
        FP8_BLOCK128, LAYOUT_OPTION, LAYOUTS, NVFP4_DENSE_ROUTED, NVFP4_ROUTED,
    };
    use serde_json::Value;

    fn manifest() -> Value {
        manifest_of(&FP8_BLOCK128, Drafter::default())
    }

    fn manifest_with(drafter: Drafter) -> Value {
        manifest_of(&FP8_BLOCK128, drafter)
    }

    fn manifest_of(quantization: &Layout, drafter: Drafter) -> Value {
        let mut components = vec![
            package_draft::frontend(StagedContentV1::bytes(crate::frontend::COMPONENT)).unwrap(),
            package_draft::tokenizer_definition(StagedContentV1::bytes(b"{}".to_vec())).unwrap(),
            package_draft::tokenizer_configuration(StagedContentV1::bytes(b"{}".to_vec())).unwrap(),
        ];
        if *quantization == FP8_BLOCK128 {
            components.push(
                precision(
                    model_glm5next_contract::prepared::test_component(),
                    quantization,
                )
                .unwrap(),
            );
        }
        let runtime = RuntimeValues { index_topk: 2048 };
        let draft = draft(
            Shape::TP4,
            quantization,
            runtime,
            components,
            StagedContentV1::bytes(b"module".to_vec()),
            drafter,
        )
        .unwrap();
        let manifest = model_compiler_sdk::plan_bundle_manifest_v1(&draft).unwrap();
        serde_json::to_value(&manifest).unwrap()
    }

    /// The weights of the layout and the host reserve.
    #[test]
    fn the_device_memory_follows_the_quantization_layout() {
        let minimum = |layout| {
            manifest_of(layout, Drafter::default())["resource_requirements"]["minimum_device_memory_bytes"]
                .as_u64()
                .unwrap()
        };
        assert_eq!(minimum(&FP8_BLOCK128), 83_370_801_144 + (1 << 30));
        assert_eq!(minimum(&NVFP4_ROUTED), 50_875_242_744 + (1 << 30));
        assert_eq!(minimum(&NVFP4_DENSE_ROUTED), 52_469_433_888 + (1 << 30));
    }

    #[test]
    fn the_manifest_declares_state_granule_topology_and_options() {
        let manifest = manifest();
        let shape = Shape::TP4;
        assert_eq!(manifest["id"], BUNDLE_ID);
        assert_eq!(
            manifest["target"]["topologies"][0]["context_parallel_sizes"],
            serde_json::json!([2, 4])
        );
        assert_eq!(manifest["batch_limits"]["prefill_granule"], 64);
        // Scored prefill rows on the granule, which the host's scheduler
        // requires of a bounded limit; verify rows apart.
        let scored = &manifest["batch_limits"]["maximum_scored_rows"];
        assert_eq!(scored, 64);
        assert_eq!(scored.as_u64().unwrap() % 64, 0);
        assert_eq!(
            manifest["speculation"]["maximum_verify_tokens"],
            model_glm5next_contract::contract::head_rows()
        );
        // The MTP drafter at the contract's maximum of five drafts by
        // default, verified in rows of 2 to 6, over the KV layouts with its
        // sparse layer.
        let speculation = &manifest["speculation"];
        assert_eq!(speculation["maximum_draft_tokens"], 5);
        assert_eq!(speculation["bundle_drafter"], true);
        assert_eq!(speculation["drafter_kind"], "mtp");
        assert_eq!(speculation["drafter_component"], RANK_IMAGE_COMPONENT);
        assert_eq!(
            speculation["row_buckets"],
            serde_json::json!([2, 3, 4, 5, 6])
        );
        assert_eq!(speculation["maximum_draft_items"], DECODE_BATCH);
        assert_eq!(manifest["batch_limits"]["rewrites_last_page"], true);
        let layouts = manifest["kv_formats"].to_string();
        assert!(
            layouts.contains("latent-paged-fp8-mtp") && layouts.contains("latent-paged-bf16-mtp")
        );
        // A package of three drafts verifies in rows of 2 to 4; host drafts
        // of up to five tokens in rows of 2 to 6 over the layouts without
        // the drafter's layer.
        let three = manifest_with(Drafter::Mtp { drafts: 3 });
        assert_eq!(
            three["speculation"]["row_buckets"],
            serde_json::json!([2, 3, 4])
        );
        let host = manifest_with(Drafter::None);
        let speculation = &host["speculation"];
        assert_eq!(speculation["maximum_draft_tokens"], 5);
        assert_eq!(speculation["bundle_drafter"], false);
        assert!(speculation.get("drafter_kind").is_none());
        assert!(!host["kv_formats"].to_string().contains("-mtp"));
        let six = draft(
            Shape::TP4,
            &FP8_BLOCK128,
            RuntimeValues { index_topk: 2048 },
            Vec::new(),
            StagedContentV1::bytes(b"module".to_vec()),
            Drafter::Mtp { drafts: 6 },
        );
        assert!(six.is_err());
        assert_eq!(manifest["batch_limits"]["maximum_sequences"], DECODE_BATCH);
        let required = manifest["resource_requirements"]["required_capabilities"]
            .as_array()
            .unwrap();
        for capability in CAPABILITIES {
            assert!(
                required.iter().any(|value| value == capability),
                "{capability}"
            );
        }
        // The bundle preallocates the KDA state before the KV pool; the
        // host reserves none.
        assert!(
            required.iter().any(
                |value| value == runtime_bundle_abi::RECURRENT_STATE_PREALLOCATED_CAPABILITY_V1
            )
        );
        let recurrent = manifest["recurrent_state_layouts"].as_array().unwrap();
        let declared: Vec<_> = recurrent
            .iter()
            .map(|layout| {
                (
                    layout["id"].as_str().unwrap(),
                    layout["bytes"].as_u64().unwrap(),
                )
            })
            .collect();
        assert_eq!(declared, shape.recurrent_state_layouts());
        assert_eq!(manifest["state_layout"]["id"], INDEXER_TAIL_STATE_ID);
        assert_eq!(
            manifest["state_layout"]["bytes"],
            shape.indexer_tail_slot_bytes(true)
        );
        assert_eq!(
            manifest["kv_layout"]["id"],
            utf8(model_glm5next_contract::contract::KV_LAYOUT_FP8_MTP)
        );
        let names: Vec<_> = manifest["model_options"]["options"]
            .as_array()
            .unwrap()
            .iter()
            .map(|option| option["name"].as_str().unwrap())
            .collect();
        assert_eq!(
            names,
            [
                LAYOUT_OPTION,
                model_glm5next_contract::options::COMM_EPILOGUE_OPTION,
                "text_config.index_topk"
            ]
        );
    }

    /// The default package declares the contract's five drafts, and each
    /// profile serves its own count within them: tp4-mtp three, tp4-mtp-k5
    /// five. The state create allocates follows the served count and the
    /// host reserves none, so tp4-mtp's state is the one a three-draft
    /// package gives, and the capacity plan at its count is a three-draft
    /// package's (665,792 KV tokens at 32 slots with FP8 KV for a text
    /// package, against tp4-mtp-k5's 366,400; a package serving images does
    /// not fit five drafts at 32 slots).
    #[test]
    fn profiles_serve_their_draft_count_within_the_package_maximum() {
        use model_glm5next_contract::capacity::{DEVICE, Planner, Serving};
        use model_glm5next_contract::contract::{MTP_DRAFTS, latent_format};
        let file: Value = serde_json::from_str(include_str!("../../profiles.json")).unwrap();
        let package: runtime_bundle::BundleManifestV1 =
            serde_json::from_value(manifest_of(&FP8_BLOCK128, Drafter::default())).unwrap();
        let maximum = package.speculation().unwrap().maximum_draft_tokens() as usize;
        assert_eq!(maximum, MAX_DRAFT_TOKENS as usize);
        let planner = Planner::new(Shape::TP4.geometry, 4).unwrap();
        let mut served = std::collections::BTreeMap::new();
        for (name, profile) in file["profiles"].as_object().unwrap() {
            let tokens = profile["speculation"]["tokens"].as_u64().unwrap() as usize;
            assert!(tokens <= maximum, "{name}");
            if profile["speculation"]["drafter"] != "mtp" {
                continue;
            }
            served.insert(name.as_str(), tokens);
            // The bundle allocates the served count's state itself, so the
            // host reserves none for either package.
            let reserved = |manifest: &runtime_bundle::BundleManifestV1| {
                runtime_host::recurrent::StateSlotPoolV1::new(manifest, 32, tokens + 1)
                    .unwrap()
                    .reserved_bytes_per_rank()
            };
            let exact: runtime_bundle::BundleManifestV1 = serde_json::from_value(manifest_of(
                &FP8_BLOCK128,
                Drafter::Mtp {
                    drafts: tokens as u32,
                },
            ))
            .unwrap();
            assert_eq!((reserved(&package), reserved(&exact)), (0, 0), "{name}");
        }
        assert_eq!(served["tp4-mtp"], MTP_DRAFTS as usize);
        assert_eq!(served["tp4-mtp-k5"], MAX_DRAFT_TOKENS as usize);
        let plan = |draft_tokens, vision| {
            planner.plan(
                Serving {
                    ranks: 4,
                    context_parallel: 1,
                    slots: 32,
                    draft_tokens,
                    // The preferred FP8 latent KV.
                    kv: latent_format(&Shape::TP4.kv_formats()[0]),
                    vision,
                },
                DEVICE,
            )
        };
        let three = plan(MTP_DRAFTS, false).unwrap();
        assert_eq!((three.state_copies, three.kv_tokens), (5, 665_792));
        let five = plan(MAX_DRAFT_TOKENS, false).unwrap();
        assert_eq!((five.state_copies, five.kv_tokens), (7, 366_400));
        assert!(plan(MAX_DRAFT_TOKENS, true).is_err());
    }

    /// Each serving profile asserts its checkpoint layout as a model option.
    /// Through the host's own parser (`serve --bundle --profile`) and the
    /// manifest resolution startup runs on the parsed options, a profile is
    /// admitted only by a package of its layout, with no explicit option or
    /// with the package's value; an explicit option naming the package's
    /// layout does not admit a profile of the other one.
    #[test]
    fn profiles_resolve_only_against_a_package_of_their_layout() {
        let profiles = concat!(env!("CARGO_MANIFEST_DIR"), "/../profiles.json");
        let file: Value = serde_json::from_str(include_str!("../../profiles.json")).unwrap();
        let mut served = std::collections::BTreeMap::new();
        for layout in LAYOUTS {
            let planned = manifest_of(layout, Drafter::default());
            let manifest: runtime_bundle::BundleManifestV1 =
                serde_json::from_value(planned.clone()).unwrap();
            let package = std::env::temp_dir().join(format!(
                "glm5next-layout-package-{}-{}",
                layout.id,
                std::process::id()
            ));
            std::fs::create_dir_all(&package).unwrap();
            std::fs::write(
                package.join(runtime_bundle::BUNDLE_MANIFEST_FILE),
                serde_json::to_vec(&planned).unwrap(),
            )
            .unwrap();
            for (name, profile) in file["profiles"].as_object().unwrap() {
                let own = profile["model_options"][LAYOUT_OPTION] == layout.id;
                for explicit in std::iter::once(None).chain(LAYOUTS.iter().map(Some)) {
                    let mut words = vec![
                        "serve".to_owned(),
                        "--bundle".into(),
                        package.display().to_string(),
                        "--checkpoint".into(),
                        package.display().to_string(),
                        "--profile".into(),
                        format!("{profiles}#{name}"),
                    ];
                    if let Some(explicit) = explicit {
                        words.push("--model-option".into());
                        words.push(format!("{LAYOUT_OPTION}={}", explicit.id));
                    }
                    let admitted = match runtime_host::parse_host_command_v1(&words) {
                        Ok(runtime_host::HostCommandV1::Serve { model_options, .. }) => {
                            manifest.resolve_model_options(&model_options).is_ok()
                        }
                        Ok(_) => panic!("{name}: not a serve command"),
                        Err(_) => false,
                    };
                    let expected = own && explicit.is_none_or(|explicit| explicit == layout);
                    assert_eq!(
                        admitted,
                        expected,
                        "{name} on {} with {:?}",
                        layout.id,
                        explicit.map(|layout| layout.id)
                    );
                }
                if own {
                    served.insert(name.as_str(), layout.id);
                }
            }
            std::fs::remove_dir_all(&package).unwrap();
            // Without a profile the option resolves to the package's layout.
            let defaults = manifest.resolve_model_options(&Default::default()).unwrap();
            assert_eq!(defaults[LAYOUT_OPTION], layout.id);
        }
        assert_eq!(
            served,
            std::collections::BTreeMap::from([
                ("tp4-ar", FP8_BLOCK128.id),
                ("tp4-mtp", FP8_BLOCK128.id),
                ("tp4-mtp-dcp2", FP8_BLOCK128.id),
                ("tp4-mtp-dcp4", FP8_BLOCK128.id),
                ("tp4-mtp-k5", FP8_BLOCK128.id),
                ("tp4-mtp-nvfp4", NVFP4_ROUTED.id),
                ("tp4-mtp-nvfp4-nvidia", NVFP4_DENSE_ROUTED.id),
            ])
        );
    }

    /// The block-FP8 package carries the precision component whole, under
    /// the role create reads it by; bytes that do not decode are refused.
    #[test]
    fn the_fp8_package_carries_its_precision_component() {
        let manifest = manifest();
        let component = manifest["components"]
            .as_array()
            .unwrap()
            .iter()
            .find(|component| component["id"] == PRECISION_COMPONENT)
            .expect("the precision component");
        assert_eq!(
            component["role"],
            model_glm5next_contract::prepared::COMPONENT_ROLE
        );
        assert_eq!(component["path"], PRECISION_PATH);
        assert!(precision(b"not a component".to_vec(), &FP8_BLOCK128).is_err());
    }
}
