//! The canonical MiMo package draft. A package carries no weights: the
//! text image, and the drafter's and encoders' images it declares, are
//! planned components the bundle's weight plan loads from the checkpoint
//! (`serve --model DIR`).

use model_compiler_sdk::package_draft::{self, whole_component};
use std::collections::BTreeMap;
use std::path::Path;

use model_compiler_sdk::{
    BundleDraftV1, CompilerError, EmittedBundleV1, FileIdentity, StagedContentV1,
    emit_bundle_directory_v1,
};
use runtime_bundle::{
    BundleBatchLimitsV1, BundleError, BundleKvFormatV1, BundleKvLayoutV1, BundleOpaqueId,
    BundleResourceRequirementsV1, BundleSpeculationV1, BundleStateLayoutV1, BundleStateLifetimeV1,
    BundleTargetV1, BundleTopologyV1, SPECULATIVE_CAPABILITY_V1,
};
// The tests name the KV dtypes.
#[cfg(test)]
use runtime_bundle::BundleKvDtypeV1;

use model_mimo26::catalog::DrafterKind;
use model_mimo26_contract::backend::{Kernels, Selected};
use model_mimo26_contract::encoders::Encoders;
use model_mimo26_contract::serving::BUNDLE_ID;
use model_mimo26_contract::serving::{PAGE_TOKENS, PREFILL_TOKENS};
use model_mimo26_contract::topology::{Tp, ga_page_bytes, ring_state_slot_bytes, topology_id};

/// The drafter-kind component: the plan has no packaged drafter index to
/// tell MTP from DFlash.
fn drafter_kind_component(kind: DrafterKind) -> Vec<u8> {
    serde_json::to_vec(&serde_json::json!({ "drafter": kind.as_str() })).expect("JSON")
}

/// The serving profiles and deployment policy every package carries.
pub(crate) const PROFILES: &str = include_str!("../../../../profiles.json");
pub(crate) const DEPLOYMENT_POLICY: &str = include_str!("../../../../deploy/policy.json");

fn id(value: &str) -> Result<BundleOpaqueId, BundleError> {
    BundleOpaqueId::new(value)
}

pub struct PackageAssets {
    pub tokenizer_definition: StagedContentV1,
    pub tokenizer_configuration: StagedContentV1,
    pub native_module: StagedContentV1,
    /// The drafter whose image the package plans, if any.
    pub drafter: Option<DrafterKind>,
    /// The media served by the planned encoders.
    pub encoders: Encoders,
    /// Topology precision components produced from admitted checkpoint catalogs.
    pub precision: BTreeMap<String, Vec<u8>>,
}

impl PackageAssets {
    /// The checkpoint's tokenizer files, staged only as `registry` pins
    /// them for `identity` (a registered or a local identity).
    pub fn pinned_tokenizer_with_registry(
        checkpoint: &Path,
        identity: &model_mimo26::checkpoint::Identity,
        registry: &model_mimo26::checkpoint::Registry,
    ) -> Result<[StagedContentV1; 2], CompilerError> {
        let geometry = model_mimo26::config::MiMoConfig::load(checkpoint)
            .and_then(|config| config.geometry())
            .map_err(|error| CompilerError::validation(error.to_string()))?;
        let contract = registry
            .tokenizer_contract_for_geometry(identity, geometry)
            .map_err(|error| CompilerError::validation(error.to_string()))?;
        let mut inputs = Vec::new();
        for identity in [contract.definition(), contract.configuration()] {
            let pinned = FileIdentity::new(identity.name(), identity.size(), identity.sha256())
                .map_err(|error| CompilerError::validation(error.to_string()))?;
            inputs.push(StagedContentV1::authenticated_file(
                checkpoint.join(identity.name()),
                pinned,
            ));
        }
        Ok(inputs.try_into().expect("two pinned tokenizer identities"))
    }
}

/// The package draft, which serves every supported rank count: at each
/// topology of `rank_bytes`, whose largest rank holds that many bytes of
/// planned images, with its per-rank resources ([`topology_draft`]).
#[cfg(test)]
fn bundle_draft(
    assets: &PackageAssets,
    identity: &model_mimo26::checkpoint::Identity,
    binding: Option<Vec<u8>>,
    rank_bytes: &[(Tp, u64)],
) -> Result<BundleDraftV1, CompilerError> {
    bundle_draft_with_registry(
        assets,
        identity,
        binding,
        rank_bytes,
        model_mimo26::checkpoint::registry(),
    )
}

fn bundle_draft_with_registry(
    assets: &PackageAssets,
    identity: &model_mimo26::checkpoint::Identity,
    binding: Option<Vec<u8>>,
    rank_bytes: &[(Tp, u64)],
    registry: &model_mimo26::checkpoint::Registry,
) -> Result<BundleDraftV1, CompilerError> {
    let drafts = rank_bytes
        .iter()
        .map(|&(tp, bytes)| topology_draft(assets, identity, binding.clone(), tp, bytes, registry))
        .collect::<Result<Vec<_>, _>>()?;
    BundleDraftV1::for_topologies(drafts)
}

/// The package draft at `tp`. Every image is declared by role and format
/// with no file, so the host asks the weight plan for each and the drafter
/// speculation can name its component; the largest rank holds `rank_bytes`
/// of them. A metadata component names the drafter's kind for the plan.
fn topology_draft(
    assets: &PackageAssets,
    identity: &model_mimo26::checkpoint::Identity,
    binding: Option<Vec<u8>>,
    tp: Tp,
    rank_bytes: u64,
    registry: &model_mimo26::checkpoint::Registry,
) -> Result<BundleDraftV1, BundleError> {
    let topology = topology_id(tp);
    let drafter_kind = assets.drafter;
    let has_encoders = assets.encoders.any();
    if has_encoders && !model_mimo26_contract::encoders::SERVING_READY {
        return Err(BundleError::validation(
            "MiMo encoder serving path is incomplete; media package is unavailable",
        ));
    }
    let planned = |component: &str, role: &str, format: &str| {
        model_compiler_sdk::ComponentDraftV1::planned(id(component)?, id(role)?, id(format)?)
    };
    let mut components = vec![planned(
        "rank-images",
        "mimo.rank-images.v1",
        "mimo.rank-image.v1",
    )?];
    let precision_id = model_mimo26_contract::prepared::component_id(tp);
    let precision = assets
        .precision
        .get(&precision_id)
        .ok_or_else(|| BundleError::validation("missing admitted topology precision component"))?;
    kernel_core::precision::component::decode(precision)
        .map_err(|error| BundleError::validation(error.to_string()))?;
    components.push(
        whole_component(
            &precision_id,
            model_mimo26_contract::prepared::COMPONENT_ROLE,
            model_mimo26_contract::prepared::COMPONENT_FORMAT,
            &model_mimo26_contract::prepared::component_path(tp),
            StagedContentV1::bytes(precision.clone()),
        )?
        .with_topologies(vec![id(topology)?]),
    );
    if let Some(kind) = drafter_kind {
        components.push(planned(
            "drafter-rank-images",
            "mimo.drafter-rank-images.v1",
            "mimo.rank-image.v1",
        )?);
        components.push(whole_component(
            "drafter-kind",
            model_mimo26_contract::weight_plan::DRAFTER_KIND_ROLE,
            "mimo.drafter-kind.json.v1",
            "components/drafter-kind.json",
            StagedContentV1::bytes(drafter_kind_component(kind)),
        )?);
    }
    if has_encoders {
        components.push(planned(
            "encoder-rank-images",
            "mimo.encoder-rank-images.v1",
            "mimo.encoder-rank-image.v1",
        )?);
    }
    if assets.encoders.any() && assets.encoders != Encoders::ALL {
        components.push(whole_component(
            "encoder-set",
            model_mimo26_contract::weight_plan::ENCODER_SET_ROLE,
            "mimo.encoder-set.json.v1",
            "components/encoder-set.json",
            StagedContentV1::bytes(model_mimo26_contract::weight_plan::encoder_set_component(
                assets.encoders,
            )),
        )?);
    }
    components.extend([
        package_draft::frontend(StagedContentV1::bytes(
            registry
                .component_for_geometry(identity, assets.encoders, tp.geometry)
                .map_err(|error| BundleError::validation(error.to_string()))?,
        ))?,
        package_draft::tokenizer_definition(assets.tokenizer_definition.clone())?,
        package_draft::tokenizer_configuration(assets.tokenizer_configuration.clone())?,
        package_draft::serve_profiles(PROFILES.as_bytes())?,
        package_draft::deployment_policy(DEPLOYMENT_POLICY.as_bytes())?,
    ]);
    if let Some(binding) = binding {
        components.push(whole_component(
            "checkpoint-binding",
            runtime_bundle::checkpoint_binding::ROLE,
            "chop.checkpoint-binding.json.v1",
            "components/checkpoint-binding.json",
            StagedContentV1::bytes(binding),
        )?);
    }
    let sequences = maximum_sequences(tp, drafter_kind, assets.encoders, rank_bytes);
    let buckets = package_draft::graph_buckets_for(
        &components,
        model_mimo26_contract::graphs::buckets_through(sequences),
        |batch| format!("mimo-decode-b{batch}"),
        "mimo-prefill",
    )?;
    // The module verifies host drafts whether or not a drafter is packaged.
    let mut capabilities = vec![
        id(runtime_bundle_abi::SERVING_KV_CAPABILITY_V1)?,
        id(runtime_bundle_abi::PROMPT_ROWS_CAPABILITY_V2)?,
        id(SPECULATIVE_CAPABILITY_V1)?,
        id(runtime_bundle_abi::PREFIX_CACHE_CAPABILITY_V1)?,
        id(runtime_bundle_abi::KV_OFFLOAD_CAPABILITY_V1)?,
        // The module reads the served item count from the create record,
        // and its ring state holds one slot per item of the largest bucket.
        id(runtime_bundle_abi::SERVED_ITEMS_CAPABILITY_V1)?,
        id(runtime_bundle_abi::SERVED_CONTEXT_CAPABILITY_V1)?,
        id(runtime_bundle_abi::STATE_PER_ITEM_CAPABILITY_V1)?,
    ];
    if has_encoders {
        capabilities.push(id(model_mimo26_contract::encoders::MEDIA_CAPABILITY)?);
    }
    capabilities.push(id(
        runtime_bundle_abi::weight_plan::WEIGHT_PLAN_CAPABILITY_V1,
    )?);
    capabilities.push(id(runtime_bundle_abi::frontend::FRONTEND_CAPABILITY_V1)?);
    let mut draft = BundleDraftV1::new(
        id(BUNDLE_ID)?,
        BundleTargetV1::new(
            vec![id(Selected::ARCHITECTURE)?],
            vec![BundleTopologyV1::new(id(topology)?, tp.ranks())?],
        )?,
        BundleResourceRequirementsV1::new(
            model_mimo26_contract::topology::minimum_rank_bytes(
                tp,
                rank_bytes,
                sequences,
                drafter_kind.map(DrafterKind::planned),
                assets.encoders,
            ),
            capabilities,
        )?,
        kv_layout(0, tp)?,
        BundleStateLayoutV1::new(
            id("mimo.swa-ring.v1")?,
            ring_state_slot_bytes(tp),
            256,
            BundleStateLifetimeV1::Process,
        )?,
    )
    .with_kv_formats(kv_formats(
        tp,
        registry
            .fp8_kv_layout(identity, tp)
            .map_err(BundleError::validation)?
            .as_deref(),
    )?)
    .with_batch_limits(
        BundleBatchLimitsV1::new(sequences, PREFILL_TOKENS)?
            .with_maximum_scored_rows(model_mimo26_contract::serving::SCORED_ROWS)?,
    )
    .with_components(components)
    .with_graph_buckets(buckets)
    .with_native_module(package_draft::native_module(assets.native_module.clone())?);
    // The bundle drafts with a packaged MTP or DFlash drafter when the module
    // can run it; otherwise the host supplies drafts.
    let bundle_drafter = drafter_kind.is_some() && model_mimo26_contract::serving::BUNDLE_DRAFTS;
    draft = draft.with_speculation(
        BundleSpeculationV1::new(
            model_mimo26_contract::serving::MAX_VERIFY_ROWS - 1,
            bundle_drafter,
            maximum_verify_tokens(drafter_kind),
            model_mimo26_contract::serving::VERIFY_ROW_BUCKETS.to_vec(),
            bundle_drafter
                .then(|| id("drafter-rank-images"))
                .transpose()?,
        )?
        .with_drafter_kind(if bundle_drafter {
            drafter_kind.map(|kind| id(kind.as_str())).transpose()?
        } else {
            None
        })?,
    );
    Ok(draft)
}

/// The paged global-attention KV layout of bundle format `format` at `tp`,
/// whose pages hold `tp`'s GA heads per rank.
fn kv_layout(format: usize, tp: Tp) -> Result<BundleKvLayoutV1, BundleError> {
    let formats = model_mimo26_contract::serving::kv_formats(tp);
    let format = &formats[format];
    package_draft::paged_kv_layout(
        std::str::from_utf8(format.layout_id).expect("layout IDs are ASCII"),
        ga_page_bytes(tp, format.fp8()),
        PAGE_TOKENS,
        tp.geometry.context_tokens() as u64,
    )
}

/// BF16 and, when calibrated, only the authenticated checkpoint's FP8 table.
fn kv_formats(tp: Tp, calibrated: Option<&str>) -> Result<Vec<BundleKvFormatV1>, BundleError> {
    model_mimo26_contract::serving::kv_formats(tp)
        .iter()
        .enumerate()
        .filter(|(_, format)| {
            !format.fp8() || calibrated.is_some_and(|id| id.as_bytes() == format.layout_id)
        })
        .map(|(index, format)| {
            Ok(BundleKvFormatV1::new(
                if format.fp8() {
                    runtime_bundle::BundleKvDtypeV1::Fp8E4m3
                } else {
                    runtime_bundle::BundleKvDtypeV1::Bf16
                },
                kv_layout(index, tp)?,
            ))
        })
        .collect()
}

/// Decode-phase rows one verify step may carry. The module's DFlash block
/// drafts every slot to its full depth, and host drafts (for example the
/// teacher-forced qualification drafts) verify through the same target
/// path; MTP drafts and catches up within the decode kernels' rows.
fn maximum_verify_tokens(drafter: Option<DrafterKind>) -> u32 {
    match drafter {
        Some(DrafterKind::Mtp) => model_mimo26_contract::serving::DECODE_ENTRY_ROWS,
        Some(DrafterKind::Dflash) | None => model_mimo26_contract::serving::VERIFY_TOKENS,
    }
}

/// The package maximum of live sequences for its largest rank's
/// `rank_bytes` of images (`model_mimo26_contract::topology::maximum_sequences`).
/// Deploy defaults serve at most 32 of them.
fn maximum_sequences(
    tp: Tp,
    drafter: Option<DrafterKind>,
    media: Encoders,
    rank_bytes: u64,
) -> u32 {
    model_mimo26_contract::topology::maximum_sequences(
        tp,
        drafter.map(DrafterKind::planned),
        media,
        rank_bytes,
    )
}

/// At every supported rank count, the most planned image bytes any rank of
/// the package holds: its text image, and the drafter's and encoders'
/// images the assets declare.
pub fn planned_rank_bytes(
    checkpoint: &Path,
    assets: &PackageAssets,
) -> Result<Vec<(Tp, u64)>, CompilerError> {
    let error = |error: Box<dyn std::error::Error + Send + Sync>| {
        CompilerError::validation(error.to_string())
    };
    let config = model_mimo26::checkpoint::registry()
        .verifier(checkpoint)
        .and_then(|verifier| verifier.config())
        .map_err(error)?;
    let catalog = model_mimo26::catalog::Catalog::load(checkpoint, &config).map_err(error)?;
    let drafter_catalog = assets
        .drafter
        .map(|kind| {
            model_mimo26::catalog::Catalog::load_drafter_with_config(checkpoint, kind, &config)
        })
        .transpose()
        .map_err(error)?;
    Tp::SUPPORTED
        .into_iter()
        .map(|ranks| {
            let tp = Tp::of_geometry(config.geometry().map_err(error)?, ranks.ranks())
                .map_err(CompilerError::validation)?;
            let mut bytes = model_mimo26::rank_bytes::estimated_rank_bytes(&catalog, None, tp)
                .map_err(error)?;
            if let (Some(kind), Some(drafter_catalog)) = (assets.drafter, &drafter_catalog) {
                let drafter_bytes = model_mimo26::rank_bytes::estimated_drafter_rank_bytes(
                    drafter_catalog,
                    kind,
                    tp,
                )
                .map_err(error)?;
                bytes
                    .iter_mut()
                    .zip(drafter_bytes)
                    .for_each(|(b, d)| *b += d);
            }
            if assets.encoders.any() {
                let encoder_bytes = model_mimo26::encoders::image::estimated_rank_bytes(
                    &catalog.encoders,
                    tp,
                    assets.encoders,
                )
                .map_err(error)?;
                bytes
                    .iter_mut()
                    .zip(encoder_bytes)
                    .for_each(|(b, e)| *b += e);
            }
            Ok((tp, bytes.into_iter().max().unwrap_or(0)))
        })
        .collect()
}

/// The implementation contract the native module's note publishes: the
/// audited kernel contract identity, which hashes every entry's wrapper and
/// declared arithmetic, and its cubins' dependency identity. A package
/// carries only a module that states it, and the package's module digest
/// authenticates the note with the module's bytes.
pub fn native_contract(
    module: &StagedContentV1,
) -> Result<runtime_bundle::preparation::ImplementationContract, CompilerError> {
    let (StagedContentV1::File(path) | StagedContentV1::AuthenticatedFile { path, .. }) = module
    else {
        return Err(CompilerError::validation(
            "the native module must be a file with its contract note",
        ));
    };
    model_compiler_sdk::native_contract::NativeModule::inspect(path)
        .map_err(|error| CompilerError::validation(error.to_string()))?
        .contracts
        .implementation_contract
        .ok_or_else(|| {
            CompilerError::validation("the native module publishes no implementation contract")
        })
}

/// A checkpoint-bound package whose largest rank at each topology holds
/// `rank_bytes` of planned images (see `bundle_draft`).
pub fn emit_verified_direct_package(
    assets: PackageAssets,
    rank_bytes: &[(Tp, u64)],
    output: &Path,
    verifier: &model_mimo26::checkpoint_provenance::Verifier,
) -> Result<EmittedBundleV1, CompilerError> {
    emit_verified(assets, output, verifier, rank_bytes)
}

/// Binds the package to the checkpoint `verifier` authenticated: its
/// tokenizer must be the pinned one, and the binding records the digest of
/// every file the package carries.
fn emit_verified(
    assets: PackageAssets,
    output: &Path,
    verifier: &model_mimo26::checkpoint_provenance::Verifier,
    rank_bytes: &[(Tp, u64)],
) -> Result<EmittedBundleV1, CompilerError> {
    native_contract(&assets.native_module)?;
    let identity = verifier.identity();
    let registry = verifier.registry();
    use runtime_bundle::checkpoint_binding::{CheckpointBinding, FileDigest, SCHEMA};
    use sha2::{Digest, Sha256};
    fn digest(content: &StagedContentV1) -> Result<FileDigest, CompilerError> {
        let (byte_length, sha256) = match content {
            StagedContentV1::Bytes(bytes) => {
                (bytes.len() as u64, format!("{:x}", Sha256::digest(bytes)))
            }
            StagedContentV1::AuthenticatedFile { identity, .. } => {
                (identity.size(), identity.sha256().to_owned())
            }
            StagedContentV1::File(path) => {
                let source = model_mimo26::catalog::open_source(path)
                    .map_err(|e| CompilerError::validation(e.to_string()))?;
                let size = source
                    .metadata()
                    .map_err(|e| CompilerError::validation(e.to_string()))?
                    .len();
                let hash = runtime_weights::hash_range(&source, 0, size)
                    .map_err(|e| CompilerError::validation(e.to_string()))?;
                source
                    .verify()
                    .map_err(|e| CompilerError::validation(e.to_string()))?;
                (size, hash)
            }
        };
        Ok(FileDigest {
            byte_length,
            sha256,
        })
    }
    for (name, content) in [
        ("tokenizer.json", &assets.tokenizer_definition),
        ("tokenizer_config.json", &assets.tokenizer_configuration),
    ] {
        let file = digest(content)?;
        verifier
            .verify_tokenizer(name, file.byte_length, &file.sha256)
            .map_err(|e| CompilerError::validation(e.to_string()))?;
    }
    let mut files = BTreeMap::new();
    files.insert(
        "components/frontend.json".into(),
        digest(&StagedContentV1::bytes(
            registry
                .component_for_geometry(
                    identity,
                    assets.encoders,
                    verifier
                        .config()
                        .and_then(|config| config.geometry())
                        .map_err(|error| CompilerError::validation(error.to_string()))?,
                )
                .map_err(|error| CompilerError::validation(error.to_string()))?,
        ))?,
    );
    files.insert(
        "components/tokenizer.json".into(),
        digest(&assets.tokenizer_definition)?,
    );
    files.insert(
        "components/tokenizer_config.json".into(),
        digest(&assets.tokenizer_configuration)?,
    );
    files.insert(
        runtime_bundle::deployment::SERVE_PROFILES_PATH_V1.into(),
        digest(&StagedContentV1::bytes(PROFILES))?,
    );
    files.insert(
        runtime_bundle::deployment::DEPLOYMENT_POLICY_PATH_V1.into(),
        digest(&StagedContentV1::bytes(DEPLOYMENT_POLICY))?,
    );
    files.insert("native/module.bin".into(), digest(&assets.native_module)?);
    for &(tp, _) in rank_bytes {
        let component = assets
            .precision
            .get(&model_mimo26_contract::prepared::component_id(tp))
            .ok_or_else(|| {
                CompilerError::validation("missing admitted topology precision component")
            })?;
        files.insert(
            model_mimo26_contract::prepared::component_path(tp),
            digest(&StagedContentV1::bytes(component.clone()))?,
        );
    }
    if let Some(kind) = assets.drafter {
        files.insert(
            "components/drafter-kind.json".into(),
            digest(&StagedContentV1::bytes(drafter_kind_component(kind)))?,
        );
    }
    if assets.encoders.any() && assets.encoders != Encoders::ALL {
        files.insert(
            "components/encoder-set.json".into(),
            digest(&StagedContentV1::bytes(
                model_mimo26_contract::weight_plan::encoder_set_component(assets.encoders),
            ))?,
        );
    }
    let binding = CheckpointBinding {
        schema: SCHEMA.into(),
        repository: identity.repository.clone(),
        revision: identity.revision.clone(),
        served_model_name: identity.served_model_name.clone(),
        source_lock_sha256: verifier.source_lock_sha256(),
        files,
    };
    let bytes =
        serde_json::to_vec(&binding).map_err(|e| CompilerError::validation(e.to_string()))?;
    emit_bundle_directory_v1(
        &bundle_draft_with_registry(&assets, identity, Some(bytes), rank_bytes, registry)?,
        output,
    )
}

/// A small admitted component for tests of package metadata and file
/// proofs, from the SM120 catalog offers.
#[cfg(test)]
#[path = "sm120.rs"]
mod sm120;
#[cfg(test)]
pub use sm120::fixture_precision;

#[cfg(test)]
mod tests {
    use super::*;
    use model_mimo26_contract::serving::DECODE_BATCH;
    use runtime_bundle::BundleManifestV1;

    /// Assets whose tokenizer and module are placeholders: planning a
    /// manifest does not open them.
    fn assets(drafter: Option<DrafterKind>, encoders: bool) -> PackageAssets {
        PackageAssets {
            tokenizer_definition: StagedContentV1::bytes(b"{}".to_vec()),
            tokenizer_configuration: StagedContentV1::bytes(b"{}".to_vec()),
            native_module: StagedContentV1::bytes(b"module".to_vec()),
            drafter,
            precision: fixture_precision(),
            encoders: if encoders {
                Encoders::ALL
            } else {
                Encoders::NONE
            },
        }
    }

    /// `rank_bytes` of planned images on the largest rank at every topology.
    fn uniform(rank_bytes: u64) -> Vec<(Tp, u64)> {
        Tp::SUPPORTED.map(|tp| (tp, rank_bytes)).to_vec()
    }

    fn manifest(assets: PackageAssets, rank_bytes: u64) -> BundleManifestV1 {
        let identity = model_mimo26::checkpoint::default_identity();
        model_compiler_sdk::plan_bundle_manifest_v1(
            &bundle_draft(&assets, identity, None, &uniform(rank_bytes)).expect("draft"),
        )
        .expect("manifest preflight")
    }

    /// The package as it serves `tp`.
    fn selected(assets: PackageAssets, rank_bytes: u64, tp: Tp) -> BundleManifestV1 {
        manifest(assets, rank_bytes)
            .for_topology(topology_id(tp))
            .expect("every rank count is a topology")
    }

    fn requires(manifest: &BundleManifestV1, capability: &str) -> bool {
        manifest
            .resource_requirements()
            .required_capabilities()
            .iter()
            .any(|required| required.as_str() == capability)
    }

    /// Records the package admission and scratch sums at each served topology.
    #[test]
    fn fixture_package_manifest_identity_covers_drafter_topologies() {
        let directory =
            std::env::var_os("CHOPINFER_MIMO26_MANIFEST_DIR").map(std::path::PathBuf::from);
        if let Some(directory) = &directory {
            std::fs::create_dir_all(directory).unwrap();
        }
        let mut identities = std::collections::BTreeSet::new();
        for (mode, drafter) in [
            ("ar", None),
            ("mtp", Some(DrafterKind::Mtp)),
            ("dflash", Some(DrafterKind::Dflash)),
        ] {
            let package = manifest(assets(drafter, false), 1 << 30);
            for tp in [Tp::TP2, Tp::TP4] {
                let selected = package.for_topology(topology_id(tp)).unwrap();
                assert_eq!(selected.target().topologies()[0].rank_count(), tp.ranks());
                assert_eq!(selected.id().as_str(), BUNDLE_ID);
                assert!(
                    selected
                        .resource_requirements()
                        .minimum_device_memory_bytes()
                        > 1 << 30
                );
                let bytes = serde_json::to_vec_pretty(&selected).unwrap();
                assert!(identities.insert(bytes.clone()));
                if let Some(directory) = &directory {
                    std::fs::write(
                        directory.join(format!("tp{}-{mode}.json", tp.ranks())),
                        bytes,
                    )
                    .unwrap();
                }
            }
        }
        assert_eq!(identities.len(), 6);
    }

    /// A package declares every image it serves as a planned component
    /// (text, and the drafter's and encoders' when packaged), carries no
    /// rank files or indexes, names its drafter's kind for the plan and
    /// requires the weight plan.
    #[test]
    fn image_free_packages_plan_every_image() {
        let identity = model_mimo26::checkpoint::identity("rl").unwrap();
        let rank_bytes =
            43_450_937_344 + 755_142_656 + model_mimo26_contract::encoders::image_bytes_flash();
        for (drafter, encoders) in [
            (None, false),
            (Some(DrafterKind::Dflash), false),
            (Some(DrafterKind::Mtp), false),
            (Some(DrafterKind::Dflash), true),
        ] {
            let root = std::env::temp_dir().join(format!(
                "package-image-free-{}-{drafter:?}-{encoders}",
                std::process::id()
            ));
            let _ = std::fs::remove_dir_all(&root);
            let assets = PackageAssets {
                native_module: StagedContentV1::file(crate::test_native::module()),
                ..assets(drafter, encoders)
            };
            let emitted = emit_bundle_directory_v1(
                &bundle_draft(&assets, identity, None, &uniform(rank_bytes)).unwrap(),
                &root,
            )
            .unwrap();
            let manifest: serde_json::Value = serde_json::from_slice(
                &std::fs::read(emitted.root().join("manifest.json")).unwrap(),
            )
            .unwrap();
            assert!(
                manifest["compiled_images"]
                    .as_array()
                    .is_none_or(Vec::is_empty)
            );
            let roles = |key: &str| -> Vec<String> {
                manifest[key]
                    .as_array()
                    .map(|list| {
                        list.iter()
                            .map(|c| c["role"].as_str().unwrap().to_owned())
                            .collect()
                    })
                    .unwrap_or_default()
            };
            let mut planned = vec!["mimo.rank-images.v1".to_owned()];
            if drafter.is_some() {
                planned.push("mimo.drafter-rank-images.v1".into());
            }
            if encoders {
                planned.push("mimo.encoder-rank-images.v1".into());
            }
            let mut declared = roles("planned_components");
            declared.sort();
            planned.sort();
            assert_eq!(declared, planned);
            let components = roles("components");
            assert!(
                components
                    .iter()
                    .all(|role| !role.contains("rank-images") && !role.contains("rank-index"))
            );
            let kind = model_mimo26_contract::weight_plan::DRAFTER_KIND_ROLE;
            assert_eq!(
                components.iter().any(|role| role == kind),
                drafter.is_some()
            );
            if let Some(drafter) = drafter {
                let bytes =
                    std::fs::read(emitted.root().join("components/drafter-kind.json")).unwrap();
                let value: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
                assert_eq!(value["drafter"], drafter.as_str());
            }
            let capabilities = &manifest["resource_requirements"]["required_capabilities"];
            assert!(
                capabilities
                    .as_array()
                    .unwrap()
                    .contains(&serde_json::json!(
                        runtime_bundle_abi::weight_plan::WEIGHT_PLAN_CAPABILITY_V1
                    ))
            );
            assert!(
                capabilities
                    .as_array()
                    .unwrap()
                    .contains(&serde_json::json!(
                        runtime_bundle_abi::frontend::FRONTEND_CAPABILITY_V1
                    ))
            );
            std::fs::remove_dir_all(root).unwrap();
        }
    }

    #[test]
    fn served_sets_have_distinct_packages_and_matching_encoder_metadata() {
        let mut identities = std::collections::BTreeSet::new();
        for set in Encoders::SETS {
            let assets = PackageAssets {
                encoders: set,
                ..assets(None, false)
            };
            let identity = model_mimo26::checkpoint::default_identity();
            let draft = bundle_draft(&assets, identity, None, &uniform(1)).unwrap();
            let manifest = model_compiler_sdk::plan_bundle_manifest_v1(&draft).unwrap();
            let value = serde_json::to_value(&manifest).unwrap();
            assert!(identities.insert(serde_json::to_vec(&value).unwrap()));
            assert_eq!(
                requires(&manifest, model_mimo26_contract::encoders::MEDIA_CAPABILITY),
                set.any()
            );
            let partial = set.any() && set != Encoders::ALL;
            assert_eq!(
                value["components"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .any(|c| c["role"] == model_mimo26_contract::weight_plan::ENCODER_SET_ROLE),
                partial
            );
            assert_eq!(
                value["planned_components"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .any(|c| c["role"] == "mimo.encoder-rank-images.v1"),
                set.any()
            );
            if partial {
                let bytes = model_mimo26_contract::weight_plan::encoder_set_component(set);
                assert_eq!(
                    model_mimo26_contract::weight_plan::parse_encoder_set(&bytes).unwrap(),
                    set
                );
            }
        }
    }

    #[test]
    fn the_package_declares_bf16_and_fp8_kv_and_prefers_bf16() {
        let manifest = selected(assets(None, false), 1, Tp::TP4);
        let preferred = manifest.kv_layout();
        assert_eq!(preferred.id().as_str(), "mimo.ga-paged-bf16.v1");
        let formats: Vec<_> = manifest
            .kv_formats()
            .iter()
            .map(|format| {
                let layout = format.layout();
                (
                    format.dtype(),
                    layout.id().as_str(),
                    layout.page_bytes(),
                    layout.bytes_per_token(),
                )
            })
            .collect();
        assert_eq!(
            formats,
            [
                (
                    BundleKvDtypeV1::Bf16,
                    "mimo.ga-paged-bf16.v1",
                    model_mimo26_contract::serving::GA_PAGE_BYTES_BF16,
                    5_760
                ),
                (
                    BundleKvDtypeV1::Fp8E4m3,
                    "mimo.ga-paged-fp8.v2",
                    model_mimo26_contract::serving::GA_PAGE_BYTES_FP8,
                    2_880
                ),
            ]
        );
        assert_eq!(manifest.kv_formats()[0].layout(), preferred);
        // The host serves BF16 for auto and bf16 and the FP8 layout for fp8.
        for (requested, dtype, layout) in [
            (
                runtime_host::KvCacheDtypeV1::Auto,
                BundleKvDtypeV1::Bf16,
                model_mimo26_contract::serving::KV_LAYOUT_BF16,
            ),
            (
                runtime_host::KvCacheDtypeV1::Bf16,
                BundleKvDtypeV1::Bf16,
                model_mimo26_contract::serving::KV_LAYOUT_BF16,
            ),
            (
                runtime_host::KvCacheDtypeV1::Fp8,
                BundleKvDtypeV1::Fp8E4m3,
                model_mimo26_contract::serving::KV_LAYOUT_FP8,
            ),
        ] {
            let choice =
                runtime_host::resolve_kv_cache_dtype_v1(&manifest, requested).expect("declared");
            assert_eq!(
                (choice.dtype, choice.automatic),
                (dtype, requested == runtime_host::KvCacheDtypeV1::Auto)
            );
            assert_eq!(
                runtime_host::kv_layout_of_v1(&manifest, dtype)
                    .map(|layout| layout.id().as_str().as_bytes()),
                Some(layout)
            );
        }
        // Serving either format needs no host capability beyond the list.
        assert!(!requires(
            &manifest,
            runtime_bundle_abi::FP8_KV_CAPABILITY_V1
        ));
    }

    #[test]
    fn encoder_package_follows_the_media_serving_gate() {
        let identity = model_mimo26::checkpoint::default_identity();
        let draft = bundle_draft(&assets(None, true), identity, None, &uniform(1));
        let refused = draft
            .as_ref()
            .err()
            .is_some_and(|error| error.to_string().contains("serving path is incomplete"));
        assert_eq!(refused, !model_mimo26_contract::encoders::SERVING_READY);
        if let Ok(draft) = draft {
            let manifest = model_compiler_sdk::plan_bundle_manifest_v1(&draft).unwrap();
            assert!(requires(
                &manifest,
                model_mimo26_contract::encoders::MEDIA_CAPABILITY
            ));
        }
    }

    /// One package serves every rank count: it declares each topology,
    /// and serves each with that rank count's per-rank resources.
    #[test]
    fn each_topology_declares_its_ids_ranks_and_per_rank_resources() {
        let package = manifest(assets(Some(DrafterKind::Mtp), false), 16);
        assert_eq!(package.id().as_str(), BUNDLE_ID);
        assert_eq!(
            package
                .target()
                .topologies()
                .iter()
                .map(|topology| (topology.id().as_str(), topology.rank_count()))
                .collect::<Vec<_>>(),
            Tp::SUPPORTED.map(|tp| (topology_id(tp), tp.ranks()))
        );
        for tp in Tp::SUPPORTED {
            let manifest = package.for_topology(topology_id(tp)).unwrap();
            let topology = &manifest.target().topologies()[0];
            assert_eq!(topology.id().as_str(), topology_id(tp));
            assert_eq!(topology.rank_count(), tp.ranks());
            // BF16 preferred, then FP8, both at this rank count's GA heads.
            assert_eq!(manifest.kv_layout().page_bytes(), ga_page_bytes(tp, false));
            let pages = manifest
                .kv_formats()
                .iter()
                .map(|format| format.layout().page_bytes())
                .collect::<Vec<_>>();
            assert_eq!(pages, [ga_page_bytes(tp, false), ga_page_bytes(tp, true)]);
            // The ring state is declared per state slot.
            assert_eq!(manifest.state_layout().bytes(), ring_state_slot_bytes(tp));
            assert_eq!(
                manifest
                    .state_bytes_v1(Some(model_mimo26_contract::serving::DECODE_ENTRY_ROWS))
                    .unwrap(),
                model_mimo26_contract::topology::ring_state_bytes_per_rank(tp)
            );
            assert!(manifest.state_per_item());
            // These packages carry the MTP drafter and no encoders.
            let limits = manifest.batch_limits().expect("declared batch limits");
            assert_eq!(
                limits.maximum_sequences(),
                maximum_sequences(tp, Some(DrafterKind::Mtp), Encoders::NONE, 16)
            );
            // Every topology counts its planned images (16 bytes per rank
            // here) and its buffers at the package maximum.
            assert_eq!(
                manifest
                    .resource_requirements()
                    .minimum_device_memory_bytes(),
                model_mimo26_contract::topology::minimum_rank_bytes(
                    tp,
                    16,
                    limits.maximum_sequences(),
                    Some(model_mimo26_contract::weight_plan::Drafter::Mtp),
                    Encoders::NONE,
                )
            );
        }
    }

    #[test]
    fn speculation_limits_and_capabilities_follow_the_drafter() {
        // Without a drafter, the module still verifies host drafts.
        let plain = selected(assets(None, false), 1, Tp::TP4);
        let speculation = plain.speculation().expect("host-draft speculation");
        assert!(!speculation.bundle_drafter());
        assert!(speculation.drafter_component().is_none());
        assert_eq!(speculation.maximum_draft_tokens(), 7);
        let limits = plain.batch_limits().expect("declared batch limits");
        assert_eq!(
            (limits.maximum_sequences(), limits.maximum_prefill_tokens()),
            (DECODE_BATCH, PREFILL_TOKENS)
        );
        assert_eq!(plain.id().as_str(), "mimo.v26.flash.v2");
        assert_eq!(
            plain.target().topologies()[0].id().as_str(),
            "mimo-v26-flash-tp4"
        );
        assert_eq!(
            plain.kv_layout().page_bytes(),
            model_mimo26_contract::serving::GA_PAGE_BYTES_BF16
        );
        for capability in [
            SPECULATIVE_CAPABILITY_V1,
            runtime_bundle_abi::SERVED_ITEMS_CAPABILITY_V1,
            "chop.prompt-rows.v2",
            "chop.prefix-cache.v1",
            runtime_bundle_abi::KV_OFFLOAD_CAPABILITY_V1,
        ] {
            assert!(requires(&plain, capability), "{capability}");
        }
        assert_eq!(
            plain.graph_buckets().len(),
            DECODE_BATCH.ilog2() as usize + 2
        );
        // An MTP drafter lets the module draft; speculation remains opt-in.
        let mtp = selected(assets(Some(DrafterKind::Mtp), false), 1, Tp::TP4);
        let speculation = mtp.speculation().unwrap();
        assert_eq!(speculation.maximum_draft_tokens(), 7);
        assert_eq!(
            speculation.bundle_drafter(),
            model_mimo26_contract::serving::BUNDLE_DRAFTS
        );
        assert_eq!(speculation.maximum_verify_tokens(), 32);
        assert_eq!(speculation.row_buckets(), &[2, 4, 8]);
        assert_eq!(
            speculation
                .drafter_component()
                .map(|component| component.as_str()),
            model_mimo26_contract::serving::BUNDLE_DRAFTS.then_some("drafter-rank-images")
        );
        assert!(requires(&mtp, SPECULATIVE_CAPABILITY_V1));
        assert_eq!(maximum_verify_tokens(None), 256);
        assert_eq!(maximum_verify_tokens(Some(DrafterKind::Dflash)), 256);
        // An MTP package serves 32 sequences, in six decode buckets.
        assert_eq!(mtp.batch_limits().unwrap().maximum_sequences(), 32);
        assert_eq!(mtp.graph_buckets().len(), 7);
        // Package maxima follow the images (the bundle's topology tests pin
        // them for the checkpoint's): tiny images leave every topology KV
        // for 256 sequences, and MTP drafts for 32; the checkpoint's TP2
        // text images leave room for 64.
        for tp in Tp::SUPPORTED {
            for media in Encoders::SETS {
                assert_eq!(maximum_sequences(tp, None, media, 16), 256);
                assert_eq!(
                    maximum_sequences(tp, Some(DrafterKind::Dflash), media, 16),
                    256
                );
                assert_eq!(maximum_sequences(tp, Some(DrafterKind::Mtp), media, 16), 32);
            }
        }
        assert_eq!(
            maximum_sequences(Tp::TP2, None, Encoders::NONE, 85_552_652_288),
            64
        );
    }

    /// A package's module must publish its kernel contract identity in its
    /// note; a module without the note, or bytes that are no module, are
    /// refused.
    #[test]
    fn packaged_modules_publish_their_kernel_contract() {
        let fixture = StagedContentV1::file(crate::test_native::fixture());
        let contract = native_contract(&fixture).unwrap();
        assert_eq!(contract.selection, crate::test_native::SELECTION);
        // A real module under `CHOPINFER_PACKAGE_NATIVE_ROOT` publishes its
        // own audited identity.
        let module = StagedContentV1::file(crate::test_native::module());
        let selection = native_contract(&module).unwrap().selection;
        assert!(runtime_bundle::preparation::valid_digest(&selection));
        let noteless = StagedContentV1::file(std::env::current_exe().unwrap());
        assert!(
            native_contract(&noteless)
                .unwrap_err()
                .to_string()
                .contains("no preparation contract note")
        );
        assert!(native_contract(&StagedContentV1::bytes(b"module".to_vec())).is_err());
    }

    /// Opt-in: `CHOPINFER_MIMO_CHECKPOINT` and `CHOPINFER_MIMO_TEST_IDENTITY`
    /// (`rl` or `mopd`). The bound package verifies and installs, and its
    /// binding names the pinned revision and every file it carries.
    #[test]
    fn checkpoint_bound_package_names_every_file() {
        use model_mimo26::checkpoint_provenance::Verifier;
        let (Ok(checkpoint), Ok(selected)) = (
            std::env::var("CHOPINFER_MIMO_CHECKPOINT"),
            std::env::var("CHOPINFER_MIMO_TEST_IDENTITY"),
        ) else {
            return;
        };
        let identity = model_mimo26::checkpoint::identity(&selected).unwrap();
        let checkpoint = Path::new(&checkpoint);
        let verifier = Verifier::new(checkpoint, identity).unwrap();
        assert!(
            verifier
                .verify_tokenizer("tokenizer.json", 2, &"0".repeat(64))
                .is_err()
        );
        let [definition, configuration] = PackageAssets::pinned_tokenizer_with_registry(
            checkpoint,
            identity,
            verifier.registry(),
        )
        .unwrap();
        let output =
            std::env::temp_dir().join(format!("mimo-bound-{}-{selected}", std::process::id()));
        let assets = PackageAssets {
            tokenizer_definition: definition,
            tokenizer_configuration: configuration,
            native_module: StagedContentV1::file(crate::test_native::module()),
            drafter: Some(DrafterKind::Mtp),
            encoders: Encoders::NONE,
            precision: fixture_precision(),
        };
        let emitted =
            emit_verified_direct_package(assets, &uniform(1 << 30), &output, &verifier).unwrap();
        let host = runtime_bundle::BundleHostCompatibilityV1::new(Selected::ARCHITECTURE, 1)
            .unwrap()
            .with_capabilities(
                emitted
                    .manifest()
                    .resource_requirements()
                    .required_capabilities()
                    .iter()
                    .cloned(),
            );
        runtime_bundle::BundlePackageV1::verify_directory(&output, &host).unwrap();
        // The package carries the model's profiles and deployment policy
        // byte for byte.
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
            let model = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
            assert_eq!(
                std::fs::read(output.join(path)).unwrap(),
                std::fs::read(model.join(file)).unwrap()
            );
        }
        let proof: runtime_bundle::checkpoint_binding::CheckpointBinding = serde_json::from_slice(
            &std::fs::read(output.join("components/checkpoint-binding.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(proof.revision, identity.revision);
        let files: Vec<_> = proof.files.keys().map(String::as_str).collect();
        assert_eq!(
            files,
            [
                "components/deployment-policy.json",
                "components/drafter-kind.json",
                "components/frontend.json",
                "components/serve-profiles.json",
                "components/tokenizer.json",
                "components/tokenizer_config.json",
                "native/module.bin",
            ]
        );
        std::fs::remove_dir_all(output).unwrap();
    }
}
