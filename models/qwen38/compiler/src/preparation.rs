//! The packages without rank images a checkpoint serves with, computed from
//! the checkpoint and the native module alone, and the family helper that
//! emits them for `serve --model`.
use crate::{
    Result,
    catalog::Catalog,
    checkpoint::CheckpointSource,
    local_frontend::LocalFrontend,
    package::{self, Drafter, PackageAssets},
    placement::{ImagePlan, bundle_shape_for},
    projection_program::{SourceProgram, UnsuppliedTopology},
};
// The serving profiles the package path serves Qwen3.8 with, which every
// package also carries.
use crate::package::PROFILES;
use kernel_core::precision::execution::PrecisionDocument;
use model_compiler_sdk::native_contract::NativeModule;
use model_compiler_sdk::preparation as shared;
use model_compiler_sdk::quantization::ActivationPolicy;
use model_qwen38_contract::contract::Shape;
use runtime_bundle::preparation::NativeProfile;
use std::path::{Path, PathBuf};

/// The Qwen3.8 family helper (`chopinfer-model-qwen38`).
pub struct QwenFamily;

/// Resolves the checkpoint's imported precision against the native module's
/// inventory under `policy` and binds it to the bundle's views.
pub fn resolve_catalog(
    catalog: &Catalog,
    policy: ActivationPolicy,
    module: &NativeModule,
) -> Result<PrecisionDocument> {
    let shape = bundle_shape_for(catalog.geometry, catalog.ranks)?;
    module.require(&QwenFamily.profile_for(catalog.geometry, catalog.ranks as u32)?)?;
    let source = SourceProgram::new(
        catalog
            .precision
            .as_ref()
            .ok_or("missing imported precision")?,
        shape,
    )?;
    let inventory = module.precision_inventory()?;
    let precision = source.resolve(policy, &inventory)?;
    model_qwen38_contract::prepared::bind(&precision, shape, &inventory)?;
    Ok(precision)
}

/// The prepared native profile of every compiled geometry and served rank
/// count, by geometry ID: `note::prepared_profile`, derived when the helper
/// is built (build.rs) so a request does not rebuild the admitted view table.
fn prepared_profiles() -> Result<&'static [(String, NativeProfile)]> {
    #[derive(serde::Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Compiled {
        geometry: String,
        profile: NativeProfile,
    }
    static PROFILES: std::sync::OnceLock<
        std::result::Result<Vec<(String, NativeProfile)>, String>,
    > = std::sync::OnceLock::new();
    PROFILES
        .get_or_init(|| {
            serde_json::from_str::<Vec<Compiled>>(include_str!(concat!(
                env!("OUT_DIR"),
                "/prepared-profiles.json"
            )))
            .map(|compiled| {
                compiled
                    .into_iter()
                    .map(|c| (c.geometry, c.profile))
                    .collect()
            })
            .map_err(|e| e.to_string())
        })
        .as_ref()
        .map(Vec::as_slice)
        .map_err(|e| e.clone().into())
}

impl QwenFamily {
    /// The native profile the bundle module must declare for `ranks`.
    #[cfg(test)]
    pub fn profile(&self, ranks: u32) -> Result<NativeProfile> {
        self.profile_for(&model_qwen38_contract::geometry::FLASH, ranks)
    }

    pub fn profile_for(
        &self,
        geometry: &'static model_qwen38_contract::geometry::Geometry,
        ranks: u32,
    ) -> Result<NativeProfile> {
        let shape = bundle_shape_for(geometry, ranks as usize)?;
        prepared_profiles()?
            .iter()
            .find(|(id, profile)| {
                id == shape.geometry.id && profile.tensor_parallel_size == shape.ranks()
            })
            .map(|(_, profile)| profile.clone())
            .ok_or_else(|| {
                format!(
                    "no prepared profile is compiled for geometry {} at TP{}",
                    shape.geometry.id,
                    shape.ranks()
                )
                .into()
            })
    }

    /// The installed native module: beside the helper or in the release lib
    /// directory.
    pub fn native_module(&self) -> Result<PathBuf> {
        shared::direct::installed_native_module("libbundle_qwen38.so", "Qwen")
    }
}

/// The checkpoint's source and its precision contract at `shape` for a
/// direct package ([`precision_for`] over the source's catalog).
pub fn direct_precision(
    checkpoint: &Path,
    module: &NativeModule,
    shape: Shape,
    requested: ActivationPolicy,
    supplied: &[PrecisionDocument],
) -> Result<(CheckpointSource, PrecisionDocument)> {
    let source = CheckpointSource::open(checkpoint, shape.ranks() as usize)?;
    if source.catalog.geometry.id != shape.geometry.id {
        return Err("checkpoint geometry differs from requested topology".into());
    }
    let precision = precision_for(&source.catalog, module, shape, requested, supplied)?;
    Ok((source, precision))
}

/// The precision contract at `shape` of the checkpoint `catalog` describes
/// (cataloged for `shape`'s rank count): resolved as cached preparation
/// resolves it, or, when precision is `supplied`, the one supplied document
/// that records this topology's operations, checked against the
/// checkpoint's imported precision and the module.
fn precision_for(
    catalog: &Catalog,
    module: &NativeModule,
    shape: Shape,
    requested: ActivationPolicy,
    supplied: &[PrecisionDocument],
) -> Result<PrecisionDocument> {
    let precision = if supplied.is_empty() {
        resolve_catalog(catalog, requested, module)?
    } else {
        module.require(&QwenFamily.profile_for(shape.geometry, shape.ranks())?)?;
        let program = SourceProgram::new(
            catalog
                .precision
                .as_ref()
                .ok_or("missing imported precision")?,
            shape,
        )?;
        let document = supplied
            .iter()
            .find(|document| program.verify_recorded(document).is_ok())
            .ok_or_else(|| {
                UnsuppliedTopology(
                    "no supplied precision component records this topology's operations".into(),
                )
            })?
            .clone();
        model_qwen38_contract::prepared::bind(&document, shape, &module.precision_inventory()?)?;
        document
    };
    for selection in &precision.operations {
        selection.require_policy(requested)?;
    }
    Ok(precision)
}

/// Why `precision` runs no MTP drafter at `shape`: the MTP tensor and
/// format the checkpoint `catalog` describes encodes without a composition,
/// or the component's missing MTP records.
pub fn drafter_refusal(
    catalog: &Catalog,
    shape: Shape,
    precision: &PrecisionDocument,
) -> Option<String> {
    let error = model_qwen38_contract::projections::support::mtp(shape, precision).err()?;
    Some(
        catalog
            .precision
            .as_ref()
            .and_then(|source| crate::projection_program::drafter_refusal(source, shape))
            .unwrap_or_else(|| error.to_string()),
    )
}

/// The serving profile a direct package carries: the rank count's MTP
/// profile when it has one and the precision executes MTP (a mixed
/// component executes it when it records the MTP layer), else its
/// autoregressive profile. Both the profile request and the emission select
/// through this one function.
fn select_profile(
    shape: Shape,
    precision: &PrecisionDocument,
) -> Result<runtime_bundle::preparation::DirectProfile> {
    let ranks = shape.ranks();
    let mtp = shape.mtp_layers() > 0
        && shared::direct::default_profile(PROFILES, ranks, &["mtp"]).is_ok()
        && model_qwen38_contract::projections::support::mtp(shape, precision).is_ok();
    let mut profile =
        shared::direct::default_profile(PROFILES, ranks, if mtp { &["mtp"] } else { &["none"] })?;
    profile.profile["topology"] = shape.topology().id.into();
    experimental_profile(shape.geometry, &mut profile)?;
    Ok(profile)
}

fn experimental_profile(
    geometry: &model_qwen38_contract::geometry::Geometry,
    profile: &mut runtime_bundle::preparation::DirectProfile,
) -> Result<()> {
    if geometry.id != model_qwen38_contract::geometry::FLASH.id {
        profile.name = format!("{}-{}", geometry.id, profile.name);
        let fields = profile
            .profile
            .as_object_mut()
            .ok_or("invalid serving profile")?;
        fields.insert(
            "qualification".into(),
            format!("unqualified geometry {}", geometry.id).into(),
        );
        fields.insert("evidence".into(), serde_json::json!([]));
        fields.insert("qualified_max_sequences".into(), serde_json::Value::Null);
        fields.insert("status".into(), "experimental".into());
    }
    Ok(())
}

/// The assets of a package without rank images for loading the checkpoint
/// `catalog` describes directly with `precision` and `drafter` (whose views
/// the planned rank images carry): the plan's per-rank sizes checked
/// against the device, and the checkpoint's local `frontend`. No tensor
/// payload is read, and nothing but the checkpoint is: any rank count the
/// placement admits works alike.
pub fn direct_assets(
    catalog: &Catalog,
    precision: PrecisionDocument,
    frontend: &LocalFrontend,
    module: &NativeModule,
    shape: Shape,
    drafter: Drafter,
    runtime: crate::config::RuntimeValues,
) -> Result<package::PreparedAssets> {
    if drafter == Drafter::Mtp && shape.mtp_layers() == 0 {
        return Err("the selected geometry has no MTP drafter".into());
    }
    if drafter == Drafter::Mtp
        && let Some(reason) = drafter_refusal(catalog, shape, &precision)
    {
        return Err(format!("this precision does not run the MTP drafter: {reason}").into());
    }
    if catalog.geometry.id != shape.geometry.id {
        return Err("catalog geometry differs from requested topology".into());
    }
    let plan = ImagePlan::with_precision(catalog, None, &precision)?;
    if plan.ranks.len() != shape.ranks() as usize {
        return Err("the Qwen image plan differs from the requested topology".into());
    }
    let ngram = if model_qwen38_contract::package::ngram_in_host_memory(shape) {
        package::NgramPlacement::Host
    } else {
        package::NgramPlacement::Gpu
    };
    for rank in &plan.ranks {
        package::check_fits(shape, rank.image_bytes, ngram)?;
    }
    let frontend = frontend.clone();
    Ok(package::PreparedAssets {
        assets: PackageAssets {
            shape,
            runtime,
            drafter,
            tokenizer_definition: model_compiler_sdk::StagedContentV1::bytes(frontend.definition),
            tokenizer_configuration: model_compiler_sdk::StagedContentV1::bytes(
                frontend.configuration,
            ),
            native_module: module.staged()?,
            encoders: None,
        },
        frontend: frontend.component,
        precision,
    })
}

/// [`shared::helper::Provider::direct_profile`] with the native module at
/// `module_path`.
fn direct_profile_with(
    root: &Path,
    request: &runtime_bundle::preparation::PrepareDirectRequest,
    module_path: &Path,
) -> Result<runtime_bundle::preparation::DirectProfile> {
    let ranks = request.tensor_parallel_size;
    crate::placement::bundle_ranks(ranks as usize)?;
    let geometry = crate::config::Qwen38Config::load(root)?
        .geometry()?
        .geometry;
    let shape = bundle_shape_for(geometry, ranks as usize)?;
    let module = NativeModule::inspect(module_path)?;
    let policy = ActivationPolicy::from_precedence(request.activation_policy.as_deref(), None)?;
    let (_, precision) = direct_precision(root, &module, shape, policy, &[])?;
    select_profile(shape, &precision)
}

/// Whether the checkpoint `catalog` describes carries (any of) the vision
/// tower.
pub fn has_vision_tower(catalog: &Catalog) -> bool {
    model_qwen38_contract::encoder::views::layout_for(catalog.geometry)
        .iter()
        .any(|view| catalog.tensors.contains_key(&view.name))
}

/// The image positions of the vision encoder a package serving
/// `request`'s modalities plans, when it serves images: the checkpoint
/// `catalog` describes provides text, and images with its vision tower.
pub fn served_encoders(
    request: &runtime_bundle::preparation::PrepareDirectRequest,
    catalog: &Catalog,
) -> Result<Option<crate::encoders::ImagePositions>> {
    use runtime_bundle::preparation::Modality;
    let served = served_modalities(request, catalog)?;
    Ok(served
        .contains(&Modality::Image)
        .then(crate::encoders::ImagePositions::default))
}

/// The concrete modalities selected from the checkpoint's encoder catalog.
pub fn served_modalities(
    request: &runtime_bundle::preparation::PrepareDirectRequest,
    catalog: &Catalog,
) -> Result<std::collections::BTreeSet<runtime_bundle::preparation::Modality>> {
    use runtime_bundle::preparation::Modality;
    let mut provides = std::collections::BTreeSet::from([Modality::Text]);
    if has_vision_tower(catalog) {
        provides.insert(Modality::Image);
    }
    Ok(request.served_modalities(&provides)?)
}

/// A requested topology's failure: an [`UnsuppliedTopology`] says the
/// topology is not served with this checkpoint's precision.
fn not_served(
    shape: Shape,
    error: Box<dyn std::error::Error + Send + Sync>,
) -> Box<dyn std::error::Error + Send + Sync> {
    if error.downcast_ref::<UnsuppliedTopology>().is_some() {
        format!(
            "{} is not served with this checkpoint's precision: {error}",
            shape.topology().id
        )
        .into()
    } else {
        error
    }
}

/// The topologies a package leaves out, each with the reason its precision
/// has no composition there.
pub type Omitted = Vec<(Shape, String)>;

/// Every topology's `draft`, in topology order, for a package that must
/// serve `required`: an [`UnsuppliedTopology`] is left out and returned with
/// its reason, any other failure fails the package, and a package serving no
/// topology fails with the first reason. `required` is drafted first, so a
/// request that cannot be served is refused for its own topology's reason
/// (an [`UnsuppliedTopology`] as not served), whatever another topology's.
#[cfg(test)]
fn topology_drafts<D>(
    required: Option<Shape>,
    draft: impl FnMut(Shape) -> Result<D>,
) -> Result<(Vec<D>, Omitted)> {
    topology_drafts_for(
        required.map_or(&model_qwen38_contract::geometry::FLASH, |s| s.geometry),
        required,
        draft,
    )
}

fn topology_drafts_for<D>(
    geometry: &'static model_qwen38_contract::geometry::Geometry,
    required: Option<Shape>,
    mut draft: impl FnMut(Shape) -> Result<D>,
) -> Result<(Vec<D>, Omitted)> {
    let mut first = match required {
        Some(shape) => Some(draft(shape).map_err(|error| not_served(shape, error))?),
        None => None,
    };
    let mut drafts = Vec::new();
    let mut refusals = Vec::new();
    let mut omitted = Vec::new();
    for shape in [2, 4].map(|n| bundle_shape_for(geometry, n)).into_iter() {
        let shape = shape?;
        let drafted = match first.take_if(|_| required == Some(shape)) {
            Some(drafted) => Ok(drafted),
            None => draft(shape),
        };
        match drafted {
            Ok(draft) => drafts.push(draft),
            Err(error) if error.downcast_ref::<UnsuppliedTopology>().is_some() => {
                eprintln!("{}: not served: {error}", shape.topology().id);
                omitted.push((shape, error.to_string()));
                refusals.push(error);
            }
            Err(error) => return Err(error),
        }
    }
    if drafts.is_empty() {
        return Err(refusals.remove(0));
    }
    Ok((drafts, omitted))
}

/// A direct package and the topologies it leaves out, each with the reason
/// its precision has no composition there.
pub struct DirectPackage {
    pub emitted: model_compiler_sdk::EmittedBundleV1,
    pub omitted: Omitted,
}

/// A topology a direct package serves: the checkpoint's catalog for its rank
/// count, and its precision contract.
struct ServedTopology {
    shape: Shape,
    runtime: crate::config::RuntimeValues,
    catalog: Catalog,
    precision: PrecisionDocument,
}

/// The topologies the checkpoint whose files are `files` (with the precision
/// `imported` from them, both read once for every topology) serves under
/// `policy`, each with its catalog and [`precision_for`]. An
/// [`UnsuppliedTopology`] is left out and reported: a rank count at which
/// the supplied kernels have no composition of an admitted operation, and
/// with `supplied` precision every topology it does not record; the
/// topologies left out are returned with their reasons. The
/// topologies must include `required` (whose [`UnsuppliedTopology`] fails
/// as not served), and any other failure fails.
fn served_topologies(
    files: &model_compiler_sdk::source::SourceFiles,
    imported: &crate::precision_source::SourcePrecision,
    module: &NativeModule,
    policy: ActivationPolicy,
    supplied: &[PrecisionDocument],
    required: Option<Shape>,
) -> Result<(Vec<ServedTopology>, Omitted)> {
    let matched = crate::config::Qwen38Config::from_bytes(
        files.inputs.get("config.json").ok_or("missing config")?,
    )?
    .geometry()?;
    let geometry = matched.geometry;
    topology_drafts_for(geometry, required, |shape| {
        let catalog = Catalog::from_imported(files, imported.clone(), shape.ranks() as usize)?;
        Ok(ServedTopology {
            shape,
            runtime: matched.runtime,
            precision: precision_for(&catalog, module, shape, policy, supplied)?,
            catalog,
        })
    })
}

/// The draft of the package without rank images serving `topology`: the
/// text image (with the MTP views of `drafter`) and, with `encoders`, the
/// vision encoders are planned components.
fn topology_draft(
    topology: ServedTopology,
    frontend: &LocalFrontend,
    module: &NativeModule,
    drafter: Drafter,
    encoders: Option<crate::encoders::ImagePositions>,
) -> Result<model_compiler_sdk::BundleDraftV1> {
    let ServedTopology {
        shape,
        runtime,
        catalog,
        precision,
    } = topology;
    let mut prepared = direct_assets(
        &catalog, precision, frontend, module, shape, drafter, runtime,
    )?;
    // A tower that is present must pass the check; it is never dropped
    // silently.
    if let Some(positions) = encoders {
        prepared.assets.encoders = Some(crate::encoders::EncoderAssets::planned(
            &catalog, positions,
        )?);
    }
    for line in model_compiler_sdk::quantization::identity::report(&prepared.precision)? {
        eprintln!("{}: {line}", shape.topology().id);
    }
    Ok(package::bundle_draft_direct(
        prepared.assets,
        Some(prepared.frontend),
        Some(&prepared.precision),
    )?)
}

/// Emits into `output` the package without rank images serving `topologies`
/// of the checkpoint whose files are `files`, with the local frontend named
/// by `checkpoint_identity`. The files must be unchanged since they were
/// captured.
fn emit_topologies(
    topologies: Vec<ServedTopology>,
    files: &model_compiler_sdk::source::SourceFiles,
    module: &NativeModule,
    checkpoint_identity: &str,
    drafter: Drafter,
    encoders: Option<crate::encoders::ImagePositions>,
    output: &Path,
) -> Result<model_compiler_sdk::EmittedBundleV1> {
    let frontend = LocalFrontend::resolve(&files.inputs, checkpoint_identity)?;
    let drafts = topologies
        .into_iter()
        .map(|topology| topology_draft(topology, &frontend, module, drafter, encoders))
        .collect::<Result<Vec<_>>>()?;
    files.verify()?;
    let emitted = model_compiler_sdk::emit_bundle_directory_v1(
        &model_compiler_sdk::BundleDraftV1::for_topologies(drafts)?,
        output,
    )?;
    module.verify()?;
    Ok(emitted)
}

/// Emits into `output` the package without rank images the checkpoint at
/// `checkpoint` serves with at every topology its precision serves
/// ([`served_topologies`], which must include `required`), from the
/// checkpoint and the native module alone. The checkpoint's files are read
/// and its precision imported once for every topology; the topologies left
/// out are returned with the package.
#[allow(clippy::too_many_arguments)]
pub fn emit_direct_package(
    checkpoint: &Path,
    module: &NativeModule,
    policy: ActivationPolicy,
    supplied: &[PrecisionDocument],
    checkpoint_identity: &str,
    drafter: Drafter,
    encoders: Option<crate::encoders::ImagePositions>,
    required: Option<Shape>,
    output: &Path,
) -> Result<DirectPackage> {
    let files = CheckpointSource::capture(checkpoint)?;
    let imported = CheckpointSource::import(&files)?;
    let (topologies, omitted) =
        served_topologies(&files, &imported, module, policy, supplied, required)?;
    let emitted = emit_topologies(
        topologies,
        &files,
        module,
        checkpoint_identity,
        drafter,
        encoders,
        output,
    )?;
    Ok(DirectPackage { emitted, omitted })
}

/// The emission inputs ([`shared::helper::Provider::direct_inputs`]) with
/// the native module at `module_path`. Besides the checkpoint, the request
/// and the helper (whose code carries the serving profiles, the package
/// declarations, the precision recipes and the encoder plans), emission
/// reads only the native module, which it packages and whose precision
/// inventory it resolves against, and [`package::WIDE_ROWS_ENV`].
fn geometry_direct_inputs(root: &Path, module: &Path) -> Result<Vec<u8>> {
    let matched = crate::config::Qwen38Config::load(root)?.geometry()?;
    let mut inputs: serde_json::Value = serde_json::from_slice(&direct_inputs_with(module)?)?;
    inputs["geometry"] = serde_json::json!({"id": matched.geometry.id,
        "execution": serde_json::from_str::<serde_json::Value>(matched.geometry.document)?,
        "indexer_budget": matched.runtime.indexer_budget});
    Ok(serde_json::to_vec(&inputs)?)
}

fn direct_inputs_with(module_path: &Path) -> Result<Vec<u8>> {
    shared::direct::native_module_inputs(module_path, &[package::WIDE_ROWS_ENV])
}

/// [`shared::helper::Provider::emit_direct_with_inputs`] with the native
/// module at `module_path`: with emission `inputs`, only the module they
/// identify.
pub(crate) fn emit_direct_with(
    root: &Path,
    request: &runtime_bundle::preparation::PrepareDirectRequest,
    checkpoint_identity: &str,
    output: &Path,
    module_path: &Path,
    inputs: Option<&[u8]>,
) -> Result<shared::direct::DirectEmission> {
    let ranks = request.tensor_parallel_size;
    crate::placement::bundle_ranks(ranks as usize)?;
    let geometry = crate::config::Qwen38Config::load(root)?
        .geometry()?
        .geometry;
    let shape = bundle_shape_for(geometry, ranks as usize)?;
    let module = shared::direct::native_module_inspected(module_path, inputs)?;
    let policy = ActivationPolicy::from_precedence(request.activation_policy.as_deref(), None)?;
    // One reading of the checkpoint for every topology; the requested one's
    // precision and catalog also select the profile and the encoders.
    let files = CheckpointSource::capture(root)?;
    let imported = CheckpointSource::import(&files)?;
    let (topologies, _) = served_topologies(&files, &imported, &module, policy, &[], Some(shape))?;
    let requested = topologies
        .iter()
        .find(|topology| topology.shape == shape)
        .ok_or("the package does not serve the requested topology")?;
    let profile = select_profile(shape, &requested.precision)?;
    let drafter = if profile.profile["speculation"]["drafter"] == "mtp" {
        Drafter::Mtp
    } else {
        Drafter::None
    };
    // Image inputs through the planned vision encoders, as the package
    // path serves them, when the checkpoint has a vision tower and the
    // request serves images.
    let encoders = served_encoders(request, &requested.catalog)?;
    let package = emit_topologies(
        topologies,
        &files,
        &module,
        checkpoint_identity,
        drafter,
        encoders,
        output,
    )?;
    Ok(shared::direct::DirectEmission { package, profile })
}

impl shared::helper::Provider for QwenFamily {
    fn name(&self) -> &str {
        "chopinfer-model-qwen38"
    }
    fn deployment_policy(&self) -> Option<&str> {
        Some(package::DEPLOYMENT_POLICY)
    }
    fn profiles(&self) -> Vec<String> {
        model_qwen38_contract::geometry::GEOMETRIES
            .iter()
            .flat_map(|g| {
                Shape::ALL.map(|s| {
                    model_qwen38_contract::note::profile_name(Shape::new(g, s.ranks), true)
                })
            })
            .collect()
    }
    fn probe(&self, path: &Path, ranks: u32) -> Result<()> {
        crate::placement::bundle_ranks(ranks as usize)?;
        let matched = crate::config::Qwen38Config::load(path)?.geometry()?;
        self.profile_for(matched.geometry, ranks)?;
        Ok(())
    }
    /// Text with the MTP drafter where the rank count has an MTP serving
    /// profile (the default the package path serves) and the precision
    /// executes it, otherwise autoregressive text; the weight plan supplies
    /// the rank images, which carry the drafter's views, and the indexes.
    /// With a vision tower, the package also plans the vision encoders, so
    /// image inputs serve with the grid positions the package path uses.
    fn emit_direct(
        &self,
        root: &Path,
        request: &runtime_bundle::preparation::PrepareDirectRequest,
        checkpoint_identity: &str,
        output: &Path,
    ) -> Result<shared::direct::DirectEmission> {
        self.emit_direct_with_inputs(root, request, checkpoint_identity, output, None)
    }
    fn direct_modalities(
        &self,
        root: &Path,
        request: &runtime_bundle::preparation::PrepareDirectRequest,
    ) -> Result<Option<std::collections::BTreeSet<runtime_bundle::preparation::Modality>>> {
        let catalog = Catalog::load_ranks(root, request.tensor_parallel_size as usize)?;
        Ok(Some(served_modalities(request, &catalog)?))
    }

    /// The installed native module and the environment the emission reads,
    /// so a warm start reuses the published package without emitting it.
    fn direct_inputs(
        &self,
        root: &Path,
        _request: &runtime_bundle::preparation::PrepareDirectRequest,
    ) -> Result<Option<Vec<u8>>> {
        geometry_direct_inputs(root, &self.native_module()?).map(Some)
    }
    fn emit_direct_with_inputs(
        &self,
        root: &Path,
        request: &runtime_bundle::preparation::PrepareDirectRequest,
        checkpoint_identity: &str,
        output: &Path,
        inputs: Option<&[u8]>,
    ) -> Result<shared::direct::DirectEmission> {
        emit_direct_with(
            root,
            request,
            checkpoint_identity,
            output,
            &self.native_module()?,
            inputs,
        )
    }
    fn names_direct_profile(&self) -> bool {
        true
    }
    fn direct_profile(
        &self,
        root: &Path,
        request: &runtime_bundle::preparation::PrepareDirectRequest,
    ) -> Result<runtime_bundle::preparation::DirectProfile> {
        direct_profile_with(root, request, &self.native_module()?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The profiles compiled into the helper are the contract's typed
    /// identity for every compiled geometry and served rank count, and
    /// `profile_for` serves them.
    #[test]
    fn compiled_prepared_profiles_equal_the_contract_derivation() {
        let mut expected = Vec::new();
        for geometry in model_qwen38_contract::geometry::GEOMETRIES {
            for ranks in [
                model_qwen38_contract::contract::Ranks::TP2,
                model_qwen38_contract::contract::Ranks::TP4,
            ] {
                let Some(shape) = Shape::of_geometry(geometry, ranks) else {
                    continue;
                };
                let profile: NativeProfile =
                    serde_json::from_value(model_qwen38_contract::note::prepared_profile(shape))
                        .unwrap();
                assert_eq!(
                    QwenFamily.profile_for(geometry, shape.ranks()).unwrap(),
                    profile,
                    "{} TP{}",
                    geometry.id,
                    shape.ranks()
                );
                expected.push((geometry.id.to_owned(), profile));
            }
        }
        assert_eq!(prepared_profiles().unwrap(), expected.as_slice());
        assert!(
            QwenFamily
                .profile_for(&model_qwen38_contract::geometry::FLASH, 3)
                .is_err()
        );
    }

    // Keep this fixture equal to runtime-host's Profile schema.
    #[derive(Debug, serde::Deserialize, serde::Serialize)]
    #[serde(deny_unknown_fields)]
    struct Qualification {
        pub bundle_id: String,
        pub topology: String,
        pub tensor_parallel: usize,
        pub devices_required: usize,
        pub devices: Vec<usize>,
        pub speculation: HostSpeculation,
        pub max_sequences: usize,
        pub qualified_max_sequences: Option<usize>,
        pub qualification: String,
        pub kv_cache_dtype: String,
        pub prefix_cache: bool,
        pub max_batch_tokens: usize,
        pub prefill_chunk_tokens: usize,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        pub prefill_budget: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        pub decode_tbt_target_ms: Option<u32>,
        pub queue_depth: Option<usize>,
        pub kv_memory_fraction: f64,
        pub status: ProfileStatus,
        pub evidence: Vec<String>,
    }

    #[derive(Debug, serde::Deserialize, serde::Serialize)]
    #[serde(rename_all = "lowercase")]
    enum ProfileStatus {
        Qualified,
        Experimental,
    }

    #[derive(Debug, serde::Deserialize, serde::Serialize)]
    #[serde(deny_unknown_fields)]
    struct HostSpeculation {
        pub mode: String,
        pub tokens: usize,
        pub draft_cost: String,
        /// Packaging requirement, interpreted by the model's preparation tools.
        pub drafter: String,
    }

    /// Each representative format mix selects the MTP drafter exactly where
    /// the pre-port gate (every text operation in a registered legacy
    /// format) did, supported and unsupported mixes alike.
    #[test]
    fn mtp_selection_keeps_the_former_legacy_gate() {
        for (label, document, _) in crate::projection_program::baseline::documents() {
            for shape in Shape::ALL {
                let former = shape.mtp_layers() > 0
                    && shared::direct::default_profile(PROFILES, shape.ranks(), &["mtp"]).is_ok()
                    && crate::projection_program::baseline::legacy_text(&document);
                let profile = select_profile(shape, &document).unwrap();
                assert_eq!(
                    profile.profile["speculation"]["drafter"] == "mtp",
                    former,
                    "{label} at TP{}",
                    shape.ranks()
                );
            }
        }
    }

    #[test]
    fn non_flash_profiles_keep_the_host_schema() {
        static GEOMETRY: model_qwen38_contract::geometry::Geometry =
            model_qwen38_contract::geometry::Geometry {
                id: "profile-test",
                ..model_qwen38_contract::geometry::FLASH
            };
        let precision = PrecisionDocument {
            version: 1,
            operations: Vec::new(),
        };
        for flash in Shape::ALL {
            let baseline = select_profile(flash, &precision).unwrap();
            let mut profile = baseline.clone();
            experimental_profile(&GEOMETRY, &mut profile).unwrap();
            assert_eq!(
                baseline
                    .profile
                    .as_object()
                    .unwrap()
                    .keys()
                    .collect::<Vec<_>>(),
                profile
                    .profile
                    .as_object()
                    .unwrap()
                    .keys()
                    .collect::<Vec<_>>()
            );
            let qualification: Qualification =
                serde_json::from_value(profile.profile.clone()).unwrap();
            assert!(matches!(qualification.status, ProfileStatus::Experimental));
            assert_eq!(
                qualification.qualification,
                "unqualified geometry profile-test"
            );
            assert!(qualification.evidence.is_empty());
            assert_eq!(qualification.qualified_max_sequences, None);
            for geometry in model_qwen38_contract::geometry::GEOMETRIES {
                let shape = Shape::new(geometry, flash.ranks);
                let selected = select_profile(shape, &precision).unwrap();
                assert_eq!(selected.profile["topology"], shape.topology().id);
                if geometry.id != model_qwen38_contract::geometry::FLASH.id {
                    let qualification: Qualification =
                        serde_json::from_value(selected.profile).unwrap();
                    assert!(matches!(qualification.status, ProfileStatus::Experimental));
                }
            }
        }
    }

    #[test]
    fn probe_refuses_unlisted_config_before_headers() {
        use shared::helper::Provider;
        let root = std::env::temp_dir().join(format!("qwen-unlisted-probe-{}", std::process::id()));
        std::fs::create_dir_all(&root).unwrap();
        let mut config: serde_json::Value =
            serde_json::from_slice(include_bytes!("../fixtures/config.json")).unwrap();
        config["text_config"]["hidden_size"] = 2304.into();
        std::fs::write(
            root.join("config.json"),
            serde_json::to_vec(&config).unwrap(),
        )
        .unwrap();
        let error = QwenFamily.probe(&root, 4).unwrap_err().to_string();
        std::fs::remove_dir_all(root).unwrap();
        assert!(
            error.contains("unsupported geometry:") && error.contains("text_config.hidden_size"),
            "{error}"
        );
        assert!(!error.contains("safetensors"), "{error}");
    }

    #[test]
    fn direct_input_keys_include_the_selected_geometry() {
        let root =
            std::env::temp_dir().join(format!("qwen-geometry-inputs-{}", std::process::id()));
        std::fs::create_dir_all(&root).unwrap();
        let module = root.join("module.bin");
        std::fs::write(&module, b"module").unwrap();
        let mut previous = None;
        for geometry in model_qwen38_contract::geometry::GEOMETRIES {
            let mut config: serde_json::Value =
                serde_json::from_slice(include_bytes!("../fixtures/config.json")).unwrap();
            fn overlay(target: &mut serde_json::Value, source: &serde_json::Value) {
                if let Some(fields) = source.as_object() {
                    for (key, value) in fields {
                        overlay(&mut target[key], value);
                    }
                } else {
                    *target = source.clone();
                }
            }
            overlay(
                &mut config,
                &serde_json::from_str(geometry.document).unwrap(),
            );
            config["text_config"]["indexer_budget"] = geometry.limits.indexer_budget.into();
            std::fs::write(
                root.join("config.json"),
                serde_json::to_vec(&config).unwrap(),
            )
            .unwrap();
            let inputs = geometry_direct_inputs(&root, &module).unwrap();
            let value: serde_json::Value = serde_json::from_slice(&inputs).unwrap();
            assert_eq!(value["geometry"]["id"], geometry.id);
            assert_eq!(
                value["geometry"]["indexer_budget"],
                geometry.limits.indexer_budget
            );
            if let Some(previous) = previous {
                assert_ne!(inputs, previous);
            }
            previous = Some(inputs);
        }
        std::fs::remove_dir_all(root).unwrap();
    }

    /// A mixed NVFP4 package serves TP2 and TP4. Where the supplied kernels
    /// compose no grouped NVFP4 operation at a rank count (TP2's are
    /// withheld here), it serves the rest and leaves that topology out,
    /// returning why, whether or not it requires TP4. CPU fixture; the real
    /// checkpoint case below is the manual check.
    #[test]
    fn a_mixed_nvfp4_package_leaves_out_only_an_unsupplied_topology() {
        use crate::projection_program::{routed_nvfp4_fixture, routed_nvfp4_fixture_without};
        let (drafts, omitted) =
            topology_drafts(None, |shape| routed_nvfp4_fixture(shape, |_| {})).unwrap();
        assert_eq!(drafts.len(), 2);
        assert!(omitted.is_empty());
        for required in [None, Some(Shape::TP4)] {
            let (drafts, omitted) = topology_drafts(required, |shape| {
                routed_nvfp4_fixture_without(shape, Some(Shape::TP2), |_| {})
            })
            .unwrap();
            assert_eq!(drafts.len(), 1);
            assert!(drafts[0].iter().all(|selection| {
                selection
                    .bindings
                    .iter()
                    .all(|b| b.implementation.stages.iter().all(|s| !s.is_empty()))
            }));
            assert_eq!(
                omitted.iter().map(|(shape, _)| *shape).collect::<Vec<_>>(),
                [Shape::TP2]
            );
            for (shape, reason) in &omitted {
                assert!(
                    reason.contains(&format!("Nvfp4 composition at TP{}", shape.ranks())),
                    "{reason}"
                );
            }
        }
    }

    /// A mixed component carries the MTP profile at TP2 and TP4 when it
    /// records the MTP layer, of either mixed layout, and the
    /// autoregressive profile when it does not (a component prepared before
    /// MTP records), which `--drafter mtp` refuses with a regenerate message.
    #[test]
    fn mixed_components_select_mtp_profiles_by_their_mtp_records() {
        use crate::projection_program::{drafter_records, routed_nvfp4_fixture};
        for shape in [Shape::TP2, Shape::TP4] {
            let text = routed_nvfp4_fixture(shape, |_| {}).unwrap();
            let document = |operations| PrecisionDocument {
                version: 1,
                operations,
            };
            let unrecorded = document(text.clone());
            assert_eq!(
                select_profile(shape, &unrecorded).unwrap().name,
                format!("tp{}-ar", shape.ranks())
            );
            let catalog = crate::catalog::synthetic(shape.ranks() as usize);
            let refusal = drafter_refusal(&catalog, shape, &unrecorded).unwrap();
            assert!(refusal.contains("regenerate"), "{refusal}");
            for layout in crate::mixed_layouts::Layout::ALL {
                let recorded = document(
                    text.iter()
                        .cloned()
                        .chain(drafter_records(layout, shape))
                        .collect(),
                );
                assert_eq!(
                    select_profile(shape, &recorded).unwrap().name,
                    format!("tp{}-mtp", shape.ranks()),
                    "{}",
                    layout.name()
                );
                assert_eq!(drafter_refusal(&catalog, shape, &recorded), None);
            }
        }
    }

    /// Requesting TP2 of a mixed NVFP4 package whose kernels compose no
    /// grouped NVFP4 operation there fails with a message that names the
    /// topology and the missing composition. CPU fixture; the
    /// real checkpoint case below is the manual check.
    #[test]
    fn requesting_tp2_of_a_mixed_nvfp4_package_fails_clearly() {
        use crate::projection_program::routed_nvfp4_fixture_without;
        let Err(error) = topology_drafts(Some(Shape::TP2), |shape| {
            routed_nvfp4_fixture_without(shape, Some(Shape::TP2), |_| {})
        }) else {
            panic!("TP2 was served");
        };
        let message = error.to_string();
        assert!(
            message.starts_with("qwen38-flash-tp2 is not served with this checkpoint's precision")
                && message.contains("Nvfp4 composition at TP2")
                && !message.contains("no supplied implementation"),
            "{message}"
        );
    }

    /// A mixed NVFP4 operation made invalid at one rank count (TP2, here
    /// with no NVFP4 composition either) fails the whole package instead of
    /// leaving TP2 out, whether validation or the inventory refuses it, and
    /// whether or not the package requires TP2. CPU fixture.
    #[test]
    fn an_invalid_operation_at_one_rank_count_aborts_the_package() {
        use crate::projection_program::routed_nvfp4_fixture_without;
        use kernel_core::precision::{CodeEncoding, NibbleOrder, ScaleConvention};
        let reciprocal = |op: &mut kernel_core::precision::OperationSpec| {
            op.weights[0].local_scale.as_mut().unwrap().convention = ScaleConvention::Reciprocal;
        };
        let even_high = |op: &mut kernel_core::precision::OperationSpec| {
            op.weights[0].encoding = CodeEncoding::E2M1(NibbleOrder::EvenHigh);
        };
        for invalid in [
            &reciprocal as &dyn Fn(&mut kernel_core::precision::OperationSpec),
            &even_high,
        ] {
            for required in [None, Some(Shape::TP4)] {
                let result = topology_drafts(required, |shape| {
                    routed_nvfp4_fixture_without(shape, Some(Shape::TP2), |op| {
                        if shape == Shape::TP2 {
                            invalid(op);
                        }
                    })
                });
                let Err(error) = result else {
                    panic!("the package left the invalid TP2 operation out");
                };
                assert!(
                    error.downcast_ref::<UnsuppliedTopology>().is_none(),
                    "{error}"
                );
                assert!(!error.to_string().contains("not served"), "{error}");
            }
        }
    }

    /// A request is refused for its own topology's reason, as the
    /// per-topology emission refused it, even when another topology fails
    /// otherwise: TP2 of the synthetic NVFP4 experts, without TP2's
    /// grouped compositions, is not served, whatever an invalid TP4
    /// operation says. CPU fixture.
    #[test]
    fn a_request_is_refused_for_its_own_topology_first() {
        use crate::projection_program::routed_nvfp4_fixture_without;
        use kernel_core::precision::ScaleConvention;
        let Err(error) = topology_drafts(Some(Shape::TP2), |shape| {
            routed_nvfp4_fixture_without(shape, Some(Shape::TP2), |op| {
                if shape == Shape::TP4 {
                    op.weights[0].local_scale.as_mut().unwrap().convention =
                        ScaleConvention::Reciprocal;
                }
            })
        }) else {
            panic!("TP2 was served");
        };
        let message = error.to_string();
        assert!(
            message.starts_with("qwen38-flash-tp2 is not served with this checkpoint's precision")
                && message.contains("Nvfp4 composition at TP2"),
            "{message}"
        );
    }

    /// The listed checkpoints whose routed experts are NVFP4: these tests
    /// run on request only.
    fn routed_nvfp4_checkpoints() -> Vec<PathBuf> {
        crate::checkpoint::extra_checkpoints(
            "NVFP4 routed experts",
            crate::checkpoint::routed_nvfp4,
        )
        .into_iter()
        .map(|(root, _)| root)
        .collect()
    }

    /// A built native module, or a failure naming its variable.
    fn native_module() -> PathBuf {
        std::env::var("CHOPINFER_QWEN38_NATIVE_MODULE")
            .expect("set CHOPINFER_QWEN38_NATIVE_MODULE to a built libbundle_qwen38.so")
            .into()
    }

    fn direct_request(
        root: &Path,
        ranks: u32,
        policy: Option<&str>,
    ) -> runtime_bundle::preparation::PrepareDirectRequest {
        runtime_bundle::preparation::PrepareDirectRequest {
            path: root.into(),
            cache_root: PathBuf::new(),
            tensor_parallel_size: ranks,
            activation_policy: policy.map(str::to_string),
            modalities: Default::default(),
        }
    }

    /// `prepare --direct` on each listed checkpoint with NVFP4 routed
    /// experts (whatever its dense, lookup, MTP and norm encodings) emits
    /// its package at TP2 and TP4 with the MTP drafter.
    #[test]
    #[ignore = "needs a checkpoint with NVFP4 routed experts and a native module: run with \
                --ignored and CHOPINFER_QWEN38_EXTRA_CHECKPOINTS and \
                CHOPINFER_QWEN38_NATIVE_MODULE set"]
    fn mixed_direct_packages_serve_tp2_and_tp4_with_mtp() {
        for root in routed_nvfp4_checkpoints() {
            mixed_direct_package_serves_tp2_and_tp4_with_mtp(&root, &native_module());
        }
    }

    /// The mixed export at `root` emits its direct package at TP2 and TP4
    /// with the MTP drafter under the `auto` and `bf16` policies, and the
    /// package serves both topologies; the `checkpoint` policy is refused
    /// at each.
    fn mixed_direct_package_serves_tp2_and_tp4_with_mtp(root: &Path, native: &Path) {
        let identity = shared::direct::checkpoint_identity(root).unwrap();
        let output = std::env::temp_dir().join(format!("qwen-mixed-direct-{}", std::process::id()));
        for ranks in [2, 4] {
            for policy in [None, Some("bf16")] {
                let _ = std::fs::remove_dir_all(&output);
                let emitted = emit_direct_with(
                    root,
                    &direct_request(root, ranks, policy),
                    &identity,
                    &output,
                    native,
                    None,
                )
                .unwrap();
                assert_eq!(emitted.profile.name, format!("tp{ranks}-mtp"), "{policy:?}");
            }
            let _ = std::fs::remove_dir_all(&output);
            let Err(error) = emit_direct_with(
                root,
                &direct_request(root, ranks, Some("checkpoint")),
                &identity,
                &output,
                native,
                None,
            ) else {
                panic!("TP{ranks} served the checkpoint policy");
            };
            assert!(!output.exists(), "{error}");
        }
        let module = NativeModule::inspect(native).unwrap();
        let _ = std::fs::remove_dir_all(&output);
        let package = emit_direct_package(
            root,
            &module,
            ActivationPolicy::Auto,
            &[],
            &identity,
            Drafter::Mtp,
            None,
            Some(Shape::TP4),
            &output,
        )
        .unwrap();
        let _ = std::fs::remove_dir_all(&output);
        assert!(package.omitted.is_empty(), "{:?}", package.omitted);
    }

    /// The package each topology emitted by reading the checkpoint on its
    /// own: its catalog, precision and frontend opened for it alone, the
    /// requested topology's precision resolved again to select the profile,
    /// and the vision and encoder catalogs loaded again. An emission reads
    /// the checkpoint once for all of them and must emit the same bytes.
    fn independently_read_package(
        root: &Path,
        module: &NativeModule,
        request: &runtime_bundle::preparation::PrepareDirectRequest,
        checkpoint_identity: &str,
        output: &Path,
    ) -> Result<shared::direct::DirectEmission> {
        let ranks = request.tensor_parallel_size;
        crate::placement::bundle_ranks(ranks as usize)?;
        let required = bundle_shape_for(
            crate::config::Qwen38Config::load(root)?
                .geometry()?
                .geometry,
            ranks as usize,
        )?;
        let policy = ActivationPolicy::from_precedence(request.activation_policy.as_deref(), None)?;
        let (_, precision) = direct_precision(root, module, required, policy, &[])
            .map_err(|error| not_served(required, error))?;
        let profile = select_profile(required, &precision)?;
        let drafter = if profile.profile["speculation"]["drafter"] == "mtp" {
            Drafter::Mtp
        } else {
            Drafter::None
        };
        let encoders = served_encoders(request, &Catalog::load_ranks(root, ranks as usize)?)?;
        let mut drafts = Vec::new();
        for shape in Shape::ALL {
            let (source, precision) = match direct_precision(root, module, shape, policy, &[]) {
                Ok(resolved) => resolved,
                Err(error)
                    if shape != required
                        && error.downcast_ref::<UnsuppliedTopology>().is_some() =>
                {
                    continue;
                }
                Err(error) => return Err(error),
            };
            let frontend = LocalFrontend::resolve(&source.files.inputs, checkpoint_identity)?;
            let mut prepared = direct_assets(
                &source.catalog,
                precision,
                &frontend,
                module,
                shape,
                drafter,
                crate::config::Qwen38Config::load(root)?.geometry()?.runtime,
            )?;
            if let Some(positions) = encoders {
                prepared.assets.encoders = Some(crate::encoders::EncoderAssets::planned(
                    &Catalog::load_ranks(root, shape.ranks() as usize)?,
                    positions,
                )?);
            }
            drafts.push(package::bundle_draft_direct(
                prepared.assets,
                Some(prepared.frontend),
                Some(&prepared.precision),
            )?);
        }
        let package = model_compiler_sdk::emit_bundle_directory_v1(
            &model_compiler_sdk::BundleDraftV1::for_topologies(drafts)?,
            output,
        )?;
        Ok(shared::direct::DirectEmission { package, profile })
    }

    /// Reading the checkpoint once per emission emits, for each requested
    /// rank count and activation policy, what reading it once per topology
    /// emits: the same manifest (so the same package identity) and the same
    /// whole serving profile where the checkpoint serves the request, and
    /// the same refusal where it does not. The block-FP8 checkpoint
    /// (`CHOPINFER_QWEN38_CHECKPOINT`) and each listed checkpoint with
    /// NVFP4 routed experts (`CHOPINFER_QWEN38_EXTRA_CHECKPOINTS`) serve TP2
    /// and TP4 under `auto` and `bf16`; none serves the `checkpoint` policy.
    #[test]
    #[ignore = "needs real checkpoints and a native module: run with --ignored and \
                CHOPINFER_QWEN38_CHECKPOINT, CHOPINFER_QWEN38_EXTRA_CHECKPOINTS and \
                CHOPINFER_QWEN38_NATIVE_MODULE set"]
    fn one_reading_emits_the_package_of_one_reading_per_topology() {
        let fp8 = PathBuf::from(
            std::env::var("CHOPINFER_QWEN38_CHECKPOINT")
                .expect("set CHOPINFER_QWEN38_CHECKPOINT to a Qwen3.8 checkpoint"),
        );
        let native = native_module();
        // Each checkpoint with the rank counts it serves.
        let mixed = routed_nvfp4_checkpoints();
        let checkpoints = std::iter::once(&fp8)
            .chain(&mixed)
            .map(|root| (root, &[2, 4][..]));
        let scratch = std::env::temp_dir().join(format!("qwen-one-reading-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&scratch);
        std::fs::create_dir_all(&scratch).unwrap();
        let module = NativeModule::inspect(&native).unwrap();
        let manifest = |package: &model_compiler_sdk::EmittedBundleV1| {
            runtime_weights::sha256_bytes(&package.manifest().to_canonical_json().unwrap())
        };
        for (index, (root, served)) in checkpoints.into_iter().enumerate() {
            let identity = shared::direct::checkpoint_identity(root).unwrap();
            for ranks in [2, 4] {
                for policy in [None, Some("bf16"), Some("checkpoint")] {
                    let case = format!("{} TP{ranks} {policy:?}", root.display());
                    let request = direct_request(root, ranks, policy);
                    let output = |kind: &str| {
                        scratch.join(format!(
                            "{kind}-{index}-{ranks}-{}",
                            policy.unwrap_or("auto")
                        ))
                    };
                    let once =
                        emit_direct_with(root, &request, &identity, &output("one"), &native, None);
                    let each = independently_read_package(
                        root,
                        &module,
                        &request,
                        &identity,
                        &output("each"),
                    );
                    match (once, each) {
                        (Ok(once), Ok(each))
                            if served.contains(&ranks) && policy != Some("checkpoint") =>
                        {
                            assert_eq!(manifest(&once.package), manifest(&each.package), "{case}");
                            assert_eq!(once.profile, each.profile, "{case}");
                        }
                        (Err(once), Err(each))
                            if !served.contains(&ranks) || policy == Some("checkpoint") =>
                        {
                            assert_eq!(once.to_string(), each.to_string(), "{case}");
                        }
                        (once, each) => panic!(
                            "{case}: one reading {:?}, one reading per topology {:?}",
                            once.map(|emission| emission.profile.name),
                            each.map(|emission| emission.profile.name)
                        ),
                    }
                }
            }
        }
        std::fs::remove_dir_all(&scratch).unwrap();
    }

    /// The direct package of a real checkpoint carries the default profile's
    /// MTP drafter and the planned vision encoders, with image inputs, and
    /// its TP4 topology serves context-parallel sizes 1 and 2 (TP2 only 1).
    #[test]
    #[ignore = "needs a real checkpoint and native module: run with --ignored and \
                CHOPINFER_QWEN38_CHECKPOINT and CHOPINFER_QWEN38_NATIVE_MODULE set"]
    fn a_real_direct_package_plans_the_drafter_and_the_encoders() {
        // Run on request only, so missing inputs fail rather than pass.
        let root = std::env::var("CHOPINFER_QWEN38_CHECKPOINT")
            .expect("set CHOPINFER_QWEN38_CHECKPOINT to a Qwen3.8 checkpoint");
        let native = std::env::var("CHOPINFER_QWEN38_NATIVE_MODULE")
            .expect("set CHOPINFER_QWEN38_NATIVE_MODULE to a built libbundle_qwen38.so");
        let output = std::env::temp_dir().join(format!("qwen-direct-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&output);
        let request = runtime_bundle::preparation::PrepareDirectRequest {
            path: root.clone().into(),
            cache_root: PathBuf::new(),
            tensor_parallel_size: 4,
            activation_policy: None,
            modalities: Default::default(),
        };
        let identity = shared::direct::checkpoint_identity(Path::new(&root)).unwrap();
        let emitted = emit_direct_with(
            Path::new(&root),
            &request,
            &identity,
            &output,
            Path::new(&native),
            None,
        )
        .unwrap();
        assert_eq!(emitted.profile.name, "tp4-mtp");
        let manifest: serde_json::Value =
            serde_json::from_slice(&emitted.package.manifest().to_canonical_json().unwrap())
                .unwrap();
        let _ = std::fs::remove_dir_all(&output);
        assert!(
            manifest["compiled_images"]
                .as_array()
                .is_none_or(Vec::is_empty)
        );
        let planned: Vec<&str> = manifest["planned_components"]
            .as_array()
            .unwrap()
            .iter()
            .map(|c| c["role"].as_str().unwrap())
            .collect();
        assert!(planned.iter().any(|r| r.contains("encoder")), "{planned:?}");
        assert_eq!(manifest["speculation"]["drafter_kind"], "mtp");
        let capabilities = manifest["resource_requirements"]["required_capabilities"].to_string();
        assert!(capabilities.contains("media"), "{capabilities}");
        assert_eq!(
            manifest["target"]["topologies"],
            serde_json::json!([
                {"id":"qwen38-flash-tp2","rank_count":2},
                {"id":"qwen38-flash-tp4","rank_count":4,"context_parallel_sizes":[2]}
            ])
        );
    }

    /// [`QwenFamily`] with the native module at a given path.
    struct At(PathBuf);

    impl shared::helper::Provider for At {
        fn name(&self) -> &str {
            QwenFamily.name()
        }
        fn profiles(&self) -> Vec<String> {
            QwenFamily.profiles()
        }
        fn probe(&self, path: &Path, ranks: u32) -> Result<()> {
            QwenFamily.probe(path, ranks)
        }
        fn direct_inputs(
            &self,
            root: &Path,
            _request: &runtime_bundle::preparation::PrepareDirectRequest,
        ) -> Result<Option<Vec<u8>>> {
            geometry_direct_inputs(root, &self.0).map(Some)
        }
        fn emit_direct_with_inputs(
            &self,
            root: &Path,
            request: &runtime_bundle::preparation::PrepareDirectRequest,
            checkpoint_identity: &str,
            output: &Path,
            inputs: Option<&[u8]>,
        ) -> Result<shared::direct::DirectEmission> {
            emit_direct_with(root, request, checkpoint_identity, output, &self.0, inputs)
        }
    }

    /// A warm start reuses the published package of a real checkpoint
    /// without emitting it; another native module emits its own package,
    /// another rank count emits once and then reuses the same package with
    /// its own profile, and a damaged package is replaced rather than reused.
    #[test]
    #[ignore = "needs a real checkpoint and native module: run with --ignored and \
                CHOPINFER_QWEN38_CHECKPOINT and CHOPINFER_QWEN38_NATIVE_MODULE set"]
    fn a_real_direct_package_is_reused_until_an_input_changes() {
        let root = std::env::var("CHOPINFER_QWEN38_CHECKPOINT")
            .expect("set CHOPINFER_QWEN38_CHECKPOINT to a Qwen3.8 checkpoint");
        let native = PathBuf::from(
            std::env::var("CHOPINFER_QWEN38_NATIVE_MODULE")
                .expect("set CHOPINFER_QWEN38_NATIVE_MODULE to a built libbundle_qwen38.so"),
        );
        let scratch = std::env::temp_dir().join(format!("qwen-reuse-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&scratch);
        std::fs::create_dir_all(scratch.join("other")).unwrap();
        let prepare = |module: &Path, ranks: u32| {
            let request = runtime_bundle::preparation::PrepareDirectRequest {
                path: root.clone().into(),
                cache_root: scratch.join("cache"),
                tensor_parallel_size: ranks,
                activation_policy: None,
                modalities: Default::default(),
            };
            // The lease ends here, so a damaged package can be moved aside.
            shared::direct::prepare_direct(&At(module.into()), &request)
                .unwrap()
                .package
        };
        let stages = |package: &runtime_bundle::preparation::DirectPackage| {
            package
                .measurements
                .stages
                .keys()
                .cloned()
                .collect::<Vec<_>>()
        };
        let cold = prepare(&native, 4);
        assert_eq!(cold.measurements.decisions, ["emitted direct package"]);
        assert_eq!(stages(&cold), ["package_assembly"]);
        let warm = prepare(&native, 4);
        assert_eq!(
            warm.measurements.decisions,
            ["reused completed direct package"]
        );
        assert_eq!(stages(&warm), ["package_reuse"]);
        assert_eq!(warm.package_identity, cold.package_identity);
        assert_eq!(warm.profile, cold.profile);
        // Another native module: its own package, the first one kept.
        let other = scratch.join("other/libbundle_qwen38.so");
        let mut bytes = std::fs::read(&native).unwrap();
        bytes.extend_from_slice(b"another module");
        std::fs::write(&other, bytes).unwrap();
        let changed = prepare(&other, 4);
        assert_eq!(changed.measurements.decisions, ["emitted direct package"]);
        assert_ne!(changed.package_identity, cold.package_identity);
        // Another rank count emits: the same package, which serves every
        // topology, with that count's profile. Then each count reuses it
        // with its own profile.
        let two = prepare(&native, 2);
        assert_eq!(stages(&two), ["package_assembly"]);
        assert_eq!(two.package_identity, cold.package_identity);
        assert_eq!(two.profile.profile["tensor_parallel"], 2);
        for (ranks, profile) in [(2, &two.profile), (4, &cold.profile)] {
            let again = prepare(&native, ranks);
            assert_eq!(stages(&again), ["package_reuse"]);
            assert_eq!(again.package_identity, cold.package_identity);
            assert_eq!(again.profile, *profile);
        }
        // A component changed after publication is not reused.
        let frontend = std::fs::read_dir(cold.bundle.join("components"))
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .find(|path| path.to_string_lossy().contains("tokenizer"))
            .expect("a tokenizer component");
        let mut bytes = std::fs::read(&frontend).unwrap();
        bytes.push(b' ');
        std::fs::write(&frontend, bytes).unwrap();
        let replaced = prepare(&native, 4);
        assert!(
            replaced.measurements.decisions[0].starts_with("replaced a damaged direct package"),
            "{:?}",
            replaced.measurements.decisions
        );
        assert_eq!(replaced.package_identity, cold.package_identity);
        assert_eq!(
            prepare(&native, 4).measurements.decisions,
            ["reused completed direct package"]
        );
        std::fs::remove_dir_all(&scratch).unwrap();
    }
}
