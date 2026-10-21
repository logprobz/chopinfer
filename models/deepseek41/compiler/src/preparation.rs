//! The preparation provider. `probe` routes a checkpoint by its
//! `config.json` alone, before any header is read. The emission admits the
//! checkpoint, loads its tokenizer against the pinned contract, plans every
//! rank of every rank count from the checkpoint's headers (the weight plan
//! the host later executes) and writes the package (`crate::package`): the
//! tokenizer, the geometry's frontend, the planned rank images, the serving
//! profiles and the installed native module, which exports the chat format.
//! A package serving images also carries the planned encoder image and the
//! image frontend. A rank count whose weights do not fit is left out, and
//! refused when it is the one requested.
use model_compiler_sdk::files::Result;
use model_compiler_sdk::preparation::direct::{self, DirectEmission, default_profile};
use model_compiler_sdk::preparation::helper::Provider;
use model_compiler_sdk::{StagedContentV1, package_draft};
use model_deepseek41_contract::admission::{self, SUPPORTED_RANKS};
use model_deepseek41_contract::geometry::Geometry;
use model_deepseek41_contract::weight_plan;
use runtime_bundle::preparation::{DirectProfile, Modality, PrepareDirectRequest};
use std::collections::BTreeSet;
use std::path::Path;

use crate::package::{PROFILES, check_fit};

/// The installed native module beside the helper.
const NATIVE_MODULE: &str = "libbundle_deepseek41.so";
const PREPARATION_DOCUMENTS: &[&str] = &[
    "config.json",
    "model.safetensors.index.json",
    "tokenizer.json",
    "tokenizer_config.json",
];

/// The geometry the checkpoint at `root` admits at `ranks`, from its
/// `config.json` alone.
fn admit(root: &Path, ranks: u32) -> Result<&'static Geometry> {
    let config = crate::checkpoint::document(root, "config.json")?;
    let geometry = admission::admit(&config)?;
    admission::check_ranks(geometry, ranks)?;
    Ok(geometry)
}

/// The modalities the checkpoint at `root` of `geometry` provides: text,
/// and images when it stores the vision tower the encoder declaration pins
/// (`encoder_image::has_tower`).
pub fn provided_modalities(root: &Path, geometry: &Geometry) -> Result<BTreeSet<Modality>> {
    let mut provided = BTreeSet::from([Modality::Text]);
    if crate::encoder_image::has_tower(root, geometry)? {
        provided.insert(Modality::Image);
    }
    Ok(provided)
}

/// The modalities a package of the checkpoint at `root` of `geometry`
/// serves for `request`, refusing what it cannot serve: an activation
/// policy other than the checkpoint's own (`auto`), or a modality
/// selection without text or naming one the checkpoint does not provide.
/// `auto` and `all` serve text and, with the vision tower, images.
fn served_modalities(
    root: &Path,
    geometry: &Geometry,
    request: &PrepareDirectRequest,
) -> Result<BTreeSet<Modality>> {
    check_activation_policy(request)?;
    select_modalities(request, &provided_modalities(root, geometry)?)
}

fn check_activation_policy(request: &PrepareDirectRequest) -> Result<()> {
    if let Some(policy) = request.activation_policy.as_deref()
        && policy != "auto"
    {
        return Err(format!(
            "--activation-policy {policy}: deepseek41 runs the checkpoint's own activation \
             recipe (auto)"
        )
        .into());
    }
    Ok(())
}

fn select_modalities(
    request: &PrepareDirectRequest,
    provided: &BTreeSet<Modality>,
) -> Result<BTreeSet<Modality>> {
    check_activation_policy(request)?;
    let served = request.served_modalities(provided)?;
    if !served.contains(&Modality::Text) {
        return Err("--modalities must include text, which every deepseek41 package serves".into());
    }
    Ok(served)
}

/// The package's DSpark profile at `ranks`, falling back to its
/// autoregressive profile when it carries no DSpark entry.
pub fn select_profile(geometry: &Geometry, ranks: u32) -> Result<DirectProfile> {
    let document = crate::package::profiles(geometry, &[ranks])?;
    default_profile(&document, ranks, &["dspark", "none"])
}

/// The most weight bytes one rank of `ranks` holds of the checkpoint at
/// `root` on its device: the arena of the plan the host executes, less the
/// mapped views the host cuts out of it.
pub fn weight_bytes(root: &Path, ranks: u32) -> Result<u64> {
    let mut snapshot = crate::checkpoint::Snapshot::open(root)?;
    let bytes = snapshot_weight_bytes(&mut snapshot, ranks)?;
    snapshot.verify()?;
    Ok(bytes)
}

fn snapshot_weight_bytes(snapshot: &mut crate::checkpoint::Snapshot, ranks: u32) -> Result<u64> {
    Ok(weight_plan::Preparation::new(snapshot.request(ranks, 0))?.maximum_device_bytes(ranks)?)
}

/// The rank counts the package of the checkpoint at `root` of `geometry`
/// serves, each with the most text weight bytes one of its ranks holds:
/// every supported count the geometry divides over and the weights fit,
/// with `extra` bytes a rank holds beyond them (the encoder's, for a
/// package serving images), which must include `required`.
pub fn served_topologies(
    root: &Path,
    geometry: &'static Geometry,
    required: u32,
    extra: u64,
) -> Result<Vec<(u32, u64)>> {
    let mut snapshot = crate::checkpoint::Snapshot::open(root)?;
    let served = snapshot_topologies(&mut snapshot, geometry, required, extra)?;
    snapshot.verify()?;
    Ok(served)
}

fn snapshot_topologies(
    snapshot: &mut crate::checkpoint::Snapshot,
    geometry: &'static Geometry,
    required: u32,
    extra: u64,
) -> Result<Vec<(u32, u64)>> {
    let mut served = Vec::new();
    let planning = weight_plan::Preparation::new(snapshot.request(required, 0))?;
    for &ranks in SUPPORTED_RANKS {
        if admission::check_ranks(geometry, ranks).is_err() {
            continue;
        }
        let bytes = planning.maximum_device_bytes(ranks)?;
        match check_fit(ranks, bytes + extra) {
            Ok(()) => served.push((ranks, bytes)),
            Err(refusal) if ranks == required => return Err(refusal.into()),
            Err(_) => {}
        }
    }
    if !served.iter().any(|&(ranks, _)| ranks == required) {
        return Err(format!("the package serves no topology of {required} ranks").into());
    }
    Ok(served)
}

/// The checkpoint's tokenizer files, loaded against the pinned contract
/// first.
fn tokenizer_components(
    snapshot: &direct::CheckpointSnapshot,
) -> Result<Vec<model_compiler_sdk::ComponentDraftV1>> {
    let definition = snapshot.regular_document("tokenizer.json")?;
    let configuration = snapshot.regular_document("tokenizer_config.json")?;
    crate::tokenizer::load_bytes(definition, configuration)?;
    Ok(vec![
        package_draft::tokenizer_definition(StagedContentV1::bytes(definition.to_vec()))?,
        package_draft::tokenizer_configuration(StagedContentV1::bytes(configuration.to_vec()))?,
    ])
}

/// Emits into `output` the package of the checkpoint at `root` for
/// `request`, with the native module at `module`, and the serving profile
/// it carries.
pub fn emit(
    root: &Path,
    request: &PrepareDirectRequest,
    module: &Path,
    output: &Path,
) -> Result<DirectEmission> {
    emit_with_inputs(root, request, module, output, None)
}

/// Emits only the native module authenticated by `inputs`, when present.
/// The helper uses the same inputs for early package reuse and publication.
pub fn emit_with_inputs(
    root: &Path,
    request: &PrepareDirectRequest,
    module: &Path,
    output: &Path,
    inputs: Option<&[u8]>,
) -> Result<DirectEmission> {
    let captured = std::sync::Arc::new(direct::CheckpointSnapshot::capture_documents(
        root,
        PREPARATION_DOCUMENTS,
    )?);
    let mut snapshot = crate::checkpoint::Snapshot::from_checkpoint(captured.clone())?;
    emit_snapshot(&mut snapshot, &captured, request, module, output, inputs)
}

fn emit_snapshot(
    snapshot: &mut crate::checkpoint::Snapshot,
    captured: &direct::CheckpointSnapshot,
    request: &PrepareDirectRequest,
    module: &Path,
    output: &Path,
    inputs: Option<&[u8]>,
) -> Result<DirectEmission> {
    let ranks = request.tensor_parallel_size;
    let geometry = admission::admit(snapshot.config())?;
    admission::check_ranks(geometry, ranks)?;
    let mut provided = BTreeSet::from([Modality::Text]);
    if crate::encoder_image::has_tower_in(snapshot.metadata(), geometry)? {
        provided.insert(Modality::Image);
    }
    let media = select_modalities(request, &provided)?.contains(&Modality::Image);
    let profile = select_profile(geometry, ranks)?;
    let mut components = tokenizer_components(captured)?;
    if media {
        model_deepseek41_contract::encoder_plan::plan(snapshot.request(ranks, 0))?;
        components.push(crate::encoder_image::component()?);
    }
    let topologies = snapshot_topologies(
        snapshot,
        geometry,
        ranks,
        crate::package::media_bytes(media)?,
    )?;
    let draft = crate::package::draft(
        geometry,
        geometry.check_runtime(snapshot.config())?,
        &topologies,
        components,
        direct::native_module_staged(module, inputs)?,
    )?;
    snapshot.verify()?;
    let package = model_compiler_sdk::emit_bundle_directory_v1(&draft, output)?;
    snapshot.verify()?;
    Ok(DirectEmission { package, profile })
}

pub struct DeepSeek41Family;

impl Provider for DeepSeek41Family {
    fn direct_snapshot(&self, root: &Path) -> Result<Option<direct::SnapshotProvider>> {
        let captured = std::sync::Arc::new(direct::CheckpointSnapshot::capture_documents(
            root,
            PREPARATION_DOCUMENTS,
        )?);
        captured.regular_document("tokenizer.json")?;
        captured.regular_document("tokenizer_config.json")?;
        let snapshot = crate::checkpoint::Snapshot::from_checkpoint(captured.clone())?;
        let geometry = admission::admit(snapshot.config())?;
        let mut provided = BTreeSet::from([Modality::Text]);
        if crate::encoder_image::has_tower_in(snapshot.metadata(), geometry)? {
            provided.insert(Modality::Image);
        }
        let family = SnapshotFamily {
            captured: captured.clone(),
            snapshot: std::sync::Mutex::new(snapshot),
            geometry,
            provided,
        };
        Ok(Some(direct::SnapshotProvider::new(
            captured,
            Box::new(family),
        )))
    }
    fn name(&self) -> &str {
        "chopinfer-model-deepseek41"
    }
    fn profiles(&self) -> Vec<String> {
        let file: serde_json::Value =
            serde_json::from_str(PROFILES).expect("the serving profiles parse");
        file["profiles"]
            .as_object()
            .map(|profiles| profiles.keys().cloned().collect())
            .unwrap_or_default()
    }
    fn probe(&self, path: &Path, ranks: u32) -> Result<()> {
        admit(path, ranks).map(|_| ())
    }
    fn emit_direct(
        &self,
        root: &Path,
        request: &PrepareDirectRequest,
        checkpoint_identity: &str,
        output: &Path,
    ) -> Result<DirectEmission> {
        self.emit_direct_with_inputs(root, request, checkpoint_identity, output, None)
    }
    fn direct_inputs(
        &self,
        _root: &Path,
        _request: &PrepareDirectRequest,
    ) -> Result<Option<Vec<u8>>> {
        // Geometry, profiles and tokenizer policy are compiled into the helper.
        // Emission reads no additional environment-controlled inputs.
        direct::native_module_inputs(&native_module()?, &[]).map(Some)
    }
    fn emit_direct_with_inputs(
        &self,
        root: &Path,
        request: &PrepareDirectRequest,
        _checkpoint_identity: &str,
        output: &Path,
        inputs: Option<&[u8]>,
    ) -> Result<DirectEmission> {
        emit_with_inputs(root, request, &native_module()?, output, inputs)
    }
    fn names_direct_profile(&self) -> bool {
        true
    }
    /// The set the emission serves, so `auto`, `all` and the list they
    /// resolve to share a published package.
    fn direct_modalities(
        &self,
        root: &Path,
        request: &PrepareDirectRequest,
    ) -> Result<Option<BTreeSet<Modality>>> {
        let geometry = admit(root, request.tensor_parallel_size)?;
        served_modalities(root, geometry, request).map(Some)
    }
    fn direct_profile(&self, root: &Path, request: &PrepareDirectRequest) -> Result<DirectProfile> {
        let geometry = admit(root, request.tensor_parallel_size)?;
        served_modalities(root, geometry, request)?;
        select_profile(geometry, request.tensor_parallel_size)
    }
}

fn native_module() -> Result<std::path::PathBuf> {
    direct::installed_native_module(NATIVE_MODULE, "DeepSeek V4.1 Flash")
}

/// Metadata selection and emission share this one parsed checkpoint. Profile
/// selection reads only immutable fields, so it does not contend with emission.
struct SnapshotFamily {
    captured: std::sync::Arc<direct::CheckpointSnapshot>,
    snapshot: std::sync::Mutex<crate::checkpoint::Snapshot>,
    geometry: &'static Geometry,
    provided: BTreeSet<Modality>,
}

impl Provider for SnapshotFamily {
    fn name(&self) -> &str {
        DeepSeek41Family.name()
    }
    fn profiles(&self) -> Vec<String> {
        DeepSeek41Family.profiles()
    }
    fn probe(&self, _: &Path, ranks: u32) -> Result<()> {
        admission::check_ranks(self.geometry, ranks).map_err(Into::into)
    }
    fn names_direct_profile(&self) -> bool {
        true
    }
    fn direct_profile(&self, _: &Path, request: &PrepareDirectRequest) -> Result<DirectProfile> {
        admission::check_ranks(self.geometry, request.tensor_parallel_size)?;
        select_modalities(request, &self.provided)?;
        select_profile(self.geometry, request.tensor_parallel_size)
    }
    fn direct_modalities(
        &self,
        _: &Path,
        request: &PrepareDirectRequest,
    ) -> Result<Option<BTreeSet<Modality>>> {
        admission::check_ranks(self.geometry, request.tensor_parallel_size)?;
        select_modalities(request, &self.provided).map(Some)
    }
    fn direct_inputs(
        &self,
        root: &Path,
        request: &PrepareDirectRequest,
    ) -> Result<Option<Vec<u8>>> {
        DeepSeek41Family.direct_inputs(root, request)
    }
    fn emit_direct_with_inputs(
        &self,
        _: &Path,
        request: &PrepareDirectRequest,
        _: &str,
        output: &Path,
        inputs: Option<&[u8]>,
    ) -> Result<DirectEmission> {
        let mut snapshot = self
            .snapshot
            .lock()
            .map_err(|_| "checkpoint snapshot poisoned")?;
        emit_snapshot(
            &mut snapshot,
            &self.captured,
            request,
            &native_module()?,
            output,
            inputs,
        )
    }
}
