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
use model_compiler_sdk::preparation::direct::{DirectEmission, default_profile};
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
    if let Some(policy) = request.activation_policy.as_deref()
        && policy != "auto"
    {
        return Err(format!(
            "--activation-policy {policy}: deepseek41 runs the checkpoint's own activation \
             recipe (auto)"
        )
        .into());
    }
    let served = request.served_modalities(&provided_modalities(root, geometry)?)?;
    if !served.contains(&Modality::Text) {
        return Err("--modalities must include text, which every deepseek41 package serves".into());
    }
    Ok(served)
}

/// The serving profile of `geometry` at `ranks`: the autoregressive entry
/// for `ranks` of the profiles a package of `geometry` carries
/// (`package::profiles`), so the selected profile is the package's own.
pub fn select_profile(geometry: &Geometry, ranks: u32) -> Result<DirectProfile> {
    let document = crate::package::profiles(geometry, &[ranks])?;
    default_profile(&document, ranks, &["none"])
}

/// The most weight bytes one rank of `ranks` holds of the checkpoint at
/// `root` on its device: the arena of the plan the host executes, less the
/// mapped views the host cuts out of it.
pub fn weight_bytes(root: &Path, ranks: u32) -> Result<u64> {
    let mut most = 0;
    for rank in 0..ranks {
        let request = crate::checkpoint::request(root, ranks, rank)?;
        let plan = weight_plan::plan(&request)?.ok_or("the text image has no weight plan")?;
        most = most.max(weight_plan::device_bytes(&plan));
    }
    Ok(most)
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
    let mut served = Vec::new();
    for &ranks in SUPPORTED_RANKS {
        if admission::check_ranks(geometry, ranks).is_err() {
            continue;
        }
        let bytes = weight_bytes(root, ranks)?;
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
fn tokenizer_components(root: &Path) -> Result<Vec<model_compiler_sdk::ComponentDraftV1>> {
    crate::tokenizer::load(root)?;
    let read = |name: &str| -> Result<StagedContentV1> {
        let file = runtime_weights::checkpoint::CheckpointFile::open(root, name)
            .map_err(|e| format!("{name}: {e}"))?;
        Ok(StagedContentV1::bytes(
            file.read_bounded(64 << 20)
                .map_err(|e| format!("{name}: {e}"))?,
        ))
    };
    Ok(vec![
        package_draft::tokenizer_definition(read("tokenizer.json")?)?,
        package_draft::tokenizer_configuration(read("tokenizer_config.json")?)?,
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
    let ranks = request.tensor_parallel_size;
    let geometry = admit(root, ranks)?;
    let media = served_modalities(root, geometry, request)?.contains(&Modality::Image);
    let profile = select_profile(geometry, ranks)?;
    let mut components = tokenizer_components(root)?;
    if media {
        components.push(crate::encoder_image::planned(root, ranks)?);
    }
    let topologies = served_topologies(root, geometry, ranks, crate::package::media_bytes(media)?)?;
    let draft = crate::package::draft(
        geometry,
        geometry.check_runtime(&crate::checkpoint::document(root, "config.json")?)?,
        &topologies,
        components,
        StagedContentV1::file(module),
    )?;
    let package = model_compiler_sdk::emit_bundle_directory_v1(&draft, output)?;
    Ok(DirectEmission { package, profile })
}

pub struct DeepSeek41Family;

impl Provider for DeepSeek41Family {
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
        _checkpoint_identity: &str,
        output: &Path,
    ) -> Result<DirectEmission> {
        let module = model_compiler_sdk::preparation::direct::installed_native_module(
            NATIVE_MODULE,
            "DeepSeek V4.1 Flash",
        )?;
        emit(root, request, &module, output)
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
