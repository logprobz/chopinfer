//! Packages without rank images for serving a checkpoint directory by loading
//! it directly. Nothing here reads or hashes a tensor payload.
use super::cache::{self, Artifact, Cache, Entry, digest};
use super::helper::Provider;
use crate::files::Result;
use crate::native_contract::NativeModule;
use crate::{FileIdentity, StagedContentV1};
use runtime_bundle::preparation::{
    CacheLease, DirectPackage, DirectProfile, Measurements, ModalitySelection,
    PrepareDirectRequest, Stage,
};
use runtime_weights::checkpoint::{CheckpointFile, header_identity};
use std::fs;
use std::os::unix::fs::FileExt;
use std::path::Path;
use std::time::Instant;

mod inputs;
use inputs::CheckpointInputs;
pub use inputs::CheckpointSnapshot;

/// A provider whose checkpoint decisions consume the same owned metadata
/// that supplies its cache identities. No wire or bundle ABI is changed.
pub struct SnapshotProvider {
    snapshot: std::sync::Arc<CheckpointSnapshot>,
    provider: Box<dyn Provider>,
}

impl SnapshotProvider {
    pub fn new(snapshot: std::sync::Arc<CheckpointSnapshot>, provider: Box<dyn Provider>) -> Self {
        Self { snapshot, provider }
    }
}

pub struct DirectPrepared {
    pub package: DirectPackage,
    pub lease: CacheLease,
}

/// What a family emits for a direct load: the package, and the serving
/// profile it selected for the checkpoint and topology.
pub struct DirectEmission {
    pub package: crate::EmittedBundleV1,
    pub profile: DirectProfile,
}

/// The profile of a model's `chopinfer-serve-profiles-v2` file that serves
/// `ranks` ranks with the first of `drafters` the family can plan for this
/// checkpoint (`"none"` last for autoregressive serving): named
/// `tp<ranks>-<drafter>` when the file has it, otherwise the only entry for
/// that rank count and drafter. The file's own entries are the profiles the
/// package path serves; this picks among them, it adds none.
pub fn default_profile(profiles: &str, ranks: u32, drafters: &[&str]) -> Result<DirectProfile> {
    let file: serde_json::Value = serde_json::from_str(profiles)?;
    if file["schema"] != "chopinfer-serve-profiles-v2" {
        return Err("unsupported serving profile schema".into());
    }
    let entries = file["profiles"]
        .as_object()
        .ok_or("serving profiles have no entries")?;
    for drafter in drafters {
        let matching: Vec<_> = entries
            .iter()
            .filter(|(_, profile)| {
                profile["tensor_parallel"] == ranks && profile["speculation"]["drafter"] == *drafter
            })
            .collect();
        let preferred = format!(
            "tp{ranks}-{}",
            if *drafter == "none" { "ar" } else { drafter }
        );
        let chosen =
            matching
                .iter()
                .find(|(name, _)| **name == preferred)
                .or(match matching.as_slice() {
                    [only] => Some(only),
                    _ => None,
                });
        if let Some((name, profile)) = chosen {
            return Ok(DirectProfile {
                name: (*name).clone(),
                profile: (*profile).clone(),
            });
        }
    }
    Err(format!(
        "no serving profile for {ranks} ranks with drafter {}",
        drafters.join(", ")
    )
    .into())
}

/// The safetensors files a direct load opens: those of `root` and of its
/// immediate subdirectories other than hidden ones, named `dir/file`.
fn safetensors_names(root: &Path) -> Result<Vec<(Option<String>, String)>> {
    let entries = |directory: &Path| -> Result<Vec<(String, bool)>> {
        fs::read_dir(directory)?
            .map(|entry| {
                let entry = entry?;
                let name = entry
                    .file_name()
                    .into_string()
                    .map_err(|_| "checkpoint name is not UTF-8")?;
                Ok((name, entry.file_type()?.is_dir()))
            })
            .collect()
    };
    let mut names = Vec::new();
    for (name, directory) in entries(root)? {
        if !directory {
            names.push((None, name));
        } else if !name.starts_with('.') {
            for (inner, nested) in entries(&root.join(&name))? {
                if !nested {
                    names.push((Some(name.clone()), inner));
                }
            }
        }
    }
    names.retain(|(_, name)| name.ends_with(".safetensors"));
    Ok(names)
}

/// The identity a direct load gives `root` (the host's `startup
/// weight_plan` line prints the same value): the SHA-256 of every
/// safetensors file's name, length and header. It authenticates no tensor
/// payload.
pub fn checkpoint_identity(root: &Path) -> Result<String> {
    let mut files = Vec::new();
    for (directory, name) in safetensors_names(root)? {
        let (parent, full) = match &directory {
            Some(directory) => (root.join(directory), format!("{directory}/{name}")),
            None => (root.to_path_buf(), name.clone()),
        };
        let file = CheckpointFile::open(&parent, &name)?;
        let length = file.metadata()?.len();
        let mut prefix = [0_u8; 8];
        file.read_exact_at(&mut prefix, 0)?;
        let header_length = u64::from_le_bytes(prefix);
        if header_length == 0 || header_length > 256 << 20 || 8 + header_length > length {
            return Err(format!("{full}: safetensors header length is outside the file").into());
        }
        let mut header = vec![0; header_length as usize];
        file.read_exact_at(&mut header, 8)?;
        files.push((full, length, header));
    }
    if files.is_empty() {
        return Err(format!("{} holds no safetensors files", root.display()).into());
    }
    let identity = header_identity(
        files
            .iter()
            .map(|(name, length, header)| (name.as_str(), *length, header.as_slice())),
    );
    Ok(identity.iter().map(|b| format!("{b:02x}")).collect())
}

/// Whether the published `key` is complete and intact: its record, and the
/// whole package (manifest, every component and the native module) against
/// the manifest `expected` names.
fn reusable(cache: &Cache, key: &str, bundle: &Path, expected: &[u8]) -> Result<()> {
    let entry = cache
        .entry(key)
        .map_err(|e| format!("no complete record: {e}"))?;
    cache.verify_files(key, &entry)?;
    let package = crate::verify_emitted_package(bundle)?;
    if package.manifest().to_canonical_json()? != expected {
        return Err("its manifest differs".into());
    }
    Ok(())
}

/// Moves a damaged or half-published package aside, as `q-<time>-<pid>-<key>`
/// (which pruning removes), unless a server still holds its lease.
fn quarantine(cache: &Cache, key: &str, bundle: &Path) -> Result<()> {
    let lease = runtime_bundle::preparation::cache_lock(&cache.root, key)?;
    lease.try_lock().map_err(|_| {
        format!("the damaged direct package {key} is in use; stop its server first")
    })?;
    let aside = cache.root.join(format!(
        "q-{}-{}-{key}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_nanos(),
        std::process::id()
    ));
    fs::rename(bundle, aside)?;
    fs::File::open(&cache.root)?.sync_all()?;
    Ok(())
}

/// The family's installed native module `file`: beside the running helper
/// or in the adjacent release `lib` directory. `family` names the model in
/// the refusal.
pub fn installed_native_module(file: &str, family: &str) -> Result<std::path::PathBuf> {
    let executable = fs::canonicalize(std::env::current_exe()?)?;
    let bin = executable
        .parent()
        .ok_or("helper has no install directory")?;
    for path in [bin.join(file), bin.join("../lib").join(file)] {
        if path.try_exists()? {
            if !fs::symlink_metadata(&path)?.is_file() {
                return Err("native module must be a regular installed file".into());
            }
            return Ok(fs::canonicalize(path)?);
        }
    }
    Err(format!(
        "install the {family} native module beside its helper or in the release lib directory"
    )
    .into())
}

/// The emission inputs of a native module at `path` ([`Provider::direct_inputs`]):
/// its file name, length and SHA-256, read once, and the value in this
/// process of each `environment` variable the emission reads (`null` when
/// unset), so a start under another value emits its own package.
pub fn native_module_inputs(path: &Path, environment: &[&str]) -> Result<Vec<u8>> {
    let bytes = fs::read(path)?;
    let environment: serde_json::Map<String, serde_json::Value> = environment
        .iter()
        .map(|name| {
            let value = std::env::var_os(name)
                .map(|value| {
                    value
                        .into_string()
                        .map_err(|_| format!("{name} must be UTF-8"))
                })
                .transpose()?;
            Ok(((*name).to_owned(), value.into()))
        })
        .collect::<Result<_>>()?;
    Ok(serde_json::to_vec(&serde_json::json!({
        "native_module": {
            "name": path.file_name().and_then(|n| n.to_str()).ok_or("invalid native module name")?,
            "bytes": bytes.len(),
            "sha256": runtime_weights::sha256_bytes(&bytes),
        },
        "environment": environment,
    }))?)
}

/// The native module that emission `inputs` ([`native_module_inputs`]) name.
fn inputs_native_module(inputs: &[u8]) -> Result<FileIdentity> {
    let inputs: serde_json::Value = serde_json::from_slice(inputs)?;
    let module = &inputs["native_module"];
    let (Some(name), Some(bytes), Some(sha256)) = (
        module["name"].as_str(),
        module["bytes"].as_u64(),
        module["sha256"].as_str(),
    ) else {
        return Err("the emission inputs name no native module".into());
    };
    Ok(FileIdentity::new(name, bytes, sha256)?)
}

/// The native module at `path` to stage in a direct package: with emission
/// `inputs` ([`native_module_inputs`]), only the module they identify, so a
/// module replaced after they were taken fails the emission; without them,
/// the file as it is.
pub fn native_module_staged(path: &Path, inputs: Option<&[u8]>) -> Result<StagedContentV1> {
    let Some(inputs) = inputs else {
        return Ok(StagedContentV1::file(path));
    };
    let identity = inputs_native_module(inputs)?;
    if path.file_name().and_then(|n| n.to_str()) != Some(identity.name()) {
        return Err("the emission inputs name another native module".into());
    }
    Ok(StagedContentV1::authenticated_file(path, identity))
}

/// The native module at `path`, inspected for a family that plans from its
/// contracts: with emission `inputs` ([`native_module_inputs`]), only the
/// module they identify, so a module replaced after they were taken fails
/// the emission. Its [`NativeModule::staged`] content is then that module.
pub fn native_module_inspected(path: &Path, inputs: Option<&[u8]>) -> Result<NativeModule> {
    let module = NativeModule::inspect(path)?;
    if let Some(inputs) = inputs {
        let name = path
            .file_name()
            .and_then(|n| n.to_str())
            .ok_or("invalid native module name")?;
        if FileIdentity::new(name, module.bytes, &module.sha256)? != inputs_native_module(inputs)? {
            return Err(
                "the native module differs from the one the emission inputs identify".into(),
            );
        }
    }
    Ok(module)
}

/// What decides a family's direct emission for a request: the checkpoint's
/// header and metadata identities, the inputs the family states
/// ([`Provider::direct_inputs`]) and, with them, their digest.
struct EmissionKey {
    identity: String,
    metadata_identity: String,
    inputs: Option<Vec<u8>>,
    digest: Option<String>,
    checkpoint_inputs: Option<CheckpointInputs>,
}

impl EmissionKey {
    /// Taken once per request: emission consumes these same inputs. The
    /// Owned snapshots bind emission to the bytes hashed here. Retained
    /// file and directory identities also detect ordinary concurrent edits.
    fn of<F: Provider + ?Sized>(
        family: &F,
        root: &Path,
        request: &PrepareDirectRequest,
        snapshot: Option<&CheckpointSnapshot>,
    ) -> Result<Self> {
        let checkpoint_inputs = snapshot
            .is_none()
            .then(|| CheckpointInputs::capture(root))
            .transpose()?;
        let (identity, metadata_identity) = match snapshot {
            Some(snapshot) => {
                snapshot.verify()?;
                (
                    snapshot.identity.clone(),
                    snapshot.metadata_identity.clone(),
                )
            }
            None => (
                checkpoint_identity(root)?,
                runtime_weights::checkpoint::metadata_identity(root)?
                    .iter()
                    .map(|b| format!("{b:02x}"))
                    .collect(),
            ),
        };
        let mut canonical = request.clone();
        if let Some(served) = family.direct_modalities(root, request)? {
            canonical.modalities = ModalitySelection::Only(served);
        }
        let request = &canonical;
        let inputs = family.direct_inputs(root, request)?;
        let digest = match &inputs {
            Some(inputs) => {
                // The executable this process runs, even if its path was
                // replaced.
                let helper = runtime_weights::sha256_bytes(&fs::read("/proc/self/exe")?);
                Some(emission_digest(
                    family.name(),
                    &helper,
                    inputs,
                    request,
                    &identity,
                    &metadata_identity,
                )?)
            }
            None => None,
        };
        if let Some(checkpoint_inputs) = &checkpoint_inputs {
            checkpoint_inputs.verify()?;
        }
        Ok(Self {
            identity,
            metadata_identity,
            inputs,
            digest,
            checkpoint_inputs,
        })
    }
}

/// The digest of what decides a family's direct emission, from the inputs
/// the family states ([`Provider::direct_inputs`]): the family and its
/// helper executable's SHA-256 `helper` (which binds the code and the data
/// compiled into it), those inputs, the request's rank count, activation
/// policy and served modalities, and the checkpoint's header and metadata
/// identities.
fn emission_digest(
    family: &str,
    helper: &str,
    inputs: &[u8],
    request: &PrepareDirectRequest,
    identity: &str,
    metadata_identity: &str,
) -> Result<String> {
    digest(&(
        "chopinfer-direct-emission-inputs-v1",
        family,
        helper,
        runtime_weights::sha256_bytes(inputs),
        request.tensor_parallel_size,
        &request.activation_policy,
        // The served list, or None for a keyword the family leaves unresolved.
        request.modalities.listed(),
        identity,
        metadata_identity,
    ))
}

/// The complete published packages whose records name these emission
/// `inputs`, with the serving profile each record states for them. A
/// package serves every rank count, so its record names the inputs of each
/// request that emitted it, with the profile each selected.
fn published_for(cache: &Cache, inputs: &str) -> Result<Vec<(String, DirectProfile)>> {
    Ok(cache
        .entries()?
        .into_iter()
        .filter(|(_, entry)| entry.kind == "package" && entry.state == "complete")
        .filter_map(|(key, entry)| {
            serde_json::from_value(entry.payload["emissions"].get(inputs)?.clone())
                .ok()
                .map(|profile| (key, profile))
        })
        .collect())
}

/// The serving profile [`prepare_direct`] returns for `request` at `root`
/// when it reuses a published package, read from the record of a package
/// published for the same emission inputs, without emitting or verifying
/// anything. `None` when the family states no inputs, `request` names no
/// existing preparation cache (this creates none) or no record matches.
/// A record names only a profile the family named itself when the package
/// was emitted. Preparation still verifies the package; a damaged one is
/// emitted again from the same inputs, with the same profile.
pub fn published_profile<F: Provider + ?Sized>(
    family: &F,
    root: &Path,
    request: &PrepareDirectRequest,
) -> Result<Option<DirectProfile>> {
    if !request.cache_root.is_absolute() || !request.cache_root.join("cache-version").is_file() {
        return Ok(None);
    }
    let captured = family.direct_snapshot(root)?;
    match captured {
        Some(captured) => published_profile_captured(
            captured.provider.as_ref(),
            root,
            request,
            Some(&captured.snapshot),
        ),
        None => published_profile_captured(family, root, request, None),
    }
}

fn published_profile_captured<F: Provider + ?Sized>(
    family: &F,
    root: &Path,
    request: &PrepareDirectRequest,
    snapshot: Option<&CheckpointSnapshot>,
) -> Result<Option<DirectProfile>> {
    let key = EmissionKey::of(family, root, request, snapshot)?;
    let Some(inputs) = key.digest.as_deref() else {
        return Ok(None);
    };
    let cache = Cache::open(&request.cache_root)?;
    let _gate = cache.gate(false)?;
    let profile = published_for(&cache, inputs)?
        .into_iter()
        .next()
        .map(|(_, profile)| profile);
    if let Some(checkpoint_inputs) = &key.checkpoint_inputs {
        checkpoint_inputs.verify()?;
    }
    if let Some(snapshot) = snapshot {
        snapshot.verify_contents(root)?;
    }
    Ok(profile)
}

/// A published package whose record names these emission `inputs`, when it
/// is complete and intact: its record's files, and the whole package
/// against its manifest, whose SHA-256 is the package's key. The package
/// lock is held while it is checked.
fn reusable_by_inputs(cache: &Cache, inputs: &str) -> Result<Option<Published>> {
    for (key, profile) in published_for(cache, inputs)? {
        let bundle = cache.path(&key)?;
        let _package = cache.lock(&format!("i-{}", digest(&("package", &key))?))?;
        let started = Instant::now();
        let intact = (|| -> Result<String> {
            cache.verify_files(&key, &cache.entry(&key)?)?;
            let package = crate::verify_emitted_package(&bundle)?;
            let manifest = package.manifest().to_canonical_json()?;
            if format!("p-{}", runtime_weights::sha256_bytes(&manifest)) != key {
                return Err("its manifest is not the one its key names".into());
            }
            Ok(package.manifest().id().as_str().to_owned())
        })();
        if let Ok(bundle_id) = intact {
            return Ok(Some(Published {
                key,
                profile,
                bundle_id,
                decision: "reused completed direct package".into(),
                stage: (
                    "package_reuse",
                    Stage {
                        seconds: started.elapsed().as_secs_f64(),
                        invocations: 1,
                        status: "verified".into(),
                        ..Default::default()
                    },
                ),
            }));
        }
    }
    Ok(None)
}

/// Emits the family's package without rank images into private staging,
/// then publishes it as `p-<manifest sha256>`, or reuses an identical
/// published package. A family that states its emission inputs has a
/// package published for the same inputs reused without emitting it again.
/// The result stays leased until the caller drops it.
pub fn prepare_direct<F: Provider + ?Sized>(
    family: &F,
    request: &PrepareDirectRequest,
) -> Result<DirectPrepared> {
    let started = Instant::now();
    let root = fs::canonicalize(&request.path)?;
    if !fs::symlink_metadata(&root)?.is_dir() {
        return Err("--model must name a local checkpoint directory".into());
    }
    family.probe(&root, request.tensor_parallel_size)?;
    // A hidden cache inside the checkpoint may create its root. Do that
    // before capturing directory identities, so our own creation is not an edit.
    let cache = Cache::open(&request.cache_root)?;
    let captured = family.direct_snapshot(&root)?;
    match captured {
        Some(captured) => prepare_captured(
            captured.provider.as_ref(),
            &root,
            request,
            &cache,
            started,
            Some(&captured.snapshot),
        ),
        None => prepare_captured(family, &root, request, &cache, started, None),
    }
}

fn prepare_captured<F: Provider + ?Sized>(
    family: &F,
    root: &Path,
    request: &PrepareDirectRequest,
    cache: &Cache,
    started: Instant,
    snapshot: Option<&CheckpointSnapshot>,
) -> Result<DirectPrepared> {
    let EmissionKey {
        identity,
        metadata_identity,
        inputs,
        digest: digest_of_inputs,
        checkpoint_inputs,
    } = EmissionKey::of(family, root, request, snapshot)?;
    let _gate = cache.gate(false)?;
    let reused = match &digest_of_inputs {
        Some(digest) => reusable_by_inputs(cache, digest)?,
        None => None,
    };
    let Published {
        key,
        profile,
        bundle_id,
        decision,
        stage,
    } = match reused {
        Some(published) => published,
        None => emit_and_publish(
            family,
            cache,
            root,
            request,
            &identity,
            inputs.as_deref(),
            digest_of_inputs.as_deref(),
            checkpoint_inputs.as_ref(),
            snapshot,
        )?,
    };
    let bundle = cache.path(&key)?;
    let package_identity = key["p-".len()..].to_owned();
    if snapshot.is_none() && checkpoint_identity(root)? != identity {
        return Err(
            "the checkpoint's safetensors files changed while its package was prepared".into(),
        );
    }
    if let Some(checkpoint_inputs) = &checkpoint_inputs {
        checkpoint_inputs.verify()?;
    }
    if let Some(snapshot) = snapshot {
        snapshot.verify_contents(root)?;
    }
    let lease = cache.select(std::slice::from_ref(&key))?;
    let mut measurements = Measurements {
        total_seconds: started.elapsed().as_secs_f64(),
        decisions: vec![decision],
        ..Default::default()
    };
    measurements.stages.insert(stage.0.into(), stage.1);
    Ok(DirectPrepared {
        lease,
        package: DirectPackage {
            bundle,
            cache_root: cache.root.clone(),
            package_identity,
            checkpoint_identity: identity,
            metadata_identity,
            bundle_id,
            tensor_parallel_size: request.tensor_parallel_size,
            profile,
            qualification: "compatible-not-evaluated".into(),
            measurements,
        },
    })
}

/// A published package selected for a request, and how.
struct Published {
    key: String,
    profile: DirectProfile,
    bundle_id: String,
    decision: String,
    stage: (&'static str, Stage),
}

/// Emits the package from the emission `inputs` into private staging and
/// publishes it, or reuses an identical published package, replacing a
/// damaged one. The record states the inputs' digest and the profile, so a
/// later request with the same digest reuses the package without emitting
/// it.
#[allow(clippy::too_many_arguments)]
fn emit_and_publish<F: Provider + ?Sized>(
    family: &F,
    cache: &Cache,
    root: &Path,
    request: &PrepareDirectRequest,
    identity: &str,
    inputs: Option<&[u8]>,
    digest_of_inputs: Option<&str>,
    checkpoint_inputs: Option<&CheckpointInputs>,
    snapshot: Option<&CheckpointSnapshot>,
) -> Result<Published> {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)?
        .as_nanos();
    let staging_key = format!(
        "i-{}",
        digest(&("direct-staging", std::process::id(), nanos.to_string()))?
    );
    let _staging_lock = cache.lock(&staging_key)?;
    let staging = cache.private_weights(&staging_key)?;
    let output = staging.join("package");
    let emission_started = Instant::now();
    // A family that names its serving profile before packaging must package
    // that profile: the host judges a start's options by the named one, and
    // a later start is answered from the profile recorded here, so a family
    // whose two selections differ is refused on every start alike. The
    // named profile is selected beside the emission, which it does not
    // read, so the check adds no time to it.
    let emission = std::thread::scope(|scope| {
        let named = family
            .names_direct_profile()
            .then(|| scope.spawn(|| family.direct_profile(root, request)));
        let emission = family.emit_direct_with_inputs(root, request, identity, &output, inputs);
        let named = named
            .map(|named| {
                named
                    .join()
                    .map_err(|_| "the serving profile selection panicked")?
            })
            .transpose();
        // The emission's own failure first.
        let emission = emission?;
        // Check before anything is published under the earlier emission key.
        if let Some(checkpoint_inputs) = &checkpoint_inputs {
            checkpoint_inputs.verify()?;
        }
        if let Some(snapshot) = snapshot {
            snapshot.verify_contents(root)?;
        }
        let named = named?;
        if let Some(named) = named
            && named != emission.profile
        {
            return Err(format!(
                "{} named serving profile {} before packaging but packaged a different {} \
                 profile",
                family.name(),
                named.name,
                emission.profile.name
            )
            .into());
        }
        Ok(emission)
    });
    let DirectEmission {
        package: emitted,
        profile,
    } = match emission {
        Ok(emitted) => emitted,
        Err(error) => {
            let _ = cache::remove_tree(&staging);
            return Err(error);
        }
    };
    let manifest = emitted.package.manifest().to_canonical_json()?;
    let bundle_id = emitted.package.manifest().id().as_str().to_owned();
    let emission_seconds = emission_started.elapsed().as_secs_f64();
    drop(emitted);
    let package_identity = runtime_weights::sha256_bytes(&manifest);
    let key = format!("p-{package_identity}");
    let bundle = cache.path(&key)?;
    let emissions = |mut known: serde_json::Map<String, serde_json::Value>| {
        if let Some(inputs) = digest_of_inputs {
            known.insert(inputs.to_owned(), serde_json::json!(&profile));
        }
        serde_json::json!({"direct": true, "checkpoint": identity, "emissions": known})
    };
    let payload = emissions(serde_json::Map::new());
    let decision: String;
    {
        let _package = cache.lock(&format!("i-{}", digest(&("package", &key))?))?;
        // Selection rewrites the same record.
        let _record = cache.record_lock(&key)?;
        let publish = || -> Result<()> {
            // The directory first, then its record: a publication cut short
            // leaves a directory without a record, which the next attempt
            // replaces.
            fs::rename(&output, &bundle)?;
            cache.save(
                &key,
                &Entry {
                    kind: "package".into(),
                    state: "complete".into(),
                    dependencies: vec![],
                    files: vec![Artifact {
                        name: "manifest.json".into(),
                        bytes: manifest.len() as u64,
                        sha256: package_identity.clone(),
                        image: false,
                    }],
                    last_selected: 0,
                    payload: payload.clone(),
                },
            )?;
            fs::File::open(&cache.root)?.sync_all()?;
            Ok(())
        };
        if bundle.try_exists()? {
            match reusable(cache, &key, &bundle, &manifest) {
                Ok(()) => {
                    // The record learns these emission inputs beside those
                    // it names, so the next request with them skips
                    // emission.
                    let mut entry = cache.entry(&key)?;
                    let known = entry.payload["emissions"]
                        .as_object()
                        .cloned()
                        .unwrap_or_default();
                    let learned = emissions(known);
                    if entry.payload != learned {
                        entry.payload = learned;
                        cache.save(&key, &entry)?;
                    }
                    decision = "reused completed direct package".into();
                }
                Err(reason) => {
                    quarantine(cache, &key, &bundle)?;
                    publish()?;
                    decision = format!("replaced a damaged direct package ({reason})");
                }
            }
        } else {
            publish()?;
            decision = "emitted direct package".into();
        }
    }
    cache::remove_tree(&staging)?;
    Ok(Published {
        key,
        profile,
        bundle_id,
        decision,
        stage: (
            "package_assembly",
            Stage {
                seconds: emission_seconds,
                invocations: 1,
                status: "emitted".into(),
                ..Default::default()
            },
        ),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    /// A family that states emission inputs but inherits the default
    /// input-aware emission; its own `emit_direct` would publish.
    struct InputsWithoutEmission;

    impl Provider for InputsWithoutEmission {
        fn name(&self) -> &str {
            "inputs-without-emission"
        }
        fn profiles(&self) -> Vec<String> {
            Vec::new()
        }
        fn probe(&self, _path: &Path, _ranks: u32) -> Result<()> {
            Ok(())
        }
        fn direct_inputs(
            &self,
            _root: &Path,
            _request: &PrepareDirectRequest,
        ) -> Result<Option<Vec<u8>>> {
            Ok(Some(b"module snapshot".to_vec()))
        }
        fn emit_direct(
            &self,
            _root: &Path,
            _request: &PrepareDirectRequest,
            _checkpoint_identity: &str,
            _output: &Path,
        ) -> Result<DirectEmission> {
            panic!("emission without the stated inputs");
        }
    }

    /// A fresh directory with a one-tensor checkpoint in `checkpoint/`.
    fn checkpoint_root(name: &str) -> std::path::PathBuf {
        let root = std::env::temp_dir().join(format!(
            "{name}-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = fs::remove_dir_all(&root);
        let checkpoint = root.join("checkpoint");
        fs::create_dir_all(&checkpoint).unwrap();
        let header = br#"{"t":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}"#;
        let mut shard = (header.len() as u64).to_le_bytes().to_vec();
        shard.extend_from_slice(header);
        shard.push(7);
        fs::write(checkpoint.join("model.safetensors"), shard).unwrap();
        root
    }

    /// The default input-aware emission fails closed: a family that states
    /// inputs and does not emit from them publishes nothing.
    #[test]
    fn stated_inputs_without_input_aware_emission_publish_nothing() {
        let root = checkpoint_root("direct-inputs-default");
        let cache = root.join("cache");
        let request = PrepareDirectRequest {
            path: root.join("checkpoint"),
            cache_root: cache.clone(),
            tensor_parallel_size: 1,
            activation_policy: None,
            modalities: ModalitySelection::Auto,
        };
        let error = prepare_direct(&InputsWithoutEmission, &request)
            .err()
            .expect("stated inputs without input-aware emission must fail")
            .to_string();
        assert!(error.contains("does not emit from them"), "{error}");
        let published: Vec<_> = fs::read_dir(&cache)
            .unwrap()
            .filter_map(|entry| entry.ok())
            .filter(|entry| entry.file_name().to_string_lossy().starts_with("p-"))
            .collect();
        assert!(published.is_empty());
        fs::remove_dir_all(&root).unwrap();
    }

    /// A family whose one package serves every rank count, selecting a
    /// profile named for the requested one.
    #[derive(Default)]
    struct EveryRankCount {
        emissions: std::sync::atomic::AtomicUsize,
    }

    impl Provider for EveryRankCount {
        fn name(&self) -> &str {
            "every-rank-count"
        }
        fn profiles(&self) -> Vec<String> {
            Vec::new()
        }
        fn probe(&self, _path: &Path, _ranks: u32) -> Result<()> {
            Ok(())
        }
        fn direct_modalities(
            &self,
            _root: &Path,
            request: &PrepareDirectRequest,
        ) -> Result<Option<std::collections::BTreeSet<runtime_bundle::preparation::Modality>>>
        {
            use runtime_bundle::preparation::Modality;
            Ok(Some(request.served_modalities(
                &[Modality::Text, Modality::Image].into(),
            )?))
        }
        fn direct_inputs(
            &self,
            _root: &Path,
            _request: &PrepareDirectRequest,
        ) -> Result<Option<Vec<u8>>> {
            Ok(Some(b"module snapshot".to_vec()))
        }
        fn emit_direct_with_inputs(
            &self,
            _root: &Path,
            request: &PrepareDirectRequest,
            _checkpoint_identity: &str,
            output: &Path,
            _inputs: Option<&[u8]>,
        ) -> Result<DirectEmission> {
            self.emissions
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            use runtime_bundle::*;
            let id = |value: &str| BundleOpaqueId::new(value).unwrap();
            let component = crate::ComponentDraftV1::new(
                id("weights"),
                id("fixture.weights.v1"),
                id("fixture.bytes.v1"),
                BundleRelativePathV1::new("components/weights.bin").unwrap(),
                crate::ComponentContentV1::Whole(crate::StagedContentV1::bytes(
                    serde_json::to_vec(&self.direct_modalities(_root, request)?)?,
                )),
            );
            let draft = crate::BundleDraftV1::new(
                id("fixture.every-rank-count.v1"),
                BundleTargetV1::new(
                    vec![id("sm_120f")],
                    vec![
                        BundleTopologyV1::new(id("tp2"), 2).unwrap(),
                        BundleTopologyV1::new(id("tp4"), 4).unwrap(),
                    ],
                )
                .unwrap(),
                BundleResourceRequirementsV1::new(1, Vec::new()).unwrap(),
                BundleKvLayoutV1::new(
                    id("fixture.kv.v1"),
                    64,
                    64,
                    1,
                    64,
                    1,
                    1_024,
                    BundleRankPlacementV1::RankLocal,
                    BundleStateLifetimeV1::Request,
                )
                .unwrap(),
                BundleStateLayoutV1::new(
                    id("fixture.state.v1"),
                    64,
                    64,
                    BundleStateLifetimeV1::Request,
                )
                .unwrap(),
            )
            .with_components(vec![component])
            .with_graph_buckets(vec![
                BundleGraphBucketV1::new(id("decode"), vec![id("weights")], Vec::new()).unwrap(),
            ]);
            Ok(DirectEmission {
                package: crate::emit_bundle_directory_v1(&draft, output)?,
                profile: DirectProfile {
                    name: format!("tp{}", request.tensor_parallel_size),
                    profile: serde_json::json!({}),
                },
            })
        }
    }

    struct ChangingCheckpoint {
        mode: &'static str,
        change: std::sync::atomic::AtomicBool,
        emitter: EveryRankCount,
    }

    impl Provider for ChangingCheckpoint {
        fn name(&self) -> &str {
            "changing-checkpoint"
        }
        fn profiles(&self) -> Vec<String> {
            Vec::new()
        }
        fn probe(&self, _: &Path, _: u32) -> Result<()> {
            Ok(())
        }
        fn direct_inputs(&self, _: &Path, _: &PrepareDirectRequest) -> Result<Option<Vec<u8>>> {
            Ok(Some(b"unchanged module".to_vec()))
        }
        fn emit_direct_with_inputs(
            &self,
            root: &Path,
            request: &PrepareDirectRequest,
            identity: &str,
            output: &Path,
            inputs: Option<&[u8]>,
        ) -> Result<DirectEmission> {
            let change = self
                .change
                .swap(false, std::sync::atomic::Ordering::Relaxed);
            if !change {
                return self
                    .emitter
                    .emit_direct_with_inputs(root, request, identity, output, inputs);
            }
            let config = root.join("config.json");
            let saved = root.parent().unwrap().join("original-config");
            match self.mode {
                "in-place" => fs::write(&config, br#"{"option":"B"}"#)?,
                "restore-inode" => {
                    fs::rename(&config, &saved)?;
                    fs::write(&config, br#"{"option":"B"}"#)?;
                }
                "optional" => fs::write(root.join("optional.json"), br#"{"option":"B"}"#)?,
                _ => unreachable!(),
            }
            // Simulate changed metadata selecting a different package, then
            // restoring A before the SDK's final header/digest checks.
            let mut changed = request.clone();
            changed.modalities =
                ModalitySelection::Only([runtime_bundle::preparation::Modality::Text].into());
            let result = self
                .emitter
                .emit_direct_with_inputs(root, &changed, identity, output, inputs);
            match self.mode {
                "in-place" => fs::write(&config, br#"{"option":"A"}"#)?,
                "restore-inode" => {
                    fs::remove_file(&config)?;
                    fs::rename(saved, &config)?;
                }
                "optional" => fs::remove_file(root.join("optional.json"))?,
                _ => unreachable!(),
            }
            result
        }
    }

    #[test]
    fn metadata_changed_and_restored_during_emission_cannot_poison_early_reuse() {
        for mode in ["in-place", "restore-inode", "optional"] {
            let root = checkpoint_root(&format!("direct-metadata-race-{mode}"));
            fs::write(root.join("checkpoint/config.json"), br#"{"option":"A"}"#).unwrap();
            let request = PrepareDirectRequest {
                path: root.join("checkpoint"),
                cache_root: root.join("cache"),
                tensor_parallel_size: 4,
                activation_policy: None,
                modalities: ModalitySelection::Auto,
            };
            let family = ChangingCheckpoint {
                mode,
                change: std::sync::atomic::AtomicBool::new(true),
                emitter: EveryRankCount::default(),
            };
            let error = prepare_direct(&family, &request)
                .err()
                .expect("changed inputs must be refused")
                .to_string();
            assert!(error.contains("checkpoint input"), "{mode}: {error}");
            let cache = Cache::open(&request.cache_root).unwrap();
            assert!(
                cache.entries().unwrap().is_empty(),
                "a refused emission publishes no record"
            );
            let good = prepare_direct(&family, &request).unwrap().package;
            assert!(good.measurements.stages.contains_key("package_assembly"));
            let reused = prepare_direct(&family, &request).unwrap().package;
            assert_eq!(good.package_identity, reused.package_identity);
            assert!(reused.measurements.stages.contains_key("package_reuse"));
            assert_eq!(
                family
                    .emitter
                    .emissions
                    .load(std::sync::atomic::Ordering::Relaxed),
                2
            );
            fs::remove_dir_all(root).unwrap();
        }
    }

    /// Requests at several rank counts share one package, and each reuses
    /// it without emitting again once its inputs are recorded, with the
    /// profile it selected.
    #[test]
    fn one_package_serves_the_requests_of_every_rank_count() {
        let root = checkpoint_root("direct-every-rank-count");
        let family = EveryRankCount::default();
        let prepare = |ranks: u32| {
            let request = PrepareDirectRequest {
                path: root.join("checkpoint"),
                cache_root: root.join("cache"),
                tensor_parallel_size: ranks,
                activation_policy: None,
                modalities: ModalitySelection::Auto,
            };
            prepare_direct(&family, &request).unwrap().package
        };
        let decisions = |package: &DirectPackage| package.measurements.decisions.clone();
        let tp2 = prepare(2);
        assert_eq!(decisions(&tp2), ["emitted direct package"]);
        let tp4 = prepare(4);
        assert_eq!(tp4.package_identity, tp2.package_identity);
        assert_eq!(decisions(&tp4), ["reused completed direct package"]);
        assert!(tp4.measurements.stages.contains_key("package_assembly"));
        for (ranks, name) in [(2, "tp2"), (4, "tp4"), (2, "tp2")] {
            // The profile request of each rank count names the profile that
            // count's request recorded, not another count's.
            let request = PrepareDirectRequest {
                path: root.join("checkpoint"),
                cache_root: fs::canonicalize(root.join("cache")).unwrap(),
                tensor_parallel_size: ranks,
                activation_policy: None,
                modalities: ModalitySelection::Auto,
            };
            let recorded = published_profile(&family, &request.path, &request)
                .unwrap()
                .expect("a recorded profile");
            assert_eq!(recorded.name, name);
            let again = prepare(ranks);
            assert_eq!(again.package_identity, tp2.package_identity);
            assert_eq!(again.profile.name, name);
            assert_eq!(
                again.measurements.stages.keys().collect::<Vec<_>>(),
                ["package_reuse"]
            );
        }
        // A rank count no request recorded has no profile to name.
        let request = PrepareDirectRequest {
            path: root.join("checkpoint"),
            cache_root: fs::canonicalize(root.join("cache")).unwrap(),
            tensor_parallel_size: 8,
            activation_policy: None,
            modalities: ModalitySelection::Auto,
        };
        assert!(
            published_profile(&family, &request.path, &request)
                .unwrap()
                .is_none()
        );
        assert_eq!(
            family.emissions.load(std::sync::atomic::Ordering::Relaxed),
            2,
            "repeat requests must not enter the emitter or topology planning"
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn auto_all_and_the_explicit_list_they_serve_reuse_without_assembly() {
        use runtime_bundle::preparation::Modality;
        let root = checkpoint_root("direct-modalities");
        let family = EveryRankCount::default();
        let full = || ModalitySelection::Only([Modality::Text, Modality::Image].into());
        for (index, (first, second)) in [
            (ModalitySelection::Auto, full()),
            (full(), ModalitySelection::All),
            (ModalitySelection::All, ModalitySelection::Auto),
        ]
        .into_iter()
        .enumerate()
        {
            let mut request = PrepareDirectRequest {
                path: root.join("checkpoint"),
                cache_root: root.join(format!("cache-{index}")),
                tensor_parallel_size: 2,
                activation_policy: None,
                modalities: first,
            };
            let initial = prepare_direct(&family, &request).unwrap().package;
            request.modalities = second;
            assert!(
                published_profile(&family, &request.path, &request)
                    .unwrap()
                    .is_some()
            );
            let reused = prepare_direct(&family, &request).unwrap().package;
            assert_eq!(initial.package_identity, reused.package_identity);
            assert_eq!(
                reused.measurements.stages.keys().collect::<Vec<_>>(),
                ["package_reuse"]
            );
            request.modalities = ModalitySelection::Only([Modality::Text].into());
            assert!(
                published_profile(&family, &request.path, &request)
                    .unwrap()
                    .is_none()
            );
            let reduced = prepare_direct(&family, &request).unwrap().package;
            assert_ne!(initial.package_identity, reduced.package_identity);
            assert!(reduced.measurements.stages.contains_key("package_assembly"));
        }
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn early_reuse_rebuilds_damaged_or_incomplete_packages() {
        for damage in [
            "truncated-record",
            "missing-record",
            "incomplete-record",
            "component",
        ] {
            let root = checkpoint_root(&format!("direct-recovery-{damage}"));
            let family = EveryRankCount::default();
            let request = PrepareDirectRequest {
                path: root.join("checkpoint"),
                cache_root: root.join("cache"),
                tensor_parallel_size: 4,
                activation_policy: None,
                modalities: ModalitySelection::Auto,
            };
            let first = prepare_direct(&family, &request).unwrap().package;
            let cache = Cache::open(&request.cache_root).unwrap();
            let key = format!("p-{}", first.package_identity);
            let record = cache.root.join("records").join(format!("{key}.json"));
            match damage {
                "truncated-record" => fs::write(record, b"{").unwrap(),
                "missing-record" => fs::remove_file(record).unwrap(),
                "incomplete-record" => {
                    let mut entry = cache.entry(&key).unwrap();
                    entry.state = "building".into();
                    cache.save(&key, &entry).unwrap();
                }
                "component" => {
                    fs::write(
                        cache.path(&key).unwrap().join("components/weights.bin"),
                        b"bad",
                    )
                    .unwrap();
                }
                _ => unreachable!(),
            }
            let repaired = prepare_direct(&family, &request).unwrap().package;
            assert_eq!(repaired.package_identity, first.package_identity);
            assert!(
                repaired
                    .measurements
                    .stages
                    .contains_key("package_assembly")
            );
            let reused = prepare_direct(&family, &request).unwrap().package;
            assert!(reused.measurements.stages.contains_key("package_reuse"));
            assert_eq!(
                family.emissions.load(std::sync::atomic::Ordering::Relaxed),
                2
            );
            fs::remove_dir_all(root).unwrap();
        }
    }

    #[test]
    fn a_new_hidden_cache_inside_the_checkpoint_does_not_invalidate_inputs() {
        let root = checkpoint_root("direct-hidden-cache");
        let family = EveryRankCount::default();
        let request = PrepareDirectRequest {
            path: root.join("checkpoint"),
            cache_root: root.join("checkpoint/.prepared-cache"),
            tensor_parallel_size: 4,
            activation_policy: None,
            modalities: ModalitySelection::Auto,
        };
        let first = prepare_direct(&family, &request).unwrap().package;
        let again = prepare_direct(&family, &request).unwrap().package;
        assert_eq!(first.package_identity, again.package_identity);
        assert!(again.measurements.stages.contains_key("package_reuse"));
        assert_eq!(
            family.emissions.load(std::sync::atomic::Ordering::Relaxed),
            1
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn owned_snapshot_preserves_identities_and_bounds_reads_before_allocation() {
        let root = checkpoint_root("direct-owned-snapshot");
        let checkpoint = root.join("checkpoint");
        fs::write(checkpoint.join("config.json"), br#"{"option":"A"}"#).unwrap();
        let snapshot = CheckpointSnapshot::capture(&checkpoint).unwrap();
        assert_eq!(snapshot.identity, checkpoint_identity(&checkpoint).unwrap());
        let metadata: String = runtime_weights::checkpoint::metadata_identity(&checkpoint)
            .unwrap()
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();
        assert_eq!(snapshot.metadata_identity, metadata);
        assert_eq!(snapshot.headers().len(), 1);
        assert_eq!(
            snapshot.document("config.json").unwrap(),
            br#"{"option":"A"}"#
        );
        snapshot.verify_contents(&checkpoint).unwrap();

        let large = checkpoint.join("oversized.json");
        fs::File::create(&large)
            .unwrap()
            .set_len((64 << 20) + 1)
            .unwrap();
        assert!(
            CheckpointSnapshot::capture(&checkpoint)
                .err()
                .unwrap()
                .to_string()
                .contains("limit")
        );
        let selected =
            CheckpointSnapshot::capture_documents(&checkpoint, &["config.json"]).unwrap();
        assert!(selected.document("oversized.json").is_err());
        selected.verify_contents(&checkpoint).unwrap();
        fs::remove_file(large).unwrap();
        let shard = checkpoint.join("model.safetensors");
        fs::write(&shard, ((256_u64 << 20) + 1).to_le_bytes()).unwrap();
        fs::OpenOptions::new()
            .write(true)
            .open(shard)
            .unwrap()
            .set_len(257 << 20)
            .unwrap();
        assert!(
            CheckpointSnapshot::capture(&checkpoint)
                .err()
                .unwrap()
                .to_string()
                .contains("bounds")
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn regular_snapshot_documents_refuse_even_confined_checkpoint_links() {
        let root = checkpoint_root("direct-snapshot-links");
        let checkpoint = root.join("cache/snapshots/revision");
        let blobs = root.join("cache/blobs");
        fs::create_dir_all(&checkpoint).unwrap();
        fs::create_dir_all(&blobs).unwrap();
        fs::copy(
            root.join("checkpoint/model.safetensors"),
            checkpoint.join("model.safetensors"),
        )
        .unwrap();
        fs::write(blobs.join("tokenizer"), b"{}").unwrap();
        std::os::unix::fs::symlink("../../blobs/tokenizer", checkpoint.join("tokenizer.json"))
            .unwrap();
        let snapshot =
            CheckpointSnapshot::capture_documents(&checkpoint, &["tokenizer.json"]).unwrap();
        assert_eq!(snapshot.document("tokenizer.json").unwrap(), b"{}");
        assert!(
            snapshot
                .regular_document("tokenizer.json")
                .err()
                .unwrap()
                .to_string()
                .contains("non-symlink")
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn snapshot_reuse_does_not_open_a_second_set_of_shards() {
        const CHILD: &str = "CHOPINFER_TEST_SNAPSHOT_FD_CHILD";
        if std::env::var_os(CHILD).is_none() {
            let output = std::process::Command::new("python3")
                .args(["-B", "-c", "import os,resource,sys; soft,hard=resource.getrlimit(resource.RLIMIT_NOFILE); resource.setrlimit(resource.RLIMIT_NOFILE,(128,hard)); os.execv(sys.argv[1],sys.argv[1:])"])
                .arg(std::env::current_exe().unwrap())
                .args(["--exact", "preparation::direct::tests::snapshot_reuse_does_not_open_a_second_set_of_shards", "--nocapture"])
                .env(CHILD, "1")
                .output().unwrap();
            assert!(
                output.status.success(),
                "{}\n{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
            assert!(String::from_utf8_lossy(&output.stdout).contains("1 passed"));
            return;
        }
        struct CapturedFamily;
        impl Provider for CapturedFamily {
            fn name(&self) -> &str {
                "snapshot-every-rank-count"
            }
            fn profiles(&self) -> Vec<String> {
                Vec::new()
            }
            fn probe(&self, _: &Path, _: u32) -> Result<()> {
                Ok(())
            }
            fn direct_snapshot(&self, root: &Path) -> Result<Option<SnapshotProvider>> {
                let snapshot =
                    std::sync::Arc::new(CheckpointSnapshot::capture_documents(root, &[])?);
                Ok(Some(SnapshotProvider::new(
                    snapshot,
                    Box::<EveryRankCount>::default(),
                )))
            }
        }
        let root = checkpoint_root("direct-snapshot-fds");
        let checkpoint = root.join("checkpoint");
        for index in 1..96 {
            fs::copy(
                checkpoint.join("model.safetensors"),
                checkpoint.join(format!("model-{index}.safetensors")),
            )
            .unwrap();
        }
        let request = PrepareDirectRequest {
            path: checkpoint,
            cache_root: root.join("cache"),
            tensor_parallel_size: 4,
            activation_policy: None,
            modalities: ModalitySelection::Auto,
        };
        let first = prepare_direct(&CapturedFamily, &request).unwrap().package;
        let reused = prepare_direct(&CapturedFamily, &request).unwrap().package;
        assert_eq!(first.package_identity, reused.package_identity);
        assert!(reused.measurements.stages.contains_key("package_reuse"));
        fs::remove_dir_all(root).unwrap();
    }

    /// Each input the emission digest binds changes it; where the
    /// checkpoint and the cache are does not.
    #[test]
    fn each_emission_input_changes_the_digest() {
        let request = PrepareDirectRequest {
            path: "/checkpoint".into(),
            cache_root: "/cache".into(),
            tensor_parallel_size: 4,
            activation_policy: None,
            modalities: ModalitySelection::Auto,
        };
        let of = |family: &str,
                  helper: &str,
                  inputs: &[u8],
                  request: &PrepareDirectRequest,
                  identity: &str,
                  metadata: &str| {
            emission_digest(family, helper, inputs, request, identity, metadata).unwrap()
        };
        let base = of(
            "family", "helper", b"inputs", &request, "header", "metadata",
        );
        let ranks = PrepareDirectRequest {
            tensor_parallel_size: 2,
            ..request.clone()
        };
        let policy = PrepareDirectRequest {
            activation_policy: Some("bf16".into()),
            ..request.clone()
        };
        let text = PrepareDirectRequest {
            modalities: ModalitySelection::Only(
                [runtime_bundle::preparation::Modality::Text].into(),
            ),
            ..request.clone()
        };
        let changed = [
            of("other", "helper", b"inputs", &request, "header", "metadata"),
            of("family", "other", b"inputs", &request, "header", "metadata"),
            of("family", "helper", b"other", &request, "header", "metadata"),
            of("family", "helper", b"inputs", &ranks, "header", "metadata"),
            of("family", "helper", b"inputs", &policy, "header", "metadata"),
            of("family", "helper", b"inputs", &text, "header", "metadata"),
            of("family", "helper", b"inputs", &request, "other", "metadata"),
            of("family", "helper", b"inputs", &request, "header", "other"),
        ];
        for (index, digest) in changed.iter().enumerate() {
            assert_ne!(*digest, base, "input {index}");
        }
        let moved = PrepareDirectRequest {
            path: "/elsewhere".into(),
            cache_root: "/another-cache".into(),
            ..request.clone()
        };
        assert_eq!(
            of("family", "helper", b"inputs", &moved, "header", "metadata"),
            base
        );
    }

    /// A native module's inputs name its bytes and the value of each
    /// environment variable the emission reads, unset ones as `null`.
    #[test]
    fn module_inputs_name_the_module_and_the_environment() {
        let root = std::env::temp_dir().join(format!(
            "direct-module-inputs-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        let module = root.join("libmodule.so");
        fs::write(&module, b"module bytes").unwrap();
        let unset = "CHOPINFER_DIRECT_INPUTS_TEST_UNSET";
        let inputs = native_module_inputs(&module, &[unset]).unwrap();
        let value: serde_json::Value = serde_json::from_slice(&inputs).unwrap();
        assert_eq!(
            value["native_module"]["sha256"],
            runtime_weights::sha256_bytes(b"module bytes")
        );
        assert_eq!(value["environment"], serde_json::json!({ unset: null }));
        assert_ne!(inputs, native_module_inputs(&module, &[]).unwrap());
        assert!(native_module_staged(&module, Some(&inputs)).is_ok());
        let renamed = root.join("libother.so");
        fs::copy(&module, &renamed).unwrap();
        assert!(native_module_staged(&renamed, Some(&inputs)).is_err());
        fs::remove_dir_all(&root).unwrap();
    }

    /// A profile request names no existing cache: it reads no record and
    /// creates no cache.
    #[test]
    fn a_profile_without_a_cache_creates_none() {
        let root = std::env::temp_dir().join(format!(
            "direct-profile-no-cache-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        for cache_root in [root.join("cache"), PathBuf::new()] {
            let request = PrepareDirectRequest {
                path: root.clone(),
                cache_root: cache_root.clone(),
                tensor_parallel_size: 1,
                activation_policy: None,
                modalities: ModalitySelection::Auto,
            };
            assert!(
                published_profile(&InputsWithoutEmission, &root, &request)
                    .unwrap()
                    .is_none()
            );
        }
        assert!(!root.join("cache").exists());
        fs::remove_dir_all(&root).unwrap();
    }
}
