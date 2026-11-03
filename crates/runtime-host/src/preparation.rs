//! Bounded installed-provider discovery and a leased canonical-package handoff.
use crate::BundleSelectionV1;
use runtime_bundle::preparation::{
    CacheLease, DIRECT_VERSION, DirectPackage, DirectProfile, Envelope, PrepareDirectRequest,
    Request, Response, Result, default_cache_root, read_record, valid_digest, write_record,
};
use runtime_weights::source_file::SourceFile;
use std::fs;
use std::io::BufReader;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Stdio};

/// The rank count `serve --model` prepares for without
/// `--tensor-parallel-size`.
pub const DEFAULT_TENSOR_PARALLEL_SIZE: u32 = 4;

/// The installed family helpers are named `chopinfer-model-<model>`.
const HELPER_PREFIX: &str = "chopinfer-model-";

#[derive(Clone, Eq, PartialEq)]
pub struct LocalModel {
    pub path: PathBuf,
    pub cache_root: Option<PathBuf>,
    pub tensor_parallel_size: u32,
    pub activation_policy: Option<String>,
    /// The modalities the package serves: `auto` serves text and image of
    /// those the checkpoint provides, `all` every one.
    pub modalities: runtime_bundle::preparation::ModalitySelection,
    /// The invocation's serve options other than the model selection and
    /// preparation options; they override the family's serving profile.
    pub serve_arguments: Vec<String>,
}

impl std::fmt::Debug for LocalModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Read option boundaries with the serving grammar: a key may look
        // like a flag, and another option's value may spell --api-key.
        let arguments: Vec<String> = crate::profiles::options(&self.serve_arguments)
            .into_iter()
            .flat_map(|(flag, values)| {
                let values = if flag == "--api-key" {
                    values.into_iter().map(|_| "[redacted]".into()).collect()
                } else {
                    values
                };
                std::iter::once(flag).chain(values)
            })
            .collect();
        f.debug_struct("LocalModel")
            .field("path", &self.path)
            .field("cache_root", &self.cache_root)
            .field("tensor_parallel_size", &self.tensor_parallel_size)
            .field("activation_policy", &self.activation_policy)
            .field("modalities", &self.modalities)
            .field("serve_arguments", &arguments)
            .finish()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ServeSelection {
    Bundle(BundleSelectionV1),
    Model(Box<LocalModel>),
}

struct Provider {
    path: PathBuf,
    source: SourceFile,
}

fn installed_providers(executable: &Path) -> Result<Vec<Provider>> {
    let executable = fs::canonicalize(executable)?;
    let root = executable
        .parent()
        .ok_or("host executable has no install directory")?;
    let metadata = fs::symlink_metadata(root)?;
    if !metadata.is_dir() || metadata.permissions().mode() & 0o022 != 0 {
        return Err("installed helper directory must not be group/world writable".into());
    }
    let mut providers = vec![];
    for (visited, entry) in fs::read_dir(root)?.enumerate() {
        if visited >= 4096 {
            return Err("installed helper directory exceeds discovery limit".into());
        }
        let entry = entry?;
        let name = entry.file_name();
        let Some(name) = name.to_str() else { continue };
        let Some(suffix) = name.strip_prefix(HELPER_PREFIX) else {
            continue;
        };
        if suffix.is_empty()
            || suffix.len() > 128
            || !suffix
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_')
        {
            continue;
        }
        let path = entry.path();
        let metadata = fs::symlink_metadata(&path)?;
        if !metadata.is_file()
            || metadata.permissions().mode() & 0o022 != 0
            || metadata.permissions().mode() & 0o111 == 0
            || fs::canonicalize(&path)? != path
        {
            return Err(format!("unsafe installed helper {}", path.display()).into());
        }
        providers.push(Provider {
            source: SourceFile::open(&path)?,
            path,
        });
        if providers.len() > 32 {
            return Err("too many installed preparation providers".into());
        }
    }
    providers.sort_by(|a, b| a.path.cmp(&b.path));
    Ok(providers)
}

struct Exchange {
    child: Child,
    input: ChildStdin,
    reply: std::sync::mpsc::Receiver<std::result::Result<Response, String>>,
}
impl Exchange {
    fn start(provider: &Provider, version: u32, request: Request) -> Result<Self> {
        provider.source.verify()?;
        let mut child = Command::new(&provider.path)
            .arg("--protocol")
            .current_dir(
                provider
                    .path
                    .parent()
                    .ok_or("helper has no install directory")?,
            )
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()?;
        let input = child.stdin.take().ok_or("helper stdin unavailable")?;
        let stdout = child.stdout.take().ok_or("helper stdout unavailable")?;
        let (sender, reply) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let _ =
                sender.send(read_record(&mut BufReader::new(stdout)).map_err(|e| e.to_string()));
        });
        let mut exchange = Self {
            child,
            input,
            reply,
        };
        write_record(&Envelope { version, request }, &mut exchange.input)?;
        provider.source.verify()?;
        Ok(exchange)
    }
    fn receive(&mut self, probe: bool) -> Result<Response> {
        let result = if probe {
            self.reply
                .recv_timeout(std::time::Duration::from_secs(10))
                .map_err(|_| "installed provider did not answer discovery within 10 seconds")?
        } else {
            self.reply
                .recv()
                .map_err(|_| "installed provider closed its response channel")?
        };
        result.map_err(Into::into)
    }
}
impl Drop for Exchange {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// The architectures `config.json` names, for refusals that say which model
/// no installed family implementation accepts. It selects nothing: each
/// installed helper probes the checkpoint itself.
fn architectures(path: &Path) -> Result<Vec<String>> {
    let config = path.join("config.json");
    let metadata = fs::metadata(&config).map_err(|e| {
        format!(
            "{} has no readable config.json ({e}); --model needs a Hugging Face checkpoint directory",
            path.display()
        )
    })?;
    if !metadata.is_file() || metadata.len() > 4 << 20 {
        return Err(format!("{} is not a config file of at most 4 MiB", config.display()).into());
    }
    let value: serde_json::Value = serde_json::from_slice(&fs::read(&config)?)
        .map_err(|e| format!("{} is not JSON: {e}", config.display()))?;
    let mut names: Vec<String> = value
        .get("architectures")
        .and_then(serde_json::Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|name| name.as_str().map(str::to_owned))
        .collect();
    if names.is_empty()
        && let Some(model_type) = value.get("model_type").and_then(serde_json::Value::as_str)
    {
        names.push(format!("model_type {model_type}"));
    }
    if names.is_empty() {
        return Err(format!("{} names no architectures or model_type", config.display()).into());
    }
    Ok(names)
}

/// Selects or emits, through the one installed family helper that accepts the
/// checkpoint, a package without rank images for loading it directly. Nothing
/// is materialized and no tensor payload is read or hashed.
///
/// `check` judges the invocation against the family's serving profile, which
/// the helper names before it emits anything, so a refusal leaves no
/// package work behind.
pub(crate) fn prepare_direct(
    model: &LocalModel,
    check: &dyn Fn(&DirectProfile) -> Result<()>,
) -> Result<(DirectPackage, CacheLease)> {
    prepare_direct_with_executable(model, &std::env::current_exe()?, check)
}

/// The canonical checkpoint directory `--model` names.
fn checkpoint_directory(path: &Path) -> Result<PathBuf> {
    let canonical =
        fs::canonicalize(path).map_err(|e| format!("--model {}: {e}", path.display()))?;
    if !fs::symlink_metadata(&canonical)?.is_dir() {
        return Err("--model must name a local checkpoint directory".into());
    }
    Ok(canonical)
}

/// The one installed family helper beside `executable` that accepts the
/// checkpoint directory `path` at `ranks`, and whether it names its serving
/// profile before packaging.
fn accepting_provider(path: &Path, ranks: u32, executable: &Path) -> Result<(Provider, bool)> {
    let architectures = architectures(path)?;
    let mut providers = installed_providers(executable)?;
    let mut accepted = vec![];
    let mut refusals = vec![];
    for (index, provider) in providers.iter().enumerate() {
        let describe =
            Exchange::start(provider, DIRECT_VERSION, Request::Describe)?.receive(true)?;
        let name = match describe {
            // A helper of protocol version 1 refuses the version 2 request.
            Response::Error { .. } => {
                refusals.push(format!(
                    "{} predates direct loading (protocol version 1); install the helper that \
                     matches this chopinfer",
                    provider.path.file_name().map_or_else(
                        || provider.path.display().to_string(),
                        |n| n.to_string_lossy().into_owned()
                    )
                ));
                continue;
            }
            Response::Description {
                version: DIRECT_VERSION,
                provider,
                capabilities,
                ..
            } => {
                if !capabilities
                    .iter()
                    .any(|c| c == runtime_bundle::preparation::PREPARE_DIRECT_CAPABILITY)
                {
                    refusals.push(format!(
                        "{provider} predates direct loading; install the helper that matches this chopinfer"
                    ));
                    continue;
                }
                let names_profile = capabilities
                    .iter()
                    .any(|c| c == runtime_bundle::preparation::DIRECT_PROFILE_CAPABILITY);
                (provider, names_profile)
            }
            _ => return Err("installed helper has an unsupported describe response".into()),
        };
        let probe = Exchange::start(
            provider,
            DIRECT_VERSION,
            Request::Probe {
                path: path.to_path_buf(),
                tensor_parallel_size: ranks,
            },
        )?
        .receive(true)?;
        match probe {
            Response::Probe {
                compatible: true, ..
            } => accepted.push((index, name.1)),
            Response::Probe { reason, .. } => refusals.push(format!("{}: {reason}", name.0)),
            _ => return Err("installed helper returned an invalid compatibility probe".into()),
        }
    }
    let (index, names_profile) = match accepted.as_slice() {
        [accepted] => *accepted,
        [] => {
            return Err(format!(
                "no installed family implementation accepts {} (config.json names {}) at \
                 --tensor-parallel-size {ranks}; install the family helper beside chopinfer{}{}",
                path.display(),
                architectures.join(", "),
                if refusals.is_empty() { "" } else { ". " },
                refusals.join("; ")
            )
            .into());
        }
        _ => {
            return Err("ambiguous checkpoint: more than one installed provider accepts it".into());
        }
    };
    Ok((providers.swap_remove(index), names_profile))
}

/// The model `<m>` (`models/<m>`) of the checkpoint directory `path`: the
/// installed helper `chopinfer-model-<m>` that `serve --model` selects for
/// it at the default rank count.
pub(crate) fn checkpoint_model(path: &Path) -> Result<String> {
    checkpoint_model_with_executable(path, &std::env::current_exe()?)
}

fn checkpoint_model_with_executable(path: &Path, executable: &Path) -> Result<String> {
    Ok(
        checkpoint_helper_with_executable(path, DEFAULT_TENSOR_PARALLEL_SIZE, executable)?
            .model()?
            .to_owned(),
    )
}

/// One installed family helper, for the catalog commands (`chopinfer
/// models`).
pub(crate) struct InstalledHelper {
    provider: Provider,
}

/// The installed family helpers beside this executable.
pub(crate) fn installed_helpers() -> Result<Vec<InstalledHelper>> {
    installed_helpers_with_executable(&std::env::current_exe()?)
}

pub(crate) fn installed_helpers_with_executable(executable: &Path) -> Result<Vec<InstalledHelper>> {
    Ok(installed_providers(executable)?
        .into_iter()
        .map(|provider| InstalledHelper { provider })
        .collect())
}

/// The installed helper `serve --model` selects for the checkpoint directory
/// `path` at `ranks`.
pub(crate) fn checkpoint_helper(path: &Path, ranks: u32) -> Result<InstalledHelper> {
    checkpoint_helper_with_executable(path, ranks, &std::env::current_exe()?)
}

pub(crate) fn checkpoint_helper_with_executable(
    path: &Path,
    ranks: u32,
    executable: &Path,
) -> Result<InstalledHelper> {
    let path = checkpoint_directory(path)?;
    let (provider, _) = accepting_provider(&path, ranks, executable)?;
    Ok(InstalledHelper { provider })
}

/// The most bytes a helper's `serving-profiles` may print.
const SERVING_PROFILES_MAX_BYTES: u64 = runtime_bundle::deployment::DEPLOYMENT_COMPONENT_MAX_BYTES;

impl InstalledHelper {
    /// Its file name, `chopinfer-model-<m>`.
    pub(crate) fn name(&self) -> String {
        self.provider.path.file_name().map_or_else(
            || self.provider.path.display().to_string(),
            |name| name.to_string_lossy().into_owned(),
        )
    }

    /// The model `<m>` its name carries.
    pub(crate) fn model(&self) -> Result<&str> {
        self.provider
            .path
            .file_name()
            .and_then(|name| name.to_str())
            .and_then(|name| name.strip_prefix(HELPER_PREFIX))
            .ok_or_else(|| "installed helper has no model name".into())
    }

    /// What its `serving-profiles` command prints: the family's serving
    /// profiles document, or nothing when it declares none. A helper that
    /// fails, outgrows the bound or takes more than 10 seconds is refused.
    pub(crate) fn serving_profiles(&self) -> Result<Vec<u8>> {
        use std::io::Read;
        let provider = &self.provider;
        provider.source.verify()?;
        let mut child = Command::new(&provider.path)
            .arg("serving-profiles")
            .current_dir(
                provider
                    .path
                    .parent()
                    .ok_or("helper has no install directory")?,
            )
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()?;
        let stdout = child.stdout.take().ok_or("helper stdout unavailable")?;
        let mut stderr = child.stderr.take().ok_or("helper stderr unavailable")?;
        let (sender, reply) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let mut bytes = Vec::new();
            let read = stdout
                .take(SERVING_PROFILES_MAX_BYTES + 1)
                .read_to_end(&mut bytes);
            let _ = sender.send(read.map(|_| bytes));
        });
        // Stderr drains beside stdout, so a helper that writes much to it
        // cannot block before it finishes its document.
        let (said, message) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let mut text = Vec::new();
            let _ = (&mut stderr).take(4096).read_to_end(&mut text);
            let _ = std::io::copy(&mut stderr, &mut std::io::sink());
            let _ = said.send(String::from_utf8_lossy(&text).into_owned());
        });
        let timeout = std::time::Duration::from_secs(10);
        let deadline = std::time::Instant::now() + timeout;
        let received = reply.recv_timeout(timeout);
        let status = loop {
            if let Some(status) = child.try_wait()? {
                break Some(status);
            }
            if std::time::Instant::now() >= deadline {
                let _ = child.kill();
                let _ = child.wait();
                break None;
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        };
        let late = || "installed helper did not print its serving profiles within 10 seconds";
        let (bytes, status) = (received.map_err(|_| late())??, status.ok_or_else(late)?);
        let message = message
            .recv_timeout(std::time::Duration::from_secs(1))
            .unwrap_or_default();
        provider.source.verify()?;
        if bytes.len() as u64 > SERVING_PROFILES_MAX_BYTES {
            return Err("installed helper's serving profiles exceed their bound".into());
        }
        if !status.success() {
            let message = message.trim();
            return Err(if message.is_empty() {
                format!("serving-profiles failed with {status}")
            } else {
                format!("serving-profiles failed: {message}")
            }
            .into());
        }
        Ok(bytes)
    }

    /// The serving profile the helper names for the checkpoint directory
    /// `path` at `ranks`, the one `serve --model` serves it with, or the
    /// helper's refusal. Nothing is prepared and no cache is read.
    pub(crate) fn profile(&self, path: &Path, ranks: u32) -> Result<DirectProfile> {
        let request = PrepareDirectRequest {
            path: checkpoint_directory(path)?,
            cache_root: PathBuf::new(),
            tensor_parallel_size: ranks,
            activation_policy: None,
            modalities: Default::default(),
        };
        match Exchange::start(
            &self.provider,
            DIRECT_VERSION,
            Request::DirectProfile(request),
        )?
        .receive(false)?
        {
            Response::DirectProfileSelected(profile) => Ok(profile),
            Response::Error { message, .. } => Err(message.into()),
            _ => Err("installed helper returned no serving profile".into()),
        }
    }
}

fn prepare_direct_with_executable(
    model: &LocalModel,
    executable: &Path,
    check: &dyn Fn(&DirectProfile) -> Result<()>,
) -> Result<(DirectPackage, CacheLease)> {
    let path = checkpoint_directory(&model.path)?;
    let discovery_started = std::time::Instant::now();
    let (provider, names_profile) =
        accepting_provider(&path, model.tensor_parallel_size, executable)?;
    let provider = &provider;
    runtime_log::info!(
        "startup provider_discovery: {:.6}s",
        discovery_started.elapsed().as_secs_f64()
    );
    // The provider runs in its install directory; resolve caller paths before sending them.
    let cache_root = std::path::absolute(
        model
            .cache_root
            .clone()
            .map_or_else(default_cache_root, Ok)?,
    )?;
    let request = PrepareDirectRequest {
        path,
        cache_root: cache_root.clone(),
        tensor_parallel_size: model.tensor_parallel_size,
        activation_policy: model.activation_policy.clone(),
        modalities: model.modalities.clone(),
    };
    // The serving profile first: options it refuses are refused before the
    // helper emits or caches anything.
    let checked = if names_profile {
        let profile = match Exchange::start(
            provider,
            DIRECT_VERSION,
            Request::DirectProfile(request.clone()),
        )?
        .receive(false)?
        {
            Response::DirectProfileSelected(profile) => profile,
            Response::Error { message, .. } => return Err(message.into()),
            _ => return Err("installed helper returned no serving profile".into()),
        };
        check(&profile)?;
        Some(profile)
    } else {
        None
    };
    let mut exchange = Exchange::start(provider, DIRECT_VERSION, Request::PrepareDirect(request))?;
    let package = match exchange.receive(false)? {
        Response::PreparedDirect(package) => *package,
        Response::Error { message, .. } => return Err(message.into()),
        _ => {
            return Err(
                "installed helper returned no direct package; install a helper that loads checkpoints directly"
                    .into(),
            );
        }
    };
    // The whole profile, not only its name: the options were checked
    // against exactly the profile the package must carry.
    if let Some(checked) = checked.filter(|checked| *checked != package.profile) {
        return Err(format!(
            "installed helper named serving profile {} before packaging but packaged a \
             different {} profile",
            checked.name, package.profile.name
        )
        .into());
    }
    let cache_root = fs::canonicalize(cache_root)?;
    if !valid_digest(&package.package_identity)
        || !valid_digest(&package.checkpoint_identity)
        || !valid_digest(&package.metadata_identity)
        || package.cache_root != cache_root
        || package.bundle != cache_root.join(format!("p-{}", package.package_identity))
        || package.tensor_parallel_size != model.tensor_parallel_size
        || package.qualification != "compatible-not-evaluated"
    {
        return Err(
            "installed helper returned an invalid package identity or cache location".into(),
        );
    }
    let lease = CacheLease::acquire(&cache_root, &[format!("p-{}", package.package_identity)])?;
    provider.source.verify()?;
    write_record(
        &Envelope {
            version: DIRECT_VERSION,
            request: Request::Acknowledge {
                package_identity: package.package_identity.clone(),
            },
        },
        &mut exchange.input,
    )?;
    // The lease is already ours, so ending this helper cannot open a prune window.
    if !exchange.child.wait()?.success() {
        return Err("preparation helper failed during lease handoff".into());
    }
    runtime_log::info!(
        "startup direct_package: {}",
        serde_json::to_string(&package)?
    );
    Ok((package, lease))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An install directory with a host binary and a fake helper per
    /// (model, accepts) that answers discovery and the probe.
    fn install(name: &str, helpers: &[(&str, bool)]) -> PathBuf {
        let root = std::env::temp_dir().join(format!("preparation-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        let bin = root.join("bin");
        fs::create_dir_all(&bin).unwrap();
        fs::set_permissions(&bin, fs::Permissions::from_mode(0o755)).unwrap();
        fs::write(bin.join("chopinfer"), b"").unwrap();
        for (model, accepts) in helpers {
            let helper = bin.join(format!("{HELPER_PREFIX}{model}"));
            let script = format!(
                "#!/bin/sh\nread -r request\ncase \"$request\" in\n\
                 *'\"describe\"'*) echo '{{\"result\":\"description\",\"version\":2,\
                 \"provider\":\"{HELPER_PREFIX}{model}\",\"profiles\":[],\
                 \"capabilities\":[\"prepare-direct\"]}}' ;;\n\
                 *'\"probe\"'*) echo '{{\"result\":\"probe\",\"compatible\":{accepts},\
                 \"reason\":\"{model} refuses it\"}}' ;;\nesac\n"
            );
            fs::write(&helper, script).unwrap();
            fs::set_permissions(&helper, fs::Permissions::from_mode(0o755)).unwrap();
        }
        let checkpoint = root.join("checkpoint");
        fs::create_dir_all(&checkpoint).unwrap();
        fs::write(
            checkpoint.join("config.json"),
            r#"{"architectures": ["ExampleForCausalLM"]}"#,
        )
        .unwrap();
        root
    }

    /// A helper that also prints serving profiles and names a profile per
    /// rank count (refusing two ranks), or one that fails `serving-profiles`
    /// and refuses every checkpoint.
    fn catalog_helper(path: &Path, profiles: &str, fails: bool) {
        let script = format!(
            "#!/bin/sh\nif [ \"$1\" = serving-profiles ]; then\n{}\nfi\n\
             read -r request\ncase \"$request\" in\n\
             *'\"describe\"'*) echo '{{\"result\":\"description\",\"version\":2,\
             \"provider\":\"chopinfer-model-alpha\",\"profiles\":[],\
             \"capabilities\":[\"prepare-direct\",\"direct-profile\"]}}' ;;\n\
             *'\"probe\"'*) echo '{{\"result\":\"probe\",\"compatible\":{},\
             \"reason\":\"compatible\"}}' ;;\n\
             *'\"tensor_parallel_size\":2'*) echo '{{\"result\":\"error\",\
             \"message\":\"two ranks need a drafter\",\"measurements\":{{\
             \"total_seconds\":0,\"stages\":{{}},\"decisions\":[]}}}}' ;;\n\
             *'\"direct-profile\"'*) echo '{{\"result\":\"direct-profile-selected\",\
             \"name\":\"tp4-ar\",\"profile\":{{\"tensor_parallel\":4}}}}' ;;\nesac\n",
            if fails {
                "echo 'profiles are missing' >&2; exit 3".to_owned()
            } else {
                format!("printf '%s' '{profiles}'; exit 0")
            },
            !fails
        );
        fs::write(path, script).unwrap();
        fs::set_permissions(path, fs::Permissions::from_mode(0o755)).unwrap();
    }

    /// The catalog commands read a helper's serving profiles and the profile
    /// it names for a checkpoint at each rank count, or its refusal; a helper
    /// whose `serving-profiles` fails is refused with what it said.
    #[test]
    fn installed_helpers_print_their_profiles_and_name_the_pick() {
        let root = install("catalog", &[("alpha", true), ("beta", false)]);
        let bin = root.join("bin");
        let document = r#"{"schema":"chopinfer-serve-profiles-v2","profiles":{}}"#;
        catalog_helper(&bin.join(format!("{HELPER_PREFIX}alpha")), document, false);
        catalog_helper(&bin.join(format!("{HELPER_PREFIX}beta")), document, true);
        let executable = bin.join("chopinfer");
        let helpers = installed_helpers_with_executable(&executable).unwrap();
        let names: Vec<String> = helpers.iter().map(InstalledHelper::name).collect();
        assert_eq!(names, ["chopinfer-model-alpha", "chopinfer-model-beta"]);
        assert_eq!(helpers[0].model().unwrap(), "alpha");
        assert_eq!(helpers[0].serving_profiles().unwrap(), document.as_bytes());
        let error = helpers[1].serving_profiles().unwrap_err().to_string();
        assert!(error.contains("profiles are missing"), "{error}");
        let checkpoint = root.join("checkpoint");
        let helper = checkpoint_helper_with_executable(&checkpoint, 4, &executable).unwrap();
        assert_eq!(helper.name(), "chopinfer-model-alpha");
        let profile = helper.profile(&checkpoint, 4).unwrap();
        assert_eq!(profile.name, "tp4-ar");
        assert_eq!(profile.profile["tensor_parallel"], 4);
        let refusal = helper.profile(&checkpoint, 2).unwrap_err().to_string();
        assert_eq!(refusal, "two ranks need a drafter");
        fs::remove_dir_all(root).unwrap();
    }

    fn model_of(root: &Path) -> Result<String> {
        checkpoint_model_with_executable(&root.join("checkpoint"), &root.join("bin/chopinfer"))
    }

    #[test]
    fn a_checkpoint_names_the_model_of_the_one_helper_that_accepts_it() {
        let root = install("one", &[("alpha", false), ("beta", true)]);
        assert_eq!(model_of(&root).unwrap(), "beta");
        fs::remove_dir_all(root).unwrap();

        let root = install("none", &[("alpha", false)]);
        let error = model_of(&root).unwrap_err().to_string();
        assert!(
            error.contains("no installed family implementation accepts")
                && error.contains("ExampleForCausalLM")
                && error.contains("alpha refuses it"),
            "{error}"
        );
        fs::remove_dir_all(root).unwrap();

        let root = install("two", &[("alpha", true), ("beta", true)]);
        let error = model_of(&root).unwrap_err().to_string();
        assert!(error.contains("ambiguous checkpoint"), "{error}");
        let file = root.join("checkpoint/config.json");
        let error = checkpoint_model_with_executable(&file, &root.join("bin/chopinfer"))
            .unwrap_err()
            .to_string();
        assert!(error.contains("local checkpoint directory"), "{error}");
        fs::remove_dir_all(root).unwrap();
    }
}
