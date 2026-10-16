//! A small CLI and wire adapter. Preparation decisions stay in the library.
use super::direct::{DirectPrepared, prepare_direct};
use super::{list, prune};
use crate::files::Result;
use runtime_bundle::preparation::{
    DIRECT_VERSION, Measurements, Modality, ModalitySelection, PrepareDirectRequest, Request,
    Response, default_cache_root, read_envelope, read_request, write_record,
};
use std::io::{BufRead, Write};
use std::path::Path;

/// A family helper's implementation. It is shared with a thread that selects
/// its named serving profile beside its emission, so it is `Sync`.
pub trait Provider: Sync {
    fn name(&self) -> &str;
    fn profiles(&self) -> Vec<String>;
    /// The family's deployment policy ([`crate::deployment_policy`]), the
    /// document its packages carry. `deployment-policy` prints it, so the
    /// image entrypoint knows every installed family's variables before
    /// any checkpoint is prepared. `None`, the default, declares none.
    fn deployment_policy(&self) -> Option<&str> {
        None
    }
    /// Metadata-only routing. Full admission still happens inside
    /// `emit_direct`.
    fn probe(&self, path: &Path, ranks: u32) -> Result<()>;
    /// Emits at `output` a package without rank images that serves the
    /// checkpoint `root` by loading it directly: its compiled images are
    /// planned components. The package carries the drafter and encoders of
    /// the serving profile it returns ([`super::direct::default_profile`]).
    /// `checkpoint_identity` is the direct load's identity of `root` (see
    /// [`super::direct::checkpoint_identity`]).
    fn emit_direct(
        &self,
        _root: &Path,
        _request: &PrepareDirectRequest,
        _checkpoint_identity: &str,
        _output: &Path,
    ) -> Result<super::direct::DirectEmission> {
        Err(format!("{} does not load checkpoints directly", self.name()).into())
    }
    /// Whether [`Self::direct_profile`] is implemented; the helper then
    /// describes the `direct-profile` capability, and the host checks an
    /// invocation against the profile before it asks for a package.
    fn names_direct_profile(&self) -> bool {
        false
    }
    /// The serving profile [`Self::emit_direct`] would return for `root`,
    /// without emitting a package. Implement it together with
    /// [`Self::names_direct_profile`]. A family that states
    /// [`Self::direct_inputs`] is asked only when no package published for
    /// the same inputs records the profile.
    fn direct_profile(
        &self,
        _root: &Path,
        _request: &PrepareDirectRequest,
    ) -> Result<runtime_bundle::preparation::DirectProfile> {
        Err(format!("{} does not load checkpoints directly", self.name()).into())
    }
    /// Resolve `auto` and `all` to the concrete served set before the
    /// emission lookup. Providers with optional encoders return the same set
    /// their emitter serves, so a keyword and the list it serves share a key.
    /// The default keeps a list and leaves the keywords unresolved, for
    /// providers without this selection.
    fn direct_modalities(
        &self,
        _root: &Path,
        request: &PrepareDirectRequest,
    ) -> Result<Option<std::collections::BTreeSet<Modality>>> {
        Ok(request.modalities.listed().cloned())
    }
    /// With the helper executable, the request and the checkpoint's header
    /// and metadata identities, bytes that decide everything [`Self::emit_direct`]
    /// emits (the native module it packages, for example). A family that
    /// states them has a published package reused without emitting it
    /// again; `None`, the default, emits on every request.
    fn direct_inputs(
        &self,
        _root: &Path,
        _request: &PrepareDirectRequest,
    ) -> Result<Option<Vec<u8>>> {
        Ok(None)
    }
    /// [`Self::emit_direct`] for a request whose emission inputs are
    /// `inputs`, what [`Self::direct_inputs`] returned for it. A family that
    /// states inputs emits exactly what they describe: it stages its native
    /// module authenticated against the identity they carry
    /// ([`super::direct::native_module_staged`]), so a module replaced after
    /// they were taken fails the emission instead of being published under
    /// them. The default emits a family without inputs, and refuses one that
    /// states them: emitting without them could publish other bytes under
    /// their digest.
    fn emit_direct_with_inputs(
        &self,
        root: &Path,
        request: &PrepareDirectRequest,
        checkpoint_identity: &str,
        output: &Path,
        inputs: Option<&[u8]>,
    ) -> Result<super::direct::DirectEmission> {
        if inputs.is_some() {
            return Err(format!(
                "{} states direct emission inputs but does not emit from them \
                 (implement emit_direct_with_inputs)",
                self.name()
            )
            .into());
        }
        self.emit_direct(root, request, checkpoint_identity, output)
    }
}

/// The reply to `request`, in the form of protocol `version`.
fn response<F: Provider>(
    family: &F,
    version: u32,
    request: Request,
) -> Result<(Response, Option<DirectPrepared>)> {
    Ok(match request {
        Request::Describe => (
            Response::Description {
                version,
                provider: family.name().into(),
                profiles: family.profiles(),
                capabilities: if version >= DIRECT_VERSION {
                    let mut capabilities =
                        vec![runtime_bundle::preparation::PREPARE_DIRECT_CAPABILITY.into()];
                    if family.names_direct_profile() {
                        capabilities
                            .push(runtime_bundle::preparation::DIRECT_PROFILE_CAPABILITY.into());
                    }
                    capabilities
                } else {
                    Vec::new()
                },
            },
            None,
        ),
        Request::Probe {
            path,
            tensor_parallel_size,
        } => {
            let result = family.probe(&path, tensor_parallel_size);
            (
                Response::Probe {
                    compatible: result.is_ok(),
                    reason: result
                        .err()
                        .map_or_else(|| "compatible metadata".into(), |e| e.to_string()),
                },
                None,
            )
        }
        Request::DirectProfile(request) => match std::fs::canonicalize(&request.path)
            .map_err(Into::into)
            .and_then(|root| {
                family.probe(&root, request.tensor_parallel_size)?;
                // A package published for the same emission inputs names
                // the profile without the family selecting it again.
                match super::direct::published_profile(family, &root, &request)? {
                    Some(profile) => Ok(profile),
                    None => family.direct_profile(&root, &request),
                }
            }) {
            Ok(profile) => (Response::DirectProfileSelected(profile), None),
            Err(error) => (
                Response::Error {
                    message: error.to_string(),
                    measurements: Measurements::default(),
                },
                None,
            ),
        },
        Request::PrepareDirect(request) => match prepare_direct(family, &request) {
            Ok(prepared) => (
                Response::PreparedDirect(Box::new(prepared.package.clone())),
                Some(prepared),
            ),
            Err(error) => (
                Response::Error {
                    message: error.to_string(),
                    measurements: Measurements::default(),
                },
                None,
            ),
        },
        Request::CacheList { cache_root } => (Response::Cache(list(&cache_root)?), None),
        Request::CachePrune {
            cache_root,
            apply,
            identities,
        } => (
            Response::Cache(prune(&cache_root, apply, &identities)?),
            None,
        ),
        Request::Acknowledge { .. } => return Err("no pending preparation handoff".into()),
    })
}

/// One request per process. A prepared result stays leased until the receiver
/// acknowledges acquiring its own lease. EOF or an invalid acknowledgement fails.
pub fn protocol<F: Provider>(
    family: &F,
    input: &mut impl BufRead,
    output: &mut impl Write,
) -> Result<()> {
    let result = read_envelope(input)
        .and_then(|envelope| response(family, envelope.version, envelope.request));
    let (reply, held) = match result {
        Ok(value) => value,
        Err(e) => (
            Response::Error {
                message: e.to_string(),
                measurements: Measurements::default(),
            },
            None,
        ),
    };
    write_record(&reply, output)?;
    if let Some(held) = held {
        match read_request(input)? {
            Request::Acknowledge { package_identity }
                if package_identity == held.package.package_identity => {}
            _ => return Err("invalid preparation handoff acknowledgement".into()),
        }
    }
    Ok(())
}

pub fn run<F: Provider>(family: &F, arguments: impl IntoIterator<Item = String>) -> Result<()> {
    let mut args = arguments.into_iter();
    let command = args.next().unwrap_or_else(|| "--help".into());
    if command == "--protocol" {
        if args.next().is_some() {
            return Err("--protocol takes no arguments".into());
        }
        return protocol(
            family,
            &mut std::io::stdin().lock(),
            &mut std::io::stdout().lock(),
        );
    }
    if command == "--help" || command == "help" {
        println!(
            "{} prepare [--direct] --model DIR [--tensor-parallel-size N] [--cache-dir DIR] [--activation-policy auto|checkpoint|bf16] [--modalities auto|all|LIST]\n{} profile --model DIR [--tensor-parallel-size N] [--activation-policy auto|checkpoint|bf16] [--modalities auto|all|LIST]\n{} cache list|prune [--cache-dir DIR] [--identity KEY] [--apply]\n{} describe\n{} deployment-policy\n{} --protocol",
            family.name(),
            family.name(),
            family.name(),
            family.name(),
            family.name(),
            family.name()
        );
        return Ok(());
    }
    // The policy document as the family's packages carry it; nothing when
    // the family declares none.
    if command == "deployment-policy" {
        if args.next().is_some() {
            return Err("deployment-policy takes no arguments".into());
        }
        if let Some(policy) = family.deployment_policy() {
            crate::deployment_policy::parse(policy.as_bytes())?;
            std::io::stdout().lock().write_all(policy.as_bytes())?;
        }
        return Ok(());
    }
    let cache_action = if command == "cache" {
        Some(args.next().ok_or("cache needs list or prune")?)
    } else {
        None
    };
    let mut cache_root = None;
    let mut model = None;
    let mut activation_policy = None;
    let mut modalities = None;
    let mut ranks = None;
    let (mut apply, mut direct_flag) = (false, false);
    let mut identities = Vec::new();
    // `profile` names the serving profile `prepare` would give the package,
    // without preparing it.
    let direct = |command: &str| matches!(command, "prepare" | "profile");
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--cache-dir" if cache_root.is_none() => {
                cache_root = Some(args.next().ok_or("missing cache directory")?.into())
            }
            "--model" if direct(&command) && model.is_none() => {
                model = Some(args.next().ok_or("missing model directory")?.into())
            }
            "--activation-policy" if direct(&command) && activation_policy.is_none() => {
                activation_policy = Some(args.next().ok_or("missing activation policy")?)
            }
            "--modalities" if direct(&command) && modalities.is_none() => {
                modalities = Some(ModalitySelection::parse(
                    &args.next().ok_or("missing modalities")?,
                )?)
            }
            "--tensor-parallel-size" if direct(&command) && ranks.is_none() => {
                ranks = Some(args.next().ok_or("missing topology")?.parse()?)
            }
            // Every package is the direct one; the flag names it explicitly.
            "--direct" if command == "prepare" && !direct_flag => direct_flag = true,
            "--snapshot" | "--refresh" | "--native-module" if command == "prepare" => {
                return Err(format!(
                    "{arg} is gone: prepare emits the package that loads the checkpoint \
                     directly and materializes no rank images"
                )
                .into());
            }
            "--apply" if cache_action.as_deref() == Some("prune") && !apply => apply = true,
            "--identity" if cache_action.as_deref() == Some("prune") => {
                identities.push(args.next().ok_or("missing cache identity")?)
            }
            _ => return Err(format!("unsupported or repeated helper argument {arg}").into()),
        }
    }
    let request = match (command.as_str(), cache_action.as_deref()) {
        ("describe", None) if cache_root.is_none() => Request::Describe,
        ("prepare", None) => Request::PrepareDirect(PrepareDirectRequest {
            path: model.ok_or("prepare needs --model DIR")?,
            cache_root: cache_root.map_or_else(default_cache_root, Ok)?,
            tensor_parallel_size: ranks.unwrap_or(4),
            activation_policy,
            modalities: modalities.unwrap_or_default(),
        }),
        ("profile", None) if cache_root.is_none() => {
            Request::DirectProfile(PrepareDirectRequest {
                path: model.ok_or("profile needs --model DIR")?,
                // A profile request prepares nothing and uses no cache.
                cache_root: std::path::PathBuf::new(),
                tensor_parallel_size: ranks.unwrap_or(4),
                activation_policy,
                modalities: modalities.unwrap_or_default(),
            })
        }
        ("cache", Some("list")) => Request::CacheList {
            cache_root: cache_root.map_or_else(default_cache_root, Ok)?,
        },
        ("cache", Some("prune")) => Request::CachePrune {
            cache_root: cache_root.map_or_else(default_cache_root, Ok)?,
            apply,
            identities,
        },
        _ => return Err("expected prepare, profile, describe, or cache list|prune".into()),
    };
    let (reply, _held) = response(family, DIRECT_VERSION, request)?;
    write_record(&reply, &mut std::io::stdout().lock())?;
    if let Response::Error { message, .. } = reply {
        return Err(message.into());
    }
    Ok(())
}
