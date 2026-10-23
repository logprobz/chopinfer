//! Process-start loading of one explicitly selected bundle directory.
//!
//! There is no discovery, no hot reload, and no schema migration. The caller
//! names a package directory, the canonical parser verifies it, and the
//! native module stays resident until the process exits.
//!
//! Verification is not the last check. Every operation the selected bundle's
//! graph buckets reference is resolved for this host's exact selected
//! architecture against this host's own catalog before the native module is
//! opened, so a bundle whose declared work this host cannot perform is
//! refused while it is still only data. Implementations the bundle declares
//! for other architectures are never candidates here.

use std::cell::Cell;
use std::collections::BTreeSet;
use std::fmt;
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::{Mutex, OnceLock};
use std::thread::ThreadId;

use runtime_bundle::{
    BundleError, BundleHostCompatibilityV1, BundleManifestV1, BundleOpaqueId, BundleOperationKeyV1,
    BundleOperationProvisioningV1, BundlePackageV1,
};
use runtime_bundle_abi::{
    BundleAbiError, BundleLibraryV1, BundleLoaderError, HostServicesTableV1, LoadedBundleInstanceV1,
};
use runtime_kernel_contract::{KernelImplementationTierV1, KernelOpaqueIdV1, KernelResolverV1};

use crate::catalog::HostGenericCatalogV1;
use crate::device::HostDeviceBackendV1;
use crate::services::{
    AUTHENTICATED_COMPONENT_ROLE_PREFIX_V1, BUNDLE_METADATA_ROLE_PREFIX_V1, DeclaredKvContractV1,
    DeclaredStateContractV1, HostInstanceConfigV1, HostServicesV1Backend, HostTopologyV1,
};

#[derive(Debug)]
pub enum HostLoadErrorV1 {
    Bundle(BundleError),
    Loader(BundleLoaderError),
    Abi(BundleAbiError),
    NoNativeModule,
    ComponentRead {
        path: String,
        message: String,
    },
    AlreadyLoaded {
        loaded: String,
        requested: String,
    },
    LoadedOnAnotherThread {
        loaded: String,
    },
    UnknownTopology(String),
    UnknownComponentRole(String),
    /// This host's own generic catalog cannot be described.
    HostCatalog(BundleError),
    /// No exact implementation of a declared operation, in any tier the
    /// bundle's own provisioning allows.
    UnsupportedOperation {
        operation: String,
        variant: String,
        reason: String,
    },
    /// An exact implementation matched, but not the one the bundle asked for.
    WrongOperationProvider {
        operation: String,
        variant: String,
        reason: String,
    },
}

impl fmt::Display for HostLoadErrorV1 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Bundle(error) => error.fmt(formatter),
            Self::Loader(error) => error.fmt(formatter),
            Self::Abi(error) => error.fmt(formatter),
            Self::NoNativeModule => {
                formatter.write_str("the selected bundle declares no native module to load")
            }
            Self::ComponentRead { path, message } => {
                write!(
                    formatter,
                    "cannot read bundle component '{path}': {message}"
                )
            }
            Self::AlreadyLoaded { loaded, requested } => write!(
                formatter,
                "this process already loaded bundle '{loaded}' and cannot also load '{requested}'"
            ),
            Self::LoadedOnAnotherThread { loaded } => write!(
                formatter,
                "this process loaded bundle '{loaded}' on another thread, and its native module \
                 stays on the thread that opened it"
            ),
            Self::UnknownTopology(topology) => {
                write!(
                    formatter,
                    "the selected bundle declares no topology '{topology}'"
                )
            }
            Self::UnknownComponentRole(role) => {
                write!(
                    formatter,
                    "the selected bundle declares no component for role '{role}'"
                )
            }
            Self::HostCatalog(error) => {
                write!(
                    formatter,
                    "this host cannot describe its own generic catalog: {error}"
                )
            }
            Self::UnsupportedOperation {
                operation,
                variant,
                reason,
            } => write!(
                formatter,
                "this host cannot run the selected bundle's operation '{operation}/{variant}': \
                 {reason}"
            ),
            Self::WrongOperationProvider {
                operation,
                variant,
                reason,
            } => write!(
                formatter,
                "the selected bundle's operation '{operation}/{variant}' resolved to an \
                 implementation it did not ask for: {reason}"
            ),
        }
    }
}

impl std::error::Error for HostLoadErrorV1 {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Bundle(error) | Self::HostCatalog(error) => Some(error),
            Self::Loader(error) => Some(error),
            Self::Abi(error) => Some(error),
            Self::NoNativeModule
            | Self::ComponentRead { .. }
            | Self::AlreadyLoaded { .. }
            | Self::LoadedOnAnotherThread { .. }
            | Self::UnknownTopology(_)
            | Self::UnknownComponentRole(_)
            | Self::UnsupportedOperation { .. }
            | Self::WrongOperationProvider { .. } => None,
        }
    }
}

impl From<BundleError> for HostLoadErrorV1 {
    fn from(error: BundleError) -> Self {
        Self::Bundle(error)
    }
}

impl From<BundleLoaderError> for HostLoadErrorV1 {
    fn from(error: BundleLoaderError) -> Self {
        Self::Loader(error)
    }
}

impl From<BundleAbiError> for HostLoadErrorV1 {
    fn from(error: BundleAbiError) -> Self {
        Self::Abi(error)
    }
}

/// Which bundle a caller selected: one package directory, by path.
///
/// Local-model preparation hands this layer the directory of the package it
/// selected in the model cache; an explicit `--bundle DIR` names one directly.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum BundleSelectionV1 {
    /// One bundle directory, by path.
    Directory(PathBuf),
}

impl BundleSelectionV1 {
    /// Names this selection for a message, without resolving it.
    #[must_use]
    pub fn display(&self) -> String {
        let Self::Directory(path) = self;
        path.display().to_string()
    }
}

/// Verifies one selected bundle directory, without loading its module.
///
/// Nothing here opens a native module: verification and loading are
/// separate, and only a process that is about to run a bundle loads one.
/// Every declared component is read in full and hashed once, and the package
/// serves those verified bytes from then on. The one `startup verify:` line
/// reports how long that took and how many component bytes it hashed;
/// `image_bytes_read` stays in the line and is always zero, because a package
/// carries no rank image to read.
pub fn verify_selected_bundle_v1(
    selection: &BundleSelectionV1,
    host: &BundleHostCompatibilityV1,
) -> Result<BundlePackageV1, BundleError> {
    let started = std::time::Instant::now();
    let BundleSelectionV1::Directory(path) = selection;
    let result = BundlePackageV1::verify_directory(path, host);
    eprintln!(
        "startup verify: {:.6}s, image_bytes_read=0, component_bytes_hashed={}, status={}",
        started.elapsed().as_secs_f64(),
        result
            .as_ref()
            .map_or(0, BundlePackageV1::component_bytes_verified),
        if result.is_ok() { "verified" } else { "failed" }
    );
    result
}

/// One operation the selected bundle declared, and what will run it.
///
/// This is the record of the check, kept so a caller can see which tier and
/// which implementation each declared operation landed on rather than having to
/// take the load's success as the whole answer.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResolvedOperationV1 {
    key: BundleOperationKeyV1,
    provisioning: BundleOperationProvisioningV1,
    tier: KernelImplementationTierV1,
    implementation: KernelOpaqueIdV1,
}

impl ResolvedOperationV1 {
    /// The operation key the manifest's graph bucket referenced.
    #[must_use]
    pub const fn key(&self) -> &BundleOperationKeyV1 {
        &self.key
    }

    /// How the bundle said this operation would be provided.
    #[must_use]
    pub const fn provisioning(&self) -> &BundleOperationProvisioningV1 {
        &self.provisioning
    }

    /// The tier the exact match came from.
    #[must_use]
    pub const fn tier(&self) -> KernelImplementationTierV1 {
        self.tier
    }

    /// The identity of the implementation that matched.
    #[must_use]
    pub const fn implementation(&self) -> &KernelOpaqueIdV1 {
        &self.implementation
    }
}

/// One verified installed bundle and its open native module.
pub struct InstalledBundleHostV1 {
    package: BundlePackageV1,
    library: BundleLibraryV1,
    model_options: std::cell::OnceCell<std::collections::BTreeMap<String, serde_json::Value>>,
}

impl InstalledBundleHostV1 {
    pub fn with_model_options(
        &self,
        overrides: &std::collections::BTreeMap<String, String>,
    ) -> Result<&Self, HostLoadErrorV1> {
        let values = self
            .manifest()
            .resolve_model_options(overrides)
            .map_err(HostLoadErrorV1::Bundle)?;
        if let Some(configured) = self.model_options.get() {
            if configured != &values {
                return Err(HostLoadErrorV1::Bundle(BundleError::validation(
                    "model options are fixed for the process",
                )));
            }
        } else {
            self.model_options
                .set(values)
                .expect("options are configured once on the owning thread");
        }
        Ok(self)
    }

    fn with_model_options_defaults(&self) -> Result<(), HostLoadErrorV1> {
        if self.model_options.get().is_none() {
            self.with_model_options(&std::collections::BTreeMap::new())?;
        }
        Ok(())
    }

    #[must_use]
    pub const fn manifest(&self) -> &BundleManifestV1 {
        self.package.manifest()
    }

    #[must_use]
    pub fn library(&self) -> &BundleLibraryV1 {
        &self.library
    }

    /// The verified package this process selected.
    #[must_use]
    pub const fn package(&self) -> &BundlePackageV1 {
        &self.package
    }

    /// Reads one component this bundle declares, by the role it serves.
    ///
    /// The bytes are the ones the process-selected bundle authenticated: a
    /// manifest may declare only one component per role, and the read copies
    /// the bytes verification hashed rather than reading a path again.
    /// Nothing here knows what the
    /// role means, so a caller that reads a frontend declaration and a caller
    /// that reads a weight arena take the same path.
    pub fn component_bytes(&self, role: &str) -> Result<Vec<u8>, HostLoadErrorV1> {
        verified_component_by_role_v1(&self.package, role)
    }

    /// Builds the host service configuration this bundle's manifest declares.
    ///
    /// Every value comes from the manifest, so nothing model-specific has to be
    /// compiled into the host to serve it.
    pub fn instance_config(
        &self,
        topology_id: &str,
        device_ordinals: Vec<i32>,
    ) -> Result<HostInstanceConfigV1, HostLoadErrorV1> {
        self.instance_config_with_kv_dtype(topology_id, device_ordinals, None)
    }

    fn instance_config_with_kv_dtype(
        &self,
        topology_id: &str,
        device_ordinals: Vec<i32>,
        kv_dtype: Option<runtime_bundle::BundleKvDtypeV1>,
    ) -> Result<HostInstanceConfigV1, HostLoadErrorV1> {
        let manifest = self.manifest();
        let kv_layout = served_kv_layout_v1(manifest, kv_dtype)?;
        let topology = HostTopologyV1::new(device_ordinals)
            .map_err(|error| BundleError::validation(error.to_string()))?;
        let declared = manifest
            .target()
            .topologies()
            .iter()
            .find(|declared| declared.id().as_str() == topology_id)
            .ok_or_else(|| HostLoadErrorV1::UnknownTopology(topology_id.to_owned()))?;
        if declared.rank_count() != topology.rank_count() {
            return Err(HostLoadErrorV1::Bundle(BundleError::validation(format!(
                "topology '{topology_id}' declares {} ranks, the host offered {}",
                declared.rank_count(),
                topology.rank_count()
            ))));
        }
        let page_layout = manifest
            .declared_page_layout_for(kv_layout, declared.id())
            .map_err(HostLoadErrorV1::Bundle)?;
        // The scored row limit the host's scheduler applies, so the bundle
        // refuses at create a limit it cannot execute.
        let mut batch_limits = runtime_bundle_abi::BundleRawBatchLimitsV1::unreported();
        batch_limits.maximum_scored_rows = manifest.batch_limits().map_or(
            runtime_bundle::DEFAULT_MAXIMUM_SCORED_ROWS,
            runtime_bundle::BundleBatchLimitsV1::maximum_scored_rows,
        );
        let mut config = HostInstanceConfigV1::new(topology)
            .with_batch_limits(batch_limits)
            .with_arena_source(crate::weight_keeper::selected_arena_source_v1());
        if manifest.model_options().is_some() {
            self.with_model_options_defaults()?;
            config = config.with_model_options(
                self.model_options
                    .get()
                    .expect("model options configured")
                    .clone(),
            );
        }
        // A package whose bundle reads the served count serves its maximum
        // unless the caller names fewer sequences.
        let served = served_sequences_v1(
            manifest.resource_requirements().required_capabilities(),
            manifest.batch_limits(),
            None,
        )?;
        if let Some(sequences) = served {
            config = config.with_maximum_sequences(sequences);
        }
        config = config
            .with_kv_contract(DeclaredKvContractV1::new(
                kv_layout.id().as_str().as_bytes().to_vec(),
                page_layout,
            ))
            .with_state_contract(state_contract_for_v1(manifest, kv_layout, served)?);
        // A selected checkpoint: the bundle's weight plans produce the rank
        // images and the components they declare, which replace any package
        // component of the same role.
        let selected = crate::direct::selected_checkpoint_v1()
            .map_err(|message| HostLoadErrorV1::Bundle(BundleError::validation(message)))?;
        let direct = selected.is_some();
        let planned = match selected {
            Some(checkpoint) => self.planned_images_v1(checkpoint, declared.rank_count())?,
            None => Vec::new(),
        };
        let planned_roles: Vec<&str> = planned.iter().map(|image| image.image_role()).collect();
        check_planned_components_v1(manifest, direct.then_some(&planned_roles[..]))
            .map_err(|message| HostLoadErrorV1::Bundle(BundleError::validation(message)))?;
        let replaced = |role: &[u8]| {
            planned.iter().any(|image| {
                image.image_role().as_bytes() == role
                    || image.components().iter().any(|c| c.name.as_bytes() == role)
            })
        };
        for component in manifest.components() {
            let role = component.role().as_str().as_bytes().to_vec();
            if replaced(&role) {
                continue;
            }
            let file = self
                .package
                .open_component_file(component)
                .map_err(|error| HostLoadErrorV1::ComponentRead {
                    path: component.path().as_str().to_owned(),
                    message: error.to_string(),
                })?;
            config = if role.starts_with(AUTHENTICATED_COMPONENT_ROLE_PREFIX_V1) {
                config.with_authenticated_component_file(role, file)
            } else if role.starts_with(BUNDLE_METADATA_ROLE_PREFIX_V1) {
                config.with_bundle_metadata_file(role, file)
            } else {
                config.with_component(role, file.bytes().to_vec())
            }
            .map_err(|error| BundleError::validation(error.to_string()))?;
        }
        for image in planned {
            config = config
                .with_planned_compiled_image(image)
                .map_err(|error| BundleError::validation(error.to_string()))?;
        }
        Ok(config)
    }

    /// Asks the bundle for every rank's weight plan over `checkpoint`, for
    /// each planned component, and returns the images it plans. A package
    /// that plans no component is asked once with an empty role, for the
    /// bundle's primary image. Each request carries the package's bundle
    /// metadata; it names no host views, because where a view lives is the
    /// plan's own decision.
    fn planned_images_v1(
        &self,
        checkpoint: std::sync::Arc<crate::direct::DirectCheckpointV1>,
        ranks: u32,
    ) -> Result<Vec<std::sync::Arc<crate::direct::PlannedImageV1>>, HostLoadErrorV1> {
        let manifest = self.manifest();
        let mut components = Vec::new();
        for component in manifest.components() {
            let role = component.role().as_str();
            if role.as_bytes().starts_with(BUNDLE_METADATA_ROLE_PREFIX_V1) {
                components.push(runtime_bundle_abi::weight_plan::NamedBytesV1 {
                    name: role.to_owned(),
                    bytes: verified_component_by_role_v1(&self.package, role)?,
                });
            }
        }
        components.sort_by(|left, right| left.name.cmp(&right.name));
        // Planned identities cover the module that runs bundle-owned
        // transforms; the package's verified manifest states its digest.
        let bundle_module = manifest
            .native_module()
            .map(|module| sha256_array_v1(module.sha256()))
            .transpose()
            .map_err(|message| HostLoadErrorV1::Bundle(BundleError::validation(message)))?
            .unwrap_or([0; 32]);
        // A planned component has no other source: the plan must supply it.
        let mut roles: Vec<String> = manifest
            .planned_components()
            .iter()
            .map(|component| component.role().as_str().to_owned())
            .collect();
        if roles.is_empty() {
            roles.push(String::new());
        }
        let mut planned = Vec::new();
        for role in roles {
            if let Some(image) = crate::direct::plan_image_v1(
                &self.library,
                bundle_module,
                checkpoint.clone(),
                &role,
                ranks,
                components.clone(),
                Vec::new(),
            )
            .map_err(|message| HostLoadErrorV1::Bundle(BundleError::validation(message)))?
            {
                planned.push(std::sync::Arc::new(image));
            }
        }
        if planned.is_empty() {
            return Err(HostLoadErrorV1::Bundle(BundleError::validation(
                "the bundle plans no rank image from the checkpoint",
            )));
        }
        Ok(planned)
    }

    /// Creates a served instance whose capable bundle can acquire the host pool.
    pub fn create_instance_with_serving_kv<D: HostDeviceBackendV1 + 'static>(
        &self,
        device: D,
        topology_id: &str,
        device_ordinals: Vec<i32>,
        serving_kv: Option<crate::services::ServingKvPolicyV1>,
    ) -> Result<(Rc<HostServicesV1Backend<D>>, LoadedBundleInstanceV1), HostLoadErrorV1> {
        self.create_instance_with_options(
            device,
            topology_id,
            device_ordinals,
            serving_kv,
            runtime_bundle_abi::BundleRawSpeculationV1::disabled(),
        )
    }

    pub fn create_instance_with_options<D: HostDeviceBackendV1 + 'static>(
        &self,
        device: D,
        topology_id: &str,
        device_ordinals: Vec<i32>,
        serving_kv: Option<crate::services::ServingKvPolicyV1>,
        speculation: runtime_bundle_abi::BundleRawSpeculationV1,
    ) -> Result<(Rc<HostServicesV1Backend<D>>, LoadedBundleInstanceV1), HostLoadErrorV1> {
        self.create_instance_with_serving_options(
            device,
            topology_id,
            device_ordinals,
            serving_kv,
            speculation,
            None,
            None,
        )
    }

    /// `maximum_sequences` names the live sequences served, at most the
    /// declared maximum; see [`served_sequences_v1`] for which packages are
    /// told. `kv_dtype` names the declared KV layout the instance runs,
    /// whether or not it acquires the serving pool; `None` runs the
    /// preferred `kv_layout`.
    #[allow(clippy::too_many_arguments)]
    pub fn create_instance_with_serving_options<D: HostDeviceBackendV1 + 'static>(
        &self,
        device: D,
        topology_id: &str,
        device_ordinals: Vec<i32>,
        serving_kv: Option<crate::services::ServingKvPolicyV1>,
        speculation: runtime_bundle_abi::BundleRawSpeculationV1,
        maximum_sequences: Option<u32>,
        kv_dtype: Option<runtime_bundle::BundleKvDtypeV1>,
    ) -> Result<(Rc<HostServicesV1Backend<D>>, LoadedBundleInstanceV1), HostLoadErrorV1> {
        self.create_served_instance(
            device,
            topology_id,
            device_ordinals,
            serving_kv,
            speculation,
            maximum_sequences,
            kv_dtype,
            1,
        )
    }

    /// [`Self::create_instance_with_serving_options`] with `context_parallel_size`
    /// ranks per context-parallel group, a size the caller checked against
    /// the topology's declaration.
    #[allow(clippy::too_many_arguments)]
    pub fn create_served_instance<D: HostDeviceBackendV1 + 'static>(
        &self,
        device: D,
        topology_id: &str,
        device_ordinals: Vec<i32>,
        serving_kv: Option<crate::services::ServingKvPolicyV1>,
        speculation: runtime_bundle_abi::BundleRawSpeculationV1,
        maximum_sequences: Option<u32>,
        kv_dtype: Option<runtime_bundle::BundleKvDtypeV1>,
        context_parallel_size: usize,
    ) -> Result<(Rc<HostServicesV1Backend<D>>, LoadedBundleInstanceV1), HostLoadErrorV1> {
        let mut config =
            self.instance_config_with_kv_dtype(topology_id, device_ordinals, kv_dtype)?;
        config = config
            .with_speculation(speculation)
            .with_context_parallel_size(context_parallel_size);
        if maximum_sequences.is_some()
            && let Some(sequences) = served_sequences_v1(
                self.manifest()
                    .resource_requirements()
                    .required_capabilities(),
                self.manifest().batch_limits(),
                maximum_sequences,
            )?
        {
            let kv_layout = served_kv_layout_v1(self.manifest(), kv_dtype)?;
            config = config
                .with_maximum_sequences(sequences)
                .with_state_contract(state_contract_for_v1(
                    self.manifest(),
                    kv_layout,
                    Some(sequences),
                )?);
        }
        if let Some(policy) = serving_kv {
            config = config.with_serving_kv(policy);
        }
        if self
            .manifest()
            .resource_requirements()
            .required_capabilities()
            .iter()
            .any(|id| id.as_str() == runtime_bundle_abi::SERVED_CONTEXT_CAPABILITY_V1)
        {
            if served_sequences_v1(
                self.manifest()
                    .resource_requirements()
                    .required_capabilities(),
                self.manifest().batch_limits(),
                maximum_sequences,
            )?
            .is_none()
            {
                return Err(HostLoadErrorV1::Bundle(BundleError::validation(
                    "served context requires served items and batch limits",
                )));
            }
            let declared = self.manifest().kv_layout().maximum_sequence_tokens();
            let tokens = serving_kv
                .and_then(|policy| policy.maximum_context_tokens)
                .map_or(declared, |tokens| tokens as u64);
            if tokens == 0 || tokens > declared {
                return Err(HostLoadErrorV1::Bundle(BundleError::validation(
                    "invalid served context",
                )));
            }
            config = config.with_maximum_context_tokens(tokens);
        }
        if let Some(nonce) = crate::services::tap_producer_v1() {
            config = config.with_tap_producer(nonce.to_owned());
        }
        let configuration = config.encode_v1(topology_id.as_bytes());
        let backend = Rc::new(HostServicesV1Backend::new(device, config));
        let table = HostServicesTableV1::new(Rc::clone(&backend));
        let selected = self.manifest().for_topology(topology_id)?;
        let preferred_kv =
            served_kv_layout_v1(&selected, kv_dtype)?.id() == selected.kv_layout().id();
        let instance = if crate::executor::device_results_selected(&selected, preferred_kv) {
            self.library
                .create_device_results_v1(table, &configuration)?
        } else {
            self.library.create(table, &configuration)?
        };
        Ok((backend, instance))
    }
}

/// Checks that what a package plans from a checkpoint has a source.
///
/// `planned_roles` are the roles of the images the selected checkpoint's
/// weight plans supply, or `None` when no checkpoint was selected. A planned
/// component has no file, so it is served only from a checkpoint and only
/// when the plan supplies it; a package that plans components, or that
/// requires the weight plan, therefore needs `--checkpoint DIR`. A package
/// with neither serves without a checkpoint and uploads no rank image.
pub fn check_planned_components_v1(
    manifest: &BundleManifestV1,
    planned_roles: Option<&[&str]>,
) -> Result<(), String> {
    let Some(planned_roles) = planned_roles else {
        if let Some(planned) = manifest.planned_components().first() {
            return Err(format!(
                "the package plans compiled image '{}' from a checkpoint and carries no file \
                 for it; serve it with --checkpoint DIR",
                planned.id().as_str()
            ));
        }
        if requires_weight_plan_v1(manifest) {
            return Err(
                "the package loads its weights through the weight plan; serve it with \
                 --checkpoint DIR"
                    .to_owned(),
            );
        }
        return Ok(());
    };
    for planned in manifest.planned_components() {
        let role = planned.role().as_str();
        if !planned_roles.contains(&role) {
            return Err(format!(
                "the bundle declined planned compiled image '{}' (role '{role}'), which has no \
                 other source",
                planned.id().as_str()
            ));
        }
    }
    Ok(())
}

/// A lowercase hex SHA-256 as its 32 bytes.
fn sha256_array_v1(hex: &str) -> Result<[u8; 32], String> {
    let digit = |byte: u8| match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        _ => None,
    };
    let bytes = hex.as_bytes();
    if bytes.len() != 64 {
        return Err(format!("{hex} is not a SHA-256 digest"));
    }
    let mut digest = [0; 32];
    for (index, pair) in bytes.chunks_exact(2).enumerate() {
        digest[index] = digit(pair[0])
            .zip(digit(pair[1]))
            .map(|(high, low)| high << 4 | low)
            .ok_or_else(|| format!("{hex} is not a SHA-256 digest"))?;
    }
    Ok(digest)
}

/// Whether a package requires the direct weight plan, so that it serves only
/// from a selected checkpoint.
pub fn requires_weight_plan_v1(manifest: &BundleManifestV1) -> bool {
    manifest
        .resource_requirements()
        .required_capabilities()
        .iter()
        .any(|capability| {
            capability.as_str() == runtime_bundle_abi::weight_plan::WEIGHT_PLAN_CAPABILITY_V1
        })
}

/// The live-sequence count a package's create record closes with, or None
/// when the record must not carry one.
///
/// Only a package that requires `chop.served-items.v1` and declares batch
/// limits is told: bundles built before the field existed refuse any byte
/// after the component roles, so everything else keeps the record it always
/// had. `requested` is `--max-sequences`, from 1 to the declared maximum;
/// None serves the declared maximum.
pub fn served_sequences_v1(
    required_capabilities: &[runtime_bundle::BundleOpaqueId],
    limits: Option<runtime_bundle::BundleBatchLimitsV1>,
    requested: Option<u32>,
) -> Result<Option<u32>, HostLoadErrorV1> {
    let reads_items = required_capabilities
        .iter()
        .any(|capability| capability.as_str() == runtime_bundle_abi::SERVED_ITEMS_CAPABILITY_V1);
    let Some(limits) = limits.filter(|_| reads_items) else {
        return Ok(None);
    };
    let maximum = limits.maximum_sequences();
    match requested {
        None => Ok(Some(maximum)),
        Some(requested) if (1..=maximum).contains(&requested) => Ok(Some(requested)),
        Some(requested) => Err(HostLoadErrorV1::Bundle(BundleError::validation(format!(
            "{requested} live sequences requested; the package serves 1 to {maximum}"
        )))),
    }
}

/// The state contract an instance serving `served` items is created with:
/// the declared layout, whose bytes a package requiring
/// `chop.state-per-item.v1` declares per state slot (see
/// [`runtime_bundle::BundleManifestV1::state_bytes_v1`]). Every other package
/// keeps its declared bytes, whatever it serves.
pub fn state_contract_v1(
    manifest: &BundleManifestV1,
    served: Option<u32>,
) -> Result<DeclaredStateContractV1, HostLoadErrorV1> {
    state_contract_for_v1(manifest, manifest.kv_layout(), served)
}

/// [`state_contract_v1`] of an instance serving `kv_layout`, one of the
/// manifest's KV layouts: a format may size the state for its element
/// ([`BundleManifestV1::state_layout_for_v1`]).
pub fn state_contract_for_v1(
    manifest: &BundleManifestV1,
    kv_layout: &runtime_bundle::BundleKvLayoutV1,
    served: Option<u32>,
) -> Result<DeclaredStateContractV1, HostLoadErrorV1> {
    let layout = manifest.state_layout_for_v1(kv_layout);
    Ok(DeclaredStateContractV1::new(
        layout.id().as_str().as_bytes().to_vec(),
        manifest
            .state_bytes_for_v1(kv_layout, served)
            .map_err(HostLoadErrorV1::Bundle)?,
        layout.alignment_bytes(),
    ))
}

/// The declared KV layout the host serves: the preferred one unless the
/// serving policy names another element.
fn served_kv_layout_v1(
    manifest: &BundleManifestV1,
    kv_dtype: Option<runtime_bundle::BundleKvDtypeV1>,
) -> Result<&runtime_bundle::BundleKvLayoutV1, HostLoadErrorV1> {
    match kv_dtype {
        None => Ok(manifest.kv_layout()),
        Some(dtype) => crate::kv_cache_dtype::kv_layout_of_v1(manifest, dtype).ok_or_else(|| {
            HostLoadErrorV1::Bundle(BundleError::validation(format!(
                "bundle declares no {} KV layout",
                dtype.as_str()
            )))
        }),
    }
}

/// Reads one verified component of a package, by the role it serves.
///
/// This is the only way a caller outside the host reaches component bytes, and
/// it reads them through the package that verified them. There is no second
/// manifest parser and no second lookup of the path.
pub fn verified_component_by_role_v1(
    package: &BundlePackageV1,
    role: &str,
) -> Result<Vec<u8>, HostLoadErrorV1> {
    let component = package
        .manifest()
        .components()
        .iter()
        .find(|component| component.role().as_str() == role)
        .ok_or_else(|| HostLoadErrorV1::UnknownComponentRole(role.to_owned()))?;
    package
        .read_component(component)
        .map_err(|error| HostLoadErrorV1::ComponentRead {
            path: component.path().as_str().to_owned(),
            message: error.to_string(),
        })
}

/// Which bundle this process selected, and which thread owns its library.
///
/// The selection is a process decision, so it lives in one process-wide record
/// rather than in per-thread state that a second thread could contradict. The
/// loaded host itself stays on its loading thread, because the native module
/// behind it is not `Send`.
struct ProcessSelectionV1 {
    selection: BundleSelectionV1,
    owner: ThreadId,
}

fn process_selection_v1() -> &'static Mutex<Option<ProcessSelectionV1>> {
    static SELECTION: OnceLock<Mutex<Option<ProcessSelectionV1>>> = OnceLock::new();
    SELECTION.get_or_init(|| Mutex::new(None))
}

thread_local! {
    /// The loaded host, on the one thread that owns its native module.
    ///
    /// The loaded host is not `Sync`, so the reference stays here while the
    /// decision to load it stays process-wide.
    static LOADED_BUNDLE_V1: Cell<Option<&'static InstalledBundleHostV1>> =
        const { Cell::new(None) };
}

/// Loads the one bundle this process will run, and keeps it resident.
///
/// A second call naming the same bundle, on the thread that loaded it, returns
/// the same loaded instance. A call naming a different bundle is refused
/// anywhere in the process: this cycle loads one bundle at process start and
/// never swaps it. A call naming the selected bundle from another thread is
/// refused too, because the loaded library stays on the thread that opened it.
///
/// This is the whole of process-start loading, and both selection forms take
/// it. The bundle is re-verified here, immediately before its native module is
/// opened, and the process is bound to that one selection afterwards.
pub fn load_selected_process_bundle_v1(
    requested: &BundleSelectionV1,
    host: &BundleHostCompatibilityV1,
) -> Result<&'static InstalledBundleHostV1, HostLoadErrorV1> {
    load_process_bundle_with_package_v1(requested, host, None, None, None)
}

/// [`load_selected_process_bundle_v1`] as the bundle serves `topology`, one
/// of the topologies it declares, with that topology's resources. A process
/// that already loaded the bundle for another topology refuses.
pub fn load_selected_topology_bundle_v1(
    requested: &BundleSelectionV1,
    host: &BundleHostCompatibilityV1,
    topology: &str,
) -> Result<&'static InstalledBundleHostV1, HostLoadErrorV1> {
    load_process_bundle_with_package_v1(requested, host, None, None, Some(topology))
}

/// Loads a package already verified for this serve, keeping its open component
/// handles instead of hashing every file again on the generation worker, and
/// the native module `library` when the caller already opened it
/// ([`open_native_module_v1`]).
pub(crate) fn load_verified_process_bundle_v1(
    requested: &BundleSelectionV1,
    host: &BundleHostCompatibilityV1,
    package: BundlePackageV1,
    library: Option<BundleLibraryV1>,
) -> Result<&'static InstalledBundleHostV1, HostLoadErrorV1> {
    load_process_bundle_with_package_v1(requested, host, Some(package), library, None)
}

fn load_process_bundle_with_package_v1(
    requested: &BundleSelectionV1,
    host: &BundleHostCompatibilityV1,
    verified: Option<BundlePackageV1>,
    library: Option<BundleLibraryV1>,
    topology: Option<&str>,
) -> Result<&'static InstalledBundleHostV1, HostLoadErrorV1> {
    // The lock is held across the load, so two threads racing to select cannot
    // both decide they are first.
    let mut selection = process_selection_v1()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if let Some(current) = selection.as_ref() {
        if &current.selection != requested {
            return Err(HostLoadErrorV1::AlreadyLoaded {
                loaded: current.selection.display(),
                requested: requested.display(),
            });
        }
        if current.owner != std::thread::current().id() {
            return Err(HostLoadErrorV1::LoadedOnAnotherThread {
                loaded: current.selection.display(),
            });
        }
        let loaded = LOADED_BUNDLE_V1.with(Cell::get).ok_or_else(|| {
            HostLoadErrorV1::LoadedOnAnotherThread {
                loaded: current.selection.display(),
            }
        })?;
        if let Some(topology) = topology
            && !matches!(loaded.manifest().target().topologies(), [only] if only.id().as_str() == topology)
        {
            return Err(HostLoadErrorV1::Bundle(BundleError::validation(format!(
                "this process loaded {} for another topology than {topology}",
                current.selection.display()
            ))));
        }
        return Ok(loaded);
    }
    let package = match verified {
        Some(package) => package,
        None => verify_selected_bundle_v1(requested, host)?,
    };
    let package = match topology {
        Some(topology) => package.for_topology(topology)?,
        None => package,
    };
    let loaded = open_package_v1(package, host, library)?;
    // The library, its API table, and the service tables its instances hand out
    // all outlive every instance, so the load is deliberately never undone.
    let loaded: &'static InstalledBundleHostV1 = Box::leak(Box::new(loaded));
    LOADED_BUNDLE_V1.with(|cell| cell.set(Some(loaded)));
    *selection = Some(ProcessSelectionV1 {
        selection: requested.clone(),
        owner: std::thread::current().id(),
    });
    Ok(loaded)
}

fn open_package_v1(
    package: BundlePackageV1,
    host: &BundleHostCompatibilityV1,
    library: Option<BundleLibraryV1>,
) -> Result<InstalledBundleHostV1, HostLoadErrorV1> {
    let library = match library {
        Some(library) => library,
        None => open_native_module_v1(&package, host)?,
    };
    let _defaults = package
        .manifest()
        .resolve_model_options(&std::collections::BTreeMap::new())
        .map_err(HostLoadErrorV1::Bundle)?;
    Ok(InstalledBundleHostV1 {
        package,
        library,
        model_options: std::cell::OnceCell::new(),
    })
}

/// Opens a verified package's native module.
///
/// A process loads it once ([`load_selected_process_bundle_v1`] and its
/// forms); a serve opens it before its deployment, whose frontend is the
/// module's export, and hands the same library to the generation worker.
pub(crate) fn open_native_module_v1(
    package: &BundlePackageV1,
    host: &BundleHostCompatibilityV1,
) -> Result<BundleLibraryV1, HostLoadErrorV1> {
    // Every declared operation is resolved next, while the bundle is still
    // only data. A bundle whose work this host cannot perform is refused here
    // rather than after its native code is mapped and running. Resolution is
    // for the architecture this host selected: implementations the bundle
    // declares for other architectures are not candidates for this process.
    let catalog = HostGenericCatalogV1::builtin_v1().map_err(HostLoadErrorV1::HostCatalog)?;
    resolve_declared_operations_for_architecture_v1(
        package.manifest(),
        &catalog,
        host.architecture(),
    )?;
    // The module's declared length, digest, and entry symbol are all checked
    // against one owned byte vector, and the loader is handed that vector
    // rather than the package path. It stages its own immutable copy, so
    // neither replacing the install-root entry nor rewriting the same inode
    // afterwards changes what this process loads.
    let module = package
        .open_native_module()?
        .ok_or(HostLoadErrorV1::NoNativeModule)?;
    let started = std::time::Instant::now();
    let library = BundleLibraryV1::open_verified_image(module.bytes(), module.display())?;
    eprintln!(
        "startup module load: {:.3}s",
        started.elapsed().as_secs_f64()
    );
    drop(module);
    Ok(library)
}

/// Resolves every operation the manifest's graph buckets reference, for one
/// exact selected architecture.
///
/// The resolver is the manifest's own, built with this host's real catalog
/// and filtered to the bundle implementations that declare this architecture,
/// so the tiers it visits are the ones the bundle's provisioning allows: an
/// operation the bundle keeps private is satisfied only by the bundle's own
/// declaration, and one it left to the host is satisfied only by an exact
/// generic candidate. A missing, near-miss, ambiguous, wrong-tier, or
/// bundle-private violation all end here, before a native module is opened.
/// An operation whose implementations all declare other architectures is
/// unsupported for this one; nothing declared for another architecture is
/// selected in its place.
pub fn resolve_declared_operations_for_architecture_v1(
    manifest: &BundleManifestV1,
    catalog: &HostGenericCatalogV1,
    architecture: &BundleOpaqueId,
) -> Result<Vec<ResolvedOperationV1>, HostLoadErrorV1> {
    let candidates = catalog.candidates().map_err(HostLoadErrorV1::HostCatalog)?;
    // A host candidate that claimed the bundle tier, or a catalog that is
    // ambiguous against this bundle's own declarations, is refused here.
    let resolver = manifest
        .kernel_resolver_for_architecture(architecture, candidates)
        .map_err(HostLoadErrorV1::Bundle)?;
    resolve_with_resolver_v1(manifest, catalog, &resolver, Some(architecture))
}

/// Resolves every operation the manifest's graph buckets reference, with no
/// architecture selected.
///
/// This is the manifest-level form of
/// [`resolve_declared_operations_for_architecture_v1`], defined only for a
/// manifest that declares one architecture, where the bundle's resolution is
/// unambiguous without one. A manifest that declares several architectures
/// must be resolved per architecture by a host that names the architecture it
/// is about to run; resolving it here would be the cross-architecture
/// fallback the bundle contract forbids.
pub fn resolve_declared_operations_v1(
    manifest: &BundleManifestV1,
    catalog: &HostGenericCatalogV1,
) -> Result<Vec<ResolvedOperationV1>, HostLoadErrorV1> {
    let architectures = manifest.target().architectures();
    let [architecture] = architectures else {
        let named = architectures
            .iter()
            .map(|architecture| architecture.as_str())
            .collect::<Vec<_>>()
            .join(", ");
        return Err(HostLoadErrorV1::Bundle(BundleError::validation(format!(
            "this form resolves a bundle that declares one architecture, and this bundle declares \
             {}: {named}; resolve one selected architecture with \
             resolve_declared_operations_for_architecture_v1",
            architectures.len()
        ))));
    };
    resolve_declared_operations_for_architecture_v1(manifest, catalog, architecture)
}

/// Resolves each declared operation once, against the resolver already built.
fn resolve_with_resolver_v1(
    manifest: &BundleManifestV1,
    catalog: &HostGenericCatalogV1,
    resolver: &KernelResolverV1,
    architecture: Option<&BundleOpaqueId>,
) -> Result<Vec<ResolvedOperationV1>, HostLoadErrorV1> {
    let semantic_keys = manifest.semantic_keys().map_err(HostLoadErrorV1::Bundle)?;
    let mut resolved = Vec::new();
    let mut seen = BTreeSet::new();
    for bucket in manifest.graph_buckets() {
        for key in bucket.operations() {
            if !seen.insert(key.clone()) {
                continue;
            }
            let requirement = manifest
                .operation_requirements()
                .iter()
                .find(|requirement| requirement.key() == key)
                .ok_or_else(|| unsupported_v1(key, "the bundle declares no requirement for it"))?;
            let semantic_key = semantic_keys
                .get(key)
                .ok_or_else(|| unsupported_v1(key, "its declared contract has no semantic key"))?;
            let resolution = resolver.resolve(semantic_key).map_err(|error| {
                unsupported_for_architecture_v1(manifest, key, &error.to_string(), architecture)
            })?;
            let implementation = resolution.implementation_id().clone();
            let tier = resolution.tier();
            check_provisioning_v1(
                manifest,
                catalog,
                key,
                requirement.provisioning(),
                tier,
                &implementation,
            )?;
            resolved.push(ResolvedOperationV1 {
                key: key.clone(),
                provisioning: requirement.provisioning().clone(),
                tier,
                implementation,
            });
        }
    }
    Ok(resolved)
}

/// Builds the refusal reason for an operation this host cannot run.
///
/// When the bundle declares implementations of the operation only for other
/// architectures, the reason names them. That is the failure a fat
/// multi-architecture bundle produces on a host whose architecture carries no
/// implementation, and it must read as exactly that: nothing declared for
/// another architecture may stand in for the selected one.
fn unsupported_for_architecture_v1(
    manifest: &BundleManifestV1,
    key: &BundleOperationKeyV1,
    reason: &str,
    architecture: Option<&BundleOpaqueId>,
) -> HostLoadErrorV1 {
    let Some(architecture) = architecture else {
        return unsupported_v1(key, reason);
    };
    let mut elsewhere: BTreeSet<&str> = BTreeSet::new();
    for implementation in manifest.implementations() {
        if implementation.key() == key
            && implementation.detail().target_architecture() != architecture
        {
            elsewhere.insert(implementation.detail().target_architecture().as_str());
        }
    }
    if elsewhere.is_empty() {
        return unsupported_v1(key, reason);
    }
    unsupported_v1(
        key,
        format!(
            "{reason} for the selected architecture {architecture}; the bundle declares \
             implementations of it for {}",
            Vec::from_iter(elsewhere).join(", ")
        ),
    )
}

/// Refuses a match that is not the one the bundle's provisioning asked for.
///
/// Exact semantic matching says the contract is the same. It does not say the
/// implementation came from where the bundle expected: a custom operation must
/// be satisfied by this bundle's own declaration, and one left to the host must
/// be satisfied by a candidate this host's catalog offers under the primitive
/// the bundle named.
fn check_provisioning_v1(
    manifest: &BundleManifestV1,
    catalog: &HostGenericCatalogV1,
    key: &BundleOperationKeyV1,
    provisioning: &BundleOperationProvisioningV1,
    tier: KernelImplementationTierV1,
    implementation: &KernelOpaqueIdV1,
) -> Result<(), HostLoadErrorV1> {
    match provisioning {
        BundleOperationProvisioningV1::BundleCustomOperation => {
            let declared_here = manifest
                .implementations()
                .iter()
                .any(|declared| declared.id().as_str() == implementation.as_str());
            if tier != KernelImplementationTierV1::Bundle || !declared_here {
                return Err(HostLoadErrorV1::WrongOperationProvider {
                    operation: key.operation().as_str().to_owned(),
                    variant: key.variant().as_str().to_owned(),
                    reason: format!(
                        "only this bundle may implement it, and {implementation} is a {tier} \
                         implementation the bundle does not declare"
                    ),
                });
            }
            Ok(())
        }
        BundleOperationProvisioningV1::HostGenericPrimitive(primitive) => {
            let offered_for = catalog.primitive_of(implementation);
            if tier != KernelImplementationTierV1::Generic
                || offered_for.is_none_or(|offered| offered.as_str() != primitive.as_str())
            {
                return Err(HostLoadErrorV1::WrongOperationProvider {
                    operation: key.operation().as_str().to_owned(),
                    variant: key.variant().as_str().to_owned(),
                    reason: format!(
                        "it asked for host primitive {primitive}, and {implementation} is a \
                         {tier} implementation this host does not offer under that primitive"
                    ),
                });
            }
            Ok(())
        }
    }
}

fn unsupported_v1(key: &BundleOperationKeyV1, reason: impl Into<String>) -> HostLoadErrorV1 {
    HostLoadErrorV1::UnsupportedOperation {
        operation: key.operation().as_str().to_owned(),
        variant: key.variant().as_str().to_owned(),
        reason: reason.into(),
    }
}

#[cfg(test)]
mod tests {
    use runtime_bundle::{
        BundleComponentV1, BundleGraphBucketV1, BundleHostCompatibilityV1, BundleKvLayoutV1,
        BundleManifestV1, BundleOpaqueId, BundleRankPlacementV1, BundleRelativePathV1,
        BundleResourceRequirementsV1, BundleStateLayoutV1, BundleStateLifetimeV1, BundleTargetV1,
        BundleTopologyV1,
    };

    use super::{
        BundleSelectionV1, HostLoadErrorV1, ProcessSelectionV1, load_selected_process_bundle_v1,
        process_selection_v1,
    };

    const SELECTED: &str = "fixture.selected.v1";
    const OTHER: &str = "fixture.other.v1";

    fn host() -> BundleHostCompatibilityV1 {
        BundleHostCompatibilityV1::new("sm_120f", 1).expect("the test host descriptor is valid")
    }

    fn directory(name: &str) -> BundleSelectionV1 {
        BundleSelectionV1::Directory(std::path::PathBuf::from("/nonexistent").join(name))
    }

    fn load(name: &str) -> Result<&'static super::InstalledBundleHostV1, HostLoadErrorV1> {
        load_selected_process_bundle_v1(&directory(name), &host())
    }

    /// One test owns the process selection, because there is only one.
    #[test]
    fn the_process_selection_binds_every_thread_and_not_only_the_loading_one() {
        // A refused load selects nothing, so the decision stays open.
        let error = load("fixture.absent.v1")
            .err()
            .expect("an absent directory cannot load");
        assert!(matches!(error, HostLoadErrorV1::Bundle(_)));
        {
            let mut selection = process_selection_v1()
                .lock()
                .expect("no other test holds the selection");
            assert!(
                selection.is_none(),
                "this process must not have selected a bundle yet"
            );
            *selection = Some(ProcessSelectionV1 {
                selection: directory(SELECTED),
                owner: std::thread::current().id(),
            });
        }

        // A second thread sees the same process decision, so it can neither
        // load a different bundle nor take over the selected one.
        let (different, same) = std::thread::spawn(|| {
            (
                matches!(load(OTHER), Err(HostLoadErrorV1::AlreadyLoaded { .. })),
                matches!(
                    load(SELECTED),
                    Err(HostLoadErrorV1::LoadedOnAnotherThread { .. })
                ),
            )
        })
        .join()
        .expect("the second thread finished");
        assert!(different, "a second thread must not select another bundle");
        assert!(same, "a second thread must not adopt the loaded bundle");

        // The selecting thread is bound by the same decision.
        assert!(matches!(
            load(OTHER),
            Err(HostLoadErrorV1::AlreadyLoaded { .. })
        ));

        *process_selection_v1()
            .lock()
            .expect("no other test holds the selection") = None;

        // With the selection cleared, a refused load again selects nothing.
        assert!(load(OTHER).is_err());
        assert!(
            process_selection_v1()
                .lock()
                .expect("no other test holds the selection")
                .is_none(),
            "a refused load must leave this process unselected"
        );
    }

    const DIGEST: &str = "0000000000000000000000000000000000000000000000000000000000000000";

    fn id(value: &str) -> BundleOpaqueId {
        BundleOpaqueId::new(value).expect("the test identifier is valid")
    }

    /// One component and two topologies: one rank for tp1, two ranks for tp2.
    fn two_topology_manifest() -> BundleManifestV1 {
        let component = BundleComponentV1::new(
            id("weights"),
            id("fixture.weights.v1"),
            id("fixture.bytes.v1"),
            BundleRelativePathV1::new("components/weights.bin").expect("the test path is relative"),
            16,
            DIGEST,
        )
        .expect("the test component is valid");
        BundleManifestV1::new(
            id("fixture.selection.v1"),
            BundleTargetV1::new(
                vec![id("sm_120f")],
                vec![
                    BundleTopologyV1::new(id("tp1"), 1).expect("tp1 is valid"),
                    BundleTopologyV1::new(id("tp2"), 2).expect("tp2 is valid"),
                ],
            )
            .expect("the test target is valid"),
            vec![component],
            BundleResourceRequirementsV1::new(1, Vec::new())
                .expect("the test requirements are valid"),
            BundleKvLayoutV1::new(
                id("fixture.kv.v1"),
                64,
                64,
                1,
                64,
                1,
                1024,
                BundleRankPlacementV1::Replicated,
                BundleStateLifetimeV1::Transaction,
            )
            .expect("the test KV layout is valid"),
            BundleStateLayoutV1::new(
                id("fixture.state.v1"),
                64,
                64,
                BundleStateLifetimeV1::Request,
            )
            .expect("the test state layout is valid"),
            vec![
                BundleGraphBucketV1::new(id("bucket.one"), vec![id("weights")], Vec::new())
                    .expect("the test bucket is valid"),
            ],
            Vec::new(),
            Vec::new(),
            None,
        )
        .expect("the test manifest is valid")
    }

    /// `chop.state-per-item.v1` scales the state contract (the create record's
    /// state bytes and the allocation `allocate_state` accepts) by the served
    /// count's state slots; every other package keeps its declared bytes.
    #[test]
    fn per_item_state_contracts_follow_the_served_count() {
        use crate::services::DeclaredStateContractV1;
        use runtime_bundle::BundleBatchLimitsV1;
        let contract =
            |bytes: u64| DeclaredStateContractV1::new(b"fixture.state.v1".to_vec(), bytes, 64);
        let limits = BundleBatchLimitsV1::new(64, 8192).expect("the test limits are valid");
        let old = two_topology_manifest()
            .with_batch_limits(limits)
            .expect("an old package takes batch limits");
        for served in [None, Some(1), Some(33), Some(64)] {
            assert_eq!(
                super::state_contract_v1(&old, served).unwrap(),
                contract(64)
            );
        }
        let requirements = |capabilities: &[&str]| {
            BundleResourceRequirementsV1::new(1, capabilities.iter().map(|c| id(c)).collect())
                .expect("the test requirements are valid")
        };
        let manifest = |capabilities: &[&str]| {
            let base = two_topology_manifest();
            BundleManifestV1::new(
                base.id().clone(),
                base.target().clone(),
                base.components().to_vec(),
                requirements(capabilities),
                base.kv_layout().clone(),
                base.state_layout().clone(),
                base.graph_buckets().to_vec(),
                Vec::new(),
                Vec::new(),
                None,
            )
        };
        assert!(manifest(&[runtime_bundle::STATE_PER_ITEM_CAPABILITY_V1]).is_err());
        let per_item = manifest(&[
            runtime_bundle::SERVED_ITEMS_CAPABILITY_V1,
            runtime_bundle::STATE_PER_ITEM_CAPABILITY_V1,
        ])
        .expect("a per-item package is valid");
        assert!(super::state_contract_v1(&per_item, Some(1)).is_err());
        let per_item = per_item.with_batch_limits(limits).unwrap();
        for (served, slots) in [(1, 1), (3, 4), (32, 32), (33, 64), (64, 64)] {
            assert_eq!(
                super::state_contract_v1(&per_item, Some(served)).unwrap(),
                contract(64 * slots)
            );
        }
        assert!(super::state_contract_v1(&per_item, None).is_err());
        assert!(super::state_contract_v1(&per_item, Some(65)).is_err());
        // The host and the bundle ABI name the same capability and slots.
        assert_eq!(
            runtime_bundle::STATE_PER_ITEM_CAPABILITY_V1,
            runtime_bundle_abi::STATE_PER_ITEM_CAPABILITY_V1
        );
        assert_eq!(
            runtime_bundle::SERVED_ITEMS_CAPABILITY_V1,
            runtime_bundle_abi::SERVED_ITEMS_CAPABILITY_V1
        );
        for served in 1..=256 {
            assert_eq!(
                runtime_bundle::state_slots_v1(served),
                runtime_bundle_abi::raw_state_slots_v1(served)
            );
        }
    }
    /// An instance serving a KV format that sizes the state for its element
    /// is created with that state, per served slot like the manifest's.
    #[test]
    fn the_served_kv_format_sizes_the_state_contract() {
        use crate::services::DeclaredStateContractV1;
        use runtime_bundle::{BundleBatchLimitsV1, BundleKvDtypeV1, BundleKvFormatV1};
        let base = two_topology_manifest();
        let requirements = BundleResourceRequirementsV1::new(
            1,
            vec![
                id(runtime_bundle::SERVED_ITEMS_CAPABILITY_V1),
                id(runtime_bundle::STATE_PER_ITEM_CAPABILITY_V1),
            ],
        )
        .expect("the test requirements are valid");
        let wide = BundleKvLayoutV1::new(
            id("fixture.kv-wide.v1"),
            128,
            64,
            1,
            128,
            1,
            1024,
            BundleRankPlacementV1::Replicated,
            BundleStateLifetimeV1::Transaction,
        )
        .expect("the wide KV layout is valid");
        let larger = BundleStateLayoutV1::new(
            id("fixture.state.v1"),
            128,
            64,
            BundleStateLifetimeV1::Request,
        )
        .expect("the larger state layout is valid");
        let manifest = BundleManifestV1::new(
            base.id().clone(),
            base.target().clone(),
            base.components().to_vec(),
            requirements,
            base.kv_layout().clone(),
            base.state_layout().clone(),
            base.graph_buckets().to_vec(),
            Vec::new(),
            Vec::new(),
            None,
        )
        .and_then(|m| m.with_batch_limits(BundleBatchLimitsV1::new(64, 8192)?))
        .and_then(|m| {
            m.with_kv_formats(vec![
                BundleKvFormatV1::new(BundleKvDtypeV1::Bf16, base.kv_layout().clone()),
                BundleKvFormatV1::new(BundleKvDtypeV1::Fp8E4m3, wide.clone())
                    .with_state_layout(larger)?,
            ])
        })
        .expect("the test manifest is valid");
        let served = |dtype| super::served_kv_layout_v1(&manifest, dtype).map(|l| l.id().clone());
        assert_eq!(served(None).unwrap(), id("fixture.kv.v1"));
        assert_eq!(
            served(Some(BundleKvDtypeV1::Fp8E4m3)).unwrap(),
            id("fixture.kv-wide.v1")
        );
        assert!(served(Some(BundleKvDtypeV1::MixedPlanes)).is_err());
        let contract =
            |bytes: u64| DeclaredStateContractV1::new(b"fixture.state.v1".to_vec(), bytes, 64);
        for (served, slots) in [(1, 1), (33, 64)] {
            assert_eq!(
                super::state_contract_for_v1(&manifest, &wide, Some(served)).unwrap(),
                contract(128 * slots)
            );
            assert_eq!(
                super::state_contract_v1(&manifest, Some(served)).unwrap(),
                contract(64 * slots)
            );
        }
    }
}
