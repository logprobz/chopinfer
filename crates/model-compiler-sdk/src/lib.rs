#![forbid(unsafe_code)]

//! The emitter every model compiler publishes its bundle through.
//!
//! A compiler describes one bundle as typed Rust records: components and the
//! bytes behind them, the components its weight plan supplies from a
//! checkpoint, the layouts and requirements its model declares, an optional
//! native module, and the ordered symbol sequences that module resolves. This crate stages those bytes into one directory,
//! measures every file it wrote, builds the one canonical
//! `chop-model-bundle-v1` manifest from those measurements, and publishes the
//! directory only after `runtime-bundle` has verified it as a closed package.
//!
//! `manifest.json` is the only metadata that reaches disk. There is no second
//! manifest, no envelope, and no side file: a record this crate cannot express
//! in the canonical manifest is a record it refuses to emit.
//!
//! Nothing here knows a model. Roles, formats, view identifiers, and paths are
//! opaque values the caller chose, and the crate links no kernel catalog, no
//! device toolchain, and no model crate.

use std::collections::BTreeSet;
use std::error::Error;
use std::fmt;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use runtime_bundle::{
    BUNDLE_MANIFEST_FILE, BUNDLE_NATIVE_ENTRY_SYMBOL_V1, BundleArtifactIdentityV1,
    BundleBatchLimitsV1, BundleBuildIdentityV1, BundleComponentV1, BundleError,
    BundleGraphBucketV1, BundleHostCompatibilityV1, BundleImplementationDetailV1,
    BundleImplementationTierV1, BundleImplementationV1, BundleKvFormatV1, BundleKvLayoutV1,
    BundleLaunchContractV1, BundleManifestV1, BundleNativeModuleV1, BundleNativeSymbolSequenceV1,
    BundleOpaqueId, BundleOperationKeyV1, BundleOperationRequirementV1, BundlePackageV1,
    BundlePlannedComponentV1, BundleRelativePathV1, BundleResourceRequirementsV1,
    BundleSourceIdentityV1, BundleSpecializationV1, BundleSpeculationV1, BundleStateLayoutV1,
    BundleTargetV1, BundleTopologyResourcesV1, BundleWorkspaceContractV1,
    MODEL_BUNDLE_V1_ABI_VERSION, publish_verified_staging_no_replace,
};
use runtime_weights::{Sha256State, sha256_reader};

pub mod deployment_policy;
pub mod device;
pub mod files;
pub mod geometry;
pub mod geometry_build;
pub mod metadata;
pub mod native_contract;
mod native_declarations;
pub mod package_cli;
pub mod package_draft;
pub mod preparation;
pub mod provenance;
pub mod quantization;
pub mod rank_index;
pub mod source;
pub mod source_file;
pub mod weight_plan;

/// The pinned identity an authenticated input is staged against.
///
/// Re-exported so a compiler names one type for the file it pins, whether it
/// received that identity from a manifest, a lock file, or its own earlier run.
pub use runtime_weights::FileIdentity;

/// The largest single file this emitter stages: the most one component may
/// declare, because package verification reads every component whole into
/// memory.
pub const MAX_STAGED_FILE_BYTES: u64 = runtime_bundle::MAX_VERIFIED_COMPONENT_BYTES;

/// The largest native module this emitter stages, matching the loader bound.
pub const MAX_STAGED_NATIVE_MODULE_BYTES: u64 = 512 * 1024 * 1024;

/// The largest number of files one emitted bundle may contain.
pub const MAX_STAGED_FILES: usize = 512;

const STAGING_ATTEMPTS: u64 = 32;

/// `O_NOFOLLOW`, which makes an open refuse a symlinked final component.
#[cfg(target_os = "linux")]
const NO_FOLLOW_FLAG: i32 = 0x2_0000;
#[cfg(target_os = "macos")]
const NO_FOLLOW_FLAG: i32 = 0x0100;

static NEXT_STAGING_DIRECTORY: AtomicU64 = AtomicU64::new(0);

/// Why one bundle was not emitted.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CompilerError(String);

impl CompilerError {
    #[must_use]
    pub fn validation(message: impl Into<String>) -> Self {
        Self(message.into())
    }

    fn io(action: &str, path: &Path, error: &io::Error) -> Self {
        Self(format!("cannot {action} {}: {error}", path.display()))
    }
}

impl fmt::Display for CompilerError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl Error for CompilerError {}

impl From<kernel_core::precision::PrecisionError> for CompilerError {
    fn from(error: kernel_core::precision::PrecisionError) -> Self {
        Self::validation(error.to_string())
    }
}

impl From<BundleError> for CompilerError {
    fn from(error: BundleError) -> Self {
        Self(error.to_string())
    }
}

/// Where one staged file's bytes come from.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum StagedContentV1 {
    /// Bytes the compiler already holds, such as a plan it just rendered.
    Bytes(Vec<u8>),
    /// One regular file the compiler produced elsewhere.
    File(PathBuf),
    /// One regular file the compiler did not produce, staged only if it is
    /// exactly the file the pinned identity names.
    ///
    /// The emitter trusts the identity and nothing else about the path: it
    /// checks the name and length before staging and authenticates the source
    /// against the pinned digest over the same read that copies it. A file
    /// that is not the pinned one fails the emission, so it is never described
    /// by a manifest and never published.
    AuthenticatedFile {
        path: PathBuf,
        identity: FileIdentity,
    },
}

impl StagedContentV1 {
    /// Stages one file the compiler produced on disk.
    pub fn file(path: impl Into<PathBuf>) -> Self {
        Self::File(path.into())
    }

    /// Stages bytes the compiler rendered in memory.
    pub fn bytes(bytes: impl Into<Vec<u8>>) -> Self {
        Self::Bytes(bytes.into())
    }

    /// Stages one file the compiler did not produce, against a pinned identity.
    ///
    /// Use this for every input whose bytes came from somewhere the compiler
    /// does not control. The pinned name, length, and digest are checked before
    /// publication.
    pub fn authenticated_file(path: impl Into<PathBuf>, identity: FileIdentity) -> Self {
        Self::AuthenticatedFile {
            path: path.into(),
            identity,
        }
    }
}

/// What one component's file holds.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ComponentContentV1 {
    /// One opaque file: a graph plan, a compiled module, tokenizer data,
    /// anything a model declares a role and a format for.
    Whole(StagedContentV1),
    /// A compiled image the bundle's weight plan supplies: declared by role
    /// and format, never staged (`runtime_bundle::BundlePlannedComponentV1`).
    Planned,
}

/// One component of the bundle being emitted.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ComponentDraftV1 {
    id: BundleOpaqueId,
    role: BundleOpaqueId,
    format: BundleOpaqueId,
    path: BundleRelativePathV1,
    content: ComponentContentV1,
    topologies: Vec<BundleOpaqueId>,
}

impl ComponentDraftV1 {
    #[must_use]
    pub const fn new(
        id: BundleOpaqueId,
        role: BundleOpaqueId,
        format: BundleOpaqueId,
        path: BundleRelativePathV1,
        content: ComponentContentV1,
    ) -> Self {
        Self {
            id,
            role,
            format,
            path,
            content,
            topologies: Vec::new(),
        }
    }

    /// This component serving only `topologies` of the bundle
    /// ([`BundleComponentV1::with_topologies`]); planned components serve
    /// every topology.
    #[must_use]
    pub fn with_topologies(mut self, topologies: Vec<BundleOpaqueId>) -> Self {
        self.topologies = topologies;
        self
    }

    fn planned_declaration(&self) -> Result<BundlePlannedComponentV1, BundleError> {
        if !self.topologies.is_empty() {
            return Err(BundleError::validation(format!(
                "planned component {} serves every topology",
                self.id
            )));
        }
        BundlePlannedComponentV1::new(self.id.clone(), self.role.clone(), self.format.clone())
    }

    /// The component as the manifest declares it, from its measured file.
    fn declared(&self, byte_length: u64, sha256: &str) -> Result<BundleComponentV1, BundleError> {
        BundleComponentV1::new(
            self.id.clone(),
            self.role.clone(),
            self.format.clone(),
            self.path.clone(),
            byte_length,
            sha256,
        )?
        .with_topologies(self.topologies.clone())
    }

    /// A planned component: a compiled image the weight plan supplies, with
    /// no file. Its path is a placeholder that is never staged or declared.
    pub fn planned(
        id: BundleOpaqueId,
        role: BundleOpaqueId,
        format: BundleOpaqueId,
    ) -> Result<Self, BundleError> {
        let path = BundleRelativePathV1::new(format!("planned/{id}"))?;
        Ok(Self::new(
            id,
            role,
            format,
            path,
            ComponentContentV1::Planned,
        ))
    }

    #[must_use]
    pub const fn id(&self) -> &BundleOpaqueId {
        &self.id
    }

    /// The role this component is resolved by.
    #[must_use]
    pub const fn role(&self) -> &BundleOpaqueId {
        &self.role
    }

    /// The format this component declares its bytes to be in.
    #[must_use]
    pub const fn format(&self) -> &BundleOpaqueId {
        &self.format
    }

    /// The path inside the package this component is staged at.
    #[must_use]
    pub const fn path(&self) -> &BundleRelativePathV1 {
        &self.path
    }
}

/// The optional native module of the bundle being emitted.
///
/// The entry symbol is not a parameter: a V1 bundle exports exactly
/// [`BUNDLE_NATIVE_ENTRY_SYMBOL_V1`], and the emitted manifest says so.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NativeModuleDraftV1 {
    path: BundleRelativePathV1,
    content: StagedContentV1,
}

impl NativeModuleDraftV1 {
    #[must_use]
    pub const fn new(path: BundleRelativePathV1, content: StagedContentV1) -> Self {
        Self { path, content }
    }
}

/// Which staged file an implementation's artifact will be.
///
/// The emitter measures the file, so this names it and nothing else. A
/// component-backed implementation names the component that carries it; a
/// native-backed one names no file at all, because a package has exactly one
/// native module and the emitter already staged it.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ImplementationArtifactDraftV1 {
    /// The declared component that carries this implementation.
    Component(BundleOpaqueId),
    /// This package's own native module, which carries the executable code.
    NativeModule,
}

/// One implementation this bundle provides for one required operation.
///
/// The artifact identity is not a parameter either. The implementation names
/// which staged file carries it, and the emitter fills the format, length, and
/// digest in from the file it staged, so a declared implementation can never
/// point outside the closed file set or at bytes that were never written.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ImplementationDraftV1 {
    id: BundleOpaqueId,
    key: BundleOperationKeyV1,
    artifact: ImplementationArtifactDraftV1,
    symbol: BundleOpaqueId,
    target_architecture: BundleOpaqueId,
    launch: BundleLaunchContractV1,
    workspace: Option<BundleWorkspaceContractV1>,
    specializations: Vec<BundleSpecializationV1>,
    source: BundleSourceIdentityV1,
    build: BundleBuildIdentityV1,
}

impl ImplementationDraftV1 {
    #[allow(clippy::too_many_arguments)]
    #[must_use]
    pub const fn new(
        id: BundleOpaqueId,
        key: BundleOperationKeyV1,
        artifact: ImplementationArtifactDraftV1,
        symbol: BundleOpaqueId,
        target_architecture: BundleOpaqueId,
        launch: BundleLaunchContractV1,
        workspace: Option<BundleWorkspaceContractV1>,
        specializations: Vec<BundleSpecializationV1>,
        source: BundleSourceIdentityV1,
        build: BundleBuildIdentityV1,
    ) -> Self {
        Self {
            id,
            key,
            artifact,
            symbol,
            target_architecture,
            launch,
            workspace,
            specializations,
            source,
            build,
        }
    }

    /// Which staged file this implementation will be measured from.
    #[must_use]
    pub const fn artifact(&self) -> &ImplementationArtifactDraftV1 {
        &self.artifact
    }

    /// The operation this implementation is declared for.
    ///
    /// A manifest refuses an implementation whose operation it declares no
    /// requirement for, so a compiler that assembles both lists separately needs
    /// to read this rather than track the key beside the draft.
    #[must_use]
    pub const fn key(&self) -> &BundleOperationKeyV1 {
        &self.key
    }

    /// The identifier this implementation is declared under.
    ///
    /// One bundle declares an identifier once, so a compiler that builds several
    /// declarations from one kernel identity reads this to check that it did.
    #[must_use]
    pub const fn id(&self) -> &BundleOpaqueId {
        &self.id
    }
}

/// One complete description of a bundle to emit.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BundleDraftV1 {
    id: BundleOpaqueId,
    target: BundleTargetV1,
    resource_requirements: BundleResourceRequirementsV1,
    kv_layout: BundleKvLayoutV1,
    kv_formats: Vec<BundleKvFormatV1>,
    state_layout: BundleStateLayoutV1,
    recurrent_state_layouts: Vec<BundleStateLayoutV1>,
    components: Vec<ComponentDraftV1>,
    graph_buckets: Vec<BundleGraphBucketV1>,
    operation_requirements: Vec<BundleOperationRequirementV1>,
    implementations: Vec<ImplementationDraftV1>,
    native_module: Option<NativeModuleDraftV1>,
    native_symbol_sequences: Vec<BundleNativeSymbolSequenceV1>,
    speculation: Option<BundleSpeculationV1>,
    model_options: Option<runtime_bundle::model_options::ModelOptionsV1>,
    batch_limits: Option<BundleBatchLimitsV1>,
    topology_resources: Vec<BundleTopologyResourcesV1>,
}

impl BundleDraftV1 {
    #[must_use]
    pub const fn new(
        id: BundleOpaqueId,
        target: BundleTargetV1,
        resource_requirements: BundleResourceRequirementsV1,
        kv_layout: BundleKvLayoutV1,
        state_layout: BundleStateLayoutV1,
    ) -> Self {
        Self {
            id,
            target,
            resource_requirements,
            kv_layout,
            kv_formats: Vec::new(),
            state_layout,
            recurrent_state_layouts: Vec::new(),
            components: Vec::new(),
            graph_buckets: Vec::new(),
            operation_requirements: Vec::new(),
            implementations: Vec::new(),
            native_module: None,
            native_symbol_sequences: Vec::new(),
            speculation: None,
            model_options: None,
            batch_limits: None,
            topology_resources: Vec::new(),
        }
    }

    /// One draft that serves every topology of `drafts`, drafts of one
    /// bundle for one topology each that differ only in that topology and
    /// its resources. The draft of the first topology in ID order keeps its
    /// resources as the bundle's own; each other becomes that topology's
    /// resources.
    pub fn for_topologies(mut drafts: Vec<Self>) -> Result<Self, CompilerError> {
        let topology = |draft: &Self| match draft.target.topologies() {
            [topology] if draft.topology_resources.is_empty() => Ok(topology.clone()),
            _ => Err(CompilerError::validation(
                "a draft joined with others serves one topology".to_owned(),
            )),
        };
        drafts.sort_by(|left, right| {
            let id = |draft: &Self| draft.target.topologies().first().map(|t| t.id().clone());
            id(left).cmp(&id(right))
        });
        let mut drafts = drafts.into_iter();
        // Components of some topologies join as they are; the rest are the
        // same in every draft.
        let restricted = |draft: &mut Self| {
            let (restricted, every): (Vec<_>, Vec<_>) = std::mem::take(&mut draft.components)
                .into_iter()
                .partition(|component| !component.topologies.is_empty());
            draft.components = every;
            restricted
        };
        let mut first = drafts
            .next()
            .ok_or_else(|| CompilerError::validation("no draft to join".to_owned()))?;
        let mut topologies = vec![topology(&first)?];
        let mut components = restricted(&mut first);
        for mut draft in drafts {
            topologies.push(topology(&draft)?);
            components.extend(restricted(&mut draft));
            let resources = BundleTopologyResourcesV1::new(
                topologies[topologies.len() - 1].id().clone(),
                draft.resource_requirements.clone(),
                draft.kv_layout.clone(),
                draft.kv_formats.clone(),
                draft.state_layout.clone(),
                draft.recurrent_state_layouts.clone(),
                draft.batch_limits,
                draft.graph_buckets.clone(),
            );
            let rest = Self {
                target: first.target.clone(),
                resource_requirements: first.resource_requirements.clone(),
                kv_layout: first.kv_layout.clone(),
                kv_formats: first.kv_formats.clone(),
                state_layout: first.state_layout.clone(),
                recurrent_state_layouts: first.recurrent_state_layouts.clone(),
                batch_limits: first.batch_limits,
                graph_buckets: first.graph_buckets.clone(),
                topology_resources: first.topology_resources.clone(),
                ..draft
            };
            if rest != first {
                return Err(CompilerError::validation(format!(
                    "the draft for topology {} differs from the first beyond its resources",
                    resources.topology()
                )));
            }
            first.topology_resources.push(resources);
        }
        first.target = BundleTargetV1::new(first.target.architectures().to_vec(), topologies)?;
        first.components.extend(components);
        Ok(first)
    }

    /// Adds an option declaration and its capability. Apply to each topology
    /// draft before joining; every topology must carry the same declaration.
    pub fn with_model_options(
        mut self,
        options: runtime_bundle::model_options::ModelOptionsV1,
    ) -> Result<Self, BundleError> {
        options.validate()?;
        if !self.topology_resources.is_empty() {
            return Err(BundleError::validation(
                "declare model options before joining topology drafts",
            ));
        }
        let mut capabilities = self.resource_requirements.required_capabilities().to_vec();
        let capability = BundleOpaqueId::new(runtime_bundle::model_options::CAPABILITY)?;
        if !capabilities.contains(&capability) {
            capabilities.push(capability);
        }
        self.resource_requirements = BundleResourceRequirementsV1::new(
            self.resource_requirements.minimum_device_memory_bytes(),
            capabilities,
        )?;
        self.model_options = Some(options);
        Ok(self)
    }

    /// Every serving KV layout the bundle can run, the preferred `kv_layout`
    /// among them; the host picks one at startup.
    #[must_use]
    pub fn with_kv_formats(mut self, formats: Vec<BundleKvFormatV1>) -> Self {
        self.kv_formats = formats;
        self
    }

    #[must_use]
    pub fn with_recurrent_state_layouts(mut self, layouts: Vec<BundleStateLayoutV1>) -> Self {
        self.recurrent_state_layouts = layouts;
        self
    }

    #[must_use]
    pub fn with_resource_requirements(
        mut self,
        resource_requirements: BundleResourceRequirementsV1,
    ) -> Self {
        self.resource_requirements = resource_requirements;
        self
    }

    #[must_use]
    pub fn with_components(mut self, components: Vec<ComponentDraftV1>) -> Self {
        self.components = components;
        self
    }

    #[must_use]
    pub fn with_graph_buckets(mut self, graph_buckets: Vec<BundleGraphBucketV1>) -> Self {
        self.graph_buckets = graph_buckets;
        self
    }

    #[must_use]
    pub fn with_operation_requirements(
        mut self,
        operation_requirements: Vec<BundleOperationRequirementV1>,
    ) -> Self {
        self.operation_requirements = operation_requirements;
        self
    }

    #[must_use]
    pub fn with_implementations(mut self, implementations: Vec<ImplementationDraftV1>) -> Self {
        self.implementations = implementations;
        self
    }

    #[must_use]
    pub fn with_native_module(mut self, native_module: NativeModuleDraftV1) -> Self {
        self.native_module = Some(native_module);
        self
    }

    /// Declares the ordered symbol sequences the emitted manifest carries.
    ///
    /// The emitter measures nothing here: a sequence names symbols, not files,
    /// so what the compiler declares is what the manifest states. What the
    /// emitter does own is the order the sequences are written in, which the
    /// manifest sorts by identifier.
    #[must_use]
    pub fn with_native_symbol_sequences(
        mut self,
        native_symbol_sequences: Vec<BundleNativeSymbolSequenceV1>,
    ) -> Self {
        self.native_symbol_sequences = native_symbol_sequences;
        self
    }

    #[must_use]
    pub fn with_speculation(mut self, speculation: BundleSpeculationV1) -> Self {
        self.speculation = Some(speculation);
        self
    }

    #[must_use]
    pub const fn with_batch_limits(mut self, limits: BundleBatchLimitsV1) -> Self {
        self.batch_limits = Some(limits);
        self
    }

    #[must_use]
    pub fn components(&self) -> &[ComponentDraftV1] {
        &self.components
    }

    /// Checks everything that can be known before a byte is written.
    ///
    /// Duplicate identity is checked here rather than left to the manifest,
    /// because two components sharing a path would otherwise be discovered only
    /// after one of them had been written over the other.
    fn validate(&self) -> Result<(), CompilerError> {
        let file_count = self.components.len()
            - self
                .components
                .iter()
                .filter(|component| matches!(component.content, ComponentContentV1::Planned))
                .count()
            + usize::from(self.native_module.is_some())
            + 1;
        if file_count > MAX_STAGED_FILES {
            return Err(CompilerError::validation(format!(
                "bundle {} declares {file_count} files, the emitter stages at most \
                 {MAX_STAGED_FILES}",
                self.id
            )));
        }
        let mut ids = BTreeSet::new();
        let mut planned = BTreeSet::new();
        let mut roles = BTreeSet::new();
        let mut paths = BTreeSet::new();
        paths.insert(BUNDLE_MANIFEST_FILE.to_owned());
        for component in &self.components {
            if !ids.insert(component.id.as_str()) {
                return Err(CompilerError::validation(format!(
                    "bundle {} declares component {} more than once",
                    self.id, component.id
                )));
            }
            // Components of some topologies share a role with those of
            // others; the manifest checks each topology's.
            if component.topologies.is_empty() && !roles.insert(component.role.as_str()) {
                return Err(CompilerError::validation(format!(
                    "bundle {} declares role {} on more than one component",
                    self.id, component.role
                )));
            }
            if !paths.insert(component.path.as_str().to_owned()) {
                return Err(CompilerError::validation(format!(
                    "bundle {} stages more than one file at {}",
                    self.id, component.path
                )));
            }
            if matches!(component.content, ComponentContentV1::Planned) {
                planned.insert(component.id.as_str());
            }
        }
        if let Some(native_module) = &self.native_module
            && !paths.insert(native_module.path.as_str().to_owned())
        {
            return Err(CompilerError::validation(format!(
                "bundle {} stages more than one file at {}",
                self.id, native_module.path
            )));
        }
        let mut implementations = BTreeSet::new();
        for implementation in &self.implementations {
            if !implementations.insert(implementation.id.as_str()) {
                return Err(CompilerError::validation(format!(
                    "bundle {} declares implementation {} more than once",
                    self.id, implementation.id
                )));
            }
            match &implementation.artifact {
                ImplementationArtifactDraftV1::Component(component) => {
                    if !ids.contains(component.as_str()) || planned.contains(component.as_str()) {
                        return Err(CompilerError::validation(format!(
                            "bundle {} implementation {} names artifact component {component}, \
                             which it does not declare",
                            self.id, implementation.id
                        )));
                    }
                }
                ImplementationArtifactDraftV1::NativeModule => {
                    if self.native_module.is_none() {
                        return Err(CompilerError::validation(format!(
                            "bundle {} implementation {} is backed by a native module, which it \
                             does not declare",
                            self.id, implementation.id
                        )));
                    }
                }
            }
        }
        // A sequence names symbols the package's own module resolves, so a
        // draft that stages no module has nothing to resolve them from. The
        // manifest refuses this too; refusing it here keeps it a declaration
        // the emitter never opened a file for.
        if !self.native_symbol_sequences.is_empty() && self.native_module.is_none() {
            return Err(CompilerError::validation(format!(
                "bundle {} declares {} native symbol sequences and no native module",
                self.id,
                self.native_symbol_sequences.len()
            )));
        }
        Ok(())
    }
}

/// One emitted bundle directory, as the canonical verifier reads it back.
#[derive(Clone, Debug)]
pub struct EmittedBundleV1 {
    package: BundlePackageV1,
    pub measurements: EmissionMeasurements,
}

/// How long staging and the canonical verification of the result took.
#[derive(Clone, Debug, Default)]
pub struct EmissionMeasurements {
    pub assembly_seconds: f64,
    pub verification_seconds: f64,
}

// Artifact equality excludes the observations made while producing it.
impl PartialEq for EmittedBundleV1 {
    fn eq(&self, other: &Self) -> bool {
        self.package == other.package
    }
}
impl Eq for EmittedBundleV1 {}

impl EmittedBundleV1 {
    #[must_use]
    pub fn root(&self) -> &Path {
        self.package.root()
    }

    #[must_use]
    pub const fn manifest(&self) -> &BundleManifestV1 {
        self.package.manifest()
    }

    #[must_use]
    pub const fn package(&self) -> &BundlePackageV1 {
        &self.package
    }
}

/// Authenticates every declared input and computes the exact manifest without writing.
///
/// This follows the same limits, digest, implementation, planned-component, and
/// symbol-sequence paths as emission.
pub fn plan_bundle_manifest_v1(draft: &BundleDraftV1) -> Result<BundleManifestV1, CompilerError> {
    draft.validate()?;
    let mut components = Vec::with_capacity(draft.components.len());
    let mut planned = Vec::new();
    let mut measured_artifacts = Vec::with_capacity(draft.components.len());
    for component in &draft.components {
        let content = match &component.content {
            ComponentContentV1::Whole(content) => content,
            ComponentContentV1::Planned => {
                planned.push(component.planned_declaration()?);
                continue;
            }
        };
        let measured = measure_file(
            &component.path,
            vec![PlannedPartV1::content(content)?],
            MAX_STAGED_FILE_BYTES,
        )?;
        components.push(component.declared(measured.byte_length, &measured.sha256)?);
        measured_artifacts.push((component.id.clone(), component.format.clone(), measured));
    }
    let native_module = match &draft.native_module {
        Some(module) => {
            let measured = measure_file(
                &module.path,
                vec![PlannedPartV1::content(&module.content)?],
                MAX_STAGED_NATIVE_MODULE_BYTES,
            )?;
            Some(BundleNativeModuleV1::new(
                module.path.clone(),
                measured.byte_length,
                measured.sha256,
                BUNDLE_NATIVE_ENTRY_SYMBOL_V1,
            )?)
        }
        None => None,
    };
    let implementations = draft
        .implementations
        .iter()
        .map(|implementation| implementation.declare(&measured_artifacts, native_module.as_ref()))
        .collect::<Result<Vec<_>, _>>()?;
    manifest_v1(draft, components, planned, implementations, native_module)
}

/// Emits one canonical `chop-model-bundle-v1` directory at `output`.
///
/// The bytes are staged beside `output` first, measured as they are written,
/// described by one manifest built from those measurements, and verified as a
/// closed package before anything is published. `output` must not exist:
/// publication reserves the directory and links each file into it, so an
/// emitted bundle never replaces an installed one or merges with a partial one.
///
/// What a failure leaves behind depends on where it happened. A failure before
/// publication removes the staging directory and leaves `output` absent. A
/// failure during publication leaves the reserved directory, because removing a
/// directory this call no longer has exclusive knowledge of could destroy what
/// another party put in it. What that directory never gets is `manifest.json`,
/// which is linked last as the commit marker: a package without it is refused
/// by every verifier and loader, so no failed publication leaves something a
/// host will load. Retrying into the same path is refused until an operator
/// removes the partial directory, which is the intended outcome rather than a
/// silent overwrite.
pub fn emit_bundle_directory_v1(
    draft: &BundleDraftV1,
    output: &Path,
) -> Result<EmittedBundleV1, CompilerError> {
    draft.validate()?;
    let parent = match output.parent() {
        Some(parent) if !parent.as_os_str().is_empty() => parent,
        _ => Path::new("."),
    };
    validate_directory(parent, "bundle output parent")?;
    if path_exists(output)? {
        return Err(CompilerError::validation(format!(
            "refusing to replace the existing bundle output {}",
            output.display()
        )));
    }
    let staging = create_staging_directory(parent, output)?;
    match emit_into_staging(draft, &staging, output) {
        Ok((host, mut measurements)) => {
            // Every published file is a link to a staged one, so the staging
            // copy holds no bytes of its own and its removal cannot damage what
            // was verified at `output`.
            let _ = fs::remove_dir_all(&staging);
            let started = std::time::Instant::now();
            let package = BundlePackageV1::load_directory(output, &host)?;
            measurements.verification_seconds += started.elapsed().as_secs_f64();
            Ok(EmittedBundleV1 {
                package,
                measurements,
            })
        }
        Err(error) => match fs::remove_dir_all(&staging) {
            Ok(()) => Err(error),
            Err(cleanup) => Err(CompilerError::validation(format!(
                "bundle emission failed ({error}) and the staging directory {} could not be \
                 removed ({cleanup})",
                staging.display()
            ))),
        },
    }
}

fn emit_into_staging(
    draft: &BundleDraftV1,
    staging: &Path,
    output: &Path,
) -> Result<(BundleHostCompatibilityV1, EmissionMeasurements), CompilerError> {
    let started = std::time::Instant::now();
    let staged = stage_bundle_v1(draft, staging)?;
    let mut measurements = EmissionMeasurements {
        assembly_seconds: started.elapsed().as_secs_f64(),
        ..Default::default()
    };

    // The emitter believes nothing it computed. What it publishes is what the
    // canonical verifier accepted from disk, digest by digest, file set and all.
    let host = emitter_host(&staged.manifest)?;
    let started = std::time::Instant::now();
    BundlePackageV1::verify_directory(staging, &host)?;
    measurements.verification_seconds += started.elapsed().as_secs_f64();
    publish_verified_staging_no_replace(staging, output, &staged.manifest)?;
    Ok((host, measurements))
}

/// One staged bundle: the manifest it describes.
struct StagedBundleV1 {
    manifest: BundleManifestV1,
}

/// Integrity verification without executing a module. Host admission still runs
/// separately against the actual serving host's capabilities.
pub fn verify_emitted_package(output: &Path) -> Result<BundlePackageV1, CompilerError> {
    let bytes = runtime_weights::checkpoint::CheckpointFile::open(output, BUNDLE_MANIFEST_FILE)
        .and_then(|f| f.read_bounded(runtime_bundle::BUNDLE_MANIFEST_MAX_BYTES as u64))
        .map_err(|e| CompilerError::validation(e.to_string()))?;
    let manifest = BundleManifestV1::from_canonical_json(&bytes)?;
    Ok(BundlePackageV1::verify_directory(
        output,
        &emitter_host(&manifest)?,
    )?)
}

/// Writes every staged file, then builds the manifest from what was written.
///
/// Writes every component and builds the manifest from the staged bytes.
fn stage_bundle_v1(draft: &BundleDraftV1, staging: &Path) -> Result<StagedBundleV1, CompilerError> {
    let mut components = Vec::with_capacity(draft.components.len());
    let mut planned = Vec::new();
    let mut staged_artifacts = Vec::with_capacity(draft.components.len());
    for component in &draft.components {
        let content = match &component.content {
            ComponentContentV1::Whole(content) => content,
            ComponentContentV1::Planned => {
                planned.push(component.planned_declaration()?);
                continue;
            }
        };
        let staged = stage_file(
            staging,
            &component.path,
            vec![PlannedPartV1::content(content)?],
            MAX_STAGED_FILE_BYTES,
        )?;
        components.push(component.declared(staged.byte_length, &staged.sha256)?);
        staged_artifacts.push((component.id.clone(), component.format.clone(), staged));
    }

    let native_module = match &draft.native_module {
        Some(native_module) => {
            let staged = stage_file(
                staging,
                &native_module.path,
                vec![PlannedPartV1::content(&native_module.content)?],
                MAX_STAGED_NATIVE_MODULE_BYTES,
            )?;
            Some(BundleNativeModuleV1::new(
                native_module.path.clone(),
                staged.byte_length,
                staged.sha256,
                BUNDLE_NATIVE_ENTRY_SYMBOL_V1,
            )?)
        }
        None => None,
    };

    let mut implementations = Vec::with_capacity(draft.implementations.len());
    for implementation in &draft.implementations {
        implementations.push(implementation.declare(&staged_artifacts, native_module.as_ref())?);
    }

    let manifest = manifest_v1(draft, components, planned, implementations, native_module)?;
    let manifest_path = BundleRelativePathV1::new(BUNDLE_MANIFEST_FILE)?;
    stage_file(
        staging,
        &manifest_path,
        vec![PlannedPartV1::bytes(manifest.to_canonical_json()?)],
        MAX_STAGED_FILE_BYTES,
    )?;
    sync_directory(staging)?;
    Ok(StagedBundleV1 { manifest })
}

/// The manifest of `draft` with its declared files. A draft of several
/// topologies is declared for its first first, with the components that
/// serve it, then joined by the others with every component.
fn manifest_v1(
    draft: &BundleDraftV1,
    components: Vec<BundleComponentV1>,
    planned: Vec<BundlePlannedComponentV1>,
    implementations: Vec<BundleImplementationV1>,
    native_module: Option<BundleNativeModuleV1>,
) -> Result<BundleManifestV1, CompilerError> {
    let topologies = draft.target.topologies();
    let first = topologies[0].id();
    let joined = topologies.len() > 1 || components.iter().any(|c| !c.topologies().is_empty());
    let first_components = components
        .iter()
        .filter(|component| component.serves(first.as_str()))
        .map(|component| component.for_topology(first))
        .collect();
    let mut manifest = BundleManifestV1::new_with_planned_v1(
        draft.id.clone(),
        BundleTargetV1::new(
            draft.target.architectures().to_vec(),
            vec![topologies[0].clone()],
        )?,
        first_components,
        planned,
        draft.resource_requirements.clone(),
        draft.kv_layout.clone(),
        draft.state_layout.clone(),
        draft.graph_buckets.clone(),
        draft.operation_requirements.clone(),
        implementations,
        native_module,
    )?
    .with_native_symbol_sequences(draft.native_symbol_sequences.clone())?
    .with_recurrent_state_layouts(draft.recurrent_state_layouts.clone())?
    .with_kv_formats(draft.kv_formats.clone())?;
    if let Some(speculation) = &draft.speculation {
        manifest = manifest.with_speculation(speculation.clone())?;
    }
    if let Some(limits) = draft.batch_limits {
        manifest = manifest.with_batch_limits(limits)?;
    }
    if joined {
        manifest = manifest.with_topologies(
            draft.target.clone(),
            components,
            draft.topology_resources.clone(),
        )?;
    }
    if let Some(options) = &draft.model_options {
        manifest = manifest.with_model_options(options.clone())?;
    }
    Ok(manifest)
}

/// The host descriptor the emitter verifies its own output against.
///
/// This is not a compatibility check: it restates what the manifest declares so
/// the verifier runs its file, digest, and closed-set checks. Whether a real
/// host can run the bundle is that host's question, asked at install and load.
fn emitter_host(manifest: &BundleManifestV1) -> Result<BundleHostCompatibilityV1, CompilerError> {
    let architecture = manifest.target().architectures().first().ok_or_else(|| {
        CompilerError::validation("bundle target declares no architecture".to_owned())
    })?;
    Ok(
        BundleHostCompatibilityV1::new(architecture.as_str(), MODEL_BUNDLE_V1_ABI_VERSION)?
            .with_capabilities(manifest.capabilities_of_every_topology().cloned()),
    )
}

impl ImplementationDraftV1 {
    /// Builds this implementation's record from the files that were staged.
    ///
    /// Both origins are measured the same way: the length and digest come from
    /// the file the emitter actually wrote, never from the draft. A native
    /// module is staged before any implementation is declared, so its record is
    /// already the measured one by the time this reads it.
    fn declare(
        &self,
        staged: &[(BundleOpaqueId, BundleOpaqueId, StagedFileV1)],
        native_module: Option<&BundleNativeModuleV1>,
    ) -> Result<BundleImplementationV1, CompilerError> {
        let artifact = match &self.artifact {
            ImplementationArtifactDraftV1::Component(component) => {
                let (_, format, file) = staged
                    .iter()
                    .find(|(id, _, _)| id == component)
                    .ok_or_else(|| {
                        CompilerError::validation(format!(
                            "implementation {} names artifact component {component}, which was \
                             not staged",
                            self.id
                        ))
                    })?;
                BundleArtifactIdentityV1::from_component(
                    component.clone(),
                    format.clone(),
                    file.byte_length,
                    file.sha256.clone(),
                )?
            }
            ImplementationArtifactDraftV1::NativeModule => {
                let native_module = native_module.ok_or_else(|| {
                    CompilerError::validation(format!(
                        "implementation {} is backed by a native module and this bundle stages \
                         none",
                        self.id
                    ))
                })?;
                BundleArtifactIdentityV1::from_native_module(
                    native_module.byte_length(),
                    native_module.sha256(),
                )?
            }
        };
        let detail = BundleImplementationDetailV1::new(
            artifact,
            self.symbol.clone(),
            self.target_architecture.clone(),
            self.launch,
            self.workspace.clone(),
            self.specializations.clone(),
            self.source.clone(),
            self.build.clone(),
        )?;
        Ok(BundleImplementationV1::new(
            self.id.clone(),
            self.key.clone(),
            BundleImplementationTierV1::Bundle,
            detail,
        )?)
    }
}

/// One part of one staged file, and where it came from.
#[derive(Clone, Debug)]
struct PlannedPartV1 {
    source: PlannedSourceV1,
    byte_length: u64,
}

#[derive(Clone, Debug)]
enum PlannedSourceV1 {
    Bytes(Vec<u8>),
    File(PathBuf),
    AuthenticatedFile {
        path: PathBuf,
        identity: FileIdentity,
    },
}

impl PlannedPartV1 {
    fn bytes(bytes: Vec<u8>) -> Self {
        let byte_length = bytes.len() as u64;
        Self {
            source: PlannedSourceV1::Bytes(bytes),
            byte_length,
        }
    }

    /// Measures one declared input before anything is opened for writing.
    ///
    /// A source that is a symlink, a directory, a device, or larger than the
    /// emitter stages is refused here, so the failure happens before a
    /// destination file exists rather than half-way through one.
    fn content(content: &StagedContentV1) -> Result<Self, CompilerError> {
        match content {
            StagedContentV1::Bytes(bytes) => Ok(Self::bytes(bytes.clone())),
            StagedContentV1::File(path) => {
                let metadata = regular_file_metadata(path)?;
                if metadata.len() > MAX_STAGED_FILE_BYTES {
                    return Err(CompilerError::validation(format!(
                        "bundle input {} has {} bytes, the emitter stages at most \
                         {MAX_STAGED_FILE_BYTES}",
                        path.display(),
                        metadata.len()
                    )));
                }
                Ok(Self {
                    source: PlannedSourceV1::File(path.clone()),
                    byte_length: metadata.len(),
                })
            }
            StagedContentV1::AuthenticatedFile { path, identity } => {
                if identity.size() > MAX_STAGED_FILE_BYTES {
                    return Err(CompilerError::validation(format!(
                        "bundle input {} is pinned at {} bytes, the emitter stages at most \
                         {MAX_STAGED_FILE_BYTES}",
                        path.display(),
                        identity.size()
                    )));
                }
                // The name is half of what the identity pins, so a path that
                // ends in some other name is refused here rather than staged
                // and then found to hash correctly for a different file.
                let name = path.file_name().and_then(|name| name.to_str());
                if name != Some(identity.name()) {
                    return Err(CompilerError::validation(format!(
                        "bundle input {} is not named {}, which its pinned identity requires",
                        path.display(),
                        identity.name()
                    )));
                }
                let metadata = regular_file_metadata(path)?;
                if metadata.len() != identity.size() {
                    return Err(CompilerError::validation(format!(
                        "bundle input {} has {} bytes, its pinned identity declares {}",
                        path.display(),
                        metadata.len(),
                        identity.size()
                    )));
                }
                Ok(Self {
                    source: PlannedSourceV1::AuthenticatedFile {
                        path: path.clone(),
                        identity: identity.clone(),
                    },
                    byte_length: identity.size(),
                })
            }
        }
    }

    fn open(&self) -> Result<PartReaderV1, CompilerError> {
        Ok(match &self.source {
            PlannedSourceV1::Bytes(bytes) => PartReaderV1::Bytes(io::Cursor::new(bytes.clone())),
            PlannedSourceV1::File(path) => PartReaderV1::File(self.open_source(path)?),
            PlannedSourceV1::AuthenticatedFile { path, identity } => {
                PartReaderV1::AuthenticatedFile(AuthenticatedPartV1 {
                    file: self.open_source(path)?,
                    path: path.clone(),
                    identity: identity.clone(),
                    digest: Sha256State::new(),
                    read: 0,
                })
            }
        })
    }

    /// Opens one measured source and checks the descriptor it actually got.
    ///
    /// Measuring a path and opening it are two lookups, and an ordinary open
    /// reads whatever the name points at by the second one. The open below
    /// refuses a symlink itself, and the type and length are then checked on the
    /// descriptor that is read, so a source swapped after it was measured is
    /// refused rather than followed. Everything after this point asks the
    /// descriptor, never the name.
    fn open_source(&self, path: &Path) -> Result<File, CompilerError> {
        let file = open_no_follow(path)?;
        let metadata = file
            .metadata()
            .map_err(|error| CompilerError::io("inspect", path, &error))?;
        if !metadata.is_file() || metadata.len() != self.byte_length {
            return Err(CompilerError::validation(format!(
                "bundle input {} changed before it was staged",
                path.display()
            )));
        }
        Ok(file)
    }
}

/// One pinned source, hashed over the same read that writes it.
struct AuthenticatedPartV1 {
    file: File,
    path: PathBuf,
    identity: FileIdentity,
    digest: Sha256State,
    read: u64,
}

impl AuthenticatedPartV1 {
    /// Hands out the next bytes of the pinned file, and hashes exactly those.
    ///
    /// The read is capped at what the identity declares is left, so a file that
    /// grew under the emitter cannot lengthen the staged component. The growth
    /// itself is what the descriptor check in [`Self::finish`] refuses.
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        let remaining = self.identity.size().saturating_sub(self.read);
        if remaining == 0 {
            return Ok(0);
        }
        let bound = usize::try_from(remaining)
            .unwrap_or(usize::MAX)
            .min(buffer.len());
        let count = self.file.read(&mut buffer[..bound])?;
        if count > 0 {
            self.digest
                .update(&buffer[..count])
                .map_err(|error| io::Error::other(error.to_string()))?;
            self.read += count as u64;
        }
        Ok(count)
    }

    /// Decides whether what was written is the file that was pinned.
    ///
    /// The descriptor is asked again here rather than the path, because the
    /// path may name something else by now and the bytes that were written came
    /// from this descriptor. Length, type, and digest all have to still hold.
    fn finish(self) -> Result<(), CompilerError> {
        let metadata = self
            .file
            .metadata()
            .map_err(|error| CompilerError::io("inspect", &self.path, &error))?;
        if !metadata.is_file() || metadata.len() != self.identity.size() {
            return Err(CompilerError::validation(format!(
                "bundle input {} changed while it was staged",
                self.path.display()
            )));
        }
        if self.read != self.identity.size() {
            return Err(CompilerError::validation(format!(
                "bundle input {} yielded {} bytes, its pinned identity declares {}",
                self.path.display(),
                self.read,
                self.identity.size()
            )));
        }
        let digest = self.digest.finalize();
        if digest != self.identity.sha256() {
            return Err(CompilerError::validation(format!(
                "bundle input {} has SHA-256 {digest}, its pinned identity declares {}",
                self.path.display(),
                self.identity.sha256()
            )));
        }
        Ok(())
    }
}

enum PartReaderV1 {
    Bytes(io::Cursor<Vec<u8>>),
    File(File),
    AuthenticatedFile(AuthenticatedPartV1),
}

impl PartReaderV1 {
    /// What this part still has to prove once it has yielded its last byte.
    fn finish(self) -> Result<(), CompilerError> {
        match self {
            Self::AuthenticatedFile(part) => part.finish(),
            Self::Bytes(_) | Self::File(_) => Ok(()),
        }
    }
}

impl Read for PartReaderV1 {
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        match self {
            Self::Bytes(bytes) => bytes.read(buffer),
            Self::File(file) => file.read(buffer),
            Self::AuthenticatedFile(part) => part.read(buffer),
        }
    }
}

/// The parts of one staged file, read end to end as a single stream.
///
/// A part that fails to open, or that fails what it has to prove once it ends,
/// stops the stream. The reason is kept here as well as reported as an I/O
/// error, because the digest this reader feeds can only report the latter and
/// the caller needs to say which input it refused and why.
struct ConcatReaderV1 {
    parts: Vec<PlannedPartV1>,
    index: usize,
    open: Option<PartReaderV1>,
    refusal: Option<CompilerError>,
}

impl ConcatReaderV1 {
    fn new(parts: Vec<PlannedPartV1>) -> Self {
        Self {
            parts,
            index: 0,
            open: None,
            refusal: None,
        }
    }

    fn refuse(&mut self, error: CompilerError) -> io::Error {
        let failure = io::Error::other(error.to_string());
        self.refusal = Some(error);
        failure
    }
}

impl Read for ConcatReaderV1 {
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        loop {
            if self.open.is_none() {
                let Some(part) = self.parts.get(self.index) else {
                    return Ok(0);
                };
                self.index += 1;
                match part.open() {
                    Ok(reader) => self.open = Some(reader),
                    Err(error) => return Err(self.refuse(error)),
                }
            }
            let reader = self.open.as_mut().expect("a part is open");
            let count = reader.read(buffer)?;
            if count > 0 {
                return Ok(count);
            }
            let ended = self.open.take().expect("a part is open");
            if let Err(error) = ended.finish() {
                return Err(self.refuse(error));
            }
        }
    }
}

/// A reader that writes every byte it hands out into the staged file.
///
/// Hashing reads through this, so the digest is taken over exactly the bytes
/// that reached the file rather than over a second read of the path.
struct TeeWriterV1<'a> {
    source: ConcatReaderV1,
    sink: &'a mut File,
    written: u64,
    failure: Option<io::Error>,
}

impl Read for TeeWriterV1<'_> {
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        let count = self.source.read(buffer)?;
        if count > 0 {
            if let Err(error) = self.sink.write_all(&buffer[..count]) {
                self.failure = Some(io::Error::new(error.kind(), error.to_string()));
                return Err(error);
            }
            self.written += count as u64;
        }
        Ok(count)
    }
}

/// One file after it was written: its length and SHA-256.
#[derive(Clone, Debug, Eq, PartialEq)]
struct StagedFileV1 {
    byte_length: u64,
    sha256: String,
}

fn measure_file(
    relative_path: &BundleRelativePathV1,
    parts: Vec<PlannedPartV1>,
    maximum_bytes: u64,
) -> Result<StagedFileV1, CompilerError> {
    let byte_length = parts.iter().try_fold(0_u64, |total, part| {
        total.checked_add(part.byte_length).ok_or_else(|| {
            CompilerError::validation(format!("staged file {relative_path} length overflows u64"))
        })
    })?;
    let source = ConcatReaderV1::new(parts);
    if byte_length > maximum_bytes {
        return Err(CompilerError::validation(format!(
            "staged file {relative_path} would hold {byte_length} bytes, the emitter stages at most {maximum_bytes}"
        )));
    }
    let mut measured = CountingReaderV1::new(source);
    let sha256 = sha256_reader(&mut measured).map_err(|error| {
        measured.inner.refusal.take().unwrap_or_else(|| {
            CompilerError::validation(format!(
                "cannot measure {}: {error}",
                relative_path.as_str()
            ))
        })
    })?;
    if measured.read != byte_length {
        return Err(CompilerError::validation(format!(
            "staged file {relative_path} declared {byte_length} bytes but measurement read {}",
            measured.read
        )));
    }
    Ok(StagedFileV1 {
        byte_length,
        sha256,
    })
}

struct CountingReaderV1<R> {
    inner: R,
    read: u64,
}

impl<R> CountingReaderV1<R> {
    const fn new(inner: R) -> Self {
        Self { inner, read: 0 }
    }
}

impl<R: Read> Read for CountingReaderV1<R> {
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        let count = self.inner.read(buffer)?;
        self.read = self
            .read
            .checked_add(count as u64)
            .ok_or_else(|| io::Error::other("measured byte count overflows u64"))?;
        Ok(count)
    }
}

fn stage_file(
    staging: &Path,
    relative_path: &BundleRelativePathV1,
    parts: Vec<PlannedPartV1>,
    maximum_bytes: u64,
) -> Result<StagedFileV1, CompilerError> {
    let offset = parts.iter().try_fold(0_u64, |total, part| {
        total.checked_add(part.byte_length).ok_or_else(|| {
            CompilerError::validation(format!("staged file {relative_path} length overflows u64"))
        })
    })?;
    if offset > maximum_bytes {
        return Err(CompilerError::validation(format!(
            "staged file {relative_path} would hold {offset} bytes, the emitter stages at most \
             {maximum_bytes}"
        )));
    }

    let path = staging.join(relative_path.as_path());
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .map_err(|error| CompilerError::io("create staging directory", parent, &error))?;
    }
    let mut file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&path)
        .map_err(|error| CompilerError::io("create staging file", &path, &error))?;
    let mut tee = TeeWriterV1 {
        source: ConcatReaderV1::new(parts),
        sink: &mut file,
        written: 0,
        failure: None,
    };
    let hashed = sha256_reader(&mut tee);
    let written = tee.written;
    let failure = tee.failure.take();
    let refusal = tee.source.refusal.take();
    let sha256 = hashed.map_err(|error| match (refusal, failure) {
        // A part that refused itself says so in its own words. The write and the
        // digest both stopped because of it, so neither has anything to add.
        (Some(refusal), _) => refusal,
        (None, Some(failure)) => CompilerError::io("write staging file", &path, &failure),
        (None, None) => {
            CompilerError::validation(format!("cannot stage {}: {error}", relative_path.as_str()))
        }
    })?;
    if written != offset {
        return Err(CompilerError::validation(format!(
            "staged file {relative_path} received {written} bytes where {offset} were planned"
        )));
    }
    file.sync_all()
        .map_err(|error| CompilerError::io("sync staging file", &path, &error))?;
    Ok(StagedFileV1 {
        byte_length: written,
        sha256,
    })
}

fn create_staging_directory(parent: &Path, output: &Path) -> Result<PathBuf, CompilerError> {
    let name = output
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| {
            CompilerError::validation(format!(
                "bundle output {} has no usable directory name",
                output.display()
            ))
        })?;
    for _ in 0..STAGING_ATTEMPTS {
        let ordinal = NEXT_STAGING_DIRECTORY.fetch_add(1, Ordering::Relaxed);
        let staging = parent.join(format!(
            ".{name}.compiling-{}-{ordinal}",
            std::process::id()
        ));
        match fs::create_dir(&staging) {
            Ok(()) => return Ok(staging),
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(error) => {
                return Err(CompilerError::io(
                    "create staging directory",
                    &staging,
                    &error,
                ));
            }
        }
    }
    Err(CompilerError::validation(format!(
        "cannot allocate a unique staging directory beside {}",
        output.display()
    )))
}

fn validate_directory(path: &Path, label: &str) -> Result<(), CompilerError> {
    let metadata =
        fs::symlink_metadata(path).map_err(|error| CompilerError::io("inspect", path, &error))?;
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        return Err(CompilerError::validation(format!(
            "{label} must be a directory and not a symlink: {}",
            path.display()
        )));
    }
    Ok(())
}

/// Opens one declared input, refusing a symlink at the final path component.
///
/// The refusal is the open itself rather than a check before it, so nothing
/// happens between deciding the path is a regular file and reading the file the
/// name resolves to. The caller still checks the type and length on the
/// returned descriptor, because a no-follow open says nothing about either.
#[cfg(any(target_os = "linux", target_os = "macos"))]
fn open_no_follow(path: &Path) -> Result<File, CompilerError> {
    use std::os::unix::fs::OpenOptionsExt;

    OpenOptions::new()
        .read(true)
        .custom_flags(NO_FOLLOW_FLAG)
        .open(path)
        .map_err(|error| CompilerError::io("open", path, &error))
}

/// The two-lookup fallback for hosts this compiler is not built for.
///
/// It cannot see a swap between the check and the open, so it says so here
/// rather than pretending the guarantee above holds everywhere.
#[cfg(not(any(target_os = "linux", target_os = "macos")))]
fn open_no_follow(path: &Path) -> Result<File, CompilerError> {
    regular_file_metadata(path)?;
    File::open(path).map_err(|error| CompilerError::io("open", path, &error))
}

fn regular_file_metadata(path: &Path) -> Result<fs::Metadata, CompilerError> {
    let metadata =
        fs::symlink_metadata(path).map_err(|error| CompilerError::io("inspect", path, &error))?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err(CompilerError::validation(format!(
            "bundle input must be a regular file and not a symlink: {}",
            path.display()
        )));
    }
    Ok(metadata)
}

fn sync_directory(path: &Path) -> Result<(), CompilerError> {
    File::open(path)
        .and_then(|directory| directory.sync_all())
        .map_err(|error| CompilerError::io("sync directory", path, &error))
}

fn path_exists(path: &Path) -> Result<bool, CompilerError> {
    match fs::symlink_metadata(path) {
        Ok(_) => Ok(true),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(false),
        Err(error) => Err(CompilerError::io("inspect", path, &error)),
    }
}

#[cfg(all(test, unix))]
mod tests {
    use std::fs;
    use std::io::Read;
    use std::os::unix::fs::symlink;
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicU64, Ordering};

    use super::{
        BundleRelativePathV1, MAX_STAGED_FILE_BYTES, PlannedPartV1, StagedContentV1, measure_file,
        open_no_follow,
    };

    /// The open itself refuses a symlink, whatever the path looked like before.
    #[test]
    fn the_input_open_refuses_a_symlink() {
        let root = ScratchDirectory::new("model-compiler-sdk-no-follow");
        let target = root.path().join("target.bin");
        fs::write(&target, b"target bytes").expect("the target file is writable");
        let link = root.path().join("link.bin");
        symlink(&target, &link).expect("the fixture symlink is creatable");

        let mut opened = open_no_follow(&target).expect("a regular file opens");
        let mut bytes = Vec::new();
        opened.read_to_end(&mut bytes).expect("the file reads");
        assert_eq!(bytes, b"target bytes");

        let error = open_no_follow(&link).expect_err("a symlinked input must not open");
        assert!(
            error.to_string().starts_with("cannot open ") && error.to_string().contains("link.bin"),
            "the open itself must be what refuses the symlink: {error}"
        );
    }

    /// A source swapped for a symlink after it was measured is refused at the
    /// open, which is the only lookup that can still see the swap.
    #[test]
    fn a_source_swapped_after_it_was_measured_is_refused() {
        let root = ScratchDirectory::new("model-compiler-sdk-swap");
        let source = root.path().join("source.bin");
        fs::write(&source, b"measured bytes").expect("the source file is writable");
        let part = PlannedPartV1::content(&StagedContentV1::file(&source))
            .expect("a regular file measures");

        // The same name, the same length, and a symlink where the regular file
        // was: only the open can tell the difference now.
        let elsewhere = root.path().join("elsewhere.bin");
        fs::write(&elsewhere, b"swapped bytes!").expect("the swapped file is writable");
        fs::remove_file(&source).expect("the measured file is removable");
        symlink(&elsewhere, &source).expect("the swap is creatable");

        let Err(error) = part.open() else {
            panic!("a swapped source must not be staged");
        };
        assert!(
            error.to_string().starts_with("cannot open ")
                && error.to_string().contains("source.bin"),
            "the open, not the earlier measurement, must be what refuses the swap: {error}"
        );
    }

    #[test]
    fn manifest_planning_refuses_short_and_long_reads() {
        for (label, declared) in [("short", 4_u64), ("long", 2_u64)] {
            let relative = BundleRelativePathV1::new("components/drift.bin").expect("path");
            let mut part = PlannedPartV1::bytes(vec![1, 2, 3]);
            part.byte_length = declared;
            let error = measure_file(&relative, vec![part], MAX_STAGED_FILE_BYTES)
                .expect_err("a read length mismatch must be refused");
            assert!(
                error.to_string().contains("measurement read 3"),
                "{label}: {error}"
            );
        }
    }

    struct ScratchDirectory {
        path: PathBuf,
    }

    impl ScratchDirectory {
        fn new(label: &str) -> Self {
            static NEXT: AtomicU64 = AtomicU64::new(0);

            let path = std::env::temp_dir().join(format!(
                "{label}-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            let _ = fs::remove_dir_all(&path);
            fs::create_dir_all(&path).expect("the scratch directory is creatable");
            Self { path }
        }

        fn path(&self) -> &Path {
            &self.path
        }
    }

    impl Drop for ScratchDirectory {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.path);
        }
    }
}
