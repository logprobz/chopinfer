#![deny(unsafe_code)]

//! Canonical parsing, verification, installation, and load-time checks for
//! `chop-model-bundle-v1` directories.
//!
//! The model-neutral operations a bundle may leave to its host are declared in
//! [`generic`], because both the manifest builder that requires one and the
//! host catalog that offers it have to name the same contract.

#[cfg(target_os = "linux")]
mod atomic_publish;
pub mod checkpoint_binding;
pub mod deployment;
/// The model-neutral generic primitives a bundle may require from its host.
mod generic;
pub mod rank_index;

/// Offline metadata shared by compiler diagnostics and bundle producers.
pub use runtime_bundle_sdk::diagnostic;

pub use generic::{
    DTYPE_F32_V1, DTYPE_I32_V1, DTYPE_U32_V1, GENERIC_COPY_F32_ABI_VERSION_V1,
    GENERIC_COPY_F32_ALIGNMENT_V1, GENERIC_COPY_F32_BLOCK_ELEMENTS_V1,
    GENERIC_COPY_F32_DESTINATION_ROLE_V1, GENERIC_COPY_F32_ELEMENT_BYTES_V1,
    GENERIC_COPY_F32_ELEMENT_COUNT_V1, GENERIC_COPY_F32_OPERATION_V1,
    GENERIC_COPY_F32_PRIMITIVE_V1, GENERIC_COPY_F32_SOURCE_ROLE_V1, GENERIC_COPY_F32_VARIANT_V1,
    LAYOUT_DENSE_ROW_MAJOR_V1, NUMERICS_EXACT_COPY_V1, SCALE_NONE_V1, generic_copy_f32_bytes_v1,
    generic_copy_f32_contract_v1, generic_copy_f32_implementation_id_v1, generic_copy_f32_key_v1,
};

use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read};
#[cfg(unix)]
use std::os::unix::fs::OpenOptionsExt;
use std::path::{Component, Path, PathBuf};
use std::sync::Arc;

use object::{BinaryFormat, Object};
use runtime_kernel_contract::{
    KernelAliasContractV1, KernelArtifactIdentityV1, KernelBuildIdentityV1,
    KernelImplementationCandidateV1, KernelImplementationDetailPartsV1,
    KernelImplementationDetailV1, KernelImplementationTierV1, KernelLaunchContractV1,
    KernelOpaqueIdV1, KernelOperationAbiV1, KernelResolverV1, KernelScalarArgumentV1,
    KernelSemanticKeyPartsV1, KernelSemanticKeyV1, KernelSha256V1, KernelSourceIdentityV1,
    KernelSpecializationV1, KernelTensorAccessV1, KernelTensorDependencyV1,
    KernelTensorRequirementV1, KernelWorkspaceContractV1,
};
use runtime_kv::{DeclaredPageLayoutV1, RankPlacementV1, StateLifetimeV1};
use runtime_weights::sha256_bytes;
use serde::de::{self, DeserializeSeed, MapAccess, SeqAccess, Visitor};
use serde::{Deserialize, Serialize};
use serde_json::Value;

pub const MODEL_BUNDLE_V1_SCHEMA: &str = "chop-model-bundle-v1";
pub const MODEL_BUNDLE_V1_ABI_VERSION: u32 = 1;
pub const BUNDLE_MANIFEST_FILE: &str = "manifest.json";
pub mod model_options;
pub mod preparation;

pub const BUNDLE_NATIVE_ENTRY_SYMBOL_V1: &str = "chop_bundle_get_api_v1";
pub const SPECULATIVE_CAPABILITY_V1: &str = "chop.speculative.v1";
/// Required by a package whose weight plan places views in host memory; the
/// bundle ABI states what the host does for it.
pub const HOST_RESIDENT_VIEWS_CAPABILITY_V1: &str = "chop.host-resident-views.v1";
/// Required by a package whose weight plan maps checkpoint tensors into host
/// memory; the bundle ABI states what the host does for it.
pub const HOST_MAPPED_GATHER_CAPABILITY_V1: &str = "chop.host-mapped-gather.v1";
/// Required by a package whose bundle reads the served item count; the bundle
/// ABI states what the host sends for it.
pub const SERVED_ITEMS_CAPABILITY_V1: &str = "chop.served-items.v1";
/// Required by a package whose bundle plans weights from a checkpoint
/// (`runtime_bundle_abi::weight_plan`), and so by a package with planned
/// components.
pub const WEIGHT_PLAN_CAPABILITY_V1: &str = "chop.weight-plan.v1";
/// Required by a package whose `state_layout` bytes are per state slot: see
/// [`BundleManifestV1::state_bytes_v1`] and the bundle ABI.
pub const STATE_PER_ITEM_CAPABILITY_V1: &str = "chop.state-per-item.v1";

/// State slots of an instance serving `served` items under
/// `chop.state-per-item.v1`: the count rounded up to a power of two, the
/// largest graph bucket, which per-slot kernels are built for.
#[must_use]
pub const fn state_slots_v1(served: u32) -> u64 {
    (served as u64).next_power_of_two()
}
pub const BUNDLE_MANIFEST_MAX_BYTES: usize = 256 * 1024;

const MAX_JSON_DEPTH: usize = 16;
const MAX_ID_BYTES: usize = 128;
const MAX_PATH_BYTES: usize = 512;
const MAX_COMPONENTS: usize = 256;
const MAX_ARCHITECTURES: usize = 16;
const MAX_TOPOLOGIES: usize = 16;
const MAX_CAPABILITIES: usize = 64;
const MAX_BUCKETS: usize = 256;
const MAX_BUCKET_COMPONENTS: usize = 256;
const MAX_BUCKET_OPERATIONS: usize = 1_024;
const MAX_REQUIREMENTS: usize = 1_024;
const MAX_IMPLEMENTATIONS: usize = 1_024;
const MAX_COMPONENT_BYTES: u64 = 1 << 50;
const MAX_NATIVE_MODULE_BYTES: u64 = 512 * 1024 * 1024;
const MAX_TOPOLOGY_RANKS: u32 = 64;
const MAX_TENSOR_DIMENSIONS: usize = 8;
const MAX_TENSOR_DEPENDENCIES: usize = 16;
const MAX_OPERATION_TENSORS: usize = 64;
const MAX_OPERATION_SCALARS: usize = 64;
const MAX_SPECIALIZATIONS: usize = 64;
const MAX_NATIVE_SYMBOL_SEQUENCES: usize = 256;
const MAX_SEQUENCE_SYMBOLS: usize = 1_024;

/// Errors from canonical bundle handling.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BundleError(String);

impl BundleError {
    #[must_use]
    pub fn validation(message: impl Into<String>) -> Self {
        Self(message.into())
    }

    fn io(action: &str, path: &Path, error: io::Error) -> Self {
        Self(format!("cannot {action} {}: {error}", path.display()))
    }
}

impl fmt::Display for BundleError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl Error for BundleError {}

/// A bounded opaque identifier used in the V1 manifest.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(transparent)]
pub struct BundleOpaqueId(String);

impl BundleOpaqueId {
    pub fn new(value: impl Into<String>) -> Result<Self, BundleError> {
        let value = value.into();
        validate_opaque_id(&value)?;
        Ok(Self(value))
    }

    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for BundleOpaqueId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

/// A relative path inside a bundle directory.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(transparent)]
pub struct BundleRelativePathV1(String);

impl BundleRelativePathV1 {
    pub fn new(value: impl Into<String>) -> Result<Self, BundleError> {
        let value = value.into();
        validate_relative_path(&value)?;
        Ok(Self(value))
    }

    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }

    #[must_use]
    pub fn as_path(&self) -> &Path {
        Path::new(&self.0)
    }
}

impl fmt::Display for BundleRelativePathV1 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

/// A target topology accepted by a bundle.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BundleTopologyV1 {
    id: BundleOpaqueId,
    rank_count: u32,
    /// The context-parallel sizes above 1 this topology serves, ascending.
    /// Size 1 is always served, and a topology without others names none,
    /// so this stays out of the canonical JSON of every manifest before it.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    context_parallel_sizes: Vec<u32>,
}

impl BundleTopologyV1 {
    pub fn new(id: BundleOpaqueId, rank_count: u32) -> Result<Self, BundleError> {
        let topology = Self {
            id,
            rank_count,
            context_parallel_sizes: Vec::new(),
        };
        topology.validate()?;
        Ok(topology)
    }

    /// This topology also serving the context-parallel `sizes`, each above 1
    /// and dividing its rank count.
    pub fn with_context_parallel_sizes(mut self, mut sizes: Vec<u32>) -> Result<Self, BundleError> {
        sizes.sort_unstable();
        self.context_parallel_sizes = sizes;
        self.validate()?;
        Ok(self)
    }

    /// The declared context-parallel sizes above 1.
    #[must_use]
    pub fn context_parallel_sizes(&self) -> &[u32] {
        &self.context_parallel_sizes
    }

    /// Whether this topology serves context-parallel `size`: 1 always, any
    /// other size only when declared.
    #[must_use]
    pub fn serves_context_parallel(&self, size: u32) -> bool {
        size == 1 || self.context_parallel_sizes.contains(&size)
    }

    #[must_use]
    pub const fn id(&self) -> &BundleOpaqueId {
        &self.id
    }

    #[must_use]
    pub const fn rank_count(&self) -> u32 {
        self.rank_count
    }

    fn validate(&self) -> Result<(), BundleError> {
        validate_opaque_id(self.id.as_str())?;
        if self.rank_count == 0 || self.rank_count > MAX_TOPOLOGY_RANKS {
            return Err(BundleError::validation(format!(
                "bundle topology {} rank count {} is outside 1..={MAX_TOPOLOGY_RANKS}",
                self.id, self.rank_count
            )));
        }
        for size in &self.context_parallel_sizes {
            if *size < 2 || !self.rank_count.is_multiple_of(*size) {
                return Err(BundleError::validation(format!(
                    "bundle topology {} context-parallel size {size} must be above 1 and divide \
                     its {} ranks",
                    self.id, self.rank_count
                )));
            }
        }
        validate_strict_order("context-parallel sizes", &self.context_parallel_sizes)?;
        Ok(())
    }
}

/// The architecture and topology constraints for a V1 bundle.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BundleTargetV1 {
    architectures: Vec<BundleOpaqueId>,
    topologies: Vec<BundleTopologyV1>,
}

impl BundleTargetV1 {
    pub fn new(
        mut architectures: Vec<BundleOpaqueId>,
        mut topologies: Vec<BundleTopologyV1>,
    ) -> Result<Self, BundleError> {
        architectures.sort();
        topologies.sort_by(|left, right| left.id.cmp(&right.id));
        let target = Self {
            architectures,
            topologies,
        };
        target.validate()?;
        Ok(target)
    }

    #[must_use]
    pub fn architectures(&self) -> &[BundleOpaqueId] {
        &self.architectures
    }

    #[must_use]
    pub fn topologies(&self) -> &[BundleTopologyV1] {
        &self.topologies
    }

    fn validate(&self) -> Result<(), BundleError> {
        validate_nonempty_count(
            "target architectures",
            self.architectures.len(),
            MAX_ARCHITECTURES,
        )?;
        validate_nonempty_count("target topologies", self.topologies.len(), MAX_TOPOLOGIES)?;
        for architecture in &self.architectures {
            validate_opaque_id(architecture.as_str())?;
        }
        for topology in &self.topologies {
            topology.validate()?;
        }
        validate_strict_order("target architectures", &self.architectures)?;
        validate_strict_order_by("target topologies", &self.topologies, |value| value.id())
    }
}

/// A bundle component with its opaque role, format, path, and exact identity.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BundleComponentV1 {
    id: BundleOpaqueId,
    role: BundleOpaqueId,
    format: BundleOpaqueId,
    path: BundleRelativePathV1,
    byte_length: u64,
    sha256: String,
    /// The topologies this component serves when it does not serve every
    /// topology the bundle declares. A component of every topology names
    /// none, so this stays out of the canonical JSON of every manifest that
    /// predates it.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    topologies: Vec<BundleOpaqueId>,
}

impl BundleComponentV1 {
    pub fn new(
        id: BundleOpaqueId,
        role: BundleOpaqueId,
        format: BundleOpaqueId,
        path: BundleRelativePathV1,
        byte_length: u64,
        sha256: impl Into<String>,
    ) -> Result<Self, BundleError> {
        let component = Self {
            id,
            role,
            format,
            path,
            byte_length,
            sha256: sha256.into(),
            topologies: Vec::new(),
        };
        component.validate()?;
        Ok(component)
    }

    /// This component serving only `topologies`, which the bundle declares.
    pub fn with_topologies(
        mut self,
        mut topologies: Vec<BundleOpaqueId>,
    ) -> Result<Self, BundleError> {
        topologies.sort();
        self.topologies = topologies;
        self.validate()?;
        Ok(self)
    }

    /// The topologies this component serves; empty when it serves every one.
    #[must_use]
    pub fn topologies(&self) -> &[BundleOpaqueId] {
        &self.topologies
    }

    /// Whether this component serves `topology`.
    #[must_use]
    pub fn serves(&self, topology: &str) -> bool {
        self.topologies.is_empty()
            || self
                .topologies
                .iter()
                .any(|candidate| candidate.as_str() == topology)
    }

    /// This component, which serves `topology`, in a manifest of that
    /// topology alone: a restricted component names only it.
    #[must_use]
    pub fn for_topology(&self, topology: &BundleOpaqueId) -> Self {
        let mut component = self.clone();
        if !component.topologies.is_empty() {
            component.topologies = vec![topology.clone()];
        }
        component
    }

    #[must_use]
    pub const fn id(&self) -> &BundleOpaqueId {
        &self.id
    }

    #[must_use]
    pub const fn role(&self) -> &BundleOpaqueId {
        &self.role
    }

    #[must_use]
    pub const fn format(&self) -> &BundleOpaqueId {
        &self.format
    }

    #[must_use]
    pub const fn path(&self) -> &BundleRelativePathV1 {
        &self.path
    }

    #[must_use]
    pub const fn byte_length(&self) -> u64 {
        self.byte_length
    }

    #[must_use]
    pub fn sha256(&self) -> &str {
        &self.sha256
    }

    fn validate(&self) -> Result<(), BundleError> {
        validate_opaque_id(self.id.as_str())?;
        validate_opaque_id(self.role.as_str())?;
        validate_opaque_id(self.format.as_str())?;
        validate_relative_path(self.path.as_str())?;
        validate_declared_file("component", self.byte_length, MAX_COMPONENT_BYTES)?;
        validate_count(
            "component topologies",
            self.topologies.len(),
            MAX_TOPOLOGIES,
        )?;
        validate_strict_order_by("component topologies", &self.topologies, |id| id)?;
        validate_sha256(&self.sha256)
    }
}

/// A compiled image the bundle's weight plan supplies, declared without a
/// file: its role and format, which the host asks the plan for and a create
/// path opens, and nothing else. Sizes and alignment come from the plan. A
/// package with one serves only by a direct checkpoint load.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BundlePlannedComponentV1 {
    id: BundleOpaqueId,
    role: BundleOpaqueId,
    format: BundleOpaqueId,
}

impl BundlePlannedComponentV1 {
    pub fn new(
        id: BundleOpaqueId,
        role: BundleOpaqueId,
        format: BundleOpaqueId,
    ) -> Result<Self, BundleError> {
        let component = Self { id, role, format };
        component.validate()?;
        Ok(component)
    }

    #[must_use]
    pub const fn id(&self) -> &BundleOpaqueId {
        &self.id
    }

    #[must_use]
    pub const fn role(&self) -> &BundleOpaqueId {
        &self.role
    }

    #[must_use]
    pub const fn format(&self) -> &BundleOpaqueId {
        &self.format
    }

    fn validate(&self) -> Result<(), BundleError> {
        validate_opaque_id(self.id.as_str())?;
        validate_opaque_id(self.role.as_str())?;
        validate_opaque_id(self.format.as_str())
    }
}

/// Resource requirements that the host checks before loading a bundle.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BundleResourceRequirementsV1 {
    minimum_device_memory_bytes: u64,
    required_capabilities: Vec<BundleOpaqueId>,
}

impl BundleResourceRequirementsV1 {
    pub fn new(
        minimum_device_memory_bytes: u64,
        mut required_capabilities: Vec<BundleOpaqueId>,
    ) -> Result<Self, BundleError> {
        required_capabilities.sort();
        let requirements = Self {
            minimum_device_memory_bytes,
            required_capabilities,
        };
        requirements.validate()?;
        Ok(requirements)
    }

    #[must_use]
    pub const fn minimum_device_memory_bytes(&self) -> u64 {
        self.minimum_device_memory_bytes
    }

    #[must_use]
    pub fn required_capabilities(&self) -> &[BundleOpaqueId] {
        &self.required_capabilities
    }

    /// These requirements with `capability` among them.
    pub fn with_capability(&self, capability: &BundleOpaqueId) -> Result<Self, BundleError> {
        let mut capabilities = self.required_capabilities.clone();
        if !capabilities.contains(capability) {
            capabilities.push(capability.clone());
        }
        Self::new(self.minimum_device_memory_bytes, capabilities)
    }

    /// These requirements without `capability`.
    pub fn without_capability(&self, capability: &BundleOpaqueId) -> Result<Self, BundleError> {
        let capabilities = self
            .required_capabilities
            .iter()
            .filter(|required| *required != capability)
            .cloned()
            .collect();
        Self::new(self.minimum_device_memory_bytes, capabilities)
    }

    /// These requirements with the capability `formats` need: a mixed-plane
    /// element requires [`MIXED_PLANE_KV_CAPABILITY_V1`].
    pub fn for_kv_formats(&self, formats: &[BundleKvFormatV1]) -> Result<Self, BundleError> {
        if formats
            .iter()
            .any(|format| format.dtype == BundleKvDtypeV1::MixedPlanes)
        {
            self.with_capability(&BundleOpaqueId::new(MIXED_PLANE_KV_CAPABILITY_V1)?)
        } else {
            Ok(self.clone())
        }
    }

    fn validate(&self) -> Result<(), BundleError> {
        if self.minimum_device_memory_bytes == 0 {
            return Err(BundleError::validation(
                "bundle minimum device memory must be greater than zero",
            ));
        }
        validate_count(
            "required capabilities",
            self.required_capabilities.len(),
            MAX_CAPABILITIES,
        )?;
        for capability in &self.required_capabilities {
            validate_opaque_id(capability.as_str())?;
        }
        validate_strict_order("required capabilities", &self.required_capabilities)
    }
}

/// Where one logical KV page is placed across ranks.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum BundleRankPlacementV1 {
    Replicated,
    RankLocal,
    Sharded,
}

/// The lifetime of declared state.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum BundleStateLifetimeV1 {
    Transaction,
    Request,
    Process,
}

/// The element a serving KV layout stores its K and V planes in.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BundleKvDtypeV1 {
    Bf16,
    Fp8E4m3,
    /// Named planes, each with its own code and scale format, which the
    /// format lists (`BundleKvFormatV1::planes`). A manifest that declares it
    /// requires `MIXED_PLANE_KV_CAPABILITY_V1`.
    MixedPlanes,
}

impl BundleKvDtypeV1 {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Bf16 => "bf16",
            Self::Fp8E4m3 => "fp8_e4m3",
            Self::MixedPlanes => "mixed_planes",
        }
    }
}

/// Capability of bundles that declare a mixed-plane serving KV element. A
/// host without it refuses the package by this name.
pub const MIXED_PLANE_KV_CAPABILITY_V1: &str = "chop.kv-cache.mixed-planes.v1";

/// The most planes one mixed-plane KV element declares.
const MAX_KV_PLANES: usize = 16;

/// The code a mixed-plane KV plane stores each value in.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BundleKvCodeV1 {
    Bf16,
    Fp8E4m3,
    Fp4E2m1,
}

impl BundleKvCodeV1 {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Bf16 => "bf16",
            Self::Fp8E4m3 => "fp8_e4m3",
            Self::Fp4E2m1 => "fp4_e2m1",
        }
    }
}

/// The element of a KV plane's block scales.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BundleKvScaleFormatV1 {
    /// A power-of-two exponent byte.
    Ue8m0,
    E4m3,
}

impl BundleKvScaleFormatV1 {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Ue8m0 => "ue8m0",
            Self::E4m3 => "e4m3",
        }
    }
}

/// One scale for each `block` consecutive values of a plane, one byte each,
/// stored from `offset` within each token's bytes.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BundleKvScaleV1 {
    block: u32,
    format: BundleKvScaleFormatV1,
    offset: u64,
}

impl BundleKvScaleV1 {
    pub fn new(
        format: BundleKvScaleFormatV1,
        block: u32,
        offset: u64,
    ) -> Result<Self, BundleError> {
        let scale = Self {
            block,
            format,
            offset,
        };
        scale.validate()?;
        Ok(scale)
    }

    #[must_use]
    pub const fn format(&self) -> BundleKvScaleFormatV1 {
        self.format
    }

    #[must_use]
    pub const fn block(&self) -> u32 {
        self.block
    }

    /// Where the plane's scales start within each token's bytes.
    #[must_use]
    pub const fn offset(&self) -> u64 {
        self.offset
    }

    fn validate(&self) -> Result<(), BundleError> {
        if self.block == 0 {
            return Err(BundleError::validation(
                "bundle KV plane scale block must be greater than zero",
            ));
        }
        Ok(())
    }
}

/// One named plane of a mixed-plane KV element: the `values` each token
/// stores in it, the code they are stored in from `offset` within each
/// token's bytes and, for a scaled code, the format and place of their block
/// scales.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BundleKvPlaneV1 {
    code: BundleKvCodeV1,
    name: BundleOpaqueId,
    offset: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    scale: Option<BundleKvScaleV1>,
    values: u32,
}

impl BundleKvPlaneV1 {
    pub fn new(
        name: BundleOpaqueId,
        code: BundleKvCodeV1,
        values: u32,
        offset: u64,
        scale: Option<BundleKvScaleV1>,
    ) -> Result<Self, BundleError> {
        let plane = Self {
            code,
            name,
            offset,
            scale,
            values,
        };
        plane.validate()?;
        Ok(plane)
    }

    #[must_use]
    pub const fn name(&self) -> &BundleOpaqueId {
        &self.name
    }

    #[must_use]
    pub const fn code(&self) -> BundleKvCodeV1 {
        self.code
    }

    #[must_use]
    pub const fn scale(&self) -> Option<BundleKvScaleV1> {
        self.scale
    }

    /// The values each token stores in this plane.
    #[must_use]
    pub const fn values(&self) -> u32 {
        self.values
    }

    /// Where the plane's codes start within each token's bytes.
    #[must_use]
    pub const fn offset(&self) -> u64 {
        self.offset
    }

    /// The byte ranges of each token's bytes this plane stores: its codes,
    /// then its scales when it has them.
    fn regions(&self) -> Option<Vec<(u64, u64)>> {
        let bits = match self.code {
            BundleKvCodeV1::Bf16 => 16,
            BundleKvCodeV1::Fp8E4m3 => 8,
            BundleKvCodeV1::Fp4E2m1 => 4,
        };
        let codes = u64::from(self.values) * bits / 8;
        let mut regions = vec![(self.offset, self.offset.checked_add(codes)?)];
        if let Some(scale) = self.scale {
            let scales = u64::from(self.values / scale.block);
            regions.push((scale.offset, scale.offset.checked_add(scales)?));
        }
        Some(regions)
    }

    /// The plane as `code` or `code/scale x block`, for messages.
    #[must_use]
    pub fn describe(&self) -> String {
        match self.scale {
            None => format!("{} {}", self.name, self.code.as_str()),
            Some(scale) => format!(
                "{} {}/{}x{}",
                self.name,
                self.code.as_str(),
                scale.format.as_str(),
                scale.block
            ),
        }
    }

    fn validate(&self) -> Result<(), BundleError> {
        validate_opaque_id(self.name.as_str())?;
        if self.values == 0
            || self.code == BundleKvCodeV1::Fp4E2m1 && self.values % 2 != 0
            || self
                .scale
                .is_some_and(|scale| scale.block != 0 && self.values % scale.block != 0)
        {
            return Err(BundleError::validation(format!(
                "bundle KV plane {} must store a positive number of values in whole bytes \
                 and whole scale blocks, not {}",
                self.name, self.values
            )));
        }
        match (self.code, self.scale) {
            (BundleKvCodeV1::Bf16, Some(_)) => Err(BundleError::validation(format!(
                "bundle KV plane {} stores BF16 and takes no scale",
                self.name
            ))),
            (BundleKvCodeV1::Fp4E2m1, None) => Err(BundleError::validation(format!(
                "bundle KV plane {} stores FP4 E2M1 and needs its block scale",
                self.name
            ))),
            (_, Some(scale)) => scale.validate(),
            (_, None) => Ok(()),
        }
    }
}

/// One serving KV layout a bundle can run, with its element. A manifest that
/// lists these lets the host choose the KV cache dtype at startup.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BundleKvFormatV1 {
    dtype: BundleKvDtypeV1,
    layout: BundleKvLayoutV1,
    /// The planes of a mixed-plane element, in the bundle's order; empty for
    /// every other element, so this stays out of the canonical JSON of every
    /// format that predates it.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    planes: Vec<BundleKvPlaneV1>,
}

impl BundleKvFormatV1 {
    /// A BF16 or FP8 E4M3 format; a mixed-plane one comes from
    /// [`Self::mixed_planes`].
    #[must_use]
    pub const fn new(dtype: BundleKvDtypeV1, layout: BundleKvLayoutV1) -> Self {
        Self {
            dtype,
            layout,
            planes: Vec::new(),
        }
    }

    /// A mixed-plane format of `planes`, distinctly named, whose codes and
    /// scales fit apart within each token of `layout`.
    ///
    /// Plane ranges may leave gaps: a token's bytes need not be covered, so
    /// a 32-value FP4 plane with its codes at 0 and its block-32 scale at 16
    /// fits 32 bytes per token and leaves 15 unused. Alignment is the
    /// layout's, checked on the page (`page_bytes` a multiple of
    /// `alignment_bytes`); a plane's offsets carry no alignment rule of
    /// their own.
    pub fn mixed_planes(
        planes: Vec<BundleKvPlaneV1>,
        layout: BundleKvLayoutV1,
    ) -> Result<Self, BundleError> {
        let format = Self {
            dtype: BundleKvDtypeV1::MixedPlanes,
            layout,
            planes,
        };
        format.validate()?;
        Ok(format)
    }

    #[must_use]
    pub const fn dtype(&self) -> BundleKvDtypeV1 {
        self.dtype
    }

    #[must_use]
    pub const fn layout(&self) -> &BundleKvLayoutV1 {
        &self.layout
    }

    /// The planes of a mixed-plane element; empty for every other element.
    #[must_use]
    pub fn planes(&self) -> &[BundleKvPlaneV1] {
        &self.planes
    }

    fn validate(&self) -> Result<(), BundleError> {
        self.layout.validate()?;
        let mixed = self.dtype == BundleKvDtypeV1::MixedPlanes;
        if mixed == self.planes.is_empty() {
            return Err(BundleError::validation(format!(
                "bundle KV format {} must declare planes exactly when its element is mixed_planes",
                self.layout.id
            )));
        }
        validate_count("KV planes", self.planes.len(), MAX_KV_PLANES)?;
        for (index, plane) in self.planes.iter().enumerate() {
            plane.validate()?;
            if self.planes[..index]
                .iter()
                .any(|other| other.name == plane.name)
            {
                return Err(BundleError::validation(format!(
                    "bundle KV format {} declares plane {} twice",
                    self.layout.id, plane.name
                )));
            }
        }
        self.validate_plane_bytes()
    }

    /// Every plane's codes and scales lie within each token's
    /// `bytes_per_token` and share no byte with another's, so the page the
    /// layout allocates holds every plane it declares.
    fn validate_plane_bytes(&self) -> Result<(), BundleError> {
        let mut regions = Vec::new();
        for plane in &self.planes {
            let within = plane.regions().filter(|regions| {
                regions
                    .iter()
                    .all(|&(_, end)| end <= self.layout.bytes_per_token)
            });
            let Some(within) = within else {
                return Err(BundleError::validation(format!(
                    "bundle KV plane {} of {} values does not fit the {} bytes per token \
                     of KV layout {}",
                    plane.name, plane.values, self.layout.bytes_per_token, self.layout.id
                )));
            };
            regions.extend(within.into_iter().map(|region| (region, &plane.name)));
        }
        regions.sort();
        for pair in regions.windows(2) {
            let (((_, end), first), ((start, _), second)) = (pair[0], pair[1]);
            if start < end {
                return Err(BundleError::validation(format!(
                    "bundle KV planes {first} and {second} of KV layout {} share bytes",
                    self.layout.id
                )));
            }
        }
        Ok(())
    }
}

/// A generic, bundle-declared KV layout.
///
/// One page holds `page_tokens` tokens in `segments_per_page` equally shaped
/// segments of `bytes_per_token` per token. The declared `page_bytes` must
/// equal that product exactly; the runtime never infers it.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BundleKvLayoutV1 {
    alignment_bytes: u64,
    bytes_per_token: u64,
    id: BundleOpaqueId,
    maximum_sequence_tokens: u64,
    page_bytes: u64,
    page_tokens: u32,
    rank_placement: BundleRankPlacementV1,
    segments_per_page: u32,
    state_lifetime: BundleStateLifetimeV1,
}

impl BundleKvLayoutV1 {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        id: BundleOpaqueId,
        page_bytes: u64,
        alignment_bytes: u64,
        page_tokens: u32,
        bytes_per_token: u64,
        segments_per_page: u32,
        maximum_sequence_tokens: u64,
        rank_placement: BundleRankPlacementV1,
        state_lifetime: BundleStateLifetimeV1,
    ) -> Result<Self, BundleError> {
        let layout = Self {
            alignment_bytes,
            bytes_per_token,
            id,
            maximum_sequence_tokens,
            page_bytes,
            page_tokens,
            rank_placement,
            segments_per_page,
            state_lifetime,
        };
        layout.validate()?;
        Ok(layout)
    }

    #[must_use]
    pub const fn id(&self) -> &BundleOpaqueId {
        &self.id
    }

    #[must_use]
    pub const fn page_bytes(&self) -> u64 {
        self.page_bytes
    }

    #[must_use]
    pub const fn alignment_bytes(&self) -> u64 {
        self.alignment_bytes
    }

    #[must_use]
    pub const fn page_tokens(&self) -> u32 {
        self.page_tokens
    }

    #[must_use]
    pub const fn bytes_per_token(&self) -> u64 {
        self.bytes_per_token
    }

    #[must_use]
    pub const fn segments_per_page(&self) -> u32 {
        self.segments_per_page
    }

    #[must_use]
    pub const fn maximum_sequence_tokens(&self) -> u64 {
        self.maximum_sequence_tokens
    }

    #[must_use]
    pub const fn rank_placement(&self) -> BundleRankPlacementV1 {
        self.rank_placement
    }

    #[must_use]
    pub const fn state_lifetime(&self) -> BundleStateLifetimeV1 {
        self.state_lifetime
    }

    fn validate(&self) -> Result<(), BundleError> {
        validate_opaque_id(self.id.as_str())?;
        validate_aligned_size("KV page", self.page_bytes, self.alignment_bytes)?;
        if self.page_tokens == 0 {
            return Err(BundleError::validation(
                "bundle KV page tokens must be greater than zero",
            ));
        }
        if self.bytes_per_token == 0 {
            return Err(BundleError::validation(
                "bundle KV bytes per token must be greater than zero",
            ));
        }
        if self.segments_per_page == 0 {
            return Err(BundleError::validation(
                "bundle KV segments per page must be greater than zero",
            ));
        }
        if self.maximum_sequence_tokens == 0 {
            return Err(BundleError::validation(
                "bundle KV maximum sequence tokens must be greater than zero",
            ));
        }
        let declared = self
            .bytes_per_token
            .checked_mul(u64::from(self.page_tokens))
            .and_then(|segment| segment.checked_mul(u64::from(self.segments_per_page)))
            .ok_or_else(|| {
                BundleError::validation("bundle KV page geometry overflows u64".to_owned())
            })?;
        if declared != self.page_bytes {
            return Err(BundleError::validation(format!(
                "bundle KV page of {} bytes does not match {} bytes per token \
                 across {} tokens and {} segments",
                self.page_bytes, self.bytes_per_token, self.page_tokens, self.segments_per_page
            )));
        }
        Ok(())
    }

    /// Builds the runtime page layout for one declared topology.
    pub fn declared_page_layout(
        &self,
        topology: &BundleTopologyV1,
    ) -> Result<DeclaredPageLayoutV1, BundleError> {
        topology.validate()?;
        self.validate()?;
        let bytes_per_token = u32::try_from(self.bytes_per_token).map_err(|_| {
            BundleError::validation(format!(
                "bundle KV bytes per token {} exceeds the runtime page layout bound",
                self.bytes_per_token
            ))
        })?;
        let rank_count = u8::try_from(topology.rank_count()).map_err(|_| {
            BundleError::validation(format!(
                "bundle topology {} rank count {} exceeds the runtime page layout bound",
                topology.id(),
                topology.rank_count()
            ))
        })?;
        let maximum_sequence_tokens =
            usize::try_from(self.maximum_sequence_tokens).map_err(|_| {
                BundleError::validation(format!(
                    "bundle KV maximum sequence tokens {} is not representable on this host",
                    self.maximum_sequence_tokens
                ))
            })?;
        DeclaredPageLayoutV1::new(
            bytes_per_token,
            self.segments_per_page,
            self.page_tokens,
            self.alignment_bytes,
            rank_count,
            self.rank_placement.into(),
            self.state_lifetime.into(),
            maximum_sequence_tokens,
        )
        .map_err(|error| {
            BundleError::validation(format!("bundle KV layout is not a runtime layout: {error}"))
        })
    }
}

impl From<BundleRankPlacementV1> for RankPlacementV1 {
    fn from(placement: BundleRankPlacementV1) -> Self {
        match placement {
            BundleRankPlacementV1::Replicated => Self::Replicated,
            BundleRankPlacementV1::RankLocal => Self::RankLocal,
            BundleRankPlacementV1::Sharded => Self::Sharded,
        }
    }
}

impl From<BundleStateLifetimeV1> for StateLifetimeV1 {
    fn from(lifetime: BundleStateLifetimeV1) -> Self {
        match lifetime {
            BundleStateLifetimeV1::Transaction => Self::Transaction,
            BundleStateLifetimeV1::Request => Self::Request,
            BundleStateLifetimeV1::Process => Self::Process,
        }
    }
}

impl From<BundleImplementationTierV1> for KernelImplementationTierV1 {
    fn from(tier: BundleImplementationTierV1) -> Self {
        match tier {
            BundleImplementationTierV1::Bundle => Self::Bundle,
            BundleImplementationTierV1::HostOptimized => Self::HostOptimized,
            BundleImplementationTierV1::Generic => Self::Generic,
        }
    }
}

/// A generic, bundle-declared persistent state layout.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BundleStateLayoutV1 {
    id: BundleOpaqueId,
    bytes: u64,
    alignment_bytes: u64,
    lifetime: BundleStateLifetimeV1,
}

impl BundleStateLayoutV1 {
    pub fn new(
        id: BundleOpaqueId,
        bytes: u64,
        alignment_bytes: u64,
        lifetime: BundleStateLifetimeV1,
    ) -> Result<Self, BundleError> {
        let layout = Self {
            id,
            bytes,
            alignment_bytes,
            lifetime,
        };
        layout.validate()?;
        Ok(layout)
    }

    #[must_use]
    pub const fn id(&self) -> &BundleOpaqueId {
        &self.id
    }

    #[must_use]
    pub const fn bytes(&self) -> u64 {
        self.bytes
    }

    #[must_use]
    pub const fn alignment_bytes(&self) -> u64 {
        self.alignment_bytes
    }

    #[must_use]
    pub const fn lifetime(&self) -> BundleStateLifetimeV1 {
        self.lifetime
    }

    fn validate(&self) -> Result<(), BundleError> {
        validate_opaque_id(self.id.as_str())?;
        validate_aligned_size("state", self.bytes, self.alignment_bytes)
    }
}

/// An opaque operation key used by requirements and implementations.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BundleOperationKeyV1 {
    operation: BundleOpaqueId,
    variant: BundleOpaqueId,
}

impl BundleOperationKeyV1 {
    #[must_use]
    pub const fn new(operation: BundleOpaqueId, variant: BundleOpaqueId) -> Self {
        Self { operation, variant }
    }

    #[must_use]
    pub const fn operation(&self) -> &BundleOpaqueId {
        &self.operation
    }

    #[must_use]
    pub const fn variant(&self) -> &BundleOpaqueId {
        &self.variant
    }

    fn validate(&self) -> Result<(), BundleError> {
        validate_opaque_id(self.operation.as_str())?;
        validate_opaque_id(self.variant.as_str())
    }
}

/// The kind of implementation expected for a required operation.
#[derive(Clone, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
#[serde(
    tag = "kind",
    content = "primitive_id",
    rename_all = "kebab-case",
    deny_unknown_fields
)]
pub enum BundleOperationProvisioningV1 {
    BundleCustomOperation,
    HostGenericPrimitive(BundleOpaqueId),
}

impl BundleOperationProvisioningV1 {
    #[must_use]
    pub const fn primitive_id(&self) -> Option<&BundleOpaqueId> {
        match self {
            Self::BundleCustomOperation => None,
            Self::HostGenericPrimitive(primitive_id) => Some(primitive_id),
        }
    }

    fn validate(&self) -> Result<(), BundleError> {
        match self {
            Self::BundleCustomOperation => Ok(()),
            Self::HostGenericPrimitive(primitive_id) => validate_opaque_id(primitive_id.as_str()),
        }
    }
}

/// How one declared operand is read or written.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum BundleTensorAccessV1 {
    ReadOnly,
    WriteOnly,
    ReadWrite,
}

impl From<BundleTensorAccessV1> for KernelTensorAccessV1 {
    fn from(access: BundleTensorAccessV1) -> Self {
        match access {
            BundleTensorAccessV1::ReadOnly => Self::ReadOnly,
            BundleTensorAccessV1::WriteOnly => Self::WriteOnly,
            BundleTensorAccessV1::ReadWrite => Self::ReadWrite,
        }
    }
}

/// Whether one declared operand may share storage with another.
#[derive(Clone, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
#[serde(
    tag = "kind",
    content = "group",
    rename_all = "kebab-case",
    deny_unknown_fields
)]
pub enum BundleTensorAliasV1 {
    Disjoint,
    AliasGroup(BundleOpaqueId),
}

impl BundleTensorAliasV1 {
    fn validate(&self) -> Result<(), BundleError> {
        match self {
            Self::Disjoint => Ok(()),
            Self::AliasGroup(group) => validate_opaque_id(group.as_str()),
        }
    }

    fn kernel_alias(&self) -> Result<KernelAliasContractV1, BundleError> {
        Ok(match self {
            Self::Disjoint => KernelAliasContractV1::Disjoint,
            Self::AliasGroup(group) => KernelAliasContractV1::AliasGroup(kernel_id(group)?),
        })
    }
}

/// An ordering constraint from one declared operand onto an earlier one.
#[derive(Clone, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BundleTensorDependencyV1 {
    dependency: BundleOpaqueId,
    tensor_index: u16,
}

impl BundleTensorDependencyV1 {
    #[must_use]
    pub const fn new(tensor_index: u16, dependency: BundleOpaqueId) -> Self {
        Self {
            dependency,
            tensor_index,
        }
    }

    #[must_use]
    pub const fn tensor_index(&self) -> u16 {
        self.tensor_index
    }

    #[must_use]
    pub const fn dependency(&self) -> &BundleOpaqueId {
        &self.dependency
    }

    fn validate(&self) -> Result<(), BundleError> {
        validate_opaque_id(self.dependency.as_str())
    }
}

/// One declared operand of a required operation.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BundleTensorRequirementV1 {
    access: BundleTensorAccessV1,
    alias: BundleTensorAliasV1,
    alignment_bytes: u64,
    dependencies: Vec<BundleTensorDependencyV1>,
    dimensions: Vec<u64>,
    dtype: BundleOpaqueId,
    layout: BundleOpaqueId,
    role: BundleOpaqueId,
    scale_block_dimensions: Vec<u64>,
    scale_format: BundleOpaqueId,
}

impl BundleTensorRequirementV1 {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        role: BundleOpaqueId,
        dimensions: Vec<u64>,
        dtype: BundleOpaqueId,
        layout: BundleOpaqueId,
        scale_format: BundleOpaqueId,
        scale_block_dimensions: Vec<u64>,
        alignment_bytes: u64,
        access: BundleTensorAccessV1,
        alias: BundleTensorAliasV1,
        dependencies: Vec<BundleTensorDependencyV1>,
    ) -> Result<Self, BundleError> {
        let requirement = Self {
            access,
            alias,
            alignment_bytes,
            dependencies,
            dimensions,
            dtype,
            layout,
            role,
            scale_block_dimensions,
            scale_format,
        };
        requirement.validate()?;
        Ok(requirement)
    }

    #[must_use]
    pub const fn role(&self) -> &BundleOpaqueId {
        &self.role
    }

    #[must_use]
    pub fn dimensions(&self) -> &[u64] {
        &self.dimensions
    }

    #[must_use]
    pub const fn dtype(&self) -> &BundleOpaqueId {
        &self.dtype
    }

    #[must_use]
    pub const fn layout(&self) -> &BundleOpaqueId {
        &self.layout
    }

    #[must_use]
    pub const fn scale_format(&self) -> &BundleOpaqueId {
        &self.scale_format
    }

    #[must_use]
    pub fn scale_block_dimensions(&self) -> &[u64] {
        &self.scale_block_dimensions
    }

    #[must_use]
    pub const fn alignment_bytes(&self) -> u64 {
        self.alignment_bytes
    }

    #[must_use]
    pub const fn access(&self) -> BundleTensorAccessV1 {
        self.access
    }

    #[must_use]
    pub const fn alias(&self) -> &BundleTensorAliasV1 {
        &self.alias
    }

    #[must_use]
    pub fn dependencies(&self) -> &[BundleTensorDependencyV1] {
        &self.dependencies
    }

    fn validate(&self) -> Result<(), BundleError> {
        validate_opaque_id(self.role.as_str())?;
        validate_opaque_id(self.dtype.as_str())?;
        validate_opaque_id(self.layout.as_str())?;
        validate_opaque_id(self.scale_format.as_str())?;
        self.alias.validate()?;
        validate_nonempty_count(
            "tensor dimensions",
            self.dimensions.len(),
            MAX_TENSOR_DIMENSIONS,
        )?;
        if self.dimensions.contains(&0) {
            return Err(BundleError::validation(format!(
                "bundle tensor {} has a zero dimension",
                self.role
            )));
        }
        validate_count(
            "tensor scale block dimensions",
            self.scale_block_dimensions.len(),
            MAX_TENSOR_DIMENSIONS,
        )?;
        if self.scale_block_dimensions.contains(&0) {
            return Err(BundleError::validation(format!(
                "bundle tensor {} has a zero scale block dimension",
                self.role
            )));
        }
        if self.alignment_bytes == 0 || !self.alignment_bytes.is_power_of_two() {
            return Err(BundleError::validation(format!(
                "bundle tensor {} alignment {} is not a nonzero power of two",
                self.role, self.alignment_bytes
            )));
        }
        validate_count(
            "tensor dependencies",
            self.dependencies.len(),
            MAX_TENSOR_DEPENDENCIES,
        )?;
        for dependency in &self.dependencies {
            dependency.validate()?;
        }
        Ok(())
    }

    fn kernel_requirement(&self) -> Result<KernelTensorRequirementV1, BundleError> {
        let requirement = KernelTensorRequirementV1::new(
            kernel_id(&self.role)?,
            self.dimensions.clone(),
            kernel_id(&self.dtype)?,
            kernel_id(&self.layout)?,
            kernel_id(&self.scale_format)?,
            self.alignment_bytes,
        )
        .map_err(|error| {
            BundleError::validation(format!(
                "bundle tensor {} is not a semantic requirement: {error}",
                self.role
            ))
        })?
        .with_scale_block_dimensions(self.scale_block_dimensions.clone())
        .map_err(|error| {
            BundleError::validation(format!(
                "bundle tensor {} has an invalid scale contract: {error}",
                self.role
            ))
        })?
        .with_access(self.access.into(), self.alias.kernel_alias()?);
        let mut dependencies = Vec::with_capacity(self.dependencies.len());
        for dependency in &self.dependencies {
            dependencies.push(KernelTensorDependencyV1::new(
                dependency.tensor_index(),
                kernel_id(dependency.dependency())?,
            ));
        }
        Ok(requirement.with_dependencies(dependencies))
    }
}

/// One declared scalar argument of a required operation.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BundleScalarArgumentV1 {
    argument: BundleOpaqueId,
    dtype: BundleOpaqueId,
    width_bits: u16,
}

impl BundleScalarArgumentV1 {
    pub fn new(
        argument: BundleOpaqueId,
        dtype: BundleOpaqueId,
        width_bits: u16,
    ) -> Result<Self, BundleError> {
        let scalar = Self {
            argument,
            dtype,
            width_bits,
        };
        scalar.validate()?;
        Ok(scalar)
    }

    #[must_use]
    pub const fn argument(&self) -> &BundleOpaqueId {
        &self.argument
    }

    #[must_use]
    pub const fn dtype(&self) -> &BundleOpaqueId {
        &self.dtype
    }

    #[must_use]
    pub const fn width_bits(&self) -> u16 {
        self.width_bits
    }

    fn validate(&self) -> Result<(), BundleError> {
        validate_opaque_id(self.argument.as_str())?;
        validate_opaque_id(self.dtype.as_str())?;
        if self.width_bits == 0 || !self.width_bits.is_multiple_of(8) {
            return Err(BundleError::validation(format!(
                "bundle scalar {} width {} is not whole nonzero bytes",
                self.argument, self.width_bits
            )));
        }
        Ok(())
    }

    fn kernel_argument(&self) -> Result<KernelScalarArgumentV1, BundleError> {
        KernelScalarArgumentV1::new(
            kernel_id(&self.argument)?,
            kernel_id(&self.dtype)?,
            self.width_bits,
        )
        .map_err(|error| {
            BundleError::validation(format!(
                "bundle scalar {} is not a semantic argument: {error}",
                self.argument
            ))
        })
    }
}

/// The complete semantic contract of one required operation. This is what a
/// model-independent host matches on; nothing outside the manifest is needed.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BundleSemanticContractV1 {
    numerical_contract: BundleOpaqueId,
    operation_abi: BundleOpaqueId,
    operation_abi_version: u32,
    required_capabilities: Vec<BundleOpaqueId>,
    scalar_arguments: Vec<BundleScalarArgumentV1>,
    tensors: Vec<BundleTensorRequirementV1>,
}

impl BundleSemanticContractV1 {
    pub fn new(
        operation_abi: BundleOpaqueId,
        operation_abi_version: u32,
        tensors: Vec<BundleTensorRequirementV1>,
        scalar_arguments: Vec<BundleScalarArgumentV1>,
        numerical_contract: BundleOpaqueId,
        mut required_capabilities: Vec<BundleOpaqueId>,
    ) -> Result<Self, BundleError> {
        required_capabilities.sort();
        let contract = Self {
            numerical_contract,
            operation_abi,
            operation_abi_version,
            required_capabilities,
            scalar_arguments,
            tensors,
        };
        contract.validate()?;
        Ok(contract)
    }

    #[must_use]
    pub const fn operation_abi(&self) -> &BundleOpaqueId {
        &self.operation_abi
    }

    #[must_use]
    pub const fn operation_abi_version(&self) -> u32 {
        self.operation_abi_version
    }

    #[must_use]
    pub fn tensors(&self) -> &[BundleTensorRequirementV1] {
        &self.tensors
    }

    #[must_use]
    pub fn scalar_arguments(&self) -> &[BundleScalarArgumentV1] {
        &self.scalar_arguments
    }

    #[must_use]
    pub const fn numerical_contract(&self) -> &BundleOpaqueId {
        &self.numerical_contract
    }

    #[must_use]
    pub fn required_capabilities(&self) -> &[BundleOpaqueId] {
        &self.required_capabilities
    }

    fn validate(&self) -> Result<(), BundleError> {
        validate_opaque_id(self.operation_abi.as_str())?;
        validate_opaque_id(self.numerical_contract.as_str())?;
        if self.operation_abi_version == 0 {
            return Err(BundleError::validation(format!(
                "bundle operation ABI {} has version zero",
                self.operation_abi
            )));
        }
        validate_nonempty_count(
            "operation tensors",
            self.tensors.len(),
            MAX_OPERATION_TENSORS,
        )?;
        validate_count(
            "operation scalar arguments",
            self.scalar_arguments.len(),
            MAX_OPERATION_SCALARS,
        )?;
        validate_count(
            "operation required capabilities",
            self.required_capabilities.len(),
            MAX_CAPABILITIES,
        )?;
        for tensor in &self.tensors {
            tensor.validate()?;
        }
        for argument in &self.scalar_arguments {
            argument.validate()?;
        }
        for capability in &self.required_capabilities {
            validate_opaque_id(capability.as_str())?;
        }
        validate_strict_order(
            "operation required capabilities",
            &self.required_capabilities,
        )?;
        // The semantic key is the contract. Building it here means a manifest
        // that parses is a manifest the resolver can use.
        self.semantic_key().map(|_| ())
    }

    /// Converts this declaration into the shared semantic key.
    pub fn semantic_key(&self) -> Result<KernelSemanticKeyV1, BundleError> {
        let mut tensors = Vec::with_capacity(self.tensors.len());
        for tensor in &self.tensors {
            tensors.push(tensor.kernel_requirement()?);
        }
        let mut scalar_arguments = Vec::with_capacity(self.scalar_arguments.len());
        for argument in &self.scalar_arguments {
            scalar_arguments.push(argument.kernel_argument()?);
        }
        let mut required_capabilities = BTreeSet::new();
        for capability in &self.required_capabilities {
            required_capabilities.insert(kernel_id(capability)?);
        }
        KernelSemanticKeyV1::new(KernelSemanticKeyPartsV1 {
            operation_abi: KernelOperationAbiV1::new(
                kernel_id(&self.operation_abi)?,
                self.operation_abi_version,
            )
            .map_err(|error| {
                BundleError::validation(format!(
                    "bundle operation ABI {} is invalid: {error}",
                    self.operation_abi
                ))
            })?,
            tensors,
            scalar_arguments,
            numerical_contract: kernel_id(&self.numerical_contract)?,
            required_capabilities,
        })
        .map_err(|error| {
            BundleError::validation(format!(
                "bundle operation {} has no valid semantic contract: {error}",
                self.operation_abi
            ))
        })
    }
}

/// A required semantic operation and the complete contract it must satisfy.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BundleOperationRequirementV1 {
    key: BundleOperationKeyV1,
    provisioning: BundleOperationProvisioningV1,
    semantic_contract: BundleSemanticContractV1,
}

impl BundleOperationRequirementV1 {
    pub fn new(
        key: BundleOperationKeyV1,
        provisioning: BundleOperationProvisioningV1,
        semantic_contract: BundleSemanticContractV1,
    ) -> Result<Self, BundleError> {
        let requirement = Self {
            key,
            provisioning,
            semantic_contract,
        };
        requirement.validate()?;
        Ok(requirement)
    }

    #[must_use]
    pub const fn key(&self) -> &BundleOperationKeyV1 {
        &self.key
    }

    #[must_use]
    pub const fn provisioning(&self) -> &BundleOperationProvisioningV1 {
        &self.provisioning
    }

    #[must_use]
    pub const fn semantic_contract(&self) -> &BundleSemanticContractV1 {
        &self.semantic_contract
    }

    fn validate(&self) -> Result<(), BundleError> {
        self.key.validate()?;
        self.provisioning.validate()?;
        self.semantic_contract.validate()
    }
}

/// The provider tier of one concrete implementation.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum BundleImplementationTierV1 {
    Bundle,
    HostOptimized,
    Generic,
}

/// Which file of the closed package an implementation's artifact is.
///
/// A bundle carries two kinds of file an implementation can be built from: a
/// declared component, and the package's own native module. Both are named
/// explicitly, because the difference is provenance a host acts on: a record
/// pointing at a component describes bytes the host uploads, and one pointing
/// at the native module describes the code the host is about to call.
#[derive(Clone, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
#[serde(
    tag = "kind",
    content = "component",
    rename_all = "kebab-case",
    deny_unknown_fields
)]
pub enum BundleArtifactOriginV1 {
    /// One declared component, by identifier.
    Component(BundleOpaqueId),
    /// The package's own native module, which it declares exactly one of.
    NativeModule,
}

impl BundleArtifactOriginV1 {
    /// The component this origin names, where it names one.
    #[must_use]
    pub const fn component(&self) -> Option<&BundleOpaqueId> {
        match self {
            Self::Component(component) => Some(component),
            Self::NativeModule => None,
        }
    }

    fn validate(&self) -> Result<(), BundleError> {
        match self {
            Self::Component(component) => validate_opaque_id(component.as_str()),
            Self::NativeModule => Ok(()),
        }
    }
}

/// The artifact format every native-module-backed implementation declares.
///
/// A native module has one format: it is the module this package ships, opened
/// through the V1 entry symbol. Requiring the name here is what lets validation
/// compare a native-backed record against the manifest's own native module
/// record without a second declaration of what that file is.
pub const BUNDLE_NATIVE_MODULE_FORMAT_V1: &str = "chop.bundle.native-module.v1";

/// The exact artifact one implementation is built from, anchored to the
/// package file that carries it.
///
/// The runtime descriptor keeps its own format, length, and digest, but the
/// manifest also names which file of the closed set those must equal, so a
/// verified package can never select an artifact outside it. An implementation
/// whose executable code is compiled into the package's native module says so,
/// rather than naming a component that only happens to travel beside it.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BundleArtifactIdentityV1 {
    byte_length: u64,
    format: BundleOpaqueId,
    origin: BundleArtifactOriginV1,
    sha256: String,
}

impl BundleArtifactIdentityV1 {
    /// One artifact carried by a declared component.
    pub fn from_component(
        component: BundleOpaqueId,
        format: BundleOpaqueId,
        byte_length: u64,
        sha256: impl Into<String>,
    ) -> Result<Self, BundleError> {
        Self::declare(
            BundleArtifactOriginV1::Component(component),
            format,
            byte_length,
            sha256,
        )
    }

    /// One artifact that is the package's own native module.
    ///
    /// The format is not a parameter: a native module has one, and validation
    /// compares the length and digest against the manifest's own record of the
    /// file, so this can never describe bytes the package does not ship.
    pub fn from_native_module(
        byte_length: u64,
        sha256: impl Into<String>,
    ) -> Result<Self, BundleError> {
        Self::declare(
            BundleArtifactOriginV1::NativeModule,
            BundleOpaqueId::new(BUNDLE_NATIVE_MODULE_FORMAT_V1)?,
            byte_length,
            sha256,
        )
    }

    fn declare(
        origin: BundleArtifactOriginV1,
        format: BundleOpaqueId,
        byte_length: u64,
        sha256: impl Into<String>,
    ) -> Result<Self, BundleError> {
        let artifact = Self {
            byte_length,
            format,
            origin,
            sha256: sha256.into(),
        };
        artifact.validate()?;
        Ok(artifact)
    }

    /// Which file of the closed package set this artifact is.
    #[must_use]
    pub const fn origin(&self) -> &BundleArtifactOriginV1 {
        &self.origin
    }

    /// The component this artifact names, where it names one.
    #[must_use]
    pub const fn component(&self) -> Option<&BundleOpaqueId> {
        self.origin.component()
    }

    #[must_use]
    pub const fn format(&self) -> &BundleOpaqueId {
        &self.format
    }

    #[must_use]
    pub const fn byte_length(&self) -> u64 {
        self.byte_length
    }

    #[must_use]
    pub fn sha256(&self) -> &str {
        &self.sha256
    }

    fn validate(&self) -> Result<(), BundleError> {
        self.origin.validate()?;
        validate_opaque_id(self.format.as_str())?;
        if matches!(self.origin, BundleArtifactOriginV1::NativeModule)
            && self.format.as_str() != BUNDLE_NATIVE_MODULE_FORMAT_V1
        {
            return Err(BundleError::validation(format!(
                "a native-module artifact declares format {}, and a native module has format \
                 {BUNDLE_NATIVE_MODULE_FORMAT_V1}",
                self.format
            )));
        }
        validate_declared_file(
            "implementation artifact",
            self.byte_length,
            MAX_COMPONENT_BYTES,
        )?;
        validate_sha256(&self.sha256)
    }

    /// Checks that this identity is exactly the package file it names.
    ///
    /// A component-backed artifact must name one declared component and carry
    /// that component's format, length, and digest. A native-backed one must
    /// find a native module in the package at all, and carry the length and
    /// digest the manifest recorded for it. Either way the record describes a
    /// file this package ships, measured, rather than a label beside it.
    fn verify_origin_linkage(
        &self,
        implementation: &BundleOpaqueId,
        components: &[BundleComponentV1],
        native_module: Option<&BundleNativeModuleV1>,
    ) -> Result<(), BundleError> {
        let component = match &self.origin {
            BundleArtifactOriginV1::NativeModule => {
                return self.verify_native_module_linkage(implementation, native_module);
            }
            BundleArtifactOriginV1::Component(component) => component,
        };
        let mut matches = components
            .iter()
            .filter(|candidate| candidate.id() == component);
        let Some(declared) = matches.next() else {
            return Err(BundleError::validation(format!(
                "bundle implementation {implementation} references unknown artifact component \
                 {component}"
            )));
        };
        if matches.next().is_some() {
            return Err(BundleError::validation(format!(
                "bundle implementation {implementation} references ambiguous artifact component \
                 {component}"
            )));
        }
        if declared.format() != &self.format
            || declared.byte_length() != self.byte_length
            || declared.sha256() != self.sha256
        {
            return Err(BundleError::validation(format!(
                "bundle implementation {implementation} artifact does not match component \
                 {component}: declared {} of {} bytes with SHA-256 {}, component is {} of {} \
                 bytes with SHA-256 {}",
                self.format,
                self.byte_length,
                self.sha256,
                declared.format(),
                declared.byte_length(),
                declared.sha256()
            )));
        }
        Ok(())
    }

    fn verify_native_module_linkage(
        &self,
        implementation: &BundleOpaqueId,
        native_module: Option<&BundleNativeModuleV1>,
    ) -> Result<(), BundleError> {
        let Some(native_module) = native_module else {
            return Err(BundleError::validation(format!(
                "bundle implementation {implementation} is backed by a native module and this \
                 package declares none"
            )));
        };
        if native_module.byte_length() != self.byte_length || native_module.sha256() != self.sha256
        {
            return Err(BundleError::validation(format!(
                "bundle implementation {implementation} artifact does not match this package's \
                 native module: declared {} bytes with SHA-256 {}, the module is {} bytes with \
                 SHA-256 {}",
                self.byte_length,
                self.sha256,
                native_module.byte_length(),
                native_module.sha256()
            )));
        }
        Ok(())
    }

    fn kernel_artifact(&self) -> Result<KernelArtifactIdentityV1, BundleError> {
        KernelArtifactIdentityV1::new(
            kernel_id(&self.format)?,
            self.byte_length,
            kernel_sha256(&self.sha256)?,
        )
        .map_err(|error| {
            BundleError::validation(format!("bundle artifact identity is invalid: {error}"))
        })
    }
}

/// The exact source one implementation was compiled from.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BundleSourceIdentityV1 {
    byte_length: u64,
    path: BundleOpaqueId,
    revision: BundleOpaqueId,
    sha256: String,
    source: BundleOpaqueId,
}

impl BundleSourceIdentityV1 {
    pub fn new(
        source: BundleOpaqueId,
        revision: BundleOpaqueId,
        path: BundleOpaqueId,
        byte_length: u64,
        sha256: impl Into<String>,
    ) -> Result<Self, BundleError> {
        let identity = Self {
            byte_length,
            path,
            revision,
            sha256: sha256.into(),
            source,
        };
        identity.validate()?;
        Ok(identity)
    }

    #[must_use]
    pub const fn source(&self) -> &BundleOpaqueId {
        &self.source
    }

    #[must_use]
    pub const fn revision(&self) -> &BundleOpaqueId {
        &self.revision
    }

    #[must_use]
    pub const fn path(&self) -> &BundleOpaqueId {
        &self.path
    }

    #[must_use]
    pub const fn byte_length(&self) -> u64 {
        self.byte_length
    }

    #[must_use]
    pub fn sha256(&self) -> &str {
        &self.sha256
    }

    fn validate(&self) -> Result<(), BundleError> {
        validate_opaque_id(self.source.as_str())?;
        validate_opaque_id(self.revision.as_str())?;
        validate_opaque_id(self.path.as_str())?;
        validate_declared_file(
            "implementation source",
            self.byte_length,
            MAX_COMPONENT_BYTES,
        )?;
        validate_sha256(&self.sha256)
    }

    fn kernel_source(&self) -> Result<KernelSourceIdentityV1, BundleError> {
        KernelSourceIdentityV1::new(
            kernel_id(&self.source)?,
            kernel_id(&self.revision)?,
            kernel_id(&self.path)?,
            self.byte_length,
            kernel_sha256(&self.sha256)?,
        )
        .map_err(|error| {
            BundleError::validation(format!("bundle source identity is invalid: {error}"))
        })
    }
}

/// The exact toolchain and recipe one implementation was built with.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BundleBuildIdentityV1 {
    build: BundleOpaqueId,
    recipe_sha256: String,
    toolchain: BundleOpaqueId,
    toolchain_version: BundleOpaqueId,
}

impl BundleBuildIdentityV1 {
    pub fn new(
        toolchain: BundleOpaqueId,
        toolchain_version: BundleOpaqueId,
        build: BundleOpaqueId,
        recipe_sha256: impl Into<String>,
    ) -> Result<Self, BundleError> {
        let identity = Self {
            build,
            recipe_sha256: recipe_sha256.into(),
            toolchain,
            toolchain_version,
        };
        identity.validate()?;
        Ok(identity)
    }

    #[must_use]
    pub const fn toolchain(&self) -> &BundleOpaqueId {
        &self.toolchain
    }

    #[must_use]
    pub const fn toolchain_version(&self) -> &BundleOpaqueId {
        &self.toolchain_version
    }

    #[must_use]
    pub const fn build(&self) -> &BundleOpaqueId {
        &self.build
    }

    #[must_use]
    pub fn recipe_sha256(&self) -> &str {
        &self.recipe_sha256
    }

    fn validate(&self) -> Result<(), BundleError> {
        validate_opaque_id(self.toolchain.as_str())?;
        validate_opaque_id(self.toolchain_version.as_str())?;
        validate_opaque_id(self.build.as_str())?;
        validate_sha256(&self.recipe_sha256)
    }

    fn kernel_build(&self) -> Result<KernelBuildIdentityV1, BundleError> {
        Ok(KernelBuildIdentityV1::new(
            kernel_id(&self.toolchain)?,
            kernel_id(&self.toolchain_version)?,
            kernel_id(&self.build)?,
            kernel_sha256(&self.recipe_sha256)?,
        ))
    }
}

/// The exact launch geometry one implementation runs with.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BundleLaunchContractV1 {
    block: [u32; 3],
    dynamic_shared_memory_bytes: u32,
    grid: [u32; 3],
}

impl BundleLaunchContractV1 {
    pub fn new(
        grid: [u32; 3],
        block: [u32; 3],
        dynamic_shared_memory_bytes: u32,
    ) -> Result<Self, BundleError> {
        let launch = Self {
            block,
            dynamic_shared_memory_bytes,
            grid,
        };
        launch.validate()?;
        Ok(launch)
    }

    #[must_use]
    pub const fn grid(&self) -> [u32; 3] {
        self.grid
    }

    #[must_use]
    pub const fn block(&self) -> [u32; 3] {
        self.block
    }

    #[must_use]
    pub const fn dynamic_shared_memory_bytes(&self) -> u32 {
        self.dynamic_shared_memory_bytes
    }

    fn validate(&self) -> Result<(), BundleError> {
        if self.grid.contains(&0) || self.block.contains(&0) {
            return Err(BundleError::validation(
                "bundle launch geometry has a zero dimension",
            ));
        }
        Ok(())
    }

    fn kernel_launch(&self) -> Result<KernelLaunchContractV1, BundleError> {
        KernelLaunchContractV1::new(self.grid, self.block, self.dynamic_shared_memory_bytes)
            .map_err(|error| {
                BundleError::validation(format!("bundle launch contract is invalid: {error}"))
            })
    }
}

/// The scratch storage one implementation needs.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BundleWorkspaceContractV1 {
    access: BundleTensorAccessV1,
    alias: BundleTensorAliasV1,
    alignment_bytes: u64,
    byte_length: u64,
    layout: BundleOpaqueId,
}

impl BundleWorkspaceContractV1 {
    pub fn new(
        byte_length: u64,
        alignment_bytes: u64,
        layout: BundleOpaqueId,
        access: BundleTensorAccessV1,
        alias: BundleTensorAliasV1,
    ) -> Result<Self, BundleError> {
        let workspace = Self {
            access,
            alias,
            alignment_bytes,
            byte_length,
            layout,
        };
        workspace.validate()?;
        Ok(workspace)
    }

    #[must_use]
    pub const fn byte_length(&self) -> u64 {
        self.byte_length
    }

    #[must_use]
    pub const fn alignment_bytes(&self) -> u64 {
        self.alignment_bytes
    }

    #[must_use]
    pub const fn layout(&self) -> &BundleOpaqueId {
        &self.layout
    }

    fn validate(&self) -> Result<(), BundleError> {
        validate_opaque_id(self.layout.as_str())?;
        self.alias.validate()?;
        validate_aligned_size("workspace", self.byte_length, self.alignment_bytes)
    }

    fn kernel_workspace(&self) -> Result<KernelWorkspaceContractV1, BundleError> {
        KernelWorkspaceContractV1::new(
            self.byte_length,
            self.alignment_bytes,
            kernel_id(&self.layout)?,
            self.access.into(),
            self.alias.kernel_alias()?,
        )
        .map_err(|error| {
            BundleError::validation(format!("bundle workspace contract is invalid: {error}"))
        })
    }
}

/// One compile-time choice baked into an implementation.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BundleSpecializationV1 {
    id: BundleOpaqueId,
    value: BundleOpaqueId,
}

impl BundleSpecializationV1 {
    pub fn new(id: BundleOpaqueId, value: BundleOpaqueId) -> Result<Self, BundleError> {
        let specialization = Self { id, value };
        specialization.validate()?;
        Ok(specialization)
    }

    #[must_use]
    pub const fn id(&self) -> &BundleOpaqueId {
        &self.id
    }

    #[must_use]
    pub const fn value(&self) -> &BundleOpaqueId {
        &self.value
    }

    fn validate(&self) -> Result<(), BundleError> {
        validate_opaque_id(self.id.as_str())?;
        validate_opaque_id(self.value.as_str())
    }
}

/// The complete identity and provenance of one declared implementation.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BundleImplementationDetailV1 {
    artifact: BundleArtifactIdentityV1,
    build: BundleBuildIdentityV1,
    launch: BundleLaunchContractV1,
    source: BundleSourceIdentityV1,
    specializations: Vec<BundleSpecializationV1>,
    symbol: BundleOpaqueId,
    target_architecture: BundleOpaqueId,
    #[serde(skip_serializing_if = "Option::is_none")]
    workspace: Option<BundleWorkspaceContractV1>,
}

impl BundleImplementationDetailV1 {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        artifact: BundleArtifactIdentityV1,
        symbol: BundleOpaqueId,
        target_architecture: BundleOpaqueId,
        launch: BundleLaunchContractV1,
        workspace: Option<BundleWorkspaceContractV1>,
        mut specializations: Vec<BundleSpecializationV1>,
        source: BundleSourceIdentityV1,
        build: BundleBuildIdentityV1,
    ) -> Result<Self, BundleError> {
        specializations.sort_by(|left, right| left.id.cmp(&right.id));
        let detail = Self {
            artifact,
            build,
            launch,
            source,
            specializations,
            symbol,
            target_architecture,
            workspace,
        };
        detail.validate()?;
        Ok(detail)
    }

    #[must_use]
    pub const fn artifact(&self) -> &BundleArtifactIdentityV1 {
        &self.artifact
    }

    #[must_use]
    pub const fn symbol(&self) -> &BundleOpaqueId {
        &self.symbol
    }

    #[must_use]
    pub const fn target_architecture(&self) -> &BundleOpaqueId {
        &self.target_architecture
    }

    #[must_use]
    pub const fn launch(&self) -> &BundleLaunchContractV1 {
        &self.launch
    }

    #[must_use]
    pub const fn workspace(&self) -> Option<&BundleWorkspaceContractV1> {
        self.workspace.as_ref()
    }

    #[must_use]
    pub fn specializations(&self) -> &[BundleSpecializationV1] {
        &self.specializations
    }

    #[must_use]
    pub const fn source(&self) -> &BundleSourceIdentityV1 {
        &self.source
    }

    #[must_use]
    pub const fn build(&self) -> &BundleBuildIdentityV1 {
        &self.build
    }

    fn validate(&self) -> Result<(), BundleError> {
        validate_opaque_id(self.symbol.as_str())?;
        validate_opaque_id(self.target_architecture.as_str())?;
        self.artifact.validate()?;
        self.launch.validate()?;
        if let Some(workspace) = &self.workspace {
            workspace.validate()?;
        }
        validate_count(
            "implementation specializations",
            self.specializations.len(),
            MAX_SPECIALIZATIONS,
        )?;
        for specialization in &self.specializations {
            specialization.validate()?;
        }
        validate_strict_order_by(
            "implementation specializations",
            &self.specializations,
            |value| value.id(),
        )?;
        self.source.validate()?;
        self.build.validate()?;
        // Building the shared descriptor here means a manifest that parses
        // already carries a usable implementation identity.
        self.kernel_detail().map(|_| ())
    }

    /// Converts this declaration into the shared implementation descriptor.
    pub fn kernel_detail(&self) -> Result<KernelImplementationDetailV1, BundleError> {
        let workspace = match &self.workspace {
            Some(workspace) => Some(workspace.kernel_workspace()?),
            None => None,
        };
        let mut specializations = Vec::with_capacity(self.specializations.len());
        for specialization in &self.specializations {
            specializations.push(KernelSpecializationV1::new(
                kernel_id(specialization.id())?,
                kernel_id(specialization.value())?,
            ));
        }
        KernelImplementationDetailV1::new(KernelImplementationDetailPartsV1 {
            artifact: self.artifact.kernel_artifact()?,
            symbol: kernel_id(&self.symbol)?,
            target_architecture_id: kernel_id(&self.target_architecture)?,
            launch: self.launch.kernel_launch()?,
            workspace,
            specializations,
            source: self.source.kernel_source()?,
            build: self.build.kernel_build()?,
        })
        .map_err(|error| {
            BundleError::validation(format!("bundle implementation detail is invalid: {error}"))
        })
    }
}

/// One concrete implementation of one exact operation key, with the complete
/// identity and provenance a host needs to select it.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BundleImplementationV1 {
    detail: BundleImplementationDetailV1,
    id: BundleOpaqueId,
    key: BundleOperationKeyV1,
    tier: BundleImplementationTierV1,
}

impl BundleImplementationV1 {
    pub fn new(
        id: BundleOpaqueId,
        key: BundleOperationKeyV1,
        tier: BundleImplementationTierV1,
        detail: BundleImplementationDetailV1,
    ) -> Result<Self, BundleError> {
        let implementation = Self {
            detail,
            id,
            key,
            tier,
        };
        implementation.validate()?;
        Ok(implementation)
    }

    #[must_use]
    pub const fn id(&self) -> &BundleOpaqueId {
        &self.id
    }

    #[must_use]
    pub const fn detail(&self) -> &BundleImplementationDetailV1 {
        &self.detail
    }

    #[must_use]
    pub const fn key(&self) -> &BundleOperationKeyV1 {
        &self.key
    }

    #[must_use]
    pub const fn tier(&self) -> BundleImplementationTierV1 {
        self.tier
    }

    fn validate(&self) -> Result<(), BundleError> {
        validate_opaque_id(self.id.as_str())?;
        self.key.validate()?;
        self.detail.validate()
    }
}

/// A graph bucket that references declared components and operations.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BundleGraphBucketV1 {
    id: BundleOpaqueId,
    components: Vec<BundleOpaqueId>,
    operations: Vec<BundleOperationKeyV1>,
}

impl BundleGraphBucketV1 {
    pub fn new(
        id: BundleOpaqueId,
        mut components: Vec<BundleOpaqueId>,
        mut operations: Vec<BundleOperationKeyV1>,
    ) -> Result<Self, BundleError> {
        components.sort();
        operations.sort();
        let bucket = Self {
            id,
            components,
            operations,
        };
        bucket.validate_local()?;
        Ok(bucket)
    }

    #[must_use]
    pub const fn id(&self) -> &BundleOpaqueId {
        &self.id
    }

    #[must_use]
    pub fn components(&self) -> &[BundleOpaqueId] {
        &self.components
    }

    #[must_use]
    pub fn operations(&self) -> &[BundleOperationKeyV1] {
        &self.operations
    }

    fn validate_local(&self) -> Result<(), BundleError> {
        validate_opaque_id(self.id.as_str())?;
        validate_nonempty_count(
            "graph bucket components",
            self.components.len(),
            MAX_BUCKET_COMPONENTS,
        )?;
        validate_count(
            "graph bucket operations",
            self.operations.len(),
            MAX_BUCKET_OPERATIONS,
        )?;
        for component in &self.components {
            validate_opaque_id(component.as_str())?;
        }
        for operation in &self.operations {
            operation.validate()?;
        }
        validate_strict_order("graph bucket components", &self.components)?;
        validate_strict_order("graph bucket operations", &self.operations)
    }
}

/// The optional native module declared by a V1 bundle.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BundleNativeModuleV1 {
    path: BundleRelativePathV1,
    byte_length: u64,
    sha256: String,
    entry_symbol: String,
}

impl BundleNativeModuleV1 {
    pub fn new(
        path: BundleRelativePathV1,
        byte_length: u64,
        sha256: impl Into<String>,
        entry_symbol: impl Into<String>,
    ) -> Result<Self, BundleError> {
        let native_module = Self {
            path,
            byte_length,
            sha256: sha256.into(),
            entry_symbol: entry_symbol.into(),
        };
        native_module.validate()?;
        Ok(native_module)
    }

    #[must_use]
    pub const fn path(&self) -> &BundleRelativePathV1 {
        &self.path
    }

    #[must_use]
    pub const fn byte_length(&self) -> u64 {
        self.byte_length
    }

    #[must_use]
    pub fn sha256(&self) -> &str {
        &self.sha256
    }

    #[must_use]
    pub fn entry_symbol(&self) -> &str {
        &self.entry_symbol
    }

    fn validate(&self) -> Result<(), BundleError> {
        validate_relative_path(self.path.as_str())?;
        validate_declared_file("native module", self.byte_length, MAX_NATIVE_MODULE_BYTES)?;
        validate_sha256(&self.sha256)?;
        if self.entry_symbol != BUNDLE_NATIVE_ENTRY_SYMBOL_V1 {
            return Err(BundleError::validation(format!(
                "bundle native module entry symbol '{}' is not {BUNDLE_NATIVE_ENTRY_SYMBOL_V1}",
                self.entry_symbol
            )));
        }
        Ok(())
    }
}

/// One named ordered sequence of native symbols a bundle resolves.
///
/// Position is the whole content of this record: the names are resolved in
/// exactly the order they are written down, and the sequence says nothing else.
/// It does not say what a symbol computes, which image inside the native module
/// exports it, or that resolving one launches anything, because a symbol name
/// here is an opaque value the bundle chose and the host never reads into it.
///
/// The symbols are therefore the one list in this manifest that is not sorted.
/// The sequences that carry them are, so a manifest still has one canonical
/// form, but sorting the names inside a sequence would erase the only thing it
/// states. A name repeated inside one sequence is refused instead: two
/// positions claiming one name leaves which of them owns it undefined.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BundleNativeSymbolSequenceV1 {
    id: BundleOpaqueId,
    symbols: Vec<BundleOpaqueId>,
}

impl BundleNativeSymbolSequenceV1 {
    /// Declares one sequence, keeping `symbols` in the order it was given.
    pub fn new(id: BundleOpaqueId, symbols: Vec<BundleOpaqueId>) -> Result<Self, BundleError> {
        let sequence = Self { id, symbols };
        sequence.validate()?;
        Ok(sequence)
    }

    #[must_use]
    pub const fn id(&self) -> &BundleOpaqueId {
        &self.id
    }

    /// The declared names, in resolution order.
    #[must_use]
    pub fn symbols(&self) -> &[BundleOpaqueId] {
        &self.symbols
    }

    fn validate(&self) -> Result<(), BundleError> {
        validate_opaque_id(self.id.as_str())?;
        validate_nonempty_count(
            "native symbol sequence symbols",
            self.symbols.len(),
            MAX_SEQUENCE_SYMBOLS,
        )?;
        let mut seen = BTreeSet::new();
        for symbol in &self.symbols {
            validate_opaque_id(symbol.as_str())?;
            if !seen.insert(symbol) {
                return Err(BundleError::validation(format!(
                    "bundle native symbol sequence {} names symbol {symbol} more than once",
                    self.id
                )));
            }
        }
        Ok(())
    }
}

/// Optional speculation limits and drafter provenance in a bundle manifest.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BundleSpeculationV1 {
    maximum_draft_tokens: u32,
    bundle_drafter: bool,
    maximum_verify_tokens: u32,
    row_buckets: Vec<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    drafter_component: Option<BundleOpaqueId>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    drafter_kind: Option<BundleOpaqueId>,
    /// The most items that draft in one step; absent, every item may.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    maximum_draft_items: Option<u32>,
}

impl BundleSpeculationV1 {
    pub fn new(
        maximum_draft_tokens: u32,
        bundle_drafter: bool,
        maximum_verify_tokens: u32,
        mut row_buckets: Vec<u32>,
        drafter_component: Option<BundleOpaqueId>,
    ) -> Result<Self, BundleError> {
        row_buckets.sort_unstable();
        let value = Self {
            maximum_draft_tokens,
            bundle_drafter,
            maximum_verify_tokens,
            row_buckets,
            drafter_component,
            drafter_kind: None,
            maximum_draft_items: None,
        };
        value.validate()?;
        Ok(value)
    }

    /// Optional generic identity for checking a serving profile against this package.
    /// Older declarations remain valid, but cannot satisfy a named drafter profile.
    pub fn with_drafter_kind(mut self, kind: Option<BundleOpaqueId>) -> Result<Self, BundleError> {
        self.drafter_kind = kind;
        self.validate()?;
        Ok(self)
    }

    /// Bounds the items that draft in one step, for a bundle whose
    /// per-step drafting state holds at most `items` of them. The host
    /// chooses which items draft; the others decode one row in the step.
    pub fn with_maximum_draft_items(mut self, items: u32) -> Result<Self, BundleError> {
        self.maximum_draft_items = Some(items);
        self.validate()?;
        Ok(self)
    }

    #[must_use]
    pub fn drafter_kind(&self) -> Option<&str> {
        self.drafter_kind.as_ref().map(BundleOpaqueId::as_str)
    }

    #[must_use]
    pub const fn maximum_draft_items(&self) -> Option<u32> {
        self.maximum_draft_items
    }

    #[must_use]
    pub const fn maximum_draft_tokens(&self) -> u32 {
        self.maximum_draft_tokens
    }

    #[must_use]
    pub const fn bundle_drafter(&self) -> bool {
        self.bundle_drafter
    }

    #[must_use]
    pub const fn maximum_verify_tokens(&self) -> u32 {
        self.maximum_verify_tokens
    }

    #[must_use]
    pub fn row_buckets(&self) -> &[u32] {
        &self.row_buckets
    }

    #[must_use]
    pub const fn drafter_component(&self) -> Option<&BundleOpaqueId> {
        self.drafter_component.as_ref()
    }

    fn validate(&self) -> Result<(), BundleError> {
        let largest_single_item = self
            .maximum_draft_tokens
            .checked_add(1)
            .ok_or_else(|| BundleError::validation("bundle maximum draft tokens overflow"))?;
        if self.maximum_draft_tokens == 0 || self.maximum_verify_tokens < largest_single_item {
            return Err(BundleError::validation(
                "bundle speculation has invalid token limits",
            ));
        }
        if self.bundle_drafter != self.drafter_component.is_some() {
            return Err(BundleError::validation(
                "bundle drafter provenance does not match bundle_drafter",
            ));
        }
        if let Some(kind) = &self.drafter_kind {
            validate_opaque_id(kind.as_str())?;
            if !self.bundle_drafter || kind.as_str() == "none" {
                return Err(BundleError::validation(
                    "drafter kind requires a bundle drafter and a non-none identity",
                ));
            }
        }
        if self.maximum_draft_items == Some(0) {
            return Err(BundleError::validation(
                "bundle speculation allows no drafting items",
            ));
        }
        if let Some(component) = &self.drafter_component {
            validate_opaque_id(component.as_str())?;
        }
        if self.row_buckets.is_empty()
            || self
                .row_buckets
                .iter()
                .any(|&rows| rows < 2 || rows > self.maximum_verify_tokens)
            || !self.row_buckets.windows(2).all(|pair| pair[0] < pair[1])
            || self.row_buckets.last().copied().unwrap_or(0) < largest_single_item
        {
            return Err(BundleError::validation(
                "bundle speculation row buckets are invalid or not in canonical order",
            ));
        }
        Ok(())
    }
}

/// Scored rows a manifest without `maximum_scored_rows` allows.
pub const DEFAULT_MAXIMUM_SCORED_ROWS: u32 = 512;

/// The most work one bundle instance runs at once.
///
/// A serving host refuses a deployment that would schedule more than this, so
/// an option the bundle cannot execute fails at startup instead of under load.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BundleBatchLimitsV1 {
    /// Live sequences, and so items in one batch.
    maximum_sequences: u32,
    /// Prompt tokens one batch prefills, over all its items.
    maximum_prefill_tokens: u32,
    /// Logits rows in one prefill phase that scores prompt rows: every scored
    /// row and every ordinary final-token row the phase also carries. The
    /// host schedules scored prompts in chunks of at most this many rows,
    /// and the bundle refuses to create with a larger value than it holds.
    /// Phases without scored rows are bounded by `maximum_sequences` alone.
    /// When omitted (every manifest that predates the field, and limits
    /// built with [`Self::new`] alone) it is
    /// [`DEFAULT_MAXIMUM_SCORED_ROWS`], and the canonical JSON is unchanged.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    maximum_scored_rows: Option<u32>,
    /// Sequence positions a prefill chunk may end at before the prompt end,
    /// and prefix anchors sit at: multiples of this many tokens. A bundle
    /// whose prefill arithmetic follows fixed position blocks (a chunked
    /// recurrence) declares its block, so a prompt computes the same rows
    /// whatever its chunking and wherever a prefix-cache hit resumes it.
    /// When omitted it is 1 (any position), and the canonical JSON is
    /// unchanged.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    prefill_granule: Option<u32>,
    /// Whether a request's next step may rewrite committed rows of its last
    /// KV page (drafter rows that need the following tokens). The host then
    /// copies a shared last page before a write even when it is full, so a
    /// prefix hit may resume at a page boundary. When omitted it is false,
    /// and the canonical JSON is unchanged.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    rewrites_last_page: bool,
    /// Prompt-ending prefill items per step whose state the bundle can keep
    /// at the last multiple of the prefill granule inside the item (V3 item
    /// flag `BATCH_ITEM_FLAG_ANCHOR_V3`), so the host captures the prompt-end
    /// anchor there without ending a chunk. When omitted it is 0 (every such
    /// anchor ends a chunk), and the canonical JSON is unchanged.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    anchor_items: Option<u32>,
}

impl BundleBatchLimitsV1 {
    /// Most live sequences any package may declare. This is a deliberate
    /// package-wide cap for every runtime, not an SM120 detail: it equals the
    /// shared SM120 runtime's per-item bound
    /// (`bundle_runtime::geometry::MAX_ITEMS`), and a runtime able to
    /// run more raises it here, with that bound, by a contract change.
    pub const MAXIMUM_SEQUENCES: u32 = 256;

    pub fn new(maximum_sequences: u32, maximum_prefill_tokens: u32) -> Result<Self, BundleError> {
        let limits = Self {
            maximum_sequences,
            maximum_prefill_tokens,
            maximum_scored_rows: None,
            prefill_granule: None,
            rewrites_last_page: false,
            anchor_items: None,
        };
        limits.validate()?;
        Ok(limits)
    }

    #[must_use]
    pub const fn maximum_sequences(self) -> u32 {
        self.maximum_sequences
    }

    #[must_use]
    pub const fn maximum_prefill_tokens(self) -> u32 {
        self.maximum_prefill_tokens
    }

    pub fn with_maximum_scored_rows(mut self, rows: u32) -> Result<Self, BundleError> {
        self.maximum_scored_rows = Some(rows);
        self.validate()?;
        Ok(self)
    }

    #[must_use]
    pub const fn maximum_scored_rows(self) -> u32 {
        match self.maximum_scored_rows {
            Some(rows) => rows,
            None => DEFAULT_MAXIMUM_SCORED_ROWS,
        }
    }

    pub fn with_prefill_granule(mut self, tokens: u32) -> Result<Self, BundleError> {
        self.prefill_granule = Some(tokens);
        self.validate()?;
        Ok(self)
    }

    #[must_use]
    pub const fn prefill_granule(self) -> u32 {
        match self.prefill_granule {
            Some(tokens) => tokens,
            None => 1,
        }
    }

    #[must_use]
    pub const fn with_last_page_rewrites(mut self, rewrites: bool) -> Self {
        self.rewrites_last_page = rewrites;
        self
    }

    #[must_use]
    pub const fn rewrites_last_page(self) -> bool {
        self.rewrites_last_page
    }

    pub fn with_anchor_items(mut self, items: u32) -> Result<Self, BundleError> {
        self.anchor_items = Some(items);
        self.validate()?;
        Ok(self)
    }

    #[must_use]
    pub const fn anchor_items(self) -> u32 {
        match self.anchor_items {
            Some(items) => items,
            None => 0,
        }
    }

    fn validate(self) -> Result<(), BundleError> {
        if self.maximum_scored_rows() == 0 {
            return Err(BundleError::validation(
                "bundle scored row limit must be positive",
            ));
        }
        if self
            .maximum_scored_rows
            .is_some_and(|rows| rows > self.maximum_prefill_tokens)
        {
            return Err(BundleError::validation(
                "bundle scored row limit exceeds its prefill token limit",
            ));
        }
        if self.maximum_sequences == 0 || self.maximum_prefill_tokens == 0 {
            return Err(BundleError::validation(
                "bundle batch limits must be positive",
            ));
        }
        if self.prefill_granule() == 0
            || !self
                .maximum_prefill_tokens
                .is_multiple_of(self.prefill_granule())
        {
            return Err(BundleError::validation(
                "bundle prefill granule must divide its prefill token limit",
            ));
        }
        if self.anchor_items() > self.maximum_sequences {
            return Err(BundleError::validation(
                "bundle anchor items exceed its sequences",
            ));
        }
        if self.maximum_sequences > Self::MAXIMUM_SEQUENCES {
            return Err(BundleError::validation(format!(
                "bundle batch limits declare {} sequences; a package runs at most {}",
                self.maximum_sequences,
                Self::MAXIMUM_SEQUENCES
            )));
        }
        Ok(())
    }
}

/// What one topology of a manifest that declares several runs with: the
/// fields that differ by rank count. The manifest's own fields are its first
/// topology's; each other topology it declares has one entry.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BundleTopologyResourcesV1 {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    batch_limits: Option<BundleBatchLimitsV1>,
    graph_buckets: Vec<BundleGraphBucketV1>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    kv_formats: Vec<BundleKvFormatV1>,
    kv_layout: BundleKvLayoutV1,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    recurrent_state_layouts: Vec<BundleStateLayoutV1>,
    resource_requirements: BundleResourceRequirementsV1,
    state_layout: BundleStateLayoutV1,
    topology: BundleOpaqueId,
}

impl BundleTopologyResourcesV1 {
    /// The resources of `topology`, in the canonical order of their lists.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        topology: BundleOpaqueId,
        resource_requirements: BundleResourceRequirementsV1,
        kv_layout: BundleKvLayoutV1,
        mut kv_formats: Vec<BundleKvFormatV1>,
        state_layout: BundleStateLayoutV1,
        mut recurrent_state_layouts: Vec<BundleStateLayoutV1>,
        batch_limits: Option<BundleBatchLimitsV1>,
        mut graph_buckets: Vec<BundleGraphBucketV1>,
    ) -> Self {
        kv_formats.sort_by_key(BundleKvFormatV1::dtype);
        recurrent_state_layouts.sort_by(|left, right| left.id.cmp(&right.id));
        graph_buckets.sort_by(|left, right| left.id.cmp(&right.id));
        Self {
            batch_limits,
            graph_buckets,
            kv_formats,
            kv_layout,
            recurrent_state_layouts,
            resource_requirements,
            state_layout,
            topology,
        }
    }

    #[must_use]
    pub const fn topology(&self) -> &BundleOpaqueId {
        &self.topology
    }

    #[must_use]
    pub const fn resource_requirements(&self) -> &BundleResourceRequirementsV1 {
        &self.resource_requirements
    }
}

/// The only accepted manifest representation for `chop-model-bundle-v1`.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BundleManifestV1 {
    abi_version: u32,
    /// A manifest without batch limits keeps its canonical JSON unchanged.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    batch_limits: Option<BundleBatchLimitsV1>,
    components: Vec<BundleComponentV1>,
    graph_buckets: Vec<BundleGraphBucketV1>,
    id: BundleOpaqueId,
    implementations: Vec<BundleImplementationV1>,
    /// Every serving KV layout the bundle can run, one per element, in
    /// element order; `kv_layout` is one of them, the package's preference.
    /// A manifest that declares only `kv_layout` has none, so this stays out
    /// of the canonical JSON of every manifest that predates it.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    kv_formats: Vec<BundleKvFormatV1>,
    kv_layout: BundleKvLayoutV1,
    #[serde(skip_serializing_if = "Option::is_none")]
    native_module: Option<BundleNativeModuleV1>,
    /// The ordered symbol sequences this bundle's native module resolves.
    ///
    /// A bundle that resolves no symbol in a declared order declares none, so
    /// this stays out of the canonical JSON of every manifest that predates it.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    native_symbol_sequences: Vec<BundleNativeSymbolSequenceV1>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    model_options: Option<model_options::ModelOptionsV1>,
    operation_requirements: Vec<BundleOperationRequirementV1>,
    /// Compiled images the weight plan supplies, declared without files.
    /// A package that plans none keeps its canonical JSON unchanged.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    planned_components: Vec<BundlePlannedComponentV1>,
    resource_requirements: BundleResourceRequirementsV1,
    schema: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    speculation: Option<BundleSpeculationV1>,
    state_layout: BundleStateLayoutV1,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    recurrent_state_layouts: Vec<BundleStateLayoutV1>,
    target: BundleTargetV1,
    /// The resources of each declared topology after the first, which runs
    /// with the manifest's own. A manifest that declares one topology has
    /// none, so this stays out of the canonical JSON of every manifest that
    /// predates it.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    topology_resources: Vec<BundleTopologyResourcesV1>,
}

impl BundleManifestV1 {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        id: BundleOpaqueId,
        target: BundleTargetV1,
        components: Vec<BundleComponentV1>,
        resource_requirements: BundleResourceRequirementsV1,
        kv_layout: BundleKvLayoutV1,
        state_layout: BundleStateLayoutV1,
        graph_buckets: Vec<BundleGraphBucketV1>,
        operation_requirements: Vec<BundleOperationRequirementV1>,
        implementations: Vec<BundleImplementationV1>,
        native_module: Option<BundleNativeModuleV1>,
    ) -> Result<Self, BundleError> {
        Self::new_with_planned_v1(
            id,
            target,
            components,
            Vec::new(),
            resource_requirements,
            kv_layout,
            state_layout,
            graph_buckets,
            operation_requirements,
            implementations,
            native_module,
        )
    }

    /// [`Self::new`] with planned components, which graph buckets and the
    /// speculation drafter may name like components.
    #[allow(clippy::too_many_arguments)]
    pub fn new_with_planned_v1(
        id: BundleOpaqueId,
        target: BundleTargetV1,
        mut components: Vec<BundleComponentV1>,
        mut planned_components: Vec<BundlePlannedComponentV1>,
        resource_requirements: BundleResourceRequirementsV1,
        kv_layout: BundleKvLayoutV1,
        state_layout: BundleStateLayoutV1,
        mut graph_buckets: Vec<BundleGraphBucketV1>,
        mut operation_requirements: Vec<BundleOperationRequirementV1>,
        mut implementations: Vec<BundleImplementationV1>,
        native_module: Option<BundleNativeModuleV1>,
    ) -> Result<Self, BundleError> {
        components.sort_by(|left, right| left.id.cmp(&right.id));
        planned_components.sort_by(|left, right| left.id.cmp(&right.id));
        graph_buckets.sort_by(|left, right| left.id.cmp(&right.id));
        operation_requirements.sort_by(|left, right| left.key.cmp(&right.key));
        implementations.sort_by(|left, right| {
            implementation_order_v1(left).cmp(&implementation_order_v1(right))
        });
        let manifest = Self {
            abi_version: MODEL_BUNDLE_V1_ABI_VERSION,
            batch_limits: None,
            components,
            graph_buckets,
            id,
            implementations,
            kv_formats: Vec::new(),
            kv_layout,
            native_module,
            native_symbol_sequences: Vec::new(),
            model_options: None,
            operation_requirements,
            planned_components,
            resource_requirements,
            schema: MODEL_BUNDLE_V1_SCHEMA.to_owned(),
            speculation: None,
            recurrent_state_layouts: Vec::new(),
            state_layout,
            target,
            topology_resources: Vec::new(),
        };
        manifest.validate()?;
        Ok(manifest)
    }

    pub fn with_model_options(
        mut self,
        options: model_options::ModelOptionsV1,
    ) -> Result<Self, BundleError> {
        self.model_options = Some(options);
        self.validate()?;
        Ok(self)
    }

    pub fn model_options(&self) -> Option<&model_options::ModelOptionsV1> {
        self.model_options.as_ref()
    }

    pub fn resolve_model_options(
        &self,
        overrides: &BTreeMap<String, String>,
    ) -> Result<BTreeMap<String, serde_json::Value>, BundleError> {
        match &self.model_options {
            Some(options) => options.resolve(overrides),
            None if self.requires(model_options::CAPABILITY) => Err(BundleError::validation(
                "model options capability requires a declaration",
            )),
            None if overrides.is_empty() => Ok(BTreeMap::new()),
            None => Err(BundleError::validation("package declares no model options")),
        }
    }

    /// Declares request state in canonical layout-ID order. Each layout's bytes
    /// are per sequence slot per rank, including all layers using that layout.
    pub fn with_recurrent_state_layouts(
        mut self,
        mut layouts: Vec<BundleStateLayoutV1>,
    ) -> Result<Self, BundleError> {
        layouts.sort_by(|left, right| left.id.cmp(&right.id));
        self.recurrent_state_layouts = layouts;
        self.validate()?;
        Ok(self)
    }

    /// Declares every serving KV layout the bundle can run, `kv_layout`
    /// among them, one per element. A mixed-plane element brings
    /// [`MIXED_PLANE_KV_CAPABILITY_V1`] into the resource requirements in the
    /// same step, so neither is declared without the other.
    pub fn with_kv_formats(
        mut self,
        mut formats: Vec<BundleKvFormatV1>,
    ) -> Result<Self, BundleError> {
        formats.sort_by_key(BundleKvFormatV1::dtype);
        self.resource_requirements = self.resource_requirements.for_kv_formats(&formats)?;
        self.kv_formats = formats;
        self.validate()?;
        Ok(self)
    }

    /// The declared serving KV layouts; empty when the manifest declares only
    /// `kv_layout`.
    #[must_use]
    pub fn kv_formats(&self) -> &[BundleKvFormatV1] {
        &self.kv_formats
    }

    #[must_use]
    pub fn recurrent_state_layouts(&self) -> &[BundleStateLayoutV1] {
        &self.recurrent_state_layouts
    }

    /// Bytes required by one complete state checkpoint on each rank.
    pub fn recurrent_state_bytes_per_slot(&self) -> Result<u64, BundleError> {
        self.recurrent_state_layouts
            .iter()
            .try_fold(0_u64, |bytes, layout| {
                bytes
                    .checked_add(layout.bytes)
                    .ok_or_else(|| BundleError::validation("request state size overflow"))
            })
    }

    /// Declares the ordered symbol sequences the native module resolves.
    ///
    /// The sequences are sorted by identifier so the manifest has one canonical
    /// form. The symbols inside each one are left exactly as they arrived,
    /// because their order is what the sequence declares.
    pub fn with_native_symbol_sequences(
        mut self,
        mut native_symbol_sequences: Vec<BundleNativeSymbolSequenceV1>,
    ) -> Result<Self, BundleError> {
        native_symbol_sequences.sort_by(|left, right| left.id.cmp(&right.id));
        self.native_symbol_sequences = native_symbol_sequences;
        self.validate()?;
        Ok(self)
    }

    /// Declares the most work one instance runs at once.
    pub fn with_batch_limits(mut self, limits: BundleBatchLimitsV1) -> Result<Self, BundleError> {
        self.batch_limits = Some(limits);
        self.validate()?;
        Ok(self)
    }

    /// Declares the bundle's draft and verification limits.
    pub fn with_speculation(
        mut self,
        speculation: BundleSpeculationV1,
    ) -> Result<Self, BundleError> {
        self.speculation = Some(speculation);
        self.validate()?;
        Ok(self)
    }

    /// This manifest, of its first topology alone, serving every topology of
    /// `target` with all its `components`: the others with their
    /// `resources`, one entry each.
    pub fn with_topologies(
        mut self,
        target: BundleTargetV1,
        mut components: Vec<BundleComponentV1>,
        mut resources: Vec<BundleTopologyResourcesV1>,
    ) -> Result<Self, BundleError> {
        resources.sort_by(|left, right| left.topology.cmp(&right.topology));
        components.sort_by(|left, right| left.id.cmp(&right.id));
        self.target = target;
        self.components = components;
        self.topology_resources = resources;
        self.validate()?;
        Ok(self)
    }

    /// The resources of each declared topology after the first.
    #[must_use]
    pub fn topology_resources(&self) -> &[BundleTopologyResourcesV1] {
        &self.topology_resources
    }

    /// This manifest as it serves `topology` alone: that topology its only
    /// target, and its resources the manifest's own.
    pub fn for_topology(&self, topology: &str) -> Result<Self, BundleError> {
        let selected = self.selected_topology_v1(topology)?;
        selected.validate()?;
        Ok(selected)
    }

    /// [`Self::for_topology`] before it is validated.
    fn selected_topology_v1(&self, topology: &str) -> Result<Self, BundleError> {
        let declared = self
            .target
            .topologies()
            .iter()
            .find(|candidate| candidate.id().as_str() == topology)
            .ok_or_else(|| {
                BundleError::validation(format!("bundle declares no topology {topology}"))
            })?;
        let mut selected = Self {
            target: BundleTargetV1 {
                architectures: self.target.architectures.clone(),
                topologies: vec![declared.clone()],
            },
            topology_resources: Vec::new(),
            ..self.clone()
        };
        selected.components = self
            .components
            .iter()
            .filter(|component| component.serves(topology))
            .map(|component| component.for_topology(declared.id()))
            .collect();
        if let Some(resources) = self
            .topology_resources
            .iter()
            .find(|resources| resources.topology.as_str() == topology)
        {
            let resources = resources.clone();
            selected.batch_limits = resources.batch_limits;
            selected.graph_buckets = resources.graph_buckets;
            selected.kv_formats = resources.kv_formats;
            selected.kv_layout = resources.kv_layout;
            selected.recurrent_state_layouts = resources.recurrent_state_layouts;
            selected.resource_requirements = resources.resource_requirements;
            selected.state_layout = resources.state_layout;
        }
        Ok(selected)
    }

    /// The capabilities any declared topology requires: a host with them
    /// verifies every topology the manifest declares.
    pub fn capabilities_of_every_topology(&self) -> impl Iterator<Item = &BundleOpaqueId> {
        let mut capabilities: Vec<_> = self
            .topology_requirements_v1()
            .flat_map(|(_, requirements)| requirements.required_capabilities())
            .collect();
        capabilities.sort();
        capabilities.dedup();
        capabilities.into_iter()
    }

    /// Each declared topology with the resource requirements it runs with.
    fn topology_requirements_v1(
        &self,
    ) -> impl Iterator<Item = (&BundleTopologyV1, &BundleResourceRequirementsV1)> {
        self.target.topologies().iter().map(|topology| {
            let requirements = self
                .topology_resources
                .iter()
                .find(|resources| resources.topology == topology.id)
                .map_or(&self.resource_requirements, |resources| {
                    &resources.resource_requirements
                });
            (topology, requirements)
        })
    }

    /// Components serve declared topologies. With resources per topology,
    /// every topology after the first has exactly one entry, in topology
    /// order; and when topologies differ in their resources or components,
    /// each serves as a valid manifest of its own.
    fn validate_topology_resources(&self) -> Result<(), BundleError> {
        let topologies = self.target.topologies();
        let declared = |id: &BundleOpaqueId| topologies.iter().any(|topology| topology.id() == id);
        if !self
            .components
            .iter()
            .flat_map(BundleComponentV1::topologies)
            .all(declared)
        {
            return Err(BundleError::validation(
                "a component serves a topology the bundle does not declare",
            ));
        }
        if !self.topology_resources.is_empty()
            && self
                .topology_resources
                .iter()
                .map(BundleTopologyResourcesV1::topology)
                .ne(topologies[1..].iter().map(BundleTopologyV1::id))
        {
            return Err(BundleError::validation(
                "topology resources must name each declared topology after the first once, in order",
            ));
        }
        let differ = !self.topology_resources.is_empty()
            || self
                .components
                .iter()
                .any(|component| !component.topologies.is_empty());
        if topologies.len() > 1 && differ {
            for topology in topologies {
                let topology = topology.id();
                self.selected_topology_v1(topology.as_str())?
                    .validate()
                    .map_err(|error| {
                        BundleError::validation(format!("topology {topology}: {error}"))
                    })?;
            }
        }
        Ok(())
    }

    pub fn from_canonical_json(bytes: &[u8]) -> Result<Self, BundleError> {
        if bytes.is_empty() || bytes.len() > BUNDLE_MANIFEST_MAX_BYTES {
            return Err(BundleError::validation(format!(
                "bundle manifest byte length {} is outside 1..={BUNDLE_MANIFEST_MAX_BYTES}",
                bytes.len()
            )));
        }
        std::str::from_utf8(bytes).map_err(|error| {
            BundleError::validation(format!("bundle manifest is not UTF-8: {error}"))
        })?;
        let value = parse_strict_json(bytes)?;
        if json_depth(&value) > MAX_JSON_DEPTH {
            return Err(BundleError::validation(format!(
                "bundle manifest nesting exceeds {MAX_JSON_DEPTH}"
            )));
        }
        validate_schema_and_abi_before_body(&value)?;
        refuse_compiled_images_v1(&value)?;
        let manifest: Self = serde_json::from_value(value).map_err(|error| {
            BundleError::validation(format!("bundle manifest JSON is invalid: {error}"))
        })?;
        manifest.validate()?;
        let canonical = manifest.to_canonical_json()?;
        if canonical != bytes {
            return Err(BundleError::validation(
                "bundle manifest is not in exact canonical JSON form",
            ));
        }
        Ok(manifest)
    }

    pub fn to_canonical_json(&self) -> Result<Vec<u8>, BundleError> {
        self.validate()?;
        if self.model_options.is_none()
            && self
                .capabilities_of_every_topology()
                .any(|id| id.as_str() == model_options::CAPABILITY)
        {
            return Err(BundleError::validation(
                "model options capability requires a declaration",
            ));
        }
        let mut value = serde_json::to_value(self).map_err(|error| {
            BundleError::validation(format!("bundle manifest JSON is invalid: {error}"))
        })?;
        sort_json(&mut value);
        let bytes = serde_json::to_vec(&value).map_err(|error| {
            BundleError::validation(format!("bundle manifest JSON is invalid: {error}"))
        })?;
        if bytes.len() > BUNDLE_MANIFEST_MAX_BYTES {
            return Err(BundleError::validation(format!(
                "bundle manifest byte length {} exceeds {BUNDLE_MANIFEST_MAX_BYTES}",
                bytes.len()
            )));
        }
        Ok(bytes)
    }

    pub fn validate(&self) -> Result<(), BundleError> {
        if self.schema != MODEL_BUNDLE_V1_SCHEMA {
            return Err(BundleError::validation(format!(
                "unsupported bundle schema '{}'; expected '{}'; rebuild and repackage with the current Chopinfer compiler and matching bundle library; do not edit the manifest in place",
                self.schema, MODEL_BUNDLE_V1_SCHEMA
            )));
        }
        if self.abi_version != MODEL_BUNDLE_V1_ABI_VERSION {
            return Err(BundleError::validation(format!(
                "unsupported bundle ABI version {}",
                self.abi_version
            )));
        }
        validate_opaque_id(self.id.as_str())?;
        self.target.validate()?;
        self.validate_topology_resources()?;
        if let Some(options) = &self.model_options {
            options.validate()?;
        }
        for (_, resources) in self.topology_requirements_v1() {
            let required = resources
                .required_capabilities()
                .iter()
                .any(|id| id.as_str() == model_options::CAPABILITY);
            if self.model_options.is_some() && !required {
                return Err(BundleError::validation(
                    "model option declarations require chop.model-options.v1 on every topology",
                ));
            }
        }
        validate_nonempty_count("components", self.components.len(), MAX_COMPONENTS)?;
        validate_nonempty_count("graph buckets", self.graph_buckets.len(), MAX_BUCKETS)?;
        validate_count(
            "operation requirements",
            self.operation_requirements.len(),
            MAX_REQUIREMENTS,
        )?;
        validate_count(
            "implementations",
            self.implementations.len(),
            MAX_IMPLEMENTATIONS,
        )?;
        self.resource_requirements.validate()?;
        self.kv_layout.validate()?;
        self.validate_kv_formats()?;
        self.state_layout.validate()?;
        validate_count(
            "recurrent state layouts",
            self.recurrent_state_layouts.len(),
            1024,
        )?;
        for (index, layout) in self.recurrent_state_layouts.iter().enumerate() {
            layout.validate()?;
            if layout.lifetime != BundleStateLifetimeV1::Request
                || (index > 0 && self.recurrent_state_layouts[index - 1].id >= layout.id)
            {
                return Err(BundleError::validation(
                    "recurrent state layouts require request lifetime and unique sorted IDs",
                ));
            }
        }
        self.recurrent_state_bytes_per_slot()?;
        if let Some(limits) = self.batch_limits {
            limits.validate()?;
        }
        // Per-item state needs the served count. Batch limits may arrive after
        // construction (`with_batch_limits`); without them the state has no
        // size, so a host refuses to create an instance (`state_bytes_v1`).
        if self.state_per_item() {
            if !self.requires(SERVED_ITEMS_CAPABILITY_V1) {
                return Err(BundleError::validation(format!(
                    "{STATE_PER_ITEM_CAPABILITY_V1} requires {SERVED_ITEMS_CAPABILITY_V1}"
                )));
            }
            if let Some(limits) = self.batch_limits {
                self.state_bytes_v1(Some(limits.maximum_sequences()))?;
            }
        }
        if let Some(speculation) = &self.speculation {
            speculation.validate()?;
            if !self
                .resource_requirements
                .required_capabilities()
                .iter()
                .any(|capability| capability.as_str() == SPECULATIVE_CAPABILITY_V1)
            {
                return Err(BundleError::validation(
                    "bundle speculation requires chop.speculative.v1",
                ));
            }
            if let Some(component) = speculation.drafter_component()
                && !self
                    .components
                    .iter()
                    .any(|candidate| candidate.id() == component)
                && !self
                    .planned_components
                    .iter()
                    .any(|candidate| candidate.id() == component)
            {
                return Err(BundleError::validation(
                    "bundle drafter component is not declared",
                ));
            }
        }
        for component in &self.components {
            component.validate()?;
        }
        validate_count(
            "planned components",
            self.planned_components.len(),
            MAX_COMPONENTS,
        )?;
        for component in &self.planned_components {
            component.validate()?;
        }
        validate_strict_order_by("planned component IDs", &self.planned_components, |value| {
            value.id()
        })?;
        if !self.planned_components.is_empty() && !self.requires(WEIGHT_PLAN_CAPABILITY_V1) {
            return Err(BundleError::validation(format!(
                "planned components require {WEIGHT_PLAN_CAPABILITY_V1}"
            )));
        }
        for bucket in &self.graph_buckets {
            bucket.validate_local()?;
        }
        for requirement in &self.operation_requirements {
            requirement.validate()?;
        }
        for implementation in &self.implementations {
            implementation.validate()?;
        }
        if let Some(native_module) = &self.native_module {
            native_module.validate()?;
        }

        validate_count(
            "native symbol sequences",
            self.native_symbol_sequences.len(),
            MAX_NATIVE_SYMBOL_SEQUENCES,
        )?;
        for sequence in &self.native_symbol_sequences {
            sequence.validate()?;
        }
        validate_strict_order_by(
            "native symbol sequence IDs",
            &self.native_symbol_sequences,
            |value| value.id(),
        )?;
        // The names are resolved out of the code this package ships, so a
        // package that ships none has nothing to resolve them from.
        if !self.native_symbol_sequences.is_empty() && self.native_module.is_none() {
            return Err(BundleError::validation(format!(
                "bundle declares {} native symbol sequences and no native module",
                self.native_symbol_sequences.len()
            )));
        }

        validate_strict_order_by("component IDs", &self.components, |value| value.id())?;
        validate_strict_order_by("graph bucket IDs", &self.graph_buckets, |value| value.id())?;
        validate_strict_order_by(
            "operation requirement keys",
            &self.operation_requirements,
            |value| value.key(),
        )?;
        // Implementations are canonically ordered by operation key, provider
        // tier, target architecture, and then identifier. The architecture is
        // what qualifies a record for a device, so two records of one
        // operation and tier order by architecture before any identity, and an
        // identifier-sorted listing is not canonical when the two disagree.
        if !self
            .implementations
            .windows(2)
            .all(|pair| implementation_order_v1(&pair[0]) < implementation_order_v1(&pair[1]))
        {
            return Err(BundleError::validation(
                "bundle implementations must be unique by operation, tier, and target \
                 architecture, in canonical order",
            ));
        }
        // Ordering no longer implies global identifier uniqueness, because two
        // records may share neither the ordering dimensions nor the identity.
        let mut implementation_ids = BTreeSet::new();
        for implementation in &self.implementations {
            if !implementation_ids.insert(implementation.id()) {
                return Err(BundleError::validation(format!(
                    "bundle implementation ID {} is declared more than once",
                    implementation.id()
                )));
            }
        }

        let mut paths = BTreeSet::new();
        let mut roles = BTreeSet::new();
        for component in &self.components {
            if !paths.insert(component.path.as_str()) {
                return Err(BundleError::validation(format!(
                    "bundle component path {} is duplicated",
                    component.path
                )));
            }
            // A host resolves a component by role, so two components sharing one
            // role would leave which of them a bundle opens undefined. Components
            // of some of several topologies are checked per topology
            // (`validate_topology_resources`).
            let every_topology =
                component.topologies.is_empty() || self.target.topologies().len() == 1;
            if every_topology && !roles.insert(component.role.as_str()) {
                return Err(BundleError::validation(format!(
                    "bundle component role {} is declared by more than one component",
                    component.role
                )));
            }
        }
        for component in &self.planned_components {
            if !roles.insert(component.role.as_str()) {
                return Err(BundleError::validation(format!(
                    "bundle component role {} is declared by more than one component",
                    component.role
                )));
            }
        }
        if let Some(native_module) = &self.native_module
            && !paths.insert(native_module.path.as_str())
        {
            return Err(BundleError::validation(format!(
                "bundle native module path {} conflicts with a component",
                native_module.path
            )));
        }
        if paths.contains(BUNDLE_MANIFEST_FILE) {
            return Err(BundleError::validation(
                "bundle components and native modules cannot use manifest.json",
            ));
        }

        let component_ids: BTreeSet<_> = self
            .components
            .iter()
            .map(|value| value.id())
            .chain(self.planned_components.iter().map(|value| value.id()))
            .collect();
        if component_ids.len() != self.components.len() + self.planned_components.len() {
            return Err(BundleError::validation(
                "a planned component shares its ID with a component",
            ));
        }
        let requirements: BTreeMap<_, _> = self
            .operation_requirements
            .iter()
            .map(|value| (value.key(), value.provisioning()))
            .collect();
        // The manifest's own buckets are its first topology's, and name the
        // components that serve it.
        let first = self.target.topologies()[0].id().as_str();
        let serving: BTreeSet<_> = self
            .components
            .iter()
            .filter(|component| component.serves(first))
            .map(BundleComponentV1::id)
            .chain(self.planned_components.iter().map(|value| value.id()))
            .collect();
        let mut referenced_components = BTreeSet::new();
        let mut referenced_operations = BTreeSet::new();
        for bucket in &self.graph_buckets {
            for component in bucket.components() {
                if !serving.contains(component) {
                    return Err(BundleError::validation(format!(
                        "bundle graph bucket {} references unknown component {component}",
                        bucket.id()
                    )));
                }
                referenced_components.insert(component);
            }
            for operation in bucket.operations() {
                if !requirements.contains_key(operation) {
                    return Err(BundleError::validation(format!(
                        "bundle graph bucket {} references unknown operation {}/{}",
                        bucket.id(),
                        operation.operation(),
                        operation.variant()
                    )));
                }
                referenced_operations.insert(operation);
            }
        }
        for component in serving {
            if !referenced_components.contains(component) {
                return Err(BundleError::validation(format!(
                    "bundle component {component} is not referenced by a graph bucket"
                )));
            }
        }
        for requirement in &self.operation_requirements {
            if !referenced_operations.contains(requirement.key()) {
                return Err(BundleError::validation(format!(
                    "bundle operation requirement {}/{} is not referenced by a graph bucket",
                    requirement.key().operation(),
                    requirement.key().variant()
                )));
            }
        }

        let mut implementation_tiers = BTreeSet::new();
        for implementation in &self.implementations {
            if !requirements.contains_key(implementation.key()) {
                return Err(BundleError::validation(format!(
                    "bundle implementation {} references unknown operation {}/{}",
                    implementation.id(),
                    implementation.key().operation(),
                    implementation.key().variant()
                )));
            }
            // One semantic operation and tier is declared at most once per
            // architecture. Records for different architectures are different
            // exact implementations of one logical bundle; two records for one
            // architecture would leave the choice undefined.
            if !implementation_tiers.insert((
                implementation.key(),
                implementation.tier(),
                implementation.detail().target_architecture(),
            )) {
                return Err(BundleError::validation(format!(
                    "bundle has duplicate {} implementations for operation {}/{} on architecture {}",
                    implementation_tier_name(implementation.tier()),
                    implementation.key().operation(),
                    implementation.key().variant(),
                    implementation.detail().target_architecture()
                )));
            }
            // A record physically declared by a bundle is a bundle-provided
            // implementation. Host-optimized and generic implementations come
            // from the host catalog, never from bundle JSON, whatever the
            // requirement's provisioning says.
            if implementation.tier() != BundleImplementationTierV1::Bundle {
                return Err(BundleError::validation(format!(
                    "bundle implementation {} declares {} tier, which only the host catalog \
                     provides",
                    implementation.id(),
                    implementation_tier_name(implementation.tier())
                )));
            }
            let detail = implementation.detail();
            // A selected artifact must be a file this package actually carries,
            // and must be the file it says it is: a declared component, or this
            // package's own native module.
            detail.artifact().verify_origin_linkage(
                implementation.id(),
                &self.components,
                self.native_module.as_ref(),
            )?;
            if !self
                .target
                .architectures()
                .iter()
                .any(|architecture| architecture == detail.target_architecture())
            {
                return Err(BundleError::validation(format!(
                    "bundle implementation {} targets architecture {}, which the bundle does not \
                     declare",
                    implementation.id(),
                    detail.target_architecture()
                )));
            }
        }

        // Two different operation keys can encode the same semantic key. The
        // resolver stores bundle privacy by semantic key, so two such keys must
        // agree on provisioning or ownership of that key is ambiguous.
        let mut owners: BTreeMap<KernelSemanticKeyV1, &BundleOperationRequirementV1> =
            BTreeMap::new();
        for requirement in &self.operation_requirements {
            let key = requirement.semantic_contract().semantic_key()?;
            if let Some(first) = owners.get(&key)
                && first.provisioning() != requirement.provisioning()
            {
                return Err(BundleError::validation(format!(
                    "bundle operations {}/{} and {}/{} share one semantic contract but disagree \
                     on provisioning",
                    first.key().operation(),
                    first.key().variant(),
                    requirement.key().operation(),
                    requirement.key().variant()
                )));
            }
            owners.entry(key).or_insert(requirement);
        }

        // Build the bundle-only catalog here so a package that verifies is one
        // whose own declarations are already unambiguous. The check runs per
        // declared architecture: implementations of one operation and tier for
        // different architectures are distinct exact candidates, so only two
        // records inside one architecture are a conflict.
        for architecture in self.target.architectures() {
            KernelResolverV1::new(self.bundle_candidates_for_architecture(architecture)?).map_err(
                |error| {
                    BundleError::validation(format!(
                        "bundle kernel implementations conflict on architecture {architecture}: \
                         {error}"
                    ))
                },
            )?;
        }
        Ok(())
    }

    /// The declared implementations of one architecture, as resolver
    /// candidates.
    ///
    /// Implementations declared for another architecture are not candidates
    /// here: a host resolves one exact architecture, and nothing compiled for
    /// a different one may stand in for it.
    fn bundle_candidates_for_architecture(
        &self,
        architecture: &BundleOpaqueId,
    ) -> Result<Vec<KernelImplementationCandidateV1>, BundleError> {
        let keys = self.semantic_keys()?;
        let mut candidates = Vec::new();
        for implementation in &self.implementations {
            if implementation.detail().target_architecture() != architecture {
                continue;
            }
            candidates.push(self.implementation_candidate(implementation, &keys)?);
        }
        Ok(candidates)
    }

    fn implementation_candidate(
        &self,
        implementation: &BundleImplementationV1,
        keys: &BTreeMap<BundleOperationKeyV1, KernelSemanticKeyV1>,
    ) -> Result<KernelImplementationCandidateV1, BundleError> {
        let key = keys.get(implementation.key()).ok_or_else(|| {
            BundleError::validation(format!(
                "bundle implementation {} declares operation {}/{} with no requirement",
                implementation.id(),
                implementation.key().operation(),
                implementation.key().variant()
            ))
        })?;
        KernelImplementationCandidateV1::with_detail(
            kernel_id(implementation.id())?,
            key.clone(),
            KernelImplementationTierV1::from(implementation.tier()),
            implementation.detail().kernel_detail()?,
        )
        .map_err(|error| {
            BundleError::validation(format!(
                "bundle implementation {} is malformed: {error}",
                implementation.id()
            ))
        })
    }

    /// The semantic keys of every operation only this bundle may implement.
    fn bundle_private_keys(&self) -> Result<Vec<KernelSemanticKeyV1>, BundleError> {
        let mut keys = Vec::new();
        for requirement in &self.operation_requirements {
            if requirement.provisioning() == &BundleOperationProvisioningV1::BundleCustomOperation {
                keys.push(requirement.semantic_contract().semantic_key()?);
            }
        }
        Ok(keys)
    }

    #[must_use]
    pub fn schema(&self) -> &str {
        &self.schema
    }

    #[must_use]
    pub const fn abi_version(&self) -> u32 {
        self.abi_version
    }

    #[must_use]
    pub const fn id(&self) -> &BundleOpaqueId {
        &self.id
    }

    #[must_use]
    pub const fn target(&self) -> &BundleTargetV1 {
        &self.target
    }

    #[must_use]
    pub fn components(&self) -> &[BundleComponentV1] {
        &self.components
    }

    /// Compiled images the weight plan supplies, declared without files.
    #[must_use]
    pub fn planned_components(&self) -> &[BundlePlannedComponentV1] {
        &self.planned_components
    }

    #[must_use]
    pub const fn resource_requirements(&self) -> &BundleResourceRequirementsV1 {
        &self.resource_requirements
    }

    #[must_use]
    pub const fn kv_layout(&self) -> &BundleKvLayoutV1 {
        &self.kv_layout
    }

    #[must_use]
    pub const fn state_layout(&self) -> &BundleStateLayoutV1 {
        &self.state_layout
    }

    /// Whether `state_layout` bytes are per state slot
    /// (`chop.state-per-item.v1`).
    #[must_use]
    pub fn state_per_item(&self) -> bool {
        self.requires(STATE_PER_ITEM_CAPABILITY_V1)
    }

    /// Per-rank bytes of the instance state for `served` items: the declared
    /// bytes, or under `chop.state-per-item.v1` the declared bytes times
    /// [`state_slots_v1`] of the served count, which such a package must
    /// name (1 to its maximum sequences).
    pub fn state_bytes_v1(&self, served: Option<u32>) -> Result<u64, BundleError> {
        if !self.state_per_item() {
            return Ok(self.state_layout.bytes);
        }
        let maximum = self
            .batch_limits
            .map(BundleBatchLimitsV1::maximum_sequences)
            .ok_or_else(|| {
                BundleError::validation("per-item state requires bundle batch limits")
            })?;
        let served = served
            .filter(|served| (1..=maximum).contains(served))
            .ok_or_else(|| {
                BundleError::validation(format!(
                    "per-item state needs a served count from 1 to {maximum}"
                ))
            })?;
        self.state_layout
            .bytes
            .checked_mul(state_slots_v1(served))
            .ok_or_else(|| BundleError::validation("per-item state bytes overflow u64"))
    }

    fn requires(&self, capability: &str) -> bool {
        self.resource_requirements
            .required_capabilities()
            .iter()
            .any(|required| required.as_str() == capability)
    }

    #[must_use]
    pub const fn speculation(&self) -> Option<&BundleSpeculationV1> {
        self.speculation.as_ref()
    }

    #[must_use]
    pub const fn batch_limits(&self) -> Option<BundleBatchLimitsV1> {
        self.batch_limits
    }

    #[must_use]
    pub fn graph_buckets(&self) -> &[BundleGraphBucketV1] {
        &self.graph_buckets
    }

    #[must_use]
    pub fn operation_requirements(&self) -> &[BundleOperationRequirementV1] {
        &self.operation_requirements
    }

    #[must_use]
    pub fn implementations(&self) -> &[BundleImplementationV1] {
        &self.implementations
    }

    #[must_use]
    pub const fn native_module(&self) -> Option<&BundleNativeModuleV1> {
        self.native_module.as_ref()
    }

    #[must_use]
    pub fn native_symbol_sequences(&self) -> &[BundleNativeSymbolSequenceV1] {
        &self.native_symbol_sequences
    }

    /// Builds the runtime page layout this bundle declares for one topology.
    pub fn declared_page_layout(
        &self,
        topology_id: &BundleOpaqueId,
    ) -> Result<DeclaredPageLayoutV1, BundleError> {
        self.declared_page_layout_for(&self.kv_layout, topology_id)
    }

    /// `declared_page_layout` of one of the declared KV layouts.
    pub fn declared_page_layout_for(
        &self,
        layout: &BundleKvLayoutV1,
        topology_id: &BundleOpaqueId,
    ) -> Result<DeclaredPageLayoutV1, BundleError> {
        if !self.topology_resources.is_empty() {
            return Err(BundleError::validation(
                "a bundle that declares several topologies declares pages for one it serves; \
                 select it first",
            ));
        }
        if *layout != self.kv_layout
            && !self
                .kv_formats
                .iter()
                .any(|format| format.layout == *layout)
        {
            return Err(BundleError::validation(format!(
                "bundle declares no KV layout {}",
                layout.id
            )));
        }
        let topology = self
            .target
            .topologies()
            .iter()
            .find(|topology| topology.id() == topology_id)
            .ok_or_else(|| {
                BundleError::validation(format!("bundle declares no topology {topology_id}"))
            })?;
        layout.declared_page_layout(topology)
    }

    /// Declared KV layouts agree on everything but their pages: the same page
    /// tokens, sequence limit, placement and lifetime, so a host sizes
    /// sequences alike whichever it serves. One per element, the preferred
    /// `kv_layout` among them. A mixed-plane element and the capability that
    /// names it come together, so a host without it refuses the package by
    /// name and never infers another element for its layout.
    fn validate_kv_formats(&self) -> Result<(), BundleError> {
        let mixed = self
            .kv_formats
            .iter()
            .any(|format| format.dtype == BundleKvDtypeV1::MixedPlanes);
        let required = self
            .resource_requirements
            .required_capabilities
            .iter()
            .any(|id| id.as_str() == MIXED_PLANE_KV_CAPABILITY_V1);
        if mixed != required {
            return Err(BundleError::validation(format!(
                "a mixed-plane KV format and {MIXED_PLANE_KV_CAPABILITY_V1} must be declared together"
            )));
        }
        if self.kv_formats.is_empty() {
            return Ok(());
        }
        validate_count("KV formats", self.kv_formats.len(), 8)?;
        let primary = &self.kv_layout;
        for (index, format) in self.kv_formats.iter().enumerate() {
            format.validate()?;
            let layout = &format.layout;
            if layout.page_tokens != primary.page_tokens
                || layout.maximum_sequence_tokens != primary.maximum_sequence_tokens
                || layout.rank_placement != primary.rank_placement
                || layout.state_lifetime != primary.state_lifetime
            {
                return Err(BundleError::validation(format!(
                    "bundle KV layout {} differs from {} in more than its pages",
                    layout.id, primary.id
                )));
            }
            let earlier = &self.kv_formats[..index];
            if index > 0 && earlier[index - 1].dtype >= format.dtype
                || earlier.iter().any(|other| other.layout.id == layout.id)
            {
                return Err(BundleError::validation(
                    "bundle KV formats must have distinct elements in order and distinct IDs",
                ));
            }
        }
        if !self
            .kv_formats
            .iter()
            .any(|format| format.layout == *primary)
        {
            return Err(BundleError::validation(
                "bundle KV formats must include the preferred KV layout",
            ));
        }
        Ok(())
    }

    /// Returns the semantic key of every operation this bundle requires.
    ///
    /// Every contract comes from the manifest itself, so a model-independent
    /// host needs no model-specific code to build them.
    pub fn semantic_keys(
        &self,
    ) -> Result<BTreeMap<BundleOperationKeyV1, KernelSemanticKeyV1>, BundleError> {
        let mut keys = BTreeMap::new();
        for requirement in &self.operation_requirements {
            keys.insert(
                requirement.key().clone(),
                requirement.semantic_contract().semantic_key()?,
            );
        }
        Ok(keys)
    }

    /// Builds the tiered kernel resolver for a manifest that declares exactly
    /// one architecture, from the manifest alone.
    ///
    /// This is the manifest-level form of
    /// [`BundleManifestV1::kernel_resolver_for_architecture`]: it delegates to
    /// it with the one declared architecture, so the resolver holds exactly
    /// the implementations that architecture qualified. A manifest that
    /// declares several architectures has no architecture-free resolution;
    /// resolving it here would be the cross-architecture selection the bundle
    /// contract forbids, so the call is refused and the caller must name the
    /// architecture it selected. Every declared implementation carries its own
    /// complete detail, so each candidate is built with `with_detail`. A
    /// declared implementation whose operation is not required, or whose
    /// identity or detail is malformed, rejects the bundle. An operation with
    /// no implementation record in a tier is absent, not malformed, so
    /// `host_candidates` may still satisfy it lower down, unless the bundle
    /// declared it a custom operation, which only this bundle may implement.
    pub fn kernel_resolver(
        &self,
        host_candidates: impl IntoIterator<Item = KernelImplementationCandidateV1>,
    ) -> Result<KernelResolverV1, BundleError> {
        let architectures = self.target.architectures();
        let [architecture] = architectures else {
            let named = architectures
                .iter()
                .map(|architecture| architecture.as_str())
                .collect::<Vec<_>>()
                .join(", ");
            return Err(BundleError::validation(format!(
                "this form resolves a bundle that declares one architecture, and this bundle \
                 declares {}: {named}; resolve one selected architecture with \
                 kernel_resolver_for_architecture",
                architectures.len()
            )));
        };
        self.kernel_resolver_for_architecture(architecture, host_candidates)
    }

    /// Builds the tiered kernel resolver for one exact architecture.
    ///
    /// The bundle's own implementations are filtered to records whose declared
    /// target architecture is exactly this one before any tier is resolved.
    /// There is no cross-architecture fallback: an operation left with no
    /// implementation for this architecture is unsupported, whatever else the
    /// bundle declares for other architectures. The architecture must be one
    /// the bundle declares. Host candidates join the lower tiers unchanged,
    /// because this host compiled them for itself, and none may claim the
    /// bundle tier.
    pub fn kernel_resolver_for_architecture(
        &self,
        architecture: &BundleOpaqueId,
        host_candidates: impl IntoIterator<Item = KernelImplementationCandidateV1>,
    ) -> Result<KernelResolverV1, BundleError> {
        if !self
            .target
            .architectures()
            .iter()
            .any(|declared| declared == architecture)
        {
            return Err(BundleError::validation(format!(
                "bundle does not declare architecture {architecture}"
            )));
        }
        let mut candidates = self.bundle_candidates_for_architecture(architecture)?;
        push_host_candidates(&mut candidates, host_candidates)?;
        KernelResolverV1::new_with_bundle_private(candidates, self.bundle_private_keys()?).map_err(
            |error| {
                BundleError::validation(format!("bundle kernel implementations conflict: {error}"))
            },
        )
    }

    fn declared_files(&self) -> BTreeSet<BundleRelativePathV1> {
        let mut paths = BTreeSet::new();
        paths.insert(BundleRelativePathV1(BUNDLE_MANIFEST_FILE.to_owned()));
        paths.extend(
            self.components
                .iter()
                .map(|component| component.path.clone()),
        );
        if let Some(native_module) = &self.native_module {
            paths.insert(native_module.path.clone());
        }
        paths
    }

    fn verify_host(&self, host: &BundleHostCompatibilityV1) -> Result<(), BundleError> {
        if host.abi_version != MODEL_BUNDLE_V1_ABI_VERSION {
            return Err(BundleError::validation(format!(
                "host ABI version {} is not {MODEL_BUNDLE_V1_ABI_VERSION}",
                host.abi_version
            )));
        }
        if !self
            .target
            .architectures()
            .iter()
            .any(|architecture| architecture == &host.architecture)
        {
            return Err(BundleError::validation(format!(
                "bundle does not support host architecture {}",
                host.architecture
            )));
        }
        if let Some(topology) = &host.topology
            && !self
                .target
                .topologies()
                .iter()
                .any(|candidate| candidate == topology)
        {
            return Err(BundleError::validation(format!(
                "bundle does not support host topology {} with {} ranks",
                topology.id(),
                topology.rank_count()
            )));
        }
        // The host's topology, or without one every declared topology.
        for (topology, requirements) in self.topology_requirements_v1() {
            if host.topology.as_ref().is_some_and(|host| host != topology) {
                continue;
            }
            for capability in requirements.required_capabilities() {
                if !host.capabilities.contains(capability) {
                    return Err(BundleError::validation(format!(
                        "bundle requires unsupported host capability {capability}"
                    )));
                }
            }
            if let Some(device_memory_bytes) = host.device_memory_bytes
                && device_memory_bytes < requirements.minimum_device_memory_bytes()
            {
                return Err(BundleError::validation(format!(
                    "bundle requires at least {} device bytes, host has {device_memory_bytes}",
                    requirements.minimum_device_memory_bytes()
                )));
            }
        }
        Ok(())
    }
}

/// Compatibility information supplied by a later runtime host.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BundleHostCompatibilityV1 {
    architecture: BundleOpaqueId,
    abi_version: u32,
    topology: Option<BundleTopologyV1>,
    capabilities: BTreeSet<BundleOpaqueId>,
    device_memory_bytes: Option<u64>,
}

impl BundleHostCompatibilityV1 {
    pub fn new(architecture: impl Into<String>, abi_version: u32) -> Result<Self, BundleError> {
        if abi_version != MODEL_BUNDLE_V1_ABI_VERSION {
            return Err(BundleError::validation(format!(
                "host ABI version {abi_version} is not {MODEL_BUNDLE_V1_ABI_VERSION}"
            )));
        }
        Ok(Self {
            architecture: BundleOpaqueId::new(architecture)?,
            abi_version,
            topology: None,
            capabilities: BTreeSet::new(),
            device_memory_bytes: None,
        })
    }

    pub fn with_topology(mut self, topology: BundleTopologyV1) -> Self {
        self.topology = Some(topology);
        self
    }

    pub fn with_capabilities(
        mut self,
        capabilities: impl IntoIterator<Item = BundleOpaqueId>,
    ) -> Self {
        self.capabilities.extend(capabilities);
        self
    }

    #[must_use]
    pub const fn with_device_memory_bytes(mut self, device_memory_bytes: u64) -> Self {
        self.device_memory_bytes = Some(device_memory_bytes);
        self
    }

    #[must_use]
    pub const fn architecture(&self) -> &BundleOpaqueId {
        &self.architecture
    }

    #[must_use]
    pub const fn abi_version(&self) -> u32 {
        self.abi_version
    }

    #[must_use]
    pub const fn topology(&self) -> Option<&BundleTopologyV1> {
        self.topology.as_ref()
    }

    pub fn capabilities(&self) -> impl Iterator<Item = &BundleOpaqueId> {
        self.capabilities.iter()
    }

    #[must_use]
    pub const fn device_memory_bytes(&self) -> Option<u64> {
        self.device_memory_bytes
    }
}

/// A verified directory package. Loading calls the same verifier again.
///
/// Verification reads every declared component and the native module once,
/// in full, checks each against its declared length and SHA-256, and keeps
/// the bytes it hashed. Everything this package hands out afterwards comes
/// from those bytes, never from the directory, so a file rewritten,
/// truncated or replaced after verification cannot change what is served.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BundlePackageV1 {
    root: PathBuf,
    manifest: BundleManifestV1,
    component_files: BTreeMap<BundleOpaqueId, VerifiedComponentFileV1>,
    native_module: Option<VerifiedNativeModuleV1>,
}

/// The most bytes one component may declare to be verified into memory.
///
/// Every component is read whole and kept in memory for as long as its
/// package is loaded, so this bounds the allocation one declaration can ask
/// for. A package today ships about 45 MiB in total, its largest file being
/// the native module at about 31 MiB; weights come from the checkpoint
/// through the weight plan and are never package components. The bound
/// matches the native module's and leaves an order of magnitude of headroom.
pub const MAX_VERIFIED_COMPONENT_BYTES: u64 = 512 * 1024 * 1024;

/// The most bytes verification holds for one package: its declared
/// components and native module together.
///
/// [`MAX_VERIFIED_COMPONENT_BYTES`] bounds one file, but a package may
/// declare up to 256 components, and verification keeps every one. This
/// bounds their sum before any is read: room for the largest native module
/// and as much again in components, over twenty times today's package.
pub const MAX_VERIFIED_PACKAGE_BYTES: u64 = 1024 * 1024 * 1024;

/// Refuses a manifest whose components and native module together declare
/// more than [`MAX_VERIFIED_PACKAGE_BYTES`].
fn check_verified_package_bytes(manifest: &BundleManifestV1) -> Result<(), BundleError> {
    let total = manifest
        .components()
        .iter()
        .map(BundleComponentV1::byte_length)
        .chain(
            manifest
                .native_module()
                .map(BundleNativeModuleV1::byte_length),
        )
        .try_fold(0_u64, u64::checked_add);
    match total {
        Some(total) if total <= MAX_VERIFIED_PACKAGE_BYTES => Ok(()),
        _ => Err(BundleError::validation(format!(
            "bundle components and native module declare {} bytes in total, maximum is \
             {MAX_VERIFIED_PACKAGE_BYTES} for a package verified into memory",
            total.map_or_else(|| "more than 2^64".into(), |total| total.to_string())
        ))),
    }
}

/// One declared component, authenticated and held in memory.
///
/// Opening reads the whole regular file once, without following a symlink at
/// its final path component, refuses it unless it has exactly the declared
/// length and SHA-256, and keeps the bytes it hashed. Every later read copies
/// from those bytes and never touches the file again, so a rewrite, a
/// truncation, a store through a shared mapping or a replacement of the
/// directory entry after verification cannot change what is read. Cloning
/// shares the bytes.
#[derive(Clone)]
pub struct VerifiedComponentFileV1 {
    display: PathBuf,
    sha256: String,
    bytes: Arc<[u8]>,
}

impl fmt::Debug for VerifiedComponentFileV1 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("VerifiedComponentFileV1")
            .field("display", &self.display)
            .field("byte_length", &self.bytes.len())
            .field("sha256", &self.sha256)
            .finish_non_exhaustive()
    }
}

/// Equal when verified at the same path to the same digest and length, which
/// is what equal bytes means for authenticated content.
impl PartialEq for VerifiedComponentFileV1 {
    fn eq(&self, other: &Self) -> bool {
        self.display == other.display
            && self.sha256 == other.sha256
            && self.bytes.len() == other.bytes.len()
    }
}

impl Eq for VerifiedComponentFileV1 {}

impl VerifiedComponentFileV1 {
    /// Reads, authenticates and keeps one regular file of exactly
    /// `expected_length` bytes whose SHA-256 is `expected_sha256`.
    ///
    /// A declaration above [`MAX_VERIFIED_COMPONENT_BYTES`] is refused before
    /// anything is read.
    pub fn open(
        path: impl AsRef<Path>,
        expected_length: u64,
        expected_sha256: &str,
    ) -> Result<Self, BundleError> {
        let path = path.as_ref();
        validate_sha256(expected_sha256)?;
        let bytes = read_authenticated_file(
            path,
            "component",
            expected_length,
            expected_sha256,
            MAX_VERIFIED_COMPONENT_BYTES,
        )?;
        Ok(Self {
            display: path.to_path_buf(),
            sha256: expected_sha256.to_owned(),
            bytes: bytes.into(),
        })
    }

    #[must_use]
    pub fn byte_length(&self) -> u64 {
        self.bytes.len() as u64
    }

    /// The manifest digest the bytes were checked against.
    #[must_use]
    pub fn sha256(&self) -> &str {
        &self.sha256
    }

    /// The path this component was verified at, for diagnostics only.
    #[must_use]
    pub fn display(&self) -> &Path {
        &self.display
    }

    /// The authenticated bytes themselves.
    #[must_use]
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// The authenticated bytes, shared rather than copied.
    #[must_use]
    pub fn shared_bytes(&self) -> Arc<[u8]> {
        Arc::clone(&self.bytes)
    }

    /// Copies one exact bounded range of the authenticated bytes.
    pub fn read_exact_range(&self, offset: u64, destination: &mut [u8]) -> Result<(), BundleError> {
        let source = usize::try_from(offset)
            .ok()
            .and_then(|start| self.bytes.get(start..start.checked_add(destination.len())?))
            .ok_or_else(|| {
                BundleError::validation(format!(
                    "component range at {offset} of {} bytes exceeds {} bytes",
                    destination.len(),
                    self.bytes.len()
                ))
            })?;
        destination.copy_from_slice(source);
        Ok(())
    }
}

impl BundlePackageV1 {
    /// Verifies one package directory for `host`.
    ///
    /// The manifest is parsed canonically and checked against the host, the
    /// directory must hold exactly the declared files, every component and
    /// the native module are read in full and hashed, and the manifest is
    /// read again at the end and must be the same bytes.
    pub fn verify_directory(
        root: impl AsRef<Path>,
        host: &BundleHostCompatibilityV1,
    ) -> Result<Self, BundleError> {
        verify_directory_v1(root.as_ref(), host)
    }

    /// Re-verify a package directory at process start before it is loaded.
    pub fn load_directory(
        root: impl AsRef<Path>,
        host: &BundleHostCompatibilityV1,
    ) -> Result<Self, BundleError> {
        Self::verify_directory(root, host)
    }

    #[must_use]
    pub fn root(&self) -> &Path {
        &self.root
    }

    #[must_use]
    pub const fn manifest(&self) -> &BundleManifestV1 {
        &self.manifest
    }

    /// This package as it serves `topology` alone
    /// ([`BundleManifestV1::for_topology`]); its files stay verified.
    pub fn for_topology(mut self, topology: &str) -> Result<Self, BundleError> {
        self.manifest = self.manifest.for_topology(topology)?;
        Ok(self)
    }

    /// Bytes read and hashed to authenticate the components, excluding the
    /// native module and the manifest.
    #[must_use]
    pub fn component_bytes_verified(&self) -> u64 {
        self.component_files
            .values()
            .map(VerifiedComponentFileV1::byte_length)
            .sum()
    }

    /// Returns a copy of one declared component's verified bytes.
    pub fn read_component(&self, component: &BundleComponentV1) -> Result<Vec<u8>, BundleError> {
        Ok(self.open_component_file(component)?.bytes().to_vec())
    }

    /// Returns the component authenticated during package verification.
    ///
    /// Cloning shares the verified bytes and never reopens or rehashes the
    /// component.
    pub fn open_component_file(
        &self,
        component: &BundleComponentV1,
    ) -> Result<VerifiedComponentFileV1, BundleError> {
        let declared = self
            .manifest
            .components()
            .iter()
            .find(|declared| declared.id() == component.id())
            .ok_or_else(|| BundleError::validation("component is not declared by this package"))?;
        if declared != component {
            return Err(BundleError::validation(
                "component declaration differs from the verified package",
            ));
        }
        self.component_files
            .get(component.id())
            .cloned()
            .ok_or_else(|| {
                BundleError::validation(format!(
                    "component {} has no verified bytes",
                    component.id()
                ))
            })
    }

    /// Returns a copy of the native bytes authenticated at package verification.
    pub fn read_native_module(&self) -> Result<Option<Vec<u8>>, BundleError> {
        Ok(self
            .native_module
            .as_ref()
            .map(|module| module.bytes().to_vec()))
    }

    /// Reads the declared native module and keeps the bytes it authenticated.
    ///
    /// Verifying a path and then loading that path are two lookups, and
    /// whatever replaced the entry, or rewrote the same inode, in between is
    /// what a loader that repeats the lookup would map. The digest, the length,
    /// and the entry-symbol check therefore all run on one owned byte vector,
    /// and that vector is what the loader is given.
    pub fn open_native_module(&self) -> Result<Option<VerifiedNativeModuleV1>, BundleError> {
        Ok(self.native_module.clone())
    }
}

/// One native module's authenticated bytes, detached from its package path.
///
/// Nothing here reads the filesystem again. [`Self::bytes`] is the exact vector
/// the declared length, digest, and entry symbol were checked against, so a
/// loader handed it maps what verified rather than what the path names by then.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VerifiedNativeModuleV1 {
    display: String,
    bytes: Arc<[u8]>,
}

impl VerifiedNativeModuleV1 {
    /// Returns the package path this module was verified at, for diagnostics.
    #[must_use]
    pub fn display(&self) -> &str {
        &self.display
    }

    /// Returns the authenticated bytes themselves.
    #[must_use]
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }
}

/// Reads one whole regular file and returns its bytes only if they have
/// exactly the declared length and SHA-256.
///
/// The final path component must not be a symlink, before or at open, and the
/// open descriptor must be a regular file of the declared length. At most one
/// byte past the declared length is read, so a file that grew is refused
/// without reading the growth.
fn read_authenticated_file(
    path: &Path,
    label: &str,
    expected_length: u64,
    expected_sha256: &str,
    maximum_bytes: u64,
) -> Result<Vec<u8>, BundleError> {
    if expected_length > maximum_bytes {
        return Err(BundleError::validation(format!(
            "bundle {label} {} declares {expected_length} bytes, maximum is {maximum_bytes}",
            path.display()
        )));
    }
    regular_file_metadata(path)?;
    let file =
        open_component_no_follow_v1(path).map_err(|error| BundleError::io("open", path, error))?;
    let opened = file
        .metadata()
        .map_err(|error| BundleError::io("inspect", path, error))?;
    if !opened.is_file() || opened.len() != expected_length {
        return Err(BundleError::validation(format!(
            "bundle {label} {} has {} bytes, the manifest declares {expected_length}",
            path.display(),
            opened.len()
        )));
    }
    let capacity = usize::try_from(expected_length).map_err(|_| {
        BundleError::validation(format!(
            "bundle {label} {} length does not fit this platform",
            path.display()
        ))
    })?;
    let mut bytes = Vec::new();
    bytes.try_reserve_exact(capacity).map_err(|_| {
        BundleError::validation(format!(
            "cannot allocate {capacity} bytes for bundle {label} {}",
            path.display()
        ))
    })?;
    file.take(expected_length.saturating_add(1))
        .read_to_end(&mut bytes)
        .map_err(|error| BundleError::io("read", path, error))?;
    if bytes.len() != capacity {
        return Err(BundleError::validation(format!(
            "bundle {label} {} changed while it was read",
            path.display()
        )));
    }
    let actual_sha256 = sha256_bytes(&bytes);
    if actual_sha256 != expected_sha256 {
        return Err(BundleError::validation(format!(
            "bundle {label} {} SHA-256 {actual_sha256} differs from the manifest's {expected_sha256}",
            path.display()
        )));
    }
    Ok(bytes)
}

/// The one directory verifier.
///
/// The manifest is read once and parsed once. Every decision below comes from
/// that one parse, and the file is re-read at the end and required to be the
/// same bytes: a manifest replaced while the components were being hashed
/// would otherwise leave a package describing a document that is no longer
/// there.
fn verify_directory_v1(
    root: &Path,
    host: &BundleHostCompatibilityV1,
) -> Result<BundlePackageV1, BundleError> {
    validate_directory(root, "bundle root")?;
    let manifest_path = root.join(BUNDLE_MANIFEST_FILE);
    let manifest_bytes = read_regular_file(&manifest_path, BUNDLE_MANIFEST_MAX_BYTES as u64)?;
    let manifest = BundleManifestV1::from_canonical_json(&manifest_bytes)?;
    manifest.verify_host(host)?;
    check_verified_package_bytes(&manifest)?;
    verify_closed_file_set_v1(root, &manifest)?;
    let component_files = verify_package_files(root, &manifest)?;
    checkpoint_binding::verify(&manifest, &component_files)?;
    let native_module = manifest
        .native_module()
        .map(|module| verify_native_module(root, module))
        .transpose()?;
    if read_regular_file(&manifest_path, BUNDLE_MANIFEST_MAX_BYTES as u64)? != manifest_bytes {
        return Err(BundleError::validation(format!(
            "bundle manifest {} changed while its package was being verified",
            manifest_path.display()
        )));
    }
    Ok(BundlePackageV1 {
        root: root.to_path_buf(),
        manifest,
        component_files,
        native_module,
    })
}

/// Reads and authenticates every declared component, once each.
fn verify_package_files(
    root: &Path,
    manifest: &BundleManifestV1,
) -> Result<BTreeMap<BundleOpaqueId, VerifiedComponentFileV1>, BundleError> {
    manifest
        .components()
        .iter()
        .map(|component| {
            VerifiedComponentFileV1::open(
                root.join(component.path().as_path()),
                component.byte_length(),
                component.sha256(),
            )
            .map(|file| (component.id().clone(), file))
        })
        .collect()
}

fn verify_native_module(
    root: &Path,
    native_module: &BundleNativeModuleV1,
) -> Result<VerifiedNativeModuleV1, BundleError> {
    let path = root.join(native_module.path().as_path());
    let bytes = read_authenticated_file(
        &path,
        "native module",
        native_module.byte_length(),
        native_module.sha256(),
        MAX_NATIVE_MODULE_BYTES,
    )?;
    verify_native_module_object(native_module, &bytes)?;
    Ok(VerifiedNativeModuleV1 {
        display: path.display().to_string(),
        bytes: bytes.into(),
    })
}

/// Checks that one native module's bytes export the declared entry symbol.
fn verify_native_module_object(
    native_module: &BundleNativeModuleV1,
    bytes: &[u8],
) -> Result<(), BundleError> {
    let file = object::File::parse(bytes).map_err(|error| {
        BundleError::validation(format!(
            "bundle native module {} is not a supported object file: {error}",
            native_module.path()
        ))
    })?;
    let entry = native_module.entry_symbol().as_bytes();
    let entry_with_macho_prefix = [b"_".as_slice(), entry].concat();
    let has_entry = file
        .exports()
        .map_err(|error| {
            BundleError::validation(format!(
                "bundle native module {} exports cannot be read: {error}",
                native_module.path()
            ))
        })?
        .into_iter()
        .any(|export| {
            export.name() == entry
                || (file.format() == BinaryFormat::MachO
                    && export.name() == entry_with_macho_prefix)
        });
    if !has_entry {
        return Err(BundleError::validation(format!(
            "bundle native module {} does not export {}",
            native_module.path(),
            native_module.entry_symbol()
        )));
    }
    Ok(())
}

/// Opens one final component without following a symlink at that pathname.
fn open_component_no_follow_v1(path: &Path) -> io::Result<File> {
    let mut options = OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    options.custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC);
    options.open(path)
}

/// The directory holds exactly the files the manifest declares: every
/// declared file is present and nothing else is.
fn verify_closed_file_set_v1(root: &Path, manifest: &BundleManifestV1) -> Result<(), BundleError> {
    let expected_files = manifest.declared_files();
    let (found_files, _) = collect_tree(root)?;
    if found_files != expected_files {
        return Err(BundleError::validation(format!(
            "bundle file set differs from its manifest: expected {:?}, found {:?}",
            expected_files, found_files
        )));
    }
    Ok(())
}

fn collect_tree(
    root: &Path,
) -> Result<
    (
        BTreeSet<BundleRelativePathV1>,
        BTreeSet<BundleRelativePathV1>,
    ),
    BundleError,
> {
    let mut files = BTreeSet::new();
    let mut directories = BTreeSet::new();
    let mut pending = vec![PathBuf::new()];
    while let Some(relative_directory) = pending.pop() {
        let directory = root.join(&relative_directory);
        for entry in fs::read_dir(&directory)
            .map_err(|error| BundleError::io("read bundle directory", &directory, error))?
        {
            let entry = entry.map_err(|error| {
                BundleError::io("read bundle directory entry", &directory, error)
            })?;
            let name = entry.file_name();
            let name = name.to_str().ok_or_else(|| {
                BundleError::validation(format!(
                    "bundle path under {} is not UTF-8",
                    directory.display()
                ))
            })?;
            let relative_path = relative_directory.join(name);
            let relative_text = relative_path.to_str().ok_or_else(|| {
                BundleError::validation(format!(
                    "bundle path under {} is not UTF-8",
                    root.display()
                ))
            })?;
            let relative = BundleRelativePathV1::new(relative_text)?;
            let path = root.join(relative.as_path());
            let metadata = fs::symlink_metadata(&path)
                .map_err(|error| BundleError::io("inspect", &path, error))?;
            if metadata.file_type().is_symlink() {
                return Err(BundleError::validation(format!(
                    "bundle contains a symlink: {}",
                    relative
                )));
            }
            if metadata.is_file() {
                files.insert(relative);
            } else if metadata.is_dir() {
                directories.insert(relative.clone());
                pending.push(relative_path);
            } else {
                return Err(BundleError::validation(format!(
                    "bundle path is not a regular file or directory: {}",
                    relative
                )));
            }
        }
    }
    Ok((files, directories))
}

/// Publishes one already-verified staging directory into a destination that
/// must not exist.
///
/// This is the only way a directory becomes an emitted bundle.
/// The destination is reserved first, every declared file is linked into it
/// without replacing anything, and the manifest is linked last, so a directory
/// that carries a manifest carries the whole package it describes. Every
/// emitter publishes through here, so none can invent a laxer rule.
pub fn publish_staging_no_replace(
    staging: &Path,
    destination: &Path,
    manifest: &BundleManifestV1,
) -> Result<(), BundleError> {
    reserve_destination_no_replace(destination)?;
    let declared_files = manifest.declared_files();
    create_destination_directories_no_replace(destination, &declared_files)?;
    for relative_path in &declared_files {
        if relative_path.as_str() == BUNDLE_MANIFEST_FILE {
            continue;
        }
        hard_link_no_replace(
            &staging.join(relative_path.as_path()),
            &destination.join(relative_path.as_path()),
        )?;
    }
    sync_directory(destination)?;
    // The manifest is the commit marker. Loaders reject directories without it.
    hard_link_no_replace(
        &staging.join(BUNDLE_MANIFEST_FILE),
        &destination.join(BUNDLE_MANIFEST_FILE),
    )?;
    sync_directory(destination)
}

/// Atomically publishes a directory that the canonical package verifier has
/// already accepted. The destination must not exist and is never replaced.
pub fn publish_verified_staging_no_replace(
    staging: &Path,
    destination: &Path,
    manifest: &BundleManifestV1,
) -> Result<(), BundleError> {
    sync_directory(staging)?;
    #[cfg(target_os = "linux")]
    {
        let _ = manifest;
        atomic_publish::rename_no_replace(staging, destination).map_err(|error| {
            if error.kind() == io::ErrorKind::AlreadyExists {
                BundleError::validation(format!(
                    "refusing to replace published package at {}",
                    destination.display()
                ))
            } else {
                BundleError::io("publish verified bundle directory", destination, error)
            }
        })?;
        sync_directory(containing_directory(destination))
    }
    #[cfg(not(target_os = "linux"))]
    publish_staging_no_replace(staging, destination, manifest)
}

/// The directory that holds `path`: its parent, or the working directory for
/// a relative path of one component (whose parent is empty).
fn containing_directory(path: &Path) -> &Path {
    match path.parent() {
        Some(parent) if !parent.as_os_str().is_empty() => parent,
        _ => Path::new("."),
    }
}

fn reserve_destination_no_replace(destination: &Path) -> Result<(), BundleError> {
    match fs::create_dir(destination) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {
            Err(BundleError::validation(format!(
                "refusing to replace published package at {}",
                destination.display()
            )))
        }
        Err(error) => Err(BundleError::io(
            "reserve package directory",
            destination,
            error,
        )),
    }
}

fn create_destination_directories_no_replace(
    destination: &Path,
    declared_files: &BTreeSet<BundleRelativePathV1>,
) -> Result<(), BundleError> {
    for relative_directory in declared_parent_directories(declared_files)? {
        let directory = destination.join(relative_directory.as_path());
        match fs::create_dir(&directory) {
            Ok(()) => {}
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {
                return Err(BundleError::validation(format!(
                    "refusing to merge a published package with existing directory {}",
                    directory.display()
                )));
            }
            Err(error) => {
                return Err(BundleError::io(
                    "create package directory",
                    &directory,
                    error,
                ));
            }
        }
    }
    Ok(())
}

fn declared_parent_directories(
    declared_files: &BTreeSet<BundleRelativePathV1>,
) -> Result<BTreeSet<BundleRelativePathV1>, BundleError> {
    let mut directories = BTreeSet::new();
    for file in declared_files {
        let segments = file.as_str().split('/').collect::<Vec<_>>();
        for end in 1..segments.len() {
            directories.insert(BundleRelativePathV1::new(segments[..end].join("/"))?);
        }
    }
    Ok(directories)
}

fn hard_link_no_replace(source: &Path, destination: &Path) -> Result<(), BundleError> {
    regular_file_metadata(source)?;
    match fs::hard_link(source, destination) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {
            Err(BundleError::validation(format!(
                "refusing to replace published package file {}",
                destination.display()
            )))
        }
        Err(error) => Err(BundleError::io("publish package file", destination, error)),
    }
}

fn validate_directory(path: &Path, label: &str) -> Result<(), BundleError> {
    let metadata =
        fs::symlink_metadata(path).map_err(|error| BundleError::io("inspect", path, error))?;
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        return Err(BundleError::validation(format!(
            "{label} must be a directory and not a symlink: {}",
            path.display()
        )));
    }
    Ok(())
}

fn regular_file_metadata(path: &Path) -> Result<fs::Metadata, BundleError> {
    let metadata =
        fs::symlink_metadata(path).map_err(|error| BundleError::io("inspect", path, error))?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err(BundleError::validation(format!(
            "bundle path must be a regular file and not a symlink: {}",
            path.display()
        )));
    }
    Ok(metadata)
}

fn read_regular_file(path: &Path, maximum_bytes: u64) -> Result<Vec<u8>, BundleError> {
    let metadata = regular_file_metadata(path)?;
    if metadata.len() > maximum_bytes {
        return Err(BundleError::validation(format!(
            "bundle file {} has {} bytes, maximum is {maximum_bytes}",
            path.display(),
            metadata.len()
        )));
    }
    let capacity = usize::try_from(metadata.len()).map_err(|_| {
        BundleError::validation(format!(
            "bundle file {} length does not fit this platform",
            path.display()
        ))
    })?;
    let mut file = File::open(path).map_err(|error| BundleError::io("open", path, error))?;
    let mut bytes = Vec::with_capacity(capacity);
    file.read_to_end(&mut bytes)
        .map_err(|error| BundleError::io("read", path, error))?;
    if bytes.len() != capacity {
        return Err(BundleError::validation(format!(
            "bundle file {} changed while it was read",
            path.display()
        )));
    }
    Ok(bytes)
}

fn sync_directory(path: &Path) -> Result<(), BundleError> {
    File::open(path)
        .and_then(|directory| directory.sync_all())
        .map_err(|error| BundleError::io("sync directory", path, error))
}

fn validate_opaque_id(value: &str) -> Result<(), BundleError> {
    if value.is_empty()
        || value.len() > MAX_ID_BYTES
        || value == "."
        || value == ".."
        || !value.bytes().all(|byte| {
            byte.is_ascii_lowercase() || byte.is_ascii_digit() || b"._-".contains(&byte)
        })
    {
        return Err(BundleError::validation(format!(
            "bundle opaque ID '{value}' must contain 1..={MAX_ID_BYTES} lowercase ASCII letters, digits, '.', '_', or '-'"
        )));
    }
    Ok(())
}

fn validate_relative_path(value: &str) -> Result<(), BundleError> {
    if value.is_empty()
        || value.len() > MAX_PATH_BYTES
        || value.contains('\\')
        || value.split('/').any(str::is_empty)
    {
        return Err(BundleError::validation(format!(
            "bundle path '{value}' must be a bounded slash-separated relative file path"
        )));
    }
    let path = Path::new(value);
    if path.is_absolute()
        || path
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
    {
        return Err(BundleError::validation(format!(
            "bundle path '{value}' must not traverse outside the bundle"
        )));
    }
    Ok(())
}

fn validate_sha256(value: &str) -> Result<(), BundleError> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(BundleError::validation(format!(
            "bundle SHA-256 '{value}' must contain exactly 64 lowercase hexadecimal characters"
        )));
    }
    Ok(())
}

fn validate_declared_file(label: &str, byte_length: u64, maximum: u64) -> Result<(), BundleError> {
    if byte_length == 0 || byte_length > maximum {
        return Err(BundleError::validation(format!(
            "bundle {label} byte length {byte_length} is outside 1..={maximum}"
        )));
    }
    Ok(())
}

fn validate_aligned_size(label: &str, bytes: u64, alignment_bytes: u64) -> Result<(), BundleError> {
    if bytes == 0
        || alignment_bytes == 0
        || !alignment_bytes.is_power_of_two()
        || bytes % alignment_bytes != 0
    {
        return Err(BundleError::validation(format!(
            "bundle {label} bytes {bytes} must be a nonzero multiple of power-of-two alignment {alignment_bytes}"
        )));
    }
    Ok(())
}

fn validate_count(field: &str, actual: usize, maximum: usize) -> Result<(), BundleError> {
    if actual > maximum {
        Err(BundleError::validation(format!(
            "bundle {field} count {actual} exceeds {maximum}"
        )))
    } else {
        Ok(())
    }
}

fn validate_nonempty_count(field: &str, actual: usize, maximum: usize) -> Result<(), BundleError> {
    if actual == 0 {
        Err(BundleError::validation(format!(
            "bundle {field} must not be empty"
        )))
    } else {
        validate_count(field, actual, maximum)
    }
}

fn validate_strict_order<T: Ord>(field: &str, values: &[T]) -> Result<(), BundleError> {
    if values.windows(2).all(|pair| pair[0] < pair[1]) {
        Ok(())
    } else {
        Err(BundleError::validation(format!(
            "bundle {field} must be unique and in canonical order"
        )))
    }
}

fn validate_strict_order_by<T, K: Ord>(
    field: &str,
    values: &[T],
    key: impl Fn(&T) -> &K,
) -> Result<(), BundleError> {
    if values.windows(2).all(|pair| key(&pair[0]) < key(&pair[1])) {
        Ok(())
    } else {
        Err(BundleError::validation(format!(
            "bundle {field} must be unique and in canonical order"
        )))
    }
}

fn implementation_tier_name(tier: BundleImplementationTierV1) -> &'static str {
    match tier {
        BundleImplementationTierV1::Bundle => "bundle",
        BundleImplementationTierV1::HostOptimized => "host-optimized",
        BundleImplementationTierV1::Generic => "generic",
    }
}

/// The canonical ordering key of one declared implementation: operation key,
/// provider tier, target architecture, and then implementation identifier.
///
/// The architecture is what qualifies a record for a device, so two records of
/// one operation and tier order by architecture before any identity. The
/// identifier still participates, so the order is total and stable.
fn implementation_order_v1(
    implementation: &BundleImplementationV1,
) -> (
    &BundleOperationKeyV1,
    BundleImplementationTierV1,
    &BundleOpaqueId,
    &BundleOpaqueId,
) {
    (
        implementation.key(),
        implementation.tier(),
        implementation.detail().target_architecture(),
        implementation.id(),
    )
}

/// Appends the host's own candidates to a bundle's candidate list.
///
/// Only a bundle provides the bundle tier. A host candidate that
/// claims it could satisfy an operation the bundle keeps private to itself,
/// so both resolver builders refuse such a candidate here.
fn push_host_candidates(
    candidates: &mut Vec<KernelImplementationCandidateV1>,
    host_candidates: impl IntoIterator<Item = KernelImplementationCandidateV1>,
) -> Result<(), BundleError> {
    for candidate in host_candidates {
        if candidate.tier() == KernelImplementationTierV1::Bundle {
            return Err(BundleError::validation(format!(
                "host candidate {} claims bundle tier, which only a bundle declares",
                candidate.implementation_id()
            )));
        }
        candidates.push(candidate);
    }
    Ok(())
}

fn validate_schema_and_abi_before_body(value: &Value) -> Result<(), BundleError> {
    let object = value.as_object().ok_or_else(|| {
        BundleError::validation("bundle manifest top-level value must be an object")
    })?;
    let schema = object
        .get("schema")
        .and_then(Value::as_str)
        .ok_or_else(|| BundleError::validation("bundle manifest schema is missing"))?;
    if schema != MODEL_BUNDLE_V1_SCHEMA {
        return Err(BundleError::validation(format!(
            "unsupported bundle schema '{schema}'; expected '{MODEL_BUNDLE_V1_SCHEMA}'; rebuild and repackage with the current Chopinfer compiler and matching bundle library; do not edit the manifest in place"
        )));
    }
    let abi_version = object
        .get("abi_version")
        .and_then(Value::as_u64)
        .ok_or_else(|| BundleError::validation("bundle manifest ABI version is missing"))?;
    if abi_version != u64::from(MODEL_BUNDLE_V1_ABI_VERSION) {
        return Err(BundleError::validation(format!(
            "unsupported bundle ABI version {abi_version}"
        )));
    }
    Ok(())
}

/// Refuses a manifest that divides components into rank images.
///
/// Such a package carried compiled rank images, which are no longer served:
/// every weight now loads directly from the checkpoint through the bundle's
/// weight plan. The field is gone from the manifest, so without this check
/// the refusal would be a bare unknown-field error that says nothing about
/// what to do instead. Packages that never declared rank images never wrote
/// the key, so their canonical bytes and identities are unchanged.
fn refuse_compiled_images_v1(value: &Value) -> Result<(), BundleError> {
    if value.get("compiled_images").is_some() {
        return Err(BundleError::validation(
            "bundle manifest declares compiled_images: this package carries compiled rank \
             images, which are no longer served; serve the model from its checkpoint with \
             `serve --model DIR`",
        ));
    }
    Ok(())
}

fn sort_json(value: &mut Value) {
    match value {
        Value::Array(values) => {
            for value in values {
                sort_json(value);
            }
        }
        Value::Object(object) => {
            let mut entries = std::mem::take(object).into_iter().collect::<Vec<_>>();
            for (_, value) in &mut entries {
                sort_json(value);
            }
            entries.sort_unstable_by(|(left, _), (right, _)| left.cmp(right));
            object.extend(entries);
        }
        _ => {}
    }
}

fn json_depth(value: &Value) -> usize {
    match value {
        Value::Array(values) => 1 + values.iter().map(json_depth).max().unwrap_or(0),
        Value::Object(values) => 1 + values.values().map(json_depth).max().unwrap_or(0),
        _ => 1,
    }
}

struct StrictValueSeed;

impl<'de> DeserializeSeed<'de> for StrictValueSeed {
    type Value = Value;

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        deserializer.deserialize_any(StrictValueVisitor)
    }
}

struct StrictValueVisitor;

impl<'de> Visitor<'de> for StrictValueVisitor {
    type Value = Value;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a JSON value without duplicate object keys")
    }

    fn visit_bool<E>(self, value: bool) -> Result<Self::Value, E> {
        Ok(Value::Bool(value))
    }

    fn visit_i64<E>(self, value: i64) -> Result<Self::Value, E> {
        Ok(Value::Number(value.into()))
    }

    fn visit_u64<E>(self, value: u64) -> Result<Self::Value, E> {
        Ok(Value::Number(value.into()))
    }

    fn visit_i128<E: de::Error>(self, value: i128) -> Result<Self::Value, E> {
        serde_json::Number::from_i128(value)
            .map(Value::Number)
            .ok_or_else(|| E::custom("integer out of range"))
    }

    fn visit_u128<E: de::Error>(self, value: u128) -> Result<Self::Value, E> {
        serde_json::Number::from_u128(value)
            .map(Value::Number)
            .ok_or_else(|| E::custom("integer out of range"))
    }

    fn visit_f64<E>(self, value: f64) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        serde_json::Number::from_f64(value)
            .map(Value::Number)
            .ok_or_else(|| E::custom("non-finite JSON number"))
    }

    fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        Ok(Value::String(value.to_owned()))
    }

    fn visit_string<E>(self, value: String) -> Result<Self::Value, E> {
        Ok(Value::String(value))
    }

    fn visit_none<E>(self) -> Result<Self::Value, E> {
        Ok(Value::Null)
    }

    fn visit_unit<E>(self) -> Result<Self::Value, E> {
        Ok(Value::Null)
    }

    fn visit_some<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        StrictValueSeed.deserialize(deserializer)
    }

    fn visit_seq<A>(self, mut sequence: A) -> Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        let mut values = Vec::new();
        while let Some(value) = sequence.next_element_seed(StrictValueSeed)? {
            values.push(value);
        }
        Ok(Value::Array(values))
    }

    fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
    where
        A: MapAccess<'de>,
    {
        let sized_map = map.size_hint().is_some();
        let first = map.next_key::<StrictJsonKey>()?;
        if !sized_map && first.as_ref().is_some_and(|key| key.number) {
            let text = map.next_value::<String>()?;
            if map.next_key::<String>()?.is_some() {
                return Err(de::Error::custom("invalid JSON number representation"));
            }
            let number: serde_json::Number = text.parse().map_err(de::Error::custom)?;
            if text.contains(['.', 'e', 'E']) {
                let value = number
                    .as_f64()
                    .filter(|v| v.is_finite())
                    .ok_or_else(|| de::Error::custom("non-finite JSON number"))?;
                return self.visit_f64(value);
            }
            return Ok(Value::Number(number));
        }
        let mut values = serde_json::Map::new();
        let mut next = first.map(|key| key.text);
        while let Some(key) = next {
            let value = map.next_value_seed(StrictValueSeed)?;
            if values.insert(key.clone(), value).is_some() {
                return Err(de::Error::custom(format!("duplicate JSON key: {key}")));
            }
            next = map.next_key::<String>()?;
        }
        Ok(Value::Object(values))
    }
}

// The number wrapper forwards option deserialization directly to a borrowed
// string. Real JSON map keys visit_some instead. Owned Value keys visit_string,
// and borrowed Value maps expose a size hint. Keep ordinary keys ordinary even
// when they spell serde_json's private marker.
struct StrictJsonKey {
    text: String,
    number: bool,
}

impl<'de> Deserialize<'de> for StrictJsonKey {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct KeyVisitor;
        impl<'de> Visitor<'de> for KeyVisitor {
            type Value = StrictJsonKey;
            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("an object key or internal number marker")
            }
            fn visit_some<D: serde::Deserializer<'de>>(
                self,
                deserializer: D,
            ) -> Result<Self::Value, D::Error> {
                String::deserialize(deserializer).map(|text| StrictJsonKey {
                    text,
                    number: false,
                })
            }
            fn visit_str<E: de::Error>(self, text: &str) -> Result<Self::Value, E> {
                Ok(StrictJsonKey {
                    text: text.to_owned(),
                    number: text == "$serde_json::private::Number",
                })
            }
            fn visit_string<E: de::Error>(self, text: String) -> Result<Self::Value, E> {
                Ok(StrictJsonKey {
                    text,
                    number: false,
                })
            }
        }
        deserializer.deserialize_option(KeyVisitor)
    }
}

/// Read JSON with unique object keys, exact integers, and finite binary64 floats.
/// Object insertion order and serde's nesting limit are preserved.
pub fn parse_unique_json(bytes: &[u8]) -> Result<Value, serde_json::Error> {
    let mut deserializer = serde_json::Deserializer::from_slice(bytes);
    let value = StrictValueSeed.deserialize(&mut deserializer)?;
    deserializer.end()?;
    Ok(value)
}

fn parse_strict_json(bytes: &[u8]) -> Result<Value, BundleError> {
    parse_unique_json(bytes).map_err(|error| {
        BundleError::validation(format!("bundle manifest JSON is invalid: {error}"))
    })
}

fn kernel_id(value: &BundleOpaqueId) -> Result<KernelOpaqueIdV1, BundleError> {
    KernelOpaqueIdV1::new(value.as_str()).map_err(|error| {
        BundleError::validation(format!("bundle ID {value} is not a kernel ID: {error}"))
    })
}

fn kernel_sha256(value: &str) -> Result<KernelSha256V1, BundleError> {
    KernelSha256V1::new(value).map_err(|error| {
        BundleError::validation(format!(
            "bundle digest {value} is not a kernel digest: {error}"
        ))
    })
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicU64, Ordering};

    use runtime_weights::sha256_bytes;

    use super::{
        BundleComponentV1, BundleGraphBucketV1, BundleHostCompatibilityV1, BundleKvCodeV1,
        BundleKvDtypeV1, BundleKvFormatV1, BundleKvLayoutV1, BundleKvPlaneV1,
        BundleKvScaleFormatV1, BundleKvScaleV1, BundleManifestV1, BundleOpaqueId, BundlePackageV1,
        BundleRankPlacementV1, BundleRelativePathV1, BundleResourceRequirementsV1,
        BundleStateLayoutV1, BundleStateLifetimeV1, BundleTargetV1, BundleTopologyV1,
        MAX_KV_PLANES, MIXED_PLANE_KV_CAPABILITY_V1, MODEL_BUNDLE_V1_ABI_VERSION,
        VerifiedComponentFileV1, containing_directory, declared_parent_directories,
        publish_staging_no_replace, publish_verified_staging_no_replace,
    };

    /// A relative output of one component publishes into the working
    /// directory, which is the directory synchronized after the rename.
    #[test]
    fn a_relative_output_syncs_the_working_directory() {
        assert_eq!(containing_directory(Path::new("package")), Path::new("."));
        assert_eq!(
            containing_directory(Path::new("out/package")),
            Path::new("out")
        );
        assert_eq!(
            containing_directory(Path::new("/srv/package")),
            Path::new("/srv")
        );
        assert_eq!(containing_directory(Path::new("/")), Path::new("."));
    }

    /// Components each within the per-file bound but together past the
    /// package-wide one are refused before any file is opened (none exists
    /// here).
    #[test]
    fn verification_bounds_the_package_bytes_it_holds_before_reading() {
        let root = TestDirectory::new("runtime-bundle-package-bytes");
        let package = root.path().join("package");
        fs::create_dir(&package).unwrap();
        let each = 400 * 1024 * 1024;
        assert!(each <= super::MAX_VERIFIED_COMPONENT_BYTES);
        assert!(3 * each > super::MAX_VERIFIED_PACKAGE_BYTES);
        let components: Vec<_> = ["a", "b", "c"]
            .map(|name| {
                BundleComponentV1::new(
                    id(name),
                    id(&format!("fixture.{name}.v1")),
                    id("fixture.bytes.v1"),
                    BundleRelativePathV1::new(format!("components/{name}.bin")).unwrap(),
                    each,
                    sha256_bytes(name.as_bytes()),
                )
                .unwrap()
            })
            .to_vec();
        let manifest = BundleManifestV1::new(
            id("fixture.package.v1"),
            BundleTargetV1::new(
                vec![id("sm_120f")],
                vec![BundleTopologyV1::new(id("tp1"), 1).unwrap()],
            )
            .unwrap(),
            components,
            BundleResourceRequirementsV1::new(1, Vec::new()).unwrap(),
            BundleKvLayoutV1::new(
                id("fixture.kv.v1"),
                64,
                64,
                1,
                64,
                1,
                1_024,
                BundleRankPlacementV1::Replicated,
                BundleStateLifetimeV1::Transaction,
            )
            .unwrap(),
            BundleStateLayoutV1::new(
                id("fixture.state.v1"),
                64,
                64,
                BundleStateLifetimeV1::Request,
            )
            .unwrap(),
            vec![
                BundleGraphBucketV1::new(
                    id("decode-1"),
                    vec![id("a"), id("b"), id("c")],
                    Vec::new(),
                )
                .unwrap(),
            ],
            Vec::new(),
            Vec::new(),
            None,
        )
        .unwrap();
        fs::write(
            package.join("manifest.json"),
            manifest.to_canonical_json().unwrap(),
        )
        .unwrap();
        let error = BundlePackageV1::verify_directory(&package, &host())
            .expect_err("three 400 MiB components exceed the package bound")
            .to_string();
        assert!(error.contains("in total"), "{error}");
    }

    fn host() -> BundleHostCompatibilityV1 {
        BundleHostCompatibilityV1::new("sm_120f", MODEL_BUNDLE_V1_ABI_VERSION)
            .unwrap()
            .with_topology(BundleTopologyV1::new(id("tp1"), 1).unwrap())
    }

    #[test]
    fn a_manifest_with_compiled_images_is_refused_naming_serve_model() {
        let canonical = test_manifest().to_canonical_json().unwrap();
        let mut value: serde_json::Value = serde_json::from_slice(&canonical).unwrap();
        value["compiled_images"] = serde_json::json!([{
            "alignment_bytes": 1,
            "component": "weights",
            "ranks": [{"byte_length": 1, "offset": 0, "rank": 0}],
            "topology": "tp1",
            "views": []
        }]);
        super::sort_json(&mut value);
        let error = BundleManifestV1::from_canonical_json(&serde_json::to_vec(&value).unwrap())
            .expect_err("a manifest carrying rank images must be refused");
        let message = error.to_string();
        assert!(message.contains("compiled_images"), "{message}");
        assert!(message.contains("serve --model DIR"), "{message}");

        // An empty list is still the old field, and is refused the same way.
        value["compiled_images"] = serde_json::json!([]);
        let error = BundleManifestV1::from_canonical_json(&serde_json::to_vec(&value).unwrap())
            .expect_err("an empty compiled_images list must be refused too");
        assert!(error.to_string().contains("serve --model DIR"));
    }

    #[test]
    fn a_component_rewritten_after_verification_serves_the_verified_bytes() {
        let root = TestDirectory::new("runtime-bundle-verified-bytes");
        let package = root.path().join("package");
        let manifest = test_manifest();
        write_staging_package(&package, &manifest);
        let verified = BundlePackageV1::verify_directory(&package, &host()).unwrap();
        let component = &manifest.components()[0];
        let file = verified.open_component_file(component).unwrap();
        assert_eq!(verified.component_bytes_verified(), 1);

        // Same length, different bytes, then a replacement of the entry.
        let path = package.join("components/weights.bin");
        fs::write(&path, b"y").unwrap();
        assert_eq!(verified.read_component(component).unwrap(), b"x");
        let mut byte = [0_u8; 1];
        file.read_exact_range(0, &mut byte).unwrap();
        assert_eq!(&byte, b"x");
        fs::remove_file(&path).unwrap();
        fs::write(&path, b"zz").unwrap();
        assert_eq!(verified.read_component(component).unwrap(), b"x");
        assert_eq!(file.bytes(), b"x");
        assert!(file.read_exact_range(0, &mut [0_u8; 2]).is_err());
        assert!(file.read_exact_range(1, &mut byte).is_err());

        // A fresh verification reads the file as it is now and refuses it.
        assert!(BundlePackageV1::verify_directory(&package, &host()).is_err());
    }

    #[test]
    fn a_component_that_differs_from_its_digest_is_refused_at_verification() {
        let root = TestDirectory::new("runtime-bundle-digest-mismatch");
        let package = root.path().join("package");
        write_staging_package(&package, &test_manifest());
        fs::write(package.join("components/weights.bin"), b"y").unwrap();
        let error = BundlePackageV1::verify_directory(&package, &host())
            .expect_err("a component whose bytes differ from the manifest must be refused");
        assert!(error.to_string().contains("SHA-256"), "{error}");

        let path = package.join("components/weights.bin");
        let refused = |length: u64, sha256: &str| {
            VerifiedComponentFileV1::open(&path, length, sha256).expect_err("must be refused")
        };
        assert!(
            refused(1, &sha256_bytes(b"x"))
                .to_string()
                .contains("SHA-256")
        );
        assert!(
            refused(2, &sha256_bytes(b"yy"))
                .to_string()
                .contains("declares 2")
        );
        let above = super::MAX_VERIFIED_COMPONENT_BYTES + 1;
        assert!(
            refused(above, &sha256_bytes(b"y"))
                .to_string()
                .contains("maximum")
        );
        #[cfg(unix)]
        {
            let link = package.join("components/link.bin");
            std::os::unix::fs::symlink(&path, &link).unwrap();
            assert!(VerifiedComponentFileV1::open(&link, 1, &sha256_bytes(b"y")).is_err());
        }
        assert_eq!(
            VerifiedComponentFileV1::open(&path, 1, &sha256_bytes(b"y"))
                .unwrap()
                .bytes(),
            b"y"
        );
    }

    #[test]
    fn publication_refuses_an_empty_destination_created_after_staging_verification() {
        let root = TestDirectory::new("runtime-bundle-no-clobber");
        let staging = root.path().join("staging");
        let destination = root.path().join("fixture.package.v1");
        let manifest = test_manifest();
        write_staging_package(&staging, &manifest);

        fs::create_dir(&destination).expect("racing destination is creatable");
        let error = publish_staging_no_replace(&staging, &destination, &manifest)
            .expect_err("publication must not replace the racing empty directory");

        assert!(error.to_string().contains("refusing to replace"));
        assert!(
            fs::read_dir(&destination)
                .expect("racing destination is readable")
                .next()
                .is_none(),
            "publication must leave the racing directory untouched"
        );
        assert!(
            staging.join("manifest.json").is_file(),
            "failed publication must retain the verified staging package"
        );
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn atomic_publication_preserves_file_identity_and_refuses_a_racing_destination() {
        use std::os::unix::fs::MetadataExt;

        let root = TestDirectory::new("runtime-bundle-atomic-publication");
        let staging = root.path().join("staging");
        let destination = root.path().join("fixture.package.v1");
        let manifest = test_manifest();
        write_staging_package(&staging, &manifest);
        let component = staging.join("components/weights.bin");
        let before = fs::metadata(&component).unwrap();

        fs::create_dir(&destination).unwrap();
        assert!(publish_verified_staging_no_replace(&staging, &destination, &manifest).is_err());
        assert!(component.exists());
        fs::remove_dir(&destination).unwrap();

        publish_verified_staging_no_replace(&staging, &destination, &manifest).unwrap();
        assert!(!staging.exists());
        let after = fs::metadata(destination.join("components/weights.bin")).unwrap();
        assert_eq!((before.dev(), before.ino()), (after.dev(), after.ino()));
        assert_eq!(
            (before.ctime(), before.ctime_nsec()),
            (after.ctime(), after.ctime_nsec())
        );
    }

    #[test]
    fn context_parallel_sizes_serialize_only_when_declared_and_divide_the_ranks() {
        let plain = BundleTopologyV1::new(id("tp4"), 4).unwrap();
        let json = serde_json::to_string(&plain).unwrap();
        assert_eq!(json, r#"{"id":"tp4","rank_count":4}"#);
        assert!(plain.serves_context_parallel(1));
        assert!(!plain.serves_context_parallel(2));

        let declared = plain.with_context_parallel_sizes(vec![4, 2]).unwrap();
        assert_eq!(declared.context_parallel_sizes(), [2, 4]);
        let json = serde_json::to_string(&declared).unwrap();
        assert_eq!(
            json,
            r#"{"id":"tp4","rank_count":4,"context_parallel_sizes":[2,4]}"#
        );
        assert_eq!(
            serde_json::from_str::<BundleTopologyV1>(&json).unwrap(),
            declared
        );
        assert!(declared.serves_context_parallel(2));
        assert!(!declared.serves_context_parallel(8));
        for sizes in [vec![1], vec![3], vec![2, 2], vec![8]] {
            assert!(
                BundleTopologyV1::new(id("tp4"), 4)
                    .unwrap()
                    .with_context_parallel_sizes(sizes.clone())
                    .is_err(),
                "{sizes:?}"
            );
        }
    }

    fn kv_layout(
        name: &str,
        bytes_per_token: u64,
        maximum_sequence_tokens: u64,
    ) -> BundleKvLayoutV1 {
        BundleKvLayoutV1::new(
            id(name),
            bytes_per_token,
            bytes_per_token,
            1,
            bytes_per_token,
            1,
            maximum_sequence_tokens,
            BundleRankPlacementV1::Replicated,
            BundleStateLifetimeV1::Transaction,
        )
        .unwrap()
    }

    #[test]
    fn declared_kv_formats_include_the_preferred_layout_and_share_its_geometry() {
        let manifest = test_manifest();
        let legacy = manifest.to_canonical_json().unwrap();
        assert!(!String::from_utf8_lossy(&legacy).contains("kv_formats"));
        let primary = manifest.kv_layout().clone();
        let fp8 = kv_layout("fixture.kv-fp8.v1", 32, 1_024);
        let both = manifest
            .clone()
            .with_kv_formats(vec![
                BundleKvFormatV1::new(BundleKvDtypeV1::Fp8E4m3, fp8.clone()),
                BundleKvFormatV1::new(BundleKvDtypeV1::Bf16, primary.clone()),
            ])
            .unwrap();
        let dtypes: Vec<_> = both
            .kv_formats()
            .iter()
            .map(BundleKvFormatV1::dtype)
            .collect();
        assert_eq!(dtypes, [BundleKvDtypeV1::Bf16, BundleKvDtypeV1::Fp8E4m3]);
        let parsed =
            BundleManifestV1::from_canonical_json(&both.to_canonical_json().unwrap()).unwrap();
        assert_eq!(parsed, both);
        let topology = id("tp1");
        assert_eq!(
            both.declared_page_layout_for(&fp8, &topology)
                .unwrap()
                .bytes_per_token(),
            32
        );
        assert!(manifest.declared_page_layout_for(&fp8, &topology).is_err());

        let refused =
            |formats: Vec<BundleKvFormatV1>| manifest.clone().with_kv_formats(formats).is_err();
        // Without the preferred layout, two of one element, one ID twice, and
        // another sequence limit.
        assert!(refused(vec![BundleKvFormatV1::new(
            BundleKvDtypeV1::Fp8E4m3,
            fp8.clone()
        )]));
        assert!(refused(vec![
            BundleKvFormatV1::new(BundleKvDtypeV1::Bf16, primary.clone()),
            BundleKvFormatV1::new(BundleKvDtypeV1::Bf16, fp8.clone()),
        ]));
        assert!(refused(vec![
            BundleKvFormatV1::new(BundleKvDtypeV1::Bf16, primary.clone()),
            BundleKvFormatV1::new(
                BundleKvDtypeV1::Fp8E4m3,
                kv_layout("fixture.kv.v1", 32, 1_024)
            ),
        ]));
        assert!(refused(vec![
            BundleKvFormatV1::new(BundleKvDtypeV1::Bf16, primary),
            BundleKvFormatV1::new(
                BundleKvDtypeV1::Fp8E4m3,
                kv_layout("fixture.kv-fp8.v1", 32, 2_048)
            ),
        ]));
    }

    /// A plane of `values` codes from `offset`, with its scales' format,
    /// block and offset when it is scaled.
    fn plane(
        name: &str,
        code: BundleKvCodeV1,
        values: u32,
        offset: u64,
        scale: Option<(BundleKvScaleFormatV1, u32, u64)>,
    ) -> BundleKvPlaneV1 {
        let scale = scale
            .map(|(format, block, offset)| BundleKvScaleV1::new(format, block, offset).unwrap());
        BundleKvPlaneV1::new(id(name), code, values, offset, scale).unwrap()
    }

    /// Three planes in 98 of a token's 128 bytes: 32 FP8 values and their
    /// scale, 32 FP4 values and their two scales, 64 FP4 values and their two.
    fn native_planes() -> Vec<BundleKvPlaneV1> {
        use BundleKvCodeV1::{Fp4E2m1, Fp8E4m3};
        use BundleKvScaleFormatV1::{E4m3, Ue8m0};
        vec![
            plane("window", Fp8E4m3, 32, 0, Some((Ue8m0, 32, 32))),
            plane("compressed", Fp4E2m1, 32, 40, Some((E4m3, 16, 56))),
            plane("index", Fp4E2m1, 64, 64, Some((Ue8m0, 32, 96))),
        ]
    }

    fn with_capability(mut manifest: BundleManifestV1, capability: &str) -> BundleManifestV1 {
        let mut capabilities = manifest.resource_requirements.required_capabilities.clone();
        capabilities.push(id(capability));
        manifest.resource_requirements = BundleResourceRequirementsV1::new(
            manifest.resource_requirements.minimum_device_memory_bytes,
            capabilities,
        )
        .unwrap();
        manifest
    }

    #[test]
    fn a_mixed_plane_kv_format_names_its_planes_and_its_capability() {
        let manifest = test_manifest();
        let primary = manifest.kv_layout().clone();
        let native = kv_layout("fixture.kv-native.v1", 128, 1_024);
        let mixed = BundleKvFormatV1::mixed_planes(native_planes(), native.clone()).unwrap();
        let formats = vec![
            mixed.clone(),
            BundleKvFormatV1::new(BundleKvDtypeV1::Bf16, primary.clone()),
        ];

        // Declaring the formats declares the capability with them; formats
        // without a mixed-plane element add none.
        assert!(!manifest.requires(MIXED_PLANE_KV_CAPABILITY_V1));
        let both = manifest.clone().with_kv_formats(formats).unwrap();
        assert!(both.requires(MIXED_PLANE_KV_CAPABILITY_V1));
        let bf16 = manifest
            .clone()
            .with_kv_formats(vec![BundleKvFormatV1::new(
                BundleKvDtypeV1::Bf16,
                primary.clone(),
            )])
            .unwrap();
        assert_eq!(bf16.resource_requirements, manifest.resource_requirements);
        // A manifest that states one without the other is refused, naming the
        // capability: the formats without it, and the capability without
        // them, whose layout a host would otherwise read as BF16.
        let mut without = both.clone();
        without.resource_requirements = manifest.resource_requirements.clone();
        let declares = with_capability(manifest.clone(), MIXED_PLANE_KV_CAPABILITY_V1);
        for refused in [without, declares] {
            let error = refused.validate().unwrap_err().to_string();
            assert!(error.contains(MIXED_PLANE_KV_CAPABILITY_V1), "{error}");
            let mut value = serde_json::to_value(&refused).unwrap();
            super::sort_json(&mut value);
            let bytes = serde_json::to_vec(&value).unwrap();
            assert!(BundleManifestV1::from_canonical_json(&bytes).is_err());
        }

        let elements: Vec<_> = both
            .kv_formats()
            .iter()
            .map(BundleKvFormatV1::dtype)
            .collect();
        assert_eq!(
            elements,
            [BundleKvDtypeV1::Bf16, BundleKvDtypeV1::MixedPlanes]
        );
        assert_eq!(both.kv_formats()[1], mixed);
        // The planes are part of the declaration's bytes and come back exactly,
        // in the bundle's order; the BF16 format serializes as before.
        let canonical = both.to_canonical_json().unwrap();
        let value: serde_json::Value = serde_json::from_slice(&canonical).unwrap();
        assert_eq!(
            value["kv_formats"][0],
            serde_json::json!({"dtype": "bf16", "layout": serde_json::to_value(&primary).unwrap()})
        );
        assert_eq!(
            value["kv_formats"][1]["planes"],
            serde_json::json!([
                {"code": "fp8_e4m3", "name": "window", "offset": 0, "values": 32,
                 "scale": {"block": 32, "format": "ue8m0", "offset": 32}},
                {"code": "fp4_e2m1", "name": "compressed", "offset": 40, "values": 32,
                 "scale": {"block": 16, "format": "e4m3", "offset": 56}},
                {"code": "fp4_e2m1", "name": "index", "offset": 64, "values": 64,
                 "scale": {"block": 32, "format": "ue8m0", "offset": 96}},
            ])
        );
        assert_eq!(value["kv_formats"][1]["dtype"], "mixed_planes");
        let parsed = BundleManifestV1::from_canonical_json(&canonical).unwrap();
        assert_eq!(parsed, both);
        assert_eq!(
            both.declared_page_layout_for(&native, &id("tp1"))
                .unwrap()
                .bytes_per_token(),
            128
        );

        // A host without the capability refuses the package by its name; a
        // host with it accepts.
        let error = both.verify_host(&host()).unwrap_err().to_string();
        assert!(
            error.contains(&format!(
                "unsupported host capability {MIXED_PLANE_KV_CAPABILITY_V1}"
            )),
            "{error}"
        );
        both.verify_host(&host().with_capabilities([id(MIXED_PLANE_KV_CAPABILITY_V1)]))
            .unwrap();
    }

    #[test]
    fn mixed_planes_are_declared_whole_and_never_on_another_element() {
        use BundleKvScaleFormatV1::{E4m3, Ue8m0};
        let layout = kv_layout("fixture.kv-native.v1", 128, 1_024);
        let refused = |planes: Vec<BundleKvPlaneV1>| {
            BundleKvFormatV1::mixed_planes(planes, layout.clone()).is_err()
        };
        assert!(refused(Vec::new()));
        let mut twice = native_planes();
        twice.push(plane("window", BundleKvCodeV1::Bf16, 8, 100, None));
        assert!(refused(twice));
        assert!(refused(
            (0..=MAX_KV_PLANES)
                .map(|index| {
                    let offset = 2 * index as u64;
                    plane(
                        &format!("plane{index}"),
                        BundleKvCodeV1::Bf16,
                        1,
                        offset,
                        None,
                    )
                })
                .collect()
        ));
        // FP4 codes carry their scale, BF16 takes none, and a block holds a value.
        let fp4 = BundleKvPlaneV1::new(id("compressed"), BundleKvCodeV1::Fp4E2m1, 32, 0, None);
        assert!(fp4.is_err());
        let bf16 = BundleKvPlaneV1::new(
            id("window"),
            BundleKvCodeV1::Bf16,
            16,
            0,
            Some(BundleKvScaleV1::new(E4m3, 16, 32).unwrap()),
        );
        assert!(bf16.is_err());
        assert!(BundleKvScaleV1::new(Ue8m0, 0, 0).is_err());
        assert_eq!(
            plane("window", BundleKvCodeV1::Fp8E4m3, 32, 0, None).describe(),
            "window fp8_e4m3"
        );
        assert_eq!(native_planes()[1].describe(), "compressed fp4_e2m1/e4m3x16");

        // A decoded format is checked by the same rules: planes on a BF16
        // element, a mixed-plane element without planes, an unscaled FP4 plane,
        // and a plane wider than its layout's token.
        let bf16 =
            serde_json::to_value(BundleKvFormatV1::new(BundleKvDtypeV1::Bf16, layout.clone()))
                .unwrap();
        let mixed = serde_json::to_value(
            BundleKvFormatV1::mixed_planes(native_planes(), layout.clone()).unwrap(),
        )
        .unwrap();
        let mut with_planes = bf16;
        with_planes["planes"] = mixed["planes"].clone();
        let mut without_planes = mixed.clone();
        without_planes.as_object_mut().unwrap().remove("planes");
        let mut unscaled = mixed.clone();
        unscaled["planes"][1]
            .as_object_mut()
            .unwrap()
            .remove("scale");
        let mut wide = mixed;
        wide["planes"][2]["values"] = 1_024.into();
        for value in [with_planes, without_planes, unscaled, wide] {
            let format: BundleKvFormatV1 = serde_json::from_value(value.clone()).unwrap();
            assert!(format.validate().is_err(), "{value}");
        }
    }

    #[test]
    fn mixed_planes_fit_the_bytes_their_layout_allocates() {
        use BundleKvCodeV1::{Bf16, Fp4E2m1, Fp8E4m3};
        use BundleKvScaleFormatV1::Ue8m0;
        // A plane states its width and place: one without them is refused.
        let bare = serde_json::json!(
            {"name": "p", "code": "fp4_e2m1", "scale": {"format": "ue8m0", "block": 32}}
        );
        assert!(serde_json::from_value::<BundleKvPlaneV1>(bare).is_err());

        // One block of 32 FP4 values needs 16 code bytes and one scale byte
        // per token: a page of one byte per token, or 16, cannot hold it.
        let one_block = plane("p", Fp4E2m1, 32, 0, Some((Ue8m0, 32, 16)));
        let format = |planes: Vec<BundleKvPlaneV1>, bytes: u64| {
            BundleKvFormatV1::mixed_planes(planes, kv_layout("fixture.kv-planes.v1", bytes, 1_024))
        };
        for bytes in [1, 16] {
            let error = format(vec![one_block.clone()], bytes)
                .unwrap_err()
                .to_string();
            assert!(error.contains("does not fit"), "{error}");
        }
        format(vec![one_block.clone()], 32).unwrap();

        // Values come in whole bytes and whole scale blocks.
        let scale = |offset| Some(BundleKvScaleV1::new(Ue8m0, 32, offset).unwrap());
        for (code, values) in [(Fp4E2m1, 0), (Fp4E2m1, 31), (Fp8E4m3, 48), (Bf16, 0)] {
            let scale = if code == Bf16 { None } else { scale(64) };
            assert!(
                BundleKvPlaneV1::new(id("p"), code, values, 0, scale).is_err(),
                "{code:?} {values}"
            );
        }

        // Codes and scales share no byte, within a plane or across planes,
        // and an offset past u64 is refused rather than wrapped.
        let overlap = |planes| format(planes, 64).unwrap_err().to_string();
        let own = plane("p", Fp4E2m1, 32, 0, Some((Ue8m0, 32, 15)));
        assert!(overlap(vec![own]).contains("share bytes"));
        let other = plane("q", Fp8E4m3, 8, 16, None);
        let error = overlap(vec![one_block.clone(), other]);
        assert!(
            error.contains("p and q") || error.contains("q and p"),
            "{error}"
        );
        let past = plane("p", Fp8E4m3, 8, u64::MAX - 2, None);
        assert!(format(vec![past], 64).is_err());
        let beside = plane("q", Fp8E4m3, 8, 17, None);
        format(vec![one_block, beside], 32).unwrap();
    }

    #[test]
    fn model_options_require_a_complete_declaration_at_canonicalization() {
        let mut manifest = test_manifest();
        let legacy = manifest.to_canonical_json().unwrap();
        assert!(
            !serde_json::from_slice::<serde_json::Value>(&legacy)
                .unwrap()
                .as_object()
                .unwrap()
                .contains_key("model_options")
        );
        let mut capabilities = manifest.resource_requirements.required_capabilities.clone();
        capabilities.push(BundleOpaqueId::new(crate::model_options::CAPABILITY).unwrap());
        manifest.resource_requirements = BundleResourceRequirementsV1::new(
            manifest.resource_requirements.minimum_device_memory_bytes,
            capabilities,
        )
        .unwrap();
        assert!(manifest.to_canonical_json().is_err());
        let manifest = manifest
            .with_model_options(crate::model_options::ModelOptionsV1 {
                schema: crate::model_options::CAPABILITY.into(),
                options: vec![crate::model_options::ModelOptionV1 {
                    name: "budget".into(),
                    definition: crate::model_options::OptionDefinitionV1::Integer {
                        default: 8,
                        minimum: 4,
                        maximum: 16,
                    },
                    default_is_qualified: true,
                }],
            })
            .unwrap();
        let bytes = manifest.to_canonical_json().unwrap();
        assert_eq!(
            BundleManifestV1::from_canonical_json(&bytes).unwrap(),
            manifest
        );
    }

    fn test_manifest() -> BundleManifestV1 {
        let component = BundleComponentV1::new(
            id("weights"),
            id("fixture.weights.v1"),
            id("fixture.bytes.v1"),
            BundleRelativePathV1::new("components/weights.bin").unwrap(),
            1,
            sha256_bytes(b"x"),
        )
        .unwrap();
        BundleManifestV1::new(
            id("fixture.package.v1"),
            BundleTargetV1::new(
                vec![id("sm_120f")],
                vec![BundleTopologyV1::new(id("tp1"), 1).unwrap()],
            )
            .unwrap(),
            vec![component],
            BundleResourceRequirementsV1::new(1, Vec::new()).unwrap(),
            BundleKvLayoutV1::new(
                id("fixture.kv.v1"),
                64,
                64,
                1,
                64,
                1,
                1_024,
                BundleRankPlacementV1::Replicated,
                BundleStateLifetimeV1::Transaction,
            )
            .unwrap(),
            BundleStateLayoutV1::new(
                id("fixture.state.v1"),
                64,
                64,
                BundleStateLifetimeV1::Request,
            )
            .unwrap(),
            vec![
                BundleGraphBucketV1::new(id("decode-1"), vec![id("weights")], Vec::new()).unwrap(),
            ],
            Vec::new(),
            Vec::new(),
            None,
        )
        .unwrap()
    }

    fn write_staging_package(staging: &Path, manifest: &BundleManifestV1) {
        fs::create_dir(staging).expect("staging directory is creatable");
        let declared_files = manifest.declared_files();
        for directory in declared_parent_directories(&declared_files).unwrap() {
            fs::create_dir(staging.join(directory.as_path()))
                .expect("staging component directory is creatable");
        }
        fs::write(staging.join("components/weights.bin"), b"x")
            .expect("staging component is writable");
        fs::write(
            staging.join("manifest.json"),
            manifest.to_canonical_json().unwrap(),
        )
        .expect("staging manifest is writable");
    }

    fn id(value: &str) -> BundleOpaqueId {
        BundleOpaqueId::new(value).unwrap()
    }

    struct TestDirectory {
        path: PathBuf,
    }

    impl TestDirectory {
        fn new(label: &str) -> Self {
            static NEXT: AtomicU64 = AtomicU64::new(0);

            let path = std::env::temp_dir().join(format!(
                "{label}-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            fs::create_dir(&path).expect("test directory is creatable");
            Self { path }
        }

        fn path(&self) -> &Path {
            &self.path
        }
    }

    impl Drop for TestDirectory {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.path);
        }
    }
}

#[cfg(test)]
mod scored_rows_tests {
    use super::*;

    #[test]
    fn legacy_limits_keep_canonical_json_and_default_to_512() {
        let json = r#"{"maximum_sequences":32,"maximum_prefill_tokens":8192}"#;
        let limits: BundleBatchLimitsV1 = serde_json::from_str(json).unwrap();
        limits.validate().unwrap();
        assert_eq!(limits.maximum_scored_rows(), DEFAULT_MAXIMUM_SCORED_ROWS);
        assert_eq!(serde_json::to_string(&limits).unwrap(), json);
        let built = BundleBatchLimitsV1::new(32, 8192).unwrap();
        assert_eq!(serde_json::to_string(&built).unwrap(), json);
        let explicit = built.with_maximum_scored_rows(512).unwrap();
        assert_eq!(
            serde_json::to_value(explicit).unwrap()["maximum_scored_rows"],
            512
        );
        assert!(explicit.with_maximum_scored_rows(0).is_err());
        assert!(explicit.with_maximum_scored_rows(8193).is_err());
        assert!(explicit.with_maximum_scored_rows(8192).is_ok());
        assert_eq!(
            explicit
                .with_maximum_scored_rows(7)
                .unwrap()
                .maximum_scored_rows(),
            7
        );
        let invalid: BundleBatchLimitsV1 = serde_json::from_str(
            r#"{"maximum_sequences":32,"maximum_prefill_tokens":8192,"maximum_scored_rows":0}"#,
        )
        .unwrap();
        assert!(invalid.validate().is_err());
    }

    #[test]
    fn the_prefill_granule_defaults_to_one_and_divides_the_prefill_tokens() {
        let json = r#"{"maximum_sequences":32,"maximum_prefill_tokens":8192}"#;
        let limits: BundleBatchLimitsV1 = serde_json::from_str(json).unwrap();
        assert_eq!(limits.prefill_granule(), 1);
        let aligned = limits.with_prefill_granule(64).unwrap();
        assert_eq!(aligned.prefill_granule(), 64);
        assert_eq!(
            serde_json::to_string(&aligned).unwrap(),
            r#"{"maximum_sequences":32,"maximum_prefill_tokens":8192,"prefill_granule":64}"#
        );
        assert!(limits.with_prefill_granule(0).is_err());
        assert!(limits.with_prefill_granule(48).is_err());
    }

    #[test]
    fn last_page_rewrites_default_to_off_and_serialize_only_when_on() {
        let json = r#"{"maximum_sequences":32,"maximum_prefill_tokens":8192}"#;
        let limits: BundleBatchLimitsV1 = serde_json::from_str(json).unwrap();
        assert!(!limits.rewrites_last_page());
        assert_eq!(serde_json::to_string(&limits).unwrap(), json);
        let rewrites = limits.with_last_page_rewrites(true);
        let encoded = serde_json::to_string(&rewrites).unwrap();
        assert_eq!(
            encoded,
            r#"{"maximum_sequences":32,"maximum_prefill_tokens":8192,"rewrites_last_page":true}"#
        );
        assert_eq!(
            serde_json::from_str::<BundleBatchLimitsV1>(&encoded).unwrap(),
            rewrites
        );
    }

    #[test]
    fn anchor_items_default_to_none_and_stay_within_the_sequences() {
        let json = r#"{"maximum_sequences":32,"maximum_prefill_tokens":8192}"#;
        let limits: BundleBatchLimitsV1 = serde_json::from_str(json).unwrap();
        assert_eq!(limits.anchor_items(), 0);
        let anchors = limits.with_anchor_items(8).unwrap();
        let encoded = serde_json::to_string(&anchors).unwrap();
        assert_eq!(
            encoded,
            r#"{"maximum_sequences":32,"maximum_prefill_tokens":8192,"anchor_items":8}"#
        );
        assert_eq!(
            serde_json::from_str::<BundleBatchLimitsV1>(&encoded).unwrap(),
            anchors
        );
        assert!(limits.with_anchor_items(33).is_err());
    }
}
