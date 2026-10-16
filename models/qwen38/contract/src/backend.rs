//! The kernel backend Qwen's contract plans for. Ops a bundle launches are
//! prepared at create through `kernel_core::backend`; what the contract and
//! the compiler themselves need of the backend they read through
//! [`Kernels`]. Only [`Selected`] names the backend; Qwen's SM120 module
//! implements it.

use crate::context_parallel::GatherBytes;
use crate::contract::Shape;
use crate::precision::{DenseRole, Projection};
use kernel_core::contract::{DType, Refusal, Target, WeightFormat};
use kernel_core::precision::capability::OperationCapability;
use kernel_core::precision::{ExecutionPhase, OperationSpec, OperationUse};
use kernel_core::properties::Invariance;
use projection_spec::derive::AdapterValue;
use std::collections::BTreeMap;

/// The backend Qwen's contract and bundle are built for.
pub type Selected = crate::sm120::Sm120;

/// A legacy adapter stage whose entry the backend chooses: the registered
/// fragment experts of one projection, the gated output projection, or the
/// hyper-connection stream mix.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LegacyStage {
    Experts(Projection),
    GatedOutput,
    StreamMix,
}

/// The backend's choice for a legacy adapter stage: the entry it runs and
/// the facts that describe that entry (its schedule, mix entries and
/// native capacity) beside the declaration's own facts.
#[derive(Clone, Debug, PartialEq)]
pub struct StageChoice {
    pub symbol: String,
    pub facts: BTreeMap<String, AdapterValue>,
}

/// One of the bundle's own kernels whose invariance [`crate::owned`]
/// declares; the backend names its entries.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OwnedKernel {
    PleResidual,
    PleStage,
    PleCheckpoints,
    PleCheckpointsIndexed,
    PleAnchor,
    PleConvVerify,
    NgramVerify,
}

/// A member of a registered input fusion: a dense role, or a
/// hyper-connection member by its projection ID.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FusionMember<'a> {
    Dense(DenseRole),
    HyperConnection(&'a str),
}

/// The compiled entry of one registered input fusion group: its symbol,
/// the output rows of each weight, K and the rows it is compiled for.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FusionChoice {
    pub symbol: String,
    pub segments: Vec<u32>,
    pub k: u32,
    pub native_capacity: u32,
}

/// The checkpoint sizes a backend's engine admission reads, named after
/// their config fields.
#[derive(Clone, Debug, PartialEq)]
pub struct EngineConfig {
    pub vision: EngineVision,
    pub text: EngineText,
}

/// `vision_config`'s sizes an engine admission reads.
#[derive(Clone, Debug, PartialEq)]
pub struct EngineVision {
    pub num_position_embeddings: usize,
    pub num_heads: usize,
    pub hidden_size: usize,
}

/// `text_config`'s sizes an engine admission reads (the `rope_*` and
/// `mrope_section` fields are `text_config.rope_parameters`', the
/// `mtp_rope_theta` field `text_config.mtp.rope_theta`).
#[derive(Clone, Debug, PartialEq)]
pub struct EngineText {
    pub hc_count: usize,
    pub head_dim: usize,
    pub heads_per_ngram: usize,
    pub hidden_size: usize,
    pub indexer_budget: usize,
    pub indexer_compress_ratio: usize,
    pub linear_num_key_heads: usize,
    pub linear_num_value_heads: usize,
    pub linear_value_head_dim: usize,
    pub max_position_embeddings: usize,
    pub ngram_size: usize,
    pub num_attention_heads: usize,
    pub num_experts: usize,
    pub num_key_value_heads: usize,
    pub ple_embed_dim: usize,
    pub shared_expert_intermediate_size: usize,
    pub rms_norm_eps: f64,
    pub rope_partial_rotary_factor: f64,
    pub rope_theta: f64,
    pub mrope_section: Vec<usize>,
    pub mtp_rope_theta: f64,
}

/// An engine admission's refusal of the config field at `path`.
pub fn engine_refusal(path: &str, restriction: &str, actual: impl std::fmt::Display) -> String {
    format!("unsupported engine shape {path}: {restriction}, received {actual}")
}

/// What Qwen's contract and compiler need of a kernel backend.
pub trait Kernels {
    /// The target the backend's operations are specified for
    /// (`OperationSpec::target`, `ImportedProjection::target`).
    const TARGET: Target;

    /// The device architecture a package built for the backend declares
    /// and a host admits it on (`BundleTargetV1`'s architectures,
    /// `BundleHostCompatibilityV1`).
    const ARCHITECTURE: &'static str;

    // Serving granules and scratch sizes (`crate::contract`, `crate::buffers`).

    /// Rows of the linear attention's delta-rule chunk.
    const LINEAR_CHUNK: u32;

    /// Bytes of the parallel decode route's arrival count.
    const ROUTE_TICKET_BYTES: u64;

    /// Rows the grouped experts' sorted routes take for `routed` (token,
    /// expert) pairs over `experts` local experts, each expert's padded.
    fn expert_route_rows(routed: u32, experts: u32) -> Result<u32, Refusal>;

    // Registered projections (`crate::projections`, `crate::precision`).

    /// The capability of the compiled fusion that serves a registered BF16
    /// dense `role` at `usage`.
    fn registered_dense(
        shape: Shape,
        role: DenseRole,
        usage: OperationUse,
    ) -> Result<OperationCapability, Refusal>;

    /// The capability of the grouped entry that serves a registered FP8
    /// fragment expert `projection` at `usage`.
    fn registered_expert(
        shape: Shape,
        projection: Projection,
        usage: OperationUse,
    ) -> Result<OperationCapability, Refusal>;

    /// The capability of the row entry that serves the embedding (or, with
    /// `ngram`, the n-gram) lookup at `usage`.
    fn registered_lookup(
        shape: Shape,
        ngram: bool,
        usage: OperationUse,
    ) -> Result<OperationCapability, Refusal>;

    /// The capabilities of the generated offers of `geometry`'s served
    /// topologies, or of every geometry's where `None`, in geometry order.
    fn generated_capabilities(
        geometry: Option<&'static crate::geometry::Geometry>,
    ) -> Result<Vec<OperationCapability>, String>;

    /// The ports `shape`'s declaration offers generated entries for, with
    /// their weight formats.
    fn published_ports(shape: Shape) -> Result<Vec<(&'static str, WeightFormat)>, String>;

    /// The capability of the generated entry that serves port `port` with
    /// `weight` weights at `usage`.
    fn generated_capability(
        shape: Shape,
        port: &str,
        weight: WeightFormat,
        usage: OperationUse,
    ) -> Result<OperationCapability, String>;

    /// The effective operation a capability's generated entry computes,
    /// with local scales of `local_scale` (the format's canonical dtype when
    /// `None`).
    fn capability_domain(
        capability: &OperationCapability,
        local_scale: Option<DType>,
    ) -> Result<OperationSpec, String>;

    /// The entry and entry facts of a legacy adapter `stage` at `usage`.
    fn legacy_stage(
        shape: Shape,
        stage: LegacyStage,
        usage: OperationUse,
    ) -> Result<StageChoice, String>;

    /// The compiled entry of the registered input fusion group led by
    /// `member` at `usage`.
    fn legacy_fusion(
        shape: Shape,
        member: FusionMember<'_>,
        usage: OperationUse,
    ) -> Result<FusionChoice, String>;

    // The bundle's own kernels (`crate::owned`).

    /// A declaration of one of the bundle's own entries, which the
    /// backend's invariance checks read.
    type Declared;

    /// The declaration of the bundle's own entry `symbol`.
    fn bundle_declaration(symbol: String, invariance: Invariance) -> Self::Declared;

    /// The entry of the bundle's own `kernel` at the topology symbol tag
    /// `tag` (the geometry's kernel tag, `_tp{N}` outside TP4).
    fn owned_entry(kernel: OwnedKernel, tag: &str) -> String;

    /// The prefill and verify symbols of the registered n-gram hash entries
    /// of every topology that publishes the packed n-gram lookup.
    fn ngram_hash_symbols() -> Result<Vec<(String, String)>, String>;

    /// The registered n-gram hash entry of `shape`'s topology at `phase`
    /// (the packed n-gram lookup's standalone hash stage).
    fn ngram_hash_symbol(shape: Shape, phase: ExecutionPhase) -> String;

    /// Whether the bundle module `name` holds the publisher's mixed
    /// projection entries (prepared packages bind them; legacy ones load
    /// none).
    fn mixed_cubin(name: &str) -> bool;

    // Encoder kernels (`crate::encoder`).

    /// One kernel an encoder declaration runs.
    type EncoderKernel: Send + Sync + 'static;

    /// The kernels `declarations` run, in declaration order.
    fn encoder_kernels(
        declarations: &[encoder_spec::Declaration],
    ) -> Result<Vec<Self::EncoderKernel>, encoder_spec::Error>;

    // Engine admission (the compiler's geometry admission).

    /// The geometry rules' engine-domain phases, in the order
    /// [`Kernels::admit_engine`] checks them.
    const ENGINE_PHASES: [&'static str; 5];

    /// Admit a checkpoint's sizes to the backend's engines: the first
    /// refusal in the backend's check order, calling `limits` (the
    /// geometry rules' declared engine domain) at each of
    /// [`Kernels::ENGINE_PHASES`] between its own checks. `geometry` is the
    /// matched geometry, if any.
    fn admit_engine(
        config: &EngineConfig,
        geometry: Option<&'static crate::geometry::Geometry>,
        limits: &mut dyn FnMut(&str) -> Result<(), String>,
    ) -> Result<(), String>;

    // Context-parallel groups (`crate::context_parallel`).

    /// Whether the backend instantiates every context-parallel group entry
    /// at `shape`'s geometry.
    fn context_parallel_entries(shape: Shape) -> bool;

    /// The bytes of a group all-gather of `words` 32-bit words from each of
    /// `size` members.
    fn group_gather(size: u32, words: u64) -> Result<GatherBytes, Refusal>;

    /// Bytes of the per-rank selection's split workspace for `rows` rows at
    /// `shape`'s geometry.
    fn selection_workspace(shape: Shape, rows: u32) -> u64;
}
