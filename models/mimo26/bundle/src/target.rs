//! What MiMo's generic code asks of the GPU target its instance runs on,
//! beyond the ops the target's backend prepares (`crate::ops`): the backend
//! and own kernels themselves, the attention queries its catalog states, how
//! its row kernels grid prefill rows, which graphs fold the MoE combine into
//! the MLP all-reduce, the modules an instance loads, the comm entries whose
//! residency create checks, the pipelined prefill's lanes and grids, how it
//! serves the projections the precision component binds, and its encoders.
//!
//! An instance builds its target once at create, with its own-op set
//! (`instance::create`); `crate::sm120` implements it for SM120.

use std::collections::BTreeMap;
use std::ops::RangeInclusive;

use bundle_runtime::encoder::Encoder;
use bundle_runtime::geometry::Items;
use bundle_runtime::residency::SpinningEntry;
use kernel_core::backend::Backend;
use kernel_core::contract::{AttentionLayout, AttentionPhase, AttentionQuery, DType, Refusal};
use kernel_core::gated_mlp::GatedMlpProjectionQuery;
use kernel_core::precision::execution::{Binding, BoundUse};
use kernel_core::precision::{OperationSpec, OperationUse};
use model_mimo26_contract::encoders::Encoders;
use model_mimo26_contract::own_ops::OwnOps;
use runtime_bundle_abi::{BundleAbiError, DeviceBufferSpanV1 as Span, HostServicesV1};

use crate::overlap::{Config, Facts, Lane};
use crate::pipeline::PreparedLaunch;
use crate::selection::RowBuckets;
use crate::switches::Switches;

/// One topology's target.
pub trait Target: Send + Sync {
    /// The backend whose ops an instance with `switches` prepares. Graphs
    /// take the wide all-reduce entries only when `wide`.
    fn backend(&self, switches: Switches, wide: bool) -> Result<Box<dyn Backend>, Refusal>;

    /// The bundle's own kernels.
    fn own(&self) -> &dyn OwnOps;

    /// The row buckets eager prefill row kernels grid
    /// (`MimoPhase::row_grid`).
    fn row_buckets(&self) -> RowBuckets;

    /// Whether a decode or verify graph of `rows` rows folds the MoE combine
    /// into its MLP all-reduce.
    fn folds_combine(&self, rows: u32) -> bool;

    /// Whether a decode or verify bucket of `rows` rows that does not route
    /// as prefill does routes on the fused router, whose route op projects
    /// the router logits itself.
    fn fused_router(&self, rows: u32) -> bool;

    /// The query of the target attention over a `layout` cache of `kv`
    /// entries at `phase`, as the catalog states it.
    fn attention_query(
        &self,
        layout: AttentionLayout,
        phase: AttentionPhase,
        kv: DType,
    ) -> Result<AttentionQuery, Refusal>;

    /// The query of DFlash's block attention.
    fn block_attention_query(&self) -> Result<AttentionQuery, Refusal>;

    /// Whether an instance with `switches` loads the target's `cubin`, as far
    /// as the target decides: modules of entries the switches leave unused
    /// stay unloaded.
    fn loads(&self, cubin: &str, switches: Switches) -> bool;

    /// The dynamic shared memory the compiled entry `symbol` requests in an
    /// instance serving `items`.
    fn dynamic_shared_bytes(&self, symbol: &str, items: Items) -> Result<u32, Refusal>;

    /// The spinning comm entries of an instance whose graphs reach `rows`
    /// rows, which create checks against the device. With the pipelined
    /// prefill's `lanes`, the phased pair also runs at the comm lane's
    /// grid, which the target refuses past what it can keep resident.
    fn spinning_entries(
        &self,
        rows: u32,
        lanes: Option<&Config>,
    ) -> Result<Vec<SpinningEntry>, Refusal>;

    /// The pipelined prefill's lanes (`crate::overlap`) from the process
    /// environment, if it is on: the settings the target measured them at.
    fn lanes(&self) -> Option<Config>;

    /// Moves `launch`'s grid on `lane` under `config`, by the facts of the
    /// op that made it: the target decides which grids may change.
    fn adjust(&self, launch: &mut PreparedLaunch, facts: Facts, lane: Lane, config: &Config);

    /// The backend that prepares the projections the precision component
    /// binds to it.
    fn projection_backend(&self) -> Box<dyn Backend>;

    /// The most rows the projection backend serves the use of `operation`
    /// at `usage` that the component binds, under the identity the
    /// component authenticates; none for a use a registered launcher serves
    /// ([`Target::registered_projection`]).
    fn projection_rows(
        &self,
        operation: &OperationSpec,
        usage: OperationUse,
    ) -> Result<Binding<Option<u32>>, String>;

    /// What MiMo's registered launcher of the use of `operation` at `usage`
    /// launches on; none for a use the projection backend prepares.
    fn registered_projection(
        &self,
        operation: &OperationSpec,
        usage: OperationUse,
    ) -> Result<Option<Registered>, String>;

    /// The alternative the component binds for projection `id` whose use
    /// `bound` the target launches instead over part of the use's live rows
    /// under `switches`: the alternative's id and the live rows it serves.
    fn row_alternative(
        &self,
        id: &str,
        bound: &BoundUse,
        switches: Switches,
    ) -> Result<Option<(String, RangeInclusive<u32>)>, String>;

    /// The encoders serving `media`, created at instance create on the
    /// target's encoder kernels and own encoder ops over each rank's encoder
    /// image `weights`, every rank's regions allocated with `allocate`.
    fn encoders(
        &self,
        services: &HostServicesV1,
        media: Encoders,
        weights: Vec<BTreeMap<String, Span>>,
        allocate: &mut dyn FnMut(usize, u64) -> Result<Span, BundleAbiError>,
    ) -> Result<Encoder, BundleAbiError>;
}

/// What a registered projection launcher launches on.
pub enum Registered {
    /// A grouped expert entry: the backend that prepares its gated MLP
    /// projection, the contract that computes, and the workspace bytes its
    /// launches need.
    Experts {
        backend: Box<dyn Backend>,
        op: GatedMlpProjectionQuery,
        workspace_bytes: u64,
    },
    /// An embedding lookup: the backend that prepares its gathers.
    Lookup { backend: Box<dyn Backend> },
}
