//! The kernel backend GLM's contract plans for. Ops a bundle launches are
//! prepared at create through `kernel_core::backend`; what the contract and
//! the compiler themselves need of the backend they read through
//! [`Kernels`]. Only [`Selected`] names the backend; GLM's SM120 module
//! implements it.

use kernel_core::contract::{Target, WeightFormat};
use kernel_core::precision::OperationUse;
use kernel_core::precision::capability::OperationCapability;

use crate::contract::Shape;

/// The backend GLM's contract and bundle are built for.
pub type Selected = crate::sm120::Sm120;

/// The entry a lowered projection use binds, with the capability whose
/// stages name what the use runs and the identity of the kernels the
/// binding depends on.
pub struct Bound<E> {
    pub entry: E,
    pub capability: OperationCapability,
    pub dependency: String,
}

/// The scratch an entry that runs in an own op's launch needs: its
/// workspace and the semaphores zeroed before its first launch.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct OwnLaunch {
    pub workspace_bytes: u64,
    pub semaphore_bytes: u64,
}

/// The flag and epoch bytes one comm protocol needs in a comm state.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct StateBytes {
    pub flags: u64,
    pub epoch: u64,
}

pub use bundle_runtime::context_parallel::GatherBytes;

/// What GLM's contract needs of a kernel backend.
pub trait Kernels {
    /// The target the backend's operations are specified for
    /// (`OperationSpec::target`, `ImportedProjection::target`).
    const TARGET: Target;

    /// The device architecture a package built for the backend declares
    /// and a host admits it on (`BundleTargetV1`'s architectures).
    const ARCHITECTURE: &'static str;

    /// The entry a lowered projection use binds, which the bundle launches.
    type Entry: Clone;

    /// The entry serving lowered projection `id` with `weight` rows at
    /// `usage` (`crate::prepared::bind`).
    fn bind(
        id: &str,
        weight: WeightFormat,
        usage: OperationUse,
    ) -> Result<Bound<Self::Entry>, String>;

    /// The scratch of `entry`'s launch where the entry runs only in a
    /// launch the bundle's own ops prepare, not as a projection the backend
    /// prepares on its own; `None` for any other entry.
    fn own_launch(entry: &Self::Entry) -> Result<Option<OwnLaunch>, String>;

    // Scratch a step binds (the bundle's buffers).

    /// The logits row gather's state.
    const GATHER_ROWS_STATE: StateBytes;

    /// The rows of the MoE's largest route: every routed row plus each
    /// local expert's padding.
    fn sorted_rows(shape: Shape) -> Result<u32, String>;

    /// The Sinkhorn mixer's split partials and tickets of `rows` rows, in
    /// bytes.
    fn mix_scratch_bytes(shape: Shape, rows: u32) -> [u64; 2];

    /// The largest workspace any of the backend's latent attention, pooled
    /// indexer and KDA entries takes: latent attention at up to
    /// `step_rows` rows (its split partials grow with the rows), the
    /// indexer at every bucket and KDA at a prefill of `prefill_rows` rows.
    fn workspace_bytes(shape: Shape, step_rows: u32, prefill_rows: u32) -> u64;

    /// The rows per item of each KDA verify entry's checkpoint table (its
    /// compiled `max_rows`).
    fn kda_checkpoint_rows(shape: Shape) -> Vec<u32>;

    /// One rank's state of the MoE sublayers' F32 all-reduce among `ranks`
    /// ranks of `hidden` wide rows: the low-latency round entry's and the
    /// phased entries', in bytes.
    fn moe_all_reduce_state_bytes(ranks: u32, hidden: u32) -> Result<u64, String>;

    // Context parallelism (`crate::context_parallel`).

    /// One group all-gather's state among `size` members, each sending
    /// `words` 32-bit words.
    fn group_gather(size: u32, words: u64) -> Result<GatherBytes, String>;
}
