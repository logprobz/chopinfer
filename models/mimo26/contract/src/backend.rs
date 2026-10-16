//! The kernel backend MiMo's contract plans for. Ops a bundle launches are
//! prepared at create through `kernel_core::backend`; what the contract
//! itself needs of the backend it reads through [`Kernels`]. Only
//! [`Selected`] names the backend; MiMo's SM120 module implements it.

use crate::Tp;
use crate::capabilities::Projection;
use kernel_core::contract::{InstantiationSpec, Refusal, Target};
use kernel_core::precision::OperationUse;
use kernel_core::precision::capability::{CapabilityDocument, OperationCapability};

/// The backend MiMo's contract and bundle are built for.
pub type Selected = crate::sm120::Sm120;

/// The flag and epoch bytes one comm protocol needs in a comm state.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct StateBytes {
    pub flags: u64,
    pub epoch: u64,
}

/// Which of a declared projection's bound alternatives a use binds: its
/// served entry, or for experts the inexact one and for dense projections
/// the medium-M one (the operation ID's `.inexact` or `.mid` suffix).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Alternative {
    Served,
    Inexact,
    Medium,
}

/// The kernels a routing stage names.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RouteKernels {
    /// The router or route kernel, or the fused router's operation.
    pub symbol: String,
    /// The tiled permutation's kernels, in launch order, where the stage
    /// routes as prefill does (`serving::prefill_route`); empty otherwise.
    pub permutation: Vec<String>,
}

/// What MiMo's contract needs of a kernel backend.
pub trait Kernels {
    /// The target the backend's operations are specified for
    /// (`OperationSpec::target`, `ImportedProjection::target`).
    const TARGET: Target;

    /// The device architecture a package built for the backend declares
    /// and a host admits it on (`BundleTargetV1`'s architectures,
    /// `BundleHostCompatibilityV1`).
    const ARCHITECTURE: &'static str;

    /// The entry of the bundle's own kernel `name` (`ring_backup`,
    /// `dflash_mask_rows`, ...) in `tp`'s cubins.
    fn bundle_entry(tp: Tp, name: &str) -> String;

    /// The bundle's own module `name` (`ring_slots`, ...) of `tp`'s cubins.
    fn bundle_cubin(tp: Tp, name: &str) -> String;

    // Comm and scratch sizes (`crate::resources`).

    /// The hidden all-reduces' state at `ranks` ranks for up to `rows` rows.
    fn all_reduce_state(ranks: u32, rows: u32) -> Result<StateBytes, Refusal>;

    /// The inbox bytes the hidden all-reduces push `rows` rows through at
    /// `ranks` ranks.
    fn all_reduce_inbox_bytes(ranks: u32, rows: u32) -> Result<u64, Refusal>;

    /// The logits row gather's state.
    const GATHER_ROWS_STATE: StateBytes;

    /// The logits row exchange's state.
    const EXCHANGE_ROWS_STATE: StateBytes;

    /// Bytes of the parallel decode route's arrival count.
    const ROUTE_TICKET_BYTES: u64;

    /// The largest scratch workspace any kernel a `tp` rank serving `items`
    /// can select needs.
    fn workspace_bytes(tp: Tp, items: bundle_runtime::geometry::Items) -> Result<u64, Refusal>;

    // Capability offers (`crate::capabilities`).

    /// Every offer of every projection of `geometry` at every supported rank
    /// count.
    fn capabilities(
        geometry: &'static crate::geometry::Geometry,
    ) -> Result<CapabilityDocument, String>;

    /// The catalog offer of a drafter projection, which the generated
    /// inventory does not list.
    fn drafter_capability(
        projection: Projection,
        tp: Tp,
        usage: OperationUse,
    ) -> Result<OperationCapability, Refusal>;

    // Precision bindings (`crate::prepared`).

    /// The entry a projection use binds, which the bundle launches.
    type Entry: Clone;

    /// The entry serving `projection` at `usage` as `alternative`, given
    /// the projection's capability, whose stages name the entry. The
    /// backend may restate the capability: experts take their grouped
    /// entry's, medium-M uses their entry's stage.
    fn bind(
        tp: Tp,
        projection: Projection,
        usage: OperationUse,
        alternative: Alternative,
        capability: &mut OperationCapability,
    ) -> Result<Self::Entry, String>;

    /// Whether a dense entry has a medium-M stand-in; `None` for an entry
    /// that is not dense.
    fn has_medium(entry: &Self::Entry) -> Option<bool>;

    /// The identity of every kernel a `tp` component's bindings can select,
    /// whichever the process's switches choose.
    fn dependency(tp: Tp) -> String;

    // Stage kernels (`crate::projections::variants`).

    /// The fused gate/up entry the legacy dense stage of `usage` names
    /// (DFlash's own where `dflash`), with its symbol.
    fn gate_up_entry(
        tp: Tp,
        usage: OperationUse,
        dflash: bool,
    ) -> Result<(String, &'static InstantiationSpec), Refusal>;

    /// Whether the routing stage of `usage` runs the router fused with its
    /// route.
    fn fused_router(usage: OperationUse) -> bool;

    /// The kernels of `usage`'s routing stage: the fused router where
    /// `fused`, else the standalone router or route.
    fn route_kernels(tp: Tp, usage: OperationUse, fused: bool) -> RouteKernels;

    /// The standalone expert combine of the stages that do not fold it.
    fn combine_symbol(tp: Tp) -> String;

    // Encoder kernels (the bundle's encoder admission).

    /// The symbol of every kernel the backend runs `geometry`'s encoder
    /// declarations (`crate::encoders::spec`) on, its own kernels for MiMo's
    /// registered stages included: an instance serves media only where its
    /// modules hold them all.
    fn encoder_symbols(
        geometry: &'static crate::geometry::Geometry,
    ) -> Result<Vec<String>, encoder_spec::Error>;
}
