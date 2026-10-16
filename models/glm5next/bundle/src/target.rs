//! What GLM's generic bundle code asks of the GPU target an instance runs
//! on, beyond the ops the target prepares (`crate::ops`): the backend and
//! own kernels those ops are prepared on, the modules an instance loads and
//! the dynamic shared memory their entries request, the spinning comm
//! entries create admits against the devices' residency, the bounds of the
//! phased all-reduce's state, and the vision encoder's kernels.
//!
//! An instance builds its target once at create (`instance::target`);
//! `crate::sm120` implements it for SM120.

use std::collections::BTreeMap;

use bundle_runtime::acquire::{Log, RankHandles};
use bundle_runtime::encoder::Encoder;
use bundle_runtime::pipeline::PreparedLaunch;
use bundle_runtime::residency::SpinningEntry;
use bundle_runtime::resources::Held;
use kernel_core::backend::{Backend, Need, Refused};
use kernel_core::precision::OperationUse;
use model_glm5next_contract::own_ops::OwnOps;
use model_glm5next_contract::quantization::Layout;
use runtime_bundle_abi::{
    BundleAbiError, BundleCudaSymbolHandleV1, DeviceBufferSpanV1 as Span, HostServicesV1,
};

use crate::ops::Served;
use crate::ops::own::StackedOps;

/// What an instance runs, as far as the modules it loads depend on it.
#[derive(Clone, Copy, Debug)]
pub struct Uses {
    /// The checkpoint layout, which picks the MoE projections' formats.
    pub layout: &'static Layout,
    /// Context-parallel groups above size 1.
    pub context_parallel: bool,
    /// The Sinkhorn mHC comm epilogue (`options::COMM_EPILOGUE_OPTION`).
    pub comm_epilogue: bool,
    /// The MTP drafter, whose argmax gathers its pairs through the
    /// context-parallel group all-gather
    /// (`model_glm5next_contract::draft_argmax`).
    pub drafter: bool,
}

/// The spans the indexer's pooling gate binds: its BF16 side tensor, the
/// normed rows it reads, the logits it writes and the projections' scratch
/// partial and semaphores.
#[derive(Clone, Copy, Debug)]
pub struct PoolGateSpans {
    pub weight: Span,
    pub activation: Span,
    pub output: Span,
    pub partial: Span,
    pub semaphores: Span,
}

/// One topology's target.
pub trait Target: Send + Sync {
    /// The backend whose ops an instance serving a checkpoint stored in
    /// `layout` prepares, the queries its entries serve and GLM's own
    /// kernels (`Served`).
    fn backend(&self, layout: &'static Layout) -> Result<(Box<dyn Backend>, Served), String>;

    /// The dynamic shared memory each entry of the modules an instance
    /// serving `layout` loads requests, by symbol.
    fn shared_bytes(&self, layout: &'static Layout) -> Result<BTreeMap<String, u32>, String>;

    /// Whether an instance that runs `uses` loads the text model's `cubin`
    /// at create: modules of entries it leaves unused stay unloaded (the
    /// vision encoder's load with the encoder).
    fn loads(&self, cubin: &str, uses: Uses) -> bool;

    /// The low-latency all-reduce entries (the sublayers' of BF16 partials,
    /// the MoE sublayers' rounded one of F32 partials), each at its largest
    /// plan, which create admits against the devices' residency together.
    fn low_latency(&self) -> Result<Vec<SpinningEntry>, String>;

    /// The phased all-reduce's served entries (of BF16 partials, then the
    /// MoE sublayers' F32 rounding ones), each at its largest plan.
    fn phased(&self) -> Result<Vec<SpinningEntry>, String>;

    /// The runs in the phased all-reduce's scope after which every rank's
    /// phased flags and epoch are zeroed while idle.
    fn phased_reset_runs(&self) -> u64;

    /// The most collective calls one rank's program of a run may make, which
    /// that reset bound counts on.
    fn phased_calls_per_run(&self) -> u64;

    /// The indexer's pooling gate logits of `rows` rows: a GEMM of the
    /// entry the layer's key projection `key` runs at `usage`, over the
    /// gate's BF16 side tensor, which no op prepares (it is not a declared
    /// projection).
    fn pool_gate(
        &self,
        key: &str,
        usage: OperationUse,
        rows: u32,
        spans: PoolGateSpans,
    ) -> Result<PreparedLaunch, String>;

    /// The KDA layers' stacked input projections prepared on `own` at each
    /// decode and verify use a stacked entry runs, each for the need `need`
    /// gives its use; none on a target without one.
    fn stacked(
        &self,
        own: &dyn OwnOps,
        need: &dyn Fn(OperationUse) -> Need,
    ) -> Result<StackedOps, Refused>;

    /// Loads the vision encoder's cubins on every rank: per rank, each
    /// entry's symbol, configured for the dynamic shared memory it plans.
    fn load_encoder(
        &self,
        log: Log,
        ranks: &[RankHandles],
        held: &mut Vec<Held>,
    ) -> Result<Vec<BTreeMap<String, BundleCudaSymbolHandleV1>>, BundleAbiError>;

    /// The vision encoder's runtime over each rank's encoder image
    /// `weights`, created on the target's encoder kernels, every rank's
    /// regions allocated with `allocate`.
    fn encoder(
        &self,
        services: &HostServicesV1,
        weights: Vec<BTreeMap<String, Span>>,
        allocate: &mut dyn FnMut(usize, u64) -> Result<Span, BundleAbiError>,
    ) -> Result<Encoder, BundleAbiError>;

    /// The smallest row capacity bucket of the target's row kernels that
    /// holds `rows` rows: a row launch there writes what one at any larger
    /// bucket writes, without the larger grid's empty CTAs.
    fn row_bucket(&self, rows: u32) -> Option<u32>;
}
