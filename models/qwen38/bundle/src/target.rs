//! What Qwen's generic code asks of the GPU target an instance runs on,
//! beyond the ops the target's backend prepares (`crate::ops`): the backend
//! and own kernels themselves, which compiled entry serves an op where the
//! op's contract depends on it, the modules, workspace and shared memory an
//! instance loads and sizes at create, its comm state's bytes, the comm
//! entries whose residency create checks, the pipelined prefill's lanes and
//! the grids they move, what the precision component's registered entries
//! are, and the vision encoder's kernels.
//!
//! An instance builds its target once at create (`instance::target`);
//! `crate::sm120` implements it for SM120.

use bundle_runtime::lanes::{Config, Lane, LaunchFacts};
use bundle_runtime::projection::Needs;
use bundle_runtime::residency::SpinningEntry;
use kernel_core::backend::{Backend, Refused};
use kernel_core::contract::{AttentionPhase, Refusal};
use kernel_core::gated_mlp::{GatedMlpProjection, GatedMlpProjectionQuery};
use model_qwen38_contract::own_ops::OwnOps;

use crate::ops::Key;
use crate::pipeline::PreparedLaunch;
use crate::scratch::{CommBytes, Served};

/// The vision encoder's kernels: the backend that prepares its slots, and
/// what each slot (stage, part, block) runs.
pub(crate) struct EncoderKernels {
    pub backend: Box<dyn Backend>,
    pub slots: Box<bundle_runtime::encoder::Slots<'static>>,
}

/// What an instance runs, as far as the modules it loads depend on it.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Uses {
    /// The rows of the instance's largest graph bucket.
    pub graph_rows: u32,
    /// The HC producer bottleneck (`Switches::hc_precompute`).
    pub hc_precompute: bool,
    /// The padded up mix layout (`Switches::hc_padded_mix`).
    pub hc_padded_mix: bool,
    /// Context-parallel groups above size 1.
    pub context_parallel: bool,
}

/// One topology's target.
pub(crate) trait Target: Send + Sync {
    /// The backend whose ops an instance prepares. Graphs take the wide
    /// all-reduce entries only when `wide_comm`.
    fn backend(&self, wide_comm: bool) -> Box<dyn Backend>;

    /// Qwen's own kernels.
    fn own(&self) -> &dyn OwnOps;

    /// The contract of the grouped expert projection `projection` a step of
    /// `key` runs: the entry's query states its routed capacity, so Qwen
    /// asks for the contract of the entry the step runs.
    fn expert_contract(
        &self,
        projection: GatedMlpProjection,
        key: Key,
    ) -> Result<GatedMlpProjectionQuery, Refused>;

    /// The phase of the selected partials a context-parallel step of `key`
    /// runs.
    fn partials_phase(&self, key: Key) -> AttentionPhase;

    /// The pipelined prefill's lanes, from the environment, if the target
    /// pipelines prefills at the instance's rank count.
    fn lanes(&self) -> Option<Config>;

    /// Moves `launch`'s grid for `lane` by the facts of the op that made it,
    /// as the target's kernels allow.
    fn adjust(&self, launch: &mut PreparedLaunch, facts: LaunchFacts, lane: Lane, config: &Config);

    /// The lane facts of a launch Qwen built without an op (a registered
    /// adapter's, a stored projection's, the media inject), by its symbol.
    fn built(&self, launch: &PreparedLaunch) -> LaunchFacts;

    /// Whether an instance that runs `uses` loads the target's `cubin`, as
    /// far as the target decides: modules of entries it leaves unused stay
    /// unloaded.
    fn loads(&self, cubin: &str, uses: Uses) -> bool;

    /// Whether an instance that runs `uses` loads `symbol` of a cubin it
    /// loads.
    fn loads_symbol(&self, symbol: &str, uses: Uses) -> bool;

    /// The comm state of a rank serving `served`: the decode all-reduce's
    /// grows for the wide entry when graphs reach above the decode entries'
    /// rows.
    fn comm_bytes(&self, served: Served) -> CommBytes;

    /// The largest workspace any stage of a rank serving `served` asks for.
    fn workspace_bytes(&self, served: Served) -> Result<u64, Refusal>;

    /// The dynamic shared memory the compiled entry `symbol` requests at
    /// its largest plan.
    fn dynamic_shared_bytes(&self, symbol: &str) -> Result<u32, Refusal>;

    /// The spinning comm entries of an instance whose graph buckets reach
    /// `rows` rows, each at its largest plan, with the F32 norm-weight forms
    /// when `f32_norms` and the comm lane's grids when it pipelines on
    /// `lanes`; create checks each against the device.
    fn spinning_entries(
        &self,
        rows: u32,
        f32_norms: bool,
        lanes: Option<&Config>,
    ) -> Result<Vec<SpinningEntry>, Refusal>;

    /// What the registered dense entry `symbol` needs beyond its views.
    fn registered_dense(&self, symbol: &str) -> Result<Needs, String>;

    /// The contract the registered grouped expert entry `symbol` computes,
    /// and the bytes of its output at its routed capacity.
    fn registered_experts(&self, symbol: &str) -> Result<(GatedMlpProjectionQuery, u64), String>;

    /// The kernels of the vision encoder of the target's geometry.
    fn encoder(&self) -> Result<EncoderKernels, String>;
}
