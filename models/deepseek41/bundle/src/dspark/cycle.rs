//! Storage owned by the overlapped target path. The lifecycle owns commits.
use bundle_runtime::graphs::GraphCache;
use model_deepseek41_contract::serving::DeepSeek;
use runtime_bundle_abi::{BundleMappedGatherHandleV1, DeviceBufferSpanV1};

use super::execution::{DraftInput, Input};
use super::staging::DeviceTarget;
use crate::ops::engram_hash::{HashDevice, HashMemory};

pub(crate) const BOUNDARY_BYTES: u64 = 64 * 1024;

pub(crate) struct Pending {
    pub inputs: Vec<Input>,
    pub drafts: Vec<DraftInput>,
    pub history: Vec<i64>,
    pub text: Vec<u8>,
    pub result: Option<DeviceTarget>,
    /// The step's allocation, when the host allocated its drafts.
    pub allocation: Option<super::allocation::Step>,
    pub packed_diagnostics: std::cell::RefCell<Option<Vec<u8>>>,
    pub feedback: std::cell::RefCell<Option<super::prefetch::Feedback>>,
}

pub(crate) struct Cycle {
    pub hash: HashDevice,
    pub memory: Option<HashMemory>,
    pub diagnostics: Option<DeviceBufferSpanV1>,
    pub layers: Vec<LayerGather>,
    pub gathering: bool,
    /// An undrafted decode step's ids, hashed and awaiting its target
    /// launch, or a mixed step's verify ids awaiting its prefill's last
    /// Engram join.
    pub deferred: Option<Deferred>,
    pub draft_done: Vec<runtime_bundle_abi::BundleEventHandleV1>,
    pub draft_joined: Option<runtime_bundle_abi::BundleEventHandleV1>,
    pub boundary_ready: Vec<runtime_bundle_abi::BundleEventHandleV1>,
    pub pending: Option<Pending>,
    /// Rank 0's side stream producing the early ids, so their reads wait
    /// for no queued device work, and the ids' storage on it. Both are
    /// acquired at create.
    pub early_stream: Option<runtime_bundle_abi::BundleStreamHandleV1>,
    pub early_ids: Option<DeviceBufferSpanV1>,
    pub early: Option<Early>,
}

/// A decode step's host-hashed ids, laid out as the device hash writes
/// them: each layer's `stride` rows in turn.
pub(crate) struct Deferred {
    pub ids: Vec<u8>,
    pub rows: u32,
    /// The gathers of a mixed step's verify rows; none for an undrafted
    /// decode step, whose target launch begins them.
    pub mixed: Option<Mixed>,
}

/// A mixed step's verify-row gathers (`Instance::admits_mixed_engram`),
/// which its prefill begins after its last Engram join: the layers begun,
/// and the joins the verify launch made, layer by layer and each layer
/// rank by rank, so a failed step's settlement makes only the rest.
#[derive(Default)]
pub(crate) struct Mixed {
    pub begun: std::cell::Cell<usize>,
    pub joined: std::cell::Cell<usize>,
}

/// A drafted step's early gathers. Each item's first target row (its bonus
/// or decode token) is known before the drafter runs, so the host hashes
/// those rows and these gathers read them while the drafter runs. The late
/// gathers (`LayerGather`) then find those rows in the resident row cache.
/// The late gathers still copy every row the target reads; the early copies
/// land in scratch that no kernel reads.
pub(crate) struct Early {
    /// The host-hashed ids, laid out as the device hash writes them.
    pub ids: DeviceBufferSpanV1,
    /// One gather per Engram layer, in layer order.
    pub gathers: Vec<BundleMappedGatherHandleV1>,
    /// The rows of the step whose early gathers are in flight.
    pub rows: Option<u32>,
}

/// A target layer attaches its graph cache to the common gather slot.
pub(crate) type LayerGather = crate::instance::layer_gather::LayerGather<GraphCache<DeepSeek>>;
impl Cycle {
    pub fn new(hash: HashDevice) -> Self {
        Self {
            hash,
            memory: None,
            diagnostics: None,
            layers: Vec::new(),
            gathering: false,
            deferred: None,
            draft_done: Vec::new(),
            draft_joined: None,
            boundary_ready: Vec::new(),
            pending: None,
            early_stream: None,
            early_ids: None,
            early: None,
        }
    }
}
