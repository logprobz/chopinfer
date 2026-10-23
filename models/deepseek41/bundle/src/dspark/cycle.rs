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
    pub packed_diagnostics: std::cell::RefCell<Option<Vec<u8>>>,
    pub feedback: std::cell::RefCell<Option<super::prefetch::Feedback>>,
}

pub(crate) struct Cycle {
    pub hash: HashDevice,
    pub memory: Option<HashMemory>,
    pub diagnostics: Option<DeviceBufferSpanV1>,
    pub layers: Vec<LayerGather>,
    pub gathering: bool,
    /// Whether decode steps without drafts upload their host-hashed rows
    /// through `layers` (`instance::ar_engram`'s switch).
    pub undrafted: bool,
    /// Such a step's ids, hashed and awaiting its target launch.
    pub deferred: Option<Deferred>,
    pub draft_done: Vec<runtime_bundle_abi::BundleEventHandleV1>,
    pub draft_joined: Option<runtime_bundle_abi::BundleEventHandleV1>,
    pub boundary_ready: Vec<runtime_bundle_abi::BundleEventHandleV1>,
    pub boundary_enabled: bool,
    pub pending: Option<Pending>,
    /// Whether drafted steps gather each item's first rows early
    /// (`early_enabled`), decided at create.
    pub early_enabled: bool,
    /// Rank 0's side stream producing the early ids, so their reads wait
    /// for no queued device work, and the ids' storage on it. Both are
    /// acquired at create when enabled.
    pub early_stream: Option<runtime_bundle_abi::BundleStreamHandleV1>,
    pub early_ids: Option<DeviceBufferSpanV1>,
    pub early: Option<Early>,
}

/// A decode step's host-hashed ids, laid out as the device hash writes
/// them: each layer's `stride` rows in turn.
pub(crate) struct Deferred {
    pub ids: Vec<u8>,
    pub rows: u32,
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

/// Whether drafted steps gather each item's first Engram rows before the
/// drafter: on unless `CHOPINFER_DSPARK_ENGRAM_EARLY` holds something other
/// than `1`, so `0` keeps every row behind the drafter.
pub(crate) fn early_enabled() -> bool {
    #[cfg(test)]
    return EARLY_IN_TESTS.get();
    #[cfg(not(test))]
    early_decision(std::env::var_os("CHOPINFER_DSPARK_ENGRAM_EARLY").as_deref())
}

/// The switch's decision for its raw value, `None` being unset.
fn early_decision(value: Option<&std::ffi::OsStr>) -> bool {
    value.is_none_or(|value| value == "1")
}

#[cfg(test)]
thread_local! {
    /// The switch in tests. Tests gather every row behind the drafter unless
    /// they set it, so their pinned journals stay that route's.
    static EARLY_IN_TESTS: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Runs `run` as if the early switch held `value`.
#[cfg(test)]
pub(crate) fn with_early<T>(value: bool, run: impl FnOnce() -> T) -> T {
    struct Restore(bool);
    impl Drop for Restore {
        fn drop(&mut self) {
            EARLY_IN_TESTS.set(self.0);
        }
    }
    let _restore = Restore(EARLY_IN_TESTS.replace(value));
    run()
}

#[cfg(test)]
#[test]
fn early_rows_default_on_and_keep_the_control() {
    use std::ffi::OsStr;
    assert!(early_decision(None));
    assert!(early_decision(Some(OsStr::new("1"))));
    for value in ["0", "", "invalid"] {
        assert!(!early_decision(Some(OsStr::new(value))));
    }
}

/// A layer owns its pinned gather slot and the target piece consuming it.
pub(crate) struct LayerGather {
    pub layer: u32,
    pub gather: BundleMappedGatherHandleV1,
    pub indices: DeviceBufferSpanV1,
    pub destinations: Vec<DeviceBufferSpanV1>,
    pub graphs: GraphCache<DeepSeek>,
}
impl Cycle {
    pub fn new(hash: HashDevice) -> Self {
        Self {
            hash,
            memory: None,
            diagnostics: None,
            layers: Vec::new(),
            gathering: false,
            undrafted: false,
            deferred: None,
            draft_done: Vec::new(),
            draft_joined: None,
            boundary_ready: Vec::new(),
            boundary_enabled: std::env::var_os("CHOPINFER_DSPARK_BOUNDARY_PACKET")
                .is_none_or(|value| value == "1"),
            pending: None,
            early_enabled: early_enabled(),
            early_stream: None,
            early_ids: None,
            early: None,
        }
    }
}
