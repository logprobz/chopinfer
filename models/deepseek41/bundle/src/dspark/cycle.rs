//! Storage owned by the overlapped target path. The lifecycle owns commits.
use bundle_runtime::graphs::GraphCache;
use model_deepseek41_contract::serving::DeepSeek;
use runtime_bundle_abi::{BundleMappedGatherHandleV1, DeviceBufferSpanV1};

use super::execution::{DraftInput, Input};
use super::staging::DeviceTarget;
use crate::ops::engram_hash::{HashDevice, HashMemory};

pub(crate) struct Pending {
    pub inputs: Vec<Input>,
    pub drafts: Vec<DraftInput>,
    pub history: Vec<i64>,
    pub text: Vec<u8>,
    pub result: Option<DeviceTarget>,
}

pub(crate) struct Cycle {
    pub hash: HashDevice,
    pub memory: Option<HashMemory>,
    pub diagnostics: Option<DeviceBufferSpanV1>,
    pub gather: Option<BundleMappedGatherHandleV1>,
    pub gathering: bool,
    pub draft_done: Vec<runtime_bundle_abi::BundleEventHandleV1>,
    pub draft_joined: Option<runtime_bundle_abi::BundleEventHandleV1>,
    pub pending: Option<Pending>,
    pub tail: Option<GraphCache<DeepSeek>>,
}
impl Cycle {
    pub fn new(hash: HashDevice) -> Self {
        Self {
            hash,
            memory: None,
            diagnostics: None,
            gather: None,
            gathering: false,
            draft_done: Vec::new(),
            draft_joined: None,
            pending: None,
            tail: None,
        }
    }
}
