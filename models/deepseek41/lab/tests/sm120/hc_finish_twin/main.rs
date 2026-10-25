//! Verify-6 twin32 finish against the two-stage reference on captured HC boundaries.
mod artifacts;
#[path = "../hc_replay_buffers.rs"]
mod buffers;
#[path = "../hc_replay_corpus.rs"]
#[allow(dead_code, reason = "shared authenticated capture loader")]
mod corpus;
mod device;
mod replay;
mod served;
mod timing;
use bundle_deepseek41::sm120::hc_finish_twin as twin;
