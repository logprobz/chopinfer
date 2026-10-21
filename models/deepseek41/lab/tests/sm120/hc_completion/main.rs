//! One-launch HC boundaries and the two-CTA finish against the served
//! producer and fused tail on every real captured boundary of the selected
//! row counts (C1 and C8 decode, Verify-6).
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
use bundle_deepseek41::sm120::hc_completion as registration;
use bundle_deepseek41::sm120::hc_finish_twin as twin;
