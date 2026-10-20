//! One-launch decode HC boundaries against the served producer and fused
//! tail on every real captured C1 and C8 boundary.
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
