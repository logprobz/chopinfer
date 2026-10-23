//! Wide decode HC boundaries: the column-block producer and split finish,
//! and the served producer with the two-CTA finish, against the served
//! two-launch boundary at Decode-16 and Decode-32 on real captured rows.
mod artifacts;
#[path = "../hc_replay_buffers.rs"]
#[allow(
    dead_code,
    reason = "shared replay buffers; Input::new reads one capture"
)]
mod buffers;
mod cases;
#[path = "../hc_replay_corpus.rs"]
#[allow(dead_code, reason = "shared authenticated capture loader")]
mod corpus;
mod device;
mod profile;
mod replay;
mod timing;
use bundle_deepseek41::sm120::hc_cols as cols;
use bundle_deepseek41::sm120::hc_finish_twin as twin;
