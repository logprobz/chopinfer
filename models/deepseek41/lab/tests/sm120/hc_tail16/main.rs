//! Decode-16 HC boundaries: the served rounded expansion mixer followed by
//! the BF16 input norm against the two-launch finish/norm tail, on 16-row
//! boundaries of real captured rows.
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
mod replay;
mod timing;
