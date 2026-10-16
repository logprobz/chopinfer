//! The per-channel-decay delta rule's streamed prefill stages and their
//! mutants (`linear::channel_decay::streamed_lab_cubin_specs`).

use kernel_build::CubinSpec;

pub const FAMILY: &str = "channel_decay_streamed";

pub fn cubins() -> Vec<CubinSpec> {
    kernels_sm120::linear::channel_decay::streamed_lab_cubin_specs()
}
