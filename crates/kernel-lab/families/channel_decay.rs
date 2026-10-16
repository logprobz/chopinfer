//! The per-channel-decay delta rule engine (`linear::channel_decay`).

use kernel_build::CubinSpec;

pub const FAMILY: &str = "channel_decay";

pub fn cubins() -> Vec<CubinSpec> {
    kernels_sm120::linear::channel_decay::cubin_specs()
}
