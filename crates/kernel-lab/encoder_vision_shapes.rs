//! The vision encoder row entries' lab profile, at the served tower's shapes
//! and capacity (32,768 rows): the zero-column pad of the 1176-wide patches
//! to 1216, the per-head q/k RMSNorm of 16 + 16 heads of 64, the clamped
//! SwiGLU row passes of 4096 and 10240, and the F32 GELU of 4096. The symbols are the ones a declaration's catalog derives for the
//! same variant ids, so the lab runs the served bodies. Shared by build.rs,
//! which builds the cubin in the `encoder_vision` lock family, and
//! `tests/encoder_vision_rows.rs`.

use kernel_core::contract::MultimodalQuery;
use kernels_sm120::encoder::{Entry, Kernel};

/// The lock family of the profile's cubin in `kernels.lock.d`.
#[allow(dead_code)] // Read by the build script only.
pub const FAMILY: &str = "encoder_vision";

/// The profile's cubin name.
pub const CUBIN: &str = "encoder_vision_rows_sm120";

pub const CAPACITY: u32 = 32_768;
pub const EPS: f32 = 1e-5;
pub const LIMIT: f32 = 10.0;

pub fn entries() -> Vec<Entry> {
    let swiglu = |hidden| MultimodalQuery::SwigluClampedF32 {
        hidden,
        limit_bits: LIMIT.to_bits(),
        capacity: CAPACITY,
    };
    [
        Kernel::new(
            "pad.zero-columns.v1",
            None,
            MultimodalQuery::PadColumnsF32 {
                input: 1176,
                output: 1216,
                capacity: CAPACITY,
            },
        ),
        Kernel::new(
            "attention.varlen.qk-rms.pairs.v1",
            Some("qk_norm"),
            MultimodalQuery::QkRmsNormF32 {
                q_heads: 16,
                kv_heads: 16,
                head_dim: 64,
                eps_bits: EPS.to_bits(),
                capacity: CAPACITY,
            },
        ),
        Kernel::new("mlp.swiglu-clamped.v1", Some("swiglu"), swiglu(4096)),
        Kernel::new("mlp.swiglu-clamped.v1", Some("swiglu"), swiglu(10240)),
        Kernel::new(
            "projector.norm-gelu-mlp.v1",
            Some("gelu"),
            MultimodalQuery::GeluRowsF32 {
                width: 4096,
                capacity: CAPACITY,
            },
        ),
    ]
    .iter()
    .map(|kernel| kernel.entry().expect("lab vision row entries are valid"))
    .collect()
}

/// The profile's cubin.
#[allow(dead_code)] // Read by the build script only.
pub fn cubin() -> kernel_build::CubinSpec {
    let entries = entries();
    kernels_sm120::encoder::kernel::cubin_spec(
        CUBIN,
        vec![],
        entries
            .iter()
            .map(|entry| (entry.symbol.as_str(), &entry.query, entry.params.as_slice())),
    )
    .expect("lab vision row cubin")
}
