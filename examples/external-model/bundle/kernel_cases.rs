//! Example-owned synthetic cases of the example's ops (`ops` in the
//! contract crate). These declarations do not qualify deployment.

use super::ops::{self, CAPACITY, WIDTH};
use kernel_core::codec::{bf16_to_f32, f32_to_bf16};
use kernel_core::contract::{Refusal, RmsNormKind};
use kernel_core::oracle::{self, OracleOutput};

pub const LIVE_ROWS: [u32; 4] = [0, 1, 3, CAPACITY];
pub const PADDING: u16 = 0x42c8;
pub const PROBE_VALUES: [u32; 3] = [0, 1, 8];

pub fn input() -> Vec<u16> {
    (0..WIDTH * CAPACITY as usize)
        .map(|i| f32_to_bf16(((i % 29) as f32 - 14.0) * 0.0625))
        .collect()
}

pub fn weights() -> Vec<u16> {
    (0..WIDTH)
        .map(|i| f32_to_bf16(0.8125 + (i % 7) as f32 * 0.0625))
        .collect()
}

/// Use the existing BF16 oracle bound in kernel-core::oracle::check_numeric.
/// Zero live rows has no arithmetic outputs; its case checks untouched padding.
pub fn rms_reference(live_rows: u32) -> Result<Option<OracleOutput>, Refusal> {
    if live_rows > CAPACITY {
        return Err(Refusal("example live rows exceed capacity"));
    }
    if live_rows == 0 {
        return Ok(None);
    }
    let decode = |v: &u16| f64::from(bf16_to_f32(*v));
    oracle::rmsnorm(
        &input()[..WIDTH * live_rows as usize]
            .iter()
            .map(decode)
            .collect::<Vec<_>>(),
        WIDTH,
        &weights().iter().map(decode).collect::<Vec<_>>(),
        f64::from(ops::EPSILON),
        RmsNormKind::Plain,
        None,
    )
    .map(Some)
}
