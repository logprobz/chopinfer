//! Fixed-input device replay and its exact-bit evidence format.
#![cfg_attr(not(feature = "linux-cuda-host"), allow(dead_code))]
#[cfg(all(target_os = "linux", feature = "linux-cuda-host"))]
mod device;
mod diagnostic;
mod record;
#[cfg(test)]
mod tests;
mod trace;
