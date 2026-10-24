//! Fixed-input device replay and its exact-bit evidence format.
#![cfg_attr(not(feature = "linux-cuda-host"), allow(dead_code))]
#[cfg(all(target_os = "linux", feature = "linux-cuda-host"))]
mod device;
mod diagnostic;
mod record;
#[cfg(test)]
mod tests;
mod trace;
mod transition;
#[cfg(test)]
mod transition_tests;

#[cfg(all(target_os = "linux", feature = "linux-cuda-host"))]
pub(super) fn replay(configuration: &std::path::Path) -> Result<String, String> {
    device::replay(configuration).map_err(|error| error.to_string())?;
    Ok("fixed-batch replay passed".into())
}
