//! The target device the fake host models (test code): its residency
//! reports, and each entry's resources from the bundle's `kernels.lock`.

use bundle_runtime::fake_host::{DeviceModel, device_report};
use kernels_sm120::device::{MEASURED_CLUSTERS_OF_8_AT_TWO_CTAS, TARGET};
use runtime_bundle_abi::BundleRawClusterResidencyV1;

/// A report of all the target's SMs with `clusters` clusters resident at
/// once, and `flags`.
pub fn residency(clusters: u32, flags: u32) -> BundleRawClusterResidencyV1 {
    device_report(&TARGET, clusters, flags)
}

/// What the target reports for the comm entries at two CTAs per SM.
pub fn comm_residency() -> BundleRawClusterResidencyV1 {
    residency(MEASURED_CLUSTERS_OF_8_AT_TWO_CTAS, 0)
}

/// The target, running each entry at the resources the bundle's
/// `kernels.lock` records.
pub fn locked() -> DeviceModel {
    DeviceModel {
        device: TARGET,
        resources: kernels_sm120::backend::locked_resources(include_str!("../../kernels.lock"))
            .expect("the bundle's kernels.lock lists resources"),
    }
}
