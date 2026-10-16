//! The target device the fake host models (test code): its residency
//! reports, and each entry's resources from a `kernels.lock`.

use std::collections::BTreeMap;

use bundle_runtime::fake_host::{DeviceModel, device_report, modelled_report};
use kernel_core::occupancy::EntryResources;
use kernels_sm120::device::{MEASURED_CLUSTERS_OF_8_AT_TWO_CTAS, TARGET};
use runtime_bundle_abi::{BundleRawClusterResidencyV1, LaunchConfigV1};

/// A report of all the target's SMs with `clusters` clusters resident at
/// once, and `flags`.
pub fn residency(clusters: u32, flags: u32) -> BundleRawClusterResidencyV1 {
    device_report(&TARGET, clusters, flags)
}

/// What the target reports for the comm entries at two CTAs per SM.
pub fn comm_residency() -> BundleRawClusterResidencyV1 {
    residency(MEASURED_CLUSTERS_OF_8_AT_TWO_CTAS, 0)
}

/// What the target's driver reports for an entry with `resources` launched
/// at `config`.
pub fn report(resources: EntryResources, config: LaunchConfigV1) -> BundleRawClusterResidencyV1 {
    modelled_report(&TARGET, resources, config)
}

/// Each entry's resources in the `kernels.lock` text `lock`, by symbol.
pub fn locked_resources(lock: &str) -> BTreeMap<String, EntryResources> {
    kernels_sm120::backend::locked_resources(lock).expect("a kernels.lock lists resources")
}

/// The target, running each entry at `resources`.
pub fn model(resources: BTreeMap<String, EntryResources>) -> DeviceModel {
    DeviceModel {
        device: TARGET,
        resources,
    }
}
