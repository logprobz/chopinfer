//! The comm residency check against the host's own residency service, over
//! `HostServicesV1Backend<FakeHostDeviceV1>`: one rank per device, a wide
//! entry that falls back and an entry that does not.

use std::collections::BTreeMap;
use std::rc::Rc;

use bundle_runtime::residency::{SpinningEntry, check_entries};
use kernel_core::contract::LaunchPlan;
use runtime_bundle_abi::{
    BUNDLE_CLUSTER_RESIDENCY_V1_CLUSTERS_REPORTED, BUNDLE_CLUSTER_RESIDENCY_V1_MPS_CLIENT,
    BundleRawClusterResidencyV1, HostServicesTableV1,
};
use runtime_host::fake::FakeHostDeviceV1;
use runtime_host::{HostInstanceConfigV1, HostServicesV1Backend, HostTopologyV1};

const ORDINALS: [i32; 4] = [0, 1, 2, 3];

/// 136-CTA grids (17 clusters): a wide entry and one without a fallback.
const ENTRIES: [SpinningEntry; 2] = [
    SpinningEntry {
        symbol: "wide_v1",
        plan: LaunchPlan {
            grid: [136, 1, 1],
            block: [256, 1, 1],
            dynamic_shared_bytes: 0,
            workspace_bytes: 0,
            cluster: [8, 1, 1],
            pdl: true,
        },
        wide: true,
    },
    SpinningEntry {
        symbol: "phased_v1",
        plan: LaunchPlan {
            grid: [136, 1, 1],
            block: [256, 1, 1],
            dynamic_shared_bytes: 0,
            workspace_bytes: 0,
            cluster: [8, 1, 1],
            pdl: true,
        },
        wide: false,
    },
];

/// Runs the check on a host whose device reports `report` (none when
/// unset), returning its result and the host's log lines.
fn run(
    report: Option<BundleRawClusterResidencyV1>,
    unchecked: bool,
) -> (Result<bool, String>, Vec<String>) {
    let config = HostInstanceConfigV1::new(HostTopologyV1::new(ORDINALS.to_vec()).unwrap());
    let backend = Rc::new(HostServicesV1Backend::new(FakeHostDeviceV1::new(), config));
    if let Some(report) = report {
        backend.with_device_mut(|device| device.report_cluster_residency(report));
    }
    let table = HostServicesTableV1::new(Rc::clone(&backend));
    let services = table.caller();
    let symbols: Vec<BTreeMap<String, _>> = (0..ORDINALS.len())
        .map(|rank| {
            let context = services
                .select_context(rank as u32, ORDINALS[rank])
                .unwrap();
            let module = services.load_cuda_module(context, b"fake-module").unwrap();
            ENTRIES
                .iter()
                .map(|entry| entry.symbol.to_owned())
                .map(|name| {
                    let symbol = services
                        .resolve_cuda_symbol(module, name.as_bytes())
                        .unwrap();
                    (name, symbol)
                })
                .collect()
        })
        .collect();
    let result = check_entries(
        services, "Test", &ORDINALS, &symbols, &ENTRIES, true, unchecked,
    );
    let logs = backend.with_device(|device| {
        device
            .logs()
            .iter()
            .map(|line| String::from_utf8_lossy(line).into_owned())
            .collect()
    });
    (result, logs)
}

fn report(clusters: u32, flags: u32) -> BundleRawClusterResidencyV1 {
    BundleRawClusterResidencyV1 {
        multiprocessor_count: 188,
        usable_multiprocessors: 188,
        max_active_clusters: clusters,
        flags: flags | BUNDLE_CLUSTER_RESIDENCY_V1_CLUSTERS_REPORTED,
        ..BundleRawClusterResidencyV1::unreported()
    }
}

#[test]
fn grids_that_fit_keep_the_wide_entry() {
    // One rank's 17 clusters against 46, as measured on the target, and a
    // line per rank records it.
    let (result, logs) = run(Some(report(46, 0)), false);
    assert_eq!(result, Ok(true));
    assert!(
        logs.iter().any(|line| line
            == "Test comm residency rank 3 (1 rank(s) on the device), clusters needed of \
                resident: wide_v1 17 of 46, phased_v1 17 of 46"),
        "{logs:?}"
    );
}

#[test]
fn a_host_that_reports_nothing_authorizes_nothing_unless_unchecked() {
    let (result, logs) = run(None, false);
    let error = result.unwrap_err();
    // The wide entry was turned off before the refusal, with its reason.
    assert!(
        logs.iter()
            .any(|line| line.contains("Test wide all-reduce off")),
        "{logs:?}"
    );
    assert!(error.contains("reports no cluster occupancy"), "{error}");
    assert!(
        error.contains("CHOPINFER_COMM_RESIDENCY=unchecked"),
        "{error}"
    );
    // The operator's override creates, with the wide entry off.
    assert_eq!(run(None, true).0, Ok(false));
    // A count reported without its flag is not a report either.
    let mut unflagged = report(46, 0);
    unflagged.flags = 0;
    assert!(run(Some(unflagged), false).0.is_err());
}

#[test]
fn an_mps_client_is_refused_unless_unchecked() {
    let mps = report(46, BUNDLE_CLUSTER_RESIDENCY_V1_MPS_CLIENT);
    let error = run(Some(mps), false).0.unwrap_err();
    assert!(error.contains("MPS"), "{error}");
    assert_eq!(run(Some(mps), true).0, Ok(false));
}

#[test]
fn a_measured_shortfall_fails_even_unchecked() {
    // 16 clusters cannot hold one rank's 17.
    for unchecked in [false, true] {
        let error = run(Some(report(16, 0)), unchecked).0.unwrap_err();
        assert!(error.contains("phased_v1"), "{error}");
        assert!(error.contains("fewer clusters"), "{error}");
    }
    // A reported zero holds nothing.
    assert!(run(Some(report(0, 0)), true).0.is_err());
}
