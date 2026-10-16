//! Residency of comm grids whose CTAs spin on flags other grids release.
//!
//! A peer-to-peer all-reduce's workers wait for data their own rank's and the
//! other ranks' sending CTAs publish. If spinning CTAs fill the SMs before a
//! sending cluster is placed, the call never finishes: a grid that is only
//! partly resident hangs rather than slows down. An MPS client shares SMs
//! with other processes' spinning grids.
//!
//! So a bundle asks the host at create, per rank and spinning entry, what the
//! device reports (`query_cluster_residency_v1`) and checks the largest plan
//! it will launch against it with the ranks that share the device. The check
//! reads the driver's own answer, `cuOccupancyMaxActiveClusters` for the
//! entry's symbol and shape, rather than a model of the GPU, and fails closed:
//! a report it cannot trust (none at all, an SM-limited context, an MPS
//! client) authorizes nothing unless the operator says so ([`UNCHECKED`]).

use kernel_core::contract::LaunchPlan;
use runtime_bundle_abi::{
    BUNDLE_CLUSTER_RESIDENCY_V1_CLUSTERS_REPORTED, BUNDLE_CLUSTER_RESIDENCY_V1_MPS_CLIENT,
    BundleAbiError, BundleCudaSymbolHandleV1, BundleRawClusterResidencyV1, HostServicesV1,
};

use std::collections::BTreeMap;

use crate::dispatch::launch_config;

/// `CHOPINFER_COMM_RESIDENCY=unchecked` lets an instance create when the
/// device cannot vouch for its spinning comm grids (no report, an SM-limited
/// context, an MPS client), logging each entry it could not check. The wide
/// all-reduce stays off, and a measured shortfall still fails create.
pub const UNCHECKED: &str = "CHOPINFER_COMM_RESIDENCY";

/// Whether the operator set [`UNCHECKED`] to `unchecked`.
#[must_use]
pub fn unchecked_from_environment() -> bool {
    std::env::var(UNCHECKED).as_deref() == Ok("unchecked")
}

/// One comm entry whose CTAs spin on flags other grids release, at the
/// largest plan an instance launches it with.
#[derive(Clone, Copy, Debug)]
pub struct SpinningEntry {
    pub symbol: &'static str,
    pub plan: LaunchPlan,
    /// The wide all-reduce, which graphs replace with the phased stages when
    /// it may not fit; the other entries have no fallback.
    pub wide: bool,
}

/// Keeps, per (symbol, cluster size), the entry with the largest grid: the
/// one a residency check needs.
#[must_use]
pub fn largest(entries: impl IntoIterator<Item = SpinningEntry>) -> Vec<SpinningEntry> {
    let grid = |entry: &SpinningEntry| entry.plan.grid.iter().product::<u32>();
    let mut kept: BTreeMap<(&'static str, u32), SpinningEntry> = BTreeMap::new();
    for entry in entries {
        let key = (entry.symbol, entry.plan.cluster.iter().product::<u32>());
        if kept.get(&key).is_none_or(|old| grid(old) < grid(&entry)) {
            kept.insert(key, entry);
        }
    }
    kept.into_values().collect()
}

/// Ranks whose ordinal equals `rank`'s: the grids that run on its device at
/// once.
#[must_use]
pub fn ranks_on_device(ordinals: &[i32], rank: usize) -> u32 {
    ordinals.get(rank).map_or(0, |own| {
        ordinals.iter().filter(|&o| o == own).count() as u32
    })
}

/// What the host reports for `symbol` launched at `plan`'s shape.
pub fn query(
    services: &HostServicesV1,
    symbol: BundleCudaSymbolHandleV1,
    plan: LaunchPlan,
) -> Result<BundleRawClusterResidencyV1, BundleAbiError> {
    let config =
        launch_config(plan).map_err(|_| BundleAbiError::InvalidRange("residency launch plan"))?;
    services.query_cluster_residency(symbol, config)
}

/// Whether grids of one entry are resident at once.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Verdict {
    /// They fit the clusters the driver keeps resident.
    Fits,
    /// The device cannot vouch for them: no cluster count reported (an older
    /// host, a query that could not run), or a context limited to fewer SMs
    /// than the device has, whose relation to the driver's count is not
    /// pinned down.
    Unknown(&'static str),
    /// The process may be an MPS client: other clients' spinning grids share
    /// its SMs, whatever this process's own grids need.
    Mps,
    /// The device keeps fewer clusters resident than the grids need.
    Short,
}

/// Whether `ranks` grids of `plan` are resident at once on the reporting
/// device: `ranks` times the plan's clusters against the clusters the driver
/// reports it keeps resident.
#[must_use]
pub fn check(plan: &LaunchPlan, ranks: u32, report: &BundleRawClusterResidencyV1) -> Verdict {
    if report.flags & BUNDLE_CLUSTER_RESIDENCY_V1_MPS_CLIENT != 0 {
        return Verdict::Mps;
    }
    if report.flags & BUNDLE_CLUSTER_RESIDENCY_V1_CLUSTERS_REPORTED == 0 {
        return Verdict::Unknown("the host reports no cluster occupancy");
    }
    let cluster = plan.cluster.iter().product::<u32>().max(1);
    let clusters = plan.grid.iter().product::<u32>().div_ceil(cluster);
    if u64::from(ranks) * u64::from(clusters) > u64::from(report.max_active_clusters) {
        return Verdict::Short;
    }
    if report.usable_multiprocessors != 0
        && report.multiprocessor_count != 0
        && report.usable_multiprocessors < report.multiprocessor_count
    {
        return Verdict::Unknown("the context may use fewer SMs than the device has");
    }
    Verdict::Fits
}

/// Checks every entry on every rank against what the host reports, with one
/// rank per device (`ordinals`). An MPS client may share SMs with other
/// processes. Returns whether the wide all-reduce stays on (`wide` in, and
/// it fits on every rank).
///
/// Fails closed: an entry without a fallback that falls short is an error,
/// and so is one the device cannot vouch for (no report, an SM-limited
/// context, an MPS client) unless `unchecked` ([`UNCHECKED`]), which logs it
/// instead. The wide all-reduce is turned off, with a log line, unless it
/// fits. Entries a rank did not load are skipped.
pub fn check_entries(
    services: &HostServicesV1,
    model: &str,
    ordinals: &[i32],
    symbols: &[BTreeMap<String, BundleCudaSymbolHandleV1>],
    entries: &[SpinningEntry],
    mut wide: bool,
    unchecked: bool,
) -> Result<bool, String> {
    for (rank, symbols) in symbols.iter().enumerate() {
        let sharing = ranks_on_device(ordinals, rank);
        // What fit, for one log line per rank: each entry's clusters for the
        // ranks on the device against the clusters the driver reports.
        let mut fitted = Vec::new();
        for entry in entries {
            if entry.wide && !wide {
                continue;
            }
            let name = entry.symbol;
            let Some(&symbol) = symbols.get(name) else {
                continue;
            };
            let report = query(services, symbol, entry.plan)
                .map_err(|error| format!("rank {rank} {name} residency: {error}"))?;
            let verdict = check(&entry.plan, sharing, &report);
            let need = u64::from(sharing)
                * u64::from(
                    entry
                        .plan
                        .grid
                        .iter()
                        .product::<u32>()
                        .div_ceil(entry.plan.cluster.iter().product::<u32>().max(1)),
                );
            let reason = match verdict {
                Verdict::Fits => {
                    fitted.push(format!("{name} {need} of {}", report.max_active_clusters));
                    continue;
                }
                Verdict::Unknown(reason) => reason,
                Verdict::Mps => {
                    "the process may be an MPS client, and multi-rank serving under MPS is \
                     unsupported: other clients' spinning grids can hold the SMs these need"
                }
                Verdict::Short => "the device keeps fewer clusters resident than these grids need",
            };
            let message = format!(
                "rank {rank}: {name} ({} CTAs in clusters of {}, {sharing} grid(s) on the device, \
                 {} clusters resident, {} of {} SMs): {reason}",
                entry.plan.grid.iter().product::<u32>(),
                entry.plan.cluster.iter().product::<u32>(),
                report.max_active_clusters,
                report.usable_multiprocessors,
                report.multiprocessor_count,
            );
            if entry.wide {
                let _ = services.log(
                    format!(
                        "{model} wide all-reduce off, graphs take the phased stages: {message}"
                    )
                    .as_bytes(),
                );
                wide = false;
            } else if unchecked && verdict != Verdict::Short {
                let _ = services.log(
                    format!("{model} comm residency unchecked ({UNCHECKED}=unchecked): {message}")
                        .as_bytes(),
                );
            } else if verdict == Verdict::Short {
                return Err(message);
            } else {
                return Err(format!(
                    "{message}; set {UNCHECKED}=unchecked to create anyway"
                ));
            }
        }
        if !fitted.is_empty() {
            let placement = format!("{sharing} rank(s) on the device");
            let _ = services.log(
                format!(
                    "{model} comm residency rank {rank} ({placement}), \
                     clusters needed of resident: {}",
                    fitted.join(", ")
                )
                .as_bytes(),
            );
        }
    }
    Ok(wide)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plan(grid: u32) -> LaunchPlan {
        LaunchPlan {
            grid: [grid, 1, 1],
            block: [256, 1, 1],
            dynamic_shared_bytes: 0,
            workspace_bytes: 0,
            cluster: [8, 1, 1],
            pdl: true,
        }
    }

    fn report(sms: u32, usable: u32, clusters: u32, flags: u32) -> BundleRawClusterResidencyV1 {
        BundleRawClusterResidencyV1 {
            multiprocessor_count: sms,
            usable_multiprocessors: usable,
            max_active_clusters: clusters,
            flags,
            ..BundleRawClusterResidencyV1::unreported()
        }
    }

    const REPORTED: u32 = BUNDLE_CLUSTER_RESIDENCY_V1_CLUSTERS_REPORTED;

    #[test]
    fn ranks_count_every_grid_on_their_device() {
        assert_eq!(ranks_on_device(&[0, 1, 2, 3], 2), 1);
        assert_eq!(ranks_on_device(&[0, 0, 1, 1, 2, 2, 3, 3], 5), 2);
        assert_eq!(ranks_on_device(&[0, 0, 0, 1], 0), 3);
        assert_eq!(ranks_on_device(&[0, 0, 0, 1], 3), 1);
        assert_eq!(ranks_on_device(&[0, 1], 4), 0);
    }

    #[test]
    fn grids_fit_the_clusters_the_driver_keeps_resident() {
        // 136 CTAs are 17 clusters: two ranks need 34, three 51.
        let full = report(188, 188, 44, REPORTED);
        assert_eq!(check(&plan(136), 1, &full), Verdict::Fits);
        assert_eq!(check(&plan(136), 2, &full), Verdict::Fits);
        assert_eq!(check(&plan(136), 3, &full), Verdict::Short);
        assert_eq!(check(&plan(72), 4, &full), Verdict::Fits);
        assert_eq!(check(&plan(72), 5, &full), Verdict::Short);
        // A reported count of zero holds nothing.
        assert_eq!(
            check(&plan(16), 1, &report(188, 188, 0, REPORTED)),
            Verdict::Short
        );
        // An SM-limited context is short when its count says so, and
        // otherwise unknown: its count's semantics are not pinned down.
        assert_eq!(
            check(&plan(136), 3, &report(188, 94, 44, REPORTED)),
            Verdict::Short
        );
        assert!(matches!(
            check(&plan(16), 1, &report(188, 94, 44, REPORTED)),
            Verdict::Unknown(_)
        ));
        // An MPS client never fits, whatever its count.
        let mps = report(
            188,
            188,
            44,
            REPORTED | BUNDLE_CLUSTER_RESIDENCY_V1_MPS_CLIENT,
        );
        assert_eq!(check(&plan(16), 1, &mps), Verdict::Mps);
        // No report authorizes nothing; an unreported affinity is no limit.
        assert!(matches!(
            check(&plan(16), 1, &BundleRawClusterResidencyV1::unreported()),
            Verdict::Unknown(_)
        ));
        assert!(matches!(
            check(&plan(16), 1, &report(188, 188, 44, 0)),
            Verdict::Unknown(_)
        ));
        assert_eq!(
            check(&plan(136), 2, &report(188, 0, 44, REPORTED)),
            Verdict::Fits
        );
    }

    #[test]
    fn the_largest_grid_of_each_symbol_and_cluster_is_kept() {
        let entry = |symbol, grid, cluster| SpinningEntry {
            symbol,
            plan: LaunchPlan {
                cluster: [cluster, 1, 1],
                ..plan(grid)
            },
            wide: false,
        };
        let kept = largest([
            entry("a", 16, 8),
            entry("a", 40, 8),
            entry("a", 24, 8),
            entry("a", 4, 2),
            entry("b", 8, 8),
        ]);
        let shapes: Vec<_> = kept
            .iter()
            .map(|entry| (entry.symbol, entry.plan.grid[0], entry.plan.cluster[0]))
            .collect();
        assert_eq!(shapes, [("a", 4, 2), ("a", 40, 8), ("b", 8, 8)]);
    }
}
