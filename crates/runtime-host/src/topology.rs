//! Device selection uses CUDA's visible ordinals, never physical ordinals.

/// Rank-ordered visible CUDA ordinals requested by a serving command.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DeviceSelectionV1(pub Vec<i32>);

pub(crate) fn parse_devices(value: &str) -> Result<Vec<i32>, String> {
    let mut devices = Vec::new();
    for part in value.split(',') {
        let ordinal = if !part.is_empty() && part.bytes().all(|byte| byte.is_ascii_digit()) {
            part.parse::<i32>().ok()
        } else {
            None
        }
        .ok_or_else(|| {
            format!("--devices requires comma-separated nonnegative ordinals, got '{value}'")
        })?;
        if devices.contains(&ordinal) {
            return Err(format!(
                "--devices contains duplicate visible ordinal {ordinal}"
            ));
        }
        devices.push(ordinal);
    }
    Ok(devices)
}

fn select_devices(
    topology: &str,
    ranks: u32,
    visible: usize,
    requested: Option<&[i32]>,
    visibility: Option<&std::ffi::OsStr>,
) -> Result<Vec<i32>, String> {
    let count = requested.map_or(visible, <[i32]>::len);
    let visibility = visibility.map_or_else(|| "unset".to_owned(), |value| format!("{value:?}"));
    if count != ranks as usize || ranks == 0 {
        let selected = requested.map_or(String::new(), |_| format!("; --devices selects {count}"));
        return Err(format!(
            "package topology {topology} needs {ranks} devices; {visible} are visible \
             (CUDA_VISIBLE_DEVICES {visibility}){selected}; set CUDA_VISIBLE_DEVICES or pass \
             --devices with exactly {ranks} distinct visible ordinals"
        ));
    }
    let devices = match requested {
        Some(devices) => devices.to_vec(),
        None => (0..ranks)
            .map(|rank| i32::try_from(rank).map_err(|_| "the declared rank count is too large"))
            .collect::<Result<_, _>>()?,
    };
    for (rank, &ordinal) in devices.iter().enumerate() {
        if ordinal < 0 || ordinal as usize >= visible {
            return Err(format!(
                "--devices ordinal {ordinal} for rank {rank} is out of range; {visible} devices \
                 are visible (CUDA_VISIBLE_DEVICES {visibility}); use ordinals within the visible set"
            ));
        }
        if devices[..rank].contains(&ordinal) {
            return Err(format!(
                "--devices contains duplicate visible ordinal {ordinal}"
            ));
        }
    }
    Ok(devices)
}

pub(crate) fn serving_devices(
    topology: &str,
    ranks: u32,
    requested: Option<&[i32]>,
    cuda: bool,
) -> Result<Vec<i32>, String> {
    #[cfg(all(target_os = "linux", feature = "linux-cuda-host"))]
    if cuda {
        let driver = runtime_cuda::Driver::initialize()
            .map_err(|error| format!("cannot open the device driver: {error}"))?;
        let visible = driver
            .visible_device_count()
            .map_err(|error| error.to_string())?;
        let devices = select_devices(
            topology,
            ranks,
            visible,
            requested,
            std::env::var_os("CUDA_VISIBLE_DEVICES").as_deref(),
        )?;
        precheck_peers(&driver, &devices)?;
        for (rank, &ordinal) in devices.iter().enumerate() {
            let pci_bus_id = driver
                .device_pci_bus_id(ordinal)
                .map_err(|error| error.to_string())?;
            let uuid = driver
                .device_uuid(ordinal)
                .map_err(|error| error.to_string())?;
            eprintln!(
                "startup device rank={rank} visible_ordinal={ordinal} pci_bus_id={pci_bus_id} uuid={uuid}"
            );
        }
        return Ok(devices);
    }
    // Host-memory services have one synthetic device per declared rank and do
    // not consult CUDA or its process environment.
    let _ = cuda;
    select_devices(topology, ranks, ranks as usize, requested, None)
}

#[cfg(all(target_os = "linux", feature = "linux-cuda-host"))]
fn precheck_peers(driver: &runtime_cuda::Driver, devices: &[i32]) -> Result<(), String> {
    let mut checked = std::collections::BTreeSet::new();
    for (rank, &device) in devices.iter().enumerate() {
        for (peer_rank, &peer) in devices.iter().enumerate() {
            if device == peer || !checked.insert((device, peer)) {
                continue;
            }
            let pair = format!(
                "rank {rank} (visible ordinal {device}) -> rank {peer_rank} (visible ordinal {peer})"
            );
            let capable = driver
                .devices_can_access_peer(device, peer)
                .map_err(|error| format!("cannot check CUDA peer access for {pair}: {error}"))?;
            if !capable {
                return Err(format!(
                    "CUDA peer access is unavailable for {pair}; select devices with peer access in both directions"
                ));
            }
        }
    }
    Ok(())
}

/// Each device's ordinal, rank minimum and free bytes.
#[cfg(all(target_os = "linux", feature = "linux-cuda-host"))]
fn device_floors(devices: &[i32], minimum_per_rank: u64) -> Result<Vec<(i32, u64, u64)>, String> {
    let driver = runtime_cuda::Driver::initialize().map_err(|error| error.to_string())?;
    let ordinals = devices
        .iter()
        .copied()
        .collect::<std::collections::BTreeSet<_>>();
    ordinals
        .into_iter()
        .map(|ordinal| {
            let context = driver
                .retain_primary_context(ordinal)
                .map_err(|error| error.to_string())?;
            let (free, _) = context.memory_info().map_err(|error| error.to_string())?;
            Ok((ordinal, minimum_per_rank, free))
        })
        .collect()
}

/// The warnings for devices with less free memory than their rank's minimum,
/// before any bundle uploads, without refusing them: a package's minimum
/// keeps a reserve for modules, sampling and KV that a device below it
/// leaves short, and the rank serves with a smaller KV pool.
#[cfg(all(target_os = "linux", feature = "linux-cuda-host"))]
pub(crate) fn memory_warnings(devices: &[i32], minimum_per_rank: u64) -> Vec<String> {
    match device_floors(devices, minimum_per_rank) {
        Ok(floors) => floors
            .into_iter()
            .filter(|&(_, required, free)| required > free)
            .map(|(ordinal, required, free)| {
                format!(
                    "startup memory warning: device {ordinal} has {free} bytes free, below the package's {required}-byte minimum; serving continues with less than the reserve that minimum keeps"
                )
            })
            .collect(),
        Err(error) => vec![format!(
            "startup memory warning: free device memory was not checked against the package's minimum: {error}"
        )],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::OsStr;

    #[cfg(all(target_os = "linux", feature = "cuda-fake-driver-tests"))]
    #[test]
    fn ordinal_memory_queries_are_sorted_and_unique() {
        use runtime_cuda::fake_driver;
        fake_driver::reset();
        let floors = device_floors(&[3, 1, 3, 0, 2], 1024).unwrap();
        assert_eq!(
            floors.iter().map(|floor| floor.0).collect::<Vec<_>>(),
            [0, 1, 2, 3]
        );
        assert!(floors.iter().all(|floor| floor.1 == 1024));
        assert_eq!(fake_driver::call_count(22), 4);
        let warnings = memory_warnings(&[3, 1, 0, 2], 64 << 30);
        assert_eq!(warnings.len(), 4);
        for (ordinal, warning) in warnings.iter().enumerate() {
            assert!(warning.contains(&format!("device {ordinal}")), "{warning}");
        }
    }

    /// Ranks below their minimum are warned about, not refused.
    #[cfg(all(target_os = "linux", feature = "cuda-fake-driver-tests"))]
    #[test]
    fn ranks_below_their_minimum_warn() {
        use runtime_cuda::fake_driver;
        let devices = [0, 1];
        fake_driver::reset();
        assert!(memory_warnings(&devices, 1024).is_empty());
        let warnings = memory_warnings(&devices, 64 << 30);
        assert_eq!(warnings.len(), 2);
        assert!(warnings[0].contains("device 0") && warnings[0].contains("less than the reserve"));
        assert_eq!(fake_driver::call_count(8), 0);
    }

    #[test]
    fn every_rank_count_uses_all_visible_devices_by_default() {
        for ranks in [1, 2, 4, 8] {
            for visibility in [None, Some(OsStr::new("GPU-example,2,3,4"))] {
                assert_eq!(
                    select_devices("test", ranks, ranks as usize, None, visibility).unwrap(),
                    (0..ranks as i32).collect::<Vec<_>>()
                );
            }
        }
    }

    #[test]
    fn subsets_and_rank_order_use_visible_ordinals() {
        assert_eq!(
            select_devices("test", 2, 4, Some(&[3, 1]), None).unwrap(),
            [3, 1]
        );
        assert_eq!(
            select_devices("test", 2, 2, Some(&[1, 0]), Some(OsStr::new("2,3"))).unwrap(),
            [1, 0]
        );
        assert_eq!(select_devices("test", 1, 8, Some(&[7]), None).unwrap(), [7]);
    }

    #[test]
    fn count_refusals_explain_visibility_and_recovery() {
        for (ranks, visible, requested, visibility, expected) in [
            (2, 4, None, None, "CUDA_VISIBLE_DEVICES unset"),
            (
                4,
                2,
                None,
                Some(OsStr::new("2,3")),
                "CUDA_VISIBLE_DEVICES \"2,3\"",
            ),
            (4, 0, None, Some(OsStr::new("")), "0 are visible"),
            (2, 4, Some(&[0][..]), None, "--devices selects 1"),
        ] {
            let error = select_devices("test", ranks, visible, requested, visibility).unwrap_err();
            assert!(
                error.contains(&format!("package topology test needs {ranks} devices")),
                "{error}"
            );
            assert!(error.contains(expected), "{error}");
            assert!(
                error.contains("set CUDA_VISIBLE_DEVICES or pass --devices"),
                "{error}"
            );
        }
    }

    #[test]
    fn invalid_ordinals_are_refused() {
        for devices in [&[1, 1], &[0, 4], &[-1, 0]] {
            assert!(select_devices("test", 2, 4, Some(devices), None).is_err());
        }
        for input in ["", "0,", "0,,1", "-1", "+1", "GPU-id", "0,0", "2147483648"] {
            assert!(parse_devices(input).is_err(), "{input}");
        }
        assert_eq!(parse_devices("3,1,0,2").unwrap(), [3, 1, 0, 2]);
    }

    #[test]
    fn host_memory_mapping_stays_rank_generic() {
        for ranks in [1, 2, 4, 8] {
            assert_eq!(
                serving_devices("test", ranks, None, false).unwrap(),
                (0..ranks as i32).collect::<Vec<_>>()
            );
        }
    }

    #[cfg(all(target_os = "linux", feature = "cuda-fake-driver-tests"))]
    #[test]
    fn fake_driver_visibility_and_prechecks() {
        use runtime_cuda::fake_driver;
        const CHILD: &str = "CHOPINFER_TOPOLOGY_TEST_CHILD";
        if let Ok(case) = std::env::var(CHILD) {
            fake_driver::reset();
            let (ranks, requested, expected) = match case.as_str() {
                "unset" => (8, None, (0..8).collect::<Vec<_>>()),
                "one" => (1, None, vec![0]),
                "two" => (2, None, vec![0, 1]),
                "four" => (4, None, vec![0, 1, 2, 3]),
                "subset" => (2, Some(vec![3, 1]), vec![3, 1]),
                "mismatch" | "empty" => (4, None, vec![]),
                "duplicates" => (2, Some(vec![0, 0]), vec![]),
                "out-of-range" => (2, Some(vec![0, 2]), vec![]),
                _ => panic!("unknown child case"),
            };
            let result = serving_devices("fixture", ranks, requested.as_deref(), true);
            if expected.is_empty() {
                let expected_error = match case.as_str() {
                    "duplicates" => "duplicate visible ordinal 0",
                    "out-of-range" => "ordinal 2 for rank 1 is out of range",
                    _ => "needs 4 devices",
                };
                let error = result.unwrap_err();
                assert!(error.contains(expected_error), "{error}");
                assert_eq!(fake_driver::call_count(21), 0);
            } else {
                assert_eq!(result.unwrap(), expected);
                assert_eq!(fake_driver::call_count(21), (ranks * (ranks - 1)) as usize);
                let driver = runtime_cuda::Driver::initialize().unwrap();
                if case == "two" {
                    assert_eq!(driver.device_pci_bus_id(0).unwrap(), "0000:03:00.0");
                    assert!(driver.device_uuid(0).unwrap().starts_with("GPU-03030303"));
                }
                if ranks > 1 {
                    fake_driver::deny_peer_access(expected[1], expected[0]);
                    let error = precheck_peers(&driver, &expected).unwrap_err();
                    assert!(error.contains("rank 1"), "{error}");
                    assert!(error.contains("peer access is unavailable"), "{error}");
                    fake_driver::fail_next(21, 801);
                    assert!(
                        precheck_peers(&driver, &expected)
                            .unwrap_err()
                            .contains("cannot check CUDA peer access")
                    );
                }
            }
            // Preflight never retains a primary context or uploads weights.
            assert_eq!(fake_driver::call_count(22), 0);
            assert_eq!(fake_driver::call_count(8), 0);
            return;
        }
        for (case, visible) in [
            ("unset", None),
            ("one", Some("7")),
            ("two", Some("2,3")),
            ("four", Some("0,1,2,3")),
            ("subset", Some("4,5,6,7")),
            ("mismatch", Some("2,3")),
            ("empty", Some("")),
            ("duplicates", Some("2,3")),
            ("out-of-range", Some("2,3")),
        ] {
            let mut command = std::process::Command::new(std::env::current_exe().unwrap());
            command
                .args([
                    "--exact",
                    "topology::tests::fake_driver_visibility_and_prechecks",
                    "--nocapture",
                ])
                .env(CHILD, case)
                .env_remove("CUDA_VISIBLE_DEVICES");
            if let Some(visible) = visible {
                command.env("CUDA_VISIBLE_DEVICES", visible);
            }
            let output = command.output().unwrap();
            assert!(
                output.status.success(),
                "{case}: {}{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
            if case == "two" {
                let log = String::from_utf8_lossy(&output.stderr);
                assert!(
                    log.contains(
                        "rank=0 visible_ordinal=0 pci_bus_id=0000:03:00.0 uuid=GPU-03030303"
                    ),
                    "{log}"
                );
            }
        }
    }
}
