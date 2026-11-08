//! The successor comparison controls and dumps of a state-capture build
//! (`instance::dspark::successor_capture`), and their absence from every
//! other build.
#[cfg(feature = "state-capture")]
use super::results::batch;
use super::wide_successor::{cycle, prompted};
use super::*;
use std::path::{Path, PathBuf};

/// A fresh capture directory under `$CHOPINFER_DATA`, enabled, holding
/// `controls` and a `successor.json` for the eight items of [`batch`],
/// each accepting `accepted` drafts with bonus 7.
fn capture_directory(label: &str, controls: &[&str], allocated: bool, accepted: u32) -> PathBuf {
    let root = PathBuf::from(std::env::var_os("CHOPINFER_DATA").expect("external test data"))
        .join("deepseek41-tests")
        .join(format!("dspark-successor-{label}-{}", std::process::id()));
    if root.exists() {
        std::fs::remove_dir_all(&root).unwrap();
    }
    std::fs::create_dir_all(&root).unwrap();
    for name in ["enabled"].iter().chain(controls) {
        std::fs::write(root.join(name), []).unwrap();
    }
    let items: Vec<_> = (1..=8)
        .map(|request| serde_json::json!({"request": request, "accepted": accepted, "bonus": 7}))
        .collect();
    std::fs::write(
        root.join("successor.json"),
        serde_json::to_vec(
            &serde_json::json!({"version": 1, "allocated": allocated, "items": items}),
        )
        .unwrap(),
    )
    .unwrap();
    root
}

fn dumps(root: &Path) -> Vec<String> {
    let mut names: Vec<_> = std::fs::read_dir(root)
        .unwrap()
        .map(|entry| entry.unwrap().file_name().into_string().unwrap())
        .filter(|name| name.starts_with("successor-step-"))
        .collect();
    names.sort();
    names
}

/// The controls and dumps exist only in a state-capture build. The same
/// capture directory, named by `CHOPINFER_DSPARK_STATE_DIR` as a host
/// names it, holds every control: there the device-result step queues no
/// successor and dumps itself, and every other build queues the successor
/// and writes nothing. The step runs in a fresh process of this test
/// binary, so the variable reaches only that process's instance.
#[test]
fn successor_controls_exist_only_in_state_capture_builds() {
    let root = capture_directory("build", &["successors-off", "successor-steps"], false, 3);
    let status = std::process::Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "fake_host_tests::dspark::successor_capture::a_device_result_step_under_every_control",
            "--ignored",
            "--nocapture",
        ])
        .env("CHOPINFER_DSPARK_STATE_DIR", &root)
        .status()
        .unwrap();
    assert!(status.success());
    let expected: &[&str] = if cfg!(feature = "state-capture") {
        &["successor-step-2"]
    } else {
        &[]
    };
    assert_eq!(dumps(&root), expected);
    std::fs::remove_dir_all(root).unwrap();
}

/// The step [`successor_controls_exist_only_in_state_capture_builds`]
/// runs: eight items' device-result verify step, which queues a successor
/// unless a control turns it off.
#[test]
#[ignore = "run by successor_controls_exist_only_in_state_capture_builds in a fresh process"]
fn a_device_result_step_under_every_control() {
    assert!(std::env::var_os("CHOPINFER_DSPARK_STATE_DIR").is_some());
    let (host, mut instance) = prompted(8, 64, &[]);
    cycle(&host, &mut instance, (8, 3, 3), (24, 3), 1).unwrap();
    assert_eq!(
        crate::instance::dspark_successor_for_test(instance.handle()),
        (!cfg!(feature = "state-capture")).then_some(8)
    );
    instance
        .commit_accepted(&(1..=8).map(|id| kept(id, 4)).collect::<Vec<_>>())
        .unwrap();
    for request in 1..=8 {
        instance.retire_request(request).unwrap();
    }
    instance.destroy().unwrap();
    assert_eq!(host.acquired.get(), host.released.get());
}

/// A legacy verify step queues the successor `successor.json` names, and
/// the next step reuses it whole: an allocated five-head one. With
/// `successors-off` the same steps queue none and draft inline. Each
/// drafted step dumps every rank's saved status, draft outputs and item
/// rings, and its heads and successor outcome, once: its commit writes
/// none after the dump is taken away.
#[cfg(feature = "state-capture")]
#[test]
fn forced_feedback_and_the_off_switch_drive_both_arms_and_dump_each_step() {
    for (label, off, heads) in [("whole", false, 5), ("inline", true, 3)] {
        let mut controls = vec!["successor-steps"];
        if off {
            controls.push("successors-off");
        }
        let root = capture_directory(label, &controls, true, 2);
        let (host, mut instance) = prompted(8, 64, &[]);
        let handle = instance.handle();
        crate::instance::set_dspark_state_capture_root_for_test(handle, Some(root.clone()));
        execute(&mut instance, &host, &batch(8, 3, 3));
        std::fs::remove_file(root.join("successor.json")).unwrap();
        assert_eq!(
            crate::instance::dspark_successor_for_test(handle),
            (!off).then_some(8),
            "{label}"
        );
        // A test takes each dump away; the commit writes none.
        let take = |step: u32| {
            assert_eq!(dumps(&root), [format!("successor-step-{step}")]);
            std::fs::rename(
                root.join(format!("successor-step-{step}")),
                root.join(format!("taken-step-{step}")),
            )
            .unwrap();
        };
        take(2);
        instance
            .commit_accepted(&(1..=8).map(|id| kept(id, 3)).collect::<Vec<_>>())
            .unwrap();
        assert!(dumps(&root).is_empty(), "{label}");
        execute(&mut instance, &host, &batch(8, 6, 3));
        take(3);
        let record = |step: u32| -> serde_json::Value {
            serde_json::from_slice(
                &std::fs::read(root.join(format!("taken-step-{step}/step.json"))).unwrap(),
            )
            .unwrap()
        };
        let (queued, next) = (record(2), record(3));
        assert_eq!(queued["consumed"], "none", "{label}");
        assert_eq!(queued["queued"], !off, "{label}");
        assert_eq!(queued["computed_positions"], 3, "{label}");
        assert_eq!(
            next["consumed"],
            if off { "none" } else { "reused" },
            "{label}"
        );
        assert_eq!(next["queued"], false, "{label}");
        assert_eq!(next["computed_positions"], heads, "{label}");
        assert_eq!(next["drafting"], 8);
        // Every decoder's context holds its projected rows at each stage:
        // the prompt's 3 after the queueing step's catch-up, 6 once the
        // next step adopted or caught up the 3 committed rows.
        for (step, record) in [(2, &queued), (3, &next)] {
            let projected = if step == 2 { 3 } else { 6 };
            let rings = record["rings"].as_array().unwrap();
            assert_eq!(rings.len(), 8, "{label}");
            for ring in rings {
                let positions = ring["positions"].as_array().unwrap();
                assert_eq!(positions.len() as u32, FLASH.dspark_stages());
                assert!(
                    positions
                        .iter()
                        .all(|range| *range == serde_json::json!([0, projected])),
                    "{label} step {step}: {ring}"
                );
            }
        }
        for step in [2, 3] {
            let ring = std::fs::metadata(root.join(format!("taken-step-{step}/ring-rank-0.bin")))
                .unwrap()
                .len();
            assert!(ring > 0 && ring % 8 == 0);
            for rank in 0..4 {
                let file = |name: &str| {
                    std::fs::metadata(
                        root.join(format!("taken-step-{step}/{name}-rank-{rank}.bin")),
                    )
                    .unwrap()
                    .len()
                };
                assert_eq!(file("drafts"), 8 * 5 * 4 * 2, "{label}");
                assert_eq!(file("ring"), ring, "{label}");
                assert!(file("status") > 0, "{label}");
            }
        }
        instance
            .commit_accepted(&(1..=8).map(|id| kept(id, 1)).collect::<Vec<_>>())
            .unwrap();
        for request in 1..=8 {
            instance.retire_request(request).unwrap();
        }
        instance.destroy().unwrap();
        assert_eq!(host.acquired.get(), host.released.get());
        std::fs::remove_dir_all(root).unwrap();
    }
}

/// The GPU test's file protocol, walked end to end on the fake host: each
/// arm runs that test's drafted steps (warm, queueing, consumer and
/// follow-up) through the shared protocol (`successor_protocol`), which
/// writes the controls, takes every step's dump and compares the arms as
/// the GPU test does. The consumer is a decode step at C8 and C9 (wide).
/// On this device every arm's bytes match; a missing dump file, an
/// unchecked status or a wrong outcome fails here before any GPU run.
#[cfg(feature = "state-capture")]
#[test]
fn the_gpu_comparison_protocol_walks_every_file_on_the_fake_host() {
    use super::successor_protocol as protocol;
    let root = PathBuf::from(std::env::var_os("CHOPINFER_DATA").expect("external test data"))
        .join("deepseek41-tests")
        .join(format!("dspark-successor-protocol-{}", std::process::id()));
    if root.exists() {
        std::fs::remove_dir_all(&root).unwrap();
    }
    let state = root.join("state");
    std::fs::create_dir_all(&state).unwrap();
    protocol::start(&state).unwrap();
    for (label, items) in [("c8-decode", 8), ("c9-wide", 9)] {
        let case = root.join("output").join(label);
        for inline in [false, true] {
            let (host, mut instance) =
                super::wide_successor::prompted_in(items, items + 1, 64, &[]);
            crate::instance::set_dspark_state_capture_root_for_test(
                instance.handle(),
                Some(state.clone()),
            );
            protocol::select_arm(&state, inline).unwrap();
            let arm = protocol::arm_directory(&case, inline);
            let decoders = |rows: u32| {
                (1..=u64::from(items))
                    .map(|id| kept(id, rows))
                    .collect::<Vec<_>>()
            };
            execute(&mut instance, &host, &batch(items, 3, 3));
            protocol::take_dump(&state, &arm.join("warm")).unwrap();
            instance.commit_accepted(&decoders(1)).unwrap();
            let accepted: Vec<_> = (1..=u64::from(items)).map(|id| (id, 2, 7)).collect();
            protocol::write_feedback(&state, false, &accepted).unwrap();
            execute(&mut instance, &host, &batch(items, 4, 3));
            protocol::take_dump(&state, &arm.join("queue")).unwrap();
            assert_eq!(protocol::take_feedback(&state).unwrap(), accepted);
            instance.commit_accepted(&decoders(3)).unwrap();
            execute(&mut instance, &host, &batch(items, 7, 3));
            let record = protocol::take_dump(&state, &arm.join("consume")).unwrap();
            assert_eq!(record["computed_positions"], if items < 9 { 3 } else { 5 });
            instance.commit_accepted(&decoders(1)).unwrap();
            execute(&mut instance, &host, &batch(items, 8, 3));
            protocol::take_dump(&state, &arm.join("follow")).unwrap();
            instance.commit_accepted(&decoders(1)).unwrap();
            for request in 1..=u64::from(items) {
                instance.retire_request(request).unwrap();
            }
            instance.destroy().unwrap();
            assert_eq!(host.acquired.get(), host.released.get());
        }
        protocol::compare_dumps(&case).unwrap_or_else(|error| panic!("{label}: {error}"));
    }
    std::fs::remove_dir_all(root).unwrap();
}
