use super::*;
use runtime_bundle_sdk::{
    BatchAppendV1, BatchItemV1, BatchRequestV1, BatchWireVersionV1, DraftSamplingV1,
};
use serde_json::{Value, json};
use std::path::{Path, PathBuf};

fn read(root: &Path, boundary: &str, step: u64, name: &str) -> Value {
    serde_json::from_slice(
        &std::fs::read(
            root.join(format!("transition-{boundary}-{step}"))
                .join(name),
        )
        .unwrap(),
    )
    .unwrap()
}

fn run(root: &Path, enabled: bool) {
    std::fs::create_dir_all(root).unwrap();
    std::fs::write(root.join("enabled"), []).unwrap();
    std::fs::write(
        root.join("transitions.json"),
        serde_json::to_vec(&json!({
            "version":1, "before_draft":[4,5,6], "after_commit":[3,4,5,6], "requests":[1],
            "max_bytes": 1_u64 << 30,
        }))
        .unwrap(),
    )
    .unwrap();
    let (host, mut instance) = create_with_sampling(1, enabled, true);
    crate::instance::set_dspark_state_capture_root_for_test(instance.handle(), Some(root.into()));
    instance
        .execute(&prompt(1, 0, 0, 193, true), &mut Views::default())
        .unwrap();
    instance.commit().unwrap();
    for position in 193..195 {
        let mut result = sink(&host, 1, position);
        instance
            .execute(&decode(&[(1, 0, position, 17)]), &mut result)
            .unwrap();
        assert_eq!(result.ready.is_empty(), !enabled);
        instance.commit().unwrap();
    }
    let after_zero = read(root, "after-commit", 3, "request-1/state.json");
    assert_eq!(after_zero["end"], 195);
    assert!(
        after_zero["projected"]
            .as_array()
            .unwrap()
            .iter()
            .all(|end| *end == 193)
    );
    assert_eq!(after_zero["retained_start"], 65);
    for (step, depth, kept) in [(4, 1, 1), (5, 1, 2), (6, 3, 4)] {
        let start = crate::instance::dspark_context(instance.handle(), 1)
            .unwrap()
            .end;
        let rows = depth + 1;
        let request = BatchRequestV1::new(
            BatchWireVersionV1::V3,
            vec![BatchItemV1::verify(1, rows, true, vec![7])],
            vec![BatchAppendV1::new(
                start,
                u64::from(rows),
                start + u64::from(rows),
                (0..(start + u64::from(rows)).div_ceil(64)).collect(),
                None,
            )],
        )
        .with_state_slots(vec![0])
        .with_draft_sampling(vec![
            DraftSamplingV1::new(91, start - 192, 1.0, 1.0, 0.0, 0, 0).unwrap(),
        ])
        .encode()
        .unwrap();
        let mut result = bundle_runtime::fake_host::RecordingSink::<
            crate::fake_host_tests::conformance::Limits,
        >::default();
        instance
            .execute(&request, &mut result)
            .unwrap_or_else(|error| panic!("{error:?}, {:?}", host.logs.borrow()));
        let before = read(root, "before-draft", step, "request-1/state.json");
        assert_eq!(before["end"], start);
        assert!(
            before["projected"]
                .as_array()
                .unwrap()
                .iter()
                .all(|end| *end == start)
        );
        let host_before = read(root, "before-draft", step, "host.json");
        assert_eq!(host_before["sampling_inputs"][0]["seed"], 91);
        assert_eq!(host_before["sampling_inputs"][0]["generated"], start - 192);
        assert_eq!(
            host_before["sampling_inputs"][0]["temperature_bits"],
            1.0_f64.to_bits()
        );
        assert!(
            host_before["state_debug"]
                .as_str()
                .unwrap()
                .contains("pending: Some")
        );
        if step == 4 {
            let host_zero = read(root, "after-commit", 3, "host.json");
            assert_eq!(
                host_before["staging"]["histories"],
                host_zero["staging"]["histories"]
            );
        }
        instance
            .commit_accepted(&[BundleKeptRowsV1 {
                struct_size: std::mem::size_of::<BundleKeptRowsV1>() as u32,
                reserved: 0,
                request_id: 1,
                kept_rows: kept,
                reserved_tail: 0,
            }])
            .unwrap();
        let after = read(root, "after-commit", step, "request-1/state.json");
        assert_eq!(after["end"], start + u64::from(kept));
        assert!(
            after["projected"]
                .as_array()
                .unwrap()
                .iter()
                .all(|end| *end == start)
        );
        let host_after = read(root, "after-commit", step, "host.json");
        assert!(
            host_after["state_debug"]
                .as_str()
                .unwrap()
                .contains("pending: None")
        );
        assert_eq!(host_after["staging"]["pending"], json!([]));
        let features = after["pieces"]
            .as_array()
            .unwrap()
            .iter()
            .find(|piece| piece["buffer"] == "dspark.features")
            .unwrap();
        assert_eq!(features["bytes"], 134_u64 * 3 * 5120 * 2);
    }
    instance.retire_request(1).unwrap();
    assert!(crate::instance::dspark_state_capture_requests_for_test(instance.handle()).is_empty());
    instance.destroy().unwrap();
    assert_eq!(host.acquired.get(), host.released.get());
    assert_eq!(
        std::fs::read_dir(root).unwrap().count(),
        9,
        "only selected boundaries may capture"
    );
}

fn files(root: &Path) -> BTreeMap<PathBuf, Vec<u8>> {
    let mut output = BTreeMap::new();
    let mut pending = vec![root.to_owned()];
    while let Some(directory) = pending.pop() {
        for item in std::fs::read_dir(directory).unwrap() {
            let path = item.unwrap().path();
            if path.is_dir() {
                pending.push(path);
            } else {
                output.insert(
                    path.strip_prefix(root).unwrap().to_owned(),
                    std::fs::read(path).unwrap(),
                );
            }
        }
    }
    output
}

#[test]
fn passive_full_state_capture_keeps_native_readiness_and_committed_cursors() {
    let root = PathBuf::from(std::env::var_os("CHOPINFER_DATA").unwrap())
        .join("deepseek41-tests")
        .join(format!("passive-ready-{}", std::process::id()));
    run(&root.join("legacy"), false);
    run(&root.join("ready"), true);
    let expected = files(&root.join("legacy"));
    let actual = files(&root.join("ready"));
    assert_eq!(
        expected.keys().collect::<Vec<_>>(),
        actual.keys().collect::<Vec<_>>()
    );
    for (name, bytes) in &expected {
        assert!(
            bytes == &actual[name],
            "capture bytes differ: {}",
            name.display()
        );
    }
    assert!(
        expected
            .values()
            .map(|bytes| bytes.len() as u64)
            .sum::<u64>()
            < 1 << 30
    );
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn passive_capture_budget_failure_keeps_a_committed_instance_terminal() {
    let root = PathBuf::from(std::env::var_os("CHOPINFER_DATA").unwrap())
        .join("deepseek41-tests")
        .join(format!("passive-budget-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(root.join("enabled"), []).unwrap();
    std::fs::write(
        root.join("transitions.json"),
        serde_json::to_vec(&json!({
            "version":1,"before_draft":[],"after_commit":[2],"requests":[1],"max_bytes":1,
        }))
        .unwrap(),
    )
    .unwrap();
    let (host, mut instance) = create(1, true);
    crate::instance::set_dspark_state_capture_root_for_test(instance.handle(), Some(root.clone()));
    prefill(&host, &mut instance, 1);
    let mut result = sink(&host, 1, 9);
    instance
        .execute(&decode(&[(1, 0, 3, 17)]), &mut result)
        .unwrap();
    assert!(!result.ready.is_empty());
    assert!(instance.commit().is_err());
    assert!(!root.join("transition-after-commit-2").exists());
    let from = host.journal.borrow().len();
    assert!(
        instance
            .execute(&decode(&[(1, 0, 4, 17)]), &mut sink(&host, 1, 10))
            .is_err()
    );
    assert!(instance.cancel().is_err());
    assert!(
        host.journal.borrow()[from..]
            .iter()
            .all(|line| !line.starts_with("launch ") && !line.starts_with("replay "))
    );
    let _ = instance.destroy();
    std::fs::remove_dir_all(root).unwrap();
}
