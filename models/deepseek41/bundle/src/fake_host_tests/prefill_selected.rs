//! Capture lifecycle and unchanged eager launches on the CPU fake host.
use super::*;

#[test]
fn selected_lists_follow_commit_and_cancel_without_changing_launches() {
    let shape = Shape::new(&FLASH, Ranks::TP4).unwrap();
    if std::env::var_os(TAPS_CHILD).is_some() {
        let host = FakeHost::new(shape, KV_LAYOUTS[0]);
        let mut instance = create(&host, 1);
        let root = std::path::PathBuf::from(std::env::var_os("CHOPINFER_TAPS_DIR").unwrap());
        for request in [7, 9] {
            let start = host.device.journal.borrow().len();
            instance
                .execute(&prompt(request, 0, 0, 8192, true), &mut Views::default())
                .unwrap_or_else(|error| panic!("{error:?} {:?}", host.logs.borrow()));
            if request == 7 {
                std::fs::write(
                    root.join("launches.json"),
                    serde_json::to_vec(&launch_lines(&host, start)).unwrap(),
                )
                .unwrap();
                instance.commit().unwrap();
            } else {
                instance.cancel().unwrap();
            }
            instance.retire_request(request).unwrap();
        }
        instance.destroy().unwrap();
        assert_eq!(host.acquired.get(), host.released.get());
        return;
    }
    let host = FakeHost::new(shape, KV_LAYOUTS[0]);
    let mut instance = create(&host, 1);
    let start = host.device.journal.borrow().len();
    instance
        .execute(&prompt(7, 0, 0, 8192, true), &mut Views::default())
        .unwrap_or_else(|error| panic!("{error:?} {:?}", host.logs.borrow()));
    let untapped = launch_lines(&host, start);
    instance.commit().unwrap();
    instance.destroy().unwrap();
    let root =
        std::env::temp_dir().join(format!("deepseek-selected-capture-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    in_child(
        "fake_host_tests::prefill_selected::selected_lists_follow_commit_and_cancel_without_changing_launches",
        &root,
        &[
            ("CHOPINFER_TAPS_PREFILL_SELECTED", "1"),
            ("CHOPINFER_TAPS_SITES", "L02.x.hc.attn.pre"),
            ("CHOPINFER_TAPS_FROM", "0"),
        ],
    );
    let launches: BTreeMap<String, Vec<String>> =
        serde_json::from_slice(&std::fs::read(root.join("launches.json")).unwrap()).unwrap();
    assert_eq!(launches, untapped);
    let step = root.join("step0000");
    for layer in 2..40 {
        let dir = step.join(format!("selected-L{layer:02}"));
        let meta: serde_json::Value =
            serde_json::from_slice(&std::fs::read(dir.join("selected.json")).unwrap()).unwrap();
        assert_eq!(meta["offset"], 0);
        assert_eq!(meta["rows"], 8192);
        assert_eq!(meta["positions"][8191], 8191);
        assert_eq!(
            std::fs::metadata(dir.join("selected.i32")).unwrap().len(),
            8192 * 512 * 4
        );
    }
    assert_eq!(
        std::fs::metadata(step.join("window-keys.i32"))
            .unwrap()
            .len(),
        8192 * 128 * 4
    );
    let completion: serde_json::Value =
        serde_json::from_slice(&std::fs::read(step.join("completion.json")).unwrap()).unwrap();
    assert_eq!(completion["completion"], "full", "{completion}");
    assert_eq!(completion["expected"], 1);
    assert_eq!(completion["observed"], 1);
    assert_eq!(
        std::fs::metadata(step.join("L02.x.hc.attn.pre.r0.bin"))
            .unwrap()
            .len(),
        8192 * 4 * 4
    );
    assert!(!root.join("step0001").exists());
    std::fs::remove_dir_all(root).unwrap();
}
