//! Complete prefill journals for the vector gate binding audit.
use super::*;

#[test]
fn prefill_gate_journals_cover_ar_and_dspark_instances() {
    // BF16 KV serves AR only; the native-package tests cover its DSpark refusal.
    for (kv, drafter) in [
        (KV_LAYOUTS[0], false),
        (KV_LAYOUTS[1], false),
        (KV_LAYOUTS[0], true),
    ] {
        for rows in [4096, 8192] {
            let shape = Shape::new(&FLASH, Ranks::TP4).unwrap();
            let host = FakeHost::new(shape, kv);
            host.drafter.set(drafter);
            if drafter {
                host.speculation.set(Some((5, 32)));
            }
            let record = crate::config::tests::record(&shape, kv, 1, false);
            let services =
                HostServicesTableV1::new(Rc::new(dspark::MappedHost::new(Rc::clone(&host))));
            let mut instance = api().create(services, &record).unwrap_or_else(|e| {
                panic!(
                    "prefill create {kv:?}, drafter={drafter}: {e:?}, {:?}",
                    host.logs.borrow()
                )
            });
            instance
                .execute(&prompt(7, 0, 0, rows, true), &mut Views::default())
                .unwrap();
            instance.commit().unwrap();
            instance.retire_request(7).unwrap();
            instance.destroy().unwrap();
            assert_eq!(host.acquired.get(), host.released.get());
            let lines = host.device.journal.borrow();
            assert_launches_fit_configured_shared(&lines, &shape.topology());
            if let Some(root) = std::env::var_os("CHOPINFER_DEEPSEEK41_PREFILL_GATE_JOURNAL_DIR") {
                let root = std::path::PathBuf::from(root);
                std::fs::create_dir_all(&root).unwrap();
                let layout = if kv == KV_LAYOUTS[0] {
                    "native"
                } else {
                    "bf16"
                };
                let mode = if drafter { "dspark" } else { "ar" };
                std::fs::write(
                    root.join(format!("tp4-{layout}-{mode}-{rows}.journal")),
                    lines.join("\n"),
                )
                .unwrap();
            }
        }
    }
}
