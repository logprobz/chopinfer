//! Journal admission of the prefill-only router specialization.
use super::*;
#[test]
fn router_prefill_journal_preserves_decode_and_configures_shared_memory() {
    for drafter in [false, true] {
        let shape = Shape::new(&FLASH, Ranks::TP4).unwrap();
        let host = FakeHost::new(shape, KV_LAYOUTS[0]);
        host.drafter.set(drafter);
        if drafter {
            host.speculation.set(Some((5, 32)));
        }
        let record = crate::config::tests::record(&shape, KV_LAYOUTS[0], 1, false);
        let services = HostServicesTableV1::new(Rc::new(dspark::MappedHost::new(Rc::clone(&host))));
        let mut instance = api().create(services, &record).unwrap();
        for (id, rows) in [(7, 8192), (8, 320), (9, 17)] {
            let from = host.device.journal.borrow().len();
            instance
                .execute(&prompt(id, 0, 0, rows, true), &mut Views::default())
                .unwrap();
            instance.commit().unwrap();
            instance.retire_request(id).unwrap();
            let lines = host.device.journal.borrow();
            let zero = lines[from..]
                .iter()
                .filter(|l| l.starts_with("launch ") && l.contains("_zero_m32_v1"))
                .count();
            assert_eq!(
                zero > 0,
                rows >= 320,
                "rows {rows}: expected measured router route"
            );
            assert_launches_fit_configured_shared(&lines, &shape.topology());
        }
        instance.destroy().unwrap();
        assert_eq!(host.acquired.get(), host.released.get());
        if let Some(root) = std::env::var_os("CHOPINFER_ROUTER_JOURNAL_DIR") {
            let root = std::path::PathBuf::from(root);
            std::fs::create_dir_all(&root).unwrap();
            std::fs::write(
                root.join(if drafter {
                    "dspark.journal"
                } else {
                    "ar.journal"
                }),
                host.device.journal.borrow().join("\n"),
            )
            .unwrap();
        }
    }
}
