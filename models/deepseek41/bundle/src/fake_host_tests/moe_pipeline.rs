//! The measured gate replacement owns a queue and a disjoint decoded operand view.
use super::*;
fn spans(line: &str) -> Vec<(u64, u64, u64)> {
    line.split("DeviceBufferSpanV1 { buffer: BundleDeviceBufferHandleV1 { raw: ")
        .skip(1)
        .map(|part| {
            let (buffer, rest) = part.split_once(" }, offset: ").unwrap();
            let (offset, rest) = rest.split_once(", length: ").unwrap();
            let length = rest.split_once(" }").unwrap().0;
            (
                buffer.parse().unwrap(),
                offset.parse().unwrap(),
                length.parse().unwrap(),
            )
        })
        .collect()
}
#[test]
fn prefill_gate_pipeline_preserves_down_and_bounds_decoded_storage() {
    let serving_rows = u64::from(model_deepseek41_contract::serving::PREFILL_TOKENS);
    assert_eq!(
        serving_rows, 8192,
        "qualify a new default chunk before routing it"
    );
    for kv in KV_LAYOUTS {
        for rows in [2048, 4096, serving_rows - 1, serving_rows] {
            let shape = Shape::new(&FLASH, Ranks::TP4).unwrap();
            let host = FakeHost::new(shape, kv);
            let record = crate::config::tests::record(&shape, kv, 1, false);
            let mut instance = api()
                .create(HostServicesTableV1::new(Rc::clone(&host)), &record)
                .unwrap();
            instance
                .execute(&prompt(7, 0, 0, rows, true), &mut Views::default())
                .unwrap();
            let journal = host.device.journal.borrow();
            assert_launches_fit_configured_shared(&journal, &shape.topology());
            let candidate = journal
                .iter()
                .filter(|l| l.starts_with("launch ") && l.contains("chop_exact_prefill_"))
                .collect::<Vec<_>>();
            assert_eq!(candidate.len(), if rows == 8192 { 4 * 40 * 6 } else { 0 });
            let down = journal
                .iter()
                .filter(|l| l.starts_with("launch ") && l.contains("before_down_e384k6_"))
                .collect::<Vec<_>>();
            assert_eq!(
                down.len(),
                4 * 40 * if rows == 2048 || rows == 4096 { 4 } else { 1 }
            );
            for group in candidate.chunks_exact(6) {
                assert!(
                    group
                        .iter()
                        .all(|l| l.contains("e384_before_gateup_decoded128_")
                            && l.contains("_vector_v1"))
                );
                let pack = spans(group[0]);
                let predecode = spans(group[1]);
                assert_eq!(pack.len(), 5);
                assert_eq!(predecode.len(), 4);
                let queue = pack[4];
                let decoded = predecode[3];
                assert_eq!(queue.0, decoded.0);
                assert_eq!(queue.2, 83_923_200);
                assert_eq!(decoded, (queue.0, queue.1 + 37_120, 83_886_080));
                for line in &group[2..] {
                    let args = spans(line);
                    assert_eq!(args.len(), 10);
                    assert_eq!(args[0], decoded);
                    assert_eq!(args[7], queue);
                    assert!(line.contains("NullDevicePointer"));
                    for operand in [&args[1..7], &args[8..10]].into_iter().flatten() {
                        assert!(
                            operand.0 != queue.0
                                || operand.1 + operand.2 <= queue.1
                                || operand.1 >= queue.1 + queue.2
                        );
                    }
                }
            }
            drop(journal);
            instance.commit().unwrap();
            instance.retire_request(7).unwrap();
            instance.destroy().unwrap();
            assert_eq!(host.acquired.get(), host.released.get());
        }
    }
}
