//! A single eager item can use the shared arena at larger serving capacity.
use super::*;

const DATA_BYTES: u64 = 75_632_640;
const TABLE_BYTES: u64 = 65_536;

fn scalar_u32s(line: &str) -> Vec<u32> {
    line.split("ScalarU32(")
        .skip(1)
        .map(|part| part.split(')').next().unwrap().parse().unwrap())
        .collect()
}

/// Check both data and table ownership, including uploads and peer operations.
pub(super) fn assert_arena_streams(
    host: &FakeHost,
    caches: &[DeviceBufferSpanV1],
    lines: &[String],
    enabled: bool,
) {
    let allocations = host.device.buffer_order.borrow();
    let ranks = host.device.allocation_ranks.borrow();
    let mains = host.device.rank_streams.borrow();
    let mut owners = std::collections::BTreeMap::new();
    for cache in caches {
        let data = cache.buffer.slot();
        let at = allocations.iter().position(|&id| id == data).unwrap();
        let table = allocations[at + 1];
        let (rank, bytes) = ranks[&data];
        assert_eq!(bytes, DATA_BYTES);
        assert_eq!(ranks[&table], (rank, TABLE_BYTES));
        owners.insert(cache.buffer.raw, (rank, false));
        owners.insert(
            BundleDeviceBufferHandleV1::from_parts(1, table).raw,
            (rank, true),
        );
    }
    for line in lines {
        for span in super::shared_overlap::journal_spans(line) {
            if let Some(&(rank, table)) = owners.get(&span.0) {
                assert!(enabled, "disabled arena used: {line}");
                let stream: u32 = line.split_whitespace().nth(1).unwrap().parse().unwrap();
                assert_eq!(
                    stream, mains[&rank],
                    "arena leaves its rank's main stream: {line}"
                );
                assert!(
                    (line.starts_with("upload ") && table)
                        || (line.starts_with("launch ")
                            && (line.contains("chop_latent_prefill_")
                                || line.contains("_decoded_"))),
                    "unexpected arena operation: {line}"
                );
            }
        }
    }
}

#[test]
fn capacity32_decoded_prefill_reuses_one_arena_for_nonzero_slots() {
    let shape = Shape::new(&FLASH, Ranks::TP4).unwrap();
    let host = FakeHost::new(shape, KV_LAYOUTS[0]);
    host.kv_pages.set(1024);
    let record = crate::config::tests::record(&shape, KV_LAYOUTS[0], 32, false);
    let mut instance = api()
        .create(HostServicesTableV1::new(Rc::clone(&host)), &record)
        .unwrap();
    let caches = crate::instance::prefill_cache_for_test(instance.handle(), true);
    assert_eq!(caches.len(), 4, "one cache per rank at capacity 32");
    assert!(caches.iter().all(|span| span.length == DATA_BYTES));
    let before = host.device.journal.borrow().clone();
    let kv = before.iter().position(|line| line == "serving kv").unwrap();
    let cache_allocations = before
        .iter()
        .enumerate()
        .filter(|(_, line)| **line == format!("allocate {DATA_BYTES} 256"))
        .map(|(index, _)| index)
        .collect::<Vec<_>>();
    assert_eq!(cache_allocations.len(), 4);
    for index in cache_allocations {
        assert!(index + 1 < kv);
        assert_eq!(before[index + 1], format!("allocate {TABLE_BYTES} 256"));
    }
    // Alternate requests in nonzero persistent state slots. Each phase's
    // item index remains zero, independently of its state slot.
    for offset in [0, 8192, 16384, 24576, 32768] {
        for (request, slot) in [(7, 17), (9, 31)] {
            let from = host.device.journal.borrow().len();
            step(
                &mut instance,
                &prompt(request, slot, offset, 8192, true),
                &mut Views::default(),
            );
            let journal = host.device.journal.borrow();
            let lines = &journal[from..];
            let attention = lines
                .iter()
                .filter(|line| {
                    line.starts_with("launch ")
                        && line.contains("chop_latent_sources_")
                        && !line.contains("_merge_")
                })
                .collect::<Vec<_>>();
            assert_eq!(attention.len(), 160);
            let decoded = attention
                .iter()
                .filter(|line| line.contains("_decoded_"))
                .count();
            assert_eq!(
                decoded,
                if offset == 0 {
                    8
                } else if offset < 32768 {
                    160
                } else {
                    0
                }
            );
            for line in attention {
                let scalars = scalar_u32s(line);
                if line.contains("_decoded_") {
                    assert_eq!(&scalars[1..3], &[1, 1], "window sequences/items");
                    if line.contains("_mixed_") {
                        assert_eq!(scalars[8], 1, "compressed sequences");
                    }
                } else {
                    assert_eq!(scalars[2], 32, "native fallback keeps capacity");
                }
            }
            assert_arena_streams(&host, &caches, lines, true);
        }
    }
    assert_eq!(
        crate::instance::prefill_cache_for_test(instance.handle(), true),
        caches
    );
    let from = host.device.journal.borrow().len();
    instance
        .execute(
            &decode(&[(7, 17, 40960, 1), (9, 31, 40960, 2)]),
            &mut Views::default(),
        )
        .unwrap();
    instance.cancel().unwrap();
    assert!(
        host.device.journal.borrow()[from..]
            .iter()
            .all(|line| { !line.contains("_decoded_") && !line.contains("chop_latent_prefill_") })
    );
    instance.retire_request(7).unwrap();
    instance.retire_request(9).unwrap();
    // Reuse a retired state slot and the same arena for a fresh request.
    step(
        &mut instance,
        &prompt(11, 31, 0, 8192, true),
        &mut Views::default(),
    );
    step(
        &mut instance,
        &prompt(11, 31, 8192, 8192, true),
        &mut Views::default(),
    );
    instance.retire_request(11).unwrap();
    instance.destroy().unwrap();
    assert_eq!(host.acquired.get(), host.released.get());
}

#[test]
fn capacity32_multiple_live_prefill_items_keep_native_attention() {
    use runtime_bundle_sdk::{BatchAppendV1, BatchItemV1, BatchRequestV1, BatchWireVersionV1};
    let shape = Shape::new(&FLASH, Ranks::TP4).unwrap();
    let host = FakeHost::new(shape, KV_LAYOUTS[0]);
    host.kv_pages.set(1024);
    let mut instance = create(&host, 32);
    assert_eq!(
        crate::instance::prefill_cache_for_test(instance.handle(), true).len(),
        4
    );
    let batch = BatchRequestV1::new(
        BatchWireVersionV1::V3,
        vec![
            BatchItemV1::new(7, false, 4096, true, vec![3; 4096]),
            BatchItemV1::new(9, false, 4096, true, vec![7; 4096]),
        ],
        vec![
            BatchAppendV1::new(0, 4096, 4096, (0..64).collect(), None),
            BatchAppendV1::new(0, 4096, 4096, (64..128).collect(), None),
        ],
    )
    .with_state_slots(vec![17, 31])
    .encode()
    .unwrap();
    let from = host.device.journal.borrow().len();
    step(&mut instance, &batch, &mut Views::default());
    let lines = host.device.journal.borrow();
    let attention = lines[from..]
        .iter()
        .filter(|line| {
            line.starts_with("launch ")
                && line.contains("chop_latent_sources_")
                && !line.contains("_merge_")
        })
        .collect::<Vec<_>>();
    assert_eq!(attention.len(), 160);
    assert!(
        attention
            .iter()
            .all(|line| !line.contains("_decoded_") && scalar_u32s(line)[2] == 32)
    );
    drop(lines);
    instance.retire_request(7).unwrap();
    instance.retire_request(9).unwrap();
    instance.destroy().unwrap();
    assert_eq!(host.acquired.get(), host.released.get());
}
