use super::*;

fn download_lengths(host: &Host, from: usize) -> Vec<u64> {
    host.journal.borrow()[from..]
        .iter()
        .filter(|line| line.starts_with("download "))
        .map(|line| {
            line.split_once("length: ")
                .unwrap()
                .1
                .split_whitespace()
                .next()
                .unwrap()
                .parse()
                .unwrap()
        })
        .collect()
}

#[test]
fn undrafted_status_accepts_equal_capacity_and_refuses_one_extra_word() {
    let (host, mut instance) = create(1, true);
    prefill(&host, &mut instance, 1);
    let from = host.journal.borrow().len();
    instance
        .execute(&decode(&[(1, 0, 3, 100)]), &mut sink(&host, 1, 800))
        .unwrap();
    let needed = download_lengths(&host, from).into_iter().sum::<u64>();
    assert!(needed > 4 && needed < 65_536);
    instance.commit().unwrap();
    for (position, capacity, downloads) in [(4, needed, 1), (5, needed - 4, 4)] {
        crate::instance::dspark_boundary_for_test(instance.handle(), None, Some(capacity));
        let from = host.journal.borrow().len();
        instance
            .execute(
                &decode(&[(1, 0, position, 100)]),
                &mut sink(&host, 1, 800 + position),
            )
            .unwrap();
        let lengths = download_lengths(&host, from);
        assert_eq!(lengths.iter().sum::<u64>(), needed);
        assert_eq!(lengths.len(), downloads);
        instance.commit().unwrap();
    }
    instance.retire_request(1).unwrap();
    instance.destroy().unwrap();
    assert_eq!(host.acquired.get(), host.released.get());
}

#[test]
fn undrafted_status_falls_back_only_when_saved_bytes_exceed_packet_capacity() {
    for count in [16_u32, 32] {
        let (host, mut instance) = create(count, true);
        let lengths = [193_u64, 192, 255, 256, 133, 134, 267, 268];
        let mut positions = Vec::new();
        for slot in 0..count {
            let length = lengths[slot as usize % lengths.len()];
            instance
                .execute(
                    &prompt(u64::from(slot) + 1, u64::from(slot), 0, length, true),
                    &mut Views::default(),
                )
                .unwrap();
            instance.commit().unwrap();
            positions.push(length);
        }
        for step in 0..2 {
            let request = decode(
                &(0..count)
                    .map(|slot| {
                        (
                            u64::from(slot) + 1,
                            u64::from(slot),
                            positions[slot as usize],
                            100,
                        )
                    })
                    .collect::<Vec<_>>(),
            );
            let mut result = sink(&host, count, 900 + step);
            let from = host.journal.borrow().len();
            instance.execute(&request, &mut result).unwrap();
            let boundary = crate::instance::dspark_boundary_for_test(instance.handle(), None, None);
            let lengths = download_lengths(&host, from);
            let saved_bytes = lengths.iter().sum::<u64>();
            assert_eq!(boundary.packet.length, 65_536);
            let overflow = saved_bytes > boundary.packet.length;
            assert_eq!(overflow, count == 32 && step == 0);
            let record = host
                .logs
                .borrow()
                .iter()
                .rev()
                .find_map(|line| {
                    serde_json::from_str::<serde_json::Value>(line)
                        .ok()
                        .filter(|value| value["schema"] == "chopinfer.deepseek41.step-shape.v1")
                })
                .unwrap();
            assert_eq!(
                record["bulk_passes"],
                if step == 0 { count * 3 / 2 } else { 0 }
            );
            assert_eq!(record["undrafted_status_packets"], u32::from(!overflow));
            assert_eq!(record["undrafted_ready"], 1);
            assert!(!result.ready.is_empty());
            assert!(result.legacy.is_empty());
            assert_eq!(lengths.len(), if overflow { 4 } else { 1 });
            instance.commit().unwrap();
            for position in &mut positions {
                *position += 1;
            }
        }
        for request in 1..=u64::from(count) {
            instance.retire_request(request).unwrap();
        }
        instance.destroy().unwrap();
        assert_eq!(host.acquired.get(), host.released.get());
    }
}
