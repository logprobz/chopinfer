use super::*;
use crate::dspark::device::slice;
use crate::instance::dspark_boundary_for_test;

#[test]
fn undrafted_status_switch_preserves_the_drafted_boundary() {
    crate::dspark::cycle::with_undrafted_status(true, || {
        c1_boundary_preserves_all_status_sources_and_confidence_words();
    });
}

fn status_bytes(rank: usize) -> u64 {
    let indexes = if rank == 0 {
        FLASH.shape.csa.index_sources.len() as u64
    } else {
        0
    };
    4 * (2 * (3 + 5) + 6 * (indexes + 2))
}

#[test]
fn c1_boundary_preserves_all_status_sources_and_confidence_words() {
    for ranks in [Ranks::TP1, Ranks::TP2, Ranks::TP4] {
        let (host, mut instance) = create(Shape::new(&FLASH, ranks).unwrap());
        execute(
            &mut instance,
            &host,
            &super::super::prompt(7, 0, 0, 3, true),
        );
        instance.commit().unwrap();
        let view = dspark_boundary_for_test(instance.handle(), None, None);
        assert!(view.enabled);
        let prefix: u64 = (0..ranks.count()).map(status_bytes).sum();
        let ids = [11_u32, 12, 13, 14, 15];
        let confidence = [0x8000_0000_u32, 1, 0x7fc0_1234, 0x7f80_0000, 0xff80_0000];
        let mut bytes = vec![0; prefix as usize];
        bytes.extend(ids.into_iter().chain(confidence).flat_map(u32::to_le_bytes));
        host.injected.replace(Some(bytes));
        let begin = host.journal.borrow().len();
        let sink = execute(&mut instance, &host, &verify(7, 0, 3, 5));
        assert!(host.injected.borrow().is_none());
        assert_eq!(sink.draft_tokens(), [(7, ids.to_vec())]);
        let memory = host.memory.borrow();
        let mut offset = 0;
        for (rank, &source) in view.status.iter().enumerate() {
            let per_rank = status_bytes(rank);
            assert_ne!(source.buffer, view.packet.buffer);
            assert_eq!(
                memory.read(slice(view.packet, offset, per_rank).unwrap()),
                memory.read(DeviceBufferSpanV1 {
                    length: per_rank,
                    ..source
                }),
            );
            offset += per_rank;
        }
        for (part, source) in view.diagnostics.into_iter().enumerate() {
            assert_ne!(source.buffer, view.packet.buffer);
            assert_eq!(
                memory.read(slice(view.packet, prefix + part as u64 * 20, 20).unwrap()),
                memory.read(slice(source, 0, 20).unwrap()),
            );
        }
        drop(memory);
        let journal = host.journal.borrow();
        let step = &journal[begin..];
        let copied = step
            .iter()
            .position(|s| s.starts_with("peer results ") || s.starts_with("download "))
            .unwrap();
        for (rank, &(stream, event)) in view.ready.iter().enumerate() {
            assert!(
                step.iter()
                    .position(|s| s == &format!("record {stream} {event}"))
                    .unwrap()
                    < copied
            );
            if rank != 0 {
                assert!(
                    step.iter()
                        .position(|s| s == &format!("wait {} {event}", view.ready[0].0))
                        .unwrap()
                        < copied
                );
            }
        }
        drop(journal);
        assert_eq!(
            host.journal.borrow()[begin..]
                .iter()
                .filter(|s| s.starts_with("download "))
                .count(),
            1
        );
        instance.commit_accepted(&[kept(7, 3)]).unwrap();
        let log = host.logs.borrow();
        let record: serde_json::Value = serde_json::from_str(
            log.iter()
                .rev()
                .find(|s| s.contains(crate::dspark::calibration::SCHEMA))
                .unwrap(),
        )
        .unwrap();
        assert_eq!(record["confidence_f32_bits"], serde_json::json!(confidence));
        assert_eq!(record["draft_ids"], serde_json::json!(ids));
        drop(log);
        assert_eq!(state(&instance, 7).end, 6);
        instance.retire_request(7).unwrap();
        instance.destroy().unwrap();
        assert_eq!(host.acquired.get(), host.released.get());
    }
}

#[test]
fn every_rank_status_failure_still_refuses_publication() {
    for rank in 0..4 {
        let (host, mut instance) = create(Shape::new(&FLASH, Ranks::TP4).unwrap());
        execute(
            &mut instance,
            &host,
            &super::super::prompt(7, 0, 0, 3, true),
        );
        instance.commit().unwrap();
        let length: u64 = (0..4).map(status_bytes).sum();
        let mut bytes = vec![0; length as usize + 40];
        let offset: u64 = (0..rank).map(status_bytes).sum();
        bytes[offset as usize] = 1;
        host.injected.replace(Some(bytes));
        let mut sink = Sink::default();
        assert!(instance.execute(&verify(7, 0, 3, 5), &mut sink).is_err());
        assert!(host.injected.borrow().is_none());
        assert!(sink.views.is_empty() && sink.draft_tokens().is_empty());
        assert!(
            host.logs
                .borrow()
                .iter()
                .any(|s| s.contains(&format!("rank {rank} is invalid")))
        );
        assert_eq!(state(&instance, 7).end, 3);
        assert!(instance.commit_accepted(&[kept(7, 6)]).is_err());
        instance.destroy().unwrap();
        assert_eq!(host.acquired.get(), host.released.get());
    }
}

#[test]
fn disabled_or_small_packet_keeps_the_legacy_boundary() {
    for (enabled, capacity) in [(Some(false), None), (None, Some(40))] {
        let (host, mut instance) = create(Shape::new(&FLASH, Ranks::TP4).unwrap());
        execute(
            &mut instance,
            &host,
            &super::super::prompt(7, 0, 0, 3, true),
        );
        instance.commit().unwrap();
        dspark_boundary_for_test(instance.handle(), enabled, capacity);
        let begin = host.journal.borrow().len();
        let sink = execute(&mut instance, &host, &verify(7, 0, 3, 5));
        assert_eq!(sink.draft_tokens(), [(7, vec![0; 5])]);
        assert_eq!(
            host.journal.borrow()[begin..]
                .iter()
                .filter(|s| s.starts_with("download "))
                .count(),
            5
        );
        instance.cancel().unwrap();
        assert!(crate::instance::dspark_context(instance.handle(), 7).is_none());
        instance.retire_request(7).unwrap();
        execute(
            &mut instance,
            &host,
            &super::super::prompt(8, 0, 0, 3, true),
        );
        instance.commit().unwrap();
        execute(&mut instance, &host, &verify(8, 0, 3, 5));
        instance.commit_accepted(&[kept(8, 1)]).unwrap();
        assert_eq!(state(&instance, 8).end, 4);
        instance.retire_request(8).unwrap();
        instance.destroy().unwrap();
        assert_eq!(host.acquired.get(), host.released.get());
    }
}
