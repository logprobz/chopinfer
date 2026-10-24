//! Inspect exact retained-row bindings; fake kernels do not execute copies.
use super::*;

const SYMBOL: &str = "chop_deepseek41_dspark_retain_rows_v1";
const ORDER: [usize; 8] = [5, 0, 7, 2, 4, 1, 6, 3];

fn batch(items: &[(u64, u64, u64)], depth: u32) -> Vec<u8> {
    if depth == 0 {
        super::super::decode(
            &items
                .iter()
                .map(|&(request, slot, start)| (request, slot, start, 7))
                .collect::<Vec<_>>(),
        )
    } else {
        verify_all(items, depth)
    }
}

fn create_indexed(enabled: bool) -> (Rc<Host>, BundleInstanceV1) {
    let result = crate::dspark::retention::with_enabled(enabled, || {
        create_with_limits(Shape::new(&FLASH, Ranks::TP4).unwrap(), 16, 32)
    });
    crate::instance::dspark_step_trace_for_test(result.1.handle(), true);
    result
}

fn prepare_prompts(host: &Host, instance: &mut BundleInstanceV1, positions: &[u64]) {
    for (slot, &end) in positions.iter().enumerate() {
        execute(
            instance,
            host,
            &super::super::prompt(slot as u64 + 1, slot as u64, 0, end, true),
        );
        let from = host.journal.borrow().len();
        instance.commit().unwrap();
        assert!(
            !host.journal.borrow()[from..]
                .iter()
                .any(|line| line.contains(SYMBOL)),
            "prefill must keep the original retention path"
        );
    }
}

fn expected(positions: &[u64], depth: u32, kept: &[u32]) -> Vec<u8> {
    let mut words = Vec::new();
    let mut source = 0;
    for (&slot, &kept) in ORDER.iter().zip(kept) {
        let mut copied = 0;
        while copied < kept {
            let cell = (positions[slot] + u64::from(copied)) % 134;
            let rows = (kept - copied).min(134 - cell as u32);
            words.extend([source + copied, slot as u32 * 134 + cell as u32, rows]);
            copied += rows;
        }
        // Source advances over rejected provisional rows as well.
        source += depth + 1;
    }
    words.into_iter().flat_map(u32::to_le_bytes).collect()
}

#[test]
fn indexed_retention_preserves_wrapped_mappings_and_partial_commit_cursors() {
    let mut contexts = Vec::new();
    for enabled in [false, true] {
        let (host, mut instance) = create_indexed(enabled);
        let mut positions = [133, 134, 267, 268, 127, 128, 3, 193];
        prepare_prompts(&host, &mut instance, &positions);
        let metadata = crate::instance::retention_metadata_for_test(instance.handle());
        assert_eq!(metadata.len(), if enabled { 4 } else { 0 });
        let mut snapshots = Vec::new();
        for depth in [0, 1, 3, 1] {
            let items: Vec<_> = ORDER
                .iter()
                .map(|&slot| (slot as u64 + 1, slot as u64, positions[slot]))
                .collect();
            execute(&mut instance, &host, &batch(&items, depth));
            let kept: Vec<_> = (0..8)
                .map(|item| if item % 2 == 0 { depth + 1 } else { 1 })
                .collect();
            let wanted = expected(&positions, depth, &kept);
            let from = host.journal.borrow().len();
            if depth == 0 {
                instance.commit().unwrap();
            } else {
                instance
                    .commit_accepted(
                        &items
                            .iter()
                            .zip(&kept)
                            .map(|(item, &count)| super::kept(item.0, count))
                            .collect::<Vec<_>>(),
                    )
                    .unwrap();
            }
            let journal = host.journal.borrow();
            let lines = &journal[from..];
            let launches = lines
                .iter()
                .filter(|line| line.starts_with("launch ") && line.contains(SYMBOL))
                .count();
            assert_eq!(launches, if enabled { 4 } else { 0 });
            let copies = lines
                .iter()
                .filter(|line| line.starts_with("copy "))
                .count();
            assert_eq!(copies, if enabled { 0 } else { wanted.len() / 12 * 4 });
            drop(journal);
            let record: serde_json::Value = host
                .logs
                .borrow()
                .iter()
                .rev()
                .find_map(|line| {
                    serde_json::from_str::<serde_json::Value>(line)
                        .ok()
                        .filter(|value| value["schema"] == "chopinfer.deepseek41.commit-shape.v1")
                })
                .unwrap();
            assert_eq!(
                record["retention_kernel_launches"],
                if enabled { 4 } else { 0 }
            );
            assert_eq!(
                record["retention_copy_calls"],
                if enabled { 0 } else { wanted.len() / 12 * 4 }
            );
            assert_eq!(
                record["retention_metadata_bytes"],
                if enabled { wanted.len() * 4 } else { 0 }
            );
            for span in &metadata {
                assert_eq!(
                    host.uploads.borrow()[&(span.buffer.raw, span.offset)],
                    wanted
                );
            }
            for (&slot, &count) in ORDER.iter().zip(&kept) {
                positions[slot] += u64::from(count);
                let context = state(&instance, slot as u64 + 1);
                assert_eq!(context.end, positions[slot]);
                snapshots.push(context);
            }
        }
        contexts.push(snapshots);
        // Below the copy threshold, the original scalar path still runs.
        execute(&mut instance, &host, &verify(1, 0, positions[0], 0));
        let from = host.journal.borrow().len();
        instance.commit().unwrap();
        assert!(
            !host.journal.borrow()[from..]
                .iter()
                .any(|line| line.contains(SYMBOL))
        );
        for id in 1..=8 {
            instance.retire_request(id).unwrap();
        }
        instance.destroy().unwrap();
        assert_eq!(host.acquired.get(), host.released.get());
    }
    assert_eq!(contexts[0], contexts[1]);
}

#[test]
fn indexed_retention_submission_failure_does_not_publish_or_resume() {
    for failed_launch in [0, 1] {
        let (host, mut instance) = create_indexed(true);
        prepare_prompts(&host, &mut instance, &[3; 8]);
        let items: Vec<_> = (0..8).map(|slot| (slot + 1, slot, 3)).collect();
        execute(&mut instance, &host, &batch(&items, 0));
        host.device.fail_launch.set(Some(failed_launch));
        assert!(instance.commit().is_err());
        for id in 1..=8 {
            assert_eq!(state(&instance, id).end, 3);
        }
        let from = host.journal.borrow().len();
        assert!(
            instance
                .execute(&batch(&items, 0), &mut Sink::default())
                .is_err()
        );
        assert!(
            host.journal.borrow()[from..]
                .iter()
                .all(|line| !line.starts_with("launch ") && !line.starts_with("replay "))
        );
        assert!(instance.cancel().is_err());
        let _ = instance.destroy();
    }
}
