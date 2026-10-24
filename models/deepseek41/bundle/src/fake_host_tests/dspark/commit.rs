//! Commit row plans and failure boundaries, without numerical GPU claims.
use super::*;

fn packed(starts: &[u64], lengths: &[u32]) -> Vec<u8> {
    let slots = [15, 3, 8, 1, 10, 2];
    BatchRequestV1::new(
        BatchWireVersionV1::V3,
        lengths
            .iter()
            .enumerate()
            .map(|(i, &rows)| {
                BatchItemV1::new(i as u64 + 1, false, rows, true, vec![7; rows as usize])
            })
            .collect(),
        lengths
            .iter()
            .zip(starts)
            .enumerate()
            .map(|(i, (&rows, &start))| {
                let end = start + u64::from(rows);
                BatchAppendV1::new(
                    start,
                    u64::from(rows),
                    end,
                    (0..end.div_ceil(64)).map(|p| i as u64 * 32 + p).collect(),
                    None,
                )
            })
            .collect(),
    )
    .with_state_slots(slots[..lengths.len()].to_vec())
    .encode()
    .unwrap()
}

fn commits(host: &Host) -> Vec<serde_json::Value> {
    host.logs
        .borrow()
        .iter()
        .filter_map(|line| {
            let record: serde_json::Value = serde_json::from_str(line).ok()?;
            (record["schema"] == "chopinfer.deepseek41.commit-shape.v1").then_some(record)
        })
        .collect()
}

#[test]
fn batched_commit_preserves_continued_and_mixed_prompt_cursors() {
    let mut evidence = Vec::new();
    for enabled in [false, true] {
        let (host, mut instance) = crate::dspark::device::with_batch_commit(enabled, || {
            create_with_limits(Shape::new(&FLASH, Ranks::TP4).unwrap(), 16, 32)
        });
        crate::instance::dspark_step_trace_for_test(instance.handle(), true);
        let mut starts = [0; 6];
        let mut contexts = Vec::new();
        for lengths in [[192, 193, 255, 256, 7, 789], [191, 195, 257, 260, 1, 789]] {
            execute(&mut instance, &host, &packed(&starts, &lengths));
            instance.commit().unwrap();
            for (i, start) in starts.iter_mut().enumerate() {
                *start += u64::from(lengths[i]);
                let context = state(&instance, i as u64 + 1);
                assert_eq!(context.end, *start);
                contexts.push((
                    context.end,
                    context.retained_start,
                    context.projected().to_vec(),
                ));
            }
        }
        let records = commits(&host);
        assert_eq!(records.len(), 2);
        for record in &records {
            let passes = record["projection_passes"].as_u64().unwrap();
            if enabled {
                assert_eq!(passes, 3);
            } else {
                assert!(passes > 3);
            }
        }
        evidence.push((
            contexts,
            records
                .iter()
                .map(|r| r["projection_rows"].clone())
                .collect::<Vec<_>>(),
        ));
        for id in 1..=6 {
            instance.retire_request(id).unwrap();
        }
        instance.destroy().unwrap();
        assert_eq!(host.acquired.get(), host.released.get());
    }
    assert_eq!(
        evidence[0], evidence[1],
        "batching must keep every projected row and cursor"
    );
}

#[test]
fn batched_commit_configures_large_kv_projection_before_launch() {
    let (host, mut instance) = crate::dspark::device::with_batch_commit(true, || {
        create_with_limits(Shape::new(&FLASH, Ranks::TP4).unwrap(), 16, 32)
    });
    execute(
        &mut instance,
        &host,
        &packed(&[0; 4], &[192, 193, 255, 256]),
    );
    let begin = host.journal.borrow().len();
    instance.commit().unwrap();
    let journal = host.journal.borrow();
    super::super::assert_launches_fit_configured_shared(&journal, "batched drafter commit");
    let mut launches = 0;
    for line in &journal[begin..] {
        let words = line.split_whitespace().collect::<Vec<_>>();
        if words.first() != Some(&"launch")
            || !words[2].contains("_n512_k5120_")
            || !words[2].ends_with("_prefill_tma_v1_packed_v1")
        {
            continue;
        }
        let shared = 66_560;
        assert!(shared > bundle_runtime::acquire::DEFAULT_SHARED_BYTES);
        assert!(line.contains(&format!("shared_memory_bytes: {shared},")));
        launches += 1;
    }
    assert_eq!(launches, 4 * FLASH.dspark_stages());
    drop(journal);
    for request in 1..=4 {
        instance.retire_request(request).unwrap();
    }
    instance.destroy().unwrap();
    assert_eq!(host.acquired.get(), host.released.get());
}

#[test]
fn batched_commit_errors_cannot_publish_cursors_or_resume() {
    for failure in ["gather", "launch", "status"] {
        let (host, mut instance) = crate::dspark::device::with_batch_commit(true, || {
            create_with_limits(Shape::new(&FLASH, Ranks::TP1).unwrap(), 16, 32)
        });
        execute(&mut instance, &host, &packed(&[0], &[193]));
        match failure {
            "gather" => host.device.fail_copy.set(Some(0)),
            "launch" => host.device.fail_launch.set(Some(0)),
            "status" => {
                // All three stages inspect 193 rows and save both planes.
                let mut bytes = vec![0; 193 * 3 * 8];
                bytes[0] = 1;
                host.injected.replace(Some(bytes));
            }
            _ => unreachable!(),
        }
        assert!(instance.commit().is_err(), "{failure}");
        assert!(
            crate::instance::dspark_context(instance.handle(), 1).is_none(),
            "{failure} published a commit"
        );
        let before = host.journal.borrow().len();
        assert!(
            instance
                .execute(&packed(&[0], &[193]), &mut Sink::default())
                .is_err()
        );
        assert!(
            host.journal.borrow()[before..]
                .iter()
                .all(|line| !line.starts_with("launch ") && !line.starts_with("replay "))
        );
        // Device submission errors may leave the host's instance terminal.
        let _ = instance.destroy();
    }
}
