//! The candidate changes only transfer scheduling and explicit layer joins.
use super::*;
use crate::engram_rows::{EngramRows, Step};
use crate::instance::prefill_engram::{indices, with_enabled};
use model_deepseek41_contract::own_ops::engram::HashTables;
use runtime_bundle_sdk::prefill::{Boundary, encode};

fn create_mapped(host: &Rc<Host>, on: bool, skip: bool) -> BundleInstanceV1 {
    try_create_mapped(host, on, skip).unwrap_or_else(|e| panic!("{e:?}: {:?}", host.logs.borrow()))
}

fn try_create_mapped(
    host: &Rc<Host>,
    on: bool,
    skip: bool,
) -> Result<BundleInstanceV1, BundleAbiError> {
    create_backend(
        host,
        on,
        skip,
        dspark::host::MappedHost::with_delayed_completion(Rc::clone(host)),
    )
}

fn create_backend(
    host: &Rc<Host>,
    on: bool,
    skip: bool,
    backend: dspark::host::MappedHost,
) -> Result<BundleInstanceV1, BundleAbiError> {
    host.drafter.set(true);
    let mut record = crate::config::tests::record(
        &host.shape,
        host.kv,
        if host.speculation.get().is_some() {
            16
        } else {
            1
        },
        false,
    );
    let tag = record
        .windows(MODEL_OPTIONS_TAG_V1.len())
        .position(|x| x == MODEL_OPTIONS_TAG_V1)
        .unwrap();
    let start = tag + MODEL_OPTIONS_TAG_V1.len();
    let mut options: serde_json::Value = serde_json::from_slice(&record[start + 4..]).unwrap();
    options["prefill_chunk_tokens"] = serde_json::json!(8192);
    options["prefill_upper_skip"] = serde_json::json!(skip);
    record.truncate(start);
    let options = serde_json::to_vec(&options).unwrap();
    record.extend_from_slice(&(options.len() as u32).to_le_bytes());
    record.extend_from_slice(&options);
    with_enabled(on, || {
        api()
            .with_prefill_api(crate::api::BUNDLE_DEEPSEEK41_PREFILL_API_V1)
            .unwrap()
            .create(HostServicesTableV1::new(Rc::new(backend)), &record)
    })
}

#[test]
fn dspark_prefill_gather_failures_are_terminal_before_publication() {
    for (failure, after) in [("enqueue", 1), ("wait", 1), ("complete", 0)] {
        let host = FakeHost::new(Shape::new(&FLASH, Ranks::TP4).unwrap(), KV_LAYOUTS[0]);
        host.speculation.set(Some((5, 32)));
        let backend =
            dspark::host::MappedHost::with_gather_failure(Rc::clone(&host), failure, after);
        let mut instance = create_backend(&host, true, true, backend).unwrap();
        let request = encode(
            &prompt(7, 0, 0, 193, true),
            &[Boundary {
                request_id: 7,
                prompt_tokens: 193,
                chunk_end: 193,
                chunk_tokens: 8192,
            }],
        )
        .unwrap();
        let mut sink = Views::default();
        assert!(instance.execute_prefill(&request, &mut sink).is_err());
        assert_eq!(sink.rows(), 0, "{failure} published logits");
        assert!(crate::instance::dspark_context(instance.handle(), 7).is_none());
        assert!(
            host.journal
                .borrow()
                .iter()
                .any(|line| line == &format!("async mapped failure {failure}"))
        );
        let _ = instance.cancel();
        let begin = host.journal.borrow().len();
        assert!(
            instance
                .execute_prefill(&request, &mut Views::default())
                .is_err()
        );
        assert!(
            host.journal.borrow()[begin..]
                .iter()
                .all(|line| !line.starts_with("launch ") && !line.starts_with("async mapped ids "))
        );
        match instance.destroy() {
            Ok(()) => assert_eq!(host.acquired.get(), host.released.get()),
            Err(_) => assert!(instance.is_poisoned()),
        }
    }
}

fn launches(host: &Host) -> BTreeMap<String, Vec<String>> {
    let mut result = BTreeMap::<String, Vec<String>>::new();
    for line in host
        .journal
        .borrow()
        .iter()
        .filter(|s| s.starts_with("launch "))
    {
        let stream = line.split_whitespace().nth(1).unwrap().to_owned();
        result.entry(stream).or_default().push(line.clone());
    }
    result
}

#[test]
fn eager_and_upper_skip_join_each_layer_without_changing_launches() {
    for (ranks, kv, rows, skip) in [
        (Ranks::TP1, KV_LAYOUTS[0], 64, false),
        (Ranks::TP2, KV_LAYOUTS[1], 257, true),
        (Ranks::TP4, KV_LAYOUTS[0], 8192, true),
        (Ranks::TP1, KV_LAYOUTS[1], 8191, true),
    ] {
        let mut reference = None;
        for on in [false, true] {
            let host = FakeHost::new(Shape::new(&FLASH, ranks).unwrap(), kv);
            let mut instance = create_mapped(&host, on, skip);
            let spans = indices(instance.handle());
            assert_eq!(spans.len(), if on { 2 } else { 0 });
            let begin = host.journal.borrow().len();
            let request = encode(
                &prompt(7, 0, 0, rows, true),
                &[Boundary {
                    request_id: 7,
                    prompt_tokens: rows,
                    chunk_end: rows,
                    chunk_tokens: 8192,
                }],
            )
            .unwrap();
            instance
                .execute_prefill(&request, &mut Views::default())
                .unwrap_or_else(|e| panic!("{e:?}: {:?}", host.logs.borrow()));
            if on {
                let logs = host.logs.borrow();
                let metric: serde_json::Value = serde_json::from_str(
                    logs.iter()
                        .find(|s| s.contains("\"phase\":\"prefill-engram\""))
                        .expect("candidate actually executed"),
                )
                .unwrap();
                assert_eq!(metric["rows"], rows);
                assert_eq!(metric["start"], 0);
                assert_eq!(metric["waits"], ranks.get() * 2);
                assert_eq!(metric["layers"][0]["layer"], 1);
                assert_eq!(metric["layers"][1]["layer"], 14);
                let journal = host.journal.borrow();
                let step = &journal[begin..];
                let first_launch = step.iter().position(|s| s.starts_with("launch ")).unwrap();
                assert_eq!(
                    step[..first_launch]
                        .iter()
                        .filter(|s| s.starts_with("async mapped ids "))
                        .count(),
                    2
                );
                let first = step
                    .iter()
                    .position(|s| s.contains("layers.1.engram.embed.weight"))
                    .unwrap();
                let second = step
                    .iter()
                    .position(|s| s.contains("layers.14.engram.embed.weight"))
                    .unwrap();
                assert!(step[..first].iter().any(|s| s.starts_with("launch ")));
                assert!(
                    step[first..second]
                        .iter()
                        .filter(|s| s.starts_with("launch "))
                        .count()
                        > 100
                );
                assert_eq!(
                    step.iter()
                        .filter(|s| s.starts_with("async mapped complete"))
                        .count(),
                    2
                );
                assert_eq!(
                    step.iter()
                        .filter(|s| s.starts_with("async mapped wait "))
                        .count(),
                    2 * ranks.count()
                );
                let hash = HashTables::of(&FLASH).unwrap();
                let columns = hash.columns();
                let oracle = EngramRows::new(hash);
                let tokens: Vec<_> = (0..rows as u32).collect();
                let hashed = oracle
                    .hash(&[Step {
                        request: 7,
                        tokens: &tokens,
                        text: None,
                    }])
                    .unwrap();
                for (layer, (_, span)) in spans.iter().enumerate() {
                    let wanted: Vec<_> = hashed
                        .ids()
                        .chunks_exact(columns * 2)
                        .flat_map(|row| &row[layer * columns..(layer + 1) * columns])
                        .flat_map(|id| id.to_le_bytes())
                        .collect();
                    assert_eq!(
                        host.uploads.borrow()[&(span.buffer.raw, span.offset)],
                        wanted
                    );
                }
            }
            instance.commit().unwrap();
            let before_decode = host.journal.borrow().len();
            instance
                .execute(&decode(&[(7, 0, rows, 7)]), &mut Views::default())
                .unwrap();
            instance.commit().unwrap();
            assert!(
                !host.journal.borrow()[before_decode..]
                    .iter()
                    .any(|s| s.starts_with("async mapped ids "))
            );
            let observed = launches(&host);
            if let Some(reference) = &reference {
                assert_eq!(&observed, reference, "ranks={} rows={rows}", ranks.get());
            } else {
                reference = Some(observed);
            }
            instance.retire_request(7).unwrap();
            instance.destroy().unwrap();
            assert_eq!(host.acquired.get(), host.released.get());
        }
    }
}

#[test]
fn layer_handles_reuse_after_partial_prefill_and_cancel() {
    let host = FakeHost::new(Shape::new(&FLASH, Ranks::TP4).unwrap(), KV_LAYOUTS[0]);
    let mut instance = create_mapped(&host, true, true);
    let initial = indices(instance.handle());
    for (start, rows) in [(0, 8192), (8192, 17)] {
        let request = encode(
            &prompt(7, 0, start, rows, start + rows == 8209),
            &[Boundary {
                request_id: 7,
                prompt_tokens: 8209,
                chunk_end: start + rows,
                chunk_tokens: 8192,
            }],
        )
        .unwrap();
        instance
            .execute_prefill(&request, &mut Views::default())
            .unwrap();
        assert_eq!(indices(instance.handle()), initial);
        if start == 0 {
            instance.commit().unwrap();
        } else {
            instance.cancel().unwrap();
        }
    }
    let request = encode(
        &prompt(8, 0, 0, 17, true),
        &[Boundary {
            request_id: 8,
            prompt_tokens: 17,
            chunk_end: 17,
            chunk_tokens: 8192,
        }],
    )
    .unwrap();
    instance
        .execute_prefill(&request, &mut Views::default())
        .unwrap();
    instance.commit().unwrap();
    assert_eq!(
        host.logs
            .borrow()
            .iter()
            .filter(|s| s.contains("\"phase\":\"prefill-engram\""))
            .count(),
        3
    );
    instance.retire_request(8).unwrap();
    instance.destroy().unwrap();
    assert_eq!(host.acquired.get(), host.released.get());
}

#[test]
fn private_id_allocation_failures_release_prior_resources() {
    let shape = Shape::new(&FLASH, Ranks::TP4).unwrap();
    let host = FakeHost::new(shape, KV_LAYOUTS[0]);
    let mut instance = create_mapped(&host, true, true);
    let count = host
        .journal
        .borrow()
        .iter()
        .filter(|s| s.starts_with("allocate "))
        .count();
    instance.destroy().unwrap();
    assert_eq!(host.acquired.get(), host.released.get());
    for at in [count - 2, count - 1] {
        let host = FakeHost::new(shape, KV_LAYOUTS[0]);
        host.device.fail_allocate.set(Some(at as u32));
        assert!(try_create_mapped(&host, true, true).is_err());
        assert_eq!(host.device.fail_allocate.get(), None);
        // The fake increments acquisition before injecting allocation failure;
        // no handle is returned for that one counter increment.
        assert_eq!(host.device.allocation_failures.get(), 1);
        assert_eq!(host.acquired.get(), host.released.get() + 1);
        assert_eq!(host.device.buffer_order.borrow().len(), at);
        for id in host.device.buffer_order.borrow().iter() {
            assert!(!host.device.extents.borrow().contains_key(id));
        }
    }
}

#[test]
fn dspark_prefill_joins_after_submission_and_finishes_after_status() {
    use runtime_bundle_sdk::{BatchAppendV1, BatchItemV1, BatchRequestV1, BatchWireVersionV1};
    for skip in [false, true] {
        let mut reference = None;
        for on in [false, true] {
            let host = FakeHost::new(Shape::new(&FLASH, Ranks::TP4).unwrap(), KV_LAYOUTS[0]);
            host.speculation.set(Some((5, 32)));
            let mut instance = create_mapped(&host, on, skip);
            let spans = indices(instance.handle());
            assert_eq!(spans.len(), if on { 2 } else { 0 });
            let mut oracle = EngramRows::new(HashTables::of(&FLASH).unwrap());
            let mut starts = [0_u64; 4];
            let slots = [3, 7, 0, 15];
            let mut contexts = Vec::new();
            for (lengths, cancel) in [
                ([192, 193, 255, 256], false),
                ([64, 65, 127, 128], false),
                ([17, 19, 23, 29], true),
                ([17, 19, 23, 29], false),
            ] {
                let tokens: Vec<Vec<u32>> = lengths
                    .iter()
                    .zip(starts)
                    .enumerate()
                    .map(|(i, (&rows, start))| {
                        (0..rows)
                            .map(|row| ((start + u64::from(row) + i as u64 * 7) % 500) as u32)
                            .collect()
                    })
                    .collect();
                let steps: Vec<_> = tokens
                    .iter()
                    .enumerate()
                    .map(|(i, tokens)| Step {
                        request: i as u64 + 1,
                        tokens,
                        text: None,
                    })
                    .collect();
                let hashed = oracle.hash(&steps).unwrap();
                let batch = BatchRequestV1::new(
                    BatchWireVersionV1::V3,
                    tokens
                        .iter()
                        .enumerate()
                        .map(|(i, tokens)| {
                            BatchItemV1::new(i as u64 + 1, false, lengths[i], true, tokens.clone())
                        })
                        .collect(),
                    lengths
                        .iter()
                        .enumerate()
                        .map(|(i, &rows)| {
                            let end = starts[i] + u64::from(rows);
                            BatchAppendV1::new(
                                starts[i],
                                u64::from(rows),
                                end,
                                (0..end.div_ceil(64)).map(|p| i as u64 * 32 + p).collect(),
                                None,
                            )
                        })
                        .collect(),
                )
                .with_state_slots(slots.to_vec())
                .encode()
                .unwrap();
                let request = encode(
                    &batch,
                    &lengths
                        .iter()
                        .enumerate()
                        .map(|(i, &rows)| Boundary {
                            request_id: i as u64 + 1,
                            prompt_tokens: starts[i] + u64::from(rows),
                            chunk_end: starts[i] + u64::from(rows),
                            chunk_tokens: 8192,
                        })
                        .collect::<Vec<_>>(),
                )
                .unwrap();
                let begin = host.journal.borrow().len();
                instance
                    .execute_prefill(&request, &mut Views::default())
                    .unwrap_or_else(|e| panic!("{e:?}: {:?}", host.logs.borrow()));
                if on {
                    let columns = HashTables::of(&FLASH).unwrap().columns();
                    for (layer, (_, span)) in spans.iter().enumerate() {
                        let expected: Vec<_> = hashed
                            .ids()
                            .chunks_exact(columns * 2)
                            .flat_map(|row| &row[layer * columns..(layer + 1) * columns])
                            .flat_map(|id| id.to_le_bytes())
                            .collect();
                        assert_eq!(
                            host.uploads.borrow()[&(span.buffer.raw, span.offset)],
                            expected
                        );
                    }
                    let journal = host.journal.borrow();
                    let lines = &journal[begin..];
                    let enqueues: Vec<_> = lines
                        .iter()
                        .enumerate()
                        .filter_map(|(at, s)| s.starts_with("async mapped ids ").then_some(at))
                        .collect();
                    assert_eq!(enqueues.len(), 2);
                    assert!(
                        lines[enqueues[0]..enqueues[1]]
                            .iter()
                            .all(|s| !s.starts_with("launch "))
                    );
                    let wait = lines
                        .iter()
                        .position(|s| s.starts_with("async mapped wait "))
                        .unwrap();
                    let complete = lines
                        .iter()
                        .position(|s| s.starts_with("async mapped complete"))
                        .unwrap();
                    assert!(
                        lines[enqueues[1]..wait]
                            .iter()
                            .any(|s| s.starts_with("launch ")),
                        "target compute must overlap the gather"
                    );
                    let last_wait = lines
                        .iter()
                        .rposition(|s| s.starts_with("async mapped wait "))
                        .unwrap();
                    assert!(
                        lines[last_wait..complete]
                            .iter()
                            .any(|s| s.starts_with("download ")),
                        "finish must follow the consolidated status readback"
                    );
                    assert_eq!(
                        lines
                            .iter()
                            .filter(|s| s.starts_with("async mapped wait "))
                            .count(),
                        8
                    );
                    assert_eq!(
                        lines
                            .iter()
                            .filter(|s| s.starts_with("async mapped complete"))
                            .count(),
                        2
                    );
                }
                if cancel {
                    instance.cancel().unwrap();
                    starts.fill(0);
                    for id in 1..=4 {
                        oracle.forget(id);
                    }
                } else {
                    instance.commit().unwrap();
                    for (i, step) in steps.iter().enumerate() {
                        oracle.commit(step, lengths[i] as usize).unwrap();
                        starts[i] += u64::from(lengths[i]);
                    }
                }
                for id in 1..=4 {
                    let context = crate::instance::dspark_context(instance.handle(), id);
                    assert_eq!(context.is_none(), cancel);
                    contexts.push(context.map(|context| {
                        (
                            context.end,
                            context.retained_start,
                            context.projected().to_vec(),
                        )
                    }));
                }
            }
            let observed = (launches(&host), contexts);
            if let Some(reference) = &reference {
                assert_eq!(&observed, reference);
            } else {
                reference = Some(observed);
            }
            for id in 1..=4 {
                instance.retire_request(id).unwrap();
            }
            instance.destroy().unwrap();
            assert_eq!(host.acquired.get(), host.released.get());
        }
    }
}
