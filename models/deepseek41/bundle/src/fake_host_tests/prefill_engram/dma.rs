use super::*;

#[test]
fn dma_prefill_enqueues_both_layers_and_joins_before_decoders() {
    for (speculative, rows) in [(false, 512), (false, 8192), (true, 512), (true, 8192)] {
        for delayed in [false, true] {
            let host = FakeHost::new(Shape::new(&FLASH, Ranks::TP4).unwrap(), KV_LAYOUTS[0]);
            if speculative {
                host.speculation.set(Some((5, 32)));
            }
            let mut instance = create_backend_with_options(
                &host,
                true,
                completion_backend(&host, delayed),
                &[("prefill_dma_engram", true.into())],
            )
            .unwrap_or_else(|e| panic!("{e:?}: {:?}", host.logs.borrow()));
            let begin = host.journal.borrow().len();
            let mapped = host.mapped_gather_calls.get();
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
            assert_eq!(host.mapped_gather_calls.get(), mapped);
            {
                let journal = host.journal.borrow();
                let step = &journal[begin..];
                assert!(
                    step.iter().any(|s| s.contains("chop_comm_owned_sum_")),
                    "DMA arithmetic must execute"
                );
                let first_launch = step.iter().position(|s| s.starts_with("launch ")).unwrap();
                assert_eq!(
                    step[..first_launch]
                        .iter()
                        .filter(|s| s.starts_with("async mapped ids "))
                        .count(),
                    2
                );
                assert_eq!(
                    step.iter()
                        .filter(|s| s.starts_with("async mapped wait "))
                        .count(),
                    8
                );
                assert_eq!(
                    step.iter()
                        .filter(|s| s.starts_with("async mapped complete"))
                        .count(),
                    2
                );
                for layer in [1, 14] {
                    let consumer = step
                        .iter()
                        .position(|s| s.contains(&format!("layers.{layer}.engram.embed.weight")))
                        .unwrap();
                    assert!(step[..consumer].iter().any(|s| s.starts_with("launch ")));
                }
            }
            instance.commit().unwrap();
            instance.retire_request(7).unwrap();
            instance.destroy().unwrap();
            assert_eq!(host.acquired.get(), host.released.get());
        }
    }
}

#[test]
fn dma_prefill_failure_between_enqueue_and_join_retains_destinations() {
    for speculative in [false, true] {
        for (failure, after) in [("enqueue", 1), ("wait", 0), ("wait", 4), ("complete", 0)] {
            let host = FakeHost::new(Shape::new(&FLASH, Ranks::TP4).unwrap(), KV_LAYOUTS[0]);
            if speculative {
                host.speculation.set(Some((5, 32)));
            }
            let backend =
                dspark::host::MappedHost::with_gather_failure(Rc::clone(&host), failure, after);
            let mut instance = create_backend_with_options(
                &host,
                true,
                backend,
                &[("prefill_dma_engram", true.into())],
            )
            .unwrap();
            let request = encode(
                &prompt(7, 0, 0, 512, true),
                &[Boundary {
                    request_id: 7,
                    prompt_tokens: 512,
                    chunk_end: 512,
                    chunk_tokens: 8192,
                }],
            )
            .unwrap();
            let mut sink = Views::default();
            assert!(instance.execute_prefill(&request, &mut sink).is_err());
            assert_eq!(sink.rows(), 0);
            assert!(
                host.journal
                    .borrow()
                    .iter()
                    .any(|s| s == &format!("async mapped failure {failure}"))
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
                    .all(|s| !s.starts_with("launch ") && !s.starts_with("async mapped ids "))
            );
            match instance.destroy() {
                Ok(()) => assert_eq!(host.acquired.get(), host.released.get()),
                Err(_) => {
                    assert!(instance.is_poisoned());
                    assert!(
                        host.journal.borrow()[begin..]
                            .iter()
                            .all(|s| !s.starts_with("async mapped release "))
                    );
                }
            }
        }
    }
}

#[test]
fn dma_prefill_option_preserves_launches_through_cancel_and_reuse() {
    for speculative in [false, true] {
        let mut reference = None;
        for enabled in [false, true] {
            let host = FakeHost::new(Shape::new(&FLASH, Ranks::TP4).unwrap(), KV_LAYOUTS[0]);
            if speculative {
                host.speculation.set(Some((5, 32)));
            }
            let mut instance = create_backend_with_options(
                &host,
                true,
                completion_backend(&host, true),
                &[("prefill_dma_engram", enabled.into())],
            )
            .unwrap();
            let initial = indices(instance.handle());
            for (request_id, start, rows, end, cancel) in [
                (7, 0, 8192, 8704, false),
                (7, 8192, 512, 8704, true),
                (8, 0, 512, 512, false),
            ] {
                let request = encode(
                    &prompt(request_id, 0, start, rows, start + rows == end),
                    &[Boundary {
                        request_id,
                        prompt_tokens: end,
                        chunk_end: start + rows,
                        chunk_tokens: 8192,
                    }],
                )
                .unwrap();
                instance
                    .execute_prefill(&request, &mut Views::default())
                    .unwrap_or_else(|e| panic!("{e:?}: {:?}", host.logs.borrow()));
                assert_eq!(indices(instance.handle()), initial);
                if cancel {
                    instance.cancel().unwrap();
                } else {
                    instance.commit().unwrap();
                }
            }
            let observed = launches(&host);
            if let Some(reference) = &reference {
                assert_eq!(&observed, reference);
            } else {
                reference = Some(observed);
            }
            instance.retire_request(8).unwrap();
            instance.destroy().unwrap();
            assert_eq!(host.acquired.get(), host.released.get());
        }
    }
}

#[test]
fn dma_owner_route_counts_match_submitted_kernels() {
    for speculative in [false, true] {
        for enabled in [false, true] {
            bundle_runtime::routes::with_logging(|| {
                let host = FakeHost::new(Shape::new(&FLASH, Ranks::TP4).unwrap(), KV_LAYOUTS[0]);
                if speculative {
                    host.speculation.set(Some((5, 32)));
                }
                let mut instance = create_backend_with_options(
                    &host,
                    true,
                    completion_backend(&host, true),
                    &[("prefill_dma_engram", enabled.into())],
                )
                .unwrap();
                let request = encode(
                    &prompt(7, 0, 0, 512, true),
                    &[Boundary {
                        request_id: 7,
                        prompt_tokens: 512,
                        chunk_end: 512,
                        chunk_tokens: 8192,
                    }],
                )
                .unwrap();
                let begin = host.journal.borrow().len();
                let log_begin = host.logs.borrow().len();
                instance
                    .execute_prefill(&request, &mut Views::default())
                    .unwrap();
                let issued = host.journal.borrow()[begin..]
                    .iter()
                    .filter(|line| {
                        line.starts_with("launch ") && line.contains("chop_comm_owned_sum_")
                    })
                    .count() as u64;
                assert!(issued > 0);
                let counted: u64 = host.logs.borrow()[log_begin..]
                    .iter()
                    .filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
                    .filter(|value| value["schema"] == "chopinfer.route.v1")
                    .flat_map(|value| {
                        value["entries"]
                            .as_object()
                            .unwrap()
                            .values()
                            .cloned()
                            .collect::<Vec<_>>()
                    })
                    .filter(|entry| entry["family"] == "prefill_dma")
                    .map(|entry| entry["launches"].as_u64().unwrap())
                    .sum();
                assert_eq!(
                    counted, issued,
                    "every submitted DMA owner sum must be witnessed"
                );
                instance.commit().unwrap();
                instance.retire_request(7).unwrap();
                instance.destroy().unwrap();
            });
        }
    }
}

/// on every stream, the first launch that reads a layer's
/// emulated Engram rows comes after that stream's join for the layer.
#[test]
fn dma_join_precedes_first_destination_read_on_every_stream() {
    fn spans(line: &str) -> Vec<(u64, u64, u64)> {
        let mut out = Vec::new();
        let mut rest = line;
        while let Some(at) = rest.find("raw: ") {
            rest = &rest[at + 5..];
            let raw: u64 = rest[..rest.find(' ').unwrap()].parse().unwrap();
            let at = rest.find("offset: ").unwrap();
            rest = &rest[at + 8..];
            let offset: u64 = rest[..rest.find(',').unwrap()].parse().unwrap();
            let at = rest.find("length: ").unwrap();
            rest = &rest[at + 8..];
            let end = rest.find(|c: char| !c.is_ascii_digit()).unwrap();
            out.push((raw, offset, rest[..end].parse().unwrap()));
        }
        out
    }
    for speculative in [false, true] {
        for rows in [512, 769, 8192] {
            let host = FakeHost::new(Shape::new(&FLASH, Ranks::TP4).unwrap(), KV_LAYOUTS[0]);
            if speculative {
                host.speculation.set(Some((5, 32)));
            }
            let mut instance = create_backend_with_options(
                &host,
                true,
                completion_backend(&host, true),
                &[("prefill_dma_engram", true.into())],
            )
            .unwrap();
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
                .unwrap();
            {
                let journal = host.journal.borrow();
                let step = &journal[begin..];
                let mut checked = 0;
                for (at, line) in step.iter().enumerate() {
                    let Some(rest) = line.strip_prefix("async mapped wait ") else {
                        continue;
                    };
                    let stream = rest.trim().to_owned();
                    let uploads: Vec<_> = step[at + 1..]
                        .iter()
                        .take_while(|l| {
                            !l.starts_with("launch ") && !l.starts_with("async mapped wait ")
                        })
                        .filter(|l| l.starts_with(&format!("upload {stream} ")))
                        .flat_map(|l| spans(l))
                        .collect();
                    assert!(!uploads.is_empty(), "{line}: no emulated rows");
                    let reads = |l: &str| {
                        spans(l).iter().any(|&(raw, offset, length)| {
                            uploads
                                .iter()
                                .any(|&(r, o, n)| r == raw && offset < o + n && o < offset + length)
                        })
                    };
                    let first = step
                        .iter()
                        .position(|l| l.starts_with(&format!("launch {stream} ")) && reads(l))
                        .expect("a consumer reads the joined rows");
                    assert!(
                        first > at,
                        "speculative={speculative} rows={rows}: stream {stream} reads Engram rows at {first} before its join at {at}"
                    );
                    checked += 1;
                }
                assert_eq!(checked, 8);
            }
            instance.commit().unwrap();
            instance.retire_request(7).unwrap();
            instance.destroy().unwrap();
        }
    }
}

fn refused_prefill(
    host: &Rc<Host>,
    instance: &mut BundleInstanceV1,
    speculative: bool,
    rows: u64,
) -> Vec<String> {
    let sources = FLASH.shape.csa.index_sources.len();
    // Large AR prefills read one status plane at a time. DSpark prefills
    // have no draft cycle, so rank 0 reads its saved planes together.
    let planes = if speculative { sources + 2 } else { 1 };
    let mut status = vec![0; planes * rows as usize * 4];
    let offset = if speculative {
        sources * rows as usize * 4
    } else {
        0
    };
    status[offset] = 1;
    host.injected_after
        .set(if speculative { 0 } else { sources });
    host.injected.replace(Some(status));
    let begin = host.journal.borrow().len();
    let mut sink = Views::default();
    assert_eq!(
        instance
            .execute_prefill(&status_request(7, rows), &mut sink)
            .unwrap_err(),
        BundleAbiError::Status(BundleStatusV1::BundleError)
    );
    assert_eq!(sink.rows(), 0);
    assert!(
        host.injected.borrow().is_none(),
        "row-status refusal must consume its injected read"
    );
    assert!(
        host.logs
            .borrow()
            .iter()
            .any(|line| line.contains("quantized input on rank 0"))
    );
    host.journal.borrow()[begin..].to_vec()
}

fn status_request(id: u64, rows: u64) -> Vec<u8> {
    encode(
        &prompt(id, 0, 0, rows, true),
        &[Boundary {
            request_id: id,
            prompt_tokens: rows,
            chunk_end: rows,
            chunk_tokens: 8192,
        }],
    )
    .unwrap()
}

fn assert_gathers_settled(refused: &[String]) {
    assert_eq!(
        refused
            .iter()
            .filter(|s| s.starts_with("async mapped wait "))
            .count(),
        8
    );
    let complete = refused
        .iter()
        .position(|line| line == "async mapped complete")
        .expect("the refused target must release pending gather ownership");
    assert_eq!(
        refused
            .iter()
            .filter(|line| *line == "async mapped complete")
            .count(),
        2
    );
    for stream in 2..=5 {
        let joined = refused
            .iter()
            .rposition(|line| line == &format!("async mapped wait {stream}"))
            .unwrap();
        assert!(
            refused[joined + 1..complete]
                .iter()
                .any(|line| line == &format!("synchronize {stream}"))
        );
    }
}

#[test]
fn prefill_status_refusal_settles_gathers_and_serves_next_step() {
    for rows in [193, 512, 769] {
        let host = FakeHost::new(Shape::new(&FLASH, Ranks::TP4).unwrap(), KV_LAYOUTS[0]);
        let mut instance = create_backend_with_options(
            &host,
            true,
            completion_backend(&host, true),
            &[("prefill_dma_engram", true.into())],
        )
        .unwrap();
        let refused = refused_prefill(&host, &mut instance, false, rows);
        assert!(instance.commit().is_err());
        instance.retire_request(7).unwrap();
        instance
            .execute_prefill(&status_request(9, rows), &mut Views::default())
            .unwrap_or_else(|e| panic!("rows={rows}: {e:?}: {:?}", host.logs.borrow()));
        assert_gathers_settled(&refused);
        instance.commit().unwrap();
        instance.retire_request(9).unwrap();
        instance.destroy().unwrap();
        assert_eq!(host.acquired.get(), host.released.get());
    }
}

#[test]
fn dspark_status_refusal_settles_gathers_before_terminal_poison() {
    for rows in [193, 512, 769] {
        for enabled in [false, true] {
            let host = FakeHost::new(Shape::new(&FLASH, Ranks::TP4).unwrap(), KV_LAYOUTS[0]);
            host.speculation.set(Some((5, 32)));
            let mut instance = create_backend_with_options(
                &host,
                true,
                completion_backend(&host, true),
                &[("prefill_dma_engram", enabled.into())],
            )
            .unwrap();
            let refused = refused_prefill(&host, &mut instance, true, rows);
            if enabled || rows < 512 {
                assert_gathers_settled(&refused);
            }
            // execute_packed settles the target before returning the error to
            // Runtime::host_result, which retains DSpark's terminal policy.
            assert_eq!(
                crate::instance::prefill_engram::refusal_state(instance.handle()),
                (true, true)
            );
            assert_eq!(
                instance.retire_request(7).unwrap_err(),
                BundleAbiError::Status(BundleStatusV1::InvalidState)
            );
            assert!(
                instance
                    .execute_prefill(&status_request(9, rows), &mut Views::default())
                    .is_err()
            );
            match instance.destroy() {
                Ok(()) => assert_eq!(host.acquired.get(), host.released.get()),
                Err(error) => {
                    assert_eq!(error, BundleAbiError::Poisoned);
                    assert!(instance.is_poisoned());
                }
            }
        }
    }
}

#[cfg(not(feature = "diagnostic-taps"))]
#[test]
fn ready_prefill_refusal_settles_after_publication_and_serves_next_cycle() {
    // runtime-host/src/executor.rs:1222 rejects device-result sampling on the
    // negotiated prefill envelope. Use ordinary AR execute for this order;
    // the negotiated-prefill tests above cover DMA separately.
    use super::super::device_results::ReadySink;
    for (rows, enabled) in [(193, false), (512, true)] {
        let host = FakeHost::new(Shape::new(&FLASH, Ranks::TP4).unwrap(), KV_LAYOUTS[0]);
        let mut instance = create_backend_with_options(
            &host,
            true,
            completion_backend(&host, true),
            &[("prefill_dma_engram", enabled.into())],
        )
        .unwrap();
        let sources = FLASH.shape.csa.index_sources.len();
        let mut status = vec![0; rows as usize * 4];
        status[0] = 1;
        host.injected_after.set(sources);
        host.injected.replace(Some(status));
        for (epoch, id) in [(9, 7), (10, 9)] {
            let mut sink = ReadySink {
                host: host.clone(),
                epoch,
                ids: vec![id],
                ready: Vec::new(),
                legacy: Vec::new(),
            };
            let from = host.journal.borrow().len();
            let result = instance.execute(&prompt(id, 0, 0, rows, true), &mut sink);
            assert!(sink.legacy.is_empty());
            assert_eq!(
                sink.ready
                    .iter()
                    .map(|view| view.logits.row_count)
                    .sum::<u32>(),
                1
            );
            let lines = host.journal.borrow()[from..].to_vec();
            let ready = lines
                .iter()
                .position(|line| line == "ready-sample")
                .unwrap();
            let status = lines
                .iter()
                .position(|line| line.starts_with("download "))
                .unwrap();
            let complete = lines
                .iter()
                .position(|line| line == "async mapped complete")
                .unwrap();
            assert!(
                ready < status && status < complete,
                "device result, status, gather completion order: {rows}"
            );
            if id == 7 {
                assert_eq!(
                    result,
                    Err(BundleAbiError::Status(BundleStatusV1::BundleError))
                );
                assert!(host.injected.borrow().is_none());
                assert!(instance.commit().is_err());
                assert_gathers_settled(&lines);
            } else {
                result.unwrap();
                instance.commit().unwrap();
            }
            instance.retire_request(id).unwrap();
        }
        instance.destroy().unwrap();
        assert_eq!(host.acquired.get(), host.released.get());
    }
}

#[test]
fn poisoned_dma_prefill_never_waits_on_healthy_pending_gathers() {
    let create = || {
        let host = FakeHost::new(Shape::new(&FLASH, Ranks::TP4).unwrap(), KV_LAYOUTS[0]);
        let instance = create_backend_with_options(
            &host,
            true,
            completion_backend(&host, true),
            &[("prefill_dma_engram", true.into())],
        )
        .unwrap();
        (host, instance)
    };
    let (reference, mut instance) = create();
    let from = reference.journal.borrow().len();
    instance
        .execute_prefill(&status_request(7, 769), &mut Views::default())
        .unwrap();
    let lines = reference.journal.borrow()[from..].to_vec();
    instance.commit().unwrap();
    instance.retire_request(7).unwrap();
    instance.destroy().unwrap();
    let at = lines
        .iter()
        .position(|line| line.starts_with("launch ") && line.contains("chop_comm_owned_sum_"))
        .unwrap();
    let before = lines[..at]
        .iter()
        .filter(|line| line.starts_with("launch "))
        .count();
    let (host, mut instance) = create();
    host.device.fail_launch.set(Some(before as u32));
    let from = host.journal.borrow().len();
    assert_eq!(
        instance.execute_prefill(&status_request(7, 769), &mut Views::default()),
        Err(BundleAbiError::Status(BundleStatusV1::HostError))
    );
    let failed = host.journal.borrow()[from..].to_vec();
    assert_eq!(
        failed
            .iter()
            .filter(|line| line.starts_with("async mapped ids "))
            .count(),
        2
    );
    assert!(!failed.iter().any(|line| line == "async mapped complete"));
    assert!(
        host.logs
            .borrow()
            .iter()
            .any(|line| line.contains("collectives unmatched"))
    );
    let from = host.journal.borrow().len();
    assert_eq!(
        instance.execute_prefill(&status_request(9, 769), &mut Views::default()),
        Err(BundleAbiError::Status(BundleStatusV1::InvalidState))
    );
    // The public retry refuses at the table poison check first. Pin the settle
    // invariant directly too, so that earlier refusal cannot mask a missing guard.
    let settled = crate::instance::prefill_engram::settle_refused(instance.handle());
    assert!(
        host.journal.borrow()[from..].iter().all(
            |line| !line.starts_with("synchronize ") && !line.starts_with("async mapped wait ")
        ),
        "poisoned collective must refuse before any gather join or stream fence"
    );
    assert_eq!(
        settled,
        Err(BundleAbiError::Status(BundleStatusV1::InvalidState))
    );
    assert!(instance.destroy().is_err());
}
