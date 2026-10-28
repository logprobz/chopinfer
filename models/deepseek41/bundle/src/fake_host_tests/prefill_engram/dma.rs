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
