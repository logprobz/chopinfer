//! Anchored templates rebuild slot seeds and ledger ownership on every step.
use super::*;
use crate::instance::execute::{plan_templates, plan_timing, skip};
use bundle_runtime::fake_host::RecordingSink;
use runtime_bundle_sdk::prefill::{Boundary, encode_with_cuts};
use runtime_bundle_sdk::{BatchAppendV1, BatchItemV1, BatchRequestV1, BatchWireVersionV1};

type Sink = RecordingSink<super::conformance::Limits>;

fn host(dspark: bool) -> Rc<Host> {
    let host = FakeHost::new(Shape::new(&FLASH, Ranks::TP4).unwrap(), KV_LAYOUTS[0]);
    host.drafter.set(dspark);
    if dspark {
        host.speculation.set(Some((5, 32)));
    }
    host
}

fn create(host: &Rc<Host>, sequences: u32) -> BundleInstanceV1 {
    let mut record = crate::config::tests::record(&host.shape, host.kv, sequences, false);
    for (name, value) in [
        ("prefill_chunk_tokens", 8192.into()),
        ("prefill_upper_skip", true.into()),
        ("prefill_plan_cache_verify", true.into()),
    ] {
        record = crate::config::tests::with_option(record, name, value);
    }
    api()
        .with_prefill_api(crate::api::BUNDLE_DEEPSEEK41_PREFILL_API_V1)
        .unwrap()
        .with_prefill_cuts_api(crate::instance::prefill::PREFILL_CUTS_API)
        .unwrap()
        .with_prefill_anchors_api(crate::api::BUNDLE_DEEPSEEK41_PREFILL_ANCHORS_API_V2)
        .unwrap()
        .create(
            HostServicesTableV1::new(Rc::new(dspark::MappedHost::new(Rc::clone(host)))),
            &record,
        )
        .unwrap_or_else(|e| panic!("{e:?}: {:?}", host.logs.borrow()))
}

/// (request, slot, prefix, rows), all prompt-ending anchored items.
fn batch(items: &[(u64, u64, u64, u64)]) -> (Vec<u8>, Vec<u8>) {
    let mut batch_items = Vec::new();
    let mut appends = Vec::new();
    let mut slots = Vec::new();
    let mut boundaries = Vec::new();
    for &(request, slot, prefix, rows) in items {
        let end = prefix + rows;
        batch_items.push(
            BatchItemV1::new(
                request,
                false,
                rows as u32,
                true,
                (prefix..end).map(|p| p as u32).collect(),
            )
            .with_anchor(true),
        );
        appends.push(BatchAppendV1::new(
            prefix,
            rows,
            end,
            (0..end.div_ceil(64)).map(|p| slot * 32 + p).collect(),
            None,
        ));
        slots.push(slot);
        boundaries.push(Boundary {
            request_id: request,
            prompt_tokens: end,
            chunk_end: end,
            chunk_tokens: 8192,
        });
    }
    let original = BatchRequestV1::new(BatchWireVersionV1::V3, batch_items, appends)
        .with_state_slots(slots)
        .encode()
        .unwrap();
    let envelope = encode_with_cuts(&original, &boundaries).unwrap();
    (original, envelope)
}

#[test]
fn anchor_allowance_is_capped_at_served_sequences() {
    for dspark in [false, true] {
        // Three also checks that the cap is the served count, not its bucket.
        for sequences in [8, 4, 3, 1] {
            let host = host(dspark);
            let mut instance = create(&host, sequences);
            assert_eq!(instance.prefill_anchor_items(), Ok(Some(sequences)));
            let items: Vec<_> = (0..u64::from(sequences))
                .map(|slot| (slot + 1, slot, 0, 65))
                .collect();
            instance
                .execute_prefill_cuts(&batch(&items).1, &mut Sink::default())
                .unwrap();
            instance
                .commit()
                .unwrap_or_else(|e| panic!("{e:?}: {:?}", host.logs.borrow()));
            let pages: Vec<_> = (130..130 + instance.describe_prefix_state().unwrap()).collect();
            for (request, _, _, _) in items {
                instance.capture_prefix_state(request, 64, &pages).unwrap();
                instance.retire_request(request).unwrap();
            }
            instance.destroy().unwrap();
            assert_eq!(host.acquired.get(), host.released.get());
        }
    }
}

/// The per-step cap remains eight even when the instance serves more items.
#[test]
fn more_than_eight_anchor_items_are_refused_before_launching() {
    for dspark in [false, true] {
        let host = host(dspark);
        let mut instance = create(&host, 9);
        assert_eq!(instance.prefill_anchor_items(), Ok(Some(8)));
        let items: Vec<_> = (0..9).map(|slot| (slot + 1, slot, 0, 65)).collect();
        let before = host.device.journal.borrow().len();
        assert_eq!(
            instance.execute_prefill_cuts(&batch(&items).1, &mut Sink::default()),
            Err(BundleAbiError::Status(BundleStatusV1::InvalidArgument))
        );
        assert_eq!(host.device.journal.borrow().len(), before);
        instance.destroy().unwrap();
        assert_eq!(host.acquired.get(), host.released.get());
    }
}

/// Device spans in the fake-host journal, in argument order.
fn spans(line: &str) -> Vec<DeviceBufferSpanV1> {
    let mut rest = line;
    let mut found = Vec::new();
    while let Some(at) = rest.find("raw: ") {
        rest = &rest[at + 5..];
        let number = |s: &str| {
            s.split(|c: char| !c.is_ascii_digit())
                .next()
                .unwrap()
                .parse::<u64>()
                .unwrap()
        };
        let buffer = BundleDeviceBufferHandleV1 { raw: number(rest) };
        rest = &rest[rest.find("offset: ").unwrap() + 8..];
        let offset = number(rest);
        rest = &rest[rest.find("length: ").unwrap() + 8..];
        found.push(DeviceBufferSpanV1::new(buffer, offset, number(rest)));
    }
    found
}

fn inside(span: DeviceBufferSpanV1, buffer: DeviceBufferSpanV1) -> bool {
    span.buffer == buffer.buffer
        && span.offset >= buffer.offset
        && span.offset + span.length <= buffer.offset + buffer.length
}

/// Every rank seeds its own anchor before the tails seed their writable
/// entries, and before any kernel (including DMA producers) can write its
/// slot state. Other ranks' uploads and Engram gathers may run earlier.
fn check_seed_order(instance: &BundleInstanceV1, lines: &[String], anchors: usize) {
    let (spares, states) = crate::instance::anchor_spans(instance.handle());
    let copies: Vec<_> = lines
        .iter()
        .enumerate()
        .filter(|(_, l)| l.starts_with("copy "))
        .map(|(at, l)| (at, spans(l)))
        .collect();
    for (spare, state) in spares.iter().zip(&states) {
        let tails = state.named("compressor.tail").unwrap();
        let rings = state.named("window.ring").unwrap();
        let seeded: Vec<_> = copies
            .iter()
            .filter(|(_, views)| spare.buffers.iter().flatten().any(|&b| inside(views[1], b)))
            .collect();
        assert!(seeded.iter().any(|(_, views)| inside(views[0], rings)));
        assert_eq!(
            seeded
                .iter()
                .filter(|(_, views)| inside(views[0], tails))
                .count(),
            3 * anchors
        );
        let own: Vec<_> = copies
            .iter()
            .filter(|(_, views)| inside(views[1], tails))
            .collect();
        assert_eq!(own.len(), 3 * anchors);
        assert!(seeded.last().unwrap().0 < own[0].0);
        let first_state_launch = lines
            .iter()
            .position(|line| {
                line.starts_with("launch ")
                    && spans(line)
                        .iter()
                        .any(|&s| inside(s, rings) || inside(s, tails))
            })
            .expect("a kernel using this rank's slot state");
        assert!(seeded.last().unwrap().0 < first_state_launch);
    }
}

/// Tag target state by rank and offset, so using another slot's seeds
/// changes the snapshot bytes. The fake host does not run model arithmetic.
fn tag_target_state(host: &Host, instance: &BundleInstanceV1) {
    let (_, states) = crate::instance::anchor_spans(instance.handle());
    for (rank, state) in states.iter().enumerate() {
        for (buffer, name) in ["window.ring", "compressor.tail"].iter().enumerate() {
            let span = state.named(name).unwrap();
            let bytes: Rc<[u8]> = (0..span.length)
                .map(|at| ((at / 4096 + rank as u64 * 31 + buffer as u64 * 7) % 251) as u8)
                .collect::<Vec<_>>()
                .into();
            host.memory
                .borrow_mut()
                .write(span, vec![(span.length, Origin::Uploaded(bytes, 0))]);
        }
    }
}

/// The target part of a snapshot: all ring and compressor-tail
/// bytes on all ranks, without masking any cells. DSpark drafter state
/// has separate coverage.
fn target_snapshot(
    host: &Host,
    instance: &mut BundleInstanceV1,
    request: u64,
    tokens: u64,
    interior: bool,
) -> Vec<u8> {
    let (spares, states) = crate::instance::anchor_spans(instance.handle());
    let targets: Vec<_> = if interior {
        spares
            .iter()
            .flat_map(|s| s.buffers[..2].iter().flatten().copied())
            .collect()
    } else {
        states
            .iter()
            .flat_map(|state| {
                ["window.ring", "compressor.tail"].map(|name| state.named(name).unwrap())
            })
            .collect()
    };
    let pages: Vec<_> = (130..130 + instance.describe_prefix_state().unwrap()).collect();
    let before = host.device.journal.borrow().len();
    instance
        .capture_prefix_state(request, tokens, &pages)
        .unwrap_or_else(|e| {
            panic!(
                "request={request} tokens={tokens}: {e:?}: {:?}",
                host.logs.borrow()
            )
        });
    let bytes: Vec<_> = host.device.journal.borrow()[before..]
        .iter()
        .filter(|l| super::prefill_anchor::is_copy(l))
        .map(|l| spans(l))
        .filter(|v| targets.iter().any(|&s| inside(v[0], s)))
        .flat_map(|v| host.memory.borrow().bytes(v[1]))
        .collect();
    assert!(!bytes.is_empty());
    bytes
}

fn snapshots(dspark: bool, reuse: bool, packed: bool) -> Vec<Vec<u8>> {
    let host = host(dspark);
    let mut instance = create(&host, 4);
    let mut snapshots = Vec::new();
    // Each anchor is 67 packed rows after an odd start, at a different
    // absolute 64-token boundary. Both ring seeds and tail seeds are needed.
    let batches = if packed {
        // Swap request order and move both slots. Request 7 uses spare 0
        // in the first step and spare 1 in the reused template's step.
        vec![
            vec![(7, 0, 829, 68), (19, 2, 957, 68)],
            vec![(19, 3, 1085, 68), (7, 1, 1021, 68)],
        ]
    } else {
        vec![
            vec![(7, 0, 829, 68)],
            vec![(19, 2, 957, 68)],
            vec![(23, 1, 1021, 68)],
        ]
    };
    for (index, items) in batches.iter().enumerate() {
        if !reuse && index > 0 {
            instance.destroy().unwrap();
            instance = create(&host, 4);
        }
        for &(request, slot, prefix, _) in items {
            plan_timing::restore_without_snapshot(
                instance.handle(),
                request,
                slot as u32,
                &(0..prefix as u32).collect::<Vec<_>>(),
            )
            .unwrap();
        }
        tag_target_state(&host, &instance);
        let before = plan_templates::counts(instance.handle());
        let first = host.device.journal.borrow().len();
        instance
            .execute_prefill_cuts(&batch(items).1, &mut Sink::default())
            .unwrap_or_else(|e| panic!("{e:?}: {:?}", host.logs.borrow()));
        let after = plan_templates::counts(instance.handle());
        assert_eq!(after.hits - before.hits, u64::from(reuse && index > 0));
        assert_eq!(after.kept - before.kept, u64::from(!reuse || index == 0));
        check_seed_order(
            &instance,
            &host.device.journal.borrow()[first..],
            items.len(),
        );
        instance.commit().unwrap();
        let captured = snapshots.len();
        for &(request, slot, prefix, rows) in items {
            assert_eq!(
                crate::instance::ledger_state(instance.handle(), slot as u32),
                (Some((request, prefix + rows)), false)
            );
            // The interior capture reads this request's spare. The later
            // prompt-end capture must still find its own slot and boundary.
            for (tokens, interior) in [((prefix + rows) / 64 * 64, true), (prefix + rows, false)] {
                snapshots.push(target_snapshot(
                    &host,
                    &mut instance,
                    request,
                    tokens,
                    interior,
                ));
            }
            instance.retire_request(request).unwrap();
        }
        if packed {
            assert!(
                snapshots[captured] != snapshots[captured + 2],
                "the two spares hold distinct slot bytes"
            );
        }
    }
    instance.destroy().unwrap();
    assert_eq!(host.acquired.get(), host.released.get());
    snapshots
}

#[test]
fn anchored_templates_rebuild_ownership_boundaries_and_seeds() {
    for dspark in [false, true] {
        for dma in [false, true] {
            skip::with_dma(dma, || {
                assert!(
                    snapshots(dspark, true, false) == snapshots(dspark, false, false),
                    "dspark={dspark} dma={dma}"
                );
            });
        }
    }
}

#[test]
fn two_anchor_templates_follow_reordered_requests_and_slots() {
    for dspark in [false, true] {
        assert!(
            snapshots(dspark, true, true) == snapshots(dspark, false, true),
            "dspark={dspark}"
        );
    }
}

#[test]
fn different_packed_anchor_rows_do_not_share_a_template() {
    for dspark in [false, true] {
        let host = host(dspark);
        let mut instance = create(&host, 4);
        for (request, slot, prefix, hit) in
            [(7, 0, 829, false), (9, 1, 830, false), (11, 2, 957, true)]
        {
            let before = plan_templates::counts(instance.handle());
            plan_timing::plan(instance.handle(), &batch(&[(request, slot, prefix, 68)]).0).unwrap();
            let after = plan_templates::counts(instance.handle());
            assert_eq!(after.hits - before.hits, u64::from(hit));
            assert_eq!(after.kept - before.kept, u64::from(!hit));
        }
        instance.destroy().unwrap();
        assert_eq!(host.acquired.get(), host.released.get());
    }
}
