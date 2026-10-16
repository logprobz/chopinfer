use super::*;

#[test]
fn owned_decode_views_follow_bucket_owners_and_replay() {
    for shape in Shape::ALL {
        let host = Rc::new(FakeHost::at(shape));
        let mut instance = crate::instance::with_row_owners(true, || create(&host));
        let prefill = (0..16)
            .map(|i| (10 + i, i, 0, &PROMPT[..]))
            .collect::<Vec<_>>();
        run(&host, &mut instance, "prefill", step(&prefill));
        assert_eq!(
            count(&launched(&host, 0), "chop_comm_exchange_rows_f32_v1"),
            0
        );
        instance.commit().unwrap();
        // Changing buckets and replaying one exercises shared exchange state.
        let mut lengths = [3; 16];
        for (iteration, live) in [1_u64, 5, 16, 5].into_iter().enumerate() {
            let before = host.launches.borrow().len();
            let journal = host.journal.borrow().len();
            let request = step(
                &(0..live)
                    .map(|i| (10 + i, i, lengths[i as usize], &[9][..]))
                    .collect::<Vec<_>>(),
            );
            let mut sink = ViewSink::default();
            instance
                .execute(&request.encode().unwrap(), &mut sink)
                .unwrap();
            let bucket = (live as u32).next_power_of_two();
            let block = bucket.div_ceil(shape.ranks());
            let buffers = crate::instance::logits_spans(instance.handle());
            let mut first = 0;
            for view in &sink.0 {
                let view = view.raw();
                let owner = first / block;
                assert_eq!(view.rank, owner);
                assert_eq!(
                    view.context,
                    BundleContextHandleV1::from_parts(1, owner + 1)
                );
                assert_eq!(view.first_item_index, first);
                assert_eq!(view.row_count, (live as u32 - first).min(block));
                assert_eq!(view.element_type, BUNDLE_LOGITS_ELEMENT_V1_F32);
                let buffer = buffers[owner as usize][3];
                assert_eq!(view.buffer.buffer, buffer.buffer);
                assert_eq!(
                    view.buffer.offset,
                    buffer.offset + u64::from(first) * u64::from(VOCAB) * 4
                );
                first += view.row_count;
            }
            assert_eq!(first, live as u32);
            let synchronizes = host.journal.borrow()[journal..]
                .iter()
                .filter(|line| line.starts_with("synchronize "))
                .count();
            assert!(synchronizes >= shape.ranks() as usize);
            if iteration < 3 {
                assert_eq!(
                    count(&launched(&host, before), "chop_comm_exchange_rows_f32_v1"),
                    shape.ranks() as usize
                );
            } else {
                assert_eq!(
                    count(&launched(&host, before), "chop_comm_exchange_rows_f32_v1"),
                    0
                );
            }
            instance.commit().unwrap();
            lengths[..live as usize]
                .iter_mut()
                .for_each(|length| *length += 1);
        }
        instance.destroy().unwrap();
        assert_eq!(host.acquired.get(), host.released.get());
    }
}

#[test]
fn mixed_prefill_and_owned_decode_keep_item_order() {
    let host = Rc::new(FakeHost::default());
    let mut instance = crate::instance::with_row_owners(true, || create(&host));
    run(
        &host,
        &mut instance,
        "prefill",
        step(&[(10, 0, 0, &PROMPT), (12, 2, 0, &PROMPT)]),
    );
    instance.commit().unwrap();
    let mut sink = ViewSink::default();
    instance
        .execute(
            &step(&[(10, 0, 3, &[9]), (11, 1, 0, &PROMPT), (12, 2, 3, &[8])])
                .encode()
                .unwrap(),
            &mut sink,
        )
        .unwrap();
    assert_eq!(
        sink.0
            .iter()
            .map(|view| {
                let view = view.raw();
                (view.first_item_index, view.row_count, view.rank)
            })
            .collect::<Vec<_>>(),
        [(0, 1, 0), (1, 1, 0), (2, 1, 1)]
    );
    instance.commit().unwrap();
    instance.destroy().unwrap();
}

#[test]
fn owned_wide_decode_exchanges_the_prefill_heads_output() {
    let host = Rc::new(FakeHost::default());
    host.served.set(Some(256));
    let mut instance = crate::instance::with_row_owners(true, || create(&host));
    // Each slot's committed tokens.
    let request = |live: u32, previous: &[u64], tokens: &[u32]| {
        let batch = (0..live)
            .map(|slot| {
                BatchItemV1::new(
                    u64::from(slot) + 10,
                    previous[slot as usize] != 0,
                    tokens.len() as u32,
                    true,
                    tokens.to_vec(),
                )
            })
            .collect();
        let appends = (0..live)
            .map(|slot| {
                BatchAppendV1::new(
                    previous[slot as usize],
                    tokens.len() as u64,
                    previous[slot as usize] + tokens.len() as u64,
                    vec![u64::from(slot)],
                    None,
                )
            })
            .collect();
        BatchRequestV1::new(BatchWireVersionV1::V3, batch, appends)
            .with_state_slots((0..live).map(u64::from).collect())
    };
    run(
        &host,
        &mut instance,
        "prefill",
        request(256, &[0; 256], &PROMPT),
    );
    instance.commit().unwrap();
    let mut lengths = [PROMPT.len() as u64; 256];
    for live in [33_u32, 65, 129, 256] {
        let before = host.launches.borrow().len();
        let mut sink = ViewSink::default();
        instance
            .execute(&request(live, &lengths, &[9]).encode().unwrap(), &mut sink)
            .unwrap_or_else(|error| panic!("{live}: {error:?} {:?}", host.logs.borrow()));
        let block = live.next_power_of_two().div_ceil(4);
        let mut first = 0;
        for view in &sink.0 {
            let view = view.raw();
            assert_eq!(view.first_item_index, first);
            assert_eq!(view.rank, first / block);
            assert_eq!(view.row_count, (live - first).min(block).min(64));
            first += view.row_count;
        }
        assert_eq!(first, live);
        let launches = host.launches.borrow();
        let arguments = host.launch_arguments.borrow();
        let mut exchanges = 0;
        for (index, (stream, symbol, _)) in launches.iter().enumerate().skip(before) {
            if symbol != "chop_comm_exchange_rows_f32_v1" {
                continue;
            }
            let head = launches[..index]
                .iter()
                .rposition(|(other, name, _)| {
                    other == stream && name == "chop_gemm_qwen38_head_f32_prefill_v1"
                })
                .unwrap();
            assert_eq!(
                arguments[index][0], arguments[head][2],
                "exchange reads the actual wide head output"
            );
            exchanges += 1;
        }
        assert_eq!(exchanges, if live == 256 { 0 } else { 4 });
        drop(arguments);
        drop(launches);
        instance.commit().unwrap();
        lengths[..live as usize]
            .iter_mut()
            .for_each(|length| *length += 1);
    }
    instance.destroy().unwrap();
    assert_eq!(host.acquired.get(), host.released.get());
}
