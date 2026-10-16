use super::*;

fn compact_step(items: &[Item<'_>]) -> BatchRequestV1 {
    BatchRequestV1::new(
        BatchWireVersionV1::V3,
        items
            .iter()
            .map(|&(request, _, previous, inputs)| {
                BatchItemV1::new(
                    request,
                    previous > 0,
                    inputs.len() as u32,
                    true,
                    inputs.to_vec(),
                )
            })
            .collect(),
        items
            .iter()
            .map(|&(_, slot, previous, inputs)| {
                BatchAppendV1::new(
                    previous,
                    inputs.len() as u64,
                    previous + inputs.len() as u64,
                    vec![slot],
                    None,
                )
            })
            .collect(),
    )
    .with_state_slots(items.iter().map(|&(_, slot, _, _)| slot).collect())
}

#[test]
fn hc_precompute_selects_only_tp4_qualified_buckets_and_can_be_disabled() {
    for enabled in [false, true] {
        for shape in Shape::ALL {
            let host = Rc::new(FakeHost::at(shape));
            let mut instance = crate::instance::with_hc_precompute(enabled, || create(&host));
            let prefill = (0..32)
                .map(|i| (10 + i, i, 0, &PROMPT[..]))
                .collect::<Vec<_>>();
            run(&host, &mut instance, "prefill", compact_step(&prefill));
            assert!(!launched(&host, 0).iter().any(|s| s.contains("precompute")));
            instance.commit().unwrap();
            // Decode steps commit: a cancel invalidates in-place linear state.
            let mut lengths = [3_u64; 32];
            for live in [1_u64, 8, 9, 16, 17, 32, 16] {
                let before = host.launches.borrow().len();
                let replays = host.replays.get();
                let request = (0..live)
                    .map(|i| (10 + i, i, lengths[i as usize], &[9][..]))
                    .collect::<Vec<_>>();
                run(&host, &mut instance, "decode", compact_step(&request));
                let names = launched(&host, before);
                let qualified = enabled && live > 8;
                // A repeated bucket replays its graph, without issuing new launches.
                if names.is_empty() {
                    assert!(host.replays.get() > replays);
                } else {
                    assert_eq!(names.iter().any(|s| s.contains("precompute")), qualified);
                    if qualified {
                        check_bottleneck_bindings(&host, before);
                    }
                }
                instance.commit().unwrap();
                for length in &mut lengths[..live as usize] {
                    *length += 1;
                }
            }
            instance.destroy().unwrap();
            assert_eq!(host.acquired.get(), host.released.get());
        }
    }
}

pub(super) fn check_bottleneck_bindings(host: &FakeHost, before: usize) {
    let launches = host.launches.borrow();
    let arguments = host.launch_arguments.borrow();
    let mut producers = BTreeMap::new();
    let mut pairs = 0;
    for ((stream, name, _), args) in launches[before..].iter().zip(&arguments[before..]) {
        if name.contains("hc_in_precompute") || name.contains("hc_down_precompute") {
            let arg = args.last().unwrap();
            let KernelArgumentV1::DeviceBuffer(span) = arg else {
                panic!("producer output is not a buffer")
            };
            producers.insert(*stream, *span);
        } else if name == super::sm120::HC_PRECOMPUTED_MIX || name == super::sm120::HC_PADDED_MIX {
            let KernelArgumentV1::DeviceBuffer(source) = args[1] else {
                panic!("mix input is not a buffer")
            };
            let producer = producers
                .remove(stream)
                .expect("mix needs its producer in the same stream");
            assert_eq!(
                (source.buffer, source.offset, source.length),
                (producer.buffer, producer.offset, producer.length)
            );
            pairs += 1;
        }
    }
    assert!(pairs > 0);
    assert!(producers.is_empty());
}

#[test]
fn padded_mix_is_latched_and_only_changes_precomputed_decode() {
    use super::sm120::{HC_PADDED_MIX, HC_PRECOMPUTED_MIX};
    for enabled in [false, true] {
        for precompute in [false, true] {
            for shape in Shape::ALL {
                let host = Rc::new(FakeHost::at(shape));
                let mut instance = crate::instance::with_hc_padded_mix(enabled, || {
                    crate::instance::with_hc_precompute(precompute, || {
                        let instance = create(&host);
                        assert_resolved(&host);
                        instance
                    })
                });
                let prefill = (0..17)
                    .map(|i| (10 + i, i, 0, &PROMPT[..]))
                    .collect::<Vec<_>>();
                run(&host, &mut instance, "prefill", compact_step(&prefill));
                assert!(!launched(&host, 0).iter().any(|s| s == HC_PADDED_MIX));
                instance.commit().unwrap();
                for (index, live) in [8_u64, 16, 17].into_iter().enumerate() {
                    let before = host.launches.borrow().len();
                    let request = (0..live)
                        .map(|i| {
                            let previous =
                                3 + [8, 16][..index].iter().filter(|&&n| i < n).count() as u64;
                            (10 + i, i, previous, &[9][..])
                        })
                        .collect::<Vec<_>>();
                    run(&host, &mut instance, "decode", compact_step(&request));
                    let names = launched(&host, before);
                    let eligible = precompute && live > 8;
                    assert_eq!(
                        names.iter().any(|s| s == HC_PADDED_MIX),
                        eligible && enabled
                    );
                    assert_eq!(
                        names.iter().any(|s| s == HC_PRECOMPUTED_MIX),
                        eligible && !enabled
                    );
                    if eligible {
                        check_bottleneck_bindings(&host, before);
                    }
                    instance.commit().unwrap();
                }
                instance.destroy().unwrap();
                assert_eq!(host.acquired.get(), host.released.get());
            }
        }
    }
}
