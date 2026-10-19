use super::*;

fn run(k: u32, ids: &[u32], remaining: u64, stops: &[u32], statuses: &[u32]) -> Decision {
    let mut samples = vec![0xa5; 6 * OUTPUT_BYTES];
    for (r, &token) in ids.iter().enumerate() {
        let bytes = &mut samples[r * OUTPUT_BYTES..][..OUTPUT_BYTES];
        bytes[..4].copy_from_slice(&token.to_le_bytes());
        bytes[4..8].copy_from_slice(&statuses.get(r).copied().unwrap_or(0).to_le_bytes());
        bytes[8..16].copy_from_slice(&(-0.25_f64).to_le_bytes());
    }
    oracle(
        Acceptance {
            vocab: 100,
            sample_rows: 6,
            draft_items: u32::from(k > 0),
            stop_ids: stops.len() as u32,
        },
        Input {
            items: &[Item {
                epoch: 19,
                remaining,
                first_sample: 0,
                draft_item: if k > 0 { 0 } else { NONE },
                k,
                stop_offset: 0,
                stop_count: stops.len() as u32,
                reserved: 0,
            }],
            proposals: &[[10, 11, 12, 13, 14]],
            samples: &samples,
            stops,
        },
    )
    .unwrap()[0]
}

#[test]
fn all_budgets_acceptance_depths_stops_and_limits_preserve_host_order() {
    for k in 0..=5 {
        for accepted in 0..=k {
            let mut ids: Vec<_> = (10..=10 + k).collect();
            if accepted < k {
                ids[accepted as usize] = 99;
            }
            for remaining in 1..=7 {
                for stop_at in 0..=k + 1 {
                    let stops = if stop_at <= k {
                        vec![ids[stop_at as usize]]
                    } else {
                        vec![]
                    };
                    let result = run(k, &ids, remaining, &stops, &[]);
                    let mut expected = Decision::failed(19, OK, NONE);
                    // This follows the engine's token walk, independently of the packed oracle.
                    for (position, &token) in ids.iter().enumerate() {
                        let same = position < k as usize && token == 10 + position as u32;
                        if same {
                            expected.accepted += 1;
                        }
                        expected.kept += 1;
                        if position < k as usize && !same {
                            expected.flags |= REJECTED;
                        }
                        if stops.contains(&token) {
                            expected.flags |= TERMINAL;
                            break;
                        }
                        if u64::from(expected.kept) == remaining || !same {
                            break;
                        }
                    }
                    assert_eq!(
                        result, expected,
                        "k={k} accepted={accepted} remaining={remaining} stop={stop_at}"
                    );
                    assert_eq!(Decision::decode(&result.encode()).unwrap(), result);
                }
            }
        }
    }
    let eos = run(5, &[10, 11, 12, 13, 14, 15], 6, &[10], &[]);
    assert_eq!((eos.accepted, eos.kept, eos.flags), (1, 1, TERMINAL));
}

#[test]
fn rejected_empty_suffix_is_ignored_but_corrupt_suffix_refuses_the_view() {
    let ids = [99, 11, 12, 13, 14, 15];
    assert_eq!(run(5, &ids, 6, &[], &[0, 2, 2, 2, 2, 2]).failure, OK);
    for status in [1, 3, 4, u32::MAX] {
        let result = run(5, &ids, 6, &[], &[0, 0, 0, 0, 0, status]);
        assert_eq!(
            (result.failure, result.failure_row, result.kept),
            (BAD_SAMPLE, 5, 0)
        );
    }
    let result = run(5, &[10, 11, 12, 13, 14, 15], 6, &[], &[0, 0, 2, 0, 0, 0]);
    assert_eq!(
        (
            result.failure,
            result.failure_row,
            result.accepted,
            result.kept
        ),
        (EMPTY, 2, 0, 0)
    );
    assert_eq!(run(5, &ids, 6, &[], &[2, 0, 0, 0, 0, 0]).failure, EMPTY);
}

#[test]
fn packed_records_have_exact_offsets_and_refuse_bad_device_metadata() {
    let item = Item {
        epoch: 0x1122334455667788,
        remaining: 9,
        first_sample: 0,
        draft_item: 0,
        k: 5,
        stop_offset: 0,
        stop_count: 1,
        reserved: 0,
    };
    let encoded = item.encode();
    assert_eq!(u64_at(&encoded, 0), item.epoch);
    assert_eq!(u64_at(&encoded, 8), 9);
    assert_eq!(u32_at(&encoded, 24), 5);
    assert_eq!(u32_at(&encoded, 32), 1);
    let op = Acceptance {
        vocab: 100,
        sample_rows: 6,
        draft_items: 1,
        stop_ids: 1,
    };
    let mut samples = vec![0; 6 * OUTPUT_BYTES];
    for r in 0..6 {
        samples[r * OUTPUT_BYTES..r * OUTPUT_BYTES + 4]
            .copy_from_slice(&(10_u32 + r as u32).to_le_bytes());
    }
    for case in 0..10 {
        let mut changed = item;
        let mut proposals = [[10, 11, 12, 13, 14]];
        let mut rows = samples.clone();
        match case {
            0 => changed.remaining = 0,
            1 => changed.k = 6,
            2 => changed.first_sample = 1,
            3 => changed.draft_item = NONE,
            4 => changed.stop_count = 2,
            5 => changed.reserved = 1,
            6 => proposals[0][4] = 100,
            7 => {
                rows[5 * OUTPUT_BYTES..5 * OUTPUT_BYTES + 4].copy_from_slice(&100_u32.to_le_bytes())
            }
            8 => rows[5 * OUTPUT_BYTES + 8..5 * OUTPUT_BYTES + 16]
                .copy_from_slice(&f64::NAN.to_le_bytes()),
            _ => rows[5 * OUTPUT_BYTES + 8..5 * OUTPUT_BYTES + 16]
                .copy_from_slice(&f64::NEG_INFINITY.to_le_bytes()),
        }
        let actual = oracle(
            op,
            Input {
                items: &[changed],
                proposals: &proposals,
                samples: &rows,
                stops: &[99],
            },
        )
        .unwrap()[0];
        assert_eq!(
            actual.failure,
            if case < 6 {
                BAD_DESCRIPTOR
            } else if case == 6 {
                BAD_PROPOSAL
            } else {
                BAD_SAMPLE
            }
        );
        assert_eq!(actual.kept, 0);
    }
}

#[test]
fn mixed_ar_items_use_no_draft_and_padding_is_never_read() {
    let items = [
        Item {
            epoch: 7,
            remaining: 8,
            first_sample: 0,
            draft_item: NONE,
            k: 0,
            stop_offset: 0,
            stop_count: 0,
            reserved: 0,
        },
        Item {
            epoch: 8,
            remaining: 8,
            first_sample: 1,
            draft_item: 0,
            k: 2,
            stop_offset: 0,
            stop_count: 0,
            reserved: 0,
        },
    ];
    let mut samples = vec![0xa5; 6 * OUTPUT_BYTES];
    for (r, token) in [42_u32, 10, 11, 19].into_iter().enumerate() {
        samples[r * OUTPUT_BYTES..r * OUTPUT_BYTES + 4].copy_from_slice(&token.to_le_bytes());
        samples[r * OUTPUT_BYTES + 4..r * OUTPUT_BYTES + 16].fill(0);
    }
    let input = Input {
        items: &items,
        proposals: &[[10, 11, 12, 13, 14]],
        samples: &samples,
        stops: &[],
    };
    let decisions = oracle(
        Acceptance {
            vocab: 100,
            sample_rows: 6,
            draft_items: 1,
            stop_ids: 0,
        },
        input,
    )
    .unwrap();
    assert_eq!((decisions[0].accepted, decisions[0].kept), (0, 1));
    assert_eq!((decisions[1].accepted, decisions[1].kept), (2, 3));
    assert!(decisions.iter().all(|d| d.failure == 0));
}
