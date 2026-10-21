use super::*;
use runtime_bundle_abi::BundleDeviceBufferHandleV1;

fn spans(first: u32, bytes: u64) -> Vec<Span> {
    (first..first + 4)
        .map(|slot| Span::new(BundleDeviceBufferHandleV1::from_parts(1, slot), 256, bytes))
        .collect()
}

#[test]
fn uneven_rows_scatter_each_peer_once_and_gather_every_owner() {
    for rows in [512_u32, 513, 1023, 2048, 8191, 8192] {
        let width = 16;
        let elements = u64::from(rows) * u64::from(width);
        let inputs = spans(1, elements * 4);
        let outputs = spans(5, elements * 2);
        let inboxes = spans(9, u64::from(rows.div_ceil(4)) * u64::from(width) * 16);
        let call = Call::layout(&inboxes, &inputs, &outputs, rows, width).unwrap();
        assert_eq!(call.owners.iter().map(|o| o.rows).sum::<u32>(), rows);
        for destination in 0..4 {
            let owner = call.owners[destination];
            let scatter: Vec<_> = call
                .scatter
                .iter()
                .flatten()
                .filter(|t| t.destination == destination)
                .collect();
            let mut sources = scatter.iter().map(|t| t.source).collect::<Vec<_>>();
            sources.sort_unstable();
            assert_eq!(
                sources,
                (0..4).filter(|&r| r != destination).collect::<Vec<_>>()
            );
            for t in scatter {
                assert_eq!(t.input.offset, owner.input.offset);
                assert_eq!(t.input.length, owner.input.length);
                assert_eq!(
                    t.output.offset,
                    inboxes[destination].offset + t.source as u64 * owner.slot_elements * 4
                );
                assert_eq!(t.input.length, t.output.length);
            }
            let mut intervals = vec![(owner.output.offset, owner.output.length)];
            for t in call
                .gather
                .iter()
                .flatten()
                .filter(|t| t.destination == destination)
            {
                assert_eq!(t.input, call.owners[t.source].output);
                assert_eq!(t.output.buffer, outputs[destination].buffer);
                intervals.push((t.output.offset, t.output.length));
            }
            intervals.sort_unstable();
            let mut frontier = outputs[destination].offset;
            for (offset, length) in intervals {
                assert_eq!(offset, frontier);
                frontier += length;
            }
            assert_eq!(
                frontier,
                outputs[destination].offset + outputs[destination].length
            );
        }
        let moved = call
            .scatter
            .iter()
            .chain(&call.gather)
            .flatten()
            .map(|t| t.input.length)
            .sum::<u64>();
        assert_eq!(moved, elements * 18);
    }
}

#[test]
fn truncated_misaligned_and_overflowing_views_are_refused() {
    let input = spans(1, 512 * 16 * 4);
    let output = spans(5, 512 * 16 * 2);
    let inbox = spans(9, 512 * 16 * 4);
    for bytes in [0, 512 * 16 * 4 - 1] {
        let mut bad = inbox.clone();
        bad[0].length = bytes;
        assert!(Call::layout(&bad, &input, &output, 512, 16).is_err());
    }
    let mut bad = input.clone();
    bad[2].offset += 2;
    assert!(Call::layout(&inbox, &bad, &output, 512, 16).is_err());
    assert!(Call::layout(&inbox, &input, &output, 512, 15).is_err());
    assert!(Call::layout(&inbox, &input, &output, 0, 16).is_err());
    let mut overflow = input[0];
    overflow.offset = u64::MAX - 1;
    assert!(subspan(overflow, 4, 4).is_err());
    assert!(subspan(input[0], u64::MAX, 4).is_err());
}
