use super::*;
use crate::instance::execute::overlap::with_mode;
use crate::ops::with_dma;
use std::collections::BTreeSet;

const SUM: &str = "chop_comm_owned_sum_round_f32_v1";
type Span = (u64, u64, u64);

fn host() -> Rc<Host> {
    let host = FakeHost::new(Shape::new(&FLASH, Ranks::TP4).unwrap(), KV_LAYOUTS[0]);
    host.device.ordinary_symbols.borrow_mut().insert(SUM.into());
    host
}

fn capture(mode: u8, rows: u64) -> Vec<String> {
    let host = host();
    let mut instance = with_dma(true, || with_mode(mode, || create(&host, 4)));
    host.device.journal.borrow_mut().clear();
    let batch = prompt(7, 0, 0, rows, true);
    instance
        .execute(&batch, &mut Views::default())
        .unwrap_or_else(|error| panic!("{error:?}: {:?}", host.logs.borrow()));
    instance.commit().unwrap();
    let journal = host.device.journal.borrow().clone();
    instance.destroy().unwrap();
    assert_eq!(host.acquired.get(), host.released.get());
    journal
}

fn intersects(a: Span, b: Span) -> bool {
    a.0 == b.0 && a.1 < b.1 + b.2 && b.1 < a.1 + a.2 && a.2 != 0 && b.2 != 0
}

/// Reconstruct GPU happens-before from submitted stream order and event
/// records/waits. Host submission order alone creates no cross-stream edge.
/// Check DMA reads and writes, including remote sources, against all compute
/// spans while retaining only histories subsumed by later ordered accesses.
fn audit(journal: &[String]) -> Result<usize, String> {
    let spans = super::shared_overlap::journal_spans;
    let relevant = |s: &&String| {
        matches!(
            s.split_whitespace().next(),
            Some("launch" | "peer-copy" | "record" | "wait")
        )
    };
    let streams = journal
        .iter()
        .filter(relevant)
        .map(|s| s.split_whitespace().nth(1).unwrap().parse::<u32>().unwrap())
        .collect::<BTreeSet<_>>();
    let streams = streams
        .into_iter()
        .enumerate()
        .map(|(i, stream)| (stream, i))
        .collect::<BTreeMap<_, _>>();
    let interesting = journal
        .iter()
        .filter(|s| s.starts_with("peer-copy ") || (s.starts_with("launch ") && s.contains(SUM)))
        .flat_map(|s| spans(s))
        .map(|s| s.0)
        .collect::<BTreeSet<_>>();
    let mut clocks = vec![vec![0_u64; streams.len()]; streams.len()];
    let mut events = BTreeMap::<u32, Vec<u64>>::new();
    let mut history = BTreeMap::<u64, Vec<(Span, Vec<u64>, bool)>>::new();
    let ordered = |a: &[u64], b: &[u64]| a.iter().zip(b).all(|(a, b)| a <= b);
    let mut accesses = 0;
    for line in journal.iter().filter(relevant) {
        let words = line.split_whitespace().collect::<Vec<_>>();
        let stream = streams[&words[1].parse::<u32>().unwrap()];
        if words[0] == "wait" {
            let event = words[2].parse::<u32>().unwrap();
            let before = events.get(&event).ok_or("wait precedes its event record")?;
            for (after, before) in clocks[stream].iter_mut().zip(before) {
                *after = (*after).max(*before);
            }
        }
        clocks[stream][stream] += 1;
        if words[0] == "record" {
            events.insert(words[2].parse().unwrap(), clocks[stream].clone());
        }
        let transport = words[0] == "peer-copy" || (words[0] == "launch" && line.contains(SUM));
        for span in spans(line)
            .into_iter()
            .filter(|s| interesting.contains(&s.0))
        {
            let previous = history.entry(span.0).or_default();
            for (other, clock, other_transport) in previous.iter() {
                if transport != *other_transport
                    && intersects(span, *other)
                    && !ordered(clock, &clocks[stream])
                {
                    return Err(format!(
                        "unordered compute/DMA alias {span:?} and {other:?}: {line}"
                    ));
                }
            }
            previous.retain(|(other, clock, kind)| {
                !(*kind == transport
                    && span.1 <= other.1
                    && span.1 + span.2 >= other.1 + other.2
                    && ordered(clock, &clocks[stream]))
            });
            previous.push((span, clocks[stream].clone(), transport));
            accesses += usize::from(transport);
        }
    }
    Ok(accesses)
}

#[test]
fn prefill_dma_preserves_all_owner_intervals_and_orders_every_transfer() {
    for (mode, rows, calls) in [(0, 8192, 81), (0, 8191, 81), (2, 8191, 162)] {
        let journal = capture(mode, rows);
        let copies = journal
            .iter()
            .filter(|s| s.starts_with("peer-copy "))
            .collect::<Vec<_>>();
        assert_eq!(copies.len(), calls * 24);
        assert_eq!(
            journal
                .iter()
                .filter(|s| s.starts_with("launch ") && s.contains(SUM))
                .count(),
            calls * 4
        );
        assert!(
            !journal
                .iter()
                .any(|s| s.starts_with("launch ") && s.contains("chop_comm_reduce_scatter_f32_v1"))
        );
        let bytes = copies
            .iter()
            .map(|s| {
                let spans = super::shared_overlap::journal_spans(s);
                assert_eq!(spans.len(), 2);
                assert_eq!(spans[0].2, spans[1].2);
                spans[0].2
            })
            .sum::<u64>();
        assert_eq!(bytes, 81 * 18 * rows * u64::from(FLASH.shape.hidden));
        assert_eq!(audit(&journal).unwrap(), calls * (24 * 2 + 4 * 3));
        let missing_waits = journal
            .into_iter()
            .filter(|s| !s.starts_with("wait "))
            .collect::<Vec<_>>();
        assert!(
            audit(&missing_waits).is_err(),
            "a removed producer/join wait must fail ordering"
        );
    }
}

#[test]
fn prefill_dma_keeps_short_prefills_and_decode_on_the_original_schedule() {
    let short = capture(0, 511);
    assert!(!short.iter().any(|s| s.starts_with("peer-copy ")));
    let host = host();
    let mut instance = with_dma(true, || create(&host, 4));
    step(
        &mut instance,
        &prompt(7, 0, 0, 8192, true),
        &mut Views::default(),
    );
    host.device.journal.borrow_mut().clear();
    step(
        &mut instance,
        &decode(&[(7, 0, 8192, 9)]),
        &mut Views::default(),
    );
    assert!(
        !host
            .device
            .journal
            .borrow()
            .iter()
            .any(|s| s.starts_with("peer-copy ") || s.contains(SUM))
    );
    instance.destroy().unwrap();
}

#[test]
fn prefill_dma_orders_a_small_part_that_uses_the_composed_transport() {
    let journal = capture(2, 1057);
    assert_eq!(
        journal
            .iter()
            .filter(|s| s.starts_with("launch ") && s.contains(SUM))
            .count(),
        81 * 4
    );
    assert_eq!(
        journal
            .iter()
            .filter(|s| s.starts_with("launch ") && s.contains("chop_comm_reduce_scatter_f32_v1"))
            .count(),
        81 * 4
    );
    assert!(audit(&journal).unwrap() > 0);
}

#[test]
fn prefill_dma_partial_copy_failure_poisoning_prevents_capture_or_retry() {
    for mode in [0, 2] {
        let host = host();
        let mut instance = with_dma(true, || with_mode(mode, || create(&host, 4)));
        host.device.fail_copy.set(Some(2));
        let batch = prompt(7, 0, 0, 8191, true);
        assert_eq!(
            instance.execute(&batch, &mut Views::default()),
            Err(BundleAbiError::Status(BundleStatusV1::HostError))
        );
        let submitted = host.device.journal.borrow().clone();
        assert_eq!(
            submitted
                .iter()
                .filter(|s| s.starts_with("peer-copy "))
                .count(),
            2
        );
        assert_eq!(
            instance.capture_prefix_state(7, 8191, &[200]),
            Err(BundleAbiError::Status(BundleStatusV1::InvalidState))
        );
        assert!(instance.execute(&batch, &mut Views::default()).is_err());
        assert!(instance.destroy().is_err());
        assert_eq!(*host.device.journal.borrow(), submitted);
    }
}
