use super::*;
use crate::instance::execute::overlap::with_mode;
use crate::ops::with_dma;
use std::collections::BTreeSet;

mod bf16;
mod defaults;
mod skip;

const SUM: &str = "chop_comm_owned_sum_round_f32_v1";
const BF16_ROUND: &str = "chop_comm_owned_round_bf16_v1";
const BF16_SUM: &str = "chop_comm_owned_sum_bf16_v1";
type Span = (u64, u64, u64);

fn host() -> Rc<Host> {
    let host = FakeHost::new(Shape::new(&FLASH, Ranks::TP4).unwrap(), KV_LAYOUTS[0]);
    host.device
        .ordinary_symbols
        .borrow_mut()
        .extend([SUM, BF16_ROUND, BF16_SUM].map(String::from));
    host
}

/// A launch of the transport's owner sums. The BF16 rounding is checked as
/// compute: the copies that read its rows must be ordered after it.
fn transport_launch(line: &str) -> bool {
    line.starts_with("launch ") && [SUM, BF16_SUM].iter().any(|s| line.contains(s))
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
        .filter(|s| s.starts_with("peer-copy ") || transport_launch(s))
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
        let transport = words[0] == "peer-copy" || transport_launch(line);
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

mod owner_host;

#[test]
fn prefill_dma_owns_peer_buffers_away_from_compute_and_copy_streams() {
    use owner_host::OwnerHost;
    for mode in [0, 2] {
        let host = host();
        let observed = Rc::new(OwnerHost::new(Rc::clone(&host)));
        let record = crate::config::tests::record(&host.shape, host.kv, 4, false);
        let mut instance = with_dma(true, || {
            with_mode(mode, || {
                api()
                    .create(HostServicesTableV1::new(Rc::clone(&observed)), &record)
                    .unwrap()
            })
        });
        let created = host.device.journal.borrow().clone();
        host.device.journal.borrow_mut().clear();
        step(
            &mut instance,
            &prompt(7, 0, 0, 8191, true),
            &mut Views::default(),
        );
        step(
            &mut instance,
            &decode(&[(7, 0, 8191, 9)]),
            &mut Views::default(),
        );
        let journal = host.device.journal.borrow().clone();
        let execution = journal
            .iter()
            .filter(|line| line.starts_with("launch ") || line.starts_with("peer-copy "))
            .map(|line| {
                line.split_whitespace()
                    .nth(1)
                    .unwrap()
                    .parse::<u32>()
                    .unwrap()
            })
            .collect::<BTreeSet<_>>();
        let peers = journal
            .iter()
            .filter(|line| line.starts_with("peer-copy "))
            .flat_map(|line| {
                super::shared_overlap::journal_spans(line)
                    .into_iter()
                    .map(|span| span.0)
            })
            .collect::<BTreeSet<_>>();
        assert_eq!(peers.len(), 16);
        let owners = observed.owners.borrow();
        let lifetime = peers
            .iter()
            .map(|buffer| owners[buffer].0)
            .collect::<BTreeSet<_>>();
        assert_eq!(lifetime.len(), 4);
        assert!(
            lifetime.is_disjoint(&execution),
            "peer ownership must not fence compute or copy streams"
        );
        assert_eq!(
            peers.iter().map(|buffer| owners[buffer].1).sum::<u64>(),
            4 * 480 * 1024 * 1024
        );
        for &stream in &lifetime {
            let last = created
                .iter()
                .rposition(|line| {
                    line.starts_with("owned-allocate ")
                        && line.split_whitespace().nth(2) == Some(&stream.to_string())
                })
                .unwrap();
            assert!(
                created[last + 1..]
                    .iter()
                    .any(|line| line == &format!("synchronize {stream}")),
                "allocation-ready fence"
            );
        }
        drop(owners);
        assert!(audit(&journal).unwrap() > 0);
        assert!(
            audit(
                &journal
                    .iter()
                    .filter(|line| !line.starts_with("wait "))
                    .cloned()
                    .collect::<Vec<_>>()
            )
            .is_err()
        );
        host.device.journal.borrow_mut().clear();
        instance.destroy().unwrap();
        let released = host.device.journal.borrow();
        let first_free = released
            .iter()
            .position(|line| line.starts_with("free "))
            .unwrap();
        for stream in execution {
            assert!(
                released[..first_free]
                    .iter()
                    .any(|line| line == &format!("synchronize {stream}")),
                "drain final compute and DMA before free"
            );
        }
        assert!(observed.owners.borrow().is_empty());
        assert_eq!(host.acquired.get(), host.released.get());
    }
}

#[test]
fn prefill_dma_failed_teardown_retains_private_buffers() {
    let host = host();
    let observed = Rc::new(owner_host::OwnerHost::new(Rc::clone(&host)));
    let record = crate::config::tests::record(&host.shape, host.kv, 4, false);
    let mut instance = with_dma(true, || {
        with_mode(2, || {
            api()
                .create(HostServicesTableV1::new(Rc::clone(&observed)), &record)
                .unwrap()
        })
    });
    step(
        &mut instance,
        &prompt(7, 0, 0, 8191, true),
        &mut Views::default(),
    );
    let main = *host.device.rank_streams.borrow().get(&0).unwrap();
    host.device.journal.borrow_mut().clear();
    observed.fail_sync.set(Some(main));
    assert_eq!(
        instance.destroy(),
        Err(BundleAbiError::Status(BundleStatusV1::HostError))
    );
    let retained = host.device.journal.borrow().clone();
    assert!(!retained.iter().any(|line| line.starts_with("free ")));
    assert!(!observed.owners.borrow().is_empty());
    assert!(host.acquired.get() > host.released.get());
    // The SDK makes a failed destroy terminal and retains the resources.
    assert_eq!(instance.destroy(), Err(BundleAbiError::Poisoned));
    assert_eq!(*host.device.journal.borrow(), retained);
}
