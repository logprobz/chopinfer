use super::*;
use std::collections::BTreeSet;

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

/// Each peer copy's issuing, source and destination rank, in order.
fn copy_ranks(host: &Host, journal: &[String]) -> Vec<(u32, u32, u32)> {
    let ranks = host.device.stream_ranks.borrow();
    journal
        .iter()
        .filter(|l| l.starts_with("peer-copy "))
        .map(|l| {
            let words = l.split_whitespace().collect::<Vec<_>>();
            let number = |at: usize| words[at].parse::<u32>().unwrap();
            (ranks[&number(1)], number(2) - 1, number(3) - 1)
        })
        .collect()
}

/// Whether span `index` of `line` is only read: a peer copy's source, and
/// the reduced rows an HC expand reads. Every other span counts as written.
fn read_only(line: &str, index: usize) -> bool {
    (line.starts_with("peer-copy ") && index == 0)
        || (line.starts_with("launch ")
            && line.contains(" chop_sinkhorn_hc_deepseek41_expand_f32_v1 ")
            && index == 1)
}

fn intersects(a: Span, b: Span) -> bool {
    a.0 == b.0 && a.1 < b.1 + b.2 && b.1 < a.1 + a.2 && a.2 != 0 && b.2 != 0
}

/// Reconstruct GPU happens-before from submitted stream order and event
/// records/waits. Host submission order alone creates no cross-stream edge.
/// Check DMA reads and writes, including remote sources, against all compute
/// spans while retaining only histories subsumed by later ordered accesses.
/// Two reads never conflict.
fn audit(journal: &[String]) -> Result<usize, String> {
    audit_with(journal, false)
}

/// As `audit`, and also between transport accesses: each owner sum after
/// the copies into its inbox, each gather after the sum it reads.
fn audit_transport(journal: &[String]) -> Result<usize, String> {
    audit_with(journal, true)
}

fn audit_with(journal: &[String], transport_too: bool) -> Result<usize, String> {
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
    let mut history = BTreeMap::<u64, Vec<(Span, Vec<u64>, bool, bool)>>::new();
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
        for (index, span) in spans(line)
            .into_iter()
            .enumerate()
            .filter(|(_, s)| interesting.contains(&s.0))
        {
            let read = read_only(line, index);
            let previous = history.entry(span.0).or_default();
            for (other, clock, other_transport, other_read) in previous.iter() {
                if (transport != *other_transport
                    || (transport_too && transport && *other_transport))
                    && !(read && *other_read)
                    && intersects(span, *other)
                    && !ordered(clock, &clocks[stream])
                {
                    return Err(format!("unordered alias {span:?} and {other:?}: {line}"));
                }
            }
            // A later read never stands in for an earlier write.
            previous.retain(|(other, clock, kind, other_read)| {
                !(*kind == transport
                    && (!read || *other_read)
                    && span.1 <= other.1
                    && span.1 + span.2 >= other.1 + other.2
                    && ordered(clock, &clocks[stream]))
            });
            previous.push((span, clocks[stream].clone(), transport, read));
            accesses += usize::from(transport);
        }
    }
    Ok(accesses)
}

mod owner_host;

fn sparse_request(rows: u64) -> Vec<u8> {
    use runtime_bundle_sdk::prefill::{Boundary, encode};
    encode(
        &prompt(7, 0, 0, rows, true),
        &[Boundary {
            request_id: 7,
            prompt_tokens: rows,
            chunk_end: rows,
            chunk_tokens: 8192,
        }],
    )
    .unwrap()
}

#[test]
fn sparse_dma_owns_peer_buffers_and_drains_compute_before_free() {
    for fail_drain in [false, true] {
        let host = host();
        let observed = Rc::new(owner_host::OwnerHost::new(Rc::clone(&host)));
        let mut instance =
            skip::create_with_services(&host, HostServicesTableV1::new(Rc::clone(&observed)));
        let created = host.device.journal.borrow().clone();
        host.device.journal.borrow_mut().clear();
        instance
            .execute_prefill(&sparse_request(8192), &mut Views::default())
            .unwrap();
        instance.commit().unwrap();
        let prefill_end = host.device.journal.borrow().len();
        step(
            &mut instance,
            &decode(&[(7, 0, 8192, 9)]),
            &mut Views::default(),
        );
        let journal = host.device.journal.borrow().clone();
        let execution: BTreeSet<u32> = journal
            .iter()
            .filter(|line| line.starts_with("launch ") || line.starts_with("peer-copy "))
            .map(|line| line.split_whitespace().nth(1).unwrap().parse().unwrap())
            .collect();
        let peers: BTreeSet<u64> = journal[..prefill_end]
            .iter()
            .filter(|line| line.starts_with("peer-copy "))
            .flat_map(|line| {
                super::shared_overlap::journal_spans(line)
                    .into_iter()
                    .map(|span| span.0)
            })
            .collect();
        assert!(!peers.is_empty());
        let owners = observed.owners.borrow();
        let lifetime: BTreeSet<u32> = peers.iter().map(|buffer| owners[buffer].0).collect();
        assert_eq!(lifetime.len(), 4);
        assert!(
            lifetime.is_disjoint(&execution),
            "allocation ownership must not fence compute or DMA"
        );
        for stream in lifetime {
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
                    .any(|line| line == &format!("synchronize {stream}"))
            );
        }
        drop(owners);
        assert!(audit(&journal).unwrap() > 0);
        host.device.journal.borrow_mut().clear();
        if fail_drain {
            observed.fail_sync.set(execution.iter().next().copied());
            let released = host.released.get();
            assert!(instance.destroy().is_err());
            assert_eq!(host.released.get(), released);
            assert!(!observed.owners.borrow().is_empty());
            assert!(
                host.device
                    .journal
                    .borrow()
                    .iter()
                    .all(|line| !line.starts_with("free "))
            );
        } else {
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
                        .any(|line| line == &format!("synchronize {stream}"))
                );
            }
            assert!(observed.owners.borrow().is_empty());
            assert_eq!(host.acquired.get(), host.released.get());
        }
    }
}

#[test]
fn sparse_dma_copy_failure_poisoning_prevents_capture_or_retry() {
    let host = host();
    let mut instance =
        skip::create_with_services(&host, HostServicesTableV1::new(Rc::clone(&host)));
    host.device.journal.borrow_mut().clear();
    host.device.fail_copy.set(Some(2));
    let request = sparse_request(8192);
    assert_eq!(
        instance.execute_prefill(&request, &mut Views::default()),
        Err(BundleAbiError::Status(BundleStatusV1::HostError))
    );
    let submitted = host.device.journal.borrow().clone();
    assert_eq!(
        submitted
            .iter()
            .filter(|line| line.starts_with("peer-copy "))
            .count(),
        2
    );
    assert_eq!(
        instance.capture_prefix_state(7, 8192, &[200]),
        Err(BundleAbiError::Status(BundleStatusV1::InvalidState))
    );
    assert!(
        instance
            .execute_prefill(&request, &mut Views::default())
            .is_err()
    );
    assert!(instance.destroy().is_err());
    assert_eq!(*host.device.journal.borrow(), submitted);
}
