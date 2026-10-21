//! Audit only the collective replacement; all arithmetic and state work stays.
use super::*;
use crate::prefill_dma::with_enabled;

mod owner_host;
use owner_host::OwnerHost;

const SUM: &str = "chop_comm_owned_sum_round_f32_v1";
const OLD: &str = "chop_comm_allreduce_phased_round_f32_v1";
type Span = (u64, u64, u64);

fn host() -> Rc<Host> {
    let host = FakeHost::new((0, 0));
    host.device.ordinary_symbols.borrow_mut().insert(SUM.into());
    host
}

fn create_dma(host: &Rc<Host>, enabled: bool) -> (Rc<OwnerHost>, BundleInstanceV1) {
    let observer = Rc::new(OwnerHost::new(Rc::clone(host)));
    let instance = with_enabled(enabled, || {
        api().create(
            HostServicesTableV1::new(Rc::clone(&observer)),
            &crate::config::tests::record_naming(
                DECODE_BATCH,
                &options(2048),
                host.format,
                &[crate::projections::COMPONENT_ROLE.as_bytes().to_vec()],
            ),
        )
    })
    .unwrap_or_else(|e| panic!("{e:?}: {:?}", host.logs.borrow()));
    (observer, instance)
}

fn spans(line: &str) -> Vec<Span> {
    line.split("DeviceBufferSpanV1 { buffer: BundleDeviceBufferHandleV1 { raw: ")
        .skip(1)
        .map(|part| {
            let (buffer, part) = part.split_once(" }, offset: ").unwrap();
            let (offset, part) = part.split_once(", length: ").unwrap();
            (
                buffer.parse().unwrap(),
                offset.parse().unwrap(),
                part.split_once(" }").unwrap().0.parse().unwrap(),
            )
        })
        .collect()
}

fn intersects(a: Span, b: Span) -> bool {
    a.0 == b.0 && a.1 < b.1 + b.2 && b.1 < a.1 + a.2 && a.2 != 0 && b.2 != 0
}

/// Vector clocks model GPU stream order and event waits, never CPU issue
/// order. Copies read their source and write their destination; owner sums
/// read two inputs and write output. Other accesses are conservatively writes.
fn audit(lines: &[String]) -> Result<usize, String> {
    let relevant = |line: &&String| {
        matches!(
            line.split_whitespace().next(),
            Some("launch" | "peer-copy" | "record" | "wait")
        )
    };
    let streams = lines
        .iter()
        .filter(relevant)
        .map(|l| l.split_whitespace().nth(1).unwrap().parse::<u32>().unwrap())
        .collect::<BTreeSet<_>>();
    let streams = streams
        .into_iter()
        .enumerate()
        .map(|(i, s)| (s, i))
        .collect::<BTreeMap<_, _>>();
    let interesting = lines
        .iter()
        .filter(|l| l.starts_with("peer-copy ") || (l.starts_with("launch ") && l.contains(SUM)))
        .flat_map(|l| spans(l))
        .map(|s| s.0)
        .collect::<BTreeSet<_>>();
    let mut clocks = vec![vec![0_u64; streams.len()]; streams.len()];
    let mut events = BTreeMap::<u32, Vec<u64>>::new();
    let mut history = BTreeMap::<u64, Vec<(Span, Vec<u64>, bool)>>::new();
    let ordered = |a: &[u64], b: &[u64]| a.iter().zip(b).all(|(a, b)| a <= b);
    let mut accesses = 0;
    for line in lines.iter().filter(relevant) {
        let words = line.split_whitespace().collect::<Vec<_>>();
        let stream = streams[&words[1].parse::<u32>().unwrap()];
        if words[0] == "wait" {
            let event = words[2].parse::<u32>().unwrap();
            let before = events.get(&event).ok_or("wait before record")?;
            for (after, before) in clocks[stream].iter_mut().zip(before) {
                *after = (*after).max(*before);
            }
        }
        clocks[stream][stream] += 1;
        if words[0] == "record" {
            events.insert(words[2].parse().unwrap(), clocks[stream].clone());
        }
        let copied = words[0] == "peer-copy";
        let summed = words[0] == "launch" && line.contains(SUM);
        // Existing collectives synchronize their remote pointer tables
        // through their own protocol. Model their direct local arguments;
        // a pointer-table entry is not a whole-buffer write by this rank.
        let access_line = if words[0] == "launch" && line.contains("chop_comm_") && !summed {
            line.split("DevicePointerArray").next().unwrap()
        } else {
            line.as_str()
        };
        for (index, span) in spans(access_line)
            .into_iter()
            .enumerate()
            .filter(|(_, s)| interesting.contains(&s.0))
        {
            let write = if copied {
                index == 1
            } else if summed {
                index == 2
            } else {
                true
            };
            let previous = history.entry(span.0).or_default();
            for (other, clock, other_write) in previous.iter() {
                if (write || *other_write)
                    && intersects(span, *other)
                    && !ordered(clock, &clocks[stream])
                {
                    return Err(format!("unordered access {span:?} and {other:?}: {line}"));
                }
            }
            previous.retain(|(other, clock, _)| {
                !(span.1 <= other.1
                    && span.1 + span.2 >= other.1 + other.2
                    && ordered(clock, &clocks[stream]))
            });
            previous.push((span, clocks[stream].clone(), write));
            accesses += usize::from(copied || summed);
        }
    }
    Ok(accesses)
}

fn capture(rows: u64, enabled: bool) -> Vec<String> {
    let host = host();
    let (_observer, mut instance) = create_dma(&host, enabled);
    host.device.journal.borrow_mut().clear();
    instance
        .execute(&prompt_chunk(7, 0, rows, true), &mut NoRows::default())
        .unwrap_or_else(|e| panic!("{e:?}: {:?}", host.logs.borrow()));
    instance.commit().unwrap();
    let journal = host.device.journal.borrow().clone();
    instance.destroy().unwrap();
    assert_eq!(host.acquired.get(), host.released.get());
    journal
}

#[test]
fn dma_orders_odd_and_full_prefills_and_replaces_only_moe_collectives() {
    for rows in [512, 513, 8191, 8192] {
        let old = capture(rows, false);
        let new = capture(rows, true);
        let launches = |lines: &[String], skipped: &str| {
            lines
                .iter()
                .filter(|l| l.starts_with("launch ") && !l.contains(skipped))
                .map(|l| l.split_whitespace().nth(2).unwrap().to_owned())
                .collect::<Vec<_>>()
        };
        assert_eq!(launches(&old, OLD), launches(&new, SUM));
        assert_eq!(
            new.iter()
                .filter(|l| l.starts_with("launch ") && l.contains(SUM))
                .count(),
            42 * 4
        );
        assert!(
            !new.iter()
                .any(|l| l.starts_with("launch ") && l.contains(OLD))
        );
        let copies = new
            .iter()
            .filter(|l| l.starts_with("peer-copy "))
            .collect::<Vec<_>>();
        assert_eq!(copies.len(), 42 * 24);
        assert_eq!(
            copies.iter().map(|l| spans(l)[0].2).sum::<u64>(),
            42 * 18 * rows * 4096
        );
        assert_eq!(audit(&new).unwrap(), 42 * (24 * 2 + 4 * 3));
        let missing = new
            .into_iter()
            .filter(|l| !l.starts_with("wait "))
            .collect::<Vec<_>>();
        assert!(audit(&missing).is_err());
    }
}

#[test]
fn dma_keeps_short_prefills_and_decode_on_existing_entries() {
    let short = capture(511, true);
    assert!(
        !short
            .iter()
            .any(|l| l.starts_with("peer-copy ") || l.contains(SUM))
    );
    let host = host();
    let (_observer, mut instance) = create_dma(&host, true);
    instance
        .execute(&prompt_chunk(7, 0, 512, true), &mut NoRows::default())
        .unwrap();
    instance.commit().unwrap();
    host.device.journal.borrow_mut().clear();
    instance
        .execute(&step(&[(7, 0, 512, &[11])]), &mut NoRows::default())
        .unwrap();
    instance.commit().unwrap();
    assert!(
        !host
            .device
            .journal
            .borrow()
            .iter()
            .any(|l| l.starts_with("peer-copy ") || l.contains(SUM))
    );
    instance.destroy().unwrap();
}

#[test]
fn dma_partial_copy_failure_poisons_before_reuse_or_release() {
    let host = host();
    let (_observer, mut instance) = create_dma(&host, true);
    host.device.journal.borrow_mut().clear();
    host.fail_copy.set(Some(2));
    let batch = prompt_chunk(7, 0, 512, true);
    assert!(instance.execute(&batch, &mut NoRows::default()).is_err());
    let submitted = host.device.journal.borrow().clone();
    assert_eq!(
        submitted
            .iter()
            .filter(|l| l.starts_with("peer-copy "))
            .count(),
        2
    );
    assert!(instance.execute(&batch, &mut NoRows::default()).is_err());
    assert!(instance.destroy().is_err());
    assert_eq!(*host.device.journal.borrow(), submitted);
}

#[test]
fn dma_repeated_chunks_keep_transport_and_mtp_target_eligibility() {
    let host = host();
    let (_observer, mut instance) = create_dma(&host, true);
    host.device.journal.borrow_mut().clear();
    for (previous, last) in [(0, false), (512, false), (1024, true)] {
        instance
            .execute(
                &prompt_chunk(7, previous, 512, last),
                &mut NoRows::default(),
            )
            .unwrap();
        instance.commit().unwrap();
    }
    assert_eq!(
        host.device
            .journal
            .borrow()
            .iter()
            .filter(|l| l.starts_with("launch ") && l.contains(SUM))
            .count(),
        3 * 42 * 4
    );
    assert!(
        !host
            .device
            .journal
            .borrow()
            .iter()
            .any(|l| l.starts_with("capture ") || l.starts_with("replay "))
    );
    instance.destroy().unwrap();

    let host = FakeHost::with_format((3, 32), Shape::TP4.kv_formats()[1]);
    host.device.ordinary_symbols.borrow_mut().insert(SUM.into());
    let (_observer, mut instance) = create_dma(&host, true);
    host.device.journal.borrow_mut().clear();
    instance
        .execute(&prompt_chunk(7, 0, 512, true), &mut DraftSink::default())
        .unwrap();
    instance.commit().unwrap();
    assert_eq!(
        host.device
            .journal
            .borrow()
            .iter()
            .filter(|l| l.starts_with("launch ") && l.contains(SUM))
            .count(),
        42 * 4
    );
    instance.destroy().unwrap();
}

#[test]
fn dma_uses_private_owners_and_failed_drain_retains_buffers() {
    let host = host();
    let (observer, mut instance) = create_dma(&host, true);
    host.device.journal.borrow_mut().clear();
    host.launch_log.borrow_mut().clear();
    instance
        .execute(&prompt_chunk(7, 0, 512, true), &mut NoRows::default())
        .unwrap();
    instance.commit().unwrap();
    let active = host
        .launch_log
        .borrow()
        .iter()
        .map(|(s, _, _, _)| *s)
        .collect::<BTreeSet<_>>();
    let copied = host
        .device
        .journal
        .borrow()
        .iter()
        .filter(|l| l.starts_with("peer-copy "))
        .flat_map(|l| spans(l))
        .map(|s| s.0)
        .collect::<BTreeSet<_>>();
    let owners = observer.owners.borrow().clone();
    assert_eq!(copied.len(), 12);
    for buffer in &copied {
        assert!(!active.contains(&owners[buffer].0));
    }
    observer.fail_sync.set(active.first().copied());
    assert!(instance.destroy().is_err());
    assert_eq!(*observer.owners.borrow(), owners);
    // The safe instance wrapper refuses reuse after a failed destruction.
    assert!(instance.destroy().is_err());
    assert_eq!(*observer.owners.borrow(), owners);
}

#[test]
fn dma_journal_is_pinned() {
    let lines = capture(513, true);
    assert_eq!(audit(&lines).unwrap(), 42 * (24 * 2 + 4 * 3));
    check_journal(
        &lines,
        (12817, 0x773e1f61bb6db6cf),
        "glm5next-prefill-dma-513.txt",
    )
    .unwrap();
}
