use super::*;
use crate::instance::execute::overlap::with_enabled;

fn capture(enabled: bool, rows: u32) -> (Rc<Host>, Vec<String>) {
    let host = FakeHost::new(Shape::new(&FLASH, Ranks::TP4).unwrap(), KV_LAYOUTS[0]);
    let mut instance = with_enabled(enabled, || create(&host, 4));
    host.device.journal.borrow_mut().clear();
    step(
        &mut instance,
        &prompt(7, 0, 0, rows.into(), true),
        &mut Views::default(),
    );
    let journal = host.device.journal.borrow().clone();
    instance.destroy().unwrap();
    assert_eq!(host.acquired.get(), host.released.get());
    (host, journal)
}

#[test]
fn prefill_overlap_preserves_mutation_order_and_joins_both_halves() {
    for rows in [1024, 1057, 8191, 8192] {
        let (_, serial) = capture(false, rows);
        let (host, parallel) = capture(true, rows);
        let mains = host.device.rank_streams.borrow();
        let mut streams = BTreeMap::<u32, usize>::new();
        let mut recorded = BTreeMap::new();
        let mut pending = BTreeMap::new();
        for line in &parallel {
            let words = line.split_whitespace().collect::<Vec<_>>();
            match words.first().copied() {
                Some("launch") if line.contains("chop_comm_reduce_scatter_f32_v1") => {
                    *streams.entry(words[1].parse().unwrap()).or_default() += 1;
                }
                Some("record") => {
                    let stream: u32 = words[1].parse().unwrap();
                    let event: u32 = words[2].parse().unwrap();
                    recorded.insert(event, stream);
                    if !mains.values().any(|&s| s == stream) {
                        pending.insert(event, stream);
                    }
                }
                Some("wait") => {
                    let stream: u32 = words[1].parse().unwrap();
                    let event: u32 = words[2].parse().unwrap();
                    assert!(recorded.contains_key(&event));
                    if let Some(side) = pending.remove(&event) {
                        assert_eq!(stream, mains[&host.device.stream_ranks.borrow()[&side]]);
                    }
                }
                _ => {}
            }
        }
        assert_eq!(streams.len(), 8, "four main and four reduction streams");
        for (stream, calls) in streams {
            assert_eq!(
                calls,
                if mains.values().any(|&s| s == stream) {
                    41
                } else {
                    80
                }
            );
        }
        assert!(pending.is_empty(), "every side operation rejoins main");
        // Compare every state-mutating launch, in order, on every rank.
        let mutations = |journal: &[String]| {
            journal
                .iter()
                .filter(|s| s.starts_with("launch "))
                .filter(|s| {
                    ["compress", "kv_write", "ring", "index", "window"]
                        .iter()
                        .any(|name| s.contains(name))
                })
                .map(|s| s.splitn(3, ' ').nth(2).unwrap().to_owned())
                .collect::<Vec<_>>()
        };
        assert!(!mutations(&serial).is_empty());
        assert_eq!(mutations(&serial), mutations(&parallel));
    }
}

#[test]
fn prefill_overlap_keeps_short_chunks_sequential() {
    let (_, serial) = capture(false, 1023);
    let (_, parallel) = capture(true, 1023);
    let launches = |journal: &[String]| {
        journal
            .iter()
            .filter(|s| s.starts_with("launch "))
            .map(|s| s.splitn(3, ' ').nth(2).unwrap().to_owned())
            .collect::<Vec<_>>()
    };
    assert_eq!(launches(&serial), launches(&parallel));
}

#[test]
fn prefill_overlap_journal_has_disjoint_concurrent_spans() {
    let (host, journal) = capture(true, 1057);
    let mains = host.device.rank_streams.borrow();
    let mut regions = 0;
    for (&rank, &main) in mains.iter() {
        let lines = journal
            .iter()
            .filter(|line| {
                let words = line.split_whitespace().collect::<Vec<_>>();
                matches!(words.first(), Some(&"launch" | &"record" | &"wait"))
                    && words[1]
                        .parse::<u32>()
                        .is_ok_and(|s| host.device.stream_ranks.borrow().get(&s) == Some(&rank))
            })
            .collect::<Vec<_>>();
        let records = lines
            .iter()
            .enumerate()
            .filter(|(_, line)| line.starts_with(&format!("record {main} ")))
            .collect::<Vec<_>>();
        assert_eq!(records.len(), 80);
        for pair in records.chunks_exact(2) {
            let (first, record) = pair[0];
            let end = pair[1].0;
            let event = record.split_whitespace().nth(2).unwrap();
            let wait = (end + 1..lines.len())
                .find(|&i| {
                    lines[i].starts_with("wait ") && lines[i].ends_with(&format!(" {event}"))
                })
                .unwrap();
            let side = lines[wait].split_whitespace().nth(1).unwrap();
            let done = (wait + 1..lines.len())
                .find(|&i| lines[i].starts_with(&format!("record {side} ")))
                .unwrap();
            let spans = |start, stop, stream: &str| {
                lines[start..stop]
                    .iter()
                    .filter(|line| line.starts_with(&format!("launch {stream} ")))
                    .flat_map(|line| super::shared_overlap::journal_spans(line))
                    .collect::<Vec<_>>()
            };
            let compute = spans(first + 1, end, &main.to_string());
            let comm = spans(wait + 1, done, side);
            assert!(!compute.is_empty() && !comm.is_empty());
            for a in &compute {
                for b in &comm {
                    assert!(
                        !(a.0 == b.0 && a.2 != 0 && b.2 != 0 && a.1 < b.1 + b.2 && b.1 < a.1 + a.2),
                        "unordered spans {a:?} {b:?}"
                    );
                }
            }
            regions += 1;
        }
    }
    assert_eq!(regions, 4 * 40);
}

#[test]
fn prefill_overlap_poisoning_stops_after_event_or_partial_rank_failure() {
    let (_, journal) = capture(true, 1057);
    let first_record = journal
        .iter()
        .position(|s| s.starts_with("record "))
        .unwrap();
    let first_wait = journal.iter().position(|s| s.starts_with("wait ")).unwrap();
    let side = journal[first_wait].split_whitespace().nth(1).unwrap();
    let first_side = journal
        .iter()
        .position(|s| s.starts_with(&format!("launch {side} ")))
        .unwrap();
    for (kind, at, marker) in [
        (0, first_record, "record "),
        (1, first_wait, "wait "),
        (2, first_side + 1, "launch "),
    ] {
        let host = FakeHost::new(Shape::new(&FLASH, Ranks::TP4).unwrap(), KV_LAYOUTS[0]);
        let mut instance = with_enabled(true, || create(&host, 4));
        let successes = journal[..at]
            .iter()
            .filter(|line| line.starts_with(marker))
            .count() as u32;
        let fault = match kind {
            0 => &host.device.fail_record,
            1 => &host.device.fail_wait,
            _ => &host.device.fail_launch,
        };
        fault.set(Some(successes));
        let mut views = Views::default();
        assert_eq!(
            instance.execute(&prompt(7, 0, 0, 1057, true), &mut views),
            Err(BundleAbiError::Status(BundleStatusV1::HostError))
        );
        assert_eq!(fault.get(), None);
        assert!(views.0.is_empty());
        let queued = host.device.journal.borrow().clone();
        assert_eq!(
            instance.execute(&prompt(8, 0, 0, 1, true), &mut views),
            Err(BundleAbiError::Status(BundleStatusV1::InvalidState))
        );
        assert_eq!(
            instance.destroy(),
            Err(BundleAbiError::Status(BundleStatusV1::InvalidState))
        );
        assert_eq!(*host.device.journal.borrow(), queued);
    }
}
