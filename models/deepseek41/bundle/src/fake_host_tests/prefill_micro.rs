use super::*;
use crate::instance::execute::overlap::with_mode;

fn capture(rows: u64) -> (Rc<Host>, Vec<String>) {
    capture_mode(2, rows)
}

fn capture_mode(mode: u8, rows: u64) -> (Rc<Host>, Vec<String>) {
    let host = FakeHost::new(Shape::new(&FLASH, Ranks::TP4).unwrap(), KV_LAYOUTS[0]);
    let mut instance = with_mode(mode, || create(&host, 4));
    host.device.journal.borrow_mut().clear();
    step(
        &mut instance,
        &prompt(7, 0, 0, rows, true),
        &mut Views::default(),
    );
    let journal = host.device.journal.borrow().clone();
    instance.destroy().unwrap();
    assert_eq!(host.acquired.get(), host.released.get());
    (host, journal)
}

#[test]
fn prefill_micro_keeps_each_parts_selections_and_window_rows() {
    let (_, sequential) = capture_mode(0, 8192);
    let (host, parallel) = capture(8192);
    let stride = u64::from(FLASH.shape.indexer.top_k) * 4;
    let key_stride = u64::from(FLASH.shape.attention.window) * 4;
    let spans = super::shared_overlap::journal_spans;
    let is_attention = |line: &&String| {
        line.starts_with("launch ")
            && line.contains("chop_latent_sources_")
            && line.contains("_prefill")
    };
    for &main in host.device.rank_streams.borrow().values() {
        let on_main = |line: &&String| line.starts_with(&format!("launch {main} "));
        let full = sequential
            .iter()
            .filter(on_main)
            .filter(is_attention)
            .filter(|line| spans(line).iter().any(|s| s.2 == 8192 * stride))
            .collect::<Vec<_>>();
        let parts = parallel
            .iter()
            .filter(on_main)
            .filter(is_attention)
            .filter(|line| spans(line).iter().any(|s| s.2 == 4096 * stride))
            .collect::<Vec<_>>();
        assert_eq!(full.len(), 38);
        assert_eq!(parts.len(), full.len() * 2);
        let mut selected_planes = Vec::new();
        for (whole, pair) in full.iter().zip(parts.chunks_exact(2)) {
            let full_spans = spans(whole);
            for width in [stride, key_stride] {
                let plane = full_spans
                    .iter()
                    .filter(|s| s.2 == 8192 * width)
                    .collect::<Vec<_>>();
                assert_eq!(plane.len(), 1, "one retained row plane of width {width}");
                let &(buffer, base, _) = plane[0];
                for (part, launch) in pair.iter().enumerate() {
                    assert!(spans(launch).contains(&(
                        buffer,
                        base + part as u64 * 4096 * width,
                        4096 * width
                    )));
                }
                if width == stride {
                    selected_planes.push((buffer, base, 8192 * width));
                }
            }
        }
        selected_planes.sort_unstable();
        selected_planes.dedup();
        let mut marks = 0;
        let mut writes = [0, 0];
        for line in &parallel {
            if line.starts_with(&format!("record {main} ")) {
                marks += 1;
            }
            if !line.starts_with(&format!("launch {main} ")) || !line.contains("chop_position_") {
                continue;
            }
            for view in spans(line) {
                for &(buffer, base, length) in &selected_planes {
                    if view.0 != buffer || view.1 < base || view.1 >= base + length {
                        continue;
                    }
                    let part = marks % 2;
                    let first = base + part as u64 * 4096 * stride;
                    assert!(
                        view.1 >= first && view.1 + view.2 <= first + 4096 * stride,
                        "index writer crosses part {part}: {view:?}"
                    );
                    writes[part] += 1;
                }
            }
        }
        assert!(writes.iter().all(|&n| n > 0));
        assert_eq!(writes[0], writes[1]);
    }
}

#[test]
fn prefill_micro_long_context_odd_endpoint_captures_after_join() {
    let shape = Shape::new(&FLASH, Ranks::TP4).unwrap();
    let host = FakeHost::new(shape, KV_LAYOUTS[0]);
    host.kv_pages.set(1024);
    let mut instance = with_mode(2, || create(&host, 4));
    for offset in [0, 8192, 16384] {
        step(
            &mut instance,
            &prompt(7, 0, offset, 8192, false),
            &mut Views::default(),
        );
    }
    host.device.journal.borrow_mut().clear();
    let count = instance.describe_prefix_state().unwrap();
    // Snapshot pages stay beyond all 512 prompt pages and the restored
    // slot's shifted page table; they must not overwrite live KV.
    let pages = (0..count).map(|p| 600 + p).collect::<Vec<_>>();
    assert!(pages.last().is_some_and(|&p| p < host.kv_pages.get()));
    instance
        .execute(&prompt(7, 0, 24576, 8191, true), &mut Views::default())
        .unwrap();
    assert!(instance.capture_prefix_state(7, 32767, &pages).is_err());
    instance.commit().unwrap();
    let submitted = host.device.journal.borrow().clone();
    assert_eq!(
        submitted
            .iter()
            .filter(|s| s.starts_with("launch ") && s.contains("chop_comm_reduce_scatter_f32_v1"))
            .count(),
        4 * 162
    );
    instance.capture_prefix_state(7, 32767, &pages).unwrap();
    instance.retire_request(7).unwrap();
    instance
        .restore_prefix_state_in_slot(9, &(0..32767).collect::<Vec<_>>(), &pages, 2)
        .unwrap();
    assert_state_moved(&host, 4, 0, 0, 2);
    step(
        &mut instance,
        &decode(&[(9, 2, 32767, 5)]),
        &mut Views::default(),
    );
    instance.retire_request(9).unwrap();
    instance.destroy().unwrap();
    assert_eq!(host.acquired.get(), host.released.get());
}

#[test]
fn prefill_micro_poisoning_prevents_snapshot_or_retry_after_failure() {
    let (_, journal) = capture(1057);
    let record = journal
        .iter()
        .position(|s| s.starts_with("record "))
        .unwrap();
    let wait = journal.iter().position(|s| s.starts_with("wait ")).unwrap();
    let side = journal[wait].split_whitespace().nth(1).unwrap();
    let launch = journal
        .iter()
        .position(|s| s.starts_with(&format!("launch {side} ")))
        .unwrap();
    for (kind, at, marker) in [
        (0, record, "record "),
        (1, wait, "wait "),
        (2, launch + 1, "launch "),
    ] {
        let host = FakeHost::new(Shape::new(&FLASH, Ranks::TP4).unwrap(), KV_LAYOUTS[0]);
        let mut instance = with_mode(2, || create(&host, 4));
        let fault = match kind {
            0 => &host.device.fail_record,
            1 => &host.device.fail_wait,
            _ => &host.device.fail_launch,
        };
        fault.set(Some(
            journal[..at]
                .iter()
                .filter(|s| s.starts_with(marker))
                .count() as u32,
        ));
        let mut views = Views::default();
        assert_eq!(
            instance.execute(&prompt(7, 0, 0, 1057, true), &mut views),
            Err(BundleAbiError::Status(BundleStatusV1::HostError))
        );
        assert!(views.0.is_empty());
        assert_eq!(fault.get(), None);
        let queued = host.device.journal.borrow().clone();
        assert_eq!(
            instance.capture_prefix_state(7, 1057, &[40]),
            Err(BundleAbiError::Status(BundleStatusV1::InvalidState))
        );
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

#[test]
fn prefill_micro_reduces_both_parts_at_every_boundary() {
    for rows in [1024, 1057, 8191, 8192] {
        let (host, journal) = capture(rows);
        let mains = host.device.rank_streams.borrow();
        let mut counts = BTreeMap::<u32, usize>::new();
        for line in &journal {
            if line.starts_with("launch ") && line.contains("chop_comm_reduce_scatter_f32_v1") {
                let stream = line.split_whitespace().nth(1).unwrap().parse().unwrap();
                *counts.entry(stream).or_default() += 1;
                assert!(!mains.values().any(|&m| m == stream));
            }
        }
        assert_eq!(counts.len(), 4);
        assert!(counts.values().all(|&n| n == 162), "{counts:?}");
        let bytes = [
            ((rows as u32).div_ceil(2).div_ceil(64) * 64).to_le_bytes(),
            (rows as u32 - (rows as u32).div_ceil(2).div_ceil(64) * 64).to_le_bytes(),
        ]
        .concat();
        assert_eq!(
            host.uploads
                .borrow()
                .values()
                .filter(|v| **v == bytes)
                .count(),
            4
        );
    }
}

#[test]
fn prefill_micro_journal_preserves_cross_boundary_scratch_lifetimes() {
    let (host, journal) = capture(8191);
    audit_journal(&host, &journal, "chop_comm_reduce_scatter_f32_v1");
}

pub(super) fn audit_journal(host: &Host, journal: &[String], comm_symbol: &str) {
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
        let records = |stream: u32| {
            lines
                .iter()
                .enumerate()
                .filter(|(_, line)| line.starts_with(&format!("record {stream} ")))
                .map(|(i, _)| i)
                .collect::<Vec<_>>()
        };
        let side = lines
            .iter()
            .filter(|s| s.starts_with("launch ") && s.contains(comm_symbol))
            .map(|s| s.split_whitespace().nth(1).unwrap().parse::<u32>().unwrap())
            .next()
            .unwrap();
        let compute_marks = records(main);
        let side_marks = records(side);
        assert_eq!(compute_marks.len(), 162);
        assert_eq!(side_marks.len(), 162);
        let spans = |first, end, stream| {
            lines[first..end]
                .iter()
                .filter(|line| line.starts_with(&format!("launch {stream} ")))
                .flat_map(|line| super::shared_overlap::journal_spans(line))
                .collect::<Vec<_>>()
        };
        let disjoint = |a: &[(u64, u64, u64)], b: &[(u64, u64, u64)]| {
            assert!(!a.is_empty() && !b.is_empty());
            for a in a {
                for b in b {
                    assert!(
                        !(a.0 == b.0 && a.2 != 0 && b.2 != 0 && a.1 < b.1 + b.2 && b.1 < a.1 + a.2),
                        "unordered spans {a:?} {b:?}"
                    );
                }
            }
        };
        for block in 0..81 {
            let i = block * 2;
            let reduce_a = spans(
                if i == 0 { 0 } else { side_marks[i - 1] + 1 },
                side_marks[i],
                side,
            );
            let compute_b = spans(compute_marks[i] + 1, compute_marks[i + 1], main);
            disjoint(&reduce_a, &compute_b);
            regions += 1;
            if block > 0 {
                let reduce_b = spans(side_marks[i - 2] + 1, side_marks[i - 1], side);
                let compute_a = spans(compute_marks[i - 1] + 1, compute_marks[i], main);
                disjoint(&reduce_b, &compute_a);
                regions += 1;
            }
        }
        // Every wait refers to a previously submitted record, and the last
        // two main waits join both parts before the full-row epilogue.
        let mut recorded = BTreeMap::<u32, u32>::new();
        let mut joins = 0;
        for line in &lines {
            let words = line.split_whitespace().collect::<Vec<_>>();
            if words[0] == "record" {
                recorded.insert(words[2].parse().unwrap(), words[1].parse().unwrap());
            }
            if words[0] == "wait" {
                let event = words[2].parse().unwrap();
                assert!(recorded.contains_key(&event));
                if words[1].parse::<u32>().unwrap() == main {
                    assert_eq!(recorded[&event], side);
                    joins += 1;
                }
            }
        }
        assert_eq!(joins, 162);
    }
    assert_eq!(regions, 4 * (81 + 80));
}
