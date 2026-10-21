use super::*;
use crate::shared_overlap::{Settings, with_settings_in_tests};

/// The launch index of the next `launch` line after `at`.
fn next_launch(lines: &[String], at: usize) -> Option<usize> {
    (at + 1..lines.len()).find(|&i| lines[i].starts_with("launch "))
}

/// A journal launch line's stream, symbol, launch flags and device ranges
/// (buffer, offset, length).
fn launch(line: &str) -> (&str, &str, u32, Vec<(u64, u64, u64)>) {
    let mut words = line.splitn(4, ' ');
    let (_, stream, symbol, rest) = (
        words.next(),
        words.next().unwrap(),
        words.next().unwrap(),
        words.next().unwrap(),
    );
    let number = |text: &str, key: &str| -> u64 {
        let at = text.find(key).unwrap() + key.len();
        text[at..]
            .chars()
            .take_while(char::is_ascii_digit)
            .collect::<String>()
            .parse()
            .unwrap()
    };
    let spans = rest
        .split("DeviceBufferSpanV1 {")
        .skip(1)
        .map(|span| {
            (
                number(span, "raw: "),
                number(span, "offset: "),
                number(span, "length: "),
            )
        })
        .collect();
    (stream, symbol, number(rest, "flags: ") as u32, spans)
}

fn overlaps(a: (u64, u64, u64), b: (u64, u64, u64)) -> bool {
    a.0 == b.0 && a.1 < b.1 + b.2 && b.1 < a.1 + a.2
}

/// For the r2 finish launch at `at`: the gate's launch flags, and every
/// (routed symbol, side symbol) pair whose ranges overlap, from the gate
/// through the down finish against the shared expert between its fork and
/// join, apart from the fork's inputs both branches read (the shared
/// initialize's activation codes, scales and live-row count).
fn audit(lines: &[String], at: usize) -> (u32, Vec<(String, String)>) {
    let (main, ..) = launch(&lines[at]);
    let side = lines[at + 1].split(' ').nth(1).unwrap();
    let fork = (0..at)
        .rev()
        .find(|&i| lines[i].starts_with(&format!("wait {side} ")))
        .unwrap();
    let shared = (fork..at)
        .filter(|&i| lines[i].starts_with(&format!("launch {side} ")))
        .map(|i| launch(&lines[i]))
        .collect::<Vec<_>>();
    let init = shared
        .iter()
        .find(|l| l.1.contains("_pair_") && l.1.ends_with("_initialize_v3"))
        .unwrap();
    let inputs = [init.3[0], init.3[1], init.3[3]];
    let mut routed = Vec::new();
    for line in &lines[at + 3..] {
        if line.starts_with(&format!("launch {main} ")) {
            routed.push(launch(line));
            if routed.last().unwrap().1.contains("_down_finish_") {
                break;
            }
        }
    }
    assert!(routed.last().unwrap().1.contains("_down_finish_"));
    let mut shares = Vec::new();
    for r in &routed {
        for &a in &r.3 {
            for s in &shared {
                if s.3.iter().any(|&b| !inputs.contains(&b) && overlaps(a, b)) {
                    shares.push((r.1.to_string(), s.1.to_string()));
                }
            }
        }
    }
    shares.sort();
    shares.dedup();
    (routed[0].2, shares)
}

/// At the backbone AR decode buckets that serve the r2 router finish
/// (Decode-8 and Decode-32), every sublayer's shared expert joins directly
/// after that finish and before the routed gate, as at 32 verify rows:
/// the finish launch, the side stream's record, the main stream's wait,
/// then the gate small queue, 40 layers on each of four ranks. The gate
/// launches without programmatic serialization, and from it through the
/// down finish no routed launch shares a range with the shared expert
/// except the fork's inputs and the down finish's read of its output.
#[test]
fn ar_decode_joins_the_shared_expert_before_the_gate_after_the_r2_finish() {
    for items in [8u32, 32] {
        let shape = Shape::new(&FLASH, Ranks::TP4).unwrap();
        let host = FakeHost::new(shape, KV_LAYOUTS[0]);
        let mut instance = with_settings_in_tests(Settings::ALL, || create(&host, items));
        for slot in 0..u64::from(items) {
            step(
                &mut instance,
                &prompt(7 + slot, slot, 0, 3, true),
                &mut Views::default(),
            );
        }
        let from = host.device.journal.borrow().len();
        let batch = (0..u64::from(items))
            .map(|slot| (7 + slot, slot, 3, 9))
            .collect::<Vec<_>>();
        let mut views = Views::default();
        step(&mut instance, &decode(&batch), &mut views);
        assert_eq!(views.rows(), items);
        let lines = host.device.journal.borrow()[from..].to_vec();
        instance.destroy().unwrap();
        let finishes = lines
            .iter()
            .enumerate()
            .filter(|(_, l)| {
                l.starts_with("launch ") && l.contains("_router_finish_e384_k6_finish_r2v")
            })
            .map(|(i, _)| i)
            .collect::<Vec<_>>();
        let mut joined = 0;
        let mut after = Vec::new();
        let mut audited = std::collections::BTreeSet::new();
        for &at in &finishes {
            let main = lines[at].split(' ').nth(1).unwrap();
            let record = &lines[at + 1];
            let wait = &lines[at + 2];
            let gate = next_launch(&lines, at)
                .map(|i| lines[i].clone())
                .unwrap_or_default();
            if record.starts_with("record ")
                && !record.starts_with(&format!("record {main} "))
                && wait.starts_with(&format!("wait {main} "))
                && wait.split(' ').nth(2) == record.split(' ').nth(2)
                && next_launch(&lines, at) == Some(at + 3)
                && gate.contains("gate")
            {
                joined += 1;
                let (flags, shares) = audit(&lines, at);
                // The gate waits for the joined shared expert in full.
                assert_eq!(flags, 0, "{gate}");
                audited.extend(shares);
            } else {
                after.push(
                    lines[at + 1..(at + 4).min(lines.len())]
                        .iter()
                        .map(|l| l.chars().take(120).collect::<String>())
                        .collect::<Vec<_>>(),
                );
            }
        }
        eprintln!(
            "DECODE_JOIN items={items} launches={} r2_finishes={} joined_before_gate={joined}",
            lines.iter().filter(|l| l.starts_with("launch ")).count(),
            finishes.len()
        );
        for a in after.iter().take(2) {
            eprintln!("DECODE_JOIN items={items} not joined after finish: {a:#?}");
        }
        assert_eq!(finishes.len(), 40 * 4, "items={items}");
        assert_eq!(joined, finishes.len(), "items={items}");
        // Apart from the fork's inputs, only the down finish reads anything
        // of the shared expert's: its output, from the shared down kernel.
        eprintln!("DECODE_JOIN items={items} shared ranges: {audited:?}");
        assert!(!audited.is_empty(), "items={items}");
        assert!(
            audited
                .iter()
                .all(|(r, s)| r.contains("_down_finish_") && s.contains("_n5120_k576_")),
            "items={items}: {audited:?}"
        );
    }
}
