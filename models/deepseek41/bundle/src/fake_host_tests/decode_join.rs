use super::*;
use crate::shared_overlap::{Settings, with_settings_in_tests};

/// The launch index of the next `launch` line after `at`.
fn next_launch(lines: &[String], at: usize) -> Option<usize> {
    (at + 1..lines.len()).find(|&i| lines[i].starts_with("launch "))
}

/// At the backbone AR decode buckets that serve the r2 router finish
/// (Decode-8 and Decode-32), every sublayer's shared expert joins directly
/// after that finish and before the routed gate, as at 32 verify rows:
/// the finish launch, the side stream's record, the main stream's wait,
/// then the gate small queue, 40 layers on each of four ranks.
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
    }
}
