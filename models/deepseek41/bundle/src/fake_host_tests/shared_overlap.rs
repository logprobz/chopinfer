use super::*;
use crate::shared_overlap::{Settings, with_settings_in_tests};

fn captured(settings: Settings, ranks: Ranks) -> Vec<String> {
    let shape = Shape::new(&FLASH, ranks).unwrap();
    let host = FakeHost::new(shape, KV_LAYOUTS[0]);
    let mut instance = with_settings_in_tests(settings, || create(&host, 4));
    step(
        &mut instance,
        &prompt(7, 0, 0, 3, true),
        &mut Views::default(),
    );
    let from = host.device.journal.borrow().len();
    step(
        &mut instance,
        &decode(&[(7, 0, 3, 9)]),
        &mut Views::default(),
    );
    let lines = host.device.journal.borrow()[from..].to_vec();
    let count = lines
        .iter()
        .filter(|line| line.starts_with("launch "))
        .count();
    assert!(count > 100);
    let from = host.device.journal.borrow().len();
    step(
        &mut instance,
        &decode(&[(7, 0, 4, 9)]),
        &mut Views::default(),
    );
    assert!(
        !host.device.journal.borrow()[from..]
            .iter()
            .any(|s| s.starts_with("capture "))
    );
    instance.destroy().unwrap();
    assert_eq!(host.acquired.get(), host.released.get());
    lines
}

#[test]
fn shared_overlap_captures_identical_launches_and_reuses_the_graph() {
    for ranks in [Ranks::TP2, Ranks::TP4] {
        let serial = captured(Settings::OFF, ranks);
        let parallel = captured(Settings::ALL, ranks);
        let launches = |lines: &[String]| {
            let mut values = lines
                .iter()
                .filter_map(|s| s.strip_prefix("launch "))
                .map(|s| s.split_once(' ').unwrap().1.to_owned())
                .collect::<Vec<_>>();
            values.sort();
            values
        };
        assert_eq!(launches(&serial), launches(&parallel));
        let waits = |lines: &[String]| lines.iter().filter(|s| s.starts_with("wait ")).count();
        assert!(waits(&parallel) > waits(&serial));
        let streams = |lines: &[String]| {
            lines
                .iter()
                .filter_map(|s| s.strip_prefix("launch "))
                .map(|s| s.split_once(' ').unwrap().0.to_owned())
                .collect::<std::collections::BTreeSet<_>>()
        };
        let count = if ranks == Ranks::TP2 { 2 } else { 4 };
        assert_eq!(streams(&serial).len(), count);
        assert_eq!(streams(&parallel).len(), count * 2);
        assert_eq!(
            serial
                .iter()
                .any(|s| s.starts_with("launch ") && s.contains("_stream_merge_v1")),
            ranks == Ranks::TP4
        );
    }
}

#[test]
fn shared_overlap_closes_captures_after_fork_or_join_failure() {
    for fault in 0..4 {
        let shape = Shape::new(&FLASH, Ranks::TP2).unwrap();
        let host = FakeHost::new(shape, KV_LAYOUTS[0]);
        let mut instance = create(&host, 4);
        step(
            &mut instance,
            &prompt(7, 0, 0, 3, true),
            &mut Views::default(),
        );
        if fault < 2 {
            host.device.fail_record.set(Some(fault));
        } else {
            host.device.fail_wait.set(Some(fault - 2));
        }
        let mut views = Views::default();
        assert!(
            instance
                .execute(&decode(&[(7, 0, 3, 9)]), &mut views)
                .is_err()
        );
        assert_eq!(views.rows(), 0);
        instance.destroy().unwrap();
        assert_eq!(host.acquired.get(), host.released.get());
    }
}

#[test]
fn shared_overlap_replay_failure_poisoning_prevents_a_second_submission() {
    let shape = Shape::new(&FLASH, Ranks::TP2).unwrap();
    let host = FakeHost::new(shape, KV_LAYOUTS[0]);
    let mut instance = create(&host, 4);
    step(
        &mut instance,
        &prompt(7, 0, 0, 3, true),
        &mut Views::default(),
    );
    step(
        &mut instance,
        &decode(&[(7, 0, 3, 9)]),
        &mut Views::default(),
    );
    host.device.fail_graph_launch.set(Some(1));
    let batch = decode(&[(7, 0, 4, 9)]);
    assert!(instance.execute(&batch, &mut Views::default()).is_err());
    let before = host.device.journal.borrow().len();
    assert!(instance.execute(&batch, &mut Views::default()).is_err());
    assert_eq!(host.device.journal.borrow().len(), before);
    let released = host.released.get();
    assert!(instance.destroy().is_err());
    assert!(instance.is_poisoned());
    // A rank may still be blocked in a collective. Its resources stay held
    // rather than being released underneath the queued work.
    assert_eq!(host.released.get(), released);
    assert_eq!(host.device.journal.borrow().len(), before);
}

#[test]
fn shared_overlap_refuses_invalid_quantized_decode_rows() {
    for (settings, plane) in [Settings::OFF, Settings::ALL]
        .into_iter()
        .flat_map(|s| [(s, 0), (s, 1)])
    {
        let shape = Shape::new(&FLASH, Ranks::TP2).unwrap();
        let host = FakeHost::new(shape, KV_LAYOUTS[0]);
        let mut instance = with_settings_in_tests(settings, || create(&host, 4));
        step(
            &mut instance,
            &prompt(7, 0, 0, 3, true),
            &mut Views::default(),
        );
        host.injected.replace(Some(vec![1, 0, 0, 0]));
        host.injected_after
            .set(FLASH.shape.csa.index_sources.len() + 1 + plane);
        let mut views = Views::default();
        assert_eq!(
            instance
                .execute(&decode(&[(7, 0, 3, 9)]), &mut views)
                .unwrap_err(),
            BundleAbiError::Status(BundleStatusV1::BundleError)
        );
        assert_eq!(views.rows(), 0);
        assert!(host.injected.borrow().is_none());
        instance.destroy().unwrap();
        assert_eq!(host.acquired.get(), host.released.get());
    }
}

type Span = (u64, u64, u64);

fn journal_spans(line: &str) -> Vec<Span> {
    line.split("DeviceBufferSpanV1 { buffer: BundleDeviceBufferHandleV1 { raw: ")
        .skip(1)
        .map(|part| {
            let (handle, part) = part.split_once(" }, offset: ").unwrap();
            let (offset, part) = part.split_once(", length: ").unwrap();
            let length = part.split_once(" }").unwrap().0;
            (
                handle.parse().unwrap(),
                offset.parse().unwrap(),
                length.parse().unwrap(),
            )
        })
        .collect()
}

/// Audit the submitted journal independently of the production checker.
/// The quantizer before each fork identifies the three immutable inputs.
fn audit_scratch(lines: &[String]) -> (usize, usize) {
    let mut regions = 0;
    let mut collisions = 0;
    for at in 1..lines.len().saturating_sub(2) {
        let Some(record) = lines[at].strip_prefix("record ") else {
            continue;
        };
        let Some(wait) = lines[at + 1].strip_prefix("wait ") else {
            continue;
        };
        let (main, fork) = record.split_once(' ').unwrap();
        let (side, event) = wait.split_once(' ').unwrap();
        if fork != event {
            continue;
        }
        let (codes, scales, live) = if lines[at - 1].contains("moe_ceil_w5120_") {
            (1, 2, 3)
        } else if lines[at - 1].contains("chop_norm_quantize_deepseek41_input_v1") {
            // The MLP norm includes the widening output before codes/scales.
            (4, 5, 6)
        } else {
            continue;
        };
        let inputs = journal_spans(&lines[at - 1]);
        // Routed kernels bind the full input capacity even though the
        // quantizer binds only this bucket's rows of those same planes.
        let capacity = u64::from(crate::buffers::scratch_rows());
        let immutable = [
            (inputs[codes].0, inputs[codes].1, capacity * 5120),
            (inputs[scales].0, inputs[scales].1, capacity * 5120 / 32),
            inputs[live],
        ];
        let joined = (at + 2..lines.len())
            .find(|&i| lines[i].starts_with(&format!("record {side} ")))
            .unwrap();
        assert!(lines[joined + 1].starts_with(&format!("wait {main} ")));
        regions += 1;
        let branch = |stream: &str| {
            lines[at + 2..joined]
                .iter()
                .filter(|line| line.starts_with(&format!("launch {stream} ")))
                .flat_map(|line| journal_spans(line))
                .collect::<Vec<_>>()
        };
        let compute = branch(main);
        let shared = branch(side);
        assert!(!compute.is_empty() && !shared.is_empty());
        let contains = |a: Span, b: Span| a.0 == b.0 && a.1 <= b.1 && b.1 + b.2 <= a.1 + a.2;
        if shared.iter().any(|&a| {
            compute.iter().any(|&b| {
                a.0 == b.0
                    && a.2 != 0
                    && b.2 != 0
                    && a.1 < b.1 + b.2
                    && b.1 < a.1 + a.2
                    && !immutable
                        .iter()
                        .any(|&input| contains(input, a) && contains(input, b))
            })
        }) {
            collisions += 1;
        }
    }
    (regions, collisions)
}

#[test]
fn shared_overlap_journal_has_no_unordered_scratch_bindings() {
    for ranks in [Ranks::TP2, Ranks::TP4] {
        let shape = Shape::new(&FLASH, ranks).unwrap();
        let host = FakeHost::new(shape, KV_LAYOUTS[0]);
        let mut instance = with_settings_in_tests(Settings::ALL, || create(&host, 4));
        step(
            &mut instance,
            &prompt(7, 0, 0, 3, true),
            &mut Views::default(),
        );
        let from = host.device.journal.borrow().len();
        step(
            &mut instance,
            &decode(&[(7, 0, 3, 9)]),
            &mut Views::default(),
        );
        let lines = host.device.journal.borrow()[from..].to_vec();
        instance.destroy().unwrap();
        assert_eq!(host.acquired.get(), host.released.get());
        let (regions, collisions) = audit_scratch(&lines);
        assert_eq!(
            regions,
            FLASH.layers() as usize * if ranks == Ranks::TP2 { 2 } else { 4 }
        );
        assert_eq!(collisions, 0, "shared and routed scratch must be disjoint");
    }
    // An external frozen failing journal can additionally prove that this
    // same audit rejects the composed router and shared split projections.
    if let Some(path) = std::env::var_os("CHOPINFER_DEEPSEEK41_OVERLAP_NEGATIVE_JOURNAL") {
        let text = std::fs::read_to_string(path).unwrap();
        let lines = text.lines().map(str::to_owned).collect::<Vec<_>>();
        let (regions, collisions) = audit_scratch(&lines);
        assert_eq!(regions, FLASH.layers() as usize * 4);
        assert_eq!(
            collisions, regions,
            "negative journal did not reproduce the race"
        );
    }
}
