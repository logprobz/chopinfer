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
        let parallel = captured(Settings::ALL, ranks);
        // Each launch with its config and operands. The routed gate after
        // an r2 join (directly after its wait) drops programmatic
        // serialization; restore it to compare.
        let launches = |lines: &[String]| {
            let mut joined = 0;
            let mut values = lines
                .iter()
                .enumerate()
                .filter_map(|(i, s)| Some((i, s.strip_prefix("launch ")?)))
                .map(|(i, s)| {
                    let launch = s.split_once(' ').unwrap().1.to_owned();
                    if i > 0 && lines[i - 1].starts_with("wait ") && launch.contains("flags: 0,") {
                        joined += 1;
                        launch.replacen("flags: 0,", "flags: 1,", 1)
                    } else {
                        launch
                    }
                })
                .collect::<Vec<_>>();
            values.sort();
            (values, joined)
        };
        let (_, joined) = launches(&parallel);
        let finishes = parallel
            .iter()
            .filter(|s| s.starts_with("launch ") && s.contains("_finish_r2v"))
            .count();
        assert_eq!(joined, finishes);
        let waits = |lines: &[String]| lines.iter().filter(|s| s.starts_with("wait ")).count();
        assert!(waits(&parallel) > 0);
        let streams = |lines: &[String]| {
            lines
                .iter()
                .filter_map(|s| s.strip_prefix("launch "))
                .map(|s| s.split_once(' ').unwrap().0.to_owned())
                .collect::<std::collections::BTreeSet<_>>()
        };
        let count = if ranks == Ranks::TP2 { 2 } else { 4 };
        assert_eq!(streams(&parallel).len(), count * 2);
        // TP4 decode runs the direct-load MoE buckets.
        assert_eq!(
            parallel
                .iter()
                .any(|s| s.starts_with("launch ") && s.contains("_moe_bandwidth_direct_")),
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
    for plane in [0, 1] {
        let shape = Shape::new(&FLASH, Ranks::TP2).unwrap();
        let host = FakeHost::new(shape, KV_LAYOUTS[0]);
        let mut instance = create(&host, 4);
        step(
            &mut instance,
            &prompt(7, 0, 0, 3, true),
            &mut Views::default(),
        );
        let sources = FLASH.shape.csa.index_sources.len();
        let mut status = vec![0_u8; (sources + 2 * 2) * 4];
        status[(sources + 1 + plane) * 4] = 1;
        host.injected.replace(Some(status));
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

pub(super) fn journal_spans(line: &str) -> Vec<Span> {
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

/// Fork/join regions of a journal and the regions whose concurrent
/// branches bind overlapping scratch.
pub(super) struct Regions {
    /// MoE sublayers: routed work beside the shared expert.
    pub moe: usize,
    /// Attention sublayers: the query chain beside the window KV projection.
    pub kv: usize,
    pub collisions: usize,
}

/// The MoE regions and the collisions of every region ([`audit_regions`]).
pub(super) fn audit_scratch(lines: &[String]) -> (usize, usize) {
    let regions = audit_regions(lines);
    (regions.moe, regions.collisions)
}

/// Audit the submitted journal independently of the production checker.
/// In an MoE region the quantizer before the fork identifies the three
/// immutable inputs. In an attention region (the compute branch ends in the
/// query RoPE) they are the spans both branches' first launches read that
/// a launch since the previous region binds: the layer's quantized input
/// and the live count.
pub(super) fn audit_regions(lines: &[String]) -> Regions {
    let mut regions = Regions {
        moe: 0,
        kv: 0,
        collisions: 0,
    };
    let contains = |a: Span, b: Span| a.0 == b.0 && a.1 <= b.1 && b.1 + b.2 <= a.1 + a.2;
    for at in 1..lines.len().saturating_sub(2) {
        let Some(record) = lines[at].strip_prefix("record ") else {
            continue;
        };
        let Some(wait) = lines[at + 1].strip_prefix("wait ") else {
            continue;
        };
        let (main, fork) = record.split_once(' ').unwrap();
        let (side, event) = wait.split_once(' ').unwrap();
        if fork != event || !lines[at + 2].starts_with("launch ") {
            continue;
        }
        let Some(joined) =
            (at + 2..lines.len()).find(|&i| lines[i].starts_with(&format!("record {side} ")))
        else {
            continue;
        };
        let launches = |stream: &str| {
            lines[at + 2..joined]
                .iter()
                .filter(|line| line.starts_with(&format!("launch {stream} ")))
                .collect::<Vec<_>>()
        };
        let producer = journal_spans(&lines[at - 1]);
        let immutable: Vec<Span> = if launches(main)
            .iter()
            .any(|line| line.contains("_query_rope"))
        {
            let (query, kv) = (launches(main), launches(side));
            let (query, kv) = (journal_spans(query[0]), journal_spans(kv[0]));
            // A graph piece can start at the fork, and HC launches can
            // follow the norm: the producer is any launch since the
            // previous region's join.
            let producer = lines[..at]
                .iter()
                .rev()
                .take_while(|line| !line.starts_with("wait "))
                .filter(|line| line.starts_with("launch "))
                .flat_map(|line| journal_spans(line))
                .collect::<Vec<_>>();
            regions.kv += 1;
            kv.into_iter()
                .filter(|s| query.contains(s) && producer.iter().any(|&p| contains(p, *s)))
                .collect()
        } else {
            let (codes, scales, live) = if lines[at - 1].contains("moe_ceil_w5120_") {
                (1, 2, 3)
            } else if lines[at - 1].contains("chop_deepseek41_hc_finish_norm_w32_v1")
                || lines[at - 1].contains("chop_deepseek41_hc_finish_twin_w")
                || lines[at - 1].contains("chop_deepseek41_hc_finish_split_w")
            {
                // MLP norms always widen; the fused ABI has ten HC arguments.
                // The two-CTA and split tails keep that ABI.
                (13, 14, 9)
            } else if [
                "chop_norm_quantize_deepseek41_input_v1",
                "chop_norm_quantize_deepseek41_input_tiles_v1",
                "chop_norm_quantize_deepseek41_input_cached32_v1",
            ]
            .iter()
            .any(|symbol| lines[at - 1].contains(symbol))
            {
                // The MLP norm includes the widening output before codes/scales.
                (4, 5, 6)
            } else {
                continue;
            };
            regions.moe += 1;
            // Routed kernels bind the full input capacity even though the
            // quantizer binds only this bucket's rows of those same planes.
            let capacity = u64::from(crate::buffers::scratch_rows());
            vec![
                (producer[codes].0, producer[codes].1, capacity * 5120),
                (producer[scales].0, producer[scales].1, capacity * 5120 / 32),
                producer[live],
            ]
        };
        assert!(lines[joined + 1].starts_with(&format!("wait {main} ")));
        let branch = |stream: &str| {
            launches(stream)
                .into_iter()
                .flat_map(|line| journal_spans(line))
                .collect::<Vec<_>>()
        };
        let compute = branch(main);
        let shared = branch(side);
        assert!(!compute.is_empty() && !shared.is_empty());
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
            regions.collisions += 1;
        }
    }
    regions
}

#[test]
fn shared_overlap_journal_has_no_unordered_scratch_bindings() {
    for ranks in [Ranks::TP1, Ranks::TP2, Ranks::TP4] {
        let shape = Shape::new(&FLASH, ranks).unwrap();
        for kv in [false, true] {
            let host = FakeHost::new(shape, KV_LAYOUTS[0]);
            let settings = Settings {
                kv,
                ..Settings::ALL
            };
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
            instance.destroy().unwrap();
            assert_eq!(host.acquired.get(), host.released.get());
            let regions = audit_regions(&lines);
            let layers = FLASH.layers() as usize * shape.ranks() as usize;
            // The window KV projection overlaps unless the C1 dual fold
            // computes it with the query.
            let dual = lines.iter().any(|l| l.contains("chop_mxfp8_dual_"));
            assert_eq!(regions.moe, layers);
            assert_eq!(regions.kv, if kv && !dual { layers } else { 0 });
            assert_eq!(regions.collisions, 0, "concurrent scratch must be disjoint");
        }
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

#[test]
fn scratch_audit_recognizes_norm_tiles_and_rejects_overlapping_writes() {
    let span = |handle, offset, length| {
        format!(
            "DeviceBufferSpanV1 {{ buffer: BundleDeviceBufferHandleV1 {{ raw: {handle} }}, offset: {offset}, length: {length} }}"
        )
    };
    for symbol in [
        "chop_norm_quantize_deepseek41_input_v1",
        "chop_norm_quantize_deepseek41_input_tiles_v1",
        "chop_norm_quantize_deepseek41_input_cached32_v1",
        "chop_deepseek41_hc_finish_norm_w32_v1",
    ] {
        let arguments = (1..=if symbol.contains("hc_finish_norm") {
            16
        } else {
            8
        })
            .map(|handle| span(handle, 0, 5120))
            .collect::<Vec<_>>()
            .join(", ");
        let mut lines = vec![
            format!("launch 1 {symbol} [{arguments}]"),
            "record 1 7".into(),
            "wait 2 7".into(),
            format!("launch 1 compute [{}]", span(19, 0, 64)),
            format!("launch 2 shared [{}]", span(20, 0, 64)),
            "record 2 8".into(),
            "wait 1 8".into(),
        ];
        assert_eq!(audit_scratch(&lines), (1, 0));
        lines[4] = format!("launch 2 shared [{}]", span(19, 32, 64));
        assert_eq!(audit_scratch(&lines), (1, 1));
    }
}

#[test]
fn the_scratch_audit_takes_attention_inputs_from_the_fork() {
    // Both projections read the input the launch before the fork binds; the
    // window KV projection may not write what the query chain reads.
    let span = |handle: u64, offset: u64, length: u64| {
        format!(
            "DeviceBufferSpanV1 {{ buffer: BundleDeviceBufferHandleV1 {{ raw: {handle} }}, offset: {offset}, length: {length} }}"
        )
    };
    let region = |kv_output: String| {
        let input = [span(3, 0, 5120), span(4, 0, 160), span(5, 0, 4)].join(", ");
        vec![
            format!("launch 1 norm [{input}]"),
            "record 1 7".into(),
            "wait 2 7".into(),
            format!("launch 1 query [{input}, {}]", span(6, 0, 1024)),
            format!("launch 2 kv [{input}, {kv_output}]"),
            format!(
                "launch 1 chop_deepseek41_latent_query_rope_v1 [{}]",
                span(6, 0, 1024)
            ),
            "record 2 8".into(),
            "wait 1 8".into(),
        ]
    };
    let audit = |lines: Vec<String>| {
        let r = audit_regions(&lines);
        (r.moe, r.kv, r.collisions)
    };
    assert_eq!(audit(region(span(7, 0, 2048))), (0, 1, 0));
    assert_eq!(audit(region(span(6, 512, 64))), (0, 1, 1));
}
