use super::*;
use crate::shared_overlap::{Settings, with_kv_reference, with_settings_in_tests};

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
        let (serial_launches, unjoined) = launches(&serial);
        let (parallel_launches, joined) = launches(&parallel);
        assert_eq!(serial_launches, parallel_launches);
        assert_eq!(unjoined, 0);
        let finishes = parallel
            .iter()
            .filter(|s| s.starts_with("launch ") && s.contains("_finish_r2v"))
            .count();
        assert_eq!(joined, finishes);
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
        // TP4 decode runs the direct-load MoE buckets under both settings.
        assert_eq!(
            serial
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
        if fork != event {
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
            {
                // MLP norms always widen; the fused ABI has ten HC arguments.
                // The two-CTA tail keeps that ABI.
                (13, 14, 9)
            } else if lines[at - 1].contains("_hc_completion_split_w") {
                // The one-launch boundary binds the pair's fourteen HC roles,
                // the workspace, then the norm's live count, weight, normed,
                // widened, codes and scales, status and the ticket.
                (19, 20, 15)
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

/// One lifecycle's journal from its first step, in phase cases: a 32-row
/// prompt, 31 one-row prompts, decode at 1, 8 and 32 rows and, on the
/// native layout, a 32-row verify. `reference` binds each window KV
/// projection as before the overlap.
fn kv_lifecycle(ranks: Ranks, kv: KvLayout, settings: Settings, reference: bool) -> Vec<String> {
    let run = || {
        let shape = Shape::new(&FLASH, ranks).unwrap();
        let host = FakeHost::new(shape, kv);
        if kv.storage == KvStorage::Native {
            host.speculation.set(Some((5, 32)));
        }
        let mut instance = with_settings_in_tests(settings, || create(&host, 32));
        let mut records = Vec::new();
        let mut ends = [1_u64; 32];
        let mark = |records: &mut Vec<String>, case: String, from: usize| {
            records.push(case);
            records.extend_from_slice(&host.device.journal.borrow()[from..]);
        };
        for slot in 0..32 {
            let count = if slot == 0 { 32 } else { 1 };
            let from = host.device.journal.borrow().len();
            step(
                &mut instance,
                &prompt(slot + 1, slot, 0, count, true),
                &mut Views::default(),
            );
            mark(&mut records, format!("case prefill{count}"), from);
            ends[slot as usize] = count;
        }
        for count in [1, 8, 32] {
            let from = host.device.journal.borrow().len();
            let items = (0..count)
                .map(|i| (i + 1, i, ends[i as usize], 7))
                .collect::<Vec<_>>();
            step(&mut instance, &decode(&items), &mut Views::default());
            mark(&mut records, format!("case decode{count}"), from);
            for end in &mut ends[..count as usize] {
                *end += 1;
            }
        }
        if kv.storage == KvStorage::Native {
            let from = host.device.journal.borrow().len();
            let tokens = vec![7; 4];
            let items = (0..8)
                .map(|i| (i + 1, i, ends[i as usize], tokens.as_slice()))
                .collect::<Vec<_>>();
            instance
                .execute(&verify(&items), &mut Views::default())
                .unwrap();
            instance
                .commit_accepted(&(1..=8).map(|i| kept(i, 4)).collect::<Vec<_>>())
                .unwrap();
            mark(&mut records, "case verify32".into(), from);
        }
        for id in 1..=32 {
            instance.retire_request(id).unwrap();
        }
        instance.destroy().unwrap();
        assert_eq!(host.acquired.get(), host.released.get());
        records
    };
    if reference {
        with_kv_reference(run)
    } else {
        run()
    }
}

/// A launch line without its stream and with each span reduced to its
/// length.
fn placement_free(line: &str) -> String {
    let rest = line
        .strip_prefix("launch ")
        .and_then(|s| s.split_once(' '))
        .map_or(line, |(_, rest)| rest);
    let mut parts = rest.split("DeviceBufferSpanV1 { buffer: BundleDeviceBufferHandleV1 { raw: ");
    let mut out = parts.next().unwrap().to_owned();
    for part in parts {
        out.push_str("span length: ");
        out.push_str(part.split_once(", length: ").unwrap().1);
    }
    out
}

/// Whether `line` launches a window KV projection kernel of `shape`, not
/// the C1 dual fold that computes it with the query.
fn kv_launch(shape: &Shape, line: &str) -> bool {
    let q = crate::ops::dense::projections(shape)
        .into_iter()
        .find(|d| d.id == "attention.kv")
        .unwrap()
        .query;
    line.starts_with("launch ")
        && line.contains("chop_")
        && !line.contains("chop_mxfp8_dual_")
        && line.contains(&format!("_n{}_k{}_g{}_", q.n, q.k, q.groups))
}

/// The window KV projection binds the shared expert's workspace and row
/// status in decode and verify phases, and, overlapped, runs beside the
/// query chain. Against the journal before the change, relocation alone
/// changes only those launches' spans; the overlap adds only fork and join
/// events and moves those launches to the side stream in captures. Every
/// other line, prefill included, is identical.
#[test]
fn the_window_kv_overlap_changes_only_its_projection() {
    let serial = Settings {
        kv: false,
        ..Settings::ALL
    };
    for (ranks, kv) in [
        (Ranks::TP4, KV_LAYOUTS[0]),
        (Ranks::TP4, KV_LAYOUTS[1]),
        (Ranks::TP2, KV_LAYOUTS[0]),
    ] {
        let shape = Shape::new(&FLASH, ranks).unwrap();
        let before = kv_lifecycle(ranks, kv, serial, true);
        let relocated = kv_lifecycle(ranks, kv, serial, false);
        let overlapped = kv_lifecycle(ranks, kv, Settings::ALL, false);
        // Relocation: equal lengths, only decode and verify KV launches differ.
        assert_eq!(before.len(), relocated.len());
        let mut case = String::new();
        let mut moved = 0;
        for (a, b) in before.iter().zip(&relocated) {
            if a.starts_with("case ") {
                assert_eq!(a, b);
                case = a.clone();
            }
            let decode = case.starts_with("case decode") || case.starts_with("case verify");
            if kv_launch(&shape, a) && decode {
                assert_ne!(a, b, "{case}: an unrelocated KV launch");
                assert_eq!(placement_free(a), placement_free(b), "{case}");
                moved += 1;
            } else {
                assert_eq!(a, b, "{case}");
            }
        }
        assert!(moved > 0);
        // Overlap: the same lines plus fork and join events; each decode
        // or verify KV launch moves, with the same operands, to the side
        // stream between a fork and its join. Graph pieces split only
        // between whole regions, so their boundaries may move.
        let event = |line: &str| {
            line.starts_with("record ")
                || line.starts_with("wait ")
                || line.starts_with("capture ")
                || line.starts_with("captured ")
        };
        let kv_of = |lines: &[String]| {
            let mut case = String::new();
            let (mut kept, mut moved) = (Vec::new(), Vec::new());
            for line in lines {
                if line.starts_with("case ") {
                    case = line.clone();
                }
                let decode = case.starts_with("case decode") || case.starts_with("case verify");
                if event(line) {
                    continue;
                } else if decode && kv_launch(&shape, line) {
                    moved.push(line.clone());
                } else {
                    kept.push(line.clone());
                }
            }
            (kept, moved)
        };
        let (kept, serial) = kv_of(&relocated);
        let (sided_kept, sided) = kv_of(&overlapped);
        if let Some(at) =
            (0..kept.len().max(sided_kept.len())).find(|&i| kept.get(i) != sided_kept.get(i))
        {
            panic!(
                "overlap changed line {at}: {:?} became {:?}",
                kept.get(at),
                sided_kept.get(at)
            );
        }
        assert_eq!(sided.len(), serial.len());
        for (a, b) in sided.iter().zip(&serial) {
            assert_eq!(placement_free(a), placement_free(b));
        }
        // Each KV region records and waits for a fork and a join.
        let fork_join = |lines: &[String]| {
            lines
                .iter()
                .filter(|l| l.starts_with("record ") || l.starts_with("wait "))
                .count()
        };
        let events = fork_join(&overlapped) - fork_join(&relocated);
        assert!(events > 0 && events % 4 == 0, "{events}");
        // Every moved launch sits on the side stream inside a region.
        let count = sided.len();
        let sided: std::collections::HashSet<&String> = sided.iter().collect();
        let split = |prefix: &str, line: &str| {
            line.strip_prefix(prefix)
                .and_then(|rest| rest.split_once(' '))
                .map(|(stream, event)| (stream.to_owned(), event.to_owned()))
        };
        let mut open: Option<String> = None;
        let mut inside = 0;
        for (i, line) in overlapped.iter().enumerate() {
            if let (Some((recorder, recorded)), Some((waiter, waited))) = (
                split("record ", line),
                overlapped.get(i + 1).and_then(|next| split("wait ", next)),
            ) && recorded == waited
            {
                open = match open {
                    // The side stream records the join.
                    Some(side) if side == recorder => None,
                    _ => Some(waiter),
                };
            }
            if sided.contains(line) {
                let side = open.as_ref().expect("a KV launch outside a region");
                assert!(line.starts_with(&format!("launch {side} ")), "{line}");
                inside += 1;
            }
        }
        assert!(open.is_none());
        assert_eq!(inside, count);
        assert!(inside > 0);
        // Each decode and verify case's regions bind disjoint scratch, with a
        // KV region per backbone layer and rank where the KV projection runs
        // on its own (the C1 dual fold computes it with the query at TP4).
        let mut cases = overlapped.split(|line| line.starts_with("case ")).skip(1);
        for name in overlapped.iter().filter(|line| line.starts_with("case ")) {
            let lines = cases.next().unwrap();
            let regions = audit_regions(lines);
            assert_eq!(regions.collisions, 0, "{name}");
            if name.starts_with("case decode") || name.starts_with("case verify") {
                let dual = lines.iter().any(|l| l.contains("chop_mxfp8_dual_"));
                let layers = (FLASH.layers() * shape.ranks()) as usize;
                if dual {
                    assert_eq!(regions.kv, 0, "{name}");
                } else {
                    assert!(regions.kv >= layers, "{name}: {}", regions.kv);
                }
            }
        }
    }
}
