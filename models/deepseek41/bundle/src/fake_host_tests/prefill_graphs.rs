//! Short text prefills split into graph segments
//! (`instance::execute::prefill_graphs`).
use super::indexer_workloads::skip_host;
use super::*;
use bundle_runtime::fake_host::RecordingSink;
use model_deepseek41_contract::execution_options::DIAGNOSTIC_STEP_LOGS;
use runtime_bundle_sdk::prefill::{Boundary, encode};

type Sink = RecordingSink<super::conformance::Limits>;

/// A whole prompt of `rows` tokens of `request` in `slot`, after `previous`
/// tokens, ending its prompt. Requests take slots of the four served items
/// in turn.
fn whole_prompt(request: u64, slot: u64, previous: u64, rows: u64) -> Vec<u8> {
    encode(
        &prompt(request, slot, previous, rows, true),
        &[Boundary {
            request_id: request,
            prompt_tokens: previous + rows,
            chunk_end: previous + rows,
            chunk_tokens: 8192,
        }],
    )
    .unwrap()
}

fn records(host: &Host) -> Vec<serde_json::Value> {
    host.logs
        .borrow()
        .iter()
        .filter(|line| line.contains("chopinfer.deepseek41.prefill-program.v1"))
        .map(|line| serde_json::from_str(line).unwrap())
        .collect()
}

/// With diagnostic step logs, every upper-skip prefill logs its program
/// record: an eligible one its split and keys, a longer one only its rows.
/// Without them, none is logged.
#[test]
fn diagnostic_logs_record_each_prefill_program() {
    for logs in [false, true] {
        let (host, mut instance) = skip_host(
            Ranks::TP4,
            false,
            true,
            4,
            &[(DIAGNOSTIC_STEP_LOGS, logs.into())],
        );
        for (request, rows) in [(1, 31), (2, 31), (3, 600)] {
            instance
                .execute_prefill(
                    &whole_prompt(request, request, 0, rows),
                    &mut Sink::default(),
                )
                .unwrap_or_else(|e| panic!("{e:?}: {:?}", host.logs.borrow()));
            instance.commit().unwrap();
            instance.retire_request(request).unwrap();
        }
        let records = records(&host);
        if !logs {
            assert!(records.is_empty());
        } else {
            assert_eq!(records.len(), 3, "{records:?}");
            let [first, second, long] = &records[..] else {
                unreachable!()
            };
            assert_eq!(first["rows"], 31);
            assert_eq!(first["eligible"], true);
            assert_eq!(first["segments"], 7);
            assert_eq!(first["handles"], 7);
            assert_eq!(first["joins"], 2);
            assert!(first["eager_launches"].as_u64().unwrap() > 0);
            assert!(first["graph_launches"].as_u64().unwrap() > 1000);
            // The same rows in another slot are the same program.
            assert_eq!(first["body"], second["body"]);
            assert_eq!(first["head"], second["head"]);
            assert_eq!(long["rows"], 600);
            assert_eq!(long["eligible"], false);
            assert!(long.get("body").is_none());
        }
        instance.destroy().unwrap();
        assert_eq!(host.acquired.get(), host.released.get());
    }
}

use model_deepseek41_contract::execution_options::PREFILL_GRAPH_PROGRAMS;

/// The journal lines from `from` on.
fn journal(host: &Host, from: usize) -> Vec<String> {
    host.device.journal.borrow()[from..].to_vec()
}

fn count(lines: &[String], prefix: &str) -> usize {
    lines.iter().filter(|line| line.starts_with(prefix)).count()
}

/// Each launch line, sorted: what a step issued, whatever the order.
fn launches(lines: &[String]) -> Vec<String> {
    let mut launches: Vec<_> = lines
        .iter()
        .filter(|line| line.starts_with("launch "))
        .cloned()
        .collect();
    launches.sort();
    launches
}

/// Decodes one token of `request` after `previous` tokens and commits it.
fn decode_one(host: &Host, instance: &mut BundleInstanceV1, request: u64, previous: u64) {
    instance
        .execute(
            &decode(&[(request, request % 4, previous, 5)]),
            &mut Views::default(),
        )
        .unwrap_or_else(|e| panic!("{e:?}: {:?}", host.logs.borrow()));
    instance.commit().unwrap();
}

/// Runs a whole prompt and commits it, returning its journal lines.
fn step(
    host: &Host,
    instance: &mut BundleInstanceV1,
    request: u64,
    previous: u64,
    rows: u64,
) -> Vec<String> {
    let from = host.device.journal.borrow().len();
    instance
        .execute_prefill(
            &whole_prompt(request, request % 4, previous, rows),
            &mut Sink::default(),
        )
        .unwrap_or_else(|e| panic!("{e:?}: {:?}", host.logs.borrow()));
    instance.commit().unwrap();
    journal(host, from)
}

/// A 64-row prompt of `request`, then the next turn: `rows` rows after
/// the prompt and, when `decoded`, one decoded token. Returns the turn's
/// journal lines. Prefix-zero prompts never replay as graphs, so only the
/// turn may.
fn turn(
    host: &Host,
    instance: &mut BundleInstanceV1,
    request: u64,
    rows: u64,
    decoded: bool,
) -> Vec<String> {
    let prompt = step(host, instance, request, 0, 64);
    assert_eq!(count(&prompt, "capture "), 0);
    assert_eq!(count(&prompt, "replay "), 0);
    if decoded {
        decode_one(host, instance, request, 64);
    }
    step(host, instance, request, 64 + u64::from(decoded), rows)
}

/// A short turn runs eagerly the first time its program is seen,
/// captures its body segments and head at the second and replays them
/// from then on, in another slot on other pages too. The capture records
/// exactly the launches the eager run issued, and a replay launches only
/// the eager selections between its segments.
#[test]
fn a_repeated_short_prefill_captures_once_and_replays_its_eager_launches() {
    let (host, mut instance) = skip_host(
        Ranks::TP4,
        false,
        true,
        4,
        &[(PREFILL_GRAPH_PROGRAMS, 4.into())],
    );
    let eager = turn(&host, &mut instance, 1, 31, true);
    assert_eq!(count(&eager, "capture "), 0);
    assert_eq!(count(&eager, "replay "), 0);
    let captured = turn(&host, &mut instance, 2, 31, true);
    // Seven body segments and the head on each of four ranks.
    assert_eq!(count(&captured, "capture "), 8 * 4);
    assert_eq!(count(&captured, "replay "), 8 * 4);
    assert_eq!(launches(&captured), launches(&eager));
    let replayed = turn(&host, &mut instance, 3, 31, true);
    // Diagnostic logs are off: no decision records.
    assert!(
        host.logs
            .borrow()
            .iter()
            .all(|line| !line.contains("prefill-graph.v1"))
    );
    assert_eq!(count(&replayed, "capture "), 0);
    assert_eq!(count(&replayed, "replay "), 8 * 4);
    let selections = launches(&replayed);
    assert!(!selections.is_empty());
    assert!(
        selections
            .iter()
            .all(|line| line.contains("_indexer_") || line.contains("candidate_block_ids")),
        "{selections:?}"
    );
    // The eager run's launches less the graphs' are the replay's.
    let mut rest = launches(&eager);
    for line in &selections {
        let at = rest
            .iter()
            .position(|l| l == line)
            .expect("an eager launch");
        rest.remove(at);
    }
    assert!(rest.len() > 1000);
    for request in 1..=3 {
        instance.retire_request(request).unwrap();
    }
    instance.destroy().unwrap();
    assert_eq!(host.acquired.get(), host.released.get());
}

/// Only continuations replay as graphs: repeated prefix-zero prompts, as
/// startup warmup runs them, and prefills of 512 rows or more stay eager,
/// and without the option nothing captures.
#[test]
fn prefill_graphs_capture_only_admitted_short_continuations() {
    for programs in [0, 4] {
        let (host, mut instance) = skip_host(
            Ranks::TP4,
            false,
            true,
            4,
            &[(PREFILL_GRAPH_PROGRAMS, programs.into())],
        );
        let mut captures = 0;
        for request in 1..=2 {
            captures += count(&turn(&host, &mut instance, request, 37, true), "capture ");
        }
        for request in 3..=4 {
            let lines = step(&host, &mut instance, request, 0, 600);
            assert_eq!(count(&lines, "capture ") + count(&lines, "replay "), 0);
        }
        // The 37-row turn captures its seven body segments and its head
        // at its second run.
        assert_eq!(captures, if programs == 0 { 0 } else { 8 * 4 });
        for request in 1..=4 {
            instance.retire_request(request).unwrap();
        }
        instance.destroy().unwrap();
        assert_eq!(host.acquired.get(), host.released.get());
    }
}

/// The capture log lines of the prefill stores: body programs and heads.
fn captures(host: &Host) -> (usize, usize) {
    let logs = host.logs.borrow();
    let of = |label: &str| logs.iter().filter(|line| line.contains(label)).count();
    (
        of("prefill graph capture: "),
        of("prefill-head graph capture: "),
    )
}

/// A DSpark target's short prefill captures its seven body segments and
/// head into the target's bounded ledger, and runs eagerly instead when
/// the capture would leave less than the decode graphs' headroom free.
#[test]
fn dspark_prefill_graphs_keep_the_ledger_headroom() {
    use model_deepseek41_contract::execution_options::DSPARK_GRAPH_HANDLES_PER_RANK;
    // At 130 handles per rank the one-graph head fits beside the 128
    // handles of headroom and the seven-graph body does not.
    for (handles, programs) in [(512, (1, 1)), (130, (0, 1))] {
        let (host, mut instance) = skip_host(
            Ranks::TP4,
            true,
            true,
            4,
            &[
                (PREFILL_GRAPH_PROGRAMS, 4.into()),
                (DSPARK_GRAPH_HANDLES_PER_RANK, handles.into()),
            ],
        );
        for request in 1..=3 {
            turn(&host, &mut instance, request, 31, false);
        }
        assert_eq!(captures(&host), programs, "{handles} handles");
        if programs.0 > 0 {
            let logs = host.logs.borrow();
            let line = logs
                .iter()
                .find(|line| line.contains("prefill graph capture: "))
                .unwrap();
            assert!(line.contains("31 rows, 7 graphs"), "{line}");
        }
        for request in 1..=3 {
            instance.retire_request(request).unwrap();
        }
        instance.destroy().unwrap();
        assert_eq!(host.acquired.get(), host.released.get());
    }
}

/// With diagnostic step logs, each graph decision is recorded with its
/// program's key: eager at the first run, then capture, then replay, for
/// the body and the head alike.
#[test]
fn diagnostic_logs_record_each_graph_decision() {
    let (host, mut instance) = skip_host(
        Ranks::TP4,
        false,
        true,
        4,
        &[
            (PREFILL_GRAPH_PROGRAMS, 4.into()),
            (DIAGNOSTIC_STEP_LOGS, true.into()),
        ],
    );
    for request in 1..=3 {
        turn(&host, &mut instance, request, 31, true);
    }
    let decisions: Vec<(String, String, String)> = host
        .logs
        .borrow()
        .iter()
        .filter(|line| line.contains("chopinfer.deepseek41.prefill-graph.v1"))
        .map(|line| {
            let record: serde_json::Value = serde_json::from_str(line).unwrap();
            let field = |name: &str| record[name].as_str().unwrap().to_owned();
            (field("program"), field("decision"), field("key"))
        })
        .collect();
    let expected = ["eager-firstrun", "capture", "replay"];
    for program in ["prefill", "prefill-head"] {
        let mine: Vec<_> = decisions.iter().filter(|d| d.0 == program).collect();
        assert_eq!(
            mine.iter().map(|d| d.1.as_str()).collect::<Vec<_>>(),
            expected,
            "{program}"
        );
        assert!(mine.iter().all(|d| d.2 == mine[0].2));
    }
    // The body key is the turns' program record digest; the prompts at
    // position zero are eligible programs that never replay.
    let programs = records(&host);
    assert_eq!(programs.len(), 6);
    let body = &decisions.iter().find(|d| d.0 == "prefill").unwrap().2;
    let turns: Vec<_> = programs.iter().filter(|r| r["start"] != 0).collect();
    assert_eq!(turns.len(), 3);
    assert!(turns.iter().all(|record| record["body"] == body.as_str()));
    for request in 1..=3 {
        instance.retire_request(request).unwrap();
    }
    instance.destroy().unwrap();
    assert_eq!(host.acquired.get(), host.released.get());
}

/// Three turns of an instance with graphs, `dspark` or not, each
/// request's 64-row prompt, a decoded token, then a 31-row continuation,
/// poisoning its scratch first when `poison`.
#[cfg(feature = "diagnostic-taps")]
struct PoisonedTurns {
    /// Each step's journal lines, in order.
    steps: Vec<Vec<String>>,
    /// Each step's recorded copies, uploads and reads.
    ops: Vec<Vec<bundle_runtime::fake_host::SnapshotOp>>,
    /// What arming recorded: the poison sources' uploads.
    armed: Vec<bundle_runtime::fake_host::SnapshotOp>,
    /// Prefills poisoned and DSpark steps left alone.
    counts: (u64, u64),
    /// Each rank's poison source and scratch fields.
    targets: crate::instance::PoisonTargets,
}

#[cfg(feature = "diagnostic-taps")]
fn poisoned_turns(dspark: bool, poison: bool) -> PoisonedTurns {
    let (host, mut instance) = skip_host(
        Ranks::TP4,
        dspark,
        true,
        4,
        &[(PREFILL_GRAPH_PROGRAMS, 4.into())],
    );
    let record = || *host.device.snapshot_ops.borrow_mut() = Some(Vec::new());
    let take = || host.device.snapshot_ops.borrow_mut().take().unwrap();
    record();
    let (armed, targets) = if poison {
        crate::instance::poison_prefill_scratch_for_test(instance.handle());
        (
            take(),
            crate::instance::poison_targets_for_test(instance.handle()),
        )
    } else {
        take();
        (Vec::new(), Vec::new())
    };
    let (mut steps, mut ops) = (Vec::new(), Vec::new());
    for request in 1..=3 {
        record();
        steps.push(step(&host, &mut instance, request, 0, 64));
        ops.push(take());
        decode_one(&host, &mut instance, request, 64);
        record();
        steps.push(step(&host, &mut instance, request, 65, 31));
        ops.push(take());
    }
    let counts = if poison {
        crate::instance::poison_prefill_scratch_for_test(instance.handle())
    } else {
        (0, 0)
    };
    for request in 1..=3 {
        instance.retire_request(request).unwrap();
    }
    instance.destroy().unwrap();
    assert_eq!(host.acquired.get(), host.released.get());
    PoisonedTurns {
        steps,
        ops,
        armed,
        counts,
        targets,
    }
}

/// A journal line without the graph ID of a capture, replay or release.
#[cfg(feature = "diagnostic-taps")]
fn without_graph_id(line: &str) -> String {
    let words: Vec<&str> = line.split(' ').collect();
    match words.as_slice() {
        ["captured", stream, _, nodes] => format!("captured {stream} {nodes}"),
        ["replay", stream, _] => format!("replay {stream}"),
        ["release", "graph", _] => "release graph".into(),
        _ => line.to_owned(),
    }
}

/// `spans` merged into disjoint byte ranges by buffer.
#[cfg(feature = "diagnostic-taps")]
fn merged(
    spans: impl IntoIterator<Item = runtime_bundle_abi::DeviceBufferSpanV1>,
) -> Vec<(u64, u64, u64)> {
    let mut ranges: Vec<_> = spans
        .into_iter()
        .map(|span| (span.buffer.raw, span.offset, span.offset + span.length))
        .collect();
    ranges.sort();
    let mut out: Vec<(u64, u64, u64)> = Vec::new();
    for (buffer, start, end) in ranges {
        match out.last_mut() {
            Some(last) if last.0 == buffer && start <= last.2 => last.2 = last.2.max(end),
            _ => out.push((buffer, start, end)),
        }
    }
    out
}

/// A diagnostic build's scratch poison fills, before each admitted short
/// prefill uploads, clears or gathers anything, every scratch field but
/// the text constants and immutable count table (including the head's
/// operands in the poisoned spans), from sources
/// of 0xff bytes, on every rank of an AR and of a DSpark instance; the
/// rest of each step's journal is unchanged, and prompts at position zero
/// are left alone.
#[cfg(feature = "diagnostic-taps")]
#[test]
fn scratch_poison_fills_every_per_step_scratch_field_before_each_admitted_prefill() {
    use bundle_runtime::fake_host::SnapshotOp;
    for dspark in [false, true] {
        let clean = poisoned_turns(dspark, false);
        let poisoned = poisoned_turns(dspark, true);
        assert_eq!(poisoned.counts, (3, 0), "dspark {dspark}");
        // Each source holds 0xff bytes.
        assert_eq!(poisoned.targets.len(), 4);
        for (source, _) in &poisoned.targets {
            let uploads: Vec<_> = poisoned
                .armed
                .iter()
                .filter_map(|op| match op {
                    SnapshotOp::Upload(_, span, bytes) if span == source => Some(bytes),
                    _ => None,
                })
                .collect();
            assert_eq!(uploads.len(), 1);
            assert_eq!(uploads[0].len() as u64, source.length);
            assert!(uploads[0].iter().all(|&byte| byte == 0xff));
        }
        for (_, fields) in &poisoned.targets {
            let filled = |name: &str| fields.iter().any(|f| f.0 == name && f.2);
            for name in [
                "normed_wide",
                "head_input",
                "head",
                "streams",
                "engram_codes",
                "row_status",
                "block_live",
                "sorted_rows",
                "workspace",
                "shared_workspace",
            ] {
                assert!(filled(name), "{name}");
            }
            for name in ["first_pre", "bias_select", "prefill_counts"] {
                assert!(fields.iter().any(|f| f.0 == name && !f.2), "{name}");
            }
            if dspark {
                for name in ["dspark.taps", "dspark.prefill_packed"] {
                    assert!(filled(name), "{name}");
                }
                for name in ["dspark.decode_packed", "dspark.feature_indices"] {
                    assert!(fields.iter().any(|f| f.0 == name && !f.2), "{name}");
                }
            }
        }
        for (index, (clean_lines, lines)) in clean.steps.iter().zip(&poisoned.steps).enumerate() {
            let copies = |source: runtime_bundle_abi::DeviceBufferSpanV1| -> Vec<_> {
                let owner = poisoned
                    .armed
                    .iter()
                    .find_map(|op| match op {
                        SnapshotOp::Upload(stream, span, _) if *span == source => Some(*stream),
                        _ => None,
                    })
                    .expect("source uploaded on its owning stream");
                poisoned.ops[index]
                    .iter()
                    .filter_map(|op| match op {
                        SnapshotOp::Copy(stream, from, to) if from.buffer == source.buffer => {
                            assert_eq!(*stream, owner);
                            assert_eq!(from.offset, source.offset);
                            assert_eq!(from.length, to.length);
                            assert!(from.length <= source.length);
                            Some(*to)
                        }
                        _ => None,
                    })
                    .collect()
            };
            if index % 2 == 0 {
                assert_eq!(clean_lines, lines, "dspark {dspark} prompt step {index}");
                assert!(
                    poisoned
                        .targets
                        .iter()
                        .all(|(source, _)| copies(*source).is_empty())
                );
                continue;
            }
            for (source, fields) in &poisoned.targets {
                let targets = copies(*source);
                let expected = fields.iter().filter(|f| f.2).map(|f| f.1);
                assert_eq!(
                    merged(targets.clone()),
                    merged(expected),
                    "dspark {dspark} step {index}"
                );
                let kept = merged(fields.iter().filter(|f| !f.2).map(|f| f.1));
                for (buffer, start, end) in merged(targets) {
                    assert!(
                        kept.iter()
                            .all(|k| k.0 != buffer || k.2 <= start || end <= k.1),
                        "a poison copy reaches a kept field"
                    );
                }
            }
            let at = clean_lines
                .iter()
                .zip(lines)
                .position(|(a, b)| a != b)
                .expect("poison copies");
            let added = lines.len() - clean_lines.len();
            assert!(
                lines[at..at + added]
                    .iter()
                    .all(|line| line.starts_with("copy "))
            );
            // The poison sources take four more handles, so graph IDs
            // shift; every other field is compared.
            let rest: Vec<_> = lines[at + added..]
                .iter()
                .map(|l| without_graph_id(l))
                .collect();
            let clean_rest: Vec<_> = clean_lines[at..]
                .iter()
                .map(|l| without_graph_id(l))
                .collect();
            let differs = rest.iter().zip(&clean_rest).position(|(a, b)| a != b);
            assert_eq!(
                rest.len(),
                clean_lines.len() - at,
                "dspark {dspark} step {index}"
            );
            assert!(
                differs.is_none(),
                "dspark {dspark} step {index}: {:?} instead of {:?}",
                differs.map(|d| &rest[d]),
                differs.map(|d| &clean_rest[d])
            );
            let first = |prefix: &str| clean_lines.iter().position(|line| line.starts_with(prefix));
            for prefix in ["upload ", "zero ", "gather ", "launch "] {
                assert!(
                    first(prefix).is_none_or(|line| at <= line),
                    "{prefix} step {index}"
                );
            }
        }
    }
}

/// A diagnostic build's program record places the step: its request's
/// slot and page table, which differ between the requests one graph
/// replays for.
#[cfg(feature = "diagnostic-taps")]
#[test]
fn diagnostic_program_records_place_each_step() {
    let (host, mut instance) = skip_host(
        Ranks::TP4,
        false,
        true,
        4,
        &[
            (PREFILL_GRAPH_PROGRAMS, 4.into()),
            (DIAGNOSTIC_STEP_LOGS, true.into()),
        ],
    );
    for request in 1..=3 {
        turn(&host, &mut instance, request, 31, true);
    }
    let turns: Vec<_> = records(&host)
        .into_iter()
        .filter(|record| record["start"] != 0)
        .collect();
    assert_eq!(turns.len(), 3);
    for (request, record) in (1..=3).zip(&turns) {
        assert_eq!(record["slot"], request % 4, "{record}");
        assert!(!record["pages"].as_array().unwrap().is_empty(), "{record}");
    }
    assert_ne!(turns[0]["pages"], turns[1]["pages"]);
    assert_ne!(turns[1]["pages"], turns[2]["pages"]);
    for request in 1..=3 {
        instance.retire_request(request).unwrap();
    }
    instance.destroy().unwrap();
    assert_eq!(host.acquired.get(), host.released.get());
}
