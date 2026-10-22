//! One target graph serves every item count of its rows. A graph instance
//! and an eager instance run the same schedule: rows held while the live
//! items rise and fall, the rows split differently at the same items,
//! requests reordered across physical slots and pages, and a slot reused
//! by a new request. Every stream must receive the same work in both,
//! each replay expanded to the launches its capture recorded: the same
//! uploads (by content), copies, zeroes, launches with their arguments,
//! and event records and waits. So no replay reads metadata a step left
//! stale or bakes a live count, and the drafts and committed contexts
//! agree.
use std::collections::BTreeMap;

use super::*;

/// The page of `request`'s `page`: each request owns its own pages, so a
/// reused slot changes pages too.
fn page(request: u64, page: u64) -> u64 {
    request * 4 + page
}

fn admit(instance: &mut BundleInstanceV1, host: &Host, request: u64, slot: u64) {
    let batch = BatchRequestV1::new(
        BatchWireVersionV1::V3,
        vec![BatchItemV1::new(request, false, 3, true, vec![1, 2, 3])],
        vec![BatchAppendV1::new(0, 3, 3, vec![page(request, 0)], None)],
    )
    .with_state_slots(vec![slot])
    .encode()
    .unwrap();
    execute(instance, host, &batch);
    instance.commit().unwrap();
}

/// One verify step of (request, slot, drafts) items, every item keeping
/// one row. Returns the drafts it produced.
fn step(
    instance: &mut BundleInstanceV1,
    host: &Host,
    items: &[(u64, u64, u32)],
) -> Vec<(u64, Vec<u32>)> {
    let mut batch = Vec::new();
    let mut appends = Vec::new();
    let mut kept_rows = Vec::new();
    for &(request, _, drafts) in items {
        let start = state(instance, request).end;
        let rows = u64::from(drafts + 1);
        batch.push(if drafts == 0 {
            BatchItemV1::new(request, true, 1, true, vec![7])
        } else {
            kept_rows.push(kept(request, 1));
            BatchItemV1::verify(request, drafts + 1, true, vec![7])
        });
        appends.push(BatchAppendV1::new(
            start,
            rows,
            start + rows,
            (0..(start + rows).div_ceil(64))
                .map(|p| page(request, p))
                .collect(),
            None,
        ));
    }
    let batch = BatchRequestV1::new(BatchWireVersionV1::V3, batch, appends)
        .with_state_slots(items.iter().map(|&(_, slot, _)| slot).collect())
        .encode()
        .unwrap();
    let sink = execute(instance, host, &batch);
    instance.commit_accepted(&kept_rows).unwrap();
    sink.draft_tokens()
}

/// Items `requests` in order, each in slot `request - 1`, with `drafts`.
fn items(requests: impl IntoIterator<Item = u64>, drafts: &[u32]) -> Vec<(u64, u64, u32)> {
    requests
        .into_iter()
        .zip(drafts)
        .map(|(request, &drafts)| (request, request - 1, drafts))
        .collect()
}

/// A step's (request, slot, drafts) items, after the requests it replaces
/// first as (retired, admitted, slot).
type Step = (Vec<(u64, u64, u32)>, Vec<(u64, u64, u64)>);

/// The schedule of steps.
fn schedule() -> Vec<Step> {
    let d = |depth: u32, count: usize| vec![depth; count];
    let cat = |a: Vec<u32>, b: Vec<u32>| a.into_iter().chain(b).collect::<Vec<_>>();
    let mut steps = vec![
        // 24 rows over 4, 6, 8, 12 and 20 items, then back down.
        (items(1..=4, &d(5, 4)), vec![]),
        (items(1..=6, &d(3, 6)), vec![]),
        (items(1..=8, &d(2, 8)), vec![]),
        (items(1..=12, &d(1, 12)), vec![]),
        (items(1..=20, &cat(d(1, 4), d(0, 16))), vec![]),
        (items(1..=12, &d(1, 12)), vec![]),
        (items(1..=8, &cat(d(0, 4), d(4, 4))), vec![]),
        (items(1..=6, &d(3, 6)), vec![]),
        (items(1..=4, &d(5, 4)), vec![]),
        // Eight items again, split another way.
        (items(1..=8, &cat(d(1, 4), d(3, 4))), vec![]),
        // Other requests in the last slots, in reverse order.
        (items((25..=32).rev(), &d(2, 8)), vec![]),
        // 32 rows over 8, 16 and 10 items; 12 rows over 2, 3 and 11.
        (items(9..=16, &d(3, 8)), vec![]),
        (items(1..=16, &d(1, 16)), vec![]),
        (items(17..=26, &cat(d(5, 4), cat(d(1, 2), d(0, 4)))), vec![]),
        (items(1..=2, &d(5, 2)), vec![]),
        (items(3..=5, &d(3, 3)), vec![]),
        (items(20..=30, &cat(d(1, 1), d(0, 10))), vec![]),
    ];
    // New requests in the first four slots, with their own pages, beside
    // older ones: 24 rows over 8 items, then 12 rows over 3.
    let reused = [(41, 0, 5), (42, 1, 5), (43, 2, 5), (44, 3, 5)];
    let mut late = vec![(44, 3, 2), (43, 2, 2), (42, 1, 2), (41, 0, 2)];
    late.extend(items(5..=8, &d(2, 4)));
    steps.push((late, (1..=4).map(|old| (old, old + 40, old - 1)).collect()));
    steps.push((reused[..2].to_vec(), vec![]));
    steps.push((vec![(43, 2, 3), (9, 8, 3), (10, 9, 3)], vec![]));
    steps
}

/// Each stream's work in journal order: a replay is the work its capture
/// recorded, and a capture's own recording is not work. Also returns the
/// rank stream that forks each side lane: the stream that recorded the
/// event the lane first waits for.
fn executed(journal: &[String]) -> (BTreeMap<String, Vec<String>>, BTreeMap<String, String>) {
    let mut graphs: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let mut streams: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let mut recorded: BTreeMap<String, String> = BTreeMap::new();
    let mut lanes: BTreeMap<String, String> = BTreeMap::new();
    let mut capture: Option<Vec<String>> = None;
    let mut push = |line: &String| {
        let fields: Vec<&str> = line.split_whitespace().collect();
        if !matches!(
            fields[0],
            "launch" | "record" | "wait" | "zero" | "copy" | "upload" | "download" | "synchronize"
        ) {
            return;
        }
        let stream = fields[1].to_owned();
        if fields[0] == "record" {
            recorded.insert(fields[2].to_owned(), stream.clone());
        }
        if fields[0] == "wait" && !streams.contains_key(&stream) {
            let by = &recorded[fields[2]];
            let by = lanes.get(by).unwrap_or(by).clone();
            lanes.insert(stream.clone(), by);
        }
        streams.entry(stream).or_default().push(line.clone());
    };
    for line in journal {
        if line.starts_with("capture ") {
            assert!(capture.replace(Vec::new()).is_none());
        } else if let Some(rest) = line.strip_prefix("captured ") {
            let graph = rest.split_whitespace().nth(1).unwrap().to_owned();
            graphs.insert(graph, capture.take().unwrap());
        } else if let Some(block) = capture.as_mut() {
            block.push(line.clone());
        } else if let Some(rest) = line.strip_prefix("replay ") {
            let graph = rest.split_whitespace().nth(1).unwrap();
            for line in &graphs[graph] {
                push(line);
            }
        } else {
            push(line);
        }
    }
    (streams, lanes)
}

/// `work` without its event records and waits.
fn computed(work: &[String]) -> Vec<&String> {
    work.iter()
        .filter(|line| !line.starts_with("record ") && !line.starts_with("wait "))
        .collect()
}

/// The first difference of two streams' work, for a failure.
fn compare(stream: &str, a: &[impl AsRef<str>], b: &[impl AsRef<str>]) {
    let at = a.iter().zip(b).position(|(x, y)| x.as_ref() != y.as_ref());
    assert!(
        a.len() == b.len() && at.is_none(),
        "stream {stream}: {} and {} operations, first difference at {at:?}:\n{:?}\n{:?}",
        a.len(),
        b.len(),
        at.map(|at| a[at].as_ref()),
        at.map(|at| b[at].as_ref()),
    );
}

/// The eager rank stream's work is its graph stream's work and its side
/// lanes' work interleaved, each in its own order. Lane lines name the
/// lane's stream, and a launch after a lane joins its stream cannot
/// overlap its predecessor's tail (PDL), so launch flags are not compared.
fn interleaves(stream: &str, eager: &[&String], own: &[&String], lane: &[&String]) {
    let plain = |line: &str| {
        let mut fields: Vec<&str> = line.splitn(3, ' ').collect();
        fields[1] = stream;
        let line = fields.join(" ");
        match line.find("LaunchConfigV1 { flags: ") {
            Some(at) => {
                let rest = &line[at + 24..];
                let end = rest.find(',').unwrap();
                format!("{}{}", &line[..at + 24], &rest[end..])
            }
            None => line,
        }
    };
    let (mut i, mut j) = (0, 0);
    for (at, line) in eager.iter().enumerate() {
        let line = plain(line);
        if own.get(i).is_some_and(|own| plain(own) == line) {
            i += 1;
        } else if lane.get(j).is_some_and(|lane| plain(lane) == line) {
            j += 1;
        } else {
            panic!(
                "stream {stream}: eager operation {at} is neither the graph's next \
                 ({:?}) nor its lane's next ({:?}):\n{line}",
                own.get(i),
                lane.get(j)
            );
        }
    }
    assert_eq!((i, j), (own.len(), lane.len()), "stream {stream}");
}

#[derive(Clone, Copy, Eq, PartialEq)]
enum Graphs {
    /// Every step replays its key's graph, captured at its first step.
    Shared,
    /// Every graph is released before each step, which captures afresh at
    /// its own items.
    Fresh,
    /// No graph handles: every step runs eagerly.
    Eager,
}

struct Run {
    host: Rc<Host>,
    instance: BundleInstanceV1,
    drafts: Vec<Vec<(u64, Vec<u32>)>>,
    /// Target captures and replays, and the replays of a graph captured
    /// at another live item count.
    captures: usize,
    replays: usize,
    cross: usize,
}

fn run(graphs: Graphs) -> Run {
    let (host, mut instance) = create_with_limits(Shape::new(&FLASH, Ranks::TP4).unwrap(), 32, 32);
    let capacity = if graphs == Graphs::Eager { 0 } else { 1 << 20 };
    crate::instance::dspark_graphs_for_test(instance.handle(), Some(capacity));
    for request in 1..=32 {
        admit(&mut instance, &host, request, request - 1);
    }
    let mut captured_at: BTreeMap<String, usize> = BTreeMap::new();
    let (mut cross, mut captures, mut replays) = (0, 0, 0);
    let mut drafts = Vec::new();
    for (items, replace) in schedule() {
        for (old, new, slot) in replace {
            instance.retire_request(old).unwrap();
            admit(&mut instance, &host, new, slot);
        }
        if graphs == Graphs::Fresh {
            crate::instance::release_graphs_for_test(instance.handle());
        }
        let journal_from = host.journal.borrow().len();
        let logs_from = host.logs.borrow().len();
        drafts.push(step(&mut instance, &host, &items));
        let journal = host.journal.borrow();
        let logs = host.logs.borrow();
        // Pair the step's capture events with its capture blocks in order.
        let families: Vec<String> = logs[logs_from..]
            .iter()
            .filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
            .filter(|r| r["schema"] == "chopinfer.graph-cache.v1" && r["event"] == "capture")
            .map(|r| r["family"].as_str().unwrap().to_owned())
            .collect();
        let handles: Vec<String> = journal[journal_from..]
            .iter()
            .filter_map(|line| line.strip_prefix("captured "))
            .map(|rest| rest.split_whitespace().nth(1).unwrap().to_owned())
            .collect();
        assert_eq!(families.len(), handles.len());
        for (family, graph) in families.iter().zip(&handles) {
            if family == "target" {
                captured_at.insert(graph.clone(), items.len());
                captures += 1;
            }
        }
        for line in &journal[journal_from..] {
            if let Some(rest) = line.strip_prefix("replay ") {
                let graph = rest.split_whitespace().nth(1).unwrap();
                if let Some(&at) = captured_at.get(graph) {
                    replays += 1;
                    cross += usize::from(at != items.len());
                }
            }
        }
    }
    Run {
        host,
        instance,
        drafts,
        captures,
        replays,
        cross,
    }
}

#[test]
fn target_graphs_replay_across_item_counts_as_fresh_captures_and_eager_steps_run() {
    let shared = run(Graphs::Shared);
    let fresh = run(Graphs::Fresh);
    let eager = run(Graphs::Eager);
    // Three row counts, each captured once: three pieces on four ranks.
    // Every step replays its key on four ranks, three pieces each; 15 of
    // the 20 steps run at an item count other than their key's capture.
    assert_eq!(
        (shared.captures, shared.replays, shared.cross),
        (3 * 12, 20 * 12, 15 * 12)
    );
    assert_eq!(
        (fresh.captures, fresh.replays, fresh.cross),
        (20 * 12, 20 * 12, 0)
    );
    assert_eq!((eager.captures, eager.replays), (0, 0));
    for other in [&fresh, &eager] {
        assert_eq!(shared.drafts, other.drafts);
        for request in (5..=32).chain(41..=44) {
            assert_eq!(
                state(&shared.instance, request),
                state(&other.instance, request),
                "request {request}"
            );
        }
    }
    let (work, lanes) = executed(&shared.host.journal.borrow());
    // Shared graphs do on every stream, events included, exactly what
    // graphs captured afresh at each step's items do.
    let (again, _) = executed(&fresh.host.journal.borrow());
    assert_eq!(
        work.keys().collect::<Vec<_>>(),
        again.keys().collect::<Vec<_>>()
    );
    for (stream, done) in &work {
        compare(stream, done, &again[stream]);
    }
    // And each rank computes what its eager steps compute.
    let (eager_work, eager_lanes) = executed(&eager.host.journal.borrow());
    assert!(eager_lanes.is_empty());
    assert_eq!(lanes.len(), 4);
    for (stream, done) in &eager_work {
        let own = work.get(stream).map(|w| computed(w)).unwrap_or_default();
        let lane: Vec<&String> = lanes
            .iter()
            .filter(|(_, by)| *by == stream)
            .flat_map(|(lane, _)| computed(&work[lane]))
            .collect();
        interleaves(stream, &computed(done), &own, &lane);
    }
    for mut run in [shared, fresh, eager] {
        for request in (5..=32).chain(41..=44) {
            run.instance.retire_request(request).unwrap();
        }
        run.instance.destroy().unwrap();
        assert_eq!(run.host.acquired.get(), run.host.released.get());
    }
}
