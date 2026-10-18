//! The shared expert overlap (`crate::shared_overlap`) on the fake host:
//! the same scenarios with the overlap off and on run the same launches
//! (per rank, every eager launch and the launches of every graph each
//! replay names, the overlapped compute and side streams' interleaving
//! into the one-stream order line for line) around the same uploads and
//! copies; only the side streams' launches, the fork and join events and
//! the graph pieces differ. Each replay must name a live graph its own
//! rank captured, and each run of replays every piece of one capture group
//! per rank in order, piece-major, the overlapped runs those of their own
//! cache and bucket. Vector clocks over each
//! capture's launch, record and wait lines show every launch pair ordered
//! as one stream orders it except a shared gate/up or down projection and
//! its own sublayer's routed work, which share only bytes neither writes;
//! each capture ends with the side stream joined, waiting only on events
//! recorded inside it.

use std::collections::BTreeMap;
use std::rc::Rc;

use bundle_runtime::kv::KvFormat;
use model_glm5next_contract::contract::{DECODE_BATCH, Shape};
use runtime_bundle_abi::*;

use super::{
    DraftSink, FakeHost, Host, IMAGE_SLOT, KV_SLOT, LaunchRecord, NoRows, api, assert_poisoned,
    drafted, kept, matched, prompt_chunk, step,
};
use crate::buffers::Buffer;
use crate::config::tests::{options, record_naming};
use crate::instance::{Overlapped, RunKey};
use crate::program::Stage;
use crate::shared_overlap::{Op, Settings, with_settings_in_tests};

const ALL: Settings = Settings::ALL;

#[derive(Clone, Copy, Debug)]
enum Scenario {
    /// Decode at buckets 2 and 1 and a verify step of 5 rows over 2 items.
    Target,
    /// A prompt, then two drafted verify steps: the drafter's extend passes
    /// and draft steps and the target's verify.
    Drafter,
}

/// What one scenario left on its host.
struct Trace {
    /// The journal from the first step on.
    journal: Vec<String>,
    /// Where each step's lines start in `journal`.
    steps: Vec<usize>,
    /// Every launch's stream, symbol and arguments from the first step on.
    args: Vec<LaunchRecord>,
    captures: Vec<Overlapped>,
    /// The key of every overlapped run that replayed, in order.
    replays: Vec<RunKey>,
    /// Each stream's rank, and whether it is a side stream.
    streams: BTreeMap<u32, (usize, bool)>,
    /// The overlap's event slots.
    events: Vec<u32>,
    normed: Vec<DeviceBufferSpanV1>,
    intermediate: Vec<DeviceBufferSpanV1>,
    /// The shared down projection's output and zero residual.
    mlp_out: Vec<DeviceBufferSpanV1>,
    zeros: Vec<DeviceBufferSpanV1>,
    /// Each rank's projection scratch and the side stream's.
    scratch: Vec<[Vec<DeviceBufferSpanV1>; 2]>,
    inputs: Vec<Vec<DeviceBufferSpanV1>>,
}

/// An instance of the served program, its projections included, over
/// `format`.
fn create_in(host: &Rc<Host>, format: KvFormat) -> BundleInstanceV1 {
    let roles = [crate::projections::COMPONENT_ROLE.as_bytes().to_vec()];
    let record = record_naming(DECODE_BATCH, &options(2048), format, &roles);
    api()
        .create(HostServicesTableV1::new(Rc::clone(host)), &record)
        .unwrap_or_else(|error| panic!("fake-host create: {error:?} {:?}", host.logs.borrow()))
}

fn format(scenario: Scenario) -> KvFormat {
    match scenario {
        Scenario::Target => Shape::TP4.kv_formats()[0],
        Scenario::Drafter => Shape::TP4.kv_formats()[1],
    }
}

/// Runs `scenario`'s steps; returns the journal length before each.
fn run_steps(host: &Rc<Host>, instance: &mut BundleInstanceV1, scenario: Scenario) -> Vec<usize> {
    let handle = instance.handle();
    let fail = |error: BundleAbiError| panic!("{error:?} {:?}", host.logs.borrow());
    let mut starts = Vec::new();
    let mut mark = || starts.push(host.device.journal.borrow().len());
    match scenario {
        Scenario::Target => {
            crate::instance::seed(handle, 7, 0, 100);
            crate::instance::seed(handle, 8, 5, 64);
            mark();
            let decode = step(&[(7, 0, 100, &[11]), (8, 5, 64, &[12])]);
            instance
                .execute(&decode, &mut NoRows::default())
                .unwrap_or_else(fail);
            instance.commit().unwrap();
            mark();
            let verify = step(&[(7, 0, 101, &[21, 22, 23, 24]), (8, 5, 65, &[13])]);
            instance
                .execute(&verify, &mut NoRows::default())
                .unwrap_or_else(fail);
            instance.commit_accepted(&[kept(7, 2)]).unwrap();
            mark();
            instance
                .execute(&step(&[(7, 0, 103, &[31])]), &mut NoRows::default())
                .unwrap_or_else(fail);
            instance.commit().unwrap();
        }
        Scenario::Drafter => {
            mark();
            instance
                .execute(&prompt_chunk(7, 0, 10, true), &mut DraftSink::default())
                .unwrap_or_else(fail);
            instance.commit().unwrap();
            for previous in [10, 12] {
                mark();
                let mut sink = DraftSink::default();
                instance
                    .execute(&drafted(7, previous, 4, 21), &mut sink)
                    .unwrap_or_else(fail);
                assert_eq!(sink.drafts.len(), 1);
                instance.commit_accepted(&[kept(7, 2)]).unwrap();
            }
        }
    }
    starts
}

fn run(settings: Settings, scenario: Scenario) -> Trace {
    let format = format(scenario);
    trace_with(
        settings,
        FakeHost::with_format((3, 32), format),
        |host| create_in(host, format),
        |host, instance| run_steps(host, instance, scenario),
    )
}

/// Traces `steps` (which return the journal length before each step) on
/// the instance `create` makes on `host` with `settings`.
fn trace_with(
    settings: Settings,
    host: Rc<Host>,
    create: impl FnOnce(&Rc<Host>) -> BundleInstanceV1,
    steps: impl FnOnce(&Rc<Host>, &mut BundleInstanceV1) -> Vec<usize>,
) -> Trace {
    let mut instance = with_settings_in_tests(settings, || create(&host));
    let handle = instance.handle();
    // Only decode, verify and drafter graphs replay.
    crate::instance::keep_last_chunks_eager(handle);
    let _ = crate::instance::overlapped_captures();
    let _ = crate::instance::overlapped_replays();
    *host.launch_args.borrow_mut() = Some(Vec::new());
    let start = host.device.journal.borrow().len();
    let steps = steps(&host, &mut instance)
        .into_iter()
        .map(|at| at - start)
        .collect();
    let lanes = crate::instance::side_lanes(handle);
    let mut streams = host
        .device
        .stream_ranks
        .borrow()
        .iter()
        .map(|(&stream, &rank)| (stream, (rank as usize, false)))
        .collect::<BTreeMap<_, _>>();
    for lane in &lanes {
        streams.get_mut(&lane.side.slot()).unwrap().1 = true;
    }
    let ranks = 0..Shape::TP4.ranks() as usize;
    let trace = Trace {
        journal: host.device.journal.borrow()[start..].to_vec(),
        steps,
        args: host.launch_args.borrow_mut().take().unwrap(),
        captures: crate::instance::overlapped_captures(),
        replays: crate::instance::overlapped_replays(),
        streams,
        events: lanes
            .iter()
            .flat_map(|lane| [lane.fork.slot(), lane.join.slot()])
            .collect(),
        normed: ranks
            .clone()
            .map(|rank| crate::instance::scratch_span(handle, rank, Buffer::Normed))
            .collect(),
        intermediate: ranks
            .clone()
            .map(|rank| crate::instance::scratch_span(handle, rank, Buffer::DenseIntermediate))
            .collect(),
        mlp_out: ranks
            .clone()
            .map(|rank| crate::instance::scratch_span(handle, rank, Buffer::MlpOut))
            .collect(),
        zeros: ranks
            .clone()
            .map(|rank| crate::instance::scratch_span(handle, rank, Buffer::ZeroRows))
            .collect(),
        scratch: ranks
            .clone()
            .map(|rank| crate::instance::projection_scratch_spans(handle, rank))
            .collect(),
        inputs: ranks
            .map(|rank| {
                let mut spans = crate::instance::input_spans(handle, rank);
                spans.extend(crate::instance::draft_input_spans(handle, rank));
                spans
            })
            .collect(),
    };
    instance.destroy().unwrap();
    assert_eq!(host.acquired.get(), host.released.get());
    trace
}

/// The stream a journal line of `kind` (`launch`, `record`, `wait`,
/// `capture`, `captured`, `replay`) names.
fn stream_of(line: &str, kind: &str) -> Option<u32> {
    line.strip_prefix(kind)?
        .strip_prefix(' ')?
        .split(' ')
        .next()?
        .parse()
        .ok()
}

/// A launch line without its stream.
fn launch_body(line: &str) -> Option<&str> {
    let rest = line.strip_prefix("launch ")?;
    Some(&rest[rest.find(' ')? + 1..])
}

/// The stream and launch count a `captured` line names.
fn captured_nodes(line: &str) -> Option<(u32, usize)> {
    let stream = stream_of(line, "captured")?;
    let count = line.rsplit(' ').next()?.parse().ok()?;
    Some((stream, count))
}

/// Whether `line` records or waits on one of `events`.
fn overlap_event(line: &str, events: &[u32]) -> bool {
    ["record ", "wait "].iter().any(|prefix| {
        line.strip_prefix(prefix)
            .and_then(|rest| rest.split(' ').nth(1))
            .and_then(|event| event.parse().ok())
            .is_some_and(|event: u32| events.contains(&event))
    })
}

/// The journal's lines but launches, the overlap's events and captures,
/// each run of consecutive replays as one `replay` line and releases
/// without their graph handles (the overlap's streams and events take
/// handle IDs before the graphs', and the graphs' pieces differ; what
/// each replay runs is [`executed`]'s).
fn other_lines(trace: &Trace) -> Vec<String> {
    let mut lines: Vec<String> = Vec::new();
    for line in &trace.journal {
        if line.starts_with("launch ")
            || line.starts_with("capture ")
            || line.starts_with("captured ")
            || overlap_event(line, &trace.events)
        {
            continue;
        }
        if line.starts_with("replay ") {
            if lines.last().is_none_or(|last| last != "replay") {
                lines.push("replay".to_owned());
            }
        } else if line.starts_with("release graph ") {
            lines.push("release graph".to_owned());
        } else {
            lines.push(line.clone());
        }
    }
    lines
}

/// A launch line without its stream, and whether it ran on a side stream.
type Launch = (String, bool);

/// One captured graph piece: its rank, the capture group it belongs to
/// (the pieces one rank captured back to back, a graph cache entry) and
/// its launches.
struct Graph {
    rank: usize,
    group: usize,
    launches: Vec<Launch>,
}

/// One capture group: its rank, its pieces' graphs in order and, for an
/// overlapped trace, the key it was captured for.
struct Group {
    rank: usize,
    graphs: Vec<u32>,
    key: Option<RunKey>,
}

/// Per rank, what its device runs from the first step on, in order: each
/// eager launch, and the launches of each graph each replay names. Every
/// replay must name a captured and unreleased graph of its own rank; each
/// run of consecutive replays must replay, per rank, every piece of one of
/// its capture groups once, in order, piece-major across ranks (as
/// `GraphCache::replay_piece_major` does). With `keyed`, the capture groups
/// take the keys of the trace's overlapped captures in order, and replay
/// run `i` must replay groups of the `i`-th overlapped replay's key.
fn executed(trace: &Trace, keyed: bool) -> Vec<Vec<Launch>> {
    let ranks = Shape::TP4.ranks() as usize;
    let mut done = vec![Vec::new(); ranks];
    let mut graphs = BTreeMap::<u32, Graph>::new();
    let mut groups = Vec::<Group>::new();
    let mut capturing: Option<(u32, Vec<Launch>)> = None;
    let mut previous: Option<&str> = None;
    let mut run: Vec<(usize, u32)> = Vec::new();
    let mut runs = 0;
    let mut finish = |run: &mut Vec<(usize, u32)>,
                      graphs: &BTreeMap<u32, Graph>,
                      groups: &[Group],
                      done: &mut Vec<Vec<Launch>>| {
        // Consecutive replays may hold several runs: each starts with every
        // rank's first piece of one group, rank by rank.
        let mut at = 0;
        while at < run.len() {
            let mut owners = BTreeMap::new();
            for &(rank, graph) in &run[at..] {
                let group = graphs[&graph].group;
                if owners.contains_key(&rank) || groups[group].graphs.first() != Some(&graph) {
                    break;
                }
                owners.insert(rank, group);
            }
            assert_eq!(
                owners.len(),
                ranks,
                "a replay run is not every piece of its groups, piece-major: {:?}",
                &run[at..]
            );
            let depth = owners
                .values()
                .map(|&group| groups[group].graphs.len())
                .max()
                .unwrap_or(0);
            let mut order = Vec::new();
            for piece in 0..depth {
                for (&rank, &group) in &owners {
                    if let Some(&graph) = groups[group].graphs.get(piece) {
                        order.push((rank, graph));
                    }
                }
            }
            let end = (at + order.len()).min(run.len());
            assert_eq!(
                run[at..end],
                order[..],
                "a replay run is not every piece of its groups, piece-major"
            );
            if keyed {
                let key = trace.replays.get(runs).expect("an overlapped replay");
                for &group in owners.values() {
                    assert_eq!(
                        groups[group].key.as_ref(),
                        Some(key),
                        "a replay of another cache's graphs"
                    );
                }
            }
            runs += 1;
            at = end;
        }
        for &(rank, graph) in run.iter() {
            done[rank].extend(graphs[&graph].launches.iter().cloned());
        }
        run.clear();
    };
    let mut keys = trace.captures.iter().map(|capture| capture.key);
    for line in &trace.journal {
        if let Some(stream) = stream_of(line, "replay") {
            let graph: u32 = line.rsplit(' ').next().unwrap().parse().unwrap();
            let (rank, side) = trace.streams[&stream];
            assert!(!side, "a replay on a side stream");
            let known = graphs
                .get(&graph)
                .unwrap_or_else(|| panic!("replay of unknown graph {graph}"));
            assert_eq!(
                known.rank, rank,
                "rank {rank} replays rank {}'s graph",
                known.rank
            );
            run.push((rank, graph));
            previous = Some(line);
            continue;
        }
        finish(&mut run, &graphs, &groups, &mut done);
        if let Some(stream) = stream_of(line, "capture") {
            assert!(capturing.is_none());
            let rank = trace.streams[&stream].0;
            let continues = previous
                .and_then(captured_nodes)
                .is_some_and(|(captured, _)| captured == stream);
            if !continues {
                let key = if keyed {
                    Some(keys.next().expect("an overlapped capture"))
                } else {
                    None
                };
                groups.push(Group {
                    rank,
                    graphs: Vec::new(),
                    key,
                });
            }
            capturing = Some((stream, Vec::new()));
        } else if let Some((stream, nodes)) = captured_nodes(line) {
            let (capturing_stream, launches) = capturing.take().expect("an open capture");
            assert_eq!(stream, capturing_stream);
            assert_eq!(launches.len(), nodes);
            let graph: u32 = line.split(' ').nth(2).unwrap().parse().unwrap();
            let group = groups.len() - 1;
            assert_eq!(groups[group].rank, trace.streams[&stream].0);
            groups[group].graphs.push(graph);
            graphs.insert(
                graph,
                Graph {
                    rank: groups[group].rank,
                    group,
                    launches,
                },
            );
        } else if let Some(graph) = line.strip_prefix("release graph ") {
            graphs.remove(&graph.parse().unwrap());
        } else if let (Some(stream), Some(body)) = (stream_of(line, "launch"), launch_body(line)) {
            let (rank, side) = trace.streams[&stream];
            match capturing.as_mut() {
                Some((capturing_stream, launches)) => {
                    assert_eq!(
                        trace.streams[capturing_stream].0, rank,
                        "a launch beside another rank's capture"
                    );
                    launches.push((body.to_owned(), side));
                }
                None => {
                    assert!(!side, "a side launch outside a capture");
                    done[rank].push((body.to_owned(), false));
                }
            }
        }
        previous = Some(line);
    }
    finish(&mut run, &graphs, &groups, &mut done);
    assert!(capturing.is_none());
    if keyed {
        assert!(
            keys.next().is_none(),
            "every overlapped capture is in the journal"
        );
        assert_eq!(runs, trace.replays.len(), "every overlapped run replayed");
    }
    done
}

/// `on` runs `off`'s launches: per rank, the launches its device runs
/// ([`executed`]: eager launches and each replayed graph's) interleave its
/// compute and side streams' launches into `off`'s, line for line, and
/// every other line but the overlap's events and the graph pieces is the
/// same. With `keyed`, `on`'s replays run the graphs of their own runs'
/// keys. Returns the side launches per rank.
fn check_equivalent(off: &Trace, on: &Trace, keyed: bool) -> Vec<usize> {
    assert_eq!(other_lines(off), other_lines(on));
    assert!(off.events.is_empty());
    let mut sides = Vec::new();
    for (rank, (one, two)) in executed(off, false)
        .into_iter()
        .zip(executed(on, keyed))
        .enumerate()
    {
        assert!(one.iter().all(|(_, side)| !side));
        let compute = two.iter().filter(|(_, side)| !side).map(|(line, _)| line);
        let side = two.iter().filter(|(_, side)| *side).map(|(line, _)| line);
        let (mut compute, mut side) = (compute.peekable(), side.peekable());
        let mut on_side = 0;
        for (index, (line, _)) in one.iter().enumerate() {
            if compute.peek() == Some(&line) {
                compute.next();
            } else if side.peek() == Some(&line) {
                side.next();
                on_side += 1;
            } else {
                panic!("rank {rank}'s launch {index} is not the next on either stream: {line}");
            }
        }
        assert!(
            compute.next().is_none() && side.next().is_none(),
            "rank {rank}"
        );
        sides.push(on_side);
    }
    sides
}

/// One captured launch: its index in the rank's program, its stage and
/// stream, the graph piece it is captured in and its vector clock over
/// the compute (0) and side (1) streams.
struct Captured {
    index: usize,
    stage: Stage,
    side: bool,
    piece: usize,
    clock: [u64; 2],
    spans: Vec<DeviceBufferSpanV1>,
}

impl Captured {
    /// Whether `self` happens before `other`: an earlier piece's graph
    /// ends before a later one starts; inside a piece, by the clocks.
    fn before(&self, other: &Self) -> bool {
        let lane = usize::from(self.side);
        self.piece < other.piece
            || (self.piece == other.piece && self.clock[lane] <= other.clock[lane])
    }
}

fn spans(arguments: &[KernelArgumentV1]) -> Vec<DeviceBufferSpanV1> {
    let mut spans = Vec::new();
    for argument in arguments {
        match argument {
            KernelArgumentV1::DeviceBuffer(span) => spans.push(*span),
            KernelArgumentV1::DevicePointerArray(array) => spans.extend(array.iter().flatten()),
            KernelArgumentV1::TensorMap(map) => spans.push(map.buffer),
            _ => {}
        }
    }
    spans
}

fn overlaps(a: DeviceBufferSpanV1, b: DeviceBufferSpanV1) -> bool {
    a.buffer == b.buffer && a.offset < b.offset + b.length && b.offset < a.offset + a.length
}

fn within(a: DeviceBufferSpanV1, b: DeviceBufferSpanV1) -> bool {
    a.buffer == b.buffer && a.offset >= b.offset && a.offset + a.length <= b.offset + b.length
}

/// Layer `layer`'s routed work, which runs beside its shared expert.
fn routed(stage: Stage, layer: u32) -> bool {
    match stage {
        Stage::Project(l, "moe.router" | "mtp.moe.router") => l == layer,
        Stage::Route(l)
        | Stage::LocalRoutes(l)
        | Stage::ExpertsGateUp(l)
        | Stage::ExpertsDown(l) => l == layer,
        _ => false,
    }
}

/// The layer of a shared expert's gate/up or down projection.
fn shared(stage: Stage) -> Option<u32> {
    match stage {
        Stage::SharedGateUp(layer)
        | Stage::Project(layer, "moe.shared.down" | "mtp.moe.shared.down") => Some(layer),
        _ => None,
    }
}

/// Every overlapped capture of `trace`, by vector clocks: each piece's
/// waits name events recorded in it, and it ends with the side stream
/// joined; each capture's launches keep the one-stream order but for a
/// shared gate/up or down projection and its sublayer's routed work, which
/// run unordered and share only bytes neither writes. Returns the
/// overlapped sublayers.
fn check_order(trace: &Trace) -> usize {
    let mut captures = trace.captures.iter();
    let mut args = trace.args.iter();
    // The open capture: its owner's issue order, the launches so far, the
    // streams' clocks and the events recorded in this piece.
    let mut current: Option<(&Overlapped, Vec<usize>, Vec<Captured>)> = None;
    let mut open: Option<OpenCapture> = None;
    let mut piece = 0;
    let mut regions = 0;
    for line in &trace.journal {
        if let Some(stream) = stream_of(line, "launch") {
            let (stream_rank, side) = trace.streams[&stream];
            let (_, name, arguments) = args.next().expect("a launch record");
            assert!(line.contains(name.as_str()));
            let Some((_, clocks, _)) = open.as_mut() else {
                assert!(!side, "a side launch outside a capture");
                continue;
            };
            let (capture, order, launched) = current.as_mut().expect("an overlapped capture");
            assert_eq!(stream_rank, capture.rank);
            let lane = usize::from(side);
            clocks[lane][lane] += 1;
            let index = order[launched.len()];
            launched.push(Captured {
                index,
                stage: capture.stages[index],
                side,
                piece,
                clock: clocks[lane],
                spans: spans(arguments),
            });
        } else if let Some(stream) = stream_of(line, "capture") {
            assert!(open.is_none());
            open = Some((stream, [[0; 2]; 2], BTreeMap::new()));
            piece += 1;
            if current.is_none() {
                let capture = captures.next().expect("an overlapped capture");
                let order = capture
                    .units
                    .iter()
                    .flatten()
                    .flat_map(|op| match op {
                        Op::Compute(range) | Op::Side(range) => range.clone(),
                        Op::Fork | Op::Join => 0..0,
                    })
                    .collect::<Vec<_>>();
                regions += capture.units.iter().filter(|unit| unit.len() > 1).count();
                current = Some((capture, order, Vec::new()));
            }
            let (capture, _, _) = current.as_ref().unwrap();
            assert_eq!(trace.streams[&stream], (capture.rank, false));
        } else if let Some((stream, _)) = captured_nodes(line) {
            let (capturing, clocks, _) = open.take().expect("an open capture");
            assert_eq!(stream, capturing);
            // The compute stream waited for every side launch.
            assert_eq!(clocks[0][1], clocks[1][1], "the side stream joins");
            let done = current
                .as_ref()
                .is_some_and(|(_, order, launched)| order.len() == launched.len());
            if done {
                let (capture, _, launched) = current.take().unwrap();
                check_pairs(trace, capture.rank, &launched);
            }
        } else if overlap_event(line, &trace.events) {
            let (_, clocks, events) = open.as_mut().expect("events only inside a capture");
            let mut words = line.split(' ');
            let kind = words.next().unwrap();
            let stream = words.next().unwrap().parse::<u32>().unwrap();
            let event = words.next().unwrap().parse::<u32>().unwrap();
            let lane = usize::from(trace.streams[&stream].1);
            if kind == "record" {
                events.insert(event, clocks[lane]);
            } else {
                let recorded = events
                    .get(&event)
                    .expect("a wait on an event recorded in its own capture");
                for (clock, &at) in clocks[lane].iter_mut().zip(recorded) {
                    *clock = (*clock).max(at);
                }
            }
        }
    }
    assert!(open.is_none() && current.is_none());
    assert!(captures.next().is_none(), "every overlapped capture ran");
    assert!(args.next().is_none());
    regions
}

/// An open capture's stream, its compute and side streams' clocks and the
/// clocks of the events recorded in it.
type OpenCapture = (u32, [[u64; 2]; 2], BTreeMap<u32, [u64; 2]>);

/// One rank's overlapped capture (see [`check_order`]).
fn check_pairs(trace: &Trace, rank: usize, launched: &[Captured]) {
    let mut launches = launched.iter().collect::<Vec<_>>();
    launches.sort_by_key(|launch| launch.index);
    assert!(
        launches
            .iter()
            .enumerate()
            .all(|(at, launch)| launch.index == at)
    );
    let readonly = |span: DeviceBufferSpanV1| {
        (IMAGE_SLOT..KV_SLOT).contains(&span.buffer.slot())
            || trace.inputs[rank].iter().any(|&input| within(span, input))
    };
    let (normed, intermediate) = (trace.normed[rank], trace.intermediate[rank]);
    let (mlp_out, zeros) = (trace.mlp_out[rank], trace.zeros[rank]);
    let [_, side_scratch] = &trace.scratch[rank];
    let mut unordered = BTreeMap::<usize, usize>::new();
    for (at, a) in launches.iter().enumerate() {
        for b in &launches[at + 1..] {
            if a.before(b) {
                continue;
            }
            assert!(!b.before(a), "{:?} runs before {:?}", b.stage, a.stage);
            let (side, compute) = match (a.side, b.side) {
                (true, false) => (a, b),
                (false, true) => (b, a),
                _ => panic!("{:?} and {:?} are unordered", a.stage, b.stage),
            };
            let Some(layer) = shared(side.stage) else {
                panic!("{:?} runs on the side stream", side.stage);
            };
            assert!(
                routed(compute.stage, layer),
                "{:?} runs beside layer {layer}'s shared expert",
                compute.stage
            );
            // The shared gate/up reads the normalized rows, weights and
            // inputs and writes its rows of the intermediate; the shared
            // down projection reads the intermediate, the zero rows, weights
            // and inputs, writes the MLP output and binds the side stream's
            // projection scratch. The routed work binds none of the
            // intermediate, the MLP output, the zero rows and the side
            // scratch, and the normalized rows only to read them.
            for &span in &side.spans {
                let own = match side.stage {
                    Stage::SharedGateUp(_) => within(span, normed) || span == intermediate,
                    _ => {
                        span == intermediate
                            || span == mlp_out
                            || span == zeros
                            || side_scratch.iter().any(|&scratch| within(span, scratch))
                    }
                };
                assert!(own || readonly(span), "{:?} {span:?}", side.stage);
            }
            for &span in &compute.spans {
                for &theirs in [intermediate, mlp_out, zeros].iter().chain(side_scratch) {
                    assert!(!overlaps(span, theirs), "{:?} {span:?}", compute.stage);
                }
                if overlaps(span, normed) {
                    assert!(
                        matches!(
                            compute.stage,
                            Stage::Project(_, "moe.router" | "mtp.moe.router")
                                | Stage::ExpertsGateUp(_)
                        ),
                        "{:?} binds the normalized rows",
                        compute.stage
                    );
                }
            }
            *unordered.entry(side.index).or_default() += 1;
        }
    }
    // Each shared gate/up and down projection runs beside every launch of
    // its sublayer's routed work.
    for launch in launches.iter().filter(|launch| launch.side) {
        let layer = shared(launch.stage).unwrap();
        let routed = launches
            .iter()
            .filter(|other| routed(other.stage, layer))
            .count();
        assert!(routed >= 5, "{layer}");
        assert_eq!(unordered.get(&launch.index), Some(&routed), "{layer}");
    }
}

/// With the overlap on, every graph run issues the launches of the run
/// with it off over two streams per rank, and every capture keeps the
/// one-stream order but for the shared experts beside their routed work.
#[test]
fn overlapped_graphs_issue_the_one_stream_launches_beside_the_routed_experts() {
    for scenario in [Scenario::Target, Scenario::Drafter] {
        let off = run(Settings::OFF, scenario);
        let on = run(ALL, scenario);
        assert!(off.captures.is_empty());
        let sides = check_equivalent(&off, &on, true);
        let regions = check_order(&on);
        // Each graph run captured every rank, each rank's capture forking
        // at each of its 42 target MoE sublayers or the drafter's one.
        let ranks = on
            .captures
            .iter()
            .map(|capture| capture.rank)
            .collect::<Vec<_>>();
        assert!(
            ranks.chunks(4).all(|chunk| chunk == [0, 1, 2, 3]),
            "{ranks:?}"
        );
        let target = on
            .captures
            .iter()
            .filter(|capture| capture.stages.contains(&Stage::SharedGateUp(3)))
            .count();
        let drafter = on.captures.len() - target;
        assert_eq!(regions, 42 * target + drafter, "{scenario:?}");
        // Each replay runs its key's shared gate/ups and down projections on
        // every rank, once each on the side streams.
        let replayed = on
            .replays
            .iter()
            .flat_map(|key| {
                on.captures
                    .iter()
                    .filter(move |capture| capture.key == *key)
            })
            .map(|capture| {
                capture
                    .stages
                    .iter()
                    .filter(|&&stage| shared(stage).is_some())
                    .count()
            })
            .sum::<usize>();
        assert!(replayed >= 2 * regions, "{scenario:?}");
        assert_eq!(sides.iter().sum::<usize>(), replayed, "{scenario:?}");
        match scenario {
            // Decode buckets 2 and 1 and a verify bucket.
            Scenario::Target => assert_eq!((target, drafter), (12, 0)),
            Scenario::Drafter => assert!(target >= 4 && drafter >= 8, "{target} {drafter}"),
        }
        // A fork and a join per sublayer, each a record and a wait.
        let events = on
            .journal
            .iter()
            .filter(|line| overlap_event(line, &on.events))
            .count();
        assert_eq!(events, 4 * regions);
    }
}

/// The switch names the largest bucket that overlaps: at 1 row only the
/// one-item decode bucket's graphs fork; the others issue the one-stream
/// journal.
#[test]
fn only_buckets_within_the_switch_overlap() {
    let off = run(Settings::OFF, Scenario::Target);
    let one = run(Settings { max_rows: 1 }, Scenario::Target);
    check_equivalent(&off, &one, false);
    // The two-item decode and the verify step fork nowhere.
    let third = one.steps[2];
    assert!(
        !one.journal[..third]
            .iter()
            .any(|line| overlap_event(line, &one.events))
    );
    let launches = one.journal[..third]
        .iter()
        .filter(|line| line.starts_with("launch "))
        .count();
    let last = Trace {
        journal: one.journal[third..].to_vec(),
        steps: Vec::new(),
        args: one.args[launches..].to_vec(),
        captures: one.captures.clone(),
        replays: one.replays.clone(),
        streams: one.streams.clone(),
        events: one.events.clone(),
        normed: one.normed.clone(),
        intermediate: one.intermediate.clone(),
        mlp_out: one.mlp_out.clone(),
        zeros: one.zeros.clone(),
        scratch: one.scratch.clone(),
        inputs: one.inputs.clone(),
    };
    assert_eq!(one.captures.len(), 4);
    assert_eq!(check_order(&last), 4 * 42);
}

/// A failed fork or join in an overlapped capture (rank 1's sixth
/// sublayer), its record or its wait, fails the step before any rank
/// replays: the instance is not poisoned, every capture ended, the failing
/// rank's pieces are released, and the step runs again, capturing the
/// ranks still missing their graphs.
#[test]
fn a_failed_fork_or_join_fails_the_capture_before_any_replay() {
    for (wait, join) in [(true, false), (true, true), (false, false), (false, true)] {
        let format = format(Scenario::Target);
        let host = FakeHost::with_format((3, 32), format);
        let mut instance = with_settings_in_tests(ALL, || create_in(&host, format));
        let handle = instance.handle();
        crate::instance::seed(handle, 7, 0, 100);
        let _ = crate::instance::overlapped_captures();
        // Each sublayer records (and waits for) its fork, then its join;
        // rank 0 has 42. A countdown of `n` lets `n` calls pass and fails
        // the next: 94 passes rank 0's 84 and rank 1's first five
        // sublayers' 10, so the sixth sublayer's fork fails, and 95 its
        // join.
        let switch = if wait {
            &host.device.fail_wait
        } else {
            &host.device.fail_record
        };
        let passed = 2 * 42 + 10 + u32::from(join);
        switch.set(Some(passed));
        let start = host.device.journal.borrow().len();
        let decode = step(&[(7, 0, 100, &[1])]);
        assert_eq!(
            instance
                .execute(&decode, &mut NoRows::default())
                .unwrap_err(),
            BundleAbiError::Status(BundleStatusV1::HostError)
        );
        assert!(switch.get().is_none(), "the switch fired");
        assert!(!instance.is_poisoned());
        let journal = host.device.journal.borrow()[start..].to_vec();
        // The failed call is the sixth sublayer's fork or join: the calls
        // of its kind before it alternate fork, join from rank 0's first.
        let lanes = crate::instance::side_lanes(handle);
        let kind = if wait { "wait " } else { "record " };
        let events = lanes
            .iter()
            .flat_map(|lane| [lane.fork.slot(), lane.join.slot()])
            .collect::<Vec<_>>();
        let named = journal
            .iter()
            .filter(|line| line.starts_with(kind) && overlap_event(line, &events))
            .map(|line| line.split(' ').nth(2).unwrap().parse::<u32>().unwrap())
            .collect::<Vec<_>>();
        let event = |region: usize, join: bool| {
            let lane = &lanes[region / 42];
            if join {
                lane.join.slot()
            } else {
                lane.fork.slot()
            }
        };
        let before = (0..passed as usize)
            .map(|call| event(call / 2, call % 2 == 1))
            .collect::<Vec<_>>();
        assert_eq!(named[..passed as usize], before[..]);
        // After a failed fork wait or join record the join is still issued
        // (a record, then a wait, unless the record failed); after a
        // failed fork record nothing more.
        let after = &named[passed as usize..];
        match (wait, join) {
            (true, false) => assert_eq!(after, [event(47, true)]),
            (false, false) | (true, true) | (false, true) => assert!(after.is_empty(), "{after:?}"),
        }
        assert!(!journal.iter().any(|line| line.starts_with("replay ")));
        let ranks = |captures: Vec<Overlapped>| {
            captures
                .into_iter()
                .map(|capture| capture.rank)
                .collect::<Vec<_>>()
        };
        assert_eq!(ranks(crate::instance::overlapped_captures()), [0, 1]);
        // Every capture ended; rank 1's pieces were released.
        let count = |prefix: &str| {
            journal
                .iter()
                .filter(|line| line.starts_with(prefix))
                .count()
        };
        assert_eq!(count("capture "), count("captured "));
        assert!(count("release graph ") > 0);
        // No collective was queued.
        assert!(host.device.collectives.borrow().is_empty());
        // The cancelled step left its slot invalid: the request again, then
        // the step; ranks 1 to 3 capture, every rank replays.
        let _ = instance.retire_request(7);
        crate::instance::seed(handle, 7, 0, 100);
        let start = host.device.journal.borrow().len();
        instance
            .execute(&decode, &mut NoRows::default())
            .unwrap_or_else(|error| panic!("{error:?} {:?}", host.logs.borrow()));
        instance.commit().unwrap();
        assert_eq!(ranks(crate::instance::overlapped_captures()), [1, 2, 3]);
        assert!(
            host.device.journal.borrow()[start..]
                .iter()
                .any(|line| line.starts_with("replay "))
        );
        assert!(matched(&host));
        instance.destroy().unwrap();
        assert_eq!(host.acquired.get(), host.released.get());
    }
}

/// An overlapped decode replay that fails after the first rank queued its
/// first piece poisons the instance, as a one-stream replay does.
#[test]
fn a_partial_overlapped_replay_poisons_the_instance() {
    let format = format(Scenario::Target);
    let host = FakeHost::with_format((3, 32), format);
    let mut instance = with_settings_in_tests(ALL, || create_in(&host, format));
    let handle = instance.handle();
    crate::instance::seed(handle, 7, 0, 100);
    instance
        .execute(&step(&[(7, 0, 100, &[1])]), &mut NoRows::default())
        .unwrap();
    instance.commit().unwrap();
    assert_eq!(crate::instance::overlapped_captures().len(), 4);
    host.device.fail_graph_launch.set(Some(1));
    let journal = host.device.journal.borrow().len();
    assert_eq!(
        instance
            .execute(&step(&[(7, 0, 101, &[2])]), &mut NoRows::default())
            .unwrap_err(),
        BundleAbiError::Status(BundleStatusV1::HostError)
    );
    assert_poisoned(&host, &mut instance, journal, |line| {
        line.starts_with("replay ")
    });
}

/// A panic once the first rank queued its overlapped replay poisons the
/// instance, as in a one-stream replay.
#[test]
fn a_panic_in_an_overlapped_replay_poisons_the_instance() {
    let format = format(Scenario::Target);
    let host = FakeHost::with_format((3, 32), format);
    let mut instance = with_settings_in_tests(ALL, || create_in(&host, format));
    let handle = instance.handle();
    crate::instance::seed(handle, 7, 0, 100);
    crate::instance::panic_after(0);
    assert_eq!(
        instance
            .execute(&step(&[(7, 0, 100, &[1])]), &mut NoRows::default())
            .unwrap_err(),
        BundleAbiError::Status(BundleStatusV1::Panic)
    );
    assert_eq!(crate::instance::overlapped_captures().len(), 4);
    assert!(!matched(&host));
    let released = host.released.get();
    let journal = host.device.journal.borrow().len();
    assert!(
        instance
            .execute(&step(&[(7, 0, 100, &[1])]), &mut NoRows::default())
            .is_err()
    );
    assert!(instance.destroy().is_err());
    assert!(instance.is_poisoned());
    assert_eq!(host.released.get(), released);
    assert_eq!(host.device.journal.borrow().len(), journal);
}

/// Without the switch an instance creates no side stream or event, and
/// with it one stream and two events per rank, after everything else.
#[test]
fn the_side_streams_and_events_are_created_last_and_only_with_the_switch() {
    let format = format(Scenario::Target);
    let mut journals = Vec::new();
    for settings in [Settings::OFF, ALL] {
        let host = FakeHost::with_format((3, 32), format);
        let mut instance = with_settings_in_tests(settings, || create_in(&host, format));
        let lanes = crate::instance::side_lanes(instance.handle());
        assert_eq!(lanes.len(), if settings.on() { 4 } else { 0 });
        journals.push(host.device.journal.borrow().clone());
        instance.destroy().unwrap();
        assert_eq!(host.acquired.get(), host.released.get());
    }
    let [off, on] = &journals[..] else {
        unreachable!()
    };
    // The same journal up to the last synchronization of acquire; the
    // overlapped instance creates 4 streams and 8 events before it.
    let added = on
        .iter()
        .filter(|line| !off.contains(line))
        .cloned()
        .collect::<Vec<_>>();
    assert_eq!(added.len(), 12, "{added:?}");
    assert_eq!(
        added
            .iter()
            .filter(|line| line.starts_with("stream "))
            .count(),
        4
    );
    assert_eq!(
        added
            .iter()
            .filter(|line| line.starts_with("event "))
            .count(),
        8
    );
    let without = on
        .iter()
        .filter(|line| !added.contains(line))
        .cloned()
        .collect::<Vec<_>>();
    assert_eq!(&without, off);
}

impl Trace {
    fn mutated(&self, journal: Vec<String>) -> Self {
        Self {
            journal,
            steps: self.steps.clone(),
            args: self.args.clone(),
            captures: self.captures.clone(),
            replays: self.replays.clone(),
            streams: self.streams.clone(),
            events: self.events.clone(),
            normed: self.normed.clone(),
            intermediate: self.intermediate.clone(),
            mlp_out: self.mlp_out.clone(),
            zeros: self.zeros.clone(),
            scratch: self.scratch.clone(),
            inputs: self.inputs.clone(),
        }
    }

    /// The journal without the waits of `stream`s of `side`.
    fn without_waits(&self, side: bool) -> Self {
        self.mutated(
            self.journal
                .iter()
                .filter(|line| {
                    !(overlap_event(line, &self.events)
                        && stream_of(line, "wait")
                            .is_some_and(|stream| self.streams[&stream].1 == side))
                })
                .cloned()
                .collect(),
        )
    }
}

/// The message `check` panics with, if it does.
fn failure(check: impl FnOnce()) -> Option<String> {
    let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(check)).err()?;
    Some(match panicked.downcast::<String>() {
        Ok(message) => *message,
        Err(panicked) => panicked
            .downcast::<&str>()
            .map_or_else(|_| String::new(), |message| (*message).to_owned()),
    })
}

/// The checks catch broken schedules: compute streams that never wait for
/// the join, side streams that never wait for the fork, a routed launch
/// that binds the shared expert's intermediate, its MLP output or the side
/// stream's projection scratch, two compute launches issued out of order,
/// and replays of the wrong graphs (a piece twice, pieces reversed, an
/// unknown graph, another bucket's graphs).
#[test]
fn the_overlap_checks_fail_on_broken_schedules() {
    let off = run(Settings::OFF, Scenario::Target);
    let on = run(ALL, Scenario::Target);
    let order = |trace: &Trace| {
        failure(|| {
            check_order(trace);
        })
    };
    assert_eq!(order(&on), None);
    // Without the joins the compute stream never waits for a side launch.
    let message = order(&on.without_waits(false)).unwrap();
    assert!(message.contains("the side stream joins"), "{message}");
    // Without the forks a shared gate/up runs beside its norm.
    let message = order(&on.without_waits(true)).unwrap();
    assert!(
        message.contains("runs beside layer 3's shared expert"),
        "{message}"
    );
    // The first captured routed experts' gate/up also binds the
    // intermediate.
    let crate::instance::Overlapped {
        rank,
        units,
        stages,
        ..
    } = &on.captures[0];
    let gate_up = stages
        .iter()
        .position(|stage| matches!(stage, Stage::ExpertsGateUp(_)))
        .unwrap();
    let issued = units
        .iter()
        .flatten()
        .flat_map(|op| match op {
            Op::Compute(range) | Op::Side(range) => range.clone(),
            Op::Fork | Op::Join => 0..0,
        })
        .position(|index| index == gate_up)
        .unwrap();
    let first = on
        .journal
        .iter()
        .take_while(|line| !line.starts_with("capture "))
        .filter(|line| line.starts_with("launch "))
        .count();
    let mut binding = on.mutated(on.journal.clone());
    let (_, name, arguments) = &mut binding.args[first + issued];
    assert!(name.contains("moe_chain"), "{name}");
    arguments.push(KernelArgumentV1::DeviceBuffer(on.intermediate[*rank]));
    let message = order(&binding).unwrap();
    assert!(message.contains("ExpertsGateUp"), "{message}");
    // The same launch binding the MLP output the shared down projection
    // writes, or the side stream's projection scratch.
    for span in [on.mlp_out[*rank], on.scratch[*rank][1][0]] {
        let mut binding = on.mutated(on.journal.clone());
        binding.args[first + issued]
            .2
            .push(KernelArgumentV1::DeviceBuffer(span));
        let message = order(&binding).unwrap();
        assert!(message.contains("ExpertsGateUp"), "{message}");
    }
    // Two compute launches swapped.
    let mut swapped = on.journal.clone();
    let launches = swapped
        .iter()
        .enumerate()
        .filter(|(_, line)| stream_of(line, "launch").is_some_and(|stream| !on.streams[&stream].1))
        .map(|(index, _)| index)
        .take(2)
        .collect::<Vec<_>>();
    swapped.swap(launches[0], launches[1]);
    let message = failure(|| {
        check_equivalent(&off, &on.mutated(swapped), true);
    })
    .unwrap();
    assert!(
        message.contains("is not the next on either stream"),
        "{message}"
    );
    // Replays: each step's run replays 4 ranks' 3 pieces piece-major.
    let equivalent = |trace: &Trace| {
        failure(|| {
            check_equivalent(&off, trace, true);
        })
    };
    assert_eq!(equivalent(&on), None);
    let runs = replay_runs(&on);
    assert_eq!(runs.len(), 3);
    assert!(runs.iter().all(|run| run.len() == 12));
    let graph = |line: usize| on.journal[line].rsplit(' ').next().unwrap().to_owned();
    // Rank 0's piece at `run`, `piece`: its journal line.
    let at = |run: usize, piece: usize| runs[run][4 * piece];
    for (mutant, edits, expected) in [
        (
            "rank 0 replays its first piece twice and never its second",
            vec![(at(0, 1), graph(at(0, 0)))],
            "is not every piece of its groups",
        ),
        (
            "rank 0 replays its first two pieces in reverse",
            vec![(at(0, 0), graph(at(0, 1))), (at(0, 1), graph(at(0, 0)))],
            "is not every piece of its groups",
        ),
        (
            "rank 0 replays a graph nobody captured",
            vec![(at(0, 2), "4000000000".to_owned())],
            "replay of unknown graph",
        ),
        (
            "the one-item decode step's rank 0 replays the two-item bucket's graphs",
            (0..3)
                .map(|piece| (at(2, piece), graph(at(0, piece))))
                .collect(),
            "a replay of another cache's graphs",
        ),
    ] {
        let mut journal = on.journal.clone();
        for (line, graph) in edits {
            let mut words = journal[line]
                .split(' ')
                .map(str::to_owned)
                .collect::<Vec<_>>();
            *words.last_mut().unwrap() = graph;
            journal[line] = words.join(" ");
        }
        let message = equivalent(&on.mutated(journal)).unwrap_or_else(|| panic!("{mutant} passes"));
        assert!(message.contains(expected), "{mutant}: {message}");
    }
}

/// The journal lines of each run of consecutive replays.
fn replay_runs(trace: &Trace) -> Vec<Vec<usize>> {
    let mut runs: Vec<Vec<usize>> = Vec::new();
    for (index, line) in trace.journal.iter().enumerate() {
        if !line.starts_with("replay ") {
            continue;
        }
        match runs.last_mut() {
            Some(run) if run.last() == Some(&(index - 1)) => run.push(index),
            _ => runs.push(vec![index]),
        }
    }
    runs
}

/// `on` runs `off`'s program with the overlap ([`check_equivalent`], its
/// replays bound to their own runs' graphs) in the order
/// [`check_order`] proves; returns the overlapped sublayers.
fn check_overlap(off: &Trace, on: &Trace) -> usize {
    assert!(off.captures.is_empty() && off.events.is_empty());
    check_equivalent(off, on, true);
    let regions = check_order(on);
    assert!(regions >= 4, "{regions}");
    regions
}

/// The configurations whose pinned journals the default overlap re-pinned
/// hold the same proof: context-parallel groups of 2 and 4 (the served
/// target steps, and the drafter's passes), the mixed layouts (decode,
/// verify and the drafter), and the Sinkhorn mHC comm epilogue.
#[test]
fn context_parallel_groups_overlap_alike() {
    use crate::context_parallel_served_tests::{record, served_steps};
    for size in [2, 4] {
        for drafting in [false, true] {
            let format = Shape::TP4.kv_formats()[usize::from(drafting)];
            let trace = |settings| {
                trace_with(
                    settings,
                    FakeHost::with_format((3, 32), format),
                    |host| {
                        {
                            api().create(
                                HostServicesTableV1::new(Rc::clone(host)),
                                &record(size, format),
                            )
                        }
                        .unwrap_or_else(|error| panic!("{error:?} {:?}", host.logs.borrow()))
                    },
                    |host, instance| {
                        let at = host.device.journal.borrow().len();
                        served_steps(host, instance, size, drafting);
                        vec![at]
                    },
                )
            };
            let regions = check_overlap(&trace(Settings::OFF), &trace(ALL));
            eprintln!("size {size} drafting {drafting}: {regions} sublayers");
        }
    }
}

/// See [`context_parallel_groups_overlap_alike`].
#[test]
fn mixed_layouts_overlap_alike() {
    use super::nvfp4::{DENSE_ROUTED, FP8_ROUTED, Mix, ROUTED};
    for export in [&ROUTED, &DENSE_ROUTED, &FP8_ROUTED] {
        for (scenario, host) in [
            (Scenario::Target, Mix::host as fn(&Mix) -> Rc<Host>),
            (Scenario::Drafter, Mix::mtp_host),
        ] {
            let trace = |settings| {
                trace_with(
                    settings,
                    host(export),
                    |host| export.create(host),
                    |host, instance| run_steps(host, instance, scenario),
                )
            };
            let regions = check_overlap(&trace(Settings::OFF), &trace(ALL));
            eprintln!("{} {scenario:?}: {regions} sublayers", export.layout.id);
        }
    }
}

/// See [`context_parallel_groups_overlap_alike`].
#[test]
fn the_comm_epilogue_overlaps_alike() {
    let format = format(Scenario::Target);
    let trace = |settings| {
        trace_with(
            settings,
            FakeHost::with_format((3, 32), format),
            |host| {
                let roles = [crate::projections::COMPONENT_ROLE.as_bytes().to_vec()];
                let options = crate::config::tests::options_comm_epilogue(2048, "true");
                let record = record_naming(DECODE_BATCH, &options, format, &roles);
                api()
                    .create(HostServicesTableV1::new(Rc::clone(host)), &record)
                    .unwrap_or_else(|error| panic!("{error:?} {:?}", host.logs.borrow()))
            },
            |host, instance| run_steps(host, instance, Scenario::Target),
        )
    };
    assert_eq!(check_overlap(&trace(Settings::OFF), &trace(ALL)), 12 * 42);
}
