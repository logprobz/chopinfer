//! Scratch regions (`buffers::REGIONS`) alias buffers whose contents are
//! never live at once. These tests serve instances whose scratch is not
//! aliased on the fake host, then replay each journal, create included,
//! over the bytes a grouping would share.
//!
//! Every launch argument, copy, upload, zero, gather and download touches
//! the span it names. A `const` entry parameter only reads. A host copy,
//! upload, zero or gather writes its whole destination. A kernel argument
//! that `OUTPUTS` lists writes every byte of its span that a later touch
//! reads, as its entry's output, and a `SPLIT_OUTPUTS` pair of launches
//! does so together when the second follows the first on its stream with
//! no other touch of the span between them. Any other kernel argument may
//! read and write. A grouping is refused when a member reads bytes that
//! another member wrote; when one launch binds two members, which its
//! threads then use at once; when two members touch the region without a
//! happens-before order between them (stream order, recorded and awaited
//! events, host synchronization, and the fork and join inside a captured
//! graph, which runs at each replay); or when a member is a destination of
//! an asynchronous host gather, whose writes the journal does not order.

use std::collections::{BTreeMap, BTreeSet};

use bundle_runtime::scratch::Region;
use runtime_bundle_sdk::prefill::{Boundary, encode};

use super::*;

type Span = (u64, u64, u64);
type Scratch = Vec<Vec<(String, DeviceBufferSpanV1)>>;

/// Destinations of asynchronous host gathers (prefill Engram and the
/// DSpark cycle): the journal names neither their bytes nor when the host
/// writes them.
const ASYNC_TARGETS: [&str; 2] = ["engram_codes", "engram_scales"];

/// A symbol pattern and the index of one of its entries' arguments.
type Listed = (&'static str, usize);

/// Kernel arguments that write every byte of their span a later touch
/// reads, without reading what it held: the argument at `index` of the
/// entries whose symbols match `pattern` (`*` stands for any text, `#` for
/// a number). Each was checked against its entry's source (2026-10-23).
/// Every one writes all of its row's columns for each row below the step's
/// device live count and leaves the span's rows past it, and every reader
/// of these buffers stops at the same live count, so the kill covers what
/// is read.
const OUTPUTS: &[Listed] = &[
    // MXFP8 fold projections: the merging CTA or launch stores each output
    // element once, zero for a refused row; split partials stay in the
    // workspace.
    ("chop_deepseek41_mxfp8_stream_*_f32_v1", 6),
    ("chop_deepseek41_mxfp8_prefetch_*_f32_v3", 6),
    ("chop_deepseek41_mxfp8_ticket_*_f32_v2", 6),
    ("chop_deepseek41_mxfp8_direct_*_single_n8k4_v1", 6),
    ("chop_deepseek41_mxfp8_direct_*_single_n32k1_v1", 6),
    ("chop_deepseek41_mxfp8_fold_m*_f32_v2", 4),
    ("chop_deepseek41_mxfp8_fold_m*_f32_v2_tma_scales_s2_v1", 6),
    ("chop_deepseek41_mxfp8_fold_m*_f32_v2_prefill_tma_v1", 6),
    (
        "chop_deepseek41_mxfp8_fold_m*_f32_v2_matrix_m#_n128_s#_v1",
        4,
    ),
    ("chop_deepseek41_mxfp8_decode_*_f32_v1", 1),
    // The status folds (`sm120::fold_fused`): the cluster form's owning split
    // and the 48-column row-local form's thread store each output element
    // once, zero for a failed row. WO_a's also stores every block of each
    // live row's WO_b input codes, scale codes and flags.
    (
        "chop_deepseek41_mxfp8_fused_*_cluster#_n#_c#s#_status_v1",
        4,
    ),
    (
        "chop_deepseek41_mxfp8_fused_*_cluster#_n#_c#s#_statusp_v1",
        4,
    ),
    (
        "chop_deepseek41_mxfp8_fused_*_cluster#_n#_c#s#_statusq_v1",
        4,
    ),
    (
        "chop_deepseek41_mxfp8_fused_*_cluster#_n#_c#s#_statusq_v1",
        5,
    ),
    (
        "chop_deepseek41_mxfp8_fused_*_cluster#_n#_c#s#_statusq_v1",
        6,
    ),
    (
        "chop_deepseek41_mxfp8_fused_*_cluster#_n#_c#s#_statusq_v1",
        7,
    ),
    ("chop_deepseek41_mxfp8_fused_*_local#_n#_s#x_status_v1", 6),
    // The one-row query and KV folds in one launch: each projection's
    // stream task merges into its own output, as above.
    ("chop_mxfp8_dual_*_f32_v1", 4),
    ("chop_mxfp8_dual_*_f32_v1", 9),
    // The experts' down projection into each live route's row of `routed`.
    ("chop_moe_native_down_*_tiles_v1", 9),
    // Its `Scanned` policy, behind the admission scan: the same tile stores.
    ("chop_moe_native_down_*_tiles_scanned_v1", 9),
    ("chop_deepseek41_moe_floor_*_down_finish_v1", 4),
    (
        "deepseek41_chop_moe_chain_mxfp4_a8ceil_before_down_*_r16_t32_v1",
        9,
    ),
    (
        "deepseek41_chop_moe_chain_mxfp4_a8ceil_before_down_*_r64_t8192_v1",
        9,
    ),
    // Latent attention's merge of the workspace partials, or the fused
    // heads' finish, into the BF16 attention output.
    ("chop_latent_sources_deepseek41_*_merge_v1", 2),
    ("chop_latent_sources_deepseek41_*_head_shared_finish_v1", 27),
    (
        "chop_latent_sources_deepseek41_*_decoded_head_async_finish_v1",
        27,
    ),
    // The index scores of every slot of a live row (padding as -inf), and
    // the hierarchical merge's group lists in the scores' tail.
    (
        "chop_position_blocks_deepseek41_r*_scores_v3_valid_v1_mma_fp4_guard_fast_v1",
        13,
    ),
    (
        "chop_position_blocks_deepseek41_r*_scores_v3_valid_v1_mma_fp4_prefill_tile_v1",
        13,
    ),
    (
        "chop_position_blocks_deepseek41_r*_bf16_scores_v3_valid_v1",
        13,
    ),
    (
        "chop_position_blocks_deepseek41_r*_select_v3_valid_merge_v1_hier_group_v1",
        17,
    ),
    (
        "chop_position_indexer_deepseek41_r*_all_scores_v3_live_v1_mma_fp4_guard_fast_v1",
        9,
    ),
    (
        "chop_position_indexer_deepseek41_r*_all_scores_v3_live_v1_mma_fp4_tile4_mode0_trace0_v1",
        9,
    ),
    (
        "chop_position_indexer_deepseek41_r*_all_bf16_scores_v3_live_v1",
        9,
    ),
    (
        "chop_position_indexer_deepseek41_r*_select_v3_live_merge_v1_hier_group_v1",
        13,
    ),
    // The query RoPE's BF16 attention query, every column of every head.
    ("chop_deepseek41_latent_query_rope_v1", 1),
    ("chop_deepseek41_latent_query_rope_heads_v1", 1),
    // One Engram layer's table rows decoded to BF16.
    ("chop_row_mxfp8_decode_v1", 3),
    // The final weighted collapse of the streams.
    ("chop_sinkhorn_hc_deepseek41_weighted_head_f32_v1", 2),
    // The rank's embedding rows, zero for other shards' ids.
    ("chop_row_embedding_deepseek41_tp*_v1", 2),
    ("chop_deepseek41_embedding_replicated_tp*_v1", 2),
];

/// Kernel arguments that write every byte of their span a later touch
/// reads only as a pair of launches, neither reading what it held: (symbol
/// pattern, argument index) of the first launch and of the second. A split
/// prefill fold writes its full 128-row tiles, then its tail rows.
const SPLIT_OUTPUTS: &[(Listed, Listed)] = &[(
    (
        "chop_deepseek41_mxfp8_fold_m*_f32_v2_prefill_tma_v1_split_full_v1",
        6,
    ),
    (
        "chop_deepseek41_mxfp8_fold_m*_f32_v2_prefill_tma_v1_split_tail_v1",
        4,
    ),
)];

/// Whether `symbol` matches `pattern`, in which `*` stands for any text and
/// `#` for a decimal number.
fn matches(pattern: &str, symbol: &str) -> bool {
    match pattern.as_bytes().first() {
        None => symbol.is_empty(),
        Some(b'*') => (0..=symbol.len()).any(|at| matches(&pattern[1..], &symbol[at..])),
        Some(b'#') => {
            let digits = symbol.bytes().take_while(u8::is_ascii_digit).count();
            (1..=digits).any(|at| matches(&pattern[1..], &symbol[at..]))
        }
        Some(&c) => symbol.as_bytes().first() == Some(&c) && matches(&pattern[1..], &symbol[1..]),
    }
}

/// How a touch accesses its span.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Access {
    Read,
    Write,
    Modify,
    /// The first launch of a `SPLIT_OUTPUTS` pair.
    Begin,
    /// The second launch of a `SPLIT_OUTPUTS` pair, which writes the whole
    /// span with the first only when it follows that on its stream and
    /// nothing touched the span between them.
    Finish,
}

/// Whether argument `argument` of `symbol` is the index of an entry of
/// `table` whose pattern the symbol matches. A packed fold entry has its
/// row-major twin's arguments (`gemm::mxfp8_fold_packed`).
fn listed<'a>(mut table: impl Iterator<Item = &'a Listed>, symbol: &str, argument: usize) -> bool {
    let symbol = crate::sm120::test_symbols::row_major_fold_symbol(symbol).unwrap_or(symbol);
    table.any(|&(pattern, index)| index == argument && matches(pattern, symbol))
}

fn access(symbol: &str, argument: usize) -> Access {
    if crate::Kernels::const_input_argument(symbol, argument) {
        Access::Read
    } else if listed(OUTPUTS.iter(), symbol, argument) {
        Access::Write
    } else if listed(
        SPLIT_OUTPUTS.iter().map(|(first, _)| first),
        symbol,
        argument,
    ) {
        Access::Begin
    } else if listed(
        SPLIT_OUTPUTS.iter().map(|(_, second)| second),
        symbol,
        argument,
    ) {
        Access::Finish
    } else {
        Access::Modify
    }
}

/// One journal operation that touches device memory or orders streams.
#[derive(Clone, Debug)]
enum Op {
    Touch {
        line: usize,
        stream: u32,
        what: String,
        touches: Vec<(Span, Access)>,
    },
    Record {
        stream: u32,
        event: u32,
    },
    Wait {
        stream: u32,
        event: u32,
    },
    Synchronize {
        stream: u32,
    },
    Capture {
        stream: u32,
    },
    Captured {
        stream: u32,
        graph: u32,
    },
    Replay {
        stream: u32,
        graph: u32,
    },
}

/// The stream, symbol and each argument's spans of a launch line.
fn launch_arguments(line: &str) -> Option<(u32, String, Vec<Vec<Span>>)> {
    let rest = line.strip_prefix("launch ")?;
    let (stream, rest) = rest.split_once(' ')?;
    let (symbol, rest) = rest.split_once(' ')?;
    // The argument list is the first bracket outside the launch config.
    let mut depth = 0_i32;
    let mut start = None;
    for (at, c) in rest.char_indices() {
        match c {
            '[' if depth == 0 => {
                start = Some(at);
                break;
            }
            '{' | '(' | '[' => depth += 1,
            '}' | ')' | ']' => depth -= 1,
            _ => {}
        }
    }
    let list = &rest[start? + 1..];
    let mut arguments = Vec::new();
    let mut depth = 0_i32;
    let mut from = 0;
    for (at, c) in list.char_indices() {
        match c {
            ']' if depth == 0 => {
                if at > from {
                    arguments.push(&list[from..at]);
                }
                break;
            }
            '{' | '(' | '[' => depth += 1,
            '}' | ')' | ']' => depth -= 1,
            ',' if depth == 0 => {
                arguments.push(&list[from..at]);
                from = at + 2;
            }
            _ => {}
        }
    }
    Some((
        stream.parse().ok()?,
        symbol.to_owned(),
        arguments
            .into_iter()
            .map(super::shared_overlap::journal_spans)
            .collect(),
    ))
}

/// The operation of journal line `line`, or none for a line that neither
/// touches device memory nor orders streams. A line that names spans and is
/// not classified here fails the test.
fn parse(line: usize, text: &str) -> Option<Op> {
    // The rank image's upload names no scratch span.
    if text.starts_with("upload arena ") {
        return None;
    }
    let words: Vec<&str> = text.split_whitespace().collect();
    let number = |at: usize| -> u32 { words[at].parse().unwrap() };
    let spans = super::shared_overlap::journal_spans(text);
    let touch = |stream: u32, touches: Vec<(Span, Access)>| Op::Touch {
        line,
        stream,
        what: words[0].to_owned(),
        touches,
    };
    Some(match words.first().copied()? {
        "launch" => {
            let (stream, symbol, arguments) = launch_arguments(text).unwrap();
            let touches = arguments
                .iter()
                .enumerate()
                .flat_map(|(index, spans)| {
                    let access = access(&symbol, index);
                    spans.iter().map(move |&span| (span, access))
                })
                .collect();
            Op::Touch {
                line,
                stream,
                what: symbol,
                touches,
            }
        }
        "copy" | "peer-copy" => {
            let [source, destination] = spans[..] else {
                panic!("a copy names two spans: {text}");
            };
            touch(
                number(1),
                vec![(source, Access::Read), (destination, Access::Write)],
            )
        }
        "zero" | "upload" | "gather" => {
            let [destination] = spans[..] else {
                panic!("a write names one span: {text}");
            };
            touch(number(1), vec![(destination, Access::Write)])
        }
        "download" => {
            let [source] = spans[..] else {
                panic!("a download names one span: {text}");
            };
            touch(number(1), vec![(source, Access::Read)])
        }
        "record" => Op::Record {
            stream: number(1),
            event: number(2),
        },
        "wait" => Op::Wait {
            stream: number(1),
            event: number(2),
        },
        "synchronize" => Op::Synchronize { stream: number(1) },
        "capture" => Op::Capture { stream: number(1) },
        "captured" => Op::Captured {
            stream: number(1),
            graph: number(2),
        },
        "replay" => Op::Replay {
            stream: number(1),
            graph: number(2),
        },
        _ => {
            assert!(spans.is_empty(), "unclassified journal line: {text}");
            return None;
        }
    })
}

/// A vector clock: the latest touch of each stream an operation follows.
type Clock = BTreeMap<u32, u64>;

fn join(into: &mut Clock, from: &Clock) {
    for (&stream, &tick) in from {
        let at = into.entry(stream).or_default();
        *at = (*at).max(tick);
    }
}

/// One rank's region: which member last wrote each byte range, each
/// member's latest touch on each stream, and the split writes that wait
/// for their second launch (range, member and stream).
#[derive(Default)]
struct RegionState {
    owners: Vec<(u64, u64, usize)>,
    latest: BTreeMap<(usize, u32), u64>,
    splits: Vec<(u64, u64, usize, u32)>,
}

/// A region member's buffer on one rank.
#[derive(Clone, Copy)]
struct Member {
    rank: usize,
    region: usize,
    member: usize,
    base: u64,
}

/// Replays a journal over a grouping's shared bytes.
struct Check<'a> {
    regions: &'a [Region],
    members: BTreeMap<u64, Member>,
    streams: BTreeMap<u32, Clock>,
    events: BTreeMap<u32, Clock>,
    host: Clock,
    states: BTreeMap<(usize, usize), RegionState>,
    capturing: BTreeMap<u32, Vec<Op>>,
    capture_of: BTreeMap<u32, u32>,
    captured_events: BTreeMap<u32, u32>,
    graphs: BTreeMap<u32, (u32, Vec<Op>)>,
    fresh: u32,
    /// Each refusal once, with the first journal line that showed it.
    refusals: BTreeMap<String, String>,
}

impl<'a> Check<'a> {
    fn new(regions: &'a [Region], scratch: &Scratch) -> Self {
        let mut check = Self {
            regions,
            members: BTreeMap::new(),
            streams: BTreeMap::new(),
            events: BTreeMap::new(),
            host: Clock::new(),
            states: BTreeMap::new(),
            capturing: BTreeMap::new(),
            capture_of: BTreeMap::new(),
            captured_events: BTreeMap::new(),
            graphs: BTreeMap::new(),
            fresh: 1 << 30,
            refusals: BTreeMap::new(),
        };
        for (region, declared) in regions.iter().enumerate() {
            for (member, &name) in declared.members.iter().enumerate() {
                if ASYNC_TARGETS.contains(&name) {
                    check.refuse(
                        format!("{name} is written by asynchronous host gathers"),
                        String::new(),
                    );
                }
                for (rank, named) in scratch.iter().enumerate() {
                    let Some(&(_, span)) = named.iter().find(|(path, _)| path == name) else {
                        check.refuse(format!("{name} names no scratch buffer"), String::new());
                        continue;
                    };
                    let previous = check.members.insert(
                        span.buffer.raw,
                        Member {
                            rank,
                            region,
                            member,
                            base: span.offset,
                        },
                    );
                    if previous.is_some() {
                        check.refuse(format!("{name} is a member twice"), String::new());
                    }
                }
            }
        }
        check
    }

    fn refuse(&mut self, refusal: String, example: String) {
        self.refusals.entry(refusal).or_insert(example);
    }

    fn fresh(&mut self) -> u32 {
        self.fresh += 1;
        self.fresh
    }

    /// Issues `op` at the host, recording it into the capture its stream
    /// is part of instead of running it.
    fn issue(&mut self, op: Op) {
        match op {
            Op::Capture { stream } => {
                self.capturing.insert(stream, Vec::new());
                self.capture_of.insert(stream, stream);
            }
            Op::Captured { stream, graph } => {
                let ops = self.capturing.remove(&stream).expect("a capture began");
                self.graphs.insert(graph, (stream, ops));
                self.capture_of.retain(|_, capture| *capture != stream);
                self.captured_events.retain(|_, capture| *capture != stream);
            }
            Op::Replay { stream, graph } => self.replay(stream, graph),
            Op::Record { stream, event } => match self.capture_of.get(&stream).copied() {
                Some(capture) => {
                    self.captured_events.insert(event, capture);
                    self.capturing.get_mut(&capture).unwrap().push(op);
                }
                None => {
                    self.captured_events.remove(&event);
                    self.run(op);
                }
            },
            Op::Wait { stream, event } => {
                // Waiting on a captured event joins the stream to its capture.
                if let Some(&capture) = self.captured_events.get(&event) {
                    self.capture_of.insert(stream, capture);
                }
                match self.capture_of.get(&stream) {
                    Some(capture) => self.capturing.get_mut(capture).unwrap().push(op),
                    None => self.run(op),
                }
            }
            Op::Touch { stream, .. } | Op::Synchronize { stream } => {
                match self.capture_of.get(&stream) {
                    Some(capture) => self.capturing.get_mut(capture).unwrap().push(op),
                    None => self.run(op),
                }
            }
        }
    }

    /// Runs graph `graph` on `stream`: its captured operations in capture
    /// order, the capture's own stream as `stream` and every other stream
    /// and event of the graph as a new one that starts after `stream`'s
    /// earlier work; `stream`'s later work follows all of them.
    fn replay(&mut self, stream: u32, graph: u32) {
        let (captured, ops) = self.graphs.get(&graph).expect("a captured graph").clone();
        let mut start = self.streams.get(&stream).cloned().unwrap_or_default();
        join(&mut start, &self.host);
        let mut streams = BTreeMap::from([(captured, stream)]);
        let mut events = BTreeMap::new();
        for op in ops {
            let mut on = |of: u32, check: &mut Self| {
                *streams.entry(of).or_insert_with(|| {
                    let id = check.fresh();
                    check.streams.insert(id, start.clone());
                    id
                })
            };
            let mut event =
                |of: u32, check: &mut Self| *events.entry(of).or_insert_with(|| check.fresh());
            let op = match op {
                Op::Touch {
                    line,
                    stream,
                    what,
                    touches,
                } => Op::Touch {
                    line,
                    stream: on(stream, self),
                    what,
                    touches,
                },
                Op::Record { stream, event: of } => Op::Record {
                    stream: on(stream, self),
                    event: event(of, self),
                },
                Op::Wait { stream, event: of } => Op::Wait {
                    stream: on(stream, self),
                    event: event(of, self),
                },
                other => panic!("graph {graph} holds {other:?}"),
            };
            self.run(op);
        }
        let mut done = self.streams.get(&stream).cloned().unwrap_or_default();
        for id in streams.values() {
            join(&mut done, &self.streams[id].clone());
        }
        self.streams.insert(stream, done);
    }

    fn run(&mut self, op: Op) {
        match op {
            Op::Touch {
                line,
                stream,
                what,
                touches,
            } => {
                let clock = self.streams.entry(stream).or_default();
                join(clock, &self.host);
                *clock.entry(stream).or_default() += 1;
                let now = clock.clone();
                let example = format!("line {line}: {what}");
                // One launch's threads use every member it binds at once.
                let mut bound = BTreeMap::<(usize, usize), BTreeSet<usize>>::new();
                for (span, _) in &touches {
                    if let Some(member) = self.members.get(&span.0).filter(|_| span.2 > 0) {
                        bound
                            .entry((member.rank, member.region))
                            .or_default()
                            .insert(member.member);
                    }
                }
                for ((_, region), members) in bound {
                    if members.len() > 1 {
                        let names: Vec<_> = members
                            .iter()
                            .map(|&member| self.regions[region].members[member])
                            .collect();
                        self.refuse(
                            format!("one launch binds {}", names.join(" and ")),
                            example.clone(),
                        );
                    }
                }
                for (span, access) in touches {
                    self.touch(&example, stream, &now, span, access);
                }
            }
            Op::Record { stream, event } => {
                let clock = self.streams.entry(stream).or_default();
                join(clock, &self.host);
                self.events.insert(event, clock.clone());
            }
            Op::Wait { stream, event } => {
                let recorded = self.events.get(&event).cloned().unwrap_or_default();
                join(self.streams.entry(stream).or_default(), &recorded);
            }
            Op::Synchronize { stream } => {
                let clock = self.streams.get(&stream).cloned().unwrap_or_default();
                join(&mut self.host, &clock);
            }
            other => unreachable!("{other:?} is issued, not run"),
        }
    }

    fn touch(&mut self, example: &str, stream: u32, now: &Clock, span: Span, access: Access) {
        let Some(&member) = self.members.get(&span.0) else {
            return;
        };
        if span.2 == 0 {
            return;
        }
        let lo = span.1 - member.base;
        let hi = lo + span.2;
        let region = &self.regions[member.region];
        let name = region.members[member.member];
        let state = self.states.entry((member.rank, member.region)).or_default();
        let mut refusals = Vec::new();
        for (&(other, on), &tick) in &state.latest {
            if other != member.member && now.get(&on).copied().unwrap_or(0) < tick {
                refusals.push(format!(
                    "{name} and {} touch region {} unordered",
                    region.members[other], region.name
                ));
            }
        }
        state.latest.insert((member.member, stream), now[&stream]);
        let split = (lo, hi, member.member, stream);
        let finishes = access == Access::Finish && state.splits.contains(&split);
        for &(a, b, writer, _) in &state.splits {
            if a < hi && lo < b && !finishes {
                refusals.push(format!(
                    "{name} touches bytes of a split write of {} before its second launch in region {}",
                    region.members[writer], region.name
                ));
            }
        }
        let access = match access {
            Access::Begin => {
                state.splits.push(split);
                Access::Write
            }
            Access::Finish if finishes => {
                state.splits.retain(|&pending| pending != split);
                Access::Write
            }
            Access::Finish => Access::Modify,
            access => access,
        };
        if access != Access::Write {
            for &(a, b, owner) in &state.owners {
                if owner != member.member && a < hi && lo < b {
                    refusals.push(format!(
                        "{name} reads bytes {} wrote in region {}",
                        region.members[owner], region.name
                    ));
                }
            }
        }
        if access != Access::Read {
            let mut owners = Vec::with_capacity(state.owners.len() + 2);
            for &(a, b, owner) in &state.owners {
                if b <= lo || hi <= a {
                    owners.push((a, b, owner));
                    continue;
                }
                if a < lo {
                    owners.push((a, lo, owner));
                }
                if hi < b {
                    owners.push((hi, b, owner));
                }
            }
            owners.push((lo, hi, member.member));
            state.owners = owners;
        }
        for refusal in refusals {
            self.refuse(refusal, example.to_owned());
        }
    }
}

/// What a check of `regions` over `journal` refused, each with an example;
/// empty when the grouping holds.
fn check(regions: &[Region], scratch: &Scratch, journal: &[String]) -> BTreeMap<String, String> {
    let mut check = Check::new(regions, scratch);
    for (line, text) in journal.iter().enumerate() {
        if let Some(op) = parse(line, text) {
            check.issue(op);
        }
    }
    assert!(check.capturing.is_empty(), "a capture never ended");
    let unfinished: Vec<_> = check
        .states
        .iter()
        .flat_map(|(&(_, region), state)| {
            state
                .splits
                .iter()
                .map(move |&(_, _, member, _)| (region, member))
        })
        .collect();
    for (region, member) in unfinished {
        let region = &regions[region];
        check.refuse(
            format!(
                "a split write of {} has no second launch in region {}",
                region.members[member], region.name
            ),
            String::new(),
        );
    }
    check.refusals
}

/// The unaliased scratch of each rank and the whole journal of the
/// instance `create` makes on `host`, through `steps`.
fn trace(
    host: &Rc<Host>,
    create: impl FnOnce(&Rc<Host>) -> BundleInstanceV1,
    steps: impl FnOnce(&mut BundleInstanceV1),
) -> (Scratch, Vec<String>) {
    host.kv_pages.set(1024);
    let mut instance = crate::buffers::unaliased(|| create(host));
    let scratch = crate::instance::scratch_for_test(instance.handle());
    steps(&mut instance);
    instance.destroy().unwrap();
    assert_eq!(host.acquired.get(), host.released.get());
    let journal = host.device.journal.borrow().clone();
    (scratch, journal)
}

/// Short prompts in three slots, decodes of each bucket that holds them
/// (each captured once, then replayed), and a scored continuation.
fn short_steps(instance: &mut BundleInstanceV1) {
    let mut views = Views::default();
    step(instance, &prompt(7, 0, 0, 64, false), &mut views);
    step(instance, &prompt(7, 0, 64, 6, true), &mut views);
    step(instance, &prompt(9, 1, 0, 3, true), &mut views);
    step(
        instance,
        &decode(&[(7, 0, 70, 1), (9, 1, 3, 2)]),
        &mut views,
    );
    step(
        instance,
        &decode(&[(7, 0, 71, 1), (9, 1, 4, 2)]),
        &mut views,
    );
    step(instance, &decode(&[(7, 0, 72, 1)]), &mut views);
    step(instance, &prompt(11, 2, 0, 5, true), &mut views);
    step(
        instance,
        &decode(&[(7, 0, 73, 1), (9, 1, 5, 2), (11, 2, 5, 3)]),
        &mut views,
    );
}

/// Two full prefill chunks of one request, then a decode.
fn long_steps(instance: &mut BundleInstanceV1) {
    let chunk = u64::from(model_deepseek41_contract::serving::PREFILL_TOKENS);
    let mut views = Views::default();
    step(instance, &prompt(7, 0, 0, chunk, false), &mut views);
    step(instance, &prompt(7, 0, chunk, chunk, true), &mut views);
    step(instance, &decode(&[(7, 0, 2 * chunk, 1)]), &mut views);
}

type Sink = bundle_runtime::fake_host::RecordingSink<super::conformance::Limits>;

#[track_caller]
fn execute(instance: &mut BundleInstanceV1, batch: &[u8]) {
    if let Err(error) = instance.execute(batch, &mut Sink::default()) {
        panic!("{error:?}: {:?}", instance.last_execute_refusal());
    }
}

/// A DSpark prompt, two verify steps that draft, and a decode.
fn draft_steps(instance: &mut BundleInstanceV1) {
    use runtime_bundle_sdk::{BatchAppendV1, BatchItemV1, BatchRequestV1, BatchWireVersionV1};
    let verify = |start: u64, drafts: u32| {
        let rows = u64::from(drafts + 1);
        BatchRequestV1::new(
            BatchWireVersionV1::V3,
            vec![BatchItemV1::verify(7, drafts + 1, true, vec![7])],
            vec![BatchAppendV1::new(
                start,
                rows,
                start + rows,
                (0..(start + rows).div_ceil(64)).collect(),
                None,
            )],
        )
        .with_state_slots(vec![0])
        .encode()
        .unwrap()
    };
    execute(instance, &prompt(7, 0, 0, 3, true));
    instance.commit().unwrap();
    execute(instance, &verify(3, 5));
    instance.commit_accepted(&[super::kept(7, 6)]).unwrap();
    execute(instance, &verify(9, 5));
    instance.commit_accepted(&[super::kept(7, 2)]).unwrap();
    execute(instance, &decode(&[(7, 0, 11, 7)]));
    instance.commit().unwrap();
}

/// One full prefill chunk through the prefill API, then a decode.
fn profile_steps(instance: &mut BundleInstanceV1) {
    let chunk = model_deepseek41_contract::serving::PREFILL_TOKENS;
    let request = encode(
        &prompt(7, 0, 0, u64::from(chunk), true),
        &[Boundary {
            request_id: 7,
            prompt_tokens: u64::from(chunk),
            chunk_end: u64::from(chunk),
            chunk_tokens: chunk,
        }],
    )
    .unwrap();
    if let Err(error) = instance.execute_prefill(&request, &mut Sink::default()) {
        panic!("{error:?}: {:?}", instance.last_execute_refusal());
    }
    instance.commit().unwrap();
    execute(instance, &decode(&[(7, 0, u64::from(chunk), 1)]));
    instance.commit().unwrap();
}

/// Runs `each` over every traced lifecycle: each topology and KV layout's
/// short lifecycle; DSpark drafting at each topology; full prefill chunks;
/// and at four ranks the opt-in prefill schedules (overlapped parts, layer
/// micro parts, DMA reduction, and the upper-skip profile with
/// asynchronous Engram gathers).
fn lifecycles(mut each: impl FnMut(&str, &Scratch, &[String])) {
    let tp4 = Shape::new(&FLASH, Ranks::TP4).unwrap();
    for shape in [Ranks::TP1, Ranks::TP2, Ranks::TP4].map(|r| Shape::new(&FLASH, r).unwrap()) {
        for kv in KV_LAYOUTS {
            let host = FakeHost::new(shape, kv);
            let (scratch, journal) = trace(&host, |host| create(host, 4), short_steps);
            each(
                &format!("{} {:?}", shape.topology(), kv.storage),
                &scratch,
                &journal,
            );
        }
        let host = FakeHost::new(shape, KV_LAYOUTS[0]);
        host.speculation.set(Some((5, 32)));
        let (scratch, journal) = trace(&host, |host| create(host, 4), draft_steps);
        each(&format!("{} DSpark", shape.topology()), &scratch, &journal);
    }
    for kv in KV_LAYOUTS {
        let host = FakeHost::new(tp4, kv);
        let (scratch, journal) = trace(&host, |host| create(host, 1), long_steps);
        each(&format!("full chunks {:?}", kv.storage), &scratch, &journal);
    }
    let host = FakeHost::new(tp4, KV_LAYOUTS[0]);
    let overlapped =
        |host: &Rc<Host>| crate::instance::execute::overlap::with_enabled(true, || create(host, 4));
    let (scratch, journal) = trace(&host, overlapped, long_steps);
    each("overlapped prefill", &scratch, &journal);
    let host = FakeHost::new(tp4, KV_LAYOUTS[0]);
    let micro =
        |host: &Rc<Host>| crate::instance::execute::overlap::with_mode(2, || create(host, 4));
    let (scratch, journal) = trace(&host, micro, long_steps);
    each("micro prefill", &scratch, &journal);
    for mode in [0, 2] {
        let host = FakeHost::new(tp4, KV_LAYOUTS[0]);
        host.device
            .ordinary_symbols
            .borrow_mut()
            .insert("chop_comm_owned_sum_round_f32_v1".into());
        let dma = |host: &Rc<Host>| {
            crate::ops::with_dma(true, || {
                crate::instance::execute::overlap::with_mode(mode, || create(host, 4))
            })
        };
        let (scratch, journal) = trace(&host, dma, long_steps);
        each(&format!("DMA prefill mode {mode}"), &scratch, &journal);
    }
    let host = FakeHost::new(tp4, KV_LAYOUTS[0]);
    let profile = |host: &Rc<Host>| {
        crate::instance::prefill_engram::with_enabled(true, || {
            super::prefill_skip::create_profile(host, 8192, true)
        })
    };
    let (scratch, journal) = trace(&host, profile, profile_steps);
    each("upper skip with async Engram", &scratch, &journal);
}

#[test]
fn regions_hold_over_every_lifecycle() {
    // The peer-written planes allocate on their DMA owner by name.
    for region in crate::buffers::REGIONS {
        for member in region.members {
            assert!(!matches!(*member, "partial" | "reduced" | "embedded"));
        }
    }
    let mut traced = 0;
    lifecycles(|label, scratch, journal| {
        let refusals = check(crate::buffers::REGIONS, scratch, journal);
        assert!(refusals.is_empty(), "{label}: {refusals:#?}");
        traced += 1;
    });
    assert_eq!(traced, 6 + 3 + 2 + 5);
}

/// Regions of `a` and `b`, two 64-byte buffers of one rank.
const PAIR: &[Region] = &[Region {
    name: "r",
    members: &["a", "b"],
}];

fn span(name: &str, offset: u64, length: u64) -> String {
    let raw = if name == "a" { 101 } else { 102 };
    format!(
        "{:?}",
        DeviceBufferSpanV1::new(BundleDeviceBufferHandleV1::from_raw(raw), offset, length)
    )
}

/// The refusals of `PAIR` over `lines`, where `<a>` and `<b>` stand for
/// the whole of `a` and `b`.
fn pair(lines: &[&str]) -> Vec<String> {
    let scratch = vec![vec![
        (
            "a".to_owned(),
            DeviceBufferSpanV1::new(BundleDeviceBufferHandleV1::from_raw(101), 0, 64),
        ),
        (
            "b".to_owned(),
            DeviceBufferSpanV1::new(BundleDeviceBufferHandleV1::from_raw(102), 0, 64),
        ),
    ]];
    let journal: Vec<String> = lines
        .iter()
        .map(|line| {
            line.replace("<a>", &span("a", 0, 64))
                .replace("<b>", &span("b", 0, 64))
        })
        .collect();
    check(PAIR, &scratch, &journal).into_keys().collect()
}

#[test]
fn a_member_reads_only_bytes_it_wrote() {
    assert!(
        pair(&[
            "zero 1 <a>",
            "download 1 <a>",
            "zero 1 <b>",
            "download 1 <b>"
        ])
        .is_empty()
    );
    assert_eq!(
        pair(&["zero 1 <a>", "zero 1 <b>", "download 1 <a>"]),
        ["a reads bytes b wrote in region r"]
    );
    // Bytes `b` did not write keep `a`'s value.
    let upper = format!("zero 1 {}", span("b", 32, 32));
    let lower = format!("download 1 {}", span("a", 0, 32));
    assert!(pair(&["zero 1 <a>", &upper, &lower]).is_empty());
    assert_eq!(
        pair(&["zero 1 <a>", &upper, "download 1 <a>"]),
        ["a reads bytes b wrote in region r"]
    );
    // A kernel argument outside `OUTPUTS` reads what it writes over.
    let launch = "launch 1 test_kernel LaunchConfigV1 { flags: 0 } [DeviceBuffer(<b>)]";
    assert_eq!(
        pair(&["zero 1 <a>", launch]),
        ["b reads bytes a wrote in region r"]
    );
    // Its threads use both members at once, whatever their order.
    assert_eq!(
        pair(&[
            "launch 1 test_kernel LaunchConfigV1 { flags: 0 } [DeviceBuffer(<a>), ScalarU32(1), DeviceBuffer(<b>)]"
        ]),
        [
            "b reads bytes a wrote in region r",
            "one launch binds a and b"
        ]
    );
}

#[test]
fn split_outputs_write_their_span_only_as_a_pair() {
    // A split fold's launch on `stream` whose output, argument `at`, is `b`.
    let launch = |stream: u32, part: &str, at: usize| {
        format!(
            "launch {stream} chop_deepseek41_mxfp8_fold_m8_f32_v2_prefill_tma_v1_split_{part}_v1 \
             LaunchConfigV1 {{ flags: 0 }} [{}DeviceBuffer(<b>)]",
            "ScalarU32(0), ".repeat(at)
        )
    };
    let (full, tail, other_tail) = (
        &launch(1, "full", 6),
        &launch(1, "tail", 4),
        &launch(2, "tail", 4),
    );
    assert!(pair(&["zero 1 <a>", "download 1 <a>", full, tail, "download 1 <b>"]).is_empty());
    // The first launch leaves the tail rows' bytes as they were.
    assert_eq!(
        pair(&["zero 1 <a>", full, "download 1 <b>"]),
        [
            "a split write of b has no second launch in region r",
            "b touches bytes of a split write of b before its second launch in region r",
        ]
    );
    // The second leaves the full tiles' bytes.
    assert_eq!(
        pair(&["zero 1 <a>", tail]),
        ["b reads bytes a wrote in region r"]
    );
    // The second launch must follow the first on its stream.
    assert_eq!(
        pair(&["zero 1 <a>", full, "record 1 9", "wait 2 9", other_tail]),
        [
            "a split write of b has no second launch in region r",
            "b touches bytes of a split write of b before its second launch in region r",
        ]
    );
}

#[test]
fn output_patterns_match_whole_symbols() {
    let matrix = "chop_deepseek41_mxfp8_fold_m*_f32_v2_matrix_m#_n128_s#_v1";
    let symbol = "chop_deepseek41_mxfp8_fold_m8192_n1024_k4096_g2_f32_v2_matrix_m32_n128_s3_v1";
    assert!(matches(matrix, symbol));
    // The variant that leaves out the first fold is not an output.
    assert!(!matches(matrix, &format!("{symbol}_omit_first_v1")));
    assert!(!matches(matrix, &symbol.replace("_m32_", "_m_")));
    assert!(!matches("a*b", "acbc"));
}

#[test]
fn members_on_two_streams_need_an_order() {
    let unordered = ["b and a touch region r unordered"];
    assert_eq!(
        pair(&["zero 1 <a>", "download 1 <a>", "zero 2 <b>"]),
        unordered
    );
    let ordered = [
        &[
            "zero 1 <a>",
            "download 1 <a>",
            "record 1 9",
            "wait 2 9",
            "zero 2 <b>",
        ][..],
        &[
            "zero 1 <a>",
            "download 1 <a>",
            "synchronize 1",
            "zero 2 <b>",
        ],
    ];
    for lines in ordered {
        assert!(pair(lines).is_empty(), "{lines:?}");
    }
    // An event recorded before `a`'s last touch does not order it.
    assert_eq!(
        pair(&[
            "zero 1 <a>",
            "record 1 9",
            "download 1 <a>",
            "wait 2 9",
            "zero 2 <b>"
        ]),
        unordered
    );
}

#[test]
fn captured_graphs_run_at_each_replay_with_their_fork_and_join() {
    // The side branch forks after the main stream's earlier work and joins
    // before its later work.
    let forked = [
        "capture 1",
        "record 1 5",
        "wait 2 5",
        "zero 2 <b>",
        "download 2 <b>",
        "record 2 6",
        "wait 1 6",
        "captured 1 40 2",
    ];
    let mut lines = vec!["zero 1 <a>", "download 1 <a>"];
    lines.extend(forked);
    lines.extend(["replay 1 40", "zero 1 <a>", "download 1 <a>"]);
    assert!(pair(&lines).is_empty());
    // Captured work runs only when replayed.
    assert!(
        pair(&[
            "zero 1 <a>",
            "capture 1",
            "zero 1 <b>",
            "captured 1 40 1",
            "download 1 <a>"
        ])
        .is_empty()
    );
    assert_eq!(
        pair(&[
            "zero 1 <a>",
            "capture 1",
            "zero 1 <b>",
            "captured 1 40 1",
            "replay 1 40",
            "download 1 <a>"
        ]),
        ["a reads bytes b wrote in region r"]
    );
    // Branches between the fork and the join run concurrently.
    let concurrent = [
        "capture 1",
        "record 1 5",
        "wait 2 5",
        "zero 2 <b>",
        "zero 1 <a>",
        "record 2 6",
        "wait 1 6",
        "captured 1 40 2",
        "replay 1 40",
    ];
    assert_eq!(pair(&concurrent), ["a and b touch region r unordered"]);
}

#[test]
fn launch_arguments_keep_their_positions() {
    let line = format!(
        "launch 3 test_kernel LaunchConfigV1 {{ flags: 1, cluster: [1, 1, 1] }} [DeviceBuffer({}), ScalarU32(5), DevicePointerArray([Some({}), None]), TensorMap(TensorMapV1 {{ buffer: {}, rank: 2, global_dims: [1, 2] }})]",
        span("a", 0, 64),
        span("b", 8, 16),
        span("a", 32, 32)
    );
    let (stream, symbol, arguments) = launch_arguments(&line).unwrap();
    assert_eq!((stream, symbol.as_str()), (3, "test_kernel"));
    assert_eq!(
        arguments,
        [
            vec![(101, 0, 64)],
            vec![],
            vec![(102, 8, 16)],
            vec![(101, 32, 32)]
        ]
    );
}

#[test]
fn unmodelled_members_are_refused() {
    let scratch = vec![vec![(
        "engram_codes".to_owned(),
        DeviceBufferSpanV1::new(BundleDeviceBufferHandleV1::from_raw(7), 0, 64),
    )]];
    for (members, refusal) in [
        (
            &["engram_codes"][..],
            "engram_codes is written by asynchronous host gathers",
        ),
        (&["missing"], "missing names no scratch buffer"),
    ] {
        let regions = [Region { name: "r", members }];
        let refusals: Vec<_> = check(&regions, &scratch, &[]).into_keys().collect();
        assert_eq!(refusals, [refusal]);
    }
}

/// A deliberately overlapping grouping of the four-rank lifecycle is
/// refused: two members one launch binds, a member another overwrites
/// while it is live, members on concurrent branches (the router beside the
/// shared expert in the decode graphs), and an asynchronous gather target.
#[test]
fn overlapping_groupings_are_refused() {
    let shape = Shape::new(&FLASH, Ranks::TP4).unwrap();
    let host = FakeHost::new(shape, KV_LAYOUTS[0]);
    let (scratch, journal) = trace(&host, |host| create(host, 4), short_steps);
    for (members, refusal) in [
        (
            &["query", "native_q"][..],
            "one launch binds query and native_q",
        ),
        (
            &["native_q", "index_query"],
            "native_q reads bytes index_query wrote in region bad",
        ),
        (
            &["router_logits", "shared_gate"],
            "router_logits and shared_gate touch region bad unordered",
        ),
        (
            &["routed", "engram_codes"],
            "engram_codes is written by asynchronous host gathers",
        ),
    ] {
        let regions = [Region {
            name: "bad",
            members,
        }];
        let refusals = check(&regions, &scratch, &journal);
        assert!(refusals.contains_key(refusal), "{members:?}: {refusals:#?}");
    }
}
