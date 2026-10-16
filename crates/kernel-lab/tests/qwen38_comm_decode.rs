//! The decode hyper-connection all-reduce against the general entry: both
//! run on the same inputs at every decode bucket and live count, and their
//! outputs must agree bit for bit on every rank, eagerly and in PDL graphs.
//! They run at the selected rank count: 2, 4 or 8 ranks
//! (`CHOPINFER_KERNEL_GPUS`, distinct devices). The test names
//! keep their TP4 prefix from before the entries took a rank count. Replays
//! use one captured graph per rank through `RankReplay`.

use kernel_core::codec::f32_to_bf16;
use kernel_core::contract::{ArgumentList, DType, LaunchPlan, SigmoidMeanHyperConnectionQuery};
use kernel_lab::launch::{self, CollectiveModules};
use kernel_lab::poison::{self, Fill, POISON, WriteMap};
use kernel_lab::topology::RankReplay;
use kernel_lab::{LabResult, Session, gpu_test_multi};
use model_qwen38_contract::sm120::comm::{self as hc, decode as hc_decode};
use runtime_cuda::{DeviceBuffer, GraphExec};
use serde_json::json;
use std::collections::BTreeMap;

#[path = "common/comm_inputs.rs"]
mod comm_inputs;
#[path = "common/comm_setup.rs"]
mod comm_setup;

fn sync_inputs(ranks: usize, rows: usize, call: u64) -> Vec<Vec<u16>> {
    comm_inputs::inputs(ranks, rows, WIDTH, call)
}

#[test]
fn sync_inputs_expose_wrong_rank_order() {
    for ranks in [4, 8] {
        for rows in [ROWS, 64] {
            for call in [0, 17, 191] {
                comm_inputs::require_order_sensitive(&sync_inputs(ranks, rows, call), WIDTH)
                    .unwrap();
            }
        }
    }
}

const WIDTH: usize = 2560;
const WIDE: usize = WIDTH * 4;
const ROWS: usize = 32;

fn codes(len: usize, seed: usize) -> Vec<u16> {
    (0..len)
        .map(|i| f32_to_bf16((((i * 31 + seed * 73) % 509) as f32 - 254.0) / 128.0))
        .collect()
}

fn bytes(s: &Session, data: &[u8]) -> LabResult<DeviceBuffer> {
    let b = s.allocate(data.len())?;
    b.copy_from_bytes(data, &s.stream)
        .map_err(|e| e.to_string())?;
    Ok(b)
}

/// Poisons buffers `ids` of every rank's fixture once every rank is idle.
fn poison_ids(
    sessions: &[Session],
    fixtures: &[BTreeMap<u64, DeviceBuffer>],
    ids: &[u64],
) -> LabResult<()> {
    poison::synchronize(sessions)?;
    for (s, fixture) in sessions.iter().zip(fixtures) {
        let targets = ids.iter().map(|id| &fixture[id]).collect::<Vec<_>>();
        poison::poison(s, &targets)?;
    }
    Ok(())
}

/// The first `live` rows of a poisoned output after its call: the general,
/// decode and wide entries write residual and normed rows below the live
/// count only (qwen38.cuh and qwen38_decode.cuh return from every row at or
/// past it), so every byte past them must still hold poison.
fn live_rows(s: &Session, buffer: &DeviceBuffer, live: usize, label: &str) -> LabResult<Vec<u16>> {
    let bytes = poison::read(s, buffer)?;
    poison::check(
        label,
        &bytes,
        &WriteMap::of(DType::Bf16).prefix(live * WIDE)?,
        Fill::Pattern(&[POISON]),
    )?;
    Ok(bytes[..live * WIDE * 2]
        .chunks_exact(2)
        .map(|pair| u16::from_ne_bytes([pair[0], pair[1]]))
        .collect())
}

/// The selected rank count, which the entries must support.
fn rank_count(sessions: &[Session]) -> LabResult<usize> {
    let ranks = sessions.len();
    if !hc::RANKS.contains(&(ranks as u32)) {
        return Err(format!("Qwen comm takes 2, 4 or 8 ranks, not {ranks}"));
    }
    Ok(ranks)
}

/// Pointer-array handles `first`, `first + 1`, ... of every rank.
fn slots(first: u64, ranks: usize) -> Vec<u64> {
    (first..first + ranks as u64).collect()
}

struct Lab<'a> {
    sessions: &'a [Session],
    fixtures: Vec<BTreeMap<u64, DeviceBuffer>>,
    general: CollectiveModules,
    decode: CollectiveModules,
    decode_symbol: &'static str,
    query: SigmoidMeanHyperConnectionQuery,
}

impl Lab<'_> {
    fn bindings(&self, rank: usize, inbox: u64, flags: u64) -> launch::Bindings<'_> {
        let mut buffers: BTreeMap<_, _> =
            self.fixtures[rank].iter().map(|(&k, v)| (k, v)).collect();
        for (peer, fixture) in self.fixtures.iter().enumerate() {
            buffers.insert(peer as u64 + 1, &fixture[&inbox]);
            buffers.insert(peer as u64 + 11, &fixture[&flags]);
        }
        launch::Bindings { buffers }
    }

    /// One general entry call on every rank.
    fn general(&self) -> LabResult<()> {
        let mut plan = hc::plan(self.query, ROWS as u32).map_err(|e| e.to_string())?;
        plan.pdl = false;
        launch::launch_collective(
            self.sessions,
            &self.general,
            hc::SYMBOL,
            plan,
            None,
            |rank| {
                let args = hc::arguments(hc::Call {
                    input: 100,
                    residual: 101,
                    logits: 102,
                    weight: 103,
                    residual_out: 104,
                    normed: 105,
                    inboxes: slots(1, self.sessions.len()),
                    flags: slots(11, self.sessions.len()),
                    epoch: 108,
                    live_rows: 109,
                    capacity: ROWS as u32,
                    rank: rank as u32,
                })
                .map_err(|e| e.to_string())?;
                Ok(launch::RankCall {
                    args,
                    bindings: self.bindings(rank, 106, 107),
                })
            },
        )
    }

    /// One decode entry call on every rank, or only on `only_rank`.
    fn decode(&self, only_rank: Option<usize>, rows: u32, pdl: bool) -> LabResult<()> {
        let mut plan = hc_decode::plan(rows).map_err(|e| e.to_string())?;
        plan.pdl = pdl;
        launch::launch_collective(
            self.sessions,
            &self.decode,
            self.decode_symbol,
            plan,
            only_rank,
            |rank| {
                let args = hc_decode::arguments(hc_decode::Call {
                    input: 100,
                    residual: 101,
                    logits: 102,
                    weight: 103,
                    residual_out: 114,
                    normed: 115,
                    inboxes: slots(1, self.sessions.len()),
                    flags: slots(11, self.sessions.len()),
                    epochs: 118,
                    live_rows: 109,
                    rows,
                    rank: rank as u32,
                })
                .map_err(|e| e.to_string())?;
                Ok(launch::RankCall {
                    args,
                    bindings: self.bindings(rank, 116, 117),
                })
            },
        )
    }

    fn sync(&self) -> LabResult<()> {
        for s in self.sessions {
            s.stream.synchronize().map_err(|e| e.to_string())?;
        }
        Ok(())
    }

    fn set_live(&self, live: u32) -> LabResult<()> {
        for (r, s) in self.sessions.iter().enumerate() {
            self.fixtures[r][&109]
                .copy_from_bytes(&live.to_ne_bytes(), &s.stream)
                .map_err(|e| e.to_string())?;
        }
        self.sync()
    }

    /// Poisons outputs `ids` of every rank.
    fn poison(&self, ids: &[u64]) -> LabResult<()> {
        poison_ids(self.sessions, &self.fixtures, ids)
    }

    fn reset_decode_outputs(&self) -> LabResult<()> {
        self.poison(&[114, 115])
    }

    /// The live rows of outputs `ids` of `rank`, whose later rows must
    /// still hold poison.
    fn outputs(
        &self,
        rank: usize,
        ids: [u64; 2],
        live: usize,
        what: &str,
    ) -> LabResult<[Vec<u16>; 2]> {
        let s = &self.sessions[rank];
        let rows = |id: u64| {
            live_rows(
                s,
                &self.fixtures[rank][&id],
                live,
                &format!("{what}: rank {rank} output {id}"),
            )
        };
        Ok([rows(ids[0])?, rows(ids[1])?])
    }

    /// Decode outputs equal the general entry's on the live rows and leave
    /// the rest poisoned, as the general entry does, on every rank.
    fn check(&self, live: usize, what: &str) -> LabResult<()> {
        for rank in 0..self.sessions.len() {
            let expected = self.outputs(rank, [104, 105], live, what)?;
            let actual = self.outputs(rank, [114, 115], live, what)?;
            for (e, a) in expected.iter().zip(&actual) {
                if a != e {
                    return Err(format!(
                        "{what}: rank {rank} differs from the general entry"
                    ));
                }
            }
        }
        Ok(())
    }

    /// `calls` PDL calls per rank, replayable as captured graphs.
    fn replay(
        &self,
        rows: u32,
        calls: usize,
    ) -> LabResult<RankReplay<impl Fn(usize, &Session) -> LabResult<()> + '_>> {
        RankReplay::new(self.sessions, move |r, _| {
            (0..calls).try_for_each(|_| self.decode(Some(r), rows, true))
        })
    }

    /// Captured graphs for timing, which needs distinct devices.
    fn graphs(&self, rows: u32, calls: usize) -> LabResult<Vec<GraphExec>> {
        self.sessions
            .iter()
            .enumerate()
            .map(|(r, s)| {
                s.capture(|_| {
                    for _ in 0..calls {
                        self.decode(Some(r), rows, true)?;
                    }
                    Ok(())
                })
            })
            .collect()
    }
}

fn run_cases(benchmark: bool) {
    gpu_test_multi(256 << 20, |sessions| {
        let ranks = rank_count(sessions)?;
        if benchmark {
            for session in sessions {
                evidence::exclusive(session)?;
            }
        }
        let query = SigmoidMeanHyperConnectionQuery::AllReduceInjectNorm {
            ranks: ranks as u32,
            streams: 4,
            width: WIDTH as u32,
            eps_bits: 1e-6_f32.to_bits(),
            capacity: ROWS as u32,
        };
        let general_sizes = hc::buffers(query).map_err(|e| e.to_string())?;
        let decode_sizes = hc_decode::buffers(ranks as u32).map_err(|e| e.to_string())?;
        // Both decode entries, on one state: v3 shares v2's state and bits.
        let symbols = [
            hc_decode::symbol_v2(ranks as u32),
            hc_decode::symbol_v3(ranks as u32),
        ]
        .into_iter()
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| e.to_string())?;
        let x = codes(ROWS * WIDE, 1);
        let logits = codes(ROWS * 4, 2);
        let weight = codes(WIDE, 3);
        let mut fixtures = Vec::new();
        for (rank, s) in sessions.iter().enumerate() {
            let mut b = BTreeMap::new();
            let partial = codes(ROWS * WIDTH, rank + 4);
            for (id, values) in [(100, &partial), (101, &x), (102, &logits), (103, &weight)] {
                b.insert(id, s.upload_bf16(values)?);
            }
            // Outputs, poisoned before every launch.
            for id in [104, 105, 114, 115] {
                b.insert(id, s.allocate(ROWS * WIDE * 2)?);
            }
            b.insert(106, s.allocate(general_sizes.inbox_bytes as usize)?);
            b.insert(107, bytes(s, &vec![0; general_sizes.flags_bytes as usize])?);
            b.insert(108, bytes(s, &vec![0; general_sizes.epoch_bytes as usize])?);
            b.insert(109, bytes(s, &[0; 4])?);
            // Only the LL region must start zeroed; the rest holds garbage.
            let mut inbox = vec![0xff; decode_sizes.inbox_bytes as usize];
            inbox[..decode_sizes.zeroed_inbox_bytes as usize].fill(0);
            b.insert(116, bytes(s, &inbox)?);
            b.insert(117, bytes(s, &vec![0; decode_sizes.flags_bytes as usize])?);
            b.insert(118, bytes(s, &vec![0; decode_sizes.epoch_bytes as usize])?);
            fixtures.push(b);
        }
        // Load and resolve every entry before any launch.
        let general = CollectiveModules::load(sessions, "comm_qwen38")?;
        general.function(hc::SYMBOL)?;
        let decode = CollectiveModules::load(sessions, "comm_qwen38_decode")?;
        for symbol in &symbols {
            decode.function(symbol)?;
        }
        let mut lab = Lab {
            sessions,
            fixtures,
            general,
            decode,
            decode_symbol: symbols[0],
            query,
        };
        lab.sync()?;
        let buckets = [1_u32, 2, 4, 8, 16, 32];
        if benchmark {
            const CALLS: usize = 16;
            let (peak, peak_source) = evidence::peak(&sessions[0], "kernel_push")?;
            for (symbol, rows) in symbols
                .iter()
                .flat_map(|&symbol| buckets.map(|rows| (symbol, rows)))
            {
                lab.decode_symbol = symbol;
                lab.set_live(rows)?;
                let graphs = lab.graphs(rows, CALLS)?;
                let timing = kernel_lab::time_multi_graph(sessions, &graphs, CALLS, 3, 9)?;
                let per_call = f64::from(timing.critical_path.median_ms) * 1000.0 / CALLS as f64;
                let data = BTreeMap::from([
                    ("artifact".into(), evidence::identity("comm_qwen38_decode")?),
                    ("symbol".into(), json!(symbol)),
                    ("rows".into(), json!(rows)),
                    ("ranks".into(), json!(ranks)),
                    ("calls_per_graph".into(), json!(CALLS)),
                    ("median_us_per_call".into(), json!(per_call)),
                    ("rounds".into(), json!(timing.critical_path.rounds)),
                    (
                        "pcie_lower_bound_us".into(),
                        json!(f64::from(rows) * (WIDTH * 2 * (ranks - 1)) as f64 / peak * 1e6),
                    ),
                    ("peak_source".into(), json!(peak_source)),
                ]);
                evidence::receipt(&sessions[0], "qwen38-comm-decode-g4", data)?;
            }
            return Ok(());
        }
        for &symbol in &symbols {
            lab.decode_symbol = symbol;
            // Eager: every bucket at full, partial and single-row live
            // counts, with buckets interleaved so the LL row epochs and the
            // one-shot epoch advance across protocols.
            for rows in buckets.into_iter().chain([1, 4, 2, 32, 1]) {
                let mut lives = vec![rows, 1];
                if rows > 2 {
                    lives.push(rows - 1);
                }
                for live in lives {
                    lab.set_live(live)?;
                    lab.poison(&[104, 105, 114, 115])?;
                    lab.general()?;
                    lab.sync()?;
                    lab.decode(None, rows, false)?;
                    lab.sync()?;
                    lab.check(
                        live as usize,
                        &format!("{symbol}: eager rows {rows} live {live}"),
                    )?;
                }
            }
            // PDL graphs of consecutive calls, replayed: the same inputs give
            // the same outputs each call.
            for rows in buckets {
                lab.set_live(rows)?;
                lab.poison(&[104, 105])?;
                lab.general()?;
                lab.sync()?;
                let replays = lab.replay(rows, 4)?;
                for replay in 0..3 {
                    lab.reset_decode_outputs()?;
                    replays.launch(sessions)?;
                    lab.sync()?;
                    lab.check(
                        rows as usize,
                        &format!("{symbol}: graph rows {rows} replay {replay}"),
                    )?;
                }
            }
        }
        sessions[0].receipt(
            "qwen38-comm-decode-g1",
            BTreeMap::from([
                ("artifact".into(), evidence::identity("comm_qwen38_decode")?),
                ("symbols".into(), json!(symbols)),
                ("ranks".into(), json!(ranks)),
                ("buckets".into(), json!(buckets)),
                ("bitwise_equal_to".into(), json!(hc::SYMBOL)),
                ("graph_determinism_pdl_rank_agreement".into(), json!(true)),
            ]),
        )?;
        Ok(())
    })
    .expect("Qwen decode HC all-reduce");
}

#[path = "common/evidence.rs"]
mod evidence;

#[test]
fn tp4_decode_hyper_connection_matches_general_entry() {
    run_cases(false);
}

#[test]
#[ignore = "requires the parent's exclusive KERNEL window"]
fn tp4_decode_hyper_connection_g4() {
    run_cases(true);
}

/// (prepared rows, live rows) of the synchronization sequence: every bucket
/// protocol in turn, dead rows, and live counts of zero between busy calls.
const SEQUENCE: [(u32, u32); 16] = [
    (1, 1),
    (2, 2),
    (4, 3),
    (32, 32),
    (2, 1),
    (16, 16),
    (16, 0),
    (16, 16),
    (16, 0),
    (16, 16),
    (1, 1),
    (4, 4),
    (32, 17),
    (2, 0),
    (1, 1),
    (8, 5),
];

/// (version, prepared rows, live rows), where version 4 is the wide entry.
/// Alternate all families on one state. Wide rows 16 and 64 cover both of
/// its protocols at four ranks; two ranks use its one-shot at both sizes.
const MIXED_SEQUENCE: [(u32, u32, u32); 16] = [
    (3, 1, 1),
    (2, 4, 3),
    (2, 2, 2),
    (4, 64, 33),
    (3, 2, 1),
    (2, 32, 31),
    (2, 1, 1),
    (4, 16, 15),
    (3, 1, 0),
    (2, 2, 2),
    (2, 16, 15),
    (4, 64, 1),
    (3, 2, 2),
    (2, 1, 1),
    (2, 32, 32),
    (4, 64, 0),
];

/// The skew build's name of a decode entry. Its lag: in every call the rank
/// chosen by the epoch waits 20 us before each LL step, each one-shot push
/// and each one-shot wait.
fn skew_symbol(symbol: &str) -> String {
    symbol.replacen("chop_comm_qwen38_", "chop_comm_qwen38_skew_", 1)
}

/// Cross-rank synchronization of the decode entry, as comm_sync.rs does for
/// the push protocols: SEQUENCE queued on every rank without host barriers,
/// eagerly and as one captured graph per rank, with the skew lag, on one
/// state whose epochs start near the u32 and 2^31 wraps. Every call has its
/// own input and outputs and must equal the general entry on its live rows,
/// on every rank, and leave dead rows untouched.
#[test]
fn tp4_decode_hyper_connection_v2_sync_g1() {
    run_sync(2, false);
}

/// Whole-row LL readers against v2, including slot reuse across flag wraps.
#[test]
fn tp4_decode_hyper_connection_v3_sync_g1() {
    run_sync(3, false);
}

#[test]
fn decode_versions_and_wide_share_state_g1() {
    run_sync(3, true);
}

fn sync_query(ranks: usize, capacity: usize) -> SigmoidMeanHyperConnectionQuery {
    SigmoidMeanHyperConnectionQuery::AllReduceInjectNorm {
        ranks: ranks as u32,
        streams: 4,
        width: WIDTH as u32,
        eps_bits: 1e-6_f32.to_bits(),
        capacity: capacity as u32,
    }
}

fn sync_states(
    ranks: usize,
    capacity: usize,
    mixed: bool,
    separate_v2: bool,
) -> LabResult<(hc_decode::Buffers, hc::Buffers)> {
    let general = hc::buffers(sync_query(ranks, capacity)).map_err(|e| e.to_string())?;
    let candidate = if mixed {
        hc_decode::wide_buffers(ranks as u32)
    } else {
        hc_decode::buffers(ranks as u32)
    }
    .map_err(|e| e.to_string())?;
    // The candidate keeps the bundle's decode prefix, extended for wide
    // calls. The general oracle owns separate one-shot banks at capacity.
    let oracle = if separate_v2 {
        let decode = hc_decode::buffers(ranks as u32).map_err(|e| e.to_string())?;
        hc::Buffers {
            inbox_bytes: decode.inbox_bytes,
            flags_bytes: decode.flags_bytes,
            epoch_bytes: decode.epoch_bytes,
        }
    } else {
        hc::Buffers {
            inbox_bytes: hc::standalone_inbox_bytes(ranks as u32, capacity as u32),
            ..general
        }
    };
    Ok((candidate, oracle))
}

fn sync_candidate_call(
    ranks: usize,
    rank: usize,
    k: usize,
    entry: u32,
    rows: u32,
) -> LabResult<(LaunchPlan, ArgumentList)> {
    let k = k as u64;
    let call = hc_decode::Call {
        input: 100 + k,
        residual: 1,
        logits: 2,
        weight: 3,
        residual_out: 200 + 2 * k,
        normed: 201 + 2 * k,
        inboxes: slots(11, ranks),
        flags: slots(21, ranks),
        epochs: 10,
        live_rows: 400 + k,
        rows,
        rank: rank as u32,
    };
    let (plan, args) = if entry == 4 {
        (hc_decode::wide_plan(rows), hc_decode::wide_arguments(call))
    } else {
        (hc_decode::plan(rows), hc_decode::arguments(call))
    };
    Ok((
        plan.map_err(|e| e.to_string())?,
        args.map_err(|e| e.to_string())?,
    ))
}

fn sync_oracle_call(
    ranks: usize,
    rank: usize,
    capacity: usize,
    k: usize,
    separate_v2: bool,
) -> LabResult<(LaunchPlan, ArgumentList)> {
    let k = k as u64;
    let (plan, args) = if separate_v2 {
        (
            hc_decode::plan(capacity as u32),
            hc_decode::arguments(hc_decode::Call {
                input: 100 + k,
                residual: 1,
                logits: 2,
                weight: 3,
                residual_out: 300,
                normed: 301,
                inboxes: slots(11, ranks),
                flags: slots(21, ranks),
                epochs: 6,
                live_rows: 7,
                rows: capacity as u32,
                rank: rank as u32,
            }),
        )
    } else {
        (
            hc::plan(sync_query(ranks, capacity), capacity as u32),
            hc::standalone_arguments(hc::Call {
                input: 100 + k,
                residual: 1,
                logits: 2,
                weight: 3,
                residual_out: 300,
                normed: 301,
                inboxes: slots(11, ranks),
                flags: slots(21, ranks),
                epoch: 6,
                live_rows: 7,
                capacity: capacity as u32,
                rank: rank as u32,
            }),
        )
    };
    let mut plan = plan.map_err(|e| e.to_string())?;
    plan.pdl = false;
    Ok((plan, args.map_err(|e| e.to_string())?))
}

#[test]
fn sync_launch_setup_is_valid_without_cuda() {
    for ranks in [2, 4, 8] {
        for (version, mixed) in [(2, false), (3, false), (3, true)] {
            let capacity = if mixed { 64 } else { ROWS };
            let separate_v2 = version == 3 && !mixed;
            let sequence: Vec<_> = if mixed {
                MIXED_SEQUENCE.to_vec()
            } else {
                SEQUENCE
                    .iter()
                    .map(|&(rows, live)| (version, rows, live))
                    .collect()
            };
            let (candidate, oracle) = sync_states(ranks, capacity, mixed, separate_v2).unwrap();
            let narrow = hc_decode::buffers(ranks as u32).unwrap();
            let wanted = if mixed {
                hc_decode::wide_buffers(ranks as u32).unwrap()
            } else {
                narrow
            };
            assert_eq!(
                (
                    candidate.inbox_bytes,
                    candidate.flags_bytes,
                    candidate.epoch_bytes
                ),
                (wanted.inbox_bytes, wanted.flags_bytes, wanted.epoch_bytes)
            );
            assert_eq!(candidate.zeroed_inbox_bytes, narrow.zeroed_inbox_bytes);
            assert_eq!(candidate.epoch_bytes, ((ROWS + 2) * 4) as u64);
            let oracle_inbox = if separate_v2 {
                narrow.inbox_bytes
            } else {
                hc::standalone_inbox_bytes(ranks as u32, capacity as u32)
            };
            assert_eq!(oracle.inbox_bytes, oracle_inbox);
            let decode_cubin = format!("{}_skew", hc_decode::TRANSPORT.cubin);
            let wide_cubin = format!("{}_skew", hc_decode::TRANSPORT.wide_cubin);
            for (k, &(entry, rows, live)) in sequence.iter().enumerate() {
                assert!(live <= rows && rows as usize <= capacity);
                let input_bytes = (capacity * WIDTH * 2) as u64;
                let output_bytes = (capacity * WIDE * 2) as u64;
                let k64 = k as u64;
                let mut buffers = BTreeMap::from([
                    (1, output_bytes),
                    (2, (capacity * 4 * 2) as u64),
                    (3, (WIDE * 2) as u64),
                    (6, oracle.epoch_bytes),
                    (7, 4),
                    (10, candidate.epoch_bytes),
                    (100 + k64, input_bytes),
                    (200 + 2 * k64, output_bytes),
                    (201 + 2 * k64, output_bytes),
                    (300, output_bytes),
                    (301, output_bytes),
                    (400 + k64, 4),
                ]);
                let symbol = match entry {
                    4 => hc_decode::wide_symbol(ranks as u32),
                    3 => hc_decode::symbol_v3(ranks as u32),
                    _ => hc_decode::symbol_v2(ranks as u32),
                }
                .unwrap();
                let cubin = if entry == 4 {
                    &wide_cubin
                } else {
                    &decode_cubin
                };
                for rank in 0..ranks {
                    for peer in 0..ranks as u64 {
                        buffers.insert(11 + peer, candidate.inbox_bytes);
                        buffers.insert(21 + peer, candidate.flags_bytes);
                    }
                    let (plan, args) = sync_candidate_call(ranks, rank, k, entry, rows).unwrap();
                    comm_setup::check(cubin, &skew_symbol(symbol), plan, &args, &buffers).unwrap();
                    for peer in 0..ranks as u64 {
                        buffers.insert(11 + peer, oracle.inbox_bytes);
                        buffers.insert(21 + peer, oracle.flags_bytes);
                    }
                    let (plan, args) = sync_oracle_call(ranks, rank, capacity, k, separate_v2)
                        .unwrap_or_else(|e| {
                            panic!("ranks {ranks} mixed {mixed} call {k} oracle: {e}")
                        });
                    let (oracle_cubin, oracle_symbol) = if separate_v2 {
                        (
                            decode_cubin.as_str(),
                            skew_symbol(hc_decode::symbol_v2(ranks as u32).unwrap()),
                        )
                    } else {
                        (hc::TRANSPORT.cubin, hc::SYMBOL.to_owned())
                    };
                    comm_setup::check(oracle_cubin, &oracle_symbol, plan, &args, &buffers).unwrap();
                }
            }
        }
    }
}

fn run_sync(version: u32, mixed: bool) {
    gpu_test_multi(512 << 20, |sessions| {
        let ranks = rank_count(sessions)?;
        let capacity = if mixed { 64 } else { ROWS };
        let sequence: Vec<_> = if mixed {
            MIXED_SEQUENCE.to_vec()
        } else {
            SEQUENCE.iter().map(|&(rows, live)| (version, rows, live)).collect()
        };
        let separate_v2 = version == 3 && !mixed;
        let (decode_sizes, oracle_sizes) = sync_states(ranks, capacity, mixed, separate_v2)?;
        let symbols = sequence.iter().map(|&(entry, _, _)| {
            let symbol = match entry {
                4 => hc_decode::wide_symbol(ranks as u32),
                3 => hc_decode::symbol_v3(ranks as u32),
                _ => hc_decode::symbol_v2(ranks as u32),
            }.map_err(|e| e.to_string())?;
            Ok(skew_symbol(symbol))
        }).collect::<LabResult<Vec<_>>>()?;
        let calls = sequence.len();
        for (k, &(entry, rows, _)) in sequence.iter().enumerate() {
            for rank in 0..ranks {
                sync_candidate_call(ranks, rank, k, entry, rows)?;
                sync_oracle_call(ranks, rank, capacity, k, separate_v2)?;
            }
        }
        // State 4/5/6 belongs to the oracle, state 8/9/10 to the candidate.
        // Calls have distinct inputs, outputs and live-count addresses.
        let mut fixtures = Vec::new();
        for s in sessions {
            let mut b = BTreeMap::new();
            for (id, mut values) in [
                (1, codes(capacity * WIDE, 1)),
                (2, codes(capacity * 4, 2)),
                (3, codes(WIDE, 3)),
            ] {
                if id == 1 {
                    comm_inputs::clear_marker_residuals(&mut values, ranks);
                }
                b.insert(id, s.upload_bf16(&values)?);
            }
            b.insert(
                4,
                bytes(
                    s,
                    &vec![0; oracle_sizes.inbox_bytes as usize],
                )?,
            );
            b.insert(
                5,
                bytes(
                    s,
                    &vec![0; oracle_sizes.flags_bytes as usize],
                )?,
            );
            b.insert(
                6,
                bytes(
                    s,
                    &vec![0; oracle_sizes.epoch_bytes as usize],
                )?,
            );
            b.insert(7, bytes(s, &(capacity as u32).to_ne_bytes())?);
            b.insert(8, s.allocate(decode_sizes.inbox_bytes as usize)?);
            b.insert(9, s.allocate(decode_sizes.flags_bytes as usize)?);
            b.insert(10, s.allocate(decode_sizes.epoch_bytes as usize)?);
            for (k, &(_, _, live)) in sequence.iter().enumerate() {
                let k = k as u64;
                b.insert(100 + k, s.allocate(capacity * WIDTH * 2)?);
                b.insert(200 + 2 * k, s.allocate(capacity * WIDE * 2)?);
                b.insert(201 + 2 * k, s.allocate(capacity * WIDE * 2)?);
                b.insert(400 + k, bytes(s, &live.to_ne_bytes())?);
            }
            b.insert(300, s.allocate(capacity * WIDE * 2)?);
            b.insert(301, s.allocate(capacity * WIDE * 2)?);
            fixtures.push(b);
        }
        let general = CollectiveModules::load(sessions, "comm_qwen38")?;
        general.function(hc::SYMBOL)?;
        let decode = CollectiveModules::load(sessions, "comm_qwen38_decode_skew")?;
        let wide_cubin = format!("{}_skew", hc_decode::TRANSPORT.wide_cubin);
        let wide = if mixed {
            Some(CollectiveModules::load(sessions, &wide_cubin)?)
        } else {
            None
        };
        for (&(entry, _, _), symbol) in sequence.iter().zip(&symbols) {
            if entry == 4 {
                wide.as_ref().expect("wide sequence module").function(symbol)?;
            } else {
                decode.function(symbol)?;
            }
        }
        let reference = if separate_v2 {
            skew_symbol(hc_decode::symbol_v2(ranks as u32).map_err(|e| e.to_string())?)
        } else {
            hc::SYMBOL.to_owned()
        };
        if separate_v2 {
            decode.function(&reference)?;
        }
        let bindings = |rank: usize, inbox: u64, flags: u64| {
            let mut buffers: BTreeMap<_, _> = fixtures[rank].iter().map(|(&k, v)| (k, v)).collect();
            for (peer, fixture) in fixtures.iter().enumerate() {
                buffers.insert(peer as u64 + 11, &fixture[&inbox]);
                buffers.insert(peer as u64 + 21, &fixture[&flags]);
            }
            launch::Bindings { buffers }
        };
        let sync = || poison::synchronize(sessions);
        let decode_call = |only_rank: Option<usize>, k: usize, pdl: bool| -> LabResult<()> {
            let (entry, rows, _) = sequence[k];
            let symbol = &symbols[k];
            let module = if entry == 4 {
                wide.as_ref().expect("wide sequence module")
            } else {
                &decode
            };
            let (mut plan, _) = sync_candidate_call(ranks, 0, k, entry, rows)?;
            plan.pdl = pdl;
            launch::launch_collective(sessions, module, symbol, plan, only_rank, |rank| {
                let (_, args) = sync_candidate_call(ranks, rank, k, entry, rows)?;
                Ok(launch::RankCall { args, bindings: bindings(rank, 8, 9) })
            })
        };
        let oracle_call = |k: usize| -> LabResult<()> {
            poison_ids(sessions, &fixtures, &[300, 301])?;
            let (plan, _) = sync_oracle_call(ranks, 0, capacity, k, separate_v2)?;
            let module = if separate_v2 { &decode } else { &general };
            launch::launch_collective(sessions, module, &reference, plan, None, |rank| {
                let (_, args) = sync_oracle_call(ranks, rank, capacity, k, separate_v2)?;
                Ok(launch::RankCall { args, bindings: bindings(rank, 4, 5) })
            })?;
            sync()
        };
        let reset = |ll: u32, one_shot: u32| -> LabResult<()> {
            sync()?;
            for (rank, s) in sessions.iter().enumerate() {
                let mut inbox = vec![0xff; decode_sizes.inbox_bytes as usize];
                inbox[..decode_sizes.zeroed_inbox_bytes as usize].fill(0);
                let mut epochs = vec![ll; 32];
                epochs.extend([one_shot, 0]);
                let epochs: Vec<u8> = epochs.iter().flat_map(|v| v.to_ne_bytes()).collect();
                for (id, data) in [
                    (8, inbox),
                    (9, vec![0; decode_sizes.flags_bytes as usize]),
                    (10, epochs),
                ] {
                    fixtures[rank][&id]
                        .copy_from_bytes(&data, &s.stream)
                        .map_err(|e| e.to_string())?;
                }
            }
            sync()
        };
        let set_inputs = |generation: u64| -> LabResult<()> {
            sync()?;
            for k in 0..calls {
                let inputs = sync_inputs(ranks, capacity, generation * calls as u64 + k as u64);
                comm_inputs::require_order_sensitive(&inputs, WIDTH)?;
                for (rank, s) in sessions.iter().enumerate() {
                    fixtures[rank][&(100 + k as u64)]
                        .copy_from_bytes(&comm_inputs::bytes(&inputs[rank]), &s.stream)
                        .map_err(|e| e.to_string())?;
                }
            }
            sync()
        };
        let check = |what: &str| -> LabResult<()> {
            sync()?;
            if version == 3 {
                for (rank, s) in sessions.iter().enumerate() {
                    let inbox = poison::read(s, &fixtures[rank][&8])?;
                    if inbox[..decode_sizes.zeroed_inbox_bytes as usize]
                        .iter()
                        .any(|&b| b != 0)
                    {
                        return Err(format!("{what}: rank {rank} left an LL slot uncleared"));
                    }
                }
            }
            // The candidate finishes first, and the oracle never writes its
            // inbox, flags, epoch or output buffers. Compare every call.
            for (k, &(entry, rows, live)) in sequence.iter().enumerate() {
                oracle_call(k)?;
                for (rank, s) in sessions.iter().enumerate() {
                    for half in 0..2 {
                        let want = live_rows(s, &fixtures[rank][&(300 + half)], capacity, "oracle")?;
                        let got = live_rows(
                            s,
                            &fixtures[rank][&(200 + 2 * k as u64 + half)],
                            live as usize,
                            &format!("{what}: rank {rank} call {k} entry {entry} rows {rows} output {half}"),
                        )?;
                        if got != want[..live as usize * WIDE] {
                            return Err(format!(
                                "{what}: rank {rank} call {k} entry {entry} rows {rows} differs from {reference}"
                            ));
                        }
                    }
                }
            }
            Ok(())
        };
        let replays = RankReplay::new(sessions, |rank, _| {
            (0..calls).try_for_each(|k| decode_call(Some(rank), k, true))
        })?;
        let outputs: Vec<_> = (0..calls as u64)
            .flat_map(|k| [200 + 2 * k, 201 + 2 * k])
            .collect();
        let seeds = [
            ((1 << 12) - 3, 0),
            (u32::MAX - 3, 0x7fff_fffd),
            (u32::MAX - 3, u32::MAX - 3),
        ];
        let mut generation = 0;
        for (ll, one_shot) in seeds {
            reset(ll, one_shot)?;
            // Do not reset between replays: the same state crosses the wrap
            // with different payloads, while every output is poisoned anew.
            for phase in 0..4 {
                set_inputs(generation)?;
                poison_ids(sessions, &fixtures, &outputs)?;
                if phase == 0 {
                    for k in 0..calls {
                        decode_call(None, k, true)?;
                    }
                } else {
                    replays.launch(sessions)?;
                }
                check(&format!(
                    "seed {ll:#x}/{one_shot:#x}, phase {phase}, generation {generation}"
                ))?;
                generation += 1;
            }
        }
        sessions[0].receipt(
            "qwen38-comm-decode-sync-g1",
            BTreeMap::from([
                (
                    "artifact".into(),
                    evidence::identity("comm_qwen38_decode_skew")?,
                ),
                ("symbols".into(), json!(symbols)),
                ("sequence".into(), json!(sequence)),
                ("mixed_entries".into(), json!(mixed)),
                ("wide_artifact".into(), if mixed {
                    evidence::identity(&wide_cubin)?
                } else {
                    serde_json::Value::Null
                }),
                ("ranks".into(), json!(ranks)),
                ("lag_ns".into(), json!(20_000)),
                ("bitwise_equal_to".into(), json!(reference)),
                ("input_generations".into(), json!(generation * calls as u64)),
                ("candidate_first".into(), json!(true)),
            ]),
        )?;
        Ok(())
    })
    .expect("Qwen decode HC synchronization");
}
