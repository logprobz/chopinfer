//! The wide decode hyper-connection all-reduce (graph buckets above 32
//! rows) against the general entry on a state of its own: at 33 to 256
//! prepared rows, full, partial, single and zero live rows, eagerly and in
//! PDL graphs, with decode entry calls on the same state, under the skew
//! build's lag across the epoch wraps, and timed against the phased pair
//! that serves these rows today. Runs at 2, 4 or 8 ranks
//! (`CHOPINFER_KERNEL_GPUS`, distinct devices).

use kernel_core::codec::f32_to_bf16;
use kernel_core::contract::{DType, SigmoidMeanHyperConnectionQuery};
use kernel_lab::launch::{self, CollectiveModules};
use kernel_lab::poison::{self, Fill, POISON, WriteMap};
use kernel_lab::topology::RankReplay;
use kernel_lab::{LabResult, Session, gpu_test_multi};
use model_qwen38_contract::sm120::comm::decode::{LL_MAX_ROWS, symbol_v2, symbol_v3};
use model_qwen38_contract::sm120::comm::{self as qwen38, decode as qwen38_decode};
use runtime_cuda::DeviceBuffer;
use serde_json::json;
use std::collections::BTreeMap;

const WIDTH: usize = 2560;
const WIDE: usize = WIDTH * 4;
/// Rows every buffer holds: the wide entry's largest bucket.
const ROWS: usize = 256;
const PACKAGE_ROWS: [u32; 6] = [33, 64, 100, 128, 255, 256];

fn codes(len: usize, seed: usize) -> Vec<u16> {
    (0..len)
        .map(|i| f32_to_bf16((((i * 31 + seed * 73) % 509) as f32 - 254.0) / 128.0))
        .collect()
}

/// Partials whose exponents span 16 binades, so F32 sums round and a
/// different summation order would change their bits.
fn partials(len: usize, seed: usize) -> Vec<u16> {
    use kernel_core::codec::bf16_to_f32;
    codes(len, seed)
        .into_iter()
        .enumerate()
        .map(|(i, bits)| {
            let scale = f32::powi(2.0, ((i * 5 + seed * 3) % 16) as i32 - 8);
            f32_to_bf16(bf16_to_f32(bits) * scale)
        })
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
/// decode, wide and phased entries write residual and normed rows below the
/// live count only (qwen38.cuh, qwen38_decode.cuh and
/// qwen38_decode_wide.cuh skip every row at or past it), so every byte past
/// them must still hold poison.
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

/// Flag bytes of the shared state, the planner's.
fn flag_bytes(sizes: &qwen38_decode::Buffers) -> usize {
    sizes.flags_bytes as usize
}

fn slots(first: u64, ranks: usize) -> Vec<u64> {
    (first..first + ranks as u64).collect()
}

fn rank_count(sessions: &[Session]) -> LabResult<usize> {
    let ranks = sessions.len();
    if !qwen38::RANKS.contains(&(ranks as u32)) {
        return Err(format!("Qwen comm takes 2, 4 or 8 ranks, not {ranks}"));
    }
    Ok(ranks)
}

fn general_query(ranks: usize) -> SigmoidMeanHyperConnectionQuery {
    SigmoidMeanHyperConnectionQuery::AllReduceInjectNorm {
        ranks: ranks as u32,
        streams: 4,
        width: WIDTH as u32,
        eps_bits: 1e-6_f32.to_bits(),
        capacity: ROWS as u32,
    }
}

fn sync(sessions: &[Session]) -> LabResult<()> {
    for s in sessions {
        s.stream.synchronize().map_err(|e| e.to_string())?;
    }
    Ok(())
}

/// One call on the shared state: the wide entry, or the decode entry for a
/// bucket of up to 32 rows.
#[derive(Clone, Copy, Debug, serde::Serialize)]
struct Shape {
    wide: bool,
    rows: u32,
    live: u32,
}

/// Buffers per rank. Shared: 1 residual streams, 2 inject logits, 3 norm
/// weights; the general entry's state 4 to 7 (7 holds its live count); the
/// shared decode and wide state 8 to 10; per call k: input 100 + k,
/// outputs 200 + 2k and 201 + 2k, live count 400 + k; the general entry's
/// outputs 300 and 301. Peers' decode inboxes and flags are 11.. and 21..,
/// the general entry's 31.. and 41...
struct Lab<'a> {
    sessions: &'a [Session],
    fixtures: Vec<BTreeMap<u64, DeviceBuffer>>,
    general: CollectiveModules,
    decode: CollectiveModules,
    wide: CollectiveModules,
    /// The narrow calls' entries: v2, and v3 at the LL prepared rows.
    decode_symbols: [String; 2],
    wide_symbol: String,
    sizes: qwen38_decode::Buffers,
    calls: Vec<Shape>,
    /// Each rank's worker CTAs for wide calls; empty takes the planner's.
    workers: Vec<u32>,
}

impl<'a> Lab<'a> {
    fn new(sessions: &'a [Session], calls: Vec<Shape>, skew: bool) -> LabResult<Self> {
        let ranks = rank_count(sessions)?;
        let general_sizes = qwen38::buffers(general_query(ranks)).map_err(|e| e.to_string())?;
        let sizes = qwen38_decode::wide_buffers(ranks as u32).map_err(|e| e.to_string())?;
        let decode_symbols = [
            symbol_v2(ranks as u32).map_err(|e| e.to_string())?,
            symbol_v3(ranks as u32).map_err(|e| e.to_string())?,
        ];
        let wide_symbol = qwen38_decode::wide_symbol(ranks as u32).map_err(|e| e.to_string())?;
        let skewed = |symbol: &str| {
            if skew {
                symbol.replacen("chop_comm_qwen38_", "chop_comm_qwen38_skew_", 1)
            } else {
                symbol.to_owned()
            }
        };
        let mut fixtures = Vec::new();
        for (rank, s) in sessions.iter().enumerate() {
            let mut b = BTreeMap::new();
            for (id, values) in [
                (1, codes(ROWS * WIDE, 1)),
                (2, codes(ROWS * 4, 2)),
                (3, codes(WIDE, 3)),
            ] {
                b.insert(id, s.upload_bf16(&values)?);
            }
            b.insert(
                4,
                s.allocate(qwen38::standalone_inbox_bytes(ranks as u32, ROWS as u32) as usize)?,
            );
            b.insert(5, bytes(s, &vec![0; general_sizes.flags_bytes as usize])?);
            b.insert(6, bytes(s, &vec![0; general_sizes.epoch_bytes as usize])?);
            b.insert(7, bytes(s, &[0; 4])?);
            b.insert(8, s.allocate(sizes.inbox_bytes as usize)?);
            b.insert(9, s.allocate(flag_bytes(&sizes))?);
            b.insert(10, s.allocate(sizes.epoch_bytes as usize)?);
            for (k, shape) in calls.iter().enumerate() {
                let k = k as u64;
                b.insert(
                    100 + k,
                    s.upload_bf16(&partials(ROWS * WIDTH, 16 * rank + k as usize + 4))?,
                );
                b.insert(200 + 2 * k, s.allocate(ROWS * WIDE * 2)?);
                b.insert(201 + 2 * k, s.allocate(ROWS * WIDE * 2)?);
                b.insert(400 + k, bytes(s, &shape.live.to_ne_bytes())?);
            }
            b.insert(300, s.allocate(ROWS * WIDE * 2)?);
            b.insert(301, s.allocate(ROWS * WIDE * 2)?);
            fixtures.push(b);
        }
        let (decode_cubin, wide_cubin) = if skew {
            ("comm_qwen38_decode_skew", "comm_qwen38_decode_wide_skew")
        } else {
            ("comm_qwen38_decode", "comm_qwen38_decode_wide")
        };
        let wide_symbol = skewed(wide_symbol);
        let decode_symbols = decode_symbols.map(skewed);
        // Load and resolve every entry before any launch.
        let general = CollectiveModules::load(sessions, "comm_qwen38")?;
        general.function(qwen38::SYMBOL)?;
        let decode = CollectiveModules::load(sessions, decode_cubin)?;
        for symbol in &decode_symbols {
            decode.function(symbol)?;
        }
        let wide = CollectiveModules::load(sessions, wide_cubin)?;
        wide.function(&wide_symbol)?;
        let lab = Lab {
            sessions,
            fixtures,
            general,
            decode,
            wide,
            decode_symbols,
            wide_symbol,
            sizes,
            calls,
            workers: Vec::new(),
        };
        sync(sessions)?;
        Ok(lab)
    }

    fn bindings(&self, rank: usize) -> launch::Bindings<'_> {
        let mut buffers: BTreeMap<_, _> =
            self.fixtures[rank].iter().map(|(&k, v)| (k, v)).collect();
        for (peer, fixture) in self.fixtures.iter().enumerate() {
            buffers.insert(peer as u64 + 11, &fixture[&8]);
            buffers.insert(peer as u64 + 21, &fixture[&9]);
            buffers.insert(peer as u64 + 31, &fixture[&4]);
            buffers.insert(peer as u64 + 41, &fixture[&5]);
        }
        launch::Bindings { buffers }
    }

    /// Call k on the shared state, on every rank or only on `only_rank`.
    /// Per-rank worker counts launch rank by rank, as one call takes one
    /// grid shape on every rank.
    fn call(&self, only_rank: Option<usize>, k: usize, pdl: bool) -> LabResult<()> {
        let shape = self.calls[k];
        if shape.wide && only_rank.is_none() && !self.workers.is_empty() {
            return kernel_lab::topology::each_rank(self.sessions.len(), |rank| {
                self.call(Some(rank), k, pdl)
            });
        }
        let ranks = self.sessions.len();
        let k = k as u64;
        let call = |rank: usize| qwen38_decode::Call {
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
            rows: shape.rows,
            rank: rank as u32,
        };
        let (modules, symbol, plan) = if shape.wide {
            (
                &self.wide,
                &self.wide_symbol,
                qwen38_decode::wide_plan(shape.rows),
            )
        } else {
            (
                &self.decode,
                &self.decode_symbols[usize::from(shape.rows <= LL_MAX_ROWS)],
                qwen38_decode::plan(shape.rows),
            )
        };
        let mut plan = plan.map_err(|e| e.to_string())?;
        plan.pdl = pdl;
        if let (true, Some(&workers)) = (shape.wide, only_rank.and_then(|r| self.workers.get(r))) {
            plan.grid[0] = 8 + workers.next_multiple_of(8);
            // Keep the worker count within the target layout.
            kernels_sm120::comm::wide_fits_target_layout(&plan).map_err(|e| e.to_string())?;
        }
        launch::launch_collective(self.sessions, modules, symbol, plan, only_rank, |rank| {
            let args = if shape.wide {
                qwen38_decode::wide_arguments(call(rank))
            } else {
                qwen38_decode::arguments(call(rank))
            }
            .map_err(|e| e.to_string())?;
            Ok(launch::RankCall {
                args,
                bindings: self.bindings(rank),
            })
        })
    }

    /// The general entry's 256 rows of every call's input, each on poisoned
    /// outputs.
    fn expected(&self) -> LabResult<Vec<Vec<[Vec<u16>; 2]>>> {
        let ranks = self.sessions.len();
        let mut expected = vec![Vec::new(); ranks];
        for (rank, s) in self.sessions.iter().enumerate() {
            self.fixtures[rank][&7]
                .copy_from_bytes(&(ROWS as u32).to_ne_bytes(), &s.stream)
                .map_err(|e| e.to_string())?;
        }
        for k in 0..self.calls.len() as u64 {
            poison_ids(self.sessions, &self.fixtures, &[300, 301])?;
            let mut plan =
                qwen38::plan(general_query(ranks), ROWS as u32).map_err(|e| e.to_string())?;
            plan.pdl = false;
            launch::launch_collective(
                self.sessions,
                &self.general,
                qwen38::SYMBOL,
                plan,
                None,
                |rank| {
                    let args = qwen38::standalone_arguments(qwen38::Call {
                        input: 100 + k,
                        residual: 1,
                        logits: 2,
                        weight: 3,
                        residual_out: 300,
                        normed: 301,
                        inboxes: slots(31, ranks),
                        flags: slots(41, ranks),
                        epoch: 6,
                        live_rows: 7,
                        capacity: ROWS as u32,
                        rank: rank as u32,
                    })
                    .map_err(|e| e.to_string())?;
                    Ok(launch::RankCall {
                        args,
                        bindings: self.bindings(rank),
                    })
                },
            )?;
            sync(self.sessions)?;
            for (rank, s) in self.sessions.iter().enumerate() {
                let rows = |id: u64| {
                    live_rows(
                        s,
                        &self.fixtures[rank][&id],
                        ROWS,
                        &format!("general entry call {k} rank {rank} output {id}"),
                    )
                };
                expected[rank].push([rows(300)?, rows(301)?]);
            }
        }
        Ok(expected)
    }

    /// Zeroed LL region, garbage elsewhere, zero flags, the given epoch
    /// seeds, and poisoned outputs.
    fn reset(&self, ll: u32, one_shot: u32) -> LabResult<()> {
        for (rank, s) in self.sessions.iter().enumerate() {
            let f = &self.fixtures[rank];
            let mut inbox = vec![0xff; self.sizes.inbox_bytes as usize];
            inbox[..self.sizes.zeroed_inbox_bytes as usize].fill(0);
            let mut epochs = vec![ll; 32];
            epochs.extend([one_shot, 0]);
            let epochs: Vec<u8> = epochs.iter().flat_map(|v| v.to_ne_bytes()).collect();
            for (id, data) in [
                (8, inbox),
                (9, vec![0; flag_bytes(&self.sizes)]),
                (10, epochs),
            ] {
                f[&id]
                    .copy_from_bytes(&data, &s.stream)
                    .map_err(|e| e.to_string())?;
            }
        }
        let outputs = (0..self.calls.len() as u64)
            .flat_map(|k| [200 + 2 * k, 201 + 2 * k])
            .collect::<Vec<_>>();
        poison_ids(self.sessions, &self.fixtures, &outputs)
    }

    /// Every call's live rows equal the general entry's and its dead rows
    /// keep their poison, on every rank.
    fn check(&self, expected: &[Vec<[Vec<u16>; 2]>], what: &str) -> LabResult<()> {
        sync(self.sessions)?;
        for (rank, s) in self.sessions.iter().enumerate() {
            for (k, shape) in self.calls.iter().enumerate() {
                let live = shape.live as usize;
                for (half, want) in expected[rank][k].iter().enumerate() {
                    let id = 200 + 2 * k as u64 + half as u64;
                    let got = live_rows(
                        s,
                        &self.fixtures[rank][&id],
                        live,
                        &format!("{what}: rank {rank} call {k} {shape:?} output {half}"),
                    )?;
                    if got[..] != want[..live * WIDE] {
                        return Err(format!(
                            "{what}: rank {rank} call {k} {shape:?} differs from the general entry"
                        ));
                    }
                }
            }
        }
        Ok(())
    }

    /// The sequence once eagerly, then as one captured graph per rank
    /// replayed three times, from each pair of epoch seeds; every pass must
    /// match the general entry.
    fn run(&self, seeds: &[(u32, u32)], label: &str) -> LabResult<()> {
        let expected = self.expected()?;
        for &(ll, one_shot) in seeds {
            let seeded = format!("{label}, LL seed {ll:#x}, one-shot seed {one_shot:#x}");
            self.reset(ll, one_shot)?;
            for k in 0..self.calls.len() {
                self.call(None, k, true)?;
            }
            self.check(&expected, &format!("eager, {seeded}"))?;
            let replays = RankReplay::new(self.sessions, |rank, _| {
                (0..self.calls.len()).try_for_each(|k| self.call(Some(rank), k, true))
            })?;
            for replay in 0..3 {
                self.reset(ll.wrapping_add(replay), one_shot.wrapping_add(replay))?;
                replays.launch(self.sessions)?;
                self.check(&expected, &format!("graph replay {replay}, {seeded}"))?;
            }
        }
        Ok(())
    }
}

/// Every package row count at full, partial, single and zero live rows,
/// each wide call followed by a decode call on the same state, and wide
/// calls of 16 and 32 rows, which take the one-shot at every rank count.
fn reference_calls() -> Vec<Shape> {
    let mut calls = vec![
        Shape {
            wide: true,
            rows: 16,
            live: 11,
        },
        Shape {
            wide: true,
            rows: 32,
            live: 32,
        },
    ];
    for rows in PACKAGE_ROWS {
        for live in [rows, rows - 1, 1, 0] {
            calls.push(Shape {
                wide: true,
                rows,
                live,
            });
        }
        calls.push(Shape {
            wide: false,
            rows: 16,
            live: 11,
        });
    }
    calls
}

#[test]
fn wide_hyper_connection_matches_general_entry() {
    gpu_test_multi(1 << 30, |sessions| {
        let ranks = rank_count(sessions)?;
        let calls = reference_calls();
        let lab = Lab::new(sessions, calls.clone(), false)?;
        lab.run(&[(0, 0)], "reference")?;
        let symbol = qwen38_decode::wide_symbol(ranks as u32).map_err(|e| e.to_string())?;
        for (rank, session) in sessions.iter().enumerate() {
            session.receipt(
                "qwen38-comm-wide",
                BTreeMap::from([
                    (
                        "case".into(),
                        json!("wide_hyper_connection_matches_general_entry"),
                    ),
                    (
                        "artifact".into(),
                        evidence::identity("comm_qwen38_decode_wide")?,
                    ),
                    ("symbol".into(), json!(symbol)),
                    ("ranks".into(), json!(ranks)),
                    ("rank".into(), json!(rank)),
                    ("calls".into(), json!(calls)),
                    ("bitwise_equal_to".into(), json!(qwen38::SYMBOL)),
                    ("graph_replays".into(), json!(3)),
                ]),
            )?;
        }
        Ok(())
    })
    .expect("Qwen wide decode HC all-reduce");
}

/// The worker count is a launch knob: rank 0 with 8 workers and the others
/// with 128 give the general entry's outputs, at both protocols.
#[test]
fn wide_hyper_connection_takes_any_worker_count_per_rank() {
    let calls: Vec<Shape> = [(33, 33), (256, 200), (16, 11), (128, 128)]
        .into_iter()
        .map(|(rows, live)| Shape {
            wide: true,
            rows,
            live,
        })
        .collect();
    gpu_test_multi(1 << 30, |sessions| {
        let mut lab = Lab::new(sessions, calls.clone(), false)?;
        lab.workers = (0..sessions.len())
            .map(|rank| if rank == 0 { 8 } else { 128 })
            .collect();
        lab.run(&[(0, 0)], "per-rank workers")
    })
    .expect("Qwen wide decode HC with per-rank worker counts");
}

/// Wide and decode calls in one sequence on one state, with dead rows and
/// zero live rows between busy calls, under the skew build's lag (the rank
/// chosen by the epoch waits 20 us before each push and wait) and from
/// epoch seeds near the u32 and 2^31 wraps.
#[test]
fn wide_hyper_connection_sync_g1() {
    let calls: Vec<Shape> = [
        (true, 33, 33),
        (false, 1, 1),
        (true, 256, 256),
        (true, 64, 0),
        (false, 32, 32),
        (true, 128, 100),
        (false, 2, 0),
        (true, 255, 255),
        (true, 33, 1),
        (false, 16, 16),
        (true, 100, 0),
        (true, 256, 129),
        (false, 4, 3),
        (true, 32, 30),
        (true, 64, 64),
        (true, 16, 0),
        (true, 40, 40),
    ]
    .into_iter()
    .map(|(wide, rows, live)| Shape { wide, rows, live })
    .collect();
    gpu_test_multi(1 << 30, |sessions| {
        let ranks = rank_count(sessions)?;
        let lab = Lab::new(sessions, calls.clone(), true)?;
        let seeds = [(u32::MAX - 3, 0x7fff_fffd), (u32::MAX - 3, u32::MAX - 3)];
        lab.run(&seeds, "skew")?;
        sessions[0].receipt(
            "qwen38-comm-wide",
            BTreeMap::from([
                ("case".into(), json!("wide_hyper_connection_sync_g1")),
                (
                    "artifact".into(),
                    evidence::identity("comm_qwen38_decode_wide_skew")?,
                ),
                ("ranks".into(), json!(ranks)),
                ("sequence".into(), json!(calls)),
                ("lag_ns".into(), json!(20_000)),
                ("bitwise_equal_to".into(), json!(qwen38::SYMBOL)),
            ]),
        )?;
        Ok(())
    })
    .expect("Qwen wide decode HC synchronization");
}

/// Per-call time of the wide entry against the phased pair (reduce-scatter
/// and gather on the prefill state, which graphs above 32 rows use today)
/// at every package row count with all rows live. Needs distinct devices
/// and `CHOPINFER_KERNEL_BENCH=1`; `CHOPINFER_KERNEL_BENCH_ROWS` takes a
/// list, `CHOPINFER_KERNEL_BENCH_WIDE_WORKERS` overrides the worker CTAs.
#[test]
fn wide_hyper_connection_bench() {
    if std::env::var("CHOPINFER_KERNEL_BENCH").as_deref() != Ok("1") {
        kernel_lab::skip("Qwen wide comm benchmark: set CHOPINFER_KERNEL_BENCH=1").unwrap();
        return;
    }
    const CALLS: usize = 16;
    let rows_list: Vec<u32> = std::env::var("CHOPINFER_KERNEL_BENCH_ROWS")
        .ok()
        .map(|value| {
            value
                .split(',')
                .map(|rows| rows.trim().parse().expect("integer row count"))
                .collect()
        })
        .unwrap_or_else(|| PACKAGE_ROWS.to_vec());
    let workers: Option<u32> = std::env::var("CHOPINFER_KERNEL_BENCH_WIDE_WORKERS")
        .ok()
        .and_then(|value| value.parse().ok());
    gpu_test_multi(1 << 30, |sessions| {
        let ranks = rank_count(sessions)?;
        let calls: Vec<Shape> = rows_list
            .iter()
            .map(|&rows| Shape {
                wide: true,
                rows,
                live: rows,
            })
            .collect();
        let lab = Lab::new(sessions, calls.clone(), false)?;
        lab.reset(0, 0)?;
        // The phased pair on the prefill state, capacity 256.
        let query = general_query(ranks);
        let phased_sizes = qwen38::buffers(query).map_err(|e| e.to_string())?;
        let mut phased_state = Vec::new();
        for s in sessions {
            phased_state.push([
                s.allocate(phased_sizes.inbox_bytes as usize)?,
                bytes(s, &vec![0; phased_sizes.flags_bytes as usize])?,
                bytes(s, &vec![0; phased_sizes.epoch_bytes as usize])?,
            ]);
        }
        // The phased pair's entries, from the lab's comm_qwen38 modules.
        lab.general.function(qwen38::SCATTER_SYMBOL)?;
        lab.general.function(qwen38::GATHER_SYMBOL)?;
        sync(sessions)?;
        for (k, shape) in calls.iter().enumerate() {
            let mut results = Vec::new();
            for arm in ["wide", "phased"] {
                // One call on rank `only`'s stream.
                let issue = |only: usize| -> LabResult<()> {
                    if arm == "wide" {
                        let mut plan =
                            qwen38_decode::wide_plan(shape.rows).map_err(|e| e.to_string())?;
                        if let Some(workers) = workers {
                            plan.grid[0] = 8 + workers.max(1).next_multiple_of(8);
                        }
                        // Keep the worker count within the target layout.
                        kernels_sm120::comm::wide_fits_target_layout(&plan).map_err(|e| e.to_string())?;
                        return launch::launch_collective(
                            sessions,
                            &lab.wide,
                            &lab.wide_symbol,
                            plan,
                            Some(only),
                            |rank| {
                                let args = qwen38_decode::wide_arguments(qwen38_decode::Call {
                                    input: 100 + k as u64,
                                    residual: 1,
                                    logits: 2,
                                    weight: 3,
                                    residual_out: 200,
                                    normed: 201,
                                    inboxes: slots(11, ranks),
                                    flags: slots(21, ranks),
                                    epochs: 10,
                                    live_rows: 400 + k as u64,
                                    rows: shape.rows,
                                    rank: rank as u32,
                                })
                                .map_err(|e| e.to_string())?;
                                Ok(launch::RankCall {
                                    args,
                                    bindings: lab.bindings(rank),
                                })
                            },
                        );
                    }
                    let phased = |rank: usize| -> LabResult<_> {
                        let mut bindings = lab.bindings(rank);
                        for (peer, state) in phased_state.iter().enumerate() {
                            bindings.buffers.insert(peer as u64 + 31, &state[0]);
                            bindings.buffers.insert(peer as u64 + 41, &state[1]);
                        }
                        bindings.buffers.insert(6, &phased_state[rank][2]);
                        let launches = qwen38::launches(
                            query,
                            shape.rows,
                            qwen38::Call {
                                input: 100 + k as u64,
                                residual: 1,
                                logits: 2,
                                weight: 3,
                                residual_out: 300,
                                normed: 301,
                                inboxes: slots(31, ranks),
                                flags: slots(41, ranks),
                                epoch: 6,
                                live_rows: 400 + k as u64,
                                capacity: ROWS as u32,
                                rank: rank as u32,
                            },
                        )
                        .map_err(|e| e.to_string())?;
                        Ok((launches, bindings))
                    };
                    for (index, item) in phased(only)?.0.into_iter().enumerate() {
                        launch::launch_collective(
                            sessions,
                            &lab.general,
                            item.symbol,
                            item.plan,
                            Some(only),
                            |rank| {
                                let (mut launches, bindings) = phased(rank)?;
                                Ok(launch::RankCall {
                                    args: launches.swap_remove(index).arguments,
                                    bindings,
                                })
                            },
                        )?;
                    }
                    Ok(())
                };
                let graphs = sessions
                    .iter()
                    .enumerate()
                    .map(|(rank, s)| s.capture(|_| (0..CALLS).try_for_each(|_| issue(rank))))
                    .collect::<LabResult<Vec<_>>>()?;
                let timing = kernel_lab::time_multi_graph(sessions, &graphs, CALLS, 2, 7)?;
                results.push((
                    arm,
                    f64::from(timing.critical_path.median_ms) * 1000.0 / CALLS as f64,
                ));
                // One more replay on poisoned outputs, whose rows are checked
                // below.
                let outputs = if arm == "wide" { [200, 201] } else { [300, 301] };
                poison_ids(sessions, &lab.fixtures, &outputs)?;
                for (s, graph) in sessions.iter().zip(&graphs) {
                    graph.launch(&s.stream).map_err(|e| e.to_string())?;
                }
                sync(sessions)?;
            }
            // Outputs of both arms on this call's input agree on every rank
            // on the live rows, and neither wrote past them.
            for (rank, s) in sessions.iter().enumerate() {
                for (wide_id, phased_id) in [(200, 300), (201, 301)] {
                    let rows = shape.live as usize;
                    let label = |id: u64| format!("{} rows rank {rank} output {id}", shape.rows);
                    let wide = live_rows(s, &lab.fixtures[rank][&wide_id], rows, &label(wide_id))?;
                    let phased =
                        live_rows(s, &lab.fixtures[rank][&phased_id], rows, &label(phased_id))?;
                    if wide != phased {
                        return Err(format!(
                            "{} rows rank {rank}: wide and phased outputs differ",
                            shape.rows
                        ));
                    }
                }
            }
            let wide_symbol =
                qwen38_decode::wide_symbol(ranks as u32).map_err(|e| e.to_string())?;
            for (arm, us) in &results {
                sessions[0].receipt(
                    "qwen38-comm-wide-bench",
                    BTreeMap::from([
                        ("arm".into(), json!(arm)),
                        ("symbol".into(), json!(wide_symbol)),
                        (
                            "protocol".into(),
                            json!(format!(
                                "{:?}",
                                qwen38_decode::wide_protocol(ranks as u32, shape.rows)
                                    .map_err(|e| e.to_string())?
                            )),
                        ),
                        ("rows".into(), json!(shape.rows)),
                        ("ranks".into(), json!(ranks)),
                        ("workers".into(), json!(workers)),
                        ("calls_per_graph".into(), json!(CALLS)),
                        ("median_us_per_call".into(), json!(us)),
                        ("time_over_wide".into(), json!(us / results[0].1)),
                    ]),
                )?;
            }
            eprintln!(
                "qwen wide rows {} ranks {ranks} workers {workers:?}: wide {:.1} us, phased {:.1} us",
                shape.rows,
                results[0].1,
                results[1].1
            );
        }
        Ok(())
    })
    .expect("Qwen wide decode HC benchmark");
}

#[allow(dead_code)]
#[path = "common/evidence.rs"]
mod evidence;
