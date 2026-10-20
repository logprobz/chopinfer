//! The drafter's Markov embedding: the old shard gather, widening and
//! composed rounded all-reduce against the peer lookup that replaces them,
//! on every selected rank, eagerly and in graph replays, with paired timing.
use super::{
    dspark_entries::rounding_symbol,
    fusion_checks::{Failures, compare_roles, live_counts},
    norm_quantize::{download, upload},
};
use bundle_deepseek41::sm120::{dspark_peer_embedding, fold_cubin_for_symbol};
use kernel_core::{
    backend::{Backend, Counts, Launch, Need, Prepared, View},
    collective::{AllReduceBuffers, RankViews},
    contract::{Argument, ArgumentList, CommQuery},
    replicated_embedding::ReplicatedEmbeddingBuffers,
};
use kernel_harness::{
    Cubins, LabResult, Session,
    launch::{self, Bindings},
    topology,
};
use kernels_sm120::{
    backend::Sm120,
    row::rounding::{self, Rounding},
};
use model_deepseek41_contract::geometry::FLASH;
use runtime_cuda::{DeviceBuffer, Function, Stream};
use serde_json::json;
use std::collections::BTreeMap;

const W: usize = 256;
const ROWS: [u32; 6] = [1, 2, 4, 8, 16, 32];
/// Eight ID sets of 32 rows reach 256 table rows per owner: every BF16
/// code under the `all_codes` fill.
const SETS: u32 = 8;
const ROLES: [&str; 3] = ["shard", "wide", "output"];
const PROBE_SYMBOLS: &[&str] = &[
    "chop_probe_gate_v1",
    "chop_probe_gate_release_v1",
    "chop_probe_flush_v1",
    "chop_probe_read_v1",
];

fn embedding_symbol(ranks: u32) -> String {
    format!("chop_row_dspark_markov_{}_tp{ranks}_v1", FLASH.kernel_tag)
}

fn shard(ranks: u32) -> u32 {
    FLASH.shape.vocab / ranks
}

fn reduce(ranks: u32, rank: u32) -> Prepared<AllReduceBuffers<View>> {
    let entry = |kind| rounding::entry(kind, W as u32, rounding_symbol(&FLASH, kind));
    Sm120::new()
        .with_partial_rounding(rounding::PartialRounding {
            width: W as u32,
            round: entry(Rounding::Round),
            widen: entry(Rounding::Widen),
            invariance: rounding::shared_invariance(),
        })
        .all_reduce(
            &CommQuery::AllReduceRounded {
                ranks,
                elements: W as u64,
            },
            rank,
            Need::rows(32),
        )
        .unwrap()
}

fn view(handle: u64, bytes: u64) -> View {
    View { handle, bytes }
}

/// Buffer ids: 1 ids, 2 shard, 3 wide, 4 output, 5 broadcast copy, 6 live
/// count, 10 + r rank r's table, 200 + r rank r's old output, 1000 + 100 r
/// + i rank r's reduction state i.
fn views(rows: u32, ranks: u32) -> BTreeMap<u64, u64> {
    let n = u64::from(rows) * W as u64;
    let mut v = BTreeMap::from([
        (1, u64::from(rows) * 4),
        (2, n * 2),
        (3, n * 4),
        (4, n * 2),
        (5, n * 2),
        (6, 4),
    ]);
    for r in 0..u64::from(ranks) {
        v.insert(10 + r, u64::from(shard(ranks)) * W as u64 * 2);
        v.insert(200 + r, n * 2);
        for (i, &bytes) in reduce(ranks, r as u32).info().state.iter().enumerate() {
            v.insert(1000 + r * 100 + i as u64, bytes);
        }
    }
    v
}

/// Each arm's launches on `rank`: the old chain, then the peer lookup.
fn launches(rows: u32, ranks: u32, rank: u32) -> LabResult<[Vec<Launch>; 2]> {
    use Argument::{DeviceBuffer as D, U32};
    let sizes = views(32, ranks);
    let v = |id: u64| view(id, sizes[&id]);
    let plan = rounding::plan(rows).map_err(|e| e.to_string())?;
    let raw = |symbol: String, arguments: Vec<Argument>| Launch {
        symbol,
        plan,
        rows,
        arguments: ArgumentList(arguments),
        collective: false,
        persistent: false,
    };
    let mut old = vec![
        raw(
            embedding_symbol(ranks),
            vec![
                D(10 + u64::from(rank)),
                D(1),
                D(2),
                D(6),
                U32(rank * shard(ranks)),
                U32(shard(ranks)),
            ],
        ),
        raw(
            rounding_symbol(&FLASH, Rounding::Widen),
            vec![D(2), D(3), D(6)],
        ),
    ];
    let op = reduce(ranks, rank);
    let buffers = AllReduceBuffers {
        input: v(3),
        output: v(4),
        live_rows: Some(v(6)),
        ranks: (0..u64::from(ranks))
            .map(|r| RankViews {
                output: v(200 + r),
                state: (0..op.info().state.len() as u64)
                    .map(|i| v(1000 + r * 100 + i))
                    .collect(),
            })
            .collect(),
    };
    old.extend(
        op.launches(&buffers, View::EMPTY, Counts::rows(rows))
            .map_err(|e| e.to_string())?,
    );
    let entry = dspark_peer_embedding::entry(&FLASH, ranks);
    let peer = Sm120::new()
        .with_replicated_embedding(entry.clone())
        .replicated_embedding(&entry.query, rank, Need::rows(32))
        .map_err(|e| e.to_string())?;
    let buffers = ReplicatedEmbeddingBuffers {
        tables: (0..u64::from(ranks)).map(|r| v(10 + r)).collect(),
        ids: v(1),
        shard: v(2),
        wide: v(3),
        output: v(4),
        expanded: v(5),
        live_rows: v(6),
    };
    let new = peer
        .launches(&buffers, View::EMPTY, Counts::rows(rows))
        .map_err(|e| e.to_string())?;
    Ok([old, new])
}

#[test]
fn dspark_peer_embedding_cpu_replaces_the_five_launch_chain() {
    for ranks in [1, 2, 4] {
        for rows in ROWS {
            for rank in 0..ranks {
                let [old, new] = launches(rows, ranks, rank).unwrap();
                let symbols: Vec<_> = old.iter().map(|l| l.symbol.as_str()).collect();
                assert_eq!(
                    symbols,
                    [
                        embedding_symbol(ranks).as_str(),
                        "chop_deepseek41_dspark_markov_widen_v1",
                        "chop_comm_reduce_scatter_f32_v1",
                        "chop_comm_allgather_scattered_f32_v1",
                        "chop_deepseek41_dspark_markov_round_v1",
                    ]
                );
                assert_eq!(new.len(), 1);
                assert_eq!(
                    new[0].symbol,
                    dspark_peer_embedding::entry(&FLASH, ranks).symbol
                );
                assert_eq!(new[0].plan.grid, [rows, 1, 1]);
                assert!(!new[0].collective);
                for launch in old.iter().chain(&new) {
                    assert!(
                        fold_cubin_for_symbol(&launch.symbol).is_some(),
                        "{} is embedded",
                        launch.symbol
                    );
                }
            }
        }
    }
    assert_eq!(FLASH.shape.dspark.markov_rank as usize, W);
}

/// Old and new outputs of one rank: shard, wide, output, broadcast copy.
struct Output {
    data: Vec<DeviceBuffer>,
    lengths: Vec<usize>,
}

impl Output {
    fn new(s: &Session, rows: u32) -> LabResult<Self> {
        let n = rows as usize * W;
        let lengths = vec![n * 2, n * 4, n * 2, n * 2];
        Ok(Self {
            data: lengths
                .iter()
                .map(|&n| s.allocate(n))
                .collect::<LabResult<_>>()?,
            lengths,
        })
    }
    fn reset(&self, s: &Session) -> LabResult<()> {
        for (b, &n) in self.data.iter().zip(&self.lengths) {
            upload(b, &vec![0xa5; n], &s.stream)?;
        }
        Ok(())
    }
    fn read(&self, s: &Session) -> LabResult<Vec<Vec<u8>>> {
        self.data
            .iter()
            .zip(&self.lengths)
            .map(|(b, &n)| download(b, n, &s.stream))
            .collect()
    }
}

struct Rank {
    table: DeviceBuffer,
    ids: DeviceBuffer,
    live: DeviceBuffer,
    old: Output,
    new: Output,
    state: Vec<DeviceBuffer>,
}

struct Loaded {
    _modules: Vec<runtime_cuda::Module>,
    functions: BTreeMap<String, Function>,
}

fn load(s: &Session, symbols: &[String]) -> LabResult<Loaded> {
    let mut modules = Vec::new();
    let mut functions = BTreeMap::new();
    for symbol in symbols {
        let (_, bytes) = fold_cubin_for_symbol(symbol).ok_or("unembedded symbol")?;
        let module = s.load_cubin(bytes)?;
        functions.insert(
            symbol.clone(),
            module.function(symbol).map_err(|e| e.to_string())?,
        );
        modules.push(module);
    }
    Ok(Loaded {
        _modules: modules,
        functions,
    })
}

fn call(
    all: &[Rank],
    s: &Session,
    loaded: &Loaded,
    rank: usize,
    rows: u32,
    arm: usize,
    stream: &Stream,
) -> LabResult<()> {
    let r = &all[rank];
    let output = if arm == 1 { &r.new } else { &r.old };
    let mut map = BTreeMap::from([
        (1, &r.ids),
        (2, &output.data[0]),
        (3, &output.data[1]),
        (4, &output.data[2]),
        (5, &output.data[3]),
        (6, &r.live),
    ]);
    for (peer, p) in all.iter().enumerate() {
        map.insert(10 + peer as u64, &p.table);
        map.insert(200 + peer as u64, &p.old.data[2]);
        for (i, b) in p.state.iter().enumerate() {
            map.insert(1000 + peer as u64 * 100 + i as u64, b);
        }
    }
    let arms = launches(rows, all.len() as u32, rank as u32)?;
    for l in &arms[arm] {
        launch::launch(
            &s.context,
            &loaded.functions[&l.symbol],
            stream,
            l.plan,
            &l.arguments,
            &Bindings {
                buffers: map.clone(),
            },
        )?;
    }
    Ok(())
}

/// Table rows 0..256 and the last row of every shard, and each rank's IDs.
fn inputs(sessions: &[Session], all: &[Rank], rows: u32, pattern: &str, set: u32) -> LabResult<()> {
    let ranks = all.len() as u32;
    let shard = shard(ranks);
    for (rank, s) in sessions.iter().enumerate() {
        for local in (0..256).chain([shard - 1]) {
            let bytes: Vec<u8> = (0..W)
                .flat_map(|c| {
                    let at = local as usize * W + c + rank * 977;
                    let code = if pattern == "finite" {
                        0x3c00 + (at % 2048) as u16
                    } else {
                        at as u16
                    };
                    code.to_ne_bytes()
                })
                .collect();
            all[rank]
                .table
                .copy_from_bytes_at(local as usize * W * 2, &bytes, &s.stream)
                .map_err(|e| e.to_string())?;
        }
        let vocab = FLASH.shape.vocab as i32;
        let edges = [
            -1,
            0,
            shard as i32 - 1,
            shard as i32,
            vocab - 1,
            vocab,
            i32::MAX,
            i32::MIN,
        ];
        let ids: Vec<u8> = (0..rows)
            .flat_map(|r| {
                let id = if pattern == "edges" {
                    edges[r as usize % edges.len()]
                } else {
                    let owner = (r + set) % ranks;
                    (owner * shard + (set * 32 + r) % 256) as i32
                };
                id.to_ne_bytes()
            })
            .collect();
        upload(&all[rank].ids, &ids, &s.stream)?;
    }
    Ok(())
}

#[test]
fn dspark_peer_embedding_gpu_matches_the_old_chain() {
    if kernel_harness::selected_ordinals().unwrap().is_none() {
        kernel_harness::skip("the peer embedding comparison needs idle GPUs").unwrap();
        return;
    }
    let probe = std::env::var_os("CHOPINFER_DSPARK_PEER_PROBE")
        .map(|path| &*Box::leak(std::fs::read(path).unwrap().into_boxed_slice()));
    let cubins = probe.map_or(Cubins::NONE, |probe| {
        Cubins::new(
            Box::leak(Box::new([("probe_sm120", probe)])),
            &[("probe_sm120", PROBE_SYMBOLS)],
        )
    });
    let result =
        kernel_harness::gpu_test_multi_with(cubins, 1 << 30, |s| compare(s, probe.is_some()))
            .unwrap();
    assert!(result.is_some(), "the comparison ran");
}

fn compare(sessions: &[Session], timed: bool) -> LabResult<()> {
    let ranks = sessions.len() as u32;
    if ![2, 4].contains(&ranks) {
        return Err(format!(
            "the peer comparison runs on 2 or 4 ranks, not {ranks}"
        ));
    }
    let symbols: Vec<String> = {
        let mut names: Vec<String> = (0..ranks)
            .flat_map(|rank| launches(32, ranks, rank).unwrap())
            .flatten()
            .map(|l| l.symbol)
            .collect();
        names.sort();
        names.dedup();
        names
    };
    let loaded = sessions
        .iter()
        .map(|s| load(s, &symbols))
        .collect::<LabResult<Vec<_>>>()?;
    let mut failures = Failures::new("PEER");
    let mut bitwise = 0;
    let mut negative = 0;
    let mut timings = Vec::new();
    for rows in ROWS {
        let mark = failures.len();
        let label = format!("peer_embedding ranks={ranks} rows={rows}");
        let all = sessions
            .iter()
            .enumerate()
            .map(|(rank, s)| {
                let state = reduce(ranks, rank as u32)
                    .info()
                    .state
                    .iter()
                    .map(|&n| s.allocate(n as usize))
                    .collect::<LabResult<Vec<_>>>()?;
                for b in &state {
                    upload(b, &vec![0; b.byte_len()], &s.stream)?;
                }
                Ok(Rank {
                    table: s.allocate(shard(ranks) as usize * W * 2)?,
                    ids: s.allocate(rows as usize * 4)?,
                    live: s.allocate(4)?,
                    old: Output::new(s, rows)?,
                    new: Output::new(s, rows)?,
                    state,
                })
            })
            .collect::<LabResult<Vec<_>>>()?;
        let graphs = (0..2)
            .map(|arm| {
                sessions
                    .iter()
                    .enumerate()
                    .map(|(rank, s)| {
                        s.capture(|stream| call(&all, s, &loaded[rank], rank, rows, arm, stream))
                    })
                    .collect::<LabResult<Vec<_>>>()
            })
            .collect::<LabResult<Vec<_>>>()?;
        let cases = (0..SETS)
            .map(|set| ("all_codes", set))
            .chain([("finite", 0), ("edges", 0)]);
        for (pattern, set) in cases {
            let case_mark = failures.len();
            inputs(sessions, &all, rows, pattern, set)?;
            for live in live_counts(rows) {
                for (rank, s) in sessions.iter().enumerate() {
                    upload(&all[rank].live, &live.to_ne_bytes(), &s.stream)?;
                }
                for replay in 0..3 {
                    for (rank, s) in sessions.iter().enumerate() {
                        all[rank].old.reset(s)?;
                        all[rank].new.reset(s)?;
                    }
                    for (arm, gs) in graphs.iter().enumerate() {
                        if replay == 0 {
                            topology::each_rank(sessions.len(), |rank| {
                                let s = &sessions[rank];
                                call(&all, s, &loaded[rank], rank, rows, arm, &s.stream)
                            })?;
                        } else {
                            for (rank, s) in sessions.iter().enumerate() {
                                gs[rank].launch(&s.stream).map_err(|e| e.to_string())?;
                            }
                        }
                    }
                    for s in sessions {
                        s.stream.synchronize().map_err(|e| e.to_string())?;
                    }
                    for (rank, s) in sessions.iter().enumerate() {
                        let old = all[rank].old.read(s)?;
                        let new = all[rank].new.read(s)?;
                        let context = format!(
                            "{label} pattern={pattern} set={set} live={live} replay={replay} rank={rank}"
                        );
                        failures.record(&context, compare_roles(&ROLES, &old[..3], &new[..3]));
                        let active = W * 2 * live as usize;
                        failures.require(
                            &format!("{context} role=copy"),
                            new[3][..active] == new[2][..active],
                            "broadcast copy differs from the output",
                        );
                        for (arm, data) in [&old, &new].into_iter().enumerate() {
                            for (role, bytes) in data.iter().enumerate() {
                                if arm == 0 && role == 3 {
                                    continue;
                                }
                                failures.require(
                                    &format!("{context} arm={arm} role={role}"),
                                    bytes[bytes.len() / rows as usize * live as usize..]
                                        .iter()
                                        .all(|&b| b == 0xa5),
                                    "inactive output changed",
                                );
                            }
                        }
                    }
                }
            }
            bitwise += 1;
            eprintln!(
                "PEER_BITWISE {label} pattern={pattern} set={set} {}",
                if failures.len() == case_mark {
                    "PASS"
                } else {
                    "FAIL"
                }
            );
        }
        for (rank, s) in sessions.iter().enumerate() {
            let original = all[rank].new.read(s)?;
            for role in 0..ROLES.len() {
                let mut changed = original[role].clone();
                changed[0] ^= 1;
                upload(&all[rank].new.data[role], &changed, &s.stream)?;
                failures.require(
                    &format!("{label} rank={rank} negative={}", ROLES[role]),
                    compare_roles(&ROLES, &original[..3], &all[rank].new.read(s)?[..3]).is_err(),
                    "mutated output escaped",
                );
                upload(&all[rank].new.data[role], &original[role], &s.stream)?;
                negative += 1;
            }
        }
        if timed && failures.len() == mark {
            inputs(sessions, &all, rows, "finite", 0)?;
            for (rank, s) in sessions.iter().enumerate() {
                upload(&all[rank].live, &rows.to_ne_bytes(), &s.stream)?;
            }
            let gs = (0..2)
                .map(|arm| {
                    sessions
                        .iter()
                        .enumerate()
                        .map(|(rank, s)| {
                            s.capture(|stream| {
                                for _ in 0..32 {
                                    call(&all, s, &loaded[rank], rank, rows, arm, stream)?;
                                }
                                Ok(())
                            })
                        })
                        .collect::<LabResult<Vec<_>>>()
                })
                .collect::<LabResult<Vec<_>>>()?;
            let mut us = Vec::new();
            for (arm, graphs) in gs.iter().enumerate() {
                let t = kernel_harness::with_timing_label(&format!("{label} arm={arm}"), || {
                    kernel_harness::time_multi_graph(sessions, graphs, 32, 3, 15)
                })?;
                us.push(t.critical_path.median_ms * 1000. / 32.);
            }
            eprintln!(
                "PEER_TIMING {label} old_us={:.3} new_us={:.3} ratio={:.3}",
                us[0],
                us[1],
                us[1] / us[0]
            );
            timings.push(json!({"ranks": ranks, "rows": rows, "old_us": us[0], "new_us": us[1]}));
        }
    }
    sessions[0].receipt(
        "dspark-peer-embedding",
        BTreeMap::from([
            ("ranks".into(), json!(ranks)),
            ("bitwise".into(), json!(bitwise)),
            ("negative".into(), json!(negative)),
            ("timings".into(), json!(timings)),
            ("mismatches".into(), json!(failures.messages())),
        ]),
    )?;
    eprintln!(
        "PEER_COMPLETE ranks={ranks} bitwise={bitwise} negative={negative} timings={} mismatches={}",
        timings.len(),
        failures.len()
    );
    failures.finish()
}
