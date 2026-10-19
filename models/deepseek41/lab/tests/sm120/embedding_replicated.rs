//! Actual old collective chain versus immutable peer gathers on two ranks.
use super::{
    fusion_checks::{Failures, compare_roles, live_counts},
    norm_quantize::{download, upload},
    round_artifacts as artifacts,
};
use bundle_deepseek41::sm120::replicated_embedding;
use kernel_core::{
    backend::{Backend, Counts, Need, View},
    collective::{AllReduceBuffers, RankViews},
    contract::{Argument, ArgumentList, CommQuery, LaunchPlan},
    replicated_embedding::ReplicatedEmbeddingBuffers,
};
use kernel_harness::{
    LabResult, Session,
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
const H: usize = 5120;
const ROWS: [u32; 6] = [1, 2, 4, 8, 16, 32];
const PATTERNS: [&str; 3] = ["all_codes", "finite", "edges"];
const ROLES: [&str; 4] = ["shard", "wide", "output", "expanded"];
const ROUND: &str = "chop_row_rounding_round_deepseek41_w5120_v1";
const WIDEN: &str = "chop_row_rounding_widen_deepseek41_w5120_v1";
const BROADCAST: &str = "chop_row_copy_broadcast_streams_v1";
pub(super) fn symbols() -> Vec<String> {
    [
        ROUND,
        WIDEN,
        BROADCAST,
        "chop_comm_reduce_scatter_f32_v1",
        "chop_comm_allgather_scattered_f32_v1",
    ]
    .into_iter()
    .map(str::to_owned)
    .chain([1, 2, 4].into_iter().flat_map(|r| {
        [
            format!("chop_row_embedding_deepseek41_tp{r}_v1"),
            replicated_embedding::entry(&FLASH, r).symbol,
        ]
    }))
    .collect()
}
pub(super) fn coverage() -> serde_json::Value {
    json!({"cases":ROWS.map(|rows|json!({"label":"peer_embedding","rows":rows})),"patterns":PATTERNS,"bitwise":ROWS.len()*PATTERNS.len(),"negative":ROWS.len()*ROLES.len()*2,"timings":ROWS.len(),"ranks":2})
}
fn check(rows: u32, live: u32, ranks: u32) -> LabResult<()> {
    if !ROWS.contains(&rows) || live > rows || ![1, 2, 4].contains(&ranks) {
        return Err(format!(
            "invalid embedding case rows={rows} live={live} ranks={ranks}"
        ));
    }
    Ok(())
}
fn row_plan(rows: u32) -> LabResult<LaunchPlan> {
    rounding::plan(rows).map_err(|e| e.to_string())
}
fn reduce(ranks: u32, rank: u32) -> kernel_core::backend::Prepared<AllReduceBuffers<View>> {
    Sm120::new()
        .with_partial_rounding(rounding::PartialRounding {
            width: H as u32,
            round: rounding::entry(Rounding::Round, H as u32, ROUND),
            widen: rounding::entry(Rounding::Widen, H as u32, WIDEN),
            invariance: rounding::shared_invariance(),
        })
        .all_reduce(
            &CommQuery::AllReduceRounded {
                ranks,
                elements: H as u64,
            },
            rank,
            Need::rows(32),
        )
        .unwrap()
}
struct Output {
    data: Vec<DeviceBuffer>,
    lengths: Vec<usize>,
}
impl Output {
    fn new(s: &Session, rows: u32) -> LabResult<Self> {
        let n = rows as usize * H;
        let lengths = vec![n * 2, n * 4, n * 2, n * 4 * 2];
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
fn view(handle: u64, bytes: u64) -> View {
    View { handle, bytes }
}
fn call(
    sessions: &[Session],
    all: &[Rank],
    functions: &[BTreeMap<String, &Function>],
    rank: usize,
    rows: u32,
    fused: bool,
    stream: &Stream,
) -> LabResult<()> {
    use Argument::{DeviceBuffer as D, U32};
    let s = &sessions[rank];
    let r = &all[rank];
    let output = if fused { &r.new } else { &r.old };
    let mut map = BTreeMap::from([
        (1, &r.ids),
        (2, &output.data[0]),
        (3, &output.data[1]),
        (4, &output.data[2]),
        (5, &output.data[3]),
        (6, &r.live),
    ]);
    for (peer, r) in all.iter().enumerate() {
        map.insert(10 + peer as u64, &r.table);
        map.insert(200 + peer as u64, &r.old.data[2]);
        for (i, b) in r.state.iter().enumerate() {
            map.insert(1000 + peer as u64 * 100 + i as u64, b);
        }
    }
    let v = |id| view(id, map[&id].byte_len() as u64);
    let run = |symbol: &str, p, args: &ArgumentList| {
        launch::launch(
            &s.context,
            functions[rank][symbol],
            stream,
            p,
            args,
            &Bindings {
                buffers: map.clone(),
            },
        )
    };
    if fused {
        let e = replicated_embedding::entry(&FLASH, all.len() as u32);
        let op = Sm120::new()
            .with_replicated_embedding(e.clone())
            .replicated_embedding(&e.query, rank as u32, Need::rows(32))
            .map_err(|e| e.to_string())?;
        let b = ReplicatedEmbeddingBuffers {
            tables: (0..all.len()).map(|r| v(10 + r as u64)).collect(),
            ids: v(1),
            shard: v(2),
            wide: v(3),
            output: v(4),
            expanded: v(5),
            live_rows: v(6),
        };
        for l in op
            .launches(&b, View::EMPTY, Counts::rows(rows))
            .map_err(|e| e.to_string())?
        {
            run(&l.symbol, l.plan, &l.arguments)?;
        }
    } else {
        let shard = FLASH.shape.vocab / all.len() as u32;
        run(
            &format!("chop_row_embedding_deepseek41_tp{}_v1", all.len()),
            row_plan(rows)?,
            &ArgumentList(vec![
                D(10 + rank as u64),
                D(1),
                D(2),
                D(6),
                U32(rank as u32 * shard),
                U32(shard),
            ]),
        )?;
        run(
            WIDEN,
            row_plan(rows)?,
            &ArgumentList(vec![D(2), D(3), D(6)]),
        )?;
        let op = reduce(all.len() as u32, rank as u32);
        let b = AllReduceBuffers {
            input: v(3),
            output: v(4),
            live_rows: Some(v(6)),
            ranks: (0..all.len())
                .map(|r| RankViews {
                    output: v(200 + r as u64),
                    state: (0..op.info().state.len())
                        .map(|i| v(1000 + r as u64 * 100 + i as u64))
                        .collect(),
                })
                .collect(),
        };
        for l in op
            .launches(&b, View::EMPTY, Counts::rows(rows))
            .map_err(|e| e.to_string())?
        {
            run(&l.symbol, l.plan, &l.arguments)?;
        }
        run(
            BROADCAST,
            row_plan(rows)?,
            &ArgumentList(vec![D(4), D(5), D(6), U32((H * 2 / 16) as u32), U32(4)]),
        )?;
    }
    Ok(())
}
fn inputs(sessions: &[Session], all: &[Rank], rows: u32, pattern: &str) -> LabResult<()> {
    let shard = FLASH.shape.vocab / all.len() as u32;
    for (rank, s) in sessions.iter().enumerate() {
        for local in (0..16).chain([shard - 1]) {
            let bytes: Vec<u8> = (0..H)
                .flat_map(|c| {
                    let at = local as usize * H + c;
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
                .copy_from_bytes_at(local as usize * H * 2, &bytes, &s.stream)
                .map_err(|e| e.to_string())?;
        }
        let edges = [
            -1,
            0,
            shard as i32 - 1,
            shard as i32,
            FLASH.shape.vocab as i32 - 1,
            FLASH.shape.vocab as i32,
            i32::MAX,
            i32::MIN,
        ];
        let ids: Vec<u8> = (0..rows as usize)
            .flat_map(|r| {
                let id = if pattern == "edges" {
                    edges[r % edges.len()]
                } else {
                    (r % all.len()) as i32 * shard as i32 + (r / all.len()) as i32 % 16
                };
                id.to_ne_bytes()
            })
            .collect();
        upload(&all[rank].ids, &ids, &s.stream)?;
    }
    Ok(())
}
#[test]
fn parameters_symbols_and_old_launch_chain_are_complete() {
    for ranks in [1, 2, 4] {
        for rows in ROWS {
            for live in live_counts(rows) {
                check(rows, live, ranks).unwrap();
            }
            assert!(check(rows, rows + 1, ranks).is_err());
        }
    }
    assert!(check(0, 0, 2).is_err());
    assert!(check(33, 0, 2).is_err());
    assert!(check(1, 1, 3).is_err());
    for sym in symbols() {
        assert!(
            bundle_deepseek41::sm120::fold_cubin_for_symbol(&sym).is_some(),
            "{sym}"
        );
    }
    assert_eq!(FLASH.shape.hidden as usize, H);
    assert_eq!(FLASH.shape.hc.streams, 4);
}
#[test]
fn bitwise_peer_graphs_and_timing_on_gpu() {
    for rows in ROWS {
        for live in live_counts(rows) {
            check(rows, live, 2).unwrap();
        }
    }
    if kernel_harness::selected_ordinals().unwrap().is_none() {
        kernel_harness::skip("replicated embedding needs two idle GPUs").unwrap();
        return;
    }
    let result =
        kernel_harness::gpu_test_multi_with(artifacts::timing_cubins().unwrap(), 2 << 30, |s| {
            compare_on_gpu(s, false)
        })
        .unwrap();
    if kernel_harness::selected_ordinals().unwrap().is_some() {
        assert!(result.is_some());
    }
}
fn compare_on_gpu(sessions: &[Session], profile: bool) -> LabResult<()> {
    if sessions.len() != 2 {
        return Err("the staged peer matrix requires exactly two ranks".into());
    }
    let loaded = sessions
        .iter()
        .map(|s| artifacts::load(s, &symbols()))
        .collect::<LabResult<Vec<_>>>()?;
    let names = symbols();
    let functions: Vec<BTreeMap<_, _>> = loaded
        .iter()
        .map(|l| names.iter().cloned().zip(l.functions.iter()).collect())
        .collect();
    let mut failures = Failures::new("EMBED");
    let mut bitwise = 0;
    let mut negative = 0;
    let mut timings = Vec::new();
    for rows in ROWS {
        let mark = failures.len();
        let label = format!("peer_embedding rows={rows}");
        let all = sessions
            .iter()
            .enumerate()
            .map(|(rank, s)| {
                let op = reduce(sessions.len() as u32, rank as u32);
                let state = op
                    .info()
                    .state
                    .iter()
                    .map(|&n| s.allocate(n as usize))
                    .collect::<LabResult<Vec<_>>>()?;
                for b in &state {
                    upload(b, &vec![0; b.byte_len()], &s.stream)?;
                }
                Ok(Rank {
                    table: s.allocate(FLASH.shape.vocab as usize / sessions.len() * H * 2)?,
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
                        s.capture(|stream| {
                            call(sessions, &all, &functions, rank, rows, arm == 1, stream)
                        })
                    })
                    .collect::<LabResult<Vec<_>>>()
            })
            .collect::<LabResult<Vec<_>>>()?;
        if profile {
            inputs(sessions, &all, rows, "finite")?;
            for (rank, s) in sessions.iter().enumerate() {
                upload(&all[rank].live, &rows.to_ne_bytes(), &s.stream)?;
            }
            for _ in 0..128 {
                for graphs in &graphs {
                    for (rank, s) in sessions.iter().enumerate() {
                        graphs[rank].launch(&s.stream).map_err(|e| e.to_string())?;
                    }
                }
            }
            for s in sessions {
                s.stream.synchronize().map_err(|e| e.to_string())?;
            }
            eprintln!("EMBED_PROFILE rows={rows} calls=128");
            continue;
        }
        for pattern in PATTERNS {
            let case_mark = failures.len();
            inputs(sessions, &all, rows, pattern)?;
            for live in live_counts(rows) {
                check(rows, live, 2)?;
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
                                call(
                                    sessions,
                                    &all,
                                    &functions,
                                    rank,
                                    rows,
                                    arm == 1,
                                    &sessions[rank].stream,
                                )
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
                            "{label} pattern={pattern} live={live} replay={replay} rank={rank}"
                        );
                        failures.record(&context, compare_roles(&ROLES, &old, &new));
                        for data in [&old, &new] {
                            for (role, bytes) in data.iter().enumerate() {
                                failures.require(
                                    &format!("{context} role={}", ROLES[role]),
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
                "EMBED_BITWISE {label} pattern={pattern} {}",
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
                    compare_roles(&ROLES, &original, &all[rank].new.read(s)?).is_err(),
                    "mutated output escaped",
                );
                upload(&all[rank].new.data[role], &original[role], &s.stream)?;
                negative += 1;
            }
        }
        if failures.len() == mark {
            inputs(sessions, &all, rows, "finite")?;
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
                                    call(sessions, &all, &functions, rank, rows, arm == 1, stream)?;
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
                "EMBED_TIMING {label} old_us={} new_us={} ratio={}",
                us[0],
                us[1],
                us[1] / us[0]
            );
            timings.push(json!({"rows":rows,"old_us":us[0],"new_us":us[1]}));
        }
    }
    if profile {
        return Ok(());
    }
    sessions[0].receipt(
        "embedding-replicated",
        BTreeMap::from([
            ("bitwise".into(), json!(bitwise)),
            ("negative".into(), json!(negative)),
            ("timings".into(), json!(timings)),
            ("mismatches".into(), json!(failures.messages())),
        ]),
    )?;
    eprintln!(
        "EMBED_COMPLETE bitwise={bitwise} negative={negative} timings={} mismatches={}",
        timings.len(),
        failures.len()
    );
    failures.finish()
}

#[test]
fn exclusive_profile_on_gpu() {
    if kernel_harness::selected_ordinals().unwrap().is_none() {
        kernel_harness::skip("embedding profile needs two idle GPUs").unwrap();
        return;
    }
    assert!(
        kernel_harness::gpu_test_multi_with(artifacts::timing_cubins().unwrap(), 2 << 30, |s| {
            compare_on_gpu(s, true)
        })
        .unwrap()
        .is_some()
    );
}
