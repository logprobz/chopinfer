//! The bundle's shared BF16 transports on frozen real F32 rank partials.
//! No timing claims: exact transport, graph replay, row and chunk invariance.
use bundle_deepseek41::sm120::fold_cubin_for_symbol;
use kernel_core::{
    backend::{Backend, Counts, Need, Prepared, View},
    codec::{bf16_to_f32, f32_to_bf16},
    collective::{AllReduceBuffers, RankViews},
    contract::{CommQuery, DType},
    oracle,
};
use kernel_harness::{
    LabResult, Session,
    launch::{self, Bindings},
    topology,
};
use kernels_sm120::backend::Sm120;
use runtime_cuda::{DeviceBuffer, Function, Module};
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
};

const WIDTH: usize = 5120;
const REAL_CASES: usize = 249;
const CAPACITY: u32 = 8192;
const SYMBOLS: [&str; 5] = [
    "chop_comm_allreduce_phased_bf16_v1",
    "chop_comm_allreduce_phased_oneshot_bf16_v1",
    "chop_comm_allreduce_phased_turns_bf16_v1",
    "chop_comm_allreduce_phased_round_f32_v1",
    "chop_comm_allreduce_phased_oneshot_round_f32_v1",
];
type Op = Prepared<AllReduceBuffers<View>>;

fn err(e: impl std::fmt::Display) -> String {
    e.to_string()
}
fn root() -> PathBuf {
    std::env::var_os("CHOPINFER_BF16_PARTIAL_INPUTS")
        .expect("frozen inputs required")
        .into()
}
fn regular(path: &Path) -> LabResult<Vec<u8>> {
    if !path.symlink_metadata().map_err(err)?.file_type().is_file() {
        return Err("expected regular artifact".into());
    }
    std::fs::read(path).map_err(err)
}
fn manifest() -> LabResult<Vec<Value>> {
    let value: Value =
        serde_json::from_slice(&regular(&root().join("manifest.json"))?).map_err(err)?;
    let cases = value["cases"].as_array().ok_or("missing frozen cases")?;
    if cases.len() != REAL_CASES || value["capture_count"] != 9 {
        return Err("real input coverage differs".into());
    }
    Ok(cases.clone())
}
fn inputs(case: &Value) -> LabResult<Vec<Vec<f32>>> {
    let name = case["file"].as_str().ok_or("input filename")?;
    if Path::new(name).file_name().and_then(|p| p.to_str()) != Some(name) {
        return Err("input must be a local filename".into());
    }
    let bytes = regular(&root().join(name))?;
    let rows = case["rows"].as_u64().ok_or("input rows")? as usize;
    if rows == 0
        || rows > 40
        || case["width"] != WIDTH
        || case["ranks"] != 4
        || bytes.len() != 4 * rows * WIDTH * 4
        || case["bytes"] != bytes.len()
        || case["sha256"] != kernel_build::digest(&bytes)
    {
        return Err("input shape or digest differs".into());
    }
    let values: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    if values.iter().any(|x| !x.is_finite()) {
        return Err("nonfinite captured input".into());
    }
    Ok(values
        .chunks_exact(rows * WIDTH)
        .map(<[f32]>::to_vec)
        .collect())
}
fn op(rank: usize, candidate: bool, capacity: u32) -> LabResult<Op> {
    let (backend, query) = if candidate {
        (
            Sm120::new().with_phased_bf16_all_reduce(),
            CommQuery::AllReduce {
                ranks: 4,
                elements: WIDTH as u64,
                dtype: DType::Bf16,
                fused_residual: false,
                fused_rmsnorm: None,
            },
        )
    } else {
        (
            Sm120::new().with_phased_f32_all_reduce(),
            CommQuery::AllReduceRounded {
                ranks: 4,
                elements: WIDTH as u64,
            },
        )
    };
    backend
        .all_reduce(&query, rank as u32, Need::rows(capacity))
        .map_err(err)
}

struct Rank {
    input: DeviceBuffer,
    output: DeviceBuffer,
    live: DeviceBuffer,
    state: Vec<DeviceBuffer>,
}
struct Fixture {
    capacity: u32,
    ranks: Vec<Rank>,
    ops: Vec<Op>,
    functions: Vec<BTreeMap<String, Function>>,
    _modules: Vec<Module>,
}
impl Fixture {
    fn new(sessions: &[Session], candidate: bool, capacity: u32) -> LabResult<Self> {
        let mut modules = Vec::new();
        let mut functions = Vec::new();
        let mut ranks = Vec::new();
        let mut ops = Vec::new();
        for (rank, s) in sessions.iter().enumerate() {
            let prepared = op(rank, candidate, capacity)?;
            let mut fns = BTreeMap::new();
            for name in SYMBOLS {
                let (_, cubin) = fold_cubin_for_symbol(name)
                    .ok_or_else(|| format!("missing served symbol {name}"))?;
                let module = s.load_cubin(cubin)?;
                fns.insert(name.into(), module.function(name).map_err(err)?);
                modules.push(module);
            }
            let state = prepared
                .info()
                .state
                .iter()
                .map(|&n| s.allocate(n as usize))
                .collect::<LabResult<Vec<_>>>()?;
            for b in &state {
                b.copy_from_bytes(&vec![0; b.byte_len()], &s.stream)
                    .map_err(err)?;
            }
            ranks.push(Rank {
                input: s.allocate(capacity as usize * WIDTH * 4)?,
                output: s.allocate(capacity as usize * WIDTH * 2)?,
                live: s.allocate(4)?,
                state,
            });
            functions.push(fns);
            ops.push(prepared);
        }
        synchronize(sessions)?;
        Ok(Self {
            capacity,
            ranks,
            ops,
            functions,
            _modules: modules,
        })
    }
    fn upload(&self, sessions: &[Session], values: &[Vec<f32>], live: u32) -> LabResult<()> {
        for (r, s) in self.ranks.iter().zip(sessions).enumerate() {
            let (buffers, session) = s;
            buffers
                .input
                .copy_from_f32(&values[r], &session.stream)
                .map_err(err)?;
            buffers
                .live
                .copy_from_bytes(&live.to_ne_bytes(), &session.stream)
                .map_err(err)?;
            buffers
                .output
                .copy_from_bytes(
                    &vec![0xa5; self.capacity as usize * WIDTH * 2],
                    &session.stream,
                )
                .map_err(err)?;
        }
        synchronize(sessions)
    }
    fn launch(&self, s: &Session, rank: usize, rows: u32) -> LabResult<()> {
        let r = &self.ranks[rank];
        let mut map = BTreeMap::from([(1, &r.input), (2, &r.output), (3, &r.live)]);
        for (peer, r) in self.ranks.iter().enumerate() {
            map.insert(100 + peer as u64, &r.output);
            for (i, b) in r.state.iter().enumerate() {
                map.insert(1000 + peer as u64 * 100 + i as u64, b);
            }
        }
        let view = |id| View {
            handle: id,
            bytes: map[&id].byte_len() as u64,
        };
        let buffers = AllReduceBuffers {
            input: view(1),
            output: view(2),
            live_rows: Some(view(3)),
            ranks: (0..4)
                .map(|peer| RankViews {
                    output: view(100 + peer),
                    state: (0..r.state.len())
                        .map(|i| view(1000 + peer * 100 + i as u64))
                        .collect(),
                })
                .collect(),
        };
        for call in self.ops[rank]
            .launches(&buffers, View::EMPTY, Counts::rows(rows))
            .map_err(err)?
        {
            if !SYMBOLS.contains(&call.symbol.as_str()) {
                return Err("unexpected composed fallback".into());
            }
            launch::launch(
                &s.context,
                &self.functions[rank][&call.symbol],
                &s.stream,
                call.plan,
                &call.arguments,
                &Bindings {
                    buffers: map.clone(),
                },
            )?;
        }
        Ok(())
    }
    fn check(&self, sessions: &[Session], expected: &[u16], live: usize) -> LabResult<()> {
        synchronize(sessions)?;
        for (r, s) in self.ranks.iter().zip(sessions) {
            let actual = s.download_bf16(&r.output, self.capacity as usize * WIDTH)?;
            if actual[..live * WIDTH] != expected[..live * WIDTH] {
                return Err("GPU differs from the exact rank-ordered oracle".into());
            }
            if actual[live * WIDTH..].iter().any(|&x| x != 0xa5a5) {
                return Err("GPU overwrote a dead row".into());
            }
        }
        Ok(())
    }
    fn exercise(
        &self,
        sessions: &[Session],
        values: &[Vec<f32>],
        rows: u32,
        live: u32,
        candidate: bool,
    ) -> LabResult<()> {
        let expected = if candidate {
            oracle::all_reduce_bf16_partials(values)
        } else {
            oracle::all_reduce_rounded(values)
        }
        .map_err(err)?;
        self.upload(sessions, values, live)?;
        topology::each_rank(4, |r| self.launch(&sessions[r], r, rows))?;
        self.check(sessions, &expected, live as usize)?;
        let graph = topology::RankReplay::new(sessions, |r, s| self.launch(s, r, rows))?;
        for _ in 0..3 {
            self.upload(sessions, values, live)?;
            graph.launch(sessions)?;
            self.check(sessions, &expected, live as usize)?;
        }
        Ok(())
    }
}
fn synchronize(sessions: &[Session]) -> LabResult<()> {
    for s in sessions {
        s.stream.synchronize().map_err(err)?;
    }
    Ok(())
}

fn f64_bounds(values: &[Vec<f32>]) -> LabResult<Value> {
    let original: Vec<Vec<f64>> = values
        .iter()
        .map(|r| r.iter().map(|&x| f64::from(x)).collect())
        .collect();
    let rounded: Vec<Vec<f64>> = values
        .iter()
        .map(|r| {
            r.iter()
                .map(|&x| f64::from(bf16_to_f32(f32_to_bf16(x))))
                .collect()
        })
        .collect();
    let old = oracle::all_reduce_rounded(values).map_err(err)?;
    let new = oracle::all_reduce_bf16_partials(values).map_err(err)?;
    let original = oracle::all_reduce(&original, None, None).map_err(err)?;
    let rounded = oracle::all_reduce(&rounded, None, None).map_err(err)?;
    let mut old_outside = 0;
    let mut new_outside = 0;
    let mut rounded_outside = 0;
    for i in 0..old.len() {
        old_outside += usize::from(
            (f64::from(bf16_to_f32(old[i])) - original.values[i]).abs()
                > oracle::numeric_bound(&original, i, DType::Bf16),
        );
        new_outside += usize::from(
            (f64::from(bf16_to_f32(new[i])) - original.values[i]).abs()
                > oracle::numeric_bound(&original, i, DType::Bf16),
        );
        rounded_outside += usize::from(
            (f64::from(bf16_to_f32(new[i])) - rounded.values[i]).abs()
                > oracle::numeric_bound(&rounded, i, DType::Bf16),
        );
    }
    if old_outside != 0 || rounded_outside != 0 {
        return Err("a transport's own input contract exceeds its f64 bound".into());
    }
    Ok(
        json!({"elements": old.len(), "current_outside_original_bound": old_outside,
        "candidate_outside_original_bound_report_only": new_outside, "candidate_outside_rounded_input_bound": rounded_outside}),
    )
}

#[test]
fn declared_entries_and_oracle_witness() {
    for symbol in SYMBOLS {
        let (_, bytes) = fold_cubin_for_symbol(symbol).expect("bundle embeds the shared entry");
        assert!(
            bytes
                .windows(symbol.len() + 1)
                .any(|w| w[..symbol.len()] == *symbol.as_bytes() && w[symbol.len()] == 0)
        );
    }
    for candidate in [false, true] {
        for rank in 0..4 {
            op(rank, candidate, CAPACITY).unwrap();
        }
    }
    let witness = vec![
        vec![f32::from_bits(0x3f80_8000)],
        vec![-1.0],
        vec![0.0],
        vec![0.0],
    ];
    assert_eq!(
        f64_bounds(&witness).unwrap()["candidate_outside_original_bound_report_only"],
        1
    );
}

#[test]
#[ignore = "requires a staged directory named by CHOPINFER_BF16_PARTIAL_INPUTS"]
fn frozen_inputs_are_complete() {
    for case in manifest().unwrap() {
        inputs(&case).unwrap();
    }
    eprintln!("BF16_FROZEN_INPUTS cases={REAL_CASES} PASS");
}

#[test]
#[ignore = "requires a staged directory named by CHOPINFER_BF16_PARTIAL_INPUTS and four idle GPUs"]
fn real_partials_replays_and_chunks_on_gpu() {
    let cases = manifest().unwrap();
    let result = kernel_harness::gpu_test_multi(4 << 30, |sessions| {
        if sessions.len() != 4 { return Err("exactly four ranks required".into()); }
        let old = Fixture::new(sessions, false, 40)?;
        let new = Fixture::new(sessions, true, 40)?;
        let mut bounds = Vec::new();
        for case in &cases {
            let values = inputs(case)?;
            let rows = case["rows"].as_u64().unwrap() as u32;
            old.exercise(sessions, &values, rows, rows, false)?;
            new.exercise(sessions, &values, rows, rows, true)?;
            bounds.push(json!({"file":case["file"], "bounds": f64_bounds(&values)?}));
        }
        drop(old); drop(new);
        let wide = Fixture::new(sessions, true, CAPACITY)?;
        // One global row pattern is sliced at each boundary. Equality to its
        // one oracle proves row order and every tested chunk partition.
        let values: Vec<Vec<f32>> = (0..4).map(|rank| (0..CAPACITY as usize * WIDTH).map(|i| {
            let row = i / WIDTH;
            let base = ((i.wrapping_mul(1664525).wrapping_add(rank * 1013904223)) % 65536) as f32 / 32768.0 - 1.0;
            if i % 53 == 0 { [16777216.0, 1.0, 1.0, -16777216.0][rank] }
            else { base * 2.0_f32.powi((row % 9) as i32 - 4) }
        }).collect()).collect();
        let expected = oracle::all_reduce_bf16_partials(&values).map_err(err)?;
        let mut chunks = 0;
        for (start, rows, live) in [(0,8192,8192), (0,8191,8191), (0,4096,4096), (4096,4096,4096),
            (0,4095,4095), (4095,4096,4096), (0,1,1), (1,8,8), (9,32,31), (40,33,33),
            (73,1023,1023), (1096,1024,1023), (2120,32,0)] {
            let part: Vec<Vec<f32>> = values.iter().map(|r| r[start*WIDTH..(start+rows)*WIDTH].to_vec()).collect();
            wide.exercise(sessions, &part, rows as u32, live as u32, true)?;
            wide.check(sessions, &expected[start*WIDTH..(start+rows)*WIDTH], live)?;
            chunks += 1;
        }
        let report = json!({"status":"passed", "real_cases":cases.len(), "real_executions":cases.len()*2*4,
            "chunk_cases":chunks, "chunk_executions":chunks*4, "ranks":4, "bounds":bounds});
        let output = PathBuf::from(std::env::var_os("CHOPINFER_BF16_PARTIAL_REPORT").ok_or("report path required")?);
        std::fs::write(output, serde_json::to_vec_pretty(&report).map_err(err)?).map_err(err)?;
        eprintln!("BF16_GPU real_cases={} chunk_cases={chunks} PASS", cases.len());
        Ok(())
    }).unwrap();
    assert!(
        result.is_some(),
        "the staged job must execute all four GPUs"
    );
}
