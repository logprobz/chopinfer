//! The latent attention's column merge (`kernels_sm120::latent::
//! merge_columns`) against the split merge it replaces, at the latent lab
//! shape (`kernel_lab::latent_shapes`: 16 heads, latent 512, 64 splits), on
//! the GPU. Run with `CHOPINFER_KERNEL_GPU=<ordinal> cargo test -p
//! kernel-lab --release --test latent_merge_columns -- --test-threads=1`;
//! without a GPU only the CPU contract test runs.
//!
//! Each case writes a workspace of split partials (maximum, denominator and
//! numerator per row, split and head): ordinary rows, rows with empty
//! splits (the exact empty partial -inf, +0, +0), all-empty rows, rows whose
//! maxima spread far enough that most weights underflow, rows whose maxima
//! are signed zeros, and rows whose only live split is the last.
//!
//! - Every splitting entry's column merge (decode and verify, BF16 and FP8
//!   pages, normalized and partial outputs) writes the split merge's outputs
//!   bit for bit at buckets of 1 to 256 rows.
//! - Poisoned replays of each served form at every live count 0..=capacity
//!   of the capacities 1, 4, 8 and 32 (256 sampled by
//!   `poison::live_counts`), from one graph captured at the capacity: eager
//!   (compared with the split merge at that live count), relaunch, the graph
//!   and PDL behind the live-count producer; the receipt holds the covered
//!   live counts and each one's phases, asserted complete.
//! - Mutants, each caught only when an output differs from the served
//!   merge's bits: the last split left out (`_skip_last`), the numerators
//!   folded in descending split order (`_descending`), and the dependency
//!   wait after the reads (`_late_wait`, behind the live-count producer;
//!   reported inconclusive, without a cause, when it reproduces the eager
//!   bits).
//!
//! With `CHOPINFER_KERNEL_BENCH=1`, `latent_merge_columns_timing` times
//! graphs of eight merges of either kind at 1 to 256 rows.

#[path = "common/support.rs"]
mod support;

use std::collections::BTreeMap;

use kernel_core::contract::{DType, LaunchPlan};
use kernel_lab::latent_shapes::{self, GEOMETRY, PROFILE};
use kernel_lab::poison::{self, LiveProducer, Replay, WriteMap};
use kernel_lab::{LabResult, Session, launch};
use kernels_sm120::latent::merge_columns::{self, MUTANTS, cubin_name};
use kernels_sm120::latent::{Entry, MAX_SPLITS};
use runtime_cuda::{DeviceBuffer, Function, KernelArgument, Module};
use serde_json::{Value, json};
use support::{Rng, download, upload};

const HEADS: usize = GEOMETRY.heads as usize;
const LATENT: usize = GEOMETRY.latent as usize;
const SPLITS: usize = MAX_SPLITS as usize;
const WIDTH: usize = LATENT + 2;
/// Prepared row buckets: decode, verify items, the largest graphs.
const BUCKETS: [u32; 8] = [1, 2, 4, 6, 8, 32, 64, 256];

fn err(e: impl ToString) -> String {
    e.to_string()
}

/// The lab's splitting attention entries: decode and verify over BF16 and
/// FP8 pages, normalized and partial.
fn splitting() -> Vec<Entry> {
    latent_shapes::catalog()
        .into_iter()
        .filter(|entry| merge_columns::Entry::of(entry).is_ok())
        .collect()
}

fn partial_output(entry: &Entry) -> bool {
    entry.name.ends_with("_partial")
}

/// How one row's split partials look.
#[derive(Clone, Copy, Debug)]
enum Pattern {
    Ordinary,
    /// About a third of the splits empty.
    SomeEmpty,
    /// Every split empty.
    AllEmpty,
    /// Maxima from -100 to 100, so most weights underflow.
    Spread,
    /// Maxima +0 or -0.
    SignedZeros,
    /// Only the last split live.
    LastOnly,
}

const PATTERNS: [Pattern; 6] = [
    Pattern::Ordinary,
    Pattern::SomeEmpty,
    Pattern::Spread,
    Pattern::AllEmpty,
    Pattern::SignedZeros,
    Pattern::LastOnly,
];

/// The F32 workspace `[rows][SPLITS][HEADS][WIDTH]`.
fn workspace(rows: usize, patterns: &[Pattern], seed: u64) -> Vec<f32> {
    let mut rng = Rng(seed | 1);
    let mut words = vec![0.0_f32; rows * SPLITS * HEADS * WIDTH];
    for row in 0..rows {
        let pattern = patterns[row % patterns.len()];
        for split in 0..SPLITS {
            for head in 0..HEADS {
                let at = ((row * SPLITS + split) * HEADS + head) * WIDTH;
                let empty = match pattern {
                    Pattern::SomeEmpty => rng.next() % 3 == 0,
                    Pattern::AllEmpty => true,
                    Pattern::LastOnly => split + 1 != SPLITS,
                    _ => false,
                };
                if empty {
                    words[at] = f32::NEG_INFINITY;
                    continue;
                }
                words[at] = match pattern {
                    Pattern::Spread => rng.unit() * 100.0,
                    Pattern::SignedZeros => {
                        if rng.next() & 1 == 0 {
                            0.0
                        } else {
                            -0.0
                        }
                    }
                    _ => rng.unit() * 5.0,
                };
                let denominator = 1.0 + rng.unit().abs() * 31.0;
                words[at + 1] = denominator;
                for d in 0..LATENT {
                    words[at + 2 + d] = rng.unit() * denominator;
                }
            }
        }
    }
    words
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect()
}

/// One merge launch's operands.
struct Fixture {
    workspace: DeviceBuffer,
    live: DeviceBuffer,
    /// BF16 `out`, or the partial's numerator.
    out: DeviceBuffer,
    out_max: DeviceBuffer,
    out_den: DeviceBuffer,
    partial: bool,
    reference: Function,
    columns: Function,
    mutants: BTreeMap<&'static str, Function>,
    plan: LaunchPlan,
    columns_plan: LaunchPlan,
}

impl Fixture {
    fn new(
        session: &Session,
        modules: &Modules,
        entry: &Entry,
        rows: usize,
        patterns: &[Pattern],
        seed: u64,
    ) -> LabResult<Fixture> {
        let columns = merge_columns::Entry::of(entry).map_err(|e| e.0.to_string())?;
        let partial = partial_output(entry);
        let mut mutants = BTreeMap::new();
        for (suffix, _) in MUTANTS {
            let symbol = merge_columns::Entry {
                suffix: suffix.into(),
                ..columns.clone()
            }
            .symbol();
            mutants.insert(
                suffix,
                modules.mutants[suffix].function(&symbol).map_err(err)?,
            );
        }
        let element = if partial { 4 } else { 2 };
        Ok(Fixture {
            workspace: upload(session, &f32_bytes(&workspace(rows, patterns, seed)))?,
            live: upload(session, &(rows as u32).to_ne_bytes())?,
            out: session.allocate(rows * HEADS * LATENT * element)?,
            out_max: session.allocate((rows * HEADS * 4).max(16))?,
            out_den: session.allocate((rows * HEADS * 4).max(16))?,
            partial,
            reference: modules
                .reference
                .function(&entry.merge_symbol().expect("a splitting entry's merge"))
                .map_err(err)?,
            columns: modules.columns.function(&columns.symbol()).map_err(err)?,
            mutants,
            plan: entry
                .merge_plan(rows as u32)
                .map_err(|e| e.0.to_string())?
                .expect("a splitting entry's merge plan"),
            columns_plan: columns.plan(rows as u32).map_err(|e| e.0.to_string())?,
        })
    }

    fn set_live(&self, session: &Session, live: u32) -> LabResult<()> {
        poison::write(session, &self.live, 0, &live.to_ne_bytes())
    }

    fn launch(&self, session: &Session, function: &Function, plan: LaunchPlan) -> LabResult<()> {
        let mut arguments = vec![
            KernelArgument::Buffer(&self.workspace),
            KernelArgument::Buffer(&self.live),
        ];
        if self.partial {
            arguments.extend([
                KernelArgument::Buffer(&self.out_max),
                KernelArgument::Buffer(&self.out_den),
                KernelArgument::Buffer(&self.out),
            ]);
        } else {
            arguments.push(KernelArgument::Buffer(&self.out));
        }
        arguments.push(KernelArgument::U32(SPLITS as u32));
        function
            .launch(&session.stream, launch::config(plan)?, &arguments)
            .map_err(err)
    }

    fn reference(&self, session: &Session) -> LabResult<()> {
        self.launch(session, &self.reference, self.plan)
    }

    fn columns(&self, session: &Session) -> LabResult<()> {
        self.launch(session, &self.columns, self.columns_plan)
    }

    fn mutant(&self, session: &Session, suffix: &str) -> LabResult<()> {
        self.launch(session, &self.mutants[suffix], self.columns_plan)
    }

    /// The outputs, each with its element dtype and the bytes of `live`
    /// rows.
    fn outputs(&self, live: usize) -> Vec<(&'static str, &DeviceBuffer, DType, usize)> {
        if self.partial {
            vec![
                ("out_max", &self.out_max, DType::F32, live * HEADS * 4),
                ("out_den", &self.out_den, DType::F32, live * HEADS * 4),
                ("out_num", &self.out, DType::F32, live * HEADS * LATENT * 4),
            ]
        } else {
            vec![("out", &self.out, DType::Bf16, live * HEADS * LATENT * 2)]
        }
    }

    /// Poisons every output, runs `launch` and reads the first `live` rows
    /// of each back.
    fn run(
        &self,
        session: &Session,
        live: usize,
        launch: impl FnOnce() -> LabResult<()>,
    ) -> LabResult<Vec<Vec<u8>>> {
        session.stream.synchronize().map_err(err)?;
        for (_, buffer, _, _) in self.outputs(live) {
            poison::fill_all(session, buffer, &[poison::POISON])?;
        }
        launch()?;
        self.outputs(live)
            .into_iter()
            .map(|(_, buffer, _, bytes)| download(session, buffer, bytes))
            .collect()
    }
}

/// The lab cubins: the latent cubin's split merges, the served column
/// merges and each mutant's.
struct Modules {
    reference: Module,
    columns: Module,
    mutants: BTreeMap<&'static str, Module>,
}

impl Modules {
    fn new(session: &Session) -> LabResult<Modules> {
        let mut mutants = BTreeMap::new();
        for (suffix, _) in MUTANTS {
            mutants.insert(suffix, session.module(&cubin_name(PROFILE, suffix))?);
        }
        Ok(Modules {
            reference: session.module(&latent_shapes::cubin_name())?,
            columns: session.module(&cubin_name(PROFILE, ""))?,
            mutants,
        })
    }
}

fn same(label: &str, actual: &[Vec<u8>], expected: &[Vec<u8>]) -> LabResult<()> {
    for (index, (a, b)) in actual.iter().zip(expected).enumerate() {
        if a != b {
            let at = a.iter().zip(b).position(|(x, y)| x != y).unwrap_or(0);
            return Err(format!(
                "{label}: output {index} differs first at byte {at} of {}",
                b.len()
            ));
        }
    }
    if actual.len() != expected.len() {
        return Err(format!("{label}: output count differs"));
    }
    Ok(())
}

fn bitwise_cases(session: &Session, modules: &Modules, entry: &Entry) -> LabResult<Value> {
    let mut evidence = Vec::new();
    for rows in BUCKETS {
        let rows = rows as usize;
        let fixture = Fixture::new(
            session,
            modules,
            entry,
            rows,
            &PATTERNS,
            0x6d65 + rows as u64,
        )?;
        let reference = fixture.run(session, rows, || fixture.reference(session))?;
        let columns = fixture.run(session, rows, || fixture.columns(session))?;
        same(
            &format!(
                "{} rows {rows}: column merge against the split merge",
                entry.name
            ),
            &columns,
            &reference,
        )?;
        evidence.push(json!({"rows": rows, "grid": fixture.columns_plan.grid}));
    }
    Ok(json!({"entry": entry.name, "buckets": evidence}))
}

/// Capacities whose replays cover every live count 0..=capacity.
const EVERY_LIVE_COUNT: [usize; 4] = [1, 4, 8, 32];
const REQUIRED_PHASES: [&str; 3] = ["deterministic_bitwise", "graph_bitwise", "pdl_bitwise"];

fn replay_case(
    session: &Session,
    modules: &Modules,
    entry: &Entry,
    capacity: usize,
) -> LabResult<Value> {
    let fixture = Fixture::new(
        session,
        modules,
        entry,
        capacity,
        &PATTERNS,
        0x7265 + capacity as u64,
    )?;
    let every = EVERY_LIVE_COUNT.contains(&capacity);
    let expected: Vec<usize> = if every {
        (0..=capacity).collect()
    } else {
        poison::live_counts(capacity as u32)
            .into_iter()
            .map(|live| live as usize)
            .collect()
    };
    let graph = session.capture(|_| fixture.columns(session))?;
    let producer = LiveProducer::new(session)?;
    let mut covered = Vec::new();
    let mut evidence = Vec::new();
    for &live in &expected {
        let label = format!("{} capacity {capacity}, live {live}", entry.name);
        fixture.set_live(session, live as u32)?;
        let reference = fixture.run(session, live, || fixture.reference(session))?;
        let mut replay = Replay::new(session);
        for (name, buffer, dtype, bytes) in fixture.outputs(live) {
            replay = replay.output(name, buffer, WriteMap::of(dtype).byte_prefix(bytes)?);
        }
        replay.eager(|| fixture.columns(session))?;
        let eager = fixture
            .outputs(live)
            .into_iter()
            .map(|(_, buffer, _, bytes)| download(session, buffer, bytes))
            .collect::<LabResult<Vec<_>>>()?;
        same(
            &format!("{label}: eager column merge against the split merge"),
            &eager,
            &reference,
        )?;
        replay.relaunch(|| fixture.columns(session))?;
        replay.graph(&graph)?;
        replay.pdl(
            || producer.launch(session, &fixture.live),
            || fixture.columns(session),
            true,
        )?;
        let mut receipt = BTreeMap::new();
        replay.record(&mut receipt);
        let missing: Vec<&str> = REQUIRED_PHASES
            .into_iter()
            .filter(|phase| receipt.get(*phase) != Some(&json!(true)))
            .collect();
        if !missing.is_empty() {
            return Err(format!("{label}: replay phases {missing:?} did not pass"));
        }
        covered.push(live);
        evidence.push(json!({"live": live, "replay": receipt}));
    }
    if covered != expected {
        return Err(format!(
            "{}: covered {covered:?}, expected {expected:?}",
            entry.name
        ));
    }
    Ok(json!({
        "entry": entry.name,
        "capacity": capacity,
        "coverage": if every { "every live count 0..=capacity" } else { "sampled (poison::live_counts)" },
        "covered_live_counts": covered,
        "required_phases": REQUIRED_PHASES,
        "live_counts": evidence,
    }))
}

fn mutant_cases(session: &Session, modules: &Modules, entry: &Entry) -> LabResult<Value> {
    let mut outcomes = Vec::new();
    for rows in [1_usize, 4, 32] {
        for (suffix, patterns) in [
            ("_skip_last", &[Pattern::LastOnly][..]),
            ("_descending", &[Pattern::Ordinary][..]),
        ] {
            let fixture = Fixture::new(
                session,
                modules,
                entry,
                rows,
                patterns,
                0x5eed + rows as u64,
            )?;
            let served = fixture.run(session, rows, || fixture.columns(session))?;
            let reference = fixture.run(session, rows, || fixture.reference(session))?;
            same(
                &format!("{} {suffix} rows {rows}: served", entry.name),
                &served,
                &reference,
            )?;
            let mutant = fixture.run(session, rows, || fixture.mutant(session, suffix))?;
            if mutant == served {
                return Err(format!(
                    "{} {suffix} rows {rows}: the mutant reproduced the served bits",
                    entry.name
                ));
            }
            outcomes.push(json!({"mutant": suffix, "rows": rows, "outcome": "caught"}));
        }
        let fixture = Fixture::new(session, modules, entry, rows, &[Pattern::Ordinary], 0x1a7e)?;
        let eager = fixture.run(session, rows, || fixture.columns(session))?;
        let producer = LiveProducer::new(session)?;
        let behind = |late: bool| -> LabResult<Vec<Vec<u8>>> {
            fixture.run(session, rows, || {
                producer.launch(session, &fixture.live)?;
                if late {
                    fixture.mutant(session, "_late_wait")
                } else {
                    fixture.columns(session)
                }
            })
        };
        if behind(false)? != eager {
            return Err(format!(
                "{} rows {rows}: the served merge behind the live-count producer changed its output",
                entry.name
            ));
        }
        let outcome = if behind(true)? == eager {
            eprintln!(
                "{} _late_wait rows {rows}: inconclusive: the mutant reproduced the eager bits",
                entry.name
            );
            "inconclusive"
        } else {
            "caught"
        };
        outcomes.push(json!({"mutant": "_late_wait", "rows": rows, "outcome": outcome}));
    }
    Ok(json!({"entry": entry.name, "mutants": outcomes}))
}

#[test]
fn latent_merge_columns_cpu_contract() {
    let entries = splitting();
    // Decode and verify, BF16 and FP8 pages, normalized and partial.
    assert_eq!(entries.len(), 8);
    for entry in &entries {
        let columns = merge_columns::Entry::of(entry).unwrap();
        assert_eq!(columns.params(), entry.merge_params());
        for rows in BUCKETS {
            let plan = columns.plan(rows).unwrap();
            let reference = entry.merge_plan(rows).unwrap().unwrap();
            assert_eq!(plan.grid, [rows, HEADS as u32, (LATENT / 128) as u32]);
            assert_eq!(plan.workspace_bytes, reference.workspace_bytes);
            assert_eq!(entry.splits(), MAX_SPLITS);
        }
    }
    // Every pattern's row has the partial layout the kernels read.
    let words = workspace(PATTERNS.len(), &PATTERNS, 7);
    assert_eq!(words.len(), PATTERNS.len() * SPLITS * HEADS * WIDTH);
    let all_empty = 3 * SPLITS * HEADS * WIDTH;
    assert!(
        words[all_empty..all_empty + SPLITS * HEADS * WIDTH]
            .chunks(WIDTH)
            .all(
                |partial| partial[0] == f32::NEG_INFINITY && partial[1..].iter().all(|&x| x == 0.0)
            )
    );
}

#[test]
fn latent_merge_columns_match_the_split_merge_bitwise() {
    kernel_lab::gpu_test(4 << 30, |session| {
        let modules = Modules::new(session)?;
        let mut evidence = Vec::new();
        for entry in splitting() {
            evidence.push(bitwise_cases(session, &modules, &entry)?);
        }
        let mut receipt = BTreeMap::new();
        receipt.insert("op".into(), json!("latent_merge_columns_bitwise"));
        receipt.insert("entries".into(), json!(evidence));
        session.receipt("latent_merge_columns_bitwise", receipt)?;
        Ok(())
    })
    .expect("column merge against the split merge");
}

#[test]
fn latent_merge_columns_replay_every_live_count() {
    kernel_lab::gpu_test(4 << 30, |session| {
        let modules = Modules::new(session)?;
        let mut evidence = Vec::new();
        for entry in splitting()
            .iter()
            .filter(|entry| entry.name.contains("fp8"))
        {
            for capacity in EVERY_LIVE_COUNT.into_iter().chain([256]) {
                evidence.push(replay_case(session, &modules, entry, capacity)?);
            }
        }
        let mut receipt = BTreeMap::new();
        receipt.insert("op".into(), json!("latent_merge_columns_replay"));
        receipt.insert("cases".into(), json!(evidence));
        session.receipt("latent_merge_columns_replay", receipt)?;
        Ok(())
    })
    .expect("column merge poisoned replays at every live count");
}

#[test]
fn latent_merge_columns_mutants_are_caught() {
    kernel_lab::gpu_test(4 << 30, |session| {
        let modules = Modules::new(session)?;
        let mut evidence = Vec::new();
        for entry in splitting()
            .iter()
            .filter(|entry| entry.name.contains("fp8"))
        {
            evidence.push(mutant_cases(session, &modules, entry)?);
        }
        let mut receipt = BTreeMap::new();
        receipt.insert("op".into(), json!("latent_merge_columns_mutants"));
        receipt.insert("entries".into(), json!(evidence));
        session.receipt("latent_merge_columns_mutants", receipt)?;
        Ok(())
    })
    .expect("column merge mutants");
}

/// Graph replays of eight merges, the split merge (arm 0) or the column
/// merge (arm 1), each launch PDL after the previous as the bundle runs them.
#[test]
fn latent_merge_columns_timing() {
    if std::env::var("CHOPINFER_KERNEL_BENCH").as_deref() != Ok("1") {
        kernel_lab::skip("latent_merge_columns timing: set CHOPINFER_KERNEL_BENCH=1").unwrap();
        return;
    }
    const MERGES: usize = 8;
    kernel_lab::gpu_test(4 << 30, |session| {
        let modules = Modules::new(session)?;
        let mut rows_out = Vec::new();
        for entry in splitting()
            .iter()
            .filter(|entry| entry.name.contains("fp8"))
        {
            for rows in [1_usize, 2, 4, 8, 16, 32, 64] {
                let fixture =
                    Fixture::new(session, &modules, entry, rows, &[Pattern::Ordinary], 0x7173)?;
                let graphs = [false, true].map(|columns| {
                    session.capture(|_| {
                        for _ in 0..MERGES {
                            if columns {
                                fixture.columns(session)?;
                            } else {
                                fixture.reference(session)?;
                            }
                        }
                        Ok(())
                    })
                });
                let [reference, columns] = graphs;
                let graphs = [reference?, columns?];
                let paired = kernel_lab::time_ab(
                    session,
                    3,
                    15,
                    [kernel_lab::eviction::EvictionMode::Write; 2],
                    |arm, stream| graphs[arm].launch(stream).map_err(err),
                )?;
                let per = |ms: f32| f64::from(ms) * 1e3 / MERGES as f64;
                let row = json!({
                    "entry": entry.name,
                    "rows": rows,
                    "split_merge_us": per(paired.arms[0].median_ms),
                    "column_merge_us": per(paired.arms[1].median_ms),
                    "columns_over_split": paired.paired_ratio,
                });
                eprintln!("latent_merge_columns timing: {row}");
                rows_out.push(row);
            }
        }
        let mut receipt = BTreeMap::new();
        receipt.insert("op".into(), json!("latent_merge_columns_timing"));
        receipt.insert("buckets".into(), json!(rows_out));
        session.receipt("latent_merge_columns_timing", receipt)?;
        Ok(())
    })
    .expect("column merge timing");
}
