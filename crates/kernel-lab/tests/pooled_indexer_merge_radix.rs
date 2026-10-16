//! The pooled indexer's radix merge (`kernels_sm120::pooled_indexer::
//! merge_radix`) against the global selection's `select_merge` and the tree
//! merge it replaces, at the lab profile (`pooled_indexer_shapes.rs`;
//! README "Radix merge" in crates/kernels-sm120/cuda/pooled_indexer), on
//! the GPU. Run with `CHOPINFER_KERNEL_GPU=<ordinal> cargo test -p
//! kernel-lab --release --test pooled_indexer_merge_radix --
//! --test-threads=1`; without a GPU only the CPU contract test runs.
//!
//! Each case writes the workspace `select_split` writes: every visible pool
//! gets a score, each split keeps the best `top_k / 4` keys of its pool range
//! descending, then zeros. Scores come from a few values, so most are tied,
//! with NaN and both zeros among them; one case adds rows with both
//! infinities and rows with no visible pool; rows that select without
//! scoring and rows past the live count hold all-ones lists the merges must
//! not read. The radix merge reads a second copy of the workspace in which
//! every word it must not use (the zero padding and the lists of empty
//! splits) is all-ones, the largest key, so a use of a word past a list's
//! kept keys or of an empty split changes the selection; a row of at most
//! `CACHE_KEYS` candidates, which the radix select takes, also has each
//! list's kept keys shuffled (the tree fold past that needs them sorted).
//!
//! - The key lists of the radix merge, the tree merge and `select_merge` are
//!   equal bit for bit, and equal to the host's selection of the same keys,
//!   at every served row bucket's split count and at split counts from 1 to
//!   64 (powers of two or not), runtime top-k from 4 to 2048 tokens, and
//!   contexts up to 262,144 pools, with candidate counts on both sides of
//!   `CACHE_KEYS` (the radix select and the tree fold).
//! - Poisoned replays of the radix merge at every live count 0..=capacity of
//!   the capacities 1, 4, 6, 16, 32, 64, 128 and 192 (the bundle runs a
//!   split merge below 376 rows; together they take every served split
//!   count from 64 to 2), each capacity's graph captured once and replayed
//!   at every live count: eager, relaunch, the capacity graph and PDL behind
//!   the live-count producer. Each live count's eager output is compared
//!   with `select_merge`'s and the tree merge's at that live count, and the
//!   receipt holds the
//!   covered live counts and each one's passed phases, asserted complete.
//!   2048 rows (one split) replays at `poison::live_counts`' sample only, and
//!   its receipt says so.
//! - Mutants, each caught only when an output differs from the served
//!   merge's bits (launch, synchronization and readback errors propagate):
//!   the last live split's list left out (`_skip_last`, on rows whose best
//!   pool is in the last split), equal scores ranking the higher pool first
//!   (`_ties_high`, on rows of equal scores), and the dependency wait after
//!   the reads of the live count and the lists (`_late_wait`, behind the
//!   lab's live-count producer; reported inconclusive, without a cause, when
//!   the mutant reproduces the eager bits), each on rows the radix select
//!   takes and on rows the tree fold takes.
//!
//! With `CHOPINFER_KERNEL_BENCH=1`, `pooled_indexer_merge_radix_timing`
//! times graphs of eight tree or radix merges at the decode, verify and
//! prefill buckets over 3K- to 128K-token contexts.

#[path = "../pooled_indexer_shapes.rs"]
mod shapes;

#[path = "common/support.rs"]
mod support;

use std::collections::{BTreeMap, BTreeSet};

use kernel_core::contract::{DType, LaunchPlan};
use kernel_lab::poison::{self, LiveProducer, Replay, WriteMap};
use kernel_lab::{LabResult, Session, launch};
use kernels_sm120::pooled_indexer::merge_radix::{self, CACHE_KEYS, cubin_name};
use kernels_sm120::pooled_indexer::merge_tree;
use kernels_sm120::pooled_indexer::{Entry, Kind, TILE_POOLS};
use runtime_cuda::{DeviceBuffer, Function, KernelArgument, Module};
use serde_json::{Value, json};
use support::{Rng, bytes_i32, download, upload};

const GROUP: usize = 4;
const MAX_KEPT: usize = 512;
const WIDTH: usize = 2051;
const ROW_BYTES: usize = WIDTH * 4;
/// The fill of key list outputs: no position, and not -1.
const LIST_FILL: [u8; 4] = [0xa5, 0x5a, 0xc3, 0x3c];
/// Prepared row buckets: decode, verify items of 6 rows, prefill chunks.
const BUCKETS: [u32; 12] = [1, 2, 3, 4, 8, 16, 32, 64, 128, 192, 512, 2048];

fn err(e: impl ToString) -> String {
    e.to_string()
}

/// The lab's global selection entry, the one the radix merge serves.
fn selection() -> Entry {
    shapes::lab_entries()
        .into_iter()
        .find(|entry| matches!(entry.kind, Kind::Select { ranks: None }))
        .expect("the lab profile has a global selection")
}

/// `candidate_key` in common.cuh.
fn candidate_key(score: f32, pool: u32) -> u64 {
    let bits = if score == 0.0 { 0 } else { score.to_bits() };
    let ordered = if score.is_nan() {
        0
    } else if bits & 0x8000_0000 != 0 {
        !bits
    } else {
        bits | 0x8000_0000
    };
    u64::from(ordered) << 32 | u64::from(!pool)
}

/// `split_range` in select.cuh over one rank's `pools`.
fn split_range(pools: usize, split: usize, splits: usize) -> (usize, usize) {
    let chunk = pools.div_ceil(splits);
    let aligned = chunk.div_ceil(TILE_POOLS as usize) * TILE_POOLS as usize;
    let begin = pools.min(split * aligned);
    (begin, pools.min(begin + aligned))
}

/// How one row scores its pools.
#[derive(Clone, Copy, Debug)]
enum Scores {
    /// Eight values, so most pools tie, with NaN and both zeros among them.
    Tied,
    /// Distinct-ish values from a wide range.
    Spread,
    /// Every pool scores the same.
    Equal,
    /// Tied, with the last visible pool the best.
    BestLast,
    /// Tied values with both infinities among them.
    Extremes,
}

impl Scores {
    fn score(self, rng: &mut Rng, pool: usize, pools: usize) -> f32 {
        const TIED: [f32; 8] = [1.5, -0.0, 0.0, 0.25, f32::NAN, 1.5, -2.0, 0.75];
        match self {
            Scores::Tied => TIED[(rng.next() % 8) as usize],
            Scores::Spread => rng.wide(),
            Scores::Equal => 0.5,
            Scores::BestLast if pool + 1 == pools => 1e30,
            Scores::BestLast => TIED[(rng.next() % 8) as usize],
            Scores::Extremes => {
                [f32::INFINITY, f32::NEG_INFINITY, f32::NAN, 1.5, -0.0][(rng.next() % 5) as usize]
            }
        }
    }
}

/// One row of a case: its position and score pattern.
#[derive(Clone, Copy, Debug)]
struct Row {
    position: i32,
    scores: Scores,
}

/// One merge launch's operands and the host's expected outputs.
struct Case {
    label: String,
    rows: Vec<Row>,
    splits: usize,
    top_k: u32,
    /// u64 `[rows][splits][MAX_KEPT]` as `select_split` writes it, all-ones
    /// where no list is written.
    workspace: Vec<u64>,
    /// The same lists with every other word all-ones, and each list's kept
    /// keys shuffled in rows of at most `CACHE_KEYS` candidates: the radix
    /// merge's input.
    sparse: Vec<u64>,
    /// The expected key list of every row.
    expected: Vec<Vec<u8>>,
}

impl Case {
    fn new(label: String, rows: Vec<Row>, splits: usize, top_k: u32, seed: u64) -> Case {
        let kept = top_k as usize / GROUP;
        let mut rng = Rng(seed | 1);
        let mut workspace = vec![u64::MAX; rows.len() * splits * MAX_KEPT];
        let mut sparse = workspace.clone();
        let mut expected = Vec::new();
        for (r, row) in rows.iter().enumerate() {
            let pools = (row.position as usize + 1) / GROUP;
            if pools <= kept {
                expected.push(global_list(
                    row.position,
                    &(0..pools as u32).collect::<Vec<_>>(),
                ));
                continue;
            }
            let keys = (0..pools)
                .map(|pool| candidate_key(row.scores.score(&mut rng, pool, pools), pool as u32))
                .collect::<Vec<_>>();
            let candidates: usize = (0..splits)
                .map(|split| {
                    let (begin, end) = split_range(pools, split, splits);
                    (end - begin).min(kept)
                })
                .sum();
            let shuffle = candidates <= CACHE_KEYS as usize;
            for split in 0..splits {
                let (begin, end) = split_range(pools, split, splits);
                let mut best = keys[begin..end].to_vec();
                best.sort_unstable_by(|a, b| b.cmp(a));
                best.truncate(kept);
                let at = (r * splits + split) * MAX_KEPT;
                // Fisher-Yates over the kept keys.
                let mut shuffled = best.clone();
                if shuffle {
                    for i in (1..shuffled.len()).rev() {
                        shuffled.swap(i, (rng.next() % (i as u64 + 1)) as usize);
                    }
                }
                sparse[at..at + shuffled.len()].copy_from_slice(&shuffled);
                best.resize(MAX_KEPT, 0);
                workspace[at..at + MAX_KEPT].copy_from_slice(&best);
            }
            let mut best = keys;
            best.sort_unstable_by(|a, b| b.cmp(a));
            best.truncate(kept);
            let mut pools = best.iter().map(|&key| !(key as u32)).collect::<Vec<_>>();
            pools.sort_unstable();
            expected.push(global_list(row.position, &pools));
        }
        Case {
            label,
            rows,
            splits,
            top_k,
            workspace,
            sparse,
            expected,
        }
    }
}

/// `write_list` in select.cuh: the kept pools' positions, ascending, the
/// always-selected tail, then -1.
fn global_list(position: i32, pools: &[u32]) -> Vec<u8> {
    let base = (position + 1) / GROUP as i32 * GROUP as i32;
    let tail = (position + 1 - base) as usize;
    let count = pools.len() * GROUP;
    let list = (0..WIDTH)
        .map(|e| {
            if e < count {
                (pools[e / GROUP] as usize * GROUP + e % GROUP) as i32
            } else if e - count < tail {
                base + (e - count) as i32
            } else {
                -1
            }
        })
        .collect::<Vec<_>>();
    bytes_i32(&list)
}

fn u64_bytes(words: &[u64]) -> Vec<u8> {
    words.iter().flat_map(|word| word.to_ne_bytes()).collect()
}

/// A case's device operands and the reference, tree and radix merges.
struct Fixture<'a> {
    case: &'a Case,
    workspace: DeviceBuffer,
    sparse: DeviceBuffer,
    positions: DeviceBuffer,
    live: DeviceBuffer,
    output: DeviceBuffer,
    reference: Function,
    tree: Function,
    radix: Function,
    /// The radix merge's mutants by suffix.
    mutants: BTreeMap<&'static str, Function>,
    plan: LaunchPlan,
    tree_plan: LaunchPlan,
    radix_plan: LaunchPlan,
}

impl<'a> Fixture<'a> {
    fn new(session: &Session, modules: &Modules, case: &'a Case) -> LabResult<Fixture<'a>> {
        let entry = selection();
        let rows = case.rows.len() as u32;
        let tree_entry = merge_tree::Entry::of(&entry).map_err(|e| e.0.to_string())?;
        let radix_entry = merge_radix::Entry::of(&entry).map_err(|e| e.0.to_string())?;
        let positions = case.rows.iter().map(|row| row.position).collect::<Vec<_>>();
        let mut mutants = BTreeMap::new();
        for (suffix, _) in merge_radix::MUTANTS {
            let symbol = merge_radix::Entry {
                suffix: suffix.into(),
                ..radix_entry.clone()
            }
            .symbol();
            mutants.insert(
                suffix,
                modules.mutants[suffix].function(&symbol).map_err(err)?,
            );
        }
        Ok(Fixture {
            case,
            workspace: upload(session, &u64_bytes(&case.workspace))?,
            sparse: upload(session, &u64_bytes(&case.sparse))?,
            positions: upload(session, &bytes_i32(&positions))?,
            live: upload(session, &rows.to_ne_bytes())?,
            output: session.allocate(rows as usize * ROW_BYTES)?,
            reference: modules
                .reference
                .function(&entry.merge_symbol().expect("a selection's merge"))
                .map_err(err)?,
            tree: modules.tree.function(&tree_entry.symbol()).map_err(err)?,
            radix: modules.radix.function(&radix_entry.symbol()).map_err(err)?,
            mutants,
            plan: entry
                .merge_plan(rows)
                .map_err(|e| e.0.to_string())?
                .expect("a selection's merge plan"),
            tree_plan: tree_entry.plan(rows).map_err(|e| e.0.to_string())?,
            radix_plan: radix_entry.plan(rows).map_err(|e| e.0.to_string())?,
        })
    }

    fn set_live(&self, session: &Session, live: u32) -> LabResult<()> {
        poison::write(session, &self.live, 0, &live.to_ne_bytes())
    }

    fn launch(
        &self,
        session: &Session,
        function: &Function,
        plan: LaunchPlan,
        workspace: &DeviceBuffer,
    ) -> LabResult<()> {
        function
            .launch(
                &session.stream,
                launch::config(plan)?,
                &[
                    KernelArgument::Buffer(workspace),
                    KernelArgument::Buffer(&self.positions),
                    KernelArgument::Buffer(&self.live),
                    KernelArgument::Buffer(&self.output),
                    KernelArgument::U32(self.case.top_k),
                    KernelArgument::U32(self.case.splits as u32),
                ],
            )
            .map_err(err)
    }

    fn reference(&self, session: &Session) -> LabResult<()> {
        self.launch(session, &self.reference, self.plan, &self.workspace)
    }

    fn tree(&self, session: &Session) -> LabResult<()> {
        self.launch(session, &self.tree, self.tree_plan, &self.workspace)
    }

    fn radix(&self, session: &Session) -> LabResult<()> {
        self.launch(session, &self.radix, self.radix_plan, &self.sparse)
    }

    /// The radix merge over the workspace as `select_split` writes it.
    fn radix_dense(&self, session: &Session) -> LabResult<()> {
        self.launch(session, &self.radix, self.radix_plan, &self.workspace)
    }

    fn mutant(&self, session: &Session, suffix: &str) -> LabResult<()> {
        self.launch(
            session,
            &self.mutants[suffix],
            self.radix_plan,
            &self.sparse,
        )
    }

    /// Fills the output, runs `launch` and reads the first `live` rows back.
    fn run(
        &self,
        session: &Session,
        live: usize,
        launch: impl FnOnce() -> LabResult<()>,
    ) -> LabResult<Vec<u8>> {
        session.stream.synchronize().map_err(err)?;
        poison::fill_all(session, &self.output, &LIST_FILL)?;
        launch()?;
        download(session, &self.output, live * ROW_BYTES)
    }
}

/// The lab cubins: the selection's merges, the tree merges, the served
/// radix merges and each mutant's.
struct Modules {
    reference: Module,
    tree: Module,
    radix: Module,
    mutants: BTreeMap<&'static str, Module>,
}

impl Modules {
    fn new(session: &Session) -> LabResult<Modules> {
        let mut mutants = BTreeMap::new();
        for (suffix, _) in merge_radix::MUTANTS {
            mutants.insert(
                suffix,
                session.module(&cubin_name(shapes::PROFILE, suffix))?,
            );
        }
        Ok(Modules {
            reference: session.module(shapes::CUBIN)?,
            tree: session.module(&merge_tree::cubin_name(shapes::PROFILE, ""))?,
            radix: session.module(&cubin_name(shapes::PROFILE, ""))?,
            mutants,
        })
    }
}

fn first_difference(label: &str, actual: &[u8], expected: &[u8]) -> String {
    let at = actual
        .iter()
        .zip(expected)
        .position(|(a, b)| a != b)
        .unwrap_or(actual.len().min(expected.len()));
    format!(
        "{label}: {} of {} bytes, first difference at row {} byte {}",
        actual.len(),
        expected.len(),
        at / ROW_BYTES,
        at % ROW_BYTES
    )
}

fn same(label: &str, actual: &[u8], expected: &[u8]) -> LabResult<()> {
    if actual != expected {
        return Err(first_difference(label, actual, expected));
    }
    Ok(())
}

/// A row bucket of `rows` rows over contexts from one pool past the kept
/// count up to `long` pools, some rows selecting without scoring.
fn bucket_rows(rows: u32, top_k: u32, long: usize, seed: u64) -> Vec<Row> {
    let kept = top_k as usize / GROUP;
    let mut rng = Rng(seed | 1);
    let patterns = [Scores::Tied, Scores::Spread, Scores::Tied, Scores::Equal];
    (0..rows as usize)
        .map(|r| {
            let pools = match r % 6 {
                0 => kept + 1,
                1 => (kept / 2).max(1),
                2 => 780,
                3 => long,
                _ => kept + 1 + (rng.next() % (long.max(kept + 2) - kept - 1) as u64) as usize,
            };
            // Positions with every tail length.
            Row {
                position: (pools * GROUP - 1 + r % GROUP) as i32,
                scores: patterns[r % patterns.len()],
            }
        })
        .collect()
}

/// Every bucket at its served split count and the served top-k, then split
/// counts from 1 to 64 and runtime top-k values at a few buckets, contexts
/// around the shared cache's 4096 candidates, and the extremes.
fn cases() -> Vec<Case> {
    let entry = selection();
    let mut cases = Vec::new();
    for rows in BUCKETS {
        let splits = entry.splits(rows) as usize;
        let long = if rows <= 4 { 262_144 } else { 8_192 };
        cases.push(Case::new(
            format!("rows {rows} splits {splits} top-k 2048"),
            bucket_rows(rows, 2048, long, 0x51ec7 + u64::from(rows)),
            splits,
            2048,
            0x9e37 + u64::from(rows),
        ));
    }
    for (rows, splits, top_k) in [
        (4, 1, 2048),
        (4, 2, 2048),
        (3, 3, 2044),
        (5, 5, 1024),
        (6, 17, 256),
        (4, 31, 2048),
        (2, 64, 4),
        (8, 64, 64),
        (16, 63, 2048),
    ] {
        cases.push(Case::new(
            format!("rows {rows} splits {splits} top-k {top_k}"),
            bucket_rows(
                rows,
                top_k,
                20_000,
                0xc0ffee + u64::from(rows) * 131 + splits as u64,
            ),
            splits,
            top_k,
            0x7a11 + splits as u64,
        ));
    }
    // Candidate counts at and around `CACHE_KEYS`: 64 splits of 64 pools
    // each hold every pool, so the count is the pools'.
    for pools in [4_095_usize, 4_096, 4_097, 4_160] {
        let rows = (0..4)
            .map(|r| Row {
                position: (pools * GROUP - 1 + r) as i32,
                scores: [Scores::Spread, Scores::Tied][r % 2],
            })
            .collect();
        cases.push(Case::new(
            format!("rows 4 splits 64 pools {pools}"),
            rows,
            64,
            2048,
            0xcace + pools as u64,
        ));
    }
    // Both infinities, and rows with no visible pool (positions 0 to 2).
    let mut rows = bucket_rows(8, 2048, 9_000, 0xe7);
    for (r, row) in rows.iter_mut().enumerate() {
        row.scores = Scores::Extremes;
        if r % 4 == 1 {
            row.position = (r % 3) as i32;
        }
    }
    cases.push(Case::new(
        "rows 8 extremes and empty rows".into(),
        rows,
        entry.splits(8) as usize,
        2048,
        0x1f1f,
    ));
    cases
}

/// The three merges at full live count against the host's selection; the
/// radix merge over both workspaces.
fn bitwise_case(session: &Session, modules: &Modules, case: &Case) -> LabResult<Value> {
    let fixture = Fixture::new(session, modules, case)?;
    let rows = case.rows.len();
    let expected = case.expected.concat();
    let reference = fixture.run(session, rows, || fixture.reference(session))?;
    same(
        &format!("{}: select_merge against the host", case.label),
        &reference,
        &expected,
    )?;
    let tree = fixture.run(session, rows, || fixture.tree(session))?;
    same(
        &format!("{}: tree merge against select_merge", case.label),
        &tree,
        &reference,
    )?;
    let radix = fixture.run(session, rows, || fixture.radix(session))?;
    same(
        &format!("{}: radix merge against select_merge", case.label),
        &radix,
        &reference,
    )?;
    let dense = fixture.run(session, rows, || fixture.radix_dense(session))?;
    same(
        &format!(
            "{}: radix merge over select_split's workspace against select_merge",
            case.label
        ),
        &dense,
        &reference,
    )?;
    Ok(json!({
        "case": case.label,
        "rows": rows,
        "splits": case.splits,
        "top_k": case.top_k,
        "grid": fixture.radix_plan.grid,
        "block": fixture.radix_plan.block,
    }))
}

/// Capacities whose replays cover every live count 0..=capacity.
/// With 2048 rows (sampled), one capacity at each served split count: 64 (1,
/// 4, 6), 32 (16), 16 (32), 8 (64), 4 (128, C32 verify items of 4 rows), 2
/// (192) and 1 (2048).
const EVERY_LIVE_COUNT: [usize; 8] = [1, 4, 6, 16, 32, 64, 128, 192];

/// The receipt fields a live count's replay must hold: relaunch, graph and
/// PDL (with programmatic serialization) bitwise against the eager bits.
const REQUIRED_PHASES: [&str; 3] = ["deterministic_bitwise", "graph_bitwise", "pdl_bitwise"];

/// The radix merge on poisoned outputs, from one graph captured at the
/// case's capacity: at every live count 0..=capacity for the capacities of
/// `EVERY_LIVE_COUNT`, else at `poison::live_counts`' sample. Per live
/// count: eager (compared with `select_merge`'s output at that live count),
/// relaunch, the capacity graph and PDL behind the live-count producer. The
/// receipt holds the covered live counts and each one's phases; a missing
/// live count or phase fails.
fn replay_case(session: &Session, modules: &Modules, case: &Case) -> LabResult<Value> {
    let fixture = Fixture::new(session, modules, case)?;
    let map = WriteMap::of(DType::I32);
    let capacity = case.rows.len();
    let every = EVERY_LIVE_COUNT.contains(&capacity);
    let expected: Vec<usize> = if every {
        (0..=capacity).collect()
    } else {
        poison::live_counts(capacity as u32)
            .into_iter()
            .map(|live| live as usize)
            .collect()
    };
    let graph = session.capture(|_| fixture.radix(session))?;
    let producer = LiveProducer::new(session)?;
    let mut evidence = Vec::new();
    let mut covered = Vec::new();
    for &live in &expected {
        let label = format!("{}, live {live}", case.label);
        fixture.set_live(session, live as u32)?;
        let reference = fixture.run(session, live, || fixture.reference(session))?;
        let tree = fixture.run(session, live, || fixture.tree(session))?;
        same(
            &format!("{label}: tree merge against select_merge"),
            &tree,
            &reference,
        )?;
        let mut replay = Replay::new(session).output_with(
            "selection",
            &fixture.output,
            map.byte_prefix(live * ROW_BYTES)?,
            &LIST_FILL,
        );
        replay.eager(|| fixture.radix(session))?;
        let eager = download(session, &fixture.output, live * ROW_BYTES)?;
        same(
            &format!("{label}: eager radix merge against select_merge and the tree merge"),
            &eager,
            &reference,
        )?;
        replay.relaunch(|| fixture.radix(session))?;
        replay.graph(&graph)?;
        replay.pdl(
            || producer.launch(session, &fixture.live),
            || fixture.radix(session),
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
            "{}: covered live counts {covered:?}, expected {expected:?}",
            case.label
        ));
    }
    Ok(json!({
        "case": case.label,
        "capacity": capacity,
        "splits": case.splits,
        "coverage": if every { "every live count 0..=capacity" } else { "sampled (poison::live_counts)" },
        "covered_live_counts": covered,
        "required_phases": REQUIRED_PHASES,
        "graph": "one capture at the capacity, replayed at every covered live count",
        "live_counts": evidence,
    }))
}

/// The radix merge's mutants, each against the served merge's bits on rows
/// that expose it.
fn mutant_cases(session: &Session, modules: &Modules) -> LabResult<Value> {
    let entry = selection();
    let mut outcomes = Vec::new();
    // 4096 pools: every split holds pools (64 per split at 64 splits), and
    // the radix select takes the row; 16384 pools: the tree fold does.
    for (rows, pools) in [
        (1_u32, 4_096_usize),
        (4, 4_096),
        (32, 4_096),
        (128, 4_096),
        (1, 16_384),
        (4, 16_384),
    ] {
        let splits = entry.splits(rows) as usize;
        let row = |scores| Row {
            position: (pools * GROUP + 2) as i32,
            scores,
        };
        for (suffix, scores) in [
            ("_skip_last", Scores::BestLast),
            ("_ties_high", Scores::Equal),
        ] {
            let case = Case::new(
                format!("{suffix} rows {rows} splits {splits} pools {pools}"),
                vec![row(scores); rows as usize],
                splits,
                2048,
                0x5eed + u64::from(rows),
            );
            let fixture = Fixture::new(session, modules, &case)?;
            let served = fixture.run(session, rows as usize, || fixture.radix(session))?;
            same(
                &format!("{}: served", case.label),
                &served,
                &case.expected.concat(),
            )?;
            let mutant = fixture.run(session, rows as usize, || fixture.mutant(session, suffix))?;
            if mutant == served {
                return Err(format!(
                    "{}: the mutant reproduced the served bits",
                    case.label
                ));
            }
            outcomes
                .push(json!({"mutant": suffix, "rows": rows, "pools": pools, "outcome": "caught"}));
        }

        // The dependency wait behind the live-count producer, which holds
        // the live count at zero through its tail.
        let case = Case::new(
            format!("_late_wait rows {rows} splits {splits} pools {pools}"),
            vec![row(Scores::Tied); rows as usize],
            splits,
            2048,
            0x1a7e + u64::from(rows),
        );
        let fixture = Fixture::new(session, modules, &case)?;
        let eager = fixture.run(session, rows as usize, || fixture.radix(session))?;
        let producer = LiveProducer::new(session)?;
        let behind = |late: bool| -> LabResult<Vec<u8>> {
            fixture.run(session, rows as usize, || {
                producer.launch(session, &fixture.live)?;
                if late {
                    fixture.mutant(session, "_late_wait")
                } else {
                    fixture.radix(session)
                }
            })
        };
        if behind(false)? != eager {
            return Err(format!(
                "{}: the served merge behind the live-count producer changed its output",
                case.label
            ));
        }
        let outcome = if behind(true)? == eager {
            eprintln!(
                "{}: inconclusive: the mutant reproduced the eager bits",
                case.label
            );
            "inconclusive"
        } else {
            "caught"
        };
        outcomes.push(
            json!({"mutant": "_late_wait", "rows": rows, "pools": pools, "outcome": outcome}),
        );
    }
    Ok(json!({"entry": entry.name, "mutants": outcomes}))
}

#[test]
fn pooled_indexer_merge_radix_cpu_contract() {
    let entry = selection();
    let radix = merge_radix::Entry::of(&entry).unwrap();
    for rows in BUCKETS {
        let plan = radix.plan(rows).unwrap();
        assert_eq!(
            plan.grid,
            entry.merge_plan(rows).unwrap().unwrap().grid,
            "one CTA per row, as the selection's merge"
        );
    }
    assert_eq!(radix.params(), entry.merge_params());
    // The host mirrors order keys as the kernels do.
    let ranked = [
        candidate_key(1.0, 7),
        candidate_key(1.0, 9),
        candidate_key(0.0, 3),
        candidate_key(-0.0, 4),
        candidate_key(-1.0, 1),
        candidate_key(f32::NEG_INFINITY, 2),
        candidate_key(f32::NAN, 5),
    ];
    assert!(ranked.windows(2).all(|pair| pair[0] > pair[1]));
    // Every case's rows have their expected outputs; the sparse workspace
    // holds exactly the keys the dense one holds, every other word
    // all-ones; cases cover both of the merge's paths; and the replay test
    // has a case at each capacity it covers at every live count.
    let cases = cases();
    for case in &cases {
        assert_eq!(case.expected.len(), case.rows.len(), "{}", case.label);
        for (dense, sparse) in case
            .workspace
            .chunks(MAX_KEPT)
            .zip(case.sparse.chunks(MAX_KEPT))
        {
            let mut kept = dense
                .iter()
                .copied()
                .filter(|&key| key != 0 && key != u64::MAX)
                .collect::<Vec<_>>();
            let mut shuffled = sparse[..kept.len()].to_vec();
            kept.sort_unstable();
            shuffled.sort_unstable();
            assert_eq!(kept, shuffled, "{}", case.label);
            assert!(
                sparse[kept.len()..].iter().all(|&key| key == u64::MAX),
                "{}",
                case.label
            );
        }
    }
    let candidates = |case: &Case, row: &Row| -> usize {
        let pools = (row.position as usize + 1) / GROUP;
        let kept = case.top_k as usize / GROUP;
        (0..case.splits)
            .map(|split| {
                let (begin, end) = split_range(pools, split, case.splits);
                (end - begin).min(kept)
            })
            .sum()
    };
    let rows = cases
        .iter()
        .flat_map(|case| case.rows.iter().map(move |row| (case, row)))
        .filter(|(case, row)| (row.position as usize + 1) / GROUP > case.top_k as usize / GROUP)
        .map(|(case, row)| candidates(case, row))
        .collect::<Vec<_>>();
    for bound in [CACHE_KEYS as usize, CACHE_KEYS as usize + 1] {
        assert!(rows.contains(&bound), "no row of {bound} candidates");
    }
    assert!(rows.iter().any(|&count| count < 1024));
    assert!(rows.iter().any(|&count| count >= 16 * 1024));
    for capacity in EVERY_LIVE_COUNT {
        assert!(
            cases.iter().any(|case| case.rows.len() == capacity),
            "no replay case of {capacity} rows"
        );
    }
    // Every split count a prepared bucket can take has a bitwise case and a
    // replayed capacity at a bucket with that count, run at that count.
    let served = (1..=2048)
        .map(|rows| entry.splits(rows))
        .collect::<BTreeSet<_>>();
    assert_eq!(served, BTreeSet::from([1, 2, 4, 8, 16, 32, 64]));
    for splits in served {
        let at = |rows: usize| entry.splits(rows as u32) == splits;
        assert!(
            cases
                .iter()
                .any(|case| at(case.rows.len()) && case.splits == splits as usize),
            "no bitwise case at a bucket of {splits} splits"
        );
        assert!(
            cases.iter().any(|case| at(case.rows.len())
                && case.splits == splits as usize
                && (EVERY_LIVE_COUNT.contains(&case.rows.len()) || case.rows.len() == 2048)),
            "no replay case at a bucket of {splits} splits"
        );
    }
}

#[test]
fn pooled_indexer_merge_radix_matches_select_merge_bitwise() {
    kernel_lab::gpu_test(4 << 30, |session| {
        let modules = Modules::new(session)?;
        let mut evidence = Vec::new();
        for case in cases() {
            evidence.push(bitwise_case(session, &modules, &case)?);
        }
        let mut receipt = BTreeMap::new();
        receipt.insert("op".into(), json!("pooled_indexer_merge_radix_bitwise"));
        receipt.insert("cases".into(), json!(evidence));
        session.receipt("pooled_indexer_merge_radix_bitwise", receipt)?;
        Ok(())
    })
    .expect("radix merge against select_merge");
}

#[test]
fn pooled_indexer_merge_radix_replays_every_live_count() {
    kernel_lab::gpu_test(4 << 30, |session| {
        let modules = Modules::new(session)?;
        let mut evidence = Vec::new();
        for case in cases()
            .into_iter()
            .filter(|case| EVERY_LIVE_COUNT.contains(&case.rows.len()) || case.rows.len() == 2048)
        {
            evidence.push(replay_case(session, &modules, &case)?);
        }
        let mut receipt = BTreeMap::new();
        receipt.insert("op".into(), json!("pooled_indexer_merge_radix_replay"));
        receipt.insert("cases".into(), json!(evidence));
        session.receipt("pooled_indexer_merge_radix_replay", receipt)?;
        Ok(())
    })
    .expect("radix merge poisoned replays at every live count");
}

#[test]
fn pooled_indexer_merge_radix_mutants_are_caught() {
    kernel_lab::gpu_test(4 << 30, |session| {
        let modules = Modules::new(session)?;
        let evidence = mutant_cases(session, &modules)?;
        let mut receipt = BTreeMap::new();
        receipt.insert("op".into(), json!("pooled_indexer_merge_radix_mutants"));
        receipt.insert("entries".into(), json!([evidence]));
        session.receipt("pooled_indexer_merge_radix_mutants", receipt)?;
        Ok(())
    })
    .expect("radix merge mutants");
}

/// Graph replays of eight merges, the tree merge (arm 0) or the radix
/// merge (arm 1), each launch PDL after the previous as the bundle runs them.
#[test]
fn pooled_indexer_merge_radix_timing() {
    if std::env::var("CHOPINFER_KERNEL_BENCH").as_deref() != Ok("1") {
        kernel_lab::skip("pooled_indexer_merge_radix timing: set CHOPINFER_KERNEL_BENCH=1")
            .unwrap();
        return;
    }
    const MERGES: usize = 8;
    kernel_lab::gpu_test(4 << 30, |session| {
        let modules = Modules::new(session)?;
        let entry = selection();
        let mut rows_out = Vec::new();
        for (context, pools) in [
            ("3k", 780_usize),
            ("16k", 4_096),
            ("32k", 8_192),
            ("64k", 16_384),
            ("128k", 32_768),
        ] {
            for rows in [1_u32, 2, 4, 8, 16, 32, 128, 192] {
                let splits = entry.splits(rows) as usize;
                let case = Case::new(
                    format!("timing {context} rows {rows}"),
                    vec![
                        Row {
                            position: (pools * GROUP + 1) as i32,
                            scores: Scores::Spread,
                        };
                        rows as usize
                    ],
                    splits,
                    2048,
                    0x7173 + u64::from(rows),
                );
                let fixture = Fixture::new(session, &modules, &case)?;
                let graphs = [false, true].map(|radix| {
                    session.capture(|_| {
                        for _ in 0..MERGES {
                            if radix {
                                fixture.radix_dense(session)?;
                            } else {
                                fixture.tree(session)?;
                            }
                        }
                        Ok(())
                    })
                });
                let [tree, radix] = graphs;
                let graphs = [tree?, radix?];
                let paired = kernel_lab::time_ab(
                    session,
                    3,
                    15,
                    [kernel_lab::eviction::EvictionMode::Write; 2],
                    |arm, stream| graphs[arm].launch(stream).map_err(err),
                )?;
                let per = |ms: f32| f64::from(ms) * 1e3 / MERGES as f64;
                let row = json!({
                    "context": context,
                    "rows": rows,
                    "splits": splits,
                    "tree_merge_us": per(paired.arms[0].median_ms),
                    "radix_merge_us": per(paired.arms[1].median_ms),
                    "radix_over_tree": paired.paired_ratio,
                });
                eprintln!("pooled_indexer_merge_radix timing: {row}");
                rows_out.push(row);
            }
        }
        let mut receipt = BTreeMap::new();
        receipt.insert("op".into(), json!("pooled_indexer_merge_radix_timing"));
        receipt.insert("buckets".into(), json!(rows_out));
        session.receipt("pooled_indexer_merge_radix_timing", receipt)?;
        Ok(())
    })
    .expect("radix merge timing");
}
