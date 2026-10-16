//! The one-split selection (`kernels_sm120::pooled_indexer::rows`, README
//! "One-split selection" in crates/kernels-sm120/cuda/pooled_indexer)
//! against the selection it replaces, its split kernel and split merge, at
//! the lab's indexer shape (`pooled_indexer_shapes.rs`: 32 heads of 128,
//! pools of 4, top 2048 with the tail, 64-token pages), on the GPU. Run
//! with `CHOPINFER_KERNEL_GPU=<ordinal> cargo test -p kernel-lab --release
//! --test pooled_indexer_rows -- --test-threads=1`; without a GPU only the
//! CPU contract test runs.
//!
//! The pooled keys are random BF16 rows on pages scattered over one pool,
//! for sequences of 3,000, 9,000, 33,000 and 1,048,576 tokens (262,144
//! pools). One page in eight copies another page's keys, so its pools tie
//! with the copied pools' scores bit for bit; some pools are all zeros
//! (scores of exactly zero), and some hold infinities (infinite and NaN
//! scores) or NaNs (dots the ReLU zeroes). Head weights are random BF16,
//! all negative for some rows (every score at most zero: wide ties at +0).
//!
//! - Both forms (one and two rows per CTA) write the key lists, or the
//!   per-rank pairs, of the selection's split kernel and merge at the same
//!   rows byte for byte over filled outputs: mixed rows at 1 to 2048 rows
//!   (runs of one sequence and changes of sequence, positions across the
//!   2,048-token boundary, odd counts that leave a two-row CTA one row),
//!   whole 2048-row pieces
//!   ending at 2K, 8K and 32K, rows at 1M contexts on both sides of the one-row
//!   bitmap's bound (262,144 pools), the runtime top-k of
//!   1024 and 16, and the per-rank selections at every rank of 2 and 4 and
//!   at 0 of 1.
//! - Poisoned replays (eager, relaunch, graph, PDL behind the live-count
//!   producer) at every live count `poison::live_counts` lists, each live
//!   count's eager output equal to the replaced selection's.
//! - Mutants of each form, each caught only when an output differs from the
//!   served bits (launch and readback errors propagate): copies from the
//!   previous tile's pages (`_stale_page`), a final cut that keeps one key
//!   too many (`_keep_extra`), and the dependency wait after the query loads
//!   (`_late_wait`, behind a producer that writes the queries after
//!   releasing it; up to four attempts, and the test fails unless one
//!   observes a difference).
//! - One captured graph per capacity (1, 31, 52, 375, 376 and 2048 rows) of
//!   each form, global and per rank, launched on filled outputs at every
//!   live count the device count is set to: rows below it hold the
//!   selection's bytes, the rest keep the fill.
//!
//! With `CHOPINFER_KERNEL_BENCH=1`, `pooled_indexer_rows_timing` times the
//! selection and both forms at 1, 16, 2048 and 8192 rows (four 2048-row
//! pieces) ending at 2K, 8K and 32K contexts.

#[path = "../pooled_indexer_shapes.rs"]
mod shapes;

#[path = "common/support.rs"]
mod support;

use std::collections::BTreeMap;

use kernel_build::EntryParam;
use kernel_core::codec::f32_to_bf16;
use kernel_core::contract::{DType, LaunchPlan};
use kernel_lab::latent_shapes;
use kernel_lab::poison::{self, CopyProducer, Fill, POISON, Replay, WriteMap, replay_all};
use kernel_lab::{LabResult, Session, launch};
use kernels_sm120::pooled_indexer::rows::{self, cubin_name};
use kernels_sm120::pooled_indexer::{Entry, Kind};
use runtime_cuda::{DeviceBuffer, Function, KernelArgument, Module};
use serde_json::{Value, json};
use support::{Rng, bytes_i32, bytes_u16, download, upload};

const HEADS: usize = 32;
const DIM: usize = 128;
const GROUP: usize = 4;
const PAGE: usize = 64;
const PAGE_POOLS: usize = PAGE / GROUP;
const MAX_KEPT: usize = 512;
const WIDTH: usize = 2048 + GROUP - 1;
/// A page's pooled keys, then bytes no kernel reads.
const PAGE_BYTES: usize = PAGE_POOLS * DIM * 2;
const STRIDE: usize = PAGE_BYTES + 512;
/// The lab's sequences: 3K, 9K, 33K and 1M tokens and a page, whose last
/// rows see more pools than the one-row form's bitmap holds.
const LENGTHS: [usize; 4] = [3_000, 9_000, 33_000, 1_048_576 + PAGE];
/// Pools the one-row form writes through its bitmap: 1M tokens.
const BITMAP_POOLS: usize = 262_144;
/// The fill of key list outputs: no position, and not -1.
const LIST_FILL: [u8; 4] = [0xa5, 0x5a, 0xc3, 0x3c];
/// The fill of per-rank pairs: a NaN score no pair holds.
const PAIR_FILL: [u8; 4] = [0xff, 0xff, 0xff, 0x7f];

fn err(e: impl ToString) -> String {
    e.to_string()
}

/// The lab's selection entries: global, then per rank at 2 and 4 ranks.
fn selections() -> Vec<Entry> {
    shapes::lab_entries()
        .into_iter()
        .filter(|entry| matches!(entry.kind, Kind::Select { .. }))
        .collect()
}

fn per_rank(entry: &Entry) -> bool {
    matches!(entry.kind, Kind::Select { ranks: Some(_) })
}

/// A pool of pooled key pages for [`LENGTHS`], their tables and device
/// copies.
struct Cache {
    max_pages: usize,
    d_keys: DeviceBuffer,
    d_tables: DeviceBuffer,
}

impl Cache {
    fn new(session: &Session, rng: &mut Rng) -> LabResult<Cache> {
        let pages_of = |len: usize| len.div_ceil(PAGE);
        let max_pages = LENGTHS.iter().map(|&l| pages_of(l)).max().unwrap_or(1);
        let pages = LENGTHS.iter().map(|&l| pages_of(l)).sum::<usize>();
        // Page ids in a shuffled order, so neighbours in a table are not
        // neighbours in the pool.
        let mut ids = (0..pages as i32).collect::<Vec<_>>();
        for i in (1..ids.len()).rev() {
            ids.swap(i, (rng.next() % (i as u64 + 1)) as usize);
        }
        let mut tables = vec![-1; LENGTHS.len() * max_pages];
        let mut next = 0;
        for (s, &len) in LENGTHS.iter().enumerate() {
            for page in 0..pages_of(len) {
                tables[s * max_pages + page] = ids[next];
                next += 1;
            }
        }
        let mut keys = vec![0_u8; pages * STRIDE];
        for page in 0..pages {
            let row = &mut keys[page * STRIDE..][..STRIDE];
            for pool in 0..PAGE_POOLS {
                let kind = rng.next() % 64;
                for d in 0..DIM {
                    let value = match kind {
                        // All zeros: a score of exactly zero.
                        0 => 0.0,
                        // An infinity: an infinite or NaN score.
                        1 if d == 5 => f32::INFINITY,
                        2 if d == 9 => f32::NEG_INFINITY,
                        // A NaN: NaN dots, which the ReLU zeroes.
                        3 if d == 17 => f32::NAN,
                        _ => rng.unit() * 2.0,
                    };
                    let at = pool * DIM * 2 + d * 2;
                    row[at..at + 2].copy_from_slice(&f32_to_bf16(value).to_ne_bytes());
                }
            }
            for byte in &mut row[PAGE_BYTES..] {
                *byte = rng.next() as u8;
            }
        }
        // One page in eight copies an earlier page: ties across pools.
        for page in (8..pages).step_by(8) {
            let from = (rng.next() % page as u64) as usize;
            keys.copy_within(from * STRIDE..from * STRIDE + PAGE_BYTES, page * STRIDE);
        }
        Ok(Cache {
            d_keys: upload(session, &keys)?,
            d_tables: upload(session, &bytes_i32(&tables))?,
            max_pages,
        })
    }
}

/// One row of a launch.
#[derive(Clone, Copy, Debug)]
struct Row {
    sequence: usize,
    position: usize,
    /// Every head weight negative.
    negative: bool,
}

/// One launch's rows and operands.
struct Case {
    label: String,
    rows: Vec<Row>,
    top_k: u32,
    q: Vec<u16>,
    weights: Vec<u16>,
}

impl Case {
    fn new(label: String, rows: Vec<Row>, top_k: u32, seed: u64) -> Case {
        let mut rng = Rng(seed | 1);
        let mut q = Vec::with_capacity(rows.len() * HEADS * DIM);
        let mut weights = Vec::with_capacity(rows.len() * HEADS);
        for row in &rows {
            assert!(row.position < LENGTHS[row.sequence], "{label}");
            q.extend((0..HEADS * DIM).map(|_| f32_to_bf16(rng.unit())));
            weights.extend((0..HEADS).map(|_| {
                let w = rng.unit();
                f32_to_bf16(if row.negative { -w.abs() - 0.01 } else { w })
            }));
        }
        Case {
            label,
            rows,
            top_k,
            q,
            weights,
        }
    }

    /// `rows` rows ending at `context` in the 33K sequence.
    fn ending_at(label: String, rows: usize, context: usize, seed: u64) -> Case {
        let rows = (0..rows)
            .map(|r| Row {
                sequence: 2,
                position: context - rows + r,
                negative: r % 97 == 5,
            })
            .collect();
        Case::new(label, rows, 2048, seed)
    }

    /// `rows` rows in runs of one to seven consecutive positions of one
    /// sequence, the sequences taking turns: the first runs start across
    /// the 2,048-token boundary and at the ends of the sequences, at most
    /// six rows sit in the 1M sequence.
    fn mixed(label: String, rows: usize, top_k: u32, seed: u64) -> Case {
        const STARTS: [(usize, usize); 8] = [
            (0, 0),
            (1, 2044),
            (2, 2047),
            (0, 2990),
            (1, 8189),
            (2, 32_760),
            (3, 1_048_570),
            (2, 8191),
        ];
        let mut rng = Rng(seed ^ 0x5eed);
        let mut out = Vec::with_capacity(rows);
        let mut run = 0;
        let mut long = 0;
        while out.len() < rows {
            let (sequence, start) = if run < STARTS.len() {
                STARTS[run]
            } else {
                let sequence = if long < 6 { run % 4 } else { run % 3 };
                (sequence, (rng.next() % LENGTHS[sequence] as u64) as usize)
            };
            let length = 1 + (rng.next() % 7) as usize;
            for k in 0..length.min(rows - out.len()) {
                let position = (start + k).min(LENGTHS[sequence] - 1);
                long += usize::from(sequence == 3);
                out.push(Row {
                    sequence,
                    position,
                    negative: (run + k) % 11 == 3,
                });
            }
            run += 1;
        }
        Case::new(label, out, top_k, seed)
    }
}

/// A case's device operands and outputs.
struct Fixture<'a> {
    cache: &'a Cache,
    case: &'a Case,
    q: DeviceBuffer,
    weights: DeviceBuffer,
    row_batch: DeviceBuffer,
    positions: DeviceBuffer,
    live: DeviceBuffer,
    workspace: DeviceBuffer,
    output: DeviceBuffer,
    /// The context-parallel group rank and size.
    cp: (u32, u32),
}

impl<'a> Fixture<'a> {
    fn new(
        session: &Session,
        cache: &'a Cache,
        case: &'a Case,
        entry: &Entry,
        cp: (u32, u32),
    ) -> LabResult<Fixture<'a>> {
        let rows = case.rows.len();
        let batch = case
            .rows
            .iter()
            .map(|row| row.sequence as i32)
            .collect::<Vec<_>>();
        let positions = case
            .rows
            .iter()
            .map(|row| row.position as i32)
            .collect::<Vec<_>>();
        let workspace = entry
            .plan(rows as u32)
            .map_err(|e| e.0.to_string())?
            .workspace_bytes as usize;
        Ok(Fixture {
            cache,
            case,
            q: upload(session, &bytes_u16(&case.q))?,
            weights: upload(session, &bytes_u16(&case.weights))?,
            row_batch: upload(session, &bytes_i32(&batch))?,
            positions: upload(session, &bytes_i32(&positions))?,
            live: upload(session, &(rows as u32).to_ne_bytes())?,
            workspace: session.allocate(workspace.max(16))?,
            output: session.allocate(rows * row_bytes(entry))?,
            cp,
        })
    }

    fn set_live(&self, session: &Session, live: u32) -> LabResult<()> {
        poison::write(session, &self.live, 0, &live.to_ne_bytes())
    }

    /// Kernel arguments for `params` by name.
    fn args(&self, params: &[EntryParam], splits: u32) -> LabResult<Vec<KernelArgument<'_>>> {
        params
            .iter()
            .map(|param| match param {
                EntryParam::DevicePointer { name, .. } => {
                    Ok(KernelArgument::Buffer(match name.as_str() {
                        "q" => &self.q,
                        "head_weights" => &self.weights,
                        "pooled_keys" => &self.cache.d_keys,
                        "page_tables" => &self.cache.d_tables,
                        "row_batch" => &self.row_batch,
                        "positions" => &self.positions,
                        "live_rows" => &self.live,
                        "workspace" => &self.workspace,
                        "output" => &self.output,
                        other => return Err(format!("unexpected pointer {other}")),
                    }))
                }
                EntryParam::U32 { name } => Ok(KernelArgument::U32(match name.as_str() {
                    "max_pages" => self.cache.max_pages as u32,
                    "top_k" => self.case.top_k,
                    "splits" => splits,
                    "cp_rank" => self.cp.0,
                    "cp_size" => self.cp.1,
                    other => return Err(format!("unexpected scalar {other}")),
                })),
                EntryParam::U64 { name } if name == "page_stride_bytes" => {
                    Ok(KernelArgument::U64(STRIDE as u64))
                }
                other => Err(format!("unexpected parameter {other:?}")),
            })
            .collect()
    }

    fn launch(
        &self,
        session: &Session,
        function: &Function,
        params: &[EntryParam],
        mut plan: LaunchPlan,
        splits: u32,
        pdl: bool,
    ) -> LabResult<()> {
        plan.pdl = pdl;
        function
            .launch(
                &session.stream,
                launch::config(plan)?,
                &self.args(params, splits)?,
            )
            .map_err(err)
    }

    fn fill(&self, entry: &Entry) -> &'static [u8] {
        if per_rank(entry) {
            &PAIR_FILL
        } else {
            &LIST_FILL
        }
    }

    /// The output's write map at `live` rows.
    fn map(&self, entry: &Entry, live: usize) -> LabResult<WriteMap> {
        if per_rank(entry) {
            WriteMap::of_records::<8>().byte_prefix(live * row_bytes(entry))
        } else {
            WriteMap::of(DType::I32).byte_prefix(live * row_bytes(entry))
        }
    }

    /// Fills the output, runs `launch` and reads the output back.
    fn run(
        &self,
        session: &Session,
        entry: &Entry,
        launch: impl FnOnce() -> LabResult<()>,
    ) -> LabResult<Vec<u8>> {
        session.stream.synchronize().map_err(err)?;
        poison::fill_all(session, &self.output, self.fill(entry))?;
        launch()?;
        download(session, &self.output, self.output.byte_len())
    }
}

/// Output bytes per row: a key list or `MAX_KEPT` pairs.
fn row_bytes(entry: &Entry) -> usize {
    if per_rank(entry) {
        MAX_KEPT * 8
    } else {
        WIDTH * 4
    }
}

/// The lab cubins: the pooled indexer's, the one-split forms' and each
/// mutant's.
struct Modules {
    indexer: Module,
    rows: Module,
    mutants: BTreeMap<&'static str, Module>,
}

impl Modules {
    fn new(session: &Session) -> LabResult<Modules> {
        let mut mutants = BTreeMap::new();
        for (suffix, _) in rows::MUTANTS {
            mutants.insert(
                suffix,
                session.module(&cubin_name(shapes::PROFILE, suffix))?,
            );
        }
        Ok(Modules {
            indexer: session.module(shapes::CUBIN)?,
            rows: session.module(&cubin_name(shapes::PROFILE, ""))?,
            mutants,
        })
    }
}

fn function(module: &Module, symbol: &str, shared: u32) -> LabResult<Function> {
    let function = module.function(symbol).map_err(err)?;
    if shared > 48 * 1024 {
        function
            .set_max_dynamic_shared_memory(shared)
            .map_err(err)?;
    }
    Ok(function)
}

/// A selection's split kernel and merge, and one of its one-split forms,
/// at a row count.
struct Pair {
    entry: Entry,
    form: rows::Entry,
    split: Function,
    merge: Function,
    served: Function,
    plan: LaunchPlan,
    merge_plan: LaunchPlan,
    form_plan: LaunchPlan,
}

impl Pair {
    fn new(modules: &Modules, entry: &Entry, form: u32, rows: usize) -> LabResult<Pair> {
        let form = rows::Entry::of(entry, form).map_err(|e| e.0.to_string())?;
        let plan = entry.plan(rows as u32).map_err(|e| e.0.to_string())?;
        let merge_plan = entry
            .merge_plan(rows as u32)
            .map_err(|e| e.0.to_string())?
            .ok_or("a selection has a merge")?;
        let form_plan = form.plan(rows as u32).map_err(|e| e.0.to_string())?;
        Ok(Pair {
            split: function(&modules.indexer, &entry.symbol(), plan.dynamic_shared_bytes)?,
            merge: function(
                &modules.indexer,
                &entry.merge_symbol().ok_or("a selection has a merge")?,
                0,
            )?,
            served: function(
                &modules.rows,
                &form.symbol(),
                form_plan.dynamic_shared_bytes,
            )?,
            entry: entry.clone(),
            form,
            plan,
            merge_plan,
            form_plan,
        })
    }

    fn mutant(&self, modules: &Modules, suffix: &str) -> LabResult<Function> {
        let symbol = rows::Entry {
            suffix: suffix.into(),
            ..self.form.clone()
        }
        .symbol();
        function(
            &modules.mutants[suffix],
            &symbol,
            self.form_plan.dynamic_shared_bytes,
        )
    }

    /// The replaced selection: its split kernel, then its merge.
    fn reference(&self, session: &Session, fixture: &Fixture, pdl: bool) -> LabResult<()> {
        let splits = self.plan.grid[1];
        fixture.launch(
            session,
            &self.split,
            &self.entry.params(),
            self.plan,
            splits,
            pdl,
        )?;
        fixture.launch(
            session,
            &self.merge,
            &self.entry.merge_params(),
            self.merge_plan,
            splits,
            pdl,
        )
    }

    fn served(&self, session: &Session, fixture: &Fixture, pdl: bool) -> LabResult<()> {
        self.with(session, fixture, &self.served, pdl)
    }

    fn with(
        &self,
        session: &Session,
        fixture: &Fixture,
        function: &Function,
        pdl: bool,
    ) -> LabResult<()> {
        fixture.launch(
            session,
            function,
            &self.form.params(),
            self.form_plan,
            1,
            pdl,
        )
    }
}

fn same(label: &str, actual: &[u8], expected: &[u8]) -> LabResult<()> {
    if actual.len() != expected.len() {
        return Err(format!(
            "{label}: {} bytes against {}",
            actual.len(),
            expected.len()
        ));
    }
    if let Some(at) = actual.iter().zip(expected).position(|(a, b)| a != b) {
        let differing = actual.iter().zip(expected).filter(|(a, b)| a != b).count();
        return Err(format!(
            "{label}: {differing} of {} bytes differ, the first at byte {at}",
            actual.len()
        ));
    }
    Ok(())
}

/// The global bitwise cases: mixed rows at every row count, whole 2048-row
/// pieces ending at 2K, 8K and 32K, 1M contexts, and the runtime top-k of
/// 1024 and 16.
fn cases() -> Vec<Case> {
    let mut cases = Vec::new();
    for rows in [1, 2, 3, 30, 36, 377, 512, 2048] {
        cases.push(Case::mixed(
            format!("mixed rows {rows}"),
            rows,
            2048,
            0x51ce + rows as u64,
        ));
    }
    for context in [2_048, 8_192, 32_768] {
        cases.push(Case::ending_at(
            format!("rows 2048 ending at {context}"),
            2048,
            context,
            0xc0de + context as u64,
        ));
    }
    // Rows at 1M tokens: the last rows whose visible pools the one-row
    // bitmap holds, the first that sort their lists past it, and the
    // sequence's last row.
    let bound = BITMAP_POOLS * GROUP - 1;
    cases.push(Case::new(
        "1M contexts across the bitmap bound".into(),
        (bound - 5..bound + 9)
            .chain([LENGTHS[3] - 1])
            .enumerate()
            .map(|(r, position)| Row {
                sequence: 3,
                position,
                negative: r == 4,
            })
            .collect(),
        2048,
        0x1e6,
    ));
    for top_k in [1_024, 16] {
        for rows in [36, 2048] {
            cases.push(Case::mixed(
                format!("mixed rows {rows} top-k {top_k}"),
                rows,
                top_k,
                0x70b + u64::from(top_k) + rows as u64,
            ));
        }
    }
    cases
}

/// The per-rank cases: mixed rows and a 2048-row piece ending at 32K.
fn per_rank_cases() -> Vec<Case> {
    vec![
        Case::mixed("per-rank mixed rows 36".into(), 36, 2048, 0x9a1),
        Case::ending_at(
            "per-rank rows 2048 ending at 32768".into(),
            2048,
            32_768,
            0x9a2,
        ),
    ]
}

#[test]
fn pooled_indexer_rows_cpu_contract() {
    let entries = selections();
    assert_eq!(entries.len(), 3);
    for entry in &entries {
        for form in rows::FORMS {
            let one = rows::Entry::of(entry, form).unwrap();
            let bitmap = if form == 1 { BITMAP_POOLS } else { 0 };
            assert_eq!(one.bitmap_pools() as usize, bitmap);
            for count in [1, 7, 376, 2048] {
                let plan = one.plan(count).unwrap();
                assert_eq!(plan.grid, [count.div_ceil(form), 1, 1]);
                assert_eq!(plan.workspace_bytes, 0);
            }
        }
    }
    for case in cases().iter().chain(&per_rank_cases()) {
        assert_eq!(
            case.q.len(),
            case.rows.len() * HEADS * DIM,
            "{}",
            case.label
        );
        assert_eq!(
            case.weights.len(),
            case.rows.len() * HEADS,
            "{}",
            case.label
        );
    }
    // The mixed rows change sequence, and the 2,048-row cases hold runs.
    let mixed = Case::mixed("check".into(), 2048, 2048, 7);
    assert!(
        mixed
            .rows
            .windows(2)
            .any(|w| w[0].sequence != w[1].sequence)
    );
    assert!(
        mixed
            .rows
            .windows(2)
            .any(|w| w[0].sequence == w[1].sequence)
    );
    assert!(mixed.rows.iter().filter(|row| row.sequence == 3).count() <= 12);
    assert!(mixed.rows.iter().any(|row| row.position < 2047));
    assert!(mixed.rows.iter().any(|row| row.negative));
}

/// The selection and a form at full live count over the same operands.
fn bitwise_case(
    session: &Session,
    modules: &Modules,
    cache: &Cache,
    entry: &Entry,
    form: u32,
    case: &Case,
    cp: (u32, u32),
) -> LabResult<Value> {
    let fixture = Fixture::new(session, cache, case, entry, cp)?;
    let rows = case.rows.len();
    let pair = Pair::new(modules, entry, form, rows)?;
    let reference = fixture.run(session, entry, || pair.reference(session, &fixture, false))?;
    let served = fixture.run(session, entry, || pair.served(session, &fixture, false))?;
    same(
        &format!(
            "{} {} cp {cp:?}: one-split form against the selection",
            pair.form.symbol(),
            case.label
        ),
        &served,
        &reference,
    )?;
    Ok(json!({
        "entry": pair.form.symbol(),
        "case": case.label,
        "rows": rows,
        "top_k": case.top_k,
        "cp": [cp.0, cp.1],
        "reference_splits": pair.plan.grid[1],
        "grid": pair.form_plan.grid,
        "dynamic_shared_bytes": pair.form_plan.dynamic_shared_bytes,
    }))
}

#[test]
fn pooled_indexer_rows_match_the_selection_bitwise() {
    kernel_lab::gpu_test(8 << 30, |session| {
        let modules = Modules::new(session)?;
        let cache = Cache::new(session, &mut Rng(0x91d3))?;
        let entries = selections();
        let mut evidence = Vec::new();
        for case in cases() {
            for form in rows::FORMS {
                evidence.push(bitwise_case(
                    session,
                    &modules,
                    &cache,
                    &entries[0],
                    form,
                    &case,
                    (0, 1),
                )?);
            }
        }
        for case in per_rank_cases() {
            for entry in &entries[1..] {
                let Kind::Select { ranks: Some(ranks) } = entry.kind else {
                    unreachable!("a per-rank selection")
                };
                let groups = (0..ranks).map(|rank| (rank, ranks)).chain([(0, 1)]);
                for cp in groups {
                    for form in rows::FORMS {
                        evidence.push(bitwise_case(
                            session, &modules, &cache, entry, form, &case, cp,
                        )?);
                    }
                }
            }
        }
        let mut receipt = BTreeMap::new();
        receipt.insert("op".into(), json!("pooled_indexer_rows_bitwise"));
        receipt.insert("cases".into(), json!(evidence));
        session.receipt("pooled_indexer_rows_bitwise", receipt)?;
        Ok(())
    })
    .expect("one-split selection against the selection");
}

/// Each form on poisoned outputs at every live count, each live count's
/// eager output compared with the selection's.
#[test]
fn pooled_indexer_rows_replay_on_poisoned_outputs() {
    kernel_lab::gpu_test(8 << 30, |session| {
        let modules = Modules::new(session)?;
        let cache = Cache::new(session, &mut Rng(0x91d5))?;
        let entries = selections();
        let mut evidence = Vec::new();
        for rows in [1_usize, 31, 2048] {
            let case = Case::mixed(
                format!("replay rows {rows}"),
                rows,
                2048,
                0x7e9 + rows as u64,
            );
            for (entry, cp) in [(&entries[0], (0, 1)), (&entries[2], (1, 4))] {
                for form in rows::FORMS {
                    let fixture = Fixture::new(session, &cache, &case, entry, cp)?;
                    let pair = Pair::new(&modules, entry, form, rows)?;
                    let mut counts = Vec::new();
                    for live in poison::live_counts(rows as u32) {
                        let live = live as usize;
                        let label = format!("{} {}, live {live}", pair.form.symbol(), case.label);
                        fixture.set_live(session, live as u32)?;
                        let reference = fixture
                            .run(session, entry, || pair.reference(session, &fixture, false))?;
                        let replay = Replay::new(session).output_with(
                            "selection",
                            &fixture.output,
                            fixture.map(entry, live)?,
                            fixture.fill(entry),
                        );
                        let mut receipt = BTreeMap::new();
                        replay_all(
                            session,
                            replay,
                            &fixture.live,
                            || pair.served(session, &fixture, true),
                            &mut receipt,
                        )?;
                        let eager = download(session, &fixture.output, fixture.output.byte_len())?;
                        // Rows past the live count keep the fill on both
                        // sides; the replay leaves its own fill there.
                        let written = live * row_bytes(entry);
                        same(
                            &format!("{label}: replayed form against the selection"),
                            &eager[..written],
                            &reference[..written],
                        )?;
                        counts.push(json!({"live": live, "replay": receipt}));
                    }
                    evidence.push(json!({
                        "entry": pair.form.symbol(),
                        "case": case.label,
                        "cp": [cp.0, cp.1],
                        "live_counts": counts,
                    }));
                }
            }
        }
        let mut receipt = BTreeMap::new();
        receipt.insert("op".into(), json!("pooled_indexer_rows_replay"));
        receipt.insert("cases".into(), json!(evidence));
        session.receipt("pooled_indexer_rows_replay", receipt)?;
        Ok(())
    })
    .expect("one-split selection poisoned replays");
}

/// One captured graph of each form per capacity, global and per rank 1 of
/// 4, launched on filled outputs at every live count: rows below the live
/// count hold the selection's bytes at full live count, the rest keep the
/// fill.
#[test]
fn pooled_indexer_rows_every_live_count_from_one_graph() {
    kernel_lab::gpu_test(8 << 30, |session| {
        let modules = Modules::new(session)?;
        let cache = Cache::new(session, &mut Rng(0x91e1))?;
        let entries = selections();
        let mut evidence = Vec::new();
        for rows in [1_usize, 31, 52, 375, 376, 2048] {
            let case = Case::mixed(
                format!("live counts rows {rows}"),
                rows,
                2048,
                0x11fe + rows as u64,
            );
            for (entry, cp) in [(&entries[0], (0, 1)), (&entries[2], (1, 4))] {
                for form in rows::FORMS {
                    let fixture = Fixture::new(session, &cache, &case, entry, cp)?;
                    let pair = Pair::new(&modules, entry, form, rows)?;
                    let reference =
                        fixture.run(session, entry, || pair.reference(session, &fixture, false))?;
                    let graph = session.capture(|_| pair.served(session, &fixture, true))?;
                    for live in 0..=rows as u32 {
                        fixture.set_live(session, live)?;
                        let bytes = fixture.run(session, entry, || {
                            graph.launch(&session.stream).map_err(err)
                        })?;
                        let label = format!(
                            "{} {} cp {cp:?}, live {live}",
                            pair.form.symbol(),
                            case.label
                        );
                        let map = fixture.map(entry, live as usize)?;
                        poison::check(&label, &bytes, &map, Fill::Pattern(fixture.fill(entry)))?;
                        let written = live as usize * row_bytes(entry);
                        same(&label, &bytes[..written], &reference[..written])?;
                    }
                    evidence.push(json!({
                        "entry": pair.form.symbol(),
                        "rows": rows,
                        "cp": [cp.0, cp.1],
                        "live_counts": rows + 1,
                    }));
                }
            }
        }
        let mut receipt = BTreeMap::new();
        receipt.insert("op".into(), json!("pooled_indexer_rows_live_matrix"));
        receipt.insert("cases".into(), json!(evidence));
        session.receipt("pooled_indexer_rows_live_matrix", receipt)?;
        Ok(())
    })
    .expect("one-split selection from one graph at every live count");
}

#[test]
fn pooled_indexer_rows_mutants_are_caught() {
    kernel_lab::gpu_test(4 << 30, |session| {
        let modules = Modules::new(session)?;
        let cache = Cache::new(session, &mut Rng(0x91d9))?;
        let lab = session.module(&latent_shapes::cubin_name())?;
        let producer = CopyProducer::new(&lab)?;
        let entry = &selections()[0];
        let mut outcomes = Vec::new();
        for form in rows::FORMS {
            for suffix in ["_stale_page", "_keep_extra"] {
                let case = Case::ending_at(
                    format!("rows{form} {suffix}"),
                    30,
                    20_000,
                    0x3e7 + suffix.len() as u64,
                );
                let fixture = Fixture::new(session, &cache, &case, entry, (0, 1))?;
                let pair = Pair::new(&modules, entry, form, 30)?;
                let served =
                    fixture.run(session, entry, || pair.served(session, &fixture, false))?;
                let mutant = pair.mutant(&modules, suffix)?;
                let caught = fixture.run(session, entry, || {
                    pair.with(session, &fixture, &mutant, false)
                })?;
                if caught == served {
                    return Err(format!(
                        "{}: the mutant reproduced the served bits",
                        case.label
                    ));
                }
                outcomes.push(json!({"form": form, "mutant": suffix, "outcome": "caught"}));
            }

            // The queries arrive from a producer that releases its
            // dependent first; the operand starts poisoned.
            let case = Case::ending_at(format!("rows{form} _late_wait"), 30, 20_000, 0x1a7e);
            let fixture = Fixture::new(session, &cache, &case, entry, (0, 1))?;
            let pair = Pair::new(&modules, entry, form, 30)?;
            let source = upload(session, &bytes_u16(&case.q))?;
            let eager = fixture.run(session, entry, || pair.served(session, &fixture, false))?;
            let mutant = pair.mutant(&modules, "_late_wait")?;
            let behind = |late: bool| -> LabResult<Vec<u8>> {
                poison::fill_all(session, &fixture.q, &[POISON])?;
                fixture.run(session, entry, || {
                    producer.copy(&session.stream, &source, &fixture.q)?;
                    if late {
                        pair.with(session, &fixture, &mutant, true)
                    } else {
                        pair.served(session, &fixture, true)
                    }
                })
            };
            if behind(false)? != eager {
                return Err(format!(
                    "{}: the served form behind the query producer changed its output",
                    case.label
                ));
            }
            // A caught attempt is an observed difference; equal bytes on
            // every attempt fail the test.
            let mut attempts = 0;
            let mut caught = false;
            while attempts < 4 && !caught {
                attempts += 1;
                caught = behind(true)? != eager;
            }
            if !caught {
                return Err(format!(
                    "{}: the mutant reproduced the eager bits in {attempts} attempts",
                    case.label
                ));
            }
            outcomes.push(json!({
                "form": form,
                "mutant": "_late_wait",
                "outcome": "caught",
                "attempts": attempts,
            }));
        }
        let mut receipt = BTreeMap::new();
        receipt.insert("op".into(), json!("pooled_indexer_rows_mutants"));
        receipt.insert("mutants".into(), json!(outcomes));
        session.receipt("pooled_indexer_rows_mutants", receipt)?;
        Ok(())
    })
    .expect("one-split selection mutants");
}

/// With `CHOPINFER_KERNEL_PROFILE=1`: one eager launch each of the global
/// selection's split kernel and merge and of both forms on a 2048-row piece
/// ending at 32K, for a profiler (`ncu --kernel-name regex:pooled_indexer`);
/// then both forms' phase-clock builds (`rows::CLOCKS_DEFINE`) on pieces
/// ending at 8K and 32K and on single rows, warm, each row's cycles per
/// phase summarized per tile.
#[test]
fn pooled_indexer_rows_profile() {
    if std::env::var("CHOPINFER_KERNEL_PROFILE").as_deref() != Ok("1") {
        kernel_lab::skip("pooled_indexer_rows profile: set CHOPINFER_KERNEL_PROFILE=1").unwrap();
        return;
    }
    kernel_lab::gpu_test(4 << 30, |session| {
        let modules = Modules::new(session)?;
        let clocks = session.module(&cubin_name(shapes::PROFILE, rows::CLOCKS_SUFFIX))?;
        let cache = Cache::new(session, &mut Rng(0x7173))?;
        let entry = &selections()[0];
        let case = Case::ending_at("profile 32k rows 2048".into(), 2048, 32_768, 0x9f0);
        let fixture = Fixture::new(session, &cache, &case, entry, (0, 1))?;
        for form in rows::FORMS {
            let pair = Pair::new(&modules, entry, form, 2048)?;
            if form == rows::FORMS[0] {
                pair.reference(session, &fixture, false)?;
            }
            pair.served(session, &fixture, false)?;
        }
        session.stream.synchronize().map_err(err)?;
        const PHASES: [&str; 8] = [
            "copy_wait",
            "barrier",
            "fill",
            "issue",
            "mma",
            "chain",
            "prologue",
            "final",
        ];
        let mut out = Vec::new();
        for (label, rows, context) in [
            ("32k 2048", 2048_usize, 32_768_usize),
            ("8k 2048", 2048, 8_192),
            ("32k 1", 1, 32_768),
        ] {
            let case = Case::ending_at(format!("clocks {label}"), rows, context, 0x9f1);
            let fixture = Fixture::new(session, &cache, &case, entry, (0, 1))?;
            for form in rows::FORMS {
                let pair = Pair::new(&modules, entry, form, rows)?;
                let symbol = rows::Entry {
                    suffix: rows::CLOCKS_SUFFIX.into(),
                    ..pair.form.clone()
                }
                .symbol();
                let function = function(&clocks, &symbol, pair.form_plan.dynamic_shared_bytes)?;
                // Warm: the second launch's clocks.
                pair.with(session, &fixture, &function, false)?;
                let bytes = fixture.run(session, entry, || {
                    pair.with(session, &fixture, &function, false)
                })?;
                let mut sums = [0_f64; 10];
                let mut scored = 0;
                // Thread 0 marks the first row of its CTA.
                for row in (0..rows).step_by(form as usize) {
                    let at = row * WIDTH * 4;
                    let word = |i: usize| {
                        u32::from_ne_bytes(bytes[at + 4 * i..at + 4 * i + 4].try_into().unwrap())
                    };
                    let tiles = u64::from(word(18)) | u64::from(word(19)) << 32;
                    if tiles == 0 {
                        continue;
                    }
                    scored += 1;
                    for (phase, sum) in sums.iter_mut().enumerate() {
                        *sum += (u64::from(word(2 * phase)) | u64::from(word(2 * phase + 1)) << 32)
                            as f64;
                    }
                }
                let tiles = sums[9].max(1.0);
                let mut row = json!({
                    "case": label,
                    "form": form,
                    "rows_scored": scored,
                    "tiles_per_row": sums[9] / f64::from(scored.max(1)),
                    "reductions_per_row": sums[8] / f64::from(scored.max(1)),
                });
                for (phase, name) in PHASES.iter().enumerate() {
                    let per = if phase < 6 {
                        tiles
                    } else {
                        f64::from(scored.max(1))
                    };
                    row[format!("{name}_cycles")] = json!(sums[phase] / per);
                }
                eprintln!("pooled_indexer_rows clocks: {row}");
                out.push(row);
            }
        }
        let mut receipt = BTreeMap::new();
        receipt.insert("op".into(), json!("pooled_indexer_rows_clocks"));
        receipt.insert("cases".into(), json!(out));
        session.receipt("pooled_indexer_rows_clocks", receipt)?;
        Ok(())
    })
    .expect("one-split selection profile");
}

/// Graph replays of the global selection (split kernel and merge, arm 0)
/// against each form (arm 1), at row counts ending at 2K, 8K and 32K
/// contexts: eight launches below 2048 rows, one at 2048, and four
/// consecutive 2048-row pieces for 8192 rows.
#[test]
fn pooled_indexer_rows_timing() {
    if std::env::var("CHOPINFER_KERNEL_BENCH").as_deref() != Ok("1") {
        kernel_lab::skip("pooled_indexer_rows timing: set CHOPINFER_KERNEL_BENCH=1").unwrap();
        return;
    }
    kernel_lab::gpu_test(8 << 30, |session| {
        let modules = Modules::new(session)?;
        let cache = Cache::new(session, &mut Rng(0x7173))?;
        let entry = &selections()[0];
        let mut rows_out = Vec::new();
        for (context_name, context) in [("2k", 2_048_usize), ("8k", 8_192), ("32k", 32_768)] {
            for rows in [1_usize, 16, 2048, 8192] {
                if rows > context {
                    continue;
                }
                let piece = rows.min(2048);
                let pieces = rows / piece;
                let cases = (0..pieces)
                    .map(|p| {
                        Case::ending_at(
                            format!("timing {context_name} rows {rows} piece {p}"),
                            piece,
                            context - (pieces - 1 - p) * piece,
                            0x7173 + (rows + p) as u64,
                        )
                    })
                    .collect::<Vec<_>>();
                let fixtures = cases
                    .iter()
                    .map(|case| Fixture::new(session, &cache, case, entry, (0, 1)))
                    .collect::<LabResult<Vec<_>>>()?;
                let launches = if rows < 2048 { 8 } else { 1 };
                let mut row = json!({"context": context_name, "rows": rows});
                for form in rows::FORMS {
                    let pair = Pair::new(&modules, entry, form, piece)?;
                    let graphs = [false, true].map(|one_split| {
                        session.capture(|_| {
                            for _ in 0..launches {
                                for fixture in &fixtures {
                                    if one_split {
                                        pair.served(session, fixture, true)?;
                                    } else {
                                        pair.reference(session, fixture, true)?;
                                    }
                                }
                            }
                            Ok(())
                        })
                    });
                    let [reference, one_split] = graphs;
                    let graphs = [reference?, one_split?];
                    let paired = kernel_lab::time_ab(
                        session,
                        3,
                        15,
                        [kernel_lab::eviction::EvictionMode::Write; 2],
                        |arm, stream| graphs[arm].launch(stream).map_err(err),
                    )?;
                    let per = |ms: f32| f64::from(ms) * 1e3 / launches as f64;
                    row[format!("select_us_vs_rows{form}")] = json!(per(paired.arms[0].median_ms));
                    row[format!("rows{form}_us")] = json!(per(paired.arms[1].median_ms));
                    row[format!("rows{form}_over_select")] = json!(paired.paired_ratio);
                }
                eprintln!("pooled_indexer_rows timing: {row}");
                rows_out.push(row);
            }
        }
        let mut receipt = BTreeMap::new();
        receipt.insert("op".into(), json!("pooled_indexer_rows_timing"));
        receipt.insert("buckets".into(), json!(rows_out));
        session.receipt("pooled_indexer_rows_timing", receipt)?;
        Ok(())
    })
    .expect("one-split selection timing");
}
