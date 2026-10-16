//! The head-sum selection (`head_sum_shapes.rs`; README "Head-sum
//! selection" in crates/kernels-sm120/cuda/pooled_indexer) against QSA's
//! served block selection, which it must reproduce bit for bit. Each row is
//! its own sequence over one set of block-key pages; QSA's
//! `block_scores_{decode,verify,prefill}` and `block_topk` run on the rows,
//! and the head-sum entries read the same queries and the same block-key
//! pages, QSA's layout bound directly:
//!
//! - DCP 1: the global key list's blocks must equal every QSA phase's
//!   selected block list, and the per-rank pairs at rank 0 of 1 must carry
//!   QSA's F32 scores of their blocks bit for bit.
//! - DCP 2: each rank's per-rank selection over its own pages (placed by
//!   `ContextParallelPagesV1`), gathered and merged by `topk_merge2`, must
//!   equal the DCP 1 key list.
//! - The kept scores hold the 128-K gate against the f64 oracle. On exact
//!   operands they must equal kernel-core's F32 emulation; elsewhere the
//!   emulation's differences are counted.
//!
//! Synthetic cases run every head-sum launch under the poison replays
//! (eager, relaunch, graph, PDL). `CHOPINFER_QSA_OPERANDS` names a
//! `qwen38-oracle qsa-capture --export` directory, or a directory holding
//! such directories (`exports-v2`): QSA's own `index_key_write` and
//! `index_query` run on each layer's projection rows and feed both engines.
//!
//! Run in release with `CHOPINFER_KERNEL_GPU=<ordinal> cargo test -p
//! kernel-lab --release --test qsa_head_sum_selection`.

#[path = "../head_sum_shapes.rs"]
mod shapes;

#[path = "common/support.rs"]
mod support;

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use kernel_core::codec::{bf16_to_f32, f32_to_bf16};
use kernel_core::contract::{DType, StaticQuery};
use kernel_core::oracle::OracleOutput;
use kernel_core::pooled_indexer::PooledIndexerQuery;
use kernel_core::pooled_indexer_head_sum_oracle::{head_sum_scores, head_sum_scores_f32};
use kernel_lab::poison::{Replay, WriteMap, replay_all};
use kernel_lab::{LabResult, Session, gpu_test, launch};
use kernels_sm120::attention::sparse;
use kernels_sm120::pooled_indexer::head_sum::{Entry, Kind};
use runtime_cuda::{DeviceBuffer, Function, KernelArgument, Module};
use runtime_kv::ContextParallelPagesV1;
use serde_json::{Value, json};
use support::{Rng, bytes_f32, bytes_i32, bytes_u16, download, upload};

const HEADS: usize = 4;
const DIM: usize = 128;
const BLOCK: usize = 4;
const PAGE: usize = 64;
const TOP: usize = 512;
const TOP_K: u32 = 2048;
const MAX_KEPT: usize = 512;
const WIDTH: usize = 2051;
/// Projection row width: the query heads, then the raw key.
const PROJECTION: usize = (HEADS + 1) * DIM;
/// Block-key page stride: one page's 4 KiB of block keys, then a 1 KiB gap.
const PAGE_STRIDE: usize = 5120;
/// Rows of one history key write launch.
const HISTORY_ROWS: usize = 8191;
/// Rows of one selection step: QSA's decode and verify capacity.
const STEP_ROWS: usize = 256;
/// The fill of key list outputs: no position, and not -1.
const LIST_FILL: [u8; 4] = [0xa5, 0x5a, 0xc3, 0x3c];
/// The fill of per-rank pairs: a NaN score no pair holds.
const PAIR_FILL: [u8; 4] = [0xff, 0xff, 0xff, 0x7f];
const PHASES: [&str; 3] = ["decode", "verify", "prefill"];

fn err(e: impl ToString) -> String {
    e.to_string()
}

/// Draws this file adds to the shared generator.
trait Draws {
    /// `k * 2^-shift` for an integer k in -range..=range, as BF16.
    fn int(&mut self, range: u64, shift: i32) -> u16;
}

impl Draws for Rng {
    fn int(&mut self, range: u64, shift: i32) -> u16 {
        let k = (self.next() % (2 * range + 1)) as i64 - range as i64;
        f32_to_bf16(k as f32 * 2_f32.powi(-shift))
    }
}

fn entry(suffix: &str) -> Entry {
    shapes::lab_entries()
        .into_iter()
        .find(|entry| entry.name.ends_with(suffix))
        .unwrap_or_else(|| panic!("no head-sum entry {suffix}"))
}

fn qsa_entry(name: &str) -> sparse::Entry {
    sparse::catalog(kernel_lab::catalogs())
        .into_iter()
        .find(|entry| entry.name == format!("qwen38_{name}"))
        .unwrap_or_else(|| panic!("no QSA entry {name}"))
}

fn function(module: &Module, symbol: &str, dynamic_shared_bytes: u32) -> LabResult<Function> {
    let function = module.function(symbol).map_err(err)?;
    if dynamic_shared_bytes > 48 * 1024 {
        function
            .set_max_dynamic_shared_memory(dynamic_shared_bytes)
            .map_err(err)?;
    }
    Ok(function)
}

fn words<T>(data: &[u8], from: fn([u8; 4]) -> T) -> Vec<T> {
    data.chunks_exact(4)
        .map(|c| from([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn halves(data: &[u8]) -> Vec<u16> {
    data.chunks_exact(2)
        .map(|c| u16::from_ne_bytes([c[0], c[1]]))
        .collect()
}

fn decode(bits: &[u16]) -> Vec<f64> {
    bits.iter().map(|&b| f64::from(bf16_to_f32(b))).collect()
}

/// Block keys `[blocks][128]` in QSA's paged layout: page j at `table[j]`
/// with `PAGE_STRIDE` bytes per page, each row its own sequence.
struct Keys {
    buffer: DeviceBuffer,
    /// One table row: the page per stripe.
    table: Vec<i32>,
    rank: u32,
    size: u32,
}

impl Keys {
    /// The pages QSA's key write fills, in page order (`upload_block_keys`
    /// of the QSA tests).
    fn single(session: &Session, keys: &[u16]) -> LabResult<Keys> {
        let blocks = keys.len() / DIM;
        let pages = blocks.div_ceil(PAGE / BLOCK).max(1);
        let mut image = vec![0xff_u8; pages * PAGE_STRIDE];
        for block in 0..blocks {
            let at = block / (PAGE / BLOCK) * PAGE_STRIDE + block % (PAGE / BLOCK) * DIM * 2;
            image[at..at + DIM * 2].copy_from_slice(&bytes_u16(&keys[block * DIM..][..DIM]));
        }
        Ok(Keys {
            buffer: upload(session, &image)?,
            table: (0..pages as i32).collect(),
            rank: 0,
            size: 1,
        })
    }

    /// Group rank `rank` of `size`'s own pages as the host places them
    /// (`ContextParallelPagesV1::rank_pages`), local page ids in an order
    /// unlike the stripe index.
    fn rank(session: &Session, keys: &[u16], size: usize, rank: usize) -> LabResult<Keys> {
        let tokens = keys.len() / DIM * BLOCK;
        let pages = ContextParallelPagesV1::new(size, PAGE, 8).map_err(err)?;
        let stripes = tokens.div_ceil(pages.stripe_tokens()).max(1);
        let table = (0..stripes).map(|s| stripes - 1 - s).collect::<Vec<_>>();
        let mut image = vec![0xff_u8; stripes * PAGE_STRIDE];
        for page in pages
            .rank_pages(rank, table.iter().copied(), tokens)
            .map_err(err)?
        {
            for offset in (0..page.tokens).step_by(BLOCK) {
                let block = (page.first_position + offset) / BLOCK;
                let at = page.local_page * PAGE_STRIDE + offset / BLOCK * DIM * 2;
                image[at..at + DIM * 2].copy_from_slice(&bytes_u16(&keys[block * DIM..][..DIM]));
            }
        }
        Ok(Keys {
            buffer: upload(session, &image)?,
            table: table.into_iter().map(|page| page as i32).collect(),
            rank: rank as u32,
            size: size as u32,
        })
    }

    fn tables(&self, rows: usize) -> Vec<i32> {
        (0..rows).flat_map(|_| self.table.iter().copied()).collect()
    }
}

/// One step's rows: queries `[rows][4][128]` BF16, each row its own
/// sequence at `positions[row]`.
struct Rows {
    q: DeviceBuffer,
    row_batch: DeviceBuffer,
    positions: DeviceBuffer,
    live: DeviceBuffer,
    host_positions: Vec<i32>,
}

impl Rows {
    fn new(session: &Session, q: &[u16], positions: &[i32]) -> LabResult<Rows> {
        let rows = positions.len();
        Ok(Rows {
            q: upload(session, &bytes_u16(q))?,
            row_batch: upload(session, &bytes_i32(&(0..rows as i32).collect::<Vec<_>>()))?,
            positions: upload(session, &bytes_i32(positions))?,
            live: upload(session, &(rows as u32).to_ne_bytes())?,
            host_positions: positions.to_vec(),
        })
    }

    fn count(&self) -> usize {
        self.host_positions.len()
    }

    /// The tokens the longest row sees.
    fn context(&self) -> u32 {
        self.host_positions
            .iter()
            .max()
            .map_or(1, |&p| p as u32 + 1)
    }
}

/// Launches QSA entry `entry` for `rows` rows that see at most `context`
/// tokens.
fn launch_qsa(
    session: &Session,
    module: &Module,
    entry: &sparse::Entry,
    rows: u32,
    context: u32,
    arguments: &[KernelArgument<'_>],
) -> LabResult<()> {
    let function = function(module, &entry.symbol(), entry.dynamic_shared_bytes())?;
    let plan = entry.plan(rows, context).map_err(|e| e.0.to_string())?;
    function
        .launch(&session.stream, launch::config(plan)?, arguments)
        .map_err(err)?;
    kernel_lab::launched(&entry.symbol());
    Ok(())
}

/// One QSA phase's scores `[rows][stride]` and selected blocks per row.
struct QsaSelection {
    stride: usize,
    scores: Vec<f32>,
    blocks: Vec<Vec<i32>>,
}

/// QSA's `block_scores_{phase}` and `block_topk` on `rows` over `keys`.
fn qsa_select(
    session: &Session,
    module: &Module,
    phase: &str,
    keys: &Keys,
    rows: &Rows,
) -> LabResult<QsaSelection> {
    let geometry = model_qwen38_contract::sm120::attention::SPARSE;
    let n = rows.count();
    let context = rows.context();
    let stride = geometry.score_stride(context) as usize;
    let table = upload(session, &bytes_i32(&keys.tables(n)))?;
    let scores = upload(session, &vec![0xff; n * stride * 4])?;
    let selected = upload(session, &vec![0xff; n * TOP * 4])?;
    launch_qsa(
        session,
        module,
        &qsa_entry(&format!("block_scores_{phase}")),
        n as u32,
        context,
        &[
            KernelArgument::Buffer(&rows.q),
            KernelArgument::Buffer(&keys.buffer),
            KernelArgument::Buffer(&table),
            KernelArgument::Buffer(&rows.row_batch),
            KernelArgument::Buffer(&rows.positions),
            KernelArgument::Buffer(&rows.live),
            KernelArgument::Buffer(&scores),
            KernelArgument::U32(keys.table.len() as u32),
            KernelArgument::U64(PAGE_STRIDE as u64),
            KernelArgument::U32(stride as u32),
        ],
    )?;
    launch_qsa(
        session,
        module,
        &qsa_entry("block_topk"),
        n as u32,
        context,
        &[
            KernelArgument::Buffer(&scores),
            KernelArgument::Buffer(&rows.positions),
            KernelArgument::Buffer(&rows.live),
            KernelArgument::Buffer(&selected),
            KernelArgument::U32(stride as u32),
        ],
    )?;
    let all = words(
        &download(session, &selected, n * TOP * 4)?,
        i32::from_ne_bytes,
    );
    let blocks = rows
        .host_positions
        .iter()
        .enumerate()
        .map(|(r, &p)| all[r * TOP..][..TOP.min((p as usize + 1) / BLOCK)].to_vec())
        .collect();
    Ok(QsaSelection {
        stride,
        scores: words(
            &download(session, &scores, n * stride * 4)?,
            f32::from_ne_bytes,
        ),
        blocks,
    })
}

/// Launches a head-sum selection entry's split and merge kernels.
#[allow(clippy::too_many_arguments)]
fn launch_select(
    session: &Session,
    module: &Module,
    entry: &Entry,
    keys: &Keys,
    rows: &Rows,
    table: &DeviceBuffer,
    workspace: &DeviceBuffer,
    output: &DeviceBuffer,
) -> LabResult<()> {
    entry
        .check_context_parallel(keys.rank, keys.size)
        .map_err(|e| e.0.to_string())?;
    let n = rows.count() as u32;
    let plan = entry.plan(n).map_err(|e| e.0.to_string())?;
    let merge_plan = entry
        .merge_plan(n)
        .map_err(|e| e.0.to_string())?
        .expect("a selection has a merge");
    let split = function(module, &entry.symbol(), plan.dynamic_shared_bytes)?;
    let merge = function(module, &entry.merge_symbol().unwrap(), 0)?;
    let splits = plan.grid[1];
    split
        .launch(
            &session.stream,
            launch::config(plan)?,
            &[
                KernelArgument::Buffer(&rows.q),
                KernelArgument::Buffer(&keys.buffer),
                KernelArgument::Buffer(table),
                KernelArgument::Buffer(&rows.row_batch),
                KernelArgument::Buffer(&rows.positions),
                KernelArgument::Buffer(&rows.live),
                KernelArgument::Buffer(workspace),
                KernelArgument::U32(keys.table.len() as u32),
                KernelArgument::U64(PAGE_STRIDE as u64),
                KernelArgument::U32(TOP_K),
                KernelArgument::U32(splits),
                KernelArgument::U32(keys.rank),
                KernelArgument::U32(keys.size),
            ],
        )
        .map_err(err)?;
    merge
        .launch(
            &session.stream,
            launch::config(merge_plan)?,
            &[
                KernelArgument::Buffer(workspace),
                KernelArgument::Buffer(&rows.positions),
                KernelArgument::Buffer(&rows.live),
                KernelArgument::Buffer(output),
                KernelArgument::U32(TOP_K),
                KernelArgument::U32(splits),
            ],
        )
        .map_err(err)
}

/// Runs a head-sum selection and returns the key lists (`Global`) or pair
/// bytes (`PerRank`) of the rows.
fn run_select(
    session: &Session,
    module: &Module,
    entry: &Entry,
    keys: &Keys,
    rows: &Rows,
    poison: bool,
    case: &str,
) -> LabResult<Vec<u8>> {
    let n = rows.count();
    let per_rank = matches!(entry.kind, Kind::Select { ranks: Some(_) });
    let row_bytes = if per_rank { MAX_KEPT * 8 } else { WIDTH * 4 };
    let output = session.allocate(n * row_bytes)?;
    let table = upload(session, &bytes_i32(&keys.tables(n)))?;
    let plan = entry.plan(n as u32).map_err(|e| e.0.to_string())?;
    let workspace = session.allocate((plan.workspace_bytes as usize).max(16))?;
    let replay = Replay::new(session)
        .output_with(
            "selection",
            &output,
            if per_rank {
                WriteMap::of_records::<8>()
            } else {
                WriteMap::of(DType::I32)
            }
            .byte_prefix(n * row_bytes)?,
            if per_rank { &PAIR_FILL } else { &LIST_FILL },
        )
        .scratch_within(
            "workspace",
            &workspace,
            std::iter::once(0..plan.workspace_bytes as usize).collect(),
        );
    let mut receipt = BTreeMap::from([
        ("entry".into(), json!(entry.name)),
        ("case".into(), json!(case)),
        ("rank".into(), json!(keys.rank)),
        ("size".into(), json!(keys.size)),
        ("rows".into(), json!(n)),
        ("grid".into(), json!(plan.grid)),
    ]);
    let run = || {
        launch_select(
            session, module, entry, keys, rows, &table, &workspace, &output,
        )
    };
    if poison {
        replay_all(session, replay, &rows.live, run, &mut receipt)?;
        session.receipt("head_sum_select", receipt)?;
    } else {
        run()?;
    }
    download(session, &output, n * row_bytes)
}

/// `topk_merge2` over the ranks' pair bytes, each `[rows][512][2]`.
fn run_merge(
    session: &Session,
    module: &Module,
    rows: &Rows,
    ranks: &[Vec<u8>],
    poison: bool,
    case: &str,
) -> LabResult<Vec<u8>> {
    let n = rows.count();
    let merge = entry(&format!("_topk_merge{}", ranks.len()));
    let gathered = upload(session, &ranks.concat())?;
    let lists = session.allocate(n * WIDTH * 4)?;
    let plan = merge.plan(n as u32).map_err(|e| e.0.to_string())?;
    let function = function(module, &merge.symbol(), 0)?;
    let replay = Replay::new(session).output_with(
        "merged",
        &lists,
        WriteMap::of(DType::I32).prefix(n * WIDTH)?,
        &LIST_FILL,
    );
    let mut receipt = BTreeMap::from([
        ("entry".into(), json!(merge.name)),
        ("case".into(), json!(case)),
        ("rows".into(), json!(n)),
    ]);
    let run = || {
        function
            .launch(
                &session.stream,
                launch::config(plan)?,
                &[
                    KernelArgument::Buffer(&gathered),
                    KernelArgument::Buffer(&rows.positions),
                    KernelArgument::Buffer(&rows.live),
                    KernelArgument::Buffer(&lists),
                    KernelArgument::U32(TOP_K),
                    KernelArgument::U32(n as u32),
                ],
            )
            .map_err(err)
    };
    if poison {
        replay_all(session, replay, &rows.live, run, &mut receipt)?;
        session.receipt("head_sum_merge", receipt)?;
    } else {
        run()?;
    }
    download(session, &lists, n * WIDTH * 4)
}

/// The blocks of a key list row at `position`, ascending: every fourth
/// position before the tail.
fn list_blocks(list: &[i32], position: i32) -> Vec<i32> {
    let complete = (position as usize + 1) / BLOCK;
    list.iter()
        .filter(|&&t| t >= 0 && (t as usize) < complete * BLOCK)
        .step_by(BLOCK)
        .map(|&t| t / BLOCK as i32)
        .collect()
}

fn select_query(entry: &Entry) -> PooledIndexerQuery {
    match entry.query() {
        StaticQuery::PooledIndexer(query) => query,
        _ => unreachable!("a selection entry"),
    }
}

/// What one case measured.
#[derive(Default)]
struct Tally {
    rows: usize,
    /// Rows whose kept scores QSA also scored (more than 512 blocks).
    scored_rows: usize,
    pairs: usize,
    emulation_mismatches: usize,
}

/// Runs both engines on `q` (`[rows][4][128]`) at `positions` over the flat
/// block keys `host_keys`, bound at DCP 1 as `single` and split over two
/// ranks as `ranks`, and requires identical selections and scores.
#[allow(clippy::too_many_arguments)]
fn compare(
    session: &Session,
    modules: &(Module, Module),
    case: &str,
    q: &[u16],
    positions: &[i32],
    host_keys: &[u16],
    (single, ranks): (&Keys, &[Keys; 2]),
    poison: bool,
    exact: bool,
    tally: &mut Tally,
) -> LabResult<()> {
    let (qsa, head_sum) = modules;
    let global = entry("_select");
    let per_rank = entry("_select_per_rank2");
    let query = select_query(&per_rank);
    let rows = Rows::new(session, q, positions)?;
    let n = rows.count();
    let lists = words(
        &run_select(session, head_sum, &global, single, &rows, poison, case)?,
        i32::from_ne_bytes,
    );
    let pairs = words(
        &run_select(session, head_sum, &per_rank, single, &rows, poison, case)?,
        u32::from_ne_bytes,
    );
    let rank_pairs = ranks
        .iter()
        .map(|keys| run_select(session, head_sum, &per_rank, keys, &rows, poison, case))
        .collect::<LabResult<Vec<_>>>()?;
    let merged = words(
        &run_merge(session, head_sum, &rows, &rank_pairs, poison, case)?,
        i32::from_ne_bytes,
    );
    if merged != lists {
        let row = (0..n)
            .find(|&r| merged[r * WIDTH..][..WIDTH] != lists[r * WIDTH..][..WIDTH])
            .unwrap();
        return Err(format!(
            "{case} row {row}: the DCP 2 merge differs from the DCP 1 list"
        ));
    }
    let width = HEADS * DIM;
    let mut actual = Vec::new();
    let mut values = Vec::new();
    let mut s = Vec::new();
    for phase in PHASES {
        let reference = qsa_select(session, qsa, phase, single, &rows)?;
        for (r, &position) in positions.iter().enumerate() {
            let blocks = list_blocks(&lists[r * WIDTH..][..WIDTH], position);
            if blocks != reference.blocks[r] {
                return Err(format!(
                    "{case} row {r} at {position}: the head-sum blocks differ from QSA {phase}'s"
                ));
            }
            let complete = (position as usize + 1) / BLOCK;
            let live = pairs[r * MAX_KEPT * 2..][..MAX_KEPT * 2]
                .chunks_exact(2)
                .filter(|p| p[1] as i32 >= 0)
                .map(|p| (p[0], p[1] as usize))
                .collect::<Vec<_>>();
            if live.len() != complete.min(MAX_KEPT) {
                return Err(format!("{case} row {r}: {} live pairs", live.len()));
            }
            if complete > TOP {
                for &(bits, block) in &live {
                    let qsa_bits = reference.scores[r * reference.stride + block].to_bits();
                    if bits != qsa_bits {
                        return Err(format!(
                            "{case} row {r} block {block}: score {bits:#010x}, QSA {phase} \
                             {qsa_bits:#010x}"
                        ));
                    }
                }
            }
            if phase != "decode" {
                continue;
            }
            tally.rows += 1;
            tally.scored_rows += usize::from(complete > TOP);
            tally.pairs += live.len();
            // The kept blocks' keys, for the emulation and the f64 oracle.
            let kept = live
                .iter()
                .flat_map(|&(_, block)| host_keys[block * DIM..][..DIM].iter().copied())
                .collect::<Vec<_>>();
            let q_row = &q[r * width..][..width];
            let emulated = head_sum_scores_f32(query, q_row, &kept).map_err(|e| e.0.to_string())?;
            let mismatches = live
                .iter()
                .zip(&emulated)
                .filter(|((bits, _), e)| *bits != e.to_bits())
                .count();
            if exact && mismatches > 0 {
                return Err(format!(
                    "{case} row {r}: {mismatches} scores differ from the exact emulation"
                ));
            }
            tally.emulation_mismatches += mismatches;
            let oracle = head_sum_scores(query, &decode(q_row), &decode(&kept))
                .map_err(|e| e.0.to_string())?;
            actual.extend(live.iter().map(|&(bits, _)| f32::from_bits(bits)));
            values.extend(oracle.values);
            s.extend(oracle.sum_abs_products);
        }
    }
    if !actual.is_empty() {
        let reference = OracleOutput::new(values, s, DIM + HEADS).map_err(|e| e.0.to_string())?;
        let mut receipt = BTreeMap::from([("case".into(), json!(case))]);
        kernel_lab::check_output(&actual, &reference, DType::F32, &mut receipt)
            .map_err(|e| format!("{case}: scores outside the 128-K gate: {e}"))?;
    }
    Ok(())
}

fn modules(session: &Session) -> LabResult<(Module, Module)> {
    Ok((
        session.module("sparse_attention_qwen38_sm120")?,
        session.module(shapes::CUBIN)?,
    ))
}

/// The lab geometry is QSA's.
fn check_geometry() {
    let qsa = model_qwen38_contract::sm120::attention::SPARSE;
    let g = shapes::GEOMETRY;
    assert_eq!(
        (g.heads, g.head_dim, g.group, g.page_size, g.max_top_k),
        (
            qsa.index_heads,
            qsa.index_dim,
            qsa.block,
            qsa.page_size,
            qsa.top_blocks * qsa.block
        )
    );
    assert!(g.always_tail);
    assert_eq!(
        (HEADS, DIM, BLOCK, PAGE, TOP),
        (4, 128, 4, 64, qsa.top_blocks as usize)
    );
}

/// Positions of the synthetic rows: the dense limit's sides, a page
/// boundary, short rows and long rows up to 131,072 tokens.
const POSITIONS: [i32; 12] = [
    131_071, 131_068, 100_001, 65_535, 8191, 4097, 2055, 2052, 2051, 2050, 7, 0,
];
const SYNTHETIC_BLOCKS: usize = 131_072 / BLOCK;

/// Synthetic operands: small-integer values with many ties (exact), the
/// chain and head-tree order probes (exact), zero queries (every block
/// +0), general BF16 values and values built to cancel (not exact).
fn synthetic(case: &str, rng: &mut Rng) -> (Vec<u16>, Vec<u16>, bool) {
    let rows = POSITIONS.len();
    let one = f32_to_bf16(1.0);
    let tiny = f32_to_bf16(2_f32.powi(-12));
    match case {
        "integers" => {
            let keys = (0..SYNTHETIC_BLOCKS * DIM).map(|_| rng.int(1, 2)).collect();
            let q = (0..rows * HEADS * DIM)
                .map(|i| match i / (HEADS * DIM) % 3 {
                    // Only head 0 sees the keys, so few scores are distinct.
                    0 if i % (HEADS * DIM) >= DIM => 0,
                    pattern => rng.int(2, pattern as i32),
                })
                .collect();
            (keys, q, true)
        }
        "order" => {
            // Every block scores 1 from head 0's dimension 0. Blocks with
            // flag a add 2^-24 at k16 steps 1 and 3, which two chains keep
            // (one would not); blocks with flag b give heads 1 to 3 dots of
            // 2^-24, which the tree keeps (a head-order sum would not).
            let mut keys = vec![0_u16; SYNTHETIC_BLOCKS * DIM];
            for (block, key) in keys.chunks_exact_mut(DIM).enumerate() {
                let hash = (block as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15) >> 60;
                key[0] = one;
                if hash & 1 == 1 {
                    key[16] = tiny;
                    key[48] = tiny;
                }
                if hash & 2 == 2 {
                    key[1] = one;
                }
            }
            let mut head = vec![0_u16; HEADS * DIM];
            head[0] = one;
            head[16] = tiny;
            head[48] = tiny;
            for h in 1..HEADS {
                head[h * DIM + 1] = f32_to_bf16(2_f32.powi(-24));
            }
            (keys, head.repeat(rows), true)
        }
        "zero_query" => {
            let keys = (0..SYNTHETIC_BLOCKS * DIM).map(|_| rng.int(3, 1)).collect();
            (keys, vec![0; rows * HEADS * DIM], true)
        }
        "cancelling" => {
            // Keys alternate +-64 in channel pairs that the queries weight
            // nearly equally, so each dot is a small sum of large products.
            let keys = (0..SYNTHETIC_BLOCKS * DIM)
                .map(|i| f32_to_bf16(if i % 2 == 0 { 64.0 } else { -64.0 } + rng.unit() / 8.0))
                .collect();
            let q = (0..rows * HEADS * DIM)
                .map(|_| f32_to_bf16(1.0 + rng.unit() / 64.0))
                .collect();
            (keys, q, false)
        }
        "general" => {
            let keys = (0..SYNTHETIC_BLOCKS * DIM)
                .map(|_| f32_to_bf16(2.0 * rng.unit()))
                .collect();
            let q = (0..rows * HEADS * DIM)
                .map(|_| f32_to_bf16(rng.unit()))
                .collect();
            (keys, q, false)
        }
        _ => unreachable!("a synthetic case"),
    }
}

/// QSA's served selection and the head-sum selection agree bit for bit on
/// synthetic operands at DCP 1, and the DCP 2 merge equals DCP 1; every
/// head-sum launch runs under the poison replays.
#[test]
fn head_sum_selection_equals_qsa_on_synthetic_operands() {
    check_geometry();
    gpu_test(4 << 30, |session| {
        let modules = modules(session)?;
        let mut rng = Rng(0x4ead_5a11_0000_0007);
        for case in ["integers", "order", "zero_query", "general", "cancelling"] {
            let (keys, q, exact) = synthetic(case, &mut rng);
            let single = Keys::single(session, &keys)?;
            let ranks = [
                Keys::rank(session, &keys, 2, 0)?,
                Keys::rank(session, &keys, 2, 1)?,
            ];
            let mut tally = Tally::default();
            compare(
                session,
                &modules,
                case,
                &q,
                &POSITIONS,
                &keys,
                (&single, &ranks),
                true,
                exact,
                &mut tally,
            )?;
            eprintln!(
                "{case}: {} rows ({} scored by QSA), {} kept scores, {} differ from the \
                 emulation",
                tally.rows, tally.scored_rows, tally.pairs, tally.emulation_mismatches
            );
            session.receipt(
                "head_sum_selection",
                BTreeMap::from([
                    ("case".into(), json!(case)),
                    ("rows".into(), json!(tally.rows)),
                    ("qsa_scored_rows".into(), json!(tally.scored_rows)),
                    ("kept_scores".into(), json!(tally.pairs)),
                    (
                        "emulation_mismatches".into(),
                        json!(tally.emulation_mismatches),
                    ),
                    ("equals_qsa".into(), json!(true)),
                    ("dcp2_equals_dcp1".into(), json!(true)),
                    ("correct".into(), json!(true)),
                ]),
            )?;
        }
        Ok(())
    })
    .expect("head-sum selection equals QSA on synthetic operands");
}

/// Directories under `root` (itself included, two levels down) that hold
/// layer manifests.
fn export_dirs(root: &Path) -> Vec<PathBuf> {
    let holds_layers = |dir: &Path| {
        fs::read_dir(dir).is_ok_and(|entries| {
            entries.filter_map(Result::ok).any(|e| {
                let name = e.file_name().to_string_lossy().into_owned();
                name.starts_with("layer") && name.ends_with(".json")
            })
        })
    };
    let subdirs = |dir: &Path| -> Vec<PathBuf> {
        let mut dirs = fs::read_dir(dir)
            .map(|entries| {
                entries
                    .filter_map(Result::ok)
                    .map(|e| e.path())
                    .filter(|p| p.is_dir())
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        dirs.sort();
        dirs
    };
    if holds_layers(root) {
        return vec![root.to_path_buf()];
    }
    let mut found = Vec::new();
    for dir in subdirs(root) {
        if holds_layers(&dir) {
            found.push(dir);
        } else {
            found.extend(subdirs(&dir).into_iter().filter(|d| holds_layers(d)));
        }
    }
    found
}

/// One exported layer's projection rows `[tokens][640]`, its indexer norm
/// weights (query then key, `[2][128]`) and its exported rows.
struct Layer {
    qk: Vec<u16>,
    norms: Vec<u16>,
    rows: Vec<usize>,
}

fn read_layer(dir: &Path, layer: u64) -> LabResult<Layer> {
    let bits = |name: &str| -> LabResult<Vec<u16>> {
        let path = dir.join(format!("layer{layer:02}.{name}"));
        let data = fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        if data.len() % 2 != 0 {
            return Err(format!("{}: odd byte count", path.display()));
        }
        Ok(data
            .chunks_exact(2)
            .map(|b| u16::from_le_bytes([b[0], b[1]]))
            .collect())
    };
    let manifest: Value =
        serde_json::from_slice(&fs::read(dir.join(format!("layer{layer:02}.json"))).map_err(err)?)
            .map_err(err)?;
    if manifest["table"] != "hf" {
        return Err(format!(
            "{} layer {layer}: operands use table {}, the kernels read the HF table",
            dir.display(),
            manifest["table"]
        ));
    }
    let layer = Layer {
        qk: bits("qk.bf16")?,
        norms: bits("norm.bf16")?,
        rows: serde_json::from_value(manifest["rows"].clone()).map_err(err)?,
    };
    let tokens = layer.qk.len() / PROJECTION;
    if layer.rows.is_empty()
        || layer.qk.len() != tokens * PROJECTION
        || layer.norms.len() != 2 * DIM
        || layer.rows.iter().any(|&row| row >= tokens)
    {
        return Err(format!(
            "{}: layer shapes disagree or no rows",
            dir.display()
        ));
    }
    Ok(layer)
}

/// The F32 cos/sin table QSA's kernels read, `[position][pair][cos, sin]`.
fn cos_sin_table(positions: usize) -> Vec<f32> {
    let geometry = model_qwen38_contract::sm120::attention::SPARSE;
    (0..positions)
        .flat_map(|position| {
            kernel_core::gated_attention_oracle::mrope_cos_sin(
                [position; 3],
                geometry.rotary_dim as usize,
                f64::from(geometry.theta),
                geometry.mrope,
            )
            .into_iter()
            .flat_map(|(cos, sin)| [cos as f32, sin as f32])
        })
        .collect()
}

/// QSA's own `index_key_write` over every projection row and `index_query`
/// on the exported rows: the block-key pages as written (with their flat
/// copy) and the queries `[rows][4][128]`.
fn qsa_operands(
    session: &Session,
    module: &Module,
    layer: &Layer,
) -> LabResult<(Keys, Vec<u16>, Vec<u16>)> {
    let geometry = model_qwen38_contract::sm120::attention::SPARSE;
    let eps = geometry.eps.to_bits();
    let ring = geometry.ring as usize;
    let tokens = layer.qk.len() / PROJECTION;
    let context = tokens as u32;
    let cos_sin = upload(session, &bytes_f32(&cos_sin_table(tokens)))?;
    let q_norm = upload(session, &bytes_u16(&layer.norms[..DIM]))?;
    let k_norm = upload(session, &bytes_u16(&layer.norms[DIM..]))?;
    let blocks = tokens / BLOCK;
    let pages = blocks.div_ceil(PAGE / BLOCK).max(1);
    let block_keys = upload(session, &vec![0xff; pages * PAGE_STRIDE])?;
    let table = upload(session, &bytes_i32(&(0..pages as i32).collect::<Vec<_>>()))?;
    let slots = upload(session, &bytes_i32(&[0]))?;
    let ring_keys = upload(session, &vec![0xff; ring * DIM * 2])?;
    let ring_positions = upload(session, &vec![0xff; ring * 3 * 4])?;
    let no_anchor = upload(session, &bytes_i32(&[-1]))?;
    let key_write = qsa_entry("index_key_write");
    let mut first = 0;
    while first < tokens {
        let rows = HISTORY_ROWS.min(tokens - first);
        let projection = upload(
            session,
            &bytes_u16(&layer.qk[first * PROJECTION..(first + rows) * PROJECTION]),
        )?;
        let positions: Vec<i32> = (first as i32..(first + rows) as i32).collect();
        let rope: Vec<i32> = positions.iter().flat_map(|&p| [p, p, p]).collect();
        let rope = upload(session, &bytes_i32(&rope))?;
        let row_batch = upload(session, &bytes_i32(&vec![0; rows]))?;
        let positions = upload(session, &bytes_i32(&positions))?;
        let live = upload(session, &(rows as u32).to_ne_bytes())?;
        launch_qsa(
            session,
            module,
            &key_write,
            rows as u32,
            context,
            &[
                KernelArgument::Buffer(&projection),
                KernelArgument::Buffer(&k_norm),
                KernelArgument::Buffer(&rope),
                KernelArgument::Buffer(&cos_sin),
                KernelArgument::Buffer(&row_batch),
                KernelArgument::Buffer(&positions),
                KernelArgument::Buffer(&live),
                KernelArgument::Buffer(&slots),
                KernelArgument::Buffer(&ring_keys),
                KernelArgument::Buffer(&ring_positions),
                KernelArgument::Buffer(&no_anchor),
                KernelArgument::Buffer(&no_anchor),
                KernelArgument::Buffer(&no_anchor),
                KernelArgument::Buffer(&no_anchor),
                KernelArgument::Buffer(&table),
                KernelArgument::Buffer(&block_keys),
                KernelArgument::U32(pages as u32),
                KernelArgument::U64(PAGE_STRIDE as u64),
                KernelArgument::U32(eps),
            ],
        )?;
        first += rows;
    }
    let paged = halves(&download(session, &block_keys, pages * PAGE_STRIDE)?);
    let flat = (0..blocks)
        .flat_map(|b| {
            let at = b / (PAGE / BLOCK) * (PAGE_STRIDE / 2) + b % (PAGE / BLOCK) * DIM;
            paged[at..at + DIM].to_vec()
        })
        .collect();
    let query = qsa_entry("index_query");
    let mut q = Vec::with_capacity(layer.rows.len() * HEADS * DIM);
    for chunk in layer.rows.chunks(HISTORY_ROWS) {
        let projection: Vec<u16> = chunk
            .iter()
            .flat_map(|&r| layer.qk[r * PROJECTION..(r + 1) * PROJECTION].to_vec())
            .collect();
        let rope: Vec<i32> = chunk.iter().flat_map(|&r| [r as i32; 3]).collect();
        let projection = upload(session, &bytes_u16(&projection))?;
        let rope = upload(session, &bytes_i32(&rope))?;
        let live = upload(session, &(chunk.len() as u32).to_ne_bytes())?;
        let out = upload(session, &vec![0xff; chunk.len() * HEADS * DIM * 2])?;
        launch_qsa(
            session,
            module,
            &query,
            chunk.len() as u32,
            context,
            &[
                KernelArgument::Buffer(&projection),
                KernelArgument::Buffer(&q_norm),
                KernelArgument::Buffer(&rope),
                KernelArgument::Buffer(&cos_sin),
                KernelArgument::Buffer(&live),
                KernelArgument::Buffer(&out),
                KernelArgument::U32(eps),
            ],
        )?;
        q.extend(halves(&download(
            session,
            &out,
            chunk.len() * HEADS * DIM * 2,
        )?));
    }
    let keys = Keys {
        buffer: block_keys,
        table: (0..pages as i32).collect(),
        rank: 0,
        size: 1,
    };
    Ok((keys, flat, q))
}

/// QSA's served selection and the head-sum selection agree bit for bit on
/// the captured operands of `CHOPINFER_QSA_OPERANDS`, fed by QSA's own key
/// write and query, at DCP 1, and the DCP 2 merge equals DCP 1. With
/// `CHOPINFER_QSA_OPERANDS_OUT` the summary is written to that new file.
#[test]
fn head_sum_selection_equals_qsa_on_captured_operands() {
    check_geometry();
    let Some(root) = std::env::var_os("CHOPINFER_QSA_OPERANDS").map(PathBuf::from) else {
        kernel_lab::skip("the captured-operand comparison: CHOPINFER_QSA_OPERANDS is unset")
            .expect("head-sum selection on captured operands");
        return;
    };
    let out = std::env::var_os("CHOPINFER_QSA_OPERANDS_OUT").map(PathBuf::from);
    gpu_test(8 << 30, |session| {
        let modules = modules(session)?;
        let dirs = export_dirs(&root);
        if dirs.is_empty() {
            return Err(format!("{}: no layer manifests", root.display()));
        }
        let mut summary = Vec::new();
        let mut checked = 0;
        for dir in dirs {
            let mut layers: Vec<u64> = fs::read_dir(&dir)
                .map_err(|e| format!("{}: {e}", dir.display()))?
                .filter_map(|entry| {
                    let name = entry.ok()?.file_name().into_string().ok()?;
                    name.strip_prefix("layer")?
                        .strip_suffix(".json")?
                        .parse()
                        .ok()
                })
                .collect();
            layers.sort_unstable();
            for number in layers {
                let layer = read_layer(&dir, number)?;
                let (single, keys, q) = qsa_operands(session, &modules.0, &layer)?;
                let ranks = [
                    Keys::rank(session, &keys, 2, 0)?,
                    Keys::rank(session, &keys, 2, 1)?,
                ];
                let case = format!("{} layer {number}", dir.display());
                let mut tally = Tally::default();
                let width = HEADS * DIM;
                for (index, chunk) in layer.rows.chunks(STEP_ROWS).enumerate() {
                    let first = index * STEP_ROWS;
                    let positions = chunk.iter().map(|&row| row as i32).collect::<Vec<_>>();
                    compare(
                        session,
                        &modules,
                        &case,
                        &q[first * width..(first + chunk.len()) * width],
                        &positions,
                        &keys,
                        (&single, &ranks),
                        false,
                        false,
                        &mut tally,
                    )?;
                }
                checked += tally.rows;
                eprintln!(
                    "{case}: {} rows ({} scored by QSA) equal, {} kept scores, {} differ from \
                     the emulation",
                    tally.rows, tally.scored_rows, tally.pairs, tally.emulation_mismatches
                );
                let record = json!({
                    "operands": dir.display().to_string(),
                    "layer": number,
                    "rows": tally.rows,
                    "qsa_scored_rows": tally.scored_rows,
                    "kept_scores": tally.pairs,
                    "emulation_mismatches": tally.emulation_mismatches,
                    "equals_qsa": true,
                    "dcp2_equals_dcp1": true,
                });
                session.receipt(
                    "head_sum_selection",
                    BTreeMap::from([
                        ("case".into(), json!(case)),
                        ("result".into(), record.clone()),
                        ("correct".into(), json!(true)),
                    ]),
                )?;
                summary.push(record);
            }
        }
        if checked == 0 {
            return Err(format!("{}: no row was checked", root.display()));
        }
        if let Some(out) = &out {
            use std::io::Write;
            let value = json!({"operands": root.display().to_string(), "layers": summary});
            fs::File::create_new(out)
                .and_then(|mut f| f.write_all(&serde_json::to_vec_pretty(&value).expect("json")))
                .map_err(|e| format!("{}: {e}", out.display()))?;
        }
        Ok(())
    })
    .expect("head-sum selection equals QSA on captured operands");
}

/// The export walk finds layer directories at the root or one and two
/// levels down, so `exports-v2` covers every context and copy.
#[test]
fn export_walk_finds_nested_layer_directories() {
    let root = std::env::temp_dir().join(format!("head-sum-exports-{}", std::process::id()));
    let nested = root.join("long_4k").join("copy0");
    let flat = root.join("direct");
    fs::create_dir_all(&nested).unwrap();
    fs::create_dir_all(&flat).unwrap();
    fs::write(nested.join("layer03.json"), "{}").unwrap();
    fs::write(flat.join("layer07.json"), "{}").unwrap();
    fs::create_dir_all(root.join("empty")).unwrap();
    assert_eq!(export_dirs(&root), vec![flat.clone(), nested.clone()]);
    assert_eq!(export_dirs(&nested), vec![nested]);
    fs::remove_dir_all(&root).unwrap();
}
