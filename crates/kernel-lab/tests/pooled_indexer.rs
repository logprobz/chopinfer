//! The pooled top-k indexer against its f64 oracles at the served shape
//! (`pooled_indexer_shapes.rs`; README in
//! crates/kernels-sm120/cuda/pooled_indexer).
//!
//! - Selection is exact: with small-integer queries, keys and head weights
//!   every dot product is exact, so the kernel's F32 scores are reproduced on
//!   the host bit for bit and the key lists and per-rank pairs must equal the
//!   oracle's selection of those scores, ties included, from one pool up to
//!   1,048,576 tokens, for decode, verify and prefill rows.
//! - Scores of general operands, and of operands built to cancel, hold the
//!   decision-0081 gate against the Neumaier-compensated f64 oracle (the
//!   128-K accumulation test: one 128-K chain per head).
//! - Per-rank selections over rank-owned pages merge into the global list.
//! - The key write's pooled keys and tail state (prefill launches that
//!   straddle groups, decode and verify checkpoints) hold the oracle.
//! - Every launch runs eager, relaunched, from a graph and behind a PDL
//!   producer on poisoned outputs (`kernel_lab::poison`).
//!
//! Run in release with
//! `CHOPINFER_KERNEL_GPU=<ordinal> cargo test -p kernel-lab --release --test pooled_indexer`.

#[path = "../pooled_indexer_shapes.rs"]
mod shapes;

#[path = "common/support.rs"]
mod support;

use std::collections::BTreeMap;
use std::ops::Range;

use kernel_core::codec::{bf16_to_f32, f32_to_bf16, f64_to_bf16};
use kernel_core::contract::{DType, RecurrentPhase, StaticQuery};
use kernel_core::oracle::{OracleOutput, numeric_bound};
use kernel_core::pooled_indexer::PooledIndexerQuery;
use kernel_core::pooled_indexer_oracle::{self as oracle, KeyWriteOperands, TailState};
use kernel_lab::poison::{Replay, WriteMap, replay_all};
use kernel_lab::{LabResult, Session, gpu_test, launch};
use kernels_sm120::pooled_indexer::{Entry, Kind, pool_address};
use runtime_cuda::{DeviceBuffer, Function, KernelArgument, Module};
use runtime_kv::ContextParallelPagesV1;
use serde_json::json;
use support::{Rng, bytes_i32, bytes_u16, download, upload};

const HEADS: usize = 32;
const DIM: usize = 128;
const GROUP: usize = 4;
const PAGE: usize = 64;
const MAX_KEPT: usize = 512;
const WIDTH: usize = 2051;
/// Pooled key page stride: a page's 4 KiB of pooled keys, then a 1 KiB gap
/// that must stay untouched.
const PAGE_STRIDE: usize = 5120;
/// The fill of key list outputs: no position, and not -1.
const LIST_FILL: [u8; 4] = [0xa5, 0x5a, 0xc3, 0x3c];
/// The fill of per-rank pairs: a NaN score no pair holds.
const PAIR_FILL: [u8; 4] = [0xff, 0xff, 0xff, 0x7f];

fn err(e: impl ToString) -> String {
    e.to_string()
}

/// Draws this file adds to the shared generator.
trait Draws {
    /// An integer in -range..=range.
    fn int(&mut self, range: i64) -> f32;
}

impl Draws for Rng {
    fn int(&mut self, range: i64) -> f32 {
        ((self.next() % (2 * range as u64 + 1)) as i64 - range) as f32
    }
}

fn entry(suffix: &str) -> Entry {
    shapes::lab_entries()
        .into_iter()
        .find(|entry| entry.name.ends_with(suffix))
        .unwrap_or_else(|| panic!("no lab entry {suffix}"))
}

fn module(session: &Session) -> LabResult<Module> {
    session.module(shapes::CUBIN)
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

/// A sequence's pooled keys `[pools][128]` as group rank `rank` of `size`
/// holds them, placed by the host's page ownership
/// (`ContextParallelPagesV1::rank_pages`): one local page per stripe of
/// `size * 64` tokens, the local page ids in an order unlike the stripe
/// index, and the rank's own pages only.
struct Bank {
    buffer: DeviceBuffer,
    /// The rank's table row: its local page per stripe.
    table: Vec<i32>,
    rank: u32,
    size: u32,
}

impl Bank {
    fn new(session: &Session, keys: &[u16], size: usize, rank: usize) -> LabResult<Bank> {
        let tokens = keys.len() / DIM * GROUP;
        let pages = ContextParallelPagesV1::new(size, PAGE, 8).map_err(err)?;
        let stripes = tokens.div_ceil(pages.stripe_tokens()).max(1);
        let table = (0..stripes).map(|s| stripes - 1 - s).collect::<Vec<_>>();
        let mut image = vec![0_u8; stripes * PAGE_STRIDE];
        for page in pages
            .rank_pages(rank, table.iter().copied(), tokens)
            .map_err(err)?
        {
            for offset in (0..page.tokens).step_by(GROUP) {
                let pool = (page.first_position + offset) / GROUP;
                let at = page.local_page * PAGE_STRIDE + offset / GROUP * DIM * 2;
                image[at..at + DIM * 2].copy_from_slice(&bytes_u16(&keys[pool * DIM..][..DIM]));
            }
        }
        Ok(Bank {
            buffer: upload(session, &image)?,
            table: table.into_iter().map(|page| page as i32).collect(),
            rank: rank as u32,
            size: size as u32,
        })
    }

    /// The page tables of `rows` rows, each its own sequence over these keys.
    fn tables(&self, rows: usize) -> Vec<i32> {
        (0..rows).flat_map(|_| self.table.iter().copied()).collect()
    }
}

/// One selection step's device operands.
struct Step {
    q: DeviceBuffer,
    weights: DeviceBuffer,
    table: DeviceBuffer,
    row_batch: DeviceBuffer,
    positions: DeviceBuffer,
    live: DeviceBuffer,
    rows: u32,
}

impl Step {
    fn new(
        session: &Session,
        q: &[u16],
        weights: &[u16],
        table: &[i32],
        positions: &[i32],
    ) -> LabResult<Step> {
        let rows = positions.len();
        Ok(Step {
            q: upload(session, &bytes_u16(q))?,
            weights: upload(session, &bytes_u16(weights))?,
            table: upload(session, &bytes_i32(table))?,
            row_batch: upload(session, &bytes_i32(&(0..rows as i32).collect::<Vec<_>>()))?,
            positions: upload(session, &bytes_i32(positions))?,
            live: upload(session, &(rows as u32).to_ne_bytes())?,
            rows: rows as u32,
        })
    }
}

/// Launches a selection entry's split and merge kernels.
#[allow(clippy::too_many_arguments)]
fn select(
    session: &Session,
    module: &Module,
    entry: &Entry,
    bank: &Bank,
    step: &Step,
    workspace: &DeviceBuffer,
    output: &DeviceBuffer,
    top_k: u32,
) -> LabResult<()> {
    entry
        .check_context_parallel(bank.rank, bank.size)
        .map_err(|e| e.0.to_string())?;
    let plan = entry.plan(step.rows).map_err(|e| e.0.to_string())?;
    let merge_plan = entry
        .merge_plan(step.rows)
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
                KernelArgument::Buffer(&step.q),
                KernelArgument::Buffer(&step.weights),
                KernelArgument::Buffer(&bank.buffer),
                KernelArgument::Buffer(&step.table),
                KernelArgument::Buffer(&step.row_batch),
                KernelArgument::Buffer(&step.positions),
                KernelArgument::Buffer(&step.live),
                KernelArgument::Buffer(workspace),
                KernelArgument::U32(bank.table.len() as u32),
                KernelArgument::U64(PAGE_STRIDE as u64),
                KernelArgument::U32(top_k),
                KernelArgument::U32(splits),
                KernelArgument::U32(bank.rank),
                KernelArgument::U32(bank.size),
            ],
        )
        .map_err(err)?;
    merge
        .launch(
            &session.stream,
            launch::config(merge_plan)?,
            &[
                KernelArgument::Buffer(workspace),
                KernelArgument::Buffer(&step.positions),
                KernelArgument::Buffer(&step.live),
                KernelArgument::Buffer(output),
                KernelArgument::U32(top_k),
                KernelArgument::U32(splits),
            ],
        )
        .map_err(err)
}

fn select_query(entry: &Entry) -> PooledIndexerQuery {
    match entry.query() {
        StaticQuery::PooledIndexer(query) => query,
        _ => unreachable!("a selection entry"),
    }
}

/// Runs a selection under the poison replays and returns the key lists
/// (`Global`) or pair words (`PerRank`) of the live rows.
#[allow(clippy::too_many_arguments)]
fn run_select(
    session: &Session,
    module: &Module,
    entry: &Entry,
    bank: &Bank,
    step: &Step,
    top_k: u32,
    case: &str,
) -> LabResult<Vec<u8>> {
    let rows = step.rows as usize;
    let per_rank = matches!(entry.kind, Kind::Select { ranks: Some(_) });
    let row_bytes = if per_rank { MAX_KEPT * 8 } else { WIDTH * 4 };
    let output = session.allocate(rows * row_bytes)?;
    let plan = entry.plan(step.rows).map_err(|e| e.0.to_string())?;
    let workspace = session.allocate((plan.workspace_bytes as usize).max(16))?;
    let map = if per_rank {
        WriteMap::of_records::<8>()
    } else {
        WriteMap::of(DType::I32)
    }
    .byte_prefix(rows * row_bytes)?;
    let replay = Replay::new(session)
        .output_with(
            "selection",
            &output,
            map,
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
        ("rows".into(), json!(rows)),
        ("top_k".into(), json!(top_k)),
        ("grid".into(), json!(plan.grid)),
        ("workspace_bytes".into(), json!(plan.workspace_bytes)),
    ]);
    replay_all(
        session,
        replay,
        &step.live,
        || {
            select(
                session, module, entry, bank, step, &workspace, &output, top_k,
            )
        },
        &mut receipt,
    )?;
    session.receipt("pooled_indexer_select", receipt)?;
    download(session, &output, rows * row_bytes)
}

/// Small-integer BF16 values in -range..=range, with zeros.
fn integers(rng: &mut Rng, len: usize, range: i64) -> Vec<u16> {
    (0..len).map(|_| f32_to_bf16(rng.int(range))).collect()
}

/// The kernel's F32 scores of integer operands, reproduced on the host:
/// exact dots, `relu(dot * f32(128^-0.5))`, `w = bf16 * f32(32^-0.5)`, F32
/// FMA from zero in head order.
struct Mirror {
    /// Exact dots `[pattern][pool][head]`.
    dots: Vec<Vec<i32>>,
}

impl Mirror {
    fn new(patterns: &[Vec<u16>], keys: &[u16]) -> Mirror {
        let value = |b: u16| bf16_to_f32(b) as i32;
        let dots = patterns
            .iter()
            .map(|q| {
                keys.chunks_exact(DIM)
                    .flat_map(|key| {
                        q.chunks_exact(DIM).map(move |head| {
                            head.iter()
                                .zip(key)
                                .map(|(&a, &b)| value(a) * value(b))
                                .sum()
                        })
                    })
                    .collect()
            })
            .collect();
        Mirror { dots }
    }

    fn scores(&self, pattern: usize, weights: &[u16], pools: usize) -> Vec<f32> {
        let scale = (1.0 / (DIM as f64).sqrt()) as f32;
        let head_scale = (1.0 / (HEADS as f64).sqrt()) as f32;
        let w = weights
            .iter()
            .map(|&b| bf16_to_f32(b) * head_scale)
            .collect::<Vec<_>>();
        (0..pools)
            .map(|pool| {
                let dots = &self.dots[pattern][pool * HEADS..][..HEADS];
                let mut score = 0_f32;
                for (h, &dot) in dots.iter().enumerate() {
                    let x = dot as f32 * scale;
                    score = w[h].mul_add(if x > 0.0 { x } else { 0.0 }, score);
                }
                score
            })
            .collect()
    }
}

/// Selection is exact: key lists and per-rank pairs equal the oracle's
/// selection of the host-reproduced F32 scores, ties to the lower pool.
#[test]
fn selection_matches_the_oracle_bitwise() {
    gpu_test(4 << 30, |session| {
        let module = module(session)?;
        let global = entry("_select");
        let per_rank = entry("_select_per_rank4");
        let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
        let max_pools = 1_048_576 / GROUP;
        // Few distinct values, so many pools tie.
        let keys = integers(&mut rng, max_pools * DIM, 1);
        let patterns = (0..4)
            .map(|_| integers(&mut rng, HEADS * DIM, 2))
            .collect::<Vec<_>>();
        let mirror = Mirror::new(&patterns, &keys);
        let bank = Bank::new(session, &keys, 1, 0)?;
        let decode = [
            1_048_575, 1_048_574, 0, 2, 3, 6, 7, 2047, 2048, 2050, 2051, 2052, 4095, 4099, 8191,
            65_535, 65_536, 262_143, 262_144, 524_287, 1_000_003, 777_777, 4, 5, 11, 63, 64, 2055,
            2063, 16_384, 131_071, 1_048_573,
        ];
        let verify = (1_048_570..1_048_576).collect::<Vec<i32>>();
        let prefill_long = (65_536 - 2048..65_536).collect::<Vec<i32>>();
        let prefill_short = (0..2048).collect::<Vec<i32>>();
        for (case, positions) in [
            ("decode_1", &decode[..1]),
            ("decode_32", &decode[..]),
            ("verify_6", &verify[..]),
            ("prefill_2048_at_64k", &prefill_long[..]),
            ("prefill_2048_from_0", &prefill_short[..]),
            ("special_weights", &decode[..8]),
        ] {
            let rows = positions.len();
            let q = (0..rows)
                .flat_map(|r| patterns[r % 4].iter().copied())
                .collect::<Vec<_>>();
            let weights = if case == "special_weights" {
                // Head 0 weighs +inf, -inf or NaN, giving +-inf and NaN
                // scores (inf * +0 is NaN); or every weight is -0, so every
                // pool scores +0 and the lowest pools win across splits.
                (0..rows * HEADS)
                    .map(|i| match (i / HEADS % 4, i % HEADS) {
                        (0, 0) => 0x7f80,
                        (1, 0) => 0xff80,
                        (2, 0) => 0x7fc0,
                        (3, _) => 0x8000,
                        _ => integers(&mut rng, 1, 3)[0],
                    })
                    .collect()
            } else {
                integers(&mut rng, rows * HEADS, 3)
            };
            let table = bank.tables(rows);
            let step = Step::new(session, &q, &weights, &table, positions)?;
            for top_k in [2048_u32, 1024] {
                let lists = words(
                    &run_select(session, &module, &global, &bank, &step, top_k, case)?,
                    i32::from_ne_bytes,
                );
                let pairs = words(
                    &run_select(session, &module, &per_rank, &bank, &step, top_k, case)?,
                    u32::from_ne_bytes,
                );
                for (r, &position) in positions.iter().enumerate() {
                    let visible = (position as usize + 1) / GROUP;
                    let scores = mirror.scores(r % 4, &weights[r * HEADS..][..HEADS], visible);
                    let expected =
                        oracle::select(select_query(&global), top_k, position as u64, &scores)
                            .map_err(|e| e.0.to_string())?;
                    if lists[r * WIDTH..][..WIDTH] != expected[..] {
                        return Err(format!("{case} top {top_k} row {r}: key list differs"));
                    }
                    let owned = vec![true; visible];
                    let expected = oracle::select_per_rank(
                        select_query(&per_rank),
                        top_k,
                        position as u64,
                        &scores,
                        &owned,
                    )
                    .map_err(|e| e.0.to_string())?;
                    let actual = pairs[r * MAX_KEPT * 2..][..MAX_KEPT * 2]
                        .chunks_exact(2)
                        .map(|p| (f32::from_bits(p[0]), p[1] as i32))
                        .collect::<Vec<_>>();
                    if actual
                        .iter()
                        .zip(&expected)
                        .any(|(a, e)| a.0.to_bits() != e.0.to_bits() || a.1 != e.1)
                    {
                        return Err(format!("{case} top {top_k} row {r}: per-rank pairs differ"));
                    }
                }
            }
        }
        Ok(())
    })
    .expect("pooled indexer selection is exact");
}

/// Scores of general and cancelling operands hold the gate against the
/// compensated f64 oracle: `|y - y_ref| <= 2^-22 K S` with K = 128 + 32.
#[test]
fn scores_hold_the_128k_accumulation_gate() {
    gpu_test(2 << 30, |session| {
        let module = module(session)?;
        let per_rank = entry("_select_per_rank2");
        let query = select_query(&per_rank);
        let mut rng = Rng(0x2545_f491_4f6c_dd1d);
        let pools = 65_536 / GROUP;
        for (case, cancel) in [("general", false), ("cancelling", true)] {
            // Cancelling keys alternate +-64 in channel pairs that the
            // queries weight equally, so each dot is a small sum of large
            // products.
            let keys = (0..pools * DIM)
                .map(|i| {
                    let v = if cancel {
                        (if i % 2 == 0 { 64.0 } else { -64.0 }) + rng.unit() / 8.0
                    } else {
                        2.0 * rng.unit()
                    };
                    f32_to_bf16(v)
                })
                .collect::<Vec<_>>();
            let bank = Bank::new(session, &keys, 1, 0)?;
            let positions = [65_535_i32, 65_534, 40_000, 9_999];
            let q = (0..positions.len() * HEADS * DIM)
                .map(|_| {
                    f32_to_bf16(if cancel {
                        1.0 + rng.unit() / 64.0
                    } else {
                        rng.unit()
                    })
                })
                .collect::<Vec<_>>();
            let weights = (0..positions.len() * HEADS)
                .map(|_| f32_to_bf16(rng.unit()))
                .collect::<Vec<_>>();
            let table = bank.tables(positions.len());
            let step = Step::new(session, &q, &weights, &table, &positions)?;
            let pairs = words(
                &run_select(session, &module, &per_rank, &bank, &step, 2048, case)?,
                u32::from_ne_bytes,
            );
            let decode = |bits: &[u16]| {
                bits.iter()
                    .map(|&b| f64::from(bf16_to_f32(b)))
                    .collect::<Vec<_>>()
            };
            let keys = decode(&keys);
            let mut receipt = BTreeMap::from([
                ("case".into(), json!(case)),
                ("chain_k".into(), json!(DIM)),
                ("reduction_k".into(), json!(DIM + HEADS)),
            ]);
            let (mut actual, mut values, mut s) = (Vec::new(), Vec::new(), Vec::new());
            for (r, &position) in positions.iter().enumerate() {
                let visible = (position as usize + 1) / GROUP;
                let reference = oracle::pool_scores(
                    query,
                    &decode(&q[r * HEADS * DIM..][..HEADS * DIM]),
                    &decode(&weights[r * HEADS..][..HEADS]),
                    &keys[..visible * DIM],
                )
                .map_err(|e| e.0.to_string())?;
                for pair in pairs[r * MAX_KEPT * 2..][..MAX_KEPT * 2].chunks_exact(2) {
                    let pool = pair[1] as usize;
                    actual.push(f32::from_bits(pair[0]));
                    values.push(reference.values[pool]);
                    s.push(reference.sum_abs_products[pool]);
                }
            }
            let reference =
                OracleOutput::new(values, s, DIM + HEADS).map_err(|e| e.0.to_string())?;
            kernel_lab::check_output(&actual, &reference, DType::F32, &mut receipt)?;
            session.receipt("pooled_indexer_scores", receipt)?;
        }
        Ok(())
    })
    .expect("pooled indexer scores hold the gate");
}

/// Per-rank selections, each over its rank's local pages and stripe table
/// as the host builds them, merge into exactly the global key list, at 2 and
/// 4 ranks with ties across ranks.
#[test]
fn per_rank_selections_merge_into_the_global_list() {
    gpu_test(2 << 30, |session| {
        let module = module(session)?;
        let global = entry("_select");
        let mut rng = Rng(0x0123_4567_89ab_cdef);
        let pools = 262_144 / GROUP;
        let keys = integers(&mut rng, pools * DIM, 1);
        let pattern = integers(&mut rng, HEADS * DIM, 2);
        let bank = Bank::new(session, &keys, 1, 0)?;
        let positions = [262_143_i32, 262_140, 100_000, 2052, 2051, 7];
        let rows = positions.len();
        let q = (0..rows)
            .flat_map(|_| pattern.iter().copied())
            .collect::<Vec<_>>();
        let weights = integers(&mut rng, rows * HEADS, 3);
        let full = Step::new(session, &q, &weights, &bank.tables(rows), &positions)?;
        let expected = run_select(
            session,
            &module,
            &global,
            &bank,
            &full,
            2048,
            "merge_global",
        )?;
        for ranks in [2_usize, 4] {
            let per_rank = entry(&format!("_select_per_rank{ranks}"));
            let gathered = session.allocate(ranks * rows * MAX_KEPT * 8)?;
            for rank in 0..ranks {
                // The rank's own local pages, as the host places them.
                let local = Bank::new(session, &keys, ranks, rank)?;
                let step = Step::new(session, &q, &weights, &local.tables(rows), &positions)?;
                let pairs = run_select(
                    session,
                    &module,
                    &per_rank,
                    &local,
                    &step,
                    2048,
                    &format!("rank {rank} of {ranks}"),
                )?;
                gathered
                    .copy_from_bytes_at(rank * rows * MAX_KEPT * 8, &pairs, &session.stream)
                    .map_err(err)?;
            }
            let merge = entry(&format!("_topk_merge{ranks}"));
            let plan = merge.plan(rows as u32).map_err(|e| e.0.to_string())?;
            let function = function(&module, &merge.symbol(), 0)?;
            let lists = session.allocate(rows * WIDTH * 4)?;
            let replay = Replay::new(session).output_with(
                "merged",
                &lists,
                WriteMap::of(DType::I32).prefix(rows * WIDTH)?,
                &LIST_FILL,
            );
            let mut receipt = BTreeMap::from([("ranks".into(), json!(ranks))]);
            replay_all(
                session,
                replay,
                &full.live,
                || {
                    function
                        .launch(
                            &session.stream,
                            launch::config(plan)?,
                            &[
                                KernelArgument::Buffer(&gathered),
                                KernelArgument::Buffer(&full.positions),
                                KernelArgument::Buffer(&full.live),
                                KernelArgument::Buffer(&lists),
                                KernelArgument::U32(2048),
                                KernelArgument::U32(rows as u32),
                            ],
                        )
                        .map_err(err)
                },
                &mut receipt,
            )?;
            if download(session, &lists, rows * WIDTH * 4)? != expected {
                return Err(format!("{ranks} ranks: the merged lists differ"));
            }
            receipt.insert("equals_single_rank".into(), json!(true));
            session.receipt("pooled_indexer_merge", receipt)?;
        }
        // The compaction gathers the merged rows.
        let compact = entry("_compact");
        let function = function(&module, &compact.symbol(), 0)?;
        let source = upload(session, &bytes_i32(&[5, 0, 0, 3]))?;
        let live = upload(session, &4_u32.to_ne_bytes())?;
        let lists = upload(session, &expected)?;
        let out = session.allocate(4 * WIDTH * 4)?;
        let plan = compact.plan(4).map_err(|e| e.0.to_string())?;
        let replay = Replay::new(session).output_with(
            "compacted",
            &out,
            WriteMap::of(DType::I32).prefix(4 * WIDTH)?,
            &LIST_FILL,
        );
        let mut receipt = BTreeMap::new();
        replay_all(
            session,
            replay,
            &live,
            || {
                function
                    .launch(
                        &session.stream,
                        launch::config(plan)?,
                        &[
                            KernelArgument::Buffer(&lists),
                            KernelArgument::Buffer(&source),
                            KernelArgument::Buffer(&live),
                            KernelArgument::Buffer(&out),
                        ],
                    )
                    .map_err(err)
            },
            &mut receipt,
        )?;
        let actual = words(&download(session, &out, 4 * WIDTH * 4)?, i32::from_ne_bytes);
        let all = words(&expected, i32::from_ne_bytes);
        let reference = oracle::compact(select_query(&compact), &all, &[5, 0, 0, 3])
            .map_err(|e| e.0.to_string())?;
        if actual != reference {
            return Err("the compacted lists differ".into());
        }
        session.receipt("pooled_indexer_compact", receipt)?;
        Ok(())
    })
    .expect("per-rank selections merge exactly");
}

/// Context-parallel groups of the key write test: one rank alone, and both
/// ranks of a pair (each stores only the pools of its own pages).
const KEY_WRITE_GROUPS: [(usize, usize); 3] = [(1, 0), (2, 0), (2, 1)];
const SEQUENCES: usize = 3;
/// Positions of each sequence's history.
const HISTORY: usize = 192;
/// Key and gate row stride, in elements.
const STRIDE: usize = 160;
/// Tail state entry of each sequence; verify checkpoints follow.
const SLOTS: [usize; SEQUENCES] = [2, 0, 1];
const VERIFY_ROWS: usize = 6;
const STATE_ELEMENTS: usize = 2 * (GROUP - 1) * DIM;

/// One key write launch: the positions each sequence's rows take.
struct KeyLaunch {
    name: &'static str,
    phase: RecurrentPhase,
    ranges: [Range<usize>; SEQUENCES],
}

/// Prefill launches whose rows straddle groups, pages and launches, then a
/// decode launch and a verify launch.
fn key_launches() -> [KeyLaunch; 4] {
    [
        KeyLaunch {
            name: "prefill",
            phase: RecurrentPhase::Prefill,
            ranges: [0..70, 0..5, 0..2],
        },
        KeyLaunch {
            name: "prefill",
            phase: RecurrentPhase::Prefill,
            ranges: [70..72, 5..14, 2..3],
        },
        KeyLaunch {
            name: "decode",
            phase: RecurrentPhase::Decode,
            ranges: [72..73, 14..15, 3..4],
        },
        KeyLaunch {
            name: "verify",
            phase: RecurrentPhase::Verify {
                max_rows: VERIFY_ROWS as u32,
            },
            ranges: [73..79, 15..18, 4..5],
        },
    ]
}

/// A key write test's page placement: group rank `rank` of `size`, with
/// `stripes` stripes per sequence and the sequences' local pages.
struct KeyPages {
    size: usize,
    rank: usize,
    stripes: usize,
    table: Vec<i32>,
}

impl KeyPages {
    fn new(size: usize, rank: usize) -> KeyPages {
        let stripes = HISTORY.div_ceil(size * PAGE);
        // Sequence s stripe i at local page `SEQUENCES * stripes - 1 - (s
        // * stripes + i)`.
        let table = (0..SEQUENCES * stripes)
            .map(|i| (SEQUENCES * stripes - 1 - i) as i32)
            .collect();
        KeyPages {
            size,
            rank,
            stripes,
            table,
        }
    }

    /// The byte offset of pool `pool` of sequence `s` in the rank's pooled
    /// key pages, when the rank owns it.
    fn pool_offset(&self, s: usize, pool: usize) -> Option<usize> {
        let address = pool_address(
            &shapes::geometry(),
            pool as u64,
            self.rank as u32,
            self.size as u32,
        )?;
        let page = self.table[s * self.stripes + address.stripe as usize] as usize;
        Some(page * PAGE_STRIDE + address.row as usize * DIM * 2)
    }
}

/// Byte ranges of the key and gate rows of tail slot `slot` in state entry
/// `entry`.
fn slot_rows(entry: usize, slot: usize) -> [Range<usize>; 2] {
    let key = (entry * STATE_ELEMENTS + slot * DIM) * 2;
    let gate = key + (GROUP - 1) * DIM * 2;
    [key..key + DIM * 2, gate..gate + DIM * 2]
}

/// What a launch may write. `pools`: the pooled rows of the groups it
/// completes on the rank's pages, written without reading. `tail`: tail
/// slots it writes without reading. `updated`: slots it reads and then
/// rewrites (an in-place tail whose straddling group it pools). Every
/// other byte (earlier pools, the verify source, dead slots, other
/// sequences) must keep its value.
struct KeyMaps {
    pools: Vec<Range<usize>>,
    tail: Vec<Range<usize>>,
    updated: Vec<Range<usize>>,
}

fn key_write_maps(launch: &KeyLaunch, pages: &KeyPages) -> KeyMaps {
    let verify = matches!(launch.phase, RecurrentPhase::Verify { .. });
    let mut maps = KeyMaps {
        pools: Vec::new(),
        tail: Vec::new(),
        updated: Vec::new(),
    };
    for (s, range) in launch.ranges.iter().enumerate() {
        for position in range.clone().filter(|p| p % GROUP == GROUP - 1) {
            if let Some(at) = pages.pool_offset(s, position / GROUP) {
                maps.pools.push(at..at + DIM * 2);
            }
        }
        // Source slots before the launch that a completed straddling group
        // reads.
        let offset = range.start % GROUP;
        let read = if offset > 0 && range.len() >= GROUP - offset {
            offset
        } else {
            0
        };
        if verify {
            for (r, position) in range.clone().enumerate() {
                let entry = SEQUENCES + s * VERIFY_ROWS + r;
                for slot in 0..(position + 1) % GROUP {
                    maps.tail.extend(slot_rows(entry, slot));
                }
            }
        } else {
            let live = range.end % GROUP;
            let base = range.end - live;
            for slot in (0..live).filter(|slot| base + slot >= range.start) {
                let rows = slot_rows(SLOTS[s], slot);
                if slot < read {
                    maps.updated.extend(rows);
                } else {
                    maps.tail.extend(rows);
                }
            }
        }
    }
    maps
}

/// Fails unless the value, its reference and its bound are finite and the
/// value is within the bound.
fn check_within(label: &str, actual: f64, reference: f64, bound: f64) -> LabResult<()> {
    if actual.is_finite()
        && reference.is_finite()
        && bound.is_finite()
        && (actual - reference).abs() <= bound
    {
        return Ok(());
    }
    Err(format!("{label}: {actual} vs {reference} (bound {bound})"))
}

/// A pooled key row against the pooling oracle's gate.
fn check_pooled(label: &str, actual: &[u16], reference: &OracleOutput) -> LabResult<()> {
    for (d, &bits) in actual.iter().enumerate() {
        check_within(
            &format!("{label} channel {d}"),
            f64::from(bf16_to_f32(bits)),
            reference.values[d],
            numeric_bound(reference, d, DType::Bf16),
        )?;
    }
    Ok(())
}

/// BF16 rows that must match bit for bit; no NaN on either side.
fn check_bits(label: &str, actual: &[u16], expected: &[u16]) -> LabResult<()> {
    let nan = |bits: u16| bf16_to_f32(bits).is_nan();
    match actual
        .iter()
        .zip(expected)
        .position(|(&a, &e)| a != e || nan(a) || nan(e))
    {
        None if actual.len() == expected.len() => Ok(()),
        None => Err(format!(
            "{label}: {} of {} values",
            actual.len(),
            expected.len()
        )),
        Some(d) => Err(format!(
            "{label} channel {d}: {:#06x} vs {:#06x}",
            actual[d], expected[d]
        )),
    }
}

/// The single BF16 value every real in `[value - allowance, value +
/// allowance]` rounds to, or `None` when the interval holds a rounding
/// boundary. Rounding is monotonic, so the endpoints decide.
fn unambiguous_bf16(value: f64, allowance: f64) -> Option<u16> {
    let low = f64_to_bf16(value - allowance);
    (low == f64_to_bf16(value + allowance)).then_some(low)
}

/// The norm's arithmetic allowance `2^-22 K S` of one normalized value.
fn norm_allowance(s: f64) -> f64 {
    2_f64.powi(-22) * DIM as f64 * s
}

fn decode_bf16(bits: &[u16]) -> Vec<f64> {
    bits.iter().map(|&b| f64::from(bf16_to_f32(b))).collect()
}

/// One sequence's history: raw keys and gate logits per position, drawn so
/// that every normalized value rounds to one BF16 value within the norm's
/// arithmetic allowance. A kernel whose F32 norm holds the gate then stores
/// exactly the oracle's BF16 keys, and pooling is checked on the operands
/// the kernel pooled.
///
/// A row is `m + s * k_d` with an integer shift m, a scale s of 1/2, 1 or 2
/// and signs k_d, half +1 and half -1 in a random order: its mean is m and
/// its variance s^2 exactly, so its normalized values are `b_d + k_d * w_d *
/// s * rsqrt(s^2 + eps)`, within 2e-6 of `b_d +- w_d`. With the weights 1/2
/// or 1 and biases -2, 1.5, 2.5 or 3 of `key_norm_operands`, those are BF16
/// values of magnitude at least 1/2, far from any rounding boundary; every row
/// is still checked.
struct History {
    raw: Vec<u16>,
    gates: Vec<u16>,
    /// The BF16 normalized keys.
    keys: Vec<u16>,
}

impl History {
    fn draw(rng: &mut Rng, query: PooledIndexerQuery, weight: &[f64], bias: &[f64]) -> History {
        let (mut raw, mut keys) = (Vec::new(), Vec::new());
        for position in 0..HISTORY {
            let shift = rng.int(3);
            let scale = [0.5, 1.0, 2.0][(rng.next() % 3) as usize];
            let mut signs = [[1.0_f32; DIM / 2], [-1.0; DIM / 2]].concat();
            for i in (1..DIM).rev() {
                signs.swap(i, (rng.next() % (i as u64 + 1)) as usize);
            }
            let row = signs
                .iter()
                .map(|k| f32_to_bf16(shift + scale * k))
                .collect::<Vec<_>>();
            let norm = oracle::key_norm(query, &decode_bf16(&row), weight, bias).unwrap();
            let rounded = norm
                .values
                .iter()
                .zip(&norm.sum_abs_products)
                .map(|(&v, &s)| unambiguous_bf16(v, norm_allowance(s)))
                .collect::<Option<Vec<_>>>()
                .unwrap_or_else(|| panic!("position {position}: an ambiguous normalized key"));
            raw.extend(row);
            keys.extend(rounded);
        }
        History {
            raw,
            gates: (0..HISTORY * DIM)
                .map(|_| f32_to_bf16(2.0 * rng.unit()))
                .collect(),
            keys,
        }
    }
}

/// LayerNorm weight and bias whose sums and differences are BF16 values of
/// magnitude at least 1/2 (`History`).
fn key_norm_operands(rng: &mut Rng) -> (Vec<u16>, Vec<u16>) {
    let weight = (0..DIM)
        .map(|_| f32_to_bf16([0.5, 1.0][(rng.next() % 2) as usize]))
        .collect();
    let bias = (0..DIM)
        .map(|_| f32_to_bf16([-2.0, 1.5, 2.5, 3.0][(rng.next() % 4) as usize]))
        .collect();
    (weight, bias)
}

fn key_query(entry: &Entry) -> PooledIndexerQuery {
    match entry.query() {
        StaticQuery::PooledIndexer(query) => query,
        _ => unreachable!("a key write entry"),
    }
}

/// The key write against the oracle: prefill launches that straddle groups,
/// pages and launches, a decode launch and a verify launch, at one rank and
/// at both ranks of a context-parallel pair. Each launch runs under exact
/// write maps over canary-filled state and pages; then its new pooled keys
/// hold the pooling gate on the kernel's own BF16 operands and every live
/// tail slot holds the oracle's normalized key and gate logits bit for bit.
#[test]
fn key_write_matches_the_oracle() {
    gpu_test(1 << 30, |session| {
        let module = module(session)?;
        let mut rng = Rng(0xdead_beef_cafe_f00d);
        let prefill = entry("_key_write_prefill");
        let query = key_query(&prefill);
        let (weight, bias) = key_norm_operands(&mut rng);
        let position = (0..GROUP * DIM)
            .map(|_| f32_to_bf16(rng.unit()))
            .collect::<Vec<_>>();
        let (weight_f, bias_f) = (decode_bf16(&weight), decode_bf16(&bias));
        let history = (0..SEQUENCES)
            .map(|_| History::draw(&mut rng, query, &weight_f, &bias_f))
            .collect::<Vec<_>>();
        let d_norm = upload(
            session,
            &bytes_u16(&[weight.clone(), bias.clone()].concat()),
        )?;
        let d_position = upload(session, &bytes_u16(&position))?;
        let slot_ids = SLOTS.map(|slot| slot as i32);
        let d_slots = upload(session, &bytes_i32(&slot_ids))?;
        let checkpoints = (0..SEQUENCES * VERIFY_ROWS)
            .map(|i| (SEQUENCES + i) as i32)
            .collect::<Vec<_>>();
        let d_checkpoints = upload(session, &bytes_i32(&checkpoints))?;
        let entries = SEQUENCES + SEQUENCES * VERIFY_ROWS;
        // Canaries: no byte of the initial state or pages is zero or 0xff.
        let mut canary = |len: usize| {
            (0..len)
                .map(|_| (rng.next() % 254 + 1) as u8)
                .collect::<Vec<_>>()
        };
        for (size, rank) in KEY_WRITE_GROUPS {
            let pages = KeyPages::new(size, rank);
            let d_table = upload(session, &bytes_i32(&pages.table))?;
            let bank = upload(session, &canary(SEQUENCES * pages.stripes * PAGE_STRIDE))?;
            let state = upload(session, &canary(entries * STATE_ELEMENTS * 2))?;
            let mut tails = (0..SEQUENCES)
                .map(|_| TailState::empty(query).map_err(|e| e.0.to_string()))
                .collect::<LabResult<Vec<_>>>()?;
            for launch in key_launches() {
                let entry = entry(&format!("_key_write_{}", launch.name));
                entry
                    .check_context_parallel(rank as u32, size as u32)
                    .map_err(|e| e.0.to_string())?;
                let (mut keys, mut gates, mut row_batch, mut positions) =
                    (Vec::new(), Vec::new(), Vec::new(), Vec::new());
                for (s, range) in launch.ranges.iter().enumerate() {
                    for p in range.clone() {
                        keys.extend_from_slice(&history[s].raw[p * DIM..][..DIM]);
                        keys.extend(std::iter::repeat_n(0, STRIDE - DIM));
                        gates.extend_from_slice(&history[s].gates[p * DIM..][..DIM]);
                        gates.extend(std::iter::repeat_n(0, STRIDE - DIM));
                        row_batch.push(s as i32);
                        positions.push(p as i32);
                    }
                }
                let rows = positions.len();
                let d_keys = upload(session, &bytes_u16(&keys))?;
                let d_gates = upload(session, &bytes_u16(&gates))?;
                let d_batch = upload(session, &bytes_i32(&row_batch))?;
                let d_positions = upload(session, &bytes_i32(&positions))?;
                let live = upload(session, &(rows as u32).to_ne_bytes())?;
                let plan = entry.plan(rows as u32).map_err(|e| e.0.to_string())?;
                let function = function(&module, &entry.symbol(), 0)?;
                let verify = matches!(launch.phase, RecurrentPhase::Verify { .. });
                let run = || {
                    let mut args = vec![
                        KernelArgument::Buffer(&d_keys),
                        KernelArgument::U32(STRIDE as u32),
                        KernelArgument::Buffer(&d_gates),
                        KernelArgument::U32(STRIDE as u32),
                        KernelArgument::Range {
                            buffer: &d_norm,
                            offset: 0,
                            byte_len: DIM * 2,
                        },
                        KernelArgument::Range {
                            buffer: &d_norm,
                            offset: DIM * 2,
                            byte_len: DIM * 2,
                        },
                        KernelArgument::Buffer(&d_position),
                        KernelArgument::Buffer(&d_batch),
                        KernelArgument::Buffer(&d_positions),
                        KernelArgument::Buffer(&live),
                        KernelArgument::Buffer(&d_slots),
                    ];
                    if verify {
                        args.push(KernelArgument::Buffer(&d_checkpoints));
                    }
                    args.extend([
                        KernelArgument::Buffer(&state),
                        KernelArgument::Buffer(&d_table),
                        KernelArgument::Buffer(&bank),
                        KernelArgument::U32(pages.stripes as u32),
                        KernelArgument::U64(PAGE_STRIDE as u64),
                        KernelArgument::U32(1e-6_f32.to_bits()),
                        KernelArgument::U32(rank as u32),
                        KernelArgument::U32(size as u32),
                    ]);
                    function
                        .launch(&session.stream, launch::config(plan)?, &args)
                        .map_err(err)
                };
                let maps = key_write_maps(&launch, &pages);
                let replay = Replay::new(session)
                    .state(
                        "tail_state",
                        &state,
                        download(session, &state, state.byte_len())?,
                        WriteMap::of(DType::Bf16).byte_ranges(maps.tail.clone())?,
                        maps.updated.clone(),
                    )
                    .state(
                        "pooled_keys",
                        &bank,
                        download(session, &bank, bank.byte_len())?,
                        WriteMap::of(DType::Bf16).byte_ranges(maps.pools.clone())?,
                        Vec::new(),
                    );
                let mut receipt = BTreeMap::from([
                    ("launch".into(), json!(launch.name)),
                    ("rows".into(), json!(rows)),
                    ("cp_size".into(), json!(size)),
                    ("cp_rank".into(), json!(rank)),
                    ("written_pool_rows".into(), json!(maps.pools.len())),
                ]);
                replay_all(session, replay, &live, run, &mut receipt)?;
                let state_now = download(session, &state, state.byte_len())?
                    .chunks_exact(2)
                    .map(|b| u16::from_ne_bytes([b[0], b[1]]))
                    .collect::<Vec<_>>();
                let bank_now = download(session, &bank, bank.byte_len())?
                    .chunks_exact(2)
                    .map(|b| u16::from_ne_bytes([b[0], b[1]]))
                    .collect::<Vec<_>>();
                let label = format!("{} cp {rank} of {size}", launch.name);
                for (s, range) in launch.ranges.iter().enumerate() {
                    let out = oracle::key_write(
                        key_query(&entry),
                        range.start as u64,
                        &tails[s],
                        KeyWriteOperands {
                            raw: &decode_bf16(&history[s].raw[range.start * DIM..range.end * DIM]),
                            gates: &decode_bf16(
                                &history[s].gates[range.start * DIM..range.end * DIM],
                            ),
                            norm_weight: &weight_f,
                            norm_bias: &bias_f,
                            position: &decode_bf16(&position),
                        },
                    )
                    .map_err(|e| e.0.to_string())?;
                    // The oracle pools the same BF16 operands the kernel
                    // stores (the history is unambiguous).
                    for (i, p) in range.clone().enumerate() {
                        let rounded = out.normalized.values[i * DIM..][..DIM]
                            .iter()
                            .map(|&v| f32_to_bf16(v as f32))
                            .collect::<Vec<_>>();
                        check_bits(
                            &format!("{label}: oracle operands of position {p}"),
                            &rounded,
                            &history[s].keys[p * DIM..][..DIM],
                        )?;
                    }
                    for pooled in &out.pools {
                        let Some(at) = pages.pool_offset(s, pooled.pool as usize) else {
                            continue;
                        };
                        check_pooled(
                            &format!("{label}: sequence {s} pool {}", pooled.pool),
                            &bank_now[at / 2..][..DIM],
                            &pooled.key,
                        )?;
                    }
                    let targets = if verify {
                        range
                            .clone()
                            .enumerate()
                            .map(|(r, p)| (SEQUENCES + s * VERIFY_ROWS + r, p))
                            .collect::<Vec<_>>()
                    } else {
                        vec![(SLOTS[s], range.end - 1)]
                    };
                    for (target, last) in targets {
                        let base = (last + 1) / GROUP * GROUP;
                        for slot in 0..(last + 1) % GROUP {
                            let token = base + slot;
                            let [key, gate] = slot_rows(target, slot);
                            let slot_label =
                                format!("{label}: sequence {s} entry {target} slot {slot}");
                            check_bits(
                                &format!("{slot_label} key"),
                                &state_now[key.start / 2..key.end / 2],
                                &history[s].keys[token * DIM..][..DIM],
                            )?;
                            check_bits(
                                &format!("{slot_label} gate"),
                                &state_now[gate.start / 2..gate.end / 2],
                                &history[s].gates[token * DIM..][..DIM],
                            )?;
                        }
                    }
                    if !verify && let Some(last) = out.checkpoints.last() {
                        tails[s] = last.clone();
                    }
                }
                session.receipt("pooled_indexer_key_write", receipt)?;
            }
        }
        Ok(())
    })
    .expect("pooled indexer key write holds the oracle");
}

/// The key write maps hold exactly the launch's writes: a new pool and new
/// tail slots are in them; an earlier pool, a pool on the other rank's page,
/// the verify source, dead slots and read-only tail slots are not.
#[test]
fn key_write_maps_are_exact() {
    let contains = |ranges: &[Range<usize>], at: usize| ranges.iter().any(|r| r.contains(&at));
    let launches = key_launches();
    let single = KeyPages::new(1, 0);
    let first = key_write_maps(&launches[0], &single);
    let second = key_write_maps(&launches[1], &single);
    let pool = |pages: &KeyPages, s, pool| pages.pool_offset(s, pool).unwrap();
    // Sequence 0 completes pools 0..17 in the first prefill, 17 in the
    // second.
    assert!(contains(&first.pools, pool(&single, 0, 0)));
    assert!(!contains(&second.pools, pool(&single, 0, 0)));
    assert!(contains(&second.pools, pool(&single, 0, 17)));
    assert_eq!(first.pools.len(), 17 + 1);
    // Sequence 1's second prefill (5..14) reads slot 0 (position 4) to pool
    // 4..8 and rewrites it with position 12: an update, not a write.
    assert!(contains(&second.updated, slot_rows(SLOTS[1], 0)[0].start));
    assert!(contains(&second.tail, slot_rows(SLOTS[1], 1)[1].start));
    assert!(!contains(&second.tail, slot_rows(SLOTS[1], 2)[0].start));
    // Sequence 2's second prefill (position 2) writes slot 2 only.
    assert!(contains(&second.tail, slot_rows(SLOTS[2], 2)[0].start));
    assert!(!contains(&second.tail, slot_rows(SLOTS[2], 0)[0].start));
    // Verify writes checkpoints only: never a source entry, never a dead
    // slot of a checkpoint.
    let verify = key_write_maps(&launches[3], &single);
    for entry in 0..SEQUENCES {
        for slot in 0..GROUP - 1 {
            for row in slot_rows(entry, slot) {
                assert!(!contains(&verify.tail, row.start));
                assert!(!contains(&verify.updated, row.start));
            }
        }
    }
    // Sequence 1 rows 15..18: 15 completes a pool, 16 leaves one slot, 17
    // two.
    let checkpoint = |r| SEQUENCES + VERIFY_ROWS + r;
    assert!(!contains(
        &verify.tail,
        slot_rows(checkpoint(0), 0)[0].start
    ));
    assert!(contains(&verify.tail, slot_rows(checkpoint(1), 0)[0].start));
    assert!(!contains(
        &verify.tail,
        slot_rows(checkpoint(1), 1)[0].start
    ));
    assert!(contains(&verify.tail, slot_rows(checkpoint(2), 1)[1].start));
    // At two ranks, pool 16 (page 1) is rank 1's; pool 0 is rank 0's.
    let (zero, one) = (KeyPages::new(2, 0), KeyPages::new(2, 1));
    assert!(zero.pool_offset(0, 16).is_none() && one.pool_offset(0, 0).is_none());
    let rank_one = key_write_maps(&launches[0], &one);
    assert!(contains(&rank_one.pools, pool(&one, 0, 16)));
    assert_eq!(rank_one.pools.len(), 1);
    assert_eq!(key_write_maps(&launches[0], &zero).pools.len(), 17);
}

/// Every drawn history row normalizes to unambiguous BF16 keys, which are
/// the oracle's own rounding of its normalized values.
#[test]
fn key_write_histories_are_unambiguous() {
    let mut rng = Rng(0xdead_beef_cafe_f00d);
    let query = key_query(&entry("_key_write_prefill"));
    let (weight, bias) = key_norm_operands(&mut rng);
    let (weight, bias) = (decode_bf16(&weight), decode_bf16(&bias));
    let history = History::draw(&mut rng, query, &weight, &bias);
    let norm = oracle::key_norm(query, &decode_bf16(&history.raw), &weight, &bias).unwrap();
    let rounded = norm
        .values
        .iter()
        .map(|&v| f32_to_bf16(v as f32))
        .collect::<Vec<_>>();
    assert_eq!(rounded, history.keys);
    assert!(history.keys.iter().all(|&k| bf16_to_f32(k).abs() >= 0.49));
}

/// The key write checks fail closed: a NaN or a corrupted value in a pooled
/// key or a tail slot fails, and the unambiguous rounding is two-sided.
#[test]
fn key_write_checks_reject_nan_and_corruption() {
    let pooling = kernel_core::gated_pooling::GatedGroupPooling {
        group: GROUP as u32,
        width: DIM as u32,
        position_dtype: DType::Bf16,
    };
    let mut rng = Rng(7);
    let keys = (0..GROUP * DIM)
        .map(|_| f64::from(bf16_to_f32(f32_to_bf16(rng.unit()))))
        .collect::<Vec<_>>();
    let gates = (0..GROUP * DIM)
        .map(|_| f64::from(rng.unit()))
        .collect::<Vec<_>>();
    let reference =
        kernel_core::gated_pooling_oracle::pool(pooling, &keys, &gates, &[0.0; GROUP * DIM])
            .unwrap();
    let good = reference
        .values
        .iter()
        .map(|&v| f32_to_bf16(v as f32))
        .collect::<Vec<_>>();
    check_pooled("good", &good, &reference).unwrap();
    for bad in [0x7fc0_u16, 0xffc0, 0x7f80] {
        let mut mutated = good.clone();
        mutated[5] = bad;
        assert!(check_pooled("mutated", &mutated, &reference).is_err());
    }
    let mut corrupted = good.clone();
    corrupted[9] ^= 0x0100;
    assert!(check_pooled("corrupted", &corrupted, &reference).is_err());
    check_bits("good", &good, &good).unwrap();
    let mut nan = good.clone();
    nan[0] = 0x7fc0;
    assert!(check_bits("nan", &nan, &good).is_err());
    assert!(check_bits("nan both", &nan, &nan).is_err());
    assert!(check_within("nan", f64::NAN, 1.0, 1.0).is_err());
    assert!(check_within("inf bound", 1.0, 1.0, f64::INFINITY).is_err());
    // Below 1 the BF16 spacing is 2^-8, above it 2^-7: the interval around
    // 0.998046875 + 2^-21 reaches 1 - 2^-9 (the midpoint below 1), and the
    // one around 1 + 2^-8 - 2^-20 reaches the midpoint above.
    assert_eq!(
        unambiguous_bf16(0.998_046_875 + 2_f64.powi(-21), 2_f64.powi(-14)),
        None
    );
    assert_eq!(
        unambiguous_bf16(1.0 + 2_f64.powi(-8) - 2_f64.powi(-20), 2_f64.powi(-14)),
        None
    );
    assert_eq!(
        unambiguous_bf16(-(1.0 + 2_f64.powi(-8)) + 2_f64.powi(-20), 2_f64.powi(-14)),
        None
    );
    assert_eq!(
        unambiguous_bf16(1.25, 2_f64.powi(-14)),
        Some(f32_to_bf16(1.25))
    );
    assert_eq!(
        unambiguous_bf16(-0.75, 2_f64.powi(-14)),
        Some(f32_to_bf16(-0.75))
    );
    // Clear of the midpoint by more than the allowance: one rounding.
    let midpoint = 1.0 + 2_f64.powi(-8);
    assert_eq!(
        unambiguous_bf16(midpoint + 2_f64.powi(-12), 2_f64.powi(-14)),
        Some(f32_to_bf16(1.0078125))
    );
}

/// The selection workspace does not depend on the context: each split's
/// kept keys for every prepared row bucket.
#[test]
fn selection_workspace_is_bounded() {
    let select = entry("_select");
    for rows in [1_u32, 2, 6, 32, 192, 2048] {
        let plan = select.plan(rows).unwrap();
        assert_eq!(
            plan.workspace_bytes,
            u64::from(rows) * u64::from(plan.grid[1]) * MAX_KEPT as u64 * 8
        );
        assert!(plan.workspace_bytes <= 8 << 20);
    }
}
