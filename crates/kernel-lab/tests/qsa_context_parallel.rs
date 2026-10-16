//! QSA's full-attention layer at context-parallel size 2, emulated on one
//! GPU against QSA's served DCP 1 layer (`owned_writes_shapes.rs`; READMEs
//! in crates/kernels-sm120/cuda/owned_writes and cuda/pooled_indexer).
//!
//! DCP 1 runs the served kernels: the gated norm, RoPE and KV write, the
//! indexer key write and query, `block_scores` and `block_topk`, and the
//! selected attention with its split merge, once per TP rank of the KV
//! head's replica set (ranks 0 and 1 share KV head 0). DCP 2 runs, per
//! group rank d:
//!
//! 1. the position localization and the owner-routed K/V write of TP rank
//!    d's rows into its own pages;
//! 2. the owner-routed indexer key write (the ring on both ranks);
//! 3. the head-sum per-rank selection over its own block keys;
//! 4. the gathered pairs merged into block ids (`topk_merge_blocks2`);
//! 5. the selected partials of both ranks' queries over its own pages; the
//!    owner gathers (on one GPU, a copy) and `partial_merge2` give each TP
//!    rank's attention.
//!
//! Required: the owned K/V rows and block keys equal the served bytes at
//! their pages, the rings equal, and the Q rows equal (bitwise). The DCP 2
//! block lists equal QSA's (bitwise). QSA's DCP 1 attention passes the f64
//! block-sparse attention's gate, and the DCP 2 merged attention passes the
//! same gate plus the merge's `exp2f` allowance. Passing both gates bounds
//! the two outputs' difference only by the sum of the bounds; their paired
//! difference is reported as a diagnostic and not gated.
//!
//! The owner-routed writes and the block merges also run alone under the
//! poison replays (eager, relaunch, graph, PDL) against the served kernels
//! and kernel-core's placement oracles. `CHOPINFER_QSA_OPERANDS` (a
//! `qwen38-oracle qsa-capture --export` directory or a directory of them)
//! adds captured indexer operands of contexts up to `CAPTURED_TOKENS` to
//! the chain, with synthetic attention operands.
//!
//! Run in release with `CHOPINFER_KERNEL_GPU=<ordinal> cargo test -p
//! kernel-lab --release --test qsa_context_parallel`.

#[allow(dead_code)]
#[path = "../context_parallel_shapes.rs"]
mod cp_shapes;
#[path = "common/selected_partials.rs"]
mod fixture;
#[allow(dead_code)]
#[path = "../head_sum_shapes.rs"]
mod head_sum_shapes;
#[allow(dead_code)]
#[path = "../paged_partials_shapes.rs"]
mod merge_shapes;
#[path = "../owned_writes_shapes.rs"]
mod shapes;
#[allow(dead_code)]
#[path = "../selected_partials_shapes.rs"]
mod sp_shapes;

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use fixture::{
    Kv, Launch, Partials, RankCache, Rng, Step, bf16_values, download, err, full_reference, gate,
    gate_merged, merge, run, upload, words,
};
use kernel_core::codec::{bf16_to_f32, f32_to_bf16};
use kernel_core::context_parallel_oracle::{owned_block_row, owned_kv_positions};
use kernel_core::contract::DType;
use kernel_core::oracle::e4m3_table;
use kernel_core::pooled_indexer_oracle::merge_blocks;
use kernel_lab::poison::{Replay, WriteMap, replay_all};
use kernel_lab::{LabResult, Session, gpu_test, launch};
use kernels_sm120::attention::{gated, sparse};
use kernels_sm120::owned_writes;
use runtime_cuda::{DeviceBuffer, Function, KernelArgument, Module};
use serde_json::{Value, json};

const HEADS: usize = 6;
const DIM: usize = 256;
const ROW: usize = HEADS * 2 * DIM + 2 * DIM;
const PAGE: usize = 64;
const BLOCK: usize = 4;
const TOP: usize = 512;
const INDEX_HEADS: usize = 4;
const INDEX_DIM: usize = 128;
const PROJECTION: usize = (INDEX_HEADS + 1) * INDEX_DIM;
const RING: usize = 16;
/// Slots of the anchor ring; the anchored sequence uses the last, so the
/// first must stay untouched.
const ANCHOR_SLOTS: usize = 2;
/// Bytes between pages that a read or write outside a page would hit.
const DECOY: usize = 512;
/// Block-key page stride: 4 KiB of block keys and a 1 KiB gap.
const KEY_STRIDE: usize = 5120;
/// Rows of one history write launch: not a multiple of the block, so blocks
/// straddle launches and the key write pools them from the ring.
const CHUNK: usize = 4093;
const SOFTMAX_SCALE: f32 = 0.0625;
/// The longest captured context the chain takes.
const CAPTURED_TOKENS: usize = 32_768;
const GROUP: usize = 2;

fn function(module: &Module, symbol: &str, shared: u32) -> LabResult<Function> {
    let function = module.function(symbol).map_err(err)?;
    if shared > 48 * 1024 {
        function
            .set_max_dynamic_shared_memory(shared)
            .map_err(err)?;
    }
    Ok(function)
}

fn halves(values: &[u16]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_ne_bytes()).collect()
}

fn floats_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_ne_bytes()).collect()
}

fn ints(bytes: &[u8]) -> Vec<i32> {
    bytes
        .chunks_exact(4)
        .map(|c| i32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn u16s(bytes: &[u8]) -> Vec<u16> {
    bytes
        .chunks_exact(2)
        .map(|c| u16::from_ne_bytes([c[0], c[1]]))
        .collect()
}

fn qsa_entry(name: &str) -> sparse::Entry {
    sparse::catalog(kernel_lab::catalogs())
        .into_iter()
        .find(|entry| entry.name == format!("qwen38_{name}"))
        .unwrap_or_else(|| panic!("no QSA entry {name}"))
}

fn gated_entry(fp8: bool) -> gated::Entry {
    let symbol = format!(
        "chop_attn_qwen38_qk_norm_rope_kv_{}_v1",
        if fp8 { "fp8" } else { "bf16" }
    );
    gated::catalog(kernel_lab::catalogs())
        .into_iter()
        .find(|entry| entry.symbol == symbol)
        .unwrap_or_else(|| panic!("no gated entry {symbol}"))
}

fn owned_entry(name: &str) -> owned_writes::Entry {
    shapes::lab_entries()
        .into_iter()
        .find(|entry| entry.name.ends_with(name))
        .unwrap_or_else(|| panic!("no owner-routed entry {name}"))
}

/// The lab geometry is QSA's TP4 rank and indexer.
fn check_geometry() -> LabResult<()> {
    let qsa = model_qwen38_contract::sm120::attention::SPARSE;
    let served = model_qwen38_contract::sm120::attention::GATED;
    let (g, i) = (shapes::GATED, shapes::INDEXER);
    let same_gated = (
        g.q_heads,
        g.kv_heads,
        g.head_dim,
        g.rotary,
        g.theta.to_bits(),
        g.sections,
        g.eps.to_bits(),
        g.page,
        g.max_rows,
    ) == (
        served.q_heads,
        served.kv_heads,
        served.head_dim,
        served.rotary,
        served.theta.to_bits(),
        served.sections,
        served.eps.to_bits(),
        served.page,
        served.max_rows,
    );
    let same_indexer = (
        i.index_heads,
        i.index_dim,
        i.rotary_dim,
        i.theta.to_bits(),
        i.mrope,
        i.eps.to_bits(),
        i.block,
        i.ring,
        i.top_blocks,
        i.page_size,
        i.softmax_scale.to_bits(),
    ) == (
        qsa.index_heads,
        qsa.index_dim,
        qsa.rotary_dim,
        qsa.theta.to_bits(),
        qsa.mrope,
        qsa.eps.to_bits(),
        qsa.block,
        qsa.ring,
        qsa.top_blocks,
        qsa.page_size,
        qsa.softmax_scale.to_bits(),
    );
    if !same_gated || !same_indexer || SOFTMAX_SCALE != qsa.softmax_scale {
        return Err("the lab geometry is not QSA's TP4 rank".into());
    }
    Ok(())
}

/// The F32 cos/sin table `[position][pair][cos, sin]` both writes read.
fn cos_sin_table(tokens: usize) -> Vec<f32> {
    let g = shapes::GATED;
    (0..tokens)
        .flat_map(|position| {
            kernel_core::gated_attention_oracle::mrope_cos_sin(
                [position; 3],
                g.rotary as usize,
                f64::from(g.theta),
                g.sections,
            )
            .into_iter()
            .flat_map(|(cos, sin)| [cos as f32, sin as f32])
        })
        .collect()
}

/// One sequence's operands: per TP rank of the replica set, packed q|k|v
/// rows sharing the K and V columns; the indexer projection rows and norms;
/// the Q and K norm weights; and the step's rows.
struct Sequence {
    tokens: usize,
    qkv: [Vec<u16>; GROUP],
    projection: Vec<u16>,
    index_norms: Vec<u16>,
    q_norm: Vec<u16>,
    k_norm: Vec<u16>,
    /// The step's positions, consecutive at the end of the sequence.
    rows: Vec<usize>,
    /// `decode` (QSA's graph entries) or `prefill`.
    phase: &'static str,
}

impl Sequence {
    fn synthetic(tokens: usize, rows: usize, phase: &'static str, rng: &mut Rng) -> Sequence {
        let projection = (0..tokens * PROJECTION)
            .map(|_| f32_to_bf16(rng.unit()))
            .collect();
        let index_norms = (0..2 * INDEX_DIM)
            .map(|_| f32_to_bf16(0.25 * rng.unit()))
            .collect();
        Self::with_indexer(tokens, rows, phase, projection, index_norms, rng)
    }

    fn with_indexer(
        tokens: usize,
        rows: usize,
        phase: &'static str,
        projection: Vec<u16>,
        index_norms: Vec<u16>,
        rng: &mut Rng,
    ) -> Sequence {
        let kv = (0..tokens * 2 * DIM)
            .map(|_| f32_to_bf16(rng.unit()))
            .collect::<Vec<_>>();
        let qkv = std::array::from_fn(|_| {
            let mut rows = vec![0_u16; tokens * ROW];
            for t in 0..tokens {
                for (i, value) in rows[t * ROW..][..HEADS * 2 * DIM].iter_mut().enumerate() {
                    *value = f32_to_bf16(if i % (2 * DIM) < DIM {
                        2.0 * rng.unit()
                    } else {
                        rng.unit()
                    });
                }
                rows[t * ROW + HEADS * 2 * DIM..][..2 * DIM]
                    .copy_from_slice(&kv[t * 2 * DIM..][..2 * DIM]);
            }
            rows
        });
        Sequence {
            tokens,
            qkv,
            projection,
            index_norms,
            q_norm: (0..DIM).map(|_| f32_to_bf16(0.25 * rng.unit())).collect(),
            k_norm: (0..DIM).map(|_| f32_to_bf16(0.25 * rng.unit())).collect(),
            rows: (tokens - rows..tokens).collect(),
            phase,
        }
    }

    fn pages(&self) -> usize {
        self.tokens.div_ceil(PAGE)
    }
}

/// A rank's pages: K and V planes `[page][PAGE][DIM]` with decoy gaps,
/// block keys `[page][PAGE / BLOCK][INDEX_DIM]` at `KEY_STRIDE`, and the
/// table row of local pages per stripe.
struct Pages {
    k: DeviceBuffer,
    v: DeviceBuffer,
    keys: DeviceBuffer,
    ring_keys: DeviceBuffer,
    ring_positions: DeviceBuffer,
    /// The anchor ring `[ANCHOR_SLOTS][RING]` of raw keys and positions.
    anchor_keys: DeviceBuffer,
    anchor_positions: DeviceBuffer,
    /// The anchored sequence's `anchor_ends`: the launch holding positions
    /// `end - RING..end` also copies them into the last anchor slot.
    anchor_end: Option<usize>,
    table: Vec<i32>,
    stride: usize,
    /// Group rank and size: (0, 1) for DCP 1.
    rank: usize,
    size: usize,
}

impl Pages {
    /// DCP 1: every page, in order. DCP 2: the rank's own pages, local
    /// pages in an order unlike the stripe index.
    fn new(
        session: &Session,
        seq: &Sequence,
        fp8: bool,
        rank: usize,
        size: usize,
    ) -> LabResult<Pages> {
        let stripes = seq.pages().div_ceil(size);
        let table = if size == 1 {
            (0..stripes as i32).collect()
        } else {
            (0..stripes as i32).rev().collect::<Vec<_>>()
        };
        let element = if fp8 { 1 } else { 2 };
        let stride = PAGE * DIM * element + DECOY;
        let decoy = if fp8 { 0x7e } else { 0x7f };
        Ok(Pages {
            k: upload(session, &vec![decoy; stripes * stride])?,
            v: upload(session, &vec![decoy; stripes * stride])?,
            keys: upload(session, &vec![0x7f; stripes * KEY_STRIDE])?,
            ring_keys: upload(session, &vec![0xff; RING * INDEX_DIM * 2])?,
            ring_positions: upload(session, &[0xff; RING * 3 * 4])?,
            anchor_keys: upload(session, &vec![0xff; ANCHOR_SLOTS * RING * INDEX_DIM * 2])?,
            anchor_positions: upload(session, &vec![0xff; ANCHOR_SLOTS * RING * 3 * 4])?,
            anchor_end: None,
            table,
            stride,
            rank,
            size,
        })
    }

    /// Anchors the sequence's ring at `end` (see `anchor_end`).
    fn anchored(mut self, end: usize) -> Pages {
        self.anchor_end = Some(end);
        self
    }

    /// The local page and slot of position `t`, or `None` off the rank.
    fn place(&self, t: usize) -> Option<(usize, usize)> {
        place(&self.table, self.rank, self.size, t)
    }
}

/// The local page and slot of position `t` on group rank `rank` of `size`
/// with table row `table`, or `None` when another rank owns its page.
fn place(table: &[i32], rank: usize, size: usize, t: usize) -> Option<(usize, usize)> {
    let physical = t / PAGE;
    (physical % size == rank).then(|| (table[physical / size] as usize, t % PAGE))
}

/// What a K/V write of positions `first..first + n` may write in one plane
/// of pages strided by `stride`: each owned row, in whole elements of the
/// cache dtype. A BF16 element is two bytes, so a value such as 0x3eff,
/// whose low byte is the poison byte, still counts as written (only the
/// all-0xff NaN never is); an FP8 E4M3 code is one byte and 0xff is NaN.
fn kv_write_map(
    table: &[i32],
    (rank, size): (usize, usize),
    stride: usize,
    fp8: bool,
    (first, n): (usize, usize),
) -> LabResult<WriteMap> {
    let cache = WriteMap::of(if fp8 { DType::Fp8E4M3 } else { DType::Bf16 });
    let element = cache.bytes()?;
    let rows = (first..first + n)
        .filter_map(|t| place(table, rank, size, t))
        .map(|(page, slot)| {
            let at = page * stride + slot * DIM * element;
            at..at + DIM * element
        });
    cache.byte_ranges(rows)
}

/// One launch's step rows of the sequence: positions `first..first + n`.
struct Chunk {
    first: usize,
    n: usize,
}

/// Launches the gated K/V write of `chunk` for TP rank `source`: the served
/// entry (DCP 1) or the owner-routed one over localized positions, under the
/// poison replays when `poison`. Returns the Q rows `[n][HEADS][DIM]`.
#[allow(clippy::too_many_arguments)]
fn kv_write(
    session: &Session,
    modules: &Modules,
    seq: &Sequence,
    source: usize,
    chunk: &Chunk,
    pages: &Pages,
    fp8: bool,
    cos_sin: &DeviceBuffer,
    poison: bool,
) -> LabResult<Vec<u16>> {
    let n = chunk.n;
    let positions = (chunk.first..chunk.first + n)
        .map(|p| p as i32)
        .collect::<Vec<_>>();
    let rope = positions.iter().flat_map(|&p| [p; 3]).collect::<Vec<_>>();
    let qkv = upload(
        session,
        &halves(&seq.qkv[source][chunk.first * ROW..(chunk.first + n) * ROW]),
    )?;
    let q_norm = upload(session, &halves(&seq.q_norm))?;
    let k_norm = upload(session, &halves(&seq.k_norm))?;
    let rope = upload(session, &words(&rope))?;
    let row_batch = upload(session, &words(&vec![0; n]))?;
    let live = upload(session, &(n as u32).to_ne_bytes())?;
    let table = upload(session, &words(&pages.table))?;
    let q = session.allocate(n * HEADS * DIM * 2)?;
    let owned = pages.size > 1;
    let (symbol, plan, position_buffer) = if owned {
        // The localization a step runs once for every layer.
        let localize = cp_shapes::lab_entries()
            .into_iter()
            .find(|entry| entry.name.ends_with("localize_positions2"))
            .ok_or("no lab position localization")?;
        localize
            .check_rank(pages.rank as u32)
            .map_err(|e| e.0.to_string())?;
        let global = upload(session, &words(&positions))?;
        let local = session.allocate(n * 4)?;
        function(&modules.localize, &localize.symbol(), 0)?
            .launch(
                &session.stream,
                launch::config(localize.plan(n as u32).map_err(|e| e.0.to_string())?)?,
                &[
                    KernelArgument::Buffer(&global),
                    KernelArgument::Buffer(&live),
                    KernelArgument::Buffer(&local),
                    KernelArgument::U32(pages.rank as u32),
                ],
            )
            .map_err(err)?;
        let entry = owned_entry(if fp8 {
            "_kv_write_fp8"
        } else {
            "_kv_write_bf16"
        });
        let expected = owned_kv_positions(entry.query(), pages.rank as u32, &positions)
            .map_err(|e| e.0.to_string())?;
        if ints(&download(session, &local, n * 4)?) != expected {
            return Err("the device's localized positions differ from the oracle's".into());
        }
        (
            entry.symbol(),
            entry.plan(n as u32).map_err(|e| e.0.to_string())?,
            local,
        )
    } else {
        let entry = gated_entry(fp8);
        (
            entry.symbol.clone(),
            entry.plan(n as u32).map_err(|e| e.0.to_string())?,
            upload(session, &words(&positions))?,
        )
    };
    let module = if owned {
        &modules.owned
    } else {
        &modules.gated
    };
    let kernel = function(module, &symbol, 0)?;
    let eps = shapes::GATED.eps.to_bits();
    let launch_kv = || {
        kernel
            .launch(
                &session.stream,
                launch::config(plan)?,
                &[
                    KernelArgument::Buffer(&qkv),
                    KernelArgument::Buffer(&q_norm),
                    KernelArgument::Buffer(&k_norm),
                    KernelArgument::Buffer(&rope),
                    KernelArgument::Buffer(cos_sin),
                    KernelArgument::Buffer(&row_batch),
                    KernelArgument::Buffer(&position_buffer),
                    KernelArgument::Buffer(&live),
                    KernelArgument::Buffer(&table),
                    KernelArgument::Buffer(&pages.k),
                    KernelArgument::Buffer(&pages.v),
                    KernelArgument::Buffer(&q),
                    KernelArgument::U32(pages.table.len() as u32),
                    KernelArgument::U64(pages.stride as u64),
                    KernelArgument::U64(pages.stride as u64),
                    KernelArgument::U32(eps),
                    KernelArgument::U32(1.0_f32.to_bits()),
                    KernelArgument::U32(1.0_f32.to_bits()),
                ],
            )
            .map_err(err)
    };
    if poison {
        let map = kv_write_map(
            &pages.table,
            (pages.rank, pages.size),
            pages.stride,
            fp8,
            (chunk.first, n),
        )?;
        let k_initial = download(session, &pages.k, pages.k.byte_len())?;
        let v_initial = download(session, &pages.v, pages.v.byte_len())?;
        let replay = Replay::new(session)
            .output("q", &q, WriteMap::of(DType::Bf16).prefix(n * HEADS * DIM)?)
            .state("k_cache", &pages.k, k_initial, map.clone(), Vec::new())
            .state("v_cache", &pages.v, v_initial, map, Vec::new());
        let mut receipt = BTreeMap::from([
            ("entry".into(), json!(symbol)),
            ("rank".into(), json!(pages.rank)),
            ("size".into(), json!(pages.size)),
            ("rows".into(), json!(n)),
        ]);
        replay_all(session, replay, &live, launch_kv, &mut receipt)?;
        session.receipt("owned_writes", receipt)?;
    } else {
        launch_kv()?;
    }
    Ok(u16s(&download(session, &q, n * HEADS * DIM * 2)?))
}

/// Launches the indexer key write of `chunk`: the served entry (DCP 1) or
/// the owner-routed one, under the poison replays when `poison`.
#[allow(clippy::too_many_arguments)]
fn key_write(
    session: &Session,
    modules: &Modules,
    seq: &Sequence,
    chunk: &Chunk,
    pages: &Pages,
    cos_sin: &DeviceBuffer,
    norms: &DeviceBuffer,
    poison: bool,
) -> LabResult<()> {
    let n = chunk.n;
    let positions = (chunk.first..chunk.first + n)
        .map(|p| p as i32)
        .collect::<Vec<_>>();
    let rope = positions.iter().flat_map(|&p| [p; 3]).collect::<Vec<_>>();
    let projection = upload(
        session,
        &halves(&seq.projection[chunk.first * PROJECTION..(chunk.first + n) * PROJECTION]),
    )?;
    let rope = upload(session, &words(&rope))?;
    let row_batch = upload(session, &words(&vec![0; n]))?;
    let positions_buffer = upload(session, &words(&positions))?;
    let live = upload(session, &(n as u32).to_ne_bytes())?;
    let slots = upload(session, &words(&[0]))?;
    // The anchor applies to the launch that holds its last ring rows.
    let anchor = pages
        .anchor_end
        .filter(|&end| end >= chunk.first + RING && end <= chunk.first + n);
    let anchor_slots = upload(
        session,
        &words(&[anchor.map_or(-1, |_| ANCHOR_SLOTS as i32 - 1)]),
    )?;
    let anchor_ends = upload(session, &words(&[anchor.map_or(-1, |end| end as i32)]))?;
    let table = upload(session, &words(&pages.table))?;
    let owned = pages.size > 1;
    let eps = shapes::INDEXER.eps.to_bits();
    let (kernel, plan, symbol) = if owned {
        let entry = owned_entry("_index_key_write");
        entry
            .check_rank(pages.rank as u32, pages.size as u32)
            .map_err(|e| e.0.to_string())?;
        (
            function(&modules.owned, &entry.symbol(), 0)?,
            entry.plan(n as u32).map_err(|e| e.0.to_string())?,
            entry.symbol(),
        )
    } else {
        let entry = qsa_entry("index_key_write");
        (
            function(&modules.qsa, &entry.symbol(), entry.dynamic_shared_bytes())?,
            entry
                .plan(n as u32, seq.tokens as u32)
                .map_err(|e| e.0.to_string())?,
            entry.symbol(),
        )
    };
    let launch_keys = || {
        let mut arguments = vec![
            KernelArgument::Buffer(&projection),
            KernelArgument::Range {
                buffer: norms,
                offset: INDEX_DIM * 2,
                byte_len: INDEX_DIM * 2,
            },
            KernelArgument::Buffer(&rope),
            KernelArgument::Buffer(cos_sin),
            KernelArgument::Buffer(&row_batch),
            KernelArgument::Buffer(&positions_buffer),
            KernelArgument::Buffer(&live),
            KernelArgument::Buffer(&slots),
            KernelArgument::Buffer(&pages.ring_keys),
            KernelArgument::Buffer(&pages.ring_positions),
            KernelArgument::Buffer(&anchor_slots),
            KernelArgument::Buffer(&anchor_ends),
            KernelArgument::Buffer(&pages.anchor_keys),
            KernelArgument::Buffer(&pages.anchor_positions),
            KernelArgument::Buffer(&table),
            KernelArgument::Buffer(&pages.keys),
            KernelArgument::U32(pages.table.len() as u32),
            KernelArgument::U64(KEY_STRIDE as u64),
            KernelArgument::U32(eps),
        ];
        if owned {
            arguments.push(KernelArgument::U32(pages.rank as u32));
            arguments.push(KernelArgument::U32(pages.size as u32));
        }
        kernel
            .launch(&session.stream, launch::config(plan)?, &arguments)
            .map_err(err)
    };
    if poison {
        // Every block whose last token is in the launch is written once,
        // by its owner.
        let query = owned_entry("_index_key_write").query();
        let mut rows = Vec::new();
        let first_block = chunk.first.saturating_sub(BLOCK - 1).div_ceil(BLOCK);
        for b in (first_block..).take_while(|b| b * BLOCK + BLOCK - 1 < chunk.first + n) {
            if let Some((stripe, row)) =
                owned_block_row(query, pages.rank as u32, b as u64).map_err(|e| e.0.to_string())?
            {
                let at = pages.table[stripe as usize] as usize * KEY_STRIDE
                    + row as usize * INDEX_DIM * 2;
                rows.push(at..at + INDEX_DIM * 2);
            }
        }
        let keys_initial = download(session, &pages.keys, pages.keys.byte_len())?;
        let ring_initial = download(session, &pages.ring_keys, pages.ring_keys.byte_len())?;
        let rope_initial = download(
            session,
            &pages.ring_positions,
            pages.ring_positions.byte_len(),
        )?;
        let ring_bytes = ring_initial.len();
        let rope_bytes = rope_initial.len();
        // The anchored slot's rows of positions `end - RING..end`, written
        // without reading; every other anchor byte keeps its value.
        let anchored = anchor
            .map(|end| {
                (end - RING..end)
                    .map(|p| (ANCHOR_SLOTS - 1) * RING + p % RING)
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        let anchor_keys_map =
            WriteMap::of(DType::Bf16).rows(INDEX_DIM, anchored.iter().copied())?;
        let anchor_positions_map = WriteMap::of(DType::I32).rows(3, anchored.iter().copied())?;
        let anchor_keys_initial =
            download(session, &pages.anchor_keys, pages.anchor_keys.byte_len())?;
        let anchor_positions_initial = download(
            session,
            &pages.anchor_positions,
            pages.anchor_positions.byte_len(),
        )?;
        let replay = Replay::new(session)
            .state(
                "block_keys",
                &pages.keys,
                keys_initial,
                WriteMap::of(DType::Bf16).byte_ranges(rows)?,
                Vec::new(),
            )
            .state(
                "ring_keys",
                &pages.ring_keys,
                ring_initial,
                WriteMap::of(DType::Bf16).none()?,
                std::iter::once(0..ring_bytes).collect(),
            )
            .state(
                "ring_positions",
                &pages.ring_positions,
                rope_initial,
                WriteMap::of(DType::I32).none()?,
                std::iter::once(0..rope_bytes).collect(),
            )
            .state(
                "anchor_keys",
                &pages.anchor_keys,
                anchor_keys_initial,
                anchor_keys_map,
                Vec::new(),
            )
            .state(
                "anchor_positions",
                &pages.anchor_positions,
                anchor_positions_initial,
                anchor_positions_map,
                Vec::new(),
            );
        let mut receipt = BTreeMap::from([
            ("entry".into(), json!(symbol)),
            ("rank".into(), json!(pages.rank)),
            ("size".into(), json!(pages.size)),
            ("rows".into(), json!(n)),
            ("anchor_end".into(), json!(anchor)),
        ]);
        replay_all(session, replay, &live, launch_keys, &mut receipt)?;
        session.receipt("owned_writes", receipt)?;
        Ok(())
    } else {
        launch_keys()
    }
}

struct Modules {
    qsa: Module,
    gated: Module,
    owned: Module,
    merges: Module,
    localize: Module,
    head_sum: Module,
    partials: Module,
    partial_merge: Module,
}

impl Modules {
    fn load(session: &Session) -> LabResult<Modules> {
        Ok(Modules {
            qsa: session.module("sparse_attention_qwen38_sm120")?,
            gated: session.module("attention_qwen38_sm120")?,
            owned: session.module(shapes::CUBIN)?,
            merges: session.module(shapes::MERGE_CUBIN)?,
            localize: session.module(cp_shapes::CUBIN)?,
            head_sum: session.module(head_sum_shapes::CUBIN)?,
            partials: session.module(sp_shapes::CUBIN)?,
            partial_merge: session.module(merge_shapes::MERGE_CUBIN)?,
        })
    }
}

/// The history in chunks of `CHUNK` rows: blocks straddle launches.
fn chunks(tokens: usize) -> Vec<Chunk> {
    (0..tokens)
        .step_by(CHUNK)
        .map(|first| Chunk {
            first,
            n: CHUNK.min(tokens - first),
        })
        .collect()
}

/// Writes the whole sequence through `pages` for TP rank `source` (K/V
/// writes) and the indexer (key writes), and returns the Q rows of the
/// step `[rows][HEADS][DIM]`.
#[allow(clippy::too_many_arguments)]
fn write_sequence(
    session: &Session,
    modules: &Modules,
    seq: &Sequence,
    source: usize,
    pages: &Pages,
    fp8: bool,
    cos_sin: &DeviceBuffer,
    norms: &DeviceBuffer,
    keys: bool,
    poison: bool,
) -> LabResult<Vec<u16>> {
    let mut q = Vec::with_capacity(seq.tokens * HEADS * DIM);
    for chunk in chunks(seq.tokens) {
        q.extend(kv_write(
            session, modules, seq, source, &chunk, pages, fp8, cos_sin, poison,
        )?);
        if keys {
            key_write(session, modules, seq, &chunk, pages, cos_sin, norms, poison)?;
        }
    }
    let width = HEADS * DIM;
    Ok(q[seq.rows[0] * width..(seq.rows[seq.rows.len() - 1] + 1) * width].to_vec())
}

/// The served layer's bytes at DCP 1 against a rank's owned pages: every
/// owned K/V row and block key equal at its pages, and the rings equal.
fn check_owned(
    session: &Session,
    seq: &Sequence,
    full: &Pages,
    rank: &Pages,
    fp8: bool,
) -> LabResult<()> {
    let element = if fp8 { 1 } else { 2 };
    let row = DIM * element;
    for (name, a, b) in [("K", &full.k, &rank.k), ("V", &full.v, &rank.v)] {
        let served = download(session, a, a.byte_len())?;
        let owned = download(session, b, b.byte_len())?;
        for t in 0..seq.tokens {
            let Some((page, slot)) = rank.place(t) else {
                continue;
            };
            let (from, to) = (
                t / PAGE * full.stride + slot * row,
                page * rank.stride + slot * row,
            );
            if served[from..from + row] != owned[to..to + row] {
                return Err(format!("rank {}: {name} row {t} differs", rank.rank));
            }
        }
    }
    let served = download(session, &full.keys, full.keys.byte_len())?;
    let owned = download(session, &rank.keys, rank.keys.byte_len())?;
    let query = owned_entry("_index_key_write").query();
    for b in 0..seq.tokens / BLOCK {
        let Some((stripe, row)) =
            owned_block_row(query, rank.rank as u32, b as u64).map_err(|e| e.0.to_string())?
        else {
            continue;
        };
        let from = b / (PAGE / BLOCK) * KEY_STRIDE + b % (PAGE / BLOCK) * INDEX_DIM * 2;
        let to = rank.table[stripe as usize] as usize * KEY_STRIDE + row as usize * INDEX_DIM * 2;
        if served[from..from + INDEX_DIM * 2] != owned[to..to + INDEX_DIM * 2] {
            return Err(format!("rank {}: block key {b} differs", rank.rank));
        }
    }
    for (name, a, b) in [
        ("ring keys", &full.ring_keys, &rank.ring_keys),
        ("ring positions", &full.ring_positions, &rank.ring_positions),
        ("anchor keys", &full.anchor_keys, &rank.anchor_keys),
        (
            "anchor positions",
            &full.anchor_positions,
            &rank.anchor_positions,
        ),
    ] {
        if download(session, a, a.byte_len())? != download(session, b, b.byte_len())? {
            return Err(format!("rank {}: {name} differ", rank.rank));
        }
    }
    Ok(())
}

/// QSA's indexer queries of the step rows.
fn index_queries(
    session: &Session,
    modules: &Modules,
    seq: &Sequence,
    cos_sin: &DeviceBuffer,
    norms: &DeviceBuffer,
) -> LabResult<DeviceBuffer> {
    let n = seq.rows.len();
    let projection = seq
        .rows
        .iter()
        .flat_map(|&r| halves(&seq.projection[r * PROJECTION..(r + 1) * PROJECTION]))
        .collect::<Vec<_>>();
    let projection = upload(session, &projection)?;
    let rope = upload(
        session,
        &words(
            &seq.rows
                .iter()
                .flat_map(|&r| [r as i32; 3])
                .collect::<Vec<_>>(),
        ),
    )?;
    let live = upload(session, &(n as u32).to_ne_bytes())?;
    let q = session.allocate(n * INDEX_HEADS * INDEX_DIM * 2)?;
    let entry = qsa_entry("index_query");
    function(&modules.qsa, &entry.symbol(), entry.dynamic_shared_bytes())?
        .launch(
            &session.stream,
            launch::config(
                entry
                    .plan(n as u32, seq.tokens as u32)
                    .map_err(|e| e.0.to_string())?,
            )?,
            &[
                KernelArgument::Buffer(&projection),
                KernelArgument::Range {
                    buffer: norms,
                    offset: 0,
                    byte_len: INDEX_DIM * 2,
                },
                KernelArgument::Buffer(&rope),
                KernelArgument::Buffer(cos_sin),
                KernelArgument::Buffer(&live),
                KernelArgument::Buffer(&q),
                KernelArgument::U32(shapes::INDEXER.eps.to_bits()),
            ],
        )
        .map_err(err)?;
    Ok(q)
}

/// The step's device row inputs: one item at the step's positions.
struct StepRows {
    row_batch: DeviceBuffer,
    positions: DeviceBuffer,
    live: DeviceBuffer,
    n: usize,
}

impl StepRows {
    fn new(session: &Session, seq: &Sequence) -> LabResult<StepRows> {
        let n = seq.rows.len();
        Ok(StepRows {
            row_batch: upload(session, &words(&vec![0; n]))?,
            positions: upload(
                session,
                &words(&seq.rows.iter().map(|&r| r as i32).collect::<Vec<_>>()),
            )?,
            live: upload(session, &(n as u32).to_ne_bytes())?,
            n,
        })
    }
}

/// QSA's served selection of the step over the DCP 1 pages: `[rows][TOP]`
/// block lists, -1 past each row's count.
fn qsa_selection(
    session: &Session,
    modules: &Modules,
    seq: &Sequence,
    full: &Pages,
    q_index: &DeviceBuffer,
    step: &StepRows,
) -> LabResult<Vec<i32>> {
    let geometry = model_qwen38_contract::sm120::attention::SPARSE;
    let context = seq.tokens as u32;
    let stride = geometry.score_stride(context) as usize;
    let scores = upload(session, &vec![0xff; step.n * stride * 4])?;
    let selected = upload(session, &vec![0xff; step.n * TOP * 4])?;
    let table = upload(session, &words(&full.table))?;
    let scorer = qsa_entry(&format!("block_scores_{}", seq.phase));
    function(
        &modules.qsa,
        &scorer.symbol(),
        scorer.dynamic_shared_bytes(),
    )?
    .launch(
        &session.stream,
        launch::config(
            scorer
                .plan(step.n as u32, context)
                .map_err(|e| e.0.to_string())?,
        )?,
        &[
            KernelArgument::Buffer(q_index),
            KernelArgument::Buffer(&full.keys),
            KernelArgument::Buffer(&table),
            KernelArgument::Buffer(&step.row_batch),
            KernelArgument::Buffer(&step.positions),
            KernelArgument::Buffer(&step.live),
            KernelArgument::Buffer(&scores),
            KernelArgument::U32(full.table.len() as u32),
            KernelArgument::U64(KEY_STRIDE as u64),
            KernelArgument::U32(stride as u32),
        ],
    )
    .map_err(err)?;
    let topk = qsa_entry("block_topk");
    function(&modules.qsa, &topk.symbol(), topk.dynamic_shared_bytes())?
        .launch(
            &session.stream,
            launch::config(
                topk.plan(step.n as u32, context)
                    .map_err(|e| e.0.to_string())?,
            )?,
            &[
                KernelArgument::Buffer(&scores),
                KernelArgument::Buffer(&step.positions),
                KernelArgument::Buffer(&step.live),
                KernelArgument::Buffer(&selected),
                KernelArgument::U32(stride as u32),
            ],
        )
        .map_err(err)?;
    let all = ints(&download(session, &selected, step.n * TOP * 4)?);
    let mut lists = vec![-1; step.n * TOP];
    for (i, &p) in seq.rows.iter().enumerate() {
        let count = TOP.min((p + 1) / BLOCK);
        lists[i * TOP..][..count].copy_from_slice(&all[i * TOP..][..count]);
    }
    Ok(lists)
}

/// The head-sum per-rank selection of the step over a rank's block keys:
/// its (score, block) pair bytes `[rows][TOP][2]`.
fn per_rank_pairs(
    session: &Session,
    modules: &Modules,
    pages: &Pages,
    q_index: &DeviceBuffer,
    step: &StepRows,
) -> LabResult<Vec<u8>> {
    let entry = head_sum_shapes::lab_entries()
        .into_iter()
        .find(|e| e.name.ends_with("_select_per_rank2"))
        .ok_or("no head-sum per-rank selection")?;
    entry
        .check_context_parallel(pages.rank as u32, pages.size as u32)
        .map_err(|e| e.0.to_string())?;
    let n = step.n as u32;
    let plan = entry.plan(n).map_err(|e| e.0.to_string())?;
    let merge_plan = entry
        .merge_plan(n)
        .map_err(|e| e.0.to_string())?
        .ok_or("a selection has a merge")?;
    let splits = plan.grid[1];
    let workspace = session.allocate((plan.workspace_bytes as usize).max(16))?;
    let pairs = session.allocate(step.n * TOP * 8)?;
    let table = upload(session, &words(&pages.table))?;
    function(
        &modules.head_sum,
        &entry.symbol(),
        plan.dynamic_shared_bytes,
    )?
    .launch(
        &session.stream,
        launch::config(plan)?,
        &[
            KernelArgument::Buffer(q_index),
            KernelArgument::Buffer(&pages.keys),
            KernelArgument::Buffer(&table),
            KernelArgument::Buffer(&step.row_batch),
            KernelArgument::Buffer(&step.positions),
            KernelArgument::Buffer(&step.live),
            KernelArgument::Buffer(&workspace),
            KernelArgument::U32(pages.table.len() as u32),
            KernelArgument::U64(KEY_STRIDE as u64),
            KernelArgument::U32(2048),
            KernelArgument::U32(splits),
            KernelArgument::U32(pages.rank as u32),
            KernelArgument::U32(pages.size as u32),
        ],
    )
    .map_err(err)?;
    function(&modules.head_sum, &entry.merge_symbol().unwrap(), 0)?
        .launch(
            &session.stream,
            launch::config(merge_plan)?,
            &[
                KernelArgument::Buffer(&workspace),
                KernelArgument::Buffer(&step.positions),
                KernelArgument::Buffer(&step.live),
                KernelArgument::Buffer(&pairs),
                KernelArgument::U32(2048),
                KernelArgument::U32(splits),
            ],
        )
        .map_err(err)?;
    download(session, &pairs, step.n * TOP * 8)
}

/// `topk_merge_blocks{N}` over the ranks' pair bytes (gathered rank-major),
/// under the poison replays, checked against kernel-core's
/// `merge_blocks` of the same pairs.
fn merge_pairs(
    session: &Session,
    modules: &Modules,
    ranks: &[Vec<u8>],
    rows: usize,
    label: &str,
) -> LabResult<Vec<i32>> {
    let entry = shapes::merge_entries()
        .into_iter()
        .find(|e| e.ranks == ranks.len() as u32)
        .ok_or("no lab block merge")?;
    let pairs = upload(session, &ranks.concat())?;
    let live = upload(session, &(rows as u32).to_ne_bytes())?;
    let blocks = session.allocate(rows * TOP * 4)?;
    let plan = entry.plan(rows as u32).map_err(|e| e.0.to_string())?;
    let kernel = function(&modules.merges, &entry.symbol(), 0)?;
    let replay = Replay::new(session).output_with(
        "blocks",
        &blocks,
        WriteMap::of(DType::I32).prefix(rows * TOP)?,
        &[0xa5, 0x5a, 0xc3, 0x3c],
    );
    let mut receipt = BTreeMap::from([
        ("entry".into(), json!(entry.name)),
        ("case".into(), json!(label)),
        ("rows".into(), json!(rows)),
    ]);
    replay_all(
        session,
        replay,
        &live,
        || {
            kernel
                .launch(
                    &session.stream,
                    launch::config(plan)?,
                    &[
                        KernelArgument::Buffer(&pairs),
                        KernelArgument::Buffer(&live),
                        KernelArgument::Buffer(&blocks),
                        KernelArgument::U32(2048),
                        KernelArgument::U32(rows as u32),
                    ],
                )
                .map_err(err)
        },
        &mut receipt,
    )?;
    session.receipt("block_merge", receipt)?;
    let device = ints(&download(session, &blocks, rows * TOP * 4)?);
    let decoded = ranks
        .iter()
        .map(|bytes| {
            ints(bytes)
                .chunks_exact(2)
                .map(|p| (f32::from_bits(p[0] as u32), p[1]))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    for row in 0..rows {
        let lists = decoded
            .iter()
            .map(|pairs| &pairs[row * TOP..][..TOP])
            .collect::<Vec<_>>();
        let expected = merge_blocks(entry.query(), 2048, &lists).map_err(|e| e.0.to_string())?;
        if device[row * TOP..][..TOP] != expected[..] {
            return Err(format!(
                "{label} row {row}: the block merge differs from the oracle"
            ));
        }
    }
    Ok(device)
}

/// QSA's served selected attention of TP rank `q`'s step rows over the DCP 1
/// pages: BF16 `[rows][HEADS][DIM]` as F32.
#[allow(clippy::too_many_arguments)]
fn qsa_attention(
    session: &Session,
    modules: &Modules,
    seq: &Sequence,
    full: &Pages,
    q: &[u16],
    selected: &[i32],
    step: &StepRows,
    fp8: bool,
) -> LabResult<Vec<f32>> {
    let entry = qsa_entry(&format!(
        "attn_{}_{}",
        if fp8 { "fp8" } else { "bf16" },
        seq.phase
    ));
    let plan = entry
        .plan(step.n as u32, seq.tokens as u32)
        .map_err(|e| e.0.to_string())?;
    let splits = plan.grid[2];
    let q = upload(session, &halves(q))?;
    let selected = upload(session, &words(selected))?;
    let table = upload(session, &words(&full.table))?;
    let workspace = session.allocate((plan.workspace_bytes as usize).max(16))?;
    let out = session.allocate(step.n * HEADS * DIM * 2)?;
    function(&modules.qsa, &entry.symbol(), entry.dynamic_shared_bytes())?
        .launch(
            &session.stream,
            launch::config(plan)?,
            &[
                KernelArgument::Buffer(&q),
                KernelArgument::Buffer(&full.k),
                KernelArgument::Buffer(&full.v),
                KernelArgument::Buffer(&table),
                KernelArgument::Buffer(&step.row_batch),
                KernelArgument::Buffer(&step.positions),
                KernelArgument::Buffer(&step.live),
                KernelArgument::Buffer(&selected),
                KernelArgument::Buffer(&workspace),
                KernelArgument::Buffer(&out),
                KernelArgument::U32(full.table.len() as u32),
                KernelArgument::U64(full.stride as u64),
                KernelArgument::U64(full.stride as u64),
                KernelArgument::U32(splits),
                KernelArgument::U32(SOFTMAX_SCALE.to_bits()),
                KernelArgument::U32(1.0_f32.to_bits()),
                KernelArgument::U32(1.0_f32.to_bits()),
            ],
        )
        .map_err(err)?;
    if let Some(merge_plan) = entry
        .merge_plan(step.n as u32)
        .map_err(|e| e.0.to_string())?
    {
        let symbol = entry.merge_symbol().ok_or("an attention entry")?;
        function(&modules.qsa, &symbol, 0)?
            .launch(
                &session.stream,
                launch::config(merge_plan)?,
                &[
                    KernelArgument::Buffer(&workspace),
                    KernelArgument::Buffer(&step.live),
                    KernelArgument::Buffer(&out),
                    KernelArgument::U32(splits),
                ],
            )
            .map_err(err)?;
    }
    Ok(bf16_values(&download(
        session,
        &out,
        step.n * HEADS * DIM * 2,
    )?))
}

/// A rank's pages as the selected partials' host cache: its K and V bytes
/// and its table.
fn rank_cache(session: &Session, pages: &Pages) -> LabResult<RankCache> {
    Ok(RankCache {
        k: download(session, &pages.k, pages.k.byte_len())?,
        v: download(session, &pages.v, pages.v.byte_len())?,
        stride: pages.stride,
        max_pages: pages.table.len(),
        tables: vec![pages.table.iter().map(|&p| p as usize).collect()],
        keys: Vec::new(),
        values: Vec::new(),
    })
}

/// The served DCP 1 cache decoded for the f64 attention: one item.
fn decoded_kv(session: &Session, seq: &Sequence, full: &Pages, fp8: bool) -> LabResult<Kv> {
    let table = e4m3_table();
    let element = if fp8 { 1 } else { 2 };
    let row = DIM * element;
    let planes = [
        download(session, &full.k, full.k.byte_len())?,
        download(session, &full.v, full.v.byte_len())?,
    ];
    let [(stored_k, keys), (stored_v, values)] = planes.map(|plane| {
        (0..seq.tokens)
            .map(|t| {
                let at = t / PAGE * full.stride + t % PAGE * row;
                let bytes = plane[at..at + row].to_vec();
                let decoded = if fp8 {
                    bytes.iter().map(|&b| table[b as usize]).collect()
                } else {
                    u16s(&bytes)
                        .iter()
                        .map(|&b| f64::from(bf16_to_f32(b)))
                        .collect()
                };
                (bytes, decoded)
            })
            .unzip::<_, _, Vec<_>, Vec<_>>()
    });
    Ok(Kv {
        fp8,
        stored_k: vec![stored_k],
        stored_v: vec![stored_v],
        keys: vec![keys],
        values: vec![values],
    })
}

/// The paired difference of two BF16 outputs: a diagnostic, not gated.
fn paired(merged: &[f32], served: &[f32]) -> Value {
    let ordered = |x: f32| {
        let bits = (x.to_bits() >> 16) as i32;
        if bits & 0x8000 != 0 {
            -(bits & 0x7fff)
        } else {
            bits
        }
    };
    let width = HEADS * DIM;
    let mut equal_rows = 0;
    let (mut differing, mut largest, mut ulps) = (0_usize, 0_f64, 0_u32);
    for (m, s) in merged.chunks_exact(width).zip(served.chunks_exact(width)) {
        let mut equal = true;
        for (&a, &b) in m.iter().zip(s) {
            if a.to_bits() == b.to_bits() {
                continue;
            }
            equal = false;
            differing += 1;
            largest = largest.max((f64::from(a) - f64::from(b)).abs());
            ulps = ulps.max(ordered(a).abs_diff(ordered(b)));
        }
        equal_rows += usize::from(equal);
    }
    json!({
        "bitwise_equal_rows": equal_rows,
        "elements": merged.len(),
        "differing_elements": differing,
        "max_abs_difference": largest,
        "max_bf16_ulps": ulps,
    })
}

/// The whole layer at DCP 1 (served) and DCP 2 (owner-routed writes,
/// per-rank selection, block merge, partials, merge) on `seq`.
fn chain(
    session: &Session,
    modules: &Modules,
    seq: &Sequence,
    fp8: bool,
    label: &str,
) -> LabResult<Value> {
    let cos_sin = upload(session, &floats_bytes(&cos_sin_table(seq.tokens)))?;
    let norms = upload(session, &halves(&seq.index_norms))?;
    // DCP 1: one set of pages holds every key; both TP ranks of the replica
    // set write the same K and V.
    let full = Pages::new(session, seq, fp8, 0, 1)?;
    let mut served_q = Vec::new();
    for source in 0..GROUP {
        served_q.push(write_sequence(
            session,
            modules,
            seq,
            source,
            &full,
            fp8,
            &cos_sin,
            &norms,
            source == 0,
            false,
        )?);
    }
    let q_index = index_queries(session, modules, seq, &cos_sin, &norms)?;
    let step = StepRows::new(session, seq)?;
    let selected = qsa_selection(session, modules, seq, &full, &q_index, &step)?;
    let served = (0..GROUP)
        .map(|s| {
            qsa_attention(
                session,
                modules,
                seq,
                &full,
                &served_q[s],
                &selected,
                &step,
                fp8,
            )
        })
        .collect::<LabResult<Vec<_>>>()?;
    // DCP 2: group rank d is TP rank d, with its own pages.
    let ranks = (0..GROUP)
        .map(|d| Pages::new(session, seq, fp8, d, GROUP))
        .collect::<LabResult<Vec<_>>>()?;
    let mut owned_q = Vec::new();
    for (d, pages) in ranks.iter().enumerate() {
        owned_q.push(write_sequence(
            session, modules, seq, d, pages, fp8, &cos_sin, &norms, true, false,
        )?);
        check_owned(session, seq, &full, pages, fp8)?;
        if owned_q[d] != served_q[d] {
            return Err(format!(
                "{label}: rank {d}'s Q rows differ from the served write"
            ));
        }
    }
    let pairs = ranks
        .iter()
        .map(|pages| per_rank_pairs(session, modules, pages, &q_index, &step))
        .collect::<LabResult<Vec<_>>>()?;
    let blocks = merge_pairs(session, modules, &pairs, step.n, label)?;
    for (i, &p) in seq.rows.iter().enumerate() {
        let count = TOP.min((p + 1) / BLOCK);
        if blocks[i * TOP..][..count] != selected[i * TOP..][..count] {
            return Err(format!(
                "{label} row {p}: the DCP 2 block list differs from QSA's selection"
            ));
        }
    }
    // Partials of both ranks' queries over each rank's pages, gathered to
    // the queries' owners and merged.
    let host_step = Step {
        lengths: vec![seq.tokens],
        rows: seq.rows.iter().map(|&p| (0, p)).collect(),
    };
    let entry = sp_shapes::lab_entries()
        .into_iter()
        .find(|e| {
            e.name
                == format!(
                    "{}_{}_{}",
                    sp_shapes::PROFILE.tag(),
                    if fp8 { "fp8" } else { "bf16" },
                    seq.phase
                )
        })
        .ok_or("no lab selected partial entry")?;
    let partial = modules.partials.function(&entry.symbol()).map_err(err)?;
    let caches = ranks
        .iter()
        .map(|pages| rank_cache(session, pages))
        .collect::<LabResult<Vec<_>>>()?;
    let mut parts: Vec<Vec<Partials>> = (0..GROUP).map(|_| Vec::new()).collect();
    for (d, cache) in caches.iter().enumerate() {
        for (s, q) in owned_q.iter().enumerate() {
            let operands = Launch {
                step: &host_step,
                q,
                selected: &blocks,
                cache,
                scales: (SOFTMAX_SCALE, 1.0, 1.0),
                rank: d,
                size: GROUP,
                prepared: step.n,
            };
            parts[s].push(run(session, &partial, &entry, &operands, false, label)?);
        }
    }
    let kv = decoded_kv(session, seq, &full, fp8)?;
    let merge_entries = merge_shapes::merge_entries();
    let mut diagnostics = Vec::new();
    for s in 0..GROUP {
        let reference = full_reference(
            &host_step,
            &served_q[s],
            &selected,
            &kv,
            (SOFTMAX_SCALE, 1.0, 1.0),
        )?;
        gate(
            &format!("{label} QSA attention of rank {s}"),
            &served[s].iter().map(|&x| f64::from(x)).collect::<Vec<_>>(),
            &reference,
            DType::Bf16,
        )?;
        let merged = merge(
            session,
            &modules.partial_merge,
            &merge_entries,
            &parts[s],
            step.n,
        )?;
        gate_merged(
            &format!("{label} DCP 2 attention of rank {s}"),
            &merged,
            &parts[s],
            &reference,
        )?;
        diagnostics.push(paired(&merged, &served[s]));
    }
    Ok(json!({
        "tokens": seq.tokens,
        "rows": step.n,
        "phase": seq.phase,
        "kv": if fp8 { "fp8" } else { "bf16" },
        "owned_writes_equal_served": true,
        "selection_equals_qsa": true,
        "both_pass_oracle_gates": true,
        "paired_difference_diagnostic": diagnostics,
    }))
}

fn record(session: &Session, case: &str, result: &Value) -> LabResult<()> {
    eprintln!("{case}: {result}");
    session.receipt(
        "qsa_context_parallel",
        BTreeMap::from([
            ("case".into(), json!(case)),
            ("result".into(), result.clone()),
            ("correct".into(), json!(true)),
        ]),
    )?;
    Ok(())
}

/// The K/V write map counts whole cache elements: a BF16 row whose values
/// carry a 0xff byte (0x3eff, 0xff3e) passes as written, an all-0xff BF16
/// element or FP8 code (NaN, never written) fails, and the ranges are the
/// rank's owned rows only. GPU-free.
#[test]
fn kv_write_maps_count_whole_cache_elements() {
    use kernel_lab::poison::{Fill, POISON, check};
    let table = [1, 0];
    let stride = PAGE * DIM * 2 + DECOY;
    // Rank 0 of 2 owns positions 0..64 (local page 1) and 128..192 (page 0).
    let map = kv_write_map(&table, (0, 2), stride, false, (60, 80)).unwrap();
    let owned = (60..140).filter(|t| t / PAGE % 2 == 0).collect::<Vec<_>>();
    assert_eq!(map.written_bytes(), owned.len() * DIM * 2);
    let mut image = vec![POISON; 2 * stride];
    for &t in &owned {
        let (page, slot) = place(&table, 0, 2, t).unwrap();
        let at = page * stride + slot * DIM * 2;
        for (i, pair) in image[at..at + DIM * 2].chunks_exact_mut(2).enumerate() {
            let value: u16 = if i % 2 == 0 { 0x3eff } else { 0xff3e };
            pair.copy_from_slice(&value.to_ne_bytes());
        }
    }
    check("bf16 row", &image, &map, Fill::Pattern(&[POISON])).unwrap();
    // The same image read byte by byte would claim the 0xff bytes unwritten.
    let bytewise = WriteMap::of(DType::U8)
        .byte_ranges(map.ranges().to_vec())
        .unwrap();
    assert!(check("bytes", &image, &bytewise, Fill::Pattern(&[POISON])).is_err());
    // A NaN element left in an owned row is a miss.
    let (page, slot) = place(&table, 0, 2, owned[3]).unwrap();
    let at = page * stride + slot * DIM * 2 + 8;
    image[at..at + 2].fill(POISON);
    assert!(check("missed", &image, &map, Fill::Pattern(&[POISON])).is_err());
    // FP8: one-byte elements, the other rank's rows only.
    let fp8_stride = PAGE * DIM + DECOY;
    let fp8 = kv_write_map(&table, (1, 2), fp8_stride, true, (60, 80)).unwrap();
    assert_eq!(fp8.element_bytes(), 1);
    assert_eq!(fp8.written_bytes(), (64..128).count() * DIM);
}

/// The owner-routed writes alone, under the poison replays, against the
/// served writes: Q, owned K/V rows, owned block keys, the rings and the
/// anchor rings equal, over a history written in launches whose blocks
/// straddle them, with the sequence anchored inside the second launch.
#[test]
fn owned_writes_equal_the_served_writes() {
    gpu_test(4 << 30, |session| {
        check_geometry()?;
        let modules = Modules::load(session)?;
        let mut rng = Rng(0x0e1e_c7ed);
        for fp8 in [false, true] {
            let seq = Sequence::synthetic(CHUNK + 1301, 4, "decode", &mut rng);
            let cos_sin = upload(session, &floats_bytes(&cos_sin_table(seq.tokens)))?;
            let norms = upload(session, &halves(&seq.index_norms))?;
            // Rows `end - RING..end` lie inside the second launch.
            let anchor_end = CHUNK + 37;
            let full = Pages::new(session, &seq, fp8, 0, 1)?.anchored(anchor_end);
            let served = write_sequence(
                session, &modules, &seq, 1, &full, fp8, &cos_sin, &norms, true, false,
            )?;
            let anchor = download(session, &full.anchor_positions, ANCHOR_SLOTS * RING * 12)?;
            let written = ints(&anchor[RING * 12..])
                .chunks_exact(3)
                .map(|p| p[0])
                .collect::<Vec<_>>();
            let mut expected = (anchor_end - RING..anchor_end)
                .map(|p| p as i32)
                .collect::<Vec<_>>();
            expected.rotate_left((RING - anchor_end % RING) % RING);
            if written != expected || anchor[..RING * 12].iter().any(|&b| b != 0xff) {
                return Err("the served writer's anchor ring is not the anchored rows".into());
            }
            for rank in 0..GROUP {
                let pages = Pages::new(session, &seq, fp8, rank, GROUP)?.anchored(anchor_end);
                let owned = write_sequence(
                    session, &modules, &seq, 1, &pages, fp8, &cos_sin, &norms, true, true,
                )?;
                check_owned(session, &seq, &full, &pages, fp8)?;
                if owned != served {
                    return Err(format!("rank {rank}: the Q rows differ"));
                }
            }
        }
        Ok(())
    })
    .expect("owner-routed writes equal the served writes");
}

/// The layer at DCP 2 against QSA's served DCP 1 layer on synthetic
/// operands: decode rows of a 9000-token sequence and prefill rows across
/// the 2051-token dense limit, BF16 and FP8 KV.
#[test]
fn dcp2_chain_matches_qsa_on_synthetic_operands() {
    gpu_test(8 << 30, |session| {
        check_geometry()?;
        let modules = Modules::load(session)?;
        let mut rng = Rng(0xdc92_c4a1);
        for (tokens, rows, phase) in [
            (9000, 4, "decode"),
            (2100, 64, "prefill"),
            (1500, 3, "decode"),
        ] {
            let seq = Sequence::synthetic(tokens, rows, phase, &mut rng);
            for fp8 in [false, true] {
                let case = format!(
                    "synthetic {tokens} tokens {phase} {}",
                    if fp8 { "fp8" } else { "bf16" }
                );
                let result = chain(session, &modules, &seq, fp8, &case)?;
                record(session, &case, &result)?;
            }
        }
        Ok(())
    })
    .expect("the DCP 2 chain matches QSA on synthetic operands");
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

/// A layer's indexer projection rows and norms (query then key), from an
/// HF-table export.
fn read_indexer(dir: &Path, layer: u64) -> LabResult<(Vec<u16>, Vec<u16>)> {
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
            "{} layer {layer}: the operands use table {}",
            dir.display(),
            manifest["table"]
        ));
    }
    let (qk, norms) = (bits("qk.bf16")?, bits("norm.bf16")?);
    if qk.is_empty() || qk.len() % PROJECTION != 0 || norms.len() != 2 * INDEX_DIM {
        return Err(format!("{} layer {layer}: shapes disagree", dir.display()));
    }
    Ok((qk, norms))
}

/// The layer at DCP 2 against QSA's served DCP 1 layer with captured
/// indexer operands (`CHOPINFER_QSA_OPERANDS`) of contexts up to
/// `CAPTURED_TOKENS`: the last four rows of each layer as a decode step, with
/// synthetic attention operands. With `CHOPINFER_QSA_OPERANDS_OUT` the
/// summary is written to that new file.
#[test]
fn dcp2_chain_matches_qsa_on_captured_indexer_operands() {
    let Some(root) = std::env::var_os("CHOPINFER_QSA_OPERANDS").map(PathBuf::from) else {
        kernel_lab::skip("the captured DCP 2 chain: CHOPINFER_QSA_OPERANDS is unset")
            .expect("the DCP 2 chain on captured operands");
        return;
    };
    let out = std::env::var_os("CHOPINFER_QSA_OPERANDS_OUT").map(PathBuf::from);
    gpu_test(8 << 30, |session| {
        check_geometry()?;
        let modules = Modules::load(session)?;
        let mut rng = Rng(0xca97_dc92);
        let mut summary = Vec::new();
        let mut skipped = 0;
        for dir in export_dirs(&root) {
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
            for layer in layers {
                let (projection, norms) = read_indexer(&dir, layer)?;
                let tokens = projection.len() / PROJECTION;
                if tokens > CAPTURED_TOKENS {
                    skipped += 1;
                    continue;
                }
                let seq = Sequence::with_indexer(tokens, 4, "decode", projection, norms, &mut rng);
                for fp8 in [false, true] {
                    let case = format!(
                        "{} layer {layer} {}",
                        dir.display(),
                        if fp8 { "fp8" } else { "bf16" }
                    );
                    let mut result = chain(session, &modules, &seq, fp8, &case)?;
                    record(session, &case, &result)?;
                    result["case"] = json!(case);
                    summary.push(result);
                }
            }
        }
        if summary.is_empty() {
            return Err(format!(
                "{}: no layer of at most {CAPTURED_TOKENS} tokens",
                root.display()
            ));
        }
        eprintln!(
            "{} layer sets checked, {skipped} longer ones skipped",
            summary.len() / 2
        );
        if let Some(out) = &out {
            use std::io::Write;
            let value = json!({
                "operands": root.display().to_string(),
                "skipped_longer_layers": skipped,
                "layers": summary,
            });
            fs::File::create_new(out)
                .and_then(|mut f| f.write_all(&serde_json::to_vec_pretty(&value).expect("json")))
                .map_err(|e| format!("{}: {e}", out.display()))?;
        }
        Ok(())
    })
    .expect("the DCP 2 chain on captured indexer operands");
}
