//! Shared fixtures of the selected attention partial tests: KV caches of
//! BF16 or FP8 rows placed on group ranks' shuffled pages, selected block
//! lists, the partial launches under the poison replays, the f64 references
//! and gates, and the partial merge of six heads of 256.
#![allow(dead_code)] // Shared by test binaries that use different parts.

#[path = "support.rs"]
mod support;

use std::collections::BTreeMap;

use kernel_core::codec::{bf16_to_f32, f32_to_bf16};
use kernel_core::context_parallel_oracle::{PagedPartialsInput, selected_attention_partials};
use kernel_core::contract::{AttentionPhase, DType, SparseAttentionQuery};
use kernel_core::latent_attention_oracle::LatentPartials;
use kernel_core::oracle::{OracleOutput, e4m3_table, numeric_bound};
use kernel_core::sparse_attention_oracle::{SparseKv, sparse_attention};
use kernel_lab::poison::{Replay, WriteMap, replay_all};
use kernel_lab::{LabResult, Session, launch};
use kernels_sm120::context_parallel::{self, Kind, MergeLayout};
use kernels_sm120::selected_partials::Entry;
use runtime_cuda::{Function, KernelArgument, Module};
use serde_json::json;
pub use support::{Rng, download, upload};

pub const Q_HEADS: usize = 6;
pub const DIM: usize = 256;
pub const PAGE: usize = 64;
pub const BLOCK: usize = 4;
pub const TOP: usize = 512;
/// Bytes of decoy between pages: BF16 3.3e38 or FP8 448, which any read
/// outside a page turns into a failing score.
pub const DECOY: usize = 512;

pub fn err(e: impl ToString) -> String {
    e.to_string()
}

pub fn floats(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

pub fn bf16_values(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|c| bf16_to_f32(u16::from_ne_bytes([c[0], c[1]])))
        .collect()
}

pub fn words(values: &[i32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_ne_bytes()).collect()
}

/// A step's items and rows.
pub struct Step {
    /// Tokens of each item's sequence.
    pub lengths: Vec<usize>,
    /// Each row's item and position.
    pub rows: Vec<(usize, usize)>,
}

impl Step {
    /// Decode: each item's last token.
    pub fn decode(lengths: &[usize]) -> Step {
        Step {
            lengths: lengths.to_vec(),
            rows: lengths
                .iter()
                .enumerate()
                .map(|(i, &n)| (i, n - 1))
                .collect(),
        }
    }

    /// Verify: each item's last `k` tokens.
    pub fn verify(lengths: &[usize], k: usize) -> Step {
        Step {
            lengths: lengths.to_vec(),
            rows: lengths
                .iter()
                .enumerate()
                .flat_map(|(i, &n)| (n - k..n).map(move |p| (i, p)))
                .collect(),
        }
    }

    /// Prefill: one item's `rows` tokens after a cached `prefix`.
    pub fn prefill(prefix: usize, rows: usize) -> Step {
        Step {
            lengths: vec![prefix + rows],
            rows: (prefix..prefix + rows).map(|p| (0, p)).collect(),
        }
    }
}

/// Selected block lists `[rows, TOP]`: a row with at most `TOP` complete
/// blocks lists them all; a longer row lists `TOP` distinct random blocks
/// ascending, always with its first and last complete block. Unused entries
/// are -1.
pub fn selections(step: &Step, rng: &mut Rng) -> Vec<i32> {
    let mut lists = vec![-1; step.rows.len() * TOP];
    for (row, &(_, p)) in step.rows.iter().enumerate() {
        let complete = (p + 1) / BLOCK;
        let list = &mut lists[row * TOP..][..TOP];
        if complete <= TOP {
            for (b, entry) in list.iter_mut().take(complete).enumerate() {
                *entry = b as i32;
            }
            continue;
        }
        let mut blocks = (1..complete - 1).collect::<Vec<_>>();
        for i in 0..TOP - 2 {
            let j = i + (rng.next() % (blocks.len() - i) as u64) as usize;
            blocks.swap(i, j);
        }
        let mut chosen = blocks[..TOP - 2].to_vec();
        chosen.extend([0, complete - 1]);
        chosen.sort_unstable();
        for (entry, &b) in list.iter_mut().zip(&chosen) {
            *entry = b as i32;
        }
    }
    lists
}

/// The sequences' K and V rows, as stored (BF16 bits or FP8 bytes), and
/// decoded to f64 before their layer scale: `[item][token][dim]`.
pub struct Kv {
    pub fp8: bool,
    pub stored_k: Vec<Vec<Vec<u8>>>,
    pub stored_v: Vec<Vec<Vec<u8>>>,
    pub keys: Vec<Vec<Vec<f64>>>,
    pub values: Vec<Vec<Vec<f64>>>,
}

impl Kv {
    pub fn new(lengths: &[usize], fp8: bool, wide: bool, rng: &mut Rng) -> Kv {
        let table = e4m3_table();
        let row = |rng: &mut Rng| -> (Vec<u8>, Vec<f64>) {
            if fp8 {
                // Every finite E4M3 byte (0x7f and 0xff are NaN).
                let bytes = (0..DIM)
                    .map(|_| {
                        loop {
                            let b = (rng.next() >> 32) as u8;
                            if b & 0x7f != 0x7f {
                                break b;
                            }
                        }
                    })
                    .collect::<Vec<_>>();
                let decoded = bytes.iter().map(|&b| table[b as usize]).collect();
                (bytes, decoded)
            } else {
                let bits = (0..DIM)
                    .map(|_| f32_to_bf16(if wide { rng.wide() } else { rng.unit() }))
                    .collect::<Vec<_>>();
                let decoded = bits.iter().map(|&b| f64::from(bf16_to_f32(b))).collect();
                (bits.iter().flat_map(|b| b.to_ne_bytes()).collect(), decoded)
            }
        };
        let (mut stored_k, mut stored_v, mut keys, mut values) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for &n in lengths {
            let (mut sk, mut sv, mut k, mut v) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
            for _ in 0..n {
                let (bytes, decoded) = row(rng);
                sk.push(bytes);
                k.push(decoded);
                let (bytes, decoded) = row(rng);
                sv.push(bytes);
                v.push(decoded);
            }
            stored_k.push(sk);
            stored_v.push(sv);
            keys.push(k);
            values.push(v);
        }
        Kv {
            fp8,
            stored_k,
            stored_v,
            keys,
            values,
        }
    }

    pub fn element_bytes(&self) -> usize {
        if self.fp8 { 1 } else { 2 }
    }
}

/// One group rank's cache: its own pages of every item, shuffled, each
/// followed by decoy bytes; its block tables of local pages; and, when
/// decoded, the same pages for the oracle.
pub struct RankCache {
    pub k: Vec<u8>,
    pub v: Vec<u8>,
    pub stride: usize,
    pub max_pages: usize,
    pub tables: Vec<Vec<usize>>,
    pub keys: Vec<f64>,
    pub values: Vec<f64>,
}

impl RankCache {
    /// The block tables `[items, max_pages]`, -1 past each item's pages.
    pub fn table_words(&self) -> Vec<i32> {
        let mut tables = vec![-1_i32; self.tables.len() * self.max_pages];
        for (item, table) in self.tables.iter().enumerate() {
            for (local, &slot) in table.iter().enumerate() {
                tables[item * self.max_pages + local] = slot as i32;
            }
        }
        tables
    }
}

pub fn rank_cache(kv: &Kv, rank: usize, size: usize, decoded: bool, rng: &mut Rng) -> RankCache {
    let page_bytes = PAGE * DIM * kv.element_bytes();
    let stride = page_bytes + DECOY;
    let owned = kv
        .stored_k
        .iter()
        .map(|tokens| {
            (0..tokens.len().div_ceil(PAGE))
                .filter(|j| j % size == rank)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let count = owned.iter().map(Vec::len).sum::<usize>();
    let mut slots = (0..count).collect::<Vec<_>>();
    for i in (1..slots.len()).rev() {
        slots.swap(i, (rng.next() % (i as u64 + 1)) as usize);
    }
    let pool = count.max(1);
    let decoy_byte = if kv.fp8 { 0x7e } else { 0x7f };
    let mut k = vec![decoy_byte; pool * stride];
    let mut v = vec![decoy_byte; pool * stride];
    let floats = if decoded { pool * PAGE * DIM } else { 0 };
    let (mut keys, mut values) = (vec![0.0; floats], vec![0.0; floats]);
    let mut tables = Vec::new();
    let mut next = 0;
    for (item, pages) in owned.iter().enumerate() {
        let mut table = Vec::new();
        for &physical in pages {
            let slot = slots[next];
            next += 1;
            table.push(slot);
            for t in 0..PAGE {
                let token = physical * PAGE + t;
                if token >= kv.stored_k[item].len() {
                    // Slots past the sequence keep the decoy; never read.
                    continue;
                }
                let len = DIM * kv.element_bytes();
                let at = slot * stride + t * len;
                k[at..at + len].copy_from_slice(&kv.stored_k[item][token]);
                v[at..at + len].copy_from_slice(&kv.stored_v[item][token]);
                if decoded {
                    let base = (slot * PAGE + t) * DIM;
                    keys[base..base + DIM].copy_from_slice(&kv.keys[item][token]);
                    values[base..base + DIM].copy_from_slice(&kv.values[item][token]);
                }
            }
        }
        tables.push(table);
    }
    let max_pages = tables.iter().map(Vec::len).max().unwrap_or(0).max(1) + 1;
    RankCache {
        k,
        v,
        stride,
        max_pages,
        tables,
        keys,
        values,
    }
}

/// Scales of a case: softmax 1/16 (head dim 256), and FP8's layer scales
/// (powers of two, so scaled FP8 values are exact).
pub fn scales(fp8: bool) -> (f32, f32, f32) {
    if fp8 {
        (0.0625, 0.0078125, 0.015625)
    } else {
        (0.0625, 1.0, 1.0)
    }
}

pub fn queries(rows: usize, wide: bool, rng: &mut Rng) -> Vec<u16> {
    (0..rows * Q_HEADS * DIM)
        .map(|_| f32_to_bf16(if wide { rng.wide() } else { 2.0 * rng.unit() }))
        .collect()
}

/// A kernel's partials of the live rows.
pub struct Partials {
    pub maximum: Vec<f32>,
    pub denominator: Vec<f32>,
    pub numerator: Vec<f32>,
}

/// The operands of one partial launch.
pub struct Launch<'a> {
    pub step: &'a Step,
    pub q: &'a [u16],
    pub selected: &'a [i32],
    pub cache: &'a RankCache,
    pub scales: (f32, f32, f32),
    pub rank: usize,
    pub size: usize,
    /// Rows planned, at least the live rows.
    pub prepared: usize,
}

/// Runs `entry` for one rank's launch, under the poison replays when
/// `poison`, and returns the live rows' partials.
pub fn run(
    session: &Session,
    function: &Function,
    entry: &Entry,
    operands: &Launch<'_>,
    poison: bool,
    label: &str,
) -> LabResult<Partials> {
    let Launch {
        step,
        q,
        selected,
        cache,
        scales,
        rank,
        size,
        prepared,
    } = *operands;
    let rows = step.rows.len();
    let heads = entry.heads() as usize;
    let q_bytes = q.iter().flat_map(|b| b.to_ne_bytes()).collect::<Vec<_>>();
    let row_batch = step.rows.iter().map(|&(i, _)| i as i32).collect::<Vec<_>>();
    let positions = step.rows.iter().map(|&(_, p)| p as i32).collect::<Vec<_>>();
    let q_buffer = upload(session, &q_bytes)?;
    let k = upload(session, &cache.k)?;
    let v = upload(session, &cache.v)?;
    let tables_buffer = upload(session, &words(&cache.table_words()))?;
    let row_batch = upload(session, &words(&row_batch))?;
    let positions = upload(session, &words(&positions))?;
    let selected = upload(session, &words(selected))?;
    let live = upload(session, &(rows as u32).to_ne_bytes())?;
    let out_max = session.allocate(prepared * heads * 4)?;
    let out_den = session.allocate(prepared * heads * 4)?;
    let out_num = session.allocate(prepared * heads * DIM * 4)?;
    entry
        .check_rank(rank as u32, size as u32)
        .map_err(|e| e.0.to_string())?;
    let config = launch::config(entry.plan(prepared as u32).map_err(|e| e.0.to_string())?)?;
    let (scale, k_scale, v_scale) = scales;
    let launch = || {
        function
            .launch(
                &session.stream,
                config,
                &[
                    KernelArgument::Buffer(&q_buffer),
                    KernelArgument::Buffer(&k),
                    KernelArgument::Buffer(&v),
                    KernelArgument::Buffer(&tables_buffer),
                    KernelArgument::Buffer(&row_batch),
                    KernelArgument::Buffer(&positions),
                    KernelArgument::Buffer(&live),
                    KernelArgument::Buffer(&selected),
                    KernelArgument::Buffer(&out_max),
                    KernelArgument::Buffer(&out_den),
                    KernelArgument::Buffer(&out_num),
                    KernelArgument::U32(cache.max_pages as u32),
                    KernelArgument::U64(cache.stride as u64),
                    KernelArgument::U64(cache.stride as u64),
                    KernelArgument::U32(scale.to_bits()),
                    KernelArgument::U32(k_scale.to_bits()),
                    KernelArgument::U32(v_scale.to_bits()),
                    KernelArgument::U32(rank as u32),
                    KernelArgument::U32(size as u32),
                ],
            )
            .map_err(err)
    };
    let replay = Replay::new(session)
        .output(
            "maximum",
            &out_max,
            WriteMap::of(DType::F32).prefix(rows * heads)?,
        )
        .output(
            "denominator",
            &out_den,
            WriteMap::of(DType::F32).prefix(rows * heads)?,
        )
        .output(
            "numerator",
            &out_num,
            WriteMap::of(DType::F32).prefix(rows * heads * DIM)?,
        );
    let mut receipt = BTreeMap::from([
        ("entry".into(), json!(entry.name)),
        ("case".into(), json!(label)),
        ("rows".into(), json!(rows)),
        ("prepared".into(), json!(prepared)),
        ("rank".into(), json!(rank)),
        ("size".into(), json!(size)),
    ]);
    if poison {
        replay_all(session, replay, &live, launch, &mut receipt)?;
        session.receipt("selected_partials", receipt)?;
    } else {
        launch()?;
    }
    Ok(Partials {
        maximum: floats(&download(session, &out_max, rows * heads * 4)?),
        denominator: floats(&download(session, &out_den, rows * heads * 4)?),
        numerator: floats(&download(session, &out_num, rows * heads * DIM * 4)?),
    })
}

/// The f64 partials of one rank's launch.
pub fn reference(entry: &Entry, operands: &Launch<'_>) -> LabResult<LatentPartials> {
    let q = operands
        .q
        .iter()
        .map(|&b| f64::from(bf16_to_f32(b)))
        .collect::<Vec<_>>();
    let row_batch = operands
        .step
        .rows
        .iter()
        .map(|&(i, _)| i)
        .collect::<Vec<_>>();
    let positions = operands
        .step
        .rows
        .iter()
        .map(|&(_, p)| p)
        .collect::<Vec<_>>();
    let (scale, k_scale, v_scale) = operands.scales;
    selected_attention_partials(
        entry.query(),
        PagedPartialsInput {
            q: &q,
            keys: &operands.cache.keys,
            values: &operands.cache.values,
            tables: &operands.cache.tables,
            row_batch: &row_batch,
            positions: &positions,
            scale: f64::from(scale),
            k_scale: f64::from(k_scale),
            v_scale: f64::from(v_scale),
            cp_rank: operands.rank,
            cp_size: operands.size,
        },
        operands.selected,
    )
    .map_err(|e| e.0.to_string())
}

/// The block-sparse attention's f64 output `[rows, heads, DIM]` of every
/// row over its key set (selected blocks, then the tail), with the layer
/// scales applied to the decoded K and V.
pub fn full_reference(
    step: &Step,
    q: &[u16],
    selected: &[i32],
    kv: &Kv,
    scales: (f32, f32, f32),
) -> LabResult<OracleOutput> {
    let (scale, k_scale, v_scale) = scales;
    let query = |rows: u32| SparseAttentionQuery::SparseAttention {
        q_heads: Q_HEADS as u32,
        kv_heads: 1,
        head_dim: DIM as u32,
        block: BLOCK as u32,
        top_blocks: TOP as u32,
        softmax_scale_bits: scale.to_bits(),
        page_size: PAGE as u32,
        kv_dtype: if kv.fp8 { DType::Fp8E4M3 } else { DType::Bf16 },
        phase: AttentionPhase::Decode,
        capacity: rows,
    };
    let mut output = OracleOutput {
        values: Vec::new(),
        sum_abs_products: Vec::new(),
        reduction_lengths: Vec::new(),
    };
    // Each item's cache decoded and scaled, in page order.
    let caches = kv
        .keys
        .iter()
        .zip(&kv.values)
        .map(|(keys, values)| {
            let pages = keys.len().div_ceil(PAGE).max(1);
            let mut k = vec![0.0; pages * PAGE * DIM];
            let mut v = vec![0.0; pages * PAGE * DIM];
            for (t, (key, value)) in keys.iter().zip(values).enumerate() {
                for d in 0..DIM {
                    k[t * DIM + d] = key[d] * f64::from(k_scale);
                    v[t * DIM + d] = value[d] * f64::from(v_scale);
                }
            }
            (k, v, (0..pages).collect::<Vec<_>>())
        })
        .collect::<Vec<_>>();
    let width = Q_HEADS * DIM;
    for (row, &(item, p)) in step.rows.iter().enumerate() {
        let complete = (p + 1) / BLOCK;
        let set = selected[row * TOP..][..complete.min(TOP)]
            .iter()
            .flat_map(|&b| b as usize * BLOCK..(b as usize + 1) * BLOCK)
            .chain(complete * BLOCK..=p)
            .collect::<Vec<_>>();
        let q_row = q[row * width..][..width]
            .iter()
            .map(|&b| f64::from(bf16_to_f32(b)))
            .collect::<Vec<_>>();
        let (k, v, table) = &caches[item];
        let one = sparse_attention(
            query(1),
            &q_row,
            &[set],
            SparseKv {
                keys: k,
                values: v,
                block_table: table,
            },
        )
        .map_err(|e| e.0.to_string())?;
        output.values.extend(one.values);
        output.sum_abs_products.extend(one.sum_abs_products);
        output.reduction_lengths.extend(one.reduction_lengths);
    }
    Ok(output)
}

/// The first element of `actual` outside `reference`'s gate.
pub fn gate(name: &str, actual: &[f64], reference: &OracleOutput, output: DType) -> LabResult<()> {
    for (index, &value) in actual.iter().enumerate() {
        let expected = reference.values[index];
        let bound = numeric_bound(reference, index, output);
        if !value.is_finite() || (value - expected).abs() > bound {
            return Err(format!(
                "{name} element {index}: {value} vs {expected} (bound {bound})"
            ));
        }
    }
    Ok(())
}

/// Gates a kernel partial against the oracle's: the maximum within one
/// score's bound, then the denominator and numerator against the oracle
/// rescaled to the kernel's maximum. Empty rows must give exactly -inf, 0
/// and 0.
pub fn gate_partial(label: &str, actual: &Partials, reference: &LatentPartials) -> LabResult<()> {
    let pairs = reference.maximum.values.len();
    let empty_output = || OracleOutput {
        values: Vec::new(),
        sum_abs_products: Vec::new(),
        reduction_lengths: Vec::new(),
    };
    let (mut maximum, mut denominator, mut numerator) = (
        (Vec::new(), empty_output()),
        (Vec::new(), empty_output()),
        (Vec::new(), empty_output()),
    );
    let push = |into: &mut (Vec<f64>, OracleOutput), actual: f64, r: &OracleOutput, i, f: f64| {
        into.0.push(actual);
        into.1.values.push(r.values[i] * f);
        into.1.sum_abs_products.push(r.sum_abs_products[i] * f);
        into.1.reduction_lengths.push(r.reduction_lengths[i]);
    };
    for pair in 0..pairs {
        let m_ref = reference.maximum.values[pair];
        let m = f64::from(actual.maximum[pair]);
        if m_ref == f64::NEG_INFINITY {
            let zeros = actual.numerator[pair * DIM..(pair + 1) * DIM]
                .iter()
                .all(|&x| x.to_bits() == 0);
            if m != f64::NEG_INFINITY || actual.denominator[pair].to_bits() != 0 || !zeros {
                return Err(format!(
                    "{label}: pair {pair} has no keys but a nonempty partial"
                ));
            }
            continue;
        }
        push(&mut maximum, m, &reference.maximum, pair, 1.0);
        let factor = (m_ref - m).exp();
        push(
            &mut denominator,
            f64::from(actual.denominator[pair]),
            &reference.denominator,
            pair,
            factor,
        );
        for d in pair * DIM..(pair + 1) * DIM {
            push(
                &mut numerator,
                f64::from(actual.numerator[d]),
                &reference.numerator,
                d,
                factor,
            );
        }
    }
    gate(
        &format!("{label} maximum"),
        &maximum.0,
        &maximum.1,
        DType::F32,
    )?;
    gate(
        &format!("{label} denominator"),
        &denominator.0,
        &denominator.1,
        DType::F32,
    )?;
    gate(
        &format!("{label} numerator"),
        &numerator.0,
        &numerator.1,
        DType::F32,
    )
}

/// The merge's `exp2f` weights' allowance over the shared gate
/// (`tests/context_parallel.rs`): `2 * max_r (2^-22 + 3 * 2^-24 |m_r - M|) *
/// S`.
pub fn exp2_allowance(maxima: &[f32], top: f32, s: f64) -> f64 {
    let worst = maxima
        .iter()
        .filter(|&&m| m != f32::NEG_INFINITY)
        .map(|&m| 2_f64.powi(-22) + 3.0 * 2_f64.powi(-24) * (f64::from(top) - f64::from(m)))
        .fold(0.0, f64::max);
    2.0 * worst * s
}

/// `partial_merge{N}` over the ranks' partials in the exchanged layout
/// (each rank's segment `m`, `l`, `o`): the BF16 output `[rows, heads,
/// DIM]` as F32.
pub fn merge(
    session: &Session,
    merges: &Module,
    entries: &[context_parallel::Entry],
    parts: &[Partials],
    rows: usize,
) -> LabResult<Vec<f32>> {
    let size = parts.len();
    let segments = parts
        .iter()
        .flat_map(|p| {
            p.maximum
                .iter()
                .chain(&p.denominator)
                .chain(&p.numerator)
                .flat_map(|v| v.to_ne_bytes())
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let merge = entries
        .iter()
        .find(|e| e.kind == Kind::PartialMerge { ranks: size as u32 })
        .ok_or("no lab merge entry")?;
    let layout = MergeLayout::exchanged(rows as u32, Q_HEADS as u32);
    merge
        .check_merge(rows as u32, layout)
        .map_err(|e| e.0.to_string())?;
    let function = merges.function(&merge.symbol()).map_err(err)?;
    let partials = upload(session, &segments)?;
    let live = upload(session, &(rows as u32).to_ne_bytes())?;
    let out = session.allocate(rows * Q_HEADS * DIM * 2)?;
    function
        .launch(
            &session.stream,
            launch::config(merge.plan(rows as u32).map_err(|e| e.0.to_string())?)?,
            &[
                KernelArgument::Buffer(&partials),
                KernelArgument::Buffer(&live),
                KernelArgument::Buffer(&out),
                KernelArgument::U32(layout.rank_rows),
                KernelArgument::U32(layout.input_heads),
                KernelArgument::U32(layout.head_offset),
            ],
        )
        .map_err(err)?;
    Ok(bf16_values(&download(
        session,
        &out,
        rows * Q_HEADS * DIM * 2,
    )?))
}

/// The first merged element outside the full attention's gate plus the
/// merge's `exp2f` allowance.
pub fn gate_merged(
    label: &str,
    merged: &[f32],
    parts: &[Partials],
    full: &OracleOutput,
) -> LabResult<()> {
    for (index, &actual) in merged.iter().enumerate() {
        let pair = index / DIM;
        let maxima = parts.iter().map(|p| p.maximum[pair]).collect::<Vec<_>>();
        let top = maxima.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let expected = full.values[index];
        let bound = numeric_bound(full, index, DType::Bf16)
            + exp2_allowance(&maxima, top, full.sum_abs_products[index]);
        if !actual.is_finite() || (f64::from(actual) - expected).abs() > bound {
            return Err(format!(
                "{label} pair {pair} column {}: {actual} vs {expected} (bound {bound})",
                index % DIM
            ));
        }
    }
    Ok(())
}
