//! Paged attention partials (`paged_partials_shapes.rs`; README in
//! crates/kernels-sm120/cuda/paged_partials) at a TP4 rank of six query
//! heads over one KV head of 256 and 64-token pages, BF16 and FP8 KV:
//!
//! - every rank's partials of decode, verify and prefill rows against the
//!   f64 reference (`context_parallel_oracle::paged_attention_partials`):
//!   the maximum within one score's bound, the denominator and numerator
//!   within `2^-22 K S` of the reference rescaled to the kernel's maximum,
//!   and rows without a kept key exactly -inf, +0, +0;
//! - the 128-K accumulation rule: queries and keys spanning 2^-8..2^8 with
//!   random signs over 4096 keys, against Neumaier-compensated f64;
//! - every group rank of sizes 2 and 4, each over its own pages, merged by
//!   `partial_merge{N}` of six heads of 256, against the full-key f64
//!   attention within the shared gate (half a BF16 ulp plus `2^-22 K S`)
//!   plus the merge's `exp2f` allowance;
//! - pages are shuffled, strided past decoy bytes that would poison any
//!   read outside a page, and rows past the live count stay untouched; every
//!   launch runs eager, relaunched, from a graph and behind a PDL producer
//!   on poisoned outputs (`kernel_lab::poison`).
//!
//! Run in release with
//! `CHOPINFER_KERNEL_GPU=<ordinal> cargo test -p kernel-lab --release --test paged_partials`.

#[path = "../paged_partials_shapes.rs"]
mod shapes;

#[path = "common/support.rs"]
mod support;

use std::collections::BTreeMap;

use kernel_core::codec::{bf16_to_f32, f32_to_bf16};
use kernel_core::context_parallel_oracle::{PagedPartialsInput, paged_attention_partials};
use kernel_core::contract::DType;
use kernel_core::latent_attention_oracle::{LatentPartials, merge_partials};
use kernel_core::oracle::{OracleOutput, e4m3_table, numeric_bound};
use kernel_lab::poison::{Replay, WriteMap, replay_all};
use kernel_lab::{LabResult, Session, gpu_test, launch};
use kernels_sm120::context_parallel::{Kind, MergeLayout};
use kernels_sm120::paged_partials::Entry;
use runtime_cuda::{Function, KernelArgument};
use serde_json::json;
use support::{Rng, download, upload};

const Q_HEADS: usize = 6;
const DIM: usize = 256;
const PAGE: usize = 64;
/// Bytes of decoy between pages: BF16 3.3e38 or FP8 448, which any read
/// outside a page turns into a failing score.
const DECOY: usize = 512;

fn err(e: impl ToString) -> String {
    e.to_string()
}

fn floats(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn words(values: &[i32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_ne_bytes()).collect()
}

/// One item's tokens and its rows: (positions of the rows).
struct Step {
    /// Tokens of each item's sequence.
    lengths: Vec<usize>,
    /// Each row's item and position.
    rows: Vec<(usize, usize)>,
}

impl Step {
    /// Decode: each item's last token.
    fn decode(lengths: &[usize]) -> Step {
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
    fn verify(lengths: &[usize], k: usize) -> Step {
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
    fn prefill(prefix: usize, rows: usize) -> Step {
        Step {
            lengths: vec![prefix + rows],
            rows: (prefix..prefix + rows).map(|p| (0, p)).collect(),
        }
    }
}

/// The sequences' K and V rows, as stored (BF16 bits or FP8 bytes), and
/// decoded to f64 before their layer scale: `[item][token][dim]`.
struct Kv {
    fp8: bool,
    stored_k: Vec<Vec<Vec<u8>>>,
    stored_v: Vec<Vec<Vec<u8>>>,
    keys: Vec<Vec<Vec<f64>>>,
    values: Vec<Vec<Vec<f64>>>,
}

impl Kv {
    fn new(lengths: &[usize], fp8: bool, wide: bool, rng: &mut Rng) -> Kv {
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

    fn element_bytes(&self) -> usize {
        if self.fp8 { 1 } else { 2 }
    }
}

/// One group rank's cache: its own pages of every item, shuffled, each
/// followed by decoy bytes; its block tables of local pages; and the same
/// pages decoded for the oracle.
struct RankCache {
    k: Vec<u8>,
    v: Vec<u8>,
    stride: usize,
    max_pages: usize,
    tables: Vec<Vec<usize>>,
    keys: Vec<f64>,
    values: Vec<f64>,
}

fn rank_cache(kv: &Kv, rank: usize, size: usize, rng: &mut Rng) -> RankCache {
    let page_bytes = PAGE * DIM * kv.element_bytes();
    let stride = page_bytes + DECOY;
    // Every item's owned physical pages, in local order.
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
    // A shuffled pool: local page l of an item at pool slot `slots[...]`.
    let mut slots = (0..count).collect::<Vec<_>>();
    for i in (1..slots.len()).rev() {
        slots.swap(i, (rng.next() % (i as u64 + 1)) as usize);
    }
    let pool = count.max(1);
    let decoy_byte = if kv.fp8 { 0x7e } else { 0x7f };
    let mut k = vec![decoy_byte; pool * stride];
    let mut v = vec![decoy_byte; pool * stride];
    let mut keys = vec![0.0; pool * PAGE * DIM];
    let mut values = vec![0.0; pool * PAGE * DIM];
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
                let at = slot * stride + t * DIM * kv.element_bytes();
                let len = DIM * kv.element_bytes();
                if token < kv.stored_k[item].len() {
                    k[at..at + len].copy_from_slice(&kv.stored_k[item][token]);
                    v[at..at + len].copy_from_slice(&kv.stored_v[item][token]);
                    let base = (slot * PAGE + t) * DIM;
                    keys[base..base + DIM].copy_from_slice(&kv.keys[item][token]);
                    values[base..base + DIM].copy_from_slice(&kv.values[item][token]);
                } else {
                    // Slots past the sequence are never read.
                    k[at..at + len].fill(decoy_byte);
                    v[at..at + len].fill(decoy_byte);
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

/// The kernel's partials of the live rows.
struct Partials {
    maximum: Vec<f32>,
    denominator: Vec<f32>,
    numerator: Vec<f32>,
}

/// Runs `entry` for group rank `rank` of `size` over `cache` with
/// `prepared` rows planned (at least the live rows), through every replay
/// path, and returns the live rows' partials.
#[allow(clippy::too_many_arguments)]
fn run(
    session: &Session,
    function: &Function,
    entry: &Entry,
    step: &Step,
    q: &[u16],
    cache: &RankCache,
    scales: (f32, f32, f32),
    rank: usize,
    size: usize,
    prepared: usize,
    label: &str,
) -> LabResult<Partials> {
    let rows = step.rows.len();
    let heads = entry.heads() as usize;
    let q_bytes = q.iter().flat_map(|b| b.to_ne_bytes()).collect::<Vec<_>>();
    let mut tables = vec![-1_i32; step.lengths.len() * cache.max_pages];
    for (item, table) in cache.tables.iter().enumerate() {
        for (local, &slot) in table.iter().enumerate() {
            tables[item * cache.max_pages + local] = slot as i32;
        }
    }
    let row_batch = step.rows.iter().map(|&(i, _)| i as i32).collect::<Vec<_>>();
    let positions = step.rows.iter().map(|&(_, p)| p as i32).collect::<Vec<_>>();
    let q_buffer = upload(session, &q_bytes)?;
    let k = upload(session, &cache.k)?;
    let v = upload(session, &cache.v)?;
    let tables_buffer = upload(session, &words(&tables))?;
    let row_batch = upload(session, &words(&row_batch))?;
    let positions = upload(session, &words(&positions))?;
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
    replay_all(session, replay, &live, launch, &mut receipt)?;
    session.receipt("paged_partials", receipt)?;
    Ok(Partials {
        maximum: floats(&download(session, &out_max, rows * heads * 4)?),
        denominator: floats(&download(session, &out_den, rows * heads * 4)?),
        numerator: floats(&download(session, &out_num, rows * heads * DIM * 4)?),
    })
}

/// The f64 partials of group rank `rank` of `size` over its cache.
fn reference(
    entry: &Entry,
    step: &Step,
    q: &[u16],
    cache: &RankCache,
    scales: (f32, f32, f32),
    rank: usize,
    size: usize,
) -> LabResult<LatentPartials> {
    let q = q
        .iter()
        .map(|&b| f64::from(bf16_to_f32(b)))
        .collect::<Vec<_>>();
    let row_batch = step.rows.iter().map(|&(i, _)| i).collect::<Vec<_>>();
    let positions = step.rows.iter().map(|&(_, p)| p).collect::<Vec<_>>();
    paged_attention_partials(
        entry.query(),
        PagedPartialsInput {
            q: &q,
            keys: &cache.keys,
            values: &cache.values,
            tables: &cache.tables,
            row_batch: &row_batch,
            positions: &positions,
            scale: f64::from(scales.0),
            k_scale: f64::from(scales.1),
            v_scale: f64::from(scales.2),
            cp_rank: rank,
            cp_size: size,
        },
    )
    .map_err(|e| e.0.to_string())
}

/// The first element of `actual` outside `reference`'s gate.
fn gate(name: &str, actual: &[f64], reference: &OracleOutput, output: DType) -> LabResult<()> {
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
fn gate_partial(label: &str, actual: &Partials, reference: &LatentPartials) -> LabResult<()> {
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

fn entry(kv: &str, phase: &str) -> Entry {
    shapes::lab_entries()
        .into_iter()
        .find(|entry| entry.name == format!("{}_{kv}_{phase}", shapes::PROFILE.tag()))
        .unwrap_or_else(|| panic!("no lab entry {kv} {phase}"))
}

fn queries(rows: usize, wide: bool, rng: &mut Rng) -> Vec<u16> {
    (0..rows * Q_HEADS * DIM)
        .map(|_| f32_to_bf16(if wide { rng.wide() } else { 2.0 * rng.unit() }))
        .collect()
}

/// Scales of a case: softmax 1/16 (head dim 256), and FP8's layer scales.
fn scales(fp8: bool) -> (f32, f32, f32) {
    if fp8 {
        (0.0625, 0.0078125, 0.015625)
    } else {
        (0.0625, 1.0, 1.0)
    }
}

/// Every group rank's partials at group sizes 1, 2 and 4 against the f64
/// reference, for decode, verify and prefill rows of BF16 and FP8 KV, with
/// rows that see no key of a rank.
#[test]
fn partials_match_their_references() {
    gpu_test(4 << 30, |session| {
        let module = session.module(shapes::CUBIN)?;
        let mut rng = Rng(0x5eed_0042);
        let mut failures = Vec::new();
        for (kv_name, fp8) in [("bf16", false), ("fp8", true)] {
            let cases = [
                (
                    "decode",
                    Step::decode(&[1, 7, 64, 65, 130, 1000, 4097, 300]),
                    8,
                ),
                ("decode", Step::decode(&[50, 129]), 32),
                ("verify", Step::verify(&[4, 70, 200, 2051], 4), 32),
                ("prefill", Step::prefill(0, 300), 300),
                ("prefill", Step::prefill(190, 200), 512),
            ];
            for (phase, step, prepared) in cases {
                let entry = entry(kv_name, phase);
                let function = module.function(&entry.symbol()).map_err(err)?;
                let kv = Kv::new(&step.lengths, fp8, false, &mut rng);
                let q = queries(step.rows.len(), false, &mut rng);
                for size in [1, 2, 4] {
                    for rank in 0..size {
                        let label = format!(
                            "{} rows {} prepared {prepared} rank {rank} of {size}",
                            entry.name,
                            step.rows.len()
                        );
                        let cache = rank_cache(&kv, rank, size, &mut rng);
                        let result = run(
                            session,
                            &function,
                            &entry,
                            &step,
                            &q,
                            &cache,
                            scales(fp8),
                            rank,
                            size,
                            prepared,
                            &label,
                        )
                        .and_then(|actual| {
                            let reference =
                                reference(&entry, &step, &q, &cache, scales(fp8), rank, size)?;
                            gate_partial(&label, &actual, &reference)
                        });
                        if let Err(error) = result {
                            failures.push(error);
                        }
                    }
                }
            }
        }
        if failures.is_empty() {
            Ok(())
        } else {
            Err(failures.join("\n"))
        }
    })
    .expect("paged attention partials match their references");
}

/// Queries and keys spanning 2^-8..2^8 with random signs: a score sums 256
/// products and a partial 4096 keys, against Neumaier-compensated f64
/// within 2^-22 K S.
#[test]
fn partials_hold_the_accumulation_rule() {
    gpu_test(4 << 30, |session| {
        let module = session.module(shapes::CUBIN)?;
        let mut rng = Rng(0xacc0_0128);
        let entry = entry("bf16", "decode");
        let function = module.function(&entry.symbol()).map_err(err)?;
        let step = Step::decode(&[4096, 4000]);
        let kv = Kv::new(&step.lengths, false, true, &mut rng);
        let q = queries(step.rows.len(), true, &mut rng);
        // Scores of hundreds keep the weights from all vanishing.
        let scales = (2_f32.powi(-12), 1.0, 1.0);
        for size in [1, 2] {
            for rank in 0..size {
                let label = format!("accumulation rank {rank} of {size}");
                let cache = rank_cache(&kv, rank, size, &mut rng);
                let actual = run(
                    session, &function, &entry, &step, &q, &cache, scales, rank, size, 2, &label,
                )?;
                let reference = reference(&entry, &step, &q, &cache, scales, rank, size)?;
                gate_partial(&label, &actual, &reference)?;
            }
        }
        Ok(())
    })
    .expect("paged attention partials hold the 128-K accumulation rule");
}

/// The merge's `exp2f` weights' allowance over the shared gate
/// (`tests/context_parallel.rs`): `2 * max_r (2^-22 + 3 * 2^-24 |m_r - M|) *
/// S`.
fn exp2_allowance(maxima: &[f32], top: f32, s: f64) -> f64 {
    let worst = maxima
        .iter()
        .filter(|&&m| m != f32::NEG_INFINITY)
        .map(|&m| 2_f64.powi(-22) + 3.0 * 2_f64.powi(-24) * (f64::from(top) - f64::from(m)))
        .fold(0.0, f64::max);
    2.0 * worst * s
}

/// Every group rank's partials over its own pages, merged by
/// `partial_merge{N}` (six heads of 256) from the exchanged layout, equal
/// the full-key attention within the shared gate plus the merge's `exp2f`
/// allowance, at group sizes 2 and 4.
#[test]
fn rank_partials_merge_into_the_full_attention() {
    gpu_test(4 << 30, |session| {
        let module = session.module(shapes::CUBIN)?;
        let merges = session.module(shapes::MERGE_CUBIN)?;
        let mut rng = Rng(0x3e26_0042);
        let mut failures = Vec::new();
        for (kv_name, fp8) in [("bf16", false), ("fp8", true)] {
            for (phase, step) in [
                ("decode", Step::decode(&[3, 64, 129, 700, 2051, 5000])),
                ("verify", Step::verify(&[6, 130, 1500], 4)),
                ("prefill", Step::prefill(100, 260)),
            ] {
                let entry = entry(kv_name, phase);
                let function = module.function(&entry.symbol()).map_err(err)?;
                let kv = Kv::new(&step.lengths, fp8, false, &mut rng);
                let q = queries(step.rows.len(), false, &mut rng);
                let rows = step.rows.len();
                let heads = Q_HEADS;
                let full_cache = rank_cache(&kv, 0, 1, &mut rng);
                let full = merge_partials(&[reference(
                    &entry,
                    &step,
                    &q,
                    &full_cache,
                    scales(fp8),
                    0,
                    1,
                )?])
                .map_err(|e| e.0.to_string())?;
                for size in [2_usize, 4] {
                    let label = format!("{} rows {rows} merged over {size}", entry.name);
                    let parts = (0..size)
                        .map(|rank| {
                            let cache = rank_cache(&kv, rank, size, &mut rng);
                            run(
                                session,
                                &function,
                                &entry,
                                &step,
                                &q,
                                &cache,
                                scales(fp8),
                                rank,
                                size,
                                rows,
                                &label,
                            )
                        })
                        .collect::<LabResult<Vec<_>>>()?;
                    // The exchanged layout: each rank's segment `m`, `l`, `o`.
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
                    let merge = shapes::merge_entries()
                        .into_iter()
                        .find(|e| e.kind == Kind::PartialMerge { ranks: size as u32 })
                        .ok_or("no lab merge entry")?;
                    let layout = MergeLayout::exchanged(rows as u32, heads as u32);
                    merge
                        .check_merge(rows as u32, layout)
                        .map_err(|e| e.0.to_string())?;
                    let merge_function = merges.function(&merge.symbol()).map_err(err)?;
                    let partials = upload(session, &segments)?;
                    let live = upload(session, &(rows as u32).to_ne_bytes())?;
                    let out = session.allocate(rows * heads * DIM * 2)?;
                    merge_function
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
                    let merged = download(session, &out, rows * heads * DIM * 2)?
                        .chunks_exact(2)
                        .map(|c| bf16_to_f32(u16::from_ne_bytes([c[0], c[1]])))
                        .collect::<Vec<_>>();
                    for (index, &actual) in merged.iter().enumerate() {
                        let pair = index / DIM;
                        let maxima = parts.iter().map(|p| p.maximum[pair]).collect::<Vec<_>>();
                        let top = maxima.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                        let expected = full.values[index];
                        let bound = numeric_bound(&full, index, DType::Bf16)
                            + exp2_allowance(&maxima, top, full.sum_abs_products[index]);
                        if !actual.is_finite() || (f64::from(actual) - expected).abs() > bound {
                            failures.push(format!(
                                "{label} pair {pair} column {}: {actual} vs {expected} (bound \
                                 {bound})",
                                index % DIM
                            ));
                            break;
                        }
                    }
                }
            }
        }
        if failures.is_empty() {
            Ok(())
        } else {
            Err(failures.join("\n"))
        }
    })
    .expect("rank partials merge into the full attention");
}
