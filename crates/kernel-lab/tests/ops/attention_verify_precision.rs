//! Verify attention against decode attention on identical inputs, each
//! against the f64 oracle, to locate the DFlash speculative accuracy gap
//! (docs/numerics.md, "Batch shape"). A verify step of 5 to 64 items holds
//! items of one to eight rows at mixed contexts up to 32,761 tokens, at most
//! 256 tokens, as pooled DFlash steps do. Every token row also runs as the
//! decode row at its position over the same Q, K, V and sink, at decode
//! buckets 64 (the gate's AR bucket) and 8. Global attention's decode rows
//! read the step's pages, which hold the step's own rows as the row engine
//! writes them before attention; each sliding-window decode row gets a ring
//! slot holding its window. Both entries are prepared for 64 items, as the
//! C64 servers that measured the gap prepare them.
//!
//! Inputs come in two regimes. `grid` is the 1/64 grid of the other
//! attention labs, whose scores are nearly flat. `model` is BF16 with random
//! mantissas: normal Q and K with uneven channel scales and scores of about
//! `SCORE_SD` nats standard deviation, keys `SPIKE_NATS` above the rest at
//! position 0 and at about one position in 64, V scaled by the model's value
//! scale, and sinks across the checkpoint's range.
//!
//! Receipts (`verify_precision`) hold each step's per-stratum error sums
//! (`verify_drift_stats::Precision`) by context class, row index in the item
//! and path, which `$CHOPINFER_DATA/dflash-accuracy/analysis/lab.py` reads.
//! The test fails only on unwritten, non-finite or grossly wrong outputs.
//!
//! `attention_verify_pipeline_against_ar` sends packed QKV rows through the
//! verify phase's row RoPE and K/V write and the wide verify entries, and
//! through AR's folded one-row decode steps, at item buckets 8, 32 and 64.
//! The K/V both write must agree bit for bit, and both outputs are measured
//! against the oracle over the rotated rows. As in
//! `attention_exact_rows_pipeline_equals_ar`, a PDL producer precedes the row
//! RoPE so attention's pre-wait copies can overlap it; a nine-token step
//! must show the overlap.

use super::exact_rows::{enqueue, prepare, run};
use super::*;
use crate::verify_drift_stats::{self as stats, Precision};
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Score standard deviation of the model-like regime, in nats.
const SCORE_SD: f64 = 2.5;
/// How far a spiked key's score rises, in nats.
const SPIKE_NATS: f64 = 6.0;
/// The checkpoint's sliding-window sink biases lie in [-6.375, 2.969].
const SINK_RANGE: (f64, f64) = (-6.5, 3.0);
/// Item count both entries are prepared for, as at C64.
const SERVED_ITEMS: u32 = 64;
const MAX_TOKENS: usize = 256;
const PIPELINE_ANGLES: u64 = 50;
const PIPELINE_LIVE_TOKENS: u64 = 51;
const PIPELINE_QKV: u64 = 52;
const PRODUCER_CYCLES: u64 = 10_000_000;
/// The verify and decode variants of global and sliding-window attention.
const VARIANTS: [(&str, &str); 2] = [
    ("mimo_ga_bf16_verify", "mimo_ga_bf16_decode"),
    ("mimo_swa_verify", "mimo_swa_decode"),
];

#[derive(Clone, Copy, Debug)]
struct Item {
    prefix: usize,
    len: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Regime {
    Grid,
    Model,
}

impl Regime {
    fn name(self) -> &'static str {
        match self {
            Regime::Grid => "grid",
            Regime::Model => "model",
        }
    }
}

/// An item's K and V at positions `first` ..= its last token, [position,
/// kv head, dim], as BF16.
struct Sequence {
    first: usize,
    k: Vec<u16>,
    v: Vec<u16>,
}

/// One step's inputs: Q [token, q head, qk], each item's sequence, sinks.
struct Inputs {
    q: Vec<u16>,
    sequences: Vec<Sequence>,
    sink: Vec<u16>,
}

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed | 1)
    }
    fn unit(&mut self) -> f64 {
        (f64::from(random_word(&mut self.0)) + 0.5) / 4_294_967_296.0
    }
    fn normal(&mut self) -> f64 {
        let (a, b) = (self.unit(), self.unit());
        (-2.0 * a.ln()).sqrt() * (std::f64::consts::TAU * b).cos()
    }
    fn grid(&mut self) -> f32 {
        ((random_word(&mut self.0) % 33) as i32 - 16) as f32 / 64.0
    }
}

fn bf16(value: f64) -> u16 {
    f32_to_bf16(value as f32)
}

/// The model-like regime's shared geometry: channel scales whose fourth
/// powers average one, and one unit spike direction per KV head.
struct Model {
    channels: Vec<f64>,
    directions: Vec<f64>,
    q_sigma: f64,
    q_spike: f64,
    k_spike: f64,
}

impl Model {
    fn new(v: Variant, rng: &mut Rng) -> Self {
        let (kh, qd) = (v.query.kv_heads as usize, v.query.qk_dim as usize);
        let mut channels = (0..qd)
            .map(|_| 2.0_f64.powf(2.0 * rng.unit() - 1.0))
            .collect::<Vec<_>>();
        let fourth = (channels.iter().map(|c| c.powi(4)).sum::<f64>() / qd as f64).powf(0.25);
        channels.iter_mut().for_each(|c| *c /= fourth);
        let mut directions = Vec::with_capacity(kh * qd);
        for _ in 0..kh {
            let raw = (0..qd).map(|_| rng.normal()).collect::<Vec<_>>();
            let norm = raw.iter().map(|x| x * x).sum::<f64>().sqrt();
            directions.extend(raw.iter().map(|x| x / norm));
        }
        let scale = f64::from(v.query.softmax_scale);
        let q_sigma = SCORE_SD / (scale * (qd as f64).sqrt());
        let q_spike = 2.0 * q_sigma;
        Self {
            channels,
            directions,
            q_sigma,
            q_spike,
            k_spike: SPIKE_NATS / (q_spike * scale),
        }
    }
}

/// The step's items: lengths cycle through eight, four, two and one rows
/// (one-row items are zero-draft items); up to four items carry 16K or 32K
/// contexts; later items drop to one row until the step holds at most 256
/// tokens, as pooled DFlash steps do.
fn step_items(live: usize, salt: usize) -> Vec<Item> {
    const PREFIXES: [usize; 8] = [3, 120, 127, 128, 500, 1017, 4095, 8191];
    const LENS: [usize; 6] = [8, 4, 2, 8, 1, 2];
    const LONG: [usize; 2] = [16_383, 32_761];
    let mut items = (0..live)
        .map(|i| {
            if i % 4 == 1 && i < 16 {
                Item {
                    prefix: LONG[(i / 4 + salt) % 2],
                    len: 8,
                }
            } else {
                Item {
                    prefix: PREFIXES[(3 * i + salt) % PREFIXES.len()],
                    len: LENS[(i + salt) % LENS.len()],
                }
            }
        })
        .collect::<Vec<_>>();
    let mut tokens: usize = items.iter().map(|item| item.len).sum();
    for item in items.iter_mut().rev() {
        if tokens <= MAX_TOKENS {
            break;
        }
        tokens -= item.len - 1;
        item.len = 1;
    }
    items
}

/// First position each item's sequence holds: all of a paged item, the ring
/// window before a sliding-window item's prefix.
fn first_position(v: Variant, item: Item) -> usize {
    if v.layout == CacheLayout::Ring {
        item.prefix.saturating_sub(v.query.page_size as usize)
    } else {
        0
    }
}

fn inputs(v: Variant, items: &[Item], regime: Regime, seed: u64) -> Inputs {
    let (qh, kh) = (v.query.q_heads as usize, v.query.kv_heads as usize);
    let (qd, vd) = (v.query.qk_dim as usize, v.query.v_dim as usize);
    let group = qh / kh;
    let mut rng = Rng::new(0x5eed_7e1f_9a3c_0d21 ^ seed);
    let model = Model::new(v, &mut rng);
    let sink = (0..qh)
        .map(|_| match regime {
            Regime::Grid => f32_to_bf16(rng.grid()),
            Regime::Model => bf16(SINK_RANGE.0 + (SINK_RANGE.1 - SINK_RANGE.0) * rng.unit()),
        })
        .collect();
    let tokens: usize = items.iter().map(|item| item.len).sum();
    let mut q = Vec::with_capacity(tokens * qh * qd);
    for _ in 0..tokens {
        for h in 0..qh {
            let direction = &model.directions[(h / group) * qd..(h / group + 1) * qd];
            q.extend((0..qd).map(|d| match regime {
                Regime::Grid => f32_to_bf16(rng.grid()),
                Regime::Model => bf16(
                    model.q_sigma * model.channels[d] * rng.normal() + model.q_spike * direction[d],
                ),
            }));
        }
    }
    let sequences = items
        .iter()
        .enumerate()
        .map(|(index, &item)| {
            let mut rng =
                Rng::new(seed ^ (index as u64) << 40 ^ (item.prefix as u64) << 8 ^ 0x51_ab_c0_de);
            let first = first_position(v, item);
            let positions = item.prefix + item.len - first;
            let mut k = Vec::with_capacity(positions * kh * qd);
            let mut values = Vec::with_capacity(positions * kh * vd);
            for position in first..item.prefix + item.len {
                let spike = if position == 0 {
                    1.0
                } else if random_word(&mut rng.0) % 64 == 0 {
                    0.5 + 0.5 * rng.unit()
                } else {
                    0.0
                };
                for head in 0..kh {
                    let direction = &model.directions[head * qd..(head + 1) * qd];
                    k.extend((0..qd).map(|d| match regime {
                        Regime::Grid => f32_to_bf16(rng.grid()),
                        Regime::Model => bf16(
                            model.channels[d] * rng.normal() + spike * model.k_spike * direction[d],
                        ),
                    }));
                    values.extend((0..vd).map(|_| match regime {
                        Regime::Grid => f32_to_bf16(rng.grid()),
                        Regime::Model => bf16(f64::from(V_MULTIPLIER) * rng.normal()),
                    }));
                }
            }
            Sequence {
                first,
                k,
                v: values,
            }
        })
        .collect();
    Inputs { q, sequences, sink }
}

/// Device layout of a step's caches: each item's pages in reverse physical
/// order, or ring slot `item` holding the positions before its prefix.
struct Layout {
    k: Vec<u16>,
    v: Vec<u16>,
    tables: Vec<i32>,
    max_pages: usize,
}

fn max_pages(v: Variant) -> usize {
    (v.query.max_context_capacity as usize).div_ceil(v.query.page_size as usize)
}

fn layout(v: Variant, items: &[Item], sequences: &[Sequence]) -> Layout {
    let ring = v.layout == CacheLayout::Ring;
    let (kh, qd, vd) = (
        v.query.kv_heads as usize,
        v.query.qk_dim as usize,
        v.query.v_dim as usize,
    );
    let page = v.query.page_size as usize;
    let max_pages = max_pages(v);
    let slots = if ring {
        items.len()
    } else {
        items
            .iter()
            .map(|item| (item.prefix + item.len).div_ceil(page))
            .sum()
    };
    let mut k = vec![0_u16; slots * page * kh * qd];
    let mut values = vec![0_u16; slots * page * kh * vd];
    let mut tables = vec![0_i32; if ring { 0 } else { items.len() * max_pages }];
    let mut base = 0;
    for (index, (item, sequence)) in items.iter().zip(sequences).enumerate() {
        let pages = (item.prefix + item.len).div_ceil(page);
        if !ring {
            for p in 0..pages {
                tables[index * max_pages + p] = (base + pages - 1 - p) as i32;
            }
        }
        let end = if ring {
            item.prefix
        } else {
            item.prefix + item.len
        };
        for position in sequence.first..end {
            let row = if ring {
                index * page + position % page
            } else {
                tables[index * max_pages + position / page] as usize * page + position % page
            };
            let at = position - sequence.first;
            k[row * kh * qd..(row + 1) * kh * qd]
                .copy_from_slice(&sequence.k[at * kh * qd..(at + 1) * kh * qd]);
            values[row * kh * vd..(row + 1) * kh * vd]
                .copy_from_slice(&sequence.v[at * kh * vd..(at + 1) * kh * vd]);
        }
        base += pages;
    }
    Layout {
        k,
        v: values,
        tables,
        max_pages,
    }
}

/// The step's own rows, [token, kv head, dim], in item order.
fn own_rows(v: Variant, items: &[Item], sequences: &[Sequence]) -> (Vec<u16>, Vec<u16>) {
    let (kh, qd, vd) = (
        v.query.kv_heads as usize,
        v.query.qk_dim as usize,
        v.query.v_dim as usize,
    );
    let mut k = Vec::new();
    let mut values = Vec::new();
    for (item, sequence) in items.iter().zip(sequences) {
        let at = item.prefix - sequence.first;
        k.extend_from_slice(&sequence.k[at * kh * qd..(at + item.len) * kh * qd]);
        values.extend_from_slice(&sequence.v[at * kh * vd..(at + item.len) * kh * vd]);
    }
    (k, values)
}

fn decoded(bits: &[u16]) -> Vec<f64> {
    bits.iter().map(|&b| f64::from(bf16_to_f32(b))).collect()
}

/// The f64 oracle of every token row in item order: values and summed
/// absolute products, one item per oracle call, items spread over threads.
fn oracle_rows(
    v: Variant,
    items: &[Item],
    q: &[u16],
    sequences: &[Sequence],
    sink: &[u16],
) -> LabResult<(Vec<f64>, Vec<f64>)> {
    let ring = v.layout == CacheLayout::Ring;
    let (qh, kh) = (v.query.q_heads as usize, v.query.kv_heads as usize);
    let (qd, vd) = (v.query.qk_dim as usize, v.query.v_dim as usize);
    let page = v.query.page_size as usize;
    let sink = decoded(sink);
    let starts = items
        .iter()
        .scan(0, |next, item| {
            let start = *next;
            *next += item.len;
            Some(start)
        })
        .collect::<Vec<_>>();
    let one = |index: usize| -> LabResult<(Vec<f64>, Vec<f64>)> {
        let item = items[index];
        let sequence = &sequences[index];
        let q = decoded(&q[starts[index] * qh * qd..(starts[index] + item.len) * qh * qd]);
        let sinks = v.query.sink.then_some(sink.as_slice());
        let output = if ring {
            let mut keys = vec![0.0; page * kh * qd];
            let mut values = vec![0.0; page * kh * vd];
            for position in sequence.first..item.prefix {
                let (row, at) = (position % page, position - sequence.first);
                for (target, source) in keys[row * kh * qd..(row + 1) * kh * qd]
                    .iter_mut()
                    .zip(&sequence.k[at * kh * qd..(at + 1) * kh * qd])
                {
                    *target = f64::from(bf16_to_f32(*source));
                }
                for (target, source) in values[row * kh * vd..(row + 1) * kh * vd]
                    .iter_mut()
                    .zip(&sequence.v[at * kh * vd..(at + 1) * kh * vd])
                {
                    *target = f64::from(bf16_to_f32(*source));
                }
            }
            let at = item.prefix - sequence.first;
            let k_new = decoded(&sequence.k[at * kh * qd..(at + item.len) * kh * qd]);
            let v_new = decoded(&sequence.v[at * kh * vd..(at + item.len) * kh * vd]);
            oracle::ring_attention(
                &v.query,
                &RingAttention {
                    q: &q,
                    keys: &keys,
                    values: &values,
                    slot_ids: &[0],
                    query_lengths: &[item.len],
                    prefix_lengths: &[item.prefix],
                    k_new: Some(&k_new),
                    v_new: Some(&v_new),
                    sink_logits: sinks,
                },
            )
        } else {
            let pages = (item.prefix + item.len).div_ceil(page);
            let mut keys = decoded(&sequence.k);
            let mut values = decoded(&sequence.v);
            keys.resize(pages * page * kh * qd, 0.0);
            values.resize(pages * page * kh * vd, 0.0);
            oracle::attention(
                &v.query,
                &PagedAttention {
                    q: &q,
                    keys: &keys,
                    values: &values,
                    page_tables: &[(0..pages).collect()],
                    query_lengths: &[item.len],
                    query_starts: &[item.prefix],
                    key_lengths: &[item.prefix + item.len],
                    sink_logits: sinks,
                },
            )
        }
        .map_err(|e| format!("oracle item {index}: {e}"))?;
        Ok((output.values, output.sum_abs_products))
    };
    let results = Mutex::new((0..items.len()).map(|_| None).collect::<Vec<_>>());
    let next = AtomicUsize::new(0);
    let threads = std::thread::available_parallelism()
        .map_or(1, |n| n.get())
        .clamp(1, 32)
        .min(items.len());
    std::thread::scope(|scope| {
        for _ in 0..threads {
            scope.spawn(|| {
                loop {
                    let index = next.fetch_add(1, Ordering::Relaxed);
                    if index >= items.len() {
                        break;
                    }
                    let result = one(index);
                    results.lock().unwrap()[index] = Some(result);
                }
            });
        }
    });
    let mut values = Vec::new();
    let mut sums = Vec::new();
    for result in results.into_inner().unwrap() {
        let (item_values, item_sums) = result.ok_or("oracle item not computed")??;
        values.extend(item_values);
        sums.extend(item_sums);
    }
    if values.len() != q.len() / qd * vd {
        return Err("oracle rows do not cover the step".into());
    }
    Ok((values, sums))
}

/// The variant as the C64 servers prepare it.
fn served(name: &str) -> LabResult<Variant> {
    named(name)?
        .with_batch_capacity(SERVED_ITEMS)
        .map_err(|e| format!("{name}: {e}"))
}

fn arguments(v: Variant, verify: bool) -> Buffers {
    let ring = v.layout == CacheLayout::Ring;
    let (kh, qd, vd) = (
        v.query.kv_heads as usize,
        v.query.qk_dim as usize,
        v.query.v_dim as usize,
    );
    let page = v.query.page_size as usize;
    Buffers {
        q: Q,
        k_cache: K,
        v_cache: V,
        k_page_stride_bytes: (page * kh * qd * 2) as u64,
        v_page_stride_bytes: (page * kh * vd * 2) as u64,
        k_map: None,
        v_map: None,
        block_tables: (!ring).then_some(TABLE),
        slot_ids: ring.then_some(SLOT),
        seq_lens: Some(SEQ),
        q_start: verify.then_some(START),
        q_len: verify.then_some(LEN),
        prefix_len: verify.then_some(PREFIX),
        live_batch: LIVE,
        k_new: verify.then_some(K_NEW),
        v_new: verify.then_some(V_NEW),
        sink: v.query.sink.then_some(SINK),
        output: OUT,
        workspace: WORK,
        k_scale: 1.0,
        v_scale: 1.0,
    }
}

fn step_counts(items: &[Item], rows: usize, batch: usize) -> RuntimeCounts {
    RuntimeCounts {
        m: rows as u32,
        batch: batch as u32,
        context: items
            .iter()
            .map(|item| item.prefix + item.len)
            .max()
            .unwrap_or(1) as u32,
        routed_rows: 0,
        split_k: 1,
    }
}

/// A poisoned output of `rows` token rows.
fn output(session: &Session, rows: usize, width: usize) -> LabResult<DeviceBuffer> {
    let out = session.allocate(rows * width * 2)?;
    poison::poison(session, &[&out])?;
    Ok(out)
}

/// Downloads `rows` written token rows of `buffer`.
fn written(
    session: &Session,
    buffer: &DeviceBuffer,
    rows: usize,
    width: usize,
    label: &str,
) -> LabResult<Vec<u16>> {
    let out = session.download_bf16(buffer, rows * width)?;
    let bytes = out.iter().flat_map(|b| b.to_ne_bytes()).collect::<Vec<_>>();
    poison::check(
        label,
        &bytes,
        &WriteMap::of(DType::Bf16).byte_prefix(bytes.len())?,
        Fill::Pattern(&[POISON]),
    )?;
    Ok(out)
}

fn step_buffers(
    session: &Session,
    items: &[Item],
    q: &[u16],
    cache: &Layout,
    own: &(Vec<u16>, Vec<u16>),
    sink: &[u16],
) -> LabResult<BTreeMap<u64, DeviceBuffer>> {
    let starts = items
        .iter()
        .scan(0, |next, item| {
            let start = *next;
            *next += item.len;
            Some(start as i32)
        })
        .collect::<Vec<_>>();
    let mut buffers = BTreeMap::new();
    buffers.insert(Q, session.upload_bf16(q)?);
    buffers.insert(K, session.upload_bf16(&cache.k)?);
    buffers.insert(V, session.upload_bf16(&cache.v)?);
    buffers.insert(TABLE, put_i32(session, &cache.tables)?);
    buffers.insert(
        SLOT,
        put_i32(session, &(0..items.len() as i32).collect::<Vec<_>>())?,
    );
    let ints = |f: &dyn Fn(&Item) -> usize| items.iter().map(|i| f(i) as i32).collect::<Vec<_>>();
    buffers.insert(SEQ, put_i32(session, &ints(&|i| i.prefix + i.len))?);
    buffers.insert(START, put_i32(session, &starts)?);
    buffers.insert(LEN, put_i32(session, &ints(&|i| i.len))?);
    buffers.insert(PREFIX, put_i32(session, &ints(&|i| i.prefix))?);
    buffers.insert(LIVE, put_i32(session, &[items.len() as i32])?);
    buffers.insert(K_NEW, session.upload_bf16(&own.0)?);
    buffers.insert(V_NEW, session.upload_bf16(&own.1)?);
    buffers.insert(SINK, session.upload_bf16(sink)?);
    Ok(buffers)
}

/// Each token row's item, index within the item and position.
fn token_rows(items: &[Item]) -> Vec<(usize, usize, usize)> {
    items
        .iter()
        .enumerate()
        .flat_map(|(index, item)| (0..item.len).map(move |t| (index, t, item.prefix + t)))
        .collect()
}

/// Decode rows of every token of the step at decode bucket `bucket`, run
/// in chunks of `bucket` rows. Paged rows read the step's pages; each
/// sliding-window row reads its own ring slot holding its window.
#[allow(clippy::too_many_arguments)]
fn decode_rows(
    session: &Session,
    cubin: &Module,
    v: Variant,
    bucket: usize,
    items: &[Item],
    inputs: &Inputs,
    cache: &Layout,
    label: &str,
) -> LabResult<(Vec<u16>, String)> {
    let ring = v.layout == CacheLayout::Ring;
    let (qh, kh) = (v.query.q_heads as usize, v.query.kv_heads as usize);
    let (qd, vd) = (v.query.qk_dim as usize, v.query.v_dim as usize);
    let page = v.query.page_size as usize;
    let width = qh * vd;
    let rows = token_rows(items);
    let mut buffers = BTreeMap::new();
    if !ring {
        buffers.insert(K, session.upload_bf16(&cache.k)?);
        buffers.insert(V, session.upload_bf16(&cache.v)?);
    }
    buffers.insert(SINK, session.upload_bf16(&inputs.sink)?);
    let mut outputs = Vec::with_capacity(rows.len() * width);
    let mut symbol = String::new();
    for (chunk_index, chunk) in rows.chunks(bucket).enumerate() {
        let first = chunk_index * bucket;
        buffers.insert(
            Q,
            session.upload_bf16(&inputs.q[first * qh * qd..(first + chunk.len()) * qh * qd])?,
        );
        buffers.insert(
            SEQ,
            put_i32(
                session,
                &chunk
                    .iter()
                    .map(|&(_, _, p)| p as i32 + 1)
                    .collect::<Vec<_>>(),
            )?,
        );
        buffers.insert(LIVE, put_i32(session, &[chunk.len() as i32])?);
        if ring {
            let mut k = vec![0_u16; chunk.len() * page * kh * qd];
            let mut values = vec![0_u16; chunk.len() * page * kh * vd];
            for (slot, &(item, _, position)) in chunk.iter().enumerate() {
                let sequence = &inputs.sequences[item];
                let start = position.saturating_sub(page - 1).max(sequence.first);
                for at in start..=position {
                    let row = slot * page + at % page;
                    let source = at - sequence.first;
                    k[row * kh * qd..(row + 1) * kh * qd]
                        .copy_from_slice(&sequence.k[source * kh * qd..(source + 1) * kh * qd]);
                    values[row * kh * vd..(row + 1) * kh * vd]
                        .copy_from_slice(&sequence.v[source * kh * vd..(source + 1) * kh * vd]);
                }
            }
            buffers.insert(K, session.upload_bf16(&k)?);
            buffers.insert(V, session.upload_bf16(&values)?);
            buffers.insert(
                SLOT,
                put_i32(session, &(0..chunk.len() as i32).collect::<Vec<_>>())?,
            );
        } else {
            let mut tables = Vec::with_capacity(chunk.len() * cache.max_pages);
            for &(item, _, _) in chunk {
                tables.extend_from_slice(
                    &cache.tables[item * cache.max_pages..(item + 1) * cache.max_pages],
                );
            }
            buffers.insert(TABLE, put_i32(session, &tables)?);
        }
        buffers.insert(OUT, output(session, chunk.len(), width)?);
        let planned = v
            .plan_for_bucket(
                bucket as u32,
                step_counts(items, chunk.len(), chunk.len()),
                arguments(v, false),
            )
            .map_err(|e| format!("{label} decode b{bucket}: {e}"))?;
        symbol = planned.partial_symbol.clone();
        run(session, cubin, &planned, &mut buffers)?;
        outputs.extend(written(
            session,
            &buffers[&OUT],
            chunk.len(),
            width,
            &format!("{label} decode b{bucket} chunk {chunk_index}"),
        )?);
    }
    Ok((outputs, symbol))
}

/// Each token row's context class, by position.
fn class(position: usize) -> &'static str {
    match position {
        0..127 => "p0-126",
        127..1024 => "p127-1023",
        1024..8192 => "p1024-8191",
        _ => "p8192+",
    }
}

/// Measures `paths` (name, outputs in token order) against the oracle by
/// context class, row index in the item and path. Each path but the last
/// also counts elements that differ bit for bit from the next path, and
/// where it is more than two BF16 values from the rounded oracle while the
/// next is within one. Fails on non-finite outputs and errors above a 64th
/// of the summed absolute products.
fn measure(
    items: &[Item],
    width: usize,
    reference: &(Vec<f64>, Vec<f64>),
    paths: &[(&str, &[u16])],
    label: &str,
) -> LabResult<Value> {
    let mut strata: BTreeMap<(&str, usize, &str), (Precision, u64, u64)> = BTreeMap::new();
    for (token, (_, t, position)) in token_rows(items).into_iter().enumerate() {
        let range = token * width..(token + 1) * width;
        let (want, products) = (&reference.0[range.clone()], &reference.1[range.clone()]);
        let mut steps_by_path = Vec::new();
        for &(path, out) in paths {
            let row = out[range.clone()]
                .iter()
                .map(|&b| bf16_to_f32(b))
                .collect::<Vec<_>>();
            for (e, (&value, (&w, &s))) in row.iter().zip(want.iter().zip(products)).enumerate() {
                if !value.is_finite() || (f64::from(value) - w).abs() > s / 64.0 + 1e-30 {
                    return Err(format!(
                        "{label} {path}: token {token} (position {position}) element {e} is {value}, oracle {w} (S {s})"
                    ));
                }
            }
            let entry = strata.entry((class(position), t, path)).or_default();
            steps_by_path.push(entry.0.row(&row, want, products));
        }
        for (index, pair) in paths.windows(2).enumerate() {
            let (out, next) = (&pair[0].1[range.clone()], &pair[1].1[range.clone()]);
            let entry = strata
                .get_mut(&(class(position), t, pair[0].0))
                .expect("path stratum");
            for e in 0..width {
                entry.1 += u64::from(out[e] != next[e]);
                entry.2 +=
                    u64::from(steps_by_path[index][e] > 2 && steps_by_path[index + 1][e] <= 1);
            }
        }
    }
    Ok(Value::Array(
        strata
            .into_iter()
            .map(|((class, t, path), (precision, differs, worse))| {
                let mut record = precision.json();
                record["class"] = json!(class);
                record["row"] = json!(t);
                record["path"] = json!(path);
                record["differs_from_next_path"] = json!(differs);
                record["over_two_where_next_within_one"] = json!(worse);
                record
            })
            .collect(),
    ))
}

#[allow(clippy::too_many_arguments)]
fn precision_step(
    session: &Session,
    cubin: &Module,
    verify_variant: Variant,
    decode_variant: Variant,
    regime: Regime,
    bucket: usize,
    live: usize,
    salt: usize,
) -> LabResult<()> {
    let label = format!(
        "{} {} b{bucket} live {live}",
        verify_variant.name,
        regime.name()
    );
    let items = step_items(live, salt);
    let tokens: usize = items.iter().map(|item| item.len).sum();
    let width = (verify_variant.query.q_heads * verify_variant.query.v_dim) as usize;
    let seed = (bucket as u64) << 16 ^ (live as u64) << 24 ^ salt as u64;
    let inputs = inputs(verify_variant, &items, regime, seed);
    let cache = layout(verify_variant, &items, &inputs.sequences);
    let own = own_rows(verify_variant, &items, &inputs.sequences);

    let mut buffers = step_buffers(session, &items, &inputs.q, &cache, &own, &inputs.sink)?;
    buffers.insert(OUT, output(session, tokens, width)?);
    let planned = verify_variant
        .plan_for_bucket(
            bucket as u32,
            step_counts(&items, tokens, live),
            arguments(verify_variant, true),
        )
        .map_err(|e| format!("{label}: {e}"))?;
    run(session, cubin, &planned, &mut buffers)?;
    let verify = written(session, &buffers[&OUT], tokens, width, &label)?;
    drop(buffers);

    let (decode_64, decode_64_symbol) = decode_rows(
        session,
        cubin,
        decode_variant,
        64,
        &items,
        &inputs,
        &cache,
        &label,
    )?;
    let (decode_8, decode_8_symbol) = decode_rows(
        session,
        cubin,
        decode_variant,
        8,
        &items,
        &inputs,
        &cache,
        &label,
    )?;
    let reference = oracle_rows(
        verify_variant,
        &items,
        &inputs.q,
        &inputs.sequences,
        &inputs.sink,
    )?;
    // Verify is compared with decode at B64, and decode at B64 with B8.
    let strata = measure(
        &items,
        width,
        &reference,
        &[
            ("verify", &verify),
            ("decode_b64", &decode_64),
            ("decode_b8", &decode_8),
        ],
        &label,
    )?;
    let mut receipt = BTreeMap::new();
    receipt.insert("case".into(), json!("verify_precision"));
    receipt.insert("label".into(), json!(label));
    receipt.insert(
        "layer".into(),
        json!(if verify_variant.layout == CacheLayout::Ring {
            "swa"
        } else {
            "ga"
        }),
    );
    receipt.insert("regime".into(), json!(regime.name()));
    receipt.insert("item_bucket".into(), json!(bucket));
    receipt.insert("live_items".into(), json!(live));
    receipt.insert("tokens".into(), json!(tokens));
    receipt.insert(
        "items".into(),
        json!(
            items
                .iter()
                .map(|item| [item.prefix, item.len])
                .collect::<Vec<_>>()
        ),
    );
    receipt.insert("verify_symbol".into(), json!(planned.partial_symbol));
    receipt.insert(
        "verify_split_capacity".into(),
        json!(planned.split_capacity),
    );
    receipt.insert("verify_direct_output".into(), json!(planned.direct_output));
    receipt.insert("decode_b64_symbol".into(), json!(decode_64_symbol));
    receipt.insert("decode_b8_symbol".into(), json!(decode_8_symbol));
    receipt.insert("strata".into(), strata);
    session.receipt("verify_precision", receipt)?;
    eprintln!("{label}: {tokens} tokens measured");
    Ok(())
}

#[test]
fn attention_verify_precision_against_decode() {
    gpu_test(8 * 1024 * 1024 * 1024, |session| {
        let filter = std::env::var("CHOPINFER_ATTENTION_CASE").ok();
        let mut steps = 0;
        for (verify, decode) in VARIANTS {
            let verify_variant = served(verify)?;
            let decode_variant = served(decode)?;
            let cubin = module(session, verify_variant.profile)?;
            for regime in [Regime::Grid, Regime::Model] {
                for (salt, (bucket, partial)) in [(8, 5), (16, 12), (32, 24), (64, 40)]
                    .into_iter()
                    .enumerate()
                {
                    for live in [bucket, partial] {
                        let label = format!("{verify} {} b{bucket} live {live}", regime.name());
                        if filter.as_ref().is_some_and(|f| !label.starts_with(f)) {
                            continue;
                        }
                        precision_step(
                            session,
                            &cubin,
                            verify_variant,
                            decode_variant,
                            regime,
                            bucket,
                            live,
                            salt,
                        )?;
                        steps += 1;
                    }
                }
            }
        }
        if steps == 0 {
            return Err("no verify precision step ran".into());
        }
        eprintln!("verify precision: {steps} steps measured");
        Ok(())
    })
    .expect("verify attention precision against decode");
}

/// The step's packed QKV rows through the verify phase's row RoPE and K/V
/// write and the wide verify entries, against AR's folded one-row decode
/// steps from the same rows. A `small` step (one two-row item, the rest one
/// row) keeps the row RoPE's grid within one wave, so its pre-wait part,
/// and attention's after it, must overlap the PDL producer. Larger steps'
/// RoPE blocks cannot all be resident before the producer ends; they record
/// whether they overlapped.
fn pipeline_step(
    session: &Session,
    verify_variant: Variant,
    decode_variant: Variant,
    bucket: usize,
    small: bool,
) -> LabResult<()> {
    let label = format!(
        "{} pipeline b{bucket}{}",
        verify_variant.name,
        if small { " small" } else { "" }
    );
    let v = verify_variant;
    let ring = v.layout == CacheLayout::Ring;
    let (qh, kh) = (v.query.q_heads as usize, v.query.kv_heads as usize);
    let (qk, vd) = (v.query.qk_dim as usize, v.query.v_dim as usize);
    let page = v.query.page_size as usize;
    let width = qh * vd;
    let row_width = qh * qk + kh * (qk + vd);
    const PREFIXES: [usize; 5] = [127, 1017, 4095, 500, 3];
    const LENS: [usize; 3] = [2, 4, 8];
    let mut items = (0..bucket)
        .map(|i| Item {
            prefix: PREFIXES[i % PREFIXES.len()],
            len: match (small, i) {
                (true, 0) => 2,
                (true, _) => 1,
                (false, _) => LENS[i % LENS.len()],
            },
        })
        .collect::<Vec<_>>();
    let mut tokens: usize = items.iter().map(|item| item.len).sum();
    for item in items.iter_mut().rev() {
        if tokens <= MAX_TOKENS {
            break;
        }
        tokens -= item.len - 1;
        item.len = 1;
    }
    let context = items
        .iter()
        .map(|item| item.prefix + item.len)
        .max()
        .unwrap_or(1);
    let base = inputs(v, &items, Regime::Grid, 0x9196 ^ bucket as u64);
    let cache = layout(v, &items, &base.sequences);
    let theta = if ring { 1e4_f64 } else { 1e7 };
    let angles = (0..context)
        .flat_map(|p| {
            (0..32).flat_map(move |pair| {
                let (c, s) = kernel_core::hf_rope::cos_sin(p, pair, theta);
                [c as f32, s as f32]
            })
        })
        .collect::<Vec<_>>();
    let mut state = 0x9e37_79b9_7f4a_7c15_u64 ^ bucket as u64;
    let qkv: Vec<u16> = (0..tokens * row_width)
        .map(|_| f32_to_bf16(((random_word(&mut state) % 2001) as f32 - 1000.0) / 250.0))
        .collect();
    let cubin = module(session, v.profile)?;
    let rows = session.load_cubin(ROW_CUBIN)?;
    let positions = token_rows(&items);

    // AR: one folded decode step per token, items one after another, on
    // one cache copy (items own disjoint pages or ring slots).
    let mut ar = BTreeMap::new();
    ar.insert(K, session.upload_bf16(&cache.k)?);
    ar.insert(V, session.upload_bf16(&cache.v)?);
    ar.insert(LIVE, put_i32(session, &[1])?);
    ar.insert(SINK, session.upload_bf16(&base.sink)?);
    ar.insert(PIPELINE_ANGLES, session.upload_f32(&angles)?);
    let mut ar_rows = Vec::with_capacity(tokens * width);
    for (token, &(item, _, position)) in positions.iter().enumerate() {
        ar.insert(
            PIPELINE_QKV,
            session.upload_bf16(&qkv[token * row_width..(token + 1) * row_width])?,
        );
        ar.insert(SEQ, put_i32(session, &[position as i32 + 1])?);
        if ring {
            ar.insert(SLOT, put_i32(session, &[item as i32])?);
        } else {
            ar.insert(
                TABLE,
                put_i32(
                    session,
                    &cache.tables[item * cache.max_pages..(item + 1) * cache.max_pages],
                )?,
            );
        }
        ar.insert(OUT, output(session, 1, width)?);
        let folded = decode_variant
            .plan_folded_table(
                1,
                step_counts(&items, 1, 1),
                Buffers {
                    q: PIPELINE_QKV,
                    ..arguments(decode_variant, false)
                },
                RopeInputs {
                    qkv: PIPELINE_QKV,
                    theta: theta as f32,
                    v_multiplier: V_MULTIPLIER,
                },
                PIPELINE_ANGLES,
            )
            .map_err(|e| format!("{label}: {e}"))?;
        run(session, &cubin, &folded, &mut ar)?;
        ar_rows.extend(written(session, &ar[&OUT], 1, width, &label)?);
    }

    // Verify: the verify phase's row RoPE, then the wide verify entries.
    let own = own_rows(v, &items, &base.sequences);
    let mut verify = step_buffers(session, &items, &base.q, &cache, &own, &base.sink)?;
    verify.insert(PIPELINE_QKV, session.upload_bf16(&qkv)?);
    verify.insert(Q, session.allocate(tokens * qh * qk * 2)?);
    verify.insert(K_NEW, session.allocate(tokens * kh * qk * 2)?);
    verify.insert(V_NEW, session.allocate(tokens * kh * vd * 2)?);
    verify.insert(
        TOKEN_BATCH,
        put_i32(
            session,
            &positions
                .iter()
                .map(|&(item, _, _)| item as i32)
                .collect::<Vec<_>>(),
        )?,
    );
    verify.insert(
        POSITIONS,
        put_i32(
            session,
            &positions
                .iter()
                .map(|&(_, _, p)| p as i32)
                .collect::<Vec<_>>(),
        )?,
    );
    verify.insert(
        PIPELINE_LIVE_TOKENS,
        put_bytes(session, &(tokens as u32).to_ne_bytes())?,
    );
    verify.insert(PIPELINE_ANGLES, session.upload_f32(&angles)?);
    verify.insert(OUT, output(session, tokens, width)?);
    let (tag, suffix) = if ring {
        ("swa_prefill", "p32")
    } else {
        ("ga_bf16", "p64")
    };
    let entry = kernels_sm120::row::catalog(kernel_lab::catalogs())
        .into_iter()
        .find(|entry| entry.symbol == format!("chop_row_rope_mimo_{tag}_table_{suffix}_pos32_v1"))
        .ok_or("verify row RoPE entry missing")?;
    let token_bucket = tokens.next_power_of_two().max(2) as u32;
    let row_plan = entry
        .plan(token_bucket, step_counts(&items, tokens, items.len()))
        .map_err(|e| format!("{label}: {e}"))?;
    let row_args = entry
        .pack_arguments(vec![
            Argument::DeviceBuffer(PIPELINE_QKV),
            Argument::DeviceBuffer(Q),
            Argument::DeviceBuffer(K_NEW),
            Argument::DeviceBuffer(V_NEW),
            Argument::DeviceBuffer(K),
            Argument::DeviceBuffer(V),
            if ring {
                Argument::NullDevicePointer
            } else {
                Argument::DeviceBuffer(TABLE)
            },
            if ring {
                Argument::DeviceBuffer(SLOT)
            } else {
                Argument::NullDevicePointer
            },
            Argument::DeviceBuffer(TOKEN_BATCH),
            Argument::DeviceBuffer(POSITIONS),
            Argument::DeviceBuffer(PIPELINE_ANGLES),
            Argument::DeviceBuffer(PIPELINE_LIVE_TOKENS),
            Argument::U32(cache.max_pages as u32),
            Argument::U32(if ring { v.query.page_size } else { 0 }),
            Argument::U64(arguments(v, true).k_page_stride_bytes),
            Argument::U64(arguments(v, true).v_page_stride_bytes),
            Argument::U32((theta as f32).to_bits()),
            Argument::U32(V_MULTIPLIER.to_bits()),
            Argument::U32(1.0_f32.to_bits()),
            Argument::U32(1.0_f32.to_bits()),
        ])
        .map_err(|e| e.to_string())?;
    let planned = v
        .plan_for_bucket(
            bucket as u32,
            step_counts(&items, tokens, items.len()),
            arguments(v, true),
        )
        .map_err(|e| format!("{label}: {e}"))?;
    if !row_plan.pdl || !planned.partial.pdl {
        return Err(format!(
            "{label}: the row RoPE and attention must launch with PDL"
        ));
    }
    prepare(session, &cubin, &planned, &mut verify)?;
    let producer = PdlProducer::new(session)?;
    let function = rows.function(&entry.symbol).map_err(|e| e.to_string())?;
    let row_rope = |buffers: &BTreeMap<u64, DeviceBuffer>| {
        let bindings = Bindings {
            buffers: buffers.iter().map(|(&id, buffer)| (id, buffer)).collect(),
        };
        launch::launch(
            &session.context,
            &function,
            &session.stream,
            row_plan,
            &row_args,
            &bindings,
        )
    };
    session
        .capture(|_| {
            producer.launch_for(session, PRODUCER_CYCLES)?;
            row_rope(&verify)?;
            enqueue(session, &cubin, &planned, &verify)
        })?
        .launch(&session.stream)
        .map_err(|e| e.to_string())?;
    session.stream.synchronize().map_err(|e| e.to_string())?;
    // Void unless the kernel after the row RoPE starts before the producer
    // ends.
    session
        .capture(|_| {
            producer.launch_for(session, PRODUCER_CYCLES)?;
            row_rope(&verify)?;
            producer.launch_second(session)
        })?
        .launch(&session.stream)
        .map_err(|e| e.to_string())?;
    session.stream.synchronize().map_err(|e| e.to_string())?;
    let [_, producer_end, second_start, _] = producer.stamps(session)?;
    let overlap_ns = producer_end as i64 - second_start as i64;
    if small && overlap_ns <= 0 {
        return Err(format!(
            "{label}: the kernel after the row RoPE started {} ns after the producer ended; PDL did not overlap",
            second_start - producer_end
        ));
    }
    let verify_rows = written(session, &verify[&OUT], tokens, width, &label)?;

    // The K and V both leave: GA pages, and for the ring AR's slots against
    // the verify step's k_new/v_new rows.
    let ar_k = session.download_bf16(&ar[&K], cache.k.len())?;
    let ar_v = session.download_bf16(&ar[&V], cache.v.len())?;
    let (verify_k, verify_v) = if ring {
        (
            session.download_bf16(&verify[&K_NEW], tokens * kh * qk)?,
            session.download_bf16(&verify[&V_NEW], tokens * kh * vd)?,
        )
    } else {
        (
            session.download_bf16(&verify[&K], cache.k.len())?,
            session.download_bf16(&verify[&V], cache.v.len())?,
        )
    };
    let cache_row = |item: usize, position: usize| {
        if ring {
            item * page + position % page
        } else {
            cache.tables[item * cache.max_pages + position / page] as usize * page + position % page
        }
    };
    // Positions before an item's prefix keep the step's initial cache; AR's
    // ring has since overwritten its oldest slots with the step's rows.
    let mut sequences = Vec::with_capacity(items.len());
    for (index, item) in items.iter().enumerate() {
        let first = first_position(v, *item);
        let mut k = Vec::new();
        let mut values = Vec::new();
        for position in first..item.prefix + item.len {
            let row = cache_row(index, position);
            if position < item.prefix {
                k.extend_from_slice(&cache.k[row * kh * qk..(row + 1) * kh * qk]);
                values.extend_from_slice(&cache.v[row * kh * vd..(row + 1) * kh * vd]);
                continue;
            }
            let (k_ar, v_ar) = (
                &ar_k[row * kh * qk..(row + 1) * kh * qk],
                &ar_v[row * kh * vd..(row + 1) * kh * vd],
            );
            let token = positions
                .iter()
                .position(|&(i, _, p)| i == index && p == position)
                .expect("own position");
            let (k_verify, v_verify) = if ring {
                (
                    &verify_k[token * kh * qk..(token + 1) * kh * qk],
                    &verify_v[token * kh * vd..(token + 1) * kh * vd],
                )
            } else {
                (
                    &verify_k[row * kh * qk..(row + 1) * kh * qk],
                    &verify_v[row * kh * vd..(row + 1) * kh * vd],
                )
            };
            if k_ar != k_verify || v_ar != v_verify {
                return Err(format!(
                    "{label}: item {index} position {position}: K or V differs from AR"
                ));
            }
            k.extend_from_slice(k_verify);
            values.extend_from_slice(v_verify);
        }
        sequences.push(Sequence {
            first,
            k,
            v: values,
        });
    }
    let rotated_q = session.download_bf16(&verify[&Q], tokens * qh * qk)?;
    let reference = oracle_rows(v, &items, &rotated_q, &sequences, &base.sink)?;
    let strata = measure(
        &items,
        width,
        &reference,
        &[("verify", &verify_rows), ("ar_b1", &ar_rows)],
        &label,
    )?;
    let mut receipt = BTreeMap::new();
    receipt.insert("case".into(), json!("verify_pipeline_precision"));
    receipt.insert("label".into(), json!(label));
    receipt.insert("layer".into(), json!(if ring { "swa" } else { "ga" }));
    receipt.insert("item_bucket".into(), json!(bucket));
    receipt.insert("tokens".into(), json!(tokens));
    receipt.insert("kv_bitwise_equal".into(), json!(true));
    receipt.insert("small".into(), json!(small));
    receipt.insert("pdl_overlap_ns".into(), json!(overlap_ns));
    receipt.insert("verify_symbol".into(), json!(planned.partial_symbol));
    receipt.insert("row_symbol".into(), json!(entry.symbol));
    receipt.insert("strata".into(), strata);
    session.receipt("verify_precision", receipt)?;
    eprintln!("{label}: K/V bitwise equal to AR over {tokens} tokens");
    Ok(())
}

#[test]
fn attention_verify_pipeline_against_ar() {
    gpu_test(4 * 1024 * 1024 * 1024, |session| {
        let mut steps = 0;
        for (verify, decode) in VARIANTS {
            let verify_variant = served(verify)?;
            let decode_variant = served(decode)?;
            for (bucket, small) in [(8, true), (8, false), (32, false), (64, false)] {
                pipeline_step(session, verify_variant, decode_variant, bucket, small)?;
                steps += 1;
            }
        }
        eprintln!("verify pipeline: {steps} steps with K/V bitwise equal to AR");
        Ok(())
    })
    .expect("verify pipeline against AR");
}

#[test]
fn verify_precision_steps_hold_pooled_shapes() {
    for (bucket, partial) in [(8, 5), (16, 12), (32, 24), (64, 40)] {
        for live in [bucket, partial] {
            let items = step_items(live, 0);
            let tokens: usize = items.iter().map(|item| item.len).sum();
            assert_eq!(items.len(), live);
            assert!(tokens <= MAX_TOKENS, "{live} items hold {tokens} tokens");
            assert!(items.iter().all(|item| (1..=8).contains(&item.len)));
            let long = items.iter().filter(|item| item.prefix > 8191).count();
            assert!(long <= 4 && (live < 2 || long >= 1));
        }
    }
    assert_eq!(stats::bf16_step(1.0), 1.0 / 128.0);
    assert_eq!(stats::bf16_step(0.75), 1.0 / 256.0);
}
