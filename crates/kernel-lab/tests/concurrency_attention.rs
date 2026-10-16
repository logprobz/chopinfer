//! MiMo above 32 sequences: MiMo attention decode and verify above 32 rows on one GPU,
//! with the entries the bundle launches (decode folds RoPE and the KV write
//! in). Decode: 32 rows (the reference) to 256 at uniform 2K, 8K and 32K
//! contexts, plus a mixed step of one 32K row among 2K rows; 33 and 65 rows
//! in the 64 and 128 buckets (padding cost); and each wide bucket's split
//! limit swept over 8 to 64. Verify: 256 query tokens over 32 (K7), 64 (K3),
//! 128 (K1) and 256 (one row each) items after 2K and 8K prefixes.
//!
//! Correctness at these buckets is the `ops` suite's `attention_wide_*`
//! tests. Here every timed graph's output equals its eager launch bit for
//! bit, and a swept split's output stays within 4 BF16 steps of the tuned
//! split's. Each graph holds one partial and merge with L2 flushed before it.

use kernel_core::codec::{bf16_to_f32, f32_to_bf16};
use kernel_core::contract::{AttentionPhase, DType, RuntimeCounts};
use kernel_core::kv_fp8;
use kernel_lab::launch::{self, Bindings};
use kernel_lab::{LabResult, Session, gpu_test};
use kernels_sm120::attention::{self, Buffers, CacheLayout, PlannedAttention, RopeInputs, Variant};
use runtime_cuda::{DeviceBuffer, Function, Module};
use serde_json::{Value, json};
use std::collections::BTreeMap;
#[path = "common/concurrency.rs"]
mod conc;

const CUBIN: &str = "attention_mimo_sm120";
use kernel_lab::poison::{Phase, Replay, WriteMap};
use std::sync::atomic::{AtomicUsize, Ordering};
static MEASURED: AtomicUsize = AtomicUsize::new(0);
fn selected(name: &str) -> bool {
    std::env::var("CHOPINFER_CONC_ATTENTION_FILTER")
        .ok()
        .is_none_or(|filter| filter.split(',').any(|wanted| name == wanted))
}

const Q: u64 = 1;
const K: u64 = 2;
const V: u64 = 3;
const TABLE: u64 = 4;
const SLOT: u64 = 5;
const SEQ: u64 = 6;
const START: u64 = 7;
const LEN: u64 = 8;
const PREFIX: u64 = 9;
const LIVE: u64 = 10;
const K_NEW: u64 = 11;
const V_NEW: u64 = 12;
const SINK: u64 = 13;
const OUT: u64 = 14;
const WORK: u64 = 15;
/// Distinct random pages repeated through the paged cache; a prime count so
/// neighbouring rows read different page sequences.
const TEMPLATE_PAGES: usize = 61;
/// Most pages any case reads: 256 rows of 32K tokens in 64-token pages.
const CACHE_PAGES: usize = 256 * 32_768 / 64;
const LONG: usize = 32_768;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 32) as u32
    }
    /// A value in [-0.75, 0.75].
    fn value(&mut self) -> f32 {
        (f64::from(self.next()) / f64::from(u32::MAX) * 1.5 - 0.75) as f32
    }
    fn bf16_bytes(&mut self, count: usize) -> Vec<u8> {
        (0..count)
            .flat_map(|_| f32_to_bf16(self.value()).to_ne_bytes())
            .collect()
    }
    fn fp8_bytes(&mut self, count: usize, scale: f32) -> Vec<u8> {
        (0..count)
            .map(|_| kv_fp8::encode(self.value(), scale))
            .collect()
    }
}

fn upload(s: &Session, bytes: &[u8]) -> LabResult<DeviceBuffer> {
    let buffer = s.allocate(bytes.len().max(4))?;
    buffer
        .copy_from_bytes(bytes, &s.stream)
        .map_err(|e| e.to_string())?;
    Ok(buffer)
}

fn upload_i32(s: &Session, values: &[i32]) -> LabResult<DeviceBuffer> {
    upload(
        s,
        &values
            .iter()
            .flat_map(|v| v.to_ne_bytes())
            .collect::<Vec<_>>(),
    )
}

/// A cache of `pages` pages of `page_bytes`, holding `TEMPLATE_PAGES` random
/// pages repeated.
fn repeated(s: &Session, pages: usize, template: &[u8]) -> LabResult<DeviceBuffer> {
    let page_bytes = template.len() / TEMPLATE_PAGES;
    let buffer = s.allocate(pages * page_bytes)?;
    buffer
        .copy_from_bytes_at(0, template, &s.stream)
        .map_err(|e| e.to_string())?;
    let mut filled = TEMPLATE_PAGES;
    while filled < pages {
        let count = filled.min(pages - filled);
        buffer
            .copy_within_async(0, filled * page_bytes, count * page_bytes, &s.stream)
            .map_err(|e| e.to_string())?;
        filled += count;
    }
    s.stream.synchronize().map_err(|e| e.to_string())?;
    Ok(buffer)
}

/// The paged K and V caches of one KV dtype.
struct Paged {
    k: DeviceBuffer,
    v: DeviceBuffer,
    fp8: bool,
}

fn paged(s: &Session, variant: Variant, rng: &mut Rng) -> LabResult<Paged> {
    let query = variant.query;
    let fp8 = query.kv_dtype == DType::Fp8E4M3;
    let tokens = TEMPLATE_PAGES * query.page_size as usize * query.kv_heads as usize;
    let (k, v) = if fp8 {
        (
            rng.fp8_bytes(tokens * query.qk_dim as usize, 0.5),
            rng.fp8_bytes(tokens * query.v_dim as usize, 0.25),
        )
    } else {
        (
            rng.bf16_bytes(tokens * query.qk_dim as usize),
            rng.bf16_bytes(tokens * query.v_dim as usize),
        )
    };
    Ok(Paged {
        k: repeated(s, CACHE_PAGES, &k)?,
        v: repeated(s, CACHE_PAGES, &v)?,
        fp8,
    })
}

/// One step's inputs, bound by handle next to the shared caches.
struct Step {
    owned: BTreeMap<u64, DeviceBuffer>,
    buffers: Buffers,
    counts: RuntimeCounts,
    output_len: usize,
    /// KV bytes the step reads plus its Q and output bytes.
    logical_bytes: u64,
}

impl Step {
    fn bindings<'a>(&'a self, cache: (&'a DeviceBuffer, &'a DeviceBuffer)) -> Bindings<'a> {
        let mut buffers: BTreeMap<u64, &DeviceBuffer> =
            self.owned.iter().map(|(&id, b)| (id, b)).collect();
        buffers.insert(K, cache.0);
        buffers.insert(V, cache.1);
        Bindings { buffers }
    }
}

fn page_table(variant: Variant, keys: &[usize]) -> (Vec<i32>, usize) {
    let page = variant.query.page_size as usize;
    let max_pages = (variant.query.max_context_capacity as usize).div_ceil(page);
    let mut table = vec![0; keys.len() * max_pages];
    let mut next = 0;
    for (row, &count) in keys.iter().enumerate() {
        for logical in 0..count.div_ceil(page) {
            table[row * max_pages + logical] = next as i32;
            next += 1;
        }
    }
    assert!(next <= CACHE_PAGES, "the cache holds every page");
    (table, next)
}

fn scales(fp8: bool) -> (f32, f32) {
    if fp8 { (0.5, 0.25) } else { (1.0, 1.0) }
}

/// A decode step of rows attending `contexts` tokens (the current one
/// included), with packed QKV rows for the folded entry.
fn decode_step(
    s: &Session,
    variant: Variant,
    contexts: &[usize],
    rng: &mut Rng,
) -> LabResult<Step> {
    let query = variant.query;
    let rows = contexts.len();
    let ring = variant.layout == CacheLayout::Ring;
    let (qh, kh) = (query.q_heads as usize, query.kv_heads as usize);
    let (qk, vd) = (query.qk_dim as usize, query.v_dim as usize);
    let fp8 = query.kv_dtype == DType::Fp8E4M3;
    let mut owned = BTreeMap::new();
    owned.insert(
        Q,
        upload(s, &rng.bf16_bytes(rows * (qh * qk + kh * (qk + vd))))?,
    );
    if ring {
        owned.insert(
            SLOT,
            upload_i32(s, &(0..rows as i32).rev().collect::<Vec<_>>())?,
        );
        owned.insert(SINK, upload(s, &rng.bf16_bytes(qh))?);
    } else {
        owned.insert(TABLE, upload_i32(s, &page_table(variant, contexts).0)?);
    }
    owned.insert(
        SEQ,
        upload_i32(s, &contexts.iter().map(|&c| c as i32).collect::<Vec<_>>())?,
    );
    owned.insert(LIVE, upload_i32(s, &[rows as i32])?);
    let output_len = rows * qh * vd;
    owned.insert(OUT, s.allocate(output_len * 2)?);
    let window = query
        .window_left
        .map_or(usize::MAX, |left| left as usize + 1);
    let visible: usize = contexts.iter().map(|&c| c.min(window)).sum();
    let element = if fp8 { 1 } else { 2 };
    let (k_scale, v_scale) = scales(fp8);
    let page = query.page_size as u64;
    Ok(Step {
        owned,
        buffers: Buffers {
            q: Q,
            k_cache: K,
            v_cache: V,
            k_page_stride_bytes: page * kh as u64 * qk as u64 * element,
            v_page_stride_bytes: page * kh as u64 * vd as u64 * element,
            k_map: None,
            v_map: None,
            block_tables: (!ring).then_some(TABLE),
            slot_ids: ring.then_some(SLOT),
            seq_lens: Some(SEQ),
            q_start: None,
            q_len: None,
            prefix_len: None,
            live_batch: LIVE,
            k_new: None,
            v_new: None,
            sink: query.sink.then_some(SINK),
            output: OUT,
            workspace: WORK,
            k_scale,
            v_scale,
        },
        counts: RuntimeCounts {
            m: rows as u32,
            batch: rows as u32,
            context: *contexts.iter().max().unwrap() as u32,
            routed_rows: 0,
            split_k: 1,
        },
        output_len,
        logical_bytes: (visible * kh * (qk + vd) * element as usize
            + rows * (qh * qk + kh * (qk + vd) + qh * vd) * 2) as u64,
    })
}

/// A verify step of `items` items of `tokens` query tokens after `prefix`
/// cached tokens each.
fn verify_step(
    s: &Session,
    variant: Variant,
    items: usize,
    tokens: usize,
    prefix: usize,
    rng: &mut Rng,
) -> LabResult<Step> {
    let query = variant.query;
    let ring = variant.layout == CacheLayout::Ring;
    let (qh, kh) = (query.q_heads as usize, query.kv_heads as usize);
    let (qk, vd) = (query.qk_dim as usize, query.v_dim as usize);
    let packed = items * tokens;
    let mut owned = BTreeMap::new();
    owned.insert(Q, upload(s, &rng.bf16_bytes(packed * qh * qk))?);
    let keys = vec![prefix + tokens; items];
    if ring {
        owned.insert(
            SLOT,
            upload_i32(s, &(0..items as i32).rev().collect::<Vec<_>>())?,
        );
        owned.insert(K_NEW, upload(s, &rng.bf16_bytes(packed * kh * qk))?);
        owned.insert(V_NEW, upload(s, &rng.bf16_bytes(packed * kh * vd))?);
        owned.insert(SINK, upload(s, &rng.bf16_bytes(qh))?);
    } else {
        owned.insert(TABLE, upload_i32(s, &page_table(variant, &keys).0)?);
    }
    let starts: Vec<i32> = (0..items).map(|i| (i * tokens) as i32).collect();
    owned.insert(START, upload_i32(s, &starts)?);
    owned.insert(LEN, upload_i32(s, &vec![tokens as i32; items])?);
    owned.insert(PREFIX, upload_i32(s, &vec![prefix as i32; items])?);
    owned.insert(LIVE, upload_i32(s, &[items as i32])?);
    let output_len = packed * qh * vd;
    owned.insert(OUT, s.allocate(output_len * 2)?);
    let window = query
        .window_left
        .map_or(usize::MAX, |left| left as usize + 1);
    let visible: usize = keys
        .iter()
        .map(|&c| c.min(window.saturating_add(tokens - 1)))
        .sum();
    let page = query.page_size as u64;
    Ok(Step {
        owned,
        buffers: Buffers {
            q: Q,
            k_cache: K,
            v_cache: V,
            k_page_stride_bytes: page * kh as u64 * qk as u64 * 2,
            v_page_stride_bytes: page * kh as u64 * vd as u64 * 2,
            k_map: None,
            v_map: None,
            block_tables: (!ring).then_some(TABLE),
            slot_ids: ring.then_some(SLOT),
            seq_lens: None,
            q_start: Some(START),
            q_len: Some(LEN),
            prefix_len: Some(PREFIX),
            live_batch: LIVE,
            k_new: ring.then_some(K_NEW),
            v_new: ring.then_some(V_NEW),
            sink: query.sink.then_some(SINK),
            output: OUT,
            workspace: WORK,
            k_scale: 1.0,
            v_scale: 1.0,
        },
        counts: RuntimeCounts {
            m: packed as u32,
            batch: items as u32,
            context: (prefix + tokens) as u32,
            routed_rows: 0,
            split_k: 1,
        },
        output_len,
        logical_bytes: (visible * kh * (qk + vd) * 2 + packed * qh * (qk + vd) * 2) as u64,
    })
}

/// `planned` with `splits` owner slots per row: the partial and merge
/// arguments and the workspace.
fn with_splits(
    mut planned: PlannedAttention,
    variant: Variant,
    bucket: u32,
    splits: u32,
) -> LabResult<PlannedAttention> {
    use kernel_core::contract::Argument;
    let current = Argument::U32(planned.split_capacity);
    if planned.partial_args.0[16] != current || planned.merge_args.0[8] != current {
        return Err("decode split capacity is not where the planner puts it".into());
    }
    planned.partial_args.0[16] = Argument::U32(splits);
    planned.merge_args.0[8] = Argument::U32(splits);
    planned.split_capacity = splits;
    let workspace = u64::from(bucket)
        * u64::from(variant.query.q_heads)
        * u64::from(splits)
        * u64::from(variant.query.v_dim + 2)
        * 4;
    planned.partial.workspace_bytes = workspace;
    planned.merge.workspace_bytes = workspace;
    Ok(planned)
}

fn function(module: &Module, symbol: &str, dynamic: u32) -> LabResult<Function> {
    let function = module.function(symbol).map_err(|e| e.to_string())?;
    if dynamic > 48 * 1024 {
        function
            .set_max_dynamic_shared_memory(dynamic)
            .map_err(|e| e.to_string())?;
    }
    Ok(function)
}

fn words(s: &Session, buffer: &DeviceBuffer) -> LabResult<Vec<usize>> {
    Ok(kernel_lab::poison::read(s, buffer)?
        .chunks_exact(4)
        .map(|word| i32::from_ne_bytes(word.try_into().expect("word")) as usize)
        .collect())
}

// Pieces per row (verify: per item) of the row-local split (decode.cuh
// `piece_units`): a row of u units is cut into pieces of max(split_units,
// ceil(u / min(splits, target))) units from its first unit; verify packs
// the records by live query token.
fn split_pieces(
    units: &[usize],
    splits: usize,
    target: usize,
    split_units: usize,
) -> LabResult<Vec<usize>> {
    if units.is_empty() || units.contains(&0) || splits < 2 || target == 0 || split_units == 0 {
        return Err("invalid split-piece geometry".into());
    }
    let most = splits.min(target);
    Ok(units
        .iter()
        .map(|&count| count.div_ceil(split_units.max(count.div_ceil(most))))
        .collect())
}

fn maps(
    s: &Session,
    variant: Variant,
    planned: &PlannedAttention,
    step: &Step,
) -> LabResult<(WriteMap, Option<(WriteMap, WriteMap)>)> {
    let q = variant.query;
    let (kh, group, vd) = (
        q.kv_heads as usize,
        (q.q_heads / q.kv_heads) as usize,
        q.v_dim as usize,
    );
    let splits = planned.split_capacity as usize;
    let keys = if q.qk_dim + q.v_dim <= 320 { 32 } else { 16 };
    let window = q.window_left.map(|n| n as usize);
    let decode = step.buffers.seq_lens.is_some();
    let lengths = if decode {
        words(s, &step.owned[&SEQ])?
    } else {
        words(s, &step.owned[&LEN])?
    };
    let prefixes = if decode {
        vec![0; lengths.len()]
    } else {
        words(s, &step.owned[&PREFIX])?
    };
    let units = lengths
        .iter()
        .enumerate()
        .map(|(item, &n)| {
            let last = if decode {
                n - 1
            } else {
                prefixes[item] + n - 1
            };
            let first = window.map_or(0, |w| {
                if decode {
                    last.saturating_sub(w)
                } else {
                    prefixes[item].saturating_sub(w)
                }
            });
            last / keys - first / keys + 1
        })
        .collect::<Vec<_>>();
    let mut ranges = Vec::new();
    if !planned.direct_output && splits > 1 {
        // Split target and split_units follow the split capacity
        // (partial arguments 17 and 18).
        let argument = |index: usize| match planned.partial_args.0[index] {
            kernel_core::contract::Argument::U32(value) => Ok(value as usize),
            _ => Err(format!("partial argument {index} is not a U32")),
        };
        let pieces = split_pieces(&units, splits, argument(17)?, argument(18)?)?;
        let mut token = 0;
        for (item, &pieces) in pieces.iter().enumerate() {
            for _ in 0..if decode { 1 } else { lengths[item] } {
                for head in 0..kh {
                    for piece in 0..pieces {
                        for grouped in 0..group {
                            let start = (((token * kh + head) * splits + piece) * group + grouped)
                                * (vd + 2)
                                * 4;
                            ranges.push(start..start + (vd + 2) * 4);
                        }
                    }
                }
                token += 1;
            }
        }
    }
    let workspace = WriteMap::of(DType::F32).byte_ranges(ranges)?;
    let caches = if planned.writes_cache {
        let cache = WriteMap::of(if q.kv_dtype == DType::Fp8E4M3 {
            DType::Fp8E4M3
        } else {
            DType::Bf16
        });
        let element = cache.bytes()?;
        let table = if variant.layout == CacheLayout::Paged {
            words(s, &step.owned[&TABLE])?
        } else {
            words(s, &step.owned[&SLOT])?
        };
        let mut k = Vec::new();
        let mut v = Vec::new();
        let page = q.page_size as usize;
        let max_pages = (q.max_context_capacity as usize).div_ceil(page);
        for (row, &context) in lengths.iter().enumerate() {
            let position = context - 1;
            let slot = if variant.layout == CacheLayout::Paged {
                table[row * max_pages + position / page]
            } else {
                table[row]
            };
            for (ranges, dim) in [(&mut k, q.qk_dim as usize), (&mut v, vd)] {
                let start = (slot * page + position % page) * kh * dim * element;
                ranges.push(start..start + kh * dim * element);
            }
        }
        Some((cache.byte_ranges(k)?, cache.byte_ranges(v)?))
    } else {
        None
    };
    Ok((workspace, caches))
}

/// Times one step's partial and merge as a one-call graph after its eager
/// launch, requires the replays' output to equal the eager one, and records
/// the receipt. Returns the eager output.
#[allow(clippy::too_many_arguments)]
fn measure(
    s: &Session,
    module: &Module,
    variant: Variant,
    planned: &PlannedAttention,
    step: &mut Step,
    cache: (&DeviceBuffer, &DeviceBuffer),
    operation: &str,
    mut extra: BTreeMap<String, Value>,
) -> LabResult<Vec<u16>> {
    step.owned.insert(
        WORK,
        s.allocate(planned.partial.workspace_bytes.max(4) as usize)?,
    );
    let partial = function(
        module,
        &planned.partial_symbol,
        planned.partial.dynamic_shared_bytes,
    )?;
    let merge = function(
        module,
        &format!("chop_attn_{}_merge_v1", variant.name),
        planned.merge.dynamic_shared_bytes,
    )?;
    let step = &*step;
    let bindings = step.bindings(cache);
    let run = || -> LabResult<()> {
        launch::launch(
            &s.context,
            &partial,
            &s.stream,
            planned.partial,
            &planned.partial_args,
            &bindings,
        )?;
        if planned.direct_output {
            return Ok(());
        }
        launch::launch(
            &s.context,
            &merge,
            &s.stream,
            planned.merge,
            &planned.merge_args,
            &bindings,
        )
    };
    let (work_map, cache_maps) = maps(s, variant, planned, step)?;
    let mut replay = Replay::new(s)
        .output(
            "attention",
            &step.owned[&OUT],
            WriteMap::of(DType::Bf16).prefix(step.output_len)?,
        )
        .output("attention partials", &step.owned[&WORK], work_map);
    if let Some((k_map, v_map)) = cache_maps {
        // Current-token cells are write-only; all other cache bytes retain
        // their canonical image. Descriptors and QKV remain valid inputs.
        replay = replay
            .state(
                "K cache",
                cache.0,
                kernel_lab::poison::read(s, cache.0)?,
                k_map,
                Vec::new(),
            )
            .state(
                "V cache",
                cache.1,
                kernel_lab::poison::read(s, cache.1)?,
                v_map,
                Vec::new(),
            );
    }
    replay.eager(run)?;
    let output = || s.download_bf16(&step.owned[&OUT], step.output_len);
    let eager = output()?;
    if eager.iter().any(|&bits| !bf16_to_f32(bits).is_finite()) {
        return Err(format!("{operation}: output is not finite"));
    }
    let graph = s.capture(|_| run())?;
    let timing = kernel_lab::time(s, 3, 9, |stream| {
        graph.launch(stream).map_err(|e| e.to_string())
    })?;
    if output()? != eager {
        return Err(format!("{operation}: graph replay differs from eager"));
    }
    replay.graph(&graph)?;
    replay.phase(Phase::Relaunch, run, planned.partial.pdl)?;
    replay.phase(Phase::Pdl, run, planned.partial.pdl)?;
    replay.record(&mut extra);
    MEASURED.fetch_add(1, Ordering::Relaxed);
    let median_us = f64::from(timing.median_ms) * 1000.0;
    extra.extend(BTreeMap::from([
        ("variant".into(), json!(variant.name)),
        ("partial_symbol".into(), json!(planned.partial_symbol)),
        ("split_capacity".into(), json!(planned.split_capacity)),
        ("partial_grid".into(), json!(planned.partial.grid)),
        ("merge_grid".into(), json!(planned.merge.grid)),
        ("direct_output".into(), json!(planned.direct_output)),
        (
            "workspace_bytes".into(),
            json!(planned.partial.workspace_bytes),
        ),
        ("logical_bytes".into(), json!(step.logical_bytes)),
        (
            "logical_gb_per_second".into(),
            json!(step.logical_bytes as f64 / median_us / 1000.0),
        ),
        ("calls_per_graph".into(), json!(1)),
        ("cache_policy".into(), json!("L2 flushed before each graph")),
        ("rounds".into(), json!(9)),
    ]));
    conc::record(
        s,
        "M5",
        operation,
        step.counts.batch as usize,
        CUBIN,
        median_us,
        f64::from(timing.min_ms) * 1000.0,
        extra,
    )?;
    Ok(eager)
}

/// Within `ulps` BF16 steps of the larger magnitude, at least of 1/128.
fn close(operation: &str, expected: &[u16], actual: &[u16], ulps: f32) -> LabResult<()> {
    for (index, (&a, &b)) in actual.iter().zip(expected).enumerate() {
        let (a, b) = (bf16_to_f32(a), bf16_to_f32(b));
        // Outputs are weighted means of V in [-0.75, 0.75]: a split's
        // rounding scales with V, not with an output that cancels near zero.
        let step = a.abs().max(b.abs()).max(1.0 / 128.0) / 128.0;
        if !(a - b).abs().le(&(ulps * step)) {
            return Err(format!(
                "{operation} differs from the tuned split at {index}: {a} against {b}"
            ));
        }
    }
    Ok(())
}

fn catalog(name: &str) -> LabResult<Variant> {
    attention::catalog(kernel_lab::catalogs())
        .into_iter()
        .find(|variant| variant.name == name)
        .ok_or_else(|| format!("missing attention variant {name}"))?
        .with_batch_capacity(attention::MAX_BATCH_ROWS)
        .map_err(|e| e.to_string())
}

#[derive(Clone, Copy)]
enum Contexts {
    Uniform(usize),
    /// One 32K row among 2K rows: the widest row's split slots bind.
    Mixed,
}

impl Contexts {
    fn rows(self, rows: usize) -> Vec<usize> {
        match self {
            Self::Uniform(context) => vec![context; rows],
            Self::Mixed => {
                let mut contexts = vec![2048; rows];
                contexts[0] = LONG;
                contexts
            }
        }
    }
    fn label(self) -> String {
        match self {
            Self::Uniform(context) => format!("ctx{context}"),
            Self::Mixed => "mixed32k".into(),
        }
    }
}

fn decode(s: &Session, module: &Module, rng: &mut Rng) -> LabResult<()> {
    let mut failures = conc::replay::Failures::default();
    let sweep = [8, 16, 32, 64];
    for name in ["mimo_ga_bf16_decode", "mimo_ga_fp8_decode"] {
        if !selected(name) {
            continue;
        }
        let variant = catalog(name)?;
        let cache = paged(s, variant, rng)?;
        let shapes: Vec<Contexts> = [2048, 8192, LONG]
            .into_iter()
            .map(Contexts::Uniform)
            .chain((!cache.fp8).then_some(Contexts::Mixed))
            .collect();
        // (rows, bucket): the 32-row reference, the wide buckets, and 33 and
        // 65 rows padded into the 64 and 128 buckets.
        let mut shapes_rows: Vec<(usize, u32)> = std::iter::once(32)
            .chain(conc::ROWS)
            .map(|rows| (rows, rows as u32))
            .collect();
        if !cache.fp8 {
            shapes_rows.extend([(33, 64), (65, 128)]);
        }
        for &(rows, bucket) in &shapes_rows {
            for &contexts in &shapes {
                if rows != bucket as usize && matches!(contexts, Contexts::Mixed) {
                    continue;
                }
                let result = (|| -> LabResult<()> {
                    let mut step = decode_step(s, variant, &contexts.rows(rows), rng)?;
                    let planned = variant
                        .plan_folded(
                            bucket,
                            step.counts,
                            step.buffers.clone(),
                            RopeInputs {
                                qkv: Q,
                                theta: 1e7,
                                v_multiplier: 0.707,
                            },
                        )
                        .map_err(|e| e.to_string())?;
                    let label = format!("{name}_rows{rows}_bucket{bucket}_{}", contexts.label());
                    let extra = |split: u32| {
                        BTreeMap::from([
                            ("bucket".into(), json!(bucket)),
                            ("contexts".into(), json!(contexts.label())),
                            ("tuned_split".into(), json!(split == planned.split_capacity)),
                        ])
                    };
                    let tuned = match measure(
                        s,
                        module,
                        variant,
                        &planned,
                        &mut step,
                        (&cache.k, &cache.v),
                        &label,
                        extra(planned.split_capacity),
                    ) {
                        Ok(output) => Some(output),
                        Err(error) => {
                            failures.check(&label, Err(error));
                            None
                        }
                    };
                    let swept = !cache.fp8
                        && rows == bucket as usize
                        && rows > 32
                        && !matches!(contexts, Contexts::Uniform(8192));
                    for split in sweep
                        .into_iter()
                        .filter(|&split| swept && split != planned.split_capacity)
                    {
                        let result = (|| -> LabResult<()> {
                            let planned = with_splits(planned.clone(), variant, bucket, split)?;
                            let operation = format!("{label}_split{split}");
                            let output = measure(
                                s,
                                module,
                                variant,
                                &planned,
                                &mut step,
                                (&cache.k, &cache.v),
                                &operation,
                                extra(split),
                            )?;
                            if let Some(tuned) = &tuned {
                                close(&operation, tuned, &output, 4.0)?;
                            } else {
                                return Err("tuned split failed; comparison unavailable".into());
                            }
                            Ok(())
                        })();
                        failures.check(&format!("{label} split={split}"), result);
                    }
                    Ok(())
                })();
                failures.check(&format!("{name} rows={rows} {}", contexts.label()), result);
            }
        }
    }
    // Windowed rows: one CTA per row and KV head over at most 128 keys.
    let variant = catalog("mimo_swa_decode")?;
    if !selected(variant.name) {
        return failures.finish();
    }
    let query = variant.query;
    let ring_tokens = 256 * query.page_size as usize * query.kv_heads as usize;
    let k = upload(s, &rng.bf16_bytes(ring_tokens * query.qk_dim as usize))?;
    let v = upload(s, &rng.bf16_bytes(ring_tokens * query.v_dim as usize))?;
    for (rows, bucket) in [(32, 32), (64, 64), (65, 128), (128, 128), (256, 256)] {
        let result = (|| -> LabResult<()> {
            let mut step = decode_step(s, variant, &vec![2048; rows], rng)?;
            let planned = variant
                .plan_folded(
                    bucket,
                    step.counts,
                    step.buffers.clone(),
                    RopeInputs {
                        qkv: Q,
                        theta: 1e4,
                        v_multiplier: 0.707,
                    },
                )
                .map_err(|e| e.to_string())?;
            measure(
                s,
                module,
                variant,
                &planned,
                &mut step,
                (&k, &v),
                &format!("mimo_swa_decode_rows{rows}_bucket{bucket}_ctx2048"),
                BTreeMap::from([("bucket".into(), json!(bucket))]),
            )?;

            Ok(())
        })();
        failures.check(&format!("ring rows={rows}"), result);
    }
    failures.finish()
}

fn verify(s: &Session, module: &Module, rng: &mut Rng) -> LabResult<()> {
    let mut failures = conc::replay::Failures::default();
    for name in ["mimo_ga_bf16_verify", "mimo_swa_verify"] {
        if !selected(name) {
            continue;
        }
        let variant = catalog(name)?;
        assert_eq!(variant.query.phase, AttentionPhase::Verify);
        let query = variant.query;
        let (k, v) = if variant.layout == CacheLayout::Ring {
            let tokens = 256 * query.page_size as usize * query.kv_heads as usize;
            (
                upload(s, &rng.bf16_bytes(tokens * query.qk_dim as usize))?,
                upload(s, &rng.bf16_bytes(tokens * query.v_dim as usize))?,
            )
        } else {
            let cache = paged(s, variant, rng)?;
            (cache.k, cache.v)
        };
        for (items, tokens) in [(32, 8), (64, 4), (128, 2), (256, 1)] {
            for prefix in [2048, 8192] {
                let result = (|| -> LabResult<()> {
                    let mut step = verify_step(s, variant, items, tokens, prefix, rng)?;
                    let bucket = items as u32;
                    let planned = variant
                        .plan_for_bucket(bucket, step.counts, step.buffers.clone())
                        .map_err(|e| e.to_string())?;
                    measure(
                        s,
                        module,
                        variant,
                        &planned,
                        &mut step,
                        (&k, &v),
                        &format!("{name}_items{items}_tokens{tokens}_prefix{prefix}"),
                        BTreeMap::from([
                            ("bucket".into(), json!(bucket)),
                            ("query_tokens".into(), json!(items * tokens)),
                            ("prefix".into(), json!(prefix)),
                        ]),
                    )?;

                    Ok(())
                })();
                failures.check(&format!("{name} items={items} prefix={prefix}"), result);
            }
        }
    }
    failures.finish()
}

#[test]
fn m5_mimo_attention_wide_batches() {
    if !conc::enabled("m5") {
        return;
    }
    gpu_test(20 << 30, |s| {
        conc::exclusive(s)?;
        let module = s
            .context
            .load_module(include_bytes!(concat!(
                env!("OUT_DIR"),
                "/attention_mimo_sm120.cubin"
            )))
            .map_err(|e| e.to_string())?;
        let mut rng = Rng(0x5eed_c0c0_a77e_0005);
        MEASURED.store(0, Ordering::Relaxed);
        let mut failures = conc::replay::Failures::default();
        failures.check("decode", decode(s, &module, &mut rng));
        failures.check("verify", verify(s, &module, &mut rng));
        failures.finish()?;
        if MEASURED.load(Ordering::Relaxed) == 0 {
            return Err("attention filter selected no cases".into());
        }
        Ok(())
    })
    .expect("M5 attention")
    .expect("GPU required");
}

#[test]
fn split_piece_contract_cpu() {
    assert_eq!(split_pieces(&[1, 4, 1], 3, 256, 1).unwrap(), [1, 2, 1]);
    assert_eq!(split_pieces(&[4, 4], 3, 2, 1).unwrap(), [2, 2]);
    assert_eq!(split_pieces(&[9, 100], 64, 47, 2).unwrap(), [5, 34]);
    assert_eq!(split_pieces(&[1, 1, 1], 8, 256, 1).unwrap(), [1, 1, 1]);
    assert!(split_pieces(&[0], 3, 256, 1).is_err());
    assert!(split_pieces(&[1], 1, 256, 1).is_err());
    assert!(split_pieces(&[1], 3, 256, 0).is_err());
    let pieces = split_pieces(&[1, 4, 1], 3, 256, 1).unwrap();
    let map = WriteMap::of(DType::F32)
        .byte_ranges(
            pieces
                .iter()
                .enumerate()
                .map(|(row, &n)| row * 12..row * 12 + n * 4),
        )
        .unwrap();
    let mut bytes = vec![0xff; 36];
    for range in map.ranges() {
        bytes[range.clone()].fill(0);
    }
    use kernel_lab::poison::{Fill, check};
    assert!(check("pieces", &bytes, &map, Fill::Pattern(&[0xff])).is_ok());
    bytes[16..20].fill(0xff);
    assert!(check("missing piece", &bytes, &map, Fill::Pattern(&[0xff])).is_err());
    bytes[16..20].fill(0);
    bytes[4] = 0;
    assert!(check("unused piece slot", &bytes, &map, Fill::Pattern(&[0xff])).is_err());
}
