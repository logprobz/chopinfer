use crate::pool_offsets::{self, SentinelPool};
use kernel_core::codec::{SmallFloat, bf16_to_f32, f32_to_bf16};
use kernel_core::contract::{
    Argument, ArgumentList, AttentionPhase, DType, LaunchPlan, RuntimeCounts,
};
use kernel_core::kv_fp8;
use kernel_core::oracle::{self, PagedAttention, RingAttention};
use kernel_lab::launch::{self, Bindings};
use kernel_lab::poison::{self, Evidence, Fill, POISON, Replay, WriteMap};
use kernel_lab::{LabResult, Session, check_output, gpu_test, time};
use kernels_sm120::Profile;
use kernels_sm120::attention::{self, Buffers, CacheLayout, RopeInputs, Variant};

use runtime_cuda::{DeviceBuffer, Function, KernelArgument, Module};
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;

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
const QKV: u64 = 16;
const POSITIONS: u64 = 17;
const TOKEN_BATCH: u64 = 18;

const ROW_CUBIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/row_engine.cubin"));

#[path = "attention_chunks.rs"]
mod chunks;
#[path = "attention_composition.rs"]
mod composition;
#[path = "attention_invariance.rs"]
pub(crate) mod invariance;
#[path = "attention_live.rs"]
mod live;
#[path = "qwen38_attention.rs"]
mod qwen38;
#[path = "attention_verify_composition.rs"]
mod verify_composition;

#[derive(Clone, Copy)]
struct Case {
    variant: Variant,
    batch: usize,
    rows: usize,
    prefix: usize,
    /// Row b attends `context() >> (b % 7)` tokens instead of `context()`.
    /// Items of up to 8 query rows (verify) also hold `rows - b % rows`
    /// tokens, packed without gaps.
    mixed: bool,
}

impl Case {
    fn context(self) -> usize {
        self.prefix + self.rows
    }
    fn contexts(self) -> Vec<usize> {
        (0..self.batch)
            .map(|b| {
                if self.mixed {
                    (self.context() >> (b % 7)).max(self.rows)
                } else {
                    self.context()
                }
            })
            .collect()
    }
    fn label(self) -> String {
        format!(
            "{}_b{}_q{}_prefix{}{}",
            self.variant.name,
            self.batch,
            self.rows,
            self.prefix,
            if self.mixed { "_mixed" } else { "" }
        )
    }
}

struct Fixture {
    buffers: BTreeMap<u64, DeviceBuffer>,
    arguments: Buffers,
    planned: attention::PlannedAttention,
    reference: oracle::OracleOutput,
    /// Packed query tokens covered by `reference`; `None` checks every token.
    sampled: Option<Vec<usize>>,
    output_len: usize,
    logical_bytes: u64,
    /// Each paged sequence's physical pages, in logical order.
    page_lists: Vec<Vec<usize>>,
    /// Each ring sequence's slot.
    slot_ids: Vec<usize>,
    /// The variant, bucket and counts `planned` was planned with.
    variant: Variant,
    bucket: u32,
    counts: RuntimeCounts,
}

/// Paged cases above this many query-key pairs check a sample of tokens.
const FULL_ORACLE_PAIRS: usize = 2048 * 4096;
const SAMPLED_TOKENS: usize = 48;

impl Fixture {
    fn bindings(&self) -> Bindings<'_> {
        Bindings {
            buffers: self
                .buffers
                .iter()
                .map(|(&id, buffer)| (id, buffer))
                .collect(),
        }
    }
    fn output(&self, session: &Session) -> LabResult<Vec<u16>> {
        session.download_bf16(&self.buffers[&OUT], self.output_len)
    }
    fn launch(&self, session: &Session, partial: &Function, merge: &Function) -> LabResult<()> {
        self.launch_with_cache(session, partial, merge, None)
    }
    fn launch_with_cache(
        &self,
        session: &Session,
        partial: &Function,
        merge: &Function,
        cache: Option<(&DeviceBuffer, &DeviceBuffer)>,
    ) -> LabResult<()> {
        self.launch_planned(session, &self.planned, partial, merge, cache)
    }
    fn cache_bindings<'a>(
        &'a self,
        cache: Option<(&'a DeviceBuffer, &'a DeviceBuffer)>,
    ) -> Bindings<'a> {
        let mut bindings = self.bindings();
        if let Some((k, v)) = cache {
            bindings.buffers.insert(K, k);
            bindings.buffers.insert(V, v);
        }
        bindings
    }
    fn launch_planned(
        &self,
        session: &Session,
        planned: &attention::PlannedAttention,
        partial: &Function,
        merge: &Function,
        cache: Option<(&DeviceBuffer, &DeviceBuffer)>,
    ) -> LabResult<()> {
        let bindings = self.cache_bindings(cache);
        launch::launch(
            &session.context,
            partial,
            &session.stream,
            planned.partial,
            &planned.partial_args,
            &bindings,
        )?;
        if planned.direct_output {
            return Ok(());
        }
        launch::launch(
            &session.context,
            merge,
            &session.stream,
            planned.merge,
            &planned.merge_args,
            &bindings,
        )
    }
    /// The planned partial and merge with some buffers rebound, separately
    /// for each launch: test-only wiring for graphs that write elsewhere.
    fn launch_rebound<'a>(
        &'a self,
        session: &Session,
        partial: &Function,
        merge: &Function,
        partial_buffers: &[(u64, &'a DeviceBuffer)],
        merge_buffers: &[(u64, &'a DeviceBuffer)],
    ) -> LabResult<()> {
        let mut partial_bindings = self.bindings();
        partial_bindings
            .buffers
            .extend(partial_buffers.iter().copied());
        let mut merge_bindings = self.bindings();
        merge_bindings.buffers.extend(merge_buffers.iter().copied());
        launch::launch(
            &session.context,
            partial,
            &session.stream,
            self.planned.partial,
            &self.planned.partial_args,
            &partial_bindings,
        )?;
        if self.planned.direct_output {
            return Ok(());
        }
        launch::launch(
            &session.context,
            merge,
            &session.stream,
            self.planned.merge,
            &self.planned.merge_args,
            &merge_bindings,
        )
    }
    /// Checks for the output, and the workspace a merge reads, poisoned
    /// before every launch.
    fn replay<'a>(&'a self, session: &'a Session) -> LabResult<Replay<'a>> {
        let out = &self.buffers[&OUT];
        Ok(Replay::new(session)
            .output(
                "attention output",
                out,
                WriteMap::of(DType::Bf16).byte_prefix(out.byte_len())?,
            )
            .scratch("attention workspace", &self.buffers[&WORK]))
    }
}

const GRAPH_LAYERS: usize = 48;

fn random_word(state: &mut u64) -> u32 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    (*state >> 32) as u32
}

fn samples(count: usize, state: &mut u64) -> Vec<f32> {
    (0..count)
        .map(|_| ((random_word(state) % 33) as i32 - 16) as f32 / 64.0)
        .collect()
}

fn put_bytes(session: &Session, bytes: &[u8]) -> LabResult<DeviceBuffer> {
    let buffer = session.allocate(bytes.len().max(1))?;
    if !bytes.is_empty() {
        buffer
            .copy_from_bytes(bytes, &session.stream)
            .map_err(|e| e.to_string())?;
    }
    Ok(buffer)
}

fn put_i32(session: &Session, values: &[i32]) -> LabResult<DeviceBuffer> {
    put_bytes(
        session,
        &values
            .iter()
            .flat_map(|v| v.to_ne_bytes())
            .collect::<Vec<_>>(),
    )
}

fn encode(values: &[f32], fp8: bool, scale: f32) -> (Vec<u8>, Vec<f64>) {
    let mut bytes = Vec::with_capacity(values.len() * if fp8 { 1 } else { 2 });
    let mut decoded = Vec::with_capacity(values.len());
    let fp8_codes: Vec<u8> = (-16..=16)
        .map(|n| kv_fp8::encode(n as f32, 64.0 * scale))
        .collect();
    for &value in values {
        if fp8 {
            let index = (value * 64.0) as i32 + 16;
            let code = fp8_codes[index as usize];
            bytes.push(code);
            decoded.push(kv_fp8::decode(code, scale));
        } else {
            let code = f32_to_bf16(value);
            bytes.extend_from_slice(&code.to_ne_bytes());
            decoded.push(f64::from(bf16_to_f32(code)));
        }
    }
    (bytes, decoded)
}

fn prepare(session: &Session, case: Case) -> LabResult<Fixture> {
    prepare_with_page_layers(session, case, 1)
}

/// A sink far above every key score with V amplified by an exact power of
/// two, so the output stays a normal BF16 value.
#[derive(Clone, Copy)]
struct SinkStress {
    sink: f32,
    v_scale: f32,
}

fn prepare_with_page_layers(
    session: &Session,
    case: Case,
    page_layers: usize,
) -> LabResult<Fixture> {
    prepare_stressed(session, case, page_layers, None)
}

fn prepare_stressed(
    session: &Session,
    case: Case,
    page_layers: usize,
    stress: Option<SinkStress>,
) -> LabResult<Fixture> {
    Ok(prepare_values(session, case, page_layers, stress, None)?.0)
}

/// Uniform BF16 values in [-0.75, 0.75] with random mantissas, which FP8
/// E4M3 cannot hold exactly.
fn general(count: usize, state: &mut u64) -> Vec<f32> {
    (0..count)
        .map(|_| {
            let unit = f64::from(random_word(state)) / f64::from(u32::MAX);
            bf16_to_f32(f32_to_bf16(((2.0 * unit - 1.0) * 0.75) as f32))
        })
        .collect()
}

/// The rounded fixtures' values spread over their range; a constant would
/// make attention exact and the FP8 check vacuous.
#[test]
fn general_values_spread_over_their_range() {
    let mut state = 0xadb3_d519_3317_62a5;
    let values = general(4096, &mut state);
    let mean = values.iter().map(|&v| f64::from(v)).sum::<f64>() / 4096.0;
    let rms = (values.iter().map(|&v| f64::from(v).powi(2)).sum::<f64>() / 4096.0).sqrt();
    assert!(
        mean.abs() < 0.05 && rms > 0.4 && rms < 0.5,
        "mean {mean} rms {rms}"
    );
    let distinct: std::collections::BTreeSet<u32> = values.iter().map(|v| v.to_bits()).collect();
    assert!(distinct.len() > 1000);
}

/// FP8 E4M3 codes of `value / scale` rounded to nearest, as the KV-write
/// kernels store them, and their decoded values.
fn encode_rounded(values: &[f32], scale: f32) -> (Vec<u8>, Vec<f64>) {
    values
        .iter()
        .map(|&value| {
            let code = kv_fp8::encode(value, scale);
            (code, kv_fp8::decode(code, scale))
        })
        .unzip()
}

/// The unrounded inputs of a fixture whose FP8 K and V were rounded from
/// general BF16 values.
struct Rounded {
    q: Vec<f64>,
    k: Vec<f64>,
    v: Vec<f64>,
    sink: Vec<f64>,
    lens: Vec<usize>,
    prefixes: Vec<usize>,
    scales: [f32; 2],
}

/// With `rounded`, K and V are general BF16 values stored as FP8 with that
/// scale; the fixture's reference is the oracle over the stored codes.
fn prepare_values(
    session: &Session,
    case: Case,
    page_layers: usize,
    stress: Option<SinkStress>,
    rounded: Option<[f32; 2]>,
) -> LabResult<(Fixture, Option<Rounded>)> {
    let mut v = case.variant;
    let decode = v.query.phase == AttentionPhase::Decode;
    let verify = v.query.phase == AttentionPhase::Verify;
    if !decode && !verify {
        v.query_capacity = (case.batch * case.rows) as u32;
    }
    let ring = v.layout == CacheLayout::Ring;
    let fp8 = v.query.kv_dtype == DType::Fp8E4M3;
    let batch = case.batch;
    let rows = case.rows;
    let context = case.context();
    let contexts = case.contexts();
    let qh = v.query.q_heads as usize;
    let kh = v.query.kv_heads as usize;
    let qd = v.query.qk_dim as usize;
    let vd = v.query.v_dim as usize;
    let page = v.query.page_size as usize;
    let slots = if ring {
        batch
    } else {
        contexts.iter().map(|c| c.div_ceil(page)).sum()
    };
    let capacity_tokens = slots * page;
    let mut state = 0xadb3_d519_3317_62a5_u64 ^ (context as u64) ^ ((batch as u64) << 24);
    let mut q_values = samples(batch * rows * qh * qd, &mut state);
    if v.query.softcap.is_some() {
        q_values.iter_mut().for_each(|value| *value *= 16.0);
    }
    let (q_bytes, q_ref) = encode(&q_values, false, 1.0);
    let mut k_values = samples(capacity_tokens * kh * qd, &mut state);
    if v.query.softcap.is_some() {
        k_values.iter_mut().for_each(|value| *value *= 16.0);
    }
    let v_amplify = stress.map_or(1.0, |stress| stress.v_scale);
    let mut v_values: Vec<f32> = samples(capacity_tokens * kh * vd, &mut state)
        .into_iter()
        .map(|value| value * v_amplify)
        .collect();
    // FP8 K and V scales differ, so a swapped pair cannot pass.
    let ([k_scale, v_scale], (k_bytes, k_ref), (v_bytes, v_ref)) = match rounded {
        Some([k_scale, v_scale]) if fp8 => {
            k_values = general(k_values.len(), &mut state);
            v_values = general(v_values.len(), &mut state);
            (
                [k_scale, v_scale],
                encode_rounded(&k_values, k_scale),
                encode_rounded(&v_values, v_scale),
            )
        }
        Some(_) => return Err("rounded KV needs an FP8 variant".into()),
        None => {
            let [k_scale, v_scale] = if fp8 { [0.5, 0.25] } else { [1.0, 1.0] };
            (
                [k_scale, v_scale],
                encode(&k_values, fp8, k_scale),
                encode(&v_values, fp8, v_scale),
            )
        }
    };
    let k_page_bytes = page * kh * qd * if fp8 { 1 } else { 2 };
    let v_page_bytes = page * kh * vd * if fp8 { 1 } else { 2 };
    let page_stride = page_layers * (k_page_bytes + v_page_bytes);
    let k_stride = if ring || page_layers == 1 {
        k_page_bytes
    } else {
        page_stride
    };
    let v_stride = if ring || page_layers == 1 {
        v_page_bytes
    } else {
        page_stride
    };
    let scatter_pages = |source: &[u8], block_bytes: usize, stride: usize| {
        if stride == block_bytes {
            return source.to_vec();
        }
        let mut storage = vec![0xa5; slots * stride];
        for physical in 0..slots {
            let source_range = physical * block_bytes..(physical + 1) * block_bytes;
            let target = physical * stride;
            storage[target..target + block_bytes].copy_from_slice(&source[source_range]);
        }
        storage
    };
    let k_storage = scatter_pages(&k_bytes, k_page_bytes, k_stride);
    let v_storage = scatter_pages(&v_bytes, v_page_bytes, v_stride);
    let new_values_k = samples(batch * rows * kh * qd, &mut state);
    let new_values_v: Vec<f32> = samples(batch * rows * kh * vd, &mut state)
        .into_iter()
        .map(|value| value * v_amplify)
        .collect();
    let (new_k_bytes, new_k_ref) = encode(&new_values_k, false, 1.0);
    let (new_v_bytes, new_v_ref) = encode(&new_values_v, false, 1.0);
    let sink_values: Vec<f32> = (0..qh)
        .map(|h| stress.map_or((h as f32 - 8.0) / 16.0, |stress| stress.sink))
        .collect();
    let (sink_bytes, sink_ref) = encode(&sink_values, false, 1.0);
    let max_pages = (v.query.max_context_capacity as usize).div_ceil(page);
    let mut tables = if ring {
        Vec::new()
    } else {
        vec![0_i32; batch * max_pages]
    };
    let mut page_lists = Vec::new();
    if !ring {
        let mut base = 0;
        for (b, &row_context) in contexts.iter().enumerate() {
            let pages_per_seq = row_context.div_ceil(page);
            let list: Vec<_> = (0..pages_per_seq)
                .map(|p| base + pages_per_seq - p - 1)
                .collect();
            for (p, &physical) in list.iter().enumerate() {
                tables[b * max_pages + p] = physical as i32;
            }
            page_lists.push(list);
            base += pages_per_seq;
        }
    }
    let prefixes: Vec<usize> = contexts.iter().map(|c| c - rows).collect();
    let lens: Vec<usize> = (0..batch)
        .map(|b| {
            if case.mixed && !decode && rows <= 8 {
                rows - b % rows
            } else {
                rows
            }
        })
        .collect();
    let tokens: usize = lens.iter().sum();
    let key_lengths: Vec<usize> = prefixes.iter().zip(&lens).map(|(p, l)| p + l).collect();
    let slot_ids: Vec<usize> = (0..batch).map(|b| batch - b - 1).collect();
    let seq_lens: Vec<i32> = contexts.iter().map(|&c| c as i32).collect();
    let q_start: Vec<i32> = lens
        .iter()
        .scan(0, |next, &len| {
            let start = *next;
            *next += len;
            Some(start as i32)
        })
        .collect();
    let q_len: Vec<i32> = lens.iter().map(|&len| len as i32).collect();
    let prefix_len: Vec<i32> = prefixes.iter().map(|&p| p as i32).collect();
    let mut buffers = BTreeMap::new();
    buffers.insert(Q, put_bytes(session, &q_bytes)?);
    buffers.insert(K, put_bytes(session, &k_storage)?);
    buffers.insert(V, put_bytes(session, &v_storage)?);
    buffers.insert(TABLE, put_i32(session, &tables)?);
    buffers.insert(
        SLOT,
        put_i32(
            session,
            &slot_ids.iter().map(|&s| s as i32).collect::<Vec<_>>(),
        )?,
    );
    buffers.insert(SEQ, put_i32(session, &seq_lens)?);
    buffers.insert(START, put_i32(session, &q_start)?);
    buffers.insert(LEN, put_i32(session, &q_len)?);
    buffers.insert(PREFIX, put_i32(session, &prefix_len)?);
    buffers.insert(LIVE, put_i32(session, &[batch as i32])?);
    buffers.insert(K_NEW, put_bytes(session, &new_k_bytes)?);
    buffers.insert(V_NEW, put_bytes(session, &new_v_bytes)?);
    buffers.insert(SINK, put_bytes(session, &sink_bytes)?);
    let output_len = tokens * qh * vd;
    let out = session.allocate(output_len * 2)?;
    poison::poison(session, &[&out])?;
    buffers.insert(OUT, out);
    let (k_map, v_map) = if decode {
        (None, None)
    } else {
        let (k, v) = v
            .cache_maps_with_strides(slots as u32, K, V, k_stride as u64, v_stride as u64)
            .map_err(|e| e.to_string())?;
        (Some(k), Some(v))
    };
    let arguments = Buffers {
        q: Q,
        k_cache: K,
        v_cache: V,
        k_page_stride_bytes: k_stride as u64,
        v_page_stride_bytes: v_stride as u64,
        k_map,
        v_map,
        block_tables: (!ring).then_some(TABLE),
        slot_ids: ring.then_some(SLOT),
        seq_lens: Some(SEQ),
        q_start: (!decode).then_some(START),
        q_len: (!decode).then_some(LEN),
        prefix_len: (!decode).then_some(PREFIX),
        live_batch: LIVE,
        k_new: (ring && !decode).then_some(K_NEW),
        v_new: (ring && !decode).then_some(V_NEW),
        sink: v.query.sink.then_some(SINK),
        output: OUT,
        workspace: WORK,
        k_scale,
        v_scale,
    };
    let bucket = batch.next_power_of_two() as u32;
    let counts = RuntimeCounts {
        m: tokens as u32,
        batch: batch as u32,
        context: context as u32,
        routed_rows: 0,
        split_k: 1,
    };
    let planned = v
        .plan_for_bucket(bucket, counts, arguments.clone())
        .map_err(|e| e.to_string())?;
    let work = session.allocate(planned.partial.workspace_bytes as usize)?;
    poison::poison(session, &[&work])?;
    buffers.insert(WORK, work);
    let sample = (!ring && batch == 1 && rows * context > FULL_ORACLE_PAIRS).then(|| {
        (0..SAMPLED_TOKENS)
            .map(|i| i * (rows - 1) / (SAMPLED_TOKENS - 1))
            .collect::<Vec<_>>()
    });
    let reference = if let Some(tokens) = &sample {
        // Each sampled token is a one-row oracle batch over the same pages.
        let row = qh * qd;
        let mut reference = oracle::OracleOutput {
            values: Vec::new(),
            sum_abs_products: Vec::new(),
            reduction_lengths: Vec::new(),
        };
        for chunk in tokens.chunks(v.query.max_batch_capacity as usize) {
            let q_rows: Vec<f64> = chunk
                .iter()
                .flat_map(|&t| q_ref[t * row..(t + 1) * row].iter().copied())
                .collect();
            let positions: Vec<usize> = chunk.iter().map(|&t| case.prefix + t).collect();
            let part = oracle::attention(
                &v.query,
                &PagedAttention {
                    q: &q_rows,
                    keys: &k_ref,
                    values: &v_ref,
                    page_tables: &vec![page_lists[0].clone(); chunk.len()],
                    query_lengths: &vec![1; chunk.len()],
                    query_starts: &positions,
                    key_lengths: &positions.iter().map(|p| p + 1).collect::<Vec<_>>(),
                    sink_logits: v.query.sink.then_some(sink_ref.as_slice()),
                },
            )
            .map_err(|e| e.to_string())?;
            reference.values.extend(part.values);
            reference.sum_abs_products.extend(part.sum_abs_products);
            reference.reduction_lengths.extend(part.reduction_lengths);
        }
        reference
    } else if ring {
        oracle::ring_attention(
            &v.query,
            &RingAttention {
                q: &q_ref[..tokens * qh * qd],
                keys: &k_ref,
                values: &v_ref,
                slot_ids: &slot_ids,
                query_lengths: &lens,
                prefix_lengths: &prefixes,
                k_new: (!decode).then_some(&new_k_ref[..tokens * kh * qd]),
                v_new: (!decode).then_some(&new_v_ref[..tokens * kh * vd]),
                sink_logits: v.query.sink.then_some(sink_ref.as_slice()),
            },
        )
        .map_err(|e| e.to_string())?
    } else {
        oracle::attention(
            &v.query,
            &PagedAttention {
                q: &q_ref[..tokens * qh * qd],
                keys: &k_ref,
                values: &v_ref,
                page_tables: &page_lists,
                query_lengths: &lens,
                query_starts: &prefixes,
                key_lengths: &key_lengths,
                sink_logits: v.query.sink.then_some(sink_ref.as_slice()),
            },
        )
        .map_err(|e| e.to_string())?
    };
    let window = v
        .query
        .window_left
        .map_or(usize::MAX, |left| left as usize + 1);
    let visible: usize = key_lengths
        .iter()
        .zip(&lens)
        .map(|(&c, &len)| if ring { c.min(window + len - 1) } else { c })
        .sum();
    let logical_bytes =
        (visible * kh * (qd + vd) * if fp8 { 1 } else { 2 } + tokens * qh * (qd + vd) * 2) as u64;
    let rounded = rounded.map(|scales| Rounded {
        q: q_ref,
        k: k_values.iter().map(|&x| f64::from(x)).collect(),
        v: v_values.iter().map(|&x| f64::from(x)).collect(),
        sink: sink_ref,
        lens,
        prefixes,
        scales,
    });
    Ok((
        Fixture {
            buffers,
            arguments,
            planned,
            reference,
            sampled: sample,
            output_len,
            logical_bytes,
            page_lists,
            slot_ids,
            variant: v,
            bucket,
            counts,
        },
        rounded,
    ))
}

/// The attention of the unrounded keys and values of every packed token and
/// head, and each output element's E4M3 storage-error standard deviation.
fn fp8_exact(case: Case, fixture: &Fixture, rounded: &Rounded) -> (Vec<f64>, Vec<f64>) {
    let query = case.variant.query;
    let (qh, kh) = (query.q_heads as usize, query.kv_heads as usize);
    let (qd, vd) = (query.qk_dim as usize, query.v_dim as usize);
    let page = query.page_size as usize;
    let [k_scale, v_scale] = rounded.scales.map(f64::from);
    let (mut exact, mut sigma) = (Vec::new(), Vec::new());
    let mut token = 0;
    for (b, (&len, &prefix)) in rounded.lens.iter().zip(&rounded.prefixes).enumerate() {
        for row in 0..len {
            let visible = prefix + row + 1;
            for head in 0..qh {
                let kv_head = head / (qh / kh);
                let (mut keys, mut values) = (Vec::new(), Vec::new());
                for position in 0..visible {
                    let slot = (fixture.page_lists[b][position / page] * page + position % page)
                        * kh
                        + kv_head;
                    keys.extend_from_slice(&rounded.k[slot * qd..(slot + 1) * qd]);
                    values.extend_from_slice(&rounded.v[slot * vd..(slot + 1) * vd]);
                }
                let (output, deviation) =
                    kernel_core::kv_fp8::attention_sigma(kernel_core::kv_fp8::Fp8Attention {
                        query: &rounded.q[(token * qh + head) * qd..][..qd],
                        keys: &keys,
                        values: &values,
                        qk_dim: qd,
                        v_dim: vd,
                        softmax_scale: f64::from(query.softmax_scale),
                        k_scale,
                        v_scale,
                        sink: query.sink.then(|| rounded.sink[head]),
                    });
                exact.extend(output);
                sigma.extend(deviation);
            }
            token += 1;
        }
    }
    (exact, sigma)
}

fn module(session: &Session, profile: Profile) -> LabResult<Module> {
    let bytes: &[u8] = match profile {
        model_qwen38_contract::sm120::config::PROFILE => {
            include_bytes!(concat!(env!("OUT_DIR"), "/attention_qwen38_sm120.cubin"))
        }
        model_mimo26_contract::sm120::config::PROFILE => {
            include_bytes!(concat!(env!("OUT_DIR"), "/attention_mimo_sm120.cubin"))
        }
        p if p == model_mimo26_contract::sm120::config::MODEL.at(2) => {
            include_bytes!(concat!(env!("OUT_DIR"), "/attention_mimo_tp2_sm120.cubin"))
        }
        Profile::StdDenseMoe => {
            include_bytes!(concat!(env!("OUT_DIR"), "/attention_std_sm120.cubin"))
        }
        // Qwen's TP2 cubin holds only its gated rows.
        p if p == model_qwen38_contract::sm120::config::MODEL.at(2) => {
            include_bytes!(concat!(
                env!("OUT_DIR"),
                "/attention_qwen38_tp2_sm120.cubin"
            ))
        }
        p if kernel_lab::lab_shapes::PROFILES.contains(&p) => {
            return session.module(&format!("attention_{}_sm120", p.tag()));
        }
        p => return Err(format!("no attention cubin for {}", p.as_str())),
    };
    session
        .context
        .load_module(bytes)
        .map_err(|e| e.to_string())
}

fn functions(
    module: &Module,
    variant: Variant,
    planned: &attention::PlannedAttention,
) -> LabResult<(Function, Function)> {
    let partial = module
        .function(&planned.partial_symbol)
        .map_err(|e| e.to_string())?;
    let merge = module
        .function(&format!("chop_attn_{}_merge_v1", variant.name))
        .map_err(|e| e.to_string())?;
    for (function, plan) in [(&partial, planned.partial), (&merge, planned.merge)] {
        if plan.dynamic_shared_bytes > 48 * 1024 {
            function
                .set_max_dynamic_shared_memory(plan.dynamic_shared_bytes)
                .map_err(|e| e.to_string())?;
        }
    }
    Ok((partial, merge))
}

fn validate(
    session: &Session,
    case: Case,
    fixture: &Fixture,
    actual: &[u16],
    phase: &str,
) -> LabResult<()> {
    // Sampled cases check only some tokens against the oracle; every token
    // of the poisoned output must still be written.
    let written = actual
        .iter()
        .flat_map(|bits| bits.to_ne_bytes())
        .collect::<Vec<_>>();
    poison::check(
        &format!("{} {phase} output", case.label()),
        &written,
        &WriteMap::of(DType::Bf16).byte_prefix(written.len())?,
        Fill::Pattern(&[POISON]),
    )?;
    let mut receipt = BTreeMap::new();
    receipt.insert("case".into(), json!(case.label()));
    receipt.insert("phase".into(), json!(phase));
    receipt.insert("profile".into(), json!(case.variant.profile.as_str()));
    receipt.insert("layout".into(), json!(format!("{:?}", case.variant.layout)));
    receipt.insert(
        "split_capacity".into(),
        json!(fixture.planned.split_capacity),
    );
    let row = (case.variant.query.q_heads * case.variant.query.v_dim) as usize;
    let actual_f32: Vec<f32> = match &fixture.sampled {
        Some(tokens) => tokens
            .iter()
            .flat_map(|&t| actual[t * row..(t + 1) * row].iter())
            .map(|&bits| bf16_to_f32(bits))
            .collect(),
        None => actual.iter().map(|&bits| bf16_to_f32(bits)).collect(),
    };
    if let Some(tokens) = &fixture.sampled {
        receipt.insert("sampled_tokens".into(), json!(tokens.len()));
    }
    let result = check_output(&actual_f32, &fixture.reference, DType::Bf16, &mut receipt);
    session.receipt("attention", receipt.clone())?;
    result.map_err(|e| format!("{}: {e}; worst={}", case.label(), receipt["worst_element"]))
}

fn exact(case: Case, expected: &[u16], actual: &[u16], phase: &str) -> LabResult<()> {
    oracle::check_exact(actual, expected).map_err(|index| {
        format!(
            "{} {phase} differs at {index}: actual={:?}, eager={:?}",
            case.label(),
            actual.get(index),
            expected.get(index)
        )
    })
}

fn run(session: &Session, cases: &[Case], replay: bool) -> LabResult<()> {
    run_with_page_layers(session, cases, replay, 1)
}

fn run_with_page_layers(
    session: &Session,
    cases: &[Case],
    replay: bool,
    page_layers: usize,
) -> LabResult<()> {
    for &case in cases {
        let cubin = module(session, case.variant.profile)?;
        let fixture = prepare_with_page_layers(session, case, page_layers)?;
        let (partial, merge) = functions(&cubin, case.variant, &fixture.planned)?;
        let labelled = |error: String| format!("{}: {error}", case.label());
        let mut checks = fixture.replay(session)?;
        checks
            .eager(|| fixture.launch(session, &partial, &merge))
            .map_err(labelled)?;
        let eager = fixture.output(session)?;
        validate(session, case, &fixture, &eager, "eager")?;
        if replay {
            checks
                .relaunch(|| fixture.launch(session, &partial, &merge))
                .map_err(labelled)?;
            let graph = session.capture(|_| fixture.launch(session, &partial, &merge))?;
            checks.graph(&graph).map_err(labelled)?;
            let producer = PdlProducer::new(session)?;
            checks
                .pdl(
                    || producer.launch(session),
                    || fixture.launch(session, &partial, &merge),
                    fixture.planned.partial.pdl,
                )
                .map_err(labelled)?;
            let mut receipt = BTreeMap::new();
            receipt.insert("case".into(), json!(case.label()));
            checks.record(&mut receipt);
            session.receipt("attention", receipt)?;
        }
    }
    Ok(())
}

/// `chop_probe_pdl_first_v1`: releases its dependents at once, then spins,
/// so a PDL dependent runs its pre-wait code while the producer is live.
struct PdlProducer {
    function: Function,
    second: Function,
    stamps: DeviceBuffer,
}

impl PdlProducer {
    fn new(session: &Session) -> LabResult<Self> {
        let probe = session.module("probe_sm120")?;
        let function = probe
            .function("chop_probe_pdl_first_v1")
            .map_err(|e| e.to_string())?;
        let second = probe
            .function("chop_probe_pdl_second_v1")
            .map_err(|e| e.to_string())?;
        Ok(Self {
            function,
            second,
            stamps: session.allocate(32)?,
        })
    }
    /// Launches `chop_probe_pdl_second_v1` with PDL: it stamps
    /// %globaltimer before and after its wait.
    fn launch_second(&self, session: &Session) -> LabResult<()> {
        self.second
            .launch(
                &session.stream,
                launch::simple_config((1, 1, 1), (32, 1, 1), 0, (1, 1, 1), true)?,
                &[KernelArgument::Buffer(&self.stamps)],
            )
            .map_err(|e| e.to_string())
    }
    /// The producer's start and end and the second probe's stamps before
    /// and after its wait, in %globaltimer nanoseconds.
    fn stamps(&self, session: &Session) -> LabResult<[u64; 4]> {
        let bytes = bytes(session, &self.stamps)?;
        Ok(std::array::from_fn(|i| {
            u64::from_ne_bytes(bytes[8 * i..8 * i + 8].try_into().unwrap())
        }))
    }
    fn launch(&self, session: &Session) -> LabResult<()> {
        self.launch_for(session, 1000)
    }
    /// Launches the producer spinning `cycles` clock cycles.
    fn launch_for(&self, session: &Session, cycles: u64) -> LabResult<()> {
        self.function
            .launch(
                &session.stream,
                launch::simple_config((1, 1, 1), (32, 1, 1), 0, (1, 1, 1), false)?,
                &[
                    KernelArgument::Buffer(&self.stamps),
                    KernelArgument::U64(cycles),
                ],
            )
            .map_err(|e| e.to_string())
    }
}

/// A graph that writes elsewhere must fail the poisoned replay check, though
/// the unpoisoned comparison it replaces accepts it: once for a graph whose
/// merge writes a decoy output, once for a graph whose partial writes a
/// decoy workspace, so the merge reads poisoned partials.
#[test]
fn poisoned_replay_catches_a_graph_writing_elsewhere() {
    gpu_test(1 << 30, |session| {
        let case = Case {
            variant: named("mimo_ga_bf16_decode")?,
            batch: 1,
            rows: 1,
            prefix: 128,
            mixed: false,
        };
        let cubin = module(session, case.variant.profile)?;
        let fixture = prepare(session, case)?;
        if fixture.planned.direct_output {
            return Err("the negative control needs a partial and a merge".into());
        }
        let (partial, merge) = functions(&cubin, case.variant, &fixture.planned)?;
        let out = &fixture.buffers[&OUT];
        let work = &fixture.buffers[&WORK];
        let mut checks = fixture.replay(session)?;
        checks.eager(|| fixture.launch(session, &partial, &merge))?;
        let eager = fixture.output(session)?;
        validate(session, case, &fixture, &eager, "negative_control_eager")?;
        let decoy_out = session.allocate(out.byte_len())?;
        let decoy_work = session.allocate(work.byte_len())?;
        let elsewhere = session.capture(|_| {
            fixture.launch_rebound(
                session,
                &partial,
                &merge,
                &[(OUT, &decoy_out)],
                &[(OUT, &decoy_out)],
            )
        })?;
        let stale = session.capture(|_| {
            fixture.launch_rebound(session, &partial, &merge, &[(WORK, &decoy_work)], &[])
        })?;
        let mut receipt = BTreeMap::new();
        receipt.insert("case".into(), json!(case.label()));
        for (name, graph) in [("decoy output", &elsewhere), ("decoy workspace", &stale)] {
            // Without poison: an eager launch leaves the output and the
            // workspace current, and the graph that writes elsewhere passes.
            fixture.launch(session, &partial, &merge)?;
            graph.launch(&session.stream).map_err(|e| e.to_string())?;
            exact(case, &eager, &fixture.output(session)?, "unpoisoned")
                .map_err(|e| format!("{name}: the unpoisoned check already fails: {e}"))?;
            // Poisoned, the same graph fails.
            let error = match checks.graph(graph) {
                Ok(()) => return Err(format!("{name}: the poisoned check passed")),
                Err(error) => error,
            };
            eprintln!("{name}: poisoned replay check failed as required: {error}");
            receipt.insert(format!("{name} error"), json!(error));
        }
        exact(
            case,
            &eager,
            &session.download_bf16(&decoy_out, fixture.output_len)?,
            "decoy output",
        )?;
        receipt.insert("unpoisoned_check_passed".into(), json!(true));
        receipt.insert("poisoned_check_failed".into(), json!(true));
        session.receipt("attention_negative_control", receipt)?;
        Ok(())
    })
    .expect("poisoned replay negative control");
}

fn find_peak(ordinal: i32) -> (f64, f64, String) {
    let directory = kernel_lab::receipt_root().join("probes");
    let latest = fs::read_dir(directory)
        .ok()
        .into_iter()
        .flatten()
        .filter_map(Result::ok)
        .filter_map(|entry| {
            let value: Value = serde_json::from_slice(&fs::read(entry.path()).ok()?).ok()?;
            (value["ordinal"].as_i64() == Some(i64::from(ordinal))
                && value
                    .pointer("/dram_read/gb_per_second")
                    .and_then(Value::as_f64)
                    .is_some())
            .then_some((entry, value))
        })
        .max_by_key(|(entry, _)| entry.file_name());
    if let Some((entry, value)) = latest
        && let Some(peak) = value
            .pointer("/dram_read/gb_per_second")
            .and_then(Value::as_f64)
        && let Some(mma) = value
            .pointer("/mma_tflops/bf16/tflops")
            .and_then(Value::as_f64)
    {
        return (peak, mma, entry.path().display().to_string());
    }
    (
        1800.0,
        300.0,
        "fallback: 1.8 TB/s DRAM, 300 TFLOPS BF16".into(),
    )
}

fn benchmark(session: &Session, cases: &[Case]) -> LabResult<()> {
    let (peak_gbps, peak_tflops, peak_source) = find_peak(session.ordinal);
    let probe = session.module("probe_sm120")?;
    let tiny = probe
        .function("chop_probe_sm120_v1")
        .map_err(|e| e.to_string())?;
    let tiny_output = session.allocate(4)?;
    let launch_floor = kernel_lab::time_named(session, "launch_floor", 2, 3, |stream| {
        tiny.launch(
            stream,
            launch::simple_config((1, 1, 1), (32, 1, 1), 0, (1, 1, 1), false)?,
            &[KernelArgument::Buffer(&tiny_output)],
        )
        .map_err(|e| e.to_string())
    })?
    .median_ms
        * 1000.0;
    for &case in cases {
        if std::env::var("CHOPINFER_ATTENTION_CASE")
            .ok()
            .is_some_and(|filter| case.label() != filter)
        {
            continue;
        }
        let cubin = module(session, case.variant.profile)?;
        let fixture = prepare(session, case)?;
        let (partial, merge) = functions(&cubin, case.variant, &fixture.planned)?;
        fixture.launch(session, &partial, &merge)?;
        let actual = fixture.output(session)?;
        validate(session, case, &fixture, &actual, "benchmark_correctness")?;
        let timing =
            kernel_lab::time_named(session, &format!("{}::total", case.label()), 2, 3, |_| {
                fixture.launch(session, &partial, &merge)
            })?;
        let bindings = fixture.bindings();
        let partial_time =
            kernel_lab::time_named(session, &format!("{}::partial", case.label()), 2, 3, |_| {
                launch::launch(
                    &session.context,
                    &partial,
                    &session.stream,
                    fixture.planned.partial,
                    &fixture.planned.partial_args,
                    &bindings,
                )
            })?;
        let merge_time = if fixture.planned.direct_output {
            kernel_lab::Timing {
                median_ms: 0.0,
                min_ms: 0.0,
                rounds: 0,
            }
        } else {
            kernel_lab::time_named(session, &format!("{}::merge", case.label()), 2, 3, |_| {
                launch::launch(
                    &session.context,
                    &merge,
                    &session.stream,
                    fixture.planned.merge,
                    &fixture.planned.merge_args,
                    &bindings,
                )
            })?
        };
        let achieved_gbps =
            fixture.logical_bytes as f64 / f64::from(timing.median_ms) / 1_000_000.0;
        let mut receipt = BTreeMap::new();
        receipt.insert("case".into(), json!(case.label()));
        receipt.insert("profile".into(), json!(case.variant.profile.as_str()));
        receipt.insert("median_us".into(), json!(timing.median_ms * 1000.0));
        receipt.insert("min_us".into(), json!(timing.min_ms * 1000.0));
        receipt.insert("rounds".into(), json!(timing.rounds));
        receipt.insert("logical_bytes".into(), json!(fixture.logical_bytes));
        receipt.insert("achieved_gb_per_second".into(), json!(achieved_gbps));
        receipt.insert("peak_gb_per_second".into(), json!(peak_gbps));
        receipt.insert("peak_source".into(), json!(peak_source));
        receipt.insert("partial_us".into(), json!(partial_time.median_ms * 1000.0));
        receipt.insert("merge_us".into(), json!(merge_time.median_ms * 1000.0));
        receipt.insert("partial_grid".into(), json!(fixture.planned.partial.grid));
        receipt.insert("merge_grid".into(), json!(fixture.planned.merge.grid));
        receipt.insert("single_launch_floor_us".into(), json!(launch_floor));
        receipt.insert(
            "dominant_stage".into(),
            json!(if partial_time.median_ms >= merge_time.median_ms {
                "partial"
            } else {
                "merge"
            }),
        );
        if case.variant.query.phase == AttentionPhase::Prefill {
            let visible = if case.variant.layout == CacheLayout::Ring {
                128
            } else {
                case.context()
            };
            let keys = (0..case.rows)
                .map(|row| (case.prefix + row + 1).min(visible))
                .sum::<usize>();
            let flops = 2.0
                * (case.batch
                    * case.variant.query.q_heads as usize
                    * keys
                    * (case.variant.query.qk_dim + case.variant.query.v_dim) as usize)
                    as f64;
            receipt.insert("logical_flops".into(), json!(flops));
            receipt.insert(
                "achieved_tflops".into(),
                json!(flops / f64::from(timing.median_ms) / 1e9),
            );
            receipt.insert("mma_peak_tflops".into(), json!(peak_tflops));
            receipt.insert(
                "percent_mma_peak".into(),
                json!(flops / f64::from(timing.median_ms) / 1e9 / peak_tflops * 100.0),
            );
            receipt.insert("roofline_kind".into(), json!("causal BF16 MMA compute"));
            receipt.insert("roofline_us".into(), json!(flops / peak_tflops / 1e6));
            receipt.insert(
                "percent_peak".into(),
                json!(flops / f64::from(timing.median_ms) / 1e9 / peak_tflops * 100.0),
            );
        } else {
            receipt.insert("roofline_kind".into(), json!("DRAM read"));
            receipt.insert(
                "roofline_us".into(),
                json!(fixture.logical_bytes as f64 / peak_gbps / 1000.0),
            );
            receipt.insert(
                "percent_peak".into(),
                json!(100.0 * achieved_gbps / peak_gbps),
            );
        }
        session.receipt("bench", receipt)?;
    }
    Ok(())
}

fn graph_benchmark(session: &Session, cases: &[Case]) -> LabResult<()> {
    let probe = session.module("probe_sm120")?;
    let producer = probe
        .function("chop_probe_pdl_first_v1")
        .map_err(|e| e.to_string())?;
    let stamps = session.allocate(24)?;
    let producer_config = launch::simple_config((1, 1, 1), (32, 1, 1), 0, (1, 1, 1), false)?;
    let producer_graph = session.capture(|stream| {
        for _ in 0..GRAPH_LAYERS {
            producer
                .launch(
                    stream,
                    producer_config,
                    &[KernelArgument::Buffer(&stamps), KernelArgument::U64(1000)],
                )
                .map_err(|e| e.to_string())?;
        }
        Ok(())
    })?;
    let producer_time = time(session, 2, 5, |stream| {
        producer_graph.launch(stream).map_err(|e| e.to_string())
    })?;
    for &case in cases {
        if std::env::var("CHOPINFER_ATTENTION_CASE")
            .ok()
            .is_some_and(|filter| case.label() != filter)
        {
            continue;
        }
        let cubin = module(session, case.variant.profile)?;
        let fixture = prepare(session, case)?;
        let (partial, merge) = functions(&cubin, case.variant, &fixture.planned)?;
        let labelled = |error: String| format!("{}: {error}", case.label());
        let mut checks = fixture.replay(session)?;
        checks
            .eager(|| fixture.launch(session, &partial, &merge))
            .map_err(labelled)?;
        let eager = fixture.output(session)?;
        validate(session, case, &fixture, &eager, "graph_benchmark_eager")?;
        let mut caches = Vec::with_capacity(GRAPH_LAYERS - 1);
        for _ in 1..GRAPH_LAYERS {
            let k = session.allocate(fixture.buffers[&K].byte_len())?;
            let v = session.allocate(fixture.buffers[&V].byte_len())?;
            k.copy_range_from_async(0, &fixture.buffers[&K], 0, k.byte_len(), &session.stream)
                .map_err(|e| e.to_string())?;
            v.copy_range_from_async(0, &fixture.buffers[&V], 0, v.byte_len(), &session.stream)
                .map_err(|e| e.to_string())?;
            caches.push((k, v));
        }
        session.stream.synchronize().map_err(|e| e.to_string())?;
        let graph = session.capture(|stream| {
            for layer in 0..GRAPH_LAYERS {
                producer
                    .launch(
                        stream,
                        producer_config,
                        &[KernelArgument::Buffer(&stamps), KernelArgument::U64(1000)],
                    )
                    .map_err(|e| e.to_string())?;
                let cache = layer
                    .checked_sub(1)
                    .map(|index| (&caches[index].0, &caches[index].1));
                fixture.launch_with_cache(session, &partial, &merge, cache)?;
            }
            Ok(())
        })?;
        checks.graph(&graph).map_err(labelled)?;
        let timing = time(session, 2, 5, |stream| {
            graph.launch(stream).map_err(|e| e.to_string())
        })?;
        let pair_graph = session.capture(|_| {
            for layer in 0..GRAPH_LAYERS {
                let cache = layer
                    .checked_sub(1)
                    .map(|index| (&caches[index].0, &caches[index].1));
                fixture.launch_with_cache(session, &partial, &merge, cache)?;
            }
            Ok(())
        })?;
        let pair_timing = time(session, 2, 5, |stream| {
            pair_graph.launch(stream).map_err(|e| e.to_string())
        })?;
        let partial_graph = session.capture(|stream| {
            for layer in 0..GRAPH_LAYERS {
                let mut bindings = fixture.bindings();
                if let Some(index) = layer.checked_sub(1) {
                    bindings.buffers.insert(K, &caches[index].0);
                    bindings.buffers.insert(V, &caches[index].1);
                }
                launch::launch(
                    &session.context,
                    &partial,
                    stream,
                    fixture.planned.partial,
                    &fixture.planned.partial_args,
                    &bindings,
                )?;
            }
            Ok(())
        })?;
        let partial_timing = time(session, 2, 5, |stream| {
            partial_graph.launch(stream).map_err(|e| e.to_string())
        })?;
        let producer_us = producer_time.median_ms * 1000.0 / GRAPH_LAYERS as f32;
        let total_us = timing.median_ms * 1000.0 / GRAPH_LAYERS as f32;
        let pair_us = pair_timing.median_ms * 1000.0 / GRAPH_LAYERS as f32;
        let partial_us = partial_timing.median_ms * 1000.0 / GRAPH_LAYERS as f32;
        let mut receipt = BTreeMap::new();
        receipt.insert("case".into(), json!(case.label()));
        receipt.insert("layers".into(), json!(GRAPH_LAYERS));
        receipt.insert("distinct_kv_caches".into(), json!(true));
        receipt.insert("producer_pdl".into(), json!(true));
        receipt.insert("producer_us_per_layer".into(), json!(producer_us));
        receipt.insert("total_us_per_layer".into(), json!(total_us));
        receipt.insert("pair_us_per_layer".into(), json!(pair_us));
        receipt.insert("partial_us_per_layer".into(), json!(partial_us));
        receipt.insert("full_graph_median_ms".into(), json!(timing.median_ms));
        receipt.insert(
            "producer_graph_median_ms".into(),
            json!(producer_time.median_ms),
        );
        receipt.insert("partial_grid".into(), json!(fixture.planned.partial.grid));
        receipt.insert("merge_grid".into(), json!(fixture.planned.merge.grid));
        receipt.insert(
            "split_capacity".into(),
            json!(fixture.planned.split_capacity),
        );
        session.receipt("bench", receipt)?;
        eprintln!(
            "{}: 48-layer graph {:.2} us/layer producer+pair, {:.2} us/layer pair, {:.2} us/layer partial",
            case.label(),
            total_us,
            pair_us,
            partial_us,
        );
    }
    Ok(())
}

/// In-graph decode layers, each preceded by a PDL producer kernel. Layers
/// cycle through distinct KV caches totalling at least 512 MiB, four times
/// L2, so every layer reads its cache from DRAM as it would between weight
/// streams in serving.
fn decode_roofline_benchmark(session: &Session, cases: &[Case]) -> LabResult<()> {
    let (peak_gbps, _, peak_source) = find_peak(session.ordinal);
    let probe = session.module("probe_sm120")?;
    // A trivial PDL kernel that releases dependents and waits on its own
    // predecessor stands in for the RoPE and KV write before attention.
    let producer = probe
        .function("chop_probe_pdl_second_v1")
        .map_err(|e| e.to_string())?;
    let stamps = session.allocate(32)?;
    let producer_config = launch::simple_config((1, 1, 1), (32, 1, 1), 0, (1, 1, 1), true)?;
    for &case in cases {
        if std::env::var("CHOPINFER_ATTENTION_CASE")
            .ok()
            .is_some_and(|filter| !case.label().starts_with(&filter))
        {
            continue;
        }
        let cubin = module(session, case.variant.profile)?;
        let fixture = prepare(session, case)?;
        let (partial, merge) = functions(&cubin, case.variant, &fixture.planned)?;
        let labelled = |error: String| format!("{}: {error}", case.label());
        let mut checks = fixture.replay(session)?;
        checks
            .eager(|| fixture.launch(session, &partial, &merge))
            .map_err(labelled)?;
        let eager = fixture.output(session)?;
        validate(session, case, &fixture, &eager, "decode_roofline_eager")?;
        let cache_bytes = fixture.buffers[&K].byte_len() + fixture.buffers[&V].byte_len();
        let distinct = (512_usize << 20).div_ceil(cache_bytes).clamp(1, 256);
        let layers = distinct.max(GRAPH_LAYERS);
        let mut caches = Vec::with_capacity(distinct - 1);
        for _ in 1..distinct {
            let k = session.allocate(fixture.buffers[&K].byte_len())?;
            let v = session.allocate(fixture.buffers[&V].byte_len())?;
            k.copy_range_from_async(0, &fixture.buffers[&K], 0, k.byte_len(), &session.stream)
                .map_err(|e| e.to_string())?;
            v.copy_range_from_async(0, &fixture.buffers[&V], 0, v.byte_len(), &session.stream)
                .map_err(|e| e.to_string())?;
            caches.push((k, v));
        }
        session.stream.synchronize().map_err(|e| e.to_string())?;
        let cache = |layer: usize| {
            (layer % distinct)
                .checked_sub(1)
                .map(|index| (&caches[index].0, &caches[index].1))
        };
        let capture = |with_producer: bool| {
            session.capture(|stream| {
                for layer in 0..layers {
                    if with_producer {
                        producer
                            .launch(stream, producer_config, &[KernelArgument::Buffer(&stamps)])
                            .map_err(|e| e.to_string())?;
                    }
                    fixture.launch_with_cache(session, &partial, &merge, cache(layer))?;
                }
                Ok(())
            })
        };
        let graph = capture(true)?;
        checks.graph(&graph).map_err(labelled)?;
        let timing = time(session, 2, 5, |stream| {
            graph.launch(stream).map_err(|e| e.to_string())
        })?;
        let pair_graph = capture(false)?;
        let pair_timing = time(session, 2, 5, |stream| {
            pair_graph.launch(stream).map_err(|e| e.to_string())
        })?;
        let partial_graph = session.capture(|stream| {
            for layer in 0..layers {
                let mut bindings = fixture.bindings();
                if let Some((k, v)) = cache(layer) {
                    bindings.buffers.insert(K, k);
                    bindings.buffers.insert(V, v);
                }
                launch::launch(
                    &session.context,
                    &partial,
                    stream,
                    fixture.planned.partial,
                    &fixture.planned.partial_args,
                    &bindings,
                )?;
            }
            Ok(())
        })?;
        let partial_timing = time(session, 2, 5, |stream| {
            partial_graph.launch(stream).map_err(|e| e.to_string())
        })?;
        let total_us = timing.median_ms * 1000.0 / layers as f32;
        let pair_us = pair_timing.median_ms * 1000.0 / layers as f32;
        let partial_us = partial_timing.median_ms * 1000.0 / layers as f32;
        let roofline_us = fixture.logical_bytes as f64 / peak_gbps / 1000.0;
        let mut receipt = BTreeMap::new();
        receipt.insert("case".into(), json!(case.label()));
        receipt.insert("contexts".into(), json!(case.contexts()));
        receipt.insert("layers".into(), json!(layers));
        receipt.insert("distinct_kv_caches".into(), json!(distinct));
        receipt.insert("producer_pdl".into(), json!(true));
        receipt.insert("total_us_per_layer".into(), json!(total_us));
        receipt.insert("pair_us_per_layer".into(), json!(pair_us));
        receipt.insert("partial_us_per_layer".into(), json!(partial_us));
        receipt.insert("logical_bytes".into(), json!(fixture.logical_bytes));
        receipt.insert("roofline_kind".into(), json!("DRAM read"));
        receipt.insert("roofline_us".into(), json!(roofline_us));
        receipt.insert(
            "percent_peak".into(),
            json!(roofline_us / f64::from(total_us) * 100.0),
        );
        receipt.insert("peak_gb_per_second".into(), json!(peak_gbps));
        receipt.insert("peak_source".into(), json!(peak_source));
        receipt.insert("partial_grid".into(), json!(fixture.planned.partial.grid));
        receipt.insert("merge_grid".into(), json!(fixture.planned.merge.grid));
        receipt.insert(
            "split_capacity".into(),
            json!(fixture.planned.split_capacity),
        );
        session.receipt("bench", receipt)?;
        eprintln!(
            "{}: {:.2} us/layer with producer, {:.2} us/layer pair, {:.2} us/layer partial, \
             roofline {:.2} us",
            case.label(),
            total_us,
            pair_us,
            partial_us,
            roofline_us,
        );
    }
    Ok(())
}

/// MiMo's V multiplier; its RoPE theta is 1e4 on SWA layers and 1e7 on GA
/// layers.
const V_MULTIPLIER: f32 = 0.707;

fn rope_theta(variant: Variant) -> f32 {
    if variant.layout == CacheLayout::Ring {
        1e4
    } else {
        1e7
    }
}

/// A decode case with packed QKV rows and both ways to consume them: the row
/// engine's RoPE and KV write followed by the qualified entries, or the
/// entry that folds them in.
struct FoldFixture {
    variant: Variant,
    fixture: Fixture,
    folded: attention::PlannedAttention,
    row_symbol: String,
    row_plan: LaunchPlan,
    row_args: ArgumentList,
}

fn prepare_fold(session: &Session, case: Case) -> LabResult<FoldFixture> {
    prepare_fold_with_page_layers(session, case, 1)
}

/// `prepare_fold` over pages of `page_layers` layers.
fn prepare_fold_with_page_layers(
    session: &Session,
    case: Case,
    page_layers: usize,
) -> LabResult<FoldFixture> {
    let mut fixture = prepare_with_page_layers(session, case, page_layers)?;
    let v = case.variant;
    let batch = case.batch;
    let (qh, kh) = (v.query.q_heads as usize, v.query.kv_heads as usize);
    let (qk, vd) = (v.query.qk_dim as usize, v.query.v_dim as usize);
    let ring = v.layout == CacheLayout::Ring;
    // Values in [-4, 4] on a 1/250 grid, rounded to BF16, so rotation and
    // FP8 quantization both round.
    let mut state = 0x9e37_79b9_7f4a_7c15_u64 ^ (case.context() as u64) ^ ((batch as u64) << 32);
    let qkv: Vec<u16> = (0..batch * (qh * qk + kh * (qk + vd)))
        .map(|_| f32_to_bf16(((random_word(&mut state) % 2001) as f32 - 1000.0) / 250.0))
        .collect();
    let positions: Vec<i32> = case.contexts().iter().map(|&c| c as i32 - 1).collect();
    fixture.buffers.insert(QKV, session.upload_bf16(&qkv)?);
    fixture
        .buffers
        .insert(POSITIONS, put_i32(session, &positions)?);
    fixture.buffers.insert(
        TOKEN_BATCH,
        put_i32(session, &(0..batch as i32).collect::<Vec<_>>())?,
    );
    // The row entry of the same rank count, e.g. mimo_tp2_swa_decode.
    let (tag, base) = ["_tp2"]
        .into_iter()
        .find_map(|tag| Some((tag, v.name.strip_prefix("mimo")?.strip_prefix(tag)?)))
        .unwrap_or(("", v.name.strip_prefix("mimo").unwrap_or(v.name)));
    let row_name = match base {
        "_ga_bf16_decode" => "ga_bf16_computed_p64_pos32",
        "_ga_fp8_decode" => "ga_fp8_computed_p64_pos32",
        "_swa_decode" => "swa_decode_computed_p32_pos32",
        _ => return Err(format!("{} folds no RoPE", v.name)),
    };
    let entry = kernels_sm120::row::catalog(kernel_lab::catalogs())
        .into_iter()
        .find(|entry| entry.symbol == format!("chop_row_rope_mimo{tag}_{row_name}_v1"))
        .ok_or("row RoPE entry missing")?;
    let counts = RuntimeCounts {
        m: batch as u32,
        batch: batch as u32,
        context: case.context() as u32,
        routed_rows: 0,
        split_k: 1,
    };
    let bucket = batch.next_power_of_two() as u32;
    let arguments = fixture.arguments.clone();
    let row_plan = entry.plan(bucket, counts).map_err(|e| e.to_string())?;
    let row_args = entry
        .pack_arguments(vec![
            Argument::DeviceBuffer(QKV),
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
            Argument::NullDevicePointer,
            Argument::DeviceBuffer(LIVE),
            Argument::U32(1_048_576_u32.div_ceil(v.query.page_size)),
            Argument::U32(if ring { v.query.page_size } else { 0 }),
            Argument::U64(arguments.k_page_stride_bytes),
            Argument::U64(arguments.v_page_stride_bytes),
            Argument::U32(rope_theta(v).to_bits()),
            Argument::U32(V_MULTIPLIER.to_bits()),
            Argument::U32(arguments.k_scale.to_bits()),
            Argument::U32(arguments.v_scale.to_bits()),
        ])
        .map_err(|e| e.to_string())?;
    let folded = v
        .plan_folded(
            bucket,
            counts,
            arguments,
            RopeInputs {
                qkv: QKV,
                theta: rope_theta(v),
                v_multiplier: V_MULTIPLIER,
            },
        )
        .map_err(|e| e.to_string())?;
    if !folded.writes_cache || folded.partial_symbol != v.rope_symbol() {
        return Err(format!(
            "{} folded plan does not write the cache",
            case.label()
        ));
    }
    Ok(FoldFixture {
        variant: v,
        fixture,
        folded,
        row_symbol: entry.symbol,
        row_plan,
        row_args,
    })
}

/// The row engine's function and the attention entries of a fold case.
struct FoldFunctions {
    row: Function,
    partial: Function,
    merge: Function,
    folded: Function,
}

impl FoldFixture {
    fn functions(&self, session: &Session, rows: &Module) -> LabResult<FoldFunctions> {
        self.functions_in(&module(session, self.variant.profile)?, rows)
    }
    /// The case's functions from `cubin`, an attention cubin of its profile.
    fn functions_in(&self, cubin: &Module, rows: &Module) -> LabResult<FoldFunctions> {
        let (partial, merge) = functions(cubin, self.variant, &self.fixture.planned)?;
        let folded = cubin
            .function(&self.folded.partial_symbol)
            .map_err(|e| e.to_string())?;
        if self.folded.partial.dynamic_shared_bytes > 48 * 1024 {
            folded
                .set_max_dynamic_shared_memory(self.folded.partial.dynamic_shared_bytes)
                .map_err(|e| e.to_string())?;
        }
        let row = rows.function(&self.row_symbol).map_err(|e| e.to_string())?;
        Ok(FoldFunctions {
            row,
            partial,
            merge,
            folded,
        })
    }
    /// RoPE and the KV write, then the qualified partial and merge.
    fn launch_unfused(
        &self,
        session: &Session,
        functions: &FoldFunctions,
        cache: Option<(&DeviceBuffer, &DeviceBuffer)>,
    ) -> LabResult<()> {
        launch::launch(
            &session.context,
            &functions.row,
            &session.stream,
            self.row_plan,
            &self.row_args,
            &self.fixture.cache_bindings(cache),
        )?;
        self.fixture
            .launch_with_cache(session, &functions.partial, &functions.merge, cache)
    }
    fn launch_folded(
        &self,
        session: &Session,
        functions: &FoldFunctions,
        cache: Option<(&DeviceBuffer, &DeviceBuffer)>,
    ) -> LabResult<()> {
        self.fixture.launch_planned(
            session,
            &self.folded,
            &functions.folded,
            &functions.merge,
            cache,
        )
    }
}

fn bytes(session: &Session, buffer: &DeviceBuffer) -> LabResult<Vec<u8>> {
    let mut result = vec![0; buffer.byte_len()];
    buffer
        .copy_to_bytes(&mut result, &session.stream)
        .map_err(|e| e.to_string())?;
    Ok(result)
}

/// The eager bytes of two poisoned checks agree for every named buffer.
fn same_eager(
    case: Case,
    phase: &str,
    expected: &Replay,
    actual: &Replay,
    names: &[&str],
) -> LabResult<()> {
    for &name in names {
        let (Some(left), Some(right)) = (expected.expected(name), actual.expected(name)) else {
            return Err(format!("{} {phase}: no eager {name}", case.label()));
        };
        if left.len() != right.len() {
            return Err(format!("{} {phase}: {name} sizes differ", case.label()));
        }
        if let Some(offset) = left.iter().zip(right).position(|(a, b)| a != b) {
            return Err(format!(
                "{} {phase}: {name} differs at byte {offset}",
                case.label()
            ));
        }
    }
    Ok(())
}

/// Bytes of every sequence's current row (position `seq_len - 1`) in the K
/// and V caches: the rows the row RoPE entry and the folded decode entries
/// write without reading (`cuda/attention/README.md`).
fn current_rows(case: Case, fixture: &Fixture) -> LabResult<(WriteMap, WriteMap)> {
    let query = case.variant.query;
    let cache = WriteMap::of(if query.kv_dtype == DType::Fp8E4M3 {
        DType::Fp8E4M3
    } else {
        DType::Bf16
    });
    let element = cache.bytes()?;
    let (kh, qd, vd) = (
        query.kv_heads as usize,
        query.qk_dim as usize,
        query.v_dim as usize,
    );
    let page = query.page_size as usize;
    let ring = case.variant.layout == CacheLayout::Ring;
    let (k_row, v_row) = (kh * qd * element, kh * vd * element);
    let (mut k_rows, mut v_rows) = (Vec::new(), Vec::new());
    for (b, &context) in case.contexts().iter().enumerate() {
        let position = context - 1;
        let (k_start, v_start) = if ring {
            // Ring slots run in reverse batch order (`prepare_values`).
            let row = (case.batch - 1 - b) * page + position % page;
            (row * k_row, row * v_row)
        } else {
            let physical = fixture.page_lists[b][position / page];
            let within = position % page;
            (
                physical * fixture.arguments.k_page_stride_bytes as usize + within * k_row,
                physical * fixture.arguments.v_page_stride_bytes as usize + within * v_row,
            )
        };
        k_rows.push(k_start..k_start + k_row);
        v_rows.push(v_start..v_start + v_row);
    }
    Ok((cache.byte_ranges(k_rows)?, cache.byte_ranges(v_rows)?))
}

/// A fold case's unfused reference and folded entry on poisoned buffers.
/// The reference (RoPE and the KV write, then the unfused entries) runs on
/// the fixture's caches; the folded entry runs eagerly, again, in a graph and
/// behind a PDL producer on copies of the starting caches, each restored with
/// the current rows poisoned before every launch. Both must match the
/// reference bit for bit. `row_kv_new` says whether the reference's row step
/// writes the rotated K and V rows (`K_NEW`, `V_NEW`); otherwise they must
/// keep their poison.
fn fold_checks(
    session: &Session,
    case: Case,
    fixture: &Fixture,
    unfused: impl FnOnce() -> LabResult<()>,
    folded: impl Fn(Option<(&DeviceBuffer, &DeviceBuffer)>) -> LabResult<()>,
    pdl: bool,
    row_kv_new: bool,
) -> LabResult<Evidence> {
    let labelled = |error: String| format!("{}: {error}", case.label());
    let (k, v) = (&fixture.buffers[&K], &fixture.buffers[&V]);
    let (k_start, v_start) = (bytes(session, k)?, bytes(session, v)?);
    let (k_rows, v_rows) = current_rows(case, fixture)?;
    // The MiMo RoPE row writes every live token's rotated K and V rows
    // (rope_kv.cuh, mapped as in row.rs); the buffers hold exactly the live
    // batch. The Qwen3.8 norm, RoPE and KV-write row has no such outputs.
    let (k_new, v_new) = (&fixture.buffers[&K_NEW], &fixture.buffers[&V_NEW]);
    let row_map = |buffer: &DeviceBuffer| {
        if row_kv_new {
            WriteMap::of(DType::Bf16).byte_prefix(buffer.byte_len())
        } else {
            WriteMap::of(DType::Bf16).none()
        }
    };
    let mut reference = fixture
        .replay(session)?
        .scratch("row Q", &fixture.buffers[&Q])
        .output("row K new", k_new, row_map(k_new)?)
        .output("row V new", v_new, row_map(v_new)?)
        .state("K cache", k, k_start.clone(), k_rows.clone(), Vec::new())
        .state("V cache", v, v_start.clone(), v_rows.clone(), Vec::new());
    reference.eager(unfused).map_err(labelled)?;
    let fold_k = put_bytes(session, &k_start)?;
    let fold_v = put_bytes(session, &v_start)?;
    let cache = Some((&fold_k, &fold_v));
    let mut checks = fixture
        .replay(session)?
        .state("K cache", &fold_k, k_start, k_rows, Vec::new())
        .state("V cache", &fold_v, v_start, v_rows, Vec::new());
    checks.eager(|| folded(cache)).map_err(labelled)?;
    same_eager(
        case,
        "folded against unfused",
        &reference,
        &checks,
        &["attention output", "K cache", "V cache"],
    )?;
    checks.relaunch(|| folded(cache)).map_err(labelled)?;
    let graph = session.capture(|_| folded(cache))?;
    checks.graph(&graph).map_err(labelled)?;
    let producer = PdlProducer::new(session)?;
    checks
        .pdl(|| producer.launch(session), || folded(cache), pdl)
        .map_err(labelled)?;
    Ok(checks.evidence().clone())
}

/// The folded entry against RoPE and the KV write followed by the unfused
/// entries, from the same caches: the output and both caches bit for bit,
/// then deterministic replay, graph replay and a PDL launch.
fn run_fold(session: &Session, cases: &[Case]) -> LabResult<()> {
    let rows = session
        .context
        .load_module(ROW_CUBIN)
        .map_err(|e| e.to_string())?;
    for &case in cases {
        let fold = prepare_fold(session, case)?;
        let functions = fold.functions(session, &rows)?;
        let evidence = fold_checks(
            session,
            case,
            &fold.fixture,
            || fold.launch_unfused(session, &functions, None),
            |cache| fold.launch_folded(session, &functions, cache),
            fold.folded.partial.pdl,
            true,
        )?;
        let mut receipt = BTreeMap::new();
        receipt.insert("case".into(), json!(case.label()));
        receipt.insert("contexts".into(), json!(case.contexts()));
        receipt.insert("folded_symbol".into(), json!(fold.folded.partial_symbol));
        receipt.insert("unfused_row_symbol".into(), json!(fold.row_symbol));
        receipt.insert("output_bitwise".into(), json!(true));
        receipt.insert("cache_bitwise".into(), json!(true));
        evidence.record(&mut receipt);
        session.receipt("attention", receipt)?;
    }
    Ok(())
}

/// In-graph decode layers between the MiMo decode QKV and O GEMMs of the
/// bucket, as in serving: RoPE and the KV write followed by the unfused
/// entries, against the folded entry. The GEMMs keep each layer's kernels as
/// cold in the SM instruction cache as serving does; with only the attention
/// kernels in the loop, the unfused pair's 62 KB of code stays resident and
/// three alternating kernels do not. Layers cycle through distinct KV caches
/// totalling at least 512 MiB, as in `decode_roofline_benchmark`.
fn rope_fold_benchmark(session: &Session, cases: &[Case]) -> LabResult<()> {
    let rows = session
        .context
        .load_module(ROW_CUBIN)
        .map_err(|e| e.to_string())?;
    for &case in cases {
        let fold = prepare_fold(session, case)?;
        let functions = fold.functions(session, &rows)?;
        let fixture = &fold.fixture;
        let bucket = case.batch.next_power_of_two() as u32;
        let qkv = super::gemm::Case::prepare(
            session,
            &format!(
                "chop_gemm_mimo_qkv_{}_m{bucket}_v1",
                if case.variant.layout == CacheLayout::Ring {
                    "swa"
                } else {
                    "ga"
                }
            ),
            case.batch as u32,
            false,
        )?;
        let o = super::gemm::Case::prepare(
            session,
            &format!("chop_gemm_mimo_o_m{bucket}_v1"),
            case.batch as u32,
            false,
        )?;
        let (qkv_function, o_function) = (qkv.function(session)?, o.function(session)?);
        let cache_bytes = fixture.buffers[&K].byte_len() + fixture.buffers[&V].byte_len();
        let distinct = (512_usize << 20).div_ceil(cache_bytes).clamp(1, 256);
        let layers = distinct.max(GRAPH_LAYERS);
        let mut caches = Vec::with_capacity(distinct - 1);
        for _ in 1..distinct {
            let k = session.allocate(fixture.buffers[&K].byte_len())?;
            let v = session.allocate(fixture.buffers[&V].byte_len())?;
            k.copy_range_from_async(0, &fixture.buffers[&K], 0, k.byte_len(), &session.stream)
                .map_err(|e| e.to_string())?;
            v.copy_range_from_async(0, &fixture.buffers[&V], 0, v.byte_len(), &session.stream)
                .map_err(|e| e.to_string())?;
            caches.push((k, v));
        }
        session.stream.synchronize().map_err(|e| e.to_string())?;
        let cache = |layer: usize| {
            (layer % distinct)
                .checked_sub(1)
                .map(|index| (&caches[index].0, &caches[index].1))
        };
        // `Some(folded)` runs one attention path between the GEMMs; `None`
        // runs the GEMMs alone.
        let capture = |attention: Option<bool>| {
            session.capture(|stream| {
                for layer in 0..layers {
                    qkv.launch(session, &qkv_function, stream)?;
                    match attention {
                        Some(true) => fold.launch_folded(session, &functions, cache(layer))?,
                        Some(false) => fold.launch_unfused(session, &functions, cache(layer))?,
                        None => {}
                    }
                    o.launch(session, &o_function, stream)?;
                }
                Ok(())
            })
        };
        let unfused = capture(Some(false))?;
        let folded = capture(Some(true))?;
        // The unfused graph is the reference the folded graph must match.
        let labelled = |error: String| format!("{}: {error}", case.label());
        let mut checks = fixture
            .replay(session)?
            .scratch("row Q", &fixture.buffers[&Q]);
        checks
            .eager(|| unfused.launch(&session.stream).map_err(|e| e.to_string()))
            .map_err(labelled)?;
        checks.graph(&folded).map_err(labelled)?;
        let gemms = capture(None)?;
        let per_layer = |graph: &runtime_cuda::GraphExec| -> LabResult<f32> {
            let timing = time(session, 2, 5, |stream| {
                graph.launch(stream).map_err(|e| e.to_string())
            })?;
            Ok(timing.median_ms * 1000.0 / layers as f32)
        };
        let unfused_us = per_layer(&unfused)?;
        let folded_us = per_layer(&folded)?;
        let gemm_us = per_layer(&gemms)?;
        let mut receipt = BTreeMap::new();
        receipt.insert("case".into(), json!(case.label()));
        receipt.insert("contexts".into(), json!(case.contexts()));
        receipt.insert("layers".into(), json!(layers));
        receipt.insert("distinct_kv_caches".into(), json!(distinct));
        receipt.insert(
            "neighbours".into(),
            json!("decode QKV and O GEMMs of the bucket"),
        );
        receipt.insert("unfused_us_per_layer".into(), json!(unfused_us));
        receipt.insert("folded_us_per_layer".into(), json!(folded_us));
        receipt.insert("gemms_only_us_per_layer".into(), json!(gemm_us));
        receipt.insert("saving_us_per_layer".into(), json!(unfused_us - folded_us));
        receipt.insert("unfused_row_symbol".into(), json!(fold.row_symbol));
        receipt.insert("folded_symbol".into(), json!(fold.folded.partial_symbol));
        receipt.insert("partial_grid".into(), json!(fold.folded.partial.grid));
        session.receipt("bench", receipt)?;
        eprintln!(
            "{}: {:.2} us/layer QKV + RoPE + attention + O, {:.2} us/layer folded, \
             {:.2} us/layer GEMMs alone",
            case.label(),
            unfused_us,
            folded_us,
            gemm_us,
        );
    }
    Ok(())
}

fn folded_variants() -> Vec<Variant> {
    variants()
        .into_iter()
        .filter(|variant| variant.rope_fold.is_some())
        .collect()
}

/// The MiMo and std variants; `qwen38` covers the Qwen3.8 Flash ones.
fn variants() -> Vec<Variant> {
    attention::catalog(kernel_lab::catalogs())
        .into_iter()
        .filter(|variant| {
            variant.profile != model_qwen38_contract::sm120::config::PROFILE
                && !kernel_lab::lab_shapes::PROFILES.contains(&variant.profile)
        })
        .collect()
}

#[test]
fn attention_decode_contexts_and_batches() {
    gpu_test(512 * 1024 * 1024, |session| {
        let mut cases = Vec::new();
        for variant in variants()
            .into_iter()
            .filter(|v| v.query.phase == AttentionPhase::Decode)
        {
            for batch in [1, 8, 32] {
                for context in [1, 127, 128, 129, 4096] {
                    cases.push(Case {
                        variant,
                        batch,
                        rows: 1,
                        prefix: context - 1,
                        mixed: false,
                    });
                }
            }
            // Rows of 4096 down to 64 tokens share one stream-K schedule.
            for batch in [8, 32] {
                cases.push(Case {
                    variant,
                    batch,
                    rows: 1,
                    prefix: 4095,
                    mixed: true,
                });
            }
        }
        run(session, &cases, false)
    })
    .expect("attention decode correctness");
}

/// Paged decode at buckets 64, 128 and 256 against the oracle: balanced
/// rows of 4,096 tokens and mixed rows of 16,384 down to 256 tokens, whose
/// pieces range from the bucket's smallest to the split-slot cap, with
/// poisoned eager, relaunch, graph and PDL launches. Every paged decode
/// variant runs prepared for 256 rows.
#[test]
fn attention_decode_large_buckets_match_oracle() {
    gpu_test(8 * 1024 * 1024 * 1024, |session| {
        let mut cases = Vec::new();
        let paged = variants()
            .into_iter()
            .filter(|v| v.query.phase == AttentionPhase::Decode && v.layout == CacheLayout::Paged)
            .collect::<Vec<_>>();
        for variant in &paged {
            let variant = variant
                .with_batch_capacity(attention::MAX_BATCH_ROWS)
                .map_err(|e| format!("{}: {e}", variant.name))?;
            for batch in [64_usize, 128, 256] {
                for (prefix, mixed) in [(4095, false), (16_383, true)] {
                    cases.push(Case {
                        variant,
                        batch,
                        rows: 1,
                        prefix,
                        mixed,
                    });
                }
            }
        }
        if paged.is_empty() || cases.len() != paged.len() * 6 {
            return Err(format!(
                "{} large-bucket cases for {} paged decode variants",
                cases.len(),
                paged.len()
            ));
        }
        run(session, &cases, true)
    })
    .expect("attention decode at large buckets");
}

/// FP8 KV contract: FP8 KV rounded from general BF16 values with a non-unit
/// scale matches the oracle over its codes, and the attention of the
/// unrounded values within the storage tolerance, in every paged FP8
/// variant and phase.
#[test]
fn attention_fp8_kv_tracks_bf16_kv_within_the_storage_tolerance() {
    gpu_test(512 * 1024 * 1024, |session| {
        for variant in variants().into_iter().filter(|v| {
            v.query.kv_dtype == DType::Fp8E4M3
                && v.layout == CacheLayout::Paged
                && v.query.softcap.is_none()
                && v.query.window_left.is_none()
        }) {
            let shapes: &[(usize, usize, usize, bool)] = match variant.query.phase {
                AttentionPhase::Decode => &[(1, 1, 128, false), (8, 1, 4095, true)],
                AttentionPhase::Verify => &[(8, 4, 1020, true)],
                AttentionPhase::Prefill => &[(1, 256, 768, false), (2, 64, 64, false)],
            };
            for &(batch, rows, prefix, mixed) in shapes {
                let case = Case {
                    variant,
                    batch,
                    rows,
                    prefix,
                    mixed,
                };
                let cubin = module(session, variant.profile)?;
                let (fixture, rounded) =
                    prepare_values(session, case, 1, None, Some([0.0625, 0.015625]))?;
                let rounded = rounded.ok_or("rounded fixture")?;
                let (partial, merge) = functions(&cubin, variant, &fixture.planned)?;
                fixture.launch(session, &partial, &merge)?;
                let output = fixture.output(session)?;
                validate(session, case, &fixture, &output, "fp8 rounded")?;
                let (exact, sigma) = fp8_exact(case, &fixture, &rounded);
                let actual: Vec<f32> = output.iter().map(|&bits| bf16_to_f32(bits)).collect();
                let mut receipt = BTreeMap::new();
                receipt.insert("case".into(), json!(case.label()));
                receipt.insert("phase".into(), json!("fp8 kv vs bf16 kv"));
                receipt.insert("kv_scales".into(), json!(rounded.scales));
                let result = kernel_lab::check_fp8_kv_output(
                    &actual,
                    &exact,
                    &sigma,
                    &fixture.reference,
                    &mut receipt,
                );
                session.receipt("attention", receipt)?;
                result.map_err(|e| format!("{}: {e}", case.label()))?;
            }
        }
        Ok(())
    })
    .expect("FP8 KV attention against BF16 KV");
}

#[test]
fn attention_decode_rope_fold_matches_row_engine() {
    gpu_test(1024 * 1024 * 1024, |session| {
        let mut cases = Vec::new();
        for variant in folded_variants() {
            // Current tokens at the start, end and inside of a 64-token page,
            // past one and many 128-slot ring wraps, and over split rows.
            for context in [1, 2, 63, 64, 65, 127, 128, 129, 4096, 4097, 70_001] {
                cases.push(Case {
                    variant,
                    batch: 1,
                    rows: 1,
                    prefix: context - 1,
                    mixed: false,
                });
            }
            for batch in [2, 8, 32] {
                for context in [1, 64, 129, 4096] {
                    cases.push(Case {
                        variant,
                        batch,
                        rows: 1,
                        prefix: context - 1,
                        mixed: false,
                    });
                }
                // Mixed rows: stream-K CTAs span several rows.
                cases.push(Case {
                    variant,
                    batch,
                    rows: 1,
                    prefix: 4095,
                    mixed: true,
                });
            }
        }
        if let Ok(filter) = std::env::var("CHOPINFER_ATTENTION_CASE") {
            cases.retain(|case| case.label().starts_with(&filter));
        }
        run_fold(session, &cases)
    })
    .expect("folded RoPE decode matches RoPE and KV write plus decode");
}

#[test]
fn attention_prefill_chunks_and_ring_wrap() {
    gpu_test(512 * 1024 * 1024, |session| {
        let mut cases: Vec<_> = variants()
            .into_iter()
            .filter(|v| v.query.phase == AttentionPhase::Prefill)
            .flat_map(|variant| {
                // Other rank profiles change only the head counts; their
                // 2048-row oracles would add minutes of host time.
                let rows: &[usize] = if variant.name.contains("_tp") {
                    &[512]
                } else {
                    &[512, 2048]
                };
                rows.iter().map(move |&rows| Case {
                    variant,
                    batch: 1,
                    rows,
                    prefix: if variant.layout == CacheLayout::Ring {
                        129
                    } else {
                        0
                    },
                    mixed: false,
                })
            })
            .collect();
        for variant in variants()
            .into_iter()
            .filter(|v| matches!(v.name, "mimo_ga_bf16_prefill" | "std128_fp8_prefill"))
        {
            cases.push(Case {
                variant,
                batch: 1,
                rows: 512,
                prefix: 129,
                mixed: false,
            });
        }
        // Rows with at least 32768 visible keys use one probability term.
        cases.push(Case {
            variant: variants()
                .into_iter()
                .find(|v| v.name == "mimo_ga_bf16_prefill")
                .unwrap(),
            batch: 1,
            rows: 256,
            prefix: 32_768,
            mixed: false,
        });
        if let Ok(filter) = std::env::var("CHOPINFER_ATTENTION_CASE") {
            cases.retain(|case| case.label().starts_with(&filter));
        }
        run(session, &cases, false)
    })
    .expect("attention prefill correctness");
}

#[test]
fn attention_strided_multilayer_pages() {
    gpu_test(512 * 1024 * 1024, |session| {
        let mut cases = Vec::new();
        for name in [
            "mimo_ga_bf16_decode",
            "mimo_ga_fp8_decode",
            "mimo_ga_bf16_prefill",
            "std128_fp8_prefill",
        ] {
            let variant = variants()
                .into_iter()
                .find(|variant| variant.name == name)
                .ok_or_else(|| format!("missing attention variant {name}"))?;
            let decode = variant.query.phase == AttentionPhase::Decode;
            cases.push(Case {
                variant,
                batch: 1,
                rows: if decode { 1 } else { 512 },
                prefix: if decode { 4095 } else { 129 },
                mixed: false,
            });
        }
        run_with_page_layers(session, &cases, false, 3)
    })
    .expect("strided multi-layer paged KV attention");
}

fn named(name: &str) -> LabResult<Variant> {
    variants()
        .into_iter()
        .find(|variant| variant.name == name)
        .ok_or_else(|| format!("missing attention variant {name}"))
}

/// A fixture's caches with every live page or ring slot moved to the ids
/// `pool_offsets::high_ids` picks, in sentinel pools, and the block table or
/// slot list that names them. Paged pages alternate between group A (past
/// `i32::MAX` bytes) and group B (at 2^32 elements); a ring gives one
/// sequence each group of the K slot stride and of the V slot stride.
struct HighCache {
    k: SentinelPool,
    v: SentinelPool,
    /// `TABLE` for a paged layout, `SLOT` for a ring, and its high ids.
    handle: u64,
    ids: DeviceBuffer,
    /// Each live low page or slot and the high id that holds it.
    moved: Vec<(u64, u64)>,
    k_entry: u64,
    v_entry: u64,
    k_stride: u64,
    v_stride: u64,
    /// Entries of each pool, the tensor maps' outer extent.
    entries: u64,
    groups: Value,
}

impl HighCache {
    fn new(session: &Session, fixture: &Fixture) -> LabResult<Self> {
        let variant = fixture.variant;
        let ring = variant.layout == CacheLayout::Ring;
        let element = if variant.query.kv_dtype == DType::Fp8E4M3 {
            1
        } else {
            2
        };
        let tokens = u64::from(variant.query.page_size) * u64::from(variant.query.kv_heads);
        let k_entry = tokens * u64::from(variant.query.qk_dim) * element;
        let v_entry = tokens * u64::from(variant.query.v_dim) * element;
        let k_stride = fixture.arguments.k_page_stride_bytes;
        let v_stride = fixture.arguments.v_page_stride_bytes;
        let (low, high, groups) = if ring {
            let batch = fixture.slot_ids.len();
            let k = pool_offsets::high_ids(k_stride, element, batch);
            let v = pool_offsets::high_ids(v_stride, element, batch);
            let mut high = Vec::new();
            for i in 0..batch {
                for id in [k.a[i], k.b[i], v.a[i], v.b[i]] {
                    if !high.contains(&id) {
                        high.push(id);
                    }
                }
            }
            high.truncate(batch);
            let low = fixture.slot_ids.iter().map(|&slot| slot as u64).collect();
            (low, high, json!({"k": k.describe(), "v": v.describe()}))
        } else {
            if k_stride != v_stride {
                return Err("high-id paged cases share one K and V page stride".into());
            }
            let mut pages: Vec<u64> = fixture
                .page_lists
                .iter()
                .flatten()
                .map(|&page| page as u64)
                .collect();
            pages.sort_unstable();
            pages.dedup();
            let ids = pool_offsets::high_ids(k_stride, element, pages.len().div_ceil(2));
            let high = ids.alternating(pages.len());
            (pages, high, json!({"pages": ids.describe()}))
        };
        let entries = high.iter().copied().max().unwrap_or(0) + 1;
        let mut k = SentinelPool::new(session, entries * k_stride)?;
        let mut v = SentinelPool::new(session, entries * v_stride)?;
        for (&from, &to) in low.iter().zip(&high) {
            k.copy_from(
                session,
                to * k_stride,
                &fixture.buffers[&K],
                from * k_stride,
                k_entry,
            )?;
            v.copy_from(
                session,
                to * v_stride,
                &fixture.buffers[&V],
                from * v_stride,
                v_entry,
            )?;
        }
        let moved: Vec<(u64, u64)> = low.into_iter().zip(high.iter().copied()).collect();
        let (handle, ids) = if ring {
            (SLOT, put_i32(session, &pool_offsets::i32_ids(&high))?)
        } else {
            let map: BTreeMap<u64, u64> = moved.iter().copied().collect();
            let max_pages = (variant.query.max_context_capacity as usize)
                .div_ceil(variant.query.page_size as usize);
            let mut tables = vec![0_i32; fixture.page_lists.len() * max_pages];
            for (b, list) in fixture.page_lists.iter().enumerate() {
                for (p, &physical) in list.iter().enumerate() {
                    tables[b * max_pages + p] =
                        pool_offsets::i32_ids(&[map[&(physical as u64)]])[0];
                }
            }
            (TABLE, put_i32(session, &tables)?)
        };
        session.stream.synchronize().map_err(|e| e.to_string())?;
        Ok(Self {
            k,
            v,
            handle,
            ids,
            moved,
            k_entry,
            v_entry,
            k_stride,
            v_stride,
            entries,
            groups,
        })
    }

    /// The fixture's plan over these pools: prefill plans its tensor maps
    /// over their entries; decode and verify keep theirs.
    fn plan(&self, fixture: &Fixture) -> LabResult<attention::PlannedAttention> {
        let variant = fixture.variant;
        if variant.query.phase != AttentionPhase::Prefill {
            return Ok(fixture.planned.clone());
        }
        let mut arguments = fixture.arguments.clone();
        let (k_map, v_map) = variant
            .cache_maps_with_strides(
                u32::try_from(self.entries).map_err(|e| e.to_string())?,
                K,
                V,
                self.k_stride,
                self.v_stride,
            )
            .map_err(|e| e.to_string())?;
        arguments.k_map = Some(k_map);
        arguments.v_map = Some(v_map);
        variant
            .plan_for_bucket(fixture.bucket, fixture.counts, arguments)
            .map_err(|e| e.to_string())
    }

    /// Binds the high ids in place of the fixture's table or slots, and
    /// poisons its output and workspace so a result this run never writes
    /// cannot pass as the low run's.
    fn bind(&self, session: &Session, fixture: &mut Fixture) -> LabResult<()> {
        fixture.buffers.insert(self.handle, self.ids.clone());
        pool_offsets::poison(session, &fixture.buffers[&OUT])?;
        pool_offsets::poison(session, &fixture.buffers[&WORK])
    }

    fn cache(&self) -> Option<(&DeviceBuffer, &DeviceBuffer)> {
        Some((&self.k.buffer, &self.v.buffer))
    }

    /// Every moved entry holds the bytes `k` and `v` hold at its low id.
    fn check_entries(
        &self,
        session: &Session,
        case: &str,
        k: &DeviceBuffer,
        v: &DeviceBuffer,
    ) -> LabResult<()> {
        for &(from, to) in &self.moved {
            pool_offsets::same(
                case,
                &format!("K entry {to}"),
                &pool_offsets::read(session, k, from * self.k_stride, self.k_entry)?,
                &self.k.read(session, to * self.k_stride, self.k_entry)?,
            )?;
            pool_offsets::same(
                case,
                &format!("V entry {to}"),
                &pool_offsets::read(session, v, from * self.v_stride, self.v_entry)?,
                &self.v.read(session, to * self.v_stride, self.v_entry)?,
            )?;
        }
        Ok(())
    }

    fn describe(&self) -> Value {
        let highest = self.moved.iter().map(|&(_, to)| to).max().unwrap_or(0);
        json!({
            "groups": self.groups,
            "entries_moved": self.moved.len(),
            "k_pool_bytes": self.k.bytes(),
            "v_pool_bytes": self.v.bytes(),
            "highest_k_offset": highest * self.k_stride,
            "highest_v_offset": highest * self.v_stride,
        })
    }
}

/// Page layers of the serving page of a paged variant: MiMo's nine GA
/// layers, Qwen's twelve full-attention layers.
fn serving_page_layers(variant: Variant) -> usize {
    match (variant.layout, variant.profile) {
        (CacheLayout::Ring, _) => 1,
        (_, model_qwen38_contract::sm120::config::PROFILE) => 12,
        _ => 9,
    }
}

/// One case at low ids, checked against the oracle, then with its live
/// pages or slots at high ids: the output must match bit for bit.
fn run_high_ids(session: &Session, case: Case) -> LabResult<()> {
    let page_layers = serving_page_layers(case.variant);
    let cubin = module(session, case.variant.profile)?;
    let mut fixture = prepare_with_page_layers(session, case, page_layers)?;
    let (partial, merge) = functions(&cubin, case.variant, &fixture.planned)?;
    fixture.launch(session, &partial, &merge)?;
    let low = fixture.output(session)?;
    validate(session, case, &fixture, &low, "low ids")?;
    let high = HighCache::new(session, &fixture)?;
    let planned = high.plan(&fixture)?;
    high.bind(session, &mut fixture)?;
    fixture.launch_planned(session, &planned, &partial, &merge, high.cache())?;
    exact(case, &low, &fixture.output(session)?, "high ids")?;
    pool_offsets::receipt(
        session,
        "attention",
        &case.label(),
        json!({"page_layers": page_layers, "cache": high.describe(), "output_bitwise": true}),
    )
}

/// Paged pages and ring slots past the 32-bit lines (`pool_offsets`):
/// decode, verify and prefill of MiMo's GA caches in the TP4 serving page,
/// its SWA ring and the DFlash block ring give the low-id output bit for
/// bit. GA BF16 decode runs first.
#[test]
fn attention_high_ids_match_low_ids() {
    gpu_test(pool_offsets::need_bytes(22 << 30), |session| {
        let case = |name: &str, batch, rows, prefix, mixed| -> LabResult<Case> {
            Ok(Case {
                variant: named(name)?,
                batch,
                rows,
                prefix,
                mixed,
            })
        };
        for case in [
            case("mimo_ga_bf16_decode", 4, 1, 4095, true)?,
            case("mimo_ga_fp8_decode", 4, 1, 4095, true)?,
            case("mimo_ga_bf16_verify", 4, 8, 2040, true)?,
            case("mimo_ga_fp8_verify", 4, 8, 2040, true)?,
            case("mimo_ga_bf16_prefill", 2, 256, 1024, false)?,
            case("mimo_ga_fp8_prefill", 2, 256, 1024, false)?,
            case("mimo_swa_decode", 4, 1, 1000, false)?,
            case("mimo_swa_verify", 4, 8, 1000, true)?,
            case("mimo_swa_prefill", 4, 64, 200, false)?,
            case("mimo_dflash_block", 4, 8, 1000, true)?,
        ] {
            run_high_ids(session, case)?;
        }
        Ok(())
    })
    .expect("attention at high page and slot ids");
}

/// The MiMo decode entries that fold RoPE and the KV write, at high page
/// and slot ids: the output and every written cache entry equal the low-id
/// run's bit for bit, and every other pool byte keeps the sentinel.
#[test]
fn attention_high_ids_rope_fold_matches_low_ids() {
    gpu_test(pool_offsets::need_bytes(22 << 30), |session| {
        let rows = session
            .context
            .load_module(ROW_CUBIN)
            .map_err(|e| e.to_string())?;
        for (name, prefix) in [
            ("mimo_ga_bf16_decode", 4095),
            ("mimo_ga_fp8_decode", 4095),
            ("mimo_swa_decode", 1000),
        ] {
            let case = Case {
                variant: named(name)?,
                batch: 4,
                rows: 1,
                prefix,
                mixed: true,
            };
            let page_layers = serving_page_layers(case.variant);
            let mut fold = prepare_fold_with_page_layers(session, case, page_layers)?;
            let functions = fold.functions(session, &rows)?;
            let low_k = put_bytes(session, &bytes(session, &fold.fixture.buffers[&K])?)?;
            let low_v = put_bytes(session, &bytes(session, &fold.fixture.buffers[&V])?)?;
            fold.launch_folded(session, &functions, Some((&low_k, &low_v)))?;
            let low = fold.fixture.output(session)?;
            // The fixture's own caches still hold the starting rows.
            let high = HighCache::new(session, &fold.fixture)?;
            high.bind(session, &mut fold.fixture)?;
            fold.launch_folded(session, &functions, high.cache())?;
            let label = case.label();
            exact(
                case,
                &low,
                &fold.fixture.output(session)?,
                "folded at high ids",
            )?;
            high.check_entries(session, &label, &low_k, &low_v)?;
            let scanned = high.k.check_untouched(session, &label)?
                + high.v.check_untouched(session, &label)?;
            pool_offsets::receipt(
                session,
                "attention_rope_fold",
                &label,
                json!({
                    "folded_symbol": fold.folded.partial_symbol,
                    "page_layers": page_layers,
                    "cache": high.describe(),
                    "output_bitwise": true,
                    "entries_bitwise": true,
                    "sentinel_bytes_checked": scanned,
                }),
            )?;
        }
        Ok(())
    })
    .expect("folded RoPE decode at high page and slot ids");
}

/// Verify items: one to eight queries per item after prefixes from 0 to 32K
/// tokens, uniform and mixed batches, including windows that start inside
/// the prefix, the ring's wrap and items shorter than their bucket.
fn verify_cases(variant: Variant) -> Vec<Case> {
    let mut cases = Vec::new();
    for (batch, rows) in [(1, 8), (1, 1), (3, 4), (16, 2), (32, 1), (32, 2), (5, 8)] {
        for prefix in [0, 1, 120, 127, 128, 1017, 4095, 32_761] {
            if batch > 1 && prefix > 4095 {
                continue;
            }
            for mixed in [false, true] {
                if mixed && batch == 1 {
                    continue;
                }
                cases.push(Case {
                    variant,
                    batch,
                    rows,
                    prefix,
                    mixed,
                });
            }
        }
    }
    cases
}

/// Two BF16 outputs agree within `ulps` BF16 steps of the larger magnitude.
fn close(case: Case, expected: &[u16], actual: &[u16], ulps: f32, phase: &str) -> LabResult<()> {
    for (index, (&a, &b)) in actual.iter().zip(expected).enumerate() {
        let (a, b) = (bf16_to_f32(a), bf16_to_f32(b));
        let step = a.abs().max(b.abs()).max(1.0 / 1024.0) / 128.0;
        if !(a - b).abs().le(&(ulps * step)) {
            return Err(format!(
                "{} {phase} differs at {index}: actual={a}, expected={b}",
                case.label()
            ));
        }
    }
    Ok(())
}

#[test]
fn attention_verify_matches_oracle_and_prefill() {
    gpu_test(2 * 1024 * 1024 * 1024, |session| {
        // TP4, then TP2, whose verify variants MTP and DFlash run.
        let filter = std::env::var("CHOPINFER_ATTENTION_CASE").ok();
        for tag in ["", "_tp2"] {
            for (verify, prefill) in [
                ("ga_bf16_verify", Some("ga_bf16_prefill")),
                ("ga_fp8_verify", Some("ga_fp8_prefill")),
                ("swa_verify", Some("swa_prefill")),
                ("dflash_block", None),
            ] {
                let variant = named(&format!("mimo{tag}_{verify}"))?;
                let prefill = prefill.map(|prefill| format!("mimo{tag}_{prefill}"));
                let cubin = module(session, variant.profile)?;
                for case in verify_cases(variant) {
                    if filter
                        .as_ref()
                        .is_some_and(|filter| !case.label().starts_with(filter))
                    {
                        continue;
                    }
                    let fixture = prepare(session, case)?;
                    let (partial, merge) = functions(&cubin, case.variant, &fixture.planned)?;
                    fixture.launch(session, &partial, &merge)?;
                    let output = fixture.output(session)?;
                    validate(session, case, &fixture, &output, "verify")?;
                    let Some(prefill) = &prefill else { continue };
                    let prefill_case = Case {
                        variant: named(prefill)?,
                        ..case
                    };
                    let reference = prepare(session, prefill_case)?;
                    let (partial, merge) =
                        functions(&cubin, prefill_case.variant, &reference.planned)?;
                    reference.launch(session, &partial, &merge)?;
                    close(
                        case,
                        &reference.output(session)?,
                        &output,
                        4.0,
                        "prefill kernel",
                    )?;
                }
            }
        }
        Ok(())
    })
    .expect("verify attention correctness");
}

/// Sinks near and past the F32 overflow of exp2(sink * log2(e) - maximum)
/// with small key scores, in the unsplit SWA verify path (batch 32) and the
/// split one (batch 3). V is amplified by 2^60 so the true output stays well
/// inside the normal BF16 range. Sinks more than about 126 / log2(e) above
/// the scores make the rescaling weight an F32 subnormal, as in prefill, so
/// the cases stop at 90.
#[test]
fn attention_verify_large_sink_matches_oracle() {
    gpu_test(2 * 1024 * 1024 * 1024, |session| {
        let variant = named("mimo_swa_verify")?;
        if !variant.query.sink || variant.query.kv_dtype != DType::Bf16 {
            return Err("mimo_swa_verify is not a BF16 sink variant".into());
        }
        let cubin = module(session, variant.profile)?;
        for (batch, rows, prefix, mixed) in [
            (32, 1, 0, false),
            (32, 3, 100, false),
            (32, 8, 60, true),
            (3, 8, 1017, false),
        ] {
            let case = Case {
                variant,
                batch,
                rows,
                prefix,
                mixed,
            };
            for sink in [0.0, 60.0, 88.0, 88.5, 89.0, 90.0] {
                let stress = Some(SinkStress {
                    sink,
                    v_scale: 2.0_f32.powi(60),
                });
                let fixture = prepare_stressed(session, case, 1, stress)?;
                if (batch == 32) != (fixture.planned.split_capacity == 1) {
                    return Err(format!(
                        "{} split capacity {} is not the intended path",
                        case.label(),
                        fixture.planned.split_capacity
                    ));
                }
                let (partial, merge) = functions(&cubin, case.variant, &fixture.planned)?;
                fixture.launch(session, &partial, &merge)?;
                let output = fixture.output(session)?;
                let phase = format!("verify_sink{sink}");
                // Within two BF16 steps of the magnitude of each output's
                // terms, which a zero from an overflowed denominator fails.
                let reference = &fixture.reference;
                for (index, &bits) in output.iter().enumerate() {
                    let actual = f64::from(bf16_to_f32(bits));
                    let error = (actual - reference.values[index]).abs();
                    if error > reference.sum_abs_products[index] / 64.0 {
                        return Err(format!(
                            "{} {phase} differs at {index}: actual={actual:e}, oracle={:e}",
                            case.label(),
                            reference.values[index]
                        ));
                    }
                }
            }
        }
        Ok(())
    })
    .expect("verify attention with large sinks");
}

#[test]
fn attention_verify_graph_determinism_and_pdl() {
    gpu_test(512 * 1024 * 1024, |session| {
        let mut cases = Vec::new();
        for name in [
            "mimo_ga_bf16_verify",
            "mimo_ga_fp8_verify",
            "mimo_swa_verify",
            "mimo_dflash_block",
        ] {
            for (batch, rows, mixed) in [(1, 8, false), (5, 8, true), (16, 2, true)] {
                cases.push(Case {
                    variant: named(name)?,
                    batch,
                    rows,
                    prefix: 1500,
                    mixed,
                });
            }
        }
        run(session, &cases, true)
    })
    .expect("verify attention graph replay");
}

#[test]
fn attention_long_context_split_kv() {
    gpu_test(512 * 1024 * 1024, |session| {
        let cases: Vec<Case> = variants()
            .into_iter()
            .filter(|v| matches!(v.name, "mimo_ga_bf16_decode" | "mimo_ga_fp8_decode"))
            .map(|variant| Case {
                variant,
                batch: 1,
                rows: 1,
                prefix: 65_535,
                mixed: false,
            })
            .collect();
        assert_eq!(cases.len(), 2);
        run(session, &cases, false)
    })
    .expect("attention long context split KV");
}

/// A catalog variant prepared for MiMo steps of up to 256 items.
fn wide(name: &str) -> LabResult<Variant> {
    named(name)?
        .with_batch_capacity(attention::MAX_BATCH_ROWS)
        .map_err(|e| e.to_string())
}

/// Wide steps of `variant`: bucket edges (33 and 65 items pad into the 64
/// and 128 buckets), uniform and mixed contexts, and the widest bucket.
fn wide_cases(variant: Variant, shapes: &[(usize, usize, usize, bool)]) -> Vec<Case> {
    shapes
        .iter()
        .map(|&(batch, rows, prefix, mixed)| Case {
            variant,
            batch,
            rows,
            prefix,
            mixed,
        })
        .collect()
}

/// MiMo decode steps above 32 rows: the shared tables' scan and
/// search, the wide buckets' split plans and merges, and the windowed rows,
/// against the oracle. Mixed rows of 32K down to 512 tokens share one
/// stream-K schedule; the widest bucket also replays bit for bit, as a graph
/// and after a PDL producer.
#[test]
fn attention_wide_decode_matches_oracle() {
    gpu_test(8 * 1024 * 1024 * 1024, |session| {
        for name in [
            "mimo_ga_bf16_decode",
            "mimo_ga_fp8_decode",
            "mimo_swa_decode",
        ] {
            let variant = wide(name)?;
            let cases = wide_cases(
                variant,
                &[
                    (33, 1, 2047, true),
                    (64, 1, 2047, false),
                    (65, 1, 4095, true),
                    (128, 1, 1023, true),
                    (129, 1, 2047, true),
                    (255, 1, 1023, true),
                    (64, 1, 32_767, true),
                    (256, 1, 127, false),
                ],
            );
            run(session, &cases, false)?;
            run(session, &wide_cases(variant, &[(256, 1, 2047, true)]), true)?;
        }
        Ok(())
    })
    .expect("wide attention decode correctness");
}

/// The folded RoPE entries in wide buckets against RoPE and the KV write
/// followed by the unfused entries, bit for bit.
#[test]
fn attention_wide_decode_rope_fold_matches_row_engine() {
    gpu_test(4 * 1024 * 1024 * 1024, |session| {
        let mut cases = Vec::new();
        for variant in folded_variants() {
            let variant = wide(variant.name)?;
            cases.extend(wide_cases(
                variant,
                &[(65, 1, 4095, true), (256, 1, 2047, true)],
            ));
        }
        run_fold(session, &cases)
    })
    .expect("wide folded RoPE decode matches RoPE and KV write plus decode");
}

/// MiMo verify steps above 32 items: the item table sized from the
/// launch's shared bytes, token-packed partials and their merge, against the
/// oracle, at up to 256 query tokens over 33 to 256 items; one wide bucket
/// also replays bit for bit.
#[test]
fn attention_wide_verify_matches_oracle() {
    gpu_test(4 * 1024 * 1024 * 1024, |session| {
        for name in [
            "mimo_ga_bf16_verify",
            "mimo_ga_fp8_verify",
            "mimo_swa_verify",
            "mimo_dflash_block",
        ] {
            let variant = wide(name)?;
            let cases = wide_cases(
                variant,
                &[
                    (33, 1, 1500, true),
                    (64, 4, 1017, true),
                    (65, 3, 4095, true),
                    (128, 2, 127, true),
                    (255, 1, 2047, true),
                    (256, 1, 0, false),
                ],
            );
            run(session, &cases, false)?;
            run(session, &wide_cases(variant, &[(64, 4, 1500, true)]), true)?;
        }
        Ok(())
    })
    .expect("wide verify attention correctness");
}

/// MiMo prefill phases above 32 items: at 256 served
/// sequences one mixed step may carry more than 32 prefill chunks. The
/// prefill kernels walk the live items from the step inputs (no item
/// tables), so the planners' raised capacity is the only change: paged GA
/// (BF16, FP8) and ring SWA prefill at 33 to 256 items, mixed prefixes and
/// chunk lengths up to 8,192 tokens, against the oracle; one step also
/// replays bit for bit, as a graph and after a PDL producer.
#[test]
fn attention_wide_prefill_matches_oracle() {
    gpu_test(4 * 1024 * 1024 * 1024, |session| {
        for name in [
            "mimo_ga_bf16_prefill",
            "mimo_ga_fp8_prefill",
            "mimo_swa_prefill",
        ] {
            let variant = wide(name)?;
            let cases = wide_cases(
                variant,
                &[
                    (33, 8, 1017, true),
                    (64, 16, 500, false),
                    (100, 4, 4095, true),
                    (128, 2, 0, false),
                    (129, 5, 127, true),
                    (255, 3, 2047, true),
                    (256, 32, 1, false),
                ],
            );
            run(session, &cases, false)?;
            run(session, &wide_cases(variant, &[(200, 6, 1500, true)]), true)?;
        }
        Ok(())
    })
    .expect("wide prefill attention correctness");
}

/// The base tree's attention cuda sources (`crates/kernels-sm120/cuda` of
/// the commit before the 256-item tables) and where their cubins go, for
/// the base-versus-tip check.
fn base_paths() -> Option<(PathBuf, PathBuf)> {
    Some((
        PathBuf::from(std::env::var_os("CHOPINFER_ATTENTION_BASE_CUDA")?),
        PathBuf::from(std::env::var_os("CHOPINFER_ATTENTION_BASE_OUT")?),
    ))
}

/// Builds the base tree's MiMo and Qwen3.8 attention cubins (CPU only). The
/// entries are this catalog's, whose parameter lists and bodies the change
/// kept, compiled against the base sources.
#[test]
fn attention_base_cubins_build() {
    let Some((cuda, out)) = base_paths() else {
        kernel_lab::skip("attention_base_cubins_build: set CHOPINFER_ATTENTION_BASE_CUDA and CHOPINFER_ATTENTION_BASE_OUT").unwrap();
        return;
    };
    let specs: Vec<_> = [
        model_mimo26_contract::sm120::config::PROFILE,
        model_qwen38_contract::sm120::config::PROFILE,
    ]
    .into_iter()
    .flat_map(|profile| attention::cubin_specs(kernel_lab::catalogs(), Some(profile)))
    .map(|mut spec| {
        spec.sources_root = cuda.clone();
        spec
    })
    .collect();
    let toolchain = kernel_build::Toolchain::detect().expect("CUDA toolchain");
    let built = kernel_build::build(
        &toolchain,
        &specs,
        &out,
        &out.join("kernels.lock"),
        kernel_build::BuildOptions {
            lock_update: true,
            emit_cargo_directives: false,
            profile_line_info: false,
        },
    )
    .expect("base attention cubins");
    for cubin in built {
        println!(
            "base {} {} {}",
            cubin.name,
            cubin.sha256,
            cubin.path.display()
        );
    }
}

/// Steps of at most 32 items give the base kernels' output
/// bits. Every MiMo and Qwen3.8 decode and verify entry, at buckets 1 to 32
/// with mixed contexts and query lengths, runs this tree's cubin and the
/// base tree's (from `attention_base_cubins_build`) on the same inputs; the
/// folded MiMo decode entries also compare the cache rows they write. The
/// launches are this tree's plans, which are the base plans at these buckets.
#[test]
fn attention_matches_base_bit_for_bit() {
    let Some((_, out)) = base_paths() else {
        kernel_lab::skip("attention_matches_base_bit_for_bit: set CHOPINFER_ATTENTION_BASE_CUDA and CHOPINFER_ATTENTION_BASE_OUT").unwrap();
        return;
    };
    gpu_test(4 * 1024 * 1024 * 1024, |session| {
        let base = |profile: Profile| -> LabResult<Module> {
            let name = match profile {
                model_mimo26_contract::sm120::config::PROFILE => "attention_mimo_sm120",
                model_qwen38_contract::sm120::config::PROFILE => "attention_qwen38_sm120",
                Profile::StdDenseMoe => return Err("no base std cubin".into()),
                // TP2 builds only the gated rows, no paged attention.
                p if p
                    .ranks_of(model_qwen38_contract::sm120::config::MODEL)
                    .is_some() =>
                {
                    return Err("no base cubin for a Qwen3.8 rank profile".into());
                }
                // MiMo's TP2 has no base tree; this test covers TP4.
                _ => return Err("no base cubin at TP2".into()),
            };
            let bytes = fs::read(out.join(format!("{name}.cubin")))
                .map_err(|e| format!("base cubin {name}: {e}; run attention_base_cubins_build"))?;
            session
                .context
                .load_module(&bytes)
                .map_err(|e| e.to_string())
        };
        let shapes = [
            (1, 1500, false),
            (1, 4095, false),
            (2, 1500, true),
            (3, 1500, true),
            (4, 4095, true),
            (8, 1500, true),
            (13, 4095, true),
            (16, 1500, true),
            (32, 1500, true),
            (32, 2043, false),
        ];
        let mut compared = 0;
        for name in [
            "mimo_ga_bf16_decode",
            "mimo_ga_fp8_decode",
            "mimo_swa_decode",
            "mimo_ga_bf16_verify",
            "mimo_ga_fp8_verify",
            "mimo_swa_verify",
            "mimo_dflash_block",
            "qwen38_bf16_decode",
            "qwen38_fp8_decode",
            "qwen38_bf16_verify",
            "qwen38_fp8_verify",
        ] {
            let variant = attention::catalog(kernel_lab::catalogs())
                .into_iter()
                .find(|variant| variant.name == name)
                .ok_or_else(|| format!("missing attention variant {name}"))?;
            let (tip, old) = (module(session, variant.profile)?, base(variant.profile)?);
            let rows = if variant.query.phase == AttentionPhase::Verify {
                8
            } else {
                1
            };
            for (batch, prefix, mixed) in shapes {
                let case = Case {
                    variant,
                    batch,
                    rows,
                    prefix,
                    mixed,
                };
                let fixture = prepare(session, case)?;
                let labelled = |error: String| format!("{}: {error}", case.label());
                let (partial, merge) = functions(&tip, variant, &fixture.planned)?;
                let mut tip_checks = fixture.replay(session)?;
                tip_checks
                    .eager(|| fixture.launch(session, &partial, &merge))
                    .map_err(labelled)?;
                let (partial, merge) = functions(&old, variant, &fixture.planned)?;
                let mut base_checks = fixture.replay(session)?;
                base_checks
                    .eager(|| fixture.launch(session, &partial, &merge))
                    .map_err(labelled)?;
                same_eager(
                    case,
                    "base",
                    &tip_checks,
                    &base_checks,
                    &["attention output"],
                )?;
                compared += 1;
            }
        }
        let rows = session
            .context
            .load_module(ROW_CUBIN)
            .map_err(|e| e.to_string())?;
        for variant in folded_variants() {
            let (tip, old) = (module(session, variant.profile)?, base(variant.profile)?);
            for (batch, prefix, mixed) in shapes {
                let case = Case {
                    variant,
                    batch,
                    rows: 1,
                    prefix,
                    mixed,
                };
                let fold = prepare_fold(session, case)?;
                let fixture = &fold.fixture;
                let labelled = |error: String| format!("{}: {error}", case.label());
                let (k, v) = (&fixture.buffers[&K], &fixture.buffers[&V]);
                let (k_start, v_start) = (bytes(session, k)?, bytes(session, v)?);
                let (k_rows, v_rows) = current_rows(case, fixture)?;
                let (old_k, old_v) = (put_bytes(session, &k_start)?, put_bytes(session, &v_start)?);
                let tip_functions = fold.functions_in(&tip, &rows)?;
                let mut tip_checks = fixture
                    .replay(session)?
                    .state("K cache", k, k_start.clone(), k_rows.clone(), Vec::new())
                    .state("V cache", v, v_start.clone(), v_rows.clone(), Vec::new());
                tip_checks
                    .eager(|| fold.launch_folded(session, &tip_functions, None))
                    .map_err(labelled)?;
                let base_functions = fold.functions_in(&old, &rows)?;
                let mut base_checks = fixture
                    .replay(session)?
                    .state("K cache", &old_k, k_start, k_rows, Vec::new())
                    .state("V cache", &old_v, v_start, v_rows, Vec::new());
                base_checks
                    .eager(|| fold.launch_folded(session, &base_functions, Some((&old_k, &old_v))))
                    .map_err(labelled)?;
                same_eager(
                    case,
                    "base folded",
                    &tip_checks,
                    &base_checks,
                    &["attention output", "K cache", "V cache"],
                )?;
                compared += 1;
            }
        }
        let mut receipt = BTreeMap::new();
        receipt.insert("check".into(), json!("attention_matches_base_bit_for_bit"));
        receipt.insert("cases".into(), json!(compared));
        receipt.insert("base_cubins".into(), json!(out.display().to_string()));
        receipt.insert("output_bitwise".into(), json!(true));
        session.receipt("attention", receipt)?;
        eprintln!("base-versus-tip: {compared} cases bit for bit");
        Ok(())
    })
    .expect("attention output matches the base kernels");
}

#[test]
fn attention_graph_determinism_and_pdl() {
    gpu_test(512 * 1024 * 1024, |session| {
        let cases: Vec<_> = variants()
            .into_iter()
            .filter(|v| {
                matches!(
                    v.name,
                    "mimo_ga_bf16_decode"
                        | "mimo_swa_prefill"
                        | "std128_fp8_decode"
                        | "std128_bf16_prefill"
                )
            })
            .map(|variant| Case {
                variant,
                batch: 1,
                rows: if variant.query.phase == AttentionPhase::Decode {
                    1
                } else {
                    512
                },
                prefix: if variant.query.phase == AttentionPhase::Decode {
                    128
                } else {
                    129
                },
                mixed: false,
            })
            .collect();
        run(session, &cases, true)
    })
    .expect("attention graph, deterministic replay, and PDL");
}

#[test]
fn attention_benchmarks() {
    if std::env::var("CHOPINFER_KERNEL_BENCH").as_deref() != Ok("1") {
        kernel_lab::skip("attention benchmarks: set CHOPINFER_KERNEL_BENCH=1").unwrap();
        return;
    }
    gpu_test(768 * 1024 * 1024, |session| {
        let mut cases = Vec::new();
        for variant in variants() {
            if variant.profile == model_mimo26_contract::sm120::config::PROFILE
                || matches!(variant.name, "std128_bf16_decode" | "std128_bf16_prefill")
            {
                if variant.query.phase == AttentionPhase::Decode {
                    for batch in [1, 8, 32] {
                        for context in [1, 127, 128, 129, 4096] {
                            cases.push(Case {
                                variant,
                                batch,
                                rows: 1,
                                prefix: context - 1,
                                mixed: false,
                            });
                        }
                    }
                    if variant.name == "mimo_ga_bf16_decode" {
                        cases.push(Case {
                            variant,
                            batch: 1,
                            rows: 1,
                            prefix: 65_535,
                            mixed: false,
                        });
                    }
                } else {
                    for rows in [512, 2048, 8192] {
                        cases.push(Case {
                            variant,
                            batch: 1,
                            rows,
                            prefix: if variant.layout == CacheLayout::Ring {
                                129
                            } else {
                                0
                            },
                            mixed: false,
                        });
                    }
                    if variant.name == "mimo_ga_bf16_prefill" {
                        cases.push(Case {
                            variant,
                            batch: 1,
                            rows: 8192,
                            prefix: 32_768,
                            mixed: false,
                        });
                    }
                }
            }
        }
        benchmark(session, &cases)
    })
    .expect("attention benchmarks");
}

#[test]
fn attention_decode_roofline_benchmarks() {
    if std::env::var("CHOPINFER_KERNEL_GRAPH_BENCH").as_deref() != Ok("1") {
        kernel_lab::skip("attention decode roofline: set CHOPINFER_KERNEL_GRAPH_BENCH=1").unwrap();
        return;
    }
    gpu_test(8 * 1024 * 1024 * 1024, |session| {
        let variants = variants();
        let mut cases = Vec::new();
        for name in [
            "mimo_ga_bf16_decode",
            "mimo_ga_fp8_decode",
            "mimo_swa_decode",
        ] {
            let variant = *variants
                .iter()
                .find(|variant| variant.name == name)
                .ok_or_else(|| format!("missing decode variant {name}"))?;
            let contexts: &[usize] = if variant.layout == CacheLayout::Ring {
                &[4096]
            } else {
                &[4096, 32_768, 65_536]
            };
            let batches: &[usize] = if variant.layout == CacheLayout::Ring {
                &[1, 2, 4, 8, 16, 32]
            } else {
                &[1, 8, 16, 32]
            };
            for &batch in batches {
                for &context in contexts {
                    for mixed in [false, true] {
                        if mixed && (batch == 1 || variant.layout == CacheLayout::Ring) {
                            continue;
                        }
                        cases.push(Case {
                            variant,
                            batch,
                            rows: 1,
                            prefix: context - 1,
                            mixed,
                        });
                    }
                }
            }
        }
        decode_roofline_benchmark(session, &cases)
    })
    .expect("attention decode roofline benchmarks");
}

/// Verify steps at C1 (one item of 8 rows), C16 (16 items of 2) and C32
/// (32 items of 1 or 2), against the prefill kernel on the same inputs, and
/// the DFlash block, in the decode roofline graph.
#[test]
fn attention_rope_fold_benchmarks() {
    if std::env::var("CHOPINFER_KERNEL_GRAPH_BENCH").as_deref() != Ok("1") {
        kernel_lab::skip("attention RoPE fold benchmarks: set CHOPINFER_KERNEL_GRAPH_BENCH=1")
            .unwrap();
        return;
    }
    gpu_test(8 * 1024 * 1024 * 1024, |session| {
        let mut cases = Vec::new();
        for variant in folded_variants() {
            for (batch, context, mixed) in [
                (1, 1024, false),
                (1, 4096, false),
                (1, 32_768, false),
                (8, 4096, false),
                (16, 4096, false),
                (32, 4096, false),
                (32, 4096, true),
            ] {
                cases.push(Case {
                    variant,
                    batch,
                    rows: 1,
                    prefix: context - 1,
                    mixed,
                });
            }
        }
        if let Ok(filter) = std::env::var("CHOPINFER_ATTENTION_CASE") {
            cases.retain(|case| case.label().starts_with(&filter));
        }
        rope_fold_benchmark(session, &cases)
    })
    .expect("attention RoPE fold benchmarks");
}

#[test]
fn attention_verify_benchmarks() {
    if std::env::var("CHOPINFER_KERNEL_GRAPH_BENCH").as_deref() != Ok("1") {
        kernel_lab::skip("attention verify benchmarks: set CHOPINFER_KERNEL_GRAPH_BENCH=1")
            .unwrap();
        return;
    }
    gpu_test(8 * 1024 * 1024 * 1024, |session| {
        let mut cases = Vec::new();
        for name in [
            "mimo_ga_bf16_verify",
            "mimo_ga_bf16_prefill",
            "mimo_ga_fp8_verify",
            "mimo_ga_fp8_prefill",
            "mimo_swa_verify",
            "mimo_swa_prefill",
        ] {
            let variant = named(name)?;
            let contexts: &[usize] = if variant.layout == CacheLayout::Ring {
                &[4096]
            } else {
                &[4096, 32_768]
            };
            for &context in contexts {
                for (batch, rows) in [(1, 8), (16, 2), (32, 1), (32, 2)] {
                    cases.push(Case {
                        variant,
                        batch,
                        rows,
                        prefix: context - rows,
                        mixed: false,
                    });
                }
            }
        }
        for batch in [1, 16, 32] {
            cases.push(Case {
                variant: named("mimo_dflash_block")?,
                batch,
                rows: 8,
                prefix: 4096,
                mixed: false,
            });
        }
        decode_roofline_benchmark(session, &cases)
    })
    .expect("attention verify benchmarks");
}

#[test]
fn attention_graph_benchmarks() {
    if std::env::var("CHOPINFER_KERNEL_GRAPH_BENCH").as_deref() != Ok("1") {
        kernel_lab::skip("attention graph benchmarks: set CHOPINFER_KERNEL_GRAPH_BENCH=1").unwrap();
        return;
    }
    gpu_test(3 * 1024 * 1024 * 1024, |session| {
        let variants = variants();
        let mut cases = Vec::new();
        for (name, batch, prefix) in [
            ("mimo_swa_decode", 1, 127),
            ("mimo_swa_decode", 8, 127),
            ("mimo_swa_decode", 32, 127),
            ("mimo_ga_bf16_decode", 1, 4095),
            ("mimo_ga_bf16_decode", 1, 65_535),
        ] {
            let variant = *variants
                .iter()
                .find(|variant| variant.name == name)
                .ok_or_else(|| format!("missing attention graph benchmark variant {name}"))?;
            cases.push(Case {
                variant,
                batch,
                rows: 1,
                prefix,
                mixed: false,
            });
        }
        graph_benchmark(session, &cases)
    })
    .expect("attention graph benchmarks");
}

#[path = "attention_exact_rows.rs"]
mod exact_rows;
#[path = "attention_verify_drift.rs"]
mod verify_drift;
#[path = "attention_verify_precision.rs"]
mod verify_precision;

#[rustfmt::skip] #[path = "rope_fold_positions.rs"] mod rope_fold_positions;
