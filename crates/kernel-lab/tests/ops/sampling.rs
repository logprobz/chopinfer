use kernel_core::codec::{bf16_to_f32, f32_to_bf16};
use kernel_core::contract::{DType, RuntimeCounts};
use kernel_lab::launch::{self, Bindings};
use kernel_lab::poison::{self, Replay, WriteMap};
use kernel_lab::{LabResult, Session, gpu_test, time};
use kernels_sm120::Profile;
use kernels_sm120::sampling::{
    self, FilteredSamplingBuffers, SamplingBuffers, SamplingQuery, SimpleSamplingBuffers,
};
use runtime_cuda::{DeviceBuffer, Function, KernelArgument};
use runtime_sampling::{
    LogitBias, Sample, SampleContext, SamplingParameters, SamplingState, sample,
};
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;

const LOGITS: u64 = 1;
const META: u64 = 2;
const COUNTS: u64 = 3;
const BIASES: u64 = 4;
const EOS: u64 = 5;
const ADJUSTED: u64 = 6;
const ORDER: u64 = 7;
const OUTPUT: u64 = 8;
const PARTIALS: u64 = 9;
const FILTER: u64 = 10;
const MASKS: u64 = 11;
const RECORD_BYTES: usize = 256;
const META_BYTES: usize = 112;
const LOGPROB_TOLERANCE: f64 = 1e-6;

#[derive(Clone)]
struct Row {
    logits: Vec<f32>,
    parameters: SamplingParameters,
    state: SamplingState,
    history: Vec<u32>,
    emitted: usize,
    eos: Vec<u32>,
    /// Constraint mask: bit `t % 32` of word `t / 32` admits token `t`.
    mask: Option<Vec<u32>>,
}

impl Row {
    fn expected(&self) -> LabResult<Sample> {
        let mut state = self.state;
        let context = SampleContext::new(&self.history, self.emitted, &self.eos);
        let context = match &self.mask {
            Some(mask) => context.with_allowed_tokens(mask),
            None => context,
        };
        sample(&self.parameters, &self.logits, context, &mut state)
            .map_err(|error| error.to_string())
    }

    /// Whether EOS stays masked: the minimum length yields when the
    /// constraint admits nothing but stop tokens, as on the host.
    fn masks_eos(&self) -> bool {
        self.emitted < self.parameters.minimum_new_tokens()
            && !self.mask.as_ref().is_some_and(|mask| {
                let mut rest = mask.clone();
                for &token in &self.eos {
                    rest[token as usize / 32] &= !(1 << (token % 32));
                }
                rest.iter().all(|word| *word == 0)
            })
    }
}

#[derive(Clone, Debug, PartialEq)]
struct ResultRow {
    token: u32,
    logprob: f64,
    top: Vec<(u32, f64)>,
}

struct Encoded {
    meta: Vec<u8>,
    counts: Vec<u8>,
    biases: Vec<u8>,
    eos: Vec<u8>,
    /// Constraint masks, `mask_words` per row, or empty when no row has one.
    masks: Vec<u8>,
    mask_words: usize,
}

fn u32_at(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(
        bytes[offset..offset + 4]
            .try_into()
            .expect("bounded record"),
    )
}

fn f64_at(bytes: &[u8], offset: usize) -> f64 {
    f64::from_bits(u64::from_le_bytes(
        bytes[offset..offset + 8]
            .try_into()
            .expect("bounded record"),
    ))
}

fn put_u32(bytes: &mut [u8], offset: usize, value: u32) {
    bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

fn put_u64(bytes: &mut [u8], offset: usize, value: u64) {
    bytes[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
}

fn put_f64(bytes: &mut [u8], offset: usize, value: f64) {
    put_u64(bytes, offset, value.to_bits());
}

fn encode(rows: &[Row]) -> Encoded {
    let mut meta = vec![0; rows.len() * META_BYTES];
    let mut counts = Vec::new();
    let mut biases = Vec::new();
    let mut eos = Vec::new();
    for (index, row) in rows.iter().enumerate() {
        let record = &mut meta[index * META_BYTES..(index + 1) * META_BYTES];
        let count_start = counts.len() / 16;
        let bias_start = biases.len() / 16;
        let eos_start = eos.len() / 4;
        let prompt = row.history.len() - row.emitted;
        let mut seen = BTreeMap::<u32, (bool, u32)>::new();
        for (position, &token) in row.history.iter().enumerate() {
            let entry = seen.entry(token).or_default();
            if position < prompt {
                entry.0 = true;
            } else {
                entry.1 += 1;
            }
        }
        for (token, (prompt_seen, output_count)) in seen {
            counts.extend_from_slice(&token.to_le_bytes());
            counts.extend_from_slice(&u32::from(prompt_seen).to_le_bytes());
            counts.extend_from_slice(&output_count.to_le_bytes());
            counts.extend_from_slice(&0_u32.to_le_bytes());
        }
        for bias in row.parameters.logit_bias() {
            biases.extend_from_slice(&bias.token_id().to_le_bytes());
            biases.extend_from_slice(&0_u32.to_le_bytes());
            biases.extend_from_slice(&bias.bias().to_le_bytes());
        }
        let mask_eos = row.masks_eos();
        if mask_eos {
            for &token in &row.eos {
                eos.extend_from_slice(&token.to_le_bytes());
            }
        }
        put_u64(record, 0, row.state.seed());
        put_u64(record, 8, row.state.generated());
        put_u64(record, 16, count_start as u64);
        put_u64(record, 24, bias_start as u64);
        put_u64(record, 32, eos_start as u64);
        put_u32(record, 40, (counts.len() / 16 - count_start) as u32);
        put_u32(record, 44, (biases.len() / 16 - bias_start) as u32);
        put_u32(record, 48, (eos.len() / 4 - eos_start) as u32);
        put_u32(record, 52, row.parameters.top_k() as u32);
        put_u32(record, 56, row.parameters.top_logprobs() as u32);
        put_u32(record, 60, u32::from(mask_eos));
        put_f64(record, 64, row.parameters.temperature());
        put_f64(record, 72, row.parameters.top_p());
        put_f64(record, 80, row.parameters.min_p());
        put_f64(record, 88, row.parameters.repetition_penalty());
        put_f64(record, 96, row.parameters.frequency_penalty());
        put_f64(record, 104, row.parameters.presence_penalty());
    }
    let width = rows[0].logits.len();
    let mask_words = if rows.iter().any(|row| row.mask.is_some()) {
        width.div_ceil(32)
    } else {
        0
    };
    let mut masks = Vec::with_capacity(rows.len() * mask_words * 4);
    for row in rows {
        match &row.mask {
            Some(mask) => mask
                .iter()
                .for_each(|word| masks.extend_from_slice(&word.to_le_bytes())),
            None => masks.resize(masks.len() + mask_words * 4, 0xff),
        }
    }
    Encoded {
        meta,
        counts,
        biases,
        eos,
        masks,
        mask_words,
    }
}

fn random_word(state: &mut u64) -> u32 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    (*state >> 32) as u32
}

fn random_row(width: usize, dtype: DType, seed: u64, temperature: f64) -> Row {
    let mut rng = seed;
    let logits = (0..width)
        .map(|_| {
            let value = ((random_word(&mut rng) % 2049) as i32 - 1024) as f32 / 128.0;
            if dtype == DType::Bf16 {
                bf16_to_f32(f32_to_bf16(value))
            } else {
                value
            }
        })
        .collect();
    Row {
        logits,
        parameters: SamplingParameters::new(temperature).expect("valid temperature"),
        state: SamplingState::new(seed ^ 0xace5_8219_68ce_1029),
        history: Vec::new(),
        emitted: 0,
        eos: Vec::new(),
        mask: None,
    }
}

/// The record bytes a sampling launch writes for `rows` (sampling.cuh
/// `simple_finish`, filtered.cuh): each live row's token, status and logprob,
/// and its first `min(top_logprobs, admitted)` top ids and logprobs. The host
/// fills later ranks of masked rows, and rows past the live count are never
/// written. Elements are 4 bytes, the top-id width, so a write to one id
/// past the requested count fails; each half of an 8-byte logprob must
/// change.
fn record_map(rows: &[Row]) -> LabResult<WriteMap> {
    let mut ranges = Vec::new();
    for (index, row) in rows.iter().enumerate() {
        let base = index * RECORD_BYTES;
        let admitted = row.mask.as_ref().map_or(row.logits.len(), |mask| {
            mask.iter().map(|word| word.count_ones() as usize).sum()
        });
        let top = row.parameters.top_logprobs().min(admitted);
        ranges.push(base..base + 16);
        ranges.push(base + 16..base + 16 + 4 * top);
        ranges.push(base + 96..base + 96 + 8 * top);
    }
    WriteMap::of(DType::F32).byte_ranges(ranges)
}

#[test]
fn record_map_rejects_an_extra_top_id() {
    let mut row = random_row(64, DType::F32, 7, 1.0);
    row.parameters = row.parameters.with_top_logprobs(1).unwrap();
    let map = record_map(std::slice::from_ref(&row)).unwrap();
    let mut record = vec![0xff; RECORD_BYTES];
    record[..16].fill(0);
    record[16..20].fill(0);
    record[96..104].fill(0);
    poison::check("one top id", &record, &map, poison::Fill::Pattern(&[0xff])).unwrap();
    record[20..24].fill(0);
    assert!(
        poison::check("stray top id", &record, &map, poison::Fill::Pattern(&[0xff])).is_err(),
        "a write to top_ids[1] with one requested id passed"
    );
}

struct Fixture {
    buffers: BTreeMap<u64, DeviceBuffer>,
    width: usize,
    dtype: DType,
    capacity: usize,
    /// Whether the uploaded batch carries constraint masks.
    masked: std::cell::Cell<bool>,
}

impl Fixture {
    fn new(session: &Session, width: usize, dtype: DType, capacity: usize) -> LabResult<Self> {
        let element_bytes = if dtype == DType::Bf16 { 2 } else { 4 };
        let mut buffers = BTreeMap::new();
        for (id, bytes) in [
            (LOGITS, capacity * width * element_bytes),
            (META, capacity * META_BYTES),
            (COUNTS, capacity * 64 * 16),
            (BIASES, capacity * 16 * 16),
            (EOS, capacity * 16 * 4),
            (ADJUSTED, capacity * width * 8),
            (ORDER, capacity * width * 4),
            (OUTPUT, capacity * RECORD_BYTES),
            (
                PARTIALS,
                capacity
                    * (width as u32).div_ceil(sampling::SIMPLE_TILE) as usize
                    * sampling::SIMPLE_PARTIAL_BYTES as usize,
            ),
            (FILTER, capacity * sampling::FILTER_ROW_BYTES as usize),
            (MASKS, capacity * width.div_ceil(32) * 4),
        ] {
            buffers.insert(id, session.allocate(bytes.max(1))?);
        }
        let fixture = Self {
            buffers,
            width,
            dtype,
            capacity,
            masked: std::cell::Cell::new(false),
        };
        fixture.poison(session)?;
        Ok(fixture)
    }

    /// Poisons the output records and the scratch the stages write before
    /// later stages read it: partials, filter rows, adjusted values and
    /// orders (`filtered_stats` resets every filter bin and state).
    fn poison(&self, session: &Session) -> LabResult<()> {
        let buffers = [OUTPUT, PARTIALS, FILTER, ADJUSTED, ORDER].map(|id| &self.buffers[&id]);
        poison::poison(session, &buffers)
    }

    /// The buffers of an independent second invocation: its records and the
    /// scratch its stages write (`poison`).
    fn second(&self, session: &Session) -> LabResult<BTreeMap<u64, DeviceBuffer>> {
        [OUTPUT, PARTIALS, FILTER, ADJUSTED, ORDER]
            .into_iter()
            .map(|id| Ok((id, session.allocate(self.buffers[&id].byte_len())?)))
            .collect()
    }

    /// Poisoned checks of one sampling launch over `rows` (`record_map`),
    /// with the partials, and on the filtered path the filter rows, adjusted
    /// values and orders, as scratch.
    fn replay<'a>(
        &'a self,
        session: &'a Session,
        rows: &[Row],
        filtered: bool,
    ) -> LabResult<Replay<'a>> {
        let mut checks = Replay::new(session)
            .output("records", &self.buffers[&OUTPUT], record_map(rows)?)
            .scratch("partials", &self.buffers[&PARTIALS]);
        if filtered {
            checks = checks
                .scratch("filter rows", &self.buffers[&FILTER])
                .scratch("adjusted values", &self.buffers[&ADJUSTED])
                .scratch("orders", &self.buffers[&ORDER]);
        }
        Ok(checks)
    }

    fn upload(&self, session: &Session, rows: &[Row]) -> LabResult<()> {
        if rows.is_empty() || rows.len() > self.capacity {
            return Err("sampling batch outside fixture capacity".into());
        }
        let mut logits = Vec::with_capacity(rows.len() * self.width * 4);
        for row in rows {
            if row.logits.len() != self.width {
                return Err("sampling row width mismatch".into());
            }
            for &value in &row.logits {
                if self.dtype == DType::Bf16 {
                    logits.extend_from_slice(&f32_to_bf16(value).to_le_bytes());
                } else {
                    logits.extend_from_slice(&value.to_le_bytes());
                }
            }
        }
        let encoded = encode(rows);
        self.masked.set(encoded.mask_words != 0);
        for (id, bytes) in [
            (LOGITS, logits.as_slice()),
            (META, encoded.meta.as_slice()),
            (COUNTS, encoded.counts.as_slice()),
            (BIASES, encoded.biases.as_slice()),
            (EOS, encoded.eos.as_slice()),
            (MASKS, encoded.masks.as_slice()),
        ] {
            if bytes.len() > self.buffers[&id].byte_len() {
                return Err(format!("sampling buffer {id} is too small"));
            }
            if !bytes.is_empty() {
                self.buffers[&id]
                    .copy_from_bytes(bytes, &session.stream)
                    .map_err(|error| error.to_string())?;
            }
        }
        session
            .stream
            .synchronize()
            .map_err(|error| error.to_string())
    }

    fn launch(&self, session: &Session, function: &Function, rows: usize) -> LabResult<()> {
        let query = SamplingQuery {
            row_capacity: self.capacity as u32,
            width_capacity: self.width as u32,
            dtype: self.dtype,
        };
        let planned = sampling::plan(
            query,
            RuntimeCounts {
                m: rows as u32,
                batch: rows as u32,
                context: 0,
                routed_rows: 0,
                split_k: 1,
            },
            self.width as u32,
            self.width as u64,
            SamplingBuffers {
                logits: LOGITS,
                metadata: META,
                counts: COUNTS,
                biases: BIASES,
                eos: EOS,
                adjusted: ADJUSTED,
                order: ORDER,
                output: OUTPUT,
                masks: self.masked.get().then_some(MASKS),
            },
        )
        .map_err(|error| error.to_string())?;
        let bindings = Bindings {
            buffers: self
                .buffers
                .iter()
                .map(|(&id, buffer)| (id, buffer))
                .collect(),
        };
        launch::launch(
            &session.context,
            function,
            &session.stream,
            planned.launch,
            &planned.arguments,
            &bindings,
        )
    }

    fn launch_simple(
        &self,
        session: &Session,
        partials: &Function,
        finish: &Function,
        rows: usize,
        top: bool,
    ) -> LabResult<()> {
        let planned = sampling::plan_simple(
            SamplingQuery {
                row_capacity: self.capacity as u32,
                width_capacity: self.width as u32,
                dtype: self.dtype,
            },
            RuntimeCounts {
                m: rows as u32,
                batch: rows as u32,
                context: 0,
                routed_rows: 0,
                split_k: 1,
            },
            self.width as u32,
            self.width as u64,
            SimpleSamplingBuffers {
                logits: LOGITS,
                metadata: META,
                partials: PARTIALS,
                output: OUTPUT,
            },
            top,
        )
        .map_err(|error| error.to_string())?;
        let bindings = Bindings {
            buffers: self
                .buffers
                .iter()
                .map(|(&id, buffer)| (id, buffer))
                .collect(),
        };
        launch::launch(
            &session.context,
            partials,
            &session.stream,
            planned.partials,
            &planned.partial_args,
            &bindings,
        )?;
        launch::launch(
            &session.context,
            finish,
            &session.stream,
            planned.finish,
            &planned.finish_args,
            &bindings,
        )
    }

    fn launch_filtered(
        &self,
        session: &Session,
        functions: &FilteredFunctions,
        rows: usize,
    ) -> LabResult<()> {
        self.launch_filtered_rebound(session, functions, rows, &[], true)
    }

    /// `launch_filtered` with the buffers of the given ids replaced, and
    /// without the stages before `finish` unless `producers`.
    fn launch_filtered_rebound(
        &self,
        session: &Session,
        functions: &FilteredFunctions,
        rows: usize,
        rebound: &[(u64, &DeviceBuffer)],
        producers: bool,
    ) -> LabResult<()> {
        let planned = sampling::plan_filtered(
            SamplingQuery {
                row_capacity: self.capacity as u32,
                width_capacity: self.width as u32,
                dtype: self.dtype,
            },
            RuntimeCounts {
                m: rows as u32,
                batch: rows as u32,
                context: 0,
                routed_rows: 0,
                split_k: 1,
            },
            self.width as u32,
            self.width as u64,
            FilteredSamplingBuffers {
                logits: LOGITS,
                metadata: META,
                counts: COUNTS,
                biases: BIASES,
                eos: EOS,
                partials: PARTIALS,
                filter: FILTER,
                adjusted: ADJUSTED,
                order: ORDER,
                output: OUTPUT,
                masks: self.masked.get().then_some(MASKS),
            },
        )
        .map_err(|error| error.to_string())?;
        let mut bindings = Bindings {
            buffers: self
                .buffers
                .iter()
                .map(|(&id, buffer)| (id, buffer))
                .collect(),
        };
        bindings.buffers.extend(rebound.iter().copied());
        let stages = [
            (&functions.stats, planned.stats),
            (&functions.histogram, planned.histogram),
            (&functions.collect, planned.collect),
            (&functions.resolve, planned.resolve),
            (&functions.gumbel, planned.gumbel),
        ];
        let producing = if producers { functions.stages } else { 0 };
        for (function, plan) in stages.into_iter().take(producing) {
            launch::launch(
                &session.context,
                function,
                &session.stream,
                plan,
                &planned.tile_args,
                &bindings,
            )?;
        }
        if functions.stages < 6 {
            return Ok(());
        }
        launch::launch(
            &session.context,
            &functions.finish,
            &session.stream,
            planned.finish,
            &planned.finish_args,
            &bindings,
        )
    }

    /// Rows whose cutoff needed the exact sequential fallback.
    fn exact_fallbacks(&self, session: &Session, rows: usize) -> LabResult<Vec<bool>> {
        let record = sampling::FILTER_ROW_BYTES as usize;
        let mut bytes = vec![0; rows * record];
        self.buffers[&FILTER]
            .copy_to_bytes(&mut bytes, &session.stream)
            .map_err(|error| error.to_string())?;
        Ok((0..rows)
            .map(|row| u32_at(&bytes, row * record + 36) != 0)
            .collect())
    }

    /// Candidates each row collected for its cutoff window.
    fn candidates(&self, session: &Session, rows: usize) -> LabResult<Vec<u32>> {
        let record = sampling::FILTER_ROW_BYTES as usize;
        let mut bytes = vec![0; rows * record];
        self.buffers[&FILTER]
            .copy_to_bytes(&mut bytes, &session.stream)
            .map_err(|error| error.to_string())?;
        Ok((0..rows)
            .map(|row| u32_at(&bytes, row * record + 8))
            .collect())
    }

    fn output(&self, session: &Session, rows: &[Row]) -> LabResult<Vec<ResultRow>> {
        let mut bytes = vec![0; rows.len() * RECORD_BYTES];
        self.buffers[&OUTPUT]
            .copy_to_bytes(&mut bytes, &session.stream)
            .map_err(|error| error.to_string())?;
        bytes
            .chunks_exact(RECORD_BYTES)
            .zip(rows)
            .enumerate()
            .map(|(index, (record, row))| {
                let status = u32_at(record, 4);
                if status != 0 {
                    return Err(format!("sampling row {index} returned status {status}"));
                }
                // A constrained row ranks only the tokens its mask admits.
                let admitted = row.mask.as_ref().map_or(row.logits.len(), |mask| {
                    mask.iter().map(|word| word.count_ones() as usize).sum()
                });
                let mut top: Vec<_> = (0..row.parameters.top_logprobs().min(admitted))
                    .map(|rank| (u32_at(record, 16 + 4 * rank), f64_at(record, 96 + 8 * rank)))
                    .collect();
                if let Some(mask) = &row.mask {
                    let remaining = row.parameters.top_logprobs().min(row.logits.len()) - top.len();
                    top.extend(
                        (0..row.logits.len())
                            .filter(|token| mask[token / 32] & (1 << (token % 32)) == 0)
                            .take(remaining)
                            .map(|token| (token as u32, f64::NEG_INFINITY)),
                    );
                }
                Ok(ResultRow {
                    token: u32_at(record, 0),
                    logprob: f64_at(record, 8),
                    top,
                })
            })
            .collect()
    }
}

fn function(session: &Session, dtype: DType) -> LabResult<(runtime_cuda::Module, Function)> {
    let cubin = include_bytes!(concat!(env!("OUT_DIR"), "/sampling_sm120.cubin"));
    let module = session
        .context
        .load_module(cubin)
        .map_err(|error| error.to_string())?;
    let symbol = if dtype == DType::Bf16 {
        "chop_sampling_bf16_v1"
    } else {
        "chop_sampling_f32_v1"
    };
    let function = module.function(symbol).map_err(|error| error.to_string())?;
    Ok((module, function))
}

fn simple_functions(
    module: &runtime_cuda::Module,
    dtype: DType,
    top: bool,
) -> LabResult<(Function, Function)> {
    let suffix = if dtype == DType::Bf16 { "bf16" } else { "f32" };
    let partial_symbol = if top {
        format!("chop_sampling_simple_top_partials_{suffix}_v1")
    } else {
        format!("chop_sampling_simple_partials_{suffix}_v1")
    };
    Ok((
        module
            .function(&partial_symbol)
            .map_err(|error| error.to_string())?,
        module
            .function(&format!("chop_sampling_simple_finish_{suffix}_v1"))
            .map_err(|error| error.to_string())?,
    ))
}

fn greedy_functions(
    module: &runtime_cuda::Module,
    dtype: DType,
) -> LabResult<(Function, Function)> {
    let suffix = if dtype == DType::Bf16 { "bf16" } else { "f32" };
    Ok((
        module
            .function(&format!("chop_sampling_greedy_partials_1024_{suffix}_v1"))
            .map_err(|error| error.to_string())?,
        module
            .function(&format!("chop_sampling_simple_finish_{suffix}_v1"))
            .map_err(|error| error.to_string())?,
    ))
}

struct FilteredFunctions {
    stats: Function,
    histogram: Function,
    collect: Function,
    resolve: Function,
    gumbel: Function,
    finish: Function,
    /// Launch only this many stages; six runs the whole path.
    stages: usize,
}

fn filtered_functions(module: &runtime_cuda::Module, dtype: DType) -> LabResult<FilteredFunctions> {
    let suffix = if dtype == DType::Bf16 { "bf16" } else { "f32" };
    let load = |stage: &str| {
        module
            .function(&format!("chop_sampling_filtered_{stage}_{suffix}_v1"))
            .map_err(|error| error.to_string())
    };
    Ok(FilteredFunctions {
        stats: load("stats")?,
        histogram: load("histogram")?,
        collect: load("collect")?,
        resolve: load("resolve")?,
        gumbel: load("gumbel")?,
        finish: load("finish")?,
        stages: 6,
    })
}

/// CPU oracle samples, computed on all cores.
fn expected_rows(rows: &[Row]) -> LabResult<Vec<Sample>> {
    let threads = std::thread::available_parallelism().map_or(8, |n| n.get());
    let chunk = rows.len().div_ceil(threads).max(1);
    std::thread::scope(|scope| {
        let handles: Vec<_> = rows
            .chunks(chunk)
            .map(|part| scope.spawn(move || part.iter().map(Row::expected).collect::<Vec<_>>()))
            .collect();
        let mut all = Vec::with_capacity(rows.len());
        for handle in handles {
            for sample in handle.join().map_err(|_| "CPU oracle thread panicked")? {
                all.push(sample?);
            }
        }
        Ok(all)
    })
}

/// Token-exact comparison without a receipt; returns the worst logprob error.
fn check_tokens(
    label: &str,
    expected: &[Sample],
    actual: &[ResultRow],
    offset: usize,
) -> LabResult<f64> {
    let mut worst = 0.0_f64;
    for (local, (want, got)) in expected.iter().zip(actual).enumerate() {
        let index = offset + local;
        if got.token != want.token_id() {
            return Err(format!(
                "{label} row {index}: token {} != CPU {}",
                got.token,
                want.token_id()
            ));
        }
        if got.top.len() != want.top_logprobs().len() {
            return Err(format!("{label} row {index}: top-logprob count differs"));
        }
        worst = worst.max((got.logprob - want.logprob()).abs());
        for (rank, ((token, value), oracle)) in got.top.iter().zip(want.top_logprobs()).enumerate()
        {
            if *token != oracle.token_id() {
                return Err(format!(
                    "{label} row {index} top rank {rank}: token {token} != CPU {}",
                    oracle.token_id()
                ));
            }
            if *value == f64::NEG_INFINITY || oracle.logprob() == f64::NEG_INFINITY {
                if *value != oracle.logprob() {
                    return Err(format!("{label} row {index}: masked tail differs"));
                }
            } else {
                worst = worst.max((value - oracle.logprob()).abs());
            }
        }
    }
    if worst > LOGPROB_TOLERANCE {
        return Err(format!(
            "{label}: logprob error {worst} exceeds {LOGPROB_TOLERANCE}"
        ));
    }
    Ok(worst)
}

fn compare(
    session: &Session,
    label: &str,
    dtype: DType,
    rows: &[Row],
    actual: &[ResultRow],
    offset: usize,
) -> LabResult<()> {
    let mut worst = (0_usize, 0_f64, "selected", 0_f64, 0_f64);
    let oracle = expected_rows(rows)?;
    for (local, ((_, got), expected)) in rows.iter().zip(actual).zip(oracle).enumerate() {
        let index = offset + local;
        if got.token != expected.token_id() {
            return failed_token(
                session,
                label,
                index,
                "selected",
                got.token,
                expected.token_id(),
            );
        }
        if got.top.len() != expected.top_logprobs().len() {
            return Err(format!(
                "{label} row {index}: top-logprob count {} != CPU {}",
                got.top.len(),
                expected.top_logprobs().len()
            ));
        }
        let selected_error = (got.logprob - expected.logprob()).abs();
        if selected_error > worst.1 {
            worst = (
                index,
                selected_error,
                "selected",
                got.logprob,
                expected.logprob(),
            );
        }
        for (rank, ((token, value), oracle)) in
            got.top.iter().zip(expected.top_logprobs()).enumerate()
        {
            if *token != oracle.token_id() {
                return failed_token(
                    session,
                    label,
                    index,
                    &format!("top rank {rank}"),
                    *token,
                    oracle.token_id(),
                );
            }
            let error = (value - oracle.logprob()).abs();
            if error > worst.1 {
                worst = (index, error, "top", *value, oracle.logprob());
            }
        }
    }
    let mut receipt = BTreeMap::new();
    receipt.insert("case".into(), json!(label));
    receipt.insert("dtype".into(), json!(format!("{dtype:?}")));
    receipt.insert("rows".into(), json!(rows.len()));
    receipt.insert("row_offset".into(), json!(offset));
    receipt.insert("worst_row".into(), json!(worst.0));
    receipt.insert("worst_kind".into(), json!(worst.2));
    receipt.insert("worst_actual".into(), json!(worst.3));
    receipt.insert("worst_expected".into(), json!(worst.4));
    receipt.insert("worst_logprob_error".into(), json!(worst.1));
    receipt.insert("logprob_tolerance".into(), json!(LOGPROB_TOLERANCE));
    receipt.insert("correct".into(), json!(worst.1 <= LOGPROB_TOLERANCE));
    session.receipt("sampling", receipt)?;
    if worst.1 > LOGPROB_TOLERANCE {
        return Err(format!(
            "{label} row {} {} logprob {} != CPU {}, error {} exceeds {}",
            worst.0, worst.2, worst.3, worst.4, worst.1, LOGPROB_TOLERANCE
        ));
    }
    Ok(())
}

fn failed_token(
    session: &Session,
    label: &str,
    row: usize,
    field: &str,
    actual: u32,
    expected: u32,
) -> LabResult<()> {
    let mut receipt = BTreeMap::new();
    receipt.insert("case".into(), json!(label));
    receipt.insert("correct".into(), json!(false));
    receipt.insert("worst_row".into(), json!(row));
    receipt.insert("worst_kind".into(), json!(field));
    receipt.insert("worst_actual".into(), json!(actual));
    receipt.insert("worst_expected".into(), json!(expected));
    session.receipt("sampling", receipt)?;
    Err(format!(
        "{label} row {row} {field}: token {actual} != CPU {expected}"
    ))
}

fn check_exact(label: &str, expected: &[ResultRow], actual: &[ResultRow]) -> LabResult<()> {
    for (index, (left, right)) in expected.iter().zip(actual).enumerate() {
        if left.token != right.token
            || left.logprob.to_bits() != right.logprob.to_bits()
            || left.top.len() != right.top.len()
            || left
                .top
                .iter()
                .zip(&right.top)
                .any(|(a, b)| a.0 != b.0 || a.1.to_bits() != b.1.to_bits())
        {
            return Err(format!("{label} row {index} differs on replay"));
        }
    }
    Ok(())
}

fn run_once(
    session: &Session,
    rows: &[Row],
    dtype: DType,
    label: &str,
    simple: bool,
) -> LabResult<()> {
    let fixture = Fixture::new(session, rows[0].logits.len(), dtype, rows.len())?;
    fixture.upload(session, rows)?;
    let (module, kernel) = function(session, dtype)?;
    if simple {
        let top = rows.iter().any(|row| row.parameters.top_logprobs() != 0);
        let (partials, finish) = if top {
            simple_functions(&module, dtype, true)?
        } else if rows.iter().all(|row| row.parameters.temperature() == 0.0) {
            greedy_functions(&module, dtype)?
        } else {
            simple_functions(&module, dtype, false)?
        };
        fixture.launch_simple(session, &partials, &finish, rows.len(), top)?;
    } else {
        fixture.launch(session, &kernel, rows.len())?;
    }
    let actual = fixture.output(session, rows)?;
    compare(session, label, dtype, rows, &actual, 0)
}

/// `count` random greedy rows from `seed` with tied maxima at tokens 3 and 7.
fn greedy_exact_rows(width: usize, dtype: DType, seed: u64, count: u64) -> Vec<Row> {
    (0..count)
        .map(|index| {
            let mut row = random_row(width, dtype, seed + index, 0.0);
            row.logits[3] = 20.0;
            row.logits[7] = 20.0;
            row
        })
        .collect()
}

/// `greedy_exact`'s cases: eight greedy rows, the same with 20 top
/// logprobs, and at the widest width 32 greedy rows.
fn greedy_exact_cases(width: usize, dtype: DType) -> Vec<(String, Vec<Row>)> {
    let rows = greedy_exact_rows(width, dtype, 0x915b_3211, 8);
    let mut top = rows.clone();
    for row in &mut top {
        row.parameters = SamplingParameters::new(0.0)
            .unwrap()
            .with_top_logprobs(20)
            .unwrap();
    }
    let mut cases = vec![
        (format!("greedy_w{width}"), rows),
        (format!("greedy_top20_w{width}"), top),
    ];
    if width == 152_576 {
        cases.push((
            "greedy_b32_w152576".into(),
            greedy_exact_rows(width, dtype, 0x7842_1afa, 32),
        ));
    }
    cases
}

#[test]
fn sampling_greedy_exact() {
    gpu_test(512 * 1024 * 1024, |session| {
        for dtype in [DType::F32, DType::Bf16] {
            for width in [38_144, 152_576] {
                for (label, rows) in greedy_exact_cases(width, dtype) {
                    run_once(session, &rows, dtype, &label, true)?;
                }
            }
        }
        Ok(())
    })
    .expect("sampling greedy exactness");
}

/// 32 rows from `start` with two top logprobs, greedy and sampled
/// alternately: ties across tile boundaries, adjacent representable maxima
/// and all-zero rows with a negative zero.
fn top_one_rows(width: usize, dtype: DType, start: u64) -> Vec<Row> {
    (start..start + 32)
        .map(|index| {
            let temperature = if index % 2 == 0 { 0.0 } else { 0.8 };
            let mut row = random_row(width, dtype, 0xabc0_0000 + index, temperature);
            row.logits[3] = 20.0;
            row.logits[1024] = if index % 4 < 2 {
                20.0
            } else {
                match dtype {
                    DType::Bf16 => 20.125,
                    _ => f32::from_bits(20.0_f32.to_bits() + 1),
                }
            };
            if index % 16 < 2 {
                row.logits.fill(0.0);
                row.logits[0] = -0.0;
            }
            row.parameters = row.parameters.with_top_logprobs(2).unwrap();
            row
        })
        .collect()
}

#[test]
fn sampling_top_one_matches_sorted_path() {
    gpu_test(512 * 1024 * 1024, |session| {
        for dtype in [DType::F32, DType::Bf16] {
            for width in [1025, 38_144, 152_576] {
                let fixture = Fixture::new(session, width, dtype, 32)?;
                let (module, _) = function(session, dtype)?;
                let (partials, finish) = simple_functions(&module, dtype, true)?;
                for start in (0..512).step_by(32) {
                    let mut rows = top_one_rows(width, dtype, start);
                    fixture.upload(session, &rows)?;
                    let mut sorted = fixture.replay(session, &rows, false)?;
                    sorted.eager(|| {
                        fixture.launch_simple(session, &partials, &finish, rows.len(), true)
                    })?;
                    let reference = fixture.output(session, &rows)?;
                    for row in &mut rows {
                        row.parameters = row.parameters.clone().with_top_logprobs(1).unwrap();
                    }
                    fixture.upload(session, &rows)?;
                    let mut checks = fixture.replay(session, &rows, false)?;
                    checks.eager(|| {
                        fixture.launch_simple(session, &partials, &finish, rows.len(), true)
                    })?;
                    let actual = fixture.output(session, &rows)?;
                    for (left, right) in actual.iter().zip(&reference) {
                        if left.token != right.token
                            || left.top[0].0 != right.top[0].0
                            || (left.logprob - right.logprob).abs() > LOGPROB_TOLERANCE
                            || (left.top[0].1 - right.top[0].1).abs() > LOGPROB_TOLERANCE
                        {
                            return Err(format!(
                                "top-one differs from sorted path: {left:?} vs {right:?}"
                            ));
                        }
                    }
                    compare(session, "top_one", dtype, &rows, &actual, start as usize)?;
                    let graph = session.capture(|_| {
                        fixture.launch_simple(session, &partials, &finish, rows.len(), true)
                    })?;
                    checks.graph(&graph)?;
                    check_exact("top-one graph", &actual, &fixture.output(session, &rows)?)?;
                }
            }
        }
        Ok(())
    })
    .expect("top-one preserves greedy and sampled choices");
}

/// Nine rows with two top logprobs and the largest logit at token 900.
fn planner_rows(width: usize, dtype: DType) -> Vec<Row> {
    (0..9)
        .map(|index| {
            let temperature = if index % 2 == 0 { 0.0 } else { 0.8 };
            let mut row = random_row(width, dtype, 0x900 + index, temperature);
            row.logits[900] = 20.0;
            row.parameters = row.parameters.with_top_logprobs(2).unwrap();
            row
        })
        .collect()
}

/// Nine live rows in a 16-row bucket take the planner's many-row geometry;
/// top logprobs still need the 256-thread tile sort, which must see the
/// largest logit at token 900 of the first tile.
#[test]
fn sampling_top_logprobs_planner_geometry() {
    gpu_test(512 * 1024 * 1024, |session| {
        for dtype in [DType::F32, DType::Bf16] {
            let width = 38_144;
            let fixture = Fixture::new(session, width, dtype, 16)?;
            let (module, _) = function(session, dtype)?;
            let (partials, finish) = simple_functions(&module, dtype, true)?;
            let rows = planner_rows(width, dtype);
            fixture.upload(session, &rows)?;
            fixture.launch_simple(session, &partials, &finish, rows.len(), true)?;
            let actual = fixture.output(session, &rows)?;
            compare(session, "top_planner_b9", dtype, &rows, &actual, 0)?;
        }
        Ok(())
    })
    .expect("top logprobs use the planner's geometry");
}

/// The 64 seeded random rows at `start` (fewer at the end) of 10,000.
fn seeded_rows(width: usize, dtype: DType, start: usize) -> Vec<Row> {
    (start..(start + 64).min(10_000))
        .map(|index| random_row(width, dtype, 0x7031_a62b_9940_0001_u64 ^ index as u64, 0.8))
        .collect()
}

#[test]
fn sampling_seeded_large_random_rows() {
    gpu_test(512 * 1024 * 1024, |session| {
        for dtype in [DType::F32, DType::Bf16] {
            for width in [38_144, 152_576] {
                let fixture = Fixture::new(session, width, dtype, 64)?;
                let (module, _kernel) = function(session, dtype)?;
                let (partials, finish) = simple_functions(&module, dtype, false)?;
                for start in (0..10_000).step_by(64) {
                    let rows = seeded_rows(width, dtype, start);
                    fixture.upload(session, &rows)?;
                    fixture.launch_simple(session, &partials, &finish, rows.len(), false)?;
                    let actual = fixture.output(session, &rows)?;
                    compare(
                        session,
                        &format!("seeded_w{width}"),
                        dtype,
                        &rows,
                        &actual,
                        start,
                    )?;
                }
            }
        }
        Ok(())
    })
    .expect("sampling 10,000 seeded random rows per width and dtype");
}

/// Eight rows with every penalty, bias, an EOS mask, every filter and 20
/// top logprobs; the first is greedy.
fn penalty_rows(width: usize, dtype: DType) -> Vec<Row> {
    let mut rows: Vec<_> = (0..8)
        .map(|index| random_row(width, dtype, 0x3094_4562 + index, 0.73))
        .collect();
    for (index, row) in rows.iter_mut().enumerate() {
        row.history = vec![3, 7, 3, 11, 7, 7];
        row.emitted = 3;
        row.eos = vec![13];
        row.parameters = SamplingParameters::new(if index == 0 { 0.0 } else { 0.73 })
            .unwrap()
            .with_penalties(0.5, 0.25, 1.15)
            .unwrap()
            .with_logit_bias(vec![
                LogitBias::new(3, 1.25).unwrap(),
                LogitBias::new(19, -0.75).unwrap(),
            ])
            .unwrap()
            .with_minimum_new_tokens(4)
            .unwrap()
            .with_top_k(64)
            .unwrap()
            .with_top_p(0.92)
            .unwrap()
            .with_min_p(0.01)
            .unwrap()
            .with_top_logprobs(20)
            .unwrap();
    }
    rows
}

#[test]
fn sampling_penalties_bias_and_top_logprobs() {
    gpu_test(512 * 1024 * 1024, |session| {
        for dtype in [DType::F32, DType::Bf16] {
            for width in [38_144, 152_576] {
                let rows = penalty_rows(width, dtype);
                run_once(session, &rows, dtype, &format!("filtered_w{width}"), false)?;
                let fixture = Fixture::new(session, width, dtype, rows.len())?;
                fixture.upload(session, &rows)?;
                let (module, _) = function(session, dtype)?;
                fixture.launch_filtered(
                    session,
                    &filtered_functions(&module, dtype)?,
                    rows.len(),
                )?;
                let actual = fixture.output(session, &rows)?;
                compare(
                    session,
                    &format!("fast_filtered_w{width}"),
                    dtype,
                    &rows,
                    &actual,
                    0,
                )?;
            }
        }
        Ok(())
    })
    .expect("sampling penalties, bias, filters, and raw top logprobs");
}

/// Eight sampled rows of width 38,144.
fn simple_graph_rows(dtype: DType) -> Vec<Row> {
    (0..8)
        .map(|index| random_row(38_144, dtype, 0x7614_29fe + index, 0.8))
        .collect()
}

#[test]
fn sampling_graph_replay_determinism_and_pdl() {
    gpu_test(512 * 1024 * 1024, |session| {
        let dtype = DType::F32;
        let rows = simple_graph_rows(dtype);
        let fixture = Fixture::new(session, 38_144, dtype, rows.len())?;
        fixture.upload(session, &rows)?;
        let (module, _kernel) = function(session, dtype)?;
        let (partials, finish) = simple_functions(&module, dtype, false)?;
        let launch = || fixture.launch_simple(session, &partials, &finish, rows.len(), false);
        let mut checks = fixture.replay(session, &rows, false)?;
        checks.eager(launch)?;
        let eager = fixture.output(session, &rows)?;
        compare(session, "graph_eager", dtype, &rows, &eager, 0)?;
        checks.relaunch(launch)?;
        check_exact("determinism", &eager, &fixture.output(session, &rows)?)?;
        let graph = session.capture(|_| launch())?;
        checks.graph(&graph)?;
        check_exact("graph", &eager, &fixture.output(session, &rows)?)?;
        let producer = PdlProducer::new(session)?;
        checks.pdl(|| producer.launch(session), launch, true)?;
        check_exact("pdl", &eager, &fixture.output(session, &rows)?)?;
        let mut receipt = BTreeMap::new();
        receipt.insert("case".into(), json!("graph_replay_determinism_pdl"));
        receipt.insert("rows".into(), json!(rows.len()));
        checks.record(&mut receipt);
        session.receipt("sampling", receipt)?;
        Ok(())
    })
    .expect("sampling graph replay, determinism, and PDL");
}

/// `chop_probe_pdl_first_v1`: releases its dependents at once, then spins.
struct PdlProducer {
    function: Function,
    stamps: DeviceBuffer,
}

impl PdlProducer {
    fn new(session: &Session) -> LabResult<Self> {
        let function = session
            .module("probe_sm120")?
            .function("chop_probe_pdl_first_v1")
            .map_err(|error| error.to_string())?;
        Ok(Self {
            function,
            stamps: session.allocate(24)?,
        })
    }
    fn launch(&self, session: &Session) -> LabResult<()> {
        self.function
            .launch(
                &session.stream,
                launch::simple_config((1, 1, 1), (32, 1, 1), 0, (1, 1, 1), false)?,
                &[
                    KernelArgument::Buffer(&self.stamps),
                    KernelArgument::U64(1000),
                ],
            )
            .map_err(|error| error.to_string())
    }
}

fn find_peak(ordinal: i32) -> (f64, String) {
    let directory = kernel_lab::receipt_root().join("probes");
    let latest = fs::read_dir(directory)
        .ok()
        .into_iter()
        .flatten()
        .filter_map(Result::ok)
        .filter_map(|entry| {
            let contents = fs::read(entry.path()).ok()?;
            let value = serde_json::from_slice::<Value>(&contents).ok()?;
            if value.get("ordinal")?.as_i64()? != i64::from(ordinal)
                || !value.pointer("/dram_read/passed")?.as_bool()?
            {
                return None;
            }
            let peak = value.pointer("/dram_read/gb_per_second")?.as_f64()?;
            Some((entry.file_name(), peak, entry.path().display().to_string()))
        })
        .max_by_key(|(name, _, _)| name.clone());
    if let Some((_, peak, path)) = latest {
        return (peak, path);
    }
    (1800.0, "fallback: 1.8 TB/s DRAM bandwidth".into())
}

#[test]
fn sampling_top_one_benchmarks() {
    if std::env::var("CHOPINFER_KERNEL_BENCH").as_deref() != Ok("1") {
        kernel_lab::skip("sampling_top_one_benchmarks: set CHOPINFER_KERNEL_BENCH=1").unwrap();
        return;
    }
    gpu_test(512 * 1024 * 1024, |session| {
        let baseline_path = std::env::var("CHOPINFER_SAMPLING_BASELINE_CUBIN")
            .map_err(|_| "set CHOPINFER_SAMPLING_BASELINE_CUBIN to the frozen baseline cubin")?;
        let baseline_bytes = fs::read(&baseline_path).map_err(|e| e.to_string())?;
        let baseline = session
            .context
            .load_module(&baseline_bytes)
            .map_err(|e| e.to_string())?;
        for dtype in [DType::F32, DType::Bf16] {
            let (module, _) = function(session, dtype)?;
            let (before_partial, before_finish) = simple_functions(&baseline, dtype, true)?;
            let (after_partial, after_finish) = simple_functions(&module, dtype, true)?;
            for batch in [1, 4, 16, 32] {
                for temperature in [0.0, 0.8] {
                    let greedy = greedy_functions(&module, dtype)?;
                    let after_partial = if temperature == 0.0 {
                        &greedy.0
                    } else {
                        &after_partial
                    };
                    let rows: Vec<_> = (0..batch)
                        .map(|index| {
                            let mut row =
                                random_row(152_576, dtype, 0xa134 + index as u64, temperature);
                            row.parameters = row.parameters.with_top_logprobs(1).unwrap();
                            row
                        })
                        .collect();
                    let fixture = Fixture::new(session, 152_576, dtype, batch)?;
                    fixture.upload(session, &rows)?;
                    fixture.launch_simple(session, &before_partial, &before_finish, batch, true)?;
                    let reference = fixture.output(session, &rows)?;
                    fixture.poison(session)?;
                    fixture.launch_simple(session, after_partial, &after_finish, batch, true)?;
                    let actual = fixture.output(session, &rows)?;
                    for (left, right) in actual.iter().zip(&reference) {
                        if left.token != right.token
                            || left.top[0].0 != right.top[0].0
                            || (left.logprob - right.logprob).abs() > LOGPROB_TOLERANCE
                            || (left.top[0].1 - right.top[0].1).abs() > LOGPROB_TOLERANCE
                        {
                            return Err("top-one differs from frozen baseline".into());
                        }
                    }
                    let measure = |partial: &Function, finish: &Function| -> LabResult<f32> {
                        let graph = session.capture(|_| {
                            for _ in 0..47 {
                                fixture.launch_simple(session, partial, finish, batch, true)?;
                            }
                            Ok(())
                        })?;
                        Ok(time(session, 5, 15, |stream| {
                            graph.launch(stream).map_err(|e| e.to_string())
                        })?
                        .median_ms
                            * 1000.0
                            / 47.0)
                    };
                    let before = measure(&before_partial, &before_finish)?;
                    let after = measure(after_partial, &after_finish)?;
                    session.receipt(
                        "sampling",
                        BTreeMap::from([
                            ("case".into(), json!("route_samp_top_one")),
                            ("dtype".into(), json!(format!("{dtype:?}"))),
                            ("rows".into(), json!(batch)),
                            ("temperature".into(), json!(temperature)),
                            ("before_us".into(), json!(before)),
                            ("after_us".into(), json!(after)),
                            ("baseline_cubin".into(), json!(baseline_path)),
                        ]),
                    )?;
                }
            }
        }
        Ok(())
    })
    .expect("top-one sampling benchmark and frozen-baseline parity");
}

#[test]
fn sampling_benchmarks() {
    if std::env::var("CHOPINFER_KERNEL_BENCH").as_deref() != Ok("1") {
        kernel_lab::skip("sampling benchmarks: set CHOPINFER_KERNEL_BENCH=1").unwrap();
        return;
    }
    gpu_test(512 * 1024 * 1024, |session| {
        let (peak_gbps, peak_source) = find_peak(session.ordinal);
        for dtype in [DType::F32, DType::Bf16] {
            for (profile, width, batches) in [
                (model_mimo26_contract::sm120::config::PROFILE, 38_144, &[1_usize, 8, 32, 64][..]),
                (model_mimo26_contract::sm120::config::PROFILE, 152_576, &[1, 8, 16, 32, 64][..]),
                (Profile::StdDenseMoe, 32_768, &[1, 32][..]),
            ] {
                let (module, _kernel) = function(session, dtype)?;
                let (partials, finish) = simple_functions(&module, dtype, false)?;
                for &batch in batches {
                    if std::env::var("CHOPINFER_KERNEL_BENCH_TARGET").as_deref() == Ok("1")
                        && (width != 152_576 || ![1, 16, 32].contains(&batch))
                    {
                        continue;
                    }
                    let rows: Vec<_> = (0..batch)
                        .map(|index| random_row(width, dtype, 0x8114_28c6 + index as u64, 0.8))
                        .collect();
                    let fixture = Fixture::new(session, width, dtype, batch)?;
                    fixture.upload(session, &rows)?;
                    fixture.launch_simple(session, &partials, &finish, batch, false)?;
                    compare(
                        session,
                        &format!("bench_correctness_w{width}_b{batch}"),
                        dtype,
                        &rows,
                        &fixture.output(session, &rows)?,
                        0,
                    )?;
                    let timing = time(session, 2, 3, |_| {
                        fixture.launch_simple(session, &partials, &finish, batch, false)
                    })?;
                    let planned = sampling::plan_simple(
                        SamplingQuery {
                            row_capacity: batch as u32,
                            width_capacity: width as u32,
                            dtype,
                        },
                        RuntimeCounts {
                            m: batch as u32,
                            batch: batch as u32,
                            context: 0,
                            routed_rows: 0,
                            split_k: 1,
                        },
                        width as u32,
                        width as u64,
                        SimpleSamplingBuffers {
                            logits: LOGITS,
                            metadata: META,
                            partials: PARTIALS,
                            output: OUTPUT,
                        },
                        false,
                    )
                    .map_err(|error| error.to_string())?;
                    let bindings = Bindings {
                        buffers: fixture
                            .buffers
                            .iter()
                            .map(|(&id, buffer)| (id, buffer))
                            .collect(),
                    };
                    let partial_timing = time(session, 2, 3, |_| {
                        launch::launch(
                            &session.context,
                            &partials,
                            &session.stream,
                            planned.partials,
                            &planned.partial_args,
                            &bindings,
                        )
                    })?;
                    let finish_timing = time(session, 2, 3, |_| {
                        launch::launch(
                            &session.context,
                            &finish,
                            &session.stream,
                            planned.finish,
                            &planned.finish_args,
                            &bindings,
                        )
                    })?;
                    let element_bytes = if dtype == DType::Bf16 { 2 } else { 4 };
                    let logical_bytes = (batch * width * element_bytes) as u64;
                    let achieved_gbps =
                        logical_bytes as f64 / f64::from(timing.median_ms) / 1_000_000.0;
                    let mut receipt = BTreeMap::new();
                    receipt.insert("case".into(), json!(format!("w{width}_b{batch}_{dtype:?}")));
                    receipt.insert("profile".into(), json!(profile.as_str()));
                    receipt.insert("median_us".into(), json!(timing.median_ms * 1000.0));
                    receipt.insert(
                        "partials_us".into(),
                        json!(partial_timing.median_ms * 1000.0),
                    );
                    receipt.insert("finish_us".into(), json!(finish_timing.median_ms * 1000.0));
                    receipt.insert("min_us".into(), json!(timing.min_ms * 1000.0));
                    receipt.insert("rounds".into(), json!(timing.rounds));
                    receipt.insert("logical_bytes".into(), json!(logical_bytes));
                    receipt.insert("achieved_gb_per_second".into(), json!(achieved_gbps));
                    receipt.insert("peak_gb_per_second".into(), json!(peak_gbps));
                    receipt.insert("peak_source".into(), json!(peak_source));
                    receipt.insert(
                        "percent_peak".into(),
                        json!(100.0 * achieved_gbps / peak_gbps),
                    );
                    session.receipt("bench", receipt)?;
                    if width == 152_576 && [1, 16, 32].contains(&batch) {
                        let mut greedy = rows.clone();
                        for row in &mut greedy {
                            row.parameters = SamplingParameters::new(0.0).unwrap();
                        }
                        fixture.upload(session, &greedy)?;
                        let (greedy_partials, greedy_finish_function) =
                            greedy_functions(&module, dtype)?;
                        let greedy_plan = sampling::plan_simple(
                            SamplingQuery {
                                row_capacity: batch as u32,
                                width_capacity: width as u32,
                                dtype,
                            },
                            RuntimeCounts {
                                m: batch as u32,
                                batch: batch as u32,
                                context: 0,
                                routed_rows: 0,
                                split_k: 1,
                            },
                            width as u32,
                            width as u64,
                            SimpleSamplingBuffers {
                                logits: LOGITS,
                                metadata: META,
                                partials: PARTIALS,
                                output: OUTPUT,
                            },
                            false,
                        )
                        .map_err(|error| error.to_string())?;
                        fixture.launch_simple(
                            session,
                            &greedy_partials,
                            &greedy_finish_function,
                            batch,
                            false,
                        )?;
                        compare(
                            session,
                            &format!("bench_greedy_w{width}_b{batch}"),
                            dtype,
                            &greedy,
                            &fixture.output(session, &greedy)?,
                            0,
                        )?;
                        let greedy_timing = time(session, 2, 3, |_| {
                            fixture.launch_simple(
                                session,
                                &greedy_partials,
                                &greedy_finish_function,
                                batch,
                                false,
                            )
                        })?;
                        let greedy_graph = session.capture(|_| {
                            fixture.launch_simple(
                                session,
                                &greedy_partials,
                                &greedy_finish_function,
                                batch,
                                false,
                            )
                        })?;
                        let greedy_graph_timing = time(session, 2, 3, |stream| {
                            greedy_graph
                                .launch(stream)
                                .map_err(|error| error.to_string())
                        })?;
                        let greedy_partial = time(session, 2, 3, |_| {
                            launch::launch(
                                &session.context,
                                &greedy_partials,
                                &session.stream,
                                greedy_plan.partials,
                                &greedy_plan.partial_args,
                                &bindings,
                            )
                        })?;
                        let greedy_finish = time(session, 2, 3, |_| {
                            launch::launch(
                                &session.context,
                                &greedy_finish_function,
                                &session.stream,
                                greedy_plan.finish,
                                &greedy_plan.finish_args,
                                &bindings,
                            )
                        })?;
                        let mut greedy_receipt = BTreeMap::new();
                        greedy_receipt.insert(
                            "case".into(),
                            json!(format!("w{width}_b{batch}_{dtype:?}_greedy")),
                        );
                        greedy_receipt
                            .insert("median_us".into(), json!(greedy_timing.median_ms * 1000.0));
                        greedy_receipt.insert(
                            "graph_median_us".into(),
                            json!(greedy_graph_timing.median_ms * 1000.0),
                        );
                        greedy_receipt.insert(
                            "partials_us".into(),
                            json!(greedy_partial.median_ms * 1000.0),
                        );
                        greedy_receipt
                            .insert("finish_us".into(), json!(greedy_finish.median_ms * 1000.0));
                        session.receipt("bench", greedy_receipt)?;
                        for row in &mut greedy {
                            row.parameters = SamplingParameters::new(0.0)
                                .unwrap()
                                .with_top_logprobs(20)
                                .unwrap();
                        }
                        let (top_partials, _) = simple_functions(&module, dtype, true)?;
                        fixture.upload(session, &greedy)?;
                        fixture.launch_simple(session, &top_partials, &finish, batch, true)?;
                        compare(
                            session,
                            &format!("bench_greedy_top20_w{width}_b{batch}"),
                            dtype,
                            &greedy,
                            &fixture.output(session, &greedy)?,
                            0,
                        )?;
                        let top_timing = time(session, 2, 3, |_| {
                            fixture.launch_simple(session, &top_partials, &finish, batch, true)
                        })?;
                        let mut top_receipt = BTreeMap::new();
                        top_receipt.insert(
                            "case".into(),
                            json!(format!("w{width}_b{batch}_{dtype:?}_greedy_top20")),
                        );
                        top_receipt
                            .insert("median_us".into(), json!(top_timing.median_ms * 1000.0));
                        session.receipt("bench", top_receipt)?;
                    }
                }
            }
        }
        Ok(())
    })
    .expect("sampling benchmarks");
}

/// Uniform standard normal from two xorshift words.
fn normal(state: &mut u64) -> f64 {
    let first = (f64::from(random_word(state)) + 0.5) / 4_294_967_296.0;
    let second = f64::from(random_word(state)) / 4_294_967_296.0;
    (-2.0 * first.ln()).sqrt() * (std::f64::consts::TAU * second).cos()
}

/// Language-model-shaped logits: a normal bulk below a head whose size and
/// decay vary per row, so nucleus sizes range from one token to thousands.
/// Returns the row and its head token IDs, best first.
fn lm_row(width: usize, dtype: DType, seed: u64) -> (Vec<f32>, Vec<u32>) {
    let mut rng = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    for _ in 0..4 {
        random_word(&mut rng);
    }
    let spread = [1.2, 2.0, 2.8, 3.5][random_word(&mut rng) as usize % 4];
    let mut logits: Vec<f32> = (0..width)
        .map(|_| (-4.0 + spread * normal(&mut rng)) as f32)
        .collect();
    let head_len = 1 + random_word(&mut rng) as usize % 96;
    let top = 6.0 + f64::from(random_word(&mut rng) % 1600) / 100.0;
    let decay = [0.3, 0.8, 1.5, 3.0][random_word(&mut rng) as usize % 4];
    let mut head = Vec::with_capacity(head_len);
    for rank in 0..head_len {
        let token = random_word(&mut rng) as usize % width;
        logits[token] = (top - decay * (1.0 + rank as f64).ln()) as f32;
        head.push(token as u32);
    }
    if dtype == DType::Bf16 {
        for value in &mut logits {
            *value = bf16_to_f32(f32_to_bf16(*value));
        }
    }
    (logits, head)
}

/// Deterministic parameter mix for row `index`: production defaults most
/// often, plus top-k, min-p, penalties, bias, EOS masking and top logprobs.
fn filtered_row(width: usize, dtype: DType, index: u64) -> Row {
    let mut rng = 0x6a09_e667_f3bc_c908_u64 ^ index.wrapping_mul(0x9e37_79b9_7f4a_7c15);
    let mut pick = |values: &[f64]| values[random_word(&mut rng) as usize % values.len()];
    let temperature = pick(&[1.0, 1.0, 1.0, 0.8, 0.6, 1.3, 0.0]);
    let top_p = pick(&[0.95, 0.95, 0.95, 0.9, 0.5, 0.99, 0.7, 1.0]);
    let top_k = pick(&[0.0, 0.0, 0.0, 0.0, 1.0, 20.0, 50.0, 1000.0, 5000.0]) as usize;
    let min_p = pick(&[0.0, 0.0, 0.0, 0.01, 0.05, 0.2]);
    let top_logprobs = pick(&[0.0, 0.0, 1.0, 5.0, 20.0]) as usize;
    let extras = random_word(&mut rng);
    let (logits, head) = if extras % 5 == 0 {
        let row = random_row(width, dtype, index ^ 0x51ed, 1.0);
        let head = (0..8).map(|i| (i * 131 % width) as u32).collect();
        (row.logits, head)
    } else {
        lm_row(width, dtype, index ^ 0x2545_f491_4f6c_dd1d)
    };
    let mut parameters = SamplingParameters::new(temperature)
        .unwrap()
        .with_top_p(top_p)
        .unwrap()
        .with_top_k(top_k)
        .unwrap()
        .with_min_p(min_p)
        .unwrap()
        .with_top_logprobs(top_logprobs)
        .unwrap();
    let mut history = Vec::new();
    let mut emitted = 0;
    let mut eos = Vec::new();
    if extras & 0x10 != 0 {
        let penalties = [(0.5, 0.25, 1.15), (0.0, 0.0, 1.3), (-0.5, 0.4, 0.9)];
        let (presence, frequency, repetition) = penalties[(extras >> 8) as usize % 3];
        parameters = parameters
            .with_penalties(presence, frequency, repetition)
            .unwrap();
        for i in 0..48 {
            let word = random_word(&mut rng) as usize;
            history.push(if i % 3 == 0 {
                (word % width) as u32
            } else {
                head[word % head.len()]
            });
        }
        emitted = 20;
    }
    if extras & 0x20 != 0 {
        let mut biases = vec![
            LogitBias::new(head[0], -2.5).unwrap(),
            LogitBias::new(((extras >> 12) as usize % width) as u32, 30.0).unwrap(),
        ];
        if head.len() > 1 && head[1] != head[0] {
            biases.push(LogitBias::new(head[1], 1.75).unwrap());
        }
        biases.sort_by_key(|bias| bias.token_id());
        biases.dedup_by_key(|bias| bias.token_id());
        parameters = parameters.with_logit_bias(biases).unwrap();
    }
    if extras & 0x40 != 0 {
        eos = vec![head[head.len() - 1], 7 % width as u32];
        parameters = parameters.with_minimum_new_tokens(emitted + 2).unwrap();
    }
    // Constraint masks: sparse with the head, dense, or stop tokens only.
    let words = width.div_ceil(32);
    let admit = |tokens: &mut Vec<u32>, token: u32| {
        tokens[token as usize / 32] |= 1 << (token % 32);
    };
    let mask = match (extras >> 7) & 7 {
        0 => {
            let mut mask = vec![0_u32; words];
            for _ in 0..width / 100 + 1 {
                let token = random_word(&mut rng) % width as u32;
                admit(&mut mask, token);
            }
            for &token in head.iter().take(3) {
                admit(&mut mask, token);
            }
            Some(mask)
        }
        1 => {
            let mut mask: Vec<u32> = (0..words).map(|_| random_word(&mut rng)).collect();
            admit(&mut mask, head[0]);
            Some(mask)
        }
        2 if !eos.is_empty() => {
            let mut mask = vec![0_u32; words];
            for &token in &eos {
                admit(&mut mask, token);
            }
            Some(mask)
        }
        _ => None,
    };
    let mask = mask.map(|mut mask| {
        if width % 32 != 0 {
            mask[words - 1] &= (1_u32 << (width % 32)) - 1;
        }
        mask
    });
    Row {
        logits,
        parameters,
        state: SamplingState::new(index.wrapping_mul(0xd1b5_4a32_d192_ed03)).after(index % 97),
        history,
        emitted,
        eos,
        mask,
    }
}

fn run_filtered_batches(
    session: &Session,
    label: &str,
    dtype: DType,
    rows: &[Row],
    batch: usize,
) -> LabResult<(f64, usize)> {
    let width = rows[0].logits.len();
    let fixture = Fixture::new(session, width, dtype, batch)?;
    let (module, _) = function(session, dtype)?;
    let functions = filtered_functions(&module, dtype)?;
    let mut worst = 0.0_f64;
    let mut fallbacks = 0;
    for (chunk, part) in rows.chunks(batch).enumerate() {
        fixture.upload(session, part)?;
        fixture.launch_filtered(session, &functions, part.len())?;
        let actual = fixture.output(session, part)?;
        let expected = expected_rows(part)?;
        worst = worst.max(check_tokens(label, &expected, &actual, chunk * batch)?);
        let flags = fixture.exact_fallbacks(session, part.len())?;
        fallbacks += flags.into_iter().filter(|&used| used).count();
    }
    Ok((worst, fallbacks))
}

/// The randomized filtered sweep's widths and row counts.
const FILTERED_SWEEP: [(usize, u64); 3] = [(152_576, 2048), (38_144, 1024), (1025, 1024)];

fn filtered_sweep_rows(width: usize, dtype: DType, count: u64) -> Vec<Row> {
    (0..count)
        .map(|index| filtered_row(width, dtype, index + (width as u64) * 7))
        .collect()
}

#[test]
fn sampling_filtered_matches_cpu() {
    gpu_test(2048 * 1024 * 1024, |session| {
        for dtype in [DType::F32, DType::Bf16] {
            for (width, count) in FILTERED_SWEEP {
                let rows = filtered_sweep_rows(width, dtype, count);
                let label = format!("filtered_sweep_w{width}");
                let (worst, fallbacks) = run_filtered_batches(session, &label, dtype, &rows, 64)?;
                eprintln!(
                    "{label} {dtype:?}: {count} rows, worst logprob error {worst:e}, {fallbacks} exact fallbacks"
                );
                session.receipt(
                    "sampling",
                    BTreeMap::from([
                        ("case".into(), json!(label)),
                        ("dtype".into(), json!(format!("{dtype:?}"))),
                        ("rows".into(), json!(count)),
                        ("tokens_exact".into(), json!(true)),
                        ("worst_logprob_error".into(), json!(worst)),
                        ("logprob_tolerance".into(), json!(LOGPROB_TOLERANCE)),
                        ("exact_fallbacks".into(), json!(fallbacks)),
                        ("correct".into(), json!(true)),
                    ]),
                )?;
            }
        }
        Ok(())
    })
    .expect("filtered sampling matches the CPU oracle");
}

/// Temperatures so small that the F32 estimate `key / T` overflows (1/T
/// itself at 1e-40): all-negative, all-zero and language-model rows, masks
/// admitting two tokens or sparse tokens in a few tiles, and no-filter,
/// top-k, top-p and min-p settings.
fn tiny_temperature_rows(width: usize, dtype: DType) -> Vec<Row> {
    let mut rows = Vec::new();
    for (index, temperature) in [1e-40, 1e-37, 1e-12].into_iter().enumerate() {
        for pattern in 0..3 {
            let logits = match pattern {
                0 => (0..width)
                    .map(|t| if t == 0 { -1.0 } else { -2.0 })
                    .collect(),
                1 => vec![0.0; width],
                _ => lm_row(width, dtype, 0x7e37 + index as u64).0,
            };
            for masked in 0..3 {
                let words = width.div_ceil(32);
                let mask = match masked {
                    0 => None,
                    1 => Some((0..words).map(|w| u32::from(w == 0) * 3).collect()),
                    _ => Some(
                        (0..words)
                            .map(|w| if w % 1000 == 7 { 0x8001_0010 } else { 0 })
                            .collect(),
                    ),
                };
                for filter in 0..4 {
                    let parameters = SamplingParameters::new(temperature).unwrap();
                    let parameters = match filter {
                        0 => parameters,
                        1 => parameters.with_top_k(20).unwrap(),
                        2 => parameters.with_top_p(0.9).unwrap(),
                        _ => parameters.with_min_p(0.1).unwrap(),
                    };
                    let seed = rows.len() as u64;
                    rows.push(Row {
                        logits: logits.clone(),
                        parameters,
                        state: SamplingState::new(0x71a7 ^ seed),
                        history: Vec::new(),
                        emitted: 0,
                        eos: Vec::new(),
                        mask: mask.clone(),
                    });
                }
            }
        }
    }
    rows
}

/// At tiny temperatures (`tiny_temperature_rows`) an overflowed estimate
/// must send an admitted token to the exact F64 pass rather than read as a
/// rejection.
#[test]
fn sampling_filtered_tiny_temperature() {
    gpu_test(2048 * 1024 * 1024, |session| {
        let width = 152_576;
        for dtype in [DType::F32, DType::Bf16] {
            let rows = tiny_temperature_rows(width, dtype);
            let label = format!("filtered_tiny_temperature_{dtype:?}");
            let (worst, _) = run_filtered_batches(session, &label, dtype, &rows, 64)?;
            session.receipt(
                "sampling",
                BTreeMap::from([
                    ("case".into(), json!(label)),
                    ("rows".into(), json!(rows.len())),
                    ("tokens_exact".into(), json!(true)),
                    ("worst_logprob_error".into(), json!(worst)),
                    ("correct".into(), json!(true)),
                ]),
            )?;
        }
        Ok(())
    })
    .expect("filtered sampling at tiny temperatures");
}

/// Four BF16 rows of MiMo V2.6 logits at width 152,576 from the numerics
/// audit (fixtures/mimo26/sampling/README.md).
fn fixture_logits() -> LabResult<Vec<Vec<f32>>> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures/mimo26/sampling/real-logits-4x152576.bf16");
    let bytes = fs::read(&path).map_err(|error| format!("{}: {error}", path.display()))?;
    if bytes.len() != 4 * 152_576 * 2 {
        return Err(format!(
            "{} is not four 152,576-wide BF16 rows",
            path.display()
        ));
    }
    Ok(bytes
        .chunks_exact(152_576 * 2)
        .map(|row| {
            row.chunks_exact(2)
                .map(|code| bf16_to_f32(u16::from_le_bytes([code[0], code[1]])))
                .collect()
        })
        .collect())
}

/// Production defaults with many seeds, then the mixed parameters of the
/// synthetic sweep, on the fixture's real rows.
fn real_fixture_rows(real: &[Vec<f32>], dtype: DType) -> Vec<Row> {
    (0..768_u64)
        .map(|index| {
            let mut row = filtered_row(152_576, dtype, index + 0x5eed);
            row.logits = real[index as usize % real.len()].clone();
            if index < 256 {
                row.parameters = SamplingParameters::new(1.0)
                    .unwrap()
                    .with_top_p(0.95)
                    .unwrap()
                    .with_top_logprobs(1)
                    .unwrap();
                row.state = SamplingState::new(index);
                row.history.clear();
                row.emitted = 0;
                row.eos.clear();
                row.mask = None;
            }
            row
        })
        .collect()
}

#[test]
fn sampling_filtered_real_rows_match_cpu() {
    gpu_test(2048 * 1024 * 1024, |session| {
        let real = fixture_logits()?;
        for dtype in [DType::F32, DType::Bf16] {
            let rows = real_fixture_rows(&real, dtype);
            let (worst, fallbacks) =
                run_filtered_batches(session, "filtered_real_rows", dtype, &rows, 64)?;
            eprintln!(
                "filtered_real_rows {dtype:?}: {} rows, worst logprob error {worst:e}, {fallbacks} exact fallbacks",
                rows.len()
            );
            session.receipt(
                "sampling",
                BTreeMap::from([
                    ("case".into(), json!("filtered_real_rows")),
                    ("dtype".into(), json!(format!("{dtype:?}"))),
                    ("rows".into(), json!(rows.len())),
                    ("tokens_exact".into(), json!(true)),
                    ("worst_logprob_error".into(), json!(worst)),
                    ("logprob_tolerance".into(), json!(LOGPROB_TOLERANCE)),
                    ("exact_fallbacks".into(), json!(fallbacks)),
                    ("correct".into(), json!(true)),
                ]),
            )?;
        }
        Ok(())
    })
    .expect("filtered sampling matches the CPU oracle on real rows");
}

/// A row whose live logits are `values` followed by tokens that contribute
/// exactly zero probability mass.
fn tie_row(width: usize, values: &[f32], parameters: SamplingParameters, seed: u64) -> Row {
    let mut logits = vec![-1000.0_f32; width];
    for (index, &value) in values.iter().enumerate() {
        logits[index * 97 % width] = value;
    }
    Row {
        logits,
        parameters,
        state: SamplingState::new(seed),
        history: Vec::new(),
        emitted: 0,
        eos: Vec::new(),
        mask: None,
    }
}

fn next_after(value: f64, up: bool) -> f64 {
    let bits = value.to_bits();
    f64::from_bits(if up { bits + 1 } else { bits - 1 })
}

/// The CPU oracle's sequential top-p prefix and total for sorted terms.
fn oracle_prefix(logits: &[f32], temperature: f64, keep: usize) -> (f64, f64) {
    let mut values: Vec<f64> = logits
        .iter()
        .map(|&value| f64::from(value) / temperature)
        .collect();
    values.sort_by(|left, right| right.total_cmp(left));
    let terms: Vec<f64> = values
        .iter()
        .map(|value| (value - values[0]).exp())
        .collect();
    (terms[..keep].iter().sum(), terms.iter().sum())
}

/// `sampling_filtered_boundary_ties`'s rows and whether each must take the
/// exact fallback: exact and adjacent top-p prefixes, top-p at the oracle's
/// own prefix ratio, wide windows and top-k cutoffs inside equal runs.
fn boundary_tie_rows(width: usize, dtype: DType) -> (Vec<Row>, Vec<bool>) {
    let mut rows = Vec::new();
    let mut expect_fallback = Vec::new();
    // Exactly representable prefixes: P_j equals top_p * total.
    for count in [2_usize, 3, 4, 8, 10] {
        for keep in 1..count {
            let top_p = keep as f64 / count as f64;
            for (p, exact_tie) in [
                (top_p, (top_p * count as f64) == keep as f64),
                (next_after(top_p, true), false),
                (next_after(top_p, false), false),
            ] {
                for seed in 0..8 {
                    let parameters = SamplingParameters::new(1.0).unwrap().with_top_p(p).unwrap();
                    rows.push(tie_row(width, &vec![5.0; count], parameters, seed));
                    expect_fallback.push(exact_tie);
                }
            }
        }
    }
    // Top-p set to the oracle's own prefix ratio on LM-shaped rows.
    for index in 0..256_u64 {
        let (logits, _) = lm_row(width, dtype, 0x7e57 + index);
        let temperature = [1.0, 0.8][index as usize % 2];
        let keep = 1 + index as usize % 40;
        let (prefix, total) = oracle_prefix(&logits, temperature, keep);
        let ratio = (prefix / total).min(next_after(1.0, false));
        let top_p = match index % 3 {
            0 => ratio,
            1 => next_after(ratio, true).min(1.0),
            _ => next_after(ratio, false),
        };
        rows.push(Row {
            logits,
            parameters: SamplingParameters::new(temperature)
                .unwrap()
                .with_top_p(top_p)
                .unwrap()
                .with_top_k([0, 30, 5000][index as usize % 3])
                .unwrap(),
            state: SamplingState::new(index),
            history: Vec::new(),
            emitted: 0,
            eos: Vec::new(),
            mask: None,
        });
        expect_fallback.push(false);
    }
    // Wide windows: equal or coarsely quantized logits put thousands of
    // candidates in one bin and exercise the global-memory sort.
    for index in 0..32_u64 {
        let mut rng = 0xbead_u64 + index;
        let logits: Vec<f32> = (0..width)
            .map(|_| {
                if index % 2 == 0 {
                    0.0
                } else {
                    -0.5 * (random_word(&mut rng) % 12) as f32
                }
            })
            .collect();
        let parameters = SamplingParameters::new(1.0)
            .unwrap()
            .with_top_p([0.3, 0.95, 0.999, 0.6][index as usize % 4])
            .unwrap()
            .with_top_k([0, 100_000, 3000, 0][index as usize / 4 % 4])
            .unwrap();
        rows.push(Row {
            logits,
            parameters,
            state: SamplingState::new(index * 31),
            history: Vec::new(),
            emitted: 0,
            eos: Vec::new(),
            mask: None,
        });
        expect_fallback.push(false);
    }
    // Top-k cutoffs inside a run of equal values keep the lowest IDs.
    for index in 0..16_u64 {
        let mut values = vec![3.0_f32; 40];
        values[..5].fill(4.0);
        let parameters = SamplingParameters::new(1.0)
            .unwrap()
            .with_top_k(6 + index as usize)
            .unwrap()
            .with_top_p(if index % 2 == 0 { 1.0 } else { 0.97 })
            .unwrap();
        rows.push(tie_row(width, &values, parameters, index));
        expect_fallback.push(false);
    }
    (rows, expect_fallback)
}

#[test]
fn sampling_filtered_boundary_ties() {
    gpu_test(2048 * 1024 * 1024, |session| {
        for dtype in [DType::F32, DType::Bf16] {
            let width = 152_576;
            let (rows, expect_fallback) = boundary_tie_rows(width, dtype);
            let fixture = Fixture::new(session, width, dtype, 64)?;
            let (module, _) = function(session, dtype)?;
            let functions = filtered_functions(&module, dtype)?;
            let mut worst = 0.0_f64;
            let mut fallbacks = 0;
            for (chunk, part) in rows.chunks(64).enumerate() {
                let offset = chunk * 64;
                fixture.upload(session, part)?;
                fixture.launch_filtered(session, &functions, part.len())?;
                let actual = fixture.output(session, part)?;
                let expected = expected_rows(part)?;
                worst = worst.max(check_tokens("boundary_ties", &expected, &actual, offset)?);
                for (local, used) in fixture.exact_fallbacks(session, part.len())?.into_iter().enumerate() {
                    fallbacks += usize::from(used);
                    if expect_fallback[offset + local] && !used {
                        return Err(format!(
                            "boundary_ties row {}: an exact tie was certified",
                            offset + local
                        ));
                    }
                }
            }
            eprintln!(
                "boundary ties {dtype:?}: {} rows, worst logprob error {worst:e}, {fallbacks} exact fallbacks",
                rows.len()
            );
            session.receipt(
                "sampling",
                BTreeMap::from([
                    ("case".into(), json!("filtered_boundary_ties")),
                    ("dtype".into(), json!(format!("{dtype:?}"))),
                    ("rows".into(), json!(rows.len())),
                    ("tokens_exact".into(), json!(true)),
                    ("worst_logprob_error".into(), json!(worst)),
                    ("exact_fallbacks".into(), json!(fallbacks)),
                    ("correct".into(), json!(true)),
                ]),
            )?;
        }
        Ok(())
    })
    .expect("filtered sampling boundary ties match the CPU oracle");
}

/// Fourteen mixed filtered rows, one exact-fallback row (row 14) and one
/// wide-window row.
fn filtered_graph_rows(width: usize, dtype: DType) -> Vec<Row> {
    let mut rows: Vec<_> = (0..14)
        .map(|index| filtered_row(width, dtype, 900 + index))
        .collect();
    rows.push(tie_row(
        width,
        &[5.0; 4],
        SamplingParameters::new(1.0)
            .unwrap()
            .with_top_p(0.5)
            .unwrap(),
        3,
    ));
    rows.push(Row {
        logits: vec![0.0; width],
        parameters: SamplingParameters::new(1.0)
            .unwrap()
            .with_top_p(0.3)
            .unwrap(),
        state: SamplingState::new(5),
        history: Vec::new(),
        emitted: 0,
        eos: Vec::new(),
        mask: None,
    });
    rows
}

#[test]
fn sampling_filtered_graph_replay_determinism_and_pdl() {
    gpu_test(1024 * 1024 * 1024, |session| {
        let dtype = DType::Bf16;
        let width = 152_576;
        let rows = filtered_graph_rows(width, dtype);
        let fixture = Fixture::new(session, width, dtype, rows.len())?;
        fixture.upload(session, &rows)?;
        let (module, _) = function(session, dtype)?;
        let functions = filtered_functions(&module, dtype)?;
        let launch = || fixture.launch_filtered(session, &functions, rows.len());
        let mut checks = fixture.replay(session, &rows, true)?;
        checks.eager(launch)?;
        let eager = fixture.output(session, &rows)?;
        check_tokens("filtered_graph_eager", &expected_rows(&rows)?, &eager, 0)?;
        if !fixture.exact_fallbacks(session, rows.len())?[14] {
            return Err("the exact tie row did not use the fallback".into());
        }
        checks.relaunch(launch)?;
        check_exact(
            "filtered determinism",
            &eager,
            &fixture.output(session, &rows)?,
        )?;
        // The second launch writes its own records and scratch, so each
        // launch's records are checked and neither reads the other's scratch.
        let second = fixture.second(session)?;
        let graph = session.capture(|_| {
            launch()?;
            fixture.launch_filtered_rebound(session, &functions, rows.len(), &rebind(&second), true)
        })?;
        for _ in 0..3 {
            graph_pair(session, &mut checks, &graph, &second, &rows)?;
            check_exact("filtered graph", &eager, &fixture.output(session, &rows)?)?;
        }
        let producer = PdlProducer::new(session)?;
        checks.pdl(|| producer.launch(session), launch, true)?;
        check_exact("filtered pdl", &eager, &fixture.output(session, &rows)?)?;
        let mut receipt = BTreeMap::from([
            (
                "case".into(),
                json!("filtered_graph_replay_determinism_pdl"),
            ),
            ("rows".into(), json!(rows.len())),
        ]);
        checks.record(&mut receipt);
        session.receipt("sampling", receipt)?;
        Ok(())
    })
    .expect("filtered sampling graph replay, determinism, and PDL");
}

fn rebind(buffers: &BTreeMap<u64, DeviceBuffer>) -> Vec<(u64, &DeviceBuffer)> {
    buffers.iter().map(|(&id, buffer)| (id, buffer)).collect()
}

/// Replays a graph of two filtered launches, the first on the fixture's
/// buffers and the second on `second` (its records at `OUTPUT`), all
/// poisoned first; the second launch's records must pass the same write map
/// and equal the eager bytes.
fn graph_pair(
    session: &Session,
    checks: &mut Replay<'_>,
    graph: &runtime_cuda::GraphExec,
    second: &BTreeMap<u64, DeviceBuffer>,
    rows: &[Row],
) -> LabResult<()> {
    poison::poison(session, &second.values().collect::<Vec<_>>())?;
    checks.graph(graph)?;
    let bytes = poison::read(session, &second[&OUTPUT])?;
    poison::check(
        "graph second launch records",
        &bytes,
        &record_map(rows)?,
        poison::Fill::Pattern(&[poison::POISON]),
    )?;
    let eager = checks.expected("records").ok_or("no eager records")?;
    match eager.iter().zip(&bytes).position(|(a, b)| a != b) {
        Some(byte) => Err(format!(
            "graph second launch records differ from eager at byte {byte}"
        )),
        None => Ok(()),
    }
}

/// Second launches that miss writes pass the checks `graph_pair` replaced
/// and fail it: one writing its records to a decoy passes a comparison of
/// the shared records after the pair, and one running only `finish` over the
/// first launch's scratch passes a pair check that shares that scratch.
#[test]
fn sampling_filtered_graph_pair_negative_controls() {
    gpu_test(1024 * 1024 * 1024, |session| {
        let dtype = DType::Bf16;
        let width = 152_576;
        let rows = filtered_graph_rows(width, dtype)[..4].to_vec();
        let fixture = Fixture::new(session, width, dtype, rows.len())?;
        fixture.upload(session, &rows)?;
        let (module, _) = function(session, dtype)?;
        let functions = filtered_functions(&module, dtype)?;
        let launch = || fixture.launch_filtered(session, &functions, rows.len());
        let mut checks = fixture.replay(session, &rows, true)?;
        checks.eager(launch)?;
        let eager = fixture.output(session, &rows)?;
        let second = fixture.second(session)?;
        let mut decoy = fixture.second(session)?;
        let decoy_records = decoy
            .remove(&OUTPUT)
            .ok_or("second invocation without records")?;
        let mut elsewhere_buffers = rebind(&decoy);
        elsewhere_buffers.push((OUTPUT, &decoy_records));
        let elsewhere = session.capture(|_| {
            launch()?;
            fixture.launch_filtered_rebound(session, &functions, rows.len(), &elsewhere_buffers, true)
        })?;
        // The first replaced check: poison before the pair, compare the
        // shared records.
        fixture.poison(session)?;
        elsewhere
            .launch(&session.stream)
            .map_err(|error| error.to_string())?;
        check_exact("pair records", &eager, &fixture.output(session, &rows)?)
            .map_err(|error| format!("the replaced record check already fails: {error}"))?;
        let decoy_error = match graph_pair(session, &mut checks, &elsewhere, &second, &rows) {
            Ok(()) => return Err("the graph pair check passed a decoy second launch".into()),
            Err(error) => error,
        };
        eprintln!("decoy second launch failed the pair check as required: {decoy_error}");
        // The second replaced check: the second launch's own records, but
        // the first launch's scratch.
        let shared = BTreeMap::from([(
            OUTPUT,
            session.allocate(fixture.buffers[&OUTPUT].byte_len())?,
        )]);
        let finish_only = |buffers: &BTreeMap<u64, DeviceBuffer>| {
            session.capture(|_| {
                launch()?;
                fixture.launch_filtered_rebound(session, &functions, rows.len(), &rebind(buffers), false)
            })
        };
        graph_pair(session, &mut checks, &finish_only(&shared)?, &shared, &rows)
            .map_err(|error| format!("the shared-scratch pair check already fails: {error}"))?;
        let stale_error = match graph_pair(session, &mut checks, &finish_only(&second)?, &second, &rows)
        {
            Ok(()) => {
                return Err("the graph pair check passed a second launch without its stages".into());
            }
            Err(error) => error,
        };
        eprintln!("finish-only second launch failed the pair check as required: {stale_error}");
        let receipt = BTreeMap::from([
            ("case".into(), json!("filtered_graph_pair_negative_controls")),
            ("rows".into(), json!(rows.len())),
            ("decoy_error".into(), json!(decoy_error)),
            ("finish_only_error".into(), json!(stale_error)),
            ("replaced_checks_passed".into(), json!(true)),
            ("pair_checks_failed".into(), json!(true)),
        ]);
        session.receipt("sampling_negative_control", receipt)?;
        Ok(())
    })
    .expect("filtered sampling graph pair negative controls");
}

/// Production-default rows: temperature 1, top-p 0.95, LM-shaped logits.
fn default_rows(batch: usize, dtype: DType, temperature: f64) -> Vec<Row> {
    (0..batch)
        .map(|index| {
            let (logits, _) = lm_row(152_576, dtype, 0xbe9c + index as u64);
            Row {
                logits,
                parameters: SamplingParameters::new(temperature)
                    .unwrap()
                    .with_top_p(0.95)
                    .unwrap(),
                state: SamplingState::new(index as u64),
                history: Vec::new(),
                emitted: 0,
                eos: Vec::new(),
                mask: None,
            }
        })
        .collect()
}

#[test]
fn sampling_filtered_benchmarks() {
    if std::env::var("CHOPINFER_KERNEL_BENCH").as_deref() != Ok("1") {
        kernel_lab::skip("sampling_filtered_benchmarks: set CHOPINFER_KERNEL_BENCH=1").unwrap();
        return;
    }
    gpu_test(2048 * 1024 * 1024, |session| {
        let with_general = std::env::var("CHOPINFER_SAMPLING_GENERAL_BASELINE").as_deref() == Ok("1");
        for dtype in [DType::Bf16, DType::F32] {
            let (module, general) = function(session, dtype)?;
            let mut functions = filtered_functions(&module, dtype)?;
            for temperature in [1.0, 0.8] {
                for batch in [1_usize, 4, 16, 32] {
                    let rows = default_rows(batch, dtype, temperature);
                    let fixture = Fixture::new(session, 152_576, dtype, batch)?;
                    fixture.upload(session, &rows)?;
                    functions.stages = 6;
                    fixture.launch_filtered(session, &functions, batch)?;
                    check_tokens("bench_filtered", &expected_rows(&rows)?, &fixture.output(session, &rows)?, 0)?;
                    // One call per graph and many rounds: on a shared device the
                    // minimum approximates an uncontended time slice.
                    let measure = |launch: &dyn Fn() -> LabResult<()>| -> LabResult<(f64, f64)> {
                        let graph = session.capture(|_| launch())?;
                        let timing = time(session, 5, 41, |stream| {
                            graph.launch(stream).map_err(|e| e.to_string())
                        })?;
                        Ok((
                            f64::from(timing.min_ms) * 1000.0,
                            f64::from(timing.median_ms) * 1000.0,
                        ))
                    };
                    let mut prefix = Vec::new();
                    for stages in 1..=6 {
                        functions.stages = stages;
                        prefix.push(measure(&|| {
                            fixture.launch_filtered(session, &functions, batch)
                        })?);
                    }
                    functions.stages = 6;
                    let (after_min, after_median) = prefix[5];
                    let (simple_partials, simple_finish) = simple_functions(&module, dtype, false)?;
                    let unfiltered = measure(&|| {
                        fixture.launch_simple(session, &simple_partials, &simple_finish, batch, false)
                    })?;
                    let (greedy_partials, greedy_finish) = greedy_functions(&module, dtype)?;
                    let greedy = measure(&|| {
                        fixture.launch_simple(session, &greedy_partials, &greedy_finish, batch, false)
                    })?;
                    let before = if with_general && temperature == 1.0 {
                        let timing = time(session, 1, 3, |_| fixture.launch(session, &general, batch))?;
                        Some(f64::from(timing.min_ms) * 1000.0)
                    } else {
                        None
                    };
                    let mut previous = 0.0;
                    let breakdown: Vec<f64> = prefix
                        .iter()
                        .map(|&(total, _)| {
                            let stage = total - previous;
                            previous = total;
                            stage
                        })
                        .collect();
                    eprintln!(
                        "filtered {dtype:?} t={temperature} rows={batch}: min {after_min:.2} us, median {after_median:.2} us, general {before:?} us, unfiltered {unfiltered:.2?}, greedy {greedy:.2?}, stage mins {breakdown:.2?}"
                    );
                    session.receipt(
                        "sampling",
                        BTreeMap::from([
                            ("case".into(), json!("filtered_default_top_p")),
                            ("dtype".into(), json!(format!("{dtype:?}"))),
                            ("rows".into(), json!(batch)),
                            ("temperature".into(), json!(temperature)),
                            ("top_p".into(), json!(0.95)),
                            ("width".into(), json!(152_576)),
                            ("after_min_us".into(), json!(after_min)),
                            ("after_median_us".into(), json!(after_median)),
                            ("general_before_min_us".into(), json!(before)),
                            ("unfiltered_sampled_min_median_us".into(), json!([unfiltered.0, unfiltered.1])),
                            ("greedy_min_median_us".into(), json!([greedy.0, greedy.1])),
                            ("stage_min_us".into(), json!({
                                "stats": breakdown[0],
                                "histogram_select": breakdown[1],
                                "collect": breakdown[2],
                                "resolve": breakdown[3],
                                "gumbel": breakdown[4],
                                "finish": breakdown[5],
                            })),
                            ("rounds".into(), json!(41)),
                        ]),
                    )?;
                }
            }
            // Worst case: an exact top-p tie sorts all 152,576 survivors.
            for batch in [1_usize, 32] {
                let rows: Vec<_> = (0..batch)
                    .map(|index| {
                        tie_row(
                            152_576,
                            &[5.0; 4],
                            SamplingParameters::new(1.0).unwrap().with_top_p(0.5).unwrap(),
                            index as u64,
                        )
                    })
                    .collect();
                let fixture = Fixture::new(session, 152_576, dtype, batch)?;
                fixture.upload(session, &rows)?;
                functions.stages = 6;
                fixture.launch_filtered(session, &functions, batch)?;
                check_tokens("bench_fallback", &expected_rows(&rows)?, &fixture.output(session, &rows)?, 0)?;
                if !fixture.exact_fallbacks(session, batch)?.into_iter().all(|used| used) {
                    return Err("fallback benchmark rows did not use the exact path".into());
                }
                let timing = time(session, 1, 3, |_| fixture.launch_filtered(session, &functions, batch))?;
                let fallback_us = f64::from(timing.median_ms) * 1000.0;
                eprintln!("filtered {dtype:?} exact fallback rows={batch}: {fallback_us:.1} us");
                session.receipt(
                    "sampling",
                    BTreeMap::from([
                        ("case".into(), json!("filtered_exact_fallback")),
                        ("dtype".into(), json!(format!("{dtype:?}"))),
                        ("rows".into(), json!(batch)),
                        ("survivors_per_row".into(), json!(152_576)),
                        ("median_us".into(), json!(fallback_us)),
                    ]),
                )?;
            }
        }
        Ok(())
    })
    .expect("filtered sampling benchmarks");
}

/// Rows of real F32 logits at width 152,576 from the files listed in
/// `CHOPINFER_SAMPLING_REAL_LOGITS` (colon separated), or none when unset.
fn real_logits() -> LabResult<Vec<Vec<f32>>> {
    let Ok(paths) = std::env::var("CHOPINFER_SAMPLING_REAL_LOGITS") else {
        return Ok(Vec::new());
    };
    let mut rows = Vec::new();
    for path in paths.split(':').filter(|path| !path.is_empty()) {
        let bytes = fs::read(path).map_err(|error| format!("{path}: {error}"))?;
        if bytes.len() % (152_576 * 4) != 0 {
            return Err(format!(
                "{path} is not a whole number of 152,576-wide F32 rows"
            ));
        }
        for row in bytes.chunks_exact(152_576 * 4) {
            rows.push(
                row.chunks_exact(4)
                    .map(|word| f32::from_le_bytes(word.try_into().expect("four bytes")))
                    .collect(),
            );
        }
    }
    Ok(rows)
}

/// Decode rows over real logits as in the serving cost comparison: default
/// sampling or greedy, both with EOS masked by `min_tokens` and one top
/// logprob.
fn real_rows(
    real: &[Vec<f32>],
    dtype: DType,
    batch: usize,
    offset: usize,
    greedy: bool,
) -> Vec<Row> {
    (0..batch)
        .map(|index| {
            let mut logits = real[(offset + index) % real.len()].clone();
            if dtype == DType::Bf16 {
                for value in &mut logits {
                    *value = bf16_to_f32(f32_to_bf16(*value));
                }
            }
            let parameters = if greedy {
                SamplingParameters::new(0.0).unwrap()
            } else {
                SamplingParameters::new(1.0)
                    .unwrap()
                    .with_top_p(0.95)
                    .unwrap()
            };
            Row {
                logits,
                parameters: parameters
                    .with_minimum_new_tokens(256)
                    .unwrap()
                    .with_top_logprobs(1)
                    .unwrap(),
                state: SamplingState::new((offset + index) as u64),
                history: Vec::new(),
                emitted: 0,
                eos: vec![151_643, 151_645],
                mask: None,
            }
        })
        .collect()
}

#[test]
fn sampling_filtered_real_benchmarks() {
    if std::env::var("CHOPINFER_KERNEL_BENCH").as_deref() != Ok("1") {
        kernel_lab::skip("sampling_filtered_real_benchmarks: set CHOPINFER_KERNEL_BENCH=1").unwrap();
        return;
    }
    let real = real_logits().expect("real logits");
    if real.is_empty() {
        return;
    }
    gpu_test(2048 * 1024 * 1024, |session| {
        let dtype = DType::Bf16;
        let (module, _) = function(session, dtype)?;
        let mut functions = filtered_functions(&module, dtype)?;
        for batch in [1_usize, 16, 32] {
            for greedy in [false, true] {
                let fixture = Fixture::new(session, 152_576, dtype, batch)?;
                let mut totals = Vec::new();
                let mut stage_sums = [0.0_f64; 6];
                let mut fallbacks = 0;
                let mut windows = Vec::new();
                // CHOPINFER_SAMPLING_REAL_SETS raises the eight row sets.
                let sets = real.len().div_ceil(batch).min(
                    std::env::var("CHOPINFER_SAMPLING_REAL_SETS")
                        .ok()
                        .and_then(|sets| sets.parse().ok())
                        .unwrap_or(8),
                );
                for set in 0..sets {
                    let rows = real_rows(&real, dtype, batch, set * batch, greedy);
                    fixture.upload(session, &rows)?;
                    functions.stages = 6;
                    fixture.launch_filtered(session, &functions, batch)?;
                    check_tokens("real_filtered", &expected_rows(&rows)?, &fixture.output(session, &rows)?, 0)?;
                    fallbacks += fixture.exact_fallbacks(session, batch)?.into_iter().filter(|&used| used).count();
                    windows.extend(fixture.candidates(session, batch)?);
                    let mut previous = 0.0;
                    for stages in 1..=6 {
                        functions.stages = stages;
                        let graph = session.capture(|_| fixture.launch_filtered(session, &functions, batch))?;
                        let timing = time(session, 3, 21, |stream| {
                            graph.launch(stream).map_err(|e| e.to_string())
                        })?;
                        let total = f64::from(timing.min_ms) * 1000.0;
                        stage_sums[stages - 1] += total - previous;
                        previous = total;
                    }
                    totals.push(previous);
                }
                windows.sort_unstable();
                let stages: Vec<f64> = stage_sums.iter().map(|sum| sum / sets as f64).collect();
                let mean = totals.iter().sum::<f64>() / totals.len() as f64;
                eprintln!(
                    "real {dtype:?} greedy={greedy} rows={batch}: mean {mean:.1} us over {sets} sets {totals:.1?}, stages {stages:.1?}, fallbacks {fallbacks}, candidates median {} max {}",
                    windows[windows.len() / 2],
                    windows[windows.len() - 1]
                );
                session.receipt(
                    "sampling",
                    BTreeMap::from([
                        ("case".into(), json!("filtered_real_logits")),
                        ("dtype".into(), json!(format!("{dtype:?}"))),
                        ("rows".into(), json!(batch)),
                        ("greedy".into(), json!(greedy)),
                        ("sets".into(), json!(sets)),
                        ("mean_min_us".into(), json!(mean)),
                        ("set_min_us".into(), json!(totals)),
                        ("stage_mean_min_us".into(), json!(stages)),
                        ("exact_fallbacks".into(), json!(fallbacks)),
                        ("candidates_median".into(), json!(windows[windows.len() / 2])),
                        ("candidates_max".into(), json!(windows[windows.len() - 1])),
                    ]),
                )?;
            }
        }
        Ok(())
    })
    .expect("real-logit filtered sampling benchmarks");
}

/// Rows whose logits share a common offset (zero, +-1e20 on all-zero rows,
/// +-128 on a tiled pattern), sampled at 0.7 with three top logprobs; the
/// filtered rows add top-k and top-p and, for even seeds, a mask across tile
/// edges. Each offset row samples the token of its unshifted row on the CPU.
fn common_offset_rows(width: usize, dtype: DType, filtered: bool) -> LabResult<Vec<Row>> {
    let mut rows = Vec::new();
    for pattern in 0..2 {
        let base: Vec<f32> = (0..width)
            .map(|token| {
                if pattern == 0 {
                    0.0
                } else {
                    -2.0 * ((token / 1024) % 3) as f32
                }
            })
            .collect();
        for seed in 0..4 {
            for offset in if pattern == 0 {
                [0.0, 1e20, -1e20]
            } else {
                [0.0, 128.0, -128.0]
            } {
                let logits = base
                    .iter()
                    .map(|value| value + offset)
                    .map(|value| {
                        if dtype == DType::Bf16 {
                            bf16_to_f32(f32_to_bf16(value))
                        } else {
                            value
                        }
                    })
                    .collect();
                let parameters = SamplingParameters::new(0.7)
                    .unwrap()
                    .with_top_logprobs(3)
                    .unwrap();
                let parameters = if filtered {
                    parameters.with_top_k(20).unwrap().with_top_p(0.9).unwrap()
                } else {
                    parameters
                };
                let mut row = Row {
                    logits,
                    parameters,
                    state: SamplingState::new(seed),
                    history: vec![],
                    emitted: 0,
                    eos: vec![],
                    mask: None,
                };
                if filtered && seed % 2 == 0 {
                    let mut mask = vec![0; width.div_ceil(32)];
                    for token in [0, 1023, 1024, width - 1] {
                        mask[token / 32] |= 1 << (token % 32);
                    }
                    row.mask = Some(mask);
                }
                let mut unshifted = row.clone();
                unshifted.logits = base.clone();
                assert_eq!(row.expected()?.token_id(), unshifted.expected()?.token_id());
                rows.push(row);
            }
        }
    }
    Ok(rows)
}

#[test]
fn sampling_common_offsets_preserve_random_selection() {
    gpu_test(2048 * 1024 * 1024, |session| {
        for dtype in [DType::F32, DType::Bf16] {
            for width in [1025_usize, 152_576, 248_320] {
                for filtered in [false, true] {
                    let rows = common_offset_rows(width, dtype, filtered)?;
                    if filtered {
                        let (worst, fallbacks) = run_filtered_batches(
                            session,
                            "common_offset_filtered",
                            dtype,
                            &rows,
                            24,
                        )?;
                        session.receipt(
                            "sampling",
                            BTreeMap::from([
                                ("case".into(), json!("common_offset_filtered")),
                                ("dtype".into(), json!(format!("{dtype:?}"))),
                                ("vocab".into(), json!(width)),
                                ("rows".into(), json!(rows.len())),
                                ("tokens_exact".into(), json!(true)),
                                ("worst_logprob_error".into(), json!(worst)),
                                ("logprob_tolerance".into(), json!(LOGPROB_TOLERANCE)),
                                ("exact_fallbacks".into(), json!(fallbacks)),
                                ("correct".into(), json!(true)),
                            ]),
                        )?;
                    } else {
                        for chunk in rows.chunks(24) {
                            run_once(session, chunk, dtype, "common_offset_simple", true)?;
                            run_once(session, chunk, dtype, "common_offset_serial", false)?;
                        }
                    }
                }
            }
        }
        Ok(())
    })
    .expect("sampling preserves a common logit offset");
}

/// A greedy and a sampled row whose mask admits only token 1024, with three
/// top logprobs.
fn singleton_mask_rows(width: usize) -> Vec<Row> {
    let mut mask = vec![0; width.div_ceil(32)];
    mask[32] = 1;
    [0.0, 1.0]
        .into_iter()
        .map(|temperature| Row {
            logits: vec![1.0; width],
            parameters: SamplingParameters::new(temperature)
                .unwrap()
                .with_top_logprobs(3)
                .unwrap(),
            state: SamplingState::new(7),
            history: vec![],
            emitted: 0,
            eos: vec![],
            mask: Some(mask.clone()),
        })
        .collect()
}

#[test]
fn sampling_singleton_mask_preserves_top_logprob_tail() {
    gpu_test(512 * 1024 * 1024, |session| {
        for dtype in [DType::F32, DType::Bf16] {
            let width = 1025_usize;
            let rows = singleton_mask_rows(width);
            run_once(session, &rows, dtype, "singleton_mask_serial", false)?;
            let (worst, fallbacks) =
                run_filtered_batches(session, "singleton_mask_filtered", dtype, &rows, 2)?;
            session.receipt(
                "sampling",
                BTreeMap::from([
                    ("case".into(), json!("singleton_mask_filtered")),
                    ("dtype".into(), json!(format!("{dtype:?}"))),
                    ("vocab".into(), json!(width)),
                    ("rows".into(), json!(rows.len())),
                    ("tokens_exact".into(), json!(true)),
                    ("worst_logprob_error".into(), json!(worst)),
                    ("logprob_tolerance".into(), json!(LOGPROB_TOLERANCE)),
                    ("exact_fallbacks".into(), json!(fallbacks)),
                    ("correct".into(), json!(true)),
                ]),
            )?;
        }
        Ok(())
    })
    .expect("masked top-logprob tails match the host decoder");
}

/// Every output record of one launch over `rows`, after poisoning.
fn launch_records(
    session: &Session,
    fixture: &Fixture,
    rows: &[Row],
    launch: impl Fn() -> LabResult<()>,
) -> LabResult<Vec<u8>> {
    fixture.upload(session, rows)?;
    fixture.poison(session)?;
    launch()?;
    let mut bytes = vec![0; rows.len() * RECORD_BYTES];
    fixture.buffers[&OUTPUT]
        .copy_to_bytes(&mut bytes, &session.stream)
        .map_err(|error| error.to_string())?;
    Ok(bytes)
}

/// Compares the old and new records of one view. A row in `empty` has no
/// candidate: the old finish wrote status 1 and the new one status 2, and
/// neither wrote any other byte. Every other row is sampled and its whole
/// record, top entries and untouched poison included, is bitwise the old
/// one. Returns the number of sampled rows.
fn same_records(
    label: &str,
    rows: &[Row],
    empty: &[bool],
    before: &[u8],
    after: &[u8],
) -> LabResult<usize> {
    let mut sampled = 0;
    for (index, (old, new)) in before
        .chunks_exact(RECORD_BYTES)
        .zip(after.chunks_exact(RECORD_BYTES))
        .enumerate()
    {
        let (old_status, new_status) = (u32_at(old, 4), u32_at(new, 4));
        if empty[index] {
            if (old_status, new_status) != (1, 2) || old[..4] != new[..4] || old[8..] != new[8..] {
                return Err(format!(
                    "{label} row {index}: no-candidate statuses {old_status} -> {new_status} or other bytes differ"
                ));
            }
            let expected = rows[index].expected().err();
            if expected != Some(runtime_sampling::SamplingError::AllTokensMasked.to_string()) {
                return Err(format!(
                    "{label} row {index}: the CPU sampler gives {expected:?}"
                ));
            }
        } else {
            if new_status != 0 || old != new {
                return Err(format!(
                    "{label} row {index}: record differs from the baseline (status {old_status} -> {new_status})"
                ));
            }
            sampled += 1;
        }
    }
    Ok(sampled)
}

#[test]
fn same_records_accepts_status_2_only_for_rows_without_a_candidate() {
    let width = 256;
    let mut masked = filtered_row(width, DType::F32, 3);
    masked.mask = Some(vec![0; width / 32]);
    let mut stops = filtered_row(width, DType::F32, 4);
    stops.mask = None;
    stops.eos = (0..width as u32).collect();
    stops.parameters = stops
        .parameters
        .clone()
        .with_minimum_new_tokens(stops.emitted + 1)
        .unwrap();
    let sampled = random_row(width, DType::F32, 5, 1.0);
    let rows = [masked, stops, sampled];
    let empty = [true, true, false];
    let mut before = vec![0xa5; 3 * RECORD_BYTES];
    for (row, status) in [(0, 1), (1, 1), (2, 0)] {
        put_u32(&mut before, row * RECORD_BYTES + 4, status);
    }
    let mut after = before.clone();
    for row in 0..2 {
        put_u32(&mut after, row * RECORD_BYTES + 4, 2);
    }
    assert_eq!(same_records("ok", &rows, &empty, &before, &after), Ok(1));
    for (offset, value) in [
        (4, 1),
        (8, 0),
        (2 * RECORD_BYTES + 4, 2),
        (2 * RECORD_BYTES + 8, 0),
    ] {
        let mut wrong = after.clone();
        put_u32(&mut wrong, offset, value);
        assert!(
            same_records("wrong", &rows, &empty, &before, &wrong).is_err(),
            "{offset}"
        );
    }
    let mut sampled_row = rows.clone();
    sampled_row[0].mask = None;
    assert!(same_records("sampled", &sampled_row, &empty, &before, &after).is_err());
}

/// The sampling path a parity view launches.
#[derive(Clone, Copy, PartialEq)]
enum Path {
    /// The one-block entry `chop_sampling_*_v1`, which the host does not
    /// launch.
    Serial,
    Greedy,
    Top,
    Plain,
    Filtered,
}

impl Path {
    /// The simple path `run_once` takes for `rows`.
    fn simple(rows: &[Row]) -> Path {
        if rows.iter().any(|row| row.parameters.top_logprobs() != 0) {
            Path::Top
        } else if rows.iter().all(|row| row.parameters.temperature() == 0.0) {
            Path::Greedy
        } else {
            Path::Plain
        }
    }
}

/// One module's entries for a path.
enum Kernels {
    Serial(Function),
    Simple(Function, Function, bool),
    Filtered(FilteredFunctions),
}

impl Kernels {
    fn new(module: &runtime_cuda::Module, dtype: DType, path: Path) -> LabResult<Self> {
        let simple = |top| -> LabResult<Kernels> {
            let (partials, finish) = simple_functions(module, dtype, top)?;
            Ok(Kernels::Simple(partials, finish, top))
        };
        Ok(match path {
            Path::Serial => Kernels::Serial(
                module
                    .function(if dtype == DType::Bf16 {
                        "chop_sampling_bf16_v1"
                    } else {
                        "chop_sampling_f32_v1"
                    })
                    .map_err(|error| error.to_string())?,
            ),
            Path::Greedy => {
                let (partials, finish) = greedy_functions(module, dtype)?;
                Kernels::Simple(partials, finish, false)
            }
            Path::Top => simple(true)?,
            Path::Plain => simple(false)?,
            Path::Filtered => Kernels::Filtered(filtered_functions(module, dtype)?),
        })
    }

    fn launch(&self, session: &Session, fixture: &Fixture, rows: usize) -> LabResult<()> {
        match self {
            Kernels::Serial(function) => fixture.launch(session, function, rows),
            Kernels::Simple(partials, finish, top) => {
                fixture.launch_simple(session, partials, finish, rows, *top)
            }
            Kernels::Filtered(functions) => fixture.launch_filtered(session, functions, rows),
        }
    }
}

/// The sampling cubin before the finish entries reported a row without a
/// candidate as status 2 (`CHOPINFER_SAMPLING_BASELINE_CUBIN`) against this
/// crate's, over the same views.
struct Parity<'a> {
    session: &'a Session,
    path: String,
    old: runtime_cuda::Module,
    new: runtime_cuda::Module,
    dtype: DType,
    views: usize,
    sampled: usize,
    empty: usize,
    replays: usize,
}

impl<'a> Parity<'a> {
    fn new(session: &'a Session, path: String, dtype: DType) -> LabResult<Self> {
        let bytes = fs::read(&path).map_err(|error| format!("{path}: {error}"))?;
        let old = session
            .context
            .load_module(&bytes)
            .map_err(|error| error.to_string())?;
        let (new, _) = function(session, dtype)?;
        Ok(Self {
            session,
            path,
            old,
            new,
            dtype,
            views: 0,
            sampled: 0,
            empty: 0,
            replays: 0,
        })
    }

    /// Launches `rows` in views of `batch` on a fixture of `capacity` rows,
    /// the case's own geometry, with both cubins (`same_records`). `empty`
    /// marks planted rows without a candidate.
    fn views(
        &mut self,
        label: &str,
        path: Path,
        rows: &[Row],
        empty: Option<&[bool]>,
        batch: usize,
        capacity: usize,
    ) -> LabResult<()> {
        let (session, dtype) = (self.session, self.dtype);
        let fixture = Fixture::new(session, rows[0].logits.len(), dtype, capacity)?;
        let old = Kernels::new(&self.old, dtype, path)?;
        let new = Kernels::new(&self.new, dtype, path)?;
        let none = vec![false; rows.len()];
        let empty = empty.unwrap_or(&none);
        for (view, (part, holes)) in rows.chunks(batch).zip(empty.chunks(batch)).enumerate() {
            let before = launch_records(session, &fixture, part, || {
                old.launch(session, &fixture, part.len())
            })?;
            let after = launch_records(session, &fixture, part, || {
                new.launch(session, &fixture, part.len())
            })?;
            let label = format!("{label} {dtype:?} view {view}");
            let sampled = same_records(&label, part, holes, &before, &after)?;
            self.views += 1;
            self.sampled += sampled;
            self.empty += part.len() - sampled;
        }
        Ok(())
    }

    /// One view of `rows`, with each cubin: the eager launch, a relaunch, a
    /// graph replay, on the filtered path a graph of two launches on their
    /// own records and scratch (`graph_pair`), and a PDL launch, each
    /// checked against the write maps and bitwise against its eager launch.
    /// The two cubins' eager records must be equal.
    fn replays(&mut self, label: &str, path: Path, rows: &[Row]) -> LabResult<()> {
        let (session, dtype) = (self.session, self.dtype);
        let fixture = Fixture::new(session, rows[0].logits.len(), dtype, rows.len())?;
        fixture.upload(session, rows)?;
        let producer = PdlProducer::new(session)?;
        let mut eager = Vec::new();
        for module in [&self.old, &self.new] {
            let kernels = Kernels::new(module, dtype, path)?;
            let launch = || kernels.launch(session, &fixture, rows.len());
            let mut checks = fixture.replay(session, rows, path == Path::Filtered)?;
            checks.eager(launch)?;
            checks.relaunch(launch)?;
            let graph = session.capture(|_| launch())?;
            checks.graph(&graph)?;
            if let Kernels::Filtered(functions) = &kernels {
                let second = fixture.second(session)?;
                let pair = session.capture(|_| {
                    launch()?;
                    fixture.launch_filtered_rebound(
                        session,
                        functions,
                        rows.len(),
                        &rebind(&second),
                        true,
                    )
                })?;
                graph_pair(session, &mut checks, &pair, &second, rows)?;
            }
            checks.pdl(|| producer.launch(session), launch, true)?;
            eager.push(
                checks
                    .expected("records")
                    .ok_or("no eager records")?
                    .to_vec(),
            );
        }
        if eager[0] != eager[1] {
            return Err(format!(
                "{label} {dtype:?}: replayed records differ from the baseline"
            ));
        }
        self.replays += 1;
        Ok(())
    }

    fn receipt(&self, case: &str) -> LabResult<()> {
        self.session.receipt(
            "sampling",
            BTreeMap::from([
                ("case".into(), json!(case)),
                ("dtype".into(), json!(format!("{:?}", self.dtype))),
                ("baseline_cubin".into(), json!(self.path)),
                ("views".into(), json!(self.views)),
                ("sampled_rows_bitwise".into(), json!(self.sampled)),
                ("no_candidate_rows_status_2".into(), json!(self.empty)),
                ("replayed_views".into(), json!(self.replays)),
                ("correct".into(), json!(true)),
            ]),
        )?;
        Ok(())
    }
}

/// Runs `cases` for F32 and BF16 when `CHOPINFER_SAMPLING_BASELINE_CUBIN`
/// names the old cubin.
fn baseline_parity(case: &str, cases: impl Fn(&mut Parity<'_>) -> LabResult<()>) {
    let Ok(path) = std::env::var("CHOPINFER_SAMPLING_BASELINE_CUBIN") else {
        kernel_lab::skip(&format!(
            "{case}: set CHOPINFER_SAMPLING_BASELINE_CUBIN to the old sampling cubin"
        ))
        .unwrap();
        return;
    };
    gpu_test(4096 * 1024 * 1024, |session| {
        for dtype in [DType::F32, DType::Bf16] {
            let mut parity = Parity::new(session, path.clone(), dtype)?;
            cases(&mut parity)?;
            parity.receipt(case)?;
        }
        Ok(())
    })
    .expect("sampling records match the baseline cubin");
}

/// The old cubin against the new over the simple-path and one-block cases
/// of this file, each with its own rows, widths, view sizes and fixture
/// capacities: `greedy_exact`, `top_one` (with its graph replays), the
/// planner geometry, the 10,000 seeded rows per width, the penalty rows on
/// the one-block entry, the graph replay rows, the common offsets and the
/// singleton mask, plus a randomized sweep of the greedy, top and plain
/// partials. Every record must be bitwise the old one.
#[test]
fn sampling_baseline_parity_simple() {
    baseline_parity("baseline_parity_simple", |parity| {
        let dtype = parity.dtype;
        for width in [38_144, 152_576] {
            for (label, rows) in greedy_exact_cases(width, dtype) {
                parity.views(
                    &label,
                    Path::simple(&rows),
                    &rows,
                    None,
                    rows.len(),
                    rows.len(),
                )?;
            }
        }
        for width in [1025, 38_144, 152_576] {
            for start in (0..512).step_by(32) {
                let mut rows = top_one_rows(width, dtype, start);
                parity.views("top_one_sorted", Path::Top, &rows, None, 32, 32)?;
                for row in &mut rows {
                    row.parameters = row.parameters.clone().with_top_logprobs(1).unwrap();
                }
                parity.views("top_one", Path::Top, &rows, None, 32, 32)?;
                parity.replays("top_one", Path::Top, &rows)?;
            }
        }
        parity.views(
            "top_planner_b9",
            Path::Top,
            &planner_rows(38_144, dtype),
            None,
            9,
            16,
        )?;
        for width in [38_144, 152_576] {
            for start in (0..10_000).step_by(64) {
                let rows = seeded_rows(width, dtype, start);
                parity.views("seeded", Path::Plain, &rows, None, 64, 64)?;
            }
            let rows = penalty_rows(width, dtype);
            parity.views("penalties_serial", Path::Serial, &rows, None, 8, 8)?;
        }
        let rows = simple_graph_rows(dtype);
        parity.views("graph", Path::Plain, &rows, None, 8, 8)?;
        parity.replays("graph", Path::Plain, &rows)?;
        for width in [1025, 152_576, 248_320] {
            let rows = common_offset_rows(width, dtype, false)?;
            for chunk in rows.chunks(24) {
                let path = Path::simple(chunk);
                parity.views("common_offset_simple", path, chunk, None, 24, chunk.len())?;
                parity.views(
                    "common_offset_serial",
                    Path::Serial,
                    chunk,
                    None,
                    24,
                    chunk.len(),
                )?;
            }
        }
        let rows = singleton_mask_rows(1025);
        parity.views("singleton_mask_serial", Path::Serial, &rows, None, 2, 2)?;
        for width in [1025, 38_144, 152_576] {
            let seed = |index: usize| (width * 1000 + index) as u64;
            let greedy: Vec<_> = (0..128)
                .map(|index| {
                    let mut row = random_row(width, dtype, seed(index), 0.0);
                    row.parameters = row.parameters.with_top_logprobs(index % 2).unwrap();
                    row
                })
                .collect();
            let temperatures = [0.0, 0.6, 1.0, 1.7, 1e-37];
            let top: Vec<_> = (0..128)
                .map(|index| {
                    let temperature = temperatures[index % temperatures.len()];
                    let mut row = random_row(width, dtype, seed(index) ^ 0x7031, temperature);
                    row.parameters = row
                        .parameters
                        .with_top_logprobs([0, 1, 5, 20][index % 4])
                        .unwrap();
                    row
                })
                .collect();
            let plain: Vec<_> = (0..128)
                .map(|index| {
                    let temperature = temperatures[index % temperatures.len()];
                    random_row(width, dtype, seed(index) ^ 0x51a1, temperature)
                })
                .collect();
            parity.views("random_greedy", Path::Greedy, &greedy, None, 64, 64)?;
            parity.views("random_top", Path::Top, &top, None, 64, 64)?;
            parity.views("random_plain", Path::Plain, &plain, None, 64, 64)?;
        }
        Ok(())
    });
}

/// The old cubin against the new over the filtered cases of this file, each
/// with its own rows, widths, view sizes and fixture capacities: the penalty
/// rows, the randomized sweep, the tiny temperatures, the fixture's real
/// rows, the boundary ties, the graph replay rows (with their graph pair),
/// the graph-pair control rows, the common offsets and the singleton mask.
/// Every record must be bitwise the old one. Then views with planted rows
/// without a candidate: an all-zero mask in two rows of every view of the
/// sweep's first 384 rows per width, and every token of a 256-token row an
/// EOS token before the minimum length. Those rows must move from status 1
/// to status 2 with no other byte written, and every other row keeps its
/// old record.
#[test]
fn sampling_baseline_parity_filtered() {
    baseline_parity("baseline_parity_filtered", |parity| {
        let dtype = parity.dtype;
        for width in [38_144, 152_576] {
            let rows = penalty_rows(width, dtype);
            parity.views("penalties", Path::Filtered, &rows, None, 8, 8)?;
        }
        for (width, count) in FILTERED_SWEEP {
            let mut rows = filtered_sweep_rows(width, dtype, count);
            parity.views("filtered_sweep", Path::Filtered, &rows, None, 64, 64)?;
            rows.truncate(384);
            let empty: Vec<_> = (0..rows.len())
                .map(|index| matches!(index % 64, 5 | 40))
                .collect();
            for (row, &planted) in rows.iter_mut().zip(&empty) {
                if planted {
                    row.mask = Some(vec![0; width.div_ceil(32)]);
                }
            }
            parity.views("empty_mask", Path::Filtered, &rows, Some(&empty), 64, 64)?;
        }
        let rows = tiny_temperature_rows(152_576, dtype);
        parity.views("tiny_temperature", Path::Filtered, &rows, None, 64, 64)?;
        let rows = real_fixture_rows(&fixture_logits()?, dtype);
        parity.views("real_rows", Path::Filtered, &rows, None, 64, 64)?;
        let (rows, _) = boundary_tie_rows(152_576, dtype);
        parity.views("boundary_ties", Path::Filtered, &rows, None, 64, 64)?;
        let rows = filtered_graph_rows(152_576, dtype);
        parity.views("filtered_graph", Path::Filtered, &rows, None, 16, 16)?;
        parity.replays("filtered_graph", Path::Filtered, &rows)?;
        parity.views(
            "graph_pair_controls",
            Path::Filtered,
            &rows[..4],
            None,
            4,
            4,
        )?;
        for width in [1025, 152_576, 248_320] {
            let rows = common_offset_rows(width, dtype, true)?;
            parity.views(
                "common_offset_filtered",
                Path::Filtered,
                &rows,
                None,
                24,
                24,
            )?;
        }
        let rows = singleton_mask_rows(1025);
        parity.views("singleton_mask_filtered", Path::Filtered, &rows, None, 2, 2)?;
        let width = 256;
        let mut rows: Vec<_> = (0..192)
            .map(|index| filtered_row(width, dtype, 0xe05 + index as u64))
            .collect();
        let empty: Vec<_> = (0..rows.len())
            .map(|index| matches!(index % 64, 9 | 50))
            .collect();
        for (row, &planted) in rows.iter_mut().zip(&empty) {
            if planted {
                row.mask = None;
                row.eos = (0..width as u32).collect();
                row.parameters = row
                    .parameters
                    .clone()
                    .with_minimum_new_tokens(row.emitted + 1)
                    .unwrap();
            }
        }
        parity.views("all_eos_w256", Path::Filtered, &rows, Some(&empty), 64, 64)
    });
}
