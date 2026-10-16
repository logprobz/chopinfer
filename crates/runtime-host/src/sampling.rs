//! Host-owned, model-neutral device sampling for a borrowed logits view.
//!
//! `kernel_core::sampling` states what the sampler computes for each row,
//! its records and its statuses; this module launches the SM120 entries that
//! compute it.
//!
//! # The SM120 entries
//!
//! `kernels_sm120::sampling` compiles one cubin; the session launches it
//! with programmatic dependent launch over a view of `rows` rows (the view's
//! own row count) and `tiles = ceil(live / 1024)` tiles. The entry choice,
//! the partials thread count and the histogram groups follow a
//! `SamplingDispatch`: the view's own, or, for an owner partition of a
//! gathered view, the gathered view's, which every partition keeps so that
//! it launches as the unsplit view would. Its row count (`dispatch_rows`
//! below: the gathered view's, at least `rows`) and its simple, top and
//! greedy classes
//! describe the whole gathered view, so a partition may take a path its own
//! rows would not (for example eight local rows of a 16-row gathered view
//! launch 64 threads), but never one they cannot take: a simple or greedy
//! dispatch over rows that are not, or a dispatch without top logprobs over
//! rows with them, is refused. Grids count the view's own `rows`.
//! - If the dispatch is simple (each of its rows has no mask, no EOS mask,
//!   no bias, identity penalties, and is greedy or without min-p, top-k and
//!   top-p): a partials entry on grid `[tiles, rows]` (256 threads when a
//!   dispatch row has top logprobs or `dispatch_rows <= 8`, else 64), then
//!   `chop_sampling_simple_finish_{f32,bf16}_v1` on grid `[rows]` with 32
//!   threads. The partials entry is
//!   `chop_sampling_greedy_partials_1024_*_v1` when every dispatch row is
//!   greedy with at most one top logprob,
//!   `chop_sampling_simple_top_partials_*_v1` when a dispatch row has top
//!   logprobs, and `chop_sampling_simple_partials_*_v1` otherwise.
//! - Otherwise the six `chop_sampling_filtered_{stats, histogram, collect,
//!   resolve, gumbel, finish}_*_v1` stages, with 256, 256, 256, 1024, 256
//!   and 32 threads, on grids `[tiles, rows]`, `[groups, rows]`,
//!   `[tiles, rows]`, `[rows]`, `[tiles, rows]` and `[rows]`, where
//!   `groups = max(ceil(tiles / 4), min(ceil(384 / dispatch_rows), tiles))`.
//!
//! Both finish entries write the row's status: `STATUS_INVALID` for a
//! non-finite live logit or more than 20 top logprobs, else `STATUS_EMPTY`
//! for a row without a candidate. A failing row's token, logprob and top
//! entries keep the bytes they held, so `decode_output` reads only the
//! status of a failing row.
//!
//! The entries' scratch, for `rows` rows and `live` tokens, is at least
//! `partials` of `rows * ceil(live / 1024) * 112` bytes, `filter` of
//! `rows * 32,896` bytes, `adjusted` of `rows * live * 8` bytes and `order`
//! of `rows * live * 4` bytes (`SamplingWorkspace` sizes them for the widest
//! view); the entries leave none of it for the next launch.
//!
//! The one-block-per-row entries `chop_sampling_{f32,bf16}_v1` compute the
//! same row but are not launched by the host; they fuse the frequency
//! product into its subtraction and report `STATUS_MALFORMED`, which the
//! served entries never write.

use std::collections::BTreeMap;

use runtime_bundle_abi::{
    BUNDLE_LOGITS_ELEMENT_V1_BF16, BUNDLE_LOGITS_ELEMENT_V1_F32, BundleLogitsViewV1, BundleStatusV1,
};
use runtime_cuda::{
    Context, CudaError, DeviceBuffer, Function, KernelArgument, LaunchConfig, MemoryPool, Module,
    PinnedHostBuffer, Stream,
};
use runtime_engine::{ModelBatchItem, RequestId, SampledRow, SamplingWork};
use runtime_sampling::{SamplingParameters, SamplingState, TokenLogprob};

use crate::sampling_dispatch::SamplingDispatch;

type SamplingResult<T> = Result<T, BundleStatusV1>;
use crate::sampling_workspace::{
    BIAS_BYTES, COUNT_BYTES, META_BYTES, OUTPUT_BYTES, SAMPLER_VIEW_ROWS, SamplingWorkspace,
};
// Unreserved callers retain the standalone sampler capacity. Serving views
// are split to the reserved workspace bound before reaching this layer.
const MAX_VIEW_ROWS: usize = 1024;

fn host_error(_: CudaError) -> BundleStatusV1 {
    BundleStatusV1::HostError
}

fn checked_u32(value: usize) -> SamplingResult<u32> {
    u32::try_from(value).map_err(|_| BundleStatusV1::InvalidArgument)
}

fn checked_u64(value: usize) -> SamplingResult<u64> {
    u64::try_from(value).map_err(|_| BundleStatusV1::InvalidArgument)
}

fn write_u32(out: &mut [u8], offset: usize, value: u32) {
    out[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

fn write_u64(out: &mut [u8], offset: usize, value: u64) {
    out[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
}

fn write_f64(out: &mut [u8], offset: usize, value: f64) {
    write_u64(out, offset, value.to_bits());
}

fn read_u32(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(
        bytes[offset..offset + 4]
            .try_into()
            .expect("bounded output record"),
    )
}

fn read_f64(bytes: &[u8], offset: usize) -> f64 {
    f64::from_bits(u64::from_le_bytes(
        bytes[offset..offset + 8]
            .try_into()
            .expect("bounded output record"),
    ))
}

struct EncodedBatch {
    bytes: Vec<u8>,
    counts_at: usize,
    biases_at: usize,
    eos_at: usize,
    masks_at: usize,
    counts_bytes: usize,
    biases_bytes: usize,
    eos_bytes: usize,
    /// Words per row of the constraint masks, or zero when no row has one.
    mask_words: usize,
    simple: bool,
    top: bool,
    greedy: bool,
}

fn row_mask(item: &ModelBatchItem) -> Option<&[u32]> {
    item.sampling()?.constraint.as_ref()?.mask()
}

/// Whether a mask admits nothing but stop tokens, in which case the minimum
/// length yields and a stop token stays allowed.
fn admits_only_stops(mask: &[u32], stops: &[u32]) -> bool {
    let mut remaining = mask.to_vec();
    for &token in stops {
        if let Some(word) = remaining.get_mut(token as usize / 32) {
            *word &= !(1 << (token % 32));
        }
    }
    remaining.iter().all(|word| *word == 0)
}

struct PenaltyCountCache {
    request_id: RequestId,
    history_len: usize,
    prompt_len: usize,
    occurrences: BTreeMap<u32, (bool, u32)>,
}

fn encode_batch(items: &[&ModelBatchItem], live: u32) -> SamplingResult<EncodedBatch> {
    let meta_bytes = items
        .len()
        .checked_mul(META_BYTES)
        .ok_or(BundleStatusV1::ResourceExhausted)?;
    let mut meta = vec![0_u8; meta_bytes];
    let mut counts = Vec::new();
    let mut biases = Vec::new();
    let mut eos = Vec::new();
    let dispatch = SamplingDispatch::for_items(items)?;
    let mut last_counts: Option<PenaltyCountCache> = None;
    for (row, item) in items.iter().enumerate() {
        let work = item.sampling().ok_or(BundleStatusV1::InvalidArgument)?;
        let settings = &work.parameters;
        let prompt_length = work
            .history
            .len()
            .checked_sub(work.emitted_tokens)
            .ok_or(BundleStatusV1::InvalidArgument)?;
        let count_start = counts.len() / COUNT_BYTES;
        let bias_start = biases.len() / BIAS_BYTES;
        let eos_start = eos.len() / 4;
        if settings.repetition_penalty() != 1.0
            || settings.frequency_penalty() != 0.0
            || settings.presence_penalty() != 0.0
        {
            if !matches!(&last_counts, Some(cache)
                if cache.request_id == item.request_id()
                    && cache.history_len <= work.history.len()
                    && cache.prompt_len == prompt_length)
            {
                last_counts = Some(PenaltyCountCache {
                    request_id: item.request_id(),
                    history_len: 0,
                    prompt_len: prompt_length,
                    occurrences: BTreeMap::new(),
                });
            }
            let cache = last_counts.as_mut().expect("count state exists");
            for (index, &token) in work.history.iter().enumerate().skip(cache.history_len) {
                if token >= live {
                    return Err(BundleStatusV1::InvalidArgument);
                }
                let entry = cache.occurrences.entry(token).or_insert((false, 0));
                if index < prompt_length {
                    entry.0 = true;
                } else {
                    entry.1 = entry
                        .1
                        .checked_add(1)
                        .ok_or(BundleStatusV1::ResourceExhausted)?;
                }
            }
            cache.history_len = work.history.len();
            for (&token, &(prompt_seen, output_count)) in &cache.occurrences {
                counts.extend_from_slice(&token.to_le_bytes());
                counts.extend_from_slice(&u32::from(prompt_seen).to_le_bytes());
                counts.extend_from_slice(&output_count.to_le_bytes());
                counts.extend_from_slice(&0_u32.to_le_bytes());
            }
        }
        for bias in settings.logit_bias() {
            if bias.token_id() >= live {
                return Err(BundleStatusV1::InvalidArgument);
            }
            biases.extend_from_slice(&bias.token_id().to_le_bytes());
            biases.extend_from_slice(&0_u32.to_le_bytes());
            biases.extend_from_slice(&bias.bias().to_le_bytes());
        }
        let mask_eos = work.emitted_tokens < settings.minimum_new_tokens()
            && !row_mask(item).is_some_and(|mask| admits_only_stops(mask, &work.stop_ids));
        if mask_eos {
            for &token in &work.stop_ids {
                if token >= live {
                    return Err(BundleStatusV1::InvalidArgument);
                }
                eos.extend_from_slice(&token.to_le_bytes());
            }
        }
        let count_len = counts.len() / COUNT_BYTES - count_start;
        let bias_len = biases.len() / BIAS_BYTES - bias_start;
        let eos_len = eos.len() / 4 - eos_start;
        let record = &mut meta[row * META_BYTES..(row + 1) * META_BYTES];
        write_u64(record, 0, work.state.seed());
        write_u64(record, 8, work.state.generated());
        write_u64(record, 16, checked_u64(count_start)?);
        write_u64(record, 24, checked_u64(bias_start)?);
        write_u64(record, 32, checked_u64(eos_start)?);
        write_u32(record, 40, checked_u32(count_len)?);
        write_u32(record, 44, checked_u32(bias_len)?);
        write_u32(record, 48, checked_u32(eos_len)?);
        write_u32(record, 52, checked_u32(settings.top_k())?);
        write_u32(record, 56, checked_u32(settings.top_logprobs())?);
        write_u32(record, 60, u32::from(mask_eos));
        write_f64(record, 64, settings.temperature());
        write_f64(record, 72, settings.top_p());
        write_f64(record, 80, settings.min_p());
        write_f64(record, 88, settings.repetition_penalty());
        write_f64(record, 96, settings.frequency_penalty());
        write_f64(record, 104, settings.presence_penalty());
    }
    let counts_bytes = counts.len();
    let biases_bytes = biases.len();
    let eos_bytes = eos.len();
    let mut bytes = meta;
    let counts_at = bytes.len();
    bytes.extend(counts);
    bytes.resize(bytes.len().max(counts_at + 16), 0);
    let biases_at = bytes.len();
    bytes.extend(biases);
    bytes.resize(bytes.len().max(biases_at + 16), 0);
    let eos_at = bytes.len();
    bytes.extend(eos);
    bytes.resize(bytes.len().max(eos_at + 8), 0);
    // Constraint masks, one row each; a row without a mask admits everything.
    let masked = items.iter().any(|item| row_mask(item).is_some());
    let mask_words = if masked {
        (live as usize).div_ceil(32)
    } else {
        0
    };
    bytes.resize(bytes.len().next_multiple_of(16), 0);
    let masks_at = bytes.len();
    for item in items {
        match row_mask(item) {
            Some(mask) if mask.len() == mask_words => {
                for word in mask {
                    bytes.extend_from_slice(&word.to_le_bytes());
                }
            }
            Some(_) => return Err(BundleStatusV1::InvalidArgument),
            None => bytes.resize(bytes.len() + mask_words * 4, 0xFF),
        }
    }
    Ok(EncodedBatch {
        bytes,
        counts_at,
        biases_at,
        eos_at,
        masks_at,
        counts_bytes,
        biases_bytes,
        eos_bytes,
        mask_words,
        // The simple kernels have no mask input; the filtered ones do.
        simple: dispatch.simple,
        top: dispatch.top,
        greedy: dispatch.greedy,
    })
}

#[cfg(test)]
pub(crate) fn assert_verify_metadata_for_test(items: &[&ModelBatchItem]) {
    let encoded = encode_batch(items, 32).expect("expanded verify metadata is valid");
    assert!(!encoded.simple);
    for (row, item) in items.iter().enumerate() {
        let work = item.sampling().unwrap();
        let prefix = work.emitted_tokens - 2;
        assert_eq!(work.history.len() - work.emitted_tokens, 2);
        let meta = &encoded.bytes[row * META_BYTES..(row + 1) * META_BYTES];
        assert_eq!(read_u32(meta, 40), 2);
        assert_eq!(read_u32(meta, 44), 1);
        assert_eq!(read_f64(meta, 80), 0.05);
        let count_at = u64::from_le_bytes(meta[16..24].try_into().unwrap()) as usize;
        let counts = &encoded.bytes[encoded.counts_at + count_at * COUNT_BYTES..];
        assert_eq!(
            (
                read_u32(counts, 0),
                read_u32(counts, 4),
                read_u32(counts, 8)
            ),
            (3, 1, prefix as u32)
        );
        assert_eq!(
            (
                read_u32(counts, 16),
                read_u32(counts, 20),
                read_u32(counts, 24)
            ),
            (9, 0, 2)
        );
        let bias_at = u64::from_le_bytes(meta[24..32].try_into().unwrap()) as usize;
        let bias = &encoded.bytes[encoded.biases_at + bias_at * BIAS_BYTES..];
        assert_eq!(read_u32(bias, 0), 9);
        assert_eq!(read_f64(bias, 8), 0.5);
        assert_eq!(
            u64::from_le_bytes(meta[..8].try_into().unwrap()),
            work.state.seed()
        );
        assert_eq!(
            u64::from_le_bytes(meta[8..16].try_into().unwrap()),
            work.state.generated()
        );
    }
}

fn ensure_device(
    buffer: &mut Option<DeviceBuffer>,
    capacity: &mut usize,
    required: usize,
    pool: &MemoryPool,
    stream: &Stream,
) -> SamplingResult<()> {
    if *capacity < required {
        let size = required
            .max(1)
            .checked_next_power_of_two()
            .ok_or(BundleStatusV1::ResourceExhausted)?;
        *buffer = Some(pool.allocate(stream, size).map_err(host_error)?);
        *capacity = size;
    }
    Ok(())
}

fn ensure_pinned(
    buffer: &mut Option<PinnedHostBuffer>,
    capacity: &mut usize,
    required: usize,
    context: &Context,
) -> SamplingResult<()> {
    if *capacity < required {
        let size = required
            .max(1)
            .checked_next_power_of_two()
            .ok_or(BundleStatusV1::ResourceExhausted)?;
        *buffer = Some(PinnedHostBuffer::new(context, size).map_err(host_error)?);
        *capacity = size;
    }
    Ok(())
}

/// Debug timing of `sample`, enabled by `CHOPINFER_SAMPLING_TIMING`: every
/// 1,000 calls it prints mean microseconds spent waiting for the context
/// (the step's forward work), encoding and issuing, and waiting for the
/// kernels and download.
#[derive(Default)]
struct SamplingTiming {
    calls: u64,
    filtered: u64,
    rows: u64,
    wait: std::time::Duration,
    issue: std::time::Duration,
    device: std::time::Duration,
}

impl SamplingTiming {
    fn record(
        &mut self,
        rows: usize,
        filtered: bool,
        wait: std::time::Duration,
        issue: std::time::Duration,
        device: std::time::Duration,
    ) {
        self.calls += 1;
        self.filtered += u64::from(filtered);
        self.rows += rows as u64;
        self.wait += wait;
        self.issue += issue;
        self.device += device;
        if self.calls == 1000 {
            let mean = |total: std::time::Duration| total.as_secs_f64() * 1e6 / 1000.0;
            runtime_engine::diagnostic!(
                "sampling timing: 1000 calls, {} filtered, {:.1} rows, wait {:.1} us, issue {:.1} us, device {:.1} us",
                self.filtered,
                self.rows as f64 / 1000.0,
                mean(self.wait),
                mean(self.issue),
                mean(self.device)
            );
            *self = Self::default();
        }
    }
}

/// Debug capture of sampled logits rows, enabled by
/// `CHOPINFER_SAMPLING_CAPTURE=<file>`: rows of sampled (nonzero temperature)
/// requests are appended to the file as raw logits of the view's element
/// type, `live_width` elements each, up to `CHOPINFER_SAMPLING_CAPTURE_ROWS`
/// rows (default 2,048).
struct LogitsCapture {
    file: std::fs::File,
    remaining: usize,
}

impl LogitsCapture {
    fn from_env() -> Option<Self> {
        let path = std::env::var_os("CHOPINFER_SAMPLING_CAPTURE")?;
        let remaining = std::env::var("CHOPINFER_SAMPLING_CAPTURE_ROWS")
            .ok()
            .and_then(|rows| rows.parse().ok())
            .unwrap_or(2048);
        match std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
        {
            Ok(file) => Some(Self { file, remaining }),
            Err(error) => {
                eprintln!(
                    "sampling capture: cannot open {}: {error}",
                    path.to_string_lossy()
                );
                None
            }
        }
    }
}

/// One context's cached sampling functions, stream, and transfer arenas.
pub(crate) struct DeviceSamplingSession {
    context: Context,
    stream: Stream,
    pool: MemoryPool,
    _module: Module,
    filtered_f32: Vec<Function>,
    filtered_bf16: Vec<Function>,
    simple_partials_f32: Function,
    simple_top_partials_f32: Function,
    simple_finish_f32: Function,
    simple_partials_bf16: Function,
    simple_top_partials_bf16: Function,
    simple_finish_bf16: Function,
    greedy_partials_1024_f32: Function,
    greedy_partials_1024_bf16: Function,
    input_device: Option<DeviceBuffer>,
    input_capacity: usize,
    output_device: Option<DeviceBuffer>,
    output_capacity: usize,
    adjusted: Option<DeviceBuffer>,
    adjusted_capacity: usize,
    order: Option<DeviceBuffer>,
    order_capacity: usize,
    simple_partials: Option<DeviceBuffer>,
    simple_partials_capacity: usize,
    filter: Option<DeviceBuffer>,
    filter_capacity: usize,
    input_pinned: Option<PinnedHostBuffer>,
    input_pinned_capacity: usize,
    output_pinned: Option<PinnedHostBuffer>,
    output_pinned_capacity: usize,
    timing: Option<SamplingTiming>,
    capture: Option<LogitsCapture>,
    pending: Option<PendingSample>,
    reserved_vocabulary: Option<usize>,
}

/// A view whose kernels and output download `collect` still has to wait for.
struct PendingSample {
    rows: usize,
    live: usize,
    output_bytes: usize,
    filtered: bool,
    waited: std::time::Duration,
    issue: std::time::Duration,
    issued: std::time::Instant,
}

impl DeviceSamplingSession {
    pub(crate) fn reserve(
        &mut self,
        vocabulary: usize,
        plan: SamplingWorkspace,
    ) -> SamplingResult<()> {
        if let Some(reserved) = self.reserved_vocabulary {
            return if reserved == vocabulary {
                Ok(())
            } else {
                Err(BundleStatusV1::InvalidArgument)
            };
        }
        for (buffer, capacity, bytes) in [
            (&mut self.input_device, &mut self.input_capacity, plan.input),
            (
                &mut self.output_device,
                &mut self.output_capacity,
                plan.output,
            ),
            (
                &mut self.adjusted,
                &mut self.adjusted_capacity,
                plan.adjusted,
            ),
            (&mut self.order, &mut self.order_capacity, plan.order),
            (
                &mut self.simple_partials,
                &mut self.simple_partials_capacity,
                plan.partials,
            ),
            (&mut self.filter, &mut self.filter_capacity, plan.filter),
        ] {
            ensure_device(buffer, capacity, bytes, &self.pool, &self.stream)?;
        }
        ensure_pinned(
            &mut self.input_pinned,
            &mut self.input_pinned_capacity,
            plan.input,
            &self.context,
        )?;
        ensure_pinned(
            &mut self.output_pinned,
            &mut self.output_pinned_capacity,
            plan.output,
            &self.context,
        )?;
        // cuMemAllocAsync must be resident before the KV service queries free memory.
        self.context.synchronize().map_err(host_error)?;
        self.reserved_vocabulary = Some(vocabulary);
        Ok(())
    }

    pub(crate) fn score_prompt(
        &mut self,
        logits: &DeviceBuffer,
        view: BundleLogitsViewV1,
        rows: &[crate::executor::DevicePromptRowV1<'_>],
        dispatch: Option<SamplingDispatch>,
    ) -> SamplingResult<(Vec<SampledRow>, Vec<SampledRow>)> {
        if rows.len() != view.row_count as usize {
            return Err(BundleStatusV1::InvalidArgument);
        }
        let mut scoring_items = Vec::with_capacity(rows.len());
        for row in rows {
            if row.target.is_some() {
                let parameters = SamplingParameters::new(0.0)
                    .and_then(|settings| settings.with_top_logprobs(row.item.prompt_top_logprobs()))
                    .map_err(|_| BundleStatusV1::InvalidArgument)?;
                scoring_items.push(
                    ModelBatchItem::new(row.item.request_id(), row.item.phase(), Vec::new(), true)
                        .with_sampling(SamplingWork {
                            parameters,
                            state: SamplingState::new(0),
                            history: Default::default(),
                            emitted_tokens: 0,
                            stop_ids: Vec::new(),
                            constraint: None,
                        }),
                );
            } else {
                scoring_items.push(row.item.clone());
            }
        }
        let scoring_refs = scoring_items.iter().collect::<Vec<_>>();
        let raw_scores = self.sample_with_dispatch(logits, view, &scoring_refs, dispatch)?;
        let element_bytes = match view.element_type {
            BUNDLE_LOGITS_ELEMENT_V1_F32 => 4_usize,
            BUNDLE_LOGITS_ELEMENT_V1_BF16 => 2_usize,
            _ => return Err(BundleStatusV1::InvalidArgument),
        };
        let mut scores = Vec::with_capacity(rows.len());
        let mut generated = Vec::new();
        for (index, (row, score)) in rows.iter().zip(raw_scores).enumerate() {
            if let Some(target) = row.target {
                // Scoring rows sample without masks, so none can fail.
                if target >= view.live_width || score.failure().is_some() {
                    return Err(BundleStatusV1::InvalidArgument);
                }
                let actual_logprob = if target == score.token_id() {
                    score.logprob()
                } else {
                    let actual = self.read_logit(logits, view, index, target, element_bytes)?;
                    let chosen =
                        self.read_logit(logits, view, index, score.token_id(), element_bytes)?;
                    score.logprob() + f64::from(actual) - f64::from(chosen)
                };
                if !actual_logprob.is_finite() {
                    return Err(BundleStatusV1::HostError);
                }
                scores.push(SampledRow::new(
                    row.item.request_id(),
                    target,
                    actual_logprob,
                    score.top_logprobs().to_vec(),
                ));
            } else {
                scores.push(score.clone());
                if row.prompt && row.output {
                    generated.push(score);
                }
            }
        }
        if rows.iter().any(|row| row.target.is_some() && row.output) {
            let output_items = rows
                .iter()
                .zip(scoring_items)
                .map(|(row, scoring)| {
                    if row.target.is_some() && row.output {
                        row.item.clone()
                    } else {
                        scoring
                    }
                })
                .collect::<Vec<_>>();
            let output_refs = output_items.iter().collect::<Vec<_>>();
            for (row, sample) in rows.iter().zip(self.sample(logits, view, &output_refs)?) {
                if row.target.is_some() && row.output {
                    generated.push(sample);
                }
            }
        }
        Ok((scores, generated))
    }

    fn read_logit(
        &self,
        logits: &DeviceBuffer,
        view: BundleLogitsViewV1,
        row: usize,
        token: u32,
        element_bytes: usize,
    ) -> SamplingResult<f32> {
        let offset = row
            .checked_mul(
                usize::try_from(view.row_stride_elements)
                    .map_err(|_| BundleStatusV1::InvalidArgument)?,
            )
            .and_then(|base| base.checked_add(token as usize))
            .and_then(|index| index.checked_mul(element_bytes))
            .and_then(|bytes| bytes.checked_add(usize::try_from(view.buffer.offset).ok()?))
            .ok_or(BundleStatusV1::InvalidArgument)?;
        let mut pinned = PinnedHostBuffer::new(&self.context, element_bytes).map_err(host_error)?;
        logits
            .copy_to_pinned_at_async(offset, &mut pinned, 0, element_bytes, &self.stream)
            .map_err(host_error)?
            .finish();
        let bytes = pinned.as_slice();
        let value = if element_bytes == 4 {
            f32::from_le_bytes(bytes[..4].try_into().expect("four bytes allocated"))
        } else {
            let bf16 = u16::from_le_bytes(bytes[..2].try_into().expect("two bytes allocated"));
            f32::from_bits(u32::from(bf16) << 16)
        };
        if !value.is_finite() {
            return Err(BundleStatusV1::HostError);
        }
        Ok(value)
    }

    pub(crate) fn new(context: Context) -> SamplingResult<Self> {
        let stream = context.stream().map_err(host_error)?;
        let pool = context.default_memory_pool().map_err(host_error)?;
        let module = context
            .load_module(runtime_kernels::SAMPLING_CUBIN)
            .map_err(host_error)?;
        let load_all = |symbols: [&str; 6]| {
            symbols
                .iter()
                .map(|symbol| module.function(symbol).map_err(host_error))
                .collect::<SamplingResult<Vec<_>>>()
        };
        let filtered_f32 = load_all(runtime_kernels::SAMPLING_FILTERED_F32_SYMBOLS)?;
        let filtered_bf16 = load_all(runtime_kernels::SAMPLING_FILTERED_BF16_SYMBOLS)?;
        let simple_partials_f32 = module
            .function(runtime_kernels::SAMPLING_SIMPLE_PARTIALS_F32_SYMBOL)
            .map_err(host_error)?;
        let simple_top_partials_f32 = module
            .function(runtime_kernels::SAMPLING_SIMPLE_TOP_PARTIALS_F32_SYMBOL)
            .map_err(host_error)?;
        let simple_finish_f32 = module
            .function(runtime_kernels::SAMPLING_SIMPLE_FINISH_F32_SYMBOL)
            .map_err(host_error)?;
        let simple_partials_bf16 = module
            .function(runtime_kernels::SAMPLING_SIMPLE_PARTIALS_BF16_SYMBOL)
            .map_err(host_error)?;
        let simple_top_partials_bf16 = module
            .function(runtime_kernels::SAMPLING_SIMPLE_TOP_PARTIALS_BF16_SYMBOL)
            .map_err(host_error)?;
        let simple_finish_bf16 = module
            .function(runtime_kernels::SAMPLING_SIMPLE_FINISH_BF16_SYMBOL)
            .map_err(host_error)?;
        let greedy_partials_1024_f32 = module
            .function(runtime_kernels::SAMPLING_GREEDY_PARTIALS_1024_F32_SYMBOL)
            .map_err(host_error)?;
        let greedy_partials_1024_bf16 = module
            .function(runtime_kernels::SAMPLING_GREEDY_PARTIALS_1024_BF16_SYMBOL)
            .map_err(host_error)?;
        Ok(Self {
            context,
            stream,
            pool,
            _module: module,
            filtered_f32,
            filtered_bf16,
            simple_partials_f32,
            simple_top_partials_f32,
            simple_finish_f32,
            simple_partials_bf16,
            simple_top_partials_bf16,
            simple_finish_bf16,
            greedy_partials_1024_f32,
            greedy_partials_1024_bf16,
            input_device: None,
            input_capacity: 0,
            output_device: None,
            output_capacity: 0,
            adjusted: None,
            adjusted_capacity: 0,
            order: None,
            order_capacity: 0,
            simple_partials: None,
            simple_partials_capacity: 0,
            filter: None,
            filter_capacity: 0,
            input_pinned: None,
            input_pinned_capacity: 0,
            output_pinned: None,
            output_pinned_capacity: 0,
            timing: std::env::var_os("CHOPINFER_SAMPLING_TIMING")
                .map(|_| SamplingTiming::default()),
            capture: LogitsCapture::from_env(),
            pending: None,
            reserved_vocabulary: None,
        })
    }

    /// Appends the sampled rows of a ready logits view to the capture file.
    fn capture_rows(
        &mut self,
        logits: &DeviceBuffer,
        offset: usize,
        stride_bytes: usize,
        row_bytes: usize,
        items: &[&ModelBatchItem],
    ) -> SamplingResult<()> {
        use std::io::Write;
        let mut pinned = PinnedHostBuffer::new(&self.context, row_bytes).map_err(host_error)?;
        let capture = self.capture.as_mut().expect("capture is enabled");
        for (row, item) in items.iter().enumerate() {
            let sampled = item
                .sampling()
                .is_some_and(|work| work.parameters.temperature() != 0.0);
            if capture.remaining == 0 || !sampled {
                continue;
            }
            logits
                .copy_to_pinned_at_async(
                    offset + row * stride_bytes,
                    &mut pinned,
                    0,
                    row_bytes,
                    &self.stream,
                )
                .map_err(host_error)?
                .finish();
            if capture
                .file
                .write_all(&pinned.as_slice()[..row_bytes])
                .is_err()
            {
                return Err(BundleStatusV1::HostError);
            }
            capture.remaining -= 1;
        }
        Ok(())
    }

    /// Starts sampling one view: waits for the view's context, then uploads
    /// the rows' metadata and launches the kernels without waiting for them.
    /// `collect` returns the rows.
    pub(crate) fn issue(
        &mut self,
        logits: &DeviceBuffer,
        view: BundleLogitsViewV1,
        items: &[&ModelBatchItem],
    ) -> SamplingResult<()> {
        self.issue_with_dispatch(logits, view, items, None)
    }

    pub(crate) fn issue_with_dispatch(
        &mut self,
        logits: &DeviceBuffer,
        view: BundleLogitsViewV1,
        items: &[&ModelBatchItem],
        dispatch: Option<SamplingDispatch>,
    ) -> SamplingResult<()> {
        let _profile =
            runtime_engine::profile::PhaseTimer::new(runtime_engine::profile::StepPhase::Sampling);
        self.retire_pending()?;
        let rows = view.row_count as usize;
        let live = view.live_width as usize;
        if rows == 0
            || rows > MAX_VIEW_ROWS
            || (self.reserved_vocabulary.is_some() && rows > SAMPLER_VIEW_ROWS)
            || rows != items.len()
            || live == 0
            || live > 1 << 20
        {
            return Err(BundleStatusV1::InvalidArgument);
        }
        let element_bytes = match view.element_type {
            BUNDLE_LOGITS_ELEMENT_V1_F32 => 4_usize,
            BUNDLE_LOGITS_ELEMENT_V1_BF16 => 2_usize,
            _ => return Err(BundleStatusV1::InvalidArgument),
        };
        let stride = usize::try_from(view.row_stride_elements)
            .map_err(|_| BundleStatusV1::InvalidArgument)?;
        let required = (rows - 1)
            .checked_mul(stride)
            .and_then(|value| value.checked_add(live))
            .and_then(|value| value.checked_mul(element_bytes))
            .ok_or(BundleStatusV1::InvalidArgument)?;
        if required as u64 > view.buffer.length {
            return Err(BundleStatusV1::InvalidArgument);
        }
        let logit_offset =
            usize::try_from(view.buffer.offset).map_err(|_| BundleStatusV1::InvalidArgument)?;
        logits
            .device_address_at(logit_offset, required)
            .map_err(host_error)?;
        let started = std::time::Instant::now();
        let mut encoded = encode_batch(items, view.live_width)?;
        let dispatch = dispatch.unwrap_or(SamplingDispatch::for_items(items)?);
        if dispatch.rows < view.row_count
            || (dispatch.simple && !encoded.simple)
            || (dispatch.greedy && !encoded.greedy)
            || (!dispatch.top && encoded.top)
        {
            return Err(BundleStatusV1::InvalidArgument);
        }
        encoded.simple = dispatch.simple;
        encoded.top = dispatch.top;
        encoded.greedy = dispatch.greedy;
        if self.reserved_vocabulary.is_some_and(|width| width != live)
            || (self.reserved_vocabulary.is_some() && encoded.bytes.len() > self.input_capacity)
        {
            return Err(BundleStatusV1::InvalidArgument);
        }

        let output_bytes = rows
            .checked_mul(OUTPUT_BYTES)
            .ok_or(BundleStatusV1::ResourceExhausted)?;
        let workspace_items = if encoded.simple {
            1
        } else {
            rows.checked_mul(live)
                .ok_or(BundleStatusV1::ResourceExhausted)?
        };
        let simple_tiles = live.div_ceil(runtime_kernels::SAMPLING_SIMPLE_TILE);
        let simple_bytes = rows
            .checked_mul(simple_tiles)
            .and_then(|items| items.checked_mul(runtime_kernels::SAMPLING_SIMPLE_PARTIAL_BYTES))
            .ok_or(BundleStatusV1::ResourceExhausted)?;
        ensure_device(
            &mut self.input_device,
            &mut self.input_capacity,
            encoded.bytes.len(),
            &self.pool,
            &self.stream,
        )?;
        ensure_pinned(
            &mut self.input_pinned,
            &mut self.input_pinned_capacity,
            encoded.bytes.len(),
            &self.context,
        )?;
        let input_pinned = self.input_pinned.as_mut().expect("pinned input exists");
        input_pinned.as_mut_slice()[..encoded.bytes.len()].copy_from_slice(&encoded.bytes);
        // The upload overlaps the wait for the logits; the stream is idle
        // once the context is.
        let upload_timer =
            runtime_engine::profile::PhaseTimer::new(runtime_engine::profile::StepPhase::Upload);
        let upload = self
            .input_device
            .as_ref()
            .expect("input allocation exists")
            .copy_from_pinned_at_async(0, input_pinned, 0, encoded.bytes.len(), &self.stream)
            .map_err(host_error)?;
        drop(upload_timer);
        crate::overlap::device_wait();
        let waiting = std::time::Instant::now();
        {
            let _wait = runtime_engine::profile::PhaseTimer::new(
                runtime_engine::profile::StepPhase::DeviceWait,
            );
            self.context.synchronize().map_err(host_error)?;
        }
        let waited = waiting.elapsed();
        upload.finish();
        if self.capture.is_some() {
            self.capture_rows(
                logits,
                logit_offset,
                stride * element_bytes,
                live * element_bytes,
                items,
            )?;
        }
        ensure_device(
            &mut self.output_device,
            &mut self.output_capacity,
            output_bytes,
            &self.pool,
            &self.stream,
        )?;
        ensure_device(
            &mut self.adjusted,
            &mut self.adjusted_capacity,
            workspace_items * 8,
            &self.pool,
            &self.stream,
        )?;
        ensure_device(
            &mut self.order,
            &mut self.order_capacity,
            workspace_items * 4,
            &self.pool,
            &self.stream,
        )?;
        ensure_device(
            &mut self.simple_partials,
            &mut self.simple_partials_capacity,
            simple_bytes,
            &self.pool,
            &self.stream,
        )?;
        if !encoded.simple {
            ensure_device(
                &mut self.filter,
                &mut self.filter_capacity,
                rows * runtime_kernels::SAMPLING_FILTER_ROW_BYTES,
                &self.pool,
                &self.stream,
            )?;
        }
        ensure_pinned(
            &mut self.output_pinned,
            &mut self.output_pinned_capacity,
            output_bytes,
            &self.context,
        )?;
        let input = self.input_device.as_ref().expect("input allocation exists");
        let output = self
            .output_device
            .as_ref()
            .expect("output allocation exists");
        let adjusted = self.adjusted.as_ref().expect("workspace allocation exists");
        let order = self.order.as_ref().expect("order allocation exists");
        let partials = self
            .simple_partials
            .as_ref()
            .expect("simple partial allocation exists");
        if encoded.simple {
            let common = [
                KernelArgument::Range {
                    buffer: logits,
                    offset: logit_offset,
                    byte_len: required,
                },
                KernelArgument::U64(view.row_stride_elements),
                KernelArgument::U32(view.live_width),
                KernelArgument::U32(view.row_count),
                KernelArgument::Range {
                    buffer: input,
                    offset: 0,
                    byte_len: rows * META_BYTES,
                },
                KernelArgument::Buffer(partials),
            ];
            let (partial_function, finish_function) =
                match (view.element_type, encoded.top, encoded.greedy) {
                    (BUNDLE_LOGITS_ELEMENT_V1_F32, _, true) => {
                        (&self.greedy_partials_1024_f32, &self.simple_finish_f32)
                    }
                    (BUNDLE_LOGITS_ELEMENT_V1_BF16, _, true) => {
                        (&self.greedy_partials_1024_bf16, &self.simple_finish_bf16)
                    }
                    (BUNDLE_LOGITS_ELEMENT_V1_F32, false, _) => {
                        (&self.simple_partials_f32, &self.simple_finish_f32)
                    }
                    (BUNDLE_LOGITS_ELEMENT_V1_F32, true, _) => {
                        (&self.simple_top_partials_f32, &self.simple_finish_f32)
                    }
                    (_, false, _) => (&self.simple_partials_bf16, &self.simple_finish_bf16),
                    (_, true, _) => (&self.simple_top_partials_bf16, &self.simple_finish_bf16),
                };
            partial_function
                .launch(
                    &self.stream,
                    LaunchConfig {
                        grid: (simple_tiles as u32, view.row_count, 1),
                        block: (dispatch.threads(), 1, 1),
                        shared_memory_bytes: 0,
                        cluster: LaunchConfig::NO_CLUSTER,
                        programmatic_stream_serialization: true,
                    },
                    &common,
                )
                .map_err(host_error)?;
            let finish_args = [
                KernelArgument::Range {
                    buffer: logits,
                    offset: logit_offset,
                    byte_len: required,
                },
                KernelArgument::U64(view.row_stride_elements),
                KernelArgument::U32(view.live_width),
                KernelArgument::U32(view.row_count),
                KernelArgument::Range {
                    buffer: input,
                    offset: 0,
                    byte_len: rows * META_BYTES,
                },
                KernelArgument::Buffer(partials),
                KernelArgument::Buffer(output),
            ];
            finish_function
                .launch(
                    &self.stream,
                    LaunchConfig {
                        grid: (view.row_count, 1, 1),
                        block: (32, 1, 1),
                        shared_memory_bytes: 0,
                        cluster: LaunchConfig::NO_CLUSTER,
                        programmatic_stream_serialization: true,
                    },
                    &finish_args,
                )
                .map_err(host_error)?;
        } else {
            let filter = self.filter.as_ref().expect("filter allocation exists");
            let tile_args = [
                KernelArgument::Range {
                    buffer: logits,
                    offset: logit_offset,
                    byte_len: required,
                },
                KernelArgument::U64(view.row_stride_elements),
                KernelArgument::U32(view.live_width),
                KernelArgument::U32(view.row_count),
                KernelArgument::Range {
                    buffer: input,
                    offset: 0,
                    byte_len: rows * META_BYTES,
                },
                KernelArgument::Range {
                    buffer: input,
                    offset: encoded.counts_at,
                    byte_len: encoded.counts_bytes.max(1),
                },
                KernelArgument::Range {
                    buffer: input,
                    offset: encoded.biases_at,
                    byte_len: encoded.biases_bytes.max(1),
                },
                KernelArgument::Range {
                    buffer: input,
                    offset: encoded.eos_at,
                    byte_len: encoded.eos_bytes.max(1),
                },
                KernelArgument::Buffer(partials),
                KernelArgument::Buffer(filter),
                KernelArgument::Buffer(adjusted),
                KernelArgument::Buffer(order),
                if encoded.mask_words == 0 {
                    KernelArgument::NullBuffer
                } else {
                    KernelArgument::Range {
                        buffer: input,
                        offset: encoded.masks_at,
                        byte_len: rows * encoded.mask_words * 4,
                    }
                },
                KernelArgument::U32(checked_u32(encoded.mask_words)?),
            ];
            let finish_args = [
                KernelArgument::Range {
                    buffer: logits,
                    offset: logit_offset,
                    byte_len: required,
                },
                KernelArgument::U64(view.row_stride_elements),
                KernelArgument::U32(view.live_width),
                KernelArgument::U32(view.row_count),
                KernelArgument::Range {
                    buffer: input,
                    offset: 0,
                    byte_len: rows * META_BYTES,
                },
                KernelArgument::Buffer(partials),
                KernelArgument::Buffer(output),
            ];
            let functions = if view.element_type == BUNDLE_LOGITS_ELEMENT_V1_F32 {
                &self.filtered_f32
            } else {
                &self.filtered_bf16
            };
            let tiles = simple_tiles as u32;
            for (stage, function) in functions.iter().enumerate() {
                let finish = stage == functions.len() - 1;
                let grid = match stage {
                    1 => (
                        runtime_kernels::sampling_filtered_histogram_groups(tiles, dispatch.rows),
                        view.row_count,
                        1,
                    ),
                    3 | 5 => (view.row_count, 1, 1),
                    _ => (tiles, view.row_count, 1),
                };
                function
                    .launch(
                        &self.stream,
                        LaunchConfig {
                            grid,
                            block: (runtime_kernels::SAMPLING_FILTERED_THREADS[stage], 1, 1),
                            shared_memory_bytes: 0,
                            cluster: LaunchConfig::NO_CLUSTER,
                            programmatic_stream_serialization: true,
                        },
                        if finish { &finish_args } else { &tile_args },
                    )
                    .map_err(host_error)?;
            }
        }
        self.pending = Some(PendingSample {
            rows,
            live,
            output_bytes,
            filtered: !encoded.simple,
            waited,
            issue: started.elapsed() - waited,
            issued: std::time::Instant::now(),
        });
        Ok(())
    }

    /// Waits for the view `issue` started last and decodes its rows; `items`
    /// are the rows `issue` received.
    pub(crate) fn collect(&mut self, items: &[&ModelBatchItem]) -> SamplingResult<Vec<SampledRow>> {
        let _profile =
            runtime_engine::profile::PhaseTimer::new(runtime_engine::profile::StepPhase::Sampling);
        let pending = self.pending.take().ok_or(BundleStatusV1::InvalidArgument)?;
        if pending.rows != items.len() {
            self.stream.synchronize().map_err(host_error)?;
            return Err(BundleStatusV1::InvalidArgument);
        }
        let output = self
            .output_device
            .as_ref()
            .expect("output allocation exists");
        let output_pinned = self.output_pinned.as_mut().expect("pinned output exists");
        let download = output
            .copy_to_pinned_at_async(0, output_pinned, 0, pending.output_bytes, &self.stream)
            .map_err(host_error)?;
        {
            crate::overlap::device_wait();
            let _wait = runtime_engine::profile::PhaseTimer::new(
                runtime_engine::profile::StepPhase::DeviceWait,
            );
            download.finish();
        }
        if let Some(timing) = &mut self.timing {
            timing.record(
                pending.rows,
                pending.filtered,
                pending.waited,
                pending.issue,
                pending.issued.elapsed(),
            );
        }
        decode_output(
            &output_pinned.as_slice()[..pending.output_bytes],
            items,
            pending.live,
        )
    }

    /// Samples one view: `issue`, then `collect`.
    pub(crate) fn sample(
        &mut self,
        logits: &DeviceBuffer,
        view: BundleLogitsViewV1,
        items: &[&ModelBatchItem],
    ) -> SamplingResult<Vec<SampledRow>> {
        self.issue(logits, view, items)?;
        self.collect(items)
    }

    pub(crate) fn sample_with_dispatch(
        &mut self,
        logits: &DeviceBuffer,
        view: BundleLogitsViewV1,
        items: &[&ModelBatchItem],
        dispatch: Option<SamplingDispatch>,
    ) -> SamplingResult<Vec<SampledRow>> {
        self.issue_with_dispatch(logits, view, items, dispatch)?;
        self.collect(items)
    }

    /// Drops a view that was issued but never collected, once its kernels
    /// no longer use the session's buffers.
    fn retire_pending(&mut self) -> SamplingResult<()> {
        if self.pending.take().is_some() {
            self.stream.synchronize().map_err(host_error)?;
        }
        Ok(())
    }
}

impl Drop for DeviceSamplingSession {
    fn drop(&mut self) {
        let _ = self.retire_pending();
    }
}

/// The sampler's row status when no token survives the row's masks
/// (`cuda/sampling/sampling.cuh`).
const EMPTY_DISTRIBUTION: u32 = 2;

fn decode_output(
    bytes: &[u8],
    items: &[&ModelBatchItem],
    live: usize,
) -> SamplingResult<Vec<SampledRow>> {
    let mut sampled = Vec::with_capacity(items.len());
    for (row, item) in items.iter().enumerate() {
        let record = &bytes[row * OUTPUT_BYTES..(row + 1) * OUTPUT_BYTES];
        match read_u32(record, 4) {
            0 => {}
            // No token survives the row's masks: the request's own failure,
            // as host sampling reports it. The engine fails that request
            // and commits the step's others.
            EMPTY_DISTRIBUTION => {
                sampled.push(SampledRow::failed(
                    item.request_id(),
                    runtime_sampling::SamplingError::AllTokensMasked,
                ));
                continue;
            }
            // A non-finite logit or malformed row parameters.
            _ => return Err(BundleStatusV1::HostError),
        }
        let work = item.sampling().ok_or(BundleStatusV1::InvalidArgument)?;
        // The kernel ranks admitted tokens; fill the masked tail on the host.
        let admitted = row_mask(item).map_or(live, |mask| {
            mask.iter().map(|word| word.count_ones() as usize).sum()
        });
        let top_count = work.parameters.top_logprobs().min(live).min(admitted);
        if top_count > 20 {
            return Err(BundleStatusV1::InvalidArgument);
        }
        let mut top = Vec::with_capacity(top_count);
        for index in 0..top_count {
            top.push(TokenLogprob::new(
                read_u32(record, 16 + index * 4),
                read_f64(record, 96 + index * 8),
            ));
        }
        if let Some(mask) = row_mask(item) {
            let remaining = work.parameters.top_logprobs().min(live) - top.len();
            top.extend(
                (0..live)
                    .filter(|token| mask[token / 32] & (1 << (token % 32)) == 0)
                    .take(remaining)
                    .map(|token| TokenLogprob::new(token as u32, f64::NEG_INFINITY)),
            );
        }
        sampled.push(SampledRow::new(
            item.request_id(),
            read_u32(record, 0),
            read_f64(record, 8),
            top,
        ));
    }
    Ok(sampled)
}

#[cfg(test)]
pub(crate) fn validate_trace_metadata_for_test(
    items: &[&ModelBatchItem],
    width: u32,
) -> SamplingResult<()> {
    encode_batch(items, width).map(|_| ())
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use runtime_engine::{ModelBatchItem, RequestId, RowConstraint, SamplingWork};
    use runtime_grammar::{
        CompiledGrammar, ConstraintState, TokenVocabulary, VocabularyToken, regex_grammar,
    };
    use runtime_sampling::{LogitBias, SampleContext, SamplingParameters, SamplingState};
    use runtime_scheduler::BatchPhase;

    use super::{DeviceSamplingSession, META_BYTES, encode_batch, read_u32};

    /// A row constrained to `pattern` over a vocabulary of single bytes whose
    /// last token is a stop token.
    fn constrained(pattern: &str, width: usize) -> RowConstraint {
        let tokens = (0..width)
            .map(|token| VocabularyToken {
                bytes: if token + 1 == width {
                    Vec::new()
                } else {
                    vec![token as u8]
                },
                special: false,
            })
            .collect();
        let vocabulary = Arc::new(TokenVocabulary::new(width, tokens).unwrap());
        let grammar = regex_grammar(pattern).unwrap();
        let grammar = Arc::new(CompiledGrammar::new(&grammar, vocabulary).unwrap());
        RowConstraint::new(ConstraintState::new(grammar, None, vec![width as u32 - 1]))
    }

    #[test]
    fn singleton_device_output_fills_masked_top_logprob_tail() {
        use super::{OUTPUT_BYTES, TokenLogprob, decode_output, write_f64, write_u32};
        let item = ModelBatchItem::new(RequestId::new(1), BatchPhase::Decode, vec![1], true)
            .with_sampling(SamplingWork {
                parameters: SamplingParameters::new(0.0)
                    .unwrap()
                    .with_top_logprobs(3)
                    .unwrap(),
                state: SamplingState::new(7),
                history: vec![1].into(),
                emitted_tokens: 0,
                stop_ids: vec![127],
                constraint: Some(constrained("a", 128)),
            });
        let mut record = vec![0; OUTPUT_BYTES];
        write_u32(&mut record, 0, 97);
        write_u32(&mut record, 16, 97);
        // The kernel writes only admitted entries. Tail bytes are not results.
        write_f64(&mut record, 104, f64::NAN);
        write_f64(&mut record, 112, f64::NAN);
        let rows = decode_output(&record, &[&item], 128).unwrap();
        let top = rows[0].top_logprobs();
        assert_eq!(top.len(), 3);
        assert_eq!(top[0], TokenLogprob::new(97, 0.0));
        assert_eq!(top[1], TokenLogprob::new(0, f64::NEG_INFINITY));
        assert_eq!(top[2], TokenLogprob::new(1, f64::NEG_INFINITY));
    }

    /// A row where no token survives its masks (status 2) fails only its
    /// request: the view's other rows decode, and the failed row carries
    /// the host sampler's error for the same case. A non-finite logit
    /// (status 1) or malformed row parameters (status 3) still fail the view.
    #[test]
    fn an_empty_distribution_fails_only_its_row() {
        use super::{OUTPUT_BYTES, decode_output, write_f64, write_u32};
        use runtime_bundle_abi::BundleStatusV1;
        use runtime_engine::SampledRow;
        let item = |id| {
            ModelBatchItem::new(RequestId::new(id), BatchPhase::Decode, vec![1], true)
                .with_sampling(SamplingWork {
                    parameters: SamplingParameters::new(0.0).unwrap(),
                    state: SamplingState::new(id),
                    history: vec![1].into(),
                    emitted_tokens: 0,
                    stop_ids: Vec::new(),
                    constraint: None,
                })
        };
        let items = [item(1), item(2), item(3)];
        let refs = items.iter().collect::<Vec<_>>();
        let records = |status: u32| {
            let mut bytes = vec![0; 3 * OUTPUT_BYTES];
            for row in 0..3 {
                let record = &mut bytes[row * OUTPUT_BYTES..(row + 1) * OUTPUT_BYTES];
                write_u32(record, 0, 5 + row as u32);
                write_f64(record, 8, -0.5);
                write_u32(record, 4, if row == 1 { status } else { 0 });
            }
            bytes
        };
        let rows = decode_output(&records(2), &refs, 128).expect("a request-local failure");
        assert_eq!(rows.len(), 3);
        assert_eq!(
            rows.iter().map(SampledRow::request_id).collect::<Vec<_>>(),
            [1, 2, 3].map(RequestId::new)
        );
        assert_eq!(
            rows[1].failure(),
            Some(runtime_sampling::SamplingError::AllTokensMasked)
        );
        for row in [&rows[0], &rows[2]] {
            assert_eq!(row.failure(), None);
            assert_eq!(row.logprob(), -0.5);
        }
        assert_eq!((rows[0].token_id(), rows[2].token_id()), (5, 7));
        for status in [1, 3] {
            assert_eq!(
                decode_output(&records(status), &refs, 128),
                Err(BundleStatusV1::HostError)
            );
        }
    }

    #[test]
    fn minimum_stop_metadata_covers_all_ids_in_gathered_and_owned_partitions() {
        let items = (0..16)
            .map(|row| {
                let work = SamplingWork {
                    parameters: SamplingParameters::new(0.0)
                        .unwrap()
                        .with_minimum_new_tokens(3)
                        .unwrap(),
                    state: SamplingState::new(0),
                    history: vec![3; row % 4 + 1].into(),
                    emitted_tokens: row % 4,
                    stop_ids: vec![1, 2],
                    constraint: None,
                };
                ModelBatchItem::new(
                    RequestId::new(row as u64 + 1),
                    BatchPhase::Decode,
                    vec![3],
                    true,
                )
                .with_sampling(work)
            })
            .collect::<Vec<_>>();
        for partition in [1, 4, 16] {
            for chunk in items.chunks(partition) {
                let encoded = encode_batch(&chunk.iter().collect::<Vec<_>>(), 4).unwrap();
                let needs_mask = chunk
                    .iter()
                    .any(|item| item.sampling().unwrap().emitted_tokens < 3);
                assert_eq!(encoded.simple, !needs_mask);
                for (row, item) in chunk.iter().enumerate() {
                    let meta = &encoded.bytes[row * META_BYTES..(row + 1) * META_BYTES];
                    let suppressed = item.sampling().unwrap().emitted_tokens < 3;
                    assert_eq!(read_u32(meta, 60), u32::from(suppressed));
                    assert_eq!(read_u32(meta, 48), if suppressed { 2 } else { 0 });
                    if suppressed {
                        let offset = u64::from_le_bytes(meta[32..40].try_into().unwrap()) as usize;
                        assert_eq!(read_u32(&encoded.bytes, encoded.eos_at + offset * 4), 1);
                        assert_eq!(
                            read_u32(&encoded.bytes, encoded.eos_at + (offset + 1) * 4),
                            2
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn masks_follow_the_metadata_and_select_the_filtered_kernels() {
        let width = 128;
        let plain = SamplingWork {
            parameters: SamplingParameters::new(0.0).unwrap(),
            state: SamplingState::new(3),
            history: vec![1].into(),
            emitted_tokens: 0,
            stop_ids: vec![127],
            constraint: None,
        };
        let masked = SamplingWork {
            parameters: SamplingParameters::new(0.0)
                .unwrap()
                .with_minimum_new_tokens(4)
                .unwrap(),
            constraint: Some(constrained("a|", width)),
            ..plain.clone()
        };
        let items = [
            ModelBatchItem::new(RequestId::new(1), BatchPhase::Decode, vec![1], true)
                .with_sampling(plain),
            ModelBatchItem::new(RequestId::new(2), BatchPhase::Decode, vec![1], true)
                .with_sampling(masked),
        ];
        let encoded = encode_batch(&items.iter().collect::<Vec<_>>(), width as u32).unwrap();
        assert!(!encoded.simple);
        assert_eq!(encoded.mask_words, 4);
        assert_eq!(encoded.masks_at % 16, 0);
        let word = |row: usize, index: usize| {
            read_u32(&encoded.bytes, encoded.masks_at + (row * 4 + index) * 4)
        };
        assert!((0..4).all(|index| word(0, index) == u32::MAX));
        // "a" (97) and the stop token (127) are admitted.
        assert_eq!(word(1, 3), (1 << 1) | (1 << 31));
        assert_eq!(word(1, 0) | word(1, 1) | word(1, 2), 0);
        // The minimum length still suppresses the stop token: "a" is admitted.
        assert_eq!(read_u32(&encoded.bytes, META_BYTES + 60), 1);
        let only_stop = SamplingWork {
            parameters: SamplingParameters::new(0.0)
                .unwrap()
                .with_minimum_new_tokens(4)
                .unwrap(),
            constraint: Some(constrained("", width)),
            ..items[0].sampling().unwrap().clone()
        };
        let item = ModelBatchItem::new(RequestId::new(3), BatchPhase::Decode, vec![1], true)
            .with_sampling(only_stop);
        let encoded = encode_batch(&[&item], width as u32).unwrap();
        assert_eq!(read_u32(&encoded.bytes, 60), 0);
        // Without masks nothing is appended and simple rows stay simple.
        let encoded = encode_batch(&[&items[0]], width as u32).unwrap();
        assert!(encoded.simple);
        assert_eq!(encoded.mask_words, 0);
    }

    #[test]
    fn compact_metadata_counts_prompt_and_output_separately() {
        let item = ModelBatchItem::new(RequestId::new(1), BatchPhase::Decode, vec![4], true)
            .with_sampling(SamplingWork {
                parameters: SamplingParameters::new(0.7)
                    .unwrap()
                    .with_penalties(0.5, 0.25, 1.2)
                    .unwrap(),
                state: SamplingState::new(17),
                history: vec![4, 8, 8].into(),
                emitted_tokens: 1,
                stop_ids: Vec::new(),
                constraint: None,
            });
        let encoded = encode_batch(&[&item], 16).unwrap();
        assert!(!encoded.simple);
        assert_eq!(encoded.counts_bytes, 2 * 16);
        assert_eq!(
            encoded.bytes[encoded.counts_at..encoded.counts_at + 4],
            4_u32.to_le_bytes()
        );
        assert_eq!(read_u32(&encoded.bytes, 40), 2);
        assert_eq!(encoded.counts_at, META_BYTES);
        assert_eq!(read_u32(&encoded.bytes, encoded.counts_at + 16 + 8), 1);
    }

    #[test]
    fn verify_row_penalty_counts_match_independent_row_encoding() {
        let work = SamplingWork {
            parameters: SamplingParameters::new(0.7)
                .unwrap()
                .with_penalties(0.5, 0.25, 1.2)
                .unwrap(),
            state: SamplingState::new(17),
            history: vec![4, 8, 8].into(),
            emitted_tokens: 1,
            stop_ids: Vec::new(),
            constraint: None,
        };
        let rows = (0..3)
            .map(|row| {
                ModelBatchItem::new(RequestId::new(1), BatchPhase::Decode, vec![8], true)
                    .with_sampling(work.for_verify_row(&[8, 5], row))
            })
            .collect::<Vec<_>>();
        let together = encode_batch(&rows.iter().collect::<Vec<_>>(), 16).unwrap();
        for (row, item) in rows.iter().enumerate() {
            let alone = encode_batch(&[item], 16).unwrap();
            let record = &together.bytes[row * META_BYTES..(row + 1) * META_BYTES];
            let offset = u64::from_le_bytes(record[16..24].try_into().unwrap()) as usize * 16;
            let count = read_u32(record, 40) as usize * 16;
            assert_eq!(
                &together.bytes[together.counts_at + offset..together.counts_at + offset + count],
                &alone.bytes[alone.counts_at..alone.counts_at + alone.counts_bytes],
            );
        }
    }

    #[test]
    fn greedy_requests_ignore_sampling_filters_for_the_fast_path() {
        let filtered = |temperature| {
            SamplingParameters::new(temperature)
                .unwrap()
                .with_top_k(40)
                .unwrap()
                .with_top_p(0.95)
                .unwrap()
                .with_min_p(0.05)
                .unwrap()
        };
        for (temperature, simple) in [(0.0, true), (0.8, false)] {
            let item = ModelBatchItem::new(RequestId::new(1), BatchPhase::Decode, vec![7], true)
                .with_sampling(SamplingWork {
                    parameters: filtered(temperature),
                    state: SamplingState::new(61),
                    history: Default::default(),
                    emitted_tokens: 0,
                    stop_ids: Vec::new(),
                    constraint: None,
                });
            let encoded = encode_batch(&[&item], 32).unwrap();
            assert_eq!(encoded.simple, simple);
            assert_eq!(encoded.greedy, temperature == 0.0);
        }
    }

    #[test]
    fn greedy_fast_path_requires_at_most_one_top_logprob() {
        for top in [0, 1, 2, 20] {
            for temperature in [0.0, 0.8] {
                let item =
                    ModelBatchItem::new(RequestId::new(1), BatchPhase::Decode, vec![7], true)
                        .with_sampling(SamplingWork {
                            parameters: SamplingParameters::new(temperature)
                                .unwrap()
                                .with_top_logprobs(top)
                                .unwrap(),
                            state: SamplingState::new(61),
                            history: Default::default(),
                            emitted_tokens: 0,
                            stop_ids: Vec::new(),
                            constraint: None,
                        });
                let encoded = encode_batch(&[&item], 32).unwrap();
                assert!(encoded.simple);
                assert_eq!(encoded.greedy, temperature == 0.0 && top <= 1);
            }
        }
    }

    #[cfg(feature = "cuda-fake-driver-tests")]
    #[test]
    fn reserved_sampler_capacities_cover_filtered_and_simple_buffers() {
        use crate::sampling_workspace::{SAMPLER_VIEW_ROWS, SamplingWorkspace};
        runtime_cuda::fake_driver::reset();
        let driver = runtime_cuda::Driver::initialize().unwrap();
        let context = driver.retain_primary_context(0).unwrap();
        let mut sampler = DeviceSamplingSession::new(context).unwrap();
        let plan = SamplingWorkspace::new(33).unwrap();
        sampler.reserve(33, plan).unwrap();
        assert_eq!(sampler.input_capacity, plan.input);
        assert_eq!(sampler.input_pinned_capacity, plan.input);
        assert_eq!(sampler.output_capacity, plan.output);
        assert_eq!(sampler.output_pinned_capacity, plan.output);
        assert_eq!(sampler.adjusted_capacity, plan.adjusted);
        assert_eq!(sampler.order_capacity, plan.order);
        assert_eq!(sampler.simple_partials_capacity, plan.partials);
        assert_eq!(sampler.filter_capacity, plan.filter);
        let address = sampler
            .adjusted
            .as_ref()
            .unwrap()
            .device_address_at(0, plan.adjusted)
            .unwrap();
        sampler.reserve(33, plan).unwrap();
        assert_eq!(
            sampler
                .adjusted
                .as_ref()
                .unwrap()
                .device_address_at(0, plan.adjusted)
                .unwrap(),
            address
        );
        let work = ModelBatchItem::new(
            RequestId::new(1),
            runtime_scheduler::BatchPhase::Decode,
            vec![0],
            true,
        )
        .with_sampling(SamplingWork {
            parameters: SamplingParameters::new(0.8)
                .unwrap()
                .with_penalties(0.5, 0.5, 1.2)
                .unwrap()
                .with_logit_bias(
                    (0..33)
                        .map(|id| runtime_sampling::LogitBias::new(id, 1.0).unwrap())
                        .collect(),
                )
                .unwrap()
                .with_minimum_new_tokens(10)
                .unwrap(),
            state: SamplingState::new(1),
            history: (0..33).collect::<Vec<_>>().into(),
            emitted_tokens: 0,
            stop_ids: (0..33).collect(),
            constraint: None,
        });
        let encoded = encode_batch(&vec![&work; SAMPLER_VIEW_ROWS], 33).unwrap();
        assert!(!encoded.simple);
        assert!(encoded.bytes.len() <= plan.input);
    }

    /// A device row whose every token is a stop token masked before the
    /// minimum length has no candidate: the filtered finish reports status
    /// 2, so the view's sample fails only that request, with the CPU
    /// sampler's error, and its other rows sample as they do alone and as
    /// the CPU sampler does.
    #[test]
    fn a_device_row_without_a_candidate_fails_only_its_request_gpu() {
        use runtime_engine::SampledRow;
        let Ok(ordinal) = std::env::var("CHOPINFER_KERNEL_GPU") else {
            return;
        };
        let ordinal: i32 = ordinal.parse().expect("GPU ordinal is an integer");
        let driver = runtime_cuda::Driver::initialize().unwrap();
        let context = driver.retain_primary_context(ordinal).unwrap();
        let stream = context.stream().unwrap();
        let pool = context.default_memory_pool().unwrap();
        let width = 128_usize;
        let logits: Vec<f32> = (0..3 * width)
            .map(|index| ((index * 17 % 71) as f32 - 35.0) / 9.0)
            .collect();
        let device = pool.allocate(&stream, logits.len() * 4).unwrap();
        device.copy_from_f32(&logits, &stream).unwrap();
        let view = |rows: usize| {
            let mut view = runtime_bundle_abi::BundleLogitsViewV1::empty();
            view.element_type = runtime_bundle_abi::BUNDLE_LOGITS_ELEMENT_V1_F32;
            view.buffer.length = (rows * width * 4) as u64;
            view.row_stride_elements = width as u64;
            view.row_count = rows as u32;
            view.live_width = width as u32;
            view
        };
        let mut sampler = DeviceSamplingSession::new(context).unwrap();
        let sampled = SamplingParameters::new(0.8)
            .unwrap()
            .with_top_p(0.9)
            .unwrap()
            .with_top_logprobs(3)
            .unwrap();
        let greedy = SamplingParameters::new(0.0)
            .unwrap()
            .with_penalties(0.3, 0.2, 1.1)
            .unwrap()
            .with_top_logprobs(2)
            .unwrap();
        let every_stop = (0..width as u32).collect::<Vec<_>>();
        let history = vec![7, 7, 13];
        let work = |parameters: &SamplingParameters, stops: &[u32], seed: u64| SamplingWork {
            parameters: parameters.clone().with_minimum_new_tokens(2).unwrap(),
            state: SamplingState::new(seed),
            history: history.clone().into(),
            emitted_tokens: 1,
            stop_ids: stops.to_vec(),
            constraint: None,
        };
        for (empty, other) in [(&sampled, &greedy), (&greedy, &sampled)] {
            let items = [
                ModelBatchItem::new(RequestId::new(1), BatchPhase::Decode, vec![7], true)
                    .with_sampling(work(other, &[3], 11)),
                ModelBatchItem::new(RequestId::new(2), BatchPhase::Decode, vec![7], true)
                    .with_sampling(work(empty, &every_stop, 12)),
                ModelBatchItem::new(RequestId::new(3), BatchPhase::Decode, vec![7], true)
                    .with_sampling(work(other, &[], 13)),
            ];
            let refs = items.iter().collect::<Vec<_>>();
            let cpu = |row: usize| {
                let work = items[row].sampling().unwrap();
                let mut state = work.state;
                runtime_sampling::sample(
                    &work.parameters,
                    &logits[row * width..(row + 1) * width],
                    SampleContext::new(&history, 1, &work.stop_ids),
                    &mut state,
                )
            };
            let rows = sampler.sample(&device, view(3), &refs).unwrap();
            assert_eq!(
                rows.iter().map(SampledRow::request_id).collect::<Vec<_>>(),
                [1, 2, 3].map(RequestId::new)
            );
            assert_eq!(
                rows[1].failure(),
                Some(runtime_sampling::SamplingError::AllTokensMasked)
            );
            for row in [0, 2] {
                let expected = cpu(row).unwrap();
                let actual = &rows[row];
                assert_eq!(actual.failure(), None);
                assert_eq!(actual.token_id(), expected.token_id());
                assert!((actual.logprob() - expected.logprob()).abs() < 1e-4);
                assert_eq!(
                    actual
                        .top_logprobs()
                        .iter()
                        .map(|top| top.token_id())
                        .collect::<Vec<_>>(),
                    expected
                        .top_logprobs()
                        .iter()
                        .map(|top| top.token_id())
                        .collect::<Vec<_>>()
                );
            }
            // The other rows' results do not depend on the failed row.
            let alone = sampler.sample(&device, view(1), &[&items[0]]).unwrap();
            assert_eq!(alone[0], rows[0]);
            assert_eq!(
                cpu(1),
                Err(runtime_sampling::SamplingError::AllTokensMasked)
            );
        }
    }

    #[test]
    fn device_sampling_round_trip_gpu() {
        let Ok(ordinal) = std::env::var("CHOPINFER_KERNEL_GPU") else {
            return;
        };
        let ordinal: i32 = ordinal.parse().expect("GPU ordinal is an integer");
        let driver = runtime_cuda::Driver::initialize().unwrap();
        let context = driver.retain_primary_context(ordinal).unwrap();
        let stream = context.stream().unwrap();
        let pool = context.default_memory_pool().unwrap();
        let width = 128_usize;
        let logits: Vec<f32> = (0..width)
            .map(|token| ((token * 17 % 71) as f32 - 35.0) / 9.0)
            .collect();
        let device = pool.allocate(&stream, width * 4).unwrap();
        device.copy_from_f32(&logits, &stream).unwrap();
        let mut view = runtime_bundle_abi::BundleLogitsViewV1::empty();
        view.element_type = runtime_bundle_abi::BUNDLE_LOGITS_ELEMENT_V1_F32;
        view.buffer.length = (width * 4) as u64;
        view.row_stride_elements = width as u64;
        view.row_count = 1;
        view.live_width = width as u32;
        let mut sampler = DeviceSamplingSession::new(context).unwrap();
        let simple = SamplingParameters::new(0.8).unwrap();
        let simple_greedy = SamplingParameters::new(0.0).unwrap();
        let simple_greedy_top = SamplingParameters::new(0.0)
            .unwrap()
            .with_top_logprobs(3)
            .unwrap();
        let filtered = SamplingParameters::new(0.8)
            .unwrap()
            .with_penalties(0.3, 0.2, 1.1)
            .unwrap()
            .with_logit_bias(vec![LogitBias::new(7, 0.5).unwrap()])
            .unwrap()
            .with_top_logprobs(3)
            .unwrap();
        let greedy_with_filters = SamplingParameters::new(0.0)
            .unwrap()
            .with_top_p(0.95)
            .unwrap()
            .with_top_k(40)
            .unwrap()
            .with_top_logprobs(3)
            .unwrap();
        let production_default = SamplingParameters::new(1.0)
            .unwrap()
            .with_top_p(0.95)
            .unwrap();
        let every_filter = SamplingParameters::new(0.7)
            .unwrap()
            .with_top_k(9)
            .unwrap()
            .with_top_p(0.8)
            .unwrap()
            .with_min_p(0.05)
            .unwrap()
            .with_penalties(0.3, 0.2, 1.1)
            .unwrap()
            .with_top_logprobs(2)
            .unwrap();
        let greedy_penalized = SamplingParameters::new(0.0)
            .unwrap()
            .with_penalties(0.0, 0.0, 3.0)
            .unwrap()
            .with_top_logprobs(1)
            .unwrap();
        for (parameters, pattern) in [
            (simple.clone(), None),
            (simple_greedy.clone(), None),
            (simple_greedy_top.clone(), None),
            (filtered.clone(), None),
            (greedy_with_filters, None),
            (production_default.clone(), None),
            (every_filter.clone(), None),
            (greedy_penalized.clone(), None),
            (simple, Some("[a-f]+")),
            (simple_greedy_top.clone(), Some("a")),
            (simple_greedy_top, Some("[0-9]")),
            (filtered, Some("x|y|z")),
            (production_default, Some("[a-z]+")),
            (every_filter, Some("[0-9a-f]")),
            (greedy_penalized, Some("x|y|z")),
        ] {
            let constraint = pattern.map(|pattern| constrained(pattern, width));
            let history = vec![7, 7, 13];
            let item = ModelBatchItem::new(RequestId::new(1), BatchPhase::Decode, vec![7], true)
                .with_sampling(SamplingWork {
                    parameters: parameters.clone(),
                    state: SamplingState::new(61),
                    history: history.clone().into(),
                    emitted_tokens: 1,
                    stop_ids: Vec::new(),
                    constraint: constraint.clone(),
                });
            let actual = sampler.sample(&device, view, &[&item]).unwrap();
            // Issue and collect return the same rows; a second issue retires
            // a view that was never collected.
            sampler.issue(&device, view, &[&item]).unwrap();
            sampler.issue(&device, view, &[&item]).unwrap();
            assert_eq!(sampler.collect(&[&item]).unwrap(), actual);
            assert!(sampler.collect(&[&item]).is_err());
            let context = SampleContext::new(&history, 1, &[]);
            let context = match constraint.as_ref().and_then(RowConstraint::mask) {
                Some(mask) => context.with_allowed_tokens(mask),
                None => context,
            };
            let expected = runtime_sampling::sample(
                &parameters,
                &logits,
                context,
                &mut SamplingState::new(61),
            )
            .unwrap();
            assert_eq!(actual[0].token_id(), expected.token_id());
            assert!((actual[0].logprob() - expected.logprob()).abs() < 1e-4);
            assert_eq!(
                actual[0].top_logprobs().len(),
                expected.top_logprobs().len()
            );
            for (got, want) in actual[0].top_logprobs().iter().zip(expected.top_logprobs()) {
                assert_eq!(got.token_id(), want.token_id());
                if want.logprob() == f64::NEG_INFINITY {
                    assert_eq!(got.logprob(), f64::NEG_INFINITY);
                } else {
                    assert!((got.logprob() - want.logprob()).abs() < 1e-4);
                }
            }
        }
    }
}
