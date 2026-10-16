//! Media inputs must reach the model exactly as if their rows were prompt
//! tokens.
//!
//! The fake executor here is a copy encoder: an encode writes its payload's
//! `u32` values into the executor's own arena rows, and an inject replaces
//! placeholder inputs with those values. Its logits depend only on the
//! resulting effective tokens, so a media request must emit exactly what a
//! text request whose prompt already holds those values emits.

use std::collections::BTreeMap;

use runtime_engine::{
    DraftSource, Engine, EngineConfig, EngineError, EngineEvent, EngineEventKind, EngineRequest,
    EngineRequestError, ExecutorError, LogitsRow, MediaConfigV1, MediaInputV1, MediaRequestError,
    MediaSpanV1, ModelBatchItem, ModelExecutor, ModelTransaction, ModelTransactionState,
    PromptLookupConfig, PromptLookupExecutor, SpeculationConfig, SpeculationMode, StepStatus,
};
use runtime_kv::{BlockId, KvGeometry, PreparedAppendBatch};
use runtime_model_api::ModelVocabulary;
use runtime_sampling::SamplingParameters;
use runtime_scheduler::{InvalidRequest, RequestId, SchedulerConfig, SubmitError};

const VOCABULARY: usize = 32;
const PLACEHOLDER: u32 = 3;
const PAGE_ROWS: u32 = 4;
const KV_BLOCKS: usize = 512;

fn vocabulary() -> ModelVocabulary {
    ModelVocabulary::new(VOCABULARY).unwrap()
}

fn mix(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn logits_for(prefix: &[u32]) -> Vec<f32> {
    let hash = prefix
        .iter()
        .fold(7u64, |hash, token| mix(hash ^ u64::from(*token)));
    let mut logits: Vec<f32> = (0..VOCABULARY as u64)
        .map(|token| (mix(hash ^ (token << 40)) >> 40) as f32 / (1u64 << 24) as f32 * 4.0 - 2.0)
        .collect();
    logits[(hash % VOCABULARY as u64) as usize] += 2.5;
    logits
}

/// The value media item `key` holds at media row `row`.
fn media_value(key: u8, row: u32) -> u32 {
    (mix(u64::from(key) * 1000 + u64::from(row)) % 1_000_000) as u32
}

fn effective(value: u32) -> u32 {
    value % VOCABULARY as u32
}

/// A copy encoder with its own arena and a record of committed inputs.
struct CopyModel {
    arena: Vec<Option<u32>>,
    kv: BTreeMap<RequestId, Vec<u32>>,
    pending: Vec<(RequestId, Vec<u32>)>,
    encodes: usize,
    fail_on_encode: bool,
    refuse_restores: bool,
    snapshots: BTreeMap<usize, Vec<u32>>,
}

impl CopyModel {
    fn new(arena_pages: usize) -> Self {
        Self {
            arena: vec![None; arena_pages * PAGE_ROWS as usize],
            kv: BTreeMap::new(),
            pending: Vec::new(),
            encodes: 0,
            fail_on_encode: false,
            refuse_restores: false,
            snapshots: BTreeMap::new(),
        }
    }
}

struct CopyState<'a> {
    model: &'a mut CopyModel,
}

impl ModelTransactionState for CopyState<'_> {
    fn commit(&mut self) -> Result<(), ExecutorError> {
        for (id, inputs) in self.model.pending.drain(..) {
            self.model.kv.entry(id).or_default().extend(inputs);
        }
        Ok(())
    }

    fn commit_kept(&mut self, kept: &[runtime_engine::KeptRows]) -> Result<(), ExecutorError> {
        for (id, inputs) in self.model.pending.drain(..) {
            let count = kept
                .iter()
                .find(|record| record.request_id == id)
                .map_or(inputs.len(), |record| record.kept_rows);
            self.model
                .kv
                .entry(id)
                .or_default()
                .extend_from_slice(&inputs[..count]);
        }
        Ok(())
    }

    fn rollback(&mut self) -> Result<(), ExecutorError> {
        self.model.pending.clear();
        Ok(())
    }
}

impl ModelExecutor for CopyModel {
    type TransactionState<'a> = CopyState<'a>;

    fn retire_request_v1(&mut self, id: RequestId) -> Result<(), ExecutorError> {
        self.kv.remove(&id);
        Ok(())
    }

    fn capture_prefix_state_v1(
        &mut self,
        id: RequestId,
        tokens: usize,
        pages: &[BlockId],
    ) -> Result<(), ExecutorError> {
        let committed = &self.kv[&id];
        assert_eq!(committed.len(), tokens);
        self.snapshots.insert(pages[0].index(), committed.clone());
        Ok(())
    }

    fn restore_prefix_state_v1(
        &mut self,
        id: RequestId,
        prefix: &[u32],
        pages: &[BlockId],
    ) -> Result<(), ExecutorError> {
        if self.refuse_restores {
            return Err(ExecutorError::new("injected restore refusal").unwrap());
        }
        let snapshot = self.snapshots[&pages[0].index()].clone();
        assert_eq!(snapshot.len(), prefix.len());
        self.kv.insert(id, snapshot);
        Ok(())
    }

    fn execute<'a>(
        &'a mut self,
        items: &[ModelBatchItem],
        kv_plan: &PreparedAppendBatch,
    ) -> Result<ModelTransaction<Self::TransactionState<'a>>, ExecutorError> {
        if self.fail_on_encode && items.iter().any(|item| !item.media_encodes().is_empty()) {
            return Err(ExecutorError::new("injected encoder failure").unwrap());
        }
        // Every encode of the step runs before any injection reads it.
        for item in items {
            for encode in item.media_encodes() {
                assert_eq!(
                    encode.pages.len(),
                    encode.output_rows.div_ceil(PAGE_ROWS) as usize
                );
                assert_eq!(encode.payload.len(), 4 * encode.output_rows as usize);
                for row in 0..encode.output_rows {
                    let at = encode.pages[(row / PAGE_ROWS) as usize] * u64::from(PAGE_ROWS)
                        + u64::from(row % PAGE_ROWS);
                    let bytes = &encode.payload[4 * row as usize..4 * row as usize + 4];
                    self.arena[at as usize] = Some(u32::from_le_bytes(bytes.try_into().unwrap()));
                }
                self.encodes += 1;
            }
        }
        let mut rows = Vec::new();
        let mut prompt_rows = Vec::new();
        for (item, append) in items.iter().zip(kv_plan.appends()) {
            let committed = self.kv.get(&item.request_id()).cloned().unwrap_or_default();
            assert_eq!(committed.len(), append.previous_tokens());
            let mut inputs = item.input_tokens().to_vec();
            for inject in item.media_injects() {
                assert!(
                    inject.arena_row % u64::from(PAGE_ROWS) + u64::from(inject.rows)
                        <= u64::from(PAGE_ROWS)
                );
                for row in 0..inject.rows {
                    let input = &mut inputs[(inject.item_row + row) as usize];
                    assert_eq!(*input, PLACEHOLDER);
                    let value = self.arena[(inject.arena_row + u64::from(row)) as usize]
                        .expect("an injected row was encoded");
                    *input = effective(value);
                }
            }
            for position in 0..item.prompt_targets().len() {
                let mut prefix = committed.clone();
                prefix.extend_from_slice(&inputs[..=position]);
                prompt_rows.push(LogitsRow::new(item.request_id(), logits_for(&prefix)).unwrap());
            }
            if item.sampling().is_some() {
                let first = if item.draft().is_some() {
                    0
                } else {
                    inputs.len() - 1
                };
                for last in first..inputs.len() {
                    let mut prefix = committed.clone();
                    prefix.extend_from_slice(&inputs[..=last]);
                    rows.push(LogitsRow::new(item.request_id(), logits_for(&prefix)).unwrap());
                }
            }
            self.pending.push((item.request_id(), inputs));
        }
        Ok(ModelTransaction::new(rows, CopyState { model: self }).with_prompt_rows(prompt_rows))
    }
}

/// One media item: its content key byte and output rows.
#[derive(Clone, Copy)]
struct Item {
    key: u8,
    rows: u32,
}

fn media_input(item: Item) -> MediaInputV1 {
    let payload: Vec<u8> = (0..item.rows)
        .flat_map(|row| media_value(item.key, row).to_le_bytes())
        .collect();
    MediaInputV1 {
        key: [item.key; 32],
        encoder: 0,
        geometry: [1, item.rows, 1, 0],
        output_rows: item.rows,
        payload: payload.into(),
    }
}

/// A prompt of text around the items, each item's rows split into two spans
/// with a text token between them. Returns the media request and the text
/// request that holds the item values in place of the placeholders.
fn requests(
    id: u64,
    items: &[Item],
    maximum: usize,
    temperature: f64,
) -> (EngineRequest, EngineRequest) {
    let text = |position: u64| (mix(id * 131 + position) % VOCABULARY as u64) as u32;
    let mut prompt = vec![text(0), text(1)];
    let mut reference = prompt.clone();
    let mut spans = Vec::new();
    for (index, item) in items.iter().enumerate() {
        let first = item.rows / 2;
        for (media_row, rows) in [(0, first), (first, item.rows - first)] {
            if rows == 0 {
                continue;
            }
            spans.push(MediaSpanV1 {
                media: index as u16,
                prompt_row: prompt.len() as u32,
                rows,
                media_row,
            });
            for row in media_row..media_row + rows {
                prompt.push(PLACEHOLDER);
                reference.push(effective(media_value(item.key, row)));
            }
            let separator = text(prompt.len() as u64);
            prompt.push(separator);
            reference.push(separator);
        }
    }
    let sampling = SamplingParameters::new(temperature).unwrap();
    let media = EngineRequest::new(
        RequestId::new(id),
        prompt,
        maximum,
        sampling.clone(),
        id,
        Vec::new(),
        vocabulary(),
    )
    .unwrap()
    .with_media(items.iter().copied().map(media_input).collect(), spans)
    .unwrap();
    let text = EngineRequest::new(
        RequestId::new(id),
        reference,
        maximum,
        sampling,
        id,
        Vec::new(),
        vocabulary(),
    )
    .unwrap();
    (media, text)
}

fn media_config(arena_pages: usize, cost_per_row_milli: u32) -> MediaConfigV1 {
    MediaConfigV1::new(arena_pages, PAGE_ROWS)
        .unwrap()
        .with_encoder(1 << 20, cost_per_row_milli)
        .unwrap()
}

fn config(budget: usize, chunk: usize, media: Option<MediaConfigV1>) -> EngineConfig {
    scored_config(budget, chunk, media, usize::MAX)
}

fn scored_config(
    budget: usize,
    chunk: usize,
    media: Option<MediaConfigV1>,
    scored_rows: usize,
) -> EngineConfig {
    let config = EngineConfig::new(
        SchedulerConfig::new(6, 64, budget, chunk, 1)
            .unwrap()
            .with_maximum_scored_rows(scored_rows)
            .unwrap(),
        KvGeometry::new(4, KV_BLOCKS).unwrap(),
        vocabulary(),
        512,
        64,
    )
    .unwrap();
    match media {
        Some(media) => config.with_media(media),
        None => config,
    }
}

type Outputs = BTreeMap<RequestId, Vec<EngineEventKind>>;

/// Submits one request every `stagger` steps and runs to idle, auditing the
/// engine, and so the arena pins, after every step.
fn run<E: ModelExecutor>(
    engine: &mut Engine<E>,
    requests: &[EngineRequest],
    stagger: usize,
) -> Outputs {
    let mut outputs = Outputs::new();
    let mut pending = requests.iter();
    for step in 0..20_000 {
        if step % stagger == 0
            && let Some(request) = pending.next()
        {
            engine.submit(request.clone()).unwrap();
        }
        let status = engine.step().unwrap();
        engine.audit().unwrap();
        for event in engine.drain_events() {
            record(&mut outputs, event);
        }
        if status == StepStatus::Idle && pending.len() == 0 {
            assert_eq!(engine.kv_stats().free_blocks(), KV_BLOCKS);
            return outputs;
        }
    }
    panic!("the engine did not finish");
}

fn record(outputs: &mut Outputs, event: EngineEvent) {
    outputs
        .entry(event.request_id())
        .or_default()
        .push(event.kind().clone());
}

fn workload(keys: &[&[Item]]) -> (Vec<EngineRequest>, Vec<EngineRequest>) {
    keys.iter()
        .enumerate()
        .map(|(index, items)| {
            requests(
                index as u64 + 1,
                items,
                6 + index % 5,
                if index % 2 == 0 { 0.0 } else { 0.9 },
            )
        })
        .unzip()
}

fn reference(requests: &[EngineRequest], stagger: usize) -> Outputs {
    let mut engine = Engine::new(config(16, 3, None), CopyModel::new(1)).unwrap();
    run(&mut engine, requests, stagger)
}

const A: Item = Item { key: 1, rows: 5 };
const B: Item = Item { key: 2, rows: 8 };
const C: Item = Item { key: 3, rows: 3 };
const D: Item = Item { key: 4, rows: 9 };
const E: Item = Item { key: 5, rows: 4 };

#[test]
fn media_requests_emit_exactly_what_their_text_equivalents_emit() {
    let (media, text) = workload(&[&[A], &[A, B], &[C], &[B, A, D], &[E], &[A], &[D, E]]);
    for stagger in [1, 2, 5] {
        let expected = reference(&text, stagger);
        for chunk in [2, 3, 8] {
            let mut engine = Engine::new(
                config(16, chunk, Some(media_config(64, 250))),
                CopyModel::new(64),
            )
            .unwrap();
            let actual = run(&mut engine, &media, stagger);
            assert_eq!(actual, expected, "stagger {stagger} chunk {chunk}");
            let metrics = engine.metrics().snapshot();
            // Five distinct keys fit the arena, so each is encoded once and
            // the other six of the eleven items are cache hits.
            assert_eq!(engine.executor().encodes, 5);
            assert_eq!(metrics.media_encodes, 5);
            assert_eq!(metrics.media_cache_hits, 6);
            assert_eq!(metrics.media_evictions, 0);
            assert!(
                metrics
                    .prometheus()
                    .contains("chop_media_cache_hits_total 6")
            );
        }
    }
}

#[test]
fn a_small_arena_evicts_unpinned_items_and_stays_exact() {
    let items: Vec<Vec<Item>> = (0..12u8)
        .map(|index| {
            vec![Item {
                key: 10 + index % 7,
                rows: 3 + u32::from(index % 4) * 2,
            }]
        })
        .collect();
    let slices: Vec<&[Item]> = items.iter().map(Vec::as_slice).collect();
    let (media, text) = workload(&slices);
    let expected = reference(&text, 2);
    // Six pages hold at most two of these items at once.
    let mut engine =
        Engine::new(config(16, 3, Some(media_config(6, 0))), CopyModel::new(6)).unwrap();
    let actual = run(&mut engine, &media, 2);
    assert_eq!(actual, expected);
    let metrics = engine.metrics().snapshot();
    assert!(metrics.media_evictions > 0);
    assert_eq!(
        metrics.media_encodes + metrics.media_cache_hits,
        items.len() as u64
    );
}

/// Each request's items total eight pages of a six-page arena, which
/// admission used to refuse. Items stream through the arena chunk by chunk,
/// and a chunk whose items would exceed it is shortened, so every request
/// still emits exactly its text equivalent.
#[test]
fn requests_whose_media_exceed_the_arena_stream_through_it_exactly() {
    let (media, text) = workload(&[&[D, B, A, E], &[B, D, C, A], &[E, A, D, B]]);
    for stagger in [1, 3] {
        let expected = reference(&text, stagger);
        for chunk in [2, 3, 8] {
            let mut engine = Engine::new(
                config(16, chunk, Some(media_config(6, 250))),
                CopyModel::new(6),
            )
            .unwrap();
            let actual = run(&mut engine, &media, stagger);
            assert_eq!(actual, expected, "stagger {stagger} chunk {chunk}");
        }
    }
}

/// Four pages hold the step demands of both prompts (one page for E or C,
/// three for D), so both start. A three-token chunk of the first covers E
/// and C together, two pages, so a step that runs it first evicts D while
/// D's request still needs it; D is encoded again when that request
/// resumes, which changes nothing it emits.
#[test]
fn an_item_evicted_between_its_chunks_is_encoded_again_exactly() {
    let (media, text) = workload(&[&[E, C], &[D]]);
    let expected = reference(&text, 1);
    let mut engine =
        Engine::new(config(16, 3, Some(media_config(4, 0))), CopyModel::new(4)).unwrap();
    let actual = run(&mut engine, &media, 1);
    assert_eq!(actual, expected);
    let metrics = engine.metrics().snapshot();
    assert_eq!(metrics.media_reencodes, 1);
    assert_eq!(
        metrics.media_encodes,
        3 + metrics.media_reencodes,
        "three distinct items, each encoded once more per re-encode"
    );
    assert!(metrics.prometheus().contains(&format!(
        "chop_media_reencodes_total {}",
        metrics.media_reencodes
    )));
}

/// Two prompts whose step demands do not fit the arena together take
/// turns: the second starts only once the first's prefill has passed D, so
/// D is never evicted while its request still needs it.
#[test]
fn media_prompts_that_cannot_share_the_arena_never_encode_again() {
    let (media, text) = workload(&[&[D], &[B, E]]);
    for chunk in [2, 3, 8] {
        let expected = reference(&text, 1);
        let mut engine = Engine::new(
            config(16, chunk, Some(media_config(4, 0))),
            CopyModel::new(4),
        )
        .unwrap();
        let actual = run(&mut engine, &media, 1);
        assert_eq!(actual, expected, "chunk {chunk}");
        let metrics = engine.metrics().snapshot();
        assert_eq!(metrics.media_reencodes, 0, "chunk {chunk}");
        assert_eq!(metrics.media_encodes, 3, "chunk {chunk}");
    }
}

/// The second request extends the first. Its items total five pages of a
/// four-page arena, which admission used to refuse. It resumes after the
/// first prompt, so it covers, places and encodes only its own new item:
/// the first item counts as neither an encode nor a cache hit again.
#[test]
fn a_prefix_hit_places_only_the_media_after_it() {
    let prompt = |extra: &[u32]| {
        let mut prompt = vec![5, 6];
        prompt.extend([PLACEHOLDER; 9]);
        prompt.extend([9, 10, 11]);
        prompt.extend_from_slice(extra);
        prompt
    };
    let request = |id: u64, extra: &[u32], items: &[Item], spans: Vec<MediaSpanV1>| {
        EngineRequest::new(
            RequestId::new(id),
            prompt(extra),
            3,
            SamplingParameters::new(0.0).unwrap(),
            id,
            Vec::new(),
            vocabulary(),
        )
        .unwrap()
        .with_media(items.iter().copied().map(media_input).collect(), spans)
        .unwrap()
    };
    let span = |media, prompt_row, rows| MediaSpanV1 {
        media,
        prompt_row,
        rows,
        media_row: 0,
    };
    let mut extra = vec![12];
    extra.extend([PLACEHOLDER; 8]);
    extra.extend([13, 14]);
    let first = request(1, &[], &[D], vec![span(0, 2, 9)]);
    let second = request(2, &extra, &[D, B], vec![span(0, 2, 9), span(1, 15, 8)]);
    let mut engine = Engine::new(
        config(16, 4, Some(media_config(4, 0))).with_prefix_cache(1),
        CopyModel::new(4),
    )
    .unwrap();
    assert_eq!(cached_tokens(&mut engine, first), None);
    assert_eq!(cached_tokens(&mut engine, second), Some(14));
    let metrics = engine.metrics().snapshot();
    assert_eq!(engine.executor().encodes, 2);
    assert_eq!(metrics.media_encodes, 2);
    assert_eq!(metrics.media_cache_hits, 0);
}

/// A payload whose live copies its lease counts: the lease is held, never
/// read, so it drops with the last copy.
struct Leased(Vec<u8>, #[allow(dead_code)] std::sync::Arc<()>);

impl AsRef<[u8]> for Leased {
    fn as_ref(&self) -> &[u8] {
        &self.0
    }
}

/// Once prefill passes an item, the engine drops its payload, which only an
/// encode reads, so its host memory returns while the request still decodes.
#[test]
fn payloads_drop_once_prefill_passes_their_items() {
    let lease = std::sync::Arc::new(());
    let items = [D, B];
    let media = items
        .iter()
        .map(|item| {
            let mut input = media_input(*item);
            let bytes = input.payload.to_vec();
            input.payload = runtime_engine::MediaPayloadV1::new(Leased(bytes, lease.clone()));
            input
        })
        .collect::<Vec<_>>();
    let mut prompt = vec![5, 6];
    prompt.extend([PLACEHOLDER; 9]);
    prompt.push(9);
    prompt.extend([PLACEHOLDER; 8]);
    prompt.push(10);
    let span = |media, prompt_row, rows| MediaSpanV1 {
        media,
        prompt_row,
        rows,
        media_row: 0,
    };
    let request = EngineRequest::new(
        RequestId::new(1),
        prompt,
        16,
        SamplingParameters::new(0.0).unwrap(),
        1,
        Vec::new(),
        vocabulary(),
    )
    .unwrap()
    .with_media(media, vec![span(0, 2, 9), span(1, 12, 8)])
    .unwrap();
    let mut engine =
        Engine::new(config(16, 4, Some(media_config(64, 0))), CopyModel::new(64)).unwrap();
    engine.submit(request).unwrap();
    assert_eq!(std::sync::Arc::strong_count(&lease), 3);
    let mut tokens = 0;
    while tokens == 0 {
        engine.step().unwrap();
        engine.audit().unwrap();
        tokens += engine
            .drain_events()
            .iter()
            .filter(|event| matches!(event.kind(), EngineEventKind::Token { .. }))
            .count();
    }
    // The prompt is prefilled and the request decodes on; neither payload
    // is held any longer.
    assert_eq!(std::sync::Arc::strong_count(&lease), 1);
    engine.run_until_idle(256).unwrap();
}

/// A prefix hit whose restore is refused prefills from the start, so the
/// items before the hit are needed again after all. Here the first item was
/// evicted from the arena in between, so it must be encoded again from its
/// payload, which the engine keeps until a restore succeeds.
#[test]
fn a_refused_restore_still_finds_the_payloads_before_its_hit() {
    let prompt = |extra: &[u32]| {
        let mut prompt = vec![5, 6];
        prompt.extend([PLACEHOLDER; 9]);
        prompt.extend([9, 10, 11]);
        prompt.extend_from_slice(extra);
        prompt
    };
    let reference_prompt = |extra: &[u32]| {
        let mut prompt = vec![5, 6];
        prompt.extend((0..9).map(|row| effective(media_value(D.key, row))));
        prompt.extend([9, 10, 11]);
        prompt.extend(extra.iter().enumerate().map(|(row, token)| {
            if *token == PLACEHOLDER {
                effective(media_value(B.key, row as u32 - 1))
            } else {
                *token
            }
        }));
        prompt
    };
    let span = |media, prompt_row, rows| MediaSpanV1 {
        media,
        prompt_row,
        rows,
        media_row: 0,
    };
    let mut extra = vec![12];
    extra.extend([PLACEHOLDER; 8]);
    extra.extend([13, 14]);
    let request = |id: u64, prompt: Vec<u32>| {
        EngineRequest::new(
            RequestId::new(id),
            prompt,
            3,
            SamplingParameters::new(0.0).unwrap(),
            id,
            Vec::new(),
            vocabulary(),
        )
        .unwrap()
    };
    let first = request(1, prompt(&[]))
        .with_media(vec![media_input(D)], vec![span(0, 2, 9)])
        .unwrap();
    // Another prompt whose item takes the pages the first item held.
    let (other, other_text) = requests(2, &[B, E], 3, 0.0);
    let second = request(3, prompt(&extra))
        .with_media(
            vec![media_input(D), media_input(B)],
            vec![span(0, 2, 9), span(1, 15, 8)],
        )
        .unwrap();
    // One request at a time, to idle; anchors keep pages, so the pool is
    // not checked for free pages as `run` does.
    let serial = |engine: &mut Engine<CopyModel>, requests: Vec<EngineRequest>| {
        let mut outputs = Outputs::new();
        for request in requests {
            engine.submit(request).unwrap();
            engine.run_until_idle(256).unwrap();
            engine.audit().unwrap();
            for event in engine.drain_events() {
                record(&mut outputs, event);
            }
        }
        outputs
    };
    let mut text = Engine::new(config(16, 3, None), CopyModel::new(1)).unwrap();
    let expected = serial(
        &mut text,
        vec![
            request(1, reference_prompt(&[])),
            other_text,
            request(3, reference_prompt(&extra)),
        ],
    );
    let mut model = CopyModel::new(4);
    model.refuse_restores = true;
    let mut engine = Engine::new(
        config(16, 4, Some(media_config(4, 0))).with_prefix_cache(1),
        model,
    )
    .unwrap();
    let actual = serial(&mut engine, vec![first, other, second]);
    assert_eq!(actual, expected);
    // D, then B and E evicting it, then D again for the full prefill.
    assert_eq!(engine.executor().encodes, 5);
}

#[test]
fn requests_the_arena_cannot_hold_or_serve_are_refused() {
    let (media, _) = requests(1, &[D], 4, 0.0);
    let mut small =
        Engine::new(config(16, 3, Some(media_config(2, 0))), CopyModel::new(2)).unwrap();
    assert_eq!(
        small.submit(media.clone()),
        Err(EngineError::Submit(SubmitError::Invalid(
            InvalidRequest::ExceedsMediaArena {
                maximum: 2,
                requested: 3
            }
        )))
    );
    let mut text_only = Engine::new(config(16, 3, None), CopyModel::new(1)).unwrap();
    assert_eq!(
        text_only.submit(media),
        Err(EngineError::InvalidRequest(EngineRequestError::Media(
            MediaRequestError::Unsupported
        )))
    );
    let request = EngineRequest::new(
        RequestId::new(2),
        vec![1, 2, 3],
        4,
        SamplingParameters::new(0.0).unwrap(),
        0,
        Vec::new(),
        vocabulary(),
    )
    .unwrap();
    let span = MediaSpanV1 {
        media: 0,
        prompt_row: 2,
        rows: 2,
        media_row: 0,
    };
    assert_eq!(
        request.with_media(vec![media_input(A)], vec![span]).err(),
        Some(EngineRequestError::Media(
            MediaRequestError::SpanOutsidePrompt
        ))
    );
}

#[test]
fn a_failed_encode_step_rolls_the_arena_back() {
    let mut model = CopyModel::new(8);
    model.fail_on_encode = true;
    let mut engine = Engine::new(config(16, 8, Some(media_config(8, 0))), model).unwrap();
    let (first, _) = requests(1, &[B], 4, 0.0);
    engine.submit(first).unwrap();
    assert!(matches!(engine.step(), Err(EngineError::Executor(_))));
    engine.audit().unwrap();
    let _ = engine.drain_events();

    engine.executor_mut().fail_on_encode = false;
    let (second, text) = requests(2, &[B], 4, 0.0);
    let expected = reference(&[text], 1);
    let actual = run(&mut engine, &[second], 1);
    assert_eq!(actual, expected);
    // The failed step's pages were freed, so the item is encoded again.
    assert_eq!(engine.executor().encodes, 1);
    assert_eq!(engine.metrics().snapshot().media_cache_hits, 0);
}

#[test]
fn an_expensive_encode_delays_decoding_by_at_most_one_step() {
    // Each encoded row costs 100 tokens, far over the 8-token budget.
    let mut engine = Engine::new(
        config(8, 4, Some(media_config(64, 100_000))),
        CopyModel::new(64),
    )
    .unwrap();
    let decoding: Vec<_> = (1..=3)
        .map(|id| {
            EngineRequest::new(
                RequestId::new(id),
                vec![1, 2],
                30,
                SamplingParameters::new(0.0).unwrap(),
                id,
                Vec::new(),
                vocabulary(),
            )
            .unwrap()
        })
        .collect();
    for request in decoding {
        engine.submit(request).unwrap();
    }
    engine.step().unwrap();
    for (id, item) in [(4, B), (5, D), (6, A)] {
        engine.submit(requests(id, &[item], 2, 0.0).0).unwrap();
    }
    let mut last_token = BTreeMap::new();
    for step in 1..25 {
        engine.step().unwrap();
        engine.audit().unwrap();
        for event in engine.drain_events() {
            if matches!(event.kind(), EngineEventKind::Token { .. }) {
                last_token.insert(event.request_id().get(), step);
            }
        }
        for id in 1..=3 {
            if let Some(last) = last_token.get(&id) {
                assert!(step - last <= 1, "request {id} waited at step {step}");
            }
        }
    }
    assert_eq!(engine.metrics().snapshot().media_encodes, 3);
}

#[test]
fn media_prefill_and_prompt_lookup_speculation_coexist() {
    let (media, text) = workload(&[&[A], &[B, C], &[A, E]]);
    let speculation =
        SpeculationConfig::new(4, 32, SpeculationMode::Fixed, DraftSource::Host).unwrap();
    let mut plain = Engine::new(
        config(16, 3, None).with_speculation(speculation),
        PromptLookupExecutor::new(
            CopyModel::new(1),
            PromptLookupConfig::new(1, 2, 64).unwrap(),
        ),
    )
    .unwrap();
    let expected = run(&mut plain, &text, 2);
    assert_eq!(expected, reference(&text, 2));
    let mut engine = Engine::new(
        config(16, 3, Some(media_config(64, 250))).with_speculation(speculation),
        PromptLookupExecutor::new(
            CopyModel::new(64),
            PromptLookupConfig::new(1, 2, 64).unwrap(),
        ),
    )
    .unwrap();
    assert_eq!(run(&mut engine, &media, 2), expected);
    assert!(engine.metrics().snapshot().spec_verify_items > 0);
}

/// Runs one request alone and returns whether it resumed from a prefix.
fn cached_tokens(engine: &mut Engine<CopyModel>, request: EngineRequest) -> Option<usize> {
    engine.submit(request).unwrap();
    engine.run_until_idle(256).unwrap();
    engine.audit().unwrap();
    engine
        .drain_events()
        .into_iter()
        .find_map(|event| match event.kind() {
            EngineEventKind::PromptCached { tokens } => Some(*tokens),
            _ => None,
        })
}

#[test]
fn equal_payloads_with_different_geometry_share_neither_cache() {
    // A 64x128 and a 128x64 image can pack to the same patch bytes and row
    // count while their grids, and so their encoded rows, differ.
    let shaped = |id: u64, geometry: [u32; 4], extra: &[u32]| {
        let mut item = media_input(B);
        item.geometry = geometry;
        let mut prompt = vec![5, 6];
        prompt.extend([PLACEHOLDER; 8]);
        prompt.extend([9, 10]);
        prompt.extend_from_slice(extra);
        EngineRequest::new(
            RequestId::new(id),
            prompt,
            2,
            SamplingParameters::new(0.0).unwrap(),
            id,
            Vec::new(),
            vocabulary(),
        )
        .unwrap()
        .with_media(
            vec![item],
            vec![MediaSpanV1 {
                media: 0,
                prompt_row: 2,
                rows: 8,
                media_row: 0,
            }],
        )
        .unwrap()
    };
    let mut engine = Engine::new(
        config(16, 8, Some(media_config(64, 0))).with_prefix_cache(1),
        CopyModel::new(64),
    )
    .unwrap();
    assert_eq!(
        cached_tokens(&mut engine, shaped(1, [1, 8, 4, 0], &[])),
        None
    );
    assert_eq!(engine.executor().encodes, 1);
    assert_eq!(
        cached_tokens(&mut engine, shaped(2, [1, 4, 8, 0], &[11, 12])),
        None,
        "another grid must not resume the first image's prefix"
    );
    assert_eq!(
        engine.executor().encodes,
        2,
        "another grid must not reuse the first image's encoded rows"
    );
    assert_eq!(
        cached_tokens(&mut engine, shaped(3, [1, 8, 4, 0], &[11, 12])),
        Some(12),
        "the same image still resumes"
    );
}

/// A scored media prompt reports the same row scores whatever its chunks
/// or the scored row cap, one event per chunk of at most the cap's rows.
#[test]
fn scored_media_prompts_do_not_depend_on_chunks_or_the_scored_cap() {
    let (media, _) = workload(&[&[B, A, D], &[E], &[A, C]]);
    let media = media
        .into_iter()
        .map(|request| request.with_prompt_logprobs(2))
        .collect::<Vec<_>>();
    // Prompt rows one by one, then every other event, per request.
    let scores = |chunk: usize, cap: usize| {
        let mut engine = Engine::new(
            scored_config(16, chunk, Some(media_config(64, 250)), cap),
            CopyModel::new(64),
        )
        .unwrap();
        run(&mut engine, &media, 2)
            .into_iter()
            .map(|(id, kinds)| {
                let mut flat = Vec::new();
                for kind in kinds {
                    match kind {
                        EngineEventKind::PromptLogprobs { positions } => {
                            assert!(!positions.is_empty() && positions.len() <= chunk.min(cap));
                            flat.extend(positions.iter().map(|row| format!("prompt {row:?}")));
                        }
                        other => flat.push(format!("{other:?}")),
                    }
                }
                (id, flat)
            })
            .collect::<BTreeMap<_, _>>()
    };
    let expected = scores(8, usize::MAX);
    // Every prompt position but the first is scored: at least the seven of
    // the shortest prompt, two text tokens and two spans of E.
    assert!(expected.values().all(|events| {
        events
            .iter()
            .filter(|event| event.starts_with("prompt "))
            .count()
            >= 7
    }));
    for chunk in [2, 3, 8] {
        for cap in [1, 3, 5] {
            assert_eq!(scores(chunk, cap), expected, "chunk {chunk} cap {cap}");
        }
    }
}

#[test]
fn media_requests_stay_pinned_under_forced_preemption() {
    let (media, text) = workload(&[&[A], &[B, C], &[A, E]]);
    let expected = reference(&text, 1);
    let mut e = Engine::new(
        config(16, 3, Some(media_config(64, 250))).with_admission(
            runtime_scheduler::AdmissionPolicy::Optimistic(
                runtime_scheduler::OptimisticAdmission::new(0, 3)
                    .unwrap()
                    .with_forced_preemption(1)
                    .unwrap(),
            ),
        ),
        CopyModel::new(64),
    )
    .unwrap();
    assert_eq!(run(&mut e, &media, 1), expected);
    let metrics = e.metrics().prometheus_compatibility("test");
    let preemptions = metrics
        .lines()
        .find(|line| line.starts_with("vllm:num_preemptions_total{"))
        .unwrap();
    assert_eq!(preemptions.rsplit(' ').next().unwrap(), "0");
}
