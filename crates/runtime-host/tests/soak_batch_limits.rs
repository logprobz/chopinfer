//! A fresh server's first minute of the mixed soak, on the CPU.
//!
//! The real engine, scheduler and bundle executor adapter serve a workload
//! shaped like `tools/qual/soak.py --prefix-mix --media-mix --structured-mix`
//! with the production TP4 serving options (32 sequences, an 8,192-token
//! batch budget, 2,048-token chunks, prefix cache, a 512-page media arena and
//! adaptive speculation with seven drafts and 32 verify rows). The bundle side
//! is the shared SM120 input packer at the served TP4 bundle's geometry, which
//! refuses exactly what a native bundle's execute refuses before it touches
//! the device.
//!
//! A fresh server encodes each media item once, so for its first requests an
//! encode can overrun the batch budget. A batch that took the overrunning
//! chunk's prompt tokens on top of a full budget had more than 8,192 prefill
//! tokens and was refused with INVALID_ARGUMENT, which stopped the worker.
//!
//! Token counts follow the soak's prompts through a word, digit and
//! punctuation split, and media rows are the served fixtures' (image 64,
//! audio 96, video 4 x 64). Drafts come from prompt lookup over a model that
//! repeats its context, so verify items appear as they do with a drafter.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::rc::Rc;

use bundle_runtime::BundleGeometry;
use bundle_runtime::inputs::{InputError, PackedStep, VerifyLimits};
use runtime_bundle_abi::{
    BundleAbiError, BundleKeptRowsV1, BundleStatusV1, OwnedLogitsRowV1, ResultSinkLimitsV1,
};
use runtime_bundle_sdk::BatchRequestV1;
use runtime_engine::{
    DraftCostModel, DraftSource, Engine, EngineConfig, EngineEventKind, EngineRequest,
    MediaConfigV1, MediaInputV1, MediaSpanV1, PromptLookupConfig, PromptLookupExecutor,
    SpeculationConfig, SpeculationMode,
};
use runtime_host::{BatchWireCapabilityV1, BundleExecutionV1, BundleModelExecutorV1};
use runtime_kv::KvGeometry;
use runtime_model_api::ModelVocabulary;
use runtime_sampling::SamplingParameters;
use runtime_scheduler::{RequestId, SchedulerCapacityProfile, SchedulerConfig};

/// The step geometry of the served TP4 bundle.
struct Tp4Bundle;

impl BundleGeometry for Tp4Bundle {
    const PAGE_TOKENS: u64 = 64;
    const PREFILL_TOKENS: u32 = 8192;
    const DECODE_BATCH: u32 = 32;
    const MAX_VERIFY_ROWS: u32 = 8;
    const TOKEN_GRANULE: u32 = 256;
}

const LIMITS: bundle_runtime::ModelLimits =
    match bundle_runtime::ModelLimits::new(152_576, 1_048_576, 64) {
        Ok(limits) => limits,
        Err(_) => panic!("invalid model limits"),
    };

const VOCABULARY: usize = 4096;
const KV_PAGES: usize = 90_450;
const SEQUENCES: usize = 32;
const QUEUE_DEPTH: usize = 1024;
const BATCH_TOKENS: usize = 8192;
const CHUNK_TOKENS: usize = 2048;
const SNAPSHOT_PAGES: usize = 46;
const DRAFT_TOKENS: usize = 7;
const VERIFY_ROWS: usize = 32;
/// Requests the workers complete: every worker's first turns.
const REQUESTS: usize = 200;

const IM_START: u32 = 1;
const IM_END: u32 = 2;
const NEWLINE: u32 = 3;
const VISION_START: u32 = 4;
const VISION_END: u32 = 5;
const IMAGE_PAD: u32 = 6;
const AUDIO_START: u32 = 7;
const AUDIO_END: u32 = 8;
const AUDIO_PAD: u32 = 9;
const VIDEO_PAD: u32 = 10;

/// What one batch asked of the bundle.
#[derive(Clone, Debug, Default)]
struct Shape {
    items: usize,
    prefill_items: usize,
    prefill_tokens: usize,
    decode_items: usize,
    verify_items: usize,
    decode_rows: usize,
    media_encodes: usize,
    media_rows: usize,
}

impl Shape {
    fn of(request: &BatchRequestV1) -> Self {
        let mut shape = Self {
            items: request.items().len(),
            media_encodes: request.media().encodes.len(),
            media_rows: request
                .media()
                .encodes
                .iter()
                .map(|encode| encode.output_rows as usize)
                .sum(),
            ..Self::default()
        };
        for item in request.items() {
            let tokens = item.phase_tokens() as usize;
            if item.is_verify() {
                shape.verify_items += 1;
                shape.decode_rows += tokens;
            } else if item.is_decode() {
                shape.decode_items += 1;
                shape.decode_rows += tokens;
            } else {
                shape.prefill_items += 1;
                shape.prefill_tokens += tokens;
            }
        }
        shape
    }
}

impl std::fmt::Display for Shape {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "{} items: {} prefill with {} tokens, {} decode and {} verify with {} rows, \
             {} media encodes with {} rows",
            self.items,
            self.prefill_items,
            self.prefill_tokens,
            self.decode_items,
            self.verify_items,
            self.decode_rows,
            self.media_encodes,
            self.media_rows
        )
    }
}

#[derive(Default)]
struct Record {
    batches: usize,
    largest_prefill: usize,
    verify_items: usize,
    media_encodes: usize,
    restores: usize,
    refused: Option<(InputError, Shape)>,
}

/// The bundle side: packs every batch the way a native SM120 bundle does
/// and keeps each request's committed tokens for the model's logits.
struct PackingExecution {
    record: Rc<RefCell<Record>>,
    slots: BTreeMap<u64, u32>,
    committed: BTreeMap<u64, Vec<u32>>,
    pending: Vec<(u64, Vec<u32>, bool)>,
}

impl PackingExecution {
    fn free_slot(&self, taken: &[u32]) -> Option<u32> {
        (0..Tp4Bundle::DECODE_BATCH)
            .find(|slot| !self.slots.values().any(|used| used == slot) && !taken.contains(slot))
    }
}

/// The model repeats the token eight positions back, so prompt lookup finds
/// drafts it accepts, as a trained drafter's are.
fn logits_after(context: &[u32]) -> Vec<u8> {
    let favoured = context
        .len()
        .checked_sub(8)
        .map_or(IM_END + 100, |back| context[back]) as usize
        % VOCABULARY;
    (0..VOCABULARY)
        .flat_map(|token| {
            let logit: f32 = if token == favoured { 8.0 } else { 0.0 };
            logit.to_le_bytes()
        })
        .collect()
}

impl BundleExecutionV1 for PackingExecution {
    fn supports_request_retirement(&self) -> bool {
        true
    }

    fn supports_speculation(&self) -> bool {
        true
    }

    fn retire_request(&mut self, request_id: u64) -> Result<(), BundleAbiError> {
        self.slots.remove(&request_id);
        self.committed.remove(&request_id);
        Ok(())
    }

    fn capture_prefix_state(
        &mut self,
        _request_id: u64,
        _tokens: u64,
        _pages: &[u64],
    ) -> Result<(), BundleAbiError> {
        Ok(())
    }

    fn restore_prefix_state(
        &mut self,
        request_id: u64,
        token_ids: &[u32],
        _pages: &[u64],
    ) -> Result<(), BundleAbiError> {
        let slot = self
            .free_slot(&[])
            .ok_or(BundleAbiError::Status(BundleStatusV1::ResourceExhausted))?;
        self.slots.insert(request_id, slot);
        self.committed.insert(request_id, token_ids.to_vec());
        self.record.borrow_mut().restores += 1;
        Ok(())
    }

    fn execute_batch(
        &mut self,
        request: &[u8],
        _limits: ResultSinkLimitsV1,
    ) -> Result<Vec<OwnedLogitsRowV1>, BundleAbiError> {
        let request = BatchRequestV1::decode(request)
            .map_err(|_| BundleAbiError::Status(BundleStatusV1::InvalidArgument))?;
        let mut taken = Vec::new();
        let mut new_slots = BTreeMap::new();
        for (item, append) in request.items().iter().zip(request.appends()) {
            if !self.slots.contains_key(&item.request_id())
                && append.previous_tokens() == 0
                && let Some(slot) = self.free_slot(&taken)
            {
                taken.push(slot);
                new_slots.insert(item.request_id(), slot);
            }
        }
        let packed = PackedStep::pack_with_verify::<Tp4Bundle>(
            LIMITS,
            &request,
            KV_PAGES as u64,
            VerifyLimits::enabled::<Tp4Bundle>(DRAFT_TOKENS as u32, VERIFY_ROWS as u32),
            |id| self.slots.get(&id).or(new_slots.get(&id)).copied(),
        );
        let shape = Shape::of(&request);
        {
            let mut record = self.record.borrow_mut();
            record.batches += 1;
            record.largest_prefill = record.largest_prefill.max(shape.prefill_tokens);
            record.verify_items += shape.verify_items;
            record.media_encodes += shape.media_encodes;
            if let Err(error) = packed {
                record.refused = Some((error, shape));
                // The status a native bundle's preflight returns.
                return Err(BundleAbiError::Status(match error {
                    InputError::MissingSlot => BundleStatusV1::NotFound,
                    InputError::UnsupportedVerify => BundleStatusV1::Unsupported,
                    _ => BundleStatusV1::InvalidArgument,
                }));
            }
        }
        self.slots.extend(new_slots);
        let mut rows = Vec::new();
        for item in request.items() {
            let id = item.request_id();
            let inputs = item.input_tokens().to_vec();
            if item.expects_logits() {
                let mut context = self.committed.get(&id).cloned().unwrap_or_default();
                let first = if item.is_verify() {
                    0
                } else {
                    inputs.len() - 1
                };
                context.extend_from_slice(&inputs[..first]);
                for &input in &inputs[first..] {
                    context.push(input);
                    rows.push(OwnedLogitsRowV1::new(id, logits_after(&context)));
                }
            }
            self.pending.push((id, inputs, item.is_verify()));
        }
        Ok(rows)
    }

    fn commit(&mut self) -> Result<(), BundleAbiError> {
        for (id, inputs, _) in self.pending.drain(..) {
            self.committed.entry(id).or_default().extend(inputs);
        }
        Ok(())
    }

    fn commit_accepted(&mut self, records: &[BundleKeptRowsV1]) -> Result<(), BundleAbiError> {
        for (id, inputs, verify) in self.pending.drain(..) {
            let kept = records
                .iter()
                .find(|record| record.request_id == id)
                .filter(|_| verify)
                .map_or(inputs.len(), |record| record.kept_rows as usize);
            self.committed
                .entry(id)
                .or_default()
                .extend_from_slice(&inputs[..kept]);
        }
        Ok(())
    }

    fn cancel(&mut self) -> Result<(), BundleAbiError> {
        self.pending.clear();
        Ok(())
    }

    fn destroy(&mut self) -> Result<(), BundleAbiError> {
        Ok(())
    }

    fn is_poisoned(&self) -> bool {
        false
    }
}

fn word(text: &str) -> u32 {
    let hash = text.bytes().fold(0xcbf2_9ce4_8422_2325_u64, |hash, byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x0100_0000_01b3)
    });
    100 + (hash % (VOCABULARY as u64 - 100)) as u32
}

/// Words, single digits and punctuation, as the served tokenizer splits them.
fn tokens(text: &str) -> Vec<u32> {
    let mut out = Vec::new();
    let mut current = String::new();
    for character in text.chars() {
        if character.is_alphabetic() {
            current.push(character);
            continue;
        }
        if !current.is_empty() {
            out.push(word(&current));
            current.clear();
        }
        if !character.is_whitespace() {
            out.push(word(&character.to_string()));
        }
    }
    if !current.is_empty() {
        out.push(word(&current));
    }
    out
}

fn message(role: &str, content: &[u32]) -> Vec<u32> {
    let mut out = vec![IM_START, word(role), NEWLINE];
    out.extend_from_slice(content);
    out.extend([IM_END, NEWLINE]);
    out
}

fn chat(messages: &[Vec<u32>]) -> Vec<u32> {
    let mut out: Vec<u32> = messages.concat();
    out.extend([IM_START, word("assistant"), NEWLINE]);
    out
}

const PROMPTS: [&str; 3] = [
    "Say hello.",
    "Summarize why a cache can improve repeated reads in one sentence.",
    "List three ways to check a numerical result. ",
];

fn prompt(index: usize) -> String {
    if index % 3 == 2 {
        PROMPTS[2].repeat(60)
    } else {
        PROMPTS[index % 3].to_owned()
    }
}

fn system(index: usize) -> String {
    let rules = (0..160)
        .map(|rule| {
            format!(
                "Rule {rule}: answer with care about topic {}.",
                index * 31 + rule
            )
        })
        .collect::<Vec<_>>()
        .join(" ");
    format!("You are assistant {index}. {rules}")
}

/// A frozen media case: its prompt, items and the spans their rows fill.
fn media_case(case: usize) -> (Vec<u32>, Vec<MediaInputV1>, Vec<MediaSpanV1>) {
    // The two audio transports carry the same file, so they share an item.
    let (encoder, runs, payload, text): (u16, &[u32], u8, &str) = match case {
        0 => (0, &[64], 1, "What color is the square?"),
        1 | 2 => (
            1,
            &[96],
            2,
            "Transcribe the speech, including the words you hear.",
        ),
        _ => (
            0,
            &[64, 64, 64, 64],
            3,
            "What colors appear in the video, in order?",
        ),
    };
    let rows: u32 = runs.iter().sum();
    let mut content = Vec::new();
    let mut spans = Vec::new();
    let mut media_row = 0;
    for (index, &run) in runs.iter().enumerate() {
        let (start, pad, end) = match (encoder, runs.len()) {
            (1, _) => (AUDIO_START, AUDIO_PAD, AUDIO_END),
            (_, 1) => (VISION_START, IMAGE_PAD, VISION_END),
            _ => (VISION_START, VIDEO_PAD, VISION_END),
        };
        if runs.len() > 1 {
            content.extend(tokens(&format!("{index}.0 seconds")));
        }
        content.push(start);
        spans.push((content.len(), run, media_row));
        content.extend(std::iter::repeat_n(pad, run as usize));
        content.push(end);
        media_row += run;
    }
    content.extend(tokens(text));
    let prompt = chat(&[message("user", &content)]);
    // The user turn's header precedes the content.
    let spans = spans
        .into_iter()
        .map(|(row, rows, media_row)| MediaSpanV1 {
            media: 0,
            prompt_row: row as u32 + 3,
            rows,
            media_row,
        })
        .collect();
    let item = MediaInputV1 {
        key: [payload; 32],
        encoder,
        geometry: [rows, 1, 1, 1],
        output_rows: rows,
        payload: vec![payload; 4096].into(),
    };
    (prompt, vec![item], spans)
}

struct Worker {
    index: usize,
    turn: usize,
    conversation: Vec<Vec<u32>>,
    live: Option<RequestId>,
    output: Vec<u32>,
}

impl Worker {
    /// The worker's next request, as `soak.py` forms it.
    fn next(&mut self, id: RequestId) -> EngineRequest {
        let (w, turn) = (self.index, self.turn);
        let sampling = SamplingParameters::new(0.0).unwrap();
        let vocabulary = ModelVocabulary::new(VOCABULARY).unwrap();
        let request = |prompt: Vec<u32>, maximum| {
            EngineRequest::new(id, prompt, maximum, sampling.clone(), 0, vec![], vocabulary)
                .unwrap()
        };
        if (w + turn) % 8 == 0 {
            let (prompt, media, spans) = media_case((w + turn) / 8 % 4);
            return request(prompt, 48).with_media(media, spans).unwrap();
        }
        if (w + turn) % 8 == 4 {
            // The strict JSON schema request; its answer is a short object.
            let text = "Return a JSON object describing a fictional person with a name, \
                        an age and a list of hobbies.";
            return request(chat(&[message("user", &tokens(text))]), 64);
        }
        let maximum = [8, 32, 96][turn % 3];
        let messages = match w % 3 {
            0 => {
                if self.conversation.is_empty() || self.conversation.len() > 12 {
                    self.conversation = vec![message(
                        "user",
                        &tokens(&format!("Worker {w} turn {turn}: {}", prompt(turn))),
                    )];
                }
                self.conversation.clone()
            }
            1 => vec![
                message("system", &tokens(&system((w + turn) % 4))),
                message(
                    "user",
                    &tokens(&format!("Worker {w} asks question {turn}.")),
                ),
            ],
            _ => vec![message(
                "user",
                &tokens(&format!("Worker {w} turn {turn}: {}", prompt(w + turn))),
            )],
        };
        request(chat(&messages), maximum)
    }

    fn finished(&mut self) {
        if (self.index + self.turn) % 8 != 0
            && (self.index + self.turn) % 8 != 4
            && self.index % 3 == 0
        {
            self.conversation.push(message("assistant", &self.output));
            self.conversation.push(message(
                "user",
                &tokens(&format!("Continue, turn {}.", self.turn + 1)),
            ));
        }
        self.output.clear();
        self.live = None;
        self.turn += 1;
    }
}

fn engine(record: Rc<RefCell<Record>>) -> Engine<impl runtime_engine::ModelExecutor> {
    let in_flight = SEQUENCES + QUEUE_DEPTH;
    let scheduler = SchedulerConfig::new(
        SEQUENCES,
        in_flight + SEQUENCES,
        BATCH_TOKENS,
        CHUNK_TOKENS,
        1,
    )
    .unwrap()
    .with_capacity_profile(SchedulerCapacityProfile::new("served", in_flight).unwrap())
    .unwrap();
    let context = LIMITS.context_tokens() as usize;
    let geometry = KvGeometry::new(64, context / 64 * SEQUENCES).unwrap();
    let media = MediaConfigV1::new(512, 64)
        .unwrap()
        .with_encoder(33_554_432, 1000)
        .unwrap()
        .with_encoder(30_720_512, 1000)
        .unwrap();
    let speculation = SpeculationConfig::new(
        DRAFT_TOKENS,
        VERIFY_ROWS,
        SpeculationMode::Adaptive,
        DraftSource::Host,
    )
    .unwrap()
    .with_draft_cost_model(DraftCostModel::PerBlock);
    let config = EngineConfig::new(
        scheduler,
        geometry,
        ModelVocabulary::new(VOCABULARY).unwrap(),
        context,
        SEQUENCES,
    )
    .unwrap()
    .with_kv_page_capacity(KV_PAGES)
    .unwrap()
    .with_prefix_cache(SNAPSHOT_PAGES)
    .with_media(media)
    .with_speculation(speculation);
    let execution = PackingExecution {
        record,
        slots: BTreeMap::new(),
        committed: BTreeMap::new(),
        pending: Vec::new(),
    };
    let executor = BundleModelExecutorV1::new_with_batch_wire(
        execution,
        VOCABULARY,
        SEQUENCES as u64,
        BatchWireCapabilityV1::V3MediaSpeculative,
    );
    let executor = PromptLookupExecutor::new(executor, PromptLookupConfig::new(1, 4, 256).unwrap());
    Engine::new(config, executor).unwrap()
}

/// Serves the soak's first minute on a fresh engine and returns what the
/// bundle side saw.
fn first_minute() -> (Record, usize) {
    let record = Rc::new(RefCell::new(Record::default()));
    let mut engine = engine(Rc::clone(&record));
    let mut next_id = 1_u64;
    let vocabulary = ModelVocabulary::new(VOCABULARY).unwrap();
    // The canary runs alone before the load starts.
    let canary = chat(&[
        message("system", &tokens(&system(0))),
        message("user", &tokens("State rule 7 in your own words.")),
    ]);
    let request = EngineRequest::new(
        RequestId::new(next_id),
        canary,
        48,
        SamplingParameters::new(0.0).unwrap(),
        0,
        vec![],
        vocabulary,
    )
    .unwrap();
    next_id += 1;
    engine.submit(request).unwrap();
    let mut workers = (0..SEQUENCES)
        .map(|index| Worker {
            index,
            turn: 0,
            conversation: Vec::new(),
            live: None,
            output: Vec::new(),
        })
        .collect::<Vec<_>>();
    let mut canary_done = false;
    let mut completed = 0;
    // Past the first turns every media item is resident in the arena.
    while completed < REQUESTS {
        if canary_done {
            for worker in &mut workers {
                if worker.live.is_none() {
                    let id = RequestId::new(next_id);
                    next_id += 1;
                    engine.submit(worker.next(id)).unwrap();
                    worker.live = Some(id);
                }
            }
        }
        if let Err(error) = engine.step() {
            assert!(record.borrow().refused.is_some(), "{error:?}");
            break;
        }
        for event in engine.drain_events() {
            let id = event.request_id();
            let worker = workers.iter_mut().find(|worker| worker.live == Some(id));
            match (event.kind(), worker) {
                (EngineEventKind::Token { token_id, .. }, Some(worker)) => {
                    worker.output.push(*token_id);
                }
                (EngineEventKind::Finished { .. }, Some(worker)) => {
                    worker.finished();
                    completed += 1;
                }
                (EngineEventKind::Finished { .. }, None) => canary_done = true,
                (EngineEventKind::Error { message }, _) => {
                    assert!(record.borrow().refused.is_some(), "{message}");
                }
                _ => {}
            }
        }
    }
    let record = record.take();
    (record, completed)
}

#[test]
fn a_fresh_server_never_forms_a_batch_the_bundle_refuses() {
    let (record, completed) = first_minute();
    if let Some((error, shape)) = &record.refused {
        panic!(
            "batch {} of a fresh server was refused with {error:?} after {completed} requests: \
             {shape} (bundle limits: {} items, {} prefill tokens, {VERIFY_ROWS} verify rows)",
            record.batches,
            Tp4Bundle::DECODE_BATCH,
            Tp4Bundle::PREFILL_TOKENS
        );
    }
    assert!(completed >= REQUESTS);
    assert!(record.largest_prefill <= Tp4Bundle::PREFILL_TOKENS as usize);
    // The load filled the prefill arena, encoded every media item, hit the
    // prefix cache and verified drafts, as the soak does.
    assert!(record.largest_prefill > BATCH_TOKENS - CHUNK_TOKENS);
    assert!(
        record.media_encodes >= 3,
        "{} encodes",
        record.media_encodes
    );
    assert!(record.restores > 0 && record.verify_items > 0);
}
