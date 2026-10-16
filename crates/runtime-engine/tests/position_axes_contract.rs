//! Explicit 3-axis positions must reach the executor for every row of a
//! request with position spans, in every phase, and for no other request.
//!
//! The fake executor checks each row's positions against an independent
//! transcription of the rule at the row's sequence index, which it reads
//! from the KV plan rather than from the engine's own bookkeeping.

use std::collections::BTreeMap;

use runtime_engine::{
    DraftSource, Engine, EngineConfig, EngineError, EngineEventKind, EngineRequest,
    EngineRequestError, ExecutorError, LogitsRow, MediaConfigV1, MediaInputV1, MediaSpanV1,
    ModelBatchItem, ModelExecutor, ModelTransaction, NoopModelTransactionState, PositionAxesError,
    PositionSpanV1, PromptLookupConfig, PromptLookupExecutor, SpeculationConfig, SpeculationMode,
};
use runtime_kv::{BlockId, KvGeometry, PreparedAppendBatch};
use runtime_model_api::ModelVocabulary;
use runtime_sampling::SamplingParameters;
use runtime_scheduler::{BatchPhase, RequestId, SchedulerConfig};

const VOCABULARY: usize = 16;
const PAD: u32 = 3;

/// The rule as the reference model code states it: walk the prompt, give
/// grid rows (start + t, start + h, start + w), advance one past the span's
/// largest position, then continue every later row at the maximum plus one.
fn reference(prompt: usize, spans: &[PositionSpanV1], total: usize) -> Vec<[i32; 3]> {
    let mut out = Vec::new();
    let mut current = 0;
    for span in spans {
        while out.len() < span.prompt_row as usize {
            out.push([current; 3]);
            current += 1;
        }
        let [t, h, w] = span.grid.map(|value| value as i32);
        for ti in 0..t {
            for hi in 0..h {
                for wi in 0..w {
                    out.push([current + ti, current + hi, current + wi]);
                }
            }
        }
        current += t.max(h).max(w);
    }
    while out.len() < prompt {
        out.push([current; 3]);
        current += 1;
    }
    let delta = out.iter().flatten().copied().max().unwrap_or(-1) + 1 - prompt as i32;
    out.extend((prompt..total).map(|row| [row as i32 + delta; 3]));
    out
}

#[derive(Default)]
struct PositionModel {
    /// The expected positions of every sequence row of a request.
    expected: BTreeMap<RequestId, Vec<[i32; 3]>>,
    /// Rows checked per phase: prefill, decode, verify.
    checked: [usize; 3],
    /// Prefill items that started past row zero of a request with spans.
    resumed: usize,
    encoded: usize,
}

impl ModelExecutor for PositionModel {
    type TransactionState<'a> = NoopModelTransactionState;

    fn retire_request_v1(&mut self, _: RequestId) -> Result<(), ExecutorError> {
        Ok(())
    }

    fn capture_prefix_state_v1(
        &mut self,
        _: RequestId,
        _: usize,
        _: &[BlockId],
    ) -> Result<(), ExecutorError> {
        Ok(())
    }

    fn restore_prefix_state_v1(
        &mut self,
        _: RequestId,
        _: &[u32],
        _: &[BlockId],
    ) -> Result<(), ExecutorError> {
        Ok(())
    }

    fn execute<'a>(
        &'a mut self,
        items: &[ModelBatchItem],
        kv_plan: &PreparedAppendBatch,
    ) -> Result<ModelTransaction<Self::TransactionState<'a>>, ExecutorError> {
        let mut rows = Vec::new();
        for (item, append) in items.iter().zip(kv_plan.appends()) {
            self.encoded += item.media_encodes().len();
            let count = append.additional_tokens();
            let first = append.previous_tokens();
            match self.expected.get(&item.request_id()) {
                Some(expected) => {
                    assert_eq!(item.position_axes(), &expected[first..first + count]);
                    let phase = match item.phase() {
                        BatchPhase::Prefill { .. } => {
                            self.resumed += usize::from(first > 0);
                            0
                        }
                        BatchPhase::Decode => 1,
                        BatchPhase::Verify { .. } => 2,
                    };
                    self.checked[phase] += count;
                }
                None => assert!(item.position_axes().is_empty()),
            }
            if item.sampling().is_some() {
                let emitted = if item.draft().is_some() { count } else { 1 };
                for row in first + count - emitted..first + count {
                    let mut logits = vec![0.0; VOCABULARY];
                    logits[(row * 5 + 1) % VOCABULARY] = 4.0;
                    rows.push(LogitsRow::new(item.request_id(), logits).unwrap());
                }
            }
        }
        Ok(ModelTransaction::new(rows, NoopModelTransactionState))
    }
}

fn vocabulary() -> ModelVocabulary {
    ModelVocabulary::new(VOCABULARY).unwrap()
}

fn config(chunk: usize) -> EngineConfig {
    EngineConfig::new(
        SchedulerConfig::new(4, 64, 32, chunk, 1).unwrap(),
        KvGeometry::new(4, 256).unwrap(),
        vocabulary(),
        256,
        64,
    )
    .unwrap()
    .with_media(
        MediaConfigV1::new(64, 4)
            .unwrap()
            .with_encoder(1 << 20, 1)
            .unwrap(),
    )
    .with_position_axes()
}

/// A prompt of text around images of the given grids; each image is one
/// span. Returns the request and the positions every row must take.
fn request(id: u64, grids: &[[u32; 3]], maximum: usize) -> (EngineRequest, Vec<[i32; 3]>) {
    // Repeated text gives prompt lookup something to draft from.
    let text = [5, 6, 7, 8, 5, 6, 7];
    let mut prompt = text.to_vec();
    let mut media = Vec::new();
    let mut spans = Vec::new();
    let mut positions = Vec::new();
    for (index, grid) in grids.iter().enumerate() {
        let rows = grid.iter().product::<u32>();
        spans.push(MediaSpanV1 {
            media: index as u16,
            prompt_row: prompt.len() as u32,
            rows,
            media_row: 0,
        });
        positions.push(PositionSpanV1 {
            prompt_row: prompt.len() as u32,
            rows,
            grid: *grid,
        });
        prompt.extend(std::iter::repeat_n(PAD, rows as usize));
        prompt.extend_from_slice(&text[..3]);
        media.push(MediaInputV1 {
            key: [16 + index as u8; 32],
            encoder: 0,
            geometry: [grid[0], 2 * grid[1], 2 * grid[2], 0],
            output_rows: rows,
            payload: vec![1_u8; 4 * rows as usize].into(),
        });
    }
    let expected = reference(prompt.len(), &positions, prompt.len() + maximum + 16);
    let request = EngineRequest::new(
        RequestId::new(id),
        prompt,
        maximum,
        SamplingParameters::new(0.0).unwrap(),
        id,
        Vec::new(),
        vocabulary(),
    )
    .unwrap()
    .with_media(media, spans)
    .unwrap()
    .with_position_axes(positions)
    .unwrap();
    (request, expected)
}

fn text_request(id: u64) -> EngineRequest {
    EngineRequest::new(
        RequestId::new(id),
        vec![5, 6, 7, 8, 5, 6, 7, 8, 5],
        6,
        SamplingParameters::new(0.0).unwrap(),
        id,
        Vec::new(),
        vocabulary(),
    )
    .unwrap()
}

fn drain<E: ModelExecutor>(engine: &mut Engine<E>) -> Vec<EngineEventKind> {
    engine.run_until_idle(512).unwrap();
    engine.audit().unwrap();
    engine
        .drain_events()
        .into_iter()
        .map(|event| event.kind().clone())
        .collect()
}

#[test]
fn chunked_prefill_and_decode_rows_carry_the_rule_and_text_requests_carry_none() {
    let mut engine = Engine::new(config(5), PositionModel::default()).unwrap();
    // Two images, the first split across three chunks of five rows.
    let (first, expected) = request(1, &[[1, 3, 4], [1, 2, 2]], 6);
    engine
        .executor_mut()
        .expected
        .insert(RequestId::new(1), expected.clone());
    // The prompt: 7 text rows, 12 grid rows at (7 + r, 7 + c), 3 text rows
    // resuming at 11, 4 grid rows from 14, 3 text rows from 16.
    assert_eq!(expected[7], [7, 7, 7]);
    assert_eq!(expected[18], [7, 9, 10]);
    assert_eq!(expected[19..22], [[11, 11, 11], [12, 12, 12], [13, 13, 13]]);
    assert_eq!(expected[25], [14, 15, 15]);
    assert_eq!(expected[26], [16, 16, 16]);
    assert_eq!(expected[29], [19, 19, 19]);
    engine.submit(first).unwrap();
    engine.submit(text_request(2)).unwrap();
    // A second image request of another grid.
    let (third, expected) = request(3, &[[1, 2, 3]], 4);
    engine
        .executor_mut()
        .expected
        .insert(RequestId::new(3), expected);
    engine.submit(third).unwrap();
    let events = drain(&mut engine);
    assert!(
        events
            .iter()
            .any(|event| matches!(event, EngineEventKind::Finished { .. }))
    );
    let checked = engine.executor().checked;
    assert_eq!(checked[0], 29 + 7 + 6 + 3);
    assert!(checked[1] >= 5 + 3);
}

#[test]
fn verify_rows_take_consecutive_generated_positions() {
    let speculation =
        SpeculationConfig::new(3, 16, SpeculationMode::Fixed, DraftSource::Host).unwrap();
    let mut engine = Engine::new(
        config(8).with_speculation(speculation),
        PromptLookupExecutor::new(
            PositionModel::default(),
            PromptLookupConfig::new(1, 2, 64).unwrap(),
        ),
    )
    .unwrap();
    let (request, expected) = request(1, &[[1, 2, 3]], 24);
    engine
        .executor_mut()
        .inner_mut()
        .expected
        .insert(RequestId::new(1), expected);
    engine.submit(request).unwrap();
    drain(&mut engine);
    assert!(engine.executor().inner().checked[2] > 0);
}

#[test]
fn a_prefix_cache_resume_starts_mid_prompt_at_the_same_positions() {
    let mut engine = Engine::new(config(8).with_prefix_cache(1), PositionModel::default()).unwrap();
    for (id, grids) in [(1, &[[1, 3, 4]][..]), (2, &[[1, 3, 4], [1, 2, 2]][..])] {
        // The second prompt extends the first, image included.
        let (request, expected) = request(id, grids, 3);
        engine
            .executor_mut()
            .expected
            .insert(RequestId::new(id), expected);
        engine.submit(request).unwrap();
        let cached = drain(&mut engine)
            .into_iter()
            .find_map(|event| match event {
                EngineEventKind::PromptCached { tokens } => Some(tokens),
                _ => None,
            });
        assert_eq!(cached.is_some(), id == 2);
    }
    assert!(engine.executor().resumed > 0);
}

#[test]
fn position_spans_need_the_engine_capability_and_a_matching_media_span() {
    let (request, _) = request(1, &[[1, 2, 2]], 2);
    let mut plain = Engine::new(
        EngineConfig::new(
            SchedulerConfig::new(4, 64, 32, 8, 1).unwrap(),
            KvGeometry::new(4, 64).unwrap(),
            vocabulary(),
            128,
            64,
        )
        .unwrap()
        .with_media(
            MediaConfigV1::new(8, 4)
                .unwrap()
                .with_encoder(1 << 20, 1)
                .unwrap(),
        ),
        PositionModel::default(),
    )
    .unwrap();
    assert_eq!(
        plain.submit(request),
        Err(EngineError::InvalidRequest(
            EngineRequestError::PositionAxes(PositionAxesError::Unsupported)
        ))
    );
    let base = || {
        EngineRequest::new(
            RequestId::new(1),
            vec![5, PAD, PAD, PAD, PAD, 6],
            2,
            SamplingParameters::new(0.0).unwrap(),
            1,
            Vec::new(),
            vocabulary(),
        )
        .unwrap()
        .with_media(
            vec![MediaInputV1 {
                key: [1; 32],
                encoder: 0,
                geometry: [1, 4, 4, 0],
                output_rows: 4,
                payload: vec![1_u8; 16].into(),
            }],
            vec![MediaSpanV1 {
                media: 0,
                prompt_row: 1,
                rows: 4,
                media_row: 0,
            }],
        )
        .unwrap()
    };
    let span = |prompt_row, grid| PositionSpanV1 {
        prompt_row,
        rows: 4,
        grid,
    };
    assert!(base().with_position_axes(vec![span(1, [1, 2, 2])]).is_ok());
    assert!(base().with_position_axes(Vec::new()).is_ok());
    assert_eq!(
        base().with_position_axes(vec![span(2, [1, 2, 2])]).err(),
        Some(EngineRequestError::PositionAxes(
            PositionAxesError::NotMediaSpan
        ))
    );
    assert_eq!(
        base().with_position_axes(vec![span(1, [1, 2, 3])]).err(),
        Some(EngineRequestError::PositionAxes(
            PositionAxesError::GridRows
        ))
    );
}

fn cached_extension(first_grid: Option<[u32; 3]>, next_grid: Option<[u32; 3]>) -> usize {
    let mut engine = Engine::new(config(8).with_prefix_cache(1), PositionModel::default()).unwrap();
    let mut cached = 0;
    for (id, grids, rule) in [
        (1, &[[1, 3, 4]][..], first_grid),
        (2, &[[1, 3, 4], [1, 2, 2]][..], next_grid),
    ] {
        let (request, _) = request(id, grids, 3);
        // The suffix image stays sequential, so changing a suffix rule must
        // not invalidate the first prompt's 22-row anchor.
        let spans: Vec<_> = rule
            .into_iter()
            .map(|grid| PositionSpanV1 {
                prompt_row: 7,
                rows: 12,
                grid,
            })
            .collect();
        if !spans.is_empty() {
            let prompt = if id == 1 { 22 } else { 29 };
            engine
                .executor_mut()
                .expected
                .insert(RequestId::new(id), reference(prompt, &spans, prompt + 19));
        }
        engine
            .submit(request.with_position_axes(spans).unwrap())
            .unwrap();
        for event in drain(&mut engine) {
            if let EngineEventKind::PromptCached { tokens } = event {
                assert_eq!(id, 2);
                cached = tokens;
            }
        }
    }
    assert_eq!(
        engine.executor().encoded,
        2,
        "position rules must not change encoder cache identity"
    );
    cached
}

#[test]
fn prefix_identity_separates_sequential_and_grid_positions() {
    for (first, next) in [(Some([1, 3, 4]), None), (None, Some([1, 3, 4]))] {
        assert!(
            cached_extension(first, next) <= 7,
            "reused image state across position rules"
        );
    }
}

#[test]
fn prefix_identity_separates_equal_product_position_grids() {
    assert!(
        cached_extension(Some([1, 3, 4]), Some([1, 2, 6])) <= 7,
        "reused image state across different position grids"
    );
}

#[test]
fn prefix_identity_preserves_unchanged_rule_extensions() {
    assert_eq!(cached_extension(Some([1, 3, 4]), Some([1, 3, 4])), 22);
    assert_eq!(cached_extension(None, None), 22);
}
