use std::collections::BTreeMap;

use runtime_engine::{
    DraftSource, Engine, EngineConfig, EngineRequest, ExecutorError, LogitsRow, ModelBatchItem,
    ModelExecutor, ModelTransaction, ModelTransactionState, SpeculationConfig, SpeculationMode,
    StepStatus,
};
use runtime_kv::{KvGeometry, PreparedAppendBatch};
use runtime_model_api::ModelVocabulary;
use runtime_sampling::SamplingParameters;
use runtime_scheduler::{BatchPhase, RequestId, SchedulerConfig};

#[derive(Clone, Copy, PartialEq)]
enum Failure {
    None,
    Execute,
    Output,
    Commit,
}

struct State(bool);
impl ModelTransactionState for State {
    fn commit(&mut self) -> Result<(), ExecutorError> {
        if self.0 {
            Err(ExecutorError::new("injected commit failure").unwrap())
        } else {
            Ok(())
        }
    }
    fn rollback(&mut self) -> Result<(), ExecutorError> {
        Ok(())
    }
}

struct Model {
    seen: Vec<(usize, usize, usize)>,
    failure: Failure,
}
impl ModelExecutor for Model {
    type TransactionState<'a> = State;

    fn step_rows_v1(&self, rows: usize) -> usize {
        rows.next_power_of_two()
    }

    fn round_draft_tokens_v1(&self, proposed: usize) -> usize {
        proposed.min(3)
    }
    fn draft_tokens_v1(
        &mut self,
        _: RequestId,
        _: &[u32],
        maximum: usize,
    ) -> Result<Vec<u32>, ExecutorError> {
        // The host drafter realizes fewer drafts than the rounded request.
        Ok(vec![1; maximum.min(2)])
    }
    fn execute<'a>(
        &'a mut self,
        items: &[ModelBatchItem],
        _: &PreparedAppendBatch,
    ) -> Result<ModelTransaction<Self::TransactionState<'a>>, ExecutorError> {
        let prefill = items
            .iter()
            .any(|item| matches!(item.phase(), BatchPhase::Prefill { .. }));
        let mut decode = 0;
        let mut verify = 0;
        for item in items
            .iter()
            .filter(|item| !matches!(item.phase(), BatchPhase::Prefill { .. }))
        {
            match item.draft() {
                Some(draft) => verify += draft.tokens() + 1,
                None => decode += 1,
            }
        }
        let capacity = (decode + verify).next_power_of_two();
        if !prefill {
            self.seen.push((decode, verify, capacity));
            if self.failure == Failure::Execute {
                return Err(ExecutorError::new("injected execution failure").unwrap());
            }
        }
        let mut rows = Vec::new();
        if prefill || self.failure != Failure::Output {
            for item in items.iter().filter(|item| item.expects_logits()) {
                for _ in 0..item.draft().map_or(1, |draft| draft.tokens() + 1) {
                    // Every host draft is wrong, but every verified row ran.
                    rows.push(
                        LogitsRow::new(item.request_id(), vec![0.0, 0.0, 10.0, 0.0]).unwrap(),
                    );
                }
            }
        }
        Ok(
            ModelTransaction::new(rows, State(!prefill && self.failure == Failure::Commit))
                .with_row_capacity(Some(capacity)),
        )
    }
}

fn engine(failure: Failure) -> Engine<Model> {
    let vocabulary = ModelVocabulary::new(4).unwrap();
    let config = EngineConfig::new(
        SchedulerConfig::new(2, 16, 16, 4, 1).unwrap(),
        KvGeometry::new(4, 32).unwrap(),
        vocabulary,
        128,
        32,
    )
    .unwrap()
    .with_speculation(
        SpeculationConfig::new(7, 16, SpeculationMode::Fixed, DraftSource::Host).unwrap(),
    );
    Engine::new(
        config,
        Model {
            seen: Vec::new(),
            failure,
        },
    )
    .unwrap()
}

fn request(id: u64, tokens: usize) -> EngineRequest {
    EngineRequest::new(
        RequestId::new(id),
        vec![0],
        tokens,
        SamplingParameters::new(0.0).unwrap(),
        id,
        Vec::new(),
        ModelVocabulary::new(4).unwrap(),
    )
    .unwrap()
}

#[test]
fn row_histogram_counts_realized_drafts_rejected_rows_and_ar_steps() {
    let mut engine = engine(Failure::None);
    engine.submit(request(1, 2)).unwrap();
    engine.submit(request(2, 10)).unwrap();
    let mut idle = false;
    for _ in 0..30 {
        if engine.step().unwrap() == StepStatus::Idle {
            idle = true;
            break;
        }
    }
    assert!(idle);
    let seen = &engine.executor().seen;
    assert!(seen.contains(&(1, 3, 4)), "{seen:?}");
    assert!(
        seen.contains(&(1, 0, 1)),
        "AR tail must be counted: {seen:?}"
    );
    let mut expected = BTreeMap::new();
    for &key in seen {
        *expected.entry(key).or_insert(0) += 1;
    }
    let text = engine.metrics().prometheus_compatibility("fixture");
    for ((decode, verify, capacity), count) in &expected {
        assert!(text.contains(&format!(
            "chop_step_rows_total{{model_name=\"fixture\",engine=\"0\",decode_rows=\"{decode}\",verify_rows=\"{verify}\",row_capacity=\"{capacity}\"}} {count}"
        )), "{text}");
    }
    assert_eq!(
        text.lines()
            .filter(|line| line.starts_with("chop_step_rows_total{"))
            .count(),
        expected.len()
    );
    let metrics = engine.metrics().snapshot();
    assert_eq!(metrics.spec_accepted_tokens, 0);
    assert_eq!(
        metrics.spec_rows,
        seen.iter()
            .map(|(_, verify, _)| *verify as u64)
            .sum::<u64>()
    );
}

#[test]
fn failed_executions_output_validation_and_commits_publish_no_row_bins() {
    for failure in [Failure::Execute, Failure::Output, Failure::Commit] {
        let mut engine = engine(failure);
        engine.submit(request(1, 10)).unwrap();
        engine.step().unwrap(); // Prefill never enters this histogram.
        assert!(engine.step().is_err());
        let text = engine.metrics().prometheus_compatibility("fixture");
        assert!(!text.contains("chop_step_rows_total{"), "{text}");
    }
}

#[test]
fn prompt_lookup_forwards_executor_buckets() {
    let wrapped = runtime_engine::PromptLookupExecutor::new(
        Model {
            seen: Vec::new(),
            failure: Failure::None,
        },
        runtime_engine::PromptLookupConfig::default(),
    );
    assert_eq!(wrapped.round_draft_tokens_v1(7), 3);
    assert_eq!(wrapped.step_rows_v1(5), 8);
}
