use super::*;
use runtime_engine::{Engine, EngineConfig, EngineEventKind, EngineRequest};
use runtime_model_api::ModelVocabulary;
use runtime_scheduler::SchedulerConfig;

mod health;

struct CollectedSampler;

impl DeviceLogitsSamplerV1 for CollectedSampler {
    fn supports_device_results(&self) -> bool {
        true
    }

    fn sample_view(
        &self,
        _view: OwnedLogitsViewV1,
        _items: &[&ModelBatchItem],
    ) -> HostServiceResultV1<Vec<SampledRow>> {
        panic!("the execution fixture emits owned logits")
    }
}

struct Execution {
    envelopes: Vec<Vec<u8>>,
    output: Vec<(u32, u64)>,
}

fn execute(device_results: bool, chunk: usize, prompt_tokens: usize) -> Execution {
    let mut instance = FakeExecutionV1::emitting(Vec::new());
    instance.prefill_chunk = chunk as u32;
    instance.dynamic_rows = true;
    instance.device_results = device_results;
    let vocabulary = ModelVocabulary::new(VOCABULARY).unwrap();
    let geometry = KvGeometry::new(64, 1024).unwrap();
    let mut config = EngineConfig::new(
        SchedulerConfig::new(1, 1, 8192, 2048, 1).unwrap(),
        geometry,
        vocabulary,
        geometry.token_capacity(),
        1,
    )
    .unwrap();
    if chunk > 0 {
        config = config.with_guaranteed_prefill_chunks(chunk).unwrap();
    }
    let executor = BundleModelExecutorV1::new(instance, VOCABULARY, 1)
        .with_view_sampler(Rc::new(CollectedSampler));
    assert_eq!(executor.device_results_v1(), device_results);
    let mut engine = Engine::new(config, executor).unwrap();
    engine
        .submit(
            EngineRequest::new(
                RequestId::new(1),
                vec![0; prompt_tokens],
                2,
                SamplingParameters::new(0.0).unwrap(),
                1,
                Vec::new(),
                vocabulary,
            )
            .unwrap(),
        )
        .unwrap();
    let prefill_steps = prompt_tokens.div_ceil(if chunk == 0 { 8192 } else { chunk });
    for _ in 0..prefill_steps + 1 {
        engine.step().unwrap();
    }
    let output = engine
        .drain_events()
        .iter()
        .filter_map(|event| match event.kind() {
            EngineEventKind::Token {
                token_id, logprob, ..
            } => Some((*token_id, logprob.to_bits())),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(output.len(), 2);
    let instance = &engine.executor().instance;
    assert_eq!(
        instance.envelopes.len(),
        if chunk == 0 { 0 } else { prefill_steps }
    );
    for (index, encoded) in instance.envelopes.iter().enumerate() {
        let envelope = runtime_bundle_sdk::prefill::decode(encoded).unwrap();
        assert_eq!(envelope.boundaries.len(), 1);
        assert_eq!(
            envelope.boundaries[0].chunk_end,
            ((index + 1) * chunk).min(prompt_tokens) as u64
        );
        assert_eq!(
            envelope.batch.items()[0].expects_logits(),
            index + 1 == prefill_steps
        );
    }
    assert!(
        instance.device_metadata[..prefill_steps]
            .iter()
            .all(|on| *on == (device_results && chunk == 0))
    );
    assert_eq!(instance.device_metadata[prefill_steps], device_results);
    Execution {
        envelopes: instance.envelopes.clone(),
        output,
    }
}

#[test]
fn collected_results_keep_terminal_and_nonterminal_prefill_negotiated() {
    for chunk in [2048, 8192] {
        for prompt in [17, chunk * 2 + 17] {
            let legacy = execute(false, chunk, prompt);
            let collected = execute(true, chunk, prompt);
            assert_eq!(
                collected.envelopes, legacy.envelopes,
                "prefill envelopes changed"
            );
            assert_eq!(
                collected.output, legacy.output,
                "output token or logprob bits changed"
            );
        }
    }
}

#[tokio::test]
async fn health_prefill_options_match_the_applied_terminal_dispatch() {
    use runtime_bundle::model_options::{
        CAPABILITY, ModelOptionV1, ModelOptionsV1, OptionDefinitionV1,
    };
    let declared = ModelOptionsV1 {
        schema: CAPABILITY.into(),
        options: vec![
            ModelOptionV1 {
                name: "prefill_chunk_tokens".into(),
                definition: OptionDefinitionV1::Integer {
                    default: 2048,
                    minimum: 0,
                    maximum: 8192,
                },
                default_is_qualified: true,
            },
            ModelOptionV1 {
                name: "prefill_upper_skip".into(),
                definition: OptionDefinitionV1::Boolean { default: true },
                default_is_qualified: true,
            },
        ],
    };
    for enabled in [false, true] {
        let options = declared
            .resolve(&BTreeMap::from([
                (
                    "prefill_chunk_tokens".into(),
                    if enabled { "2048" } else { "0" }.into(),
                ),
                ("prefill_upper_skip".into(), enabled.to_string()),
            ]))
            .unwrap();
        let reported = health::read(options.clone()).await;
        assert_eq!(reported["status"], "ready");
        assert_eq!(
            reported["serving"]["model_options"],
            serde_json::to_value(&options).unwrap()
        );
        let chunk = options["prefill_chunk_tokens"].as_u64().unwrap() as usize;
        let legacy = execute(false, chunk, 17);
        let collected = execute(true, chunk, 17);
        assert_eq!(legacy.envelopes, collected.envelopes);
        assert_eq!(legacy.output, collected.output);
        assert_eq!(
            reported["serving"]["model_options"]["prefill_upper_skip"],
            !collected.envelopes.is_empty(),
            "health must match the negotiated prefill schedule, including with device results"
        );
    }
}
