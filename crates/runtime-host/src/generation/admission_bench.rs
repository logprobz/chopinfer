//! Engine-thread admission cost of one long prompt, on CPU.
//!
//! Serves one 94,080-token prompt at a time through the backend's own
//! submission, worker and relay, over an executor that samples one text
//! token per row and keeps no prefix state, with prefix caching on, so every
//! round after the first is a prefix hit as the served admission probe is.
//! Each round asks for one greedy token with logprobs, without and then
//! with returned token IDs, which echo the prompt. Run it with the step
//! trace on and read each admitting step's `admit_*` fields from stderr; it
//! prints one `ADMISSION_ROUND` line per round, in the same order, to stdout.
//!
//! `CHOPINFER_STEP_TRACE=1 cargo test --release -p runtime-host --lib
//! long_prompt_admission_bench -- --ignored --nocapture`.
//!
//! `CHOPINFER_ADMISSION_COMPONENTS` names a directory holding a package's
//! `frontend.json`, `tokenizer.json` and `tokenizer_config.json`, to serve
//! that declaration and tokenizer instead of the test fixture's.
use super::stop_tests::{declarations, tokenizer_with};
use super::*;
use runtime_engine::{
    EngineConfig, ModelBatchItem, ModelTransaction, NoopModelTransactionState, SampledRow,
};
use runtime_kv::{BlockId, KvGeometry, PreparedAppendBatch};
use runtime_sampling::TokenLogprob as SampleTokenLogprob;
use runtime_scheduler::SchedulerConfig;

const PROMPT_TOKENS: usize = 94_080;
const WARMUP_ROUNDS: usize = 3;
const MEASURED_ROUNDS: usize = 20;

/// Word pieces the fixture tokenizer adds after its byte tokens, so the
/// prompt averages several bytes a token, some of them multi-byte.
const PIECES: [&str; 12] = [
    " the",
    " model",
    " serves",
    " long",
    " prompts",
    " and",
    "\u{4e2d}\u{6587}",
    " caf\u{e9}",
    " \"quoted\"",
    "\n\n",
    " tokens",
    " admission",
];

/// Samples `token` for every row that asks for one.
struct AdmissionExecutor {
    token: u32,
}

impl ModelExecutor for AdmissionExecutor {
    type TransactionState<'a> = NoopModelTransactionState;

    fn execute<'a>(
        &'a mut self,
        items: &[ModelBatchItem],
        _: &PreparedAppendBatch,
    ) -> Result<ModelTransaction<Self::TransactionState<'a>>, runtime_engine::ExecutorError> {
        let sampled = items
            .iter()
            .filter(|item| item.sampling().is_some())
            .map(|item| {
                let top = vec![SampleTokenLogprob::new(self.token, -0.5)];
                SampledRow::new(item.request_id(), self.token, -0.5, top)
            })
            .collect();
        Ok(ModelTransaction::no_op(Vec::new()).with_sampled(sampled))
    }

    fn capture_prefix_state_v1(
        &mut self,
        _: RequestId,
        _: usize,
        _: &[BlockId],
    ) -> Result<(), runtime_engine::ExecutorError> {
        Ok(())
    }

    fn restore_prefix_state_v1(
        &mut self,
        _: RequestId,
        _: &[u32],
        _: &[BlockId],
    ) -> Result<(), runtime_engine::ExecutorError> {
        Ok(())
    }
}

/// The served tokenizer and policy, and the long prompt and sampled token.
fn deployment() -> (Tokenizer, EngineModelPolicyV1, Vec<u32>, u32) {
    if let Some(directory) = std::env::var_os("CHOPINFER_ADMISSION_COMPONENTS") {
        let directory = PathBuf::from(directory);
        let read = |name: &str| std::fs::read(directory.join(name)).unwrap();
        let descriptor =
            FrontendDescriptorV1::from_component_bytes(&read("frontend.json")).unwrap();
        let tokenizer = runtime_frontend::tokenizer_from_components_v1(
            &descriptor,
            &read("tokenizer.json"),
            &read("tokenizer_config.json"),
        )
        .unwrap();
        let paragraph = PIECES.concat()
            + " fn admit(prompt: &[u32]) -> Result<(), Error> { Ok(()) }\n\
               Long prompts are admitted on the engine thread before each mixed step. ";
        let text = paragraph.repeat(PROMPT_TOKENS / 16);
        let mut prompt = tokenizer.encode(&text, &[]).unwrap();
        assert!(prompt.len() >= PROMPT_TOKENS, "{}", prompt.len());
        prompt.truncate(PROMPT_TOKENS);
        let token = tokenizer.encode("a", &[]).unwrap()[0];
        let policy = EngineModelPolicyV1::from_declaration(&descriptor)
            .with_parser_source(BundleParserSourceV1::from_declaration(&descriptor).unwrap());
        return (tokenizer, policy, prompt, token);
    }
    // The fixture spells its tokens in the byte-level alphabet.
    let mut alphabet = ['\0'; 256];
    let mut unprintable = 256;
    for (byte, letter) in alphabet.iter_mut().enumerate() {
        let printable = matches!(byte, 33..=126 | 161..=172 | 174..=255);
        *letter = char::from_u32(if printable { byte as u32 } else { unprintable }).unwrap();
        unprintable += u32::from(!printable);
    }
    let spelled = PIECES.map(|piece| {
        piece
            .bytes()
            .map(|byte| alphabet[usize::from(byte)])
            .collect::<String>()
    });
    let tokenizer = tokenizer_with(&spelled.each_ref().map(String::as_str));
    assert_eq!(
        tokenizer.token_bytes(259 + 6).unwrap(),
        PIECES[6].as_bytes()
    );
    let mut descriptor =
        FrontendDescriptorV1::from_component_bytes(declarations::KEY_VALUE.as_bytes()).unwrap();
    descriptor.vocabulary.size = 259 + PIECES.len() as u32;
    descriptor.vocabulary.eos_token_id = 1;
    descriptor.vocabulary.stop_token_ids = vec![1];
    // The word pieces, with a byte token every seventh.
    let prompt = (0..PROMPT_TOKENS)
        .map(|index| {
            if index % 7 == 6 {
                u32::from(b'a') + 3 + (index % 26) as u32
            } else {
                259 + (index % PIECES.len()) as u32
            }
        })
        .collect();
    let policy = EngineModelPolicyV1::from_declaration(&descriptor);
    (tokenizer, policy, prompt, u32::from(b'x') + 3)
}

#[test]
#[ignore = "release CPU admission timing; needs CHOPINFER_STEP_TRACE=1"]
fn long_prompt_admission_bench() {
    assert!(!cfg!(debug_assertions), "run with --release");
    let (tokenizer, policy, prompt, token) = deployment();
    let vocabulary = policy.vocabulary();
    let block = 256;
    let maximum = PROMPT_TOKENS + 64;
    let config = EngineConfig::new(
        SchedulerConfig::new(4, 8, 8192, 8192, 1).unwrap(),
        KvGeometry::new(block, 4 * maximum.div_ceil(block)).unwrap(),
        vocabulary,
        maximum,
        64,
    )
    .unwrap()
    .with_prefix_cache(1);
    let engine = Engine::new(config, AdmissionExecutor { token }).unwrap();
    assert!(engine.traces_steps(), "set CHOPINFER_STEP_TRACE=1");
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(4)
        .enable_all()
        .build()
        .unwrap();
    let _guard = runtime.enter();
    let backend = EngineGenerationBackend::new(
        tokenizer,
        engine,
        EngineBackendConfig::new(16).unwrap(),
        policy,
    )
    .unwrap();
    let mut round = 0;
    for echo in [false, true] {
        for _ in 0..WARMUP_ROUNDS + MEASURED_ROUNDS {
            let request = PreparedRequest {
                request_id: format!("admission-{round}"),
                client_request_id: None,
                model: "bench".into(),
                prompt: PreparedPrompt::TokenIds(prompt.clone()),
                max_new_tokens: 1,
                stop: vec![],
                decoding: runtime_server::DecodingOptions::default(),
                sampling: SamplingParameters::new(0.0)
                    .unwrap()
                    .with_top_logprobs(1)
                    .unwrap(),
                seed: 1,
                logprobs: runtime_server::RequestedLogprobs {
                    enabled: true,
                    top: 1,
                    echo: false,
                    prompt_top: None,
                    return_token_ids: echo,
                },
                chat: None,
                constraint: None,
                cache_salt: Some("admission-bench".into()),
                cancellation: CancellationSignal::new(),
            };
            let (sender, mut receiver) = tokio::sync::mpsc::channel(64);
            let started = Instant::now();
            let (first_event, finished, prompt_event) = runtime.block_on(async {
                backend
                    .submit(request, EventSender::new(sender))
                    .await
                    .unwrap();
                let mut first_event = None;
                let mut prompt_event = None;
                loop {
                    let event = receiver.recv().await.expect("the request finishes");
                    first_event.get_or_insert_with(|| started.elapsed());
                    match event {
                        GenerationEvent::Prompt {
                            token_ids, tokens, ..
                        } => {
                            assert_eq!(token_ids.len(), PROMPT_TOKENS);
                            assert_eq!(tokens.len(), PROMPT_TOKENS);
                            prompt_event = Some(started.elapsed());
                        }
                        GenerationEvent::Finished { .. } => {
                            break (first_event.unwrap(), started.elapsed(), prompt_event);
                        }
                        GenerationEvent::Failed { message, .. } => panic!("{message}"),
                        _ => {}
                    }
                }
            });
            assert_eq!(prompt_event.is_some(), echo);
            println!(
                "ADMISSION_ROUND {}",
                serde_json::json!({
                    "round": round,
                    "echo": echo,
                    "measured": round % (WARMUP_ROUNDS + MEASURED_ROUNDS) >= WARMUP_ROUNDS,
                    "prompt_tokens": PROMPT_TOKENS,
                    "first_event_us": first_event.as_micros() as u64,
                    "finished_us": finished.as_micros() as u64,
                })
            );
            round += 1;
        }
    }
    backend.shutdown();
}
