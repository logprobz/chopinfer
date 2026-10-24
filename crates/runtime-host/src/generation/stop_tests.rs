use super::*;
use runtime_engine::{
    DraftRow, DraftSource, EngineConfig, LogitsRow, ModelBatchItem, ModelTransaction,
    NoopModelTransactionState, SampledRow, SpeculationConfig, SpeculationMode,
};
use runtime_kv::{KvGeometry, PreparedAppendBatch};
use runtime_sampling::{SamplingParameters, sample_for_vocabulary};
use runtime_scheduler::SchedulerConfig;
use runtime_tokenizer::{
    BaseTokenRepresentation, ByteLevelPolicy, ConfigFieldPolicy, ConfigTokenRepresentation,
    ConfigTokenValue, FileIdentity, IgnoreMergesPolicy, MergeRecordPolicy, MergeResultIdPolicy,
    NormalizerPolicy, TokenizerContract, TokenizerDialect,
};
use runtime_weights::sha256_bytes;
use serde_json::json;

#[allow(dead_code)]
#[path = "../../../frontend-format/tests/common/mod.rs"]
pub(crate) mod declarations;

const EOS: u32 = 1;
const X: u32 = b'x' as u32 + 3;

#[test]
fn prompt_special_token_refusals_are_distinct_from_other_tokenizer_errors() {
    let mut descriptor =
        FrontendDescriptorV1::from_component_bytes(declarations::KEY_VALUE.as_bytes()).unwrap();
    descriptor.vocabulary.size = 259;
    descriptor.vocabulary.eos_token_id = EOS;
    descriptor.vocabulary.stop_token_ids = vec![EOS];
    descriptor.vocabulary.prompt_special_tokens = vec!["[BOS]".to_owned()];
    let backend = EngineGenerationBackend::new(
        tokenizer(),
        engine(vec![X], false, false),
        EngineBackendConfig::new(1)
            .unwrap()
            .with_tokenizer_input_bytes(64),
        EngineModelPolicyV1::from_declaration(&descriptor),
    )
    .unwrap();
    for text in ["[EOS]", &"x".repeat(65)] {
        let expected = if text == "[EOS]" {
            AdmissionError::InvalidParameter {
                message:
                    "prompt contains a special token the model does not admit in prompts: [EOS]"
                        .into(),
                parameter: "prompt",
            }
        } else {
            AdmissionError::InvalidPrompt
        };
        assert_eq!(backend.tokenize(text, false), Err(expected.clone()));
        assert_eq!(
            backend.prepare_prompt(PreparedPrompt::Raw(text.into())),
            Err(expected)
        );
    }
    assert!(backend.tokenize("[BOS]hello", false).is_ok());
}

pub(crate) fn tokenizer() -> Tokenizer {
    tokenizer_with(&[])
}

/// The test tokenizer with `extra_tokens` after its 259, numbered in order.
pub(super) fn tokenizer_with(extra_tokens: &[&str]) -> Tokenizer {
    let size = 259 + extra_tokens.len() as u32;
    let mut vocab = serde_json::Map::new();
    let specials = ["[BOS]", "[EOS]", "[PAD]"];
    for (id, text) in specials.iter().enumerate() {
        vocab.insert((*text).into(), json!(id));
    }
    let mut extra = 256;
    for byte in 0..256u32 {
        let encoded = if (33..=126).contains(&byte)
            || (161..=172).contains(&byte)
            || (174..=255).contains(&byte)
        {
            byte
        } else {
            let code = extra;
            extra += 1;
            code
        };
        vocab.insert(
            char::from_u32(encoded).unwrap().to_string(),
            json!(byte + 3),
        );
    }
    for (index, text) in extra_tokens.iter().enumerate() {
        vocab.insert((*text).to_owned(), json!(259 + index));
    }
    let added: Vec<_> = specials.iter().enumerate().map(|(id, text)| json!({"id":id,"content":text,"single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true})).collect();
    let definition = serde_json::to_vec(&json!({
        "version":"1.0", "truncation":null,"padding":null,"added_tokens":added,
        "normalizer":{"type":"Sequence","normalizers":[]},
        "pre_tokenizer":{"type":"Sequence","pretokenizers":[{"type":"Split","pattern":{"Regex":".+"},"behavior":"Isolated","invert":false},{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":true,"use_regex":false}]},
        "post_processor":{"type":"ByteLevel","add_prefix_space":true,"trim_offsets":false,"use_regex":true},
        "decoder":{"type":"ByteLevel","add_prefix_space":true,"trim_offsets":true,"use_regex":true},
        "model":{"type":"BPE","dropout":null,"unk_token":null,"continuing_subword_prefix":null,"end_of_word_suffix":null,"fuse_unk":false,"byte_fallback":false,"vocab":vocab,"merges":[]}
    })).unwrap();
    let special = |text: &str| json!({"__type":"AddedToken","content":text,"lstrip":false,"normalized":true,"rstrip":false,"single_word":false});
    let configuration = serde_json::to_vec(&json!({"add_bos_token":false,"add_eos_token":false,"bos_token":special("[BOS]"),"eos_token":special("[EOS]"),"pad_token":special("[EOS]"),"clean_up_tokenization_spaces":false,"legacy":true,"model_max_length":1024,"sp_model_kwargs":{},"tokenizer_class":"PreTrainedTokenizerFast","unk_token":null})).unwrap();
    let identity = |name, bytes: &[u8]| {
        FileIdentity::new(name, bytes.len() as u64, &sha256_bytes(bytes)).unwrap()
    };
    let contract = TokenizerContract::new_with_dialect(
        "test/model",
        "0123456789abcdef0123456789abcdef01234567",
        identity("tokenizer.json", &definition),
        identity("tokenizer_config.json", &configuration),
        size,
        size,
        0,
        3,
        1024,
        Some("[BOS]"),
        "[EOS]",
        "[EOS]",
        &[".+"],
        &["[BOS]"],
        // The byte tokens follow the three specials; extras follow them.
        literal_prefix_policy(size, 3),
    )
    .unwrap();
    Tokenizer::from_verified_bytes(&definition, &configuration, &contract).unwrap()
}

/// Literal special tokens below the byte tokens, and merges whose results
/// follow each other from one first identifier.
fn literal_prefix_policy(first_merge_id: u32, byte_start_id: u32) -> TokenizerDialect {
    TokenizerDialect {
        normalizer: NormalizerPolicy::IdentitySequence,
        pretokenizer_byte_level: ByteLevelPolicy {
            add_prefix_space: false,
            trim_offsets: true,
            use_regex: false,
        },
        post_processor_byte_level: ByteLevelPolicy {
            add_prefix_space: true,
            trim_offsets: false,
            use_regex: true,
        },
        decoder_byte_level: ByteLevelPolicy {
            add_prefix_space: true,
            trim_offsets: true,
            use_regex: true,
        },
        merge_results: MergeResultIdPolicy::Sequential {
            first_id: first_merge_id,
        },
        merge_records: MergeRecordPolicy::SpaceSeparatedString,
        base_tokens: BaseTokenRepresentation::LiteralPrefixThenByteLevel { byte_start_id },
        config_tokens: ConfigTokenRepresentation::AddedToken { normalized: true },
        add_bos_token: ConfigFieldPolicy::Required(false),
        bos_token: ConfigFieldPolicy::Required(ConfigTokenValue::ContractValue),
        unk_token: ConfigFieldPolicy::RequiredNull,
        tokenizer_class: "PreTrainedTokenizerFast".to_owned(),
        continuing_subword_prefix: None,
        end_of_word_suffix: None,
        ignore_merges: IgnoreMergesPolicy::Absent,
        add_eos_token: Some(false),
        legacy: Some(true),
        empty_sentencepiece_kwargs: true,
    }
}

/// Runs before each batch the scripted model executes.
pub(super) type Stall = Arc<dyn Fn(&[ModelBatchItem]) + Send + Sync>;

#[derive(Default)]
pub(super) struct ScriptedModel {
    batches: Option<Arc<Mutex<Vec<usize>>>>,
    tokens: Vec<u32>,
    sampled: bool,
    offload: bool,
    transfers: BTreeMap<runtime_kv::KvTransferId, runtime_kv::KvTransferSubmission>,
    completion_mode: Arc<AtomicU8>,
    /// Runs before each batch executes, so a test can make it slow or hang.
    pub(super) stall: Option<Stall>,
}
impl ModelExecutor for ScriptedModel {
    type TransactionState<'a> = NoopModelTransactionState;
    fn kv_offload_info_v1(&self) -> Option<runtime_engine::KvOffloadInfo> {
        self.offload.then_some(runtime_engine::KvOffloadInfo {
            host_page_capacity: 64,
            minimum_tokens: 1,
            snapshot_pages: 1,
        })
    }
    fn capture_prefix_state_v1(
        &mut self,
        _: RequestId,
        _: usize,
        _: &[runtime_kv::BlockId],
    ) -> Result<(), runtime_engine::ExecutorError> {
        Ok(())
    }
    fn offload_pages_v1(
        &mut self,
        plan: runtime_kv::KvTransferPlan,
    ) -> runtime_engine::KvTransferSubmitResult {
        self.transfers.insert(plan.id(), plan.into_submission());
        Ok(())
    }
    fn poll_kv_transfer_v1(
        &mut self,
        id: runtime_kv::KvTransferId,
    ) -> Result<runtime_engine::KvTransferPoll, runtime_engine::ExecutorError> {
        let status = match self.completion_mode.load(Ordering::Relaxed) {
            0 => return Ok(runtime_engine::KvTransferPoll::Pending),
            1 => runtime_kv::KvTransferStatus::Complete,
            _ => runtime_kv::KvTransferStatus::Unconfirmed,
        };
        Ok(runtime_engine::KvTransferPoll::Finished(
            self.transfers[&id].completion(status),
        ))
    }
    fn execute<'a>(
        &'a mut self,
        items: &[ModelBatchItem],
        _: &PreparedAppendBatch,
    ) -> Result<ModelTransaction<Self::TransactionState<'a>>, runtime_engine::ExecutorError> {
        if let Some(stall) = &self.stall {
            stall(items);
        }
        if let Some(batches) = &self.batches {
            batches.lock().unwrap().push(items.len());
        }
        let mut logits_rows = Vec::new();
        let mut sampled_rows = Vec::new();
        let mut draft_rows = Vec::new();
        for item in items {
            let Some(work) = item.sampling() else {
                continue;
            };
            let start = work.emitted_tokens;
            let drafts: Vec<_> = (0..item.draft().map_or(0, |draft| draft.tokens()))
                .map(|row| *self.tokens.get(start + row).unwrap_or(&EOS))
                .collect();
            if item.draft().is_some() {
                draft_rows.push(DraftRow::new(item.request_id(), drafts.clone()));
            }
            for row in 0..=drafts.len() {
                let mut logits = vec![0.0; 259];
                logits[*self.tokens.get(start + row).unwrap_or(&EOS) as usize] = 10.0;
                if self.sampled {
                    let row_work = work.for_verify_row(&drafts, row);
                    let context = row_work
                        .history
                        .context(row_work.emitted_tokens, &row_work.stop_ids);
                    let mask = row_work
                        .constraint
                        .as_ref()
                        .and_then(|constraint| constraint.mask());
                    let context = if let Some(mask) = mask {
                        context.with_allowed_tokens(mask)
                    } else {
                        context
                    };
                    let mut state = row_work.state;
                    let sample = sample_for_vocabulary(
                        &row_work.parameters,
                        &logits,
                        ModelVocabulary::new(259).unwrap(),
                        context,
                        &mut state,
                    )
                    .unwrap();
                    sampled_rows.push(SampledRow::new(
                        item.request_id(),
                        sample.token_id(),
                        sample.logprob(),
                        sample.top_logprobs().to_vec(),
                    ));
                } else {
                    logits_rows.push(LogitsRow::new(item.request_id(), logits)?);
                }
            }
        }
        Ok(ModelTransaction::no_op(logits_rows)
            .with_sampled(sampled_rows)
            .with_drafts(draft_rows))
    }
}

/// An engine whose model answers `tokens` after `stall` runs on each batch.
pub(super) fn stalling_engine(tokens: Vec<u32>, stall: Stall) -> Engine<ScriptedModel> {
    Engine::new(
        EngineConfig::new(
            SchedulerConfig::new(1, 32, 32, 4, 1).unwrap(),
            KvGeometry::new(4, 64).unwrap(),
            ModelVocabulary::new(259).unwrap(),
            128,
            64,
        )
        .unwrap(),
        ScriptedModel {
            tokens,
            stall: Some(stall),
            ..Default::default()
        },
    )
    .unwrap()
}

fn engine(tokens: Vec<u32>, sampled: bool, speculative: bool) -> Engine<ScriptedModel> {
    let config = EngineConfig::new(
        SchedulerConfig::new(1, 32, 32, 4, 1).unwrap(),
        KvGeometry::new(4, 64).unwrap(),
        ModelVocabulary::new(259).unwrap(),
        128,
        64,
    )
    .unwrap();
    let config = if speculative {
        config.with_speculation(
            SpeculationConfig::new(3, 32, SpeculationMode::Fixed, DraftSource::Bundle).unwrap(),
        )
    } else {
        config
    };
    Engine::new(
        config,
        ScriptedModel {
            tokens,
            sampled,
            ..Default::default()
        },
    )
    .unwrap()
}

fn active(
    minimum: usize,
    stop: &[String],
) -> (
    ActiveGeneration,
    tokio::sync::mpsc::Receiver<GenerationEvent>,
) {
    let descriptor = Arc::new(
        FrontendDescriptorV1::from_component_bytes(declarations::KEY_VALUE.as_bytes()).unwrap(),
    );
    let mut decoder = if stop.is_empty() {
        CompletionDecoderV1::new(
            Arc::clone(&descriptor),
            PromptModeV1::Chat,
            DecoderLimitsV1::declared(&descriptor),
        )
    } else {
        CompletionDecoderV1::with_client_stops(
            Arc::clone(&descriptor),
            PromptModeV1::Chat,
            DecoderLimitsV1::declared(&descriptor),
            stop,
        )
        .unwrap()
    };
    // A raw completion parses through the declared frontend, as admission
    // opens it.
    decoder.set_parser_source(
        runtime_frontend::BundleParserSourceV1::from_declaration(&descriptor).unwrap(),
    );
    decoder.set_decoding_options(Default::default(), true);
    decoder.set_minimum_new_tokens(minimum);
    let (sender, receiver) = tokio::sync::mpsc::channel(128);
    (
        ActiveGeneration {
            events: Outbox::new(EventSender::new(sender), CancellationSignal::new(), false),
            cancellation: CancellationSignal::new(),
            decoder,
            prompt_tokens: 1,
            cached_tokens: None,
            completion_tokens: 0,
            accounting: TokenAccounting::default(),
            logprob_context: [None; 4],
            prompt_logprob_context: [None; 4],
            logprobs_enabled: false,
            return_token_ids: false,
            admission: None,
            queued_since: Instant::now(),
            group_admitted: Rc::new(Cell::new(false)),
        },
        receiver,
    )
}

#[test]
fn grammar_stop_before_minimum_finishes_greedy_and_speculative_completions() {
    let tokenizer = tokenizer();
    for speculative in [false, true] {
        for sampled in [false, true] {
            for length in 1..=4 {
                let mut tokens = vec![X; length];
                tokens.push(EOS);
                let mut engine = engine(tokens, sampled, speculative);
                let vocabulary = Arc::new(
                    runtime_grammar::TokenVocabulary::new(
                        259,
                        (0..259)
                            .map(|id| runtime_grammar::VocabularyToken {
                                bytes: tokenizer.token_bytes(id).unwrap().to_vec(),
                                special: id < 3,
                            })
                            .collect(),
                    )
                    .unwrap(),
                );
                let grammar = Arc::new(
                    CompiledGrammar::new(
                        &runtime_grammar::regex_grammar(&"x".repeat(length)).unwrap(),
                        vocabulary,
                    )
                    .unwrap(),
                );
                let id = RequestId::new(1);
                engine
                    .submit(
                        EngineRequest::new(
                            id,
                            vec![X],
                            20,
                            SamplingParameters::new(0.0)
                                .unwrap()
                                .with_minimum_new_tokens(10)
                                .unwrap(),
                            0,
                            vec![EOS],
                            ModelVocabulary::new(259).unwrap(),
                        )
                        .unwrap()
                        .with_constraint(ConstraintState::new(grammar, None, vec![EOS]))
                        .unwrap(),
                    )
                    .unwrap();
                let (mut generation, mut receiver) = active(10, &[]);
                // Stop bytes must stay hidden even when other special tokens are visible.
                generation.decoder.set_decoding_options(
                    runtime_server::DecodingOptions {
                        skip_special_tokens: false,
                        ..Default::default()
                    },
                    true,
                );
                let mut active = BTreeMap::from([(id, generation)]);
                for _ in 0..20 {
                    engine.step().unwrap();
                    for event in engine.drain_events() {
                        assert!(handle_engine_event(
                            &tokenizer,
                            &mut engine,
                            &mut active,
                            event
                        ));
                    }
                    if active.is_empty() {
                        break;
                    }
                }
                let mut finished = Vec::new();
                let mut text = String::new();
                while let Ok(event) = receiver.try_recv() {
                    match event {
                        GenerationEvent::Content(content) => text.push_str(&content.delta),
                        GenerationEvent::Finished { reason, usage } => {
                            finished.push((reason, usage.completion_tokens))
                        }
                        GenerationEvent::Failed { .. } => panic!(
                            "valid grammar completion failed: speculative={speculative}, sampled={sampled}, length={length}"
                        ),
                        _ => {}
                    }
                }
                assert_eq!(finished, vec![(FinishReason::Stop, length as u64 + 1)]);
                assert_eq!(text, "x".repeat(length));
            }
        }
    }
}

#[test]
fn ignored_eos_does_not_become_an_engine_confirmed_stop() {
    let tokenizer = tokenizer();
    for (speculative, sampled) in [(false, false), (false, true), (true, false), (true, true)] {
        let mut engine = engine(vec![X, EOS, X, X], sampled, speculative);
        let id = RequestId::new(1);
        let vocabulary = Arc::new(
            runtime_grammar::TokenVocabulary::new(
                259,
                (0..259)
                    .map(|id| runtime_grammar::VocabularyToken {
                        bytes: tokenizer.token_bytes(id).unwrap().to_vec(),
                        special: id < 3,
                    })
                    .collect(),
            )
            .unwrap(),
        );
        let grammar = Arc::new(
            CompiledGrammar::new(&runtime_grammar::regex_grammar("x").unwrap(), vocabulary)
                .unwrap(),
        );
        engine
            .submit(
                EngineRequest::new(
                    id,
                    vec![X],
                    4,
                    SamplingParameters::new(0.0)
                        .unwrap()
                        .with_minimum_new_tokens(3)
                        .unwrap(),
                    0,
                    vec![],
                    ModelVocabulary::new(259).unwrap(),
                )
                .unwrap()
                .with_constraint(ConstraintState::new(grammar, None, vec![]))
                .unwrap(),
            )
            .unwrap();
        let (mut generation, mut receiver) = active(3, &[]);
        generation.decoder.set_decoding_options(
            runtime_server::DecodingOptions {
                ignore_eos: true,
                ..Default::default()
            },
            true,
        );
        let mut active = BTreeMap::from([(id, generation)]);
        for _ in 0..4 {
            engine.step().unwrap();
            for event in engine.drain_events() {
                assert!(handle_engine_event(
                    &tokenizer,
                    &mut engine,
                    &mut active,
                    event
                ));
            }
            if active.is_empty() {
                break;
            }
        }
        let mut finished = Vec::new();
        while let Ok(event) = receiver.try_recv() {
            match event {
                GenerationEvent::Finished { reason, usage } => {
                    finished.push((reason, usage.completion_tokens))
                }
                GenerationEvent::Failed { .. } => panic!("ignored EOS failed"),
                _ => {}
            }
        }
        assert_eq!(finished, vec![(FinishReason::Length, 4)]);
    }
}

fn client_stop_metrics(maximum: usize) {
    let tokenizer = tokenizer();
    for streamed in [false, true] {
        for sampled in [false, true] {
            let tokens = vec![X, b'E' as u32 + 3, b'N' as u32 + 3, b'D' as u32 + 3, X];
            let mut engine = engine(tokens, sampled, true);
            let id = RequestId::new(1);
            engine
                .submit(
                    EngineRequest::new(
                        id,
                        vec![X],
                        maximum,
                        SamplingParameters::new(0.0).unwrap(),
                        0,
                        vec![EOS],
                        ModelVocabulary::new(259).unwrap(),
                    )
                    .unwrap(),
                )
                .unwrap();
            engine.defer_completion_metrics(id);
            let (generation, mut receiver) = active(0, &["END".to_owned()]);
            let mut active = BTreeMap::from([(id, generation)]);
            let mut delivered = Vec::new();
            for _ in 0..4 {
                engine.step().unwrap();
                for event in engine.drain_events() {
                    assert!(handle_engine_event(
                        &tokenizer,
                        &mut engine,
                        &mut active,
                        event
                    ));
                    if streamed {
                        while let Ok(event) = receiver.try_recv() {
                            delivered.push(event);
                        }
                    }
                }
                if active.is_empty() {
                    break;
                }
            }
            while let Ok(event) = receiver.try_recv() {
                delivered.push(event);
            }
            let mut text = String::new();
            let mut finishes = Vec::new();
            for event in delivered {
                match event {
                    GenerationEvent::Content(content) => text.push_str(&content.delta),
                    GenerationEvent::Finished { reason, usage } => {
                        finishes.push((reason, usage.completion_tokens))
                    }
                    GenerationEvent::Failed { .. } => panic!("client stop failed"),
                    _ => {}
                }
            }
            assert_eq!(text, "x");
            assert_eq!(finishes, vec![(FinishReason::Stop, 4)]);
            let metrics = engine.metrics().prometheus_compatibility("fixture");
            for (reason, count) in [("stop", 1), ("length", 0), ("abort", 0)] {
                assert!(metrics.contains(&format!("vllm:request_success_total{{model_name=\"fixture\",engine=\"0\",finished_reason=\"{reason}\"}} {count}\n")), "{metrics}");
            }
            assert_eq!(engine.metrics().snapshot().cancellations, 0);
            assert_eq!(engine.metrics().snapshot().requests_completed, 1);
        }
    }
}

#[test]
fn client_stop_before_engine_budget_counts_as_success() {
    client_stop_metrics(20);
}

#[test]
fn client_stop_at_engine_budget_counts_as_success() {
    client_stop_metrics(5);
}

fn admitted_ignored_eos(
    script: Vec<u32>,
    minimum: usize,
    grammar_length: Option<usize>,
    sampled: bool,
    speculative: bool,
    explicit_stops: Vec<u32>,
) -> (Vec<u32>, EngineFinishReason) {
    let tokenizer = tokenizer();
    let mut descriptor =
        FrontendDescriptorV1::from_component_bytes(declarations::KEY_VALUE.as_bytes()).unwrap();
    descriptor.vocabulary.size = 259;
    descriptor.vocabulary.eos_token_id = EOS;
    descriptor.vocabulary.stop_token_ids = vec![EOS];
    let policy = EngineModelPolicyV1::from_declaration(&descriptor);
    let grammar = grammar_length.map(|length| {
        let vocabulary = Arc::new(
            runtime_grammar::TokenVocabulary::new(
                259,
                (0..259)
                    .map(|id| runtime_grammar::VocabularyToken {
                        bytes: tokenizer.token_bytes(id).unwrap().to_vec(),
                        special: id < 3,
                    })
                    .collect(),
            )
            .unwrap(),
        );
        Arc::new(
            CompiledGrammar::new(
                &runtime_grammar::regex_grammar(&"x".repeat(length)).unwrap(),
                vocabulary,
            )
            .unwrap(),
        )
    });
    let request = PreparedRequest {
        request_id: "ignored-eos".into(),
        client_request_id: None,
        model: "fixture/alpha-1".into(),
        prompt: PreparedPrompt::TokenIds(vec![X]),
        max_new_tokens: 8,
        stop: vec![],
        decoding: runtime_server::DecodingOptions {
            ignore_eos: true,
            stop_token_ids: explicit_stops,
            ..Default::default()
        },
        sampling: SamplingParameters::new(0.0)
            .unwrap()
            .with_minimum_new_tokens(minimum)
            .unwrap(),
        seed: 7,
        logprobs: runtime_server::RequestedLogprobs {
            enabled: false,
            top: 0,
            echo: false,
            prompt_top: None,
            return_token_ids: false,
        },
        chat: None,
        constraint: None,
        cache_salt: None,
        cancellation: CancellationSignal::new(),
    };
    let mut engine = engine(script, sampled, speculative);
    let mut active = BTreeMap::new();
    let (sender, mut receiver) = tokio::sync::mpsc::channel(128);
    let (response, mut admission) = oneshot::channel();
    admit(
        &tokenizer,
        &mut engine,
        &mut active,
        RequestId::new(1),
        request,
        grammar,
        EventSender::new(sender),
        response,
        65536,
        &policy,
        Delivery::Immediate,
    );
    assert_eq!(active.len(), 1, "request must reach engine admission");
    let mut tokens = Vec::new();
    let mut finish = None;
    for _ in 0..16 {
        engine.step().unwrap();
        if let Some(response) = active
            .get_mut(&RequestId::new(1))
            .and_then(|generation| generation.admission.take())
        {
            response.send(Ok(())).unwrap();
        }
        for event in engine.drain_events() {
            match event.kind() {
                EngineEventKind::Token { token_id, .. } => tokens.push(*token_id),
                EngineEventKind::Finished { reason } => finish = Some(*reason),
                _ => {}
            }
            assert!(handle_engine_event(
                &tokenizer,
                &mut engine,
                &mut active,
                event
            ));
        }
        if active.is_empty() {
            break;
        }
    }
    assert!(admission.try_recv().unwrap().is_ok());
    let mut delivered_finish = false;
    while let Ok(event) = receiver.try_recv() {
        match event {
            GenerationEvent::Finished { usage, .. } => {
                assert_eq!(usage.completion_tokens as usize, tokens.len());
                delivered_finish = true;
            }
            GenerationEvent::Failed { .. } => panic!("ignored EOS admission failed"),
            _ => {}
        }
    }
    assert!(delivered_finish);
    (tokens, finish.unwrap())
}

#[test]
fn ignored_eos_still_suppresses_minimum_tokens_through_admission() {
    for speculative in [false, true] {
        for sampled in [false, true] {
            let (tokens, reason) =
                admitted_ignored_eos(vec![EOS; 8], 5, None, sampled, speculative, vec![]);
            assert_eq!(tokens, [0, 0, 0, 0, 0, EOS, EOS, EOS]);
            assert_eq!(reason, EngineFinishReason::Length);
            let (tokens, reason) =
                admitted_ignored_eos(vec![X; 8], 0, None, sampled, speculative, vec![X]);
            assert_eq!(tokens, [X]);
            assert_eq!(reason, EngineFinishReason::Stop);
        }
    }
}

#[test]
fn ignored_eos_requires_the_grammar_terminal_transition_through_admission() {
    for speculative in [false, true] {
        for sampled in [false, true] {
            // Four x tokens finish the grammar just before the bonus row.
            for length in [1, 4] {
                let (tokens, reason) =
                    admitted_ignored_eos(vec![X; 8], 7, Some(length), sampled, speculative, vec![]);
                let mut expected = vec![X; 8];
                expected[length] = EOS;
                assert_eq!(tokens, expected);
                assert_eq!(reason, EngineFinishReason::Length);
            }
        }
    }
}

#[test]
fn textual_eos_stays_literal_in_gathered_owned_and_verify_outputs() {
    let tokenizer = tokenizer();
    for sequences in [1, 16] {
        for sampled in [false, true] {
            for speculative in [false, true] {
                let config = EngineConfig::new(
                    SchedulerConfig::new(sequences, 128, 128, 4, 1).unwrap(),
                    KvGeometry::new(4, 256).unwrap(),
                    ModelVocabulary::new(259).unwrap(),
                    512,
                    256,
                )
                .unwrap();
                let config = if speculative {
                    config.with_speculation(
                        SpeculationConfig::new(3, 128, SpeculationMode::Fixed, DraftSource::Bundle)
                            .unwrap(),
                    )
                } else {
                    config
                };
                // These are ordinary byte tokens, not the EOS ID. The frontend
                // must not turn their decoded spelling into an early stop.
                let tokens = b"[EOS]xxxxx"
                    .iter()
                    .map(|byte| u32::from(*byte) + 3)
                    .collect();
                let model = ScriptedModel {
                    tokens,
                    sampled,
                    ..Default::default()
                };
                let mut engine = Engine::new(config, model).unwrap();
                let mut active = BTreeMap::new();
                let mut receivers = Vec::new();
                for row in 0..sequences {
                    let minimum = if row % 2 == 0 { 10 } else { 0 };
                    let id = RequestId::new(row as u64 + 1);
                    engine
                        .submit(
                            EngineRequest::new(
                                id,
                                vec![X],
                                10,
                                SamplingParameters::new(0.0)
                                    .unwrap()
                                    .with_minimum_new_tokens(minimum)
                                    .unwrap(),
                                0,
                                vec![EOS],
                                ModelVocabulary::new(259).unwrap(),
                            )
                            .unwrap(),
                        )
                        .unwrap();
                    let (mut generation, receiver) = self::active(minimum, &[]);
                    generation
                        .decoder
                        .set_decoding_options(Default::default(), false);
                    active.insert(id, generation);
                    receivers.push(receiver);
                }
                for _ in 0..64 {
                    engine.step().unwrap();
                    for event in engine.drain_events() {
                        assert!(handle_engine_event(
                            &tokenizer,
                            &mut engine,
                            &mut active,
                            event
                        ));
                    }
                    if active.is_empty() {
                        break;
                    }
                }
                assert!(active.is_empty());
                for (row, receiver) in receivers.iter_mut().enumerate() {
                    let mut finished = Vec::new();
                    let mut content = String::new();
                    while let Ok(event) = receiver.try_recv() {
                        match event {
                            GenerationEvent::Content(part) => content.push_str(&part.delta),
                            GenerationEvent::Finished { reason, usage } => {
                                finished.push((reason, usage.completion_tokens))
                            }
                            GenerationEvent::Failed { .. } => panic!("textual EOS request failed"),
                            _ => {}
                        }
                    }
                    let expected = (FinishReason::Length, 10);
                    assert_eq!(
                        finished,
                        vec![expected],
                        "row={row}, sequences={sequences}, sampled={sampled}, speculative={speculative}"
                    );
                    assert_eq!(content, "[EOS]xxxxx");
                }
                if speculative {
                    assert!(engine.metrics().snapshot().spec_verify_items > 0);
                }
            }
        }
    }
}

#[test]
fn worker_idle_and_shutdown_drain_pending_copies_or_report_fatal() {
    for idle in [false, true] {
        for confirmed in [false, true] {
            let (mut engine, mode, id) = pending_copy_engine();
            mode.store(if confirmed { 1 } else { 2 }, Ordering::Relaxed);
            let metrics = engine.metrics();
            let exit = if idle {
                engine.cancel(id).unwrap();
                let (send, receive) = std::sync::mpsc::channel();
                send.send(WorkerCommand::Shutdown).unwrap();
                let descriptor =
                    FrontendDescriptorV1::from_component_bytes(declarations::KEY_VALUE.as_bytes())
                        .unwrap();
                worker_loop(
                    Arc::new(tokenizer()),
                    engine,
                    receive,
                    EngineBackendConfig::new(1)
                        .unwrap()
                        .with_queue_wait(Duration::from_secs(1))
                        .with_tokenizer_input_bytes(1024),
                    EngineModelPolicyV1::from_declaration(&descriptor),
                    &AtomicU8::new(READY),
                    &WorkerProgress::new(),
                )
            } else {
                let mut active_map = BTreeMap::new();
                let (generation, _events) = active(0, &[]);
                active_map.insert(id, generation);
                shutdown_worker(&mut engine, &mut active_map)
            };
            assert_eq!(matches!(exit, WorkerExit::Shutdown), confirmed);
            if confirmed {
                assert_eq!(metrics.snapshot().offload.used_pages, 0);
            } else {
                assert!(metrics.snapshot().offload.used_pages > 0);
            }
        }
    }
}

fn pending_copy_engine() -> (Engine<ScriptedModel>, Arc<AtomicU8>, RequestId) {
    pending_copy_engine_with_clock(Arc::new(runtime_engine::SystemMonotonicClock::default()))
}

fn pending_copy_engine_with_clock(
    clock: Arc<dyn runtime_engine::MonotonicClock>,
) -> (Engine<ScriptedModel>, Arc<AtomicU8>, RequestId) {
    pending_copy_engine_with_sequences(clock, 1)
}

fn pending_copy_engine_with_sequences(
    clock: Arc<dyn runtime_engine::MonotonicClock>,
    sequences: usize,
) -> (Engine<ScriptedModel>, Arc<AtomicU8>, RequestId) {
    let config = EngineConfig::new(
        SchedulerConfig::new(sequences, 32, 32, 4, 1).unwrap(),
        KvGeometry::new(4, 64).unwrap(),
        ModelVocabulary::new(259).unwrap(),
        128,
        64,
    )
    .unwrap()
    .with_admission(runtime_scheduler::AdmissionPolicy::Optimistic(
        runtime_scheduler::OptimisticAdmission::new(0, 3)
            .unwrap()
            .with_forced_preemption(1)
            .unwrap(),
    ))
    .with_forced_swaps();
    let mode = Arc::new(AtomicU8::new(0));
    let mut engine = Engine::new_with_observability(
        config,
        ScriptedModel {
            tokens: vec![X; 10],
            offload: true,
            completion_mode: mode.clone(),
            ..Default::default()
        },
        clock,
        EngineMetrics::default(),
    )
    .unwrap();
    let id = RequestId::new(1);
    engine
        .submit(
            EngineRequest::new(
                id,
                vec![X; 4],
                10,
                SamplingParameters::new(0.0).unwrap(),
                0,
                vec![],
                ModelVocabulary::new(259).unwrap(),
            )
            .unwrap(),
        )
        .unwrap();
    for _ in 0..10 {
        engine.step().unwrap();
        if !engine.executor().transfers.is_empty() {
            break;
        }
    }
    assert!(!engine.executor().transfers.is_empty());
    (engine, mode, id)
}

#[test]
fn idle_worker_prefills_grouped_prompts_and_finishes_a_slow_copy_without_new_commands() {
    let clock = Arc::new(TransferClock::default());
    let (mut engine, mode, id) = pending_copy_engine_with_sequences(clock.clone(), 3);
    engine.cancel(id).unwrap();
    let metrics = engine.metrics();
    let (send, receive) = std::sync::mpsc::channel();
    let client = std::thread::spawn(move || {
        // Exceed the former five-second idle poison deadline.
        std::thread::sleep(Duration::from_millis(5100));
        assert!(metrics.snapshot().offload.used_pages > 0);
        let request = PreparedRequest {
            request_id: "while-copying".into(),
            client_request_id: None,
            model: "fixture/alpha-1".into(),
            prompt: PreparedPrompt::TokenIds(vec![X]),
            max_new_tokens: 1,
            stop: vec![],
            decoding: Default::default(),
            sampling: SamplingParameters::new(0.0).unwrap(),
            seed: 0,
            logprobs: runtime_server::RequestedLogprobs {
                enabled: false,
                top: 0,
                echo: false,
                prompt_top: None,
                return_token_ids: false,
            },
            chat: None,
            constraint: None,
            cache_salt: None,
            cancellation: CancellationSignal::new(),
        };
        let before = metrics.snapshot();
        let mut submissions = Vec::new();
        let mut admissions = Vec::new();
        let mut receivers = Vec::new();
        for index in 0..3 {
            let mut choice = request.clone();
            choice.prompt = PreparedPrompt::TokenIds(vec![X + index as u32]);
            let (events, event_receiver) = tokio::sync::mpsc::channel(128);
            let (response, admission) = oneshot::channel();
            submissions.push(WorkerSubmission {
                id: RequestId::new(index + 2),
                request: Box::new(choice),
                constraint: None,
                events: EventSender::new(events),
                response,
            });
            admissions.push(admission);
            receivers.push(event_receiver);
        }
        send.send(WorkerCommand::Submit(submissions)).unwrap();
        let wait = |predicate: &dyn Fn() -> bool| {
            let deadline = Instant::now() + Duration::from_secs(3);
            while !predicate() {
                assert!(Instant::now() < deadline, "idle worker stopped progressing");
                std::thread::sleep(Duration::from_millis(1));
            }
        };
        for mut admission in admissions {
            let deadline = Instant::now() + Duration::from_secs(3);
            loop {
                match admission.try_recv() {
                    Ok(result) => {
                        result.unwrap();
                        break;
                    }
                    Err(oneshot::error::TryRecvError::Empty) if Instant::now() < deadline => {
                        std::thread::sleep(Duration::from_millis(1));
                    }
                    result => panic!("idle copies blocked grouped admission: {result:?}"),
                }
            }
        }
        wait(&|| metrics.snapshot().requests_completed == before.requests_completed + 3);
        let after = metrics.snapshot();
        assert_eq!(after.prefill_batch_items - before.prefill_batch_items, 3);
        assert_eq!(after.prefill_only_batches - before.prefill_only_batches, 1);
        assert_eq!(after.batches - before.batches, 1);
        assert!(after.offload.used_pages > 0);
        // Retain each choice's receiver until every admitted choice has finished.
        for receiver in &mut receivers {
            while receiver.try_recv().is_ok() {}
        }
        // No further command wakes the worker: it must keep polling while idle.
        clock.0.store(9_000_000, Ordering::Relaxed);
        wait(&|| metrics.snapshot().offload.oldest_pending_micros == 9_000_000);
        mode.store(1, Ordering::Relaxed);
        wait(&|| metrics.snapshot().offload.used_pages == 0);
        send.send(WorkerCommand::Shutdown).unwrap();
        metrics
    });
    let mut descriptor =
        FrontendDescriptorV1::from_component_bytes(declarations::KEY_VALUE.as_bytes()).unwrap();
    descriptor.vocabulary.size = 259;
    descriptor.vocabulary.eos_token_id = EOS;
    descriptor.vocabulary.stop_token_ids = vec![EOS];
    let exit = worker_loop(
        Arc::new(tokenizer()),
        engine,
        receive,
        EngineBackendConfig::new(1)
            .unwrap()
            .with_queue_wait(Duration::from_secs(1))
            .with_tokenizer_input_bytes(1024),
        EngineModelPolicyV1::from_declaration(&descriptor),
        &AtomicU8::new(READY),
        &WorkerProgress::new(),
    );
    let metrics = client.join().unwrap();
    assert!(matches!(exit, WorkerExit::Shutdown));
    assert_eq!(metrics.snapshot().offload.used_pages, 0);
}

#[derive(Debug, Default)]
struct TransferClock(AtomicU64);
impl runtime_engine::MonotonicClock for TransferClock {
    fn now_micros(&self) -> u64 {
        self.0.load(Ordering::Relaxed)
    }
}

#[test]
fn overdue_idle_copy_changes_readiness_once_and_recovers_after_completion() {
    let clock = Arc::new(TransferClock::default());
    let (mut engine, mode, id) = pending_copy_engine_with_clock(clock.clone());
    engine.cancel(id).unwrap();
    let metrics = engine.metrics();
    let mut descriptor =
        FrontendDescriptorV1::from_component_bytes(declarations::KEY_VALUE.as_bytes()).unwrap();
    descriptor.vocabulary.size = 259;
    descriptor.vocabulary.eos_token_id = EOS;
    descriptor.vocabulary.stop_token_ids = vec![EOS];
    let backend = EngineGenerationBackend::new(
        tokenizer(),
        engine,
        EngineBackendConfig::new(1).unwrap(),
        EngineModelPolicyV1::from_declaration(&descriptor),
    )
    .unwrap();
    let wait = |predicate: &dyn Fn() -> bool| {
        let deadline = Instant::now() + Duration::from_secs(3);
        while !predicate() {
            assert!(
                Instant::now() < deadline,
                "worker health transition did not arrive"
            );
            std::thread::sleep(Duration::from_millis(1));
        }
    };
    clock.0.store(9_000_000, Ordering::Relaxed);
    wait(&|| metrics.snapshot().offload.oldest_pending_micros == 9_000_000);
    assert_eq!(backend.readiness(), BackendReadiness::Ready);
    assert_eq!(metrics.snapshot().offload.stalls, 0);
    clock.0.store(11_000_000, Ordering::Relaxed);
    wait(&|| matches!(backend.readiness(), BackendReadiness::NotReady(_)));
    assert_eq!(metrics.snapshot().offload.stalled_transfers, 1);
    assert_eq!(metrics.snapshot().offload.stalls, 1);
    clock.0.store(20_000_000, Ordering::Relaxed);
    wait(&|| metrics.snapshot().offload.oldest_pending_micros == 20_000_000);
    assert_eq!(
        metrics.snapshot().offload.stalls,
        1,
        "one warning per transfer"
    );
    assert!(metrics.snapshot().offload.used_pages > 0);
    mode.store(1, Ordering::Relaxed);
    wait(&|| {
        backend.readiness() == BackendReadiness::Ready && metrics.snapshot().offload.used_pages == 0
    });
    assert_eq!(metrics.snapshot().offload.stalled_transfers, 0);
    assert_eq!(metrics.snapshot().offload.oldest_pending_micros, 0);
    backend.shutdown();
    assert!(matches!(
        backend.readiness(),
        BackendReadiness::Terminated(_)
    ));
}

#[test]
fn a_busy_step_detects_an_overdue_copy_and_cancellation_keeps_its_lease() {
    let clock = Arc::new(TransferClock::default());
    let (mut engine, mode, id) = pending_copy_engine_with_clock(clock.clone());
    let state = AtomicU8::new(READY);
    clock.0.store(11_000_000, Ordering::Relaxed);
    engine.step().unwrap();
    update_transfer_readiness(&engine, &state);
    assert_eq!(state.load(Ordering::Acquire), KV_STALLED);
    assert_eq!(engine.metrics().snapshot().offload.stalls, 1);
    engine.cancel(id).unwrap();
    assert!(engine.metrics().snapshot().offload.used_pages > 0);
    engine.audit().unwrap();
    mode.store(1, Ordering::Relaxed);
    assert!(!engine.poll_idle_kv_transfers().unwrap());
    update_transfer_readiness(&engine, &state);
    assert_eq!(state.load(Ordering::Acquire), READY);
    assert_eq!(engine.metrics().snapshot().offload.used_pages, 0);
    engine.step().unwrap();
    engine.audit().unwrap();
}

#[test]
fn startup_warmup_uses_real_group_admission_with_a_small_kv_pool() {
    let config = EngineConfig::new(
        SchedulerConfig::new(4, 16, 32, 32, 1).unwrap(),
        KvGeometry::new(4, 12).unwrap(),
        ModelVocabulary::new(259).unwrap(),
        48,
        16,
    )
    .unwrap()
    .with_prefix_cache(1)
    .with_speculation(
        SpeculationConfig::new(3, 32, SpeculationMode::Fixed, DraftSource::Bundle).unwrap(),
    );
    let batches = Arc::new(Mutex::new(Vec::new()));
    let engine = Engine::new(
        config,
        ScriptedModel {
            tokens: vec![X; 32],
            batches: Some(Arc::clone(&batches)),
            ..Default::default()
        },
    )
    .unwrap();
    let metrics = engine.metrics();
    let kv = engine.kv_metrics();
    let before = metrics.snapshot();
    let kv_before = kv.snapshot();
    let mut descriptor =
        FrontendDescriptorV1::from_component_bytes(declarations::KEY_VALUE.as_bytes()).unwrap();
    descriptor.vocabulary.size = 259;
    descriptor.vocabulary.eos_token_id = EOS;
    descriptor.vocabulary.stop_token_ids = vec![EOS];
    descriptor.vocabulary.prompt_special_tokens = vec!["[BOS]".to_owned()];
    let backend = EngineGenerationBackend::new(
        tokenizer(),
        engine,
        EngineBackendConfig::new(4).unwrap(),
        EngineModelPolicyV1::from_declaration(&descriptor),
    )
    .unwrap();
    assert_eq!(backend.startup_wave_capacity(21), 2);
    let steps = backend.startup_step_log();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    runtime
        .block_on(crate::startup::warmup(
            &backend,
            "test/model".into(),
            48,
            runtime_server::RequestAcceptanceProfile::General,
            4,
            false,
            Some(runtime_engine::warmup::VerifyWarmup {
                item_rows: vec![1, 2, 4],
                maximum_draft_tokens: 3,
                step_rows: 32,
                drafting_items: Some(4),
                exact_graphs: false,
                widths: Vec::new(),
                allocated: Vec::new(),
            }),
            Some(&steps),
        ))
        .unwrap();
    assert_eq!(backend.readiness(), BackendReadiness::Ready);
    // The engine reported the warmup's steps, a prefill first.
    assert!(
        steps
            .since(0)
            .first()
            .is_some_and(|step| step.prefill_tokens > 0)
    );
    assert_eq!(metrics.snapshot(), before);
    assert_eq!(kv.snapshot(), kv_before);
    assert!(batches.lock().unwrap().contains(&2));
    assert!(batches.lock().unwrap().iter().all(|&items| items <= 3));
    backend.shutdown();
}

/// Startup warmup runs without prefix caching in the engine while the
/// scheduler keeps its prefix-cache configuration. A forced preemption during
/// warmup must recompute, not capture an anchor nothing restores: the
/// generation worker would otherwise go idle with requests still active.
#[test]
fn startup_warmup_recomputes_forced_preemptions() {
    let config = EngineConfig::new(
        SchedulerConfig::new(4, 16, 32, 32, 1).unwrap(),
        KvGeometry::new(64, 64).unwrap(),
        ModelVocabulary::new(259).unwrap(),
        48,
        16,
    )
    .unwrap()
    .with_prefix_cache(1)
    .with_admission(runtime_scheduler::AdmissionPolicy::Optimistic(
        runtime_scheduler::OptimisticAdmission::new(0, 3)
            .unwrap()
            .with_forced_preemption(4)
            .unwrap(),
    ))
    .with_speculation(
        SpeculationConfig::new(3, 32, SpeculationMode::Fixed, DraftSource::Bundle).unwrap(),
    );
    let engine = Engine::new(
        config,
        ScriptedModel {
            tokens: vec![X; 32],
            ..Default::default()
        },
    )
    .unwrap();
    let mut descriptor =
        FrontendDescriptorV1::from_component_bytes(declarations::KEY_VALUE.as_bytes()).unwrap();
    descriptor.vocabulary.size = 259;
    descriptor.vocabulary.eos_token_id = EOS;
    descriptor.vocabulary.stop_token_ids = vec![EOS];
    descriptor.vocabulary.prompt_special_tokens = vec!["[BOS]".to_owned()];
    let backend = EngineGenerationBackend::new(
        tokenizer(),
        engine,
        EngineBackendConfig::new(4).unwrap(),
        EngineModelPolicyV1::from_declaration(&descriptor),
    )
    .unwrap();
    let steps = backend.startup_step_log();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let result = runtime.block_on(crate::startup::warmup(
        &backend,
        "test/model".into(),
        48,
        runtime_server::RequestAcceptanceProfile::General,
        4,
        false,
        Some(runtime_engine::warmup::VerifyWarmup {
            item_rows: vec![1, 2, 4],
            maximum_draft_tokens: 3,
            step_rows: 32,
            drafting_items: Some(4),
            exact_graphs: false,
            widths: Vec::new(),
            allocated: Vec::new(),
        }),
        Some(&steps),
    ));
    assert!(result.is_ok(), "{result:?}");
    assert_eq!(backend.readiness(), BackendReadiness::Ready);
    backend.shutdown();
}

#[test]
fn a_client_gone_before_its_prompt_echo_leaves_nothing_behind() {
    let tokenizer = tokenizer();
    let mut descriptor =
        FrontendDescriptorV1::from_component_bytes(declarations::KEY_VALUE.as_bytes()).unwrap();
    descriptor.vocabulary.size = 259;
    descriptor.vocabulary.eos_token_id = EOS;
    descriptor.vocabulary.stop_token_ids = vec![EOS];
    let policy = EngineModelPolicyV1::from_declaration(&descriptor);
    let mut engine = engine(vec![X; 8], false, false);
    let mut active = BTreeMap::new();
    for id in 1..=3 {
        let request = PreparedRequest {
            request_id: format!("echo-{id}"),
            client_request_id: None,
            model: "fixture/alpha-1".into(),
            prompt: PreparedPrompt::TokenIds(vec![X, X, X]),
            max_new_tokens: 4,
            stop: vec![],
            decoding: runtime_server::DecodingOptions::default(),
            sampling: SamplingParameters::new(0.0).unwrap(),
            seed: 7,
            // Returned token IDs echo the prompt as the first event.
            logprobs: runtime_server::RequestedLogprobs {
                enabled: false,
                top: 0,
                echo: false,
                prompt_top: None,
                return_token_ids: true,
            },
            chat: None,
            constraint: None,
            cache_salt: None,
            cancellation: CancellationSignal::new(),
        };
        // The client hung up after submitting, so the echo cannot be sent.
        let (sender, receiver) = tokio::sync::mpsc::channel(128);
        drop(receiver);
        let (response, mut admission) = oneshot::channel();
        assert!(admit(
            &tokenizer,
            &mut engine,
            &mut active,
            RequestId::new(id),
            request,
            None,
            EventSender::new(sender),
            response,
            65536,
            &policy,
            Delivery::Immediate,
        ));
        assert!(matches!(
            admission.try_recv(),
            Ok(Err(AdmissionError::WorkerUnavailable))
        ));
        let _ = engine.drain_events();
    }
    assert!(active.is_empty());
    // The deferred completion metrics were finished with each request.
    assert_eq!(engine.metrics().live_request_records(), 0);
    assert_eq!(engine.metrics().snapshot().requests_active, 0);
    assert_eq!(engine.metrics().snapshot().cancellations, 3);
}
