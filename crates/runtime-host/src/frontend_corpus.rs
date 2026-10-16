//! A model's HF differential corpus through the served frontend, for the
//! model's own tests.
//!
//! `python -m tools.refrun.frontend_corpus generate --model <m>` renders
//! seeded requests with transformers at the model's pinned chat template,
//! and `minimize` keeps the committed subset in `fixtures/<m>/frontend/`:
//! raw render vectors (`corpus.json`), round-trip streaming vectors
//! (`corpus-rounds.json`, a history turn of the template read back as a
//! completion) and the harness metadata (`corpus-coverage.json`). A model's
//! corpus test names its export, its component, its committed corpus and
//! its tokenizer in a [`FrontendCorpusV1`], and the checks are these:
//!
//! - every render vector is sent as the raw request it holds, through the
//!   checks `chopinfer bundle check-frontend` runs, and renders the
//!   reference's text and token ids or is refused as the vector states;
//! - every round trip runs as a streaming vector, whole and streamed;
//! - every round-trip completion is cut in two at each byte and decoded,
//!   whole and streamed, by the parser the served request opens, by the
//!   shared parser called directly and through the frontend documents; all
//!   three answer each push with the same updates, which read back the
//!   reference turn and finish it once;
//! - every round-trip completion, and its answer after the reasoning close,
//!   also parses in one call to the reference turn's content and calls.
//!
//! `CHOPINFER_FRONTEND_CORPUS_DIR` names a full corpus
//! (`frontend_corpus generate --output DIR`), which
//! [`FrontendCorpusV1::check_full`] runs with the same checks. With
//! `CHOPINFER_FRONTEND_CORPUS_DUMP` naming a file, the run over the
//! selected corpus (the full one when `CHOPINFER_FRONTEND_CORPUS_DIR` is
//! set, otherwise the committed one) writes each case's outcome there, so
//! two runs' files are an outcome diff: `{"cases": {id: {"rendered_text", "token_ids"} or {"refusal"}},
//! "checks": {"<kind> <id>": null or the difference}, "failed": [...]}`.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use runtime_bundle_abi::frontend::FrontendExportV1;
use runtime_frontend::{
    BundleChatStateV1, BundleFrontendV1, BundleParserSourceV1, CompletedCompletionV1,
    CompletionDecoderV1, CompletionFinishV1, CompletionUpdateV1, DecoderLimitsV1,
    FrontendDescriptorV1, GeneratedTokenV1, PromptModeV1, parse_completion_v1,
    tokenizer_from_components_v1,
};
use runtime_server::{
    DecodingOptions, FinishReason, PreparedPrompt, ServerFrontendV1, prepare_chat_request_v1,
};
use serde_json::{Map, Value, json};

use crate::frontend_check::{
    FrontendCaseOutcomeV1, FrontendCheckV1, StreamingVectorV1, read_render_vectors_v1,
    read_streaming_vectors_v1,
};

/// The full corpus a run checks besides the committed one.
pub const CORPUS_DIR_ENV_V1: &str = "CHOPINFER_FRONTEND_CORPUS_DIR";

/// The file the outcomes are written to.
pub const CORPUS_DUMP_ENV_V1: &str = "CHOPINFER_FRONTEND_CORPUS_DUMP";

/// The fields of a render vector that are not the request.
const VECTOR_FIELDS: [&str; 5] = ["id", "raw", "rendered_text", "token_ids", "refusal"];

/// One model's corpus: what its test runs the corpus files against.
pub struct FrontendCorpusV1 {
    /// The model, as a skipped run names it.
    pub model: &'static str,
    /// The export the bundle serves.
    pub export: FrontendExportV1,
    /// The served component the corpus renders with.
    pub component: Vec<u8>,
    /// The committed corpus directory, `fixtures/<m>/frontend`.
    pub committed: PathBuf,
    /// The checkpoint directory the declared tokenizer files are read from,
    /// when one is configured; without it every check is skipped.
    pub tokenizer_dir: Option<PathBuf>,
}

/// The first of `variables` that is set, as a directory.
#[must_use]
pub fn directory_from_env_v1(variables: &[&str]) -> Option<PathBuf> {
    variables
        .iter()
        .find_map(std::env::var_os)
        .map(PathBuf::from)
}

impl FrontendCorpusV1 {
    /// Runs every check over the committed corpus, and writes its outcomes
    /// when asked and no full corpus is selected.
    ///
    /// # Panics
    ///
    /// When a check fails, listing every failure.
    pub fn check_committed(&self) {
        let selected = std::env::var_os(CORPUS_DIR_ENV_V1).is_none();
        self.check(&self.committed, selected);
    }

    /// Runs every check over the full corpus `CHOPINFER_FRONTEND_CORPUS_DIR`
    /// names, and writes its outcomes when asked; skips without it.
    ///
    /// # Panics
    ///
    /// When a check fails, listing every failure.
    pub fn check_full(&self) {
        let Some(directory) = std::env::var_os(CORPUS_DIR_ENV_V1).map(PathBuf::from) else {
            eprintln!("SKIP: {CORPUS_DIR_ENV_V1} is unset");
            return;
        };
        self.check(&directory, true);
    }

    /// Checks that the transport translation the corpus's HF inputs follow
    /// (the cases module's `TRANSPORT`, recorded in the committed coverage)
    /// is what the export answers when it opens: which history reasoning
    /// reaches the render, whether it reads client tool objects and
    /// tool_reference parts, and its extra roles.
    ///
    /// # Panics
    ///
    /// When they differ.
    pub fn check_transport(&self) {
        let coverage: Value =
            serde_json::from_slice(&read(&self.committed.join("corpus-coverage.json"))).unwrap();
        let frontend = BundleFrontendV1::from_export(&self.export, &self.component).unwrap();
        assert_eq!(
            coverage["transport"],
            json!({
                "reasoning_content_fallback": frontend.reasoning_content_fallback(),
                "tool_call_reasoning_fallback": frontend.tool_call_reasoning_fallback(),
                "reads_tool_objects": frontend.reads_tool_objects(),
                "tool_references": frontend.tool_references(),
                "extra_roles": frontend.extra_roles(),
            })
        );
    }

    fn check(&self, directory: &Path, selected: bool) {
        let Some(root) = &self.tokenizer_dir else {
            eprintln!(
                "SKIP: no {} tokenizer directory is configured, skipping the frontend corpus",
                self.model
            );
            return;
        };
        let declaration = FrontendDescriptorV1::from_component_bytes(&self.component).unwrap();
        let tokenizer = tokenizer_from_components_v1(
            &declaration,
            &read(&root.join(&declaration.tokenizer.definition.name)),
            &read(&root.join(&declaration.tokenizer.configuration.name)),
        )
        .unwrap();
        let cases = read(&directory.join("corpus.json"));
        let rounds = read(&directory.join("corpus-rounds.json"));
        let case_values: Vec<Map<String, Value>> = serde_json::from_slice(&cases).unwrap();
        for case in &case_values {
            assert_eq!(
                case.get("raw"),
                Some(&Value::Bool(true)),
                "{:?} is not a raw request",
                case["id"]
            );
        }
        let render = read_render_vectors_v1(&cases).unwrap();
        let streaming = read_streaming_vectors_v1(&rounds).unwrap();
        assert!(!render.is_empty() && !streaming.is_empty());
        let check =
            FrontendCheckV1::from_component_v1(&self.export, &self.component, tokenizer).unwrap();
        let outcomes = check.run_v1(&render, &[], &streaming);
        assert_eq!(outcomes.len(), render.len() + streaming.len());
        let mut failed: Vec<String> = outcomes
            .iter()
            .filter(|outcome| outcome.result.is_err())
            .map(ToString::to_string)
            .collect();
        let round_values: Vec<Map<String, Value>> = serde_json::from_slice(&rounds).unwrap();
        let mut decodes = 0;
        for (vector, value) in streaming.iter().zip(round_values) {
            let Some(Value::Object(request)) = value.get("request") else {
                panic!("round trip {} names no request", vector.id);
            };
            match splits(&check, vector, request.clone()) {
                Ok(count) => decodes += count,
                Err(difference) => failed.push(format!("FAIL splits {}: {difference}", vector.id)),
            }
        }
        let dump = std::env::var_os(CORPUS_DUMP_ENV_V1);
        if let Some(path) = dump.filter(|_| selected) {
            write_outcomes(Path::new(&path), &check, case_values, &outcomes, &failed);
        }
        assert!(
            failed.is_empty(),
            "{} of {} corpus checks fail:\n{}",
            failed.len(),
            outcomes.len() + streaming.len(),
            failed.join("\n")
        );
        eprintln!(
            "{}: {} render and {} round-trip vectors pass, and {decodes} split decodes",
            self.model,
            render.len(),
            streaming.len()
        );
    }
}

fn read(path: &Path) -> Vec<u8> {
    std::fs::read(path).unwrap_or_else(|error| panic!("{}: {error}", path.display()))
}

/// What one decode delivered, read back the way a chat response folds it.
#[derive(Debug, Default)]
struct Delivered {
    reasoning: String,
    content: String,
    calls: Vec<(String, String)>,
    streamed: Vec<(String, String)>,
    finished: Vec<CompletedCompletionV1>,
}

impl Delivered {
    fn fold(pushes: &[Vec<CompletionUpdateV1>]) -> Self {
        let mut delivered = Self::default();
        for update in pushes.iter().flatten().cloned() {
            match update {
                CompletionUpdateV1::Reasoning(span) => delivered.reasoning.push_str(&span.text),
                CompletionUpdateV1::Content(span) => delivered.content.push_str(&span.text),
                CompletionUpdateV1::ToolCalls(group) => delivered.calls.extend(
                    group
                        .calls
                        .into_iter()
                        .map(|call| (call.name, call.arguments)),
                ),
                CompletionUpdateV1::ToolCallDeltas(deltas) => {
                    for delta in deltas {
                        match delta.name {
                            Some(name) => delivered.streamed.push((name, delta.arguments)),
                            None => delivered.streamed[delta.index].1.push_str(&delta.arguments),
                        }
                    }
                }
                CompletionUpdateV1::Finished(done) => delivered.finished.push(done),
            }
        }
        delivered
    }
}

/// The parsers a round trip decodes through: the one the served request
/// opens, the shared parser called directly, and the shared parser opened
/// through the frontend documents of a chat format that renders nothing.
const PARSER_PATHS: [&str; 3] = ["served", "shared", "document"];

/// Feeds a round trip's completion, which ends in its stop token's text, to
/// the parser in two chunks at every byte split, with tool calls delivered
/// whole and streamed. The request is prepared as served, so each decoder
/// opens in the mode its prompt leaves, with its tools. Every parser path
/// must answer each push with the same updates (text, tool calls, token
/// contributions and the terminal update alike), and what they deliver must
/// be the reference turn, finished once by the last push. The whole
/// completion, and the answer after the reasoning close, also parse in one
/// call to the reference turn's content and call names. Returns how many
/// decodes it compared.
fn splits(
    check: &FrontendCheckV1,
    vector: &StreamingVectorV1,
    request: Map<String, Value>,
) -> Result<usize, String> {
    let prepared = prepare_chat_request_v1(
        &check.frontend,
        check.max_string_bytes,
        request,
        vector.token_ids.len().max(1),
    )
    .map_err(|error| format!("the request was refused: {}", error.message))?;
    let chat = prepared.chat.as_ref().ok_or("no chat metadata")?;
    let state = BundleChatStateV1::from_state(&chat.state).ok_or("no bundle chat state")?;
    let served = state
        .parser_source()
        .or(check.policy.parser_source())
        .ok_or("no parser source")?;
    let descriptor = Arc::new(check.frontend.descriptor().clone());
    let document = BundleParserSourceV1::from_declaration(&descriptor)
        .map_err(|error| format!("the document parser: {error}"))?;
    let sources = [Some(served), None, Some(&document)];
    let bytes = check
        .tokenizer
        .decode_bytes(&vector.token_ids)
        .map_err(|error| error.to_string())?;
    let expect = &vector.expect;
    let calls: Vec<(String, String)> = expect
        .tool_calls
        .iter()
        .map(|call| (call.name.clone(), call.arguments.clone()))
        .collect();
    let finish = match expect.finish_reason {
        FinishReason::Stop => CompletionFinishV1::Stop,
        FinishReason::ToolCalls => CompletionFinishV1::ToolCalls,
        FinishReason::Length => CompletionFinishV1::Length,
    };
    whole(&descriptor, state.mode(), &bytes, vector)?;
    let mut decodes = 0;
    for stream in [false, true] {
        for split in 1..bytes.len() {
            let at = format!("stream {stream} split {split}");
            let chunks = [&bytes[..split], &bytes[split..]];
            let mut decoders = sources.map(|source| decoder(&descriptor, state, source, stream));
            let mut pushes = Vec::new();
            for (ordinal, chunk) in chunks.into_iter().enumerate() {
                let token = GeneratedTokenV1::new(ordinal as u64, 42, chunk.to_vec());
                let answers = decoders
                    .each_mut()
                    .map(|decoder| decoder.push(token.clone()));
                for (path, answer) in PARSER_PATHS.iter().zip(&answers).skip(1) {
                    if *answer != answers[0] {
                        return Err(format!(
                            "{at} push {ordinal}: the served parser answered {:?}, the {path} \
                             parser {answer:?}",
                            answers[0]
                        ));
                    }
                }
                let [served, ..] = answers;
                pushes.push(served.map_err(|error| format!("{at} push {ordinal}: {error}"))?);
            }
            let delivered = Delivered::fold(&pushes);
            if delivered.reasoning != expect.reasoning {
                return Err(format!("{at}: reasoning {:?}", delivered.reasoning));
            }
            if delivered.content != expect.content {
                return Err(format!("{at}: content {:?}", delivered.content));
            }
            if delivered.calls != calls || (stream && delivered.streamed != calls) {
                return Err(format!(
                    "{at}: calls {:?} streamed {:?}",
                    delivered.calls, delivered.streamed
                ));
            }
            let last = pushes[1].last();
            match (delivered.finished.as_slice(), last) {
                ([done], Some(CompletionUpdateV1::Finished(_)))
                    if done.reason == finish && done.generated_token_count == chunks.len() => {}
                _ => {
                    return Err(format!(
                        "{at}: finished {:?}, not once at the last push as {finish:?} after {} \
                         tokens",
                        delivered.finished,
                        chunks.len()
                    ));
                }
            }
            decodes += 1;
        }
    }
    Ok(decodes)
}

/// One decoder of the request `state` holds, through `source` or, without
/// one, the shared parser called directly.
fn decoder(
    descriptor: &Arc<FrontendDescriptorV1>,
    state: &BundleChatStateV1,
    source: Option<&BundleParserSourceV1>,
    stream: bool,
) -> CompletionDecoderV1 {
    let mut decoder = CompletionDecoderV1::new(
        Arc::clone(descriptor),
        state.mode(),
        DecoderLimitsV1::declared(descriptor),
    );
    if let Some(source) = source {
        decoder.set_parser_source(source.clone());
    }
    decoder.set_decoding_options(
        DecodingOptions {
            stream_tool_calls: stream,
            ..Default::default()
        },
        false,
    );
    decoder.set_tool_schemas(state.tools());
    decoder.set_tool_extraction(state.extracts_tool_calls());
    decoder.set_reasoning_open(state.reasoning_open());
    decoder
}

/// Parses a round trip's whole completion in one call, in the mode its
/// prompt leaves, and the answer after its reasoning close as a completion
/// that begins in the answer. Both read the reference turn's content and
/// call names; the one call reads no tool schema, so the arguments are not
/// compared.
fn whole(
    descriptor: &FrontendDescriptorV1,
    mode: PromptModeV1,
    bytes: &[u8],
    vector: &StreamingVectorV1,
) -> Result<(), String> {
    let text = std::str::from_utf8(bytes).map_err(|error| error.to_string())?;
    let close = descriptor.format["parser"]["reasoning_close"]
        .as_str()
        .ok_or("the declaration has no format.parser reasoning_close")?;
    let mut parses = vec![("whole completion", text, mode)];
    if let Some((_, answer)) = text.split_once(close) {
        parses.push(("answer", answer, PromptModeV1::Chat));
    }
    let names: Vec<&str> = vector
        .expect
        .tool_calls
        .iter()
        .map(|call| call.name.as_str())
        .collect();
    for (part, text, mode) in parses {
        let parsed = parse_completion_v1(descriptor, text, mode)
            .map_err(|error| format!("the {part} does not parse in one call: {error}"))?;
        let parsed_names: Vec<&str> = parsed
            .tool_calls
            .iter()
            .map(|call| call.name.as_str())
            .collect();
        if parsed.content != vector.expect.content || parsed_names != names {
            return Err(format!(
                "the {part} parses in one call to content {:?} and calls {parsed_names:?}",
                parsed.content
            ));
        }
    }
    Ok(())
}

/// Writes each render case's outcome, the prompt text and its token ids or
/// the refusal, with every check's result and every failure.
fn write_outcomes(
    path: &Path,
    check: &FrontendCheckV1,
    cases: Vec<Map<String, Value>>,
    outcomes: &[FrontendCaseOutcomeV1],
    failed: &[String],
) {
    let mut rendered = Map::new();
    for case in cases {
        let id = case["id"].as_str().unwrap().to_owned();
        let body: Map<String, Value> = case
            .into_iter()
            .filter(|(key, _)| !VECTOR_FIELDS.contains(&key.as_str()))
            .collect();
        let outcome =
            match prepare_chat_request_v1(&check.frontend, check.max_string_bytes, body, 1) {
                Err(error) => {
                    json!({"refusal": {"message": error.message, "parameter": error.parameter}})
                }
                Ok(prepared) => {
                    let text = match prepared.prompt {
                        PreparedPrompt::RenderedChat(text) => text,
                        PreparedPrompt::MediaChat { segments, media } => check
                            .media_prompt_text_v1(&segments, &media)
                            .unwrap_or_else(|error| panic!("{id}: {error}")),
                        _ => panic!("{id}: the transport prepared no rendered chat"),
                    };
                    let ids = check.encode_v1(&text).ok();
                    json!({"rendered_text": text, "token_ids": ids})
                }
            };
        rendered.insert(id, outcome);
    }
    let checks: Map<String, Value> = outcomes
        .iter()
        .map(|outcome| {
            (
                format!("{} {}", outcome.kind.name(), outcome.id),
                json!(outcome.result.as_ref().err()),
            )
        })
        .collect();
    let document = json!({"cases": rendered, "checks": checks, "failed": failed});
    let mut bytes = serde_json::to_vec_pretty(&document).unwrap();
    bytes.push(b'\n');
    std::fs::write(path, bytes).unwrap_or_else(|error| panic!("{}: {error}", path.display()));
    eprintln!("wrote the corpus outcomes to {}", path.display());
}
