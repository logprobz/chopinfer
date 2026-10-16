//! Per-token decode cost of the shared parsers called directly against the
//! same parsers through the frontend documents.
//!
//! Run with `cargo run --release -p runtime-frontend --features test-support
//! --example frontend_decode_cost -- [REPEATS] [--module PATH]`. For each
//! shared tool grammar (function parameters and key-value arguments, the
//! parser fixtures) and each workload (chat, reasoning, streamed tool
//! calls), it decodes one long completion in 4-byte tokens each way, checks
//! that every way reports the same updates, and prints nanoseconds per
//! token. The ways are the library called directly, a chat format's export
//! through the in-process loopback, and, with `--module`, the export of a
//! bundle module the loader opens from PATH (any package module that exports
//! a chat format over a `format.parser` declaration, such as the TP4
//! fixture's `libbundle_tp4_fixture.so`).

use std::hint::black_box;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use frontend_format::FormatDescriptorV1;
use frontend_format::test_support::fixtures::{FUNCTION_PARAMETERS, KEY_VALUE};
use runtime_frontend::{
    BundleParserSourceV1, CompletionDecoderV1, CompletionUpdateV1, DecoderLimitsV1,
    FrontendDescriptorV1, GeneratedTokenV1, PromptModeV1,
};
use runtime_server::{DecodingOptions, FrontendToolV1};
use serde_json::json;

/// Bytes per generated token.
const TOKEN_BYTES: usize = 4;
/// Sentences of text in each completion section.
const SENTENCES: usize = 200;
const SENTENCE: &str = "The quick brown fox jumps over the lazy dog. ";

/// A parser fixture with room for one long completion.
fn fixture(declaration: &str) -> FrontendDescriptorV1 {
    let mut descriptor =
        FrontendDescriptorV1::from_component_bytes(declaration.as_bytes()).unwrap();
    descriptor.limits.maximum_completion_bytes = 1 << 20;
    descriptor.limits.maximum_generated_tokens = 1 << 18;
    descriptor.limits.maximum_json_bytes = 1 << 16;
    descriptor
}

fn lookup() -> FrontendToolV1 {
    FrontendToolV1 {
        name: "lookup".to_owned(),
        description: None,
        parameters: serde_json::from_value(
            json!({"type": "object", "properties": {"query": {"type": "string"}}}),
        )
        .unwrap(),
        parameters_present: true,
        strict: None,
    }
}

/// One tool call with a long string argument, in the fixture's tool grammar.
fn call(name: &str, text: &str) -> String {
    match name {
        "function-parameter" => format!(
            "<tool_call>\n<function=lookup>\n<parameter=query>\n{text}\n</parameter>\n</function>\n</tool_call>"
        ),
        _ => format!(
            "<tool_call>lookup<arg_key>query</arg_key><arg_value>{text}</arg_value></tool_call>"
        ),
    }
}

/// Decodes `text` once, and returns the updates and the elapsed time.
fn decode(
    descriptor: &Arc<FrontendDescriptorV1>,
    source: Option<&BundleParserSourceV1>,
    mode: PromptModeV1,
    text: &[u8],
    stream: bool,
) -> (Vec<CompletionUpdateV1>, std::time::Duration) {
    let started = Instant::now();
    let mut decoder = CompletionDecoderV1::new(
        Arc::clone(descriptor),
        mode,
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
    decoder.set_tool_schemas(&[lookup()]);
    let mut updates = Vec::new();
    for (ordinal, bytes) in text.chunks(TOKEN_BYTES).enumerate() {
        updates.extend(
            decoder
                .push(GeneratedTokenV1::new(ordinal as u64, 200, bytes.to_vec()))
                .unwrap(),
        );
    }
    let elapsed = started.elapsed();
    assert!(
        matches!(updates.last(), Some(CompletionUpdateV1::Finished(_))),
        "the completion finished"
    );
    (black_box(updates), elapsed)
}

fn main() {
    let mut repeats: usize = 20;
    let mut module = None;
    let mut arguments = std::env::args().skip(1);
    while let Some(argument) = arguments.next() {
        if argument == "--module" {
            module = Some(PathBuf::from(arguments.next().expect("a module path")));
        } else {
            repeats = argument.parse().expect("a repeat count");
        }
    }
    // The loader maps a private copy of the bytes, as a host maps a
    // package's authenticated module; the export holds the library.
    let export = module.map(|path| {
        let bytes = std::fs::read(&path).expect("the module reads");
        runtime_bundle_abi::BundleLibraryV1::open_verified_image(
            &bytes,
            &path.display().to_string(),
        )
        .expect("the module loads")
        .frontend_v1()
        .expect("the module's frontend table is valid")
        .expect("the module exports a frontend")
    });
    let body = SENTENCE.repeat(SENTENCES);
    println!(
        "format             workload    tokens  library ns/token  document ns/token  added ns/token  module ns/token  added ns/token"
    );
    for (name, descriptor) in [
        ("function-parameter", fixture(FUNCTION_PARAMETERS)),
        ("key-value", fixture(KEY_VALUE)),
    ] {
        // The stop token's spelling ends every format's completion.
        let stop = descriptor.vocabulary.stop_token_ids[0];
        let end = descriptor
            .vocabulary
            .boundary_tokens
            .iter()
            .find(|boundary| boundary.id == stop)
            .unwrap()
            .text
            .clone();
        let close = FormatDescriptorV1::read(descriptor.clone())
            .unwrap()
            .parser
            .reasoning_close;
        let workloads = [
            ("chat", PromptModeV1::Chat, format!("{body}{end}"), false),
            (
                "reasoning",
                PromptModeV1::Thinking,
                format!("{body}{close}{body}{end}"),
                false,
            ),
            (
                "tool-calls",
                PromptModeV1::Chat,
                format!("Checking.{}{end}", call(name, &body)),
                true,
            ),
        ];
        let descriptor = Arc::new(descriptor);
        let source = BundleParserSourceV1::from_declaration(&descriptor).unwrap();
        // The module's export opens over the same declaration, written out.
        let loaded = export.as_ref().map(|export| {
            let component = serde_json::to_vec(descriptor.as_ref()).unwrap();
            BundleParserSourceV1::from_export(export, &component).unwrap()
        });
        for (workload, mode, text, stream) in workloads {
            let tokens = text.len().div_ceil(TOKEN_BYTES);
            let sources = [None, Some(&source), loaded.as_ref()];
            let (library_updates, _) = decode(&descriptor, None, mode, text.as_bytes(), stream);
            for source in sources.iter().skip(1).flatten() {
                let (updates, _) = decode(&descriptor, Some(source), mode, text.as_bytes(), stream);
                assert_eq!(library_updates, updates, "{name} {workload}");
            }
            let mut best = [f64::MAX; 3];
            for _ in 0..repeats {
                for (index, source) in sources.iter().enumerate() {
                    if index == 2 && source.is_none() {
                        continue;
                    }
                    let (_, elapsed) = decode(&descriptor, *source, mode, text.as_bytes(), stream);
                    best[index] = best[index].min(elapsed.as_nanos() as f64 / tokens as f64);
                }
            }
            let module = if loaded.is_some() {
                format!("{:>16.0} {:>15.0}", best[2], best[2] - best[0])
            } else {
                format!("{:>16} {:>15}", "-", "-")
            };
            println!(
                "{name:<18} {workload:<10} {tokens:>7} {:>17.0} {:>18.0} {:>15.0} {module}",
                best[0],
                best[1],
                best[1] - best[0]
            );
        }
    }
}
