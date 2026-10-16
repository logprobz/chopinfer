//! This model's frontend vectors through the checks `chopinfer bundle
//! check-frontend` runs against a built package: the render and tokenizer
//! goldens and the streaming vectors, transcribed from `decode_goldens.rs`
//! (with steps where a golden pins what each chunk publishes),
//! served by MiMo's chat format through its in-process export. The tokenizer is the
//! checkpoint's, so the vectors run with `CHOPINFER_MIMO_TOKENIZER_DIR` or
//! `CHOPINFER_MIMO_CHECKPOINT` set; a built package runs the same files with
//! `chopinfer bundle check-frontend --bundle DIR --render ... --tokenizer ...
//! --streaming ...`.

use std::path::Path;

use model_mimo26::checkpoint::default_identity;
use model_mimo26::frontend::component_for_flash;
use model_mimo26_contract::encoders::Encoders;
use runtime_frontend::{FrontendDescriptorV1, tokenizer_from_components_v1};
use runtime_host::{
    FrontendCheckV1, read_render_vectors_v1, read_streaming_vectors_v1, read_tokenizer_vectors_v1,
};

const RENDER: [&[u8]; 3] = [
    include_bytes!("../../../../fixtures/mimo26/frontend/chat-renders.json"),
    include_bytes!("../fixtures/chat-integers.json"),
    include_bytes!("../fixtures/chat-floats.json"),
];
const TOKENIZER: &[u8] =
    include_bytes!("../../../../fixtures/mimo26/frontend/tokenizer-golden.json");
const STREAMING: &[u8] = include_bytes!("../fixtures/streaming-vectors.json");

#[test]
fn every_vector_file_reads() {
    let render: usize = RENDER
        .iter()
        .map(|file| read_render_vectors_v1(file).unwrap().len())
        .sum();
    assert_eq!(render, 65);
    assert_eq!(read_tokenizer_vectors_v1(TOKENIZER).unwrap().len(), 52);
    let streaming = read_streaming_vectors_v1(STREAMING).unwrap();
    assert_eq!(streaming.len(), 34);
    // One golden pins what each chunk publishes:
    // tool_arguments_wait_for_the_function_close.
    let stepped: Vec<_> = streaming
        .iter()
        .filter(|vector| vector.steps.is_some())
        .map(|vector| vector.id.as_str())
        .collect();
    assert_eq!(stepped, ["arguments-at-function-close"]);
}

#[test]
fn vectors_pass_the_served_frontend_checks() {
    let Ok(root) = std::env::var("CHOPINFER_MIMO_TOKENIZER_DIR")
        .or_else(|_| std::env::var("CHOPINFER_MIMO_CHECKPOINT"))
    else {
        eprintln!(
            "CHOPINFER_MIMO_TOKENIZER_DIR and CHOPINFER_MIMO_CHECKPOINT are unset, skipping frontend vectors"
        );
        return;
    };
    let component = component_for_flash(default_identity(), Encoders::ALL).unwrap();
    let declaration = FrontendDescriptorV1::from_component_bytes(&component).unwrap();
    let read = |name: &str| std::fs::read(Path::new(&root).join(name)).unwrap();
    let tokenizer = tokenizer_from_components_v1(
        &declaration,
        &read(&declaration.tokenizer.definition.name),
        &read(&declaration.tokenizer.configuration.name),
    )
    .unwrap();
    let check = FrontendCheckV1::from_component_v1(
        &frontend_format::chat_format_export_v1::<model_mimo26_contract::frontend::Mimo26Format>(),
        &component,
        tokenizer,
    )
    .unwrap();
    let render: Vec<_> = RENDER
        .iter()
        .flat_map(|file| read_render_vectors_v1(file).unwrap())
        .collect();
    let outcomes = check.run_v1(
        &render,
        &read_tokenizer_vectors_v1(TOKENIZER).unwrap(),
        &read_streaming_vectors_v1(STREAMING).unwrap(),
    );
    let failed: Vec<_> = outcomes
        .iter()
        .filter(|outcome| outcome.result.is_err())
        .map(ToString::to_string)
        .collect();
    assert!(failed.is_empty(), "{}", failed.join("\n"));
    assert_eq!(outcomes.len(), 65 + 52 + 34);
}
