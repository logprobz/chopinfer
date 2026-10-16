//! This model's frontend vectors through the checks `chopinfer bundle
//! check-frontend` runs against a built package: the render and tokenizer
//! goldens and the streaming vectors, transcribed from the decode cases in
//! `frontend_goldens.rs`, served by the bundle's chat format through its
//! in-process export.
//! The tokenizer is the checkpoint's, so the vectors run with
//! `CHOPINFER_GLM5NEXT_CHECKPOINT` set; a built package runs the same files
//! with `chopinfer bundle check-frontend`.

use std::path::PathBuf;

use frontend_format::chat_format_export_v1;
use model_glm5next::frontend::COMPONENT;
use model_glm5next_contract::frontend::Glm53Format;
use runtime_frontend::{FrontendDescriptorV1, tokenizer_from_components_v1};
use runtime_host::{
    FrontendCheckV1, read_render_vectors_v1, read_streaming_vectors_v1, read_tokenizer_vectors_v1,
};

const RENDER: &[u8] = include_bytes!("../fixtures/checkpoints/glm-5.3-flash/chat-goldens.json");
const TOKENIZER: &[u8] =
    include_bytes!("../fixtures/checkpoints/glm-5.3-flash/tokenizer-golden.json");
const STREAMING: &[u8] =
    include_bytes!("../fixtures/checkpoints/glm-5.3-flash/streaming-vectors.json");

#[test]
fn every_vector_file_reads() {
    assert_eq!(read_render_vectors_v1(RENDER).unwrap().len(), 16);
    assert_eq!(read_tokenizer_vectors_v1(TOKENIZER).unwrap().len(), 27);
    assert_eq!(read_streaming_vectors_v1(STREAMING).unwrap().len(), 3);
}

#[test]
fn vectors_pass_the_served_frontend_checks() {
    let Some(root) = std::env::var_os("CHOPINFER_GLM5NEXT_CHECKPOINT").map(PathBuf::from) else {
        eprintln!("SKIP: CHOPINFER_GLM5NEXT_CHECKPOINT is unset");
        return;
    };
    let declaration = FrontendDescriptorV1::from_component_bytes(COMPONENT).unwrap();
    let read = |name: &str| std::fs::read(root.join(name)).unwrap();
    let tokenizer = tokenizer_from_components_v1(
        &declaration,
        &read(&declaration.tokenizer.definition.name),
        &read(&declaration.tokenizer.configuration.name),
    )
    .unwrap();
    let check = FrontendCheckV1::from_component_v1(
        &chat_format_export_v1::<Glm53Format>(),
        COMPONENT,
        tokenizer,
    )
    .unwrap();
    let outcomes = check.run_v1(
        &read_render_vectors_v1(RENDER).unwrap(),
        &read_tokenizer_vectors_v1(TOKENIZER).unwrap(),
        &read_streaming_vectors_v1(STREAMING).unwrap(),
    );
    let failed: Vec<_> = outcomes
        .iter()
        .filter(|outcome| outcome.result.is_err())
        .map(ToString::to_string)
        .collect();
    assert!(failed.is_empty(), "{}", failed.join("\n"));
    assert_eq!(outcomes.len(), 16 + 27 + 3);
}
