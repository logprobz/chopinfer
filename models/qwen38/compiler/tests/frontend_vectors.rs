//! This model's frontend vectors through the checks `chopinfer bundle
//! check-frontend` runs against a built package: each checkpoint's render
//! and tokenizer goldens and its streaming vectors, transcribed from
//! `decode_goldens.rs` and `chat_goldens.rs`, served by the bundle's chat
//! format through its in-process export. The tokenizer is the checkpoint's,
//! so the vectors run with the checkpoint's tokenizer or checkpoint variable
//! set; a built package runs the same files with `chopinfer bundle
//! check-frontend`.

use std::path::{Path, PathBuf};

use model_qwen38::frontend::COMPONENT;
use model_qwen38_contract::frontend::export;
use runtime_frontend::{FrontendDescriptorV1, tokenizer_from_components_v1};
use runtime_host::{
    FrontendCheckV1, read_render_vectors_v1, read_streaming_vectors_v1, read_tokenizer_vectors_v1,
};
#[allow(dead_code)]
#[path = "../src/checkpoint_index.rs"]
mod checkpoint_index;
use checkpoint_index::checkpoints;

const RENDER: [&str; 3] = [
    "chat-goldens.json",
    "chat-integers.json",
    "chat-floats.json",
];

fn fixtures(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("fixtures/checkpoints")
        .join(name)
}

fn read(path: PathBuf) -> Vec<u8> {
    std::fs::read(&path).unwrap_or_else(|error| panic!("{}: {error}", path.display()))
}

#[test]
fn every_vector_file_reads() {
    for checkpoint in checkpoints() {
        let directory = fixtures(&checkpoint.name);
        let render: usize = RENDER
            .iter()
            .map(|name| {
                read_render_vectors_v1(&read(directory.join(name)))
                    .unwrap()
                    .len()
            })
            .sum();
        assert_eq!(render, checkpoint.chat_cases + 4);
        let tokenizer = read(directory.join("tokenizer-golden.json"));
        assert_eq!(
            read_tokenizer_vectors_v1(&tokenizer).unwrap().len(),
            checkpoint.tokenizer_cases
        );
        let streaming = read(directory.join("streaming-vectors.json"));
        assert_eq!(read_streaming_vectors_v1(&streaming).unwrap().len(), 21);
    }
}

#[test]
fn vectors_pass_the_served_frontend_checks() {
    for checkpoint in checkpoints() {
        let Some(root) = checkpoint.root() else {
            eprintln!(
                "{} and {} are unset, skipping frontend vectors",
                checkpoint.tokenizer_env, checkpoint.checkpoint_env
            );
            continue;
        };
        let declaration = FrontendDescriptorV1::from_component_bytes(COMPONENT).unwrap();
        let tokenizer = tokenizer_from_components_v1(
            &declaration,
            &read(root.join(&declaration.tokenizer.definition.name)),
            &read(root.join(&declaration.tokenizer.configuration.name)),
        )
        .unwrap();
        let check = FrontendCheckV1::from_component_v1(&export(), COMPONENT, tokenizer).unwrap();
        let directory = fixtures(&checkpoint.name);
        let render: Vec<_> = RENDER
            .iter()
            .flat_map(|name| read_render_vectors_v1(&read(directory.join(name))).unwrap())
            .collect();
        let outcomes = check.run_v1(
            &render,
            &read_tokenizer_vectors_v1(&read(directory.join("tokenizer-golden.json"))).unwrap(),
            &read_streaming_vectors_v1(&read(directory.join("streaming-vectors.json"))).unwrap(),
        );
        let failed: Vec<_> = outcomes
            .iter()
            .filter(|outcome| outcome.result.is_err())
            .map(ToString::to_string)
            .collect();
        assert!(failed.is_empty(), "{}", failed.join("\n"));
        assert_eq!(
            outcomes.len(),
            checkpoint.chat_cases + 4 + checkpoint.tokenizer_cases + 21
        );
    }
}
