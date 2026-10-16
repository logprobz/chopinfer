//! Qwen3.8's HF differential corpus through the served frontend
//! (`runtime_host::frontend_corpus`): every case's raw request renders the
//! reference's prompt text and token ids, or is refused as the case states,
//! and every round-trip turn decodes back to the reference turn, through the
//! transport and through the parser at every byte split.
//!
//! The committed corpus is `fixtures/qwen38/frontend/`. The full one runs
//! with `CHOPINFER_FRONTEND_CORPUS_DIR` set, and the outcomes are written to
//! the file `CHOPINFER_FRONTEND_CORPUS_DUMP` names. The tokenizer is the
//! first indexed checkpoint's.

use std::path::Path;

use runtime_host::frontend_corpus::FrontendCorpusV1;

#[allow(dead_code)]
#[path = "../src/checkpoint_index.rs"]
mod checkpoint_index;

/// The bundle's export over the multimodal component, since the corpus
/// sends images.
fn corpus() -> FrontendCorpusV1 {
    FrontendCorpusV1 {
        model: "Qwen3.8",
        export: model_qwen38_contract::frontend::export(),
        component: model_qwen38::frontend::multimodal_component(),
        committed: Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../fixtures/qwen38/frontend"),
        tokenizer_dir: checkpoint_index::checkpoints()
            .into_iter()
            .next()
            .and_then(|checkpoint| checkpoint.root()),
    }
}

#[test]
fn committed_corpus_passes_the_served_checks() {
    corpus().check_committed();
}

#[test]
fn full_corpus_passes_the_served_checks() {
    corpus().check_full();
}

/// The transport translation the corpus's HF inputs follow
/// (`models/qwen38/frontend_cases.py` `TRANSPORT`) is what the export opens
/// with.
#[test]
fn corpus_transport_is_the_export_open_answer() {
    corpus().check_transport();
}
