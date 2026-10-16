//! GLM 5.3 Flash's HF differential corpus through the served frontend
//! (`runtime_host::frontend_corpus`): every case's raw request renders the
//! reference's prompt text and token ids, or is refused as the case states,
//! and every round-trip turn decodes back to the reference turn, through the
//! transport and through the parser at every byte split.
//!
//! The committed corpus is `fixtures/glm5next/frontend/`, the subset
//! `frontend_corpus minimize` keeps. The full one runs with
//! `CHOPINFER_FRONTEND_CORPUS_DIR` set, and the outcomes are written to the
//! file `CHOPINFER_FRONTEND_CORPUS_DUMP` names. The tokenizer is the
//! checkpoint's, so the corpus runs with `CHOPINFER_GLM5NEXT_TOKENIZER_DIR`
//! or `CHOPINFER_GLM5NEXT_CHECKPOINT` set.

use std::path::Path;

use model_glm5next::frontend::COMPONENT;
use model_glm5next_contract::frontend::Glm53Format;
use runtime_host::frontend_corpus::{FrontendCorpusV1, directory_from_env_v1};

/// The package's own export over its component.
fn corpus() -> FrontendCorpusV1 {
    FrontendCorpusV1 {
        model: "GLM 5.3 Flash",
        export: frontend_format::chat_format_export_v1::<Glm53Format>(),
        component: COMPONENT.to_vec(),
        committed: Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../../fixtures/glm5next/frontend"),
        tokenizer_dir: directory_from_env_v1(&[
            "CHOPINFER_GLM5NEXT_TOKENIZER_DIR",
            "CHOPINFER_GLM5NEXT_CHECKPOINT",
        ]),
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

/// The transport translation the corpus renders HF's input with
/// (`models/glm5next/frontend_cases.py` `TRANSPORT`) is what the export
/// opens with.
#[test]
fn corpus_transport_is_the_export_open_answer() {
    corpus().check_transport();
}
