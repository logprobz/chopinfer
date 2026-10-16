//! MiMo v2.6's HF differential corpus through the served frontend
//! (`runtime_host::frontend_corpus`): every case's raw request renders the
//! reference's prompt text and token ids, or is refused as the case states,
//! and every round-trip turn decodes back to the reference turn, through the
//! transport and through the parser at every byte split.
//!
//! The committed corpus is `fixtures/mimo26/frontend/`. The full one (for
//! example `$CHOPINFER_DATA/frontend-corpus/mimo26/full`) runs with
//! `CHOPINFER_FRONTEND_CORPUS_DIR` set, before each minimization and after
//! every pin bump, and the outcomes are written to the file
//! `CHOPINFER_FRONTEND_CORPUS_DUMP` names. It runs against the served
//! full-encoder component of the default checkpoint, the one that labels
//! every media kind. The tokenizer is the checkpoint's, so the corpus runs
//! with `CHOPINFER_MIMO_TOKENIZER_DIR` or `CHOPINFER_MIMO_CHECKPOINT` set.

use std::path::Path;

use model_mimo26::checkpoint::default_identity;
use model_mimo26::frontend::component_for_flash;
use model_mimo26_contract::encoders::Encoders;
use model_mimo26_contract::frontend::Mimo26Format;
use runtime_host::frontend_corpus::{FrontendCorpusV1, directory_from_env_v1};

fn corpus() -> FrontendCorpusV1 {
    FrontendCorpusV1 {
        model: "MiMo v2.6",
        export: frontend_format::chat_format_export_v1::<Mimo26Format>(),
        component: component_for_flash(default_identity(), Encoders::ALL).unwrap(),
        committed: Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../fixtures/mimo26/frontend"),
        tokenizer_dir: directory_from_env_v1(&[
            "CHOPINFER_MIMO_TOKENIZER_DIR",
            "CHOPINFER_MIMO_CHECKPOINT",
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

/// The transport translation the corpus's HF inputs follow
/// (`models/mimo26/frontend_cases.py` `TRANSPORT`) is what the export opens
/// with.
#[test]
fn corpus_transport_is_the_export_open_answer() {
    corpus().check_transport();
}
