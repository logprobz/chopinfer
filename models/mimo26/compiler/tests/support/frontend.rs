//! Runtime frontend construction for portable behavior tests and examples.
//! Every call runs both ways, MiMo's chat format called directly and
//! through its export's documents.
pub fn frontend() -> runtime_frontend::test_support::documents::PairedFrontendV1 {
    runtime_frontend::test_support::documents::PairedFrontendV1::chat_format::<
        model_mimo26_contract::frontend::Mimo26Format,
    >(model_mimo26::frontend::descriptor())
    .expect("the pinned frontend is valid")
}
