//! The declared chat frontend.

use runtime_frontend_contract::FrontendDescriptorV1;

/// The frontend component, rendered by `model_deepseek41_contract::frontend`
/// and checked against the vendor's encoder pairs by
/// `tests/frontend_goldens.rs`. Its `tokenizer` and `vocabulary` sections
/// are `tokenizer-contract.json`'s.
pub const COMPONENT: &[u8] = include_bytes!("frontend.json");

pub fn descriptor() -> FrontendDescriptorV1 {
    FrontendDescriptorV1::from_component_bytes(COMPONENT)
        .expect("the pinned frontend declaration is valid")
}
