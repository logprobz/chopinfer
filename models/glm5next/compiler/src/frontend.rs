//! The checkpoint's declared chat frontend.

use runtime_frontend_contract::FrontendDescriptorV1;

/// The frontend component, rendered against revision eb9eb208's chat
/// template by `tests/frontend_goldens.rs`.
pub const COMPONENT: &[u8] = include_bytes!("frontend.json");

pub fn descriptor() -> FrontendDescriptorV1 {
    FrontendDescriptorV1::from_component_bytes(COMPONENT)
        .expect("the pinned frontend declaration is valid")
}
