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

#[cfg(test)]
mod tests {
    use super::*;

    /// The serving defaults plan V6 fixes:
    /// temperature and top_p 1.0, thinking on at effort high, and the 1M
    /// context the checkpoint's `max_position_embeddings` gives. The
    /// checkpoint ships no generation_config.json. An omitted max_tokens
    /// fills the remaining context, as the transport does for every model.
    #[test]
    fn the_declaration_holds_the_planned_defaults() {
        let descriptor = descriptor();
        let sampling = descriptor.generation_config;
        assert_eq!(
            (sampling.temperature, sampling.top_p),
            (Some(1.0), Some(1.0))
        );
        assert_eq!(
            (sampling.top_k, sampling.min_p, sampling.repetition_penalty),
            (None, None, None)
        );
        assert!(descriptor.reasoning.default_enabled);
        assert_eq!(descriptor.reasoning.default_effort, "high");
        assert_eq!(descriptor.context.maximum_total_tokens, 1_048_576);
    }
}
