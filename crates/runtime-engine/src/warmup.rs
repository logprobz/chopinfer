//! Shapes that startup warmup drives through the engine; the shared rule
//! lives in `runtime_bundle_sdk::warmup`, so a bundle's CPU conformance
//! scenario drives the same shapes.
pub use runtime_bundle_sdk::warmup::{VerifyShape, VerifyWarmup, context_ladder, verify_shapes};

/// One step the engine executed during startup warmup: its rows by phase
/// and the draft tokens its verify items accepted. Warmup reads these to
/// end a verify wave once it has run what the wave is for.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct StartupStep {
    pub prefill_tokens: usize,
    pub decode_rows: usize,
    pub verify_items: usize,
    pub verify_rows: usize,
    pub accepted_tokens: usize,
}
