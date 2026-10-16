//! Shapes that startup warmup drives through the engine; the shared rule
//! lives in `runtime_bundle_sdk::warmup`, so a bundle's CPU conformance
//! scenario drives the same shapes.
pub use runtime_bundle_sdk::warmup::{VerifyShape, VerifyWarmup, context_ladder, verify_shapes};
