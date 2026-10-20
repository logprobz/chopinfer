//! The drafter's argmax partial with 1024 threads per CTA.
//!
//! Each drafting position reduces one vocabulary shard row per rank in a
//! single CTA. The shared partial's key order is total (ties to the lower
//! id), so the wider CTA returns the same pair as the 256-thread entry,
//! which stays compiled for the lab comparison.
use kernel_build::CubinSpec;
use kernels_sm120::row::argmax;

pub const FAMILY: &str = "dspark_argmax_wide";
pub const CUBIN: &str = "dspark_argmax_wide_sm120";
pub const SYMBOL: &str = "chop_dspark_argmax_f32_wide_v1";
pub const THREADS: u32 = 1024;

pub fn partial() -> argmax::Entry {
    argmax::Entry::new(argmax::Pattern::PartialF32, SYMBOL)
        .with_threads(THREADS)
        .expect("a whole-warp argmax width")
}

pub fn cubin_spec() -> CubinSpec {
    argmax::cubin_spec(CUBIN, &[partial()])
}
