//! The drafter's coupled sampling noise (`kernels_sm120::row::draft_noise`)
//! in its own lock family.
//!
//! Each drafting position adds the target sampler's Gumbel noise to every
//! sampled item's row of the rank's vocabulary shard after the Markov step,
//! so the unchanged sharded argmax proposes the coupled draft. Greedy items'
//! rows keep their bits.
use kernel_build::CubinSpec;
use kernels_sm120::row::draft_noise;

pub const FAMILY: &str = "dspark_noise";
pub const CUBIN: &str = "dspark_noise_sm120";
pub use draft_noise::SYMBOL;

pub fn entry() -> draft_noise::Entry {
    draft_noise::Entry::new(SYMBOL)
}

pub fn cubin_spec() -> CubinSpec {
    draft_noise::cubin_spec(CUBIN, &[entry()])
}
