//! Exact composition of lagged F32 stream expansion and the next mixer.
//!
//! Expansion stores BF16 before either projection or RMS reads its values.
//! The mixer consumes the previous sublayer's F32 collapse weights. Its
//! original 128-element projection chains, RMS lane chains and Sinkhorn
//! orders are preserved. Inactive rows are untouched; neither constituent
//! has a status plane. The existing expansion and mixer oracles define
//! the composition.
use crate::contract::{Refusal, Validate};
use crate::sinkhorn_hyper_connection::{HyperConnectionPlacement, SinkhornMix};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SinkhornExpandMixQuery {
    pub mix: SinkhornMix,
}
impl Validate for SinkhornExpandMixQuery {
    fn validate(&self) -> Result<(), Refusal> {
        self.mix.validate()?;
        if self.mix.placement != HyperConnectionPlacement::LaggedSinglePass {
            return Err(Refusal("expand-mix requires lagged placement"));
        }
        Ok(())
    }
}

crate::scratch_record! {
    #[derive(Clone, Copy, Debug)]
    pub struct SinkhornExpandMixBuffers {
        pub x: value,
        pub y: value,
        pub post_in: value,
        pub comb_in: value,
        pub expanded: value,
        pub fn_: value,
        pub base: value,
        pub scale: value,
        pub pre_in: value,
        pub collapsed: value,
        pub pre: value,
        /// May alias post_in: the partial kernel finishes reading it first.
        pub post: value,
        /// May alias comb_in under the same two-kernel lifetime rule.
        pub comb: value,
        pub live_rows: value,
    }
}
