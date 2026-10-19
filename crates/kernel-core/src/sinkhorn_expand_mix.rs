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

crate::scratch_record! {
    #[derive(Clone, Copy, Debug)]
    pub struct SinkhornRoundExpandMixBuffers {
        pub x: value,
        /// F32 sum, converted to BF16 before every expansion operand.
        pub sum: value,
        /// The original round output, still written for live rows.
        pub rounded: value,
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

/// The rounded producer and lagged mixer followed by a BF16 input norm.
/// The collapsed BF16 row remains observable before normalization. Each
/// constituent retains its reduction, rounding and status contract.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SinkhornRoundNormQuery {
    pub mixer: SinkhornExpandMixQuery,
    pub norm: crate::rmsnorm_quantize::RmsNormQuantizeQuery,
}
impl Validate for SinkhornRoundNormQuery {
    fn validate(&self) -> Result<(), Refusal> {
        self.mixer.validate()?;
        self.norm.validate()?;
        if self.mixer.mix.width != self.norm.norm.width
            || self.norm.norm.input != crate::contract::DType::Bf16
        {
            return Err(Refusal("rounded mixer and norm need matching BF16 rows"));
        }
        Ok(())
    }
}
use crate::rmsnorm_quantize::RmsNormQuantizeBuffers;
crate::scratch_record! {
    #[derive(Clone, Copy, Debug)]
    pub struct SinkhornRoundNormBuffers {
        pub hc: record(SinkhornRoundExpandMixBuffers),
        /// input and live_rows must equal hc.collapsed and hc.live_rows.
        /// Every additional norm role is distinct from HC roles/workspace.
        pub norm: record(RmsNormQuantizeBuffers),
    }
}
