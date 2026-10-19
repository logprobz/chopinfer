//! Composition of plain RMSNorm, optional exact widening, and ceil E4M3/32.
//!
//! The norm retains its declared output and rounding point. Quantization
//! consumes that output using `precision::quantizer::BlockQuantizeQuery`.
//! The two existing oracles define the composition. BF16 norms do not touch
//! status; F32 norms commit their flags before quantization. Quantization
//! keeps sticky row status and zeros codes with scale one on failure.
//! Inactive rows and status words outside the requested range are untouched.

use crate::contract::{DType, Refusal, RmsNormKind, Validate};
use crate::precision::quantizer::{BlockQuantization, BlockQuantizeQuery, QuantizeOutput};
use crate::rmsnorm::RmsNormRequirement;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RmsNormQuantizeQuery {
    pub norm: RmsNormRequirement,
    pub amax_floor_bits: u32,
}

impl RmsNormQuantizeQuery {
    pub fn quantizer(&self) -> BlockQuantizeQuery {
        BlockQuantizeQuery {
            width: self.norm.width,
            input: self.norm.output,
            format: BlockQuantization::E4M3Ceil32 {
                amax_floor_bits: self.amax_floor_bits,
            },
            output: QuantizeOutput::Codes,
        }
    }
}

impl Validate for RmsNormQuantizeQuery {
    fn validate(&self) -> Result<(), Refusal> {
        self.norm.validate()?;
        self.quantizer().validate()?;
        if self.norm.kind != RmsNormKind::Plain
            || self.norm.fused_residual
            || self.norm.output_quantization.is_some()
            || !matches!(
                (self.norm.input, self.norm.output),
                (DType::Bf16, DType::Bf16) | (DType::F32, DType::F32)
            )
        {
            return Err(Refusal(
                "norm quantization requires a plain matching-format norm",
            ));
        }
        Ok(())
    }
}

crate::scratch_record! {
    #[derive(Clone, Copy, Debug)]
    pub struct RmsNormQuantizeBuffers {
        pub input: value,
        pub weight: value,
        pub normed: value,
        /// Optional F32 copy of a BF16 norm's output, before quantization.
        pub widened: option,
        pub codes: value,
        pub scales: value,
        pub live_rows: value,
        pub status: value,
        pub status_row_offset: scalar(u64),
    }
}
