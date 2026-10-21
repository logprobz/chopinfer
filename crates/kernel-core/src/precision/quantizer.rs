//! Row quantization with BF16 rounding before every discrete decision.

use crate::codec::{SmallFloat, bf16_to_f32, f32_to_bf16};
use crate::contract::{DType, NumericalContract, NumericalContractId, Refusal, Validate};
use crate::quantize::{AmaxRatio, Quantized, Quantizer, ScaleRule, quantize};

use super::DynamicScale;

/// These formats have no tensor-global scale. E4M3 scales use RNE; UE8M0
/// scales use the exponent-bit ceiling of the F32 rounded reciprocal product.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum BlockQuantization {
    E4M3Ceil32 {
        amax_floor_bits: u32,
    },
    /// E2M1 per 16, E4M3 scale = RNE_E4M3(max(amax, 6 * 2^-9) / 6).
    E2M1E4M3Scale16,
    /// E2M1 per 32, UE8M0 ceil scale, amax floor 6 * 2^-126.
    E2M1Ceil32,
}

impl BlockQuantization {
    pub const E4M3_CEIL32: Self = Self::E4M3Ceil32 {
        amax_floor_bits: 1e-4_f32.to_bits(),
    };

    pub const fn group_size(self) -> u32 {
        match self {
            Self::E2M1E4M3Scale16 => 16,
            _ => 32,
        }
    }

    pub fn quantizer(self) -> Result<Quantizer, Refusal> {
        let (mut quantizer, floor) = match self {
            Self::E4M3Ceil32 { amax_floor_bits } => {
                DynamicScale::CeilPowerOfTwo { amax_floor_bits }
                    .validate()
                    .map_err(|_| Refusal("invalid ceil E4M3 amax floor"))?;
                (
                    Quantizer::mx(SmallFloat::E4M3),
                    f32::from_bits(amax_floor_bits),
                )
            }
            Self::E2M1E4M3Scale16 => (Quantizer::nvfp4(1.0), 6.0 * 2_f32.powi(-9)),
            Self::E2M1Ceil32 => (Quantizer::mx(SmallFloat::E2M1), 6.0 * f32::MIN_POSITIVE),
        };
        if self != Self::E2M1E4M3Scale16 {
            quantizer.rule = ScaleRule::CeilPowerOfTwo(AmaxRatio::MultiplyReciprocal);
        }
        quantizer.amax_floor = Some(floor);
        Ok(quantizer)
    }

    /// The exact code/scale rule. F32 inputs round to BF16 before amax and
    /// element quantization. NaNs, infinities and BF16 overflow are refused.
    /// E2M1/E4M3 scales saturate at 448, then payloads clamp to +/-6,
    /// matching the pinned TileLang SATFINITE cast
    /// (`docs/numerics.md#row-status`).
    pub fn apply(self, values: &[f32], rows: usize, width: usize) -> Result<Quantized, Refusal> {
        if !width.is_multiple_of(self.group_size() as usize) {
            return Err(Refusal("block quantization requires complete row groups"));
        }
        if values.iter().any(|v| !v.is_finite()) {
            return Err(Refusal("block quantization requires finite inputs"));
        }
        let rounded: Vec<_> = values
            .iter()
            .map(|&v| bf16_to_f32(f32_to_bf16(v)))
            .collect();
        quantize(&rounded, rows, width, self.quantizer()?)
    }

    /// The fake-quant form stores the same quantized values in BF16. Refuse
    /// values that overflow BF16 or lose a nonzero bit in its subnormal range.
    pub fn fake_quant(
        self,
        values: &[f32],
        rows: usize,
        width: usize,
    ) -> Result<Vec<u16>, Refusal> {
        self.apply(values, rows, width)?
            .decoded()
            .into_iter()
            .map(|v| {
                let bits = f32_to_bf16(v as f32);
                let decoded = f64::from(bf16_to_f32(bits));
                if !decoded.is_finite() || decoded.to_bits() != v.to_bits() {
                    Err(Refusal(
                        "fake-quant operand must be finite and exact in BF16",
                    ))
                } else {
                    Ok(bits)
                }
            })
            .collect()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum QuantizeOutput {
    Codes,
    Bf16,
}

/// Sticky row status bits. Callers clear the status plane once before an
/// ordered sequence of launches and check it before consuming the results.
pub const NONFINITE_INPUT: u32 = 1;
/// Reserved for caller-reported scale failures. Finite Format-1 scale
/// saturation is defined and does not raise this bit.
pub const SCALE_OVERFLOW: u32 = 2;
pub const UNREPRESENTABLE_OUTPUT: u32 = 4;
/// Invalid code rows use positive zero codes and finite scale 1 in either
/// encoding, never NaN scale 255. Every shared downstream decoder accepts
/// this pair, including the MXFP4 and chained E4M3 decode guards.
pub const INVALID_UE8M0_SCALE: u8 = 127;
pub const INVALID_E4M3_SCALE: u8 = 0x38;

/// Each live row is independent. `Bf16` writes the fake-quantized values;
/// `Codes` writes E4M3 bytes or even-low E2M1 nibbles plus scale bytes.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct BlockQuantizeQuery {
    pub width: u32,
    pub input: DType,
    pub format: BlockQuantization,
    pub output: QuantizeOutput,
}

impl Validate for BlockQuantizeQuery {
    fn validate(&self) -> Result<(), Refusal> {
        if self.width == 0
            || !self.width.is_multiple_of(self.format.group_size())
            || !matches!(self.input, DType::Bf16 | DType::F32)
        {
            return Err(Refusal(
                "block quantization needs BF16 or F32 input and complete row groups",
            ));
        }
        self.format.quantizer()?;
        Ok(())
    }
}

impl NumericalContract for BlockQuantizeQuery {
    fn numerical_contract_id(&self) -> Result<NumericalContractId, Refusal> {
        self.validate()?;
        let format = match self.format {
            BlockQuantization::E4M3Ceil32 { .. } => "e4m3-ceil-ue8m0-32",
            BlockQuantization::E2M1E4M3Scale16 => "e2m1-e4m3-16",
            BlockQuantization::E2M1Ceil32 => "e2m1-ceil-ue8m0-32",
        };
        let output = match self.output {
            QuantizeOutput::Codes => "codes",
            QuantizeOutput::Bf16 => "bf16",
        };
        Ok(NumericalContractId::new(format!(
            "quantize.bf16-input.{format}.{output}.status-or-zero.v2"
        )))
    }
}

crate::scratch_record! {
    /// Dense input/output rows, one optional scale byte per group and an
    /// U32 device live count. Fake-quant output has no scale buffer. No row
    /// at or past `live_rows` may be read or written.
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct BlockQuantizeBuffers {
        pub input: value,
        pub output: value,
        pub scales: option,
        pub live_rows: value,
        /// U32 words, one per launch row, with atomic OR updates. Distinct
        /// from input, output, scales and the live count. Non-live words
        /// stay unchanged. A nonzero prior or newly raised status makes the
        /// whole row positive zero; code output uses the fixed scale 1.
        pub status: value,
        /// Word offset into status. Only offset..offset+live_rows changes.
        pub status_row_offset: scalar(u64),
    }
}

impl BlockQuantizeQuery {
    /// Status oracle over decoded input rows. F32-to-BF16 overflow is a
    /// non-finite quantizer input. Finite E2M1/E4M3 scales saturate at 448.
    /// Both output forms report values not finite and exact in BF16, so
    /// packed results cannot trip the shared decoders' range guards.
    /// Each flag is independent and ORs with all caller-supplied bits.
    pub fn row_status(self, values: &[f32], prior: &[u32]) -> Result<Vec<u32>, Refusal> {
        self.validate()?;
        let width = self.width as usize;
        if prior.len().checked_mul(width) != Some(values.len()) {
            return Err(Refusal("quantizer status extent differs from input rows"));
        }
        values
            .chunks_exact(width)
            .zip(prior)
            .map(|(row, &prior)| {
                let mut status = prior;
                for group in row.chunks_exact(self.format.group_size() as usize) {
                    let mut safe = Vec::with_capacity(group.len());
                    for &value in group {
                        let rounded = bf16_to_f32(f32_to_bf16(value));
                        if !value.is_finite() || !rounded.is_finite() {
                            status |= NONFINITE_INPUT;
                            safe.push(0.0);
                        } else {
                            safe.push(rounded);
                        }
                    }
                    if self.format.fake_quant(&safe, 1, safe.len()).is_err() {
                        status |= UNREPRESENTABLE_OUTPUT;
                    }
                }
                Ok(status)
            })
            .collect()
    }

    pub const fn invalid_scale_code(self) -> u8 {
        match self.format {
            BlockQuantization::E2M1E4M3Scale16 => INVALID_E4M3_SCALE,
            _ => INVALID_UE8M0_SCALE,
        }
    }
}

/// The caller-owned status plane must cover offset plus every planned row.
pub fn status_extent(row_offset: u64, rows: u32) -> Result<u64, Refusal> {
    row_offset
        .checked_add(u64::from(rows))
        .and_then(|n| n.checked_mul(4))
        .ok_or(Refusal("quantizer status extent overflows"))
}
