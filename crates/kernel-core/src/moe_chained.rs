//! Routed MXFP4 projections with ceil MXFP8 activations, route weights
//! before down, and 128-K chains with an explicitly zero-filled tail.

use crate::contract::{
    ActivationFormat, DType, NumericalContract, NumericalContractId, OutputQuantization, Refusal,
    SwigluColumns, Validate, WeightFormat,
};
use crate::gated_mlp::{GatedMlpProjection, GatedMlpProjectionQuery, RouteWeightPoint};

/// Stored K is the row stride for both operands, in elements. Per-32
/// scales have stride `stored / 32`. The loop closes whole 128-K chains,
/// loading +0 into both tiles at k >= stored without reading either the
/// codes or scales there. A stored K of 576 therefore has loop K 640:
/// four full chains and 64 real products plus 64 exact zeros in the fifth.
/// Padding exists only in tiles, never between stored rows.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct ChainedK {
    pub stored: u32,
    pub loop_k: u32,
}

impl ChainedK {
    pub fn new(stored: u32) -> Result<Self, Refusal> {
        let loop_k = stored
            .checked_next_multiple_of(128)
            .ok_or(Refusal("chained K padding overflows"))?;
        let value = Self { stored, loop_k };
        value.validate()?;
        Ok(value)
    }
}

impl Validate for ChainedK {
    fn validate(&self) -> Result<(), Refusal> {
        if self.stored == 0
            || !self.stored.is_multiple_of(64)
            || self.stored.checked_next_multiple_of(128) != Some(self.loop_k)
        {
            return Err(Refusal(
                "chained K needs whole 64-K stored tiles padded to the next 128",
            ));
        }
        Ok(())
    }
}

/// The routed projection's additional arithmetic contract. Both inputs
/// decode to exact finite BF16 operands: MXFP4 weights preserve the
/// actual-code overflow exclusions at UE8M0 bytes 253 and 254, and MXFP8
/// activations preserve finite and BF16-exact admission, including scales
/// outside 3..=246. Byte 255 is refused for either scale format.
///
/// Activations use E4M3 per 32 with UE8M0 scales under the ceil rule:
/// round to BF16 before amax, floor amax at 1e-4, then take the smallest
/// power of two >= fl32(amax * fl32(1/448)). This is not the OCP floor rule.
///
/// Gate/up clamps its F32 accumulators, computes `silu(g) * u`, then
/// `w * intermediate`, with a rounding after each F32 product and no
/// reassociation. It rounds to BF16 before the same ceil quantization.
/// Down applies no route weight again and writes each route's F32 partial.
/// Each 128-K chain starts at +0 and folds into F32 in ascending K order.
/// The per-token routed sum and rank-partial sum are separate operations.
/// Buffers use `moe_experts::GatedMlpBuffers`: quantized outputs go to
/// `scratch.output_codes` and `scratch.output_scales`, whose strides are
/// the unpadded intermediate width and that width divided by 32.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct ChainedMlpProjectionQuery {
    pub projection: GatedMlpProjectionQuery,
    pub k: ChainedK,
}

impl Validate for ChainedMlpProjectionQuery {
    fn validate(&self) -> Result<(), Refusal> {
        self.projection.validate()?;
        self.k.validate()?;
        let p = self.projection;
        let g = p.gemm;
        if g.k != self.k.stored {
            return Err(Refusal("chained GEMM K must equal the stored stride"));
        }
        if g.grouped.is_none() || p.route_weight != Some(RouteWeightPoint::BeforeDown) {
            return Err(Refusal(
                "quantized chained experts need routed weights before down",
            ));
        }
        if g.weight != WeightFormat::Mxfp4 || g.activation != ActivationFormat::Mxfp8 {
            return Err(Refusal(
                "quantized chained experts need MXFP4 weights and ceil MXFP8 activations",
            ));
        }
        let e = g.epilogues;
        if e.residual_add || e.bias || e.gelu.is_some() || e.weighted_moe_combine {
            return Err(Refusal(
                "quantized chained experts have no residual, bias, GELU or weighted combine",
            ));
        }
        match p.projection {
            GatedMlpProjection::GateUp
                if g.output == DType::Bf16
                    && e.swiglu == Some(SwigluColumns::ConcatenatedHalves)
                    && e.output_quantization == Some(OutputQuantization::Mxfp8)
                    && (g.n / 2).is_multiple_of(32) =>
            {
                Ok(())
            }
            GatedMlpProjection::Down
                if g.output == DType::F32 && e.output_quantization.is_none() =>
            {
                Ok(())
            }
            _ => Err(Refusal(
                "chained gate/up needs BF16 then ceil MXFP8 output; down needs F32 output",
            )),
        }
    }
}

impl NumericalContract for ChainedMlpProjectionQuery {
    fn numerical_contract_id(&self) -> Result<NumericalContractId, Refusal> {
        self.validate()?;
        Ok(NumericalContractId::new(format!(
            "{}.decoded-bf16.chain128.zero-tail.ceil-mxfp8-bf16-amax.v1",
            self.projection.numerical_contract_id()?
        )))
    }
}
