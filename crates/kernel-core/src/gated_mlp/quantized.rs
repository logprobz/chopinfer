//! F32 clamped SwiGLU followed by one BF16 rounding and ceil E4M3/32.

use crate::contract::{NumericalContract, NumericalContractId, Refusal, Validate};
use crate::gated_mlp::SwigluClamp;
use crate::precision::quantizer::BlockQuantization;
use crate::quantize::Quantized;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct SiluMulQuantizeQuery {
    pub width: u32,
    pub clamp: SwigluClamp,
    pub amax_floor_bits: u32,
}

impl Validate for SiluMulQuantizeQuery {
    fn validate(&self) -> Result<(), Refusal> {
        if self.width == 0
            || !self.width.is_multiple_of(32)
            || !self.clamp.gate_max().is_finite()
            || self.clamp.gate_max() <= 0.0
            || !self.clamp.up_limit().is_finite()
            || self.clamp.up_limit() <= 0.0
        {
            return Err(Refusal(
                "quantized SwiGLU requires complete 32-value groups and finite positive clamps",
            ));
        }
        BlockQuantization::E4M3Ceil32 {
            amax_floor_bits: self.amax_floor_bits,
        }
        .quantizer()?;
        Ok(())
    }
}

impl NumericalContract for SiluMulQuantizeQuery {
    fn numerical_contract_id(&self) -> Result<NumericalContractId, Refusal> {
        self.validate()?;
        Ok(NumericalContractId::new(
            "silu-mul.clamp-f32.round-bf16.e4m3-ceil-ue8m0-32.v1".into(),
        ))
    }
}

impl SiluMulQuantizeQuery {
    /// F64 continuous-math oracle before the BF16 quantizer input boundary.
    pub fn oracle(self, gate: &[f32], up: &[f32], rows: usize) -> Result<Vec<f64>, Refusal> {
        self.check_inputs(gate, up, rows)?;
        let gate: Vec<_> = gate.iter().copied().map(f64::from).collect();
        let up: Vec<_> = up.iter().copied().map(f64::from).collect();
        crate::gated_mlp_oracle::clamped_silu_glu(Some(self.clamp), None).apply(&gate, &up)
    }

    fn check_inputs(self, gate: &[f32], up: &[f32], rows: usize) -> Result<(), Refusal> {
        self.validate()?;
        if rows == 0
            || rows.checked_mul(self.width as usize) != Some(gate.len())
            || gate.len() != up.len()
            || gate.iter().chain(up).any(|v| !v.is_finite())
        {
            return Err(Refusal(
                "quantized SwiGLU needs finite F32 gate and up matrices",
            ));
        }
        Ok(())
    }

    /// `g = min(gate, gate_max)`, `u = clamp(up, -limit, limit)`, then
    /// `(g / (1 + exp(-g))) * u`, each step in F32. Round the product to
    /// BF16 before amax and code selection. The SiLU result is not rounded
    /// to BF16 separately. Non-finite inputs are refused, never clamped away.
    pub fn apply(self, gate: &[f32], up: &[f32], rows: usize) -> Result<Quantized, Refusal> {
        self.check_inputs(gate, up, rows)?;
        let product: Vec<_> = gate
            .iter()
            .zip(up)
            .map(|(&g, &u)| {
                let g = g.min(self.clamp.gate_max());
                let u = u.clamp(-self.clamp.up_limit(), self.clamp.up_limit());
                (g / (1.0 + (-g).exp())) * u
            })
            .collect();
        BlockQuantization::E4M3Ceil32 {
            amax_floor_bits: self.amax_floor_bits,
        }
        .apply(&product, rows, self.width as usize)
    }
}

crate::scratch_record! {
    /// F32 gate/up `[rows, width]`, E4M3 output codes and UE8M0 row scales
    /// `[rows, width / 32]`. The I32 live count protects all unused rows.
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct SiluMulQuantizeBuffers {
        pub gate: value,
        pub up: value,
        pub codes: value,
        pub scales: value,
        pub live_rows: value,
    }
}
