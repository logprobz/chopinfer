//! F32 clamped SwiGLU followed by one BF16 rounding and ceil E4M3/32.

use crate::codec::{bf16_to_f32, f32_to_bf16};
use crate::contract::{DType, NumericalContract, NumericalContractId, Refusal, Validate};
use crate::gated_mlp::SwigluClamp;
use crate::precision::quantizer::{
    BlockQuantization, BlockQuantizeQuery, INVALID_UE8M0_SCALE, NONFINITE_INPUT, QuantizeOutput,
    UNREPRESENTABLE_OUTPUT,
};
use crate::quantize::{Quantized, Scales};

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
            "silu-mul.clamp-f32.round-bf16.e4m3-ceil-ue8m0-32.status-or-zero.v2".into(),
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
    /// to BF16 separately. Invalid rows are positive zero at fixed scale 1.
    /// Use `apply_with_status` to retain and inspect their sticky flags.
    pub fn apply(self, gate: &[f32], up: &[f32], rows: usize) -> Result<Quantized, Refusal> {
        self.apply_with_status(gate, up, &vec![0; rows])
            .map(|(q, _)| q)
    }

    fn products(
        self,
        gate: &[f32],
        up: &[f32],
        prior: &[u32],
    ) -> Result<(Vec<f32>, Vec<u32>), Refusal> {
        self.validate()?;
        let width = self.width as usize;
        if prior.len().checked_mul(width) != Some(gate.len()) || gate.len() != up.len() {
            return Err(Refusal(
                "quantized SwiGLU status extent differs from input rows",
            ));
        }
        let mut status = prior.to_vec();
        let product: Vec<_> = gate
            .iter()
            .zip(up)
            .enumerate()
            .map(|(at, (&g, &u))| {
                if !g.is_finite() || !u.is_finite() {
                    status[at / width] |= NONFINITE_INPUT;
                    return 0.0;
                }
                let g = g.min(self.clamp.gate_max());
                let u = u.clamp(-self.clamp.up_limit(), self.clamp.up_limit());
                let product = (g / (1.0 + (-g).exp())) * u;
                if !bf16_to_f32(f32_to_bf16(product)).is_finite() {
                    status[at / width] |= UNREPRESENTABLE_OUTPUT;
                    return 0.0;
                }
                product
            })
            .collect();
        let status = BlockQuantizeQuery {
            width: self.width,
            input: DType::F32,
            format: self.format(),
            output: QuantizeOutput::Codes,
        }
        .row_status(&product, &status)?;
        Ok((product, status))
    }

    fn format(self) -> BlockQuantization {
        BlockQuantization::E4M3Ceil32 {
            amax_floor_bits: self.amax_floor_bits,
        }
    }

    /// OR raw non-finite input and unrepresentable product/output flags
    /// into every prior word. A flagged operand skips the product entirely.
    pub fn row_status(self, gate: &[f32], up: &[f32], prior: &[u32]) -> Result<Vec<u32>, Refusal> {
        self.products(gate, up, prior).map(|(_, status)| status)
    }

    /// Status oracle and quantized bytes. All prior bits survive. A prior
    /// or new flag zeros the whole row, including groups without bad inputs.
    pub fn apply_with_status(
        self,
        gate: &[f32],
        up: &[f32],
        prior: &[u32],
    ) -> Result<(Quantized, Vec<u32>), Refusal> {
        let (mut product, status) = self.products(gate, up, prior)?;
        let width = self.width as usize;
        for (row, &flag) in product.chunks_exact_mut(width).zip(&status) {
            if flag != 0 {
                row.fill(0.0);
            }
        }
        let mut output = self.format().apply(&product, prior.len(), width)?;
        let Scales::Ue8M0(scales) = &mut output.scales else {
            unreachable!()
        };
        for (row, &flag) in status.iter().enumerate() {
            if flag != 0 {
                output.codes[row * width..(row + 1) * width].fill(0);
                scales[row * width / 32..(row + 1) * width / 32].fill(INVALID_UE8M0_SCALE);
            }
        }
        Ok((output, status))
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
        /// Caller-owned U32 words. Clear once before ordered launches and
        /// check before consuming output. Every live row atomically ORs
        /// flags; any prior or new flag zeros its codes with UE8M0 scale 127.
        /// Must be distinct from every other role and cover offset plus
        /// every planned row, including inactive rows. Missing status refuses.
        pub status: value,
        /// Offset in U32 words, with checked `(offset + rows) * 4` extent.
        pub status_row_offset: scalar(u64),
    }
}
