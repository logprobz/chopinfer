//! Exact operands for finite F32 reductions and sticky arithmetic failures.
use crate::LabResult;

pub(super) struct Operands {
    pub input: Vec<f32>,
    pub weight: Vec<u16>,
    pub prior: Vec<u32>,
}

pub(super) fn operands(width: usize, output_overflow: bool) -> LabResult<Operands> {
    if width < 2 {
        return Err("norm status boundary requires at least two columns".into());
    }
    let mut input = vec![0.0; 8 * width];
    for row in [1, 2, 3, 5, 7] {
        input[row * width] = 1.0;
    }
    if !output_overflow {
        for row in [0, 3] {
            input[row * width] = f32::from_bits(0x5f7ffff3);
            input[row * width + 1] = f32::from_bits(0x5a9ffffe);
        }
        for row in [1, 2, 5] {
            input[row * width] = f32::MAX;
        }
    }
    // The first output-overflow row stays zero, so a live prefix of one
    // still observes the window store. Later rows overflow after weighting.
    input[4 * width] = f32::NAN;
    Ok(Operands {
        input,
        weight: vec![if output_overflow { 0x7f7f } else { 0x3f80 }; width],
        prior: vec![0, 0, 1, 1, 4, 0x80, 0, 0],
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use kernel_core::contract::{DType, RmsNormKind};
    use kernel_core::latent_glue::{LatentWindowWrite, WindowAddressing};
    use kernel_core::rmsnorm::RmsNormRequirement;

    #[test]
    fn both_norm_fixtures_expose_finite_reductions_and_new_flags() {
        let norm = RmsNormRequirement {
            input: DType::F32,
            output: DType::F32,
            ..RmsNormRequirement::bf16_backend_chosen(512, 1e-20, RmsNormKind::Plain, false)
        };
        let window = LatentWindowWrite {
            kv: kernel_core::latent_attention::LatentKvFormat::Fp8E4M3Ue8M0Block32,
            latent: 512,
            rotary: 64,
            epsilon_bits: 1e-20_f32.to_bits(),
            addressing: WindowAddressing::Circular { slots: 136 },
        };
        for (overflow, wanted) in [
            (false, [0, 4, 5, 1, 5, 0x84, 0, 0]),
            (true, [0, 4, 5, 5, 5, 0x84, 0, 4]),
        ] {
            let data = operands(512, overflow).unwrap();
            let (value, status) =
                kernel_core::rmsnorm_status::oracle(norm, &data.input, &data.weight, &data.prior)
                    .unwrap();
            assert_eq!(status, wanted);
            let (latent, codes, status) = kernel_core::latent_glue_oracle::window_rows_status(
                window,
                &data.input,
                &data.weight,
                &[1.0, 0.0].repeat(8 * 32),
                &data.prior,
            )
            .unwrap();
            assert_eq!(status, wanted);
            if !overflow {
                assert!(value.values[0] > 20.0);
                assert!(latent[0] > 20.0);
                assert_ne!(codes[0].values[0], 0);
            }
        }
    }
}
