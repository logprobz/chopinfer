use kernel_core::codec::{SmallFloat, bf16_to_f32, f32_to_bf16};
use kernel_core::contract::{DType, WeightFormat};
use kernel_core::precision::packed::{MatrixView, ScaleBytes};
use kernel_core::precision::{DecodeOrder, NibbleOrder};

fn matrix<'a>(
    format: WeightFormat,
    codes: &'a [u8],
    scales: &'a [u8],
    global: &'a [u8],
) -> MatrixView<'a> {
    MatrixView {
        tensor: "test.weight",
        format,
        rows: 1,
        k: if matches!(format, WeightFormat::Nvfp4 | WeightFormat::Mxfp4) {
            codes.len() * 2
        } else {
            codes.len()
        },
        row_origin: 0,
        k_origin: 0,
        nibble_order: NibbleOrder::EvenLow,
        codes,
        local: Some(ScaleBytes {
            dtype: if format == WeightFormat::Nvfp4 {
                DType::Ue4M3
            } else {
                DType::Ue8M0
            },
            bytes: scales,
        }),
        global: (!global.is_empty()).then_some(ScaleBytes {
            dtype: DType::F32,
            bytes: global,
        }),
    }
}

#[test]
fn every_mxfp4_code_and_scale_requires_a_finite_exact_bf16_operand() {
    for code in 0..16 {
        for scale in 0..=255_u8 {
            let factor = if scale == 255 {
                f64::NAN
            } else {
                2_f64.powi(i32::from(scale) - 127)
            };
            let exact = SmallFloat::E2M1.decode_f64(code) * factor;
            let expected = kernel_core::codec::f64_to_bf16(exact);
            let admitted =
                exact.is_finite() && f64::from(bf16_to_f32(expected)).to_bits() == exact.to_bits();
            let codes = [code | code << 4; 16];
            let scales = [scale];
            let view = matrix(WeightFormat::Mxfp4, &codes, &scales, &[]);
            assert_eq!(
                view.validate_bf16_operands(DecodeOrder::ExactScaledOperand)
                    .is_ok(),
                admitted,
                "admission: code {code}, scale {scale}"
            );
            let operand = view.bf16_operand(0, 0, DecodeOrder::ExactScaledOperand);
            assert_eq!(operand.is_ok(), admitted, "code {code}, scale {scale}");
            if admitted {
                assert_eq!(operand.unwrap(), expected, "code {code}, scale {scale}");
            }
        }
    }
}

#[test]
fn every_finite_nvfp4_local_product_is_exact_in_bf16() {
    for code in 0..16 {
        for scale in 0..127 {
            let exact = SmallFloat::E2M1.decode_f64(code) * SmallFloat::Ue4M3.decode_f64(scale);
            let operand = bf16_to_f32(f32_to_bf16(exact as f32));
            assert_eq!(
                f64::from(operand).to_bits(),
                exact.to_bits(),
                "{code} * {scale}"
            );
            let codes = [code | code << 4; 8];
            let scales = [scale];
            let global = 0.0371_f32.to_le_bytes();
            let view = matrix(WeightFormat::Nvfp4, &codes, &scales, &global);
            view.validate_bf16_operands(DecodeOrder::ExactLocalThenGlobal)
                .unwrap();
            assert_eq!(
                view.bf16_operand(0, 0, DecodeOrder::ExactLocalThenGlobal)
                    .unwrap(),
                f32_to_bf16(operand)
            );
            assert_eq!(
                view.oracle_weight(0, 0, DecodeOrder::ExactLocalThenGlobal)
                    .unwrap(),
                exact * f64::from(0.0371_f32)
            );
        }
    }
}

#[test]
fn mxfp8_exactness_domain_includes_subnormals_but_not_every_scale() {
    let mut inexact = 0;
    for code in 0..=255_u8 {
        if code & 127 == 127 {
            continue;
        }
        for scale in 0..255 {
            let exact = SmallFloat::E4M3.decode_f64(code) * 2_f64.powi(scale - 127);
            let decoded = f64::from(bf16_to_f32(f32_to_bf16(exact as f32)));
            let codes = [code; 32];
            let scales = [scale as u8];
            let view = matrix(WeightFormat::Mxfp8, &codes, &scales, &[]);
            assert_eq!(
                view.validate_bf16_operands(DecodeOrder::ExactScaledOperand)
                    .is_ok(),
                decoded == exact,
                "payload admission for {code} * 2^{}",
                scale - 127
            );
            assert_eq!(
                view.bf16_operand(0, 0, DecodeOrder::ExactScaledOperand)
                    .is_ok(),
                decoded == exact
            );
            if decoded != exact {
                inexact += 1;
                // All finite E4M3 codes are exact for these scale exponents.
                assert!(!(3..=246).contains(&scale), "{code} * 2^{}", scale - 127);
                assert!(
                    exact.abs() < 2_f64.powi(-126) || exact.abs() > f64::from(bf16_to_f32(0x7f7f))
                );
            }
        }
    }
    assert!(inexact > 0);
}

#[test]
fn nvfp4_oracle_keeps_global_precision_and_lookup_rounds_only_once() {
    let codes = [0x77; 8];
    let scales = [0x7e];
    let global = 0.0371_f32.to_le_bytes();
    let view = matrix(WeightFormat::Nvfp4, &codes, &scales, &global);
    let exact = 6.0 * 448.0 * f64::from(0.0371_f32);
    assert_eq!(
        view.oracle_weight(0, 0, DecodeOrder::ExactLocalThenGlobal)
            .unwrap(),
        exact
    );
    assert_ne!(
        view.oracle_weight(0, 0, DecodeOrder::CodeThenScale)
            .unwrap(),
        exact
    );
    assert_eq!(
        view.bf16_operand(0, 0, DecodeOrder::ExactWeightToBf16)
            .unwrap(),
        kernel_core::codec::f64_to_bf16(exact)
    );
    assert!(
        view.validate_bf16_operands(DecodeOrder::ExactScaledOperand)
            .is_err()
    );
}

#[test]
fn direct_bf16_rounding_resolves_every_finite_midpoint_and_its_neighbors() {
    use kernel_core::codec::f64_to_bf16;
    for low in 0..0x7f7f {
        let a = f64::from(bf16_to_f32(low));
        let b = f64::from(bf16_to_f32(low + 1));
        let middle = (a + b) / 2.0;
        let below = f64::from_bits(middle.to_bits() - 1);
        let above = f64::from_bits(middle.to_bits() + 1);
        for (value, expected) in [(below, low), (middle, low + (low & 1)), (above, low + 1)] {
            assert_eq!(f64_to_bf16(value), expected);
            assert_eq!(f64_to_bf16(-value), expected | 0x8000);
        }
    }
    assert_eq!(f64_to_bf16(-0.0), 0x8000);
    assert_eq!(f64_to_bf16(2_f64.powi(128) - 2_f64.powi(119)), 0x7f80);
}
