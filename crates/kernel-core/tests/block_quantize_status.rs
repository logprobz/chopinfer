use kernel_core::codec::{bf16_to_f32, f32_to_bf16};
use kernel_core::contract::{DType, NumericalContract};
use kernel_core::precision::quantizer::*;

fn query(input: DType, output: QuantizeOutput, format: BlockQuantization) -> BlockQuantizeQuery {
    BlockQuantizeQuery {
        width: 64,
        input,
        output,
        format,
    }
}

#[test]
fn status_oracle_covers_nonfinite_inputs_rounding_and_scale_boundaries() {
    for dtype in [DType::F32, DType::Bf16] {
        for output in [QuantizeOutput::Codes, QuantizeOutput::Bf16] {
            let q = query(dtype, output, BlockQuantization::E2M1E4M3Scale16);
            for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
                let mut row = vec![1.; 64];
                row[63] = bad;
                assert_eq!(q.row_status(&row, &[0]).unwrap(), [NONFINITE_INPUT]);
                row[0] = 2704.;
                assert_eq!(
                    q.row_status(&row, &[0x80]).unwrap(),
                    [0x80 | NONFINITE_INPUT]
                );
            }
            for (value, flag) in [
                (2688., 0),
                (f32::from_bits(2688_f32.to_bits() + 1), 0),
                (2704., 0),
            ] {
                let value = if dtype == DType::Bf16 {
                    bf16_to_f32(f32_to_bf16(value))
                } else {
                    value
                };
                assert_eq!(q.row_status(&vec![value; 64], &[0]).unwrap(), [flag]);
            }
            assert_eq!(q.row_status(&vec![1.; 64], &[0x82]).unwrap(), [0x82]);
            assert!(q.row_status(&[1.; 63], &[0]).is_err());
            assert!(q.row_status(&[1.; 64], &[]).is_err());
            assert!(
                q.numerical_contract_id()
                    .unwrap()
                    .as_str()
                    .ends_with("status-or-zero.v2")
            );
        }
    }
}

#[test]
fn status_oracle_reports_fake_output_loss_and_bf16_input_overflow() {
    for format in [
        BlockQuantization::E4M3_CEIL32,
        BlockQuantization::E2M1Ceil32,
    ] {
        let fake = query(DType::F32, QuantizeOutput::Bf16, format);
        let max = bf16_to_f32(0x7f7f);
        assert_eq!(
            fake.row_status(&vec![max; 64], &[0]).unwrap(),
            [UNREPRESENTABLE_OUTPUT]
        );
        let codes = query(DType::F32, QuantizeOutput::Codes, format);
        assert_eq!(
            codes.row_status(&vec![max; 64], &[0]).unwrap(),
            [UNREPRESENTABLE_OUTPUT]
        );
        assert_eq!(
            fake.row_status(&vec![f32::MAX; 64], &[0]).unwrap(),
            [NONFINITE_INPUT]
        );
        let mut mixed = vec![max; 64];
        mixed[0] = f32::NAN;
        assert_eq!(
            codes.row_status(&mixed, &[SCALE_OVERFLOW]).unwrap(),
            [NONFINITE_INPUT | SCALE_OVERFLOW | UNREPRESENTABLE_OUTPUT]
        );
        assert_eq!(fake.invalid_scale_code(), 127);
    }
    assert_eq!(
        query(
            DType::Bf16,
            QuantizeOutput::Codes,
            BlockQuantization::E2M1E4M3Scale16
        )
        .invalid_scale_code(),
        0x38
    );
}

#[test]
fn status_offsets_are_words_and_extent_arithmetic_is_checked() {
    assert_eq!(status_extent(3, 16).unwrap(), 76);
    assert_eq!(
        status_extent((1_u64 << 32) + 7, 3).unwrap(),
        ((1_u64 << 32) + 10) * 4
    );
    assert!(status_extent(u64::MAX, 1).is_err());
    assert!(status_extent(u64::MAX / 4, 1).is_err());
}
