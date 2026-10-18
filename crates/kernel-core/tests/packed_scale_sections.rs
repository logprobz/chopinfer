#![allow(clippy::single_range_in_vec_init)]
use kernel_core::contract::{DType, SwigluColumns, WeightFormat};
use kernel_core::precision::layout::PackingPlan;
use kernel_core::precision::{DecodeOrder, NibbleOrder, WeightLayout};
use kernel_core::weight_transform::{
    PackedMatrixSource, PackedOperand, PackedSection, SourceTensor, WeightRecord, WeightTransform,
};

fn record(order: DecodeOrder) -> (PackedOperand, WeightRecord, Vec<u8>, Vec<u8>) {
    let operand = PackedOperand {
        plan: PackingPlan {
            layout: WeightLayout {
                id: "row-major".into(),
                version: 1,
            },
            format: WeightFormat::Mxfp4,
            experts: 1,
            projections: 1,
            rows: 12,
            k: 96,
            columns: SwigluColumns::ConcatenatedHalves,
            local_scale_dtype: Some(DType::Ue8M0),
        },
        section: PackedSection::LocalScales,
        decode_order: order,
        matrices: vec![PackedMatrixSource {
            expert: 0,
            projection: 0,
            format: WeightFormat::Mxfp4,
            rows: 16,
            k: 256,
            row_origin: 7,
            k_origin: 32,
            nibble_order: NibbleOrder::EvenLow,
            codes: 0,
            local: Some(1),
            local_dtype: Some(DType::Ue8M0),
            global: None,
            source_expert: 0,
            source_experts: 1,
            rows_kept: vec![2..5, 8..13],
            k_kept: 64..128,
        }],
    };
    let record = WeightRecord {
        transform: WeightTransform::PackedOperand(Box::new(operand.clone())),
        sources: vec![
            SourceTensor {
                dtype: "U8".into(),
                shape: vec![16, 128],
                byte_length: 2048,
            },
            SourceTensor {
                dtype: "U8".into(),
                shape: vec![16, 8],
                byte_length: 128,
            },
        ],
        axis: None,
        ranges: vec![],
        stored_shape: vec![],
    };
    (
        operand,
        record,
        (0..2048).map(|i| (i * 37) as u8).collect(),
        (0..128).map(|i| 120 + (i % 7) as u8).collect(),
    )
}

#[test]
fn independent_scales_match_whole_packing_with_origins_slices_and_padding() {
    for order in [
        DecodeOrder::ExactScaledOperand,
        DecodeOrder::NativeBlockScaledMma,
    ] {
        let (mut operand, record, codes, scales) = record(order);
        for origin in [0, 32, 64] {
            operand.matrices[0].k_origin = origin;
            for start in [0, 32, 128, 192] {
                operand.matrices[0].k_kept = start..start + 64;
                let expected = operand
                    .execute_sections(&record, &[&codes, &scales])
                    .unwrap();
                let actual = operand
                    .independent_scales(&record, &scales, vec![255; 1000])
                    .unwrap()
                    .unwrap();
                assert_eq!(actual, expected[1]);
                let reused = operand
                    .execute_sections_reusing(
                        &record,
                        &[&codes, &scales],
                        std::array::from_fn(|_| vec![255; 8192]),
                    )
                    .unwrap();
                assert_eq!(reused, expected);
            }
        }
    }
}

#[test]
fn scale_section_keeps_full_source_validation_and_code_dependent_fallback() {
    let (operand, record, codes, mut scales) = record(DecodeOrder::ExactScaledOperand);
    // This scale lies outside both the selected rows and columns.
    scales[0] = 255;
    assert!(
        operand
            .execute_sections(&record, &[&codes, &scales])
            .is_err()
    );
    assert!(
        operand
            .independent_scales(&record, &scales, vec![])
            .is_err()
    );
    scales[0] = 254;
    assert!(
        operand
            .independent_scales(&record, &scales, vec![])
            .unwrap()
            .is_none()
    );
    scales[0] = 120;
    assert!(
        operand
            .independent_scales(&record, &scales[..127], vec![])
            .is_err()
    );
    for order in [DecodeOrder::ScaleThenCode, DecodeOrder::CodeThenScale] {
        let mut checked = operand.clone();
        checked.decode_order = order;
        assert_eq!(checked.independent_scale_source(&record), None);
        assert!(
            checked
                .independent_scales(&record, &scales, vec![])
                .unwrap()
                .is_none()
        );
    }
    for bad in 0..4 {
        let mut changed = operand.clone();
        match bad {
            0 => changed.matrices[0].k_kept = 240..272,
            1 => changed.matrices[0].rows_kept = vec![15..17],
            2 => changed.matrices[0].k_origin = u64::MAX - 31,
            _ => changed.matrices[0].source_experts = 2,
        }
        assert_eq!(changed.independent_scale_source(&record), None);
    }
}
