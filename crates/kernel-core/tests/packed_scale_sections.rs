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

// Logical row-major oracle, independent of both materializers. Its seeded
// corpus is also checked against the pre-change implementation below.
fn reference(operand: &PackedOperand, codes: &[u8], scales: &[u8]) -> [Vec<u8>; 3] {
    let m = &operand.matrices[0];
    let rows = operand.plan.rows as usize;
    let k = operand.plan.k as usize;
    let mut output_codes = vec![0; rows * k / 2];
    let mut output_scales = vec![127; rows * k.div_ceil(32)];
    for (out_row, source_row) in m
        .rows_kept
        .iter()
        .flat_map(|range| range.clone())
        .enumerate()
    {
        for column in 0..m.k_kept.end - m.k_kept.start {
            let source = (source_row * m.k + m.k_kept.start + column) as usize;
            let shift = match m.nibble_order {
                NibbleOrder::EvenLow => (source % 2) * 4,
                NibbleOrder::EvenHigh => (1 - source % 2) * 4,
            };
            let code = (codes[source / 2] >> shift) & 15;
            let target = out_row * k + column as usize;
            output_codes[target / 2] |= code << ((target % 2) * 4);
        }
        for column in 0..(m.k_kept.end - m.k_kept.start).div_ceil(32) {
            let source = (source_row * (m.k / 32) + m.k_kept.start / 32 + column) as usize;
            output_scales[out_row * k.div_ceil(32) + column as usize] = scales[source];
        }
    }
    [output_codes, output_scales, vec![]]
}

fn seeded_cases() -> Vec<(PackedOperand, WeightRecord, Vec<u8>, Vec<u8>)> {
    let mut seed = 0x6d78_6670_345f_7470_u64;
    let mut next = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    (0..128)
        .map(|_| {
            let (mut operand, record, mut codes, mut scales) =
                record(DecodeOrder::ExactScaledOperand);
            for byte in &mut codes {
                *byte = next() as u8;
            }
            for byte in &mut scales {
                *byte = (next() % 252) as u8;
            }
            let matrix = &mut operand.matrices[0];
            let start = next() % 8;
            let rows = 1 + next() % (16 - start);
            matrix.rows_kept = vec![start..start + rows];
            matrix.row_origin = next() % 17;
            matrix.k_origin = (next() % 4) * 32;
            let start = (next() % 5) * 32;
            let k = 1 + next() % 96;
            matrix.k_kept = start..start + k;
            matrix.nibble_order = if next() % 2 == 0 {
                NibbleOrder::EvenLow
            } else {
                NibbleOrder::EvenHigh
            };
            operand.plan.rows = (rows + next() % 5) as u32;
            operand.plan.k = (k.div_ceil(32) * 32) as u32;
            (operand, record, codes, scales)
        })
        .collect()
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
                let expected = reference(&operand, &codes, &scales);
                assert_eq!(
                    operand
                        .execute_sections(&record, &[&codes, &scales])
                        .unwrap(),
                    expected
                );
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
fn seeded_sections_match_independent_bytes_and_the_pre_change_digest() {
    let mut digest = 0xcbf2_9ce4_8422_2325_u64;
    for (operand, record, codes, scales) in seeded_cases() {
        let expected = reference(&operand, &codes, &scales);
        assert_eq!(
            operand
                .execute_sections(&record, &[&codes, &scales])
                .unwrap(),
            expected
        );
        assert_eq!(
            operand
                .execute_sections_reusing(
                    &record,
                    &[&codes, &scales],
                    std::array::from_fn(|_| vec![255; 8192])
                )
                .unwrap(),
            expected
        );
        assert_eq!(
            operand
                .independent_scales(&record, &scales, vec![255; 8192])
                .unwrap()
                .unwrap(),
            expected[1]
        );
        for byte in expected.iter().flatten() {
            digest = (digest ^ u64::from(*byte)).wrapping_mul(0x100_0000_01b3);
        }
    }
    // Filled from the pre-change packer at b450a4ea1, never from a candidate.
    assert_eq!(digest, 0xa4e0_f18a_1038_ae61);
}

#[test]
fn reused_row_major_sections_match_the_logical_oracle() {
    for (operand, record, codes, scales) in seeded_cases() {
        let expected = reference(&operand, &codes, &scales);
        let mut storage = std::array::from_fn(|_| vec![255; 8192]);
        for _ in 0..3 {
            storage = operand
                .execute_sections_reusing(&record, &[&codes, &scales], storage)
                .unwrap();
            assert_eq!(storage, expected);
            for section in &mut storage {
                section.fill(255);
            }
        }
    }
}

#[test]
fn streamed_rows_and_fragments_overwrite_caller_storage_exactly() {
    for (mut operand, mut record, codes, scales) in seeded_cases() {
        operand.section = PackedSection::Codes;
        let expected = reference(&operand, &codes, &scales)[0].clone();
        let mut read = |index: usize, offset: u64, out: &mut [u8]| {
            let source = [&codes, &scales][index];
            out.copy_from_slice(&source[offset as usize..offset as usize + out.len()]);
            Ok(())
        };
        let mut whole = vec![255; expected.len()];
        operand
            .execute_code_rows_into(
                &record,
                0..u64::from(operand.plan.rows),
                &mut whole,
                &mut read,
            )
            .unwrap();
        assert_eq!(whole, expected);
        let mut wrong = vec![255; expected.len() + 1];
        assert!(
            operand
                .execute_code_rows_into(
                    &record,
                    0..u64::from(operand.plan.rows),
                    &mut wrong,
                    &mut read
                )
                .is_err()
        );
        assert!(wrong.iter().all(|&b| b == 255));
        record.transform = WeightTransform::PackedOperand(Box::new(operand));
        let prepared = record.prepare(expected.len() as u64).unwrap();
        let mut scratch = vec![255; expected.len() + 17];
        for range in [0..expected.len(), 1..expected.len() - 1, 3..7] {
            let mut fragment = vec![255; range.len()];
            prepared
                .produce(
                    range.start as u64..range.end as u64,
                    &mut fragment,
                    &mut read,
                    &mut scratch,
                )
                .unwrap();
            assert_eq!(fragment, expected[range]);
        }
    }
}

/// A refused row stream publishes no bytes, also when an earlier row run
/// was already admitted: a scale no BF16 operand holds in the second run
/// (source rows 8..13), or a failing read of that run's codes, leaves caller
/// storage as it was, directly and through the prepared producer.
#[test]
fn a_refused_later_row_run_leaves_caller_storage_unchanged() {
    use kernel_core::weight_transform::WeightTransformError;
    let (mut operand, mut record, codes, mut scales) = record(DecodeOrder::ExactScaledOperand);
    operand.section = PackedSection::Codes;
    record.transform = WeightTransform::PackedOperand(Box::new(operand.clone()));
    let rows = u64::from(operand.plan.rows);
    let row_bytes = u64::from(operand.plan.k) / 2;
    let fails = |codes: &[u8], scales: &[u8], late_read: bool| {
        // 128 code bytes per source row: the second run reads from row 8.
        let mut read = |index: usize, offset: u64, out: &mut [u8]| {
            if late_read && index == 0 && offset >= 8 * 128 {
                return Err(WeightTransformError("source read failed".into()));
            }
            let source = [codes, scales][index];
            out.copy_from_slice(&source[offset as usize..offset as usize + out.len()]);
            Ok(())
        };
        // The first run alone is admitted.
        let mut first = vec![0xa5; (3 * row_bytes) as usize];
        operand
            .execute_code_rows_into(&record, 0..3, &mut first, &mut read)
            .unwrap();
        let mut whole = vec![0xa5; (rows * row_bytes) as usize];
        assert!(
            operand
                .execute_code_rows_into(&record, 0..rows, &mut whole, &mut read)
                .is_err()
        );
        assert!(whole.iter().all(|&b| b == 0xa5), "the direct stream wrote");
        let prepared = record.clone().prepare(rows * row_bytes).unwrap();
        for range in [0..rows * row_bytes, 1..rows * row_bytes - 1] {
            let mut fragment = vec![0xa5; (range.end - range.start) as usize];
            assert!(
                prepared
                    .produce(range.clone(), &mut fragment, &mut read, &mut Vec::new())
                    .is_err()
            );
            assert!(
                fragment.iter().all(|&b| b == 0xa5),
                "the prepared producer wrote {range:?}"
            );
        }
    };
    fails(&codes, &scales, true);
    // Eight scale bytes per source row.
    scales[9 * 8..10 * 8].fill(255);
    fails(&codes, &scales, false);
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

#[test]
fn independent_scales_fall_back_for_padded_coordinate_overflow() {
    for axis in ["row", "column"] {
        let (mut operand, mut record, _, _) = record(DecodeOrder::ExactScaledOperand);
        operand.plan.rows = 96;
        operand.plan.k = 96;
        let matrix = &mut operand.matrices[0];
        matrix.rows = 32;
        matrix.k = 32;
        matrix.rows_kept = vec![0..32];
        matrix.k_kept = 0..32;
        matrix.row_origin = if axis == "row" { u64::MAX - 63 } else { 0 };
        matrix.k_origin = if axis == "column" { u64::MAX - 63 } else { 0 };
        record.sources[0].byte_length = 512;
        record.sources[1].byte_length = 32;
        let error = operand
            .execute_sections(&record, &[&[0; 512], &[126; 32]])
            .unwrap_err();
        assert!(
            error.0.contains(if axis == "row" {
                "scale row overflow"
            } else {
                "scale grid overflow"
            }),
            "{error}"
        );
        assert_eq!(operand.independent_scale_source(&record), None);
        assert!(
            operand
                .independent_scales(&record, &[126; 32], vec![])
                .unwrap()
                .is_none()
        );
    }
}

#[test]
fn independent_scale_refusals_keep_the_whole_packers_context() {
    let (operand, record, codes, mut scales) = record(DecodeOrder::ExactScaledOperand);
    scales[0] = 255;
    let expected = operand
        .execute_sections(&record, &[&codes, &scales])
        .unwrap_err();
    assert_eq!(
        expected.0,
        "packed operand packed operand local scale: UE8M0 scale 0xff is not finite"
    );
    assert_eq!(
        operand
            .independent_scales(&record, &scales, vec![])
            .unwrap_err()
            .0,
        expected.0
    );
}

#[test]
fn reused_row_major_groups_match_the_logical_oracle() {
    for (mut operand, record, codes, scales) in seeded_cases() {
        operand.plan.layout.id = "row-major-groups".into();
        let expected = reference(&operand, &codes, &scales);
        let mut storage = std::array::from_fn(|_| vec![255; 8192]);
        for _ in 0..3 {
            storage = operand
                .execute_sections_reusing(&record, &[&codes, &scales], storage)
                .unwrap();
            assert_eq!(storage, expected);
            for section in &mut storage {
                section.fill(255);
            }
        }
    }
}

#[test]
fn requested_sections_keep_oracle_bytes_without_materializing_companions() {
    for (mut operand, mut record, codes, scales) in seeded_cases() {
        for layout in ["row-major", "row-major-groups"] {
            operand.plan.layout.id = layout.into();
            let expected = reference(&operand, &codes, &scales);
            for (section, index) in [(PackedSection::Codes, 0), (PackedSection::LocalScales, 1)] {
                operand.section = section;
                record.transform = WeightTransform::PackedOperand(Box::new(operand.clone()));
                for wanted in [[false; 3], [true; 3]] {
                    let actual = operand
                        .execute_needed_sections_reusing(
                            &record,
                            &[&codes, &scales],
                            std::array::from_fn(|_| vec![255; 8192]),
                            wanted,
                        )
                        .unwrap();
                    for part in 0..3 {
                        if part == index || wanted[part] {
                            assert_eq!(actual[part], expected[part]);
                        } else {
                            assert!(actual[part].is_empty());
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn omitted_sections_do_not_omit_payload_or_slice_admission() {
    for section in [PackedSection::Codes, PackedSection::LocalScales] {
        for bad in 0..5 {
            let (mut operand, mut record, mut codes, mut scales) =
                record(DecodeOrder::ExactScaledOperand);
            operand.section = section;
            match bad {
                0 => scales[0] = 255,
                1 => {
                    operand.decode_order = DecodeOrder::ScaleThenCode;
                    scales[0] = 254;
                    codes[0] = 0x77;
                }
                2 => {
                    codes.pop();
                }
                3 => operand.matrices[0].rows_kept = vec![15..17],
                _ => operand.matrices[0].k_origin = u64::MAX - 31,
            }
            record.transform = WeightTransform::PackedOperand(Box::new(operand.clone()));
            let expected = operand
                .execute_sections(&record, &[&codes, &scales])
                .unwrap_err();
            let actual = operand
                .execute_needed_sections_reusing(
                    &record,
                    &[&codes, &scales],
                    std::array::from_fn(|_| vec![0xa5; 8192]),
                    [false; 3],
                )
                .unwrap_err();
            assert_eq!(actual, expected);
        }
    }
}
