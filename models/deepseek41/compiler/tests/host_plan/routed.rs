//! Decode planned records with the chained loader's row addresses, rather
//! than unpacking the layout the record itself advertises.
use super::*;
use kernel_core::codec::SmallFloat;
use kernel_core::moe_chained::ChainedK;
use kernel_core::moe_chained_oracle::{fold_f32_chains, partial_chains};

fn decoded(codes: &[u8], scales: &[u8], width: usize, row: usize, k: usize) -> f64 {
    let pair = codes[row * width / 2 + k / 2];
    let code = (pair >> (4 * (k % 2))) & 15;
    SmallFloat::E2M1.decode_f64(code)
        * 2.0_f64.powi(i32::from(scales[row * width / 32 + k / 32]) - 127)
}

/// Keep the real model record's layout and decode admission while probing
/// additional stored widths. Four source shards exercise nonzero K origins.
fn check_record_widths(record: &runtime_bundle_abi::weight_plan::WeightRecordV1) {
    use kernel_core::weight_transform::{SourceTensor, WeightRecord, WeightTransform};
    let WeightTransform::PackedOperand(original) =
        WeightTransform::decode(record.transform, &record.params).unwrap()
    else {
        panic!("experts must retain packed admission")
    };
    for k in [384, 512, 576] {
        for rank in 0..4 {
            let rows = 128;
            let source_k = 4 * k;
            let codes = (0..rows * source_k / 2)
                .map(|i| (1 + i % 7) as u8 | ((9 + (i / 3) % 7) as u8) << 4)
                .collect::<Vec<_>>();
            let scales = (0..rows * source_k / 32)
                .map(|i| 120 + ((i / (source_k / 32) * 3 + i % (source_k / 32)) % 7) as u8)
                .collect::<Vec<_>>();
            let mut operand = original.clone();
            operand.plan.rows = rows as u32;
            operand.plan.k = k as u32;
            let [matrix] = operand.matrices.as_mut_slice() else {
                panic!("one matrix per expert piece")
            };
            matrix.rows = rows as u64;
            matrix.k = source_k as u64;
            matrix.rows_kept = std::iter::once(0..rows as u64).collect();
            matrix.k_kept = (rank * k) as u64..((rank + 1) * k) as u64;
            let record = WeightRecord {
                transform: WeightTransform::PackedOperand(operand.clone()),
                sources: vec![
                    SourceTensor {
                        dtype: "U8".into(),
                        shape: vec![rows as u64, source_k as u64 / 2],
                        byte_length: codes.len() as u64,
                    },
                    SourceTensor {
                        dtype: "F8_E8M0".into(),
                        shape: vec![rows as u64, source_k as u64 / 32],
                        byte_length: scales.len() as u64,
                    },
                ],
                axis: None,
                ranges: vec![],
                stored_shape: vec![],
            };
            let packed = operand
                .execute_sections(&record, &[&codes, &scales])
                .unwrap();
            let mut tiled = operand.clone();
            tiled.plan.layout.id = "moe-packed-Absent".into();
            let wrong = tiled.execute_sections(&record, &[&codes, &scales]).unwrap();
            assert_ne!(
                packed[1], wrong[1],
                "K{k} must distinguish tiled scales even without padding"
            );
            let mut rejected_chains = 0;
            for row in 0..rows {
                let x = (0..k)
                    .map(|col| [1.0, -0.5, 2.0, 0.25][(rank + col) % 4])
                    .collect::<Vec<_>>();
                let truth = (0..k)
                    .map(|col| decoded(&codes, &scales, source_k, row, rank * k + col))
                    .collect::<Vec<_>>();
                let loaded = (0..k)
                    .map(|col| decoded(&packed[0], &packed[1], k, row, col))
                    .collect::<Vec<_>>();
                assert_eq!(loaded, truth, "K{k} rank{rank} row{row}");
                let oracle = partial_chains(&x, &truth, ChainedK::new(k as u32).unwrap()).unwrap();
                let chains = loaded
                    .chunks(128)
                    .zip(x.chunks(128))
                    .map(|(w, x)| {
                        w.iter()
                            .zip(x)
                            .fold(0.0f32, |s, (&w, &x)| s + (w * x) as f32)
                    })
                    .collect::<Vec<_>>();
                assert_eq!(
                    chains,
                    oracle.values.iter().map(|&x| x as f32).collect::<Vec<_>>()
                );
                let wrong_weight = (0..k)
                    .map(|col| decoded(&wrong[0], &wrong[1], k, row, col))
                    .collect::<Vec<_>>();
                let wrong_chains =
                    partial_chains(&x, &wrong_weight, ChainedK::new(k as u32).unwrap()).unwrap();
                rejected_chains += wrong_chains
                    .values
                    .iter()
                    .zip(&oracle.values)
                    .filter(|(wrong, expected)| wrong != expected)
                    .count();
            }
            assert!(
                rejected_chains > 0,
                "K{k} rank{rank}: tiled-scale control must change a projection chain"
            );
            let mut invalid = scales.clone();
            invalid[rank * k / 32] = 255;
            assert!(
                operand
                    .execute_sections(&record, &[&codes, &invalid])
                    .is_err(),
                "finite scale admission remains active"
            );
            if k == 384 && rank == 0 {
                let uniform = vec![127; scales.len()];
                let wrong = tiled
                    .execute_sections(&record, &[&codes, &uniform])
                    .unwrap();
                let missing = (0..64)
                    .filter_map(|row| {
                        let groups = (0..3)
                            .filter(|&group| {
                                wrong[1][row * 12 + group * 4..row * 12 + group * 4 + 4]
                                    .iter()
                                    .all(|&s| s == 0)
                            })
                            .collect::<Vec<_>>();
                        (!groups.is_empty()).then_some((row, groups))
                    })
                    .collect::<std::collections::BTreeMap<_, _>>();
                let mut expected = std::collections::BTreeMap::new();
                for (rows, groups) in [
                    (&[12, 15, 20, 36, 39, 55, 60, 63][..], &[0, 1, 2][..]),
                    (&[13, 21, 37, 61], &[0]),
                    (&[14, 38, 54, 62], &[2]),
                    (&[17, 33, 41, 57], &[1, 2]),
                    (&[18, 34, 42, 58], &[0, 1]),
                ] {
                    for &row in rows {
                        expected.insert(row, groups.to_vec());
                    }
                }
                assert_eq!(
                    missing, expected,
                    "tile padding reproduces the missing chains exactly"
                );
            }
        }
        eprintln!(
            "K{k}: packed record CPU roundtrip, F64 chains, tiled-scale control and admission passed at four K origins"
        );
    }
}

fn check_down(root: &Path, geometry: &model_deepseek41_contract::geometry::Geometry, ranks: u32) {
    let (header, data) = header(root);
    let (image, plans) = prepare(root, ranks).unwrap();
    let n = geometry.shape.hidden as usize;
    let source_k = geometry.shape.moe.intermediate as usize;
    let k = source_k / ranks as usize;
    assert!(matches!(k, 384 | 512 | 576));
    let mut base = 0;
    for (rank, plan) in plans.iter().enumerate() {
        let index =
            bundle_runtime::index::parse(&plan.components[0].bytes, rank as u32, INDEX_SCHEMA)
                .unwrap();
        if rank == 0 {
            let record = plan
                .records
                .iter()
                .find(|r| {
                    r.sources
                        .first()
                        .is_some_and(|s| s.tensor == "layers.0.ffn.experts.0.w2.weight")
                })
                .unwrap();
            check_record_widths(record);
        }
        let codes = &index.views["layers.0.ffn.experts.down.weight"];
        let scales = &index.views["layers.0.ffn.experts.down.scale"];
        for expert in [0, 1, geometry.shape.moe.experts - 1] {
            let stem = format!("layers.0.ffn.experts.{expert}.w2");
            let source_codes = source(root, &header, data, &format!("{stem}.weight"));
            let source_scales = source(root, &header, data, &format!("{stem}.scale"));
            let mut actual_codes = vec![0; n * k / 2];
            let mut actual_scales = vec![0; n * k / 32];
            image
                .read(
                    base + codes.offset + u64::from(expert) * actual_codes.len() as u64,
                    &mut actual_codes,
                )
                .unwrap();
            image
                .read(
                    base + scales.offset + u64::from(expert) * actual_scales.len() as u64,
                    &mut actual_scales,
                )
                .unwrap();
            let expected_codes = block(
                &source_codes.bytes,
                source_k as u64 / 2,
                0..n as u64,
                (rank * k / 2) as u64..((rank + 1) * k / 2) as u64,
            );
            let expected_scales = block(
                &source_scales.bytes,
                source_k as u64 / 32,
                0..n as u64,
                (rank * k / 32) as u64..((rank + 1) * k / 32) as u64,
            );
            same(
                &actual_codes,
                &expected_codes,
                &format!("K{k} rank{rank} expert{expert} chained codes"),
            );
            same(
                &actual_scales,
                &expected_scales,
                &format!("K{k} rank{rank} expert{expert} chained scales"),
            );
            for row in 0..n {
                for col in 0..k {
                    assert_eq!(
                        decoded(&actual_codes, &actual_scales, k, row, col).to_bits(),
                        decoded(
                            &source_codes.bytes,
                            &source_scales.bytes,
                            source_k,
                            row,
                            rank * k + col
                        )
                        .to_bits()
                    );
                }
            }
            // Probe two complete 64-column tiles with each K group alone.
            // This isolates a lost chain even if the full dot cancels it.
            for group in 0..k.div_ceil(128) {
                let input = (0..k)
                    .map(|i| {
                        if i / 128 == group {
                            [1.0, -0.5, 2.0, 0.25][(i + rank) % 4]
                        } else {
                            0.0
                        }
                    })
                    .collect::<Vec<_>>();
                for row in 0..128 {
                    let truth = (0..k)
                        .map(|col| {
                            decoded(
                                &source_codes.bytes,
                                &source_scales.bytes,
                                source_k,
                                row,
                                rank * k + col,
                            )
                        })
                        .collect::<Vec<_>>();
                    let loaded = (0..k)
                        .map(|col| decoded(&actual_codes, &actual_scales, k, row, col))
                        .collect::<Vec<_>>();
                    let oracle =
                        partial_chains(&input, &truth, ChainedK::new(k as u32).unwrap()).unwrap();
                    let chains = loaded
                        .chunks(128)
                        .zip(input.chunks(128))
                        .map(|(w, x)| {
                            w.iter()
                                .zip(x)
                                .fold(0.0f32, |s, (&w, &x)| s + (w * x) as f32)
                        })
                        .collect::<Vec<_>>();
                    let expected = fold_f32_chains(
                        &oracle.values.iter().map(|&x| x as f32).collect::<Vec<_>>(),
                    )
                    .unwrap();
                    assert_eq!(
                        fold_f32_chains(&chains).unwrap().to_bits(),
                        expected.to_bits(),
                        "K{k} rank{rank} expert{expert} row{row} group{group}"
                    );
                }
            }
        }
        assert_eq!(codes.layout_id, "mxfp4_e2m1_rows_v1");
        assert_eq!(scales.layout_id, "mxfp4_ue8m0_scale_rows_v1");
        assert_eq!(
            scales.bytes,
            u64::from(geometry.shape.moe.experts) * (n * k / 32) as u64
        );
        eprintln!(
            "{} TP{ranks} rank{rank}: K{k} three experts, every decoded weight and per-chain F64 projections passed",
            geometry.id
        );
        base += plan.arena_bytes;
    }
}

fn fixture(geometry: &model_deepseek41_contract::geometry::Geometry, operands: &[&str]) -> PathBuf {
    let root = sparse(geometry);
    let (header, data) = header(&root);
    let file = std::fs::OpenOptions::new()
        .write(true)
        .open(root.join(synthetic::FILE))
        .unwrap();
    for expert in [0, 1, geometry.shape.moe.experts - 1] {
        for (operand_index, operand) in operands.iter().enumerate() {
            for (part, scale) in [("weight", false), ("scale", true)] {
                let name = format!("layers.0.ffn.experts.{expert}.{operand}.{part}");
                let entry = &header[&name];
                let start = entry["data_offsets"][0].as_u64().unwrap();
                let len = entry["data_offsets"][1].as_u64().unwrap() - start;
                let width = entry["shape"][1].as_u64().unwrap() as usize;
                let bytes = (0..len as usize)
                    .map(|i| {
                        let row = i / width;
                        let col = i % width;
                        if scale {
                            120 + ((row * 3 + col + (expert as usize + operand_index * 3)) % 7)
                                as u8
                        } else {
                            let a = 1
                                + ((row + col * 2 + (expert as usize + operand_index * 3)) % 7)
                                    as u8;
                            let b = 9
                                + ((row * 3 + col + (expert as usize + operand_index * 3)) % 7)
                                    as u8;
                            a | b << 4
                        }
                    })
                    .collect::<Vec<_>>();
                file.write_all_at(&bytes, data + start).unwrap();
            }
        }
    }
    root
}

#[test]
fn routed_down_records_round_trip_through_chained_addresses() {
    for (geometry, ranks) in GEOMETRIES.iter().filter_map(|g| match g.id {
        "synthetic_small" => Some((g, 2)),
        "flash" => Some((g, 4)),
        _ => None,
    }) {
        let root = fixture(geometry, &["w2"]);
        check_down(&root, geometry, ranks);
        std::fs::remove_dir_all(root).unwrap();
    }
}

#[test]
#[ignore = "requires the external synthetic-small checkpoint"]
fn checkpoint_routed_down_records_round_trip_through_chained_addresses() {
    let root = PathBuf::from(
        std::env::var_os("CHOPINFER_DEEPSEEK_CHECKPOINT").expect("checkpoint required"),
    );
    let geometry = GEOMETRIES
        .iter()
        .find(|g| g.id == "synthetic_small")
        .expect("compile synthetic-small");
    check_down(&root, geometry, 2);
}

/// Flash is always compiled, so this checks the fused gate/up bank even
/// when no optional geometry enables the full checkpoint-view test.
#[test]
fn flash_gate_up_records_round_trip_through_chained_addresses() {
    let geometry = &model_deepseek41_contract::geometry::FLASH;
    let root = fixture(geometry, &["w1", "w3"]);
    let (header, data) = header(&root);
    let ranks = 4;
    let (image, plans) = prepare(&root, ranks).unwrap();
    let n = geometry.shape.moe.intermediate as usize / ranks as usize;
    let k = geometry.shape.hidden as usize;
    let mut base = 0;
    for (rank, plan) in plans.iter().enumerate() {
        let index =
            bundle_runtime::index::parse(&plan.components[0].bytes, rank as u32, INDEX_SCHEMA)
                .unwrap();
        let codes = &index.views["layers.0.ffn.experts.gate_up.weight"];
        let scales = &index.views["layers.0.ffn.experts.gate_up.scale"];
        for expert in [0, 1, geometry.shape.moe.experts - 1] {
            // The loader's global row is expert * (2 * n) + half * n + row.
            // Read both halves together, so the comparison also checks the
            // expert stride and w1/w3 order, independently of record offsets.
            let mut actual_codes = vec![0; 2 * n * k / 2];
            let mut actual_scales = vec![0; 2 * n * k / 32];
            image
                .read(
                    base + codes.offset + u64::from(expert) * actual_codes.len() as u64,
                    &mut actual_codes,
                )
                .unwrap();
            image
                .read(
                    base + scales.offset + u64::from(expert) * actual_scales.len() as u64,
                    &mut actual_scales,
                )
                .unwrap();
            for (half, operand) in ["w1", "w3"].into_iter().enumerate() {
                let stem = format!("layers.0.ffn.experts.{expert}.{operand}");
                let source_codes = source(&root, &header, data, &format!("{stem}.weight"));
                let source_scales = source(&root, &header, data, &format!("{stem}.scale"));
                let rows = (rank * n) as u64..((rank + 1) * n) as u64;
                let expected_codes = block(
                    &source_codes.bytes,
                    k as u64 / 2,
                    rows.clone(),
                    0..k as u64 / 2,
                );
                let expected_scales =
                    block(&source_scales.bytes, k as u64 / 32, rows, 0..k as u64 / 32);
                let at = format!("Flash TP{ranks} rank{rank} expert{expert} {operand} chained");
                same(
                    &actual_codes[half * n * k / 2..(half + 1) * n * k / 2],
                    &expected_codes,
                    &format!("{at} codes"),
                );
                same(
                    &actual_scales[half * n * k / 32..(half + 1) * n * k / 32],
                    &expected_scales,
                    &format!("{at} scales"),
                );
                for row in 0..n {
                    for col in 0..k {
                        assert_eq!(
                            decoded(&actual_codes, &actual_scales, k, half * n + row, col)
                                .to_bits(),
                            decoded(
                                &source_codes.bytes,
                                &source_scales.bytes,
                                k,
                                rank * n + row,
                                col
                            )
                            .to_bits(),
                            "{at} row{row} K{col}"
                        );
                    }
                }
            }
        }
        assert_eq!(codes.layout_id, "mxfp4_e2m1_rows_v1");
        assert_eq!(scales.layout_id, "mxfp4_ue8m0_scale_rows_v1");
        assert_eq!(
            codes.bytes,
            u64::from(geometry.shape.moe.experts) * (2 * n * k / 2) as u64
        );
        assert_eq!(
            scales.bytes,
            u64::from(geometry.shape.moe.experts) * (2 * n * k / 32) as u64
        );
        eprintln!(
            "Flash TP{ranks} rank{rank}: w1/w3 row-addressed records and every decoded weight passed for three experts"
        );
        base += plan.arena_bytes;
    }
    std::fs::remove_dir_all(root).unwrap();
}
