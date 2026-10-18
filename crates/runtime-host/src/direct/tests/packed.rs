use super::*;
use kernel_core::contract::{DType, SwigluColumns, WeightFormat};
use kernel_core::precision::layout::PackingPlan;
use kernel_core::precision::{DecodeOrder, NibbleOrder, WeightLayout};
use kernel_core::weight_transform::{PackedMatrixSource, PackedOperand, PackedSection};

/// Scale planes precede code planes, as a row-major expert bank places them.
/// All ranks own a K slice of each expert, so the source rows overlap.
fn fixture(experts: usize) -> (Temp, Vec<WeightPlanV1>, Vec<Vec<u8>>, u64) {
    let (rows, k) = (256_u64, 256_u64);
    let code_bytes = rows * k / 2;
    let scale_bytes = rows * k / 32;
    let mut tensors = Vec::new();
    for expert in 0..experts {
        tensors.push((
            format!("e{expert}.codes"),
            "U8",
            vec![rows, k / 2],
            bytes(code_bytes as usize, expert as u8),
        ));
        tensors.push((
            format!("e{expert}.scales"),
            "U8",
            vec![rows, k / 32],
            (0..scale_bytes)
                .map(|i| 120 + ((i + expert as u64) % 7) as u8)
                .collect(),
        ));
    }
    let borrowed = tensors
        .iter()
        .map(|(name, dtype, shape, data)| (name.as_str(), *dtype, shape.clone(), data.clone()))
        .collect::<Vec<_>>();
    let root = checkpoint(&borrowed);
    let mut plans = Vec::new();
    let mut expected = Vec::new();
    for rank in 0..4_u64 {
        let mut records = Vec::new();
        let mut output = Vec::new();
        for section in [PackedSection::LocalScales, PackedSection::Codes] {
            for expert in 0..experts {
                let operand = PackedOperand {
                    plan: PackingPlan {
                        layout: WeightLayout {
                            id: "row-major".into(),
                            version: 1,
                        },
                        format: WeightFormat::Mxfp4,
                        experts: 1,
                        projections: 1,
                        rows: rows as u32,
                        k: (k / 4) as u32,
                        columns: SwigluColumns::ConcatenatedHalves,
                        local_scale_dtype: Some(DType::Ue8M0),
                    },
                    section,
                    decode_order: DecodeOrder::ExactScaledOperand,
                    matrices: vec![PackedMatrixSource {
                        expert: 0,
                        projection: 0,
                        format: WeightFormat::Mxfp4,
                        rows,
                        k,
                        row_origin: 0,
                        k_origin: 0,
                        nibble_order: NibbleOrder::EvenLow,
                        codes: 0,
                        local: Some(1),
                        local_dtype: Some(DType::Ue8M0),
                        global: None,
                        source_expert: 0,
                        source_experts: 1,
                        rows_kept: vec![0..rows],
                        k_kept: rank * k / 4..(rank + 1) * k / 4,
                    }],
                };
                let transform = WeightTransform::PackedOperand(Box::new(operand));
                let record = WeightRecord {
                    transform: transform.clone(),
                    sources: vec![
                        SourceTensor {
                            dtype: "U8".into(),
                            shape: vec![rows, k / 2],
                            byte_length: code_bytes,
                        },
                        SourceTensor {
                            dtype: "U8".into(),
                            shape: vec![rows, k / 32],
                            byte_length: scale_bytes,
                        },
                    ],
                    axis: None,
                    ranges: vec![],
                    stored_shape: vec![],
                };
                let WeightTransform::PackedOperand(operand) = &transform else {
                    unreachable!()
                };
                let produced = operand
                    .execute(
                        &record,
                        &[&tensors[expert * 2].3, &tensors[expert * 2 + 1].3],
                    )
                    .unwrap();
                records.push(WeightRecordV1 {
                    arena_offset: output.len() as u64,
                    byte_length: produced.len() as u64,
                    transform: transform.id(),
                    params: transform.encode_params().unwrap(),
                    sources: ["codes", "scales"]
                        .map(|suffix| RecordSourceV1 {
                            file: 0,
                            tensor: format!("e{expert}.{suffix}"),
                        })
                        .to_vec(),
                    axis: None,
                    ranges: vec![],
                    stored_shape: vec![],
                });
                output.extend(produced);
            }
        }
        plans.push(WeightPlanV1 {
            ranks: 4,
            rank: rank as u32,
            image_role: "r".into(),
            alignment: 64,
            arena_bytes: output.len() as u64,
            views: vec![],
            components: vec![],
            records,
        });
        expected.push(output);
    }
    (
        root,
        plans,
        expected,
        (code_bytes + scale_bytes) * experts as u64,
    )
}

/// Match CUDA's joined waves, including a read that crosses a plane boundary.
fn waves(image: &PlannedImageV1, chunk: usize) -> Vec<Vec<u8>> {
    let mut outputs = (0..4)
        .map(|r| vec![0xee; image.arena_bytes(r) as usize])
        .collect::<Vec<_>>();
    for start in (0..outputs[0].len()).step_by(chunk) {
        std::thread::scope(|scope| {
            for (rank, output) in outputs.iter_mut().enumerate() {
                let end = (start + chunk).min(output.len());
                scope.spawn(move || {
                    image
                        .read(
                            image.inner.offsets[rank] + start as u64,
                            &mut output[start..end],
                        )
                        .unwrap()
                });
            }
        });
    }
    outputs
}

#[test]
fn scale_first_expert_banks_match_the_whole_packer_at_small_and_large_budgets() {
    let (root, plans, expected, unique) = fixture(32);
    let checkpoint = Arc::new(DirectCheckpointV1::open(&root.0).unwrap());
    for budget in [128 << 10, 2 << 20] {
        let image = sharing_image(&checkpoint, &plans, 2, budget);
        assert_eq!(waves(&image, 8192), expected);
        let stats = image.stats();
        eprintln!("packed bank: budget={budget} unique={unique} stats={stats:?}");
        assert_eq!(stats.unique_source_bytes, unique);
        assert!(stats.retained_peak_bytes <= budget);
        image.release_buffers();
    }
}

#[test]
fn packed_sources_are_shared_across_ranks_without_retaining_code_banks() {
    let (root, plans, expected, unique) = fixture(32);
    let checkpoint = Arc::new(DirectCheckpointV1::open(&root.0).unwrap());
    let image = sharing_image(&checkpoint, &plans, 2, 128 << 10);
    assert_eq!(waves(&image, 8192), expected);
    let stats = image.stats();
    assert!(
        stats.source_bytes <= unique * 2,
        "reread amplification: {stats:?}"
    );
    assert!(stats.source_cache_hits > 0);
}

#[test]
fn packed_replays_and_skewed_reads_keep_bytes_and_release_retained_inputs() {
    let (root, plans, expected, _) = fixture(4);
    let checkpoint = Arc::new(DirectCheckpointV1::open(&root.0).unwrap());
    for budget in [0, 4096, 128 << 10] {
        let image = sharing_image(&checkpoint, &plans, 2, budget);
        for chunk in [2048, 4093, 16387] {
            assert_eq!(
                waves(&image, chunk),
                expected,
                "budget {budget} chunk {chunk}"
            );
        }
        // Sequential fallback and overlapping replays do not imply that a
        // source will be consumed exactly once or in rank order.
        for rank in [3, 0, 2, 1, 3] {
            for range in [3..111, 0..1025, 1023..4000, 0..expected[rank].len()] {
                let mut output = vec![0xee; range.len()];
                image
                    .read(image.inner.offsets[rank] + range.start as u64, &mut output)
                    .unwrap();
                assert_eq!(output, expected[rank][range]);
            }
        }
        assert!(image.stats().retained_peak_bytes <= budget);
        assert!(image.inner.retained_held.peak.load(Ordering::Acquire) <= budget);
        assert!(image.stats().pooled_peak_bytes <= budget.min(64 << 20));
        image.release_buffers();
        assert_eq!(image.inner.retained.load(Ordering::Acquire), 0);
        assert_eq!(image.inner.retained_held.now.load(Ordering::Acquire), 0);
        assert_eq!(waves(&image, 8192), expected);
    }
}

#[test]
fn packed_scale_admission_checks_unselected_source_rows_and_cleans_up_on_error() {
    let (root, mut plans, _, _) = fixture(1);
    let checkpoint = Arc::new(DirectCheckpointV1::open(&root.0).unwrap());
    // All ranks select K >= 32; source scale 0 is outside every selection.
    for plan in &mut plans {
        for record in &mut plan.records {
            let WeightTransform::PackedOperand(mut operand) =
                WeightTransform::decode(record.transform, &record.params).unwrap()
            else {
                unreachable!()
            };
            operand.matrices[0].k_kept = 64..128;
            record.params = WeightTransform::PackedOperand(operand)
                .encode_params()
                .unwrap();
        }
    }
    let shard = &checkpoint.shards[0];
    let (_, at) = shard.tensors["e0.scales"].clone();
    std::fs::OpenOptions::new()
        .write(true)
        .open(root.0.join("model.safetensors"))
        .unwrap()
        .write_all_at(&[255], at)
        .unwrap();
    let image = sharing_image(&checkpoint, &plans, 2, 128 << 10);
    let mut scales = [0_u8; 512];
    for _ in 0..2 {
        assert!(image.read(0, &mut scales).unwrap_err().contains("finite"));
        image.release_buffers();
        assert_eq!(image.inner.retained.load(Ordering::Acquire), 0);
    }
}
