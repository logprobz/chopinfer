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
                    local_encoding: None,
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
                let (source, row_bytes, width) = match section {
                    PackedSection::Codes => (&tensors[expert * 2].3, k / 2, k / 8),
                    PackedSection::LocalScales => (&tensors[expert * 2 + 1].3, k / 32, k / 128),
                    PackedSection::GlobalScales => unreachable!(),
                };
                // Select stored row bytes directly, without invoking a packer.
                let produced: Vec<u8> = (0..rows)
                    .flat_map(|row| {
                        let start = (row * row_bytes + rank * width) as usize;
                        source[start..start + width as usize].iter().copied()
                    })
                    .collect();
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
fn packed_rank_plans_and_images_keep_pre_change_digests() {
    // SHA-256 from the original packer at b450a4ea1, including every rank's
    // encoded plan and full arena. These do not depend on the new packer.
    let plans_before = [
        "1429608f4ce4cade372438ac4e45e9094c4d26abfff14719cdbbec92828e6415",
        "014d935f75eab42b94aa9a5e290e52d0badab0d2c976a0cb7efeafa30f5b896e",
        "121a25a60ec7e9c90e54c29ca278ca395d148bd1d0e33b2d73278422dc80464b",
        "2048124912f22ce1641f8e4e47e442ddbfb90acecc64176645de21ab7495c0ad",
    ];
    let arenas_before = [
        "7c14d5343ecad69f2c78a84a8e4180760246a26c61e4f6e315f903a67803208e",
        "648261333bb9c4620789c52a11ce4578620da815142292281ce30c8b15c96934",
        "92f327eddcab82cb57aac3f9fb572722c99c614a134ff3a779a959d1e2fa908d",
        "b08a014acab30da0c793a111992708d4b93addbc18ea39bef107e8568806cb75",
    ];
    let (root, plans, expected, _) = fixture(32);
    let checkpoint = Arc::new(DirectCheckpointV1::open(&root.0).unwrap());
    let image = sharing_image(&checkpoint, &plans, 2, 128 << 10);
    let actual = waves(&image, 8192);
    for rank in 0..4 {
        assert_eq!(
            runtime_weights::sha256_bytes(&plans[rank].encode()),
            plans_before[rank]
        );
        assert_eq!(
            runtime_weights::sha256_bytes(&expected[rank]),
            arenas_before[rank]
        );
        assert_eq!(
            runtime_weights::sha256_bytes(&actual[rank]),
            arenas_before[rank]
        );
    }
}

#[test]
fn optional_io_diagnostics_preserve_bytes_and_count_real_reads() {
    let (root, plans, expected, _) = fixture(4);
    let checkpoint = Arc::new(DirectCheckpointV1::open(&root.0).unwrap());
    for detailed in [false, true] {
        let mut image = sharing_image(&checkpoint, &plans, 2, 128 << 10);
        let inner = Arc::get_mut(&mut image.inner).unwrap();
        inner.detailed_timing = detailed;
        inner.io_timing = detailed.then(io_stats::IoCounters::default);
        let before = image.stats();
        assert_eq!(waves(&image, 8192), expected);
        let stats = image.stats();
        let delta = stats.since(before);
        assert_eq!(delta.io_timing.is_some(), detailed);
        if let Some(io) = stats.io_timing {
            assert!(io.buffered_calls > 0);
            assert_eq!(io.direct_calls, 0);
            assert_eq!(io.read_errors, 0);
            assert_eq!(io.read_bytes, stats.file_read_bytes);
            assert_eq!(io.request_sizes.iter().sum::<u64>(), stats.source_reads);
            assert!(io.peak_reads > 0);
            assert_eq!(delta.io_timing.unwrap().peak_reads, 0);
        }
    }
}

#[test]
fn unused_companions_include_owed_sections_that_cannot_be_retained() {
    let (root, plans, expected, _) = fixture(1);
    let checkpoint = Arc::new(DirectCheckpointV1::open(&root.0).unwrap());
    let mut image = sharing_image(&checkpoint, &plans, 1, 128 << 10);
    let inner = Arc::get_mut(&mut image.inner).unwrap();
    inner.options.shared_bytes = 0;
    inner.io_timing = Some(io_stats::IoCounters::default());
    // Exercise whole production while a companion is still owed. With no
    // shared budget, that companion is discarded despite having consumers.
    let record = &mut inner.ranks[0].records[0];
    record.has_code_section = false;
    let shared = record.shared.unwrap();
    let WeightTransform::PackedOperand(operand) = &record.prepared.record().transform else {
        panic!("packed fixture");
    };
    let sizes = operand.plan.section_bytes().unwrap();
    inner.shared[shared].state.get_mut().unwrap().remaining[0] = sizes.codes;
    let mut out = vec![0; sizes.local_scales as usize];
    image.read(0, &mut out).unwrap();
    assert_eq!(out, expected[0][..out.len()]);
    assert_eq!(
        image.stats().io_timing.unwrap().unused_section_bytes,
        sizes.codes
    );
}

#[test]
fn consumed_scale_sections_are_not_repacked_by_whole_code_production() {
    let (root, mut plans, expected, _) = fixture(1);
    for plan in &mut plans {
        for record in &mut plan.records {
            let WeightTransform::PackedOperand(mut operand) =
                WeightTransform::decode(record.transform, &record.params).unwrap()
            else {
                unreachable!()
            };
            operand.plan.layout.id = "row-major-groups".into();
            record.params = WeightTransform::PackedOperand(operand)
                .encode_params()
                .unwrap();
        }
    }
    let checkpoint = Arc::new(DirectCheckpointV1::open(&root.0).unwrap());
    let mut image = sharing_image(&checkpoint, &plans, 1, 0);
    Arc::get_mut(&mut image.inner).unwrap().io_timing = Some(io_stats::IoCounters::default());
    assert_eq!(waves(&image, 512), expected);
    assert_eq!(image.stats().io_timing.unwrap().unused_section_bytes, 0);
    image.release_buffers();
    assert_eq!(image.inner.retained.load(Ordering::Acquire), 0);
}

#[test]
fn independent_scale_fallback_reuses_its_admitted_input_without_a_shared_budget() {
    let (root, mut plans, _, _) = fixture(1);
    for plan in &mut plans {
        for record in &mut plan.records {
            let WeightTransform::PackedOperand(mut operand) =
                WeightTransform::decode(record.transform, &record.params).unwrap()
            else {
                unreachable!()
            };
            operand.decode_order = DecodeOrder::ExactScaledOperand;
            record.params = WeightTransform::PackedOperand(operand)
                .encode_params()
                .unwrap();
        }
    }
    let checkpoint = Arc::new(DirectCheckpointV1::open(&root.0).unwrap());
    let shard = &checkpoint.shards[0];
    let file = std::fs::OpenOptions::new()
        .write(true)
        .open(root.0.join("model.safetensors"))
        .unwrap();
    let (codes, at) = &shard.tensors["e0.codes"];
    file.write_all_at(&vec![0; codes.byte_length as usize], *at)
        .unwrap();
    file.write_all_at(&[254], shard.tensors["e0.scales"].1)
        .unwrap();
    let image = sharing_image(&checkpoint, &plans, 1, 0);
    let mut scales = [0xa5; 512];
    image.read(0, &mut scales).unwrap();
    assert_eq!(scales[0], 254);
    assert_eq!(image.stats().source_reads, 2);
    assert_eq!(image.stats().retained_peak_bytes, 0);
}

#[test]
fn owned_packed_sources_read_directly_and_charge_alignment_storage() {
    let (root, mut plans, expected, _) = fixture(4);
    for plan in &mut plans {
        for record in &mut plan.records {
            let WeightTransform::PackedOperand(mut operand) =
                WeightTransform::decode(record.transform, &record.params).unwrap()
            else {
                unreachable!()
            };
            operand.plan.layout.id = "row-major-groups".into();
            record.params = WeightTransform::PackedOperand(operand)
                .encode_params()
                .unwrap();
        }
    }
    let checkpoint = Arc::new(DirectCheckpointV1::open(&root.0).unwrap());
    if !direct_reads_supported(&checkpoint) {
        eprintln!("skipped: no residency report or O_DIRECT on this file system");
        return;
    }
    for budget in [0, 4096, 128 << 10] {
        evict(&checkpoint);
        let mut image = sharing_image(&checkpoint, &plans, 2, budget);
        let inner = Arc::get_mut(&mut image.inner).unwrap();
        inner.options.direct_reads = true;
        inner.io_timing = Some(io_stats::IoCounters::default());
        assert_eq!(waves(&image, 8192), expected);
        let stats = image.stats();
        assert!(stats.direct_bytes > 0);
        assert_eq!(stats.io_timing.unwrap().bounce_bytes, 0);
        assert!(stats.retained_peak_bytes <= budget);
        assert!(image.inner.retained_held.peak.load(Ordering::Acquire) <= budget);
        image.release_buffers();
        assert_eq!(image.inner.retained.load(Ordering::Acquire), 0);
        assert_eq!(image.inner.retained_held.now.load(Ordering::Acquire), 0);
    }
}

#[test]
fn scale_first_expert_banks_match_baseline_bytes_at_small_and_large_budgets() {
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
fn scale_inputs_are_read_once_across_whole_section_cohorts_when_they_fit() {
    let (root, mut plans, expected, unique) = fixture(32);
    for plan in &mut plans {
        for record in &mut plan.records {
            let WeightTransform::PackedOperand(mut operand) =
                WeightTransform::decode(record.transform, &record.params).unwrap()
            else {
                unreachable!()
            };
            operand.plan.layout.id = "row-major-groups".into();
            record.params = WeightTransform::PackedOperand(operand)
                .encode_params()
                .unwrap();
        }
    }
    let checkpoint = Arc::new(DirectCheckpointV1::open(&root.0).unwrap());
    for budget in [4096, 2 << 20] {
        let image = sharing_image(&checkpoint, &plans, 2, budget);
        assert_eq!(waves(&image, 8192), expected);
        if budget == 2 << 20 {
            assert_eq!(image.stats().source_bytes, unique);
        }
        assert!(image.stats().retained_peak_bytes <= budget);
        image.release_buffers();
        assert_eq!(image.inner.retained.load(Ordering::Acquire), 0);
        assert_eq!(image.inner.retained_held.now.load(Ordering::Acquire), 0);
        // Evicted or released cohorts must tolerate another partial replay.
        let mut replay = vec![0xa5; 1025];
        image
            .read(image.inner.offsets[3] + 17, &mut replay)
            .unwrap();
        assert_eq!(replay, expected[3][17..17 + 1025]);
        image.release_buffers();
        assert_eq!(image.inner.retained.load(Ordering::Acquire), 0);
    }
}

#[test]
fn ordinary_reads_do_not_allocate_or_lock_a_range_union() {
    let (root, plans, expected, _) = fixture(2);
    let checkpoint = Arc::new(DirectCheckpointV1::open(&root.0).unwrap());
    let image = PlannedImageV1::new_with_range_tracking(
        checkpoint,
        &plans,
        None,
        implementation_v1([0; 32]),
        DirectLoadOptionsV1 {
            threads_per_rank: 2,
            read_ahead: false,
            direct_reads: false,
            shared_bytes: 128 << 10,
        },
        false,
    )
    .unwrap();
    assert_eq!(waves(&image, 8192), expected);
    assert!(image.inner.source_ranges.is_none());
    assert!(!image.stats().source_ranges_tracked);
    assert_eq!(image.stats().unique_source_bytes, 0);
    assert!(image.stats().source_bytes > 0);
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

#[test]
fn scale_only_plans_refuse_unreadable_code_payloads() {
    let (root, mut plans, _, _) = fixture(1);
    for plan in &mut plans {
        plan.records.truncate(1);
        plan.arena_bytes = plan.records[0].byte_length;
    }
    let checkpoint = Arc::new(DirectCheckpointV1::open(&root.0).unwrap());
    let (tensor, begin) = &checkpoint.shards[0].tensors["e0.codes"];
    checkpoint
        .read_refusal
        .set(*begin..*begin + tensor.byte_length)
        .unwrap();
    let image = sharing_image(&checkpoint, &plans, 2, 128 << 10);
    let error = image.read(0, &mut [0; 512]).unwrap_err();
    assert!(error.contains("Input/output error"), "{error}");
    checkpoint.verify_unchanged().unwrap();
    image.release_buffers();
    assert_eq!(image.inner.retained.load(Ordering::Acquire), 0);
}

#[test]
fn failed_single_rank_upload_releases_packed_inputs_before_image_teardown() {
    failed_upload_releases_retained_bytes(false);
}

#[test]
fn failed_single_rank_upload_releases_replicated_outputs_before_image_teardown() {
    failed_upload_releases_retained_bytes(true);
}

fn failed_upload_releases_retained_bytes(replicated: bool) {
    use crate::fake::{FakeHostDeviceV1, FakeRefusalV1};
    use crate::{HostInstanceConfigV1, HostServicesV1Backend, HostTopologyV1};
    use runtime_bundle_abi::HostServicesTableV1;
    use std::rc::Rc;
    let (root, mut plans, expected, _) = fixture(2);
    if replicated {
        let first = plans[0].clone();
        for plan in &mut plans {
            *plan = WeightPlanV1 {
                rank: plan.rank,
                ..first.clone()
            };
        }
    }
    let checkpoint = Arc::new(DirectCheckpointV1::open(&root.0).unwrap());
    let image = Arc::new(sharing_image(&checkpoint, &plans, 2, 128 << 10));
    let config = HostInstanceConfigV1::new(HostTopologyV1::new(vec![0, 1, 2, 3]).unwrap())
        .with_planned_compiled_image(Arc::clone(&image))
        .unwrap();
    let backend = Rc::new(HostServicesV1Backend::new(FakeHostDeviceV1::new(), config));
    let table = HostServicesTableV1::new(Rc::clone(&backend));
    let host = table.caller();
    let context = host.select_context(0, 0).unwrap();
    let pool = host.select_memory_pool(context).unwrap();
    let stream = host.create_stream(context, 0).unwrap();
    let compiled = host.open_compiled_image(b"r").unwrap();
    backend.with_device_mut(|device| device.refuse(FakeRefusalV1::SynchronizeStream));
    assert!(
        host.upload_rank_arena(context, pool, stream, compiled, 0)
            .is_err()
    );
    assert!(image.stats().source_bytes > 0);
    assert!(image.stats().retained_peak_bytes > 0);
    assert_eq!(image.inner.retained.load(Ordering::Acquire), 0);
    assert_eq!(image.inner.retained_held.now.load(Ordering::Acquire), 0);
    // Cleanup does not consume the image. A later reader still gets the
    // same bytes, even when the first upload left section consumers pending.
    let mut retry = vec![0; expected[0].len()];
    image.read(0, &mut retry).unwrap();
    assert_eq!(retry, expected[0]);
    image.release_buffers();
    assert_eq!(image.inner.retained.load(Ordering::Acquire), 0);
    host.release_compiled_image(compiled).unwrap();
    host.release_stream(stream).unwrap();
    host.release_memory_pool(pool).unwrap();
    host.release_context(context).unwrap();
    assert_eq!(backend.live_resource_count(), 0);
}
