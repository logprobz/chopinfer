//! Full chunks use the decoded cache only at qualified offsets.
use super::*;
use bundle_runtime::fake_host::conformance::ObservedHost;

#[test]
fn decoded_prefill_cache_allocation_failures_release_prior_resources() {
    for capacity in [1, 32] {
        allocation_failures(capacity);
    }
}

fn allocation_failures(capacity: u32) {
    let shape = Shape::new(&FLASH, Ranks::TP4).unwrap();
    let record = crate::config::tests::record(&shape, KV_LAYOUTS[0], capacity, false);
    let host = FakeHost::new(shape, KV_LAYOUTS[0]);
    let mut instance = api()
        .create(HostServicesTableV1::new(Rc::clone(&host)), &record)
        .unwrap();
    let bytes = crate::Kernels::prefill_cache_layout(8319, 64, crate::Kernels::PREFILL_CACHE_PAGES)
        .unwrap()
        .bytes;
    let calls = host.device.journal.borrow();
    let first = calls
        .iter()
        .filter(|l| l.starts_with("allocate "))
        .position(|l| *l == format!("allocate {bytes} 256"))
        .unwrap();
    let cache = calls
        .iter()
        .position(|l| *l == format!("allocate {bytes} 256"))
        .unwrap();
    let kv = calls.iter().position(|l| l == "serving kv").unwrap();
    assert!(cache < kv);
    drop(calls);
    instance.destroy().unwrap();
    assert_eq!(host.acquired.get(), host.released.get());
    for failed_at in [first, first + 1] {
        let host = FakeHost::new(shape, KV_LAYOUTS[0]);
        let observed = Rc::new(ObservedHost::new(Rc::clone(&host)));
        host.device
            .fail_allocate
            .set(Some(failed_at.try_into().unwrap()));
        assert!(
            api()
                .create(HostServicesTableV1::new(Rc::clone(&observed)), &record)
                .is_err()
        );
        assert_eq!(host.device.allocation_failures.get(), 1);
        assert_eq!(host.device.fail_allocate.get(), None);
        assert_eq!(host.device.buffer_order.borrow().len(), failed_at);
        observed.accounting.assert_released();
        for id in host.device.buffer_order.borrow().iter() {
            assert!(!host.device.extents.borrow().contains_key(id));
        }
    }
}

fn high_pages(previous: u64, stride: u64, slot: u64) -> Vec<u8> {
    use runtime_bundle_sdk::{BatchAppendV1, BatchItemV1, BatchRequestV1, BatchWireVersionV1};
    let end = previous + 8192;
    BatchRequestV1::new(
        BatchWireVersionV1::V3,
        vec![BatchItemV1::new(
            7,
            false,
            8192,
            true,
            (previous..end).map(|i| i as u32).collect(),
        )],
        vec![BatchAppendV1::new(
            previous,
            8192,
            end,
            (0..end.div_ceil(64)).map(|p| 65536 + p * stride).collect(),
            None,
        )],
    )
    .with_state_slots(vec![slot])
    .encode()
    .unwrap()
}

#[test]
fn decoded_prefill_rebases_high_pages_and_falls_back_for_wide_spans() {
    for (capacity, slot, stride) in [(1, 0, 1), (1, 0, 8), (32, 31, 1), (32, 31, 8)] {
        let shape = Shape::new(&FLASH, Ranks::TP4).unwrap();
        let host = FakeHost::new(shape, KV_LAYOUTS[0]);
        host.kv_pages.set(219355);
        let record = crate::config::tests::record(&shape, KV_LAYOUTS[0], capacity, false);
        let mut instance = api()
            .create(HostServicesTableV1::new(Rc::clone(&host)), &record)
            .unwrap();
        step(
            &mut instance,
            &high_pages(0, stride, slot),
            &mut Views::default(),
        );
        let from = host.device.journal.borrow().len();
        step(
            &mut instance,
            &high_pages(8192, stride, slot),
            &mut Views::default(),
        );
        let lines = host.device.journal.borrow();
        let attention: Vec<_> = lines[from..]
            .iter()
            .filter(|l| {
                l.starts_with("launch ")
                    && l.contains("chop_latent_sources_")
                    && !l.contains("_merge_")
            })
            .collect();
        assert_eq!(attention.len(), 160);
        for line in attention {
            assert_eq!(
                line.contains("_decoded_"),
                stride == 1 || line.contains("_window_")
            );
        }
        let pages = lines[from..]
            .iter()
            .find(|l| l.starts_with("launch 2 chop_latent_prefill_decode_pages_d512_p32_v1 "));
        if stride == 1 {
            let pool = DeviceBufferSpanV1::new(
                BundleDeviceBufferHandleV1::from_parts(1, KV_SLOT),
                0,
                host.kv_pages.get() * host.page_bytes(),
            );
            let source = crate::compressor::source_binding(
                &FLASH,
                2,
                KvStorage::Native,
                pool,
                host.kv_pages.get() as u32,
                FLASH.context_tokens(),
            )
            .unwrap()
            .unwrap()
            .latent;
            let last = source.scales.unwrap();
            let tail = last.offset + last.length
                - source.values.offset
                - u64::from(source.pages - 1) * source.page_stride;
            let sliced = DeviceBufferSpanV1::new(
                pool.buffer,
                source.values.offset + 65536 * source.page_stride,
                255 * source.page_stride + tail,
            );
            assert!(
                pages
                    .unwrap()
                    .contains(&format!("[DeviceBuffer({sliced:?}),"))
            );
            let mut table = (0_i32..256).flat_map(i32::to_le_bytes).collect::<Vec<_>>();
            table.resize((FLASH.context_tokens().div_ceil(64) * 4) as usize, 255);
            assert_eq!(
                host.uploads
                    .borrow()
                    .values()
                    .filter(|bytes| **bytes == table)
                    .count(),
                4
            );
        } else {
            assert!(pages.is_none());
        }
        drop(lines);
        instance.retire_request(7).unwrap();
        instance.destroy().unwrap();
        assert_eq!(host.acquired.get(), host.released.get());
    }
}

#[test]
fn decoded_prefill_routes_winning_offsets_and_fuses_mixed_outputs() {
    let shape = Shape::new(&FLASH, Ranks::TP4).unwrap();
    let host = FakeHost::new(shape, KV_LAYOUTS[0]);
    host.kv_pages.set(1024);
    let record = crate::config::tests::record(&shape, KV_LAYOUTS[0], 1, false);
    let mut instance = api()
        .create(HostServicesTableV1::new(Rc::clone(&host)), &record)
        .unwrap();
    let mut mismatches = Vec::new();
    for offset in [0, 8192, 16384, 24576, 32768] {
        let from = host.device.journal.borrow().len();
        if let Err(error) =
            instance.execute(&prompt(7, 0, offset, 8192, true), &mut Views::default())
        {
            panic!(
                "offset {offset}: {error:?}: {:?}; {:?}",
                instance.last_execute_refusal(),
                host.logs.borrow()
            );
        }
        let lines = host.device.journal.borrow();
        assert_launches_fit_configured_shared(&lines, &shape.topology());
        let launches: Vec<_> = lines[from..]
            .iter()
            .filter(|l| l.starts_with("launch "))
            .collect();
        let attention: Vec<_> = launches
            .iter()
            .filter(|l| l.contains("chop_latent_sources_") && !l.contains("_merge_"))
            .collect();
        assert_eq!(attention.len(), 4 * 40);
        for line in &attention {
            let window = line.contains("_window_");
            let cached = offset < 32768 && (window || offset > 0);
            if line.contains("_decoded_") != cached {
                mismatches.push(format!("offset {offset}: {line}"));
            }
            let head = !window && offset < 32768;
            let suffix = if head {
                if offset == 0 {
                    "_head_shared_finish_v1"
                } else {
                    "_head_async_finish_v1"
                }
            } else {
                "_reuse2_v1"
            };
            assert!(line.contains(suffix), "{line}");
            let shared = if head { 83200 } else { 85504 };
            assert!(
                line.contains(&format!("shared_memory_bytes: {shared},")),
                "{line}"
            );
            assert!(
                line.contains(if head {
                    "grid_x: 8192, grid_y: 1, grid_z: 1,"
                } else {
                    "grid_x: 4096, grid_y: 2, grid_z: 1,"
                }),
                "{line}"
            );
        }
        let merges: Vec<_> = launches
            .iter()
            .filter(|l| l.contains("chop_latent_sources_") && l.contains("_merge_"))
            .collect();
        assert_eq!(merges.len(), if offset < 32768 { 8 } else { 160 });
        assert!(merges.iter().all(|l| !l.contains("_decoded_")));
        if let Some(dir) = std::env::var_os("CHOPINFER_DEEPSEEK41_CACHE_JOURNAL_DIR") {
            std::fs::create_dir_all(&dir).unwrap();
            std::fs::write(
                std::path::Path::new(&dir).join(format!("prefill-{offset}.journal")),
                lines.join("\n"),
            )
            .unwrap();
        }
        drop(lines);
        instance.commit().unwrap();
    }
    let from = host.device.journal.borrow().len();
    instance
        .execute(&decode(&[(7, 0, 40960, 1)]), &mut Views::default())
        .unwrap();
    assert!(
        host.device.journal.borrow()[from..]
            .iter()
            .all(|l| !l.contains("_decoded_") && !l.contains("chop_latent_prefill_"))
    );
    instance.commit().unwrap();
    instance.retire_request(7).unwrap();
    instance.destroy().unwrap();
    assert_eq!(host.acquired.get(), host.released.get());
    assert!(
        mismatches.is_empty(),
        "{} unexpected attention routes; first: {:?}",
        mismatches.len(),
        mismatches.first()
    );
}

pub(super) fn composed_trace(mode: u8, dma: bool, skip: bool, cached: bool) -> Vec<String> {
    composed_trace_offsets(mode, dma, skip, cached, &[0, 8192])
}

fn composed_trace_offsets(
    mode: u8,
    dma: bool,
    skip: bool,
    cached: bool,
    offsets: &[u64],
) -> Vec<String> {
    composed_trace_capacity(mode, dma, skip, cached, offsets, 1, 0)
}

fn composed_trace_capacity(
    mode: u8,
    dma: bool,
    skip: bool,
    cached: bool,
    offsets: &[u64],
    capacity: u32,
    slot: u64,
) -> Vec<String> {
    use crate::instance::execute::overlap::with_mode;
    use crate::ops::{with_bf16_gather, with_dma};
    use runtime_bundle_sdk::prefill::{Boundary, encode};
    let shape = Shape::new(&FLASH, Ranks::TP4).unwrap();
    let host = FakeHost::new(shape, KV_LAYOUTS[0]);
    host.kv_pages.set(1024);
    host.device.residency.set(Some(BundleRawClusterResidencyV1 {
        multiprocessor_count: 188,
        usable_multiprocessors: 188,
        max_active_clusters: 188,
        flags: BUNDLE_CLUSTER_RESIDENCY_V1_CLUSTERS_REPORTED
            | BUNDLE_CLUSTER_RESIDENCY_V1_COOPERATIVE_LAUNCH,
        ..BundleRawClusterResidencyV1::unreported()
    }));
    host.device
        .ordinary_symbols
        .borrow_mut()
        .insert("chop_comm_owned_sum_round_f32_v1".into());
    let mut record = crate::config::tests::record(&shape, KV_LAYOUTS[0], capacity, false);
    if skip {
        let at = record
            .windows(MODEL_OPTIONS_TAG_V1.len())
            .position(|s| s == MODEL_OPTIONS_TAG_V1)
            .unwrap();
        let options = serde_json::to_vec(&serde_json::json!({
            "text_config.index_topk": shape.runtime.index_topk,
            "text_config.candidate_topk_blocks": shape.runtime.candidate_topk_blocks,
            "prefill_chunk_tokens": 8192, "prefill_upper_skip": true,
        }))
        .unwrap();
        record.truncate(at + MODEL_OPTIONS_TAG_V1.len());
        record.extend_from_slice(&(options.len() as u32).to_le_bytes());
        record.extend_from_slice(&options);
    }
    let mut instance = with_dma(dma, || {
        with_bf16_gather(true, || {
            with_mode(mode, || {
                api()
                    .with_prefill_api(crate::api::BUNDLE_DEEPSEEK41_PREFILL_API_V1)
                    .unwrap()
                    .create(HostServicesTableV1::new(Rc::clone(&host)), &record)
                    .unwrap()
            })
        })
    });
    let cache = crate::instance::prefill_cache_for_test(instance.handle(), cached);
    assert_eq!(cache.len(), 4);
    let mut submitted = Vec::new();
    for &offset in offsets {
        let from = host.device.journal.borrow().len();
        let original = prompt(7, slot, offset, 8192, true);
        if skip {
            let request = encode(
                &original,
                &[Boundary {
                    request_id: 7,
                    prompt_tokens: offset + 8192,
                    chunk_end: offset + 8192,
                    chunk_tokens: 8192,
                }],
            )
            .unwrap();
            instance
                .execute_prefill(&request, &mut Views::default())
                .unwrap_or_else(|e| panic!("{e:?}: {:?}", host.logs.borrow()));
        } else {
            instance
                .execute(&original, &mut Views::default())
                .unwrap_or_else(|e| panic!("{e:?}: {:?}", host.logs.borrow()));
        }
        instance.commit().unwrap();
        let journal = host.device.journal.borrow();
        assert_launches_fit_configured_shared(&journal, &shape.topology());
        let lines = &journal[from..];
        let attention = lines
            .iter()
            .filter(|s| {
                s.starts_with("launch ")
                    && s.contains("chop_latent_sources_")
                    && !s.contains("_merge_")
            })
            .collect::<Vec<_>>();
        let decoded = attention.iter().filter(|s| s.contains("_decoded_")).count();
        let full = if skip {
            20
        } else if mode == 2 {
            0
        } else {
            40
        };
        assert_eq!(
            decoded,
            if cached && full > 0 && offset < 32768 {
                4 * if offset == 0 { 2 } else { full }
            } else {
                0
            },
            "mode={mode} dma={dma} skip={skip} offset={offset}"
        );
        if mode == 2 && !skip {
            assert_eq!(attention.len(), 2 * 4 * 40);
            for line in &attention {
                let fused = matches!(offset, 0 | 24576) && line.contains("_mixed_");
                assert!(!line.contains("_decoded_"));
                assert_eq!(line.contains("_head_shared_finish_v1"), fused);
                assert_eq!(line.contains("_reuse2_"), !fused);
            }
        }
        if skip {
            let retained = attention
                .iter()
                .filter(|s| s.contains("h16_d512_mixed_normalized_s1_r64_prefill_reuse2_v1"))
                .count();
            assert_eq!(
                retained,
                4 * if matches!(offset, 0 | 24576) { 2 } else { 20 },
                "retained native-key attention"
            );
        }
        assert!(lines.iter().any(|s| s.starts_with("launch ")
            && s.contains(if dma && !skip {
                "chop_comm_owned_sum_round_f32_v1"
            } else {
                "chop_comm_allreduce_phased_round_f32_v1"
            })));
        super::prefill_cache_capacity::assert_arena_streams(&host, &cache, lines, cached);
        if let Some(root) = std::env::var_os("CHOPINFER_DEEPSEEK41_CACHE_COMPOSITION_JOURNAL_DIR") {
            std::fs::create_dir_all(&root).unwrap();
            let root = std::path::Path::new(&root);
            let prefix = if capacity == 32 {
                "capacity32-"
            } else if offsets.len() > 2 {
                "wide-"
            } else {
                ""
            };
            let name = format!(
                "{prefix}mode-{mode}-dma-{dma}-skip-{skip}-cache-{cached}-offset-{offset}.journal"
            );
            std::fs::write(root.join(&name), lines.join("\n")).unwrap();
            super::decode_attention::export_handles(&host.device, root, &name);
            std::fs::write(root.join(format!("{name}.full")), journal.join("\n")).unwrap();
        }
        submitted.extend_from_slice(lines);
    }
    instance.retire_request(7).unwrap();
    instance.destroy().unwrap();
    assert_eq!(host.acquired.get(), host.released.get());
    submitted
}

#[test]
fn fused_prefill_checks_later_parts_and_retained_cache_fallbacks() {
    for (mode, skip, cached, fused) in [
        (2, false, true, 608),
        (0, true, true, 432),
        (2, true, false, 216),
    ] {
        let trace =
            composed_trace_offsets(mode, false, skip, cached, &[0, 8192, 16384, 24576, 32768]);
        assert_eq!(
            trace
                .iter()
                .filter(|line| line.starts_with("launch ") && line.contains("_finish_v1"))
                .count(),
            fused
        );
    }
}

fn compare_composed(mode: u8, dma: bool, skip: bool) {
    let old = composed_trace(mode, dma, skip, false);
    let new = composed_trace(mode, dma, skip, true);
    compare_traces(old, new);
}

fn compare_traces(old: Vec<String>, new: Vec<String>) {
    let unchanged = |lines: Vec<String>| {
        lines
            .into_iter()
            .filter(|s| {
                (s.starts_with("launch ")
                    || s.starts_with("peer-copy ")
                    || s.starts_with("record ")
                    || s.starts_with("wait "))
                    && !s.contains("chop_latent_prefill_")
                    && !s.contains("chop_latent_sources_")
            })
            .collect::<Vec<_>>()
    };
    let attention = |lines: &[String]| {
        lines
            .iter()
            .enumerate()
            .filter(|(_, s)| {
                s.starts_with("launch ")
                    && s.contains("chop_latent_sources_")
                    && !s.contains("_merge_")
            })
            .map(|(at, line)| (at, line.clone()))
            .collect::<Vec<_>>()
    };
    let (a, b) = (attention(&old), attention(&new));
    assert_eq!(a.len(), b.len());
    for ((at, a), (_, b)) in a.iter().zip(&b) {
        if !b.contains("_decoded_") {
            assert_eq!(a, b);
        } else if b.contains("_finish_v1") {
            // A fused decoded call inherits the original final output views.
            let before = if a.contains("_finish_v1") {
                a
            } else {
                let merge = &old[at + 1];
                assert!(merge.contains("_merge_v1"));
                merge
            };
            let source = super::shared_overlap::journal_spans(before);
            let fused = super::shared_overlap::journal_spans(b);
            let output = if a.contains("_finish_v1") {
                source[source.len() - 3..].to_vec()
            } else {
                assert_eq!(source.len(), 5);
                // The merge declares output, sinks, RoPE; fusion appends
                // sinks, RoPE, output after the native attention arguments.
                vec![source[3], source[4], source[2]]
            };
            assert_eq!(output, fused[fused.len() - 3..]);
            assert!(b.contains("NullDevicePointer, ScalarU64("));
        }
    }
    assert_eq!(
        unchanged(old),
        unchanged(new),
        "all other launches, events and peer transfers"
    );
}

#[test]
fn decoded_prefill_composes_with_overlap_bf16_and_dma() {
    for (mode, dma) in [(0, false), (1, false), (2, false), (0, true), (2, true)] {
        compare_composed(mode, dma, false);
    }
}

#[test]
fn decoded_prefill_keeps_skipped_row_parts_on_native_attention() {
    for mode in [0, 2] {
        compare_composed(mode, false, true);
    }
}

#[test]
fn capacity32_decoded_prefill_composes_with_overlap_and_retained_rows() {
    for (mode, dma, skip) in [
        (0, false, false),
        (1, false, false),
        (2, false, false),
        (0, true, false),
        (2, true, false),
        (0, false, true),
        (2, false, true),
    ] {
        let trace = |cached| {
            composed_trace_capacity(mode, dma, skip, cached, &[0, 8192, 16384, 24576], 32, 31)
        };
        compare_traces(trace(false), trace(true));
    }
}
