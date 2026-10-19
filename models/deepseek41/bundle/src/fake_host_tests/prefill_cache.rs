//! Full chunks use the decoded cache only at qualified offsets.
use super::*;
use bundle_runtime::fake_host::conformance::ObservedHost;

#[test]
fn decoded_prefill_cache_allocation_failures_release_prior_resources() {
    let shape = Shape::new(&FLASH, Ranks::TP4).unwrap();
    let record = crate::config::tests::record(&shape, KV_LAYOUTS[0], 1, false);
    let host = FakeHost::new(shape, KV_LAYOUTS[0]);
    let mut instance = api()
        .create(HostServicesTableV1::new(Rc::clone(&host)), &record)
        .unwrap();
    let bytes =
        crate::sm120::prefill_cache::layout(8319, 64, crate::sm120::prefill_cache::PAGE_LIMIT)
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

fn high_pages(previous: u64, stride: u64) -> Vec<u8> {
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
    .with_state_slots(vec![0])
    .encode()
    .unwrap()
}

#[test]
fn decoded_prefill_rebases_high_pages_and_falls_back_for_wide_spans() {
    for stride in [1, 8] {
        let shape = Shape::new(&FLASH, Ranks::TP4).unwrap();
        let host = FakeHost::new(shape, KV_LAYOUTS[0]);
        host.kv_pages.set(219355);
        let record = crate::config::tests::record(&shape, KV_LAYOUTS[0], 1, false);
        let mut instance = api()
            .create(HostServicesTableV1::new(Rc::clone(&host)), &record)
            .unwrap();
        step(&mut instance, &high_pages(0, stride), &mut Views::default());
        let from = host.device.journal.borrow().len();
        step(
            &mut instance,
            &high_pages(8192, stride),
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
fn decoded_prefill_routes_winning_offsets_and_preserves_merge() {
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
            assert!(line.contains("_reuse2_v1"), "{line}");
            assert!(line.contains("shared_memory_bytes: 85504,"), "{line}");
        }
        let merges: Vec<_> = launches
            .iter()
            .filter(|l| l.contains("chop_latent_sources_") && l.contains("_merge_"))
            .collect();
        assert_eq!(merges.len(), attention.len());
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
