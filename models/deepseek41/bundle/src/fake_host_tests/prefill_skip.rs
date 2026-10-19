use super::*;
use bundle_runtime::fake_host::RecordingSink;
use runtime_bundle_sdk::prefill::{Boundary, encode};

type Sink = RecordingSink<super::conformance::Limits>;

fn create_profile(host: &Rc<Host>, chunk: u32, skip: bool) -> BundleInstanceV1 {
    let mut record = crate::config::tests::record(&host.shape, host.kv, 4, false);
    let at = record
        .windows(MODEL_OPTIONS_TAG_V1.len())
        .position(|w| w == MODEL_OPTIONS_TAG_V1)
        .unwrap();
    let options = serde_json::to_vec(&serde_json::json!({
        "text_config.index_topk": host.shape.runtime.index_topk,
        "text_config.candidate_topk_blocks": host.shape.runtime.candidate_topk_blocks,
        "prefill_chunk_tokens": chunk, "prefill_upper_skip": skip,
    }))
    .unwrap();
    record.truncate(at + MODEL_OPTIONS_TAG_V1.len());
    record.extend_from_slice(&(options.len() as u32).to_le_bytes());
    record.extend_from_slice(&options);
    let services = HostServicesTableV1::new(Rc::new(dspark::MappedHost(Rc::clone(host))));
    api()
        .with_prefill_api(crate::api::BUNDLE_DEEPSEEK41_PREFILL_API_V1)
        .unwrap()
        .create(services, &record)
        .unwrap_or_else(|e| panic!("{e:?}: {:?}", host.logs.borrow()))
}

#[test]
fn exact_prefill_profiles_execute_capture_restore_and_continue() {
    for dspark in [false, true] {
        for chunk in [2048_u32, 8192] {
            let host = FakeHost::new(Shape::new(&FLASH, Ranks::TP4).unwrap(), KV_LAYOUTS[0]);
            host.drafter.set(dspark);
            if dspark {
                host.speculation.set(Some((5, 32)));
            }
            let mut instance = create_profile(&host, chunk, true);
            assert_eq!(instance.prefill_policy().unwrap().chunk_tokens, chunk);
            let prompt_end = u64::from(chunk) + 17;
            for (start, rows) in [(0, u64::from(chunk)), (u64::from(chunk), 17)] {
                let original = prompt(1, 0, start, rows, start + rows == prompt_end);
                let request = encode(
                    &original,
                    &[Boundary {
                        request_id: 1,
                        prompt_tokens: prompt_end,
                        chunk_end: start + rows,
                        chunk_tokens: chunk,
                    }],
                )
                .unwrap();
                let mut sink = Sink::default();
                instance
                    .execute_prefill(&request, &mut sink)
                    .unwrap_or_else(|e| {
                        panic!(
                            "dspark={dspark}, chunk={chunk}: {e:?}: {:?}",
                            host.logs.borrow()
                        )
                    });
                assert_eq!(
                    sink.views.iter().map(|v| v.raw().row_count).sum::<u32>(),
                    u32::from(start > 0)
                );
                instance.commit().unwrap();
                assert_eq!(
                    crate::instance::ledger_state(instance.handle(), 0),
                    (Some((1, start + rows)), false)
                );
            }
            let pages: Vec<_> = (130..130 + instance.describe_prefix_state().unwrap()).collect();
            instance
                .capture_prefix_state(1, prompt_end, &pages)
                .unwrap();
            instance.retire_request(1).unwrap();
            let ids: Vec<_> = (0..prompt_end as u32).collect();
            instance
                .restore_prefix_state_in_slot(2, &ids, &pages, 0)
                .unwrap();
            let mut sink = Sink::default();
            instance
                .execute(&decode(&[(2, 0, prompt_end, 7)]), &mut sink)
                .unwrap();
            instance.commit().unwrap();
            assert_eq!(sink.views.iter().map(|v| v.raw().row_count).sum::<u32>(), 1);
            instance.retire_request(2).unwrap();
            instance.destroy().unwrap();
            assert_eq!(host.acquired.get(), host.released.get());
        }
    }
}

#[test]
fn legacy_execute_keeps_full_schedule_even_when_profile_requests_skip() {
    let launch_names = |skip| {
        let host = FakeHost::new(Shape::new(&FLASH, Ranks::TP1).unwrap(), KV_LAYOUTS[0]);
        let mut instance = create_profile(&host, 2048, skip);
        host.device.journal.borrow_mut().clear();
        step(
            &mut instance,
            &prompt(1, 0, 0, 257, true),
            &mut Views::default(),
        );
        let names: Vec<_> = host
            .device
            .journal
            .borrow()
            .iter()
            .filter(|s| s.starts_with("launch "))
            .map(|s| s.split_whitespace().nth(2).unwrap().to_owned())
            .collect();
        instance.retire_request(1).unwrap();
        instance.destroy().unwrap();
        names
    };
    let full = launch_names(false);
    assert!(!full.is_empty());
    assert_eq!(full, launch_names(true));
}

#[cfg(feature = "diagnostic-taps")]
#[test]
fn profile_capture_contains_chunk_state_snapshot_and_raw_logits() {
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    for dspark in [false, true] {
        let root = std::path::PathBuf::from(std::env::var_os("CHOPINFER_DATA").unwrap())
            .join("deepseek41-tests")
            .join(format!(
                "prefill-state-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            ));
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join("enabled"), []).unwrap();
        let host = FakeHost::new(Shape::new(&FLASH, Ranks::TP1).unwrap(), KV_LAYOUTS[0]);
        host.drafter.set(dspark);
        if dspark {
            host.speculation.set(Some((5, 32)));
        }
        let mut instance = create_profile(&host, 2048, true);
        crate::instance::set_dspark_state_capture_root_for_test(
            instance.handle(),
            Some(root.clone()),
        );
        let request = encode(
            &prompt(1, 0, 0, 137, true),
            &[Boundary {
                request_id: 1,
                prompt_tokens: 137,
                chunk_end: 137,
                chunk_tokens: 2048,
            }],
        )
        .unwrap();
        instance
            .execute_prefill(&request, &mut Sink::default())
            .unwrap();
        instance.commit().unwrap();
        let pages = (40..40 + instance.describe_prefix_state().unwrap()).collect::<Vec<_>>();
        instance.capture_prefix_state(1, 137, &pages).unwrap();
        for kind in ["chunk", "snapshot"] {
            let directory = root.join(format!("{kind}-137/request-1"));
            let metadata: serde_json::Value =
                serde_json::from_slice(&std::fs::read(directory.join("state.json")).unwrap())
                    .unwrap();
            assert_eq!(metadata["end"], 137);
            assert!(!metadata["pieces"].as_array().unwrap().is_empty());
            assert!(!metadata["paged_pieces"].as_array().unwrap().is_empty());
            assert!(
                std::fs::metadata(directory.join("rank-0.bin"))
                    .unwrap()
                    .len()
                    > 0
            );
            assert!(
                std::fs::metadata(directory.join("kv-rank-0.bin"))
                    .unwrap()
                    .len()
                    > 0
            );
        }
        assert_eq!(
            std::fs::metadata(root.join("logits/step-1-view-0.f32"))
                .unwrap()
                .len(),
            u64::from(FLASH.shape.vocab) * 4
        );
        instance.retire_request(1).unwrap();
        instance.destroy().unwrap();
        assert_eq!(host.acquired.get(), host.released.get());
        std::fs::remove_dir_all(root).unwrap();
    }
}
