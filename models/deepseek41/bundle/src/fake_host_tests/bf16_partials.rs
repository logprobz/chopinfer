use super::*;

const PREFIX: &str = "chop_comm_allreduce_phased_";

pub(super) fn host(admitted: bool) -> Rc<Host> {
    let host = FakeHost::new(Shape::new(&FLASH, Ranks::TP4).unwrap(), KV_LAYOUTS[0]);
    if admitted {
        host.device.residency.set(Some(BundleRawClusterResidencyV1 {
            multiprocessor_count: 188,
            usable_multiprocessors: 188,
            max_active_clusters: 188,
            flags: BUNDLE_CLUSTER_RESIDENCY_V1_CLUSTERS_REPORTED
                | BUNDLE_CLUSTER_RESIDENCY_V1_COOPERATIVE_LAUNCH,
            ..BundleRawClusterResidencyV1::unreported()
        }));
    } else {
        host.device.residency.set(None);
    }
    host
}

fn partial_entry(line: &str) -> bool {
    line.contains(PREFIX) && line.contains("bf16_v1")
}

#[test]
fn bf16_partials_refuse_unadmitted_grids_and_compose_with_dma() {
    let host = host(false);
    let record = crate::config::tests::record(&host.shape, host.kv, 1, false);
    let result = { api().create(HostServicesTableV1::new(Rc::clone(&host)), &record) };
    assert!(result.is_err());
    assert!(
        host.logs
            .borrow()
            .iter()
            .any(|line| line.contains("cannot admit"))
    );
    assert_eq!(host.acquired.get(), host.released.get());
    // Prefill DMA carries rounded BF16 partials, so the two selections
    // compose in the sparse-DMA fixtures.
    assert!(crate::Kernels::new(&host.shape).is_ok());
}

#[test]
fn prefill_only_partials_keep_f32_draft_and_verify_reductions() {
    check_draft_reductions();
}

fn check_draft_reductions() {
    use bundle_runtime::entry::Bundle;
    use bundle_runtime::fake_host::RecordingSink;
    use runtime_bundle_sdk::{BatchAppendV1, BatchItemV1, BatchRequestV1, BatchWireVersionV1};
    type Sink = RecordingSink<super::conformance::Limits>;
    let host = host(true);
    host.drafter.set(true);
    host.speculation.set(Some((5, 32)));
    let record = crate::config::tests::record(&host.shape, host.kv, 8, false);
    let mut instance = {
        api().create(
            HostServicesTableV1::new(Rc::new(super::dspark::MappedHost::new(Rc::clone(&host)))),
            &record,
        )
    }
    .unwrap_or_else(|e| panic!("BF16 draft create: {e:?} {:?}", host.logs.borrow()));
    for slot in 0..8 {
        instance
            .execute(&prompt(7 + slot, slot, 0, 1, true), &mut Sink::default())
            .unwrap();
        instance.commit().unwrap();
    }
    for start in [1, 2] {
        let reset = (start == 2).then(|| {
            <crate::instance::Instance as Bundle>::instances().with(|table| {
                table
                    .borrow_mut()
                    .get_mut(instance.handle())
                    .unwrap()
                    .reset_prefill_reduce_next_run()
            })
        });
        host.device.journal.borrow_mut().clear();
        let batch = BatchRequestV1::new(
            BatchWireVersionV1::V3,
            (0..8)
                .map(|slot| BatchItemV1::verify(7 + slot, 4, true, vec![9]))
                .collect(),
            (0..8)
                .map(|slot| BatchAppendV1::new(start, 4, start + 4, vec![slot * 8], None))
                .collect(),
        )
        .with_state_slots((0..8).collect())
        .encode()
        .unwrap();
        let mut sink = Sink::default();
        instance
            .execute(&batch, &mut sink)
            .unwrap_or_else(|e| panic!("BF16 verify: {e:?} {:?}", host.logs.borrow()));
        assert_eq!(sink.draft_tokens().len(), 8);
        if start == 1 {
            let journal = host.device.journal.borrow();
            assert!(
                !journal
                    .iter()
                    .any(|line| line.starts_with("launch ") && partial_entry(line))
            );
            assert!(journal.iter().any(|line| line.starts_with("launch ")
                && line.contains("chop_comm_reduce_scatter_f32_v1")));
        }
        if let Some(spans) = reset {
            assert_no_protocol_zero(&host.device.journal.borrow(), &spans);
        }
        instance
            .commit_accepted(
                &(0..8)
                    .map(|slot| BundleKeptRowsV1 {
                        struct_size: std::mem::size_of::<BundleKeptRowsV1>() as u32,
                        request_id: 7 + slot,
                        kept_rows: 1,
                        reserved: 0,
                        reserved_tail: 0,
                    })
                    .collect::<Vec<_>>(),
            )
            .unwrap();
    }
    instance.destroy().unwrap();
    assert_eq!(host.acquired.get(), host.released.get());
}

#[test]
fn prefill_only_partials_keep_small_rows_and_original_decode_fusion() {
    use bundle_runtime::entry::Bundle;
    for rows in [1, 63, 8191] {
        let host = host(true);
        let mut instance = create(&host, 2);
        host.device.journal.borrow_mut().clear();
        step(
            &mut instance,
            &prompt(7, 0, 0, rows, true),
            &mut Views::default(),
        );
        {
            let journal = host.device.journal.borrow();
            assert!(
                journal
                    .iter()
                    .any(|s| s.starts_with("launch ") && partial_entry(s))
            );
            assert!(
                !journal
                    .iter()
                    .any(|s| s.starts_with("launch ")
                        && s.contains("chop_comm_reduce_scatter_f32_v1"))
            );
        }
        let spans = <crate::instance::Instance as Bundle>::instances().with(|table| {
            table
                .borrow_mut()
                .get_mut(instance.handle())
                .unwrap()
                .reset_prefill_reduce_next_run()
        });
        assert_eq!(spans.len(), 12);
        host.device.journal.borrow_mut().clear();
        step(
            &mut instance,
            &decode(&[(7, 0, rows, 9)]),
            &mut Views::default(),
        );
        {
            let journal = host.device.journal.borrow();
            assert!(
                !journal
                    .iter()
                    .any(|s| s.starts_with("launch ") && partial_entry(s))
            );
            assert!(
                journal
                    .iter()
                    .any(|s| s.contains("chop_comm_reduce_scatter_f32_v1"))
            );
            // C1 decode keeps the served fusion: the column-block pair's
            // split finish carries the finish/norm tail.
            assert!(
                journal
                    .iter()
                    .any(|s| s.contains("chop_deepseek41_hc_finish_split_w32_v1"))
            );
            assert_no_protocol_zero(&journal, &spans);
        }
        // Decode must leave the pending prefill reset alone. The next
        // prefill drains and clears only its own protocol spans.
        host.device.journal.borrow_mut().clear();
        step(
            &mut instance,
            &prompt(8, 1, 0, 1, true),
            &mut Views::default(),
        );
        {
            let journal = host.device.journal.borrow();
            let first = journal.iter().position(|s| s.starts_with("zero ")).unwrap();
            let actual = journal[first..first + spans.len()]
                .iter()
                .flat_map(|s| super::shared_overlap::journal_spans(s))
                .collect::<Vec<_>>();
            assert_eq!(
                actual,
                spans
                    .iter()
                    .map(|s| (s.buffer.raw, s.offset, s.length))
                    .collect::<Vec<_>>()
            );
        }
        instance.destroy().unwrap();
        assert_eq!(host.acquired.get(), host.released.get());
    }
}

fn assert_no_protocol_zero(journal: &[String], spans: &[DeviceBufferSpanV1]) {
    for line in journal.iter().filter(|s| s.starts_with("zero ")) {
        for (raw, offset, length) in super::shared_overlap::journal_spans(line) {
            assert!(
                !spans.iter().any(|span| {
                    span.buffer.raw == raw
                        && span.offset < offset + length
                        && offset < span.offset + span.length
                }),
                "decode cleared prefill protocol: {line}"
            );
        }
    }
}

#[test]
fn unset_bf16_partials_select_prefill_and_keep_decode_f32() {
    let host = host(true);
    let mut instance = create(&host, 1);
    host.device.journal.borrow_mut().clear();
    step(
        &mut instance,
        &prompt(7, 0, 0, 63, true),
        &mut Views::default(),
    );
    assert!(
        host.device
            .journal
            .borrow()
            .iter()
            .any(|line| line.starts_with("launch ") && partial_entry(line))
    );
    host.device.journal.borrow_mut().clear();
    step(
        &mut instance,
        &decode(&[(7, 0, 63, 9)]),
        &mut Views::default(),
    );
    let journal = host.device.journal.borrow();
    assert!(
        !journal
            .iter()
            .any(|line| line.starts_with("launch ") && partial_entry(line))
    );
    assert!(
        journal
            .iter()
            .any(|line| line.contains("chop_comm_reduce_scatter_f32_v1"))
    );
    // C1 decode keeps the served fusion: the column-block pair's split
    // finish carries the finish/norm tail.
    assert!(
        journal
            .iter()
            .any(|line| line.contains("chop_deepseek41_hc_finish_split_w32_v1"))
    );
    drop(journal);
    instance.destroy().unwrap();
    assert_eq!(host.acquired.get(), host.released.get());
}

#[test]
fn unset_bf16_partials_keep_other_kv_on_f32() {
    let host = FakeHost::new(Shape::new(&FLASH, Ranks::TP4).unwrap(), KV_LAYOUTS[1]);
    assert_eq!(host.kv.storage, KvStorage::Bf16);
    let mut instance = create(&host, 1);
    assert!(
        !host
            .device
            .journal
            .borrow()
            .iter()
            .any(|line| partial_entry(line))
    );
    host.device.journal.borrow_mut().clear();
    step(
        &mut instance,
        &prompt(7, 0, 0, 63, true),
        &mut Views::default(),
    );
    let journal = host.device.journal.borrow();
    assert!(
        !journal
            .iter()
            .any(|line| line.starts_with("launch ") && partial_entry(line))
    );
    assert!(
        journal
            .iter()
            .any(|line| line.contains("chop_comm_reduce_scatter_f32_v1"))
    );
    drop(journal);
    instance.destroy().unwrap();
    assert_eq!(host.acquired.get(), host.released.get());
}

#[test]
fn bf16_prefill_partials_compose_with_native_experts() {
    for rows in [65, 8191, 8192] {
        let host = host(true);
        let mut instance = { create(&host, 1) };
        let begin = host.device.journal.borrow().len();
        step(
            &mut instance,
            &prompt(7, 0, 0, rows, true),
            &mut Views::default(),
        );
        {
            let journal = host.device.journal.borrow();
            let launches = journal[begin..]
                .iter()
                .filter(|line| line.starts_with("launch "))
                .collect::<Vec<_>>();
            let native = launches
                .iter()
                .filter(|line| line.contains("chop_moe_native_"))
                .count();
            assert!(native > 0 && native % 16 == 0, "{rows}: {native}");
            assert!(launches.iter().any(|line| partial_entry(line)), "{rows}");
            // At the exact full bucket, native replaces the grouped down.
            assert!(
                launches
                    .iter()
                    .all(|line| !line.contains("_group_m128n128")),
                "{rows}"
            );
        }
        for repeat in 0..2 {
            let begin = host.device.journal.borrow().len();
            step(
                &mut instance,
                &decode(&[(7, 0, rows + repeat, 9)]),
                &mut Views::default(),
            );
            let journal = host.device.journal.borrow();
            assert!(journal[begin..].iter().all(|line| {
                !(line.contains("chop_moe_native_")
                    || line.starts_with("launch ") && partial_entry(line))
            }));
        }
        instance.destroy().unwrap();
        assert_eq!(host.acquired.get(), host.released.get());
        if let Some(dir) = std::env::var_os("CHOPINFER_DEEPSEEK41_COMBINED_JOURNAL_DIR") {
            let dir = std::path::PathBuf::from(dir);
            std::fs::create_dir_all(&dir).unwrap();
            let name = format!("prefill-{rows}.journal");
            std::fs::write(dir.join(&name), host.device.journal.borrow().join("\n")).unwrap();
            super::decode_attention::export_handles(&host.device, &dir, &name);
        }
    }
}

#[test]
fn bf16_prefill_and_native_experts_keep_f32_draft_and_verify() {
    check_draft_reductions();
}
