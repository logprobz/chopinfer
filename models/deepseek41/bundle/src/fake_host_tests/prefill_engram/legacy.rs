use super::*;

// This adapter changes only the host-visible ABI extent; the backend still
// implements every async callback, so consulting the hidden tail is observable.
// SAFETY-REVIEWED: the real create gets a complete table copy and captures it by value.
#[allow(unsafe_code)]
unsafe extern "C" fn create_without_async_tail(
    services: *const BundleHostServicesV1,
    configuration: BundleByteViewV1,
) -> BundleCreateResultV1 {
    // SAFETY-REVIEWED: BundleApiV1::create supplies its owned, complete table.
    let mut prefix = unsafe { services.read() };
    prefix.struct_size = std::mem::offset_of!(BundleHostServicesV1, create_mapped_gather_v1) as u32;
    // SAFETY-REVIEWED: the real create copies the advertised prefix during
    // this call. The owning HostServicesTableV1 keeps its backend alive through
    // instance destruction, and the original configuration view is unchanged.
    unsafe { crate::api::BUNDLE_DEEPSEEK41_API_V1.create.unwrap()(&prefix, configuration) }
}

#[test]
fn truncated_async_tail_keeps_ar_prefill_on_synchronous_gathers() {
    // DSpark requires async gathers for its device cycle before prefill.
    for rows in [193, 512] {
        let host = FakeHost::new(Shape::new(&FLASH, Ranks::TP4).unwrap(), KV_LAYOUTS[0]);
        let mut raw = crate::api::BUNDLE_DEEPSEEK41_EXTENDED_API_V1;
        raw.base.base.base.base.create = Some(create_without_async_tail);
        let mut instance = create_backend_with_api(
            &host,
            true,
            completion_backend(&host, false),
            &[],
            BundleApiV1::new_with_prefix_state_slot(raw).unwrap(),
        )
        .unwrap_or_else(|e| panic!("{e:?}: {:?}", host.logs.borrow()));
        assert!(indices(instance.handle()).is_empty());
        let begin = host.journal.borrow().len();
        let gathered = host.mapped_gather_calls.get();
        let request = encode(
            &prompt(7, 0, 0, rows, true),
            &[Boundary {
                request_id: 7,
                prompt_tokens: rows,
                chunk_end: rows,
                chunk_tokens: 8192,
            }],
        )
        .unwrap();
        let mut sink = Views::default();
        instance
            .execute_prefill(&request, &mut sink)
            .unwrap_or_else(|e| panic!("{e:?}: {:?}", host.logs.borrow()));
        assert_eq!(sink.rows(), 1);
        assert_eq!(host.mapped_gather_calls.get() - gathered, 16);
        {
            let journal = host.journal.borrow();
            let step = &journal[begin..];
            assert_eq!(
                step.iter()
                    .filter(|line| line.starts_with("mapped gather "))
                    .count(),
                16
            );
            assert_eq!(
                step.iter()
                    .any(|line| line.contains("chop_comm_owned_sum_")),
                rows == 512
            );
        }
        instance.commit().unwrap();
        instance.retire_request(7).unwrap();
        instance.destroy().unwrap();
        assert!(host.acquired.get() > 0);
        assert_eq!(host.acquired.get(), host.released.get());
        assert!(
            host.journal
                .borrow()
                .iter()
                .all(|line| !line.starts_with("async mapped ")),
            "the truncated service tail must never create, submit or release async gathers"
        );
    }
}
