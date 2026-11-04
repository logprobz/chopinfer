use super::*;
use runtime_bundle_abi::PhysicalKvLayoutV1;

const LAYOUT: &[u8] = b"fixture.kv";
const STATE: &[u8] = b"fixture.state";

fn config() -> HostInstanceConfigV1 {
    let layout = DeclaredPageLayoutV1::new(
        8,
        1,
        8,
        64,
        2,
        runtime_kv::RankPlacementV1::Sharded,
        StateLifetimeV1::Request,
        64,
    )
    .unwrap();
    HostInstanceConfigV1::new(HostTopologyV1::new(vec![0, 1]).unwrap())
        .with_kv_contract(DeclaredKvContractV1::new(LAYOUT.to_vec(), layout))
        .with_state_contract(DeclaredStateContractV1::new(STATE.to_vec(), 64, 64))
}

fn kv_request() -> PhysicalKvRequestV1 {
    PhysicalKvRequestV1 {
        layout: PhysicalKvLayoutV1 {
            flags: 0,
            layout_id: LAYOUT.to_vec(),
            page_bytes: 64,
            page_alignment: 64,
            page_tokens: 8,
            rank_count: 2,
            state_lifetime: state_lifetime_code_v1(StateLifetimeV1::Request),
        },
        flags: 0,
        page_capacity: 8,
        request_capacity: 1,
    }
}

fn state_request() -> StateRequestV1 {
    StateRequestV1 {
        flags: 0,
        state_id: STATE.to_vec(),
        byte_length: 64,
        alignment: 64,
    }
}

#[test]
fn logical_kv_registration_failure_changes_no_sequence_or_append() {
    for slot in [false, true] {
        let s = Services::new(FakeHostDeviceV1::new(), config());
        let kv = s.allocate_physical_kv(&kv_request()).unwrap();
        let key = slot_key_v1(kv.generation(), kv.slot()).unwrap();
        let next = s.state.borrow().order.next;
        let before = s.state.borrow().kv.get(key).unwrap().pool.stats();
        if slot {
            s.state.borrow_mut().kv_requests.refuse_registration(true);
        } else {
            s.state.borrow_mut().order.next = u64::MAX;
        }
        assert_eq!(
            s.create_kv_request(kv, 16),
            Err(BundleStatusV1::ResourceExhausted)
        );
        assert_eq!(s.state.borrow().kv.get(key).unwrap().pool.stats(), before);
        s.state.borrow_mut().order.next = next;
        s.state.borrow_mut().kv_requests.refuse_registration(false);
        let request = s
            .create_kv_request(kv, 16)
            .expect("failed registration cannot consume sequence capacity");
        let next = s.state.borrow().order.next;
        let before = s.state.borrow().kv.get(key).unwrap().pool.stats();
        if slot {
            s.state
                .borrow_mut()
                .kv_transactions
                .refuse_registration(true);
        } else {
            s.state.borrow_mut().order.next = u64::MAX;
        }
        assert_eq!(
            s.prepare_kv(request, &kv_append_operation_v1(4)),
            Err(BundleStatusV1::ResourceExhausted)
        );
        assert_eq!(s.state.borrow().kv.get(key).unwrap().pool.stats(), before);
        assert_eq!(
            s.state.borrow().kv.get(key).unwrap().pool.prepared_id(),
            None
        );
        s.state.borrow_mut().order.next = next;
        s.state
            .borrow_mut()
            .kv_transactions
            .refuse_registration(false);
        let append = s.prepare_kv(request, &kv_append_operation_v1(4)).unwrap();
        s.cancel_kv(append).unwrap();
        let append = s.prepare_kv(request, &kv_append_operation_v1(4)).unwrap();
        s.commit_kv(append).unwrap();
        s.release_kv_request(request).unwrap();
        s.release_physical_kv(kv).unwrap();
        assert_eq!(s.with_device(FakeHostDeviceV1::live_object_count), 0);
    }
}

#[test]
fn compound_registration_precedes_host_rank_and_buffer_creation() {
    for physical in [false, true] {
        for slot in [false, true] {
            let s = Services::new(FakeHostDeviceV1::new(), config());
            let next = s.state.borrow().order.next;
            if slot {
                if physical {
                    s.state.borrow_mut().kv.refuse_registration(true);
                } else {
                    s.state.borrow_mut().states.refuse_registration(true);
                }
            } else {
                s.state.borrow_mut().order.next = u64::MAX;
            }
            s.with_device_mut(|device| {
                for refusal in [
                    FakeRefusalV1::ReleaseDeviceBuffer,
                    FakeRefusalV1::ReleaseStream,
                    FakeRefusalV1::ReleaseMemoryPool,
                    FakeRefusalV1::ReleaseContext,
                ] {
                    device.refuse(refusal);
                }
            });
            let result = if physical {
                s.allocate_physical_kv(&kv_request()).map(|_| ())
            } else {
                s.allocate_state(&state_request()).map(|_| ())
            };
            assert_eq!(result, Err(BundleStatusV1::ResourceExhausted));
            assert_eq!(
                s.with_device(FakeHostDeviceV1::issued_object_count),
                0,
                "reserve the parent before its children"
            );
            assert_eq!(s.with_device(FakeHostDeviceV1::live_object_count), 0);
            assert_eq!(s.state.borrow().host_rank_users, 0);
            assert!(s.state.borrow().host_ranks.is_empty());
            assert_eq!(s.live_resource_count(), 0);
            {
                let mut state = s.state.borrow_mut();
                state.order.next = next;
                state.states.refuse_registration(false);
                state.kv.refuse_registration(false);
            }
            s.with_device_mut(|device| {
                for refusal in [
                    FakeRefusalV1::ReleaseDeviceBuffer,
                    FakeRefusalV1::ReleaseStream,
                    FakeRefusalV1::ReleaseMemoryPool,
                    FakeRefusalV1::ReleaseContext,
                ] {
                    device.allow(refusal);
                }
            });
            if physical {
                let handle = s.allocate_physical_kv(&kv_request()).unwrap();
                s.release_physical_kv(handle).unwrap();
            } else {
                let handle = s.allocate_state(&state_request()).unwrap();
                s.release_state(handle).unwrap();
            }
            assert_eq!(s.with_device(FakeHostDeviceV1::live_object_count), 0);
        }
    }
}

#[test]
fn later_rank_failure_retains_buffers_and_host_rank_dependencies() {
    for alignment in [false, true] {
        let s = Services::new(FakeHostDeviceV1::new(), config());
        s.with_device_mut(|device| {
            device.refuse_after(
                if alignment {
                    FakeRefusalV1::AllocationAlignment
                } else {
                    FakeRefusalV1::AllocateDeviceBuffer
                },
                1,
            );
            device.refuse(FakeRefusalV1::ReleaseDeviceBuffer);
        });
        assert_eq!(
            s.allocate_state(&state_request()),
            Err(BundleStatusV1::HostError)
        );
        let issued = s.with_device(FakeHostDeviceV1::issued_object_count);
        let live = s.with_device(FakeHostDeviceV1::live_object_count);
        assert!(live >= 7, "buffers keep the per-rank dependencies alive");
        assert_eq!(
            s.live_resource_count(),
            1,
            "the incomplete parent stays owned"
        );
        let incomplete = match s.state.borrow().pending_compound_cleanup {
            Some(CompoundCleanupV1::Ranks(RankOwnerV1::State(key))) => key,
            _ => panic!("the failed state must retain its parent"),
        };
        assert_eq!(
            s.resolve_state_view(
                BundleStateHandleV1::from_parts(incomplete.generation, incomplete.slot),
                0
            ),
            Err(BundleStatusV1::InvalidState),
            "an incomplete parent cannot expose an earlier rank"
        );
        for _ in 0..2 {
            assert_eq!(
                s.allocate_state(&state_request()),
                Err(BundleStatusV1::HostError)
            );
            assert_eq!(s.with_device(FakeHostDeviceV1::issued_object_count), issued);
            assert_eq!(s.with_device(FakeHostDeviceV1::live_object_count), live);
        }
        s.with_device_mut(|device| device.allow(FakeRefusalV1::ReleaseDeviceBuffer));
        s.state.borrow_mut().retry_acquisition_cleanup().unwrap();
        assert_eq!(s.with_device(FakeHostDeviceV1::live_object_count), 0);
        assert_eq!(s.live_resource_count(), 0);
        let handle = s.allocate_state(&state_request()).unwrap();
        s.release_state(handle).unwrap();
        assert_eq!(s.with_device(FakeHostDeviceV1::live_object_count), 0);
    }
}

#[test]
fn serving_kv_stream_survives_failed_buffer_registration_and_cleanup() {
    let s = Services::new(
        FakeHostDeviceV1::new(),
        config().with_serving_kv(ServingKvPolicyV1 {
            maximum_pages: Some(8),
            ..Default::default()
        }),
    );
    let mut selected = Vec::new();
    for rank in 0..2 {
        let context = s.select_context(rank, rank as i32).unwrap();
        let pool = s.select_memory_pool(context).unwrap();
        selected.push((context, pool));
    }
    let live = s.with_device(FakeHostDeviceV1::live_object_count);
    s.state.borrow_mut().buffers.refuse_registration(true);
    s.with_device_mut(|device| device.refuse(FakeRefusalV1::ReleaseStream));
    assert_eq!(s.acquire_serving_kv(LAYOUT), Err(BundleStatusV1::HostError));
    let issued = s.with_device(FakeHostDeviceV1::issued_object_count);
    assert_eq!(s.with_device(FakeHostDeviceV1::live_object_count), live + 1);
    assert_eq!(s.serving_kv_pages(), None);
    assert!(s.resolve_serving_kv_pool(0).is_err());
    assert!(s.state.borrow().device_memory.is_empty());
    for _ in 0..2 {
        assert_eq!(s.acquire_serving_kv(LAYOUT), Err(BundleStatusV1::HostError));
        assert_eq!(s.with_device(FakeHostDeviceV1::issued_object_count), issued);
    }
    assert_eq!(
        s.release_memory_pool(selected[1].1),
        Err(BundleStatusV1::HostError)
    );
    s.with_device_mut(|device| device.allow(FakeRefusalV1::ReleaseStream));
    s.state.borrow_mut().retry_acquisition_cleanup().unwrap();
    assert_eq!(s.with_device(FakeHostDeviceV1::live_object_count), live);
    s.state.borrow_mut().buffers.refuse_registration(false);
    assert_eq!(s.acquire_serving_kv(LAYOUT).unwrap(), 8);
    let view = s.resolve_serving_kv_pool(0).unwrap();
    let issued = s.with_device(FakeHostDeviceV1::issued_object_count);
    assert_eq!(s.acquire_serving_kv(LAYOUT).unwrap(), 8);
    assert_eq!(s.resolve_serving_kv_pool(0).unwrap(), view);
    assert_eq!(s.with_device(FakeHostDeviceV1::issued_object_count), issued);
}

#[test]
fn logits_read_keeps_its_temporary_stream_when_sync_and_release_refuse() {
    let fixture = DirectFixture::new();
    let s = &fixture.services;
    let buffer = s
        .allocate_device_buffer(fixture.pool, fixture.stream, 8, 4)
        .unwrap();
    let view = runtime_bundle_abi::OwnedLogitsViewV1::new(runtime_bundle_abi::BundleLogitsViewV1 {
        struct_size: std::mem::size_of::<runtime_bundle_abi::BundleLogitsViewV1>() as u32,
        reserved: 0,
        context: fixture.context,
        rank: 0,
        element_type: runtime_bundle_abi::BUNDLE_LOGITS_ELEMENT_V1_F32,
        buffer: runtime_bundle_abi::BundleDeviceBufferViewV1 {
            buffer,
            offset: 0,
            length: 8,
            ..runtime_bundle_abi::BundleDeviceBufferViewV1::empty()
        },
        row_stride_elements: 2,
        row_count: 1,
        live_width: 2,
        first_item_index: 0,
        reserved_tail: 0,
    });
    let live = s.with_device(FakeHostDeviceV1::live_object_count);
    s.with_device_mut(|device| {
        device.refuse(FakeRefusalV1::SynchronizeStream);
        device.refuse(FakeRefusalV1::ReleaseStream);
    });
    assert_eq!(
        s.read_logits_view_row(view, 0),
        Err(BundleStatusV1::HostError)
    );
    assert_eq!(s.with_device(FakeHostDeviceV1::live_object_count), live + 1);
    let issued = s.with_device(FakeHostDeviceV1::issued_object_count);
    assert_eq!(
        s.read_logits_view_row(view, 0),
        Err(BundleStatusV1::HostError)
    );
    assert_eq!(s.with_device(FakeHostDeviceV1::issued_object_count), issued);
    assert_eq!(
        s.release_device_buffer(buffer),
        Err(BundleStatusV1::HostError)
    );
    s.with_device_mut(|device| {
        device.allow(FakeRefusalV1::SynchronizeStream);
        device.allow(FakeRefusalV1::ReleaseStream);
    });
    assert_eq!(s.read_logits_view_row(view, 0).unwrap(), vec![0; 8]);
    assert_eq!(s.with_device(FakeHostDeviceV1::live_object_count), live);
    s.release_device_buffer(buffer).unwrap();
    fixture.release();
}

#[test]
fn partial_host_rank_construction_retains_each_refused_dependency_phase() {
    for (constructor, allowed, release, live) in [
        (
            FakeRefusalV1::CreateMemoryPool,
            0,
            FakeRefusalV1::ReleaseContext,
            1,
        ),
        (
            FakeRefusalV1::CreateStream,
            0,
            FakeRefusalV1::ReleaseMemoryPool,
            2,
        ),
        (
            FakeRefusalV1::CreateContext,
            1,
            FakeRefusalV1::ReleaseStream,
            3,
        ),
    ] {
        let s = Services::new(FakeHostDeviceV1::new(), config());
        s.with_device_mut(|device| {
            device.refuse_after(constructor, allowed);
            device.refuse(release);
        });
        assert_eq!(
            s.allocate_physical_kv(&kv_request()),
            Err(BundleStatusV1::HostError)
        );
        assert_eq!(s.with_device(FakeHostDeviceV1::live_object_count), live);
        assert_eq!(s.live_resource_count(), 1);
        let issued = s.with_device(FakeHostDeviceV1::issued_object_count);
        assert_eq!(
            s.allocate_physical_kv(&kv_request()),
            Err(BundleStatusV1::HostError)
        );
        assert_eq!(s.with_device(FakeHostDeviceV1::issued_object_count), issued);
        s.with_device_mut(|device| device.allow(release));
        s.state.borrow_mut().retry_acquisition_cleanup().unwrap();
        assert_eq!(s.with_device(FakeHostDeviceV1::live_object_count), 0);
        assert_eq!(s.live_resource_count(), 0);
        let kv = s.allocate_physical_kv(&kv_request()).unwrap();
        s.release_physical_kv(kv).unwrap();
        assert_eq!(s.with_device(FakeHostDeviceV1::live_object_count), 0);
    }
}

#[test]
fn last_shared_rank_user_survives_refused_stream_pool_and_context_teardown() {
    for release in [
        FakeRefusalV1::ReleaseStream,
        FakeRefusalV1::ReleaseMemoryPool,
        FakeRefusalV1::ReleaseContext,
    ] {
        let s = Services::new(FakeHostDeviceV1::new(), config());
        let kv = s.allocate_physical_kv(&kv_request()).unwrap();
        let state = s.allocate_state(&state_request()).unwrap();
        assert_eq!(s.state.borrow().host_rank_users, 2);
        s.with_device_mut(|device| device.refuse(release));
        s.release_state(state).unwrap();
        assert_eq!(s.state.borrow().host_rank_users, 1);
        assert_eq!(s.release_physical_kv(kv), Err(BundleStatusV1::HostError));
        let live = s.with_device(FakeHostDeviceV1::live_object_count);
        assert!(live > 0);
        assert_eq!(s.state.borrow().host_rank_users, 1);
        assert_eq!(s.live_resource_count(), 1);
        assert_eq!(s.release_physical_kv(kv), Err(BundleStatusV1::HostError));
        assert_eq!(s.with_device(FakeHostDeviceV1::live_object_count), live);
        s.with_device_mut(|device| device.allow(release));
        s.release_physical_kv(kv).unwrap();
        assert_eq!(s.state.borrow().host_rank_users, 0);
        assert_eq!(s.with_device(FakeHostDeviceV1::live_object_count), 0);
        assert_eq!(s.live_resource_count(), 0);
    }
}

#[test]
fn borrowed_host_rank_dependencies_remain_owned_by_their_public_handles() {
    let s = Services::new(FakeHostDeviceV1::new(), config());
    let mut selected = Vec::new();
    for rank in 0..2 {
        let context = s.select_context(rank, rank as i32).unwrap();
        let pool = s.select_memory_pool(context).unwrap();
        selected.push((context, pool));
    }
    let baseline = s.with_device(FakeHostDeviceV1::live_object_count);
    s.with_device_mut(|device| {
        device.refuse_after(FakeRefusalV1::CreateStream, 1);
        device.refuse(FakeRefusalV1::ReleaseStream);
    });
    assert_eq!(
        s.allocate_state(&state_request()),
        Err(BundleStatusV1::HostError)
    );
    assert_eq!(
        s.release_memory_pool(selected[1].1),
        Err(BundleStatusV1::HostError)
    );
    assert_eq!(
        s.with_device(FakeHostDeviceV1::live_object_count),
        baseline + 1
    );
    s.with_device_mut(|device| device.allow(FakeRefusalV1::ReleaseStream));
    s.state.borrow_mut().retry_acquisition_cleanup().unwrap();
    assert_eq!(s.with_device(FakeHostDeviceV1::live_object_count), baseline);
    for (context, pool) in selected.into_iter().rev() {
        s.release_memory_pool(pool).unwrap();
        s.release_context(context).unwrap();
    }
    assert_eq!(s.with_device(FakeHostDeviceV1::live_object_count), 0);
}

#[test]
fn serving_kv_later_rank_failure_retains_independent_buffer_and_stream_phases() {
    for fault in [
        FakeRefusalV1::AllocateDeviceBuffer,
        FakeRefusalV1::AllocationAlignment,
    ] {
        let s = Services::new(
            FakeHostDeviceV1::new(),
            config().with_serving_kv(ServingKvPolicyV1 {
                maximum_pages: Some(8),
                ..Default::default()
            }),
        );
        let mut selected = Vec::new();
        for rank in 0..2 {
            let context = s.select_context(rank, rank as i32).unwrap();
            let pool = s.select_memory_pool(context).unwrap();
            selected.push((context, pool));
        }
        let baseline = s.with_device(FakeHostDeviceV1::live_object_count);
        s.with_device_mut(|device| {
            device.refuse_after(fault, 1);
            device.refuse(FakeRefusalV1::ReleaseDeviceBuffer);
            device.refuse(FakeRefusalV1::ReleaseStream);
        });
        assert_eq!(s.acquire_serving_kv(LAYOUT), Err(BundleStatusV1::HostError));
        let issued = s.with_device(FakeHostDeviceV1::issued_object_count);
        assert!(s.with_device(FakeHostDeviceV1::live_object_count) >= baseline + 3);
        assert_eq!(s.serving_kv_pages(), None);
        assert!(s.resolve_serving_kv_pool(0).is_err());
        assert!(s.state.borrow().offload.is_none());
        assert!(s.state.borrow().device_memory.is_empty());
        assert_eq!(s.acquire_serving_kv(LAYOUT), Err(BundleStatusV1::HostError));
        s.with_device_mut(|device| device.allow(FakeRefusalV1::ReleaseDeviceBuffer));
        assert_eq!(s.acquire_serving_kv(LAYOUT), Err(BundleStatusV1::HostError));
        assert_eq!(s.with_device(FakeHostDeviceV1::issued_object_count), issued);
        assert_eq!(
            s.release_context(selected[1].0),
            Err(BundleStatusV1::HostError)
        );
        s.with_device_mut(|device| device.allow(FakeRefusalV1::ReleaseStream));
        s.state.borrow_mut().retry_acquisition_cleanup().unwrap();
        assert_eq!(s.with_device(FakeHostDeviceV1::live_object_count), baseline);
        assert_eq!(s.acquire_serving_kv(LAYOUT).unwrap(), 8);
        let views = [
            s.resolve_serving_kv_pool(0).unwrap(),
            s.resolve_serving_kv_pool(1).unwrap(),
        ];
        let issued = s.with_device(FakeHostDeviceV1::issued_object_count);
        assert_eq!(s.acquire_serving_kv(LAYOUT).unwrap(), 8);
        assert_eq!(
            views,
            [
                s.resolve_serving_kv_pool(0).unwrap(),
                s.resolve_serving_kv_pool(1).unwrap()
            ]
        );
        assert_eq!(s.with_device(FakeHostDeviceV1::issued_object_count), issued);
    }
}

struct UploadFixture {
    services: Services,
    uploads: Vec<RankArenaUploadV1>,
}

impl UploadFixture {
    fn new(host_views: bool) -> Self {
        let mut views = BTreeMap::new();
        if host_views {
            views.insert(
                b"host".to_vec(),
                CompiledImageViewV1::new(
                    64,
                    (0..2)
                        .map(|rank| {
                            (
                                rank,
                                CompiledViewSpanV1 {
                                    offset: 64,
                                    byte_length: 64,
                                },
                            )
                        })
                        .collect(),
                )
                .on_host(),
            );
        }
        let config = config()
            .with_compiled_image(
                b"image".to_vec(),
                vec![17; 256],
                CompiledImageDescriptionV1::new(
                    64,
                    (0..2)
                        .map(|rank| CompiledRankImageV1 {
                            offset: rank * 128,
                            byte_length: 128,
                        })
                        .collect(),
                    views,
                ),
            )
            .unwrap();
        let services = Services::new(FakeHostDeviceV1::new(), config);
        let mut uploads = Vec::new();
        for rank in 0..2 {
            let context = services.select_context(rank, rank as i32).unwrap();
            let pool = services.select_memory_pool(context).unwrap();
            let stream = services.create_stream(context, 0).unwrap();
            // One image handle shared by both ranks, acquired after dependencies below.
            uploads.push(RankArenaUploadV1 {
                context,
                pool,
                stream,
                image: BundleCompiledImageHandleV1::from_raw(0),
                rank,
            });
        }
        let image = services.open_compiled_image(b"image").unwrap();
        for upload in &mut uploads {
            upload.image = image;
        }
        Self { services, uploads }
    }

    fn acquire(&self, group: bool) -> HostServiceResultV1<Vec<BundleRankArenaHandleV1>> {
        if group {
            self.services.upload_rank_arenas(&self.uploads)
        } else {
            let rank = self.uploads[0];
            self.services
                .upload_rank_arena(rank.context, rank.pool, rank.stream, rank.image, rank.rank)
                .map(|arena| vec![arena])
        }
    }

    fn release(self) {
        self.services
            .release_compiled_image(self.uploads[0].image)
            .unwrap();
        for rank in self.uploads.into_iter().rev() {
            self.services.release_stream(rank.stream).unwrap();
            self.services.release_memory_pool(rank.pool).unwrap();
            self.services.release_context(rank.context).unwrap();
        }
        assert_eq!(self.services.live_resource_count(), 0);
        assert_eq!(
            self.services
                .with_device(FakeHostDeviceV1::live_object_count),
            0
        );
    }
}

#[test]
fn rank_arena_registration_precedes_all_buffer_construction() {
    for slot in [false, true] {
        let fixture = UploadFixture::new(true);
        let s = &fixture.services;
        let baseline = s.with_device(FakeHostDeviceV1::issued_object_count);
        let next = s.state.borrow().order.next;
        if slot {
            s.state.borrow_mut().arenas.refuse_registration(true);
        } else {
            s.state.borrow_mut().order.next = u64::MAX;
        }
        s.with_device_mut(|device| device.refuse(FakeRefusalV1::ReleaseDeviceBuffer));
        assert_eq!(
            fixture.acquire(false),
            Err(BundleStatusV1::ResourceExhausted)
        );
        assert_eq!(
            s.with_device(FakeHostDeviceV1::issued_object_count),
            baseline
        );
        s.state.borrow_mut().order.next = next;
        s.state.borrow_mut().arenas.refuse_registration(false);
        s.with_device_mut(|device| device.allow(FakeRefusalV1::ReleaseDeviceBuffer));
        let arenas = fixture.acquire(false).unwrap();
        for arena in arenas.into_iter().rev() {
            s.release_rank_arena(arena).unwrap();
        }
        fixture.release();
    }
}

#[test]
fn single_and_group_upload_failures_retain_unpublished_arenas() {
    // A second buffer failure leaves a device buffer without its host view;
    // a later rank failure leaves an earlier complete arena; a sync refusal
    // happens only after every arena has been prepared.
    for (group, host_views, fault, allowed) in [
        (false, true, FakeRefusalV1::AllocateDeviceBuffer, 1),
        (true, false, FakeRefusalV1::AllocateDeviceBuffer, 1),
        (false, false, FakeRefusalV1::SynchronizeStream, 0),
        (true, true, FakeRefusalV1::SynchronizeStream, 0),
    ] {
        let fixture = UploadFixture::new(host_views);
        let s = &fixture.services;
        let baseline = s.with_device(FakeHostDeviceV1::live_object_count);
        let orders = s.live_resource_count();
        s.with_device_mut(|device| {
            device.refuse_after(fault, allowed);
            device.refuse(FakeRefusalV1::ReleaseDeviceBuffer);
        });
        assert_eq!(fixture.acquire(group), Err(BundleStatusV1::HostError));
        assert!(s.with_device(FakeHostDeviceV1::live_object_count) > baseline);
        assert!(s.live_resource_count() > orders);
        for (rank, key) in s.state.borrow().unpublished_arenas.iter().enumerate() {
            assert_eq!(
                s.resolve_compiled_image_view(
                    fixture.uploads[0].image,
                    BundleRankArenaHandleV1::from_parts(key.generation, key.slot),
                    rank as u32,
                    COMPILED_IMAGE_ARENA_VIEW_V1,
                ),
                Err(BundleStatusV1::InvalidState)
            );
        }
        let issued = s.with_device(FakeHostDeviceV1::issued_object_count);
        assert_eq!(fixture.acquire(group), Err(BundleStatusV1::HostError));
        assert_eq!(s.with_device(FakeHostDeviceV1::issued_object_count), issued);
        assert_eq!(
            s.release_stream(fixture.uploads[0].stream),
            Err(BundleStatusV1::HostError)
        );
        s.with_device_mut(|device| device.allow(FakeRefusalV1::ReleaseDeviceBuffer));
        s.state.borrow_mut().retry_acquisition_cleanup().unwrap();
        assert_eq!(s.with_device(FakeHostDeviceV1::live_object_count), baseline);
        assert_eq!(s.live_resource_count(), orders);
        for arena in fixture.acquire(group).unwrap().into_iter().rev() {
            s.release_rank_arena(arena).unwrap();
        }
        fixture.release();
    }
}
