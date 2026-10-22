//! The host service contract a bundle sees through the safe V1 caller.

use std::collections::BTreeMap;
use std::rc::Rc;

use runtime_bundle_abi::{
    BundleAbiError, BundleDeviceBufferHandleV1, BundleKvRequestHandleV1, BundleStatusV1,
    BundleStreamHandleV1, CollectiveRequestV1, CustomAddressSpaceV1, CustomCompletionV1,
    CustomOperandV1, DeviceBufferSpanV1, HostServicesTableV1, HostServicesV1, LentLaunchV1,
    LentStreamV1, PhysicalKvLayoutV1, PhysicalKvRequestV1, export_custom_operation_v1,
};
use runtime_host::fake::{FAKE_COLLECTIVE_SUM_F32_V1, FakeHostDeviceV1, FakeRefusalV1};
use runtime_host::{
    COLLECTIVE_SUM_F32_V1, COMPILED_IMAGE_ARENA_VIEW_V1, CompiledImageDescriptionV1,
    CompiledImageViewV1, CompiledRankImageV1, CompiledViewSpanV1, DeclaredKvContractV1,
    HostConfigErrorV1, HostInstanceConfigV1, HostServicesV1Backend, HostTopologyV1,
    ServingKvPolicyV1, assert_collective_ordering_contract_v1, assert_host_transfer_contract_v1,
    kv_append_operation_v1, kv_page_view_id_v1,
};
use runtime_kv::{DeclaredPageLayoutV1, KvGeometry, KvPool, RankPlacementV1, StateLifetimeV1};

const RANKS: u32 = 2;
const COMPONENT_ROLE: &[u8] = b"rank-images";
const LAYOUT_ID: &[u8] = b"paged-kv-layout";
const PAGE_BYTES: u64 = 1024;
const PAGE_ALIGNMENT: u64 = 64;
const PAGE_TOKENS: u64 = 8;
const PAGE_CAPACITY: u64 = 8;

fn page_layout() -> DeclaredPageLayoutV1 {
    DeclaredPageLayoutV1::new(
        64,
        2,
        PAGE_TOKENS as u32,
        PAGE_ALIGNMENT,
        RANKS as u8,
        RankPlacementV1::Sharded,
        StateLifetimeV1::Request,
        64,
    )
    .expect("the test page layout is valid")
}

fn kv_request() -> PhysicalKvRequestV1 {
    PhysicalKvRequestV1 {
        flags: 0,
        layout: PhysicalKvLayoutV1 {
            flags: 0,
            rank_count: u64::from(RANKS),
            page_bytes: PAGE_BYTES,
            page_alignment: PAGE_ALIGNMENT,
            page_tokens: PAGE_TOKENS,
            state_lifetime: 1,
            layout_id: LAYOUT_ID.to_vec(),
        },
        page_capacity: PAGE_CAPACITY,
        request_capacity: 2,
    }
}

/// Two ranks of eight distinct bytes each, so a rank slice is identifiable.
fn component_bytes() -> Vec<u8> {
    (0..16_u8).collect()
}

/// A rank image per rank, plus one named four-byte view inside each.
///
/// The names are opaque to the host, so the count and the naming scheme are the
/// bundle's business alone.
fn compiled_image_description() -> CompiledImageDescriptionV1 {
    let ranks = (0..u64::from(RANKS))
        .map(|rank| CompiledRankImageV1 {
            offset: rank * 8,
            byte_length: 8,
        })
        .collect();
    let mut views = BTreeMap::new();
    for (name, offset) in [(&b"view-head"[..], 0_u64), (&b"view-tail"[..], 4)] {
        let spans = (0..RANKS)
            .map(|rank| {
                (
                    rank,
                    CompiledViewSpanV1 {
                        offset,
                        byte_length: 4,
                    },
                )
            })
            .collect();
        views.insert(name.to_vec(), CompiledImageViewV1::new(4, spans));
    }
    CompiledImageDescriptionV1::new(8, ranks, views)
}

fn table() -> (
    HostServicesTableV1,
    Rc<HostServicesV1Backend<FakeHostDeviceV1>>,
) {
    let config = HostInstanceConfigV1::new(
        HostTopologyV1::new(vec![0, 1]).expect("the test topology is valid"),
    )
    .with_compiled_image(
        COMPONENT_ROLE,
        component_bytes(),
        compiled_image_description(),
    )
    .expect("the test component role is unique")
    .with_kv_contract(DeclaredKvContractV1::new(LAYOUT_ID.to_vec(), page_layout()));
    let backend = Rc::new(HostServicesV1Backend::new(FakeHostDeviceV1::new(), config));
    (HostServicesTableV1::new(Rc::clone(&backend)), backend)
}

fn serving_table(
    maximum_pages: Option<u64>,
) -> (
    HostServicesTableV1,
    Rc<HostServicesV1Backend<FakeHostDeviceV1>>,
) {
    serving_table_on(FakeHostDeviceV1::new(), 1_000_000, maximum_pages)
}

fn serving_table_on(
    device: FakeHostDeviceV1,
    utilization_millionths: u32,
    maximum_pages: Option<u64>,
) -> (
    HostServicesTableV1,
    Rc<HostServicesV1Backend<FakeHostDeviceV1>>,
) {
    let config = HostInstanceConfigV1::new(HostTopologyV1::new(vec![0, 1]).unwrap())
        .with_kv_contract(DeclaredKvContractV1::new(LAYOUT_ID.to_vec(), page_layout()))
        .with_serving_kv(ServingKvPolicyV1 {
            sampling_vocabulary: None,
            maximum_context_tokens: None,
            state_reserved_bytes: 0,
            utilization_millionths,
            maximum_pages,
        });
    let backend = Rc::new(HostServicesV1Backend::new(device, config));
    (HostServicesTableV1::new(Rc::clone(&backend)), backend)
}

fn serving_ranks(host: &HostServicesV1) -> Vec<BundleStreamHandleV1> {
    (0..RANKS)
        .map(|rank| {
            let context = host.select_context(rank, rank as i32).unwrap();
            host.select_memory_pool(context).unwrap();
            host.create_stream(context, 0).unwrap()
        })
        .collect()
}

fn status(error: BundleAbiError) -> BundleStatusV1 {
    match error {
        BundleAbiError::Status(status) => status,
        other => panic!("expected a host status, got {other}"),
    }
}

#[test]
fn graph_capture_keeps_host_transfers_and_synchronization_out_of_capture() {
    let (table, backend) = table();
    let host = table.caller();
    let context = host.select_context(0, 0).unwrap();
    let pool = host.select_memory_pool(context).unwrap();
    let stream = host.create_stream(context, 0).unwrap();
    let peer_stream = host.create_stream(context, 1).unwrap();
    let event = host.create_event(context).unwrap();
    let buffer = host.allocate_device_buffer(pool, stream, 8, 64).unwrap();
    let span = DeviceBufferSpanV1::new(buffer, 0, 8);

    host.begin_graph_capture(stream).unwrap();
    assert_eq!(
        status(host.begin_graph_capture(peer_stream).unwrap_err()),
        BundleStatusV1::InvalidState
    );
    host.record_event(event, stream).unwrap();
    host.wait_event(peer_stream, event).unwrap();
    assert_eq!(
        status(host.synchronize_stream(peer_stream).unwrap_err()),
        BundleStatusV1::InvalidState
    );
    assert_eq!(
        status(host.synchronize_event(event).unwrap_err()),
        BundleStatusV1::InvalidState
    );
    assert_eq!(
        status(host.copy_host_to_device(stream, span, &[0; 8]).unwrap_err()),
        BundleStatusV1::InvalidState
    );
    assert_eq!(
        status(
            host.copy_device_to_host(stream, &mut [0; 8], span)
                .unwrap_err()
        ),
        BundleStatusV1::InvalidState
    );
    assert_eq!(
        status(host.zero_device_buffer(stream, span).unwrap_err()),
        BundleStatusV1::InvalidState
    );
    assert_eq!(
        status(
            host.allocate_device_buffer(pool, stream, 8, 64)
                .unwrap_err()
        ),
        BundleStatusV1::InvalidState
    );
    assert_eq!(
        status(host.release_device_buffer(buffer).unwrap_err()),
        BundleStatusV1::InvalidState
    );
    assert_eq!(
        status(host.end_graph_capture(peer_stream).unwrap_err()),
        BundleStatusV1::InvalidState
    );

    let graph = host.end_graph_capture(stream).unwrap();
    host.launch_graph(peer_stream, graph).unwrap();
    host.release_graph(graph).unwrap();
    assert_eq!(
        status(host.launch_graph(stream, graph).unwrap_err()),
        BundleStatusV1::StaleHandle
    );
    host.synchronize_stream(stream).unwrap();
    host.release_device_buffer(buffer).unwrap();
    host.release_event(event).unwrap();
    host.release_stream(peer_stream).unwrap();
    host.release_stream(stream).unwrap();
    host.release_memory_pool(pool).unwrap();
    host.release_context(context).unwrap();
    assert_eq!(backend.live_resource_count(), 0);
}

/// A capture that ended without a graph (its instantiation failed) leaves
/// the stream free to synchronize, capture again and be released. A refusal
/// before the capture ended leaves it capturing until it is ended.
#[test]
fn a_failed_capture_end_leaves_the_stream_in_the_state_the_device_reports() {
    for refusal in [
        FakeRefusalV1::InstantiateGraph,
        FakeRefusalV1::EndGraphCapture,
    ] {
        let (table, backend) = table();
        let host = table.caller();
        let context = host.select_context(0, 0).unwrap();
        let stream = host.create_stream(context, 0).unwrap();
        host.begin_graph_capture(stream).unwrap();
        backend.with_device_mut(|device| device.refuse_after(refusal, 0));
        assert_eq!(
            status(host.end_graph_capture(stream).unwrap_err()),
            BundleStatusV1::HostError
        );
        if refusal == FakeRefusalV1::EndGraphCapture {
            assert_eq!(
                status(host.synchronize_stream(stream).unwrap_err()),
                BundleStatusV1::InvalidState,
                "{refusal:?}"
            );
            let graph = host.end_graph_capture(stream).unwrap();
            host.release_graph(graph).unwrap();
        } else {
            assert_eq!(
                status(host.end_graph_capture(stream).unwrap_err()),
                BundleStatusV1::InvalidState
            );
        }
        host.synchronize_stream(stream).unwrap();
        host.begin_graph_capture(stream).unwrap();
        let graph = host.end_graph_capture(stream).unwrap();
        host.release_graph(graph).unwrap();
        host.release_stream(stream).unwrap();
        host.release_context(context).unwrap();
        assert_eq!(backend.live_resource_count(), 0, "{refusal:?}");
    }
}

#[test]
fn a_bundle_round_trips_bytes_through_its_own_handles() {
    let (table, backend) = table();
    let host = table.caller();

    let context = host.select_context(0, 0).unwrap();
    let pool = host.select_memory_pool(context).unwrap();
    let stream = host.create_stream(context, 0).unwrap();
    let buffer = host.allocate_device_buffer(pool, stream, 8, 64).unwrap();
    let span = DeviceBufferSpanV1::new(buffer, 0, 8);
    host.copy_host_to_device(stream, span, &[1, 2, 3, 4, 5, 6, 7, 8])
        .unwrap();
    let mut read = [0_u8; 8];
    host.copy_device_to_host(stream, &mut read, span).unwrap();

    assert_eq!(read, [1, 2, 3, 4, 5, 6, 7, 8]);
    assert_eq!(backend.live_resource_count(), 4);

    host.release_device_buffer(buffer).unwrap();
    host.release_stream(stream).unwrap();
    host.release_memory_pool(pool).unwrap();
    host.release_context(context).unwrap();

    assert_eq!(backend.live_resource_count(), 0);
    assert_eq!(backend.with_device(FakeHostDeviceV1::live_object_count), 0);
}

#[test]
fn releasing_out_of_order_is_refused_and_changes_nothing() {
    let (table, backend) = table();
    let host = table.caller();
    let context = host.select_context(0, 0).unwrap();
    let pool = host.select_memory_pool(context).unwrap();

    assert_eq!(
        host.release_context(context).map_err(status),
        Err(BundleStatusV1::InvalidState)
    );
    assert_eq!(backend.live_resource_count(), 2);

    host.release_memory_pool(pool).unwrap();
    host.release_context(context).unwrap();
    assert_eq!(backend.live_resource_count(), 0);
}

#[test]
fn a_released_handle_is_rejected_for_the_life_of_its_instance() {
    let (first_table, _first) = table();
    let first = first_table.caller();

    let context = first.select_context(1, 1).unwrap();
    first.release_context(context).unwrap();

    assert_eq!(
        first.release_context(context).map_err(status),
        Err(BundleStatusV1::StaleHandle)
    );
    assert_eq!(
        first.select_memory_pool(context).map_err(status),
        Err(BundleStatusV1::StaleHandle)
    );
}

#[test]
fn a_foreign_handle_is_rejected_while_the_slot_it_collides_with_is_live() {
    let (first_table, first_backend) = table();
    let (second_table, second_backend) = table();
    let first = first_table.caller();
    let second = second_table.caller();

    let context = first.select_context(1, 1).unwrap();
    let foreign = second.select_context(1, 1).unwrap();

    // The two instances issue the same slot index, and both slots stay live for
    // this whole test, so nothing but the generation can separate the handles.
    assert_eq!(foreign.slot(), context.slot());
    assert_ne!(foreign.raw, context.raw);
    assert_eq!(first_backend.live_resource_count(), 1);
    assert_eq!(second_backend.live_resource_count(), 1);

    assert_eq!(
        first.select_memory_pool(foreign).map_err(status),
        Err(BundleStatusV1::StaleHandle)
    );
    assert_eq!(
        second.select_memory_pool(context).map_err(status),
        Err(BundleStatusV1::StaleHandle)
    );
    assert_eq!(
        first.release_context(foreign).map_err(status),
        Err(BundleStatusV1::StaleHandle)
    );

    // Each instance still owns and can release its own live context.
    first.release_context(context).unwrap();
    second.release_context(foreign).unwrap();
    assert_eq!(first_backend.live_resource_count(), 0);
    assert_eq!(second_backend.live_resource_count(), 0);
}

#[test]
fn a_rank_arena_carries_its_own_declared_rank_image_and_owns_its_allocation() {
    let (table, _backend) = table();
    let host = table.caller();
    let context = host.select_context(1, 1).unwrap();
    let pool = host.select_memory_pool(context).unwrap();
    let stream = host.create_stream(context, 0).unwrap();
    let image = host.open_compiled_image(COMPONENT_ROLE).unwrap();
    let arena = host
        .upload_rank_arena(context, pool, stream, image, 1)
        .unwrap();

    let whole = host
        .resolve_compiled_image_view(image, arena, 1, COMPILED_IMAGE_ARENA_VIEW_V1)
        .unwrap();
    assert_eq!(whole.rank, 1);
    assert_eq!(whole.buffer.length, 8);
    assert_eq!(whole.alignment, 8);
    let mut read = [0_u8; 8];
    host.copy_device_to_host(
        stream,
        &mut read,
        DeviceBufferSpanV1::new(whole.buffer.buffer, 0, 8),
    )
    .unwrap();
    assert_eq!(read, [8, 9, 10, 11, 12, 13, 14, 15]);

    // The arena owns this allocation, so the bundle cannot release it directly.
    assert_eq!(
        host.release_device_buffer(whole.buffer.buffer)
            .map_err(status),
        Err(BundleStatusV1::InvalidState)
    );

    host.release_rank_arena(arena).unwrap();
    host.release_compiled_image(image).unwrap();
    host.release_stream(stream).unwrap();
    host.release_memory_pool(pool).unwrap();
    host.release_context(context).unwrap();
}

#[test]
fn a_named_view_resolves_to_the_bounded_span_the_bundle_declared() {
    let (table, _backend) = table();
    let host = table.caller();
    let context = host.select_context(0, 0).unwrap();
    let pool = host.select_memory_pool(context).unwrap();
    let stream = host.create_stream(context, 0).unwrap();
    let image = host.open_compiled_image(COMPONENT_ROLE).unwrap();
    let arena = host
        .upload_rank_arena(context, pool, stream, image, 0)
        .unwrap();

    let head = host
        .resolve_compiled_image_view(image, arena, 0, b"view-head")
        .unwrap();
    let tail = host
        .resolve_compiled_image_view(image, arena, 0, b"view-tail")
        .unwrap();
    assert_eq!((head.buffer.offset, head.buffer.length), (0, 4));
    assert_eq!((tail.buffer.offset, tail.buffer.length), (4, 4));
    assert_eq!(head.alignment, 4);
    let mut read = [0_u8; 4];
    host.copy_device_to_host(stream, &mut read, tail.buffer)
        .unwrap();
    assert_eq!(read, [4, 5, 6, 7]);

    // Only declared names resolve, and the host never invents one.
    assert_eq!(
        host.resolve_compiled_image_view(image, arena, 0, COMPONENT_ROLE)
            .map_err(status),
        Err(BundleStatusV1::NotFound)
    );

    host.release_rank_arena(arena).unwrap();
    host.release_compiled_image(image).unwrap();
    host.release_stream(stream).unwrap();
    host.release_memory_pool(pool).unwrap();
    host.release_context(context).unwrap();
}

#[test]
fn a_component_with_no_declared_division_is_never_split_by_guesswork() {
    let config = HostInstanceConfigV1::new(
        HostTopologyV1::new(vec![0, 1]).expect("the test topology is valid"),
    )
    .with_component(COMPONENT_ROLE, component_bytes())
    .expect("the test component role is unique");
    let backend = Rc::new(HostServicesV1Backend::new(FakeHostDeviceV1::new(), config));
    let table = HostServicesTableV1::new(Rc::clone(&backend));
    let host = table.caller();
    let context = host.select_context(0, 0).unwrap();
    let pool = host.select_memory_pool(context).unwrap();
    let stream = host.create_stream(context, 0).unwrap();
    let image = host.open_compiled_image(COMPONENT_ROLE).unwrap();

    assert_eq!(
        host.upload_rank_arena(context, pool, stream, image, 0)
            .map_err(status),
        Err(BundleStatusV1::Unsupported)
    );

    host.release_compiled_image(image).unwrap();
    host.release_stream(stream).unwrap();
    host.release_memory_pool(pool).unwrap();
    host.release_context(context).unwrap();
}

#[test]
fn one_stage_of_one_rank_always_selects_the_same_stream() {
    let (table, backend) = table();
    let host = table.caller();
    let context = host.select_context(0, 0).unwrap();

    let first = host.select_stage_stream(context, 0, b"stage-7", 2).unwrap();
    let again = host.select_stage_stream(context, 0, b"stage-7", 2).unwrap();
    let other = host.select_stage_stream(context, 0, b"stage-8", 2).unwrap();

    assert_eq!(first, again);
    assert_ne!(first, other);
    assert_eq!(backend.live_resource_count(), 3);
    assert_eq!(
        host.select_stage_stream(context, 1, b"stage-7", 2)
            .map_err(status),
        Err(BundleStatusV1::InvalidArgument)
    );
}

#[test]
fn the_kv_service_honours_only_the_declared_layout() {
    let (table, _backend) = table();
    let host = table.caller();
    let mut wrong = kv_request();
    wrong.layout.page_tokens = PAGE_TOKENS + 1;

    assert_eq!(
        host.allocate_physical_kv(&wrong).map_err(status),
        Err(BundleStatusV1::InvalidArgument)
    );

    let mut renamed = kv_request();
    renamed.layout.layout_id = b"another-layout".to_vec();
    assert_eq!(
        host.allocate_physical_kv(&renamed).map_err(status),
        Err(BundleStatusV1::InvalidArgument)
    );
}

#[test]
fn serving_pool_views_are_stable_and_page_ids_address_each_rank() {
    let (table, backend) = serving_table(Some(4));
    let host = table.caller();
    let streams = serving_ranks(host);
    assert_eq!(
        host.acquire_serving_kv(b"wrong-layout").map_err(status),
        Err(BundleStatusV1::InvalidArgument)
    );
    assert_eq!(host.acquire_serving_kv(LAYOUT_ID).unwrap(), 4);
    assert_eq!(host.acquire_serving_kv(LAYOUT_ID).unwrap(), 4);
    assert_eq!(backend.serving_kv_pages(), Some(4));
    let before_capture = host.resolve_serving_kv_pool(0).unwrap();
    host.begin_graph_capture(streams[0]).unwrap();
    assert_eq!(host.resolve_serving_kv_pool(0).unwrap(), before_capture);
    let graph = host.end_graph_capture(streams[0]).unwrap();
    assert_eq!(host.resolve_serving_kv_pool(0).unwrap(), before_capture);
    host.release_graph(graph).unwrap();
    for (rank, stream) in streams.into_iter().enumerate() {
        let view = host.resolve_serving_kv_pool(rank as u32).unwrap();
        assert_eq!(view.length, 4 * PAGE_BYTES);
        let page = DeviceBufferSpanV1::new(view.buffer, 2 * PAGE_BYTES, PAGE_BYTES);
        host.copy_host_to_device(stream, page, &vec![rank as u8 + 1; PAGE_BYTES as usize])
            .unwrap();
        let mut read = vec![0; PAGE_BYTES as usize];
        host.copy_device_to_host(stream, &mut read, page).unwrap();
        assert_eq!(read, vec![rank as u8 + 1; PAGE_BYTES as usize]);
        assert_eq!(host.resolve_serving_kv_pool(rank as u32).unwrap(), view);
    }
    assert_eq!(
        host.resolve_serving_kv_pool(RANKS).map_err(status),
        Err(BundleStatusV1::InvalidArgument)
    );
}

#[test]
fn serving_pool_sizes_from_the_tightest_rank_after_bundle_allocations() {
    let (table, backend) = serving_table(None);
    let host = table.caller();
    let streams = serving_ranks(host);
    let rank_one = host.select_context(1, 1).unwrap();
    let pool_one = host.select_memory_pool(rank_one).unwrap();
    host.allocate_device_buffer(pool_one, streams[1], 4 * 1024 * 1024, 64)
        .unwrap();
    // Full utilization takes all the tightest rank's free memory past the
    // 1 GiB reserve, as the fraction 1.0 it replaces did.
    let available = 12_u64 * 1024 * 1024;
    let expected = available / PAGE_BYTES;
    assert_eq!(host.acquire_serving_kv(LAYOUT_ID).unwrap(), expected);
    assert_eq!(backend.serving_kv_pages(), Some(expected));
    assert_eq!(
        host.resolve_serving_kv_pool(0).unwrap().length,
        expected * PAGE_BYTES
    );
    assert_eq!(
        host.resolve_serving_kv_pool(1).unwrap().length,
        expected * PAGE_BYTES
    );
}

/// Pages the pool holds on a 1,088 MiB fake device of which rank 1's own
/// allocations take 4 MiB, another process `foreign` bytes on every device.
fn utilization_pages(utilization_millionths: u32, foreign: u64) -> u64 {
    let mut device = FakeHostDeviceV1::new();
    device.set_device_memory_bytes(1088 * MIB);
    device.set_foreign_device_bytes(foreign);
    let (table, _backend) = serving_table_on(device, utilization_millionths, None);
    let host = table.caller();
    let streams = serving_ranks(host);
    let rank_one = host.select_context(1, 1).unwrap();
    let pool_one = host.select_memory_pool(rank_one).unwrap();
    host.allocate_device_buffer(pool_one, streams[1], 4 * MIB, 64)
        .unwrap();
    host.acquire_serving_kv(LAYOUT_ID).unwrap()
}

const MIB: u64 = 1024 * 1024;

#[test]
fn serving_pool_below_full_utilization_caps_own_allocations_at_its_share() {
    // At 1.0 rank 1 leaves 1,084 MiB free, 60 MiB past the reserve.
    assert_eq!(utilization_pages(1_000_000, 0), 60 * MIB / PAGE_BYTES);
    // At 0.97 its own 4 MiB, the reserve and the pool fit 0.97 of 1,088 MiB.
    let share = 1088 * MIB * 97 / 100;
    let capped = (share - 4 * MIB - 1024 * MIB) / PAGE_BYTES;
    assert_eq!(utilization_pages(970_000, 0), capped);
    // Memory another process holds does not count as this process's: with
    // 30 MiB of it the share still fits in free memory and is unchanged.
    assert_eq!(utilization_pages(970_000, 30 * MIB), capped);
    // Only once it leaves less free memory than the share does it shrink
    // the pool, and at 1.0 it always does.
    assert_eq!(
        utilization_pages(970_000, 40 * MIB),
        (1044 * MIB - 1024 * MIB) / PAGE_BYTES
    );
    assert_eq!(
        utilization_pages(1_000_000, 30 * MIB),
        (1054 * MIB - 1024 * MIB) / PAGE_BYTES
    );
}

#[test]
fn serving_pool_refuses_zero_capacity_and_insufficient_memory() {
    let (table, _backend) = serving_table(Some(0));
    let host = table.caller();
    serving_ranks(host);
    assert_eq!(
        host.acquire_serving_kv(LAYOUT_ID).map_err(status),
        Err(BundleStatusV1::ResourceExhausted)
    );

    let (table, _backend) = serving_table(None);
    let host = table.caller();
    let context = host.select_context(0, 0).unwrap();
    let pool = host.select_memory_pool(context).unwrap();
    let stream = host.create_stream(context, 0).unwrap();
    let _buffer = host
        .allocate_device_buffer(pool, stream, 17 * 1024 * 1024, 64)
        .unwrap();
    let other = host.select_context(1, 1).unwrap();
    host.select_memory_pool(other).unwrap();
    assert_eq!(
        host.acquire_serving_kv(LAYOUT_ID).map_err(status),
        Err(BundleStatusV1::ResourceExhausted)
    );
}

#[test]
fn serving_pool_skips_rank_synchronization_without_page_work() {
    let (table, backend) = serving_table(Some(4));
    let host = table.caller();
    serving_ranks(host);
    host.acquire_serving_kv(LAYOUT_ID).unwrap();
    let mut logical = KvPool::new(KvGeometry::new(8, 4).unwrap(), 2).unwrap();
    let sequence = logical.create_sequence(16).unwrap();
    logical.append(sequence, 3).unwrap();
    let prepared = logical.prepare_append_batch(&[(sequence, 1)]).unwrap();
    assert!(prepared.appends()[0].copy().is_none());
    let before = backend.with_device(FakeHostDeviceV1::synchronizations);
    backend.with_device_mut(|device| device.refuse(FakeRefusalV1::SynchronizeStream));
    backend.prepare_serving_batch(&prepared).unwrap();
    assert_eq!(
        backend.with_device(FakeHostDeviceV1::synchronizations),
        before
    );
}

/// The copy covers the whole page, because a bundle's page layout need not
/// keep one token's rows together.
#[test]
fn serving_pool_copies_whole_shared_pages_before_execute() {
    let (table, backend) = serving_table(Some(4));
    let host = table.caller();
    let streams = serving_ranks(host);
    host.acquire_serving_kv(LAYOUT_ID).unwrap();
    let mut logical = KvPool::new(KvGeometry::new(8, 4).unwrap(), 2).unwrap();
    let source = logical.create_sequence(16).unwrap();
    logical.append(source, 3).unwrap();
    let child = logical.fork(source, 16).unwrap();
    let prepared = logical.prepare_append_batch(&[(child, 1)]).unwrap();
    let copy = prepared.appends()[0].copy().unwrap();
    for (rank, stream) in streams.iter().copied().enumerate() {
        let view = host.resolve_serving_kv_pool(rank as u32).unwrap();
        for segment in 0..2_u64 {
            let source_offset =
                copy.source().index() as u64 * PAGE_BYTES + segment * PAGE_BYTES / 2;
            let span = DeviceBufferSpanV1::new(view.buffer, source_offset, 3 * 64);
            host.copy_host_to_device(stream, span, &[rank as u8 + 4; 3 * 64])
                .unwrap();
        }
        let last = DeviceBufferSpanV1::new(
            view.buffer,
            (copy.source().index() as u64 + 1) * PAGE_BYTES - 1,
            1,
        );
        host.copy_host_to_device(stream, last, &[rank as u8 + 9])
            .unwrap();
    }
    let before = backend.with_device(FakeHostDeviceV1::synchronizations);
    backend.prepare_serving_batch(&prepared).unwrap();
    assert_eq!(
        backend.with_device(FakeHostDeviceV1::synchronizations) - before,
        RANKS as usize
    );
    for (rank, stream) in streams.iter().copied().enumerate() {
        let view = host.resolve_serving_kv_pool(rank as u32).unwrap();
        for segment in 0..2_u64 {
            let destination =
                copy.destination().index() as u64 * PAGE_BYTES + segment * PAGE_BYTES / 2;
            let mut bytes = vec![0; 3 * 64];
            host.copy_device_to_host(
                stream,
                &mut bytes,
                DeviceBufferSpanV1::new(view.buffer, destination, 3 * 64),
            )
            .unwrap();
            assert_eq!(bytes, vec![rank as u8 + 4; 3 * 64]);
        }
        let mut last = [0];
        host.copy_device_to_host(
            stream,
            &mut last,
            DeviceBufferSpanV1::new(
                view.buffer,
                (copy.destination().index() as u64 + 1) * PAGE_BYTES - 1,
                1,
            ),
        )
        .unwrap();
        assert_eq!(last, [rank as u8 + 9]);
    }
}

#[test]
fn a_kv_append_prepares_commits_and_releases_in_reverse_order() {
    let (table, backend) = table();
    let host = table.caller();
    let kv = host.allocate_physical_kv(&kv_request()).unwrap();
    let request = host.create_kv_request(kv, 16).unwrap();

    let rank_pool = host.resolve_kv_view(request, 1, b"").unwrap();
    assert_eq!(rank_pool.length, PAGE_BYTES * PAGE_CAPACITY);
    let page = host
        .resolve_kv_view(request, 0, &kv_page_view_id_v1(2))
        .unwrap();
    assert_eq!(page.offset, PAGE_BYTES * 2);
    assert_eq!(page.length, PAGE_BYTES);
    assert_eq!(
        host.resolve_kv_view(request, 0, &kv_page_view_id_v1(PAGE_CAPACITY))
            .map_err(status),
        Err(BundleStatusV1::NotFound)
    );

    let transaction = host
        .prepare_kv(request, &kv_append_operation_v1(8))
        .unwrap();
    assert_eq!(
        host.release_kv_request(request).map_err(status),
        Err(BundleStatusV1::InvalidState)
    );
    host.commit_kv(transaction).unwrap();
    host.release_kv_request(request).unwrap();
    let reused = host.create_kv_request(kv, 16).unwrap();
    host.release_kv_request(reused).unwrap();
    host.release_physical_kv(kv).unwrap();

    assert_eq!(backend.live_resource_count(), 0);
    assert_eq!(backend.with_device(FakeHostDeviceV1::live_object_count), 0);
}

#[test]
fn a_cancelled_kv_append_returns_every_page_it_reserved() {
    let (table, _backend) = table();
    let host = table.caller();
    let kv = host.allocate_physical_kv(&kv_request()).unwrap();
    let request = host.create_kv_request(kv, 16).unwrap();

    for _ in 0..4 {
        let transaction = host
            .prepare_kv(request, &kv_append_operation_v1(8))
            .unwrap();
        host.cancel_kv(transaction).unwrap();
    }

    // Every cancelled append released its page, so a real append still fits.
    let transaction = host
        .prepare_kv(request, &kv_append_operation_v1(8))
        .unwrap();
    host.commit_kv(transaction).unwrap();
    host.release_kv_request(request).unwrap();
    host.release_physical_kv(kv).unwrap();
}

#[test]
fn a_collective_reduces_every_rank_buffer_it_was_given() {
    let (table, _backend) = table();
    let host = table.caller();
    let context = host.select_context(0, 0).unwrap();
    let pool = host.select_memory_pool(context).unwrap();
    let stream = host.create_stream(context, 0).unwrap();
    let mut buffers = Vec::new();
    for rank in 0..RANKS {
        let buffer = host.allocate_device_buffer(pool, stream, 4, 64).unwrap();
        let span = DeviceBufferSpanV1::new(buffer, 0, 4);
        host.copy_host_to_device(stream, span, &(rank as f32 + 1.0).to_le_bytes())
            .unwrap();
        buffers.push(buffer);
    }
    let collective = host
        .create_collective(&CollectiveRequestV1 {
            flags: 0,
            rank_count: u64::from(RANKS),
            semantic_contract: FAKE_COLLECTIVE_SUM_F32_V1.to_vec(),
        })
        .unwrap();

    host.execute_collective(
        collective,
        stream,
        FAKE_COLLECTIVE_SUM_F32_V1,
        &buffers
            .iter()
            .map(|buffer| {
                runtime_bundle_abi::KernelArgumentV1::DeviceBuffer(DeviceBufferSpanV1::new(
                    *buffer, 0, 4,
                ))
            })
            .collect::<Vec<_>>(),
    )
    .unwrap();

    for buffer in &buffers {
        let mut read = [0_u8; 4];
        host.copy_device_to_host(stream, &mut read, DeviceBufferSpanV1::new(*buffer, 0, 4))
            .unwrap();
        assert_eq!(f32::from_le_bytes(read), 3.0);
    }

    host.release_collective(collective).unwrap();
    for buffer in buffers.into_iter().rev() {
        host.release_device_buffer(buffer).unwrap();
    }
    host.release_stream(stream).unwrap();
    host.release_memory_pool(pool).unwrap();
    host.release_context(context).unwrap();
}

/// What one custom operation saw of what the host lent it.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
struct CustomOperationStateV1 {
    calls: u32,
    address_space: Option<CustomAddressSpaceV1>,
    stream_is_null: bool,
    lengths: Vec<u64>,
    distinct_addresses: bool,
    /// The code this operation answers with, so a failing one can be observed.
    code: i32,
}

export_custom_operation_v1!(
    fixture_custom_operation,
    CustomOperationStateV1,
    |state: &mut CustomOperationStateV1, launch: LentLaunchV1<'_>| {
        state.calls += 1;
        state.address_space = Some(launch.address_space);
        state.stream_is_null = launch.stream == LentStreamV1::null();
        state.lengths = launch.ranges.iter().map(|range| range.length).collect();
        state.distinct_addresses = launch
            .ranges
            .windows(2)
            .all(|pair| pair[0].address != pair[1].address);
        state.code
    },
);
#[test]
fn a_custom_operation_is_lent_the_ranges_the_host_resolved() {
    let (table, backend) = table();
    let host = table.caller();
    let context = host.select_context(0, 0).unwrap();
    let pool = host.select_memory_pool(context).unwrap();
    let stream = host.create_stream(context, 0).unwrap();
    let input = host.allocate_device_buffer(pool, stream, 64, 64).unwrap();
    let output = host.allocate_device_buffer(pool, stream, 32, 64).unwrap();
    let mut state = CustomOperationStateV1::default();

    {
        let operation = fixture_custom_operation(&mut state);
        host.run_custom_operation(
            context,
            stream,
            CustomCompletionV1::MayEnqueue,
            b"fixture-operation-v1",
            &[
                CustomOperandV1::read(DeviceBufferSpanV1::new(input, 16, 32)),
                CustomOperandV1::write(DeviceBufferSpanV1::new(output, 0, 32)),
            ],
            &operation,
        )
        .expect("the host runs an operation whose operands all check out");
    }

    assert_eq!(state.calls, 1);
    // This host has no driver, and it says so rather than lending a stream or
    // an address a device kernel could use.
    assert_eq!(state.address_space, Some(CustomAddressSpaceV1::Host));
    assert!(state.stream_is_null);
    assert_eq!(state.lengths, vec![32, 32]);
    assert!(state.distinct_addresses);
    // One operation that may enqueue is waited for twice: once before it is
    // lent anything, and once for what it left behind.
    assert_eq!(backend.with_device(FakeHostDeviceV1::synchronizations), 2);

    host.release_device_buffer(output).unwrap();
    host.release_device_buffer(input).unwrap();
    host.release_stream(stream).unwrap();
    host.release_memory_pool(pool).unwrap();
    host.release_context(context).unwrap();
}

#[test]
fn a_synchronous_operation_is_waited_for_once_and_only_before_its_callback() {
    let (table, backend) = table();
    let host = table.caller();
    let context = host.select_context(0, 0).unwrap();
    let pool = host.select_memory_pool(context).unwrap();
    let stream = host.create_stream(context, 0).unwrap();
    let mut state = CustomOperationStateV1::default();

    {
        let operation = fixture_custom_operation(&mut state);
        host.run_custom_operation(
            context,
            stream,
            CustomCompletionV1::Synchronous,
            b"fixture-operation-v1",
            &[],
            &operation,
        )
        .expect("a synchronous operation runs");
    }

    // An operation that finished what it started is not waited for afterwards,
    // so no wait stands between its result and its caller, and there is no
    // split state where the operation succeeded and the call reports failure.
    assert_eq!(state.calls, 1);
    assert_eq!(backend.with_device(FakeHostDeviceV1::synchronizations), 1);

    host.release_stream(stream).unwrap();
    host.release_memory_pool(pool).unwrap();
    host.release_context(context).unwrap();
}

#[test]
fn a_wait_the_host_cannot_complete_invokes_no_callback_and_stays_retryable() {
    let (table, backend) = table();
    let host = table.caller();
    let context = host.select_context(0, 0).unwrap();
    let pool = host.select_memory_pool(context).unwrap();
    let stream = host.create_stream(context, 0).unwrap();
    let mut state = CustomOperationStateV1::default();

    backend.with_device_mut(|device| device.refuse(FakeRefusalV1::SynchronizeStream));
    let refused = {
        let operation = fixture_custom_operation(&mut state);
        host.run_custom_operation(
            context,
            stream,
            CustomCompletionV1::MayEnqueue,
            b"fixture-operation-v1",
            &[],
            &operation,
        )
    };
    assert_eq!(refused.map_err(status), Err(BundleStatusV1::HostError));
    assert_eq!(
        state.calls, 0,
        "a wait that failed before the callback invokes nothing"
    );
    assert!(
        backend
            .with_device(|device| device.custom_operations().to_vec())
            .is_empty()
    );

    // Nothing changed, so the same call is the retry.
    backend.with_device_mut(|device| device.allow(FakeRefusalV1::SynchronizeStream));
    {
        let operation = fixture_custom_operation(&mut state);
        host.run_custom_operation(
            context,
            stream,
            CustomCompletionV1::MayEnqueue,
            b"fixture-operation-v1",
            &[],
            &operation,
        )
        .expect("the retry runs once the wait can complete");
    }
    assert_eq!(state.calls, 1);

    host.release_stream(stream).unwrap();
    host.release_memory_pool(pool).unwrap();
    host.release_context(context).unwrap();
}

#[test]
fn a_failed_wait_after_an_operation_leaves_the_next_one_retryable() {
    let (table, backend) = table();
    let host = table.caller();
    let context = host.select_context(0, 0).unwrap();
    let pool = host.select_memory_pool(context).unwrap();
    let stream = host.create_stream(context, 0).unwrap();
    let mut state = CustomOperationStateV1::default();

    // The wait before the callback succeeds and the one after it fails, which
    // is the shape of a device that raised while the operation's work ran.
    backend.with_device_mut(|device| device.refuse_after(FakeRefusalV1::SynchronizeStream, 1));
    let failed = {
        let operation = fixture_custom_operation(&mut state);
        host.run_custom_operation(
            context,
            stream,
            CustomCompletionV1::MayEnqueue,
            b"fixture-operation-v1",
            &[],
            &operation,
        )
    };
    assert_eq!(failed.map_err(status), Err(BundleStatusV1::HostError));
    assert_eq!(
        state.calls, 1,
        "the callback did run before the wait failed"
    );

    // Cleanup after that failure is still reachable: its own wait comes first,
    // succeeds, and its callback runs.
    let mut cleanup = CustomOperationStateV1::default();
    {
        let operation = fixture_custom_operation(&mut cleanup);
        host.run_custom_operation(
            context,
            stream,
            CustomCompletionV1::Synchronous,
            b"fixture-cleanup-v1",
            &[],
            &operation,
        )
        .expect("cleanup runs after a failed wait");
    }
    assert_eq!(cleanup.calls, 1);
    let operations = backend.with_device(|device| {
        device
            .custom_operations()
            .iter()
            .map(|(identifier, _)| identifier.clone())
            .collect::<Vec<_>>()
    });
    assert_eq!(
        operations,
        vec![
            b"fixture-operation-v1".to_vec(),
            b"fixture-cleanup-v1".to_vec()
        ]
    );

    host.release_stream(stream).unwrap();
    host.release_memory_pool(pool).unwrap();
    host.release_context(context).unwrap();
}

#[test]
fn an_operand_or_a_stream_from_another_context_reaches_no_callback() {
    let (table, backend) = table();
    let host = table.caller();
    let first = host.select_context(0, 0).unwrap();
    let second = host.select_context(1, 1).unwrap();
    let first_pool = host.select_memory_pool(first).unwrap();
    let second_pool = host.select_memory_pool(second).unwrap();
    let first_stream = host.create_stream(first, 0).unwrap();
    let second_stream = host.create_stream(second, 0).unwrap();
    let foreign = host
        .allocate_device_buffer(second_pool, second_stream, 64, 64)
        .unwrap();
    let mut state = CustomOperationStateV1::default();

    {
        let operation = fixture_custom_operation(&mut state);
        // A stream from another context.
        assert_eq!(
            host.run_custom_operation(
                first,
                second_stream,
                CustomCompletionV1::Synchronous,
                b"fixture-operation-v1",
                &[],
                &operation,
            )
            .map_err(status),
            Err(BundleStatusV1::InvalidArgument)
        );
        // An operand from another context.
        assert_eq!(
            host.run_custom_operation(
                first,
                first_stream,
                CustomCompletionV1::Synchronous,
                b"fixture-operation-v1",
                &[CustomOperandV1::read(DeviceBufferSpanV1::new(
                    foreign, 0, 64
                ))],
                &operation,
            )
            .map_err(status),
            Err(BundleStatusV1::InvalidArgument)
        );
    }

    assert_eq!(
        state.calls, 0,
        "an address the launching context cannot reach is never lent"
    );
    assert!(
        backend
            .with_device(|device| device.custom_operations().to_vec())
            .is_empty()
    );

    host.release_device_buffer(foreign).unwrap();
    host.release_stream(second_stream).unwrap();
    host.release_stream(first_stream).unwrap();
    host.release_memory_pool(second_pool).unwrap();
    host.release_memory_pool(first_pool).unwrap();
    host.release_context(second).unwrap();
    host.release_context(first).unwrap();
}

#[test]
fn a_custom_operation_the_host_refuses_never_runs() {
    let (table, backend) = table();
    let host = table.caller();
    let context = host.select_context(0, 0).unwrap();
    let pool = host.select_memory_pool(context).unwrap();
    let stream = host.create_stream(context, 0).unwrap();
    let buffer = host.allocate_device_buffer(pool, stream, 64, 64).unwrap();
    let mut state = CustomOperationStateV1::default();
    let operation = fixture_custom_operation(&mut state);
    let span = DeviceBufferSpanV1::new(buffer, 0, 32);

    // A span that runs past its allocation, a handle from no live slot, a
    // stream this instance never issued, an unnamed operation, and a written
    // operand that overlaps one the same call reads.
    for (label, refused) in [
        (
            "a range outside its allocation",
            host.run_custom_operation(
                context,
                stream,
                CustomCompletionV1::MayEnqueue,
                b"fixture-operation-v1",
                &[CustomOperandV1::read(DeviceBufferSpanV1::new(
                    buffer, 48, 32,
                ))],
                &operation,
            ),
        ),
        (
            "a stale buffer handle",
            host.run_custom_operation(
                context,
                stream,
                CustomCompletionV1::MayEnqueue,
                b"fixture-operation-v1",
                &[CustomOperandV1::read(DeviceBufferSpanV1::new(
                    BundleDeviceBufferHandleV1::from_parts(9, 9),
                    0,
                    32,
                ))],
                &operation,
            ),
        ),
        (
            "a stream this instance did not issue",
            host.run_custom_operation(
                context,
                BundleStreamHandleV1::from_parts(9, 9),
                CustomCompletionV1::MayEnqueue,
                b"fixture-operation-v1",
                &[CustomOperandV1::read(span)],
                &operation,
            ),
        ),
        (
            "an operation with no identity",
            host.run_custom_operation(
                context,
                stream,
                CustomCompletionV1::MayEnqueue,
                b"",
                &[CustomOperandV1::read(span)],
                &operation,
            ),
        ),
        (
            "a written operand that aliases one it reads",
            host.run_custom_operation(
                context,
                stream,
                CustomCompletionV1::MayEnqueue,
                b"fixture-operation-v1",
                &[
                    CustomOperandV1::read(DeviceBufferSpanV1::new(buffer, 0, 32)),
                    CustomOperandV1::write(DeviceBufferSpanV1::new(buffer, 16, 32)),
                ],
                &operation,
            ),
        ),
    ] {
        assert!(refused.is_err(), "{label} must be refused");
    }
    assert_eq!(state.calls, 0, "no refusal reached the operation");
    assert!(
        backend
            .with_device(|device| device.custom_operations().to_vec())
            .is_empty()
    );

    // The same operands the alias rule refuses are accepted when the written
    // one starts where the read one ends, which is the boundary case.
    {
        let operation = fixture_custom_operation(&mut state);
        host.run_custom_operation(
            context,
            stream,
            CustomCompletionV1::MayEnqueue,
            b"fixture-operation-v1",
            &[
                CustomOperandV1::read(DeviceBufferSpanV1::new(buffer, 0, 32)),
                CustomOperandV1::write(DeviceBufferSpanV1::new(buffer, 32, 32)),
            ],
            &operation,
        )
        .expect("adjacent spans of one allocation are disjoint");
    }
    assert_eq!(state.calls, 1);

    host.release_device_buffer(buffer).unwrap();
    host.release_stream(stream).unwrap();
    host.release_memory_pool(pool).unwrap();
    host.release_context(context).unwrap();
}

#[test]
fn a_custom_operation_that_fails_leaves_every_resource_releasable() {
    let (table, backend) = table();
    let host = table.caller();
    let context = host.select_context(0, 0).unwrap();
    let pool = host.select_memory_pool(context).unwrap();
    let stream = host.create_stream(context, 0).unwrap();
    let buffer = host.allocate_device_buffer(pool, stream, 64, 64).unwrap();
    let mut state = CustomOperationStateV1 {
        code: 5,
        ..CustomOperationStateV1::default()
    };

    let failed = {
        let operation = fixture_custom_operation(&mut state);
        host.run_custom_operation(
            context,
            stream,
            CustomCompletionV1::MayEnqueue,
            b"fixture-operation-v1",
            &[CustomOperandV1::write(DeviceBufferSpanV1::new(
                buffer, 0, 64,
            ))],
            &operation,
        )
    };

    // The operation's own code comes back as the bundle's, not as a host
    // failure, and the call changed nothing this instance tracks.
    assert_eq!(failed, Err(BundleAbiError::CustomOperation(5)));
    assert_eq!(state.calls, 1);
    assert_eq!(
        backend.with_device(|device| device.custom_operations().len()),
        1
    );

    // A refusal from the host itself invokes nothing and stays retryable.
    backend.with_device_mut(|device| device.refuse(FakeRefusalV1::RunCustomOperation));
    let mut refused_state = CustomOperationStateV1::default();
    let refused = {
        let operation = fixture_custom_operation(&mut refused_state);
        host.run_custom_operation(
            context,
            stream,
            CustomCompletionV1::MayEnqueue,
            b"fixture-operation-v1",
            &[CustomOperandV1::write(DeviceBufferSpanV1::new(
                buffer, 0, 64,
            ))],
            &operation,
        )
    };
    assert_eq!(refused.map_err(status), Err(BundleStatusV1::HostError));
    assert_eq!(refused_state.calls, 0);
    backend.with_device_mut(|device| device.allow(FakeRefusalV1::RunCustomOperation));

    host.release_device_buffer(buffer).unwrap();
    host.release_stream(stream).unwrap();
    host.release_memory_pool(pool).unwrap();
    host.release_context(context).unwrap();
    assert_eq!(backend.live_resource_count(), 0);
    assert_eq!(backend.with_device(FakeHostDeviceV1::live_object_count), 0);
}

#[test]
fn a_range_outside_its_allocation_never_reaches_the_device() {
    let (table, _backend) = table();
    let host = table.caller();
    let context = host.select_context(0, 0).unwrap();
    let pool = host.select_memory_pool(context).unwrap();
    let stream = host.create_stream(context, 0).unwrap();
    let buffer = host.allocate_device_buffer(pool, stream, 8, 64).unwrap();

    assert_eq!(
        host.zero_device_buffer(stream, DeviceBufferSpanV1::new(buffer, 4, 8))
            .map_err(status),
        Err(BundleStatusV1::InvalidArgument)
    );
    assert_eq!(
        host.zero_device_buffer(
            stream,
            DeviceBufferSpanV1::new(BundleDeviceBufferHandleV1::from_parts(9, 9), 0, 8)
        )
        .map_err(status),
        Err(BundleStatusV1::StaleHandle)
    );

    host.release_device_buffer(buffer).unwrap();
    host.release_stream(stream).unwrap();
    host.release_memory_pool(pool).unwrap();
    host.release_context(context).unwrap();
}

#[test]
fn a_refused_release_keeps_the_resource_tracked_and_retryable() {
    let (table, backend) = table();
    let host = table.caller();
    let context = host.select_context(0, 0).unwrap();
    let pool = host.select_memory_pool(context).unwrap();
    let stream = host.create_stream(context, 0).unwrap();
    let buffer = host.allocate_device_buffer(pool, stream, 8, 64).unwrap();
    backend.with_device_mut(|device| device.refuse(FakeRefusalV1::ReleaseDeviceBuffer));

    assert_eq!(
        host.release_device_buffer(buffer).map_err(status),
        Err(BundleStatusV1::HostError)
    );

    // Nothing was dropped, so the driver's allocation is still tracked by the
    // handle that owns it and the same release can be retried.
    assert_eq!(backend.live_resource_count(), 4);
    assert_eq!(backend.with_device(FakeHostDeviceV1::live_object_count), 4);
    let span = DeviceBufferSpanV1::new(buffer, 0, 8);
    host.copy_host_to_device(stream, span, &[1, 2, 3, 4, 5, 6, 7, 8])
        .unwrap();
    // The refused release did not move the release order on either.
    assert_eq!(
        host.release_stream(stream).map_err(status),
        Err(BundleStatusV1::InvalidState)
    );

    backend.with_device_mut(|device| device.allow(FakeRefusalV1::ReleaseDeviceBuffer));
    host.release_device_buffer(buffer).unwrap();
    host.release_stream(stream).unwrap();
    host.release_memory_pool(pool).unwrap();
    host.release_context(context).unwrap();

    assert_eq!(backend.live_resource_count(), 0);
    assert_eq!(backend.with_device(FakeHostDeviceV1::live_object_count), 0);
}

#[test]
fn a_refused_rank_buffer_release_keeps_every_buffer_the_driver_still_owns() {
    let (table, backend) = table();
    let host = table.caller();
    let kv = host.allocate_physical_kv(&kv_request()).unwrap();
    // One context, pool, and stream per rank, plus one buffer per rank.
    let device_objects = backend.with_device(FakeHostDeviceV1::live_object_count);
    assert_eq!(device_objects, (RANKS as usize) * 4);
    backend.with_device_mut(|device| device.refuse(FakeRefusalV1::ReleaseDeviceBuffer));

    assert_eq!(
        host.release_physical_kv(kv).map_err(status),
        Err(BundleStatusV1::HostError)
    );
    assert_eq!(backend.live_resource_count(), 1);
    assert_eq!(
        backend.with_device(FakeHostDeviceV1::live_object_count),
        device_objects
    );

    backend.with_device_mut(|device| device.allow(FakeRefusalV1::ReleaseDeviceBuffer));
    host.release_physical_kv(kv).unwrap();

    assert_eq!(backend.live_resource_count(), 0);
    assert_eq!(backend.with_device(FakeHostDeviceV1::live_object_count), 0);
}

#[test]
fn a_committed_append_zeroes_exactly_the_page_it_reserved() {
    let (table, _backend) = table();
    let host = table.caller();
    let context = host.select_context(0, 0).unwrap();
    let pool = host.select_memory_pool(context).unwrap();
    let stream = host.create_stream(context, 0).unwrap();
    let kv = host.allocate_physical_kv(&kv_request()).unwrap();
    let request = host.create_kv_request(kv, 16).unwrap();

    // Every page starts dirty, so a page that reads back as zeroes was zeroed
    // by this commit rather than by its allocation.
    let dirt = vec![0xAB_u8; PAGE_BYTES as usize];
    for rank in 0..RANKS {
        for page in 0..PAGE_CAPACITY {
            let span = host
                .resolve_kv_view(request, rank, &kv_page_view_id_v1(page))
                .unwrap();
            host.copy_host_to_device(stream, span, &dirt).unwrap();
        }
    }

    let transaction = host
        .prepare_kv(request, &kv_append_operation_v1(PAGE_TOKENS))
        .unwrap();
    host.commit_kv(transaction).unwrap();

    for rank in 0..RANKS {
        let zeroed = (0..PAGE_CAPACITY)
            .filter(|page| {
                read_kv_page(host, stream, request, rank, *page) == vec![0; PAGE_BYTES as usize]
            })
            .collect::<Vec<_>>();
        assert_eq!(zeroed.len(), 1, "rank {rank} zeroed {zeroed:?}");
        for page in 0..PAGE_CAPACITY {
            if page != zeroed[0] {
                assert_eq!(read_kv_page(host, stream, request, rank, page), dirt);
            }
        }
    }

    host.release_kv_request(request).unwrap();
    host.release_physical_kv(kv).unwrap();
    host.release_stream(stream).unwrap();
    host.release_memory_pool(pool).unwrap();
    host.release_context(context).unwrap();
}

#[test]
fn a_refused_page_fill_leaves_the_append_uncommitted_and_cancellable() {
    let (table, backend) = table();
    let host = table.caller();
    let kv = host.allocate_physical_kv(&kv_request()).unwrap();
    let request = host.create_kv_request(kv, 16).unwrap();
    let transaction = host
        .prepare_kv(request, &kv_append_operation_v1(PAGE_TOKENS))
        .unwrap();
    backend.with_device_mut(|device| device.refuse(FakeRefusalV1::ZeroDeviceBuffer));

    assert_eq!(
        host.commit_kv(transaction).map_err(status),
        Err(BundleStatusV1::HostError)
    );

    // The append is still prepared, so the bundle can still cancel it and the
    // pages it reserved come back.
    host.cancel_kv(transaction).unwrap();
    backend.with_device_mut(|device| device.allow(FakeRefusalV1::ZeroDeviceBuffer));
    let retried = host
        .prepare_kv(request, &kv_append_operation_v1(PAGE_TOKENS))
        .unwrap();
    host.commit_kv(retried).unwrap();

    host.release_kv_request(request).unwrap();
    host.release_physical_kv(kv).unwrap();
    assert_eq!(backend.live_resource_count(), 0);
}

fn read_kv_page(
    host: &HostServicesV1,
    stream: BundleStreamHandleV1,
    request: BundleKvRequestHandleV1,
    rank: u32,
    page: u64,
) -> Vec<u8> {
    let span = host
        .resolve_kv_view(request, rank, &kv_page_view_id_v1(page))
        .unwrap();
    let mut bytes = vec![0_u8; PAGE_BYTES as usize];
    host.copy_device_to_host(stream, &mut bytes, span).unwrap();
    bytes
}

#[test]
fn an_unbound_stream_handle_still_allocates_from_its_pool() {
    let (table, _backend) = table();
    let host = table.caller();
    let context = host.select_context(0, 0).unwrap();
    let pool = host.select_memory_pool(context).unwrap();

    let buffer = host
        .allocate_device_buffer(pool, BundleStreamHandleV1::from_raw(0), 16, 16)
        .unwrap();

    host.release_device_buffer(buffer).unwrap();
    host.release_memory_pool(pool).unwrap();
    host.release_context(context).unwrap();
}

#[test]
fn a_backend_that_finishes_its_host_transfers_passes_the_transfer_contract() {
    let mut device = FakeHostDeviceV1::new();

    assert_eq!(assert_host_transfer_contract_v1(&mut device, 0), Ok(()));
    assert_eq!(device.live_object_count(), 0);
}

#[test]
fn a_backend_that_leaves_a_host_transfer_in_flight_fails_the_transfer_contract() {
    let mut device = FakeHostDeviceV1::new();
    // A driver that returns while its DMA is still running behaves this way.
    device.defer_host_transfers(true);

    assert_eq!(
        assert_host_transfer_contract_v1(&mut device, 0),
        Err(BundleStatusV1::InvalidState)
    );
    // The check owns and releases everything it made, whatever it decided.
    assert_eq!(device.live_object_count(), 0);
}

#[test]
fn only_the_declared_collective_contract_reduces_anything() {
    let (table, _backend) = table();
    let host = table.caller();
    let context = host.select_context(0, 0).unwrap();
    let pool = host.select_memory_pool(context).unwrap();
    let stream = host.create_stream(context, 0).unwrap();
    let mut buffers = Vec::new();
    for rank in 0..RANKS {
        let buffer = host.allocate_device_buffer(pool, stream, 4, 64).unwrap();
        host.copy_host_to_device(
            stream,
            DeviceBufferSpanV1::new(buffer, 0, 4),
            &(rank as f32 + 1.0).to_le_bytes(),
        )
        .unwrap();
        buffers.push(buffer);
    }
    let arguments = buffers
        .iter()
        .map(|buffer| {
            runtime_bundle_abi::KernelArgumentV1::DeviceBuffer(DeviceBufferSpanV1::new(
                *buffer, 0, 4,
            ))
        })
        .collect::<Vec<_>>();
    let collective = host
        .create_collective(&CollectiveRequestV1 {
            flags: 0,
            rank_count: u64::from(RANKS),
            semantic_contract: COLLECTIVE_SUM_F32_V1.to_vec(),
        })
        .unwrap();

    // A contract this host does not implement never touches a rank buffer.
    assert_eq!(
        host.execute_collective(collective, stream, b"product-f32-v1", &arguments)
            .map_err(status),
        Err(BundleStatusV1::Unsupported)
    );
    for buffer in &buffers {
        let mut read = [0_u8; 4];
        host.copy_device_to_host(stream, &mut read, DeviceBufferSpanV1::new(*buffer, 0, 4))
            .unwrap();
        assert_ne!(f32::from_le_bytes(read), 3.0);
    }

    host.execute_collective(collective, stream, COLLECTIVE_SUM_F32_V1, &arguments)
        .unwrap();
    for buffer in &buffers {
        let mut read = [0_u8; 4];
        host.copy_device_to_host(stream, &mut read, DeviceBufferSpanV1::new(*buffer, 0, 4))
            .unwrap();
        assert_eq!(f32::from_le_bytes(read), 3.0);
    }

    host.release_collective(collective).unwrap();
    for buffer in buffers.into_iter().rev() {
        host.release_device_buffer(buffer).unwrap();
    }
    host.release_stream(stream).unwrap();
    host.release_memory_pool(pool).unwrap();
    host.release_context(context).unwrap();
}

#[test]
fn an_allocation_that_misses_its_alignment_is_released_not_handed_over() {
    let (table, backend) = table();
    let host = table.caller();
    let context = host.select_context(0, 0).unwrap();
    let pool = host.select_memory_pool(context).unwrap();
    let stream = host.create_stream(context, 0).unwrap();
    // This allocator guarantees less than the bundle is about to ask for.
    backend.with_device_mut(|device| device.set_allocation_alignment(64));

    let weaker = host.allocate_device_buffer(pool, stream, 8, 64).unwrap();
    host.release_device_buffer(weaker).unwrap();
    assert_eq!(
        host.allocate_device_buffer(pool, stream, 8, 128)
            .map_err(status),
        Err(BundleStatusV1::Unsupported)
    );

    // The refused allocation went back to the driver and took no order entry.
    assert_eq!(backend.live_resource_count(), 3);
    assert_eq!(backend.with_device(FakeHostDeviceV1::live_object_count), 3);

    host.release_stream(stream).unwrap();
    host.release_memory_pool(pool).unwrap();
    host.release_context(context).unwrap();
    assert_eq!(backend.with_device(FakeHostDeviceV1::live_object_count), 0);
}

#[test]
fn a_kv_pool_whose_pages_cannot_be_aligned_allocates_nothing() {
    let (table, backend) = table();
    let host = table.caller();
    // The declared page layout needs 64-byte pages; this allocator gives 16.
    backend.with_device_mut(|device| device.set_allocation_alignment(16));

    assert_eq!(
        host.allocate_physical_kv(&kv_request()).map_err(status),
        Err(BundleStatusV1::Unsupported)
    );

    assert_eq!(backend.live_resource_count(), 0);
    assert_eq!(backend.with_device(FakeHostDeviceV1::live_object_count), 0);
}

#[test]
fn one_role_cannot_be_claimed_by_two_components() {
    let config = HostInstanceConfigV1::new(
        HostTopologyV1::new(vec![0, 1]).expect("the test topology is valid"),
    )
    .with_component(COMPONENT_ROLE, component_bytes())
    .expect("the first component claims the role");

    // A bundle opens a component by role, so a second claim has no answer and
    // is refused rather than replacing the first.
    assert_eq!(
        config
            .clone()
            .with_component(COMPONENT_ROLE, vec![0; 4])
            .err(),
        Some(HostConfigErrorV1::DuplicateComponentRole)
    );
    assert_eq!(
        config
            .with_compiled_image(COMPONENT_ROLE, vec![0; 4], compiled_image_description())
            .err(),
        Some(HostConfigErrorV1::DuplicateComponentRole)
    );
}

#[test]
fn a_rank_image_division_is_never_reused_across_topologies() {
    // Two ranks of eight bytes each, which is the two-rank division.
    let two_ranks = compiled_image_description();
    let config = HostInstanceConfigV1::new(
        HostTopologyV1::new(vec![0]).expect("the test topology is valid"),
    )
    .with_compiled_image(COMPONENT_ROLE, component_bytes(), two_ranks)
    .expect("the test component role is unique");
    let backend = Rc::new(HostServicesV1Backend::new(FakeHostDeviceV1::new(), config));
    let table = HostServicesTableV1::new(Rc::clone(&backend));
    let host = table.caller();
    let context = host.select_context(0, 0).unwrap();
    let pool = host.select_memory_pool(context).unwrap();
    let stream = host.create_stream(context, 0).unwrap();
    let image = host.open_compiled_image(COMPONENT_ROLE).unwrap();

    // This host serves one rank, so rank one is not a rank it has.
    assert_eq!(
        host.upload_rank_arena(context, pool, stream, image, 1)
            .map_err(status),
        Err(BundleStatusV1::InvalidArgument)
    );

    host.release_compiled_image(image).unwrap();
    host.release_stream(stream).unwrap();
    host.release_memory_pool(pool).unwrap();
    host.release_context(context).unwrap();
}

#[test]
fn a_collective_that_honours_its_stream_passes_the_ordering_contract() {
    let mut device = FakeHostDeviceV1::new();

    assert_eq!(
        assert_collective_ordering_contract_v1(&mut device, 0),
        Ok(())
    );
    assert_eq!(device.live_object_count(), 0);
}

#[test]
fn a_collective_still_honours_its_stream_when_the_work_on_it_is_pending() {
    let mut device = FakeHostDeviceV1::new();
    // The writes that fill the rank buffers are now work ordered on the
    // collective's stream rather than work already done.
    device.defer_host_transfers(true);

    assert_eq!(
        assert_collective_ordering_contract_v1(&mut device, 0),
        Ok(()),
        "the collective must let its own stream's work land before reading"
    );
    assert_eq!(device.live_object_count(), 0);
}

#[test]
fn a_collective_that_reads_before_its_stream_lands_fails_the_ordering_contract() {
    let mut device = FakeHostDeviceV1::new();
    device.defer_host_transfers(true);
    // A backend that stages through some other stream behaves this way, and
    // the reduction it returns is a plausible number over stale bytes.
    device.skip_collective_stream_ordering(true);

    assert_eq!(
        assert_collective_ordering_contract_v1(&mut device, 0),
        Err(BundleStatusV1::InvalidState)
    );
    assert_eq!(device.live_object_count(), 0);
}

/// A peer edge is a lease over shared device state, not a switch.
///
/// Two things make that necessary. The devices are shared with whoever else is
/// using them, so an edge this host found enabled belongs to that user and must
/// survive this instance. And one instance may want the same edge more than
/// once, so the transition happens on the first lease and is undone on the
/// last, not on every acquire and release.
#[test]
fn a_preexisting_context_peer_edge_is_leased_and_left_enabled() {
    let (table, backend) = table();
    let host = table.caller();
    let owner = host.select_context(0, 0).unwrap();
    let peer = host.select_context(1, 1).unwrap();
    let objects = backend.with_device(FakeHostDeviceV1::context_objects);
    // Another user of these devices enabled this edge before the bundle ran.
    backend.with_device_mut(|device| device.install_context_peer_access(objects[0], objects[1]));

    let access = host.acquire_peer_access(owner, peer).unwrap();
    assert!(
        backend
            .with_device(|device| device.enabled_context_peer_access().to_vec())
            .is_empty(),
        "an edge that was already enabled is not enabled again"
    );
    host.release_peer_access(access).unwrap();
    assert!(
        backend
            .with_device(|device| device.disabled_context_peer_access().to_vec())
            .is_empty(),
        "an edge this host did not enable is not disabled"
    );
    assert!(backend.with_device(|device| device.has_context_peer_access(objects[0], objects[1])));

    host.release_context(peer).unwrap();
    host.release_context(owner).unwrap();
}

#[test]
fn nested_context_peer_leases_transition_the_edge_once() {
    let (table, backend) = table();
    let host = table.caller();
    let owner = host.select_context(0, 0).unwrap();
    let peer = host.select_context(1, 1).unwrap();
    let objects = backend.with_device(FakeHostDeviceV1::context_objects);

    let first = host.acquire_peer_access(owner, peer).unwrap();
    let second = host.acquire_peer_access(owner, peer).unwrap();
    assert_ne!(first, second, "each acquire is its own lease");
    assert_eq!(
        backend
            .with_device(|device| device.enabled_context_peer_access().to_vec())
            .len(),
        1,
        "only the first lease reaches the device"
    );

    host.release_peer_access(second).unwrap();
    assert!(
        backend
            .with_device(|device| device.disabled_context_peer_access().to_vec())
            .is_empty(),
        "a lease that is not the last one disables nothing"
    );
    assert!(backend.with_device(|device| device.has_context_peer_access(objects[0], objects[1])));

    host.release_peer_access(first).unwrap();
    assert_eq!(
        backend.with_device(|device| device.disabled_context_peer_access().to_vec()),
        [(objects[0].get(), objects[1].get())]
    );
    assert!(!backend.with_device(|device| device.has_context_peer_access(objects[0], objects[1])));

    host.release_context(peer).unwrap();
    host.release_context(owner).unwrap();
}

#[test]
fn a_stale_or_out_of_order_context_peer_release_is_refused() {
    let (table, _backend) = table();
    let host = table.caller();
    let owner = host.select_context(0, 0).unwrap();
    let peer = host.select_context(1, 1).unwrap();
    let access = host.acquire_peer_access(owner, peer).unwrap();

    host.release_peer_access(access).unwrap();
    // The same handle a second time names a lease that is no longer there.
    assert_eq!(
        status(host.release_peer_access(access).unwrap_err()),
        BundleStatusV1::StaleHandle
    );

    let access = host.acquire_peer_access(owner, peer).unwrap();
    let stream = host.create_stream(owner, 0).unwrap();
    assert_eq!(
        status(host.release_peer_access(access).unwrap_err()),
        BundleStatusV1::InvalidState,
        "a release out of reverse order is refused"
    );
    host.release_stream(stream).unwrap();
    host.release_peer_access(access).unwrap();
    host.release_context(peer).unwrap();
    host.release_context(owner).unwrap();
}

#[test]
fn a_refused_context_peer_transition_changes_nothing_and_is_retryable() {
    let (table, backend) = table();
    let host = table.caller();
    let owner = host.select_context(0, 0).unwrap();
    let peer = host.select_context(1, 1).unwrap();
    let objects = backend.with_device(FakeHostDeviceV1::context_objects);

    backend.with_device_mut(|device| device.refuse(FakeRefusalV1::EnableContextPeerAccess));
    assert!(host.acquire_peer_access(owner, peer).is_err());
    assert_eq!(backend.live_resource_count(), 2, "nothing was recorded");
    assert!(!backend.with_device(|device| device.has_context_peer_access(objects[0], objects[1])));

    backend.with_device_mut(|device| device.allow(FakeRefusalV1::EnableContextPeerAccess));
    let access = host.acquire_peer_access(owner, peer).unwrap();

    // A refused disable leaves the lease exactly as it was.
    backend.with_device_mut(|device| device.refuse(FakeRefusalV1::ReleasePeerAccess));
    assert!(host.release_peer_access(access).is_err());
    assert_eq!(backend.live_resource_count(), 3);
    assert!(backend.with_device(|device| device.has_context_peer_access(objects[0], objects[1])));

    backend.with_device_mut(|device| device.allow(FakeRefusalV1::ReleasePeerAccess));
    host.release_peer_access(access).unwrap();
    assert!(!backend.with_device(|device| device.has_context_peer_access(objects[0], objects[1])));
    host.release_context(peer).unwrap();
    host.release_context(owner).unwrap();
}

#[test]
fn a_preexisting_pool_peer_grant_is_leased_and_left_enabled() {
    let (table, backend) = table();
    let host = table.caller();
    let owner = host.select_context(0, 0).unwrap();
    let peer = host.select_context(1, 1).unwrap();
    let pool = host.select_memory_pool(owner).unwrap();
    let contexts = backend.with_device(FakeHostDeviceV1::context_objects);
    let pools = backend.with_device(FakeHostDeviceV1::memory_pool_objects);
    backend.with_device_mut(|device| device.install_memory_pool_peer_access(pools[0], contexts[1]));

    host.set_memory_pool_peer_access(pool, peer, true).unwrap();
    assert!(
        backend
            .with_device(|device| device.enabled_memory_pool_peer_access().to_vec())
            .is_empty()
    );
    host.set_memory_pool_peer_access(pool, peer, false).unwrap();
    assert!(
        backend
            .with_device(|device| device.disabled_memory_pool_peer_access().to_vec())
            .is_empty(),
        "a grant this host did not make is not revoked"
    );
    assert!(
        backend.with_device(|device| device.has_memory_pool_peer_access(pools[0], contexts[1]))
    );

    host.release_memory_pool(pool).unwrap();
    host.release_context(peer).unwrap();
    host.release_context(owner).unwrap();
}

#[test]
fn nested_pool_peer_leases_grant_once_and_revoke_once() {
    let (table, backend) = table();
    let host = table.caller();
    let owner = host.select_context(0, 0).unwrap();
    let peer = host.select_context(1, 1).unwrap();
    let pool = host.select_memory_pool(owner).unwrap();
    let contexts = backend.with_device(FakeHostDeviceV1::context_objects);
    let pools = backend.with_device(FakeHostDeviceV1::memory_pool_objects);

    host.set_memory_pool_peer_access(pool, peer, true).unwrap();
    host.set_memory_pool_peer_access(pool, peer, true).unwrap();
    assert_eq!(
        backend.with_device(|device| device.enabled_memory_pool_peer_access().to_vec()),
        [(pools[0].get(), contexts[1].get())]
    );

    host.set_memory_pool_peer_access(pool, peer, false).unwrap();
    assert!(
        backend
            .with_device(|device| device.disabled_memory_pool_peer_access().to_vec())
            .is_empty()
    );
    host.set_memory_pool_peer_access(pool, peer, false).unwrap();
    assert_eq!(
        backend.with_device(|device| device.disabled_memory_pool_peer_access().to_vec()),
        [(pools[0].get(), contexts[1].get())]
    );

    // A release with no lease behind it is a mistake, not a no-op.
    assert_eq!(
        status(
            host.set_memory_pool_peer_access(pool, peer, false)
                .unwrap_err()
        ),
        BundleStatusV1::InvalidState
    );

    host.release_memory_pool(pool).unwrap();
    host.release_context(peer).unwrap();
    host.release_context(owner).unwrap();
}

#[test]
fn a_refused_pool_peer_transition_changes_nothing_and_is_retryable() {
    let (table, backend) = table();
    let host = table.caller();
    let owner = host.select_context(0, 0).unwrap();
    let peer = host.select_context(1, 1).unwrap();
    let pool = host.select_memory_pool(owner).unwrap();
    let contexts = backend.with_device(FakeHostDeviceV1::context_objects);
    let pools = backend.with_device(FakeHostDeviceV1::memory_pool_objects);

    backend.with_device_mut(|device| device.refuse(FakeRefusalV1::EnableMemoryPoolPeerAccess));
    assert!(host.set_memory_pool_peer_access(pool, peer, true).is_err());
    assert!(
        !backend.with_device(|device| device.has_memory_pool_peer_access(pools[0], contexts[1]))
    );
    // No lease was recorded, so this is still an unmatched release.
    assert_eq!(
        status(
            host.set_memory_pool_peer_access(pool, peer, false)
                .unwrap_err()
        ),
        BundleStatusV1::InvalidState
    );

    backend.with_device_mut(|device| device.allow(FakeRefusalV1::EnableMemoryPoolPeerAccess));
    host.set_memory_pool_peer_access(pool, peer, true).unwrap();

    backend.with_device_mut(|device| device.refuse(FakeRefusalV1::DisableMemoryPoolPeerAccess));
    assert!(host.set_memory_pool_peer_access(pool, peer, false).is_err());
    assert!(
        backend.with_device(|device| device.has_memory_pool_peer_access(pools[0], contexts[1]))
    );

    backend.with_device_mut(|device| device.allow(FakeRefusalV1::DisableMemoryPoolPeerAccess));
    host.set_memory_pool_peer_access(pool, peer, false).unwrap();
    assert!(
        !backend.with_device(|device| device.has_memory_pool_peer_access(pools[0], contexts[1]))
    );

    host.release_memory_pool(pool).unwrap();
    host.release_context(peer).unwrap();
    host.release_context(owner).unwrap();
}

/// A resource a live lease names is not this instance's to give back yet.
#[test]
fn a_pool_or_context_a_peer_lease_names_cannot_be_released() {
    let (table, _backend) = table();
    let host = table.caller();
    let owner = host.select_context(0, 0).unwrap();
    let pool = host.select_memory_pool(owner).unwrap();
    // The peer context is the newest resource, so reverse order alone would
    // allow releasing it; the lease is what refuses.
    let peer = host.select_context(1, 1).unwrap();
    host.set_memory_pool_peer_access(pool, peer, true).unwrap();

    assert_eq!(
        status(host.release_context(peer).unwrap_err()),
        BundleStatusV1::InvalidState
    );
    assert_eq!(
        status(host.release_memory_pool(pool).unwrap_err()),
        BundleStatusV1::InvalidState
    );

    host.set_memory_pool_peer_access(pool, peer, false).unwrap();
    host.release_context(peer).unwrap();
    host.release_memory_pool(pool).unwrap();
    host.release_context(owner).unwrap();
}

/// A rejected acquire enables nothing, because it fails before the transition.
///
/// The order matters: if the edge were enabled first and the handle refused
/// afterwards, the only way back would be a disable that can itself be refused,
/// and a lease with no handle would hold the contexts hostage for the life of
/// the instance. Every fallible piece of bookkeeping therefore happens first.
#[test]
fn a_peer_handle_table_that_is_full_neither_enables_nor_records_an_edge() {
    let (table, backend) = table();
    let host = table.caller();
    let owner = host.select_context(0, 0).unwrap();
    let peer = host.select_context(1, 1).unwrap();
    let objects = backend.with_device(FakeHostDeviceV1::context_objects);

    // Fill the peer handle table with leases over one edge.
    let mut held = Vec::new();
    loop {
        match host.acquire_peer_access(owner, peer) {
            Ok(access) => held.push(access),
            Err(error) => {
                assert_eq!(status(error), BundleStatusV1::ResourceExhausted);
                break;
            }
        }
    }
    assert!(held.len() > 1);
    let enabled_before = backend
        .with_device(|device| device.enabled_context_peer_access().to_vec())
        .len();
    assert_eq!(enabled_before, 1, "one edge, one transition");

    // A fresh edge now has nowhere to record its handle. Arm the disable that a
    // rollback would need, so a rollback that ran and failed would be visible.
    backend.with_device_mut(|device| device.refuse(FakeRefusalV1::ReleasePeerAccess));
    let fresh = host.acquire_peer_access(peer, owner);
    assert_eq!(
        status(fresh.unwrap_err()),
        BundleStatusV1::ResourceExhausted
    );
    assert_eq!(
        backend
            .with_device(|device| device.enabled_context_peer_access().to_vec())
            .len(),
        enabled_before,
        "the rejected acquire enabled nothing"
    );
    assert!(!backend.with_device(|device| device.has_context_peer_access(objects[1], objects[0])));
    backend.with_device_mut(|device| device.allow(FakeRefusalV1::ReleasePeerAccess));

    // No orphan lease was recorded either: the whole table gives back cleanly
    // and both contexts are then free to release.
    while let Some(access) = held.pop() {
        host.release_peer_access(access).unwrap();
    }
    assert_eq!(
        backend
            .with_device(|device| device.disabled_context_peer_access().to_vec())
            .len(),
        1
    );
    host.release_context(peer).unwrap();
    host.release_context(owner).unwrap();
    assert_eq!(backend.live_resource_count(), 0);
}

/// One rank is one context handle, and one context is one pool handle.
///
/// That is what makes a lease key exact: the handles the ABI call names are the
/// driver objects underneath, so a grant cannot be acquired through one name for
/// a pool and released through another, and every handle a lease names is
/// guarded from teardown.
#[test]
fn repeated_selection_returns_one_handle_per_underlying_object() {
    let (table, backend) = table();
    let host = table.caller();

    let first = host.select_context(0, 0).unwrap();
    let again = host.select_context(0, 0).unwrap();
    assert_eq!(first, again, "one rank is one context handle");
    assert_eq!(
        backend.with_device(FakeHostDeviceV1::context_objects).len(),
        1,
        "and one driver context"
    );

    let pool = host.select_memory_pool(first).unwrap();
    let pool_again = host.select_memory_pool(again).unwrap();
    assert_eq!(pool_again, pool, "one context is one pool handle");
    assert_eq!(
        backend
            .with_device(FakeHostDeviceV1::memory_pool_objects)
            .len(),
        1,
        "and one driver pool"
    );
    // Two selections, two users: the first release gives nothing back.
    assert_eq!(backend.live_resource_count(), 2);
    host.release_memory_pool(pool).unwrap();
    assert_eq!(backend.live_resource_count(), 2);
    host.release_memory_pool(pool_again).unwrap();
    assert_eq!(backend.live_resource_count(), 1);
    assert_eq!(
        status(host.release_memory_pool(pool).unwrap_err()),
        BundleStatusV1::StaleHandle
    );

    host.release_context(again).unwrap();
    host.release_context(first).unwrap();
    assert_eq!(backend.live_resource_count(), 0);
}

#[test]
fn a_pool_grant_is_released_through_the_pool_that_acquired_it_or_not_at_all() {
    let (table, backend) = table();
    let host = table.caller();
    let owner = host.select_context(0, 0).unwrap();
    let peer = host.select_context(1, 1).unwrap();
    let owner_pool = host.select_memory_pool(owner).unwrap();
    let peer_pool = host.select_memory_pool(peer).unwrap();
    let contexts = backend.with_device(FakeHostDeviceV1::context_objects);
    let pools = backend.with_device(FakeHostDeviceV1::memory_pool_objects);
    assert_eq!(pools.len(), 2, "one pool per context");

    host.set_memory_pool_peer_access(owner_pool, peer, true)
        .unwrap();
    // The other pool's grant is a different lease, and releasing it matches
    // nothing this instance took.
    assert_eq!(
        status(
            host.set_memory_pool_peer_access(peer_pool, owner, false)
                .unwrap_err()
        ),
        BundleStatusV1::InvalidState
    );
    assert!(
        backend
            .with_device(|device| device.disabled_memory_pool_peer_access().to_vec())
            .is_empty(),
        "an unmatched release reaches no device"
    );

    // Every handle the live grant names refuses teardown: the pool it was made
    // through, the context that owns that pool, and the peer context.
    for refused in [
        host.release_memory_pool(owner_pool).err(),
        host.release_context(owner).err(),
        host.release_context(peer).err(),
    ] {
        assert_eq!(
            status(refused.expect("a handle a live grant names is not releasable")),
            BundleStatusV1::InvalidState
        );
    }

    host.set_memory_pool_peer_access(owner_pool, peer, false)
        .unwrap();
    assert_eq!(
        backend.with_device(|device| device.disabled_memory_pool_peer_access().to_vec()),
        [(pools[0].get(), contexts[1].get())]
    );
    host.release_memory_pool(peer_pool).unwrap();
    host.release_memory_pool(owner_pool).unwrap();
    host.release_context(peer).unwrap();
    host.release_context(owner).unwrap();
    assert_eq!(backend.live_resource_count(), 0);
}

/// A second selection shares the lease rather than taking a second one.
#[test]
fn a_shared_pool_handle_grants_once_and_never_disables_early() {
    let (table, backend) = table();
    let host = table.caller();
    let owner = host.select_context(0, 0).unwrap();
    let peer = host.select_context(1, 1).unwrap();
    // Two users of the same pool, as a host runtime and a route are.
    let runtime_pool = host.select_memory_pool(owner).unwrap();
    let route_pool = host.select_memory_pool(owner).unwrap();
    assert_eq!(runtime_pool, route_pool);

    host.set_memory_pool_peer_access(runtime_pool, peer, true)
        .unwrap();
    host.set_memory_pool_peer_access(route_pool, peer, true)
        .unwrap();
    assert_eq!(
        backend
            .with_device(|device| device.enabled_memory_pool_peer_access().to_vec())
            .len(),
        1
    );

    host.set_memory_pool_peer_access(route_pool, peer, false)
        .unwrap();
    assert!(
        backend
            .with_device(|device| device.disabled_memory_pool_peer_access().to_vec())
            .is_empty(),
        "the grant one user gave back is still the other user's"
    );
    // One user giving its selection back tears nothing down, so it is not the
    // release the guard is about.
    host.release_memory_pool(route_pool).unwrap();
    assert_eq!(backend.live_resource_count(), 3);
    // The release that would give the pool back is refused while the grant
    // stands, and so is either context of it.
    for refused in [
        host.release_memory_pool(runtime_pool).err(),
        host.release_context(owner).err(),
        host.release_context(peer).err(),
    ] {
        assert_eq!(
            status(refused.expect("a handle a live grant names is not releasable")),
            BundleStatusV1::InvalidState
        );
    }

    host.set_memory_pool_peer_access(runtime_pool, peer, false)
        .unwrap();
    assert_eq!(
        backend
            .with_device(|device| device.disabled_memory_pool_peer_access().to_vec())
            .len(),
        1
    );
    host.release_memory_pool(runtime_pool).unwrap();
    host.release_context(peer).unwrap();
    host.release_context(owner).unwrap();
    assert_eq!(backend.live_resource_count(), 0);
}

/// One device is one rank, because one device is one primary context.
///
/// The exactness every peer lease key rests on comes from here: a context
/// handle names one driver context only while no two ranks share a device. Two
/// ranks on one ordinal would be two handles over one context, and their edges
/// would key two leases over one driver edge.
#[test]
fn a_topology_that_puts_two_ranks_on_one_device_is_refused() {
    assert_eq!(
        HostTopologyV1::new(vec![0, 1, 2, 3]).map(|_| ()),
        Ok(()),
        "four ranks on four devices is the supported shape"
    );
    assert_eq!(
        HostTopologyV1::new(vec![0, 0]).err(),
        Some(HostConfigErrorV1::DuplicateDeviceOrdinal { ordinal: 0 })
    );
    assert_eq!(
        HostTopologyV1::new(vec![0, 1, 2, 1]).err(),
        Some(HostConfigErrorV1::DuplicateDeviceOrdinal { ordinal: 1 }),
        "the ordinal the second rank repeated is the one named"
    );
    assert_eq!(
        HostConfigErrorV1::DuplicateDeviceOrdinal { ordinal: 3 }.to_string(),
        "two host topology ranks name device ordinal 3, and one device is one rank"
    );
    // The other topology refusals are unchanged and still distinct.
    assert_eq!(
        HostTopologyV1::new(Vec::new()).err(),
        Some(HostConfigErrorV1::EmptyTopology)
    );
    assert_eq!(
        HostTopologyV1::new(vec![0, -1]).err(),
        Some(HostConfigErrorV1::NegativeDeviceOrdinal)
    );
    // A single rank on a single device is still a topology.
    assert!(HostTopologyV1::new(vec![7]).is_ok());
}

#[test]
fn recurrent_reservations_reduce_kv_capacity_and_refuse_overcommit() {
    let capacity = |reservation| {
        let config = HostInstanceConfigV1::new(HostTopologyV1::new(vec![0, 1]).unwrap())
            .with_kv_contract(DeclaredKvContractV1::new(LAYOUT_ID.to_vec(), page_layout()))
            .with_serving_kv(ServingKvPolicyV1 {
                sampling_vocabulary: None,
                maximum_context_tokens: None,
                state_reserved_bytes: reservation,
                utilization_millionths: 1_000_000,
                maximum_pages: None,
            });
        let backend = Rc::new(HostServicesV1Backend::new(FakeHostDeviceV1::new(), config));
        let table = HostServicesTableV1::new(backend);
        let host = table.caller();
        serving_ranks(host);
        host.acquire_serving_kv(LAYOUT_ID).map_err(status)
    };
    let full = capacity(0).unwrap();
    assert_eq!(capacity(PAGE_BYTES * 3).unwrap(), full - 3);
    assert_eq!(
        capacity(17 * 1024 * 1024),
        Err(BundleStatusV1::ResourceExhausted)
    );
}

/// Bytes of declared request state per slot per rank in the state tests.
const STATE_SLOT_BYTES: u64 = 64 * 1024;

fn manifest_id(value: &str) -> runtime_bundle::BundleOpaqueId {
    runtime_bundle::BundleOpaqueId::new(value).unwrap()
}

/// A manifest that declares request state, allocated during create before the
/// serving KV pool when `preallocated`.
fn recurrent_manifest(preallocated: bool) -> runtime_bundle::BundleManifestV1 {
    use runtime_bundle::{
        BundleComponentV1, BundleGraphBucketV1, BundleKvLayoutV1, BundleManifestV1,
        BundleRankPlacementV1, BundleRelativePathV1, BundleResourceRequirementsV1,
        BundleStateLayoutV1, BundleStateLifetimeV1, BundleTargetV1, BundleTopologyV1,
    };
    let mut capabilities = vec![
        manifest_id(runtime_bundle_abi::SERVING_KV_CAPABILITY_V1),
        manifest_id(runtime_bundle_abi::RECURRENT_STATE_CAPABILITY_V1),
    ];
    if preallocated {
        capabilities.push(manifest_id(
            runtime_bundle_abi::RECURRENT_STATE_PREALLOCATED_CAPABILITY_V1,
        ));
    }
    BundleManifestV1::new(
        manifest_id("fixture.state.v1"),
        BundleTargetV1::new(
            vec![manifest_id("sm_120f")],
            vec![BundleTopologyV1::new(manifest_id("tp2"), RANKS).unwrap()],
        )
        .unwrap(),
        vec![
            BundleComponentV1::new(
                manifest_id("weights"),
                manifest_id("fixture.weights.v1"),
                manifest_id("fixture.bytes.v1"),
                BundleRelativePathV1::new("components/weights.bin").unwrap(),
                16,
                "0".repeat(64),
            )
            .unwrap(),
        ],
        BundleResourceRequirementsV1::new(1, capabilities).unwrap(),
        BundleKvLayoutV1::new(
            manifest_id("fixture.kv.v1"),
            PAGE_BYTES,
            PAGE_ALIGNMENT,
            PAGE_TOKENS as u32,
            PAGE_BYTES / PAGE_TOKENS,
            1,
            1024,
            BundleRankPlacementV1::RankLocal,
            BundleStateLifetimeV1::Transaction,
        )
        .unwrap(),
        BundleStateLayoutV1::new(
            manifest_id("fixture.ring.v1"),
            64,
            64,
            BundleStateLifetimeV1::Process,
        )
        .unwrap(),
        vec![
            BundleGraphBucketV1::new(
                manifest_id("bucket.one"),
                vec![manifest_id("weights")],
                Vec::new(),
            )
            .unwrap(),
        ],
        Vec::new(),
        Vec::new(),
        None,
    )
    .unwrap()
    .with_recurrent_state_layouts(vec![
        BundleStateLayoutV1::new(
            manifest_id("fixture.request.v1"),
            STATE_SLOT_BYTES,
            64,
            BundleStateLifetimeV1::Request,
        )
        .unwrap(),
    ])
    .unwrap()
}

/// Opens serving KV the way a bundle whose state takes `state_bytes` per rank
/// does: allocating that state before or after it acquires the pool.
fn serving_pages_beside_state(
    manifest: &runtime_bundle::BundleManifestV1,
    sequences: usize,
    allocate_first: bool,
    maximum_pages: Option<u64>,
) -> u64 {
    let slots = runtime_host::recurrent::StateSlotPoolV1::new(manifest, sequences, 1).unwrap();
    let config = HostInstanceConfigV1::new(HostTopologyV1::new(vec![0, 1]).unwrap())
        .with_kv_contract(DeclaredKvContractV1::new(LAYOUT_ID.to_vec(), page_layout()))
        .with_serving_kv(ServingKvPolicyV1 {
            sampling_vocabulary: None,
            maximum_context_tokens: None,
            state_reserved_bytes: slots.reserved_bytes_per_rank(),
            utilization_millionths: 1_000_000,
            maximum_pages,
        });
    let backend = Rc::new(HostServicesV1Backend::new(FakeHostDeviceV1::new(), config));
    let table = HostServicesTableV1::new(backend);
    let host = table.caller();
    // The committed state plus one tentative checkpoint per slot.
    let state_bytes = STATE_SLOT_BYTES * 2 * sequences as u64;
    let ranks: Vec<_> = (0..RANKS)
        .map(|rank| {
            let context = host.select_context(rank, rank as i32).unwrap();
            let pool = host.select_memory_pool(context).unwrap();
            (pool, host.create_stream(context, 0).unwrap())
        })
        .collect();
    let allocate_state = || {
        for (pool, stream) in &ranks {
            host.allocate_device_buffer(*pool, *stream, state_bytes, 64)
                .unwrap();
        }
    };
    if allocate_first {
        allocate_state();
    }
    let pages = host.acquire_serving_kv(LAYOUT_ID).map_err(status).unwrap();
    if !allocate_first {
        allocate_state();
    }
    pages
}

#[test]
fn request_state_is_charged_once_whether_allocated_before_or_after_serving_kv() {
    // The fake device leaves 16 MiB per rank past the host's 1 GiB reserve.
    let spare = 16 * 1024 * 1024;
    for sequences in [16, 32] {
        let state_bytes = STATE_SLOT_BYTES * 2 * sequences as u64;
        let expected = (spare - state_bytes) / PAGE_BYTES;
        assert_eq!(
            serving_pages_beside_state(&recurrent_manifest(true), sequences, true, None),
            expected,
            "state allocated before the pool is already outside free memory"
        );
        assert_eq!(
            serving_pages_beside_state(&recurrent_manifest(false), sequences, false, None),
            expected,
            "state allocated after the pool is reserved up front"
        );
        for preallocated in [true, false] {
            assert_eq!(
                serving_pages_beside_state(
                    &recurrent_manifest(preallocated),
                    sequences,
                    preallocated,
                    Some(1000),
                ),
                1000,
                "an explicit page cap still applies"
            );
        }
    }
}

#[test]
fn sampler_workspace_is_reserved_before_kv_and_reused() {
    let config = HostInstanceConfigV1::new(HostTopologyV1::new(vec![0, 1]).unwrap())
        .with_kv_contract(DeclaredKvContractV1::new(LAYOUT_ID.to_vec(), page_layout()))
        .with_serving_kv(ServingKvPolicyV1 {
            sampling_vocabulary: Some(33),
            maximum_context_tokens: None,
            state_reserved_bytes: 0,
            utilization_millionths: 1_000_000,
            maximum_pages: None,
        });
    let backend = Rc::new(HostServicesV1Backend::new(FakeHostDeviceV1::new(), config));
    let table = HostServicesTableV1::new(backend.clone());
    let host = table.caller();
    serving_ranks(host);
    // Independently rounded input, output, adjusted, order, partials, filter.
    let sampler_bytes = 262144 + 16384 + 32768 + 16384 + 8192 + 4194304;
    let expected = (16 * 1024 * 1024 - sampler_bytes) / PAGE_BYTES;
    assert_eq!(host.acquire_serving_kv(LAYOUT_ID).unwrap(), expected);
    assert_eq!(host.acquire_serving_kv(LAYOUT_ID).unwrap(), expected);
    assert_eq!(backend.serving_kv_pages(), Some(expected));
}

#[test]
fn sampler_capacity_failure_refuses_startup_before_any_kv_pool() {
    let config = HostInstanceConfigV1::new(HostTopologyV1::new(vec![0, 1]).unwrap())
        .with_kv_contract(DeclaredKvContractV1::new(LAYOUT_ID.to_vec(), page_layout()))
        .with_serving_kv(ServingKvPolicyV1 {
            sampling_vocabulary: Some(1 << 20),
            maximum_context_tokens: None,
            state_reserved_bytes: 0,
            utilization_millionths: 1_000_000,
            maximum_pages: Some(1),
        });
    let backend = Rc::new(HostServicesV1Backend::new(FakeHostDeviceV1::new(), config));
    let table = HostServicesTableV1::new(backend.clone());
    let host = table.caller();
    serving_ranks(host);
    assert_eq!(
        host.acquire_serving_kv(LAYOUT_ID).map_err(status),
        Err(BundleStatusV1::ResourceExhausted)
    );
    assert_eq!(backend.serving_kv_pages(), None);
}

#[test]
fn host_placed_views_leave_their_bytes_to_the_serving_kv_pool() {
    const RANK_BYTES: u64 = 4 << 20;
    const HOST_BYTES: u64 = 2 << 20;
    let capacity = |host: bool| {
        let ranks = (0..u64::from(RANKS))
            .map(|rank| CompiledRankImageV1 {
                offset: rank * RANK_BYTES,
                byte_length: RANK_BYTES,
            })
            .collect();
        let spans = (0..RANKS)
            .map(|rank| {
                (
                    rank,
                    CompiledViewSpanV1 {
                        offset: 1 << 20,
                        byte_length: HOST_BYTES,
                    },
                )
            })
            .collect();
        let view = CompiledImageViewV1::new(64, spans);
        let mut views = BTreeMap::new();
        views.insert(b"table".to_vec(), if host { view.on_host() } else { view });
        let config = HostInstanceConfigV1::new(HostTopologyV1::new(vec![0, 1]).unwrap())
            .with_compiled_image(
                COMPONENT_ROLE,
                vec![3; (RANK_BYTES * u64::from(RANKS)) as usize],
                CompiledImageDescriptionV1::new(64, ranks, views),
            )
            .unwrap()
            .with_kv_contract(DeclaredKvContractV1::new(LAYOUT_ID.to_vec(), page_layout()))
            .with_serving_kv(ServingKvPolicyV1 {
                sampling_vocabulary: None,
                maximum_context_tokens: None,
                state_reserved_bytes: 0,
                utilization_millionths: 1_000_000,
                maximum_pages: None,
            });
        let backend = Rc::new(HostServicesV1Backend::new(FakeHostDeviceV1::new(), config));
        let table = HostServicesTableV1::new(backend);
        let host = table.caller();
        let image = host.open_compiled_image(COMPONENT_ROLE).unwrap();
        for rank in 0..RANKS {
            let context = host.select_context(rank, rank as i32).unwrap();
            let pool = host.select_memory_pool(context).unwrap();
            let stream = host.create_stream(context, 0).unwrap();
            host.upload_rank_arena(context, pool, stream, image, rank)
                .unwrap();
        }
        host.acquire_serving_kv(LAYOUT_ID).unwrap()
    };
    // The pool is sized from each rank's free device memory after the
    // upload, which the host view no longer takes.
    assert_eq!(capacity(true) - capacity(false), HOST_BYTES / PAGE_BYTES);
}
