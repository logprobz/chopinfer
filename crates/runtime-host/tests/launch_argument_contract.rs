//! What the host does with each declared launch argument encoding.
//!
//! Every encoding V1 defines is checked here on the way from a bundle's span to
//! the backend's resolved argument: which handles are resolved, which ranges are
//! bounds checked, which scalars keep their width, and which descriptors are
//! refused before any device work is ordered. Nothing here names a model.
//!
//! Encoding a tensor map needs a driver, so what runs here is the half that can
//! run anywhere: the resolution and the checks. The repository contract holds
//! the CUDA backend to encoding what survives them.

use std::rc::Rc;

use runtime_bundle_abi::{
    BundleAbiError, BundleContextHandleV1, BundleCudaSymbolHandleV1, BundleDeviceBufferHandleV1,
    BundleMemoryPoolHandleV1, BundleStatusV1, BundleStreamHandleV1, DeviceBufferSpanV1,
    HostServicesTableV1, KernelArgumentV1, LaunchClusterV1, LaunchConfigV1, TensorMapElementV1,
    TensorMapSwizzleV1, TensorMapV1,
};
use runtime_host::fake::FakeHostDeviceV1;
use runtime_host::{
    DeclaredKvContractV1, DeviceArgumentV1, HostInstanceConfigV1, HostServicesV1Backend,
    HostTopologyV1,
};
use runtime_kv::{DeclaredPageLayoutV1, RankPlacementV1, StateLifetimeV1};

const MODULE: &[u8] = b"fake-module-image";
const SYMBOL: &[u8] = b"fake_entry_v1";
/// The span every argument below is cut from, in bytes.
const ARENA_BYTES: u64 = 65_536;

fn page_layout() -> DeclaredPageLayoutV1 {
    DeclaredPageLayoutV1::new(
        64,
        2,
        8,
        64,
        1,
        RankPlacementV1::Sharded,
        StateLifetimeV1::Request,
        64,
    )
    .expect("the test page layout is valid")
}

fn config() -> LaunchConfigV1 {
    LaunchConfigV1 {
        flags: 0,
        grid_x: 1,
        grid_y: 1,
        grid_z: 1,
        block_x: 1,
        block_y: 1,
        block_z: 1,
        shared_memory_bytes: 0,
        cluster: LaunchClusterV1::NONE,
    }
}

/// One host with a context, a pool, a stream, a symbol, and one allocation.
struct HarnessV1 {
    table: HostServicesTableV1,
    backend: Rc<HostServicesV1Backend<FakeHostDeviceV1>>,
    stream: BundleStreamHandleV1,
    symbol: BundleCudaSymbolHandleV1,
    buffer: BundleDeviceBufferHandleV1,
}

fn harness() -> HarnessV1 {
    let config = HostInstanceConfigV1::new(
        HostTopologyV1::new(vec![0]).expect("the test topology is valid"),
    )
    .with_kv_contract(DeclaredKvContractV1::new(
        b"paged-kv-layout".to_vec(),
        page_layout(),
    ));
    let backend = Rc::new(HostServicesV1Backend::new(FakeHostDeviceV1::new(), config));
    let table = HostServicesTableV1::new(Rc::clone(&backend));
    let services = table.caller();
    let context = services.select_context(0, 0).expect("a context");
    let pool = services.select_memory_pool(context).expect("a pool");
    let stream = services.create_stream(context, 0).expect("a stream");
    let buffer = services
        .allocate_device_buffer(pool, stream, ARENA_BYTES, 256)
        .expect("an allocation");
    let module = services
        .load_cuda_module(context, MODULE)
        .expect("a module");
    let symbol = services
        .resolve_cuda_symbol(module, SYMBOL)
        .expect("a symbol");
    HarnessV1 {
        table,
        backend,
        stream,
        symbol,
        buffer,
    }
}

impl HarnessV1 {
    fn span(&self, offset: u64, length: u64) -> DeviceBufferSpanV1 {
        DeviceBufferSpanV1::new(self.buffer, offset, length)
    }

    /// Launches one argument list and returns whatever refused it.
    ///
    /// The fake backend runs no kernel, so a list that survives every check
    /// still ends as `Unsupported`. That is the signal the arguments reached it.
    fn launch(&self, arguments: &[KernelArgumentV1]) -> BundleAbiError {
        self.table
            .caller()
            .launch_cuda_symbol(self.stream, self.symbol, config(), arguments)
            .expect_err("a fake backend never reports a kernel as run")
    }

    /// The status the host itself answered, for a list the ABI let through.
    fn host_status(&self, arguments: &[KernelArgumentV1]) -> BundleStatusV1 {
        match self.launch(arguments) {
            BundleAbiError::Status(status) => status,
            other => panic!("the ABI refused before the host was asked: {other:?}"),
        }
    }

    fn recorded(&self) -> Vec<Vec<DeviceArgumentV1>> {
        self.backend
            .with_device(|device| device.refused_launch_arguments().to_vec())
    }
}

/// One rank-two encoding over a 32 KiB span.
fn tensor_map(buffer: DeviceBufferSpanV1) -> TensorMapV1 {
    TensorMapV1 {
        buffer,
        rank: 2,
        element_type: TensorMapElementV1::U8,
        swizzle: TensorMapSwizzleV1::Bytes128,
        interleave: 0,
        l2_promotion: 3,
        oob_fill: 0,
        global_dims: [1_024, 32, 0, 0, 0],
        global_strides_bytes: [1_024, 0, 0, 0],
        box_dims: [128, 64, 0, 0, 0],
        element_strides: [1, 1, 0, 0, 0],
    }
}

#[test]
fn every_declared_encoding_reaches_the_backend_as_the_width_it_named() {
    let harness = harness();
    let span = harness.span(0, 32_768);
    let status = harness.host_status(&[
        KernelArgumentV1::DeviceBuffer(span),
        KernelArgumentV1::NullDevicePointer,
        KernelArgumentV1::ScalarU32(u32::MAX),
        // A value no 32-bit slot could carry proves the 64-bit path is not
        // narrowed on the way through.
        KernelArgumentV1::ScalarU64(u64::from(u32::MAX) + 1),
        KernelArgumentV1::TensorMap(tensor_map(span)),
        // A negative value proves the signed 32-bit path keeps its sign rather
        // than arriving as the unsigned pattern with the same bits.
        KernelArgumentV1::ScalarI32(i32::MIN),
        KernelArgumentV1::DevicePointerArray(vec![
            Some(harness.span(0, 16)),
            None,
            Some(harness.span(16, 16)),
        ]),
    ]);
    assert_eq!(status, BundleStatusV1::Unsupported);

    let recorded = harness.recorded();
    let arguments = recorded.first().expect("one launch is recorded");
    assert_eq!(arguments.len(), 7);
    // The span became a resolved range with its own bounds already checked.
    let DeviceArgumentV1::Range(range) = &arguments[0] else {
        panic!("the first argument is a resolved range");
    };
    assert_eq!((range.offset, range.length), (0, 32_768));
    assert_eq!(arguments[1], DeviceArgumentV1::NullDevicePointer);
    assert_eq!(arguments[2], DeviceArgumentV1::ScalarU32(u32::MAX));
    assert_eq!(
        arguments[3],
        DeviceArgumentV1::ScalarU64(u64::from(u32::MAX) + 1)
    );
    let DeviceArgumentV1::TensorMap(map) = &arguments[4] else {
        panic!("the last argument is a tensor map");
    };
    // The descriptor arrived intact, with the span it named resolved.
    assert_eq!(map.element_type, TensorMapElementV1::U8);
    assert_eq!(map.swizzle, TensorMapSwizzleV1::Bytes128);
    assert_eq!(map.global_dims, [1_024, 32, 0, 0, 0]);
    assert_eq!(map.global_strides_bytes[0], 1_024);
    assert_eq!(map.box_dims, [128, 64, 0, 0, 0]);
    assert_eq!((map.range.offset, map.range.length), (0, 32_768));
    assert_eq!(map.range.object, range.object);
    assert_eq!(arguments[5], DeviceArgumentV1::ScalarI32(i32::MIN));
    // Every slot arrived as a resolved range, in slot order, with the null one
    // still null rather than collapsed away.
    let DeviceArgumentV1::DevicePointerArray(slots) = &arguments[6] else {
        panic!("the last argument is a by-value address array");
    };
    assert_eq!(slots.len(), 3);
    assert_eq!(
        slots[0].map(|slot| (slot.object, slot.offset, slot.length)),
        Some((range.object, 0, 16))
    );
    assert_eq!(slots[1], None);
    assert_eq!(
        slots[2].map(|slot| (slot.object, slot.offset, slot.length)),
        Some((range.object, 16, 16))
    );
}

#[test]
fn an_address_array_slot_reaching_past_its_allocation_reaches_no_backend() {
    let harness = harness();
    let error = harness.launch(&[KernelArgumentV1::DevicePointerArray(vec![
        Some(harness.span(0, 16)),
        Some(harness.span(ARENA_BYTES, 16)),
    ])]);
    assert_eq!(
        error,
        BundleAbiError::Status(BundleStatusV1::InvalidArgument)
    );
    assert!(harness.recorded().is_empty());
}

#[test]
fn an_address_array_outside_its_own_bound_never_leaves_the_abi() {
    let harness = harness();
    // Empty and over-long arrays are refused at the boundary, before the host
    // resolves a handle, because the slot count is what makes this bounded.
    assert!(matches!(
        harness.launch(&[KernelArgumentV1::DevicePointerArray(Vec::new())]),
        BundleAbiError::InvalidDevicePointerArraySlots(0)
    ));
    let too_many = (0..9).map(|_| Some(harness.span(0, 16))).collect();
    assert!(matches!(
        harness.launch(&[KernelArgumentV1::DevicePointerArray(too_many)]),
        BundleAbiError::InvalidDevicePointerArraySlots(9)
    ));
    assert!(harness.recorded().is_empty());
}

#[test]
fn a_pointer_table_holds_exactly_one_entry_per_slot() {
    let harness = harness();
    let services = harness.table.caller();
    // Eight bytes per slot, and the destination has to be exactly that long.
    services
        .write_device_pointer_table(
            harness.stream,
            harness.span(0, 16),
            &[Some(harness.span(32, 16)), None],
        )
        .expect("a destination of exactly two entries is written");
    for (label, destination, slots) in [
        ("a destination one entry short", harness.span(0, 8), 2),
        ("a destination one entry long", harness.span(0, 24), 2),
    ] {
        let slots: Vec<_> = (0..slots).map(|_| Some(harness.span(32, 16))).collect();
        assert_eq!(
            services.write_device_pointer_table(harness.stream, destination, &slots),
            Err(BundleAbiError::Status(BundleStatusV1::InvalidArgument)),
            "{label} must be refused"
        );
    }
    // An empty table is refused at the boundary for the same reason an empty
    // by-value array is: the slot count is the whole of what bounds it.
    assert!(matches!(
        services.write_device_pointer_table(harness.stream, harness.span(0, 16), &[]),
        Err(BundleAbiError::InvalidDevicePointerArraySlots(0))
    ));
    // And a slot reaching past its allocation is refused before any write.
    assert_eq!(
        services.write_device_pointer_table(
            harness.stream,
            harness.span(0, 8),
            &[Some(harness.span(ARENA_BYTES, 16))],
        ),
        Err(BundleAbiError::Status(BundleStatusV1::InvalidArgument))
    );
}

#[test]
fn a_tensor_map_over_a_span_the_host_did_not_issue_reaches_no_backend() {
    let harness = harness();
    let stale = DeviceBufferSpanV1::new(BundleDeviceBufferHandleV1::from_parts(1, 99), 0, 32_768);
    assert_eq!(
        harness.host_status(&[KernelArgumentV1::TensorMap(tensor_map(stale))]),
        BundleStatusV1::StaleHandle
    );
    assert!(
        harness.recorded().is_empty(),
        "a stale handle must order no launch"
    );
}

/// Each declared element reaches the fake backend as its own code.
#[test]
fn every_tensor_map_element_reaches_the_backend() {
    for element_type in [
        TensorMapElementV1::U8,
        TensorMapElementV1::U16,
        TensorMapElementV1::I32,
        TensorMapElementV1::Bf16,
        TensorMapElementV1::F16,
        TensorMapElementV1::F32,
        TensorMapElementV1::Fp8E4M3,
        TensorMapElementV1::Fp8E5M2,
        TensorMapElementV1::U4Align8B,
        TensorMapElementV1::U4Align16B,
        TensorMapElementV1::U6Align16B,
    ] {
        let harness = harness();
        let mut map = tensor_map(harness.span(0, 32_768));
        map.element_type = element_type;
        // The width the element is read at changes the row, so the map is cut
        // to one the declared stride still holds.
        map.global_dims[0] = 128;
        map.box_dims[0] = 16;
        if element_type == TensorMapElementV1::U4Align8B {
            map.box_dims[0] = 32;
        }
        if matches!(
            element_type,
            TensorMapElementV1::U4Align16B | TensorMapElementV1::U6Align16B
        ) {
            map.box_dims[0] = 128;
            map.global_strides_bytes[0] = 1_024;
        }
        assert_eq!(
            harness.host_status(&[KernelArgumentV1::TensorMap(map)]),
            BundleStatusV1::Unsupported,
            "{element_type:?} is an encoding a host carries"
        );
        assert_eq!(
            harness.recorded().len(),
            1,
            "{element_type:?} must reach the backend, which then runs no kernel"
        );
    }
}

#[test]
fn a_tensor_map_reaching_past_its_allocation_reaches_no_backend() {
    let harness = harness();
    // The span fits no allocation this host issued, which only the host knows,
    // so the descriptor is well formed and the host is the one that refuses it.
    // The span is the right length for the map, and starts so near the end of
    // the allocation that it runs past it.
    let past_end = harness.span(ARENA_BYTES - 16, 32_768);
    assert_eq!(
        harness.host_status(&[KernelArgumentV1::TensorMap(tensor_map(past_end))]),
        BundleStatusV1::InvalidArgument
    );
    assert!(
        harness.recorded().is_empty(),
        "an unresolvable span must order no launch"
    );
}

#[test]
fn a_descriptor_that_contradicts_itself_never_reaches_the_host() {
    let harness = harness();
    // Each of these is a claim the descriptor makes about itself, so the ABI
    // settles it before any handle is resolved and no host is asked at all.
    let mut oversized = tensor_map(harness.span(0, 32_768));
    oversized.global_dims[1] = 64;
    let mut narrow_stride = tensor_map(harness.span(0, 32_768));
    narrow_stride.global_dims[0] = 2_048;
    let misaligned = tensor_map(harness.span(8, 32_768));
    let mut overflowing = tensor_map(harness.span(0, 32_768));
    overflowing.global_strides_bytes[0] = 1 << 63;
    overflowing.global_dims[1] = 4;

    for (surface, map) in [
        ("footprint", oversized),
        ("row width", narrow_stride),
        ("base alignment", misaligned),
        ("stride", overflowing),
    ] {
        assert!(
            matches!(
                harness.launch(&[KernelArgumentV1::TensorMap(map)]),
                BundleAbiError::InvalidTensorMapGeometry(actual) if actual == surface
            ),
            "a broken {surface} must be refused at the boundary"
        );
    }
    assert!(
        harness.recorded().is_empty(),
        "no refused descriptor may order a launch"
    );
}

#[test]
fn opaque_bytes_stay_unsupported_at_the_launch_boundary() {
    let harness = harness();
    // The argument is well formed and crosses the ABI, and the backend then
    // reports that this version declares no launch encoding for it.
    assert_eq!(
        harness.host_status(&[KernelArgumentV1::OpaqueBytes(vec![1, 2, 3])]),
        BundleStatusV1::Unsupported
    );
    assert!(
        harness.recorded().is_empty(),
        "an argument with no launch encoding orders nothing"
    );
}

#[test]
fn a_refusal_anywhere_in_the_list_stops_the_whole_launch() {
    // The invariant a driver backend owes is that nothing is built or encoded
    // until the whole launch is known to be runnable. Here the first argument
    // is a well formed tensor map and the second is one that cannot be
    // resolved, so a backend that acted argument by argument would already have
    // encoded the first by the time it learned about the second.
    let harness = harness();
    let good = harness.span(0, 32_768);
    let stale = DeviceBufferSpanV1::new(BundleDeviceBufferHandleV1::from_parts(1, 99), 0, 32_768);
    assert_eq!(
        harness.host_status(&[
            KernelArgumentV1::TensorMap(tensor_map(good)),
            KernelArgumentV1::DeviceBuffer(stale),
        ]),
        BundleStatusV1::StaleHandle
    );
    assert!(harness.recorded().is_empty());

    // The same, with the unresolvable argument first and a map after it.
    assert_eq!(
        harness.host_status(&[
            KernelArgumentV1::DeviceBuffer(stale),
            KernelArgumentV1::TensorMap(tensor_map(good)),
        ]),
        BundleStatusV1::StaleHandle
    );
    assert!(harness.recorded().is_empty());

    // An argument this version carries no launch encoding for stops the launch
    // wherever it sits, so a map beside it is never encoded either.
    assert_eq!(
        harness.host_status(&[
            KernelArgumentV1::TensorMap(tensor_map(good)),
            KernelArgumentV1::OpaqueBytes(vec![1, 2, 3]),
        ]),
        BundleStatusV1::Unsupported
    );
    assert!(harness.recorded().is_empty());
}

/// Two ranks, each with its own context, pool, stream, and one allocation.
///
/// Everything the cross-context rule needs is separate here: the consumer that
/// launches, the owner whose memory a slot names, the directed lease between
/// their contexts, and the grant on the owner's pool.
struct CrossHarnessV1 {
    backend: Rc<HostServicesV1Backend<FakeHostDeviceV1>>,
    table: HostServicesTableV1,
    consumer: BundleContextHandleV1,
    owner: BundleContextHandleV1,
    consumer_pool: BundleMemoryPoolHandleV1,
    owner_pool: BundleMemoryPoolHandleV1,
    consumer_stream: BundleStreamHandleV1,
    owner_stream: BundleStreamHandleV1,
    consumer_buffer: BundleDeviceBufferHandleV1,
    owner_buffer: BundleDeviceBufferHandleV1,
    symbol: BundleCudaSymbolHandleV1,
}

fn cross_harness() -> CrossHarnessV1 {
    let instance = HostInstanceConfigV1::new(
        HostTopologyV1::new(vec![0, 1]).expect("the two-rank topology is valid"),
    );
    let backend = Rc::new(HostServicesV1Backend::new(
        FakeHostDeviceV1::new(),
        instance,
    ));
    let table = HostServicesTableV1::new(Rc::clone(&backend));
    let services = table.caller();
    let consumer = services.select_context(0, 0).expect("the consumer context");
    let owner = services.select_context(1, 1).expect("the owner context");
    let consumer_pool = services
        .select_memory_pool(consumer)
        .expect("the consumer pool");
    let owner_pool = services.select_memory_pool(owner).expect("the owner pool");
    let consumer_stream = services
        .create_stream(consumer, 0)
        .expect("the consumer stream");
    let owner_stream = services.create_stream(owner, 0).expect("the owner stream");
    let consumer_buffer = services
        .allocate_device_buffer(consumer_pool, consumer_stream, 64, 16)
        .expect("the consumer allocation");
    let owner_buffer = services
        .allocate_device_buffer(owner_pool, owner_stream, 64, 16)
        .expect("the owner allocation");
    let module = services
        .load_cuda_module(consumer, MODULE)
        .expect("a module in the consumer context");
    let symbol = services
        .resolve_cuda_symbol(module, SYMBOL)
        .expect("a symbol");
    CrossHarnessV1 {
        backend,
        table,
        consumer,
        owner,
        consumer_pool,
        owner_pool,
        consumer_stream,
        owner_stream,
        consumer_buffer,
        owner_buffer,
        symbol,
    }
}

impl CrossHarnessV1 {
    fn services(&self) -> &runtime_bundle_abi::HostServicesV1 {
        self.table.caller()
    }

    fn span(&self, buffer: BundleDeviceBufferHandleV1) -> DeviceBufferSpanV1 {
        DeviceBufferSpanV1::new(buffer, 0, 64)
    }

    fn reached_backend(&self) -> bool {
        !self
            .backend
            .with_device(|device| device.refused_launch_arguments().to_vec())
            .is_empty()
    }

    /// Launches one by-value slot naming the owner's allocation.
    fn launch_owner_slot(&self) -> Result<(), BundleAbiError> {
        self.services().launch_cuda_symbol(
            self.consumer_stream,
            self.symbol,
            config(),
            &[KernelArgumentV1::DevicePointerArray(vec![Some(
                self.span(self.owner_buffer),
            )])],
        )
    }
}

#[test]
fn a_cross_context_address_needs_both_the_lease_and_the_grant() {
    // Neither half implies the other: the lease is the consumer's right to
    // reach that context, the grant is the owner pool's consent to be reached
    // from it, and an address missing either would fault.
    let neither = cross_harness();
    assert_eq!(
        neither.launch_owner_slot(),
        Err(BundleAbiError::Status(BundleStatusV1::InvalidState)),
        "neither half is enough"
    );
    assert!(!neither.reached_backend());

    let lease_only = cross_harness();
    let lease = lease_only
        .services()
        .acquire_peer_access(lease_only.consumer, lease_only.owner)
        .expect("the directed lease");
    assert_eq!(
        lease_only.launch_owner_slot(),
        Err(BundleAbiError::Status(BundleStatusV1::InvalidState)),
        "a lease without the grant is not enough"
    );
    assert!(!lease_only.reached_backend());

    let grant_only = cross_harness();
    grant_only
        .services()
        .set_memory_pool_peer_access(grant_only.owner_pool, grant_only.consumer, true)
        .expect("the pool grant");
    assert_eq!(
        grant_only.launch_owner_slot(),
        Err(BundleAbiError::Status(BundleStatusV1::InvalidState)),
        "a grant without the lease is not enough"
    );
    assert!(!grant_only.reached_backend());

    // Both, and the same argument reaches the backend.
    lease_only
        .services()
        .set_memory_pool_peer_access(lease_only.owner_pool, lease_only.consumer, true)
        .expect("the pool grant");
    assert_eq!(
        lease_only.launch_owner_slot(),
        Err(BundleAbiError::Status(BundleStatusV1::Unsupported)),
        "both halves let the address through"
    );
    assert!(lease_only.reached_backend());
    lease_only
        .services()
        .release_peer_access(lease)
        .expect("the lease goes back");
}

#[test]
fn peer_dma_requires_the_directed_lease_and_pool_grant() {
    let h = cross_harness();
    let copy = || {
        h.services().copy_peer_to_peer(
            h.consumer_stream,
            h.consumer,
            h.span(h.consumer_buffer),
            h.owner,
            h.span(h.owner_buffer),
        )
    };
    assert_eq!(
        copy(),
        Err(BundleAbiError::Status(BundleStatusV1::InvalidState))
    );
    let lease = h
        .services()
        .acquire_peer_access(h.consumer, h.owner)
        .unwrap();
    assert_eq!(
        copy(),
        Err(BundleAbiError::Status(BundleStatusV1::InvalidState))
    );
    h.services()
        .set_memory_pool_peer_access(h.owner_pool, h.consumer, true)
        .unwrap();
    copy().unwrap();
    h.services().release_peer_access(lease).unwrap();
    assert_eq!(
        copy(),
        Err(BundleAbiError::Status(BundleStatusV1::InvalidState))
    );
}

#[test]
fn peer_dma_contexts_must_own_their_spans_and_destination_stream() {
    let h = cross_harness();
    let lease = h
        .services()
        .acquire_peer_access(h.consumer, h.owner)
        .unwrap();
    h.services()
        .set_memory_pool_peer_access(h.owner_pool, h.consumer, true)
        .unwrap();
    for (stream, destination, source) in [
        (h.owner_stream, h.consumer, h.owner),
        (h.consumer_stream, h.owner, h.owner),
        (h.consumer_stream, h.consumer, h.consumer),
    ] {
        assert_eq!(
            h.services().copy_peer_to_peer(
                stream,
                destination,
                h.span(h.consumer_buffer),
                source,
                h.span(h.owner_buffer),
            ),
            Err(BundleAbiError::Status(BundleStatusV1::InvalidArgument))
        );
    }
    h.services().release_peer_access(lease).unwrap();
}

#[test]
fn a_lease_or_grant_pointing_the_wrong_way_is_not_the_one_needed() {
    let harness = cross_harness();
    // The lease from the owner to the consumer is a different edge from the one
    // a launch in the consumer needs, and so is a grant on the consumer's own
    // pool rather than the owner's.
    let wrong_lease = harness
        .services()
        .acquire_peer_access(harness.owner, harness.consumer)
        .expect("the reversed lease");
    harness
        .services()
        .set_memory_pool_peer_access(harness.consumer_pool, harness.owner, true)
        .expect("the grant on the wrong pool, for the wrong peer");
    assert_eq!(
        harness.launch_owner_slot(),
        Err(BundleAbiError::Status(BundleStatusV1::InvalidState))
    );
    assert!(!harness.reached_backend());

    // The right lease with a grant on the consumer's own pool is still wrong:
    // the grant has to be on the pool the address came from.
    let right_lease = harness
        .services()
        .acquire_peer_access(harness.consumer, harness.owner)
        .expect("the directed lease");
    assert_eq!(
        harness.launch_owner_slot(),
        Err(BundleAbiError::Status(BundleStatusV1::InvalidState))
    );
    assert!(!harness.reached_backend());

    harness
        .services()
        .release_peer_access(right_lease)
        .expect("the newest lease goes back first");
    harness
        .services()
        .release_peer_access(wrong_lease)
        .expect("then the other");
}

#[test]
fn the_consumers_own_allocation_needs_neither() {
    let harness = cross_harness();
    assert_eq!(
        harness.services().launch_cuda_symbol(
            harness.consumer_stream,
            harness.symbol,
            config(),
            &[KernelArgumentV1::DevicePointerArray(vec![Some(
                harness.span(harness.consumer_buffer)
            )])],
        ),
        Err(BundleAbiError::Status(BundleStatusV1::Unsupported))
    );
    assert!(harness.reached_backend());
}

#[test]
fn a_launch_on_another_contexts_stream_is_refused_before_anything_resolves() {
    let harness = cross_harness();
    // The symbol lives in the consumer's module, so the owner's stream would
    // order it in a context the symbol is not in.
    assert_eq!(
        harness.services().launch_cuda_symbol(
            harness.owner_stream,
            harness.symbol,
            config(),
            &[KernelArgumentV1::DevicePointerArray(vec![Some(
                harness.span(harness.consumer_buffer)
            )])],
        ),
        Err(BundleAbiError::Status(BundleStatusV1::InvalidArgument))
    );
    assert!(!harness.reached_backend());
}

#[test]
fn a_pointer_table_applies_the_same_rule_to_its_stream_and_destination() {
    let harness = cross_harness();
    let services = harness.services();
    // The destination has to be the writing stream's own context.
    assert_eq!(
        services.write_device_pointer_table(
            harness.consumer_stream,
            DeviceBufferSpanV1::new(harness.owner_buffer, 0, 8),
            &[Some(harness.span(harness.consumer_buffer))],
        ),
        Err(BundleAbiError::Status(BundleStatusV1::InvalidArgument)),
        "a destination in another context is refused"
    );
    // And so is a table written on a stream of some other context than the one
    // that owns the destination.
    assert_eq!(
        services.write_device_pointer_table(
            harness.owner_stream,
            DeviceBufferSpanV1::new(harness.consumer_buffer, 0, 8),
            &[Some(harness.span(harness.consumer_buffer))],
        ),
        Err(BundleAbiError::Status(BundleStatusV1::InvalidArgument)),
        "a stream in another context is refused"
    );

    // A slot naming another context needs both halves, exactly as a launch
    // argument does, with the writing stream's context as the consumer.
    assert_eq!(
        services.write_device_pointer_table(
            harness.consumer_stream,
            DeviceBufferSpanV1::new(harness.consumer_buffer, 0, 8),
            &[Some(harness.span(harness.owner_buffer))],
        ),
        Err(BundleAbiError::Status(BundleStatusV1::InvalidState)),
        "neither half is enough"
    );
    let lease = services
        .acquire_peer_access(harness.consumer, harness.owner)
        .expect("the directed lease");
    assert_eq!(
        services.write_device_pointer_table(
            harness.consumer_stream,
            DeviceBufferSpanV1::new(harness.consumer_buffer, 0, 8),
            &[Some(harness.span(harness.owner_buffer))],
        ),
        Err(BundleAbiError::Status(BundleStatusV1::InvalidState)),
        "a lease without the grant is not enough"
    );
    services
        .set_memory_pool_peer_access(harness.owner_pool, harness.consumer, true)
        .expect("the pool grant");
    services
        .write_device_pointer_table(
            harness.consumer_stream,
            DeviceBufferSpanV1::new(harness.consumer_buffer, 0, 8),
            &[Some(harness.span(harness.owner_buffer))],
        )
        .expect("both halves let the address through");
    services
        .set_memory_pool_peer_access(harness.owner_pool, harness.consumer, false)
        .expect("the grant goes back");
    services
        .release_peer_access(lease)
        .expect("the lease goes back");
}
