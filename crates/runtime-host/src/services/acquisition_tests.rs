use super::*;
use crate::fake::{FakeHostDeviceV1, FakeRefusalV1};

type Services = HostServicesV1Backend<FakeHostDeviceV1>;

#[derive(Clone, Copy, Debug)]
enum Direct {
    Context,
    Pool,
    Buffer,
    Stream,
    StageStream,
    Event,
    Module,
    Symbol,
}

const DIRECT: [Direct; 8] = [
    Direct::Context,
    Direct::Pool,
    Direct::Buffer,
    Direct::Stream,
    Direct::StageStream,
    Direct::Event,
    Direct::Module,
    Direct::Symbol,
];

struct DirectFixture {
    services: Services,
    context: BundleContextHandleV1,
    other: BundleContextHandleV1,
    pool: BundleMemoryPoolHandleV1,
    stream: BundleStreamHandleV1,
    module: BundleCudaModuleHandleV1,
}

impl DirectFixture {
    fn new() -> Self {
        let services = Services::new(
            FakeHostDeviceV1::new(),
            HostInstanceConfigV1::new(HostTopologyV1::new(vec![0, 1, 2]).unwrap()),
        );
        let context = services.select_context(0, 0).unwrap();
        let other = services.select_context(1, 1).unwrap();
        let pool = services.select_memory_pool(context).unwrap();
        let stream = services.create_stream(context, 0).unwrap();
        let module = services
            .load_cuda_module(context, b"fixture-module")
            .unwrap();
        Self {
            services,
            context,
            other,
            pool,
            stream,
            module,
        }
    }

    fn release(self) {
        self.services.release_cuda_module(self.module).unwrap();
        self.services.release_stream(self.stream).unwrap();
        self.services.release_memory_pool(self.pool).unwrap();
        self.services.release_context(self.other).unwrap();
        self.services.release_context(self.context).unwrap();
        assert_eq!(self.services.live_resource_count(), 0);
        assert_eq!(
            self.services
                .with_device(FakeHostDeviceV1::live_object_count),
            0
        );
    }
}

macro_rules! key {
    ($result:expr) => {
        $result.map(|handle| SlotKeyV1 {
            generation: handle.generation(),
            slot: handle.slot(),
        })
    };
}

impl Direct {
    fn acquire(self, fixture: &DirectFixture) -> HostServiceResultV1<SlotKeyV1> {
        let s = &fixture.services;
        match self {
            Self::Context => key!(s.select_context(2, 2)),
            Self::Pool => key!(s.select_memory_pool(fixture.other)),
            Self::Buffer => key!(s.allocate_device_buffer(fixture.pool, fixture.stream, 64, 64)),
            Self::Stream => key!(s.create_stream(fixture.context, 0)),
            Self::StageStream => {
                key!(s.select_stage_stream(fixture.context, 0, b"fixture-stage", 0))
            }
            Self::Event => key!(s.create_event(fixture.context)),
            Self::Module => key!(s.load_cuda_module(fixture.context, b"fixture-module")),
            Self::Symbol => key!(s.resolve_cuda_symbol(fixture.module, b"fixture_entry")),
        }
    }

    fn release(self, fixture: &DirectFixture, key: SlotKeyV1) {
        let s = &fixture.services;
        match self {
            Self::Context => {
                s.release_context(BundleContextHandleV1::from_parts(key.generation, key.slot))
            }
            Self::Pool => s.release_memory_pool(BundleMemoryPoolHandleV1::from_parts(
                key.generation,
                key.slot,
            )),
            Self::Buffer => s.release_device_buffer(BundleDeviceBufferHandleV1::from_parts(
                key.generation,
                key.slot,
            )),
            Self::Stream | Self::StageStream => {
                s.release_stream(BundleStreamHandleV1::from_parts(key.generation, key.slot))
            }
            Self::Event => {
                s.release_event(BundleEventHandleV1::from_parts(key.generation, key.slot))
            }
            Self::Module => s.release_cuda_module(BundleCudaModuleHandleV1::from_parts(
                key.generation,
                key.slot,
            )),
            Self::Symbol => s.release_cuda_symbol(BundleCudaSymbolHandleV1::from_parts(
                key.generation,
                key.slot,
            )),
        }
        .unwrap();
    }

    fn constructor_refusal(self) -> FakeRefusalV1 {
        match self {
            Self::Context => FakeRefusalV1::CreateContext,
            Self::Pool => FakeRefusalV1::CreateMemoryPool,
            Self::Buffer => FakeRefusalV1::AllocateDeviceBuffer,
            Self::Stream | Self::StageStream => FakeRefusalV1::CreateStream,
            Self::Event => FakeRefusalV1::CreateEvent,
            Self::Module => FakeRefusalV1::LoadModule,
            Self::Symbol => FakeRefusalV1::ResolveSymbol,
        }
    }

    fn release_refusal(self) -> FakeRefusalV1 {
        match self {
            Self::Context => FakeRefusalV1::ReleaseContext,
            Self::Pool => FakeRefusalV1::ReleaseMemoryPool,
            Self::Buffer => FakeRefusalV1::ReleaseDeviceBuffer,
            Self::Stream | Self::StageStream => FakeRefusalV1::ReleaseStream,
            Self::Event => FakeRefusalV1::ReleaseEvent,
            Self::Module => FakeRefusalV1::ReleaseModule,
            Self::Symbol => FakeRefusalV1::ReleaseSymbol,
        }
    }

    fn refuse_slot(self, state: &mut HostStateV1<FakeHostDeviceV1>, refuse: bool) {
        match self {
            Self::Context => state.contexts.refuse_registration(refuse),
            Self::Pool => state.pools.refuse_registration(refuse),
            Self::Buffer => state.buffers.refuse_registration(refuse),
            Self::Stream | Self::StageStream => state.streams.refuse_registration(refuse),
            Self::Event => state.events.refuse_registration(refuse),
            Self::Module => state.modules.refuse_registration(refuse),
            Self::Symbol => state.symbols.refuse_registration(refuse),
        }
    }
}

fn registration_failure_creates_nothing(slot: bool) {
    for kind in DIRECT {
        let fixture = DirectFixture::new();
        let s = &fixture.services;
        let issued = s.with_device(FakeHostDeviceV1::issued_object_count);
        let live = s.with_device(FakeHostDeviceV1::live_object_count);
        let (next, order, caches) = {
            let state = s.state.borrow();
            (
                state.order.next,
                state.order.live.clone(),
                (
                    state.context_by_rank.len(),
                    state.pool_by_context.len(),
                    state.stage_streams.len(),
                ),
            )
        };
        s.with_device_mut(|device| device.refuse(kind.release_refusal()));
        if slot {
            kind.refuse_slot(&mut s.state.borrow_mut(), true);
        } else {
            s.state.borrow_mut().order.next = u64::MAX;
        }
        assert_eq!(
            kind.acquire(&fixture),
            Err(BundleStatusV1::ResourceExhausted),
            "{kind:?}"
        );
        assert_eq!(
            s.with_device(FakeHostDeviceV1::issued_object_count),
            issued,
            "{kind:?}: registration must precede backend construction"
        );
        assert_eq!(
            s.with_device(FakeHostDeviceV1::live_object_count),
            live,
            "{kind:?}"
        );
        {
            let mut state = s.state.borrow_mut();
            assert_eq!(state.order.live, order, "{kind:?}");
            assert_eq!(
                (
                    state.context_by_rank.len(),
                    state.pool_by_context.len(),
                    state.stage_streams.len()
                ),
                caches,
                "{kind:?}"
            );
            state.order.next = next;
            kind.refuse_slot(&mut state, false);
        }
        s.with_device_mut(|device| device.allow(kind.release_refusal()));
        let acquired = kind.acquire(&fixture).unwrap();
        kind.release(&fixture, acquired);
        let reused = kind.acquire(&fixture).unwrap();
        assert_eq!(acquired.slot, reused.slot);
        assert_ne!(acquired.generation, reused.generation);
        kind.release(&fixture, reused);
        fixture.release();
    }
}

#[test]
fn order_exhaustion_precedes_direct_backend_construction() {
    registration_failure_creates_nothing(false);
}

#[test]
fn slot_exhaustion_precedes_direct_backend_construction() {
    registration_failure_creates_nothing(true);
}

#[test]
fn direct_constructor_refusal_rolls_back_only_registration_metadata() {
    for kind in DIRECT {
        let fixture = DirectFixture::new();
        let s = &fixture.services;
        let issued = s.with_device(FakeHostDeviceV1::issued_object_count);
        let order = s.state.borrow().order.live.clone();
        s.with_device_mut(|device| {
            device.refuse(kind.constructor_refusal());
            device.refuse(kind.release_refusal());
        });
        assert_eq!(
            kind.acquire(&fixture),
            Err(BundleStatusV1::HostError),
            "{kind:?}"
        );
        assert_eq!(
            s.with_device(FakeHostDeviceV1::issued_object_count),
            issued,
            "{kind:?}"
        );
        assert_eq!(s.state.borrow().order.live, order, "{kind:?}");
        s.with_device_mut(|device| {
            device.allow(kind.constructor_refusal());
            device.allow(kind.release_refusal());
        });
        let acquired = kind.acquire(&fixture).unwrap();
        kind.release(&fixture, acquired);
        fixture.release();
    }
}

#[derive(Clone, Copy, Debug)]
enum HostBuffer {
    Owned,
    Lendable,
    Imported,
    Mapped,
}

const HOST_BUFFERS: [HostBuffer; 4] = [
    HostBuffer::Owned,
    HostBuffer::Lendable,
    HostBuffer::Imported,
    HostBuffer::Mapped,
];

impl HostBuffer {
    fn acquire(self, fixture: &DirectFixture) -> HostServiceResultV1<SlotKeyV1> {
        let mut state = fixture.services.state.borrow_mut();
        let context = slot_key_v1(fixture.context.generation(), fixture.context.slot())?;
        let pool = slot_key_v1(fixture.pool.generation(), fixture.pool.slot())?;
        let origin = BufferOriginV1 {
            context,
            pool,
            lent: matches!(self, Self::Imported),
        };
        let context = state.contexts.get(context)?.object;
        let pool = state.pools.get(pool)?.object;
        let stream = state.stream_object(fixture.stream)?;
        match self {
            Self::Owned => state.allocate_host_buffer(pool, None, Some(stream), 64, 64),
            Self::Lendable => state.allocate_lendable_host_buffer(context, stream, 64, 64, origin),
            Self::Imported => {
                let lent = crate::weight_keeper::LentImageV1 {
                    image: crate::weight_keeper::RankImageV1 {
                        image_role: "fixture.image".into(),
                        rank: 0,
                        identity: "0".repeat(64),
                        device: "fixture-device".into(),
                        byte_length: 64,
                    },
                    fd: crate::weight_keeper::fds::anonymous_file(&[0; 64], 64).unwrap(),
                    mapped_bytes: 64,
                    read_only: true,
                };
                state.import_host_buffer(context, stream, &lent, 64, origin)
            }
            Self::Mapped => {
                state.allocate_mapped_host_buffer(stream, Some(origin), 64, 64, &mut |bytes| {
                    bytes.fill(0);
                    true
                })
            }
        }
    }
}

#[cfg(target_os = "linux")]
#[test]
fn host_buffer_registration_and_constructor_refusal_create_no_orphans() {
    for kind in HOST_BUFFERS {
        let fixture = DirectFixture::new();
        let s = &fixture.services;
        let issued = s.with_device(FakeHostDeviceV1::issued_object_count);
        let live = s.with_device(FakeHostDeviceV1::live_object_count);
        s.with_device_mut(|device| device.refuse(FakeRefusalV1::ReleaseDeviceBuffer));
        s.state.borrow_mut().buffers.refuse_registration(true);
        assert_eq!(
            kind.acquire(&fixture),
            Err(BundleStatusV1::ResourceExhausted),
            "{kind:?}"
        );
        assert_eq!(
            s.with_device(FakeHostDeviceV1::issued_object_count),
            issued,
            "{kind:?}"
        );
        assert_eq!(s.state.borrow().buffers.live_count(), 0);
        s.state.borrow_mut().buffers.refuse_registration(false);
        s.with_device_mut(|device| device.refuse(FakeRefusalV1::AllocateDeviceBuffer));
        assert_eq!(
            kind.acquire(&fixture),
            Err(BundleStatusV1::HostError),
            "{kind:?}"
        );
        assert_eq!(
            s.with_device(FakeHostDeviceV1::issued_object_count),
            issued,
            "{kind:?}"
        );
        assert_eq!(s.state.borrow().buffers.live_count(), 0);
        s.with_device_mut(|device| {
            device.allow(FakeRefusalV1::AllocateDeviceBuffer);
            device.allow(FakeRefusalV1::ReleaseDeviceBuffer);
        });
        let first = kind.acquire(&fixture).unwrap();
        s.state.borrow_mut().release_host_buffer(first).unwrap();
        let reused = kind.acquire(&fixture).unwrap();
        assert_eq!(first.slot, reused.slot);
        assert_ne!(first.generation, reused.generation);
        s.state.borrow_mut().release_host_buffer(reused).unwrap();
        assert_eq!(s.with_device(FakeHostDeviceV1::live_object_count), live);
        fixture.release();
    }
}

struct MappedFixture(std::path::PathBuf);
impl Drop for MappedFixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn mapped_fixture() -> (MappedFixture, HostInstanceConfigV1) {
    use crate::direct::{
        DirectCheckpointV1, DirectLoadOptionsV1, PlannedImageV1, implementation_v1,
    };
    use runtime_bundle_abi::weight_plan::{
        PlannedViewV1, RecordSourceV1, ViewPlacementV1, WeightPlanV1, WeightRecordV1,
    };
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let root = MappedFixture(std::env::temp_dir().join(format!(
        "acquisition-mapped-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    )));
    std::fs::create_dir(&root.0).unwrap();
    let header = serde_json::to_vec(
        &serde_json::json!({"table":{"dtype":"U8","shape":[128],"data_offsets":[0,128]}}),
    )
    .unwrap();
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend(header);
    bytes.extend([0; 128]);
    std::fs::write(root.0.join("model.safetensors"), bytes).unwrap();
    let plan = WeightPlanV1 {
        ranks: 1,
        rank: 0,
        image_role: "fixture.image".into(),
        alignment: 64,
        arena_bytes: 128,
        views: vec![
            PlannedViewV1 {
                id: "device".into(),
                alignment: 64,
                placement: ViewPlacementV1::Device,
                offset: 0,
                byte_length: 64,
            },
            PlannedViewV1 {
                id: "mapped".into(),
                alignment: 64,
                placement: ViewPlacementV1::Mapped { prewarm: false },
                offset: 64,
                byte_length: 64,
            },
        ],
        components: vec![],
        records: [0, 64]
            .into_iter()
            .map(|offset| WeightRecordV1 {
                arena_offset: offset,
                byte_length: 64,
                transform: kernel_core::weight_transform::WeightTransform::Copy.id(),
                params: vec![],
                sources: vec![RecordSourceV1 {
                    file: 0,
                    tensor: "table".into(),
                }],
                axis: Some(0),
                ranges: std::iter::once(offset..offset + 64).collect(),
                stored_shape: vec![64],
            })
            .collect(),
    };
    let image = PlannedImageV1::new(
        Arc::new(DirectCheckpointV1::open(&root.0).unwrap()),
        vec![plan],
        None,
        implementation_v1([0; 32]),
        DirectLoadOptionsV1 {
            threads_per_rank: 1,
            read_ahead: false,
            direct_reads: false,
            shared_bytes: DirectLoadOptionsV1::SHARED_BYTES_DEFAULT,
        },
    )
    .unwrap();
    let config = HostInstanceConfigV1::new(HostTopologyV1::new(vec![0]).unwrap())
        .with_planned_compiled_image(Arc::new(image))
        .unwrap();
    (root, config)
}

#[test]
fn mapped_gather_reserves_metadata_before_constructor_and_refusal() {
    for fault in ["order", "slot", "constructor"] {
        let (_root, config) = mapped_fixture();
        let s = Services::new(FakeHostDeviceV1::new(), config);
        let context = s.select_context(0, 0).unwrap();
        let pool = s.select_memory_pool(context).unwrap();
        let stream = s.create_stream(context, 0).unwrap();
        let image = s.open_compiled_image(b"fixture.image").unwrap();
        let indices = s.allocate_device_buffer(pool, stream, 16, 64).unwrap();
        let output = s.allocate_device_buffer(pool, stream, 64, 64).unwrap();
        let planes = [runtime_bundle_abi::MappedGatherPlaneV1 {
            image,
            rank: 0,
            view_id: b"mapped".to_vec(),
            row_bytes: 16,
            index_offset: 0,
            stream,
            destination: DeviceBufferSpanV1::new(output, 0, 64),
        }];
        let acquire =
            || s.create_mapped_gather(stream, DeviceBufferSpanV1::new(indices, 0, 16), &planes);
        let issued = s.with_device(FakeHostDeviceV1::issued_object_count);
        let live = s.with_device(FakeHostDeviceV1::live_object_count);
        let (next, order) = {
            let state = s.state.borrow();
            (state.order.next, state.order.live.clone())
        };
        s.with_device_mut(|device| device.refuse(FakeRefusalV1::ReleaseMappedGather));
        match fault {
            "order" => s.state.borrow_mut().order.next = u64::MAX,
            "slot" => s
                .state
                .borrow_mut()
                .mapped_gathers
                .refuse_registration(true),
            _ => s.with_device_mut(|device| device.refuse(FakeRefusalV1::CreateMappedGather)),
        }
        assert_eq!(
            acquire(),
            Err(if fault == "constructor" {
                BundleStatusV1::HostError
            } else {
                BundleStatusV1::ResourceExhausted
            }),
            "{fault}"
        );
        assert_eq!(
            s.with_device(FakeHostDeviceV1::issued_object_count),
            issued,
            "{fault}"
        );
        assert_eq!(s.with_device(FakeHostDeviceV1::live_object_count), live);
        assert_eq!(s.host_memory_v1().gather_staging_bytes, 0);
        {
            let mut state = s.state.borrow_mut();
            assert_eq!(state.order.live, order);
            assert_eq!(state.mapped_gathers.live_count(), 0);
            state.order.next = next;
            state.mapped_gathers.refuse_registration(false);
        }
        s.with_device_mut(|device| {
            device.allow(FakeRefusalV1::CreateMappedGather);
            device.allow(FakeRefusalV1::ReleaseMappedGather);
        });
        let gather = acquire().unwrap();
        s.release_mapped_gather(gather).unwrap();
        s.release_device_buffer(output).unwrap();
        s.release_device_buffer(indices).unwrap();
        s.release_compiled_image(image).unwrap();
        s.release_stream(stream).unwrap();
        s.release_memory_pool(pool).unwrap();
        s.release_context(context).unwrap();
        assert_eq!(s.with_device(FakeHostDeviceV1::live_object_count), 0);
        assert_eq!(s.live_resource_count(), 0);
    }
}

fn rejected_buffer_retains_cleanup(host: Option<HostBuffer>, query_error: bool) {
    let fixture = DirectFixture::new();
    let s = &fixture.services;
    let issued = s.with_device(FakeHostDeviceV1::issued_object_count);
    let live = s.with_device(FakeHostDeviceV1::live_object_count);
    let orders = s.live_resource_count();
    s.with_device_mut(|device| {
        if query_error {
            device.refuse(FakeRefusalV1::AllocationAlignment);
        } else {
            device.set_allocation_alignment(1);
        }
        device.refuse(FakeRefusalV1::ReleaseDeviceBuffer);
    });
    let acquire = || match host {
        Some(kind) => kind.acquire(&fixture),
        None => Direct::Buffer.acquire(&fixture),
    };
    assert_eq!(
        acquire(),
        Err(BundleStatusV1::HostError),
        "cleanup failure must be returned: {host:?}, query_error={query_error}"
    );
    assert_eq!(
        s.state.borrow().buffers.live_count(),
        1,
        "the rejected buffer must retain an internal owner"
    );
    assert_eq!(s.with_device(FakeHostDeviceV1::live_object_count), live + 1);
    assert_eq!(
        s.live_resource_count(),
        orders + usize::from(host.is_none())
    );
    for _ in 0..2 {
        assert_eq!(acquire(), Err(BundleStatusV1::HostError));
        assert_eq!(
            s.with_device(FakeHostDeviceV1::issued_object_count),
            issued + 1
        );
        assert_eq!(s.state.borrow().buffers.live_count(), 1);
    }
    assert_eq!(
        s.release_stream(fixture.stream),
        Err(BundleStatusV1::HostError)
    );
    assert_eq!(
        s.release_memory_pool(fixture.pool),
        Err(BundleStatusV1::HostError)
    );
    assert_eq!(
        s.release_context(fixture.context),
        Err(BundleStatusV1::HostError)
    );
    s.with_device_mut(|device| {
        device.allow(FakeRefusalV1::AllocationAlignment);
        device.set_allocation_alignment(64);
    });
    assert_eq!(
        acquire(),
        Err(BundleStatusV1::HostError),
        "valid new buffers still wait for retained cleanup"
    );
    assert_eq!(
        s.with_device(FakeHostDeviceV1::issued_object_count),
        issued + 1
    );
    s.with_device_mut(|device| device.allow(FakeRefusalV1::ReleaseDeviceBuffer));
    let key = acquire().unwrap();
    assert_eq!(
        s.with_device(FakeHostDeviceV1::issued_object_count),
        issued + 2
    );
    assert_eq!(s.with_device(FakeHostDeviceV1::live_object_count), live + 1);
    if host.is_some() {
        s.state.borrow_mut().release_host_buffer(key).unwrap();
    } else {
        Direct::Buffer.release(&fixture, key);
    }
    assert_eq!(s.state.borrow().buffers.live_count(), 0);
    assert_eq!(s.live_resource_count(), orders);
    fixture.release();
}

#[test]
fn rejected_bundle_buffer_retains_ownership_until_cleanup_succeeds() {
    for query_error in [false, true] {
        rejected_buffer_retains_cleanup(None, query_error);
    }
}

#[cfg(target_os = "linux")]
#[test]
fn rejected_host_buffers_retain_ownership_until_cleanup_succeeds() {
    for host in HOST_BUFFERS {
        for query_error in [false, true] {
            rejected_buffer_retains_cleanup(Some(host), query_error);
        }
    }
}
