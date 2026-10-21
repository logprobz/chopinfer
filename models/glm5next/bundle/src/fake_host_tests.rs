//! CPU fake host for the glm5next bundle: create, acquisition and
//! retirement through the ABI, the shared fake-host backend
//! (`bundle_runtime::fake_host::Backend`) over this package.
//!
//! The host runs no kernel. It serves the planned text image from the
//! direct weight plan (its rank indexes and plan identities), the serving KV
//! pool and the per-item indexer tail state, records every device-facing
//! call and counts what the instance acquires and releases.

#[path = "conformance_tests.rs"]
mod conformance_tests;

use std::cell::{Cell, RefCell};
use std::collections::{BTreeMap, BTreeSet};
use std::rc::Rc;

use bundle_runtime::fake_host::{
    Backend, Countdown, FakeDevice, FakePackage, Handle, NoObserver, check_journal,
};
use bundle_runtime::kv::KvFormat;
use bundle_runtime::state::Extent;
use model_glm5next_contract::contract::{
    CAPABILITIES, DECODE_BATCH, INDEXER_TAIL_STATE_ID, Shape, has_mtp_layer,
};
use model_glm5next_contract::direct::{RANK_IMAGE_ROLE, index_role, plan, test_support};
use model_glm5next_contract::state::request_state;
use runtime_bundle_abi::*;

use crate::config::tests::{options, record};

mod anchors;
mod capacity;
mod checks;
mod comm_epilogue;
mod host_plan;
mod invariance;
mod ll_residency;
mod nvfp4;
mod phased_reset;
mod phased_residency;
mod prefill_dma;
mod rows;
mod shared_overlap;
mod sm120;
mod taps;

/// Room for a step's pages and a prefix snapshot's 89.
const KV_PAGES: u64 = 256;
const IMAGE_SLOT: u32 = 1_000_000;
pub(crate) const KV_SLOT: u32 = 2_000_000;
const STATE_SLOT: u32 = 3_000_000;
/// Authenticated component handles of the rank indexes and the plan
/// identities.
const INDEX_COMPONENT: u32 = 100;
const IDENTITY_COMPONENT: u32 = 200;
/// The vision encoder's image (handle generation 2, its buffer slots) and
/// component handles: its rank indexes and the image file's identities.
const ENCODER: u32 = 2;
const ENCODER_IMAGE_SLOT: u32 = 4_000_000;
const ENCODER_INDEX: u32 = 300;
const ENCODER_IDENTITY: u32 = 400;
/// The precision component's handle.
const PRECISION_COMPONENT: u32 = 500;
/// The encoder image file's digest its indexes and identities name.
const ENCODER_SHA256: &str = "5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a";

/// One rank's planned image: its index component and its identity.
#[derive(Clone)]
struct Planned {
    index: Vec<u8>,
    image_bytes: u64,
    identity: String,
}

/// Each rank's plan of the pinned config, one checkpoint file holding every
/// tensor, planned once per test process (the conformance scenarios create
/// a host per allocation).
fn planned_ranks() -> Vec<Planned> {
    static PLANNED: std::sync::OnceLock<Vec<Planned>> = std::sync::OnceLock::new();
    PLANNED
        .get_or_init(|| {
            planned_ranks_of(
                &model_glm5next_contract::quantization::FP8_BLOCK128,
                include_bytes!("../../contract/fixtures/config.json"),
            )
        })
        .clone()
}

/// [`planned_ranks`] of a checkpoint with `config` stored in `layout`,
/// once the host's preparation of every rank's plan accepts it.
fn planned_ranks_of(
    layout: &model_glm5next_contract::quantization::Layout,
    config: &[u8],
) -> Vec<Planned> {
    host_plan::prepare_synthetic(layout, config)
        .unwrap_or_else(|error| panic!("{}: the host refused the plan: {error}", layout.id));
    (0..4)
        .map(|rank| {
            let request = test_support::checkpoint_request(layout, config, rank, |_| {});
            Planned::of(&plan(&request).unwrap().unwrap())
        })
        .collect()
}

impl Planned {
    /// The index and identity `plan` gives its rank.
    fn of(plan: &runtime_bundle_abi::weight_plan::WeightPlanV1) -> Self {
        let index = plan.components[0].bytes.clone();
        let parsed: serde_json::Value = serde_json::from_slice(&index).unwrap();
        Self {
            identity: parsed["direct_identity"].as_str().unwrap().to_owned(),
            image_bytes: plan.arena_bytes,
            index,
        }
    }
}

/// The package the fake host serves, and what tests set and read. Its
/// device counts what the instance acquires and releases, and holds the
/// bytes of every buffer slot handed out, for the span checks.
pub(crate) struct FakeHost {
    pub(crate) device: FakeDevice,
    format: KvFormat,
    planned: Vec<Planned>,
    /// (drafts, verify tokens) the host enables.
    speculation: (u32, u32),
    /// Every live resource in acquisition order, as the host's release
    /// order tracks them (`runtime-host`'s `ReleaseOrderV1`): a release of
    /// anything but the newest live resource is refused with INVALID_STATE
    /// and changes nothing, so it can be retried.
    order: RefCell<Vec<String>>,
    pub(crate) logs: RefCell<Vec<String>>,
    /// The bytes last uploaded to each span, which a download of exactly
    /// that span reads back (zeros otherwise): a snapshot header's round
    /// trip.
    uploaded: RefCell<BTreeMap<(u32, u64, u64), Vec<u8>>>,
    /// Fails the first eager launch of the symbol on a stream of the rank.
    fail_launch_on_rank: RefCell<Option<(String, u32)>>,
    /// The pools that hold an allocation (`POOL_ALIGNMENT`).
    pools_used: RefCell<BTreeSet<u32>>,
    /// Whether every pool already holds another allocation when create
    /// starts, so no image arena is a pool's first.
    pub(crate) pools_occupied: Cell<bool>,
    /// Fails the first upload starting where this span starts.
    pub(crate) fail_upload_to: Cell<Option<DeviceBufferSpanV1>>,
    /// While set, uploads, device copies and readbacks move real bytes,
    /// kept sparsely by buffer slot and `CHUNK` (unwritten bytes are zero).
    pub(crate) backing: Cell<bool>,
    bytes: RefCell<BTreeMap<(u32, u64), Vec<u8>>>,
    /// Fail a device copy, or a stream synchronization, after this many.
    pub(crate) fail_copy: Countdown,
    pub(crate) fail_synchronize: Countdown,
    /// Graph releases to admit, then graph releases to refuse with
    /// HOST_ERROR (each leaving its graph live).
    pub(crate) fail_graph_releases: Cell<(u32, u32)>,
    /// Every upload in order: the journal line it precedes, its span and
    /// its bytes.
    upload_log: RefCell<Vec<(usize, DeviceBufferSpanV1, Vec<u8>)>>,
    /// The items the create record serves.
    items: Cell<u32>,
    /// Whether create acquired the serving KV, and the last buffer slot
    /// allocated before it did.
    kv_acquired: Cell<bool>,
    before_kv: Cell<u32>,
    /// Bytes of every buffer slot held when create acquired the serving KV.
    held_at_kv: Cell<u64>,
    /// The precision component the package carries.
    precision: RefCell<Vec<u8>>,
    /// Every eager launch in issue order: its stream, symbol, configuration
    /// and arguments.
    pub(crate) launch_log: RefCell<Vec<LoggedLaunch>>,
    /// Every launch recorded into a graph capture, in issue order, alike.
    pub(crate) capture_log: RefCell<Vec<LoggedLaunch>>,
    /// While set, every launch's stream, symbol and arguments, eager or
    /// captured, in issue order.
    pub(crate) launch_args: RefCell<Option<Vec<LaunchRecord>>>,
}

/// A launch's stream slot, symbol and arguments.
pub(crate) type LaunchRecord = (u32, String, Vec<KernelArgumentV1>);

/// The fake host of a glm5next package.
pub(crate) type Host = Backend<FakeHost>;

impl std::ops::Deref for FakeHost {
    type Target = FakeDevice;

    fn deref(&self) -> &FakeDevice {
        &self.device
    }
}

/// One eager launch the host took: its stream, symbol, configuration and
/// arguments.
pub(crate) type LoggedLaunch = (u32, String, LaunchConfigV1, Vec<KernelArgumentV1>);

/// Bytes of one chunk of the fake device's sparse backing.
const CHUNK: u64 = 1 << 16;

/// The alignment a stream-ordered CUDA pool guarantees each allocation.
/// The fake host places a pool's first allocation at its reserved base, so
/// that one happens to be aligned further (CUDA promises no more than this
/// constant); the host refuses any allocation whose address misses the
/// alignment the request (or the image plan) asks for, as a GPU create did
/// for a 4096-byte encoder image after the text rank image.
const POOL_ALIGNMENT: u64 = 256;

/// The audited resources of the low-latency all-reduce's entries: the BF16
/// partials' and the MoE sublayers' rounded F32 partials'.
pub(crate) fn ll_resources() -> BTreeMap<String, kernel_core::occupancy::EntryResources> {
    let mut resources = crate::sm120::device::locked_resources(include_str!(
        "../kernels.lock.d/allreduce_ll_bf16.lock"
    ));
    resources.extend(crate::sm120::device::locked_resources(include_str!(
        "../kernels.lock.d/allreduce_ll_f32.lock"
    )));
    resources
}

/// The audited resources of the phased all-reduce's entries: the BF16
/// partials' and the MoE sublayers' rounded F32 partials'.
pub(crate) fn phased_resources() -> BTreeMap<String, kernel_core::occupancy::EntryResources> {
    let mut resources = crate::sm120::device::locked_resources(include_str!(
        "../kernels.lock.d/comm_phased_bf16.lock"
    ));
    resources.extend(crate::sm120::device::locked_resources(include_str!(
        "../kernels.lock.d/comm_phased_turns_bf16.lock"
    )));
    resources.extend(crate::sm120::device::locked_resources(include_str!(
        "../kernels.lock.d/comm_phased_f32.lock"
    )));
    resources
}

/// The entries whose residency create asks about: the low-latency and the
/// phased all-reduce's.
pub(crate) fn spinning_resources() -> BTreeMap<String, kernel_core::occupancy::EntryResources> {
    let mut resources = ll_resources();
    resources.extend(phased_resources());
    resources
}

impl FakeHost {
    /// Places one allocation of `alignment` in `pool` as a CUDA pool may:
    /// refused, as the host refuses it, when it is not the pool's first and
    /// asks for more than `POOL_ALIGNMENT`.
    fn place(
        &self,
        pool: BundleMemoryPoolHandleV1,
        bytes: u64,
        alignment: u64,
    ) -> HostServiceResultV1<()> {
        let first = self.pools_used.borrow_mut().insert(pool.slot()) && !self.pools_occupied.get();
        if !first && alignment > POOL_ALIGNMENT {
            self.logs.borrow_mut().push(format!(
                "host allocate_device_buffer: {bytes} bytes, pool {}: the allocation does not meet \
                 the requested {alignment}-byte alignment",
                pool.slot()
            ));
            return Err(BundleStatusV1::Unsupported);
        }
        Ok(())
    }

    pub(crate) fn new(speculation: (u32, u32)) -> Rc<Host> {
        Self::with_format(speculation, Shape::TP4.kv_formats()[0])
    }

    pub(crate) fn with_format(speculation: (u32, u32), format: KvFormat) -> Rc<Host> {
        // Residency reports are what the target's driver gives for each
        // queried entry's audited resources at the exact queried launch
        // (`crate::sm120::device`), the low-latency
        // and the phased all-reduce's from their locks.
        let device = FakeDevice::default();
        *device.modelled.borrow_mut() = Some(crate::sm120::device::model(spinning_resources()));
        Rc::new(Backend::new(Self {
            device,
            format,
            planned: planned_ranks(),
            speculation,
            order: RefCell::default(),
            logs: RefCell::default(),
            uploaded: RefCell::default(),
            fail_launch_on_rank: RefCell::default(),
            pools_used: RefCell::default(),
            pools_occupied: Cell::new(false),
            fail_upload_to: Cell::new(None),
            backing: Cell::new(false),
            bytes: RefCell::default(),
            fail_copy: Countdown::default(),
            fail_synchronize: Countdown::default(),
            fail_graph_releases: Cell::new((0, 0)),
            upload_log: RefCell::default(),
            items: Cell::new(DECODE_BATCH),
            kv_acquired: Cell::new(false),
            before_kv: Cell::new(0),
            held_at_kv: Cell::new(0),
            precision: RefCell::new(model_glm5next_contract::prepared::test_component()),
            launch_log: RefCell::default(),
            capture_log: RefCell::default(),
            launch_args: RefCell::default(),
        }))
    }

    /// A host of the package of a checkpoint with `config` stored in
    /// `layout`: its planned rank images and its precision component.
    pub(crate) fn with_layout(
        speculation: (u32, u32),
        layout: &model_glm5next_contract::quantization::Layout,
        config: &[u8],
    ) -> Rc<Host> {
        Self::with_layout_in(speculation, layout, config, Shape::TP4.kv_formats()[0])
    }

    /// [`Self::with_layout`] with the KV format `format`.
    pub(crate) fn with_layout_in(
        speculation: (u32, u32),
        layout: &model_glm5next_contract::quantization::Layout,
        config: &[u8],
        format: KvFormat,
    ) -> Rc<Host> {
        let mut host = Rc::into_inner(Self::with_format(speculation, format))
            .expect("a new host")
            .into_inner();
        host.planned = planned_ranks_of(layout, config);
        let config: serde_json::Value = serde_json::from_slice(config).unwrap();
        host.precision = RefCell::new(
            model_glm5next_contract::prepared::checkpoint_component(layout, &config).unwrap(),
        );
        Rc::new(Backend::new(host))
    }

    /// Tracks `resource` as the newest live resource.
    fn enter(&self, resource: String) {
        self.order.borrow_mut().push(resource);
    }

    /// Resources still live, newest last.
    pub(crate) fn live(&self) -> Vec<String> {
        self.order.borrow().clone()
    }

    /// Writes `data` at `span` of the sparse backing.
    pub(crate) fn write(&self, span: DeviceBufferSpanV1, data: &[u8]) {
        assert_eq!(span.length, data.len() as u64);
        let mut bytes = self.bytes.borrow_mut();
        let (mut at, mut rest) = (span.offset, data);
        while !rest.is_empty() {
            let within = (at % CHUNK) as usize;
            let count = rest.len().min(CHUNK as usize - within);
            bytes
                .entry((span.buffer.slot(), at / CHUNK))
                .or_insert_with(|| vec![0; CHUNK as usize])[within..within + count]
                .copy_from_slice(&rest[..count]);
            at += count as u64;
            rest = &rest[count..];
        }
    }

    /// The sparse backing's bytes of `span`.
    pub(crate) fn read(&self, span: DeviceBufferSpanV1) -> Vec<u8> {
        let bytes = self.bytes.borrow();
        let mut out = Vec::with_capacity(span.length as usize);
        let mut at = span.offset;
        while (out.len() as u64) < span.length {
            let within = (at % CHUNK) as usize;
            let count = (span.length as usize - out.len()).min(CHUNK as usize - within);
            match bytes.get(&(span.buffer.slot(), at / CHUNK)) {
                Some(chunk) => out.extend_from_slice(&chunk[within..within + count]),
                None => out.resize(out.len() + count, 0),
            }
            at += count as u64;
        }
        out
    }

    /// The per-item tail state the package declares for the served items.
    fn state_bytes(&self) -> u64 {
        u64::from(self.items.get().next_power_of_two())
            * Shape::TP4.indexer_tail_slot_bytes(has_mtp_layer(&self.format))
    }
}

impl FakePackage for FakeHost {
    type Observer = NoObserver;

    fn device(&self) -> &FakeDevice {
        &self.device
    }
    fn device_residency(&self) -> BundleRawClusterResidencyV1 {
        crate::sm120::device::comm_residency()
    }
    fn cubins() -> impl IntoIterator<Item = (&'static str, &'static [u8])> {
        crate::CUBINS.iter().map(|cubin| (cubin.name, cubin.bytes))
    }
    fn log(&self, message: &[u8]) {
        self.logs
            .borrow_mut()
            .push(String::from_utf8_lossy(message).into_owned());
    }
    fn on_acquire(&self, handle: Handle) {
        self.enter(handle.to_string());
    }
    /// Admits a release of the newest live resource, as the host does;
    /// refuses any other with INVALID_STATE, leaving it live.
    fn admit_release(&self, handle: Handle) -> HostServiceResultV1<()> {
        if matches!(handle, Handle::Graph(_)) {
            match self.fail_graph_releases.get() {
                (0, 0) => {}
                (0, refused) => {
                    self.fail_graph_releases.set((0, refused - 1));
                    return Err(BundleStatusV1::HostError);
                }
                (admitted, refused) => self.fail_graph_releases.set((admitted - 1, refused)),
            }
        }
        let resource = handle.to_string();
        let order = self.order.borrow();
        if order.last() != Some(&resource) {
            let refusal = format!(
                "host release order: {resource} is not the newest live resource ({:?})",
                order.last()
            );
            eprintln!("{refusal}");
            self.logs.borrow_mut().push(refusal);
            return Err(BundleStatusV1::InvalidState);
        }
        Ok(())
    }
    /// The host journals no release line of its own.
    fn release(&self, _handle: Handle) {
        self.order.borrow_mut().pop();
    }
    fn admit_allocation(
        &self,
        pool: BundleMemoryPoolHandleV1,
        byte_length: u64,
        alignment: u64,
    ) -> HostServiceResultV1<()> {
        self.place(pool, byte_length, alignment)
    }
    /// The attempted extent, and the last buffer allocated before the
    /// serving KV, stand even when the device fails the allocation.
    fn before_allocation(
        &self,
        _pool: BundleMemoryPoolHandleV1,
        buffer: BundleDeviceBufferHandleV1,
        byte_length: u64,
    ) {
        self.extents.borrow_mut().insert(buffer.slot(), byte_length);
        if !self.kv_acquired.get() {
            self.before_kv.set(buffer.slot());
        }
    }
    fn open_compiled_image(&self, role: &[u8]) -> HostServiceResultV1<BundleCompiledImageHandleV1> {
        let image = if role == crate::encoders::views::IMAGE_ROLE {
            BundleCompiledImageHandleV1::from_parts(ENCODER, self.device.id())
        } else {
            assert_eq!(role, RANK_IMAGE_ROLE.as_bytes());
            BundleCompiledImageHandleV1::from_parts(1, self.device.id())
        };
        Ok(image)
    }
    fn upload_rank_arena(
        &self,
        _context: BundleContextHandleV1,
        pool: BundleMemoryPoolHandleV1,
        _stream: BundleStreamHandleV1,
        image: BundleCompiledImageHandleV1,
        rank: u32,
    ) -> HostServiceResultV1<BundleRankArenaHandleV1> {
        // Each image's arena at the alignment its weight plan declares.
        if image.generation() == ENCODER {
            self.place(
                pool,
                crate::encoders::views::image_bytes().unwrap(),
                model_glm5next_contract::encoder_plan::ALIGNMENT,
            )?;
        } else {
            self.place(
                pool,
                self.planned[rank as usize].image_bytes,
                model_glm5next_contract::direct::ARENA_ALIGNMENT,
            )?;
        }
        let arena = if image.generation() == ENCODER {
            self.device.note(format!("upload encoder arena {rank}"));
            BundleRankArenaHandleV1::from_parts(ENCODER, self.device.id())
        } else {
            self.device.note(format!("upload arena {rank}"));
            BundleRankArenaHandleV1::from_parts(1, self.device.id())
        };
        Ok(arena)
    }
    fn resolve_compiled_image_view(
        &self,
        image: BundleCompiledImageHandleV1,
        _arena: BundleRankArenaHandleV1,
        rank: u32,
        view_id: &[u8],
    ) -> HostServiceResultV1<CompiledImageSpanV1> {
        if !view_id.is_empty() {
            return Err(BundleStatusV1::NotFound);
        }
        // Each image at its weight plan's arena alignment.
        let (bytes, slot, alignment) = match image.generation() {
            ENCODER => (
                crate::encoders::views::image_bytes().unwrap(),
                ENCODER_IMAGE_SLOT + rank,
                model_glm5next_contract::encoder_plan::ALIGNMENT,
            ),
            _ => (
                self.planned[rank as usize].image_bytes,
                IMAGE_SLOT + rank,
                model_glm5next_contract::direct::ARENA_ALIGNMENT,
            ),
        };
        self.extents.borrow_mut().insert(slot, bytes);
        Ok(CompiledImageSpanV1::new(
            rank,
            DeviceBufferSpanV1::new(BundleDeviceBufferHandleV1::from_parts(1, slot), 0, bytes),
            alignment,
        ))
    }
    fn open_authenticated_component(
        &self,
        role: &[u8],
    ) -> HostServiceResultV1<BundleAuthenticatedComponentHandleV1> {
        if role == crate::projections::COMPONENT_ROLE.as_bytes() {
            return Ok(BundleAuthenticatedComponentHandleV1::from_parts(
                1,
                PRECISION_COMPONENT,
            ));
        }
        let encoder = std::str::from_utf8(crate::encoders::views::IMAGE_ROLE).unwrap();
        let slot = (0..4)
            .find_map(|rank| {
                if role == crate::encoders::views::index_role(rank).as_bytes() {
                    Some(ENCODER_INDEX + rank)
                } else if role
                    == format!("chop.compiled-image-identity.v1/{encoder}/{rank}").as_bytes()
                {
                    Some(ENCODER_IDENTITY + rank)
                } else if role == index_role(rank).as_bytes() {
                    Some(INDEX_COMPONENT + rank)
                } else if role
                    == format!("chop.compiled-image-identity.v1/{RANK_IMAGE_ROLE}/{rank}")
                        .as_bytes()
                {
                    Some(IDENTITY_COMPONENT + rank)
                } else {
                    None
                }
            })
            .ok_or(BundleStatusV1::NotFound)?;
        Ok(BundleAuthenticatedComponentHandleV1::from_parts(1, slot))
    }
    fn component(&self, slot: u32) -> (String, Vec<u8>) {
        use crate::encoders::views;
        if slot == PRECISION_COMPONENT {
            return (
                crate::projections::COMPONENT_ROLE.into(),
                self.precision.borrow().clone(),
            );
        }
        let image = std::str::from_utf8(views::IMAGE_ROLE).unwrap();
        if let Some(rank) = slot.checked_sub(ENCODER_IDENTITY) {
            let identity = serde_json::json!({
                "byte_length": views::image_bytes().unwrap(),
                "sha256": ENCODER_SHA256,
                "direct_identity": null,
            });
            return (
                format!("chop.compiled-image-identity.v1/{image}/{rank}"),
                serde_json::to_vec(&identity).unwrap(),
            );
        }
        if let Some(rank) = slot.checked_sub(ENCODER_INDEX) {
            let index = views::index(
                rank,
                bundle_runtime::index::encoder::Identity::Sha256(ENCODER_SHA256),
            )
            .unwrap();
            return (views::index_role(rank), serde_json::to_vec(&index).unwrap());
        }
        if let Some(rank) = slot.checked_sub(IDENTITY_COMPONENT) {
            let planned = &self.planned[rank as usize];
            let identity = serde_json::json!({
                "byte_length": planned.image_bytes,
                "sha256": null,
                "direct_identity": planned.identity,
            });
            return (
                format!("chop.compiled-image-identity.v1/{RANK_IMAGE_ROLE}/{rank}"),
                serde_json::to_vec(&identity).unwrap(),
            );
        }
        let rank = slot - INDEX_COMPONENT;
        (index_role(rank), self.planned[rank as usize].index.clone())
    }
    fn admit_synchronize(&self, _stream: BundleStreamHandleV1) -> HostServiceResultV1<()> {
        if self.fail_synchronize.fails() {
            return Err(BundleStatusV1::HostError);
        }
        Ok(())
    }
    fn admit_upload(
        &self,
        _stream: BundleStreamHandleV1,
        destination: DeviceBufferSpanV1,
        _source: &[u8],
    ) -> HostServiceResultV1<()> {
        if self.fail_upload_to.get().is_some_and(|span| {
            (span.buffer.slot(), span.offset) == (destination.buffer.slot(), destination.offset)
        }) {
            self.fail_upload_to.set(None);
            return Err(BundleStatusV1::HostError);
        }
        Ok(())
    }
    fn on_upload(
        &self,
        _stream: BundleStreamHandleV1,
        destination: DeviceBufferSpanV1,
        source: &[u8],
    ) -> HostServiceResultV1<()> {
        if self.backing.get() {
            self.write(destination, source);
        }
        let line = self.device.journal.borrow().len();
        self.upload_log
            .borrow_mut()
            .push((line, destination, source.to_vec()));
        self.uploaded.borrow_mut().insert(
            (
                destination.buffer.slot(),
                destination.offset,
                destination.length,
            ),
            source.to_vec(),
        );
        Ok(())
    }
    fn read_back(
        &self,
        destination: &mut [u8],
        source: DeviceBufferSpanV1,
    ) -> HostServiceResultV1<()> {
        if self.backing.get() {
            destination.copy_from_slice(&self.read(source));
            return Ok(());
        }
        match self
            .uploaded
            .borrow()
            .get(&(source.buffer.slot(), source.offset, source.length))
        {
            Some(bytes) if bytes.len() == destination.len() => destination.copy_from_slice(bytes),
            _ => destination.fill(0),
        }
        Ok(())
    }
    /// Checks both spans before the injected copy failure.
    fn admit_copy(
        &self,
        _stream: BundleStreamHandleV1,
        destination: DeviceBufferSpanV1,
        source: DeviceBufferSpanV1,
    ) -> HostServiceResultV1<()> {
        self.check_span(source);
        self.check_span(destination);
        if self.fail_copy.fails() {
            return Err(BundleStatusV1::HostError);
        }
        Ok(())
    }
    fn on_copy(
        &self,
        _stream: BundleStreamHandleV1,
        destination: DeviceBufferSpanV1,
        source: DeviceBufferSpanV1,
    ) -> HostServiceResultV1<()> {
        if self.backing.get() {
            let bytes = self.read(source);
            self.write(destination, &bytes);
        }
        Ok(())
    }
    fn copy_peer_to_peer(
        &self,
        stream: BundleStreamHandleV1,
        _destination_context: BundleContextHandleV1,
        destination: DeviceBufferSpanV1,
        _source_context: BundleContextHandleV1,
        source: DeviceBufferSpanV1,
    ) -> HostServiceResultV1<()> {
        self.check_span(source);
        self.check_span(destination);
        assert_eq!(source.length, destination.length);
        if self.fail_copy.fails() {
            return Err(BundleStatusV1::HostError);
        }
        self.device.journal.borrow_mut().push(format!(
            "peer-copy {} {source:?} -> {destination:?}",
            stream.slot()
        ));
        if self.backing.get() {
            self.write(destination, &self.read(source));
        }
        Ok(())
    }
    /// Fails the first eager launch of the failing symbol on a stream of
    /// its rank.
    fn admit_launch(
        &self,
        stream: BundleStreamHandleV1,
        name: &str,
        _config: LaunchConfigV1,
        _arguments: &[KernelArgumentV1],
    ) -> HostServiceResultV1<()> {
        let rank = self
            .device
            .stream_ranks
            .borrow()
            .get(&stream.slot())
            .copied();
        let fails = matches!(
            &*self.fail_launch_on_rank.borrow(),
            Some((failing, on)) if *failing == name
                && rank == Some(*on)
                && self.device.capture_of(stream.slot()).is_none()
        );
        if fails {
            self.fail_launch_on_rank.borrow_mut().take();
            return Err(BundleStatusV1::HostError);
        }
        Ok(())
    }
    /// Logs each launch (`launch_log`, or `capture_log` while captured),
    /// and records it while `launch_args` is set. Commit's state copy moves
    /// real bytes, as the host copy per layer it replaced did.
    fn on_launch(
        &self,
        stream: BundleStreamHandleV1,
        name: &str,
        config: LaunchConfigV1,
        arguments: &[KernelArgumentV1],
    ) -> HostServiceResultV1<()> {
        let log = match self.device.capture_of(stream.slot()) {
            None => &self.launch_log,
            Some(_) => &self.capture_log,
        };
        log.borrow_mut()
            .push((stream.slot(), name.to_owned(), config, arguments.to_vec()));
        if let Some(launches) = self.launch_args.borrow_mut().as_mut() {
            launches.push((stream.slot(), name.to_owned(), arguments.to_vec()));
        }
        if self.backing.get() && name == model_glm5next_contract::sm120::state_copy::SYMBOL {
            let [
                KernelArgumentV1::DeviceBuffer(source),
                KernelArgumentV1::DeviceBuffer(destination),
                KernelArgumentV1::ScalarU64(source_stride),
                KernelArgumentV1::ScalarU64(destination_stride),
                KernelArgumentV1::ScalarU64(bytes),
            ] = arguments
            else {
                panic!("state copy arguments {arguments:?}");
            };
            let layer = |span: DeviceBufferSpanV1, stride: u64, layer: u64| {
                DeviceBufferSpanV1::new(span.buffer, span.offset + layer * stride, *bytes)
            };
            for l in 0..u64::from(config.grid_y) {
                let read = self.read(layer(*source, *source_stride, l));
                self.write(layer(*destination, *destination_stride, l), &read);
            }
        }
        Ok(())
    }
    fn acquire_serving_kv(&self, layout_id: &[u8]) -> HostServiceResultV1<u64> {
        assert_eq!(layout_id, self.format.layout_id);
        self.device.note(format!("serving kv {KV_PAGES}"));
        self.kv_acquired.set(true);
        self.held_at_kv.set(self.extents.borrow().values().sum());
        Ok(KV_PAGES)
    }
    fn resolve_serving_kv_pool(&self, rank: u32) -> HostServiceResultV1<DeviceBufferSpanV1> {
        let bytes = KV_PAGES * self.format.page_bytes;
        self.extents.borrow_mut().insert(KV_SLOT + rank, bytes);
        Ok(DeviceBufferSpanV1::new(
            BundleDeviceBufferHandleV1::from_parts(1, KV_SLOT + rank),
            0,
            bytes,
        ))
    }
    fn query_speculation(&self) -> HostServiceResultV1<BundleRawSpeculationV1> {
        let (drafts, verify) = self.speculation;
        let mut speculation = BundleRawSpeculationV1::disabled();
        if drafts > 0 {
            speculation.maximum_draft_tokens = drafts;
            speculation.maximum_verify_tokens = verify;
        }
        Ok(speculation)
    }
    fn allocate_state(&self, request: &StateRequestV1) -> HostServiceResultV1<BundleStateHandleV1> {
        assert_eq!(request.state_id, INDEXER_TAIL_STATE_ID.as_bytes());
        assert_eq!(request.byte_length, self.state_bytes());
        self.device.note(format!("state {}", request.byte_length));
        Ok(BundleStateHandleV1::from_parts(1, self.device.id()))
    }
    fn resolve_state_view(
        &self,
        _state: BundleStateHandleV1,
        rank: u32,
    ) -> HostServiceResultV1<DeviceBufferSpanV1> {
        let bytes = self.state_bytes();
        self.extents.borrow_mut().insert(STATE_SLOT + rank, bytes);
        Ok(DeviceBufferSpanV1::new(
            BundleDeviceBufferHandleV1::from_parts(1, STATE_SLOT + rank),
            0,
            bytes,
        ))
    }
}

pub(crate) fn api() -> BundleApiV1 {
    BundleApiV1::new_with_prefix_state_slot(crate::BUNDLE_GLM5NEXT_EXTENDED_API_V1)
        .expect("bundle API")
}

/// Creates an instance of the FP8 package serving `DECODE_BATCH` items.
fn create(host: &Rc<Host>) -> BundleInstanceV1 {
    api()
        .create(
            HostServicesTableV1::new(Rc::clone(host)),
            &record(DECODE_BATCH, &options(2048)),
        )
        .unwrap_or_else(|error| panic!("fake-host create: {error:?} {:?}", host.logs.borrow()))
}

/// Create acquires in the declared order, the KDA pools before the serving
/// KV pool, the tail in the host's per-item state; retirement succeeds and
/// destroy gives back everything.
/// Re-pinned when GLM's own draft argmax kernels left its row cubin (two
/// fewer resolves per rank at create; the ids issued after them renumber).
#[test]
fn create_acquires_retires_and_releases() {
    // Three drafts: four verify rows per item.
    let host = FakeHost::new((3, 32));
    let mut instance = create(&host);
    let journal = host.device.journal.borrow().clone();
    let position = |line: &str| {
        journal
            .iter()
            .position(|entry| entry.starts_with(line))
            .unwrap_or_else(|| panic!("no {line}"))
    };
    assert!(position("state ") < position("serving kv"));
    // Each rank's KDA conv and state pools precede the serving pool, which
    // the host sizes from what they leave: the committed entries and four
    // checkpoints of every served item, then the anchor spares; the
    // separate anchor spares (the indexer tail's) follow the state pool in
    // its allocation.
    let extent = Extent {
        served: DECODE_BATCH,
        bucket: DECODE_BATCH,
        verify_rows: 4,
    };
    let kv = position("serving kv");
    let state = request_state(Shape::TP4, &host.format, false);
    let separate: u64 = state
        .buffers
        .iter()
        .map(|buffer| buffer.spare_bytes())
        .sum();
    assert!(separate > 0);
    for name in ["kda.conv", "kda.state"] {
        let buffer = state
            .buffers
            .iter()
            .find(|buffer| buffer.name == name)
            .unwrap();
        let bytes = buffer.bytes(extent) + if name == "kda.state" { separate } else { 0 };
        let line = format!("allocate {bytes} 256");
        let lines = journal
            .iter()
            .enumerate()
            .filter(|(_, entry)| **entry == line)
            .map(|(index, _)| index)
            .collect::<Vec<_>>();
        assert_eq!(lines.len(), 4, "{name}");
        assert!(lines.iter().all(|&index| index < kv), "{name}");
    }
    assert!(
        host.logs
            .borrow()
            .iter()
            .any(|line| line == "GLM 5.3 Flash: 256 KV pages per rank, 32 items")
    );
    // Re-pinned as create grew: the KDA checkpoint table's stride (1,024
    // bytes), the prefill piece arenas, the row cubin's norm and embedding
    // symbols with three scratch buffers, then the all-reduce state (seven
    // buffers, four zeroed), the F32 partial, the head rows and the comm
    // and own-row symbols, then the logits shard, the logits moves' flags
    // and epochs (zeroed) and full rows, and their two comm symbols; then
    // the fake pool's 256 pages (was 64), which hold a prefix snapshot;
    // then the drafter's three buffers and its residual-add symbol; then
    // the one-shot flags and epochs sized for its tiles and the two-shot's
    // for the phased protocol's fixed layout; then one one-shot state per
    // served row count (16 states of three buffers, two zeroed); then the
    // logits shard and prefill rows sized for 64 scored rows; then the F32
    // MLP output and the F32 routed sum's symbol; then the drafter's
    // gathered hidden inputs (one more scratch buffer); then the drafter's
    // scratch only on drafting instances (four fewer); then the drafter's
    // three own row symbols; then the projections' cubin (its shared GEMM
    // symbols resolved and configured on every rank), whose BF16 entries
    // were then padded to the format's 64-row tile (105 symbols); then the
    // KDA gates' two low-rank scratch buffers; then the head's own use
    // profile's entries (107 symbols); then the projections' layout copies'
    // cubin (21 symbols) and the copies' contiguous temporary; then the
    // drafter's input halves' copies (14 symbols) in place of one of its
    // buffers; then the padded BF16 projections' compactions (14 symbols);
    // then the router's padded scores' buffer and their compaction's row
    // symbol; then the payload admission's cubin (3 symbols), each rank's
    // report buffer, upload, admission launches over its block scales,
    // synchronization and readback; then the layout copies' 49 entries
    // merged into one per layout (7 symbols), as capacity no longer names
    // a copy's entry; then commit's state copy cubin (one symbol); then the
    // streamed KDA prefill stages' cubin (2 symbols); then the fused Sinkhorn
    // mixer and input norm's cubin (one symbol per rank); then the phased
    // all-reduce's state after the low-latency one (three buffers, flags
    // and epoch zeroed) and its cubin (2 symbols per rank), the last cubin
    // loaded; then the shared gate/up overlap's side stream and two events
    // per rank, now the default (`fake_host_tests::shared_overlap`). The
    // selections' tree merges' cubin (3 symbols per rank) loads after the
    // pooled indexer's. Re-pinned when the strip-chain and prefill warp-tile
    // entries took the kernels that fence their ring reads before release
    // (their symbols and resolve order).
    // Re-pinned when prefill mixers from 144 rows ran the tiled mix-collapse
    // and its two symbols joined create (the handles). The KDA input
    // projections' stacked cubin (6 symbols, the 32-row entry's shared memory
    // configured) loads after the projections'.
    // The slice-widened FP8 prefill attention's cubin (2 symbols per rank)
    // resolves after the latent attention's, and the one-split selections'
    // cubin (3 symbols per rank) after the tree merges'. The projections'
    // decoded-tile cubin (6 symbols per rank, each configured) loads after the
    // layout copies', and the chained MoE's decoded-tile cubin (2 symbols per
    // rank, each configured) before its tile pairs'. Re-pinned when the
    // projections' decoded-tile cubin took the balanced decoded tiles (its
    // six symbols). The projections' staged-tile cubin (5 symbols per rank,
    // each configured) loads after the decoded-tile one. Re-pinned when the KDA
    // pools grew by the anchor spares, and the state pool by the separate ones
    // (sizes only). The decode and verify latent attention's column merges'
    // cubin (8 symbols per rank) resolves after the slice-widened cubin, and the
    // global selection's radix merge's cubin (one symbol per rank) before the
    // tree merges'. The MoE routed sum add's cubin (one symbol per rank) loads
    // last. Re-pinned when create loaded the phased turns all-reduce's cubin
    // and the phased flags grew by the turn words. Re-pinned when create
    // checked every rank index before the ranks opened: the image takes the
    // first handle, so the rank streams are 2 to 5 (were 1 to 4). Re-pinned
    // when the stream expansion and the score compaction launched the shared
    // row copies (identity change, accuracy unaffected): the row cubin
    // resolves two symbols fewer and `row_copy_sm120` (two symbols per rank)
    // loads after the other text cubins, so later handles move. Re-pinned
    // when GLM's own draft argmax kernels left the row cubin (two fewer
    // resolves per rank; the ids issued after them renumber). Re-pinned when
    // the KDA state pools moved before the serving KV pool: the one `serving
    // kv` line follows their allocations and zeroes, every other line and
    // handle as before. Re-pinned when the dense FP8 chained cubin lost its
    // two unlaunched expert-parallel entries (two fewer resolves per rank;
    // later ids renumber).
    check_journal(&journal, (2786, 0xacd61864e3f2e382), "glm5next-create.txt").unwrap();
    instance.retire_request(7).unwrap();
    instance.destroy().unwrap();
    assert_eq!(host.acquired.get(), host.released.get());
}

/// A drafting instance allocates the drafter's staging and its retained
/// rows (the declared `mtp.retained` state, which the drafter's commit and
/// prefix snapshots use) before it sizes the KV pool, which the host
/// reserves no room for them in, and the KDA pools likewise.
#[test]
fn drafting_instances_allocate_the_drafters_state_before_the_kv_pool() {
    let format = Shape::TP4.kv_formats()[1];
    let host = FakeHost::with_format((3, 32), format);
    let mut instance = create_over(&host, format);
    let handle = instance.handle();
    assert!(host.kv_acquired.get());
    let kv = host.before_kv.get();
    let (spans, table) = crate::instance::snapshot_layout(handle, 0);
    let names = request_state(Shape::TP4, &format, true)
        .buffers
        .iter()
        .map(|buffer| buffer.name)
        .collect::<Vec<_>>();
    assert!(names.contains(&"mtp.retained"));
    assert!(!table.descriptors.is_empty());
    for (rank, spans) in spans.iter().enumerate() {
        for (name, span) in names.iter().zip(spans) {
            match *name {
                "mtp.retained" | "kda.conv" | "kda.state" => {
                    assert!(span.buffer.slot() <= kv, "{rank}");
                }
                _ => {}
            }
        }
        for staged in crate::instance::drafter_staged(handle, rank) {
            assert!(staged.buffer.slot() <= kv, "{rank}");
        }
    }
    instance.destroy().unwrap();
    assert_eq!(host.acquired.get(), host.released.get());
}

/// A prompt of a text row, an image of `grid` patches (its `tokens` pad
/// rows, encoded into arena pages from 0 on) and a text row, in slot 0.
fn image_prompt(request: u64, grid: [u32; 3]) -> Vec<u8> {
    use crate::encoders::{PAGE_ROWS, PATCH_COLUMNS};
    use runtime_bundle_sdk::{
        BatchAppendV1, BatchItemV1, BatchMediaEncodeV3, BatchMediaInjectV3, BatchMediaSectionV3,
        BatchRequestV1, BatchWireVersionV1,
    };
    let patches = u64::from(grid[1] * grid[2]);
    let tokens = (patches / 4) as u32;
    let payload_bytes = patches * u64::from(PATCH_COLUMNS) * 4;
    let pages = u64::from(tokens).div_ceil(PAGE_ROWS);
    let media = BatchMediaSectionV3 {
        encodes: vec![BatchMediaEncodeV3 {
            encoder: 0,
            geometry: [grid[0], grid[1], grid[2], 0],
            output_rows: tokens,
            pages: (0..pages).collect(),
            payload_offset: 0,
            payload_bytes,
        }],
        injects: (0..pages)
            .map(|at| BatchMediaInjectV3 {
                item_index: 0,
                item_row: 1 + (at * PAGE_ROWS) as u32,
                rows: (u64::from(tokens) - at * PAGE_ROWS).min(PAGE_ROWS) as u32,
                arena_row: at * PAGE_ROWS,
            })
            .collect(),
        payload: (0..payload_bytes).map(|byte| (byte % 251) as u8).collect(),
    };
    let rows = u64::from(tokens) + 2;
    BatchRequestV1::new(
        BatchWireVersionV1::V3,
        vec![BatchItemV1::new(
            request,
            false,
            rows as u32,
            true,
            (0..rows as u32).map(|row| 10 + row % 7).collect(),
        )],
        vec![BatchAppendV1::new(
            0,
            rows,
            rows,
            (0..rows.div_ceil(64)).collect(),
            None,
        )],
    )
    .with_media(media)
    .with_state_slots(vec![0])
    .encode()
    .unwrap()
}

/// Each stream's launch lines from journal line `from` on, in issue order.
pub(crate) fn launches_by_stream(host: &FakeHost, from: usize) -> BTreeMap<u32, Vec<String>> {
    let mut streams = BTreeMap::<u32, Vec<String>>::new();
    for line in &host.device.journal.borrow()[from..] {
        let mut words = line.split(' ');
        if words.next() == Some("launch") {
            let stream = words.next().unwrap().parse().unwrap();
            streams.entry(stream).or_default().push(line.clone());
        }
    }
    streams
}

/// An instance over a record naming the vision encoder's components, over
/// the host's KV format.
fn create_with_encoder(host: &Rc<Host>) -> BundleInstanceV1 {
    let roles = crate::config::tests::encoder_roles();
    api()
        .create(
            HostServicesTableV1::new(Rc::clone(host)),
            &crate::config::tests::record_naming(DECODE_BATCH, &options(2048), host.format, &roles),
        )
        .unwrap_or_else(|error| panic!("fake-host create: {error:?} {:?}", host.logs.borrow()))
}

/// The encoder image's arena follows the text rank image in each rank's
/// pool, so it may ask only for the alignment a pool guarantees every
/// allocation: a GPU create refused the 4096-byte one ("the allocation does
/// not meet the requested 4096-byte alignment"). Neither image relies on
/// being its pool's first allocation: create succeeds when every pool
/// already holds another one.
#[test]
fn the_rank_images_ask_only_the_pool_alignment() {
    for occupied in [false, true] {
        let host = FakeHost::new((3, 32));
        host.pools_occupied.set(occupied);
        let mut instance = create_with_encoder(&host);
        assert!(
            !host
                .logs
                .borrow()
                .iter()
                .any(|line| line.contains("requested")),
            "{:?}",
            host.logs.borrow()
        );
        instance.destroy().unwrap();
        assert_eq!(host.acquired.get(), host.released.get());
    }
}

/// A record naming the vision encoder's components: create loads its
/// cubins and acquires its image, working memory and arena on every rank
/// before it sizes the KV pool, and destroy gives them back. An instance
/// without the encoder refuses media.
#[test]
fn create_acquires_the_vision_encoder_before_the_kv_pool() {
    let host = FakeHost::new((3, 32));
    let mut instance = create_with_encoder(&host);
    let journal = host.device.journal.borrow().clone();
    let kv = journal
        .iter()
        .position(|line| line.starts_with("serving kv"))
        .unwrap();
    let arenas = journal
        .iter()
        .enumerate()
        .filter(|(_, line)| line.starts_with("upload encoder arena "))
        .map(|(at, _)| at)
        .collect::<Vec<_>>();
    assert_eq!(arenas.len(), 4);
    assert!(arenas.iter().all(|&at| at < kv));
    // The KDA state pools follow the encoder, last before the KV pool.
    let extent = Extent {
        served: DECODE_BATCH,
        bucket: DECODE_BATCH,
        verify_rows: 4,
    };
    let conv = request_state(Shape::TP4, &host.format, false)
        .buffers
        .iter()
        .find(|buffer| buffer.name == "kda.conv")
        .unwrap()
        .bytes(extent);
    let pools = journal[..kv]
        .iter()
        .position(|line| *line == format!("allocate {conv} 256"))
        .unwrap();
    assert!(arenas.iter().all(|&at| at < pools));
    // The encoder's working memory, arena and source rows, per rank.
    let allocated: u64 = journal[arenas[0]..pools]
        .iter()
        .filter_map(|line| line.strip_prefix("allocate "))
        .map(|line| line.split(' ').next().unwrap().parse::<u64>().unwrap())
        .sum();
    assert_eq!(allocated, 4 * crate::encoders::buffer_bytes());
    instance.destroy().unwrap();
    assert_eq!(host.acquired.get(), host.released.get());
    let host = FakeHost::new((3, 32));
    let mut instance = create(&host);
    assert_eq!(
        instance
            .execute(&image_prompt(7, [1, 8, 8]), &mut DraftSink::default())
            .unwrap_err(),
        BundleAbiError::Status(BundleStatusV1::Unsupported)
    );
    instance.destroy().unwrap();
}

/// A prompt with an image runs the encoder on every rank before the step
/// opens (a record the encoder refuses reaches no device and opens none),
/// and the eager prefill overrides the image rows of the embedding by the
/// encoder's rows once the all-reduce assembled it, right before the
/// streams expand it.
#[test]
fn the_vision_encoder_runs_before_the_step_and_injects_into_the_prefill() {
    use crate::buffers::Buffer;
    let host = FakeHost::new((3, 32));
    let mut instance = create_with_encoder(&host);
    let handle = instance.handle();
    crate::instance::run_lowered_stages_alone(handle);
    // An odd grid the declaration refuses: no device work, no step.
    let start = host.device.journal.borrow().len();
    assert_eq!(
        instance
            .execute(&image_prompt(7, [1, 7, 8]), &mut DraftSink::default())
            .unwrap_err(),
        BundleAbiError::Status(BundleStatusV1::InvalidArgument)
    );
    assert_eq!(host.device.journal.borrow().len(), start);
    // An image of 16 tokens between two text rows.
    let start = host.device.journal.borrow().len();
    instance
        .execute(&image_prompt(7, [1, 8, 8]), &mut DraftSink::default())
        .unwrap_or_else(|error| panic!("{error:?} {:?}", host.logs.borrow()));
    let streams = launches_by_stream(&host, start);
    assert_eq!(streams.len(), 4);
    for rank in 0..4 {
        let reduced = crate::instance::scratch_span(handle, rank, Buffer::Reduced);
        let encoder = |line: &String| line.contains(" chop_enc_");
        // Peers' all-reduces bind the rank's rows too; its own stream
        // injects into them.
        let stream = streams
            .values()
            .find(|launches| {
                launches
                    .iter()
                    .any(|line| encoder(line) && line.contains(&format!("{reduced:?}")))
            })
            .unwrap();
        let inject = stream
            .iter()
            .position(|line| encoder(line) && line.contains(&format!("{reduced:?}")))
            .expect("the prefill injects the image rows");
        let text = stream.iter().position(|line| !encoder(line)).unwrap();
        // The tower, then the target's embedding and its all-reduce, the
        // inject, and the streams' expansion of the injected rows.
        assert!(text > 0 && text < inject);
        assert!(stream[..text].iter().all(encoder));
        assert!(stream[text..inject].iter().all(|line| !encoder(line)));
        assert!(
            stream[inject + 1].contains(model_glm5next_contract::sm120::engines::BROADCAST_SYMBOL),
            "{}",
            stream[inject + 1]
        );
        assert_eq!(stream.iter().filter(|line| encoder(line)).count(), text + 1);
    }
    instance.commit().unwrap();
    instance.destroy().unwrap();
    assert_eq!(host.acquired.get(), host.released.get());
}

/// The MTP drafter embeds the prompt's token ids at image rows, as vLLM's
/// GLM MTP does (its drafter takes text-only inputs: no multimodal
/// embeddings reach it), and no token at position 0: only the target's
/// prefill injects the encoder's rows, and the drafter's extend pass over
/// the completed rows uploads the next token of every row.
#[test]
fn the_drafter_embeds_token_ids_at_image_rows() {
    use crate::buffers::Buffer;
    let host = FakeHost::with_format((3, 32), Shape::TP4.kv_formats()[1]);
    let mut instance = create_with_encoder(&host);
    let handle = instance.handle();
    crate::instance::run_lowered_stages_alone(handle);
    let start = host.device.journal.borrow().len();
    let request = image_prompt(7, [1, 8, 8]);
    instance
        .execute(&request, &mut DraftSink::default())
        .unwrap_or_else(|error| panic!("{error:?} {:?}", host.logs.borrow()));
    for (rank, stream) in launches_by_stream(&host, start).values().enumerate() {
        let reduced = crate::instance::scratch_span(handle, rank, Buffer::Reduced);
        let injects = stream
            .iter()
            .filter(|line| line.contains(" chop_enc_") && line.contains(&format!("{reduced:?}")))
            .count();
        assert_eq!(injects, 1, "rank {rank}");
    }
    // The prompt's 18 tokens: the drafter's rows 0..17 read tokens 1..18,
    // the first at position 0 none.
    let batch = runtime_bundle_sdk::BatchCodecV1::decode(&request).unwrap();
    let tokens = batch.items()[0].input_tokens().to_vec();
    assert_eq!(tokens.len(), 18);
    let expected = std::iter::once(crate::drafter::NO_TOKEN)
        .chain(tokens[2..].iter().copied())
        .flat_map(u32::to_le_bytes)
        .collect::<Vec<_>>();
    let log = host.upload_log.borrow();
    let drafter = log
        .iter()
        .filter(|(line, _, bytes)| {
            *line > start
                && bytes
                    .windows(expected.len())
                    .any(|window| window == expected)
        })
        .count();
    assert_eq!(drafter, 4, "every rank uploads the drafter's token ids");
    drop(log);
    instance.commit().unwrap();
    instance.destroy().unwrap();
    assert_eq!(host.acquired.get(), host.released.get());
}

/// A package carrying the precision component: create authenticates it
/// against the projections' kernels and binds every rank's store over its
/// placed views, by checkpoint name (block FP8 codes with their scales,
/// BF16 codes with the zero rows their offers read). A component missing
/// an operation, or reading another layer's scales, is refused before the
/// KV pool, and everything create acquired is given back.
#[test]
fn create_binds_the_projection_stores() {
    use kernel_core::precision::{ExecutionPhase, OperationUse};
    let host = FakeHost::new((3, 32));
    let roles = [crate::projections::COMPONENT_ROLE.as_bytes().to_vec()];
    let record =
        crate::config::tests::record_naming(DECODE_BATCH, &options(2048), host.format, &roles);
    let mut instance = api()
        .create(HostServicesTableV1::new(Rc::clone(&host)), &record)
        .unwrap_or_else(|error| panic!("fake-host create: {error:?} {:?}", host.logs.borrow()));
    let handle = instance.handle();
    let decode = OperationUse {
        phase: ExecutionPhase::Decode,
        capacity: 8,
        graph: true,
    };
    for rank in 0..4 {
        // A BF16 KDA projection and a block-FP8 latent one of layer 3.
        let linear = crate::instance::projection_weights(handle, rank, "linear.q.0.bf16", decode)
            .expect("the KDA query projection");
        let name = "model.language_model.layers.0.self_attn.q_proj.weight";
        assert_eq!(linear.codes, crate::instance::view_span(handle, rank, name));
        assert!(linear.local_scales.is_none());
        let latent = crate::instance::projection_weights(handle, rank, "latent.q-b.3.bf16", decode)
            .expect("the latent query projection");
        let stem = "model.language_model.layers.3.self_attn.q_b_proj";
        assert_eq!(
            latent.codes,
            crate::instance::view_span(handle, rank, &format!("{stem}.weight"))
        );
        assert_eq!(
            latent.local_scales,
            Some(crate::instance::view_span(
                handle,
                rank,
                &format!("{stem}.weight_scale_inv")
            ))
        );
        // The KDA beta projection's padded rows.
        let beta = crate::instance::projection_weights(handle, rank, "linear.beta.0.bf16", decode)
            .expect("the KDA beta projection");
        let name = "model.language_model.layers.0.self_attn.b_proj.weight";
        assert_eq!(beta.codes, crate::instance::view_span(handle, rank, name));
        assert_eq!(beta.codes.length, 64 * 4096 * 2);
    }
    instance.destroy().unwrap();
    assert_eq!(host.acquired.get(), host.released.get());
    // An operation missing from the component.
    let host = FakeHost::new((3, 32));
    let mut document = kernel_core::precision::component::decode(&host.precision.borrow()).unwrap();
    document.operations.pop();
    *host.precision.borrow_mut() = kernel_core::precision::component::encode(&document).unwrap();
    let created = api().create(HostServicesTableV1::new(Rc::clone(&host)), &record);
    assert!(created.is_err());
    assert!(!host.kv_acquired.get());
    assert_eq!(host.acquired.get(), host.released.get());
    // An operation reading another layer's scales of the same size.
    let host = FakeHost::new((3, 32));
    let mut document = kernel_core::precision::component::decode(&host.precision.borrow()).unwrap();
    let selection = document
        .operations
        .iter_mut()
        .find(|s| s.source.id == "latent.q-b.3.bf16")
        .unwrap();
    let scale = "model.language_model.layers.7.self_attn.q_b_proj.weight_scale_inv";
    for operation in [&mut selection.source, &mut selection.effective] {
        operation.weights[0]
            .local_scale
            .as_mut()
            .unwrap()
            .tensor
            .tensor = scale.into();
    }
    *host.precision.borrow_mut() = kernel_core::precision::component::encode(&document).unwrap();
    let created = api().create(HostServicesTableV1::new(Rc::clone(&host)), &record);
    assert!(created.is_err());
    assert!(!host.kv_acquired.get());
    assert_eq!(host.acquired.get(), host.released.get());
    assert!(
        host.logs
            .borrow()
            .iter()
            .any(|line| line.contains("scales other than"))
    );
}

/// With the precision component, the projection stages launch through
/// each rank's store: a decode step's KDA projections of the normed input
/// (q, k, v, forget-a, beta and gate-a) as one stacked launch per layer in
/// place of their GEMMs and copies, the latent and output projections, and
/// the head's GEMM over the bucket's rows; a verify step's at its bucket; a scored prefill's
/// head at its 64 rows; a drafter's input projection and head; the
/// indexer's pooling gates through the typed GEMM planner; and each
/// router's GEMM followed by the compaction of its padded scores. Every
/// stage lowers: the steps run without leaving any stage out.
/// Re-pinned when GLM's own draft argmax kernels left its row cubin (two
/// fewer resolves per rank at create; the ids issued after them renumber).
#[test]
fn projection_stages_launch_through_the_stores() {
    use kernel_core::precision::{ExecutionPhase, OperationUse};
    let host = FakeHost::new((3, 32));
    let roles = [crate::projections::COMPONENT_ROLE.as_bytes().to_vec()];
    let record =
        crate::config::tests::record_naming(DECODE_BATCH, &options(2048), host.format, &roles);
    let mut instance = {
        api()
            .create(HostServicesTableV1::new(Rc::clone(&host)), &record)
            .unwrap_or_else(|error| panic!("fake-host create: {error:?} {:?}", host.logs.borrow()))
    };
    let handle = instance.handle();
    crate::instance::seed(handle, 7, 0, 100);
    crate::instance::seed(handle, 8, 5, 64);
    let decode = step(&[(7, 0, 100, &[11]), (8, 5, 64, &[12])]);
    let stages = |id: &str, phase, capacity| {
        crate::instance::projection_stages(
            handle,
            0,
            id,
            OperationUse {
                phase,
                capacity,
                graph: phase != ExecutionPhase::Prefill,
            },
        )
    };
    // How often each consecutive run of `symbols` launched, over every rank.
    let runs = |lines: &[String], symbols: &[String]| {
        lines
            .windows(symbols.len())
            .filter(|window| {
                window
                    .iter()
                    .zip(symbols)
                    .all(|(line, symbol)| line.contains(&format!(" {symbol} ")))
            })
            .count()
    };
    let launches = |start: usize| -> Vec<String> {
        issued(&host, start)
            .into_iter()
            .filter(|line| line.starts_with("launch "))
            .collect()
    };
    let kda = Shape::TP4.linear_layers() as usize;
    let stacked = |capacity: u32| {
        vec![format!(
            "chop_gemm_stacked_glm5next_kda_inputs_m{capacity}_s4x1_pf_v1"
        )]
    };
    let start = host.device.journal.borrow().len();
    instance.execute(&decode, &mut NoRows::default()).unwrap();
    let lines = launches(start);
    // The six KDA input projections' decode and verify uses are bound to
    // their stacked entry, which runs once per KDA layer and rank; none of
    // their layout copies runs.
    let inputs = [
        "linear.q.0.bf16",
        "linear.k.0.bf16",
        "linear.v.0.bf16",
        "linear.forget-a.0.bf16",
        "linear.beta.0.bf16",
        "linear.gate-a.0.bf16",
    ];
    for id in inputs {
        assert_eq!(stages(id, ExecutionPhase::Decode, 2), stacked(2), "{id}");
        assert_eq!(stages(id, ExecutionPhase::Verify, 8), stacked(8), "{id}");
    }
    assert_eq!(runs(&lines, &stacked(2)), 4 * kda);
    assert!(
        lines
            .iter()
            .all(|line| !line.contains(" chop_proj_bf16_copy_n2048_k1_")
                && !line.contains(" chop_proj_bf16_copy_n16_k1_"))
    );
    // Each latent layer's indexer key projection and pooling gate run one
    // `[128, 4096]` entry, which the KDA gates' down projections (stacked
    // here) run in prefill.
    let gate = stages("latent.indexer.k.3.bf16", ExecutionPhase::Decode, 2);
    assert_eq!(runs(&lines, &gate), 4 * 2 * 11);
    // Each MoE layer's router (42), then its scores' compaction (the
    // shared column slice).
    let mut router = stages("moe.router.3.f32", ExecutionPhase::Decode, 2);
    router.push(model_glm5next_contract::sm120::engines::SLICE_SYMBOL.into());
    assert_eq!(runs(&lines, &router), 4 * 42);
    for (id, count) in [("latent.q-b.3.bf16", 4 * 11), ("head.f32", 4)] {
        let symbols = stages(id, ExecutionPhase::Decode, 2);
        assert_eq!(symbols.len(), 1, "{id}");
        assert!(runs(&lines, &symbols) >= count, "{id}");
    }
    // Re-pinned when commit's state copy symbol joined create (the handles),
    // and when the shared gate/up overlap became the default (the shared gate/ups on the side
    // streams; `fake_host_tests::shared_overlap` proves the launches the
    // one-stream program's), and when the strip-chain entries took the
    // kernels that fence their ring reads before release (their symbols),
    // and when the KDA input projections became one stacked launch.
    // Re-pinned when prefill mixers from 144 rows ran the tiled mix-collapse
    // and its two symbols joined create (the handles), and when the
    // projections' staged-tile cubin joined create (the handles), and when
    // the stream expansion and the score compaction launched the shared row
    // copies (their symbols, shape arguments and the handles; identity
    // change, accuracy unaffected).
    check_journal(
        &lines,
        (4268, 0x74e2b6c1134c2d4c),
        "glm5next-decode-projections.txt",
    )
    .unwrap();
    instance.commit().unwrap();
    // A verify step of 5 rows: its 8-row bucket.
    let verify = step(&[(7, 0, 101, &[21, 22, 23, 24]), (8, 5, 65, &[13])]);
    let start = host.device.journal.borrow().len();
    instance.execute(&verify, &mut NoRows::default()).unwrap();
    let lines = launches(start);
    let symbols = stages("head.f32", ExecutionPhase::Verify, 8);
    assert_eq!(runs(&lines, &symbols), 4);
    assert_eq!(runs(&lines, &stacked(8)), 4 * kda);
    instance.commit_accepted(&[kept(7, 2)]).unwrap();
    instance.destroy().unwrap();
    assert_eq!(host.acquired.get(), host.released.get());
    // A scored prefill chunk: the head over its 64 gathered rows.
    let host = FakeHost::new((0, 0));
    let mut instance = api()
        .create(HostServicesTableV1::new(Rc::clone(&host)), &record)
        .unwrap();
    let handle = instance.handle();
    instance
        .execute(
            &prompt_chunk_scoring(9, 0, 64, false, true),
            &mut NoRows::default(),
        )
        .unwrap();
    let lines: Vec<String> = issued(&host, 0)
        .into_iter()
        .filter(|line| line.starts_with("launch "))
        .collect();
    let symbols = crate::instance::projection_stages(
        handle,
        0,
        "head.f32",
        OperationUse {
            phase: ExecutionPhase::Prefill,
            capacity: 64,
            graph: false,
        },
    );
    assert_eq!(runs(&lines, &symbols), 4);
    instance.commit().unwrap();
    instance.destroy().unwrap();
    assert_eq!(host.acquired.get(), host.released.get());
    // A drafting instance's extend pass after a prompt's last chunk: the
    // drafter's input projection over its side-by-side rows, and the head
    // over its draft rows beside the target's over the prompt's last row.
    let format = Shape::TP4.kv_formats()[1];
    let host = FakeHost::with_format((3, 32), format);
    let record = crate::config::tests::record_naming(DECODE_BATCH, &options(2048), format, &roles);
    let mut instance = api()
        .create(HostServicesTableV1::new(Rc::clone(&host)), &record)
        .unwrap();
    let handle = instance.handle();
    instance
        .execute(&prompt_chunk(7, 0, 10, true), &mut DraftSink::default())
        .unwrap();
    let lines: Vec<String> = issued(&host, 0)
        .into_iter()
        .filter(|line| line.starts_with("launch "))
        .collect();
    let prefill = OperationUse {
        phase: ExecutionPhase::Prefill,
        capacity: 8192,
        graph: false,
    };
    let eh = crate::instance::projection_stages(handle, 0, "mtp.eh.45.bf16", prefill);
    assert_eq!(runs(&lines, &eh), 4);
    let head = OperationUse {
        capacity: 64,
        ..prefill
    };
    let head = crate::instance::projection_stages(handle, 0, "head.f32", head);
    assert_eq!(runs(&lines, &head), 2 * 4);
    instance.commit().unwrap();
    instance.destroy().unwrap();
    assert_eq!(host.acquired.get(), host.released.get());
}

/// A record naming part of the encoder's components is refused before
/// create acquires anything.
#[test]
fn a_partial_encoder_is_refused_at_create() {
    let host = FakeHost::new((3, 32));
    let mut roles = crate::config::tests::encoder_roles();
    roles.pop();
    let created = api().create(
        HostServicesTableV1::new(Rc::clone(&host)),
        &crate::config::tests::record_naming(
            DECODE_BATCH,
            &options(2048),
            Shape::TP4.kv_formats()[0],
            &roles,
        ),
    );
    assert!(matches!(
        created,
        Err(BundleAbiError::Status(BundleStatusV1::Unsupported))
    ));
    assert_eq!(host.acquired.get(), 0);
    assert!(host.device.journal.borrow().is_empty());
}

/// Every text rank index is admitted and bound to the image the host
/// serves for its rank before any rank context opens or the image is
/// opened: an index naming another image creates no stream, and the log
/// names the rank.
#[test]
fn a_rank_index_naming_another_image_fails_before_any_rank_opens() {
    for rank in [0, 3] {
        let mut host = Rc::into_inner(FakeHost::new((3, 32)))
            .expect("a new host")
            .into_inner();
        host.planned[rank].identity = "f".repeat(64);
        let host = Rc::new(Backend::new(host));
        let created = api().create(
            HostServicesTableV1::new(Rc::clone(&host)),
            &record(DECODE_BATCH, &options(2048)),
        );
        assert!(created.is_err(), "rank {rank}");
        assert!(
            host.device.journal.borrow().is_empty(),
            "rank {rank}: device calls before the identity check: {:?}",
            host.device.journal.borrow()
        );
        let line = format!("GLM 5.3 Flash rank {rank} index names another image");
        assert!(
            host.logs.borrow().contains(&line),
            "{line}: {:?}",
            host.logs.borrow()
        );
        assert_eq!(host.acquired.get(), host.released.get(), "rank {rank}");
    }
}

/// The native module exports a callback for each declared capability whose
/// host path calls into the bundle, and declares each callback it exports,
/// so the host does not refuse the package before create.
#[test]
fn the_exported_api_matches_the_declared_capabilities() {
    let api = api();
    let declared = |capability: &str| CAPABILITIES.contains(&capability);
    assert_eq!(
        declared(REQUEST_RETIREMENT_CAPABILITY_V1),
        api.supports_request_retirement()
    );
    assert_eq!(
        declared(PREFIX_CACHE_CAPABILITY_V1),
        api.supports_prefix_cache()
    );
    assert_eq!(
        declared(PREFIX_CACHE_STATE_SLOT_CAPABILITY_V1),
        api.supports_prefix_state_slot()
    );
    // Offload has no callback: a swapped request's state returns through
    // the prefix snapshot and its restore into a host-chosen slot.
    if declared(KV_OFFLOAD_CAPABILITY_V1) {
        assert!(api.supports_prefix_cache() && api.supports_prefix_state_slot());
    }
    // Verify steps of host drafts commit their accepted rows.
    assert_eq!(
        declared(SPECULATIVE_CAPABILITY_V1),
        api.supports_speculation()
    );
    assert!(api.supports_speculation());
}

/// A sink of device logits views (the head itself is not lowered yet, so
/// their rows are not meaningful); it takes no copied rows.
#[derive(Default)]
pub(crate) struct NoRows(pub(crate) Vec<OwnedLogitsViewV1>);

impl LogitsRowSinkV1 for NoRows {
    fn limits(&self) -> ResultSinkLimitsV1 {
        ResultSinkLimitsV1::new(256, 1 << 20, 256 << 20)
    }

    fn emit_row(&mut self, _row: OwnedLogitsRowV1) -> HostServiceResultV1<()> {
        Err(BundleStatusV1::Unsupported)
    }

    fn emit_view(&mut self, view: OwnedLogitsViewV1) -> HostServiceResultV1<()> {
        self.0.push(view);
        Ok(())
    }
}

/// A sink that checks, as each view arrives, that its owner rank's
/// stream has no launch or replay queued after its last synchronization:
/// the host may read the rows at once.
struct ReadySink {
    host: Rc<Host>,
    /// Each view's rank, in emission order.
    ranks: Vec<u32>,
}

impl ReadySink {
    fn new(host: &Rc<Host>) -> Self {
        Self {
            host: Rc::clone(host),
            ranks: Vec::new(),
        }
    }

    /// Whether `rank`'s stream has device work queued since it was last
    /// synchronized.
    fn pending(&self, rank: u32) -> bool {
        let streams = self.host.device.stream_ranks.borrow();
        let journal = self.host.device.journal.borrow();
        streams
            .iter()
            .filter(|&(_, &on)| on == rank)
            .any(|(&stream, _)| {
                let last = |prefix: &str| {
                    let prefix = format!("{prefix} {stream} ");
                    journal.iter().rposition(|line| line.starts_with(&prefix))
                };
                let synchronized = journal
                    .iter()
                    .rposition(|line| *line == format!("synchronize {stream}"));
                last("launch").max(last("replay")) > synchronized
            })
    }
}

impl LogitsRowSinkV1 for ReadySink {
    fn limits(&self) -> ResultSinkLimitsV1 {
        ResultSinkLimitsV1::new(256, 1 << 20, 256 << 20)
    }

    fn emit_row(&mut self, _row: OwnedLogitsRowV1) -> HostServiceResultV1<()> {
        Err(BundleStatusV1::Unsupported)
    }

    fn emit_view(&mut self, view: OwnedLogitsViewV1) -> HostServiceResultV1<()> {
        let rank = view.raw().rank;
        assert!(
            !self.pending(rank),
            "rank {rank}'s rows are still being written"
        );
        self.ranks.push(rank);
        Ok(())
    }
}

/// Logits views reach the sink only once their owners completed: a
/// prefill's rows on rank 0, and a decode step's rows on each rank that
/// owns a block of them (eight items: two per rank), replayed or eager.
#[test]
fn logits_views_reach_the_sink_after_their_owners_complete() {
    let host = FakeHost::new((3, 32));
    let mut instance = create(&host);
    let handle = instance.handle();
    crate::instance::run_lowered_stages_alone(handle);
    let mut sink = ReadySink::new(&host);
    instance
        .execute(&prompt_chunk(1, 0, 64, true), &mut sink)
        .unwrap();
    instance.commit().unwrap();
    assert_eq!(sink.ranks, [0]);
    for request in 2..10 {
        crate::instance::seed(handle, request, request as u32, 64);
    }
    let tokens = (2..10).map(|request| [request as u32]).collect::<Vec<_>>();
    let items = (2..10)
        .zip(&tokens)
        .map(|(request, token)| (request, request, 64, &token[..]))
        .collect::<Vec<_>>();
    for _ in 0..2 {
        let mut sink = ReadySink::new(&host);
        instance.execute(&step(&items), &mut sink).unwrap();
        instance.cancel().unwrap();
        let mut owners = sink.ranks.clone();
        owners.dedup();
        assert_eq!(owners, [0, 1, 2, 3]);
        for request in 2..10 {
            crate::instance::seed(handle, request, request as u32, 64);
        }
    }
    instance.destroy().unwrap();
}

/// One step's items: (request, slot, previous tokens, inputs), each with
/// one KV page per 64 tokens, four pages per slot; several inputs make a
/// verify item.
fn step(items: &[(u64, u64, u64, &[u32])]) -> Vec<u8> {
    use runtime_bundle_sdk::{BatchAppendV1, BatchItemV1, BatchRequestV1, BatchWireVersionV1};
    let mut batch = Vec::new();
    let mut appends = Vec::new();
    let mut slots = Vec::new();
    for &(request, slot, previous, inputs) in items {
        let count = inputs.len() as u64;
        let end = previous + count;
        let pages = (0..end.div_ceil(64)).map(|page| slot * 4 + page).collect();
        batch.push(if count == 1 {
            BatchItemV1::new(request, true, 1, true, inputs.to_vec())
        } else {
            BatchItemV1::verify(request, count as u32, false, inputs.to_vec())
        });
        appends.push(BatchAppendV1::new(previous, count, end, pages, None));
        slots.push(slot);
    }
    BatchRequestV1::new(BatchWireVersionV1::V3, batch, appends)
        .with_state_slots(slots)
        .encode()
        .unwrap()
}

fn kept(request_id: u64, kept_rows: u32) -> BundleKeptRowsV1 {
    BundleKeptRowsV1 {
        struct_size: std::mem::size_of::<BundleKeptRowsV1>() as u32,
        reserved: 0,
        request_id,
        kept_rows,
        reserved_tail: 0,
    }
}

/// The journal lines from `from` on that are launches, uploads or copies.
pub(crate) fn issued(host: &FakeHost, from: usize) -> Vec<String> {
    host.device.journal.borrow()[from..]
        .iter()
        .filter(|line| {
            line.starts_with("launch ") || line.starts_with("upload ") || line.starts_with("copy ")
        })
        .cloned()
        .collect()
}

/// Decode and verify steps through the ledger: an incomplete program is
/// refused before the step opens; with the lowered stages alone, decode
/// uploads its inputs and issues the merged engines' launches, commit
/// advances the slots, a verify step's commit copies each item's last kept
/// checkpoint into its slot, cancel invalidates the slots the step touched,
/// and retirement frees a slot.
/// Re-pinned when GLM's own draft argmax kernels left its row cubin (two
/// fewer resolves per rank at create; the ids issued after them renumber).
#[test]
fn decode_and_verify_steps_run_through_the_ledger() {
    let host = FakeHost::new((3, 32));
    let mut instance = create(&host);
    let handle = instance.handle();
    crate::instance::seed(handle, 7, 0, 100);
    crate::instance::seed(handle, 8, 5, 64);
    let decode = step(&[(7, 0, 100, &[11]), (8, 5, 64, &[12])]);
    // Unlowered stages refuse the step before it opens.
    assert_eq!(
        instance
            .execute(&decode, &mut NoRows::default())
            .unwrap_err(),
        BundleAbiError::Status(BundleStatusV1::Unsupported)
    );
    assert!(
        host.logs
            .borrow()
            .iter()
            .any(|line| line.contains("is not lowered yet"))
    );
    assert_eq!(
        crate::instance::ledger_state(handle, 0),
        (Some((7, 100)), false)
    );

    crate::instance::run_lowered_stages_alone(handle);
    let start = host.device.journal.borrow().len();
    let mut views = NoRows::default();
    instance.execute(&decode, &mut views).unwrap();
    // One logits row per item, each on the rank that owns its block of the
    // 2-row bucket (one row per block): items 0 and 1 on ranks 0 and 1.
    let owners = views
        .0
        .iter()
        .map(|view| {
            let view = view.raw();
            (
                view.rank,
                view.row_count,
                view.live_width,
                view.first_item_index,
            )
        })
        .collect::<Vec<_>>();
    assert_eq!(owners, [(0, 1, 154_880, 0), (1, 1, 154_880, 1)]);
    let lines = issued(&host, start);
    // Every rank uploads its inputs and the KDA side inputs, then issues
    // the latent, KDA, indexer, mixer and MoE launches of a 2-item bucket.
    for symbol in [
        "latent",
        "channel_decay",
        "pooled_indexer",
        "sinkhorn",
        "moe_chain",
    ] {
        assert!(
            lines
                .iter()
                .any(|line| line.starts_with("launch ") && line.contains(symbol)),
            "{symbol}"
        );
    }
    // Re-pinned as stages lowered (norms and embedding; all-reduces,
    // widening, rounding and stream expansion; the logits exchange), as
    // create's new buffers shifted the handles, and when the fake pool grew
    // to 256 pages (the KV spans' lengths) and when the drafter's buffers
    // shifted the handles again, and when the comm flags and epochs were
    // sized for each protocol (their lengths), and when each one-shot row
    // count took its own state (the handles), and when the logits shard
    // grew to 64 scored rows (its length), and when the MoE took the F32
    // routed sum (its symbol, the MLP output's length and the handles), and
    // when the drafter's hidden inputs joined the scratch, and when a
    // non-drafting instance stopped allocating the drafter's, and when
    // create resolved the drafter's own row symbols (the handles), and when
    // it resolved the projections' symbols (the handles), and when their
    // padded BF16 entries resolved fewer (the handles), and when the
    // lowered BF16 projections' weight views took their zero rows (the
    // views' offsets), and when the KDA gates' low-rank buffers joined the
    // scratch (the handles), and when the head's own use profile added two
    // projection symbols (the handles), and when the projections' layout
    // copies and their temporary joined create (the handles), and when the
    // drafter's input halves took one buffer and 14 copies (the handles),
    // and when the padded BF16 projections' compactions joined (the
    // handles), and when the router's padded scores took a buffer and a row
    // symbol (the handles), and when the payload admission's symbols and
    // report buffer joined create (the handles), and when the layout copies
    // merged into one entry per layout (their symbols and the handles), and
    // when the latent attention kept every row's 64 fixed splits (its
    // kernels' outputs), and when every mixer ran the mix-collapse split
    // over CTAs (its symbol, grid, chain sums and tickets, and the handles),
    // and when commit's state copy symbol joined create (the handles), and
    // when create resolved the tiled KDA prefill stages (the handles), and
    // when the phased all-reduce's state and cubin joined create (the
    // handles), and when the 2-row bucket's all-reduces and rounds each
    // became one phased one-shot launch.
    // Re-pinned when prefill mixers from 144 rows ran the tiled mix-collapse
    // and its two symbols joined create (the handles), and when the
    // projections' staged-tile cubin joined create (the handles), and when
    // create checked every rank index before the ranks opened (the rank
    // streams' ids), and when the stream expansion and the score compaction
    // launched the shared row copies (their symbols, shape arguments and the
    // handles; identity change, accuracy unaffected).
    check_journal(
        &lines,
        (2836, 0xbb09f551c85c91c6),
        "glm5next-decode-step.txt",
    )
    .unwrap();
    assert!(crate::instance::ledger_state(handle, 0).1);
    instance.commit().unwrap();
    assert_eq!(
        crate::instance::ledger_state(handle, 0),
        (Some((7, 101)), false)
    );
    assert_eq!(
        crate::instance::ledger_state(handle, 5),
        (Some((8, 65)), false)
    );

    // A verify step of 4 rows for request 7 beside request 8's decode row:
    // commit keeps 2 of 7's rows, so its slot takes checkpoint row 1.
    let verify = step(&[(7, 0, 101, &[21, 22, 23, 24]), (8, 5, 65, &[13])]);
    let start = host.device.journal.borrow().len();
    let mut views = NoRows::default();
    instance.execute(&verify, &mut views).unwrap();
    // Every verify row and the decode row: five rows.
    assert_eq!(
        views.0.iter().map(|view| view.raw().row_count).sum::<u32>(),
        5
    );
    let launched = issued(&host, start).len();
    instance.commit_accepted(&[kept(7, 2)]).unwrap();
    let copies = issued(&host, start)[launched..].to_vec();
    // One state copy each of KDA conv and state over the 34 KDA layers and
    // of the tail over the paged layers, on each of the 4 ranks.
    let shape = Shape::TP4;
    assert_eq!(copies.len(), 4 * 3, "{copies:#?}");
    let state_copy = model_glm5next_contract::sm120::state_copy::SYMBOL;
    assert!(
        copies
            .iter()
            .all(|line| line.starts_with("launch ") && line.contains(state_copy))
    );
    for layers in [shape.linear_layers(), shape.paged_layers(false)] {
        assert!(
            copies
                .iter()
                .any(|line| line.contains(&format!("grid_y: {layers},"))),
            "{layers}"
        );
    }
    assert_eq!(
        crate::instance::ledger_state(handle, 0),
        (Some((7, 103)), false)
    );

    // A cancelled decode step leaves its slots invalid: both requests retire.
    instance
        .execute(
            &step(&[(7, 0, 103, &[31]), (8, 5, 66, &[14])]),
            &mut NoRows::default(),
        )
        .unwrap();
    instance.cancel().unwrap();
    assert_eq!(crate::instance::ledger_state(handle, 0), (None, false));
    assert_eq!(crate::instance::ledger_state(handle, 5), (None, false));
    crate::instance::seed(handle, 9, 2, 64);
    instance.retire_request(9).unwrap();
    assert_eq!(crate::instance::ledger_state(handle, 2), (None, false));
    instance.destroy().unwrap();
    assert_eq!(host.acquired.get(), host.released.get());
}

/// A prefill step of request `request` in slot 0: `count` prompt tokens
/// after `previous`, its pages from 0, logits only at the prompt's end.
fn prompt_chunk(request: u64, previous: u64, count: u64, last: bool) -> Vec<u8> {
    prompt_chunk_scoring(request, previous, count, last, false)
}

/// [`prompt_chunk`], with logits of every prompt row when `scored`.
fn prompt_chunk_scoring(
    request: u64,
    previous: u64,
    count: u64,
    last: bool,
    scored: bool,
) -> Vec<u8> {
    use runtime_bundle_sdk::{BatchAppendV1, BatchItemV1, BatchRequestV1, BatchWireVersionV1};
    let end = previous + count;
    let inputs = (previous..end).map(|token| token as u32).collect();
    let item = BatchItemV1::new(request, false, count as u32, last || scored, inputs)
        .with_prompt_rows(scored);
    BatchRequestV1::new(
        BatchWireVersionV1::V3,
        vec![item],
        vec![BatchAppendV1::new(
            previous,
            count,
            end,
            (0..end.div_ceil(64)).collect(),
            None,
        )],
    )
    .with_state_slots(vec![0])
    .encode()
    .unwrap()
}

/// A launch line without its device spans: the symbol, configuration and
/// scalars.
fn without_spans(line: &str) -> String {
    let mut out = String::new();
    let mut rest = line;
    while let Some(start) = rest.find("DeviceBuffer(") {
        out.push_str(&rest[..start]);
        let mut depth = 0;
        let mut end = start;
        for (offset, c) in rest[start..].char_indices() {
            match c {
                '(' => depth += 1,
                ')' => {
                    depth -= 1;
                    if depth == 0 {
                        end = start + offset + 1;
                        break;
                    }
                }
                _ => {}
            }
        }
        out.push_str("DeviceBuffer");
        rest = &rest[end..];
    }
    out.push_str(rest);
    out
}

/// The indexer launches of the journal from `from` on, without spans,
/// sorted.
fn indexer_work(host: &FakeHost, from: usize) -> Vec<String> {
    let mut work = issued(host, from)
        .iter()
        .filter(|line| line.starts_with("launch ") && line.contains("pooled_indexer"))
        .map(|line| without_spans(line))
        .collect::<Vec<_>>();
    work.sort();
    work
}

/// An eager prefill of 3000 rows runs its indexer in pieces of 2048 and
/// 952 rows, each uploading its own inputs: the same indexer work as the
/// prompt delivered as chunks of 2048 and 952 rows across two steps. The
/// step commits the prompt's tokens.
#[test]
fn a_long_prefill_runs_its_indexer_as_the_chunks_of_two_steps() {
    let one = FakeHost::new((0, 0));
    let mut instance = create(&one);
    crate::instance::run_lowered_stages_alone(instance.handle());
    let start = one.device.journal.borrow().len();
    instance
        .execute(&prompt_chunk(7, 0, 3000, true), &mut NoRows::default())
        .unwrap();
    let pieces = indexer_work(&one, start);
    // Each rank uploads both pieces' own inputs.
    let bytes = crate::instance::piece_bytes_for_tests();
    let uploads = issued(&one, start)
        .iter()
        .filter(|line| line.starts_with("upload ") && line.contains(&format!("length: {bytes} }}")))
        .count();
    assert_eq!(uploads, 2 * 4);
    instance.commit().unwrap();
    assert_eq!(
        crate::instance::ledger_state(instance.handle(), 0),
        (Some((7, 3000)), false)
    );
    instance.destroy().unwrap();

    let two = FakeHost::new((0, 0));
    let mut instance = create(&two);
    crate::instance::run_lowered_stages_alone(instance.handle());
    let start = two.device.journal.borrow().len();
    instance
        .execute(&prompt_chunk(7, 0, 2048, false), &mut NoRows::default())
        .unwrap();
    instance.commit().unwrap();
    instance
        .execute(&prompt_chunk(7, 2048, 952, true), &mut NoRows::default())
        .unwrap();
    instance.commit().unwrap();
    let chunks = indexer_work(&two, start);
    assert!(!pieces.is_empty());
    let shape = Shape::TP4;
    // Key write and selection per sparse layer, piece and rank: both pieces
    // (2,048 and 952 rows) take one split, so each selection is its
    // one-split form's one launch.
    assert_eq!(pieces.len() as u64, 2 * 2 * 4 * shape.sparse_layers());
    assert_eq!(pieces, chunks);
    assert_eq!(
        crate::instance::ledger_state(instance.handle(), 0),
        (Some((7, 3000)), false)
    );
    instance.destroy().unwrap();
}

/// A prefill phase scores up to `SCORED_ROWS` rows, the package's
/// `maximum_scored_rows` (a multiple of the prefill granule, which the
/// host's scheduler requires), above the 32 verify rows: a chunk scoring
/// its 64 prompt rows runs and emits them, one scoring 65 is refused before
/// the step opens.
#[test]
fn a_prefill_scores_up_to_its_declared_rows() {
    let host = FakeHost::new((0, 0));
    let mut instance = create(&host);
    crate::instance::run_lowered_stages_alone(instance.handle());
    assert_eq!(crate::buffers::SCORED_ROWS, 64);
    assert_eq!(
        instance
            .execute(
                &prompt_chunk_scoring(7, 0, 65, false, true),
                &mut NoRows::default()
            )
            .unwrap_err(),
        BundleAbiError::Status(BundleStatusV1::Unsupported)
    );
    assert!(
        host.logs
            .borrow()
            .iter()
            .any(|line| line.contains("65 scored rows exceed the head's 64 rows"))
    );
    let mut sink = NoRows::default();
    instance
        .execute(&prompt_chunk_scoring(7, 0, 64, false, true), &mut sink)
        .unwrap();
    let rows: u32 = sink.0.iter().map(|view| view.raw().row_count).sum();
    assert_eq!(rows, 64);
    instance.commit().unwrap();
    instance.destroy().unwrap();
}

/// Whether every rank queued the same collectives.
pub(crate) fn matched(host: &FakeHost) -> bool {
    let counts = host.device.collectives.borrow();
    counts.len() == 4
        && counts
            .values()
            .all(|count| Some(count) == counts.values().next())
}

/// After a step that left collectives unmatched (from journal line
/// `journal` on, its last collective submission `last`): the instance is
/// poisoned and says so, queued nothing after that submission, and refuses
/// every later call (execute, retirement, prefix state, destroy) without
/// touching a stream or giving back a resource.
fn assert_poisoned(
    host: &FakeHost,
    instance: &mut BundleInstanceV1,
    journal: usize,
    last: impl Fn(&str) -> bool,
) {
    assert!(!matched(host));
    assert!(
        host.device
            .logs
            .borrow()
            .iter()
            .chain(host.logs.borrow().iter())
            .any(|line| line.contains("the instance is poisoned"))
    );
    let added = host.device.journal.borrow()[journal..].to_vec();
    let at = added
        .iter()
        .rposition(|line| last(line))
        .expect("a submission");
    let queued = [
        "launch ",
        "replay ",
        "copy ",
        "upload ",
        "zero ",
        "synchronize ",
    ];
    assert!(
        added[at + 1..]
            .iter()
            .all(|line| !queued.iter().any(|prefix| line.starts_with(prefix))),
        "{:?}",
        &added[at + 1..]
    );
    let released = host.released.get();
    let journal = host.device.journal.borrow().len();
    assert!(
        instance
            .execute(&prompt_chunk(9, 0, 64, true), &mut NoRows::default())
            .is_err()
    );
    assert!(instance.retire_request(7).is_err());
    assert!(instance.describe_prefix_state().is_err());
    assert!(instance.destroy().is_err());
    assert!(instance.is_poisoned());
    assert_eq!(host.released.get(), released);
    assert_eq!(host.device.journal.borrow().len(), journal);
}

/// An eager prefill whose second rank fails to queue its first all-reduce,
/// after the first rank queued its own, leaves the first rank waiting at a
/// collective the second never joins: the instance is poisoned.
#[test]
fn a_failed_launch_after_a_queued_rank_poisons_the_instance() {
    use model_glm5next_contract::sm120::engines::{PHASED_SYMBOLS, bf16_tuned};
    let two_shot = PHASED_SYMBOLS[1];
    assert_eq!(bf16_tuned(64).symbol(), two_shot);
    let host = FakeHost::new((0, 0));
    let mut instance = create(&host);
    crate::instance::run_lowered_stages_alone(instance.handle());
    crate::instance::keep_last_chunks_eager(instance.handle());
    instance
        .execute(&prompt_chunk(7, 0, 64, false), &mut NoRows::default())
        .unwrap();
    instance.commit().unwrap();
    assert!(matched(&host));
    *host.fail_launch_on_rank.borrow_mut() = Some((two_shot.into(), 1));
    let journal = host.device.journal.borrow().len();
    assert_eq!(
        instance
            .execute(&prompt_chunk(7, 64, 64, true), &mut NoRows::default())
            .unwrap_err(),
        BundleAbiError::Status(BundleStatusV1::HostError)
    );
    assert!(
        host.fail_launch_on_rank.borrow().is_none(),
        "the all-reduce failed"
    );
    let added = host.device.journal.borrow()[journal..].to_vec();
    let launched = |symbol: &str| {
        added
            .iter()
            .filter(|line| line.starts_with("launch ") && line.contains(symbol))
            .count()
    };
    // Rank 0's first all-reduce, which rank 1 never joins; rank 1 queued
    // its launches before it, and nothing follows its failed one.
    assert_eq!(launched(two_shot), 1);
    assert_poisoned(&host, &mut instance, journal, |line| {
        line.starts_with("launch ")
    });
}

/// A decode replay that fails after the first rank queued its first piece
/// poisons the instance.
#[test]
fn a_partial_graph_replay_poisons_the_instance() {
    let host = FakeHost::new((3, 32));
    let mut instance = create(&host);
    let handle = instance.handle();
    crate::instance::run_lowered_stages_alone(handle);
    crate::instance::seed(handle, 7, 0, 100);
    instance
        .execute(&step(&[(7, 0, 100, &[1])]), &mut NoRows::default())
        .unwrap();
    instance.commit().unwrap();
    assert!(matched(&host));
    host.device.fail_graph_launch.set(Some(1));
    let journal = host.device.journal.borrow().len();
    assert_eq!(
        instance
            .execute(&step(&[(7, 0, 101, &[2])]), &mut NoRows::default())
            .unwrap_err(),
        BundleAbiError::Status(BundleStatusV1::HostError)
    );
    assert_poisoned(&host, &mut instance, journal, |line| {
        line.starts_with("replay ")
    });
}

/// A panic once the first rank queued its share of an eager prefill
/// unwinds through the submission: the instance is poisoned.
#[test]
fn a_panic_after_a_queued_rank_poisons_the_instance() {
    let host = FakeHost::new((0, 0));
    let mut instance = create(&host);
    crate::instance::run_lowered_stages_alone(instance.handle());
    let journal = host.device.journal.borrow().len();
    crate::instance::panic_after(0);
    assert_eq!(
        instance
            .execute(&prompt_chunk(7, 0, 64, true), &mut NoRows::default())
            .unwrap_err(),
        BundleAbiError::Status(BundleStatusV1::Panic)
    );
    assert!(!matched(&host));
    let released = host.released.get();
    let journal_after = host.device.journal.borrow().len();
    assert!(journal_after > journal);
    assert!(
        instance
            .execute(&prompt_chunk(9, 0, 64, true), &mut NoRows::default())
            .is_err()
    );
    assert!(instance.destroy().is_err());
    assert!(instance.is_poisoned());
    assert_eq!(host.released.get(), released);
    assert_eq!(host.device.journal.borrow().len(), journal_after);
}

/// A one-slot instance of the served program, its projections included,
/// as `--max-sequences 1` creates it.
fn create_one_slot(host: &Rc<Host>) -> BundleInstanceV1 {
    host.items.set(1);
    let roles = [crate::projections::COMPONENT_ROLE.as_bytes().to_vec()];
    let record = crate::config::tests::record_naming(1, &options(2048), host.format, &roles);
    api()
        .create(HostServicesTableV1::new(Rc::clone(host)), &record)
        .unwrap_or_else(|error| panic!("fake-host create: {error:?} {:?}", host.logs.borrow()))
}

/// Checks the eager launches of journal lines from `from` on (none of
/// them captured): every rank issues the same collectives, symbol and
/// launch shape, in the same order, and no rank has a launch queued past
/// its k-th collective before every peer queued its own k-th. A stream
/// holds a bounded number of pending launches: once a rank spins at a
/// collective its peers have not reached, a host that keeps queueing that
/// rank's program blocks in the launch call and never queues the peers.
/// Returns each rank's launch and collective counts.
fn assert_ranks_advance_together(host: &FakeHost, from: usize) -> Vec<(usize, usize)> {
    let journal = host.device.journal.borrow();
    assert!(
        journal[from..]
            .iter()
            .all(|line| !line.starts_with("capture ")),
        "an eager step captured a graph"
    );
    let stream_ranks = host.device.stream_ranks.borrow();
    let mut collectives = vec![Vec::<String>::new(); 4];
    let mut launches = vec![0; 4];
    for line in &journal[from..] {
        let mut words = line.splitn(4, ' ');
        if words.next() != Some("launch") {
            continue;
        }
        let stream: u32 = words.next().unwrap().parse().unwrap();
        let rank = stream_ranks[&stream] as usize;
        let symbol = words.next().unwrap();
        let queued = collectives[rank].len();
        let behind = collectives
            .iter()
            .enumerate()
            .find(|(_, peer)| peer.len() < queued);
        if let Some((peer, _)) = behind {
            panic!(
                "rank {rank} queues {symbol} (launch {}) past its collective {queued} before \
                 rank {peer} queued its own",
                launches[rank]
            );
        }
        launches[rank] += 1;
        if symbol.starts_with("chop_comm_") {
            // The symbol and launch configuration, without arguments.
            let config = words.next().unwrap();
            let shape = config.split(" [").next().unwrap();
            collectives[rank].push(format!("{symbol} {shape}"));
        }
    }
    for (rank, sequence) in collectives.iter().enumerate() {
        assert_eq!(sequence, &collectives[0], "rank {rank}'s collectives");
    }
    launches
        .into_iter()
        .zip(collectives.iter().map(Vec::len))
        .collect()
}

/// The first step of the host's startup warmup on a one-slot instance
/// without speculation (`--max-sequences 1 --speculative-tokens 0`): an
/// eager prefill of a short prompt that scores its last row (here kept
/// eager; served, such a last chunk captures at its first run), then its
/// first decode step. The prefill is far longer than a stream's pending
/// launches (bring-up 2 hung here: rank 0 spun at its first all-reduce
/// while the host, still queueing rank 0's program, never reached ranks 1
/// to 3), so every rank must reach each collective before any rank queues
/// past it. The decode step's graphs replay with matched collectives.
#[test]
fn the_warmup_prefill_queues_every_rank_up_to_each_collective_in_turn() {
    let host = FakeHost::new((0, 0));
    let mut instance = create_one_slot(&host);
    crate::instance::keep_last_chunks_eager(instance.handle());
    for prompt in [1, 2] {
        let request = 7 + prompt;
        let start = host.device.journal.borrow().len();
        let mut sink = NoRows::default();
        instance
            .execute(&prompt_chunk(request, 0, prompt, true), &mut sink)
            .unwrap_or_else(|error| panic!("{error:?} {:?}", host.logs.borrow()));
        let counts = assert_ranks_advance_together(&host, start);
        assert!(
            counts.iter().all(|&(launches, _)| launches > 1024),
            "{counts:?}"
        );
        assert!(counts.iter().all(|&(_, collectives)| collectives > 90));
        assert_eq!(
            sink.0.iter().map(|view| view.raw().row_count).sum::<u32>(),
            1
        );
        instance.commit().unwrap();
        assert!(matched(&host));
        let start = host.device.journal.borrow().len();
        instance
            .execute(&step(&[(request, 0, prompt, &[1])]), &mut NoRows::default())
            .unwrap_or_else(|error| panic!("{error:?} {:?}", host.logs.borrow()));
        assert!(count(&host, start, "captured ") > 0 || prompt == 2);
        assert!(count(&host, start, "replay ") > 0);
        instance.commit().unwrap();
        assert!(matched(&host));
        instance.retire_request(request).unwrap();
    }
    instance.destroy().unwrap();
}

/// A step at a bucket whose graphs every rank captured lowers no program:
/// the replay needs none, and lowering the whole program of four ranks
/// took most of a step's host time (30 to 50 ms per AR step, about four
/// times that with MTP's drafter runs). Only a missing graph's ranks are
/// lowered: the first decode step at a bucket, and again once the graphs
/// were released. With MTP, the drafter's extend pass, its draft steps
/// and the target's verify phase each lower only until their buckets are
/// captured.
#[test]
fn steps_at_captured_buckets_lower_no_program() {
    use crate::instance::lowered_programs;
    let host = FakeHost::new((0, 0));
    let mut instance = create_one_slot(&host);
    let handle = instance.handle();
    let before = lowered_programs();
    instance
        .execute(&prompt_chunk(7, 0, 16, true), &mut NoRows::default())
        .unwrap_or_else(|error| panic!("{error:?} {:?}", host.logs.borrow()));
    instance.commit().unwrap();
    assert_eq!(
        lowered_programs() - before,
        4,
        "the eager prefill lowers every rank"
    );
    let decode = |instance: &mut BundleInstanceV1, previous: u64| {
        let start = host.device.journal.borrow().len();
        let before = lowered_programs();
        instance
            .execute(&step(&[(7, 0, previous, &[1])]), &mut NoRows::default())
            .unwrap_or_else(|error| panic!("{error:?} {:?}", host.logs.borrow()));
        instance.commit().unwrap();
        assert_eq!(count(&host, start, "replay "), 12);
        (
            lowered_programs() - before,
            count(&host, start, "captured "),
        )
    };
    assert_eq!(
        decode(&mut instance, 16),
        (4, 12),
        "the first step captures"
    );
    assert_eq!(decode(&mut instance, 17), (0, 0));
    assert_eq!(decode(&mut instance, 18), (0, 0));
    crate::instance::release_graphs(handle);
    assert_eq!(
        decode(&mut instance, 19),
        (4, 12),
        "released graphs are captured again"
    );
    assert_eq!(decode(&mut instance, 20), (0, 0));
    instance.destroy().unwrap();

    let format = Shape::TP4.kv_formats()[1];
    let host = FakeHost::with_format((3, 4), format);
    host.items.set(1);
    let roles = [crate::projections::COMPONENT_ROLE.as_bytes().to_vec()];
    let record = crate::config::tests::record_naming(1, &options(2048), format, &roles);
    let mut instance = api()
        .create(HostServicesTableV1::new(Rc::clone(&host)), &record)
        .unwrap_or_else(|error| panic!("fake-host create: {error:?} {:?}", host.logs.borrow()));
    instance
        .execute(&prompt_chunk(7, 0, 16, true), &mut DraftSink::default())
        .unwrap_or_else(|error| panic!("{error:?} {:?}", host.logs.borrow()));
    instance.commit().unwrap();
    // Each step keeps two of its four rows, so from the second step on the
    // extend pass covers two rows (a verify bucket of its own).
    let mut lowered = Vec::new();
    for (index, previous) in [16, 18, 20, 22].into_iter().enumerate() {
        let start = host.device.journal.borrow().len();
        let before = lowered_programs();
        let mut sink = DraftSink::default();
        instance
            .execute(&drafted(7, previous, 4, 21 + index as u32), &mut sink)
            .unwrap_or_else(|error| panic!("{error:?} {:?}", host.logs.borrow()));
        assert_eq!(sink.drafts.len(), 1);
        instance.commit_accepted(&[kept(7, 2)]).unwrap();
        // The extend pass, two draft steps and the verify phase, three
        // pieces on every rank each.
        assert_eq!(count(&host, start, "replay "), 4 * 4 * 3);
        lowered.push((
            lowered_programs() - before,
            count(&host, start, "captured ") > 0,
        ));
    }
    // The first step lowers the extend pass, both draft steps (each is
    // planned before the first one captures their shared bucket) and the
    // verify phase; the second the extend pass at its new bucket; then
    // nothing.
    assert_eq!(lowered, [(16, true), (4, true), (0, false), (0, false)]);
    instance.destroy().unwrap();
}

/// An eager prefill whose exact shape ran before replays captured
/// programs (`instance::prefill_graphs`): a prompt-logprob request's
/// scored chunks of 64 rows run the first eagerly, capture every rank's
/// program at the second, which then replays it, and from the third on
/// lower nothing and launch only the replays. The capture holds exactly
/// the launches the eager run issued on every rank. Another shape runs
/// eagerly; released graphs are captured again from the second run on,
/// and destroy releases the rest.
#[test]
fn repeated_prefill_shapes_replay_their_captured_programs() {
    use crate::instance::lowered_programs;
    let host = FakeHost::new((0, 0));
    let mut instance = create_one_slot(&host);
    let handle = instance.handle();
    let mut previous = 0;
    let mut chunk = |instance: &mut BundleInstanceV1, rows: u64, scored: bool| {
        let start = host.device.journal.borrow().len();
        let before = lowered_programs();
        let mut sink = NoRows::default();
        instance
            .execute(
                &prompt_chunk_scoring(7, previous, rows, false, scored),
                &mut sink,
            )
            .unwrap_or_else(|error| panic!("{error:?} {:?}", host.logs.borrow()));
        if scored {
            assert_eq!(
                sink.0.iter().map(|view| view.raw().row_count).sum::<u32>(),
                rows as u32
            );
        }
        instance.commit().unwrap();
        assert!(matched(&host));
        previous += rows;
        let launches = launches_by_stream(&host, start);
        let replays = count(&host, start, "replay ");
        let captured = count(&host, start, "captured ");
        let released = count(&host, start, "release graph ");
        (
            (lowered_programs() - before, captured, replays, released),
            launches,
        )
    };
    let (first, eager) = chunk(&mut instance, 64, true);
    assert_eq!(first, (4, 0, 0, 0), "the first run is eager");
    let (second, capture) = chunk(&mut instance, 64, true);
    assert_eq!(second, (4, 12, 12, 0), "the second captures and replays");
    assert_eq!(capture, eager, "the capture holds the eager launches");
    for _ in 0..2 {
        let (later, launched) = chunk(&mut instance, 64, true);
        assert_eq!(later, (0, 0, 12, 0), "later runs only replay");
        assert!(launched.is_empty());
    }
    assert_eq!(chunk(&mut instance, 32, true).0, (4, 0, 0, 0));
    crate::instance::release_graphs(handle);
    assert_eq!(chunk(&mut instance, 64, true).0, (4, 0, 0, 0));
    assert_eq!(chunk(&mut instance, 64, true).0, (4, 12, 12, 0));
    assert_eq!(chunk(&mut instance, 64, true).0, (0, 0, 12, 0));
    instance.destroy().unwrap();
    assert_eq!(
        count(&host, 0, "release graph "),
        count(&host, 0, "captured "),
        "every captured piece is released by destroy at the latest"
    );
    assert_eq!(host.live(), Vec::<String>::new());
}

/// The host releases graphs only newest first, so a ninth repeated shape
/// cannot evict one older program from under newer graphs: its capture
/// first releases the eight programs above every other graph, here every
/// graph of the instance (their 96 pieces, newest first, as the fake host
/// enforces), then captures and replays. The released shapes run eagerly
/// again; destroy releases the rest.
#[test]
fn a_ninth_repeated_shape_releases_every_graph_before_it_captures() {
    let host = FakeHost::new((0, 0));
    let mut instance = create_one_slot(&host);
    let mut previous = 0;
    let mut chunk = |instance: &mut BundleInstanceV1, rows: u64| {
        let start = host.device.journal.borrow().len();
        instance
            .execute(
                &prompt_chunk_scoring(7, previous, rows, false, false),
                &mut NoRows::default(),
            )
            .unwrap_or_else(|error| panic!("{error:?} {:?}", host.logs.borrow()));
        instance.commit().unwrap();
        previous += rows;
        (
            count(&host, start, "captured "),
            count(&host, start, "replay "),
            count(&host, start, "release graph "),
        )
    };
    for rows in 1..=9 {
        assert_eq!(chunk(&mut instance, rows), (0, 0, 0));
        let released = if rows == 9 { 96 } else { 0 };
        assert_eq!(
            chunk(&mut instance, rows),
            (12, 12, released),
            "{rows} rows"
        );
    }
    assert_eq!(chunk(&mut instance, 9), (0, 12, 0));
    assert_eq!(chunk(&mut instance, 1), (0, 0, 0));
    instance.destroy().unwrap();
    assert_eq!(host.live(), Vec::<String>::new());
}

/// Prefill programs captured before a decode bucket (as at warmup) stay
/// under its graphs, and later shapes capture above them without
/// releasing the decode graphs: eight more shapes capture with nothing
/// released, the decode bucket replays without a capture, and the
/// seventeenth shape releases only the eight programs above the decode
/// graphs (96 pieces, newest first, as the fake host enforces). The
/// programs under the decode graphs still replay; destroy releases the
/// rest.
#[test]
fn later_prefill_shapes_release_only_the_programs_above_the_decode_graphs() {
    let host = FakeHost::new((0, 0));
    let mut instance = create_one_slot(&host);
    // Each prompt's last chunk stays eager, as any shape seen once.
    crate::instance::keep_last_chunks_eager(instance.handle());
    let mut previous = 0;
    let run = |instance: &mut BundleInstanceV1, request: &[u8]| {
        let start = host.device.journal.borrow().len();
        instance
            .execute(request, &mut NoRows::default())
            .unwrap_or_else(|error| panic!("{error:?} {:?}", host.logs.borrow()));
        instance.commit().unwrap();
        (
            count(&host, start, "captured "),
            count(&host, start, "replay "),
            count(&host, start, "release graph "),
        )
    };
    let mut shapes =
        |instance: &mut BundleInstanceV1, request, rows: std::ops::RangeInclusive<u64>| {
            let mut runs = Vec::new();
            for rows in rows {
                for _ in 0..2 {
                    runs.push(run(
                        instance,
                        &prompt_chunk_scoring(request, previous, rows, false, false),
                    ));
                    previous += rows;
                }
            }
            // The prompt's last chunk, a shape seen once, then its first
            // decode step.
            runs.push(run(
                instance,
                &prompt_chunk(request, previous, 40 + request, true),
            ));
            previous += 40 + request;
            runs.push(run(instance, &step(&[(request, 0, previous, &[1])])));
            instance.retire_request(request).unwrap();
            previous = 0;
            runs
        };
    let pinned = shapes(&mut instance, 7, 1..=8);
    let captures = |release| [(0, 0, 0), (12, 12, release)];
    assert_eq!(pinned[..16], captures(0).repeat(8));
    assert_eq!(pinned[16..], [(0, 0, 0), (12, 12, 0)], "the decode bucket");
    let above = shapes(&mut instance, 8, 9..=16);
    assert_eq!(above[..16], captures(0).repeat(8), "nothing released");
    assert_eq!(
        above[16..],
        [(0, 0, 0), (0, 12, 0)],
        "the decode bucket replays"
    );
    let newest = shapes(&mut instance, 9, 17..=17);
    assert_eq!(newest[..2], captures(96), "only the eight newest programs");
    assert_eq!(
        newest[2..],
        [(0, 0, 0), (0, 12, 0)],
        "the decode bucket replays"
    );
    assert_eq!(
        run(&mut instance, &prompt_chunk_scoring(10, 0, 1, false, false)),
        (0, 12, 0),
        "a program under the decode graphs replays"
    );
    assert_eq!(
        run(
            &mut instance,
            &prompt_chunk_scoring(10, 1, 16, false, false)
        ),
        (0, 0, 0),
        "a released program runs eagerly"
    );
    instance.destroy().unwrap();
    assert_eq!(host.live(), Vec::<String>::new());
}

/// A release of the newest programs that fails partway (here after 20
/// of their 96 pieces) leaves the failed piece's program forgotten while
/// its graphs stay in the ledger, so the capture releases every graph
/// instead and then captures; nothing is left behind. When that release
/// fails too, the step fails without poisoning the instance and the
/// release stays owed: the next capture first releases every graph, so
/// the ledger never keeps graphs no program names past a capture.
#[test]
fn a_failed_release_of_the_newest_programs_releases_every_graph() {
    let live_graphs = |host: &FakeHost| {
        host.live()
            .iter()
            .filter(|resource| resource.starts_with("graph "))
            .count()
    };
    for refused in [1, 2] {
        let host = FakeHost::new((0, 0));
        let mut instance = create_one_slot(&host);
        let (mut request, mut previous) = (7, 0);
        let mut chunk = |instance: &mut BundleInstanceV1, rows: u64| {
            let start = host.device.journal.borrow().len();
            let executed = instance.execute(
                &prompt_chunk_scoring(request, previous, rows, false, false),
                &mut NoRows::default(),
            );
            if executed.is_ok() {
                instance.commit().unwrap();
                previous += rows;
            } else {
                // The cancelled prefill left its request's state invalid:
                // the host retires it and the prompt starts over.
                let _ = instance.retire_request(request);
                (request, previous) = (request + 1, 0);
            }
            (
                executed.is_ok(),
                count(&host, start, "captured "),
                count(&host, start, "release graph "),
            )
        };
        for rows in 1..=8 {
            assert_eq!(chunk(&mut instance, rows), (true, 0, 0));
            assert_eq!(chunk(&mut instance, rows), (true, 12, 0));
        }
        assert_eq!(chunk(&mut instance, 9), (true, 0, 0));
        assert_eq!(live_graphs(&host), 96);
        host.fail_graph_releases.set((20, refused));
        if refused == 2 {
            // The full release's first piece is refused as well.
            assert_eq!(chunk(&mut instance, 9), (false, 0, 20));
            assert!(!instance.is_poisoned());
            assert_eq!(live_graphs(&host), 76, "{:?}", host.live());
            assert!(
                host.logs
                    .borrow()
                    .iter()
                    .any(|line| { line.contains("releasing the newest prefill programs failed") })
            );
            // The release forgot every key, so the shape runs eagerly once
            // more, leaving the ledger as it is.
            assert_eq!(chunk(&mut instance, 9), (true, 0, 0));
            assert_eq!(live_graphs(&host), 76);
        }
        let released = if refused == 1 { 96 } else { 76 };
        assert_eq!(chunk(&mut instance, 9), (true, 12, released));
        assert_eq!(live_graphs(&host), 12, "{:?}", host.live());
        assert_eq!(chunk(&mut instance, 9), (true, 0, 0), "it replays");
        assert_eq!(chunk(&mut instance, 1), (true, 0, 0), "forgotten, eager");
        instance.destroy().unwrap();
        assert_eq!(host.live(), Vec::<String>::new());
    }
}

/// Programs left under other graphs count toward `HELD`: 24 programs, each
/// run of eight followed by a decode bucket's graphs, leave no program
/// above every other graph, so the 25th capture releases every graph of
/// the instance (the programs' 288 pieces and the decode buckets'), then
/// captures; the decode buckets capture again at their next step.
#[test]
fn twenty_four_programs_under_decode_graphs_release_every_graph() {
    let host = FakeHost::new((0, 0));
    let roles = [crate::projections::COMPONENT_ROLE.as_bytes().to_vec()];
    let record =
        crate::config::tests::record_naming(DECODE_BATCH, &options(2048), host.format, &roles);
    let mut instance = api()
        .create(HostServicesTableV1::new(Rc::clone(&host)), &record)
        .unwrap_or_else(|error| panic!("fake-host create: {error:?} {:?}", host.logs.borrow()));
    let handle = instance.handle();
    // Requests decoding in slots 5 to 8, after 100 tokens each.
    let mut tokens = [100; 4];
    for (index, &committed) in tokens.iter().enumerate() {
        crate::instance::seed(
            handle,
            20 + index as u64,
            5 + index as u32,
            committed as u32,
        );
    }
    let run = |instance: &mut BundleInstanceV1, request: &[u8]| {
        let start = host.device.journal.borrow().len();
        instance
            .execute(request, &mut NoRows::default())
            .unwrap_or_else(|error| panic!("{error:?} {:?}", host.logs.borrow()));
        instance.commit().unwrap();
        assert!(matched(&host));
        (
            count(&host, start, "captured "),
            count(&host, start, "release graph "),
        )
    };
    let mut decode = |instance: &mut BundleInstanceV1, live: usize| {
        let items = (0..live)
            .map(|index| (20 + index as u64, 5 + index as u64, tokens[index], &[1][..]))
            .collect::<Vec<_>>();
        let counts = run(instance, &step(&items));
        for committed in &mut tokens[..live] {
            *committed += 1;
        }
        counts
    };
    let mut previous = 0;
    let mut shapes = |instance: &mut BundleInstanceV1, rows: std::ops::RangeInclusive<u64>| {
        let mut runs = Vec::new();
        for rows in rows {
            for _ in 0..2 {
                runs.push(run(
                    instance,
                    &prompt_chunk_scoring(7, previous, rows, false, false),
                ));
                previous += rows;
            }
        }
        runs
    };
    let mut bucket_pieces = 0;
    for (shape, live) in [(1..=8, 1), (9..=16, 2), (17..=24, 4)] {
        let runs = shapes(&mut instance, shape);
        assert_eq!(runs, [(0, 0), (12, 0)].repeat(8), "nothing released");
        let (captured, released) = decode(&mut instance, live);
        assert!(captured > 0, "the decode bucket of {live} captures");
        assert_eq!(released, 0);
        bucket_pieces += captured;
    }
    assert_eq!(
        shapes(&mut instance, 25..=25),
        [(0, 0), (12, 24 * 12 + bucket_pieces)],
        "every graph"
    );
    assert!(decode(&mut instance, 1).0 > 0, "the bucket captures again");
    instance.destroy().unwrap();
    assert_eq!(host.live(), Vec::<String>::new());
}

/// A request's last prompt chunk (one item, at most the 64-row granule,
/// scoring its last row) captures at its first run and from its second
/// replays, at any committed length. The capture holds exactly the
/// launches the chunk issues eagerly, and first releases the programs
/// above every other graph (here one, 12 pieces), so none is left under
/// it. Pinned programs stay captured: 30 lengths (more than `HELD`)
/// capture with nothing released, other shapes' newest run releases only
/// itself at its ninth shape (96 pieces), and every pinned length still
/// replays. A pinned capture that fails partway releases every graph, as
/// any failed capture does; its length captures at its next run, and so
/// does every released one. Destroy releases the rest.
#[test]
fn a_requests_last_chunk_captures_at_its_first_run() {
    let run = |host: &FakeHost, instance: &mut BundleInstanceV1, request: &[u8]| {
        let start = host.device.journal.borrow().len();
        instance
            .execute(request, &mut NoRows::default())
            .unwrap_or_else(|error| panic!("{error:?} {:?}", host.logs.borrow()));
        instance.commit().unwrap();
        assert!(matched(host));
        (
            (
                count(host, start, "captured "),
                count(host, start, "replay "),
                count(host, start, "release graph "),
            ),
            launches_by_stream(host, start),
        )
    };
    // Two 64-row chunks (the second captures), then the 5-row last chunk.
    let chunks = |host: &FakeHost, instance: &mut BundleInstanceV1| {
        let first = run(host, instance, &prompt_chunk(7, 0, 64, false)).0;
        let second = run(host, instance, &prompt_chunk(7, 64, 64, false)).0;
        let last = run(host, instance, &prompt_chunk(7, 128, 5, true));
        instance.retire_request(7).unwrap();
        ([first, second, last.0], last.1)
    };
    let eager_host = FakeHost::new((0, 0));
    let mut eager = create_one_slot(&eager_host);
    crate::instance::keep_last_chunks_eager(eager.handle());
    let (counts, launched) = chunks(&eager_host, &mut eager);
    assert_eq!(counts, [(0, 0, 0), (12, 12, 0), (0, 0, 0)], "kept eager");
    eager.destroy().unwrap();

    let host = FakeHost::new((0, 0));
    let mut instance = create_one_slot(&host);
    let (counts, captured) = chunks(&host, &mut instance);
    assert_eq!(counts, [(0, 0, 0), (12, 12, 0), (12, 12, 12)]);
    assert_eq!(captured, launched, "the capture holds the eager launches");
    let mut request = 8;
    let mut last = |instance: &mut BundleInstanceV1, rows: u64| {
        let counts = run(&host, instance, &prompt_chunk(request, 0, rows, true));
        instance.retire_request(request).unwrap();
        request += 1;
        counts
    };
    let (counts, launched) = last(&mut instance, 5);
    assert_eq!(counts, (0, 12, 0), "another request's last chunk replays");
    assert!(launched.is_empty());
    let lengths = 1..=30;
    for rows in lengths.clone().filter(|&rows| rows != 5) {
        assert_eq!(last(&mut instance, rows).0, (12, 12, 0), "{rows} rows");
    }
    // Other shapes: chunks that score no row, each captured at its second
    // run.
    let mut previous = 0;
    for rows in 1..=9 {
        for repeat in [false, true] {
            let counts = run(
                &host,
                &mut instance,
                &prompt_chunk_scoring(100, previous, rows, false, false),
            )
            .0;
            previous += rows;
            let released = if rows == 9 { 96 } else { 0 };
            let expected = if repeat {
                (12, 12, released)
            } else {
                (0, 0, 0)
            };
            assert_eq!(counts, expected, "{rows} rows, repeat {repeat}");
        }
    }
    instance.retire_request(100).unwrap();
    for rows in lengths.clone() {
        assert_eq!(last(&mut instance, rows).0, (0, 12, 0), "{rows} rows");
    }
    // A failed capture (rank 1's first piece) releases every graph.
    let live_graphs = || {
        host.live()
            .iter()
            .filter(|resource| resource.starts_with("graph "))
            .count()
    };
    assert!(live_graphs() > 0);
    host.device.fail_graph_end.set(Some(3));
    assert!(
        instance
            .execute(&prompt_chunk(99, 0, 40, true), &mut NoRows::default())
            .is_err()
    );
    assert!(!instance.is_poisoned());
    let _ = instance.retire_request(99);
    assert_eq!(live_graphs(), 0, "{:?}", host.live());
    assert_eq!(last(&mut instance, 40).0, (12, 12, 0));
    assert_eq!(last(&mut instance, 40).0, (0, 12, 0));
    assert_eq!(last(&mut instance, 5).0, (12, 12, 0), "captured again");
    instance.destroy().unwrap();
    assert_eq!(
        count(&host, 0, "release graph "),
        count(&host, 0, "captured "),
        "every captured piece is released by destroy at the latest"
    );
    assert_eq!(host.live(), Vec::<String>::new());
}

/// Create captures the programs of a request's last prompt chunk at every
/// length up to the 64-row granule (12 pieces each), before any other
/// graph, and each holds exactly the launches that chunk issues eagerly
/// (here at 1, 5 and 64 rows, after two 64-row chunks). A request's last
/// chunk then replays without lowering or capturing, whatever its
/// committed rows; other shapes capture above the pinned programs and
/// release among themselves; destroy releases everything.
#[test]
fn create_captures_every_last_chunk_length() {
    use crate::instance::lowered_programs;
    let pieces = 12;
    let host = FakeHost::new((0, 0));
    let mut instance = crate::instance::last_chunks_in_tests(|| create_one_slot(&host));
    assert!(host.logs.borrow().iter().any(|line| {
        line.contains("captured the programs of a request's last prompt chunk at 64 lengths")
    }));
    // Each program's launches by stream: those before its pieces' ends.
    let mut programs = vec![BTreeMap::<u32, Vec<String>>::new(); 64];
    let mut ends = 0;
    let mut capturing = false;
    for line in host.device.journal.borrow().iter() {
        if line.starts_with("capture ") {
            capturing = true;
        } else if line.starts_with("captured ") {
            ends += 1;
        } else if capturing && line.starts_with("launch ") {
            let stream = line.split(' ').nth(1).unwrap().parse().unwrap();
            programs[ends / pieces]
                .entry(stream)
                .or_default()
                .push(line.clone());
        }
    }
    assert_eq!(ends, 64 * pieces);
    assert_eq!(count(&host, 0, "release graph "), 0);
    let run = |host: &FakeHost, instance: &mut BundleInstanceV1, request: &[u8]| {
        let start = host.device.journal.borrow().len();
        let lowered = lowered_programs();
        instance
            .execute(request, &mut NoRows::default())
            .unwrap_or_else(|error| panic!("{error:?} {:?}", host.logs.borrow()));
        instance.commit().unwrap();
        assert!(matched(host));
        (
            (
                lowered_programs() - lowered,
                count(host, start, "captured "),
                count(host, start, "replay "),
                count(host, start, "release graph "),
            ),
            launches_by_stream(host, start),
        )
    };
    let eager_host = FakeHost::new((0, 0));
    let mut eager = create_one_slot(&eager_host);
    crate::instance::keep_last_chunks_eager(eager.handle());
    for (request, rows) in [(7, 1), (8, 5), (9, 64)] {
        for previous in [0, 64] {
            for (host, instance) in [(&host, &mut instance), (&eager_host, &mut eager)] {
                run(host, instance, &prompt_chunk(request, previous, 64, false));
            }
        }
        let (counts, replayed) = run(
            &host,
            &mut instance,
            &prompt_chunk(request, 128, rows, true),
        );
        assert_eq!(counts, (0, 0, pieces, 0), "{rows} rows replay");
        assert!(replayed.is_empty());
        let (counts, launched) = run(
            &eager_host,
            &mut eager,
            &prompt_chunk(request, 128, rows, true),
        );
        assert_eq!(counts, (4, 0, 0, 0), "{rows} rows kept eager");
        assert_eq!(programs[rows as usize - 1], launched, "{rows} rows");
        for instance in [&mut instance, &mut eager] {
            instance.retire_request(request).unwrap();
        }
    }
    eager.destroy().unwrap();
    // The 64-row chunks captured at their second run, above the pinned
    // programs; another request's last chunk replays.
    assert_eq!(
        run(&host, &mut instance, &prompt_chunk(10, 0, 33, true)).0,
        (0, 0, pieces, 0)
    );
    instance.destroy().unwrap();
    assert_eq!(
        count(&host, 0, "release graph "),
        count(&host, 0, "captured ")
    );
    assert_eq!(host.live(), Vec::<String>::new());
}

/// On a drafting (MTP) instance, create captures the last chunks'
/// programs as well, and each holds the target prefill's eager launches:
/// a last chunk's step on an instance that keeps it eager issues, per
/// stream, the captured program's launches and then the drafter's extend
/// pass, which the capturing instance's step issues alone (here at 1, 5
/// and 64 rows after 128 committed rows).
#[test]
fn a_drafting_instance_captures_the_same_last_chunk_programs() {
    let pieces = 12;
    let format = Shape::TP4.kv_formats()[1];
    assert!(model_glm5next_contract::contract::has_mtp_layer(&format));
    let create = |host: &Rc<Host>| {
        host.items.set(1);
        let roles = [crate::projections::COMPONENT_ROLE.as_bytes().to_vec()];
        let record = crate::config::tests::record_naming(1, &options(2048), format, &roles);
        api()
            .create(HostServicesTableV1::new(Rc::clone(host)), &record)
            .unwrap_or_else(|error| panic!("fake-host create: {error:?} {:?}", host.logs.borrow()))
    };
    let host = FakeHost::with_format((3, 4), format);
    let mut instance = crate::instance::last_chunks_in_tests(|| create(&host));
    let mut programs = vec![BTreeMap::<u32, Vec<String>>::new(); 64];
    let mut ends = 0;
    let mut capturing = false;
    for line in host.device.journal.borrow().iter() {
        if line.starts_with("capture ") {
            capturing = true;
        } else if line.starts_with("captured ") {
            ends += 1;
        } else if capturing && line.starts_with("launch ") {
            let stream = line.split(' ').nth(1).unwrap().parse().unwrap();
            programs[ends / pieces]
                .entry(stream)
                .or_default()
                .push(line.clone());
        }
    }
    assert_eq!(ends, 64 * pieces);
    let eager_host = FakeHost::with_format((3, 4), format);
    let mut eager = create(&eager_host);
    crate::instance::keep_last_chunks_eager(eager.handle());
    let run = |host: &FakeHost, instance: &mut BundleInstanceV1, request: &[u8]| {
        let start = host.device.journal.borrow().len();
        instance
            .execute(request, &mut DraftSink::default())
            .unwrap_or_else(|error| panic!("{error:?} {:?}", host.logs.borrow()));
        instance.commit().unwrap();
        assert!(matched(host));
        (
            count(host, start, "replay "),
            launches_by_stream(host, start),
        )
    };
    for (request, rows) in [(7, 1), (8, 5), (9, 64)] {
        for previous in [0, 64] {
            run(
                &host,
                &mut instance,
                &prompt_chunk(request, previous, 64, false),
            );
            run(
                &eager_host,
                &mut eager,
                &prompt_chunk(request, previous, 64, false),
            );
        }
        let (replays, drafter) = run(
            &host,
            &mut instance,
            &prompt_chunk(request, 128, rows, true),
        );
        assert!(replays >= pieces, "{rows} rows replay");
        let (_, launched) = run(
            &eager_host,
            &mut eager,
            &prompt_chunk(request, 128, rows, true),
        );
        let mut expected = programs[rows as usize - 1].clone();
        for (stream, launches) in drafter {
            expected.entry(stream).or_default().extend(launches);
        }
        assert_eq!(expected, launched, "{rows} rows");
        instance.retire_request(request).unwrap();
        eager.retire_request(request).unwrap();
    }
    eager.destroy().unwrap();
    instance.destroy().unwrap();
    assert_eq!(host.live(), Vec::<String>::new());
}

/// After a release of every graph fails, every lookup is gone while the
/// ledger keeps the graphs, so no graph may capture until that release is
/// paid: a drafted step (the drafter's extend pass and draft steps, the
/// target's verify phase) whose release fails again fails before it
/// captures anything, without poisoning the instance, and the next step
/// that needs a graph releases every graph first, then captures.
#[test]
fn an_owed_release_is_paid_before_any_graph_captures() {
    let format = Shape::TP4.kv_formats()[1];
    let host = FakeHost::with_format((3, 4), format);
    host.items.set(1);
    let roles = [crate::projections::COMPONENT_ROLE.as_bytes().to_vec()];
    let record = crate::config::tests::record_naming(1, &options(2048), format, &roles);
    let mut instance = api()
        .create(HostServicesTableV1::new(Rc::clone(&host)), &record)
        .unwrap_or_else(|error| panic!("fake-host create: {error:?} {:?}", host.logs.borrow()));
    let handle = instance.handle();
    crate::instance::keep_last_chunks_eager(handle);
    let live_graphs = || {
        host.live()
            .iter()
            .filter(|resource| resource.starts_with("graph "))
            .count()
    };
    let journal = || host.device.journal.borrow().len();
    let prompt = |instance: &mut BundleInstanceV1, request: u64, rows: u64| {
        instance
            .execute(
                &prompt_chunk(request, 0, rows, true),
                &mut DraftSink::default(),
            )
            .unwrap_or_else(|error| panic!("{error:?} {:?}", host.logs.borrow()));
        instance.commit().unwrap();
    };
    let draft = |instance: &mut BundleInstanceV1, request: u64, previous: u64, bonus: u32| {
        let executed = instance.execute(
            &drafted(request, previous, 4, bonus),
            &mut DraftSink::default(),
        );
        if executed.is_ok() {
            instance.commit_accepted(&[kept(request, 2)]).unwrap();
        }
        executed
    };
    prompt(&mut instance, 7, 16);
    draft(&mut instance, 7, 16, 21).unwrap();
    let live = live_graphs();
    assert!(live > 0);
    host.fail_graph_releases.set((0, 1));
    assert!(crate::instance::try_release_graphs(handle).is_err());
    assert_eq!(live_graphs(), live);
    // The drafted step needs every graph again; the owed release fails.
    host.fail_graph_releases.set((0, 1));
    let start = journal();
    assert!(draft(&mut instance, 7, 18, 22).is_err());
    assert!(!instance.is_poisoned());
    assert_eq!(count(&host, start, "captured "), 0, "nothing captured");
    assert_eq!(live_graphs(), live);
    let _ = instance.retire_request(7);
    // A new prompt length runs eagerly; its drafted step pays the release.
    let start = journal();
    prompt(&mut instance, 8, 17);
    assert_eq!(count(&host, start, "captured "), 0);
    assert_eq!(count(&host, start, "release graph "), 0);
    draft(&mut instance, 8, 17, 23).unwrap();
    let lines = host.device.journal.borrow()[start..].to_vec();
    let last_release = lines
        .iter()
        .rposition(|line| line.starts_with("release graph "));
    let first_capture = lines.iter().position(|line| line.starts_with("captured "));
    assert!(
        last_release < first_capture,
        "{last_release:?} {first_capture:?}"
    );
    assert_eq!(count(&host, start, "release graph "), live);
    assert_eq!(live_graphs(), count(&host, start, "captured "));
    instance.destroy().unwrap();
    assert_eq!(host.live(), Vec::<String>::new());
}

/// A wait that fails before graphs are released (to make room for a ninth
/// shape, or after a capture failed partway) poisons the instance:
/// earlier replays may still be queued, so it keeps its graphs and
/// resources and refuses every later call (execute, retirement, destroy)
/// without touching a stream or giving back a resource.
#[test]
fn a_failed_wait_before_releasing_every_graph_poisons_the_instance() {
    for full in [true, false] {
        let host = FakeHost::new((0, 0));
        let mut instance = create_one_slot(&host);
        let mut previous = 0;
        let mut chunk = |instance: &mut BundleInstanceV1, rows: u64| {
            let executed = instance.execute(
                &prompt_chunk_scoring(7, previous, rows, false, false),
                &mut NoRows::default(),
            );
            if executed.is_ok() {
                instance.commit().unwrap();
                previous += rows;
            }
            executed
        };
        let shapes = if full { 8 } else { 0 };
        for rows in 1..=shapes {
            chunk(&mut instance, rows).unwrap();
            chunk(&mut instance, rows).unwrap();
        }
        // A captured program replays; the next shape runs once eagerly.
        if full {
            chunk(&mut instance, 1).unwrap();
        }
        chunk(&mut instance, 9).unwrap();
        if !full {
            // Rank 0's three pieces end, rank 1's first does not.
            host.device.fail_graph_end.set(Some(3));
        }
        host.fail_synchronize.set(Some(0));
        let live = host.live();
        let released = host.released.get();
        assert!(chunk(&mut instance, 9).is_err());
        assert!(
            host.logs
                .borrow()
                .iter()
                .any(|line| line.contains("the instance is poisoned"))
        );
        // Nothing was released, the failed capture's graphs included.
        assert_eq!(host.released.get(), released);
        assert!(host.live().len() >= live.len());
        let journal = host.device.journal.borrow().len();
        assert!(
            instance
                .execute(
                    &prompt_chunk_scoring(7, previous, 1, false, false),
                    &mut NoRows::default()
                )
                .is_err()
        );
        assert!(instance.retire_request(7).is_err());
        assert!(instance.destroy().is_err());
        assert!(instance.is_poisoned(), "full {full}");
        assert_eq!(host.device.journal.borrow().len(), journal);
        assert_eq!(host.released.get(), released);
    }
}

/// Every graph the instance holds is released newest first, as the host
/// requires (the fake host refuses any other order): decode graphs
/// captured before a prefill program and drafter graphs interleaved with
/// the target's, by `release_graphs` and by destroy.
#[test]
fn graphs_are_released_newest_first_across_every_cache() {
    let live_graphs = |host: &FakeHost| {
        host.live()
            .iter()
            .filter(|r| r.starts_with("graph "))
            .count()
    };
    let host = FakeHost::new((0, 0));
    let mut instance = create_one_slot(&host);
    let handle = instance.handle();
    // A decode bucket's graphs, then a prefill program over them.
    instance
        .execute(&prompt_chunk(7, 0, 16, true), &mut NoRows::default())
        .unwrap();
    instance.commit().unwrap();
    instance
        .execute(&step(&[(7, 0, 16, &[1])]), &mut NoRows::default())
        .unwrap();
    instance.commit().unwrap();
    instance.retire_request(7).unwrap();
    for request in [8, 9] {
        instance
            .execute(&prompt_chunk(request, 0, 16, true), &mut NoRows::default())
            .unwrap_or_else(|error| panic!("{error:?} {:?}", host.logs.borrow()));
        instance.commit().unwrap();
        instance.retire_request(request).unwrap();
    }
    assert_eq!(
        live_graphs(&host),
        24,
        "the decode bucket's and the prefill's"
    );
    crate::instance::release_graphs(handle);
    assert_eq!(live_graphs(&host), 0);
    // Again, released by destroy.
    for request in [10, 11] {
        instance
            .execute(&prompt_chunk(request, 0, 16, true), &mut NoRows::default())
            .unwrap();
        instance.commit().unwrap();
        instance.retire_request(request).unwrap();
    }
    instance.destroy().unwrap();
    assert_eq!(host.live(), Vec::<String>::new());

    // The drafter's extend pass and draft steps capture between the
    // target's verify graphs.
    let format = Shape::TP4.kv_formats()[1];
    let host = FakeHost::with_format((3, 4), format);
    host.items.set(1);
    let roles = [crate::projections::COMPONENT_ROLE.as_bytes().to_vec()];
    let record = crate::config::tests::record_naming(1, &options(2048), format, &roles);
    let mut instance = api()
        .create(HostServicesTableV1::new(Rc::clone(&host)), &record)
        .unwrap();
    instance
        .execute(&prompt_chunk(7, 0, 16, true), &mut DraftSink::default())
        .unwrap();
    instance.commit().unwrap();
    for (index, previous) in [16, 18].into_iter().enumerate() {
        instance
            .execute(
                &drafted(7, previous, 4, 21 + index as u32),
                &mut DraftSink::default(),
            )
            .unwrap_or_else(|error| panic!("{error:?} {:?}", host.logs.borrow()));
        instance.commit_accepted(&[kept(7, 2)]).unwrap();
    }
    assert!(live_graphs(&host) > 12);
    instance.destroy().unwrap();
    assert_eq!(host.live(), Vec::<String>::new());
}

/// A prefill capture that fails partway (rank 1's first piece) publishes
/// nothing and fails the step; the graphs rank 0 captured stay owned and
/// are released with every other graph, newest first, so nothing stays
/// live behind the instance's back and destroy releases everything.
#[test]
fn a_failed_prefill_capture_releases_what_it_captured() {
    let live_graphs = |host: &FakeHost| {
        host.live()
            .iter()
            .filter(|r| r.starts_with("graph "))
            .count()
    };
    let host = FakeHost::new((0, 0));
    let mut instance = create_one_slot(&host);
    // The shape captures at its second run (a pinned last chunk at its
    // first: `a_requests_last_chunk_captures_at_its_first_run`).
    crate::instance::keep_last_chunks_eager(instance.handle());
    for (request, fails) in [(7, false), (8, true), (9, false), (10, false)] {
        // Rank 0's three pieces end, rank 1's first does not.
        host.device.fail_graph_end.set(fails.then_some(3));
        let executed =
            instance.execute(&prompt_chunk(request, 0, 16, true), &mut NoRows::default());
        if fails {
            assert!(executed.is_err());
            assert_eq!(live_graphs(&host), 0, "{:?}", host.live());
            assert!(!instance.is_poisoned());
            // The cancelled step left its request nothing to retire.
            let _ = instance.retire_request(request);
        } else {
            executed.unwrap_or_else(|error| panic!("{error:?} {:?}", host.logs.borrow()));
            instance.commit().unwrap();
            instance.retire_request(request).unwrap();
        }
    }
    // The shape captured again at its next repeat (requests 9 and 10).
    assert_eq!(live_graphs(&host), 12);
    instance.destroy().unwrap();
    assert_eq!(host.live(), Vec::<String>::new());
}

/// With the bring-up switch (`SYNC_STAGES`), every rank synchronizes
/// after each segment of an eager run and after each graph replay, logging
/// the segment's stages before the wait, so a hang's last line names its
/// stage and rank; the ranks still reach each collective in turn.
#[test]
fn the_stage_sync_switch_names_each_wait_before_it() {
    let host = FakeHost::new((0, 0));
    let mut instance = create_one_slot(&host);
    crate::instance::sync_stages(instance.handle());
    let start = host.device.journal.borrow().len();
    let logged = host.logs.borrow().len();
    instance
        .execute(&prompt_chunk(7, 0, 1, true), &mut NoRows::default())
        .unwrap_or_else(|error| panic!("{error:?} {:?}", host.logs.borrow()));
    let (_, collectives) = assert_ranks_advance_together(&host, start)[0];
    let logs = host.logs.borrow()[logged..].to_vec();
    let waits = logs
        .iter()
        .filter(|line| line.contains("waits for eager segment"))
        .count();
    assert!(waits >= 4 * collectives, "{waits} waits");
    assert!(
        logs[0].contains("stage sync: rank 0 waits for eager segment 0 of ")
            && logs[0].contains("Some(Embedding) to Some(EmbeddingReduce)"),
        "{}",
        logs[0]
    );
    assert!(logs[1].starts_with("GLM 5.3 Flash: stage sync: rank 0 finished eager segment 0"));
    // Each wait is a synchronization of its rank's stream.
    assert!(count(&host, start, "synchronize ") >= waits);
    instance.commit().unwrap();
    let logged = host.logs.borrow().len();
    instance
        .execute(&step(&[(7, 0, 1, &[1])]), &mut NoRows::default())
        .unwrap();
    let logs = host.logs.borrow()[logged..].to_vec();
    for rank in 0..4 {
        let wait = format!("rank {rank} waits for None graph Some((1, 0))");
        assert!(logs.iter().any(|line| line.contains(&wait)), "{logs:?}");
    }
    instance.commit().unwrap();
    instance.destroy().unwrap();
}

/// A failed diagnostic wait leaves queued work whose completion nothing
/// established, in an eager segment or a graph replay alike: the instance
/// is poisoned and refuses every later call that would queue device work
/// or release resources.
#[test]
fn a_failed_stage_sync_poisons_the_instance() {
    for graph in [false, true] {
        let host = FakeHost::new((0, 0));
        let mut instance = create_one_slot(&host);
        crate::instance::sync_stages(instance.handle());
        let mut request = prompt_chunk(7, 0, 1, true);
        if graph {
            instance.execute(&request, &mut NoRows::default()).unwrap();
            instance.commit().unwrap();
            request = step(&[(7, 0, 1, &[1])]);
        }
        host.fail_synchronize.set(Some(0));
        assert_eq!(
            instance
                .execute(&request, &mut NoRows::default())
                .unwrap_err(),
            BundleAbiError::Status(BundleStatusV1::HostError)
        );
        let waited = match graph {
            true => "stage sync: rank 0 waits for None graph",
            false => "stage sync: rank 0 waits for eager segment 0 of ",
        };
        let logs = host.logs.borrow().clone();
        assert!(logs.iter().any(|line| line.contains(waited)), "{logs:?}");
        assert!(
            logs.iter()
                .any(|line| line.contains("the instance is poisoned"))
        );
        let released = host.released.get();
        let journal = host.device.journal.borrow().len();
        assert!(
            instance
                .execute(&prompt_chunk(9, 0, 1, true), &mut NoRows::default())
                .is_err()
        );
        assert!(instance.retire_request(7).is_err());
        assert!(instance.destroy().is_err());
        assert!(instance.is_poisoned(), "graph {graph}");
        assert_eq!(host.released.get(), released);
        assert_eq!(host.device.journal.borrow().len(), journal);
    }
}

/// A sink of the step's drafts and logits views.
#[derive(Default)]
struct DraftSink {
    drafts: Vec<(u64, Vec<u32>)>,
    views: usize,
}

impl LogitsRowSinkV1 for DraftSink {
    fn limits(&self) -> ResultSinkLimitsV1 {
        ResultSinkLimitsV1::new(256, 1 << 20, 256 << 20)
    }

    fn emit_row(&mut self, _row: OwnedLogitsRowV1) -> HostServiceResultV1<()> {
        Err(BundleStatusV1::Unsupported)
    }

    fn emit_draft_tokens(&mut self, drafts: OwnedDraftTokensV1) -> HostServiceResultV1<()> {
        self.drafts.push((drafts.request_id, drafts.tokens));
        Ok(())
    }

    fn emit_view(&mut self, _view: OwnedLogitsViewV1) -> HostServiceResultV1<()> {
        self.views += 1;
        Ok(())
    }
}

/// A verify item of `rows` rows the MTP drafter drafts: the host sends
/// its bonus token alone.
fn drafted(request: u64, previous: u64, rows: u32, bonus: u32) -> Vec<u8> {
    drafted_in(request, previous, rows, bonus, 0)
}

/// [`drafted`] in state slot `slot`.
fn drafted_in(request: u64, previous: u64, rows: u32, bonus: u32, slot: u64) -> Vec<u8> {
    use runtime_bundle_sdk::{BatchAppendV1, BatchItemV1, BatchRequestV1, BatchWireVersionV1};
    let end = previous + u64::from(rows);
    BatchRequestV1::new(
        BatchWireVersionV1::V3,
        vec![BatchItemV1::verify(request, rows, true, vec![bonus])],
        vec![BatchAppendV1::new(
            previous,
            u64::from(rows),
            end,
            (0..end.div_ceil(64)).collect(),
            None,
        )],
    )
    .with_state_slots(vec![slot])
    .encode()
    .unwrap()
}

/// With an MTP KV format and speculation, the instance runs the MTP
/// drafter: a prefill runs the drafter's extend pass over the rows its
/// chunk completes after the target, and commit moves the frontier to the
/// last row but one. A verify item the drafter drafts first runs the
/// extend pass over the rows past the frontier, its last reading the bonus
/// token, then one draft step per further draft, each captured as graphs
/// at the first step of its bucket and replayed after; the step emits the
/// drafts before the logits, and commit keeps the frontier at the length
/// before the step and retains the kept rows' hidden rows.
#[test]
fn the_mtp_drafter_extends_drafts_and_commits_around_the_target() {
    use model_glm5next_contract::contract::has_mtp_layer;
    let format = Shape::TP4.kv_formats()[1];
    assert!(has_mtp_layer(&format));
    let host = FakeHost::with_format((3, 32), format);
    let mut instance = api()
        .create(
            HostServicesTableV1::new(Rc::clone(&host)),
            &crate::config::tests::record_with(DECODE_BATCH, &options(2048), format),
        )
        .unwrap_or_else(|error| panic!("fake-host create: {error:?} {:?}", host.logs.borrow()));
    let handle = instance.handle();
    crate::instance::run_lowered_stages_alone(handle);
    // The prompt's ten rows, nine of which the drafter runs after them.
    instance
        .execute(&prompt_chunk(7, 0, 10, true), &mut DraftSink::default())
        .unwrap_or_else(|error| panic!("{error:?} {:?}", host.logs.borrow()));
    instance.commit().unwrap();
    assert_eq!(crate::instance::drafter_frontier(handle, 0), Some(9));
    // Three drafts of a verify item of four rows.
    let start = host.device.journal.borrow().len();
    let mut sink = DraftSink::default();
    instance
        .execute(&drafted(7, 10, 4, 21), &mut sink)
        .unwrap_or_else(|error| panic!("{error:?} {:?}", host.logs.borrow()));
    assert_eq!(sink.drafts, [(7, vec![0, 0, 0])]);
    assert!(sink.views > 0);
    let captured = count(&host, start, "captured ");
    assert!(captured > 0, "the drafter's passes capture their graphs");
    // Every rank captures the extend pass and the first draft step (the
    // second replays it): each gathers its row inputs and writes its
    // drafts, the draft step compacts the shared selection.
    let launched = |symbol: &str| {
        issued(&host, start)
            .iter()
            .filter(|line| line.starts_with("launch ") && line.contains(symbol))
            .count()
    };
    assert_eq!(launched("chop_row_glm5next_draft_gather_v1"), 4 * 2);
    assert_eq!(launched("pooled_indexer_glm5next_compact"), 4);
    assert_eq!(launched("chop_row_argmax_final_v1"), 4 * 2);
    instance.commit_accepted(&[kept(7, 2)]).unwrap();
    assert_eq!(crate::instance::drafter_frontier(handle, 0), Some(10));
    // Two rows past the frontier now: the extend pass takes a verify
    // bucket, captured once; the next step of the same shape replays every
    // graph.
    let mut sink = DraftSink::default();
    instance.execute(&drafted(7, 12, 4, 22), &mut sink).unwrap();
    assert_eq!(sink.drafts.len(), 1);
    instance.commit_accepted(&[kept(7, 2)]).unwrap();
    assert_eq!(crate::instance::drafter_frontier(handle, 0), Some(12));
    // A snapshot of request 7 carries its retained rows and frontier: a
    // request restored from it in slot 1 drafts from there.
    let pages = instance.describe_prefix_state().unwrap();
    let ids = (128..128 + pages).collect::<Vec<_>>();
    instance.capture_prefix_state(7, 14, &ids).unwrap();
    let tokens = (0..14).collect::<Vec<u32>>();
    instance
        .restore_prefix_state_in_slot(8, &tokens, &ids, 1)
        .unwrap();
    assert_eq!(crate::instance::drafter_frontier(handle, 1), Some(12));
    let mut sink = DraftSink::default();
    instance
        .execute(&drafted_in(8, 14, 4, 30, 1), &mut sink)
        .unwrap();
    assert_eq!(sink.drafts.len(), 1);
    instance.commit_accepted(&[kept(8, 1)]).unwrap();
    assert_eq!(crate::instance::drafter_frontier(handle, 1), Some(14));
    let start = host.device.journal.borrow().len();
    instance
        .execute(&drafted(7, 14, 4, 23), &mut DraftSink::default())
        .unwrap();
    assert_eq!(count(&host, start, "captured "), 0);
    assert!(count(&host, start, "replay ") > 0);
    // Cancel keeps the frontier.
    instance.cancel().unwrap();
    assert_eq!(crate::instance::drafter_frontier(handle, 0), Some(12));
    instance.destroy().unwrap();
    assert_eq!(host.acquired.get(), host.released.get());
}

/// The MTP drafter's indexer tail, the indexer tail's last layer, follows
/// the drafter's frontier: every extend pass writes it in place (one row
/// with the decode writer, more with the prefill writer), never into the
/// target's verify checkpoints, and the target's commit copies its kept
/// checkpoint into every other layer only, whatever rows it keeps.
#[test]
fn target_commits_keep_the_drafters_indexer_tail() {
    let format = Shape::TP4.kv_formats()[1];
    let host = FakeHost::with_format((3, 32), format);
    let mut instance = create_over(&host, format);
    let handle = instance.handle();
    crate::instance::run_lowered_stages_alone(handle);
    host.backing.set(true);
    let drafter = Shape::TP4.sparse_layers();
    let [layer, tail] = crate::instance::indexer_tail(handle, 0, drafter, 0, None);
    let [_, target] = crate::instance::indexer_tail(handle, 0, 0, 0, None);
    instance
        .execute(&prompt_chunk(7, 0, 10, true), &mut DraftSink::default())
        .unwrap();
    instance.commit().unwrap();
    assert_eq!(crate::instance::drafter_frontier(handle, 0), Some(9));
    // (committed tokens, extend rows, kept rows, the extend's tail writer
    // when it launches eagerly or captures).
    let steps = [
        (10, 1, 2, Some("key_write_decode")),
        (12, 2, 3, Some("key_write_prefill")),
        (15, 3, 1, Some("key_write_prefill")),
        (16, 1, 4, None),
    ];
    for (index, (previous, rows, kept_rows, writer)) in steps.into_iter().enumerate() {
        let frontier = crate::instance::drafter_frontier(handle, 0).unwrap();
        assert_eq!(frontier + rows, previous);
        let start = host.device.journal.borrow().len();
        instance
            .execute(&drafted(7, previous, 4, 21), &mut DraftSink::default())
            .unwrap_or_else(|error| panic!("{error:?} {:?}", host.logs.borrow()));
        let writes = issued(&host, start)
            .into_iter()
            .filter(|line| line.starts_with("launch ") && line.contains(&format!("{layer:?}")))
            .collect::<Vec<_>>();
        assert!(writes.iter().all(|line| !line.contains("key_write_verify")));
        if let Some(writer) = writer {
            assert!(
                writes.iter().any(|line| line.contains(writer)),
                "{writes:?}"
            );
        }
        // Model the kernels' writes: the drafter's extend left its tail in
        // place, the target's verify rows their checkpoints of the other
        // layers; the drafter's layer's checkpoints stay unwritten.
        let value = 0x40 + index as u8;
        host.write(tail, &vec![value; tail.length as usize]);
        let [_, checkpoint] = crate::instance::indexer_tail(handle, 0, 0, 0, Some(kept_rows - 1));
        host.write(checkpoint, &vec![!value; checkpoint.length as usize]);
        instance.commit_accepted(&[kept(7, kept_rows)]).unwrap();
        assert_eq!(crate::instance::drafter_frontier(handle, 0), Some(previous));
        assert!(host.read(tail).iter().all(|&byte| byte == value));
        assert!(host.read(target).iter().all(|&byte| byte == !value));
    }
    instance.destroy().unwrap();
    assert_eq!(host.acquired.get(), host.released.get());
}

/// Journal lines from `from` on that start with `prefix`.
pub(crate) fn count(host: &FakeHost, from: usize, prefix: &str) -> usize {
    host.device.journal.borrow()[from..]
        .iter()
        .filter(|line| line.starts_with(prefix))
        .count()
}

/// A decode bucket's graphs are captured at its first step and replayed by
/// every later step of the bucket, whose live count and inputs the uploads
/// carry; a new bucket captures its own. Destroy releases every graph.
#[test]
fn decode_graphs_capture_once_per_bucket_and_replay() {
    let host = FakeHost::new((3, 32));
    let mut instance = create(&host);
    let handle = instance.handle();
    crate::instance::run_lowered_stages_alone(handle);
    crate::instance::seed(handle, 7, 0, 100);
    crate::instance::seed(handle, 8, 5, 64);
    let start = host.device.journal.borrow().len();
    instance
        .execute(
            &step(&[(7, 0, 100, &[1]), (8, 5, 64, &[2])]),
            &mut NoRows::default(),
        )
        .unwrap();
    instance.commit().unwrap();
    let captured = count(&host, start, "captured ");
    let launched = count(&host, start, "launch ");
    // Bucket 2's all-reduces each take one phased launch, where bucket 1's
    // take the low-latency entry and the round row.
    let phased = host.device.journal.borrow()[start..]
        .iter()
        .filter(|line| {
            line.starts_with("launch ")
                && model_glm5next_contract::sm120::engines::PHASED_SYMBOLS
                    .iter()
                    .any(|symbol| line.contains(symbol))
        })
        .count();
    assert!(phased > 0);
    assert!(captured >= 4, "every rank captures its pieces");
    assert_eq!(count(&host, start, "replay "), captured);
    // The same bucket again: uploads and replays, no capture or launch.
    let start = host.device.journal.borrow().len();
    instance
        .execute(
            &step(&[(7, 0, 101, &[3]), (8, 5, 65, &[4])]),
            &mut NoRows::default(),
        )
        .unwrap();
    instance.commit().unwrap();
    assert_eq!(count(&host, start, "captured "), 0);
    assert_eq!(count(&host, start, "launch "), 0);
    assert_eq!(count(&host, start, "replay "), captured);
    assert!(count(&host, start, "upload ") > 0);
    // One item: bucket 1 captures once.
    let start = host.device.journal.borrow().len();
    instance
        .execute(&step(&[(7, 0, 102, &[5])]), &mut NoRows::default())
        .unwrap();
    instance.commit().unwrap();
    assert_eq!(count(&host, start, "captured "), captured);
    assert!(
        count(&host, start, "launch ") > 0 && count(&host, start, "launch ") <= launched + phased
    );
    // A verify step of 4 rows for one item: tokens bucket 4, item bucket 1.
    let start = host.device.journal.borrow().len();
    instance
        .execute(&step(&[(7, 0, 103, &[6, 7, 8, 9])]), &mut NoRows::default())
        .unwrap();
    instance.commit_accepted(&[kept(7, 4)]).unwrap();
    assert!(count(&host, start, "captured ") > 0);
    instance.destroy().unwrap();
    assert_eq!(host.acquired.get(), host.released.get());
}

/// The drafter reads `h_p` as HF's MTP does, after `model.norm`: a step
/// stages the target's normalized rows (the final norm's output) for
/// commit to retain, not the collapse before it, and the drafter's extend
/// pass over a prefill's completed rows gathers those same rows.
#[test]
fn the_drafter_stages_and_gathers_the_targets_normalized_rows() {
    use crate::buffers::Buffer;
    let format = Shape::TP4.kv_formats()[1];
    let host = FakeHost::with_format((3, 32), format);
    let mut instance = create_over(&host, format);
    let handle = instance.handle();
    crate::instance::run_lowered_stages_alone(handle);
    host.backing.set(true);
    let normed = crate::instance::scratch_span(handle, 0, Buffer::Normed);
    let collapsed = crate::instance::scratch_span(handle, 0, Buffer::Collapsed);
    let row = u64::from(Shape::TP4.profile.hidden) * 2;
    let rows = |span: DeviceBufferSpanV1, value: u8| {
        host.write(
            DeviceBufferSpanV1::new(span.buffer, span.offset, 16 * row),
            &(0..16 * row)
                .map(|byte| value ^ (byte / row) as u8)
                .collect::<Vec<_>>(),
        );
    };
    rows(normed, 0x40);
    rows(collapsed, 0x80);
    let start = host.device.journal.borrow().len();
    instance
        .execute(&prompt_chunk(7, 0, 10, true), &mut DraftSink::default())
        .unwrap();
    let [staged, _] = crate::instance::drafter_staged(handle, 0);
    let at = |span: DeviceBufferSpanV1, index: u64| {
        host.read(DeviceBufferSpanV1::new(
            span.buffer,
            span.offset + index * row,
            row,
        ))
    };
    assert!(at(staged, 0) == at(normed, 9));
    assert!(at(staged, 0) != at(collapsed, 9));
    let gather = issued(&host, start)
        .into_iter()
        .find(|line| line.starts_with("launch ") && line.contains("draft_gather"))
        .expect("the extend pass gathers its rows");
    assert!(gather.contains(&format!("{normed:?}")), "{gather}");
    assert!(!gather.contains(&format!("{collapsed:?}")), "{gather}");
    instance.commit().unwrap();
    instance.destroy().unwrap();
}

/// One step of a new request's prompt chunk (`prefill`: request, slot,
/// tokens) and verify items (request, slot, previous tokens, inputs).
fn mixed(prefill: (u64, u64, u64), verify: &[(u64, u64, u64, &[u32])]) -> Vec<u8> {
    use runtime_bundle_sdk::{BatchAppendV1, BatchItemV1, BatchRequestV1, BatchWireVersionV1};
    let (request, slot, count) = prefill;
    let mut items = vec![BatchItemV1::new(
        request,
        false,
        count as u32,
        true,
        (0..count as u32).collect(),
    )];
    let mut appends = vec![BatchAppendV1::new(
        0,
        count,
        count,
        (0..count.div_ceil(64))
            .map(|page| 200 + slot * 4 + page)
            .collect(),
        None,
    )];
    let mut slots = vec![slot];
    for &(request, slot, previous, inputs) in verify {
        let end = previous + inputs.len() as u64;
        items.push(BatchItemV1::verify(
            request,
            inputs.len() as u32,
            false,
            inputs.to_vec(),
        ));
        appends.push(BatchAppendV1::new(
            previous,
            inputs.len() as u64,
            end,
            (0..end.div_ceil(64)).map(|page| slot * 4 + page).collect(),
            None,
        ));
        slots.push(slot);
    }
    BatchRequestV1::new(BatchWireVersionV1::V3, items, appends)
        .with_state_slots(slots)
        .encode()
        .unwrap()
}

/// A step with a new prompt chunk and verify items: the eager prefill
/// shares the KDA row offsets (`CuRows`, `CuSeqlens`) with the verify
/// phase, so the verify phase binds its own again before it runs, whether
/// it captures its graphs or replays them, at any item count and row
/// length.
#[test]
fn a_verify_phase_after_a_prefill_reads_its_own_row_offsets() {
    use crate::buffers::Buffer;
    let host = FakeHost::new((3, 32));
    let mut instance = create(&host);
    let handle = instance.handle();
    crate::instance::run_lowered_stages_alone(handle);
    // The prefills launch eagerly, so the verify phase's first replay is
    // its own.
    crate::instance::keep_last_chunks_eager(handle);
    crate::instance::seed(handle, 7, 0, 100);
    crate::instance::seed(handle, 8, 5, 64);
    let tables = [Buffer::CuRows, Buffer::CuSeqlens]
        .map(|buffer| crate::instance::scratch_span(handle, 0, buffer));
    // The offsets each table held when the verify phase started: its last
    // upload before the phase's first launch or replay.
    let bound = |start: usize| {
        let journal = host.device.journal.borrow();
        let verify = (start..journal.len())
            .find(|&line| {
                let line = &journal[line];
                line.starts_with("replay ")
                    || (line.starts_with("launch ") && line.contains("conv_verify"))
            })
            .expect("a verify phase");
        tables.map(|span| {
            let log = host.upload_log.borrow();
            let (_, _, bytes) = log
                .iter()
                .rev()
                .find(|&&(line, at, _)| {
                    line <= verify && (at.buffer, at.offset) == (span.buffer, span.offset)
                })
                .expect("an upload");
            bytes
                .chunks_exact(4)
                .map(|word| i32::from_le_bytes(word.try_into().unwrap()))
                .collect::<Vec<_>>()
        })
    };
    for (prefill, verify, offsets) in [
        // Captures the verify bucket.
        ((9, 3, 64), vec![(7, 0, 100, &[1, 2, 3, 4][..])], vec![0, 4]),
        // Replays it beside a prompt of another length.
        (
            (10, 4, 32),
            vec![(7, 0, 102, &[5, 6, 7, 8][..])],
            vec![0, 4],
        ),
        // Two items of other lengths.
        (
            (11, 6, 100),
            vec![(7, 0, 104, &[9, 10, 11][..]), (8, 5, 64, &[12, 13][..])],
            vec![0, 3, 5],
        ),
    ] {
        let start = host.device.journal.borrow().len();
        instance
            .execute(&mixed(prefill, &verify), &mut NoRows::default())
            .unwrap_or_else(|error| panic!("{error:?} {:?}", host.logs.borrow()));
        for table in bound(start) {
            assert_eq!(table[..offsets.len()], offsets[..]);
        }
        let kept = verify
            .iter()
            .map(|&(request, ..)| kept(request, 2))
            .collect::<Vec<_>>();
        instance.commit_accepted(&kept).unwrap();
    }
    instance.destroy().unwrap();
}

/// A verify item of six rows (five drafts) on an instance serving one
/// slot: its eight-token bucket would need two item slots, so the phase
/// runs eagerly at its live rows instead of being refused; a four-row item
/// still replays its graph.
#[test]
fn a_one_slot_instance_verifies_six_rows_eagerly() {
    let host = FakeHost::new((5, 32));
    host.items.set(1);
    let mut instance = api()
        .create(
            HostServicesTableV1::new(Rc::clone(&host)),
            &record(1, &options(2048)),
        )
        .unwrap_or_else(|error| panic!("fake-host create: {error:?} {:?}", host.logs.borrow()));
    let handle = instance.handle();
    crate::instance::run_lowered_stages_alone(handle);
    crate::instance::seed(handle, 7, 0, 100);
    let start = host.device.journal.borrow().len();
    instance
        .execute(
            &step(&[(7, 0, 100, &[1, 2, 3, 4, 5, 6])]),
            &mut NoRows::default(),
        )
        .unwrap_or_else(|error| panic!("{error:?} {:?}", host.logs.borrow()));
    instance.commit_accepted(&[kept(7, 6)]).unwrap();
    assert_eq!(count(&host, start, "captured "), 0);
    assert!(count(&host, start, "launch ") > 0);
    let start = host.device.journal.borrow().len();
    instance
        .execute(&step(&[(7, 0, 106, &[1, 2, 3, 4])]), &mut NoRows::default())
        .unwrap();
    instance.commit_accepted(&[kept(7, 4)]).unwrap();
    assert!(count(&host, start, "captured ") > 0);
    instance.destroy().unwrap();
}

/// Creates an instance over the KV `format`.
fn create_over(host: &Rc<Host>, format: KvFormat) -> BundleInstanceV1 {
    api()
        .create(
            HostServicesTableV1::new(Rc::clone(host)),
            &crate::config::tests::record_with(DECODE_BATCH, &options(2048), format),
        )
        .unwrap_or_else(|error| panic!("fake-host create: {error:?} {:?}", host.logs.borrow()))
}

/// The span of snapshot `piece` in a rank's request-state `buffers`.
fn piece_span(
    buffers: &[DeviceBufferSpanV1],
    piece: &bundle_runtime::snapshot::Descriptor,
) -> DeviceBufferSpanV1 {
    let view = buffers[piece.buffer];
    assert!(!piece.spare && piece.offset + piece.length <= view.length);
    DeviceBufferSpanV1::new(view.buffer, view.offset + piece.offset, piece.length)
}

/// Writes distinct bytes (from `seed`) into every declared piece of
/// `slot`'s committed state on every rank, and returns them per rank in
/// snapshot order.
fn fill_slot(host: &FakeHost, handle: BundleHandleV1, slot: u32, seed: u64) -> Vec<Vec<u8>> {
    let (spans, table) = crate::instance::snapshot_layout(handle, slot);
    spans
        .iter()
        .enumerate()
        .map(|(rank, buffers)| {
            let mut state = 0x9e37_79b9_7f4a_7c15_u64 ^ seed << 40 ^ (rank as u64) << 56;
            let mut all = Vec::new();
            for piece in &table.descriptors {
                let bytes = (0..piece.length.div_ceil(8))
                    .flat_map(|_| {
                        state = state
                            .wrapping_mul(6_364_136_223_846_793_005)
                            .wrapping_add(1_442_695_040_888_963_407);
                        state.to_le_bytes()
                    })
                    .take(piece.length as usize)
                    .collect::<Vec<_>>();
                host.write(piece_span(buffers, piece), &bytes);
                all.extend(bytes);
            }
            all
        })
        .collect()
}

/// Every rank's bytes of `slot`'s declared pieces, in snapshot order.
fn slot_bytes(host: &FakeHost, handle: BundleHandleV1, slot: u32) -> Vec<Vec<u8>> {
    let (spans, table) = crate::instance::snapshot_layout(handle, slot);
    spans
        .iter()
        .map(|buffers| {
            table
                .descriptors
                .iter()
                .flat_map(|piece| host.read(piece_span(buffers, piece)))
                .collect()
        })
        .collect()
}

/// The snapshot bytes rank `rank` holds in `pages`, header included.
fn snapshot_bytes(
    host: &FakeHost,
    table: &bundle_runtime::snapshot::Table,
    rank: u32,
    pages: &[u64],
) -> Vec<u8> {
    let mut bytes = pages
        .iter()
        .flat_map(|&page| {
            host.read(DeviceBufferSpanV1::new(
                BundleDeviceBufferHandleV1::from_parts(1, KV_SLOT + rank),
                page * table.page_bytes,
                table.page_bytes,
            ))
        })
        .collect::<Vec<_>>();
    bytes.truncate((table.header_at() + table.header_bytes) as usize);
    bytes
}

/// Over both KV formats, and the FP8 layout with the MTP drafter (whose
/// retained rows are a piece and whose frontier the header carries), a
/// snapshot's pages hold every declared piece of the committed slot in
/// snapshot order (noncontiguous pages, each rank its own shards, rank 0
/// the header), and a restore fills another slot's pieces from the pages,
/// not from the source slot, leaving other slots alone. A device copy that fails after rank 0's copies and one of rank
/// 1's fails the capture or restore with the attempted ranks drained: the
/// source keeps its owner, the destination stays free, and a retry
/// succeeds. A drain that fails is fatal: the instance refuses every later
/// call and keeps its resources.
#[test]
fn prefix_snapshots_move_every_byte_from_its_origin() {
    let tokens = (0..128).collect::<Vec<u32>>();
    for format in [0, 2, 1].map(|index| Shape::TP4.kv_formats()[index]) {
        let host = FakeHost::with_format((3, 32), format);
        let mut instance = create_over(&host, format);
        let handle = instance.handle();
        crate::instance::seed(handle, 7, 2, 128);
        // The drafter has run every row but the last, which waits for the
        // next token.
        let drafting = model_glm5next_contract::contract::has_mtp_layer(&format);
        if drafting {
            crate::instance::drafter_restore(handle, 2, &tokens, 127);
        }
        host.backing.set(true);
        let source = fill_slot(&host, handle, 2, 1);
        let (_, table) = crate::instance::snapshot_layout(handle, 2);
        // Pages out of order from page 100 on, a stride prime to their
        // count, so pieces cross between unrelated pages.
        let held = table.pages();
        assert_eq!(held, instance.describe_prefix_state().unwrap());
        let gcd = |mut a: u64, mut b: u64| {
            while b != 0 {
                (a, b) = (b, a % b);
            }
            a
        };
        let stride = (11..).find(|&stride| gcd(stride, held) == 1).unwrap();
        let pages = (0..held)
            .map(|index| 100 + (index * stride) % held)
            .collect::<Vec<_>>();
        let start = host.device.journal.borrow().len();
        instance.capture_prefix_state(7, 128, &pages).unwrap();
        let copies = count(&host, start, "copy ");
        assert_eq!(copies % 4, 0);
        let check_pages = |expected: &[Vec<u8>]| {
            for rank in 0..4 {
                let bytes = snapshot_bytes(&host, &table, rank, &pages);
                for piece in &table.descriptors {
                    let at = piece.at as usize..(piece.at + piece.length) as usize;
                    assert!(
                        bytes[at.clone()] == expected[rank as usize][at],
                        "{format:?} rank {rank} piece {piece:?}"
                    );
                }
                let header = &bytes[table.header_at() as usize..];
                assert_eq!(
                    header.iter().any(|&byte| byte != 0),
                    rank == 0,
                    "rank {rank}"
                );
            }
        };
        check_pages(&source);
        // The source slot changes; a restore into slot 6 takes the pages'
        // bytes, and slot 3 stays unwritten.
        let changed = fill_slot(&host, handle, 2, 2);
        instance
            .restore_prefix_state_in_slot(9, &tokens, &pages, 6)
            .unwrap();
        assert!(slot_bytes(&host, handle, 6) == source);
        assert_eq!(
            crate::instance::drafter_frontier(handle, 6),
            drafting.then_some(127)
        );
        assert!(slot_bytes(&host, handle, 2) == changed);
        let empty = slot_bytes(&host, handle, 3);
        assert!(empty.iter().flatten().all(|&byte| byte == 0));
        // A capture whose copies fail in rank 1 drains both ranks; request
        // 7 keeps slot 2, and a retry captures its new bytes.
        host.fail_copy.set(Some(copies as u32 / 4 + 1));
        let start = host.device.journal.borrow().len();
        assert_eq!(
            instance.capture_prefix_state(7, 128, &pages).unwrap_err(),
            BundleAbiError::Status(BundleStatusV1::HostError)
        );
        assert!(count(&host, start, "synchronize ") >= 2);
        assert_eq!(
            crate::instance::ledger_state(handle, 2),
            (Some((7, 128)), false)
        );
        instance.capture_prefix_state(7, 128, &pages).unwrap();
        check_pages(&changed);
        // A restore whose copies fail leaves slot 7 free; a retry fills it.
        host.fail_copy.set(Some(copies as u32 / 4 + 1));
        assert_eq!(
            instance
                .restore_prefix_state_in_slot(11, &tokens, &pages, 7)
                .unwrap_err(),
            BundleAbiError::Status(BundleStatusV1::HostError)
        );
        assert_eq!(crate::instance::ledger_state(handle, 7), (None, false));
        instance
            .restore_prefix_state_in_slot(11, &tokens, &pages, 7)
            .unwrap();
        assert!(slot_bytes(&host, handle, 7) == changed);
        assert_eq!(
            crate::instance::ledger_state(handle, 7),
            (Some((11, 128)), false)
        );
        // A drain that cannot be confirmed is fatal.
        host.fail_synchronize.set(Some(0));
        assert_eq!(
            instance.capture_prefix_state(7, 128, &pages).unwrap_err(),
            BundleAbiError::Status(BundleStatusV1::InternalError)
        );
        let released = host.released.get();
        let journal = host.device.journal.borrow().len();
        assert!(instance.capture_prefix_state(7, 128, &pages).is_err());
        assert!(instance.destroy().is_err());
        assert!(instance.is_poisoned());
        assert_eq!(host.released.get(), released);
        assert_eq!(host.device.journal.borrow().len(), journal);
    }
}

/// A committed request's KDA state and indexer tail go to snapshot pages
/// with the shared header on rank 0, and come back into another request's
/// state slot as that request's committed state; a snapshot read as other
/// tokens, a capture of tokens the request does not hold, an occupied slot
/// and a request that already holds a slot are refused.
#[test]
fn prefix_snapshots_capture_and_restore_a_slot() {
    let host = FakeHost::new((3, 32));
    let mut instance = create(&host);
    let handle = instance.handle();
    crate::instance::seed(handle, 7, 2, 128);
    let pages = instance.describe_prefix_state().unwrap();
    assert_eq!(pages, 89);
    // The snapshot's own pages, after the 4 pages per slot the steps use.
    let ids = (128..128 + pages).collect::<Vec<_>>();
    let start = host.device.journal.borrow().len();
    instance.capture_prefix_state(7, 128, &ids).unwrap();
    // Every rank copies its slot's pieces into the pages, each piece in
    // one copy over the consecutive pages; rank 0 also uploads the header,
    // and every rank synchronizes.
    let copies = count(&host, start, "copy ");
    let (_, table) = crate::instance::snapshot_layout(handle, 2);
    assert_eq!(copies, 4 * table.descriptors.len());
    assert_eq!(count(&host, start, "upload "), 1);
    assert_eq!(count(&host, start, "synchronize "), 4);
    // The restore reads the header back (the fake device keeps uploads):
    // request 9 now holds slot 6 at 128 tokens, and the copies ran back.
    let tokens = (0..128).collect::<Vec<u32>>();
    let start = host.device.journal.borrow().len();
    instance
        .restore_prefix_state_in_slot(9, &tokens, &ids, 6)
        .unwrap();
    assert_eq!(count(&host, start, "copy "), copies);
    assert_eq!(count(&host, start, "download "), 1);
    assert_eq!(
        crate::instance::ledger_state(handle, 6),
        (Some((9, 128)), false)
    );
    // A snapshot of other tokens does not match its header: the slot
    // stays free.
    let other = (0..64).collect::<Vec<u32>>();
    assert!(
        instance
            .restore_prefix_state_in_slot(10, &other, &ids, 7)
            .is_err()
    );
    assert_eq!(crate::instance::ledger_state(handle, 7), (None, false));
    // A capture of tokens the request does not hold is refused.
    assert!(instance.capture_prefix_state(7, 64, &ids).is_err());
    // An unseen request into an occupied slot (request 7's) is refused
    // before any copy, and the slot keeps its holder.
    let start = host.device.journal.borrow().len();
    assert!(
        instance
            .restore_prefix_state_in_slot(11, &tokens, &ids, 2)
            .is_err()
    );
    assert_eq!(count(&host, start, "copy "), 0);
    assert_eq!(
        crate::instance::ledger_state(handle, 2),
        (Some((7, 128)), false)
    );
    assert_eq!(crate::instance::ledger_state(handle, 7), (None, false));
    // A request that already holds a slot is refused before any copy,
    // even into a free slot.
    let start = host.device.journal.borrow().len();
    assert!(
        instance
            .restore_prefix_state_in_slot(9, &tokens, &ids, 7)
            .is_err()
    );
    assert_eq!(count(&host, start, "copy "), 0);
    assert_eq!(crate::instance::ledger_state(handle, 7), (None, false));
    instance.destroy().unwrap();
    assert_eq!(host.acquired.get(), host.released.get());
}

/// The autoregressive bring-up's shape: one served item, no speculation,
/// the precision component. A scored prompt chunk and decode steps lower
/// every stage and run through the ledger, and the prefix state is
/// described for the host's generation worker.
#[test]
fn a_one_item_autoregressive_instance_runs_whole_steps() {
    let host = FakeHost::new((0, 0));
    host.items.set(1);
    let roles = [crate::projections::COMPONENT_ROLE.as_bytes().to_vec()];
    let record = crate::config::tests::record_naming(1, &options(2048), host.format, &roles);
    let mut instance = api()
        .create(HostServicesTableV1::new(Rc::clone(&host)), &record)
        .unwrap_or_else(|error| panic!("{error:?} {:?}", host.logs.borrow()));
    instance.describe_prefix_state().unwrap();
    let mut sink = NoRows::default();
    instance
        .execute(&prompt_chunk_scoring(7, 0, 20, true, false), &mut sink)
        .unwrap_or_else(|error| panic!("{error:?} {:?}", host.logs.borrow()));
    instance.commit().unwrap();
    for (previous, token) in [(20, 154842), (21, 11), (22, 220)] {
        let decode = step(&[(7, 0, previous, &[token])]);
        instance
            .execute(&decode, &mut NoRows::default())
            .unwrap_or_else(|error| panic!("{error:?} {:?}", host.logs.borrow()));
        instance.commit().unwrap();
    }
    assert_eq!(
        crate::instance::ledger_state(instance.handle(), 0),
        (Some((7, 23)), false)
    );
    instance.destroy().unwrap();
    assert_eq!(host.acquired.get(), host.released.get());
}

/// Without speculation (`--speculative-tokens 0`) the instance serves no
/// verify rows: it describes its prefix state, in every KV format and at
/// one served item as at the full batch, and a snapshot captures and
/// restores a slot as with speculation.
#[test]
fn an_instance_without_speculation_describes_and_moves_its_prefix_state() {
    for format in Shape::TP4.kv_formats() {
        for items in [1, DECODE_BATCH] {
            let host = FakeHost::with_format((0, 0), format);
            host.items.set(items);
            let mut instance = api()
                .create(
                    HostServicesTableV1::new(Rc::clone(&host)),
                    &crate::config::tests::record_with(items, &options(2048), format),
                )
                .unwrap_or_else(|error| panic!("{error:?} {:?}", host.logs.borrow()));
            let pages = instance
                .describe_prefix_state()
                .unwrap_or_else(|error| panic!("{format:?} {items}: {error:?}"));
            let handle = instance.handle();
            crate::instance::seed(handle, 7, 0, 128);
            let ids = (128..128 + pages).collect::<Vec<_>>();
            instance.capture_prefix_state(7, 128, &ids).unwrap();
            if items > 1 {
                let tokens = (0..128).collect::<Vec<u32>>();
                instance
                    .restore_prefix_state_in_slot(9, &tokens, &ids, 1)
                    .unwrap();
                assert_eq!(
                    crate::instance::ledger_state(handle, 1),
                    (Some((9, 128)), false)
                );
            }
            instance.destroy().unwrap();
            assert_eq!(host.acquired.get(), host.released.get());
        }
    }
}

#[path = "snapshot_dump_tests.rs"]
mod snapshot_dump;
