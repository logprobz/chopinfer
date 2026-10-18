//! CPU fake host for MiMo bundle acquisition and release, over the shared
//! fake device (`bundle_runtime::fake_host`).
#[path = "conformance_tests.rs"]
mod conformance_tests;

use crate::contract::*;
use crate::sm120::device::residency;
use crate::topology::Tp;
use bundle_runtime::fake_host::{
    Backend, FakeDevice, FakeIndex, FakePackage, Handle, NoObserver, RecordingSink, SinkLimits,
    SnapshotOp,
};
use runtime_bundle_abi::*;
use std::cell::{Cell, RefCell};
use std::collections::{BTreeMap, BTreeSet};
use std::rc::Rc;
mod precision;

/// A host service a prefix-copy test makes fail.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PrefixFault {
    Readback,
    Upload,
    Launch,
    Synchronize,
}

/// (stream, symbol, grid, arguments) of one launch.
type LaunchRecord = (u32, String, [u32; 3], Vec<KernelArgumentV1>);

/// The fake host of a MiMo package.
type FakeHost = Backend<Package>;

/// A host for a package of a topology.
trait At {
    fn at(tp: Tp) -> Self;
}

impl At for FakeHost {
    fn at(tp: Tp) -> Self {
        Backend::new(Package::at(tp))
    }
}

/// The package the fake host serves, and what tests set and read.
#[derive(Default)]
struct Package {
    /// Handles, the journal, streams, events, graphs, collectives,
    /// launches, residency, the copy and launch fault switches and the
    /// snapshot record.
    device: FakeDevice,
    pool_pages: Cell<Option<u64>>,
    snapshot_memory: RefCell<BTreeMap<(u64, u64), u8>>,
    device_copies: RefCell<Vec<(DeviceBufferSpanV1, DeviceBufferSpanV1)>>,
    offload: RefCell<Option<offload::Probe>>,
    /// Serve the FP8 global-KV layout instead of BF16.
    fp8_kv: Cell<bool>,
    /// The layout selected by a create test; unset uses the historical layout.
    kv_layout_id: Cell<Option<&'static [u8]>>,
    /// Sequences the created instance serves; zero is 32.
    served: Cell<u32>,
    allocations: RefCell<Vec<u64>>,
    allocation_bytes: RefCell<BTreeMap<u32, u64>>,
    host_copies: RefCell<Vec<(u64, Vec<u8>)>>,
    host_copy_offsets: RefCell<Vec<u64>>,
    symbols: Cell<u32>,
    launches: Cell<u32>,
    launch_symbols: RefCell<Vec<String>>,
    stream_syncs: Cell<u32>,
    graph_begins: Cell<u32>,
    graph_ends: Cell<u32>,
    graph_replays: Cell<u32>,
    /// (stream slot, graph slot) per graph launch, in issue order.
    graph_launches: RefCell<Vec<(u32, u32)>>,
    graph_releases: Cell<u32>,
    /// Live graphs, oldest first. Like the host, this releases the newest.
    live_graphs: RefCell<Vec<u32>>,
    replayed_nodes: Cell<usize>,
    zeroes: Cell<u32>,
    peer_grants: Cell<u32>,
    kv_acquires: Cell<u32>,
    table_uploads_before_kv: Cell<usize>,
    kv_resolves: Cell<u32>,
    copies: RefCell<Vec<(u64, u64, u64)>>,
    /// (stream slot, symbol) per launch, and (stream slot, event slot, wait)
    /// per event operation, in issue order.
    stream_launches: RefCell<Vec<(u32, String, u32)>>,
    event_ops: RefCell<Vec<(u32, u32, bool)>>,
    /// Enabled (draft tokens, verify rows); speculation is off when unset.
    speculation: Cell<Option<(u32, u32)>>,
    mtp_images: RefCell<Vec<u32>>,
    readback_tokens: RefCell<Vec<u32>>,
    encoder_images: RefCell<Vec<u32>>,
    encoder_readbacks: Cell<bool>,
    /// Runs before each encoder readback while `encoder_readbacks` is set.
    encoder_readback_hook: RefCell<Option<Box<dyn FnMut()>>>,
    /// Every readback returns zeros (a tapped instance reads every buffer).
    tap_readbacks: Cell<bool>,
    /// The packaged drafter is DFlash rather than MTP.
    dflash: Cell<bool>,
    bad_text_index: Cell<u32>,
    bad_precision: Cell<u32>,
    bad_drafter_identity: Cell<u32>,
    /// The last rank's encoder image identity names another file.
    bad_encoder_identity: Cell<bool>,
    /// The media the package's encoders serve; unset serves every one.
    served_encoders: Cell<Option<model_mimo26_contract::encoders::Encoders>>,
    /// The encoders the encoder rank indexes and images hold, when not the
    /// served ones.
    indexed_encoders: Cell<Option<model_mimo26_contract::encoders::Encoders>>,
    drafter_uploads: Cell<u32>,
    /// Opens of the text rank image.
    text_images: Cell<u32>,
    /// Output span offsets of the DFlash row copies, in launch order.
    dflash_copy_outputs: RefCell<Vec<u64>>,
    /// History span, capacity and width of each retained-row launch.
    retain_launches: RefCell<Vec<(DeviceBufferSpanV1, u32, u32)>>,
    /// Arguments of each catch-up gather launch, as captured or issued.
    gather_launches: RefCell<Vec<Vec<KernelArgumentV1>>>,
    /// The bytes of the latest upload to each (buffer, offset).
    latest_uploads: RefCell<BTreeMap<(u64, u64), Vec<u8>>>,
    /// (stream slot, arguments) of every prefix snapshot copy, in launch order.
    prefix_copies: RefCell<Vec<(u32, Vec<KernelArgumentV1>)>>,
    /// The service call that fails, after this many more calls of it succeed.
    prefix_fault: Cell<Option<(PrefixFault, u32)>>,
    /// Streams with a prefix copy launched and not yet synchronized: device
    /// work that may still read or write snapshot pages.
    prefix_in_flight: RefCell<BTreeSet<u32>>,
    /// (stream, symbol, grid, arguments) of every launch.
    launch_records: RefCell<Vec<LaunchRecord>>,
    /// The package's rank count; zero is TP4.
    ranks: Cell<u32>,
    geometry: Cell<Option<&'static model_mimo26_contract::geometry::Geometry>>,
}

impl std::ops::Deref for Package {
    type Target = FakeDevice;

    fn deref(&self) -> &FakeDevice {
        &self.device
    }
}

/// Authenticated component slots, less one: text rank indexes from 0, drafter
/// indexes from `DRAFTER_INDEX`, then the image identities of each.
const DRAFTER_INDEX: u32 = 16;
const TEXT_IDENTITY: u32 = 32;
const DRAFTER_IDENTITY: u32 = 48;
const ENCODER_INDEX: u32 = 64;
const ENCODER_IDENTITY: u32 = 80;
const PRECISION: u32 = 96;
const ENCODER_SET: u32 = 97;

impl Package {
    fn kv(&self) -> &'static bundle_runtime::kv::KvFormat {
        &crate::contract::KV_FORMATS[usize::from(self.fp8_kv.get())]
    }
    /// The media the package's encoders serve.
    fn encoders(&self) -> model_mimo26_contract::encoders::Encoders {
        self.served_encoders
            .get()
            .unwrap_or(model_mimo26_contract::encoders::Encoders::ALL)
    }
    /// The encoders whose views the encoder rank images hold.
    fn indexed_encoders(&self) -> model_mimo26_contract::encoders::Encoders {
        self.indexed_encoders.get().unwrap_or(self.encoders())
    }
    /// A package of `tp`'s rank images.
    fn at(tp: Tp) -> Self {
        let host = Self::default();
        host.ranks.set(tp.ranks());
        host.geometry.set(Some(tp.geometry));
        host
    }
    fn tp(&self) -> Tp {
        Tp::of_geometry(
            self.geometry
                .get()
                .unwrap_or(&model_mimo26_contract::geometry::V26_FLASH),
            if self.ranks.get() == 0 {
                4
            } else {
                self.ranks.get()
            },
        )
        .unwrap()
    }
    fn served(&self) -> u32 {
        match self.served.get() {
            0 => 32,
            served => served,
        }
    }
    /// Ring slots of the served count: one per item of its largest bucket.
    fn ring_slots(&self) -> u32 {
        self.served().next_power_of_two()
    }
    /// The ring state of the served count: one slot per item of its
    /// largest bucket.
    fn ring_state_bytes(&self) -> u64 {
        crate::topology::ring_state_bytes(self.tp(), self.served().next_power_of_two())
    }
    fn inject(&self, service: PrefixFault) -> HostServiceResultV1<()> {
        match self.prefix_fault.get() {
            Some((fault, 0)) if fault == service => {
                self.prefix_fault.set(None);
                Err(BundleStatusV1::HostError)
            }
            Some((fault, count)) if fault == service => {
                self.prefix_fault.set(Some((fault, count - 1)));
                Ok(())
            }
            _ => Ok(()),
        }
    }
    fn component_bytes(&self, rank: u32) -> Vec<u8> {
        if rank == PRECISION {
            let bytes = precision::component(
                self.tp(),
                self.speculation.get().map(|_| {
                    if self.dflash.get() {
                        model_mimo26_contract::weight_plan::Drafter::Dflash
                    } else {
                        model_mimo26_contract::weight_plan::Drafter::Mtp
                    }
                }),
            );
            if self.bad_precision.get() == 0 {
                return bytes.to_vec();
            }
            let mut document = kernel_core::precision::component::decode(bytes).unwrap();
            match self.bad_precision.get() {
                1 => document.operations.retain(|s| s.source.id != "head.bf16"),
                2 => document.operations[0].bindings[0].implementation.dependency = "b".repeat(64),
                // Layer 1's QKV codes with layer 2's scales.
                3 => {
                    let selection = document
                        .operations
                        .iter_mut()
                        .find(|s| s.source.id == "swa.qkv.1.bf16")
                        .unwrap();
                    for operation in [&mut selection.source, &mut selection.effective] {
                        let scale = &mut operation.weights[0].local_scale.as_mut().unwrap().tensor;
                        assert!(scale.tensor.contains("layers.1."));
                        scale.tensor = scale.tensor.replace("layers.1.", "layers.2.");
                    }
                }
                // Layer 1's QKV codes read again as its F32 scales, or its
                // scales with another extent.
                4 | 5 => {
                    let fault = self.bad_precision.get();
                    let selection = document
                        .operations
                        .iter_mut()
                        .find(|s| s.source.id == "swa.qkv.1.bf16")
                        .unwrap();
                    for operation in [&mut selection.source, &mut selection.effective] {
                        let weight = &mut operation.weights[0];
                        let scale = &mut weight.local_scale.as_mut().unwrap().tensor;
                        if fault == 4 {
                            scale.tensor = weight.codes.tensor.clone();
                        } else {
                            scale.shape[0] += 1;
                            scale.byte_length += scale.shape[1] * 4;
                        }
                    }
                }
                _ => unreachable!(),
            }
            return kernel_core::precision::component::encode(&document).unwrap();
        }
        if rank < DRAFTER_INDEX {
            let mut index: serde_json::Value =
                serde_json::from_slice(&fake_index(rank, self.tp())).unwrap();
            let view = &mut index["views"][0];
            match self.bad_text_index.get() {
                1 => view["shape"] = serde_json::json!([4096, 152576]),
                2 => view["dtype"] = serde_json::json!("WRONG"),
                3 => view["layout_id"] = serde_json::json!("stale_layout_v0"),
                4 => index["views"][1]["offset"] = serde_json::json!(0),
                5 => {
                    let extra = index["views"][0].clone();
                    index["views"].as_array_mut().unwrap().push(extra);
                }
                6 => index["sha256"] = serde_json::json!("b".repeat(64)),
                7 => view["bytes"] = serde_json::json!(view["bytes"].as_u64().unwrap() - 2),
                8 => view["name"] = serde_json::json!("model.embed_tokens.weight2"),
                _ => {}
            }
            serde_json::to_vec(&index).unwrap()
        } else {
            self.drafter_index(rank - DRAFTER_INDEX).0
        }
    }
    fn image_identity(&self, slot: u32) -> (String, Vec<u8>) {
        if slot >= ENCODER_IDENTITY {
            let rank = slot - ENCODER_IDENTITY;
            let bytes = serde_json::to_vec(&serde_json::json!({"byte_length": self.indexed_encoders().image_bytes_for(self.tp().geometry), "sha256": if self.bad_encoder_identity.get() && rank + 1 == self.tp().ranks() { "c" } else { "a" }.repeat(64)})).unwrap();
            return (
                format!("chop.compiled-image-identity.v1/mimo.encoder-rank-images.v1/{rank}"),
                bytes,
            );
        }
        if slot < DRAFTER_IDENTITY {
            return fake_image_identity(slot - TEXT_IDENTITY, self.tp());
        }
        let rank = slot - DRAFTER_IDENTITY;
        let (bytes, length) = self.drafter_index(rank);
        let index: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        let mut identity = serde_json::json!({"byte_length": length, "sha256": index["sha256"]});
        if rank + 1 == self.tp().ranks() {
            match self.bad_drafter_identity.get() {
                1 => identity["sha256"] = serde_json::json!("b".repeat(64)),
                2 => identity["byte_length"] = serde_json::json!(length + 4096),
                _ => {}
            }
        }
        (
            format!("chop.compiled-image-identity.v1/mimo.drafter-rank-images.v1/{rank}"),
            serde_json::to_vec(&identity).unwrap(),
        )
    }
    fn encoder_index(&self, rank: u32) -> Vec<u8> {
        let mut offset = 0;
        let views = model_mimo26_contract::encoders::views_for(self.tp().geometry, self.indexed_encoders()).into_iter().map(|(name, view)| {
            let entry = serde_json::json!({"name":name,"offset":offset,"bytes":view.bytes,"shape":view.shape,"dtype":view.dtype,"layout_id":view.layout_id});
            offset += view.bytes.next_multiple_of(4096);
            entry
        }).collect::<Vec<_>>();
        serde_json::to_vec(&serde_json::json!({"schema": crate::encoders::views::SCHEMA,"rank":rank,"image":format!("encoder-rank{rank}.img"),"image_bytes":offset,"sha256":"a".repeat(64),"views":views})).unwrap()
    }
    fn drafter_index(&self, rank: u32) -> (Vec<u8>, u64) {
        if self.dflash.get() {
            fake_dflash_index(rank, self.tp())
        } else {
            fake_mtp_index(rank, self.tp())
        }
    }
}
const FAKE_IMAGE_BYTES: u64 = 64 << 30;

/// Bytes of a synthetic text rank image: 64 GiB holds every view at TP4;
/// TP2's rank weights are twice TP4's.
fn fake_image_bytes(tp: Tp) -> u64 {
    if tp.ranks() < 4 {
        2 * FAKE_IMAGE_BYTES
    } else {
        FAKE_IMAGE_BYTES
    }
}
impl FakePackage for Package {
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
        eprintln!("{}", String::from_utf8_lossy(message));
        self.device.log(message);
    }
    /// Like the host, this releases the newest graph first.
    fn admit_release(&self, handle: Handle) -> HostServiceResultV1<()> {
        if let Handle::Graph(slot) = handle {
            let mut live = self.live_graphs.borrow_mut();
            if live.last() != Some(&slot) {
                return Err(BundleStatusV1::InvalidState);
            }
            live.pop();
        }
        Ok(())
    }
    fn release(&self, handle: Handle) {
        match handle {
            // Identity inspection is CPU metadata, outside the device
            // journal.
            Handle::Component(slot) if slot > TEXT_IDENTITY => {}
            // A pool's peer grant is the pool's, outside the journal.
            Handle::PeerGrant { .. } => {}
            // A graph's release journals only the device's line.
            Handle::Graph(_) => self.graph_releases.set(self.graph_releases.get() + 1),
            _ => self.note("release".into()),
        }
    }
    fn on_acquire(&self, handle: Handle) {
        if let Handle::PeerGrant { .. } = handle {
            self.peer_grants.set(self.peer_grants.get() + 1);
        }
    }
    /// Views, KV pools and state share fixed spans the host does not track.
    fn check_span(&self, _span: DeviceBufferSpanV1) {}
    fn on_allocation(
        &self,
        _pool: BundleMemoryPoolHandleV1,
        buffer: BundleDeviceBufferHandleV1,
        byte_length: u64,
    ) {
        self.allocations.borrow_mut().push(byte_length);
        self.allocation_bytes
            .borrow_mut()
            .insert(buffer.slot(), byte_length);
    }
    fn open_compiled_image(&self, role: &[u8]) -> HostServiceResultV1<BundleCompiledImageHandleV1> {
        let id = self.id();
        if role == crate::mtp::views::IMAGE_ROLE {
            self.mtp_images.borrow_mut().push(id);
        } else if role == crate::encoders::views::IMAGE_ROLE {
            self.encoder_images.borrow_mut().push(id);
        } else {
            assert_eq!(role, b"mimo.rank-images.v1");
            self.text_images.set(self.text_images.get() + 1);
        }
        Ok(BundleCompiledImageHandleV1::from_parts(1, id))
    }
    fn upload_rank_arena(
        &self,
        _context: BundleContextHandleV1,
        _pool: BundleMemoryPoolHandleV1,
        _stream: BundleStreamHandleV1,
        image: BundleCompiledImageHandleV1,
        _rank: u32,
    ) -> HostServiceResultV1<BundleRankArenaHandleV1> {
        if self.mtp_images.borrow().contains(&image.slot()) {
            self.drafter_uploads.set(self.drafter_uploads.get() + 1);
        }
        Ok(BundleRankArenaHandleV1::from_parts(1, self.id()))
    }
    fn resolve_compiled_image_view(
        &self,
        image: BundleCompiledImageHandleV1,
        _arena: BundleRankArenaHandleV1,
        rank: u32,
        _view_id: &[u8],
    ) -> HostServiceResultV1<CompiledImageSpanV1> {
        Ok(CompiledImageSpanV1::new(
            rank,
            DeviceBufferSpanV1::new(
                BundleDeviceBufferHandleV1::from_parts(1, 1),
                0,
                if self.mtp_images.borrow().contains(&image.slot()) {
                    self.drafter_index(rank).1
                } else if self.encoder_images.borrow().contains(&image.slot()) {
                    self.indexed_encoders().image_bytes_for(self.tp().geometry)
                } else {
                    fake_image_bytes(self.tp())
                },
            ),
            4096,
        ))
    }
    fn open_authenticated_component(
        &self,
        role: &[u8],
    ) -> HostServiceResultV1<BundleAuthenticatedComponentHandleV1> {
        if role == model_mimo26_contract::prepared::COMPONENT_ROLE.as_bytes() {
            return Ok(BundleAuthenticatedComponentHandleV1::from_parts(
                1,
                PRECISION + 1,
            ));
        }
        if role == model_mimo26_contract::weight_plan::ENCODER_SET_ROLE.as_bytes() {
            // Only a package that names its encoder set holds the component.
            if !names_encoder_set(self.encoders()) {
                return Err(BundleStatusV1::NotFound);
            }
            return Ok(BundleAuthenticatedComponentHandleV1::from_parts(
                1,
                ENCODER_SET + 1,
            ));
        }
        if let Some(rank) =
            role.strip_prefix(b"chop.compiled-image-identity.v1/mimo.rank-images.v1/")
        {
            let rank = std::str::from_utf8(rank).unwrap().parse::<u32>().unwrap();
            return Ok(BundleAuthenticatedComponentHandleV1::from_parts(
                1,
                1 + TEXT_IDENTITY + rank,
            ));
        }
        if let Some(rank) =
            role.strip_prefix(b"chop.compiled-image-identity.v1/mimo.drafter-rank-images.v1/")
        {
            let rank = std::str::from_utf8(rank).unwrap().parse::<u32>().unwrap();
            return Ok(BundleAuthenticatedComponentHandleV1::from_parts(
                1,
                1 + DRAFTER_IDENTITY + rank,
            ));
        }
        if let Some(rank) =
            role.strip_prefix(b"chop.compiled-image-identity.v1/mimo.encoder-rank-images.v1/")
        {
            let rank = std::str::from_utf8(rank).unwrap().parse::<u32>().unwrap();
            return Ok(BundleAuthenticatedComponentHandleV1::from_parts(
                1,
                1 + ENCODER_IDENTITY + rank,
            ));
        }
        if let Some(rank) = std::str::from_utf8(role)
            .ok()
            .and_then(|role| role.strip_prefix("chop.bundle-metadata.mimo.encoder-rank-index.r"))
            .and_then(|rank| rank.strip_suffix(".v1"))
            .and_then(|rank| rank.parse::<u32>().ok())
        {
            return Ok(BundleAuthenticatedComponentHandleV1::from_parts(
                1,
                1 + ENCODER_INDEX + rank,
            ));
        }
        let rank = std::str::from_utf8(role)
            .ok()
            .and_then(|role| {
                role.strip_prefix("chop.bundle-metadata.mimo.rank-index.r")
                    .map(|rank| (rank, 0))
                    .or_else(|| {
                        role.strip_prefix("chop.bundle-metadata.mimo.drafter-rank-index.r")
                            .map(|rank| (rank, DRAFTER_INDEX))
                    })
            })
            .and_then(|(role, base)| role.strip_suffix(".v1").map(|rank| (rank, base)))
            .and_then(|(rank, base)| rank.parse::<u32>().ok().map(|rank| rank + base))
            .ok_or(BundleStatusV1::InvalidArgument)?;
        Ok(BundleAuthenticatedComponentHandleV1::from_parts(
            1,
            rank + 1,
        ))
    }
    fn component(&self, slot: u32) -> (String, Vec<u8>) {
        let rank = slot - 1;
        if rank == PRECISION {
            return (
                model_mimo26_contract::prepared::COMPONENT_ROLE.into(),
                self.component_bytes(rank),
            );
        }
        if rank == ENCODER_SET {
            return (
                model_mimo26_contract::weight_plan::ENCODER_SET_ROLE.into(),
                model_mimo26_contract::weight_plan::encoder_set_component(self.encoders()),
            );
        }
        if (ENCODER_INDEX..ENCODER_IDENTITY).contains(&rank) {
            let rank = rank - ENCODER_INDEX;
            return (
                format!("chop.bundle-metadata.mimo.encoder-rank-index.r{rank}.v1"),
                self.encoder_index(rank),
            );
        }
        if rank >= TEXT_IDENTITY {
            return self.image_identity(rank);
        }
        (fake_role(rank), self.component_bytes(rank))
    }
    fn acquire_serving_kv(&self, layout_id: &[u8]) -> HostServiceResultV1<u64> {
        assert_eq!(
            layout_id,
            self.kv_layout_id.get().unwrap_or(self.kv().layout_id)
        );
        self.table_uploads_before_kv.set(
            self.host_copies
                .borrow()
                .iter()
                .filter(|(_, data)| {
                    data.len() >= 256
                        && data[..256]
                            .chunks_exact(8)
                            .all(|pair| pair == [0, 0, 128, 63, 0, 0, 0, 0])
                })
                .count(),
        );
        self.kv_acquires.set(self.kv_acquires.get() + 1);
        Ok(self.pool_pages.get().unwrap_or(128))
    }
    fn resolve_serving_kv_pool(&self, rank: u32) -> HostServiceResultV1<DeviceBufferSpanV1> {
        assert!(rank < self.tp().ranks());
        assert_eq!(self.kv_acquires.get(), 1);
        self.kv_resolves.set(self.kv_resolves.get() + 1);
        Ok(DeviceBufferSpanV1::new(
            BundleDeviceBufferHandleV1::from_parts(1, rank + 100),
            0,
            self.pool_pages.get().unwrap_or(128)
                * crate::topology::ga_page_bytes(self.tp(), self.kv().fp8()),
        ))
    }
    fn allocate_state(&self, request: &StateRequestV1) -> HostServiceResultV1<BundleStateHandleV1> {
        assert_eq!(request.byte_length, self.ring_state_bytes());
        Ok(BundleStateHandleV1::from_parts(1, self.id()))
    }
    fn resolve_state_view(
        &self,
        _state: BundleStateHandleV1,
        _rank: u32,
    ) -> HostServiceResultV1<DeviceBufferSpanV1> {
        Ok(DeviceBufferSpanV1::new(
            BundleDeviceBufferHandleV1::from_parts(1, 1),
            0,
            self.ring_state_bytes(),
        ))
    }
    fn query_speculation(&self) -> HostServiceResultV1<runtime_bundle_abi::BundleRawSpeculationV1> {
        let mut enabled = runtime_bundle_abi::BundleRawSpeculationV1::disabled();
        if let Some((drafts, rows)) = self.speculation.get() {
            enabled.maximum_draft_tokens = drafts;
            enabled.maximum_verify_tokens = rows;
        }
        Ok(enabled)
    }
    fn on_synchronize(&self, stream: BundleStreamHandleV1) -> HostServiceResultV1<()> {
        self.stream_syncs.set(self.stream_syncs.get() + 1);
        self.inject(PrefixFault::Synchronize)?;
        self.prefix_in_flight.borrow_mut().remove(&stream.slot());
        Ok(())
    }
    fn on_event(&self, stream: BundleStreamHandleV1, event: BundleEventHandleV1, wait: bool) {
        self.event_ops
            .borrow_mut()
            .push((stream.slot(), event.slot(), wait));
    }
    fn admit_upload(
        &self,
        _stream: BundleStreamHandleV1,
        _destination: DeviceBufferSpanV1,
        _source: &[u8],
    ) -> HostServiceResultV1<()> {
        self.inject(PrefixFault::Upload)
    }
    fn journaled<'a>(&self, source: &'a [u8]) -> &'a [u8] {
        // Immutable RoPE tables are checked by their generator tests. Retain
        // only their prefix here, rather than duplicating 512 MiB per rank.
        let is_rope_table = source.len() >= 512
            && source[..256]
                .chunks_exact(8)
                .all(|pair| pair == [0, 0, 128, 63, 0, 0, 0, 0]);
        if is_rope_table {
            &source[..512]
        } else {
            source
        }
    }
    fn on_upload(
        &self,
        _stream: BundleStreamHandleV1,
        destination: DeviceBufferSpanV1,
        source: &[u8],
    ) -> HostServiceResultV1<()> {
        let recorded = self.journaled(source);
        if source.len() == crate::prefix_projected::HEADER_BYTES as usize
            && source.starts_with(b"DFCTX001")
        {
            let mut memory = self.snapshot_memory.borrow_mut();
            for (offset, byte) in source.iter().enumerate() {
                memory.insert(
                    (destination.buffer.raw, destination.offset + offset as u64),
                    *byte,
                );
            }
        }
        self.host_copy_offsets.borrow_mut().push(destination.offset);
        self.latest_uploads.borrow_mut().insert(
            (destination.buffer.raw, destination.offset),
            recorded.to_vec(),
        );
        self.host_copies
            .borrow_mut()
            .push((destination.length, source[..source.len().min(512)].to_vec()));
        Ok(())
    }
    fn admit_download(
        &self,
        _stream: BundleStreamHandleV1,
        _source: DeviceBufferSpanV1,
    ) -> HostServiceResultV1<()> {
        self.inject(PrefixFault::Readback)
    }
    fn download_line(&self, stream: u32, source: DeviceBufferSpanV1, bytes: usize) -> String {
        format!("readback {stream} {source:?} {bytes}")
    }
    fn read_back(
        &self,
        destination: &mut [u8],
        source: DeviceBufferSpanV1,
    ) -> HostServiceResultV1<()> {
        let tokens = self.readback_tokens.borrow();
        if self.tap_readbacks.get() {
            destination.fill(0);
            return Ok(());
        }
        if self.encoder_readbacks.get() && source.buffer.slot() >= 60000 {
            if let Some(hook) = self.encoder_readback_hook.borrow_mut().as_mut() {
                hook();
            }
            destination.fill(0);
            return Ok(());
        }
        if source.length == crate::prefix_projected::HEADER_BYTES
            && self
                .snapshot_memory
                .borrow()
                .contains_key(&(source.buffer.raw, source.offset))
        {
            let memory = self.snapshot_memory.borrow();
            for (offset, byte) in destination.iter_mut().enumerate() {
                *byte = *memory
                    .get(&(source.buffer.raw, source.offset + offset as u64))
                    .unwrap_or(&0);
            }
            return Ok(());
        }
        assert_eq!(destination.len(), tokens.len() * 4);
        for (bytes, token) in destination.chunks_exact_mut(4).zip(tokens.iter()) {
            bytes.copy_from_slice(&token.to_le_bytes());
        }
        Ok(())
    }
    fn admit_copy(
        &self,
        _stream: BundleStreamHandleV1,
        destination: DeviceBufferSpanV1,
        source: DeviceBufferSpanV1,
    ) -> HostServiceResultV1<()> {
        if self.mtp_images.borrow().is_empty() && self.encoder_images.borrow().is_empty() {
            assert_eq!(destination.buffer, source.buffer);
        }
        Ok(())
    }
    fn on_copy(
        &self,
        stream: BundleStreamHandleV1,
        destination: DeviceBufferSpanV1,
        source: DeviceBufferSpanV1,
    ) -> HostServiceResultV1<()> {
        if let Some(probe) = self.offload.borrow_mut().as_mut() {
            probe.hidden_copy(stream.slot(), source);
        }
        self.device_copies.borrow_mut().push((source, destination));
        let mut memory = self.snapshot_memory.borrow_mut();
        let bytes = memory
            .range(
                (source.buffer.raw, source.offset)
                    ..(source.buffer.raw, source.offset + source.length),
            )
            .map(|(&(_, offset), &byte)| (offset - source.offset, byte))
            .collect::<Vec<_>>();
        memory.retain(|&(buffer, offset), _| {
            buffer != destination.buffer.raw
                || !(destination.offset..destination.offset + destination.length).contains(&offset)
        });
        for (offset, byte) in bytes {
            memory.insert((destination.buffer.raw, destination.offset + offset), byte);
        }
        self.copies
            .borrow_mut()
            .push((source.offset, destination.offset, source.length));
        Ok(())
    }
    fn copy_peer_to_peer(
        &self,
        stream: BundleStreamHandleV1,
        destination_context: BundleContextHandleV1,
        destination: DeviceBufferSpanV1,
        source_context: BundleContextHandleV1,
        source: DeviceBufferSpanV1,
    ) -> HostServiceResultV1<()> {
        assert_eq!(destination.length, source.length);
        self.note(format!(
            "peer copy {} {} {source:?} {} {destination:?}",
            stream.slot(),
            source_context.slot(),
            destination_context.slot()
        ));
        self.copies
            .borrow_mut()
            .push((source.offset, destination.offset, source.length));
        Ok(())
    }
    fn on_zero(&self, _stream: BundleStreamHandleV1, _destination: DeviceBufferSpanV1) {
        self.zeroes.set(self.zeroes.get() + 1);
    }
    fn on_resolve(&self, _symbol_name: &[u8]) {
        self.symbols.set(self.symbols.get() + 1);
    }
    fn on_launch(
        &self,
        stream: BundleStreamHandleV1,
        name: &str,
        config: LaunchConfigV1,
        arguments: &[KernelArgumentV1],
    ) -> HostServiceResultV1<()> {
        self.stream_launches
            .borrow_mut()
            .push((stream.slot(), name.into(), config.grid_x));
        self.launch_records.borrow_mut().push((
            stream.slot(),
            name.into(),
            [config.grid_x, config.grid_y, config.grid_z],
            arguments.to_vec(),
        ));
        // Wide rings' copies end in `_s{slots}_v1`.
        if name.contains("_ring_prefix_copy_") {
            self.snapshot_op(|| SnapshotOp::Launch(stream.slot(), arguments.to_vec()));
        }
        if name == "chop_row_embedding_full_v1" {
            assert_eq!(arguments.len(), 6);
            // The token embedding reads the vocabulary table; the prefill
            // head's row gather reads that step's normalized hidden rows.
            if arguments[5] == KernelArgumentV1::ScalarU32(152_576) {
                assert!(matches!(
                    arguments[0],
                    KernelArgumentV1::DeviceBuffer(span) if span.offset == 0 && span.length == 1_249_902_592
                ));
            } else {
                assert!(matches!(
                    arguments[0],
                    KernelArgumentV1::DeviceBuffer(span) if span.length == u64::from(PREFILL_TOKENS) * 4096 * 2
                ));
            }
        } else if name == "chop_row_rms_mimo_hf4096_v1" {
            assert_eq!(arguments.len(), 7);
            let mut norm_offsets = vec![1_249_902_592];
            if self.dflash.get() && !self.mtp_images.borrow().is_empty() {
                let index = crate::index::parse(&fake_dflash_index(0, self.tp()).0, 0).unwrap();
                norm_offsets.push(index.views["layers.0.input_layernorm.weight"].offset);
            } else if !self.mtp_images.borrow().is_empty() {
                let index = crate::index::parse(&fake_mtp_index(0, self.tp()).0, 0).unwrap();
                for name in ["enorm", "hnorm"] {
                    norm_offsets
                        .push(index.views[&format!("model.mtp.layers.0.{name}.weight")].offset);
                }
            }
            assert!(matches!(
                arguments[1],
                KernelArgumentV1::DeviceBuffer(span) if norm_offsets.contains(&span.offset) && span.length == 8192
            ));
            assert_eq!(arguments[2], KernelArgumentV1::NullDevicePointer);
            assert_eq!(arguments[3], KernelArgumentV1::NullDevicePointer);
        } else if name.ends_with("_dflash_copy_rows_v1") {
            assert_eq!(arguments.len(), 7);
            let KernelArgumentV1::DeviceBuffer(output) = arguments[1] else {
                panic!("DFlash row copy output must be a device buffer");
            };
            self.dflash_copy_outputs.borrow_mut().push(output.offset);
        } else if name == crate::retain::retain_rows_symbol(self.tp()) {
            assert_eq!(arguments.len(), 12);
            let (
                KernelArgumentV1::DeviceBuffer(history),
                KernelArgumentV1::ScalarU32(width),
                KernelArgumentV1::ScalarU32(capacity),
            ) = (&arguments[1], &arguments[7], &arguments[10])
            else {
                panic!("retained-row launch arguments");
            };
            self.retain_launches
                .borrow_mut()
                .push((*history, *capacity, *width));
        } else if name == crate::retain::gather_rows_symbol(self.tp()) {
            assert_eq!(arguments.len(), 11);
            self.gather_launches.borrow_mut().push(arguments.to_vec());
        } else if name.ends_with("_ring_prefix_copy_v1") {
            assert_eq!(arguments.len(), 9);
            self.inject(PrefixFault::Launch)?;
            self.prefix_in_flight.borrow_mut().insert(stream.slot());
            self.prefix_copies
                .borrow_mut()
                .push((stream.slot(), arguments.to_vec()));
        } else {
            assert!(!arguments.is_empty());
        }
        if let Some(probe) = self.offload.borrow_mut().as_mut() {
            if name == crate::sm120::ring_entries::prefix_copy_symbol(self.tp(), self.ring_slots())
            {
                probe.copy(self, stream.slot(), arguments);
            }
            if name == crate::sm120::ring_entries::ring_backup_symbol(self.tp(), self.ring_slots())
                || name == crate::retain::retain_rows_symbol(self.tp())
            {
                probe.launch(self, stream.slot(), name, arguments);
            }
        }
        self.launch_symbols.borrow_mut().push(name.into());
        self.launches.set(self.launches.get() + 1);
        Ok(())
    }
    fn on_capture(&self, stream: BundleStreamHandleV1) {
        if let Some(probe) = self.offload.borrow_mut().as_mut() {
            probe.begin(stream.slot());
        }
        self.graph_begins.set(self.graph_begins.get() + 1);
    }
    fn on_captured(&self, stream: BundleStreamHandleV1, graph: BundleGraphHandleV1) {
        self.graph_ends.set(self.graph_ends.get() + 1);
        if let Some(probe) = self.offload.borrow_mut().as_mut() {
            probe.end(stream.slot(), graph.slot());
        }
        self.live_graphs.borrow_mut().push(graph.slot());
    }
    fn on_replay(&self, stream: BundleStreamHandleV1, graph: BundleGraphHandleV1, nodes: usize) {
        self.replayed_nodes.set(self.replayed_nodes.get() + nodes);
        if let Some(probe) = self.offload.borrow_mut().as_mut() {
            probe.replay(self, stream.slot(), graph.slot());
        }
        self.graph_replays.set(self.graph_replays.get() + 1);
        self.graph_launches
            .borrow_mut()
            .push((stream.slot(), graph.slot()));
    }
}
impl Package {
    /// The history cells a catch-up gather reads from the inputs last
    /// uploaded to its arena, in row order, and the history it reads; `None`
    /// when those inputs do not hold its rows.
    fn gathered_cells(&self, launch: &[KernelArgumentV1]) -> Option<(u64, Vec<u64>)> {
        let span = |index: usize| match launch[index] {
            KernelArgumentV1::DeviceBuffer(span) => span,
            _ => panic!("gather argument {index} is not a device buffer"),
        };
        let KernelArgumentV1::ScalarU32(capacity) = launch[9] else {
            panic!("gather capacity");
        };
        let uploads = self.latest_uploads.borrow();
        let words = |span: DeviceBufferSpanV1, count: usize| {
            let ((_, offset), bytes) = uploads
                .range((span.buffer.raw, 0)..=(span.buffer.raw, span.offset))
                .next_back()?;
            let start = usize::try_from(span.offset - offset).ok()?;
            let words = bytes
                .get(start..start.checked_add(4 * count)?)?
                .chunks_exact(4)
                .map(|word| i32::from_le_bytes(word.try_into().unwrap()))
                .collect::<Vec<_>>();
            Some(words)
        };
        let live = usize::try_from(*words(span(5), 1)?.first()?).ok()?;
        let batch = words(span(3), live)?;
        let positions = words(span(4), live)?;
        let items = batch.iter().map(|&item| item.max(0) as usize + 1).max()?;
        let slots = words(span(2), items)?;
        let cells = batch
            .iter()
            .zip(&positions)
            .map(|(&item, &position)| {
                crate::retain::cell(
                    slots[item as usize] as u32,
                    position as u64,
                    u64::from(capacity),
                )
            })
            .collect();
        Some((span(0).buffer.raw, cells))
    }

    /// The histories whose catch-up gathers read exactly `cells`.
    fn histories_gathering(&self, cells: &[u64]) -> BTreeSet<u64> {
        self.gather_launches
            .borrow()
            .iter()
            .filter_map(|launch| self.gathered_cells(launch))
            .filter(|(_, gathered)| gathered == cells)
            .map(|(history, _)| history)
            .collect()
    }
}

fn fake_role(rank: u32) -> String {
    if rank < DRAFTER_INDEX {
        format!("chop.bundle-metadata.mimo.rank-index.r{rank}.v1")
    } else {
        format!(
            "chop.bundle-metadata.mimo.drafter-rank-index.r{}.v1",
            rank - DRAFTER_INDEX
        )
    }
}
const SCHEMA: &str = "chopinfer.mimo26.rank-image.v1";

fn fake_dflash_index(rank: u32, tp: Tp) -> (Vec<u8>, u64) {
    let index = crate::dflash::views::tests::index_at(rank, tp);
    let mut fake = FakeIndex::new(SCHEMA, rank);
    fake.image = index.image;
    fake.sha256 = index.sha256.expect("a digest");
    fake.views = index.views.into_iter().collect();
    (fake.bytes(index.image_bytes), index.image_bytes)
}
fn fake_mtp_index(rank: u32, tp: Tp) -> (Vec<u8>, u64) {
    let mut index = FakeIndex::new(SCHEMA, rank);
    for layer in 0..tp.geometry.mtp_layers() {
        for (suffix, shape, dtype, layout) in crate::mtp::views::mtp_fields(tp) {
            let bytes = shape.iter().product::<u64>()
                * match dtype {
                    "BF16" => 2,
                    "F32" => 4,
                    _ => 1,
                };
            let name = format!("model.mtp.layers.{layer}.{suffix}");
            index.push(name, bytes, dtype, shape.to_vec(), layout, 4096);
        }
    }
    let image_bytes = index.end(4096);
    (index.bytes(image_bytes), image_bytes)
}

fn fake_index(rank: u32, tp: Tp) -> Vec<u8> {
    let mut index = FakeIndex::new(SCHEMA, rank);
    let contracts = model_mimo26_contract::text_views(tp);
    // Every view at its contract length, in a fixed order.
    let mut add = |name: String| {
        let contract = &contracts[&name];
        index.push(
            name,
            contract.bytes,
            contract.dtype,
            contract.shape.clone(),
            contract.layout_id,
            4096,
        );
    };
    add("model.embed_tokens.weight".into());
    add("model.layers.0.input_layernorm.weight".into());
    add("model.norm.weight".into());
    add("lm_head.weight".into());
    for layer in 0..tp.geometry.layers() {
        let stem = format!("model.layers.{layer}");
        if layer != 0 {
            add(format!("{stem}.input_layernorm.weight"));
        }
        add(format!("{stem}.post_attention_layernorm.weight"));
        add(format!("{stem}.self_attn.qkv_proj.weight"));
        add(format!("{stem}.self_attn.qkv_proj.weight_scale_inv"));
        add(format!("{stem}.self_attn.o_proj.weight"));
        if !tp.geometry.is_global(layer) {
            add(format!("{stem}.self_attn.attention_sink_bias"));
        }
        if layer < tp.geometry.dense_layer_end() {
            add(format!("{stem}.mlp.gate_up_proj.weight"));
            add(format!("{stem}.mlp.gate_up_proj.weight_scale_inv"));
            add(format!("{stem}.mlp.down_proj.weight"));
            add(format!("{stem}.mlp.down_proj.weight_scale_inv"));
        } else {
            add(format!("{stem}.mlp.gate.weight"));
            add(format!("{stem}.mlp.gate.e_score_correction_bias"));
            add(format!("{stem}.mlp.experts.gate_up_proj.weight"));
            add(format!("{stem}.mlp.experts.gate_up_proj.weight_scale"));
            add(format!("{stem}.mlp.experts.down_proj.weight"));
            add(format!("{stem}.mlp.experts.down_proj.weight_scale"));
        }
    }
    assert_eq!(index.views.len(), contracts.len());
    assert!(index.end(4096) <= fake_image_bytes(tp));
    index.bytes(fake_image_bytes(tp))
}
fn configuration() -> Vec<u8> {
    configuration_with_mtp(false)
}

fn configuration_with_mtp(mtp: bool) -> Vec<u8> {
    configuration_at(Tp::TP4, mtp)
}

/// The create record of a `tp` package, with a drafter when `mtp`.
fn configuration_at(tp: Tp, mtp: bool) -> Vec<u8> {
    configuration_for_at(tp, &crate::contract::KV_FORMATS[0], mtp)
}

fn configuration_for_at(tp: Tp, kv: &bundle_runtime::kv::KvFormat, mtp: bool) -> Vec<u8> {
    configuration_serving(tp, kv, mtp, 32)
}

/// The create record of an instance serving `served` sequences: its ring
/// state holds one slot per item of the largest bucket, and the record
/// closes with the count.
fn configuration_serving(
    tp: Tp,
    kv: &bundle_runtime::kv::KvFormat,
    mtp: bool,
    served: u32,
) -> Vec<u8> {
    let ordinals = (0..tp.ranks() as i32).collect::<Vec<_>>();
    configuration_on(tp, kv, mtp, &ordinals, served)
}

/// The create record of an instance whose rank r runs on `ordinals[r]`.
fn configuration_on(
    tp: Tp,
    kv: &bundle_runtime::kv::KvFormat,
    mtp: bool,
    ordinals: &[i32],
    served: u32,
) -> Vec<u8> {
    configuration_on_with_encoders(tp, kv, mtp, ordinals, served, false)
}

/// The create record of an instance with every encoder when `encoders`.
fn configuration_on_with_encoders(
    tp: Tp,
    kv: &bundle_runtime::kv::KvFormat,
    mtp: bool,
    ordinals: &[i32],
    served: u32,
    encoders: bool,
) -> Vec<u8> {
    let encoders = if encoders {
        model_mimo26_contract::encoders::Encoders::ALL
    } else {
        model_mimo26_contract::encoders::Encoders::NONE
    };
    configuration_on_serving_encoders(tp, kv, mtp, ordinals, served, encoders)
}

/// Whether a package whose encoders serve `encoders` holds the encoder-set
/// component: when they serve some media but not every one, as the compiler
/// packages it (`mimo26_package`).
fn names_encoder_set(encoders: model_mimo26_contract::encoders::Encoders) -> bool {
    encoders.any() && encoders != model_mimo26_contract::encoders::Encoders::ALL
}

/// The create record of an instance whose encoders serve `encoders`: the
/// package names its encoder set unless they serve every medium.
fn configuration_on_serving_encoders(
    tp: Tp,
    kv: &bundle_runtime::kv::KvFormat,
    mtp: bool,
    ordinals: &[i32],
    served: u32,
    encoders: model_mimo26_contract::encoders::Encoders,
) -> Vec<u8> {
    let ranks = tp.ranks();
    let mut roles = vec![
        b"mimo.rank-images.v1".to_vec(),
        model_mimo26_contract::prepared::COMPONENT_ROLE
            .as_bytes()
            .to_vec(),
    ];
    roles.extend(
        (0..ranks)
            .map(|rank| format!("chop.bundle-metadata.mimo.rank-index.r{rank}.v1").into_bytes()),
    );
    if encoders.any() {
        roles.push(crate::encoders::views::IMAGE_ROLE.to_vec());
        roles.extend((0..ranks).map(|rank| {
            format!("chop.bundle-metadata.mimo.encoder-rank-index.r{rank}.v1").into_bytes()
        }));
    }
    if names_encoder_set(encoders) {
        roles.push(
            model_mimo26_contract::weight_plan::ENCODER_SET_ROLE
                .as_bytes()
                .to_vec(),
        );
    }
    if mtp {
        roles.push(crate::mtp::views::IMAGE_ROLE.to_vec());
        roles.extend((0..ranks).map(|rank| fake_role(rank + DRAFTER_INDEX).into_bytes()));
    }
    bundle_runtime::config::test_support::encode(
        crate::topology::topology_id(tp).as_bytes(),
        ordinals,
        &bundle_runtime::config::KvLayout {
            layout_id: kv.layout_id.to_vec(),
            page_bytes: crate::topology::ga_page_bytes(tp, kv.fp8()),
            alignment: 256,
            page_tokens: PAGE_TOKENS,
            ranks: u64::from(ranks),
            lifetime: 0,
        },
        &bundle_runtime::config::StateLayout {
            state_id: b"mimo.swa-ring.v1".to_vec(),
            bytes: crate::topology::ring_state_bytes(tp, served.next_power_of_two()),
            alignment: 256,
        },
        &roles,
        Some(served),
    )
}

#[derive(Default)]
struct ViewSink(Vec<OwnedLogitsViewV1>);

/// The logits limits of the MiMo tests' sinks.
struct Limits;

/// A package built before the precision component is refused at create,
/// before any device work, with the missing component named.
#[test]
fn packages_without_the_precision_component_are_refused() {
    fn field(value: &[u8]) -> Vec<u8> {
        let mut bytes = (value.len() as u32).to_le_bytes().to_vec();
        bytes.extend_from_slice(value);
        bytes
    }
    fn find(bytes: &[u8], field: &[u8]) -> usize {
        bytes
            .windows(field.len())
            .position(|window| window == field)
            .unwrap()
    }
    let mut config = configuration_at(Tp::TP2, false);
    let precision = field(model_mimo26_contract::prepared::COMPONENT_ROLE.as_bytes());
    let at = find(&config, &precision);
    config.drain(at..at + precision.len());
    let count = find(&config, &field(b"mimo.rank-images.v1")) - 4;
    let roles = u32::from_le_bytes(config[count..count + 4].try_into().unwrap());
    config[count..count + 4].copy_from_slice(&(roles - 1).to_le_bytes());
    let api = BundleApiV1::new_with_prefix_cache(crate::BUNDLE_MIMO26_EXTENDED_API_V1).unwrap();
    let host = Rc::new(FakeHost::at(Tp::TP2));
    let created = api.create(HostServicesTableV1::new(Rc::clone(&host)), &config);
    assert!(matches!(
        created.err(),
        Some(BundleAbiError::Status(BundleStatusV1::Unsupported))
    ));
    assert!(host.allocations.borrow().is_empty());
    assert_eq!(host.launches.get(), 0);
    assert!(host.logs.borrow().iter().any(|line| {
        line.contains("lacks the precision component chop.bundle-metadata.mimo.precision.v1")
    }));
}

#[test]
fn precision_component_is_authenticated_before_rank_acquisition() {
    let api = BundleApiV1::new_with_prefix_cache(crate::BUNDLE_MIMO26_EXTENDED_API_V1).unwrap();
    for fault in [1, 2, 3, 4, 5] {
        let host = Rc::new(FakeHost::at(Tp::TP2));
        host.bad_precision.set(fault);
        assert!(
            api.create(
                HostServicesTableV1::new(Rc::clone(&host)),
                &configuration_at(Tp::TP2, false)
            )
            .is_err()
        );
        assert!(host.allocations.borrow().is_empty());
        assert_eq!(host.launches.get(), 0);
        assert!(host.host_copies.borrow().is_empty());
    }
}

impl SinkLimits for Limits {
    fn limits() -> ResultSinkLimitsV1 {
        ViewSink::default().limits()
    }
}

type DraftSink = RecordingSink<Limits>;

#[test]
fn mtp_drafts_emit_before_logits_and_catch_up_only_committed_rows() {
    use runtime_bundle_sdk::{BatchAppendV1, BatchItemV1, BatchRequestV1, BatchWireVersionV1};
    let api = BundleApiV1::new_with_prefix_cache(crate::BUNDLE_MIMO26_EXTENDED_API_V1).unwrap();
    let host = Rc::new(FakeHost::default());
    host.speculation.set(Some((7, 32)));
    let mut instance = api
        .create(
            HostServicesTableV1::new(Rc::clone(&host)),
            &configuration_with_mtp(true),
        )
        .unwrap();
    let request = |start, tokens: Vec<u32>, verify| {
        let rows = if verify { 4 } else { tokens.len() as u32 };
        BatchRequestV1::new(
            BatchWireVersionV1::V3,
            vec![if verify {
                BatchItemV1::verify(7, rows, true, tokens)
            } else {
                BatchItemV1::new(7, false, rows, true, tokens)
            }],
            vec![BatchAppendV1::new(
                start,
                u64::from(rows),
                start + u64::from(rows),
                vec![1],
                None,
            )],
        )
        .encode()
        .unwrap()
    };
    instance
        .execute(
            &request(0, vec![40, 41, 42], false),
            &mut DraftSink::default(),
        )
        .unwrap();
    instance.commit().unwrap();
    *host.readback_tokens.borrow_mut() = vec![43, 44, 45, 46];
    let before = host.launch_symbols.borrow().len();
    let mut sink = DraftSink::default();
    instance
        .execute(&request(3, vec![43], true), &mut sink)
        .unwrap();
    assert_eq!(
        sink.drafts,
        [OwnedDraftTokensV1 {
            request_id: 7,
            tokens: vec![44, 45, 46]
        }]
    );
    assert_eq!(sink.views.iter().map(|v| v.raw().row_count).sum::<u32>(), 4);
    let launched = host.launch_symbols.borrow()[before..].to_vec();
    assert_eq!(
        launched
            .iter()
            .filter(|s| s.as_str() == "chop_row_copy_column_slice_v1")
            .count(),
        if std::env::var_os("CHOPINFER_MIMO_MTP_EAGER").is_some_and(|value| value == "1") {
            12
        } else {
            8
        }
    );
    assert_eq!(
        launched
            .iter()
            .filter(|s| s.as_str() == "chop_row_dflash_argmax_final_v1")
            .count(),
        if std::env::var_os("CHOPINFER_MIMO_MTP_EAGER").is_some_and(|value| value == "1") {
            12
        } else {
            4
        }
    );
    instance.commit_accepted(&[kept(7, 2)]).unwrap();
    *host.readback_tokens.borrow_mut() = vec![90, 91, 92, 93];
    instance
        .execute(&request(5, vec![90], true), &mut DraftSink::default())
        .unwrap();
    instance.cancel().unwrap();
    // Cancel preserves the previous target rows and allows the same catch-up.
    instance
        .execute(&request(5, vec![90], true), &mut DraftSink::default())
        .unwrap();
    instance.commit_accepted(&[kept(7, 1)]).unwrap();
    instance.retire_request(7).unwrap();
    let graphs = host.graph_ends.get();
    instance.destroy().unwrap();
    assert_eq!(host.graph_releases.get(), graphs);
}

/// The MTP ring's declared cancel save (`model_mimo26_contract::state`): a
/// step whose drafter runs copies each rank's whole MTP ring to a separate
/// buffer, and cancel copies that buffer back.
#[test]
fn a_cancelled_drafting_step_restores_the_declared_mtp_ring() {
    use bundle_runtime::state::{Cancel, Extent, Region, Save, Shadow, Writer};
    use runtime_bundle_sdk::{BatchAppendV1, BatchItemV1, BatchRequestV1, BatchWireVersionV1};
    let state = model_mimo26_contract::state::request_state(
        Tp::TP4,
        &KV_FORMATS[0],
        crate::prefix::Drafter::Mtp,
    );
    let ring = state
        .buffers
        .iter()
        .find(|buffer| buffer.name == "mtp.ring")
        .unwrap();
    assert_eq!(
        (ring.shadow, &ring.cancel),
        (
            Shadow::Separate,
            &Cancel::Restore(vec![Save {
                writer: Writer::Drafter,
                region: Region::All,
            }])
        )
    );
    let extent = Extent {
        served: 1,
        bucket: 1,
        verify_rows: 4,
    };
    let (offset, length) = ring.region(extent, Region::All).unwrap();
    assert_eq!(offset, 0);
    let api = BundleApiV1::new_with_prefix_cache(crate::BUNDLE_MIMO26_EXTENDED_API_V1).unwrap();
    let host = Rc::new(FakeHost::default());
    host.speculation.set(Some((7, 32)));
    let mut instance = api
        .create(
            HostServicesTableV1::new(Rc::clone(&host)),
            &configuration_with_mtp(true),
        )
        .unwrap();
    let request = |start, tokens: Vec<u32>, verify| {
        let rows = if verify { 4 } else { tokens.len() as u32 };
        BatchRequestV1::new(
            BatchWireVersionV1::V3,
            vec![if verify {
                BatchItemV1::verify(7, rows, true, tokens)
            } else {
                BatchItemV1::new(7, false, rows, true, tokens)
            }],
            vec![BatchAppendV1::new(
                start,
                u64::from(rows),
                start + u64::from(rows),
                vec![1],
                None,
            )],
        )
        .encode()
        .unwrap()
    };
    instance
        .execute(
            &request(0, vec![40, 41, 42], false),
            &mut DraftSink::default(),
        )
        .unwrap();
    instance.commit().unwrap();
    let ring_copies = |from: usize| {
        host.device_copies.borrow()[from..]
            .iter()
            .copied()
            .filter(|(source, _)| source.length == length)
            .collect::<Vec<_>>()
    };
    let ranks = Tp::TP4.count();
    *host.readback_tokens.borrow_mut() = vec![43, 44, 45, 46];
    let before = host.device_copies.borrow().len();
    instance
        .execute(&request(3, vec![43], true), &mut DraftSink::default())
        .unwrap();
    // The saves come first; chained drafts copy the same length later.
    let saves = ring_copies(before)[..ranks].to_vec();
    let before = host.device_copies.borrow().len();
    instance.cancel().unwrap();
    let restores = ring_copies(before);
    assert_eq!(
        restores,
        saves
            .iter()
            .map(|&(ring, shadow)| (shadow, ring))
            .collect::<Vec<_>>()
    );
    for (ring, shadow) in saves {
        // Separate: neither copy overlaps the other.
        assert!(
            ring.buffer != shadow.buffer
                || ring.offset + length <= shadow.offset
                || shadow.offset + length <= ring.offset
        );
    }
    instance.retire_request(7).unwrap();
    instance.destroy().unwrap();
}

#[test]
fn mtp_batches_ragged_drafts_and_keeps_transaction_prefixes() {
    use runtime_bundle_sdk::{BatchAppendV1, BatchItemV1, BatchRequestV1, BatchWireVersionV1};
    let api = BundleApiV1::new_with_prefix_cache(crate::BUNDLE_MIMO26_EXTENDED_API_V1).unwrap();
    let host = Rc::new(FakeHost::default());
    host.speculation.set(Some((7, 32)));
    let mut instance = api
        .create(
            HostServicesTableV1::new(Rc::clone(&host)),
            &configuration_with_mtp(true),
        )
        .unwrap();
    let request = |starts: [u64; 3], verify| {
        let lengths = if verify { [4, 2, 3] } else { [1, 1, 1] };
        let items = (0..3)
            .map(|index| {
                let id = 7 + index as u64;
                if verify {
                    BatchItemV1::verify(id, lengths[index], true, vec![40])
                } else {
                    BatchItemV1::new(id, false, 1, true, vec![40])
                }
            })
            .collect();
        let appends = (0..3)
            .map(|index| {
                let rows = u64::from(lengths[index]);
                BatchAppendV1::new(
                    starts[index],
                    rows,
                    starts[index] + rows,
                    vec![index as u64 + 1],
                    None,
                )
            })
            .collect();
        BatchRequestV1::new(BatchWireVersionV1::V3, items, appends)
            .encode()
            .unwrap()
    };
    instance
        .execute(&request([0; 3], false), &mut DraftSink::default())
        .unwrap();
    instance.commit().unwrap();
    *host.readback_tokens.borrow_mut() = vec![40, 41, 42, 43, 40, 44, 40, 45, 46];
    let before = host.launch_symbols.borrow().len();
    let mut sink = DraftSink::default();
    instance.execute(&request([1; 3], true), &mut sink).unwrap();
    assert_eq!(
        sink.drafts
            .iter()
            .map(|row| row.tokens.len())
            .collect::<Vec<_>>(),
        [3, 1, 2]
    );
    let launched = host.launch_symbols.borrow()[before..].to_vec();
    for name in [
        "chop_row_copy_column_slice_v1",
        "chop_row_dflash_argmax_final_v1",
        "chop_row_dflash_argmax_partial_v1",
    ] {
        assert_eq!(
            launched
                .iter()
                .filter(|symbol| symbol.as_str() == name)
                .count(),
            4 * 3,
            "{name}"
        );
    }
    instance
        .commit_accepted(&[kept(7, 2), kept(8, 1), kept(9, 3)])
        .unwrap();
    // Different retained prefix lengths become a single catch-up batch.
    instance
        .execute(&request([3, 2, 4], true), &mut DraftSink::default())
        .unwrap();
    instance.cancel().unwrap();
    instance
        .execute(&request([3, 2, 4], true), &mut DraftSink::default())
        .unwrap();
    instance
        .commit_accepted(&[kept(7, 1), kept(8, 1), kept(9, 1)])
        .unwrap();
    for id in 7..10 {
        instance.retire_request(id).unwrap();
    }
    instance.destroy().unwrap();
}

#[test]
fn mtp_graphs_reuse_buckets_when_live_items_change() {
    use runtime_bundle_sdk::{BatchAppendV1, BatchItemV1, BatchRequestV1, BatchWireVersionV1};
    let api = BundleApiV1::new_with_prefix_cache(crate::BUNDLE_MIMO26_EXTENDED_API_V1).unwrap();
    let host = Rc::new(FakeHost::default());
    host.speculation.set(Some((7, 32)));
    let mut instance = api
        .create(
            HostServicesTableV1::new(Rc::clone(&host)),
            &configuration_with_mtp(true),
        )
        .unwrap();
    let request = |starts: &[u64], verify| {
        let rows = if verify { 2 } else { 1 };
        BatchRequestV1::new(
            BatchWireVersionV1::V3,
            starts
                .iter()
                .enumerate()
                .map(|(index, _)| {
                    if verify {
                        BatchItemV1::verify(7 + index as u64, rows, true, vec![40])
                    } else {
                        BatchItemV1::new(7 + index as u64, false, rows, true, vec![40])
                    }
                })
                .collect(),
            starts
                .iter()
                .enumerate()
                .map(|(index, &start)| {
                    BatchAppendV1::new(
                        start,
                        u64::from(rows),
                        start + u64::from(rows),
                        vec![index as u64 + 1],
                        None,
                    )
                })
                .collect(),
        )
        .encode()
        .unwrap()
    };
    instance
        .execute(&request(&[0; 4], false), &mut DraftSink::default())
        .unwrap();
    instance.commit().unwrap();
    *host.readback_tokens.borrow_mut() = vec![40; 6];
    instance
        .execute(&request(&[1; 3], true), &mut DraftSink::default())
        .unwrap();
    instance
        .commit_accepted(&[kept(7, 1), kept(8, 1), kept(9, 1)])
        .unwrap();
    let captures = host.graph_ends.get();
    let replays = host.graph_replays.get();
    let before = host.launch_symbols.borrow().len();
    *host.readback_tokens.borrow_mut() = vec![40; 8];
    instance
        .execute(&request(&[2, 2, 2, 1], true), &mut DraftSink::default())
        .unwrap();
    assert_eq!(host.graph_ends.get(), captures);
    assert!(host.graph_replays.get() > replays);
    if std::env::var_os("CHOPINFER_MIMO_MTP_EAGER").is_none_or(|value| value != "1") {
        assert!(
            !host.launch_symbols.borrow()[before..]
                .iter()
                .any(|name| name == "chop_row_copy_column_slice_v1")
        );
    }
    instance.cancel().unwrap();
    instance.destroy().unwrap();
    assert_eq!(host.graph_releases.get(), captures);
}

#[test]
fn mtp_keeps_target_prefill_overlap_enabled() {
    use runtime_bundle_sdk::{BatchAppendV1, BatchItemV1, BatchRequestV1, BatchWireVersionV1};
    let api = BundleApiV1::new_with_prefix_cache(crate::BUNDLE_MIMO26_EXTENDED_API_V1).unwrap();
    let host = Rc::new(FakeHost::default());
    host.speculation.set(Some((7, 32)));
    let mut instance = api
        .create(
            HostServicesTableV1::new(Rc::clone(&host)),
            &configuration_with_mtp(true),
        )
        .unwrap();
    crate::instance::force_overlap(
        instance.handle(),
        Some(crate::overlap::Config {
            sms: 188,
            reserved_sms: 16,
            comm_ctas: Some(24),
            min_tokens: 512,
        }),
    );
    assert!(!host.allocations.borrow().contains(&(32 * 8192 * 4096 * 2)));
    assert!(host.allocations.borrow().contains(&(32 * 8 * 4096 * 2)));
    let comm = crate::instance::comm_streams(instance.handle());
    let request = BatchRequestV1::new(
        BatchWireVersionV1::V3,
        vec![BatchItemV1::new(77, false, 1024, true, vec![7; 1024])],
        vec![BatchAppendV1::new(0, 1024, 1024, (0..16).collect(), None)],
    )
    .encode()
    .unwrap();
    instance
        .execute(&request, &mut DraftSink::default())
        .unwrap();
    assert!(
        host.stream_launches
            .borrow()
            .iter()
            .any(|(stream, _, _)| comm.contains(stream)),
        "MTP must preserve target prefill all-reduce overlap"
    );
    instance.commit().unwrap();
    instance.destroy().unwrap();
}

#[test]
fn packaged_mtp_does_not_allocate_or_run_when_speculation_is_off() {
    let api = BundleApiV1::new_with_prefix_cache(crate::BUNDLE_MIMO26_EXTENDED_API_V1).unwrap();
    let host = Rc::new(FakeHost::default());
    let mut instance = api
        .create(
            HostServicesTableV1::new(Rc::clone(&host)),
            &configuration_with_mtp(true),
        )
        .unwrap();
    assert!(host.mtp_images.borrow().is_empty());
    assert!(!host.allocations.borrow().contains(&(32 * 8192 * 4096 * 2)));
    instance.destroy().unwrap();
}

fn dflash_instance() -> (Rc<FakeHost>, runtime_bundle_abi::BundleInstanceV1) {
    let api = BundleApiV1::new_with_prefix_cache(crate::BUNDLE_MIMO26_EXTENDED_API_V1).unwrap();
    let host = Rc::new(FakeHost::default());
    host.speculation.set(Some((7, 32)));
    host.dflash.set(true);
    let instance = api
        .create(
            HostServicesTableV1::new(Rc::clone(&host)),
            &configuration_with_mtp(true),
        )
        .unwrap();
    (host, instance)
}

/// One item of request 7: a prefill chunk, or a verify item drafting
/// `rows - 1` tokens after its bonus token.
fn dflash_step(start: u64, tokens: Vec<u32>, verify: Option<u32>) -> Vec<u8> {
    use runtime_bundle_sdk::{BatchAppendV1, BatchItemV1, BatchRequestV1, BatchWireVersionV1};
    let rows = verify.unwrap_or(tokens.len() as u32);
    BatchRequestV1::new(
        BatchWireVersionV1::V3,
        vec![match verify {
            Some(rows) => BatchItemV1::verify(7, rows, true, tokens),
            None => BatchItemV1::new(7, false, rows, true, tokens),
        }],
        vec![BatchAppendV1::new(
            start,
            u64::from(rows),
            start + u64::from(rows),
            (0..(start + u64::from(rows)).div_ceil(PAGE_TOKENS)).collect(),
            None,
        )],
    )
    .encode()
    .unwrap()
}

fn count(launched: &[String], symbol: &str) -> usize {
    launched
        .iter()
        .filter(|name| name.as_str() == symbol)
        .count()
}

#[test]
fn dflash_drafts_inside_the_verify_graph_and_follows_kept_rows() {
    const CACHE_WRITE: &str = "chop_mimo_dflash_cache_write_v1";
    const COPY_ROWS: &str = "chop_mimo_dflash_copy_rows_v1";
    const DRAFTS: &str = "chop_mimo_dflash_drafts_v1";
    const MASK_ROWS: &str = "chop_mimo_dflash_mask_rows_v1";
    let (host, mut instance) = dflash_instance();
    assert!(
        host.allocations
            .borrow()
            .contains(&crate::dflash::program::cache_bytes(Tp::TP4, 32))
    );
    let before = host.launch_symbols.borrow().len();
    instance
        .execute(
            &dflash_step(0, vec![40, 41, 42], None),
            &mut DraftSink::default(),
        )
        .unwrap();
    instance.commit().unwrap();
    let launched = host.launch_symbols.borrow()[before..].to_vec();
    // Each rank retains target taps; projection waits until drafting resumes.
    assert_eq!(count(&launched, COPY_ROWS), 8);
    assert_eq!(count(&launched, CACHE_WRITE), 0);
    assert_eq!(count(&launched, "chop_gemm_mimo_dflash_fc_prefill_v1"), 0);
    assert_eq!(count(&launched, DRAFTS) + count(&launched, MASK_ROWS), 0);

    *host.readback_tokens.borrow_mut() = vec![43, 44, 45, 46];
    let graphs = host.graph_begins.get();
    let before = host.stream_launches.borrow().len();
    let mut sink = DraftSink::default();
    instance
        .execute(&dflash_step(3, vec![43], Some(4)), &mut sink)
        .unwrap();
    assert_eq!(
        sink.drafts,
        [OwnedDraftTokensV1 {
            request_id: 7,
            tokens: vec![44, 45, 46]
        }]
    );
    assert_eq!(sink.views.iter().map(|v| v.raw().row_count).sum::<u32>(), 4);
    // The draft block is captured with the verify program, in its three
    // graph pieces per rank.
    assert_eq!(host.graph_begins.get(), graphs + RANKS * 5);
    let launched = host.stream_launches.borrow()[before..].to_vec();
    let mut streams = launched
        .iter()
        .map(|(stream, _, _)| *stream)
        .collect::<Vec<_>>();
    streams.sort_unstable();
    streams.dedup();
    assert_eq!(streams.len(), 4);
    for stream in streams {
        let names = launched
            .iter()
            .filter(|(owner, _, _)| *owner == stream)
            .map(|(_, name, _)| name.as_str())
            .collect::<Vec<_>>();
        assert!(names.contains(&"chop_row_dflash_q_rope_table_pos32_v1"));
        assert!(names.contains(&"chop_row_dflash_k_rope_table_pos32_v1"));
        assert!(!names.contains(&"chop_row_dflash_q_rope_pos32_v1"));
        assert!(!names.contains(&"chop_row_dflash_k_rope_pos32_v1"));
        let at = |symbol| names.iter().position(|&name| name == symbol).unwrap();
        let embeddings = names
            .iter()
            .enumerate()
            .filter(|(_, name)| **name == "chop_row_embedding_full_v1")
            .map(|(index, _)| index)
            .collect::<Vec<_>>();
        // The block embeds and masks its rows, drafts, and only then does the
        // target embed the drafted tokens.
        assert_eq!(embeddings.len(), 2);
        assert!(embeddings[0] < at(MASK_ROWS));
        assert!(at(MASK_ROWS) < at(DRAFTS) && at(DRAFTS) < embeddings[1]);
        assert!(at(CACHE_WRITE) < at("chop_mimo_ring_backup_v1"));
        assert_eq!(
            names
                .iter()
                .filter(|&&name| name == "chop_attn_mimo_dflash_block_partial_v1")
                .count(),
            5
        );
        assert!(at(CACHE_WRITE) < at("chop_row_dflash_argmax_partial_v1"));
    }
    instance.commit_accepted(&[kept(7, 2)]).unwrap();

    // The next verify item continues at the kept end and replays the graph.
    *host.readback_tokens.borrow_mut() = vec![90, 91, 92, 93];
    let before = host.launch_symbols.borrow().len();
    instance
        .execute(
            &dflash_step(5, vec![90], Some(4)),
            &mut DraftSink::default(),
        )
        .unwrap();
    let launched = host.launch_symbols.borrow()[before..].to_vec();
    assert_eq!(count(&launched, CACHE_WRITE), 20);
    assert_eq!(count(&launched, DRAFTS), 0);
    instance.cancel().unwrap();
    // Cancel keeps the committed context, so the same step runs again.
    instance
        .execute(
            &dflash_step(5, vec![90], Some(4)),
            &mut DraftSink::default(),
        )
        .unwrap();
    instance.commit_accepted(&[kept(7, 1)]).unwrap();
    *host.readback_tokens.borrow_mut() = vec![91, 92];
    assert!(
        instance
            .execute(
                &dflash_step(7, vec![90], Some(4)),
                &mut DraftSink::default()
            )
            .is_err(),
        "a verify item must start at the committed context end"
    );
    instance
        .execute(
            &dflash_step(6, vec![91], Some(2)),
            &mut DraftSink::default(),
        )
        .unwrap();
    assert!(instance.retire_request(7).is_err());
    instance.commit_accepted(&[kept(7, 2)]).unwrap();
    // A zero-draft decode step retains raw features without context projection.
    let before = host.launch_symbols.borrow().len();
    let decode = {
        use runtime_bundle_sdk::{BatchAppendV1, BatchItemV1, BatchRequestV1, BatchWireVersionV1};
        BatchRequestV1::new(
            BatchWireVersionV1::V3,
            vec![BatchItemV1::new(7, true, 1, true, vec![93])],
            vec![BatchAppendV1::new(8, 1, 9, vec![0], None)],
        )
        .encode()
        .unwrap()
    };
    instance
        .execute(&decode, &mut DraftSink::default())
        .unwrap();
    let launched = host.launch_symbols.borrow()[before..].to_vec();
    assert_eq!(count(&launched, CACHE_WRITE), 0);
    assert_eq!(count(&launched, DRAFTS), 0);
    instance.commit().unwrap();
    *host.readback_tokens.borrow_mut() = vec![94, 95];
    instance
        .execute(
            &dflash_step(9, vec![94], Some(2)),
            &mut DraftSink::default(),
        )
        .unwrap();
    instance.commit_accepted(&[kept(7, 2)]).unwrap();
    instance.retire_request(7).unwrap();
    // A retired request drafts only after new context.
    assert!(
        instance
            .execute(
                &dflash_step(8, vec![92], Some(2)),
                &mut DraftSink::default()
            )
            .is_err()
    );
    instance.destroy().unwrap();
}

#[test]
fn dflash_drafts_every_row_of_a_verify_step_wider_than_the_decode_batch() {
    const DRAFTS: &str = "chop_mimo_dflash_drafts_v1";
    const MASK_ROWS: &str = "chop_mimo_dflash_mask_rows_v1";
    use runtime_bundle_sdk::{BatchAppendV1, BatchItemV1, BatchRequestV1, BatchWireVersionV1};
    let api = BundleApiV1::new_with_prefix_cache(crate::BUNDLE_MIMO26_EXTENDED_API_V1).unwrap();
    let host = Rc::new(FakeHost::default());
    host.speculation.set(Some((7, 256)));
    host.dflash.set(true);
    let mut instance = api
        .create(
            HostServicesTableV1::new(Rc::clone(&host)),
            &configuration_with_mtp(true),
        )
        .unwrap();
    let prefill = BatchRequestV1::new(
        BatchWireVersionV1::V3,
        (0..32)
            .map(|item| BatchItemV1::new(100 + item, false, 3, true, vec![40, 41, 42]))
            .collect(),
        (0..32)
            .map(|item| BatchAppendV1::new(0, 3, 3, vec![item], None))
            .collect(),
    );
    instance
        .execute(&prefill.encode().unwrap(), &mut DraftSink::default())
        .unwrap();
    instance.commit().unwrap();
    let step = BatchRequestV1::new(
        BatchWireVersionV1::V3,
        (0..32)
            .map(|item| BatchItemV1::verify(100 + item, 8, true, vec![43]))
            .collect(),
        (0..32)
            .map(|item| BatchAppendV1::new(3, 8, 11, vec![item], None))
            .collect(),
    );
    *host.readback_tokens.borrow_mut() = (0..256).map(|row| 1000 + row).collect();
    let graphs = host.graph_begins.get();
    let before = host.stream_launches.borrow().len();
    let mut sink = DraftSink::default();
    instance
        .execute(&step.encode().unwrap(), &mut sink)
        .unwrap();
    // One catch-up pass gathers all 32 prefills' three rows on each rank.
    let window = (0..32)
        .flat_map(|slot| {
            (0..3).map(move |position| {
                crate::retain::cell(slot, position, crate::dflash::state::HISTORY_CAPACITY)
            })
        })
        .collect::<Vec<_>>();
    assert_eq!(host.histories_gathering(&window).len(), RANKS as usize);
    // Each item drafts seven tokens after its bonus row, and all 256 rows
    // emit logits.
    assert_eq!(sink.drafts.len(), 32);
    for (item, drafts) in sink.drafts.iter().enumerate() {
        assert_eq!(drafts.request_id, 100 + item as u64);
        assert_eq!(
            drafts.tokens,
            (1..8)
                .map(|row| 1000 + 8 * item as u32 + row)
                .collect::<Vec<_>>()
        );
    }
    assert_eq!(
        sink.views.iter().map(|v| v.raw().row_count).sum::<u32>(),
        256
    );
    // The (128, 32) catch-up graph, then the draft block and target in the
    // (256, 32) verify graph, five pieces per rank. The block's projections
    // use the prefill GEMMs and its drafts kernel covers every row.
    assert_eq!(host.graph_begins.get(), graphs + RANKS * 5);
    let launched = host.stream_launches.borrow()[before..].to_vec();
    let named = |symbol: &str| {
        launched
            .iter()
            .filter(|(_, name, _)| name == symbol)
            .count()
    };
    // Medium-M entries stop at 64 rows, so the switch changes nothing here.
    assert_eq!(
        named("chop_gemm_mimo_dflash_qkv_prefill_v1"),
        5 * RANKS as usize
    );
    assert_eq!(named("chop_gemm_mid_bf16_t64_v1"), 0);
    assert_eq!(named("chop_gemm_mimo_dflash_qkv_m32_v1"), 0);
    assert_eq!(named(MASK_ROWS), RANKS as usize);
    assert_eq!(named(DRAFTS), RANKS as usize);
    assert_eq!(
        named(&crate::retain::gather_rows_symbol(Tp::TP4)),
        RANKS as usize
    );
    let wide_drafts = host
        .journal
        .borrow()
        .iter()
        .filter(|line| {
            line.contains(DRAFTS) && line.contains("grid_x: 8,") && line.contains("block_x: 32,")
        })
        .count();
    assert_eq!(wide_drafts, RANKS as usize);
    instance
        .commit_accepted(&(0..32).map(|item| kept(100 + item, 8)).collect::<Vec<_>>())
        .unwrap();
}

#[test]
fn only_dflash_and_host_drafts_verify_beyond_the_decode_batch() {
    let api = BundleApiV1::new_with_prefix_cache(crate::BUNDLE_MIMO26_EXTENDED_API_V1).unwrap();
    for (drafter, dflash, rows) in [(false, false, 256), (true, false, 32), (true, true, 256)] {
        let host = Rc::new(FakeHost::default());
        host.speculation.set(Some((7, 1024)));
        host.dflash.set(dflash);
        let mut instance = api
            .create(
                HostServicesTableV1::new(Rc::clone(&host)),
                &configuration_with_mtp(drafter),
            )
            .unwrap();
        assert_eq!(
            crate::instance::verify_rows_per_step(instance.handle()),
            Some(rows),
            "drafter {drafter}, DFlash {dflash}"
        );
        instance.destroy().unwrap();
    }
}

#[test]
fn verify_buffers_follow_the_rows_the_host_enabled() {
    let api = BundleApiV1::new_with_prefix_cache(crate::BUNDLE_MIMO26_EXTENDED_API_V1).unwrap();
    let logits = |rows: u64| rows * u64::from(VOCAB_PER_RANK) * u64::from(RANKS) * 4;
    let arena = crate::upload::InputLayout::verify_within::<Mimo>(crate::contract::ITEMS_32).end;
    // (drafter, DFlash, enabled speculation, gathered logits rows, verify arena)
    for (drafter, dflash, speculation, rows, verify) in [
        (false, false, None, 32, false),
        (true, true, None, 32, false),
        (true, true, Some((7, 256)), 256, true),
        (true, true, Some((7, 32)), 32, true),
        (true, true, Some((7, 40)), 64, true),
        (true, false, Some((7, 256)), 32, true),
        (false, false, Some((7, 256)), 256, true),
    ] {
        let host = Rc::new(FakeHost::default());
        host.speculation.set(speculation);
        host.dflash.set(dflash);
        let mut instance = api
            .create(
                HostServicesTableV1::new(Rc::clone(&host)),
                &configuration_with_mtp(drafter),
            )
            .unwrap();
        let allocations = host.allocations.borrow().clone();
        let case = format!("drafter {drafter}, DFlash {dflash}, {speculation:?}");
        // Scored shards and the two BF16 buffers can equal gathered F32 sizes.
        for other in [32, 64, 128, 256] {
            let expected = [
                rows,
                rows / 2,
                rows / 8,
                u64::from(crate::contract::SCORED_ROWS / RANKS),
            ]
            .into_iter()
            .filter(|&capacity| capacity == other)
            .count()
                * RANKS as usize;
            assert_eq!(
                allocations
                    .iter()
                    .filter(|&&bytes| bytes == logits(other))
                    .count(),
                expected,
                "{case}: {other} rows"
            );
        }
        assert_eq!(allocations.contains(&arena), verify, "{case}");
        instance.destroy().unwrap();
    }
}

#[test]
fn dflash_refuses_host_drafts_and_taps_both_pipelined_prefill_halves() {
    use runtime_bundle_sdk::{BatchAppendV1, BatchItemV1, BatchRequestV1, BatchWireVersionV1};
    let (host, mut instance) = dflash_instance();
    crate::instance::force_overlap(
        instance.handle(),
        Some(crate::overlap::Config {
            sms: 188,
            reserved_sms: 16,
            comm_ctas: Some(24),
            min_tokens: 512,
        }),
    );
    instance
        .execute(
            &dflash_step(0, vec![7; 1024], None),
            &mut DraftSink::default(),
        )
        .unwrap();
    instance.commit().unwrap();
    // Rank 0's tap of layer 0 lands at its first column, for each half's first
    // row: the second half starts at row 512 of the fc shard.
    let outputs = host.dflash_copy_outputs.borrow().clone();
    assert!(outputs.contains(&0));
    assert!(outputs.contains(&(512 * crate::dflash::program::fc_shard(Tp::TP4) * 2)));
    assert_eq!(outputs.len(), 2 * 8);
    let host_drafts = BatchRequestV1::new(
        BatchWireVersionV1::V3,
        vec![BatchItemV1::verify(7, 3, false, vec![1, 2, 3])],
        vec![BatchAppendV1::new(1024, 3, 1027, (0..17).collect(), None)],
    )
    .encode()
    .unwrap();
    assert!(
        instance
            .execute(&host_drafts, &mut DraftSink::default())
            .is_err()
    );
    *host.readback_tokens.borrow_mut() = vec![5, 6, 7];
    instance
        .execute(
            &dflash_step(1024, vec![5], Some(3)),
            &mut DraftSink::default(),
        )
        .unwrap();
    instance.commit_accepted(&[kept(7, 3)]).unwrap();
    instance.destroy().unwrap();
}

impl LogitsRowSinkV1 for ViewSink {
    fn limits(&self) -> ResultSinkLimitsV1 {
        ResultSinkLimitsV1::new(
            8192,
            u64::from(VOCAB_PER_RANK) * 4,
            8192 * u64::from(VOCAB_PER_RANK) * 4,
        )
    }

    fn emit_row(&mut self, _row: OwnedLogitsRowV1) -> HostServiceResultV1<()> {
        Err(BundleStatusV1::Unsupported)
    }

    fn emit_view(&mut self, view: OwnedLogitsViewV1) -> HostServiceResultV1<()> {
        self.0.push(view);
        Ok(())
    }
}

#[test]
fn global_prefill_uses_context_to_select_attention_splits() {
    let api = BundleApiV1::new_with_prefix_cache(crate::BUNDLE_MIMO26_EXTENDED_API_V1)
        .expect("bundle API");
    let backend = Rc::new(FakeHost::default());
    let table = HostServicesTableV1::new(backend);
    let mut instance = api
        .create(table, &configuration())
        .expect("fake-host create");
    assert_eq!(
        crate::instance::prefill_ga_split_probe(instance.handle(), 1).expect("short context"),
        1
    );
    assert_eq!(
        crate::instance::prefill_ga_split_probe(instance.handle(), 512).expect("long context"),
        4
    );
    instance.destroy().expect("destroy fake-host instance");
}

/// `CHOPINFER_MIMO_FOLD_DECODE`: a mixed step's plain decode rows run in
/// its prefill forward as one-token items, with no decode graph, and every
/// item's logits view keeps its wire index and row count. A step without a
/// prefill chunk still replays the decode graphs.
#[test]
fn folded_decode_rows_run_in_the_prefill_forward() {
    use runtime_bundle_sdk::{BatchAppendV1, BatchItemV1, BatchRequestV1, BatchWireVersionV1};

    let api = BundleApiV1::new_with_prefix_cache(crate::BUNDLE_MIMO26_EXTENDED_API_V1)
        .expect("bundle API");
    let mixed = BatchRequestV1::new(
        BatchWireVersionV1::V2,
        vec![
            BatchItemV1::new(41, true, 1, true, vec![7]),
            BatchItemV1::new(42, false, 3, true, vec![8, 9, 10]),
            BatchItemV1::new(43, true, 1, true, vec![11]),
        ],
        vec![
            BatchAppendV1::new(0, 1, 1, vec![1], None),
            BatchAppendV1::new(0, 3, 3, vec![2], None),
            BatchAppendV1::new(0, 1, 1, vec![3], None),
        ],
    )
    .encode()
    .unwrap();
    let mut views = Vec::new();
    for fold in [false, true] {
        let backend = Rc::new(FakeHost::default());
        let table = HostServicesTableV1::new(Rc::clone(&backend));
        let mut instance = api.create(table, &configuration()).expect("create");
        crate::instance::force_fold_decode(instance.handle(), fold);
        let mut sink = ViewSink::default();
        instance.execute(&mixed, &mut sink).expect("mixed step");
        views.push(
            sink.0
                .iter()
                .map(|view| {
                    let raw = view.raw();
                    (raw.first_item_index, raw.row_count, raw.element_type)
                })
                .collect::<Vec<_>>(),
        );
        let symbols = backend.launch_symbols.borrow().clone();
        let decode_kernels = symbols
            .iter()
            .filter(|symbol| !symbol.starts_with("chop_comm_"))
            .filter(|symbol| symbol.contains("_decode_") || symbol.contains("_m1_"))
            .count();
        if fold {
            // One eager forward over five rows: no graph, no decode kernel.
            assert_eq!(backend.graph_begins.get(), 0);
            assert_eq!(backend.graph_replays.get(), 0);
            assert_eq!(decode_kernels, 0, "{symbols:?}");
            let rope = backend
                .stream_launches
                .borrow()
                .iter()
                .filter(|(_, symbol, _)| symbol.starts_with("chop_row_rope_mimo_"))
                .map(|&(_, _, grid)| grid)
                .collect::<std::collections::BTreeSet<_>>();
            assert_eq!(rope, [8].into());
        } else {
            assert!(backend.graph_replays.get() > 0);
        }
        instance.commit().expect("commit");
        // A decode-only step keeps the decode graphs.
        let decode = BatchRequestV1::new(
            BatchWireVersionV1::V2,
            vec![BatchItemV1::new(41, true, 1, true, vec![12])],
            vec![BatchAppendV1::new(1, 1, 2, vec![1], None)],
        );
        let replays = backend.graph_replays.get();
        instance
            .execute(&decode.encode().unwrap(), &mut ViewSink::default())
            .expect("decode step");
        assert!(backend.graph_replays.get() > replays);
        instance.commit().expect("commit decode");
        instance.destroy().expect("destroy");
    }
    assert_eq!(views[0], views[1]);
    assert_eq!(
        views[1]
            .iter()
            .map(|&(item, rows, _)| (item, rows))
            .collect::<Vec<_>>(),
        [(0, 1), (1, 1), (2, 1)]
    );
}

#[test]
fn mixed_step_runs_every_rank_stage_and_emits_prompt_and_decode_views() {
    use runtime_bundle_sdk::{BatchAppendV1, BatchItemV1, BatchRequestV1, BatchWireVersionV1};

    let api = BundleApiV1::new_with_prefix_cache(crate::BUNDLE_MIMO26_EXTENDED_API_V1)
        .expect("bundle API");
    let backend = Rc::new(FakeHost::default());
    let table = HostServicesTableV1::new(Rc::clone(&backend));
    let mut instance = api
        .create(table, &configuration())
        .expect("fake-host create");
    // This test covers the BF16 head that `CHOPINFER_MIMO_F32_LOGITS=0`
    // selects; `f32_logits_give_prefill_and_decode_rows_the_f32_head` covers
    // the default.
    crate::instance::force_f32_logits(instance.handle(), false);
    let expected = [
        crate::selection::Phase::Prefill,
        crate::selection::Phase::Decode(1),
    ]
    .into_iter()
    .flat_map(|phase| {
        let stages = crate::program::stages()
            .into_iter()
            .filter(|stage| {
                !(phase == crate::selection::Phase::Prefill
                    && matches!(
                        stage,
                        crate::program::Stage::AttentionMerge(layer)
                            if !crate::program::GLOBAL_LAYERS.contains(layer)
                    ))
            })
            .map(|stage| {
                crate::instance::plan_stage_probe(instance.handle(), phase, stage)
                    .unwrap_or_else(|status| panic!("{phase:?} {stage:?}: {status:?}"))
            })
            .collect::<Vec<_>>();
        // Eager prefill issues each stage on every rank before the next stage;
        // a decode graph is captured one rank at a time.
        if phase == crate::selection::Phase::Prefill {
            stages
                .into_iter()
                .flat_map(|stage| std::iter::repeat_n(stage, RANKS as usize).flatten())
                .collect::<Vec<_>>()
        } else {
            std::iter::repeat_n(stages.concat(), RANKS as usize)
                .flatten()
                .collect()
        }
    })
    .collect::<Vec<_>>();
    let request = BatchRequestV1::new(
        BatchWireVersionV1::V2,
        vec![
            BatchItemV1::new(41, true, 1, true, vec![7]),
            BatchItemV1::new(42, false, 2, true, vec![8, 9]).with_prompt_rows(true),
        ],
        vec![
            BatchAppendV1::new(0, 1, 1, vec![1], None),
            BatchAppendV1::new(0, 2, 2, vec![2], None),
        ],
    );
    let mut sink = ViewSink::default();
    let zeroes_before = backend.zeroes.get();
    let syncs_before = backend.stream_syncs.get();
    let copies_before = backend.host_copies.borrow().len();
    instance
        .execute(&request.encode().unwrap(), &mut sink)
        .expect("full mixed step");
    // Prefill reads a constant all-expert count written at create, so the
    // step uploads only each phase's packed inputs and new block-table rows.
    assert_eq!(
        backend.host_copies.borrow().len() - copies_before,
        2 * 2 * RANKS as usize
    );
    // Collective state is zeroed once at create; a step clears nothing and
    // blocks on every rank, since each holds its block of decode rows.
    assert_eq!(backend.zeroes.get(), zeroes_before);
    assert_eq!(backend.stream_syncs.get() - syncs_before, RANKS);
    assert_eq!(*backend.launch_symbols.borrow(), expected);
    // Each rank's decode program is captured as three graph pieces.
    let pieces = RANKS * 3;
    assert_eq!(backend.graph_begins.get(), pieces);
    assert_eq!(backend.graph_ends.get(), pieces);
    assert_eq!(backend.graph_replays.get(), pieces);
    assert_eq!(
        sink.0
            .iter()
            .map(|view| (view.raw().first_item_index, view.raw().row_count))
            .collect::<Vec<_>>(),
        [(0, 1), (1, 2)]
    );
    assert_eq!(
        sink.0
            .iter()
            .map(|view| view.raw().element_type)
            .collect::<Vec<_>>(),
        [
            runtime_bundle_abi::BUNDLE_LOGITS_ELEMENT_V1_BF16,
            runtime_bundle_abi::BUNDLE_LOGITS_ELEMENT_V1_F32,
        ]
    );
    instance.commit().expect("commit mixed ring state");
    let replay = BatchRequestV1::new(
        BatchWireVersionV1::V2,
        vec![BatchItemV1::new(41, true, 1, true, vec![10])],
        vec![BatchAppendV1::new(1, 1, 2, vec![1], None)],
    );
    let launches = backend.launches.get();
    let zeroes_before_replay = backend.zeroes.get();
    let captured = backend.graph_launches.borrow().clone();
    instance
        .execute(&replay.encode().unwrap(), &mut ViewSink::default())
        .expect("decode graph replay");
    assert_eq!(backend.zeroes.get(), zeroes_before_replay);
    assert_eq!(backend.graph_begins.get(), pieces);
    assert_eq!(backend.graph_replays.get(), 2 * pieces);
    assert_eq!(backend.launches.get(), launches);
    // Every rank's graphs were captured before any launched, and both the
    // first run and the replay issue every rank's first piece before any
    // rank's second, the same graphs on the same streams.
    let replayed = backend.graph_launches.borrow()[captured.len()..].to_vec();
    assert_eq!(replayed, captured);
    let streams = captured
        .iter()
        .map(|&(stream, _)| stream)
        .collect::<Vec<_>>();
    assert!(
        streams
            .chunks(RANKS as usize)
            .all(|piece| piece == &streams[..RANKS as usize])
    );
    // The graph saved the decode item's entries; cancel copies them back.
    let copies_before_cancel = backend.copies.borrow().len();
    instance.cancel().expect("cancel decode ring state");
    let restored = &backend.copies.borrow()[copies_before_cancel..];
    assert_eq!(restored.len(), (RANKS as u64 * SWA_LAYERS * 2) as usize);
    assert!(
        restored
            .iter()
            .all(|&(source, target, _)| source == target + RING_STATE_BYTES_PER_RANK / 2)
    );
    instance.retire_request(41).expect("retire decode slot");
    instance.retire_request(42).expect("retire prompt slot");
    for (index, live) in [2_u32, 3, 8, 16, 32, 4].into_iter().enumerate() {
        let ids = (0..live)
            .map(|row| 1_000 + index as u64 * 100 + u64::from(row))
            .collect::<Vec<_>>();
        let batch = BatchRequestV1::new(
            BatchWireVersionV1::V2,
            ids.iter()
                .map(|&id| BatchItemV1::new(id, true, 1, true, vec![7]))
                .collect(),
            (0..live)
                .map(|page| BatchAppendV1::new(0, 1, 1, vec![u64::from(page)], None))
                .collect(),
        );
        let prior_launches = backend.launches.get();
        instance
            .execute(&batch.encode().unwrap(), &mut ViewSink::default())
            .expect("decode bucket step");
        assert_eq!(
            backend.graph_begins.get(),
            if live == 4 {
                6 * pieces
            } else {
                (index as u32 + 2) * pieces
            }
        );
        if live == 4 {
            assert_eq!(
                backend.launches.get(),
                prior_launches,
                "bucket-four graph replay"
            );
        }
        instance.commit().expect("commit decode bucket");
        for id in ids {
            instance
                .retire_request(id)
                .expect("retire decode bucket slot");
        }
    }
    assert_eq!(backend.graph_replays.get(), 8 * pieces);
    let ordinary = BatchRequestV1::new(
        BatchWireVersionV1::V2,
        vec![BatchItemV1::new(9_999, false, 1, true, vec![7])],
        vec![BatchAppendV1::new(0, 1, 1, vec![1], None)],
    );
    let prior_symbols = backend.launch_symbols.borrow().len();
    let mut ordinary_sink = ViewSink::default();
    instance
        .execute(&ordinary.encode().unwrap(), &mut ordinary_sink)
        .expect("ordinary prefill");
    assert_eq!(ordinary_sink.0.len(), 1);
    assert_eq!(
        ordinary_sink.0[0].raw().element_type,
        BUNDLE_LOGITS_ELEMENT_V1_BF16
    );
    let launched = backend.launch_symbols.borrow()[prior_symbols..].to_vec();
    let count = |name: &str| launched.iter().filter(|symbol| *symbol == name).count();
    // One logits row: gather it, then run the one-row decode head on it
    // rather than the prefill head over every prompt token.
    assert_eq!(count("chop_row_embedding_full_v1"), 2 * RANKS as usize);
    assert_eq!(count("chop_gemm_mimo_head_m1_v1"), RANKS as usize);
    assert_eq!(count("chop_gemm_mimo_head_prefill_v1"), 0);
    assert_eq!(count("chop_gemm_mimo_head_f32_prefill_v1"), 0);
    assert_eq!(count("chop_comm_gather_rows_bf16_v1"), RANKS as usize);
    instance.commit().expect("commit ordinary prefill");
    // A chunk that needs no logits runs no head and gathers nothing.
    let chunk = BatchRequestV1::new(
        BatchWireVersionV1::V2,
        vec![BatchItemV1::new(9_998, false, 2, false, vec![7, 8])],
        vec![BatchAppendV1::new(0, 2, 2, vec![2], None)],
    );
    let prior_symbols = backend.launch_symbols.borrow().len();
    let mut chunk_sink = ViewSink::default();
    instance
        .execute(&chunk.encode().unwrap(), &mut chunk_sink)
        .expect("intermediate prefill chunk");
    assert!(chunk_sink.0.is_empty());
    let launched = backend.launch_symbols.borrow()[prior_symbols..].to_vec();
    assert!(!launched.iter().any(|symbol| symbol.contains("_head")));
    assert!(!launched.iter().any(|symbol| symbol.contains("gather_rows")));
    assert_eq!(
        launched
            .iter()
            .filter(|symbol| *symbol == "chop_row_embedding_full_v1")
            .count(),
        RANKS as usize
    );
    instance.commit().expect("commit intermediate chunk");
    instance
        .retire_request(9_998)
        .expect("retire intermediate chunk");
    instance
        .retire_request(9_999)
        .expect("retire ordinary prefill");
    instance.destroy().expect("destroy fake-host instance");
}
#[test]
fn f32_logits_give_prefill_and_decode_rows_the_f32_head() {
    use runtime_bundle_sdk::{BatchAppendV1, BatchItemV1, BatchRequestV1, BatchWireVersionV1};
    let api = BundleApiV1::new_with_prefix_cache(crate::BUNDLE_MIMO26_EXTENDED_API_V1)
        .expect("bundle API");
    let backend = Rc::new(FakeHost::default());
    let table = HostServicesTableV1::new(backend.clone());
    let mut instance = api
        .create(table, &configuration())
        .expect("fake-host create");
    crate::instance::force_f32_logits(instance.handle(), true);
    let mut step = |batch: BatchRequestV1| {
        let before = backend.launch_symbols.borrow().len();
        let mut sink = ViewSink::default();
        instance
            .execute(&batch.encode().unwrap(), &mut sink)
            .expect("step");
        instance.commit().expect("commit step");
        let launched = backend.launch_symbols.borrow()[before..].to_vec();
        let types = sink
            .0
            .iter()
            .map(|view| view.raw().element_type)
            .collect::<Vec<_>>();
        (launched, types)
    };
    let count =
        |launched: &[String], name: &str| launched.iter().filter(|symbol| *symbol == name).count();
    let f32 = runtime_bundle_abi::BUNDLE_LOGITS_ELEMENT_V1_F32;
    // An unscored prefill row takes the F32 prefill head, as a scored one does.
    let (launched, types) = step(BatchRequestV1::new(
        BatchWireVersionV1::V2,
        vec![BatchItemV1::new(41, false, 1, true, vec![7])],
        vec![BatchAppendV1::new(0, 1, 1, vec![1], None)],
    ));
    assert_eq!(types, [f32]);
    assert_eq!(
        count(&launched, "chop_gemm_mimo_head_f32_prefill_v1"),
        RANKS as usize
    );
    assert_eq!(count(&launched, "chop_gemm_mimo_head_m1_v1"), 0);
    // A decode row takes the F32 head of its bucket and exchanges F32 rows.
    let (launched, types) = step(BatchRequestV1::new(
        BatchWireVersionV1::V2,
        vec![BatchItemV1::new(41, true, 1, true, vec![10])],
        vec![BatchAppendV1::new(1, 1, 2, vec![1], None)],
    ));
    assert_eq!(types, [f32]);
    assert_eq!(
        count(&launched, "chop_gemm_mimo_head_f32_m1_v1"),
        RANKS as usize
    );
    assert_eq!(count(&launched, "chop_gemm_mimo_head_m1_v1"), 0);
    assert!(
        launched
            .iter()
            .any(|symbol| symbol.contains("exchange_rows_f32"))
    );
    instance.retire_request(41).expect("retire request");
    instance.destroy().expect("destroy fake-host instance");
}

#[test]
fn row_owned_logits_exchange_decode_rows_and_view_them_on_their_ranks() {
    use runtime_bundle_sdk::{BatchAppendV1, BatchItemV1, BatchRequestV1, BatchWireVersionV1};
    let api = BundleApiV1::new_with_prefix_cache(crate::BUNDLE_MIMO26_EXTENDED_API_V1)
        .expect("bundle API");
    let backend = Rc::new(FakeHost::default());
    // Verify inputs exist only when the host enables verify steps.
    backend.speculation.set(Some((7, 32)));
    let table = HostServicesTableV1::new(backend.clone());
    // `ROW_OWNED_SPECULATION=0` keeps verify rows on the rank-0 gather.
    crate::instance::ROW_OWNED_LOGITS_FOR_TEST.with(|enabled| enabled.set(Some(true)));
    crate::instance::ROW_OWNED_SPECULATION_FOR_TEST.with(|enabled| enabled.set(Some(false)));
    let created = api.create(table, &configuration());
    crate::instance::ROW_OWNED_SPECULATION_FOR_TEST.with(|enabled| enabled.set(None));
    crate::instance::ROW_OWNED_LOGITS_FOR_TEST.with(|enabled| enabled.set(None));
    let mut instance = created.expect("fake-host create");
    crate::instance::force_f32_logits(instance.handle(), true);
    let mut step = |decode: bool, starts: u64| {
        let before = backend.launch_symbols.borrow().len();
        let syncs = backend.stream_syncs.get();
        let items = (0..5)
            .map(|index| BatchItemV1::new(41 + index, decode, 1, true, vec![7]))
            .collect();
        let appends = (0..5)
            .map(|index| BatchAppendV1::new(starts, 1, starts + 1, vec![index + 1], None))
            .collect();
        let mut sink = ViewSink::default();
        instance
            .execute(
                &BatchRequestV1::new(BatchWireVersionV1::V2, items, appends)
                    .encode()
                    .unwrap(),
                &mut sink,
            )
            .expect("step");
        instance.commit().expect("commit step");
        let launched = backend.launch_symbols.borrow()[before..].to_vec();
        let views = sink
            .0
            .iter()
            .map(|view| {
                let raw = view.raw();
                (raw.rank, raw.first_item_index, raw.row_count)
            })
            .collect::<Vec<_>>();
        (launched, views, backend.stream_syncs.get() - syncs)
    };
    let count =
        |launched: &[String], name: &str| launched.iter().filter(|symbol| *symbol == name).count();
    // Prefill keeps the rank-0 gather and one rank-0 view.
    let (launched, views, _) = step(false, 0);
    assert_eq!(count(&launched, "chop_comm_exchange_rows_f32_v1"), 0);
    assert_eq!(
        count(&launched, "chop_comm_gather_rows_f32_v1"),
        RANKS as usize
    );
    assert_eq!(views, [(0, 0, 5)]);
    // Five decode rows fill a bucket of eight: blocks of two rows on ranks
    // 0, 1 and 2, each exchanged by one launch per rank and viewed on its
    // owner, and every rank's stream is synchronized.
    let (launched, views, syncs) = step(true, 1);
    assert_eq!(
        count(&launched, "chop_comm_exchange_rows_f32_v1"),
        RANKS as usize
    );
    assert_eq!(count(&launched, "chop_comm_gather_rows_f32_v1"), 0);
    assert_eq!(views, [(0, 0, 2), (1, 2, 2), (2, 4, 1)]);
    assert!(syncs >= RANKS);
    // Without row-owned speculation, verify rows keep the rank-0 gather.
    let symbols = crate::instance::plan_stage_probe(
        instance.handle(),
        crate::selection::Phase::Verify(6),
        crate::program::Stage::GatherLogits,
    )
    .expect("verify gather");
    assert!(symbols.iter().all(|symbol| symbol.contains("gather_rows")));
    assert!(!symbols.is_empty());
    for id in 41..46 {
        instance.retire_request(id).expect("retire request");
    }
    instance.destroy().expect("destroy fake-host instance");
}

/// Runs `create` with `ROW_OWNED_LOGITS` and `ROW_OWNED_SPECULATION` set for
/// the instances it creates.
fn with_row_owned_speculation<T>(create: impl FnOnce() -> T) -> T {
    crate::instance::ROW_OWNED_LOGITS_FOR_TEST.with(|enabled| enabled.set(Some(true)));
    crate::instance::ROW_OWNED_SPECULATION_FOR_TEST.with(|enabled| enabled.set(Some(true)));
    let created = create();
    crate::instance::ROW_OWNED_SPECULATION_FOR_TEST.with(|enabled| enabled.set(None));
    crate::instance::ROW_OWNED_LOGITS_FOR_TEST.with(|enabled| enabled.set(None));
    created
}

#[test]
fn row_owned_speculation_exchanges_verify_and_dflash_decode_rows() {
    // DFlash decode and verify rows, narrow and wide, are exchanged; an
    // MTP instance keeps the rank-0 gather.
    for dflash in [true, false] {
        let api = BundleApiV1::new_with_prefix_cache(crate::BUNDLE_MIMO26_EXTENDED_API_V1)
            .expect("bundle API");
        let host = Rc::new(FakeHost::default());
        host.speculation.set(Some((7, 256)));
        host.dflash.set(dflash);
        let mut instance = with_row_owned_speculation(|| {
            api.create(
                HostServicesTableV1::new(Rc::clone(&host)),
                &configuration_with_mtp(true),
            )
        })
        .expect("fake-host create");
        for phase in [
            crate::selection::Phase::Decode(5),
            crate::selection::Phase::Verify(6),
            crate::selection::Phase::Verify(40),
            crate::selection::Phase::Verify(256),
        ] {
            let symbols = crate::instance::plan_stage_probe(
                instance.handle(),
                phase,
                crate::program::Stage::GatherLogits,
            )
            .expect("gather logits");
            let exchanged = symbols
                .iter()
                .any(|symbol| symbol.contains("exchange_rows"));
            let gathered = symbols.iter().any(|symbol| symbol.contains("gather_rows"));
            assert_eq!(
                (exchanged, gathered),
                (dflash, !dflash),
                "dflash={dflash} {phase:?}"
            );
        }
        instance.destroy().expect("destroy fake-host instance");
    }
    // A 256-row verify step (host drafts) exchanges its rows and views each
    // rank's block of 64 rows on that rank.
    let (backend, mut instance) = with_row_owned_speculation(|| wide_verify_instance((7, 256), 32));
    let before = backend.launch_symbols.borrow().len();
    let mut sink = ViewSink::default();
    instance
        .execute(&wide_verify_step(32, 8), &mut sink)
        .expect("256-row verify step");
    let launched = backend.launch_symbols.borrow()[before..].to_vec();
    assert_eq!(
        launched
            .iter()
            .filter(|symbol| symbol.contains("chop_comm_exchange_rows_"))
            .count(),
        RANKS as usize
    );
    assert!(!launched.iter().any(|symbol| symbol.contains("gather_rows")));
    let views = sink
        .0
        .iter()
        .map(|view| {
            let raw = view.raw();
            (raw.rank, raw.first_item_index, raw.row_count)
        })
        .collect::<Vec<_>>();
    assert_eq!(views, [(0, 0, 64), (1, 64, 64), (2, 128, 64), (3, 192, 64)]);
    instance.cancel().expect("cancel");
    instance.destroy().expect("destroy");
}

#[test]
fn row_owned_logits_leave_drafter_steps_on_the_rank_zero_gather() {
    let api = BundleApiV1::new_with_prefix_cache(crate::BUNDLE_MIMO26_EXTENDED_API_V1)
        .expect("bundle API");
    let host = Rc::new(FakeHost::default());
    host.speculation.set(Some((7, 32)));
    crate::instance::ROW_OWNED_LOGITS_FOR_TEST.with(|enabled| enabled.set(Some(true)));
    let created = api.create(
        HostServicesTableV1::new(Rc::clone(&host)),
        &configuration_with_mtp(true),
    );
    crate::instance::ROW_OWNED_LOGITS_FOR_TEST.with(|enabled| enabled.set(None));
    let mut instance = created.expect("fake-host create");
    for phase in [
        crate::selection::Phase::Decode(5),
        crate::selection::Phase::Verify(6),
    ] {
        let symbols = crate::instance::plan_stage_probe(
            instance.handle(),
            phase,
            crate::program::Stage::GatherLogits,
        )
        .expect("gather logits");
        assert!(
            symbols.iter().any(|symbol| symbol.contains("gather_rows"))
                && !symbols
                    .iter()
                    .any(|symbol| symbol.contains("exchange_rows")),
            "{phase:?} {symbols:?}"
        );
    }
    instance.destroy().expect("destroy fake-host instance");
}

#[test]
fn verify_rows_use_multi_query_attention_and_decode_capacity_projections() {
    let api = BundleApiV1::new_with_prefix_cache(crate::BUNDLE_MIMO26_EXTENDED_API_V1)
        .expect("bundle API");
    let backend = Rc::new(FakeHost::default());
    // Verify inputs exist only when the host enables verify steps.
    backend.speculation.set(Some((7, 32)));
    let table = HostServicesTableV1::new(backend);
    let mut instance = api
        .create(table, &configuration())
        .expect("fake-host create");
    let symbols = |phase| {
        let mut symbols = Vec::new();
        for stage in crate::program::stages() {
            symbols.extend(
                crate::instance::plan_stage_probe(instance.handle(), phase, stage)
                    .unwrap_or_else(|status| panic!("{stage:?}: {status:?}")),
            );
        }
        symbols
    };
    let count = |symbols: &[String], name: &str| {
        symbols
            .iter()
            .filter(|symbol| symbol.contains(name))
            .count()
    };
    // Six rows use the bucket-eight dense kernels and the fused decode
    // route, never the prefill GEMMs or the permute kernels. They are exact
    // rows: the experts, the all-reduce norm and attention take the one-row
    // order (`selection::EXACT_VERIFY_ROWS`).
    let exact = symbols(crate::selection::Phase::Verify(6));
    let count_exact = |name: &str| count(&exact, name);
    assert!(count_exact("chop_gemm_mimo_qkv_swa_m8_v1") > 0);
    assert_eq!(count_exact("_prefill_v1"), 0);
    assert_eq!(count_exact("permute_"), 0);
    let moe_layers = crate::program::LAYERS as usize - 1;
    assert_eq!(
        count_exact(model_mimo26_contract::sm120::moe::FUSED_ROUTER_PREFIX),
        moe_layers
    );
    assert_eq!(count_exact("chop_row_combine_mimo4096_v1"), 0);
    assert_eq!(
        count_exact(crate::sm120::comm::VERIFY_COMBINE_SYMBOL),
        moe_layers
    );
    assert_eq!(
        count_exact(crate::sm120::comm::VERIFY_SYMBOL),
        crate::program::LAYERS as usize + 1
    );
    assert_eq!(
        count_exact("chop_moe_mimo_gateup_exact_mxfp4_a16_t8_v1"),
        moe_layers
    );
    // Down's decode entry already has the one-token order.
    assert_eq!(
        count_exact("chop_moe_mimo_down_decode_mxfp4_a16_t8_v1"),
        moe_layers
    );
    assert_eq!(count_exact("_gateup_decode_mxfp4_a16_"), 0);
    assert_eq!(
        count_exact("chop_attn_mimo_swa_decode_rows_v1"),
        SWA_LAYERS as usize
    );
    // Global attention runs the row-group entries (`selection::EXACT_ATTENTION`).
    assert_eq!(
        count_exact("chop_attn_mimo_ga_bf16_decode_rows_shared_v1"),
        GA_LAYERS as usize
    );
    assert_eq!(count_exact("chop_attn_mimo_ga_bf16_decode_rows_v1"), 0);
    assert_eq!(
        count_exact("chop_attn_mimo_ga_bf16_decode_rows_merge_v1"),
        GA_LAYERS as usize
    );
    assert_eq!(count_exact("chop_attn_mimo_swa_verify"), 0);
    assert_eq!(count_exact("chop_attn_mimo_ga_bf16_verify"), 0);
    assert_eq!(count_exact("_verify_mxfp4_"), 0);
    assert_eq!(count_exact("chop_attn_mimo_swa_prefill"), 0);
    // The row engine still rotates and writes the rows ahead of attention,
    // and the sliding-window ring still after it.
    assert_eq!(count_exact("rope_mimo_swa_prefill"), SWA_LAYERS as usize);
    assert_eq!(
        count_exact("chop_row_rope_mimo_ga_bf16"),
        GA_LAYERS as usize
    );
    assert_eq!(
        count_exact("ring_write_mimo_bf16_pos32"),
        SWA_LAYERS as usize
    );
    assert_eq!(count_exact("chop_mimo_ring_backup_v1"), 1);
    // Above eight rows, verify attention shares keys across query rows.
    let wide = symbols(crate::selection::Phase::Verify(13));
    let count_wide = |name: &str| count(&wide, name);
    assert!(count_wide("chop_attn_mimo_swa_verify_partial") > 0);
    assert!(count_wide("chop_attn_mimo_swa_verify_merge") > 0);
    assert!(count_wide("chop_attn_mimo_ga_bf16_verify") > 0);
    assert_eq!(count_wide("_decode_rows"), 0);
    assert_eq!(count_wide("_exact_"), 0);
    assert_eq!(count_wide(crate::sm120::comm::VERIFY_SYMBOL), 0);
    assert_eq!(count_wide("chop_attn_mimo_swa_prefill"), 0);
    assert_eq!(count_wide("chop_attn_mimo_ga_bf16_prefill"), 0);
    assert_eq!(count_wide("chop_attn_mimo_swa_decode"), 0);
    assert_eq!(count_wide("rope_mimo_swa_prefill"), SWA_LAYERS as usize);
    assert_eq!(
        count_wide("ring_write_mimo_bf16_pos32"),
        SWA_LAYERS as usize
    );
    instance.destroy().expect("destroy fake-host instance");
}

#[test]
fn decode_attention_folds_rope_and_kv_write() {
    use crate::program::Stage;
    let api = BundleApiV1::new_with_prefix_cache(crate::BUNDLE_MIMO26_EXTENDED_API_V1)
        .expect("bundle API");
    let backend = Rc::new(FakeHost::default());
    // Verify inputs exist only when the host enables verify steps.
    backend.speculation.set(Some((7, 32)));
    let table = HostServicesTableV1::new(backend);
    let mut instance = api
        .create(table, &configuration())
        .expect("fake-host create");
    let symbols = |phase| {
        let mut symbols = Vec::new();
        for stage in crate::program::stages() {
            symbols.extend(
                crate::instance::plan_stage_probe(instance.handle(), phase, stage)
                    .unwrap_or_else(|status| panic!("{stage:?}: {status:?}")),
            );
        }
        symbols
    };
    let count = |symbols: &[String], name: &str| {
        symbols
            .iter()
            .filter(|symbol| symbol.contains(name))
            .count()
    };
    // Decode attention rotates Q and writes each row's current K and V, so
    // no RoPE or ring write runs; GA partials are still merged.
    let decode = symbols(crate::selection::Phase::Decode(3));
    assert_eq!(count(&decode, "chop_row_rope_"), 0);
    assert_eq!(count(&decode, "ring_write_mimo"), 0);
    assert_eq!(
        count(&decode, "chop_attn_mimo_swa_decode_rope_table_v1"),
        SWA_LAYERS as usize
    );
    assert_eq!(
        count(&decode, "chop_attn_mimo_ga_bf16_decode_rope_table_v1"),
        GA_LAYERS as usize
    );
    assert_eq!(
        count(&decode, "chop_attn_mimo_ga_bf16_decode_merge_v1"),
        GA_LAYERS as usize
    );
    assert_eq!(count(&decode, "_decode_partial_v1"), 0);
    assert_eq!(count(&decode, "_decode_direct_v1"), 0);
    assert_eq!(count(&decode, "chop_mimo_ring_backup_v1"), 1);
    for layer in [0, 1] {
        assert!(
            crate::instance::stage_reads_qkv(
                instance.handle(),
                crate::selection::Phase::Decode(3),
                Stage::AttentionPartial(layer),
            )
            .expect("folded attention plan")
        );
    }
    // Verify keeps the row engine's RoPE ahead of its attention.
    let verify = symbols(crate::selection::Phase::Verify(6));
    assert_eq!(count(&verify, "_rope_v1"), 0);
    assert_eq!(count(&verify, "_rope_table_v1"), 0);
    assert_eq!(
        count(&verify, "chop_row_rope_mimo_ga_bf16"),
        GA_LAYERS as usize
    );
    assert!(
        !crate::instance::stage_reads_qkv(
            instance.handle(),
            crate::selection::Phase::Verify(6),
            Stage::AttentionPartial(1),
        )
        .expect("verify attention plan")
    );
    instance.destroy().expect("destroy fake-host instance");
}

#[test]
fn verify_combine_folds_every_token_bucket() {
    use crate::program::Stage;
    let (_, mut instance) = verify_instance(Some((7, 32)));
    for rows in [2, 3, 5, 9, 17, 32] {
        let phase = crate::selection::Phase::Verify(rows);
        for stage in [
            Stage::CombineExperts(1),
            Stage::MlpReduceNorm(1),
            Stage::FinalReduceNorm,
        ] {
            let symbols = crate::instance::plan_stage_probe(instance.handle(), phase, stage)
                .expect("verify combine plan");
            if matches!(stage, Stage::CombineExperts(_)) {
                assert!(symbols.is_empty());
            } else {
                // Exact verify rows take the verify combine entry.
                let bucket = crate::graphs::verify_bucket(rows).expect("verify bucket");
                assert_eq!(
                    symbols,
                    [
                        if bucket <= crate::selection::EXACT_VERIFY_ROWS && bucket > 2 {
                            crate::sm120::comm::VERIFY_COMBINE_SYMBOL
                        } else {
                            crate::sm120::launch::comm::push_symbols(RANKS, bucket).1
                        }
                    ]
                );
            }
        }
    }
    instance.destroy().expect("destroy verify instance");
}

#[test]
fn graph_cache_keeps_decode_and_verify_shapes_separate_and_rejects_invalid_keys() {
    use crate::graphs::{GraphCache, GraphKey};
    let backend = Rc::new(FakeHost::default());
    let table = HostServicesTableV1::new(Rc::clone(&backend));
    let stream = BundleStreamHandleV1::from_parts(1, 1);
    let mut graphs =
        GraphCache::with_items(crate::contract::TOPOLOGY.ranks, crate::contract::ITEMS_32);
    for (rank, bucket, items) in [
        (4, 8, 2),
        (0, 0, 0),
        (0, 3, 0),
        (0, 64, 2),
        (0, 64, 0),
        (0, 512, 32),
        (0, 8, 3),
        (0, 8, 16),
    ] {
        let result = graphs.run::<BundleAbiError>(
            table.caller(),
            stream,
            GraphKey {
                rank,
                bucket,
                items,
            },
            96,
            |_| panic!("invalid key must not record"),
        );
        assert!(matches!(
            result,
            Err(BundleAbiError::InvalidRange("graph bucket"))
        ));
    }
    assert_eq!(backend.graph_begins.get(), 0);
    for items in [0, 1, 2, 4, 8] {
        let key = GraphKey {
            rank: 0,
            bucket: 8,
            items,
        };
        graphs
            .run::<BundleAbiError>(table.caller(), stream, key, 96, |_| Ok(()))
            .expect("capture distinct shape");
        graphs
            .run::<BundleAbiError>(table.caller(), stream, key, 0, |_| {
                panic!("cached shape must replay")
            })
            .expect("replay distinct shape");
    }
    assert_eq!(graphs.len(), 5);
    assert_eq!(backend.graph_begins.get(), 15);
    assert_eq!(backend.graph_replays.get(), 30);
    graphs
        .release_all(table.caller())
        .expect("release all pieces");
    assert!(graphs.is_empty());
    assert_eq!(backend.graph_releases.get(), 15);
}

fn kept(request_id: u64, kept_rows: u32) -> runtime_bundle_abi::BundleKeptRowsV1 {
    runtime_bundle_abi::BundleKeptRowsV1 {
        struct_size: std::mem::size_of::<runtime_bundle_abi::BundleKeptRowsV1>() as u32,
        reserved: 0,
        request_id,
        kept_rows,
        reserved_tail: 0,
    }
}

/// A speculating instance with requests 50 and 51 prefilled to three tokens.
fn verify_instance(
    speculation: Option<(u32, u32)>,
) -> (Rc<FakeHost>, runtime_bundle_abi::BundleInstanceV1) {
    use runtime_bundle_sdk::{BatchAppendV1, BatchItemV1, BatchRequestV1, BatchWireVersionV1};
    let api = BundleApiV1::new_with_prefix_cache(crate::BUNDLE_MIMO26_EXTENDED_API_V1)
        .expect("bundle API");
    assert!(api.supports_speculation());
    let backend = Rc::new(FakeHost::default());
    backend.speculation.set(speculation);
    let table = HostServicesTableV1::new(Rc::clone(&backend));
    let mut instance = api
        .create(table, &configuration())
        .expect("fake-host create");
    let prefill = BatchRequestV1::new(
        BatchWireVersionV1::V3,
        vec![
            BatchItemV1::new(50, false, 3, true, vec![1, 2, 3]),
            BatchItemV1::new(51, false, 3, true, vec![4, 5, 6]),
        ],
        vec![
            BatchAppendV1::new(0, 3, 3, vec![1], None),
            BatchAppendV1::new(0, 3, 3, vec![2], None),
        ],
    );
    instance
        .execute(&prefill.encode().unwrap(), &mut ViewSink::default())
        .expect("prefill");
    instance.commit().expect("commit prefill");
    (backend, instance)
}

fn verify_step(previous: u64, rows: u32) -> Vec<u8> {
    use runtime_bundle_sdk::{BatchAppendV1, BatchItemV1, BatchRequestV1, BatchWireVersionV1};
    BatchRequestV1::new(
        BatchWireVersionV1::V3,
        vec![
            BatchItemV1::verify(50, rows, false, (0..rows).map(|token| 10 + token).collect()),
            BatchItemV1::new(51, true, 1, true, vec![7]),
        ],
        vec![
            BatchAppendV1::new(
                previous,
                u64::from(rows),
                previous + u64::from(rows),
                vec![1],
                None,
            ),
            BatchAppendV1::new(3, 1, 4, vec![2], None),
        ],
    )
    .encode()
    .unwrap()
}

#[test]
fn a_verify_step_emits_every_row_and_restores_the_rejected_ring_tail() {
    let (backend, mut instance) = verify_instance(Some((7, 32)));
    let graphs_before = backend.graph_begins.get();
    let symbols_before = backend.launch_symbols.borrow().len();
    let mut sink = ViewSink::default();
    instance
        .execute(&verify_step(3, 4), &mut sink)
        .expect("verify step");
    // Four verify rows and the decode row are contiguous in slots and in the
    // decode logits buffer; the exchange gives each rank a block of two rows
    // of the eight-row bucket, so they share one view per owner.
    assert_eq!(
        sink.0
            .iter()
            .map(|view| (view.raw().first_item_index, view.raw().row_count))
            .collect::<Vec<_>>(),
        [(0, 2), (2, 2), (4, 1)]
    );
    // Five rows over two items capture the (8, 2) verify graph on each rank.
    assert_eq!(backend.graph_begins.get(), graphs_before + 3 * RANKS);
    let launched = backend.launch_symbols.borrow()[symbols_before..].to_vec();
    let count = |name: &str| {
        launched
            .iter()
            .filter(|symbol| symbol.contains(name))
            .count()
    };
    assert_eq!(count("chop_mimo_ring_backup_v1"), RANKS as usize);
    assert_eq!(
        count("ring_write_mimo_bf16_pos32"),
        SWA_LAYERS as usize * RANKS as usize
    );
    assert_eq!(count("chop_gemm_mimo_head_f32_m8_v1"), RANKS as usize);

    let copies_before = backend.host_copies.borrow().len();
    let symbols_before = backend.launch_symbols.borrow().len();
    instance
        .commit_accepted(&[kept(50, 2)])
        .expect("keep two of four rows");
    // One restore per rank, reading one range: slot 0, positions 5 and 6.
    let uploads = backend.host_copies.borrow()[copies_before..].to_vec();
    assert_eq!(uploads.len(), RANKS as usize);
    for (length, bytes) in &uploads {
        assert_eq!(*length, 512);
        let word =
            |offset: usize| u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
        assert_eq!((word(0), word(128), word(256), word(384)), (1, 0, 5, 2));
    }
    assert_eq!(
        backend.launch_symbols.borrow()[symbols_before..],
        vec!["chop_mimo_ring_restore_v1".to_string(); RANKS as usize]
    );
    // The next step resumes at the kept end, and every row kept restores
    // nothing.
    assert!(
        instance
            .execute(&verify_step(7, 2), &mut ViewSink::default())
            .is_err()
    );
    let (backend, mut instance) = verify_instance(Some((7, 32)));
    instance
        .execute(&verify_step(3, 4), &mut ViewSink::default())
        .expect("verify step");
    instance.commit_accepted(&[kept(50, 2)]).expect("keep two");
    let decode_resume = {
        use runtime_bundle_sdk::{BatchAppendV1, BatchItemV1, BatchRequestV1, BatchWireVersionV1};
        BatchRequestV1::new(
            BatchWireVersionV1::V3,
            vec![BatchItemV1::verify(50, 2, false, vec![1, 2])],
            vec![BatchAppendV1::new(5, 2, 7, vec![1], None)],
        )
        .encode()
        .unwrap()
    };
    instance
        .execute(&decode_resume, &mut ViewSink::default())
        .expect("resume at the kept end");
    // A same-shape verify step replays the graph it captured.
    let captured = backend.graph_launches.borrow()[3 * RANKS as usize..].to_vec();
    let replays = backend.graph_replays.get();
    let begins = backend.graph_begins.get();
    let launches = backend.launches.get();
    instance.commit_accepted(&[kept(50, 1)]).expect("keep one");
    instance
        .execute(
            &{
                use runtime_bundle_sdk::{
                    BatchAppendV1, BatchItemV1, BatchRequestV1, BatchWireVersionV1,
                };
                BatchRequestV1::new(
                    BatchWireVersionV1::V3,
                    vec![BatchItemV1::verify(50, 2, false, vec![3, 4])],
                    vec![BatchAppendV1::new(6, 2, 8, vec![1], None)],
                )
                .encode()
                .unwrap()
            },
            &mut ViewSink::default(),
        )
        .expect("same-shape verify step");
    assert_eq!(backend.graph_begins.get(), begins);
    assert_eq!(backend.graph_replays.get(), replays + 3 * RANKS);
    let replayed = backend.graph_launches.borrow()[replays as usize..].to_vec();
    assert_eq!(replayed, captured);
    // The restore of the previous step's rejected row is the only launch.
    assert_eq!(backend.launches.get(), launches + RANKS);
    // Keeping every row restores nothing.
    let symbols_before = backend.launch_symbols.borrow().len();
    instance.commit_accepted(&[kept(50, 2)]).expect("keep all");
    assert_eq!(backend.launch_symbols.borrow().len(), symbols_before);
}

/// A speculating instance with `items` requests from 100 prefilled to
/// three tokens each.
fn wide_verify_instance(
    speculation: (u32, u32),
    items: u64,
) -> (Rc<FakeHost>, runtime_bundle_abi::BundleInstanceV1) {
    wide_verify_instance_reporting(speculation, items, None)
}

/// `wide_verify_instance` on a host whose driver reports `residency` for
/// every cluster residency query.
fn wide_verify_instance_reporting(
    speculation: (u32, u32),
    items: u64,
    residency: Option<runtime_bundle_abi::BundleRawClusterResidencyV1>,
) -> (Rc<FakeHost>, runtime_bundle_abi::BundleInstanceV1) {
    let backend = Rc::new(FakeHost::default());
    backend.residency.set(residency);
    wide_verify_instance_on(backend, speculation, items)
}

/// `wide_verify_instance` on `backend`.
fn wide_verify_instance_on(
    backend: Rc<FakeHost>,
    speculation: (u32, u32),
    items: u64,
) -> (Rc<FakeHost>, runtime_bundle_abi::BundleInstanceV1) {
    use runtime_bundle_sdk::{BatchAppendV1, BatchItemV1, BatchRequestV1, BatchWireVersionV1};
    let api = BundleApiV1::new_with_prefix_cache(crate::BUNDLE_MIMO26_EXTENDED_API_V1)
        .expect("bundle API");
    backend.speculation.set(Some(speculation));
    let table = HostServicesTableV1::new(Rc::clone(&backend));
    let mut instance = api
        .create(table, &configuration())
        .expect("fake-host create");
    let prefill = BatchRequestV1::new(
        BatchWireVersionV1::V3,
        (0..items)
            .map(|item| BatchItemV1::new(100 + item, false, 3, true, vec![1, 2, 3]))
            .collect(),
        (0..items)
            .map(|item| BatchAppendV1::new(0, 3, 3, vec![item], None))
            .collect(),
    );
    instance
        .execute(&prefill.encode().unwrap(), &mut ViewSink::default())
        .expect("prefill");
    instance.commit().expect("commit prefill");
    (backend, instance)
}

/// Every one of `items` requests verifies `rows` rows after its prefill.
fn wide_verify_step(items: u64, rows: u32) -> Vec<u8> {
    use runtime_bundle_sdk::{BatchAppendV1, BatchItemV1, BatchRequestV1, BatchWireVersionV1};
    BatchRequestV1::new(
        BatchWireVersionV1::V3,
        (0..items)
            .map(|item| {
                BatchItemV1::verify(100 + item, rows, false, (0..rows).map(|t| 10 + t).collect())
            })
            .collect(),
        (0..items)
            .map(|item| {
                BatchAppendV1::new(3, u64::from(rows), 3 + u64::from(rows), vec![item], None)
            })
            .collect(),
    )
    .encode()
    .unwrap()
}

#[test]
fn verify_steps_above_the_decode_batch_run_prefill_kernels_in_their_bucket_graph() {
    // At 256 rows the dense projections run the prefill entries (medium-M
    // entries stop at 64 rows).
    let (backend, mut instance) = wide_verify_instance((7, 256), 32);
    let graphs_before = backend.graph_begins.get();
    let symbols_before = backend.launch_symbols.borrow().len();
    let mut sink = ViewSink::default();
    instance
        .execute(&wide_verify_step(32, 8), &mut sink)
        .expect("256-row verify step");
    // Every row of every item emits logits.
    assert_eq!(
        sink.0.iter().map(|view| view.raw().row_count).sum::<u32>(),
        256
    );
    // The (256, 32) verify graph is captured once per rank.
    assert_eq!(backend.graph_begins.get(), graphs_before + 3 * RANKS);
    let launched = backend.launch_symbols.borrow()[symbols_before..].to_vec();
    let count = |name: &str| {
        launched
            .iter()
            .filter(|symbol| symbol.contains(name))
            .count()
    };
    let ranks = RANKS as usize;
    let moe_layers = crate::program::LAYERS as usize - 1;
    // Token-indexed stages run the prefill kernels (GEMMs, the tiled route,
    // the combine row); the hidden all-reduces take the plain wide entry,
    // unless CHOPINFER_MIMO_WIDE_COMM=0 keeps the phased stages.
    let wide = std::env::var(crate::switches::WIDE_COMM).as_deref() != Ok("0");
    assert_eq!(
        count("chop_gemm_mimo_qkv_swa_prefill_v1"),
        SWA_LAYERS as usize * ranks
    );
    assert_eq!(count("chop_gemm_mid_"), 0);
    // The heads keep the prefill entry above 64 rows.
    assert_eq!(count("chop_gemm_mimo_head_f32_prefill_v1"), ranks);
    assert_eq!(count("_m32_v1"), 0);
    assert_eq!(count("permute_tile_scatter"), moe_layers * ranks);
    assert_eq!(
        count(model_mimo26_contract::sm120::moe::FUSED_ROUTER_PREFIX),
        0
    );
    assert_eq!(count("chop_row_combine_mimo4096_v1"), moe_layers * ranks);
    for rows in [1, 32] {
        let (plain, combine) = crate::sm120::launch::comm::push_symbols(RANKS, rows);
        assert_eq!(count(combine), 0);
        assert_eq!(count(plain), 0);
    }
    assert_eq!(
        count(crate::sm120::launch::comm::PHASED_SYMBOLS[0]),
        count(crate::sm120::launch::comm::PHASED_SYMBOLS[1])
    );
    assert_eq!(
        count(crate::sm120::launch::comm::PHASED_SYMBOLS[0]) > 0,
        !wide
    );
    assert_eq!(count(crate::sm120::comm::WIDE_COMBINE_SYMBOL), 0);
    assert_eq!(count(crate::sm120::comm::WIDE_SYMBOL) > 0, wide);
    // Verify attention and the ring's backup still cover the items.
    assert!(count("chop_attn_mimo_swa_verify_partial") > 0);
    assert_eq!(count("chop_attn_mimo_swa_prefill"), 0);
    assert_eq!(count("chop_mimo_ring_backup_v1"), ranks);
    let keep = (0..32)
        .map(|item| kept(100 + item, (item % 8 + 1) as u32))
        .collect::<Vec<_>>();
    instance.commit_accepted(&keep).expect("keep 1 to 8 rows");

    // 40 rows over five items take the (64, 8) graph; a same-bucket step
    // with fewer rows replays it.
    let (backend, mut instance) = wide_verify_instance((7, 256), 5);
    instance
        .execute(&wide_verify_step(5, 8), &mut ViewSink::default())
        .expect("40-row verify step");
    instance
        .commit_accepted(&(0..5).map(|item| kept(100 + item, 1)).collect::<Vec<_>>())
        .expect("keep one row each");
    let begins = backend.graph_begins.get();
    let replays = backend.graph_replays.get();
    let step = {
        use runtime_bundle_sdk::{BatchAppendV1, BatchItemV1, BatchRequestV1, BatchWireVersionV1};
        BatchRequestV1::new(
            BatchWireVersionV1::V3,
            (0..5)
                .map(|item| BatchItemV1::verify(100 + item, 7, false, vec![3; 7]))
                .collect(),
            (0..5)
                .map(|item| BatchAppendV1::new(4, 7, 11, vec![item], None))
                .collect(),
        )
        .encode()
        .unwrap()
    };
    instance
        .execute(&step, &mut ViewSink::default())
        .expect("35-row verify step");
    assert_eq!(backend.graph_begins.get(), begins);
    assert_eq!(backend.graph_replays.get(), replays + 3 * RANKS);

    // An instance enabled for 32 rows refuses the wide step.
    let (_, mut narrow) = wide_verify_instance((7, 32), 32);
    assert!(
        narrow
            .execute(&wide_verify_step(32, 8), &mut ViewSink::default())
            .is_err()
    );
}

/// A device report of `clusters` resident 8-CTA clusters on all 188 SMs, from
/// a device that accepts cooperative launches.
#[test]
fn a_device_short_of_wide_clusters_routes_wide_graphs_to_the_phased_stages() {
    // The wide entry's 72 CTAs are nine clusters; the phased stages' 64 are
    // eight and the push entries' 40 five, so only the wide entry is short.
    let (backend, mut instance) =
        wide_verify_instance_reporting((7, 256), 32, Some(residency(8, 0)));
    // Every rank asks for each entry, except that the wide entry, once off
    // (or off by CHOPINFER_MIMO_WIDE_COMM=0), is asked about no further.
    let queries = backend.residency_queries.borrow().clone();
    let wide_env = std::env::var(crate::switches::WIDE_COMM).as_deref() != Ok("0");
    for (symbol, grid, asked) in [
        (crate::sm120::comm::WIDE_SYMBOL, 72, usize::from(wide_env)),
        (
            crate::sm120::launch::comm::PHASED_SYMBOLS[0],
            64,
            RANKS as usize,
        ),
        (
            crate::sm120::launch::comm::PHASED_SYMBOLS[1],
            64,
            RANKS as usize,
        ),
        (
            crate::sm120::launch::comm::push_symbols(RANKS, 32).0,
            40,
            RANKS as usize,
        ),
    ] {
        // At its largest grid (the push symbols also at their low-latency
        // shape).
        assert_eq!(
            queries
                .iter()
                .filter(|query| query.0 == symbol && query.1 == grid)
                .count(),
            asked,
            "{symbol}"
        );
        assert!(
            queries
                .iter()
                .filter(|query| query.0 == symbol)
                .all(|query| query.1 <= grid)
        );
    }
    assert_eq!(
        backend
            .logs
            .borrow()
            .iter()
            .any(|line| line.contains("wide all-reduce off, graphs take the phased stages")),
        wide_env
    );
    let before = backend.launch_symbols.borrow().len();
    instance
        .execute(&wide_verify_step(32, 8), &mut ViewSink::default())
        .expect("256-row verify step");
    let launched = backend.launch_symbols.borrow()[before..].to_vec();
    let count = |name: &str| {
        launched
            .iter()
            .filter(|symbol| symbol.contains(name))
            .count()
    };
    assert_eq!(count(crate::sm120::comm::WIDE_SYMBOL), 0);
    assert!(count(crate::sm120::launch::comm::PHASED_SYMBOLS[0]) > 0);
    assert_eq!(
        count(crate::sm120::launch::comm::PHASED_SYMBOLS[0]),
        count(crate::sm120::launch::comm::PHASED_SYMBOLS[1])
    );
}

/// Creates a TP4 instance with one rank per device and 256-row verify
/// buckets, its host reporting `report`, and returns the host and whether
/// create succeeded.
fn create_reporting(
    report: runtime_bundle_abi::BundleRawClusterResidencyV1,
) -> (Rc<FakeHost>, bool) {
    let api = BundleApiV1::new_with_prefix_cache(crate::BUNDLE_MIMO26_EXTENDED_API_V1)
        .expect("bundle API");
    let backend = Rc::new(FakeHost::at(Tp::TP4));
    backend.speculation.set(Some((7, 256)));
    backend.residency.set(Some(report));
    let table = HostServicesTableV1::new(Rc::clone(&backend));
    let config = configuration_serving(Tp::TP4, &crate::contract::KV_FORMATS[0], false, 32);
    let created = api.create(table, &config).is_ok();
    (backend, created)
}

#[test]
fn repeated_ordinals_log_why_create_is_refused() {
    for tp in Tp::SUPPORTED {
        let host = Rc::new(FakeHost::at(tp));
        let mut ordinals = (0..tp.ranks() as i32).collect::<Vec<_>>();
        ordinals[1] = ordinals[0];
        let record = configuration_on(tp, host.kv(), false, &ordinals, 32);
        assert_eq!(
            BundleApiV1::new_with_prefix_cache(crate::BUNDLE_MIMO26_EXTENDED_API_V1)
                .unwrap()
                .create(HostServicesTableV1::new(Rc::clone(&host)), &record)
                .err(),
            Some(BundleAbiError::Status(BundleStatusV1::InvalidArgument))
        );
        assert!(
            host.logs
                .borrow()
                .iter()
                .any(|line| line.contains("each rank needs its own device ordinal"))
        );
        assert!(host.journal.borrow().is_empty());
    }
}

/// Every placement creates when residency is what the target device's
/// driver reports for each entry's audited resources
/// (`crate::sm120::device::locked`, from `kernels.lock`).
#[test]
fn every_placement_fits_the_target_device() {
    let model = crate::sm120::device::locked();
    let device_ordinals = (0..4).collect::<Vec<i32>>();
    for (tp, ordinals) in [
        (Tp::TP4, &device_ordinals[..4]),
        (Tp::TP2, &device_ordinals[..2]),
    ] {
        let api = BundleApiV1::new_with_prefix_cache(crate::BUNDLE_MIMO26_EXTENDED_API_V1)
            .expect("bundle API");
        let backend = Rc::new(FakeHost::at(tp));
        backend.speculation.set(Some((7, 256)));
        *backend.modelled.borrow_mut() = Some(model.clone());
        let table = HostServicesTableV1::new(Rc::clone(&backend));
        let config = configuration_on(tp, &crate::contract::KV_FORMATS[0], false, ordinals, 32);
        let created = api.create(table, &config);
        assert!(
            created.is_ok(),
            "{tp:?} on {ordinals:?}: {:?}",
            backend
                .logs
                .borrow()
                .iter()
                .filter(|line| line.contains("residency") || line.contains("failed"))
                .collect::<Vec<_>>()
        );
    }
}

#[test]
fn ranks_require_sufficient_resident_clusters() {
    // CHOPINFER_MIMO_WIDE_COMM=0 keeps the wide entry off unasked.
    let wide_env = std::env::var(crate::switches::WIDE_COMM).as_deref() != Ok("0");
    let wide_off = |backend: &FakeHost| {
        backend
            .logs
            .borrow()
            .iter()
            .any(|line| line.contains("wide all-reduce off, graphs take the phased stages"))
    };
    // Eight clusters hold one rank's push (five) and phased (eight) grids
    // but not its wide grid (nine): the wide all-reduce turns off.
    let (backend, created) = create_reporting(residency(8, 0));
    assert!(created && wide_off(&backend) == wide_env);
    // Distinct devices need no cooperative launch.
    let mut ordinary = residency(46, 0);
    ordinary.flags &= !runtime_bundle_abi::BUNDLE_CLUSTER_RESIDENCY_V1_COOPERATIVE_LAUNCH;
    let (_, created) = create_reporting(ordinary);
    assert!(created);
    // Four clusters hold no push grid (five): create fails rather than
    // leaving calls that could hang.
    let (backend, created) = create_reporting(residency(4, 0));
    assert!(!created);
    assert!(
        backend
            .logs
            .borrow()
            .iter()
            .any(|line| line.contains("create failed") && line.contains("chop_comm_"))
    );
    // An MPS client is refused: multi-rank serving under MPS is unsupported
    // (CHOPINFER_COMM_RESIDENCY=unchecked would override it).
    let (backend, created) = create_reporting(residency(
        46,
        runtime_bundle_abi::BUNDLE_CLUSTER_RESIDENCY_V1_MPS_CLIENT,
    ));
    assert!(!created);
    assert!(
        backend
            .logs
            .borrow()
            .iter()
            .any(|line| line.contains("create failed") && line.contains("MPS"))
    );
    // So is a host that reports no cluster count: nothing is authorized
    // without one.
    let mut unreported = runtime_bundle_abi::BundleRawClusterResidencyV1::unreported();
    unreported.multiprocessor_count = 188;
    let (backend, created) = create_reporting(unreported);
    assert!(!created);
    assert!(backend.logs.borrow().iter().any(|line| {
        line.contains("create failed") && line.contains("CHOPINFER_COMM_RESIDENCY=unchecked")
    }));
}

/// Every push symbol a row count up to 32 can pick is checked: a device short
/// only for the decode combine entry (which small buckets take) fails create.
#[test]
fn a_short_decode_entry_fails_create() {
    let symbol = crate::sm120::comm::DECODE_COMBINE_SYMBOL;
    let api = BundleApiV1::new_with_prefix_cache(crate::BUNDLE_MIMO26_EXTENDED_API_V1)
        .expect("bundle API");
    let backend = Rc::new(FakeHost::default());
    backend
        .residency_by_symbol
        .borrow_mut()
        .insert(symbol.into(), residency(0, 0));
    let table = HostServicesTableV1::new(Rc::clone(&backend));
    assert!(api.create(table, &configuration()).is_err());
    assert!(
        backend
            .logs
            .borrow()
            .iter()
            .any(|line| line.contains("create failed") && line.contains(symbol))
    );
}

/// One rank's shortfall turns the wide entry off for the whole instance, so
/// every rank lowers the same stages.
#[test]
fn a_later_rank_short_of_wide_clusters_turns_it_off_everywhere() {
    if std::env::var(crate::switches::WIDE_COMM).as_deref() == Ok("0") {
        return;
    }
    // Symbol handles are resolved rank by rank, so the wide entry's last
    // handle is rank 3's; a first create learns it.
    let (probe, _) = wide_verify_instance_reporting((7, 256), 1, None);
    let last = *probe
        .symbol_names
        .borrow()
        .iter()
        .filter(|(_, name)| *name == crate::sm120::comm::WIDE_SYMBOL)
        .map(|(slot, _)| slot)
        .max()
        .expect("wide symbol resolved");
    let backend = Rc::new(FakeHost::default());
    backend
        .residency_by_slot
        .borrow_mut()
        .insert(last, residency(8, 0));
    let (backend, mut instance) = wide_verify_instance_on(backend, (7, 256), 32);
    assert!(backend.logs.borrow().iter().any(|line| {
        line.contains("wide all-reduce off, graphs take the phased stages")
            && line.contains("rank 3")
    }));
    let before = backend.launch_symbols.borrow().len();
    instance
        .execute(&wide_verify_step(32, 8), &mut ViewSink::default())
        .expect("256-row verify step");
    let launched = backend.launch_symbols.borrow()[before..].to_vec();
    let count = |name: &str| launched.iter().filter(|symbol| *symbol == name).count();
    assert_eq!(count(crate::sm120::comm::WIDE_SYMBOL), 0);
    let phased = count(crate::sm120::launch::comm::PHASED_SYMBOLS[0]);
    assert!(phased > 0 && phased % RANKS as usize == 0);
}

#[test]
fn a_forced_verify_bucket_pads_narrow_steps_into_the_wide_kernels() {
    let (backend, mut instance) = verify_instance(Some((7, 256)));
    crate::instance::force_verify_bucket(instance.handle(), 256);
    let graphs = backend.graph_begins.get();
    let before = backend.launch_symbols.borrow().len();
    let mut sink = ViewSink::default();
    instance
        .execute(&verify_step(3, 4), &mut sink)
        .expect("padded verify step");
    // Five live rows emit their logits only.
    assert_eq!(
        sink.0.iter().map(|view| view.raw().row_count).sum::<u32>(),
        5
    );
    // They run the (256, 32) graph and its prefill kernels.
    assert_eq!(backend.graph_begins.get(), graphs + 3 * RANKS);
    let launched = backend.launch_symbols.borrow()[before..].to_vec();
    let count = |name: &str| {
        launched
            .iter()
            .filter(|symbol| symbol.contains(name))
            .count()
    };
    assert_eq!(count("chop_gemm_mimo_head_f32_prefill_v1"), RANKS as usize);
    assert_eq!(count("_m8_v1"), 0);
    let wide = std::env::var(crate::switches::WIDE_COMM).as_deref() != Ok("0");
    assert_eq!(count(crate::sm120::comm::WIDE_SYMBOL) > 0, wide);
    assert_eq!(
        count(crate::sm120::launch::comm::PHASED_SYMBOLS[0]) > 0,
        !wide
    );
    instance.commit_accepted(&[kept(50, 2)]).expect("keep two");
}

#[test]
fn verify_commits_refuse_plain_commit_and_out_of_range_kept_rows() {
    let (_, mut plain) = verify_instance(Some((7, 32)));
    plain
        .execute(&verify_step(3, 4), &mut ViewSink::default())
        .expect("verify step");
    assert!(plain.commit().is_err());

    for records in [
        vec![kept(50, 5)],
        vec![kept(51, 1)],
        vec![],
        vec![kept(50, 1), kept(50, 1)],
    ] {
        let (_, mut instance) = verify_instance(Some((7, 32)));
        instance
            .execute(&verify_step(3, 4), &mut ViewSink::default())
            .expect("verify step");
        assert!(instance.commit_accepted(&records).is_err(), "{records:?}");
    }
}

#[test]
fn verify_steps_are_refused_unless_the_host_enabled_them() {
    let (_, mut off) = verify_instance(None);
    assert!(
        off.execute(&verify_step(3, 4), &mut ViewSink::default())
            .is_err()
    );
    // Two drafts allow three rows per item, not four.
    let (_, mut narrow) = verify_instance(Some((2, 32)));
    assert!(
        narrow
            .execute(&verify_step(3, 4), &mut ViewSink::default())
            .is_err()
    );
    narrow
        .execute(&verify_step(3, 3), &mut ViewSink::default())
        .expect("three rows fit");
}

#[test]
fn create_acquires_all_rank_resources_and_destroy_releases_them() {
    use runtime_bundle_sdk::{BatchAppendV1, BatchItemV1, BatchRequestV1, BatchWireVersionV1};

    let api = BundleApiV1::new_with_prefix_cache(crate::BUNDLE_MIMO26_EXTENDED_API_V1)
        .expect("bundle API");
    let backend = Rc::new(FakeHost::default());
    let table = HostServicesTableV1::new(Rc::clone(&backend));
    let mut instance = api
        .create(table, &configuration())
        .expect("fake-host create");
    for phase in [
        crate::selection::Phase::Decode(1),
        crate::selection::Phase::Prefill,
    ] {
        for stage in crate::program::stages() {
            let symbols = crate::instance::plan_stage_probe(instance.handle(), phase, stage)
                .unwrap_or_else(|status| panic!("{stage:?} {phase:?}: {status:?}"));
            if stage == crate::program::Stage::MediaInject {
                assert!(symbols.is_empty(), "empty media has no injection launch");
                continue;
            }
            if matches!(phase, crate::selection::Phase::Decode(_))
                && matches!(stage, crate::program::Stage::RouterTop8(_))
            {
                assert_eq!(
                    symbols,
                    [format!(
                        "{}_t1_v1",
                        model_mimo26_contract::sm120::moe::FUSED_ROUTER_PREFIX
                    )]
                );
            }
            if matches!(phase, crate::selection::Phase::Decode(_))
                && matches!(
                    stage,
                    crate::program::Stage::MlpReduceNorm(1..)
                        | crate::program::Stage::FinalReduceNorm
                )
            {
                assert_eq!(
                    symbols,
                    [crate::sm120::launch::comm::push_symbols(RANKS, 1).1],
                    "decode MoE layers fold the combine into the all-reduce"
                );
            }
            if matches!(phase, crate::selection::Phase::Decode(1))
                && matches!(stage, crate::program::Stage::AttentionPartial(layer)
                    if crate::program::GLOBAL_LAYERS.contains(&layer))
            {
                assert_eq!(
                    symbols,
                    ["chop_attn_mimo_ga_bf16_decode_rope_table_v1"],
                    "bucket-one GA decode uses the stream-K decode entry with RoPE folded in"
                );
            }
            if matches!(phase, crate::selection::Phase::Decode(_))
                && matches!(
                    stage,
                    crate::program::Stage::RouterProjection(_)
                        | crate::program::Stage::PermuteCount(_)
                        | crate::program::Stage::PermuteScan(_)
                        | crate::program::Stage::PermuteScatter(_)
                        | crate::program::Stage::SlidingWindowWrite(_)
                        | crate::program::Stage::RopeForSlidingWindow(_)
                        | crate::program::Stage::RopeAndPagedWrite(_)
                        | crate::program::Stage::GatherHeadRows
                        | crate::program::Stage::CombineExperts(_)
                )
            {
                assert!(symbols.is_empty(), "{stage:?} has no decode launch");
                continue;
            }
            // A merge after a direct-output partial is skipped: every SWA
            // decode row, and prefill that plans one split.
            if let crate::program::Stage::AttentionMerge(layer) = stage {
                let global = crate::program::GLOBAL_LAYERS.contains(&layer);
                match phase {
                    crate::selection::Phase::Decode(_) => {
                        assert_eq!(symbols.is_empty(), !global, "{stage:?} decode merge");
                    }
                    crate::selection::Phase::Prefill => assert!(symbols.is_empty()),
                    crate::selection::Phase::Verify(_) => {
                        unreachable!("this test plans decode and prefill")
                    }
                }
                continue;
            }
            assert!(!symbols.is_empty(), "{stage:?} must launch a kernel");
            if matches!(phase, crate::selection::Phase::Prefill)
                && matches!(
                    stage,
                    crate::program::Stage::AttentionPartial(layer)
                        if !crate::program::GLOBAL_LAYERS.contains(&layer)
                )
            {
                assert!(
                    symbols
                        .iter()
                        .any(|symbol| symbol == "chop_attn_mimo_swa_prefill_direct_v1"),
                    "SWA prefill must use its direct output kernel"
                );
            }
        }
        assert!(
            crate::instance::grouped_gate_input_is_normalized(instance.handle(), phase)
                .expect("grouped gate input binding"),
            "A16 gate/up must gather from the normalized token rows"
        );
    }
    let norm = crate::instance::rank_view(instance.handle(), 0, "model.norm.weight")
        .expect("M1 named view");
    assert_eq!((norm.offset, norm.length), (1_249_910_784, 8192));
    assert_eq!(
        crate::instance::rank_view(instance.handle(), 0, "missing"),
        Err(BundleStatusV1::NotFound)
    );
    let request = BatchRequestV1::new(
        BatchWireVersionV1::V2,
        vec![BatchItemV1::new(7, true, 1, true, vec![42])],
        vec![BatchAppendV1::new(0, 1, 1, vec![3], None)],
    );
    assert_eq!(
        crate::instance::preflight_request(instance.handle(), b"bad wire"),
        BundleStatusV1::InvalidArgument
    );
    assert_eq!(
        crate::instance::preflight_request(instance.handle(), &request.encode().unwrap()),
        BundleStatusV1::Success
    );
    let media = BatchRequestV1::new(
        BatchWireVersionV1::V3,
        vec![BatchItemV1::new(7, false, 1, true, vec![151655])],
        vec![BatchAppendV1::new(0, 1, 1, vec![3], None)],
    )
    .with_media(runtime_bundle_sdk::BatchMediaSectionV3 {
        injects: vec![runtime_bundle_sdk::BatchMediaInjectV3 {
            item_index: 0,
            item_row: 0,
            rows: 1,
            arena_row: 0,
        }],
        ..runtime_bundle_sdk::BatchMediaSectionV3::default()
    });
    assert_eq!(
        crate::instance::preflight_request(instance.handle(), &media.encode().unwrap()),
        BundleStatusV1::Unsupported,
        "instances without encoder images must refuse media"
    );
    let positioned = BatchRequestV1::new(
        BatchWireVersionV1::V3,
        vec![BatchItemV1::new(7, true, 1, true, vec![42]).with_position_axes(vec![[0, 0, 0]])],
        vec![BatchAppendV1::new(0, 1, 1, vec![3], None)],
    );
    assert_eq!(
        crate::instance::preflight_request(instance.handle(), &positioned.encode().unwrap()),
        BundleStatusV1::InvalidArgument,
        "MiMo declares no position capability and refuses explicit positions"
    );
    let beyond_pool = BatchRequestV1::new(
        BatchWireVersionV1::V2,
        vec![BatchItemV1::new(7, true, 1, true, vec![42])],
        vec![BatchAppendV1::new(0, 1, 1, vec![128], None)],
    );
    assert_eq!(
        crate::instance::preflight_request(instance.handle(), &beyond_pool.encode().unwrap()),
        BundleStatusV1::InvalidArgument
    );
    let packed = crate::inputs::PackedStep::pack::<Mimo>(
        crate::contract::MODEL_LIMITS,
        &request,
        32_768,
        |_| Some(0),
    )
    .expect("packed decode");
    assert_eq!(
        crate::instance::upload_packed_phase(instance.handle(), &packed.decode),
        BundleStatusV1::Success
    );
    // Create wrote three constants and two RoPE tables per rank. Each rank
    // then takes one packed input copy and the first block-table row write.
    let layout = crate::upload::InputLayout::decode_within(crate::contract::ITEMS_32);
    assert_eq!(backend.host_copies.borrow().len(), RANKS as usize * 7);
    let (length, bytes) = backend.host_copies.borrow()[(RANKS * 5) as usize].clone();
    assert_eq!(length, layout.end);
    assert_eq!(bytes[..4], 1_u32.to_le_bytes());
    let (length, bytes) = backend.host_copies.borrow()[(RANKS * 5 + 1) as usize].clone();
    assert_eq!(length, 4 * u64::from(crate::contract::MAX_PAGES));
    assert_eq!(bytes[..8], [3, 0, 0, 0, 0xff, 0xff, 0xff, 0xff]);
    assert_eq!(
        crate::instance::upload_packed_phase(instance.handle(), &packed.decode),
        BundleStatusV1::Success
    );
    // The same step again changes no block-table entry.
    assert_eq!(backend.host_copies.borrow().len(), RANKS as usize * 8);
    assert_eq!(
        crate::instance::launch_embedding_probe(instance.handle(), None),
        BundleStatusV1::Success
    );
    assert_eq!(backend.launches.get(), 2);
    assert_eq!(
        crate::instance::launch_embedding_probe(instance.handle(), Some(1)),
        BundleStatusV1::Success
    );
    assert_eq!(
        crate::instance::launch_embedding_probe(instance.handle(), Some(1)),
        BundleStatusV1::Success
    );
    assert_eq!(backend.graph_begins.get(), 1);
    assert_eq!(backend.graph_ends.get(), 1);
    assert_eq!(backend.graph_replays.get(), 2);
    assert_eq!(backend.launches.get(), 4);
    assert_eq!(backend.graph_releases.get(), 0);
    // The device backup is the first launch of each phase, so opening the
    // transaction copies nothing, and a cancel before any phase was enqueued
    // restores nothing: those entries were never overwritten.
    assert_eq!(
        crate::instance::prepare_ring_append(instance.handle(), &[(7, 0, 1)]),
        BundleStatusV1::Success
    );
    assert!(backend.copies.borrow().is_empty());
    assert_eq!(
        bundle_runtime::entry::cancel::<crate::instance::Instance>(instance.handle()),
        BundleStatusV1::Success
    );
    assert!(backend.copies.borrow().is_empty());
    assert_eq!(
        crate::instance::prepare_ring_append(instance.handle(), &[(7, 0, 1)]),
        BundleStatusV1::Success
    );
    assert_eq!(
        bundle_runtime::entry::commit::<crate::instance::Instance>(instance.handle()),
        BundleStatusV1::Success
    );
    assert!(backend.copies.borrow().is_empty());
    assert_eq!(
        bundle_runtime::entry::retire_request::<crate::instance::Instance>(instance.handle(), 7),
        BundleStatusV1::Success
    );
    // Speculation is off, so there is no verify arena. The logits row
    // exchange adds flags and an epoch to each comm bucket up to 256 rows.
    // Each rank also owns two immutable RoPE tables.
    assert_eq!(
        backend.allocations.borrow().len(),
        RANKS as usize * (84 + 2 * 9)
    );
    assert!(backend.allocations.borrow().contains(
        &(u64::from(DECODE_ENTRY_ROWS)
            * u64::from(VOCAB_PER_RANK)
            * u64::from(RANKS)
            * crate::contract::DECODE_LOGIT_BYTES)
    ));
    assert!(backend.allocations.borrow().contains(
        &(u64::from(crate::contract::SCORED_ROWS)
            * u64::from(VOCAB_PER_RANK)
            * u64::from(RANKS)
            * crate::contract::PREFILL_LOGIT_BYTES)
    ));
    // Decode logits are F32 by default (`CHOPINFER_MIMO_F32_LOGITS`).
    assert!(backend.allocations.borrow().contains(
        &(u64::from(DECODE_ENTRY_ROWS)
            * u64::from(VOCAB_PER_RANK)
            * u64::from(RANKS)
            * crate::contract::PREFILL_LOGIT_BYTES)
    ));
    assert!(
        backend.allocations.borrow().contains(
            &crate::sm120::launch::workspace_bytes(Tp::TP4, crate::contract::ITEMS_32)
                .expect("workspace"),
        )
    );
    // With the exchange flags and epochs of nine comm buckets.
    assert_eq!(backend.zeroes.get(), RANKS * (45 + 2 * 9));
    assert_eq!(backend.kv_acquires.get(), 1);
    assert_eq!(backend.kv_resolves.get(), RANKS);
    assert_eq!(backend.peer_grants.get(), RANKS * (RANKS - 1));
    assert_eq!(
        backend.symbols.get() as usize,
        RANKS as usize
            * crate::CUBINS
                .iter()
                .filter(|cubin| {
                    crate::topology::cubin_serves(cubin.name, Tp::TP4)
                        && (!crate::sm120::dense::medium_cubin(cubin.name)
                            || crate::switches::verify_dense_mid_enabled())
                        // A 32-slot ring runs the default ring entries.
                        && !cubin.name.ends_with("_ring_slots")
                })
                .map(|cubin| cubin.symbols.len())
                .sum::<usize>()
    );
    instance.destroy().expect("fake-host destroy");
    assert_eq!(backend.graph_releases.get(), 1);
    assert!(backend.released.get() > backend.symbols.get());
}

/// A 2048-token prefill pipelined over two halves: all-reduces run on the
/// comm streams behind events recorded on the rank streams, everything else
/// on the rank streams, and the logits views match the unpipelined step.
#[test]
fn pipelined_prefill_orders_halves_across_two_streams() {
    use runtime_bundle_sdk::{BatchAppendV1, BatchItemV1, BatchRequestV1, BatchWireVersionV1};

    let request = BatchRequestV1::new(
        BatchWireVersionV1::V2,
        vec![BatchItemV1::new(77, false, 2048, true, vec![7; 2048])],
        vec![BatchAppendV1::new(0, 2048, 2048, (0..32).collect(), None)],
    )
    .encode()
    .unwrap();
    let api = BundleApiV1::new_with_prefix_cache(crate::BUNDLE_MIMO26_EXTENDED_API_V1)
        .expect("bundle API");
    let mut views = Vec::new();
    for pipelined in [false, true] {
        let backend = Rc::new(FakeHost::default());
        let table = HostServicesTableV1::new(Rc::clone(&backend));
        let mut instance = api.create(table, &configuration()).expect("create");
        let config = crate::overlap::Config {
            sms: 188,
            reserved_sms: 24,
            comm_ctas: Some(16),
            min_tokens: 512,
        };
        crate::instance::force_overlap(instance.handle(), pipelined.then_some(config));
        let comm = crate::instance::comm_streams(instance.handle());
        backend.stream_launches.borrow_mut().clear();
        let mut sink = ViewSink::default();
        let result = instance.execute(&request, &mut sink);
        assert!(result.is_ok(), "prefill pipelined {pipelined}: {result:?}");
        views.push(
            sink.0
                .iter()
                .map(|view| {
                    let raw = view.raw();
                    (raw.first_item_index, raw.row_count, raw.buffer.offset)
                })
                .collect::<Vec<_>>(),
        );
        let launches = backend.stream_launches.borrow().clone();
        let is_comm = |symbol: &str| {
            symbol.starts_with("chop_comm_reduce_scatter")
                || symbol.starts_with("chop_comm_allgather_scattered")
                || symbol.starts_with("chop_comm_allgather_presummed")
        };
        let count = |name: &str| {
            launches
                .iter()
                .filter(|(_, symbol, _)| symbol == name)
                .count()
        };
        // Row kernels launch the smallest row bucket holding their rows: the
        // whole chunk, or each half of a pipelined one (`Phase::row_grid`).
        let rows = if pipelined { 1024 } else { 2048 };
        for prefix in [
            "chop_row_rms_mimo_",
            "chop_row_rope_mimo_",
            "chop_row_ring_write_mimo_",
            "chop_row_router_mimo_",
            "chop_row_combine_mimo",
        ] {
            let grids = launches
                .iter()
                .filter(|(_, symbol, _)| symbol.starts_with(prefix))
                .map(|(_, _, grid)| *grid)
                .collect::<std::collections::BTreeSet<_>>();
            assert_eq!(grids, [rows].into(), "{prefix} pipelined {pipelined}");
        }
        if !pipelined {
            assert!(launches.iter().all(|(stream, _, _)| !comm.contains(stream)));
            assert!(backend.event_ops.borrow().is_empty());
            instance.commit().expect("commit");
            instance.destroy().expect("destroy");
            continue;
        }
        // All-reduces, and only they, run on the comm streams: two per layer
        // per half per rank, each respecting the 16-CTA comm budget.
        for (stream, symbol, grid) in &launches {
            assert_eq!(comm.contains(stream), is_comm(symbol), "{symbol}");
            if is_comm(symbol) {
                assert_eq!(*grid, 16, "{symbol}");
            }
        }
        assert_eq!(
            count(crate::sm120::launch::comm::PHASED_SYMBOLS[0]),
            2 * 48 * 2 * RANKS as usize
        );
        // The persistent GEMM and MoE grids leave 24 SMs free.
        assert!(
            launches.iter().any(|(_, symbol, grid)| {
                symbol == "chop_gemm_mimo_o_prefill_v1" && *grid == 164
            })
        );
        assert!(launches.iter().any(|(_, symbol, grid)| {
            symbol.starts_with("chop_moe_mimo_gateup_prefill") && *grid == 164
        }));
        // Whole-chunk stages run once per rank; per-half stages twice.
        assert_eq!(count("chop_mimo_ring_backup_v1"), RANKS as usize);
        assert_eq!(count("chop_gemm_mimo_head_f32_prefill_v1"), RANKS as usize);
        assert_eq!(
            count("chop_gemm_mimo_o_prefill_v1"),
            48 * 2 * RANKS as usize
        );
        // Every wait follows a record of the same event on the other stream.
        let mut recorded = std::collections::BTreeMap::new();
        for &(stream, event, wait) in backend.event_ops.borrow().iter() {
            if wait {
                let from = recorded.get(&event).copied().expect("wait follows record");
                assert_ne!(comm.contains(&from), comm.contains(&stream));
            } else {
                recorded.insert(event, stream);
            }
        }
        // Cancel restores the chunk's saved ring entries on every rank.
        let copies = backend.copies.borrow().len();
        instance.cancel().expect("cancel");
        assert_eq!(
            backend.copies.borrow().len() - copies,
            (RANKS as u64 * 128 * SWA_LAYERS * 2) as usize
        );
        instance.destroy().expect("destroy");
    }
    assert_eq!(views[0], views[1]);
}

/// The pre-port bundle's encoder buffers in its allocation order, by the
/// plan region of each one's role, that f32-pairs.v1 still has, then
/// f32-pairs.v1's own: `shared_encoder` lays the plan's regions out in this
/// order.
const PREPORT_LAYOUT: [&str; 17] = [
    "arena",
    "payload",
    "source-rows",
    "im2col",
    "joined",
    "count",
    "carry",
    "gathered",
    "cos",
    "sin",
    "reorder",
    "sequences",
    "codebook-offsets",
    "staging",
    "stream",
    "a",
    "b",
];

/// Runs `f` with the services of a fresh fake host.
pub(crate) fn with_services<R>(f: impl FnOnce(&HostServicesV1) -> R) -> R {
    let host = Rc::new(FakeHost::default());
    let table = HostServicesTableV1::new(Rc::clone(&host));
    f(table.caller())
}

/// MiMo's encoders on the shared runtime over `services`, one rank per
/// handle: each rank's regions laid out in `PREPORT_LAYOUT` order in its
/// handle, and its image views after them. Also each rank's symbols,
/// resolved in module `1, rank + 1`.
pub(crate) fn shared_encoder(
    services: &HostServicesV1,
    handles: &[BundleDeviceBufferHandleV1],
) -> (
    bundle_runtime::encoder::Encoder,
    Vec<BTreeMap<String, BundleCudaSymbolHandleV1>>,
) {
    use crate::encoders::runtime;
    // Catalog ownership can reorder rows. Keep this legacy encoder fixture's
    // symbol handles in their original order, with the same complete row set.
    let legacy_rows = include_str!("../fixtures/legacy_encoder_row_symbols.txt")
        .lines()
        .map(str::to_owned)
        .collect::<Vec<_>>();
    let selected_rows = crate::sm120::reference::row_catalog()
        .into_iter()
        .filter(|entry| !model_mimo26_contract::sm120::PROFILES[1..].contains(&entry.profile))
        .map(|entry| entry.symbol)
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(
        legacy_rows
            .iter()
            .cloned()
            .collect::<std::collections::BTreeSet<_>>(),
        selected_rows
    );
    let names = model_mimo26_contract::sm120::encoders::kernels_flash()
        .unwrap()
        .iter()
        .map(|entry| entry.symbol.clone())
        .chain(
            model_mimo26_contract::sm120::encoders::rows::kernels_flash()
                .unwrap()
                .into_iter()
                .map(|kernel| kernel.symbol),
        )
        .chain(legacy_rows)
        .collect::<Vec<_>>();
    let plan = runtime::plan(model_mimo26_contract::encoders::Encoders::ALL).unwrap();
    let mut offsets = BTreeMap::new();
    let mut next = 0;
    for name in PREPORT_LAYOUT {
        offsets.insert(name, next);
        next += plan.region(name).unwrap().bytes;
    }
    assert_eq!(offsets.len(), plan.regions.len());
    let mut symbols = Vec::new();
    let mut weights = Vec::new();
    for (rank, &handle) in handles.iter().enumerate() {
        let module = BundleCudaModuleHandleV1::from_parts(1, rank as u32 + 1);
        symbols.push(
            names
                .iter()
                .map(|name| {
                    let symbol = services
                        .resolve_cuda_symbol(module, name.as_bytes())
                        .unwrap();
                    let shared = crate::sm120::modules::dynamic_shared_bytes(name, ITEMS_32)
                        .expect("encoder symbol shared memory");
                    bundle_runtime::acquire::configure_symbol(
                        bundle_runtime::acquire::Log {
                            services,
                            bundle: "MiMo",
                        },
                        name,
                        symbol,
                        shared,
                    )
                    .unwrap();
                    (name.clone(), symbol)
                })
                .collect::<BTreeMap<_, _>>(),
        );
        let span = DeviceBufferSpanV1::new(handle, next, 1 << 30);
        let mut views = BTreeMap::new();
        for name in model_mimo26_contract::encoders::spec::declarations_flash()
            .unwrap()
            .iter()
            .flat_map(|declaration| declaration.tensors())
        {
            if let Some(stem) = name.strip_suffix(".weight") {
                views.insert(format!("{stem}.bias"), span);
            }
            views.insert(name, span);
        }
        views.insert("@codebooks".into(), span);
        views.insert("@embeddings".into(), span);
        weights.push(views);
    }
    let mut regions = handles
        .iter()
        .flat_map(|_| plan.regions.iter().map(|region| region.name))
        .collect::<Vec<_>>()
        .into_iter();
    let encoder = crate::sm120::encoders::create(
        services,
        model_mimo26_contract::encoders::Encoders::ALL,
        weights,
        |rank, bytes| {
            let name = regions.next().unwrap();
            Ok(DeviceBufferSpanV1::new(handles[rank], offsets[name], bytes))
        },
    )
    .unwrap();
    (encoder, symbols)
}

/// Runs `f` with the handle of `encoder_fault_fixture`'s live instance.
pub(crate) fn with_encoder_instance<R>(f: impl FnOnce(BundleHandleV1) -> R) -> R {
    let (_host, instance, _request) = encoder_fault_fixture();
    f(instance.handle())
}

fn encoder_fault_fixture() -> (
    Rc<FakeHost>,
    runtime_bundle_abi::BundleInstanceV1,
    runtime_bundle_sdk::BatchRequestV1,
) {
    use runtime_bundle_sdk::{
        BatchAppendV1, BatchItemV1, BatchMediaEncodeV3, BatchMediaInjectV3, BatchMediaSectionV3,
        BatchRequestV1, BatchWireVersionV1,
    };
    let (backend, instance) = fault_instance();
    let table = HostServicesTableV1::new(Rc::clone(&backend));
    let handles = (0..RANKS)
        .map(|rank| BundleDeviceBufferHandleV1::from_parts(1, 60_000 + rank))
        .collect::<Vec<_>>();
    let (encoder, _) = shared_encoder(table.caller(), &handles);
    crate::instance::force_encoder(instance.handle(), encoder);
    let request = BatchRequestV1::new(
        BatchWireVersionV1::V3,
        vec![BatchItemV1::new(7, false, 1, true, vec![151655])],
        vec![BatchAppendV1::new(0, 1, 1, vec![3], None)],
    )
    .with_media(BatchMediaSectionV3 {
        encodes: vec![BatchMediaEncodeV3 {
            encoder: 0,
            geometry: [1, 2, 2, 0],
            output_rows: 1,
            pages: vec![0],
            payload_offset: 0,
            payload_bytes: 4 * 1536 * 4,
        }],
        injects: vec![BatchMediaInjectV3 {
            item_index: 0,
            item_row: 0,
            rows: 1,
            arena_row: 0,
        }],
        payload: vec![0; 4 * 1536 * 4],
    });
    (backend, instance, request)
}

/// An encoder failure drains every stream before any text or injection
/// work, and poisons the instance only when a stream does not drain.
#[test]
fn encoder_failures_drain_before_text_or_injection_and_poison_when_undrained() {
    // First/middle rank failures, second-record failure, and source-row
    // upload. A vision record launches 280 kernels on each of four ranks.
    for (serial, launches, second_record, upload) in [
        (false, 1, false, false),
        (false, 2, false, false),
        (false, 1121, true, false),
        (false, 1120, false, true),
        (true, 1, false, false),
    ] {
        // The upload fault and the drain fault share the fake host's slot.
        for undrained in [false, true].into_iter().filter(|&u| !(u && upload)) {
            let (backend, mut instance, mut request) = encoder_fault_fixture();
            crate::instance::encoder_test_options(instance.handle(), serial, None, None);
            if second_record {
                let mut media = request.media().clone();
                let mut record = media.encodes[0].clone();
                record.pages = vec![1];
                record.payload_offset = record.payload_bytes;
                media.encodes.push(record);
                media.payload.resize(media.payload.len() * 2, 0);
                request = request.with_media(media);
            }
            if upload {
                // Each rank uploads the record's payload, then its staged
                // side inputs; the ninth upload is the source rows'.
                backend.prefix_fault.set(Some((PrefixFault::Upload, 8)));
            } else {
                backend.fail_launch.set(Some(launches));
            }
            if undrained {
                // The failed submission's first drain; a first record drains
                // its four ranks before.
                let drains = if second_record { 4 } else { 0 };
                backend
                    .prefix_fault
                    .set(Some((PrefixFault::Synchronize, drains)));
            }
            let before = backend.journal.borrow().len();
            assert!(matches!(
                instance.execute(&request.encode().unwrap(), &mut ViewSink::default()),
                Err(BundleAbiError::Status(BundleStatusV1::InternalError))
            ));
            let log = backend.journal.borrow();
            let issued = &log[before..];
            assert_eq!(
                issued
                    .iter()
                    .filter(|line| line.starts_with("launch "))
                    .count(),
                launches as usize
            );
            assert!(!issued.iter().any(|line| {
                line.contains("chop_row_embedding_full_v1")
                    || line.contains(
                        model_mimo26_contract::sm120::encoders::kernel_flash(
                            "chop_enc_layout_tokens_w4096_v1",
                        )
                        .unwrap()
                        .symbol
                        .as_str(),
                    )
            }));
            // Every stream a record or upload touched drains: all four, or
            // in the serial mode the failed rank's.
            let waits = issued
                .iter()
                .filter(|line| line.starts_with("synchronize "))
                .count();
            if !undrained {
                let expected = match (serial, second_record || upload) {
                    (true, _) => 1,
                    (false, true) => 8,
                    (false, false) => 4,
                };
                assert_eq!(waits, expected);
            }
            let after = log.len();
            drop(log);
            if !undrained {
                // Not poisoned: no step is open, and the instance releases.
                assert_eq!(
                    bundle_runtime::entry::destroy::<crate::instance::Instance>(instance.handle()),
                    BundleStatusV1::Success,
                    "{serial} {launches} {second_record} {upload}"
                );
                continue;
            }
            assert_eq!(
                bundle_runtime::entry::cancel::<crate::instance::Instance>(instance.handle()),
                BundleStatusV1::InvalidState
            );
            assert_eq!(
                bundle_runtime::entry::destroy::<crate::instance::Instance>(instance.handle()),
                BundleStatusV1::InvalidState
            );
            assert!(
                !backend.journal.borrow()[after..]
                    .iter()
                    .any(|line| queues_device_work(line))
            );
        }
    }
}

#[test]
fn invalid_later_geometry_refuses_without_poison_or_device_work() {
    for serial in [false, true] {
        let (backend, mut instance, request) = encoder_fault_fixture();
        crate::instance::encoder_test_options(instance.handle(), serial, None, None);
        let mut media = request.media().clone();
        let mut bad = media.encodes[0].clone();
        bad.geometry = [1, 3, 2, 0];
        bad.pages = vec![1];
        bad.payload_offset = bad.payload_bytes;
        media.encodes.push(bad);
        media.payload.resize(media.payload.len() * 2, 0);
        let before = backend.journal.borrow().len();
        assert!(matches!(
            instance.execute(
                &request.clone().with_media(media).encode().unwrap(),
                &mut ViewSink::default()
            ),
            Err(BundleAbiError::Status(BundleStatusV1::InvalidArgument))
        ));
        assert!(
            !backend.journal.borrow()[before..]
                .iter()
                .any(|line| queues_device_work(line))
        );
        // A subsequent encode reaches its first launch; refusal did not poison.
        let good = request.clone();
        backend.fail_launch.set(Some(1));
        assert!(
            instance
                .execute(&good.encode().unwrap(), &mut ViewSink::default())
                .is_err()
        );
        assert!(backend.journal.borrow()[before..].iter().any(|line| {
            line.contains(
                model_mimo26_contract::sm120::encoders::kernel_flash(
                    "chop_enc_linear_pairs_n1280_k1536_f32_v1",
                )
                .unwrap()
                .symbol
                .as_str(),
            )
        }));
    }
}

#[test]
fn missing_later_rank_binding_is_detected_before_any_encode_submission() {
    let (backend, mut instance, request) = encoder_fault_fixture();
    crate::instance::encoder_test_options(instance.handle(), false, None, Some(2));
    let before = backend.journal.borrow().len();
    assert!(
        instance
            .execute(&request.encode().unwrap(), &mut ViewSink::default())
            .is_err()
    );
    assert!(
        !backend.journal.borrow()[before..]
            .iter()
            .any(|line| queues_device_work(line) && !line.starts_with("synchronize "))
    );
    assert_eq!(
        bundle_runtime::entry::cancel::<crate::instance::Instance>(instance.handle()),
        BundleStatusV1::InvalidState
    );
}

#[test]
fn arena_file_errors_are_nonfatal_and_existing_captures_survive_restart() {
    let root = std::env::temp_dir().join(format!("encoder-arena-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    let blocked = root.join("file");
    std::fs::write(&blocked, b"old evidence").unwrap();
    for path in [blocked.clone(), root.clone(), root.clone()] {
        let (backend, mut instance, request) = encoder_fault_fixture();
        crate::instance::encoder_test_options(instance.handle(), false, Some(path), None);
        // Fake encoder arenas return deterministic bytes; the normal logits path
        // still uses its ordinary fake-host assertions.
        backend.encoder_readbacks.set(true);
        let before = backend.journal.borrow().len();
        instance
            .execute(&request.encode().unwrap(), &mut ViewSink::default())
            .unwrap();
        assert!(backend.journal.borrow()[before..].iter().any(|line| {
            line.contains(
                model_mimo26_contract::sm120::encoders::kernel_flash(
                    "chop_enc_layout_tokens_w4096_v1",
                )
                .unwrap()
                .symbol
                .as_str(),
            )
        }));
        instance.cancel().unwrap();
    }
    assert_eq!(std::fs::read(blocked).unwrap(), b"old evidence");
    let runs = bundle_runtime::recorded::capture_root(&root);
    assert_eq!(std::fs::read_dir(runs).unwrap().count(), 8);
    std::fs::remove_dir_all(root).unwrap();
}

/// Encoder captures are diagnostics: a capture root that is a file, or a
/// write that fails after a record's rows were written, disables them, and
/// the step still serves (the prefill injects the encoded rows).
#[test]
fn encoder_capture_file_errors_never_fail_a_step() {
    use std::os::unix::fs::PermissionsExt;
    let root = std::env::temp_dir().join(format!("encoder-capture-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();
    let blocked = root.join("file");
    std::fs::write(&blocked, b"not a directory").unwrap();
    let override_symbol =
        model_mimo26_contract::sm120::encoders::kernel_flash("chop_enc_layout_tokens_w4096_v1")
            .unwrap()
            .symbol
            .clone();
    for second_record in [false, true] {
        let (backend, mut instance, mut request) = encoder_fault_fixture();
        backend.encoder_readbacks.set(true);
        let path = if second_record {
            // Two records; the step directory turns read-only once the
            // first record's rows were read back.
            let mut media = request.media().clone();
            let mut record = media.encodes[0].clone();
            record.pages = vec![1];
            record.payload_offset = record.payload_bytes;
            media.encodes.push(record);
            media.payload.resize(media.payload.len() * 2, 0);
            request = request.with_media(media);
            let dir = root.join("steps");
            let steps = dir.clone();
            let mut reads = 0;
            *backend.encoder_readback_hook.borrow_mut() = Some(Box::new(move || {
                reads += 1;
                if reads == 2 {
                    for step in std::fs::read_dir(&steps).unwrap() {
                        let step = step.unwrap().path();
                        let readonly = std::fs::Permissions::from_mode(0o555);
                        std::fs::set_permissions(step, readonly).unwrap();
                    }
                }
            }));
            dir
        } else {
            blocked.clone()
        };
        crate::instance::encoder_test_capture(instance.handle(), path.clone());
        let before = backend.journal.borrow().len();
        instance
            .execute(&request.encode().unwrap(), &mut ViewSink::default())
            .unwrap();
        assert!(
            backend.journal.borrow()[before..]
                .iter()
                .any(|line| line.contains(&override_symbol)),
            "the prefill injects the rows"
        );
        assert!(
            backend
                .logs
                .borrow()
                .iter()
                .any(|line| line.contains("MiMo encoder capture disabled")),
            "{second_record}"
        );
        if second_record {
            // The first record's rows were written, no manifest was.
            let step = std::fs::read_dir(&path)
                .unwrap()
                .next()
                .unwrap()
                .unwrap()
                .path();
            assert!(step.join("encode-0.bin").exists());
            assert!(!step.join("manifest.json").exists());
            std::fs::set_permissions(&step, std::fs::Permissions::from_mode(0o755)).unwrap();
        }
        instance.cancel().unwrap();
    }
    assert_eq!(std::fs::read(&blocked).unwrap(), b"not a directory");
    std::fs::remove_dir_all(root).unwrap();
}

/// This host records kernels and injects readback tokens. It establishes
/// matching projection work after restoring the materialized cursor.
#[test]
fn dflash_restore_preserves_incremental_projection_tail() {
    use runtime_bundle_sdk::{BatchAppendV1, BatchItemV1, BatchRequestV1, BatchWireVersionV1};

    for prompt in [64_u64, 2048] {
        let (host, mut instance) = dflash_instance();
        instance
            .execute(
                &dflash_step(0, vec![40; prompt as usize], None),
                &mut DraftSink::default(),
            )
            .unwrap();
        instance.commit().unwrap();
        *host.readback_tokens.borrow_mut() = vec![40; 8];
        instance
            .execute(
                &dflash_step(prompt, vec![40], Some(8)),
                &mut DraftSink::default(),
            )
            .unwrap();
        instance.commit_accepted(&[kept(7, 4)]).unwrap();
        let boundary = prompt + 4;
        let pages = (10..10 + instance.describe_prefix_state().unwrap()).collect::<Vec<_>>();
        let cache_views = crate::instance::dflash_cache_views(instance.handle());
        // State-sensitive draft fixture: nontrivial BF16 bit patterns in all
        // ten projected planes, including signed zero and NaN payloads. The
        // fake host moves these bytes; it does not emulate CUDA arithmetic.
        for (rank, cache) in cache_views.iter().enumerate() {
            for slot in [0, 1, 2] {
                for (plane, range) in projected_planes(slot, cache.length).into_iter().enumerate() {
                    for offset in [
                        0,
                        16,
                        (range.end - range.start) / 2,
                        range.end - range.start - 16,
                    ] {
                        for (byte, value) in [
                            0_u8,
                            128,
                            193,
                            127,
                            rank as u8,
                            plane as u8,
                            slot as u8,
                            255,
                        ]
                        .into_iter()
                        .enumerate()
                        {
                            host.snapshot_memory.borrow_mut().insert(
                                (
                                    cache.buffer.raw,
                                    cache.offset + range.start + offset + byte as u64,
                                ),
                                value,
                            );
                        }
                    }
                }
            }
        }
        let state = |slot| {
            cache_views
                .iter()
                .flat_map(|cache| {
                    projected_planes(slot, cache.length)
                        .into_iter()
                        .flat_map(|range| {
                            host.snapshot_memory
                                .borrow()
                                .range(
                                    (cache.buffer.raw, cache.offset + range.start)
                                        ..(cache.buffer.raw, cache.offset + range.end),
                                )
                                .map(|(&(_, offset), &byte)| {
                                    (offset - cache.offset - range.start, byte)
                                })
                                .collect::<Vec<_>>()
                        })
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>()
        };
        let original = state(0);
        let neighbor = state(2);
        let draft_words = |values: &[(u64, u8)]| {
            (0..8)
                .map(|index| {
                    values.iter().fold(index + 40_u32, |hash, (offset, byte)| {
                        hash.wrapping_mul(31)
                            .wrapping_add(*offset as u32 ^ u32::from(*byte))
                    }) % 150_000
                })
                .collect::<Vec<_>>()
        };
        *host.readback_tokens.borrow_mut() = draft_words(&original);
        instance.capture_prefix_state(7, boundary, &pages).unwrap();

        let before = host.launch_symbols.borrow().len();
        instance
            .execute(
                &dflash_step(boundary, vec![40], Some(8)),
                &mut DraftSink::default(),
            )
            .unwrap();
        let incremental = host.launch_symbols.borrow()[before..].to_vec();
        let cells = (prompt..boundary)
            .map(|position| {
                crate::retain::cell(0, position, crate::dflash::state::HISTORY_CAPACITY)
            })
            .collect::<Vec<_>>();
        assert_eq!(host.histories_gathering(&cells).len(), RANKS as usize);
        let incremental_fc = incremental
            .iter()
            .filter(|name| name.contains("dflash_fc"))
            .collect::<Vec<_>>();
        assert_eq!(incremental_fc.len(), RANKS as usize);
        assert!(incremental_fc.iter().all(|name| !name.contains("prefill")));
        instance.cancel().unwrap();

        instance
            .restore_prefix_state(8, &vec![40; boundary as usize], &pages)
            .unwrap();
        assert_eq!(
            state(1),
            original,
            "restore must retain every modeled projected byte"
        );
        assert_eq!(state(2), neighbor, "another slot must be untouched");
        *host.readback_tokens.borrow_mut() = draft_words(&state(1));
        let verify = BatchRequestV1::new(
            BatchWireVersionV1::V3,
            vec![BatchItemV1::verify(8, 8, true, vec![40])],
            vec![BatchAppendV1::new(
                boundary,
                8,
                boundary + 8,
                (0..(boundary + 8).div_ceil(PAGE_TOKENS)).collect(),
                None,
            )],
        )
        .encode()
        .unwrap();
        let before = host.launch_symbols.borrow().len();
        let mut restored_sink = DraftSink::default();
        instance.execute(&verify, &mut restored_sink).unwrap();
        assert_eq!(restored_sink.drafts[0].tokens, draft_words(&original)[1..]);
        let restored = host.launch_symbols.borrow()[before..].to_vec();
        let start = prompt;
        let cells = (start..boundary)
            .map(|position| {
                crate::retain::cell(1, position, crate::dflash::state::HISTORY_CAPACITY)
            })
            .collect::<Vec<_>>();
        assert_eq!(host.histories_gathering(&cells).len(), RANKS as usize);
        assert_eq!(count(&restored, "chop_gemm_mimo_dflash_fc_prefill_v1"), 0);
        eprintln!(
            "DFlash boundary {boundary}: incremental projects 4 rows with {incremental_fc:?}; restore projects the same {} rows",
            boundary - start
        );
        instance.cancel().unwrap();
        instance.destroy().unwrap();
    }
}

/// The ranges of `slot`'s ten DFlash projected planes in a TP4 working
/// cache of `bytes`, as the declared table copies them.
fn projected_planes(slot: u32, bytes: u64) -> Vec<std::ops::Range<u64>> {
    use bundle_runtime::state::{Extent, Holder};
    let slots = (bytes / crate::dflash::program::cache_bytes(Tp::TP4, 1)) as u32;
    let state = model_mimo26_contract::state::request_state(
        Tp::TP4,
        &KV_FORMATS[0],
        crate::prefix::Drafter::Dflash,
    );
    let extent = Extent {
        served: slots,
        bucket: slots,
        verify_rows: MAX_VERIFY_ROWS,
    };
    bundle_runtime::snapshot::Table::new(&state, extent, Holder::Slot(slot))
        .unwrap()
        .descriptors
        .iter()
        .filter(|descriptor| descriptor.buffer == 2)
        .map(|descriptor| descriptor.offset..descriptor.offset + descriptor.length)
        .collect()
}

#[test]
fn a_prefix_snapshot_round_trips_the_ring_slot_and_drafter_history() {
    use runtime_bundle_sdk::{BatchAppendV1, BatchItemV1, BatchRequestV1, BatchWireVersionV1};
    for dflash in [false, true] {
        for boundary in [3, 255, 258, 2045] {
            let api =
                BundleApiV1::new_with_prefix_cache(crate::BUNDLE_MIMO26_EXTENDED_API_V1).unwrap();
            assert!(api.supports_prefix_cache());
            let host = Rc::new(FakeHost::default());
            host.speculation.set(Some((7, 32)));
            host.dflash.set(dflash);
            let mut instance = api
                .create(
                    HostServicesTableV1::new(Rc::clone(&host)),
                    &configuration_with_mtp(true),
                )
                .unwrap();
            let step = |request: u64, start: u64, tokens: Vec<u32>| {
                let rows = tokens.len() as u32;
                BatchRequestV1::new(
                    BatchWireVersionV1::V3,
                    vec![BatchItemV1::new(request, false, rows, true, tokens)],
                    vec![BatchAppendV1::new(
                        start,
                        u64::from(rows),
                        start + u64::from(rows),
                        (0..(start + u64::from(rows)).div_ceil(PAGE_TOKENS)).collect(),
                        None,
                    )],
                )
                .encode()
                .unwrap()
            };
            let tokens = (40..40 + boundary as u32).collect::<Vec<_>>();
            instance
                .execute(&step(7, 0, tokens.clone()), &mut DraftSink::default())
                .unwrap();
            instance.commit().unwrap();
            let pages = instance.describe_prefix_state().unwrap();

            assert_eq!(pages, if dflash { 75 } else { 21 });
            let pages = (10..10 + pages).collect::<Vec<_>>();

            let drafter = if dflash {
                crate::prefix::Drafter::Dflash
            } else {
                crate::prefix::Drafter::Mtp
            };
            let table = pages
                .iter()
                .flat_map(|page| page.to_le_bytes())
                .collect::<Vec<_>>();
            // Per rank: one page table upload and one copy launch, then one
            // synchronize once every rank launched. No per-piece copies.
            let check = |slot: u32, to_pages: u32| {
                let copies = host.copies.borrow().len();
                let uploads = host.host_copies.borrow().len();
                let syncs = host.stream_syncs.get();
                let launched = host.prefix_copies.borrow().len();
                let table = table.as_slice();
                move |host: &FakeHost| {
                    if dflash {
                        assert!(host.copies.borrow().len() > copies);
                    } else {
                        assert_eq!(host.copies.borrow().len(), copies);
                    }
                    assert_eq!(host.stream_syncs.get() - syncs, RANKS);
                    let uploads = host.host_copies.borrow()[uploads..].to_vec();
                    assert_eq!(
                        uploads.len(),
                        RANKS as usize * if dflash && to_pages != 0 { 2 } else { 1 }
                    );
                    let uploads = uploads
                        .iter()
                        .filter(|(length, _)| *length == table.len() as u64)
                        .collect::<Vec<_>>();
                    assert_eq!(uploads.len(), RANKS as usize);
                    // The recorder keeps an upload's first 512 bytes.
                    assert!(
                        uploads
                            .iter()
                            .all(|(length, bytes)| *length == table.len() as u64
                                && bytes.as_slice() == &table[..table.len().min(512)])
                    );
                    let launches = host.prefix_copies.borrow()[launched..].to_vec();
                    assert_eq!(launches.len(), RANKS as usize);
                    let streams = launches
                        .iter()
                        .map(|(stream, _)| *stream)
                        .collect::<std::collections::BTreeSet<_>>();
                    assert_eq!(streams.len(), RANKS as usize, "one launch per rank stream");
                    // The drafter's history: one slot of its declared buffer.
                    let piece = model_mimo26_contract::state::request_state(
                        Tp::TP4,
                        &KV_FORMATS[0],
                        drafter,
                    )
                    .buffers[1]
                        .entry_bytes();
                    for (_, arguments) in &launches {
                        assert!(matches!(arguments[0], KernelArgumentV1::DeviceBuffer(span)
                            if span.offset == 0
                                && span.length == crate::contract::RING_STATE_BYTES_PER_RANK / 2));
                        assert!(matches!(arguments[1], KernelArgumentV1::DeviceBuffer(span)
                            if span.length == piece));
                        assert_eq!(
                            &arguments[5..],
                            &[
                                KernelArgumentV1::ScalarU32(table.len() as u32 / 8),
                                KernelArgumentV1::ScalarU64(GA_PAGE_BYTES_BF16),
                                KernelArgumentV1::ScalarU32(slot),
                                KernelArgumentV1::ScalarU32(to_pages),
                            ]
                        );
                        assert_eq!(arguments[2], KernelArgumentV1::ScalarU64(piece));
                    }
                }
            };
            let captured = check(0, 1);
            instance.capture_prefix_state(7, boundary, &pages).unwrap();
            captured(&host);
            assert_eq!(
                instance.capture_prefix_state(7, boundary + 1, &pages),
                Err(BundleAbiError::Status(BundleStatusV1::InvalidState)),
                "a capture names the committed length"
            );
            assert_eq!(
                instance.capture_prefix_state(9, boundary, &pages),
                Err(BundleAbiError::Status(BundleStatusV1::NotFound))
            );
            assert!(
                instance
                    .capture_prefix_state(7, boundary, &pages[1..])
                    .is_err()
            );

            assert!(
                instance
                    .restore_prefix_state(8, &tokens, &pages[1..])
                    .is_err()
            );
            assert!(!instance.is_poisoned());
            // Request 7 still holds slot 0.
            let restored = check(1, 0);
            instance.restore_prefix_state(8, &tokens, &pages).unwrap();
            restored(&host);
            assert!(
                instance.restore_prefix_state(8, &tokens, &pages).is_err(),
                "a request the bundle knows is not restored again"
            );
            assert!(!instance.is_poisoned());
            // Draft immediately from the restored boundary, including a wrapped window.
            *host.readback_tokens.borrow_mut() = vec![44, 45, 46, 47];
            let verify = BatchRequestV1::new(
                BatchWireVersionV1::V3,
                vec![BatchItemV1::verify(8, 4, true, vec![44])],
                vec![BatchAppendV1::new(
                    boundary,
                    4,
                    boundary + 4,
                    (0..(boundary + 4).div_ceil(PAGE_TOKENS)).collect(),
                    None,
                )],
            )
            .encode()
            .unwrap();
            let mut sink = DraftSink::default();
            instance.execute(&verify, &mut sink).unwrap();
            assert_eq!(sink.drafts[0].tokens, [45, 46, 47]);
            if dflash {
                // Catch-up gathers the restored window from slot 1's cells on
                // every rank.
                use crate::dflash::state::{HISTORY_CAPACITY, HISTORY_ROWS};
                let start = boundary.saturating_sub(HISTORY_ROWS);
                let window = (start..boundary)
                    .map(|position| crate::retain::cell(1, position, HISTORY_CAPACITY))
                    .collect::<Vec<_>>();
                assert_eq!(
                    host.histories_gathering(&window).len(),
                    4,
                    "boundary {boundary}"
                );
            }
            assert!(instance.capture_prefix_state(7, boundary, &pages).is_err());
            assert!(instance.restore_prefix_state(9, &tokens, &pages).is_err());
            instance.cancel().unwrap();
            instance
                .execute(&verify, &mut DraftSink::default())
                .unwrap();
            instance.commit_accepted(&[kept(8, 2)]).unwrap();
            instance.retire_request(7).unwrap();
            instance.retire_request(8).unwrap();
            // Reuse the source slot after it held another request's drafter state.
            instance.restore_prefix_state(8, &tokens, &pages).unwrap();
            instance
                .execute(&verify, &mut DraftSink::default())
                .unwrap();
            instance.commit_accepted(&[kept(8, 1)]).unwrap();
            instance.retire_request(8).unwrap();
            instance
                .retire_request(99)
                .expect("a request cancelled before it ran retires cleanly");
            instance.destroy().unwrap();
        }
    }
}

/// A failed snapshot copy never returns while a submitted copy may still run.
///
/// The host frees or reuses snapshot pages as soon as a capture or restore
/// fails, so every rank that received work must be drained first. A
/// synchronization that fails leaves that completion unknown: the call reports
/// INTERNAL_ERROR and the wrapper poisons the instance so no page is reused.
#[test]
fn a_failed_prefix_copy_drains_every_submitted_rank_first() {
    use runtime_bundle_sdk::{BatchAppendV1, BatchItemV1, BatchRequestV1, BatchWireVersionV1};
    let boundary = 3_u64;
    let tokens = (40..40 + boundary as u32).collect::<Vec<_>>();
    for restore in [false, true] {
        for fault in [
            PrefixFault::Upload,
            PrefixFault::Launch,
            PrefixFault::Synchronize,
        ] {
            for rank in 0..RANKS {
                let api = BundleApiV1::new_with_prefix_cache(crate::BUNDLE_MIMO26_EXTENDED_API_V1)
                    .unwrap();
                let host = Rc::new(FakeHost::default());
                host.speculation.set(Some((7, 32)));
                let mut instance = api
                    .create(
                        HostServicesTableV1::new(Rc::clone(&host)),
                        &configuration_with_mtp(true),
                    )
                    .unwrap();
                let step = BatchRequestV1::new(
                    BatchWireVersionV1::V3,
                    vec![BatchItemV1::new(7, false, 3, true, tokens.clone())],
                    vec![BatchAppendV1::new(0, 3, 3, vec![0], None)],
                )
                .encode()
                .unwrap();
                instance.execute(&step, &mut DraftSink::default()).unwrap();
                instance.commit().unwrap();
                let pages =
                    (10..10 + instance.describe_prefix_state().unwrap()).collect::<Vec<_>>();
                if restore {
                    instance.capture_prefix_state(7, boundary, &pages).unwrap();
                }
                host.prefix_fault.set(Some((fault, rank)));
                let result = if restore {
                    instance.restore_prefix_state(8, &tokens, &pages)
                } else {
                    instance.capture_prefix_state(7, boundary, &pages)
                };
                let case = format!("restore {restore} {fault:?} on rank {rank}");
                assert_eq!(host.prefix_fault.get(), None, "{case}: the fault fired");
                let in_flight = host.prefix_in_flight.borrow().len();
                if fault == PrefixFault::Synchronize {
                    assert_eq!(
                        result,
                        Err(BundleAbiError::Status(BundleStatusV1::InternalError)),
                        "{case}"
                    );
                    assert!(instance.is_poisoned(), "{case}: pages must not be reused");
                    assert_eq!(in_flight, 1, "{case}: every other rank is drained");
                } else {
                    assert_eq!(
                        result,
                        Err(BundleAbiError::Status(BundleStatusV1::HostError)),
                        "{case}: the original error"
                    );
                    assert_eq!(in_flight, 0, "{case}: a copy may still use the pages");
                    assert!(!instance.is_poisoned(), "{case}");
                    // Nothing was left half done: the same call now succeeds.
                    if restore {
                        instance.restore_prefix_state(8, &tokens, &pages).unwrap();
                    } else {
                        instance.capture_prefix_state(7, boundary, &pages).unwrap();
                    }
                    assert!(host.prefix_in_flight.borrow().is_empty());
                }
            }
        }
    }
}

#[test]
fn zero_draft_maintenance_cpu_accounting() {
    use runtime_bundle_sdk::{BatchAppendV1, BatchItemV1, BatchRequestV1, BatchWireVersionV1};
    let mut ar = [0; 4];
    for concurrency in [1, 4, 16, 32] {
        for drafter in ["none", "mtp", "dflash"] {
            let api =
                BundleApiV1::new_with_prefix_cache(crate::BUNDLE_MIMO26_EXTENDED_API_V1).unwrap();
            let host = Rc::new(FakeHost::default());
            if drafter != "none" {
                host.speculation.set(Some((7, 32)));
            }
            host.dflash.set(drafter == "dflash");
            let mut instance = api
                .create(
                    HostServicesTableV1::new(Rc::clone(&host)),
                    &configuration_with_mtp(drafter != "none"),
                )
                .unwrap();
            let step = |position| {
                BatchRequestV1::new(
                    BatchWireVersionV1::V3,
                    (0..concurrency)
                        .map(|item| BatchItemV1::new(item + 1, position > 0, 1, true, vec![40]))
                        .collect(),
                    (0..concurrency)
                        .map(|item| {
                            BatchAppendV1::new(position, 1, position + 1, vec![item + 1], None)
                        })
                        .collect(),
                )
                .encode()
                .unwrap()
            };
            for position in 0..3 {
                instance
                    .execute(&step(position), &mut DraftSink::default())
                    .unwrap();
                instance.commit().unwrap();
            }
            let graphs = host.graph_replays.get();
            let nodes = host.replayed_nodes.get();
            let copies = host.copies.borrow().len();
            let uploads = host.host_copies.borrow().len();
            let started = std::time::Instant::now();
            for position in 3..11 {
                let mut sink = DraftSink::default();
                instance.execute(&step(position), &mut sink).unwrap();
                assert!(sink.drafts.is_empty());
                instance.commit().unwrap();
            }
            let elapsed = started.elapsed().as_micros() / 8;
            let copy_bytes: u64 = host.copies.borrow()[copies..]
                .iter()
                .map(|(_, _, bytes)| bytes)
                .sum();
            let counts = [
                (host.graph_replays.get() - graphs) / 8,
                ((host.replayed_nodes.get() - nodes) / 8) as u32,
                ((host.copies.borrow().len() - copies) / 8) as u32,
                ((host.host_copies.borrow().len() - uploads) / 8) as u32,
            ];
            eprintln!(
                "K0 {drafter} C{concurrency}: graphs={} nodes={} copies={} bytes={} uploads={} fake_host_us={elapsed}",
                counts[0],
                counts[1],
                counts[2],
                copy_bytes / 8,
                counts[3]
            );
            // A K=0 step matches AR except for the kernels that retain target
            // rows: one per rank for MTP, two tap kernels per rank for DFlash.
            let retain_nodes = match drafter {
                "none" => {
                    ar = counts;
                    0
                }
                "mtp" => 4,
                _ => 8,
            };
            assert_eq!(
                counts,
                [ar[0], ar[1] + retain_nodes, 0, ar[3]],
                "{drafter} C{concurrency}"
            );
            instance.destroy().unwrap();
        }
    }
}

#[test]
fn mtp_zero_draft_spans_resume_after_wrap_cancel_and_partial_commit() {
    use runtime_bundle_sdk::{BatchAppendV1, BatchItemV1, BatchRequestV1, BatchWireVersionV1};
    let api = BundleApiV1::new_with_prefix_cache(crate::BUNDLE_MIMO26_EXTENDED_API_V1).unwrap();
    let host = Rc::new(FakeHost::default());
    host.speculation.set(Some((7, 32)));
    let mut instance = api
        .create(
            HostServicesTableV1::new(Rc::clone(&host)),
            &configuration_with_mtp(true),
        )
        .unwrap();
    let step = |position, verify| {
        let count = if verify { 4 } else { 1 };
        BatchRequestV1::new(
            BatchWireVersionV1::V3,
            vec![if verify {
                BatchItemV1::verify(7, count, true, vec![40])
            } else {
                BatchItemV1::new(7, position > 0, 1, true, vec![40])
            }],
            vec![BatchAppendV1::new(
                position,
                u64::from(count),
                position + u64::from(count),
                (0..(position + u64::from(count)).div_ceil(PAGE_TOKENS)).collect(),
                None,
            )],
        )
        .encode()
        .unwrap()
    };
    for position in 0..255 {
        if position == 0 {
            instance
                .execute(&step(position, false), &mut DraftSink::default())
                .unwrap();
            instance.cancel().unwrap();
        }
        let graphs = host.graph_replays.get();
        let copies = host.copies.borrow().len();
        instance
            .execute(&step(position, false), &mut DraftSink::default())
            .unwrap();
        instance.commit().unwrap();
        if position > 1 {
            // Only the target's graphs; they retain the hidden rows.
            assert_eq!(host.graph_replays.get() - graphs, 12);
            assert_eq!(host.copies.borrow().len() - copies, 0);
        }
    }
    assert_eq!(
        count(
            &host.launch_symbols.borrow(),
            "chop_row_copy_column_slice_v1"
        ),
        0
    );
    // One retained-row kernel per rank in the decode graph, addressing the
    // history with the capacity catch-up uses.
    let retained = host.retain_launches.borrow().clone();
    assert_eq!(retained.len(), 4);
    assert!(
        retained
            .iter()
            .all(
                |&(_, capacity, width)| u64::from(capacity) == crate::prefix::MTP_SLOT_ROWS
                    && width == 4096
            )
    );
    *host.readback_tokens.borrow_mut() = vec![40, 41, 42, 43];
    let copies = host.copies.borrow().len();
    instance
        .execute(&step(255, true), &mut DraftSink::default())
        .unwrap();
    // The first draft rebuilds rows 127..254 from their cells (127 % 136 =
    // 127: 9 rows, then 118 from the start) and runs row 254 (cell 118).
    let row = 4096 * 2;
    let rebuilt = host.copies.borrow()[copies..]
        .iter()
        .filter(|&&(source, _, bytes)| {
            bytes == 9 * row && source == 127 * row
                || bytes == 118 * row && source == 0
                || bytes == row && source == 118 * row
        })
        .count();
    assert_eq!(rebuilt, 12);
    instance.cancel().unwrap();
    instance
        .execute(&step(255, true), &mut DraftSink::default())
        .unwrap();
    instance.commit_accepted(&[kept(7, 2)]).unwrap();
    instance
        .execute(&step(257, false), &mut DraftSink::default())
        .unwrap();
    instance.cancel().unwrap();
    instance
        .execute(&step(257, true), &mut DraftSink::default())
        .unwrap();
    instance.commit_accepted(&[kept(7, 1)]).unwrap();
    instance.retire_request(7).unwrap();
    instance
        .execute(&step(0, false), &mut DraftSink::default())
        .unwrap();
    instance.commit().unwrap();
    instance.destroy().unwrap();
}

#[test]
fn dflash_zero_draft_history_wraps_and_only_committed_rows_are_projected() {
    use runtime_bundle_sdk::{BatchAppendV1, BatchItemV1, BatchRequestV1, BatchWireVersionV1};
    let (host, mut instance) = dflash_instance();
    instance
        .execute(
            &dflash_step(0, vec![40; 1024], None),
            &mut DraftSink::default(),
        )
        .unwrap();
    instance.commit().unwrap();
    for position in 1024..3068 {
        let request = BatchRequestV1::new(
            BatchWireVersionV1::V3,
            vec![BatchItemV1::new(7, true, 1, true, vec![40])],
            vec![BatchAppendV1::new(
                position,
                1,
                position + 1,
                (0..(position + 1).div_ceil(PAGE_TOKENS)).collect(),
                None,
            )],
        )
        .encode()
        .unwrap();
        if position == 1500 {
            instance
                .execute(&request, &mut DraftSink::default())
                .unwrap();
            instance.cancel().unwrap();
        }
        let graphs = host.graph_replays.get();
        let copies = host.copies.borrow().len();
        instance
            .execute(&request, &mut DraftSink::default())
            .unwrap();
        instance.commit().unwrap();
        // The target graphs write the taps into the history: eight kernels
        // across the four ranks and no commit copies.
        assert_eq!(host.graph_replays.get() - graphs, 12);
        assert_eq!(host.copies.borrow().len() - copies, 0);
    }
    // Decode taps address each rank's history with the capacity that
    // catch-up and prefill commits use: two tap kernels per rank.
    let capacity = crate::dflash::state::HISTORY_CAPACITY;
    let retained = host.retain_launches.borrow().clone();
    assert_eq!(retained.len(), 8);
    assert!(
        retained
            .iter()
            .all(|&(_, launch_capacity, _)| u64::from(launch_capacity) == capacity)
    );
    let histories = retained
        .iter()
        .map(|(history, _, _)| history.buffer.slot())
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(histories.len(), 4);
    assert_eq!(
        count(
            &host.launch_symbols.borrow(),
            "chop_mimo_dflash_cache_write_v1"
        ),
        0
    );
    *host.readback_tokens.borrow_mut() = vec![40, 41, 42, 43];
    let before = host.launch_symbols.borrow().len();
    let copies = host.copies.borrow().len();
    instance
        .execute(
            &dflash_step(3068, vec![40], Some(4)),
            &mut DraftSink::default(),
        )
        .unwrap();
    let launched = host.launch_symbols.borrow()[before..].to_vec();
    assert_eq!(count(&launched, "chop_mimo_dflash_cache_write_v1"), 20);
    // Catch-up gathers the window 2045..3068 from the cells the decode taps
    // wrote, 2045 % 1031 = 1014: 17 rows, then 1006 from the start, on
    // every rank, with no copies into the taps.
    let row = crate::dflash::program::fc_shard(Tp::TP4) * 2;
    assert!(
        host.copies.borrow()[copies..]
            .iter()
            .all(|&(_, destination, _)| destination >= 1023 * row)
    );
    let window = (1014..1031).chain(0..1006).collect::<Vec<u64>>();
    assert_eq!(
        host.histories_gathering(&window)
            .into_iter()
            .map(|raw| raw as u32)
            .collect::<std::collections::BTreeSet<_>>(),
        histories
    );
    instance.cancel().unwrap();
    let before = host.graph_replays.get();
    // Projection of committed history survives cancellation. No context
    // projection graph is necessary on the retry.
    instance
        .execute(
            &dflash_step(3068, vec![40], Some(4)),
            &mut DraftSink::default(),
        )
        .unwrap();
    assert_eq!(host.graph_replays.get() - before, 12);
    instance.commit_accepted(&[kept(7, 2)]).unwrap();
    instance
        .execute(
            &dflash_step(3070, vec![40], Some(4)),
            &mut DraftSink::default(),
        )
        .unwrap();
    instance.commit_accepted(&[kept(7, 1)]).unwrap();
    instance.retire_request(7).unwrap();
    instance
        .execute(&dflash_step(0, vec![40], None), &mut DraftSink::default())
        .unwrap();
    instance.commit().unwrap();
    let before = host.graph_replays.get();
    instance
        .execute(
            &dflash_step(1, vec![40], Some(4)),
            &mut DraftSink::default(),
        )
        .unwrap();
    assert_eq!(host.graph_replays.get() - before, 20);
    instance.cancel().unwrap();
    let captured = host.graph_ends.get();
    instance.destroy().unwrap();
    assert_eq!(host.graph_releases.get(), captured);
}

/// Collective launches each rank's stream has queued, eagerly or in launched
/// graphs, when every rank queued the same number.
fn matched_collectives(host: &FakeHost) -> Option<usize> {
    let counts = host.collectives.borrow();
    let first = *counts.values().next()?;
    (counts.len() == RANKS as usize && counts.values().all(|&count| count == first))
        .then_some(first)
}

/// Device-facing journal lines: work queued on a stream or a buffer.
fn queues_device_work(line: &str) -> bool {
    [
        "launch ",
        "replay ",
        "copy ",
        "peer copy ",
        "upload ",
        "zero ",
        "record ",
        "wait ",
        "synchronize ",
        "readback ",
    ]
    .iter()
    .any(|prefix| line.starts_with(prefix))
}

fn fault_instance() -> (Rc<FakeHost>, runtime_bundle_abi::BundleInstanceV1) {
    let api = BundleApiV1::new_with_prefix_cache(crate::BUNDLE_MIMO26_EXTENDED_API_V1)
        .expect("bundle API");
    let backend = Rc::new(FakeHost::default());
    let instance = api
        .create(
            HostServicesTableV1::new(Rc::clone(&backend)),
            &configuration(),
        )
        .expect("fake-host create");
    (backend, instance)
}

/// A batch past the prefill arena is refused before any device work, the
/// log names the check and the batch shape, and the instance runs on.
#[test]
fn a_refused_batch_is_logged_with_its_shape_and_leaves_the_instance_usable() {
    use runtime_bundle_sdk::{BatchAppendV1, BatchItemV1, BatchRequestV1, BatchWireVersionV1};
    let (backend, mut instance) = fault_instance();
    let (items, appends): (Vec<_>, Vec<_>) = [2048_u32, 2048, 2048, 2048, 117]
        .into_iter()
        .enumerate()
        .map(|(index, tokens)| {
            let pages = u64::from(tokens).div_ceil(PAGE_TOKENS);
            (
                BatchItemV1::new(
                    60 + index as u64,
                    false,
                    tokens,
                    true,
                    vec![1; tokens as usize],
                ),
                BatchAppendV1::new(
                    0,
                    u64::from(tokens),
                    u64::from(tokens),
                    (0..pages).map(|page| page % 128).collect(),
                    None,
                ),
            )
        })
        .unzip();
    let oversized = BatchRequestV1::new(BatchWireVersionV1::V3, items, appends);
    let queued = backend.journal.borrow().len();
    let refused = instance.execute(&oversized.encode().unwrap(), &mut ViewSink::default());
    assert_eq!(
        refused.err(),
        Some(BundleAbiError::Status(BundleStatusV1::InvalidArgument))
    );
    assert!(
        backend.journal.borrow()[queued..]
            .iter()
            .all(|line| !queues_device_work(line))
    );
    let logs = backend.logs.borrow();
    let line = logs.last().expect("the refusal is logged");
    assert!(
        line.starts_with(
            "MiMo execute refused a batch with status 1: invalid batch; \
             5 items (at most 32); prefill: 5 items, 8309 tokens (at most 8192), largest 2048;"
        ),
        "{line}"
    );
    drop(logs);
    instance
        .execute(&fault_prefill(), &mut ViewSink::default())
        .expect("a batch within the limits runs after the refusal");
    instance.commit().expect("commit");
}

fn fault_decode(previous: u64, token: u32) -> Vec<u8> {
    use runtime_bundle_sdk::{BatchAppendV1, BatchItemV1, BatchRequestV1, BatchWireVersionV1};
    BatchRequestV1::new(
        BatchWireVersionV1::V2,
        vec![BatchItemV1::new(41, true, 1, true, vec![token])],
        vec![BatchAppendV1::new(previous, 1, previous + 1, vec![1], None)],
    )
    .encode()
    .unwrap()
}

fn fault_prefill() -> Vec<u8> {
    use runtime_bundle_sdk::{BatchAppendV1, BatchItemV1, BatchRequestV1, BatchWireVersionV1};
    BatchRequestV1::new(
        BatchWireVersionV1::V2,
        vec![BatchItemV1::new(50, false, 3, true, vec![1, 2, 3])],
        vec![BatchAppendV1::new(0, 3, 3, vec![1], None)],
    )
    .encode()
    .unwrap()
}

/// After a poisoning failure: nothing reached a stream after the failed
/// submission, every later call is refused, and destroy keeps every
/// resource for process-level recovery.
fn assert_poisoned(
    backend: &FakeHost,
    instance: &mut runtime_bundle_abi::BundleInstanceV1,
    journal: usize,
) {
    let added = backend.journal.borrow()[journal..].to_vec();
    let last = added
        .iter()
        .rposition(|line| line.starts_with("launch ") || line.starts_with("replay "))
        .expect("the step submitted work");
    assert!(
        !added[last + 1..]
            .iter()
            .any(|line| queues_device_work(line)),
        "work queued behind unmatched collectives: {:?}",
        &added[last + 1..]
    );
    let releases = backend.released.get();
    let graph_releases = backend.graph_releases.get();
    assert_eq!(
        instance.execute(&fault_decode(0, 7), &mut ViewSink::default()),
        Err(BundleAbiError::Status(BundleStatusV1::InvalidState))
    );
    assert!(instance.retire_request(41).is_err());
    assert!(instance.destroy().is_err());
    assert!(instance.is_poisoned());
    assert_eq!(backend.released.get(), releases);
    assert_eq!(backend.graph_releases.get(), graph_releases);
    assert_eq!(backend.journal.borrow().len(), journal + added.len());
}

/// A decode replay that fails after two ranks queued their first piece
/// leaves them waiting at collectives the other ranks never join.
#[test]
fn a_partial_graph_replay_poisons_the_instance() {
    let (backend, mut instance) = fault_instance();
    instance
        .execute(&fault_decode(0, 7), &mut ViewSink::default())
        .expect("capture and run the decode graph");
    instance.commit().expect("commit");
    assert!(matched_collectives(&backend).is_some_and(|count| count > 0));
    backend.fail_graph_launch.set(Some(2));
    let journal = backend.journal.borrow().len();
    assert_eq!(
        instance.execute(&fault_decode(1, 8), &mut ViewSink::default()),
        Err(BundleAbiError::Status(BundleStatusV1::HostError))
    );
    assert_eq!(matched_collectives(&backend), None);
    assert_poisoned(&backend, &mut instance, journal);
}

/// A 64-row verify replay, whose hidden all-reduces run the wide entry,
/// that fails after two ranks queued their first piece leaves them waiting
/// at collectives the other ranks never join, as a decode replay does.
#[test]
fn a_partial_wide_verify_replay_poisons_the_instance() {
    use runtime_bundle_sdk::{BatchAppendV1, BatchItemV1, BatchRequestV1, BatchWireVersionV1};
    let (backend, mut instance) = wide_verify_instance((7, 256), 5);
    // 40 rows over five items take the (64, 8) graph.
    instance
        .execute(&wide_verify_step(5, 8), &mut ViewSink::default())
        .expect("capture and run the wide verify graph");
    instance
        .commit_accepted(&(0..5).map(|item| kept(100 + item, 1)).collect::<Vec<_>>())
        .expect("keep one row each");
    let wide = std::env::var(crate::switches::WIDE_COMM).as_deref() != Ok("0");
    assert_eq!(
        backend
            .launch_symbols
            .borrow()
            .iter()
            .any(|symbol| symbol == crate::sm120::comm::WIDE_SYMBOL),
        wide
    );
    let step = BatchRequestV1::new(
        BatchWireVersionV1::V3,
        (0..5)
            .map(|item| BatchItemV1::verify(100 + item, 7, false, vec![3; 7]))
            .collect(),
        (0..5)
            .map(|item| BatchAppendV1::new(4, 7, 11, vec![item], None))
            .collect(),
    )
    .encode()
    .unwrap();
    backend.fail_graph_launch.set(Some(2));
    let journal = backend.journal.borrow().len();
    assert_eq!(
        instance.execute(&step, &mut ViewSink::default()),
        Err(BundleAbiError::Status(BundleStatusV1::HostError))
    );
    assert_eq!(matched_collectives(&backend), None);
    assert_poisoned(&backend, &mut instance, journal);
}

/// Every rank's decode graph is captured before any rank launches, so a
/// failed capture leaves nothing queued and the instance usable.
#[test]
fn a_failed_capture_queues_nothing_and_keeps_the_instance() {
    let (probe, mut clean) = fault_instance();
    clean
        .execute(&fault_decode(0, 7), &mut ViewSink::default())
        .expect("clean decode");
    let pieces = probe.graph_ends.get() / RANKS;
    let (backend, mut instance) = fault_instance();
    // Rank 0's pieces are captured; rank 1's first capture fails.
    backend.fail_graph_end.set(Some(pieces));
    assert!(
        instance
            .execute(&fault_decode(0, 7), &mut ViewSink::default())
            .is_err()
    );
    assert!(backend.graph_launches.borrow().is_empty());
    assert!(backend.collectives.borrow().is_empty());
    assert!(!instance.is_poisoned());
    instance
        .execute(&fault_decode(0, 7), &mut ViewSink::default())
        .expect("the same step captures and runs");
    instance.commit().expect("commit");
    assert!(matched_collectives(&backend).is_some_and(|count| count > 0));
    instance.destroy().expect("destroy");
}

/// An eager prefill stage that fails on rank 1 after rank 0 queued its
/// all-reduce leaves rank 0 waiting at it.
#[test]
fn a_partial_prefill_stage_poisons_the_instance() {
    // Find rank 1's first collective launch in a clean run.
    let (probe, mut clean) = fault_instance();
    let before = probe.stream_launches.borrow().len();
    clean
        .execute(&fault_prefill(), &mut ViewSink::default())
        .expect("clean prefill");
    let launches = probe.stream_launches.borrow()[before..].to_vec();
    let first = launches
        .iter()
        .position(|(_, name, _)| name.starts_with("chop_comm_"))
        .expect("a collective");
    let peer = launches[first..]
        .iter()
        .position(|(stream, name, _)| {
            name.starts_with("chop_comm_") && *stream != launches[first].0
        })
        .expect("a peer's collective")
        + first;
    let (backend, mut instance) = fault_instance();
    backend.fail_launch.set(Some(peer as u32));
    let journal = backend.journal.borrow().len();
    assert_eq!(
        instance.execute(&fault_prefill(), &mut ViewSink::default()),
        Err(BundleAbiError::Status(BundleStatusV1::HostError))
    );
    assert_eq!(matched_collectives(&backend), None);
    assert_poisoned(&backend, &mut instance, journal);
}

/// A pipelined prefill whose event record or wait fails after an all-reduce
/// reached every comm stream, but before compute waited for it, drains every
/// rank's streams before cancel's restore copies or a retry reuse the buffers
/// that all-reduce still reads and writes.
#[test]
fn a_failed_overlap_event_drains_every_stream_before_reuse() {
    use runtime_bundle_sdk::{BatchAppendV1, BatchItemV1, BatchRequestV1, BatchWireVersionV1};
    let request = BatchRequestV1::new(
        BatchWireVersionV1::V2,
        vec![BatchItemV1::new(77, false, 2048, true, vec![7; 2048])],
        vec![BatchAppendV1::new(0, 2048, 2048, (0..32).collect(), None)],
    )
    .encode()
    .unwrap();
    let pipelined = |instance: &runtime_bundle_abi::BundleInstanceV1| {
        crate::instance::force_overlap(
            instance.handle(),
            Some(crate::overlap::Config {
                sms: 188,
                reserved_sms: 16,
                comm_ctas: Some(24),
                min_tokens: 1024,
            }),
        );
    };
    // In a clean run, find the first comm record (rank 0's half-0 attention
    // all-reduce done) and the first wait of a compute stream for one.
    let (probe, mut clean) = fault_instance();
    pipelined(&clean);
    let comm = crate::instance::comm_streams(clean.handle());
    clean
        .execute(&request, &mut ViewSink::default())
        .expect("clean pipelined prefill");
    let ops = probe.event_ops.borrow().clone();
    let first = |wait: bool, on_comm: bool| {
        ops.iter()
            .filter(|op| op.2 == wait)
            .position(|op| comm.contains(&op.0) == on_comm)
            .expect("an event operation") as u32
    };
    // The record fails on rank 0, the wait on rank 3.
    for (wait, index) in [(false, first(false, true)), (true, first(true, false) + 3)] {
        let (backend, mut instance) = fault_instance();
        pipelined(&instance);
        let comm = crate::instance::comm_streams(instance.handle());
        if wait {
            backend.fail_wait.set(Some(index));
        } else {
            backend.fail_record.set(Some(index));
        }
        let journal = backend.journal.borrow().len();
        assert_eq!(
            instance.execute(&request, &mut ViewSink::default()),
            Err(BundleAbiError::Status(BundleStatusV1::HostError)),
            "wait {wait}"
        );
        assert!(!instance.is_poisoned());
        let added = backend.journal.borrow()[journal..].to_vec();
        let failed = added
            .iter()
            .rposition(|line| {
                ["launch ", "record ", "wait "]
                    .iter()
                    .any(|op| line.starts_with(op))
            })
            .expect("the schedule submitted work");
        let after = &added[failed + 1..];
        let reuse = after
            .iter()
            .position(|line| queues_device_work(line) && !line.starts_with("synchronize "))
            .expect("cancel restores the ring");
        let compute = backend
            .event_ops
            .borrow()
            .iter()
            .map(|op| op.0)
            .filter(|stream| !comm.contains(stream))
            .collect::<BTreeSet<_>>();
        for stream in comm.iter().chain(&compute) {
            assert!(
                after[..reuse].contains(&format!("synchronize {stream}")),
                "wait {wait}: stream {stream} not drained before {:?}",
                after[reuse]
            );
        }
        instance
            .execute(&request, &mut ViewSink::default())
            .expect("the same step runs again");
        instance.commit().expect("commit");
        instance.destroy().expect("destroy");
    }
    // Streams that cannot be drained poison the instance, which then
    // restores nothing and refuses every later call.
    let (backend, mut instance) = fault_instance();
    pipelined(&instance);
    backend.fail_record.set(Some(first(false, true)));
    backend
        .prefix_fault
        .set(Some((PrefixFault::Synchronize, 0)));
    let copies = backend.copies.borrow().len();
    assert_eq!(
        instance.execute(&request, &mut ViewSink::default()),
        Err(BundleAbiError::Status(BundleStatusV1::HostError))
    );
    assert_eq!(backend.copies.borrow().len(), copies);
    assert_eq!(
        instance.execute(&fault_decode(0, 7), &mut ViewSink::default()),
        Err(BundleAbiError::Status(BundleStatusV1::InvalidState))
    );
    assert!(instance.destroy().is_err());
}

/// Launches per stream and entry of the steps that capture MiMo's decode
/// buckets 1, 16 and 32, DFlash verify at C1 (K7) and C16 (K1 each), and one
/// 8,192-token prefill chunk, for per-step launch budgets:
/// `cargo test -p bundle-mimo26 --lib launch_census -- --ignored --nocapture`.
#[test]
#[ignore = "prints a report"]
fn launch_census() {
    use runtime_bundle_sdk::{BatchAppendV1, BatchItemV1, BatchRequestV1, BatchWireVersionV1};
    fn report(label: &str, host: &FakeHost, from: usize) {
        let mut streams = BTreeMap::<u32, BTreeMap<String, usize>>::new();
        for (stream, symbol, _) in &host.stream_launches.borrow()[from..] {
            *streams
                .entry(*stream)
                .or_default()
                .entry(symbol.clone())
                .or_default() += 1;
        }
        let totals = streams
            .values()
            .map(|symbols| symbols.values().sum::<usize>())
            .collect::<Vec<_>>();
        eprintln!("== {label}: launches per stream {totals:?}");
        if let Some(symbols) = streams.values().next() {
            for (symbol, count) in symbols {
                eprintln!("   {count:5} {symbol}");
            }
        }
    }
    let api = || BundleApiV1::new_with_prefix_cache(crate::BUNDLE_MIMO26_EXTENDED_API_V1).unwrap();
    let items = |live: u64, position: u64, verify: Option<u32>| {
        BatchRequestV1::new(
            BatchWireVersionV1::V3,
            (1..=live)
                .map(|id| match verify {
                    Some(rows) => BatchItemV1::verify(id, rows, true, vec![43]),
                    None => BatchItemV1::new(id, position > 0, 1, true, vec![40]),
                })
                .collect(),
            (1..=live)
                .map(|id| {
                    let rows = u64::from(verify.unwrap_or(1));
                    BatchAppendV1::new(position, rows, position + rows, vec![id], None)
                })
                .collect(),
        )
        .encode()
        .unwrap()
    };
    for live in [1, 16, 32] {
        let host = Rc::new(FakeHost::default());
        let mut instance = api()
            .create(
                HostServicesTableV1::new(Rc::clone(&host)),
                &configuration_with_mtp(false),
            )
            .unwrap();
        instance
            .execute(&items(live, 0, None), &mut ViewSink::default())
            .unwrap();
        instance.commit().unwrap();
        let from = host.stream_launches.borrow().len();
        instance
            .execute(&items(live, 1, None), &mut ViewSink::default())
            .unwrap();
        instance.commit().unwrap();
        report(&format!("decode C{live}"), &host, from);
        eprintln!("   graph nodes {:?}", host.graph_nodes.borrow().values());
        instance.destroy().unwrap();
    }
    for (live, rows) in [(1, 8), (16, 2)] {
        let (host, mut instance) = dflash_instance();
        for position in 0..3 {
            instance
                .execute(&items(live, position, None), &mut DraftSink::default())
                .unwrap();
            instance.commit().unwrap();
        }
        *host.readback_tokens.borrow_mut() = vec![44; (live * u64::from(rows)) as usize];
        let from = host.stream_launches.borrow().len();
        instance
            .execute(&items(live, 3, Some(rows)), &mut DraftSink::default())
            .unwrap();
        report(&format!("DFlash verify C{live} rows {rows}"), &host, from);
        instance.cancel().unwrap();
        instance.destroy().unwrap();
    }
    let host = Rc::new(FakeHost::default());
    let mut instance = api()
        .create(
            HostServicesTableV1::new(Rc::clone(&host)),
            &configuration_with_mtp(false),
        )
        .unwrap();
    let tokens = PREFILL_TOKENS;
    let prefill = BatchRequestV1::new(
        BatchWireVersionV1::V3,
        vec![BatchItemV1::new(
            1,
            false,
            tokens,
            true,
            vec![40; tokens as usize],
        )],
        vec![BatchAppendV1::new(
            0,
            u64::from(tokens),
            u64::from(tokens),
            (0..u64::from(tokens).div_ceil(PAGE_TOKENS)).collect(),
            None,
        )],
    );
    let from = host.stream_launches.borrow().len();
    instance
        .execute(&prefill.encode().unwrap(), &mut ViewSink::default())
        .unwrap();
    report("prefill 8192", &host, from);
    instance.commit().unwrap();
    instance.destroy().unwrap();
}

#[path = "sm120/invariance_journal_tests.rs"]
mod invariance_journal;

#[path = "launch_journal_tests.rs"]
mod launch_journal;

#[path = "taps_tests.rs"]
mod taps;

#[test]
fn malformed_text_indexes_fail_before_any_weight_upload() {
    let api = BundleApiV1::new_with_prefix_cache(crate::BUNDLE_MIMO26_EXTENDED_API_V1).unwrap();
    for mutation in 1..=8 {
        let host = Rc::new(FakeHost::default());
        host.bad_text_index.set(mutation);
        assert!(
            api.create(HostServicesTableV1::new(Rc::clone(&host)), &configuration())
                .is_err(),
            "mutation {mutation}"
        );
        assert!(
            host.journal.borrow().iter().all(|call| call == "release"),
            "device calls before admission: {:?}",
            host.journal.borrow()
        );
    }
}

/// Every text and encoder rank index is bound to the image the host serves
/// for its rank before that image is opened: an index that names another
/// image opens nothing of it, and the log names the rank.
#[test]
fn rank_indexes_naming_another_image_fail_before_the_image_opens() {
    let api = BundleApiV1::new_with_prefix_cache(crate::BUNDLE_MIMO26_EXTENDED_API_V1).unwrap();
    for tp in [Tp::TP2, Tp::TP4] {
        let ordinals = (0..tp.ranks() as i32).collect::<Vec<_>>();
        let config = configuration_on_with_encoders(
            tp,
            &crate::contract::KV_FORMATS[0],
            false,
            &ordinals,
            32,
            true,
        );
        let last = tp.ranks() - 1;
        for encoder in [false, true] {
            let host = Rc::new(FakeHost::at(tp));
            if encoder {
                host.bad_encoder_identity.set(true);
            } else {
                // Every text index names a file digest the host does not serve.
                host.bad_text_index.set(6);
            }
            let case = format!("tp{} encoder={encoder}", tp.ranks());
            assert!(
                api.create(HostServicesTableV1::new(Rc::clone(&host)), &config)
                    .is_err(),
                "{case}"
            );
            assert!(host.encoder_images.borrow().is_empty(), "{case}");
            let line = if encoder {
                // The text image opened and uploaded before the encoders.
                assert_eq!(host.text_images.get(), 1, "{case}");
                format!("MiMo encoder rank {last} index names another image")
            } else {
                assert_eq!(host.text_images.get(), 0, "{case}");
                assert!(
                    host.journal.borrow().iter().all(|call| call == "release"),
                    "{case}: device calls before the identity check: {:?}",
                    host.journal.borrow()
                );
                "MiMo rank 0 index names another image".to_owned()
            };
            assert!(
                host.logs.borrow().contains(&line),
                "{case}: {:?}",
                host.logs.borrow()
            );
        }
    }
}

fn fake_image_identity(rank: u32, tp: Tp) -> (String, Vec<u8>) {
    (
        format!("chop.compiled-image-identity.v1/mimo.rank-images.v1/{rank}"),
        serde_json::to_vec(
            &serde_json::json!({"byte_length": fake_image_bytes(tp), "sha256": "a".repeat(64)}),
        )
        .unwrap(),
    )
}

#[test]
fn drafter_identity_mismatches_fail_before_drafter_upload() {
    let api = BundleApiV1::new_with_prefix_cache(crate::BUNDLE_MIMO26_EXTENDED_API_V1).unwrap();
    for dflash in [false, true] {
        for mutation in 1..=2 {
            let host = Rc::new(FakeHost::default());
            host.dflash.set(dflash);
            host.bad_drafter_identity.set(mutation);
            host.speculation.set(Some((7, 32)));
            let result = api.create(
                HostServicesTableV1::new(Rc::clone(&host)),
                &configuration_with_mtp(true),
            );
            assert!(result.is_err(), "dflash={dflash}, mutation={mutation}");
            assert_eq!(
                host.drafter_uploads.get(),
                0,
                "no drafter rank may upload before all identities match"
            );
        }
    }
}

#[path = "drafter_rollback_tests.rs"]
mod drafter_rollback;

#[test]
fn decode_buckets_route_by_their_capacity() {
    use crate::selection::Phase;
    let api = BundleApiV1::new_with_prefix_cache(crate::BUNDLE_MIMO26_EXTENDED_API_V1)
        .expect("bundle API");
    let backend = Rc::new(FakeHost::default());
    // Verify inputs exist only when the host enables verify steps.
    backend.speculation.set(Some((7, 32)));
    let table = HostServicesTableV1::new(backend);
    let mut instance = api
        .create(table, &configuration())
        .expect("fake-host create");
    for rows in [1, 8, 9, 16, 17, 32] {
        let capacity = crate::graphs::decode_bucket(rows).unwrap();
        for phase in [Phase::Decode(rows), Phase::Verify(rows)] {
            let symbols = crate::program::stages()
                .into_iter()
                .flat_map(|stage| {
                    crate::instance::plan_stage_probe(instance.handle(), phase, stage)
                        .expect("lower stage")
                })
                .collect::<Vec<_>>();
            // Buckets above 16 rows route their experts as prefill does;
            // buckets of up to 8 rows run the fused router and route, and
            // 16 the router GEMM and the decode route.
            let prefill = capacity > 16;
            let fused = capacity <= 8;
            let count = |prefix: &str| symbols.iter().filter(|s| s.starts_with(prefix)).count();
            assert_eq!(
                symbols
                    .iter()
                    .filter(|s| s.contains("permute_tile_scatter"))
                    .count(),
                if prefill { 47 } else { 0 },
                "{phase:?}"
            );
            assert_eq!(
                count(model_mimo26_contract::sm120::moe::FUSED_ROUTER_PREFIX),
                if fused { 47 } else { 0 }
            );
            assert_eq!(
                count(model_mimo26_contract::sm120::moe::DECODE_ROUTE_SYMBOL),
                if prefill || fused { 0 } else { 47 }
            );
            assert_eq!(count("chop_gemm_mimo_router_"), if fused { 0 } else { 47 });
        }
    }
    instance.destroy().expect("fake-host destroy");
}

#[test]
fn experts_above_16_rows_route_as_prefill() {
    let ranks = RANKS as usize;
    let moe_layers = crate::program::LAYERS as usize - 1;
    for (items, rows, bucket) in [(32, 8, 256), (16, 8, 128), (5, 8, 64)] {
        let (backend, mut instance) = wide_verify_instance((7, 256), items);
        let symbols_before = backend.launch_symbols.borrow().len();
        instance
            .execute(&wide_verify_step(items, rows), &mut ViewSink::default())
            .expect("wide verify step");
        let launched = backend.launch_symbols.borrow()[symbols_before..].to_vec();
        let count = |name: &str| {
            launched
                .iter()
                .filter(|symbol| symbol.contains(name))
                .count()
        };
        for projection in ["gateup", "down"] {
            assert_eq!(
                count(&format!("chop_moe_mimo_{projection}_prefill_mxfp4_a16")),
                moe_layers * ranks,
                "{projection} at {bucket} rows"
            );
        }
        assert_eq!(count("_verify_mxfp4_a16_t"), 0);
        // The route, combine and all-reduce keep the prefill lowering.
        assert_eq!(count("permute_tile_scatter"), moe_layers * ranks);
        assert_eq!(count("chop_row_combine_mimo4096_v1"), moe_layers * ranks);
        assert_eq!(count("_decode_mxfp4_a16_t"), 0);
        instance.cancel().expect("cancel");
        instance.destroy().expect("destroy");
    }
    // Buckets of 1 to 16 rows keep the decode route and their decode
    // entries, or exact verify rows their exact entries; 32 rows routes as
    // prefill does.
    let (backend, mut instance) = wide_verify_instance((7, 256), 32);
    let symbols_before = backend.launch_symbols.borrow().len();
    for rows in [1, 2, 4, 8, 16, 32] {
        for phase in [
            crate::selection::Phase::Decode(rows),
            crate::selection::Phase::Verify(rows),
        ] {
            let symbols = crate::program::stages()
                .into_iter()
                .flat_map(|stage| {
                    crate::instance::plan_stage_probe(instance.handle(), phase, stage)
                        .expect("lower stage")
                })
                .collect::<Vec<_>>();
            let count = |name: &str| symbols.iter().filter(|s| s.contains(name)).count();
            let prefill = rows > 16;
            assert_eq!(count("_verify_mxfp4_a16_t"), 0);
            for projection in ["gateup", "down"] {
                assert_eq!(
                    count(&format!("chop_moe_mimo_{projection}_prefill_mxfp4_a16")),
                    if prefill { moe_layers } else { 0 },
                    "{projection} {phase:?}"
                );
                assert_eq!(
                    count(&format!(
                        "chop_moe_mimo_{projection}_decode_mxfp4_a16_t{rows}_v1"
                    )) + count(&format!(
                        "chop_moe_mimo_{projection}_exact_mxfp4_a16_t{rows}_v1"
                    )),
                    if prefill { 0 } else { moe_layers },
                    "{projection} {phase:?}"
                );
            }
            assert_eq!(
                count("permute_tile_scatter"),
                if prefill { moe_layers } else { 0 }
            );
            assert_eq!(
                count(model_mimo26_contract::sm120::moe::DECODE_ROUTE_SYMBOL)
                    + count(model_mimo26_contract::sm120::moe::FUSED_ROUTER_PREFIX),
                if prefill { 0 } else { moe_layers },
                "{phase:?}"
            );
        }
    }
    assert_eq!(backend.launch_symbols.borrow().len(), symbols_before);
    instance.destroy().expect("destroy");
}

#[test]
fn fresh_short_prompts_catch_up_with_mtp_and_dflash() {
    for dflash in [false, true] {
        for tokens in [3, 8, 9, 10, 20, 33, 34, 39, 40] {
            let api =
                BundleApiV1::new_with_prefix_cache(crate::BUNDLE_MIMO26_EXTENDED_API_V1).unwrap();
            let host = Rc::new(FakeHost::default());
            host.speculation.set(Some((7, 32)));
            host.dflash.set(dflash);
            let mut instance = api
                .create(
                    HostServicesTableV1::new(Rc::clone(&host)),
                    &configuration_with_mtp(true),
                )
                .unwrap();
            instance
                .execute(
                    &dflash_step(0, (0..tokens as u32).collect(), None),
                    &mut DraftSink::default(),
                )
                .unwrap();
            instance.commit().unwrap();
            *host.readback_tokens.borrow_mut() = vec![43, 44, 45, 46];
            let mut sink = DraftSink::default();
            instance
                .execute(&dflash_step(tokens, vec![43], Some(4)), &mut sink)
                .unwrap_or_else(|error| panic!("dflash={dflash}, tokens={tokens}: {error:?}"));
            assert_eq!(
                sink.drafts,
                [OwnedDraftTokensV1 {
                    request_id: 7,
                    tokens: vec![44, 45, 46],
                }]
            );
            assert_eq!(sink.views.iter().map(|v| v.raw().row_count).sum::<u32>(), 4);
            instance.cancel().unwrap();
            instance.destroy().unwrap();
        }
    }
}

#[test]
fn scored_prompt_capacity_bounds_allocations_and_refuses_before_work() {
    use runtime_bundle_sdk::{BatchAppendV1, BatchItemV1, BatchRequestV1, BatchWireVersionV1};
    let api = BundleApiV1::new_with_prefix_cache(crate::BUNDLE_MIMO26_EXTENDED_API_V1).unwrap();
    let host = Rc::new(FakeHost::default());
    let instance = api
        .create(HostServicesTableV1::new(Rc::clone(&host)), &configuration())
        .unwrap();
    let local = u64::from(crate::contract::SCORED_ROWS) * u64::from(VOCAB_PER_RANK) * 4;
    assert!(host.allocations.borrow().contains(&local));
    assert!(
        host.allocations
            .borrow()
            .contains(&(local * u64::from(RANKS)))
    );
    for rows in [512_u32, 513] {
        let request = BatchRequestV1::new(
            BatchWireVersionV1::V2,
            vec![
                BatchItemV1::new(7, false, rows, true, vec![42; rows as usize])
                    .with_prompt_rows(true),
            ],
            vec![BatchAppendV1::new(
                0,
                rows as u64,
                rows as u64,
                (0..rows.div_ceil(64)).map(u64::from).collect(),
                None,
            )],
        );
        let before = host.launch_symbols.borrow().len();
        assert_eq!(
            crate::instance::preflight_request(instance.handle(), &request.encode().unwrap()),
            if rows == 512 {
                BundleStatusV1::Success
            } else {
                BundleStatusV1::Unsupported
            }
        );
        assert_eq!(host.launch_symbols.borrow().len(), before);
    }
}

#[test]
fn create_refuses_a_scored_row_limit_past_the_bundle_capacity() {
    let api = BundleApiV1::new_with_prefix_cache(crate::BUNDLE_MIMO26_EXTENDED_API_V1).unwrap();
    let capacity = crate::contract::SCORED_ROWS;
    for (reported, accepted) in [(0, true), (capacity, true), (capacity + 1, false)] {
        let host = Rc::new(FakeHost::default());
        host.scored_rows.set(reported);
        let created = api.create(HostServicesTableV1::new(Rc::clone(&host)), &configuration());
        assert_eq!(created.is_ok(), accepted, "reported={reported}");
        match created {
            Ok(mut instance) => instance.destroy().unwrap(),
            Err(_) => {
                assert_eq!(host.allocations.borrow().len(), 0);
                assert!(host.logs.borrow().iter().any(|line| {
                    line.contains("maximum_scored_rows is 513")
                        && line.contains("holds 512 scored rows")
                }));
            }
        }
    }
}

/// Final-token rows of ordinary prefill items count against the scored cap
/// of the phase they share with a scored prompt.
#[test]
fn mixed_scored_phase_counts_final_rows_against_the_scored_cap() {
    use runtime_bundle_sdk::{BatchAppendV1, BatchItemV1, BatchRequestV1, BatchWireVersionV1};
    const BEFORE: u32 = 3;
    const AFTER: u32 = 4;
    let api = BundleApiV1::new_with_prefix_cache(crate::BUNDLE_MIMO26_EXTENDED_API_V1).unwrap();
    for total in [
        crate::contract::SCORED_ROWS,
        crate::contract::SCORED_ROWS + 1,
    ] {
        let host = Rc::new(FakeHost::default());
        let mut instance = api
            .create(HostServicesTableV1::new(Rc::clone(&host)), &configuration())
            .unwrap();
        let scored = total - BEFORE - AFTER;
        let scored_pages = u64::from(scored).div_ceil(PAGE_TOKENS);
        let mut items = Vec::new();
        let mut appends = Vec::new();
        let mut page = scored_pages;
        for index in 0..BEFORE + 1 + AFTER {
            if index == BEFORE {
                items.push(
                    BatchItemV1::new(7, false, scored, true, vec![42; scored as usize])
                        .with_prompt_rows(true),
                );
                appends.push(BatchAppendV1::new(
                    0,
                    u64::from(scored),
                    u64::from(scored),
                    (0..scored_pages).collect(),
                    None,
                ));
            } else {
                items.push(BatchItemV1::new(
                    100 + u64::from(index),
                    false,
                    3,
                    true,
                    vec![5, 6, 7],
                ));
                appends.push(BatchAppendV1::new(0, 3, 3, vec![page], None));
                page += 1;
            }
        }
        let request = BatchRequestV1::new(BatchWireVersionV1::V2, items, appends);
        let before = host.launch_symbols.borrow().len();
        let mut sink = ViewSink::default();
        let result = instance.execute(&request.encode().unwrap(), &mut sink);
        if total > crate::contract::SCORED_ROWS {
            assert!(result.is_err());
            assert_eq!(host.launch_symbols.borrow().len(), before);
            assert!(
                host.logs
                    .borrow()
                    .iter()
                    .any(|line| line.contains("exceed the scored row cap"))
            );
        } else {
            result.unwrap_or_else(|error| panic!("{error:?} {:?}", host.logs.borrow()));
            let views = sink.0.iter().map(|view| view.raw()).collect::<Vec<_>>();
            assert_eq!(views.iter().map(|view| view.row_count).sum::<u32>(), total);
            // Rows arrive in item order, all in F32: the ordinary items'
            // final rows around every row of the scored prompt.
            assert_eq!(views[0].first_item_index, 0);
            assert!(
                views
                    .windows(2)
                    .all(|pair| pair[0].first_item_index < pair[1].first_item_index)
            );
            assert!(views.iter().all(|view| {
                view.element_type == runtime_bundle_abi::BUNDLE_LOGITS_ELEMENT_V1_F32
            }));
            instance.commit().unwrap();
        }
        instance.destroy().unwrap();
    }
}

/// With F32 logits off, a 256-row verify step fills the BF16 local and
/// gathered buffers, now sized to exactly the enabled verify rows.
#[test]
fn a_full_verify_step_fits_the_bf16_logits_buffers_without_f32_logits() {
    let (backend, mut instance) = wide_verify_instance((7, 256), 32);
    crate::instance::force_f32_logits(instance.handle(), false);
    let before = backend.launch_symbols.borrow().len();
    let mut sink = ViewSink::default();
    instance
        .execute(&wide_verify_step(32, 8), &mut sink)
        .unwrap_or_else(|error| panic!("{error:?} {:?}", backend.logs.borrow()));
    assert_eq!(
        sink.0.iter().map(|view| view.raw().row_count).sum::<u32>(),
        256
    );
    assert!(sink.0.iter().all(|view| {
        view.raw().element_type == runtime_bundle_abi::BUNDLE_LOGITS_ELEMENT_V1_BF16
    }));
    let launched = backend.launch_symbols.borrow()[before..].to_vec();
    assert_eq!(
        launched
            .iter()
            .filter(|symbol| symbol.as_str() == "chop_gemm_mimo_head_prefill_v1")
            .count(),
        RANKS as usize
    );
    assert!(!launched.iter().any(|symbol| symbol.contains("head_f32")));
    let keep = (0..32)
        .map(|item| kept(100 + item, (item % 8 + 1) as u32))
        .collect::<Vec<_>>();
    instance.commit_accepted(&keep).expect("keep 1 to 8 rows");
}
#[path = "offload_tests.rs"]
mod offload;
/// TP8 is unsupported before create acquires any host resource.
#[test]
fn create_refuses_tp8_before_acquiring_resources() {
    use bundle_runtime::config::{KvLayout, StateLayout, test_support};
    let api = BundleApiV1::new_with_prefix_cache(crate::BUNDLE_MIMO26_EXTENDED_API_V1).unwrap();
    let host = Rc::new(FakeHost::default());
    let mut roles = vec![b"mimo.rank-images.v1".to_vec()];
    roles.extend(
        (0..8).map(|rank| format!("chop.bundle-metadata.mimo.rank-index.r{rank}.v1").into_bytes()),
    );
    let record = test_support::encode(
        b"mimo-v26-flash-tp8",
        &[0, 1, 2, 3, 4, 5, 6, 7],
        &KvLayout {
            layout_id: crate::contract::KV_FORMATS[0].layout_id.to_vec(),
            page_bytes: 368_640,
            alignment: 256,
            page_tokens: PAGE_TOKENS,
            ranks: 8,
            lifetime: 0,
        },
        &StateLayout {
            state_id: b"mimo.swa-ring.v1".to_vec(),
            bytes: RING_STATE_BYTES_PER_RANK / 2,
            alignment: 256,
        },
        &roles,
        Some(32),
    );
    let error = api
        .create(HostServicesTableV1::new(Rc::clone(&host)), &record)
        .err()
        .expect("TP8 create must fail");
    assert_eq!(error, BundleAbiError::Status(BundleStatusV1::Unsupported));
    assert!(host.journal.borrow().is_empty());
    assert!(host.allocations.borrow().is_empty());
    assert_eq!(host.released.get(), 0);
    assert_eq!(host.peer_grants.get(), 0);
    assert_eq!(host.kv_acquires.get(), 0);
    assert_eq!(host.kv_resolves.get(), 0);
    assert_eq!(host.symbols.get(), 0);
}

/// A create record is refused unless its topology, rank count, page bytes
/// and ring state all name one supported topology.
#[test]
fn create_accepts_each_topology_only_with_its_own_layouts() {
    let api = BundleApiV1::new_with_prefix_cache(crate::BUNDLE_MIMO26_EXTENDED_API_V1).unwrap();
    {
        let tp = Tp::TP2;
        let host = Rc::new(FakeHost::at(tp));
        let mut instance = api
            .create(
                HostServicesTableV1::new(Rc::clone(&host)),
                &configuration_at(tp, false),
            )
            .expect("create at its own topology");
        assert_eq!(host.kv_resolves.get(), tp.ranks());
        assert_eq!(
            host.peer_grants.get(),
            tp.ranks() * (tp.ranks() - 1),
            "every ordered rank pair"
        );
        instance.destroy().expect("destroy");
        // TP4's layouts under this topology's ID and rank count.
        let mut record = configuration_at(tp, false);
        let tp4_page = GA_PAGE_BYTES_BF16.to_le_bytes();
        let page = crate::topology::ga_page_bytes(tp, false).to_le_bytes();
        if page != tp4_page {
            let at = record
                .windows(8)
                .position(|window| window == page)
                .expect("page bytes in the record");
            record[at..at + 8].copy_from_slice(&tp4_page);
            let host = Rc::new(FakeHost::at(tp));
            assert!(
                api.create(HostServicesTableV1::new(Rc::clone(&host)), &record)
                    .is_err()
            );
        }
        let mut record = configuration_at(tp, false);
        let ring = crate::topology::ring_state_bytes_per_rank(tp).to_le_bytes();
        let at = record
            .windows(8)
            .position(|window| window == ring)
            .expect("ring bytes in the record");
        record[at..at + 8].copy_from_slice(&RING_STATE_BYTES_PER_RANK.to_le_bytes());
        let host = Rc::new(FakeHost::at(tp));
        assert!(
            api.create(HostServicesTableV1::new(Rc::clone(&host)), &record)
                .is_err()
        );
    }
}

/// Media records on the shared encoder runtime: grids beyond the recipes'
/// limits are refused before any device work, and the largest video
/// (16 grids of 1,024 patches, two runs) and audio (60,001 frames, eleven
/// segments and two group runs) stage their payload and every run's tables
/// within the 64 MiB staging. Every record writes its rows to its pages,
/// page by page, in one program.
#[test]
fn media_records_fit_staging_and_write_their_pages() {
    use runtime_bundle_sdk::{
        BatchMediaEncodeV3, BatchMediaSectionV3, BatchRequestV1, BatchWireVersionV1,
    };
    let backend = Rc::new(FakeHost::default());
    let table = HostServicesTableV1::new(Rc::clone(&backend));
    let services = table.caller();
    let handle = BundleDeviceBufferHandleV1::from_parts(1, 1);
    let (encoder, symbols) = shared_encoder(services, &[handle]);
    let arena = encoder.ranks[0].region("arena").unwrap();
    let section = |encoder, geometry, rows: u32, bytes: u64| BatchMediaSectionV3 {
        encodes: vec![BatchMediaEncodeV3 {
            encoder,
            geometry,
            output_rows: rows,
            pages: (0..u64::from(rows).div_ceil(64)).rev().collect(),
            payload_offset: 0,
            payload_bytes: bytes,
        }],
        payload: vec![0; bytes as usize],
        ..BatchMediaSectionV3::default()
    };
    let vision = |t: u32, h: u32, w: u32| {
        let bytes = u64::from(t * h * w) * 1536 * 4;
        section(0, [t, h, w, 0], t * h * w / 4, bytes)
    };
    let audio = |frames: u32| {
        let declaration = &model_mimo26_contract::encoders::spec::declarations_flash().unwrap()[1];
        let item = declaration.admit_record([frames, 128, 0, 0]).unwrap();
        let rows = declaration.trace(item).unwrap().last().unwrap().rows as u32;
        section(1, [frames, 128, 0, 0], rows, u64::from(frames) * 128 * 4)
    };
    let run = |media: BatchMediaSectionV3| {
        let request = BatchRequestV1::new(BatchWireVersionV1::V3, Vec::new(), Vec::new())
            .with_media(media)
            .encode()
            .unwrap();
        encoder.run(
            services,
            &[BundleStreamHandleV1::from_parts(1, 1)],
            &symbols,
            &request,
            &Default::default(),
            &Default::default(),
        )
    };
    for (t, h, w) in [(2, 64, 128), (1, 64, 128), (3, 2, 2050), (4096, 2, 2)] {
        let media = vision(t, h, w);
        let refused =
            crate::encoders::injection::plan(&media, &crate::inputs::PhaseInputs::default(), 0);
        assert!(refused.is_err(), "{t}x{h}x{w}");
        let before = backend.journal.borrow().len();
        assert_eq!(run(media), Err(BundleStatusV1::InvalidArgument));
        assert_eq!(backend.journal.borrow().len(), before, "{t}x{h}x{w}");
    }
    // The record, and the runs that write output rows.
    let cases = [
        ("video 16x32x32", vision(16, 32, 32), 2),
        ("video 10x32x32", vision(10, 32, 32), 2),
        ("video 11x32x32", vision(11, 32, 32), 2),
        ("image 1x64x64", vision(1, 64, 64), 1),
        ("image 1x2x2048", vision(1, 2, 2048), 1),
        ("video 16x2x512", vision(16, 2, 512), 1),
        ("tiny 16x2x2", vision(16, 2, 2), 1),
        ("small image 1x16x32", vision(1, 16, 32), 1),
        ("audio 60001", audio(60_001), 2),
        ("audio 32772", audio(32_772), 2),
        ("audio 6001", audio(6_001), 1),
        ("small audio 1600", audio(1_600), 1),
    ];
    for (name, media, runs) in cases {
        crate::encoders::injection::plan(&media, &crate::inputs::PhaseInputs::default(), 0)
            .expect("declared geometry passes the payload contract");
        let pages = media.encodes[0].pages.len();
        let rows = u64::from(media.encodes[0].output_rows);
        let before = backend.copies.borrow().len();
        let synchronized = backend
            .journal
            .borrow()
            .iter()
            .filter(|line| line.starts_with("synchronize "))
            .count();
        assert_eq!(run(media), Ok(false), "{name}");
        let copies = backend.copies.borrow();
        let arena_copies = copies[before..]
            .iter()
            .filter(|&&(_, dst, _)| (arena.offset..arena.offset + arena.length).contains(&dst))
            .collect::<Vec<_>>();
        // A run whose rows end inside a page splits that page's copy.
        assert!(
            arena_copies.len() >= pages && arena_copies.len() < pages + runs,
            "{name}"
        );
        assert_eq!(
            arena_copies
                .iter()
                .map(|&&(_, _, bytes)| bytes)
                .sum::<u64>(),
            rows * 4096 * 2,
            "{name}"
        );
        // One program per record, however many runs: one drain.
        let drained = backend
            .journal
            .borrow()
            .iter()
            .filter(|line| line.starts_with("synchronize "))
            .count();
        assert_eq!(drained - synchronized, 1, "{name}");
    }
}

/// Audio of 6,001 frames on the shared runtime: two tokenizer segments
/// (6,000 and one frame), each through its 24 layers (12 of them in
/// windows) and its residual quantizer, then one group run of the 1,501
/// codes padded with the last to 1,504, gathered in one copy and three,
/// through the six local layers; its 376 rows land on noncontiguous pages.
/// A capture writes its rows and its 1,501 joined codes.
#[test]
fn audio_records_restart_segments_and_write_their_pages() {
    use runtime_bundle_sdk::{
        BatchMediaEncodeV3, BatchMediaSectionV3, BatchRequestV1, BatchWireVersionV1,
    };
    let backend = Rc::new(FakeHost::default());
    let table = HostServicesTableV1::new(Rc::clone(&backend));
    let services = table.caller();
    // Readbacks of handles from 60,000 return zeros.
    backend.encoder_readbacks.set(true);
    let (mut encoder, symbols) = shared_encoder(
        services,
        &[BundleDeviceBufferHandleV1::from_parts(1, 60_001)],
    );
    let root = std::env::temp_dir().join(format!("encoder-joined-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    encoder.capture = Some(root.clone());
    let region = |name| encoder.ranks[0].region(name).unwrap();
    let frames = 6001;
    let request = BatchRequestV1::new(BatchWireVersionV1::V3, Vec::new(), Vec::new())
        .with_media(BatchMediaSectionV3 {
            encodes: vec![BatchMediaEncodeV3 {
                encoder: 1,
                geometry: [frames, 128, 0, 0],
                output_rows: 376,
                pages: vec![9, 3, 8, 2, 7, 1],
                payload_offset: 0,
                payload_bytes: u64::from(frames) * 128 * 4,
            }],
            payload: vec![0; frames as usize * 128 * 4],
            ..BatchMediaSectionV3::default()
        })
        .encode()
        .unwrap();
    let before = backend.launch_symbols.borrow().len();
    let copies = backend.copies.borrow().len();
    encoder
        .run(
            services,
            &[BundleStreamHandleV1::from_parts(1, 1)],
            &symbols,
            &request,
            &Default::default(),
            &Default::default(),
        )
        .unwrap();
    let launches = backend.launch_symbols.borrow()[before..].to_vec();
    let count = |symbol: &str| {
        launches
            .iter()
            .filter(|launched| *launched == symbol)
            .count()
    };
    let shared = |stem: &str| {
        model_mimo26_contract::sm120::encoders::kernel_flash(stem)
            .unwrap()
            .symbol
            .clone()
    };
    // MiMo's own residual quantizer and embedding sum (`rows`).
    assert_eq!(count("chop_mimo_enc_rvq20_f32_d1024_codes1024_v1"), 2);
    assert_eq!(
        count(&shared(
            "chop_enc_attention_varlen_pairs_q16_kv16_d64_causal_left128_v1"
        )),
        24
    );
    assert_eq!(
        count(&shared(
            "chop_enc_attention_varlen_pairs_q16_kv16_d64_causal_v1"
        )),
        24
    );
    assert_eq!(
        count(&shared(
            "chop_enc_attention_varlen_pairs_q16_kv16_d64_full_v1"
        )),
        6
    );
    assert_eq!(count("chop_mimo_enc_sum20_f32_vocab1280_w1024_v1"), 1);
    let copies = backend.copies.borrow()[copies..].to_vec();
    let (joined, gathered, arena) = (region("joined"), region("gathered"), region("arena"));
    let within = |span: DeviceBufferSpanV1, offset: u64| {
        (span.offset..span.offset + span.length).contains(&offset)
    };
    // Codes 0..1,501 in one copy, then code 1,500 three times.
    let gathers: Vec<(u64, u64, u64)> = copies
        .iter()
        .filter(|&&(src, dst, _)| within(joined, src) && within(gathered, dst))
        .map(|&(src, dst, bytes)| (src - joined.offset, dst - gathered.offset, bytes))
        .collect();
    assert_eq!(
        gathers,
        [
            (0, 0, 1501 * 80),
            (1500 * 80, 1501 * 80, 80),
            (1500 * 80, 1502 * 80, 80),
            (1500 * 80, 1503 * 80, 80)
        ]
    );
    // Rows 64..128 fill page 3; the last 56 rows land on page 1.
    assert!(
        copies
            .iter()
            .any(|&(_, dst, bytes)| dst == arena.offset + 3 * 64 * 4096 * 2
                && bytes == 64 * 4096 * 2)
    );
    assert!(
        copies
            .iter()
            .any(|&(_, dst, bytes)| dst == arena.offset + 64 * 4096 * 2 && bytes == 56 * 4096 * 2)
    );
    let step = std::fs::read_dir(&root)
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .path();
    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(step.join("manifest.json")).unwrap()).unwrap();
    let encode = &manifest["encodes"][0];
    assert_eq!(encode["rows"], 376);
    assert_eq!(
        encode["joined"],
        serde_json::json!({"file": "joined-0.bin", "rows": 1501, "row_bytes": 80})
    );
    assert_eq!(
        std::fs::read(step.join("joined-0.bin")).unwrap().len(),
        1501 * 80
    );
    assert_eq!(
        std::fs::read(step.join("encode-0.bin")).unwrap().len(),
        376 * 4096 * 2
    );
    std::fs::remove_dir_all(root).unwrap();
}

/// With block taps on, rank 0 reads back every transformer block's output
/// rows (and each transformer's input) of each record, every run's in run
/// order: vision's 28 blocks over
/// a two-run video's 16,384 patch rows, audio's 24 tokenizer blocks over
/// its two segments' 3,001 rows and 6 local blocks over its 1,504 grouped
/// rows. Off, a record reads nothing back.
#[test]
fn block_taps_read_every_blocks_rows_of_each_record() {
    use runtime_bundle_sdk::{
        BatchMediaEncodeV3, BatchMediaSectionV3, BatchRequestV1, BatchWireVersionV1,
    };
    let backend = Rc::new(FakeHost::default());
    // Readbacks of handles from 60,000 return zeros.
    backend.encoder_readbacks.set(true);
    let table = HostServicesTableV1::new(Rc::clone(&backend));
    let services = table.caller();
    let handle = BundleDeviceBufferHandleV1::from_parts(1, 60_001);
    let (mut encoder, symbols) = shared_encoder(services, &[handle]);
    let section = |encoder, geometry, rows: u32, bytes: u64| BatchMediaSectionV3 {
        encodes: vec![BatchMediaEncodeV3 {
            encoder,
            geometry,
            output_rows: rows,
            pages: (0..u64::from(rows).div_ceil(64)).collect(),
            payload_offset: 0,
            payload_bytes: bytes,
        }],
        payload: vec![0; bytes as usize],
        ..BatchMediaSectionV3::default()
    };
    let video = section(0, [16, 32, 32, 0], 4096, 16 * 1024 * 1536 * 4);
    let audio = section(1, [6001, 128, 0, 0], 376, 6001 * 128 * 4);
    let run = |encoder: &bundle_runtime::encoder::Encoder, media| {
        let request = BatchRequestV1::new(BatchWireVersionV1::V3, Vec::new(), Vec::new())
            .with_media(media)
            .encode()
            .unwrap();
        encoder
            .run(
                services,
                &[BundleStreamHandleV1::from_parts(1, 1)],
                &symbols,
                &request,
                &Default::default(),
                &Default::default(),
            )
            .unwrap();
        encoder.take_block_rows()
    };
    let readbacks = || {
        backend
            .journal
            .borrow()
            .iter()
            .filter(|line| line.starts_with("readback "))
            .count()
    };
    let before = readbacks();
    assert!(run(&encoder, video.clone()).is_empty());
    assert_eq!(readbacks(), before);
    encoder.block_taps = true;
    let rows = run(&encoder, video);
    let names: Vec<String> = std::iter::once("enc.vision.B00.in".to_owned())
        .chain((0..28).map(|block| format!("enc.vision.B{block:02}.out")))
        .collect();
    assert_eq!(
        rows.iter().map(|tap| tap.name.clone()).collect::<Vec<_>>(),
        names
    );
    for tap in &rows {
        assert_eq!((tap.record, tap.encoder), (0, 0));
        // The residual streams are F32 under f32-pairs.v1.
        assert_eq!(tap.bytes.len(), 16 * 1024 * 1280 * 4, "{}", tap.name);
    }
    let rows = run(&encoder, audio);
    let names: Vec<String> = std::iter::once("enc.audio.B00.in".to_owned())
        .chain((0..24).map(|block| format!("enc.audio.B{block:02}.out")))
        .chain(std::iter::once("enc.audio.B24.in".to_owned()))
        .chain((24..30).map(|block| format!("enc.audio.B{block:02}.out")))
        .collect();
    assert_eq!(
        rows.iter().map(|tap| tap.name.clone()).collect::<Vec<_>>(),
        names
    );
    for tap in &rows {
        let local = tap.name.as_str() >= "enc.audio.B24";
        let rows = if local { 1504 } else { 3001 };
        assert_eq!(tap.bytes.len(), rows * 1024 * 4, "{}", tap.name);
    }
}

#[path = "served_tests.rs"]
mod served;

#[path = "rank_memory_tests.rs"]
mod rank_memory;

#[path = "dflash_snapshot_tests.rs"]
mod dflash_snapshot;

#[path = "snapshot_dump_tests.rs"]
mod snapshot_dump;

#[test]
fn created_resources_match_contract_and_engine_planners() {
    use bundle_runtime::geometry::Items;
    use model_mimo26_contract::resources::{HIDDEN_LAYOUT_FLASH, LogitsLayout};
    let f32_decode = std::env::var_os("CHOPINFER_MIMO_F32_LOGITS").is_none_or(|value| value != "0");
    for tp in Tp::SUPPORTED {
        for (served, verify_rows, rows) in [
            (1, 0, 32),
            (17, 0, 32),
            (32, 0, 32),
            (33, 0, 64),
            (64, 0, 64),
            (32, 40, 64),
            (32, 256, 256),
            (100, 0, 128),
            (256, 0, 256),
        ] {
            let host = Rc::new(FakeHost::at(tp));
            host.served.set(served);
            host.speculation
                .set((verify_rows != 0).then_some((7, verify_rows)));
            let mut instance =
                BundleApiV1::new_with_prefix_cache(crate::BUNDLE_MIMO26_EXTENDED_API_V1)
                    .unwrap()
                    .create(
                        HostServicesTableV1::new(Rc::clone(&host)),
                        &configuration_serving(tp, &KV_FORMATS[0], false, served),
                    )
                    .unwrap();
            let expected = LogitsLayout::new(tp, rows, f32_decode);
            let workspace = crate::resource_tests::planner_maximum(tp, Items::new(served).unwrap());
            let spans = crate::instance::resource_spans(instance.handle());
            assert_eq!(spans.len(), tp.count());
            for rank in spans {
                let expected_bytes = [
                    HIDDEN_LAYOUT_FLASH.allocation_bytes(),
                    HIDDEN_LAYOUT_FLASH.allocation_bytes(),
                    expected.local,
                    expected.gathered_prefill,
                    expected.gathered_decode,
                    workspace,
                    expected.local_scored,
                    expected.gathered_scored,
                ];
                for (index, (span, bytes)) in rank.into_iter().zip(expected_bytes).enumerate() {
                    assert_eq!(
                        host.allocation_bytes.borrow()[&span.buffer.slot()],
                        bytes,
                        "TP{} served={served} verify={verify_rows} buffer={index}",
                        tp.ranks()
                    );
                    assert_eq!(
                        span.length,
                        if index < 2 {
                            HIDDEN_LAYOUT_FLASH.span_bytes()
                        } else {
                            bytes
                        }
                    );
                }
            }
            instance.destroy().unwrap();
        }
    }
}

#[test]
fn hybrid_tables_are_context_sized_before_kv_and_bound_every_phase() {
    use runtime_bundle_sdk::{BatchAppendV1, BatchItemV1, BatchRequestV1, BatchWireVersionV1};
    let api = BundleApiV1::new_with_prefix_cache(crate::BUNDLE_MIMO26_EXTENDED_API_V1).unwrap();
    let host = Rc::new(FakeHost::default());
    let mut config = configuration();
    config.extend_from_slice(&128_u64.to_le_bytes());
    let mut instance = api
        .create(HostServicesTableV1::new(Rc::clone(&host)), &config)
        .unwrap();
    assert_eq!(host.table_uploads_before_kv.get(), 2 * RANKS as usize);
    let uploads = host.host_copies.borrow();
    let tables: Vec<_> = uploads
        .iter()
        .filter(|(_, data)| {
            data.len() >= 256
                && data[..256]
                    .chunks_exact(8)
                    .all(|pair| pair == [0, 0, 128, 63, 0, 0, 0, 0])
        })
        .collect();
    assert_eq!(tables.len(), 2 * RANKS as usize);
    assert!(tables.iter().all(|(bytes, _)| *bytes == 128 * 256));
    drop(uploads);
    for rows in [128_u32, 129] {
        let request = BatchRequestV1::new(
            BatchWireVersionV1::V2,
            vec![BatchItemV1::new(
                7,
                false,
                rows,
                true,
                vec![42; rows as usize],
            )],
            vec![BatchAppendV1::new(
                0,
                rows as u64,
                rows as u64,
                (0..rows.div_ceil(64)).map(u64::from).collect(),
                None,
            )],
        );
        let before = host.launch_symbols.borrow().len();
        assert_eq!(
            crate::instance::preflight_request(instance.handle(), &request.encode().unwrap()),
            if rows == 128 {
                BundleStatusV1::Success
            } else {
                BundleStatusV1::InvalidArgument
            }
        );
        assert_eq!(host.launch_symbols.borrow().len(), before);
    }
    instance.destroy().unwrap();
}

#[test]
fn host_rope_tables_are_released_by_instance_teardown() {
    let api = BundleApiV1::new_with_prefix_cache(crate::BUNDLE_MIMO26_EXTENDED_API_V1).unwrap();
    let host = Rc::new(FakeHost::default());
    let owner = crate::rope_tables::tables(197);
    let lifetime = std::sync::Arc::downgrade(&owner);
    let mut config = configuration();
    config.extend_from_slice(&197_u64.to_le_bytes());
    let mut instance = api.create(HostServicesTableV1::new(host), &config).unwrap();
    drop(owner);
    // Synchronous rank uploads have finished, so host bytes can already go.
    assert!(lifetime.upgrade().is_none());
    instance.destroy().unwrap();
    assert!(lifetime.upgrade().is_none());
}

#[test]
fn create_refuses_unknown_geometry_before_acquiring_resources() {
    let host = Rc::new(FakeHost::default());
    let id = b"mimo-absent-tp4";
    let mut bytes = bundle_runtime::config::MAGIC.to_vec();
    bytes.extend_from_slice(&(id.len() as u32).to_le_bytes());
    bytes.extend_from_slice(id);
    assert!(matches!(
        BundleApiV1::new_with_prefix_cache(crate::BUNDLE_MIMO26_EXTENDED_API_V1)
            .unwrap()
            .create(HostServicesTableV1::new(Rc::clone(&host)), &bytes),
        Err(BundleAbiError::Status(BundleStatusV1::Unsupported))
    ));
    assert!(host.journal.borrow().is_empty());
}

/// Both encoder towers execute before target rows at every listed geometry and rank count.
#[test]
fn geometry_encoders_run_tp2_tp4_media_lifecycle() {
    use runtime_bundle_sdk::{
        BatchAppendV1, BatchItemV1, BatchMediaEncodeV3, BatchMediaInjectV3, BatchMediaSectionV3,
        BatchRequestV1, BatchWireVersionV1,
    };
    let mut exercised = 0;
    for tp in crate::topology::all().filter(|tp| tp.geometry.encoders) {
        assert!(crate::topology::encoders_available(tp));
        let host = Rc::new(FakeHost::at(tp));
        let api = BundleApiV1::new_with_prefix_cache(crate::BUNDLE_MIMO26_EXTENDED_API_V1).unwrap();
        let mut instance = api
            .create(
                HostServicesTableV1::new(Rc::clone(&host)),
                &configuration_on_with_encoders(
                    tp,
                    &crate::contract::KV_FORMATS[0],
                    false,
                    &(0..tp.ranks() as i32).collect::<Vec<_>>(),
                    32,
                    true,
                ),
            )
            .unwrap();
        let v = tp.geometry.vision();
        let pixels = 4
            * v.in_chans as u64
            * v.temporal_patch_size as u64
            * v.spatial_patch_size as u64
            * v.spatial_patch_size as u64
            * 4;
        for (request_id, encoder, geometry, rows, payload_bytes, token) in [
            (70, 0, [1, 2, 2, 0], 1, pixels, tp.geometry.tokens.image),
            (
                71,
                1,
                [80, 128, 0, 0],
                5,
                80 * 128 * 4,
                tp.geometry.tokens.audio,
            ),
        ] {
            let request = BatchRequestV1::new(
                BatchWireVersionV1::V3,
                vec![BatchItemV1::new(
                    request_id,
                    false,
                    rows,
                    true,
                    vec![token; rows as usize],
                )],
                vec![BatchAppendV1::new(
                    0,
                    rows as u64,
                    rows as u64,
                    vec![3],
                    None,
                )],
            )
            .with_media(BatchMediaSectionV3 {
                encodes: vec![BatchMediaEncodeV3 {
                    encoder,
                    geometry,
                    output_rows: rows,
                    pages: vec![0],
                    payload_offset: 0,
                    payload_bytes,
                }],
                injects: vec![BatchMediaInjectV3 {
                    item_index: 0,
                    item_row: 0,
                    rows,
                    arena_row: 0,
                }],
                payload: vec![0; payload_bytes as usize],
            })
            .encode()
            .unwrap();
            instance
                .execute(&request, &mut DraftSink::default())
                .unwrap();
            instance.cancel().unwrap();
            instance
                .execute(&request, &mut DraftSink::default())
                .unwrap();
            instance.commit().unwrap();
            instance.retire_request(request_id).unwrap();
        }
        instance.destroy().unwrap();
        exercised += 1;
    }
    assert!(exercised >= 2);
}

/// The span of `length` bytes at `offset` in buffer `slot`.
fn unknown_span(slot: u32, offset: u64, length: u64) -> DeviceBufferSpanV1 {
    DeviceBufferSpanV1::new(
        BundleDeviceBufferHandleV1::from_parts(1, slot),
        offset,
        length,
    )
}

/// An injected copy failure comes before the copy's checks: a failing copy
/// of mismatched lengths, or across buffers, fails with the injected status
/// and reaches no journal.
#[test]
fn a_copy_fault_precedes_the_copy_checks() {
    let host = FakeHost::default();
    let stream = BundleStreamHandleV1::from_parts(1, 1);
    host.fail_copy.set(Some(0));
    assert_eq!(
        host.copy_device_to_device(
            stream,
            unknown_span(900_000, 0, 8),
            unknown_span(900_000, 8, 4)
        ),
        Err(BundleStatusV1::HostError)
    );
    host.fail_copy.set(Some(0));
    assert_eq!(
        host.copy_device_to_device(
            stream,
            unknown_span(900_001, 0, 8),
            unknown_span(900_000, 0, 8)
        ),
        Err(BundleStatusV1::HostError)
    );
    assert_eq!(host.fail_copy.get(), None);
    assert!(host.journal.borrow().is_empty());
}

/// A device copy across buffers (without drafter or encoder images) panics
/// before it reaches the journal or the copy records.
#[test]
fn a_rejected_cross_buffer_copy_reaches_no_journal() {
    let host = FakeHost::default();
    let copy = std::panic::AssertUnwindSafe(|| {
        host.copy_device_to_device(
            BundleStreamHandleV1::from_parts(1, 1),
            unknown_span(900_001, 0, 8),
            unknown_span(900_000, 0, 8),
        )
    });
    assert!(std::panic::catch_unwind(copy).is_err());
    assert!(host.journal.borrow().is_empty());
    assert!(host.device_copies.borrow().is_empty());
}

/// An injected readback failure comes before the readback's length check.
#[test]
fn a_readback_fault_precedes_the_length_check() {
    let host = FakeHost::default();
    host.prefix_fault.set(Some((PrefixFault::Readback, 0)));
    assert_eq!(
        host.copy_device_to_host(
            BundleStreamHandleV1::from_parts(1, 1),
            &mut [0; 4],
            unknown_span(900_000, 0, 8)
        ),
        Err(BundleStatusV1::HostError)
    );
    assert_eq!(host.prefix_fault.get(), None);
    assert!(host.journal.borrow().is_empty());
}
/// Components, identities and peer grants count as acquired handles, and
/// their releases (journaled or not) balance them.
#[test]
fn components_and_peer_grants_balance_their_releases() {
    let host = FakeHost::default();
    let precision = host
        .open_authenticated_component(model_mimo26_contract::prepared::COMPONENT_ROLE.as_bytes())
        .unwrap();
    let identity = host
        .open_authenticated_component(b"chop.compiled-image-identity.v1/mimo.rank-images.v1/0")
        .unwrap();
    let context = host.select_context(0, 0).unwrap();
    let peer = host.select_context(1, 1).unwrap();
    let pool = host.select_memory_pool(context).unwrap();
    host.set_memory_pool_peer_access(pool, peer, true).unwrap();
    assert_eq!(host.acquired.get(), 6);
    host.set_memory_pool_peer_access(pool, peer, false).unwrap();
    host.release_memory_pool(pool).unwrap();
    host.release_context(peer).unwrap();
    host.release_context(context).unwrap();
    host.release_authenticated_component(identity).unwrap();
    host.release_authenticated_component(precision).unwrap();
    assert_eq!(host.released.get(), 6);
    assert_eq!(host.peer_grants.get(), 1);
}

/// A created instance with its drafter, run through a draft step and a
/// verify step and destroyed, released every handle the host counted as
/// acquired: contexts, pools, buffers, images, arenas, components, state,
/// streams, events, modules, symbols, peer grants and graphs.
#[test]
fn a_destroyed_instance_released_every_handle_it_acquired() {
    use runtime_bundle_sdk::{BatchAppendV1, BatchItemV1, BatchRequestV1, BatchWireVersionV1};
    let api = BundleApiV1::new_with_prefix_cache(crate::BUNDLE_MIMO26_EXTENDED_API_V1).unwrap();
    let host = Rc::new(FakeHost::default());
    host.speculation.set(Some((7, 32)));
    let mut instance = api
        .create(
            HostServicesTableV1::new(Rc::clone(&host)),
            &configuration_with_mtp(true),
        )
        .unwrap();
    let request = |start, tokens: Vec<u32>, verify| {
        let rows = if verify { 4 } else { tokens.len() as u32 };
        BatchRequestV1::new(
            BatchWireVersionV1::V3,
            vec![if verify {
                BatchItemV1::verify(7, rows, true, tokens)
            } else {
                BatchItemV1::new(7, false, rows, true, tokens)
            }],
            vec![BatchAppendV1::new(
                start,
                u64::from(rows),
                start + u64::from(rows),
                vec![1],
                None,
            )],
        )
        .encode()
        .unwrap()
    };
    instance
        .execute(
            &request(0, vec![40, 41, 42], false),
            &mut DraftSink::default(),
        )
        .unwrap();
    instance.commit().unwrap();
    *host.readback_tokens.borrow_mut() = vec![43, 44, 45, 46];
    instance
        .execute(&request(3, vec![43], true), &mut DraftSink::default())
        .unwrap();
    instance.commit_accepted(&[kept(7, 1)]).unwrap();
    instance.retire_request(7).unwrap();
    instance.destroy().unwrap();
    assert!(host.graph_releases.get() > 0);
    assert!(!host.mtp_images.borrow().is_empty());
    assert!(host.peer_grants.get() > 0);
    assert_eq!(host.acquired.get(), host.released.get());
}
/// Behavior change of the shared fake host: a symbol configures at most the
/// target device's 101,376 bytes of dynamic shared memory.
#[test]
fn symbol_configuration_is_limited_to_the_device_shared_memory() {
    let host = FakeHost::default();
    let symbol = BundleCudaSymbolHandleV1::from_parts(1, 1);
    host.configure_cuda_symbol(symbol, 101_376).unwrap();
    let oversized = std::panic::AssertUnwindSafe(|| host.configure_cuda_symbol(symbol, 101_377));
    assert!(std::panic::catch_unwind(oversized).is_err());
}

/// Behavior change of the shared fake host: a readback's destination holds
/// exactly its source span.
#[test]
fn a_readback_fills_exactly_its_destination() {
    let host = FakeHost::default();
    host.tap_readbacks.set(true);
    let stream = BundleStreamHandleV1::from_parts(1, 1);
    let span = unknown_span(900_000, 0, 8);
    host.copy_device_to_host(stream, &mut [0; 8], span).unwrap();
    let short =
        std::panic::AssertUnwindSafe(|| host.copy_device_to_host(stream, &mut [0; 4], span));
    assert!(std::panic::catch_unwind(short).is_err());
}

/// A prompt of `request` in page 3 whose `rows` are the output of one
/// media record of `encoder` at `geometry`, with a zero payload of `payload`
/// bytes: each row the medium's `token`.
fn media_step(
    request: u64,
    encoder: u32,
    geometry: [u32; 4],
    rows: u32,
    payload: u64,
    token: u32,
) -> Vec<u8> {
    use runtime_bundle_sdk::{
        BatchAppendV1, BatchItemV1, BatchMediaEncodeV3, BatchMediaInjectV3, BatchMediaSectionV3,
        BatchRequestV1, BatchWireVersionV1,
    };
    BatchRequestV1::new(
        BatchWireVersionV1::V3,
        vec![BatchItemV1::new(
            request,
            false,
            rows,
            true,
            vec![token; rows as usize],
        )],
        vec![BatchAppendV1::new(
            0,
            u64::from(rows),
            u64::from(rows),
            vec![3],
            None,
        )],
    )
    .with_media(BatchMediaSectionV3 {
        encodes: vec![BatchMediaEncodeV3 {
            encoder,
            geometry,
            output_rows: rows,
            pages: vec![0],
            payload_offset: 0,
            payload_bytes: payload,
        }],
        injects: vec![BatchMediaInjectV3 {
            item_index: 0,
            item_row: 0,
            rows,
            arena_row: 0,
        }],
        payload: vec![0; payload as usize],
    })
    .encode()
    .unwrap()
}

/// The host's default modalities (text and image) on a package whose
/// encoders serve images only: it names its encoder set, its encoder rank
/// images hold only the vision tower, and an instance creates, describes
/// its prefix state, warms up on text and serves an image at each
/// topology. An audio record finds no encoder.
#[test]
fn a_vision_only_package_creates_warms_up_and_serves_images() {
    use model_mimo26_contract::encoders::Encoders;
    use runtime_bundle_sdk::{BatchAppendV1, BatchItemV1, BatchRequestV1, BatchWireVersionV1};
    let images = Encoders::of(true, false, false);
    let api = BundleApiV1::new_with_prefix_cache(crate::BUNDLE_MIMO26_EXTENDED_API_V1).unwrap();
    let mut exercised = 0;
    for tp in crate::topology::all().filter(|tp| tp.geometry.encoders) {
        assert!(images.image_bytes_for(tp.geometry) < Encoders::ALL.image_bytes_for(tp.geometry));
        let host = Rc::new(FakeHost::at(tp));
        host.served_encoders.set(Some(images));
        let ordinals = (0..tp.ranks() as i32).collect::<Vec<_>>();
        let kv = &crate::contract::KV_FORMATS[0];
        let config = configuration_on_serving_encoders(tp, kv, false, &ordinals, 32, images);
        let mut instance = api
            .create(HostServicesTableV1::new(Rc::clone(&host)), &config)
            .unwrap();
        assert_eq!(host.encoder_images.borrow().len(), 1);
        instance.describe_prefix_state().unwrap();
        // Warmup: a prompt, then one decode row.
        let text = |start: u64, tokens: Vec<u32>| {
            let rows = tokens.len() as u64;
            BatchRequestV1::new(
                BatchWireVersionV1::V3,
                vec![BatchItemV1::new(7, start > 0, rows as u32, true, tokens)],
                vec![BatchAppendV1::new(start, rows, start + rows, vec![1], None)],
            )
            .encode()
            .unwrap()
        };
        for (start, tokens) in [(0, vec![40, 41, 42]), (3, vec![43])] {
            instance
                .execute(&text(start, tokens), &mut DraftSink::default())
                .unwrap();
            instance.commit().unwrap();
        }
        instance.retire_request(7).unwrap();
        let v = tp.geometry.vision();
        let pixels = 4
            * v.in_chans as u64
            * v.temporal_patch_size as u64
            * v.spatial_patch_size as u64
            * v.spatial_patch_size as u64
            * 4;
        let image = media_step(70, 0, [1, 2, 2, 0], 1, pixels, tp.geometry.tokens.image);
        instance.execute(&image, &mut DraftSink::default()).unwrap();
        instance.commit().unwrap();
        instance.retire_request(70).unwrap();
        // The audio record `geometry_encoders_run_tp2_tp4_media_lifecycle`
        // serves with every encoder.
        let audio = media_step(
            71,
            1,
            [80, 128, 0, 0],
            5,
            80 * 128 * 4,
            tp.geometry.tokens.audio,
        );
        assert!(
            instance.execute(&audio, &mut DraftSink::default()).is_err(),
            "tp{}: an audio record",
            tp.ranks()
        );
        assert!(
            host.logs
                .borrow()
                .iter()
                .any(|line| line == "MiMo encoder record: no encoder 1"),
            "tp{}: {:?}",
            tp.ranks(),
            host.logs.borrow()
        );
        instance.destroy().unwrap();
        assert_eq!(host.acquired.get(), host.released.get());
        exercised += 1;
    }
    assert!(exercised >= 2);
}

/// A package that names its encoder set refuses, before the encoder image
/// opens, encoder rank indexes holding other encoders than the set names;
/// so does one that names no set (every encoder) over vision-only indexes.
#[test]
fn an_encoder_set_refuses_indexes_of_other_encoders() {
    use model_mimo26_contract::encoders::Encoders;
    let api = BundleApiV1::new_with_prefix_cache(crate::BUNDLE_MIMO26_EXTENDED_API_V1).unwrap();
    let images = Encoders::of(true, false, false);
    let audio = Encoders::of(false, false, true);
    for tp in [Tp::TP2, Tp::TP4] {
        let ordinals = (0..tp.ranks() as i32).collect::<Vec<_>>();
        let kv = &crate::contract::KV_FORMATS[0];
        for (served, indexed) in [
            (images, Encoders::ALL),
            (images, audio),
            (Encoders::ALL, images),
        ] {
            let case = format!(
                "tp{} set {:?} indexes {:?}",
                tp.ranks(),
                served.names(),
                indexed.names()
            );
            let host = Rc::new(FakeHost::at(tp));
            host.served_encoders.set(Some(served));
            host.indexed_encoders.set(Some(indexed));
            let config = configuration_on_serving_encoders(tp, kv, false, &ordinals, 32, served);
            assert!(
                api.create(HostServicesTableV1::new(Rc::clone(&host)), &config)
                    .is_err(),
                "{case}"
            );
            // The text image opened and uploaded before the encoders.
            assert_eq!(host.text_images.get(), 1, "{case}");
            assert!(host.encoder_images.borrow().is_empty(), "{case}");
            let line = "MiMo create failed: MiMo encoder rank index range is invalid";
            assert!(
                host.logs.borrow().iter().any(|logged| logged == line),
                "{case}: {:?}",
                host.logs.borrow()
            );
            assert_eq!(host.acquired.get(), host.released.get(), "{case}");
        }
    }
}

/// The fake host holds the encoder-set component exactly when the compiler
/// packages it: for a partial set of media, not for every medium, none, or a
/// package that serves the default (every one).
#[test]
fn the_encoder_set_component_exists_only_for_a_partial_set() {
    use model_mimo26_contract::encoders::Encoders;
    let role = model_mimo26_contract::weight_plan::ENCODER_SET_ROLE.as_bytes();
    for served in std::iter::once(None).chain(Encoders::SETS.map(Some)) {
        let host = FakeHost::at(Tp::TP4);
        host.served_encoders.set(served);
        let encoders = served.unwrap_or(Encoders::ALL);
        let partial = encoders.any() && encoders != Encoders::ALL;
        assert_eq!(names_encoder_set(encoders), partial, "{served:?}");
        match host.open_authenticated_component(role) {
            Ok(handle) => {
                assert!(partial, "{served:?}");
                let (named, bytes) = host.component(handle.slot());
                assert_eq!(named.as_bytes(), role);
                assert_eq!(
                    model_mimo26_contract::weight_plan::parse_encoder_set(&bytes),
                    Ok(encoders)
                );
            }
            Err(status) => {
                assert!(!partial, "{served:?}");
                assert_eq!(status, BundleStatusV1::NotFound);
            }
        }
    }
}
