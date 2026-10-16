//! CPU fake host for the Qwen3.8 bundle: acquisition, launch plans, state
//! bookkeeping, graph replay and the launch journal.
//!
//! The host runs no kernel. It checks that every launch names a resolved
//! symbol, that every buffer argument lies inside a live allocation, rank
//! image or pool, and it records every device-facing call: the shared fake
//! host (`bundle_runtime::fake_host::Backend`) over this package.

#[path = "conformance_tests.rs"]
mod conformance_tests;

use std::cell::{Cell, RefCell};
use std::collections::{BTreeMap, BTreeSet};
use std::rc::Rc;

use bundle_runtime::fake_host::{
    Backend, FakeDevice, FakeIndex, FakePackage, Handle, RecordingSink, SinkLimits, SnapshotOp,
    check_journal, dump::paged_copies, fnv1a, journal_digest,
};
use runtime_bundle_abi::*;
use runtime_bundle_sdk::{BatchAppendV1, BatchItemV1, BatchRequestV1, BatchWireVersionV1};

use crate::contract::*;
use model_qwen38_contract::package::ngram_in_host_memory;

mod context_parallel;
mod diagnostics;
mod hc;
mod invariance;
mod lifecycle;
mod logits;
mod memory;
mod mixed_mtp;
mod norms;
mod paired;
pub(crate) mod precision;
mod served;
mod sm120;
mod snapshot_dump;
mod speculation;
mod taps;

const KV_PAGES: u64 = 1024;
/// The rank count of the TP4 tests.
const RANKS: u32 = Shape::TP4.ranks();
/// Handle slots of the fixed spans the fake host hands out.
const IMAGE_SLOT: u32 = 1_000_000;
const KV_SLOT: u32 = 2_000_000;
const STATE_SLOT: u32 = 3_000_000;
const ENCODER_IMAGE_SLOT: u32 = 4_000_000;
const HOST_TABLE_SLOT: u32 = 5_000_000;
/// Authenticated component handles of the encoder indexes start here.
const ENCODER_COMPONENT: u32 = 100;
/// Authenticated component handles of the served encoder images'
/// identities (`chop.compiled-image-identity.v1/...`) start here.
const ENCODER_IDENTITY_COMPONENT: u32 = 200;
/// Authenticated component handles of the served text images' identities
/// start here.
const IDENTITY_COMPONENT: u32 = 300;

/// The fake host of a Qwen3.8 package.
type FakeHost = Backend<Package>;

/// A host for a package of a topology.
trait At {
    fn at(shape: Shape) -> Self;
}

impl At for FakeHost {
    fn at(shape: Shape) -> Self {
        Backend::new(Package::at(shape))
    }
}

/// The package the fake host serves, and what tests set and read.
#[derive(Default)]
struct Package {
    /// Handles, the journal (seen by the lifecycle recorder while set),
    /// streams, events, graphs, collectives, residency, span extents and
    /// the snapshot record.
    device: FakeDevice<lifecycle::Recorder>,
    precision: RefCell<Option<kernel_core::precision::execution::PrecisionDocument>>,
    capture_value: Cell<u16>,
    capture_rows: RefCell<BTreeMap<u64, u16>>,
    fail_restore_rank: Cell<Option<u32>>,
    fail_allocation: Cell<Option<usize>>,
    allocation_attempts: Cell<usize>,
    allocated_buffers: RefCell<Vec<u32>>,
    allocation_bytes: RefCell<Vec<u64>>,
    /// The memory pool slot of each allocation, in allocation order.
    allocation_pools: RefCell<Vec<u32>>,
    released_buffers: RefCell<Vec<u32>>,
    /// (stream, symbol, grid) of every launch in issue order.
    launches: RefCell<Vec<(u32, String, [u32; 3])>>,
    /// The arguments of every launch, beside `launches`.
    launch_arguments: RefCell<Vec<Vec<KernelArgumentV1>>>,
    /// The handle slot of the opened encoder image.
    encoder_image: Cell<Option<u32>>,
    /// The identity each encoder index names in place of its image digest
    /// (`sha256` or `direct_identity`, and the value); unset names the
    /// served file's digest.
    encoder_index_identity: RefCell<Option<(&'static str, String)>>,
    /// The identity the host serves for each encoder rank; unset serves a
    /// file of the image's length with the indexes' digest.
    encoder_served_identity: RefCell<Option<serde_json::Value>>,
    /// The identity each text rank index names in place of its image digest
    /// (`sha256` or `direct_identity`, and the value); unset names the
    /// served file's digest.
    index_identity: RefCell<Option<(&'static str, String)>>,
    /// The identity the host serves for each text rank, with the rank
    /// image's length unless it names one; unset serves a file with the
    /// index's digest.
    served_identity: RefCell<Option<serde_json::Value>>,
    captures: Cell<u32>,
    replays: Cell<u32>,
    /// (source, destination) of every device copy.
    copies: RefCell<Vec<(DeviceBufferSpanV1, DeviceBufferSpanV1)>>,
    /// Destination and bytes of every small upload.
    uploads: RefCell<Vec<(DeviceBufferSpanV1, Vec<u8>)>>,
    /// Fault injection: launches of this symbol fail when the countdown
    /// reaches zero.
    fail_symbol_launch: RefCell<Option<(String, u32)>>,
    /// Fault injection: uploads of this many bytes fail when the countdown
    /// reaches zero, after the upload is journaled.
    fail_upload: Cell<Option<(u64, u32)>>,
    /// Fault injection: device-to-host copies fail, after they are journaled.
    fail_download: Cell<bool>,
    /// The package has the MTP drafter: its views, pages and ring.
    drafter: Cell<bool>,
    /// The package topology; TP4 unless a test sets another.
    topology: Cell<Option<Shape>>,
    /// The package declares FP8 main-layer K and V.
    fp8_kv: Cell<bool>,
    /// The sequences the host serves (`--max-sequences`); `SERVED` unless a
    /// test sets another count.
    served: Cell<Option<u32>>,
    /// (drafts, verify tokens) the host enables.
    speculation: Cell<(u32, u32)>,
    /// Launches of each capturing stream, for the model to replay.
    captured: RefCell<BTreeMap<u32, Vec<speculation::Captured>>>,
    /// Indexes into `launches` of each capturing stream's launches, then of
    /// each graph's, which run only when the graph is launched.
    captured_launches: RefCell<BTreeMap<u32, Vec<usize>>>,
    graph_launches: RefCell<BTreeMap<u32, Vec<usize>>>,
    /// (launches recorded before it, graph slot) of every graph launch.
    graph_replays: RefCell<Vec<(usize, u32)>>,
    sim: RefCell<speculation::Sim>,
    /// Sparse bytes of uploads and prefix snapshot copies once `backing` is
    /// set, by (buffer slot, chunk). No other kernel runs, so a test writes
    /// what a step would have left in a slot's state.
    backing: Cell<bool>,
    memory: RefCell<BTreeMap<(u32, u64), Vec<u8>>>,
    /// Failures a test injects: stream synchronization, and launches of
    /// one symbol from the given launch count on.
    refuse_synchronize: Cell<bool>,
    refuse_launch: RefCell<Option<(String, usize)>>,
    /// The package places the n-gram table in host memory: the host cut it
    /// out of the arena and answers `NGRAM_HOST_VIEW`.
    ngram_host: Cell<bool>,
    /// (slot, offset, length) of every buffer argument of every launch, in
    /// issue order, pointer arrays flattened.
    launch_spans: RefCell<Vec<Vec<(u32, u64, u64)>>>,
    /// Norm weights the prepared rank views store as F32 rows, as an export
    /// with F32 norms places them.
    f32_norms: RefCell<BTreeSet<String>>,
    /// The only rank that stores `f32_norms` as F32, when set.
    f32_norm_rank: Cell<Option<u32>>,
}

impl std::ops::Deref for Package {
    type Target = FakeDevice<lifecycle::Recorder>;

    fn deref(&self) -> &Self::Target {
        &self.device
    }
}

const CHUNK: u64 = 1 << 16;
const ZERO_CHUNK: &[u8; CHUNK as usize] = &[0; CHUNK as usize];

#[test]
fn zero_snapshot_chunks_stay_sparse() {
    let host = FakeHost::default();
    let buffer = BundleDeviceBufferHandleV1::from_parts(1, 1);
    let span = |offset, length| DeviceBufferSpanV1::new(buffer, offset, length);
    let crossing = span(CHUNK - 2, 4);
    host.write(crossing, &[0; 4]);
    assert!(host.memory.borrow().is_empty());
    host.write(crossing, &[1, 2, 3, 4]);
    assert_eq!(host.read(span(CHUNK - 3, 6)), [0, 1, 2, 3, 4, 0]);
    host.write(span(CHUNK - 2, 2), &[0; 2]);
    assert_eq!(host.memory.borrow().len(), 1);
    assert_eq!(host.read(crossing), [0, 0, 3, 4]);
    host.write(span(CHUNK, 1), &[0]);
    assert_eq!(host.read(crossing), [0, 0, 0, 4]);
    host.write(span(CHUNK + 1, 1), &[0]);
    assert!(host.memory.borrow().is_empty());
    // Clearing a neighboring range must preserve a byte outside that range.
    host.write(span(CHUNK - 3, 1), &[9]);
    host.write(crossing, &[0; 4]);
    assert_eq!(host.read(span(CHUNK - 3, 6)), [9, 0, 0, 0, 0, 0]);
    assert_eq!(host.memory.borrow().len(), 1);
}

impl Package {
    /// A package of `shape`'s topology.
    fn at(shape: Shape) -> Self {
        let host = Self::default();
        host.topology.set(Some(shape));
        host.sim.borrow_mut().topology = Some(shape);
        host
    }

    fn shape(&self) -> Shape {
        self.topology.get().unwrap_or(Shape::TP4)
    }

    /// The layout the package declared, with the ring state the host
    /// allocates for the served sequences.
    fn layout(&self) -> (&'static [u8], u64, u64) {
        let shape = self.shape();
        let kv = shape.kv_formats()
            [usize::from(self.drafter.get()) + 2 * usize::from(self.fp8_kv.get())];
        (
            kv.layout_id,
            kv.page_bytes,
            shape.ring_state_slot_bytes(self.drafter.get())
                * u64::from(self.served().next_power_of_two()),
        )
    }

    fn served(&self) -> u32 {
        self.served.get().unwrap_or(SERVED)
    }

    /// Points the device model at an instance's buffers.
    fn model(&self, handle: BundleHandleV1) {
        let mut sim = self.sim.borrow_mut();
        let parts = crate::instance::model_overlap_arenas(handle);
        for (rank, (stream, [x_start, x_final], ple_input)) in
            crate::instance::model_spans(handle).into_iter().enumerate()
        {
            sim.spans.insert(
                stream,
                speculation::ModelSpans {
                    x_start,
                    x_final,
                    ple_input,
                    parts: parts.get(rank).copied(),
                },
            );
        }
    }

    fn read(&self, span: DeviceBufferSpanV1) -> Vec<u8> {
        let memory = self.memory.borrow();
        let mut bytes = Vec::with_capacity(span.length as usize);
        let mut at = span.offset;
        while at < span.offset + span.length {
            let (chunk, within) = (at / CHUNK, at % CHUNK);
            let take = (CHUNK - within).min(span.offset + span.length - at);
            match memory.get(&(span.buffer.slot(), chunk)) {
                Some(data) => {
                    bytes.extend_from_slice(&data[within as usize..(within + take) as usize]);
                }
                None => bytes.extend_from_slice(&ZERO_CHUNK[..take as usize]),
            }
            at += take;
        }
        bytes
    }

    fn write(&self, span: DeviceBufferSpanV1, bytes: &[u8]) {
        assert_eq!(span.length, bytes.len() as u64);
        let mut memory = self.memory.borrow_mut();
        let mut at = span.offset;
        let mut from = 0;
        while at < span.offset + span.length {
            let (chunk, within) = (at / CHUNK, at % CHUNK);
            let take = (CHUNK - within).min(span.offset + span.length - at);
            let key = (span.buffer.slot(), chunk);
            let source = &bytes[from..from + take as usize];
            let zero = source == &ZERO_CHUNK[..source.len()];
            // Missing chunks already read as zero. A zero overwrite must
            // still clear existing bytes and preserve its neighbors.
            if !zero || memory.contains_key(&key) {
                let data = memory.entry(key).or_insert_with(|| vec![0; CHUNK as usize]);
                data[within as usize..(within + take) as usize].copy_from_slice(source);
                if zero && data.as_slice() == ZERO_CHUNK.as_slice() {
                    memory.remove(&key);
                }
            }
            at += take;
            from += take as usize;
        }
    }

    /// Bytes of `span`: the device model's for the buffers it tracks, else
    /// the sparse backing when a test set it.
    fn device_bytes(&self, span: DeviceBufferSpanV1) -> Vec<u8> {
        let tracked = self
            .extents
            .borrow()
            .get(&span.buffer.slot())
            .is_some_and(|&extent| extent <= speculation::TRACKED);
        if self.backing.get() && !tracked {
            self.read(span)
        } else {
            self.sim.borrow_mut().read(span, &self.extents.borrow())
        }
    }

    /// Runs `chop_qwen38_prefix_copy_v1` as `prefix_copy_spans` addresses
    /// it: on the backing bytes when a test set them, and on the device
    /// model's token histories and X-row tags.
    fn prefix_copy(&self, arguments: &[KernelArgumentV1]) {
        let KernelArgumentV1::DeviceBuffer(table) = arguments[6] else {
            panic!("the page table is {:?}", arguments[6]);
        };
        let pages = self
            .device_bytes(table)
            .chunks_exact(8)
            .map(|lane| u64::from_le_bytes(lane.try_into().unwrap()))
            .collect::<Vec<_>>();
        for (from, to) in prefix_copy_spans(self.shape(), arguments, &pages) {
            if self.backing.get() {
                self.write(to, &self.read(from));
            }
            self.sim.borrow_mut().copy(to, from, &self.extents.borrow());
        }
    }
}

/// What `prefix_copy` of `cuda/qwen38_bundle.cuh` moves for its launch
/// `arguments`, as (from, to) spans in snapshot order with the snapshot's
/// `pages`: pool entry `index` of every linear layer's convolution plane,
/// then of every delta-rule plane, PLE entry `slot`, each ring layer's keys
/// and positions of slot `slot`, then the slot's retained rows, end to end
/// across the pages. Mirrors the kernel's addressing, not the bundle's.
fn prefix_copy_spans(
    shape: Shape,
    arguments: &[KernelArgumentV1],
    pages: &[u64],
) -> Vec<(DeviceBufferSpanV1, DeviceBufferSpanV1)> {
    assert!(matches!(arguments.len(), 14 | 16));
    let span = |index: usize| match arguments[index] {
        KernelArgumentV1::DeviceBuffer(span) => Some(span),
        KernelArgumentV1::NullDevicePointer => None,
        ref other => panic!("argument {index} is {other:?}"),
    };
    let scalar = |index: usize| match arguments[index] {
        KernelArgumentV1::ScalarU32(value) => u64::from(value),
        KernelArgumentV1::ScalarU64(value) => value,
        ref other => panic!("argument {index} is {other:?}"),
    };
    let (conv, delta, ple, ring, retained) = (
        span(0).unwrap(),
        span(1).unwrap(),
        span(2).unwrap(),
        span(3).unwrap(),
        span(4),
    );
    let pool = span(5).unwrap();
    let (ring_layers, page_bytes, slot, index) = (scalar(7), scalar(8), scalar(9), scalar(10));
    let (to_pages, pool_entries, ring_slots) = (scalar(11) != 0, scalar(12), scalar(13));
    if slot >= ring_slots || index >= pool_entries || page_bytes == 0 || page_bytes % 16 != 0 {
        return Vec::new();
    }
    let (conv_entry, delta_entry) = (shape.conv_entry_bytes, shape.delta_entry_bytes);
    let (ple_entry, key_slot, position_slot) = (
        shape.ple_entry_bytes(),
        u64::from(QSA_RING) * u64::from(shape.index_dim()) * BF16_BYTES,
        u64::from(QSA_RING) * 3 * 4,
    );
    let retained_slot =
        u64::from(crate::scratch::RETAINED_ROWS) * crate::scratch::wide_row_bytes(shape);
    let ring_layer = ring_slots * (key_slot + position_slot);
    let at = |base: DeviceBufferSpanV1, offset: u64, length: u64| {
        DeviceBufferSpanV1::new(base.buffer, base.offset + offset, length)
    };
    let mut held = Vec::new();
    for layer in 0..shape.linear_layers() {
        held.push(at(
            conv,
            (layer * pool_entries + index) * conv_entry,
            conv_entry,
        ));
    }
    for layer in 0..shape.linear_layers() {
        held.push(at(
            delta,
            (layer * pool_entries + index) * delta_entry,
            delta_entry,
        ));
    }
    held.push(at(
        ple,
        (if arguments.len() == 16 {
            scalar(14)
        } else {
            slot
        }) * ple_entry,
        ple_entry,
    ));
    for layer in 0..ring_layers {
        held.push(at(ring, layer * ring_layer + slot * key_slot, key_slot));
        held.push(at(
            ring,
            layer * ring_layer + ring_slots * key_slot + slot * position_slot,
            position_slot,
        ));
    }
    if let Some(retained) = retained {
        held.push(at(retained, slot * retained_slot, retained_slot));
    }
    paged_copies(held, pool, pages, page_bytes, to_pages)
}

/// A synthetic rank index holding every required view, and the drafter's.
fn fake_index(rank: u32, drafter: bool, shape: Shape) -> (Vec<u8>, u64) {
    let mut required = crate::views::required(rank, shape);
    if drafter {
        required.extend(crate::views::drafter_required(shape));
    }
    fake_index_views(rank, required)
}

fn fake_index_views(rank: u32, required: Vec<crate::views::Required>) -> (Vec<u8>, u64) {
    let mut index = FakeIndex::new(crate::views::SCHEMA, rank);
    for view in required {
        index.push(
            view.name,
            view.bytes,
            view.dtype,
            vec![view.bytes],
            view.layout_id,
            4096,
        );
    }
    let image_bytes = index.end(4096);
    (index.bytes(image_bytes), image_bytes)
}

impl Package {
    /// A precision package's index is built from the prepared view contract
    /// the bundle validates, so it agrees with that contract by construction
    /// and cannot show a placement that names a view differently. The
    /// compiler's `planned_mixed_rank_indexes_pass_the_bundles_view_validation`
    /// runs the same validation on the placement's own rank indexes.
    fn rank_index(&self, rank: u32) -> (Vec<u8>, u64) {
        let f32_norms = self.f32_norms.borrow();
        let precision = self.precision.borrow();
        if precision.is_none() && f32_norms.is_empty() {
            return fake_index(rank, self.drafter.get(), self.shape());
        }
        let shape = self.shape();
        // Legacy attention reads Q, K and V as contiguous weights.
        let contiguous_qkv = |required: &mut Vec<crate::views::Required>| {
            required.sort_by_cached_key(|v| {
                v.name
                    .replace(".self_attn.q_proj", ".self_attn.0_q_proj")
                    .replace(".self_attn.k_proj", ".self_attn.1_k_proj")
                    .replace(".self_attn.v_proj", ".self_attn.2_v_proj")
            });
        };
        let mut required = if let Some(precision) = precision.as_ref() {
            let mut required =
                model_qwen38_contract::prepared::required(precision, rank, shape).unwrap();
            contiguous_qkv(&mut required);
            // A drafter package places the MTP views: in their recorded
            // encodings when the component records the MTP layer.
            if self.drafter.get() {
                if model_qwen38_contract::prepared::records_drafter(precision, shape).unwrap() {
                    let mut drafter =
                        model_qwen38_contract::prepared::drafter_required(precision, rank, shape)
                            .unwrap();
                    contiguous_qkv(&mut drafter);
                    required.extend(drafter);
                } else {
                    required.extend(crate::views::drafter_required(shape));
                }
            }
            required
        } else {
            let mut required = crate::views::required(rank, shape);
            if self.drafter.get() {
                required.extend(crate::views::drafter_required(shape));
            }
            required
        };
        let stored = self.f32_norm_rank.get().is_none_or(|only| only == rank);
        for view in &mut required {
            if stored && f32_norms.contains(&view.name) {
                view.dtype = "F32";
                view.layout_id = crate::views::F32_ROWS;
                view.bytes *= 2;
            }
        }
        fake_index_views(rank, required)
    }
}

/// The role under which the host serves text rank `rank`'s identity.
fn identity_role(rank: u32) -> String {
    format!(
        "chop.compiled-image-identity.v1/{}/{rank}",
        std::str::from_utf8(RANK_IMAGE_ROLE).unwrap()
    )
}

/// The role under which the host serves encoder rank `rank`'s identity.
fn encoder_identity_role(rank: u32) -> String {
    format!(
        "chop.compiled-image-identity.v1/{}/{rank}",
        std::str::from_utf8(crate::encoder::views::IMAGE_ROLE).unwrap()
    )
}

/// The encoder index of `rank`, exactly the bundle's layout.
fn fake_encoder_index(rank: u32, shape: Shape) -> Vec<u8> {
    use crate::encoder::views::*;
    let mut index = FakeIndex::new(SCHEMA, rank);
    index.image = image_name(rank);
    index.sha256 = "0".repeat(64);
    index.views = layout_for(shape.geometry)
        .iter()
        .map(|view| {
            let placed = View {
                offset: view.offset,
                bytes: view.bytes(),
                dtype: "BF16".into(),
                shape: view.shape.to_vec(),
                layout_id: BF16_ROWS.into(),
            };
            (view.name.to_string(), placed)
        })
        .collect();
    index.bytes(image_bytes_for(shape.geometry))
}

/// The role and bytes of an authenticated component handle slot.
fn component_bytes(slot: u32, drafter: bool, shape: Shape) -> (String, Vec<u8>) {
    match slot.checked_sub(ENCODER_COMPONENT + 1) {
        Some(rank) => (
            crate::encoder::views::index_role(rank),
            fake_encoder_index(rank, shape),
        ),
        None => (index_role(slot - 1), fake_index(slot - 1, drafter, shape).0),
    }
}

impl FakePackage for Package {
    type Observer = lifecycle::Recorder;

    fn device(&self) -> &FakeDevice<lifecycle::Recorder> {
        &self.device
    }
    fn device_residency(&self) -> BundleRawClusterResidencyV1 {
        crate::sm120::device::comm_residency()
    }
    fn cubins() -> impl IntoIterator<Item = (&'static str, &'static [u8])> {
        crate::CUBINS.iter().map(|cubin| (cubin.name, cubin.bytes))
    }
    fn release(&self, handle: Handle) {
        match handle {
            Handle::Buffer(slot) => {
                // Callback tables retain the fake host for the process
                // lifetime. Release simulated storage with its buffer, not
                // with the host.
                self.released_buffers.borrow_mut().push(slot);
                self.memory
                    .borrow_mut()
                    .retain(|&(buffer, _), _| buffer != slot);
                self.sim.borrow_mut().release_buffer(slot);
            }
            Handle::Graph(slot) => self.sim.borrow_mut().release_graph(slot),
            _ => {}
        }
        self.note("release".into());
        if self.released.get() == self.acquired.get()
            && self.fail_allocation.get().is_none()
            && let Some(root) = std::env::var_os("CHOPINFER_QWEN38_M0_ALLOCATIONS_DIR")
        {
            thread_local! { static CAPTURE: Cell<u32> = const { Cell::new(0) }; }
            let index = CAPTURE.with(|count| {
                let index = count.get();
                count.set(index + 1);
                index
            });
            let thread = std::thread::current();
            let name = thread.name().unwrap_or("main");
            let root = std::path::PathBuf::from(root);
            std::fs::create_dir_all(&root).unwrap();
            let ranks = self.allocation_ranks.borrow();
            let allocation = self
                .allocated_buffers
                .borrow()
                .iter()
                .enumerate()
                .map(|(index, id)| {
                    let rank = ranks
                        .get(id)
                        .map(|&(rank, _)| rank.to_string())
                        .unwrap_or_else(|| "unbound".into());
                    let bytes = self.allocation_bytes.borrow()[index];
                    format!("{index} rank={rank} handle={id} bytes={bytes}\n")
                })
                .collect::<String>();
            std::fs::write(root.join(format!("{name}-{index}.txt")), allocation).unwrap();
            if let Some(root) = std::env::var_os("CHOPINFER_QWEN38_M0_HOSTS_DIR") {
                let root = std::path::PathBuf::from(root);
                std::fs::create_dir_all(&root).unwrap();
                std::fs::write(
                    root.join(format!("{name}-{index}.txt")),
                    self.journal.borrow().join("\n"),
                )
                .unwrap();
            }
        }
    }

    fn component(&self, slot: u32) -> (String, Vec<u8>) {
        if slot == 1000 {
            return (
                model_qwen38_contract::prepared::COMPONENT_ROLE.into(),
                kernel_core::precision::component::encode(
                    self.precision.borrow().as_ref().unwrap(),
                )
                .unwrap(),
            );
        }
        if let Some((key, value)) = self.index_identity.borrow().as_ref()
            && slot >= 1
            && slot <= self.shape().ranks()
        {
            let mut index: serde_json::Value =
                serde_json::from_slice(&self.rank_index(slot - 1).0).unwrap();
            let fields = index.as_object_mut().unwrap();
            fields.remove("sha256");
            fields.insert((*key).into(), value.clone().into());
            return (index_role(slot - 1), serde_json::to_vec(&index).unwrap());
        }
        if (self.precision.borrow().is_some() || !self.f32_norms.borrow().is_empty())
            && slot >= 1
            && slot <= self.shape().ranks()
        {
            return (index_role(slot - 1), self.rank_index(slot - 1).0);
        }
        if let Some(rank) = slot
            .checked_sub(IDENTITY_COMPONENT + 1)
            .filter(|&rank| rank < self.shape().ranks())
        {
            let mut identity = self
                .served_identity
                .borrow()
                .clone()
                .unwrap_or_else(|| serde_json::json!({"sha256": "a".repeat(64)}));
            let fields = identity.as_object_mut().unwrap();
            if !fields.contains_key("byte_length") {
                fields.insert("byte_length".into(), self.rank_index(rank).1.into());
            }
            return (identity_role(rank), serde_json::to_vec(&identity).unwrap());
        }
        if let Some(rank) = slot.checked_sub(ENCODER_IDENTITY_COMPONENT + 1) {
            let identity = self
                .encoder_served_identity
                .borrow()
                .clone()
                .unwrap_or_else(|| {
                    serde_json::json!({"byte_length": crate::encoder::views::image_bytes_for(self.shape().geometry),
                    "sha256": "0".repeat(64)})
                });
            return (
                encoder_identity_role(rank),
                serde_json::to_vec(&identity).unwrap(),
            );
        }
        if let (Some(rank), Some((key, value))) = (
            slot.checked_sub(ENCODER_COMPONENT + 1),
            self.encoder_index_identity.borrow().as_ref(),
        ) {
            let mut index: serde_json::Value =
                serde_json::from_slice(&fake_encoder_index(rank, self.shape())).unwrap();
            let fields = index.as_object_mut().unwrap();
            fields.remove("sha256");
            fields.insert((*key).into(), value.clone().into());
            return (
                crate::encoder::views::index_role(rank),
                serde_json::to_vec(&index).unwrap(),
            );
        }
        component_bytes(slot, self.drafter.get(), self.shape())
    }
    fn admit_allocation(
        &self,
        _pool: BundleMemoryPoolHandleV1,
        _byte_length: u64,
        _alignment: u64,
    ) -> HostServiceResultV1<()> {
        let attempt = self.allocation_attempts.get();
        self.allocation_attempts.set(attempt + 1);
        if self.fail_allocation.get() == Some(attempt) {
            return Err(BundleStatusV1::HostError);
        }
        Ok(())
    }
    fn before_allocation(
        &self,
        pool: BundleMemoryPoolHandleV1,
        buffer: BundleDeviceBufferHandleV1,
        byte_length: u64,
    ) {
        self.allocated_buffers.borrow_mut().push(buffer.slot());
        self.allocation_bytes.borrow_mut().push(byte_length);
        self.allocation_pools.borrow_mut().push(pool.slot());
    }
    fn open_compiled_image(&self, role: &[u8]) -> HostServiceResultV1<BundleCompiledImageHandleV1> {
        let slot = self.id();
        if role == crate::encoder::views::IMAGE_ROLE {
            self.encoder_image.set(Some(slot));
        } else {
            assert_eq!(role, RANK_IMAGE_ROLE);
        }
        Ok(BundleCompiledImageHandleV1::from_parts(1, slot))
    }
    fn upload_rank_arena(
        &self,
        _context: BundleContextHandleV1,
        _pool: BundleMemoryPoolHandleV1,
        _stream: BundleStreamHandleV1,
        _image: BundleCompiledImageHandleV1,
        _rank: u32,
    ) -> HostServiceResultV1<BundleRankArenaHandleV1> {
        Ok(BundleRankArenaHandleV1::from_parts(1, self.id()))
    }
    fn resolve_compiled_image_view(
        &self,
        image: BundleCompiledImageHandleV1,
        _arena: BundleRankArenaHandleV1,
        rank: u32,
        view_id: &[u8],
    ) -> HostServiceResultV1<CompiledImageSpanV1> {
        if self.encoder_image.get() == Some(image.slot()) {
            let bytes = crate::encoder::views::image_bytes_for(self.shape().geometry);
            let slot = ENCODER_IMAGE_SLOT + rank;
            self.extents.borrow_mut().insert(slot, bytes);
            return Ok(CompiledImageSpanV1::new(
                rank,
                DeviceBufferSpanV1::new(BundleDeviceBufferHandleV1::from_parts(1, slot), 0, bytes),
                4096,
            ));
        }
        let (index, mut bytes) = self.rank_index(rank);
        let table = crate::views::parse(&index, rank).unwrap().views
            [&crate::views::ngram_view_for(self.shape())]
            .bytes;
        let (slot, alignment) = match view_id {
            b"" => (IMAGE_SLOT + rank, 4096),
            NGRAM_HOST_VIEW if self.ngram_host.get() => {
                bytes = table;
                (HOST_TABLE_SLOT + rank, 1)
            }
            _ => return Err(BundleStatusV1::NotFound),
        };
        if view_id.is_empty() && self.ngram_host.get() {
            // The host's cut: the table rounded up to the image alignment.
            bytes -= table.next_multiple_of(4096);
        }
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
        if role == model_qwen38_contract::prepared::COMPONENT_ROLE.as_bytes()
            && self.precision.borrow().is_some()
        {
            return Ok(BundleAuthenticatedComponentHandleV1::from_parts(1, 1000));
        }
        let slot = (0..self.shape().ranks())
            .find_map(|rank| {
                if role == index_role(rank).as_bytes() {
                    Some(rank + 1)
                } else if role == crate::encoder::views::index_role(rank).as_bytes() {
                    Some(ENCODER_COMPONENT + rank + 1)
                } else if role == encoder_identity_role(rank).as_bytes() {
                    Some(ENCODER_IDENTITY_COMPONENT + rank + 1)
                } else if role == identity_role(rank).as_bytes() {
                    Some(IDENTITY_COMPONENT + rank + 1)
                } else {
                    None
                }
            })
            .ok_or(BundleStatusV1::InvalidArgument)?;
        Ok(BundleAuthenticatedComponentHandleV1::from_parts(1, slot))
    }
    fn acquire_serving_kv(&self, layout_id: &[u8]) -> HostServiceResultV1<u64> {
        assert_eq!(layout_id, self.layout().0);
        Ok(KV_PAGES)
    }
    fn resolve_serving_kv_pool(&self, rank: u32) -> HostServiceResultV1<DeviceBufferSpanV1> {
        let bytes = KV_PAGES * self.layout().1;
        self.extents.borrow_mut().insert(KV_SLOT + rank, bytes);
        Ok(DeviceBufferSpanV1::new(
            BundleDeviceBufferHandleV1::from_parts(1, KV_SLOT + rank),
            0,
            bytes,
        ))
    }
    fn query_speculation(&self) -> HostServiceResultV1<BundleRawSpeculationV1> {
        let (drafts, verify) = self.speculation.get();
        if drafts == 0 {
            return Ok(BundleRawSpeculationV1::disabled());
        }
        let mut enabled = BundleRawSpeculationV1::disabled();
        enabled.maximum_draft_tokens = drafts;
        enabled.maximum_verify_tokens = verify;
        Ok(enabled)
    }
    fn allocate_state(&self, request: &StateRequestV1) -> HostServiceResultV1<BundleStateHandleV1> {
        assert_eq!(request.byte_length, self.layout().2);
        assert_eq!(request.state_id, RING_STATE_ID);
        Ok(BundleStateHandleV1::from_parts(1, self.id()))
    }
    fn resolve_state_view(
        &self,
        _state: BundleStateHandleV1,
        rank: u32,
    ) -> HostServiceResultV1<DeviceBufferSpanV1> {
        let bytes = self.layout().2;
        self.extents.borrow_mut().insert(STATE_SLOT + rank, bytes);
        Ok(DeviceBufferSpanV1::new(
            BundleDeviceBufferHandleV1::from_parts(1, STATE_SLOT + rank),
            0,
            bytes,
        ))
    }
    fn read_back(
        &self,
        destination: &mut [u8],
        source: DeviceBufferSpanV1,
    ) -> HostServiceResultV1<()> {
        if self.fail_download.get() {
            return Err(BundleStatusV1::HostError);
        }
        if self.capture_value.get() == 0 {
            destination.copy_from_slice(&self.device_bytes(source));
            return Ok(());
        }
        // Encoder captures read synthetic merged rows.
        for (row, bytes) in destination.chunks_exact_mut(2560 * 2).enumerate() {
            let value = if source.offset == 0 {
                self.capture_value.get() + row as u16
            } else {
                self.capture_rows.borrow()[&(source.offset / (2560 * 2) + row as u64)]
            };
            for element in bytes.chunks_exact_mut(2) {
                element.copy_from_slice(&value.to_le_bytes());
            }
        }
        Ok(())
    }
    fn on_upload(
        &self,
        _stream: BundleStreamHandleV1,
        destination: DeviceBufferSpanV1,
        source: &[u8],
    ) -> HostServiceResultV1<()> {
        if let Some((length, countdown)) = self.fail_upload.get()
            && length == destination.length
        {
            if countdown == 0 {
                self.fail_upload.set(None);
                return Err(BundleStatusV1::HostError);
            }
            self.fail_upload.set(Some((length, countdown - 1)));
        }
        self.sim
            .borrow_mut()
            .write(destination, source, &self.extents.borrow());
        if source.len() <= 1 << 20 {
            self.uploads
                .borrow_mut()
                .push((destination, source.to_vec()));
        }
        if self.backing.get() {
            self.write(destination, source);
        }
        Ok(())
    }
    fn on_copy(
        &self,
        _stream: BundleStreamHandleV1,
        destination: DeviceBufferSpanV1,
        source: DeviceBufferSpanV1,
    ) -> HostServiceResultV1<()> {
        self.copies.borrow_mut().push((source, destination));
        if self.fail_restore_rank.get() == destination.buffer.slot().checked_sub(STATE_SLOT)
            && source.offset
                == destination.offset
                    + (Ring {
                        shape: self.shape(),
                        layers: if self.drafter.get() {
                            self.shape().full_layers_mtp()
                        } else {
                            self.shape().full_layers()
                        },
                        slots: u64::from(self.served().next_power_of_two()),
                    })
                    .bytes()
        {
            self.fail_restore_rank.set(None);
            lifecycle::copy_result(self, destination, source, "HostError");
            return Err(BundleStatusV1::HostError);
        }
        self.sim
            .borrow_mut()
            .copy(destination, source, &self.extents.borrow());
        lifecycle::copy_result(self, destination, source, "Success");
        Ok(())
    }
    fn admit_launch(
        &self,
        _stream: BundleStreamHandleV1,
        name: &str,
        _config: LaunchConfigV1,
        _arguments: &[KernelArgumentV1],
    ) -> HostServiceResultV1<()> {
        let mut fail = self.fail_symbol_launch.borrow_mut();
        if let Some((symbol, left)) = fail.as_mut()
            && *symbol == name
        {
            if *left == 0 {
                *fail = None;
                return Err(BundleStatusV1::HostError);
            }
            *left -= 1;
        }
        Ok(())
    }
    fn admit_spans(
        &self,
        _stream: BundleStreamHandleV1,
        name: &str,
        spans: &[DeviceBufferSpanV1],
    ) -> HostServiceResultV1<()> {
        self.launch_spans.borrow_mut().push(
            spans
                .iter()
                .map(|span| (span.buffer.slot(), span.offset, span.length))
                .collect(),
        );
        if let Some((refused, from)) = &*self.refuse_launch.borrow()
            && *refused == name
            && self.launches.borrow().len() >= *from
        {
            return Err(BundleStatusV1::HostError);
        }
        Ok(())
    }
    fn on_launch(
        &self,
        stream: BundleStreamHandleV1,
        name: &str,
        config: LaunchConfigV1,
        arguments: &[KernelArgumentV1],
    ) -> HostServiceResultV1<()> {
        let grid = [config.grid_x, config.grid_y, config.grid_z];
        // A capture's member streams record into its launches.
        let capture = self.capture_of(stream.slot());
        let mut captured = self.captured.borrow_mut();
        if let Some(capture) = capture
            && let Some(launches) = captured.get_mut(&capture)
        {
            launches.push((stream.slot(), name.into(), grid, arguments.to_vec()));
            self.captured_launches
                .borrow_mut()
                .entry(capture)
                .or_default()
                .push(self.launches.borrow().len());
        } else {
            self.sim.borrow_mut().launch(
                stream.slot(),
                name,
                grid,
                arguments,
                &self.extents.borrow(),
            );
        }
        drop(captured);
        if name
            == crate::launch::per_rank(
                model_qwen38_contract::sm120::owned::PREFIX_COPY_INDEXED,
                self.shape(),
            )
        {
            self.snapshot_op(|| SnapshotOp::Launch(stream.slot(), arguments.to_vec()));
            self.prefix_copy(arguments);
        }
        self.launches
            .borrow_mut()
            .push((stream.slot(), name.into(), grid));
        self.launch_arguments.borrow_mut().push(arguments.to_vec());
        Ok(())
    }
    fn on_synchronize(&self, _stream: BundleStreamHandleV1) -> HostServiceResultV1<()> {
        if self.refuse_synchronize.get() {
            return Err(BundleStatusV1::HostError);
        }
        Ok(())
    }
    fn on_capture(&self, stream: BundleStreamHandleV1) {
        self.captured.borrow_mut().insert(stream.slot(), Vec::new());
    }
    fn on_captured(&self, stream: BundleStreamHandleV1, graph: BundleGraphHandleV1) {
        let captured = self
            .captured
            .borrow_mut()
            .remove(&stream.slot())
            .unwrap_or_default();
        self.sim.borrow_mut().capture(graph.slot(), captured);
        let indexes = self
            .captured_launches
            .borrow_mut()
            .remove(&stream.slot())
            .unwrap_or_default();
        self.graph_launches
            .borrow_mut()
            .insert(graph.slot(), indexes);
        self.captures.set(self.captures.get() + 1);
    }
    fn on_replay(&self, _stream: BundleStreamHandleV1, graph: BundleGraphHandleV1, _nodes: usize) {
        self.graph_replays
            .borrow_mut()
            .push((self.launches.borrow().len(), graph.slot()));
        self.replays.set(self.replays.get() + 1);
        self.sim
            .borrow_mut()
            .replay(graph.slot(), &self.extents.borrow());
    }
}

/// `count` served items.
fn served(count: u32) -> bundle_runtime::geometry::Items {
    bundle_runtime::geometry::Items::new(count).unwrap()
}

/// The sequences the fake host serves unless a test names another count:
/// the count every scenario and journal ran at before it was a runtime
/// value.
const SERVED: u32 = 32;

/// The KV layout id, page bytes and ring state bytes of a package without
/// or with the MTP drafter, with BF16 or FP8 main-layer K and V, serving
/// `SERVED` sequences.
fn layout(drafter: bool, fp8_kv: bool) -> (&'static [u8], u64, u64) {
    layout_served(drafter, fp8_kv, SERVED)
}

/// [`layout`] serving `served` sequences: the host allocates the ring's
/// per-slot bytes for the served count rounded up to a power of two
/// (`chop.state-per-item.v1`), or the drafter package's fixed ring.
fn layout_served(drafter: bool, fp8_kv: bool, served: u32) -> (&'static [u8], u64, u64) {
    let kv = &KV_FORMATS[usize::from(drafter) + 2 * usize::from(fp8_kv)];
    let ring = if drafter {
        RING_STATE_SLOT_BYTES_MTP * u64::from(served.next_power_of_two())
    } else {
        RING_STATE_SLOT_BYTES * u64::from(served.next_power_of_two())
    };
    (kv.layout_id, kv.page_bytes, ring)
}

fn configuration() -> Vec<u8> {
    configuration_with(false, &[])
}

/// The configuration of a package without or with the MTP drafter, with
/// `extra` component roles after the text image and indexes.
fn configuration_with(drafter: bool, extra: &[Vec<u8>]) -> Vec<u8> {
    configuration_for(layout(drafter, false), extra)
}

fn configuration_for(layout: (&[u8], u64, u64), extra: &[Vec<u8>]) -> Vec<u8> {
    configuration_at(Shape::TP4, layout, extra)
}

/// The configuration of a package of `shape`'s topology, one device
/// ordinal per rank, serving `SERVED` sequences.
fn configuration_at(shape: Shape, layout: (&[u8], u64, u64), extra: &[Vec<u8>]) -> Vec<u8> {
    configuration_served(shape, layout, extra, SERVED)
}

/// [`configuration_at`] serving `served` sequences: the record's last field
/// (`chop.served-items.v1`).
fn configuration_served(
    shape: Shape,
    (layout, page_bytes, ring_bytes): (&[u8], u64, u64),
    extra: &[Vec<u8>],
    served: u32,
) -> Vec<u8> {
    let ranks = shape.ranks();
    let ordinals = (0..ranks as i32).collect::<Vec<_>>();
    let mut roles = vec![RANK_IMAGE_ROLE.to_vec()];
    roles.extend((0..ranks).map(|rank| index_role(rank).into_bytes()));
    roles.extend_from_slice(extra);
    bundle_runtime::config::test_support::encode(
        shape.topology().id.as_bytes(),
        &ordinals,
        &bundle_runtime::config::KvLayout {
            layout_id: layout.to_vec(),
            page_bytes,
            alignment: 256,
            page_tokens: PAGE_TOKENS,
            ranks: u64::from(ranks),
            lifetime: 0,
        },
        &bundle_runtime::config::StateLayout {
            state_id: RING_STATE_ID.to_vec(),
            bytes: ring_bytes,
            alignment: 256,
        },
        &roles,
        Some(served),
    )
}

#[derive(Default)]
struct ViewSink(Vec<OwnedLogitsViewV1>);

impl LogitsRowSinkV1 for ViewSink {
    fn limits(&self) -> ResultSinkLimitsV1 {
        ResultSinkLimitsV1::new(
            8192,
            u64::from(VOCAB) * 4,
            u64::from(SCORED_ROWS) * u64::from(VOCAB) * 4,
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

/// A view sink whose limits admit a full step of scored prompt rows.
#[derive(Default)]
struct ScoredSink(Vec<OwnedLogitsViewV1>);

impl LogitsRowSinkV1 for ScoredSink {
    fn limits(&self) -> ResultSinkLimitsV1 {
        let rows = u64::from(SCORED_ROWS) + u64::from(SERVED);
        ResultSinkLimitsV1::new(rows, u64::from(VOCAB) * 4, rows * u64::from(VOCAB) * 4)
    }

    fn emit_row(&mut self, _row: OwnedLogitsRowV1) -> HostServiceResultV1<()> {
        Err(BundleStatusV1::Unsupported)
    }

    fn emit_view(&mut self, view: OwnedLogitsViewV1) -> HostServiceResultV1<()> {
        self.0.push(view);
        Ok(())
    }
}

fn api() -> BundleApiV1 {
    BundleApiV1::new_with_prefix_state_slot(crate::BUNDLE_QWEN38_EXTENDED_API_V1)
        .expect("bundle API")
}

fn create(host: &Rc<FakeHost>) -> BundleInstanceV1 {
    let extra = if host.precision.borrow().is_some() {
        vec![
            model_qwen38_contract::prepared::COMPONENT_ROLE
                .as_bytes()
                .to_vec(),
        ]
    } else {
        vec![]
    };
    let instance = api()
        .create(
            HostServicesTableV1::new(Rc::clone(host)),
            &configuration_served(host.shape(), host.layout(), &extra, host.served()),
        )
        .unwrap_or_else(|error| panic!("fake-host create: {error:?} {:?}", host.logs.borrow()));
    if let Some(document) = host.precision.borrow().as_ref() {
        let copies: std::collections::BTreeSet<_> =
            model_qwen38_contract::sm120::generated::inventory(host.shape())
                .unwrap()
                .copies
                .into_iter()
                .map(|copy| copy.stage().entry().symbol.clone())
                .collect();
        // Authenticated dimensions give an independent BF16 output bound.
        let bytes = document
            .operations
            .iter()
            .flat_map(|op| {
                op.bindings
                    .iter()
                    .filter(|binding| {
                        binding
                            .implementation
                            .stages
                            .iter()
                            .any(|stage| copies.contains(stage))
                    })
                    .map(|binding| {
                        let capacity = if binding.usage.phase
                            != kernel_core::precision::ExecutionPhase::Prefill
                            && binding.usage.capacity <= DECODE_ENTRY_ROWS
                        {
                            binding.usage.capacity
                        } else {
                            PREFILL_TOKENS
                        };
                        u64::from(capacity)
                            * op.effective.matrix_geometry().unwrap().rows
                            * BF16_BYTES
                    })
            })
            .max()
            .unwrap_or(0);
        for (prefix, ticket, temporary, index) in
            crate::instance::test_scratch_suffix(instance.handle())
        {
            let order = host.buffer_order.borrow();
            let at = order
                .iter()
                .position(|&id| id == prefix.buffer.slot())
                .unwrap();
            assert_eq!(order[at + 1], ticket.buffer.slot());
            let mut next = at + 2;
            if bytes != 0 {
                let span = temporary.expect("precision output needs its later temporary");
                assert_eq!((span.offset, span.length), (0, bytes));
                assert_eq!(host.extents.borrow()[&span.buffer.slot()], bytes);
                assert_eq!(order[next], span.buffer.slot());
                let allocated = host.allocated_buffers.borrow();
                let cutoff = allocated
                    .iter()
                    .position(|&id| id == span.buffer.slot())
                    .unwrap();
                let failed = Rc::new(FakeHost::at(host.shape()));
                *failed.precision.borrow_mut() = Some(document.clone());
                failed.served.set(Some(host.served()));
                failed.ngram_host.set(host.ngram_host.get());
                *failed.f32_norms.borrow_mut() = host.f32_norms.borrow().clone();
                failed.drafter.set(host.drafter.get());
                failed.fp8_kv.set(host.fp8_kv.get());
                failed.speculation.set(host.speculation.get());
                failed.fail_allocation.set(Some(cutoff));
                assert!(
                    api()
                        .create(
                            HostServicesTableV1::new(Rc::clone(&failed)),
                            &configuration_served(
                                host.shape(),
                                failed.layout(),
                                &extra,
                                host.served()
                            )
                        )
                        .is_err()
                );
                assert_eq!(failed.allocation_attempts.get(), cutoff + 1);
                assert_eq!(*failed.allocated_buffers.borrow(), allocated[..cutoff]);
                assert_eq!(
                    *failed.released_buffers.borrow(),
                    allocated[..cutoff]
                        .iter()
                        .rev()
                        .copied()
                        .collect::<Vec<_>>()
                );
                assert_eq!(failed.released.get(), failed.acquired.get());
                next += 1;
            } else {
                assert!(temporary.is_none());
            }
            assert_eq!(order[next], index.buffer.slot());
        }
    }
    instance
}

/// One request's item in a step: (request, slot, previous tokens, inputs).
type Item<'a> = (u64, u64, u64, &'a [u32]);

/// A step of prefill items (several inputs) and decode items (one input),
/// with one KV page per 64 tokens numbered from the slot.
fn step(items: &[Item<'_>]) -> BatchRequestV1 {
    let mut batch = Vec::new();
    let mut appends = Vec::new();
    let mut slots = Vec::new();
    for &(request, slot, previous, inputs) in items {
        let count = inputs.len() as u64;
        let end = previous + count;
        let pages = (0..end.div_ceil(PAGE_TOKENS))
            .map(|page| slot * 64 + page)
            .collect();
        let decode = previous > 0 && count == 1;
        batch.push(BatchItemV1::new(
            request,
            decode,
            count as u32,
            true,
            inputs.to_vec(),
        ));
        appends.push(BatchAppendV1::new(previous, count, end, pages, None));
        slots.push(slot);
    }
    BatchRequestV1::new(BatchWireVersionV1::V3, batch, appends).with_state_slots(slots)
}

fn run(host: &FakeHost, instance: &mut BundleInstanceV1, name: &str, request: BatchRequestV1) {
    let mut sink = ViewSink::default();
    host.note(format!("step {name}"));
    instance
        .execute(&request.encode().unwrap(), &mut sink)
        .unwrap_or_else(|error| panic!("{name}: {error:?} {:?}", host.logs.borrow()));
    for view in &sink.0 {
        host.note(format!("{name} view {:?}", view.raw()));
    }
}

fn launched(host: &FakeHost, from: usize) -> Vec<String> {
    host.launches.borrow()[from..]
        .iter()
        .map(|(_, name, _)| name.clone())
        .collect()
}

fn count(names: &[String], symbol: &str) -> usize {
    names.iter().filter(|name| *name == symbol).count()
}

const PROMPT: [u32; 3] = [1, 2, 3];

fn long_prompt() -> Vec<u32> {
    (0..2050).map(|token| token % 1000 + 5).collect()
}

#[test]
fn repeated_ordinals_log_why_create_is_refused() {
    for shape in Shape::ALL {
        let host = Rc::new(FakeHost::at(shape));
        let mut record = configuration_at(shape, host.layout(), &[]);
        let first = bundle_runtime::config::MAGIC.len() + 4 + shape.topology().id.len() + 4;
        record[first + 4..first + 8].copy_from_slice(&0_i32.to_le_bytes());
        assert_eq!(
            api()
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

#[test]
fn create_resolves_every_entry_and_destroy_releases_everything() {
    let host = Rc::new(FakeHost::default());
    let mut instance = create(&host);
    assert_resolved(&host);
    run(
        &host,
        &mut instance,
        "prefill",
        step(&[(10, 2, 0, &PROMPT)]),
    );
    instance.commit().unwrap();
    run(&host, &mut instance, "decode", step(&[(10, 2, 3, &[7])]));
    instance.commit().unwrap();
    instance.retire_request(10).unwrap();
    instance.destroy().unwrap();
    assert_eq!(host.released.get(), host.acquired.get());
}

/// The host resolved every symbol of the cubins its topology loads: the
/// TP4 cubins, which also hold the rank-invariant entries, and its own
/// per-rank cubins, and no other topology's; the wide all-reduce's only
/// when the instance serves graph buckets above 32 rows; the
/// context-parallel cubins' only above size 1, which these instances do not
/// run. The padded HC candidate is resolved only under its explicit opt-in.
fn assert_resolved(host: &FakeHost) {
    let names = host
        .symbol_names
        .borrow()
        .values()
        .cloned()
        .collect::<BTreeSet<_>>();
    let shape = host.shape();
    let wide = host.served().next_power_of_two() > DECODE_ENTRY_ROWS;
    let hc = crate::instance::hc::enabled()
        && crate::GEOMETRY_HC_PRECOMPUTE
            .iter()
            .any(|(id, available)| *id == shape.geometry.id && *available);
    for cubin in crate::CUBINS {
        let loaded = crate::GEOMETRY_CUBINS.iter().any(|(name, id, ranks)| {
            *name == cubin.name
                && (*id == "*" || *id == shape.geometry.id)
                && (*ranks == 0 || *ranks == shape.ranks())
        }) && !crate::precision::mixed_module(cubin.name)
            && !crate::F32_NORM_CUBINS.contains(&cubin.name)
            && (wide || !cubin.name.ends_with("_decode_wide"))
            && (hc || cubin.name != sm120::HC_PRECOMPUTE_CUBIN)
            && !model_qwen38_contract::sm120::context_parallel::is_cubin(cubin.name);
        for symbol in cubin.symbols {
            let selected = loaded
                && (*symbol != sm120::HC_PADDED_MIX || crate::instance::hc::padded_enabled());
            assert_eq!(names.contains(*symbol), selected, "{symbol}");
        }
    }
}

#[test]
fn every_topology_creates_resolves_its_entries_and_releases_everything() {
    for overlap in [false, true] {
        let _arm = overlap_arm(overlap);
        let shape = Shape::TP2;
        let host = Rc::new(FakeHost::at(shape));
        let mut instance = create(&host);
        assert_resolved(&host);
        // Overlap adds one communication stream per rank.
        assert_eq!(
            host.journal
                .borrow()
                .iter()
                .filter(|line| line.starts_with("stream "))
                .count(),
            shape.ranks() as usize * if overlap { 2 } else { 1 }
        );
        instance.destroy().unwrap();
        assert_eq!(host.released.get(), host.acquired.get());
    }
}

#[test]
fn prefill_decode_and_mixed_steps_launch_cataloged_entries_in_bounds() {
    // These fixed counts describe the whole-chunk default program.
    let _default_overlap = legacy_overlap_off();
    let host = Rc::new(FakeHost::default());
    let mut instance = create(&host);
    let before = host.launches.borrow().len();
    let long = long_prompt();
    run(
        &host,
        &mut instance,
        "prefill",
        step(&[(10, 0, 0, &PROMPT), (11, 1, 0, &long)]),
    );
    let prefill = launched(&host, before);
    // Two QSA pieces for 2053 rows: scores and top-k twice per full layer.
    assert_eq!(
        count(&prefill, "chop_qsa_qwen38_block_scores_prefill_v1"),
        4 * 12 * 2
    );
    assert_eq!(count(&prefill, "chop_qsa_qwen38_block_topk_v1"), 4 * 12 * 2);
    assert_eq!(
        count(&prefill, "chop_linear_qwen38_delta_prefill_scan_v1"),
        4 * 36
    );
    // PLE runs one n-gram and one convolution launch per item.
    assert_eq!(count(&prefill, "chop_row_qwen38_ngram_v1"), 4 * 2);
    assert_eq!(
        count(&prefill, "chop_row_qwen38_ple_conv_prefill_v1"),
        4 * 2
    );
    // 2053 rows take the phased all-reduce pair with 128 workers.
    assert_eq!(count(&prefill, "chop_comm_qwen38_allreduce_hc_v1"), 0);
    for symbol in prefill_comm_symbols() {
        assert_eq!(count(&prefill, symbol), 4 * 96);
        assert!(
            host.launches.borrow()[before..]
                .iter()
                .filter(|(_, name, _)| name == symbol)
                .all(|(_, _, grid)| *grid == [136, 1, 1])
        );
    }
    assert_eq!(
        count(&prefill, "chop_moe_qwen38_permute_scatter_v1"),
        4 * 48
    );
    // The prefill saved the ring before writing it.
    assert_eq!(host.copies.borrow().len(), 4);
    let committed = host.launches.borrow().len();
    instance.commit().unwrap();
    // Commit copies each rank's staged PLE state back to its slots.
    assert_eq!(
        launched(&host, committed),
        vec!["chop_qwen38_copy_entries_v1".to_string(); 4]
    );
    assert_eq!(host.captures.get(), 0);

    let before = host.launches.borrow().len();
    run(
        &host,
        &mut instance,
        "decode",
        step(&[(10, 0, 3, &[9]), (11, 1, 2050, &[8])]),
    );
    let decode = launched(&host, before);
    let score_grids: Vec<_> = host.launches.borrow()[before..]
        .iter()
        .filter(|(_, name, _)| name == "chop_qsa_qwen38_block_scores_decode_v1")
        .map(|(_, _, grid)| *grid)
        .collect();
    assert_eq!(score_grids, vec![[376, 1, 1]; 4 * 12]);
    assert_eq!(count(&decode, "chop_linear_qwen38_delta_verify_v1"), 4 * 36);
    assert_eq!(
        count(&decode, "chop_qsa_qwen38_attn_bf16_decode_merge_v1"),
        4 * 12
    );
    assert_eq!(count(&decode, "chop_moe_qwen38_decode_route_t2_v1"), 4 * 48);
    // One n-gram and one PLE convolution launch covers the bucket.
    assert_eq!(count(&decode, "chop_row_qwen38_ngram_decode_v1"), 4);
    assert_eq!(
        count(&decode, "chop_row_qwen38_ple_conv_items_decode_v1"),
        4
    );
    assert_eq!(count(&decode, "chop_row_qwen38_ngram_v1"), 0);
    assert_eq!(count(&decode, decode_comm(4, 2)), 4 * 96);
    assert_eq!(count(&decode, "chop_comm_qwen38_allreduce_hc_v1"), 0);
    // The fused up mix replaces the bottleneck, up and mix launches.
    assert_eq!(
        count(&decode, "chop_gemm_qwen38_hc_up_mix_decode_v1"),
        4 * 97
    );
    for unfused in [
        "chop_row_qwen38_hc_bottleneck_v1",
        "chop_gemm_qwen38_hc_up_decode_v1",
        "chop_row_qwen38_hc_mix_v1",
    ] {
        assert_eq!(count(&decode, unfused), 0, "{unfused}");
    }
    assert_eq!(count(&prefill, "chop_row_qwen38_hc_mix_v1"), 4 * 97);
    assert!(!decode.iter().any(|name| name.contains("prefill")));
    assert!(!decode.iter().any(|name| name.contains("scatter")));
    let captures = host.captures.get();
    assert!(captures >= 4, "every rank captures its bucket-2 graph");
    instance.commit().unwrap();

    // A mixed step: a new prefill and the two decodes, which replay the
    // bucket-2 graphs without capturing again.
    let replays = host.replays.get();
    run(
        &host,
        &mut instance,
        "mixed",
        step(&[(12, 2, 0, &PROMPT), (10, 0, 4, &[5]), (11, 1, 2051, &[6])]),
    );
    assert_eq!(host.captures.get(), captures);
    assert_eq!(host.replays.get() - replays, captures);
    instance.commit().unwrap();
    assert_eq!(crate::instance::owner(instance.handle(), 2), Some(12));
}

#[test]
fn decode_graph_holds_every_launch_of_the_lowered_program() {
    let host = Rc::new(FakeHost::default());
    let mut instance = create(&host);
    run(
        &host,
        &mut instance,
        "prefill",
        step(&[(10, 0, 0, &PROMPT)]),
    );
    instance.commit().unwrap();
    run(&host, &mut instance, "decode", step(&[(10, 0, 3, &[9])]));
    let request = step(&[(10, 0, 3, &[9])]);
    let packed = crate::inputs::PackedStep::pack::<Qwen>(
        crate::contract::MODEL_LIMITS,
        &request,
        KV_PAGES,
        |_| Some(0),
    )
    .unwrap();
    let program = crate::instance::plan_probe(instance.handle(), &packed.decode, 0).unwrap();
    let nodes = host
        .graph_nodes
        .borrow()
        .values()
        .copied()
        .collect::<Vec<_>>();
    assert_eq!(nodes.iter().sum::<usize>(), 4 * program.len());
    let pieces = crate::graphs::pieces(program.len());
    assert_eq!(nodes.len(), 4 * pieces.len());
    // Every bound span of every launch lies in a live buffer.
    for launch in &program {
        crate::dispatch::arguments(&launch.arguments, &launch.bindings).unwrap();
    }
    instance.commit().unwrap();
}

#[test]
fn recurrent_slots_follow_commit_cancel_and_retire() {
    let host = Rc::new(FakeHost::default());
    let mut instance = create(&host);
    run(
        &host,
        &mut instance,
        "prefill",
        step(&[(10, 3, 0, &PROMPT)]),
    );
    // Cancel restores the ring the prefill saved, and the slot stays free.
    let copies = host.copies.borrow().len();
    instance.cancel().unwrap();
    let restored = host.copies.borrow()[copies..].to_vec();
    assert_eq!(restored.len(), 4);
    for (source, destination) in restored {
        assert_eq!(
            source.offset,
            destination.offset + 32 * RING_STATE_SLOT_BYTES / 2
        );
    }
    assert_eq!(crate::instance::owner(instance.handle(), 3), None);
    run(
        &host,
        &mut instance,
        "prefill again",
        step(&[(10, 3, 0, &PROMPT)]),
    );
    instance.commit().unwrap();
    assert_eq!(crate::instance::owner(instance.handle(), 3), Some(10));

    // Cancellation invalidates the in-place state, so retry requires a
    // fresh prefill after host retirement.
    run(&host, &mut instance, "decode", step(&[(10, 3, 3, &[7])]));
    instance.cancel().unwrap();
    assert_eq!(crate::instance::owner(instance.handle(), 3), None);
    let mut sink = ViewSink::default();
    assert!(
        instance
            .execute(&step(&[(10, 3, 3, &[7])]).encode().unwrap(), &mut sink)
            .is_err()
    );
    instance.retire_request(10).unwrap();
    run(
        &host,
        &mut instance,
        "recompute",
        step(&[(10, 3, 0, &PROMPT)]),
    );
    instance.commit().unwrap();
    run(
        &host,
        &mut instance,
        "decode after recompute",
        step(&[(10, 3, 3, &[7])]),
    );
    instance.commit().unwrap();

    // Another request can neither continue slot 3's state nor start there,
    // and request 10 continues only from the 4 tokens it committed.
    let invalid = Err(BundleAbiError::Status(BundleStatusV1::InvalidArgument));
    for refused in [
        step(&[(11, 3, 5, &[1])]),
        step(&[(11, 3, 0, &PROMPT)]),
        step(&[(10, 3, 3, &[1])]),
        step(&[(10, 3, 5, &[1])]),
        step(&[(10, 4, 4, &[1])]),
    ] {
        let mut sink = ViewSink::default();
        let launches = host.launches.borrow().len();
        assert_eq!(
            instance.execute(&refused.encode().unwrap(), &mut sink),
            invalid
        );
        assert_eq!(host.launches.borrow().len(), launches);
    }
    assert_eq!(crate::instance::owner(instance.handle(), 3), Some(10));
    run(&host, &mut instance, "decode", step(&[(10, 3, 4, &[1])]));
    instance.commit().unwrap();
}

#[test]
fn v5_position_axes_reach_the_uploaded_rope_positions() {
    let host = Rc::new(FakeHost::default());
    let mut instance = create(&host);
    let offset = crate::side::SideLayout::new(SERVED, SERVED).rope_positions as usize;
    // Every side arena of the step holds `axes` at its rope positions.
    let uploaded = |from: usize, axes: &[[i32; 3]]| {
        let expected = axes
            .iter()
            .flatten()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        host.uploads.borrow()[from..]
            .iter()
            .filter(|(_, bytes)| bytes.get(offset..offset + expected.len()) == Some(&expected[..]))
            .count()
    };
    let with_axes = |mut request: BatchRequestV1, axes: Vec<[i32; 3]>| {
        let item = request.items()[0].clone().with_position_axes(axes);
        request = BatchRequestV1::new(
            BatchWireVersionV1::V3,
            vec![item],
            request.appends().to_vec(),
        )
        .with_state_slots(request.state_slots().unwrap().to_vec());
        assert!(request.encode().unwrap().starts_with(b"chopbatch-v5"));
        request
    };
    let prefill = [[0, 0, 0], [1, 1, 2], [1, 2, 1]];
    let from = host.uploads.borrow().len();
    run(
        &host,
        &mut instance,
        "prefill",
        with_axes(step(&[(10, 3, 0, &PROMPT)]), prefill.to_vec()),
    );
    instance.commit().unwrap();
    assert_eq!(uploaded(from, &prefill), RANKS as usize);
    let from = host.uploads.borrow().len();
    run(
        &host,
        &mut instance,
        "decode",
        with_axes(step(&[(10, 3, 3, &[7])]), vec![[40, 40, 40]]),
    );
    instance.commit().unwrap();
    assert_eq!(uploaded(from, &[[40, 40, 40]]), RANKS as usize);
}

/// A decode replay that fails after two ranks queued their first piece
/// leaves them waiting at collectives the other ranks never join: the
/// instance is poisoned, queues nothing more, refuses every later call and
/// keeps its resources.
#[test]
fn a_partial_graph_replay_poisons_the_instance() {
    let host = Rc::new(FakeHost::default());
    let mut instance = create(&host);
    run(
        &host,
        &mut instance,
        "prefill",
        step(&[(10, 0, 0, &PROMPT)]),
    );
    instance.commit().unwrap();
    run(&host, &mut instance, "decode", step(&[(10, 0, 3, &[9])]));
    instance.commit().unwrap();
    let matched = |host: &FakeHost| {
        let counts = host.collectives.borrow();
        counts.len() == RANKS as usize
            && counts
                .values()
                .all(|count| Some(count) == counts.values().next())
    };
    assert!(matched(&host));
    host.fail_graph_launch.set(Some(2));
    let journal = host.journal.borrow().len();
    let mut sink = ViewSink::default();
    assert_eq!(
        instance.execute(&step(&[(10, 0, 4, &[8])]).encode().unwrap(), &mut sink),
        Err(BundleAbiError::Status(BundleStatusV1::HostError))
    );
    assert!(!matched(&host));
    let added = host.journal.borrow()[journal..].to_vec();
    let last = added
        .iter()
        .rposition(|line| line.starts_with("replay "))
        .expect("replayed pieces");
    assert!(
        added[last + 1..].iter().all(|line| ![
            "launch ",
            "replay ",
            "copy ",
            "upload ",
            "synchronize "
        ]
        .iter()
        .any(|prefix| line.starts_with(prefix))),
        "{:?}",
        &added[last + 1..]
    );
    let released = host.released.get();
    assert!(
        instance
            .execute(&step(&[(10, 0, 4, &[8])]).encode().unwrap(), &mut sink)
            .is_err()
    );
    assert!(instance.retire_request(10).is_err());
    assert!(instance.destroy().is_err());
    assert!(instance.is_poisoned());
    assert_eq!(host.released.get(), released);
    assert_eq!(host.journal.borrow().len(), journal + added.len());
}

/// Once its collectives are unmatched, the instance refuses every
/// prefix-state call too: none reaches a stream.
#[test]
fn a_poisoned_instance_refuses_prefix_state_calls() {
    let host = Rc::new(FakeHost::default());
    let mut instance = create(&host);
    run(
        &host,
        &mut instance,
        "prefill",
        step(&[(10, 0, 0, &PROMPT)]),
    );
    instance.commit().unwrap();
    run(&host, &mut instance, "decode", step(&[(10, 0, 3, &[9])]));
    instance.commit().unwrap();
    let pages = snapshot_pages(&host, 20);
    host.fail_graph_launch.set(Some(2));
    assert_eq!(
        instance.execute(
            &step(&[(10, 0, 4, &[8])]).encode().unwrap(),
            &mut ViewSink::default()
        ),
        Err(BundleAbiError::Status(BundleStatusV1::HostError))
    );
    let journal = host.journal.borrow().len();
    let refused = BundleAbiError::Status(BundleStatusV1::InvalidState);
    assert_eq!(instance.describe_prefix_state(), Err(refused.clone()));
    assert_eq!(
        instance.capture_prefix_state(10, 4, &pages),
        Err(refused.clone())
    );
    assert_eq!(
        instance.restore_prefix_state(11, &PROMPT, &pages),
        Err(refused.clone())
    );
    assert_eq!(
        instance.restore_prefix_state_in_slot(11, &PROMPT, &pages, 1),
        Err(refused)
    );
    assert_eq!(host.journal.borrow().len(), journal);
}

/// A destroyed or unknown handle names no instance: every prefix-state
/// entry reports it stale.
#[test]
fn a_stale_handle_refuses_prefix_state_calls() {
    use crate::instance::Instance;
    use bundle_runtime::entry::{
        capture_prefix_state, describe_prefix_state, restore_prefix_state,
        restore_prefix_state_in_slot,
    };
    let host = Rc::new(FakeHost::default());
    let mut instance = create(&host);
    let destroyed = instance.handle();
    instance.destroy().unwrap();
    let request = || PrefixStateRequestV1 {
        request_id: 10,
        tokens: PROMPT.len() as u64,
        token_ids: PROMPT.to_vec(),
        pages: snapshot_pages(&host, 20),
    };
    let stale = BundleStatusV1::StaleHandle;
    for handle in [destroyed, BundleHandleV1::from_raw(0)] {
        assert_eq!(
            describe_prefix_state::<Instance>(handle),
            BundlePrefixStateInfoV1::failure(stale)
        );
        assert_eq!(capture_prefix_state::<Instance>(handle, request()), stale);
        assert_eq!(restore_prefix_state::<Instance>(handle, request()), stale);
        assert_eq!(
            restore_prefix_state_in_slot::<Instance>(handle, request(), 1),
            stale
        );
    }
}

/// A prefill whose second rank queues its phased scatter but fails to queue
/// the gather leaves the other ranks waiting for its gather flags: the
/// instance is poisoned, queues nothing more, refuses every later call and
/// keeps its resources.
#[test]
fn a_failed_phased_gather_launch_poisons_the_instance() {
    let host = Rc::new(FakeHost::default());
    let mut instance = create(&host);
    let [scatter, gather] = prefill_comm_symbols();
    // Rank 0 queues its first pair; rank 1 queues its scatter and fails.
    *host.fail_symbol_launch.borrow_mut() = Some((gather.into(), 1));
    let journal = host.journal.borrow().len();
    let mut sink = ViewSink::default();
    assert_eq!(
        instance.execute(
            &step(&[(10, 0, 0, &long_prompt())]).encode().unwrap(),
            &mut sink
        ),
        Err(BundleAbiError::Status(BundleStatusV1::HostError))
    );
    assert!(
        host.fail_symbol_launch.borrow().is_none(),
        "the gather failed"
    );
    let added = host.journal.borrow()[journal..].to_vec();
    let launched = |symbol: &str| {
        added
            .iter()
            .filter(|line| line.starts_with("launch ") && line.contains(symbol))
            .count()
    };
    assert_eq!((launched(scatter), launched(gather)), (2, 1));
    let last = added
        .iter()
        .rposition(|line| line.contains(scatter))
        .unwrap();
    assert!(
        added[last + 1..].iter().all(|line| ![
            "launch ",
            "replay ",
            "copy ",
            "upload ",
            "synchronize "
        ]
        .iter()
        .any(|prefix| line.starts_with(prefix))),
        "{:?}",
        &added[last + 1..]
    );
    assert!(
        host.logs
            .borrow()
            .iter()
            .any(|line| line.contains("the instance is poisoned"))
    );
    let released = host.released.get();
    assert!(
        instance
            .execute(&step(&[(11, 1, 0, &PROMPT)]).encode().unwrap(), &mut sink)
            .is_err()
    );
    assert!(instance.destroy().is_err());
    assert!(instance.is_poisoned());
    assert_eq!(host.released.get(), released);
    assert_eq!(host.journal.borrow().len(), journal + added.len());
}

#[test]
fn a_chunk_without_logits_skips_the_head_and_continues_its_state() {
    for overlap in [false, true] {
        let _arm = overlap_arm(overlap);
        let host = Rc::new(FakeHost::default());
        let mut instance = create(&host);
        let long = long_prompt();
        let chunk = |previous: u64, inputs: &[u32], logits: bool| {
            let end = previous + inputs.len() as u64;
            BatchRequestV1::new(
                BatchWireVersionV1::V3,
                vec![BatchItemV1::new(
                    30,
                    false,
                    inputs.len() as u32,
                    logits,
                    inputs.to_vec(),
                )],
                vec![BatchAppendV1::new(
                    previous,
                    inputs.len() as u64,
                    end,
                    (0..end.div_ceil(PAGE_TOKENS)).collect(),
                    None,
                )],
            )
            .with_state_slots(vec![4])
        };
        let before = host.launches.borrow().len();
        run(
            &host,
            &mut instance,
            "first chunk",
            chunk(0, &long[..1024], false),
        );
        let first = launched(&host, before);
        assert_eq!(
            count(&first, "chop_comm_gather_rows_bf16_v1"),
            4 * 4 * if overlap { 2 } else { 1 }
        );
        assert!(!first.iter().any(|name| name.contains("head")));
        instance.commit().unwrap();
        let before = host.launches.borrow().len();
        run(
            &host,
            &mut instance,
            "second chunk",
            chunk(1024, &long[1024..], true),
        );
        let second = launched(&host, before);
        // One head row per rank runs on the one-token F32 head entry.
        assert_eq!(count(&second, "chop_gemm_qwen38_head_f32_m1_v1"), 4);
        assert_eq!(count(&second, "chop_gemm_qwen38_head_m1_v1"), 0);
        // Continuing prefill reads and writes the same slot, without copies.
        assert!(
            !second
                .iter()
                .any(|name| name == "chop_qwen38_copy_entries_v1")
        );
        instance.commit().unwrap();
        assert_eq!(crate::instance::owner(instance.handle(), 4), Some(30));
    }
}

#[test]
fn retire_frees_the_slot_for_a_fresh_request() {
    let host = Rc::new(FakeHost::default());
    let mut instance = create(&host);
    run(
        &host,
        &mut instance,
        "prefill",
        step(&[(10, 5, 0, &PROMPT)]),
    );
    instance.commit().unwrap();
    instance.retire_request(10).unwrap();
    assert_eq!(crate::instance::owner(instance.handle(), 5), None);
    run(&host, &mut instance, "reuse", step(&[(11, 5, 0, &PROMPT)]));
    instance.commit().unwrap();
    assert_eq!(crate::instance::owner(instance.handle(), 5), Some(11));
}

#[test]
fn scored_prompt_rows_take_f32_logits_of_every_row() {
    let host = Rc::new(FakeHost::default());
    let mut instance = create(&host);
    let before = host.launches.borrow().len();
    let scored = BatchRequestV1::new(
        BatchWireVersionV1::V3,
        vec![BatchItemV1::new(12, false, 3, true, PROMPT.to_vec()).with_prompt_rows(true)],
        vec![BatchAppendV1::new(0, 3, 3, vec![0], None)],
    )
    .with_state_slots(vec![6]);
    let mut sink = ViewSink::default();
    instance
        .execute(&scored.encode().unwrap(), &mut sink)
        .unwrap_or_else(|error| panic!("{error:?} {:?}", host.logs.borrow()));
    let names = launched(&host, before);
    assert_eq!(
        count(&names, "chop_gemm_qwen38_head_f32_prefill_v1"),
        RANKS as usize
    );
    assert!(
        !names
            .iter()
            .any(|name| name.starts_with("chop_gemm_qwen38_head_m"))
    );
    let views = sink.0.iter().map(|view| view.raw()).collect::<Vec<_>>();
    assert_eq!(views.len(), 1);
    assert_eq!(
        views[0].element_type,
        runtime_bundle_abi::BUNDLE_LOGITS_ELEMENT_V1_F32
    );
    assert_eq!((views[0].row_count, views[0].live_width), (3, VOCAB));
    instance.commit().unwrap();
}

/// A prefill of `SCORED_ROWS - 1` scored prompt rows and one unscored
/// final row shares the scored F32 buffers: one compact head row list, one
/// F32 head over every row, one gather of all rows into `gathered_scored`
/// and consecutive F32 views, while the decode item beside it gathers into
/// `gathered_decode`. One more prefill logits row is refused before any
/// launch.
#[test]
fn scored_and_unscored_prefill_rows_share_the_f32_buffers() {
    let host = Rc::new(FakeHost::default());
    // Row-owned decode exchanges instead of gathering; logits.rs covers it.
    let mut instance = crate::instance::with_row_owners(false, || create(&host));
    run(
        &host,
        &mut instance,
        "decode item",
        step(&[(60, 2, 0, &PROMPT)]),
    );
    instance.commit().unwrap();
    let scored = SCORED_ROWS as usize - 1;
    let prompt = (0..scored as u32)
        .map(|token| token % 1000 + 5)
        .collect::<Vec<_>>();
    let pages = |slot: u64, end: u64| {
        (0..end.div_ceil(PAGE_TOKENS))
            .map(|page| slot * 64 + page)
            .collect()
    };
    let mixed = |scored: &[u32]| {
        BatchRequestV1::new(
            BatchWireVersionV1::V3,
            vec![
                BatchItemV1::new(61, false, scored.len() as u32, true, scored.to_vec())
                    .with_prompt_rows(true),
                BatchItemV1::new(62, false, 3, true, PROMPT.to_vec()),
                BatchItemV1::new(60, true, 1, true, vec![4]),
            ],
            vec![
                BatchAppendV1::new(
                    0,
                    scored.len() as u64,
                    scored.len() as u64,
                    pages(0, scored.len() as u64),
                    None,
                ),
                BatchAppendV1::new(0, 3, 3, pages(1, 3), None),
                BatchAppendV1::new(3, 1, 4, pages(2, 4), None),
            ],
        )
        .with_state_slots(vec![0, 1, 2])
    };
    // One row too many: refused before anything launches or uploads.
    let over = (0..scored as u32 + 1)
        .map(|token| token % 1000 + 5)
        .collect::<Vec<_>>();
    let (launches, uploads) = (host.launches.borrow().len(), host.uploads.borrow().len());
    let mut sink = ScoredSink::default();
    assert_eq!(
        instance.execute(&mixed(&over).encode().unwrap(), &mut sink),
        Err(BundleAbiError::Status(BundleStatusV1::Unsupported))
    );
    assert_eq!(host.launches.borrow().len(), launches);
    assert_eq!(host.uploads.borrow().len(), uploads);
    assert!(sink.0.is_empty());
    assert!(
        host.logs
            .borrow()
            .iter()
            .any(|line| line.contains(&format!("scores at most {SCORED_ROWS} prompt rows")))
    );
    // Every scored row, then the unscored item's final row (its third,
    // at step row `scored + 2`), in one compact head row list.
    let rows = (0..scored as u32)
        .chain([scored as u32 + 2])
        .flat_map(u32::to_le_bytes)
        .collect::<Vec<_>>();
    let before = host.launches.borrow().len();
    let uploaded = host.uploads.borrow().len();
    let mut sink = ScoredSink::default();
    instance
        .execute(&mixed(&prompt).encode().unwrap(), &mut sink)
        .unwrap_or_else(|error| panic!("{error:?} {:?}", host.logs.borrow()));
    assert!(
        host.uploads.borrow()[uploaded..]
            .iter()
            .any(|(_, bytes)| bytes.windows(rows.len()).any(|window| window == rows)),
        "no upload holds the compact head row list"
    );
    let spans = crate::instance::logits_spans(instance.handle());
    let launches = host.launches.borrow()[before..].to_vec();
    let arguments = host.launch_arguments.borrow()[before..].to_vec();
    let buffer = |argument: &KernelArgumentV1| match argument {
        KernelArgumentV1::DeviceBuffer(span) => *span,
        other => panic!("{other:?}"),
    };
    let targets = |argument: &KernelArgumentV1| match argument {
        KernelArgumentV1::DevicePointerArray(array) => {
            array.iter().map(|span| span.unwrap()).collect::<Vec<_>>()
        }
        other => panic!("{other:?}"),
    };
    let mut heads = 0;
    let mut gathers = [0; 2];
    for (index, (_, name, _)) in launches.iter().enumerate() {
        let arguments = &arguments[index];
        if name == "chop_gemm_qwen38_head_f32_prefill_v1" {
            // Output, live rows and the F32 flag.
            let output = buffer(&arguments[2]);
            assert!(spans.iter().any(|rank| output == rank[1]), "{output:?}");
            assert_eq!(arguments[5], KernelArgumentV1::ScalarU32(SCORED_ROWS));
            assert_eq!(arguments[9], KernelArgumentV1::ScalarU32(1));
            heads += 1;
        }
        if name != "chop_comm_gather_rows_bf16_v1" {
            continue;
        }
        // Logits gathers read a rank's local_scored or local_logits.
        let input = buffer(&arguments[0]);
        let Some(prefill) = spans.iter().find_map(|rank| {
            (input == rank[1])
                .then_some(true)
                .or((input == rank[0]).then_some(false))
        }) else {
            continue;
        };
        let expected = spans
            .iter()
            .map(|rank| rank[if prefill { 2 } else { 3 }])
            .collect::<Vec<_>>();
        assert_eq!(targets(&arguments[1]), expected);
        // The decode item runs in the one-item bucket.
        assert_eq!(
            arguments[5],
            KernelArgumentV1::ScalarU32(if prefill { SCORED_ROWS } else { 1 })
        );
        assert_eq!(
            arguments[6],
            KernelArgumentV1::ScalarU64(2 * u64::from(Shape::TP4.vocab_per_rank))
        );
        gathers[usize::from(!prefill)] += 1;
    }
    assert_eq!((heads, gathers), (RANKS as usize, [RANKS as usize; 2]));
    assert!(
        !launches
            .iter()
            .any(|(_, name, _)| name.starts_with("chop_gemm_qwen38_head_m"))
    );
    // The prefill views tile rank 0's gathered_scored from its start: the
    // scored rows, then the unscored row. The decode row is at the start of
    // rank 0's gathered_decode. Every view is F32.
    let row_bytes = u64::from(VOCAB) * 4;
    let mut views = sink.0.iter().map(|view| view.raw()).collect::<Vec<_>>();
    views.sort_by_key(|view| (view.buffer.buffer.raw, view.buffer.offset));
    let mut next = spans[0][2].offset;
    let mut prefill_rows = 0;
    for view in &views {
        assert_eq!(
            view.element_type,
            runtime_bundle_abi::BUNDLE_LOGITS_ELEMENT_V1_F32
        );
        assert_eq!(view.row_stride_elements, u64::from(VOCAB));
        if view.buffer.buffer == spans[0][3].buffer {
            assert_eq!(
                (view.buffer.offset, view.row_count),
                (spans[0][3].offset, 1)
            );
            continue;
        }
        assert_eq!(view.buffer.buffer, spans[0][2].buffer);
        assert_eq!(view.buffer.offset, next);
        next += u64::from(view.row_count) * row_bytes;
        prefill_rows += view.row_count;
    }
    assert_eq!(prefill_rows, SCORED_ROWS);
    assert_ne!(
        (spans[0][3].buffer, spans[0][3].offset),
        (spans[0][2].buffer, spans[0][2].offset)
    );
    instance.commit().unwrap();
}

/// Unscored prefill, decode and a full decode bucket all take F32 logits
/// from the F32 decode-bucket heads, as MiMo's default: the host's sampler,
/// logprobs, penalties and grammar read F32 rows of the whole vocabulary.
#[test]
fn every_logits_row_is_f32() {
    let host = Rc::new(FakeHost::default());
    // Row-owned decode exchanges instead of gathering; logits.rs covers it.
    let mut instance = crate::instance::with_row_owners(false, || create(&host));
    // (request, slot, previous tokens, inputs, page): one page per item.
    let request = |items: &[(u64, u64, u64, &[u32], u64)]| {
        let (batch, appends): (Vec<_>, Vec<_>) = items
            .iter()
            .map(|&(request, _, previous, inputs, page)| {
                let count = inputs.len() as u64;
                (
                    BatchItemV1::new(request, previous > 0, count as u32, true, inputs.to_vec()),
                    BatchAppendV1::new(previous, count, previous + count, vec![page], None),
                )
            })
            .unzip();
        BatchRequestV1::new(BatchWireVersionV1::V3, batch, appends)
            .with_state_slots(items.iter().map(|item| item.1).collect())
    };
    let prompts = (0..SERVED).map(|slot| [slot + 7]).collect::<Vec<_>>();
    let before = host.launches.borrow().len();
    let mut sink = ViewSink::default();
    instance
        .execute(
            &request(&[(40, 0, 0, &PROMPT, 0), (41, 1, 0, &PROMPT, 1)])
                .encode()
                .unwrap(),
            &mut sink,
        )
        .unwrap_or_else(|error| panic!("{error:?} {:?}", host.logs.borrow()));
    instance.commit().unwrap();
    // Two head rows (one per item) on the two-token F32 entry.
    assert_eq!(
        count(&launched(&host, before), "chop_gemm_qwen38_head_f32_m2_v1"),
        4
    );
    // Fill every other slot, then decode the whole 32-item bucket.
    for slot in 2..u64::from(SERVED) {
        let before = host.launches.borrow().len();
        run(
            &host,
            &mut instance,
            "fill",
            request(&[(50 + slot, slot, 0, &prompts[slot as usize], slot)]),
        );
        instance.commit().unwrap();
        assert_eq!(
            count(&launched(&host, before), "chop_gemm_qwen38_head_f32_m1_v1"),
            4
        );
    }
    let decode = (0..u64::from(SERVED))
        .map(|slot| match slot {
            0 => (40, 0, 3, &[4][..], 0),
            1 => (41, 1, 3, &[5][..], 1),
            _ => (50 + slot, slot, 1, &prompts[slot as usize][..], slot),
        })
        .collect::<Vec<_>>();
    let before = host.launches.borrow().len();
    let mut full = ViewSink::default();
    instance
        .execute(&request(&decode).encode().unwrap(), &mut full)
        .unwrap_or_else(|error| panic!("{error:?} {:?}", host.logs.borrow()));
    assert_eq!(
        count(&launched(&host, before), "chop_gemm_qwen38_head_f32_m32_v1"),
        4
    );
    let launches = host.launches.borrow();
    assert!(
        !launches
            .iter()
            .any(|(_, name, _)| name.starts_with("chop_gemm_qwen38_head_m"))
    );
    // Each target head's rank shard reaches rank 0 as F32: BF16 pairs of
    // twice the shard width.
    let arguments = host.launch_arguments.borrow();
    let mut gathers = 0;
    for (index, (stream, name, _)) in launches.iter().enumerate() {
        if !name.starts_with("chop_gemm_qwen38_head_f32_m") {
            continue;
        }
        let (next, (_, gather, _)) = launches
            .iter()
            .enumerate()
            .skip(index + 1)
            .find(|(_, (next, _, _))| next == stream)
            .unwrap();
        assert_eq!(gather, "chop_comm_gather_rows_bf16_v1");
        // The gather reads the rows the head wrote.
        assert_eq!(arguments[next][0], arguments[index][2]);
        assert!(
            arguments[next].contains(&KernelArgumentV1::ScalarU64(
                2 * u64::from(Shape::TP4.vocab_per_rank)
            )),
            "{:?}",
            arguments[next]
        );
        gathers += 1;
    }
    // The first prefill, 30 fills and the decode, on every rank.
    assert_eq!(gathers, 4 * (1 + (SERVED as usize - 2) + 1));
    let views = sink
        .0
        .iter()
        .chain(&full.0)
        .map(|view| view.raw())
        .collect::<Vec<_>>();
    // One view per phase holds its consecutive rows.
    assert_eq!(
        views.iter().map(|view| view.row_count).collect::<Vec<_>>(),
        [2, SERVED]
    );
    for view in views {
        assert_eq!(
            view.element_type,
            runtime_bundle_abi::BUNDLE_LOGITS_ELEMENT_V1_F32
        );
        assert_eq!(
            (view.live_width, view.row_stride_elements),
            (VOCAB, u64::from(VOCAB))
        );
    }
    drop((launches, arguments));
    instance.commit().unwrap();
}

/// The declared state of `host`'s instance: its topology and KV format,
/// with the drafter when the host enabled speculation on its package.
fn host_state(host: &FakeHost) -> bundle_runtime::state::RequestState {
    let drafter = host.drafter.get();
    let kv = &KV_FORMATS[usize::from(drafter) + 2 * usize::from(host.fp8_kv.get())];
    model_qwen38_contract::state::request_state(
        host.shape(),
        kv,
        drafter && host.speculation.get().0 > 0,
    )
}

/// The declared table of `slot`'s snapshot in `host`'s instance, with the
/// verify rows the host enabled.
fn host_table(host: &FakeHost, slot: u32) -> bundle_runtime::snapshot::Table {
    let drafts = host.speculation.get().0;
    let extent = bundle_runtime::state::Extent {
        served: host.served(),
        bucket: host.served().next_power_of_two(),
        verify_rows: if drafts > 0 { drafts + 1 } else { 0 },
    };
    bundle_runtime::snapshot::Table::new(
        &host_state(host),
        extent,
        bundle_runtime::state::Holder::Slot(slot),
    )
    .unwrap()
}

/// Snapshot pages of one rank, from the per-rank state sizes of
/// docs/topology-qwen.md rather than the declared state: 36 linear layers'
/// convolution and delta entries, the PLE entry, each ring layer's 16 keys
/// of 128 BF16 and 16 positions of 3 words, the drafter's eight retained
/// four-stream rows, and the 24-byte header.
fn expected_snapshot_pages(ranks: u32, page_bytes: u64, drafter: bool) -> u64 {
    let (conv, delta): (u64, u64) = match ranks {
        2 => (30_720, 1_572_864),
        4 => (15_360, 786_432),
        _ => (7_680, 393_216),
    };
    let ring_layers = if drafter { 13 } else { 12 };
    let retained = if drafter { 8 * 4 * 2560 * 2 } else { 0 };
    let bytes = 36 * (conv + delta) + 184_448 + ring_layers * (16 * 128 * 2 + 16 * 12) + retained;
    (bytes + 24).div_ceil(page_bytes)
}

/// The bytes of `slot`'s committed state on `rank`, in snapshot order.
fn slot_bytes(host: &FakeHost, handle: BundleHandleV1, rank: usize, slot: u32) -> Vec<u8> {
    let views = crate::instance::state_views(handle, rank);
    host_table(host, slot)
        .descriptors
        .into_iter()
        .flat_map(|piece| {
            let view = views[piece.buffer];
            host.read(DeviceBufferSpanV1::new(
                view.buffer,
                view.offset + piece.offset,
                piece.length,
            ))
        })
        .collect()
}

/// Writes distinct bytes where a step would have left the committed state
/// of `slot`, and returns them per rank.
fn fill_slot(host: &FakeHost, handle: BundleHandleV1, slot: u32) -> Vec<Vec<u8>> {
    (0..host.shape().ranks.count())
        .map(|rank| {
            let views = crate::instance::state_views(handle, rank);
            let mut state = 0x9e37_79b9_7f4a_7c15_u64 ^ (rank as u64) << 56 ^ u64::from(slot);
            host_table(host, slot)
                .descriptors
                .into_iter()
                .flat_map(|piece| {
                    let bytes = (0..piece.length / 8)
                        .flat_map(|_| {
                            state = state
                                .wrapping_mul(6_364_136_223_846_793_005)
                                .wrapping_add(1_442_695_040_888_963_407);
                            state.to_le_bytes()
                        })
                        .collect::<Vec<_>>();
                    let view = views[piece.buffer];
                    host.write(
                        DeviceBufferSpanV1::new(
                            view.buffer,
                            view.offset + piece.offset,
                            piece.length,
                        ),
                        &bytes,
                    );
                    bytes
                })
                .collect()
        })
        .collect()
}

/// The snapshot bytes in `pages` on `rank`.
fn snapshot(host: &FakeHost, rank: u32, pages: &[u64]) -> Vec<u8> {
    let table = host_table(host, 0);
    let mut bytes = pages
        .iter()
        .flat_map(|page| {
            host.read(DeviceBufferSpanV1::new(
                BundleDeviceBufferHandleV1::from_parts(1, KV_SLOT + rank),
                page * table.page_bytes,
                table.page_bytes,
            ))
        })
        .collect::<Vec<_>>();
    bytes.truncate(table.header_at() as usize);
    bytes
}

/// Snapshot page IDs, out of order so a piece crosses between unrelated pages.
fn snapshot_pages(host: &FakeHost, first: u64) -> Vec<u64> {
    let pages = host_table(host, 0).pages();
    // A stride prime to the page count visits every page once.
    let gcd = |mut a: u64, mut b: u64| {
        while b != 0 {
            (a, b) = (b, a % b);
        }
        a
    };
    let stride = (11..).find(|&stride| gcd(stride, pages) == 1).unwrap();
    (0..pages)
        .map(|index| first + (index * stride) % pages)
        .collect()
}

#[test]
fn describe_reports_35_pages_and_the_host_restores_through_the_slot_tail() {
    let host = Rc::new(FakeHost::default());
    let mut instance = create(&host);
    assert!(instance.supports_prefix_cache());
    assert!(instance.supports_prefix_state_slot());
    // The table exports commit_accepted_v1 beside the slot restore.
    assert!(instance.supports_speculation());
    assert_eq!(instance.describe_prefix_state().unwrap(), 35);
    run(
        &host,
        &mut instance,
        "prefill",
        step(&[(10, 3, 0, &PROMPT)]),
    );
    instance.commit().unwrap();
    // The plain restore does not know the request's slot.
    assert_eq!(
        instance.restore_prefix_state(11, &PROMPT, &snapshot_pages(&host, 100)),
        Err(BundleAbiError::Status(BundleStatusV1::Unsupported))
    );
    instance.destroy().unwrap();
}

#[test]
fn a_restored_slot_holds_the_captured_bytes_and_continues_like_the_cold_one() {
    // TP2 places its n-gram table in host memory, as its package must.
    for shape in Shape::ALL {
        for fp8_kv in [false, true] {
            let host = Rc::new(FakeHost::at(shape));
            host.ngram_host.set(ngram_in_host_memory(shape));
            host.fp8_kv.set(fp8_kv);
            restored_slot_continues_like_the_cold_one(&host);
        }
    }
}

fn restored_slot_continues_like_the_cold_one(host: &Rc<FakeHost>) {
    let host = Rc::clone(host);
    let ranks = host.shape().ranks();
    let mut instance = create(&host);
    assert_eq!(
        instance.describe_prefix_state().unwrap(),
        expected_snapshot_pages(ranks, host.layout().1, false)
    );
    let handle = instance.handle();
    host.backing.set(true);
    run(
        &host,
        &mut instance,
        "prefill",
        step(&[(10, 3, 0, &PROMPT)]),
    );
    instance.commit().unwrap();
    // The first commit made entry 1 of slot 3 the committed one.
    let cold = fill_slot(&host, handle, 3);
    let first = snapshot_pages(&host, 100);
    let launches = host.launches.borrow().len();
    instance
        .capture_prefix_state(10, 3, &first)
        .unwrap_or_else(|error| {
            panic!(
                "{:?}: {error:?} {:?}",
                host.shape().ranks,
                host.logs.borrow()
            )
        });
    assert_eq!(
        launched(&host, launches),
        vec![
            crate::launch::per_rank(
                model_qwen38_contract::sm120::owned::PREFIX_COPY_INDEXED,
                host.shape()
            );
            ranks as usize
        ]
    );
    for rank in 0..ranks {
        let expected = &cold[rank as usize];
        assert!(
            snapshot(&host, rank, &first) == *expected,
            "rank {rank} snapshot"
        );
    }
    // Restore into another slot: entry 0 of slot 7 then holds the same bytes.
    instance
        .restore_prefix_state_in_slot(11, &PROMPT, &first, 7)
        .unwrap();
    assert_eq!(crate::instance::owner(handle, 7), Some(11));
    for (rank, cold) in cold.iter().enumerate() {
        assert!(
            slot_bytes(&host, handle, rank, 7) == *cold,
            "rank {rank} restore"
        );
    }
    // Capture, restore and capture again is byte-identical.
    let second = snapshot_pages(&host, 300);
    instance.capture_prefix_state(11, 3, &second).unwrap();
    for rank in 0..ranks {
        assert!(snapshot(&host, rank, &second) == snapshot(&host, rank, &first));
    }
    // Both continue with the same chunk: the restored item is planned as a
    // continuing sequence and runs the cold item's program.
    let before = host.launches.borrow().len();
    run(&host, &mut instance, "cold", step(&[(10, 3, 3, &[9, 8])]));
    let cold_names = launched(&host, before);
    instance.commit().unwrap();
    let before = host.launches.borrow().len();
    run(&host, &mut instance, "hit", step(&[(11, 7, 3, &[9, 8])]));
    assert_eq!(launched(&host, before), cold_names);
    instance.commit().unwrap();
    instance.retire_request(10).unwrap();
    instance.retire_request(11).unwrap();
    instance.destroy().unwrap();
    assert_eq!(host.released.get(), host.acquired.get());
}

/// The declared state (`model_qwen38_contract::state`) is what a capture
/// writes and a restore reads: the snapshot pages hold the declared pieces
/// of the slot's committed entry in order, then rank 0's header, and a
/// restore into another slot fills that slot's declared pieces with them.
#[test]
fn snapshots_hold_the_declared_pieces_and_header() {
    use bundle_runtime::state::{Extent, Holder};
    for shape in Shape::ALL {
        for fp8_kv in [false, true] {
            let host = Rc::new(FakeHost::at(shape));
            host.ngram_host.set(ngram_in_host_memory(shape));
            host.fp8_kv.set(fp8_kv);
            let mut instance = create(&host);
            let handle = instance.handle();
            host.backing.set(true);
            run(
                &host,
                &mut instance,
                "prefill",
                step(&[(10, 3, 0, &PROMPT)]),
            );
            instance.commit().unwrap();
            fill_slot(&host, handle, 3);
            let state = model_qwen38_contract::state::request_state(
                shape,
                &KV_FORMATS[2 * usize::from(fp8_kv)],
                false,
            );
            let extent = Extent {
                served: host.served(),
                bucket: host.served().next_power_of_two(),
                verify_rows: 0,
            };
            let declared = |rank: usize, slot| {
                let views = crate::instance::state_views(handle, rank);
                state
                    .pieces(extent, Holder::Slot(slot))
                    .unwrap()
                    .into_iter()
                    .flat_map(|piece| {
                        let view = views[piece.buffer];
                        host.read(DeviceBufferSpanV1::new(
                            view.buffer,
                            view.offset + piece.offset,
                            piece.length,
                        ))
                    })
                    .collect::<Vec<_>>()
            };
            let pages = snapshot_pages(&host, 100);
            assert_eq!(pages.len() as u64, state.snapshot_pages());
            instance.capture_prefix_state(10, 3, &pages).unwrap();
            let header = bundle_runtime::prefix::Header {
                tokens: 3,
                frontier: None,
                owner: Some(10),
                cursor: None,
            }
            .encode(
                &crate::prefix::HEADER,
                bundle_runtime::prefix::Stamp {
                    ranks: 0,
                    digest: 0,
                },
            )
            .unwrap();
            for rank in 0..shape.ranks.count() {
                let mut bytes = pages
                    .iter()
                    .flat_map(|page| {
                        host.device_bytes(DeviceBufferSpanV1::new(
                            BundleDeviceBufferHandleV1::from_parts(1, KV_SLOT + rank as u32),
                            page * state.page_bytes(),
                            state.page_bytes(),
                        ))
                    })
                    .collect::<Vec<_>>();
                bytes.truncate(state.snapshot_bytes() as usize);
                let (pieces, written) = bytes.split_at(bytes.len() - header.len());
                assert!(pieces == declared(rank, 3), "{shape:?} rank {rank}");
                if rank == 0 {
                    assert_eq!(written, header);
                }
            }
            instance
                .restore_prefix_state_in_slot(11, &PROMPT, &pages, 7)
                .unwrap();
            for rank in 0..shape.ranks.count() {
                assert!(
                    declared(rank, 7) == declared(rank, 3),
                    "rank {rank} restore"
                );
            }
            instance.retire_request(10).unwrap();
            instance.retire_request(11).unwrap();
            instance.destroy().unwrap();
        }
    }
}

/// The QSA ring's declared cancel save: a step with a prefill copies the
/// declared region, the whole live ring, to the adjacent shadow, and cancel
/// copies it back, so the ring holds its committed bytes again.
#[test]
fn a_cancelled_prefill_restores_the_declared_ring_region() {
    use bundle_runtime::state::{Cancel, Extent, Region, Save, Writer};
    let host = Rc::new(FakeHost::default());
    let mut instance = create(&host);
    let handle = instance.handle();
    run(
        &host,
        &mut instance,
        "prefill",
        step(&[(10, 3, 0, &PROMPT)]),
    );
    instance.commit().unwrap();
    let state = model_qwen38_contract::state::request_state(Shape::TP4, &KV_FORMATS[0], false);
    let ring = &state.buffers[3];
    assert_eq!(
        ring.cancel,
        Cancel::Restore(vec![Save {
            writer: Writer::Prefill,
            region: Region::All,
        }])
    );
    let extent = Extent {
        served: host.served(),
        bucket: host.served().next_power_of_two(),
        verify_rows: 0,
    };
    let (offset, length) = ring.region(extent, Region::All).unwrap();
    let (separate, shadow) = ring.shadow_offset(extent, offset).unwrap();
    assert!(!separate);
    let span = |rank: usize, at: u64| {
        let view = crate::instance::state_views(handle, rank)[3];
        DeviceBufferSpanV1::new(view.buffer, view.offset + at, length)
    };
    let ranks = RANKS as usize;
    let fill = |seed: u8| {
        for rank in 0..ranks {
            let bytes = (0..length)
                .map(|at| (at as u8).wrapping_mul(31) ^ seed ^ rank as u8)
                .collect::<Vec<_>>();
            host.sim
                .borrow_mut()
                .write(span(rank, offset), &bytes, &host.extents.borrow());
        }
    };
    let ring_copies = |from: usize| {
        host.copies.borrow()[from..]
            .iter()
            .copied()
            .filter(|(source, _)| source.length == length)
            .collect::<Vec<_>>()
    };
    fill(0x5a);
    let committed = (0..ranks)
        .map(|rank| host.device_bytes(span(rank, offset)))
        .collect::<Vec<_>>();
    let before = host.copies.borrow().len();
    run(
        &host,
        &mut instance,
        "continuation",
        step(&[(10, 3, 3, &[9, 8])]),
    );
    let saves = (0..ranks)
        .map(|rank| (span(rank, offset), span(rank, shadow)))
        .collect::<Vec<_>>();
    assert_eq!(ring_copies(before), saves);
    // What the step's kernels would have written.
    fill(0xc3);
    let before = host.copies.borrow().len();
    instance.cancel().unwrap();
    assert_eq!(
        ring_copies(before),
        saves
            .iter()
            .map(|&(live, saved)| (saved, live))
            .collect::<Vec<_>>()
    );
    for (rank, committed) in committed.iter().enumerate() {
        assert!(
            host.device_bytes(span(rank, offset)) == *committed,
            "rank {rank}"
        );
    }
    instance.retire_request(10).unwrap();
    instance.destroy().unwrap();
}

#[test]
fn cancel_retire_and_slot_reuse_around_a_restore() {
    let host = Rc::new(FakeHost::default());
    let mut instance = create(&host);
    let handle = instance.handle();
    host.backing.set(true);
    run(
        &host,
        &mut instance,
        "prefill",
        step(&[(10, 3, 0, &PROMPT)]),
    );
    instance.commit().unwrap();
    let cold = fill_slot(&host, handle, 3);
    let pages = snapshot_pages(&host, 100);
    instance.capture_prefix_state(10, 3, &pages).unwrap();
    instance
        .restore_prefix_state_in_slot(11, &PROMPT, &pages, 7)
        .unwrap();
    // Cancellation invalidates the restored entry; capture must refuse it.
    run(&host, &mut instance, "hit", step(&[(11, 7, 3, &[9, 8])]));
    instance.cancel().unwrap();
    let again = snapshot_pages(&host, 500);
    assert!(instance.capture_prefix_state(11, 3, &again).is_err());
    // Retiring a restored request frees its slot: a fresh request prefills
    // there, and another restore can take it.
    instance.retire_request(11).unwrap();
    assert_eq!(crate::instance::owner(handle, 7), None);
    run(&host, &mut instance, "fresh", step(&[(12, 7, 0, &PROMPT)]));
    instance.commit().unwrap();
    instance.retire_request(12).unwrap();
    instance
        .restore_prefix_state_in_slot(13, &PROMPT, &pages, 7)
        .unwrap();
    for (rank, cold) in cold.iter().enumerate() {
        assert!(slot_bytes(&host, handle, rank, 7) == *cold);
    }
    instance.retire_request(13).unwrap();
    instance.retire_request(10).unwrap();
    instance.destroy().unwrap();
}

#[test]
fn refused_captures_and_restores_change_nothing() {
    let host = Rc::new(FakeHost::default());
    let mut instance = create(&host);
    let handle = instance.handle();
    // Restores read the snapshot header back from the pages.
    host.backing.set(true);
    let pages = snapshot_pages(&host, 100);
    let invalid_state = Err(BundleAbiError::Status(BundleStatusV1::InvalidState));
    let invalid_argument = Err(BundleAbiError::Status(BundleStatusV1::InvalidArgument));
    run(
        &host,
        &mut instance,
        "prefill",
        step(&[(10, 3, 0, &PROMPT)]),
    );
    // Nothing is captured or restored while a step is open.
    assert_eq!(
        instance.capture_prefix_state(10, 3, &pages),
        Err(BundleAbiError::Lifecycle("capture_prefix_state"))
    );
    instance.commit().unwrap();
    // Pages no capture wrote hold no header, and a snapshot of another
    // length does not match: both restores are refused and leave the slot
    // free.
    assert_eq!(
        instance.restore_prefix_state_in_slot(11, &PROMPT, &pages, 4),
        invalid_argument
    );
    instance.capture_prefix_state(10, 3, &pages).unwrap();
    assert_eq!(
        instance.restore_prefix_state_in_slot(11, &PROMPT[..2], &pages, 4),
        invalid_argument
    );
    assert_eq!(crate::instance::owner(handle, 4), None);
    let launches = host.launches.borrow().len();
    // Unknown requests, other token counts and foreign or taken slots.
    assert_eq!(instance.capture_prefix_state(99, 3, &pages), invalid_state);
    assert_eq!(instance.capture_prefix_state(10, 2, &pages), invalid_state);
    assert_eq!(
        instance.restore_prefix_state_in_slot(11, &PROMPT, &pages, 3),
        invalid_state
    );
    assert_eq!(
        instance.restore_prefix_state_in_slot(10, &PROMPT, &pages, 4),
        invalid_state
    );
    assert_eq!(
        instance.restore_prefix_state_in_slot(11, &PROMPT, &pages, 32),
        invalid_state
    );
    assert_eq!(
        instance.restore_prefix_state_in_slot(11, &PROMPT, &pages, 1 << 40),
        invalid_argument
    );
    // Wrong, repeated or out-of-range snapshot pages.
    assert_eq!(
        instance.restore_prefix_state_in_slot(11, &PROMPT, &pages[1..], 4),
        invalid_argument
    );
    let mut repeated = pages.clone();
    repeated[1] = repeated[0];
    assert_eq!(
        instance.restore_prefix_state_in_slot(11, &PROMPT, &repeated, 4),
        invalid_argument
    );
    let mut outside = pages.clone();
    outside[0] = KV_PAGES;
    assert_eq!(
        instance.capture_prefix_state(10, 3, &outside),
        invalid_argument
    );
    assert_eq!(host.launches.borrow().len(), launches);
    assert_eq!(crate::instance::owner(handle, 4), None);
    assert!(!instance.is_poisoned());

    // A copy that fails to launch on one rank drains the ranks that
    // received work and leaves the slot free.
    *host.refuse_launch.borrow_mut() = Some((
        model_qwen38_contract::sm120::owned::PREFIX_COPY_INDEXED.to_string(),
        launches + 2,
    ));
    let journal = host.journal.borrow().len();
    assert!(
        instance
            .restore_prefix_state_in_slot(11, &PROMPT, &pages, 4)
            .is_err()
    );
    let synchronized = host.journal.borrow()[journal..]
        .iter()
        .filter(|line| line.starts_with("synchronize"))
        .count();
    assert_eq!(synchronized, 3);
    assert_eq!(crate::instance::owner(handle, 4), None);
    *host.refuse_launch.borrow_mut() = None;
    assert!(!instance.is_poisoned());
    // The refusals left nothing behind: the restore now succeeds.
    instance
        .restore_prefix_state_in_slot(11, &PROMPT, &pages, 4)
        .unwrap();
    assert_eq!(crate::instance::owner(handle, 4), Some(11));
    instance.retire_request(11).unwrap();
    // A drain that fails cannot confirm the copy finished: INTERNAL_ERROR,
    // which poisons the instance so the host never reuses the pages.
    host.refuse_synchronize.set(true);
    assert_eq!(
        instance.restore_prefix_state_in_slot(12, &PROMPT, &pages, 4),
        Err(BundleAbiError::Status(BundleStatusV1::InternalError))
    );
    host.refuse_synchronize.set(false);
    assert_eq!(crate::instance::owner(handle, 4), None);
    assert!(instance.is_poisoned());
}

/// Journal lines and FNV-1a of the journal, re-recorded when the program
/// or a kernel catalog it launches changes (last: F32 logits for every
/// row. Each rank resolves and releases the six `head_f32_m{M}` entries
/// and allocates `local_logits` at F32 and one F32
/// `gathered_decode` in place of the BF16 `gathered_prefill` and
/// `gathered_decode` (four fewer releases). The heads switch from
/// `head_m{M}` to `head_f32_m{M}` with `output_f32` set, a prefill head
/// writes `local_scored`, the logits gathers move twice the shard width
/// (124160 BF16 pairs) into `gathered_scored` or `gathered_decode`, the
/// views are F32 (element type 1) of twice the bytes, and graph IDs follow
/// 20 later.)
/// The 512-row cap changes only the sizes of the F32 prefill buffers: their
/// eight allocations and the spans the prefill heads and logits gathers bind
/// (24 lines). Launches, uploads, output views and ordering keep their bytes.
/// The MiMo comm cubin this bundle loads then gained the wide all-reduce
/// entries: their symbol resolves and releases per rank, and the handle IDs
/// after them, are the only changes, here and in `EXPECTED_FP8`,
/// `EXPECTED_TP2`.
/// The lower-latency decode all-reduce: each rank resolves and releases the
/// comm entries it added (`hc_decode_r{N}_v2` and MiMo's two
/// `allreduce_decode_*` entries of the shared comm cubin), so handle IDs
/// after them move, and the decode all-reduce launches name `_v2` with the
/// same arguments and plans.
/// Row-owned logits then became the TP4 default: each rank allocates,
/// zeroes and releases the exchange flags and epoch, the four decode
/// `chop_comm_gather_rows_bf16_v1` launches become
/// `chop_comm_exchange_rows_f32_v1`, six owner-stream synchronizations
/// precede the views, and the decode and mixed steps emit one more owner
/// view each (32 more lines; handle IDs after the new allocations shift).
/// TP2 followed; see `EXPECTED_TP2`.
/// Parallel routing: the two-row score launches take a 376-CTA grid instead of
/// `[2, 1024]`, and each rank resolves the C32 parallel route entry and
/// allocates and zeroes its four-byte ticket (six lines per rank with their
/// releases), which moves the graph handle IDs after them.
/// Default HC adds its five entry resolves/settings and module lifecycle at
/// each rank (60 lines), shifting subsequent handles. This two-row journal
/// retains its original launch entries. Explicit HC-off kept parallel routing's digest.
/// CONC-QWEN-STATE, with HC on and off: one committed entry halves linear pools; the index
/// arena adds checkpoint-copy indices. Sixteen prefill state-copy launches
/// disappear at TP4. The narrow verification budget stays at 32 rows.
/// After normalizing handles and excluding copies, launch symbols and their
/// order match acd3934dc. Only linear pool spans and offsets change in the
/// remaining launch parameters. Kernel bodies and locks match.
// The shared comm cubin's round-once prefill pair (then MiMo's opt-in
// CHOPINFER_MIMO_PREFILL_ROUND_ONCE) adds two entries each rank resolves and
// releases, 4 lines per rank in every journal here; handle IDs after them
// move and no launch changes. PREFIX-EXACT (2026-09-30) made that pair
// MiMo's only phased pair and removed the switch and the double-rounding
// gather (`chop_comm_allgather_scattered_norm_bf16_v2`), one entry fewer
// for each rank to resolve and release; Qwen launches neither.
// HF tables change only the 64 MiB RoPE upload hash once per rank.
// Audited against the prior journals: every other byte and line is unchanged.
// The grouped-MoE schedule table's swept rows change only the grid, block
// and shared memory of the tuned expert launches, and the per-rank
// `configure` lines of entries whose rings now need (or no longer need)
// the dynamic shared memory opt-in; no other line changes.
// PREFIX-EXACT (2026-09-30), against main 6ce3d0b6b's journals as launch
// multisets with handles named: QSA prefill runs one split (grid z 1 and no
// `attn_*_prefill_merge` launch, 48 lines here); the skinny `lin_ab` and
// `hc_inject` prefill launches take their fixed prefill split counts (5 and
// 10 per 32-row tile); fragment prefill GEMMs pass their fixed segments and
// the planner workspace holds eight partial tiles per CTA, so scratch
// offsets in later launch arguments move; the removed gather is one entry
// fewer to resolve and release per rank (8 lines). Every other launch keeps
// its symbol, geometry and argument shape, and the order is unchanged.
// The prefix anchor snapshot (2026-09-30), against main fb4fe6301's journals
// as launch multisets with handles named: the linear prefill and delta scan
// launches take the anchor entries and the QSA key writes the anchor slots,
// ends and ring (arguments added); each rank allocates the anchor PLE and
// ring spares (the drafter's anchor rows with MTP), resolves and releases
// `chop_qwen38_ple_anchor_v1`, and its pools and side layout grow, so
// offsets in other launches and the side uploads' bytes move. No journal
// step keeps an anchor, so no PLE anchor launch runs; every other launch
// keeps its symbol, geometry and argument shape, in the same order. The
// FP8, TP2 and HC-off journals change in the same classes.
// The F32 vision tower (2026-10-01) has 12 encoder entries
// instead of 14, and its attention entry takes the dynamic shared memory
// opt-in. Each rank resolves and releases two entries fewer and configures
// one more (3 lines fewer per rank), and later handle IDs move. Against
// the journals before it with the encoder entries' lines dropped and handles
// renamed by first appearance, every other line is unchanged.
// The encoder catalog generated from the vision declaration (2026-10-01)
// renames the 12 encoder entries to their derived `chop_enc_*`
// symbols (re-recorded when the symbols gained their identity digests); against main 142577401's journals with the encoder entries'
// lines dropped and IDs renamed by first appearance, every other line is
// unchanged, as are line counts.
// Checkpoint ownership tables (2026-10-03): two appended entries per rank
// add resolve/release lines. Side arrays grow and move pool-index operands.
// Against the archived base, every AR launch keeps its symbol, geometry,
// scalar operands and ordering after naming buffer handles; host provenance
// and snapshot-byte tests check the new entry bindings separately.
// Decode v3 adds four resolves and four releases to each prefill mode.
// TP4 renames 384 decode launches and TP2 renames 192 from v2 to v3,
// all at two prepared rows. Larger buckets retain v2. Every other
// canonical record matches the corresponding decode-v2 journal.
// Qualified prefill v2 adds scatter/gather resolves and releases per rank
// (16 lines at TP4, 8 at TP2). Flags grow from 384 to 1408 bytes at TP4
// and 336 to 848 at TP2: allocation, zero and comm argument spans change.
// Phased launches select reduce_scatter_v2 and allgather_hc_v2. These
// journals' long prefill keeps grid 136; short rows keep the one-shot entry.
// The v2 planner uses one worker per row instead of per four rows elsewhere.
// After normalizing shifted handles, every other journal line is identical.
// Snapshot bytes and copy plans are unchanged.
// Plain decode v1 left the shared comm cubin (2026-10-05): each rank
// resolves and releases one entry fewer (8 lines at TP4, 4 at TP2), and
// every other line matches after renaming handles. The Qwen decode v1
// entries left the decode cubins the same day: one entry fewer per rank and
// loaded decode cubin (8 lines at TP4; 8 at TP2, whose ranks load the r2
// and r4 cubins). The prefill v1 pair left comm_qwen38 the same day, and
// with it the v1 pins: each rank resolves and releases two entries fewer
// than the v2 journals did (16 lines at TP4, 8 at TP2), and every other
// line matches them after renaming handles.
// Create authenticates each text rank index against the host's image
// identity first (2026-10-07): the journal starts with one release line per
// rank (4 at TP4, 2 at TP2) and every other line is unchanged, in these and
// in the FP8 and TP2 pins.
// The stream expansion runs the shared broadcast (plan step 27, an identity
// change, accuracy unaffected): the bundle cubin drops
// `chop_qwen38_expand_streams_v1` and `row_copy_sm120`, loaded after the
// other cubins, holds `chop_row_copy_broadcast_streams_v1`, so each rank
// resolves the broadcast after the other entries and releases one more
// module (4 lines at TP4, 2 at TP2), and each expansion launch has grid y 1
// and two more arguments (16-byte vectors per row, streams). Every other
// line matches after renaming handles, in these, the FP8, TP2, DCP2,
// encoder and overlapped prefill pins and the precision lifecycle dumps.
const EXPECTED: (usize, u64) = (19760, 0x4E5FF8D412C4C758);
const EXPECTED_HC_OFF: (usize, u64) = (19700, 0x31B22B84A77AEF61);

/// Frozen diagnostics select the default kernels and the legacy overlap-off program.
fn with_default_journal_settings<T>(run: impl FnOnce() -> T) -> T {
    crate::instance::with_hc_precompute(true, || {
        crate::instance::with_hc_padded_mix(false, || {
            crate::instance::with_row_owners(true, || {
                let _overlap = legacy_overlap_off();
                run()
            })
        })
    })
}

/// Fixed legacy goldens explicitly describe the overlap-off arm.
fn legacy_overlap_off() -> impl Drop {
    overlap_arm(false)
}

fn overlap_arm(enabled: bool) -> impl Drop {
    struct Restore(Option<bool>);
    impl Drop for Restore {
        fn drop(&mut self) {
            crate::instance::FORCE_OVERLAP.with(|slot| slot.set(self.0));
        }
    }
    Restore(crate::instance::FORCE_OVERLAP.with(|slot| slot.replace(Some(enabled))))
}

/// The journal test's steps: a mixed prefill, a decode, a cancelled mixed
/// step, a retirement and destroy.
fn journal_steps(host: &Rc<FakeHost>) -> Vec<String> {
    let _default_overlap = legacy_overlap_off();
    crate::pipeline::take_recorded();
    let config = configuration_served(host.shape(), host.layout(), &[], host.served());
    lifecycle::execute(host, &config, &lifecycle::actions(), false).unwrap_or_else(|error| {
        panic!(
            "{} lifecycle: {error:?}, {:?}",
            host.shape().topology().id,
            host.logs.borrow()
        )
    });
    host.journal.borrow().clone()
}

/// Writes the launches this thread recorded since the last
/// `crate::pipeline::take_recorded` (lane, symbol, rows) to
/// `$CHOPINFER_QWEN38_ROWS_DIR/<scenario>.rows.tsv` when the variable is set,
/// so the rows of every launch, which journals omit, can be compared. They
/// must be every launch `hosts` received, in the order the hosts were
/// created, by stream and symbol (`check_recorded`): no submission bypasses
/// the recorder.
fn export_rows(scenario: &str, hosts: &[&FakeHost]) {
    let recorded = crate::pipeline::take_recorded();
    assert!(!recorded.is_empty(), "{scenario}: no launch was recorded");
    assert!(!hosts.is_empty(), "{scenario}: no host to check");
    check_recorded(scenario, &recorded, hosts);
    if let Some(root) = std::env::var_os("CHOPINFER_QWEN38_ROWS_DIR") {
        let root = std::path::PathBuf::from(root);
        std::fs::create_dir_all(&root).expect("create rows directory");
        let lines = recorded
            .into_iter()
            .map(|launch| launch.line)
            .collect::<Vec<_>>();
        std::fs::write(root.join(format!("{scenario}.rows.tsv")), lines.join("\n"))
            .expect("export launch rows");
    }
}

/// Writes every launch `host` received since the last
/// `crate::pipeline::take_recorded`, with its lane, symbol and rows, its
/// plan, the grid the host saw and its resolved arguments, to
/// `$CHOPINFER_QWEN38_LAUNCHES_DIR/<scenario>.launches.tsv` when the
/// variable is set, so two trees' launches compare in full.
fn export_launches(scenario: &str, host: &FakeHost) {
    let recorded = crate::pipeline::take_recorded();
    assert!(!recorded.is_empty(), "{scenario}: no launch was recorded");
    check_recorded(scenario, &recorded, &[host]);
    if let Some(root) = std::env::var_os("CHOPINFER_QWEN38_LAUNCHES_DIR") {
        let root = std::path::PathBuf::from(root);
        std::fs::create_dir_all(&root).expect("create launches directory");
        let launches = host.launches.borrow();
        let arguments = host.launch_arguments.borrow();
        // Each launch in the host's order with its stream's next record.
        let mut records = BTreeMap::<u32, std::collections::VecDeque<_>>::new();
        for record in &recorded {
            records.entry(record.stream).or_default().push_back(record);
        }
        let lines = launches
            .iter()
            .zip(arguments.iter())
            .map(|((stream, _, grid), arguments)| {
                let record = records.get_mut(stream).and_then(|own| own.pop_front());
                let record = record.expect("checked against the host's launches");
                format!(
                    "{}\t{}\t{grid:?}\t{arguments:?}",
                    record.line,
                    record.plan.as_deref().unwrap_or("-")
                )
            })
            .collect::<Vec<_>>();
        std::fs::write(
            root.join(format!("{scenario}.launches.tsv")),
            lines.join("\n"),
        )
        .expect("export launches");
    }
}

/// Checks that `recorded` holds every launch `hosts` received, in the
/// order the hosts were created, each stream's in its order by symbol.
/// Only each stream's order is compared: `pipeline::run_ranks` records a
/// rank's eager program when it is lowered and queues the ranks collective
/// by collective.
fn check_recorded(scenario: &str, recorded: &[crate::pipeline::Recorded], hosts: &[&FakeHost]) {
    let by_stream = |launches: &mut dyn Iterator<Item = (u32, String)>| {
        let mut streams = BTreeMap::<u32, Vec<String>>::new();
        for (stream, symbol) in launches {
            streams.entry(stream).or_default().push(symbol);
        }
        streams
    };
    let mut seen = recorded
        .iter()
        .map(|launch| (launch.stream, launch.symbol.clone()));
    for (index, host) in hosts.iter().enumerate() {
        let host_launches = host.launches.borrow();
        let launched = by_stream(
            &mut host_launches
                .iter()
                .map(|(stream, symbol, _)| (*stream, symbol.clone())),
        );
        let own = by_stream(&mut seen.by_ref().take(host_launches.len()));
        for stream in launched.keys().chain(own.keys()) {
            let own = own.get(stream).map_or(&[][..], Vec::as_slice);
            let launched = launched.get(stream).map_or(&[][..], Vec::as_slice);
            if let Some(at) =
                (0..own.len().max(launched.len())).find(|&at| own.get(at) != launched.get(at))
            {
                panic!(
                    "{scenario}: host {index} stream {stream}: recorded launch {at} is {:?}, \
                     the host's is {:?} ({} and {})",
                    own.get(at),
                    launched.get(at),
                    own.len(),
                    launched.len()
                );
            }
        }
    }
    assert_eq!(
        seen.count(),
        0,
        "{scenario}: launches recorded past the hosts'"
    );
}

/// The decode all-reduce entry for `ranks` ranks and `rows` prepared rows:
/// v3 at the LL prepared rows, v2 above.
pub(crate) fn decode_comm(ranks: u32, rows: u32) -> &'static str {
    use model_qwen38_contract::sm120::comm::decode;
    if rows <= decode::LL_MAX_ROWS {
        decode::symbol_v3(ranks).unwrap()
    } else {
        decode::symbol_v2(ranks).unwrap()
    }
}

fn assert_journal(lines: &[String], expected: (usize, u64), name: &str) {
    if let Some(root) = std::env::var_os("CHOPINFER_QWEN38_JOURNAL_DIR") {
        diagnostics::export(
            &std::path::PathBuf::from(root).join(name),
            lines,
            name,
            expected,
            64 << 20,
        )
        .unwrap();
    }
    if let Err(error) = check_journal(
        lines,
        expected,
        &format!("qwen38-launch-journal-{name}.txt"),
    ) {
        panic!("journal {name} differs from {expected:?}: {error}");
    }
}

fn prefill_comm_symbols() -> [&'static str; 2] {
    use model_qwen38_contract::sm120::comm as hc;
    [hc::SCATTER_SYMBOL, hc::GATHER_SYMBOL]
}

/// No retired prefill pair is resolved or launched.
#[test]
fn no_retired_prefill_comm_entry_resolves() {
    let host = Rc::new(FakeHost::default());
    let lines = journal_steps(&host);
    for symbol in prefill_comm_symbols() {
        assert!(
            lines
                .iter()
                .any(|line| line.starts_with("launch ") && launch_symbol(line) == symbol)
        );
    }
    for symbol in [
        "chop_comm_qwen38_reduce_scatter_v1",
        "chop_comm_qwen38_allgather_hc_v1",
    ] {
        assert!(!lines.iter().any(|line| line.contains(symbol)), "{symbol}");
    }
}

#[test]
fn every_geometry_loads_each_encoder_symbol_once() {
    for geometry in model_qwen38_contract::geometry::GEOMETRIES {
        for entry in model_qwen38_contract::encoder::kernels_for(geometry).unwrap() {
            let owners = crate::CUBINS
                .iter()
                .filter(|cubin| cubin.symbols.contains(&entry.symbol.as_str()))
                .filter(|cubin| {
                    crate::GEOMETRY_CUBINS.iter().any(|(name, id, ranks)| {
                        *name == cubin.name && (*id == "*" || *id == geometry.id) && *ranks == 0
                    })
                })
                .count();
            assert_eq!(owners, 1, "{}: {}", geometry.id, entry.symbol);
        }
    }
}

/// Exercise real geometry selection and the create, execute, commit, cancel,
/// retire and destroy lifecycle at every compiled topology.
#[test]
fn every_geometry_topology_runs_the_fake_host_journal_and_lifecycle() {
    for geometry in model_qwen38_contract::geometry::GEOMETRIES {
        for ranks in [Ranks::TP2, Ranks::TP4] {
            let shape = Shape::of_geometry(geometry, ranks).expect("served geometry");
            let host = Rc::new(FakeHost::at(shape));
            let available = crate::GEOMETRY_HC_PRECOMPUTE
                .iter()
                .any(|(id, available)| *id == geometry.id && *available);
            let lines = crate::instance::with_hc_padded_mix(!available, || {
                crate::instance::with_hc_precompute(true, || journal_steps(&host))
            });
            export_rows(shape.topology().id, &[&host]);
            if !available {
                assert!(host.logs.borrow().iter().any(|line| {
                    line.contains("HC precompute: off (buckets 16/32), padded mix: off")
                }));
                assert!(
                    host.symbol_names
                        .borrow()
                        .values()
                        .all(|symbol| !symbol.contains("_precompute_")
                            && !symbol.contains("_mix_precomputed_"))
                );
            }
            assert!(!lines.is_empty(), "{}", shape.topology().id);
            assert_eq!(host.released.get(), host.acquired.get());
            let names = host.symbol_names.borrow();
            for symbol in names.values() {
                assert!(
                    crate::CUBINS.iter().any(|cubin| {
                        cubin.symbols.contains(&symbol.as_str())
                            && crate::GEOMETRY_CUBINS.iter().any(|(name, id, n)| {
                                *name == cubin.name
                                    && (*id == "*" || *id == geometry.id)
                                    && (*n == 0 || *n == shape.ranks())
                            })
                    }),
                    "{}: {symbol}",
                    shape.topology().id
                );
            }
            if let Some(root) = std::env::var_os("CHOPINFER_QWEN38_GEOMETRY_JOURNAL_DIR") {
                let expected = journal_digest(&lines);
                diagnostics::export_geometry(
                    &std::path::PathBuf::from(root).join(shape.topology().id),
                    &lines,
                    shape.topology().id,
                    expected,
                    64 << 20,
                )
                .unwrap();
            }
        }
    }
}

#[test]
fn qwen38_launch_journal_is_unchanged() {
    let host = Rc::new(FakeHost::default());
    let lines = crate::instance::with_hc_precompute(true, || journal_steps(&host));
    export_rows("bf16", &[&host]);
    assert_journal(&lines, EXPECTED, "bf16");
    let host = Rc::new(FakeHost::default());
    let old = crate::instance::with_hc_precompute(false, || journal_steps(&host));
    export_rows("bf16-hc-off", &[&host]);
    assert_journal(&old, EXPECTED_HC_OFF, "bf16-hc-off");
}

/// The FP8 KV journal, re-recorded with `EXPECTED` (last:
/// the 512-row scored cap changes the same 24 buffer-size lines as BF16;
/// then the wide and the decode all-reduce entries, as for `EXPECTED`; then
/// the TP4 row-owned default changes the same 32 lines as BF16).
/// HC module loading adds the same 60 lines as BF16.
/// Then CONC-QWEN-STATE, and the swept schedule rows, and PREFIX-EXACT, and
/// the generated encoder catalog, and the text image identity, as for
/// `EXPECTED`, and the shared stream broadcast, as for `EXPECTED`.
const EXPECTED_FP8: (usize, u64) = (19760, 0x66C88A9BFC10295E);
const EXPECTED_FP8_HC_OFF: (usize, u64) = (19700, 0xF169FEDDA14759E7);

/// The symbol of a journal launch line.
fn launch_symbol(line: &str) -> &str {
    line.split(' ').nth(2).unwrap_or_default()
}

/// Checks that `fp8` holds `bf16`'s lines except launches that switch from
/// a BF16 to the FP8 entry of the same stage or keep their symbol with other
/// pool offsets, and returns the symbols of the changed launches.
fn fp8_changes(bf16: &[String], fp8: &[String]) -> BTreeSet<String> {
    assert_eq!(bf16.len(), fp8.len());
    let mut changed = BTreeSet::new();
    // The one other difference: the prefix page table of a layout whose
    // snapshots take more pages (`prefix::page_table_entries`), once per rank.
    let table = |page_bytes| {
        format!(
            "allocate {} 256",
            crate::prefix::page_table_entries(page_bytes, Shape::TP4) * 8
        )
    };
    for (bf16, fp8) in bf16.iter().zip(fp8) {
        if bf16 == fp8 {
            continue;
        }
        if *bf16 == table(PAGE_BYTES_BF16) && *fp8 == table(PAGE_BYTES_FP8) {
            changed.insert("prefix_pages".to_owned());
            continue;
        }
        assert!(
            bf16.starts_with("launch ") && fp8.starts_with("launch "),
            "{bf16}\n{fp8}"
        );
        let (from, to) = (launch_symbol(bf16), launch_symbol(fp8));
        assert!(
            to == from || to == from.replace("_bf16_", "_fp8_"),
            "{bf16}\n{fp8}"
        );
        changed.insert(to.to_owned());
    }
    changed
}

/// The FP8 KV layout runs the BF16 program with the FP8 KV write and
/// attention entries on a pool of FP8 pages: every other journal line is the
/// BF16 journal's, and the launches that differ are exactly the ones that
/// touch the pool.
#[test]
fn qwen38_fp8_kv_journal_differs_only_in_kv_launches() {
    let bf16 =
        crate::instance::with_hc_precompute(true, || journal_steps(&Rc::new(FakeHost::default())));
    let host = Rc::new(FakeHost::default());
    host.fp8_kv.set(true);
    let fp8 = crate::instance::with_hc_precompute(true, || journal_steps(&host));
    export_rows("fp8", &[&host]);
    assert_eq!(
        fp8_changes(&bf16, &fp8),
        BTreeSet::from(
            [
                "chop_attn_qwen38_qk_norm_rope_kv_fp8_v1",
                "chop_qsa_qwen38_attn_fp8_decode_v1",
                "chop_qsa_qwen38_attn_fp8_decode_merge_v1",
                "chop_qsa_qwen38_attn_fp8_prefill_v1",
                "chop_qsa_qwen38_index_key_write_v1",
                "chop_qsa_qwen38_block_scores_decode_v1",
                "chop_qsa_qwen38_block_scores_prefill_v1",
                "prefix_pages",
            ]
            .map(str::to_owned)
        )
    );
    assert_journal(&fp8, EXPECTED_FP8, "fp8");
    let host = Rc::new(FakeHost::default());
    host.fp8_kv.set(true);
    let old = crate::instance::with_hc_precompute(false, || journal_steps(&host));
    export_rows("fp8-hc-off", &[&host]);
    assert_journal(&old, EXPECTED_FP8_HC_OFF, "fp8-hc-off");
}

/// The full layer a pool launch binds first, from its K (or block-key)
/// plane offset in pages of `format`, if it binds the pool.
fn bound_layer(line: &str, format: &bundle_runtime::kv::KvFormat, shape: Shape) -> Option<u32> {
    let offset = (0..shape.ranks()).find_map(|rank| {
        let pool = format!(
            "raw: {} }}, offset: ",
            (1_u64 << 32) + u64::from(KV_SLOT + rank)
        );
        let at = line.find(&pool)? + pool.len();
        line[at..].split(',').next()?.parse::<u64>().ok()
    })?;
    (0..=shape.layers())
        .filter(|&layer| shape.is_full(layer))
        .find(|&layer| shape.page_layer(format, layer).0 == offset)
}

/// Distinct K and V scales per main full layer.
static DISTINCT_SCALES: [[f32; 2]; FULL_LAYERS as usize] = {
    let mut scales = [[0.0; 2]; FULL_LAYERS as usize];
    let mut layer = 0;
    while layer < FULL_LAYERS as usize {
        scales[layer] = [1.0 / (2 << layer) as f32, 1.0 / (8192 << layer) as f32];
        layer += 1;
    }
    scales
};

/// Checks every FP8 or BF16 KV write and attention launch of `lines`: the
/// main layers use their FP8 entries with their own scales from
/// `DISTINCT_SCALES`, and the MTP layer, if any, its BF16 entries at 1.0.
/// Returns the (symbol, layer) pairs seen.
fn assert_layer_storage(
    lines: &[String],
    format: &bundle_runtime::kv::KvFormat,
) -> BTreeSet<(String, u32)> {
    assert_layer_storage_for(lines, format, Shape::TP4, &DISTINCT_SCALES)
}

fn assert_layer_storage_for(
    lines: &[String],
    format: &bundle_runtime::kv::KvFormat,
    shape: Shape,
    scales: &[[f32; 2]],
) -> BTreeSet<(String, u32)> {
    let mut seen = BTreeSet::new();
    for line in lines {
        let symbol = launch_symbol(line);
        let kv_launch = symbol.starts_with(&crate::launch::per_rank(
            "chop_attn_qwen38_qk_norm_rope_kv_",
            shape,
        )) || (symbol
            .starts_with(&crate::launch::per_rank("chop_qsa_qwen38_attn_", shape))
            && !symbol.contains("merge"));
        if !line.starts_with("launch ") || !kv_launch {
            continue;
        }
        let layer = bound_layer(line, format, shape)
            .unwrap_or_else(|| panic!("{symbol} binds no layer planes"));
        let (k, v) = if layer == shape.layers() {
            assert!(symbol.contains("_bf16_"), "the MTP layer runs {symbol}");
            (1.0_f32, 1.0_f32)
        } else {
            assert!(symbol.contains("_fp8_"), "layer {layer} runs {symbol}");
            let [k, v] = scales[shape.ordinal(layer) as usize];
            (k, v)
        };
        let pair = format!("ScalarU32({}), ScalarU32({})", k.to_bits(), v.to_bits());
        assert!(
            line.contains(&pair),
            "{symbol} layer {layer} lacks its scales"
        );
        seen.insert((symbol.to_owned(), layer));
    }
    seen
}

/// Each main full layer's K and V scales, in that order, reach the KV write
/// and the sparse attention of that layer in prefill and decode.
#[test]
fn fp8_kv_scales_reach_each_full_layers_launches() {
    for geometry in model_qwen38_contract::geometry::GEOMETRIES {
        for ranks in [Ranks::TP2, Ranks::TP4] {
            let shape = Shape::of_geometry(geometry, ranks).unwrap();
            let host = Rc::new(FakeHost::at(shape));
            host.fp8_kv.set(true);
            let mut instance = create(&host);
            let scales: Vec<_> = (0..shape.full_layers())
                .map(|i| [2.0_f32.powi(-1 - i as i32), 2.0_f32.powi(-13 - i as i32)])
                .collect();
            let scales: &'static [[f32; 2]] = Box::leak(scales.into_boxed_slice());
            crate::instance::force_kv_scales(instance.handle(), scales);
            let long = long_prompt();
            run(
                &host,
                &mut instance,
                "prefill",
                step(&[(20, 0, 0, &PROMPT), (21, 1, 0, &long)]),
            );
            instance.commit().unwrap();
            run(
                &host,
                &mut instance,
                "decode",
                step(&[(20, 0, 3, &[4]), (21, 1, 2050, &[5])]),
            );
            instance.commit().unwrap();
            let seen = assert_layer_storage_for(
                &host.journal.borrow(),
                &shape.kv_formats()[2],
                shape,
                scales,
            );
            for symbol in [
                "chop_attn_qwen38_qk_norm_rope_kv_fp8_v1",
                "chop_qsa_qwen38_attn_fp8_decode_v1",
                "chop_qsa_qwen38_attn_fp8_prefill_v1",
            ] {
                let layers = seen
                    .iter()
                    .filter(|(s, _)| s == &crate::launch::per_rank(symbol, shape))
                    .count();
                assert_eq!(layers, shape.full_layers() as usize, "{symbol}");
            }
            assert_eq!(seen.len(), 3 * shape.full_layers() as usize);
            instance.destroy().unwrap();
        }
    }
}

#[test]
fn create_refuses_unknown_and_malformed_geometry_before_acquiring_resources() {
    for topology in [b"qwen38-absent-tp4".as_slice(), b"qwen38-flash-tp02"] {
        let host = Rc::new(FakeHost::default());
        let mut config = configuration_at(host.shape(), host.layout(), &[]);
        let old = host.shape().topology().id;
        let at = config
            .windows(old.len())
            .position(|s| s == old.as_bytes())
            .unwrap();
        // The create wire carries a length-prefixed topology string.
        config[at - 4..at].copy_from_slice(&(topology.len() as u32).to_le_bytes());
        config.splice(at..at + old.len(), topology.iter().copied());
        let result = api().create(HostServicesTableV1::new(Rc::clone(&host)), &config);
        assert!(matches!(
            result,
            Err(BundleAbiError::Status(BundleStatusV1::Unsupported))
        ));
        assert_eq!(host.acquired.get(), 0);
        assert!(host.journal.borrow().is_empty());
    }
}

#[test]
fn create_refuses_tp8_before_acquiring_resources() {
    let host = Rc::new(FakeHost::default());
    let mut roles = vec![RANK_IMAGE_ROLE.to_vec()];
    roles.extend((0..8).map(|rank| index_role(rank).into_bytes()));
    let config = bundle_runtime::config::test_support::encode(
        b"qwen38-flash-tp8",
        &(0..8).collect::<Vec<i32>>(),
        &bundle_runtime::config::KvLayout {
            layout_id: KV_LAYOUT_BF16.to_vec(),
            page_bytes: PAGE_BYTES_BF16,
            alignment: 256,
            page_tokens: PAGE_TOKENS,
            ranks: 8,
            lifetime: 0,
        },
        &bundle_runtime::config::StateLayout {
            state_id: RING_STATE_ID.to_vec(),
            bytes: RING_STATE_SLOT_BYTES * u64::from(SERVED),
            alignment: 256,
        },
        &roles,
        Some(SERVED),
    );
    let created = api().create(HostServicesTableV1::new(Rc::clone(&host)), &config);
    assert_eq!(host.acquired.get(), 0);
    assert!(matches!(
        created,
        Err(BundleAbiError::Status(BundleStatusV1::Unsupported))
    ));
    assert_eq!(host.released.get(), 0);
    assert!(host.journal.borrow().is_empty());
}

/// Each text rank index names the identity of the image the host serves for
/// its rank: a file index names the file's digest, a planned one the plan's
/// identity. Any other pairing is refused before the image is opened.
#[test]
fn create_refuses_a_rank_index_that_names_another_image() {
    let file = serde_json::json!({"sha256": "a".repeat(64)});
    let planned = serde_json::json!({"direct_identity": "b".repeat(64)});
    let cases = [
        (None, file.clone(), true),
        (
            Some(("direct_identity", "b".repeat(64))),
            planned.clone(),
            true,
        ),
        // A file image whose digest differs from the index's.
        (None, serde_json::json!({"sha256": "c".repeat(64)}), false),
        // A file image of another length.
        (
            None,
            serde_json::json!({"byte_length": 4096, "sha256": "a".repeat(64)}),
            false,
        ),
        // A file image with an index that claims a planned identity.
        (
            Some(("direct_identity", "b".repeat(64))),
            file.clone(),
            false,
        ),
        // A planned image whose identity differs from the index's.
        (
            Some(("direct_identity", "d".repeat(64))),
            planned.clone(),
            false,
        ),
        // A planned image with an index that names a file digest.
        (None, planned.clone(), false),
    ];
    for shape in Shape::ALL {
        for (index, served, accepted) in &cases {
            let host = Rc::new(FakeHost::at(shape));
            *host.index_identity.borrow_mut() = index.clone();
            *host.served_identity.borrow_mut() = Some(served.clone());
            let result = api().create(
                HostServicesTableV1::new(Rc::clone(&host)),
                &configuration_at(shape, host.layout(), &[]),
            );
            let case = format!("{} {index:?} {served}", shape.topology().id);
            assert_eq!(result.is_ok(), *accepted, "{case}");
            match result {
                Ok(mut instance) => instance.destroy().unwrap(),
                Err(_) => assert!(
                    host.journal.borrow().iter().all(|call| call == "release"),
                    "{case}: device calls before the identity check: {:?}",
                    host.journal.borrow()
                ),
            }
            assert_eq!(host.released.get(), host.acquired.get(), "{case}");
        }
    }
}

#[test]
fn create_refuses_a_kv_layout_with_another_formats_page() {
    for (layout_id, page_bytes) in [
        (KV_LAYOUT_FP8, PAGE_BYTES_BF16),
        (KV_LAYOUT_BF16, PAGE_BYTES_FP8),
        (KV_LAYOUT_FP8_MTP, PAGE_BYTES_FP8),
        (KV_LAYOUT_FP8_MTP, PAGE_BYTES_BF16_MTP),
        (&b"qwen38.qsa-paged-fp16.v1"[..], PAGE_BYTES_BF16),
    ] {
        let host = Rc::new(FakeHost::default());
        let error = api()
            .create(
                HostServicesTableV1::new(Rc::clone(&host)),
                &configuration_for((layout_id, page_bytes, 32 * RING_STATE_SLOT_BYTES), &[]),
            )
            .map(|_| ())
            .unwrap_err();
        assert!(
            matches!(error, BundleAbiError::Status(BundleStatusV1::Unsupported)),
            "{error:?}"
        );
        assert!(host.journal.borrow().is_empty());
    }
}

/// Runs the journal scenario and returns the host with its record.
fn journal_scenario(ngram_host: bool) -> Rc<FakeHost> {
    let host = Rc::new(FakeHost::default());
    host.ngram_host.set(ngram_host);
    let mut instance = create(&host);
    let long = long_prompt();
    run(
        &host,
        &mut instance,
        "prefill",
        step(&[(20, 0, 0, &PROMPT), (21, 1, 0, &long)]),
    );
    instance.commit().unwrap();
    run(
        &host,
        &mut instance,
        "decode",
        step(&[(20, 0, 3, &[4]), (21, 1, 2050, &[5])]),
    );
    instance.commit().unwrap();
    instance.retire_request(20).unwrap();
    instance.retire_request(21).unwrap();
    instance.destroy().unwrap();
    assert_eq!(host.released.get(), host.acquired.get());
    host
}

#[test]
fn a_host_n_gram_table_binds_the_same_launches_to_the_host_view() {
    for overlap in [false, true] {
        let _arm = overlap_arm(overlap);
        let device = journal_scenario(false);
        let host = journal_scenario(true);
        assert_eq!(*host.launches.borrow(), *device.launches.borrow());
        assert_eq!(host.journal.borrow().len(), device.journal.borrow().len());
        let tables: Vec<_> = (0..RANKS)
            .map(|rank| {
                let index =
                    crate::views::parse(&fake_index(rank, false, Shape::TP4).0, rank).unwrap();
                let view = index.views[&crate::views::ngram_view_for(Shape::TP4)].clone();
                (view.offset, view.bytes, view.bytes.next_multiple_of(4096))
            })
            .collect();
        let table = |rank: u32| tables[rank as usize];
        let names = host.launches.borrow();
        let mut host_reads = 0;
        for ((host_spans, device_spans), (_, name, _)) in host
            .launch_spans
            .borrow()
            .iter()
            .zip(device.launch_spans.borrow().iter())
            .zip(names.iter())
        {
            assert_eq!(host_spans.len(), device_spans.len(), "{name}");
            for (&(slot, offset, length), &expected) in host_spans.iter().zip(device_spans) {
                // The device run's span, from the host run's: the table moved to
                // the host buffer, and arena views after it moved down by the cut.
                let unmapped = if (HOST_TABLE_SLOT..HOST_TABLE_SLOT + RANKS).contains(&slot) {
                    let rank = slot - HOST_TABLE_SLOT;
                    let (table_offset, bytes, _) = table(rank);
                    assert!(name.starts_with("chop_row_qwen38_ngram"), "{name}");
                    assert_eq!((offset, length), (0, bytes));
                    host_reads += 1;
                    (IMAGE_SLOT + rank, table_offset, bytes)
                } else if (IMAGE_SLOT..KV_SLOT).contains(&slot) {
                    let (table_offset, _, cut) = table(slot - IMAGE_SLOT);
                    let offset = if offset >= table_offset {
                        offset + cut
                    } else {
                        offset
                    };
                    (slot, offset, length)
                } else {
                    (slot, offset, length)
                };
                assert_eq!(unmapped, expected, "{name}");
            }
        }
        // Prefill: one n-gram launch per item and rank; decode: one per rank.
        assert_eq!(host_reads, 4 * if overlap { 3 } else { 2 } + 4);
    }
}

#[test]
fn zero_prefix_decode_is_refused_before_device_work() {
    let host = Rc::new(FakeHost::default());
    let mut instance = create(&host);
    let request = BatchRequestV1::new(
        BatchWireVersionV1::V3,
        vec![BatchItemV1::new(7, true, 1, true, vec![42])],
        vec![BatchAppendV1::new(0, 1, 1, vec![0], None)],
    )
    .with_state_slots(vec![0]);
    let before = host.journal.borrow().len();
    let mut sink = ViewSink::default();
    let error = instance
        .execute(&request.encode().unwrap(), &mut sink)
        .unwrap_err();
    assert!(matches!(
        error,
        BundleAbiError::Status(BundleStatusV1::InvalidArgument)
    ));
    assert_eq!(host.journal.borrow().len(), before);
    assert!(sink.0.is_empty());
    run(
        &host,
        &mut instance,
        "fresh prefill after refusal",
        step(&[(7, 0, 0, &[42])]),
    );
    instance.commit().unwrap();
    instance.retire_request(7).unwrap();
}

struct RejectView {
    remaining: usize,
    accepted: usize,
}
impl LogitsRowSinkV1 for RejectView {
    fn limits(&self) -> ResultSinkLimitsV1 {
        ViewSink::default().limits()
    }
    fn emit_row(&mut self, _row: OwnedLogitsRowV1) -> HostServiceResultV1<()> {
        Err(BundleStatusV1::HostError)
    }
    fn emit_view(&mut self, _view: OwnedLogitsViewV1) -> HostServiceResultV1<()> {
        if self.remaining == 0 {
            return Err(BundleStatusV1::HostError);
        }
        self.remaining -= 1;
        self.accepted += 1;
        Ok(())
    }
}

#[test]
fn failed_execute_cleanup_retains_a_cancellable_transaction() {
    for rank in 1..RANKS {
        for accepted in [0, 1] {
            let host = Rc::new(FakeHost::default());
            let mut instance = create(&host);
            run(
                &host,
                &mut instance,
                "initial",
                step(&[(10, 0, 0, &PROMPT), (11, 1, 0, &PROMPT)]),
            );
            instance.commit().unwrap();
            host.fail_restore_rank.set(Some(rank));
            let mut sink = RejectView {
                remaining: accepted,
                accepted: 0,
            };
            // Success with incomplete output is the ABI's retained-transaction
            // protocol: the caller discards the output and cancels.
            instance
                .execute(
                    &step(&[(10, 0, 3, &[7, 8]), (11, 1, 3, &[9])])
                        .encode()
                        .unwrap(),
                    &mut sink,
                )
                .unwrap();
            assert_eq!(sink.accepted, accepted);
            assert_eq!(
                host.fail_restore_rank.get(),
                None,
                "restoration fault was reached"
            );
            // Even a direct ABI caller cannot commit the retained failed step.
            assert_eq!(
                bundle_runtime::entry::commit::<crate::instance::Instance>(instance.handle()),
                BundleStatusV1::InvalidState
            );
            let before = host.copies.borrow().len();
            instance.cancel().unwrap();
            assert_eq!(host.copies.borrow().len() - before, RANKS as usize);
            for request in [10, 11] {
                instance.retire_request(request).unwrap();
            }
            run(
                &host,
                &mut instance,
                "reuse after recovery",
                step(&[(12, 0, 0, &PROMPT)]),
            );
            instance.commit().unwrap();
            instance.retire_request(12).unwrap();
        }
    }
}

#[test]
fn reused_slot_submits_fresh_prefill_inputs_and_output_views() {
    let reused_host = Rc::new(FakeHost::default());
    let mut reused = create(&reused_host);
    run(
        &reused_host,
        &mut reused,
        "history",
        step(&[(10, 5, 0, &PROMPT)]),
    );
    reused.commit().unwrap();
    run(
        &reused_host,
        &mut reused,
        "cancelled token",
        step(&[(10, 5, 3, &[7])]),
    );
    reused.cancel().unwrap();
    assert_eq!(crate::instance::owner(reused.handle(), 5), None);
    reused.retire_request(10).unwrap();
    let fresh_host = Rc::new(FakeHost::default());
    let mut fresh = create(&fresh_host);
    let request = step(&[(11, 5, 0, &PROMPT)]);
    let packed = crate::inputs::PackedStep::pack::<Qwen>(
        crate::contract::MODEL_LIMITS,
        &request,
        KV_PAGES,
        |_| Some(5),
    )
    .unwrap();
    let layout =
        crate::upload::InputLayout::prefill_within::<Qwen>(&packed.prefill, 1, served(SERVED));
    let side = crate::side::SideLayout::new(layout.tokens as u32, SERVED);
    let sizes = [layout.end, side.end];
    let observe = |host: &Rc<FakeHost>, instance: &mut BundleInstanceV1| {
        let uploads = host.uploads.borrow().len();
        let launches = host.launches.borrow().len();
        let mut sink = ViewSink::default();
        instance
            .execute(&request.encode().unwrap(), &mut sink)
            .unwrap();
        let arenas = host.uploads.borrow()[uploads..]
            .iter()
            .filter(|(span, _)| sizes.contains(&span.length))
            .cloned()
            .collect::<Vec<_>>();
        assert_eq!(arenas.len(), 2 * RANKS as usize);
        let views = sink
            .0
            .iter()
            .map(|v| format!("{:?}", v.raw()))
            .collect::<Vec<_>>();
        (arenas, launched(host, launches), views)
    };
    // This host executes no arithmetic: compare the PLE/linear/QSA input
    // contracts and logits views, not numerical state or logits values.
    assert_eq!(
        observe(&reused_host, &mut reused),
        observe(&fresh_host, &mut fresh)
    );
    reused.commit().unwrap();
    fresh.commit().unwrap();
}

mod encoder;

/// The TP2 journals of the journal test's steps, with the n-gram table in
/// host memory as its package places it, re-recorded with
/// `EXPECTED` when the program or a kernel catalog changes (last: target
/// logits rows became F32 through each topology's `head_f32_m{M}` entries).
/// The 512-row scored cap changes only the F32 prefill buffer sizes: their
/// allocations and the spans the prefill heads and logits gathers bind (12
/// lines at TP2). Then the wide and the decode all-reduce
/// entries, as for `EXPECTED`.
/// Row-owned logits then became the default here as at TP4: per rank the
/// exchange flags and epoch are allocated, zeroed and released, and the
/// decode `chop_comm_gather_rows_bf16_v1` launches become
/// `chop_comm_exchange_rows_f32_v1`; owner-stream synchronizations (2 at
/// TP2) and one more owner view in each of the decode and mixed
/// steps follow (16 more lines at TP2).
/// Parallel routing changes TP2 as it changes `EXPECTED`, resolving only the
/// topology's own C32 parallel route.
/// Then CONC-QWEN-STATE, and the swept schedule rows, as for `EXPECTED`.
/// HC precompute runs at every rank count: each rank also loads the HC
/// precompute module, whose entries the steps' decode buckets do not launch
/// (`hc_module_is_the_only_difference`). The `_HC_OFF` journals are the
/// ones recorded before, unchanged.
/// PREFIX-EXACT changes TP2 as it changes `EXPECTED` (24
/// merge launches and 4 resolve and release lines fewer), and so does
/// the F32 vision tower (6 lines fewer). The generated encoder
/// catalog renames symbols only, as for `EXPECTED`. The text image
/// identity adds 2 release lines first, here and in `_HC_OFF`. The shared
/// stream broadcast changes the expansion launches, as for `EXPECTED`.
const EXPECTED_TP2: (usize, u64) = (10414, 0x4CEA7EC8A602A0DC);
const EXPECTED_TP2_HC_OFF: (usize, u64) = (10384, 0x62AE923A04209806);

/// `on` is `off` plus the HC precompute module: each rank resolves and
/// configures its entries and releases them and the module at destroy.
/// Handle numbers shift, so lines compare with their digits masked.
fn hc_module_is_the_only_difference(off: &[String], on: &[String], ranks: u32) {
    let module = crate::CUBINS
        .iter()
        .find(|cubin| cubin.name == sm120::HC_PRECOMPUTE_CUBIN)
        .unwrap();
    // Each run of digits becomes one `#`.
    let mask = |line: &str| {
        let mut masked = String::with_capacity(line.len());
        for c in line.chars() {
            if !c.is_ascii_digit() {
                masked.push(c);
            } else if !masked.ends_with('#') {
                masked.push('#');
            }
        }
        masked
    };
    let mut kept = Vec::new();
    let mut resolved = 0;
    let mut skip_configure = false;
    for line in on {
        if line.starts_with("resolve ")
            && module
                .symbols
                .iter()
                .any(|symbol| line.ends_with(&format!(" {symbol}")))
        {
            resolved += 1;
            skip_configure = true;
            continue;
        }
        if std::mem::take(&mut skip_configure) && line.starts_with("configure ") {
            continue;
        }
        kept.push(line.as_str());
    }
    // The padded mix candidate resolves only under its opt-in.
    let entries = module
        .symbols
        .iter()
        .filter(|symbol| **symbol != sm120::HC_PADDED_MIX || crate::instance::hc::padded_enabled())
        .count();
    assert_eq!(resolved, entries * ranks as usize);
    // Destroy releases each entry and each rank's module.
    let releases = |lines: &[&str]| lines.iter().filter(|line| **line == "release").count();
    let off: Vec<&str> = off.iter().map(String::as_str).collect();
    assert_eq!(releases(&kept), releases(&off) + resolved + ranks as usize);
    let without = |lines: &[&str]| {
        lines
            .iter()
            .filter(|line| **line != "release")
            .map(|line| mask(line))
            .collect::<Vec<_>>()
    };
    let (kept, off) = (without(&kept), without(&off));
    assert_eq!(kept.len(), off.len());
    if let Some(index) = (0..kept.len()).find(|&index| kept[index] != off[index]) {
        panic!("line {index}: {} instead of {}", kept[index], off[index]);
    }
}

#[test]
fn tp2_launch_journal_is_unchanged() {
    let (shape, ngram_host, expected, expected_hc_off) =
        (Shape::TP2, true, EXPECTED_TP2, EXPECTED_TP2_HC_OFF);
    let off = Rc::new(FakeHost::at(shape));
    off.ngram_host.set(ngram_host);
    let off_host = off;
    let off = crate::instance::with_hc_precompute(false, || journal_steps(&off_host));
    export_rows(&format!("{}-hc-off", shape.tag()), &[&off_host]);
    assert_journal(&off, expected_hc_off, &format!("{}-hc-off", shape.tag()));
    let host = Rc::new(FakeHost::at(shape));
    host.ngram_host.set(ngram_host);
    let lines = crate::instance::with_hc_precompute(true, || journal_steps(&host));
    export_rows(shape.tag(), &[&host]);
    hc_module_is_the_only_difference(&off, &lines, shape.ranks());
    assert_eq!(host.released.get(), host.acquired.get());
    // Every rank launches the same program on its own stream, with this
    // topology's per-rank entries.
    let mut per_stream = BTreeMap::<&str, Vec<&str>>::new();
    for line in lines.iter().filter(|line| line.starts_with("launch ")) {
        let stream = line.split(' ').nth(1).unwrap();
        per_stream
            .entry(stream)
            .or_default()
            .push(launch_symbol(line));
    }
    assert_eq!(per_stream.len(), shape.ranks() as usize);
    let programs = per_stream.values().collect::<Vec<_>>();
    assert!(programs.iter().all(|p| p.len() == programs[0].len()));
    let symbols = programs[0].iter().copied().collect::<BTreeSet<_>>();
    for per_rank in [
        "chop_linear_{t}_delta_prefill_scan_v1",
        "chop_qsa_{t}_attn_bf16_prefill_v1",
        "chop_attn_{t}_qk_norm_rope_kv_bf16_v1",
        "chop_gemm_{t}_lin_in_c2_v1",
        "chop_gemm_{t}_head_f32_m2_v1",
        "chop_moe_{t}_decode_route_t2_v1",
        "chop_row_{t}_ngram_decode_v1",
    ] {
        let symbol = per_rank
            .replace("{t}", shape.tag())
            .replace("{n}", &shape.ranks().to_string());
        assert!(symbols.contains(symbol.as_str()), "{symbol}");
    }
    assert!(symbols.contains(decode_comm(shape.ranks(), 2)));
    assert!(symbols.contains("chop_moe_qwen38_router_v1"));
    assert!(
        !lines
            .iter()
            .any(|line| line.contains("chop_moe_qwen38_decode_route_parallel_t32_v1"))
    );
    let parallel = format!("chop_moe_{}_decode_route_parallel_t32_v1", shape.tag());
    assert!(lines.iter().any(|line| line.contains(&parallel)));
    assert_journal(&lines, expected, shape.tag());
}

/// Every placement creates, with and without the MTP drafter, when
/// residency is what the target device's driver reports for each entry's
/// audited resources (`crate::sm120::device::locked`, from
/// `kernels.lock`).
#[test]
fn every_placement_fits_the_target_device() {
    let model = crate::sm120::device::locked();
    for shape in [Shape::TP4, Shape::TP2] {
        for drafter in [false, true] {
            let host = Rc::new(FakeHost::at(shape));
            host.drafter.set(drafter);
            if drafter {
                host.speculation.set((3, MAX_VERIFY_TOKENS));
            }
            *host.modelled.borrow_mut() = Some(model.clone());
            let created = api().create(
                HostServicesTableV1::new(Rc::clone(&host)),
                &configuration_at(shape, host.layout(), &[]),
            );
            let logs = host
                .logs
                .borrow()
                .iter()
                .filter(|line| line.contains("residency") || line.contains("failed"))
                .cloned()
                .collect::<Vec<_>>();
            let mut instance = created
                .unwrap_or_else(|error| panic!("{shape:?} MTP {drafter}: {error:?} {logs:?}"));
            instance.destroy().unwrap();
        }
    }
}

#[test]
fn scored_prompt_capacity_bounds_allocations_and_refuses_before_work() {
    for rows in [512_u32, 513] {
        let host = Rc::new(FakeHost::default());
        let mut instance = create(&host);
        let local = u64::from(SCORED_ROWS) * u64::from(Shape::TP4.vocab_per_rank) * 4;
        assert!(host.extents.borrow().values().any(|&bytes| bytes == local));
        assert!(
            host.extents
                .borrow()
                .values()
                .any(|&bytes| bytes == local * u64::from(RANKS))
        );
        let request = BatchRequestV1::new(
            BatchWireVersionV1::V3,
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
        )
        .with_state_slots(vec![6]);
        let before = host.launches.borrow().len();
        let mut sink = ViewSink::default();
        let result = instance.execute(&request.encode().unwrap(), &mut sink);
        if rows > SCORED_ROWS {
            assert!(result.is_err());
            assert_eq!(host.launches.borrow().len(), before);
            assert!(
                host.logs
                    .borrow()
                    .iter()
                    .any(|line| line.contains("at most 512 prompt rows"))
            );
        } else {
            result.unwrap_or_else(|error| panic!("{error:?} {:?}", host.logs.borrow()));
            assert_eq!(
                sink.0.iter().map(|view| view.raw().row_count).sum::<u32>(),
                rows
            );
            instance.commit().unwrap();
        }
    }
}

#[test]
fn create_refuses_a_scored_row_limit_past_the_bundle_capacity() {
    for (reported, accepted) in [(0, true), (SCORED_ROWS, true), (SCORED_ROWS + 1, false)] {
        let host = Rc::new(FakeHost::default());
        host.scored_rows.set(reported);
        let created = api().create(
            HostServicesTableV1::new(Rc::clone(&host)),
            &configuration_for(host.layout(), &[]),
        );
        assert_eq!(created.is_ok(), accepted, "reported={reported}");
        match created {
            Ok(mut instance) => instance.destroy().unwrap(),
            Err(_) => {
                assert!(
                    host.allocated_buffers
                        .borrow()
                        .iter()
                        .all(|id| !host.extents.borrow().contains_key(id))
                );
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
    const BEFORE: u32 = 3;
    const AFTER: u32 = 4;
    for total in [SCORED_ROWS, SCORED_ROWS + 1] {
        let host = Rc::new(FakeHost::default());
        let mut instance = create(&host);
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
                    PROMPT.to_vec(),
                ));
                appends.push(BatchAppendV1::new(0, 3, 3, vec![page], None));
                page += 1;
            }
        }
        let request = BatchRequestV1::new(BatchWireVersionV1::V3, items, appends)
            .with_state_slots((1..=u64::from(BEFORE + 1 + AFTER)).collect());
        let before = host.launches.borrow().len();
        let mut sink = ViewSink::default();
        let result = instance.execute(&request.encode().unwrap(), &mut sink);
        if total > SCORED_ROWS {
            assert!(result.is_err());
            assert_eq!(host.launches.borrow().len(), before);
            assert!(
                host.logs
                    .borrow()
                    .iter()
                    .any(|line| line.contains("at most 512 prompt rows"))
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
            assert!(
                views
                    .iter()
                    .all(|view| view.element_type == BUNDLE_LOGITS_ELEMENT_V1_F32)
            );
            instance.commit().unwrap();
        }
        instance.destroy().unwrap();
    }
}

#[test]
fn explicit_rope_axes_refuse_out_of_range_before_upload() {
    let host = Rc::new(FakeHost::default());
    let mut instance = create(&host);
    for axis in 0..3 {
        for value in [-1, crate::contract::CONTEXT_TOKENS as i32] {
            let request = step(&[(10, 3, 0, &[7])]);
            let mut axes = [0; 3];
            axes[axis] = value;
            let item = request.items()[0].clone().with_position_axes(vec![axes]);
            let request = BatchRequestV1::new(
                BatchWireVersionV1::V3,
                vec![item],
                request.appends().to_vec(),
            )
            .with_state_slots(request.state_slots().unwrap().to_vec());
            let uploads = host.uploads.borrow().len();
            let journal = host.journal.borrow().len();
            assert_eq!(
                instance.execute(&request.encode().unwrap(), &mut ViewSink::default()),
                Err(BundleAbiError::Status(BundleStatusV1::InvalidArgument))
            );
            assert_eq!(host.uploads.borrow().len(), uploads);
            assert_eq!(host.journal.borrow().len(), journal);
        }
    }
    let request = step(&[(10, 3, 0, &[7])]);
    let item = request.items()[0].clone().with_position_axes(vec![[
        0,
        crate::contract::CONTEXT_TOKENS as i32 - 1,
        0,
    ]]);
    let request = BatchRequestV1::new(
        BatchWireVersionV1::V3,
        vec![item],
        request.appends().to_vec(),
    )
    .with_state_slots(request.state_slots().unwrap().to_vec());
    instance
        .execute(&request.encode().unwrap(), &mut ViewSink::default())
        .unwrap();
}

fn overlap_instance(host: &Rc<FakeHost>) -> BundleInstanceV1 {
    struct Reset(Option<bool>, Option<u32>);
    impl Drop for Reset {
        fn drop(&mut self) {
            crate::instance::FORCE_OVERLAP.with(|value| value.set(self.0));
            crate::instance::FORCE_OVERLAP_MIN.with(|value| value.set(self.1));
        }
    }
    let _reset = Reset(
        crate::instance::FORCE_OVERLAP.with(|value| value.replace(Some(true))),
        crate::instance::FORCE_OVERLAP_MIN.with(|value| value.replace(Some(128))),
    );
    create(host)
}

#[test]
fn overlapped_prefill_keeps_scored_rows_anchors_and_kernel_invariance() {
    use sm120::Launched;
    let mut pins = Vec::new();
    for shape in [Shape::TP2, Shape::TP4] {
        let host = Rc::new(FakeHost::at(shape));
        host.ngram_host.set(ngram_in_host_memory(shape));
        crate::pipeline::take_recorded();
        let mut instance = overlap_instance(&host);
        let tokens = vec![7; 510];
        let request = BatchRequestV1::new(
            BatchWireVersionV1::V3,
            vec![
                BatchItemV1::new(11, false, 510, true, tokens)
                    .with_prompt_rows(true)
                    .with_anchor(true),
                BatchItemV1::new(10, false, 3, true, PROMPT.to_vec()),
            ],
            vec![
                BatchAppendV1::new(0, 510, 510, (0..8).collect(), None),
                BatchAppendV1::new(0, 3, 3, vec![64], None),
            ],
        )
        .with_state_slots(vec![0, 1]);
        let before = host.launches.borrow().len();
        let journal = host.journal.borrow().len();
        crate::instance::PLANNED.with(|planned| planned.borrow_mut().clear());
        crate::sm120::overlap::COMPARED.with(|compared| compared.set([0; 4]));
        let mut sink = ViewSink::default();
        instance
            .execute(&request.encode().unwrap(), &mut sink)
            .unwrap_or_else(|error| panic!("{shape:?}: {error:?} {:?}", host.logs.borrow()));
        assert_eq!(
            sink.0.iter().map(|view| view.raw().row_count).sum::<u32>(),
            511
        );
        // Each pipelined launch's grid moved by its op's facts as by the
        // symbol rule (`sm120::overlap::compare_with_symbols`): persistent
        // grids shrank, and phased pairs ran on the comm lane.
        let [all, shrunk, phased, comm] = crate::sm120::overlap::COMPARED.with(|c| c.get());
        assert!(
            all > 0 && shrunk > 0 && phased > 0 && comm > 0,
            "{shape:?}: {all} {shrunk} {phased} {comm}"
        );
        let planned = crate::instance::PLANNED.with(|planned| planned.take());
        let mut lowered = planned
            .iter()
            .map(|(_, symbol, _)| symbol.clone())
            .collect::<Vec<_>>();
        let mut submitted = host.launches.borrow()[before..]
            .iter()
            .map(|(_, symbol, _)| symbol.clone())
            .collect::<Vec<_>>();
        lowered.sort();
        submitted.sort();
        assert_eq!(lowered, submitted);
        let errors = invariance::PROGRAM.check_journal(
            model_qwen38_contract::sm120::catalogs(),
            planned
                .iter()
                .map(|(_, symbol, _)| (Launched::Prefill, symbol.as_str())),
            &model_qwen38_contract::owned::declared(),
        );
        assert!(errors.is_empty(), "{}", errors.join("\n"));
        assert!(
            planned
                .iter()
                .any(|(_, symbol, _)| symbol == model_qwen38_contract::sm120::owned::PLE_ANCHOR)
        );
        let names = launched(&host, before);
        assert_eq!(
            count(
                &names,
                &crate::launch::per_rank("chop_gemm_qwen38_head_f32_prefill_v1", shape)
            ),
            shape.ranks.count()
        );
        let lines = host.journal.borrow()[journal..].to_vec();
        let mut recorded = BTreeSet::new();
        for line in &lines {
            let words = line.split_whitespace().collect::<Vec<_>>();
            match words.first().copied() {
                Some("record") => {
                    recorded.insert(words[2]);
                }
                Some("wait") => assert!(recorded.contains(words[2])),
                _ => {}
            }
        }
        assert!(!recorded.is_empty());
        let uploaded = |view: DeviceBufferSpanV1, count: usize| {
            let uploads = host.uploads.borrow();
            let (span, bytes) = uploads
                .iter()
                .find(|(span, _)| {
                    span.buffer == view.buffer
                        && span.offset <= view.offset
                        && span.offset + span.length >= view.offset + (4 * count) as u64
                })
                .unwrap();
            let at = (view.offset - span.offset) as usize;
            bytes[at..at + 4 * count]
                .chunks_exact(4)
                .map(|bytes| i32::from_le_bytes(bytes.try_into().unwrap()))
                .collect::<Vec<_>>()
        };
        let launches = host.launches.borrow();
        let arguments = host.launch_arguments.borrow();
        let conv = launches
            .iter()
            .zip(arguments.iter())
            .filter(|((_, symbol, _), _)| symbol.contains("_conv_prefill_"))
            .take(2 * shape.ranks.count())
            .collect::<Vec<_>>();
        assert_eq!(conv.len(), 2 * shape.ranks.count());
        for (index, (_, args)) in conv.iter().enumerate() {
            let KernelArgumentV1::DeviceBuffer(initial) = args[7] else {
                panic!("linear initial argument");
            };
            assert_eq!(
                uploaded(initial, if index < shape.ranks.count() { 1 } else { 2 }),
                if index < shape.ranks.count() {
                    vec![0]
                } else {
                    vec![1, 0]
                }
            );
        }
        drop(arguments);
        drop(launches);
        let histories = host.copies.borrow();
        let histories = histories
            .iter()
            .filter(|(from, to)| {
                from.length == shape.ple_entry_bytes() && to.length == shape.ple_entry_bytes()
            })
            .collect::<Vec<_>>();
        assert_eq!(histories.len(), 2 * shape.ranks.count());
        assert!(histories.iter().all(|(from, to)| from.buffer != to.buffer));
        instance.commit().unwrap();
        instance.destroy().unwrap();
        export_rows(&format!("overlap-{}", shape.tag()), &[&host]);
        // Final row views reuse compute-only intermediates; only comm
        // operands and the final head input have disjoint row backing.
        if let Some(root) = std::env::var_os("CHOPINFER_QWEN38_JOURNAL_DIR") {
            let name = format!("overlap-{}", shape.tag());
            diagnostics::export(
                &std::path::PathBuf::from(root).join(&name),
                &lines,
                &name,
                journal_digest(&lines),
                64 << 20,
            )
            .unwrap();
        }
        // Main's checkpoint side arrays (2026-10-03) lengthen side uploads
        // and move their operand offsets by 1152 bytes, as in the eager
        // journals; every other line is unchanged. Plain decode v1 leaving
        // the shared comm cubin and the Qwen decode v1 entries leaving the
        // decode cubins (2026-10-05) move only handle IDs, and so does the
        // prefill v1 pair leaving comm_qwen38 against the v2 pins. The
        // shared stream broadcast changes the expansion launches, as for
        // `EXPECTED`.
        pins.push((
            journal_digest(&lines),
            match shape.ranks() {
                2 => (8083, 11605953476443707514),
                4 => (16181, 6988794815195098612),
                _ => unreachable!(),
            },
        ));
    }
    assert!(
        pins.iter().all(|(actual, expected)| actual == expected),
        "{pins:?}"
    );
}

#[test]
fn failed_overlap_events_drain_both_lanes_before_restore() {
    for wait in [false, true] {
        let host = Rc::new(FakeHost::default());
        let mut instance = overlap_instance(&host);
        let before = host.journal.borrow().len();
        if wait {
            host.device.fail_wait.set(Some(8));
        } else {
            host.device.fail_record.set(Some(8));
        }
        assert_eq!(
            instance.execute(
                &step(&[(10, 0, 0, &vec![7; 2048])]).encode().unwrap(),
                &mut ViewSink::default()
            ),
            Err(BundleAbiError::Status(BundleStatusV1::HostError))
        );
        assert!(!instance.is_poisoned());
        let lines = host.journal.borrow()[before..].to_vec();
        let last = lines
            .iter()
            .rposition(|line| {
                ["launch ", "record ", "wait "]
                    .iter()
                    .any(|prefix| line.starts_with(prefix))
            })
            .unwrap();
        let drained = lines[last + 1..]
            .iter()
            .take_while(|line| line.starts_with("synchronize "))
            .collect::<Vec<_>>();
        let comm = host
            .launches
            .borrow()
            .iter()
            .filter(|(_, symbol, _)| symbol == model_qwen38_contract::sm120::comm::SCATTER_SYMBOL)
            .map(|(stream, _, _)| format!("synchronize {stream}"))
            .collect::<BTreeSet<_>>();
        assert_eq!(comm.len(), 4);
        assert!(drained[..4].iter().all(|line| comm.contains(line.as_str())));
        assert!(
            drained[4..]
                .iter()
                .all(|line| !comm.contains(line.as_str()))
        );
        assert_eq!(drained.len(), 8);
        assert_eq!(drained.iter().collect::<BTreeSet<_>>().len(), 8);
        run(
            &host,
            &mut instance,
            "retry",
            step(&[(10, 0, 0, &vec![7; 2048])]),
        );
        instance.commit().unwrap();
        instance.destroy().unwrap();
    }
}

#[cfg(feature = "diagnostic-taps")]
#[test]
fn overlap_taps_capture_parts_and_live_state_bytes() {
    const CHILD: &str = "CHOPINFER_TEST_OVERLAP_TAPS_CHILD";
    const NAME: &str = "fake_host_tests::overlap_taps_capture_parts_and_live_state_bytes";
    if let Ok(enabled) = std::env::var(CHILD) {
        let host = Rc::new(FakeHost::default());
        let mut instance = if enabled == "1" {
            overlap_instance(&host)
        } else {
            create(&host)
        };
        run(
            &host,
            &mut instance,
            "tapped",
            BatchRequestV1::new(
                BatchWireVersionV1::V3,
                vec![BatchItemV1::new(10, false, 257, true, vec![7; 257]).with_anchor(true)],
                vec![BatchAppendV1::new(0, 257, 257, (0..5).collect(), None)],
            )
            .with_state_slots(vec![0]),
        );
        instance.commit().unwrap();
        instance.destroy().unwrap();
        return;
    }
    let root = std::env::temp_dir().join(format!("qwen-overlap-taps-{}", std::process::id()));
    for enabled in ["0", "1"] {
        let dir = root.join(enabled);
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args([NAME, "--exact", "--test-threads=1"])
            .env(CHILD, enabled)
            .env("CHOPINFER_QWEN38_PREFILL_OVERLAP", enabled)
            .env("CHOPINFER_TAPS_DIR", &dir)
            .env("CHOPINFER_TAPS_SITES", "embed.out,final.x.hc.mixer")
            .env("CHOPINFER_QWEN38_OVERLAP_EVIDENCE", "1")
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        let step = std::fs::read_dir(&dir)
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .find(|path| path.join("capture.json").is_file())
            .unwrap();
        let completion: serde_json::Value =
            serde_json::from_slice(&std::fs::read(step.join("completion.json")).unwrap()).unwrap();
        assert_eq!(completion["completion"], "full");
        let split: serde_json::Value =
            serde_json::from_slice(&std::fs::read(step.join("overlap.json")).unwrap()).unwrap();
        assert_eq!(split["rows"], 257);
        assert_eq!(
            split["split"],
            if enabled == "1" {
                serde_json::json!(128)
            } else {
                serde_json::Value::Null
            }
        );
        let evidence = step.join("overlap-evidence");
        let rows: serde_json::Value =
            serde_json::from_slice(&std::fs::read(evidence.join("rows.json")).unwrap()).unwrap();
        assert_eq!(rows["tokens"].as_array().unwrap().len(), 257);
        assert_eq!(rows["anchors"], serde_json::json!([256]));
        for rank in 0..RANKS {
            assert_eq!(
                std::fs::metadata(evidence.join(format!("logits.r{rank}.bin")))
                    .unwrap()
                    .len(),
                u64::from(Shape::TP4.vocab_per_rank) * 4
            );
            for name in [
                "kv.i0.p0",
                "ple.i0",
                "conv.l0.i0",
                "delta.l0.i0",
                "ring-key.l0.i0",
                "ring-position.l0.i0",
                "anchor-ple.i0",
                "anchor-conv.l0.i0",
                "anchor-delta.l0.i0",
                "anchor-ring-key.l0.i0",
                "anchor-ring-position.l0.i0",
            ] {
                assert!(
                    std::fs::metadata(evidence.join(format!("{name}.r{rank}.bin")))
                        .unwrap()
                        .len()
                        > 0
                );
            }
        }
    }
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn failed_overlap_drain_poison_stops_before_restore() {
    let host = Rc::new(FakeHost::default());
    let mut instance = overlap_instance(&host);
    host.device.fail_record.set(Some(8));
    host.refuse_synchronize.set(true);
    let before = host.journal.borrow().len();
    assert!(
        instance
            .execute(
                &step(&[(10, 0, 0, &vec![7; 2048])]).encode().unwrap(),
                &mut ViewSink::default()
            )
            .is_err()
    );
    let refused_at = host.journal.borrow().len();
    assert_eq!(
        instance.execute(
            &step(&[(10, 0, 0, &PROMPT)]).encode().unwrap(),
            &mut ViewSink::default()
        ),
        Err(BundleAbiError::Status(BundleStatusV1::InvalidState))
    );
    assert!(instance.destroy().is_err());
    assert_eq!(host.journal.borrow().len(), refused_at);
    let lines = host.journal.borrow()[before..].to_vec();
    let last = lines
        .iter()
        .rposition(|line| line.starts_with("launch "))
        .unwrap();
    let after = &lines[last + 1..];
    assert_eq!(
        after
            .iter()
            .filter(|line| line.starts_with("synchronize "))
            .count(),
        1
    );
    assert!(after.iter().all(|line| {
        !["copy ", "upload ", "launch "]
            .iter()
            .any(|prefix| line.starts_with(prefix))
    }));
}

#[test]
fn refused_overlap_split_runs_the_whole_prefill() {
    let host = Rc::new(FakeHost::default());
    let previous = crate::instance::FORCE_OVERLAP_MIN.with(|value| value.replace(Some(1)));
    let enabled = crate::instance::FORCE_OVERLAP.with(|value| value.replace(Some(true)));
    let mut instance = create(&host);
    crate::instance::FORCE_OVERLAP_MIN.with(|value| value.set(previous));
    crate::instance::FORCE_OVERLAP.with(|value| value.set(enabled));
    let request = BatchRequestV1::new(
        BatchWireVersionV1::V3,
        vec![BatchItemV1::new(10, false, 65, true, vec![7; 65]).with_anchor(true)],
        vec![BatchAppendV1::new(0, 65, 65, vec![0, 1], None)],
    )
    .with_state_slots(vec![0]);
    let before = host.launches.borrow().len();
    instance
        .execute(&request.encode().unwrap(), &mut ViewSink::default())
        .unwrap();
    let names = launched(&host, before);
    assert_eq!(
        count(
            &names,
            &crate::launch::per_rank("chop_gemm_qwen38_head_f32_m1_v1", Shape::TP4)
        ),
        RANKS as usize
    );
    assert_eq!(
        host.copies
            .borrow()
            .iter()
            .filter(|(from, to)| from.length == Shape::TP4.ple_entry_bytes()
                && to.length == from.length
                && from.buffer != to.buffer)
            .count(),
        0
    );
    instance.commit().unwrap();
    instance.destroy().unwrap();
}

#[test]
fn overlapped_continuation_preserves_nonzero_ple_item_offsets() {
    let host = Rc::new(FakeHost::default());
    let mut instance = overlap_instance(&host);
    run(
        &host,
        &mut instance,
        "warm",
        step(&[(20, 1, 0, &vec![7; 192])]),
    );
    instance.commit().unwrap();
    let before = host.launches.borrow().len();
    let copies = host.copies.borrow().len();
    run(
        &host,
        &mut instance,
        "continuing",
        step(&[(10, 0, 0, &PROMPT), (20, 1, 192, &vec![7; 257])]),
    );
    let launches = host.launches.borrow();
    let args = host.launch_arguments.borrow();
    let ngram = launches[before..]
        .iter()
        .zip(&args[before..])
        .filter(|((_, name, _), _)| {
            name == &crate::launch::per_rank(crate::sm120::entries::NGRAM, Shape::TP4)
        })
        .map(|(_, args)| args)
        .collect::<Vec<_>>();
    assert_eq!(ngram.len(), 3 * RANKS as usize);
    let histories = host.copies.borrow();
    let histories = histories[copies..]
        .iter()
        .filter(|(from, to)| {
            from.length == Shape::TP4.ple_entry_bytes()
                && to.length == from.length
                && from.buffer != to.buffer
        })
        .collect::<Vec<_>>();
    assert_eq!(histories.len(), RANKS as usize);
    let pointer = |arg: &KernelArgumentV1| match arg {
        KernelArgumentV1::DeviceBuffer(span) => *span,
        _ => panic!("PLE pointer"),
    };
    for rank in 0..RANKS as usize {
        let first = pointer(&ngram[2 * rank][9]);
        let crossing = pointer(&ngram[2 * rank + 1][9]);
        let continued = pointer(&ngram[2 * RANKS as usize + rank][9]);
        assert_eq!(crossing.offset - first.offset, Shape::TP4.ple_entry_bytes());
        assert_eq!(
            (crossing.buffer, crossing.offset),
            (continued.buffer, continued.offset)
        );
        assert_eq!(
            histories[rank].0.offset + Shape::TP4.ple_conv_bytes(),
            crossing.offset
        );
        assert_eq!(
            histories[rank].1.offset + Shape::TP4.ple_conv_bytes(),
            pointer(&ngram[2 * RANKS as usize + rank][1]).offset
        );
    }
    let conv = launches[before..]
        .iter()
        .zip(&args[before..])
        .filter(|((_, name, _), _)| name.contains("_conv_prefill_"))
        .take(2 * RANKS as usize)
        .collect::<Vec<_>>();
    for (index, (_, args)) in conv.iter().enumerate() {
        let source = pointer(&args[7]);
        let uploads = host.uploads.borrow();
        let (span, bytes) = uploads
            .iter()
            .rev()
            .find(|(span, _)| {
                span.buffer == source.buffer
                    && span.offset <= source.offset
                    && span.offset + span.length >= source.offset + source.length
            })
            .unwrap();
        let start = (source.offset - span.offset) as usize;
        let values = bytes[start..start + 4 * if index < RANKS as usize { 2 } else { 1 }]
            .chunks_exact(4)
            .map(|v| i32::from_le_bytes(v.try_into().unwrap()))
            .collect::<Vec<_>>();
        assert_eq!(
            values,
            if index < RANKS as usize {
                vec![0, 1]
            } else {
                vec![1]
            }
        );
    }
    drop(args);
    drop(launches);
    instance.commit().unwrap();
    instance.destroy().unwrap();
}

fn check_overlap_anchor_boundary(first: usize, second: usize) {
    let host = Rc::new(FakeHost::default());
    let mut instance = overlap_instance(&host);
    let request = BatchRequestV1::new(
        BatchWireVersionV1::V3,
        vec![
            BatchItemV1::new(10, false, first as u32, true, vec![7; first]).with_anchor(true),
            BatchItemV1::new(20, false, second as u32, true, vec![7; second]),
        ],
        vec![
            BatchAppendV1::new(
                0,
                first as u64,
                first as u64,
                (0..(first as u64).div_ceil(64)).collect(),
                None,
            ),
            BatchAppendV1::new(
                0,
                second as u64,
                second as u64,
                (64..64 + (second as u64).div_ceil(64)).collect(),
                None,
            ),
        ],
    )
    .with_state_slots(vec![0, 1]);
    let before = host.launches.borrow().len();
    instance
        .execute(&request.encode().unwrap(), &mut ViewSink::default())
        .unwrap();
    let launches = host.launches.borrow();
    let arguments = host.launch_arguments.borrow();
    let uploads = host.uploads.borrow();
    let read = |argument: &KernelArgumentV1, count: usize| {
        let KernelArgumentV1::DeviceBuffer(source) = argument else {
            panic!("linear side pointer");
        };
        let (span, bytes) = uploads
            .iter()
            .find(|(span, _)| {
                span.buffer == source.buffer
                    && span.offset <= source.offset
                    && span.offset + span.length >= source.offset + count as u64 * 4
            })
            .unwrap();
        let start = (source.offset - span.offset) as usize;
        bytes[start..start + count * 4]
            .chunks_exact(4)
            .map(|v| i32::from_le_bytes(v.try_into().unwrap()))
            .collect::<Vec<_>>()
    };
    let conv = launches[before..]
        .iter()
        .zip(&arguments[before..])
        .filter(|((_, name, _), _)| name.contains("_conv_prefill_"))
        .take(RANKS as usize)
        .collect::<Vec<_>>();
    assert_eq!(conv.len(), RANKS as usize);
    for (_, args) in conv {
        assert_eq!(read(&args[5], 2), vec![0, first as i32]);
        assert!(read(&args[8], 1)[0] >= 0);
        let rows = read(&args[5], 2);
        assert_ne!(
            (rows[1] - rows[0]) % 64,
            0,
            "the launched anchor owner must retain a partial granule"
        );
    }
    let delta = launches[before..]
        .iter()
        .zip(&arguments[before..])
        .filter(|((_, name, _), _)| name.contains("_delta_prefill_scan_"))
        .take(RANKS as usize)
        .collect::<Vec<_>>();
    assert_eq!(delta.len(), RANKS as usize);
    for (_, args) in delta {
        let rows = read(&args[5], 2);
        assert_eq!(rows, vec![0, first as i32]);
        assert!(read(&args[8], 1)[0] >= 0);
        assert_ne!((rows[1] - rows[0]) % 64, 0);
    }
    drop(uploads);
    drop(arguments);
    drop(launches);
    instance.commit().unwrap();
    instance.destroy().unwrap();
}

#[test]
fn overlapped_anchor_boundary_for_1025_and_1023_prompts() {
    check_overlap_anchor_boundary(1025, 1023);
}

#[test]
fn overlapped_anchor_boundary_for_1030_and_1000_prompts() {
    check_overlap_anchor_boundary(1030, 1000);
}

#[test]
fn unaligned_continuation_uses_safe_item_boundaries_and_eager_fallback() {
    let host = Rc::new(FakeHost::default());
    let mut instance = overlap_instance(&host);
    run(
        &host,
        &mut instance,
        "warm",
        step(&[(20, 1, 0, &vec![7; 193])]),
    );
    instance.commit().unwrap();
    let before = host.launches.borrow().len();
    run(
        &host,
        &mut instance,
        "boundary",
        step(&[(10, 0, 0, &PROMPT), (20, 1, 193, &vec![7; 257])]),
    );
    let names = launched(&host, before);
    assert_eq!(
        count(
            &names,
            &crate::launch::per_rank(crate::sm120::entries::NGRAM, Shape::TP4)
        ),
        2 * RANKS as usize
    );
    instance.commit().unwrap();
    let before = host.launches.borrow().len();
    let copies = host.copies.borrow().len();
    run(
        &host,
        &mut instance,
        "whole continuation",
        step(&[(20, 1, 450, &vec![7; 2048])]),
    );
    let names = launched(&host, before);
    assert_eq!(
        count(
            &names,
            &crate::launch::per_rank(crate::sm120::entries::NGRAM, Shape::TP4)
        ),
        RANKS as usize
    );
    assert!(
        !host.copies.borrow()[copies..]
            .iter()
            .any(|(from, to)| from.length == Shape::TP4.ple_entry_bytes()
                && to.length == from.length
                && from.buffer != to.buffer)
    );
    instance.commit().unwrap();
    instance.destroy().unwrap();
}

#[test]
fn every_allocation_failure_stops_and_releases_the_acquired_prefix() {
    use bundle_runtime::resources::{Held, allocate_span, release_all};
    for shape in [Shape::TP2, Shape::TP4] {
        for (verify_rows, drafter) in [
            (0_u32, false),
            (2, false),
            (4, false),
            (8, false),
            (8, true),
        ] {
            for overlap in [false, true] {
                let served = crate::scratch::Served::new(Items::new(33).unwrap());
                let speculation = crate::scratch::Speculation {
                    verify_rows,
                    drafter,
                };
                let comm = crate::instance::target(shape, false).comm_bytes(served);
                let rows = crate::scratch::sizes(
                    &comm,
                    123_456,
                    speculation,
                    shape.page_bytes_bf16_mtp(),
                    shape,
                    served,
                );
                for cutoff in 0..rows.len() {
                    let host = Rc::new(FakeHost::at(shape));
                    host.fail_allocation.set(Some(cutoff));
                    let table = HostServicesTableV1::new(Rc::clone(&host));
                    let services = table.caller();
                    let stream = services
                        .create_stream(BundleContextHandleV1::from_parts(1, 1), 0)
                        .unwrap();
                    let mut held = vec![Held::Stream(stream)];
                    let mut attempted = Vec::new();
                    let result = crate::scratch::RankScratch::allocate_with_overlap(
                        &comm,
                        123_456,
                        speculation,
                        shape.page_bytes_bf16_mtp(),
                        shape,
                        served,
                        overlap,
                        1,
                        |bytes| {
                            attempted.push(bytes);
                            allocate_span(
                                services,
                                BundleMemoryPoolHandleV1::from_parts(1, 1),
                                stream,
                                bytes,
                                &mut held,
                            )
                        },
                    );
                    assert!(result.is_err());
                    let expected = rows[..=cutoff]
                        .iter()
                        .map(|(name, bytes)| {
                            bytes
                                * if overlap
                                    && matches!(
                                        &**name,
                                        "x0" | "x1" | "xn" | "hc_inject" | "u" | "partial"
                                    )
                                {
                                    2
                                } else {
                                    1
                                }
                        })
                        .collect::<Vec<_>>();
                    assert_eq!(attempted, expected);
                    assert_eq!(host.allocation_attempts.get(), cutoff + 1);
                    assert_eq!(host.allocated_buffers.borrow().len(), cutoff);
                    let allocated = host.allocated_buffers.borrow().clone();
                    assert_eq!(held.len(), cutoff + 1);
                    release_all(services, &mut held).unwrap();
                    assert!(held.is_empty());
                    assert_eq!(
                        *host.released_buffers.borrow(),
                        allocated.iter().rev().copied().collect::<Vec<_>>()
                    );
                    assert_eq!(host.released.get(), host.acquired.get());
                    assert!(host.extents.borrow().is_empty());
                }
            }
        }
    }
}

/// Capture both KV layouts and every allocation mode without changing execution defaults.
#[test]
fn mechanics_modes_preserve_commit_cancel_and_ranked_allocations() {
    let _default_overlap = legacy_overlap_off();
    for shape in [Shape::TP2, Shape::TP4] {
        for fp8 in [false, true] {
            for (rows, drafter) in [
                (0_u32, false),
                (2, false),
                (4, false),
                (8, false),
                (8, true),
            ] {
                let host = Rc::new(FakeHost::at(shape));
                host.ngram_host.set(ngram_in_host_memory(shape));
                host.drafter.set(drafter);
                host.fp8_kv.set(fp8);
                host.speculation
                    .set((rows.saturating_sub(1), MAX_VERIFY_TOKENS));
                let mut instance = create(&host);
                host.model(instance.handle());
                let tokens = (0..5).map(speculation::truth).collect::<Vec<_>>();
                speculation::execute(
                    &host,
                    &mut instance,
                    &speculation::batch(vec![speculation::plain(10, 1, 0, &tokens)]),
                );
                instance.commit().unwrap();
                let verify = |previous| {
                    if drafter {
                        speculation::drafted(
                            10,
                            1,
                            previous,
                            speculation::truth(previous),
                            rows - 1,
                        )
                    } else {
                        let tokens = (previous..previous + u64::from(rows))
                            .map(speculation::truth)
                            .collect();
                        (
                            BatchItemV1::verify(10, rows, false, tokens),
                            BatchAppendV1::new(
                                previous,
                                u64::from(rows),
                                previous + u64::from(rows),
                                vec![64],
                                None,
                            ),
                            1,
                        )
                    }
                };
                if rows != 0 {
                    // Decode-first drafter records join a prefill-first ledger by request ID.
                    speculation::execute(
                        &host,
                        &mut instance,
                        &speculation::batch(vec![speculation::plain(20, 2, 0, &tokens), verify(5)]),
                    );
                    let commit_start = host.journal.borrow().len();
                    instance
                        .commit_accepted(&speculation::kept(&[(10, 1)]))
                        .unwrap();
                    let journal = host.journal.borrow();
                    let observed = journal[commit_start..]
                        .iter()
                        .filter_map(|line| {
                            if line.starts_with("launch ") {
                                Some("launch")
                            } else if line.starts_with("upload ") {
                                Some("upload")
                            } else {
                                None
                            }
                        })
                        .collect::<Vec<_>>();
                    let ranks = shape.ranks() as usize;
                    let mut expected = vec!["launch"; ranks];
                    if drafter {
                        for _ in 0..2 {
                            expected.extend(vec!["upload"; ranks]);
                            expected.extend(vec!["launch"; ranks]);
                        }
                    }
                    assert_eq!(
                        observed, expected,
                        "PLE copies precede retained indices and copies"
                    );
                    drop(journal);
                    speculation::execute(
                        &host,
                        &mut instance,
                        &speculation::batch(vec![verify(6)]),
                    );
                } else {
                    speculation::execute(
                        &host,
                        &mut instance,
                        &speculation::batch(vec![
                            speculation::plain(10, 1, 5, &[speculation::truth(5)]),
                            speculation::plain(20, 2, 0, &tokens),
                        ]),
                    );
                    let commit_start = host.journal.borrow().len();
                    instance.commit().unwrap();
                    let journal = host.journal.borrow();
                    let effects = journal[commit_start..]
                        .iter()
                        .filter(|line| line.starts_with("launch ") || line.starts_with("upload "))
                        .collect::<Vec<_>>();
                    assert_eq!(effects.len(), 2 * shape.ranks() as usize);
                    assert!(effects.iter().all(|line| line.starts_with("launch ")
                        && line.contains(model_qwen38_contract::sm120::owned::COPY_ENTRIES)));
                    drop(journal);
                    speculation::execute(
                        &host,
                        &mut instance,
                        &speculation::batch(vec![speculation::plain(
                            10,
                            1,
                            6,
                            &[speculation::truth(6)],
                        )]),
                    );
                }
                let before = host.copies.borrow().len();
                let rings = (0..shape.ranks() as usize)
                    .map(|rank| crate::instance::state_views(instance.handle(), rank)[3].buffer)
                    .collect::<Vec<_>>();
                instance.cancel().unwrap();
                assert!(!crate::instance::pending_step(instance.handle()).unwrap());
                if drafter {
                    let copies = host.copies.borrow();
                    let restores = &copies[before..];
                    assert_eq!(restores.len(), shape.ranks() as usize);
                    assert_eq!(
                        restores.iter().map(|(_, to)| to.buffer).collect::<Vec<_>>(),
                        rings
                    );
                    assert!(restores.iter().all(|(from, to)| from.buffer == to.buffer
                        && from.offset > to.offset
                        && from.length == to.length));
                }
                instance.retire_request(10).unwrap();
                instance.retire_request(20).unwrap();
                instance.destroy().unwrap();
                assert_eq!(host.released.get(), host.acquired.get());
                if let Some(root) = std::env::var_os("CHOPINFER_QWEN38_M0_MODES_DIR") {
                    let name = format!(
                        "tp{}-{}-rows{rows}-drafter{drafter}",
                        shape.ranks(),
                        if fp8 { "fp8" } else { "bf16" }
                    );
                    let root = std::path::PathBuf::from(root).join(&name);
                    let lines = host.journal.borrow();
                    diagnostics::export(&root, &lines, &name, journal_digest(&lines), 64 << 20)
                        .unwrap();
                    let ranks = host.allocation_ranks.borrow();
                    let allocation = host
                        .allocated_buffers
                        .borrow()
                        .iter()
                        .enumerate()
                        .map(|(index, id)| {
                            let (rank, bytes) = ranks[id];
                            format!("{index} rank={rank} handle={id} bytes={bytes}\n")
                        })
                        .collect::<String>();
                    std::fs::write(root.join("allocations.txt"), allocation).unwrap();
                }
            }
        }
    }
}

/// Device failures must happen before ledger apply on both commit entry paths.
#[test]
fn plain_and_drafter_commit_failures_keep_the_ledger_open() {
    let _default_overlap = legacy_overlap_off();
    for drafter in [false, true] {
        let host = Rc::new(FakeHost::default());
        host.drafter.set(drafter);
        if drafter {
            host.speculation.set((3, MAX_VERIFY_TOKENS));
        }
        let mut instance = create(&host);
        host.model(instance.handle());
        let tokens = (0..5).map(speculation::truth).collect::<Vec<_>>();
        speculation::execute(
            &host,
            &mut instance,
            &speculation::batch(vec![speculation::plain(10, 1, 0, &tokens)]),
        );
        instance.commit().unwrap();
        let request = if drafter {
            speculation::drafted(10, 1, 5, speculation::truth(5), 3)
        } else {
            speculation::plain(20, 2, 0, &tokens)
        };
        speculation::execute(&host, &mut instance, &speculation::batch(vec![request]));
        let handle = instance.handle();
        assert_eq!(crate::instance::pending_step(handle), Some(true));
        // Plain commit fails its staged PLE copy. Drafter commit fails the retained
        // history copy after CPU bookkeeping and before checkpoint exchange.
        *host.fail_symbol_launch.borrow_mut() = Some((
            model_qwen38_contract::sm120::owned::COPY_ENTRIES.into(),
            Shape::TP4.ranks() - 1,
        ));
        let result = if drafter {
            instance.commit_accepted(&speculation::kept(&[(10, 2)]))
        } else {
            instance.commit()
        };
        assert_eq!(
            result,
            Err(BundleAbiError::Status(BundleStatusV1::HostError))
        );
        assert!(
            host.fail_symbol_launch.borrow().is_none(),
            "copy fault was reached"
        );
        assert_eq!(
            crate::instance::pending_step(handle),
            Some(true),
            "ledger apply must follow model effects"
        );
        assert!(instance.is_poisoned());
    }
}

#[test]
fn entry_exchange_boundaries_keep_ledger_and_both_maps_ordered() {
    use crate::instance::{
        TestExchangeFault, set_test_exchange_fault, test_entry_state, test_exchange_observation,
    };
    let _default_overlap = legacy_overlap_off();
    for fault in [
        TestExchangeFault::Before,
        TestExchangeFault::After,
        TestExchangeFault::StalePle {
            slot: 0,
            checkpoint: SERVED as i32 + 1,
        },
    ] {
        let host = Rc::new(FakeHost::default());
        host.drafter.set(true);
        host.speculation.set((3, MAX_VERIFY_TOKENS));
        let mut instance = create(&host);
        host.model(instance.handle());
        let tokens = (0..5).map(speculation::truth).collect::<Vec<_>>();
        speculation::execute(
            &host,
            &mut instance,
            &speculation::batch(vec![speculation::plain(10, 1, 0, &tokens)]),
        );
        instance.commit().unwrap();
        speculation::execute(
            &host,
            &mut instance,
            &speculation::batch(vec![speculation::drafted(
                10,
                1,
                5,
                speculation::truth(5),
                3,
            )]),
        );
        let handle = instance.handle();
        let before = test_entry_state(handle);
        assert!(before.ledger_open);
        assert_eq!((before.linear_slots[1], before.ple_slots[1]), (1, 1));
        let mut expected = before.clone();
        match fault {
            TestExchangeFault::Before => {}
            TestExchangeFault::After => {
                expected.linear_slots[1] = SERVED as i32 + 1;
                expected.linear_workspace[1] = 1;
                expected.ple_slots[1] = SERVED as i32 + 1;
                expected.ple_workspace[1] = 1;
            }
            TestExchangeFault::StalePle { .. } => {
                // Invalidate only the prepared PLE exchange; linear remains valid.
                expected.ple_slots[0] = SERVED as i32 + 1;
                expected.ple_workspace[1] = 0;
            }
        }
        set_test_exchange_fault(fault);
        assert_eq!(
            instance.commit_accepted(&speculation::kept(&[(10, 2)])),
            Err(BundleAbiError::Status(BundleStatusV1::HostError))
        );
        assert_eq!(
            test_exchange_observation().unwrap(),
            expected,
            "exchange boundary {fault:?}"
        );
        assert_eq!(
            test_entry_state(handle),
            expected,
            "both maps and ledger after refusal {fault:?}"
        );
        assert!(instance.is_poisoned());
    }
}

/// A named launch failure is consumed before the launch's buffer arguments
/// are checked: a launch that also names a buffer the host never handed out
/// fails with the injected status, records nothing and clears the fault.
#[test]
fn a_named_launch_failure_precedes_the_span_checks() {
    let host = FakeHost::default();
    let context = host.select_context(0, 0).unwrap();
    let stream = host.create_stream(context, 0).unwrap();
    let module = host.load_cuda_module(context, &[]).unwrap();
    let symbol = host.resolve_cuda_symbol(module, b"failing").unwrap();
    *host.fail_symbol_launch.borrow_mut() = Some(("failing".into(), 0));
    let config = LaunchConfigV1 {
        flags: BUNDLE_LAUNCH_FLAG_V1_PROGRAMMATIC_STREAM_SERIALIZATION,
        grid_x: 1,
        grid_y: 1,
        grid_z: 1,
        block_x: 1,
        block_y: 1,
        block_z: 1,
        shared_memory_bytes: 0,
        cluster: LaunchClusterV1::NONE,
    };
    let unknown = DeviceBufferSpanV1::new(BundleDeviceBufferHandleV1::from_parts(1, 900_000), 0, 8);
    let lines = host.journal.borrow().len();
    assert_eq!(
        host.launch_cuda_symbol(
            stream,
            symbol,
            config,
            &[KernelArgumentV1::DeviceBuffer(unknown)]
        ),
        Err(BundleStatusV1::HostError)
    );
    assert!(host.fail_symbol_launch.borrow().is_none());
    assert!(host.launch_spans.borrow().is_empty());
    assert_eq!(host.journal.borrow().len(), lines);
}

/// An injected device allocation failure panics after the host recorded the
/// attempt's buffer, bytes and pool.
#[test]
fn a_failed_device_allocation_keeps_its_attempt_record() {
    let host = FakeHost::default();
    host.fail_allocate.set(Some(0));
    let allocate = std::panic::AssertUnwindSafe(|| {
        host.allocate_device_buffer(
            BundleMemoryPoolHandleV1::from_parts(1, 1),
            BundleStreamHandleV1::from_parts(1, 1),
            8,
            8,
        )
    });
    assert!(std::panic::catch_unwind(allocate).is_err());
    assert_eq!(host.allocation_failures.get(), 1);
    assert_eq!(*host.allocated_buffers.borrow(), [1]);
    assert_eq!(*host.allocation_bytes.borrow(), [8]);
    assert_eq!(*host.allocation_pools.borrow(), [1]);
}
