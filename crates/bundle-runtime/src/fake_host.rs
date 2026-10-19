//! The shared core of a bundle's CPU fake host, for its journal and snapshot
//! tests.
//!
//! A bundle's fake host implements `HostServicesBackendV1` with a
//! [`FakeDevice`] inside it (and derefs to it): the device hands out handle
//! IDs, records the journal of device-facing calls, tracks streams, events,
//! graph captures and the collectives each stream has queued, answers the
//! residency, batch-limit queries, counts the handles the host acquires and
//! releases, keeps the extent of every span it handed out, and holds the
//! fault switches and the snapshot copy record. The bundle's host answers
//! the services that read its package (components, images, views, KV pools,
//! state), checks what its model's launches carry, and keeps its scenarios.
//! [`Backend`] is that host over a model's [`FakePackage`].
//!
//! A journal line names the service and its operands, so a pinned
//! [`journal_digest`] notices any change in what reaches the device:
//! `allocate`, `stream`, `event`, `record`, `wait`, `resolve`, `configure`,
//! `launch`, `zero`,
//! `synchronize`, `capture`, `captured`, `replay`, `release graph` and
//! `release` lines come from here, and [`Backend`] adds `upload`, `copy` and
//! `download` lines; a package adds its own.

mod backend;
pub mod conformance;
pub mod dump;

pub use backend::{Backend, FakePackage, Handle};

use std::cell::{Cell, RefCell};
use std::collections::{BTreeMap, BTreeSet};
use std::marker::PhantomData;

use runtime_bundle_abi::{
    BUNDLE_CLUSTER_RESIDENCY_V1_CLUSTERS_REPORTED, BUNDLE_CLUSTER_RESIDENCY_V1_COOPERATIVE_LAUNCH,
    BUNDLE_LAUNCH_FLAG_V1_COOPERATIVE, BUNDLE_LAUNCH_FLAG_V1_PROGRAMMATIC_STREAM_SERIALIZATION,
    BundleContextHandleV1, BundleCudaModuleHandleV1, BundleCudaSymbolHandleV1,
    BundleDeviceBufferHandleV1, BundleEventHandleV1, BundleGraphHandleV1, BundleMemoryPoolHandleV1,
    BundleRawBatchLimitsV1, BundleRawClusterResidencyV1, BundleStatusV1, BundleStreamHandleV1,
    DeviceBufferSpanV1, HostServiceResultV1, KernelArgumentV1, LaunchConfigV1, LogitsRowSinkV1,
    OwnedDraftTokensV1, OwnedLogitsRowV1, OwnedLogitsViewV1, ResultSinkLimitsV1,
};

use kernel_core::occupancy::{Device, EntryResources};

use crate::index::View;

/// 64-bit FNV-1a: enough to notice a changed upload or journal, and needs
/// no dependency.
pub fn fnv1a(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |hash, &byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

/// The (line count, FNV-1a of the lines joined by newlines) a test pins a
/// journal by.
pub fn journal_digest(lines: &[String]) -> (usize, u64) {
    (lines.len(), fnv1a(lines.join("\n").as_bytes()))
}

/// Checks `lines` against its pinned digest. On a mismatch the journal goes
/// to `file` in the temporary directory, for diffing, and the error names
/// the observed digest and the path.
pub fn check_journal(lines: &[String], expected: (usize, u64), file: &str) -> Result<(), String> {
    let observed = journal_digest(lines);
    if observed == expected {
        return Ok(());
    }
    let path = std::env::temp_dir().join(file);
    std::fs::write(&path, lines.join("\n")).expect("write journal");
    Err(format!(
        "({}, {:#018x}), journal in {}",
        observed.0,
        observed.1,
        path.display()
    ))
}

/// A fault switch: the service fails when its countdown reaches zero, and
/// the switch then turns off.
#[derive(Debug, Default)]
pub struct Countdown(Cell<Option<u32>>);

impl Countdown {
    /// Fails the call after `calls` more succeed; `None` turns it off.
    pub fn set(&self, calls: Option<u32>) {
        self.0.set(calls);
    }

    pub fn get(&self) -> Option<u32> {
        self.0.get()
    }

    /// Counts one call down; true when this call must fail.
    pub fn fails(&self) -> bool {
        match self.0.get() {
            Some(0) => {
                self.0.set(None);
                true
            }
            Some(left) => {
                self.0.set(Some(left - 1));
                false
            }
            None => false,
        }
    }
}

/// One snapshot copy step on a stream, for the snapshot byte dump.
#[derive(Clone, Debug)]
pub enum SnapshotOp {
    Upload(u32, DeviceBufferSpanV1, Vec<u8>),
    Copy(u32, DeviceBufferSpanV1, DeviceBufferSpanV1),
    /// A snapshot copy kernel's arguments.
    Launch(u32, Vec<KernelArgumentV1>),
    Read(u32, DeviceBufferSpanV1),
}

/// Sees each journal line as it is recorded, with its index.
pub trait JournalObserver {
    fn journal(&mut self, line: usize, entry: &str);
}

/// Observes nothing.
#[derive(Default)]
pub struct NoObserver;

impl JournalObserver for NoObserver {
    fn journal(&mut self, _line: usize, _entry: &str) {}
}

/// A device's limits and the audited resources of the entries it runs, by
/// symbol (a bundle's `kernels.lock`): the fake host answers each residency
/// query with what the device's driver would report for the queried entry
/// at the queried launch ([`modelled_report`]).
#[derive(Clone, Debug)]
pub struct DeviceModel {
    pub device: Device,
    pub resources: BTreeMap<String, EntryResources>,
}

/// A residency report of all of `device`'s SMs: `clusters` clusters
/// resident at once, cooperative launch supported, and `flags`.
pub fn device_report(device: &Device, clusters: u32, flags: u32) -> BundleRawClusterResidencyV1 {
    BundleRawClusterResidencyV1 {
        multiprocessor_count: device.multiprocessors,
        usable_multiprocessors: device.multiprocessors,
        max_active_clusters: clusters,
        flags: flags
            | BUNDLE_CLUSTER_RESIDENCY_V1_CLUSTERS_REPORTED
            | BUNDLE_CLUSTER_RESIDENCY_V1_COOPERATIVE_LAUNCH,
        ..BundleRawClusterResidencyV1::unreported()
    }
}

/// The residency report `device`'s driver would give for an entry with
/// `resources` launched at `config`.
pub fn modelled_report(
    device: &Device,
    resources: EntryResources,
    config: LaunchConfigV1,
) -> BundleRawClusterResidencyV1 {
    let threads = config.block_x * config.block_y * config.block_z;
    let per_sm = device.ctas_per_sm(resources, threads, config.shared_memory_bytes);
    device_report(
        device,
        device.resident_clusters(per_sm, config.cluster.x),
        0,
    )
}

/// Whether a symbol is a collective that waits for its peers.
fn collective(symbol: &str) -> bool {
    symbol.starts_with("chop_comm_")
}

/// The device side of a fake host. Fields are public so tests can set
/// switches and read what the bundle did.
#[derive(Default)]
pub struct FakeDevice<O = NoObserver> {
    /// The last handle ID handed out.
    pub next: Cell<u32>,
    /// Every service call that reaches the device, in issue order.
    pub journal: RefCell<Vec<String>>,
    /// Sees each journal line while set.
    pub observer: RefCell<Option<O>>,
    /// Every message the bundle logged, in order.
    pub logs: RefCell<Vec<String>>,
    pub symbol_names: RefCell<BTreeMap<u32, String>>,
    /// Entries explicitly declared to use ordinary stream ordering. Every
    /// other entry retains the fake host's PDL launch requirement.
    pub ordinary_symbols: RefCell<BTreeSet<String>>,
    /// The configured maximum dynamic shared bytes, by symbol slot.
    symbol_dynamic_shared_bytes: RefCell<BTreeMap<u32, u32>>,
    /// The module of each symbol, and the cubin of each loaded module.
    pub symbol_modules: RefCell<BTreeMap<u32, u32>>,
    pub module_cubins: RefCell<BTreeMap<u32, &'static str>>,
    /// Each stream's rank (the context it was created in), and each rank's
    /// first stream.
    pub stream_ranks: RefCell<BTreeMap<u32, u32>>,
    pub rank_streams: RefCell<BTreeMap<u32, u32>>,
    /// Buffer slots in allocation order, and each allocation's rank and
    /// bytes.
    pub buffer_order: RefCell<Vec<u32>>,
    pub allocation_ranks: RefCell<BTreeMap<u32, (u32, u64)>>,
    /// Host handles acquired and released, which a destroyed instance
    /// leaves equal.
    pub acquired: Cell<u32>,
    pub released: Cell<u32>,
    /// Bytes of every span a call may point into, by buffer slot.
    pub extents: RefCell<BTreeMap<u32, u64>>,
    /// (stream, collective) of every launch, eager or captured, for the
    /// captures to count their nodes in.
    launched: RefCell<Vec<(u32, bool)>>,
    /// Open captures by capturing stream: where their launches start.
    pub capture_starts: RefCell<BTreeMap<u32, usize>>,
    /// Events recorded inside a capture, by event: its capturing stream.
    pub event_captures: RefCell<BTreeMap<u32, u32>>,
    /// Streams that joined a capture by waiting on its event, by capturing
    /// stream.
    pub capture_members: RefCell<BTreeMap<u32, BTreeSet<u32>>>,
    /// Launches each captured graph holds.
    pub graph_nodes: RefCell<BTreeMap<u32, usize>>,
    /// Collective launches queued per stream, eagerly or in launched graphs,
    /// and per captured graph. Ranks whose streams differ wait at a
    /// collective their peers never join.
    pub collectives: RefCell<BTreeMap<u32, usize>>,
    pub graph_collectives: RefCell<BTreeMap<u32, BTreeMap<u32, usize>>>,
    /// Residency as a modelled device reports it ([`DeviceModel`]), before
    /// anything below.
    pub modelled: RefCell<Option<DeviceModel>>,
    /// What the cluster residency query reports; unset, the package
    /// answers (`FakePackage::device_residency`). Reports for one symbol
    /// handle by slot, or one
    /// symbol by name, come first.
    pub residency: Cell<Option<BundleRawClusterResidencyV1>>,
    pub residency_by_slot: RefCell<BTreeMap<u32, BundleRawClusterResidencyV1>>,
    pub residency_by_symbol: RefCell<BTreeMap<String, BundleRawClusterResidencyV1>>,
    /// (symbol, grid) of every cluster residency query, in issue order.
    pub residency_queries: RefCell<Vec<(String, u32)>>,
    /// The scored row limit the host reports; zero reports none.
    pub scored_rows: Cell<u32>,
    /// While set, every upload, device copy, readback and snapshot copy
    /// kernel the bundle's host records, in issue order.
    pub snapshot_ops: RefCell<Option<Vec<SnapshotOp>>>,
    /// Fault switches of the services this device answers.
    /// Allocation failure panics before acquiring a buffer. The safe host
    /// trampoline converts it to Panic, preserving its output ABI.
    pub fail_allocate: Countdown,
    /// Failed allocation attempts, whose caller may already have minted an ID.
    pub allocation_failures: Cell<u32>,
    pub fail_record: Countdown,
    pub fail_wait: Countdown,
    pub fail_copy: Countdown,
    pub fail_launch: Countdown,
    pub fail_graph_end: Countdown,
    pub fail_graph_launch: Countdown,
}

impl<O: JournalObserver> FakeDevice<O> {
    /// The next handle ID.
    pub fn id(&self) -> u32 {
        let next = self.next.get() + 1;
        self.next.set(next);
        next
    }

    /// The next handle ID, counted as acquired.
    pub fn acquire(&self) -> u32 {
        self.acquired.set(self.acquired.get() + 1);
        self.id()
    }

    /// Panics unless `span` lies inside a span the host handed out.
    pub fn check_span(&self, span: DeviceBufferSpanV1) {
        let extent = *self
            .extents
            .borrow()
            .get(&span.buffer.slot())
            .unwrap_or_else(|| panic!("launch reads unknown buffer {}", span.buffer.slot()));
        assert!(
            span.offset
                .checked_add(span.length)
                .is_some_and(|end| end <= extent),
            "span {span:?} passes its buffer's {extent} bytes"
        );
    }

    /// Records a journal line.
    pub fn note(&self, entry: String) {
        if let Some(observer) = self.observer.borrow_mut().as_mut() {
            observer.journal(self.journal.borrow().len(), &entry);
        }
        self.journal.borrow_mut().push(entry);
    }

    pub fn log(&self, message: &[u8]) {
        self.logs
            .borrow_mut()
            .push(String::from_utf8_lossy(message).into_owned());
    }

    /// Records `op` while the snapshot record is on.
    pub fn snapshot_op(&self, op: impl FnOnce() -> SnapshotOp) {
        if let Some(ops) = self.snapshot_ops.borrow_mut().as_mut() {
            ops.push(op());
        }
    }

    /// The capture `stream` records into: its own, or the one it joined.
    /// Whether any stream on `stream`'s device (its rank) is capturing, as
    /// the host asks before work a graph must not hold.
    pub fn capturing_on_device_of(&self, stream: u32) -> bool {
        let ranks = self.stream_ranks.borrow();
        let rank = ranks.get(&stream);
        self.capture_starts
            .borrow()
            .keys()
            .any(|capturing| *capturing == stream || ranks.get(capturing) == rank)
    }

    pub fn capture_of(&self, stream: u32) -> Option<u32> {
        if self.capture_starts.borrow().contains_key(&stream) {
            return Some(stream);
        }
        self.capture_members
            .borrow()
            .iter()
            .find(|(_, members)| members.contains(&stream))
            .map(|(&capture, _)| capture)
    }

    /// The buffer `id` of `byte_length` bytes, allocated on `stream`.
    pub fn allocate(
        &self,
        id: u32,
        stream: BundleStreamHandleV1,
        byte_length: u64,
        alignment: u64,
    ) -> BundleDeviceBufferHandleV1 {
        if self.fail_allocate.fails() {
            self.allocation_failures
                .set(self.allocation_failures.get() + 1);
            panic!("injected allocation failure");
        }
        self.note(format!("allocate {byte_length} {alignment}"));
        if let Some(&rank) = self.stream_ranks.borrow().get(&stream.slot()) {
            self.allocation_ranks
                .borrow_mut()
                .insert(id, (rank, byte_length));
        }
        self.buffer_order.borrow_mut().push(id);
        BundleDeviceBufferHandleV1::from_parts(1, id)
    }

    /// The stream `id` in `context`, whose slot is its rank plus one.
    pub fn create_stream(&self, id: u32, context: BundleContextHandleV1) -> BundleStreamHandleV1 {
        self.note(format!("stream {id}"));
        let rank = context.slot() - 1;
        self.stream_ranks.borrow_mut().insert(id, rank);
        self.rank_streams.borrow_mut().entry(rank).or_insert(id);
        BundleStreamHandleV1::from_parts(1, id)
    }

    pub fn create_event(&self, id: u32) -> BundleEventHandleV1 {
        self.note(format!("event {id}"));
        BundleEventHandleV1::from_parts(1, id)
    }

    pub fn record_event(
        &self,
        event: BundleEventHandleV1,
        stream: BundleStreamHandleV1,
    ) -> HostServiceResultV1<()> {
        if self.fail_record.fails() {
            return Err(BundleStatusV1::HostError);
        }
        self.note(format!("record {} {}", stream.slot(), event.slot()));
        // An event recorded inside a capture forks that capture.
        match self.capture_of(stream.slot()) {
            Some(capture) => self
                .event_captures
                .borrow_mut()
                .insert(event.slot(), capture),
            None => self.event_captures.borrow_mut().remove(&event.slot()),
        };
        Ok(())
    }

    pub fn wait_event(
        &self,
        stream: BundleStreamHandleV1,
        event: BundleEventHandleV1,
    ) -> HostServiceResultV1<()> {
        if self.fail_wait.fails() {
            return Err(BundleStatusV1::HostError);
        }
        self.note(format!("wait {} {}", stream.slot(), event.slot()));
        // Waiting on a capture's event joins the stream to it.
        if let Some(&capture) = self.event_captures.borrow().get(&event.slot())
            && capture != stream.slot()
        {
            self.capture_members
                .borrow_mut()
                .entry(capture)
                .or_default()
                .insert(stream.slot());
        }
        Ok(())
    }

    pub fn synchronize_stream(&self, stream: BundleStreamHandleV1) {
        self.note(format!("synchronize {}", stream.slot()));
    }

    pub fn zero(&self, stream: BundleStreamHandleV1, destination: DeviceBufferSpanV1) {
        self.note(format!("zero {} {destination:?}", stream.slot()));
    }

    /// The module `id`, naming its cubin when `module_bytes` is one of
    /// `cubins` (by address).
    pub fn load_module(
        &self,
        id: u32,
        module_bytes: &[u8],
        cubins: impl IntoIterator<Item = (&'static str, &'static [u8])>,
    ) -> BundleCudaModuleHandleV1 {
        if let Some((name, _)) = cubins
            .into_iter()
            .find(|(_, bytes)| std::ptr::eq(*bytes, module_bytes))
        {
            self.module_cubins.borrow_mut().insert(id, name);
        }
        BundleCudaModuleHandleV1::from_parts(1, id)
    }

    /// The symbol `id` of `module`.
    pub fn resolve_symbol(
        &self,
        id: u32,
        module: BundleCudaModuleHandleV1,
        symbol_name: &[u8],
    ) -> HostServiceResultV1<BundleCudaSymbolHandleV1> {
        self.symbol_modules.borrow_mut().insert(id, module.slot());
        let name =
            String::from_utf8(symbol_name.to_vec()).map_err(|_| BundleStatusV1::InvalidArgument)?;
        self.note(format!("resolve {id} {name}"));
        self.symbol_names.borrow_mut().insert(id, name);
        Ok(BundleCudaSymbolHandleV1::from_parts(1, id))
    }

    pub fn configure_symbol(&self, symbol: BundleCudaSymbolHandleV1, dynamic_shared_bytes: u32) {
        self.symbol_dynamic_shared_bytes
            .borrow_mut()
            .insert(symbol.slot(), dynamic_shared_bytes);
        self.note(format!(
            "configure {} {dynamic_shared_bytes}",
            symbol.slot()
        ));
    }

    /// The name `symbol` resolved, or `missing`.
    pub fn symbol_name(
        &self,
        symbol: BundleCudaSymbolHandleV1,
        missing: BundleStatusV1,
    ) -> HostServiceResultV1<String> {
        self.symbol_names
            .borrow()
            .get(&symbol.slot())
            .cloned()
            .ok_or(missing)
    }

    /// Every bundle launch serializes programmatically.
    pub fn check_launch(&self, config: LaunchConfigV1) {
        self.check_launch_for("", config);
    }

    /// Check an explicitly declared ordinary entry or the default PDL form.
    pub fn check_launch_for(&self, name: &str, config: LaunchConfigV1) {
        assert_eq!(
            config.flags,
            if self.ordinary_symbols.borrow().contains(name) {
                0
            } else {
                BUNDLE_LAUNCH_FLAG_V1_PROGRAMMATIC_STREAM_SERIALIZATION
            }
        );
    }

    /// The most launches a stream queued behind its earliest collective
    /// that another stream which queues collectives had not yet queued:
    /// nearly a whole program when the ranks' programs are queued one after
    /// another, none when they are queued collective by collective
    /// (`pipeline::launch_segments`). Each launch counts behind every
    /// collective of its stream still unmatched when it is queued, so a
    /// backlog keeps the launches queued before a peer reached an earlier
    /// collective. For a host that captured no graph.
    pub fn launches_past_unmatched_collectives(&self) -> usize {
        assert!(self.graph_nodes.borrow().is_empty(), "an eager host");
        let launched = self.launched.borrow();
        let mut queued = launched
            .iter()
            .filter(|(_, comm)| *comm)
            .map(|&(stream, _)| (stream, 0))
            .collect::<BTreeMap<u32, usize>>();
        let mut behind = BTreeMap::<(u32, usize), usize>::new();
        for &(stream, comm) in launched.iter() {
            let Some(&own) = queued.get(&stream) else {
                continue;
            };
            // The stream's collectives some peer has not reached.
            let earliest = queued.values().min().copied().unwrap_or(0) + 1;
            for collective in earliest..=own {
                *behind.entry((stream, collective)).or_default() += 1;
            }
            if comm {
                queued.insert(stream, own + 1);
            }
        }
        behind.into_values().max().unwrap_or(0)
    }

    /// Records a launch the device accepted: its journal line, and an eager
    /// collective's place in its stream's queue.
    pub fn launched(
        &self,
        stream: BundleStreamHandleV1,
        name: &str,
        config: LaunchConfigV1,
        arguments: &[KernelArgumentV1],
    ) {
        let comm = collective(name);
        if comm && self.capture_of(stream.slot()).is_none() {
            *self
                .collectives
                .borrow_mut()
                .entry(stream.slot())
                .or_default() += 1;
        }
        self.note(format!(
            "launch {} {name} {config:?} {arguments:?}",
            stream.slot()
        ));
        self.launched.borrow_mut().push((stream.slot(), comm));
    }

    pub fn begin_capture(&self, stream: BundleStreamHandleV1) {
        self.note(format!("capture {}", stream.slot()));
        self.capture_starts
            .borrow_mut()
            .insert(stream.slot(), self.launched.borrow().len());
    }

    /// Closes `stream`'s capture into the graph `id()` names: the capturing
    /// stream's launches and those of the streams that joined it. Returns
    /// the graph and its launch count.
    pub fn end_capture(
        &self,
        stream: BundleStreamHandleV1,
        id: impl FnOnce() -> u32,
    ) -> HostServiceResultV1<(BundleGraphHandleV1, usize)> {
        let start = self
            .capture_starts
            .borrow_mut()
            .remove(&stream.slot())
            .expect("capture began");
        if self.fail_graph_end.fails() {
            return Err(BundleStatusV1::HostError);
        }
        let id = id();
        let members = self
            .capture_members
            .borrow_mut()
            .remove(&stream.slot())
            .unwrap_or_default();
        self.event_captures
            .borrow_mut()
            .retain(|_, capture| *capture != stream.slot());
        let captured = self.launched.borrow()[start..]
            .iter()
            .filter(|(slot, _)| *slot == stream.slot() || members.contains(slot))
            .copied()
            .collect::<Vec<_>>();
        let nodes = captured.len();
        self.graph_nodes.borrow_mut().insert(id, nodes);
        let mut collectives = BTreeMap::<u32, usize>::new();
        for (slot, comm) in captured {
            if comm {
                *collectives.entry(slot).or_default() += 1;
            }
        }
        self.graph_collectives.borrow_mut().insert(id, collectives);
        self.note(format!("captured {} {id} {nodes}", stream.slot()));
        Ok((BundleGraphHandleV1::from_parts(1, id), nodes))
    }

    /// Replays `graph` on `stream`: each stream it captured work of queues
    /// its collectives. Returns its launch count.
    pub fn launch_graph(
        &self,
        stream: BundleStreamHandleV1,
        graph: BundleGraphHandleV1,
    ) -> HostServiceResultV1<usize> {
        let nodes = *self
            .graph_nodes
            .borrow()
            .get(&graph.slot())
            .expect("a captured graph");
        if self.fail_graph_launch.fails() {
            return Err(BundleStatusV1::HostError);
        }
        for (&slot, &count) in &self.graph_collectives.borrow()[&graph.slot()] {
            *self.collectives.borrow_mut().entry(slot).or_default() += count;
        }
        self.note(format!("replay {} {}", stream.slot(), graph.slot()));
        Ok(nodes)
    }

    pub fn release_graph(&self, graph: BundleGraphHandleV1) {
        self.note(format!("release graph {}", graph.slot()));
        self.graph_nodes.borrow_mut().remove(&graph.slot());
        self.graph_collectives.borrow_mut().remove(&graph.slot());
    }

    /// The residency report for `symbol` at `config`, which must be the
    /// descriptor its launch submits; `None` when nothing here answers it.
    pub fn query_cluster_residency(
        &self,
        symbol: BundleCudaSymbolHandleV1,
        config: LaunchConfigV1,
    ) -> HostServiceResultV1<Option<BundleRawClusterResidencyV1>> {
        let name = self.symbol_name(symbol, BundleStatusV1::StaleHandle)?;
        self.residency_queries
            .borrow_mut()
            .push((name.clone(), config.grid_x));
        assert_eq!(
            (config.grid_y, config.grid_z),
            (1, 1),
            "{name}: residency asked for another grid than it launches"
        );
        assert_eq!(
            config.flags & BUNDLE_LAUNCH_FLAG_V1_COOPERATIVE,
            0,
            "{name}: residency asked with another cooperative flag than it launches"
        );
        if let Some(model) = &*self.modelled.borrow() {
            let entry = model
                .resources
                .get(&name)
                .unwrap_or_else(|| panic!("{name} has no locked resources"));
            return Ok(Some(modelled_report(&model.device, *entry, config)));
        }
        let by_slot = self.residency_by_slot.borrow().get(&symbol.slot()).copied();
        let by_symbol = self.residency_by_symbol.borrow().get(&name).copied();
        Ok(by_slot.or(by_symbol).or(self.residency.get()))
    }

    pub fn query_batch_limits(&self) -> HostServiceResultV1<BundleRawBatchLimitsV1> {
        let mut limits = BundleRawBatchLimitsV1::unreported();
        limits.maximum_scored_rows = self.scored_rows.get();
        Ok(limits)
    }

    /// The memory pool of `context`, whose slot is the context's.
    pub fn memory_pool(&self, context: BundleContextHandleV1) -> BundleMemoryPoolHandleV1 {
        BundleMemoryPoolHandleV1::from_parts(1, context.slot())
    }
}

/// A synthetic rank index in the JSON form `index::parse` reads, its views
/// in the order they were added.
pub struct FakeIndex {
    pub schema: String,
    pub rank: u32,
    pub image: String,
    pub sha256: String,
    pub views: Vec<(String, View)>,
}

impl FakeIndex {
    /// An index of `rank` without views, of the image `rank<rank>.img`
    /// with a placeholder digest.
    pub fn new(schema: &str, rank: u32) -> Self {
        Self {
            schema: schema.into(),
            rank,
            image: format!("rank{rank}.img"),
            sha256: "a".repeat(64),
            views: Vec::new(),
        }
    }

    /// Where the next view starts: the end of the last one rounded up to
    /// `alignment`.
    pub fn end(&self, alignment: u64) -> u64 {
        self.views.last().map_or(0, |(_, view)| {
            view.offset + view.bytes.next_multiple_of(alignment)
        })
    }

    /// Adds a view at [`Self::end`], returning its offset.
    pub fn push(
        &mut self,
        name: impl Into<String>,
        bytes: u64,
        dtype: &str,
        shape: Vec<u64>,
        layout_id: &str,
        alignment: u64,
    ) -> u64 {
        let offset = self.end(alignment);
        self.views.push((
            name.into(),
            View {
                offset,
                bytes,
                dtype: dtype.into(),
                shape,
                layout_id: layout_id.into(),
            },
        ));
        offset
    }

    /// The index's bytes for an image of `image_bytes`.
    pub fn bytes(&self, image_bytes: u64) -> Vec<u8> {
        let views = self
            .views
            .iter()
            .map(|(name, view)| {
                serde_json::json!({"name": name, "offset": view.offset, "bytes": view.bytes,
                    "dtype": view.dtype, "shape": view.shape, "layout_id": view.layout_id})
            })
            .collect::<Vec<_>>();
        serde_json::to_vec(
            &serde_json::json!({"schema": self.schema, "rank": self.rank,
            "image": self.image, "image_bytes": image_bytes, "sha256": self.sha256,
            "views": views}),
        )
        .expect("synthetic rank index")
    }
}

/// The limits a [`RecordingSink`] reports: a bundle's rows and row widths.
pub trait SinkLimits {
    fn limits() -> ResultSinkLimitsV1;
}

/// A result sink that records the draft tokens and logits views a step
/// emits, refusing rows. Drafts must precede every view.
pub struct RecordingSink<L> {
    pub drafts: Vec<OwnedDraftTokensV1>,
    pub views: Vec<OwnedLogitsViewV1>,
    limits: PhantomData<L>,
}

impl<L> Default for RecordingSink<L> {
    fn default() -> Self {
        Self {
            drafts: Vec::new(),
            views: Vec::new(),
            limits: PhantomData,
        }
    }
}

impl<L> RecordingSink<L> {
    /// Each draft as (request, tokens).
    pub fn draft_tokens(&self) -> Vec<(u64, Vec<u32>)> {
        self.drafts
            .iter()
            .map(|drafts| (drafts.request_id, drafts.tokens.clone()))
            .collect()
    }
}

impl<L: SinkLimits> LogitsRowSinkV1 for RecordingSink<L> {
    fn limits(&self) -> ResultSinkLimitsV1 {
        L::limits()
    }

    fn emit_row(&mut self, _row: OwnedLogitsRowV1) -> HostServiceResultV1<()> {
        Err(BundleStatusV1::Unsupported)
    }

    fn emit_view(&mut self, view: OwnedLogitsViewV1) -> HostServiceResultV1<()> {
        self.views.push(view);
        Ok(())
    }

    fn emit_draft_tokens(&mut self, drafts: OwnedDraftTokensV1) -> HostServiceResultV1<()> {
        assert!(
            self.views.is_empty(),
            "drafts must precede every logits view"
        );
        self.drafts.push(drafts);
        Ok(())
    }
}

/// Implements host services a fake host does not offer: each reports
/// `Unsupported`. With no arguments it implements those no bundle's fake
/// host offers (physical KV, KV requests, collectives, stage streams, event
/// synchronization, pointer tables and custom operations).
#[macro_export]
macro_rules! unused_host_services {
    ($($name:ident($($argument:ident: $type:ty),* $(,)?) -> $result:ty;)+) => {
        $(
            fn $name(&self, $($argument: $type),*) -> ::runtime_bundle_abi::HostServiceResultV1<$result> {
                Err(::runtime_bundle_abi::BundleStatusV1::Unsupported)
            }
        )+
    };
    () => {
        $crate::unused_host_services! {
            allocate_physical_kv(_request: &::runtime_bundle_abi::PhysicalKvRequestV1)
                -> ::runtime_bundle_abi::BundlePhysicalKvHandleV1;
            release_physical_kv(_kv: ::runtime_bundle_abi::BundlePhysicalKvHandleV1) -> ();
            create_kv_request(
                _kv: ::runtime_bundle_abi::BundlePhysicalKvHandleV1,
                _quota_tokens: u64,
            ) -> ::runtime_bundle_abi::BundleKvRequestHandleV1;
            prepare_kv(
                _request: ::runtime_bundle_abi::BundleKvRequestHandleV1,
                _operation: &[u8],
            ) -> ::runtime_bundle_abi::BundleKvTransactionHandleV1;
            commit_kv(_transaction: ::runtime_bundle_abi::BundleKvTransactionHandleV1) -> ();
            cancel_kv(_transaction: ::runtime_bundle_abi::BundleKvTransactionHandleV1) -> ();
            resolve_kv_view(
                _request: ::runtime_bundle_abi::BundleKvRequestHandleV1,
                _rank: u32,
                _view_id: &[u8],
            ) -> ::runtime_bundle_abi::DeviceBufferSpanV1;
            release_kv_request(_request: ::runtime_bundle_abi::BundleKvRequestHandleV1) -> ();
            create_collective(_request: &::runtime_bundle_abi::CollectiveRequestV1)
                -> ::runtime_bundle_abi::BundleCollectiveHandleV1;
            execute_collective(
                _collective: ::runtime_bundle_abi::BundleCollectiveHandleV1,
                _stream: ::runtime_bundle_abi::BundleStreamHandleV1,
                _operation: &[u8],
                _arguments: &[::runtime_bundle_abi::KernelArgumentV1],
            ) -> ();
            release_collective(_collective: ::runtime_bundle_abi::BundleCollectiveHandleV1) -> ();
            select_stage_stream(
                _context: ::runtime_bundle_abi::BundleContextHandleV1,
                _rank: u32,
                _stage_id: &[u8],
                _stream_kind: u32,
            ) -> ::runtime_bundle_abi::BundleStreamHandleV1;
            synchronize_event(_event: ::runtime_bundle_abi::BundleEventHandleV1) -> ();
            write_device_pointer_table(
                _stream: ::runtime_bundle_abi::BundleStreamHandleV1,
                _destination: ::runtime_bundle_abi::DeviceBufferSpanV1,
                _slots: &[Option<::runtime_bundle_abi::DeviceBufferSpanV1>],
            ) -> ();
            run_custom_operation(
                _context: ::runtime_bundle_abi::BundleContextHandleV1,
                _stream: ::runtime_bundle_abi::BundleStreamHandleV1,
                _completion: ::runtime_bundle_abi::CustomCompletionV1,
                _operation_id: &[u8],
                _operands: &[::runtime_bundle_abi::CustomOperandV1],
                _operation: &::runtime_bundle_abi::CustomOperationV1<'_>,
            ) -> i32;
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ordinary_launches_require_an_exact_symbol_declaration() {
        let device = FakeDevice::<NoObserver>::default();
        let mut config = LaunchConfigV1 {
            flags: 0,
            grid_x: 1,
            grid_y: 1,
            grid_z: 1,
            block_x: 32,
            block_y: 1,
            block_z: 1,
            shared_memory_bytes: 0,
            cluster: runtime_bundle_abi::LaunchClusterV1::NONE,
        };
        let rejects = |name, config| {
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                device.check_launch_for(name, config)
            }))
            .is_err()
        };
        assert!(rejects("local_sum", config));
        device
            .ordinary_symbols
            .borrow_mut()
            .insert("local_sum".into());
        device.check_launch_for("local_sum", config);
        assert!(rejects("another_sum", config));
        config.flags = BUNDLE_LAUNCH_FLAG_V1_PROGRAMMATIC_STREAM_SERIALIZATION;
        assert!(rejects("local_sum", config));
        device.check_launch(config);
    }

    /// A residency query answers from the modelled device first, then the
    /// fixed reports, and leaves one nothing answers to the package.
    #[test]
    fn residency_answers_from_the_modelled_device_then_the_fixed_reports() {
        let device = FakeDevice::<NoObserver>::default();
        device
            .symbol_names
            .borrow_mut()
            .insert(1, "entry_v1".into());
        let symbol = BundleCudaSymbolHandleV1::from_parts(1, 1);
        let config = LaunchConfigV1 {
            flags: 0,
            grid_x: 8,
            grid_y: 1,
            grid_z: 1,
            block_x: 128,
            block_y: 1,
            block_z: 1,
            shared_memory_bytes: 0,
            cluster: runtime_bundle_abi::LaunchClusterV1::NONE,
        };
        assert_eq!(device.query_cluster_residency(symbol, config), Ok(None));
        let target = Device {
            multiprocessors: 4,
            registers_per_sm: 8192,
            register_unit: 256,
            shared_bytes_per_sm: 16_384,
            shared_reserved_per_cta: 1024,
            threads_per_sm: 512,
            ctas_per_sm: 3,
            clusters_of_8_per_slot: 1,
        };
        device.residency.set(Some(device_report(&target, 5, 0)));
        assert_eq!(
            device.query_cluster_residency(symbol, config),
            Ok(Some(device_report(&target, 5, 0)))
        );
        let resources = EntryResources {
            registers: 32,
            shared_static_bytes: 0,
        };
        *device.modelled.borrow_mut() = Some(DeviceModel {
            device: target,
            resources: BTreeMap::from([("entry_v1".to_owned(), resources)]),
        });
        // Two CTAs of 128 threads per SM, by registers: 16 clusters of one.
        let modelled = device.query_cluster_residency(symbol, config).unwrap();
        assert_eq!(modelled, Some(modelled_report(&target, resources, config)));
        assert_eq!(modelled.unwrap().max_active_clusters, 16);
        assert_eq!(modelled.unwrap().multiprocessor_count, 4);
        assert_eq!(device.residency_queries.borrow().len(), 3);
    }

    #[test]
    fn allocation_fault_acquires_nothing_and_fires_once() {
        let device = FakeDevice::<NoObserver>::default();
        let stream = device.create_stream(device.id(), BundleContextHandleV1::from_parts(1, 1));
        device.fail_allocate.set(Some(1));
        let first = device.allocate(device.id(), stream, 64, 16);
        let journal = device.journal.borrow().clone();
        let failed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            device.allocate(device.id(), stream, 128, 16)
        }));
        assert!(failed.is_err());
        assert_eq!(*device.journal.borrow(), journal);
        assert_eq!(*device.buffer_order.borrow(), vec![first.slot()]);
        assert_eq!(device.allocation_ranks.borrow().len(), 1);
        assert_eq!(device.allocation_failures.get(), 1);
        assert_eq!(device.fail_allocate.get(), None);
        let third = device.allocate(device.id(), stream, 256, 16);
        assert_eq!(
            *device.buffer_order.borrow(),
            vec![first.slot(), third.slot()]
        );
        assert_eq!(device.allocation_ranks.borrow()[&third.slot()], (0, 256));
    }

    #[test]
    #[should_panic(expected = "a captured graph")]
    fn a_released_graph_cannot_replay() {
        let device = FakeDevice::<NoObserver>::default();
        let stream = device.create_stream(device.id(), BundleContextHandleV1::from_parts(1, 1));
        device.begin_capture(stream);
        let (graph, _) = device.end_capture(stream, || device.id()).unwrap();
        device.release_graph(graph);
        let _ = device.launch_graph(stream, graph);
    }

    #[test]
    fn countdowns_fail_once_after_their_calls() {
        let countdown = Countdown::default();
        assert!(!countdown.fails());
        countdown.set(Some(2));
        assert_eq!(
            [(); 4].map(|()| countdown.fails()),
            [false, false, true, false]
        );
        assert_eq!(countdown.get(), None);
    }

    /// Launches behind a stream's unmatched collectives count until every
    /// peer queues them, later collectives of the stream included, and stay
    /// counted behind a later collective when a peer reaches an earlier
    /// one; once the peers catch up nothing is behind.
    #[test]
    fn launches_behind_the_earliest_unmatched_collective_are_counted() {
        let device = FakeDevice::<NoObserver>::default();
        let context = |rank: u32| BundleContextHandleV1::from_parts(1, rank + 1);
        let (first, second) = (
            device.create_stream(device.id(), context(0)),
            device.create_stream(device.id(), context(1)),
        );
        let config = LaunchConfigV1 {
            flags: BUNDLE_LAUNCH_FLAG_V1_PROGRAMMATIC_STREAM_SERIALIZATION,
            grid_x: 1,
            grid_y: 1,
            grid_z: 1,
            block_x: 32,
            block_y: 1,
            block_z: 1,
            shared_memory_bytes: 0,
            cluster: runtime_bundle_abi::LaunchClusterV1 { x: 1, y: 1, z: 1 },
        };
        let launch = |stream, comm: bool| {
            let name = if comm {
                "chop_comm_sum_v1"
            } else {
                "chop_row_v1"
            };
            device.launched(stream, name, config, &[]);
        };
        // C1, local, C2, local, local before the peer queues either.
        for comm in [true, false, true, false, false] {
            launch(first, comm);
        }
        for comm in [true, true] {
            launch(second, comm);
        }
        assert_eq!(device.launches_past_unmatched_collectives(), 4);
        // C1, C2 and a local launch; the peer reaches C1; two more local
        // launches; the peer reaches C2. All three wait behind C2.
        let device = FakeDevice::<NoObserver>::default();
        let (first, second) = (
            device.create_stream(device.id(), context(0)),
            device.create_stream(device.id(), context(1)),
        );
        for (stream, comm) in [
            (first, true),
            (first, true),
            (first, false),
            (second, true),
            (first, false),
            (first, false),
            (second, true),
        ] {
            device.launched(
                stream,
                if comm {
                    "chop_comm_sum_v1"
                } else {
                    "chop_row_v1"
                },
                config,
                &[],
            );
        }
        assert_eq!(device.launches_past_unmatched_collectives(), 3);
        // Queued collective by collective, nothing waits behind one.
        let device = FakeDevice::<NoObserver>::default();
        let (first, second) = (
            device.create_stream(device.id(), context(0)),
            device.create_stream(device.id(), context(1)),
        );
        for (stream, comm) in [
            (first, false),
            (first, true),
            (second, false),
            (second, true),
            (first, false),
            (first, true),
            (second, true),
            (first, false),
        ] {
            device.launched(
                stream,
                if comm {
                    "chop_comm_sum_v1"
                } else {
                    "chop_row_v1"
                },
                config,
                &[],
            );
        }
        assert_eq!(device.launches_past_unmatched_collectives(), 0);
    }

    #[test]
    fn graphs_count_the_launches_and_collectives_of_their_joined_streams() {
        let device = FakeDevice::<NoObserver>::default();
        let context = |rank: u32| BundleContextHandleV1::from_parts(1, rank + 1);
        let (lead, member, other) = (
            device.create_stream(device.id(), context(0)),
            device.create_stream(device.id(), context(1)),
            device.create_stream(device.id(), context(2)),
        );
        let event = device.create_event(device.id());
        let config = LaunchConfigV1 {
            flags: BUNDLE_LAUNCH_FLAG_V1_PROGRAMMATIC_STREAM_SERIALIZATION,
            grid_x: 1,
            grid_y: 1,
            grid_z: 1,
            block_x: 32,
            block_y: 1,
            block_z: 1,
            shared_memory_bytes: 0,
            cluster: runtime_bundle_abi::LaunchClusterV1 { x: 1, y: 1, z: 1 },
        };
        device.begin_capture(lead);
        device.launched(lead, "chop_comm_sum_v1", config, &[]);
        device.record_event(event, lead).unwrap();
        device.wait_event(member, event).unwrap();
        device.launched(member, "chop_row_v1", config, &[]);
        device.launched(other, "chop_comm_sum_v1", config, &[]);
        let (graph, nodes) = device.end_capture(lead, || device.id()).unwrap();
        assert_eq!(nodes, 2);
        assert_eq!(device.collectives.borrow()[&other.slot()], 1);
        assert!(!device.collectives.borrow().contains_key(&lead.slot()));
        assert_eq!(device.launch_graph(member, graph), Ok(2));
        assert_eq!(device.collectives.borrow()[&lead.slot()], 1);
        device.release_graph(graph);
        assert!(device.graph_nodes.borrow().is_empty());
        assert!(device.graph_collectives.borrow().is_empty());
        let (lines, _) = journal_digest(&device.journal.borrow());
        assert_eq!(lines, 13);
        assert_eq!(device.journal.borrow()[10], "captured 1 5 2");
    }
}
