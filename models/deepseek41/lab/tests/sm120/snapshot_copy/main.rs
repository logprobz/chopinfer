//! The shared snapshot copy kernel (`prefix_snapshot_copy_kernel`) against
//! the device copies it replaces, over DeepSeek V4.1 Flash's prefix
//! snapshot tables. Both arms plan from the bundle's table
//! (`bundle_runtime::snapshot`): the copies arm issues `Table::actions`'
//! copies as the served host does, one `cuMemcpyDtoDAsync` each, and the
//! kernel arm `Table::kernel_actions`' table upload and launch, through the
//! backend op the bundle registers and the cubin it embeds and locks. The
//! upload goes through a `PinnedUploadRing`, stream-ordered with no host
//! wait, as the served host's `copy_host_to_device` service does
//! (`copy_host_to_device_ordered`); the staged and fenced upload
//! `DeviceBuffer::copy_from_bytes_at` makes, which allocates page-locked
//! staging per call and waits for the stream, is timed beside it. The
//! launch resolves its ABI arguments as the host does: each span is the
//! address of its first byte.
//!
//! The spans lie inside a rank's allocations as a served rank's do: the
//! state's declared buffers one after another in one allocation, the
//! anchor spares in another, the pool span and the table (at an eight-byte
//! boundary that is no sixteen-byte one) at offsets of theirs. In the
//! packed layout they start near the start of their allocations over a
//! 512-page pool; in the high layout the state's first view crosses 4 GiB
//! of its allocation, the spares' crosses 2 GiB, and the pool runs past
//! 4 GiB, its snapshot pages straddling both boundaries.
//!
//! The per-rank state declaration does not depend on the rank count, so
//! the cases cover each KV storage, DSpark on and off and anchor spares on
//! and off; a slot's table at the first and last of the decode batch's
//! slots and an anchor item's at the first and last anchor; and the
//! snapshot's pages ascending, scattered over the pool and descending.
//! Slots capture and restore; anchors only capture, as served.
//!
//! [`bitwise_gpu`] compares every byte of every destination allocation, and
//! the sources after, with the destinations poisoned (twice in the packed
//! layout), across both arms and the CPU oracle
//! (`kernel_core::snapshot_copy::snapshot_copy`).
//! [`pair_gpu`] times both arms with CUDA events, host wall time and host
//! issue time in a pair lab with A/A controls
//! (docs/contributing.md#pair-lab-kernel-timing), then splits the kernel
//! arm: each upload alone, the launch alone over a resident table, the
//! device time of back-to-back launches, and one contiguous device copy of
//! the snapshot's bytes as a bandwidth reference.
//! [`inventory`] plans every case on the CPU.

use bundle_deepseek41::sm120::{fold_cubin_for_symbol, snapshot_copy as bundle_kernel};
use bundle_runtime::geometry::Items;
use bundle_runtime::snapshot::{
    Action, BoundLaunch, Descriptor, HeaderAction, HeldBuffers, KernelCopy, Pages, Table,
    launched_copies,
};
use bundle_runtime::state::{Extent, Holder, RequestState};
use kernel_core::backend::{Backend, Need, Prepared, View};
use kernel_core::snapshot_copy::{Direction, SnapshotCopy, SnapshotCopyBuffers};
use kernel_harness::{LabResult, Session};
use kernels_sm120::backend::Sm120;
use kernels_sm120::snapshot_copy::SYMBOL;
use model_deepseek41_contract::geometry::FLASH;
use model_deepseek41_contract::serving::DECODE_BATCH;
use model_deepseek41_contract::state::{KvStorage, request_state_with_anchors};
use runtime_bundle_abi::{BundleDeviceBufferHandleV1, DeviceBufferSpanV1, KernelArgumentV1};
use runtime_cuda::{
    DeviceBuffer, Function, KernelArgument, PinnedHostBuffer, PinnedUploadRing, Stream,
};
use serde_json::{Value, json};
use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};
use std::time::Instant;

/// Anchor items an anchored instance declares at TP4.
const ANCHORS: u32 = 4;
/// Pages of the packed layout's pool.
const PACKED_POOL_PAGES: u64 = 512;
/// The destinations' fills: a write of either value shows under the other.
const POISONS: [u8; 2] = [0xa5, 0x00];
/// The bitwise sweep stops here, with no result, so its job stays in its
/// bound.
const BUDGET_SECONDS: u64 = 240;
/// The pair lab's bound: its job times one GPU after the other.
const PAIR_BUDGET_SECONDS: u64 = 120;
/// The served host's upload ring per stream (runtime-host's
/// `PINNED_UPLOAD_RING_BYTES`).
const RING_BYTES: usize = 4 << 20;
/// Back-to-back launches whose events time the kernel on the device.
const LAUNCHES: u32 = 32;
/// The lab's allocations, by buffer handle: a rank's state, its anchor
/// spares, its pool and its table buffer.
const STATE: u64 = 1;
const SPARES: u64 = 2;
const POOL: u64 = 3;
const TABLE: u64 = 4;
/// Where the pool span starts in its allocation.
const POOL_OFFSET: u64 = 4096;
/// Where the table starts in its allocation: an eight-byte boundary, the
/// table's alignment, that is no sixteen-byte one.
const TABLE_OFFSET: u64 = 4104;
/// The alignment of the spans packed one after another, a host pool's.
const SPAN_ALIGNMENT: u64 = 256;
const GIB: u64 = 1 << 30;
/// How far below 4 GiB (state) and 2 GiB (spares) the high layout's first
/// span starts.
const BELOW: u64 = 64 << 10;
/// Bytes a verification reads back at once.
const CHUNK_BYTES: u64 = 64 << 20;

/// One per-rank state declaration.
#[derive(Clone, Copy, Debug)]
struct Declared {
    storage: KvStorage,
    dspark: bool,
    anchors: u32,
}

const DECLARED: [Declared; 6] = [
    Declared {
        storage: KvStorage::Native,
        dspark: false,
        anchors: 0,
    },
    Declared {
        storage: KvStorage::Native,
        dspark: false,
        anchors: ANCHORS,
    },
    Declared {
        storage: KvStorage::Native,
        dspark: true,
        anchors: 0,
    },
    Declared {
        storage: KvStorage::Native,
        dspark: true,
        anchors: ANCHORS,
    },
    Declared {
        storage: KvStorage::Bf16,
        dspark: false,
        anchors: 0,
    },
    Declared {
        storage: KvStorage::Bf16,
        dspark: false,
        anchors: ANCHORS,
    },
];

/// Where a bench's spans lie in their allocations.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Layout {
    /// From the start of each allocation, over a 512-page pool.
    Packed,
    /// The state's spans from 64 KiB below 4 GiB, the spares' from 64 KiB
    /// below 2 GiB, over a pool past 4 GiB.
    High,
}

#[derive(Clone, Copy, Debug)]
enum Order {
    Ascending,
    Scattered,
    Descending,
    /// The pages that straddle 2 GiB and 4 GiB of the pool and their
    /// neighbours, the pool's last page, then scattered pages.
    Straddling,
}

const ORDERS: [Order; 3] = [Order::Ascending, Order::Scattered, Order::Descending];

/// The first `count` distinct pages of `ids`, or none when they run short.
fn distinct(ids: impl Iterator<Item = u64>, count: u64, pool_pages: u64) -> LabResult<Vec<u64>> {
    let mut seen = BTreeSet::new();
    let ids = ids
        .take(4 * pool_pages as usize + 16)
        .filter(|&id| id < pool_pages && seen.insert(id))
        .take(count as usize)
        .collect::<Vec<_>>();
    if ids.len() as u64 != count {
        return Err(format!(
            "{pool_pages} pool pages give no {count} distinct pages"
        ));
    }
    Ok(ids)
}

/// `count` distinct pages of a pool of `pool_pages` in `order`.
fn page_ids(order: Order, count: u64, pool_pages: u64, page_bytes: u64) -> LabResult<Vec<u64>> {
    let scattered = (0..).map(move |i: u64| (i * 211 + 17) % pool_pages);
    match order {
        Order::Ascending => distinct((0..count).map(|i| 3 + i), count, pool_pages),
        Order::Scattered => distinct(scattered, count, pool_pages),
        Order::Descending => distinct(
            (0..count).map(|i| pool_pages.saturating_sub(1 + 2 * i)),
            count,
            pool_pages,
        ),
        Order::Straddling => {
            let two = 2 * GIB / page_bytes;
            let four = 4 * GIB / page_bytes;
            let first = [
                four,
                two,
                four - 1,
                two + 1,
                four + 1,
                two - 1,
                pool_pages - 1,
            ];
            distinct(first.into_iter().chain(scattered), count, pool_pages)
        }
    }
}

fn handle(raw: u64) -> BundleDeviceBufferHandleV1 {
    BundleDeviceBufferHandleV1 { raw }
}

fn refused(error: impl std::fmt::Debug) -> String {
    format!("{error:?}")
}

fn align(bytes: u64) -> u64 {
    bytes.div_ceil(SPAN_ALIGNMENT) * SPAN_ALIGNMENT
}

/// One declaration's state in an instance serving the decode batch, laid
/// out in `layout`: its held buffers and spares, the pool and the table
/// buffer, each allocation's size, and the kernel ops prepared as the
/// bundle prepares them: both directions over the snapshot's pages and
/// views, for the most pieces a slot's or an anchor item's table has.
struct Bench {
    declared: Declared,
    layout: Layout,
    state: RequestState,
    extent: Extent,
    held: HeldBuffers,
    pool: DeviceBufferSpanV1,
    pool_pages: u64,
    table: DeviceBufferSpanV1,
    /// Each allocation's buffer handle and bytes.
    arenas: Vec<(u64, u64)>,
    capture: (SnapshotCopy, Prepared<SnapshotCopyBuffers<View>>),
    restore: (SnapshotCopy, Prepared<SnapshotCopyBuffers<View>>),
    /// The synthetic bench's own table, in place of a holder's.
    fixed: Option<Table>,
}

/// The kernel ops of a snapshot of `pages` over `views` for `capacity`
/// pieces, prepared through the backend the bundle registers.
fn ops(
    views: u32,
    page_bytes: u64,
    pages: u64,
    capacity: u32,
) -> LabResult<[(SnapshotCopy, Prepared<SnapshotCopyBuffers<View>>); 2]> {
    let backend = bundle_kernel::register(Sm120::new());
    let prepare = |direction| -> LabResult<_> {
        let op = SnapshotCopy {
            direction,
            views,
            page_bytes,
            pages: pages as u32,
        };
        let prepared = backend
            .snapshot_copy(&op, Need::rows(capacity))
            .map_err(|refusal| refusal.to_string())?;
        Ok((op, prepared))
    };
    Ok([prepare(Direction::ToPages)?, prepare(Direction::FromPages)?])
}

/// Every bench: each declaration packed and high, then the synthetic one.
fn benches() -> LabResult<Vec<Bench>> {
    let mut benches = Vec::new();
    for layout in [Layout::Packed, Layout::High] {
        for declared in DECLARED {
            benches.push(Bench::new(declared, layout)?);
        }
    }
    benches.push(Bench::synthetic()?);
    Ok(benches)
}

impl Bench {
    fn new(declared: Declared, layout: Layout) -> LabResult<Self> {
        let state =
            request_state_with_anchors(&FLASH, declared.storage, declared.dspark, declared.anchors);
        let items = Items::new(DECODE_BATCH).ok_or("the decode batch is no item count")?;
        let extent = Extent {
            served: items.get(),
            bucket: items.capacity(),
            // Five drafts and the bonus row.
            verify_rows: if declared.dspark { 6 } else { 0 },
        };
        let (state_base, spare_base) = match layout {
            Layout::Packed => (0, 0),
            Layout::High => (4 * GIB - BELOW, 2 * GIB - BELOW),
        };
        let mut at = state_base;
        let buffers = state
            .buffers
            .iter()
            .map(|buffer| {
                let span = DeviceBufferSpanV1::new(handle(STATE), at, buffer.bytes(extent));
                at = align(at + span.length);
                Some(span)
            })
            .collect();
        let state_bytes = at;
        let mut at = spare_base;
        let spares = state
            .buffers
            .iter()
            .map(|buffer| {
                (buffer.spare_bytes() > 0).then(|| {
                    let span = DeviceBufferSpanV1::new(handle(SPARES), at, buffer.spare_bytes());
                    at = align(at + span.length);
                    span
                })
            })
            .collect::<Vec<_>>();
        let spare_bytes = at;
        let held = HeldBuffers {
            buffers,
            spares: spares.clone(),
        };
        let mut tables = vec![Table::new(&state, extent, Holder::Slot(0)).map_err(refused)?];
        if declared.anchors > 0 {
            tables.push(Table::new(&state, extent, Holder::Anchor(0)).map_err(refused)?);
        }
        let views = tables[0].segments(&held).map_err(refused)?.0.len() as u32;
        let capacity = tables
            .iter()
            .map(|table| table.descriptors.len())
            .max()
            .unwrap() as u32;
        let page_bytes = tables[0].page_bytes;
        let [capture, restore] = ops(views, page_bytes, tables[0].pages(), capacity)?;
        let table_bytes = capture.0.table_bytes(capacity).map_err(refused)?;
        let pool_pages = match layout {
            Layout::Packed => PACKED_POOL_PAGES,
            Layout::High => 4 * GIB / page_bytes + 64,
        };
        let pool = DeviceBufferSpanV1::new(handle(POOL), POOL_OFFSET, pool_pages * page_bytes);
        let table = DeviceBufferSpanV1::new(handle(TABLE), TABLE_OFFSET, table_bytes);
        let mut arenas = vec![
            (STATE, state_bytes),
            (POOL, POOL_OFFSET + pool.length),
            (TABLE, TABLE_OFFSET + table_bytes),
        ];
        if spares.iter().any(Option::is_some) {
            arenas.push((SPARES, spare_bytes));
        }
        Ok(Self {
            declared,
            layout,
            state,
            extent,
            held,
            pool,
            pool_pages,
            table,
            arenas,
            capture,
            restore,
            fixed: None,
        })
    }

    /// A bench of its own table, whose pieces the declarations never make:
    /// views at offsets off a 16-byte boundary but one, pieces starting,
    /// ending and lying at odd bytes, byte tails and pieces across pages,
    /// so the kernel moves most bytes one at a time. Packed, over 64 pages
    /// of 1,040 bytes.
    fn synthetic() -> LabResult<Self> {
        let declared = DECLARED[0];
        let state =
            request_state_with_anchors(&FLASH, declared.storage, declared.dspark, declared.anchors);
        let items = Items::new(DECODE_BATCH).ok_or("the decode batch is no item count")?;
        let extent = Extent {
            served: items.get(),
            bucket: items.capacity(),
            verify_rows: 0,
        };
        let spans = [(3, 4096), (8192, 8192), (16384 + 11, 4096)]
            .map(|(offset, length)| DeviceBufferSpanV1::new(handle(STATE), offset, length));
        let held = HeldBuffers {
            buffers: spans.map(Some).to_vec(),
            spares: vec![None; spans.len()],
        };
        let mut at = 0;
        let descriptors = [
            (0, 3, 301),
            (1, 17, 255),
            (2, 1, 600),
            (0, 400, 413),
            (1, 1024, 4096),
            (2, 700, 150),
            (0, 2048, 15),
            (1, 5120, 2080),
            (2, 1000, 2051),
        ]
        .map(|(buffer, offset, length)| {
            let descriptor = Descriptor {
                buffer,
                spare: false,
                offset,
                length,
                at,
            };
            at += length;
            descriptor
        })
        .to_vec();
        let table = Table {
            descriptors,
            header_bytes: 0,
            page_bytes: 1040,
        };
        let capacity = table.descriptors.len() as u32;
        let [capture, restore] = ops(3, table.page_bytes, table.pages(), capacity)?;
        let table_bytes = capture.0.table_bytes(capacity).map_err(refused)?;
        let pool_pages = 64;
        let pool =
            DeviceBufferSpanV1::new(handle(POOL), POOL_OFFSET, pool_pages * table.page_bytes);
        let state_bytes = align(spans[2].offset + spans[2].length);
        Ok(Self {
            declared,
            layout: Layout::Packed,
            state,
            extent,
            held,
            pool,
            pool_pages,
            table: DeviceBufferSpanV1::new(handle(TABLE), TABLE_OFFSET, table_bytes),
            arenas: vec![
                (STATE, state_bytes),
                (POOL, POOL_OFFSET + pool.length),
                (TABLE, TABLE_OFFSET + table_bytes),
            ],
            capture,
            restore,
            fixed: Some(table),
        })
    }

    /// The held buffers' and spares' spans.
    fn held_spans(&self) -> Vec<DeviceBufferSpanV1> {
        let held = self.held.buffers.iter().chain(&self.held.spares).flatten();
        held.copied().collect()
    }

    /// The allocations a capture (the pool) or a restore (the state and
    /// spares) writes.
    fn destinations(&self, to_pages: bool) -> Vec<u64> {
        self.arenas
            .iter()
            .map(|&(raw, _)| raw)
            .filter(|&raw| {
                if to_pages {
                    raw == POOL
                } else {
                    raw == STATE || raw == SPARES
                }
            })
            .collect()
    }

    /// The holders of its snapshots: the decode batch's first and last
    /// slots and its first and last anchor items; in the high layout the
    /// last of each.
    fn holders(&self) -> Vec<Holder> {
        if self.fixed.is_some() {
            return vec![Holder::Slot(0)];
        }
        let last = Holder::Slot(self.extent.served - 1);
        let mut holders = match self.layout {
            Layout::Packed => vec![Holder::Slot(0), last],
            Layout::High => vec![last],
        };
        if self.declared.anchors > 0 {
            let anchor = Holder::Anchor(self.declared.anchors - 1);
            match self.layout {
                Layout::Packed => holders.extend([Holder::Anchor(0), anchor]),
                Layout::High => holders.push(anchor),
            }
        }
        holders
    }

    /// The page orders its cases take.
    fn orders(&self) -> &'static [Order] {
        if self.fixed.is_some() {
            return &[Order::Scattered, Order::Descending];
        }
        match self.layout {
            Layout::Packed => &ORDERS,
            Layout::High => &[Order::Straddling],
        }
    }

    fn name(&self) -> String {
        if self.fixed.is_some() {
            return "synthetic".into();
        }
        let d = self.declared;
        format!(
            "{}-{}-a{}",
            match d.storage {
                KvStorage::Native => "native",
                KvStorage::Bf16 => "bf16",
            },
            if d.dspark { "dspark" } else { "ar" },
            d.anchors
        )
    }

    fn page_span(&self, id: u64, page_bytes: u64) -> DeviceBufferSpanV1 {
        DeviceBufferSpanV1::new(
            self.pool.buffer,
            self.pool.offset + id * page_bytes,
            page_bytes,
        )
    }
}

/// One case's two plans: the device copies as (source, destination), and
/// the kernel's table upload and launch.
struct Plans {
    table: Table,
    ids: Vec<u64>,
    copies: Vec<(DeviceBufferSpanV1, DeviceBufferSpanV1)>,
    upload: (DeviceBufferSpanV1, Vec<u8>),
    launch: BoundLaunch,
}

fn plans(bench: &Bench, holder: Holder, order: Order, to_pages: bool) -> LabResult<Plans> {
    let table = match &bench.fixed {
        Some(table) => table.clone(),
        None => Table::new(&bench.state, bench.extent, holder).map_err(refused)?,
    };
    let ids = page_ids(order, table.pages(), bench.pool_pages, table.page_bytes)?;
    let paged = Pages {
        pool: bench.pool,
        ids: &ids,
        page_bytes: table.page_bytes,
    };
    paged.check(&table, bench.pool_pages).map_err(refused)?;
    let copies = table
        .actions::<()>(&bench.held, &paged, to_pages, HeaderAction::None)
        .map_err(refused)?
        .into_iter()
        .map(|action| match action {
            Action::Copy { from, to } => Ok((from, to)),
            _ => Err("the device copies plan only copies".to_owned()),
        })
        .collect::<LabResult<Vec<_>>>()?;
    let (op, prepared) = if to_pages {
        &bench.capture
    } else {
        &bench.restore
    };
    let kernel = KernelCopy {
        op: *op,
        prepared: prepared.as_ref(),
        table: bench.table,
    };
    let actions = table
        .kernel_actions(&bench.held, &paged, to_pages, HeaderAction::None, &kernel)
        .map_err(refused)?;
    let Ok([Action::Upload { to, bytes }, Action::Launch(launch)]) = <[_; 2]>::try_from(actions)
    else {
        return Err("the kernel plan is one table upload and one launch".into());
    };
    if launch.launch.symbol != SYMBOL {
        return Err(format!("the kernel plan launches {}", launch.launch.symbol));
    }
    Ok(Plans {
        table,
        ids,
        copies,
        upload: (to, bytes),
        launch,
    })
}

/// The device side of one bench: an allocation per arena, by buffer
/// handle, the kernel's function, the stream's upload ring and, for the
/// pair lab, a scratch buffer the bandwidth reference copies into.
struct Device<'a> {
    session: &'a Session,
    buffers: BTreeMap<u64, DeviceBuffer>,
    function: Function,
    ring: RefCell<PinnedUploadRing>,
    scratch: Option<DeviceBuffer>,
}

/// What one timed or compared run does on the stream.
#[derive(Clone, Copy, Debug)]
enum Arm {
    /// The device copies.
    Copies,
    /// The served kernel path: the ring upload, then the launch.
    Kernel,
    /// The fenced upload, then the launch: the first pair lab's kernel arm.
    KernelFenced,
    /// The launch alone, over the table the last upload left.
    Launch,
    UploadRing,
    UploadFenced,
    /// Back-to-back launches over a resident table.
    Launches(u32),
    /// One device copy of the snapshot's byte count from the pool.
    Reference,
}

impl Arm {
    fn name(self) -> String {
        match self {
            Arm::Copies => "copies".into(),
            Arm::Kernel => "kernel".into(),
            Arm::KernelFenced => "kernel_fenced".into(),
            Arm::Launch => "launch_resident".into(),
            Arm::UploadRing => "upload_ring".into(),
            Arm::UploadFenced => "upload_fenced".into(),
            Arm::Launches(count) => format!("launches_{count}"),
            Arm::Reference => "reference".into(),
        }
    }
}

impl<'a> Device<'a> {
    fn new(session: &'a Session, bench: &Bench, module: &runtime_cuda::Module) -> LabResult<Self> {
        let mut buffers = BTreeMap::new();
        for &(raw, bytes) in &bench.arenas {
            buffers.insert(raw, session.allocate(bytes as usize)?);
        }
        let function = module.function(SYMBOL).map_err(|e| e.to_string())?;
        let staging =
            PinnedHostBuffer::new(&session.context, RING_BYTES).map_err(|e| e.to_string())?;
        Ok(Self {
            session,
            buffers,
            function,
            ring: RefCell::new(PinnedUploadRing::new(staging)),
            scratch: None,
        })
    }

    fn arena(&self, raw: u64) -> LabResult<&DeviceBuffer> {
        self.buffers
            .get(&raw)
            .ok_or_else(|| format!("no allocation {raw}"))
    }

    fn buffer(&self, span: DeviceBufferSpanV1) -> LabResult<&DeviceBuffer> {
        self.arena(span.buffer.raw)
    }

    /// The device address of `span`'s first byte, as the host resolves it.
    fn address(&self, span: DeviceBufferSpanV1) -> LabResult<u64> {
        self.buffer(span)?
            .device_address_at(span.offset as usize, span.length as usize)
            .map_err(|e| e.to_string())
    }

    /// The device copies, as the host issues them.
    fn copies(&self, plans: &Plans, stream: &Stream) -> LabResult<()> {
        for &(from, to) in &plans.copies {
            self.buffer(to)?
                .copy_range_from_async(
                    to.offset as usize,
                    self.buffer(from)?,
                    from.offset as usize,
                    to.length as usize,
                    stream,
                )
                .map_err(|e| e.to_string())?;
        }
        Ok(())
    }

    /// The table upload as the served host makes it: through the stream's
    /// pinned ring, ordered after the stream's work, with no host wait.
    fn upload_ring(&self, plans: &Plans, stream: &Stream) -> LabResult<()> {
        let (to, bytes) = &plans.upload;
        self.ring
            .borrow_mut()
            .upload(self.buffer(*to)?, to.offset as usize, bytes, stream)
            .map_err(|e| e.to_string())
    }

    /// The table upload staged in page-locked memory allocated for the
    /// call, waited for on the host.
    fn upload_fenced(&self, plans: &Plans, stream: &Stream) -> LabResult<()> {
        let (to, bytes) = &plans.upload;
        self.buffer(*to)?
            .copy_from_bytes_at(to.offset as usize, bytes, stream)
            .map_err(|e| e.to_string())
    }

    /// The launch of the bound plan's ABI arguments, each span the address
    /// of its first byte and each pointer array's slots resolved alike, as
    /// the served host launches them.
    fn launch(&self, plans: &Plans, stream: &Stream) -> LabResult<()> {
        let arguments = plans.launch.arguments().map_err(refused)?;
        let arrays = arguments
            .iter()
            .filter_map(|argument| match argument {
                KernelArgumentV1::DevicePointerArray(slots) => Some(
                    slots
                        .iter()
                        .map(|slot| slot.map_or(Ok(0), |span| self.address(span)))
                        .collect::<LabResult<Vec<_>>>(),
                ),
                _ => None,
            })
            .collect::<LabResult<Vec<_>>>()?;
        let mut arrays = arrays.iter();
        let mut values = Vec::with_capacity(arguments.len());
        for argument in &arguments {
            values.push(match argument {
                KernelArgumentV1::DeviceBuffer(span) => KernelArgument::Range {
                    buffer: self.buffer(*span)?,
                    offset: span.offset as usize,
                    byte_len: span.length as usize,
                },
                KernelArgumentV1::DevicePointerArray(_) => {
                    KernelArgument::DevicePointerArray(arrays.next().ok_or("a pointer array")?)
                }
                KernelArgumentV1::ScalarU32(value) => KernelArgument::U32(*value),
                KernelArgumentV1::ScalarU64(value) => KernelArgument::U64(*value),
                KernelArgumentV1::ScalarI32(value) => KernelArgument::I32(*value),
                KernelArgumentV1::NullDevicePointer => KernelArgument::NullBuffer,
                other => return Err(format!("the lab launches no {other:?}")),
            });
        }
        let config = kernel_harness::launch::config(plans.launch.launch.plan)?;
        kernel_harness::launch::launch_with(&self.function, stream, config, &values)
    }

    fn arm(&self, arm: Arm, plans: &Plans, stream: &Stream) -> LabResult<()> {
        match arm {
            Arm::Copies => self.copies(plans, stream),
            Arm::Kernel => {
                self.upload_ring(plans, stream)?;
                self.launch(plans, stream)
            }
            Arm::KernelFenced => {
                self.upload_fenced(plans, stream)?;
                self.launch(plans, stream)
            }
            Arm::Launch => self.launch(plans, stream),
            Arm::UploadRing => self.upload_ring(plans, stream),
            Arm::UploadFenced => self.upload_fenced(plans, stream),
            Arm::Launches(count) => {
                for _ in 0..count {
                    self.launch(plans, stream)?;
                }
                Ok(())
            }
            Arm::Reference => {
                let scratch = self
                    .scratch
                    .as_ref()
                    .ok_or("the reference copy needs scratch")?;
                let bytes = (plans.table.pages() * plans.table.page_bytes) as usize;
                scratch
                    .copy_range_from_async(
                        0,
                        self.arena(POOL)?,
                        POOL_OFFSET as usize,
                        bytes,
                        stream,
                    )
                    .map_err(|e| e.to_string())
            }
        }
    }

    /// Writes `bytes` at `span`, staged and waited for.
    fn write(&self, span: DeviceBufferSpanV1, bytes: &[u8]) -> LabResult<()> {
        self.buffer(span)?
            .copy_from_bytes_at(span.offset as usize, bytes, &self.session.stream)
            .map_err(|e| e.to_string())
    }

    /// Fills all of allocation `raw` with `value`.
    fn fill(&self, raw: u64, value: u8) -> LabResult<()> {
        kernel_harness::poison::fill_all(self.session, self.arena(raw)?, &[value])
    }

    /// The bytes of `length` at `offset` of allocation `raw`.
    fn read(&self, raw: u64, offset: u64, bytes: &mut [u8]) -> LabResult<()> {
        let stream = &self.session.stream;
        self.arena(raw)?
            .copy_to_bytes_at_async(offset as usize, bytes, stream)
            .map_err(|e| e.to_string())?;
        stream.synchronize().map_err(|e| e.to_string())
    }

    fn read_span(&self, span: DeviceBufferSpanV1) -> LabResult<Vec<u8>> {
        let mut bytes = vec![0; span.length as usize];
        self.read(span.buffer.raw, span.offset, &mut bytes)?;
        Ok(bytes)
    }

    /// Checks every byte of allocation `expected.arena` against
    /// `expected`, reading it back a chunk at a time; returns its bytes.
    fn verify(&self, expected: &Expected, fill: u8) -> LabResult<u64> {
        let total = self.arena(expected.arena)?.byte_len() as u64;
        let mut chunk = vec![0; CHUNK_BYTES.min(total) as usize];
        let mut at = 0;
        while at < total {
            let length = CHUNK_BYTES.min(total - at);
            let observed = &mut chunk[..length as usize];
            self.read(expected.arena, at, observed)?;
            expected.check(at, observed, fill)?;
            at += length;
        }
        Ok(total)
    }
}

/// Seeded bytes, distinct for every salt.
fn seeded(length: u64, salt: u64) -> Vec<u8> {
    let mut x = salt.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    let mut bytes = Vec::with_capacity(length as usize + 8);
    while (bytes.len() as u64) < length {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        bytes.extend_from_slice(&x.to_le_bytes());
    }
    bytes.truncate(length as usize);
    bytes
}

/// The bytes a case starts from: a capture's every held span, or a
/// restore's snapshot pages, compact in page order.
struct Seeds {
    spans: Vec<(DeviceBufferSpanV1, Vec<u8>)>,
    pages: Vec<u8>,
}

/// The sources' seeds, for the host alone or also written to `device`.
fn seeds(
    bench: &Bench,
    plans: &Plans,
    to_pages: bool,
    salt: u64,
    device: Option<&Device<'_>>,
) -> LabResult<Seeds> {
    let mut seeds = Seeds {
        spans: Vec::new(),
        pages: Vec::new(),
    };
    if to_pages {
        for (index, span) in bench.held_spans().into_iter().enumerate() {
            let bytes = seeded(span.length, salt * 1000 + index as u64);
            if let Some(device) = device {
                device.write(span, &bytes)?;
            }
            seeds.spans.push((span, bytes));
        }
    } else {
        let page_bytes = plans.table.page_bytes;
        for (index, &id) in plans.ids.iter().enumerate() {
            let bytes = seeded(page_bytes, salt * 1000 + 500 + index as u64);
            if let Some(device) = device {
                device.write(bench.page_span(id, page_bytes), &bytes)?;
            }
            seeds.pages.extend(bytes);
        }
    }
    Ok(seeds)
}

/// What an allocation holds after a case: these bytes at these offsets,
/// sorted and apart, and the fill everywhere else.
struct Expected {
    arena: u64,
    regions: Vec<(u64, Vec<u8>)>,
}

impl Expected {
    /// Checks `observed`, the allocation's bytes from `at`.
    fn check(&self, at: u64, observed: &[u8], fill: u8) -> LabResult<()> {
        let end = at + observed.len() as u64;
        let mut cursor = at;
        let mut next = self
            .regions
            .partition_point(|(start, bytes)| start + bytes.len() as u64 <= at);
        while cursor < end {
            let region = self.regions.get(next);
            let start = region.map_or(u64::MAX, |(start, _)| *start);
            let (stop, wanted) = if cursor < start {
                (start.min(end), None)
            } else {
                let (start, bytes) = region.unwrap();
                let stop = (start + bytes.len() as u64).min(end);
                next += 1;
                (
                    stop,
                    Some(&bytes[(cursor - start) as usize..(stop - start) as usize]),
                )
            };
            let seen = &observed[(cursor - at) as usize..(stop - at) as usize];
            let differs = match wanted {
                Some(wanted) => first_difference(seen, wanted),
                None => seen.iter().position(|&byte| byte != fill),
            };
            if let Some(offset) = differs {
                return Err(format!(
                    "byte {} of allocation {} differs from the oracle's ({:#04x}{})",
                    cursor + offset as u64,
                    self.arena,
                    seen[offset],
                    if wanted.is_some() {
                        ", a snapshot byte"
                    } else {
                        ", outside the snapshot"
                    }
                ));
            }
            cursor = stop;
        }
        Ok(())
    }
}

/// What the oracle leaves in each destination allocation, filled with
/// `fill` beforehand, from `seeds`. The oracle runs over a compact pool
/// of the snapshot's pages in order, page `i` standing for `plans.ids[i]`.
fn oracle(
    bench: &Bench,
    plans: &Plans,
    to_pages: bool,
    seeds: &Seeds,
    fill: u8,
) -> LabResult<Vec<Expected>> {
    let (views, segments) = plans.table.segments(&bench.held).map_err(refused)?;
    let page_bytes = plans.table.page_bytes as usize;
    let count = plans.ids.len();
    let compact = (0..count as u64).collect::<Vec<_>>();
    let op = if to_pages {
        bench.capture.0
    } else {
        bench.restore.0
    };
    let mut held = views
        .iter()
        .map(|view| {
            if to_pages {
                seeds
                    .spans
                    .iter()
                    .find(|(span, _)| span == view)
                    .map(|(_, bytes)| bytes.clone())
                    .ok_or_else(|| format!("no seeds for {view:?}"))
            } else {
                Ok(vec![fill; view.length as usize])
            }
        })
        .collect::<LabResult<Vec<_>>>()?;
    let mut pool = if to_pages {
        vec![fill; count * page_bytes]
    } else {
        seeds.pages.clone()
    };
    let mut slices = held
        .iter_mut()
        .map(|bytes| bytes.as_mut_slice())
        .collect::<Vec<_>>();
    kernel_core::snapshot_copy::snapshot_copy(&op, &mut slices, &mut pool, &compact, &segments)
        .map_err(refused)?;
    let mut expected = bench
        .destinations(to_pages)
        .into_iter()
        .map(|arena| (arena, Vec::new()))
        .collect::<BTreeMap<_, _>>();
    if to_pages {
        let regions = expected.get_mut(&POOL).ok_or("no pool")?;
        for (index, &id) in plans.ids.iter().enumerate() {
            let bytes = pool[index * page_bytes..(index + 1) * page_bytes].to_vec();
            regions.push((bench.page_span(id, page_bytes as u64).offset, bytes));
        }
    } else {
        for (view, bytes) in views.iter().zip(held) {
            let regions = expected
                .get_mut(&view.buffer.raw)
                .ok_or("a view outside the state")?;
            regions.push((view.offset, bytes));
        }
    }
    Ok(expected
        .into_iter()
        .map(|(arena, mut regions)| {
            regions.sort_by_key(|(start, _)| *start);
            Expected { arena, regions }
        })
        .collect())
}

/// The first byte at which two buffers differ.
fn first_difference(left: &[u8], right: &[u8]) -> Option<usize> {
    if left.len() != right.len() {
        return Some(left.len().min(right.len()));
    }
    left.iter().zip(right).position(|(l, r)| l != r)
}

/// Runs both arms with every destination allocation filled with each
/// poison (the first alone in the high layout) and compares all of their
/// bytes with the oracle's, then the sources with their seeds. Returns the
/// bytes compared.
fn compare(
    bench: &Bench,
    device: &Device<'_>,
    plans: &Plans,
    to_pages: bool,
    seeds: &Seeds,
) -> LabResult<u64> {
    let poisons = match bench.layout {
        Layout::Packed => &POISONS[..],
        Layout::High => &POISONS[..1],
    };
    let mut compared = 0;
    for &poison in poisons {
        let expected = oracle(bench, plans, to_pages, seeds, poison)?;
        for arm in [Arm::Copies, Arm::Kernel] {
            for destination in &expected {
                device.fill(destination.arena, poison)?;
            }
            device.arm(arm, plans, &device.session.stream)?;
            device
                .session
                .stream
                .synchronize()
                .map_err(|e| e.to_string())?;
            for destination in &expected {
                compared += device.verify(destination, poison).map_err(|error| {
                    format!("{} arm, poison {poison:#04x}: {error}", arm.name())
                })?;
            }
        }
    }
    if to_pages {
        for (span, bytes) in &seeds.spans {
            if device.read_span(*span)? != *bytes {
                return Err(format!("source span {span:?} changed"));
            }
        }
    } else {
        let page_bytes = plans.table.page_bytes;
        for (index, &id) in plans.ids.iter().enumerate() {
            let at = index * page_bytes as usize;
            if device.read_span(bench.page_span(id, page_bytes))?
                != seeds.pages[at..at + page_bytes as usize]
            {
                return Err(format!("source page {id} changed"));
            }
        }
    }
    Ok(compared)
}

fn directions(holder: Holder) -> &'static [bool] {
    match holder {
        Holder::Slot(_) => &[true, false],
        Holder::Anchor(_) => &[true],
    }
}

fn holder_name(holder: Holder) -> String {
    match holder {
        Holder::Slot(slot) => format!("slot{slot}"),
        Holder::Anchor(item) => format!("anchor{item}"),
    }
}

fn layout_name(layout: Layout) -> &'static str {
    match layout {
        Layout::Packed => "packed",
        Layout::High => "high",
    }
}

fn module(session: &Session) -> LabResult<runtime_cuda::Module> {
    let (_, bytes) =
        fold_cubin_for_symbol(SYMBOL).ok_or("the bundle embeds no snapshot copy cubin")?;
    session.load_cubin(bytes)
}

/// Every case: 90 packed, 15 high and 4 synthetic.
const CASES: u64 = 109;

/// Every case plans on the CPU: the bundle's ops prepare, the copies are
/// the table's fragments, the kernel plan is one table upload and one
/// launch of the embedded cubin's entry, its plan valid, and its arguments
/// and uploaded table name exactly the device copies. The high layout's
/// pages straddle 2 GiB and 4 GiB of the pool and its views' offsets pass
/// 2 GiB and 4 GiB.
#[test]
fn inventory() {
    let (cubin, _) =
        fold_cubin_for_symbol(SYMBOL).expect("the bundle embeds the snapshot copy cubin");
    assert_eq!(cubin, bundle_kernel::CUBIN);
    let mut cases = 0;
    for bench in benches().unwrap() {
        let layout = bench.layout;
        for holder in bench.holders() {
            for &to_pages in directions(holder) {
                for &order in bench.orders() {
                    let plans = plans(&bench, holder, order, to_pages).unwrap();
                    let (views, segments) = plans.table.segments(&bench.held).unwrap();
                    let op = if to_pages {
                        bench.capture.0
                    } else {
                        bench.restore.0
                    };
                    let fragments = op.fragments(&plans.ids, &segments).unwrap();
                    assert_eq!(plans.copies.len(), fragments.len());
                    let arguments = plans.launch.arguments().unwrap();
                    let (table, bytes) = &plans.upload;
                    assert_eq!(table.offset, TABLE_OFFSET);
                    let launched = launched_copies(&arguments, |read| {
                        assert_eq!(read, *table);
                        bytes.clone()
                    })
                    .unwrap();
                    assert_eq!(launched, plans.copies);
                    assert!(views.len() as u32 == op.views && op.views <= 8);
                    assert_eq!(plans.launch.launch.plan.workspace_bytes, 0);
                    assert!(plans.launch.launch.plan.pdl);
                    if layout == Layout::High {
                        let page_bytes = plans.table.page_bytes;
                        for boundary in [2 * GIB, 4 * GIB] {
                            assert!(plans.ids.iter().any(|&id| {
                                id * page_bytes < boundary && (id + 1) * page_bytes > boundary
                            }));
                        }
                        // Views from below each boundary to past it.
                        for (arena, boundary) in [(STATE, 4 * GIB), (SPARES, 2 * GIB)] {
                            let named = views.iter().filter(|view| view.buffer.raw == arena);
                            assert!(named.clone().all(|view| view.offset >= boundary - BELOW));
                            if named.clone().next().is_some() {
                                assert!(
                                    named
                                        .clone()
                                        .any(|view| view.offset + view.length > boundary)
                                );
                            }
                        }
                    }
                    println!(
                        "snapshot_copy_inventory {}",
                        json!({"layout": layout_name(layout), "state": bench.name(),
                                "holder": holder_name(holder),
                                "direction": if to_pages { "capture" } else { "restore" },
                                "order": format!("{order:?}"), "pages": plans.ids.len(),
                                "page_bytes": plans.table.page_bytes, "views": views.len(),
                                "view_offsets": views.iter().map(|view| view.offset).collect::<Vec<_>>(),
                                "segments": segments.len(), "copies": plans.copies.len(),
                                "table_bytes": plans.upload.1.len(),
                                "grid": plans.launch.launch.plan.grid[0]})
                    );
                    cases += 1;
                }
            }
        }
    }
    // Packed: six states, 30 holder directions, three page orders; high:
    // 15 holder directions over straddling pages; synthetic: both
    // directions over two page orders.
    assert_eq!(cases, CASES);
    println!(
        "snapshot_copy_inventory_complete {}",
        json!({"cases": cases})
    );
}

/// The lab's oracle leaves what the copies plan, run on the host over the
/// packed allocations, leaves: a TP4 slot's capture and restore over
/// scattered pages, an anchor's capture, and the synthetic table's capture
/// and restore.
#[test]
fn host_copies_match_the_oracle() {
    let declared = Bench::new(DECLARED[1], Layout::Packed).unwrap();
    let synthetic = Bench::synthetic().unwrap();
    let cases = [
        (&declared, Holder::Slot(DECODE_BATCH - 1), true),
        (&declared, Holder::Slot(DECODE_BATCH - 1), false),
        (&declared, Holder::Anchor(ANCHORS - 1), true),
        (&synthetic, Holder::Slot(0), true),
        (&synthetic, Holder::Slot(0), false),
    ];
    for (bench, holder, to_pages) in cases {
        let plans = plans(bench, holder, Order::Scattered, to_pages).unwrap();
        let seeds = seeds(bench, &plans, to_pages, 3, None).unwrap();
        let fill = POISONS[0];
        let expected = oracle(bench, &plans, to_pages, &seeds, fill).unwrap();
        let mut arenas = bench
            .arenas
            .iter()
            .map(|&(raw, bytes)| (raw, vec![0x5c; bytes as usize]))
            .collect::<BTreeMap<_, _>>();
        for destination in &expected {
            arenas.get_mut(&destination.arena).unwrap().fill(fill);
        }
        let write =
            |arenas: &mut BTreeMap<u64, Vec<u8>>, span: DeviceBufferSpanV1, bytes: &[u8]| {
                arenas.get_mut(&span.buffer.raw).unwrap()
                    [span.offset as usize..(span.offset + span.length) as usize]
                    .copy_from_slice(bytes);
            };
        for (span, bytes) in &seeds.spans {
            write(&mut arenas, *span, bytes);
        }
        let page_bytes = plans.table.page_bytes;
        for (index, &id) in plans.ids.iter().enumerate() {
            if !to_pages {
                let at = index * page_bytes as usize;
                let page = seeds.pages[at..at + page_bytes as usize].to_vec();
                write(&mut arenas, bench.page_span(id, page_bytes), &page);
            }
        }
        for &(from, to) in &plans.copies {
            let bytes = arenas[&from.buffer.raw]
                [from.offset as usize..(from.offset + from.length) as usize]
                .to_vec();
            write(&mut arenas, to, &bytes);
        }
        let mut moved = 0;
        for destination in &expected {
            destination
                .check(0, &arenas[&destination.arena], fill)
                .unwrap_or_else(|error| panic!("{holder:?} {to_pages}: {error}"));
            moved += destination
                .regions
                .iter()
                .map(|(_, bytes)| bytes.iter().filter(|&&byte| byte != fill).count())
                .sum::<usize>();
        }
        assert!(moved > 0, "{holder:?} {to_pages}");
        // A wrong byte inside the snapshot, or a stray one outside it, shows.
        for destination in expected
            .iter()
            .filter(|expected| !expected.regions.is_empty())
        {
            let observed = &arenas[&destination.arena];
            let (start, region) = &destination.regions[0];
            let mut bytes = observed.clone();
            bytes[*start as usize] = !region[0];
            assert!(destination.check(0, &bytes, fill).is_err());
            if let Some(at) = outside(destination, observed.len() as u64) {
                let mut bytes = observed.clone();
                bytes[at as usize] = !fill;
                assert!(destination.check(0, &bytes, fill).is_err());
            }
        }
    }
}

/// The first byte of an allocation of `length` that none of `expected`'s
/// regions covers.
fn outside(expected: &Expected, length: u64) -> Option<u64> {
    let mut cursor = 0;
    for (start, bytes) in &expected.regions {
        if *start > cursor {
            return Some(cursor);
        }
        cursor = cursor.max(start + bytes.len() as u64);
    }
    (cursor < length).then_some(cursor)
}

/// Every case's destination allocations are bitwise the oracle's after
/// both arms, under each poison, and its sources unchanged.
#[test]
fn bitwise_gpu() {
    kernel_harness::gpu_test(16 << 30, |s| {
        let module = module(s)?;
        let started = Instant::now();
        let mut cases = 0;
        let mut compared = 0;
        for (index, bench) in benches()?.into_iter().enumerate() {
            let layout = bench.layout;
            let device = Device::new(s, &bench, &module)?;
            for holder in bench.holders() {
                for &to_pages in directions(holder) {
                    for &order in bench.orders() {
                        if started.elapsed().as_secs() >= BUDGET_SECONDS {
                            return Err(
                                "the bitwise sweep exceeded its budget; coverage is incomplete"
                                    .into(),
                            );
                        }
                        let plans = plans(&bench, holder, order, to_pages)?;
                        let salt = index as u64 * 100 + cases;
                        let seeds = seeds(&bench, &plans, to_pages, salt, Some(&device))?;
                        let bytes = compare(&bench, &device, &plans, to_pages, &seeds)?;
                        println!(
                            "snapshot_copy_bitwise {}",
                            json!({"layout": layout_name(layout), "state": bench.name(),
                                    "holder": holder_name(holder),
                                    "direction": if to_pages { "capture" } else { "restore" },
                                    "order": format!("{order:?}"), "copies": plans.copies.len(),
                                    "bytes_compared": bytes, "bitwise": true})
                        );
                        compared += bytes;
                        cases += 1;
                    }
                }
            }
        }
        if cases != CASES {
            return Err(format!("only {cases} of {CASES} cases ran"));
        }
        let mut data = BTreeMap::new();
        data.insert("cases".into(), json!(cases));
        data.insert("bytes_compared".into(), json!(compared));
        data.insert("symbol".into(), json!(SYMBOL));
        let receipt = s.receipt("snapshot_copy_bitwise", data)?;
        println!(
            "snapshot_copy_bitwise_complete {}",
            json!({"cases": cases, "bytes_compared": compared, "receipt": receipt})
        );
        Ok(())
    })
    .unwrap();
}

fn median(values: &[f64]) -> f64 {
    let mut values = values.to_vec();
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

/// One sample of `arm`: its CUDA-event time, its host wall time from before
/// the first event to after the stream drains, and its host issue time
/// from before the first event to the arm's return, in ms.
fn sample(
    device: &Device<'_>,
    run: &dyn Fn(&Stream) -> LabResult<()>,
    events: &(runtime_cuda::Event, runtime_cuda::Event),
) -> LabResult<[f64; 3]> {
    let s = device.session;
    let wall = Instant::now();
    events.0.record(&s.stream).map_err(|e| e.to_string())?;
    run(&s.stream)?;
    let issue = wall.elapsed().as_secs_f64() * 1e3;
    events.1.record(&s.stream).map_err(|e| e.to_string())?;
    s.stream.synchronize().map_err(|e| e.to_string())?;
    let host = wall.elapsed().as_secs_f64() * 1e3;
    let event = f64::from(events.0.elapsed_ms(&events.1).map_err(|e| e.to_string())?);
    Ok([event, host, issue])
}

const MEASURES: [&str; 3] = ["event_ms", "host_ms", "issue_ms"];

fn events(device: &Device<'_>) -> LabResult<(runtime_cuda::Event, runtime_cuda::Event)> {
    let context = &device.session.context;
    Ok((
        context.timed_event().map_err(|e| e.to_string())?,
        context.timed_event().map_err(|e| e.to_string())?,
    ))
}

/// Five interleaved rounds of three samples per arm after three warm-up
/// runs of each; `arms` names each arm.
fn pair(device: &Device<'_>, plans: &Plans, arms: [Arm; 2]) -> LabResult<Value> {
    let s = device.session;
    let events = events(device)?;
    for _ in 0..3 {
        for arm in arms {
            device.arm(arm, plans, &s.stream)?;
        }
    }
    s.stream.synchronize().map_err(|e| e.to_string())?;
    let mut rounds = Vec::new();
    let mut medians = [
        [Vec::new(), Vec::new()],
        [Vec::new(), Vec::new()],
        [Vec::new(), Vec::new()],
    ];
    for round in 0..5 {
        let mut values = [
            [Vec::new(), Vec::new()],
            [Vec::new(), Vec::new()],
            [Vec::new(), Vec::new()],
        ];
        for sample_index in 0..3 {
            for step in 0..2 {
                let arm = (round + sample_index + step) % 2;
                let run = |stream: &Stream| device.arm(arms[arm], plans, stream);
                let measured = sample(device, &run, &events)?;
                for (kind, value) in measured.into_iter().enumerate() {
                    values[kind][arm].push(value);
                }
            }
        }
        for kind in 0..3 {
            for arm in 0..2 {
                medians[kind][arm].push(median(&values[kind][arm]));
            }
        }
        rounds.push(values);
    }
    let mut record = json!({"arms": arms.map(Arm::name), "round_samples_ms": rounds});
    for (kind, name) in MEASURES.into_iter().enumerate() {
        record[name] = json!([median(&medians[kind][0]), median(&medians[kind][1])]);
    }
    Ok(record)
}

/// Fifteen samples of `run` after three warm-up runs: each measure's
/// median, minimum and maximum.
fn series(device: &Device<'_>, run: &dyn Fn(&Stream) -> LabResult<()>) -> LabResult<Value> {
    let s = device.session;
    let events = events(device)?;
    for _ in 0..3 {
        run(&s.stream)?;
    }
    s.stream.synchronize().map_err(|e| e.to_string())?;
    let mut values = [Vec::new(), Vec::new(), Vec::new()];
    for _ in 0..15 {
        let measured = sample(device, run, &events)?;
        for (kind, value) in measured.into_iter().enumerate() {
            values[kind].push(value);
        }
    }
    let mut record = json!({});
    for (kind, name) in MEASURES.into_iter().enumerate() {
        let min = values[kind].iter().copied().fold(f64::INFINITY, f64::min);
        let max = values[kind].iter().copied().fold(0.0, f64::max);
        record[name] = json!({"median": median(&values[kind]), "min": min, "max": max});
    }
    Ok(record)
}

/// The kernel arm split into its parts, each timed alone, and the copies
/// and the bandwidth reference beside them. The ring upload's series
/// leaves this case's table resident for the launches'. The launches run
/// eagerly, where the host may pace them, and replayed from one graph of
/// [`LAUNCHES`], which times the device alone.
fn split(device: &Device<'_>, plans: &Plans) -> LabResult<Value> {
    let mut record = json!({});
    for arm in [
        Arm::UploadFenced,
        Arm::UploadRing,
        Arm::Launch,
        Arm::Launches(LAUNCHES),
        Arm::KernelFenced,
        Arm::Kernel,
        Arm::Copies,
        Arm::Reference,
    ] {
        record[arm.name()] = series(device, &|stream: &Stream| device.arm(arm, plans, stream))?;
    }
    let graph = device.session.capture(|stream| {
        for _ in 0..LAUNCHES {
            device.launch(plans, stream)?;
        }
        Ok(())
    })?;
    let replay = |stream: &Stream| graph.launch(stream).map_err(|e| e.to_string());
    record[format!("graph_{LAUNCHES}")] = series(device, &replay)?;
    Ok(record)
}

/// The timed cases: TP4's served states, each a slot at the end of the
/// decode batch capturing and restoring over scattered pages, and the
/// anchored DSpark state's last anchor capturing.
fn timed() -> Vec<(Declared, Holder, bool)> {
    let last = DECODE_BATCH - 1;
    vec![
        (DECLARED[1], Holder::Slot(last), true),
        (DECLARED[1], Holder::Slot(last), false),
        (DECLARED[3], Holder::Slot(last), true),
        (DECLARED[3], Holder::Slot(last), false),
        (DECLARED[3], Holder::Anchor(ANCHORS - 1), true),
        (DECLARED[5], Holder::Slot(last), true),
        (DECLARED[5], Holder::Slot(last), false),
    ]
}

/// Pair-lab timing of the device copies against the kernel. Per case,
/// after a bitwise comparison of both arms: copies A/A, copies against
/// kernel, kernel A/A and a closing copies A/A, the split of the kernel arm
/// ([`split`]), then the comparison again.
#[test]
#[ignore = "pair-lab timing; run on an idle GPU through its job"]
fn pair_gpu() {
    kernel_harness::gpu_test(4 << 30, |s| {
        let module = module(s)?;
        let started = Instant::now();
        let mut records = Vec::new();
        for (index, (declared, holder, to_pages)) in timed().into_iter().enumerate() {
            if started.elapsed().as_secs() >= PAIR_BUDGET_SECONDS {
                return Err("the pair lab exceeded its budget; coverage is incomplete".into());
            }
            let bench = Bench::new(declared, Layout::Packed)?;
            let mut device = Device::new(s, &bench, &module)?;
            let plans = plans(&bench, holder, Order::Scattered, to_pages)?;
            let paged_bytes = plans.table.pages() * plans.table.page_bytes;
            device.scratch = Some(s.allocate(paged_bytes as usize)?);
            let seeds = seeds(&bench, &plans, to_pages, 7000 + index as u64, Some(&device))?;
            compare(&bench, &device, &plans, to_pages, &seeds)?;
            let aa_before = pair(&device, &plans, [Arm::Copies, Arm::Copies])?;
            let ab = pair(&device, &plans, [Arm::Copies, Arm::Kernel])?;
            let bb = pair(&device, &plans, [Arm::Kernel, Arm::Kernel])?;
            let aa_after = pair(&device, &plans, [Arm::Copies, Arm::Copies])?;
            let split = split(&device, &plans)?;
            compare(&bench, &device, &plans, to_pages, &seeds)?;
            let plan = plans.launch.launch.plan;
            let record = json!({"state": bench.name(), "holder": holder_name(holder),
                "direction": if to_pages { "capture" } else { "restore" },
                "pages": plans.ids.len(), "copies": plans.copies.len(),
                "segments": plans.table.descriptors.len(),
                "snapshot_bytes": plans.table.bytes(), "paged_bytes": paged_bytes,
                "table_bytes": plans.upload.1.len(), "grid": plan.grid, "block": plan.block,
                "launches": LAUNCHES, "bitwise": true,
                "aa_before": aa_before, "ab": ab, "bb": bb, "aa_after": aa_after, "split": split});
            println!("snapshot_copy_pair {record}");
            records.push(record);
        }
        let mut data = BTreeMap::new();
        data.insert("records".into(), json!(records));
        data.insert("symbol".into(), json!(SYMBOL));
        let receipt = s.receipt("snapshot_copy_pair", data)?;
        println!(
            "snapshot_copy_pair_complete {}",
            json!({"cases": records.len(), "receipt": receipt})
        );
        Ok(())
    })
    .unwrap();
}
