//! The shared snapshot copy kernel (`prefix_snapshot_copy_kernel`) against
//! the device copies it replaces, over DeepSeek V4.1 Flash's prefix
//! snapshot tables. Both arms plan from the bundle's table
//! (`bundle_runtime::snapshot`): the copies arm issues `Table::actions`'
//! copies as the served host does, one `cuMemcpyDtoDAsync` each, and the
//! kernel arm `Table::kernel_actions`' table upload (staged and fenced, as
//! the host's upload is) and launch, through the backend op the bundle
//! registers and the cubin it embeds and locks.
//!
//! The per-rank state declaration does not depend on the rank count, so
//! the cases cover each KV storage, DSpark on and off and anchor spares on
//! and off; a slot's table at the first and last of the decode batch's
//! slots and an anchor item's at the first and last anchor; and the
//! snapshot's pages ascending, scattered over the pool and descending.
//! Slots capture and restore; anchors only capture, as served.
//!
//! [`bitwise_gpu`] compares every byte of every destination buffer, and
//! the sources after, with the destinations poisoned twice, across both
//! arms and the CPU oracle (`kernel_core::snapshot_copy::snapshot_copy`).
//! [`pair_gpu`] times both arms with CUDA events and host wall time in a
//! pair lab with A/A controls (docs/contributing.md#pair-lab-kernel-timing).
//! [`inventory`] plans every case on the CPU.

use bundle_deepseek41::sm120::{fold_cubin_for_symbol, snapshot_copy as bundle_kernel};
use bundle_runtime::geometry::Items;
use bundle_runtime::snapshot::{
    Action, BoundLaunch, HeaderAction, HeldBuffers, KernelCopy, Pages, Table, launched_copies,
};
use bundle_runtime::state::{Extent, Holder, RequestState};
use kernel_core::backend::{Backend, Need, Prepared, View};
use kernel_core::contract::Argument;
use kernel_core::snapshot_copy::{Direction, SnapshotCopy, SnapshotCopyBuffers};
use kernel_harness::launch::{Bindings, launch};
use kernel_harness::{LabResult, Session};
use kernels_sm120::backend::Sm120;
use kernels_sm120::snapshot_copy::SYMBOL;
use model_deepseek41_contract::geometry::FLASH;
use model_deepseek41_contract::serving::DECODE_BATCH;
use model_deepseek41_contract::state::{KvStorage, request_state_with_anchors};
use runtime_bundle_abi::{BundleDeviceBufferHandleV1, DeviceBufferSpanV1};
use runtime_cuda::{DeviceBuffer, Function, Stream};
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::time::Instant;

/// Anchor items an anchored instance declares at TP4.
const ANCHORS: u32 = 4;
/// Pages of the lab's pool.
const POOL_PAGES: u64 = 512;
/// The destinations' fills: a write of either value shows under the other.
const POISONS: [u8; 2] = [0xa5, 0x00];
/// The bitwise sweep stops here, with no result, so its job stays in its
/// bound.
const BUDGET_SECONDS: u64 = 240;
/// The pair lab's bound: its job times one GPU after the other.
const PAIR_BUDGET_SECONDS: u64 = 120;
/// Buffer handles of the lab's spans: declared buffer `i` is `BUFFER + i`
/// and its spare `SPARE + i`.
const BUFFER: u64 = 1;
const SPARE: u64 = 101;
const POOL: u64 = 201;
const TABLE: u64 = 301;

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

#[derive(Clone, Copy, Debug)]
enum Order {
    Ascending,
    Scattered,
    Descending,
}

const ORDERS: [Order; 3] = [Order::Ascending, Order::Scattered, Order::Descending];

/// `count` distinct pool pages in `order`.
fn page_ids(order: Order, count: u64) -> Vec<u64> {
    match order {
        Order::Ascending => (0..count).map(|i| 3 + i).collect(),
        // 211 is odd, so it permutes the 512 pages.
        Order::Scattered => (0..count).map(|i| (i * 211 + 17) % POOL_PAGES).collect(),
        Order::Descending => (0..count).map(|i| POOL_PAGES - 1 - 2 * i).collect(),
    }
}

fn span(raw: u64, length: u64) -> DeviceBufferSpanV1 {
    DeviceBufferSpanV1::new(BundleDeviceBufferHandleV1 { raw }, 0, length)
}

fn refused(error: impl std::fmt::Debug) -> String {
    format!("{error:?}")
}

/// One declaration's state in an instance serving the decode batch, its
/// held buffers (each declared buffer and spare its own allocation), the
/// pool and the table buffer, and the kernel ops prepared as the bundle
/// prepares them: both directions over the snapshot's pages and views, for
/// the most pieces a slot's or an anchor item's table has.
struct Bench {
    declared: Declared,
    state: RequestState,
    extent: Extent,
    held: HeldBuffers,
    pool: DeviceBufferSpanV1,
    table: DeviceBufferSpanV1,
    capture: (SnapshotCopy, Prepared<SnapshotCopyBuffers<View>>),
    restore: (SnapshotCopy, Prepared<SnapshotCopyBuffers<View>>),
}

impl Bench {
    fn new(declared: Declared) -> LabResult<Self> {
        let state =
            request_state_with_anchors(&FLASH, declared.storage, declared.dspark, declared.anchors);
        let items = Items::new(DECODE_BATCH).ok_or("the decode batch is no item count")?;
        let extent = Extent {
            served: items.get(),
            bucket: items.capacity(),
            // Five drafts and the bonus row.
            verify_rows: if declared.dspark { 6 } else { 0 },
        };
        let held = HeldBuffers {
            buffers: (BUFFER..)
                .zip(&state.buffers)
                .map(|(raw, buffer)| Some(span(raw, buffer.bytes(extent))))
                .collect(),
            spares: (SPARE..)
                .zip(&state.buffers)
                .map(|(raw, buffer)| {
                    (buffer.spare_bytes() > 0).then(|| span(raw, buffer.spare_bytes()))
                })
                .collect(),
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
        let op = |direction| SnapshotCopy {
            direction,
            views,
            page_bytes: tables[0].page_bytes,
            pages: tables[0].pages() as u32,
        };
        let backend = bundle_kernel::register(Sm120::new());
        let prepare = |direction| -> LabResult<_> {
            let op = op(direction);
            let prepared = backend
                .snapshot_copy(&op, Need::rows(capacity))
                .map_err(|refusal| refusal.to_string())?;
            Ok((op, prepared))
        };
        let capture = prepare(Direction::ToPages)?;
        let restore = prepare(Direction::FromPages)?;
        let table_bytes = capture.0.table_bytes(capacity).map_err(refused)?;
        Ok(Self {
            declared,
            pool: span(POOL, POOL_PAGES * tables[0].page_bytes),
            table: span(TABLE, table_bytes),
            state,
            extent,
            held,
            capture,
            restore,
        })
    }

    /// Every span the lab allocates: the held buffers and spares, the pool
    /// and the table.
    fn spans(&self) -> Vec<DeviceBufferSpanV1> {
        let held = self.held.buffers.iter().chain(&self.held.spares).flatten();
        held.copied().chain([self.pool, self.table]).collect()
    }

    /// The held buffers and spares.
    fn state_spans(&self) -> Vec<DeviceBufferSpanV1> {
        let held = self.held.buffers.iter().chain(&self.held.spares).flatten();
        held.copied().collect()
    }

    /// The holders of its snapshots: the decode batch's first and last
    /// slots and its first and last anchor items.
    fn holders(&self) -> Vec<Holder> {
        let mut holders = vec![Holder::Slot(0), Holder::Slot(self.extent.served - 1)];
        if self.declared.anchors > 0 {
            holders.extend([Holder::Anchor(0), Holder::Anchor(self.declared.anchors - 1)]);
        }
        holders
    }

    fn name(&self) -> String {
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
    let table = Table::new(&bench.state, bench.extent, holder).map_err(refused)?;
    let ids = page_ids(order, table.pages());
    let paged = Pages {
        pool: bench.pool,
        ids: &ids,
        page_bytes: table.page_bytes,
    };
    paged.check(&table, POOL_PAGES).map_err(refused)?;
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

/// The device side of one bench: an allocation per span, by buffer handle,
/// and the kernel's function.
struct Device<'a> {
    session: &'a Session,
    buffers: BTreeMap<u64, DeviceBuffer>,
    function: Function,
}

impl<'a> Device<'a> {
    fn new(session: &'a Session, bench: &Bench, module: &runtime_cuda::Module) -> LabResult<Self> {
        let mut buffers = BTreeMap::new();
        for span in bench.spans() {
            buffers.insert(span.buffer.raw, session.allocate(span.length as usize)?);
        }
        let function = module.function(SYMBOL).map_err(|e| e.to_string())?;
        Ok(Self {
            session,
            buffers,
            function,
        })
    }

    fn buffer(&self, span: DeviceBufferSpanV1) -> LabResult<&DeviceBuffer> {
        self.buffers
            .get(&span.buffer.raw)
            .ok_or_else(|| format!("no allocation for {span:?}"))
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

    /// The table upload, staged and fenced like the host's, then the
    /// launch, its views bound to whole allocations.
    fn kernel(&self, plans: &Plans, stream: &Stream) -> LabResult<()> {
        let (to, bytes) = &plans.upload;
        self.buffer(*to)?
            .copy_from_bytes_at(to.offset as usize, bytes, stream)
            .map_err(|e| e.to_string())?;
        let launched = &plans.launch;
        let mut bound = BTreeMap::new();
        for argument in &launched.launch.arguments.0 {
            let handles = match argument {
                Argument::DeviceBuffer(handle) => vec![*handle],
                Argument::DevicePointerArray(handles) => handles.clone(),
                _ => Vec::new(),
            };
            for handle in handles {
                let span = launched.bindings.get(handle).map_err(refused)?;
                if span.offset != 0 {
                    return Err(format!("the lab binds whole allocations, not {span:?}"));
                }
                bound.insert(handle, self.buffer(span)?);
            }
        }
        launch(
            &self.session.context,
            &self.function,
            stream,
            launched.launch.plan,
            &launched.launch.arguments,
            &Bindings { buffers: bound },
        )
    }

    fn arm(&self, kernel: bool, plans: &Plans, stream: &Stream) -> LabResult<()> {
        if kernel {
            self.kernel(plans, stream)
        } else {
            self.copies(plans, stream)
        }
    }

    fn write(&self, span: DeviceBufferSpanV1, bytes: &[u8]) -> LabResult<()> {
        self.buffer(span)?
            .copy_from_bytes(bytes, &self.session.stream)
            .map_err(|e| e.to_string())
    }

    fn fill(&self, span: DeviceBufferSpanV1, value: u8) -> LabResult<()> {
        kernel_harness::poison::fill_all(self.session, self.buffer(span)?, &[value])
    }

    fn read(&self, span: DeviceBufferSpanV1) -> LabResult<Vec<u8>> {
        let mut bytes = kernel_harness::poison::read(self.session, self.buffer(span)?)?;
        bytes.truncate(span.length as usize);
        Ok(bytes)
    }
}

/// Seeded bytes, distinct for every buffer.
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

/// The sources' and destinations' spans of a direction.
fn sides(bench: &Bench, to_pages: bool) -> (Vec<DeviceBufferSpanV1>, Vec<DeviceBufferSpanV1>) {
    if to_pages {
        (bench.state_spans(), vec![bench.pool])
    } else {
        (vec![bench.pool], bench.state_spans())
    }
}

/// What the oracle leaves in `destinations` (each filled with `poison`)
/// from `sources`.
fn oracle(
    bench: &Bench,
    plans: &Plans,
    to_pages: bool,
    sources: &BTreeMap<u64, Vec<u8>>,
    destinations: &[DeviceBufferSpanV1],
    poison: u8,
) -> LabResult<BTreeMap<u64, Vec<u8>>> {
    let mut bytes = sources.clone();
    for span in destinations {
        bytes.insert(span.buffer.raw, vec![poison; span.length as usize]);
    }
    let (views, segments) = plans.table.segments(&bench.held).map_err(refused)?;
    let mut pool = bytes.remove(&POOL).ok_or("no pool bytes")?;
    let mut held = views
        .iter()
        .map(|view| {
            bytes
                .remove(&view.buffer.raw)
                .map(|bytes| (view.buffer.raw, bytes))
                .ok_or("no view bytes")
        })
        .collect::<Result<Vec<_>, _>>()?;
    let op = if to_pages {
        bench.capture.0
    } else {
        bench.restore.0
    };
    let mut slices = held
        .iter_mut()
        .map(|(_, bytes)| bytes.as_mut_slice())
        .collect::<Vec<_>>();
    kernel_core::snapshot_copy::snapshot_copy(&op, &mut slices, &mut pool, &plans.ids, &segments)
        .map_err(refused)?;
    bytes.extend(held);
    bytes.insert(POOL, pool);
    Ok(destinations
        .iter()
        .map(|span| (span.buffer.raw, bytes.remove(&span.buffer.raw).unwrap()))
        .collect())
}

/// Seeds the sources; returns their bytes.
fn seed(
    device: &Device<'_>,
    sources: &[DeviceBufferSpanV1],
    salt: u64,
) -> LabResult<BTreeMap<u64, Vec<u8>>> {
    let mut written = BTreeMap::new();
    for span in sources {
        let bytes = seeded(span.length, salt * 1000 + span.buffer.raw);
        device.write(*span, &bytes)?;
        written.insert(span.buffer.raw, bytes);
    }
    Ok(written)
}

/// The first byte at which two buffers differ.
fn first_difference(left: &[u8], right: &[u8]) -> Option<usize> {
    if left.len() != right.len() {
        return Some(left.len().min(right.len()));
    }
    left.iter().zip(right).position(|(l, r)| l != r)
}

/// Runs both arms with every destination poisoned by each fill and
/// compares all of their bytes with the oracle's, then the sources with
/// their seeds. Returns the bytes compared.
fn compare(
    bench: &Bench,
    device: &Device<'_>,
    plans: &Plans,
    to_pages: bool,
    sources: &BTreeMap<u64, Vec<u8>>,
) -> LabResult<u64> {
    let (source_spans, destinations) = sides(bench, to_pages);
    let mut compared = 0;
    for poison in POISONS {
        let expected = oracle(bench, plans, to_pages, sources, &destinations, poison)?;
        for kernel in [false, true] {
            for span in &destinations {
                device.fill(*span, poison)?;
            }
            device.arm(kernel, plans, &device.session.stream)?;
            device
                .session
                .stream
                .synchronize()
                .map_err(|e| e.to_string())?;
            for span in &destinations {
                let observed = device.read(*span)?;
                if let Some(at) = first_difference(&observed, &expected[&span.buffer.raw]) {
                    return Err(format!(
                        "{} arm differs from the oracle in buffer {} at byte {at} (poison {poison:#04x})",
                        if kernel { "kernel" } else { "copies" },
                        span.buffer.raw
                    ));
                }
                compared += span.length;
            }
        }
    }
    for span in &source_spans {
        if device.read(*span)? != sources[&span.buffer.raw] {
            return Err(format!("source buffer {} changed", span.buffer.raw));
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

fn module(session: &Session) -> LabResult<runtime_cuda::Module> {
    let (_, bytes) =
        fold_cubin_for_symbol(SYMBOL).ok_or("the bundle embeds no snapshot copy cubin")?;
    session.load_cubin(bytes)
}

/// Every case plans on the CPU: the bundle's ops prepare, the copies are
/// the table's fragments, the kernel plan is one table upload and one
/// launch of the embedded cubin's entry, its plan valid.
#[test]
fn inventory() {
    let (cubin, _) =
        fold_cubin_for_symbol(SYMBOL).expect("the bundle embeds the snapshot copy cubin");
    assert_eq!(cubin, bundle_kernel::CUBIN);
    let mut cases = 0;
    for declared in DECLARED {
        let bench = Bench::new(declared).unwrap();
        for holder in bench.holders() {
            for &to_pages in directions(holder) {
                for order in ORDERS {
                    let plans = plans(&bench, holder, order, to_pages).unwrap();
                    let (views, segments) = plans.table.segments(&bench.held).unwrap();
                    let op = if to_pages {
                        bench.capture.0
                    } else {
                        bench.restore.0
                    };
                    let fragments = op.fragments(&plans.ids, &segments).unwrap();
                    assert_eq!(plans.copies.len(), fragments.len());
                    // The launch and its uploaded table name the copies.
                    let arguments = plans.launch.arguments().unwrap();
                    let (table, bytes) = &plans.upload;
                    let launched = launched_copies(&arguments, |read| {
                        assert_eq!(read, *table);
                        bytes.clone()
                    })
                    .unwrap();
                    assert_eq!(launched, plans.copies);
                    assert!(views.len() as u32 == op.views && op.views <= 8);
                    assert_eq!(plans.launch.launch.plan.workspace_bytes, 0);
                    assert!(plans.launch.launch.plan.pdl);
                    println!(
                        "snapshot_copy_inventory {}",
                        json!({"state": bench.name(), "holder": holder_name(holder),
                            "direction": if to_pages { "capture" } else { "restore" },
                            "order": format!("{order:?}"), "pages": plans.ids.len(),
                            "page_bytes": plans.table.page_bytes, "views": views.len(),
                            "segments": segments.len(), "copies": plans.copies.len(),
                            "table_bytes": plans.upload.1.len(),
                            "grid": plans.launch.launch.plan.grid[0]})
                    );
                    cases += 1;
                }
            }
        }
    }
    // Six states, 18 holders, 30 holder directions, three page orders.
    assert_eq!(cases, 90);
    println!(
        "snapshot_copy_inventory_complete {}",
        json!({"cases": cases})
    );
}

/// The lab's oracle leaves what the copies plan, run on the host, leaves:
/// a TP4 slot's capture and restore over scattered pages.
#[test]
fn host_copies_match_the_oracle() {
    let bench = Bench::new(DECLARED[1]).unwrap();
    for to_pages in [true, false] {
        let plans = plans(
            &bench,
            Holder::Slot(DECODE_BATCH - 1),
            Order::Scattered,
            to_pages,
        )
        .unwrap();
        let (sources, destinations) = sides(&bench, to_pages);
        let seeds = sources
            .iter()
            .map(|span| (span.buffer.raw, seeded(span.length, span.buffer.raw)))
            .collect::<BTreeMap<_, _>>();
        let expected = oracle(&bench, &plans, to_pages, &seeds, &destinations, POISONS[0]).unwrap();
        let mut copied = seeds.clone();
        for span in &destinations {
            copied.insert(span.buffer.raw, vec![POISONS[0]; span.length as usize]);
        }
        for &(from, to) in &plans.copies {
            let bytes = copied[&from.buffer.raw]
                [from.offset as usize..(from.offset + from.length) as usize]
                .to_vec();
            copied.get_mut(&to.buffer.raw).unwrap()
                [to.offset as usize..(to.offset + to.length) as usize]
                .copy_from_slice(&bytes);
        }
        for span in &destinations {
            assert!(
                copied[&span.buffer.raw] == expected[&span.buffer.raw],
                "{to_pages}"
            );
        }
        // Some bytes moved, and some stayed poisoned.
        let moved = destinations
            .iter()
            .map(|span| {
                expected[&span.buffer.raw]
                    .iter()
                    .filter(|&&b| b != POISONS[0])
                    .count()
            })
            .sum::<usize>();
        assert!(moved > 0 && (moved as u64) < destinations.iter().map(|span| span.length).sum());
    }
}

/// Every case's destinations are bitwise the oracle's after both arms,
/// under both poisons, and its sources unchanged.
#[test]
fn bitwise_gpu() {
    kernel_harness::gpu_test(4 << 30, |s| {
        let module = module(s)?;
        let started = Instant::now();
        let mut cases = 0;
        let mut compared = 0;
        for (index, declared) in DECLARED.into_iter().enumerate() {
            let bench = Bench::new(declared)?;
            let device = Device::new(s, &bench, &module)?;
            for holder in bench.holders() {
                for &to_pages in directions(holder) {
                    for order in ORDERS {
                        if started.elapsed().as_secs() >= BUDGET_SECONDS {
                            return Err(
                                "the bitwise sweep exceeded its budget; coverage is incomplete"
                                    .into(),
                            );
                        }
                        let plans = plans(&bench, holder, order, to_pages)?;
                        let (sources, _) = sides(&bench, to_pages);
                        let seeds = seed(&device, &sources, index as u64 * 100 + cases)?;
                        let bytes = compare(&bench, &device, &plans, to_pages, &seeds)?;
                        println!(
                            "snapshot_copy_bitwise {}",
                            json!({"state": bench.name(), "holder": holder_name(holder),
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
        if cases != 90 {
            return Err(format!("only {cases} of 90 cases ran"));
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

/// Five interleaved rounds of three samples per arm, each sample one run
/// of the arm between CUDA events, its host wall time from before the
/// first event to after the stream drains; `kernel[arm]` names each arm.
fn pair(device: &Device<'_>, plans: &Plans, kernel: [bool; 2]) -> LabResult<Value> {
    let s = device.session;
    let start = s.context.timed_event().map_err(|e| e.to_string())?;
    let end = s.context.timed_event().map_err(|e| e.to_string())?;
    for _ in 0..3 {
        for arm in kernel {
            device.arm(arm, plans, &s.stream)?;
        }
    }
    s.stream.synchronize().map_err(|e| e.to_string())?;
    let mut rounds = Vec::new();
    let mut medians = [[Vec::new(), Vec::new()], [Vec::new(), Vec::new()]];
    for round in 0..5 {
        let mut values = [[Vec::new(), Vec::new()], [Vec::new(), Vec::new()]];
        for sample in 0..3 {
            for step in 0..2 {
                let arm = (round + sample + step) % 2;
                let wall = Instant::now();
                start.record(&s.stream).map_err(|e| e.to_string())?;
                device.arm(kernel[arm], plans, &s.stream)?;
                end.record(&s.stream).map_err(|e| e.to_string())?;
                s.stream.synchronize().map_err(|e| e.to_string())?;
                let host = wall.elapsed().as_secs_f64() * 1e3;
                let event = f64::from(start.elapsed_ms(&end).map_err(|e| e.to_string())?);
                values[0][arm].push(event);
                values[1][arm].push(host);
            }
        }
        for kind in 0..2 {
            for arm in 0..2 {
                medians[kind][arm].push(median(&values[kind][arm]));
            }
        }
        rounds.push(values);
    }
    let summary = |kind: usize| [median(&medians[kind][0]), median(&medians[kind][1])];
    Ok(
        json!({"arms": kernel.map(|k| if k { "kernel" } else { "copies" }),
        "event_ms": summary(0), "host_ms": summary(1), "round_samples_ms": rounds}),
    )
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
/// kernel, kernel A/A and a closing copies A/A, then the comparison again.
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
            let bench = Bench::new(declared)?;
            let device = Device::new(s, &bench, &module)?;
            let plans = plans(&bench, holder, Order::Scattered, to_pages)?;
            let (sources, _) = sides(&bench, to_pages);
            let seeds = seed(&device, &sources, 7000 + index as u64)?;
            compare(&bench, &device, &plans, to_pages, &seeds)?;
            let aa_before = pair(&device, &plans, [false, false])?;
            let ab = pair(&device, &plans, [false, true])?;
            let bb = pair(&device, &plans, [true, true])?;
            let aa_after = pair(&device, &plans, [false, false])?;
            compare(&bench, &device, &plans, to_pages, &seeds)?;
            let record = json!({"state": bench.name(), "holder": holder_name(holder),
                "direction": if to_pages { "capture" } else { "restore" },
                "pages": plans.ids.len(), "copies": plans.copies.len(),
                "snapshot_bytes": plans.table.bytes(), "bitwise": true,
                "aa_before": aa_before, "ab": ab, "bb": bb, "aa_after": aa_after});
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
