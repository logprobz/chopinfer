//! The snapshot copy kernel's actions against the device copies they
//! replace: the same bytes in the same pages and slots, every refusal
//! before any rank submits, and the drain rules of [`copy_prefix`]. The
//! kernel is SM120's prepared op; the fake streams run each launch through
//! the device copies its arguments and uploaded table name
//! ([`launched_copies`]).

use kernel_core::backend::{Backend, Need, Prepared};
use kernels_sm120::backend::Sm120;

use super::super::fake::{Streams, kinds, span};
use super::super::{
    Action, Descriptor, HeaderAction, HeldBuffers, Pages, RankStreams, Table, agreed, copy_prefix,
};
use super::*;
use crate::state::{Anchors, Capture, Extent, Holder, RequestState, synthetic};

const EXTENT: Extent = Extent {
    served: 6,
    bucket: 8,
    verify_rows: 3,
};
/// The pool's buffer, the table's buffer, and the first declared buffer's;
/// buffer `i` is `BUFFERS + i`, its spare `SPARES + i`.
const POOL: u32 = 1;
const TABLE: u32 = 20;
const BUFFERS: u32 = 2;
const SPARES: u32 = 10;
const PAGE_IDS: [u64; 7] = [9, 3, 7, 0, 5, 1, 12];

/// Rank streams over host memory whose launches run the snapshot copy
/// kernel as the device copies it makes; a rank's launch fails when it is
/// `fail`.
struct KernelStreams {
    streams: Streams,
    fail: Option<usize>,
}

impl RankStreams for KernelStreams {
    type Launch = BoundLaunch;
    type Error = &'static str;

    fn ranks(&self) -> usize {
        self.streams.ranks()
    }
    fn upload(
        &self,
        rank: usize,
        to: DeviceBufferSpanV1,
        bytes: &[u8],
    ) -> Result<(), &'static str> {
        self.streams.upload(rank, to, bytes)
    }
    fn launch(&self, rank: usize, launch: BoundLaunch) -> Result<(), &'static str> {
        self.streams.calls.borrow_mut().push((rank, "launch"));
        if self.fail == Some(rank) {
            return Err("launch");
        }
        assert_eq!(launch.launch.symbol, kernels_sm120::snapshot_copy::SYMBOL);
        let arguments = launch.arguments().map_err(|_| "unbound launch")?;
        let copies = launched_copies(&arguments, |table| self.streams.bytes(rank, table))
            .map_err(|refusal| refusal.0)?;
        for (from, to) in copies {
            let bytes = self.streams.bytes(rank, from);
            self.streams.write(rank, to, &bytes);
        }
        Ok(())
    }
    fn copy(
        &self,
        rank: usize,
        from: DeviceBufferSpanV1,
        to: DeviceBufferSpanV1,
    ) -> Result<(), &'static str> {
        self.streams.copy(rank, from, to)
    }
    fn read(
        &self,
        rank: usize,
        from: DeviceBufferSpanV1,
        into: &mut [u8],
    ) -> Result<(), &'static str> {
        self.streams.read(rank, from, into)
    }
    fn drain(&self, rank: usize) -> bool {
        self.streams.drain(rank)
    }
    fn refused(&self, refusal: Refusal) -> &'static str {
        self.streams.refused(refusal)
    }
    fn unconfirmed(&self) -> &'static str {
        self.streams.unconfirmed()
    }
}

/// The synthetic declaration with a 128-byte header, which crosses a page,
/// and with `anchors` separate spares in each captured buffer.
fn state(anchors: u32) -> RequestState {
    let mut state = synthetic();
    state.header_bytes = 128;
    if anchors > 0 {
        for buffer in &mut state.buffers {
            if buffer.capture != Capture::None {
                buffer.anchors = Anchors::Spare {
                    per_step: anchors,
                    pooled: false,
                };
            }
        }
    }
    state
}

/// Every declared buffer and spare of `state`, held whole in its own
/// buffer.
fn held(state: &RequestState) -> HeldBuffers {
    HeldBuffers {
        buffers: state
            .buffers
            .iter()
            .enumerate()
            .map(|(index, buffer)| Some(span(BUFFERS + index as u32, buffer.bytes(EXTENT))))
            .collect(),
        spares: state
            .buffers
            .iter()
            .enumerate()
            .map(|(index, buffer)| {
                (buffer.spare_bytes() > 0)
                    .then(|| span(SPARES + index as u32, buffer.spare_bytes()))
            })
            .collect(),
    }
}

/// The pool, past a 512-byte offset in its buffer.
fn pages(page_bytes: u64) -> Pages<'static> {
    Pages {
        pool: DeviceBufferSpanV1::new(span(POOL, 0).buffer, 512, 16 * page_bytes),
        ids: &PAGE_IDS,
        page_bytes,
    }
}

/// `ranks` ranks' memory: every held buffer and spare a pattern of its own
/// on each rank, the pool and the table filled.
fn memory(state: &RequestState, ranks: usize) -> Streams {
    let streams = Streams::new(ranks);
    let held = held(state);
    let ids = (BUFFERS..)
        .zip(&held.buffers)
        .chain((SPARES..).zip(&held.spares));
    for (id, span) in ids {
        if let Some(span) = span {
            streams.fill(id, span.length as usize, |rank, at| {
                (at * 7 + rank * 3 + id as usize * 11) as u8
            });
        }
    }
    streams.fill(POOL, 512 + 16 * 256, |_, _| 0xee);
    streams.fill(TABLE, 4096, |_, _| 0xcc);
    streams
}

fn op(table: &Table, views: u32, to_pages: bool) -> SnapshotCopy {
    SnapshotCopy {
        direction: if to_pages {
            Direction::ToPages
        } else {
            Direction::FromPages
        },
        views,
        page_bytes: table.page_bytes,
        pages: table.pages() as u32,
    }
}

fn prepared(op: SnapshotCopy, capacity: u32) -> Prepared<SnapshotCopyBuffers<View>> {
    Sm120::new()
        .with_snapshot_copy(kernels_sm120::snapshot_copy::Entry::new(
            kernels_sm120::snapshot_copy::SYMBOL,
        ))
        .snapshot_copy(&op, Need::rows(capacity))
        .expect("the SM120 snapshot copy")
}

/// Rank 0 writes or reads the header; the others move only the pieces.
fn header(rank: usize, to_pages: bool, bytes: &[u8]) -> HeaderAction {
    match (rank, to_pages) {
        (0, true) => HeaderAction::Write(bytes.to_vec()),
        (0, false) => HeaderAction::Read,
        _ => HeaderAction::None,
    }
}

/// Every rank's plan of `holder`'s snapshot through the kernel.
fn kernel_plans(
    state: &RequestState,
    holder: Holder,
    to_pages: bool,
    ranks: usize,
    bytes: &[u8],
) -> Vec<Vec<Action<BoundLaunch>>> {
    let table = Table::new(state, EXTENT, holder).unwrap();
    let pages = pages(table.page_bytes);
    pages.check(&table, 16).unwrap();
    let op = op(&table, 3, to_pages);
    let prepared = prepared(op, table.descriptors.len() as u32);
    let kernel = KernelCopy {
        op,
        prepared: prepared.as_ref(),
        table: span(TABLE, 4096),
    };
    (0..ranks)
        .map(|rank| {
            table
                .kernel_actions(
                    &held(state),
                    &pages,
                    to_pages,
                    header(rank, to_pages, bytes),
                    &kernel,
                )
                .unwrap()
        })
        .collect()
}

/// Every rank's plan of `holder`'s snapshot through device copies.
fn copy_plans(
    state: &RequestState,
    holder: Holder,
    to_pages: bool,
    ranks: usize,
    bytes: &[u8],
) -> Vec<Vec<Action<&'static str>>> {
    let table = Table::new(state, EXTENT, holder).unwrap();
    let pages = pages(table.page_bytes);
    (0..ranks)
        .map(|rank| {
            table
                .actions(
                    &held(state),
                    &pages,
                    to_pages,
                    header(rank, to_pages, bytes),
                )
                .unwrap()
        })
        .collect()
}

fn header_bytes() -> Vec<u8> {
    (0..128).map(|byte| byte as u8 ^ 0x5a).collect()
}

/// One launch moves exactly the device copies the table issues without
/// the kernel, piece by piece and page fragment by page fragment, in both
/// directions, for slots and anchor spares: its uploaded table and bound
/// views name the same spans. The header stays separate actions after it.
#[test]
fn a_launch_moves_exactly_the_tables_device_copies() {
    for (state, holder) in [
        (state(0), Holder::Slot(1)),
        (state(0), Holder::Slot(5)),
        (state(2), Holder::Anchor(1)),
    ] {
        let table = Table::new(&state, EXTENT, holder).unwrap();
        let pages = pages(table.page_bytes);
        assert_eq!((table.header_at(), table.pages()), (1440, 7));
        for to_pages in [true, false] {
            let plan = &kernel_plans(&state, holder, to_pages, 1, &header_bytes())[0];
            let headers = if to_pages {
                ["upload", "upload"]
            } else {
                ["read", "read"]
            };
            assert_eq!(
                kinds(plan),
                ["upload", "launch", headers[0], headers[1]],
                "{holder:?}"
            );
            let (Action::Upload { to, bytes }, Action::Launch(launch)) = (&plan[0], &plan[1])
            else {
                panic!("{plan:?}");
            };
            // The table: seven page IDs and a segment per piece.
            assert_eq!(*to, span(TABLE, bytes.len() as u64));
            assert_eq!(bytes.len(), 7 * 8 + table.descriptors.len() * 32);
            assert_eq!(launch.launch.rows, table.descriptors.len() as u32);
            let arguments = launch.arguments().unwrap();
            let copies = launched_copies(&arguments, |read| {
                assert_eq!(read, *to);
                bytes.clone()
            })
            .unwrap();
            let expected = table
                .copies(0..usize::MAX, &held(&state), &pages, to_pages)
                .unwrap();
            assert_eq!(copies, expected, "{holder:?} {to_pages}");
            // Some pieces cross pages.
            assert!(expected.len() > table.descriptors.len());
        }
    }
}

/// A capture through the kernel writes the pages the device copies write,
/// and a restore through it into another slot of another instance fills
/// that slot with the bytes the device copies restore, rank by rank; the
/// header round-trips beside it and nothing else moves.
#[test]
fn kernel_snapshots_match_the_device_copies_byte_for_byte() {
    for (state, from, to) in [
        (state(0), Holder::Slot(1), Holder::Slot(4)),
        (state(2), Holder::Anchor(1), Holder::Slot(3)),
    ] {
        let header = header_bytes();
        let pool = pages(state.page_bytes()).pool;
        let (copied, launched) = (
            memory(&state, 2),
            KernelStreams {
                streams: memory(&state, 2),
                fail: None,
            },
        );
        assert_eq!(
            copy_prefix(&copied, copy_plans(&state, from, true, 2, &header)),
            Ok(vec![vec![], vec![]])
        );
        assert_eq!(
            copy_prefix(&launched, kernel_plans(&state, from, true, 2, &header)),
            Ok(vec![vec![], vec![]])
        );
        for rank in 0..2 {
            assert_eq!(
                launched.streams.bytes(rank, pool),
                copied.bytes(rank, pool),
                "rank {rank} pages"
            );
        }
        // The pages move to fresh instances, whose held state is zero.
        let (target, launched_target) = (
            memory(&state, 2),
            KernelStreams {
                streams: memory(&state, 2),
                fail: None,
            },
        );
        for rank in 0..2 {
            let pages = copied.bytes(rank, pool);
            for streams in [&target, &launched_target.streams] {
                for span in held(&state).buffers.iter().flatten() {
                    streams.write(rank, *span, &vec![0; span.length as usize]);
                }
                streams.write(rank, pool, &pages);
            }
        }
        let reads = copy_prefix(&target, copy_plans(&state, to, false, 2, &header)).unwrap();
        let launched_reads = copy_prefix(
            &launched_target,
            kernel_plans(&state, to, false, 2, &header),
        )
        .unwrap();
        assert_eq!(agreed(&reads), Some(header.as_slice()));
        assert_eq!(launched_reads, reads);
        let source = Table::new(&state, EXTENT, from).unwrap();
        let restored = Table::new(&state, EXTENT, to).unwrap();
        for rank in 0..2 {
            for span in held(&state).buffers.iter().flatten() {
                assert_eq!(
                    launched_target.streams.bytes(rank, *span),
                    target.bytes(rank, *span),
                    "rank {rank} restored buffer {span:?}"
                );
            }
            // Each restored piece holds the captured piece's bytes, and the
            // uncaptured scratch buffer stays zero.
            for (captured, restored) in source.descriptors.iter().zip(&restored.descriptors) {
                let at = |descriptor: &Descriptor| {
                    let base = if descriptor.spare { SPARES } else { BUFFERS };
                    DeviceBufferSpanV1::new(
                        span(base + descriptor.buffer as u32, 0).buffer,
                        descriptor.offset,
                        descriptor.length,
                    )
                };
                assert_eq!(
                    launched.streams.bytes(rank, at(captured)),
                    launched_target.streams.bytes(rank, at(restored))
                );
            }
            let scratch = held(&state).buffers[3].unwrap();
            assert!(
                launched_target
                    .streams
                    .bytes(rank, scratch)
                    .iter()
                    .all(|&byte| byte == 0)
            );
        }
    }
}

/// Pieces of odd lengths at odd offsets, which no declaration has (its
/// planes are whole vectors), leave later pieces unaligned in the snapshot
/// and end in byte tails; the kernel moves the bytes the device copies
/// move, both ways.
#[test]
fn unaligned_pieces_and_byte_tails_match_the_device_copies() {
    let state = state(0);
    let mut at = 0;
    let descriptors = [
        (0, 3, 301),
        (1, 17, 255),
        (2, 1, 600),
        (0, 400, 413),
        (2, 700, 150),
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
        page_bytes: 256,
    };
    assert_eq!(table.pages(), 7);
    let pages = pages(256);
    pages.check(&table, 16).unwrap();
    let held = held(&state);
    for to_pages in [true, false] {
        let op = op(&table, 3, to_pages);
        let prepared = prepared(op, 5);
        let kernel = kernel_of(op, &prepared, 4096);
        let copied = memory(&state, 1);
        let launched = KernelStreams {
            streams: memory(&state, 1),
            fail: None,
        };
        let plans = vec![
            table
                .actions(&held, &pages, to_pages, HeaderAction::None)
                .unwrap(),
        ];
        copy_prefix(&copied, plans).unwrap();
        let plans = vec![
            table
                .kernel_actions(&held, &pages, to_pages, HeaderAction::None, &kernel)
                .unwrap(),
        ];
        copy_prefix(&launched, plans).unwrap();
        assert_eq!(
            launched.streams.bytes(0, pages.pool),
            copied.bytes(0, pages.pool)
        );
        for span in held.buffers.iter().flatten() {
            assert_eq!(launched.streams.bytes(0, *span), copied.bytes(0, *span));
        }
    }
}

/// Every plan the kernel cannot serve, or that would alias, is refused
/// while the plans are built: one rank's refusal fails the whole set before
/// [`copy_prefix`] sees any of it, so no rank receives anything.
#[test]
fn kernel_plans_refuse_before_any_rank_submits() {
    let state = state(0);
    let table = Table::new(&state, EXTENT, Holder::Slot(1)).unwrap();
    let pages = pages(table.page_bytes);
    let segments = table.descriptors.len() as u32;
    let capture = op(&table, 3, true);
    let full = prepared(capture, segments);
    let short = prepared(capture, segments - 1);
    let restore = prepared(op(&table, 3, false), segments);
    let two_views = prepared(op(&table, 2, true), segments);
    let fewer_pages = SnapshotCopy {
        pages: capture.pages - 1,
        ..capture
    };
    let fewer_pages_op = prepared(fewer_pages, segments);
    let kernel = kernel_of;
    let good = held(&state);
    let mut missing = good.clone();
    missing.buffers[2] = None;
    let mut outside = good.clone();
    outside.buffers[0] = Some(span(BUFFERS, 1024));
    // The history inside the window ring's buffer, over its pieces.
    let mut aliased = good.clone();
    aliased.buffers[2] = good.buffers[0];
    // A held buffer inside the pool, over the snapshot's pages.
    let mut on_pages = good.clone();
    on_pages.buffers[2] = Some(DeviceBufferSpanV1::new(span(POOL, 0).buffer, 512, 4096));
    let cases: Vec<(&'static str, HeldBuffers, KernelCopy)> = vec![
        (
            "the snapshot copy kernel is prepared for another snapshot",
            good.clone(),
            kernel(op(&table, 3, false), &restore, 4096),
        ),
        (
            "snapshot copy segments exceed the prepared capacity",
            good.clone(),
            kernel(capture, &short, 4096),
        ),
        (
            "the snapshot copy kernel is prepared for another view count",
            good.clone(),
            kernel(op(&table, 2, true), &two_views, 4096),
        ),
        (
            "the snapshot copy kernel is prepared for another snapshot",
            good.clone(),
            kernel(fewer_pages, &fewer_pages_op, 4096),
        ),
        (
            "a snapshot copy table exceeds its buffer",
            good.clone(),
            kernel(capture, &full, 7 * 8 + 32),
        ),
        (
            "a snapshot piece's buffer is not held",
            missing,
            kernel(capture, &full, 4096),
        ),
        (
            "a snapshot piece lies outside its buffer",
            outside,
            kernel(capture, &full, 4096),
        ),
        (
            "a snapshot copy's pieces, pages and table overlap",
            aliased,
            kernel(capture, &full, 4096),
        ),
        (
            "a snapshot copy's pieces, pages and table overlap",
            on_pages,
            kernel(capture, &full, 4096),
        ),
    ];
    for (what, held, kernel) in cases {
        // Rank 0's plan is good; rank 1's is refused, so the plans never
        // reach the streams.
        let plans = (0..2)
            .map(|rank| {
                let (held, kernel) = if rank == 0 {
                    (good.clone(), kernel_of(capture, &full, 4096))
                } else {
                    (held.clone(), kernel_ref(&kernel))
                };
                table.kernel_actions(&held, &pages, true, HeaderAction::None, &kernel)
            })
            .collect::<Result<Vec<_>, _>>();
        assert_eq!(plans.unwrap_err(), Refusal(what));
    }
    // Nor may the table start off an eight-byte boundary, which the kernel
    // reads as u64 words.
    for offset in [1, 4, 12] {
        let unaligned = KernelCopy {
            op: capture,
            prepared: full.as_ref(),
            table: DeviceBufferSpanV1::new(span(TABLE, 0).buffer, offset, 4096),
        };
        assert_eq!(
            table
                .kernel_actions(&good, &pages, true, HeaderAction::None, &unaligned)
                .unwrap_err(),
            Refusal("a snapshot copy table starts at an eight-byte boundary")
        );
    }
    let aligned = KernelCopy {
        op: capture,
        prepared: full.as_ref(),
        table: DeviceBufferSpanV1::new(span(TABLE, 0).buffer, 8, 4096),
    };
    assert!(
        table
            .kernel_actions(&good, &pages, true, HeaderAction::None, &aligned)
            .is_ok()
    );
    // The table may not lie on the pages either, nor the pages differ in
    // size from the table's.
    let on_pool = KernelCopy {
        op: capture,
        prepared: full.as_ref(),
        table: DeviceBufferSpanV1::new(span(POOL, 0).buffer, 512 + 3 * 256, 4096),
    };
    assert_eq!(
        table
            .kernel_actions(&good, &pages, true, HeaderAction::None, &on_pool)
            .unwrap_err(),
        Refusal("a snapshot copy's pieces, pages and table overlap")
    );
    let other = Pages {
        page_bytes: 128,
        ..pages
    };
    assert!(
        table
            .kernel_actions(
                &good,
                &other,
                true,
                HeaderAction::None,
                &kernel_of(capture, &full, 4096)
            )
            .is_err()
    );
}

fn kernel_of(
    op: SnapshotCopy,
    prepared: &Prepared<SnapshotCopyBuffers<View>>,
    table: u64,
) -> KernelCopy<'_> {
    KernelCopy {
        op,
        prepared: prepared.as_ref(),
        table: span(TABLE, table),
    }
}

fn kernel_ref<'a>(kernel: &KernelCopy<'a>) -> KernelCopy<'a> {
    KernelCopy {
        op: kernel.op,
        prepared: kernel.prepared,
        table: kernel.table,
    }
}

/// A table naming more than eight held buffers is more than the kernel
/// binds; the caller keeps the device copies for it.
#[test]
fn more_than_eight_views_are_refused() {
    let descriptors = (0..9)
        .map(|buffer| Descriptor {
            buffer,
            spare: false,
            offset: 0,
            length: 16,
            at: 16 * buffer as u64,
        })
        .collect();
    let table = Table {
        descriptors,
        header_bytes: 0,
        page_bytes: 256,
    };
    let held = HeldBuffers {
        buffers: (0..9).map(|buffer| Some(span(30 + buffer, 16))).collect(),
        spares: Vec::new(),
    };
    assert_eq!(
        table.segments(&held).unwrap_err(),
        Refusal("a snapshot copy kernel binds at most eight views")
    );
    let held = HeldBuffers {
        buffers: held.buffers[..8].to_vec(),
        spares: Vec::new(),
    };
    assert!(table.segments(&held).is_err());
}

/// A failed launch drains every rank that received work, a stuck rank's
/// drain makes the copy unconfirmed, and both leave the error the host
/// poisons on; a good plan drains every rank once.
#[test]
fn kernel_copies_drain_every_rank_that_received_work() {
    let state = state(0);
    let header = header_bytes();
    let failing = KernelStreams {
        streams: memory(&state, 2),
        fail: Some(1),
    };
    assert_eq!(
        copy_prefix(
            &failing,
            kernel_plans(&state, Holder::Slot(1), true, 2, &header)
        ),
        Err("launch")
    );
    assert_eq!(
        *failing.streams.calls.borrow(),
        [
            (0, "upload"),
            (0, "launch"),
            (0, "upload"),
            (0, "upload"),
            (1, "upload"),
            (1, "launch"),
            (0, "drain"),
            (1, "drain"),
        ]
    );
    let stuck = KernelStreams {
        streams: Streams {
            stuck: Some(1),
            ..memory(&state, 2)
        },
        fail: None,
    };
    assert_eq!(
        copy_prefix(
            &stuck,
            kernel_plans(&state, Holder::Slot(1), false, 2, &header)
        ),
        Err("unconfirmed")
    );
    let calls = stuck.streams.calls.borrow();
    assert_eq!(calls.iter().filter(|call| call.1 == "drain").count(), 2);
    let good = KernelStreams {
        streams: memory(&state, 2),
        fail: None,
    };
    copy_prefix(
        &good,
        kernel_plans(&state, Holder::Slot(1), true, 2, &header),
    )
    .unwrap();
    assert_eq!(
        good.streams.calls.borrow().last().copied(),
        Some((1, "drain"))
    );
}
