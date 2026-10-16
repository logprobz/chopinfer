//! Prefix-cache snapshots in serving-pool pages.
//!
//! A declaration ([`crate::state::RequestState`]) yields one [`Table`] of
//! descriptors per holder: the pieces a snapshot copies, in snapshot order,
//! each at its byte in the snapshot, then the header. The pieces lie end to
//! end across the snapshot's pages, so a piece may continue on the next
//! page: snapshot byte `c` lives at `pages[c / page_bytes] * page_bytes +
//! c % page_bytes` of the serving pool.
//!
//! A bundle moves the pieces with device copies split at page boundaries
//! ([`Table::copies`]), or covers some of them with a copy kernel of its own
//! that mirrors the table. [`copy_prefix`] submits every rank's actions,
//! built and checked before any rank starts, and drains every rank that
//! received work.

use std::collections::BTreeSet;
use std::ops::Range;

use kernel_core::contract::Refusal;
use runtime_bundle_abi::{
    BundleAbiError, BundleStreamHandleV1, DeviceBufferSpanV1, HostServicesV1,
};

use crate::state::{Extent, Holder, Invalid, RequestState};

/// One piece of a snapshot: `length` bytes at `offset` of a declared
/// buffer's working planes (or of its separate spare buffer), at byte `at`
/// of the snapshot.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Descriptor {
    /// The buffer's index in the declaration.
    pub buffer: usize,
    pub spare: bool,
    pub offset: u64,
    pub length: u64,
    pub at: u64,
}

/// What one holder's snapshot copies in an instance of one extent: the
/// pieces in snapshot order, then `header_bytes` the host writes.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Table {
    pub descriptors: Vec<Descriptor>,
    pub header_bytes: u64,
    /// Bytes of one serving page of the declaration.
    pub page_bytes: u64,
}

impl Table {
    /// The table of `holder`'s snapshot, after checking the declaration and
    /// that an instance of `extent` can hold it.
    pub fn new(state: &RequestState, extent: Extent, holder: Holder) -> Result<Self, Invalid> {
        Self::with_entries(state, extent, holder, &[])
    }

    /// Build addresses from physical entries instead of modifying logical-slot
    /// offsets after construction. Unmapped buffers retain their declared entry.
    pub fn with_entries(
        state: &RequestState,
        extent: Extent,
        holder: Holder,
        mapped: &[(usize, u64)],
    ) -> Result<Self, Invalid> {
        state.validate()?;
        state.validate_extent(extent)?;
        let mut at = 0;
        let descriptors = state
            .pieces_with_entries(extent, holder, mapped)?
            .into_iter()
            .map(|piece| {
                let descriptor = Descriptor {
                    buffer: piece.buffer,
                    spare: piece.spare,
                    offset: piece.offset,
                    length: piece.length,
                    at,
                };
                at += piece.length;
                descriptor
            })
            .collect();
        Ok(Self {
            descriptors,
            header_bytes: state.header_bytes,
            page_bytes: state.page_bytes(),
        })
    }

    /// The snapshot byte the header starts at, right after the pieces.
    pub fn header_at(&self) -> u64 {
        self.descriptors
            .last()
            .map_or(0, |last| last.at + last.length)
    }

    /// Bytes of the snapshot, its header included.
    pub fn bytes(&self) -> u64 {
        self.header_at() + self.header_bytes
    }

    /// Serving pages the snapshot takes.
    pub fn pages(&self) -> u64 {
        self.bytes().div_ceil(self.page_bytes)
    }

    /// The device copies that move the pieces of the declared buffers
    /// `buffers` between where `held` holds them and `pages`, as `(source,
    /// destination)` in snapshot order: into the pages when `to_pages`,
    /// otherwise back.
    pub fn copies(
        &self,
        buffers: Range<usize>,
        held: &HeldBuffers,
        pages: &Pages,
        to_pages: bool,
    ) -> Result<Vec<(DeviceBufferSpanV1, DeviceBufferSpanV1)>, Refusal> {
        let mut copies = Vec::new();
        for descriptor in self
            .descriptors
            .iter()
            .filter(|descriptor| buffers.contains(&descriptor.buffer))
        {
            let span = held
                .span(descriptor.buffer, descriptor.spare)
                .ok_or(Refusal("a snapshot piece's buffer is not held"))?;
            if descriptor
                .offset
                .checked_add(descriptor.length)
                .is_none_or(|end| end > span.length)
            {
                return Err(Refusal("a snapshot piece lies outside its buffer"));
            }
            let piece = DeviceBufferSpanV1::new(
                span.buffer,
                span.offset + descriptor.offset,
                descriptor.length,
            );
            copies.extend(pages.copies(piece, descriptor.at, to_pages)?);
        }
        Ok(copies)
    }

    /// One rank's actions with no copy kernel: every piece as device copies,
    /// then `header`'s.
    pub fn actions<L>(
        &self,
        held: &HeldBuffers,
        pages: &Pages,
        to_pages: bool,
        header: HeaderAction,
    ) -> Result<Vec<Action<L>>, Refusal> {
        let mut actions = self
            .copies(0..usize::MAX, held, pages, to_pages)?
            .into_iter()
            .map(|(from, to)| Action::Copy { from, to })
            .collect::<Vec<_>>();
        actions.extend(pages.header_actions(self, header)?);
        Ok(actions)
    }
}

/// Where one rank holds each declared buffer's working planes, by
/// declaration index, and each separate anchor spare buffer. A buffer the
/// snapshot does not copy may be `None`.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct HeldBuffers {
    pub buffers: Vec<Option<DeviceBufferSpanV1>>,
    pub spares: Vec<Option<DeviceBufferSpanV1>>,
}

impl HeldBuffers {
    fn span(&self, buffer: usize, spare: bool) -> Option<DeviceBufferSpanV1> {
        let spans = if spare { &self.spares } else { &self.buffers };
        spans.get(buffer).copied().flatten()
    }
}

/// What a rank does with the header after the pieces.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum HeaderAction {
    /// Nothing: the snapshot has no header, or another rank holds it.
    None,
    /// Write these bytes (a capture).
    Write(Vec<u8>),
    /// Read it back (a restore).
    Read,
}

/// One rank's view of a snapshot's pages: the serving pool, addressed by
/// page ID, and the snapshot's page IDs.
#[derive(Clone, Copy, Debug)]
pub struct Pages<'a> {
    pub pool: DeviceBufferSpanV1,
    pub ids: &'a [u64],
    pub page_bytes: u64,
}

impl Pages<'_> {
    /// Checks that the pages hold `table`'s snapshot in a pool of
    /// `capacity` pages: the declared page size in whole 16-byte vectors,
    /// as many pages as the snapshot takes, each a distinct page of the
    /// pool.
    pub fn check(&self, table: &Table, capacity: u64) -> Result<(), Refusal> {
        if self.page_bytes != table.page_bytes
            || self.page_bytes == 0
            || !self.page_bytes.is_multiple_of(16)
            || capacity
                .checked_mul(self.page_bytes)
                .is_none_or(|bytes| bytes > self.pool.length)
        {
            return Err(Refusal("snapshot pages do not match the pool"));
        }
        if self.ids.len() as u64 != table.pages()
            || self.ids.iter().any(|&page| page >= capacity)
            || self.ids.iter().collect::<BTreeSet<_>>().len() != self.ids.len()
        {
            return Err(Refusal("snapshot pages are not the snapshot's"));
        }
        Ok(())
    }

    /// The pool spans that hold snapshot bytes `at..at + length`, one per
    /// page they touch.
    pub fn spans(&self, at: u64, length: u64) -> Result<Vec<DeviceBufferSpanV1>, Refusal> {
        if self.page_bytes == 0
            || at
                .checked_add(length)
                .is_none_or(|end| end > self.ids.len() as u64 * self.page_bytes)
            || self.pool.offset.checked_add(self.pool.length).is_none()
        {
            return Err(Refusal("snapshot range outside its pages"));
        }
        let mut spans = Vec::new();
        let mut done = 0;
        while done < length {
            let at = at + done;
            let bytes = (self.page_bytes - at % self.page_bytes).min(length - done);
            let physical = self.ids[(at / self.page_bytes) as usize]
                .checked_mul(self.page_bytes)
                .and_then(|base| base.checked_add(at % self.page_bytes))
                .filter(|base| {
                    base.checked_add(bytes)
                        .is_some_and(|end| end <= self.pool.length)
                })
                .ok_or(Refusal("snapshot page outside the pool"))?;
            spans.push(DeviceBufferSpanV1::new(
                self.pool.buffer,
                self.pool.offset + physical,
                bytes,
            ));
            done += bytes;
        }
        Ok(spans)
    }

    /// The device copies that move `held` to snapshot bytes `at..` (when
    /// `to_pages`) or back, as `(source, destination)`.
    pub fn copies(
        &self,
        held: DeviceBufferSpanV1,
        at: u64,
        to_pages: bool,
    ) -> Result<Vec<(DeviceBufferSpanV1, DeviceBufferSpanV1)>, Refusal> {
        if held.offset.checked_add(held.length).is_none() {
            return Err(Refusal("snapshot piece range overflows"));
        }
        let mut done = 0;
        Ok(self
            .spans(at, held.length)?
            .into_iter()
            .map(|paged| {
                let part = DeviceBufferSpanV1::new(held.buffer, held.offset + done, paged.length);
                done += paged.length;
                if to_pages {
                    (part, paged)
                } else {
                    (paged, part)
                }
            })
            .collect())
    }

    /// The actions that write `table`'s header straight into its pages, or
    /// read it back, one per page it touches.
    pub fn header_actions<L>(
        &self,
        table: &Table,
        header: HeaderAction,
    ) -> Result<Vec<Action<L>>, Refusal> {
        let spans = || self.spans(table.header_at(), table.header_bytes);
        Ok(match header {
            HeaderAction::None => Vec::new(),
            HeaderAction::Write(bytes) => {
                if bytes.len() as u64 != table.header_bytes {
                    return Err(Refusal("the header is not the declared size"));
                }
                let mut rest = bytes.as_slice();
                spans()?
                    .into_iter()
                    .map(|to| {
                        let (part, tail) = rest.split_at(to.length as usize);
                        rest = tail;
                        Action::Upload {
                            to,
                            bytes: part.to_vec(),
                        }
                    })
                    .collect()
            }
            HeaderAction::Read => spans()?
                .into_iter()
                .map(|from| Action::Read { from })
                .collect(),
        })
    }
}

/// One host action of a snapshot copy on one rank's stream.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Action<L> {
    /// Write host bytes to the device: the page table, a header.
    Upload {
        to: DeviceBufferSpanV1,
        bytes: Vec<u8>,
    },
    /// A copy kernel the bundle prepared.
    Launch(L),
    Copy {
        from: DeviceBufferSpanV1,
        to: DeviceBufferSpanV1,
    },
    /// Read device bytes back: a restored header.
    Read { from: DeviceBufferSpanV1 },
}

/// Bytes one [`Action::Read`] may return.
pub const MAX_READ_BYTES: u64 = 4096;

/// The rank streams a snapshot copy runs on.
pub trait RankStreams {
    type Launch;
    type Error;
    fn ranks(&self) -> usize;
    fn upload(&self, rank: usize, to: DeviceBufferSpanV1, bytes: &[u8]) -> Result<(), Self::Error>;
    fn launch(&self, rank: usize, launch: Self::Launch) -> Result<(), Self::Error>;
    fn copy(
        &self,
        rank: usize,
        from: DeviceBufferSpanV1,
        to: DeviceBufferSpanV1,
    ) -> Result<(), Self::Error>;
    fn read(
        &self,
        rank: usize,
        from: DeviceBufferSpanV1,
        into: &mut [u8],
    ) -> Result<(), Self::Error>;
    /// Waits for the rank's work; false when that cannot be confirmed.
    fn drain(&self, rank: usize) -> bool;
    /// The error of a plan refused before any rank starts.
    fn refused(&self, refusal: Refusal) -> Self::Error;
    /// The error of a copy whose completion a drain did not confirm. Device
    /// work may still access the pages, so the host must not reuse them.
    fn unconfirmed(&self) -> Self::Error;
}

/// Borrowed host streams with model-owned launch and error policies.
/// The stream slice is in rank order and contains exactly the served ranks.
pub struct HostRankStreams<'a, L, E> {
    pub services: &'a HostServicesV1,
    /// Exactly one stream per topology rank, in rank order.
    pub streams: &'a [BundleStreamHandleV1],
    pub launch: &'a dyn Fn(usize, BundleStreamHandleV1, L) -> Result<(), E>,
    pub host_error: &'a dyn Fn(BundleAbiError) -> E,
    pub refusal: &'a dyn Fn(Refusal) -> E,
    pub unconfirmed: &'a dyn Fn() -> E,
}

impl<L, E> RankStreams for HostRankStreams<'_, L, E> {
    type Launch = L;
    type Error = E;

    fn ranks(&self) -> usize {
        self.streams.len()
    }

    fn upload(&self, rank: usize, to: DeviceBufferSpanV1, bytes: &[u8]) -> Result<(), E> {
        crate::resources::upload_bytes(self.services, self.streams[rank], to, bytes)
            .map_err(self.host_error)
    }

    fn launch(&self, rank: usize, launch: L) -> Result<(), E> {
        (self.launch)(rank, self.streams[rank], launch)
    }

    fn copy(&self, rank: usize, from: DeviceBufferSpanV1, to: DeviceBufferSpanV1) -> Result<(), E> {
        self.services
            .copy_device_to_device(self.streams[rank], to, from)
            .map_err(self.host_error)
    }

    fn read(&self, rank: usize, from: DeviceBufferSpanV1, into: &mut [u8]) -> Result<(), E> {
        self.services
            .copy_device_to_host(self.streams[rank], into, from)
            .map_err(self.host_error)
    }

    fn drain(&self, rank: usize) -> bool {
        self.services.synchronize_stream(self.streams[rank]).is_ok()
    }

    fn refused(&self, refusal: Refusal) -> E {
        (self.refusal)(refusal)
    }
    fn unconfirmed(&self) -> E {
        (self.unconfirmed)()
    }
}

/// Submits each rank's actions in rank order and returns once every rank
/// that received work is drained, with each rank's reads end to end.
///
/// The host frees or reuses the pages as soon as this fails, so the plans
/// are complete before any rank submits: one per rank, every read bounded.
/// A failed drain takes precedence over any other error.
pub fn copy_prefix<S: RankStreams>(
    streams: &S,
    plans: Vec<Vec<Action<S::Launch>>>,
) -> Result<Vec<Vec<u8>>, S::Error> {
    if plans.len() != streams.ranks() {
        return Err(streams.refused(Refusal("a snapshot copy needs a plan per rank")));
    }
    if plans
        .iter()
        .flatten()
        .any(|action| matches!(action, Action::Read { from } if from.length > MAX_READ_BYTES))
    {
        return Err(streams.refused(Refusal("a snapshot read is too long")));
    }
    let mut reads = vec![Vec::new(); plans.len()];
    submit_and_drain(
        plans,
        |rank, actions| {
            for action in actions {
                match action {
                    Action::Upload { to, bytes } => streams.upload(rank, to, &bytes)?,
                    Action::Launch(launch) => streams.launch(rank, launch)?,
                    Action::Copy { from, to } => streams.copy(rank, from, to)?,
                    Action::Read { from } => {
                        let mut bytes = vec![0; from.length as usize];
                        streams.read(rank, from, &mut bytes)?;
                        reads[rank].extend(bytes);
                    }
                }
            }
            Ok(())
        },
        |rank| streams.drain(rank),
        streams.unconfirmed(),
    )?;
    Ok(reads)
}

#[cfg(test)]
#[path = "snapshot/host_tests.rs"]
mod host_tests;

/// The bytes every rank that read anything read, when they agree.
pub fn agreed(reads: &[Vec<u8>]) -> Option<&[u8]> {
    let mut read = reads.iter().filter(|bytes| !bytes.is_empty());
    let first = read.next()?;
    read.all(|bytes| bytes == first).then_some(first.as_slice())
}

/// Submit prevalidated copies in rank order, then drain every rank that
/// received work, including a rank whose submission failed partway through.
/// A failed drain takes precedence: the host must not reuse pages that device
/// work may still access. Model layouts and copy launches stay with the caller.
pub fn submit_and_drain<T, E>(
    copies: impl IntoIterator<Item = T>,
    mut submit: impl FnMut(usize, T) -> Result<(), E>,
    mut drain: impl FnMut(usize) -> bool,
    unconfirmed: E,
) -> Result<(), E> {
    let mut submitted = 0;
    let mut result = Ok(());
    for (rank, copy) in copies.into_iter().enumerate() {
        submitted = rank + 1;
        result = submit(rank, copy);
        if result.is_err() {
            break;
        }
    }
    let mut drained = true;
    for rank in 0..submitted {
        drained &= drain(rank);
    }
    if drained { result } else { Err(unconfirmed) }
}

#[cfg(test)]
pub(crate) mod fake {
    //! Rank streams over host memory, for the shared mechanics' tests.

    use std::cell::RefCell;
    use std::collections::BTreeMap;

    use runtime_bundle_abi::{BundleDeviceBufferHandleV1, DeviceBufferSpanV1};

    use super::{Action, RankStreams, Refusal};

    /// A whole device buffer `id` of `length` bytes.
    pub fn span(id: u32, length: u64) -> DeviceBufferSpanV1 {
        DeviceBufferSpanV1::new(BundleDeviceBufferHandleV1::from_parts(1, id), 0, length)
    }

    /// Every rank's buffers by handle, and the calls each rank received.
    #[derive(Default)]
    pub struct Streams {
        pub memory: Vec<RefCell<BTreeMap<u64, Vec<u8>>>>,
        pub calls: RefCell<Vec<(usize, &'static str)>>,
        /// A rank whose drain fails.
        pub stuck: Option<usize>,
    }

    impl Streams {
        pub fn new(ranks: usize) -> Self {
            Self {
                memory: (0..ranks).map(|_| RefCell::default()).collect(),
                ..Self::default()
            }
        }

        /// Gives every rank buffer `id`, filled by `fill(rank, byte)`.
        pub fn fill(&self, id: u32, length: usize, fill: impl Fn(usize, usize) -> u8) {
            for (rank, memory) in self.memory.iter().enumerate() {
                let bytes = (0..length).map(|at| fill(rank, at)).collect();
                memory.borrow_mut().insert(span(id, 0).buffer.raw, bytes);
            }
        }

        pub fn bytes(&self, rank: usize, span: DeviceBufferSpanV1) -> Vec<u8> {
            let memory = self.memory[rank].borrow();
            let at = span.offset as usize;
            memory[&span.buffer.raw][at..at + span.length as usize].to_vec()
        }

        pub fn write(&self, rank: usize, span: DeviceBufferSpanV1, bytes: &[u8]) {
            let mut memory = self.memory[rank].borrow_mut();
            let at = span.offset as usize;
            memory.get_mut(&span.buffer.raw).unwrap()[at..at + bytes.len()].copy_from_slice(bytes);
        }
    }

    impl RankStreams for Streams {
        type Launch = &'static str;
        type Error = &'static str;

        fn ranks(&self) -> usize {
            self.memory.len()
        }

        fn upload(
            &self,
            rank: usize,
            to: DeviceBufferSpanV1,
            bytes: &[u8],
        ) -> Result<(), &'static str> {
            self.calls.borrow_mut().push((rank, "upload"));
            self.write(rank, to, bytes);
            Ok(())
        }

        fn launch(&self, rank: usize, launch: &'static str) -> Result<(), &'static str> {
            self.calls.borrow_mut().push((rank, launch));
            if launch == "fail" {
                Err("launch")
            } else {
                Ok(())
            }
        }

        fn copy(
            &self,
            rank: usize,
            from: DeviceBufferSpanV1,
            to: DeviceBufferSpanV1,
        ) -> Result<(), &'static str> {
            self.calls.borrow_mut().push((rank, "copy"));
            assert_eq!(from.length, to.length);
            let bytes = self.bytes(rank, from);
            self.write(rank, to, &bytes);
            Ok(())
        }

        fn read(
            &self,
            rank: usize,
            from: DeviceBufferSpanV1,
            into: &mut [u8],
        ) -> Result<(), &'static str> {
            self.calls.borrow_mut().push((rank, "read"));
            into.copy_from_slice(&self.bytes(rank, from));
            Ok(())
        }

        fn drain(&self, rank: usize) -> bool {
            self.calls.borrow_mut().push((rank, "drain"));
            self.stuck != Some(rank)
        }

        fn refused(&self, refusal: Refusal) -> &'static str {
            refusal.0
        }

        fn unconfirmed(&self) -> &'static str {
            "unconfirmed"
        }
    }

    /// The actions' kinds, for comparing plans.
    pub fn kinds<L>(actions: &[Action<L>]) -> Vec<&'static str> {
        actions
            .iter()
            .map(|action| match action {
                Action::Upload { .. } => "upload",
                Action::Launch(_) => "launch",
                Action::Copy { .. } => "copy",
                Action::Read { .. } => "read",
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::fake::{Streams, kinds, span};
    use super::*;
    use crate::state::{Capture, synthetic};
    use std::cell::RefCell;

    const EXTENT: Extent = Extent {
        served: 6,
        bucket: 8,
        verify_rows: 3,
    };

    #[test]
    fn mapped_entries_address_checkpoint_pool_planes_at_construction() {
        let state = synthetic();
        let logical = Table::new(&state, EXTENT, Holder::Slot(1)).unwrap();
        let mapped = Table::with_entries(&state, EXTENT, Holder::Slot(1), &[(1, 8)]).unwrap();
        let entries = state.buffers[1].entries(EXTENT);
        let bytes = state.buffers[1].planes[0];
        for (before, after) in logical.descriptors.iter().zip(&mapped.descriptors) {
            assert_eq!(before.at, after.at);
            assert_eq!(before.length, after.length);
            if after.buffer == 1 {
                assert_eq!(after.offset, before.offset + 7 * bytes);
                assert_eq!((after.offset / bytes) % entries, 8);
            } else {
                assert_eq!(before, after);
            }
        }
        for overrides in [
            vec![(1, entries)],
            vec![(1, 8), (1, 9)],
            vec![(9, 0)],
            vec![(3, 0)],
        ] {
            assert!(Table::with_entries(&state, EXTENT, Holder::Slot(1), &overrides).is_err());
        }
        assert!(Table::with_entries(&state, EXTENT, Holder::Anchor(0), &[(1, 8)]).is_err());
    }

    #[test]
    fn descriptors_follow_the_declared_pieces_end_to_end() {
        let state = synthetic();
        let table = Table::new(&state, EXTENT, Holder::Slot(5)).unwrap();
        let pieces = state.pieces(EXTENT, Holder::Slot(5)).unwrap();
        assert_eq!(table.descriptors.len(), pieces.len());
        // Three ring layers of K and V, two slot layers, one history.
        assert_eq!(pieces.len(), 3 * 2 + 2 + 1);
        let mut at = 0;
        for (descriptor, piece) in table.descriptors.iter().zip(&pieces) {
            assert_eq!(
                (
                    descriptor.buffer,
                    descriptor.spare,
                    descriptor.offset,
                    descriptor.length
                ),
                (piece.buffer, piece.spare, piece.offset, piece.length)
            );
            assert_eq!(descriptor.at, at);
            at += piece.length;
        }
        // Slot 5's K in the second ring layer: past the first layer's eight
        // K and V entries, then five K entries.
        assert_eq!(table.descriptors[2].offset, 8 * 384 + 5 * 256);
        // The slot buffer's second layer: six slots and two items' three
        // checkpoints in each layer.
        assert_eq!(table.descriptors[7].offset, (6 + 6 + 5) * 48);
        assert_eq!(table.header_at(), 3 * 384 + 2 * 48 + 192);
        assert_eq!(table.header_at() + 64, state.snapshot_bytes());
        assert_eq!((table.page_bytes, table.pages()), (256, 6));
        assert_eq!(table.pages(), state.snapshot_pages());
    }

    #[test]
    fn inconsistent_declarations_and_extents_are_refused() {
        let mut state = synthetic();
        state.buffers[2].capture = Capture::Full;
        assert_eq!(
            Table::new(&state, EXTENT, Holder::Slot(0)).unwrap_err(),
            Invalid {
                name: "history",
                rule: "the capture does not fit the scope"
            }
        );
        let state = synthetic();
        let extent = |served, bucket, verify_rows| Extent {
            served,
            bucket,
            verify_rows,
        };
        // More served slots than the bucket, and a verify item longer than
        // the ring.
        assert!(Table::new(&state, extent(9, 8, 0), Holder::Slot(0)).is_err());
        assert_eq!(
            Table::new(&state, extent(4, 4, 9), Holder::Slot(0))
                .unwrap_err()
                .rule,
            "a verify item's rows must fit the positions"
        );
        assert!(Table::new(&state, EXTENT, Holder::Slot(6)).is_err());
        assert!(Table::new(&state, EXTENT, Holder::Anchor(0)).is_err());
    }

    #[test]
    fn pages_must_be_the_snapshots_distinct_pool_pages() {
        let table = Table::new(&synthetic(), EXTENT, Holder::Slot(0)).unwrap();
        let ids = [9, 3, 7, 0, 5, 1];
        let pages = |ids, page_bytes| Pages {
            pool: span(1, 16 * 256),
            ids,
            page_bytes,
        };
        assert_eq!(pages(&ids, 256).check(&table, 16), Ok(()));
        for (ids, page_bytes, capacity) in [
            (&ids[..5], 256, 16),
            (&[9, 3, 7, 0, 5, 9][..], 256, 16),
            (&[9, 3, 7, 0, 5, 16][..], 256, 16),
            (&ids[..], 256, 17),
            (&ids[..], 128, 16),
        ] {
            assert!(pages(ids, page_bytes).check(&table, capacity).is_err());
        }
    }

    /// Copies split at page boundaries and land at the page each snapshot
    /// byte maps to, and a header that crosses a page is written and read
    /// in two parts.
    #[test]
    fn pieces_and_header_cross_noncontiguous_pages() {
        let mut state = synthetic();
        state.header_bytes = 128;
        let table = Table::new(&state, EXTENT, Holder::Slot(1)).unwrap();
        assert_eq!((table.header_at(), table.pages()), (1440, 7));
        let pages = Pages {
            pool: DeviceBufferSpanV1::new(span(1, 0).buffer, 512, 16 * 256),
            ids: &[9, 3, 7, 0, 5, 1, 12],
            page_bytes: 256,
        };
        let header = (0..128).collect::<Vec<u8>>();
        let actions = pages
            .header_actions::<()>(&table, HeaderAction::Write(header.clone()))
            .unwrap();
        let Action::Upload { to, bytes } = &actions[1] else {
            panic!("{actions:?}");
        };
        // 1440..1536 on page 1, then 1536..1568 at the start of page 12.
        assert_eq!((to.offset, to.length), (512 + 12 * 256, 32));
        assert_eq!(bytes.as_slice(), &header[96..]);
        assert_eq!(kinds(&actions), ["upload", "upload"]);
        let back = pages
            .header_actions::<()>(&table, HeaderAction::Read)
            .unwrap();
        assert_eq!(kinds(&back), ["read", "read"]);
        assert!(
            pages
                .header_actions::<()>(&table, HeaderAction::Write(vec![0; 64]))
                .is_err()
        );
        // The first ring piece, 256 bytes at snapshot byte 0, fills page 9.
        let held = HeldBuffers {
            buffers: vec![
                Some(span(2, 1 << 20)),
                Some(span(3, 1 << 20)),
                Some(span(4, 1 << 20)),
            ],
            spares: vec![],
        };
        let copies = table.copies(0..1, &held, &pages, true).unwrap();
        assert_eq!(copies[0].1.offset, 512 + 9 * 256);
        // The V piece of 128 bytes continues on page 3.
        assert_eq!(
            (copies[1].1.offset, copies[1].1.length),
            (512 + 3 * 256, 128)
        );
        let restore = table.copies(0..1, &held, &pages, false).unwrap();
        assert_eq!(
            restore,
            copies
                .iter()
                .map(|&(from, to)| (to, from))
                .collect::<Vec<_>>()
        );
        // A piece outside its buffer, or a buffer not held.
        let short = HeldBuffers {
            buffers: vec![Some(span(2, 1024))],
            spares: vec![],
        };
        assert!(table.copies(0..1, &short, &pages, true).is_err());
        assert!(table.copies(1..2, &short, &pages, true).is_err());
    }

    /// A capture from slot 1 of one instance and a restore into slot 4 of
    /// another move every declared piece and the header, rank by rank.
    #[test]
    fn a_snapshot_restores_into_another_slot_byte_for_byte() {
        let state = synthetic();
        let (source, target) = (Streams::new(2), Streams::new(2));
        let sizes = state
            .buffers
            .iter()
            .map(|buffer| buffer.bytes(EXTENT) as usize)
            .collect::<Vec<_>>();
        for (index, &bytes) in sizes.iter().enumerate() {
            let id = index as u32 + 2;
            source.fill(id, bytes, |rank, at| (at * 7 + rank * 3 + index) as u8);
            target.fill(id, bytes, |_, _| 0);
        }
        for streams in [&source, &target] {
            streams.fill(1, 16 * 256, |_, _| 0xee);
        }
        let held = HeldBuffers {
            buffers: sizes
                .iter()
                .enumerate()
                .map(|(index, &bytes)| Some(span(index as u32 + 2, bytes as u64)))
                .collect(),
            spares: vec![],
        };
        let ids = [9, 3, 7, 0, 5, 1];
        let pages = Pages {
            pool: span(1, 16 * 256),
            ids: &ids,
            page_bytes: 256,
        };
        let header = (0..64).map(|byte| byte as u8 ^ 0x5a).collect::<Vec<_>>();
        let capture = Table::new(&state, EXTENT, Holder::Slot(1)).unwrap();
        pages.check(&capture, 16).unwrap();
        let plans = (0..2)
            .map(|rank| {
                capture.actions(
                    &held,
                    &pages,
                    true,
                    if rank == 0 {
                        HeaderAction::Write(header.clone())
                    } else {
                        HeaderAction::None
                    },
                )
            })
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        assert_eq!(copy_prefix(&source, plans), Ok(vec![vec![], vec![]]));
        // The pool pages move to the restoring instance.
        for rank in 0..2 {
            let pool = source.bytes(rank, pages.pool);
            target.write(rank, pages.pool, &pool);
        }
        let restore = Table::new(&state, EXTENT, Holder::Slot(4)).unwrap();
        let plans = (0..2)
            .map(|rank| {
                restore.actions(
                    &held,
                    &pages,
                    false,
                    if rank == 0 {
                        HeaderAction::Read
                    } else {
                        HeaderAction::None
                    },
                )
            })
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        let reads = copy_prefix(&target, plans).unwrap();
        assert_eq!(agreed(&reads), Some(header.as_slice()));
        for rank in 0..2 {
            for (from, to) in capture.descriptors.iter().zip(&restore.descriptors) {
                let held = |descriptor: &Descriptor| {
                    DeviceBufferSpanV1::new(
                        span(descriptor.buffer as u32 + 2, 0).buffer,
                        descriptor.offset,
                        descriptor.length,
                    )
                };
                assert_eq!(source.bytes(rank, held(from)), target.bytes(rank, held(to)));
            }
            // Nothing else of the target moved.
            let scratch = span(5, sizes[3] as u64);
            assert!(target.bytes(rank, scratch).iter().all(|&byte| byte == 0));
        }
        let calls = target.calls.borrow();
        assert_eq!(calls.last(), Some(&(1, "drain")));
        assert_eq!(calls.iter().filter(|call| call.1 == "read").count(), 1);
    }

    #[test]
    fn plans_are_checked_before_any_rank_starts() {
        let streams = Streams::new(2);
        let read = |length| Action::Read {
            from: span(1, length),
        };
        assert_eq!(
            copy_prefix(&streams, vec![vec![Action::Launch("copy")]]),
            Err("a snapshot copy needs a plan per rank")
        );
        assert_eq!(
            copy_prefix(&streams, vec![vec![], vec![read(MAX_READ_BYTES + 1)]]),
            Err("a snapshot read is too long")
        );
        assert!(streams.calls.borrow().is_empty());
        // A failed launch drains the ranks that received work; a failed
        // drain takes precedence.
        let plans = || vec![vec![Action::Launch("copy")], vec![Action::Launch("fail")]];
        assert_eq!(copy_prefix(&streams, plans()), Err("launch"));
        assert_eq!(
            *streams.calls.borrow(),
            [(0, "copy"), (1, "fail"), (0, "drain"), (1, "drain")]
        );
        let stuck = Streams {
            stuck: Some(0),
            ..Streams::new(2)
        };
        assert_eq!(copy_prefix(&stuck, plans()), Err("unconfirmed"));
        assert_eq!(agreed(&[vec![1], vec![], vec![2]]), None);
        assert_eq!(agreed(&[vec![], vec![]]), None);
    }

    #[test]
    fn submission_failure_drains_the_failed_rank_and_all_earlier_ranks() {
        for failed in 0..4 {
            let calls = RefCell::new(Vec::new());
            let result = submit_and_drain(
                0..4,
                |rank, copy| {
                    assert_eq!(rank, copy);
                    calls.borrow_mut().push(("submit", rank));
                    if rank == failed {
                        Err("submit")
                    } else {
                        Ok(())
                    }
                },
                |rank| {
                    calls.borrow_mut().push(("drain", rank));
                    true
                },
                "unconfirmed",
            );
            assert_eq!(result, Err("submit"));
            let expected = (0..=failed)
                .map(|rank| ("submit", rank))
                .chain((0..=failed).map(|rank| ("drain", rank)))
                .collect::<Vec<_>>();
            assert_eq!(*calls.borrow(), expected);
        }
    }

    #[test]
    fn a_failed_drain_does_not_skip_other_ranks_and_overrides_submission_errors() {
        for failed in 0..4 {
            let mut drained = Vec::new();
            let result = submit_and_drain(
                0..4,
                |rank, _| if rank == 3 { Err("submit") } else { Ok(()) },
                |rank| {
                    drained.push(rank);
                    rank != failed
                },
                "unconfirmed",
            );
            assert_eq!(result, Err("unconfirmed"));
            assert_eq!(drained, [0, 1, 2, 3]);
        }
    }

    #[test]
    fn successful_copies_finish_after_every_rank_drains() {
        let mut drained = Vec::new();
        assert_eq!(
            submit_and_drain(
                0..3,
                |_, _| Ok(()),
                |rank| {
                    drained.push(rank);
                    true
                },
                "unconfirmed"
            ),
            Ok(())
        );
        assert_eq!(drained, [0, 1, 2]);
    }
}
