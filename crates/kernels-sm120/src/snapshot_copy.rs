//! The snapshot copy (`kernel_core::snapshot_copy`) on SM120
//! (`cuda/snapshot/copy.cuh`, `cuda/snapshot/README.md`): one thread per
//! 16-byte vector of the snapshot's pages, which finds its segments by
//! binary search and moves each one's part of the vector, whole when it is
//! one segment's aligned vector and byte by byte otherwise. The grid covers
//! the op's pages; the segment count is the launch's. A build compiles the
//! entry under the shared symbol ([`entries`]) or its own.

use kernel_build::{CubinSpec, EntryParam, EntrySpec};
use kernel_core::contract::{Argument, ArgumentList, LaunchPlan, Refusal, Validate};
use kernel_core::properties::Invariance;
use kernel_core::snapshot_copy::{Direction, MAX_VIEWS, SnapshotCopy};

use crate::invariance::{Declared, OpKind};

/// The kernel's header, relative to the CUDA source root.
pub const HEADER: &str = "snapshot/copy.cuh";
/// Threads of a block.
pub const THREADS: u32 = 256;
/// Bytes of the vector each thread moves; pages hold whole vectors.
pub const VECTOR_BYTES: u64 = 16;
/// The shared entry.
pub const SYMBOL: &str = "chop_snapshot_copy_v1";
/// Slots of the entry's view pointer array: every view a table may name.
pub const VIEW_SLOTS: u8 = MAX_VIEWS as u8;
/// What a launch keeps: it moves bytes and computes nothing, so every byte
/// it writes is its source byte at any segment count up to the most an
/// SM120 op serves.
pub const INVARIANCE: Invariance = Invariance::any_rows(crate::backend::MAX_CAPACITY);

/// A snapshot copy entry a build compiles, under `symbol`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Entry {
    pub symbol: String,
}

impl Entry {
    pub fn new(symbol: impl Into<String>) -> Self {
        Self {
            symbol: symbol.into(),
        }
    }

    /// The entry's code under its symbol.
    pub fn spec(&self) -> EntrySpec {
        let pointer = |name: &str, cxx_type: &str| EntryParam::DevicePointer {
            name: name.into(),
            cxx_type: cxx_type.into(),
        };
        EntrySpec::new(
            self.symbol.clone(),
            vec![
                EntryParam::PointerArray {
                    name: "views".into(),
                    slots: VIEW_SLOTS,
                    cxx_type: "uint8_t".into(),
                },
                pointer("pool", "uint8_t"),
                pointer("table", "const uint64_t"),
                EntryParam::U32 {
                    name: "segments".into(),
                },
                EntryParam::U32 {
                    name: "pages".into(),
                },
                EntryParam::U64 {
                    name: "page_bytes".into(),
                },
                EntryParam::U32 {
                    name: "to_pages".into(),
                },
            ],
            "chop::snapshot::copy(views,pool,table,segments,pages,page_bytes,to_pages);".into(),
            THREADS,
            1,
        )
    }
}

/// The shared entry under its kernel-base symbol.
pub fn entries() -> Vec<Entry> {
    vec![Entry::new(SYMBOL)]
}

/// The declaration of the shared entry: a copy every profile shares,
/// claiming [`INVARIANCE`].
pub fn declared_entries() -> Vec<Declared> {
    entries()
        .into_iter()
        .map(|entry| Declared {
            symbol: entry.symbol,
            profile: None,
            op: OpKind::Row,
            invariance: INVARIANCE,
        })
        .collect()
}

/// A cubin `name` holding `entries`.
pub fn cubin_spec(name: impl Into<String>, entries: &[Entry]) -> CubinSpec {
    CubinSpec {
        name: name.into(),
        sources_root: crate::cuda_source_root(),
        include_dirs: vec![],
        headers: vec![format!("#include \"{HEADER}\"")],
        defines: vec![],
        entries: entries.iter().map(Entry::spec).collect(),
    }
}

/// A snapshot copy of up to `capacity` segments through entry `symbol`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Copy {
    op: SnapshotCopy,
    capacity: u32,
    symbol: String,
}

impl Copy {
    /// Refuses pages of a partial vector, a snapshot whose vectors the grid
    /// cannot index and a capacity of no segments or past the backend's:
    /// SM120's limits, not the contract's. Views and pages may lie at any
    /// alignment; unaligned parts move byte by byte.
    pub fn new(op: SnapshotCopy, capacity: u32, symbol: &str) -> Result<Self, Refusal> {
        op.validate()?;
        if !op.page_bytes.is_multiple_of(VECTOR_BYTES) {
            return Err(Refusal(
                "an SM120 snapshot copy needs pages of whole 16-byte vectors",
            ));
        }
        if capacity == 0 || capacity > crate::backend::MAX_CAPACITY {
            return Err(Refusal(
                "an SM120 snapshot copy serves 1 to MAX_CAPACITY segments",
            ));
        }
        let copy = Self {
            op,
            capacity,
            symbol: symbol.into(),
        };
        copy.blocks()?;
        op.table_bytes(capacity)?;
        Ok(copy)
    }

    pub fn op(&self) -> SnapshotCopy {
        self.op
    }

    pub fn capacity(&self) -> u32 {
        self.capacity
    }

    pub fn symbol(&self) -> &str {
        &self.symbol
    }

    /// Blocks of the grid: one thread per vector of the op's pages.
    fn blocks(&self) -> Result<u32, Refusal> {
        let vectors = self.op.snapshot_bytes()? / VECTOR_BYTES;
        u32::try_from(vectors.div_ceil(u64::from(THREADS)))
            .ok()
            .filter(|&blocks| blocks <= i32::MAX as u32)
            .ok_or(Refusal("an SM120 snapshot copy's pages exceed its grid"))
    }

    /// The grid over the op's pages, whatever the segment count.
    pub fn plan(&self) -> Result<LaunchPlan, Refusal> {
        let plan = LaunchPlan {
            grid: [self.blocks()?, 1, 1],
            block: [THREADS, 1, 1],
            dynamic_shared_bytes: 0,
            workspace_bytes: 0,
            cluster: [1, 1, 1],
            pdl: true,
        };
        plan.validate()?;
        Ok(plan)
    }

    /// The arguments of a launch of `segments` segments over the bound
    /// `views` (in view order), `pool` and `table`.
    pub fn arguments(
        &self,
        views: &[u64],
        pool: u64,
        table: u64,
        segments: u32,
    ) -> Result<ArgumentList, Refusal> {
        if segments == 0 || segments > self.capacity {
            return Err(Refusal(
                "snapshot copy segments exceed the prepared capacity",
            ));
        }
        if views.len() as u64 != u64::from(self.op.views) {
            return Err(Refusal("a snapshot copy binds each of its views"));
        }
        if views.iter().any(|&view| view == pool || view == table) || pool == table {
            return Err(Refusal(
                "a snapshot copy's views, pool and table are distinct buffers",
            ));
        }
        let args = ArgumentList(vec![
            Argument::DevicePointerArray(views.to_vec()),
            Argument::DeviceBuffer(pool),
            Argument::DeviceBuffer(table),
            Argument::U32(segments),
            Argument::U32(self.op.pages),
            Argument::U64(self.op.page_bytes),
            Argument::U32(u32::from(self.op.direction == Direction::ToPages)),
        ]);
        args.validate()?;
        Ok(args)
    }
}

/// What the kernel's threads move, run on the CPU in its thread order: the
/// twin of `copy.cuh`'s index mapping (vector, binary search, page lookup
/// and each segment's part of the vector), which tests compare with the
/// oracle. Views are byte slices here, so every part moves byte by byte;
/// the device moves an aligned whole vector in one access, the same bytes.
#[cfg(test)]
pub(crate) fn emulate(
    copy: &Copy,
    views: &mut [&mut [u8]],
    pool: &mut [u8],
    table: &[u8],
    segments: u32,
) -> Result<(), Refusal> {
    let op = copy.op;
    let (pages, listed) = op.decode(table, segments)?;
    let plan = copy.plan()?;
    let threads = u64::from(plan.grid[0]) * u64::from(plan.block[0]);
    for thread in 0..threads {
        let start = thread * VECTOR_BYTES;
        let page = start / op.page_bytes;
        if page >= u64::from(op.pages) {
            continue;
        }
        let low = listed.partition_point(|segment| segment.at + segment.length <= start);
        let end = start + VECTOR_BYTES;
        let paged = (pages[page as usize] * op.page_bytes + start % op.page_bytes) as usize;
        for segment in listed[low..].iter().take_while(|segment| segment.at < end) {
            let from = segment.at.max(start);
            let to = (segment.at + segment.length).min(end);
            let held = (segment.offset + (from - segment.at)) as usize;
            let in_page = paged + (from - start) as usize;
            let bytes = (to - from) as usize;
            let view = &mut views[segment.view as usize];
            match op.direction {
                Direction::ToPages => {
                    pool[in_page..in_page + bytes].copy_from_slice(&view[held..held + bytes]);
                }
                Direction::FromPages => {
                    view[held..held + bytes].copy_from_slice(&pool[in_page..in_page + bytes]);
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use kernel_core::snapshot_copy::{Segment, snapshot_copy};

    /// A Flash-like table: rings of 528-byte cells' planes and odd
    /// fragments, one piece crossing several pages, and a byte tail.
    fn table() -> (SnapshotCopy, Vec<u64>, Vec<Segment>, Vec<u64>) {
        let page_bytes = 1_008;
        let lengths = [
            (0, 4096),
            (0, 128),
            (1, 3_005),
            (2, 17),
            (0, 1_000),
            (2, 4_099),
            (1, 1),
        ];
        let mut at = 0;
        let mut offsets = [32_u64, 3, 0];
        let segments = lengths
            .iter()
            .map(|&(view, length)| {
                let segment = Segment {
                    view,
                    offset: offsets[view as usize],
                    at,
                    length,
                };
                offsets[view as usize] += length + 48;
                at += length;
                segment
            })
            .collect::<Vec<_>>();
        let pages = at.div_ceil(page_bytes) as u32 + 1;
        let ids = (0..u64::from(pages))
            .map(|page| (page * 7 + 3) % 31)
            .collect();
        let op = SnapshotCopy {
            direction: Direction::ToPages,
            views: 3,
            page_bytes,
            pages,
        };
        (op, ids, segments, offsets.to_vec())
    }

    fn pattern(bytes: u64, seed: u64) -> Vec<u8> {
        (0..bytes)
            .map(|i| (i.wrapping_mul(131) ^ (i >> 7) ^ seed) as u8)
            .collect()
    }

    /// The kernel's thread mapping writes exactly the oracle's bytes both
    /// ways, through segments that share vectors, cross pages and end in
    /// byte tails.
    #[test]
    fn the_thread_mapping_moves_the_oracles_bytes() {
        let (op, pages, segments, ends) = table();
        for direction in [Direction::ToPages, Direction::FromPages] {
            let op = SnapshotCopy { direction, ..op };
            let copy = Copy::new(op, 16, SYMBOL).unwrap();
            let encoded = op.encode(&pages, &segments).unwrap();
            let fresh = || {
                let views = ends
                    .iter()
                    .enumerate()
                    .map(|(view, &bytes)| pattern(bytes, view as u64 * 77))
                    .collect::<Vec<_>>();
                (views, pattern(31 * op.page_bytes, 5))
            };
            let (mut expected_views, mut expected_pool) = fresh();
            let (mut views, mut pool) = fresh();
            {
                let mut slices = expected_views
                    .iter_mut()
                    .map(Vec::as_mut_slice)
                    .collect::<Vec<_>>();
                snapshot_copy(&op, &mut slices, &mut expected_pool, &pages, &segments).unwrap();
            }
            let mut slices = views.iter_mut().map(Vec::as_mut_slice).collect::<Vec<_>>();
            emulate(
                &copy,
                &mut slices,
                &mut pool,
                &encoded,
                segments.len() as u32,
            )
            .unwrap();
            assert_eq!(pool, expected_pool, "{direction:?}");
            assert_eq!(views, expected_views, "{direction:?}");
            // The copy moved something both ways.
            assert_ne!(fresh(), (views, pool), "{direction:?}");
        }
    }

    #[test]
    fn the_grid_covers_every_vector_of_the_pages() {
        let (op, ..) = table();
        let copy = Copy::new(op, 7, SYMBOL).unwrap();
        let plan = copy.plan().unwrap();
        let threads = u64::from(plan.grid[0] * plan.block[0]);
        let vectors = op.snapshot_bytes().unwrap() / VECTOR_BYTES;
        assert!(threads >= vectors && threads < vectors + u64::from(THREADS));
        assert!(plan.pdl);
        assert_eq!(
            (plan.block, plan.dynamic_shared_bytes),
            ([THREADS, 1, 1], 0)
        );
    }

    #[test]
    fn arguments_bind_every_view_and_the_launch_segments() {
        let (op, ..) = table();
        let copy = Copy::new(op, 7, SYMBOL).unwrap();
        let arguments = copy.arguments(&[1, 2, 3], 4, 5, 7).unwrap();
        assert_eq!(
            arguments.0,
            [
                Argument::DevicePointerArray(vec![1, 2, 3]),
                Argument::DeviceBuffer(4),
                Argument::DeviceBuffer(5),
                Argument::U32(7),
                Argument::U32(op.pages),
                Argument::U64(op.page_bytes),
                Argument::U32(1),
            ]
        );
        let restore = Copy::new(
            SnapshotCopy {
                direction: Direction::FromPages,
                ..op
            },
            7,
            SYMBOL,
        )
        .unwrap();
        assert_eq!(
            restore.arguments(&[1, 2, 3], 4, 5, 1).unwrap().0[6],
            Argument::U32(0)
        );
        for (views, pool, table, segments) in [
            (&[1, 2, 3][..], 4, 5, 0),
            (&[1, 2, 3][..], 4, 5, 8),
            (&[1, 2][..], 4, 5, 7),
            (&[1, 2, 4][..], 4, 5, 7),
            (&[1, 2, 3][..], 4, 4, 7),
            (&[1, 2, 5][..], 4, 5, 7),
        ] {
            assert!(
                copy.arguments(views, pool, table, segments).is_err(),
                "{views:?} {pool} {table} {segments}"
            );
        }
    }

    #[test]
    fn sm120_limits_refuse_what_the_kernel_cannot_index() {
        let (op, ..) = table();
        for (op, capacity) in [
            (
                SnapshotCopy {
                    page_bytes: 1_000,
                    ..op
                },
                7,
            ),
            (op, 0),
            (op, crate::backend::MAX_CAPACITY + 1),
            (
                SnapshotCopy {
                    page_bytes: 1 << 40,
                    pages: u32::MAX,
                    ..op
                },
                7,
            ),
        ] {
            assert!(
                Copy::new(op, capacity, SYMBOL).is_err(),
                "{op:?} {capacity}"
            );
        }
    }

    #[test]
    fn the_entry_takes_eight_view_slots_and_the_table_sizes() {
        let spec = Entry::new(SYMBOL).spec();
        assert_eq!(spec.symbol, SYMBOL);
        assert!(matches!(
            &spec.params[0],
            EntryParam::PointerArray { slots: 8, .. }
        ));
        assert_eq!(spec.params.len(), 7);
        let cubin = cubin_spec("snapshot_copy_sm120", &entries());
        assert_eq!(cubin.headers, ["#include \"snapshot/copy.cuh\""]);
        assert_eq!(declared_entries()[0].invariance, INVARIANCE);
    }
}
