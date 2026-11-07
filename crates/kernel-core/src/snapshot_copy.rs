//! Snapshot copies: byte segments moved between bound views and the pages
//! of a paged pool, nothing computed ([`SnapshotCopy`]).
//!
//! A snapshot is a run of bytes laid across pool pages: snapshot byte `c`
//! lives at byte `pages[c / page_bytes] * page_bytes + c % page_bytes` of
//! the pool. Segment `s` is `s.length` bytes at byte `s.offset` of view
//! `s.view`, at snapshot byte `s.at`, so a segment may continue on the next
//! page. A copy to the pages writes every segment's snapshot bytes from its
//! view; a copy from the pages writes every segment's view bytes from its
//! snapshot bytes. Every other byte of the pool and of the views keeps its
//! value, and bytes are copied without interpretation. A segment may have
//! any length and lie at any offset: how a backend moves the bytes (vector
//! widths, threads, grids) is its own choice, and every choice writes the
//! same bytes.
//!
//! A launch reads its table from one device buffer: the page IDs as
//! little-endian u64, then the segments, each four little-endian u64
//! (offset, at, length, view) ([`SnapshotCopy::encode`]).
//! [`SnapshotCopy::check`] states what makes a table valid: distinct pages
//! inside the pool, every segment inside its view and the snapshot, the
//! segments in snapshot order and disjoint, and no two segments of one view
//! overlapping. The host checks a table, and that its views, its pool and
//! its table buffer do not overlap one another, before it uploads or
//! launches anything; a backend reads the table's size, never its contents.
//! The oracle is [`snapshot_copy`], whose page-split copies
//! ([`SnapshotCopy::fragments`]) are the device copies a host would issue
//! instead.

use crate::contract::{Refusal, Validate};

/// The numerical contract: bytes moved, none interpreted.
pub const CONTRACT: &str = "snapshot-copy.paged-segments.bytes.v1";
/// The most views one launch binds.
pub const MAX_VIEWS: u32 = 8;
/// Bytes of one page ID in the table.
pub const PAGE_ID_BYTES: u64 = 8;
/// Bytes of one segment in the table.
pub const SEGMENT_BYTES: u64 = 32;

/// Which way a snapshot copy moves bytes.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum Direction {
    /// From the views into the pages (a capture).
    ToPages,
    /// From the pages into the views (a restore).
    FromPages,
}

/// `length` bytes at byte `offset` of view `view`, at snapshot byte `at`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Segment {
    pub view: u32,
    pub offset: u64,
    pub at: u64,
    pub length: u64,
}

impl Segment {
    /// The snapshot byte after the segment.
    fn end(&self) -> Result<u64, Refusal> {
        self.at
            .checked_add(self.length)
            .ok_or(Refusal("a snapshot segment overflows u64"))
    }
}

/// The part of a segment that lies in one page: `length` bytes at byte
/// `offset` of view `view` and at byte `pool` of the pool.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Fragment {
    pub view: u32,
    pub offset: u64,
    pub pool: u64,
    pub length: u64,
}

/// A copy between `views` bound views and a snapshot of `pages` pool pages
/// of `page_bytes` each, in `direction`. The segment count is the launch's;
/// a backend prepares the op for a most segments (`Need::capacity`).
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct SnapshotCopy {
    pub direction: Direction,
    pub views: u32,
    pub page_bytes: u64,
    pub pages: u32,
}

impl SnapshotCopy {
    /// Bytes the snapshot's pages hold: every segment ends at or before.
    pub fn snapshot_bytes(&self) -> Result<u64, Refusal> {
        self.page_bytes
            .checked_mul(u64::from(self.pages))
            .ok_or(Refusal("snapshot copy sizes overflow u64"))
    }

    /// Bytes of a table of `segments` segments.
    pub fn table_bytes(&self, segments: u32) -> Result<u64, Refusal> {
        (u64::from(segments) * SEGMENT_BYTES)
            .checked_add(u64::from(self.pages) * PAGE_ID_BYTES)
            .ok_or(Refusal("snapshot copy sizes overflow u64"))
    }

    /// Checks `segments` over `pages` for views of `view_bytes` bytes each
    /// and a pool of `pool_bytes`: the op's page count of distinct pages,
    /// each inside the pool; every segment nonempty, inside its view and
    /// the snapshot; the segments in snapshot order and disjoint; and no
    /// two segments of one view overlapping, so the bytes a copy writes do
    /// not depend on the order it writes them in.
    pub fn check(
        &self,
        pages: &[u64],
        segments: &[Segment],
        view_bytes: &[u64],
        pool_bytes: u64,
    ) -> Result<(), Refusal> {
        self.validate()?;
        if pages.len() as u64 != u64::from(self.pages)
            || view_bytes.len() as u64 != u64::from(self.views)
        {
            return Err(Refusal(
                "a snapshot copy names one page ID a page and one size a view",
            ));
        }
        let mut seen = std::collections::BTreeSet::new();
        for &page in pages {
            if !seen.insert(page) {
                return Err(Refusal("a snapshot copy names one page twice"));
            }
            if page
                .checked_add(1)
                .and_then(|end| end.checked_mul(self.page_bytes))
                .is_none_or(|end| end > pool_bytes)
            {
                return Err(Refusal("a snapshot page lies outside the pool"));
            }
        }
        let snapshot = self.snapshot_bytes()?;
        let mut held = Vec::with_capacity(segments.len());
        let mut previous = 0;
        for segment in segments {
            let bytes = *view_bytes
                .get(segment.view as usize)
                .ok_or(Refusal("a snapshot segment names no bound view"))?;
            let end = segment.end()?;
            if segment.length == 0 {
                return Err(Refusal("a snapshot segment holds no bytes"));
            }
            if segment
                .offset
                .checked_add(segment.length)
                .is_none_or(|end| end > bytes)
            {
                return Err(Refusal("a snapshot segment lies outside its view"));
            }
            if end > snapshot {
                return Err(Refusal("a snapshot segment lies outside its pages"));
            }
            if segment.at < previous {
                return Err(Refusal(
                    "snapshot segments are not in snapshot order or overlap",
                ));
            }
            previous = end;
            held.push((
                segment.view,
                segment.offset,
                segment.offset + segment.length,
            ));
        }
        held.sort_unstable();
        if held
            .windows(2)
            .any(|pair| pair[0].0 == pair[1].0 && pair[1].1 < pair[0].2)
        {
            return Err(Refusal("snapshot segments of one view overlap"));
        }
        Ok(())
    }

    /// The table a launch reads: `pages`, then `segments`.
    pub fn encode(&self, pages: &[u64], segments: &[Segment]) -> Result<Vec<u8>, Refusal> {
        if pages.len() as u64 != u64::from(self.pages) {
            return Err(Refusal("a snapshot copy names one page ID a page"));
        }
        let count =
            u32::try_from(segments.len()).map_err(|_| Refusal("too many snapshot segments"))?;
        let mut table = Vec::with_capacity(self.table_bytes(count)? as usize);
        for &page in pages {
            table.extend(page.to_le_bytes());
        }
        for segment in segments {
            for word in [
                segment.offset,
                segment.at,
                segment.length,
                u64::from(segment.view),
            ] {
                table.extend(word.to_le_bytes());
            }
        }
        Ok(table)
    }

    /// The page IDs and the first `segments` segments of `table`.
    pub fn decode(&self, table: &[u8], segments: u32) -> Result<(Vec<u64>, Vec<Segment>), Refusal> {
        if (table.len() as u64) < self.table_bytes(segments)? {
            return Err(Refusal("a snapshot table is shorter than its segments"));
        }
        let mut words = table
            .chunks_exact(8)
            .map(|word| u64::from_le_bytes(word.try_into().expect("eight bytes")));
        let pages = words.by_ref().take(self.pages as usize).collect();
        let segments = (0..segments)
            .map(|_| {
                let mut next = || words.next().expect("the table holds its segments");
                let (offset, at, length) = (next(), next(), next());
                let view = u32::try_from(next())
                    .map_err(|_| Refusal("a snapshot segment names no bound view"))?;
                Ok(Segment {
                    view,
                    offset,
                    at,
                    length,
                })
            })
            .collect::<Result<_, Refusal>>()?;
        Ok((pages, segments))
    }

    /// Each segment split at page boundaries, in snapshot order: where each
    /// run of it lies in its view and in the pool.
    pub fn fragments(&self, pages: &[u64], segments: &[Segment]) -> Result<Vec<Fragment>, Refusal> {
        self.validate()?;
        let snapshot = self.snapshot_bytes()?;
        if pages.len() as u64 != u64::from(self.pages) {
            return Err(Refusal("a snapshot copy names one page ID a page"));
        }
        let mut fragments = Vec::new();
        for segment in segments {
            if segment.end()? > snapshot {
                return Err(Refusal("a snapshot segment lies outside its pages"));
            }
            let mut done = 0;
            while done < segment.length {
                let byte = segment.at + done;
                let within = byte % self.page_bytes;
                let length = (self.page_bytes - within).min(segment.length - done);
                let pool = pages[(byte / self.page_bytes) as usize]
                    .checked_mul(self.page_bytes)
                    .and_then(|base| base.checked_add(within))
                    .ok_or(Refusal("a snapshot page offset overflows u64"))?;
                fragments.push(Fragment {
                    view: segment.view,
                    offset: segment.offset + done,
                    pool,
                    length,
                });
                done += length;
            }
        }
        Ok(fragments)
    }
}

impl Validate for SnapshotCopy {
    fn validate(&self) -> Result<(), Refusal> {
        if self.views == 0 || self.views > MAX_VIEWS {
            return Err(Refusal("a snapshot copy binds one to eight views"));
        }
        if self.page_bytes == 0 || self.pages == 0 {
            return Err(Refusal("a snapshot copy needs pages of bytes"));
        }
        self.snapshot_bytes().map(|_| ())
    }
}

crate::scratch_record! {
    /// A snapshot copy's buffers: the `views` its segments name, in view
    /// order, the `pool` the page IDs address, and the `table` of page IDs
    /// and segments ([`SnapshotCopy::encode`]), which holds at least the
    /// launch's segments.
    #[derive(Clone, Debug, Eq, PartialEq)]
    pub struct SnapshotCopyBuffers {
        pub views: values,
        pub pool: value,
        pub table: value,
    }
}

/// The snapshot copy of `op` over `pages` and `segments` between `views`
/// and `pool`, after checking them ([`SnapshotCopy::check`]). Nothing is
/// written when the check fails.
pub fn snapshot_copy(
    op: &SnapshotCopy,
    views: &mut [&mut [u8]],
    pool: &mut [u8],
    pages: &[u64],
    segments: &[Segment],
) -> Result<(), Refusal> {
    let view_bytes = views
        .iter()
        .map(|view| view.len() as u64)
        .collect::<Vec<_>>();
    op.check(pages, segments, &view_bytes, pool.len() as u64)?;
    for fragment in op.fragments(pages, segments)? {
        let view = &mut views[fragment.view as usize];
        let held = fragment.offset as usize..(fragment.offset + fragment.length) as usize;
        let paged = fragment.pool as usize..(fragment.pool + fragment.length) as usize;
        match op.direction {
            Direction::ToPages => pool[paged].copy_from_slice(&view[held]),
            Direction::FromPages => view[held].copy_from_slice(&pool[paged]),
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const PAGE: u64 = 64;

    fn op(direction: Direction) -> SnapshotCopy {
        SnapshotCopy {
            direction,
            views: 2,
            page_bytes: PAGE,
            pages: 4,
        }
    }

    /// Three segments end to end over two views: an aligned one that fills
    /// the first page, an odd-length one that crosses into the third page,
    /// and an unaligned tail that crosses again.
    fn segments() -> Vec<Segment> {
        vec![
            Segment {
                view: 1,
                offset: 32,
                at: 0,
                length: 64,
            },
            Segment {
                view: 0,
                offset: 5,
                at: 64,
                length: 77,
            },
            Segment {
                view: 1,
                offset: 100,
                at: 141,
                length: 51,
            },
        ]
    }

    const PAGES: [u64; 4] = [6, 2, 9, 0];

    fn pattern(bytes: usize, seed: u8) -> Vec<u8> {
        (0..bytes)
            .map(|i| (i as u8).wrapping_mul(31).wrapping_add(seed))
            .collect()
    }

    /// Snapshot byte `c` of each segment lands on page `PAGES[c / 64]`, at
    /// `c % 64`, and nothing else of the pool changes; the restore puts
    /// every byte back.
    #[test]
    fn segments_map_bytes_through_the_pages() {
        let (mut first, mut second) = (pattern(96, 3), pattern(200, 101));
        let original = (first.clone(), second.clone());
        let mut pool = vec![0xee; 10 * PAGE as usize];
        snapshot_copy(
            &op(Direction::ToPages),
            &mut [&mut first, &mut second],
            &mut pool,
            &PAGES,
            &segments(),
        )
        .unwrap();
        let views = [&original.0, &original.1];
        let mut covered = vec![false; pool.len()];
        for segment in segments() {
            for byte in 0..segment.length {
                let c = segment.at + byte;
                let pooled = (PAGES[(c / PAGE) as usize] * PAGE + c % PAGE) as usize;
                assert_eq!(
                    pool[pooled],
                    views[segment.view as usize][(segment.offset + byte) as usize]
                );
                covered[pooled] = true;
            }
        }
        assert!(
            pool.iter()
                .zip(&covered)
                .all(|(&byte, &covered)| covered || byte == 0xee)
        );
        let (mut first, mut second) = (vec![0; 96], vec![0; 200]);
        snapshot_copy(
            &op(Direction::FromPages),
            &mut [&mut first, &mut second],
            &mut pool,
            &PAGES,
            &segments(),
        )
        .unwrap();
        for segment in segments() {
            let held = segment.offset as usize..(segment.offset + segment.length) as usize;
            let restored = if segment.view == 0 { &first } else { &second };
            assert_eq!(restored[held.clone()], views[segment.view as usize][held]);
        }
        // Bytes no segment names stay as they were.
        assert!(first[..5].iter().all(|&byte| byte == 0));
        assert!(second[..32].iter().all(|&byte| byte == 0));
    }

    /// Each segment splits at every page boundary it crosses, in order.
    #[test]
    fn fragments_split_at_page_boundaries() {
        let fragments = op(Direction::ToPages)
            .fragments(&PAGES, &segments())
            .unwrap();
        let runs = fragments
            .iter()
            .map(|f| (f.view, f.offset, f.pool, f.length))
            .collect::<Vec<_>>();
        assert_eq!(
            runs,
            [
                (1, 32, 6 * PAGE, 64),
                (0, 5, 2 * PAGE, 64),
                (0, 69, 9 * PAGE, 13),
                (1, 100, 9 * PAGE + 13, 51),
            ]
        );
    }

    #[test]
    fn tables_round_trip_through_their_encoding() {
        let op = op(Direction::ToPages);
        let table = op.encode(&PAGES, &segments()).unwrap();
        assert_eq!(table.len() as u64, op.table_bytes(3).unwrap());
        assert_eq!(table.len(), 4 * 8 + 3 * 32);
        assert_eq!(op.decode(&table, 3).unwrap(), (PAGES.to_vec(), segments()));
        assert!(op.decode(&table[..table.len() - 1], 3).is_err());
        assert!(op.encode(&PAGES[..3], &segments()).is_err());
    }

    /// Every invalid table is refused before any byte moves.
    #[test]
    fn invalid_tables_are_refused_and_write_nothing() {
        let segment = |view, offset, at, length| Segment {
            view,
            offset,
            at,
            length,
        };
        // A segment outside its view, outside the pages, empty, of a view
        // not bound, out of order, overlapping another in the snapshot,
        // overlapping another of its view, or past u64.
        let bad = [
            vec![segment(0, 90, 0, 7)],
            vec![segment(1, 0, 250, 7)],
            vec![segment(0, 0, 0, 0)],
            vec![segment(2, 0, 0, 4)],
            vec![segment(0, 0, 10, 4), segment(1, 0, 0, 4)],
            vec![segment(0, 0, 0, 8), segment(1, 0, 7, 4)],
            vec![segment(1, 0, 0, 8), segment(1, 7, 8, 4)],
            vec![segment(0, u64::MAX, 0, 1)],
            vec![segment(0, 0, u64::MAX, 1)],
        ];
        let pages_cases: [&[u64]; 3] = [&[6, 2, 9, 6], &[6, 2, 9, 10], &[6, 2, 9]];
        let check = |pages: &[u64], segments: &[Segment]| {
            let (mut first, mut second) = (pattern(96, 3), pattern(200, 101));
            let mut pool = vec![0xee; 10 * PAGE as usize];
            let result = snapshot_copy(
                &op(Direction::FromPages),
                &mut [&mut first, &mut second],
                &mut pool,
                pages,
                segments,
            );
            if result.is_err() {
                assert_eq!(first, pattern(96, 3));
                assert_eq!(second, pattern(200, 101));
            }
            result
        };
        for segments in &bad {
            assert!(check(&PAGES, segments).is_err(), "{segments:?}");
        }
        for pages in pages_cases {
            assert!(check(pages, &segments()).is_err(), "{pages:?}");
        }
        assert_eq!(check(&PAGES, &segments()).map(|_| ()), Ok(()));
        // Segments of different views may share offsets; adjacent segments
        // of one view may touch.
        assert!(
            check(
                &PAGES,
                &[
                    segment(0, 0, 0, 8),
                    segment(1, 0, 8, 8),
                    segment(1, 8, 16, 8)
                ]
            )
            .is_ok()
        );
        for op in [
            SnapshotCopy {
                views: 0,
                ..op(Direction::ToPages)
            },
            SnapshotCopy {
                views: MAX_VIEWS + 1,
                ..op(Direction::ToPages)
            },
            SnapshotCopy {
                page_bytes: 0,
                ..op(Direction::ToPages)
            },
            SnapshotCopy {
                pages: 0,
                ..op(Direction::ToPages)
            },
            SnapshotCopy {
                page_bytes: u64::MAX,
                ..op(Direction::ToPages)
            },
        ] {
            assert!(op.validate().is_err(), "{op:?}");
        }
    }
}
