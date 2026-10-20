//! Host-mapped rows: read-only mappings of checkpoint bytes whose rows the
//! host gathers on the CPU (`chop.host-mapped-gather.v1`).
//!
//! A planned view placed `ViewPlacementV1::Mapped` is one tensor, or one run
//! of its leading-axis rows, stored unchanged. The direct load checks the
//! record against the checkpoint header and maps that byte range of the file
//! it opened without following links (`runtime_weights::checkpoint`), with
//! `PROT_READ` and `MAP_SHARED`: the page cache holds the bytes, and neither
//! the device nor any page-locked budget does. One mapping serves every rank
//! and image of the process that names the same bytes of the same file; the
//! registry finds it while some image still holds it.
//!
//! A mapping reads what the file holds. A file rewritten in place changes
//! what later reads see, and one truncated while mapped makes reads of the
//! lost pages fault, so a server maps only a checkpoint nobody rewrites while
//! it runs (the same rule its loads already state; `chopinfer verify`
//! checks the payload).

// SAFETY-REVIEWED: this module contains the mmap, madvise and munmap calls.
#![allow(unsafe_code)]

pub(crate) mod gather;
pub(crate) mod resident;

use std::collections::BTreeMap;
use std::fs::File;
use std::io;
use std::sync::{Arc, Mutex, Weak};

/// What the kernel is told about how a mapping is read.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RowAdviceV1 {
    /// Rows are read at random: a fault reads its own page and no
    /// read-ahead (`MADV_RANDOM`). The default, since a gather touches a few
    /// hundred bytes of each page it faults.
    Random,
    /// The kernel's default read-ahead (`MADV_NORMAL`).
    Normal,
    /// Read the whole range into the page cache now (`MADV_WILLNEED`).
    WillNeed,
    /// Drop this mapping's resident pages (`MADV_DONTNEED`): the next read
    /// faults again, from the page cache if it still holds the page. For a
    /// measurement that evicts the file's pages afterwards.
    DontNeed,
}

/// Which bytes a mapping holds: the file (device and inode) and the range.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct RowsKeyV1 {
    identity: Option<[u8; 32]>,
    device: u64,
    inode: u64,
    offset: u64,
    length: u64,
}

/// One read-only mapping of `length` bytes at `offset` of one file.
pub struct MappedRowsV1 {
    /// The mapping's start, at the page boundary at or below `offset`.
    base: *mut libc::c_void,
    mapped_length: usize,
    /// Where the requested bytes start within the mapping.
    lead: usize,
    length: usize,
    key: RowsKeyV1,
    resident: std::sync::OnceLock<resident::Plane>,
}

// SAFETY-REVIEWED: the mapping is read-only memory no one writes through, so
// any thread may read it and drop the last reference.
unsafe impl Send for MappedRowsV1 {}
// SAFETY-REVIEWED: shared references only read the mapping.
unsafe impl Sync for MappedRowsV1 {}

impl std::fmt::Debug for MappedRowsV1 {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("MappedRowsV1")
            .field("offset", &self.key.offset)
            .field("length", &self.length)
            .finish_non_exhaustive()
    }
}

impl MappedRowsV1 {
    pub(crate) fn resident_plane(&self, row_bytes: usize) -> Option<&resident::Plane> {
        self.resident.get().filter(|plane| plane.width == row_bytes)
    }

    /// The mapped bytes.
    #[must_use]
    pub fn bytes(&self) -> &[u8] {
        // SAFETY-REVIEWED: `base + lead` starts `length` readable bytes this
        // value maps until it is dropped.
        unsafe {
            std::slice::from_raw_parts(
                self.base.cast::<u8>().cast_const().add(self.lead),
                self.length,
            )
        }
    }

    #[must_use]
    pub fn len(&self) -> u64 {
        self.length as u64
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.length == 0
    }

    /// The file offset the bytes start at.
    #[must_use]
    pub const fn file_offset(&self) -> u64 {
        self.key.offset
    }

    /// Tells the kernel how the rows are read.
    pub fn advise(&self, advice: RowAdviceV1) -> io::Result<()> {
        let advice = match advice {
            RowAdviceV1::Random => libc::MADV_RANDOM,
            RowAdviceV1::Normal => libc::MADV_NORMAL,
            RowAdviceV1::WillNeed => libc::MADV_WILLNEED,
            RowAdviceV1::DontNeed => libc::MADV_DONTNEED,
        };
        // SAFETY-REVIEWED: the range is exactly the mapping this value owns.
        if unsafe { libc::madvise(self.base, self.mapped_length, advice) } != 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(())
    }

    /// Asks the kernel to read ahead (`MADV_WILLNEED`) the pages holding
    /// `rows`, each `row_bytes` long from the start of [`Self::bytes`], so a
    /// gather's faults on them overlap instead of waiting one at a time. A
    /// hint only: the bytes later read are the same, and errors are ignored.
    pub fn will_need_rows(&self, row_bytes: usize, rows: &[u32]) {
        let page = page_bytes_v1();
        let mut previous = None;
        for &row in rows {
            let Some(range) = row_pages_v1(self.lead, row_bytes, row as usize, page).filter(
                |&(start, length)| start + length <= self.mapped_length.next_multiple_of(page),
            ) else {
                continue;
            };
            if previous == Some(range) {
                continue;
            }
            previous = Some(range);
            // SAFETY-REVIEWED: the page-aligned range lies within the pages of
            // the mapping this value owns.
            unsafe {
                libc::madvise(
                    self.base.cast::<u8>().add(range.0).cast(),
                    range.1,
                    libc::MADV_WILLNEED,
                );
            }
        }
    }

    /// The pages of one window of the mapping: its start within the mapping
    /// and length, page-aligned, the last window ending with the mapping.
    #[must_use]
    pub fn windows(&self, window: usize) -> Vec<(usize, usize)> {
        let page = page_bytes_v1();
        let window = window.max(page).next_multiple_of(page);
        (0..self.mapped_length)
            .step_by(window)
            .map(|start| (start, window.min(self.mapped_length - start)))
            .collect()
    }

    /// Reads one window ([`Self::windows`]) into the page cache and maps it:
    /// asks the kernel to read it ahead (`MADV_WILLNEED`), then touches each
    /// of its pages, so a later read neither waits for the file nor faults.
    pub fn prewarm_window(&self, start: usize, length: usize) -> io::Result<()> {
        let page = page_bytes_v1();
        if !start.is_multiple_of(page)
            || start
                .checked_add(length)
                .is_none_or(|end| end > self.mapped_length)
        {
            return Err(io::Error::other("prewarm window lies outside the mapping"));
        }
        // SAFETY-REVIEWED: the page-aligned range lies within the mapping this
        // value owns.
        if unsafe {
            libc::madvise(
                self.base.cast::<u8>().add(start).cast(),
                length,
                libc::MADV_WILLNEED,
            )
        } != 0
        {
            return Err(io::Error::last_os_error());
        }
        let bytes = self.bytes();
        let mut sum = 0_u8;
        for page_start in (start..start + length).step_by(page) {
            // The page's first byte the requested range holds, if any.
            let index = page_start.max(self.lead) - self.lead;
            if let Some(byte) = bytes.get(index) {
                sum = sum.wrapping_add(*byte);
            }
        }
        std::hint::black_box(sum);
        Ok(())
    }

    /// How many of the mapping's pages are resident now (`mincore`): the
    /// pages a read would not fault from the file.
    pub fn resident_pages(&self) -> io::Result<(u64, u64)> {
        let page = page_bytes_v1();
        let pages = self.mapped_length.div_ceil(page);
        let mut residency = vec![0_u8; pages];
        // SAFETY-REVIEWED: the range is the mapping this value owns, and the
        // vector holds one byte per page of it.
        if unsafe { libc::mincore(self.base, self.mapped_length, residency.as_mut_ptr().cast()) }
            != 0
        {
            return Err(io::Error::last_os_error());
        }
        let resident = residency.iter().filter(|page| **page & 1 != 0).count();
        Ok((resident as u64, pages as u64))
    }
}

impl Drop for MappedRowsV1 {
    fn drop(&mut self) {
        // SAFETY-REVIEWED: this value owns the mapping and no reference to its
        // bytes outlives it.
        unsafe {
            libc::munmap(self.base, self.mapped_length);
        }
    }
}

/// The host's page size.
#[must_use]
pub fn page_bytes_v1() -> usize {
    // SAFETY-REVIEWED: sysconf reads one configuration value.
    let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    usize::try_from(page)
        .ok()
        .filter(|page| page.is_power_of_two())
        .unwrap_or(4096)
}

/// The page-aligned range of a mapping that holds row `row` of `row_bytes`
/// bytes, counted from `lead` bytes into the mapping: its start and length
/// from the mapping's base, covering every page the row touches.
fn row_pages_v1(lead: usize, row_bytes: usize, row: usize, page: usize) -> Option<(usize, usize)> {
    let first = row.checked_mul(row_bytes)?.checked_add(lead)?;
    let end = first
        .checked_add(row_bytes)?
        .checked_next_multiple_of(page)?;
    let start = first / page * page;
    (row_bytes > 0).then_some((start, end - start))
}

/// Every mapping some image still holds, by the bytes it maps.
static REGISTRY: Mutex<BTreeMap<RowsKeyV1, Weak<MappedRowsV1>>> = Mutex::new(BTreeMap::new());

/// Maps `length` bytes at `offset` of `file` read-only, or answers the
/// mapping of the same bytes of the same file this process already holds.
/// The caller has checked that the range is the tensor bytes it names; this
/// refuses an empty range and one past the file's current end.
pub fn map_rows_v1(file: &File, offset: u64, length: u64) -> io::Result<Arc<MappedRowsV1>> {
    map_rows_inner(file, offset, length, None)
}

/// Map a range after the direct loader has authenticated its checkpoint and
/// checked its source tensor against the immutable shard header.
pub(crate) fn map_authenticated_rows_v1(
    file: &File,
    offset: u64,
    length: u64,
    identity: [u8; 32],
) -> io::Result<Arc<MappedRowsV1>> {
    map_rows_inner(file, offset, length, Some(identity))
}

fn map_rows_inner(
    file: &File,
    offset: u64,
    length: u64,
    identity: Option<[u8; 32]>,
) -> io::Result<Arc<MappedRowsV1>> {
    use std::os::unix::fs::MetadataExt;
    let metadata = file.metadata()?;
    if length == 0
        || offset
            .checked_add(length)
            .is_none_or(|end| end > metadata.len())
    {
        return Err(io::Error::other("mapped rows lie outside the file"));
    }
    let key = RowsKeyV1 {
        identity,
        device: metadata.dev(),
        inode: metadata.ino(),
        offset,
        length,
    };
    let mut registry = REGISTRY
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    registry.retain(|_, mapping| mapping.strong_count() > 0);
    if let Some(mapping) = registry.get(&key).and_then(Weak::upgrade) {
        return Ok(mapping);
    }
    let page = page_bytes_v1() as u64;
    let start = offset - offset % page;
    let lead = usize::try_from(offset - start).map_err(io::Error::other)?;
    let length = usize::try_from(length).map_err(io::Error::other)?;
    let mapped_length = lead
        .checked_add(length)
        .ok_or_else(|| io::Error::other("mapped rows exceed this host's addresses"))?;
    let file_offset = libc::off_t::try_from(start).map_err(io::Error::other)?;
    use std::os::fd::AsRawFd;
    // SAFETY-REVIEWED: a fresh read-only shared mapping of a descriptor `file`
    // keeps open for the call; the kernel chooses the address.
    let base = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            mapped_length,
            libc::PROT_READ,
            libc::MAP_SHARED,
            file.as_raw_fd(),
            file_offset,
        )
    };
    if base == libc::MAP_FAILED {
        return Err(io::Error::last_os_error());
    }
    let mapping = Arc::new(MappedRowsV1 {
        base,
        mapped_length,
        lead,
        length,
        key,
        resident: std::sync::OnceLock::new(),
    });
    mapping.advise(RowAdviceV1::Random)?;
    registry.insert(key, Arc::downgrade(&mapping));
    Ok(mapping)
}

/// Bytes every live mapping of this process maps, each counted once.
#[must_use]
pub fn mapped_bytes_v1() -> u64 {
    REGISTRY
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .values()
        .filter_map(Weak::upgrade)
        .map(|mapping| mapping.len())
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Temp(std::path::PathBuf);
    impl Drop for Temp {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    fn file(name: &str, bytes: &[u8]) -> (Temp, File) {
        let path = std::env::temp_dir().join(format!("host-rows-{name}-{}", std::process::id()));
        std::fs::write(&path, bytes).unwrap();
        let file = File::open(&path).unwrap();
        (Temp(path), file)
    }

    #[test]
    fn row_pages_cover_every_page_a_row_touches() {
        let page = 4096;
        // Rows inside one page, ending on a boundary, and crossing one.
        assert_eq!(row_pages_v1(0, 256, 0, page), Some((0, 4096)));
        assert_eq!(row_pages_v1(0, 256, 15, page), Some((0, 4096)));
        assert_eq!(row_pages_v1(0, 256, 16, page), Some((4096, 4096)));
        assert_eq!(row_pages_v1(100, 256, 15, page), Some((0, 8192)));
        assert_eq!(row_pages_v1(100, 256, 16, page), Some((4096, 4096)));
        // A row as long as a page, and one longer than two.
        assert_eq!(row_pages_v1(0, 4096, 3, page), Some((12_288, 4096)));
        assert_eq!(row_pages_v1(4095, 9000, 1, page), Some((12_288, 12_288)));
        assert_eq!(row_pages_v1(0, 0, 3, page), None);
        assert_eq!(row_pages_v1(0, usize::MAX, 2, page), None);
    }

    #[test]
    fn read_ahead_leaves_the_mapped_bytes_unchanged() {
        let bytes: Vec<u8> = (0..40_000_u32).map(|i| (i * 13 + 5) as u8).collect();
        let (_temp, file) = file("ahead", &bytes);
        let rows = map_rows_v1(&file, 300, 39_000).unwrap();
        rows.will_need_rows(256, &[0, 15, 16, 151, 151, 3]);
        // Rows past the end are skipped, not advised outside the mapping.
        rows.will_need_rows(256, &[10_000]);
        rows.will_need_rows(0, &[1]);
        assert_eq!(rows.bytes(), &bytes[300..39_300]);
    }

    #[test]
    fn a_mapping_reads_the_file_bytes_at_any_offset_and_is_shared() {
        let bytes: Vec<u8> = (0..20_000_u32).map(|i| (i * 7 + 3) as u8).collect();
        let (_temp, file) = file("bytes", &bytes);
        for (offset, length) in [(0, 1), (1, 4095), (4095, 2), (4096, 8000), (12_345, 7655)] {
            let rows = map_rows_v1(&file, offset, length).unwrap();
            assert_eq!(
                rows.bytes(),
                &bytes[offset as usize..(offset + length) as usize]
            );
            assert_eq!((rows.file_offset(), rows.len()), (offset, length));
        }
        let first = map_rows_v1(&file, 100, 900).unwrap();
        let second = map_rows_v1(&file, 100, 900).unwrap();
        assert!(Arc::ptr_eq(&first, &second));
        assert!(!Arc::ptr_eq(&first, &map_rows_v1(&file, 100, 901).unwrap()));
        // Another descriptor of the same file finds the same mapping.
        let again = File::open(&_temp.0).unwrap();
        assert!(Arc::ptr_eq(&first, &map_rows_v1(&again, 100, 900).unwrap()));
        assert!(mapped_bytes_v1() >= 900);
        let (resident, pages) = first.resident_pages().unwrap();
        assert!(resident <= pages && pages >= 1);
        for advice in [
            RowAdviceV1::Normal,
            RowAdviceV1::WillNeed,
            RowAdviceV1::Random,
        ] {
            first.advise(advice).unwrap();
        }
        // Prewarming reads and maps every page, window by window.
        // The mapping starts at the page holding offset 100.
        let windows = first.windows(1);
        assert_eq!(windows, [(0, 100 % page_bytes_v1() + 900)]);
        for (start, length) in windows {
            first.prewarm_window(start, length).unwrap();
        }
        let (resident, pages) = first.resident_pages().unwrap();
        assert_eq!(resident, pages);
        assert!(first.prewarm_window(1, 4096).is_err());
        // Once dropped, the same bytes map afresh.
        drop((first, second, again));
        let fresh = map_rows_v1(&file, 100, 900).unwrap();
        assert_eq!(fresh.bytes(), &bytes[100..1000]);
    }

    #[test]
    fn ranges_outside_the_file_or_empty_are_refused() {
        let (_temp, file) = file("bounds", &[1; 100]);
        assert!(map_rows_v1(&file, 0, 0).is_err());
        assert!(map_rows_v1(&file, 0, 101).is_err());
        assert!(map_rows_v1(&file, 100, 1).is_err());
        assert!(map_rows_v1(&file, u64::MAX, 2).is_err());
        assert_eq!(map_rows_v1(&file, 99, 1).unwrap().bytes(), &[1]);
    }
}
