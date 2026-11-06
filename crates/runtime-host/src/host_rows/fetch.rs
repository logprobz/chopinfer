//! Bounded descriptor reads for ordered resident-cache misses.
//!
//! A miss is read through the mapping's descriptor, never the mapping, into
//! owned bytes. Rows the page cache holds are copied at once; the device
//! reads of the others start together, so a batch of cold rows waits about
//! one device latency ([`start_reads`]). Gathers too large for the resident
//! cache read all their rows the same way ([`copy_large`]).
pub(crate) mod legacy;
mod pool;

use super::MappedRowsV1;
use std::collections::BTreeMap;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};

const ENV: &str = "CHOPINFER_MAPPED_MISS_FETCH";
const READERS_ENV: &str = "CHOPINFER_MAPPED_MISS_READERS";
const START_ENV: &str = "CHOPINFER_MAPPED_MISS_START";
const JOINT_ENV: &str = "CHOPINFER_MAPPED_MISS_JOINT_PLANES";
const LARGE_ENV: &str = "CHOPINFER_MAPPED_LARGE_GATHER_FETCH";
/// Reader threads one batch's rows are split across. A drafted DSpark step
/// gathers its first Engram layer after the drafter, and at 32 rows on rows
/// no request read before about 450 of each plane's 768 rows miss; with 16
/// blocking reads in flight the device waited 1.1-2.1 ms for them per C8
/// step. 450 random 256-byte reads of an Engram codes plane took 2.0 ms at
/// 16 in flight, 0.84 ms at 64 and 0.76 ms at 128 on the serving host's NVMe
/// drive. Started reads ([`start_reads`]) keep every cold row of a batch in
/// flight whatever this count.
const DEFAULT_READERS: usize = 64;
const TILE_BYTES: usize = 4 << 20;
/// The most output bytes one plane of a fetch writes.
const MAX_OUT_BYTES: usize = 256 << 20;
static BATCHES: AtomicU64 = AtomicU64::new(0);
static ROWS: AtomicU64 = AtomicU64::new(0);
static READS: AtomicU64 = AtomicU64::new(0);
static BYTES: AtomicU64 = AtomicU64::new(0);
static NANOS: AtomicU64 = AtomicU64::new(0);
static SERIAL: AtomicU64 = AtomicU64::new(0);
static MAPPED: AtomicU64 = AtomicU64::new(0);
static CACHED: AtomicU64 = AtomicU64::new(0);
static COLD: AtomicU64 = AtomicU64::new(0);
static JOINT: AtomicU64 = AtomicU64::new(0);
static LARGE_GATHERS: AtomicU64 = AtomicU64::new(0);
static LARGE_ROWS: AtomicU64 = AtomicU64::new(0);
static FANOUT_COPIES: AtomicU64 = AtomicU64::new(0);

fn configured(value: Option<std::ffi::OsString>) -> bool {
    value.is_none_or(|value| value == "1")
}

pub(crate) fn enabled() -> bool {
    selection().load(Ordering::Relaxed)
}

fn selection() -> &'static AtomicBool {
    static ENABLED: OnceLock<AtomicBool> = OnceLock::new();
    ENABLED.get_or_init(|| AtomicBool::new(configured(std::env::var_os(ENV))))
}

/// The explicit measurement controller changes this between idle cohorts.
pub(super) fn set_for_measurement(enabled: bool) {
    selection().store(enabled, Ordering::Relaxed);
}

/// `CHOPINFER_MAPPED_MISS_READERS`: 1 to the reader thread count; anything
/// else keeps the default.
fn configured_readers(value: Option<std::ffi::OsString>) -> usize {
    value
        .and_then(|value| value.to_str()?.parse::<usize>().ok())
        .filter(|&readers| valid_readers(readers))
        .unwrap_or(DEFAULT_READERS)
}

fn reader_selection() -> &'static AtomicUsize {
    static READERS: OnceLock<AtomicUsize> = OnceLock::new();
    READERS.get_or_init(|| AtomicUsize::new(configured_readers(std::env::var_os(READERS_ENV))))
}

pub(crate) fn readers() -> usize {
    reader_selection().load(Ordering::Relaxed)
}

pub(super) fn valid_readers(readers: usize) -> bool {
    (1..=pool::WORKERS).contains(&readers)
}

/// The explicit measurement controller changes this between idle cohorts.
pub(super) fn set_readers_for_measurement(readers: usize) -> Result<(), String> {
    if !valid_readers(readers) {
        return Err("mapped miss readers must be 1 to the reader thread count".into());
    }
    reader_selection().store(readers, Ordering::Relaxed);
    Ok(())
}

/// `CHOPINFER_MAPPED_MISS_START`: unset or `1` (the default) starts the
/// device reads of every row of a batch the page cache lacks before waiting
/// for any of them; anything else reads each row with one blocking read, at
/// most [`readers`] in flight. A device busy with another reader (a server
/// loading its checkpoint from the same drive) takes about 1 to 5 ms per
/// 4 KiB read, so blocking reads waited that long for every 64 cold rows.
/// The bytes copied are the same either way.
pub(crate) fn start_reads() -> bool {
    static START: OnceLock<bool> = OnceLock::new();
    *START.get_or_init(|| configured(std::env::var_os(START_ENV)))
}

/// `CHOPINFER_MAPPED_MISS_JOINT_PLANES`: with started reads, unset or `1`
/// (the default) reads the misses of every plane a gather joins
/// ([`copy_planes`]) in one batch, so the gather waits for the device once,
/// not once per plane; anything else reads each plane's misses on their own.
/// Without started reads every plane reads alone, as before them.
pub(crate) fn joint_planes() -> bool {
    static JOINT_PLANES: OnceLock<bool> = OnceLock::new();
    start_reads() && *JOINT_PLANES.get_or_init(|| configured(std::env::var_os(JOINT_ENV)))
}

/// `CHOPINFER_MAPPED_LARGE_GATHER_FETCH`: with started reads, unset or `1`
/// (the default) reads a large gather, one too large for the resident cache
/// ([`super::gather::use_resident`]), through the reader threads
/// ([`copy_large`]); anything else copies it from the mapping as before.
/// Without started reads large gathers copy from the mapping.
pub(crate) fn large_reads() -> bool {
    static LARGE: OnceLock<bool> = OnceLock::new();
    enabled() && start_reads() && *LARGE.get_or_init(|| configured(std::env::var_os(LARGE_ENV)))
}

/// Whether [`copy_large`] reads `bytes` output bytes of `mapping`: large
/// reads are on, the mapping kept its descriptor, and the output fits one
/// fetch's bound.
pub(crate) fn reads_large(mapping: &MappedRowsV1, bytes: usize) -> bool {
    large_reads() && mapping.source.is_some() && bytes <= MAX_OUT_BYTES
}

/// Started reads count the rows the page cache served without waiting and
/// the rows they had to start.
fn count_cold_rows(cached: usize, cold: usize) {
    CACHED.fetch_add(cached as u64, Ordering::Relaxed);
    COLD.fetch_add(cold as u64, Ordering::Relaxed);
}

pub(crate) fn snapshot() -> serde_json::Value {
    let (legacy_gathers, legacy_misses) = legacy::snapshot();
    serde_json::json!({
        "enabled":enabled(), "configured_workers":pool::WORKERS, "readers":readers(),
        "batches":BATCHES.load(Ordering::Relaxed),
        "miss_positions":ROWS.load(Ordering::Relaxed), "unique_reads":READS.load(Ordering::Relaxed),
        "successful_bytes":BYTES.load(Ordering::Relaxed), "nanoseconds":NANOS.load(Ordering::Relaxed),
        "serial_fallback_batches":SERIAL.load(Ordering::Relaxed), "start_reads":start_reads(),
        "page_cached_reads":CACHED.load(Ordering::Relaxed), "started_reads":COLD.load(Ordering::Relaxed),
        "joint_planes":joint_planes(), "joint_batches":JOINT.load(Ordering::Relaxed),
        "large_reads":large_reads(), "large_gathers":LARGE_GATHERS.load(Ordering::Relaxed),
        "large_rows":LARGE_ROWS.load(Ordering::Relaxed),
        "large_fanout_copies":FANOUT_COPIES.load(Ordering::Relaxed),
        "mapped_fallback_rows":MAPPED.load(Ordering::Relaxed),
        "legacy_async_gathers":legacy_gathers, "legacy_async_misses":legacy_misses,
    })
}

fn validate(mapping: &MappedRowsV1, width: usize, rows: &[u32], out: &[u8]) -> Result<(), String> {
    if width == 0
        || !mapping.len().is_multiple_of(width as u64)
        || rows.len().checked_mul(width) != Some(out.len())
        || out.len() > MAX_OUT_BYTES
        || rows
            .iter()
            .any(|&row| u64::from(row) >= mapping.len() / width as u64)
    {
        return Err("mapped miss fetch has invalid rows or output extent".into());
    }
    Ok(())
}

/// Fill hits first, then submit the ordered miss set. The owning caller keeps
/// its existing upload and fanout ordering after this function returns.
pub(crate) fn copy_rows(
    mapping: &MappedRowsV1,
    width: usize,
    rows: &[u32],
    out: &mut [u8],
) -> Result<(), String> {
    validate(mapping, width, rows, out)?;
    let misses = mapping.resident_plane(width).map_or_else(
        || (0..rows.len()).collect(),
        |plane| plane.copy_resident_rows(rows, out),
    );
    copy_misses(mapping, width, rows, &misses, out)
}

/// Consume original unfilled output positions from the bulk resident lookup.
/// Fetched bytes stay owned until copied and published, regardless of whether
/// their source pages remain in the OS cache.
pub(crate) fn copy_misses(
    mapping: &MappedRowsV1,
    width: usize,
    rows: &[u32],
    misses: &[usize],
    out: &mut [u8],
) -> Result<(), String> {
    copy_misses_from(mapping, width, rows, misses, out, mapping.source.as_ref())
}

fn copy_misses_from(
    mapping: &MappedRowsV1,
    width: usize,
    rows: &[u32],
    misses: &[usize],
    out: &mut [u8],
    source: Option<&std::sync::Arc<std::fs::File>>,
) -> Result<(), String> {
    validate(mapping, width, rows, out)?;
    if misses.iter().any(|&position| position >= rows.len())
        || misses.windows(2).any(|pair| pair[0] >= pair[1])
    {
        return Err("mapped miss positions must be unique and ordered".into());
    }
    if misses.is_empty() {
        return Ok(());
    }
    let started = std::time::Instant::now();
    BATCHES.fetch_add(1, Ordering::Relaxed);
    ROWS.fetch_add(misses.len() as u64, Ordering::Relaxed);
    let Some(source) = source else {
        MAPPED.fetch_add(misses.len() as u64, Ordering::Relaxed);
        pool::fallback("mapped source descriptor is unavailable");
        for &position in misses {
            let row = rows[position];
            let target = &mut out[position * width..][..width];
            if let Some(plane) = mapping.resident_plane(width) {
                plane.copy(mapping.bytes(), row, 0, target);
            } else {
                target.copy_from_slice(&mapping.bytes()[row as usize * width..][..width]);
            }
        }
        elapsed(started);
        return Ok(());
    };
    for tile in misses.chunks(tile_rows(width)) {
        let (requests, slots) = requests(mapping, width, rows, tile);
        let (data, serial) = pool::read(source, &requests)?;
        count_reads(requests.len(), width, serial);
        publish(mapping, width, rows, tile, &slots, &data, out);
    }
    elapsed(started);
    Ok(())
}

fn tile_rows(width: usize) -> usize {
    (TILE_BYTES / width).clamp(1, 4096)
}

/// One read per distinct row of `misses`, and each miss's read.
fn requests(
    mapping: &MappedRowsV1,
    width: usize,
    rows: &[u32],
    misses: &[usize],
) -> (Vec<(u64, usize)>, Vec<usize>) {
    let mut index = BTreeMap::new();
    let mut requests = Vec::new();
    let mut slots = Vec::with_capacity(misses.len());
    for &position in misses {
        let row = rows[position];
        let slot = *index.entry(row).or_insert_with(|| {
            let slot = requests.len();
            // validate established row * width < mapping length, and the
            // mapping constructor checked its whole file interval.
            requests.push((mapping.file_offset() + u64::from(row) * width as u64, width));
            slot
        });
        slots.push(slot);
    }
    (requests, slots)
}

/// Copies each miss's fetched row (row `slots[i]` of `data`, which holds the
/// fetched rows end to end) into its output position and publishes it to the
/// resident cache, in position order: completion order never changes
/// publication order.
fn publish(
    mapping: &MappedRowsV1,
    width: usize,
    rows: &[u32],
    misses: &[usize],
    slots: &[usize],
    data: &[u8],
    out: &mut [u8],
) {
    let resident = mapping.resident_plane(width);
    for (&position, &slot) in misses.iter().zip(slots) {
        let bytes = &data[slot * width..][..width];
        out[position * width..][..width].copy_from_slice(bytes);
        if let Some(plane) = resident {
            plane.publish_fetched(rows[position], bytes);
        }
    }
}

fn count_reads(reads: usize, width: usize, serial: bool) {
    READS.fetch_add(reads as u64, Ordering::Relaxed);
    BYTES.fetch_add((reads * width) as u64, Ordering::Relaxed);
    if serial {
        SERIAL.fetch_add(1, Ordering::Relaxed);
    }
}

fn elapsed(started: std::time::Instant) {
    NANOS.fetch_add(
        started.elapsed().as_nanos().min(u128::from(u64::MAX)) as u64,
        Ordering::Relaxed,
    );
}

/// One plane of a joint copy ([`copy_planes`]).
pub(crate) struct PlaneCopy<'a> {
    pub(crate) mapping: &'a MappedRowsV1,
    pub(crate) width: usize,
    pub(crate) rows: &'a [u32],
    pub(crate) out: &'a mut [u8],
}

/// [`copy_rows`] for several planes with one wait for the device: each
/// plane's resident hits are copied first, then the descriptor reads of every
/// plane's misses go out as one batch, then each plane's fetched rows are
/// copied and published, plane by plane in order. Where a plane has no
/// descriptor or more distinct misses than one tile, every plane reads its
/// misses on its own instead. The bytes copied equal [`copy_rows`] plane by
/// plane: every row is an exact read of its file range, or a resident copy
/// of one.
pub(crate) fn copy_planes(planes: &mut [PlaneCopy<'_>]) -> Result<(), String> {
    copy_planes_with(planes, pool::read_groups)
}

fn copy_planes_with(
    planes: &mut [PlaneCopy<'_>],
    read: impl FnOnce(&[pool::Group<'_>]) -> Result<pool::GroupRows, String>,
) -> Result<(), String> {
    for plane in planes.iter() {
        validate(plane.mapping, plane.width, plane.rows, plane.out)?;
    }
    let misses = planes
        .iter_mut()
        .map(|plane| {
            plane.mapping.resident_plane(plane.width).map_or_else(
                || (0..plane.rows.len()).collect::<Vec<_>>(),
                |resident| resident.copy_resident_rows(plane.rows, plane.out),
            )
        })
        .collect::<Vec<_>>();
    let work = planes
        .iter()
        .zip(&misses)
        .map(|(plane, misses)| requests(plane.mapping, plane.width, plane.rows, misses))
        .collect::<Vec<_>>();
    let joint = planes.iter().zip(&work).all(|(plane, (requests, _))| {
        plane.mapping.source.is_some() && requests.len() <= tile_rows(plane.width)
    });
    if !joint {
        for (plane, misses) in planes.iter_mut().zip(&misses) {
            copy_misses(plane.mapping, plane.width, plane.rows, misses, plane.out)?;
        }
        return Ok(());
    }
    let missing = misses.iter().filter(|misses| !misses.is_empty()).count();
    if missing == 0 {
        return Ok(());
    }
    let started = std::time::Instant::now();
    BATCHES.fetch_add(missing as u64, Ordering::Relaxed);
    ROWS.fetch_add(
        misses.iter().map(Vec::len).sum::<usize>() as u64,
        Ordering::Relaxed,
    );
    if missing > 1 {
        JOINT.fetch_add(1, Ordering::Relaxed);
    }
    let groups = planes
        .iter()
        .zip(&work)
        .filter_map(|(plane, (requests, _))| {
            let source = plane.mapping.source.as_ref()?;
            (!requests.is_empty()).then_some((source, requests.as_slice()))
        })
        .collect::<Vec<_>>();
    let (data, serial) = read(&groups)?;
    if data.len() != groups.len()
        || data.iter().zip(&groups).any(|(rows, (_, requests))| {
            rows.len() != requests.iter().map(|&(_, length)| length).sum::<usize>()
        })
    {
        return Err("mapped joint read answered another shape".into());
    }
    let mut data = data.into_iter();
    for ((plane, misses), (requests, slots)) in planes.iter_mut().zip(&misses).zip(&work) {
        if requests.is_empty() {
            continue;
        }
        let rows = data.next().expect("one answer per group");
        count_reads(requests.len(), plane.width, serial);
        publish(
            plane.mapping,
            plane.width,
            plane.rows,
            misses,
            slots,
            &rows,
            plane.out,
        );
    }
    elapsed(started);
    Ok(())
}

/// Copies every plane's rows as exact reads of their file ranges through the
/// reader threads, all planes in one batch, so the gather waits for the
/// device once: the large-gather route ([`reads_large`]). Every position is
/// read, duplicates included, as the mapped copy reads them; the resident
/// cache is neither consulted nor filled, so a streaming gather cannot
/// displace the rows decode steps reuse. Refuses a bad row or a plane
/// without a descriptor before any write.
pub(crate) fn copy_large(planes: &mut [PlaneCopy<'_>]) -> Result<(), String> {
    copy_large_with(planes, pool::read_groups_into)
}

fn copy_large_with(
    planes: &mut [PlaneCopy<'_>],
    read: impl FnOnce(&[pool::Group<'_>], &mut [&mut [u8]]) -> Result<bool, String>,
) -> Result<(), String> {
    let mut sources = Vec::with_capacity(planes.len());
    for plane in planes.iter() {
        validate(plane.mapping, plane.width, plane.rows, plane.out)?;
        sources.push(
            plane
                .mapping
                .source
                .as_ref()
                .ok_or("a large mapped gather needs the mapping's descriptor")?,
        );
    }
    let started = std::time::Instant::now();
    let requests = planes
        .iter()
        .map(|plane| {
            let base = plane.mapping.file_offset();
            // validate established row * width < mapping length, and the
            // mapping constructor checked its whole file interval.
            plane
                .rows
                .iter()
                .map(|&row| (base + u64::from(row) * plane.width as u64, plane.width))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let groups = sources
        .into_iter()
        .zip(&requests)
        .map(|(source, requests)| (source, requests.as_slice()))
        .collect::<Vec<_>>();
    let serial = read(
        &groups,
        &mut planes
            .iter_mut()
            .map(|plane| &mut *plane.out)
            .collect::<Vec<_>>(),
    )?;
    let rows = planes.iter().map(|plane| plane.rows.len()).sum::<usize>() as u64;
    BATCHES.fetch_add(planes.len() as u64, Ordering::Relaxed);
    ROWS.fetch_add(rows, Ordering::Relaxed);
    for plane in planes.iter() {
        count_reads(plane.rows.len(), plane.width, false);
    }
    if serial {
        SERIAL.fetch_add(1, Ordering::Relaxed);
    }
    if planes.len() > 1 {
        JOINT.fetch_add(1, Ordering::Relaxed);
    }
    LARGE_GATHERS.fetch_add(1, Ordering::Relaxed);
    LARGE_ROWS.fetch_add(rows, Ordering::Relaxed);
    elapsed(started);
    Ok(())
}

/// A large gather's bytes for the ranks that gather the same rows after
/// the first: a tensor-parallel layer gathers its rows of one shared mapping
/// once per rank ([`LargeFanout::rows`]).
#[derive(Default)]
pub(crate) struct LargeFanout {
    kept: Vec<Kept>,
}

struct Kept {
    mapping: std::sync::Weak<MappedRowsV1>,
    width: usize,
    rows: Vec<u32>,
    bytes: std::sync::Arc<Vec<u8>>,
    left: usize,
}

/// The most bytes a fanout keeps at once.
const FANOUT_BYTES: usize = 1 << 30;

impl LargeFanout {
    /// `rows` of `mapping`, `width` bytes each, end to end: the copy an
    /// earlier gather of the same rows kept, or read now ([`copy_large`])
    /// and kept for `later` more gathers of them. A kept copy is released
    /// after its last use, when another gather of its mapping and width
    /// replaces it, or with its mapping. The bytes equal a fresh read: the
    /// mapped file never changes.
    pub(crate) fn rows(
        &mut self,
        mapping: &std::sync::Arc<MappedRowsV1>,
        width: usize,
        rows: &[u32],
        later: usize,
    ) -> Result<std::sync::Arc<Vec<u8>>, String> {
        let same = |kept: &Kept| {
            kept.width == width
                && kept
                    .mapping
                    .upgrade()
                    .is_some_and(|kept| std::sync::Arc::ptr_eq(&kept, mapping))
        };
        if let Some(index) = self
            .kept
            .iter()
            .position(|kept| same(kept) && kept.rows == rows)
        {
            let kept = &mut self.kept[index];
            kept.left -= 1;
            let bytes = std::sync::Arc::clone(&kept.bytes);
            if kept.left == 0 {
                self.kept.swap_remove(index);
            }
            FANOUT_COPIES.fetch_add(1, Ordering::Relaxed);
            return Ok(bytes);
        }
        self.kept
            .retain(|kept| kept.mapping.strong_count() > 0 && !same(kept));
        let length = rows
            .len()
            .checked_mul(width)
            .ok_or("mapped miss fetch has invalid rows or output extent")?;
        let mut bytes = vec![0; length];
        copy_large(&mut [PlaneCopy {
            mapping,
            width,
            rows,
            out: &mut bytes,
        }])?;
        let bytes = std::sync::Arc::new(bytes);
        let held = self.kept.iter().map(|kept| kept.bytes.len()).sum::<usize>();
        if later > 0 && held + length <= FANOUT_BYTES {
            self.kept.push(Kept {
                mapping: std::sync::Arc::downgrade(mapping),
                width,
                rows: rows.to_vec(),
                bytes: std::sync::Arc::clone(&bytes),
                left: later,
            });
        }
        Ok(bytes)
    }
}

#[cfg(test)]
mod tests;
