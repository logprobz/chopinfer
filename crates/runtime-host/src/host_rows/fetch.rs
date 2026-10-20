//! Bounded descriptor reads for ordered resident-cache misses.
mod pool;

use super::MappedRowsV1;
use std::collections::BTreeMap;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

const ENV: &str = "CHOPINFER_MAPPED_MISS_FETCH";
const TILE_BYTES: usize = 4 << 20;
static BATCHES: AtomicU64 = AtomicU64::new(0);
static ROWS: AtomicU64 = AtomicU64::new(0);
static READS: AtomicU64 = AtomicU64::new(0);
static BYTES: AtomicU64 = AtomicU64::new(0);
static NANOS: AtomicU64 = AtomicU64::new(0);
static SERIAL: AtomicU64 = AtomicU64::new(0);
static MAPPED: AtomicU64 = AtomicU64::new(0);

fn configured(value: Option<std::ffi::OsString>) -> bool {
    value.is_none_or(|value| value == "1")
}

pub(crate) fn enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| configured(std::env::var_os(ENV)))
}

pub(crate) fn snapshot() -> serde_json::Value {
    serde_json::json!({
        "enabled":enabled(), "configured_workers":pool::WORKERS, "batches":BATCHES.load(Ordering::Relaxed),
        "miss_positions":ROWS.load(Ordering::Relaxed), "unique_reads":READS.load(Ordering::Relaxed),
        "successful_bytes":BYTES.load(Ordering::Relaxed), "nanoseconds":NANOS.load(Ordering::Relaxed),
        "serial_fallback_batches":SERIAL.load(Ordering::Relaxed),
        "mapped_fallback_rows":MAPPED.load(Ordering::Relaxed),
    })
}

fn validate(mapping: &MappedRowsV1, width: usize, rows: &[u32], out: &[u8]) -> Result<(), String> {
    if width == 0
        || !mapping.len().is_multiple_of(width as u64)
        || rows.len().checked_mul(width) != Some(out.len())
        || out.len() > 256 << 20
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
        NANOS.fetch_add(
            started.elapsed().as_nanos().min(u128::from(u64::MAX)) as u64,
            Ordering::Relaxed,
        );
        return Ok(());
    };
    let tile_rows = (TILE_BYTES / width).clamp(1, 4096);
    for tile in misses.chunks(tile_rows) {
        let mut index = BTreeMap::new();
        let mut requests = Vec::new();
        let mut positions = Vec::with_capacity(tile.len());
        for &position in tile {
            let row = rows[position];
            let slot = *index.entry(row).or_insert_with(|| {
                let slot = requests.len();
                // validate established row * width < mapping length, and the
                // mapping constructor checked its whole file interval.
                requests.push((mapping.file_offset() + u64::from(row) * width as u64, width));
                slot
            });
            positions.push(slot);
        }
        let (data, serial) = pool::read(source, &requests)?;
        READS.fetch_add(requests.len() as u64, Ordering::Relaxed);
        BYTES.fetch_add((requests.len() * width) as u64, Ordering::Relaxed);
        if serial {
            SERIAL.fetch_add(1, Ordering::Relaxed);
        }
        // Completion order never changes publication order.
        for (&position, slot) in tile.iter().zip(positions) {
            let bytes = &data[slot];
            out[position * width..][..width].copy_from_slice(bytes);
            if let Some(plane) = mapping.resident_plane(width) {
                plane.publish_fetched(rows[position], bytes);
            }
        }
    }
    NANOS.fetch_add(
        started.elapsed().as_nanos().min(u128::from(u64::MAX)) as u64,
        Ordering::Relaxed,
    );
    Ok(())
}

#[cfg(test)]
mod tests;
