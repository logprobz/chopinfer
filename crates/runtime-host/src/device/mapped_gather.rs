//! Checked plans for event-ordered gathers of host-mapped rows.
use std::sync::Arc;

use super::{DeviceObjectV1, DeviceRangeV1, HostDeviceResultV1};
use crate::host_rows::MappedRowsV1;
use crate::host_rows::gather::{GatherProbe, MappedProbe, ReadAhead, gather_span_v1};
use runtime_bundle_abi::BundleStatusV1;

/// One mapped plane. Its input indices are U32 little-endian words.
#[derive(Clone)]
pub struct MappedGatherPlaneV1 {
    pub mapping: Arc<MappedRowsV1>,
    pub row_bytes: usize,
    pub index_offset: usize,
    pub stream: DeviceObjectV1,
    pub destination: DeviceRangeV1,
}

/// Every capacity is fixed at preparation; enqueue changes only the live rows.
#[derive(Clone)]
pub struct MappedGatherPlanV1 {
    pub producer: DeviceObjectV1,
    pub indices: DeviceRangeV1,
    pub planes: Vec<MappedGatherPlaneV1>,
}

impl MappedGatherPlanV1 {
    /// Validate capacities before any pinned allocation or device operation.
    pub fn validate(&self) -> HostDeviceResultV1<()> {
        let invalid = BundleStatusV1::InvalidArgument;
        if self.planes.is_empty()
            || self.planes.len() > 256
            || self.indices.length == 0
            || !self.indices.length.is_multiple_of(4)
            || !self.indices.offset.is_multiple_of(4)
        {
            return Err(invalid);
        }
        let mut pinned = self.indices.length;
        for (index, plane) in self.planes.iter().enumerate() {
            if plane.row_bytes == 0
                || !plane.mapping.bytes().len().is_multiple_of(plane.row_bytes)
                || plane.mapping.bytes().is_empty()
                || !plane.index_offset.is_multiple_of(4)
                || plane.destination.length == 0
                || !plane
                    .destination
                    .length
                    .is_multiple_of(plane.row_bytes as u64)
            {
                return Err(invalid);
            }
            let rows = plane.destination.length / plane.row_bytes as u64;
            if (plane.index_offset as u64)
                .checked_add(rows.checked_mul(4).ok_or(invalid)?)
                .is_none_or(|end| end > self.indices.length)
            {
                return Err(invalid);
            }
            let overlaps = |a: DeviceRangeV1, b: DeviceRangeV1| -> HostDeviceResultV1<bool> {
                let a_end = a.offset.checked_add(a.length).ok_or(invalid)?;
                let b_end = b.offset.checked_add(b.length).ok_or(invalid)?;
                Ok(a.object == b.object && a.offset < b_end && b.offset < a_end)
            };
            if overlaps(self.indices, plane.destination)? {
                return Err(invalid);
            }
            for earlier in &self.planes[..index] {
                if overlaps(earlier.destination, plane.destination)? {
                    return Err(invalid);
                }
            }
            pinned = pinned
                .checked_add(plane.destination.length)
                .ok_or(invalid)?;
        }
        // A prepared slot pins its bounded working set, never the mapped tables.
        if pinned > 256 * 1024 * 1024 {
            return Err(BundleStatusV1::ResourceExhausted);
        }
        Ok(())
    }

    pub fn validate_rows(&self, rows: u64) -> HostDeviceResultV1<()> {
        if rows == 0
            || self
                .planes
                .iter()
                .any(|p| rows > p.destination.length / p.row_bytes as u64)
        {
            return Err(BundleStatusV1::InvalidArgument);
        }
        Ok(())
    }
}

/// Gather raw bytes only. Validate every index before writing any output;
/// device-supplied indices cannot become unchecked mmap accesses.
pub fn gather_plane_v1(
    plane: &MappedGatherPlaneV1,
    input: &[u8],
    output: &mut [u8],
) -> Result<(), String> {
    // Apply the complete gather's cache policy before queued resident lookups.
    // Invalid widths still reach the checked mapped path and are refused.
    if crate::host_rows::fetch::enabled()
        && plane.row_bytes != 0
        && crate::host_rows::gather::use_resident(output.len() / plane.row_bytes)
    {
        let ids = checked_ids(plane, input, output.len())?;
        // Keep the existing gather counters. Queued descriptor reads supply
        // the misses directly, so this path needs no mmap read-ahead hint.
        let _ = crate::host_rows::gather::mapped_async_read_ahead_v1();
        return crate::host_rows::fetch::copy_rows(&plane.mapping, plane.row_bytes, &ids, output);
    }
    let probe = MappedProbe::asynchronous(&plane.mapping, plane.row_bytes);
    crate::host_rows::fetch::legacy::observe(|| {
        let result = gather_plane_with_probe(
            plane,
            input,
            output,
            &probe,
            crate::host_rows::gather::mapped_async_read_ahead_v1(),
        );
        if result.is_ok() && plane.mapping.resident_plane(plane.row_bytes).is_none() {
            crate::host_rows::fetch::legacy::record((output.len() / plane.row_bytes) as u64);
        }
        result
    })
}

fn gather_plane_with_probe(
    plane: &MappedGatherPlaneV1,
    input: &[u8],
    output: &mut [u8],
    probe: &dyn GatherProbe,
    read_ahead: bool,
) -> Result<(), String> {
    let ids = checked_ids(plane, input, output.len())?;
    let table = plane.mapping.bytes();
    let ahead = if read_ahead {
        ReadAhead::new(probe)
    } else {
        ReadAhead::disabled(probe)
    };
    let ahead = ahead.with_resident(plane.mapping.resident_plane(plane.row_bytes));
    gather_span_v1(table, plane.row_bytes, &ids, 0, output, &ahead);
    Ok(())
}

fn checked_ids(
    plane: &MappedGatherPlaneV1,
    input: &[u8],
    output_length: usize,
) -> Result<Vec<u32>, String> {
    if plane.row_bytes == 0 || !output_length.is_multiple_of(plane.row_bytes) {
        return Err("mapped gather output differs from its row width".into());
    }
    let rows = output_length / plane.row_bytes;
    let bytes = rows
        .checked_mul(4)
        .ok_or("mapped gather index length overflows")?;
    let indices = input
        .get(
            plane.index_offset
                ..plane
                    .index_offset
                    .checked_add(bytes)
                    .ok_or("mapped gather input overflows")?,
        )
        .ok_or("mapped gather input is truncated")?;
    let table = plane.mapping.bytes();
    let limit = table.len() / plane.row_bytes;
    let ids: Vec<u32> = indices
        .chunks_exact(4)
        .map(|word| u32::from_le_bytes(word.try_into().expect("U32 index")))
        .collect();
    if ids.iter().any(|&row| row as usize >= limit) {
        return Err("mapped gather device index is outside the table".into());
    }
    Ok(ids)
}

#[cfg(test)]
#[path = "mapped_gather_tests.rs"]
mod tests;
