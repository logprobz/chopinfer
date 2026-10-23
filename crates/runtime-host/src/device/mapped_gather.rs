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

/// Gathers every plane of one prepared gather, `outputs[i]` holding plane
/// `i`'s rows. With joint plane reads (`CHOPINFER_MAPPED_MISS_JOINT_PLANES`,
/// on by default with started reads), the first plane of each mapping and
/// row width that takes the queued-read path (every gather below the
/// large-gather size) leads: the leaders' ids are all checked before any
/// write, and their misses are read as one batch, so the gather waits for
/// the device once rather than once per plane. Every other plane (another
/// rank's copy of a leader's rows, or a large gather on the mapped path)
/// then gathers on its own, as [`gather_plane_v1`] does; another rank's copy
/// finds the leaders' rows in the resident cache, or without one in the page
/// cache. The bytes equal gathering plane by plane: each is the mapped
/// table's row or a resident copy of it.
pub fn gather_planes_v1(
    planes: &[MappedGatherPlaneV1],
    input: &[u8],
    outputs: &mut [&mut [u8]],
) -> Result<(), String> {
    if planes.len() != outputs.len() {
        return Err("mapped gather outputs differ from its planes".into());
    }
    let mut leads = vec![false; planes.len()];
    if crate::host_rows::fetch::enabled() && crate::host_rows::fetch::joint_planes() {
        for (index, plane) in planes.iter().enumerate() {
            let served = plane.row_bytes != 0
                && crate::host_rows::gather::use_resident(outputs[index].len() / plane.row_bytes);
            let first = !planes[..index].iter().zip(&leads).any(|(earlier, &led)| {
                led && earlier.row_bytes == plane.row_bytes
                    && Arc::ptr_eq(&earlier.mapping, &plane.mapping)
            });
            leads[index] = served && first;
        }
    }
    // One leader gains nothing from a joint read.
    if leads.iter().filter(|&&led| led).count() < 2 {
        leads.fill(false);
    } else {
        let ids = planes
            .iter()
            .zip(outputs.iter())
            .zip(&leads)
            .filter(|(_, led)| **led)
            .map(|((plane, output), _)| checked_ids(plane, input, output.len()))
            .collect::<Result<Vec<_>, _>>()?;
        let mut copies = Vec::with_capacity(ids.len());
        for (((plane, output), _), ids) in planes
            .iter()
            .zip(outputs.iter_mut())
            .zip(&leads)
            .filter(|(_, led)| **led)
            .zip(&ids)
        {
            // Keep the gather counters each plane's own gather would count.
            let _ = crate::host_rows::gather::mapped_async_read_ahead_v1();
            copies.push(crate::host_rows::fetch::PlaneCopy {
                mapping: &plane.mapping,
                width: plane.row_bytes,
                rows: ids,
                out: output,
            });
        }
        crate::host_rows::fetch::copy_planes(&mut copies)?;
    }
    for ((plane, output), &led) in planes.iter().zip(outputs.iter_mut()).zip(&leads) {
        if !led {
            gather_plane_v1(plane, input, output)?;
        }
    }
    Ok(())
}

/// Each plane's output inside a gather's per-stream outputs, in plane order:
/// plane `i` writes `lengths[i]` bytes from `places[i]`, an output index and
/// a start within it. Refuses ranges that overlap or leave their output.
/// Only the CUDA backend's callback has per-stream outputs.
#[cfg_attr(
    not(all(target_os = "linux", feature = "linux-cuda-host")),
    allow(dead_code)
)]
pub fn plane_outputs<'a>(
    outputs: &'a mut [&mut [u8]],
    places: &[(usize, usize)],
    lengths: &[usize],
) -> Result<Vec<&'a mut [u8]>, String> {
    if places.len() != lengths.len() {
        return Err("mapped gather plane places differ from their lengths".into());
    }
    let mut order = (0..places.len()).collect::<Vec<_>>();
    order.sort_by_key(|&plane| places[plane]);
    // What is left of each output after the planes placed so far, and where
    // that rest starts.
    let mut rest = outputs
        .iter_mut()
        .map(|output| Some((&mut **output, 0_usize)))
        .collect::<Vec<_>>();
    let mut slices = (0..places.len()).map(|_| None).collect::<Vec<_>>();
    for plane in order {
        let (output, start) = places[plane];
        let (buffer, at) = rest
            .get_mut(output)
            .and_then(Option::take)
            .ok_or("mapped gather plane names no output")?;
        let skip = start
            .checked_sub(at)
            .filter(|&skip| skip <= buffer.len())
            .ok_or("mapped gather plane outputs overlap or leave their buffer")?;
        let (_, tail) = buffer.split_at_mut(skip);
        if tail.len() < lengths[plane] {
            return Err("mapped gather plane output leaves its buffer".into());
        }
        let (slice, after) = tail.split_at_mut(lengths[plane]);
        slices[plane] = Some(slice);
        rest[output] = Some((after, start + lengths[plane]));
    }
    Ok(slices
        .into_iter()
        .map(|slice| slice.expect("every plane placed"))
        .collect())
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
