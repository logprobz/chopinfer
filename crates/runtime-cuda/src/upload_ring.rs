//! Host-to-device copies that return once the source bytes are staged.
//!
//! A [`PinnedUploadRing`] copies each upload into its own page-locked ring,
//! starts the DMA on the caller's stream and records an event behind it. The
//! host waits only when a new upload needs ring space an earlier one still
//! owns. Each copy keeps its destination allocation and stream alive until it
//! is known to be complete, and dropping the ring waits for every copy.

use std::collections::VecDeque;

/// Copies kept in flight before the oldest is waited for. Small uploads would
/// otherwise hold an event and a destination each until the ring wraps.
const MAX_IN_FLIGHT: usize = 64;

use crate::{
    CudaError, DeviceBuffer, Event, PinnedHostBuffer, Stream, checked, ffi, require_same_context,
    set_current_inner,
};

/// Where uploads sit in a ring of `capacity` bytes, apart from the driver.
#[derive(Debug)]
struct Regions {
    capacity: usize,
    head: usize,
    /// Byte ranges still owned by copies, oldest first.
    live: VecDeque<(usize, usize)>,
}

impl Regions {
    fn new(capacity: usize) -> Self {
        Self {
            capacity,
            head: 0,
            live: VecDeque::new(),
        }
    }

    /// The start of the next `bytes`-byte region, and how many of the oldest
    /// live regions must complete before it may be written. An upload never
    /// straddles the end of the ring.
    fn place(&self, bytes: usize) -> (usize, usize) {
        let start = if self.head + bytes > self.capacity {
            0
        } else {
            self.head
        };
        let end = start + bytes;
        let overlapping = self
            .live
            .iter()
            .rposition(|&(first, last)| first < end && start < last)
            .map_or(0, |index| index + 1);
        (start, overlapping)
    }

    fn retire_oldest(&mut self) {
        self.live.pop_front();
    }

    fn occupy(&mut self, start: usize, bytes: usize) {
        self.live.push_back((start, start + bytes));
        self.head = start + bytes;
    }
}

struct InFlight {
    event: Event,
    _destination: DeviceBuffer,
    _stream: Stream,
}

/// Stream-ordered host-to-device copies staged through page-locked memory.
pub struct PinnedUploadRing {
    staging: PinnedHostBuffer,
    regions: Regions,
    in_flight: VecDeque<InFlight>,
    spare_events: Vec<Event>,
}

impl PinnedUploadRing {
    /// Stages uploads through `staging`, which the ring owns from now on.
    #[must_use]
    pub fn new(staging: PinnedHostBuffer) -> Self {
        let capacity = staging.len();
        Self {
            staging,
            regions: Regions::new(capacity),
            in_flight: VecDeque::new(),
            spare_events: Vec::new(),
        }
    }

    /// Copies `source` to `destination` at `offset`, ordered on `stream` after
    /// its earlier work. The host is done reading `source` when this returns;
    /// the device copy completes later on the stream.
    pub fn upload(
        &mut self,
        destination: &DeviceBuffer,
        offset: usize,
        source: &[u8],
        stream: &Stream,
    ) -> Result<(), CudaError> {
        self.upload_with(
            destination,
            offset,
            source.len(),
            stream,
            &mut |at, chunk| {
                chunk.copy_from_slice(&source[at..at + chunk.len()]);
            },
        )
    }

    /// Copies `length` bytes that `fill` writes to `destination` at `offset`,
    /// ordered on `stream` after its earlier work, as [`Self::upload`] does:
    /// `fill(at, chunk)` writes bytes `at..at + chunk.len()` of the upload
    /// straight into the ring, chunk by chunk in order, before this returns.
    pub fn upload_with(
        &mut self,
        destination: &DeviceBuffer,
        offset: usize,
        length: usize,
        stream: &Stream,
        fill: &mut dyn FnMut(usize, &mut [u8]),
    ) -> Result<(), CudaError> {
        require_same_context(&destination.stream.context, &self.staging.context)?;
        require_same_context(&destination.stream.context, &stream.inner.context)?;
        stream.require_live()?;
        destination.pointer_at(offset, length)?;
        let capacity = self.regions.capacity;
        for first in (0..length).step_by(capacity) {
            let bytes = capacity.min(length - first);
            let at = offset + first;
            let (start, waits) = self.regions.place(bytes);
            for _ in 0..waits.max((self.in_flight.len() + 1).saturating_sub(MAX_IN_FLIGHT)) {
                self.retire_oldest();
            }
            fill(
                first,
                &mut self.staging.as_mut_slice()[start..start + bytes],
            );
            let pointer = destination.pointer_at(at, bytes)?;
            set_current_inner(&stream.inner.context)?;
            checked(
                "cuMemcpyHtoDAsync",
                ffi::copy_pinned_to_device(
                    pointer,
                    &self.staging.allocation,
                    start,
                    bytes,
                    stream.inner.handle,
                ),
            )?;
            let event = match self.spare_events.pop() {
                Some(event) => Ok(event),
                None => stream.context().event(),
            };
            let recorded = event.and_then(|event| event.record(stream).map(|()| event));
            let Ok(event) = recorded else {
                // The copy is queued with nothing to wait on, so the stream
                // itself has to finish before the region can be reused.
                if stream.synchronize().is_err() {
                    std::process::abort();
                }
                return recorded.map(|_| ());
            };
            self.regions.occupy(start, bytes);
            self.in_flight.push_back(InFlight {
                event,
                _destination: destination.clone(),
                _stream: stream.clone(),
            });
        }
        Ok(())
    }

    /// The page-locked bytes the ring holds.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.regions.capacity
    }

    /// Waits for every copy still in flight and lets go of what they held.
    pub fn synchronize(&mut self) {
        while !self.in_flight.is_empty() {
            self.retire_oldest();
        }
    }

    /// Waits for the oldest copy. Its staging bytes are reused next, so a copy
    /// whose completion cannot be established stops the process.
    fn retire_oldest(&mut self) {
        if let Some(copy) = self.in_flight.pop_front() {
            if copy.event.synchronize().is_err() {
                std::process::abort();
            }
            self.spare_events.push(copy.event);
        }
        self.regions.retire_oldest();
    }
}

impl Drop for PinnedUploadRing {
    fn drop(&mut self) {
        self.synchronize();
    }
}

#[cfg(test)]
mod tests {
    use super::Regions;

    #[test]
    fn regions_wait_only_for_overlapping_copies() {
        let mut regions = Regions::new(100);
        assert_eq!(regions.place(40), (0, 0));
        regions.occupy(0, 40);
        assert_eq!(regions.place(40), (40, 0));
        regions.occupy(40, 40);
        // 30 more bytes do not fit after 80, so they wrap onto the first copy.
        assert_eq!(regions.place(30), (0, 1));
        regions.retire_oldest();
        regions.occupy(0, 30);
        assert_eq!(regions.place(10), (30, 0));
        regions.occupy(30, 10);
        // [40, 80) is still the oldest live copy; [30, 40) is newer.
        assert_eq!(regions.place(15), (40, 1));
    }

    #[test]
    fn a_wrap_waits_for_every_older_copy_up_to_the_last_overlap() {
        let mut regions = Regions::new(100);
        regions.occupy(0, 90);
        regions.retire_oldest();
        regions.occupy(90, 10);
        regions.occupy(0, 60);
        assert_eq!(regions.place(30), (60, 0));
        regions.occupy(60, 30);
        // Wrapped: [90, 100) is oldest and does not overlap [0, 20), but the
        // newer [0, 60) does, so both are waited for, oldest first.
        assert_eq!(regions.place(20), (0, 2));
    }
}
