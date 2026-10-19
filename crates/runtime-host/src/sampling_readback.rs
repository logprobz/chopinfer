//! Two retained D2H slots on a private stream. Only the validated device-result
//! service can enqueue work, and it refuses events from an open capture.
use runtime_bundle_abi::BundleStatusV1;
use runtime_cuda::{Context, DeviceBuffer, Event, PinnedHostBuffer, Stream};

type Result<T> = std::result::Result<T, BundleStatusV1>;
const MAX_PACKET_BYTES: usize = 1 << 20;

struct Slot {
    pinned: PinnedHostBuffer,
    done: Event,
    source: Option<DeviceBuffer>,
    ticket: u64,
    bytes: usize,
    recorded: bool,
}

pub(crate) struct SamplingReadback {
    // The stream is never exposed, so callers cannot begin or join capture on it.
    stream: Stream,
    slots: [Slot; 2],
    next: usize,
    next_ticket: u64,
    failed: bool,
}

impl SamplingReadback {
    #[cfg(test)]
    fn new(context: &Context) -> Result<Self> {
        Self::on_private_stream(
            context,
            context.stream().map_err(|_| BundleStatusV1::HostError)?,
        )
    }

    pub(crate) fn on_private_stream(context: &Context, stream: Stream) -> Result<Self> {
        let slot = || -> Result<Slot> {
            Ok(Slot {
                pinned: PinnedHostBuffer::new(context, MAX_PACKET_BYTES)
                    .map_err(|_| BundleStatusV1::HostError)?,
                done: context.event().map_err(|_| BundleStatusV1::HostError)?,
                source: None,
                ticket: 0,
                bytes: 0,
                recorded: false,
            })
        };
        Ok(Self {
            stream,
            slots: [slot()?, slot()?],
            next: 0,
            next_ticket: 1,
            failed: false,
        })
    }

    pub(crate) fn enqueue(
        &mut self,
        source: &DeviceBuffer,
        offset: usize,
        bytes: usize,
        ready: &Event,
    ) -> Result<u64> {
        if self.failed {
            return Err(BundleStatusV1::InvalidState);
        }
        if bytes == 0 || bytes > MAX_PACKET_BYTES {
            return Err(BundleStatusV1::InvalidArgument);
        }
        source
            .device_address_at(offset, bytes)
            .map_err(|_| BundleStatusV1::InvalidArgument)?;
        let slot = &mut self.slots[self.next];
        if slot.source.is_some() {
            return Err(BundleStatusV1::ResourceExhausted);
        }
        let next_ticket = self
            .next_ticket
            .checked_add(1)
            .ok_or(BundleStatusV1::ResourceExhausted)?;
        // Retain before submitting. Even an error after a partial enqueue keeps
        // every pointer live until cancel/drop establishes a successful fence.
        slot.source = Some(source.clone());
        slot.ticket = self.next_ticket;
        slot.bytes = bytes;
        slot.recorded = false;
        let result = self
            .stream
            .wait_event(ready)
            .and_then(|()| {
                source.copy_to_bytes_at_async(
                    offset,
                    &mut slot.pinned.as_mut_slice()[..bytes],
                    &self.stream,
                )
            })
            .and_then(|()| slot.done.record(&self.stream));
        if result.is_err() {
            self.failed = true;
            return Err(BundleStatusV1::HostError);
        }
        slot.recorded = true;
        self.next = (self.next + 1) % 2;
        let ticket = self.next_ticket;
        self.next_ticket = next_ticket;
        Ok(ticket)
    }

    pub(crate) fn collect(&mut self, ticket: u64) -> Result<Vec<u8>> {
        if self.failed {
            return Err(BundleStatusV1::InvalidState);
        }
        let slot = self
            .slots
            .iter_mut()
            .find(|slot| slot.source.is_some() && slot.ticket == ticket)
            .ok_or(BundleStatusV1::StaleHandle)?;
        if !slot.recorded || slot.done.synchronize().is_err() {
            self.failed = true;
            return Err(BundleStatusV1::HostError);
        }
        let bytes = slot.pinned.as_slice()[..slot.bytes].to_vec();
        slot.source = None;
        Ok(bytes)
    }

    pub(crate) fn discard(&mut self, ticket: u64) -> Result<()> {
        let index = self
            .slots
            .iter()
            .position(|slot| slot.source.is_some() && slot.ticket == ticket)
            .ok_or(BundleStatusV1::StaleHandle)?;
        self.stream
            .synchronize()
            .map_err(|_| BundleStatusV1::HostError)?;
        self.slots[index].source = None;
        Ok(())
    }

    pub(crate) fn cancel(&mut self) -> Result<()> {
        if self.slots.iter().any(|slot| slot.source.is_some()) {
            self.stream
                .synchronize()
                .map_err(|_| BundleStatusV1::HostError)?;
            for slot in &mut self.slots {
                slot.source = None;
            }
        }
        Ok(())
    }
}

impl Drop for SamplingReadback {
    fn drop(&mut self) {
        if self.cancel().is_err() && self.cancel().is_err() {
            // DMA ownership cannot end without a confirmed private-stream fence.
            std::process::abort();
        }
    }
}

#[cfg(all(test, feature = "cuda-fake-driver-tests"))]
mod tests {
    use super::*;
    use runtime_cuda::{Driver, fake_driver};
    fn fixture() -> (Context, Stream, DeviceBuffer, Event) {
        fake_driver::reset();
        let context = Driver::initialize()
            .unwrap()
            .retain_primary_context(0)
            .unwrap();
        let stream = context.stream().unwrap();
        let source = context
            .default_memory_pool()
            .unwrap()
            .allocate(&stream, 64)
            .unwrap();
        source.copy_from_bytes_at(0, &[17; 64], &stream).unwrap();
        let ready = context.event().unwrap();
        ready.record(&stream).unwrap();
        (context, stream, source, ready)
    }
    #[test]
    fn alternates_without_overwriting_in_flight_or_accepting_stale_tickets() {
        let (context, _producer, source, ready) = fixture();
        let mut readback = SamplingReadback::new(&context).unwrap();
        let first = readback.enqueue(&source, 0, 16, &ready).unwrap();
        let second = readback.enqueue(&source, 16, 32, &ready).unwrap();
        assert_eq!(
            readback.enqueue(&source, 0, 16, &ready),
            Err(BundleStatusV1::ResourceExhausted)
        );
        assert_eq!(readback.collect(first).unwrap(), vec![17; 16]);
        let third = readback.enqueue(&source, 0, 16, &ready).unwrap();
        assert_eq!(readback.collect(first), Err(BundleStatusV1::StaleHandle));
        assert_eq!(readback.collect(second).unwrap(), vec![17; 32]);
        assert_eq!(readback.collect(third).unwrap(), vec![17; 16]);
        assert_eq!(fake_driver::ordering_violations(), 0);
    }
    #[test]
    fn partial_copy_and_event_failures_hold_sources_until_cancel() {
        for operation in [10, 30] {
            let (context, _producer, source, ready) = fixture();
            let mut readback = SamplingReadback::new(&context).unwrap();
            fake_driver::fail_next(operation, 719);
            assert_eq!(
                readback.enqueue(&source, 0, 16, &ready),
                Err(BundleStatusV1::HostError)
            );
            assert!(
                source.release().is_err(),
                "pending readback must retain its source"
            );
            assert_eq!(readback.collect(1), Err(BundleStatusV1::InvalidState));
            readback.cancel().unwrap();
            source.release().unwrap();
            assert_eq!(fake_driver::ordering_violations(), 0);
        }
    }

    #[test]
    fn failed_collection_and_cancel_retry_cannot_publish_or_release_dma_storage() {
        let (context, _producer, source, ready) = fixture();
        let mut readback = SamplingReadback::new(&context).unwrap();
        let ticket = readback.enqueue(&source, 0, 16, &ready).unwrap();
        fake_driver::fail_next(1, 201);
        assert_eq!(readback.collect(ticket), Err(BundleStatusV1::HostError));
        fake_driver::fail_next(5, 719);
        assert_eq!(readback.cancel(), Err(BundleStatusV1::HostError));
        assert!(source.release().is_err());
        readback.cancel().unwrap();
        assert_eq!(readback.collect(ticket), Err(BundleStatusV1::InvalidState));
        source.release().unwrap();
    }

    #[test]
    fn cancellation_after_two_submissions_and_drop_retry_fence_before_free() {
        let (context, _producer, source, ready) = fixture();
        let mut readback = SamplingReadback::new(&context).unwrap();
        readback.enqueue(&source, 0, 16, &ready).unwrap();
        readback.enqueue(&source, 16, 16, &ready).unwrap();
        let before = fake_driver::call_count(5);
        fake_driver::fail_next(5, 719);
        drop(readback);
        assert_eq!(fake_driver::call_count(5), before + 2);
        source.release().unwrap();
    }
}
