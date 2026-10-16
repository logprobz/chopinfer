//! Owned pinned pages on a dedicated stream. No host slice escapes while DMA
//! can access the arena. At most 128 bounded chunks may be outstanding (at most 1 GiB per rank).
use super::*;
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PageCopyDirection {
    ToHost,
    ToDevice,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PageCopyPair {
    pub device_page: usize,
    pub host_page: usize,
}
struct Pending {
    event: Event,
    pages: Vec<PageCopyPair>,
}
pub struct PageCopyArena {
    buffers: Vec<PinnedHostBuffer>,
    device: DeviceBuffer,
    stream: Stream,
    page_bytes: usize,
    pages_per_buffer: usize,
    capacity: usize,
    pending: BTreeMap<u64, Pending>,
    next: u64,
    poisoned: bool,
}
impl PageCopyArena {
    pub fn new(
        device: DeviceBuffer,
        page_bytes: usize,
        capacity: usize,
    ) -> Result<Self, CudaError> {
        if page_bytes == 0 || page_bytes > 1024 * 1024 * 1024 || capacity == 0 {
            return Err(CudaError::InvalidLaunch);
        }
        let pages_per_buffer = (1024 * 1024 * 1024) / page_bytes;
        let context = device.owner_context();
        let stream = context.stream()?;
        let mut buffers = Vec::new();
        let mut remaining = capacity;
        while remaining != 0 {
            let pages = remaining.min(pages_per_buffer);
            buffers.push(PinnedHostBuffer::new(&context, pages * page_bytes)?);
            remaining -= pages;
        }
        Ok(Self {
            buffers,
            device,
            stream,
            page_bytes,
            pages_per_buffer,
            capacity,
            pending: BTreeMap::new(),
            next: 1,
            poisoned: false,
        })
    }
    pub fn submit(
        &mut self,
        direction: PageCopyDirection,
        pages: &[PageCopyPair],
    ) -> Result<u64, CudaError> {
        if self.poisoned
            || self.pending.len() >= 128
            || self
                .pending
                .values()
                .map(|p| p.pages.len())
                .sum::<usize>()
                .saturating_add(pages.len())
                .saturating_mul(self.page_bytes)
                > 1024 * 1024 * 1024
            || pages.is_empty()
            || pages.len() > (8 * 1024 * 1024 / self.page_bytes).max(1)
        {
            return Err(CudaError::InvalidLaunch);
        }
        let next = self.next.checked_add(1).ok_or(CudaError::InvalidLaunch)?;
        let mut seen_host = BTreeSet::new();
        let mut seen_device = BTreeSet::new();
        let mut destinations = Vec::new();
        let mut sources = Vec::new();
        self.device.require_live()?;
        self.stream.require_live()?;
        for pair in pages {
            if pair.host_page >= self.capacity
                || !seen_host.insert(pair.host_page)
                || !seen_device.insert(pair.device_page)
                || self.pending.values().any(|p| {
                    p.pages.iter().any(|old| {
                        old.host_page == pair.host_page || old.device_page == pair.device_page
                    })
                })
            {
                return Err(CudaError::InvalidLaunch);
            }
            let offset = pair
                .device_page
                .checked_mul(self.page_bytes)
                .ok_or(CudaError::InvalidLaunch)?;
            let device = self.device.pointer_at(offset, self.page_bytes)?;
            let host = ffi::pinned_address(
                &self.buffers[pair.host_page / self.pages_per_buffer].allocation,
                (pair.host_page % self.pages_per_buffer) * self.page_bytes,
            );
            let (dst, src) = match direction {
                PageCopyDirection::ToHost => (host, device),
                PageCopyDirection::ToDevice => (device, host),
            };
            destinations.push(dst);
            sources.push(src);
        }
        let event = self.device.owner_context().event()?;
        let mut sizes = vec![self.page_bytes; pages.len()];
        set_current_inner(&self.stream.inner.context)?;
        // Track ownership even when submission or event recording fails: a
        // driver may have accepted part of the list before reporting failure.
        let id = self.next;
        self.next = next;
        self.pending.insert(
            id,
            Pending {
                event,
                pages: pages.to_vec(),
            },
        );
        let result = checked(
            "cuMemcpyBatchAsync",
            ffi::copy_pages(
                &mut destinations,
                &mut sources,
                &mut sizes,
                self.stream.inner.handle,
            ),
        )
        .and_then(|()| self.pending[&id].event.record(&self.stream));
        if let Err(error) = result {
            self.poisoned = true;
            if self.stream.synchronize().is_ok() {
                self.pending.clear();
            }
            return Err(error);
        }
        Ok(id)
    }
    /// Removes only a completed event; false leaves ownership intact.
    pub fn query(&mut self, id: u64) -> Result<bool, CudaError> {
        if self.poisoned {
            return Err(CudaError::InvalidLaunch);
        }
        let pending = self.pending.get(&id).ok_or(CudaError::InvalidLaunch)?;
        match pending.event.query() {
            Ok(false) => return Ok(false),
            Ok(true) => {}
            Err(e) => {
                self.poisoned = true;
                return Err(e);
            }
        }
        self.pending.remove(&id);
        Ok(true)
    }
    pub fn wait(&mut self, id: u64) -> Result<(), CudaError> {
        if self.poisoned {
            return Err(CudaError::InvalidLaunch);
        }
        self.pending
            .get(&id)
            .ok_or(CudaError::InvalidLaunch)?
            .event
            .synchronize()?;
        self.pending.remove(&id);
        Ok(())
    }
    /// A successful fence proves every submitted copy idle, including a
    /// partially submitted failed list. Poisoned arenas remain unavailable.
    pub fn drain(&mut self) -> Result<(), CudaError> {
        if self.pending.is_empty() {
            return Ok(());
        }
        self.stream.synchronize()?;
        self.pending.clear();
        Ok(())
    }
}
impl Drop for PageCopyArena {
    fn drop(&mut self) {
        if self.drain().is_err() {
            eprintln!(
                "runtime-cuda: could not fence KV copies; retaining pinned and device allocations until process exit"
            );
            std::mem::forget(std::mem::take(&mut self.buffers));
            std::mem::forget(std::mem::take(&mut self.pending));
            std::mem::forget(self.device.clone());
            std::mem::forget(self.stream.clone());
        }
    }
}

#[cfg(all(test, feature = "fake-driver-tests"))]
mod tests {
    use super::*;
    fn arena() -> PageCopyArena {
        ffi::fake_driver_reset();
        let context = crate::tests::live_fake_context(0, 0x8200);
        let stream = context.stream().unwrap();
        let device = context
            .default_memory_pool()
            .unwrap()
            .allocate(&stream, 64)
            .unwrap();
        PageCopyArena::new(device, 16, 4).unwrap()
    }
    fn pair() -> [PageCopyPair; 1] {
        [PageCopyPair {
            device_page: 0,
            host_page: 0,
        }]
    }
    #[test]
    fn failed_batch_and_event_record_fence_before_releasing_ownership() {
        for operation in [29, 30] {
            let mut a = arena();
            ffi::fake_driver_fail_next(operation, 999);
            assert!(a.submit(PageCopyDirection::ToHost, &pair()).is_err());
            assert_eq!(ffi::fake_driver_call_count(5), 1);
            assert!(a.pending.is_empty());
            assert!(a.submit(PageCopyDirection::ToHost, &pair()).is_err());
            drop(a);
            assert_eq!(ffi::fake_driver_call_count(5), 1);
            assert_eq!(ffi::fake_driver_call_count(17), 1);
        }
    }
    #[test]
    fn query_errors_retain_pages_until_a_successful_fence() {
        let mut a = arena();
        let id = a.submit(PageCopyDirection::ToHost, &pair()).unwrap();
        ffi::fake_driver_fail_next(31, 999);
        assert!(a.query(id).is_err());
        assert_eq!(a.pending.len(), 1);
        ffi::fake_driver_fail_next(5, 999);
        assert!(a.drain().is_err());
        assert_eq!(a.pending.len(), 1);
        a.drain().unwrap();
        assert!(a.pending.is_empty());
        assert!(a.submit(PageCopyDirection::ToHost, &pair()).is_err());
    }
    #[test]
    fn drop_fences_pending_copies_but_does_not_fence_an_empty_arena() {
        let mut a = arena();
        a.submit(PageCopyDirection::ToHost, &pair()).unwrap();
        drop(a);
        assert_eq!(ffi::fake_driver_call_count(5), 1);
        assert_eq!(ffi::fake_driver_call_count(17), 1);
        let a = arena();
        ffi::fake_driver_fail_next(5, 999);
        drop(a);
        assert_eq!(ffi::fake_driver_call_count(5), 0);
        assert_eq!(ffi::fake_driver_call_count(17), 1);
        ffi::fake_driver_reset();
    }
    #[test]
    fn failed_drop_fence_retains_every_dma_owner_without_aborting() {
        let mut a = arena();
        a.submit(PageCopyDirection::ToHost, &pair()).unwrap();
        ffi::fake_driver_fail_next(5, 999);
        drop(a);
        assert_eq!(ffi::fake_driver_call_count(5), 1);
        assert_eq!(ffi::fake_driver_call_count(17), 0);
        assert_eq!(ffi::fake_driver_call_count(4), 0);
        assert_eq!(ffi::fake_driver_call_count(2), 0);
    }
    #[test]
    fn page_lists_round_trip_and_query_without_early_reuse() {
        let context = crate::tests::live_fake_context(0, 0x8100);
        let stream = context.stream().unwrap();
        let pool = context.default_memory_pool().unwrap();
        let device = pool.allocate(&stream, 64).unwrap();
        device
            .copy_from_bytes(&(0..64u8).collect::<Vec<_>>(), &stream)
            .unwrap();
        let mut arena = PageCopyArena::new(device.clone(), 16, 4).unwrap();
        let list = [
            PageCopyPair {
                device_page: 3,
                host_page: 0,
            },
            PageCopyPair {
                device_page: 1,
                host_page: 2,
            },
        ];
        let id = arena.submit(PageCopyDirection::ToHost, &list).unwrap();
        assert!(!arena.query(id).unwrap());
        assert!(arena.submit(PageCopyDirection::ToDevice, &list).is_err());
        assert!(arena.query(id).unwrap());
        let back = [
            PageCopyPair {
                device_page: 0,
                host_page: 0,
            },
            PageCopyPair {
                device_page: 2,
                host_page: 2,
            },
        ];
        let id = arena.submit(PageCopyDirection::ToDevice, &back).unwrap();
        arena.wait(id).unwrap();
        let mut bytes = [0; 64];
        device
            .copy_to_bytes_at_async(0, &mut bytes, &stream)
            .unwrap();
        stream.synchronize().unwrap();
        assert_eq!(&bytes[..16], &(48..64u8).collect::<Vec<_>>());
        assert_eq!(&bytes[32..48], &(16..32u8).collect::<Vec<_>>());
        assert!(
            arena
                .submit(
                    PageCopyDirection::ToHost,
                    &[PageCopyPair {
                        device_page: 4,
                        host_page: 0
                    }]
                )
                .is_err()
        );
        assert!(
            arena
                .submit(
                    PageCopyDirection::ToHost,
                    &[PageCopyPair {
                        device_page: 0,
                        host_page: 4
                    }]
                )
                .is_err()
        );
        assert!(arena.query(id).is_err());
    }
}
