//! Exclusive pinned byte ranges that CPU producers can return for DMA.

use crate::{
    CudaError, DeviceBuffer, Event, PinnedHostBuffer, Stream, checked, checked_host_range, ffi,
    require_same_context, set_current_inner,
};
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, Ordering};

/// An exclusive borrow of pinned bytes, movable to a CPU worker. The CUDA
/// allocation and its context stay on their owning thread. Only the owner
/// can construct this range, and its borrow keeps the allocation live.
pub struct PinnedChunk<'a> {
    bytes: &'a mut [u8],
    context: usize,
    in_flight: &'a AtomicBool,
}

impl PinnedHostBuffer {
    /// Lends the bytes until the producer and any subsequent DMA return them.
    pub fn lend(&mut self) -> PinnedChunk<'_> {
        self.require_no_pending_chunk();
        let context = Rc::as_ptr(&self.context) as usize;
        PinnedChunk {
            bytes: self.allocation.as_mut_slice(),
            context,
            in_flight: &self.chunk_in_flight,
        }
    }
}

impl PinnedChunk<'_> {
    #[must_use]
    pub fn len(&self) -> usize {
        self.bytes.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        self.bytes
    }
}

/// Owns a chunk's exclusive borrow and both CUDA resources until its copy
/// completes. Dropping a pending transfer drains the stream before releasing
/// the borrow, including failure to allocate or record its completion event.
/// Forgetting this guard makes subsequent access to or destruction of the
/// pinned owner abort, since completion can no longer be established.
pub struct PinnedChunkTransfer<'a> {
    chunk: Option<PinnedChunk<'a>>,
    _destination: DeviceBuffer,
    stream: Stream,
    event: Option<Event>,
    start_event: Option<Event>,
}

impl DeviceBuffer {
    /// Starts a copy from the first `bytes` of `chunk` and takes its borrow
    /// until completion. The returned guard stays on the CUDA owning thread.
    pub fn upload_pinned_chunk<'a>(
        &self,
        chunk: PinnedChunk<'a>,
        offset: usize,
        bytes: usize,
        stream: &Stream,
    ) -> Result<PinnedChunkTransfer<'a>, CudaError> {
        self.upload_chunk(chunk, offset, bytes, stream, false)
    }

    /// Diagnostic variant with device timestamps around the copy. No fence
    /// is added; elapsed time is available after normal completion polling.
    pub fn upload_pinned_chunk_timed<'a>(
        &self,
        chunk: PinnedChunk<'a>,
        offset: usize,
        bytes: usize,
        stream: &Stream,
    ) -> Result<PinnedChunkTransfer<'a>, CudaError> {
        self.upload_chunk(chunk, offset, bytes, stream, true)
    }

    fn upload_chunk<'a>(
        &self,
        chunk: PinnedChunk<'a>,
        offset: usize,
        bytes: usize,
        stream: &Stream,
        timed: bool,
    ) -> Result<PinnedChunkTransfer<'a>, CudaError> {
        self.require_live()?;
        stream.require_live()?;
        require_same_context(&self.stream.context, &stream.inner.context)?;
        if chunk.context != Rc::as_ptr(&self.stream.context) as usize {
            return Err(CudaError::ContextMismatch);
        }
        let pointer = self.pointer_at(offset, bytes)?;
        checked_host_range(chunk.len(), 0, bytes)?;
        set_current_inner(&self.stream.context)?;
        let mut transfer = PinnedChunkTransfer {
            chunk: Some(chunk),
            _destination: self.clone(),
            stream: stream.clone(),
            event: None,
            start_event: None,
        };
        // Allocate both timed events before submission so their creation
        // cannot become part of the measured device interval.
        let end = if timed {
            transfer.start_event = Some(stream.context().timed_event()?);
            let end = stream.context().timed_event()?;
            transfer.start_event.as_ref().unwrap().record(stream)?;
            Some(end)
        } else {
            None
        };
        let chunk = transfer
            .chunk
            .as_ref()
            .expect("a new transfer owns its chunk");
        chunk.in_flight.store(true, Ordering::Relaxed);
        checked(
            "cuMemcpyHtoDAsync",
            ffi::copy_pinned_bytes_to_device(pointer, &chunk.bytes[..bytes], stream.inner.handle),
        )?;
        let event = match end {
            Some(event) => event,
            None => stream.context().event()?,
        };
        event.record(stream)?;
        transfer.event = Some(event);
        Ok(transfer)
    }
}

impl<'a> PinnedChunkTransfer<'a> {
    /// The interval between device timestamps, when requested. Call only
    /// after `ready` succeeds; this never synchronizes the stream.
    pub fn elapsed_seconds(&self) -> Result<Option<f64>, CudaError> {
        self.start_event
            .as_ref()
            .map(|start| {
                start
                    .elapsed_ms(
                        self.event
                            .as_ref()
                            .expect("a returned transfer has an event"),
                    )
                    .map(|ms| f64::from(ms) * 1e-3)
            })
            .transpose()
    }

    /// Polls this copy's event without waiting for other ranks or later work.
    pub fn ready(&self) -> Result<bool, CudaError> {
        self.event
            .as_ref()
            .expect("a returned transfer has an event")
            .query()
    }

    /// Waits for this copy and returns its bytes for the next CPU read.
    pub fn finish(mut self) -> PinnedChunk<'a> {
        if self
            .event
            .as_ref()
            .expect("a returned transfer has an event")
            .synchronize()
            .is_err()
        {
            std::process::abort();
        }
        let chunk = self
            .chunk
            .take()
            .expect("a pending transfer owns its chunk");
        chunk.in_flight.store(false, Ordering::Relaxed);
        chunk
    }
}

impl Drop for PinnedChunkTransfer<'_> {
    fn drop(&mut self) {
        if let Some(chunk) = &self.chunk {
            if self.stream.synchronize().is_err() {
                std::process::abort();
            }
            chunk.in_flight.store(false, Ordering::Relaxed);
        }
    }
}

#[cfg(all(test, feature = "fake-driver-tests"))]
mod tests {
    use super::*;
    use crate::{Driver, fake_driver};

    #[test]
    fn diagnostic_timestamps_preserve_copy_bytes_and_add_no_submission_fence() {
        for timed in [false, true] {
            fake_driver::reset();
            let context = Driver::initialize()
                .unwrap()
                .retain_primary_context(0)
                .unwrap();
            let stream = context.stream().unwrap();
            let destination = context
                .default_memory_pool()
                .unwrap()
                .allocate(&stream, 8)
                .unwrap();
            destination.copy_from_bytes(&[0; 8], &stream).unwrap();
            let mut staging = PinnedHostBuffer::new(&context, 4).unwrap();
            staging.as_mut_slice().copy_from_slice(&[3, 5, 7, 9]);
            let syncs = fake_driver::call_count(5);
            let transfer = if timed {
                destination.upload_pinned_chunk_timed(staging.lend(), 2, 4, &stream)
            } else {
                destination.upload_pinned_chunk(staging.lend(), 2, 4, &stream)
            }
            .unwrap();
            assert_eq!(fake_driver::call_count(5), syncs);
            while !transfer.ready().unwrap() {}
            assert_eq!(transfer.elapsed_seconds().unwrap(), timed.then_some(0.012));
            drop(transfer.finish());
            let mut bytes = [0; 8];
            destination.copy_to_bytes(&mut bytes, &stream).unwrap();
            assert_eq!(bytes, [0, 0, 3, 5, 7, 9, 0, 0]);
        }
    }

    #[test]
    fn failed_timed_uploads_release_staging_after_draining() {
        for operation in [8, 30] {
            fake_driver::reset();
            let context = Driver::initialize()
                .unwrap()
                .retain_primary_context(0)
                .unwrap();
            let stream = context.stream().unwrap();
            let destination = context
                .default_memory_pool()
                .unwrap()
                .allocate(&stream, 4)
                .unwrap();
            let mut staging = PinnedHostBuffer::new(&context, 4).unwrap();
            fake_driver::defer_host_copies(100);
            fake_driver::fail_next(operation, 999);
            assert!(
                destination
                    .upload_pinned_chunk_timed(staging.lend(), 0, 4, &stream)
                    .is_err()
            );
            assert_eq!(fake_driver::pending_host_copies(), 0);
            staging.as_mut_slice().fill(1);
        }
    }

    #[test]
    fn forgotten_transfers_prevent_pinned_owner_reuse_and_destruction() {
        use std::os::unix::process::ExitStatusExt;

        const CHILD_CASE: &str = "CHOPINFER_TEST_FORGOTTEN_CHUNK";
        if let Ok(case) = std::env::var(CHILD_CASE) {
            fake_driver::reset();
            let context = Driver::initialize()
                .unwrap()
                .retain_primary_context(0)
                .unwrap();
            let stream = context.stream().unwrap();
            let destination = context
                .default_memory_pool()
                .unwrap()
                .allocate(&stream, 4)
                .unwrap();
            let mut staging = PinnedHostBuffer::new(&context, 4).unwrap();
            staging.as_mut_slice().copy_from_slice(&[1, 2, 3, 4]);
            fake_driver::defer_host_copies(100);
            let transfer = destination
                .upload_pinned_chunk(staging.lend(), 0, 4, &stream)
                .unwrap();
            assert_eq!(fake_driver::pending_host_copies(), 1);
            std::mem::forget(transfer);
            eprintln!("forgotten transfer is still pending: {case}");
            match case.as_str() {
                "write" => staging.as_mut_slice().fill(9),
                "read" => {
                    let _ = staging.as_slice();
                }
                "length" => {
                    let _ = staging.len();
                }
                "lend" => {
                    let _ = staging.lend();
                }
                "drop" => drop(staging),
                "upload" => {
                    let _ = destination.copy_from_pinned_at_async(0, &staging, 0, 4, &stream);
                }
                "download" => {
                    let _ = destination.copy_to_pinned_at_async(0, &mut staging, 0, 4, &stream);
                }
                "ring" => {
                    let _ = crate::PinnedUploadRing::new(staging);
                }
                _ => panic!("unknown child case"),
            }
            // Avoid cleanup masking a missing check at the attempted access.
            std::process::exit(0);
        }
        for case in [
            "write", "read", "length", "lend", "drop", "upload", "download", "ring",
        ] {
            let output = std::process::Command::new("bash")
                .args(["-c", "ulimit -c 0; exec \"$@\"", "pinned-chunk-test"])
                .arg(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "upload_chunk::tests::forgotten_transfers_prevent_pinned_owner_reuse_and_destruction",
                    "--nocapture",
                ])
                .env(CHILD_CASE, case)
                .current_dir(std::env::temp_dir())
                .output()
                .unwrap();
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                stderr.contains("forgotten transfer is still pending"),
                "{case}: {stderr}"
            );
            assert_eq!(output.status.signal(), Some(6), "{case}: {stderr}");
        }
    }

    #[test]
    fn failed_submissions_and_events_release_the_owner_only_after_draining() {
        for operation in [8, 30] {
            fake_driver::reset();
            let context = Driver::initialize()
                .unwrap()
                .retain_primary_context(0)
                .unwrap();
            let stream = context.stream().unwrap();
            let destination = context
                .default_memory_pool()
                .unwrap()
                .allocate(&stream, 4)
                .unwrap();
            let mut staging = PinnedHostBuffer::new(&context, 4).unwrap();
            staging.as_mut_slice().fill(17);
            fake_driver::defer_host_copies(100);
            fake_driver::fail_next(operation, 999);
            assert!(
                destination
                    .upload_pinned_chunk(staging.lend(), 0, 4, &stream)
                    .is_err()
            );
            assert_eq!(fake_driver::pending_host_copies(), 0);
            staging.as_mut_slice().fill(9);
            if operation != 8 {
                let mut result = [0; 4];
                destination.copy_to_bytes(&mut result, &stream).unwrap();
                assert_eq!(result, [17; 4]);
            }
        }
    }

    #[test]
    fn a_cpu_worker_returns_exclusive_pinned_bytes_for_upload() {
        fake_driver::reset();
        let context = Driver::initialize()
            .unwrap()
            .retain_primary_context(0)
            .unwrap();
        let stream = context.stream().unwrap();
        let destination = context
            .default_memory_pool()
            .unwrap()
            .allocate(&stream, 8)
            .unwrap();
        destination.copy_from_bytes(&[0; 8], &stream).unwrap();
        let mut staging = PinnedHostBuffer::new(&context, 4).unwrap();
        let chunk = std::thread::scope(|scope| {
            let mut chunk = staging.lend();
            scope
                .spawn(move || {
                    chunk.as_mut_slice().copy_from_slice(&[7, 8, 9, 10]);
                    chunk
                })
                .join()
                .unwrap()
        });
        let copy = destination
            .upload_pinned_chunk(chunk, 2, 3, &stream)
            .unwrap();
        assert!(!copy.ready().unwrap());
        assert!(copy.ready().unwrap());
        let mut returned = copy.finish();
        returned.as_mut_slice().fill(99);
        let mut result = [0; 8];
        destination.copy_to_bytes(&mut result, &stream).unwrap();
        assert_eq!(result, [0, 0, 7, 8, 9, 0, 0, 0]);
    }

    #[test]
    fn chunks_check_context_and_both_ranges_before_copying() {
        fake_driver::reset();
        let driver = Driver::initialize().unwrap();
        let context = driver.retain_primary_context(0).unwrap();
        let other = driver.retain_primary_context(1).unwrap();
        let stream = context.stream().unwrap();
        let destination = context
            .default_memory_pool()
            .unwrap()
            .allocate(&stream, 8)
            .unwrap();
        let mut wrong_context = PinnedHostBuffer::new(&other, 8).unwrap();
        assert!(matches!(
            destination.upload_pinned_chunk(wrong_context.lend(), 0, 8, &stream),
            Err(CudaError::ContextMismatch)
        ));
        let mut staging = PinnedHostBuffer::new(&context, 4).unwrap();
        for (offset, count) in [(0, 5), (7, 2)] {
            assert!(matches!(
                destination.upload_pinned_chunk(staging.lend(), offset, count, &stream),
                Err(CudaError::BufferRange { .. })
            ));
        }
        assert_eq!(fake_driver::call_count(8), 0);
    }

    #[test]
    fn event_record_failure_drains_the_submitted_copy() {
        fake_driver::reset();
        let context = Driver::initialize()
            .unwrap()
            .retain_primary_context(0)
            .unwrap();
        let stream = context.stream().unwrap();
        let destination = context
            .default_memory_pool()
            .unwrap()
            .allocate(&stream, 4)
            .unwrap();
        let mut staging = PinnedHostBuffer::new(&context, 4).unwrap();
        staging.as_mut_slice().fill(17);
        let syncs = fake_driver::call_count(5);
        fake_driver::fail_next(30, 999);
        assert!(
            destination
                .upload_pinned_chunk(staging.lend(), 0, 4, &stream)
                .is_err()
        );
        assert_eq!(fake_driver::call_count(8), 1);
        assert_eq!(fake_driver::call_count(5), syncs + 1);
        staging.as_mut_slice().fill(0);
        let mut result = [0; 4];
        destination.copy_to_bytes(&mut result, &stream).unwrap();
        assert_eq!(result, [17; 4]);
    }

    #[test]
    fn dropping_a_transfer_drains_before_releasing_its_borrow() {
        fake_driver::reset();
        let context = Driver::initialize()
            .unwrap()
            .retain_primary_context(0)
            .unwrap();
        let stream = context.stream().unwrap();
        let destination = context
            .default_memory_pool()
            .unwrap()
            .allocate(&stream, 4)
            .unwrap();
        let mut staging = PinnedHostBuffer::new(&context, 4).unwrap();
        let transfer = destination
            .upload_pinned_chunk(staging.lend(), 0, 4, &stream)
            .unwrap();
        assert!(matches!(
            destination.release(),
            Err(CudaError::ResourceShared(_))
        ));
        let syncs = fake_driver::call_count(5);
        drop(transfer);
        assert_eq!(fake_driver::call_count(5), syncs + 1);
        staging.as_mut_slice().fill(1);
        destination.release().unwrap();
    }
}
