//! Bounded pinned-buffer staging for startup rank uploads.

use super::*;
use runtime_cuda::{PinnedChunk, PinnedChunkTransfer};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, SyncSender, sync_channel};
use std::time::{Duration, Instant};

#[derive(Clone, Default, serde::Serialize)]
struct RankTiming {
    chunks: u64,
    bytes: u64,
    read_seconds: f64,
    ready_wait_seconds: f64,
    buffer_wait_seconds: f64,
    transfer_wait_seconds: f64,
    submit_seconds: f64,
    device_copy_seconds: f64,
    frontier_wait_seconds: f64,
}

pub(super) fn upload(
    uploads: &mut [PinnedFileUploadV1<'_>],
    streams: &[Stream],
    destinations: &[DeviceBuffer],
) -> HostDeviceResultV1<()> {
    let mut timing =
        crate::direct::detailed_timing_v1().then(|| vec![RankTiming::default(); uploads.len()]);
    let result = upload_chunks(
        uploads,
        streams,
        destinations,
        &mut timing,
        PINNED_RANK_UPLOAD_CHUNK_BYTES,
    );
    if let Some(timing) = timing {
        eprintln!(
            "startup weight_pipeline: {}",
            serde_json::json!({
                "ranks": timing, "status": if result.is_ok() { "complete" } else { "failed" },
            })
        );
    }
    result
}

#[derive(Clone, Copy)]
struct Source {
    offset: u64,
    bytes: u64,
}

struct Read<'a> {
    chunk: PinnedChunk<'a>,
    source: u64,
    target: usize,
    count: usize,
}

struct Ready<'a> {
    rank: usize,
    read: Read<'a>,
    seconds: f64,
    at: Option<Instant>,
}

struct Rank<'a> {
    free: Vec<PinnedChunk<'a>>,
    pending: VecDeque<PinnedChunkTransfer<'a>>,
    next: u64,
    issued: u64,
    submitted: u64,
    submitted_chunks: u64,
    reading: usize,
    buffer_wait: Option<Instant>,
    frontier_wait: Option<Instant>,
}

fn upload_chunks(
    uploads: &mut [PinnedFileUploadV1<'_>],
    streams: &[Stream],
    destinations: &[DeviceBuffer],
    timing: &mut Option<Vec<RankTiming>>,
    chunk_bytes: usize,
) -> HostDeviceResultV1<()> {
    if chunk_bytes == 0 || uploads.len() != streams.len() || uploads.len() != destinations.len() {
        return Err(BundleStatusV1::InvalidArgument);
    }
    let mut sources = Vec::with_capacity(uploads.len());
    let mut buffers = Vec::with_capacity(uploads.len());
    for (upload, stream) in uploads.iter().zip(streams) {
        upload
            .source_offset
            .checked_add(upload.byte_length)
            .ok_or(BundleStatusV1::InvalidArgument)?;
        sources.push(Source {
            offset: upload.source_offset,
            bytes: upload.byte_length,
        });
        let capacity = usize::try_from(upload.byte_length)
            .unwrap_or(usize::MAX)
            .clamp(1, chunk_bytes);
        buffers.push([
            PinnedHostBuffer::new(&stream.context(), capacity).map_err(driver_status)?,
            PinnedHostBuffer::new(&stream.context(), capacity).map_err(driver_status)?,
        ]);
    }
    let stopped = AtomicBool::new(false);
    let detailed = timing.is_some();
    std::thread::scope(|scope| {
        let (ready_tx, ready_rx) = sync_channel(uploads.len().saturating_mul(2).max(1));
        let mut jobs = Vec::with_capacity(uploads.len());
        let mut readers = Vec::with_capacity(uploads.len());
        let mut ranks = Vec::with_capacity(uploads.len());
        for (rank, (upload, pair)) in uploads.iter_mut().zip(&mut buffers).enumerate() {
            let (send, receive) = sync_channel::<Read<'_>>(2);
            let ready = ready_tx.clone();
            let stopped = &stopped;
            let read_chunk = &mut *upload.read_chunk;
            readers.push(
                std::thread::Builder::new()
                    .name(format!("weight-reader-{rank}"))
                    .spawn_scoped(scope, move || {
                        while let Ok(mut read) = receive.recv() {
                            if stopped.load(Ordering::Acquire) {
                                break;
                            }
                            let start = detailed.then(Instant::now);
                            let good =
                                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                                    read_chunk(
                                        read.source,
                                        &mut read.chunk.as_mut_slice()[..read.count],
                                    )
                                }))
                                .unwrap_or(false);
                            if !good {
                                stopped.store(true, Ordering::Release);
                                break;
                            }
                            let seconds = start.map(|s| s.elapsed().as_secs_f64()).unwrap_or(0.0);
                            if ready
                                .send(Ready {
                                    rank,
                                    read,
                                    seconds,
                                    at: detailed.then(Instant::now),
                                })
                                .is_err()
                            {
                                break;
                            }
                        }
                    })
                    .map_err(|_| BundleStatusV1::HostError)?,
            );
            jobs.push(send);
            ranks.push(Rank {
                free: pair.iter_mut().map(PinnedHostBuffer::lend).collect(),
                pending: VecDeque::new(),
                next: 0,
                issued: 0,
                submitted: 0,
                submitted_chunks: 0,
                reading: 0,
                buffer_wait: None,
                frontier_wait: None,
            });
        }
        drop(ready_tx);
        let mut result = coordinate(
            &sources,
            streams,
            destinations,
            &jobs,
            &ready_rx,
            &mut ranks,
            timing,
            &stopped,
        );
        stopped.store(true, Ordering::Release);
        // Unblock idle readers and readers returning a completed chunk before
        // joining. The rank states retain every submitted transfer until all
        // CPU writers have stopped; dropping them drains remaining DMA.
        drop(jobs);
        drop(ready_rx);
        for reader in readers {
            if reader.join().is_err() && result.is_ok() {
                result = Err(BundleStatusV1::HostError);
            }
        }
        drop(ranks);
        result
    })
}

#[allow(clippy::too_many_arguments)]
fn coordinate<'a>(
    sources: &[Source],
    streams: &[Stream],
    destinations: &[DeviceBuffer],
    jobs: &[SyncSender<Read<'a>>],
    ready: &Receiver<Ready<'a>>,
    ranks: &mut [Rank<'a>],
    timing: &mut Option<Vec<RankTiming>>,
    stopped: &AtomicBool,
) -> HostDeviceResultV1<()> {
    loop {
        if stopped.load(Ordering::Acquire) {
            return Err(BundleStatusV1::HostError);
        }
        for (rank, state) in ranks.iter_mut().enumerate() {
            while state
                .pending
                .front()
                .map(|p| p.ready().map_err(driver_status))
                .transpose()?
                .unwrap_or(false)
            {
                if let Some(timing) = &mut *timing {
                    timing[rank].device_copy_seconds += state
                        .pending
                        .front()
                        .unwrap()
                        .elapsed_seconds()
                        .map_err(driver_status)?
                        .unwrap_or(0.0);
                }
                let start = timing.is_some().then(Instant::now);
                state.free.push(state.pending.pop_front().unwrap().finish());
                if let (Some(timing), Some(start)) = (&mut *timing, start) {
                    timing[rank].transfer_wait_seconds += start.elapsed().as_secs_f64();
                }
            }
        }
        // A rank may prepare two chunks beyond the slowest active rank's
        // submitted frontier. Finished smaller arenas do not hold up others.
        let frontier = ranks
            .iter()
            .zip(sources)
            .filter(|(rank, source)| rank.submitted < source.bytes)
            .map(|(rank, _)| rank.submitted_chunks)
            .min()
            .unwrap_or(u64::MAX);
        for (rank, (state, source)) in ranks.iter_mut().zip(sources).enumerate() {
            let frontier_blocked = state.next < source.bytes
                && !state.free.is_empty()
                && state.issued >= frontier.saturating_add(2);
            if let Some(timing) = &mut *timing {
                if frontier_blocked {
                    state.frontier_wait.get_or_insert_with(Instant::now);
                } else if let Some(start) = state.frontier_wait.take() {
                    timing[rank].frontier_wait_seconds += start.elapsed().as_secs_f64();
                }
            }
            if (!state.free.is_empty() || state.next == source.bytes)
                && let (Some(start), Some(timing)) = (state.buffer_wait.take(), &mut *timing)
            {
                timing[rank].buffer_wait_seconds += start.elapsed().as_secs_f64();
            }
            while state.next < source.bytes && state.issued < frontier.saturating_add(2) {
                let Some(chunk) = state.free.pop() else {
                    if timing.is_some() && state.buffer_wait.is_none() {
                        state.buffer_wait = Some(Instant::now());
                    }
                    break;
                };
                let count = usize::try_from(source.bytes - state.next)
                    .unwrap_or(usize::MAX)
                    .min(chunk.len());
                let read = Read {
                    chunk,
                    source: source.offset + state.next,
                    target: usize::try_from(state.next)
                        .map_err(|_| BundleStatusV1::InvalidArgument)?,
                    count,
                };
                jobs[rank]
                    .send(read)
                    .map_err(|_| BundleStatusV1::HostError)?;
                state.next += count as u64;
                state.issued += 1;
                state.reading += 1;
            }
        }
        if ranks.iter().zip(sources).all(|(rank, source)| {
            rank.submitted == source.bytes && rank.reading == 0 && rank.pending.is_empty()
        }) {
            return Ok(());
        }
        let received = match ready.recv_timeout(Duration::from_millis(1)) {
            Ok(read) => read,
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => continue,
            Err(_) => return Err(BundleStatusV1::HostError),
        };
        if stopped.load(Ordering::Acquire) {
            return Err(BundleStatusV1::HostError);
        }
        let Ready {
            rank,
            read,
            seconds,
            at,
        } = received;
        let state = &mut ranks[rank];
        if let Some(timing) = timing {
            timing[rank].read_seconds += seconds;
            timing[rank].ready_wait_seconds += at.map(|s| s.elapsed().as_secs_f64()).unwrap_or(0.0);
            timing[rank].chunks += 1;
            timing[rank].bytes += read.count as u64;
        }
        let start = timing.is_some().then(Instant::now);
        let transfer = if timing.is_some() {
            destinations[rank].upload_pinned_chunk_timed(
                read.chunk,
                read.target,
                read.count,
                &streams[rank],
            )
        } else {
            destinations[rank].upload_pinned_chunk(
                read.chunk,
                read.target,
                read.count,
                &streams[rank],
            )
        }
        .map_err(driver_status)?;
        if let (Some(start), Some(timing)) = (start, &mut *timing) {
            timing[rank].submit_seconds += start.elapsed().as_secs_f64();
        }
        state.pending.push_back(transfer);
        state.reading -= 1;
        state.submitted += read.count as u64;
        state.submitted_chunks += 1;
    }
}

#[cfg(all(test, feature = "cuda-fake-driver-tests"))]
mod tests {
    use super::*;
    use std::sync::mpsc::channel;
    use std::time::Duration;

    type Reader = Box<dyn FnMut(u64, &mut [u8]) -> bool + Send>;

    struct Fixture {
        device: CudaHostDeviceV1,
        stream_handles: Vec<DeviceObjectV1>,
        buffer_handles: Vec<DeviceObjectV1>,
        streams: Vec<Stream>,
        destinations: Vec<DeviceBuffer>,
        lengths: Vec<u64>,
    }

    impl Fixture {
        fn new(lengths: &[u64]) -> Self {
            runtime_cuda::fake_driver::reset();
            let mut device = CudaHostDeviceV1::new().unwrap();
            let mut stream_handles = Vec::new();
            let mut buffer_handles = Vec::new();
            for (rank, &bytes) in lengths.iter().enumerate() {
                let context = device.create_context(rank as u32, rank as i32).unwrap();
                let pool = device.create_memory_pool(context).unwrap();
                let stream = device.create_stream(context, 0).unwrap();
                let buffer = device
                    .allocate_device_buffer(pool, Some(stream), bytes.max(1), 1)
                    .unwrap();
                device
                    .copy_host_to_device(
                        stream,
                        DeviceRangeV1 {
                            object: buffer,
                            offset: 0,
                            length: bytes.max(1),
                        },
                        &vec![0xee; bytes.max(1) as usize],
                    )
                    .unwrap();
                stream_handles.push(stream);
                buffer_handles.push(buffer);
            }
            let streams = stream_handles
                .iter()
                .map(|&s| device.stream(s).unwrap().clone())
                .collect();
            let destinations = buffer_handles
                .iter()
                .map(|&b| device.buffer(b).unwrap().clone())
                .collect();
            Self {
                device,
                stream_handles,
                buffer_handles,
                streams,
                destinations,
                lengths: lengths.to_vec(),
            }
        }

        fn run(&mut self, readers: &mut [Reader]) -> (HostDeviceResultV1<()>, Vec<RankTiming>) {
            let mut uploads = readers
                .iter_mut()
                .enumerate()
                .map(|(rank, read)| PinnedFileUploadV1 {
                    stream: self.stream_handles[rank],
                    destination: self.buffer_handles[rank],
                    source_offset: 0,
                    byte_length: self.lengths[rank],
                    read_chunk: &mut **read,
                })
                .collect::<Vec<_>>();
            let mut timing = Some(vec![RankTiming::default(); uploads.len()]);
            let result = upload_chunks(
                &mut uploads,
                &self.streams,
                &self.destinations,
                &mut timing,
                16,
            );
            (result, timing.unwrap())
        }

        fn bytes(&mut self, rank: usize) -> Vec<u8> {
            let mut bytes = vec![0; self.lengths[rank] as usize];
            if !bytes.is_empty() {
                self.device
                    .copy_device_to_host(
                        self.stream_handles[rank],
                        &mut bytes,
                        DeviceRangeV1 {
                            object: self.buffer_handles[rank],
                            offset: 0,
                            length: self.lengths[rank],
                        },
                    )
                    .unwrap();
            }
            bytes
        }

        fn release(mut self) {
            self.destinations.clear();
            for buffer in self.buffer_handles {
                self.device.release_device_buffer(buffer).unwrap();
            }
        }
    }

    #[test]
    fn unequal_arenas_and_empty_ranks_keep_their_bytes_and_counts() {
        let mut fixture = Fixture::new(&[0, 35, 80]);
        runtime_cuda::fake_driver::defer_host_copies(5);
        let mut readers: Vec<Reader> = (0..3)
            .map(|rank| {
                Box::new(move |offset: u64, out: &mut [u8]| {
                    assert_ne!(rank, 0, "empty rank must not read");
                    for (index, byte) in out.iter_mut().enumerate() {
                        *byte = (offset + index as u64 + rank as u64 * 17) as u8;
                    }
                    true
                }) as Reader
            })
            .collect();
        let (result, timing) = fixture.run(&mut readers);
        result.unwrap();
        assert_eq!(
            timing.iter().map(|t| t.chunks).collect::<Vec<_>>(),
            [0, 3, 5]
        );
        assert_eq!(
            timing.iter().map(|t| t.bytes).collect::<Vec<_>>(),
            [0, 35, 80]
        );
        assert_eq!(runtime_cuda::fake_driver::pending_host_copies(), 0);
        for rank in 0..3 {
            let expected: Vec<_> = (0..fixture.lengths[rank])
                .map(|b| (b + rank as u64 * 17) as u8)
                .collect();
            assert_eq!(fixture.bytes(rank), expected);
        }
        fixture.release();
    }

    #[test]
    fn failed_and_panicking_readers_join_another_read_in_progress() {
        for panic in [false, true] {
            let mut fixture = Fixture::new(&[40, 40]);
            let (started, start) = channel();
            let (released, release) = channel();
            let joined = std::sync::Arc::new(AtomicBool::new(false));
            let completed = joined.clone();
            let mut readers: Vec<Reader> = vec![
                Box::new(move |offset, out| {
                    if offset == 0 {
                        started.send(()).unwrap();
                        release.recv_timeout(Duration::from_secs(5)).unwrap();
                    }
                    out.fill(1);
                    completed.store(true, Ordering::Release);
                    true
                }),
                Box::new(move |_, out| {
                    start.recv_timeout(Duration::from_secs(5)).unwrap();
                    out.fill(99);
                    released.send(()).unwrap();
                    assert!(!panic, "injected producer panic");
                    false
                }),
            ];
            assert_eq!(fixture.run(&mut readers).0, Err(BundleStatusV1::HostError));
            assert!(joined.load(Ordering::Acquire));
            assert_eq!(fixture.bytes(1), vec![0xee; 40]);
            fixture.release();
        }
    }

    #[test]
    fn cancellation_drains_deferred_transfers_and_never_submits_failed_bytes() {
        let mut fixture = Fixture::new(&[80, 80]);
        runtime_cuda::fake_driver::defer_host_copies(20);
        // The first event releases a buffer. Later copies cannot finish by
        // polling, so cancellation must drain them through synchronization.
        runtime_cuda::fake_driver::hold_later_host_copies(true);
        let (progress, ready) = channel();
        let mut readers: Vec<Reader> = vec![
            Box::new(move |offset, out| {
                out.fill((offset / 16 + 1) as u8);
                if offset == 32 {
                    progress.send(()).unwrap();
                }
                true
            }),
            Box::new(move |offset, out| {
                if offset != 0 {
                    ready.recv_timeout(Duration::from_secs(5)).unwrap();
                    out.fill(99);
                    return false;
                }
                out.fill(7);
                true
            }),
        ];
        assert_eq!(fixture.run(&mut readers).0, Err(BundleStatusV1::HostError));
        assert_eq!(runtime_cuda::fake_driver::pending_host_copies(), 0);
        let fast = fixture.bytes(0);
        assert_eq!(&fast[..16], &[1; 16]);
        assert_eq!(&fast[16..32], &[2; 16]);
        let slow = fixture.bytes(1);
        assert_eq!(&slow[..16], &[7; 16]);
        assert_eq!(&slow[16..], &[0xee; 64]);
        fixture.release();
    }

    #[test]
    fn producer_lead_is_bounded_while_a_rank_is_blocked() {
        let mut fixture = Fixture::new(&[64, 64]);
        let (progress, ready) = channel();
        let mut readers: Vec<Reader> = vec![
            Box::new(move |offset, out| {
                if offset == 0 {
                    assert_eq!(ready.recv_timeout(Duration::from_secs(5)).unwrap(), 0);
                    assert_eq!(ready.recv_timeout(Duration::from_secs(5)).unwrap(), 1);
                    assert!(
                        ready.recv_timeout(Duration::from_millis(50)).is_err(),
                        "the fast rank prepared a third chunk beyond the frontier"
                    );
                }
                out.fill(1);
                true
            }),
            Box::new(move |offset, out| {
                progress.send(offset / 16).unwrap();
                out.fill(2);
                true
            }),
        ];
        fixture.run(&mut readers).0.unwrap();
        assert_eq!(fixture.bytes(0), vec![1; 64]);
        assert_eq!(fixture.bytes(1), vec![2; 64]);
        fixture.release();
    }

    #[test]
    fn driver_failures_leave_no_transfer_borrow_behind() {
        for operation in [8, 30, 31] {
            let mut fixture = Fixture::new(&[40, 40]);
            let mut readers: Vec<Reader> = (0..2)
                .map(|_| {
                    Box::new(|_, out: &mut [u8]| {
                        out.fill(3);
                        true
                    }) as Reader
                })
                .collect();
            runtime_cuda::fake_driver::fail_next(operation, 999);
            assert!(
                fixture.run(&mut readers).0.is_err(),
                "operation {operation}"
            );
            fixture.release();
        }
    }

    #[test]
    fn a_ready_rank_advances_while_another_rank_is_still_reading() {
        runtime_cuda::fake_driver::reset();
        let mut device = CudaHostDeviceV1::new().unwrap();
        let context = device.create_context(0, 0).unwrap();
        let pool = device.create_memory_pool(context).unwrap();
        let stream_handles = [
            device.create_stream(context, 0).unwrap(),
            device.create_stream(context, 0).unwrap(),
        ];
        let buffers = stream_handles.map(|stream| {
            device
                .allocate_device_buffer(pool, Some(stream), 40, 1)
                .unwrap()
        });
        let streams = stream_handles.map(|handle| device.stream(handle).unwrap().clone());
        let destinations = buffers.map(|handle| device.buffer(handle).unwrap().clone());
        let (progress, ready) = channel();
        let mut fast = move |offset: u64, out: &mut [u8]| {
            if offset == 7 + 16 {
                progress.send(()).unwrap();
            }
            for (index, byte) in out.iter_mut().enumerate() {
                *byte = (offset + index as u64) as u8;
            }
            true
        };
        let mut slow = move |offset: u64, out: &mut [u8]| {
            if offset == 11 && ready.recv_timeout(Duration::from_secs(5)).is_err() {
                return false;
            }
            for (index, byte) in out.iter_mut().enumerate() {
                *byte = (offset + index as u64) as u8;
            }
            true
        };
        let mut uploads = [
            PinnedFileUploadV1 {
                stream: stream_handles[0],
                destination: buffers[0],
                source_offset: 7,
                byte_length: 40,
                read_chunk: &mut fast,
            },
            PinnedFileUploadV1 {
                stream: stream_handles[1],
                destination: buffers[1],
                source_offset: 11,
                byte_length: 40,
                read_chunk: &mut slow,
            },
        ];
        let result = upload_chunks(&mut uploads, &streams, &destinations, &mut None, 16);
        assert_eq!(
            result,
            Ok(()),
            "a slow rank blocked the ready rank's next chunk"
        );
        for (rank, start) in [7_u8, 11].into_iter().enumerate() {
            let mut actual = [0; 40];
            device
                .copy_device_to_host(
                    stream_handles[rank],
                    &mut actual,
                    DeviceRangeV1 {
                        object: buffers[rank],
                        offset: 0,
                        length: 40,
                    },
                )
                .unwrap();
            assert_eq!(actual.as_slice(), &(start..start + 40).collect::<Vec<_>>());
        }
    }
}

#[cfg(all(test, not(feature = "cuda-fake-driver-tests")))]
mod real_driver_tests {
    use super::*;
    use std::sync::mpsc::channel;

    #[test]
    #[ignore = "requires four selected idle GPUs and a coordinated quiet host"]
    fn unequal_rank_uploads_keep_every_gpu_byte() {
        assert_eq!(std::env::var("CHOPINFER_KERNEL_GPUS").unwrap(), "0,1,2,3");
        let chunk = PINNED_RANK_UPLOAD_CHUNK_BYTES as u64;
        let lengths = [0, chunk + 19, 2 * chunk + 41, (7 << 20) + 3];
        let starts = [0, 7, 11, 23];
        let mut device = CudaHostDeviceV1::new().unwrap();
        let mut streams = Vec::new();
        let mut buffers = Vec::new();
        for (rank, &bytes) in lengths.iter().enumerate() {
            let context = device.create_context(rank as u32, rank as i32).unwrap();
            let pool = device.create_memory_pool(context).unwrap();
            let stream = device.create_stream(context, 0).unwrap();
            let buffer = device
                .allocate_device_buffer(pool, Some(stream), bytes.max(1), 256)
                .unwrap();
            streams.push(stream);
            buffers.push(buffer);
        }
        let pattern = |rank: usize, offset: u64| (offset.wrapping_mul(13) + rank as u64 * 17) as u8;
        let (progress, ready) = channel();
        let mut progress = Some(progress);
        let mut ready = Some(ready);
        type Reader = Box<dyn FnMut(u64, &mut [u8]) -> bool + Send>;
        let mut readers: Vec<Reader> = (0..4)
            .map(|rank| {
                let sender = if rank == 2 { progress.take() } else { None };
                let receiver = if rank == 1 { ready.take() } else { None };
                Box::new(move |offset: u64, out: &mut [u8]| {
                    if rank == 1
                        && offset == starts[rank]
                        && receiver
                            .as_ref()
                            .unwrap()
                            .recv_timeout(Duration::from_secs(10))
                            .is_err()
                    {
                        return false;
                    }
                    if rank == 2 && offset == starts[rank] + chunk {
                        sender.as_ref().unwrap().send(()).unwrap();
                    }
                    for (index, byte) in out.iter_mut().enumerate() {
                        *byte = pattern(rank, offset + index as u64);
                    }
                    true
                }) as Reader
            })
            .collect();
        let mut uploads: Vec<_> = readers
            .iter_mut()
            .enumerate()
            .map(|(rank, read)| PinnedFileUploadV1 {
                stream: streams[rank],
                destination: buffers[rank],
                source_offset: starts[rank],
                byte_length: lengths[rank],
                read_chunk: &mut **read,
            })
            .collect();
        device
            .upload_pinned_file_ranges(&mut uploads)
            .unwrap()
            .unwrap();
        let mut checked = 0;
        for rank in 0..4 {
            let mut offset = 0;
            let mut bytes = vec![0; 4 << 20];
            while offset < lengths[rank] {
                let count = (lengths[rank] - offset).min(bytes.len() as u64) as usize;
                device
                    .copy_device_to_host(
                        streams[rank],
                        &mut bytes[..count],
                        DeviceRangeV1 {
                            object: buffers[rank],
                            offset,
                            length: count as u64,
                        },
                    )
                    .unwrap();
                assert!(
                    bytes[..count].iter().enumerate().all(|(index, &byte)| byte
                        == pattern(rank, starts[rank] + offset + index as u64)),
                    "rank {rank} differs at readback chunk {offset}"
                );
                offset += count as u64;
                checked += count as u64;
            }
        }
        assert_eq!(checked, lengths.iter().sum::<u64>());
        assert!(checked > 3 * chunk);
        for buffer in buffers {
            device.release_device_buffer(buffer).unwrap();
        }
        println!(
            "upload_gpu_probe: {}",
            serde_json::json!({"ranks":4,"checked_bytes":checked,"status":"passed"})
        );
    }
}
