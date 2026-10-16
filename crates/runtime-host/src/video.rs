//! Video frame extraction for bundles that take media inputs.
//!
//! A video part is demuxed and planned on the caller's thread: the container's
//! first H.264 track becomes Annex B access units stamped with their
//! decode-order sample index, and the frame plan names the samples to keep.
//! A small pool of worker threads then decodes, each worker owning one
//! decoder: on the CUDA backend, NVDEC on a CUDA context of its own, apart
//! from the contexts the engine serves on. Workers copy only the planned
//! frames and convert each to RGB. A request waits for its own frames only,
//! with a deadline and its cancellation, so a slow or stuck decode costs that
//! request and one worker, never admission. None of this runs unless the
//! selected bundle requires `chop.media-inputs.v1`.

use runtime_bundle::BundleManifestV1;
use runtime_media::mp4::{Mp4Error, Mp4Limits, VideoCodec, demux_video};
use runtime_media::video_frames::{FrameSampling, VideoFrameError, frame_plan, nv12_to_rgb};
use std::collections::BTreeMap;
use std::error::Error;
use std::fmt;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, mpsc};
use std::thread;
use std::time::{Duration, Instant};

pub use crate::executor::MEDIA_INPUTS_CAPABILITY_V1;

/// Whether the verified bundle takes media inputs, so this host must be able
/// to decode them.
#[must_use]
pub fn media_inputs_required_v1(manifest: &BundleManifestV1) -> bool {
    requires_media_inputs_v1(
        manifest
            .resource_requirements()
            .required_capabilities()
            .iter()
            .map(|capability| capability.as_str()),
    )
}

fn requires_media_inputs_v1<'a>(mut capabilities: impl Iterator<Item = &'a str>) -> bool {
    capabilities.any(|capability| capability == MEDIA_INPUTS_CAPABILITY_V1)
}

/// Bounds on one video input.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct VideoLimitsV1 {
    /// The most samples the container may declare.
    pub maximum_samples: u32,
    /// The largest display width and height decoded; the decoder's own
    /// capacity may lower them.
    pub maximum_width: u32,
    pub maximum_height: u32,
    pub sampling: FrameSampling,
}

/// A refusal or failure while extracting frames. Every message is safe to
/// return to the client.
#[derive(Debug)]
pub enum VideoErrorV1 {
    Container(Mp4Error),
    Frames(VideoFrameError),
    /// This host cannot decode video.
    Unavailable(String),
    /// The decoder refused or failed on this input.
    Decode(String),
    /// The decode pool or its frame budget is full; a later retry may succeed.
    Busy(String),
    /// The frames were not ready within the decode deadline.
    TimedOut(Duration),
    /// The request was cancelled while it waited for its frames.
    Cancelled,
}

impl fmt::Display for VideoErrorV1 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Container(error) => write!(formatter, "{error}"),
            Self::Frames(error) => write!(formatter, "{error}"),
            Self::Unavailable(reason) => write!(formatter, "video input is unavailable: {reason}"),
            Self::Decode(reason) => write!(formatter, "video could not be decoded: {reason}"),
            Self::Busy(reason) => write!(formatter, "video decoding is busy: {reason}"),
            Self::TimedOut(deadline) => write!(
                formatter,
                "video decoding did not finish within {} seconds",
                deadline.as_secs_f64()
            ),
            Self::Cancelled => formatter.write_str("video decoding was cancelled"),
        }
    }
}

impl Error for VideoErrorV1 {}

/// One kept frame as interleaved RGB bytes.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VideoFrameV1 {
    pub width: u32,
    pub height: u32,
    pub rgb: Vec<u8>,
    pub label_seconds: u32,
}

/// A demuxed video ready for the decoder.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PreparedVideoV1 {
    /// Annex B access units in decode order, parameter sets first.
    pub stream: Vec<u8>,
    /// The end offset of each access unit in `stream`.
    pub unit_ends: Vec<u64>,
    /// Each unit's decode-order sample index, which NVDEC returns with the
    /// frame the unit produces.
    pub timestamps: Vec<i64>,
    /// The kept frames as (sample index, label seconds), in display order.
    pub plan: Vec<(i64, u32)>,
}

/// Demuxes `file` and plans which frames to keep.
pub fn prepare_video_v1(
    file: &[u8],
    limits: &VideoLimitsV1,
) -> Result<PreparedVideoV1, VideoErrorV1> {
    let track = demux_video(
        file,
        &Mp4Limits {
            maximum_samples: limits.maximum_samples,
        },
    )
    .map_err(VideoErrorV1::Container)?;
    let VideoCodec::Avc(config) = &track.codec;
    let presentable = track.presentable_order();
    let plan = frame_plan(track.samples.len(), presentable.len(), limits.sampling)
        .map_err(VideoErrorV1::Frames)?
        .into_iter()
        .map(|frame| (presentable[frame.display_index] as i64, frame.label_seconds))
        .collect();
    let mut stream = Vec::new();
    config.write_parameter_sets(&mut stream);
    let mut unit_ends = Vec::with_capacity(track.samples.len());
    for index in 0..track.samples.len() {
        let sample = track
            .sample_bytes(file, index)
            .map_err(VideoErrorV1::Container)?;
        config
            .write_annex_b(sample, &mut stream)
            .map_err(VideoErrorV1::Container)?;
        unit_ends.push(stream.len() as u64);
    }
    Ok(PreparedVideoV1 {
        stream,
        unit_ends,
        timestamps: (0..track.samples.len() as i64).collect(),
        plan,
    })
}

/// How the decode pool bounds the work and memory of video inputs.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct VideoDecodeConfigV1 {
    /// Worker threads, each owning one decoder.
    pub workers: usize,
    /// Jobs that may wait for a worker; a full queue refuses new video.
    pub queue: usize,
    /// How long a request waits for its frames, queueing included.
    pub deadline: Duration,
    /// RGB bytes all decoded videos may hold at once, from their first kept
    /// frame until the request has turned them into patches.
    pub frame_bytes: u64,
}

impl Default for VideoDecodeConfigV1 {
    fn default() -> Self {
        Self {
            workers: 2,
            queue: 8,
            deadline: Duration::from_secs(30),
            frame_bytes: 4 << 30,
        }
    }
}

/// One decoder, owned by one pool worker for the worker's lifetime.
pub trait VideoDecoderV1 {
    /// The largest frame this decoder decodes.
    fn capacity(&self) -> (u32, u32);

    /// Decodes `video`, offering each displayed frame to `sink` in display
    /// order. A sink that refuses a frame stops the decode; its own failure
    /// takes precedence over what this returns.
    fn decode(
        &mut self,
        video: &PreparedVideoV1,
        limits: &VideoLimitsV1,
        sink: &mut dyn FrameSinkV1,
    ) -> Result<(), VideoErrorV1>;
}

/// Where a decoder delivers frames.
pub trait FrameSinkV1 {
    /// Whether the displayed frame from this sample should be copied out.
    fn wants(&self, sample: i64) -> bool;

    /// Takes one copied NV12 frame; `false` stops the decode.
    fn accept(&mut self, sample: i64, width: u32, height: u32, nv12: &[u8]) -> bool;
}

/// Frames of one video and the share of the frame budget they hold. Keep it
/// until the frames are no longer needed.
#[derive(Debug)]
pub struct DecodedVideoV1 {
    pub frames: Vec<VideoFrameV1>,
    reservation: FrameReservationV1,
}

impl DecodedVideoV1 {
    /// The frames, and the reservation to hold while they or their copies live.
    #[must_use]
    pub fn into_parts(self) -> (Vec<VideoFrameV1>, FrameReservationV1) {
        (self.frames, self.reservation)
    }
}

#[derive(Debug)]
struct FrameBudgetV1 {
    limit: u64,
    used: AtomicU64,
}

impl FrameBudgetV1 {
    fn reserve(self: &Arc<Self>, bytes: u64) -> Option<FrameReservationV1> {
        self.used
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |used| {
                used.checked_add(bytes).filter(|&total| total <= self.limit)
            })
            .ok()
            .map(|_| FrameReservationV1 {
                budget: Arc::clone(self),
                bytes,
            })
    }
}

/// Bytes of the shared frame budget, returned on drop.
#[derive(Debug)]
pub struct FrameReservationV1 {
    budget: Arc<FrameBudgetV1>,
    bytes: u64,
}

impl Drop for FrameReservationV1 {
    fn drop(&mut self) {
        self.budget.used.fetch_sub(self.bytes, Ordering::AcqRel);
    }
}

/// What a job's caller and its worker share.
#[derive(Debug)]
struct JobStateV1 {
    cancelled: AtomicBool,
    deadline: Instant,
}

impl JobStateV1 {
    fn abandoned(&self) -> bool {
        self.cancelled.load(Ordering::Acquire) || Instant::now() >= self.deadline
    }
}

/// Collects decoded frames against a plan.
///
/// Frames arrive keyed by sample index. A planned frame the decoder never
/// produced is dropped, and the remaining frames are relabeled in order, as
/// the reference loader does for a frame it fails to read. The first kept
/// frame reserves RGB bytes for every planned frame from the shared budget,
/// before any is converted.
struct FrameAssemblyV1 {
    positions: BTreeMap<i64, usize>,
    frames: Vec<Option<VideoFrameV1>>,
    seconds_per_frame: u32,
    job: Arc<JobStateV1>,
    budget: Arc<FrameBudgetV1>,
    reservation: Option<(FrameReservationV1, u64)>,
    failure: Option<VideoErrorV1>,
}

impl FrameAssemblyV1 {
    fn new(
        plan: &[(i64, u32)],
        seconds_per_frame: u32,
        job: Arc<JobStateV1>,
        budget: Arc<FrameBudgetV1>,
    ) -> Self {
        Self {
            positions: plan
                .iter()
                .enumerate()
                .map(|(position, &(sample, _))| (sample, position))
                .collect(),
            frames: vec![None; plan.len()],
            seconds_per_frame,
            job,
            budget,
            reservation: None,
            failure: None,
        }
    }

    fn stop(&mut self, failure: VideoErrorV1) -> bool {
        self.failure.get_or_insert(failure);
        false
    }

    fn finish(self) -> Result<DecodedVideoV1, VideoErrorV1> {
        let frames: Vec<_> = self
            .frames
            .into_iter()
            .flatten()
            .enumerate()
            .map(|(position, frame)| VideoFrameV1 {
                label_seconds: (position as u32).saturating_mul(self.seconds_per_frame),
                ..frame
            })
            .collect();
        match self.reservation {
            Some((reservation, _)) if !frames.is_empty() => Ok(DecodedVideoV1 {
                frames,
                reservation,
            }),
            _ => Err(VideoErrorV1::Frames(VideoFrameError::NoFrames)),
        }
    }
}

// Opt-in diagnostics describe only copied frame bytes, never GPU addresses.
fn frame_checksums(
    sample: i64,
    width: u32,
    height: u32,
    nv12: &[u8],
    rgb: &[u8],
) -> serde_json::Value {
    serde_json::json!({
        "sample": sample,
        "width": width,
        "height": height,
        "nv12_bytes": nv12.len(),
        "nv12_sha256": runtime_weights::sha256_bytes(nv12),
        "rgb_bytes": rgb.len(),
        "rgb_sha256": runtime_weights::sha256_bytes(rgb),
    })
}

impl FrameSinkV1 for FrameAssemblyV1 {
    fn wants(&self, sample: i64) -> bool {
        self.failure.is_none()
            && !self.job.cancelled.load(Ordering::Acquire)
            && self
                .positions
                .get(&sample)
                .is_some_and(|&position| self.frames[position].is_none())
    }

    fn accept(&mut self, sample: i64, width: u32, height: u32, nv12: &[u8]) -> bool {
        if self.job.cancelled.load(Ordering::Acquire) {
            return self.stop(VideoErrorV1::Cancelled);
        }
        let Some(&position) = self.positions.get(&sample) else {
            return true;
        };
        let frame_bytes = u64::from(width) * u64::from(height) * 3;
        match &self.reservation {
            Some((_, reserved)) if frame_bytes > *reserved => {
                return self.stop(VideoErrorV1::Decode(
                    "the frame size changed mid-stream".to_owned(),
                ));
            }
            Some(_) => {}
            None => {
                let bytes = frame_bytes.saturating_mul(self.frames.len() as u64);
                let Some(reservation) = self.budget.reserve(bytes) else {
                    return self.stop(VideoErrorV1::Busy(format!(
                        "its {bytes} bytes of decoded frames do not fit the {}-byte frame \
                         budget beside other videos",
                        self.budget.limit
                    )));
                };
                self.reservation = Some((reservation, frame_bytes));
            }
        }
        match nv12_to_rgb(width, height, nv12) {
            Ok(rgb) => {
                if std::env::var("CHOPINFER_VIDEO_FRAME_CHECKSUMS").as_deref() == Ok("1") {
                    eprintln!(
                        "video frame checksum {}",
                        frame_checksums(sample, width, height, nv12, &rgb)
                    );
                }
                self.frames[position] = Some(VideoFrameV1 {
                    width,
                    height,
                    rgb,
                    label_seconds: 0,
                });
                true
            }
            Err(error) => self.stop(VideoErrorV1::Frames(error)),
        }
    }
}

type Reply = mpsc::SyncSender<Result<DecodedVideoV1, VideoErrorV1>>;

struct Job {
    video: PreparedVideoV1,
    limits: VideoLimitsV1,
    state: Arc<JobStateV1>,
    reply: Reply,
}

struct PoolSharedV1 {
    budget: Arc<FrameBudgetV1>,
    /// The deadline of the job each worker is running, if any.
    running: Vec<Mutex<Option<Instant>>>,
}

impl PoolSharedV1 {
    /// Workers still inside a job whose caller has given up on it. A decode
    /// stuck in the driver cannot be interrupted, so its worker stays here
    /// until the call returns.
    fn overdue(&self, now: Instant) -> usize {
        self.running
            .iter()
            .filter(|slot| {
                slot.lock()
                    .map(|deadline| deadline.is_some_and(|deadline| now >= deadline))
                    .unwrap_or(true)
            })
            .count()
    }

    fn set_running(&self, worker: usize, deadline: Option<Instant>) {
        if let Ok(mut slot) = self.running[worker].lock() {
            *slot = deadline;
        }
    }
}

/// A pool of video decoders, each owned by its own worker thread.
///
/// A request waits for its own job only, and never longer than the deadline
/// or past its cancellation. A queued job whose caller left is skipped, and a
/// running one stops copying frames. A job stuck inside a decoder call keeps
/// only its worker; once every worker is overdue, or the queue is full, new
/// video is refused at once rather than queued. Dropping the pool ends idle
/// workers and leaves stuck ones to exit when their call returns.
pub struct VideoDecodeServiceV1 {
    jobs: mpsc::SyncSender<Job>,
    shared: Arc<PoolSharedV1>,
    capacity: (u32, u32),
    config: VideoDecodeConfigV1,
}

/// How often a waiting request looks at its cancellation.
const CANCELLATION_POLL: Duration = Duration::from_millis(20);

impl VideoDecodeServiceV1 {
    /// Starts `config.workers` workers, each building its decoder with
    /// `open` on its own thread. Any worker's refusal refuses the pool.
    pub fn start_with<F>(config: VideoDecodeConfigV1, open: F) -> Result<Self, VideoErrorV1>
    where
        F: Fn() -> Result<Box<dyn VideoDecoderV1>, VideoErrorV1> + Send + Sync + 'static,
    {
        if config.workers == 0 {
            return Err(VideoErrorV1::Unavailable(
                "the video decode pool has no workers".to_owned(),
            ));
        }
        let (jobs, queue) = mpsc::sync_channel::<Job>(config.queue);
        let queue = Arc::new(Mutex::new(queue));
        let open = Arc::new(open);
        let shared = Arc::new(PoolSharedV1 {
            budget: Arc::new(FrameBudgetV1 {
                limit: config.frame_bytes,
                used: AtomicU64::new(0),
            }),
            running: (0..config.workers).map(|_| Mutex::new(None)).collect(),
        });
        let (ready, readiness) = mpsc::channel();
        for worker in 0..config.workers {
            let (queue, open, shared, ready) = (
                Arc::clone(&queue),
                Arc::clone(&open),
                Arc::clone(&shared),
                ready.clone(),
            );
            thread::Builder::new()
                .name("video-decode".to_owned())
                .spawn(move || {
                    let mut decoder = match open() {
                        Ok(decoder) => {
                            let _ = ready.send(Ok(decoder.capacity()));
                            decoder
                        }
                        Err(error) => {
                            let _ = ready.send(Err(error));
                            return;
                        }
                    };
                    drop(ready);
                    run_worker(worker, decoder.as_mut(), &queue, &shared);
                })
                .map_err(|error| VideoErrorV1::Unavailable(error.to_string()))?;
        }
        drop(ready);
        let mut capacity = None;
        for _ in 0..config.workers {
            let reported = readiness
                .recv()
                .map_err(|_| VideoErrorV1::Unavailable("a decode worker exited".to_owned()))??;
            capacity.get_or_insert(reported);
        }
        Ok(Self {
            jobs,
            shared,
            capacity: capacity.unwrap_or_default(),
            config,
        })
    }

    /// The largest frame the decoders decode.
    #[must_use]
    pub fn capacity(&self) -> (u32, u32) {
        self.capacity
    }

    /// Decodes the planned frames of one prepared video.
    ///
    /// Returns once the frames are ready, `is_cancelled` reports true, or the
    /// deadline passes, whichever is first. Never waits for a worker to free
    /// up beyond that, and refuses at once when no worker could take the job.
    pub fn decode(
        &self,
        video: PreparedVideoV1,
        limits: VideoLimitsV1,
        is_cancelled: &dyn Fn() -> bool,
    ) -> Result<DecodedVideoV1, VideoErrorV1> {
        let now = Instant::now();
        let workers = self.config.workers;
        if self.shared.overdue(now) >= workers {
            return Err(VideoErrorV1::Unavailable(format!(
                "all {workers} video decode workers are stuck past their deadline"
            )));
        }
        let state = Arc::new(JobStateV1 {
            cancelled: AtomicBool::new(false),
            deadline: now + self.config.deadline,
        });
        let (reply, answer) = mpsc::sync_channel(1);
        let job = Job {
            video,
            limits,
            state: Arc::clone(&state),
            reply,
        };
        match self.jobs.try_send(job) {
            Ok(()) => {}
            Err(mpsc::TrySendError::Full(_)) => {
                return Err(VideoErrorV1::Busy(format!(
                    "{} videos are already waiting for a decoder",
                    self.config.queue
                )));
            }
            Err(mpsc::TrySendError::Disconnected(_)) => {
                return Err(VideoErrorV1::Unavailable(
                    "the decode workers exited".to_owned(),
                ));
            }
        }
        loop {
            if is_cancelled() {
                state.cancelled.store(true, Ordering::Release);
                return Err(VideoErrorV1::Cancelled);
            }
            let remaining = state.deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                state.cancelled.store(true, Ordering::Release);
                let overdue = self.shared.overdue(Instant::now());
                eprintln!(
                    "video decode: a job passed its {:?} deadline; {overdue} of {workers} \
                     workers are overdue",
                    self.config.deadline
                );
                return Err(VideoErrorV1::TimedOut(self.config.deadline));
            }
            match answer.recv_timeout(remaining.min(CANCELLATION_POLL)) {
                Ok(result) => return result,
                Err(mpsc::RecvTimeoutError::Timeout) => {}
                Err(mpsc::RecvTimeoutError::Disconnected) => {
                    return Err(VideoErrorV1::Unavailable(
                        "the decode worker exited".to_owned(),
                    ));
                }
            }
        }
    }
}

fn run_worker(
    worker: usize,
    decoder: &mut dyn VideoDecoderV1,
    queue: &Mutex<mpsc::Receiver<Job>>,
    shared: &PoolSharedV1,
) {
    loop {
        let job = match queue.lock() {
            Ok(queue) => match queue.recv() {
                Ok(job) => job,
                Err(_) => return,
            },
            Err(_) => return,
        };
        if job.state.abandoned() {
            continue;
        }
        shared.set_running(worker, Some(job.state.deadline));
        let result = decode_job(decoder, &job, &shared.budget);
        shared.set_running(worker, None);
        let _ = job.reply.try_send(result);
    }
}

fn decode_job(
    decoder: &mut dyn VideoDecoderV1,
    job: &Job,
    budget: &Arc<FrameBudgetV1>,
) -> Result<DecodedVideoV1, VideoErrorV1> {
    let mut assembly = FrameAssemblyV1::new(
        &job.video.plan,
        job.limits.sampling.seconds_per_frame,
        Arc::clone(&job.state),
        Arc::clone(budget),
    );
    let decoded = catch_unwind(AssertUnwindSafe(|| {
        decoder.decode(&job.video, &job.limits, &mut assembly)
    }))
    .unwrap_or_else(|_| Err(VideoErrorV1::Decode("the decoder panicked".to_owned())));
    match (assembly.failure.take(), decoded) {
        (Some(failure), _) => Err(failure),
        (None, Err(error)) => Err(error),
        (None, Ok(())) => assembly.finish(),
    }
}

#[cfg(all(target_os = "linux", feature = "linux-cuda-host"))]
mod nvdec {
    use super::{
        FrameSinkV1, PreparedVideoV1, VideoDecodeConfigV1, VideoDecodeServiceV1, VideoDecoderV1,
        VideoErrorV1, VideoLimitsV1,
    };
    use runtime_cuda::{AnnexBStream, NvdecSession};
    use std::cell::RefCell;

    /// NVDEC on a CUDA context of its own; see [`NvdecSession`].
    struct NvdecDecoderV1 {
        session: NvdecSession,
        capacity: (u32, u32),
    }

    impl VideoDecoderV1 for NvdecDecoderV1 {
        fn capacity(&self) -> (u32, u32) {
            self.capacity
        }

        fn decode(
            &mut self,
            video: &PreparedVideoV1,
            limits: &VideoLimitsV1,
            sink: &mut dyn FrameSinkV1,
        ) -> Result<(), VideoErrorV1> {
            let mut stream = AnnexBStream::new();
            let mut start = 0;
            for (&end, &timestamp) in video.unit_ends.iter().zip(&video.timestamps) {
                let unit = video.stream.get(start..end as usize).ok_or_else(|| {
                    VideoErrorV1::Decode("access units overrun the stream".to_owned())
                })?;
                stream
                    .push_unit(timestamp, |out| {
                        out.extend_from_slice(unit);
                        Ok::<(), ()>(())
                    })
                    .map_err(|()| VideoErrorV1::Decode("access unit write failed".to_owned()))?;
                start = end as usize;
            }
            let sink = RefCell::new(sink);
            self.session
                .decode_h264(
                    &stream,
                    limits.maximum_width,
                    limits.maximum_height,
                    &mut |timestamp| sink.borrow().wants(timestamp),
                    &mut |frame| {
                        sink.borrow_mut().accept(
                            frame.timestamp,
                            frame.width,
                            frame.height,
                            frame.nv12,
                        )
                    },
                )
                .map_err(|error| VideoErrorV1::Decode(error.to_string()))
        }
    }

    impl VideoDecodeServiceV1 {
        /// Starts NVDEC workers on `device_ordinal`. A host without
        /// `libnvcuvid`, or a device without H.264 decode, is refused here, so
        /// the caller can refuse video parts with the reason.
        pub fn start(device_ordinal: i32) -> Result<Self, VideoErrorV1> {
            Self::start_with(VideoDecodeConfigV1::default(), move || {
                let session = NvdecSession::open(device_ordinal)
                    .map_err(|error| VideoErrorV1::Unavailable(error.to_string()))?;
                let capacity = session
                    .h264_capacity()
                    .map_err(|error| VideoErrorV1::Unavailable(error.to_string()))?;
                if std::env::var("CHOPINFER_VIDEO_FRAME_CHECKSUMS").as_deref() == Ok("1") {
                    eprintln!("video decoder visible_ordinal={device_ordinal}");
                }
                Ok(Box::new(NvdecDecoderV1 { session, capacity }) as Box<dyn VideoDecoderV1>)
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FIXTURE: &[u8] = include_bytes!("../../runtime-media/tests/fixtures/red-then-blue.mp4");

    fn limits(maximum_frames: usize) -> VideoLimitsV1 {
        VideoLimitsV1 {
            maximum_samples: 1 << 16,
            maximum_width: 4096,
            maximum_height: 4096,
            sampling: FrameSampling {
                maximum_frames,
                seconds_per_frame: 1,
            },
        }
    }

    #[test]
    fn checksums_cover_both_decoded_planes_and_converted_pixels() {
        let rgb = nv12_to_rgb(2, 2, &RED).unwrap();
        let before = frame_checksums(7, 2, 2, &RED, &rgb);
        assert_eq!(before["sample"], 7);
        assert_eq!(before["width"], 2);
        assert_eq!(before["height"], 2);
        assert_eq!(before["nv12_bytes"], 6);
        assert_eq!(before["rgb_bytes"], 12);
        for index in [0, 4, 5] {
            let mut changed = RED;
            changed[index] = changed[index].saturating_sub(10);
            let changed_rgb = nv12_to_rgb(2, 2, &changed).unwrap();
            let after = frame_checksums(7, 2, 2, &changed, &changed_rgb);
            assert_ne!(before["nv12_sha256"], after["nv12_sha256"]);
            assert_ne!(before["rgb_sha256"], after["rgb_sha256"]);
        }
    }

    #[test]
    fn only_bundles_requiring_media_inputs_start_video_decoding() {
        assert!(requires_media_inputs_v1(
            ["chop.serving-kv.v1", MEDIA_INPUTS_CAPABILITY_V1].into_iter()
        ));
        assert!(!requires_media_inputs_v1(
            ["chop.prompt-rows.v2"].into_iter()
        ));
    }

    #[test]
    fn fixture_prepares_decode_order_units_and_a_display_order_plan() {
        let video = prepare_video_v1(FIXTURE, &limits(32)).unwrap();
        assert_eq!(video.unit_ends.len(), 8);
        assert_eq!(video.timestamps, (0..8).collect::<Vec<_>>());
        assert_eq!(&video.stream[..5], &[0, 0, 0, 1, 0x67]);
        assert_eq!(
            *video.unit_ends.last().unwrap() as usize,
            video.stream.len()
        );
        let samples: Vec<_> = video.plan.iter().map(|&(sample, _)| sample).collect();
        assert_eq!(samples, [0, 2, 3, 1, 4, 6, 7, 5]);
        let labels: Vec<_> = video.plan.iter().map(|&(_, label)| label).collect();
        assert_eq!(labels, (0..8).collect::<Vec<_>>());

        let sparse = prepare_video_v1(FIXTURE, &limits(3)).unwrap();
        // linspace(0, 7, 3) over display order picks display frames 0, 3 and 7.
        assert_eq!(sparse.plan, [(0, 0), (1, 1), (5, 2)]);
    }

    fn job(deadline: Duration) -> Arc<JobStateV1> {
        Arc::new(JobStateV1 {
            cancelled: AtomicBool::new(false),
            deadline: Instant::now() + deadline,
        })
    }

    fn budget(limit: u64) -> Arc<FrameBudgetV1> {
        Arc::new(FrameBudgetV1 {
            limit,
            used: AtomicU64::new(0),
        })
    }

    const RED: [u8; 6] = [81, 81, 81, 81, 90, 240];

    #[test]
    fn assembly_orders_frames_by_plan_and_relabels_missing_ones() {
        let plan = [(0, 0), (2, 1), (1, 2)];
        let frames = budget(1 << 20);
        let mut assembly =
            FrameAssemblyV1::new(&plan, 1, job(Duration::from_secs(60)), Arc::clone(&frames));
        assert!(assembly.wants(2) && !assembly.wants(5));
        assert!(assembly.accept(2, 2, 2, &RED));
        // Three planned 2x2 RGB frames are reserved by the first kept one.
        assert_eq!(frames.used.load(Ordering::Acquire), 36);
        assert!(!assembly.wants(2));
        assert!(assembly.accept(0, 2, 2, &RED));
        let decoded = assembly.finish().unwrap();
        assert_eq!(decoded.frames.len(), 2);
        assert_eq!(decoded.frames[1].label_seconds, 1);
        assert_eq!(&decoded.frames[0].rgb[..3], &[253, 0, 0]);
        drop(decoded);
        assert_eq!(frames.used.load(Ordering::Acquire), 0);
        assert!(matches!(
            FrameAssemblyV1::new(&plan, 1, job(Duration::from_secs(60)), frames).finish(),
            Err(VideoErrorV1::Frames(VideoFrameError::NoFrames))
        ));
    }

    #[test]
    fn assembly_refuses_frames_past_the_budget_or_after_cancellation() {
        let plan = [(0, 0), (1, 1)];
        let frames = budget(23);
        let mut assembly =
            FrameAssemblyV1::new(&plan, 1, job(Duration::from_secs(60)), Arc::clone(&frames));
        assert!(!assembly.accept(0, 2, 2, &RED));
        assert!(!assembly.wants(1));
        assert!(matches!(assembly.failure, Some(VideoErrorV1::Busy(_))));

        let cancelled = job(Duration::from_secs(60));
        let mut assembly = FrameAssemblyV1::new(&plan, 1, Arc::clone(&cancelled), frames);
        cancelled.cancelled.store(true, Ordering::Release);
        assert!(!assembly.wants(0));
        assert!(!assembly.accept(0, 2, 2, &RED));
        assert!(matches!(assembly.failure, Some(VideoErrorV1::Cancelled)));
    }

    /// Whether a fake decode may proceed; tests open it to let stuck
    /// decodes return.
    #[derive(Default)]
    struct Gate {
        open: Mutex<bool>,
        opened: std::sync::Condvar,
    }

    impl Gate {
        fn wait(&self) {
            let mut open = self.open.lock().unwrap();
            while !*open {
                open = self.opened.wait(open).unwrap();
            }
        }

        fn release(&self) {
            *self.open.lock().unwrap() = true;
            self.opened.notify_all();
        }
    }

    /// Decodes by the first stream byte: 0 delivers every planned frame as a
    /// red 2x2 NV12 frame, 1 first waits for the gate, 2 panics.
    struct FakeDecoder {
        gate: Arc<Gate>,
        started: Arc<AtomicU64>,
    }

    impl VideoDecoderV1 for FakeDecoder {
        fn capacity(&self) -> (u32, u32) {
            (4096, 4096)
        }

        fn decode(
            &mut self,
            video: &PreparedVideoV1,
            _limits: &VideoLimitsV1,
            sink: &mut dyn FrameSinkV1,
        ) -> Result<(), VideoErrorV1> {
            self.started.fetch_add(1, Ordering::AcqRel);
            match video.stream[0] {
                1 => self.gate.wait(),
                2 => panic!("fake decoder failure"),
                _ => {}
            }
            for &sample in &video.timestamps {
                if sink.wants(sample) && !sink.accept(sample, 2, 2, &RED) {
                    return Err(VideoErrorV1::Decode("stopped".to_owned()));
                }
            }
            Ok(())
        }
    }

    struct FakePool {
        pool: VideoDecodeServiceV1,
        gate: Arc<Gate>,
        started: Arc<AtomicU64>,
    }

    impl Drop for FakePool {
        fn drop(&mut self) {
            self.gate.release();
        }
    }

    fn fake_pool(config: VideoDecodeConfigV1) -> FakePool {
        let gate = Arc::new(Gate::default());
        let started = Arc::new(AtomicU64::new(0));
        let (opened_gate, opened_started) = (Arc::clone(&gate), Arc::clone(&started));
        let pool = VideoDecodeServiceV1::start_with(config, move || {
            Ok(Box::new(FakeDecoder {
                gate: Arc::clone(&opened_gate),
                started: Arc::clone(&opened_started),
            }) as Box<dyn VideoDecoderV1>)
        })
        .unwrap();
        FakePool {
            pool,
            gate,
            started,
        }
    }

    fn config(workers: usize, queue: usize, deadline_ms: u64) -> VideoDecodeConfigV1 {
        VideoDecodeConfigV1 {
            workers,
            queue,
            deadline: Duration::from_millis(deadline_ms),
            frame_bytes: 1 << 20,
        }
    }

    fn fake_video(mode: u8) -> PreparedVideoV1 {
        PreparedVideoV1 {
            stream: vec![mode],
            unit_ends: vec![1, 1, 1],
            timestamps: vec![0, 1, 2],
            plan: vec![(0, 0), (2, 1), (1, 2)],
        }
    }

    fn never() -> bool {
        false
    }

    /// Waits until the fake decoders have started `count` decodes.
    fn started(pool: &FakePool, count: u64) {
        let until = Instant::now() + Duration::from_secs(10);
        while pool.started.load(Ordering::Acquire) < count {
            assert!(Instant::now() < until, "decodes did not start");
            thread::sleep(Duration::from_millis(1));
        }
    }

    #[test]
    fn pool_returns_frames_in_plan_order_with_their_labels() {
        let pool = fake_pool(config(2, 4, 10_000));
        assert_eq!(pool.pool.capacity(), (4096, 4096));
        let (frames, reservation) = pool
            .pool
            .decode(fake_video(0), limits(32), &never)
            .unwrap()
            .into_parts();
        let labels: Vec<_> = frames.iter().map(|frame| frame.label_seconds).collect();
        assert_eq!(labels, [0, 1, 2]);
        assert!(frames.iter().all(|frame| frame.rgb[..3] == [253, 0, 0]));
        assert_eq!(pool.pool.shared.budget.used.load(Ordering::Acquire), 36);
        drop(reservation);
        assert_eq!(pool.pool.shared.budget.used.load(Ordering::Acquire), 0);
    }

    #[test]
    fn a_stuck_decode_times_out_alone_while_another_worker_keeps_decoding() {
        let pool = fake_pool(config(2, 4, 200));
        let stuck = thread::scope(|scope| {
            let stuck = scope.spawn(|| {
                let started = Instant::now();
                let error = pool
                    .pool
                    .decode(fake_video(1), limits(32), &never)
                    .unwrap_err();
                (error, started.elapsed())
            });
            started(&pool, 1);
            // The other worker serves a new video while the first is stuck.
            let decoded = pool.pool.decode(fake_video(0), limits(32), &never).unwrap();
            assert_eq!(decoded.frames.len(), 3);
            stuck.join().unwrap()
        });
        assert!(
            matches!(stuck.0, VideoErrorV1::TimedOut(_)),
            "{:?}",
            stuck.0
        );
        assert!(stuck.1 >= Duration::from_millis(200) && stuck.1 < Duration::from_secs(5));
        assert_eq!(
            stuck.0.to_string(),
            "video decoding did not finish within 0.2 seconds"
        );
    }

    #[test]
    fn a_cancelled_request_stops_waiting_and_its_queued_job_is_skipped() {
        let pool = fake_pool(config(1, 4, 10_000));
        let cancelled = AtomicBool::new(false);
        thread::scope(|scope| {
            let blocked = scope.spawn(|| {
                pool.pool.decode(fake_video(1), limits(32), &|| {
                    cancelled.load(Ordering::Acquire)
                })
            });
            started(&pool, 1);
            let queued = scope.spawn(|| {
                pool.pool.decode(fake_video(0), limits(32), &|| {
                    cancelled.load(Ordering::Acquire)
                })
            });
            thread::sleep(Duration::from_millis(50));
            let cancelled_at = Instant::now();
            cancelled.store(true, Ordering::Release);
            assert!(matches!(
                blocked.join().unwrap(),
                Err(VideoErrorV1::Cancelled)
            ));
            assert!(matches!(
                queued.join().unwrap(),
                Err(VideoErrorV1::Cancelled)
            ));
            assert!(cancelled_at.elapsed() < Duration::from_secs(1));
        });
        // The running decode sees the cancellation at its next frame, and the
        // queued one never reaches the decoder.
        pool.gate.release();
        let decoded = pool.pool.decode(fake_video(0), limits(32), &never).unwrap();
        assert_eq!(decoded.frames.len(), 3);
        assert_eq!(pool.started.load(Ordering::Acquire), 2);
    }

    #[test]
    fn a_pool_of_stuck_workers_or_a_full_queue_refuses_at_once() {
        let pool = fake_pool(config(1, 1, 100));
        thread::scope(|scope| {
            let stuck = scope.spawn(|| pool.pool.decode(fake_video(1), limits(32), &never));
            started(&pool, 1);
            // One job may wait; the next finds the queue full.
            let waiting = scope.spawn(|| pool.pool.decode(fake_video(0), limits(32), &never));
            thread::sleep(Duration::from_millis(20));
            let refused = Instant::now();
            let full = pool.pool.decode(fake_video(0), limits(32), &never);
            assert!(matches!(full, Err(VideoErrorV1::Busy(_))), "{full:?}");
            assert!(refused.elapsed() < Duration::from_millis(50));
            assert!(matches!(
                stuck.join().unwrap(),
                Err(VideoErrorV1::TimedOut(_))
            ));
            assert!(matches!(
                waiting.join().unwrap(),
                Err(VideoErrorV1::TimedOut(_))
            ));
        });
        // The only worker is still inside its decode past the deadline.
        let refused = Instant::now();
        let exhausted = pool.pool.decode(fake_video(0), limits(32), &never);
        assert!(refused.elapsed() < Duration::from_millis(50));
        let Err(VideoErrorV1::Unavailable(message)) = exhausted else {
            panic!("{exhausted:?}");
        };
        assert_eq!(
            message,
            "all 1 video decode workers are stuck past their deadline"
        );
        // Once the call returns the worker serves again, skipping the
        // abandoned job it had queued.
        pool.gate.release();
        let until = Instant::now() + Duration::from_secs(10);
        let decoded = loop {
            match pool.pool.decode(fake_video(0), limits(32), &never) {
                Ok(decoded) => break decoded,
                Err(_) if Instant::now() < until => thread::sleep(Duration::from_millis(5)),
                Err(error) => panic!("{error}"),
            }
        };
        assert_eq!(decoded.frames.len(), 3);
        assert_eq!(pool.started.load(Ordering::Acquire), 2);
    }

    #[test]
    fn frames_that_do_not_fit_the_budget_are_refused_until_it_frees() {
        let pool = fake_pool(VideoDecodeConfigV1 {
            frame_bytes: 36,
            ..config(2, 4, 10_000)
        });
        let held = pool.pool.decode(fake_video(0), limits(32), &never).unwrap();
        let refused = pool.pool.decode(fake_video(0), limits(32), &never);
        let Err(error @ VideoErrorV1::Busy(_)) = refused else {
            panic!("{refused:?}");
        };
        assert_eq!(
            error.to_string(),
            "video decoding is busy: its 36 bytes of decoded frames do not fit the \
             36-byte frame budget beside other videos"
        );
        drop(held);
        assert!(pool.pool.decode(fake_video(0), limits(32), &never).is_ok());
    }

    #[test]
    fn a_panicking_decoder_fails_its_request_and_keeps_its_worker() {
        let pool = fake_pool(config(1, 4, 10_000));
        let panicked = pool.pool.decode(fake_video(2), limits(32), &never);
        assert!(
            matches!(panicked, Err(VideoErrorV1::Decode(_))),
            "{panicked:?}"
        );
        assert!(pool.pool.decode(fake_video(0), limits(32), &never).is_ok());
    }

    #[test]
    fn a_decoder_that_cannot_open_refuses_the_pool() {
        let refused = VideoDecodeServiceV1::start_with(config(2, 4, 100), || {
            Err(VideoErrorV1::Unavailable("no decoder here".to_owned()))
        });
        assert!(matches!(
            refused,
            Err(VideoErrorV1::Unavailable(message)) if message == "no decoder here"
        ));
    }

    #[test]
    fn broken_containers_are_refused_with_the_demuxer_reason() {
        let error = prepare_video_v1(&FIXTURE[..100], &limits(32)).unwrap_err();
        assert!(matches!(error, VideoErrorV1::Container(_)));
        assert_eq!(error.to_string(), "MP4 file is truncated");
    }
}
