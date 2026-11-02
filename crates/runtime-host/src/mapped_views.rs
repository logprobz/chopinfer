//! Model-neutral background warming and page-cache residency of mapped views.
//! The sampler reads no file bytes: it counts residency through windowed
//! `mincore` calls. Visibility requires ownership on a mount that is neither
//! idmapped nor overlay, or write access on a read-write mount. Hidden
//! residency is reported explicitly, never as resident bytes.

use std::io;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender};
use std::sync::{Arc, Mutex, Weak};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use crate::direct::{PrewarmRoomV1, PrewarmV1, prewarm_v1};
use crate::host_rows::{MappedRowsV1, authenticated_mappings_v1, page_bytes_v1};

const RESIDENCY_INTERVAL_V1: Duration = Duration::from_secs(60);
const RESIDENCY_PERCENT_V1: u64 = 95;
const STOP_WAIT_V1: Duration = Duration::from_secs(1);

/// Whether mapped views are warmed after serving starts.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum MappedPrewarmV1 {
    #[default]
    Off,
    On,
    Auto,
}

impl MappedPrewarmV1 {
    #[must_use]
    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "off" => Some(Self::Off),
            "on" => Some(Self::On),
            "auto" => Some(Self::Auto),
            _ => None,
        }
    }

    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::On => "on",
            Self::Auto => "auto",
        }
    }
}

#[must_use]
pub fn should_warm(mode: MappedPrewarmV1, room: &PrewarmRoomV1) -> bool {
    match mode {
        MappedPrewarmV1::Off => false,
        MappedPrewarmV1::On => true,
        MappedPrewarmV1::Auto => room.available_bytes.is_some_and(|available| {
            room.bytes
                .checked_add(room.margin_bytes)
                .is_some_and(|needed| available >= needed)
        }),
    }
}

/// One pass over live mappings. Both byte counts include whole mapped pages.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ResidencySampleV1 {
    pub mapped_bytes: u64,
    /// Absent if the kernel hides any mapping's residency.
    pub resident_bytes: Option<u64>,
    pub seconds: f64,
}

/// The latest successful residency sample, absent before the first pass.
#[derive(Clone, Debug, Default)]
pub struct MappedResidencyReportV1(Arc<Mutex<Option<ResidencySampleV1>>>);

impl MappedResidencyReportV1 {
    pub(crate) fn publish(&self, sample: ResidencySampleV1) {
        if let Ok(mut held) = self.0.lock() {
            *held = Some(sample);
        }
    }

    #[must_use]
    pub fn sample(&self) -> Option<ResidencySampleV1> {
        self.0.lock().ok().and_then(|sample| *sample)
    }

    #[must_use]
    pub fn prometheus(&self) -> String {
        let Some(sample) = self.sample() else {
            return String::new();
        };
        let mut output = String::new();
        for (name, value) in [
            ("chop_mapped_view_bytes", sample.mapped_bytes),
            (
                "chop_mapped_view_residency_visible",
                u64::from(sample.resident_bytes.is_some()),
            ),
        ] {
            output.push_str(&format!("# TYPE {name} gauge\n{name} {value}\n"));
        }
        if let Some(resident) = sample.resident_bytes {
            let name = "chop_mapped_view_resident_bytes";
            output.push_str(&format!("# TYPE {name} gauge\n{name} {resident}\n"));
        }
        let name = "chop_mapped_view_residency_sample_seconds";
        output.push_str(&format!("# TYPE {name} gauge\n{name} {}\n", sample.seconds));
        output
    }

    fn publish_result_v1(
        &self,
        result: io::Result<Option<(ResidencySampleV1, Option<String>)>>,
    ) -> Option<(ResidencySampleV1, Option<String>)> {
        match result {
            Ok(Some((sample, hidden))) => {
                self.publish(sample);
                Some((sample, hidden))
            }
            Ok(None) => None,
            Err(error) => {
                runtime_log::warn!(
                    "mapped residency: sampling failed: {error}; keeping the previous sample"
                );
                None
            }
        }
    }
}

fn mapped_page_bytes_v1(mapping: &MappedRowsV1) -> u64 {
    let page = page_bytes_v1() as u64;
    (mapping.file_offset() % page + mapping.len()).div_ceil(page) * page
}

fn sample_v1(
    mappings: &[Arc<MappedRowsV1>],
    stop: &AtomicBool,
) -> io::Result<Option<(ResidencySampleV1, Option<String>)>> {
    sample_with_v1(mappings, stop, MappedRowsV1::resident_pages)
}

fn sample_with_v1(
    mappings: &[Arc<MappedRowsV1>],
    stop: &AtomicBool,
    mut resident_pages: impl FnMut(&MappedRowsV1) -> io::Result<(u64, u64)>,
) -> io::Result<Option<(ResidencySampleV1, Option<String>)>> {
    let started = Instant::now();
    let page = page_bytes_v1() as u64;
    let mut mapped_bytes = 0;
    let mut resident_bytes = Some(0);
    let mut hidden = None;
    for mapping in mappings {
        if stop.load(Ordering::Acquire) {
            return Ok(None);
        }
        mapped_bytes += mapped_page_bytes_v1(mapping);
        if resident_bytes.is_none() {
            continue;
        }
        match resident_pages(mapping) {
            Ok((resident, _)) => {
                resident_bytes = resident_bytes.map(|bytes| bytes + resident * page);
            }
            Err(error) if error.kind() == io::ErrorKind::PermissionDenied => {
                resident_bytes = None;
                hidden.get_or_insert_with(|| error.to_string());
            }
            Err(error) => return Err(error),
        }
    }
    if stop.load(Ordering::Acquire) {
        return Ok(None);
    }
    Ok(Some((
        ResidencySampleV1 {
            mapped_bytes,
            resident_bytes,
            seconds: started.elapsed().as_secs_f64(),
        },
        hidden,
    )))
}

fn resident_enough_v1(resident: u64, mapped: u64) -> bool {
    u128::from(resident) * 100 >= u128::from(mapped) * u128::from(RESIDENCY_PERCENT_V1)
}

#[derive(Debug, Eq, PartialEq)]
enum ResidencyChangeV1 {
    Low,
    Recovered,
}

#[derive(Default)]
struct ResidencyTrackerV1 {
    armed: bool,
    low: bool,
    warned_hidden: bool,
}

impl ResidencyTrackerV1 {
    fn hidden_warning(&mut self, reason: Option<&str>) -> Option<String> {
        let reason = reason.filter(|_| !self.warned_hidden)?;
        self.warned_hidden = true;
        Some(format!(
            "mapped residency: warning residency-hidden: {reason}; the kernel shows \
             page-cache residency only to a process that owns the file on a mount \
             that is neither idmapped nor overlay, or may write it on a read-write \
             mount; run the server as the checkpoint's owner on a plain mount to see it"
        ))
    }

    fn observe(&mut self, sample: ResidencySampleV1) -> Option<ResidencyChangeV1> {
        let resident = sample.resident_bytes?;
        if resident_enough_v1(resident, sample.mapped_bytes) {
            self.armed = true;
            if std::mem::take(&mut self.low) {
                return Some(ResidencyChangeV1::Recovered);
            }
        } else if self.armed && !self.low {
            self.low = true;
            return Some(ResidencyChangeV1::Low);
        }
        None
    }
}

fn residency_line_v1(sample: ResidencySampleV1) -> String {
    let mapped = sample.mapped_bytes;
    let seconds = sample.seconds;
    match sample.resident_bytes {
        Some(resident) => {
            let percent = if mapped == 0 {
                100.0
            } else {
                resident as f64 / mapped as f64 * 100.0
            };
            format!(
                "{resident} resident bytes, {mapped} mapped bytes, {percent:.2}%, {seconds:.6} sample seconds"
            )
        }
        None => {
            format!("resident bytes hidden, {mapped} mapped bytes, {seconds:.6} sample seconds")
        }
    }
}

fn sample_loop_v1(
    mappings: Vec<Weak<MappedRowsV1>>,
    report: MappedResidencyReportV1,
    interval: Duration,
    stop: Arc<AtomicBool>,
    stopped: Receiver<()>,
) {
    let mut first = true;
    let mut tracker = ResidencyTrackerV1::default();
    loop {
        if stop.load(Ordering::Acquire) {
            break;
        }
        // Strong references live only for the sample, never across the wait.
        let result = sample_v1(
            &mappings
                .iter()
                .filter_map(Weak::upgrade)
                .collect::<Vec<_>>(),
            &stop,
        );
        if let Some((sample, hidden)) = report.publish_result_v1(result) {
            if first {
                runtime_log::info!(
                    "mapped residency: {}; interval {} seconds",
                    residency_line_v1(sample),
                    interval.as_secs_f64()
                );
                first = false;
            }
            if let Some(line) = tracker.hidden_warning(hidden.as_deref()) {
                runtime_log::warn!("{line}");
            }
            match tracker.observe(sample) {
                Some(ResidencyChangeV1::Low) => runtime_log::warn!(
                    "mapped residency: warning resident-low: {}",
                    residency_line_v1(sample)
                ),
                Some(ResidencyChangeV1::Recovered) => runtime_log::info!(
                    "mapped residency: resident-recovered: {}",
                    residency_line_v1(sample)
                ),
                None => {}
            }
        }
        match stopped.recv_timeout(interval) {
            Err(RecvTimeoutError::Timeout) => {}
            Ok(()) | Err(RecvTimeoutError::Disconnected) => break,
        }
    }
}

/// The background warm's decision or result. A stopped warm reports its
/// partial work through [`PrewarmV1::warmed_bytes`] and [`PrewarmV1::stopped`].
#[derive(Clone, Debug, PartialEq)]
pub enum MappedPrewarmOutcomeV1 {
    Declined(PrewarmRoomV1),
    Warmed(PrewarmV1),
    Failed(String),
}

fn prewarm_done_line_v1(
    mode: MappedPrewarmV1,
    warm: &PrewarmV1,
    mapped: u64,
) -> (runtime_log::Level, String) {
    let resident_bytes = warm.resident_bytes.filter(|_| !warm.stopped);
    let level = if resident_bytes.is_some_and(|resident| !resident_enough_v1(resident, mapped)) {
        runtime_log::Level::Warn
    } else {
        runtime_log::Level::Info
    };
    let json = serde_json::json!({
        "mode": mode.name(),
        "bytes": warm.bytes,
        "seconds": warm.seconds,
        "resident_bytes": resident_bytes,
        "available_bytes": warm.available_bytes,
        "margin_bytes": warm.margin_bytes,
        "stopped": warm.stopped,
    });
    (level, format!("mapped prewarm: {json}"))
}

fn warm_v1(
    mode: MappedPrewarmV1,
    mappings: &[Arc<MappedRowsV1>],
    room: PrewarmRoomV1,
    stop: &AtomicBool,
) -> MappedPrewarmOutcomeV1 {
    if let Some(available) = room.available_bytes.filter(|_| room.is_short()) {
        let action = if mode == MappedPrewarmV1::Auto {
            "not warming (auto)"
        } else {
            "warming anyway (on)"
        };
        runtime_log::warn!(
            "mapped prewarm: warning host-ram-short: {available} bytes available, below the \
             {} mapped bytes plus a {}-byte margin; {action}",
            room.bytes,
            room.margin_bytes
        );
    } else if room.available_bytes.is_none() && mode == MappedPrewarmV1::Auto {
        runtime_log::warn!(
            "mapped prewarm: warning host-ram-short: MemAvailable is unreadable; not warming (auto)"
        );
    }
    if !should_warm(mode, &room) {
        return MappedPrewarmOutcomeV1::Declined(room);
    }
    match prewarm_v1(mappings, &room, Some(stop)) {
        Ok(warm) => {
            let mapped = mappings.iter().map(|rows| mapped_page_bytes_v1(rows)).sum();
            let (level, line) = prewarm_done_line_v1(mode, &warm, mapped);
            if level == runtime_log::Level::Warn {
                runtime_log::warn!("{line}");
            } else {
                runtime_log::info!("{line}");
            }
            MappedPrewarmOutcomeV1::Warmed(warm)
        }
        Err(error) => {
            runtime_log::warn!("mapped prewarm: failed: {error}; serving continues");
            MappedPrewarmOutcomeV1::Failed(error)
        }
    }
}

/// Starts sampling every live authenticated mapping and optionally warms it.
pub struct MappedViewsV1 {
    pub mode: MappedPrewarmV1,
    pub report: MappedResidencyReportV1,
}

impl MappedViewsV1 {
    #[must_use]
    pub fn start(self) -> MappedViewsGuardV1 {
        self.start_with_v1(
            authenticated_mappings_v1(),
            PrewarmRoomV1::measure,
            RESIDENCY_INTERVAL_V1,
        )
    }

    fn start_with_v1(
        self,
        mappings: Vec<Arc<MappedRowsV1>>,
        room: impl FnOnce(u64) -> PrewarmRoomV1 + Send + 'static,
        interval: Duration,
    ) -> MappedViewsGuardV1 {
        let (stop_sender, stopped) = mpsc::channel();
        let (exit_sender, exits) = mpsc::channel();
        let mut guard = MappedViewsGuardV1 {
            stop: Arc::new(AtomicBool::new(false)),
            stop_sender: Some(stop_sender),
            exits,
            residency_thread: None,
            prewarm_thread: None,
            outcome: None,
        };
        if mappings.is_empty() {
            if self.mode != MappedPrewarmV1::Off {
                runtime_log::info!("mapped prewarm: no mapped views");
            }
            return guard;
        }
        let weak = mappings.iter().map(Arc::downgrade).collect();
        let stop = Arc::clone(&guard.stop);
        let exit = exit_sender.clone();
        match std::thread::Builder::new()
            .name("chopinfer-mapped-residency".into())
            .spawn(move || {
                let _exit = exit;
                sample_loop_v1(weak, self.report, interval, stop, stopped);
            }) {
            Ok(thread) => guard.residency_thread = Some(thread),
            Err(error) => runtime_log::warn!(
                "mapped residency: could not start sampler: {error}; serving continues"
            ),
        }
        if self.mode != MappedPrewarmV1::Off {
            let stop = Arc::clone(&guard.stop);
            match std::thread::Builder::new()
                .name("chopinfer-mapped-prewarm".into())
                .spawn(move || {
                    let _exit = exit_sender;
                    let room = room(mappings.iter().map(|rows| rows.len()).sum());
                    warm_v1(self.mode, &mappings, room, &stop)
                }) {
                Ok(thread) => guard.prewarm_thread = Some(thread),
                Err(error) => {
                    runtime_log::warn!(
                        "mapped prewarm: could not start worker: {error}; serving continues"
                    );
                    guard.outcome = Some(MappedPrewarmOutcomeV1::Failed(error.to_string()));
                }
            }
        }
        guard
    }
}

/// Stops both workers on drop, waiting at most one second in total. A worker
/// blocked in a file operation is left to finish that operation and stop.
pub struct MappedViewsGuardV1 {
    stop: Arc<AtomicBool>,
    stop_sender: Option<Sender<()>>,
    exits: Receiver<()>,
    residency_thread: Option<JoinHandle<()>>,
    prewarm_thread: Option<JoinHandle<MappedPrewarmOutcomeV1>>,
    outcome: Option<MappedPrewarmOutcomeV1>,
}

impl MappedViewsGuardV1 {
    /// Blocks until the warm thread finishes and returns its outcome, for
    /// tests and callers that require warming to finish before proceeding.
    /// Returns `None` when no warm was requested or no mappings existed.
    pub fn wait_for_prewarm(&mut self) -> Option<MappedPrewarmOutcomeV1> {
        if let Some(thread) = self.prewarm_thread.take() {
            self.outcome = Some(thread.join().unwrap_or_else(|_| {
                let error = "the prewarm thread panicked";
                runtime_log::warn!("mapped prewarm: failed: {error}; serving continues");
                MappedPrewarmOutcomeV1::Failed(error.into())
            }));
        }
        self.outcome.clone()
    }
}

impl Drop for MappedViewsGuardV1 {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        self.stop_sender.take();
        match self.exits.recv_timeout(STOP_WAIT_V1) {
            Err(RecvTimeoutError::Disconnected) => {
                if let Some(thread) = self.residency_thread.take() {
                    let _ = thread.join();
                }
                self.wait_for_prewarm();
            }
            Ok(()) | Err(RecvTimeoutError::Timeout) => runtime_log::warn!(
                "mapped views: a worker is still running after {STOP_WAIT_V1:?}; leaving it to finish its current operation"
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::host_rows::{map_authenticated_rows_v1, map_rows_v1};
    use crate::page_cache::{CacheAdviceV1, advise_v1};
    use std::fs::File;
    use std::path::PathBuf;

    const TEST_INTERVAL: Duration = Duration::from_millis(5);
    const TEST_TIMEOUT: Duration = Duration::from_secs(5);

    #[test]
    fn modes_round_trip_and_refuse_other_values() {
        assert_eq!(MappedPrewarmV1::default(), MappedPrewarmV1::Off);
        for mode in [
            MappedPrewarmV1::Off,
            MappedPrewarmV1::On,
            MappedPrewarmV1::Auto,
        ] {
            assert_eq!(MappedPrewarmV1::parse(mode.name()), Some(mode));
        }
        for value in ["", "ON", "Auto", " off", "on ", "true", "false", "yes"] {
            assert_eq!(MappedPrewarmV1::parse(value), None, "{value}");
        }
    }

    #[test]
    fn warming_policy_requires_known_sufficient_room_only_for_auto() {
        for (available_bytes, fits) in [
            (Some(109), false),
            (Some(110), true),
            (Some(111), true),
            (None, false),
        ] {
            let room = PrewarmRoomV1 {
                bytes: 100,
                available_bytes,
                margin_bytes: 10,
            };
            assert!(!should_warm(MappedPrewarmV1::Off, &room));
            assert!(should_warm(MappedPrewarmV1::On, &room));
            assert_eq!(should_warm(MappedPrewarmV1::Auto, &room), fits);
        }
        assert!(!should_warm(
            MappedPrewarmV1::Auto,
            &PrewarmRoomV1 {
                bytes: u64::MAX,
                available_bytes: Some(u64::MAX),
                margin_bytes: 1,
            }
        ));
    }

    #[test]
    fn tracker_arms_warns_once_and_recovers_ignoring_hidden_samples() {
        let mut tracker = ResidencyTrackerV1::default();
        let page = page_bytes_v1() as u64;
        for (resident, expected) in [
            (Some(94), None),
            (None, None),
            (Some(94), None),
            (Some(95), None),
            (None, None),
            (Some(94), Some(ResidencyChangeV1::Low)),
            (Some(0), None),
            (None, None),
            (Some(94), None),
            (Some(95), Some(ResidencyChangeV1::Recovered)),
            (Some(100), None),
            (Some(94), Some(ResidencyChangeV1::Low)),
        ] {
            assert_eq!(
                tracker.observe(ResidencySampleV1 {
                    mapped_bytes: 100 * page,
                    resident_bytes: resident.map(|pages| pages * page),
                    seconds: 0.25,
                }),
                expected
            );
        }
        assert!(resident_enough_v1(u64::MAX, u64::MAX));
    }

    #[test]
    fn tracker_warns_about_hidden_residency_once() {
        let mut tracker = ResidencyTrackerV1::default();
        assert_eq!(tracker.hidden_warning(None), None);
        let line = tracker
            .hidden_warning(Some("fixture access denied"))
            .unwrap();
        assert_eq!(
            line,
            "mapped residency: warning residency-hidden: fixture access denied; \
            the kernel shows page-cache residency only to a process that owns the file on a \
            mount that is neither idmapped nor overlay, or may write it on a read-write mount; \
            run the server as the checkpoint's owner on a plain mount to see it"
        );
        assert_eq!(tracker.hidden_warning(Some("fixture access denied")), None);
        assert_eq!(tracker.hidden_warning(None), None);
        assert_eq!(tracker.hidden_warning(Some("another hidden mapping")), None);
    }

    #[test]
    fn prewarm_done_line_reports_all_fields_and_warns_only_for_visible_low_residency() {
        use runtime_log::Level::{Info, Warn};
        let page = page_bytes_v1() as u64;
        let mapped = 100 * page;
        for (resident_pages, stopped, level) in [
            (Some(100), false, Info),
            (Some(95), false, Info),
            (Some(94), false, Warn),
            (None, false, Info),
            (None, true, Info),
        ] {
            let resident_bytes = resident_pages.map(|pages| pages * page);
            for (mode, name, available_bytes) in [
                (MappedPrewarmV1::On, "on", None),
                (MappedPrewarmV1::Auto, "auto", Some(2 * mapped)),
            ] {
                let warm = PrewarmV1 {
                    bytes: mapped - 3,
                    warmed_bytes: if stopped { 0 } else { mapped },
                    stopped,
                    seconds: 0.25,
                    resident_bytes,
                    available_bytes,
                    margin_bytes: page,
                };
                let (actual_level, text) = prewarm_done_line_v1(mode, &warm, mapped);
                assert_eq!(actual_level, level);
                let json: serde_json::Value =
                    serde_json::from_str(text.strip_prefix("mapped prewarm: ").unwrap()).unwrap();
                assert_eq!(json.as_object().unwrap().len(), 7);
                assert_eq!(
                    json,
                    serde_json::json!({
                        "mode": name,
                        "bytes": mapped - 3,
                        "seconds": 0.25,
                        "resident_bytes": resident_bytes,
                        "available_bytes": available_bytes,
                        "margin_bytes": page,
                        "stopped": stopped,
                    })
                );
                if resident_bytes.is_none() {
                    assert!(text.contains("\"resident_bytes\":null"));
                }
                if stopped {
                    assert!(text.contains("\"stopped\":true"));
                }
            }
        }
    }

    #[test]
    fn report_renders_only_observed_gauges_and_keeps_the_previous_sample_on_error() {
        let report = MappedResidencyReportV1::default();
        assert_eq!(report.sample(), None);
        assert_eq!(report.prometheus(), "");
        let sample = ResidencySampleV1 {
            mapped_bytes: 8192,
            resident_bytes: Some(4096),
            seconds: 0.25,
        };
        report.publish_result_v1(Ok(Some((sample, None))));
        assert_eq!(report.sample(), Some(sample));
        assert_eq!(
            report.prometheus(),
            concat!(
                "# TYPE chop_mapped_view_bytes gauge\nchop_mapped_view_bytes 8192\n",
                "# TYPE chop_mapped_view_residency_visible gauge\nchop_mapped_view_residency_visible 1\n",
                "# TYPE chop_mapped_view_resident_bytes gauge\nchop_mapped_view_resident_bytes 4096\n",
                "# TYPE chop_mapped_view_residency_sample_seconds gauge\nchop_mapped_view_residency_sample_seconds 0.25\n",
            )
        );
        assert!(
            report
                .publish_result_v1(Err(io::Error::other("sample failed")))
                .is_none()
        );
        assert_eq!(report.sample(), Some(sample));
        report.publish(ResidencySampleV1 {
            resident_bytes: None,
            ..sample
        });
        assert_eq!(
            report.prometheus(),
            concat!(
                "# TYPE chop_mapped_view_bytes gauge\nchop_mapped_view_bytes 8192\n",
                "# TYPE chop_mapped_view_residency_visible gauge\nchop_mapped_view_residency_visible 0\n",
                "# TYPE chop_mapped_view_residency_sample_seconds gauge\nchop_mapped_view_residency_sample_seconds 0.25\n",
            )
        );
    }

    struct Temp(PathBuf);

    impl Drop for Temp {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    fn mapping(label: &str) -> (Temp, Arc<MappedRowsV1>, bool) {
        let temp =
            Temp(std::env::temp_dir().join(format!("mapped-views-{label}-{}", std::process::id())));
        let page = page_bytes_v1();
        std::fs::write(&temp.0, vec![7; 5 * page]).unwrap();
        let file = File::open(&temp.0).unwrap();
        file.sync_all().unwrap();
        advise_v1(&file, 0, 0, CacheAdviceV1::DontNeed).unwrap();
        let rows = map_authenticated_rows_v1(&file, 3, (4 * page) as u64, [71; 32]).unwrap();
        let (resident, pages) = rows.resident_pages().unwrap();
        assert_eq!(pages, 5);
        let cold = resident == 0;
        if !cold {
            eprintln!("eviction advice did not remove every page; cold-state assertion omitted");
        }
        (temp, rows, cold)
    }

    fn wait_for_sample(
        report: &MappedResidencyReportV1,
        predicate: impl Fn(ResidencySampleV1) -> bool,
    ) -> ResidencySampleV1 {
        let deadline = Instant::now() + TEST_TIMEOUT;
        loop {
            if let Some(sample) = report.sample().filter(|sample| predicate(*sample)) {
                return sample;
            }
            assert!(
                Instant::now() < deadline,
                "sample was not published: {:?}",
                report.sample()
            );
            std::thread::sleep(TEST_INTERVAL);
        }
    }

    fn short_room(bytes: u64) -> PrewarmRoomV1 {
        PrewarmRoomV1 {
            bytes,
            available_bytes: Some(0),
            margin_bytes: 10,
        }
    }

    #[test]
    fn sampling_stops_between_mappings_without_publishing_partial_results() {
        let (_temp, rows, _) = mapping("sample-stop");
        let mappings = vec![Arc::clone(&rows); 3];
        let report = MappedResidencyReportV1::default();
        let previous = ResidencySampleV1 {
            mapped_bytes: 8192,
            resident_bytes: Some(4096),
            seconds: 0.25,
        };
        report.publish(previous);
        // Stop before the pass, between mappings, and after the last mapping.
        for stop_after in [0, 1, 3] {
            let stop = AtomicBool::new(stop_after == 0);
            let mut sampled = 0;
            let result = sample_with_v1(&mappings, &stop, |_| {
                sampled += 1;
                if sampled == stop_after {
                    stop.store(true, Ordering::Release);
                }
                Ok((0, 5))
            });
            assert_eq!(sampled, stop_after);
            assert!(result.as_ref().unwrap().is_none());
            assert!(report.publish_result_v1(result).is_none());
            assert_eq!(report.sample(), Some(previous));
        }
    }

    #[test]
    fn hidden_residency_skips_remaining_reads_but_counts_all_mapped_pages() {
        let (_temp, rows, _) = mapping("sample-hidden");
        let mappings = vec![Arc::clone(&rows); 3];
        let stop = AtomicBool::new(false);
        let mut sampled = 0;
        let (sample, hidden) = sample_with_v1(&mappings, &stop, |_| {
            sampled += 1;
            assert_eq!(sampled, 1, "read residency after a hidden result");
            Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "fixture hidden",
            ))
        })
        .unwrap()
        .unwrap();
        assert_eq!(sampled, 1);
        assert_eq!(sample.mapped_bytes, 15 * page_bytes_v1() as u64);
        assert_eq!(sample.resident_bytes, None);
        assert_eq!(hidden.as_deref(), Some("fixture hidden"));
    }

    #[test]
    fn background_on_warms_and_publishes_whole_pages_even_when_room_is_short() {
        let (_temp, rows, cold) = mapping("on");
        let (before, _) = sample_v1(&[Arc::clone(&rows)], &AtomicBool::new(false))
            .unwrap()
            .unwrap();
        if cold {
            assert_eq!(before.resident_bytes, Some(0));
        }
        let report = MappedResidencyReportV1::default();
        let mut guard = MappedViewsV1 {
            mode: MappedPrewarmV1::On,
            report: report.clone(),
        }
        .start_with_v1(vec![Arc::clone(&rows)], short_room, TEST_INTERVAL);
        let Some(MappedPrewarmOutcomeV1::Warmed(warm)) = guard.wait_for_prewarm() else {
            panic!("on did not warm");
        };
        let (resident, pages) = rows.resident_pages().unwrap();
        assert_eq!(resident, pages);
        assert_eq!(warm.bytes, rows.len());
        assert_eq!(warm.warmed_bytes, 5 * page_bytes_v1() as u64);
        assert_eq!(warm.resident_bytes, Some(warm.warmed_bytes));
        assert!(!warm.stopped);
        let sample = wait_for_sample(&report, |sample| {
            sample.resident_bytes == Some(sample.mapped_bytes)
        });
        assert_eq!(sample.mapped_bytes, warm.warmed_bytes);
        assert!(sample.seconds >= 0.0);
    }

    #[test]
    fn background_auto_declines_short_room_and_leaves_cold_pages_cold() {
        let (_temp, rows, cold) = mapping("auto");
        let report = MappedResidencyReportV1::default();
        let mut guard = MappedViewsV1 {
            mode: MappedPrewarmV1::Auto,
            report: report.clone(),
        }
        .start_with_v1(vec![Arc::clone(&rows)], short_room, TEST_INTERVAL);
        assert_eq!(
            guard.wait_for_prewarm(),
            Some(MappedPrewarmOutcomeV1::Declined(short_room(rows.len())))
        );
        let sample = wait_for_sample(&report, |_| true);
        assert_eq!(sample.mapped_bytes, 5 * page_bytes_v1() as u64);
        if cold {
            assert_eq!(sample.resident_bytes, Some(0));
            assert_eq!(rows.resident_pages().unwrap().0, 0);
        }
    }

    #[test]
    fn stop_before_background_warm_reads_zero_bytes() {
        let (_temp, rows, cold) = mapping("stop");
        let (ready, waiting) = mpsc::channel();
        let (release, released) = mpsc::channel();
        let mut guard = MappedViewsV1 {
            mode: MappedPrewarmV1::On,
            report: MappedResidencyReportV1::default(),
        }
        .start_with_v1(
            vec![Arc::clone(&rows)],
            move |bytes| {
                ready.send(()).unwrap();
                released.recv().unwrap();
                short_room(bytes)
            },
            TEST_INTERVAL,
        );
        waiting.recv_timeout(TEST_TIMEOUT).unwrap();
        guard.stop.store(true, Ordering::Release);
        release.send(()).unwrap();
        let Some(MappedPrewarmOutcomeV1::Warmed(warm)) = guard.wait_for_prewarm() else {
            panic!("on did not report the stopped warm");
        };
        assert!(warm.stopped);
        assert_eq!(warm.warmed_bytes, 0);
        assert_eq!(warm.resident_bytes, None);
        if cold {
            assert_eq!(rows.resident_pages().unwrap().0, 0);
        }
    }

    #[test]
    fn off_samples_without_retaining_dropped_mappings() {
        let (_temp, rows, _) = mapping("off");
        let weak = Arc::downgrade(&rows);
        let report = MappedResidencyReportV1::default();
        let mut guard = MappedViewsV1 {
            mode: MappedPrewarmV1::Off,
            report: report.clone(),
        }
        .start_with_v1(vec![rows], |_| panic!("off measured room"), TEST_INTERVAL);
        assert_eq!(guard.wait_for_prewarm(), None);
        let sample = wait_for_sample(&report, |sample| sample.mapped_bytes == 0);
        assert_eq!(sample.resident_bytes, Some(0));
        assert!(weak.upgrade().is_none());
    }

    #[test]
    fn no_mappings_start_no_workers_or_samples() {
        for mode in [
            MappedPrewarmV1::Off,
            MappedPrewarmV1::On,
            MappedPrewarmV1::Auto,
        ] {
            let report = MappedResidencyReportV1::default();
            let mut guard = MappedViewsV1 {
                mode,
                report: report.clone(),
            }
            .start_with_v1(
                vec![],
                |_| panic!("no mappings measured room"),
                TEST_INTERVAL,
            );
            assert!(guard.residency_thread.is_none());
            assert!(guard.prewarm_thread.is_none());
            assert_eq!(guard.wait_for_prewarm(), None);
            assert_eq!(report.sample(), None);
        }
    }

    #[test]
    fn drop_stops_the_sampler_without_waiting_for_its_interval() {
        let (_temp, rows, _) = mapping("drop");
        let report = MappedResidencyReportV1::default();
        let guard = MappedViewsV1 {
            mode: MappedPrewarmV1::Off,
            report: report.clone(),
        }
        .start_with_v1(vec![Arc::clone(&rows)], short_room, RESIDENCY_INTERVAL_V1);
        wait_for_sample(&report, |_| true);
        let (finished, done) = mpsc::channel();
        let thread = std::thread::spawn(move || {
            drop(guard);
            finished.send(()).unwrap();
        });
        done.recv_timeout(TEST_TIMEOUT).unwrap();
        thread.join().unwrap();
        assert_eq!(Arc::strong_count(&rows), 1);
        // Only the process registry may retain a weak reference after shutdown.
        assert_eq!(Arc::weak_count(&rows), 1);
    }

    #[test]
    fn drop_bounds_the_wait_for_a_blocked_warm_and_stops_it_after_release() {
        let (_temp, rows, cold) = mapping("blocked");
        let (ready, waiting) = mpsc::channel();
        let (release, released) = mpsc::channel();
        let guard = MappedViewsV1 {
            mode: MappedPrewarmV1::On,
            report: MappedResidencyReportV1::default(),
        }
        .start_with_v1(
            vec![Arc::clone(&rows)],
            move |bytes| {
                ready.send(()).unwrap();
                released.recv().unwrap();
                short_room(bytes)
            },
            TEST_INTERVAL,
        );
        waiting.recv_timeout(TEST_TIMEOUT).unwrap();
        let stop = Arc::clone(&guard.stop);
        let (finished, done) = mpsc::channel();
        let thread = std::thread::spawn(move || {
            drop(guard);
            finished.send(()).unwrap();
        });
        let bounded = done.recv_timeout(TEST_TIMEOUT);
        release.send(()).unwrap();
        thread.join().unwrap();
        bounded.unwrap();
        assert!(stop.load(Ordering::Acquire));
        let deadline = Instant::now() + TEST_TIMEOUT;
        while Arc::strong_count(&rows) > 1 {
            assert!(
                Instant::now() < deadline,
                "stopped warm retained its mapping"
            );
            std::thread::sleep(TEST_INTERVAL);
        }
        if cold {
            assert_eq!(rows.resident_pages().unwrap().0, 0);
        }
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn non_owned_read_only_file_sample_is_hidden() {
        use std::os::unix::fs::MetadataExt;
        let file = File::open("/etc/passwd").unwrap();
        let metadata = file.metadata().unwrap();
        if crate::host_rows::filesystem_uid_v1() == Some(metadata.uid()) {
            eprintln!("skipped non-owned sample case: file owner equals process fsuid");
            return;
        }
        if crate::host_rows::file_writable_v1(&file) {
            eprintln!("skipped non-owned sample case: file is writable");
            return;
        }
        let rows = map_rows_v1(&file, 0, metadata.len()).unwrap();
        let (_temp, visible, _) = mapping("hidden-mix");
        // A hidden mapping hides the whole sample in either iteration order.
        for mappings in [
            vec![Arc::clone(&rows), Arc::clone(&visible)],
            vec![visible, rows],
        ] {
            let (sample, hidden) = sample_v1(&mappings, &AtomicBool::new(false))
                .unwrap()
                .unwrap();
            let expected: u64 = mappings.iter().map(|rows| mapped_page_bytes_v1(rows)).sum();
            assert_eq!(sample.mapped_bytes, expected);
            assert_eq!(sample.resident_bytes, None);
            assert!(hidden.unwrap().contains("page-cache residency is hidden"));
            let report = MappedResidencyReportV1::default();
            report.publish(sample);
            let text = report.prometheus();
            assert!(text.contains("chop_mapped_view_residency_visible 0\n"));
            assert!(!text.contains("chop_mapped_view_resident_bytes"));
        }
        eprintln!("non-owned /etc/passwd sample case ran: residency hidden, resident gauge absent");
    }
}
