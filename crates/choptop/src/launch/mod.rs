//! Starting serve processes from the loader: who holds each GPU and port,
//! where each instance goes, the command line each one runs, and the
//! launches themselves.
//!
//! A switch queues one launch per instance. Each waits until what it
//! replaces has exited: the serve processes on its GPUs or port, which
//! choptop sends SIGTERM through a pidfd, and earlier launches, which it
//! cancels or stops through their process group. It then checks its port
//! once more and starts. A launch that is queued or starting is a
//! [`Reservation`], which the occupancy and the placement read.

pub mod command;
pub mod occupancy;
pub mod placement;
pub mod process;
pub mod records;
pub mod switch;

use std::os::unix::process::ExitStatusExt;
use std::path::{Path, PathBuf};
use std::process::{Child, ExitStatus};

use crate::instances::{Followup, Launched, Registry, Tone};
use process::Identity;
use records::{Record, Records};

/// Seconds serve drains admitted requests after SIGTERM by default.
pub const DRAIN_SECONDS: u32 = 30;
/// Seconds between rereads of a running launch's log.
const TAIL_SECONDS: f64 = 1.0;

/// A choptop launch that holds its GPUs and port from the moment it is
/// queued until it is cancelled or its process exits, including while it
/// prepares its package and shows neither a GPU process nor a listener.
#[derive(Clone, Debug, PartialEq)]
pub struct Reservation {
    /// The row it shows as: `launch A` until it has a pid, `proc PID` after.
    pub label: String,
    /// What it serves, for the loader's placement rows.
    pub model: String,
    /// The UUIDs of its GPUs.
    pub gpus: Vec<String>,
    pub port: u16,
    /// Its process once started, by pid and start ticks; `None` while it
    /// waits for what it replaces to exit.
    pub process: Option<(u32, u64)>,
    /// The launch's id.
    pub launch: u64,
}

impl Reservation {
    /// Whether it has not started yet, so a switch cancels it rather than
    /// stopping it.
    pub fn waiting(&self) -> bool {
        self.process.is_none()
    }
}

/// The reservation of launch `id` among `reservations`.
pub fn reserved(reservations: &[Reservation], id: u64) -> Option<&Reservation> {
    reservations
        .iter()
        .find(|reservation| reservation.launch == id)
}

/// What a launch waits for before it starts.
#[derive(Clone, Debug, PartialEq)]
pub enum Wait {
    /// A serve process the switch stops, as recorded when it was confirmed.
    Serve(Identity),
    /// An earlier launch the switch cancels or stops, by its id.
    Launch(u64),
}

/// Where a launch is.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Phase {
    Waiting,
    Running,
    /// Sent SIGTERM at this time.
    Stopping(f64),
    /// Exited on its own at this time, with its status when known.
    Exited(f64, Option<i32>),
}

/// One instance to queue.
#[derive(Clone, Debug)]
pub struct Queued {
    pub record: Record,
    /// The chopinfer it runs.
    pub binary: PathBuf,
    /// Its serve arguments, `--listen` included.
    pub arguments: Vec<String>,
    /// Variables added to choptop's own environment, secrets included.
    pub environment: Vec<(String, String)>,
    /// The environment and command line as logged: full UUIDs, secrets
    /// hidden.
    pub logged: String,
    /// The command line as its row shows it.
    pub shown: String,
    pub waits: Vec<Wait>,
}

/// One launch.
#[derive(Debug)]
pub struct Launch {
    pub id: u64,
    pub record: Record,
    pub binary: PathBuf,
    pub arguments: Vec<String>,
    environment: Vec<(String, String)>,
    pub logged: String,
    pub shown: String,
    pub waits: Vec<Wait>,
    pub phase: Phase,
    /// Its process while choptop has not reaped it; `None` before it starts
    /// and for a launch an earlier choptop started.
    child: Option<Child>,
    pub queued_at: f64,
    pub started_at: Option<f64>,
    /// The end of its log, sanitized.
    pub tail: Vec<String>,
    tail_read: f64,
}

impl Launch {
    /// `launch A` until it has a pid, `proc PID` after.
    pub fn label(&self) -> String {
        match self.record.pid {
            Some(pid) => format!("proc {pid}"),
            None => format!("launch {}", self.record.letter),
        }
    }

    /// `GPUs 0,1`.
    pub fn gpus(&self) -> String {
        format!("GPUs {}", indices(&self.record.gpus))
    }

    fn reread(&mut self, now: f64) {
        self.tail = process::tail(&self.record.log);
        self.tail_read = now;
    }
}

/// A serve process choptop sent SIGTERM, watched until it exits.
#[derive(Clone, Debug, PartialEq)]
pub struct Stop {
    pub identity: Identity,
    pub since: f64,
    /// The GPUs it frees, by NVML index.
    pub gpus: Vec<u32>,
}

/// `0,1,2,3`.
pub fn indices(gpus: &[u32]) -> String {
    gpus.iter()
        .map(u32::to_string)
        .collect::<Vec<_>>()
        .join(",")
}

/// `exited 1`, `exited after signal 9`.
fn exit_text(status: Option<i32>, signal: Option<i32>) -> String {
    match (status, signal) {
        (Some(status), _) => format!("exited {status}"),
        (None, Some(signal)) => format!("exited after signal {signal}"),
        (None, None) => "exited".to_owned(),
    }
}

/// Every launch of this choptop and the serve processes it is stopping.
#[derive(Debug, Default)]
pub struct Launches {
    pub launches: Vec<Launch>,
    pub stops: Vec<Stop>,
    pub records: Records,
    /// Where logs go; `None` without a state directory.
    pub logs: Option<PathBuf>,
    /// The log directory as shown, `~` for the home directory.
    pub logs_shown: String,
    /// What the pollers and the discovery thread should do, taken by the
    /// caller after each change.
    pub followups: Vec<Followup>,
    next: u64,
}

// Production launches keep running when the dashboard exits. Tests own
// their fakes and must reap them even when an assertion unwinds.
#[cfg(test)]
impl Drop for Launches {
    fn drop(&mut self) {
        for launch in &mut self.launches {
            if let Some(child) = &mut launch.child {
                let _ = process::terminate_group(child);
                let _ = child.wait();
            }
        }
    }
}

impl Launches {
    /// Launches whose records and logs go in `state`, choptop's state
    /// directory; a message when its records cannot be read.
    pub fn new(state: Option<&Path>, logs_shown: &str) -> (Self, Option<String>) {
        let (records, message) = Records::load(state.map(|state| state.join("launches.json")));
        let launches = Self {
            records,
            logs: state.map(|state| state.join("logs")),
            logs_shown: logs_shown.to_owned(),
            launches: Vec::new(),
            stops: Vec::new(),
            followups: Vec::new(),
            next: 0,
        };
        (launches, message)
    }

    pub fn get(&self, id: u64) -> Option<&Launch> {
        self.launches.iter().find(|launch| launch.id == id)
    }

    /// The launches that hold their GPUs and port: every one that has not
    /// exited.
    pub fn reservations(&self) -> Vec<Reservation> {
        self.launches
            .iter()
            .filter(|launch| !matches!(launch.phase, Phase::Exited(..)))
            .map(|launch| Reservation {
                label: launch.label(),
                model: launch.record.served_model_name.clone(),
                gpus: launch.record.uuids.clone(),
                port: launch.record.port,
                process: launch.record.pid.zip(launch.record.start_ticks),
                launch: launch.id,
            })
            .collect()
    }

    /// Whether the serve process `pid` is being stopped.
    pub fn stopping(&self, pid: u32, start_ticks: u64) -> bool {
        self.stops
            .iter()
            .any(|stop| stop.identity.pid == pid && stop.identity.start_ticks == start_ticks)
    }

    /// Keeps `record` and writes the records, noting why they cannot be
    /// written.
    fn keep(&mut self, record: Record, registry: &mut Registry, now: f64) {
        if let Some(message) = self.records.put(record) {
            registry.note(now, "choptop", message, Tone::Warning);
        }
        if let Err(error) = self.records.save() {
            registry.note(
                now,
                "choptop",
                format!("cannot write the launch records: {error}"),
                Tone::Warning,
            );
        }
    }

    /// Takes up the launches an earlier choptop started that still run, by
    /// pid and start time, so they keep their GPUs and port; the records
    /// of the others say how they ended.
    pub fn relink(&mut self, registry: &mut Registry, now: f64) {
        let records = self.records.records.clone();
        for mut record in records.into_iter().filter(|record| !record.over()) {
            let running = record
                .pid
                .zip(record.start_ticks)
                .is_some_and(|(pid, ticks)| process::alive(pid, ticks));
            if !running {
                record.ended = Some(match record.pid {
                    Some(_) => "exited while choptop was not running".to_owned(),
                    None => "cancelled: choptop exited before it started".to_owned(),
                });
                self.keep(record, registry, now);
                continue;
            }
            let arguments = record
                .pid
                .and_then(Identity::read)
                .map(|identity| identity.arguments.join(" "))
                .unwrap_or_default();
            if record.command.is_empty() {
                record.command = crate::ui::widgets::sanitize(&arguments);
            }
            let id = self.adopt(record, Phase::Running, registry, now);
            if let Some(launch) = self.launches.iter_mut().find(|launch| launch.id == id) {
                launch.shown = crate::ui::widgets::sanitize(&arguments);
                let what = format!(
                    "relinked: launch {} of {} on {}, started by an earlier choptop",
                    launch.record.letter,
                    launch.record.profile,
                    launch.gpus()
                );
                registry.note(now, launch.label(), what, Tone::Normal);
            }
        }
    }

    /// Takes up `record` as a launch in `phase` that choptop holds no
    /// process handle of, its row with it: how a restarted choptop shows a
    /// launch an earlier one started, and how the screen tests show launch
    /// rows without starting a process.
    pub fn adopt(
        &mut self,
        record: Record,
        phase: Phase,
        registry: &mut Registry,
        now: f64,
    ) -> u64 {
        self.next += 1;
        let id = self.next;
        match record.pid.zip(record.start_ticks) {
            Some((pid, ticks)) => {
                registry.launch_process(id, pid, ticks, &record.served_model_name, now);
            }
            None => {
                registry.add_launch(id, record.letter, &record.served_model_name, now);
            }
        }
        if let Some(row) = registry.launch_row(id) {
            row.launch = Some((
                id,
                match phase {
                    Phase::Waiting => Launched::Waiting,
                    Phase::Running | Phase::Stopping(_) => Launched::Running,
                    Phase::Exited(_, status) => Launched::Exited(status),
                },
            ));
            if let Phase::Stopping(since) = phase {
                row.stopping = Some(since);
            }
        }
        let mut launch = Launch {
            id,
            shown: String::new(),
            logged: record.command.clone(),
            binary: PathBuf::new(),
            arguments: Vec::new(),
            environment: Vec::new(),
            waits: Vec::new(),
            phase,
            child: None,
            queued_at: now,
            started_at: record.pid.map(|_| now),
            tail: Vec::new(),
            tail_read: now,
            record,
        };
        launch.reread(now);
        self.launches.push(launch);
        id
    }

    /// Queues `queued`; its row waits as `launch LETTER`.
    pub fn queue(&mut self, queued: Queued, registry: &mut Registry, now: f64) -> u64 {
        self.next += 1;
        let id = self.next;
        let mut record = queued.record;
        record.command = queued.logged.clone();
        registry.add_launch(id, record.letter, &record.served_model_name, now);
        let after = queued
            .waits
            .iter()
            .map(|wait| match wait {
                Wait::Serve(identity) => format!("proc {}", identity.pid),
                Wait::Launch(other) => self
                    .get(*other)
                    .map_or_else(|| "a launch".to_owned(), Launch::label),
            })
            .collect::<Vec<_>>();
        let launch = Launch {
            id,
            binary: queued.binary,
            arguments: queued.arguments,
            environment: queued.environment,
            logged: queued.logged,
            shown: queued.shown,
            waits: queued.waits,
            phase: Phase::Waiting,
            child: None,
            queued_at: now,
            started_at: None,
            tail: Vec::new(),
            tail_read: f64::NEG_INFINITY,
            record: record.clone(),
        };
        registry.note(
            now,
            launch.label(),
            format!(
                "queued: {} on {} at {}:{}{}",
                record.profile,
                launch.gpus(),
                record.host,
                record.port,
                if after.is_empty() {
                    String::new()
                } else {
                    format!(
                        ", after {} exit{}",
                        after.join(" and "),
                        if after.len() == 1 { "s" } else { "" }
                    )
                }
            ),
            Tone::Normal,
        );
        self.launches.push(launch);
        self.keep(record, registry, now);
        id
    }

    /// Takes launch `id` out, its row with it, noting `what` for it, and
    /// records how it ended.
    fn drop_launch(
        &mut self,
        id: u64,
        what: String,
        tone: Tone,
        registry: &mut Registry,
        now: f64,
    ) {
        let Some(position) = self.launches.iter().position(|launch| launch.id == id) else {
            return;
        };
        let launch = self.launches.remove(position);
        registry.note(now, launch.label(), what, tone);
        if let Some(row) = registry.launch_row(id).map(|row| row.id) {
            self.followups.extend(registry.forget(row));
        }
        let mut record = launch.record;
        if !record.over() {
            record.ended = Some("cancelled".to_owned());
        }
        self.keep(record, registry, now);
    }

    /// Cancels the waiting launch `id` because `why`.
    pub fn cancel(&mut self, id: u64, why: &str, registry: &mut Registry, now: f64) {
        if let Some(launch) = self.launches.iter_mut().find(|launch| launch.id == id) {
            launch.record.ended = Some(format!("cancelled: {why}"));
        }
        self.drop_launch(
            id,
            format!("cancelled: {why}"),
            Tone::Warning,
            registry,
            now,
        );
    }

    /// Stops launch `id` because `why`: a waiting one is cancelled, a
    /// running one choptop started gets SIGTERM through its process group
    /// while it is not reaped, and one an earlier choptop started gets it
    /// through a pidfd once it is still the process recorded.
    pub fn stop(
        &mut self,
        id: u64,
        why: &str,
        registry: &mut Registry,
        now: f64,
    ) -> Result<(), String> {
        let result = self.stop_launch(id, why, registry, now);
        if let Err(error) = &result {
            let label = self
                .get(id)
                .map_or_else(|| "a launch".to_owned(), Launch::label);
            registry.note(now, &label, format!("not stopped: {error}"), Tone::Bad);
            let waiting = self
                .launches
                .iter()
                .filter(|launch| {
                    launch.phase == Phase::Waiting && launch.waits.contains(&Wait::Launch(id))
                })
                .map(|launch| launch.id)
                .collect::<Vec<_>>();
            for dependent in waiting {
                self.cancel(
                    dependent,
                    &format!("{label} was not stopped"),
                    registry,
                    now,
                );
            }
        }
        result
    }

    fn stop_launch(
        &mut self,
        id: u64,
        why: &str,
        registry: &mut Registry,
        now: f64,
    ) -> Result<(), String> {
        let Some(launch) = self.launches.iter_mut().find(|launch| launch.id == id) else {
            return Err("the launch is gone".to_owned());
        };
        match launch.phase {
            Phase::Waiting => {
                self.cancel(id, why, registry, now);
                return Ok(());
            }
            Phase::Stopping(_) => return Ok(()),
            Phase::Exited(..) => return Err(format!("{} has exited", launch.label())),
            Phase::Running => {}
        }
        match &mut launch.child {
            Some(child) => process::terminate_group(child)?,
            None => {
                let (Some(pid), Some(ticks)) = (launch.record.pid, launch.record.start_ticks)
                else {
                    return Err("the launch has no process".to_owned());
                };
                Identity::record(pid, ticks)?.terminate()?;
            }
        }
        launch.phase = Phase::Stopping(now);
        let label = launch.label();
        if let Some(row) = registry.launch_row(id) {
            row.stopping = Some(now);
        }
        registry.note(
            now,
            label,
            format!(
                "stopping: SIGTERM from choptop {why}; it ran {}",
                launch.logged
            ),
            Tone::Warning,
        );
        Ok(())
    }

    /// Sends the serve process `identity`, recorded when the switch was
    /// confirmed, SIGTERM through a pidfd once it is still that process,
    /// and watches it until it exits. A process that changed is not
    /// signalled, and the launches waiting for it are cancelled.
    pub fn stop_serve(
        &mut self,
        identity: &Identity,
        gpus: &[u32],
        why: &str,
        registry: &mut Registry,
        now: f64,
    ) -> Result<(), String> {
        let label = format!("proc {}", identity.pid);
        if self.stopping(identity.pid, identity.start_ticks) {
            return Ok(());
        }
        if let Err(error) = identity.terminate() {
            registry.note(
                now,
                label.clone(),
                format!("not stopped: {error}"),
                Tone::Bad,
            );
            let waiting = self
                .launches
                .iter()
                .filter(|launch| launch.waits.contains(&Wait::Serve(identity.clone())))
                .map(|launch| launch.id)
                .collect::<Vec<_>>();
            for id in waiting {
                self.cancel(id, &format!("{label} was not stopped"), registry, now);
            }
            return Err(error);
        }
        self.stops.push(Stop {
            identity: identity.clone(),
            since: now,
            gpus: gpus.to_vec(),
        });
        let source = crate::instances::Source::Process {
            pid: identity.pid,
            start_ticks: identity.start_ticks,
        };
        if let Some(row) = registry
            .instances
            .iter_mut()
            .find(|row| row.source == source)
        {
            row.stopping = Some(now);
        }
        registry.note(
            now,
            label,
            format!(
                "stopping: SIGTERM from choptop {why}; it ran {}",
                identity.arguments.join(" ")
            ),
            Tone::Warning,
        );
        Ok(())
    }

    /// Takes the exited launch `id`'s row out of the list.
    pub fn dismiss(&mut self, id: u64, registry: &mut Registry) {
        let Some(position) = self
            .launches
            .iter()
            .position(|launch| launch.id == id && matches!(launch.phase, Phase::Exited(..)))
        else {
            return;
        };
        self.launches.remove(position);
        if let Some(row) = registry.launch_row(id).map(|row| row.id) {
            self.followups.extend(registry.forget(row));
        }
    }

    /// Whether what `wait` names has exited.
    fn over(&self, wait: &Wait) -> bool {
        match wait {
            Wait::Serve(identity) => !identity.running(),
            Wait::Launch(id) => self
                .get(*id)
                .is_none_or(|launch| matches!(launch.phase, Phase::Exited(..))),
        }
    }

    /// One turn of the loop: starts each waiting launch whose waits are
    /// over once `probe` finds its port free, reaps launches that exited,
    /// and notes stopped serve processes that are gone. Asks for a scan
    /// when anything ended or started; says whether anything did.
    pub fn supervise(
        &mut self,
        registry: &mut Registry,
        now: f64,
        probe: &mut dyn FnMut(&str, u16) -> bool,
    ) -> bool {
        let mut changed = false;
        // Stopped serve processes that exited.
        let gone = self
            .stops
            .iter()
            .filter(|stop| !stop.identity.running())
            .cloned()
            .collect::<Vec<_>>();
        for stop in gone {
            self.stops.retain(|seen| *seen != stop);
            let label = format!("proc {}", stop.identity.pid);
            registry.note(
                now,
                label,
                format!(
                    "exited after draining {}; GPUs {} are free",
                    crate::ui::widgets::age(now - stop.since),
                    indices(&stop.gpus)
                ),
                Tone::Normal,
            );
            let source = crate::instances::Source::Process {
                pid: stop.identity.pid,
                start_ticks: stop.identity.start_ticks,
            };
            let row = registry
                .instances
                .iter()
                .find(|row| row.source == source && row.launch.is_none())
                .map(|row| row.id);
            if let Some(row) = row {
                self.followups.extend(registry.forget(row));
            }
            changed = true;
        }
        let ids = self
            .launches
            .iter()
            .map(|launch| launch.id)
            .collect::<Vec<_>>();
        for id in ids {
            let Some(index) = self.launches.iter().position(|launch| launch.id == id) else {
                continue;
            };
            match self.launches[index].phase {
                Phase::Waiting => {
                    let waits = self.launches[index].waits.clone();
                    if waits.iter().all(|wait| self.over(wait)) {
                        self.begin(id, registry, now, probe);
                        changed = true;
                    }
                }
                Phase::Running | Phase::Stopping(_) => {
                    if let Some(ended) = self.reap(index) {
                        self.ended(id, ended, registry, now);
                        changed = true;
                    } else if now - self.launches[index].tail_read >= TAIL_SECONDS {
                        self.launches[index].reread(now);
                    }
                }
                Phase::Exited(..) => {}
            }
        }
        if changed {
            self.followups.push(Followup::Request);
        }
        changed
    }

    /// The exit status and signal of the launch at `index` once its process
    /// has exited.
    fn reap(&mut self, index: usize) -> Option<(Option<i32>, Option<i32>)> {
        let launch = &mut self.launches[index];
        match &mut launch.child {
            Some(child) => match child.try_wait() {
                Ok(Some(status)) => Some(status_of(status)),
                Ok(None) => None,
                Err(_) => Some((None, None)),
            },
            None => {
                let (pid, ticks) = launch.record.pid.zip(launch.record.start_ticks)?;
                (!process::alive(pid, ticks)).then_some((None, None))
            }
        }
    }

    /// Launch `id` exited with `status` or `signal`: one choptop stopped
    /// leaves the list, one that exited on its own stays as an exited row
    /// that tells how it ended.
    fn ended(
        &mut self,
        id: u64,
        (status, signal): (Option<i32>, Option<i32>),
        registry: &mut Registry,
        now: f64,
    ) {
        let Some(launch) = self.launches.iter_mut().find(|launch| launch.id == id) else {
            return;
        };
        launch.child = None;
        launch.reread(now);
        launch.record.exit_status = status;
        if status.is_none() {
            launch.record.ended = Some(match signal {
                Some(signal) => format!("ended by signal {signal}"),
                None => "exited".to_owned(),
            });
        }
        let record = launch.record.clone();
        if let Phase::Stopping(since) = launch.phase {
            let what = format!(
                "exited after draining {}; {} are free",
                crate::ui::widgets::age(now - since),
                launch.gpus()
            );
            self.drop_launch(id, what, Tone::Normal, registry, now);
            return;
        }
        launch.phase = Phase::Exited(now, status);
        let label = launch.label();
        let last = launch.tail.last().cloned();
        let answered = registry.launch_row(id).is_some_and(|row| row.answered);
        if let Some(row) = registry.launch_row(id) {
            row.launch = Some((id, Launched::Exited(status)));
            row.reachable = false;
        }
        let exited = exit_text(status, signal);
        let what = match (answered, last) {
            (false, Some(last)) => format!("{exited} during startup: {last}"),
            (false, None) => format!("{exited} during startup"),
            (true, _) => exited,
        };
        registry.note(now, label, what, Tone::Bad);
        self.keep(record, registry, now);
    }

    /// Starts the waiting launch `id` once its port is free.
    fn begin(
        &mut self,
        id: u64,
        registry: &mut Registry,
        now: f64,
        probe: &mut dyn FnMut(&str, u16) -> bool,
    ) {
        let Some(index) = self.launches.iter().position(|launch| launch.id == id) else {
            return;
        };
        let (host, port) = {
            let record = &self.launches[index].record;
            (record.host.clone(), record.port)
        };
        if !probe(&host, port) {
            let why = format!("port {port} is in use");
            self.launches[index].record.ended = Some(why.clone());
            self.drop_launch(id, format!("not started: {why}"), Tone::Bad, registry, now);
            return;
        }
        if let Err(error) = self.spawn(index, now) {
            self.launches[index].record.ended = Some(format!("not started: {error}"));
            self.drop_launch(
                id,
                format!("not started: {error}"),
                Tone::Bad,
                registry,
                now,
            );
            return;
        }
        let launch = &mut self.launches[index];
        let (Some(pid), Some(ticks)) = (launch.record.pid, launch.record.start_ticks) else {
            return;
        };
        registry.launch_process(id, pid, ticks, &launch.record.served_model_name, now);
        let file = launch
            .record
            .log
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_default();
        let what = format!(
            "launched by choptop: {} on {}; log {}/{file}",
            launch.record.profile,
            launch.gpus(),
            self.logs_shown
        );
        let label = launch.label();
        let record = launch.record.clone();
        registry.note(now, label, what, Tone::Normal);
        self.keep(record, registry, now);
    }

    /// Starts the process of the launch at `index`.
    fn spawn(&mut self, index: usize, now: f64) -> Result<(), String> {
        let directory = self
            .logs
            .clone()
            .ok_or("no directory for logs: neither XDG_STATE_HOME nor HOME is set")?;
        process::private_directory(&directory)?;
        let launch = &mut self.launches[index];
        let stem = launch
            .record
            .log
            .file_stem()
            .map(|stem| stem.to_string_lossy().into_owned())
            .unwrap_or_else(|| format!("launch-{}", launch.record.letter));
        let log = process::log_path(&directory, &stem);
        let header = format!(
            "# choptop launch {} of {} {}, switch {}\n# GPUs {}: {}\n# {}\n",
            launch.record.letter,
            launch.record.model,
            launch.record.profile,
            launch.record.switch,
            indices(&launch.record.gpus),
            launch.record.uuids.join(","),
            launch.logged,
        );
        let child = process::start(
            &launch.binary,
            &launch.arguments,
            &launch.environment,
            &log,
            &header,
        )?;
        let pid = child.id();
        launch.record.pid = Some(pid);
        launch.record.start_ticks = process::start_ticks_of(pid);
        launch.record.log = log;
        launch.child = Some(child);
        launch.phase = Phase::Running;
        launch.started_at = Some(now);
        launch.reread(now);
        Ok(())
    }
}

/// The exit status, or the signal that ended the process.
fn status_of(status: ExitStatus) -> (Option<i32>, Option<i32>) {
    (status.code(), status.signal())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::instances::{Source, State};
    use crate::launch::process::tests::{
        CLOSE, Guard, UNTIL_TERM, fake_serve, scratch, wait_for, write_script,
    };
    use crate::launch::records::tests::record;
    use std::os::unix::fs::MetadataExt;

    const LOGS: &str = "~/.local/state/choptop/logs";

    /// Instance `letter` of a switch, run by `binary` on `port`.
    fn queued(binary: &Path, letter: char, port: u16, waits: Vec<Wait>) -> Queued {
        let mut record = record("20261102-153012", letter, &[u32::from(letter as u8 - b'A')]);
        record.port = port;
        record.log = PathBuf::from(format!(
            "qwen38-tp2-mtp-{}-20261102-153012.log",
            letter.to_ascii_lowercase()
        ));
        Queued {
            record,
            binary: binary.to_path_buf(),
            arguments: vec![
                "serve".into(),
                "--listen".into(),
                format!("127.0.0.1:{port}"),
            ],
            environment: vec![("CHOPINFER_LOG_LEVEL".into(), "info".into())],
            logged: format!("CHOPINFER_LOG_LEVEL=info {} serve", binary.display()),
            shown: "chopinfer serve".into(),
            waits,
        }
    }

    fn launches(root: &Path) -> Launches {
        let (launches, message) = Launches::new(Some(&root.join("state")), LOGS);
        assert_eq!(message, None);
        launches
    }

    /// Supervises until `done`, every probe finding its port free.
    fn until(
        launches: &mut Launches,
        registry: &mut Registry,
        mut done: impl FnMut(&Launches, &Registry) -> bool,
    ) {
        wait_for(|| {
            launches.supervise(registry, 5.0, &mut |_, _| true);
            done(launches, registry)
        });
    }

    fn last_event(registry: &Registry) -> (String, String, Tone) {
        let event = registry.events.back().unwrap();
        (event.who.clone(), event.what.clone(), event.tone)
    }

    /// Stops what is still running at the end of a test.
    fn stop_all(launches: &mut Launches, registry: &mut Registry) {
        let ids = launches
            .launches
            .iter()
            .map(|launch| launch.id)
            .collect::<Vec<_>>();
        for id in ids {
            let _ = launches.stop(id, "at the end of the test", registry, 9.0);
        }
        until(launches, registry, |launches, _| {
            launches
                .launches
                .iter()
                .all(|launch| matches!(launch.phase, Phase::Exited(..)))
        });
    }

    #[test]
    fn a_panicking_launch_owner_reaps_its_children() {
        let root = scratch("panic-owner");
        let binary = root.join("sleeps");
        write_script(&binary, &format!("#!/bin/sh\n{CLOSE}{UNTIL_TERM}"));
        let mut spawned = None;
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut launches = launches(&root);
            let mut registry = Registry::new(1.0);
            let id = launches.queue(queued(&binary, 'A', 8310, Vec::new()), &mut registry, 1.0);
            launches.supervise(&mut registry, 2.0, &mut |_, _| true);
            let pid = launches.get(id).unwrap().record.pid.unwrap();
            spawned = Identity::read(pid);
            panic!("fixture assertion");
        }));
        assert!(result.is_err());
        let spawned = spawned.unwrap();
        let survived = spawned.running();
        // Clean up even when this regression is run with Drop disabled.
        if survived {
            spawned.terminate().unwrap();
            wait_for(|| !spawned.running());
            let _ = rustix::process::waitpid(
                rustix::process::Pid::from_raw(spawned.pid as i32),
                rustix::process::WaitOptions::empty(),
            );
        }
        assert!(!survived, "the panic left a test launch running");
        assert!(
            process::start_ticks_of(spawned.pid).is_none(),
            "child was not reaped"
        );
    }

    #[test]
    fn a_launch_that_exits_during_startup_keeps_its_status_and_log_tail() {
        let root = scratch("exits");
        let binary = root.join("fails");
        write_script(
            &binary,
            &format!(
                "#!/bin/sh\n{CLOSE}echo 'loading weights'\necho 'error: CUDA out of memory on GPU 0'\nexit 1\n"
            ),
        );
        let mut launches = launches(&root);
        let mut registry = Registry::new(1.0);
        let id = launches.queue(queued(&binary, 'A', 8310, Vec::new()), &mut registry, 1.0);
        let row = registry.launch_row(id).unwrap();
        assert_eq!(row.source, Source::Launch { letter: 'A' });
        assert_eq!(row.state(1.0), State::Waiting);
        assert_eq!(
            last_event(&registry),
            (
                "launch A".to_owned(),
                "queued: tp2-mtp on GPUs 0 at 127.0.0.1:8310".to_owned(),
                Tone::Normal
            )
        );
        launches.supervise(&mut registry, 2.0, &mut |_, _| true);
        let pid = launches.get(id).unwrap().record.pid.unwrap();
        let log = launches.get(id).unwrap().record.log.clone();
        assert!(matches!(
            registry.launch_row(id).unwrap().source,
            Source::Process { pid: seen, .. } if seen == pid
        ));
        let launched = registry
            .events
            .iter()
            .find(|event| event.what.starts_with("launched"))
            .unwrap();
        assert_eq!(launched.who, format!("proc {pid}"));
        assert_eq!(
            launched.what,
            format!(
                "launched by choptop: tp2-mtp on GPUs 0; log {LOGS}/qwen38-tp2-mtp-a-20261102-153012.log"
            )
        );
        until(&mut launches, &mut registry, |launches, _| {
            matches!(launches.get(id).unwrap().phase, Phase::Exited(..))
        });
        let launch = launches.get(id).unwrap();
        assert_eq!(launch.record.exit_status, Some(1));
        assert_eq!(
            launch.tail[0],
            "# choptop launch A of qwen38 tp2-mtp, switch 20261102-153012"
        );
        assert_eq!(
            &launch.tail[launch.tail.len() - 2..],
            ["loading weights", "error: CUDA out of memory on GPU 0"]
        );
        assert_eq!(
            registry.launch_row(id).unwrap().state(5.0),
            State::Exited(Some(1))
        );
        assert_eq!(
            last_event(&registry),
            (
                format!("proc {pid}"),
                "exited 1 during startup: error: CUDA out of memory on GPU 0".to_owned(),
                Tone::Bad
            )
        );
        // The record on disk, and the log, readable by this user only.
        let (again, _) = Records::load(Some(root.join("state/launches.json")));
        assert_eq!(again.records.len(), 1);
        assert_eq!(
            (again.records[0].pid, again.records[0].exit_status),
            (Some(pid), Some(1))
        );
        assert_eq!(again.records[0].log, log);
        assert_eq!(fs_mode(&log), 0o600);
        assert_eq!(fs_mode(log.parent().unwrap()), 0o700);
        // An exited launch's row stays until it is dismissed.
        assert!(launches.reservations().is_empty());
        launches.dismiss(id, &mut registry);
        assert!(registry.launch_row(id).is_none() && launches.get(id).is_none());
    }

    fn fs_mode(path: &Path) -> u32 {
        std::fs::metadata(path).unwrap().mode() & 0o777
    }

    fn identity(guard: &Guard) -> Identity {
        let ticks = process::start_ticks_of(guard.0.id()).unwrap();
        Identity::record(guard.0.id(), ticks).unwrap()
    }

    #[test]
    fn a_launch_waits_for_its_gpu_owner_and_its_port_holder_then_probes_its_port() {
        let root = scratch("waits");
        let owner = fake_serve(&root);
        let holder = fake_serve(&root);
        let (owner_identity, holder_identity) = (identity(&owner), identity(&holder));
        let binary = root.join("sleeps");
        write_script(
            &binary,
            &format!("#!/bin/sh\n{CLOSE}echo \"started $*\"\n{UNTIL_TERM}"),
        );
        let mut launches = launches(&root);
        let mut registry = Registry::new(1.0);
        let waits = vec![
            Wait::Serve(owner_identity.clone()),
            Wait::Serve(holder_identity.clone()),
        ];
        let id = launches.queue(queued(&binary, 'A', 8310, waits), &mut registry, 1.0);
        assert_eq!(
            last_event(&registry).1,
            format!(
                "queued: tp2-mtp on GPUs 0 at 127.0.0.1:8310, after proc {} and proc {} exit",
                owner_identity.pid, holder_identity.pid
            )
        );
        let probed = std::cell::Cell::new(0);
        let mut probe = |_: &str, _: u16| {
            probed.set(probed.get() + 1);
            true
        };
        launches.supervise(&mut registry, 2.0, &mut probe);
        assert_eq!(launches.get(id).unwrap().phase, Phase::Waiting);
        // The GPU's owner exits; the port's holder still runs.
        launches
            .stop_serve(
                &owner_identity,
                &[0],
                "to start Qwen3.8-Flash tp2-mtp on GPUs 0",
                &mut registry,
                2.0,
            )
            .unwrap();
        let (who, what, tone) = last_event(&registry);
        assert_eq!(
            (who, tone),
            (format!("proc {}", owner_identity.pid), Tone::Warning)
        );
        assert!(what.starts_with(
            "stopping: SIGTERM from choptop to start Qwen3.8-Flash tp2-mtp on GPUs 0; it ran "
        ));
        assert!(what.ends_with("chopinfer serve"), "{what}");
        wait_for(|| {
            launches.supervise(&mut registry, 3.0, &mut probe);
            launches.stops.is_empty()
        });
        assert_eq!(
            last_event(&registry).1,
            "exited after draining 1s; GPUs 0 are free"
        );
        assert_eq!(launches.get(id).unwrap().phase, Phase::Waiting);
        assert_eq!(probed.get(), 0);
        assert!(holder_identity.running());
        // Once the port's holder exits too, the port is probed and the
        // launch starts.
        launches
            .stop_serve(
                &holder_identity,
                &[],
                "to start Qwen3.8-Flash tp2-mtp on GPUs 0",
                &mut registry,
                3.0,
            )
            .unwrap();
        wait_for(|| {
            launches.supervise(&mut registry, 4.0, &mut probe);
            launches.get(id).unwrap().phase == Phase::Running
        });
        assert_eq!(probed.get(), 1);
        // A launch whose port is taken at its final probe does not start.
        let taken = launches.queue(queued(&binary, 'B', 8311, Vec::new()), &mut registry, 4.0);
        launches.supervise(&mut registry, 4.5, &mut |_, port| port != 8311);
        assert!(launches.get(taken).is_none() && registry.launch_row(taken).is_none());
        assert_eq!(
            last_event(&registry),
            (
                "launch B".to_owned(),
                "not started: port 8311 is in use".to_owned(),
                Tone::Bad
            )
        );
        let (again, _) = Records::load(Some(root.join("state/launches.json")));
        let record = again
            .records
            .iter()
            .find(|record| record.letter == 'B')
            .unwrap();
        assert_eq!(record.ended.as_deref(), Some("port 8311 is in use"));
        stop_all(&mut launches, &mut registry);
    }

    #[test]
    fn stopping_a_launch_ends_its_group_and_its_row() {
        let root = scratch("stop");
        let binary = root.join("sleeps");
        write_script(
            &binary,
            &format!(
                "#!/bin/sh\n{CLOSE}sleep 30 &\necho $! > \"$(dirname \"$0\")/grandchild\"\n{UNTIL_TERM}"
            ),
        );
        let mut launches = launches(&root);
        let mut registry = Registry::new(1.0);
        let id = launches.queue(queued(&binary, 'A', 8310, Vec::new()), &mut registry, 1.0);
        launches.supervise(&mut registry, 2.0, &mut |_, _| true);
        wait_for(|| root.join("grandchild").exists());
        let grandchild = std::fs::read_to_string(root.join("grandchild"))
            .unwrap()
            .trim()
            .parse::<u32>()
            .unwrap();
        let grandchild_ticks = process::start_ticks_of(grandchild).unwrap();
        let pid = launches.get(id).unwrap().record.pid.unwrap();
        launches
            .stop(id, "as asked with X", &mut registry, 3.0)
            .unwrap();
        assert_eq!(registry.launch_row(id).unwrap().state(3.0), State::Stopping);
        assert_eq!(
            last_event(&registry),
            (
                format!("proc {pid}"),
                format!(
                    "stopping: SIGTERM from choptop as asked with X; it ran CHOPINFER_LOG_LEVEL=info {} serve",
                    binary.display()
                ),
                Tone::Warning
            )
        );
        until(&mut launches, &mut registry, |launches, _| {
            launches.get(id).is_none()
        });
        assert!(registry.launch_row(id).is_none());
        assert_eq!(
            last_event(&registry).1,
            "exited after draining 2s; GPUs 0 are free"
        );
        wait_for(|| !process::alive(grandchild, grandchild_ticks));
        let (again, _) = Records::load(Some(root.join("state/launches.json")));
        assert_eq!(again.records[0].exit_status, Some(0));
    }

    #[test]
    fn a_serve_process_that_changed_is_not_signalled_and_its_launch_is_cancelled() {
        let root = scratch("changed");
        let serve = fake_serve(&root);
        let recorded = identity(&serve);
        // The pid now names a process that started at another time.
        let changed = Identity {
            start_ticks: recorded.start_ticks + 1,
            ..recorded.clone()
        };
        let binary = root.join("never");
        write_script(&binary, &format!("#!/bin/sh\n{CLOSE}exit 0\n"));
        let mut launches = launches(&root);
        let mut registry = Registry::new(1.0);
        let id = launches.queue(
            queued(&binary, 'A', 8310, vec![Wait::Serve(changed.clone())]),
            &mut registry,
            1.0,
        );
        let error = launches
            .stop_serve(
                &changed,
                &[0],
                "to start Qwen3.8-Flash tp2-mtp on GPUs 0",
                &mut registry,
                1.0,
            )
            .unwrap_err();
        assert!(error.contains("start time changed"), "{error}");
        assert!(launches.get(id).is_none() && registry.launch_row(id).is_none());
        let events = registry
            .events
            .iter()
            .rev()
            .take(2)
            .map(|event| (event.who.clone(), event.what.clone()))
            .collect::<Vec<_>>();
        assert_eq!(
            events,
            [
                (
                    "launch A".to_owned(),
                    format!("cancelled: proc {} was not stopped", recorded.pid)
                ),
                (
                    format!("proc {}", recorded.pid),
                    format!("not stopped: {error}")
                ),
            ]
        );
        // The process runs on, never signalled, and the launch never ran.
        std::thread::sleep(std::time::Duration::from_millis(200));
        assert!(recorded.running());
        let (again, _) = Records::load(Some(root.join("state/launches.json")));
        assert_eq!(again.records[0].pid, None);
        assert_eq!(
            again.records[0].ended,
            Some(format!("cancelled: proc {} was not stopped", recorded.pid))
        );
    }

    #[test]
    fn own_and_adopted_stop_events_keep_the_redacted_command() {
        let root = scratch("stop-command");
        let binary = root.join("sleeps");
        write_script(&binary, &format!("#!/bin/sh\n{CLOSE}{UNTIL_TERM}"));
        let command = "CHOPINFER_API_KEY=**** chopinfer serve --model /models/example";
        let mut registry = Registry::new(1.0);
        let mut launches = launches(&root);
        let mut item = queued(&binary, 'A', 8310, Vec::new());
        item.environment
            .push(("CHOPINFER_API_KEY".into(), "placeholder-secret".into()));
        item.logged = command.into();
        let id = launches.queue(item, &mut registry, 1.0);
        launches.supervise(&mut registry, 2.0, &mut |_, _| true);
        launches
            .stop(id, "as asked with X", &mut registry, 3.0)
            .unwrap();
        assert!(last_event(&registry).1.ends_with(command));
        let saved = Records::load(Some(root.join("state/launches.json"))).0;
        assert_eq!(saved.records[0].command, command);
        let guard = fake_serve(&root.join("adopted"));
        let mut record = saved.records[0].clone();
        record.pid = Some(guard.0.id());
        record.start_ticks = process::start_ticks_of(guard.0.id());
        let adopted = launches.adopt(record, Phase::Running, &mut registry, 4.0);
        launches
            .stop(adopted, "as asked with X", &mut registry, 5.0)
            .unwrap();
        assert!(last_event(&registry).1.ends_with(command));
        assert!(
            registry
                .events
                .iter()
                .all(|event| !event.what.contains("placeholder-secret"))
        );
    }

    #[test]
    fn a_restarted_choptop_relinks_its_running_launches_and_closes_the_others() {
        let root = scratch("relink");
        let serve = fake_serve(&root);
        let running = identity(&serve);
        let mut records = Records::load(Some(root.join("state/launches.json"))).0;
        let mut live = record("20261102-153012", 'A', &[0, 1]);
        (live.pid, live.start_ticks) = (Some(running.pid), Some(running.start_ticks));
        let mut gone = record("20261102-153012", 'B', &[2, 3]);
        (gone.pid, gone.start_ticks) = (Some(running.pid), Some(running.start_ticks + 1));
        let queued = record("20261102-153012", 'C', &[4]);
        for record in [live, gone, queued] {
            records.put(record);
        }
        records.save().unwrap();
        let mut launches = launches(&root);
        let mut registry = Registry::new(1.0);
        launches.relink(&mut registry, 1.0);
        assert_eq!(launches.launches.len(), 1);
        let reservations = launches.reservations();
        assert_eq!(reservations.len(), 1);
        assert_eq!(
            reservations[0].process,
            Some((running.pid, running.start_ticks))
        );
        assert_eq!(reservations[0].label, format!("proc {}", running.pid));
        let ended = launches
            .records
            .records
            .iter()
            .map(|record| (record.letter, record.ended.clone()))
            .collect::<Vec<_>>();
        assert_eq!(
            ended,
            [
                ('A', None),
                ('B', Some("exited while choptop was not running".to_owned())),
                (
                    'C',
                    Some("cancelled: choptop exited before it started".to_owned())
                ),
            ]
        );
        // Its exit is seen by start time, since no Child is held.
        let id = launches.launches[0].id;
        drop(serve);
        until(&mut launches, &mut registry, |launches, _| {
            matches!(launches.get(id).unwrap().phase, Phase::Exited(..))
        });
        assert_eq!(
            registry.launch_row(id).unwrap().state(5.0),
            State::Exited(None)
        );
    }
}
