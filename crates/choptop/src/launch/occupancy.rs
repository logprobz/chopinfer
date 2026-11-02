//! Who holds each GPU and port, from the latest readings: a serve process
//! of this user that a switch may stop, a pending choptop launch, something
//! choptop leaves alone, or nothing. A GPU whose holders cannot be told is
//! unknown and cannot be picked; a failed or missing process list is never
//! read as an empty GPU.
//!
//! Any compute process that is not a serve row blocks its GPU, even beside
//! a serve process a switch would stop. A graphics process, such as a
//! desktop's display server, does not: the GPU stays free or held, and the
//! process is carried along for the loader to show and warn about.
//!
//! The readings are the latest the GPU poller and the scanner delivered,
//! kept in [`Live`] as they arrive: pausing holds the list's frame, not
//! these.

use std::collections::BTreeMap;

use crate::discover::{self, Found, listen_address};
use crate::gpu::poll::Report;
use crate::gpu::{Device, Kind, Process, is_uuid};
use crate::instances::{Instance, Source};
use crate::ui::widgets::{bytes, sanitize};

use super::Reservation;

/// Whether a queued launch may proceed under the latest GPU reading.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Gate {
    Clear,
    Hold,
    Cancel(String),
}

/// A reservation never overrides a newer unknown or blocked reading.
/// Expected owners hold the launch until the reading also shows them gone;
/// `gpus` must count the processes it waits for as what they stand for
/// (see [`Waited`]).
pub fn gate(launch: &super::Launch, gpus: &[Gpu]) -> Gate {
    if launch.record.gpus.len() != launch.record.uuids.len() {
        return Gate::Cancel("the saved GPU UUIDs are incomplete".into());
    }
    let mut hold = false;
    for (index, uuid) in launch.record.gpus.iter().zip(&launch.record.uuids) {
        let Some(gpu) = gpus.iter().find(|gpu| gpu.index == *index) else {
            return Gate::Cancel(format!("GPU {index} is gone"));
        };
        if gpu.uuid.is_none() && matches!(gpu.occupancy, Occupancy::Unknown(_)) {
            hold = true;
            continue;
        }
        if gpu.uuid.as_ref() != Some(uuid) {
            return Gate::Cancel(format!("GPU {index}'s UUID changed"));
        }
        match &gpu.occupancy {
            Occupancy::Free => {}
            Occupancy::Unknown(_) => hold = true,
            Occupancy::Blocked(why) => {
                return Gate::Cancel(format!("GPU {index} is held by {why}"));
            }
            Occupancy::Held(holders) => {
                for holder in holders {
                    if *holder == Holder::Launch(launch.id) {
                        continue;
                    }
                    let expected = launch.waits.iter().any(|wait| match (wait, holder) {
                        (super::Wait::Launch(id), Holder::Launch(other)) => id == other,
                        (super::Wait::Serve(identity), Holder::Serve(serve)) => {
                            identity.pid == serve.pid && identity.start_ticks == serve.start_ticks
                        }
                        _ => false,
                    });
                    if !expected {
                        return Gate::Cancel(format!("GPU {index} has a new owner"));
                    }
                    hold = true;
                }
            }
        }
    }
    if hold { Gate::Hold } else { Gate::Clear }
}

/// What is known of one GPU's processes.
#[derive(Clone, Debug, Default, PartialEq)]
struct Listing {
    /// It said it is lost and has not answered since.
    lost: bool,
    /// The latest process query failed: the list before it says nothing
    /// now.
    failed: bool,
    /// The latest process list read, and when.
    listed: Option<(f64, Vec<Process>)>,
}

impl Listing {
    /// The processes, while the latest query read them.
    fn current(&self) -> Option<&(f64, Vec<Process>)> {
        self.listed.as_ref().filter(|_| !self.failed)
    }
}

/// The latest GPU readings and discovery scan, taken as they arrive, even
/// while the screen is paused.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Live {
    pub devices: Vec<Device>,
    listings: Vec<Listing>,
    /// Why the GPUs cannot be read; `None` while they are or before NVML
    /// says.
    pub unavailable: Option<String>,
    /// Every instance the latest scan found.
    pub found: Vec<Found>,
    /// When the latest scan began, and when the latest GPU reading was
    /// taken; a start waits for both to be newer than its request.
    pub scanned: Option<f64>,
    pub polled: Option<f64>,
    /// Whether discovery runs; without it no scan is waited for.
    pub scanning: bool,
}

impl Live {
    /// Takes a GPU poller report.
    pub fn gpu(&mut self, report: &Report) {
        match report {
            Report::Opened { devices, .. } => {
                self.devices = devices.clone();
                self.listings = vec![Listing::default(); devices.len()];
                self.unavailable = None;
            }
            Report::Unavailable(reason) => {
                self.devices.clear();
                self.listings.clear();
                self.unavailable = Some(reason.clone());
            }
            Report::Reopened { index, device } => {
                if let Some(slot) = self.devices.get_mut(*index) {
                    *slot = device.clone();
                }
            }
            Report::Polled(readings) => {
                self.polled = readings
                    .iter()
                    .map(|reading| reading.at)
                    .reduce(f64::min)
                    .or(self.polled);
                for (listing, reading) in self.listings.iter_mut().zip(readings) {
                    // As the interface reads it: lost until a query answers.
                    if reading.lost {
                        listing.lost = true;
                    } else if reading.answered {
                        listing.lost = false;
                    }
                    // A failed query leaves the GPU unknown until the next
                    // one reads it.
                    match &reading.processes {
                        Some(processes) => {
                            listing.listed = Some((reading.at, processes.clone()));
                            listing.failed = false;
                        }
                        None => listing.failed = true,
                    }
                }
            }
        }
    }

    /// Whether a scan began and the GPUs were read at or after `since`;
    /// GPUs that cannot be read, and scans while discovery is off, are not
    /// waited for.
    pub fn fresh(&self, since: f64) -> bool {
        let read = self.unavailable.is_some() || self.polled.is_some_and(|at| at >= since);
        let scanned = !self.scanning || self.scanned.is_some_and(|at| at >= since);
        read && scanned
    }

    /// Takes a discovery report.
    pub fn scan(&mut self, report: &discover::Report) {
        match report {
            discover::Report::Scan { found, started, .. } => {
                self.found = found.clone();
                self.scanned = Some(*started);
            }
            discover::Report::Listening {
                pid,
                start_ticks,
                addresses,
            } => {
                for found in &mut self.found {
                    if let Found::Process(process) = found
                        && process.pid == *pid
                        && process.start_ticks == *start_ticks
                    {
                        process.listening = addresses.clone();
                    }
                }
            }
        }
    }
}

/// A serve process of this user.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Serve {
    pub pid: u32,
    pub start_ticks: u64,
    /// What it serves, as its row or command line says.
    pub name: Option<String>,
    /// The ports it listens on, or the one its `--listen` names while it
    /// has no listener.
    pub ports: Vec<u16>,
    /// It holds no GPU yet: it is loading.
    pub starting: bool,
}

impl Serve {
    /// `proc 48211`.
    pub fn label(&self) -> String {
        format!("proc {}", self.pid)
    }

    /// `proc 48211 MiMo-V2.6-Flash`.
    pub fn described(&self) -> String {
        match &self.name {
            Some(name) => format!("proc {} {name}", self.pid),
            None => self.label(),
        }
    }
}

/// What holds a GPU that a switch may stop or replace.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Holder {
    /// A serve process of this user, stopped before the launch starts.
    Serve(Serve),
    /// A pending choptop launch, by its launch id.
    Launch(u64),
}

/// A process a queued launch waits for, by pid and start ticks, and the
/// holder it stands for. A GPU reading taken before it exited may still
/// show it after the scan has dropped it or its launch has left the list;
/// it is still that holder, not a new owner.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Waited {
    pub pid: u32,
    pub start_ticks: u64,
    pub holder: Holder,
}

/// Who holds one GPU.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Occupancy {
    Free,
    /// Rows a switch may stop or replace.
    Held(Vec<Holder>),
    /// Something a switch does not stop, and who, as it reads after "held
    /// by": a container, a remote target, a process that is not serve, a
    /// serve process that is starting there.
    Blocked(String),
    /// Not known, and why; it cannot be picked.
    Unknown(String),
}

/// One GPU as the loader shows it.
#[derive(Clone, Debug, PartialEq)]
pub struct Gpu {
    /// NVML's index, the one shown; not a CUDA ordinal.
    pub index: u32,
    pub uuid: Option<String>,
    /// Bytes.
    pub memory_total: Option<u64>,
    pub occupancy: Occupancy,
    /// Graphics processes on it, which keep running beside a launch.
    pub graphics: Vec<Process>,
}

/// Who holds a port.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PortHolder {
    Serve(Serve),
    /// A pending choptop launch, by its launch id.
    Launch(u64),
    /// Something choptop does not stop, and what.
    Other(String),
}

/// What occupancy is read from.
pub struct Inputs<'a> {
    pub live: &'a Live,
    /// The registry's rows, which name containers and targets.
    pub instances: &'a [Instance],
    pub reservations: &'a [Reservation],
    /// The processes queued launches wait for; empty but for deciding
    /// whether those launches go ahead.
    pub waited: &'a [Waited],
    /// Seconds between GPU polls: a process list older than two is stale.
    pub interval: f64,
    pub now: f64,
}

impl Inputs<'_> {
    /// The serve processes of this user the latest scan found, with the
    /// name each one's row or command line gives.
    fn serves(&self, holding: &[u32]) -> Vec<Serve> {
        self.live
            .found
            .iter()
            .filter_map(|found| match found {
                Found::Process(process) => Some(process),
                Found::Container(_) => None,
            })
            .map(|process| {
                let row = self.instances.iter().find(|instance| {
                    instance.source
                        == Source::Process {
                            pid: process.pid,
                            start_ticks: process.start_ticks,
                        }
                });
                let mut ports = process
                    .listening
                    .iter()
                    .map(|address| address.port())
                    .collect::<Vec<_>>();
                if ports.is_empty()
                    && let Ok(address) =
                        listen_address(&process.arguments).parse::<std::net::SocketAddr>()
                {
                    ports.push(address.port());
                }
                ports.dedup();
                Serve {
                    pid: process.pid,
                    start_ticks: process.start_ticks,
                    name: row
                        .and_then(Instance::name)
                        .map(str::to_owned)
                        .or_else(|| process.model_hint.clone())
                        .map(|name| sanitize(&name)),
                    ports,
                    starting: !holding.contains(&process.pid),
                }
            })
            .collect()
    }

    /// The launch id of the reservation whose started process is `pid`.
    fn launch_of(&self, pid: u32, start_ticks: Option<u64>) -> Option<u64> {
        self.reservations
            .iter()
            .find(|reservation| {
                reservation.process.is_some_and(|(seen, ticks)| {
                    seen == pid && start_ticks.is_none_or(|start| start == ticks)
                })
            })
            .map(|reservation| reservation.launch)
    }

    /// The waited process `pid`; a GPU reading that could not read its
    /// start ticks matches by pid, as [`Inputs::launch_of`] does.
    fn waited(&self, pid: u32, start_ticks: Option<u64>) -> Option<&Waited> {
        self.waited.iter().find(|waited| {
            waited.pid == pid && start_ticks.is_none_or(|start| start == waited.start_ticks)
        })
    }

    /// The container row named `id`, as `docker glm-tp4`.
    fn container(&self, id: &str) -> String {
        self.instances
            .iter()
            .find_map(|instance| match &instance.source {
                Source::Container { id: seen, name, .. } if seen == id => {
                    Some(format!("docker {}", sanitize(name)))
                }
                _ => None,
            })
            .unwrap_or_else(|| format!("container {}", &id[..id.len().min(12)]))
    }
}

/// Why GPU `index` of `live` cannot be read now; `None` when its process
/// list is fresh.
fn unknown(live: &Live, index: usize, interval: f64, now: f64) -> Option<String> {
    let device = &live.devices[index];
    if let Some(reason) = &device.unreadable {
        return Some(format!("NVML cannot open it: {reason}"));
    }
    if device.uuid.is_none() {
        return Some("its UUID cannot be read".to_owned());
    }
    let listing = live.listings.get(index)?;
    if listing.lost {
        return Some("lost or off the bus".to_owned());
    }
    if listing.failed {
        return Some("its process list could not be read".to_owned());
    }
    match &listing.listed {
        None => Some("its processes have not been read".to_owned()),
        Some((at, _)) if now - at > 2.0 * interval => Some(format!(
            "its process list is {:.0}s old",
            (now - at).max(0.0)
        )),
        Some(_) => None,
    }
}

/// `pid 51877 (python3, 1.4 GiB)`.
pub fn described(process: &Process) -> String {
    let name = process.owner.name.as_deref().unwrap_or("-");
    match process.memory {
        Some(memory) => format!("pid {} ({name}, {})", process.pid, bytes(memory as f64)),
        None => format!("pid {} ({name})", process.pid),
    }
}

/// Who holds each GPU of `inputs.live`, in device order.
pub fn occupancy(inputs: &Inputs) -> Vec<Gpu> {
    let live = inputs.live;
    // The processes on GPUs whose lists are fresh.
    let fresh = (0..live.devices.len())
        .map(|index| {
            unknown(live, index, inputs.interval, inputs.now)
                .is_none()
                .then(|| {
                    live.listings[index]
                        .current()
                        .map_or(&[][..], |(_, processes)| processes.as_slice())
                })
        })
        .collect::<Vec<_>>();
    let holding = fresh
        .iter()
        .flatten()
        .flat_map(|processes| processes.iter().map(|process| process.pid))
        .collect::<Vec<_>>();
    let serves = inputs.serves(&holding);
    let mut gpus = live
        .devices
        .iter()
        .enumerate()
        .map(|(index, device)| {
            let mut gpu = Gpu {
                index: device.index,
                uuid: device.uuid.clone(),
                memory_total: device.memory_total,
                occupancy: Occupancy::Free,
                graphics: Vec::new(),
            };
            match fresh[index] {
                None => {
                    gpu.occupancy = Occupancy::Unknown(
                        unknown(live, index, inputs.interval, inputs.now).unwrap_or_default(),
                    );
                }
                Some(processes) => {
                    (gpu.occupancy, gpu.graphics) = holders(inputs, &serves, processes);
                }
            }
            gpu
        })
        .collect::<Vec<_>>();
    // Pending launches hold their GPUs before their processes show.
    for reservation in inputs.reservations {
        let launch = reservation.launch;
        for gpu in gpus.iter_mut().filter(|gpu| {
            gpu.uuid
                .as_ref()
                .is_some_and(|uuid| reservation.gpus.contains(uuid))
        }) {
            match &mut gpu.occupancy {
                Occupancy::Free => gpu.occupancy = Occupancy::Held(vec![Holder::Launch(launch)]),
                Occupancy::Held(holders) if !holders.contains(&Holder::Launch(launch)) => {
                    holders.push(Holder::Launch(launch));
                }
                _ => {}
            }
        }
    }
    // A serve process of this user that holds no GPU yet is loading; the
    // GPUs its CUDA_VISIBLE_DEVICES names are its own. Without UUIDs there,
    // every GPU might be.
    for serve in serves.iter().filter(|serve| serve.starting) {
        // A launch's process, or one a launch waits for, which may have
        // let go of its GPUs while it drains.
        if inputs
            .launch_of(serve.pid, Some(serve.start_ticks))
            .is_some()
            || inputs.waited(serve.pid, Some(serve.start_ticks)).is_some()
        {
            continue;
        }
        let visible = live.found.iter().find_map(|found| match found {
            Found::Process(process) if process.pid == serve.pid => {
                Some(process.cuda_visible_devices.clone())
            }
            _ => None,
        });
        let entries = visible
            .flatten()
            .map(|value| {
                value
                    .split(',')
                    .map(str::trim)
                    .filter(|entry| !entry.is_empty())
                    .map(str::to_owned)
                    .collect::<Vec<_>>()
            })
            .filter(|entries| entries.iter().all(|entry| is_uuid(entry)));
        match entries {
            Some(uuids) => {
                for gpu in gpus.iter_mut().filter(|gpu| {
                    gpu.uuid.as_ref().is_some_and(|uuid| uuids.contains(uuid))
                        && !matches!(gpu.occupancy, Occupancy::Unknown(_))
                }) {
                    gpu.occupancy =
                        Occupancy::Blocked(format!("{}, which is starting", serve.described()));
                }
            }
            None => {
                for gpu in gpus
                    .iter_mut()
                    .filter(|gpu| !matches!(gpu.occupancy, Occupancy::Unknown(_)))
                {
                    gpu.occupancy = Occupancy::Unknown(format!(
                        "{} is starting without GPU UUIDs in CUDA_VISIBLE_DEVICES",
                        serve.label()
                    ));
                }
            }
        }
    }
    gpus
}

/// Who holds a GPU with these `processes`, and the graphics processes on
/// it.
fn holders(inputs: &Inputs, serves: &[Serve], processes: &[Process]) -> (Occupancy, Vec<Process>) {
    let mut holders = Vec::new();
    let mut blocked = None;
    let mut graphics = Vec::new();
    for process in processes {
        let owner = &process.owner;
        if let Some(launch) = inputs.launch_of(process.pid, owner.start_ticks) {
            holders.push(Holder::Launch(launch));
        } else if let Some(waited) = inputs.waited(process.pid, owner.start_ticks) {
            holders.push(waited.holder.clone());
        } else if let Some(serve) = serves
            .iter()
            .find(|serve| serve.pid == process.pid && owner.start_ticks == Some(serve.start_ticks))
        {
            holders.push(Holder::Serve(serve.clone()));
        } else if owner.listen.is_some() {
            // serve, but not a process of this user in this namespace.
            let what = match &owner.container {
                Some(id) => inputs.container(id),
                None => match crate::gpu::owners::row(inputs.instances, process)
                    .and_then(|row| inputs.instances.iter().find(|seen| seen.id == row))
                {
                    Some(row) => sanitize(&row.label()),
                    None => format!("serve pid {} of another user", process.pid),
                },
            };
            blocked.get_or_insert(what);
        } else if process.kind == Kind::Graphics {
            graphics.push(process.clone());
        } else {
            // Any other compute process: choptop does not stop it, and a
            // launch would share the GPU with it.
            blocked.get_or_insert_with(|| described(process));
        }
    }
    holders.dedup();
    let occupancy = match blocked {
        Some(what) => Occupancy::Blocked(what),
        None if holders.is_empty() => Occupancy::Free,
        None => Occupancy::Held(holders),
    };
    (occupancy, graphics)
}

/// Who holds each port: pending launches, discovered instances, and a
/// loading process's `--listen`.
pub fn ports(inputs: &Inputs) -> BTreeMap<u16, PortHolder> {
    let holding = inputs
        .live
        .listings
        .iter()
        .filter_map(Listing::current)
        .flat_map(|(_, processes)| processes.iter().map(|process| process.pid))
        .collect::<Vec<_>>();
    let mut ports = BTreeMap::new();
    for reservation in inputs.reservations {
        ports
            .entry(reservation.port)
            .or_insert(PortHolder::Launch(reservation.launch));
    }
    for serve in inputs.serves(&holding) {
        let holder = match inputs.launch_of(serve.pid, Some(serve.start_ticks)) {
            Some(launch) => PortHolder::Launch(launch),
            None => PortHolder::Serve(serve.clone()),
        };
        for port in &serve.ports {
            ports.entry(*port).or_insert_with(|| holder.clone());
        }
    }
    for found in &inputs.live.found {
        if let Found::Container(container) = found {
            for address in &container.addresses {
                ports
                    .entry(address.port())
                    .or_insert_with(|| PortHolder::Other(inputs.container(&container.id)));
            }
        }
    }
    ports
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::discover::docker::ServeContainer;
    use crate::discover::proc::ServeProcess;
    use crate::gpu::{Owner, Reading};
    use crate::instances::Registry;
    use std::path::PathBuf;

    /// The made-up UUID of GPU `index`, written so that no UUID appears in
    /// the source.
    pub(crate) fn uuid(index: u32) -> String {
        format!("GPU-{}-0000-0000-0000-{index:012}", "0".repeat(8))
    }

    pub(crate) fn devices(count: u32) -> Vec<Device> {
        (0..count)
            .map(|index| Device {
                index,
                uuid: Some(uuid(index)),
                memory_total: Some(96 << 30),
                ..Device::default()
            })
            .collect()
    }

    pub(crate) fn serve_process(pid: u32, listening: &[&str], visible: Option<&str>) -> Found {
        Found::Process(ServeProcess {
            pid,
            start_ticks: u64::from(pid) * 10,
            executable: "chopinfer".into(),
            arguments: vec!["chopinfer".into(), "serve".into()],
            model_hint: Some(format!("model-{pid}")),
            age: None,
            listening: listening.iter().map(|text| text.parse().unwrap()).collect(),
            cuda_visible_devices: visible.map(str::to_owned),
        })
    }

    /// The GPU process of serve process `pid` from [`serve_process`].
    pub(crate) fn on_gpu(pid: u32) -> Process {
        Process {
            pid,
            memory: Some(80 << 30),
            kind: Kind::Compute,
            owner: Owner {
                start_ticks: Some(u64::from(pid) * 10),
                name: Some("chopinfer".into()),
                listen: Some("127.0.0.1:8310".into()),
                ..Owner::default()
            },
        }
    }

    fn other(pid: u32, name: &str, kind: Kind) -> Process {
        Process {
            pid,
            memory: Some(3 << 29),
            kind,
            owner: Owner {
                start_ticks: Some(1),
                name: Some(name.into()),
                ..Owner::default()
            },
        }
    }

    fn polled(at: f64, lists: Vec<Option<Vec<Process>>>) -> Report {
        Report::Polled(
            lists
                .into_iter()
                .map(|processes| Reading {
                    at,
                    processes,
                    answered: true,
                    ..Reading::default()
                })
                .collect(),
        )
    }

    fn states(live: &Live, reservations: &[Reservation], now: f64) -> Vec<Occupancy> {
        occupancy(&Inputs {
            live,
            instances: &[],
            reservations,
            waited: &[],
            interval: 1.0,
            now,
        })
        .into_iter()
        .map(|gpu| gpu.occupancy)
        .collect()
    }

    fn unknown_reason(occupancy: &Occupancy) -> &str {
        match occupancy {
            Occupancy::Unknown(reason) => reason,
            other => panic!("{other:?} is not unknown"),
        }
    }

    #[test]
    fn a_gpu_is_known_only_from_a_fresh_process_list() {
        let mut live = Live::default();
        live.gpu(&Report::Opened {
            driver: None,
            devices: devices(3),
        });
        // No read yet: nothing is free.
        let before = states(&live, &[], 0.5);
        assert!(
            before
                .iter()
                .all(|state| matches!(state, Occupancy::Unknown(_)))
        );
        assert_eq!(
            unknown_reason(&before[0]),
            "its processes have not been read"
        );
        live.gpu(&polled(1.0, vec![Some(vec![]), Some(vec![]), None]));
        let now = states(&live, &[], 1.5);
        assert_eq!(now[0], Occupancy::Free);
        // A failed list is unknown, never free.
        assert_eq!(
            unknown_reason(&now[2]),
            "its process list could not be read"
        );
        // A failed read right after a good one is unknown too: the list
        // before it does not stand in...
        live.gpu(&polled(2.0, vec![None, Some(vec![]), Some(vec![])]));
        let failed = states(&live, &[], 2.0);
        assert_eq!(
            unknown_reason(&failed[0]),
            "its process list could not be read"
        );
        assert_eq!(failed[2], Occupancy::Free);
        // ...until the next read succeeds.
        live.gpu(&polled(3.0, vec![Some(vec![]), Some(vec![]), Some(vec![])]));
        assert_eq!(states(&live, &[], 3.0)[0], Occupancy::Free);
        // A list read more than two polls ago is stale.
        let stale = states(&live, &[], 6.0);
        assert_eq!(unknown_reason(&stale[0]), "its process list is 3s old");
        // Readings stopped: every list goes stale.
        let later = states(&live, &[], 10.0);
        assert!(
            later
                .iter()
                .all(|state| matches!(state, Occupancy::Unknown(_)))
        );
    }

    #[test]
    fn a_lost_gpu_or_one_without_a_uuid_is_unknown() {
        let mut live = Live::default();
        let mut devices = devices(3);
        devices[1].uuid = None;
        devices[2].unreadable = Some("no permission".into());
        live.gpu(&Report::Opened {
            driver: None,
            devices,
        });
        live.gpu(&polled(1.0, vec![Some(vec![]); 3]));
        let mut lost = polled(2.0, vec![None, Some(vec![]), Some(vec![])]);
        if let Report::Polled(readings) = &mut lost {
            readings[0].lost = true;
            readings[0].answered = false;
        }
        live.gpu(&lost);
        let now = states(&live, &[], 2.0);
        assert_eq!(unknown_reason(&now[0]), "lost or off the bus");
        assert_eq!(unknown_reason(&now[1]), "its UUID cannot be read");
        assert_eq!(
            unknown_reason(&now[2]),
            "NVML cannot open it: no permission"
        );
        live.gpu(&Report::Unavailable("no driver".into()));
        assert!(states(&live, &[], 2.0).is_empty());
        assert_eq!(live.unavailable.as_deref(), Some("no driver"));
    }

    #[test]
    fn serve_processes_containers_and_strangers_hold_gpus() {
        let mut registry = Registry::new(1.0);
        registry.reconcile(
            vec![
                serve_process(48211, &["127.0.0.1:8311"], None),
                Found::Container(ServeContainer {
                    daemon: PathBuf::from("/run/docker.sock"),
                    id: "4f2a".into(),
                    name: "glm-tp4".into(),
                    arguments: Vec::new(),
                    model_hint: None,
                    seen_for: 0.0,
                    addresses: vec!["127.0.0.1:8600".parse().unwrap()],
                }),
            ],
            0.0,
        );
        let mut live = Live::default();
        live.gpu(&Report::Opened {
            driver: None,
            devices: devices(5),
        });
        live.scan(&discover::Report::Scan {
            found: vec![serve_process(48211, &["127.0.0.1:8311"], None)],
            events: Vec::new(),
            started: 0.0,
        });
        let mut in_container = on_gpu(700);
        in_container.owner.container = Some("4f2a".into());
        live.gpu(&polled(
            1.0,
            vec![
                Some(vec![on_gpu(48211), other(2210, "Xorg", Kind::Graphics)]),
                Some(vec![on_gpu(48211), other(51877, "python3", Kind::Compute)]),
                Some(vec![in_container]),
                Some(vec![other(51877, "python3", Kind::Compute)]),
                Some(vec![other(2210, "Xorg", Kind::Graphics)]),
            ],
        ));
        let gpus = occupancy(&Inputs {
            live: &live,
            instances: &registry.instances,
            reservations: &[],
            waited: &[],
            interval: 1.0,
            now: 1.0,
        });
        let serve = Serve {
            pid: 48211,
            start_ticks: 482_110,
            name: Some("model-48211".into()),
            ports: vec![8311],
            starting: false,
        };
        // A graphics process beside a serve process leaves it held, and is
        // carried along.
        assert_eq!(
            gpus[0].occupancy,
            Occupancy::Held(vec![Holder::Serve(serve)])
        );
        assert_eq!(described(&gpus[0].graphics[0]), "pid 2210 (Xorg, 1.5 GiB)");
        // A compute process that is not serve blocks the GPU, even beside a
        // serve process a switch would stop.
        assert_eq!(
            gpus[1].occupancy,
            Occupancy::Blocked("pid 51877 (python3, 1.5 GiB)".into())
        );
        assert!(gpus[1].graphics.is_empty());
        assert_eq!(
            gpus[2].occupancy,
            Occupancy::Blocked("docker glm-tp4".into())
        );
        // So does one alone.
        assert_eq!(
            gpus[3].occupancy,
            Occupancy::Blocked("pid 51877 (python3, 1.5 GiB)".into())
        );
        // A GPU with only a desktop's graphics process is free, and says
        // which process is there.
        assert_eq!(gpus[4].occupancy, Occupancy::Free);
        assert_eq!(gpus[4].graphics.len(), 1);
        assert_eq!(gpus[4].graphics[0].pid, 2210);
        let ports = ports(&Inputs {
            live: &live,
            instances: &registry.instances,
            reservations: &[],
            waited: &[],
            interval: 1.0,
            now: 1.0,
        });
        assert!(matches!(&ports[&8311], PortHolder::Serve(serve) if serve.pid == 48211));
        assert_eq!(ports.get(&8600), None);
    }

    /// GPU 0 of a host with `processes` on it. Serve process 48211 of this
    /// user is found when it is one of them; found elsewhere, it would be
    /// loading, which leaves every GPU unknown.
    fn alone(processes: Vec<Process>) -> Gpu {
        let mut live = Live::default();
        live.gpu(&Report::Opened {
            driver: None,
            devices: devices(1),
        });
        let serving = processes.iter().any(|process| process.pid == 48211);
        live.scan(&discover::Report::Scan {
            found: serving
                .then(|| serve_process(48211, &["127.0.0.1:8310"], None))
                .into_iter()
                .collect(),
            events: Vec::new(),
            started: 0.0,
        });
        live.gpu(&polled(1.0, vec![Some(processes)]));
        occupancy(&Inputs {
            live: &live,
            instances: &[],
            reservations: &[],
            waited: &[],
            interval: 1.0,
            now: 1.0,
        })
        .remove(0)
    }

    #[test]
    fn a_compute_process_that_is_not_serve_blocks_even_beside_serve() {
        for kind in [Kind::Compute, Kind::Both] {
            let gpu = alone(vec![on_gpu(48211), other(51877, "python3", kind)]);
            assert_eq!(
                gpu.occupancy,
                Occupancy::Blocked("pid 51877 (python3, 1.5 GiB)".into()),
                "{kind:?}"
            );
        }
    }

    #[test]
    fn a_graphics_process_does_not_block_and_is_carried_as_a_warning() {
        let gpu = alone(vec![on_gpu(48211), other(2210, "Xorg", Kind::Graphics)]);
        assert!(
            matches!(&gpu.occupancy, Occupancy::Held(holders) if matches!(&holders[..], [Holder::Serve(serve)] if serve.pid == 48211))
        );
        assert_eq!(
            gpu.graphics
                .iter()
                .map(|process| process.pid)
                .collect::<Vec<_>>(),
            [2210]
        );
    }

    #[test]
    fn a_gpu_with_only_graphics_processes_is_free_and_names_them() {
        let gpu = alone(vec![other(2210, "Xorg", Kind::Graphics)]);
        assert_eq!(gpu.occupancy, Occupancy::Free);
        assert_eq!(described(&gpu.graphics[0]), "pid 2210 (Xorg, 1.5 GiB)");
        let empty = alone(Vec::new());
        assert_eq!(empty.occupancy, Occupancy::Free);
        assert!(empty.graphics.is_empty());
    }

    fn reservation(
        label: &str,
        gpus: &[u32],
        port: u16,
        process: Option<(u32, u64)>,
    ) -> Reservation {
        Reservation {
            label: label.into(),
            model: "Qwen3.8-Flash".into(),
            gpus: gpus.iter().map(|index| uuid(*index)).collect(),
            port,
            process,
            launch: u64::from(port),
        }
    }

    #[test]
    fn a_launch_holds_its_gpus_and_port_before_its_process_shows() {
        let mut live = Live::default();
        live.gpu(&Report::Opened {
            driver: None,
            devices: devices(4),
        });
        live.gpu(&polled(1.0, vec![Some(vec![]); 4]));
        // One waits to start; one started and is loading, with no GPU
        // process and no listener yet.
        live.scan(&discover::Report::Scan {
            found: vec![serve_process(61204, &[], None)],
            events: Vec::new(),
            started: 0.0,
        });
        let reservations = [
            reservation("launch A", &[0, 1], 8310, None),
            reservation("proc 61204", &[2, 3], 8311, Some((61204, 612_040))),
        ];
        let inputs = Inputs {
            live: &live,
            instances: &[],
            reservations: &reservations,
            waited: &[],
            interval: 1.0,
            now: 1.5,
        };
        let held = occupancy(&inputs)
            .into_iter()
            .map(|gpu| gpu.occupancy)
            .collect::<Vec<_>>();
        assert_eq!(
            held,
            [
                Occupancy::Held(vec![Holder::Launch(8310)]),
                Occupancy::Held(vec![Holder::Launch(8310)]),
                Occupancy::Held(vec![Holder::Launch(8311)]),
                Occupancy::Held(vec![Holder::Launch(8311)]),
            ]
        );
        let ports = ports(&inputs);
        assert_eq!(ports[&8310], PortHolder::Launch(8310));
        assert_eq!(ports[&8311], PortHolder::Launch(8311));
        // Once its process holds the GPUs it is still that launch.
        live.gpu(&polled(
            2.0,
            vec![
                None,
                None,
                Some(vec![on_gpu(61204)]),
                Some(vec![on_gpu(61204)]),
            ],
        ));
        let gpus = occupancy(&Inputs {
            live: &live,
            instances: &[],
            reservations: &reservations,
            waited: &[],
            interval: 1.0,
            now: 2.0,
        });
        assert_eq!(
            gpus[2].occupancy,
            Occupancy::Held(vec![Holder::Launch(8311)])
        );
    }

    #[test]
    fn a_starting_process_reserves_the_gpus_its_uuids_name() {
        let mut live = Live::default();
        live.gpu(&Report::Opened {
            driver: None,
            devices: devices(4),
        });
        live.gpu(&polled(1.0, vec![Some(vec![]); 4]));
        let named = format!("{},{}", uuid(2), uuid(3));
        live.scan(&discover::Report::Scan {
            found: vec![serve_process(51022, &[], Some(&named))],
            events: Vec::new(),
            started: 0.0,
        });
        let held = states(&live, &[], 1.0);
        assert_eq!(held[0], Occupancy::Free);
        assert_eq!(
            held[2],
            Occupancy::Blocked("proc 51022 model-51022, which is starting".into())
        );
        assert_eq!(held[3], held[2]);
        // Each reads after "held by".
        let gpus = occupancy(&Inputs {
            live: &live,
            instances: &[],
            reservations: &[],
            waited: &[],
            interval: 1.0,
            now: 1.0,
        });
        let placement = crate::launch::placement::place(
            &crate::launch::placement::Field {
                gpus: &gpus,
                ports: &BTreeMap::new(),
                reservations: &[],
            },
            &[2, 3].into(),
            2,
            None,
            &mut |_, _| true,
        );
        assert_eq!(
            placement.problems,
            [
                "GPU 2 is held by proc 51022 model-51022, which is starting",
                "GPU 3 is held by proc 51022 model-51022, which is starting",
            ]
        );
        // With ordinals, or without the variable, any GPU may be its.
        for visible in [Some("2,3"), None, Some(&format!("{},1", uuid(2))[..])] {
            live.scan(&discover::Report::Scan {
                found: vec![serve_process(51022, &[], visible)],
                events: Vec::new(),
                started: 0.0,
            });
            for state in states(&live, &[], 1.0) {
                assert_eq!(
                    unknown_reason(&state),
                    "proc 51022 is starting without GPU UUIDs in CUDA_VISIBLE_DEVICES",
                    "{visible:?}"
                );
            }
        }
        // Once it shows up on its GPUs it holds them like any other.
        live.gpu(&polled(
            2.0,
            vec![
                Some(vec![]),
                Some(vec![]),
                Some(vec![on_gpu(51022)]),
                Some(vec![on_gpu(51022)]),
            ],
        ));
        let held = states(&live, &[], 2.0);
        assert_eq!(held[0], Occupancy::Free);
        assert!(
            matches!(&held[2], Occupancy::Held(holders) if matches!(&holders[0], Holder::Serve(serve) if serve.pid == 51022))
        );
    }

    #[test]
    fn a_waited_process_holds_its_gpu_as_what_it_stands_for() {
        let mut live = Live::default();
        live.gpu(&Report::Opened {
            driver: None,
            devices: devices(4),
        });
        // A reading taken before a replaced launch (500) and a stopped serve
        // process (600) exited, the latter's start time no longer readable,
        // and a new process that reuses pid 500.
        let gone = Process {
            owner: Owner::default(),
            ..on_gpu(600)
        };
        let reused = Process {
            owner: Owner {
                start_ticks: Some(1),
                ..Owner::default()
            },
            ..on_gpu(500)
        };
        live.gpu(&polled(
            1.0,
            vec![
                Some(vec![on_gpu(500)]),
                Some(vec![gone]),
                Some(vec![reused]),
                Some(Vec::new()),
            ],
        ));
        // A scan from before the replaced launch exited, and another
        // replaced launch that has let go of GPU 3 while it drains.
        live.scan(&discover::Report::Scan {
            found: vec![
                serve_process(500, &["127.0.0.1:8310"], Some(&uuid(0))),
                serve_process(700, &[], Some(&uuid(3))),
            ],
            events: Vec::new(),
            started: 0.5,
        });
        let serve = Serve {
            pid: 600,
            start_ticks: 6000,
            name: None,
            ports: Vec::new(),
            starting: false,
        };
        let waited = [
            Waited {
                pid: 500,
                start_ticks: 5000,
                holder: Holder::Launch(3),
            },
            Waited {
                pid: 600,
                start_ticks: 6000,
                holder: Holder::Serve(serve.clone()),
            },
            Waited {
                pid: 700,
                start_ticks: 7000,
                holder: Holder::Launch(4),
            },
        ];
        let held = |waited: &[Waited]| {
            occupancy(&Inputs {
                live: &live,
                instances: &[],
                reservations: &[],
                waited,
                interval: 1.0,
                now: 1.0,
            })
            .into_iter()
            .map(|gpu| gpu.occupancy)
            .collect::<Vec<_>>()
        };
        assert_eq!(
            held(&waited),
            [
                Occupancy::Held(vec![Holder::Launch(3)]),
                Occupancy::Held(vec![Holder::Serve(serve)]),
                Occupancy::Blocked("pid 500 (-, 80 GiB)".into()),
                Occupancy::Free,
            ]
        );
        // Unwaited, the first is a serve process of this user, the second a
        // stranger, and the last a serve process that is loading.
        let unwaited = held(&[]);
        assert!(
            matches!(&unwaited[0], Occupancy::Held(holders) if matches!(&holders[0], Holder::Serve(serve) if serve.pid == 500))
        );
        assert_eq!(
            unwaited[1],
            Occupancy::Blocked("pid 600 (-, 80 GiB)".into())
        );
        assert_eq!(
            unwaited[3],
            Occupancy::Blocked("proc 700 model-700, which is starting".into())
        );
    }

    #[test]
    fn readings_are_fresh_once_a_scan_and_a_gpu_read_follow_the_request() {
        let mut live = Live {
            scanning: true,
            ..Live::default()
        };
        assert!(!live.fresh(5.0));
        live.gpu(&Report::Opened {
            driver: None,
            devices: devices(2),
        });
        live.gpu(&polled(5.5, vec![Some(vec![]); 2]));
        live.scan(&discover::Report::Scan {
            found: Vec::new(),
            events: Vec::new(),
            started: 4.9,
        });
        // The scan began before the request.
        assert!(!live.fresh(5.0));
        live.scan(&discover::Report::Scan {
            found: Vec::new(),
            events: Vec::new(),
            started: 5.1,
        });
        assert!(live.fresh(5.0));
        assert!(!live.fresh(5.6));
        // Without GPUs there is no read to wait for, and without
        // discovery no scan.
        live.gpu(&Report::Unavailable("no driver".into()));
        assert!(live.fresh(5.1));
        assert!(!live.fresh(5.2));
        live.scanning = false;
        assert!(live.fresh(5.2));
    }

    #[test]
    fn a_listener_that_opens_later_updates_the_scan() {
        let mut live = Live::default();
        live.scan(&discover::Report::Scan {
            found: vec![serve_process(9, &[], None)],
            events: Vec::new(),
            started: 0.0,
        });
        live.scan(&discover::Report::Listening {
            pid: 9,
            start_ticks: 90,
            addresses: vec!["127.0.0.1:8399".parse().unwrap()],
        });
        let Found::Process(process) = &live.found[0] else {
            panic!("a process")
        };
        assert_eq!(process.listening, ["127.0.0.1:8399".parse().unwrap()]);
    }
}
