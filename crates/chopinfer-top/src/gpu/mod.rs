//! The GPUs of this host, read through NVML: temperature, power, clocks,
//! memory, activity and the processes on each, and which instance those
//! processes are.
//!
//! Only the host chopinfer-top runs on is visible; an instance on another
//! host shows no GPUs.

pub mod nvml;
pub mod owners;
pub mod poll;

use std::collections::VecDeque;

use crate::instances::{Instance, Registry, Tone};
use crate::stats::{HISTORY_SECONDS, RATE_POLLS};
use crate::ui::widgets::integer;
pub use owners::Owner;
use poll::Report;

/// What does not change about one GPU while chopinfer-top runs.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Device {
    /// NVML's index, the one nvidia-smi shows; not a CUDA ordinal.
    pub index: u32,
    pub name: String,
    /// Bytes.
    pub memory_total: Option<u64>,
    pub sm_max_mhz: Option<u32>,
    /// The fastest PCIe link, as `(generation, width)`.
    pub pcie_max: Option<(u32, u32)>,
    /// The temperature headroom counts down to: the maximum operating
    /// temperature, else the slowdown threshold.
    pub temperature_limit: Option<i32>,
    /// Whether ECC is on.
    pub ecc: Option<bool>,
    /// Whether GPM counters can be read: Hopper and newer.
    pub gpm: bool,
    /// Why NVML gave no handle for it, such as no permission; `None` when
    /// it opened.
    pub unreadable: Option<String>,
}

impl Device {
    /// The name without the vendor and edition: `RTX PRO 6000 Max-Q`.
    pub fn short_name(&self) -> String {
        let name = self.name.strip_prefix("NVIDIA ").unwrap_or(&self.name);
        let name = name
            .strip_suffix(" Workstation Edition")
            .or_else(|| name.strip_suffix(" Server Edition"))
            .unwrap_or(name);
        name.replace(" Blackwell", "")
    }
}

/// The clock event reasons of `nvmlDeviceGetCurrentClocksEventReasons`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Reasons(pub u64);

impl Reasons {
    pub const IDLE: u64 = 0x1;
    pub const APPLICATION_CLOCKS: u64 = 0x2;
    pub const POWER_CAP: u64 = 0x4;
    pub const HARDWARE: u64 = 0x8;
    pub const SYNC_BOOST: u64 = 0x10;
    pub const SOFTWARE_THERMAL: u64 = 0x20;
    pub const HARDWARE_THERMAL: u64 = 0x40;
    pub const POWER_BRAKE: u64 = 0x80;
    pub const DISPLAY_CLOCKS: u64 = 0x100;
    /// Newer than the CUDA 13.2 header; nvidia-smi calls it Reliability.
    pub const RELIABILITY: u64 = 0x400;

    fn has(self, bits: u64) -> bool {
        self.0 & bits != 0
    }

    pub fn thermal(self) -> bool {
        self.has(Self::SOFTWARE_THERMAL | Self::HARDWARE_THERMAL)
    }

    /// A slowdown beyond the power cap: thermal, a hardware slowdown or the
    /// power brake.
    pub fn slowdown(self) -> bool {
        self.has(
            Self::SOFTWARE_THERMAL | Self::HARDWARE_THERMAL | Self::HARDWARE | Self::POWER_BRAKE,
        )
    }

    /// What holds the clocks back, the most serious reason first; `None`
    /// when nothing does.
    pub fn label(self) -> Option<&'static str> {
        [
            (Self::HARDWARE_THERMAL, "hardware thermal slowdown"),
            (Self::SOFTWARE_THERMAL, "thermal slowdown"),
            (Self::POWER_BRAKE, "power brake slowdown"),
            (Self::HARDWARE, "hardware slowdown"),
            (Self::POWER_CAP, "power cap"),
            (Self::RELIABILITY, "reliability"),
            (Self::APPLICATION_CLOCKS, "application clocks"),
            (Self::SYNC_BOOST, "sync boost"),
            (Self::DISPLAY_CLOCKS, "display clocks"),
            (Self::IDLE, "idle"),
        ]
        .into_iter()
        .find(|(bit, _)| self.has(*bit))
        .map(|(_, label)| label)
    }
}

/// GPM figures between two samples: shares from 0 to 1, PCIe in bytes a
/// second.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Activity {
    /// Time any compute or graphics work ran, from any process.
    pub kernels: Option<f64>,
    pub sm: Option<f64>,
    pub occupancy: Option<f64>,
    pub tensor: Option<f64>,
    pub dram: Option<f64>,
    pub pcie_rx: Option<f64>,
    pub pcie_tx: Option<f64>,
}

/// Which NVML lists a process is on, shown as nvidia-smi does.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Kind {
    #[default]
    Compute,
    Graphics,
    Both,
}

impl Kind {
    pub fn label(self) -> &'static str {
        match self {
            Self::Compute => "C",
            Self::Graphics => "G",
            Self::Both => "C+G",
        }
    }
}

/// A process holding a GPU.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Process {
    pub pid: u32,
    /// Bytes, when the driver says.
    pub memory: Option<u64>,
    pub kind: Kind,
    pub owner: Owner,
}

/// The compute and graphics lists as one, by pid. A process on both is
/// listed once with the larger memory figure: both lists report the same
/// allocation, so their sum would count it twice. Largest first.
pub fn merge(compute: &[(u32, Option<u64>)], graphics: &[(u32, Option<u64>)]) -> Vec<Process> {
    let mut processes: Vec<Process> = Vec::new();
    for (list, kind) in [(compute, Kind::Compute), (graphics, Kind::Graphics)] {
        for (pid, memory) in list {
            match processes.iter_mut().find(|process| process.pid == *pid) {
                Some(process) => {
                    if process.kind != kind {
                        process.kind = Kind::Both;
                    }
                    process.memory = match (process.memory, *memory) {
                        (Some(a), Some(b)) => Some(a.max(b)),
                        (a, b) => a.or(b),
                    };
                }
                None => processes.push(Process {
                    pid: *pid,
                    memory: *memory,
                    kind,
                    owner: Owner::default(),
                }),
            }
        }
    }
    processes.sort_by(|a, b| {
        b.memory
            .unwrap_or(0)
            .cmp(&a.memory.unwrap_or(0))
            .then(a.pid.cmp(&b.pid))
    });
    processes
}

/// One poll of one GPU; `None` where NVML does not say.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Reading {
    /// Seconds on chopinfer-top's clock.
    pub at: f64,
    /// Celsius.
    pub temperature: Option<i32>,
    /// Degrees under [`Device::temperature_limit`].
    pub headroom: Option<i32>,
    /// Percent of the fan's top speed.
    pub fan: Option<u32>,
    /// Watts, as NVML reports power: averaged over a second on Ampere and
    /// newer.
    pub power: Option<f64>,
    /// The enforced power limit in watts.
    pub power_limit: Option<f64>,
    pub sm_mhz: Option<u32>,
    pub memory_mhz: Option<u32>,
    /// The P-state number.
    pub pstate: Option<u32>,
    pub reasons: Option<Reasons>,
    /// Bytes in use.
    pub memory_used: Option<u64>,
    /// NVML's `(gpu, memory)` utilization: the share of its sample period
    /// with a kernel running, and with device memory being read or written.
    pub utilization: Option<(f64, f64)>,
    /// The current PCIe link, as `(generation, width)`.
    pub pcie: Option<(u32, u32)>,
    /// Uncorrected ECC errors since the driver loaded, while ECC is on.
    pub ecc_uncorrected: Option<u64>,
    /// `None` when the process lists were not read: a failed query is not
    /// an empty GPU.
    pub processes: Option<Vec<Process>>,
    /// GPM figures since the previous poll, while they are read.
    pub activity: Option<Activity>,
    /// The GPU stopped answering: it fell off the bus or is otherwise lost.
    pub lost: bool,
    /// Whether any query of this reading succeeded. A reading where none
    /// did says nothing about the GPU, not even that it is back.
    pub answered: bool,
}

impl Reading {
    /// The share of time a kernel ran: GPM's measure where read, else
    /// NVML's coarser utilization.
    pub fn kernels(&self) -> Option<f64> {
        self.activity
            .and_then(|activity| activity.kernels)
            .or(self.utilization.map(|(gpu, _)| gpu))
    }
}

/// A value of a reading, for averages and plots.
pub type Field = fn(&Reading) -> Option<f64>;

/// One GPU's recent readings: five minutes of polls, as an instance keeps.
#[derive(Clone, Debug)]
pub struct History {
    ring: VecDeque<Reading>,
    capacity: usize,
    /// Readings dropped from the front since the history began.
    dropped: u64,
}

impl History {
    pub fn new(interval: f64) -> Self {
        let capacity = (HISTORY_SECONDS / interval.max(0.1)).ceil() as usize + 1;
        Self {
            ring: VecDeque::with_capacity(capacity),
            capacity,
            dropped: 0,
        }
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// The number of the oldest reading held, counting from 0 at the first.
    pub fn oldest_poll(&self) -> u64 {
        self.dropped
    }

    pub fn push(&mut self, reading: Reading) {
        if self.ring.len() == self.capacity {
            self.ring.pop_front();
            self.dropped += 1;
        }
        self.ring.push_back(reading);
    }

    pub fn latest(&self) -> Option<&Reading> {
        self.ring.back()
    }

    /// The latest reading whose process lists were read.
    pub fn listed(&self) -> Option<&Reading> {
        self.ring
            .iter()
            .rev()
            .find(|reading| reading.processes.is_some())
    }

    /// The average of `field` over the last three polls that have it, as
    /// rates are taken for instances.
    pub fn recent(&self, field: Field) -> Option<f64> {
        let values = self
            .ring
            .iter()
            .rev()
            .take(RATE_POLLS)
            .filter_map(field)
            .collect::<Vec<_>>();
        (!values.is_empty()).then(|| values.iter().sum::<f64>() / values.len() as f64)
    }

    /// `field` at each poll, 0 where missing.
    pub fn series(&self, field: Field) -> Vec<f64> {
        self.ring
            .iter()
            .map(|reading| field(reading).unwrap_or(0.0))
            .collect()
    }
}

/// Whether the GPUs can be read.
#[derive(Clone, Debug, PartialEq)]
pub enum Status {
    /// `--no-gpus`.
    Off,
    /// NVML is being opened.
    Opening,
    /// NVML could not be opened, and why.
    Unavailable(String),
    Open,
}

/// What has been said about one GPU, so that each change is one event.
#[derive(Clone, Debug, Default)]
struct Watch {
    slowdown: bool,
    /// Under [`HOT_HEADROOM`] degrees of headroom, until [`COOL_HEADROOM`].
    hot: bool,
    ecc: u64,
    lost: bool,
    /// Processes beside an instance already reported.
    strangers: Vec<u32>,
}

/// Headroom that is reported, and headroom that ends the report.
const HOT_HEADROOM: i32 = 5;
const COOL_HEADROOM: i32 = 10;

/// What the interface knows about the GPUs.
#[derive(Clone, Debug)]
pub struct Gpus {
    pub status: Status,
    pub driver: Option<String>,
    pub devices: Vec<Device>,
    /// One per device, in device order.
    pub histories: Vec<History>,
    /// Whether GPM counters may be read: not `--no-gpu-counters`.
    pub counters: bool,
    interval: f64,
    watches: Vec<Watch>,
}

impl Gpus {
    pub fn new(status: Status, counters: bool, interval: f64) -> Self {
        Self {
            status,
            driver: None,
            devices: Vec::new(),
            histories: Vec::new(),
            counters,
            interval,
            watches: Vec::new(),
        }
    }

    /// For a chopinfer-top that does not read GPUs.
    pub fn off() -> Self {
        Self::new(Status::Off, false, 1.0)
    }

    /// Takes a report at `now`, noting in `registry` what changed: GPUs
    /// found, a slowdown starting or ending, a GPU near its temperature
    /// limit, uncorrected ECC errors, a GPU lost, and a process that is no
    /// instance holding memory beside one.
    pub fn apply(&mut self, report: Report, registry: &mut Registry, now: f64) {
        match report {
            Report::Opened { driver, devices } => {
                registry.note(
                    now,
                    "GPUs",
                    format!(
                        "{} found, driver {}",
                        devices.len(),
                        driver.as_deref().unwrap_or("unknown")
                    ),
                    Tone::Normal,
                );
                self.status = Status::Open;
                self.driver = driver;
                self.histories = devices
                    .iter()
                    .map(|_| History::new(self.interval))
                    .collect();
                self.watches = vec![Watch::default(); devices.len()];
                self.devices = devices;
            }
            Report::Unavailable(reason) => {
                registry.note(now, "GPUs", format!("not read: {reason}"), Tone::Normal);
                self.status = Status::Unavailable(reason);
            }
            Report::Polled(readings) => {
                for (index, reading) in readings.into_iter().enumerate().take(self.devices.len()) {
                    self.watch(index, &reading, registry, now);
                    self.histories[index].push(reading);
                }
            }
        }
    }

    fn watch(&mut self, index: usize, reading: &Reading, registry: &mut Registry, now: f64) {
        let device = &self.devices[index];
        let who = format!("GPU {}", device.index);
        let watch = &mut self.watches[index];
        if reading.lost {
            if !watch.lost {
                watch.lost = true;
                registry.note(
                    now,
                    &who,
                    "stopped answering NVML: lost or off the bus",
                    Tone::Bad,
                );
            }
            return;
        }
        // Every query failed: nothing is known, so nothing changes.
        if !reading.answered {
            return;
        }
        if watch.lost {
            watch.lost = false;
            registry.note(now, &who, "answering NVML again", Tone::Good);
        }
        // A reading without reasons, from a failed query, says nothing
        // about a slowdown.
        if let Some(reasons) = reading.reasons
            && reasons.slowdown() != watch.slowdown
        {
            watch.slowdown = reasons.slowdown();
            let clock = reading.sm_mhz.map(|mhz| integer(f64::from(mhz)));
            if watch.slowdown {
                let mut what = reasons.label().unwrap_or("slowdown").to_owned();
                if let Some(clock) = &clock {
                    what.push_str(&format!(": SM clock {clock}"));
                    match device.sm_max_mhz {
                        Some(max) => what.push_str(&format!(" of {} MHz", integer(f64::from(max)))),
                        None => what.push_str(" MHz"),
                    }
                }
                if let Some(headroom) = reading.headroom {
                    what.push_str(&format!(", {}C under the limit", headroom.max(0)));
                }
                registry.note(now, &who, what, Tone::Warning);
            } else {
                let what = clock.map_or_else(
                    || "slowdown over".to_owned(),
                    |clock| format!("slowdown over: SM clock {clock} MHz"),
                );
                registry.note(now, &who, what, Tone::Normal);
            }
        }
        if let Some(headroom) = reading.headroom {
            if !watch.hot && headroom < HOT_HEADROOM {
                watch.hot = true;
                let at = reading
                    .temperature
                    .map_or_else(String::new, |celsius| format!(" at {celsius}C"));
                registry.note(
                    now,
                    &who,
                    format!("{}C under the temperature limit{at}", headroom.max(0)),
                    Tone::Warning,
                );
            } else if watch.hot && headroom >= COOL_HEADROOM {
                watch.hot = false;
            }
        }
        if let Some(errors) = reading.ecc_uncorrected {
            // Each reading is the new baseline, so a cleared counter counts
            // again from its new value.
            let rose = errors > watch.ecc;
            watch.ecc = errors;
            if rose {
                registry.note(
                    now,
                    &who,
                    format!(
                        "{errors} uncorrected ECC error{} since the driver loaded",
                        if errors == 1 { "" } else { "s" }
                    ),
                    Tone::Bad,
                );
            }
        }
        // Without process lists nothing is known about the processes: none
        // is new, and none has gone.
        let Some(processes) = &reading.processes else {
            return;
        };
        let rows = processes
            .iter()
            .map(|process| owners::row(&registry.instances, process))
            .collect::<Vec<_>>();
        let beside = rows
            .iter()
            .flatten()
            .next()
            .and_then(|id| registry.get(*id))
            .map(|instance| {
                instance
                    .name()
                    .map_or_else(|| instance.label(), str::to_owned)
            });
        if let Some(beside) = beside {
            for (process, row) in processes.iter().zip(&rows) {
                if row.is_some() || watch.strangers.contains(&process.pid) {
                    continue;
                }
                watch.strangers.push(process.pid);
                let name = process.owner.name.as_deref().unwrap_or("-");
                let memory = process.memory.map_or_else(
                    || "memory".to_owned(),
                    |bytes| format!("{:.1} GiB", bytes as f64 / f64::from(1 << 30)),
                );
                registry.note(
                    now,
                    &who,
                    format!(
                        "pid {} ({name}) holds {memory} beside {beside}",
                        process.pid
                    ),
                    Tone::Warning,
                );
            }
        }
        watch
            .strangers
            .retain(|pid| processes.iter().any(|process| process.pid == *pid));
    }

    pub fn latest(&self, index: usize) -> Option<&Reading> {
        self.histories.get(index).and_then(History::latest)
    }

    /// Whether GPU `index` is lost: it said so and has not answered since.
    /// A poll where every query failed leaves this as it was.
    pub fn lost(&self, index: usize) -> bool {
        self.watches.get(index).is_some_and(|watch| watch.lost)
    }

    /// Whether any GPU can give GPM figures.
    pub fn gpm(&self) -> bool {
        self.devices.iter().any(|device| device.gpm)
    }

    /// The processes of GPU `index` with the visible row each one is, as
    /// last read: a poll whose process lists failed keeps the list before
    /// it, for as long as the history holds it.
    pub fn processes<'a>(
        &'a self,
        index: usize,
        instances: &[Instance],
    ) -> Vec<(&'a Process, Option<u64>)> {
        self.histories
            .get(index)
            .and_then(History::listed)
            .and_then(|reading| reading.processes.as_ref())
            .map_or_else(Vec::new, |processes| {
                processes
                    .iter()
                    .map(|process| (process, owners::row(instances, process)))
                    .collect()
            })
    }

    /// When the processes of GPU `index` were last read; `None` when they
    /// have not been within the history.
    pub fn processes_read(&self, index: usize) -> Option<f64> {
        self.histories
            .get(index)
            .and_then(History::listed)
            .map(|reading| reading.at)
    }

    /// The GPUs that a process of visible row `id` holds.
    pub fn of_instance(&self, instances: &[Instance], id: u64) -> Vec<u32> {
        self.devices
            .iter()
            .enumerate()
            .filter(|(index, _)| {
                self.processes(*index, instances)
                    .iter()
                    .any(|(_, row)| *row == Some(id))
            })
            .map(|(_, device)| device.index)
            .collect()
    }
}

/// GPU indices as runs: `0-3`, `0,2`, `0-1,4`.
pub fn ranges(indices: &[u32]) -> String {
    let mut runs: Vec<(u32, u32)> = Vec::new();
    for index in indices {
        match runs.last_mut() {
            Some((_, end)) if *end + 1 == *index => *end = *index,
            _ => runs.push((*index, *index)),
        }
    }
    runs.iter()
        .map(|(start, end)| {
            if start == end {
                start.to_string()
            } else {
                format!("{start}-{end}")
            }
        })
        .collect::<Vec<_>>()
        .join(",")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reasons_name_the_most_serious_first() {
        assert_eq!(Reasons(0).label(), None);
        assert_eq!(Reasons(Reasons::IDLE).label(), Some("idle"));
        assert_eq!(Reasons(Reasons::POWER_CAP).label(), Some("power cap"));
        // What an idle RTX PRO 6000 reports on driver 615.
        assert_eq!(Reasons(0x400).label(), Some("reliability"));
        assert_eq!(Reasons(0x200).label(), None);
        let hot =
            Reasons(Reasons::POWER_CAP | Reasons::SOFTWARE_THERMAL | Reasons::HARDWARE_THERMAL);
        assert_eq!(hot.label(), Some("hardware thermal slowdown"));
        assert!(hot.thermal() && hot.slowdown());
        // A power cap alone is not a slowdown: it is normal for a capped card.
        assert!(!Reasons(Reasons::POWER_CAP).slowdown());
        assert!(Reasons(Reasons::POWER_BRAKE).slowdown());
        assert!(!Reasons(Reasons::POWER_BRAKE).thermal());
    }

    #[test]
    fn names_drop_the_vendor_and_edition() {
        let device = |name: &str| Device {
            name: name.to_owned(),
            ..Device::default()
        };
        assert_eq!(
            device("NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition").short_name(),
            "RTX PRO 6000 Max-Q"
        );
        assert_eq!(
            device("NVIDIA H100 80GB HBM3").short_name(),
            "H100 80GB HBM3"
        );
    }

    #[test]
    fn compute_and_graphics_lists_merge_by_pid_without_double_counting() {
        let merged = merge(
            &[(10, Some(5 << 30)), (11, None)],
            &[(10, Some(4 << 30)), (12, Some(96 << 20))],
        );
        let rows = merged
            .iter()
            .map(|process| (process.pid, process.memory, process.kind.label()))
            .collect::<Vec<_>>();
        assert_eq!(
            rows,
            [
                (10, Some(5 << 30), "C+G"),
                (12, Some(96 << 20), "G"),
                (11, None, "C"),
            ]
        );
    }

    #[test]
    fn the_kernel_share_prefers_gpm() {
        let mut reading = Reading {
            utilization: Some((0.97, 0.7)),
            ..Reading::default()
        };
        assert_eq!(reading.kernels(), Some(0.97));
        reading.activity = Some(Activity {
            kernels: Some(0.86),
            ..Activity::default()
        });
        assert_eq!(reading.kernels(), Some(0.86));
        // GPM without the graphics metric falls back to NVML.
        reading.activity = Some(Activity::default());
        assert_eq!(reading.kernels(), Some(0.97));
    }

    #[test]
    fn indices_print_as_runs() {
        assert_eq!(ranges(&[0, 1, 2, 3]), "0-3");
        assert_eq!(ranges(&[0, 2]), "0,2");
        assert_eq!(ranges(&[0, 1, 4]), "0-1,4");
        assert_eq!(ranges(&[]), "");
    }

    #[test]
    fn reports_open_poll_and_find_an_instances_gpus() {
        use crate::targets::Target;
        let mut registry = crate::instances::Registry::new(1.0);
        let row = registry.add_target(Target::parse(":8311").unwrap(), 0.0);
        let mut gpus = Gpus::new(Status::Opening, true, 1.0);
        let device = |index| Device {
            index,
            name: "NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition".into(),
            ..Device::default()
        };
        gpus.apply(
            Report::Opened {
                driver: Some("615.71.09".into()),
                devices: vec![device(0), device(1), device(2)],
            },
            &mut registry,
            0.0,
        );
        assert_eq!(gpus.status, Status::Open);
        let serving = |pid| Reading {
            processes: Some(vec![Process {
                pid,
                owner: Owner {
                    listen: Some("0.0.0.0:8311".into()),
                    ..Owner::default()
                },
                ..Process::default()
            }]),
            ..Reading::default()
        };
        gpus.apply(
            Report::Polled(vec![serving(7), Reading::default(), serving(7)]),
            &mut registry,
            1.0,
        );
        assert_eq!(gpus.of_instance(&registry.instances, row), [0, 2]);
        assert_eq!(gpus.processes(1, &registry.instances), []);
        gpus.apply(Report::Unavailable("gone".into()), &mut registry, 2.0);
        assert_eq!(gpus.status, Status::Unavailable("gone".into()));
    }

    #[test]
    fn each_change_of_a_gpu_is_one_event() {
        use crate::targets::Target;
        let mut registry = Registry::new(1.0);
        registry.add_target(Target::parse(":8311").unwrap(), 0.0);
        let mut gpus = Gpus::new(Status::Opening, true, 1.0);
        let device = Device {
            index: 2,
            sm_max_mhz: Some(3090),
            ..Device::default()
        };
        gpus.apply(
            Report::Opened {
                driver: Some("615.71.09".into()),
                devices: vec![device],
            },
            &mut registry,
            0.0,
        );
        let process = |pid, listen: Option<&str>, name: &str| Process {
            pid,
            memory: Some(3 << 29),
            owner: Owner {
                name: Some(name.into()),
                listen: listen.map(str::to_owned),
                ..Owner::default()
            },
            ..Process::default()
        };
        let reading = |reasons, headroom, ecc, lost, stranger: bool| {
            let mut processes = vec![process(7, Some("0.0.0.0:8311"), "chopinfer")];
            if stranger {
                processes.push(process(9, None, "python3"));
            }
            Reading {
                temperature: Some(92 - headroom),
                headroom: Some(headroom),
                sm_mhz: Some(1650),
                reasons: Some(Reasons(reasons)),
                ecc_uncorrected: Some(ecc),
                processes: Some(processes),
                lost,
                answered: !lost,
                ..Reading::default()
            }
        };
        let thermal = Reasons::HARDWARE_THERMAL;
        for (at, reading) in [
            (1.0, reading(Reasons::POWER_CAP, 20, 0, false, false)),
            (2.0, reading(thermal, 3, 0, false, true)),
            // Nothing new: no event.
            (3.0, reading(thermal, 2, 0, false, true)),
            (4.0, reading(Reasons::POWER_CAP, 8, 1, false, true)),
            // Still under the re-arming headroom: no second warning.
            (5.0, reading(Reasons::POWER_CAP, 4, 1, false, false)),
            (6.0, reading(0, 0, 0, true, false)),
            (7.0, reading(Reasons::POWER_CAP, 20, 1, false, false)),
        ] {
            gpus.apply(Report::Polled(vec![reading]), &mut registry, at);
        }
        let events = registry
            .events
            .iter()
            .map(|event| {
                (
                    event.at,
                    event.who.as_str(),
                    event.what.as_str(),
                    event.tone,
                )
            })
            .collect::<Vec<_>>();
        assert_eq!(
            events,
            [
                (0.0, "GPUs", "1 found, driver 615.71.09", Tone::Normal),
                (
                    2.0,
                    "GPU 2",
                    "hardware thermal slowdown: SM clock 1,650 of 3,090 MHz, 3C under the limit",
                    Tone::Warning
                ),
                (
                    2.0,
                    "GPU 2",
                    "3C under the temperature limit at 89C",
                    Tone::Warning
                ),
                (
                    2.0,
                    "GPU 2",
                    "pid 9 (python3) holds 1.5 GiB beside target 127.0.0.1:8311",
                    Tone::Warning
                ),
                (
                    4.0,
                    "GPU 2",
                    "slowdown over: SM clock 1,650 MHz",
                    Tone::Normal
                ),
                (
                    4.0,
                    "GPU 2",
                    "1 uncorrected ECC error since the driver loaded",
                    Tone::Bad
                ),
                (
                    6.0,
                    "GPU 2",
                    "stopped answering NVML: lost or off the bus",
                    Tone::Bad
                ),
                (7.0, "GPU 2", "answering NVML again", Tone::Good),
            ]
        );
    }

    #[test]
    fn a_failed_query_or_a_cleared_counter_keeps_events_true() {
        let mut registry = Registry::new(1.0);
        let mut gpus = Gpus::new(Status::Opening, true, 1.0);
        gpus.apply(
            Report::Opened {
                driver: None,
                devices: vec![Device::default()],
            },
            &mut registry,
            0.0,
        );
        let thermal = Some(Reasons(Reasons::SOFTWARE_THERMAL));
        let reading = |reasons, ecc| Reading {
            reasons,
            ecc_uncorrected: ecc,
            answered: true,
            ..Reading::default()
        };
        for (at, reading) in [
            (1.0, reading(thermal, Some(4))),
            // The reasons query failed: still slowing down, no second start.
            (2.0, reading(None, None)),
            (3.0, reading(thermal, Some(4))),
            // The counter was cleared, then one more error.
            (4.0, reading(thermal, Some(0))),
            (5.0, reading(thermal, Some(1))),
        ] {
            gpus.apply(Report::Polled(vec![reading]), &mut registry, at);
        }
        let events = registry
            .events
            .iter()
            .skip(1)
            .map(|event| (event.at, event.what.as_str()))
            .collect::<Vec<_>>();
        assert_eq!(
            events,
            [
                (1.0, "thermal slowdown"),
                (1.0, "4 uncorrected ECC errors since the driver loaded"),
                (5.0, "1 uncorrected ECC error since the driver loaded"),
            ]
        );
    }

    #[test]
    fn a_lost_gpu_is_back_only_when_a_query_answers() {
        let mut registry = Registry::new(1.0);
        let mut gpus = Gpus::new(Status::Opening, true, 1.0);
        gpus.apply(
            Report::Opened {
                driver: None,
                devices: vec![Device::default()],
            },
            &mut registry,
            0.0,
        );
        let reading = |lost, answered| Reading {
            lost,
            answered,
            ..Reading::default()
        };
        for (at, reading, lost) in [
            (1.0, reading(true, false), true),
            // Every query failed, but not as lost: no news, still lost.
            (2.0, reading(false, false), true),
            (3.0, reading(false, true), false),
        ] {
            gpus.apply(Report::Polled(vec![reading]), &mut registry, at);
            assert_eq!(gpus.lost(0), lost, "at {at}");
        }
        let events = registry
            .events
            .iter()
            .skip(1)
            .map(|event| (event.at, event.what.as_str()))
            .collect::<Vec<_>>();
        assert_eq!(
            events,
            [
                (1.0, "stopped answering NVML: lost or off the bus"),
                (3.0, "answering NVML again"),
            ]
        );
    }

    #[test]
    fn a_failed_process_query_keeps_the_list_and_its_strangers() {
        use crate::targets::Target;
        let mut registry = Registry::new(1.0);
        registry.add_target(Target::parse(":8311").unwrap(), 0.0);
        let mut gpus = Gpus::new(Status::Opening, true, 1.0);
        gpus.apply(
            Report::Opened {
                driver: None,
                devices: vec![Device::default()],
            },
            &mut registry,
            0.0,
        );
        let process = |pid, listen: Option<&str>| Process {
            pid,
            owner: Owner {
                name: Some("python3".into()),
                listen: listen.map(str::to_owned),
                ..Owner::default()
            },
            ..Process::default()
        };
        let read = |at| Reading {
            at,
            processes: Some(vec![process(7, Some("0.0.0.0:8311")), process(9, None)]),
            answered: true,
            ..Reading::default()
        };
        // The other queries answered; the process lists did not.
        let failed = Reading {
            at: 2.0,
            answered: true,
            ..Reading::default()
        };
        for (at, reading, listed) in [
            (1.0, read(1.0), 1.0),
            (2.0, failed, 1.0),
            (3.0, read(3.0), 3.0),
        ] {
            gpus.apply(Report::Polled(vec![reading]), &mut registry, at);
            let pids = gpus
                .processes(0, &registry.instances)
                .iter()
                .map(|(process, _)| process.pid)
                .collect::<Vec<_>>();
            assert_eq!(pids, [7, 9], "at {at}");
            assert_eq!(gpus.processes_read(0), Some(listed), "at {at}");
        }
        let strangers = registry
            .events
            .iter()
            .filter(|event| event.what.starts_with("pid 9"))
            .count();
        assert_eq!(strangers, 1, "{:?}", registry.events);
    }

    #[test]
    fn a_gpu_this_user_may_not_open_is_not_lost() {
        let mut registry = Registry::new(1.0);
        let mut gpus = Gpus::new(Status::Opening, true, 1.0);
        gpus.apply(
            Report::Opened {
                driver: None,
                devices: vec![Device {
                    unreadable: Some("no permission".into()),
                    ..Device::default()
                }],
            },
            &mut registry,
            0.0,
        );
        // What the poller reads for it: no handle, nothing answered.
        gpus.apply(Report::Polled(vec![Reading::default()]), &mut registry, 1.0);
        assert!(!gpus.lost(0));
        assert_eq!(gpus.processes_read(0), None);
        assert_eq!(registry.events.len(), 1, "{:?}", registry.events);
    }

    #[test]
    fn history_averages_the_last_three_polls_and_keeps_five_minutes() {
        let mut history = History::new(1.0);
        assert_eq!(history.capacity(), 301);
        for poll in 0..400 {
            history.push(Reading {
                at: f64::from(poll),
                power: (poll % 2 == 0).then_some(f64::from(poll)),
                ..Reading::default()
            });
        }
        assert_eq!(history.oldest_poll(), 99);
        assert_eq!(history.series(|r| r.power).len(), 301);
        // Polls 399, 398 and 397: only 398 has a power figure.
        assert_eq!(history.recent(|r| r.power), Some(398.0));
        assert_eq!(history.recent(|r| r.fan.map(f64::from)), None);
    }
}
