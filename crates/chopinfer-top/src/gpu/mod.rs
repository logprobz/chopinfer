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

use crate::stats::{HISTORY_SECONDS, RATE_POLLS};
pub use owners::Owner;

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
    pub processes: Vec<Process>,
    /// GPM figures since the previous poll, while they are read.
    pub activity: Option<Activity>,
    /// The GPU stopped answering: it fell off the bus or is otherwise lost.
    pub lost: bool,
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
