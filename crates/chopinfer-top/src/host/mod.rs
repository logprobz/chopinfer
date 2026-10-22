//! This host: CPU, memory, load, pressure, disk and network I/O, disk
//! space and the hwmon sensors, read from `/proc` and `/sys`.
//!
//! Inside a container `/proc/stat`, `/proc/meminfo`, `/proc/diskstats` and
//! hwmon describe the host, not the container's limits, while network
//! counters and mounts belong to the container's own namespaces.

pub mod hwmon;
pub mod poll;
pub mod proc;
pub mod space;

use std::collections::{HashSet, VecDeque};

use crate::gpu::poll::SLOW_POLLS;
use crate::instances::{Registry, Tone};
use crate::stats::{HISTORY_SECONDS, RATE_POLLS};
use crate::ui::widgets::{bytes, integer, percent};
use poll::Report;
pub use space::Filesystem;

const MIB: u64 = 1 << 20;
/// Memory in use that is reported, and the share that ends the report.
const MEMORY_HIGH: f64 = 0.95;
const MEMORY_EASED: f64 = 0.90;
/// Swap in use that is reported, and the amount that ends the report.
const SWAP_IN_USE: u64 = 64 * MIB;
const SWAP_EASED: u64 = 32 * MIB;
/// A filesystem's share in use that is reported, and the share that ends
/// the report.
const FULL: f64 = 0.90;
const FULL_EASED: f64 = 0.85;
/// Degrees under a sensor's warning level that end a warning.
const COOLED: f64 = 5.0;
/// Rounds of disk-space reads after which the sizes shown are marked old.
const SPACE_ROUNDS: f64 = 3.0;

/// The share of CPU time in each state since the previous poll.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Cpu {
    /// Everything but idle and iowait.
    pub busy: f64,
    /// User and nice time.
    pub user: f64,
    /// System, interrupt and softirq time.
    pub system: f64,
    pub iowait: f64,
}

/// `/proc/meminfo`, in bytes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Memory {
    pub total: u64,
    /// What the kernel can give to new work without swapping.
    pub available: u64,
    pub cached: u64,
    pub swap_total: u64,
    pub swap_free: u64,
}

impl Memory {
    pub fn used(&self) -> u64 {
        self.total.saturating_sub(self.available)
    }

    pub fn share(&self) -> Option<f64> {
        (self.total > 0).then(|| self.used() as f64 / self.total as f64)
    }

    pub fn swap_used(&self) -> u64 {
        self.swap_total.saturating_sub(self.swap_free)
    }
}

/// `/proc/loadavg`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Load {
    pub one: f64,
    pub five: f64,
    pub fifteen: f64,
    /// Tasks that exist, running or not.
    pub tasks: Option<u64>,
}

/// The share of the last ten seconds that some task stalled, from
/// `/proc/pressure`; `None` where the kernel has no pressure files.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Pressure {
    pub cpu: Option<f64>,
    pub memory: Option<f64>,
    pub io: Option<f64>,
}

/// Bytes a second each way: read and written, or received and sent.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Rates {
    pub incoming: f64,
    pub outgoing: f64,
}

/// One poll of this host; `None` where a file could not be read.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Reading {
    /// Seconds on chopinfer-top's clock.
    pub at: f64,
    /// Since the previous poll; `None` on the first.
    pub cpu: Option<Cpu>,
    /// The busy share of each CPU since the previous poll, in CPU order.
    pub cores: Vec<f64>,
    pub memory: Option<Memory>,
    pub load: Option<Load>,
    pub pressure: Pressure,
    /// Whole disks, each counted once.
    pub disk: Option<Rates>,
    /// Interfaces with a device, in chopinfer-top's network namespace.
    pub network: Option<Rates>,
    /// Whether any interface with a device is visible here: inside a
    /// bridged container none is, and the network is unknown, not idle.
    pub network_visible: bool,
    /// The CPU packages' power from RAPL, in watts.
    pub package: Option<f64>,
    /// Seconds since boot.
    pub uptime: Option<f64>,
}

/// A value of a reading, for averages and plots.
pub type Field = fn(&Reading) -> Option<f64>;

/// What a chip is, for grouping and order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Kind {
    Cpu,
    Nvme,
    Dimm,
    Nic,
    Board,
    Other,
}

/// What a sensor measures.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Quantity {
    Celsius,
    Rpm,
    Watts,
    Volts,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Sensor {
    /// The channel's file stem, such as `temp1` or `fan2`.
    pub channel: String,
    pub label: String,
    pub quantity: Quantity,
    /// `None` when its read failed.
    pub value: Option<f64>,
    /// The chip's critical temperature, else its maximum.
    pub limit: Option<f64>,
    /// The chip's main temperature: Tdie, Tctl, Composite or a package.
    pub primary: bool,
}

/// One hwmon chip and its sensors.
#[derive(Clone, Debug, PartialEq)]
pub struct Chip {
    /// The hwmon directory, such as `hwmon3`.
    pub id: String,
    pub kind: Kind,
    /// The kernel driver's name, such as `k10temp` or `nvme`.
    pub driver: String,
    /// The device's model, where it says.
    pub model: Option<String>,
    /// How it is shown: `CPU`, `NVMe 1`, `DIMM 3`, an interface or the driver.
    pub name: String,
    pub sensors: Vec<Sensor>,
}

impl Chip {
    /// How sensor `sensor` is named on its own, in events and lines.
    pub fn sensor_name(&self, sensor: &Sensor) -> String {
        let temperatures = self
            .sensors
            .iter()
            .filter(|sensor| sensor.quantity == Quantity::Celsius)
            .count();
        let generic = sensor.label == sensor.channel;
        if matches!(self.kind, Kind::Dimm | Kind::Nic) && temperatures == 1 && generic {
            self.name.clone()
        } else {
            format!("{} {}", self.name, sensor.label)
        }
    }
}

/// How warm a temperature is against its limit.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Level {
    Normal,
    Warm,
    Hot,
}

/// Where a sensor without a limit of its own turns warm and hot.
pub const WARM_CELSIUS: f64 = 85.0;
pub const HOT_CELSIUS: f64 = 95.0;
/// Degrees under a chip's limit where it turns warm.
pub const WARM_HEADROOM: f64 = 10.0;
/// Degrees under its limit where a CPU's temperature is reported. CPUs
/// boost until they near their limit, so their warm level is normal load.
pub const CPU_HEADROOM: f64 = 3.0;

/// Warm 10C under the chip's limit and hot at it; without a limit, warm
/// from 85C and hot from 95C.
pub fn level(celsius: f64, limit: Option<f64>) -> Level {
    let (warm, hot) = warm_hot(limit);
    if celsius >= hot {
        Level::Hot
    } else if celsius >= warm {
        Level::Warm
    } else {
        Level::Normal
    }
}

fn warm_hot(limit: Option<f64>) -> (f64, f64) {
    limit.map_or((WARM_CELSIUS, HOT_CELSIUS), |limit| {
        (limit - WARM_HEADROOM, limit)
    })
}

/// Where a temperature on a chip of `kind` is reported: 3C under the
/// limit for a CPU, 95C without one; for any other chip, where it turns
/// warm.
pub fn alert(kind: Kind, limit: Option<f64>) -> f64 {
    let (warm, hot) = warm_hot(limit);
    if kind == Kind::Cpu {
        hot - CPU_HEADROOM
    } else {
        warm
    }
}

/// How far `celsius` is from `limit`, named `what`, in whole degrees.
fn margin(celsius: f64, limit: f64, what: &str) -> String {
    let under = (limit - celsius).round();
    if under > 0.0 {
        format!("{under:.0}C under {what}")
    } else if under < 0.0 {
        format!("{:.0}C over {what}", -under)
    } else {
        format!("at {what}")
    }
}

/// How full a filesystem is: warm from 90%, hot from 95%.
pub fn fullness(share: f64) -> Level {
    if share >= 0.95 {
        Level::Hot
    } else if share >= FULL {
        Level::Warm
    } else {
        Level::Normal
    }
}

/// This host's recent readings: five minutes of polls, as an instance keeps.
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

    /// The average of `field` over the last three polls that have it.
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

    /// Each CPU's busy share averaged over the last three polls.
    pub fn cores(&self) -> Vec<f64> {
        let recent = self
            .ring
            .iter()
            .rev()
            .take(RATE_POLLS)
            .filter(|reading| !reading.cores.is_empty())
            .collect::<Vec<_>>();
        let count = recent
            .iter()
            .map(|reading| reading.cores.len())
            .min()
            .unwrap_or(0);
        (0..count)
            .map(|core| {
                recent
                    .iter()
                    .map(|reading| reading.cores[core])
                    .sum::<f64>()
                    / recent.len() as f64
            })
            .collect()
    }
}

/// Whether this host can be read.
#[derive(Clone, Debug, PartialEq)]
pub enum Status {
    /// The first reading is being taken.
    Opening,
    /// `/proc/stat` could not be read, and why.
    Unavailable(String),
    Open,
}

/// Whether the CPU packages' power can be read.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Rapl {
    /// No RAPL counters on this host.
    #[default]
    Absent,
    /// The counters are readable by root only.
    NeedsRoot,
    Readable,
}

/// What has been said about the host, so that each change is one event.
#[derive(Clone, Debug, Default)]
struct Watch {
    /// Sensors at their warning level, by `chip/channel`.
    warm: HashSet<String>,
    memory: bool,
    swap: bool,
    /// Fans that have read above 0 rpm since start, and those of them that
    /// read 0 now.
    spun: HashSet<String>,
    stopped: HashSet<String>,
    /// Filesystems at least 90% full, by mount point.
    full: HashSet<String>,
}

/// What the interface knows about this host.
#[derive(Clone, Debug)]
pub struct Host {
    pub status: Status,
    pub model: Option<String>,
    pub threads: usize,
    pub history: History,
    /// The latest sensor values.
    pub chips: Vec<Chip>,
    /// Fan channels exposed, whatever they read.
    pub fan_channels: usize,
    pub rapl: Rapl,
    /// The latest disk-space reading, and when it was taken.
    pub filesystems: Vec<Filesystem>,
    pub space_at: Option<f64>,
    interval: f64,
    watch: Watch,
}

fn key(chip: &Chip, sensor: &Sensor) -> String {
    format!("{}/{}", chip.id, sensor.channel)
}

impl Host {
    pub fn new(interval: f64) -> Self {
        Self {
            status: Status::Opening,
            model: None,
            threads: 0,
            history: History::new(interval),
            chips: Vec::new(),
            fan_channels: 0,
            rapl: Rapl::Absent,
            filesystems: Vec::new(),
            space_at: None,
            interval,
            watch: Watch::default(),
        }
    }

    pub fn latest(&self) -> Option<&Reading> {
        self.history.latest()
    }

    pub fn recent(&self, field: Field) -> Option<f64> {
        self.history.recent(field)
    }

    pub fn sensors(&self) -> usize {
        self.chips.iter().map(|chip| chip.sensors.len()).sum()
    }

    /// Whether the fan of `chip` is shown: it has read above 0 rpm since
    /// start, so a header with nothing on it stays hidden.
    pub fn fan_shown(&self, chip: &Chip, sensor: &Sensor) -> bool {
        self.watch.spun.contains(&key(chip, sensor))
    }

    /// The fans shown, with their chips.
    pub fn fans(&self) -> Vec<(&Chip, &Sensor)> {
        self.chips
            .iter()
            .flat_map(|chip| chip.sensors.iter().map(move |sensor| (chip, sensor)))
            .filter(|(chip, sensor)| {
                sensor.quantity == Quantity::Rpm && self.fan_shown(chip, sensor)
            })
            .collect()
    }

    /// Seconds since the disk sizes were read, when that is more than three
    /// rounds of reads ago: a read is stuck.
    pub fn space_age(&self, now: f64) -> Option<f64> {
        let age = now - self.space_at?;
        (age > SPACE_ROUNDS * SLOW_POLLS as f64 * self.interval).then_some(age)
    }

    /// Takes a report at `now`, noting in `registry` what changed: the
    /// host found, a chip's main temperature reaching its alert level and
    /// leaving it, memory nearly full, swap in use, a fan stopping and
    /// starting, and a filesystem nearly full.
    pub fn apply(&mut self, report: Report, registry: &mut Registry, now: f64) {
        match report {
            Report::Opened {
                model,
                threads,
                memory_total,
                sensors,
                chips,
                fans,
                rapl,
            } => {
                let mut what = format!("{threads} CPU{}", if threads == 1 { "" } else { "s" });
                if let Some(total) = memory_total {
                    what.push_str(&format!(", {}", bytes(total as f64)));
                }
                if chips == 0 {
                    what.push_str("; no sensors in /sys/class/hwmon");
                } else {
                    what.push_str(&format!("; {sensors} sensors on {chips} chips"));
                    if fans == 0 {
                        what.push_str("; no fan sensors exposed");
                    }
                }
                registry.note(now, "host", what, Tone::Normal);
                self.status = Status::Open;
                self.model = model;
                self.threads = threads;
                self.rapl = rapl;
                self.fan_channels = fans;
            }
            Report::Unavailable(reason) => {
                registry.note(now, "host", format!("not read: {reason}"), Tone::Normal);
                self.status = Status::Unavailable(reason);
            }
            Report::Polled { reading, chips } => {
                self.watch_reading(&reading, registry, now);
                self.watch_chips(&chips, registry, now);
                self.fan_channels = chips
                    .iter()
                    .flat_map(|chip| &chip.sensors)
                    .filter(|sensor| sensor.quantity == Quantity::Rpm)
                    .count();
                self.chips = chips;
                self.history.push(*reading);
            }
            Report::Space { at, filesystems } => {
                self.watch_space(&filesystems, registry, now);
                self.filesystems = filesystems;
                self.space_at = Some(at);
            }
        }
    }

    fn watch_reading(&mut self, reading: &Reading, registry: &mut Registry, now: f64) {
        let Some(memory) = reading.memory else {
            return;
        };
        if let Some(share) = memory.share() {
            if !self.watch.memory && share > MEMORY_HIGH {
                self.watch.memory = true;
                registry.note(
                    now,
                    "host",
                    format!(
                        "memory {} in use, {} available",
                        percent(Some(share)),
                        bytes(memory.available as f64)
                    ),
                    Tone::Warning,
                );
            } else if self.watch.memory && share < MEMORY_EASED {
                self.watch.memory = false;
                registry.note(
                    now,
                    "host",
                    format!("memory back to {} in use", percent(Some(share))),
                    Tone::Normal,
                );
            }
        }
        let swap = memory.swap_used();
        if !self.watch.swap && swap > SWAP_IN_USE {
            self.watch.swap = true;
            registry.note(
                now,
                "host",
                format!("swap in use: {}", bytes(swap as f64)),
                Tone::Warning,
            );
        } else if self.watch.swap && swap < SWAP_EASED {
            self.watch.swap = false;
            registry.note(now, "host", "swap back under 32 MiB", Tone::Normal);
        }
    }

    fn watch_chips(&mut self, chips: &[Chip], registry: &mut Registry, now: f64) {
        for chip in chips {
            for sensor in &chip.sensors {
                let Some(value) = sensor.value else {
                    continue;
                };
                let key = key(chip, sensor);
                let name = chip.sensor_name(sensor);
                match sensor.quantity {
                    // A chip's main temperature speaks for it: Tctl is the
                    // hottest CCD's, Composite the drive's.
                    Quantity::Celsius if sensor.primary => {
                        let alert = alert(chip.kind, sensor.limit);
                        if !self.watch.warm.contains(&key) && value >= alert {
                            self.watch.warm.insert(key);
                            let what = match sensor.limit {
                                Some(limit) => {
                                    format!(
                                        "{name} {value:.0}C, {}",
                                        margin(value, limit, "its limit")
                                    )
                                }
                                None if chip.kind == Kind::Cpu => format!(
                                    "{name} {value:.0}C, {} (no limit exposed)",
                                    margin(value, HOT_CELSIUS, &format!("{HOT_CELSIUS:.0}C"))
                                ),
                                None => format!(
                                    "{name} {value:.0}C, over {WARM_CELSIUS:.0}C (no limit exposed)"
                                ),
                            };
                            registry.note(now, "host", what, Tone::Warning);
                        } else if self.watch.warm.contains(&key)
                            // Rounded as shown, so the event never names
                            // the level it went under.
                            && value.round() < alert - COOLED
                        {
                            self.watch.warm.remove(&key);
                            registry.note(
                                now,
                                "host",
                                format!("{name} back to {value:.0}C"),
                                Tone::Normal,
                            );
                        }
                    }
                    Quantity::Rpm => {
                        if value > 0.0 {
                            self.watch.spun.insert(key.clone());
                            if self.watch.stopped.remove(&key) {
                                registry.note(
                                    now,
                                    "host",
                                    format!("{name} spinning again: {} rpm", integer(value)),
                                    Tone::Good,
                                );
                            }
                        } else if self.watch.spun.contains(&key) && self.watch.stopped.insert(key) {
                            registry.note(
                                now,
                                "host",
                                format!("{name} stopped: 0 rpm"),
                                Tone::Warning,
                            );
                        }
                    }
                    Quantity::Celsius | Quantity::Watts | Quantity::Volts => {}
                }
            }
        }
    }

    fn watch_space(&mut self, filesystems: &[Filesystem], registry: &mut Registry, now: f64) {
        for filesystem in filesystems {
            let share = filesystem.share();
            let full = self.watch.full.contains(&filesystem.mount);
            if !full && share >= FULL {
                self.watch.full.insert(filesystem.mount.clone());
                registry.note(
                    now,
                    "host",
                    format!(
                        "{} {}% full, {} free",
                        filesystem.mount,
                        filesystem.percent(),
                        bytes(filesystem.available as f64)
                    ),
                    Tone::Warning,
                );
            } else if full && share < FULL_EASED {
                self.watch.full.remove(&filesystem.mount);
                registry.note(
                    now,
                    "host",
                    format!(
                        "{} back to {}% full",
                        filesystem.mount,
                        filesystem.percent()
                    ),
                    Tone::Normal,
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temperature(channel: &str, label: &str, value: f64, limit: Option<f64>) -> Sensor {
        Sensor {
            channel: channel.to_owned(),
            label: label.to_owned(),
            quantity: Quantity::Celsius,
            value: Some(value),
            limit,
            primary: true,
        }
    }

    fn fan(value: f64) -> Sensor {
        Sensor {
            channel: "fan2".to_owned(),
            label: "fan2".to_owned(),
            quantity: Quantity::Rpm,
            value: Some(value),
            limit: None,
            primary: false,
        }
    }

    fn chip(kind: Kind, name: &str, sensors: Vec<Sensor>) -> Chip {
        Chip {
            id: format!("hwmon-{name}"),
            kind,
            driver: "driver".to_owned(),
            model: None,
            name: name.to_owned(),
            sensors,
        }
    }

    fn polled(at: f64, memory: Memory, chips: Vec<Chip>) -> Report {
        Report::Polled {
            reading: Box::new(Reading {
                at,
                memory: Some(memory),
                ..Reading::default()
            }),
            chips,
        }
    }

    fn whats(registry: &Registry) -> Vec<(String, Tone)> {
        registry
            .events
            .iter()
            .filter(|event| event.who == "host")
            .map(|event| (event.what.clone(), event.tone))
            .collect()
    }

    #[test]
    fn levels_use_the_chip_limit_else_85_and_95() {
        assert_eq!(level(74.0, Some(85.0)), Level::Normal);
        assert_eq!(level(75.0, Some(85.0)), Level::Warm);
        assert_eq!(level(85.0, Some(85.0)), Level::Hot);
        assert_eq!(level(84.0, None), Level::Normal);
        assert_eq!(level(93.0, None), Level::Warm);
        assert_eq!(level(95.0, None), Level::Hot);
        // A CPU is reported only near its limit.
        assert_eq!(alert(Kind::Cpu, None), 92.0);
        assert_eq!(alert(Kind::Cpu, Some(100.0)), 97.0);
        assert_eq!(alert(Kind::Nvme, Some(85.0)), 75.0);
        assert_eq!(alert(Kind::Nvme, None), 85.0);
        assert_eq!(fullness(0.89), Level::Normal);
        assert_eq!(fullness(0.90), Level::Warm);
        assert_eq!(fullness(0.97), Level::Hot);
    }

    #[test]
    fn each_change_is_one_event_with_hysteresis() {
        let mut registry = Registry::new(1.0);
        let mut host = Host::new(1.0);
        host.apply(
            Report::Opened {
                model: None,
                threads: 48,
                memory_total: Some(251 << 30),
                sensors: 3,
                chips: 2,
                fans: 0,
                rapl: Rapl::Absent,
            },
            &mut registry,
            0.0,
        );
        let gib = 1 << 30;
        let calm = Memory {
            total: 100 * gib,
            available: 40 * gib,
            swap_total: 8 * gib,
            swap_free: 8 * gib,
            ..Memory::default()
        };
        let pressed = Memory {
            available: 4 * gib,
            swap_free: 6 * gib,
            ..calm
        };
        let eased = Memory {
            available: 12 * gib,
            swap_free: 8 * gib - 40 * MIB,
            ..calm
        };
        // Tccd1 is as hot, but Tctl speaks for the chip.
        let cpu = |celsius| {
            let mut ccd = temperature("temp3", "Tccd1", celsius, None);
            ccd.primary = false;
            chip(
                Kind::Cpu,
                "CPU",
                vec![temperature("temp1", "Tctl", celsius, None), ccd],
            )
        };
        let nvme = |celsius| {
            chip(
                Kind::Nvme,
                "NVMe 0",
                vec![temperature("temp1", "Composite", celsius, Some(85.0))],
            )
        };
        let board = |rpm| chip(Kind::Board, "nct6799", vec![fan(rpm)]);
        let steps = [
            // A CPU at 88C, warm but boosting as designed, is not reported.
            (calm, 88.0, 40.0, 0.0),
            (pressed, 93.0, 77.0, 0.0),
            (pressed, 94.0, 78.0, 1200.0),
            // Under the alert level, but not by the 5C that ends it: 86.6C
            // shows as 87C, which is not under 92 - 5.
            (eased, 86.6, 72.0, 0.0),
            (eased, 86.0, 69.0, 0.0),
            (calm, 70.0, 60.0, 1180.0),
        ];
        for (second, (memory, celsius, drive, rpm)) in steps.into_iter().enumerate() {
            host.apply(
                polled(
                    second as f64,
                    memory,
                    vec![cpu(celsius), nvme(drive), board(rpm)],
                ),
                &mut registry,
                second as f64,
            );
        }
        let events = whats(&registry);
        assert_eq!(
            events,
            [
                (
                    "48 CPUs, 251 GiB; 3 sensors on 2 chips; no fan sensors exposed".to_owned(),
                    Tone::Normal
                ),
                (
                    "memory 96% in use, 4.0 GiB available".to_owned(),
                    Tone::Warning
                ),
                ("swap in use: 2.0 GiB".to_owned(), Tone::Warning),
                (
                    "CPU Tctl 93C, 2C under 95C (no limit exposed)".to_owned(),
                    Tone::Warning
                ),
                (
                    "NVMe 0 Composite 77C, 8C under its limit".to_owned(),
                    Tone::Warning
                ),
                // 40 MiB of swap is under 64 but not under 32: still in use.
                ("memory back to 88% in use".to_owned(), Tone::Normal),
                ("nct6799 fan2 stopped: 0 rpm".to_owned(), Tone::Warning),
                ("CPU Tctl back to 86C".to_owned(), Tone::Normal),
                ("NVMe 0 Composite back to 69C".to_owned(), Tone::Normal),
                ("swap back under 32 MiB".to_owned(), Tone::Normal),
                (
                    "nct6799 fan2 spinning again: 1,180 rpm".to_owned(),
                    Tone::Good
                ),
            ]
        );
        // A fan never seen spinning is not shown and not reported.
        assert_eq!(host.fans().len(), 1);
    }

    #[test]
    fn a_full_filesystem_is_reported_until_it_drops_under_85_percent() {
        let mut registry = Registry::new(1.0);
        let mut host = Host::new(1.0);
        let data = |used: u64| Filesystem {
            mount: "/data".to_owned(),
            used,
            available: 100 - used,
            total: 100,
        };
        for (at, used) in [
            (0.0, 80),
            (5.0, 97),
            (10.0, 96),
            (15.0, 87),
            (20.0, 84),
            (25.0, 91),
        ] {
            host.apply(
                Report::Space {
                    at,
                    filesystems: vec![data(used)],
                },
                &mut registry,
                at,
            );
        }
        assert_eq!(
            whats(&registry),
            [
                ("/data 97% full, 3 B free".to_owned(), Tone::Warning),
                ("/data back to 84% full".to_owned(), Tone::Normal),
                ("/data 91% full, 9 B free".to_owned(), Tone::Warning),
            ]
        );
    }

    #[test]
    fn sizes_older_than_three_rounds_are_marked() {
        let mut registry = Registry::new(1.0);
        let mut host = Host::new(1.0);
        assert_eq!(host.space_age(100.0), None);
        host.apply(
            Report::Space {
                at: 10.0,
                filesystems: Vec::new(),
            },
            &mut registry,
            10.0,
        );
        // Rounds are five polls of a second.
        assert_eq!(host.space_age(25.0), None);
        assert_eq!(host.space_age(26.0), Some(16.0));
    }

    #[test]
    fn cores_average_the_last_three_polls() {
        let mut history = History::new(1.0);
        for (at, cores) in [
            (0.0, vec![0.0, 0.0]),
            (1.0, vec![0.3, 0.9]),
            (2.0, vec![0.6, 0.6]),
            (3.0, vec![0.9, 0.3]),
        ] {
            history.push(Reading {
                at,
                cores,
                ..Reading::default()
            });
        }
        let cores = history.cores();
        assert!(
            (cores[0] - 0.6).abs() < 1e-9 && (cores[1] - 0.6).abs() < 1e-9,
            "{cores:?}"
        );
    }
}
