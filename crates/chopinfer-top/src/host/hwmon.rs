//! The sensors of `/sys/class/hwmon`: temperatures, fans, power and
//! labelled voltages, by chip, and the CPU packages' RAPL energy counters.
//!
//! Discovery records the channels each chip exposes, apart from what they
//! read, so a fan stopped at startup is still read when it spins. Every
//! recorded channel is read each poll.

use std::fs;
use std::io::ErrorKind;
use std::path::{Path, PathBuf};

use super::{Chip, Kind, Quantity, Rapl, Sensor};

/// One channel of a chip and the file it is read from.
#[derive(Clone, Debug, PartialEq)]
struct Channel {
    input: PathBuf,
    channel: String,
    label: String,
    quantity: Quantity,
    limit: Option<f64>,
    primary: bool,
}

/// A chip and its channels, as discovered.
#[derive(Clone, Debug, PartialEq)]
pub struct Source {
    id: String,
    kind: Kind,
    driver: String,
    model: Option<String>,
    name: String,
    channels: Vec<Channel>,
}

impl Source {
    pub fn sensors(&self) -> usize {
        self.channels.len()
    }

    pub fn fans(&self) -> usize {
        self.channels
            .iter()
            .filter(|channel| channel.quantity == Quantity::Rpm)
            .count()
    }
}

fn read_trimmed(path: &Path) -> Option<String> {
    fs::read_to_string(path)
        .ok()
        .map(|text| text.trim().to_owned())
        .filter(|text| !text.is_empty())
}

fn read_number(path: &Path) -> Option<f64> {
    read_trimmed(path)?.parse::<f64>().ok()
}

/// The number after `prefix` in a name such as `hwmon12`.
fn numbered(name: &str, prefix: &str) -> Option<u32> {
    name.strip_prefix(prefix)?.parse().ok()
}

fn kind(driver: &str, directory: &Path) -> Kind {
    match driver {
        "k10temp" | "zenpower" | "coretemp" | "cpu_thermal" => Kind::Cpu,
        "nvme" => Kind::Nvme,
        "spd5118" | "jc42" => Kind::Dimm,
        _ if directory.join("device/net").is_dir() => Kind::Nic,
        _ if ["nct", "it87", "it8", "asus", "f71", "w83"]
            .iter()
            .any(|prefix| driver.starts_with(prefix)) =>
        {
            Kind::Board
        }
        _ => Kind::Other,
    }
}

/// The labels that name a chip's main temperature, the most telling first.
const PRIMARY: [&str; 3] = ["Tdie", "Tctl", "Composite"];

fn channels(directory: &Path) -> Vec<Channel> {
    let Ok(entries) = fs::read_dir(directory) else {
        return Vec::new();
    };
    let mut found = Vec::new();
    for entry in entries.flatten() {
        let file = entry.file_name();
        let Some(stem) = file.to_str().and_then(|name| name.strip_suffix("_input")) else {
            continue;
        };
        let (quantity, order) = if stem.starts_with("temp") {
            (Quantity::Celsius, 0)
        } else if stem.starts_with("fan") {
            (Quantity::Rpm, 1)
        } else if stem.starts_with("power") {
            (Quantity::Watts, 2)
        } else if stem.starts_with("in") {
            (Quantity::Volts, 3)
        } else {
            continue;
        };
        let prefix = &stem[..stem.len() - stem.trim_start_matches(char::is_alphabetic).len()];
        let Some(number) = numbered(stem, prefix) else {
            continue;
        };
        let label = read_trimmed(&directory.join(format!("{stem}_label")));
        // Unlabelled voltages are a board's raw inputs, which mean nothing
        // without the board's divider table.
        if quantity == Quantity::Volts && label.is_none() {
            continue;
        }
        let limit = (quantity == Quantity::Celsius)
            .then(|| {
                read_number(&directory.join(format!("{stem}_crit")))
                    .or_else(|| read_number(&directory.join(format!("{stem}_max"))))
            })
            .flatten()
            .map(|millidegrees| millidegrees / 1000.0);
        found.push((
            (order, number),
            Channel {
                input: entry.path(),
                channel: stem.to_owned(),
                label: label.unwrap_or_else(|| stem.to_owned()),
                quantity,
                limit,
                primary: false,
            },
        ));
    }
    found.sort_by_key(|(order, _)| *order);
    let mut channels = found
        .into_iter()
        .map(|(_, channel)| channel)
        .collect::<Vec<_>>();
    let temperatures = || {
        channels
            .iter()
            .enumerate()
            .filter(|(_, channel)| channel.quantity == Quantity::Celsius)
    };
    let primary = PRIMARY
        .iter()
        .find_map(|label| {
            temperatures()
                .find(|(_, channel)| channel.label == *label)
                .map(|(index, _)| index)
        })
        .or_else(|| {
            temperatures()
                .find(|(_, channel)| channel.label.starts_with("Package id"))
                .map(|(index, _)| index)
        })
        .or_else(|| temperatures().next().map(|(index, _)| index));
    if let Some(primary) = primary {
        channels[primary].primary = true;
    }
    // Every package of a multi-socket host is a main temperature.
    for channel in &mut channels {
        if channel.label.starts_with("Package id") {
            channel.primary = true;
        }
    }
    channels
}

/// Every chip under `sys_root/class/hwmon`, in hwmon order, with its
/// channels. CPU chips are numbered when there are several; NVMe drives
/// take their controller's number; DIMMs are numbered in order; a network
/// device's chip is named by its interface.
pub fn discover(sys_root: &Path) -> Vec<Source> {
    let class = sys_root.join("class/hwmon");
    let Ok(entries) = fs::read_dir(&class) else {
        return Vec::new();
    };
    let mut directories = entries
        .flatten()
        .filter_map(|entry| {
            let name = entry.file_name().into_string().ok()?;
            Some((numbered(&name, "hwmon")?, name, entry.path()))
        })
        .collect::<Vec<_>>();
    directories.sort();
    let mut sources = Vec::new();
    for (_, id, directory) in directories {
        let Some(driver) = read_trimmed(&directory.join("name")) else {
            continue;
        };
        let channels = channels(&directory);
        if channels.is_empty() {
            continue;
        }
        let kind = kind(&driver, &directory);
        let device = fs::canonicalize(directory.join("device")).ok();
        let device_name = device
            .as_ref()
            .and_then(|device| device.file_name())
            .and_then(|name| name.to_str())
            .map(str::to_owned);
        let (model, name) = match kind {
            Kind::Nvme => (
                read_trimmed(&directory.join("device/model")),
                device_name
                    .as_deref()
                    .and_then(|name| numbered(name, "nvme"))
                    .map_or_else(|| "NVMe".to_owned(), |number| format!("NVMe {number}")),
            ),
            Kind::Nic => (
                None,
                fs::read_dir(directory.join("device/net"))
                    .ok()
                    .and_then(|mut interfaces| interfaces.next())
                    .and_then(|interface| interface.ok())
                    .and_then(|interface| interface.file_name().into_string().ok())
                    .unwrap_or_else(|| driver.clone()),
            ),
            Kind::Cpu => (None, "CPU".to_owned()),
            Kind::Dimm => (None, "DIMM".to_owned()),
            Kind::Board | Kind::Other => (None, driver.clone()),
        };
        sources.push(Source {
            id,
            kind,
            driver,
            model,
            name,
            channels,
        });
    }
    let count = |kind| sources.iter().filter(|source| source.kind == kind).count();
    let (cpus, dimms) = (count(Kind::Cpu), count(Kind::Dimm));
    let (mut cpu, mut dimm) = (0, 0);
    for source in &mut sources {
        if source.kind == Kind::Cpu && cpus > 1 {
            source.name = format!("CPU {cpu}");
            cpu += 1;
        } else if source.kind == Kind::Dimm {
            source.name = if dimms > 1 {
                format!("DIMM {dimm}")
            } else {
                "DIMM".to_owned()
            };
            dimm += 1;
        }
    }
    sources
}

/// Reads every channel of `sources`. The flag says whether a read failed,
/// so the chips are discovered again.
pub fn read(sources: &[Source]) -> (Vec<Chip>, bool) {
    let mut failed = false;
    let chips = sources
        .iter()
        .map(|source| Chip {
            id: source.id.clone(),
            kind: source.kind,
            driver: source.driver.clone(),
            model: source.model.clone(),
            name: source.name.clone(),
            sensors: source
                .channels
                .iter()
                .map(|channel| {
                    let scale = match channel.quantity {
                        Quantity::Celsius | Quantity::Volts => 1e-3,
                        Quantity::Rpm => 1.0,
                        Quantity::Watts => 1e-6,
                    };
                    let value = read_number(&channel.input).map(|raw| raw * scale);
                    failed |= value.is_none();
                    Sensor {
                        channel: channel.channel.clone(),
                        label: channel.label.clone(),
                        quantity: channel.quantity,
                        value,
                        limit: channel.limit,
                        primary: channel.primary,
                    }
                })
                .collect(),
        })
        .collect();
    (chips, failed)
}

/// The package zones of `sys_root/class/powercap`: their energy counters
/// and the value each wraps at.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Packages {
    zones: Vec<(PathBuf, u64)>,
}

impl Packages {
    /// The `intel-rapl:N` zones named `package-N`, which AMD's RAPL driver
    /// also provides.
    pub fn discover(sys_root: &Path) -> Self {
        let Ok(entries) = fs::read_dir(sys_root.join("class/powercap")) else {
            return Self::default();
        };
        let mut zones = entries
            .flatten()
            .filter(|entry| {
                let name = entry.file_name();
                let name = name.to_string_lossy();
                name.strip_prefix("intel-rapl:")
                    .is_some_and(|zone| zone.chars().all(|c| c.is_ascii_digit()))
                    && read_trimmed(&entry.path().join("name"))
                        .is_some_and(|zone| zone.starts_with("package"))
            })
            .map(|entry| {
                let range = read_number(&entry.path().join("max_energy_range_uj"))
                    .map_or(u64::MAX, |range| range as u64);
                (entry.path().join("energy_uj"), range)
            })
            .collect::<Vec<_>>();
        zones.sort();
        Self { zones }
    }

    /// Whether the counters can be read: they are root's alone since the
    /// Platypus side channel.
    pub fn access(&self) -> Rapl {
        match self.zones.first() {
            None => Rapl::Absent,
            Some((energy, _)) => match fs::read_to_string(energy) {
                Ok(_) => Rapl::Readable,
                Err(error) if error.kind() == ErrorKind::PermissionDenied => Rapl::NeedsRoot,
                Err(_) => Rapl::Absent,
            },
        }
    }

    /// Each zone's counter in microjoules, or `None` when one is unreadable.
    pub fn energy(&self) -> Option<Vec<u64>> {
        if self.zones.is_empty() {
            return None;
        }
        self.zones
            .iter()
            .map(|(energy, _)| read_number(energy).map(|joules| joules as u64))
            .collect()
    }

    /// Watts over `seconds` between two readings of [`Packages::energy`],
    /// a counter that wrapped counted through its range.
    pub fn watts(&self, before: &[u64], after: &[u64], seconds: f64) -> Option<f64> {
        if seconds <= 0.0 || before.len() != after.len() || before.len() != self.zones.len() {
            return None;
        }
        let microjoules = before
            .iter()
            .zip(after)
            .zip(&self.zones)
            .map(|((before, after), (_, range))| {
                if after >= before {
                    after - before
                } else {
                    range.saturating_sub(*before) + after
                }
            })
            .sum::<u64>();
        Some(microjoules as f64 / 1e6 / seconds)
    }
}

#[cfg(test)]
pub(super) mod tests {
    use super::*;
    use std::os::unix::fs::symlink;

    pub fn scratch(name: &str) -> PathBuf {
        let root =
            std::env::temp_dir().join(format!("chopinfer-top-host-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        root
    }

    /// A chip `hwmonN` whose device is `device` under `devices/`, with
    /// `files` written into it.
    pub fn chip(sys: &Path, number: u32, device: &str, files: &[(&str, &str)]) -> PathBuf {
        let target = sys
            .join("devices")
            .join(device)
            .join(format!("hwmon/hwmon{number}"));
        fs::create_dir_all(&target).unwrap();
        for (name, value) in files {
            let path = target.join(name);
            fs::create_dir_all(path.parent().unwrap()).unwrap();
            fs::write(path, format!("{value}\n")).unwrap();
        }
        let _ = symlink(sys.join("devices").join(device), target.join("device"));
        fs::create_dir_all(sys.join("class/hwmon")).unwrap();
        symlink(&target, sys.join(format!("class/hwmon/hwmon{number}"))).unwrap();
        target
    }

    #[test]
    fn chips_are_classified_with_their_main_sensors() {
        let sys = scratch("classify");
        chip(
            &sys,
            2,
            "pci0/cpu",
            &[
                ("name", "k10temp"),
                ("temp1_label", "Tctl"),
                ("temp1_input", "61625"),
                ("temp3_label", "Tccd1"),
                ("temp3_input", "58000"),
            ],
        );
        chip(
            &sys,
            0,
            "pci0/nvme/nvme1",
            &[
                ("name", "nvme"),
                ("temp1_label", "Composite"),
                ("temp1_input", "38850"),
                ("temp1_max", "81850"),
                ("temp1_crit", "84850"),
                ("temp2_label", "Sensor 1"),
                ("temp2_input", "38850"),
                ("temp2_max", "65261850"),
            ],
        );
        fs::write(
            sys.join("devices/pci0/nvme/nvme1/model"),
            "Example NVMe 4TB   \n",
        )
        .unwrap();
        chip(
            &sys,
            10,
            "i2c/1-0051",
            &[
                ("name", "spd5118"),
                ("temp1_input", "41000"),
                ("temp1_crit", "85000"),
            ],
        );
        chip(
            &sys,
            11,
            "i2c/1-0053",
            &[("name", "spd5118"), ("temp1_input", "40000")],
        );
        chip(
            &sys,
            4,
            "pci0/eth",
            &[("name", "bnxt_en"), ("temp1_input", "52000")],
        );
        fs::create_dir_all(sys.join("devices/pci0/eth/net/enp1s0f0")).unwrap();
        chip(
            &sys,
            5,
            "platform/nct6799.2592",
            &[
                ("name", "nct6799"),
                ("temp1_label", "SYSTIN"),
                ("temp1_input", "38000"),
                ("fan1_input", "1240"),
                // Nothing on this header yet; it is still recorded.
                ("fan5_input", "0"),
                ("in0_input", "1104"),
                ("in1_input", "1800"),
                ("in1_label", "+1.8V"),
            ],
        );
        // A chip without a name, and one with nothing to read, are skipped.
        chip(&sys, 7, "virtual/a", &[("temp1_input", "1000")]);
        chip(&sys, 8, "virtual/b", &[("name", "acpi_fan")]);
        let sources = discover(&sys);
        let summary = sources
            .iter()
            .map(|source| (source.id.as_str(), source.kind, source.name.as_str()))
            .collect::<Vec<_>>();
        assert_eq!(
            summary,
            [
                ("hwmon0", Kind::Nvme, "NVMe 1"),
                ("hwmon2", Kind::Cpu, "CPU"),
                ("hwmon4", Kind::Nic, "enp1s0f0"),
                ("hwmon5", Kind::Board, "nct6799"),
                ("hwmon10", Kind::Dimm, "DIMM 0"),
                ("hwmon11", Kind::Dimm, "DIMM 1"),
            ]
        );
        assert_eq!(sources[0].model.as_deref(), Some("Example NVMe 4TB"));
        let (chips, failed) = read(&sources);
        assert!(!failed);
        let nvme = &chips[0].sensors;
        // The critical temperature is preferred over the maximum.
        assert_eq!(
            (
                nvme[0].label.as_str(),
                nvme[0].value,
                nvme[0].limit,
                nvme[0].primary
            ),
            ("Composite", Some(38.85), Some(84.85), true)
        );
        assert!(!nvme[1].primary);
        let cpu = &chips[1].sensors;
        assert_eq!(
            (cpu[0].label.as_str(), cpu[0].primary, cpu[0].limit),
            ("Tctl", true, None)
        );
        assert_eq!((cpu[1].label.as_str(), cpu[1].primary), ("Tccd1", false));
        let board = chips[3]
            .sensors
            .iter()
            .map(|sensor| (sensor.label.as_str(), sensor.quantity, sensor.value))
            .collect::<Vec<_>>();
        // The unlabelled voltage is left out; the stopped fan is kept.
        assert_eq!(
            board,
            [
                ("SYSTIN", Quantity::Celsius, Some(38.0)),
                ("fan1", Quantity::Rpm, Some(1240.0)),
                ("fan5", Quantity::Rpm, Some(0.0)),
                ("+1.8V", Quantity::Volts, Some(1.8)),
            ]
        );
        assert_eq!(chips[2].sensor_name(&chips[2].sensors[0]), "enp1s0f0");
        assert_eq!(chips[4].sensor_name(&chips[4].sensors[0]), "DIMM 0");
        assert_eq!(chips[4].sensors[0].limit, Some(85.0));
        // A fan stopped at discovery is read each poll, and its new value
        // seen.
        fs::write(
            sys.join("devices/platform/nct6799.2592/hwmon/hwmon5/fan5_input"),
            "870\n",
        )
        .unwrap();
        let (chips, _) = read(&sources);
        assert_eq!(chips[3].sensors[2].value, Some(870.0));
        // A channel that goes away fails its read.
        fs::remove_file(sys.join("devices/pci0/eth/hwmon/hwmon4/temp1_input")).unwrap();
        let (chips, failed) = read(&sources);
        assert!(failed);
        assert_eq!(chips[2].sensors[0].value, None);
        fs::remove_dir_all(&sys).unwrap();
    }

    #[test]
    fn several_cpu_packages_are_numbered_and_each_is_main() {
        let sys = scratch("sockets");
        for (number, package) in [(1, 0), (2, 1)] {
            chip(
                &sys,
                number,
                &format!("platform/coretemp.{package}"),
                &[
                    ("name", "coretemp"),
                    ("temp1_label", &format!("Package id {package}")),
                    ("temp1_input", "50000"),
                    ("temp1_crit", "100000"),
                    ("temp2_label", "Core 0"),
                    ("temp2_input", "48000"),
                ],
            );
        }
        let sources = discover(&sys);
        let (chips, _) = read(&sources);
        assert_eq!(
            chips
                .iter()
                .map(|chip| chip.name.as_str())
                .collect::<Vec<_>>(),
            ["CPU 0", "CPU 1"]
        );
        assert!(
            chips
                .iter()
                .all(|chip| chip.sensors[0].primary && !chip.sensors[1].primary)
        );
        fs::remove_dir_all(&sys).unwrap();
    }

    #[test]
    fn package_power_counts_through_a_wrapped_counter() {
        let sys = scratch("rapl");
        assert_eq!(Packages::discover(&sys).access(), Rapl::Absent);
        for (zone, name) in [("intel-rapl:0", "package-0"), ("intel-rapl:0:0", "core")] {
            let directory = sys.join("class/powercap").join(zone);
            fs::create_dir_all(&directory).unwrap();
            fs::write(directory.join("name"), format!("{name}\n")).unwrap();
            fs::write(directory.join("energy_uj"), "1000\n").unwrap();
            fs::write(directory.join("max_energy_range_uj"), "65532610987\n").unwrap();
        }
        let packages = Packages::discover(&sys);
        assert_eq!(packages.zones.len(), 1);
        assert_eq!(packages.access(), Rapl::Readable);
        assert_eq!(packages.energy(), Some(vec![1000]));
        assert_eq!(packages.watts(&[1_000_000], &[3_000_000], 2.0), Some(1.0));
        let wrapped = packages
            .watts(&[65_532_610_987 - 500_000], &[500_000], 1.0)
            .unwrap();
        assert!((wrapped - 1.0).abs() < 1e-9, "{wrapped}");
        assert_eq!(packages.watts(&[1], &[2], 0.0), None);
        fs::remove_dir_all(&sys).unwrap();
    }
}
