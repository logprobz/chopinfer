//! Reading this host once a poll on a thread of its own, and its disk
//! space every few polls on another.
//!
//! A poll reads about a dozen `/proc` files and every hwmon channel, which
//! takes well under a millisecond on a workstation; CPU shares and rates
//! come from the difference with the previous poll. Disk sizes are read
//! every [`SLOW_POLLS`] polls on their own thread, so a size read stuck on
//! a failing disk stops only disk-space updates.

use std::fs;
use std::path::PathBuf;
use std::sync::mpsc::{self, RecvTimeoutError, Sender};
use std::thread;
use std::time::{Duration, Instant};

use super::hwmon::{self, Packages, Source};
use super::proc::{self, CpuTimes};
use super::space::{self, Filesystem, Statvfs};
use super::{Chip, Pressure, Rapl, Rates, Reading};
use crate::gpu::poll::SLOW_POLLS;

/// What the host poller reports.
#[derive(Clone, Debug, PartialEq)]
pub enum Report {
    /// The host can be read: its CPU, memory and the sensors found.
    Opened {
        model: Option<String>,
        threads: usize,
        memory_total: Option<u64>,
        sensors: usize,
        chips: usize,
        fans: usize,
        rapl: Rapl,
    },
    /// `/proc/stat` could not be read, and why. Nothing more follows.
    Unavailable(String),
    /// One reading of the host and its sensors.
    Polled {
        reading: Box<Reading>,
        chips: Vec<Chip>,
    },
    /// The size of every filesystem shown, read at `at`.
    Space {
        at: f64,
        filesystems: Vec<Filesystem>,
    },
}

/// The counters of one poll that rates and shares are taken against.
#[derive(Clone, Debug, Default)]
struct Counters {
    at: f64,
    cpu: Option<CpuTimes>,
    cores: Vec<(u32, CpuTimes)>,
    network: Option<(u64, u64)>,
    disk: Option<(u64, u64)>,
    energy: Option<Vec<u64>>,
}

/// Takes readings of this host from `proc_root` and `sys_root`.
pub struct Sampler {
    proc_root: PathBuf,
    sys_root: PathBuf,
    sources: Vec<Source>,
    packages: Packages,
    previous: Option<Counters>,
    polls: u64,
    /// A sensor read failed, so the chips are discovered again at the next
    /// round of slow polls.
    rediscover: bool,
}

impl Sampler {
    pub fn new(proc_root: impl Into<PathBuf>, sys_root: impl Into<PathBuf>) -> Self {
        let sys_root = sys_root.into();
        Self {
            proc_root: proc_root.into(),
            sources: hwmon::discover(&sys_root),
            packages: Packages::discover(&sys_root),
            sys_root,
            previous: None,
            polls: 0,
            rediscover: false,
        }
    }

    fn read(&self, path: &str) -> Option<String> {
        fs::read_to_string(self.proc_root.join(path)).ok()
    }

    /// What the host is, or why it cannot be read.
    pub fn open(&self) -> Report {
        let stat = match fs::read_to_string(self.proc_root.join("stat")) {
            Ok(text) => text,
            Err(error) => return Report::Unavailable(format!("/proc/stat: {error}")),
        };
        let (_, cores) = proc::stat(&stat);
        Report::Opened {
            model: self.read("cpuinfo").as_deref().and_then(proc::cpu_model),
            threads: cores.len(),
            memory_total: self
                .read("meminfo")
                .as_deref()
                .and_then(proc::meminfo)
                .map(|memory| memory.total),
            sensors: self.sources.iter().map(Source::sensors).sum(),
            chips: self.sources.len(),
            fans: self.sources.iter().map(Source::fans).sum(),
            rapl: self.packages.access(),
        }
    }

    /// Whether interface `name` has a device: lo, docker, bridges and veth
    /// pairs have none.
    fn physical_interface(&self, name: &str) -> bool {
        self.sys_root
            .join("class/net")
            .join(name)
            .join("device")
            .exists()
    }

    /// Whether `name` is a whole disk counted once: it has a device and is
    /// not a hidden multipath path. Partitions are not under `block/`, and
    /// loop, ram, zram, dm and md devices have no device, so traffic through
    /// an array or a mapper is counted only on its disks.
    fn whole_disk(&self, name: &str) -> bool {
        let block = self.sys_root.join("block").join(name);
        block.join("device").exists()
            && fs::read_to_string(block.join("hidden"))
                .ok()
                .is_none_or(|hidden| hidden.trim() != "1")
    }

    fn counters(&self, at: f64) -> Counters {
        let (cpu, cores) = self
            .read("stat")
            .map_or((None, Vec::new()), |text| proc::stat(&text));
        let network = self.read("net/dev").and_then(|text| {
            let counted = proc::net_dev(&text)
                .into_iter()
                .filter(|(name, _, _)| self.physical_interface(name))
                .collect::<Vec<_>>();
            (!counted.is_empty()).then(|| {
                counted
                    .iter()
                    .fold((0u64, 0u64), |(rx, tx), (_, received, sent)| {
                        (rx.saturating_add(*received), tx.saturating_add(*sent))
                    })
            })
        });
        let disk = self.read("diskstats").map(|text| {
            proc::diskstats(&text)
                .into_iter()
                .filter(|(name, _, _)| self.whole_disk(name))
                .fold((0u64, 0u64), |(read, written), (_, r, w)| {
                    (read.saturating_add(r), written.saturating_add(w))
                })
        });
        Counters {
            at,
            cpu,
            cores,
            network,
            disk,
            energy: self.packages.energy(),
        }
    }

    /// One reading at `at` seconds, with the sensors read now.
    pub fn poll(&mut self, at: f64) -> Report {
        if self.rediscover && self.polls % SLOW_POLLS == 0 {
            self.sources = hwmon::discover(&self.sys_root);
            self.rediscover = false;
        }
        self.polls += 1;
        let counters = self.counters(at);
        let previous = self.previous.replace(counters.clone());
        let seconds = previous.as_ref().map_or(0.0, |previous| at - previous.at);
        let rates = |now: Option<(u64, u64)>, before: Option<(u64, u64)>| {
            let ((a, b), (c, d)) = (now?, before?);
            (seconds > 0.0 && a >= c && b >= d).then(|| Rates {
                incoming: (a - c) as f64 / seconds,
                outgoing: (b - d) as f64 / seconds,
            })
        };
        let pressure = |name: &str| {
            self.read(&format!("pressure/{name}"))
                .as_deref()
                .and_then(proc::pressure)
        };
        let reading = Reading {
            at,
            cpu: previous
                .as_ref()
                .and_then(|previous| previous.cpu?.shares(&counters.cpu?)),
            cores: previous.as_ref().map_or_else(Vec::new, |previous| {
                counters
                    .cores
                    .iter()
                    .map(|(number, after)| {
                        previous
                            .cores
                            .iter()
                            .find(|(seen, _)| seen == number)
                            .and_then(|(_, before)| before.shares(after))
                            .map_or(0.0, |shares| shares.busy)
                    })
                    .collect()
            }),
            memory: self.read("meminfo").as_deref().and_then(proc::meminfo),
            load: self.read("loadavg").as_deref().and_then(proc::loadavg),
            pressure: Pressure {
                cpu: pressure("cpu"),
                memory: pressure("memory"),
                io: pressure("io"),
            },
            disk: rates(counters.disk, previous.as_ref().and_then(|p| p.disk)),
            network: rates(counters.network, previous.as_ref().and_then(|p| p.network)),
            network_visible: counters.network.is_some(),
            uptime: self.read("uptime").as_deref().and_then(proc::uptime),
            package: previous.as_ref().and_then(|previous| {
                self.packages.watts(
                    previous.energy.as_ref()?,
                    counters.energy.as_ref()?,
                    seconds,
                )
            }),
        };
        let (chips, failed) = hwmon::read(&self.sources);
        self.rediscover |= failed;
        Report::Polled {
            reading: Box::new(reading),
            chips,
        }
    }
}

/// Reads the sizes of `proc_root`'s filesystems every `seconds` and sends
/// them to `results` until `stop` is dropped or sent to.
fn space_loop(
    proc_root: PathBuf,
    statvfs: Statvfs,
    seconds: f64,
    origin: Instant,
    results: Sender<Report>,
    stop: mpsc::Receiver<()>,
) {
    loop {
        let filesystems = space::sizes(&proc_root, statvfs);
        let at = origin.elapsed().as_secs_f64();
        if results.send(Report::Space { at, filesystems }).is_err() {
            return;
        }
        match stop.recv_timeout(Duration::from_secs_f64(seconds)) {
            Err(RecvTimeoutError::Timeout) => {}
            _ => return,
        }
    }
}

/// Starts reading disk sizes with `statvfs` every [`SLOW_POLLS`] intervals.
pub fn spawn_space(
    proc_root: impl Into<PathBuf>,
    statvfs: Statvfs,
    interval: f64,
    origin: Instant,
    results: Sender<Report>,
) -> Sender<()> {
    let (stop, stopped) = mpsc::channel();
    let proc_root = proc_root.into();
    thread::spawn(move || {
        space_loop(
            proc_root,
            statvfs,
            interval * SLOW_POLLS as f64,
            origin,
            results,
            stopped,
        );
    });
    stop
}

/// Starts reading the host from `proc_root` and `sys_root` every
/// `interval` seconds.
pub fn spawn_host(
    proc_root: impl Into<PathBuf>,
    sys_root: impl Into<PathBuf>,
    interval: f64,
    origin: Instant,
    results: Sender<Report>,
) -> Sender<()> {
    let (stop, stopped) = mpsc::channel::<()>();
    let (proc_root, sys_root) = (proc_root.into(), sys_root.into());
    thread::spawn(move || {
        let mut sampler = Sampler::new(proc_root, sys_root);
        let opened = sampler.open();
        let unavailable = matches!(opened, Report::Unavailable(_));
        if results.send(opened).is_err() || unavailable {
            return;
        }
        loop {
            let at = origin.elapsed().as_secs_f64();
            if results.send(sampler.poll(at)).is_err() {
                return;
            }
            match stopped.recv_timeout(Duration::from_secs_f64(interval)) {
                Err(RecvTimeoutError::Timeout) => {}
                _ => return,
            }
        }
    });
    stop
}

/// The host and disk-space threads. Dropping it stops both; a size read
/// that never returns keeps only its own thread.
#[derive(Debug)]
pub struct HostPoller {
    stops: [Sender<()>; 2],
}

impl HostPoller {
    /// Reads this host every `interval` seconds and its disk sizes every
    /// [`SLOW_POLLS`] intervals, sending both to `results`.
    pub fn spawn(interval: f64, origin: Instant, results: Sender<Report>) -> Self {
        let space = spawn_space("/proc", space::statvfs, interval, origin, results.clone());
        let host = spawn_host("/proc", "/sys", interval, origin, results);
        Self {
            stops: [host, space],
        }
    }
}

impl Drop for HostPoller {
    fn drop(&mut self) {
        for stop in &self.stops {
            let _ = stop.send(());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::host::hwmon::tests::{chip, scratch};
    use std::os::unix::fs::symlink;
    use std::path::Path;

    fn write(root: &Path, name: &str, text: &str) {
        let path = root.join(name);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, text).unwrap();
    }

    /// A fake `/proc` at `ticks` into its life, and a `/sys` with one
    /// device-backed interface, two disks under an md array and a mapper,
    /// a hidden multipath path and a CPU sensor.
    fn tree(name: &str) -> (PathBuf, PathBuf) {
        let root = scratch(name);
        let (proc_root, sys) = (root.join("proc"), root.join("sys"));
        write(
            &proc_root,
            "meminfo",
            "MemTotal: 1000 kB\nMemAvailable: 250 kB\n",
        );
        write(&proc_root, "loadavg", "1.00 0.50 0.25 1/300 42\n");
        // Pressure files are absent, as on kernels built without PSI.
        for disk in ["nvme0n1", "nvme1n1", "nvme2c2n1"] {
            fs::create_dir_all(sys.join("block").join(disk).join("device")).unwrap();
        }
        write(&sys, "block/nvme2c2n1/hidden", "1\n");
        for virtual_disk in ["md0", "dm-0", "loop0"] {
            fs::create_dir_all(sys.join("block").join(virtual_disk)).unwrap();
        }
        fs::create_dir_all(sys.join("devices/pci0/eth0")).unwrap();
        fs::create_dir_all(sys.join("class/net/eth0")).unwrap();
        symlink(
            sys.join("devices/pci0/eth0"),
            sys.join("class/net/eth0/device"),
        )
        .unwrap();
        fs::create_dir_all(sys.join("class/net/veth1")).unwrap();
        chip(
            &sys,
            0,
            "pci0/cpu",
            &[
                ("name", "k10temp"),
                ("temp1_label", "Tctl"),
                ("temp1_input", "61000"),
            ],
        );
        (proc_root, sys)
    }

    /// The counters at `second`: every disk moves 1 MiB a second each way,
    /// eth0 receives 2 KiB and sends 1 KiB a second, and CPU 0 is half busy.
    fn tick(proc_root: &Path, second: u64) {
        let stat = format!(
            "cpu  {} 0 0 {} 0 0 0 0 0 0\ncpu0 {} 0 0 {} 0 0 0 0\ncpu1 0 0 0 {} 0 0 0 0\n",
            50 * second,
            150 * second,
            50 * second,
            50 * second,
            100 * second
        );
        write(proc_root, "stat", &stat);
        let sectors = 2048 * second;
        let disks = [
            "nvme0n1",
            "nvme1n1",
            "nvme2c2n1",
            "md0",
            "dm-0",
            "loop0",
            "nvme0n1p1",
        ]
        .iter()
        .map(|disk| format!(" 259 0 {disk} 1 0 {sectors} 1 1 0 {sectors} 1 0 1 1\n"))
        .collect::<String>();
        write(proc_root, "diskstats", &disks);
        let net = format!(
            "Inter-|\n face |\n    lo: {0} 0 0 0 0 0 0 0 {0} 0 0 0 0 0 0 0\n  eth0: {1} 0 0 0 0 0 0 0 {2} 0 0 0 0 0 0 0\n veth1: {0} 0 0 0 0 0 0 0 {0} 0 0 0 0 0 0 0\n",
            9999 * second,
            2048 * second,
            1024 * second
        );
        write(proc_root, "net/dev", &net);
    }

    /// Reads this host; run with `--ignored --nocapture` and compare with
    /// `top`, `free`, `df` and `/sys/class/hwmon`.
    #[test]
    #[ignore = "reads the host it runs on"]
    fn reads_this_host() {
        let mut sampler = Sampler::new("/proc", "/sys");
        println!("{:?}", sampler.open());
        for at in [0.0, 1.0] {
            thread::sleep(Duration::from_secs_f64(at));
            println!("{:?}", sampler.poll(at));
        }
        for filesystem in space::sizes(Path::new("/proc"), space::statvfs) {
            println!("{filesystem:?} {:.1}%", filesystem.share() * 100.0);
        }
    }

    #[test]
    fn rates_and_shares_come_from_consecutive_polls() {
        let (proc_root, sys) = tree("rates");
        tick(&proc_root, 1);
        let mut sampler = Sampler::new(&proc_root, &sys);
        let Report::Opened {
            threads,
            memory_total,
            sensors,
            fans,
            ..
        } = sampler.open()
        else {
            panic!("not opened");
        };
        assert_eq!(
            (threads, memory_total, sensors, fans),
            (2, Some(1000 * 1024), 1, 0)
        );
        let Report::Polled {
            reading: first,
            chips,
        } = sampler.poll(1.0)
        else {
            panic!("not polled");
        };
        assert_eq!(
            (first.cpu, first.network, first.network_visible),
            (None, None, true)
        );
        assert_eq!(chips[0].sensors[0].value, Some(61.0));
        assert_eq!(first.pressure, Pressure::default());
        tick(&proc_root, 3);
        let Report::Polled { reading, .. } = sampler.poll(3.0) else {
            panic!("not polled");
        };
        assert_eq!(reading.cpu.map(|cpu| cpu.busy), Some(0.25));
        assert_eq!(reading.cores, [0.5, 0.0]);
        // Two whole disks count; the array, the mapper, the loop device,
        // the hidden path and the partition do not.
        assert_eq!(
            reading.disk,
            Some(Rates {
                incoming: 2.0 * 1024.0 * 1024.0,
                outgoing: 2.0 * 1024.0 * 1024.0
            })
        );
        // Only eth0 has a device.
        assert_eq!(
            reading.network,
            Some(Rates {
                incoming: 2048.0,
                outgoing: 1024.0
            })
        );
        assert_eq!(reading.memory.map(|memory| memory.used()), Some(750 * 1024));
        assert_eq!(reading.load.map(|load| load.tasks), Some(Some(300)));
        fs::remove_dir_all(proc_root.parent().unwrap()).unwrap();
    }

    #[test]
    fn without_a_device_backed_interface_the_network_is_unknown() {
        let (proc_root, sys) = tree("netns");
        fs::remove_file(sys.join("class/net/eth0/device")).unwrap();
        let mut sampler = Sampler::new(&proc_root, &sys);
        for second in [1, 2] {
            tick(&proc_root, second);
            let Report::Polled { reading, .. } = sampler.poll(second as f64) else {
                panic!("not polled");
            };
            assert_eq!((reading.network, reading.network_visible), (None, false));
        }
        fs::remove_dir_all(proc_root.parent().unwrap()).unwrap();
    }

    #[test]
    fn an_unreadable_proc_is_unavailable() {
        let root = scratch("missing");
        let sampler = Sampler::new(root.join("proc"), root.join("sys"));
        assert!(
            matches!(sampler.open(), Report::Unavailable(reason) if reason.starts_with("/proc/stat"))
        );
        fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn a_stuck_size_read_does_not_hold_up_the_host() {
        fn stuck(_: &Path) -> std::io::Result<space::Size> {
            loop {
                thread::park();
            }
        }
        let (proc_root, sys) = tree("stuck");
        tick(&proc_root, 1);
        write(
            &proc_root,
            "self/mountinfo",
            &format!(
                "20 1 259:0 / {} rw - ext4 /dev/x rw\n",
                std::env::temp_dir().display()
            ),
        );
        let (results, reports) = mpsc::channel();
        let origin = Instant::now();
        let _space = spawn_space(&proc_root, stuck, 0.05, origin, results.clone());
        let _host = spawn_host(&proc_root, &sys, 0.05, origin, results);
        let mut polled = 0;
        let deadline = Instant::now() + Duration::from_secs(10);
        while polled < 3 && Instant::now() < deadline {
            match reports.recv_timeout(Duration::from_secs(1)) {
                Ok(Report::Polled { .. }) => polled += 1,
                Ok(Report::Space { .. }) => panic!("a stuck read sent sizes"),
                _ => {}
            }
        }
        assert_eq!(polled, 3);
        fs::remove_dir_all(proc_root.parent().unwrap()).unwrap();
    }
}
