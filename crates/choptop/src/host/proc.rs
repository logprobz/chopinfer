//! Parsers for the `/proc` files the host panel reads: CPU times, memory,
//! load, pressure, network and disk counters, the CPU model and the
//! mounts. Each takes the file's text, so tests give them fixed input.

use std::collections::{HashMap, HashSet};

use super::{Cpu, Load, Memory};

/// Ticks one CPU, or all of them, spent in each state.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CpuTimes {
    /// User and nice; guest time is already in them.
    pub user: u64,
    /// System, interrupt and softirq.
    pub system: u64,
    pub idle: u64,
    pub iowait: u64,
    pub steal: u64,
}

impl CpuTimes {
    fn total(&self) -> u64 {
        self.user + self.system + self.idle + self.iowait + self.steal
    }

    /// The share of time in each state between `self` and a later `after`;
    /// `None` when no tick passed or a counter went back.
    pub fn shares(&self, after: &Self) -> Option<Cpu> {
        let total = after.total().checked_sub(self.total())?;
        if total == 0 {
            return None;
        }
        let delta = |before: u64, after: u64| after.saturating_sub(before) as f64 / total as f64;
        let idle = delta(self.idle, after.idle);
        let iowait = delta(self.iowait, after.iowait);
        Some(Cpu {
            busy: (1.0 - idle - iowait).clamp(0.0, 1.0),
            user: delta(self.user, after.user),
            system: delta(self.system, after.system),
            iowait,
        })
    }
}

fn cpu_times(fields: &[u64]) -> Option<CpuTimes> {
    let field = |index: usize| fields.get(index).copied().unwrap_or(0);
    (fields.len() >= 4).then(|| CpuTimes {
        user: field(0) + field(1),
        system: field(2) + field(5) + field(6),
        idle: field(3),
        iowait: field(4),
        steal: field(7),
    })
}

/// The `cpu` line of `/proc/stat` and each `cpuN` line with its number.
pub fn stat(text: &str) -> (Option<CpuTimes>, Vec<(u32, CpuTimes)>) {
    let mut all = None;
    let mut cores = Vec::new();
    for line in text.lines() {
        let mut words = line.split_whitespace();
        let Some(name) = words.next() else {
            continue;
        };
        let Some(number) = name.strip_prefix("cpu") else {
            continue;
        };
        let fields = words
            .map_while(|word| word.parse().ok())
            .collect::<Vec<u64>>();
        let Some(times) = cpu_times(&fields) else {
            continue;
        };
        if number.is_empty() {
            all = Some(times);
        } else if let Ok(number) = number.parse() {
            cores.push((number, times));
        }
    }
    (all, cores)
}

/// `/proc/meminfo`; `None` without a total.
pub fn meminfo(text: &str) -> Option<Memory> {
    let mut values = HashMap::new();
    for line in text.lines() {
        let Some((name, rest)) = line.split_once(':') else {
            continue;
        };
        if let Some(kib) = rest
            .split_whitespace()
            .next()
            .and_then(|value| value.parse::<u64>().ok())
        {
            values.insert(name.trim(), kib.saturating_mul(1024));
        }
    }
    let total = *values.get("MemTotal")?;
    // Kernels before 3.14 have no MemAvailable.
    let available = values.get("MemAvailable").copied().unwrap_or_else(|| {
        values.get("MemFree").copied().unwrap_or(0) + values.get("Cached").copied().unwrap_or(0)
    });
    Some(Memory {
        total,
        available: available.min(total),
        cached: values.get("Cached").copied().unwrap_or(0),
        swap_total: values.get("SwapTotal").copied().unwrap_or(0),
        swap_free: values.get("SwapFree").copied().unwrap_or(0),
    })
}

/// `/proc/loadavg`: `2.15 4.80 5.68 2/1004 989857`.
pub fn loadavg(text: &str) -> Option<Load> {
    let mut words = text.split_whitespace();
    let mut average = || words.next()?.parse::<f64>().ok();
    let (one, five, fifteen) = (average()?, average()?, average()?);
    let tasks = text
        .split_whitespace()
        .nth(3)
        .and_then(|running| running.split_once('/'))
        .and_then(|(_, tasks)| tasks.parse().ok());
    Some(Load {
        one,
        five,
        fifteen,
        tasks,
    })
}

/// The `some avg10` share of a `/proc/pressure` file.
pub fn pressure(text: &str) -> Option<f64> {
    text.lines()
        .find(|line| line.starts_with("some "))?
        .split_whitespace()
        .find_map(|word| word.strip_prefix("avg10="))?
        .parse::<f64>()
        .ok()
        .map(|percent| percent / 100.0)
}

/// Each interface of `/proc/net/dev` with its received and sent bytes.
pub fn net_dev(text: &str) -> Vec<(String, u64, u64)> {
    text.lines()
        .filter_map(|line| {
            let (name, counters) = line.split_once(':')?;
            let counters = counters
                .split_whitespace()
                .map(|word| word.parse::<u64>().ok())
                .collect::<Option<Vec<_>>>()?;
            Some((
                name.trim().to_owned(),
                *counters.first()?,
                *counters.get(8)?,
            ))
        })
        .collect()
}

/// Each device of `/proc/diskstats` with the bytes it read and wrote.
pub fn diskstats(text: &str) -> Vec<(String, u64, u64)> {
    const SECTOR: u64 = 512;
    text.lines()
        .filter_map(|line| {
            let words = line.split_whitespace().collect::<Vec<_>>();
            let number = |index: usize| words.get(index)?.parse::<u64>().ok();
            Some((
                (*words.get(2)?).to_owned(),
                number(5)?.saturating_mul(SECTOR),
                number(9)?.saturating_mul(SECTOR),
            ))
        })
        .collect()
}

/// Seconds since boot, the first number of `/proc/uptime`.
pub fn uptime(text: &str) -> Option<f64> {
    text.split_whitespace().next()?.parse().ok()
}

/// The first `model name` of `/proc/cpuinfo`, without a trailing core
/// count or "Processor": `AMD EPYC 9555 64-Core Processor` is
/// `AMD EPYC 9555`.
pub fn cpu_model(text: &str) -> Option<String> {
    let model = text
        .lines()
        .find(|line| line.starts_with("model name"))?
        .split_once(':')?
        .1;
    let mut words = model.split_whitespace().collect::<Vec<_>>();
    while let Some(last) = words.last() {
        let lower = last.to_ascii_lowercase();
        let cores = lower
            .strip_suffix("-cores")
            .or_else(|| lower.strip_suffix("-core"))
            .is_some_and(|count| !count.is_empty() && count.chars().all(|c| c.is_ascii_digit()));
        if cores || lower == "processor" {
            words.pop();
        } else {
            break;
        }
    }
    (!words.is_empty()).then(|| words.join(" "))
}

/// One line of `/proc/self/mountinfo`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Mount {
    pub id: u64,
    pub parent: u64,
    /// `major:minor`.
    pub device: String,
    /// Where it is mounted, its octal escapes decoded.
    pub point: String,
    pub filesystem: String,
}

/// `\040` for a space, `\011` for a tab, `\012` for a newline and `\134`
/// for a backslash, as the kernel escapes mount points.
fn unescape(text: &str) -> String {
    let bytes = text.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut index = 0;
    while index < bytes.len() {
        let octal = bytes.get(index + 1..index + 4).filter(|digits| {
            bytes[index] == b'\\' && digits.iter().all(|digit| (b'0'..=b'7').contains(digit))
        });
        match octal {
            Some(digits) => {
                let value = digits
                    .iter()
                    .fold(0u32, |value, digit| value * 8 + u32::from(digit - b'0'));
                out.push(value as u8);
                index += 4;
            }
            None => {
                out.push(bytes[index]);
                index += 1;
            }
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// The lines of `/proc/self/mountinfo`.
pub fn mountinfo(text: &str) -> Vec<Mount> {
    text.lines()
        .filter_map(|line| {
            let (before, after) = line.split_once(" - ")?;
            let words = before.split_whitespace().collect::<Vec<_>>();
            Some(Mount {
                id: words.first()?.parse().ok()?,
                parent: words.get(1)?.parse().ok()?,
                device: (*words.get(2)?).to_owned(),
                point: unescape(words.get(4)?),
                filesystem: after.split_whitespace().next()?.to_owned(),
            })
        })
        .collect()
}

/// Whether each mount can be reached through its mount point. A mount is
/// hidden when another sits on the same point with it as its parent. A
/// mount not hidden that way is reached when the mount its point lies on
/// is: its parent, past any it is stacked on at the same point. The root
/// has no listed parent, or itself.
pub fn visible(mounts: &[Mount]) -> Vec<bool> {
    let by_id = mounts
        .iter()
        .enumerate()
        .map(|(index, mount)| (mount.id, index))
        .collect::<HashMap<_, _>>();
    let parent = |index: usize| {
        by_id
            .get(&mounts[index].parent)
            .copied()
            .filter(|parent| *parent != index)
    };
    let stacked = (0..mounts.len())
        .filter_map(|index| {
            let below = parent(index)?;
            (mounts[below].point == mounts[index].point).then_some(below)
        })
        .collect::<HashSet<_>>();
    // The mount a point lies on: its parent, past those it is stacked on.
    let base = |index: usize| {
        let mut below = parent(index);
        for _ in 0..mounts.len() {
            match below {
                Some(next) if mounts[next].point == mounts[index].point => below = parent(next),
                _ => break,
            }
        }
        below
    };
    let mut seen = (0..mounts.len())
        .map(|index| stacked.contains(&index).then_some(false))
        .collect::<Vec<_>>();
    for start in 0..mounts.len() {
        let mut path = Vec::new();
        let mut index = Some(start);
        let reached = loop {
            let Some(current) = index else {
                break true;
            };
            if let Some(known) = seen[current] {
                break known;
            }
            // A cycle, which the kernel never lists.
            if path.len() > mounts.len() {
                break false;
            }
            path.push(current);
            index = base(current);
        };
        for index in path {
            seen[index] = Some(reached);
        }
    }
    seen.into_iter()
        .map(|shown| shown.unwrap_or(false))
        .collect()
}

/// Filesystems on local disks.
const LOCAL: [&str; 8] = [
    "ext2", "ext3", "ext4", "xfs", "btrfs", "f2fs", "bcachefs", "zfs",
];

/// The mount points whose size is shown: the visible local disk
/// filesystems that `is_directory` accepts, one per device, the shortest
/// point winning, and the tmpfs at `/dev/shm`. `/` comes first, then the
/// rest by name. Network and FUSE filesystems are left out, as a size read
/// on an unreachable server blocks.
pub fn local(mounts: &[Mount], is_directory: impl Fn(&str) -> bool) -> Vec<String> {
    let mut chosen: HashMap<&str, &str> = HashMap::new();
    for (mount, shown) in mounts.iter().zip(visible(mounts)) {
        let kept = LOCAL.contains(&mount.filesystem.as_str())
            || (mount.filesystem == "tmpfs" && mount.point == "/dev/shm");
        if !shown || !kept || !is_directory(&mount.point) {
            continue;
        }
        let entry = chosen.entry(mount.device.as_str()).or_insert(&mount.point);
        if (mount.point.len(), mount.point.as_str()) < (entry.len(), *entry) {
            *entry = &mount.point;
        }
    }
    let mut points = chosen.into_values().map(str::to_owned).collect::<Vec<_>>();
    points.sort_by(|a, b| (a.as_str() != "/", a).cmp(&(b.as_str() != "/", b)));
    points
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_shares_come_from_deltas() {
        let before = "cpu  100 0 50 800 50 0 0 0 0 0\ncpu0 50 0 25 400 25 0 0 0\ncpu1 50 0 25 400 25 0 0 0\nintr 1 2 3\n";
        let after = "cpu  160 20 70 900 70 5 5 0 0 0\ncpu0 110 20 45 410 35 5 5 0\ncpu1 50 0 25 490 35 0 0 0\n";
        let (all, cores) = stat(before);
        let (later, later_cores) = stat(after);
        assert_eq!(
            cores.iter().map(|(number, _)| *number).collect::<Vec<_>>(),
            [0, 1]
        );
        // 230 ticks: 80 user, 30 system, 100 idle, 20 iowait.
        let shares = all.unwrap().shares(&later.unwrap()).unwrap();
        assert_eq!(
            (shares.user, shares.system, shares.iowait),
            (80.0 / 230.0, 30.0 / 230.0, 20.0 / 230.0)
        );
        assert!((shares.busy - 110.0 / 230.0).abs() < 1e-12);
        let core = |index: usize| cores[index].1.shares(&later_cores[index].1).unwrap().busy;
        assert!((core(0) - 110.0 / 130.0).abs() < 1e-12);
        assert_eq!(core(1), 0.0);
        // No tick passed, or a counter went back.
        assert_eq!(all.unwrap().shares(&all.unwrap()), None);
        assert_eq!(later.unwrap().shares(&all.unwrap()), None);
    }

    #[test]
    fn reads_memory_load_and_pressure() {
        let text = "MemTotal:       263227348 kB\nMemFree:  1000 kB\nMemAvailable:   243907840 kB\nCached:         154871612 kB\nSwapTotal:       8388604 kB\nSwapFree:        8388348 kB\n";
        let memory = meminfo(text).unwrap();
        assert_eq!(memory.total, 263_227_348 * 1024);
        assert_eq!(memory.used(), (263_227_348 - 243_907_840) * 1024);
        assert_eq!(memory.swap_used(), 256 * 1024);
        assert_eq!(meminfo("MemFree: 1 kB\n"), None);
        let old = meminfo("MemTotal: 100 kB\nMemFree: 30 kB\nCached: 20 kB\n").unwrap();
        assert_eq!(old.available, 50 * 1024);
        assert_eq!(
            loadavg("2.15 4.80 5.68 2/1004 989857\n"),
            Some(Load {
                one: 2.15,
                five: 4.80,
                fifteen: 5.68,
                tasks: Some(1004)
            })
        );
        assert_eq!(loadavg(""), None);
        assert_eq!(uptime("1051234.56 47000000.00\n"), Some(1_051_234.56));
        let psi = "some avg10=12.50 avg60=0.31 avg300=2.85 total=814697727\nfull avg10=0.00 avg60=0.00 avg300=0.00 total=0\n";
        assert_eq!(pressure(psi), Some(0.125));
        assert_eq!(pressure("full avg10=1.00\n"), None);
    }

    #[test]
    fn reads_network_and_disk_counters() {
        let net = "Inter-|   Receive                                                |  Transmit\n face |bytes    packets errs drop fifo frame compressed multicast|bytes    packets errs drop fifo colls carrier compressed\n    lo: 500 5 0 0 0 0 0 0 500 5 0 0 0 0 0 0\n  eth0: 1000 10 0 0 0 0 0 0 2000 20 0 0 0 0 0 0\n";
        assert_eq!(
            net_dev(net),
            [("lo".to_owned(), 500, 500), ("eth0".to_owned(), 1000, 2000)]
        );
        let disks = " 259       0 nvme0n1 100 0 2048 10 50 0 4096 20 0 30 30 0 0 0 0\n 259       1 nvme0n1p1 50 0 1024 5 25 0 2048 10 0 15 15\n";
        assert_eq!(
            diskstats(disks),
            [
                ("nvme0n1".to_owned(), 2048 * 512, 4096 * 512),
                ("nvme0n1p1".to_owned(), 1024 * 512, 2048 * 512)
            ]
        );
    }

    #[test]
    fn the_cpu_model_drops_core_counts_and_processor() {
        let model = |name: &str| cpu_model(&format!("processor\t: 0\nmodel name\t: {name}\n"));
        assert_eq!(
            model("AMD Ryzen Threadripper PRO 9965WX 24-Cores").as_deref(),
            Some("AMD Ryzen Threadripper PRO 9965WX")
        );
        assert_eq!(
            model("AMD EPYC 9555 64-Core Processor").as_deref(),
            Some("AMD EPYC 9555")
        );
        assert_eq!(
            model("Intel(R) Xeon(R) Platinum 8480+").as_deref(),
            Some("Intel(R) Xeon(R) Platinum 8480+")
        );
        assert_eq!(cpu_model("processor\t: 0\nBogoMIPS\t: 50.00\n"), None);
    }

    fn line(id: u64, parent: u64, device: &str, point: &str, filesystem: &str) -> String {
        format!("{id} {parent} {device} / {point} rw,relatime shared:1 - {filesystem} /dev/x rw\n")
    }

    #[test]
    fn mount_points_are_unescaped() {
        let mounts = mountinfo(&line(30, 1, "8:1", "/models\\040store", "ext4"));
        assert_eq!(mounts[0].point, "/models store");
        assert_eq!(mounts[0].device, "8:1");
        assert_eq!(unescape("a\\134b\\0"), "a\\b\\0");
    }

    #[test]
    fn only_visible_local_filesystems_are_kept_once_each() {
        let text = [
            line(20, 1, "253:0", "/", "ext4"),
            line(21, 20, "0:5", "/proc", "proc"),
            line(22, 20, "0:24", "/dev/shm", "tmpfs"),
            line(23, 20, "0:25", "/run", "tmpfs"),
            line(24, 20, "259:2", "/boot", "ext4"),
            // A bind mount of /boot's filesystem: one entry, the shorter.
            line(25, 20, "259:2", "/boot-copy", "ext4"),
            line(26, 20, "0:40", "/share", "nfs4"),
            line(27, 20, "0:41", "/smb", "cifs"),
            line(28, 20, "0:42", "/fuse", "fuse.sshfs"),
            line(
                29,
                20,
                "0:43",
                "/var/lib/docker/overlay2/x/merged",
                "overlay",
            ),
            line(31, 20, "7:0", "/snap/core/1", "squashfs"),
            // An ext4 /store hidden by NFS mounted on it, and a disk mounted
            // under the hidden /store: statvfs("/store") would ask the NFS
            // server, so neither is kept.
            line(32, 20, "259:3", "/store", "ext4"),
            line(33, 32, "259:4", "/store/models", "ext4"),
            line(34, 32, "0:44", "/store", "nfs"),
            // A disk mounted over a tmpfs is the one reached.
            line(35, 20, "0:45", "/scratch", "tmpfs"),
            line(36, 35, "259:5", "/scratch", "xfs"),
            // A file bind mount is not a directory.
            line(37, 20, "259:6", "/etc/hosts", "ext4"),
            line(38, 20, "259:7", "/a\\040b", "btrfs"),
        ]
        .concat();
        let mounts = mountinfo(&text);
        assert_eq!(mounts.len(), 18);
        let shown = visible(&mounts);
        let hidden = mounts
            .iter()
            .zip(&shown)
            .filter(|(_, shown)| !**shown)
            .map(|(mount, _)| (mount.point.as_str(), mount.filesystem.as_str()))
            .collect::<Vec<_>>();
        assert_eq!(
            hidden,
            [
                ("/store", "ext4"),
                ("/store/models", "ext4"),
                ("/scratch", "tmpfs")
            ]
        );
        assert_eq!(
            local(&mounts, |point| point != "/etc/hosts"),
            ["/", "/a b", "/boot", "/dev/shm", "/scratch"]
        );
    }

    #[test]
    fn a_namespace_root_without_a_listed_parent_is_visible() {
        // A container: the root's parent is outside its namespace.
        let text = [
            line(500, 480, "0:90", "/", "overlay"),
            line(501, 500, "259:3", "/workspace", "ext4"),
        ]
        .concat();
        let mounts = mountinfo(&text);
        assert_eq!(visible(&mounts), [true, true]);
        assert_eq!(local(&mounts, |_| true), ["/workspace"]);
    }
}
