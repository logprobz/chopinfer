//! Finding serve instances: processes of this user and Docker containers.
//!
//! Discovery only has to find new instances. A restart, or a different
//! model at an address already known, shows up through that instance's
//! poller within one poll. So a full scan runs at startup, before the first
//! frame, and then every `--rescan` seconds; a lost or restarted instance
//! asks for one sooner, and a process that is still loading has only its
//! own sockets checked every two seconds.

pub mod docker;
pub mod proc;

use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};
use std::path::PathBuf;
use std::sync::mpsc::{Receiver, RecvTimeoutError, Sender};
use std::thread;
use std::time::{Duration, Instant};

use docker::{DockerScanner, ServeContainer};
use proc::{ProcScanner, ServeProcess};

/// serve's default `--listen`.
pub const DEFAULT_LISTEN: &str = "127.0.0.1:8310";
/// Seconds between checks of a loading process's sockets.
const LOADING_SECONDS: f64 = 2.0;
/// Seconds a requested rescan waits after the previous scan.
const NUDGE_GAP_SECONDS: f64 = 5.0;
/// How often the discovery thread looks at its schedule.
const TICK: Duration = Duration::from_millis(250);

/// Whether a command line runs serve: the CLI takes its command from the
/// first argument, so an option value such as `--model serve` is not one.
pub fn is_serve(arguments: &[String]) -> bool {
    arguments.get(1).is_some_and(|command| command == "serve")
}

/// The value of `--name VALUE` or `--name=VALUE`.
fn option<'a>(arguments: &'a [String], name: &str) -> Option<&'a str> {
    let prefix = format!("{name}=");
    arguments.iter().enumerate().find_map(|(index, argument)| {
        if argument == name {
            arguments.get(index + 1).map(String::as_str)
        } else {
            argument.strip_prefix(&prefix)
        }
    })
}

/// What a command line says it serves: the first `--served-model-name`,
/// else the repository name of a Hugging Face cache snapshot
/// (`models--org--Name/snapshots/REVISION`), else the last path component
/// of `--model` or `--bundle`.
pub fn model_hint(arguments: &[String]) -> Option<String> {
    if let Some(name) =
        option(arguments, "--served-model-name").filter(|name| !name.starts_with('-'))
    {
        return Some(name.to_owned());
    }
    ["--model", "--bundle"].iter().find_map(|flag| {
        let path = option(arguments, flag)?.trim_end_matches('/');
        if let Some(repository) = path
            .split('/')
            .rev()
            .find_map(|part| part.strip_prefix("models--"))
            .and_then(|repository| repository.rsplit("--").next())
            .filter(|name| !name.is_empty())
        {
            return Some(repository.to_owned());
        }
        let name = path.rsplit('/').next()?;
        (!name.is_empty()).then(|| name.to_owned())
    })
}

/// The `--listen` address of a serve command line, or serve's default.
pub fn listen_address(arguments: &[String]) -> String {
    option(arguments, "--listen")
        .unwrap_or(DEFAULT_LISTEN)
        .to_owned()
}

/// Where this host reaches a listener: a wildcard bind is reached on
/// loopback.
pub fn reachable(address: SocketAddr) -> SocketAddr {
    match address.ip() {
        IpAddr::V4(ip) if ip.is_unspecified() => {
            SocketAddr::new(Ipv4Addr::LOCALHOST.into(), address.port())
        }
        IpAddr::V6(ip) if ip.is_unspecified() => {
            SocketAddr::new(Ipv6Addr::LOCALHOST.into(), address.port())
        }
        _ => address,
    }
}

/// What names a discovered instance across scans.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Key {
    /// The pid and start time, so a reused pid is a new process.
    Process {
        pid: u32,
        start_ticks: u64,
    },
    Container {
        daemon: PathBuf,
        id: String,
    },
}

/// One discovered instance.
#[derive(Clone, Debug, PartialEq)]
pub enum Found {
    Process(ServeProcess),
    Container(ServeContainer),
}

impl Found {
    pub fn key(&self) -> Key {
        match self {
            Self::Process(process) => Key::Process {
                pid: process.pid,
                start_ticks: process.start_ticks,
            },
            Self::Container(container) => Key::Container {
                daemon: container.daemon.clone(),
                id: container.id.clone(),
            },
        }
    }
}

/// When scans run, in seconds on one monotonic clock.
#[derive(Clone, Debug)]
pub struct Schedule {
    rescan: f64,
    next_full: f64,
    last_full: f64,
    requested: bool,
    next_loading: f64,
}

impl Schedule {
    /// A full scan at once, then every `rescan` seconds.
    pub fn new(rescan: f64) -> Self {
        Self {
            rescan,
            next_full: f64::NEG_INFINITY,
            last_full: f64::NEG_INFINITY,
            requested: false,
            next_loading: f64::NEG_INFINITY,
        }
    }

    /// Whether a full scan is due: its interval passed, or one was
    /// requested and the previous scan is at least five seconds old.
    pub fn full_due(&self, now: f64) -> bool {
        now >= self.next_full || (self.requested && now >= self.last_full + NUDGE_GAP_SECONDS)
    }

    pub fn ran_full(&mut self, now: f64) {
        self.last_full = now;
        self.next_full = now + self.rescan;
        self.requested = false;
    }

    /// Asks for a scan soon, because an instance was lost or restarted.
    pub fn request(&mut self) {
        self.requested = true;
    }

    /// Runs a full scan at the next tick, as the `r` key does.
    pub fn force(&mut self) {
        self.next_full = f64::NEG_INFINITY;
    }

    /// Whether loading processes' sockets are due a check; if so, the
    /// next one is two seconds later.
    pub fn loading_due(&mut self, now: f64) -> bool {
        if now < self.next_loading {
            return false;
        }
        self.next_loading = now + LOADING_SECONDS;
        true
    }
}

/// The scanners in use.
#[derive(Debug)]
pub struct Discovery {
    processes: Option<ProcScanner>,
    containers: Option<DockerScanner>,
    /// Processes that had no listener at the last look, by pid and start
    /// ticks.
    loading: Vec<(u32, u64)>,
}

impl Discovery {
    pub fn new(processes: Option<ProcScanner>, containers: Option<DockerScanner>) -> Self {
        Self {
            processes,
            containers,
            loading: Vec::new(),
        }
    }

    /// Every instance found now, at `now` seconds; notes worth an event go
    /// to `events`.
    pub fn scan(&mut self, now: f64, events: &mut Vec<String>) -> Vec<Found> {
        let mut found = Vec::new();
        if let Some(scanner) = &self.processes {
            let processes = scanner.scan();
            self.loading = processes
                .iter()
                .filter(|process| process.listening.is_empty())
                .map(|process| (process.pid, process.start_ticks))
                .collect();
            found.extend(processes.into_iter().map(Found::Process));
        }
        if let Some(scanner) = &mut self.containers {
            found.extend(scanner.scan(now, events).into_iter().map(Found::Container));
        }
        found
    }

    /// Loading processes that now listen, with their addresses. A process
    /// that exited, or whose pid another process took, is dropped here; the
    /// next full scan removes it.
    pub fn listening(&mut self) -> Vec<(u32, u64, Vec<SocketAddr>)> {
        let Some(scanner) = &self.processes else {
            return Vec::new();
        };
        let mut opened = Vec::new();
        self.loading.retain(
            |&(pid, start_ticks)| match scanner.recheck(pid, start_ticks) {
                Some(addresses) if addresses.is_empty() => true,
                Some(addresses) => {
                    opened.push((pid, start_ticks, addresses));
                    false
                }
                None => false,
            },
        );
        opened
    }

    /// Whether an unreachable container is due its process list reread.
    pub fn reread_due(&self, now: f64) -> bool {
        self.containers
            .as_ref()
            .is_some_and(|scanner| scanner.reread_due(now))
    }

    /// Makes the next scan read a container's process list again.
    pub fn invalidate(&mut self, key: &Key) {
        if let (Key::Container { daemon, id }, Some(scanner)) = (key, &mut self.containers) {
            scanner.invalidate(daemon, id);
        }
    }
}

/// What the discovery thread reports.
#[derive(Debug)]
pub enum Report {
    /// A full scan: every instance found, and notes worth an event.
    Scan {
        found: Vec<Found>,
        events: Vec<String>,
    },
    /// A loading process now listens.
    Listening {
        pid: u32,
        start_ticks: u64,
        addresses: Vec<SocketAddr>,
    },
}

/// What the discovery thread is asked to do.
#[derive(Debug)]
pub enum Command {
    /// Scan soon: an instance was lost or restarted.
    Request,
    /// Scan now.
    Force,
    /// Read a container's process list again at the next scan.
    Invalidate(Key),
}

/// Runs `discovery` on its own thread until `commands` closes, timing its
/// scans with `schedule` on the clock that started at `origin`.
pub fn spawn(
    mut discovery: Discovery,
    mut schedule: Schedule,
    origin: Instant,
    commands: Receiver<Command>,
    reports: Sender<Report>,
) -> thread::JoinHandle<()> {
    thread::spawn(move || {
        loop {
            match commands.recv_timeout(TICK) {
                Ok(Command::Request) => schedule.request(),
                Ok(Command::Force) => schedule.force(),
                Ok(Command::Invalidate(key)) => discovery.invalidate(&key),
                Err(RecvTimeoutError::Timeout) => {}
                Err(RecvTimeoutError::Disconnected) => return,
            }
            let now = origin.elapsed().as_secs_f64();
            if schedule.full_due(now) || discovery.reread_due(now) {
                let mut events = Vec::new();
                let found = discovery.scan(now, &mut events);
                schedule.ran_full(origin.elapsed().as_secs_f64());
                if reports.send(Report::Scan { found, events }).is_err() {
                    return;
                }
            } else if schedule.loading_due(now) {
                for (pid, start_ticks, addresses) in discovery.listening() {
                    let report = Report::Listening {
                        pid,
                        start_ticks,
                        addresses,
                    };
                    if reports.send(report).is_err() {
                        return;
                    }
                }
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn arguments(text: &str) -> Vec<String> {
        text.split_whitespace().map(str::to_owned).collect()
    }

    #[test]
    fn reads_serve_command_lines() {
        assert!(is_serve(&arguments("chopinfer serve --model m")));
        assert!(!is_serve(&arguments("serve --version")));
        assert!(!is_serve(&arguments("chopinfer verify --model m")));
        assert!(!is_serve(&arguments("chopinfer verify --model serve")));
        assert_eq!(
            model_hint(&arguments("chopinfer serve --model /m/Qwen3.8-Flash/")).as_deref(),
            Some("Qwen3.8-Flash")
        );
        assert_eq!(
            model_hint(&arguments("chopinfer serve --bundle=pkg/mimo26")).as_deref(),
            Some("mimo26")
        );
        assert_eq!(
            model_hint(&arguments(
                "chopinfer serve --model m --served-model-name glm extra"
            ))
            .as_deref(),
            Some("glm")
        );
        assert_eq!(
            model_hint(&arguments(
                "chopinfer serve --model /hub/models--zai-org--GLM-5.3-Flash/snapshots/eb9eb208 --tensor-parallel-size 4"
            ))
            .as_deref(),
            Some("GLM-5.3-Flash")
        );
        assert_eq!(model_hint(&arguments("chopinfer serve")), None);
        assert_eq!(
            listen_address(&arguments("chopinfer serve --listen=0.0.0.0:9000")),
            "0.0.0.0:9000"
        );
        assert_eq!(
            listen_address(&arguments("chopinfer serve")),
            DEFAULT_LISTEN
        );
        assert_eq!(
            reachable("0.0.0.0:1".parse().unwrap()),
            "127.0.0.1:1".parse().unwrap()
        );
        assert_eq!(
            reachable("[::]:1".parse().unwrap()),
            "[::1]:1".parse().unwrap()
        );
        assert_eq!(
            reachable("10.0.0.2:1".parse().unwrap()),
            "10.0.0.2:1".parse().unwrap()
        );
    }

    #[test]
    fn scans_at_start_then_on_the_interval_with_rate_limited_requests() {
        let mut schedule = Schedule::new(15.0);
        assert!(schedule.full_due(0.0));
        schedule.ran_full(0.0);
        assert!(!schedule.full_due(14.9));
        assert!(schedule.full_due(15.0));
        schedule.ran_full(15.0);
        // A lost instance asks for a scan: at most one every five seconds.
        schedule.request();
        assert!(!schedule.full_due(17.0));
        assert!(schedule.full_due(20.0));
        schedule.ran_full(20.0);
        assert!(!schedule.full_due(21.0));
        schedule.request();
        schedule.request();
        assert!(schedule.full_due(25.0));
        schedule.ran_full(25.0);
        assert!(!schedule.full_due(30.0));
        // The r key does not wait.
        schedule.force();
        assert!(schedule.full_due(25.1));
        // Loading sockets every two seconds.
        assert!(schedule.loading_due(0.0));
        assert!(!schedule.loading_due(1.9));
        assert!(schedule.loading_due(2.0));
    }
}
