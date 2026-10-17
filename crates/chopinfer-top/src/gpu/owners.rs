//! Which instance a process on a GPU is, from `/proc`.
//!
//! Each new pid is read once: its start time, so that a reused pid is read
//! again; the Docker container named in its cgroup; and serve's listen
//! address when its command line runs serve. Nothing is taken from
//! Docker's process lists, which are read again only when a container's
//! row asks, so a serve that restarts inside a container is found again
//! under its new pid.

use std::collections::HashMap;
use std::fs;
use std::net::{IpAddr, SocketAddr};
use std::path::PathBuf;

use super::Process;
use crate::discover::proc::{read_cmdline, start_ticks};
use crate::discover::{is_serve, listen_address};
use crate::instances::{Instance, Source};
use crate::targets::Target;
use crate::ui::widgets::sanitize;

/// What `/proc` says about a process holding a GPU; `None` where it cannot
/// be read, as for a process in another PID namespace.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Owner {
    pub start_ticks: Option<u64>,
    /// Its command name, sanitized.
    pub name: Option<String>,
    /// The Docker container its cgroup names.
    pub container: Option<String>,
    /// serve's `--listen`, when it runs serve.
    pub listen: Option<String>,
}

/// The owners read so far, by pid.
#[derive(Debug)]
pub struct Owners {
    root: PathBuf,
    known: HashMap<u32, Owner>,
}

impl Owners {
    /// Reads processes under `root`, normally `/proc`.
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self {
            root: root.into(),
            known: HashMap::new(),
        }
    }

    /// The owner of `pid`, read again when its start time changed.
    pub fn resolve(&mut self, pid: u32) -> Owner {
        let directory = self.root.join(pid.to_string());
        let ticks = start_ticks(&directory);
        if let Some(owner) = self.known.get(&pid)
            && owner.start_ticks == ticks
        {
            return owner.clone();
        }
        let arguments = read_cmdline(&directory.join("cmdline")).unwrap_or_default();
        let owner = Owner {
            start_ticks: ticks,
            name: fs::read_to_string(directory.join("comm"))
                .ok()
                .map(|name| sanitize(name.trim_end())),
            container: fs::read_to_string(directory.join("cgroup"))
                .ok()
                .and_then(|cgroup| container_id(&cgroup)),
            listen: is_serve(&arguments).then(|| listen_address(&arguments)),
        };
        self.known.insert(pid, owner.clone());
        owner
    }

    /// Forgets every pid but `pids`, the ones still on a GPU.
    pub fn keep(&mut self, pids: &[u32]) {
        self.known.retain(|pid, _| pids.contains(pid));
    }
}

/// The 64-digit container id in a `/proc/PID/cgroup`, as Docker's cgroup
/// drivers name it: `/docker/ID`, `docker-ID.scope`, under a rootless
/// daemon's user slice too. The deepest one wins.
pub fn container_id(cgroup: &str) -> Option<String> {
    cgroup.lines().find_map(|line| {
        let path = line.splitn(3, ':').nth(2)?;
        path.split(|character: char| !character.is_ascii_hexdigit())
            .rev()
            .find(|run| run.len() == 64)
            .map(str::to_owned)
    })
}

/// Whether serve listening on `listen` is `target`: the same port, and the
/// target names this host's loopback or the listen address itself.
fn serves(target: &Target, listen: &str) -> bool {
    let Ok(address) = listen.parse::<SocketAddr>() else {
        return false;
    };
    if address.port() != target.port {
        return false;
    }
    let host = target.host.trim_start_matches('[').trim_end_matches(']');
    let named = host.parse::<IpAddr>().ok();
    let loopback = host == "localhost" || named.is_some_and(|ip| ip.is_loopback());
    (loopback && (address.ip().is_loopback() || address.ip().is_unspecified()))
        || named == Some(address.ip())
}

/// The visible row `process` is: a process row by pid and start time, a
/// container row by container id, or an explicit target on this host by
/// serve's listen port. A row shown as another resolves to that one.
pub fn row(instances: &[Instance], process: &Process) -> Option<u64> {
    let owner = &process.owner;
    let found = instances.iter().find(|instance| match &instance.source {
        Source::Process { pid, start_ticks } => {
            *pid == process.pid && owner.start_ticks == Some(*start_ticks)
        }
        Source::Container { id, .. } => owner.container.as_deref() == Some(id.as_str()),
        Source::Target(target) => owner
            .listen
            .as_deref()
            .is_some_and(|listen| serves(target, listen)),
    })?;
    Some(found.alias_of.unwrap_or(found.id))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::discover::Found;
    use crate::discover::docker::ServeContainer;
    use crate::discover::proc::ServeProcess;
    use crate::instances::Registry;
    use std::path::Path;

    const ID: &str = "3f1c9a2b4d5e6f708192a3b4c5d6e7f8091a2b3c4d5e6f708192a3b4c5d6e7f8";

    #[test]
    fn container_ids_come_from_every_cgroup_driver() {
        for cgroup in [
            format!("0::/system.slice/docker-{ID}.scope\n"),
            format!("12:memory:/docker/{ID}\n0::/docker/{ID}\n"),
            format!("0::/user.slice/user-1000.slice/app.slice/docker-{ID}.scope\n"),
        ] {
            assert_eq!(container_id(&cgroup).as_deref(), Some(ID), "{cgroup}");
        }
        assert_eq!(
            container_id("0::/user.slice/user-1000.slice/session-2.scope\n"),
            None
        );
    }

    /// A fake `/proc/PID` with a command line, a start time and a cgroup.
    fn process(root: &Path, pid: u32, arguments: &[&str], ticks: u64, cgroup: &str) {
        let directory = root.join(pid.to_string());
        fs::create_dir_all(&directory).unwrap();
        fs::write(directory.join("cmdline"), arguments.join("\0") + "\0").unwrap();
        // Field 22 is the start time; the name holds a space and a paren.
        let stat = format!("{pid} (chop (x)) S {} {ticks} 0\n", "0 ".repeat(18));
        fs::write(directory.join("stat"), stat).unwrap();
        fs::write(directory.join("comm"), "chopinfer\u{1b}\n").unwrap();
        fs::write(directory.join("cgroup"), cgroup).unwrap();
    }

    fn scratch(name: &str) -> PathBuf {
        let root = std::env::temp_dir().join(format!(
            "chopinfer-top-owners-{name}-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        root
    }

    #[test]
    fn a_reused_pid_is_read_again() {
        let root = scratch("reuse");
        process(
            &root,
            41,
            &["chopinfer", "serve", "--listen", "0.0.0.0:8311"],
            100,
            "0::/\n",
        );
        let mut owners = Owners::new(&root);
        let first = owners.resolve(41);
        assert_eq!(first.start_ticks, Some(100));
        assert_eq!(first.listen.as_deref(), Some("0.0.0.0:8311"));
        assert_eq!(first.name.as_deref(), Some("chopinfer?"));
        // The same process is not read again.
        fs::write(root.join("41/cmdline"), "python3\0").unwrap();
        assert_eq!(owners.resolve(41), first);
        // Another process under the same pid is.
        process(&root, 41, &["python3", "train.py"], 900, "0::/\n");
        let second = owners.resolve(41);
        assert_eq!((second.start_ticks, second.listen), (Some(900), None));
        owners.keep(&[]);
        assert!(owners.known.is_empty());
        // A pid that /proc does not show resolves to nothing.
        assert_eq!(owners.resolve(77), Owner::default());
    }

    fn found_process(pid: u32, start_ticks: u64) -> Found {
        Found::Process(ServeProcess {
            pid,
            start_ticks,
            executable: "chopinfer".into(),
            arguments: vec!["chopinfer".into(), "serve".into()],
            model_hint: None,
            listening: Vec::new(),
            age: None,
        })
    }

    fn found_container(id: &str) -> Found {
        Found::Container(ServeContainer {
            daemon: PathBuf::from("/run/docker.sock"),
            id: id.to_owned(),
            name: "glm-tp4".to_owned(),
            arguments: vec!["chopinfer".into(), "serve".into()],
            model_hint: None,
            seen_for: 0.0,
            addresses: Vec::new(),
        })
    }

    fn gpu_process(pid: u32, owner: Owner) -> Process {
        Process {
            pid,
            owner,
            ..Process::default()
        }
    }

    #[test]
    fn processes_map_to_rows_by_pid_container_and_listen_port() {
        let mut registry = Registry::new(1.0);
        let target = registry.add_target(Target::parse(":8431").unwrap(), 0.0);
        let remote = registry.add_target(Target::parse("192.0.2.10:8311").unwrap(), 0.0);
        registry.reconcile(vec![found_process(48211, 100), found_container(ID)], 0.0);
        let id_of = |kind: &str| {
            registry
                .instances
                .iter()
                .find(|instance| instance.source.kind() == kind)
                .unwrap()
                .id
        };
        let (proc_row, container_row) = (id_of("proc"), id_of("docker"));
        let instances = &registry.instances;
        let owner = |ticks, container: Option<&str>, listen: Option<&str>| Owner {
            start_ticks: ticks,
            container: container.map(str::to_owned),
            listen: listen.map(str::to_owned),
            name: None,
        };
        assert_eq!(
            row(instances, &gpu_process(48211, owner(Some(100), None, None))),
            Some(proc_row)
        );
        // A pid reused by another process is not that row.
        assert_eq!(
            row(instances, &gpu_process(48211, owner(Some(5), None, None))),
            None
        );
        assert_eq!(
            row(instances, &gpu_process(9, owner(Some(1), Some(ID), None))),
            Some(container_row)
        );
        // A systemd unit watched as a target, listening on every address.
        let unit = owner(Some(1), None, Some("0.0.0.0:8431"));
        assert_eq!(row(instances, &gpu_process(7, unit)), Some(target));
        let other_port = owner(Some(1), None, Some("127.0.0.1:8432"));
        assert_eq!(row(instances, &gpu_process(7, other_port)), None);
        // A remote target is never a local process, whatever its port.
        let local = owner(Some(1), None, Some("0.0.0.0:8311"));
        assert_ne!(row(instances, &gpu_process(7, local)), Some(remote));
    }

    #[test]
    fn a_restarted_serve_in_an_aliased_container_maps_through_its_new_pid() {
        let root = scratch("restart");
        let cgroup = format!("0::/system.slice/docker-{ID}.scope\n");
        process(
            &root,
            500,
            &["chopinfer", "serve", "--listen", "0.0.0.0:8600"],
            10,
            &cgroup,
        );
        let mut registry = Registry::new(1.0);
        let target = registry.add_target(Target::parse(":8600").unwrap(), 0.0);
        registry.reconcile(vec![found_container(ID)], 0.0);
        // The container row is the same server as the target, shown as it.
        let container = registry
            .instances
            .iter_mut()
            .find(|instance| instance.source.kind() == "docker")
            .unwrap();
        container.alias_of = Some(target);
        let mut owners = Owners::new(&root);
        let before = gpu_process(500, owners.resolve(500));
        assert_eq!(row(&registry.instances, &before), Some(target));
        // serve restarts inside the container under a new pid; nothing about
        // the container's row changes.
        fs::remove_dir_all(root.join("500")).unwrap();
        process(
            &root,
            612,
            &["chopinfer", "serve", "--listen", "0.0.0.0:8600"],
            99,
            &cgroup,
        );
        let after = gpu_process(612, owners.resolve(612));
        assert_eq!(row(&registry.instances, &after), Some(target));
    }
}
