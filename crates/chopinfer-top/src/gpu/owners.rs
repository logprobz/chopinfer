//! Which instance a process on a GPU is, from `/proc`.
//!
//! Each new pid is read once: its start time, so that a reused pid is read
//! again; the Docker container named in its cgroup; whether it shares
//! chopinfer-top's network namespace; and serve's listen address when it is
//! chopinfer running serve. Nothing is taken from Docker's process lists,
//! which are read again only when a container's row asks, so a serve that
//! restarts inside a container is found again under its new pid.

use std::collections::HashMap;
use std::fs;
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};
use std::path::{Path, PathBuf};

use super::Process;
use crate::discover::proc::{executable, is_chopinfer, read_cmdline, start_ticks};
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
    /// Whether it shares chopinfer-top's network namespace; `None` when
    /// that cannot be read, as for another user's process.
    pub host_network: Option<bool>,
    /// serve's `--listen`, when it is chopinfer running serve.
    pub listen: Option<String>,
}

impl Owner {
    /// Whether its listener is reached on this host's addresses: it shares
    /// chopinfer-top's network namespace, or, where that cannot be read, it
    /// is in no container, as a systemd unit of another user is not.
    fn on_host_network(&self) -> bool {
        self.host_network
            .unwrap_or_else(|| self.container.is_none())
    }
}

/// The owners read so far, by pid.
#[derive(Debug)]
pub struct Owners {
    root: PathBuf,
    /// chopinfer-top's network namespace.
    namespace: Option<PathBuf>,
    known: HashMap<u32, Owner>,
}

impl Owners {
    /// Reads processes under `root`, normally `/proc`.
    pub fn new(root: impl Into<PathBuf>) -> Self {
        let root = root.into();
        let namespace = fs::read_link(root.join("self/ns/net")).ok();
        Self {
            root,
            namespace,
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
        let serve = runs_chopinfer(&directory, &arguments) && is_serve(&arguments);
        let owner = Owner {
            start_ticks: ticks,
            name: fs::read_to_string(directory.join("comm"))
                .ok()
                .map(|name| sanitize(name.trim_end())),
            container: fs::read_to_string(directory.join("cgroup"))
                .ok()
                .and_then(|cgroup| container_id(&cgroup)),
            host_network: self.namespace.as_ref().and_then(|namespace| {
                fs::read_link(directory.join("ns/net"))
                    .ok()
                    .map(|seen| seen == *namespace)
            }),
            listen: serve.then(|| listen_address(&arguments)),
        };
        self.known.insert(pid, owner.clone());
        owner
    }

    /// Forgets every pid but `pids`, the ones still on a GPU.
    pub fn keep(&mut self, pids: &[u32]) {
        self.known.retain(|pid, _| pids.contains(pid));
    }
}

/// Whether the process in `directory` is chopinfer: its executable, else,
/// for another user's process whose executable cannot be read, its first
/// argument. Other programs have a `serve` command too.
fn runs_chopinfer(directory: &Path, arguments: &[String]) -> bool {
    executable(directory)
        .or_else(|| {
            let first = Path::new(arguments.first()?);
            Some(first.file_name()?.to_str()?.to_owned())
        })
        .is_some_and(|name| is_chopinfer(&name))
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
/// target names the listen address itself, `localhost` for 127.0.0.1 or
/// ::1, or, for a wildcard bind, any loopback address.
fn serves(target: &Target, listen: &str) -> bool {
    let Ok(address) = listen.parse::<SocketAddr>() else {
        return false;
    };
    if address.port() != target.port {
        return false;
    }
    let host = target.host.trim_start_matches('[').trim_end_matches(']');
    let listen = address.ip();
    match host.parse::<IpAddr>() {
        // The address itself, or any loopback address for a wildcard bind;
        // 127.0.0.1 and 127.0.0.2 are different listeners.
        Ok(named) => named == listen || (listen.is_unspecified() && named.is_loopback()),
        Err(_) => {
            host == "localhost"
                && (listen.is_unspecified()
                    || listen == IpAddr::from(Ipv4Addr::LOCALHOST)
                    || listen == IpAddr::from(Ipv6Addr::LOCALHOST))
        }
    }
}

/// The visible row `process` is: a process row by pid and start time or a
/// container row by container id, else, for a process on this host's
/// network, an explicit target on this host by serve's listen port. A
/// container's own port is not the host's, so it never matches a target.
/// A row shown as another resolves to that one.
pub fn row(instances: &[Instance], process: &Process) -> Option<u64> {
    let owner = &process.owner;
    let exact = instances.iter().find(|instance| match &instance.source {
        Source::Process { pid, start_ticks } => {
            *pid == process.pid && owner.start_ticks == Some(*start_ticks)
        }
        Source::Container { id, .. } => owner.container.as_deref() == Some(id.as_str()),
        Source::Target(_) => false,
    });
    let found = exact.or_else(|| {
        let listen = owner
            .listen
            .as_deref()
            .filter(|_| owner.on_host_network())?;
        instances.iter().find(
            |instance| matches!(&instance.source, Source::Target(target) if serves(target, listen)),
        )
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
            ..Owner::default()
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

    #[test]
    fn another_program_with_a_serve_command_is_not_chopinfer() {
        let root = scratch("program");
        process(&root, 30, &["ollama", "serve"], 1, "0::/\n");
        std::os::unix::fs::symlink("/usr/bin/ollama", root.join("30/exe")).unwrap();
        // Its executable cannot be read and its first argument is a path.
        process(&root, 31, &["/opt/ollama/bin/ollama", "serve"], 1, "0::/\n");
        process(
            &root,
            32,
            &["/usr/local/bin/chopinfer", "serve"],
            1,
            "0::/\n",
        );
        // An executable replaced on disk is still chopinfer.
        process(&root, 33, &["chopinfer", "serve"], 1, "0::/\n");
        std::os::unix::fs::symlink("/usr/bin/chopinfer (deleted)", root.join("33/exe")).unwrap();
        let mut owners = Owners::new(&root);
        assert_eq!(owners.resolve(30).listen, None);
        assert_eq!(owners.resolve(31).listen, None);
        assert_eq!(owners.resolve(32).listen.as_deref(), Some("127.0.0.1:8310"));
        assert_eq!(owners.resolve(33).listen.as_deref(), Some("127.0.0.1:8310"));
        // So it never takes the GPUs of a target on serve's default port.
        let mut registry = Registry::new(1.0);
        registry.add_target(Target::parse(":8310").unwrap(), 0.0);
        let ollama = gpu_process(30, owners.resolve(30));
        assert_eq!(row(&registry.instances, &ollama), None);
    }

    #[test]
    fn the_network_namespace_is_read_against_chopinfer_tops_own() {
        let root = scratch("namespace");
        fs::create_dir_all(root.join("self/ns")).unwrap();
        std::os::unix::fs::symlink("net:[1]", root.join("self/ns/net")).unwrap();
        for (pid, namespace) in [(40, Some("net:[1]")), (41, Some("net:[2]")), (42, None)] {
            process(&root, pid, &["chopinfer", "serve"], 1, "0::/\n");
            if let Some(namespace) = namespace {
                fs::create_dir_all(root.join(format!("{pid}/ns"))).unwrap();
                std::os::unix::fs::symlink(namespace, root.join(format!("{pid}/ns/net"))).unwrap();
            }
        }
        let mut owners = Owners::new(&root);
        assert_eq!(owners.resolve(40).host_network, Some(true));
        assert_eq!(owners.resolve(41).host_network, Some(false));
        assert_eq!(owners.resolve(42).host_network, None);
    }

    #[test]
    fn containers_sharing_a_port_with_a_target_keep_their_own_rows() {
        const OTHER: &str = "9e8d7c6b5a4f3e2d1c0b9a8f7e6d5c4b3a2f1e0d9c8b7a6f5e4d3c2b1a0f9e8d";
        let mut registry = Registry::new(1.0);
        // Targets come before discovered rows.
        let target = registry.add_target(Target::parse(":8310").unwrap(), 0.0);
        registry.reconcile(vec![found_container(ID)], 0.0);
        let container = registry
            .instances
            .iter()
            .find(|instance| instance.source.kind() == "docker")
            .unwrap()
            .id;
        let instances = &registry.instances;
        let owner = |container: Option<&str>, host_network| Owner {
            container: container.map(str::to_owned),
            host_network,
            listen: Some("0.0.0.0:8310".to_owned()),
            ..Owner::default()
        };
        let row_of = |owner| row(instances, &gpu_process(5, owner));
        // Inside its container, published elsewhere: its own row.
        assert_eq!(row_of(owner(Some(ID), Some(false))), Some(container));
        // Another container listening on 8310 inside is not the target...
        assert_eq!(row_of(owner(Some(OTHER), Some(false))), None);
        assert_eq!(row_of(owner(Some(OTHER), None)), None);
        // ...unless it shares the host's network.
        assert_eq!(row_of(owner(Some(OTHER), Some(true))), Some(target));
        // Another user's systemd unit: no namespace to read, no container.
        assert_eq!(row_of(owner(None, None)), Some(target));
    }

    #[test]
    fn loopback_listeners_on_one_port_are_different_servers() {
        let mut registry = Registry::new(1.0);
        let first = registry.add_target(Target::parse("127.0.0.1:8310").unwrap(), 0.0);
        let second = registry.add_target(Target::parse("127.0.0.2:8310").unwrap(), 0.0);
        let row_of = |listen: &str| {
            let owner = Owner {
                listen: Some(listen.to_owned()),
                ..Owner::default()
            };
            row(&registry.instances, &gpu_process(5, owner))
        };
        assert_eq!(row_of("127.0.0.1:8310"), Some(first));
        assert_eq!(row_of("127.0.0.2:8310"), Some(second));
        assert_eq!(row_of("127.0.0.3:8310"), None);
        // A wildcard bind answers on every loopback address.
        assert_eq!(row_of("0.0.0.0:8310"), Some(first));
        let mut named = Registry::new(1.0);
        let localhost = named.add_target(Target::parse("localhost:8310").unwrap(), 0.0);
        let row_of = |listen: &str| {
            let owner = Owner {
                listen: Some(listen.to_owned()),
                ..Owner::default()
            };
            row(&named.instances, &gpu_process(5, owner))
        };
        assert_eq!(row_of("127.0.0.1:8310"), Some(localhost));
        assert_eq!(row_of("[::1]:8310"), Some(localhost));
        assert_eq!(row_of("[::]:8310"), Some(localhost));
        assert_eq!(row_of("127.0.0.2:8310"), None);
    }
}
