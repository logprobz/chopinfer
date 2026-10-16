//! Docker containers running `chopinfer serve`, found through the daemon's
//! socket.
//!
//! A container counts when Docker's process list for it shows `chopinfer
//! serve` or `chopinfer-host serve`, so any image works. The daemon runs
//! `ps` on its host side, so nothing is needed inside the container. Only
//! read-only routes are used: `/_ping`, the running-container list and each
//! container's process list.

use std::collections::HashMap;
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};
use std::path::{Path, PathBuf};

use serde_json::Value;

use super::{is_serve, listen_address, model_hint, reachable};
use crate::http::{self, DOCKER_LIMIT, Endpoint};

/// Seconds a container that did not match is checked on every scan.
const NEW_CONTAINER_SECONDS: f64 = 120.0;
/// Seconds between later checks of a container that did not match.
const RECHECK_SECONDS: f64 = 300.0;
/// Seconds between checks of a matched container the host cannot reach.
const UNREACHABLE_RECHECK_SECONDS: f64 = 60.0;
/// Seconds between attempts on a daemon that did not answer.
const RETRY_SECONDS: f64 = 60.0;
/// The process list arguments: every process, unwrapped, with its pid.
const TOP_ARGUMENTS: &str = "ps_args=-ww%20-eo%20pid%2Cargs";

/// One container running serve.
#[derive(Clone, Debug, PartialEq)]
pub struct ServeContainer {
    pub daemon: PathBuf,
    pub id: String,
    pub name: String,
    pub arguments: Vec<String>,
    pub model_hint: Option<String>,
    /// Seconds since this scanner first listed the container.
    pub seen_for: f64,
    /// Where the host reaches its listener, best first. Empty when serve
    /// listens only on the container's own loopback.
    pub addresses: Vec<SocketAddr>,
}

/// One daemon socket and what is known about it.
#[derive(Debug)]
struct Daemon {
    socket: PathBuf,
    /// `/vX.Y`, empty for a daemon that reports no version, `None` until it
    /// answers.
    prefix: Option<String>,
    retry_at: f64,
    reported: bool,
}

#[derive(Debug)]
struct Container {
    name: String,
    first_seen: f64,
    last_checked: f64,
    /// The serve command line, once matched.
    matched: Option<Vec<String>>,
    /// Matched, with no address the host can reach at the last scan.
    unreachable: bool,
    loopback_reported: bool,
}

/// Lists serve containers on every usable daemon.
#[derive(Debug, Default)]
pub struct DockerScanner {
    daemons: Vec<Daemon>,
    containers: HashMap<(PathBuf, String), Container>,
    /// Whether the sockets are the implicit candidates, which most hosts
    /// lack some of: a missing one is then not worth an event.
    implicit: bool,
}

impl DockerScanner {
    /// The daemons to use: only `DOCKER_HOST` when it names a Unix socket;
    /// otherwise the rootless and system sockets, each used when it
    /// answers. Another `DOCKER_HOST` scheme leaves Docker off, with the
    /// reason returned.
    pub fn from_environment(
        docker_host: Option<&str>,
        runtime_directory: Option<&str>,
    ) -> Result<Self, String> {
        let explicit = docker_host.filter(|host| !host.is_empty());
        let sockets = match explicit {
            Some(host) => match host.strip_prefix("unix://") {
                Some(path) => vec![PathBuf::from(path)],
                None => {
                    return Err(format!(
                        "docker: DOCKER_HOST={host} is not a unix:// socket; container discovery is off"
                    ));
                }
            },
            None => runtime_directory
                .filter(|directory| !directory.is_empty())
                .map(|directory| Path::new(directory).join("docker.sock"))
                .into_iter()
                .chain([PathBuf::from("/var/run/docker.sock")])
                .collect(),
        };
        Ok(Self {
            implicit: explicit.is_none(),
            ..Self::with_sockets(sockets)
        })
    }

    pub fn with_sockets(sockets: Vec<PathBuf>) -> Self {
        Self {
            daemons: sockets
                .into_iter()
                .map(|socket| Daemon {
                    socket,
                    prefix: None,
                    retry_at: f64::NEG_INFINITY,
                    reported: false,
                })
                .collect(),
            containers: HashMap::new(),
            implicit: false,
        }
    }

    /// Lists serve containers at time `now` (seconds), adding notes worth
    /// one event each to `events`.
    pub fn scan(&mut self, now: f64, events: &mut Vec<String>) -> Vec<ServeContainer> {
        let mut found = Vec::new();
        for index in 0..self.daemons.len() {
            if !self.connect(index, now, events) {
                continue;
            }
            let daemon = &self.daemons[index];
            let socket = daemon.socket.clone();
            let prefix = daemon.prefix.clone().unwrap_or_default();
            let endpoint = Endpoint::Unix(socket.clone());
            let listed = match get_json(&endpoint, &format!("{prefix}/containers/json")) {
                Ok(Value::Array(listed)) => listed,
                result => {
                    let daemon = &mut self.daemons[index];
                    daemon.prefix = None;
                    daemon.retry_at = now + RETRY_SECONDS;
                    let reason = match result {
                        Err(error) => error,
                        Ok(_) => "the container list is not an array".to_owned(),
                    };
                    events.push(format!(
                        "docker: {} stopped answering ({reason}); retrying every 60s",
                        socket.display()
                    ));
                    self.containers.retain(|(daemon, _), _| *daemon != socket);
                    continue;
                }
            };
            let mut present = Vec::new();
            for listing in &listed {
                let Some(id) = listing["Id"].as_str() else {
                    continue;
                };
                present.push(id.to_owned());
                let key = (socket.clone(), id.to_owned());
                let container = self.containers.entry(key).or_insert_with(|| Container {
                    name: container_name(listing).unwrap_or_else(|| id.chars().take(12).collect()),
                    first_seen: now,
                    last_checked: f64::NEG_INFINITY,
                    matched: None,
                    unreachable: false,
                    loopback_reported: false,
                });
                let unreachable = container
                    .matched
                    .as_ref()
                    .is_some_and(|arguments| addresses(listing, arguments).is_empty());
                let due = if unreachable {
                    // No poll fails to say that serve restarted on an address
                    // the host can reach, so its command is read again.
                    now - container.last_checked >= UNREACHABLE_RECHECK_SECONDS
                } else {
                    container.matched.is_none()
                        && (now - container.first_seen < NEW_CONTAINER_SECONDS
                            || now - container.last_checked >= RECHECK_SECONDS)
                };
                if due {
                    container.last_checked = now;
                    container.matched = serve_command(&endpoint, &prefix, id);
                }
                let Some(arguments) = &container.matched else {
                    continue;
                };
                let addresses = addresses(listing, arguments);
                container.unreachable = addresses.is_empty();
                if addresses.is_empty() && !container.loopback_reported {
                    container.loopback_reported = true;
                    let listen = listen_address(arguments);
                    let loopback = listen
                        .parse::<SocketAddr>()
                        .is_ok_and(|address| address.ip().is_loopback());
                    events.push(if loopback {
                        format!(
                            "docker {}: serve listens on the container's own loopback ({listen}), which the host cannot reach; listen on 0.0.0.0 and publish the port",
                            container.name
                        )
                    } else {
                        format!(
                            "docker {}: no published port or container address reaches its listener ({listen})",
                            container.name
                        )
                    });
                }
                found.push(ServeContainer {
                    daemon: socket.clone(),
                    id: id.to_owned(),
                    name: container.name.clone(),
                    arguments: arguments.clone(),
                    model_hint: model_hint(arguments),
                    seen_for: now - container.first_seen,
                    addresses,
                });
            }
            self.containers
                .retain(|(daemon, id), _| *daemon != socket || present.contains(id));
        }
        found
    }

    /// Whether a matched container the host cannot reach is due its minute
    /// reread, which a scan does whatever `--rescan` is.
    pub fn reread_due(&self, now: f64) -> bool {
        self.containers.values().any(|container| {
            container.matched.is_some()
                && container.unreachable
                && now - container.last_checked >= UNREACHABLE_RECHECK_SECONDS
        })
    }

    /// Makes `index` usable: pings it when it has not answered yet and its
    /// retry time has come. Reports a failure once per daemon, except an
    /// implicit socket that does not exist: no daemon there is the usual
    /// case.
    fn connect(&mut self, index: usize, now: f64, events: &mut Vec<String>) -> bool {
        let daemon = &mut self.daemons[index];
        if daemon.prefix.is_some() {
            return true;
        }
        if now < daemon.retry_at {
            return false;
        }
        let endpoint = Endpoint::Unix(daemon.socket.clone());
        match http::get(&endpoint, "/_ping", None, DOCKER_LIMIT) {
            Ok(response) if response.status == 200 => {
                daemon.prefix = Some(
                    response
                        .header("api-version")
                        .filter(|version| {
                            !version.is_empty()
                                && version.chars().all(|c| c.is_ascii_digit() || c == '.')
                        })
                        .map(|version| format!("/v{version}"))
                        .unwrap_or_default(),
                );
                daemon.reported = false;
                true
            }
            result => {
                daemon.retry_at = now + RETRY_SECONDS;
                let absent = self.implicit && !daemon.socket.exists();
                if !daemon.reported && !absent {
                    daemon.reported = true;
                    let reason = match result {
                        Ok(response) => format!("HTTP {}", response.status),
                        Err(error) => error.to_string(),
                    };
                    events.push(format!(
                        "docker: {} is not usable ({reason}); retrying every 60s",
                        daemon.socket.display()
                    ));
                }
                false
            }
        }
    }

    /// Forgets the matched command of a container, so the next scan reads
    /// its process list again: its serve process may have restarted on
    /// another port or with another model.
    pub fn invalidate(&mut self, daemon: &Path, id: &str) {
        if let Some(container) = self
            .containers
            .get_mut(&(daemon.to_path_buf(), id.to_owned()))
        {
            container.matched = None;
            container.last_checked = f64::NEG_INFINITY;
            container.loopback_reported = false;
        }
    }
}

fn get_json(endpoint: &Endpoint, path: &str) -> Result<Value, String> {
    let response =
        http::get(endpoint, path, None, DOCKER_LIMIT).map_err(|error| error.to_string())?;
    if response.status != 200 {
        return Err(format!("HTTP {} from {path}", response.status));
    }
    serde_json::from_slice(&response.body).map_err(|error| format!("{path}: {error}"))
}

fn container_name(listing: &Value) -> Option<String> {
    let name = listing["Names"].as_array()?.first()?.as_str()?;
    Some(name.trim_start_matches('/').to_owned())
}

/// The first serve command line in a container's process list.
fn serve_command(endpoint: &Endpoint, prefix: &str, id: &str) -> Option<Vec<String>> {
    let top = get_json(
        endpoint,
        &format!("{prefix}/containers/{id}/top?{TOP_ARGUMENTS}"),
    )
    .ok()?;
    let titles = top["Titles"].as_array()?;
    let column = titles
        .iter()
        .position(|title| matches!(title.as_str(), Some("COMMAND" | "CMD" | "ARGS")))
        .unwrap_or(titles.len().saturating_sub(1));
    top["Processes"]
        .as_array()?
        .iter()
        .filter_map(|process| process.get(column)?.as_str())
        .map(|command| {
            command
                .split_whitespace()
                .map(str::to_owned)
                .collect::<Vec<_>>()
        })
        .find(|arguments| is_serve_command(arguments))
}

/// A command whose program is `chopinfer` or `chopinfer-host` and which
/// names `serve`.
fn is_serve_command(arguments: &[String]) -> bool {
    arguments
        .first()
        .and_then(|program| Path::new(program).file_name()?.to_str())
        .is_some_and(|program| matches!(program, "chopinfer" | "chopinfer-host"))
        && is_serve(arguments)
}

/// Where the host reaches a serve container, from its listing and the
/// `--listen` of its command line.
fn addresses(listing: &Value, arguments: &[String]) -> Vec<SocketAddr> {
    let Ok(mut listen) = listen_address(arguments).parse::<SocketAddr>() else {
        return Vec::new();
    };
    // An IPv4-mapped IPv6 listener is an IPv4 one.
    if let IpAddr::V6(ip) = listen.ip()
        && let Some(ip) = ip.to_ipv4_mapped()
    {
        listen.set_ip(IpAddr::V4(ip));
    }
    if listing["HostConfig"]["NetworkMode"].as_str() == Some("host") {
        return vec![reachable(listen)];
    }
    if listen.ip().is_loopback() {
        return Vec::new();
    }
    let mut found = Vec::new();
    for port in listing["Ports"].as_array().into_iter().flatten() {
        let published = port["PublicPort"]
            .as_u64()
            .and_then(|port| u16::try_from(port).ok());
        if port["PrivatePort"].as_u64() != Some(u64::from(listen.port()))
            || port["Type"].as_str().is_some_and(|kind| kind != "tcp")
        {
            continue;
        }
        let Some(published) = published else {
            continue;
        };
        let ip = match port["IP"].as_str().unwrap_or("") {
            "" | "0.0.0.0" => IpAddr::V4(Ipv4Addr::LOCALHOST),
            "::" => IpAddr::V6(Ipv6Addr::LOCALHOST),
            other => match other.parse() {
                Ok(ip) => ip,
                Err(_) => continue,
            },
        };
        let address = SocketAddr::new(ip, published);
        if !found.contains(&address) {
            found.push(address);
        }
    }
    if found.is_empty()
        && let Some(networks) = listing["NetworkSettings"]["Networks"].as_object()
    {
        // An IPv4 listener is reached on the IPv4 address; an IPv6 one on
        // the IPv6 address, and on both when it is unspecified (dual stack).
        let v4 = !listen.is_ipv6() || listen.ip().is_unspecified();
        for network in networks.values() {
            let ips = [
                network["IPAddress"].as_str().filter(|_| v4),
                network["GlobalIPv6Address"]
                    .as_str()
                    .filter(|_| listen.is_ipv6()),
            ];
            for ip in ips.into_iter().flatten() {
                if let Ok(ip) = ip.parse::<IpAddr>() {
                    let address = SocketAddr::new(ip, listen.port());
                    if !found.contains(&address) {
                        found.push(address);
                    }
                }
            }
        }
    }
    found
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::io::{Read, Write};
    use std::os::unix::net::UnixListener;
    use std::sync::{Arc, Mutex};
    use std::thread;

    /// A fake daemon: answers `/_ping` with `version`, and container routes
    /// only with that version prefix (any prefix-free path when `version`
    /// is `None`). Records every path requested.
    pub(crate) struct FakeDaemon {
        pub path: PathBuf,
        pub state: Arc<Mutex<FakeState>>,
        directory: PathBuf,
    }

    #[derive(Default)]
    pub(crate) struct FakeState {
        pub version: Option<String>,
        pub containers: Vec<Value>,
        /// Process lists by container id.
        pub processes: HashMap<String, Vec<String>>,
        pub requests: Vec<String>,
    }

    impl FakeDaemon {
        pub(crate) fn start(label: &str, version: Option<&str>) -> Self {
            let directory = std::env::temp_dir().join(format!(
                "chopinfer-top-docker-{label}-{}",
                std::process::id()
            ));
            let _ = std::fs::remove_dir_all(&directory);
            std::fs::create_dir_all(&directory).unwrap();
            let path = directory.join("docker.sock");
            let listener = UnixListener::bind(&path).unwrap();
            let state = Arc::new(Mutex::new(FakeState {
                version: version.map(str::to_owned),
                ..FakeState::default()
            }));
            let shared = Arc::clone(&state);
            thread::spawn(move || {
                for socket in listener.incoming() {
                    let Ok(mut socket) = socket else { return };
                    let mut head = Vec::new();
                    let mut byte = [0u8; 1];
                    while !head.ends_with(b"\r\n\r\n") {
                        if socket.read(&mut byte).unwrap_or(0) == 0 {
                            break;
                        }
                        head.push(byte[0]);
                    }
                    let head = String::from_utf8_lossy(&head).into_owned();
                    let path = head.split_whitespace().nth(1).unwrap_or("/").to_owned();
                    let reply = shared.lock().unwrap().answer(&path);
                    let _ = socket.write_all(reply.as_bytes());
                }
            });
            Self {
                path,
                state,
                directory,
            }
        }
    }

    impl Drop for FakeDaemon {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.directory);
        }
    }

    impl FakeState {
        fn answer(&mut self, path: &str) -> String {
            self.requests.push(path.to_owned());
            let ok = |body: String, extra: &str| {
                format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n{extra}Content-Length: {}\r\n\r\n{body}",
                    body.len()
                )
            };
            if path == "/_ping" {
                let extra = self
                    .version
                    .as_ref()
                    .map(|version| format!("Api-Version: {version}\r\n"))
                    .unwrap_or_default();
                return ok("OK".to_owned(), &extra);
            }
            let route = match &self.version {
                Some(version) => match path.strip_prefix(&format!("/v{version}")) {
                    Some(route) => route,
                    None => {
                        let body = r#"{"message":"page not found"}"#;
                        return format!(
                            "HTTP/1.1 404 Not Found\r\nContent-Length: {}\r\n\r\n{body}",
                            body.len()
                        );
                    }
                },
                None => path,
            };
            if route == "/containers/json" {
                return ok(Value::Array(self.containers.clone()).to_string(), "");
            }
            if let Some(id) = route
                .strip_prefix("/containers/")
                .and_then(|rest| rest.split_once("/top?"))
                .map(|(id, _)| id)
            {
                let processes = self
                    .processes
                    .get(id)
                    .cloned()
                    .unwrap_or_default()
                    .into_iter()
                    .enumerate()
                    .map(|(index, command)| {
                        serde_json::json!([format!("{}", 100 + index), command])
                    })
                    .collect::<Vec<_>>();
                let body =
                    serde_json::json!({"Titles": ["PID", "COMMAND"], "Processes": processes});
                return ok(body.to_string(), "");
            }
            "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n\r\n".to_owned()
        }
    }

    pub(crate) fn container(id: &str, name: &str, network: &str, ports: Value) -> Value {
        serde_json::json!({
            "Id": id,
            "Names": [format!("/{name}")],
            "State": "running",
            "Ports": ports,
            "HostConfig": {"NetworkMode": network},
            "NetworkSettings": {"Networks": {"bridge": {"IPAddress": "172.17.0.5"}}}
        })
    }

    fn published(private: u16, public: u16) -> Value {
        serde_json::json!([
            {"IP": "0.0.0.0", "PrivatePort": private, "PublicPort": public, "Type": "tcp"},
            {"IP": "::", "PrivatePort": private, "PublicPort": public, "Type": "tcp"},
            {"PrivatePort": 9999, "Type": "tcp"}
        ])
    }

    const SERVE: &str =
        "/opt/chopinfer/bin/chopinfer serve --model /models/model --listen 0.0.0.0:8000";

    #[test]
    fn matches_containers_by_their_serve_process_with_the_daemon_version() {
        let daemon = FakeDaemon::start("match", Some("1.47"));
        {
            let mut state = daemon.state.lock().unwrap();
            state.containers = vec![
                container("aaa", "glm-tp4", "default", published(8000, 8600)),
                container("bbb", "fixture", "default", published(8000, 8601)),
                container("ccc", "postgres", "default", published(5432, 5432)),
            ];
            state.processes.insert(
                "aaa".into(),
                vec!["/bin/sh /entrypoint.sh".into(), SERVE.into()],
            );
            state.processes.insert(
                "bbb".into(),
                vec!["chopinfer-host serve --bundle /pkg --listen 0.0.0.0:8000 --served-model-name fx".into()],
            );
            state
                .processes
                .insert("ccc".into(), vec!["postgres -D /var/lib/postgresql".into()]);
        }
        let mut scanner = DockerScanner::with_sockets(vec![daemon.path.clone()]);
        let mut events = Vec::new();
        let found = scanner.scan(0.0, &mut events);
        assert_eq!(events, Vec::<String>::new());
        assert_eq!(found.len(), 2);
        assert_eq!(
            (found[0].name.as_str(), found[0].model_hint.as_deref()),
            ("glm-tp4", Some("model"))
        );
        assert_eq!(
            found[0].addresses,
            [
                "127.0.0.1:8600".parse().unwrap(),
                "[::1]:8600".parse().unwrap()
            ]
        );
        assert_eq!(
            (found[1].name.as_str(), found[1].model_hint.as_deref()),
            ("fixture", Some("fx"))
        );
        let requests = daemon.state.lock().unwrap().requests.clone();
        assert_eq!(requests[0], "/_ping");
        assert!(
            requests[1..].iter().all(|path| path.starts_with("/v1.47/")),
            "{requests:?}"
        );
        assert!(
            requests
                .iter()
                .any(|path| path.contains("ps_args=-ww%20-eo%20pid%2Cargs"))
        );
    }

    #[test]
    fn a_daemon_without_a_version_gets_unversioned_paths() {
        let daemon = FakeDaemon::start("unversioned", None);
        {
            let mut state = daemon.state.lock().unwrap();
            state.containers = vec![container("aaa", "one", "default", published(8000, 8600))];
            state.processes.insert("aaa".into(), vec![SERVE.into()]);
        }
        let mut scanner = DockerScanner::with_sockets(vec![daemon.path.clone()]);
        assert_eq!(scanner.scan(0.0, &mut Vec::new()).len(), 1);
        let requests = daemon.state.lock().unwrap().requests.clone();
        assert_eq!(requests[1], "/containers/json");
    }

    #[test]
    fn listen_addresses_follow_the_network_mode() {
        let serve = |listen: &str| {
            format!("chopinfer serve --listen {listen}")
                .split_whitespace()
                .map(str::to_owned)
                .collect::<Vec<_>>()
        };
        let host = container("a", "a", "host", Value::Null);
        assert_eq!(
            addresses(&host, &serve("0.0.0.0:9000")),
            ["127.0.0.1:9000".parse().unwrap()]
        );
        // The serve default is loopback: fine on the host network.
        let default = vec!["chopinfer".to_owned(), "serve".to_owned()];
        assert_eq!(
            addresses(&host, &default),
            ["127.0.0.1:8310".parse().unwrap()]
        );
        let bridge = container("b", "b", "default", published(8000, 8600));
        assert_eq!(
            addresses(&bridge, &serve("0.0.0.0:7000")),
            ["172.17.0.5:7000".parse().unwrap()]
        );
        assert_eq!(addresses(&bridge, &serve("[::]:8000")).len(), 2);
        assert!(addresses(&bridge, &default).is_empty());
        assert!(addresses(&bridge, &serve("127.0.0.1:8000")).is_empty());
        let mut v6 = container("c", "c", "default", Value::Null);
        v6["NetworkSettings"]["Networks"]["bridge"] =
            serde_json::json!({"IPAddress": "", "GlobalIPv6Address": "fd00::5"});
        assert_eq!(
            addresses(&v6, &serve("[::]:8000")),
            ["[fd00::5]:8000".parse().unwrap()]
        );
        assert!(addresses(&v6, &serve("0.0.0.0:8000")).is_empty());
        let mut both = container("d", "d", "default", Value::Null);
        both["NetworkSettings"]["Networks"]["bridge"]["GlobalIPv6Address"] = Value::from("fd00::6");
        assert_eq!(
            addresses(&both, &serve("[::]:8000")),
            [
                "172.17.0.5:8000".parse().unwrap(),
                "[fd00::6]:8000".parse().unwrap()
            ]
        );
        assert_eq!(
            addresses(&both, &serve("0.0.0.0:8000")),
            ["172.17.0.5:8000".parse().unwrap()]
        );
        // IPv4-mapped listeners are IPv4 ones.
        assert_eq!(
            addresses(&both, &serve("[::ffff:172.17.0.5]:8000")),
            ["172.17.0.5:8000".parse().unwrap()]
        );
        assert!(addresses(&both, &serve("[::ffff:127.0.0.1]:8000")).is_empty());
    }

    #[test]
    fn rechecks_unmatched_containers_on_a_schedule_and_forgets_removed_ones() {
        let daemon = FakeDaemon::start("schedule", Some("1.47"));
        {
            let mut state = daemon.state.lock().unwrap();
            state.containers = vec![container("aaa", "slow", "default", published(8000, 8600))];
            state
                .processes
                .insert("aaa".into(), vec!["/bin/sh /entrypoint.sh".into()]);
        }
        let tops = |daemon: &FakeDaemon| {
            daemon
                .state
                .lock()
                .unwrap()
                .requests
                .iter()
                .filter(|path| path.contains("/top?"))
                .count()
        };
        let mut scanner = DockerScanner::with_sockets(vec![daemon.path.clone()]);
        let mut events = Vec::new();
        // Checked on every scan in its first two minutes.
        for now in [0.0, 15.0, 100.0] {
            assert!(scanner.scan(now, &mut events).is_empty());
        }
        assert_eq!(tops(&daemon), 3);
        // After that, five minutes after the last check.
        for now in [130.0, 145.0] {
            assert!(scanner.scan(now, &mut events).is_empty());
        }
        assert_eq!(tops(&daemon), 3);
        daemon
            .state
            .lock()
            .unwrap()
            .processes
            .insert("aaa".into(), vec![SERVE.into()]);
        assert!(scanner.scan(300.0, &mut events).is_empty());
        assert_eq!(scanner.scan(400.0, &mut events).len(), 1);
        // A matched container is not checked again until invalidated.
        assert_eq!(scanner.scan(415.0, &mut events).len(), 1);
        assert_eq!(tops(&daemon), 4);
        // A container that leaves the list is forgotten; one that appears
        // is checked at once.
        {
            let mut state = daemon.state.lock().unwrap();
            state.containers = vec![container("ddd", "new", "default", published(8000, 8601))];
            state.processes.insert("ddd".into(), vec![SERVE.into()]);
        }
        let found = scanner.scan(430.0, &mut events);
        assert_eq!(
            found.iter().map(|c| c.name.as_str()).collect::<Vec<_>>(),
            ["new"]
        );
        assert_eq!(scanner.containers.len(), 1);
    }

    #[test]
    fn an_invalidated_match_rereads_the_process_list() {
        let daemon = FakeDaemon::start("invalidate", Some("1.47"));
        {
            let mut state = daemon.state.lock().unwrap();
            state.containers = vec![container("aaa", "sup", "default", published(8000, 8600))];
            state.processes.insert("aaa".into(), vec![SERVE.into()]);
        }
        let mut scanner = DockerScanner::with_sockets(vec![daemon.path.clone()]);
        let found = scanner.scan(0.0, &mut Vec::new());
        assert_eq!(found[0].addresses[0], "127.0.0.1:8600".parse().unwrap());
        {
            let mut state = daemon.state.lock().unwrap();
            state.containers = vec![container("aaa", "sup", "default", published(9000, 8700))];
            state.processes.insert(
                "aaa".into(),
                vec!["chopinfer serve --model /models/other --listen 0.0.0.0:9000".into()],
            );
        }
        // Still the cached command until the instance is reported lost: its
        // port 8000 is no longer published, so the container address.
        assert_eq!(
            scanner.scan(200.0, &mut Vec::new())[0].addresses,
            ["172.17.0.5:8000".parse().unwrap()]
        );
        scanner.invalidate(&daemon.path, "aaa");
        let found = scanner.scan(201.0, &mut Vec::new());
        assert_eq!(found[0].addresses[0], "127.0.0.1:8700".parse().unwrap());
        assert_eq!(found[0].model_hint.as_deref(), Some("other"));
    }

    #[test]
    fn a_loopback_listener_is_reported_once() {
        let daemon = FakeDaemon::start("loopback", Some("1.47"));
        {
            let mut state = daemon.state.lock().unwrap();
            state.containers = vec![container("aaa", "inner", "default", published(8310, 8310))];
            state
                .processes
                .insert("aaa".into(), vec!["chopinfer serve --model /m".into()]);
        }
        let mut scanner = DockerScanner::with_sockets(vec![daemon.path.clone()]);
        let mut events = Vec::new();
        assert!(scanner.scan(0.0, &mut events)[0].addresses.is_empty());
        scanner.scan(15.0, &mut events);
        assert_eq!(events.len(), 1);
        assert!(
            events[0].contains("own loopback (127.0.0.1:8310)"),
            "{events:?}"
        );
        // Serve restarts listening on every address. Nothing polls the
        // unreachable row to notice, so its command is read again a minute
        // after the last read, which asks for a scan whatever --rescan is.
        assert!(!scanner.reread_due(59.0));
        assert!(scanner.reread_due(60.0));
        daemon.state.lock().unwrap().processes.insert(
            "aaa".into(),
            vec!["chopinfer serve --model /m --listen 0.0.0.0:8310".into()],
        );
        assert!(scanner.scan(45.0, &mut events)[0].addresses.is_empty());
        assert_eq!(
            scanner.scan(60.0, &mut events)[0].addresses[0],
            "127.0.0.1:8310".parse().unwrap()
        );
        let tops = daemon
            .state
            .lock()
            .unwrap()
            .requests
            .iter()
            .filter(|path| path.contains("/top?"))
            .count();
        assert_eq!(tops, 2);
        assert!(!scanner.reread_due(1000.0));
    }

    #[test]
    fn missing_daemons_are_retried_and_both_implicit_sockets_are_used() {
        let directory =
            std::env::temp_dir().join(format!("chopinfer-top-runtime-{}", std::process::id()));
        std::fs::create_dir_all(&directory).unwrap();
        let system = FakeDaemon::start("system", Some("1.55"));
        {
            let mut state = system.state.lock().unwrap();
            state.containers = vec![container("aaa", "sys", "default", published(8000, 8600))];
            state.processes.insert("aaa".into(), vec![SERVE.into()]);
        }
        // XDG_RUNTIME_DIR set without a rootless daemon.
        let rootless = directory.join("docker.sock");
        let mut scanner = DockerScanner::with_sockets(vec![rootless.clone(), system.path.clone()]);
        scanner.implicit = true;
        let mut events = Vec::new();
        let found = scanner.scan(0.0, &mut events);
        assert_eq!(
            found.iter().map(|c| c.name.as_str()).collect::<Vec<_>>(),
            ["sys"]
        );
        // A missing implicit socket is the usual case, not an event.
        assert_eq!(events, Vec::<String>::new());
        scanner.scan(30.0, &mut events);
        assert!(events.is_empty());
        // A missing socket someone named is worth one.
        let mut named = DockerScanner::with_sockets(vec![rootless.clone()]);
        named.scan(0.0, &mut events);
        named.scan(70.0, &mut events);
        assert_eq!(events.len(), 1);
        assert!(events[0].contains("is not usable"), "{events:?}");
        events.clear();
        // The rootless daemon starts; it is used from the next retry.
        let late = FakeDaemon::start("rootless", Some("1.47"));
        {
            let mut state = late.state.lock().unwrap();
            state.containers = vec![container("bbb", "user", "default", published(8000, 8700))];
            state.processes.insert("bbb".into(), vec![SERVE.into()]);
        }
        scanner.daemons[0].socket = late.path.clone();
        assert_eq!(scanner.scan(59.0, &mut events).len(), 1);
        let found = scanner.scan(61.0, &mut events);
        assert_eq!(
            found.iter().map(|c| c.name.as_str()).collect::<Vec<_>>(),
            ["user", "sys"]
        );
        std::fs::remove_dir_all(&directory).unwrap();
    }

    #[test]
    fn docker_host_selects_one_socket_or_turns_discovery_off() {
        let scanner = DockerScanner::from_environment(
            Some("unix:///run/user/7/docker.sock"),
            Some("/run/user/7"),
        )
        .unwrap();
        assert_eq!(
            scanner
                .daemons
                .iter()
                .map(|d| d.socket.clone())
                .collect::<Vec<_>>(),
            [PathBuf::from("/run/user/7/docker.sock")]
        );
        assert!(!scanner.implicit);
        let scanner = DockerScanner::from_environment(None, Some("/run/user/7")).unwrap();
        assert!(scanner.implicit);
        assert_eq!(
            scanner
                .daemons
                .iter()
                .map(|d| d.socket.clone())
                .collect::<Vec<_>>(),
            [
                PathBuf::from("/run/user/7/docker.sock"),
                PathBuf::from("/var/run/docker.sock")
            ]
        );
        let scanner = DockerScanner::from_environment(Some(""), None).unwrap();
        assert_eq!(scanner.daemons.len(), 1);
        let error = DockerScanner::from_environment(Some("tcp://10.0.0.2:2375"), None).unwrap_err();
        assert!(error.contains("not a unix:// socket"), "{error}");
    }
}
