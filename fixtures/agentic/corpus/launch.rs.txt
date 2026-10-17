//! `eval run --launch`: the server under test, started for the run as
//! `chopinfer serve FLAGS...` in its own process group and stopped after
//! it. Only that owned process group is signalled.

use std::fs::File;
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};
use std::os::unix::process::CommandExt as _;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use serde_json::{Value, json};

use crate::http::Client;
use crate::json::sha256_hex;

/// Names the binary `--launch` starts instead of the running `chopinfer`,
/// so a test can stand a fake `serve` in.
pub const SERVE_BINARY: &str = "CHOPINFER_EVAL_SERVE_BINARY";

/// How long a stopped server may take to exit after SIGTERM before it is
/// killed.
const GRACE: Duration = Duration::from_secs(30);

/// The URL `serve FLAGS` answers on: its `--listen` address, else
/// `default`; an unspecified address is reached over loopback.
pub fn url(flags: &[String], default: Option<&str>) -> Result<String, String> {
    let listen = match flags.iter().position(|flag| flag == "--listen") {
        Some(index) => flags
            .get(index + 1)
            .map(String::as_str)
            .ok_or("--launch: --listen needs a value")?,
        None => default.ok_or("--launch: the serve flags need --listen ADDRESS:PORT")?,
    };
    let mut address: SocketAddr = listen
        .parse()
        .map_err(|error| format!("--launch: --listen {listen}: {error}"))?;
    if address.port() == 0 {
        return Err(format!("--launch: --listen {listen}: name a port"));
    }
    if address.ip().is_unspecified() {
        address.set_ip(match address.ip() {
            IpAddr::V4(_) => Ipv4Addr::LOCALHOST.into(),
            IpAddr::V6(_) => Ipv6Addr::LOCALHOST.into(),
        });
    }
    Ok(format!("http://{address}"))
}

/// The command line `--launch` runs.
pub fn command(program: &Path, flags: &[String]) -> Vec<String> {
    [program.display().to_string(), "serve".into()]
        .into_iter()
        .chain(flags.iter().cloned())
        .collect()
}

/// A launched server.
pub struct Server {
    child: Child,
    group: Option<u32>,
    command: Vec<String>,
    log: PathBuf,
    grace: Duration,
}

impl Server {
    /// Starts `PROGRAM serve FLAGS` in its own process group, its stdout
    /// and stderr written to the new file `log`.
    pub fn start(program: &Path, flags: &[String], log: &Path) -> Result<Self, String> {
        let failed = |error: std::io::Error| format!("{}: {error}", log.display());
        let output = File::create_new(log).map_err(failed)?;
        let errors = output.try_clone().map_err(failed)?;
        let child = Command::new(program)
            .arg("serve")
            .args(flags)
            .stdin(Stdio::null())
            .stdout(output)
            .stderr(errors)
            .process_group(0)
            .spawn()
            .map_err(|error| format!("cannot launch {}: {error}", program.display()))?;
        Ok(Self {
            group: Some(child.id()),
            child,
            command: command(program, flags),
            log: log.into(),
            grace: GRACE,
        })
    }

    /// Sets how long stopping waits after SIGTERM before SIGKILL (default
    /// 30 s).
    #[must_use]
    pub fn with_grace(mut self, grace: Duration) -> Self {
        self.grace = grace;
        self
    }

    /// Waits until `/health` at `url` reports ready; the server exiting
    /// first, or `timeout` passing, is an error that quotes its log.
    pub fn wait_ready(&mut self, url: &str, timeout: Duration) -> Result<(), String> {
        let started = Instant::now();
        loop {
            if let Some(status) = self.child.try_wait().map_err(|error| error.to_string())? {
                return Err(format!(
                    "the launched server exited before it was ready ({status}){}",
                    self.tail()
                ));
            }
            let left = timeout.saturating_sub(started.elapsed());
            if left.is_zero() {
                return Err(format!(
                    "the launched server was not ready within {} s{}",
                    timeout.as_secs_f64(),
                    self.tail()
                ));
            }
            let mut client = Client::new(url, left.min(Duration::from_secs(5)), 0)?;
            if let Ok(response) = client.request("GET", "/health", None)
                && response
                    .json()
                    .is_ok_and(|health| health["status"] == "ready")
            {
                return Ok(());
            }
            std::thread::sleep(left.min(Duration::from_millis(200)));
        }
    }

    /// The end of the log, for an error message.
    fn tail(&self) -> String {
        let bytes = std::fs::read(&self.log).unwrap_or_default();
        let text = String::from_utf8_lossy(&bytes[bytes.len().saturating_sub(4096)..]);
        let lines: Vec<&str> = text.lines().collect();
        format!(
            "; {} ends:\n{}",
            self.log.display(),
            lines[lines.len().saturating_sub(20)..].join("\n")
        )
    }

    fn signal(group: u32, signal: &str) -> bool {
        Command::new("kill")
            .args([signal, "--", &format!("-{group}")])
            .stderr(Stdio::null())
            .status()
            .is_ok_and(|status| status.success())
    }

    /// Releases the owned group once, even if its leader has already exited.
    fn cleanup(&mut self) -> Option<std::process::ExitStatus> {
        let group = self.group.take()?;
        Self::signal(group, "-TERM");
        let started = Instant::now();
        while Self::signal(group, "-0") && started.elapsed() < self.grace {
            let _ = self.child.try_wait();
            std::thread::sleep(Duration::from_millis(50));
        }
        if Self::signal(group, "-0") {
            Self::signal(group, "-KILL");
        }
        self.child.wait().ok()
    }

    /// Stops the server group: SIGTERM, then SIGKILL after the grace period.
    /// Returns the command, pid, log digest and the leader's exit status.
    pub fn stop(mut self) -> Value {
        let pid = self.child.id();
        let status = self.cleanup();
        let log_sha256 = std::fs::read(&self.log)
            .ok()
            .map(|bytes| sha256_hex(&bytes));
        json!({"command": self.command, "pid": pid, "log": self.log,
               "log_sha256": log_sha256, "exit": status.map(|status| status.to_string())})
    }
}

impl Drop for Server {
    /// A server not stopped (a run cut short) is not left running.
    fn drop(&mut self) {
        self.cleanup();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flags(words: &[&str]) -> Vec<String> {
        words.iter().map(|word| (*word).to_owned()).collect()
    }

    #[test]
    fn the_url_is_the_listen_address_reached_locally() {
        let default = Some("127.0.0.1:8310");
        assert_eq!(url(&[], default).unwrap(), "http://127.0.0.1:8310");
        let listen = flags(&["--bundle", "b", "--listen", "0.0.0.0:9001"]);
        assert_eq!(url(&listen, None).unwrap(), "http://127.0.0.1:9001");
        let listen = flags(&["--listen", "[::]:9002"]);
        assert_eq!(url(&listen, default).unwrap(), "http://[::1]:9002");
        let listen = flags(&["--listen", "10.0.0.5:9003"]);
        assert_eq!(url(&listen, default).unwrap(), "http://10.0.0.5:9003");
        for (words, refusal) in [
            (&[][..], "need --listen"),
            (&["--listen"][..], "needs a value"),
            (&["--listen", "9004"][..], "--listen 9004"),
            (&["--listen", "127.0.0.1:0"][..], "name a port"),
        ] {
            let error = url(&flags(words), None).unwrap_err();
            assert!(error.contains(refusal), "{error}");
        }
    }
}
