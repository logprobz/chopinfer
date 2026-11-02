//! Starting and stopping the processes of a switch.
//!
//! A launch runs in a process group of its own, its output in a new log
//! file, and keeps running after choptop exits. choptop signals only two
//! kinds of process, and only with SIGTERM: one it started and has not
//! reaped, through its process group, and a serve process it recorded when
//! the switch was confirmed, through a pidfd once the record still holds.

use std::fs::{self, DirBuilder, File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::os::unix::fs::{DirBuilderExt, MetadataExt, OpenOptionsExt};
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};

use rustix::process::{Pid, PidfdFlags, Signal, kill_process_group, pidfd_open, pidfd_send_signal};

use crate::discover::is_serve;
use crate::discover::proc::{executable, is_chopinfer, read_cmdline, start_ticks};
use crate::gpu::owners::container_id;
use crate::ui::widgets::sanitize;

/// Where processes are read.
const PROC: &str = "/proc";
/// How much of the end of a log is read.
const TAIL_BYTES: u64 = 4096;

/// What identifies a process choptop may stop. It is recorded when a switch
/// is confirmed and read again just before the signal; the signal goes only
/// when both readings are the same.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Identity {
    pub pid: u32,
    pub start_ticks: u64,
    pub uid: u32,
    pub arguments: Vec<String>,
    /// The Docker container its cgroup names.
    pub container: Option<String>,
}

impl Identity {
    /// Reads process `pid`; `None` once it is gone or cannot be read.
    pub fn read(pid: u32) -> Option<Self> {
        let directory = Path::new(PROC).join(pid.to_string());
        Some(Self {
            pid,
            start_ticks: start_ticks(&directory)?,
            uid: fs::metadata(&directory).ok()?.uid(),
            arguments: read_cmdline(&directory.join("cmdline"))?,
            container: fs::read_to_string(directory.join("cgroup"))
                .ok()
                .and_then(|cgroup| container_id(&cgroup)),
        })
    }

    /// Records the serve process `pid` started at `start_ticks` for a stop,
    /// or says why choptop may not stop it: it is gone or another process,
    /// another user's, not `chopinfer serve`, or in a container.
    pub fn record(pid: u32, start_ticks: u64) -> Result<Self, String> {
        let identity = Self::read(pid).ok_or_else(|| format!("proc {pid} is gone"))?;
        if identity.start_ticks != start_ticks {
            return Err(format!("proc {pid} is another process now"));
        }
        if identity.uid != rustix::process::getuid().as_raw() {
            return Err(format!("proc {pid} belongs to another user"));
        }
        let directory = Path::new(PROC).join(pid.to_string());
        let chopinfer = executable(&directory).is_some_and(|name| is_chopinfer(&name));
        if !chopinfer || !is_serve(&identity.arguments) {
            return Err(format!("proc {pid} is not chopinfer serve"));
        }
        if let Some(container) = &identity.container {
            return Err(format!(
                "proc {pid} runs in container {}",
                &container[..container.len().min(12)]
            ));
        }
        Ok(identity)
    }

    /// Whether the process is still running, by its start time.
    pub fn running(&self) -> bool {
        alive(self.pid, self.start_ticks)
    }

    /// Sends SIGTERM through a pidfd once the process read again matches
    /// this record; a process that changed in any way is not signalled.
    pub fn terminate(&self) -> Result<(), String> {
        let pid = i32::try_from(self.pid)
            .ok()
            .and_then(Pid::from_raw)
            .ok_or_else(|| format!("{} is not a process id", self.pid))?;
        let pidfd = pidfd_open(pid, PidfdFlags::empty())
            .map_err(|error| format!("proc {} is gone: {error}", self.pid))?;
        match Self::read(self.pid) {
            Some(now) if now == *self => pidfd_send_signal(&pidfd, Signal::TERM)
                .map_err(|error| format!("cannot signal proc {}: {error}", self.pid)),
            Some(now) if now.start_ticks != self.start_ticks => Err(format!(
                "proc {} is another process now: its start time changed",
                self.pid
            )),
            Some(_) => Err(format!(
                "proc {} changed since the start was confirmed",
                self.pid
            )),
            None => Err(format!("proc {} is gone", self.pid)),
        }
    }
}

/// When process `pid` started, in clock ticks after boot.
pub fn start_ticks_of(pid: u32) -> Option<u64> {
    start_ticks(&Path::new(PROC).join(pid.to_string()))
}

/// Whether process `pid` that started at `start_ticks` still runs. A
/// zombie has exited: its GPUs and port are free, though its parent has not
/// reaped it.
pub fn alive(pid: u32, start_ticks: u64) -> bool {
    let Ok(stat) = fs::read_to_string(Path::new(PROC).join(pid.to_string()).join("stat")) else {
        return false;
    };
    let Some(end) = stat.rfind(')') else {
        return false;
    };
    let fields = stat[end + 1..].split_whitespace().collect::<Vec<_>>();
    fields
        .first()
        .is_some_and(|state| !matches!(*state, "Z" | "X"))
        && fields.get(19).and_then(|ticks| ticks.parse().ok()) == Some(start_ticks)
}

/// Sends SIGTERM to the process group of `child`, which choptop started as
/// the leader of its own group. While it is not reaped its pid cannot be
/// reused, so nothing else is checked.
pub fn terminate_group(child: &mut Child) -> Result<(), String> {
    match child.try_wait() {
        Ok(None) => kill_process_group(Pid::from_child(child), Signal::TERM)
            .map_err(|error| format!("cannot signal proc {}: {error}", child.id())),
        Ok(Some(_)) => Err(format!("proc {} has exited", child.id())),
        Err(error) => Err(format!("cannot wait for proc {}: {error}", child.id())),
    }
}

/// Makes `directory` and its parents, private to this user.
pub fn private_directory(directory: &Path) -> Result<(), String> {
    DirBuilder::new()
        .recursive(true)
        .mode(0o700)
        .create(directory)
        .map_err(|error| format!("cannot create {}: {error}", directory.display()))
}

/// A new log file in `directory` named after `stem`, unused so far.
pub fn log_path(directory: &Path, stem: &str) -> PathBuf {
    let mut path = directory.join(format!("{stem}.log"));
    let mut number = 2;
    while path.exists() {
        path = directory.join(format!("{stem}-{number}.log"));
        number += 1;
    }
    path
}

/// Starts `binary` with `arguments` and `environment` added to choptop's
/// own, in a process group of its own, as runtime-eval starts a server:
/// stdin closed, stdout and stderr to `log`, a new file readable by this
/// user only that begins with `header`.
pub fn start(
    binary: &Path,
    arguments: &[String],
    environment: &[(String, String)],
    log: &Path,
    header: &str,
) -> Result<Child, String> {
    let failed = |error: std::io::Error| format!("{}: {error}", log.display());
    let mut output = OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(log)
        .map_err(failed)?;
    output.write_all(header.as_bytes()).map_err(failed)?;
    let errors = output.try_clone().map_err(failed)?;
    Command::new(binary)
        .args(arguments)
        .envs(environment.iter().map(|(name, value)| (name, value)))
        .stdin(Stdio::null())
        .stdout(output)
        .stderr(errors)
        .process_group(0)
        .spawn()
        .map_err(|error| format!("cannot start {}: {error}", binary.display()))
}

/// The lines of the last 4 KiB of `log`, sanitized; a line the 4 KiB cut
/// through is left out.
pub fn tail(log: &Path) -> Vec<String> {
    let Ok(mut file) = File::open(log) else {
        return Vec::new();
    };
    let length = file.metadata().map_or(0, |metadata| metadata.len());
    let from = length.saturating_sub(TAIL_BYTES);
    let mut bytes = Vec::new();
    if file.seek(SeekFrom::Start(from)).is_err() || file.read_to_end(&mut bytes).is_err() {
        return Vec::new();
    }
    let text = String::from_utf8_lossy(&bytes);
    let mut lines = text.lines().collect::<Vec<_>>();
    if from > 0 && !lines.is_empty() {
        lines.remove(0);
    }
    lines
        .into_iter()
        .map(sanitize)
        .filter(|line| !line.trim().is_empty())
        .collect()
}

/// `20261102-153012`: the UTC time `unix` seconds after the epoch.
pub fn stamp(unix: u64) -> String {
    let days = (unix / 86_400) as i64;
    let seconds = unix % 86_400;
    // Days to a civil date, after Howard Hinnant's days_from_civil inverse.
    let shifted = days + 719_468;
    let era = shifted.div_euclid(146_097);
    let day_of_era = shifted.rem_euclid(146_097);
    let year_of_era =
        (day_of_era - day_of_era / 1460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let month_index = (5 * day_of_year + 2) / 153;
    let day = day_of_year - (153 * month_index + 2) / 5 + 1;
    let month = if month_index < 10 {
        month_index + 3
    } else {
        month_index - 9
    };
    let year = year_of_era + era * 400 + i64::from(month <= 2);
    format!(
        "{year:04}{month:02}{day:02}-{:02}{:02}{:02}",
        seconds / 3600,
        seconds / 60 % 60,
        seconds % 60
    )
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::time::{Duration, Instant};

    /// The start of every fake: it closes descriptors it inherited above
    /// stderr, so a fake a failed test leaves behind holds no lock of the
    /// test's runner.
    pub(crate) const CLOSE: &str =
        "for fd in 3 4 5 6 7 8 9; do eval \"exec $fd>&-\"; done 2>/dev/null\n";
    /// A loop that runs about 30 s at most, until SIGTERM ends it.
    pub(crate) const UNTIL_TERM: &str =
        "trap 'exit 0' TERM\nn=0\nwhile [ $n -lt 600 ]; do sleep 0.05; n=$((n+1)); done\n";

    /// A process a test started, whose group is sent SIGTERM and reaped
    /// when the guard drops, a failed test's included.
    pub(crate) struct Guard(pub(crate) Child);

    impl Drop for Guard {
        fn drop(&mut self) {
            let _ = terminate_group(&mut self.0);
            let _ = self.0.wait();
        }
    }

    pub(crate) fn scratch(name: &str) -> PathBuf {
        let root =
            std::env::temp_dir().join(format!("choptop-launch-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        root
    }

    /// Writes an executable file from another process, so no child another
    /// test forks meanwhile holds it open for writing.
    pub(crate) fn write_script(path: &Path, text: &str) {
        let mut child = Command::new("/bin/sh")
            .args(["-c", "cat > \"$0\" && chmod 755 \"$0\""])
            .arg(path)
            .stdin(Stdio::piped())
            .spawn()
            .unwrap();
        child
            .stdin
            .take()
            .unwrap()
            .write_all(text.as_bytes())
            .unwrap();
        assert!(child.wait().unwrap().success());
    }

    /// A process that passes for `chopinfer serve` until SIGTERM: a copy of
    /// the shell named chopinfer, running the script `serve` in `root`, in
    /// a process group of its own.
    pub(crate) fn fake_serve(root: &Path) -> Guard {
        fake_serve_running(root, UNTIL_TERM)
    }

    /// [`fake_serve`] running `body` after closing its descriptors; the
    /// first call in `root` writes the script.
    pub(crate) fn fake_serve_running(root: &Path, body: &str) -> Guard {
        let shell = root.join("chopinfer");
        if !shell.exists() {
            fs::create_dir_all(root).unwrap();
            let status = Command::new("cp")
                .arg("/bin/sh")
                .arg(&shell)
                .status()
                .unwrap();
            assert!(status.success());
            write_script(&root.join("serve"), &format!("{CLOSE}{body}"));
        }
        let child = Command::new(&shell)
            .arg("serve")
            .current_dir(root)
            .stdin(Stdio::null())
            .process_group(0)
            .spawn()
            .unwrap();
        // Until the shell has replaced the copy's arguments with its own.
        wait_for(|| {
            Identity::read(child.id()).is_some_and(|identity| identity.arguments.len() == 2)
        });
        Guard(child)
    }

    pub(crate) fn wait_for(mut done: impl FnMut() -> bool) {
        let started = Instant::now();
        while !done() {
            assert!(started.elapsed() < Duration::from_secs(10), "timed out");
            std::thread::sleep(Duration::from_millis(20));
        }
    }

    #[test]
    fn a_recorded_serve_process_is_stopped_only_while_it_is_the_same() {
        let root = scratch("identity");
        let mut guard = fake_serve(&root);
        let serve = &mut guard.0;
        let ticks = Identity::read(serve.id()).unwrap().start_ticks;
        assert!(
            Identity::record(serve.id(), ticks + 1)
                .unwrap_err()
                .contains("another process")
        );
        let identity = Identity::record(serve.id(), ticks).unwrap();
        assert!(identity.running());
        // A record of another start time is not signalled...
        let changed = Identity {
            start_ticks: ticks + 1,
            ..identity.clone()
        };
        assert!(changed.terminate().unwrap_err().contains("start time"));
        assert!(serve.try_wait().unwrap().is_none());
        // ...the record itself is.
        identity.terminate().unwrap();
        wait_for(|| serve.try_wait().unwrap().is_some());
        assert!(!identity.running());
        assert!(identity.terminate().is_err());
        // Not chopinfer serve: a shell of the test's own.
        let other = Guard(
            Command::new("/bin/sh")
                .args(["-c", &format!("{CLOSE}{UNTIL_TERM}")])
                .process_group(0)
                .spawn()
                .unwrap(),
        );
        let ticks = Identity::read(other.0.id()).unwrap().start_ticks;
        assert!(
            Identity::record(other.0.id(), ticks)
                .unwrap_err()
                .contains("not chopinfer serve")
        );
    }

    #[test]
    fn a_launch_logs_to_a_new_private_file_and_stops_with_its_group() {
        let root = scratch("group");
        let logs = root.join("state/choptop/logs");
        private_directory(&logs).unwrap();
        assert_eq!(fs::metadata(&logs).unwrap().mode() & 0o777, 0o700);
        let script = root.join("fake");
        write_script(
            &script,
            &format!(
                "#!/bin/sh\n{CLOSE}echo \"started $*\"\nsleep 30 &\necho $! > \"$(dirname \"$0\")/grandchild\"\n{UNTIL_TERM}"
            ),
        );
        let log = log_path(&logs, "qwen38-tp2-mtp-a-20261102-153012");
        let mut guard = Guard(
            start(
                &script,
                &["serve".into(), "--listen".into(), "127.0.0.1:8310".into()],
                &[("CHOPINFER_LOG_LEVEL".into(), "info".into())],
                &log,
                "# choptop launch A\n",
            )
            .unwrap(),
        );
        let child = &mut guard.0;
        assert_eq!(fs::metadata(&log).unwrap().mode() & 0o777, 0o600);
        // A second log of the same name gets a new one.
        let again = log_path(&logs, "qwen38-tp2-mtp-a-20261102-153012");
        assert!(again.to_string_lossy().ends_with("a-20261102-153012-2.log"));
        wait_for(|| root.join("grandchild").exists() && tail(&log).len() == 2);
        assert_eq!(
            tail(&log),
            [
                "# choptop launch A",
                "started serve --listen 127.0.0.1:8310"
            ]
        );
        let grandchild = fs::read_to_string(root.join("grandchild"))
            .unwrap()
            .trim()
            .parse::<u32>()
            .unwrap();
        terminate_group(child).unwrap();
        wait_for(|| child.try_wait().unwrap().is_some());
        // The whole group ends, the grandchild with it.
        wait_for(|| Identity::read(grandchild).is_none_or(|seen| seen.arguments.is_empty()));
        assert!(terminate_group(child).is_err());
        // A log that already exists is never written over.
        assert!(start(&script, &[], &[], &log, "").is_err());
    }

    #[test]
    fn the_tail_keeps_whole_lines_of_the_last_4_kib() {
        let root = scratch("tail");
        let log = root.join("log");
        let mut text = String::new();
        for number in 0..600 {
            text.push_str(&format!("line {number}\n"));
        }
        text.push_str("error: \u{1b}[31mbad\u{1b}[0m\n");
        fs::write(&log, &text).unwrap();
        let lines = tail(&log);
        assert_eq!(lines.last().unwrap(), "error: ?[31mbad?[0m");
        // Whole lines from the first after the cut.
        let first = lines[0]
            .strip_prefix("line ")
            .unwrap()
            .parse::<usize>()
            .unwrap();
        assert!(first > 0);
        for (offset, line) in lines[..lines.len() - 1].iter().enumerate() {
            assert_eq!(*line, format!("line {}", first + offset));
        }
        assert_eq!(first + lines.len() - 1, 600);
        assert!(tail(&root.join("absent")).is_empty());
    }

    #[test]
    fn stamps_are_utc() {
        assert_eq!(stamp(0), "19700101-000000");
        assert_eq!(stamp(1_793_633_412), "20261102-153012");
        assert_eq!(stamp(951_782_400), "20000229-000000");
    }
}
