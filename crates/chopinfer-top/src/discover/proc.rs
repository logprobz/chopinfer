//! Serve processes of this user, found in `/proc`.
//!
//! A process counts when it is owned by this user (every user when running
//! as root), its executable is `chopinfer` or `chopinfer-host`, its command
//! line names `serve`, and it shares this process's network namespace, so
//! an address it listens on is reachable from here. Processes in other
//! namespaces, such as containers, are left to Docker discovery. Only owned
//! processes have their executable and command line read, and only matching
//! ones their descriptors, so a scan costs a few milliseconds.

use std::collections::HashSet;
use std::fs;
use std::net::{Ipv4Addr, Ipv6Addr, SocketAddr};
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};

use super::{is_serve, model_hint, reachable};

/// Clock ticks per second in `/proc/PID/stat`: `USER_HZ`, 100 in the x86_64
/// Linux ABI.
const USER_HZ: f64 = 100.0;
/// Largest command line read.
const MAX_CMDLINE: usize = 64 * 1024;

/// One serve process.
#[derive(Clone, Debug, PartialEq)]
pub struct ServeProcess {
    pub pid: u32,
    /// The start time in clock ticks after boot. With the pid it names the
    /// process, so a reused pid is a different process.
    pub start_ticks: u64,
    /// `chopinfer` or `chopinfer-host`.
    pub executable: String,
    pub arguments: Vec<String>,
    pub model_hint: Option<String>,
    /// Seconds since the process started, when readable.
    pub age: Option<f64>,
    /// The addresses it listens on, wildcards mapped to loopback.
    pub listening: Vec<SocketAddr>,
}

/// Reads one `/proc` tree.
#[derive(Clone, Debug)]
pub struct ProcScanner {
    root: PathBuf,
    /// The owner whose processes count; `None` counts every owner.
    uid: Option<u32>,
    /// This process's network namespace; `None` when unreadable.
    namespace: Option<PathBuf>,
}

impl ProcScanner {
    /// A scanner of `root` (normally `/proc`) for the user of `root/self`.
    pub fn new(root: impl Into<PathBuf>) -> Self {
        let root = root.into();
        let uid = fs::metadata(root.join("self"))
            .map(|metadata| metadata.uid())
            .ok()
            .filter(|uid| *uid != 0);
        let namespace = fs::read_link(root.join("self/ns/net")).ok();
        Self {
            root,
            uid,
            namespace,
        }
    }

    /// Counts processes of `uid` only, or of every owner with `None`.
    pub fn with_uid(mut self, uid: Option<u32>) -> Self {
        self.uid = uid;
        self
    }

    /// Every serve process now running.
    pub fn scan(&self) -> Vec<ServeProcess> {
        let Ok(entries) = fs::read_dir(&self.root) else {
            return Vec::new();
        };
        let uptime = fs::read_to_string(self.root.join("uptime"))
            .ok()
            .and_then(|text| text.split_whitespace().next()?.parse::<f64>().ok());
        let mut found = entries
            .filter_map(Result::ok)
            .filter_map(|entry| entry.file_name().to_str()?.parse::<u32>().ok())
            .filter_map(|pid| self.process(pid, uptime))
            .collect::<Vec<_>>();
        found.sort_by_key(|process| process.pid);
        found
    }

    fn process(&self, pid: u32, uptime: Option<f64>) -> Option<ServeProcess> {
        let directory = self.root.join(pid.to_string());
        let owner = fs::metadata(&directory).ok()?.uid();
        if self.uid.is_some_and(|uid| uid != owner) {
            return None;
        }
        let executable = fs::read_link(directory.join("exe")).ok()?;
        let executable = executable.file_name()?.to_str()?;
        let executable = executable.strip_suffix(" (deleted)").unwrap_or(executable);
        if !matches!(executable, "chopinfer" | "chopinfer-host") {
            return None;
        }
        let arguments = read_cmdline(&directory.join("cmdline"))?;
        if !is_serve(&arguments) {
            return None;
        }
        if let Some(namespace) = &self.namespace
            && fs::read_link(directory.join("ns/net")).ok().as_ref() != Some(namespace)
        {
            return None;
        }
        let start_ticks = start_ticks(&directory)?;
        Some(ServeProcess {
            pid,
            start_ticks,
            executable: executable.to_owned(),
            model_hint: model_hint(&arguments),
            arguments,
            age: uptime.map(|uptime| (uptime - start_ticks as f64 / USER_HZ).max(0.0)),
            listening: self.listening(pid).unwrap_or_default(),
        })
    }

    /// The addresses a loading serve process listens on now; `None` when
    /// it is gone or `pid` is now another process, which every filter of a
    /// scan and the start time tell apart.
    pub fn recheck(&self, pid: u32, start_ticks: u64) -> Option<Vec<SocketAddr>> {
        self.process(pid, None)
            .filter(|process| process.start_ticks == start_ticks)
            .map(|process| process.listening)
    }

    /// The addresses `pid` listens on; `None` when the process is gone.
    pub fn listening(&self, pid: u32) -> Option<Vec<SocketAddr>> {
        let directory = self.root.join(pid.to_string());
        let mut sockets = HashSet::new();
        for entry in fs::read_dir(directory.join("fd"))
            .ok()?
            .filter_map(Result::ok)
        {
            let Ok(link) = fs::read_link(entry.path()) else {
                continue;
            };
            if let Some(inode) = link
                .to_str()
                .and_then(|link| link.strip_prefix("socket:["))
                .and_then(|link| link.strip_suffix(']'))
                .and_then(|inode| inode.parse::<u64>().ok())
            {
                sockets.insert(inode);
            }
        }
        let mut addresses = Vec::new();
        for table in ["net/tcp", "net/tcp6"] {
            let Ok(text) = fs::read_to_string(directory.join(table)) else {
                continue;
            };
            for (address, inode) in listening_rows(&text) {
                let address = reachable(address);
                if sockets.contains(&inode) && !addresses.contains(&address) {
                    addresses.push(address);
                }
            }
        }
        Some(addresses)
    }
}

pub(crate) fn read_cmdline(path: &Path) -> Option<Vec<String>> {
    let bytes = fs::read(path).ok()?;
    let bytes = &bytes[..bytes.len().min(MAX_CMDLINE)];
    Some(
        bytes
            .split(|byte| *byte == 0)
            .filter(|argument| !argument.is_empty())
            .map(|argument| String::from_utf8_lossy(argument).into_owned())
            .collect(),
    )
}

/// Field 22 of `/proc/PID/stat`, counted after the command name, which can
/// itself hold spaces and parentheses.
pub(crate) fn start_ticks(directory: &Path) -> Option<u64> {
    let stat = fs::read_to_string(directory.join("stat")).ok()?;
    let after = &stat[stat.rfind(')')? + 1..];
    after.split_whitespace().nth(19)?.parse().ok()
}

/// The `(address, inode)` of every LISTEN row of a `/proc/net/tcp` or
/// `tcp6` table.
pub fn listening_rows(text: &str) -> Vec<(SocketAddr, u64)> {
    text.lines()
        .skip(1)
        .filter_map(|line| {
            let fields = line.split_whitespace().collect::<Vec<_>>();
            if fields.get(3) != Some(&"0A") {
                return None;
            }
            let address = parse_address(fields.get(1)?)?;
            let inode = fields.get(9)?.parse().ok()?;
            Some((address, inode))
        })
        .collect()
}

/// Parses `0100007F:20CF` or a 32-digit IPv6 form. The kernel prints each
/// 32-bit word of the address in host byte order and the port in hex.
fn parse_address(text: &str) -> Option<SocketAddr> {
    let (address, port) = text.split_once(':')?;
    let port = u16::from_str_radix(port, 16).ok()?;
    let word = |hex: &str| u32::from_str_radix(hex, 16).ok().map(u32::to_le_bytes);
    let ip = match address.len() {
        8 => Ipv4Addr::from(word(address)?).into(),
        32 => {
            let mut bytes = [0u8; 16];
            for (index, chunk) in bytes.chunks_mut(4).enumerate() {
                chunk.copy_from_slice(&word(address.get(index * 8..index * 8 + 8)?)?);
            }
            Ipv6Addr::from(bytes).into()
        }
        _ => return None,
    };
    Some(SocketAddr::new(ip, port))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::TcpListener;
    use std::os::unix::fs::symlink;

    const TCP: &str = "  sl  local_address rem_address   st tx_queue rx_queue tr tm->when retrnsmt   uid  timeout inode
   0: 0100007F:20CF 00000000:0000 0A 00000000:00000000 00:00000000 00000000  1000        0 5001 1 0000000000000000 100 0 0 10 0
   1: 00000000:20D0 00000000:0000 0A 00000000:00000000 00:00000000 00000000  1000        0 5002 1 0000000000000000 100 0 0 10 0
   2: 0100007F:20CF 0100007F:9C40 01 00000000:00000000 00:00000000 00000000  1000        0 5003 1 0000000000000000 100 0 0 10 0
";
    const TCP6: &str = "  sl  local_address                         remote_address                        st tx_queue rx_queue tr tm->when retrnsmt   uid  timeout inode
   0: 00000000000000000000000001000000:20D1 00000000000000000000000000000000:0000 0A 00000000:00000000 00:00000000 00000000  1000        0 5004 1 0000000000000000 100 0 0 10 0
   1: 00000000000000000000000000000000:20D2 00000000000000000000000000000000:0000 0A 00000000:00000000 00:00000000 00000000  1000        0 5005 1 0000000000000000 100 0 0 10 0
";

    #[test]
    fn parses_listen_rows_of_both_tables() {
        assert_eq!(
            listening_rows(TCP),
            [
                ("127.0.0.1:8399".parse().unwrap(), 5001),
                ("0.0.0.0:8400".parse().unwrap(), 5002),
            ]
        );
        assert_eq!(
            listening_rows(TCP6),
            [
                ("[::1]:8401".parse().unwrap(), 5004),
                ("[::]:8402".parse().unwrap(), 5005),
            ]
        );
    }

    /// A fake `/proc` with this test's uid, a network namespace, and
    /// processes added with [`FakeProc::process`].
    struct FakeProc {
        root: PathBuf,
    }

    impl FakeProc {
        fn new(label: &str) -> Self {
            let root = std::env::temp_dir()
                .join(format!("chopinfer-top-proc-{label}-{}", std::process::id()));
            let _ = fs::remove_dir_all(&root);
            fs::create_dir_all(root.join("1/ns")).unwrap();
            symlink("net:[4026531840]", root.join("1/ns/net")).unwrap();
            symlink("1", root.join("self")).unwrap();
            fs::write(root.join("uptime"), "1000.00 4000.00\n").unwrap();
            Self { root }
        }

        /// Adds a process with `arguments`, started at second `started`,
        /// holding `sockets` and seeing `tcp` as its table.
        fn process(
            &self,
            pid: u32,
            executable: &str,
            arguments: &[&str],
            sockets: &[u64],
            namespace: &str,
        ) {
            let directory = self.root.join(pid.to_string());
            fs::create_dir_all(directory.join("fd")).unwrap();
            fs::create_dir_all(directory.join("ns")).unwrap();
            fs::create_dir_all(directory.join("net")).unwrap();
            symlink(executable, directory.join("exe")).unwrap();
            symlink(namespace, directory.join("ns/net")).unwrap();
            fs::write(directory.join("cmdline"), arguments.join("\0") + "\0").unwrap();
            fs::write(
                directory.join("stat"),
                format!(
                    "{pid} (chop (x) y) S 1 1 1 0 -1 4194560 0 0 0 0 0 0 0 0 20 0 1 0 90000 0 0\n"
                ),
            )
            .unwrap();
            fs::write(directory.join("net/tcp"), TCP).unwrap();
            fs::write(directory.join("net/tcp6"), TCP6).unwrap();
            for (index, inode) in sockets.iter().enumerate() {
                symlink(
                    format!("socket:[{inode}]"),
                    directory.join(format!("fd/{}", index + 3)),
                )
                .unwrap();
            }
            symlink("/dev/null", directory.join("fd/0")).unwrap();
        }

        fn remove(&self, pid: u32) {
            fs::remove_dir_all(self.root.join(pid.to_string())).unwrap();
        }
    }

    impl Drop for FakeProc {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.root);
        }
    }

    const NAMESPACE: &str = "net:[4026531840]";

    #[test]
    fn finds_serve_processes_their_sockets_and_hints() {
        let proc = FakeProc::new("scan");
        proc.process(
            200,
            "/opt/chopinfer/bin/chopinfer",
            &[
                "chopinfer",
                "serve",
                "--model",
                "/models/Qwen3.8-Flash/",
                "--listen",
                "0.0.0.0:8400",
            ],
            &[5002, 5004],
            NAMESPACE,
        );
        proc.process(
            201,
            "/opt/chopinfer/bin/chopinfer-host (deleted)",
            &[
                "chopinfer-host",
                "serve",
                "--bundle",
                "pkg",
                "--served-model-name",
                "fixture",
                "alias",
            ],
            &[],
            NAMESPACE,
        );
        // Not serve, not chopinfer, and another network namespace.
        proc.process(
            202,
            "/opt/chopinfer/bin/chopinfer",
            &["chopinfer", "verify", "--model", "m"],
            &[5001],
            NAMESPACE,
        );
        proc.process(
            203,
            "/usr/bin/python3",
            &["python3", "serve"],
            &[5001],
            NAMESPACE,
        );
        proc.process(
            204,
            "/opt/chopinfer/bin/chopinfer",
            &["chopinfer", "serve"],
            &[5001],
            "net:[4026532000]",
        );
        let scanner = ProcScanner::new(&proc.root);
        let found = scanner.scan();
        assert_eq!(
            found.iter().map(|process| process.pid).collect::<Vec<_>>(),
            [200, 201]
        );
        let serve = &found[0];
        assert_eq!(serve.executable, "chopinfer");
        assert_eq!(serve.model_hint.as_deref(), Some("Qwen3.8-Flash"));
        assert_eq!(serve.start_ticks, 90000);
        assert_eq!(serve.age, Some(100.0));
        assert_eq!(
            serve.listening,
            [
                "127.0.0.1:8400".parse().unwrap(),
                "[::1]:8401".parse().unwrap()
            ]
        );
        let loading = &found[1];
        assert_eq!(loading.executable, "chopinfer-host");
        assert_eq!(loading.model_hint.as_deref(), Some("fixture"));
        assert!(loading.listening.is_empty());
        // Another owner's processes are not this user's.
        let others = scanner
            .clone()
            .with_uid(Some(scanner.uid.unwrap_or(1000) + 1));
        assert!(others.scan().is_empty());
        assert_eq!(scanner.clone().with_uid(None).scan().len(), 2);
    }

    #[test]
    fn rescans_see_new_processes_exits_and_sockets_that_open_later() {
        let proc = FakeProc::new("rescan");
        let scanner = ProcScanner::new(&proc.root);
        assert!(scanner.scan().is_empty());
        proc.process(
            300,
            "/bin/chopinfer",
            &["chopinfer", "serve"],
            &[],
            NAMESPACE,
        );
        let loading = scanner.scan();
        assert_eq!(loading.len(), 1);
        assert!(loading[0].listening.is_empty());
        assert_eq!(scanner.listening(300), Some(Vec::new()));
        symlink("socket:[5001]", proc.root.join("300/fd/9")).unwrap();
        assert_eq!(
            scanner.listening(300),
            Some(vec!["127.0.0.1:8399".parse().unwrap()])
        );
        proc.process(
            301,
            "/bin/chopinfer",
            &["chopinfer", "serve"],
            &[5005],
            NAMESPACE,
        );
        assert_eq!(scanner.scan().len(), 2);
        assert_eq!(
            scanner.recheck(300, 90000),
            Some(vec!["127.0.0.1:8399".parse().unwrap()])
        );
        // The pid is reused by a process that started later.
        let stat = proc.root.join("300/stat");
        let later = fs::read_to_string(&stat)
            .unwrap()
            .replace(" 90000 ", " 95000 ");
        fs::write(&stat, later).unwrap();
        assert_eq!(scanner.recheck(300, 90000), None);
        proc.remove(300);
        assert_eq!(scanner.listening(300), None);
        assert_eq!(scanner.recheck(300, 90000), None);
        let left = scanner.scan();
        assert_eq!(
            left.iter().map(|process| process.pid).collect::<Vec<_>>(),
            [301]
        );
        assert_eq!(left[0].listening, ["[::1]:8402".parse().unwrap()]);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn finds_a_live_listener_through_its_socket_inode() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let scanner = ProcScanner::new("/proc");
        let listening = scanner.listening(std::process::id()).unwrap();
        assert!(
            listening.contains(&address),
            "{listening:?} lacks {address}"
        );
    }
}
