//! Read-ahead selection and an explicit, local measurement control.
use std::ffi::OsString;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU8, AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::Duration;

const CONTROL: &str = "CHOPINFER_MAPPED_ROW_CONTROL";
const ASYNC: &str = "CHOPINFER_MAPPED_ASYNC_READ_AHEAD";
const SYNC: &str = super::MAPPED_READ_AHEAD_ENV_V1;
// Resident row caching stays enabled independently of page read-ahead.
const DEFAULT_READ_AHEAD: bool = false;

#[derive(Default)]
struct Counts {
    gathers: AtomicU64,
    advice_calls: AtomicU64,
    advice_rows: AtomicU64,
}

struct Policy {
    mode: AtomicU8,
    observed: bool,
    counts: [Counts; 2],
}

impl Policy {
    fn new(mode: u8, observed: bool) -> Self {
        Self {
            mode: AtomicU8::new(mode),
            observed,
            counts: Default::default(),
        }
    }

    fn apply(&self, command: &Command) -> Result<serde_json::Value, String> {
        let mode = mode(&command.mode)?;
        if command.reset_cache {
            crate::host_rows::resident::reset_for_measurement()?;
        }
        let replay = command
            .replay_rows
            .as_deref()
            .map(crate::host_rows::resident::replay::run)
            .transpose()?;
        let released_mapping_bytes = if command.release_page_tables {
            crate::host_rows::release_page_tables_for_measurement()
                .map_err(|error| format!("release mapped row page tables: {error}"))?
        } else {
            0
        };
        if let Some(enabled) = command.miss_fetch {
            crate::host_rows::fetch::set_for_measurement(enabled);
        }
        self.mode.store(mode, Ordering::Release);
        Ok(serde_json::json!({
            "schema":"chopinfer.mapped-row-policy.v1", "status":"applied",
            "epoch":command.epoch, "mode":command.mode, "reset_cache":command.reset_cache,
            "released_mapping_bytes":released_mapping_bytes,
            "replay":replay,
            "sync":self.snapshot(0), "async":self.snapshot(1),
            "cache":crate::host_rows::resident::measurement_snapshot(),
            "miss_fetch":crate::host_rows::fetch::snapshot(),
        }))
    }

    fn enabled(&self, asynchronous: bool) -> bool {
        self.mode.load(Ordering::Acquire) & (1 << usize::from(asynchronous)) != 0
    }

    fn snapshot(&self, index: usize) -> serde_json::Value {
        let counts = &self.counts[index];
        serde_json::json!({
            "gathers":counts.gathers.load(Ordering::Relaxed),
            "advice_calls":counts.advice_calls.load(Ordering::Relaxed),
            "advice_rows":counts.advice_rows.load(Ordering::Relaxed),
        })
    }
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct Command {
    epoch: u64,
    mode: String,
    reset_cache: bool,
    #[serde(default)]
    release_page_tables: bool,
    #[serde(default)]
    replay_rows: Option<Vec<crate::host_rows::resident::replay::Replay>>,
    #[serde(default)]
    miss_fetch: Option<bool>,
}

fn mode(value: &str) -> Result<u8, String> {
    match value {
        "off" => Ok(0),
        "sync" => Ok(1),
        "async" => Ok(2),
        "both" => Ok(3),
        _ => Err("unknown mapped-row read-ahead mode".into()),
    }
}

fn configured(get: impl Fn(&str) -> Option<OsString>) -> u8 {
    let sync = get(SYNC).map_or(DEFAULT_READ_AHEAD, |value| value == "1");
    let asynchronous = get(ASYNC).map_or(sync, |value| value == "1");
    u8::from(sync) | (u8::from(asynchronous) << 1)
}

fn read_command(path: &Path) -> Result<Command, String> {
    use std::os::unix::fs::OpenOptionsExt;
    let file = std::fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
        .open(path)
        .map_err(|error| format!("open mapped-row control: {error}"))?;
    if !file
        .metadata()
        .map_err(|error| format!("stat mapped-row control: {error}"))?
        .is_file()
    {
        return Err("mapped-row control must be a regular file".into());
    }
    let mut bytes = Vec::new();
    file.take(1025)
        .read_to_end(&mut bytes)
        .map_err(|error| format!("read mapped-row control: {error}"))?;
    if bytes.len() > 1024 {
        return Err("mapped-row control exceeds 1024 bytes".into());
    }
    let command: Command =
        serde_json::from_slice(&bytes).map_err(|_| "invalid mapped-row control record")?;
    mode(&command.mode)?;
    Ok(command)
}

fn observe(policy: Arc<Policy>, path: PathBuf) {
    let mut last = None;
    let mut last_error = None;
    loop {
        let result = read_command(&path).and_then(|command| {
            if let Some(record) = advance(&policy, &command, &mut last)? {
                runtime_engine::logging::line(record.to_string());
            }
            Ok(())
        });
        match result {
            Ok(()) => last_error = None,
            Err(error) if last_error.as_ref() != Some(&error) => {
                runtime_engine::logging::line(
                    serde_json::json!({
                        "schema":"chopinfer.mapped-row-policy.v1", "status":"refused",
                        "reason":error,
                    })
                    .to_string(),
                );
                last_error = Some(error);
            }
            Err(_) => {}
        }
        std::thread::sleep(Duration::from_millis(50));
    }
}

fn advance(
    policy: &Policy,
    command: &Command,
    last: &mut Option<u64>,
) -> Result<Option<serde_json::Value>, String> {
    if last.is_some_and(|epoch| command.epoch <= epoch) {
        return Ok(None);
    }
    let record = policy.apply(command)?;
    *last = Some(command.epoch);
    Ok(Some(record))
}

fn policy() -> &'static Arc<Policy> {
    static POLICY: OnceLock<Arc<Policy>> = OnceLock::new();
    POLICY.get_or_init(|| {
        let control = std::env::var_os(CONTROL).map(PathBuf::from);
        let policy = Arc::new(Policy::new(
            configured(|name| std::env::var_os(name)),
            control.is_some(),
        ));
        if let Some(path) = control {
            let observed = Arc::clone(&policy);
            if let Err(error) = std::thread::Builder::new()
                .name("mapped-row-policy".into())
                .spawn(move || observe(observed, path))
            {
                runtime_engine::logging::line(
                    serde_json::json!({
                        "schema":"chopinfer.mapped-row-policy.v1", "status":"refused",
                        "reason":format!("start mapped-row control: {error}"),
                    })
                    .to_string(),
                );
            }
        }
        policy
    })
}

pub(super) fn enabled(asynchronous: bool) -> bool {
    policy().enabled(asynchronous)
}

pub(super) fn gather(asynchronous: bool) {
    let policy = policy();
    if policy.observed {
        policy.counts[usize::from(asynchronous)]
            .gathers
            .fetch_add(1, Ordering::Relaxed);
    }
}

pub(super) fn advise(asynchronous: bool, rows: usize) {
    let policy = policy();
    if policy.observed {
        let counts = &policy.counts[usize::from(asynchronous)];
        counts.advice_calls.fetch_add(1, Ordering::Relaxed);
        counts.advice_rows.fetch_add(rows as u64, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_off_and_each_read_ahead_path_has_an_override() {
        assert_eq!(configured(|_| None), 0);
        assert_eq!(configured(|name| (name == SYNC).then(|| "1".into())), 3);
        assert_eq!(
            configured(|name| Some(if name == SYNC { "1" } else { "0" }.into())),
            1
        );
        assert_eq!(configured(|name| (name == ASYNC).then(|| "1".into())), 2);
        for (name, expected) in [("off", 0), ("sync", 1), ("async", 2), ("both", 3)] {
            assert_eq!(mode(name).unwrap(), expected);
        }
        assert!(mode("automatic").is_err());
    }

    #[test]
    fn acknowledged_epochs_switch_only_the_selected_path_and_reject_replays() {
        let policy = Policy::new(0, true);
        let mut last = None;
        for (epoch, (name, sync, asynchronous)) in [
            ("off", false, false),
            ("sync", true, false),
            ("async", false, true),
        ]
        .into_iter()
        .enumerate()
        {
            let command = Command {
                epoch: epoch as u64,
                mode: name.into(),
                reset_cache: false,
                release_page_tables: false,
                replay_rows: None,
                miss_fetch: None,
            };
            let record = advance(&policy, &command, &mut last).unwrap().unwrap();
            assert_eq!(record["epoch"], command.epoch);
            assert_eq!(record["mode"], name);
            assert_eq!(policy.enabled(false), sync);
            assert_eq!(policy.enabled(true), asynchronous);
            assert!(advance(&policy, &command, &mut last).unwrap().is_none());
        }
        assert!(
            advance(
                &policy,
                &Command {
                    epoch: 0,
                    mode: "both".into(),
                    reset_cache: false,
                    release_page_tables: false,
                    replay_rows: None,
                    miss_fetch: None,
                },
                &mut last
            )
            .unwrap()
            .is_none()
        );
        assert!(!policy.enabled(false) && policy.enabled(true));
    }

    #[test]
    fn control_reader_refuses_symlinks_oversized_and_unknown_records() {
        use std::os::unix::fs::symlink;
        let root = std::env::temp_dir().join(format!("mapped-row-control-{}", std::process::id()));
        std::fs::create_dir_all(&root).unwrap();
        let path = root.join("control");
        std::fs::write(&path, br#"{"epoch":1,"mode":"sync","reset_cache":false}"#).unwrap();
        let command = read_command(&path).unwrap();
        assert_eq!(command.epoch, 1);
        assert!(!command.release_page_tables);
        let policy = Policy::new(0, true);
        let record = policy.apply(&command).unwrap();
        assert_eq!(record["status"], "applied");
        assert_eq!(record["released_mapping_bytes"], 0);
        assert_eq!(policy.mode.load(Ordering::Relaxed), 1);
        let link = root.join("link");
        symlink(&path, &link).unwrap();
        assert!(read_command(&link).is_err());
        std::fs::write(&path, vec![b' '; 1025]).unwrap();
        assert!(read_command(&path).is_err());
        std::fs::write(
            &path,
            br#"{"epoch":2,"mode":"sync","reset_cache":false,"extra":1}"#,
        )
        .unwrap();
        assert!(read_command(&path).is_err());
        assert_eq!(policy.mode.load(Ordering::Relaxed), 1);
        std::fs::write(
            &path,
            br#"{"epoch":3,"mode":"off","reset_cache":false,"release_page_tables":true}"#,
        )
        .unwrap();
        assert!(read_command(&path).unwrap().release_page_tables);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn measurement_switch_acknowledges_the_actual_fetch_path() {
        const CHILD: &str = "MAPPED_FETCH_POLICY_TEST";
        if std::env::var_os(CHILD).is_none() {
            let output = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "host_rows::gather::policy::tests::measurement_switch_acknowledges_the_actual_fetch_path",
                    "--nocapture",
                ])
                .env(CHILD, "1")
                .env_remove("CHOPINFER_MAPPED_MISS_FETCH")
                .output()
                .unwrap();
            assert!(
                output.status.success(),
                "{}",
                String::from_utf8_lossy(&output.stdout)
            );
            assert!(String::from_utf8_lossy(&output.stdout).contains("1 passed; 0 failed"));
            return;
        }
        let policy = Policy::new(0, true);
        assert!(crate::host_rows::fetch::enabled());
        for selected in [Some(false), None, Some(true), None] {
            let before = crate::host_rows::fetch::enabled();
            let command: Command = serde_json::from_value(serde_json::json!({
                "epoch":1, "mode":"off", "reset_cache":false, "miss_fetch":selected,
            }))
            .unwrap();
            let record = policy.apply(&command).unwrap();
            let expected = selected.unwrap_or(before);
            assert_eq!(crate::host_rows::fetch::enabled(), expected);
            assert_eq!(record["miss_fetch"]["enabled"], expected);
        }
    }
}
