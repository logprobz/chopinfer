//! The record of choptop's launches in `launches.json` under its state
//! directory: what each instance ran and how it ended. It feeds the recent
//! launches of the loader's Models pane, and lets a restarted choptop find
//! the launches still running, by pid and start time.

use std::collections::BTreeSet;
use std::ffi::OsString;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::os::unix::fs::OpenOptionsExt;
use std::path::{Path, PathBuf};

use serde_json::{Map, Value, json};

use super::command::Key;
use super::process::private_directory;
use crate::loader::{Choice, Recent};

pub const SCHEMA: &str = "choptop.launches.v1";
/// History limit; active ownership records are always kept.
pub const LIMIT: usize = 20;

/// choptop's state directory: `$XDG_STATE_HOME/choptop`, else
/// `~/.local/state/choptop`; `None` without either.
pub fn state_directory(state: Option<OsString>, home: Option<OsString>) -> Option<PathBuf> {
    let absolute =
        |value: Option<OsString>| value.map(PathBuf::from).filter(|path| path.is_absolute());
    absolute(state)
        .map(|state| state.join("choptop"))
        .or_else(|| absolute(home).map(|home| home.join(".local/state/choptop")))
}

/// One instance of a launch.
#[derive(Clone, Debug, PartialEq)]
pub struct Record {
    /// The switch it was part of, by when it was confirmed: `20261102-153012`.
    pub switch: String,
    pub letter: char,
    pub model: String,
    pub served_model_name: String,
    pub profile: String,
    /// NVML indices, as the loader picks them.
    pub gpus: Vec<u32>,
    pub uuids: Vec<String>,
    pub host: String,
    pub port: u16,
    /// The form's values, without secrets.
    pub edits: Vec<(Key, String)>,
    pub custom: Vec<String>,
    /// The command saved for events, with secret environment values masked.
    pub command: String,
    pub pid: Option<u32>,
    pub start_ticks: Option<u64>,
    pub log: PathBuf,
    /// Seconds since the epoch when it was queued.
    pub queued: u64,
    pub exit_status: Option<i32>,
    /// How it ended when not by an exit status: cancelled, the port in use,
    /// ended by a signal.
    pub ended: Option<String>,
}

impl Record {
    /// Whether it has ended.
    pub fn over(&self) -> bool {
        self.exit_status.is_some() || self.ended.is_some()
    }

    fn to_json(&self) -> Value {
        json!({
            "switch": self.switch,
            "letter": self.letter.to_string(),
            "model": self.model,
            "served_model_name": self.served_model_name,
            "profile": self.profile,
            "gpus": self.gpus,
            "uuids": self.uuids,
            "host": self.host,
            "port": self.port,
            "edits": self.edits.iter().map(|(key, value)| {
                let (kind, name) = match key {
                    Key::Flag(name) => ("flag", name),
                    Key::Variable(name) => ("variable", name),
                    Key::ModelOption(name) => ("model-option", name),
                };
                json!({"kind": kind, "name": name, "value": value})
            }).collect::<Vec<_>>(),
            "custom": self.custom,
            "command": self.command,
            "pid": self.pid,
            "start_ticks": self.start_ticks,
            "log": self.log.display().to_string(),
            "queued": self.queued,
            "exit_status": self.exit_status,
            "ended": self.ended,
        })
    }

    fn from_json(value: &Value) -> Result<Self, String> {
        let object = value.as_object().ok_or("a record is not an object")?;
        let text = |name: &str| {
            object
                .get(name)
                .and_then(Value::as_str)
                .map(str::to_owned)
                .ok_or_else(|| format!("{name} is missing"))
        };
        let number = |name: &str| object.get(name).and_then(Value::as_u64);
        let texts = |name: &str| -> Result<Vec<String>, String> {
            object
                .get(name)
                .and_then(Value::as_array)
                .ok_or_else(|| format!("{name} is missing"))?
                .iter()
                .map(|item| {
                    item.as_str()
                        .map(str::to_owned)
                        .ok_or_else(|| format!("{name} holds a non-text item"))
                })
                .collect()
        };
        let edits = object
            .get("edits")
            .and_then(Value::as_array)
            .ok_or("edits is missing")?
            .iter()
            .map(|edit| {
                let field = |name: &str| edit.get(name).and_then(Value::as_str).map(str::to_owned);
                let (Some(kind), Some(name), Some(value)) =
                    (field("kind"), field("name"), field("value"))
                else {
                    return Err("an edit lacks its kind, name or value".to_owned());
                };
                let key = match kind.as_str() {
                    "flag" => Key::Flag(name),
                    "variable" => Key::Variable(name),
                    "model-option" => Key::ModelOption(name),
                    other => return Err(format!("an edit of kind {other}")),
                };
                Ok((key, value))
            })
            .collect::<Result<Vec<_>, String>>()?;
        Ok(Self {
            switch: text("switch")?,
            letter: text("letter")?.chars().next().ok_or("letter is empty")?,
            model: text("model")?,
            served_model_name: text("served_model_name")?,
            profile: text("profile")?,
            gpus: object
                .get("gpus")
                .and_then(Value::as_array)
                .ok_or("gpus is missing")?
                .iter()
                .map(|gpu| {
                    gpu.as_u64()
                        .and_then(|gpu| u32::try_from(gpu).ok())
                        .ok_or("gpus holds a non-index")
                })
                .collect::<Result<Vec<_>, _>>()?,
            uuids: texts("uuids")?,
            host: text("host")?,
            port: number("port")
                .and_then(|port| u16::try_from(port).ok())
                .ok_or("port is missing")?,
            edits,
            custom: texts("custom")?,
            command: object
                .get("command")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_owned(),
            pid: number("pid").and_then(|pid| u32::try_from(pid).ok()),
            start_ticks: number("start_ticks"),
            log: PathBuf::from(text("log")?),
            queued: number("queued").ok_or("queued is missing")?,
            exit_status: object
                .get("exit_status")
                .and_then(Value::as_i64)
                .and_then(|status| i32::try_from(status).ok()),
            ended: object
                .get("ended")
                .and_then(Value::as_str)
                .map(str::to_owned),
        })
    }
}

/// `Qwen` for `Qwen3.8-Flash`: a served model name up to its first digit
/// or dash.
pub fn short_name(served: &str) -> &str {
    let end = served
        .find(|character: char| character == '-' || character.is_ascii_digit())
        .unwrap_or(served.len());
    if end == 0 { served } else { &served[..end] }
}

/// `3d ago`: how long before `now` the time `then` was, both in seconds
/// since the epoch.
pub fn ago(then: u64, now: u64) -> String {
    let seconds = now.saturating_sub(then);
    match seconds {
        0..60 => "just now".to_owned(),
        60..3600 => format!("{}m ago", seconds / 60),
        3600..86_400 => format!("{}h ago", seconds / 3600),
        86_400..172_800 => "yesterday".to_owned(),
        _ => format!("{}d ago", seconds / 86_400),
    }
}

/// The records file and what it holds.
#[derive(Debug, Default)]
pub struct Records {
    /// `launches.json`; `None` keeps the records in memory only.
    pub path: Option<PathBuf>,
    pub records: Vec<Record>,
}

impl Records {
    /// Reads `path`. A missing file holds no records; one that cannot be
    /// read holds none either, and says why.
    pub fn load(path: Option<PathBuf>) -> (Self, Option<String>) {
        let mut records = Self {
            path,
            records: Vec::new(),
        };
        let Some(path) = &records.path else {
            return (records, None);
        };
        let text = match fs::read_to_string(path) {
            Ok(text) => text,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return (records, None),
            Err(error) => {
                let message = format!(
                    "{}: {error}; starting with no launch records",
                    path.display()
                );
                return (records, Some(message));
            }
        };
        match parse(&text) {
            Ok(parsed) => records.records = parsed,
            Err(error) => {
                let message = format!(
                    "{}: {error}; starting with no launch records",
                    path.display()
                );
                return (records, Some(message));
            }
        }
        (records, None)
    }

    /// Adds `record`, or replaces the one of its switch and letter, and
    /// evicts completed history first. Returns a message when active
    /// records alone exceed [`LIMIT`].
    pub fn put(&mut self, record: Record) -> Option<String> {
        match self
            .records
            .iter_mut()
            .find(|seen| seen.switch == record.switch && seen.letter == record.letter)
        {
            Some(seen) => *seen = record,
            None => self.records.push(record),
        }
        let mut excess = self.records.len().saturating_sub(LIMIT);
        self.records.retain(|record| {
            if excess > 0 && record.over() {
                excess -= 1;
                false
            } else {
                true
            }
        });
        (excess > 0).then(|| {
            format!(
                "keeping all {} active launch records, above the history limit of {LIMIT}",
                self.records.len()
            )
        })
    }

    /// Writes the records: a new file beside the old one, renamed over it,
    /// readable by this user only.
    pub fn save(&self) -> Result<(), String> {
        let Some(path) = &self.path else {
            return Ok(());
        };
        if let Some(directory) = path.parent() {
            private_directory(directory)?;
        }
        let document = json!({
            "schema": SCHEMA,
            "launches": self.records.iter().map(Record::to_json).collect::<Vec<_>>(),
        });
        let mut text =
            serde_json::to_string_pretty(&document).map_err(|error| error.to_string())?;
        text.push('\n');
        let partial = partial_path(path);
        let failed = |error: std::io::Error| format!("{}: {error}", partial.display());
        let _ = fs::remove_file(&partial);
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(&partial)
            .map_err(failed)?;
        file.write_all(text.as_bytes()).map_err(failed)?;
        file.sync_all().map_err(failed)?;
        fs::rename(&partial, path).map_err(|error| format!("{}: {error}", path.display()))
    }

    /// A name for a new switch confirmed at `stamp` that no record has:
    /// `stamp`, else `stamp-2` and so on.
    pub fn unused_switch(&self, stamp: &str) -> String {
        let taken = |name: &str| self.records.iter().any(|record| record.switch == name);
        let mut name = stamp.to_owned();
        let mut number = 2;
        while taken(&name) {
            name = format!("{stamp}-{number}");
            number += 1;
        }
        name
    }

    /// The choices of switch `switch`: the GPUs of all its instances.
    pub fn choice(&self, switch: &str) -> Option<Choice> {
        let records = self
            .records
            .iter()
            .filter(|record| record.switch == switch)
            .collect::<Vec<_>>();
        let first = records.first()?;
        Some(Choice {
            model: first.model.clone(),
            profile: first.profile.clone(),
            gpus: records
                .iter()
                .flat_map(|record| record.gpus.iter().copied())
                .collect::<BTreeSet<_>>(),
            edits: first.edits.iter().cloned().collect(),
            custom: first.custom.clone(),
        })
    }

    /// The latest switches, newest first, as the Models pane shows them.
    pub fn recent(&self, now: u64) -> Vec<Recent> {
        let mut switches = Vec::new();
        for record in self.records.iter().rev() {
            if !switches.contains(&record.switch) {
                switches.push(record.switch.clone());
            }
        }
        switches
            .into_iter()
            .filter_map(|switch| {
                let records = self
                    .records
                    .iter()
                    .filter(|record| record.switch == switch)
                    .collect::<Vec<_>>();
                let first = records.first()?;
                let count = records.len();
                Some(Recent {
                    label: format!(
                        "{} {}{}",
                        short_name(&first.served_model_name),
                        first.profile,
                        if count > 1 {
                            format!(" x{count}")
                        } else {
                            String::new()
                        }
                    ),
                    when: ago(first.queued, now),
                    choice: self.choice(&switch)?,
                })
            })
            .collect()
    }
}

fn partial_path(path: &Path) -> PathBuf {
    let mut name = path.file_name().unwrap_or_default().to_os_string();
    name.push(".partial");
    path.with_file_name(name)
}

fn parse(text: &str) -> Result<Vec<Record>, String> {
    let document: Value =
        serde_json::from_str(text).map_err(|error| format!("not a JSON document: {error}"))?;
    let object: &Map<String, Value> = document.as_object().ok_or("not a JSON object")?;
    match object.get("schema").and_then(Value::as_str) {
        Some(SCHEMA) => {}
        Some(other) => return Err(format!("schema {other}, not {SCHEMA}")),
        None => return Err("no schema".to_owned()),
    }
    object
        .get("launches")
        .and_then(Value::as_array)
        .ok_or("launches is missing")?
        .iter()
        .map(Record::from_json)
        .collect()
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::launch::process::tests::scratch;
    use std::os::unix::fs::MetadataExt;

    pub(crate) fn record(switch: &str, letter: char, gpus: &[u32]) -> Record {
        Record {
            switch: switch.into(),
            letter,
            model: "qwen38".into(),
            served_model_name: "Qwen3.8-Flash".into(),
            profile: "tp2-mtp".into(),
            gpus: gpus.to_vec(),
            uuids: Vec::new(),
            host: "127.0.0.1".into(),
            port: 8310,
            edits: vec![(Key::Flag("max-sequences".into()), "48".into())],
            custom: vec!["--kv-offload-prefix off".into()],
            command: String::new(),
            pid: None,
            start_ticks: None,
            log: PathBuf::from("/state/choptop/logs/a.log"),
            queued: 1_000_000,
            exit_status: None,
            ended: None,
        }
    }

    #[test]
    fn records_are_written_atomically_and_privately_and_read_back() {
        let root = scratch("records");
        let path = root.join("state/choptop/launches.json");
        let (mut records, message) = Records::load(Some(path.clone()));
        assert!(records.records.is_empty() && message.is_none());
        let mut first = record("20261102-153012", 'A', &[0, 1]);
        records.put(first.clone());
        records.put(record("20261102-153012", 'B', &[2, 3]));
        first.pid = Some(4242);
        first.start_ticks = Some(99);
        first.exit_status = Some(1);
        records.put(first.clone());
        records.save().unwrap();
        assert_eq!(fs::metadata(&path).unwrap().mode() & 0o777, 0o600);
        assert_eq!(
            fs::metadata(path.parent().unwrap()).unwrap().mode() & 0o777,
            0o700
        );
        assert!(!root.join("state/choptop/launches.json.partial").exists());
        let (again, message) = Records::load(Some(path.clone()));
        assert_eq!(message, None);
        assert_eq!(again.records, records.records);
        assert_eq!(again.records[0], first);
        // The live record stays beside the newest completed history.
        for number in 0..30 {
            let mut completed = record(&format!("s{number:02}"), 'A', &[0]);
            completed.exit_status = Some(1);
            records.put(completed);
        }
        assert_eq!(records.records.len(), LIMIT);
        assert_eq!(records.records[0].letter, 'B');
        assert_eq!(records.records[1].switch, "s11");
        // A switch confirmed in the same second as another gets a name of
        // its own.
        assert_eq!(records.unused_switch("s29"), "s29-2");
        records.put(record("s29-2", 'A', &[0]));
        assert_eq!(records.unused_switch("s29"), "s29-3");
        assert_eq!(records.unused_switch("s30"), "s30");
    }

    #[test]
    fn active_records_survive_history_eviction_and_exceed_the_limit_explicitly() {
        let root = scratch("active-records");
        let path = root.join("launches.json");
        let mut records = Records::load(Some(path.clone())).0;
        for number in 0..3 {
            let mut active = record(&format!("active-{number}"), 'A', &[number]);
            if number > 0 {
                active.pid = Some(4000 + number);
                active.start_ticks = Some(100);
            }
            assert_eq!(records.put(active), None);
        }
        for number in 0..30 {
            let mut completed = record(&format!("ended-{number}"), 'A', &[0]);
            completed.ended = Some("cancelled".into());
            records.put(completed);
        }
        records.save().unwrap();
        let mut again = Records::load(Some(path.clone())).0;
        assert_eq!(again.records.len(), LIMIT);
        assert_eq!(
            again.records.iter().filter(|record| !record.over()).count(),
            3
        );
        for number in 3..=LIMIT {
            let message = again.put(record(&format!("active-{number}"), 'A', &[0]));
            assert_eq!(message.is_some(), number == LIMIT);
        }
        again.save().unwrap();
        let restored = Records::load(Some(path)).0;
        assert_eq!(restored.records.len(), LIMIT + 1);
        assert!(restored.records.iter().all(|record| !record.over()));
    }

    #[test]
    fn a_malformed_file_starts_empty_and_says_why() {
        let root = scratch("malformed");
        let path = root.join("launches.json");
        fs::write(
            &path,
            "{\"schema\": \"choptop.launches.v1\", \"launches\": [1]}",
        )
        .unwrap();
        let (records, message) = Records::load(Some(path.clone()));
        assert!(records.records.is_empty());
        assert!(message.unwrap().contains("not an object"));
        fs::write(&path, "not json").unwrap();
        assert!(
            Records::load(Some(path))
                .1
                .unwrap()
                .contains("not a JSON document")
        );
    }

    #[test]
    fn recent_launches_group_a_switch_newest_first() {
        let mut records = Records::default();
        let mut older = record("20261030-090000", 'A', &[0, 1, 2, 3]);
        older.model = "deepseek41".into();
        older.served_model_name = "DeepSeek-V4.1-Flash".into();
        older.profile = "tp4-dspark".into();
        older.edits.clear();
        older.queued = 1_000_000 - 90_000;
        records.put(older);
        records.put(record("20261102-153012", 'A', &[0, 1]));
        records.put(record("20261102-153012", 'B', &[2, 3]));
        let recent = records.recent(1_000_000 + 300);
        assert_eq!(recent.len(), 2);
        assert_eq!(
            (recent[0].label.as_str(), recent[0].when.as_str()),
            ("Qwen tp2-mtp x2", "5m ago")
        );
        assert_eq!(recent[0].choice.gpus, BTreeSet::from([0, 1, 2, 3]));
        assert_eq!(recent[0].choice.edits.len(), 1);
        assert_eq!(
            (recent[1].label.as_str(), recent[1].when.as_str()),
            ("DeepSeek tp4-dspark", "yesterday")
        );
    }

    #[test]
    fn the_state_directory_follows_xdg() {
        assert_eq!(
            state_directory(Some("/state".into()), Some("/operator".into())),
            Some(PathBuf::from("/state/choptop"))
        );
        assert_eq!(
            state_directory(Some("relative".into()), Some("/operator".into())),
            Some(PathBuf::from("/operator/.local/state/choptop"))
        );
        assert_eq!(state_directory(None, None), None);
        assert_eq!(short_name("GLM-5.3-Flash"), "GLM");
        assert_eq!(short_name("MiMo-V2.6-Flash"), "MiMo");
        assert_eq!(ago(0, 7 * 86_400 + 5), "7d ago");
    }
}
