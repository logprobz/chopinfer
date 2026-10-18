//! The run report (`chopinfer.qual-report.v1`), its summary and the exit
//! code.
//!
//! A run writes `report.json`, `summary.md` and a `records/` directory (one
//! raw record per check, so a run can be rescored) into its report
//! directory, and prints a short summary, or the report itself with
//! `--json`.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use serde_json::{Map, Value, json};

use crate::json::{receipt_text, sha256_hex};
use crate::kind::Status;

pub const SCHEMA: &str = "chopinfer.qual-report.v1";

/// The documented exit codes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Exit {
    /// Every selected check passes (or is declared informational).
    Pass = 0,
    /// A gate failed.
    Fail = 1,
    /// A usage or declaration error.
    Usage = 2,
    /// Insufficient evidence: a missing, unverified or stale reference, a
    /// server that does not identify itself, or a check not run.
    Insufficient = 3,
    /// An environment error: the server is unreachable or timed out.
    Environment = 4,
}

impl Exit {
    pub fn code(self) -> ExitCode {
        ExitCode::from(self as u8)
    }

    /// The exit of a run whose checks have `statuses`: a failure first,
    /// then missing evidence.
    pub fn of(statuses: impl IntoIterator<Item = Status>) -> Self {
        let mut exit = Self::Pass;
        for status in statuses {
            exit = exit.worst(match status {
                Status::Fail => Self::Fail,
                Status::Insufficient | Status::NotRun | Status::NotApplicable => Self::Insufficient,
                Status::Pass | Status::Informational | Status::Waived => Self::Pass,
            });
        }
        exit
    }

    /// Precedence: a failed gate, then a usage error, then the
    /// environment, then insufficient evidence, then a pass.
    fn severity(self) -> u8 {
        match self {
            Self::Pass => 0,
            Self::Insufficient => 1,
            Self::Environment => 2,
            Self::Usage => 3,
            Self::Fail => 4,
        }
    }

    /// The more severe of two exits.
    pub fn worst(self, other: Self) -> Self {
        if other.severity() > self.severity() {
            other
        } else {
            self
        }
    }
}

/// The program that wrote a report.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Tool {
    pub name: String,
    pub version: String,
    /// The source revision the binary was built from, when known.
    pub revision: Option<String>,
}

impl Tool {
    pub fn to_json(&self) -> Value {
        let binary = std::env::current_exe()
            .ok()
            .and_then(|path| std::fs::read(path).ok())
            .map(|bytes| sha256_hex(&bytes));
        json!({"name": self.name, "version": self.version, "revision": self.revision,
               "binary_sha256": binary})
    }
}

/// A pinned reference a check read.
#[derive(Clone, Debug, PartialEq)]
pub struct ReferenceUse {
    pub name: String,
    pub sha256: String,
    pub binding: Value,
}

/// One check's entry in the report.
#[derive(Clone, Debug, PartialEq)]
pub struct CheckReport {
    pub id: String,
    pub kind: String,
    pub version: String,
    pub status: Status,
    pub metrics: Option<Map<String, Value>>,
    pub limits: Option<Value>,
    pub spread: Option<Value>,
    pub predicates: Vec<Value>,
    pub recorded_fields: Vec<String>,
    pub references: Vec<ReferenceUse>,
    /// The raw record (the check's receipt), written under `records/`.
    pub record: Option<Value>,
    pub seconds: f64,
    /// Informational results reported beside the verdict.
    pub context: Vec<Value>,
    pub error: Option<String>,
}

impl CheckReport {
    /// A check with a status and the reason, before or without a verdict.
    pub fn without_verdict(
        id: &str,
        kind: &str,
        version: &str,
        status: Status,
        reason: &str,
    ) -> Self {
        Self {
            id: id.into(),
            kind: kind.into(),
            version: version.into(),
            status,
            metrics: None,
            limits: None,
            spread: None,
            predicates: Vec::new(),
            recorded_fields: Vec::new(),
            references: Vec::new(),
            record: None,
            seconds: 0.0,
            context: Vec::new(),
            error: Some(reason.into()),
        }
    }

    /// The raw record's path in the report directory.
    pub fn record_path(&self) -> String {
        format!("records/{}.json", self.id)
    }

    fn to_json(&self) -> Value {
        let references: Vec<Value> = self
            .references
            .iter()
            .map(|reference| {
                json!({"name": reference.name, "sha256": reference.sha256,
                       "binding": reference.binding})
            })
            .collect();
        json!({"id": self.id, "kind": self.kind, "version": self.version,
               "status": self.status.as_str(), "metrics": self.metrics, "limits": self.limits,
               "spread": self.spread, "predicates": self.predicates,
               "recorded_fields": self.recorded_fields, "references": references,
               "record": self.record.as_ref().map(|_| self.record_path()),
               "record_sha256": self.record.as_ref().map(|record| sha256_hex(&receipt_text(record))),
               "seconds": self.seconds,
               "context": self.context, "error": self.error})
    }
}

/// A run's report.
#[derive(Clone, Debug, PartialEq)]
pub struct Report {
    pub tool: Tool,
    pub model: String,
    pub declaration_sha256: String,
    /// An external variant's overlay, lock and checkpoint lock, each path
    /// with its SHA-256 (`lock::variant_source`); `None` for a variant the
    /// tree declares.
    pub variant_source: Option<Value>,
    pub suite: String,
    pub at: String,
    /// The server's identity and the checkpoint bound to it.
    pub server: Value,
    pub checks: Vec<CheckReport>,
    pub seconds: f64,
    /// Why the run as a whole failed, whatever its checks' statuses: the
    /// server's identity, its environment or an empty selection.
    pub failure: Option<RunFailure>,
}

/// A run-level failure and the exit it gives the run at least.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RunFailure {
    pub exit: Exit,
    pub message: String,
}

impl Report {
    /// Records a run-level failure; the most severe one is kept.
    pub fn fail_run(&mut self, exit: Exit, message: String) {
        if self
            .failure
            .as_ref()
            .is_none_or(|failure| failure.exit.worst(exit) != failure.exit)
        {
            self.failure = Some(RunFailure { exit, message });
        }
    }

    pub fn exit(&self) -> Exit {
        let checks = Exit::of(self.checks.iter().map(|check| check.status));
        match &self.failure {
            Some(failure) => checks.worst(failure.exit),
            None => checks,
        }
    }

    pub fn passed(&self) -> bool {
        self.exit() == Exit::Pass
    }

    /// Qualified: the accuracy suite with every check run and passing, or
    /// failing only where its declared waivers cover it (listed in
    /// `waivers`).
    pub fn qualified(&self) -> bool {
        self.passed()
            && self.suite == "accuracy"
            && !self.checks.is_empty()
            && self
                .checks
                .iter()
                .all(|check| matches!(check.status, Status::Pass | Status::Waived))
    }

    /// Every declared waiver a check's verdict applied: the check, the
    /// predicate it covers and the waiver as applied (measured value,
    /// ceiling, reason and decision).
    pub fn waivers(&self) -> Vec<Value> {
        let mut applied = Vec::new();
        for check in &self.checks {
            for predicate in &check.predicates {
                if let Some(waiver) = predicate.get("waiver") {
                    applied.push(json!({"check": check.id, "path": predicate["path"],
                                        "op": predicate["op"], "value": predicate["value"],
                                        "waiver": waiver}));
                }
            }
        }
        applied
    }

    pub fn to_json(&self) -> Value {
        let serving = &self.server["checkpoint"];
        let profile = self.server["identity"]["health"]["serving"]["profile"].clone();
        json!({
            "schema": SCHEMA,
            "tool": self.tool.to_json(),
            "model": {"name": self.model,
                      "package_identity": serving.get("package_identity").cloned().unwrap_or(Value::Null),
                      "checkpoint_identity": serving.get("checkpoint_identity").cloned().unwrap_or(Value::Null),
                      "profile": profile},
            "declaration_sha256": self.declaration_sha256,
            "variant_source": self.variant_source,
            "suite": self.suite,
            "at": self.at,
            "server": self.server,
            "checks": self.checks.iter().map(CheckReport::to_json).collect::<Vec<_>>(),
            "passed": self.passed(),
            "qualified": self.qualified(),
            "waivers": self.waivers(),
            "seconds": self.seconds,
            "failure": self.failure.as_ref().map(|failure| {
                json!({"exit": failure.exit as u8, "message": failure.message})
            }),
            "exit": self.exit() as u8,
        })
    }

    /// The short human summary.
    pub fn summary(&self) -> String {
        let count = |status: Status| {
            self.checks
                .iter()
                .filter(|check| check.status == status)
                .count()
        };
        let mut text = format!(
            "eval {} {}: {}  {} pass, {} waived, {} fail, {} informational, {} insufficient, {} not run, {} not applicable  {:.0} s\n",
            self.model,
            self.suite,
            match self.exit() {
                Exit::Pass => "PASS",
                Exit::Fail => "FAIL",
                Exit::Insufficient => "INSUFFICIENT",
                Exit::Environment => "ENVIRONMENT",
                Exit::Usage => "USAGE",
            },
            count(Status::Pass),
            count(Status::Waived),
            count(Status::Fail),
            count(Status::Informational),
            count(Status::Insufficient),
            count(Status::NotRun),
            count(Status::NotApplicable),
            self.seconds,
        );
        if let Some(failure) = &self.failure {
            let _ = writeln!(text, "  {}", failure.message);
        }
        for check in &self.checks {
            let _ = write!(text, "  {:<14} {:<13}", check.id, check.status.as_str());
            if let Some(error) = &check.error {
                let _ = write!(text, " {error}");
            }
            text.push('\n');
        }
        text
    }

    /// `summary.md`.
    pub fn markdown(&self) -> String {
        let mut text = format!(
            "# Qualification report: {} {}\n\n- Result: {:?} (exit {})\n- Run: {}\n- Declaration sha256: {}\n- Tool: {} {}\n\n",
            self.model,
            self.suite,
            self.exit(),
            self.exit() as u8,
            self.at,
            self.declaration_sha256,
            self.tool.name,
            self.tool.version,
        );
        if let Some(failure) = &self.failure {
            let _ = writeln!(text, "Run failure: {}\n", failure.message);
        }
        text.push_str("| Check | Kind | Status | Detail |\n|---|---|---|---|\n");
        for check in &self.checks {
            let _ = writeln!(
                text,
                "| {} | {} | {} | {} |",
                check.id,
                check.version,
                check.status.as_str(),
                check.error.as_deref().unwrap_or("").replace('|', "\\|")
            );
        }
        text
    }

    /// Writes `report.json`, `summary.md` and `records/` into the run's own
    /// `directory`, creating it. Returns the report path and its sha256.
    pub fn write(&self, directory: &Path) -> Result<(PathBuf, String), String> {
        let failed = |error: std::io::Error| format!("{}: {error}", directory.display());
        std::fs::create_dir_all(directory).map_err(failed)?;
        std::fs::create_dir(directory.join("records")).map_err(failed)?;
        for check in &self.checks {
            if let Some(record) = &check.record {
                std::fs::write(directory.join(check.record_path()), receipt_text(record))
                    .map_err(failed)?;
            }
        }
        let mut bytes = serde_json::to_vec_pretty(&self.to_json()).expect("JSON values serialize");
        bytes.push(b'\n');
        let path = directory.join("report.json");
        std::fs::write(&path, &bytes).map_err(failed)?;
        std::fs::write(directory.join("summary.md"), self.markdown()).map_err(failed)?;
        Ok((path, sha256_hex(&bytes)))
    }
}

/// A UTC stamp `YYYYMMDDTHHMMSSffffffZ`, as receipts name themselves.
pub fn utc_stamp(time: std::time::SystemTime) -> String {
    let since = time
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    let seconds = since.as_secs();
    let (days, of_day) = ((seconds / 86_400) as i64, seconds % 86_400);
    // Civil date from days since 1970-01-01 (Howard Hinnant's algorithm).
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let day_of_era = z.rem_euclid(146_097);
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
        "{year:04}{month:02}{day:02}T{:02}{:02}{:02}{:06}Z",
        of_day / 3600,
        of_day / 60 % 60,
        of_day % 60,
        since.subsec_micros()
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, UNIX_EPOCH};

    #[test]
    fn stamps_are_utc_dates() {
        assert_eq!(utc_stamp(UNIX_EPOCH), "19700101T000000000000Z");
        let time = UNIX_EPOCH + Duration::new(1_790_872_445, 123_456_000);
        assert_eq!(utc_stamp(time), "20261001T163405123456Z");
        let leap = UNIX_EPOCH + Duration::from_secs(951_782_400);
        assert_eq!(utc_stamp(leap), "20000229T000000000000Z");
    }

    #[test]
    fn the_exit_is_a_failure_first_then_missing_evidence() {
        assert_eq!(Exit::of([]), Exit::Pass);
        assert_eq!(Exit::of([Status::Pass, Status::Informational]), Exit::Pass);
        assert_eq!(Exit::of([Status::NotRun, Status::Pass]), Exit::Insufficient);
        assert_eq!(
            Exit::of([Status::NotApplicable, Status::Pass]),
            Exit::Insufficient
        );
        assert_eq!(Exit::of([Status::NotApplicable, Status::Fail]), Exit::Fail);
        assert_eq!(Exit::of([Status::Insufficient, Status::Fail]), Exit::Fail);
        assert_eq!(Exit::of([Status::Fail, Status::Insufficient]), Exit::Fail);
        assert_eq!(
            Exit::Environment.worst(Exit::Insufficient),
            Exit::Environment
        );
        assert_eq!(Exit::Insufficient.worst(Exit::Fail), Exit::Fail);
    }
}
