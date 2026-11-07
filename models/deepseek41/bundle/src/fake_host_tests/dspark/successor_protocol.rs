//! The successor comparison's capture-directory protocol
//! (`instance::dspark::successor_capture`): the controls a test writes, the
//! step dumps it takes, and how two arms' dumps must match. The
//! state-capture fake-host test and the `dspark_successor` GPU test of
//! `model-deepseek41` both include this file, so a CPU run walks every file
//! the GPU test reads. Every I/O error names its path.
use serde_json::{Value, json};
use std::path::{Path, PathBuf};

/// An arm's drafted steps, in order.
pub const STEPS: [&str; 4] = ["warm", "queue", "consume", "follow"];
/// Each rank's dump files of a step.
pub const DUMPS: [&str; 3] = ["status", "drafts", "ring"];
pub const RANKS: usize = 4;

/// One item's acceptance: (request, accepted drafts, bonus).
pub type Accepted = (u64, u32, u32);

/// `result`, its error naming `path`.
fn at<T>(path: &Path, result: std::io::Result<T>) -> Result<T, String> {
    result.map_err(|error| format!("{}: {error}", path.display()))
}

pub fn read(path: &Path) -> Result<Vec<u8>, String> {
    at(path, std::fs::read(path))
}

fn write(path: &Path, bytes: &[u8]) -> Result<(), String> {
    at(path, std::fs::write(path, bytes))
}

pub fn read_json(path: &Path) -> Result<Value, String> {
    serde_json::from_slice(&read(path)?).map_err(|error| format!("{}: {error}", path.display()))
}

/// Enables `root`, an empty capture directory, with every step's dump.
pub fn start(root: &Path) -> Result<(), String> {
    if at(root, std::fs::read_dir(root))?.next().is_some() {
        return Err(format!(
            "{}: the capture directory must start empty",
            root.display()
        ));
    }
    write(&root.join("enabled"), &[])?;
    write(&root.join("successor-steps"), &[])
}

/// Selects the inline arm, which queues no successor, or the reuse arm.
pub fn select_arm(root: &Path, inline: bool) -> Result<(), String> {
    let off = root.join("successors-off");
    if inline {
        write(&off, &[])
    } else if off.exists() {
        at(&off, std::fs::remove_file(&off))
    } else {
        Ok(())
    }
}

/// An arm's directory of a case's output.
pub fn arm_directory(case: &Path, inline: bool) -> PathBuf {
    case.join(if inline { "inline" } else { "reuse" })
}

/// Names the acceptance the next drafting decode step queues its successor
/// from.
pub fn write_feedback(root: &Path, allocated: bool, items: &[Accepted]) -> Result<(), String> {
    let items: Vec<_> = items
        .iter()
        .map(|&(request, accepted, bonus)| {
            json!({"request": request, "accepted": accepted, "bonus": bonus})
        })
        .collect();
    let control = json!({"version": 1, "allocated": allocated, "items": items});
    write(
        &root.join("successor.json"),
        &serde_json::to_vec(&control).map_err(|error| error.to_string())?,
    )
}

/// Reads and removes the queueing step's feedback.
pub fn take_feedback(root: &Path) -> Result<Vec<Accepted>, String> {
    let path = root.join("successor.json");
    let control = read_json(&path)?;
    at(&path, std::fs::remove_file(&path))?;
    let invalid = || format!("{}: invalid successor feedback", path.display());
    control["items"]
        .as_array()
        .ok_or_else(invalid)?
        .iter()
        .map(|item| {
            let field = |name: &str| item[name].as_u64().ok_or_else(invalid);
            Ok((
                field("request")?,
                field("accepted")? as u32,
                field("bonus")? as u32,
            ))
        })
        .collect()
}

/// Moves the one step dump the bundle wrote since the last take to
/// `destination`, checks it holds every rank's dump files, and returns its
/// `step.json`.
pub fn take_dump(root: &Path, destination: &Path) -> Result<Value, String> {
    let mut dumps = Vec::new();
    for entry in at(root, std::fs::read_dir(root))? {
        let path = at(root, entry)?.path();
        if path
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| name.starts_with("successor-step-"))
        {
            dumps.push(path);
        }
    }
    let [dump] = dumps.as_slice() else {
        return Err(format!(
            "{}: expected one step dump, found {dumps:?}",
            root.display()
        ));
    };
    let parent = destination
        .parent()
        .ok_or_else(|| format!("{}: no parent", destination.display()))?;
    at(parent, std::fs::create_dir_all(parent))?;
    at(dump, std::fs::rename(dump, destination))?;
    for rank in 0..RANKS {
        for name in DUMPS {
            let file = destination.join(format!("{name}-rank-{rank}.bin"));
            at(&file, std::fs::metadata(&file))?;
        }
    }
    read_json(&destination.join("step.json"))
}

/// The first byte at which `a` and `b` differ, if any.
pub fn first_difference(a: &[u8], b: &[u8]) -> Option<usize> {
    a.iter()
        .zip(b)
        .position(|(x, y)| x != y)
        .or((a.len() != b.len()).then_some(a.len().min(b.len())))
}

/// Whether the two arms' dumps under `case` match: every rank's dump files
/// of every step bitwise; each step's drafting items, heads, status checks
/// (at least one) and ring positions; and the successor outcomes, where
/// the reuse arm queues at the queueing step and reuses at the consumer
/// and the inline arm does neither.
pub fn compare_dumps(case: &Path) -> Result<(), String> {
    let (reuse, inline) = (arm_directory(case, false), arm_directory(case, true));
    for step in STEPS {
        for rank in 0..RANKS {
            for name in DUMPS {
                let file = format!("{name}-rank-{rank}.bin");
                let (a, b) = (reuse.join(step).join(&file), inline.join(step).join(&file));
                if let Some(byte) = first_difference(&read(&a)?, &read(&b)?) {
                    return Err(format!(
                        "{} and {} differ at byte {byte}",
                        a.display(),
                        b.display()
                    ));
                }
            }
        }
        let path = reuse.join(step).join("step.json");
        let (a, b) = (
            read_json(&path)?,
            read_json(&inline.join(step).join("step.json"))?,
        );
        for field in ["drafting", "computed_positions", "status_checks", "rings"] {
            if a[field] != b[field] {
                return Err(format!(
                    "{}: {field} is {} with the successor and {} inline",
                    path.display(),
                    a[field],
                    b[field]
                ));
            }
        }
        if a["status_checks"].as_u64().is_none_or(|checks| checks == 0) {
            return Err(format!(
                "{}: the step checked no saved status",
                path.display()
            ));
        }
    }
    for (step, field, reused, drafted) in [
        ("queue", "queued", json!(true), json!(false)),
        ("consume", "consumed", json!("reused"), json!("none")),
        ("follow", "consumed", json!("none"), json!("none")),
    ] {
        let path = reuse.join(step).join("step.json");
        let (a, b) = (
            read_json(&path)?,
            read_json(&inline.join(step).join("step.json"))?,
        );
        if a[field] != reused || b[field] != drafted {
            return Err(format!(
                "{}: {field} is {} with the successor and {} inline",
                path.display(),
                a[field],
                b[field]
            ));
        }
    }
    Ok(())
}
