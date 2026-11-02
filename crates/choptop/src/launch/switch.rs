//! A switch between `S` and its launches: what the fresh reading changed,
//! whether `serve --check` resolves what the profile and the form say, the
//! confirm dialog's contents, and the instances it queues.

use std::collections::BTreeMap;
use std::path::Path;

use super::command::{self, Key, Origin};
use super::occupancy::{Gpu, Holder, Occupancy, Serve};
use super::placement::{Dependency, Placement};
use super::process::{Identity, stamp};
use super::records::Record;
use super::{Queued, Reservation, Wait, indices};
use crate::catalog::{Catalog, Check, Profile};
use crate::loader::Start;

/// What a change between the start checked and the fresh reading means:
/// GPUs newly held, ports moved, launches to replace. Problems of the new
/// placement are the loader's, given before these.
pub fn changed(before: &Placement, after: &Placement) -> Vec<String> {
    let mut changes = Vec::new();
    for serve in &after.stops {
        if !before.stops.contains(serve) {
            changes.push(format!("{} newly holds a GPU picked", serve.described()));
        }
    }
    for planned in &after.instances {
        match before
            .instances
            .iter()
            .find(|seen| seen.letter == planned.letter)
        {
            Some(seen) if seen.port != planned.port => changes.push(format!(
                "port {} is taken now, so {} would listen on {}",
                seen.port, planned.letter, planned.port
            )),
            Some(_) => {}
            None => changes.push(format!("instance {} is new", planned.letter)),
        }
    }
    if after.replaces.len() != before.replaces.len() {
        changes.push("the launches it replaces changed".to_owned());
    }
    changes
}

/// Two values the same: equal text, or equal numbers.
fn same(left: &str, right: &str) -> bool {
    let (left, right) = (left.trim(), right.trim());
    left == right
        || matches!(
            (left.parse::<f64>(), right.parse::<f64>()),
            (Ok(left), Ok(right)) if left == right
        )
}

/// The settings and model options the profile and the form say, by name:
/// the profile's, then the form's values and custom rows over them, and
/// the resets of the model options the profile leaves out.
fn expected(
    profile: &Profile,
    start: &Start,
) -> (BTreeMap<String, String>, BTreeMap<String, String>) {
    let mut settings = profile.settings.clone();
    let mut options = profile.model_options.clone();
    for (key, value) in &start.choice.edits {
        match key {
            Key::Flag(name) => {
                settings.insert(name.clone(), value.clone());
            }
            Key::ModelOption(name) => {
                options.insert(name.clone(), value.clone());
            }
            Key::Variable(_) => {}
        }
    }
    for (group, origin) in &start.explicit.options {
        if *origin == Origin::Set {
            continue;
        }
        match group.as_slice() {
            [flag, assignment] if flag == "--model-option" => {
                if let Some((name, value)) = assignment.split_once('=') {
                    options.insert(name.to_owned(), value.to_owned());
                }
            }
            [flag] if *origin == Origin::Custom => {
                if let Some(name) = flag.strip_prefix("--") {
                    settings.insert(name.to_owned(), "on".to_owned());
                }
            }
            [flag, value] if *origin == Origin::Custom => {
                if let Some(name) = flag.strip_prefix("--") {
                    settings.insert(name.to_owned(), value.clone());
                }
            }
            _ => {}
        }
    }
    (settings, options)
}

/// Where `serve --check` resolves other values than the profile and the
/// form say, each named. Only the settings both sides govern are compared:
/// a key on one side only is not governed there.
pub fn differences(check: &Check, profile: &Profile, start: &Start) -> Vec<String> {
    let (settings, options) = expected(profile, start);
    let mut differences = Vec::new();
    for (name, value) in &check.settings {
        if let Some(wanted) = settings.get(name)
            && !same(value, wanted)
        {
            differences.push(format!("--{name} resolves to {value}, not {wanted}"));
        }
    }
    for option in &check.model_options {
        if let Some(wanted) = options.get(&option.name)
            && !same(&option.value, wanted)
        {
            differences.push(format!(
                "model option {} resolves to {}, not {wanted}",
                option.name, option.value
            ));
        }
    }
    differences
}

/// What the switch waits for, as it was when the dialog opened.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Target {
    Serve(Serve),
    /// A launch, by its id.
    Launch(u64),
}

/// A row of the dialog's `Stops first`.
#[derive(Clone, Debug, PartialEq)]
pub struct Stopping {
    pub target: Target,
    /// `proc 48211`.
    pub label: String,
    pub model: String,
    /// The GPUs it frees, by NVML index.
    pub gpus: Vec<u32>,
    /// What choptop does to it.
    pub how: String,
}

/// The confirm dialog: what starts, what stops first, and what to know.
#[derive(Clone, Debug, PartialEq)]
pub struct Confirm {
    pub start: Box<Start>,
    /// `Qwen3.8-Flash`.
    pub served: String,
    pub tensor_parallel: u32,
    /// When it opened, in seconds since the epoch, and as the stamp the
    /// switch and its logs are named by.
    pub queued: u64,
    pub switch: String,
    /// Each instance's log file name, in instance order.
    pub logs: Vec<String>,
    /// What each instance waits for.
    pub waits: Vec<Vec<Target>>,
    pub stops: Vec<Stopping>,
    /// The profile's qualification text when it is experimental.
    pub qualification: Option<String>,
    pub warnings: Vec<String>,
    /// `host memory: 132 of 251 GiB available`.
    pub memory: Option<String>,
}

impl Confirm {
    /// `Qwen tp2-mtp`.
    pub fn title(&self) -> String {
        format!(
            "Start {} {}",
            super::records::short_name(&self.served),
            self.start.profile
        )
    }
}

/// A file name part: what is not a letter, digit, dot or dash becomes `_`.
fn file_part(text: &str) -> String {
    text.chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || matches!(character, '.' | '-') {
                character
            } else {
                '_'
            }
        })
        .collect()
}

/// The GPUs `serve` holds among `gpus`.
fn held_by(gpus: &[Gpu], holder: &Holder) -> Vec<u32> {
    gpus.iter()
        .filter(
            |gpu| matches!(&gpu.occupancy, Occupancy::Held(holders) if holders.contains(holder)),
        )
        .map(|gpu| gpu.index)
        .collect()
}

/// What the dialog is built from besides the start.
pub struct Reading<'a> {
    pub catalog: &'a Catalog,
    pub check: &'a Check,
    pub gpus: &'a [Gpu],
    pub reservations: &'a [Reservation],
    /// Serve processes being stopped already, by pid and start ticks.
    pub stopping: &'a dyn Fn(u32, u64) -> bool,
    /// The form's warnings, as the Options pane notes them.
    pub warnings: Vec<String>,
    /// `(available, total)` bytes of host memory.
    pub memory: Option<(u64, u64)>,
    /// Seconds since the epoch.
    pub wall: u64,
}

/// The confirm dialog of `start`.
pub fn confirm(start: Box<Start>, reading: &Reading) -> Confirm {
    let catalog = reading.catalog;
    let model = catalog.model(&start.model);
    let profile = catalog.profile(&start.model, &start.profile);
    let served = model.map_or_else(
        || start.model.clone(),
        |model| model.served_model_name.clone(),
    );
    let switch = stamp(reading.wall);
    let logs = start
        .placement
        .instances
        .iter()
        .map(|planned| {
            format!(
                "{}-{}-{}-{switch}.log",
                file_part(&start.model),
                file_part(&start.profile),
                planned.letter.to_ascii_lowercase()
            )
        })
        .collect();
    let target = |dependency: &Dependency| match dependency {
        Dependency::Serve(serve) => Some(Target::Serve(serve.clone())),
        Dependency::Launch(position) => reading
            .reservations
            .get(*position)
            .map(|reservation| Target::Launch(reservation.launch)),
    };
    let waits = start
        .placement
        .instances
        .iter()
        .map(|planned| planned.after.iter().filter_map(target).collect())
        .collect();
    let mut stops = Vec::new();
    for serve in &start.placement.stops {
        let how = if (reading.stopping)(serve.pid, serve.start_ticks) {
            "already stopping; the launch waits for it to exit"
        } else {
            "SIGTERM, then the GPUs are free once it exits"
        };
        stops.push(Stopping {
            target: Target::Serve(serve.clone()),
            label: serve.label(),
            model: serve.name.clone().unwrap_or_else(|| "-".to_owned()),
            gpus: held_by(reading.gpus, &Holder::Serve(serve.clone())),
            how: how.to_owned(),
        });
    }
    for position in &start.placement.replaces {
        let Some(reservation) = reading.reservations.get(*position) else {
            continue;
        };
        stops.push(Stopping {
            target: Target::Launch(reservation.launch),
            label: reservation.label.clone(),
            model: reservation.model.clone(),
            gpus: held_by(reading.gpus, &Holder::Launch(*position)),
            how: if reservation.waiting() {
                "cancelled before it starts"
            } else {
                "SIGTERM to its process group, then the GPUs are free once it exits"
            }
            .to_owned(),
        });
    }
    let mut warnings = reading.warnings.clone();
    for option in &reading.check.model_options {
        if option.value != option.default {
            warnings.push(format!(
                "model option {} {}: unqualified (package default {})",
                option.name, option.value, option.default
            ));
        }
    }
    for warning in &reading.check.warnings {
        let flag = warning.split_whitespace().next().unwrap_or_default();
        let said = flag.starts_with("--") && warnings.iter().any(|seen| seen.starts_with(flag));
        if !said {
            warnings.push(format!("serve --check: {warning}"));
        }
    }
    let picked = start
        .placement
        .instances
        .iter()
        .flat_map(|planned| planned.gpus.iter().copied())
        .collect::<Vec<_>>();
    for gpu in reading
        .gpus
        .iter()
        .filter(|gpu| picked.contains(&gpu.index))
    {
        for process in &gpu.graphics {
            warnings.push(format!(
                "GPU {} also holds {}, which choptop leaves running",
                gpu.index,
                super::occupancy::described(process)
            ));
        }
    }
    let gib = |bytes: u64| bytes as f64 / f64::from(1u32 << 30);
    Confirm {
        tensor_parallel: profile.map_or(1, |profile| profile.tensor_parallel),
        qualification: profile
            .filter(|profile| profile.qualified_max_sequences.is_none())
            .map(|profile| format!("{}: {}", profile.name, profile.qualification)),
        memory: reading.memory.map(|(available, total)| {
            format!(
                "host memory: {:.0} of {:.0} GiB available",
                gib(available),
                gib(total)
            )
        }),
        served,
        queued: reading.wall,
        switch,
        logs,
        waits,
        stops,
        warnings,
        start,
    }
}

/// The instances `confirm` queues: their records, command lines and what
/// each waits for, the serve processes as `identities` recorded them.
pub fn queued(
    confirm: &Confirm,
    catalog: &Catalog,
    identities: &[Identity],
    logs: Option<&Path>,
) -> Result<Vec<Queued>, String> {
    let start = &confirm.start;
    let binary = catalog.binary.clone()?;
    let secret = |name: &str| {
        name == command::API_KEY
            || catalog
                .options()
                .and_then(|options| options.variable(name))
                .is_some_and(|variable| variable.secret)
    };
    let edits = start
        .choice
        .edits
        .iter()
        .filter(|(key, _)| !matches!(key, Key::Variable(name) if secret(name)))
        .map(|(key, value)| (key.clone(), value.clone()))
        .collect::<Vec<_>>();
    let program = binary.file_name().map_or_else(
        || "chopinfer".to_owned(),
        |name| name.to_string_lossy().into_owned(),
    );
    let mut queued = Vec::new();
    for (index, planned) in start.placement.instances.iter().enumerate() {
        let listen = planned.listen();
        let arguments = command::arguments(&start.command, &listen);
        let environment = command::environment(&start.explicit, &planned.uuids);
        let hidden = |name: &str, value: &str| {
            if secret(name) {
                format!("{name}=****")
            } else {
                format!("{name}={value}")
            }
        };
        let mut logged = environment
            .iter()
            .map(|(name, value)| hidden(name, value))
            .collect::<Vec<_>>();
        logged.push(binary.display().to_string());
        logged.extend(arguments.iter().cloned());
        let secrets = environment
            .iter()
            .map(|(name, _)| name.as_str())
            .filter(|name| secret(name))
            .collect::<Vec<_>>();
        let shown = command::tokens(
            &program,
            &start.command,
            &start.explicit,
            &planned.gpus,
            &listen,
            &secrets,
        )
        .into_iter()
        .map(|(token, _)| token)
        .collect::<Vec<_>>()
        .join(" ");
        let mut waits = Vec::new();
        for target in confirm.waits.get(index).into_iter().flatten() {
            match target {
                Target::Serve(serve) => {
                    if let Some(identity) = identities.iter().find(|identity| {
                        identity.pid == serve.pid && identity.start_ticks == serve.start_ticks
                    }) {
                        waits.push(Wait::Serve(identity.clone()));
                    }
                }
                Target::Launch(id) => waits.push(Wait::Launch(*id)),
            }
        }
        let name = confirm.logs.get(index).cloned().unwrap_or_default();
        queued.push(Queued {
            record: Record {
                switch: confirm.switch.clone(),
                letter: planned.letter,
                model: start.model.clone(),
                served_model_name: confirm.served.clone(),
                profile: start.profile.clone(),
                gpus: planned.gpus.clone(),
                uuids: planned.uuids.clone(),
                host: planned.host.clone(),
                port: planned.port,
                edits: edits.clone(),
                custom: start.choice.custom.clone(),
                pid: None,
                start_ticks: None,
                log: logs.map_or_else(|| name.clone().into(), |logs| logs.join(&name)),
                queued: confirm.queued,
                exit_status: None,
                ended: None,
            },
            binary: binary.clone(),
            arguments,
            environment,
            logged: logged.join(" "),
            shown,
            waits,
        });
    }
    Ok(queued)
}

/// `to start Qwen3.8-Flash tp2-mtp on GPUs 0,1,2,3`: why the switch stops
/// what it stops.
pub fn reason(confirm: &Confirm) -> String {
    let gpus = confirm
        .start
        .placement
        .instances
        .iter()
        .flat_map(|planned| planned.gpus.iter().copied())
        .collect::<Vec<_>>();
    format!(
        "to start {} {} on GPUs {}",
        confirm.served,
        confirm.start.profile,
        indices(&gpus)
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::launch::placement::Planned;

    fn serve(pid: u32) -> Serve {
        Serve {
            pid,
            start_ticks: u64::from(pid),
            name: Some("MiMo-V2.6-Flash".into()),
            ports: vec![8310],
            starting: false,
        }
    }

    fn placement(stops: &[u32], ports: &[u16]) -> Placement {
        Placement {
            instances: ports
                .iter()
                .enumerate()
                .map(|(number, port)| Planned {
                    letter: char::from(b'A' + number as u8),
                    gpus: vec![number as u32],
                    uuids: Vec::new(),
                    host: "127.0.0.1".into(),
                    port: *port,
                    after: Vec::new(),
                })
                .collect(),
            stops: stops.iter().map(|pid| serve(*pid)).collect(),
            ..Placement::default()
        }
    }

    #[test]
    fn a_fresh_reading_names_gpus_newly_held_and_ports_taken() {
        let before = placement(&[48211], &[8310, 8311]);
        assert!(changed(&before, &before.clone()).is_empty());
        assert_eq!(
            changed(&before, &placement(&[48211, 61204], &[8310, 8312])),
            [
                "proc 61204 MiMo-V2.6-Flash newly holds a GPU picked",
                "port 8311 is taken now, so B would listen on 8312",
            ]
        );
        // A process that has left the GPUs is no reason to stop.
        assert!(changed(&before, &placement(&[], &[8310, 8311])).is_empty());
    }
}
