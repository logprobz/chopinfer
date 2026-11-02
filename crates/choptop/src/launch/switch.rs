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
use super::{Queued, Reservation, Wait, indices, reserved};
use crate::catalog::{Catalog, Check, Profile};
use crate::loader::Start;

/// What changed between the placement a start was confirmed or checked
/// with and the one a later reading gives: serve processes newly holding a
/// GPU picked or gone from them, launches to replace by id, ports moved,
/// and what each instance waits for. Problems of the new placement are
/// the loader's, given before these. `reservations` name the launches.
pub fn changed(before: &Placement, after: &Placement, reservations: &[Reservation]) -> Vec<String> {
    let launch = |id: &u64| {
        reserved(reservations, *id).map_or_else(
            || "a launch that has ended".to_owned(),
            |reservation| reservation.label.clone(),
        )
    };
    let mut changes = Vec::new();
    for serve in &after.stops {
        if !before.stops.contains(serve) {
            changes.push(format!("{} newly holds a GPU picked", serve.described()));
        }
    }
    for serve in &before.stops {
        if !after.stops.contains(serve) {
            changes.push(format!(
                "{} no longer holds a GPU picked",
                serve.described()
            ));
        }
    }
    for id in &after.replaces {
        if !before.replaces.contains(id) {
            changes.push(format!("{} newly holds a GPU or port picked", launch(id)));
        }
    }
    for id in &before.replaces {
        if !after.replaces.contains(id) {
            changes.push(format!(
                "{} no longer holds a GPU or port picked",
                launch(id)
            ));
        }
    }
    for planned in &after.instances {
        match before
            .instances
            .iter()
            .find(|seen| seen.letter == planned.letter)
        {
            Some(seen) if seen.gpus != planned.gpus || seen.uuids != planned.uuids => {
                changes.push(format!("instance {}'s GPUs changed", planned.letter));
            }
            Some(seen) if seen.port != planned.port => changes.push(format!(
                "port {} is taken now, so {} would listen on {}",
                seen.port, planned.letter, planned.port
            )),
            Some(seen) if seen.after != planned.after => {
                changes.push(format!("what {} waits for changed", planned.letter))
            }
            Some(_) => {}
            None => changes.push(format!("instance {} is new", planned.letter)),
        }
    }
    for seen in &before.instances {
        if !after
            .instances
            .iter()
            .any(|planned| planned.letter == seen.letter)
        {
            changes.push(format!("instance {} is gone", seen.letter));
        }
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

/// The confirm dialog of `start`, or why it cannot open: a launch the
/// placement replaces or waits for has ended since the placement was read.
pub fn confirm(start: Box<Start>, reading: &Reading) -> Result<Confirm, String> {
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
    // A launch the start replaces that has ended since its placement was
    // worked out is no reason to go on with that placement.
    let ended = |_: &u64| {
        "a launch this start replaces has ended since its placement was read; press S again"
            .to_owned()
    };
    let target = |dependency: &Dependency| match dependency {
        Dependency::Serve(serve) => Ok(Target::Serve(serve.clone())),
        Dependency::Launch(id) => reserved(reading.reservations, *id)
            .map(|reservation| Target::Launch(reservation.launch))
            .ok_or_else(|| ended(id)),
    };
    let waits = start
        .placement
        .instances
        .iter()
        .map(|planned| {
            planned
                .after
                .iter()
                .map(target)
                .collect::<Result<Vec<_>, _>>()
        })
        .collect::<Result<Vec<_>, _>>()?;
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
    for id in &start.placement.replaces {
        let reservation = reserved(reading.reservations, *id).ok_or_else(|| ended(id))?;
        stops.push(Stopping {
            target: Target::Launch(reservation.launch),
            label: reservation.label.clone(),
            model: reservation.model.clone(),
            gpus: held_by(reading.gpus, &Holder::Launch(*id)),
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
    Ok(Confirm {
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
    })
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

    fn reservation(launch: u64, label: &str) -> Reservation {
        Reservation {
            label: label.into(),
            model: "Qwen3.8-Flash".into(),
            gpus: Vec::new(),
            port: 8310,
            process: None,
            launch,
        }
    }

    #[test]
    fn a_fresh_reading_names_every_change_to_the_placement() {
        let before = placement(&[48211], &[8310, 8311]);
        assert!(changed(&before, &before.clone(), &[]).is_empty());
        assert_eq!(
            changed(&before, &placement(&[48211, 61204], &[8310, 8312]), &[]),
            [
                "proc 61204 MiMo-V2.6-Flash newly holds a GPU picked",
                "port 8311 is taken now, so B would listen on 8312",
            ]
        );
        // A process that has left the GPUs changes what the start does.
        assert_eq!(
            changed(&before, &placement(&[], &[8310, 8311]), &[]),
            ["proc 48211 MiMo-V2.6-Flash no longer holds a GPU picked"]
        );
        // The launches replaced are compared by id, not by how many.
        let reservations = [reservation(7, "launch A"), reservation(9, "launch B")];
        let mut first = before.clone();
        first.replaces = vec![7];
        let mut second = before.clone();
        second.replaces = vec![9];
        assert_eq!(
            changed(&first, &second, &reservations),
            [
                "launch B newly holds a GPU or port picked",
                "launch A no longer holds a GPU or port picked",
            ]
        );
        // So is what each instance waits for.
        second.replaces = vec![7];
        second.instances[0].after = vec![Dependency::Launch(7)];
        assert_eq!(
            changed(&first, &second, &reservations),
            ["what A waits for changed"]
        );
    }

    /// The dialog of a start that replaces launch 9 and waits for it.
    fn replacing_nine(reservations: &[Reservation]) -> Result<Confirm, String> {
        let mut placement = placement(&[], &[8310]);
        placement.replaces = vec![9];
        placement.instances[0].after = vec![Dependency::Launch(9)];
        let start = Start {
            model: "qwen38".into(),
            profile: "tp2-mtp".into(),
            placement,
            command: vec!["serve".into()],
            explicit: command::Explicit::default(),
            choice: crate::loader::Choice::default(),
        };
        let check = Check {
            profile: "tp2-mtp".into(),
            settings: BTreeMap::new(),
            model_options: Vec::new(),
            warnings: Vec::new(),
        };
        confirm(
            Box::new(start),
            &Reading {
                catalog: &Catalog::none(),
                check: &check,
                gpus: &[],
                reservations,
                stopping: &|_, _| false,
                warnings: Vec::new(),
                memory: None,
                wall: 0,
            },
        )
    }

    #[test]
    fn a_launch_that_ends_during_the_check_moves_no_dependency_to_another() {
        // Launch 7 ended while serve --check ran, so launch 9 is now first
        // among the reservations: the dialog still names launch 9.
        let confirm = replacing_nine(&[reservation(9, "launch B")]).unwrap();
        assert_eq!(confirm.waits, [vec![Target::Launch(9)]]);
        assert_eq!(confirm.stops.len(), 1);
        assert_eq!(
            (&confirm.stops[0].target, confirm.stops[0].label.as_str()),
            (&Target::Launch(9), "launch B")
        );
        // A replaced launch that ended refuses the start rather than
        // dropping what it waits for.
        let refused = replacing_nine(&[reservation(7, "launch A")]).unwrap_err();
        assert!(refused.contains("has ended"), "{refused}");
    }
}
