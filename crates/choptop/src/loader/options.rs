//! The Options pane: every serve option the catalog lists, the variables
//! serve reads, the package's own options and the custom rows, grouped,
//! each with its value, where the value comes from and a note.

use std::collections::BTreeMap;

use crate::catalog::{
    Catalog, Checkpoint, Model, PackageOption, Profile, ServeOption, Shape, Source,
};
use crate::launch::command::{API_KEY, Key, OWNED, classify};
use crate::launch::placement::parse_listen;

/// A group of the Options pane, in the order shown.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Group {
    Model,
    Scheduling,
    Memory,
    Speculation,
    Prefill,
    Cache,
    Package,
    Server,
    Media,
    Logging,
    Custom,
}

impl Group {
    pub const ALL: [Self; 11] = [
        Self::Model,
        Self::Scheduling,
        Self::Memory,
        Self::Speculation,
        Self::Prefill,
        Self::Cache,
        Self::Package,
        Self::Server,
        Self::Media,
        Self::Logging,
        Self::Custom,
    ];

    pub fn title(self) -> &'static str {
        match self {
            Self::Model => "Model",
            Self::Scheduling => "Scheduling",
            Self::Memory => "Memory and KV",
            Self::Speculation => "Speculation",
            Self::Prefill => "Prefill",
            Self::Cache => "Prefix cache and offload",
            Self::Package => "Package options",
            Self::Server => "Server and API",
            Self::Media => "Media",
            Self::Logging => "Logging and debug",
            Self::Custom => "Custom arguments and environment",
        }
    }

    /// Groups that start closed.
    pub fn starts_closed(self) -> bool {
        matches!(self, Self::Server | Self::Media | Self::Logging)
    }

    /// The pane group of a catalog group; one this choptop does not know
    /// joins Logging and debug.
    fn of(group: &str) -> Self {
        match group {
            "model" => Self::Model,
            "scheduling" => Self::Scheduling,
            "memory" => Self::Memory,
            "speculation" => Self::Speculation,
            "prefill" => Self::Prefill,
            "cache" => Self::Cache,
            "server" => Self::Server,
            "media" => Self::Media,
            _ => Self::Logging,
        }
    }

    /// The pane group of a variable, which the catalog does not group.
    fn of_variable(name: &str) -> Self {
        if name.contains("MODEL_OPTIONS") {
            Self::Package
        } else if name.contains("API_KEY") {
            Self::Server
        } else if name.contains("LOG") {
            Self::Logging
        } else {
            Self::Model
        }
    }
}

/// One row of the Options pane.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Row {
    Group {
        group: Group,
        /// Its option rows, or custom rows.
        count: usize,
        /// How many of them are set.
        set: usize,
        open: bool,
    },
    /// A serve flag or variable, or the checkpoint (`--model`).
    Option(Key),
    /// A package option, by name.
    Package(String),
    /// A line of text in a group: no package options, or why not yet.
    Info(String, Tone),
    /// A custom row, by position.
    Custom(usize),
    /// The custom row being added.
    NewCustom,
    /// `+ add an argument or VAR=value`.
    Add,
}

impl Row {
    /// The value it holds in the form, for `x` and inline edits.
    pub fn key(&self) -> Option<Key> {
        match self {
            Self::Option(key) => Some(key.clone()),
            Self::Package(name) => Some(Key::ModelOption(name.clone())),
            _ => None,
        }
    }
}

/// How a note or a value reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tone {
    /// Dim.
    Hint,
    /// Yellow.
    Warning,
    /// Red.
    Problem,
}

/// Where a value comes from, shown in the FROM column.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum From {
    Set,
    Profile,
    Package,
    Default,
    /// choptop's own environment, or a checkpoint variable.
    Environment,
    /// The remembered checkpoint.
    Saved,
    /// Worked out by choptop: the listen address.
    Auto,
    Missing,
    /// A checkpoint `models pick` has not answered for.
    Checking,
}

impl From {
    pub fn label(self) -> &'static str {
        match self {
            Self::Set => "set",
            Self::Profile => "profile",
            Self::Package => "package",
            Self::Default => "default",
            Self::Environment => "env",
            Self::Saved => "saved",
            Self::Auto => "auto",
            Self::Missing => "missing",
            Self::Checking => "checking",
        }
    }
}

/// Everything a row's value is read from.
pub struct Context<'a> {
    pub catalog: &'a Catalog,
    pub model: Option<&'a Model>,
    pub profile: Option<&'a Profile>,
    pub edits: &'a BTreeMap<Key, String>,
    pub custom: &'a [String],
    /// The package's options once read.
    pub package: Option<&'a [PackageOption]>,
    /// Why they are not read yet: `None` before they are asked for, an
    /// empty text while they are read, or the failure.
    pub package_unread: Option<&'a str>,
    /// The first instance's address.
    pub listen: Option<String>,
    pub tensor_parallel: Option<u32>,
    /// A checkpoint typed in, waiting for `models pick`.
    pub typed_checkpoint: Option<&'a std::path::Path>,
}

/// A row as the pane draws it.
#[derive(Clone, Debug, PartialEq)]
pub struct Line {
    pub name: String,
    pub value: String,
    pub from: From,
    /// The value is wrong.
    pub wrong: bool,
    pub note: Option<(String, Tone)>,
    /// ←→ changes it in place.
    pub choice: bool,
    /// What the help line names: `--max-sequences`, `CHOPINFER_API_KEY=`.
    pub flag: String,
    pub help: String,
}

/// The flags the pane leaves out: those choptop owns (the checkpoint row
/// is `--model`), package options, and the API key, which travels in the
/// environment.
fn hidden(option: &ServeOption) -> bool {
    let name = option.name();
    !option.model_launch
        || (name != "model" && OWNED.iter().any(|(owned, _)| owned[2..] == *name))
        || matches!(name, "model-option" | "list-model-options" | "api-key")
}

impl Context<'_> {
    /// The option rows of each group, in catalog order.
    fn members(&self, group: Group) -> Vec<Row> {
        match group {
            Group::Custom => (0..self.custom.len()).map(Row::Custom).collect(),
            _ => {
                let mut rows = if group == Group::Package {
                    self.package
                        .unwrap_or(&[])
                        .iter()
                        .map(|option| Row::Package(option.name.clone()))
                        .collect()
                } else {
                    Vec::new()
                };
                let Some(options) = self.catalog.options() else {
                    return rows;
                };
                rows.extend(
                    options
                        .options
                        .iter()
                        .filter(|option| {
                            !hidden(option)
                                && group != Group::Package
                                && Group::of(&option.group) == group
                        })
                        .map(|option| Row::Option(Key::Flag(option.name().to_owned()))),
                );
                rows.extend(
                    options
                        .environment
                        .iter()
                        .filter(|variable| Group::of_variable(&variable.name) == group)
                        .map(|variable| Row::Option(Key::Variable(variable.name.clone()))),
                );
                rows
            }
        }
    }

    /// How many option rows there are in all.
    pub fn option_count(&self) -> usize {
        Group::ALL
            .iter()
            .filter(|group| **group != Group::Custom)
            .map(|group| self.members(*group).len())
            .sum()
    }

    /// The pane's rows: groups and, in open groups, their rows; while
    /// `find` holds text, only matching rows, every group with one open.
    pub fn rows(&self, closed: &[Group], find: &str, adding: bool) -> Vec<Row> {
        let find = find.to_lowercase();
        let mut rows = Vec::new();
        for group in Group::ALL {
            let mut members = self.members(group);
            if !find.is_empty() {
                members.retain(|row| self.name(row).to_lowercase().contains(&find));
                if members.is_empty() {
                    continue;
                }
            }
            let set = members
                .iter()
                .filter(|row| match row {
                    Row::Custom(_) => true,
                    row => row.key().is_some_and(|key| self.edits.contains_key(&key)),
                })
                .count();
            let open = !find.is_empty() || !closed.contains(&group);
            rows.push(Row::Group {
                group,
                count: members.len(),
                set,
                open,
            });
            if !open {
                continue;
            }
            if group == Group::Package
                && find.is_empty()
                && !members.iter().any(|row| matches!(row, Row::Package(_)))
            {
                rows.push(self.package_info());
            }
            rows.extend(members);
            if group == Group::Custom && find.is_empty() {
                if adding {
                    rows.push(Row::NewCustom);
                }
                rows.push(Row::Add);
            }
        }
        rows
    }

    fn package_info(&self) -> Row {
        match (self.package, self.package_unread) {
            (Some(_), _) => Row::Info("this package declares none".to_owned(), Tone::Hint),
            (None, Some("")) => Row::Info(
                "reading them with serve --list-model-options".to_owned(),
                Tone::Hint,
            ),
            (None, Some(error)) => Row::Info(error.to_owned(), Tone::Problem),
            (None, None) => Row::Info(
                "read with serve --list-model-options once a checkpoint and GPUs are chosen"
                    .to_owned(),
                Tone::Hint,
            ),
        }
    }

    /// The name a row shows.
    pub fn name(&self, row: &Row) -> String {
        match row {
            Row::Option(Key::Flag(name)) if name == "model" => "checkpoint".to_owned(),
            Row::Option(Key::Flag(name)) => name.clone(),
            Row::Option(Key::Variable(name)) => name.clone(),
            Row::Option(Key::ModelOption(name)) | Row::Package(name) => name.clone(),
            Row::Custom(index) => self.custom.get(*index).cloned().unwrap_or_default(),
            Row::Group { group, .. } => group.title().to_owned(),
            Row::Info(text, _) => text.clone(),
            Row::NewCustom | Row::Add => String::new(),
        }
    }

    fn option(&self, name: &str) -> Option<&ServeOption> {
        self.catalog
            .options()
            .and_then(|options| options.option(name))
    }

    fn package_option(&self, name: &str) -> Option<&PackageOption> {
        self.package?.iter().find(|option| option.name == name)
    }

    fn secret(&self, name: &str) -> bool {
        name == API_KEY
            || self
                .catalog
                .options()
                .and_then(|options| options.variable(name))
                .is_some_and(|variable| variable.secret)
    }

    /// The value of `key` before any edit: the profile's, the package's or
    /// the environment's, and where it comes from.
    pub fn inherited(&self, key: &Key) -> (Option<String>, From) {
        match key {
            Key::Flag(name) if name == "listen" => (self.listen.clone(), From::Auto),
            Key::Flag(name) if name == "served-model-name" => (
                self.model.map(|model| model.served_model_name.clone()),
                From::Package,
            ),
            Key::Flag(name) => match self.profile.and_then(|profile| profile.settings.get(name)) {
                Some(value) => (Some(value.clone()), From::Profile),
                None => (None, From::Default),
            },
            Key::Variable(name) => match self.catalog.environment.get(name) {
                Some(value) => (Some(value.clone()), From::Environment),
                None => (None, From::Default),
            },
            Key::ModelOption(name) => {
                match self
                    .profile
                    .and_then(|profile| profile.model_options.get(name))
                {
                    Some(value) => (Some(value.clone()), From::Profile),
                    None => (
                        self.package_option(name).map(PackageOption::default_text),
                        From::Package,
                    ),
                }
            }
        }
    }

    /// The value of `key` with the form's edit, and where it comes from.
    pub fn effective(&self, key: &Key) -> (Option<String>, From) {
        match self.edits.get(key) {
            Some(value) => (Some(value.clone()), From::Set),
            None => self.inherited(key),
        }
    }

    /// The values ←→ cycles through; empty for a typed value.
    pub fn choices(&self, key: &Key) -> Vec<String> {
        match key {
            Key::Flag(name) => match self.option(name) {
                Some(option) if option.shape == Shape::Switch => {
                    vec!["off".to_owned(), "on".to_owned()]
                }
                Some(option) => option.choices.clone().unwrap_or_default(),
                None => Vec::new(),
            },
            Key::ModelOption(name) => self
                .package_option(name)
                .map(PackageOption::choices)
                .unwrap_or_default(),
            Key::Variable(_) => Vec::new(),
        }
    }

    /// Why the form's value of `key` is wrong; `None` when it is not.
    pub fn problem(&self, key: &Key) -> Option<String> {
        let (value, from) = self.effective(key);
        let value = value?;
        match key {
            Key::Flag(name) => {
                if name == "context-parallel-size"
                    && let (Ok(size), Some(tensor_parallel)) =
                        (value.parse::<u32>(), self.tensor_parallel)
                    && (size == 0 || tensor_parallel % size != 0)
                {
                    return Some(format!("must divide TP{tensor_parallel}"));
                }
                if name == "speculative-rows"
                    && let (Ok(rows), Some(tokens)) = (
                        value.parse::<u64>(),
                        self.effective(&Key::Flag("speculative-tokens".to_owned()))
                            .0
                            .and_then(|tokens| tokens.parse::<u64>().ok()),
                    )
                    && rows <= tokens
                {
                    return Some(format!("must exceed {tokens}"));
                }
                if from != From::Set {
                    return None;
                }
                if name == "listen" {
                    return parse_listen(&value)
                        .is_none()
                        .then(|| "HOST:PORT".to_owned());
                }
                let option = self.option(name)?;
                if let Some(choices) = &option.choices
                    && !choices.contains(&value)
                {
                    return Some(format!("one of {}", choices.join(", ")));
                }
                match option.shape {
                    Shape::Count if value.parse::<u64>().is_err() => {
                        Some("whole number".to_owned())
                    }
                    Shape::Number => match value.parse::<f64>() {
                        Ok(share) if name == "gpu-memory-utilization" => {
                            (!(share > 0.0 && share <= 1.0)).then(|| "share in (0, 1]".to_owned())
                        }
                        Ok(number) if number.is_finite() => None,
                        _ => Some("a number".to_owned()),
                    },
                    Shape::Json => serde_json::from_str::<serde_json::Value>(&value)
                        .is_err()
                        .then(|| "JSON".to_owned()),
                    _ => None,
                }
            }
            Key::Variable(name) => {
                let variable = self.catalog.options()?.variable(name)?;
                (from == From::Set
                    && variable.shape == Shape::Count
                    && value.parse::<u64>().is_err())
                .then(|| "whole number".to_owned())
            }
            Key::ModelOption(name) => self.package_option(name)?.problem(&value),
        }
    }

    /// A warning about the form's value of `key`.
    pub fn warning(&self, key: &Key) -> Option<String> {
        match key {
            Key::Flag(name) if name == "max-sequences" => {
                let sequences = self.edits.get(key)?.parse::<u64>().ok()?;
                let profile = self.profile?;
                let limit = profile
                    .qualified_max_sequences
                    .unwrap_or(profile.max_sequences);
                (sequences > limit).then(|| match profile.qualified_max_sequences {
                    Some(qualified) => format!("above evidence ({qualified})"),
                    None => "no evidence".to_owned(),
                })
            }
            Key::ModelOption(name) => {
                let option = self.package_option(name)?;
                let value = self.effective(key).0?;
                (value != option.default_text()).then(|| "unqualified".to_owned())
            }
            _ => None,
        }
    }

    /// A hint about `key`.
    fn hint(&self, key: &Key) -> Option<String> {
        match key {
            Key::Flag(name) => match name.as_str() {
                "max-sequences" => {
                    let profile = self.profile?;
                    Some(match profile.qualified_max_sequences {
                        Some(qualified) => format!("evidence through {qualified}"),
                        None => "experimental".to_owned(),
                    })
                }
                "graph-memory-reserve-mib" => Some("MiB per rank".to_owned()),
                "speculative-tokens" => Some("0 turns it off".to_owned()),
                "listen" => Some("+1 per instance".to_owned()),
                "model" if self.typed_checkpoint.is_some() => {
                    Some("checking with models pick".to_owned())
                }
                _ => None,
            },
            Key::Variable(_) => Some("environment".to_owned()),
            Key::ModelOption(_) => None,
        }
    }

    /// How `row` is drawn.
    pub fn line(&self, row: &Row) -> Line {
        let mut line = Line {
            name: self.name(row),
            value: String::new(),
            from: From::Default,
            wrong: false,
            note: None,
            choice: false,
            flag: String::new(),
            help: String::new(),
        };
        let Some(key) = row.key() else {
            match row {
                Row::Group { group, open, .. } => {
                    line.flag = group.title().to_owned();
                    line.help = match group {
                        Group::Package => format!(
                            "Options the package declares, read with serve --list-model-options. {}",
                            if *open {
                                "Enter or ← closes the group."
                            } else {
                                "Enter or → opens the group."
                            }
                        ),
                        Group::Custom => "Arguments passed to serve as written, and VAR=value set in its environment. a adds one.".to_owned(),
                        _ if *open => "Enter or ← closes the group.".to_owned(),
                        _ => "Enter or → opens the group.".to_owned(),
                    };
                }
                Row::Custom(_) => {
                    line.flag = "custom".to_owned();
                    line.help = "Passed as written. Enter edits, x removes.".to_owned();
                    let environment = classify(&line.name, None).is_ok_and(|classified| {
                        classified.arguments.is_empty() && !classified.variables.is_empty()
                    });
                    line.note = Some((
                        if environment {
                            "environment"
                        } else {
                            "serve argument"
                        }
                        .to_owned(),
                        Tone::Hint,
                    ));
                }
                Row::NewCustom => {
                    line.flag = "a".to_owned();
                    line.help = "A serve argument such as --kv-offload-prefix off, or VAR=value for the environment. A flag the options above hold is set there.".to_owned();
                }
                Row::Add => {
                    line.flag = "a".to_owned();
                    line.help = "Adds a serve argument such as --kv-offload-prefix off, or VAR=value for the environment.".to_owned();
                }
                Row::Info(..) | Row::Option(_) | Row::Package(_) => {}
            }
            return line;
        };
        let (value, from) = self.effective(&key);
        line.from = from;
        line.choice = !self.choices(&key).is_empty();
        line.value = match (&key, value) {
            (Key::Flag(name), _) if name == "model" => self.checkpoint_value(&mut line),
            (Key::Variable(name), Some(_)) if self.secret(name) => "****".to_owned(),
            (_, Some(value)) => value,
            (_, None) => "-".to_owned(),
        };
        let problem = self.problem(&key);
        line.wrong = problem.is_some();
        line.note = problem
            .map(|problem| (problem, Tone::Problem))
            .or_else(|| self.warning(&key).map(|warning| (warning, Tone::Warning)))
            .or_else(|| self.hint(&key).map(|hint| (hint, Tone::Hint)));
        match &key {
            Key::Flag(name) => {
                line.flag = format!("--{name}");
                line.help = self
                    .option(name)
                    .map(|option| option.help.clone())
                    .unwrap_or_default();
                if name == "listen" {
                    line.help = "Address of the first instance, from serve's default 127.0.0.1:8310; each further instance takes the next free port.".to_owned();
                }
            }
            Key::Variable(name) => {
                line.flag = format!("{name}=");
                line.help = self
                    .catalog
                    .options()
                    .and_then(|options| options.variable(name))
                    .map(|variable| variable.help.clone())
                    .unwrap_or_default();
            }
            Key::ModelOption(name) => {
                line.flag = format!("--model-option {name}=");
                line.help = "A value other than the package default is unqualified.".to_owned();
            }
        }
        line
    }

    /// The checkpoint row's value, which also sets its FROM column.
    fn checkpoint_value(&self, line: &mut Line) -> String {
        if let Some(directory) = self.typed_checkpoint {
            line.from = From::Set;
            return directory.display().to_string();
        }
        let Some(model) = self.model else {
            line.from = From::Missing;
            return "not set".to_owned();
        };
        match self.catalog.checkpoint(&model.model) {
            Checkpoint::Found {
                directory,
                source: Source::Saved,
            } => {
                line.from = From::Saved;
                directory.display().to_string()
            }
            Checkpoint::Found {
                source: Source::Environment(name),
                ..
            } => {
                line.from = From::Environment;
                format!("${name}")
            }
            Checkpoint::Checking => {
                line.from = From::Checking;
                "-".to_owned()
            }
            Checkpoint::Missing => {
                line.from = From::Missing;
                "not set".to_owned()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::catalog::parse_package_options;
    use crate::loader::Loader;
    use crate::loader::tests::{catalog, live};

    const PACKAGE: &str = include_str!("../../tests/fixtures/contracts/list-model-options.json");

    /// A loader on DeepSeek's tp4-dspark at TP4 with the package's options
    /// read.
    fn loader() -> (Loader, Catalog) {
        let mut catalog = catalog();
        let mut loader = Loader::open(Some(vec![0, 1, 2, 3]), Vec::new());
        loader.refresh(&mut catalog, &live(), &[], &[], 1.0, 1.5, &mut |_, _| true);
        let generation = catalog
            .outbox
            .iter()
            .find_map(|request| match request {
                crate::catalog::Request::PackageOptions { generation, .. } => Some(*generation),
                _ => None,
            })
            .unwrap();
        loader.reply(crate::catalog::Reply::PackageOptions(
            generation,
            parse_package_options(PACKAGE),
        ));
        loader.refresh(&mut catalog, &live(), &[], &[], 1.0, 1.5, &mut |_, _| true);
        (loader, catalog)
    }

    fn line(loader: &Loader, catalog: &Catalog, key: Key) -> Line {
        loader.context(catalog).line(&Row::Option(key))
    }

    #[test]
    fn groups_follow_the_catalog_and_three_start_closed() {
        let (loader, catalog) = loader();
        let rows = loader.rows(&catalog);
        let groups = rows
            .iter()
            .filter_map(|row| match row {
                Row::Group { group, open, .. } => Some((*group, *open)),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(
            groups,
            [
                (Group::Model, true),
                (Group::Scheduling, true),
                (Group::Memory, true),
                (Group::Speculation, true),
                (Group::Prefill, true),
                (Group::Cache, true),
                (Group::Package, true),
                (Group::Server, false),
                (Group::Media, false),
                (Group::Logging, false),
                (Group::Custom, true),
            ]
        );
        // --bundle does not apply to serve --model; --model is the
        // checkpoint row; the closed groups show no rows.
        let flags = rows
            .iter()
            .filter_map(|row| match row {
                Row::Option(Key::Flag(name)) => Some(name.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(
            flags,
            [
                "model",
                "modalities",
                "max-sequences",
                "gpu-memory-utilization",
                "prefix-cache"
            ]
        );
        assert_eq!(rows.last(), Some(&Row::Add));
        let context = loader.context(&catalog);
        assert_eq!(
            context.name(&Row::Option(Key::Flag("model".into()))),
            "checkpoint"
        );
        // Every flag and variable but the hidden ones, and the package's.
        assert_eq!(context.option_count(), 9 + 2 + 4);
    }

    #[test]
    fn values_say_where_they_come_from() {
        let (mut loader, catalog) = loader();
        let checkpoint = line(&loader, &catalog, Key::Flag("model".into()));
        assert_eq!(
            (checkpoint.value.as_str(), checkpoint.from),
            ("$CHOPINFER_DEEPSEEK_CHECKPOINT", From::Environment)
        );
        let default = line(&loader, &catalog, Key::Flag("max-sequences".into()));
        assert_eq!((default.value.as_str(), default.from), ("-", From::Default));
        // An experimental profile has no evidence to stay under.
        assert_eq!(default.note, Some(("experimental".to_owned(), Tone::Hint)));
        loader
            .edits
            .insert(Key::Flag("max-sequences".into()), "48".into());
        let set = line(&loader, &catalog, Key::Flag("max-sequences".into()));
        assert_eq!((set.value.as_str(), set.from), ("48", From::Set));
        assert_eq!(set.note, Some(("no evidence".to_owned(), Tone::Warning)));
        loader
            .edits
            .insert(Key::Flag("max-sequences".into()), "4x".into());
        let wrong = line(&loader, &catalog, Key::Flag("max-sequences".into()));
        assert!(wrong.wrong);
        assert_eq!(wrong.note, Some(("whole number".to_owned(), Tone::Problem)));
        loader
            .edits
            .insert(Key::Flag("gpu-memory-utilization".into()), "1.5".into());
        assert_eq!(
            line(
                &loader,
                &catalog,
                Key::Flag("gpu-memory-utilization".into())
            )
            .note,
            Some(("share in (0, 1]".to_owned(), Tone::Problem))
        );
        // A secret is never shown.
        loader
            .edits
            .insert(Key::Variable("CHOPINFER_API_KEY".into()), "secret".into());
        let key = line(&loader, &catalog, Key::Variable("CHOPINFER_API_KEY".into()));
        assert_eq!(
            (key.value.as_str(), key.flag.as_str()),
            ("****", "CHOPINFER_API_KEY=")
        );
        assert_eq!(key.note, Some(("environment".to_owned(), Tone::Hint)));
        // A choice says so; a switch reads as off until set.
        let cache = line(&loader, &catalog, Key::Flag("prefix-cache".into()));
        assert!(cache.choice);
        assert_eq!(
            loader
                .context(&catalog)
                .choices(&Key::Flag("allow-credentials".into())),
            ["off", "on"]
        );
    }

    #[test]
    fn package_options_read_their_profile_and_mark_other_values_unqualified() {
        let (mut loader, catalog) = loader();
        let rows = loader.rows(&catalog);
        assert!(rows.contains(&Row::Package("dspark_draft_chunk_items".into())));
        let context = loader.context(&catalog);
        let items = context.line(&Row::Package("dspark_draft_chunk_items".into()));
        // The profile's value is the package default: qualified.
        assert_eq!(
            (items.value.as_str(), items.from, items.note.clone()),
            ("12", From::Profile, None)
        );
        let skip = context.line(&Row::Package("prefill_upper_skip".into()));
        assert_eq!((skip.value.as_str(), skip.from), ("true", From::Profile));
        let chunks = context.line(&Row::Package("prefill_chunk_tokens".into()));
        assert_eq!(
            (chunks.value.as_str(), chunks.from),
            ("8192", From::Package)
        );
        assert!(chunks.choice);
        loader.edits.insert(
            Key::ModelOption("dspark_draft_chunk_items".into()),
            "16".into(),
        );
        let set = loader
            .context(&catalog)
            .line(&Row::Package("dspark_draft_chunk_items".into()));
        assert_eq!(set.note, Some(("unqualified".to_owned(), Tone::Warning)));
        loader.edits.insert(
            Key::ModelOption("dspark_draft_chunk_items".into()),
            "64".into(),
        );
        let wrong = loader
            .context(&catalog)
            .line(&Row::Package("dspark_draft_chunk_items".into()));
        assert_eq!(wrong.note, Some(("from 1 to 32".to_owned(), Tone::Problem)));
        assert!(loader.plan.problems.is_empty());
    }

    #[test]
    fn find_shows_matching_rows_in_open_groups() {
        let (mut loader, catalog) = loader();
        loader.find = "api".into();
        let rows = loader.rows(&catalog);
        assert_eq!(
            rows,
            [
                Row::Group {
                    group: Group::Server,
                    count: 1,
                    set: 0,
                    open: true
                },
                Row::Option(Key::Variable("CHOPINFER_API_KEY".into())),
            ]
        );
    }
}
