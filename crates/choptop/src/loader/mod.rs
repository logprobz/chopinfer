//! The model loader's state: the four panes (Models, Profiles, Placement,
//! Options), the cursor in each, the form's values and custom rows, the
//! find text and the inline edit, and what it works out from them each
//! frame: who holds the GPUs, where instances go, the command line and
//! what blocks a start.

pub mod options;

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

use crossterm::event::KeyCode;

use crate::catalog::{
    Availability, Catalog, Checkpoint, Fetch, Model, PackageOption, Pick, Profile, Reply, Request,
};
use crate::instances::Instance;
use crate::launch::Reservation;
use crate::launch::command::{self, Explicit, Form, Key, classify};
use crate::launch::occupancy::{self, Gpu, Inputs, Live, Occupancy};
use crate::launch::placement::{self, Field, Placement};
use crate::ui::widgets::sanitize;
use options::{Context, Group, Line, Row};

/// Seconds a message replaces the command box's summary.
pub const FLASH_SECONDS: f64 = 4.0;

/// The panes, in Tab order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pane {
    Models,
    Profiles,
    Placement,
    Options,
}

impl Pane {
    const ORDER: [Self; 4] = [Self::Models, Self::Profiles, Self::Placement, Self::Options];

    fn step(self, back: bool) -> Self {
        let index = Self::ORDER
            .iter()
            .position(|pane| *pane == self)
            .unwrap_or(0);
        Self::ORDER[(index + if back { 3 } else { 1 }) % 4]
    }

    pub fn title(self) -> &'static str {
        match self {
            Self::Models => "Models",
            Self::Profiles => "Profiles",
            Self::Placement => "Placement",
            Self::Options => "Options",
        }
    }

    /// The pane Tab goes to next.
    pub fn next(self) -> Self {
        self.step(false)
    }
}

/// What an inline edit changes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Target {
    Value(Key),
    Custom(usize),
    NewCustom,
}

/// An inline edit: a value cell or a custom row being typed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Edit {
    pub target: Target,
    pub text: String,
}

/// A message that replaces the command box's summary for a few seconds.
#[derive(Clone, Debug, PartialEq)]
pub struct Flash {
    pub text: String,
    pub good: bool,
    pub at: f64,
}

/// The choices of a launch, which a recent launch restores.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Choice {
    pub model: String,
    pub profile: String,
    pub gpus: BTreeSet<u32>,
    pub edits: BTreeMap<Key, String>,
    pub custom: Vec<String>,
}

/// A recent launch in the Models pane.
#[derive(Clone, Debug, PartialEq)]
pub struct Recent {
    /// `Qwen tp2-mtp x2`.
    pub label: String,
    /// When it ran, as `3d ago`.
    pub when: String,
    pub choice: Choice,
}

/// What `S` does once the form passes its checks: the switch to confirm.
#[derive(Clone, Debug, PartialEq)]
pub struct Start {
    pub model: String,
    pub profile: String,
    pub placement: Placement,
    /// The final `serve` arguments, without `--listen`.
    pub command: Vec<String>,
    pub explicit: Explicit,
}

/// What a key did in the loader.
#[derive(Clone, Debug, PartialEq)]
pub enum Outcome {
    Handled,
    /// Leave the loader.
    Back,
    Help,
    /// The form passed; confirm and launch.
    Start(Box<Start>),
    /// Not a loader key.
    Unhandled,
}

/// What the loader works out each frame.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Plan {
    /// The form's revision it was worked out for.
    pub revision: u64,
    pub gpus: Vec<Gpu>,
    pub placement: Placement,
    pub explicit: Explicit,
    pub tensor_parallel: Option<u32>,
    /// What must change before a start, counted in the command box.
    pub problems: Vec<String>,
    /// What a start waits for.
    pub waits: Vec<String>,
    /// The final `serve` arguments once `models command` answered for this
    /// form.
    pub command: Option<Vec<String>>,
    /// The checkpoint directory.
    pub checkpoint: Option<PathBuf>,
}

/// What a package options request is asked for.
#[derive(Clone, Debug, PartialEq)]
struct PackageQuery {
    checkpoint: PathBuf,
    tensor_parallel: u32,
    profile: String,
    /// The form's options, which `models command` merges over the profile.
    options: Vec<String>,
    environment: Vec<(String, String)>,
}

/// What the plan is worked out from, to tell when a key changed it.
#[derive(Clone, Debug, PartialEq)]
struct Snapshot {
    model: Option<String>,
    profile: Option<String>,
    picked: BTreeSet<u32>,
    edits: BTreeMap<Key, String>,
    custom: Vec<String>,
    typed_checkpoint: Option<PathBuf>,
}

/// The model loader.
#[derive(Clone, Debug)]
pub struct Loader {
    pub pane: Pane,
    /// The row of the Models pane: a model, then the recent launches.
    pub model_cursor: usize,
    pub model: Option<String>,
    pub profile: Option<String>,
    /// The profile was chosen in the Profiles pane; until then it follows
    /// what `serve --model` picks for the GPUs.
    profile_chosen: bool,
    pub picked: BTreeSet<u32>,
    /// GPUs to preselect at the next refresh: the opened row's, or the
    /// free ones when empty.
    preselect: Option<Vec<u32>>,
    pub gpu_cursor: usize,
    pub edits: BTreeMap<Key, String>,
    pub custom: Vec<String>,
    pub option_cursor: usize,
    pub option_scroll: usize,
    pub closed: Vec<Group>,
    pub find: String,
    pub finding: bool,
    pub edit: Option<Edit>,
    pub flash: Option<Flash>,
    pub recent: Vec<Recent>,
    /// A checkpoint typed in, kept once `models pick` names its model.
    pub typed_checkpoint: Option<PathBuf>,
    /// Counts changes to the form: the TUI takes several keys before it
    /// works out the plan again, and a plan of an earlier revision does
    /// not start.
    revision: u64,
    pub plan: Plan,
    /// The `models command` asked last, the catalog's ticket of the
    /// request whose answer is awaited, and its answer.
    command_asked: Option<(PathBuf, u32, String, Vec<String>)>,
    command_ticket: Option<u64>,
    command_answer: Option<Result<Vec<String>, String>>,
    /// The package options asked last, by what the request runs with.
    package_asked: Option<PackageQuery>,
    package_ticket: Option<u64>,
    pub package: Option<Fetch<Vec<PackageOption>>>,
}

impl Default for Loader {
    fn default() -> Self {
        Self {
            pane: Pane::Models,
            model_cursor: 0,
            model: None,
            profile: None,
            profile_chosen: false,
            picked: BTreeSet::new(),
            preselect: None,
            gpu_cursor: 0,
            edits: BTreeMap::new(),
            custom: Vec::new(),
            option_cursor: 1,
            option_scroll: 0,
            closed: Group::ALL
                .into_iter()
                .filter(|group| group.starts_closed())
                .collect(),
            find: String::new(),
            finding: false,
            edit: None,
            flash: None,
            recent: Vec::new(),
            typed_checkpoint: None,
            revision: 0,
            plan: Plan::default(),
            command_asked: None,
            command_ticket: None,
            command_answer: None,
            package_asked: None,
            package_ticket: None,
            package: None,
        }
    }
}

/// The tensor parallel size `count` GPUs allow among `sizes`: the largest
/// that splits them evenly, else the largest that fits, else the smallest.
fn allowed(sizes: &[u32], count: usize) -> Option<u32> {
    let count = u32::try_from(count).unwrap_or(u32::MAX);
    sizes
        .iter()
        .copied()
        .filter(|size| *size > 0 && *size <= count && count % size == 0)
        .max()
        .or_else(|| sizes.iter().copied().filter(|size| *size <= count).max())
        .or_else(|| sizes.iter().copied().min())
}

impl Loader {
    /// A loader opened on a row whose GPUs are `gpus`, or on none.
    pub fn open(gpus: Option<Vec<u32>>, recent: Vec<Recent>) -> Self {
        Self {
            preselect: Some(gpus.unwrap_or_default()),
            recent,
            ..Self::default()
        }
    }

    pub fn flash(&mut self, now: f64, text: impl Into<String>, good: bool) {
        self.flash = Some(Flash {
            text: text.into(),
            good,
            at: now,
        });
    }

    /// The chosen model in `catalog`.
    pub fn model<'a>(&self, catalog: &'a Catalog) -> Option<&'a Model> {
        catalog.model(self.model.as_deref()?)
    }

    pub fn profile<'a>(&self, catalog: &'a Catalog) -> Option<&'a Profile> {
        catalog.profile(self.model.as_deref()?, self.profile.as_deref()?)
    }

    /// The chosen model's profiles, with the ones only its pick names.
    pub fn profiles<'a>(&self, catalog: &'a Catalog) -> Vec<&'a Profile> {
        self.model
            .as_deref()
            .map_or_else(Vec::new, |model| catalog.profiles(model))
    }

    /// The pick for the chosen model's checkpoint, once read.
    pub fn pick<'a>(&self, catalog: &'a Catalog) -> Option<&'a Pick> {
        let model = self.model.as_deref()?;
        catalog
            .picked(catalog.checkpoint(model).directory()?)
            .filter(|pick| pick.model.as_deref() == Some(model))
    }

    /// The flags that make the pick serve the chosen profile.
    pub fn profile_arguments(&self, catalog: &Catalog) -> Vec<String> {
        let (Some(pick), Some(profile)) = (self.pick(catalog), self.profile(catalog)) else {
            return Vec::new();
        };
        match pick
            .size(profile.tensor_parallel)
            .and_then(|size| size.profile(&profile.name))
            .map(|profile| &profile.availability)
        {
            Some(Availability::Available { arguments, .. }) => arguments.clone(),
            _ => Vec::new(),
        }
    }

    /// The model options the chosen profile inherits from the pick.
    fn inherited(&self, catalog: &Catalog) -> Vec<String> {
        let (Some(pick), Some(profile)) = (self.pick(catalog), self.profile(catalog)) else {
            return Vec::new();
        };
        match pick
            .size(profile.tensor_parallel)
            .and_then(|size| size.profile(&profile.name))
            .map(|profile| &profile.availability)
        {
            Some(Availability::Available {
                inherited_model_options,
                ..
            }) => inherited_model_options.clone(),
            _ => Vec::new(),
        }
    }

    /// The options pane's context.
    pub fn context<'a>(&'a self, catalog: &'a Catalog) -> Context<'a> {
        Context {
            catalog,
            model: self.model(catalog),
            profile: self.profile(catalog),
            edits: &self.edits,
            custom: &self.custom,
            package: self
                .package
                .as_ref()
                .and_then(Fetch::ready)
                .map(Vec::as_slice),
            package_unread: match &self.package {
                Some(Fetch::Waiting) => Some(""),
                Some(Fetch::Failed(error)) => Some(error),
                _ => None,
            },
            listen: self
                .plan
                .placement
                .instances
                .first()
                .map(placement::Planned::listen),
            tensor_parallel: self.plan.tensor_parallel,
            typed_checkpoint: self.typed_checkpoint.as_deref(),
        }
    }

    /// The Options pane's rows.
    pub fn rows(&self, catalog: &Catalog) -> Vec<Row> {
        self.context(catalog).rows(
            &self.closed,
            &self.find,
            self.edit
                .as_ref()
                .is_some_and(|edit| edit.target == Target::NewCustom),
        )
    }

    /// The Options pane's rows with how each is drawn.
    pub fn lines(&self, catalog: &Catalog) -> Vec<(Row, Line)> {
        let context = self.context(catalog);
        self.rows(catalog)
            .into_iter()
            .map(|row| {
                let line = context.line(&row);
                (row, line)
            })
            .collect()
    }

    /// The profile `serve --model` picks at the size the picked GPUs allow.
    fn auto_profile(&self, catalog: &Catalog) -> Option<String> {
        let profiles = self.profiles(catalog);
        let pick = self.pick(catalog);
        let mut sizes = match pick {
            Some(pick) => pick.sizes.iter().map(|size| size.tensor_parallel).collect(),
            None => profiles
                .iter()
                .map(|profile| profile.tensor_parallel)
                .collect::<Vec<_>>(),
        };
        sizes.sort_unstable();
        sizes.dedup();
        let size = allowed(&sizes, self.picked.len())?;
        pick.and_then(|pick| pick.size(size)?.pick.clone())
            .filter(|name| profiles.iter().any(|profile| profile.name == *name))
            .or_else(|| {
                profiles
                    .iter()
                    .find(|profile| profile.tensor_parallel == size)
                    .map(|profile| profile.name.clone())
            })
            .or_else(|| profiles.first().map(|profile| profile.name.clone()))
    }

    fn choose_model(&mut self, model: String) {
        if self.model.as_ref() == Some(&model) {
            return;
        }
        self.model = Some(model);
        self.edits.clear();
        self.custom.clear();
        self.typed_checkpoint = None;
        self.profile = None;
        self.profile_chosen = false;
        self.option_cursor = 1;
        self.option_scroll = 0;
    }

    /// Works out the plan from the latest readings and catalog, asks
    /// `chopinfer` what the plan needs, and keeps a typed checkpoint once
    /// `models pick` names its model.
    #[allow(clippy::too_many_arguments)]
    pub fn refresh(
        &mut self,
        catalog: &mut Catalog,
        live: &Live,
        instances: &[Instance],
        reservations: &[Reservation],
        interval: f64,
        now: f64,
        probe: &mut dyn FnMut(&str, u16) -> bool,
    ) {
        let before = self.snapshot();
        self.keep_typed_checkpoint(catalog, now);
        let inputs = Inputs {
            live,
            instances,
            reservations,
            interval,
            now,
        };
        let gpus = occupancy::occupancy(&inputs);
        if let Some(wanted) = self.preselect.take() {
            let known = |index: &u32| {
                gpus.iter().any(|gpu| {
                    gpu.index == *index && !matches!(gpu.occupancy, Occupancy::Unknown(_))
                })
            };
            let held = wanted.into_iter().filter(known).collect::<BTreeSet<_>>();
            self.picked = if held.is_empty() {
                gpus.iter()
                    .filter(|gpu| gpu.occupancy == Occupancy::Free)
                    .map(|gpu| gpu.index)
                    .collect()
            } else {
                held
            };
        }
        if self.model.is_none() {
            let models = catalog.models();
            let first = models
                .iter()
                .find(|model| catalog.checkpoint(&model.model).directory().is_some())
                .or(models.first());
            if let Some(first) = first {
                self.model_cursor = models
                    .iter()
                    .position(|model| model.model == first.model)
                    .unwrap_or(0);
                self.model = Some(first.model.clone());
            }
        }
        let valid = self.profile(catalog).is_some();
        if !self.profile_chosen || !valid {
            self.profile = self.auto_profile(catalog);
            self.profile_chosen = self.profile_chosen && valid;
        }
        if self.snapshot() != before {
            self.revision += 1;
        }
        let ports = occupancy::ports(&inputs);
        let tensor_parallel = self.profile(catalog).map(|profile| profile.tensor_parallel);
        let explicit = command::explicit(
            &Form {
                edits: &self.edits,
                custom: &self.custom,
            },
            catalog.options(),
            &self.inherited(catalog),
            self.package
                .as_ref()
                .and_then(Fetch::ready)
                .map(Vec::as_slice),
        );
        let placement = placement::place(
            &Field {
                gpus: &gpus,
                ports: &ports,
                reservations,
            },
            &self.picked,
            tensor_parallel.unwrap_or(1),
            explicit.listen.as_deref(),
            probe,
        );
        let checkpoint = self
            .model
            .as_deref()
            .and_then(|model| catalog.checkpoint(model).directory().map(PathBuf::from));
        self.plan = Plan {
            revision: self.revision,
            gpus,
            placement,
            explicit,
            tensor_parallel,
            problems: Vec::new(),
            waits: Vec::new(),
            command: None,
            checkpoint,
        };
        self.ask(catalog);
        self.check(catalog);
    }

    /// Keeps a typed checkpoint once `models pick` names its model, or
    /// drops it with why.
    fn keep_typed_checkpoint(&mut self, catalog: &mut Catalog, now: f64) {
        let Some(directory) = self.typed_checkpoint.clone() else {
            return;
        };
        let shown = sanitize(&directory.display().to_string());
        match catalog.picks.get(&directory) {
            None => catalog.pick(directory),
            Some(Fetch::Waiting) => {}
            Some(Fetch::Failed(error)) => {
                let error = sanitize(error);
                self.typed_checkpoint = None;
                self.flash(now, format!("Not kept: {error}"), false);
            }
            Some(Fetch::Ready(Pick {
                model: None,
                refusal,
                ..
            })) => {
                let refusal = sanitize(refusal.as_deref().unwrap_or("no installed model takes it"));
                self.typed_checkpoint = None;
                self.flash(now, format!("Not kept: {shown}: {refusal}"), false);
            }
            Some(Fetch::Ready(Pick {
                model: Some(model), ..
            })) => {
                let model = model.clone();
                self.typed_checkpoint = None;
                match catalog.checkpoints.remember(&model, &directory) {
                    Err(error) => self.flash(now, sanitize(&error), false),
                    Ok(()) if self.model.as_deref() == Some(model.as_str()) => {
                        self.flash(
                            now,
                            format!("{shown} is kept as the {model} checkpoint"),
                            true,
                        );
                    }
                    Ok(()) => self.flash(
                        now,
                        format!("{shown} is a {model} checkpoint, so it is kept for {model}"),
                        true,
                    ),
                }
            }
        }
    }

    /// Asks for the package's options and the command line when what they
    /// depend on changed.
    fn ask(&mut self, catalog: &mut Catalog) {
        let (Some(checkpoint), Some(tensor_parallel), Some(profile)) = (
            self.plan.checkpoint.clone(),
            self.plan.tensor_parallel,
            self.profile.clone(),
        ) else {
            self.forget_package();
            self.forget_command();
            return;
        };
        // The package's options are read for the package the command
        // prepares: the form's options, the preparation ones among them,
        // merged over the profile by models command, with the variables the
        // first instance runs with. Model option values are left out: they
        // do not change what the package declares, and the resets of the
        // inherited ones wait for these defaults.
        let options = self
            .plan
            .explicit
            .options
            .iter()
            .filter(|(group, _)| group.first().map(String::as_str) != Some("--model-option"))
            .flat_map(|(group, _)| group.iter().cloned())
            .collect();
        let instances = &self.plan.placement.instances;
        let mut environment = command::environment(
            &self.plan.explicit,
            instances.first().map_or(&[][..], |first| &first.uuids),
        );
        // Before an instance is placed, no GPUs are named.
        if instances.is_empty() {
            environment.remove(0);
        }
        let package = PackageQuery {
            checkpoint: checkpoint.clone(),
            tensor_parallel,
            profile: profile.clone(),
            options,
            environment,
        };
        if self.package_asked.as_ref() != Some(&package) {
            let ticket = catalog.ticket();
            self.package_ticket = Some(ticket);
            self.package = Some(Fetch::Waiting);
            catalog.send(Request::PackageOptions {
                generation: ticket,
                checkpoint: package.checkpoint.clone(),
                tensor_parallel,
                profile: package.profile.clone(),
                options: package.options.clone(),
                environment: package.environment.clone(),
            });
            self.package_asked = Some(package);
        }
        let explicit = &self.plan.explicit;
        if !explicit.blocked.is_empty() || !explicit.waiting.is_empty() {
            self.forget_command();
            return;
        }
        let command_key = (checkpoint, tensor_parallel, profile, explicit.arguments());
        if self.command_asked.as_ref() != Some(&command_key) {
            let ticket = catalog.ticket();
            self.command_ticket = Some(ticket);
            self.command_answer = None;
            catalog.send(Request::Command {
                generation: ticket,
                checkpoint: command_key.0.clone(),
                tensor_parallel,
                profile: command_key.2.clone(),
                options: command_key.3.clone(),
            });
            self.command_asked = Some(command_key);
        }
    }

    /// Drops the package's options once what they were read for changed,
    /// and any answer still on its way.
    fn forget_package(&mut self) {
        if self.package_asked.take().is_some() {
            self.package_ticket = None;
            self.package = None;
        }
    }

    /// Drops the command line once the form it was asked for changed.
    fn forget_command(&mut self) {
        if self.command_asked.take().is_some() {
            self.command_ticket = None;
            self.command_answer = None;
        }
    }

    /// Takes a reply to a request the loader made: only the answer to the
    /// request it awaits, not one of an earlier form or an earlier loader.
    pub fn reply(&mut self, reply: Reply) {
        match reply {
            Reply::Command(ticket, result) if Some(ticket) == self.command_ticket => {
                self.command_answer = Some(result);
            }
            Reply::PackageOptions(ticket, result) if Some(ticket) == self.package_ticket => {
                self.package = Some(result.into());
            }
            _ => {}
        }
    }

    /// Works out what blocks a start, and what it waits for.
    fn check(&mut self, catalog: &Catalog) {
        let mut problems = Vec::new();
        let mut waits = Vec::new();
        if let Err(error) = &catalog.binary {
            problems.push(error.clone());
        }
        for fetch in [
            catalog.options.as_ref().map(|fetch| match fetch {
                Fetch::Failed(error) => Err(error.clone()),
                Fetch::Waiting => Ok(false),
                Fetch::Ready(_) => Ok(true),
            }),
            catalog.models.as_ref().map(|fetch| match fetch {
                Fetch::Failed(error) => Err(error.clone()),
                Fetch::Waiting => Ok(false),
                Fetch::Ready(_) => Ok(true),
            }),
        ]
        .into_iter()
        .flatten()
        {
            match fetch {
                Err(error) => problems.push(error),
                Ok(false) => waits.push("the catalog from chopinfer".to_owned()),
                Ok(true) => {}
            }
        }
        waits.dedup();
        let model = self.model(catalog);
        let profile = self.profile(catalog);
        if catalog
            .models
            .as_ref()
            .is_some_and(|fetch| fetch.ready().is_some())
            && model.is_none()
        {
            problems.push("no model is installed beside chopinfer".to_owned());
        }
        if let Some(model) = model {
            match (catalog.checkpoint(&model.model), &self.typed_checkpoint) {
                // A checkpoint typed in replaces the one found before it, so
                // nothing starts until models pick has named its model.
                (_, Some(_)) => waits.push("models pick for the checkpoint typed in".to_owned()),
                (Checkpoint::Missing, None) => problems.push(format!(
                    "no {} checkpoint: set its CHOPINFER_*_CHECKPOINT variable or enter one in the checkpoint row",
                    model.model
                )),
                (Checkpoint::Checking, None) => {
                    waits.push("models pick for the checkpoint".to_owned());
                }
                (Checkpoint::Found { .. }, None) => {}
            }
        }
        if let (Some(profile), Some(pick)) = (profile, self.pick(catalog)) {
            match pick.size(profile.tensor_parallel) {
                None => problems.push(format!(
                    "serve --model does not serve this checkpoint at TP{}",
                    profile.tensor_parallel
                )),
                Some(size) => match size.profile(&profile.name).map(|seen| &seen.availability) {
                    Some(Availability::Unavailable(reason)) => problems.push(format!(
                        "{} cannot run here: {}",
                        profile.name,
                        sanitize(reason)
                    )),
                    Some(Availability::Available { .. }) => {}
                    None => match &size.refusal {
                        Some(refusal) => problems.push(format!(
                            "serve --model refuses TP{}: {}",
                            profile.tensor_parallel,
                            sanitize(refusal)
                        )),
                        None => problems.push(format!(
                            "models pick says nothing of {} at TP{}",
                            profile.name, profile.tensor_parallel
                        )),
                    },
                },
            }
        }
        if let Some(grouping) = &self.plan.placement.grouping {
            problems.push(grouping.clone());
        }
        problems.extend(self.plan.placement.problems.iter().cloned());
        let context = self.context(catalog);
        let mut keys = context
            .rows(&[], "", false)
            .into_iter()
            .filter_map(|row| row.key())
            .collect::<Vec<_>>();
        keys.extend(self.edits.keys().cloned());
        keys.dedup();
        let mut seen = BTreeSet::new();
        for key in keys {
            if !seen.insert(key.clone()) {
                continue;
            }
            if let Some(problem) = context.problem(&key) {
                problems.push(format!("{}: {problem}", key.shown()));
            }
        }
        problems.extend(self.plan.explicit.blocked.iter().cloned());
        match &self.package {
            Some(Fetch::Waiting) => waits.push("the package's options".to_owned()),
            Some(Fetch::Failed(error)) => problems.push(sanitize(error)),
            _ => {}
        }
        waits.extend(self.plan.explicit.waiting.iter().cloned());
        if self.command_asked.is_some() {
            match &self.command_answer {
                None => waits.push("the command line from chopinfer models command".to_owned()),
                Some(Ok(arguments)) => self.plan.command = Some(arguments.clone()),
                Some(Err(error)) => problems.push(sanitize(error)),
            }
        }
        if self
            .plan
            .placement
            .instances
            .iter()
            .any(|planned| planned.uuids.len() != planned.gpus.len())
        {
            problems.push("a picked GPU has no UUID to place it by".to_owned());
        }
        self.plan.problems = problems;
        self.plan.waits = waits;
    }

    /// The switch `S` starts, or why not.
    fn start(&self) -> Result<Start, String> {
        // The checkpoint row names a directory the plan does not use yet.
        if self.typed_checkpoint.is_some() {
            return Err("still checking the checkpoint".to_owned());
        }
        // The checks and the command line must be the current form's.
        if self.plan.revision != self.revision {
            return Err("still computing the command".to_owned());
        }
        if !self.plan.problems.is_empty() {
            return Err(self.plan.problems.join("; "));
        }
        if !self.plan.waits.is_empty() {
            return Err(format!("waiting for {}", self.plan.waits.join(", ")));
        }
        let (Some(model), Some(profile), Some(command)) =
            (&self.model, &self.profile, &self.plan.command)
        else {
            return Err("nothing to start".to_owned());
        };
        Ok(Start {
            model: model.clone(),
            profile: profile.clone(),
            placement: self.plan.placement.clone(),
            command: command.clone(),
            explicit: self.plan.explicit.clone(),
        })
    }

    /// Sets `key` to `value`, or drops the edit when that is the value it
    /// has anyway.
    fn set(&mut self, catalog: &Catalog, key: Key, value: String) {
        let inherited = self.context(catalog).inherited(&key).0;
        if inherited.as_ref() == Some(&value) {
            self.edits.remove(&key);
        } else {
            self.edits.insert(key, value);
        }
    }

    /// Moves a choice of `key` one step.
    fn cycle(&mut self, catalog: &Catalog, key: &Key, forward: bool) {
        let context = self.context(catalog);
        let choices = context.choices(key);
        if choices.is_empty() {
            return;
        }
        let current = context.effective(key).0;
        let position = current
            .and_then(|value| choices.iter().position(|choice| *choice == value))
            .or_else(|| {
                // A switch reads as off until set.
                (choices.len() == 2 && choices[0] == "off").then_some(0)
            });
        let next = match (position, forward) {
            (Some(index), true) => (index + 1) % choices.len(),
            (Some(index), false) => (index + choices.len() - 1) % choices.len(),
            (None, true) => 0,
            (None, false) => choices.len() - 1,
        };
        self.set(catalog, key.clone(), choices[next].clone());
    }

    /// What the plan is worked out from now.
    fn snapshot(&self) -> Snapshot {
        Snapshot {
            model: self.model.clone(),
            profile: self.profile.clone(),
            picked: self.picked.clone(),
            edits: self.edits.clone(),
            custom: self.custom.clone(),
            typed_checkpoint: self.typed_checkpoint.clone(),
        }
    }

    /// Takes a key; `now` stamps any message. A key that changes the form
    /// makes the plan out of date until the next refresh.
    pub fn key(&mut self, code: KeyCode, catalog: &mut Catalog, now: f64) -> Outcome {
        let before = self.snapshot();
        let outcome = self.take_key(code, catalog, now);
        if self.snapshot() != before {
            self.revision += 1;
            self.plan.command = None;
        }
        outcome
    }

    fn take_key(&mut self, code: KeyCode, catalog: &mut Catalog, now: f64) -> Outcome {
        if self.edit.is_some() {
            match code {
                KeyCode::Enter => self.commit(catalog, now),
                KeyCode::Esc => self.edit = None,
                KeyCode::Backspace => {
                    if let Some(edit) = &mut self.edit {
                        edit.text.pop();
                    }
                }
                KeyCode::Char(character) => {
                    if let Some(edit) = &mut self.edit {
                        edit.text.push(character);
                    }
                }
                _ => {}
            }
            return Outcome::Handled;
        }
        if self.finding {
            match code {
                KeyCode::Enter => self.finding = false,
                KeyCode::Esc => {
                    self.finding = false;
                    self.find.clear();
                }
                KeyCode::Backspace => {
                    self.find.pop();
                }
                KeyCode::Char(character) => {
                    self.find.push(character);
                    self.option_cursor = 1;
                }
                _ => {}
            }
            return Outcome::Handled;
        }
        match code {
            KeyCode::Tab => self.pane = self.pane.step(false),
            KeyCode::BackTab => self.pane = self.pane.step(true),
            KeyCode::Esc if !self.find.is_empty() => self.find.clear(),
            KeyCode::Esc => return Outcome::Back,
            KeyCode::Char('?') => return Outcome::Help,
            KeyCode::Char('S') => match self.start() {
                Ok(start) => return Outcome::Start(Box::new(start)),
                Err(why) => self.flash(now, format!("Not started: {why}"), false),
            },
            KeyCode::Char('/') => {
                self.pane = Pane::Options;
                self.finding = true;
                self.find.clear();
                self.option_cursor = 1;
            }
            KeyCode::Char('a') => self.start_adding(catalog),
            _ => {
                return match self.pane {
                    Pane::Models => self.models_key(code, catalog, now),
                    Pane::Profiles => self.profiles_key(code, catalog),
                    Pane::Placement => self.placement_key(code, now),
                    Pane::Options => self.options_key(code, catalog),
                };
            }
        }
        Outcome::Handled
    }

    fn models_key(&mut self, code: KeyCode, catalog: &Catalog, now: f64) -> Outcome {
        let models = catalog.models();
        let count = models.len() + self.recent.len();
        match code {
            KeyCode::Up | KeyCode::Char('k') => {
                self.model_cursor = self.model_cursor.saturating_sub(1)
            }
            KeyCode::Down | KeyCode::Char('j') => {
                self.model_cursor = (self.model_cursor + 1).min(count.saturating_sub(1));
            }
            KeyCode::Enter => {
                if let Some(recent) = self
                    .model_cursor
                    .checked_sub(models.len())
                    .and_then(|index| self.recent.get(index))
                    .cloned()
                {
                    let choice = recent.choice;
                    self.model = Some(choice.model);
                    self.profile = Some(choice.profile);
                    self.profile_chosen = true;
                    self.picked = choice.gpus;
                    self.edits = choice.edits;
                    self.custom = choice.custom;
                    self.flash(
                        now,
                        format!("restored {} from {}", recent.label, recent.when),
                        true,
                    );
                } else {
                    self.pane = Pane::Profiles;
                }
                return Outcome::Handled;
            }
            _ => return Outcome::Unhandled,
        }
        if let Some(model) = models.get(self.model_cursor) {
            self.choose_model(model.model.clone());
        }
        Outcome::Handled
    }

    fn profiles_key(&mut self, code: KeyCode, catalog: &Catalog) -> Outcome {
        if self.model(catalog).is_none() {
            return Outcome::Unhandled;
        }
        let profiles = self.profiles(catalog);
        let index = self
            .profile
            .as_ref()
            .and_then(|name| profiles.iter().position(|profile| profile.name == *name))
            .unwrap_or(0);
        let next = match code {
            KeyCode::Up | KeyCode::Char('k') => index.saturating_sub(1),
            KeyCode::Down | KeyCode::Char('j') => (index + 1).min(profiles.len().saturating_sub(1)),
            KeyCode::Enter => {
                self.pane = Pane::Options;
                return Outcome::Handled;
            }
            _ => return Outcome::Unhandled,
        };
        if let Some(profile) = profiles.get(next) {
            self.profile = Some(profile.name.clone());
            self.profile_chosen = true;
        }
        Outcome::Handled
    }

    fn placement_key(&mut self, code: KeyCode, now: f64) -> Outcome {
        let count = self.plan.gpus.len();
        match code {
            KeyCode::Up | KeyCode::Char('k') => self.gpu_cursor = self.gpu_cursor.saturating_sub(1),
            KeyCode::Down | KeyCode::Char('j') => {
                self.gpu_cursor = (self.gpu_cursor + 1).min(count.saturating_sub(1));
            }
            KeyCode::Char(' ') | KeyCode::Enter => {
                let Some(gpu) = self.plan.gpus.get(self.gpu_cursor) else {
                    return Outcome::Handled;
                };
                if self.picked.remove(&gpu.index) {
                    return Outcome::Handled;
                }
                if let Occupancy::Unknown(reason) = &gpu.occupancy {
                    self.flash(
                        now,
                        format!("GPU {} cannot be picked: {reason}", gpu.index),
                        false,
                    );
                } else {
                    self.picked.insert(gpu.index);
                }
            }
            _ => return Outcome::Unhandled,
        }
        Outcome::Handled
    }

    fn options_key(&mut self, code: KeyCode, catalog: &mut Catalog) -> Outcome {
        let rows = self.rows(catalog);
        let last = rows.len().saturating_sub(1);
        self.option_cursor = self.option_cursor.min(last);
        match code {
            KeyCode::Up | KeyCode::Char('k') => {
                self.option_cursor = self.option_cursor.saturating_sub(1);
                return Outcome::Handled;
            }
            KeyCode::Down | KeyCode::Char('j') => {
                self.option_cursor = (self.option_cursor + 1).min(last);
                return Outcome::Handled;
            }
            KeyCode::PageUp => {
                self.option_cursor = self.option_cursor.saturating_sub(10);
                return Outcome::Handled;
            }
            KeyCode::PageDown => {
                self.option_cursor = (self.option_cursor + 10).min(last);
                return Outcome::Handled;
            }
            _ => {}
        }
        let Some(row) = rows.get(self.option_cursor).cloned() else {
            return Outcome::Unhandled;
        };
        match (code, &row) {
            (KeyCode::Left | KeyCode::Char('h'), Row::Group { group, .. }) => self.close(*group),
            (KeyCode::Right | KeyCode::Char('l'), Row::Group { group, .. }) => self.reopen(*group),
            (KeyCode::Enter | KeyCode::Char(' '), Row::Group { group, open, .. }) => {
                if *open {
                    self.close(*group);
                } else {
                    self.reopen(*group);
                }
            }
            (KeyCode::Left | KeyCode::Char('h'), row) => {
                if let Some(key) = row.key() {
                    self.cycle(catalog, &key, false);
                }
            }
            (KeyCode::Right | KeyCode::Char('l'), row) => {
                if let Some(key) = row.key() {
                    self.cycle(catalog, &key, true);
                }
            }
            (KeyCode::Enter | KeyCode::Char(' '), Row::Option(_) | Row::Package(_)) => {
                let key = row.key().unwrap_or(Key::Flag(String::new()));
                let context = self.context(catalog);
                if !context.choices(&key).is_empty() {
                    self.cycle(catalog, &key, true);
                    return Outcome::Handled;
                }
                let secret = matches!(&key, Key::Variable(name) if name == command::API_KEY)
                    || catalog
                        .options()
                        .and_then(|options| match &key {
                            Key::Variable(name) => options.variable(name),
                            _ => None,
                        })
                        .is_some_and(|variable| variable.secret);
                let text = match (&key, context.effective(&key)) {
                    (Key::Flag(name), _) if name == "model" => String::new(),
                    _ if secret => self.edits.get(&key).cloned().unwrap_or_default(),
                    (_, (Some(value), _)) => value,
                    (_, (None, _)) => String::new(),
                };
                self.edit = Some(Edit {
                    target: Target::Value(key),
                    text,
                });
            }
            (KeyCode::Enter | KeyCode::Char(' '), Row::Custom(index)) => {
                self.edit = Some(Edit {
                    target: Target::Custom(*index),
                    text: self.custom.get(*index).cloned().unwrap_or_default(),
                });
            }
            (KeyCode::Enter | KeyCode::Char(' '), Row::Add) => self.start_adding(catalog),
            (KeyCode::Char('x'), Row::Custom(index)) => {
                if *index < self.custom.len() {
                    self.custom.remove(*index);
                }
            }
            (KeyCode::Char('x'), row) => {
                if let Some(key) = row.key() {
                    self.edits.remove(&key);
                    if key == Key::Flag("model".to_owned()) {
                        self.typed_checkpoint = None;
                    }
                }
            }
            (KeyCode::Enter | KeyCode::Char(' '), _) => {}
            _ => return Outcome::Unhandled,
        }
        Outcome::Handled
    }

    fn close(&mut self, group: Group) {
        if !self.closed.contains(&group) {
            self.closed.push(group);
        }
    }

    fn reopen(&mut self, group: Group) {
        self.closed.retain(|seen| *seen != group);
        // Opening the package's options asks again after a failure.
        if group == Group::Package && matches!(self.package, Some(Fetch::Failed(_))) {
            self.package_asked = None;
        }
    }

    fn start_adding(&mut self, catalog: &Catalog) {
        self.pane = Pane::Options;
        self.find.clear();
        self.finding = false;
        self.reopen(Group::Custom);
        self.edit = Some(Edit {
            target: Target::NewCustom,
            text: String::new(),
        });
        self.option_cursor = self
            .rows(catalog)
            .iter()
            .position(|row| *row == Row::NewCustom)
            .unwrap_or(0);
    }

    /// Keeps the inline edit.
    fn commit(&mut self, catalog: &mut Catalog, now: f64) {
        let Some(edit) = self.edit.take() else {
            return;
        };
        let text = edit.text.trim().to_owned();
        match edit.target {
            Target::Value(Key::Flag(name)) if name == "model" => {
                if !text.is_empty() {
                    let directory = PathBuf::from(&text);
                    // A failed answer is asked again.
                    if matches!(catalog.picks.get(&directory), Some(Fetch::Failed(_))) {
                        catalog.picks.remove(&directory);
                    }
                    catalog.pick(directory.clone());
                    self.typed_checkpoint = Some(directory);
                }
            }
            Target::Value(key) => {
                if text.is_empty() {
                    self.edits.remove(&key);
                } else {
                    self.set(catalog, key, text);
                }
            }
            Target::Custom(index) => self.keep_custom(catalog, Some(index), &text, now),
            Target::NewCustom => {
                self.keep_custom(catalog, None, &text, now);
                self.option_cursor = self
                    .rows(catalog)
                    .iter()
                    .position(|row| *row == Row::Add)
                    .unwrap_or(self.option_cursor);
            }
        }
    }

    /// Keeps custom row `index` (a new one with `None`) as `text`: a flag
    /// or variable the form holds moves onto its row, and a flag choptop
    /// owns is refused.
    fn keep_custom(&mut self, catalog: &Catalog, index: Option<usize>, text: &str, now: f64) {
        if text.is_empty() {
            if let Some(index) = index.filter(|index| *index < self.custom.len()) {
                self.custom.remove(index);
            }
            return;
        }
        let classified = match classify(text, catalog.options()) {
            Ok(classified) => classified,
            Err(reason) => {
                self.flash(now, format!("Not added: {}", sanitize(&reason)), false);
                return;
            }
        };
        let rest = classified.rest();
        match (index.filter(|index| *index < self.custom.len()), rest) {
            (Some(index), Some(rest)) => self.custom[index] = rest,
            (Some(index), None) => {
                self.custom.remove(index);
            }
            (None, Some(rest)) => self.custom.push(rest),
            (None, None) => {}
        }
        let Some((first, _)) = classified.folds.first().cloned() else {
            return;
        };
        let mut groups = Vec::new();
        for (key, value) in classified.folds {
            let group = self.group_of(catalog, &key);
            if !groups.contains(&group) {
                groups.push(group);
            }
            self.reopen(group);
            self.edits.insert(key, value);
        }
        let text = if groups.len() == 1 {
            format!(
                "{} is a {} option, so it is set there",
                first.shown(),
                groups[0].title().trim_end_matches(" options")
            )
        } else {
            format!("{} and more are set in their groups", first.shown())
        };
        self.flash(now, text, true);
        if let Some(position) = self
            .rows(catalog)
            .iter()
            .position(|row| row.key().as_ref() == Some(&first))
        {
            self.option_cursor = position;
        }
    }

    /// The group whose row holds `key`.
    fn group_of(&self, catalog: &Catalog, key: &Key) -> Group {
        let context = self.context(catalog);
        let rows = context.rows(&[], "", false);
        let mut group = Group::Custom;
        for row in rows {
            match row {
                Row::Group { group: seen, .. } => group = seen,
                row if row.key().as_ref() == Some(key) => return group,
                _ => {}
            }
        }
        match key {
            Key::ModelOption(_) => Group::Package,
            _ => Group::Custom,
        }
    }

    /// Scrolls the Options pane so its cursor is one of `shown` rows.
    pub fn settle(&mut self, catalog: &Catalog, shown: usize) {
        let count = self.rows(catalog).len();
        self.option_cursor = self.option_cursor.min(count.saturating_sub(1));
        self.option_scroll =
            crate::ui::list::first_row(self.option_scroll, self.option_cursor, shown)
                .min(count.saturating_sub(shown.max(1)));
        self.gpu_cursor = self.gpu_cursor.min(self.plan.gpus.len().saturating_sub(1));
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::catalog::{Checkpoints, Models, ServeOptions, parse_package_options};
    use crate::discover;
    use crate::gpu::poll::Report;
    use crate::gpu::{Kind, Owner, Process, Reading};
    use crate::launch::occupancy::tests::{devices, on_gpu, serve_process};

    const SERVE_OPTIONS: &str = include_str!("../../tests/fixtures/contracts/serve-options.json");
    const MODELS: &str = include_str!("../../tests/fixtures/contracts/models.json");
    const PICK: &str = include_str!("../../tests/fixtures/contracts/models-pick.json");
    const PACKAGE: &str = include_str!("../../tests/fixtures/contracts/list-model-options.json");

    /// A catalog with the contract fixtures read, and DeepSeek's
    /// checkpoint at /models/deepseek.
    pub(crate) fn catalog() -> Catalog {
        let environment = vec![(
            "CHOPINFER_DEEPSEEK_CHECKPOINT".to_owned(),
            PathBuf::from("/models/deepseek"),
        )];
        let mut catalog = Catalog::new(
            Ok("/opt/chopinfer/bin/chopinfer".into()),
            Checkpoints::load(None, environment),
        );
        catalog.open();
        catalog.apply(Reply::Options(ServeOptions::parse(SERVE_OPTIONS)));
        catalog.apply(Reply::Models(Models::parse(MODELS)));
        catalog.apply(Reply::Pick("/models/deepseek".into(), Pick::parse(PICK)));
        catalog.outbox.clear();
        catalog
    }

    /// Four GPUs read at second 1: 0 and 1 held by serve process 48211,
    /// 2 and 3 free.
    pub(crate) fn live() -> Live {
        let mut live = Live::default();
        live.gpu(&Report::Opened {
            driver: None,
            devices: devices(4),
        });
        live.scan(&discover::Report::Scan {
            found: vec![serve_process(48211, &["127.0.0.1:8310"], None)],
            events: Vec::new(),
            started: 0.0,
        });
        let listed = |processes: Vec<Process>| Reading {
            at: 1.0,
            processes: Some(processes),
            answered: true,
            ..Reading::default()
        };
        live.gpu(&Report::Polled(vec![
            listed(vec![on_gpu(48211)]),
            listed(vec![on_gpu(48211)]),
            listed(Vec::new()),
            listed(vec![Process {
                pid: 2210,
                kind: Kind::Graphics,
                owner: Owner {
                    name: Some("Xorg".into()),
                    ..Owner::default()
                },
                ..Process::default()
            }]),
        ]));
        live
    }

    fn refresh(loader: &mut Loader, catalog: &mut Catalog, live: &Live) {
        loader.refresh(catalog, live, &[], &[], 1.0, 1.5, &mut |_, _| true);
    }

    fn press(loader: &mut Loader, catalog: &mut Catalog, code: KeyCode) -> Outcome {
        loader.key(code, catalog, 1.5)
    }

    fn typed(loader: &mut Loader, catalog: &mut Catalog, text: &str) {
        for character in text.chars() {
            press(loader, catalog, KeyCode::Char(character));
        }
        press(loader, catalog, KeyCode::Enter);
    }

    #[test]
    fn opening_on_a_row_preselects_its_gpus_else_the_free_ones() {
        let (mut catalog, live) = (catalog(), live());
        let mut loader = Loader::open(Some(vec![0, 1]), Vec::new());
        refresh(&mut loader, &mut catalog, &live);
        assert_eq!(loader.picked, BTreeSet::from([0, 1]));
        // The first model with a checkpoint, at the pick for two GPUs:
        // DeepSeek's pick refuses TP2, so its TP4 profile stays.
        assert_eq!(loader.model.as_deref(), Some("deepseek41"));
        assert_eq!(loader.profile.as_deref(), Some("tp4-dspark"));
        let mut free = Loader::open(None, Vec::new());
        refresh(&mut free, &mut catalog, &live);
        // GPU 3 holds only a desktop's graphics process.
        assert_eq!(free.picked, BTreeSet::from([2, 3]));
        // A row whose GPUs are unknown preselects the free ones.
        let mut stale = Loader::open(Some(vec![0, 1]), Vec::new());
        stale.refresh(&mut catalog, &live, &[], &[], 1.0, 9.0, &mut |_, _| true);
        assert!(stale.picked.is_empty());
    }

    #[test]
    fn a_pick_the_profiles_file_does_not_list_is_offered_and_started() {
        let mut catalog = catalog();
        let mut live = Live::default();
        live.gpu(&Report::Opened {
            driver: None,
            devices: devices(8),
        });
        live.scan(&discover::Report::Scan {
            found: Vec::new(),
            events: Vec::new(),
            started: 0.0,
        });
        live.gpu(&Report::Polled(vec![
            Reading {
                at: 1.0,
                processes: Some(Vec::new()),
                answered: true,
                ..Reading::default()
            };
            8
        ]));
        let mut loader = Loader::open(None, Vec::new());
        refresh(&mut loader, &mut catalog, &live);
        assert_eq!(loader.picked.len(), 8);
        let names = loader
            .profiles(&catalog)
            .iter()
            .map(|profile| profile.name.as_str())
            .collect::<Vec<_>>();
        assert_eq!(names, ["tp4-dspark", "tp8-dspark-wide"]);
        // Eight GPUs take serve --model's TP8 pick, which only the pick
        // names, and the command line is asked for it.
        assert_eq!(loader.profile.as_deref(), Some("tp8-dspark-wide"));
        assert_eq!(loader.plan.tensor_parallel, Some(8));
        assert!(catalog.outbox.iter().any(|request| matches!(
            request,
            Request::Command { tensor_parallel: 8, profile, .. } if profile == "tp8-dspark-wide"
        )));
        // The Profiles pane moves to it and back like any other.
        loader.pane = Pane::Profiles;
        press(&mut loader, &mut catalog, KeyCode::Up);
        assert_eq!(loader.profile.as_deref(), Some("tp4-dspark"));
        press(&mut loader, &mut catalog, KeyCode::Down);
        assert_eq!(loader.profile.as_deref(), Some("tp8-dspark-wide"));
    }

    #[test]
    fn keys_move_through_the_panes_and_change_the_form() {
        let (mut catalog, live) = (catalog(), live());
        let mut loader = Loader::open(None, Vec::new());
        refresh(&mut loader, &mut catalog, &live);
        assert_eq!(loader.pane, Pane::Models);
        press(&mut loader, &mut catalog, KeyCode::Tab);
        assert_eq!(loader.pane, Pane::Profiles);
        press(&mut loader, &mut catalog, KeyCode::BackTab);
        press(&mut loader, &mut catalog, KeyCode::BackTab);
        assert_eq!(loader.pane, Pane::Options);
        press(&mut loader, &mut catalog, KeyCode::Tab);
        assert_eq!(loader.pane, Pane::Models);
        // Moving in Models chooses a model; changing it clears edits.
        loader
            .edits
            .insert(Key::Flag("max-sequences".into()), "48".into());
        press(&mut loader, &mut catalog, KeyCode::Char('k'));
        assert_eq!(loader.model.as_deref(), Some("qwen38"));
        assert!(loader.edits.is_empty());
        press(&mut loader, &mut catalog, KeyCode::Down);
        assert_eq!(loader.model.as_deref(), Some("deepseek41"));
        // Enter on a model goes on to its profiles.
        press(&mut loader, &mut catalog, KeyCode::Enter);
        assert_eq!(loader.pane, Pane::Profiles);
        // Changing the profile keeps edits.
        press(&mut loader, &mut catalog, KeyCode::Tab);
        press(&mut loader, &mut catalog, KeyCode::Tab);
        assert_eq!(loader.pane, Pane::Options);
        refresh(&mut loader, &mut catalog, &live);
        let rows = loader.rows(&catalog);
        let sequences = rows
            .iter()
            .position(|row| *row == Row::Option(Key::Flag("max-sequences".into())))
            .unwrap();
        loader.option_cursor = sequences;
        press(&mut loader, &mut catalog, KeyCode::Enter);
        assert_eq!(
            loader.edit,
            Some(Edit {
                target: Target::Value(Key::Flag("max-sequences".into())),
                text: String::new(),
            })
        );
        typed(&mut loader, &mut catalog, "48");
        assert_eq!(loader.edits[&Key::Flag("max-sequences".into())], "48");
        // Esc cancels an edit and leaves the value.
        press(&mut loader, &mut catalog, KeyCode::Enter);
        press(&mut loader, &mut catalog, KeyCode::Backspace);
        press(&mut loader, &mut catalog, KeyCode::Esc);
        assert_eq!(loader.edits[&Key::Flag("max-sequences".into())], "48");
        // ←→ change a choice; back to the profile's value drops the edit.
        let cache = rows
            .iter()
            .position(|row| *row == Row::Option(Key::Flag("prefix-cache".into())))
            .unwrap();
        loader.option_cursor = cache;
        press(&mut loader, &mut catalog, KeyCode::Right);
        assert_eq!(loader.edits[&Key::Flag("prefix-cache".into())], "on");
        press(&mut loader, &mut catalog, KeyCode::Left);
        press(&mut loader, &mut catalog, KeyCode::Left);
        assert_eq!(loader.edits[&Key::Flag("prefix-cache".into())], "on");
        // x resets a row.
        press(&mut loader, &mut catalog, KeyCode::Char('x'));
        assert!(!loader.edits.contains_key(&Key::Flag("prefix-cache".into())));
        // ← closes a group on its title row, → opens it.
        loader.option_cursor = 0;
        press(&mut loader, &mut catalog, KeyCode::Left);
        assert!(loader.closed.contains(&Group::Model));
        press(&mut loader, &mut catalog, KeyCode::Right);
        assert!(!loader.closed.contains(&Group::Model));
        press(&mut loader, &mut catalog, KeyCode::Enter);
        assert!(loader.closed.contains(&Group::Model));
        // / finds; Esc clears the filter; a second Esc leaves.
        press(&mut loader, &mut catalog, KeyCode::Char('/'));
        assert!(loader.finding);
        typed(&mut loader, &mut catalog, "cache");
        assert!(!loader.finding);
        assert_eq!(loader.find, "cache");
        let found = loader.rows(&catalog);
        assert!(found.contains(&Row::Option(Key::Flag("prefix-cache".into()))));
        assert!(!found.contains(&Row::Option(Key::Flag("max-sequences".into()))));
        assert_eq!(
            press(&mut loader, &mut catalog, KeyCode::Esc),
            Outcome::Handled
        );
        assert!(loader.find.is_empty());
        assert_eq!(
            press(&mut loader, &mut catalog, KeyCode::Char('?')),
            Outcome::Help
        );
        assert_eq!(
            press(&mut loader, &mut catalog, KeyCode::Esc),
            Outcome::Back
        );
        assert_eq!(
            press(&mut loader, &mut catalog, KeyCode::Char('q')),
            Outcome::Unhandled
        );
    }

    #[test]
    fn space_picks_gpus_but_not_unknown_ones() {
        let (mut catalog, live) = (catalog(), live());
        let mut loader = Loader::open(Some(vec![0, 1]), Vec::new());
        refresh(&mut loader, &mut catalog, &live);
        loader.pane = Pane::Placement;
        press(&mut loader, &mut catalog, KeyCode::Down);
        press(&mut loader, &mut catalog, KeyCode::Down);
        press(&mut loader, &mut catalog, KeyCode::Char(' '));
        press(&mut loader, &mut catalog, KeyCode::Char('j'));
        press(&mut loader, &mut catalog, KeyCode::Enter);
        assert_eq!(loader.picked, BTreeSet::from([0, 1, 2, 3]));
        press(&mut loader, &mut catalog, KeyCode::Char(' '));
        assert_eq!(loader.picked, BTreeSet::from([0, 1, 2]));
        // Readings went stale: the GPU cannot be picked.
        loader.refresh(&mut catalog, &live, &[], &[], 1.0, 9.0, &mut |_, _| true);
        press(&mut loader, &mut catalog, KeyCode::Char(' '));
        assert_eq!(loader.picked, BTreeSet::from([0, 1, 2]));
        let flash = loader.flash.clone().unwrap();
        assert!(!flash.good);
        assert!(
            flash.text.starts_with("GPU 3 cannot be picked"),
            "{}",
            flash.text
        );
    }

    #[test]
    fn a_custom_row_folds_onto_its_option_or_is_refused() {
        let (mut catalog, live) = (catalog(), live());
        let mut loader = Loader::open(Some(vec![0, 1, 2, 3]), Vec::new());
        refresh(&mut loader, &mut catalog, &live);
        assert_eq!(
            press(&mut loader, &mut catalog, KeyCode::Char('a')),
            Outcome::Handled
        );
        assert_eq!(loader.pane, Pane::Options);
        assert_eq!(loader.rows(&catalog)[loader.option_cursor], Row::NewCustom);
        typed(&mut loader, &mut catalog, "--gpu-memory-utilization 0.9");
        assert_eq!(
            loader.edits[&Key::Flag("gpu-memory-utilization".into())],
            "0.9"
        );
        assert!(loader.custom.is_empty());
        let flash = loader.flash.clone().unwrap();
        assert!(flash.good);
        assert_eq!(
            flash.text,
            "--gpu-memory-utilization is a Memory and KV option, so it is set there"
        );
        press(&mut loader, &mut catalog, KeyCode::Char('a'));
        typed(&mut loader, &mut catalog, "--devices 0,1");
        assert!(loader.custom.is_empty());
        assert!(
            loader
                .flash
                .clone()
                .unwrap()
                .text
                .starts_with("Not added: --devices")
        );
        press(&mut loader, &mut catalog, KeyCode::Char('a'));
        typed(&mut loader, &mut catalog, "CUDA_VISIBLE_DEVICES=0");
        assert!(loader.custom.is_empty());
        press(&mut loader, &mut catalog, KeyCode::Char('a'));
        typed(&mut loader, &mut catalog, "--kv-offload-prefix off");
        assert_eq!(loader.custom, ["--kv-offload-prefix off"]);
        // x on a custom row removes it.
        let rows = loader.rows(&catalog);
        loader.option_cursor = rows.iter().position(|row| *row == Row::Custom(0)).unwrap();
        press(&mut loader, &mut catalog, KeyCode::Char('x'));
        assert!(loader.custom.is_empty());
    }

    /// Answers the requests in the outbox as chopinfer would: the
    /// package's options, and a command line of the options asked for.
    fn answer(loader: &mut Loader, catalog: &mut Catalog) {
        for request in std::mem::take(&mut catalog.outbox) {
            match request {
                Request::PackageOptions { generation, .. } => loader.reply(Reply::PackageOptions(
                    generation,
                    parse_package_options(PACKAGE),
                )),
                Request::Command {
                    generation,
                    checkpoint,
                    tensor_parallel,
                    options,
                    ..
                } => {
                    let mut arguments = vec![
                        "serve".to_owned(),
                        "--model".to_owned(),
                        checkpoint.display().to_string(),
                        "--tensor-parallel-size".to_owned(),
                        tensor_parallel.to_string(),
                    ];
                    arguments.extend(options);
                    loader.reply(Reply::Command(generation, Ok(arguments)));
                }
                _ => {}
            }
        }
    }

    #[test]
    fn a_start_right_after_a_change_waits_for_that_changes_plan() {
        let (mut catalog, live) = (catalog(), live());
        let mut loader = Loader::open(Some(vec![0, 1, 2, 3]), Vec::new());
        refresh(&mut loader, &mut catalog, &live);
        answer(&mut loader, &mut catalog);
        refresh(&mut loader, &mut catalog, &live);
        assert!(
            loader.plan.problems.is_empty() && loader.plan.command.is_some(),
            "{:?}",
            loader.plan
        );
        // The TUI takes a change of profile and S before it refreshes: S
        // does not start the old profile's command under the new name.
        loader.pane = Pane::Profiles;
        press(&mut loader, &mut catalog, KeyCode::Down);
        assert_eq!(loader.profile.as_deref(), Some("tp8-dspark-wide"));
        assert_eq!(
            press(&mut loader, &mut catalog, KeyCode::Char('S')),
            Outcome::Handled
        );
        assert_eq!(
            loader.flash.clone().unwrap().text,
            "Not started: still computing the command"
        );
        assert_eq!(loader.plan.command, None);
        // Back to the pick, and an edit, then S, still before a refresh.
        press(&mut loader, &mut catalog, KeyCode::Up);
        press(&mut loader, &mut catalog, KeyCode::Char('/'));
        typed(&mut loader, &mut catalog, "max-seq");
        press(&mut loader, &mut catalog, KeyCode::Enter);
        for _ in 0..8 {
            press(&mut loader, &mut catalog, KeyCode::Backspace);
        }
        typed(&mut loader, &mut catalog, "16");
        assert_eq!(
            loader.edits.get(&Key::Flag("max-sequences".into())),
            Some(&"16".to_owned())
        );
        assert!(matches!(
            press(&mut loader, &mut catalog, KeyCode::Char('S')),
            Outcome::Handled
        ));
        // Once the plan and the command line are this form's, S starts it.
        refresh(&mut loader, &mut catalog, &live);
        answer(&mut loader, &mut catalog);
        refresh(&mut loader, &mut catalog, &live);
        let Outcome::Start(start) = press(&mut loader, &mut catalog, KeyCode::Char('S')) else {
            panic!("a start: {:?}", loader.plan)
        };
        assert_eq!(start.profile, "tp4-dspark");
        assert!(
            start
                .command
                .windows(2)
                .any(|pair| pair == ["--max-sequences", "16"]),
            "{:?}",
            start.command
        );
    }

    #[test]
    fn a_checkpoint_typed_in_starts_nothing_until_its_command_is_read() {
        let (mut catalog, live) = (catalog(), live());
        let mut loader = Loader::open(Some(vec![0, 1, 2, 3]), Vec::new());
        refresh(&mut loader, &mut catalog, &live);
        answer(&mut loader, &mut catalog);
        refresh(&mut loader, &mut catalog, &live);
        assert_eq!(loader.plan.checkpoint, Some("/models/deepseek".into()));
        assert!(loader.start().is_ok(), "{:?}", loader.plan);
        // Another DeepSeek checkpoint typed in, models pick not answered.
        loader.pane = Pane::Options;
        loader.option_cursor = loader
            .rows(&catalog)
            .iter()
            .position(|row| *row == Row::Option(Key::Flag("model".into())))
            .unwrap();
        press(&mut loader, &mut catalog, KeyCode::Enter);
        typed(&mut loader, &mut catalog, "/models/new-deepseek");
        let refused = |loader: &mut Loader, catalog: &mut Catalog| {
            let outcome = press(loader, catalog, KeyCode::Char('S'));
            assert_eq!(outcome, Outcome::Handled);
            loader.flash.clone().unwrap().text
        };
        // S at once...
        assert_eq!(
            refused(&mut loader, &mut catalog),
            "Not started: still checking the checkpoint"
        );
        // ...and after a refresh, which still plans the checkpoint before.
        refresh(&mut loader, &mut catalog, &live);
        assert_eq!(loader.plan.checkpoint, Some("/models/deepseek".into()));
        assert!(
            loader
                .plan
                .waits
                .contains(&"models pick for the checkpoint typed in".to_owned()),
            "{:?}",
            loader.plan.waits
        );
        assert_eq!(
            refused(&mut loader, &mut catalog),
            "Not started: still checking the checkpoint"
        );
        // Once models pick names DeepSeek, the command line is asked for
        // the new checkpoint and S waits for that answer.
        catalog.apply(Reply::Pick(
            "/models/new-deepseek".into(),
            Pick::parse(PICK),
        ));
        refresh(&mut loader, &mut catalog, &live);
        assert_eq!(loader.typed_checkpoint, None);
        assert_eq!(loader.plan.checkpoint, Some("/models/new-deepseek".into()));
        assert!(
            refused(&mut loader, &mut catalog).contains("the command line"),
            "{:?}",
            loader.flash
        );
        answer(&mut loader, &mut catalog);
        refresh(&mut loader, &mut catalog, &live);
        let Outcome::Start(start) = press(&mut loader, &mut catalog, KeyCode::Char('S')) else {
            panic!("a start: {:?}", loader.plan)
        };
        assert_eq!(
            start.command[..3],
            ["serve", "--model", "/models/new-deepseek"]
        );
    }

    /// The contract catalog where, at TP4, serve --model picks a profile
    /// the file does not list, and tp4-dspark runs as that pick with
    /// `--max-sequences 16`.
    fn renamed_pick() -> Catalog {
        use crate::catalog::PickProfile;
        let mut catalog = catalog();
        let mut pick = Pick::parse(PICK).unwrap();
        pick.sizes[0].pick = Some("tp4-other".into());
        pick.sizes[0].profiles = vec![
            PickProfile {
                name: "tp4-other".into(),
                availability: Availability::Available {
                    arguments: Vec::new(),
                    inherited_model_options: Vec::new(),
                },
            },
            PickProfile {
                name: "tp4-dspark".into(),
                availability: Availability::Available {
                    arguments: vec!["--max-sequences".into(), "16".into()],
                    inherited_model_options: Vec::new(),
                },
            },
        ];
        catalog.apply(Reply::Pick("/models/deepseek".into(), Ok(pick)));
        catalog
    }

    /// A package options request's profile, options and variables.
    type Asked = (String, String, Vec<(String, String)>);

    /// The package options request in the outbox, if any.
    fn package_asked(catalog: &mut Catalog) -> Option<Asked> {
        std::mem::take(&mut catalog.outbox)
            .into_iter()
            .find_map(|request| match request {
                Request::PackageOptions {
                    profile,
                    options,
                    environment,
                    ..
                } => Some((profile, options.join(" "), environment)),
                _ => None,
            })
    }

    #[test]
    fn the_package_options_follow_the_profile_the_form_and_the_gpus() {
        use crate::launch::occupancy::tests::uuid;
        let (mut catalog, live) = (renamed_pick(), live());
        let mut loader = Loader::open(Some(vec![0, 1, 2, 3]), Vec::new());
        refresh(&mut loader, &mut catalog, &live);
        assert_eq!(loader.profile.as_deref(), Some("tp4-other"));
        let (profile, options, environment) = package_asked(&mut catalog).unwrap();
        assert_eq!((profile.as_str(), options.as_str()), ("tp4-other", ""));
        let uuids = (0..4).map(uuid).collect::<Vec<_>>().join(",");
        assert_eq!(environment, [("CUDA_VISIBLE_DEVICES".to_owned(), uuids)]);
        loader.reply(Reply::PackageOptions(
            loader.package_ticket.unwrap(),
            parse_package_options(PACKAGE),
        ));
        refresh(&mut loader, &mut catalog, &live);
        assert_eq!(package_asked(&mut catalog), None);
        // Another profile asks again, and its defaults are not read from
        // the other package.
        loader.pane = Pane::Profiles;
        press(&mut loader, &mut catalog, KeyCode::Up);
        refresh(&mut loader, &mut catalog, &live);
        assert_eq!(loader.profile.as_deref(), Some("tp4-dspark"));
        let (profile, options, _) = package_asked(&mut catalog).unwrap();
        assert_eq!((profile.as_str(), options.as_str()), ("tp4-dspark", ""));
        assert_eq!(loader.package, Some(Fetch::Waiting));
        // So does a preparation option set in the form, and a variable.
        loader
            .edits
            .insert(Key::Flag("modalities".into()), "text".into());
        loader.edits.insert(
            Key::Variable("CHOPINFER_LOG_SUMMARY_SECONDS".into()),
            "30".into(),
        );
        refresh(&mut loader, &mut catalog, &live);
        let (_, options, environment) = package_asked(&mut catalog).unwrap();
        assert_eq!(options, "--modalities text");
        assert_eq!(
            environment[1],
            ("CHOPINFER_LOG_SUMMARY_SECONDS".to_owned(), "30".to_owned())
        );
        // A model option's value is not part of it.
        loader.edits.insert(
            Key::ModelOption("prefill_upper_skip".into()),
            "false".into(),
        );
        refresh(&mut loader, &mut catalog, &live);
        assert_eq!(package_asked(&mut catalog), None);
        // Other GPUs are: before an instance is placed none are named.
        loader.picked = BTreeSet::from([0, 1]);
        refresh(&mut loader, &mut catalog, &live);
        let (_, _, environment) = package_asked(&mut catalog).unwrap();
        assert_eq!(
            environment,
            [("CHOPINFER_LOG_SUMMARY_SECONDS".to_owned(), "30".to_owned())]
        );
    }

    #[test]
    fn a_flag_the_profile_and_the_form_both_set_is_asked_once() {
        let (mut catalog, live) = (renamed_pick(), live());
        let mut loader = Loader::open(Some(vec![0, 1, 2, 3]), Vec::new());
        loader.profile = Some("tp4-dspark".into());
        loader.profile_chosen = true;
        loader
            .edits
            .insert(Key::Flag("max-sequences".into()), "48".into());
        refresh(&mut loader, &mut catalog, &live);
        // tp4-dspark runs as the pick with --max-sequences 16; the form's
        // 48 goes to models command, which merges it over the profile, and
        // the profile's own flag is not added beside it.
        let (profile, options, _) = package_asked(&mut catalog).unwrap();
        assert_eq!(profile, "tp4-dspark");
        assert_eq!(options, "--max-sequences 48");
    }

    #[test]
    fn s_checks_the_form_and_hands_over_the_switch() {
        let (mut catalog, live) = (catalog(), live());
        let mut loader = Loader::open(Some(vec![0, 1, 2, 3]), Vec::new());
        refresh(&mut loader, &mut catalog, &live);
        // The package's options are asked for; the command waits for them.
        let package = catalog.outbox.iter().find_map(|request| match request {
            Request::PackageOptions {
                generation,
                checkpoint,
                tensor_parallel,
                profile,
                options,
                ..
            } => Some((
                *generation,
                (checkpoint.clone(), *tensor_parallel, profile.clone()),
                options.clone(),
            )),
            _ => None,
        });
        let (generation, asked, options) = package.unwrap();
        assert_eq!(
            asked,
            (
                PathBuf::from("/models/deepseek"),
                4,
                "tp4-dspark".to_owned()
            )
        );
        assert!(options.is_empty());
        // The pick inherits nothing, so the command line is asked at once.
        let command = catalog.outbox.iter().find_map(|request| match request {
            Request::Command {
                generation,
                profile,
                ..
            } => Some((*generation, profile.clone())),
            _ => None,
        });
        let (command, profile) = command.unwrap();
        assert_eq!(profile, "tp4-dspark");
        assert!(matches!(
            press(&mut loader, &mut catalog, KeyCode::Char('S')),
            Outcome::Handled
        ));
        assert!(
            loader
                .flash
                .clone()
                .unwrap()
                .text
                .contains("waiting for the package's options")
        );
        loader.reply(Reply::PackageOptions(
            generation,
            parse_package_options(PACKAGE),
        ));
        catalog.outbox.clear();
        refresh(&mut loader, &mut catalog, &live);
        // Nothing changed, so nothing is asked again.
        assert!(catalog.outbox.is_empty(), "{:?}", catalog.outbox);
        let generation = command;
        // A stale answer is dropped.
        loader.reply(Reply::Command(generation - 1, Ok(vec!["serve".into()])));
        refresh(&mut loader, &mut catalog, &live);
        assert!(loader.plan.command.is_none());
        let arguments = vec![
            "serve".to_owned(),
            "--model".to_owned(),
            "/models/deepseek".to_owned(),
        ];
        loader.reply(Reply::Command(generation, Ok(arguments.clone())));
        refresh(&mut loader, &mut catalog, &live);
        assert!(
            loader.plan.problems.is_empty(),
            "{:?}",
            loader.plan.problems
        );
        let Outcome::Start(start) = press(&mut loader, &mut catalog, KeyCode::Char('S')) else {
            panic!("a start: {:?}", loader.plan)
        };
        assert_eq!(start.command, arguments);
        assert_eq!(start.placement.stops.len(), 1);
        assert_eq!(start.placement.instances[0].gpus, [0, 1, 2, 3]);
        // A bad value blocks it, with the reason.
        loader
            .edits
            .insert(Key::Flag("max-sequences".into()), "many".into());
        refresh(&mut loader, &mut catalog, &live);
        press(&mut loader, &mut catalog, KeyCode::Char('S'));
        let flash = loader.flash.clone().unwrap();
        assert_eq!(flash.text, "Not started: --max-sequences: whole number");
    }

    #[test]
    fn a_typed_checkpoint_is_kept_once_models_pick_names_its_model() {
        let (mut catalog, live) = (catalog(), live());
        let mut loader = Loader::open(None, Vec::new());
        refresh(&mut loader, &mut catalog, &live);
        loader.pane = Pane::Models;
        press(&mut loader, &mut catalog, KeyCode::Up);
        assert_eq!(loader.model.as_deref(), Some("qwen38"));
        loader.pane = Pane::Options;
        loader.option_cursor = loader
            .rows(&catalog)
            .iter()
            .position(|row| *row == Row::Option(Key::Flag("model".into())))
            .unwrap();
        press(&mut loader, &mut catalog, KeyCode::Enter);
        catalog.outbox.clear();
        typed(&mut loader, &mut catalog, "/models/qwen");
        assert_eq!(catalog.outbox, [Request::Pick("/models/qwen".into())]);
        refresh(&mut loader, &mut catalog, &live);
        assert_eq!(loader.typed_checkpoint, Some(PathBuf::from("/models/qwen")));
        let mut pick = Pick::parse(PICK).unwrap();
        pick.model = Some("qwen38".into());
        catalog.apply(Reply::Pick("/models/qwen".into(), Ok(pick)));
        refresh(&mut loader, &mut catalog, &live);
        assert_eq!(loader.typed_checkpoint, None);
        assert_eq!(
            catalog.checkpoints.remembered("qwen38"),
            Some(std::path::Path::new("/models/qwen"))
        );
        assert!(loader.flash.clone().unwrap().good);
        // A directory no helper takes is not kept.
        press(&mut loader, &mut catalog, KeyCode::Enter);
        typed(&mut loader, &mut catalog, "/models/none");
        catalog.apply(Reply::Pick(
            "/models/none".into(),
            Err("no helper takes it".into()),
        ));
        refresh(&mut loader, &mut catalog, &live);
        assert_eq!(
            catalog.checkpoints.remembered("qwen38"),
            Some(std::path::Path::new("/models/qwen"))
        );
        assert!(loader.flash.clone().unwrap().text.starts_with("Not kept"));
    }

    #[test]
    fn a_recent_launch_restores_its_choices() {
        let (mut catalog, live) = (catalog(), live());
        let choice = Choice {
            model: "qwen38".into(),
            profile: "tp2-ar".into(),
            gpus: BTreeSet::from([2, 3]),
            edits: BTreeMap::from([(Key::Flag("max-sequences".into()), "48".into())]),
            custom: vec!["--kv-offload-prefix off".into()],
        };
        let recent = Recent {
            label: "Qwen tp2-ar".into(),
            when: "3d ago".into(),
            choice: choice.clone(),
        };
        let mut loader = Loader::open(None, vec![recent]);
        refresh(&mut loader, &mut catalog, &live);
        press(&mut loader, &mut catalog, KeyCode::Down);
        press(&mut loader, &mut catalog, KeyCode::Down);
        press(&mut loader, &mut catalog, KeyCode::Enter);
        refresh(&mut loader, &mut catalog, &live);
        assert_eq!(loader.model.as_deref(), Some("qwen38"));
        assert_eq!(loader.profile.as_deref(), Some("tp2-ar"));
        assert_eq!(loader.picked, choice.gpus);
        assert_eq!(loader.edits, choice.edits);
        assert_eq!(loader.custom, choice.custom);
        assert_eq!(
            loader.flash.clone().unwrap().text,
            "restored Qwen tp2-ar from 3d ago"
        );
    }

    #[test]
    fn a_second_start_on_pending_launches_replaces_them() {
        use crate::launch::occupancy::tests::uuid;
        use crate::launch::placement::Dependency;
        let mut catalog = catalog();
        let mut live = live();
        // Launch A waits for 48211; launch B started and is loading, with
        // no GPU process and no listener yet.
        live.scan(&discover::Report::Scan {
            found: vec![
                serve_process(48211, &["127.0.0.1:8310"], None),
                serve_process(61204, &[], None),
            ],
            events: Vec::new(),
            started: 0.0,
        });
        let reservations = vec![
            Reservation {
                label: "launch A".into(),
                model: "DeepSeek-V4.1-Flash".into(),
                gpus: vec![uuid(0), uuid(1)],
                port: 8312,
                process: None,
                launch: 1,
            },
            Reservation {
                label: "proc 61204".into(),
                model: "DeepSeek-V4.1-Flash".into(),
                gpus: vec![uuid(2), uuid(3)],
                port: 8313,
                process: Some((61204, 612_040)),
                launch: 2,
            },
        ];
        let mut loader = Loader::open(Some(vec![0, 1, 2, 3]), Vec::new());
        loader.refresh(
            &mut catalog,
            &live,
            &[],
            &reservations,
            1.0,
            1.5,
            &mut |_, _| true,
        );
        let placement = &loader.plan.placement;
        // Both launches are replaced: A is cancelled, B stopped.
        assert_eq!(placement.replaces, [0, 1]);
        assert!(reservations[0].waiting() && !reservations[1].waiting());
        assert_eq!(
            placement.instances[0].after,
            [
                Dependency::Serve(placement.stops[0].clone()),
                Dependency::Launch(0),
                Dependency::Launch(1),
            ]
        );
        // The pending launches' ports stay theirs; 48211's is reused.
        assert_eq!(placement.instances[0].port, 8310);
    }

    #[test]
    fn another_model_drops_the_command_line_and_package_options_asked_before() {
        let (mut catalog, live) = (catalog(), live());
        let mut loader = Loader::open(Some(vec![0, 1, 2, 3]), Vec::new());
        refresh(&mut loader, &mut catalog, &live);
        for request in std::mem::take(&mut catalog.outbox) {
            match request {
                Request::PackageOptions { generation, .. } => {
                    loader.reply(Reply::PackageOptions(
                        generation,
                        parse_package_options(PACKAGE),
                    ));
                }
                Request::Command { generation, .. } => {
                    loader.reply(Reply::Command(generation, Ok(vec!["serve".into()])));
                }
                _ => {}
            }
        }
        refresh(&mut loader, &mut catalog, &live);
        assert!(loader.plan.command.is_some());
        assert!(
            loader
                .rows(&catalog)
                .contains(&Row::Package("prefill_upper_skip".into()))
        );
        // Qwen has no checkpoint: nothing of DeepSeek's stays.
        loader.pane = Pane::Models;
        press(&mut loader, &mut catalog, KeyCode::Up);
        refresh(&mut loader, &mut catalog, &live);
        assert_eq!(loader.model.as_deref(), Some("qwen38"));
        assert_eq!(loader.plan.command, None);
        assert_eq!(loader.package, None);
        assert!(
            !loader
                .rows(&catalog)
                .contains(&Row::Package("prefill_upper_skip".into()))
        );
        // An answer still on its way for DeepSeek is dropped.
        loader.reply(Reply::PackageOptions(1, parse_package_options(PACKAGE)));
        assert_eq!(loader.package, None);
    }

    #[test]
    fn sizes_follow_the_gpus_picked() {
        assert_eq!(allowed(&[2, 4], 4), Some(4));
        assert_eq!(allowed(&[2, 4], 6), Some(2));
        assert_eq!(allowed(&[2, 4], 3), Some(2));
        assert_eq!(allowed(&[4], 2), Some(4));
        assert_eq!(allowed(&[], 2), None);
    }
}
