//! The interface state: which view, which instance or GPU, which window,
//! the model loader, the launches and their confirm dialogs, and what each
//! key does.

use std::time::{SystemTime, UNIX_EPOCH};

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers, MouseButton, MouseEvent, MouseEventKind};

use crate::catalog::{Catalog, Check, Reply, Request};
use crate::gpu::Gpus;
use crate::host::Host;
use crate::instances::{Instance, Launched, Registry, Source, Tone};
use crate::launch::occupancy::Live;
use crate::launch::placement::bind_probe;
use crate::launch::process::{self, Identity};
use crate::launch::switch::{self, Confirm, Target};
use crate::launch::{Launches, Reservation, Wait, command, indices};
use crate::loader::{Loader, Outcome, Start};
use crate::stats::Window;
use crate::ui::list::{TABLE_TOP, first_row, table_rows};
use crate::ui::monitor;
use crate::ui::widgets::sanitize;

/// Seconds between two clicks on one row that open it.
const DOUBLE_CLICK_SECONDS: f64 = 0.5;
/// Seconds a start or stop waits for a fresh GPU reading and scan.
pub const READ_SECONDS: f64 = 20.0;

/// Seconds since the epoch.
fn wall() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |since| since.as_secs())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum View {
    List,
    Detail,
    /// This host: CPU, memory, disks, sensors and GPUs.
    Monitor,
    /// Starting or switching a model.
    Loader,
}

/// What the main loop should do after an input.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Action {
    Nothing,
    Redraw,
    Rescan,
    /// A start or stop waits for a fresh GPU reading and scan: the loop
    /// asks for both and calls [`App::asked`].
    Read,
    Quit,
}

/// What a fresh reading is taken for.
#[derive(Clone, Debug, PartialEq)]
pub enum Wanted {
    Start(Box<Start>),
    /// Stopping the row with this id.
    Stop(u64),
}

/// A start or stop waiting for its fresh reading, and a start then for
/// `serve --check`.
#[derive(Clone, Debug, PartialEq)]
pub struct Pending {
    pub wanted: Wanted,
    /// When the reading was asked for; `None` until the loop asks.
    pub since: Option<f64>,
    /// The ticket of the `serve --check` request, once sent.
    pub ticket: Option<u64>,
}

/// What `X` stops.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Stoppable {
    /// A serve process of this user, by pid and start time.
    Process { pid: u32, start_ticks: u64 },
    /// A choptop launch, by its id.
    Launch(u64),
}

/// The stop dialog.
#[derive(Clone, Debug, PartialEq)]
pub struct StopConfirm {
    pub label: String,
    /// `Qwen3.8-Flash at 127.0.0.1:8310 on GPUs 0,1`.
    pub what: String,
    /// The GPUs it frees, by NVML index.
    pub gpus: Vec<u32>,
    pub target: Stoppable,
    /// A launch that has not started, which Enter cancels.
    pub waiting: bool,
}

/// A confirm dialog over the view.
#[derive(Clone, Debug, PartialEq)]
pub enum Dialog {
    Start(Box<Confirm>),
    Stop(Box<StopConfirm>),
}

#[derive(Debug)]
pub struct App {
    pub registry: Registry,
    pub view: View,
    /// The selected row's id, so the selection survives rows coming and
    /// going.
    pub selected: Option<u64>,
    pub window: Window,
    pub paused: bool,
    pub help: bool,
    pub interval: f64,
    pub rescan: f64,
    /// The sources in use, for the header: `proc`, `docker`, `--target`.
    pub sources: Vec<&'static str>,
    /// The first list row shown, kept so the list scrolls only when the
    /// selection leaves it.
    pub scroll: usize,
    /// This host's CPU, memory, disks and sensors.
    pub host: Host,
    /// This host's GPUs.
    pub gpus: Gpus,
    /// The selected GPU's position.
    pub gpu: usize,
    /// The first GPU row shown.
    pub gpu_scroll: usize,
    /// The latest GPU readings and scan, which the loader reads even while
    /// the screen is paused.
    pub live: Live,
    /// choptop's launches that are queued or starting.
    pub reservations: Vec<Reservation>,
    /// What `chopinfer` said it can serve.
    pub catalog: Catalog,
    pub loader: Loader,
    /// choptop's launches and the serve processes it is stopping.
    pub launches: Launches,
    /// A start or stop waiting for its fresh reading or `serve --check`.
    pub pending: Option<Pending>,
    pub dialog: Option<Dialog>,
    /// `--read-only`: no loader and no stopping.
    pub read_only: bool,
    /// Where launch logs go, as shown.
    pub logs: String,
    /// Whether a port is free to listen on; tests replace the bind probe.
    pub probe: fn(&str, u16) -> bool,
    /// Seconds since the epoch, which name a switch and its logs; tests
    /// replace it.
    pub wall: fn() -> u64,
    /// The clock at the last [`App::tick`], which stamps the loader's
    /// messages.
    clock: f64,
    /// The views Esc returns through, the latest last.
    trail: Vec<View>,
    /// The terminal height at the last [`App::settle`], which clicks are
    /// mapped with.
    height: u16,
    /// The monitor's GPU table at the last [`App::settle`], as
    /// `(x, y, width, height)`.
    gpu_table: Option<(u16, u16, u16, u16)>,
    /// The row clicked last, and when: an instance id in the list, a GPU
    /// position in the monitor.
    last_click: Option<(View, u64, f64)>,
}

impl App {
    pub fn new(registry: Registry, interval: f64, rescan: f64, sources: Vec<&'static str>) -> Self {
        Self {
            registry,
            view: View::List,
            selected: None,
            window: Window::Minute,
            paused: false,
            help: false,
            interval,
            rescan,
            sources,
            scroll: 0,
            host: Host::new(interval),
            gpus: Gpus::off(),
            gpu: 0,
            gpu_scroll: 0,
            live: Live::default(),
            reservations: Vec::new(),
            catalog: Catalog::none(),
            loader: Loader::default(),
            launches: Launches::default(),
            pending: None,
            dialog: None,
            read_only: false,
            logs: "~/.local/state/choptop/logs".to_owned(),
            probe: bind_probe,
            wall,
            clock: 0.0,
            trail: Vec::new(),
            height: 0,
            gpu_table: None,
            last_click: None,
        }
    }

    /// Shows `view`, which Esc leaves for the view shown now. A view already
    /// on the way back is returned to instead, so the trail stays short.
    fn open(&mut self, view: View) {
        match self.trail.iter().position(|seen| *seen == view) {
            Some(position) => self.trail.truncate(position),
            None => self.trail.push(self.view),
        }
        self.view = view;
    }

    /// Goes back to the view before this one, the list at the start.
    fn back(&mut self) {
        self.view = self.trail.pop().unwrap_or(View::List);
    }

    /// Opens the monitor, or leaves it. From an instance's details, its
    /// first GPU is selected.
    fn toggle_monitor(&mut self) {
        if self.view == View::Monitor {
            self.back();
            return;
        }
        if self.view == View::Detail
            && let Some(row) = self.current()
            && let Some(first) = self
                .gpus
                .of_instance(&self.registry.instances, row.id)
                .first()
            && let Some(position) = self
                .gpus
                .devices
                .iter()
                .position(|device| device.index == *first)
        {
            self.gpu = position;
        }
        self.open(View::Monitor);
    }

    /// The row on the selected GPU, if any.
    fn gpu_instance(&self) -> Option<u64> {
        self.gpus
            .processes(self.gpu, &self.registry.instances)
            .into_iter()
            .find_map(|(_, row)| row)
    }

    /// Opens the model loader on the selected row, or in the monitor the
    /// selected GPU's: its GPUs are preselected when they are known. On an
    /// exited launch, the loader takes that launch's choices.
    fn open_loader(&mut self) {
        let row = match self.view {
            View::Monitor => self.gpu_instance(),
            _ => self.current().map(|row| row.id),
        };
        let gpus = row
            .map(|id| self.gpus.of_instance(&self.registry.instances, id))
            .filter(|gpus| !gpus.is_empty());
        let choice = row
            .and_then(|id| self.registry.get(id))
            .and_then(|row| match row.launch {
                Some((id, Launched::Exited(_))) => self.launches.get(id),
                _ => None,
            })
            .and_then(|launch| self.launches.records.choice(&launch.record.switch));
        let recent = self.launches.records.recent((self.wall)());
        self.loader = Loader::open(gpus, recent);
        if let Some(choice) = choice {
            if let Some(position) = self
                .catalog
                .models()
                .iter()
                .position(|model| model.model == choice.model)
            {
                self.loader.model_cursor = position;
            }
            self.loader.restore(choice);
        }
        self.catalog.open();
        self.open(View::Loader);
    }

    /// Notes that `--read-only` refuses `key`.
    fn read_only_note(&mut self, key: char) {
        self.registry.note(
            self.clock,
            "choptop",
            format!("{key}: choptop runs --read-only, so it starts and stops nothing"),
            Tone::Warning,
        );
    }

    /// `X` on the selected row: an exited launch leaves the list; a serve
    /// process of this user or a launch is stopped after a fresh reading
    /// and a confirm; anything else is refused with the reason.
    fn ask_stop(&mut self) -> Action {
        let row = match self.view {
            View::Monitor => self.gpu_instance().and_then(|id| self.registry.get(id)),
            _ => self.current(),
        };
        let Some(row) = row else {
            return Action::Nothing;
        };
        let (id, label) = (row.id, row.label());
        let refusal = match (&row.launch, &row.source) {
            (Some((launch, Launched::Exited(_))), _) => {
                let launch = *launch;
                self.launches.dismiss(launch, &mut self.registry);
                return Action::Redraw;
            }
            _ if row.stopping.is_some() => Some("it is stopping already".to_owned()),
            (Some(_), _) | (None, Source::Process { .. }) => None,
            (None, Source::Target(_)) => Some(
                "choptop stops only serve processes of this user, and a --target may be anywhere"
                    .to_owned(),
            ),
            (None, Source::Container { .. }) => {
                Some("choptop does not stop containers; stop it with Docker".to_owned())
            }
            (None, Source::Launch { .. }) => Some("the launch is gone".to_owned()),
        };
        if let Some(refusal) = refusal {
            self.registry.note(
                self.clock,
                label,
                format!("not stopped: {refusal}"),
                Tone::Warning,
            );
            return Action::Redraw;
        }
        self.pending = Some(Pending {
            wanted: Wanted::Stop(id),
            since: None,
            ticket: None,
        });
        Action::Read
    }

    /// The loop asked for a fresh GPU reading and scan at `now`.
    pub fn asked(&mut self, now: f64) {
        if let Some(pending) = &mut self.pending {
            pending.since.get_or_insert(now);
        }
    }

    /// What the pending start is waiting for, as the command box says it.
    pub fn pending_text(&self) -> Option<&'static str> {
        match self.pending.as_ref()? {
            Pending {
                wanted: Wanted::Start(_),
                ticket: None,
                ..
            } => Some("Checking: reading the GPUs and processes again"),
            Pending {
                wanted: Wanted::Start(_),
                ..
            } => Some("Checking: running serve --check"),
            _ => None,
        }
    }

    /// Says why the start does not go ahead.
    fn refuse_start(&mut self, why: &str) {
        self.pending = None;
        self.loader
            .flash(self.clock, format!("Not started: {}", sanitize(why)), false);
    }

    /// One turn of the loop at `now`: the launches move on, and a pending
    /// start or stop takes its fresh reading once it is in. Says whether
    /// anything changed.
    pub fn supervise(&mut self, now: f64) -> bool {
        self.clock = now;
        let mut probe = self.probe;
        let mut changed = self.launches.supervise(&mut self.registry, now, &mut probe);
        self.reservations = self.launches.reservations();
        let Some(Pending {
            since: Some(since),
            ticket: None,
            ..
        }) = &self.pending
        else {
            return changed;
        };
        let since = *since;
        if !self.live.fresh(since) {
            if now - since > READ_SECONDS {
                let why =
                    format!("no fresh GPU reading and discovery scan within {READ_SECONDS:.0} s");
                match self.pending.take().map(|pending| pending.wanted) {
                    Some(Wanted::Start(_)) => self.refuse_start(&why),
                    Some(Wanted::Stop(row)) => self.refuse_stop(row, &why),
                    None => {}
                }
                changed = true;
            }
            return changed;
        }
        match self.pending.take().map(|pending| pending.wanted) {
            Some(Wanted::Start(start)) => self.recheck(start, since, now),
            Some(Wanted::Stop(row)) => self.open_stop(row),
            None => {}
        }
        true
    }

    /// The fresh reading is in: the start goes on to `serve --check` if the
    /// GPUs and ports it uses are as they were when `S` was pressed.
    fn recheck(&mut self, start: Box<Start>, since: f64, now: f64) {
        let mut probe = self.probe;
        self.loader.refresh(
            &mut self.catalog,
            &self.live,
            &self.registry.instances,
            &self.reservations,
            self.interval,
            now,
            &mut probe,
        );
        let fresh = match self.loader.start() {
            Ok(fresh) => fresh,
            Err(why) => return self.refuse_start(&why),
        };
        let changes = switch::changed(&start.placement, &fresh.placement, &self.reservations);
        if !changes.is_empty() {
            return self.refuse_start(&format!(
                "{}; check the placement and press S again",
                changes.join("; ")
            ));
        }
        let Some(first) = fresh.placement.instances.first() else {
            return self.refuse_start("nothing to start");
        };
        let ticket = self.catalog.ticket();
        self.catalog.send(Request::Check {
            generation: ticket,
            arguments: command::arguments(&fresh.command, &first.listen()),
            environment: command::environment(&fresh.explicit, &first.uuids),
        });
        self.pending = Some(Pending {
            wanted: Wanted::Start(Box::new(fresh)),
            since: Some(since),
            ticket: Some(ticket),
        });
    }

    /// `serve --check` answered for the pending start: a refusal or a
    /// setting other than the profile and the form say blocks it, else the
    /// confirm dialog opens.
    fn checked(&mut self, result: Result<Check, String>) {
        let Some(Pending {
            wanted: Wanted::Start(start),
            ..
        }) = self.pending.take()
        else {
            return;
        };
        let check = match result {
            Ok(check) => check,
            Err(message) => return self.refuse_start(&message),
        };
        let Some(profile) = self.catalog.profile(&start.model, &start.profile) else {
            return self.refuse_start(&format!("profile {} is gone", start.profile));
        };
        let differences = switch::differences(&check, profile, &start);
        if !differences.is_empty() {
            return self.refuse_start(&format!(
                "serve --check resolves what the profile and form do not say: {}",
                differences.join("; ")
            ));
        }
        let launches = &self.launches;
        let stopping = |pid, ticks| launches.stopping(pid, ticks);
        let confirm = switch::confirm(
            start,
            &switch::Reading {
                catalog: &self.catalog,
                check: &check,
                gpus: &self.loader.plan.gpus,
                reservations: &self.reservations,
                stopping: &stopping,
                warnings: self.loader.warnings(&self.catalog),
                memory: self
                    .host
                    .latest()
                    .and_then(|reading| reading.memory)
                    .map(|memory| (memory.available, memory.total)),
                wall: (self.wall)(),
            },
        );
        match confirm {
            Ok(confirm) => self.dialog = Some(Dialog::Start(Box::new(confirm))),
            Err(why) => self.refuse_start(&why),
        }
    }

    /// Notes why row `row` is not stopped.
    fn refuse_stop(&mut self, row: u64, why: &str) {
        let label = self
            .registry
            .get(row)
            .map_or_else(|| "choptop".to_owned(), Instance::label);
        self.registry.note(
            self.clock,
            label,
            format!("not stopped: {why}"),
            Tone::Warning,
        );
    }

    /// The fresh reading is in for stopping row `row`: the stop dialog
    /// opens if the row is still there and still running.
    fn open_stop(&mut self, row: u64) {
        let Some(instance) = self.registry.get(row) else {
            return self.refuse_stop(row, "it is gone");
        };
        let (target, gpus, waiting) = match (&instance.launch, &instance.source) {
            (Some((id, state)), _) => {
                let gpus = self
                    .launches
                    .get(*id)
                    .map(|launch| launch.record.gpus.clone())
                    .unwrap_or_default();
                if matches!(state, Launched::Exited(_)) {
                    return self.refuse_stop(row, "it has exited");
                }
                (Stoppable::Launch(*id), gpus, *state == Launched::Waiting)
            }
            (None, Source::Process { pid, start_ticks }) => {
                if !process::alive(*pid, *start_ticks) {
                    return self.refuse_stop(row, "it has exited");
                }
                let gpus = self.gpus.of_instance(&self.registry.instances, row);
                (
                    Stoppable::Process {
                        pid: *pid,
                        start_ticks: *start_ticks,
                    },
                    gpus,
                    false,
                )
            }
            _ => return self.refuse_stop(row, "choptop stops only its own serve processes"),
        };
        let name = crate::ui::shown_name(instance).unwrap_or_else(|| "-".to_owned());
        let at = instance
            .address()
            .map(|address| format!(" at {}", sanitize(&address)))
            .unwrap_or_default();
        let on = if gpus.is_empty() {
            String::new()
        } else {
            format!(" on GPUs {}", indices(&gpus))
        };
        self.dialog = Some(Dialog::Stop(Box::new(StopConfirm {
            label: instance.label(),
            what: format!("{name}{at}{on}"),
            gpus,
            target,
            waiting,
        })));
    }

    /// Enter in a confirm dialog.
    fn confirm(&mut self, dialog: Dialog) {
        match dialog {
            Dialog::Start(confirm) => self.launch(*confirm),
            Dialog::Stop(stop) => self.stop(*stop),
        }
    }

    /// Runs a confirmed switch: records each serve process it stops, and
    /// refuses if one changed since the reading; cancels or stops the
    /// launches it replaces; queues its instances; then sends the serve
    /// processes SIGTERM. The list shows the launches.
    fn launch(&mut self, mut confirm: Confirm) {
        let now = self.clock;
        confirm.switch = self.launches.records.unused_switch(&confirm.switch);
        let mut identities = Vec::new();
        for stopping in &confirm.stops {
            if let Target::Serve(serve) = &stopping.target {
                match Identity::record(serve.pid, serve.start_ticks) {
                    Ok(identity) => identities.push((identity, stopping.gpus.clone())),
                    Err(why) => return self.refuse_start(&why),
                }
            }
        }
        let recorded = identities
            .iter()
            .map(|(identity, _)| identity.clone())
            .collect::<Vec<_>>();
        let queued = match switch::queued(
            &confirm,
            &self.catalog,
            &recorded,
            self.launches.logs.as_deref(),
        ) {
            Ok(queued) => queued,
            Err(why) => return self.refuse_start(&why),
        };
        let why = switch::reason(&confirm);
        for stopping in &confirm.stops {
            if let Target::Launch(id) = stopping.target
                && let Err(error) = self.launches.stop(id, &why, &mut self.registry, now)
            {
                self.registry.note(
                    now,
                    stopping.label.clone(),
                    format!("not stopped: {error}"),
                    Tone::Warning,
                );
            }
        }
        let mut first = None;
        for mut item in queued {
            let launches = &self.launches;
            item.waits.retain(|wait| match wait {
                Wait::Launch(id) => launches.get(*id).is_some(),
                Wait::Serve(_) => true,
            });
            let id = self.launches.queue(item, &mut self.registry, now);
            first.get_or_insert(id);
        }
        for (identity, gpus) in &identities {
            // A process that changed is not signalled; its launches are
            // cancelled and the reason noted.
            let _ = self
                .launches
                .stop_serve(identity, gpus, &why, &mut self.registry, now);
        }
        self.reservations = self.launches.reservations();
        self.trail.clear();
        self.view = View::List;
        if let Some(first) = first.and_then(|id| self.registry.launch_row(id)) {
            self.selected = Some(first.id);
        }
    }

    /// Runs a confirmed stop: a launch through its process group, or a
    /// serve process through a pidfd once it is still the process read.
    fn stop(&mut self, stop: StopConfirm) {
        let now = self.clock;
        let why = "as asked with X";
        let result = match stop.target {
            Stoppable::Launch(id) => self.launches.stop(id, why, &mut self.registry, now),
            Stoppable::Process { pid, start_ticks } => {
                Identity::record(pid, start_ticks).and_then(|identity| {
                    self.launches
                        .stop_serve(&identity, &stop.gpus, why, &mut self.registry, now)
                })
            }
        };
        if let Err(error) = result {
            self.registry.note(
                now,
                stop.label,
                format!("not stopped: {error}"),
                Tone::Warning,
            );
        }
    }

    /// Works out the loader's plan from the latest readings at `now`;
    /// called before each frame.
    pub fn tick(&mut self, now: f64) {
        self.clock = now;
        if self.view == View::Loader {
            let mut probe = self.probe;
            self.loader.refresh(
                &mut self.catalog,
                &self.live,
                &self.registry.instances,
                &self.reservations,
                self.interval,
                now,
                &mut probe,
            );
        }
    }

    /// Takes a reply from `chopinfer`.
    pub fn catalog_reply(&mut self, reply: Reply) {
        let Some(reply) = self.catalog.apply(reply) else {
            return;
        };
        match reply {
            Reply::Check(ticket, result)
                if self
                    .pending
                    .as_ref()
                    .is_some_and(|pending| pending.ticket == Some(ticket)) =>
            {
                self.checked(result);
            }
            reply => self.loader.reply(reply),
        }
    }

    /// Opens the details of the instance on the selected GPU, if any.
    fn open_gpu_instance(&mut self) -> bool {
        let row = self.gpu_instance();
        match row {
            Some(row) => {
                self.selected = Some(row);
                self.open(View::Detail);
                true
            }
            None => false,
        }
    }

    fn step_gpu(&mut self, forward: bool) {
        let count = self.gpus.devices.len();
        self.gpu = if forward {
            (self.gpu + 1).min(count.saturating_sub(1))
        } else {
            self.gpu.saturating_sub(1)
        };
    }

    /// Scrolls the list and the GPU table so the selected rows are shown
    /// on a terminal `width` by `height`. Called before each frame.
    pub fn settle(&mut self, width: u16, height: u16) {
        self.height = height;
        let room = table_rows(self.rows().len(), height);
        self.scroll = first_row(self.scroll, self.selected_index(), room);
        let count = self.gpus.devices.len();
        self.gpu = self.gpu.min(count.saturating_sub(1));
        self.gpu_table = monitor::layout(width, height, self).table;
        let shown = self
            .gpu_table
            .map_or(1, |(_, _, _, height)| usize::from(height.saturating_sub(3)));
        self.gpu_scroll = first_row(self.gpu_scroll, self.gpu, shown);
        if self.view == View::Loader {
            let shown = crate::ui::loader::layout(width, height, self)
                .options
                .map_or(1, |(_, _, _, height)| {
                    crate::ui::loader::option_rows(height)
                });
            self.loader.settle(&self.catalog, shown);
        }
    }

    /// The rows shown, in order.
    pub fn rows(&self) -> Vec<&Instance> {
        self.registry.visible()
    }

    /// The selected row's position, the first row when it is gone.
    pub fn selected_index(&self) -> usize {
        let rows = self.rows();
        self.selected
            .and_then(|id| rows.iter().position(|row| row.id == id))
            .unwrap_or(0)
    }

    pub fn current(&self) -> Option<&Instance> {
        self.rows().get(self.selected_index()).copied()
    }

    fn select(&mut self, index: usize) {
        self.selected = self.rows().get(index).map(|row| row.id);
    }

    fn step(&mut self, forward: bool, wrap: bool) {
        let count = self.rows().len();
        if count == 0 {
            return;
        }
        let index = self.selected_index();
        let next = match (forward, wrap) {
            (true, true) => (index + 1) % count,
            (false, true) => (index + count - 1) % count,
            (true, false) => (index + 1).min(count - 1),
            (false, false) => index.saturating_sub(1),
        };
        self.select(next);
    }

    pub fn key(&mut self, event: KeyEvent) -> Action {
        let control = event.modifiers.contains(KeyModifiers::CONTROL);
        if control && matches!(event.code, KeyCode::Char('c')) {
            return Action::Quit;
        }
        if self.help {
            self.help = false;
            return Action::Redraw;
        }
        if let Some(dialog) = self.dialog.take() {
            match event.code {
                KeyCode::Enter => self.confirm(dialog),
                KeyCode::Esc => {}
                _ => {
                    self.dialog = Some(dialog);
                    return Action::Nothing;
                }
            }
            return Action::Redraw;
        }
        if self.pending.is_some() && event.code == KeyCode::Esc {
            if matches!(
                self.pending.take().map(|pending| pending.wanted),
                Some(Wanted::Start(_))
            ) {
                self.loader.flash(self.clock, "Start cancelled", false);
            }
            return Action::Redraw;
        }
        if self.view == View::Loader {
            // The form holds still while its start is checked.
            if self.pending.is_some() {
                return Action::Nothing;
            }
            match self.loader.key(event.code, &mut self.catalog, self.clock) {
                Outcome::Handled => return Action::Redraw,
                Outcome::Back => {
                    self.back();
                    return Action::Redraw;
                }
                Outcome::Help => {
                    self.help = true;
                    return Action::Redraw;
                }
                Outcome::Start(start) => {
                    self.pending = Some(Pending {
                        wanted: Wanted::Start(start),
                        since: None,
                        ticket: None,
                    });
                    return Action::Read;
                }
                Outcome::Unhandled => {}
            }
        }
        match (self.view, event.code) {
            (_, KeyCode::Char('q')) => return Action::Quit,
            (_, KeyCode::Char('?')) => self.help = true,
            (_, KeyCode::Char('p')) => self.paused = !self.paused,
            (_, KeyCode::Char('w')) => self.window = self.window.next(),
            (_, KeyCode::Char('r')) => return Action::Rescan,
            // g opened the GPU view this replaces.
            (_, KeyCode::Char('m' | 'g')) => self.toggle_monitor(),
            (View::List | View::Detail | View::Monitor, KeyCode::Char('l')) if self.read_only => {
                self.read_only_note('l');
            }
            (View::List | View::Detail | View::Monitor, KeyCode::Char('l')) => self.open_loader(),
            (View::List | View::Detail | View::Monitor, KeyCode::Char('X')) if self.read_only => {
                self.read_only_note('X');
            }
            (View::List | View::Detail | View::Monitor, KeyCode::Char('X')) => {
                return self.ask_stop();
            }
            (View::List, KeyCode::Down | KeyCode::Char('j')) => self.step(true, false),
            (View::List, KeyCode::Up | KeyCode::Char('k')) => self.step(false, false),
            (View::List, KeyCode::Enter) => {
                if self.current().is_some() {
                    self.select(self.selected_index());
                    self.open(View::Detail);
                }
            }
            (
                View::Detail | View::Monitor,
                KeyCode::Esc | KeyCode::Backspace | KeyCode::Char('h'),
            ) => self.back(),
            (View::Monitor, KeyCode::Down | KeyCode::Char('j')) => self.step_gpu(true),
            (View::Monitor, KeyCode::Up | KeyCode::Char('k')) => self.step_gpu(false),
            (View::Monitor, KeyCode::Enter) => {
                if !self.open_gpu_instance() {
                    return Action::Nothing;
                }
            }
            (View::Detail, KeyCode::Right | KeyCode::Tab) => {
                self.step(true, true);
            }
            (View::Detail, KeyCode::Left | KeyCode::BackTab) => self.step(false, true),
            _ => return Action::Nothing,
        }
        Action::Redraw
    }

    /// A click selects a row of the list or a GPU; a second click on it
    /// within half a second opens it, or the GPU's instance. A GPU row is
    /// clicked only between the table's borders, not beside it.
    pub fn mouse(&mut self, event: MouseEvent, now: f64) -> Action {
        if matches!(self.view, View::Detail | View::Loader)
            || self.help
            || self.dialog.is_some()
            || event.kind != MouseEventKind::Down(MouseButton::Left)
        {
            return Action::Nothing;
        }
        let (top, count, selected, scroll, room) = match self.view {
            View::Monitor => {
                let Some((x, y, width, height)) = self.gpu_table else {
                    return Action::Nothing;
                };
                if event.column <= x || event.column + 1 >= x + width {
                    return Action::Nothing;
                }
                let room = usize::from(height.saturating_sub(3));
                (y, self.gpus.devices.len(), self.gpu, self.gpu_scroll, room)
            }
            _ => {
                let count = self.rows().len();
                let room = table_rows(count, self.height);
                (TABLE_TOP, count, self.selected_index(), self.scroll, room)
            }
        };
        let Some(line) = event.row.checked_sub(top + 2).map(usize::from) else {
            return Action::Nothing;
        };
        let first = first_row(scroll, selected, room);
        if line >= room || first + line >= count {
            return Action::Nothing;
        }
        let clicked = match self.view {
            View::Monitor => {
                self.gpu = first + line;
                self.gpu as u64
            }
            _ => {
                let id = self.rows()[first + line].id;
                self.selected = Some(id);
                id
            }
        };
        let view = self.view;
        let double = self.last_click.is_some_and(|(seen, id, at)| {
            seen == view && id == clicked && now - at <= DOUBLE_CLICK_SECONDS
        });
        if double {
            self.last_click = None;
            match view {
                View::Monitor => {
                    self.open_gpu_instance();
                }
                _ => self.open(View::Detail),
            }
        } else {
            self.last_click = Some((view, clicked, now));
        }
        Action::Redraw
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::catalog::{Checkpoints, parse_package_options};
    use crate::discover::Found;
    use crate::discover::proc::ServeProcess;
    use crate::gpu::poll::Report;
    use crate::gpu::{Kind, Owner, Process, Reading};
    use crate::instances::State;
    use crate::launch::command::{Explicit, Key};
    use crate::launch::occupancy::Serve;
    use crate::launch::occupancy::tests::{on_gpu, serve_process, uuid};
    use crate::launch::placement::{Placement, Planned};
    use crate::launch::process::tests::{
        CLOSE, Guard, fake_serve, fake_serve_running, scratch, wait_for, write_script,
    };
    use crate::launch::switch::Stopping;
    use crate::launch::{Phase, records::Records};
    use crate::loader::Choice;
    use std::path::Path;

    fn app(rows: usize) -> App {
        let mut registry = Registry::new(1.0);
        for port in 0..rows {
            registry.add_target(
                crate::targets::Target::parse(&format!(":{}", 8310 + port)).unwrap(),
                0.0,
            );
        }
        App::new(registry, 1.0, 15.0, vec!["--target"])
    }

    fn press(app: &mut App, code: KeyCode) -> Action {
        app.key(KeyEvent::new(code, KeyModifiers::NONE))
    }

    #[test]
    fn keys_select_open_switch_and_quit() {
        let mut app = app(3);
        assert_eq!(app.selected_index(), 0);
        press(&mut app, KeyCode::Down);
        press(&mut app, KeyCode::Char('j'));
        press(&mut app, KeyCode::Down);
        assert_eq!(app.selected_index(), 2);
        press(&mut app, KeyCode::Char('k'));
        assert_eq!(app.selected_index(), 1);
        assert_eq!(press(&mut app, KeyCode::Enter), Action::Redraw);
        assert_eq!(app.view, View::Detail);
        press(&mut app, KeyCode::Tab);
        press(&mut app, KeyCode::Right);
        assert_eq!(app.selected_index(), 0);
        press(&mut app, KeyCode::BackTab);
        assert_eq!(app.selected_index(), 2);
        press(&mut app, KeyCode::Esc);
        assert_eq!(app.view, View::List);
        press(&mut app, KeyCode::Char('w'));
        assert_eq!(app.window, Window::FiveMinutes);
        press(&mut app, KeyCode::Char('p'));
        assert!(app.paused);
        assert_eq!(press(&mut app, KeyCode::Char('r')), Action::Rescan);
        press(&mut app, KeyCode::Char('?'));
        assert!(app.help);
        // Any key closes help, even q.
        assert_eq!(press(&mut app, KeyCode::Char('q')), Action::Redraw);
        assert!(!app.help);
        assert_eq!(press(&mut app, KeyCode::Char('q')), Action::Quit);
        assert_eq!(
            app.key(KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL)),
            Action::Quit
        );
        assert_eq!(press(&mut app, KeyCode::Char('x')), Action::Nothing);
    }

    #[test]
    fn clicks_select_and_double_clicks_open() {
        let mut app = app(3);
        app.settle(80, 24);
        let click = |row| MouseEvent {
            kind: MouseEventKind::Down(MouseButton::Left),
            column: 10,
            row,
            modifiers: KeyModifiers::NONE,
        };
        assert_eq!(app.mouse(click(TABLE_TOP + 3), 1.0), Action::Redraw);
        assert_eq!((app.selected_index(), app.view), (1, View::List));
        assert_eq!(app.mouse(click(TABLE_TOP + 3), 1.3), Action::Redraw);
        assert_eq!(app.view, View::Detail);
        app.view = View::List;
        assert_eq!(app.mouse(click(TABLE_TOP + 9), 2.0), Action::Nothing);
        assert_eq!(app.mouse(click(0), 2.0), Action::Nothing);
    }

    #[test]
    fn a_long_list_scrolls_with_the_selection_and_clicks_follow() {
        let mut app = app(30);
        app.settle(80, 24);
        for _ in 0..20 {
            press(&mut app, KeyCode::Down);
            app.settle(80, 24);
        }
        // 13 rows fit; row 20 is the last one shown.
        assert_eq!((app.selected_index(), app.scroll), (20, 8));
        let click = MouseEvent {
            kind: MouseEventKind::Down(MouseButton::Left),
            column: 10,
            row: TABLE_TOP + 2,
            modifiers: KeyModifiers::NONE,
        };
        app.mouse(click, 1.0);
        app.settle(80, 24);
        assert_eq!((app.selected_index(), app.scroll), (8, 8));
        // A click below the table, on the summary, selects nothing.
        let below = MouseEvent {
            row: TABLE_TOP + 2 + 13,
            ..click
        };
        assert_eq!(app.mouse(below, 5.0), Action::Nothing);
    }

    fn with_gpus(app: &mut App, count: u32) {
        use crate::gpu::{Device, Status, poll::Report};
        let mut gpus = Gpus::new(Status::Opening, true, 1.0);
        gpus.apply(
            Report::Opened {
                driver: None,
                devices: (0..count)
                    .map(|index| Device {
                        index,
                        ..Device::default()
                    })
                    .collect(),
            },
            &mut app.registry,
            0.0,
        );
        app.gpus = gpus;
    }

    fn click(column: u16, row: u16) -> MouseEvent {
        MouseEvent {
            kind: MouseEventKind::Down(MouseButton::Left),
            column,
            row,
            modifiers: KeyModifiers::NONE,
        }
    }

    #[test]
    fn the_monitor_opens_selects_and_returns() {
        let mut app = app(2);
        with_gpus(&mut app, 3);
        app.settle(80, 24);
        press(&mut app, KeyCode::Char('m'));
        assert_eq!(app.view, View::Monitor);
        press(&mut app, KeyCode::Down);
        press(&mut app, KeyCode::Char('j'));
        press(&mut app, KeyCode::Char('j'));
        assert_eq!(app.gpu, 2);
        press(&mut app, KeyCode::Char('k'));
        assert_eq!(app.gpu, 1);
        // No process on the GPU is an instance, so Enter does nothing.
        assert_eq!(press(&mut app, KeyCode::Enter), Action::Nothing);
        // Before the host's first reading its panel is one line, so the
        // table starts on row 5 and its GPUs on row 7.
        app.settle(80, 24);
        assert_eq!(app.gpu_table, Some((0, 5, 80, 6)));
        assert_eq!(app.mouse(click(10, 7), 1.0), Action::Redraw);
        assert_eq!(app.gpu, 0);
        assert_eq!(app.mouse(click(10, 10), 1.1), Action::Nothing);
        assert_eq!(app.view, View::Monitor);
        press(&mut app, KeyCode::Char('m'));
        assert_eq!(app.view, View::List);
        // g, which opened the GPU view, opens the monitor too.
        press(&mut app, KeyCode::Char('g'));
        assert_eq!(app.view, View::Monitor);
        press(&mut app, KeyCode::Esc);
        assert_eq!(app.view, View::List);
        // Through the details and back.
        press(&mut app, KeyCode::Enter);
        press(&mut app, KeyCode::Char('m'));
        press(&mut app, KeyCode::Esc);
        assert_eq!(app.view, View::Detail);
        press(&mut app, KeyCode::Esc);
        assert_eq!(app.view, View::List);
    }

    #[test]
    fn a_click_beside_the_gpu_table_selects_nothing() {
        let mut app = app(1);
        with_gpus(&mut app, 3);
        press(&mut app, KeyCode::Char('m'));
        app.settle(200, 50);
        // The Sensors panel runs down the right from column 152.
        assert_eq!(app.gpu_table, Some((0, 5, 152, 6)));
        assert_eq!(app.mouse(click(160, 8), 1.0), Action::Nothing);
        assert_eq!(app.mouse(click(151, 8), 2.0), Action::Nothing);
        assert_eq!(app.mouse(click(0, 8), 3.0), Action::Nothing);
        assert_eq!(app.gpu, 0);
        assert_eq!(app.mouse(click(150, 8), 4.0), Action::Redraw);
        assert_eq!(app.gpu, 1);
    }

    #[test]
    fn l_opens_the_loader_from_every_view_and_esc_returns() {
        let mut app = app(3);
        with_gpus(&mut app, 2);
        assert_eq!(press(&mut app, KeyCode::Char('l')), Action::Redraw);
        assert_eq!(app.view, View::Loader);
        press(&mut app, KeyCode::Esc);
        assert_eq!(app.view, View::List);
        // In the details, l opens the loader; Tab and → still step.
        press(&mut app, KeyCode::Enter);
        press(&mut app, KeyCode::Tab);
        assert_eq!(app.selected_index(), 1);
        press(&mut app, KeyCode::Char('l'));
        assert_eq!((app.view, app.selected_index()), (View::Loader, 1));
        press(&mut app, KeyCode::Esc);
        assert_eq!(app.view, View::Detail);
        press(&mut app, KeyCode::Char('m'));
        press(&mut app, KeyCode::Char('l'));
        assert_eq!(app.view, View::Loader);
        press(&mut app, KeyCode::Esc);
        assert_eq!(app.view, View::Monitor);
        press(&mut app, KeyCode::Esc);
        press(&mut app, KeyCode::Esc);
        assert_eq!(app.view, View::List);
    }

    #[test]
    fn the_loader_takes_its_keys_and_leaves_the_rest() {
        use crate::loader::Pane;
        let mut app = app(1);
        press(&mut app, KeyCode::Char('l'));
        app.tick(1.0);
        // Tab moves between panes instead of instances.
        press(&mut app, KeyCode::Tab);
        assert_eq!(app.loader.pane, Pane::Profiles);
        press(&mut app, KeyCode::BackTab);
        assert_eq!(app.loader.pane, Pane::Models);
        // ? opens help, which any key closes.
        press(&mut app, KeyCode::Char('?'));
        assert!(app.help);
        press(&mut app, KeyCode::Char('j'));
        assert!(!app.help);
        // / finds; while finding, q is text.
        press(&mut app, KeyCode::Char('/'));
        assert_eq!(press(&mut app, KeyCode::Char('q')), Action::Redraw);
        assert_eq!(app.loader.find, "q");
        press(&mut app, KeyCode::Esc);
        // Without chopinfer, S says why it cannot start.
        assert_eq!(press(&mut app, KeyCode::Char('S')), Action::Redraw);
        let flash = app.loader.flash.clone().unwrap();
        assert!(
            flash.text.starts_with("Not started: chopinfer"),
            "{}",
            flash.text
        );
        assert!(app.pending.is_none());
        // Keys the loader leaves alone keep their meaning.
        press(&mut app, KeyCode::Char('p'));
        assert!(app.paused);
        let click = MouseEvent {
            kind: MouseEventKind::Down(MouseButton::Left),
            column: 10,
            row: 5,
            modifiers: KeyModifiers::NONE,
        };
        assert_eq!(app.mouse(click, 1.0), Action::Nothing);
        assert_eq!(press(&mut app, KeyCode::Char('q')), Action::Quit);
    }

    #[test]
    fn a_reopened_loader_takes_no_answer_meant_for_the_one_before() {
        use crate::catalog::{Fetch, Reply, Request};
        let mut app = app(1);
        app.catalog = crate::loader::tests::catalog();
        app.live = crate::loader::tests::live();
        app.probe = |_, _| true;
        // The package's options and the command line it asked for.
        let asked = |app: &mut App| {
            let outbox = std::mem::take(&mut app.catalog.outbox);
            let package = outbox.iter().find_map(|request| match request {
                Request::PackageOptions { generation, .. } => Some(*generation),
                _ => None,
            });
            let command = outbox.iter().find_map(|request| match request {
                Request::Command { generation, .. } => Some(*generation),
                _ => None,
            });
            (package.unwrap(), command.unwrap())
        };
        press(&mut app, KeyCode::Char('l'));
        app.tick(1.5);
        let (package, command) = asked(&mut app);
        press(&mut app, KeyCode::Esc);
        press(&mut app, KeyCode::Char('l'));
        app.tick(1.5);
        let (again, command_again) = asked(&mut app);
        assert!(again != package && again != command && command_again != command);
        // The first loader's answers arrive after it closed.
        app.catalog_reply(Reply::PackageOptions(package, Ok(Vec::new())));
        let stale = vec!["serve".to_owned(), "--stale".to_owned()];
        app.catalog_reply(Reply::Command(command, Ok(stale)));
        app.tick(1.5);
        assert_eq!(app.loader.package, Some(Fetch::Waiting));
        assert_eq!(app.loader.plan.command, None);
        // Its own are taken.
        app.catalog_reply(Reply::PackageOptions(again, Ok(Vec::new())));
        app.catalog_reply(Reply::Command(command_again, Ok(vec!["serve".to_owned()])));
        app.tick(1.5);
        assert_eq!(app.loader.package, Some(Fetch::Ready(Vec::new())));
        assert_eq!(app.loader.plan.command, Some(vec!["serve".to_owned()]));
    }

    // Starting and stopping.

    const LOGS: &str = "~/.local/state/choptop/logs";
    const CHECK: &str = include_str!("../tests/fixtures/contracts/serve-check.json");
    const PACKAGE: &str = include_str!("../tests/fixtures/contracts/list-model-options.json");
    /// A serve process that drains for half a second after SIGTERM.
    const DRAINS: &str = "trap 'draining=1' TERM\nn=0\nd=0\nwhile [ $n -lt 600 ]; do sleep 0.05; n=$((n+1)); if [ -n \"$draining\" ]; then d=$((d+1)); if [ $d -ge 10 ]; then exit 0; fi; fi; done\n";

    /// An app whose launches run `binary` and keep their records and logs
    /// in `root`.
    fn launching(root: &Path, binary: &Path) -> App {
        let mut app = app(0);
        app.catalog = Catalog::new(
            Ok(binary.to_path_buf()),
            Checkpoints::load(None, Vec::new()),
        );
        app.launches = Launches::new(Some(&root.join("state")), LOGS).0;
        app.probe = |_, _| true;
        app.wall = || 1_793_633_412;
        app
    }

    /// A Qwen switch of `instances`, each `(letter, GPUs, port, waits)`,
    /// that stops `stops` first.
    fn confirm(instances: &[(char, &[u32], u16, Vec<Target>)], stops: Vec<Stopping>) -> Confirm {
        let start = Start {
            model: "qwen38".into(),
            profile: "tp2-mtp".into(),
            placement: Placement {
                instances: instances
                    .iter()
                    .map(|(letter, gpus, port, _)| Planned {
                        letter: *letter,
                        gpus: gpus.to_vec(),
                        uuids: gpus.iter().map(|index| uuid(*index)).collect(),
                        host: "127.0.0.1".into(),
                        port: *port,
                        after: Vec::new(),
                    })
                    .collect(),
                ..Placement::default()
            },
            command: [
                "serve",
                "--model",
                "/models/qwen",
                "--tensor-parallel-size",
                "2",
            ]
            .map(str::to_owned)
            .to_vec(),
            explicit: Explicit::default(),
            choice: Choice {
                model: "qwen38".into(),
                profile: "tp2-mtp".into(),
                ..Choice::default()
            },
        };
        Confirm {
            start: Box::new(start),
            served: "Qwen3.8-Flash".into(),
            tensor_parallel: 2,
            queued: 1_793_633_412,
            switch: "20261102-153012".into(),
            logs: instances
                .iter()
                .map(|(letter, ..)| {
                    format!(
                        "qwen38-tp2-mtp-{}-20261102-153012.log",
                        letter.to_ascii_lowercase()
                    )
                })
                .collect(),
            waits: instances.iter().map(|(.., waits)| waits.clone()).collect(),
            stops,
            qualification: None,
            warnings: Vec::new(),
            memory: None,
        }
    }

    /// The serve process `guard` runs, as the placement names it.
    fn serve_of(guard: &Guard) -> Serve {
        let pid = guard.0.id();
        Serve {
            pid,
            start_ticks: process::start_ticks_of(pid).unwrap(),
            name: Some("MiMo-V2.6-Flash".into()),
            ports: vec![8310],
            starting: false,
        }
    }

    fn stopping(serve: &Serve) -> Stopping {
        Stopping {
            target: Target::Serve(serve.clone()),
            label: serve.label(),
            model: "MiMo-V2.6-Flash".into(),
            gpus: vec![0, 1],
            how: "SIGTERM, then the GPUs are free once it exits".into(),
        }
    }

    fn replacing(app: &App, id: u64) -> Stopping {
        Stopping {
            target: Target::Launch(id),
            label: app.launches.get(id).unwrap().label(),
            model: "Qwen3.8-Flash".into(),
            gpus: vec![0, 1],
            how: "replaced".into(),
        }
    }

    /// Confirms `confirm` with Enter in its dialog.
    fn run(app: &mut App, confirm: Confirm) {
        app.dialog = Some(Dialog::Start(Box::new(confirm)));
        assert_eq!(press(app, KeyCode::Enter), Action::Redraw);
        assert!(app.dialog.is_none());
    }

    fn newest(app: &App) -> u64 {
        app.launches.launches.last().unwrap().id
    }

    fn position(app: &App, who: &str, what: &str) -> usize {
        app.registry
            .events
            .iter()
            .position(|event| event.who == who && event.what.starts_with(what))
            .unwrap_or_else(|| panic!("no event {who}: {what} in {:?}", app.registry.events))
    }

    #[test]
    fn a_second_start_cancels_a_waiting_launch_and_stops_a_starting_one_first() {
        let root = scratch("second-start");
        let draining = fake_serve_running(&root.join("serve"), DRAINS);
        // Launches drain too, so one that starts before the launch it
        // replaces has exited is seen.
        let binary = root.join("drains");
        write_script(
            &binary,
            &format!("#!/bin/sh\n{CLOSE}echo \"started $*\"\n{DRAINS}"),
        );
        let mut app = launching(&root, &binary);
        let serve = serve_of(&draining);
        let label = serve.label();
        // The first start stops the serve process and waits for it.
        let first_start = confirm(
            &[('A', &[0, 1], 8310, vec![Target::Serve(serve.clone())])],
            vec![stopping(&serve)],
        );
        run(&mut app, first_start);
        let first = newest(&app);
        assert_eq!(app.launches.get(first).unwrap().phase, Phase::Waiting);
        assert_eq!(app.view, View::List);
        assert_eq!(
            app.selected,
            app.registry.launch_row(first).map(|row| row.id)
        );
        // A second start over the waiting launch cancels it before it starts.
        let second_start = confirm(
            &[(
                'A',
                &[0, 1],
                8310,
                vec![Target::Serve(serve.clone()), Target::Launch(first)],
            )],
            vec![stopping(&serve), replacing(&app, first)],
        );
        run(&mut app, second_start);
        assert!(app.launches.get(first).is_none());
        position(
            &app,
            "launch A",
            "cancelled: to start Qwen3.8-Flash tp2-mtp on GPUs 0,1",
        );
        let second = newest(&app);
        assert_eq!(
            app.launches.get(second).unwrap().waits.len(),
            1,
            "the cancelled launch is not waited for"
        );
        // The serve process got one SIGTERM, not one per start.
        let signals = app
            .registry
            .events
            .iter()
            .filter(|event| event.who == label && event.what.starts_with("stopping:"))
            .count();
        assert_eq!(signals, 1);
        wait_for(|| {
            app.supervise(5.0);
            app.launches.get(second).unwrap().phase == Phase::Running
        });
        assert!(
            position(&app, &label, "exited after draining")
                < position(
                    &app,
                    &app.launches.get(second).unwrap().label(),
                    "launched by choptop"
                )
        );
        let records = Records::load(Some(root.join("state/launches.json"))).0;
        let cancelled = &records.records[0];
        assert_eq!(
            (cancelled.pid, cancelled.ended.as_deref()),
            (
                None,
                Some("cancelled: to start Qwen3.8-Flash tp2-mtp on GPUs 0,1")
            )
        );
        // A third start over the starting launch stops its group first and
        // waits for it to exit.
        let second_label = app.launches.get(second).unwrap().label();
        let third_start = confirm(
            &[('A', &[0, 1], 8310, vec![Target::Launch(second)])],
            vec![replacing(&app, second)],
        );
        run(&mut app, third_start);
        assert!(matches!(
            app.launches.get(second).unwrap().phase,
            Phase::Stopping(_)
        ));
        let third = newest(&app);
        wait_for(|| {
            app.supervise(6.0);
            let started = app
                .launches
                .get(third)
                .is_some_and(|launch| launch.record.pid.is_some());
            if started {
                assert!(
                    app.launches.get(second).is_none(),
                    "started before the launch it replaces exited"
                );
            }
            started
        });
        let third_label = app.launches.get(third).unwrap().label();
        assert!(
            position(&app, &second_label, "exited after draining")
                < position(&app, &third_label, "launched by choptop")
        );
        app.launches
            .stop(third, "at the end of the test", &mut app.registry, 7.0)
            .unwrap();
        wait_for(|| {
            app.supervise(8.0);
            app.launches.launches.is_empty()
        });
    }

    #[test]
    fn a_serve_process_that_changed_since_the_reading_is_neither_stopped_nor_replaced() {
        let root = scratch("changed-at-confirm");
        let guard = fake_serve(&root.join("serve"));
        let binary = root.join("never");
        write_script(&binary, &format!("#!/bin/sh\n{CLOSE}exit 0\n"));
        let mut app = launching(&root, &binary);
        let mut serve = serve_of(&guard);
        serve.start_ticks += 1;
        let start = confirm(
            &[('A', &[0, 1], 8310, vec![Target::Serve(serve.clone())])],
            vec![stopping(&serve)],
        );
        run(&mut app, start);
        assert!(app.launches.launches.is_empty());
        assert_eq!(
            app.loader.flash.clone().unwrap().text,
            format!("Not started: proc {} is another process now", serve.pid)
        );
        std::thread::sleep(std::time::Duration::from_millis(200));
        assert!(process::alive(guard.0.id(), serve.start_ticks - 1));
    }

    /// DeepSeek at TP4 on GPUs 0 to 3, with `edits`.
    fn deepseek(edits: &[(Key, &str)]) -> Box<Start> {
        Box::new(Start {
            model: "deepseek41".into(),
            profile: "tp4-dspark".into(),
            placement: Placement {
                instances: vec![Planned {
                    letter: 'A',
                    gpus: vec![0, 1, 2, 3],
                    uuids: (0..4).map(uuid).collect(),
                    host: "127.0.0.1".into(),
                    port: 8310,
                    after: Vec::new(),
                }],
                ..Placement::default()
            },
            command: vec!["serve".into()],
            explicit: Explicit::default(),
            choice: Choice {
                model: "deepseek41".into(),
                profile: "tp4-dspark".into(),
                gpus: (0..4).collect(),
                edits: edits
                    .iter()
                    .map(|(key, value)| (key.clone(), (*value).to_owned()))
                    .collect(),
                custom: Vec::new(),
            },
        })
    }

    fn checking(app: &mut App, start: Box<Start>, ticket: u64) {
        app.pending = Some(Pending {
            wanted: Wanted::Start(start),
            since: Some(1.0),
            ticket: Some(ticket),
        });
    }

    #[test]
    fn serve_check_must_resolve_what_the_profile_and_the_form_say() {
        let mut app = app(0);
        app.catalog = crate::loader::tests::catalog();
        let check = Check::parse(CHECK).unwrap();
        let edits = [
            (Key::ModelOption("prefill_upper_skip".into()), "false"),
            (Key::Flag("max-sequences".into()), "48"),
        ];
        // It resolves the profile's settings with the form's values: the
        // dialog opens.
        checking(&mut app, deepseek(&edits), 7);
        app.catalog_reply(Reply::Check(8, Ok(check.clone())));
        assert!(app.dialog.is_none() && app.pending.is_some());
        app.catalog_reply(Reply::Check(7, Ok(check.clone())));
        let Some(Dialog::Start(confirm)) = &app.dialog else {
            panic!("no dialog: {:?}", app.loader.flash)
        };
        assert_eq!(confirm.title(), "Start DeepSeek tp4-dspark");
        assert!(
            confirm.warnings.contains(
                &"model option verify_allocation confidence: unqualified (package default fixed)"
                    .to_owned()
            ),
            "{:?}",
            confirm.warnings
        );
        app.dialog = None;
        // serve refuses: its own line is shown.
        checking(&mut app, deepseek(&edits), 9);
        app.catalog_reply(Reply::Check(
            9,
            Err("chopinfer: --context-parallel-size 3 does not divide 4".into()),
        ));
        assert!(app.dialog.is_none() && app.pending.is_none());
        assert_eq!(
            app.loader.flash.clone().unwrap().text,
            "Not started: chopinfer: --context-parallel-size 3 does not divide 4"
        );
        // A governed setting or model option other than the profile and
        // the form say is named, and blocks.
        let mut other = check.clone();
        other
            .settings
            .insert("prefill-budget".into(), "fixed".into());
        checking(&mut app, deepseek(&edits[1..]), 10);
        app.catalog_reply(Reply::Check(10, Ok(other)));
        assert!(app.dialog.is_none());
        assert_eq!(
            app.loader.flash.clone().unwrap().text,
            "Not started: serve --check resolves what the profile and form do not say: \
             --prefill-budget resolves to fixed, not tbt; \
             model option prefill_upper_skip resolves to false, not true"
        );
        // A key one side leaves out is not governed there: the check's
        // admission and the profile's speculation are not compared.
        assert!(check.settings.contains_key("admission"));
        assert!(!check.settings.contains_key("speculation"));
    }

    /// Answers the loader's package and command requests.
    fn answer(app: &mut App) {
        for request in std::mem::take(&mut app.catalog.outbox) {
            match request {
                Request::PackageOptions { generation, .. } => app.catalog_reply(
                    Reply::PackageOptions(generation, parse_package_options(PACKAGE)),
                ),
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
                    app.catalog_reply(Reply::Command(generation, Ok(arguments)));
                }
                _ => {}
            }
        }
    }

    /// A GPU reading and a scan at `at`: serve process 48211 on GPUs 0 and
    /// 1, and on GPU 2 `other` when given; GPU 3 is not read when
    /// `unknown`.
    fn reread(app: &mut App, at: f64, other: Option<Process>, unknown: bool) {
        app.live.scan(&crate::discover::Report::Scan {
            found: vec![serve_process(48211, &["127.0.0.1:8310"], None)],
            events: Vec::new(),
            started: at,
        });
        let listed = |processes: Option<Vec<Process>>| Reading {
            at,
            processes,
            answered: true,
            ..Reading::default()
        };
        app.live.gpu(&Report::Polled(vec![
            listed(Some(vec![on_gpu(48211)])),
            listed(Some(vec![on_gpu(48211)])),
            listed(Some(other.into_iter().collect())),
            listed((!unknown).then(Vec::new)),
        ]));
    }

    /// The loader on DeepSeek at TP4 on all four GPUs, ready to start.
    fn ready() -> App {
        let mut app = app(0);
        app.catalog = crate::loader::tests::catalog();
        app.live = crate::loader::tests::live();
        app.live.scanning = true;
        app.probe = |_, _| true;
        press(&mut app, KeyCode::Char('l'));
        app.tick(1.5);
        app.loader.picked = (0..4).collect();
        app.tick(1.5);
        answer(&mut app);
        app.tick(1.5);
        assert!(app.loader.start().is_ok(), "{:?}", app.loader.plan);
        app
    }

    #[test]
    fn a_start_waits_for_a_fresh_reading_and_stops_on_what_it_changed() {
        let mut app = ready();
        assert_eq!(press(&mut app, KeyCode::Char('S')), Action::Read);
        // The form holds still while the start is checked.
        assert_eq!(press(&mut app, KeyCode::Char('j')), Action::Nothing);
        app.asked(2.0);
        app.supervise(2.1);
        assert_eq!(
            app.pending_text(),
            Some("Checking: reading the GPUs and processes again")
        );
        // A scan begun before the request is not fresh.
        reread(&mut app, 1.9, None, false);
        app.supervise(2.2);
        assert!(app.catalog.outbox.is_empty());
        reread(&mut app, 2.2, None, false);
        app.supervise(2.3);
        assert_eq!(app.pending_text(), Some("Checking: running serve --check"));
        let Some(Request::Check {
            arguments,
            environment,
            ..
        }) = app.catalog.outbox.pop()
        else {
            panic!("no check: {:?}", app.catalog.outbox)
        };
        assert_eq!(arguments[..2], ["serve", "--model"]);
        assert_eq!(
            arguments[arguments.len() - 2..],
            ["--listen", "127.0.0.1:8310"]
        );
        assert_eq!(
            environment[0],
            (
                "CUDA_VISIBLE_DEVICES".to_owned(),
                (0..4).map(uuid).collect::<Vec<_>>().join(",")
            )
        );
        // Esc drops the start.
        press(&mut app, KeyCode::Esc);
        assert!(app.pending.is_none());
        assert_eq!(app.loader.flash.clone().unwrap().text, "Start cancelled");
        // A GPU newly held by a process choptop does not stop blocks it.
        let python = Process {
            pid: 7777,
            memory: Some(1 << 30),
            kind: Kind::Compute,
            owner: Owner {
                name: Some("python3".into()),
                ..Owner::default()
            },
        };
        assert_eq!(press(&mut app, KeyCode::Char('S')), Action::Read);
        app.asked(3.0);
        reread(&mut app, 3.1, Some(python), false);
        app.supervise(3.2);
        assert!(app.pending.is_none() && app.catalog.outbox.is_empty());
        let text = app.loader.flash.clone().unwrap().text;
        assert!(
            text.starts_with("Not started: GPU 2 is held by pid 7777"),
            "{text}"
        );
        // So does a GPU whose processes cannot be read now.
        reread(&mut app, 3.3, None, false);
        app.tick(3.4);
        assert_eq!(press(&mut app, KeyCode::Char('S')), Action::Read);
        app.asked(4.0);
        reread(&mut app, 4.1, None, true);
        app.supervise(4.2);
        let text = app.loader.flash.clone().unwrap().text;
        assert!(
            text.starts_with("Not started: GPU 3 is unknown: its process list could not be read"),
            "{text}"
        );
        // And no fresh reading at all, in time.
        reread(&mut app, 4.3, None, false);
        app.tick(4.4);
        assert_eq!(press(&mut app, KeyCode::Char('S')), Action::Read);
        app.asked(5.0);
        app.supervise(5.0 + READ_SECONDS + 1.0);
        let text = app.loader.flash.clone().unwrap().text;
        assert!(
            text.contains("no fresh GPU reading and discovery scan"),
            "{text}"
        );
    }

    #[test]
    fn x_stops_a_serve_process_of_this_user_after_a_fresh_reading_and_a_confirm() {
        let root = scratch("x-stop");
        let guard = fake_serve(&root.join("serve"));
        let mut app = launching(&root, &root.join("unused"));
        let serve = serve_of(&guard);
        app.registry.reconcile(
            vec![Found::Process(ServeProcess {
                pid: serve.pid,
                start_ticks: serve.start_ticks,
                executable: "chopinfer".into(),
                arguments: vec!["chopinfer".into(), "serve".into()],
                model_hint: Some("MiMo-V2.6-Flash".into()),
                age: None,
                listening: Vec::new(),
                cuda_visible_devices: None,
            })],
            1.0,
        );
        app.live.unavailable = Some("--no-gpus".into());
        assert_eq!(press(&mut app, KeyCode::Char('X')), Action::Read);
        app.asked(1.0);
        app.supervise(1.1);
        let Some(Dialog::Stop(stop)) = &app.dialog else {
            panic!("no stop dialog")
        };
        assert_eq!(stop.label, serve.label());
        assert_eq!(stop.what, "MiMo-V2.6-Flash");
        // Esc keeps it running.
        press(&mut app, KeyCode::Esc);
        assert!(app.dialog.is_none() && process::alive(serve.pid, serve.start_ticks));
        press(&mut app, KeyCode::Char('X'));
        app.asked(2.0);
        app.supervise(2.1);
        press(&mut app, KeyCode::Enter);
        assert_eq!(app.current().unwrap().state(2.1), State::Stopping);
        position(
            &app,
            &serve.label(),
            "stopping: SIGTERM from choptop as asked with X; it ran ",
        );
        wait_for(|| {
            app.supervise(3.5);
            app.registry.instances.is_empty()
        });
        position(&app, &serve.label(), "exited after draining 1s");
    }

    #[test]
    fn x_refuses_containers_and_targets_and_dismisses_an_exited_launch() {
        let mut app = app(1);
        assert_eq!(press(&mut app, KeyCode::Char('X')), Action::Redraw);
        assert!(app.pending.is_none());
        let event = app.registry.events.back().unwrap();
        assert!(
            event
                .what
                .starts_with("not stopped: choptop stops only serve processes of this user")
        );
        app.registry.reconcile(
            vec![Found::Container(crate::discover::docker::ServeContainer {
                daemon: "/var/run/docker.sock".into(),
                id: "c0ffee".into(),
                name: "glm-tp4".into(),
                arguments: Vec::new(),
                model_hint: None,
                seen_for: 0.0,
                addresses: vec!["127.0.0.1:8600".parse().unwrap()],
            })],
            1.0,
        );
        press(&mut app, KeyCode::Down);
        assert_eq!(press(&mut app, KeyCode::Char('X')), Action::Redraw);
        assert_eq!(
            app.registry.events.back().unwrap().what,
            "not stopped: choptop does not stop containers; stop it with Docker"
        );
        // An exited launch's row leaves with X.
        let mut record = crate::launch::records::tests::record("20261102-153012", 'A', &[0, 1]);
        (record.pid, record.start_ticks, record.exit_status) = (Some(4242), Some(1), Some(1));
        let id = app
            .launches
            .adopt(record, Phase::Exited(2.0, Some(1)), &mut app.registry, 2.0);
        let row = app.registry.launch_row(id).unwrap().id;
        app.selected = Some(row);
        assert_eq!(app.current().unwrap().state(2.0), State::Exited(Some(1)));
        assert_eq!(press(&mut app, KeyCode::Char('X')), Action::Redraw);
        assert!(app.registry.get(row).is_none() && app.launches.get(id).is_none());
    }

    #[test]
    fn l_on_an_exited_launch_opens_the_loader_with_its_choices() {
        let mut app = app(0);
        app.catalog = crate::loader::tests::catalog();
        let mut record = crate::launch::records::tests::record("20261102-153012", 'A', &[0, 1]);
        record.model = "deepseek41".into();
        record.profile = "tp4-dspark".into();
        (record.pid, record.start_ticks, record.exit_status) = (Some(4242), Some(1), Some(1));
        let mut second = record.clone();
        (second.letter, second.gpus) = ('B', vec![2, 3]);
        app.launches.records.put(record.clone());
        app.launches.records.put(second);
        let id = app
            .launches
            .adopt(record, Phase::Exited(2.0, Some(1)), &mut app.registry, 2.0);
        app.selected = app.registry.launch_row(id).map(|row| row.id);
        press(&mut app, KeyCode::Char('l'));
        assert_eq!(app.view, View::Loader);
        assert_eq!(app.loader.model.as_deref(), Some("deepseek41"));
        assert_eq!(app.loader.profile.as_deref(), Some("tp4-dspark"));
        assert_eq!(app.loader.picked, (0..4).collect());
        assert_eq!(
            app.loader.edits.get(&Key::Flag("max-sequences".into())),
            Some(&"48".to_owned())
        );
    }

    #[test]
    fn read_only_refuses_l_and_x() {
        let mut app = app(2);
        app.read_only = true;
        for key in ['l', 'X'] {
            assert_eq!(press(&mut app, KeyCode::Char(key)), Action::Redraw);
            assert_eq!(app.view, View::List);
            assert!(app.pending.is_none());
            assert_eq!(
                app.registry.events.back().unwrap().what,
                format!("{key}: choptop runs --read-only, so it starts and stops nothing")
            );
        }
        let screen = crate::ui::draw(&app, 132, 38, 1.0).text();
        let footer = screen.lines().last().unwrap();
        assert!(
            !footer.contains("load model") && !footer.contains("X stop"),
            "{footer}"
        );
        app.help = true;
        let help = crate::ui::draw(&app, 132, 38, 1.0).text();
        assert!(
            help.contains("Sources")
                && !help.contains("load a model")
                && !help.contains("  launch ")
        );
        assert!(!help.contains("l and X start and stop"));
    }

    #[test]
    fn the_selection_follows_its_row() {
        let mut app = app(3);
        press(&mut app, KeyCode::Down);
        press(&mut app, KeyCode::Down);
        let id = app.current().unwrap().id;
        app.registry.instances.remove(0);
        assert_eq!(app.current().unwrap().id, id);
        app.registry.instances.clear();
        assert!(app.current().is_none());
        assert_eq!(press(&mut app, KeyCode::Enter), Action::Redraw);
        assert_eq!(app.view, View::List);
    }
}
