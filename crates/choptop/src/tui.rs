//! The full-screen interface: the terminal, the input loop, and the
//! threads it drives.

use std::collections::{BTreeMap, HashMap};
use std::io::{self, Write};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender};
use std::thread;
use std::time::{Duration, Instant};

use crossterm::cursor::{Hide, Show};
use crossterm::event::{self, DisableMouseCapture, EnableMouseCapture, Event, KeyEventKind};
use crossterm::execute;
use crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};

use crate::app::{Action, App, View};
use crate::catalog::{self, Catalog, Checkpoints};
use crate::discover::proc::ProcScanner;
use crate::discover::{self, Command, Discovery, Report, Schedule};
use crate::gpu::poll::{self as gpu_poll, GpuPoller};
use crate::gpu::{Gpus, Status};
use crate::host::poll::{self as host_poll, HostPoller};
use crate::instances::{EVENT_LIMIT, Followup, Registry, Source, Tone};
use crate::launch::Launches;
use crate::launch::occupancy::Live;
use crate::poll::{Poll, Poller};
use crate::targets::Target;
use crate::ui::{self, buffer::Buffer, buffer::flush};

/// Seconds the first frame waits for the startup scan.
const FIRST_SCAN: Duration = Duration::from_secs(3);
/// How long one wait for input lasts.
const INPUT_WAIT: Duration = Duration::from_millis(100);
/// Most instances polled at once.
const MAXIMUM_INSTANCES: usize = 64;

/// What the interface needs to start.
pub struct Setup {
    pub targets: Vec<Target>,
    pub discovery: Option<Discovery>,
    pub notes: Vec<String>,
    pub interval: f64,
    pub rescan: f64,
    pub bearer: Option<String>,
    pub sources: Vec<&'static str>,
    /// Whether to read this host's GPUs, and whether GPM counters may be
    /// sampled while the monitor is open.
    pub gpus: bool,
    pub gpu_counters: bool,
    /// The `chopinfer` the loader runs, or why there is none.
    pub chopinfer: Result<PathBuf, String>,
    pub checkpoints: Checkpoints,
    /// choptop's own `CHOPINFER_*` variables, which launches inherit.
    pub environment: BTreeMap<String, String>,
    /// Where launch logs go, as shown.
    pub logs: String,
    /// choptop's state directory, for launch logs and records.
    pub state: Option<PathBuf>,
    /// `--read-only`: no loader and no stopping.
    pub read_only: bool,
}

/// Restores the terminal when dropped, including while a panic unwinds.
struct Terminal;

impl Terminal {
    fn enter() -> io::Result<Self> {
        enable_raw_mode()?;
        let terminal = Self;
        execute!(io::stdout(), EnterAlternateScreen, Hide, EnableMouseCapture)?;
        Ok(terminal)
    }
}

fn restore() {
    let _ = execute!(
        io::stdout(),
        DisableMouseCapture,
        Show,
        LeaveAlternateScreen
    );
    let _ = disable_raw_mode();
}

impl Drop for Terminal {
    fn drop(&mut self) {
        restore();
    }
}

/// The pollers, the discovery thread's command channel, and how to start
/// a poller.
struct Workers {
    pollers: HashMap<u64, Poller>,
    results: Sender<(u64, Poll)>,
    commands: Option<Sender<Command>>,
    origin: Instant,
    interval: f64,
    bearer: Option<String>,
    /// Most pollers at once.
    limit: usize,
    limit_reported: bool,
    /// The threads that run `chopinfer` for the loader, started at its
    /// first request.
    catalog: Option<catalog::Worker>,
    catalog_replies: Sender<catalog::Reply>,
    /// Those threads' children while they run, which discovery leaves out.
    children: catalog::Children,
}

impl Workers {
    fn follow(&mut self, registry: &mut Registry, followups: Vec<Followup>, now: f64) {
        for followup in followups {
            match followup {
                Followup::Poll(id) => self.start(registry, id, now),
                Followup::Stop(id) => {
                    self.pollers.remove(&id);
                }
                Followup::Request => self.command(Command::Request),
                Followup::Invalidate(key) => self.command(Command::Invalidate(key)),
                Followup::Watch { pid, start_ticks } => {
                    self.command(Command::Watch { pid, start_ticks });
                }
            }
        }
        // Rows skipped at the limit are polled once a poller is free.
        if self.pollers.len() < self.limit {
            let waiting = registry
                .visible()
                .iter()
                .filter(|instance| !self.pollers.contains_key(&instance.id))
                .map(|instance| instance.id)
                .take(self.limit - self.pollers.len())
                .collect::<Vec<_>>();
            for id in waiting {
                self.start(registry, id, now);
            }
        }
    }

    /// Polls row `id`, or polls its new endpoints.
    fn start(&mut self, registry: &mut Registry, id: u64, now: f64) {
        let Some(instance) = registry.get(id) else {
            return;
        };
        if instance.alias_of.is_some() {
            return;
        }
        let endpoints = instance.endpoints.clone();
        if let Some(poller) = self.pollers.get(&id) {
            poller.set_endpoints(endpoints);
            return;
        }
        if self.pollers.len() >= self.limit {
            if !self.limit_reported {
                self.limit_reported = true;
                registry.note(
                    now,
                    "choptop",
                    format!(
                        "more than {} instances; the rest are polled as others go",
                        self.limit
                    ),
                    Tone::Warning,
                );
            }
            return;
        }
        let bearer = matches!(instance.source, Source::Target(_))
            .then(|| self.bearer.clone())
            .flatten();
        let poller = Poller::spawn(
            id,
            endpoints,
            bearer,
            self.interval,
            self.origin,
            self.results.clone(),
        );
        self.pollers.insert(id, poller);
    }

    fn command(&self, command: Command) {
        if let Some(commands) = &self.commands {
            let _ = commands.send(command);
        }
    }

    /// Sends the loader's requests to `chopinfer`'s threads.
    fn ask(&mut self, app: &mut App) {
        if app.catalog.outbox.is_empty() {
            return;
        }
        let requests = std::mem::take(&mut app.catalog.outbox);
        let Ok(binary) = &app.catalog.binary else {
            return;
        };
        let worker = self.catalog.get_or_insert_with(|| {
            catalog::Worker::spawn(
                binary.clone(),
                self.catalog_replies.clone(),
                self.children.clone(),
            )
        });
        for request in requests {
            worker.send(request);
        }
    }
}

/// Holds `report` while paused. A scan lists every instance, so it replaces
/// the reports held before it, keeping their latest events up to what the
/// event log holds: a long pause holds one scan and a bounded log.
fn hold(held: &mut Vec<Report>, report: Report) {
    let Report::Scan {
        found,
        events,
        started,
    } = report
    else {
        held.push(report);
        return;
    };
    let mut kept = Vec::new();
    for earlier in held.drain(..) {
        if let Report::Scan { events, .. } = earlier {
            kept.extend(events);
        }
    }
    kept.extend(events);
    kept.drain(..kept.len().saturating_sub(EVENT_LIMIT));
    held.push(Report::Scan {
        found,
        events: kept,
        started,
    });
}

/// Holds a GPU report while paused: the latest reading replaces the ones
/// held before it.
fn hold_gpu(held: &mut Vec<gpu_poll::Report>, report: gpu_poll::Report) {
    if matches!(report, gpu_poll::Report::Polled(_)) {
        held.retain(|earlier| !matches!(earlier, gpu_poll::Report::Polled(_)));
    }
    held.push(report);
}

/// Holds a host report while paused: the latest reading and the latest
/// disk sizes replace the ones held before them.
fn hold_host(held: &mut Vec<host_poll::Report>, report: host_poll::Report) {
    use host_poll::Report::{Polled, Space};
    held.retain(|earlier| {
        !matches!(
            (earlier, &report),
            (Polled { .. }, Polled { .. }) | (Space { .. }, Space { .. })
        )
    });
    held.push(report);
}

/// Takes a discovery report: the loader's readings at once, the screen's
/// held while paused.
fn receive_scan(app: &mut App, held: &mut Vec<Report>, report: Report) {
    app.live.scan(&report);
    hold(held, report);
}

/// Takes a GPU report: the loader's readings at once, the screen's held
/// while paused.
fn receive_gpu(app: &mut App, held: &mut Vec<gpu_poll::Report>, report: gpu_poll::Report) {
    app.live.gpu(&report);
    hold_gpu(held, report);
}

fn take_report(registry: &mut Registry, report: Report, now: f64) -> Vec<Followup> {
    match report {
        Report::Scan { found, events, .. } => {
            for event in events {
                registry.note(now, "discovery", event, Tone::Warning);
            }
            registry.reconcile(found, now)
        }
        Report::Listening {
            pid,
            start_ticks,
            addresses,
        } => registry.listening(pid, start_ticks, addresses, now),
    }
}

/// Runs the interface until the user quits or a signal arrives.
pub fn run(setup: Setup) -> io::Result<()> {
    let origin = Instant::now();
    let now = || origin.elapsed().as_secs_f64();
    let mut registry = Registry::new(setup.interval);
    for note in setup.notes {
        registry.note(now(), "choptop", note, Tone::Warning);
    }
    let mut followups = Vec::new();
    for target in setup.targets {
        followups.push(Followup::Poll(registry.add_target(target, now())));
    }
    let (results, poll_results) = mpsc::channel();
    let (reports_sender, reports) = mpsc::channel::<Report>();
    let mut live = Live::default();
    let children = catalog::Children::default();
    let scanning = setup.discovery.is_some();
    // With discovery off, choptop's own launches are still watched until
    // they listen, so they are polled.
    let discovery = setup.discovery.or_else(|| {
        (!setup.read_only).then(|| Discovery::launches_only(ProcScanner::new("/proc")))
    });
    let commands = discovery.map(|mut discovery| {
        discovery.exclude(&children);
        let (commands, receiver) = mpsc::channel();
        discover::spawn(
            discovery,
            Schedule::new(setup.rescan),
            origin,
            receiver,
            reports_sender,
        );
        // The startup scan, before the first frame.
        if scanning && let Ok(report) = reports.recv_timeout(FIRST_SCAN) {
            live.scan(&report);
            followups.extend(take_report(&mut registry, report, now()));
        }
        commands
    });
    let (catalog_replies, catalog_reports) = mpsc::channel();
    let mut workers = Workers {
        pollers: HashMap::new(),
        results,
        commands,
        origin,
        interval: setup.interval,
        bearer: setup.bearer,
        limit: MAXIMUM_INSTANCES,
        limit_reported: false,
        catalog: None,
        catalog_replies,
        children,
    };
    // Launches an earlier choptop started that still run keep their rows,
    // GPUs and port. A read-only choptop has no launches.
    let (mut launches, message) = if setup.read_only {
        (Launches::default(), None)
    } else {
        Launches::new(setup.state.as_deref(), &setup.logs)
    };
    if let Some(message) = message {
        registry.note(now(), "choptop", message, Tone::Warning);
    }
    launches.relink(&mut registry, now());
    followups.append(&mut launches.followups);
    workers.follow(&mut registry, followups, now());
    let mut app = App::new(registry, setup.interval, setup.rescan, setup.sources);
    app.live = live;
    app.live.scanning = scanning;
    app.catalog = Catalog::new(setup.chopinfer, setup.checkpoints);
    app.catalog.environment = setup.environment;
    app.logs = setup.logs;
    app.read_only = setup.read_only;
    app.reservations = launches.reservations();
    app.launches = launches;
    // GPM counters are sampled only while the monitor is open.
    let counters = Arc::new(AtomicBool::new(false));
    let (gpu_results, gpu_reports) = mpsc::channel();
    if !setup.gpus {
        app.live.unavailable = Some("--no-gpus".to_owned());
    }
    let gpu_poller = setup.gpus.then(|| {
        app.gpus = Gpus::new(Status::Opening, setup.gpu_counters, setup.interval);
        GpuPoller::spawn(setup.interval, origin, Arc::clone(&counters), gpu_results)
    });
    let (host_results, host_reports) = mpsc::channel();
    let _host_poller = HostPoller::spawn(setup.interval, origin, host_results);

    let stop = Arc::new(AtomicBool::new(false));
    for signal in [
        signal_hook::consts::SIGTERM,
        signal_hook::consts::SIGHUP,
        signal_hook::consts::SIGINT,
    ] {
        signal_hook::flag::register(signal, Arc::clone(&stop))?;
    }
    // Set by every resize, before its event is read, and cleared by each
    // frame.
    let resized = Arc::new(AtomicBool::new(false));
    signal_hook::flag::register(signal_hook::consts::SIGWINCH, Arc::clone(&resized))?;
    let default_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        restore();
        default_hook(info);
    }));
    let color = std::env::var_os("NO_COLOR").is_none_or(|value| value.is_empty());
    let terminal = Terminal::enter()?;
    let result = run_loop(
        &mut app,
        &mut workers,
        Inbox {
            reports: &reports,
            polls: &poll_results,
            gpus: &gpu_reports,
            host: &host_reports,
            catalog: &catalog_reports,
        },
        (&stop, &counters, &resized),
        gpu_poller.as_ref(),
        color,
        now,
    );
    drop(terminal);
    result
}

/// Reads terminal input on its own thread. crossterm's reader spins
/// without returning once the terminal hangs up, so the input loop must not
/// run on the thread that watches for signals and draws: a hung-up terminal
/// then fails the next draw, and a signal is seen within one input wait.
/// The thread ends with the process.
fn read_input() -> Receiver<Event> {
    let (sender, receiver) = mpsc::channel();
    thread::spawn(move || {
        while let Ok(event) = event::read() {
            if sender.send(event).is_err() {
                return;
            }
        }
    });
    receiver
}

/// What discovery, the instance, GPU and host pollers and `chopinfer`'s
/// threads send the loop.
struct Inbox<'a> {
    reports: &'a Receiver<Report>,
    polls: &'a Receiver<(u64, Poll)>,
    gpus: &'a Receiver<gpu_poll::Report>,
    host: &'a Receiver<host_poll::Report>,
    catalog: &'a Receiver<catalog::Reply>,
}

/// Hands input `event` to `app` at `now`. A key goes with the terminal's
/// size as `size` reads it then, not as the last resize said: a resize can
/// be read after a key typed since, and Enter must not confirm a dialog
/// that a frame of another size drew. Nor one the terminal was resized
/// away from and back to since the last frame, which `resized`, set by
/// SIGWINCH and cleared by each frame, tells whether its event is read
/// yet or not.
pub fn handle(
    app: &mut App,
    event: Event,
    size: impl FnOnce() -> Option<(u16, u16)>,
    resized: &AtomicBool,
    now: f64,
) -> Action {
    if resized.load(Ordering::SeqCst) {
        app.resized();
    }
    match event {
        Event::Key(key) if key.kind == KeyEventKind::Press => app.key(key, size()),
        Event::Mouse(mouse) => match app.mouse(mouse, now) {
            Action::Nothing => Action::Nothing,
            _ => Action::Redraw,
        },
        Event::Resize(_, _) => {
            app.resized();
            Action::Redraw
        }
        _ => Action::Nothing,
    }
}

fn run_loop(
    app: &mut App,
    workers: &mut Workers,
    Inbox {
        reports,
        polls: poll_results,
        gpus: gpu_reports,
        host: host_reports,
        catalog: catalog_reports,
    }: Inbox<'_>,
    (stop, counters, resized): (&AtomicBool, &AtomicBool, &AtomicBool),
    gpu_poller: Option<&GpuPoller>,
    color: bool,
    now: impl Fn() -> f64,
) -> io::Result<()> {
    let mut stdout = io::stdout();
    let mut previous: Option<Buffer> = None;
    let mut drawn_at = f64::NEG_INFINITY;
    // While paused, the latest result of each poller and every report wait
    // here, so the screen holds still and nothing is lost.
    let mut held_polls: HashMap<u64, Poll> = HashMap::new();
    let mut held_reports: Vec<Report> = Vec::new();
    let mut held_gpus: Vec<gpu_poll::Report> = Vec::new();
    let mut held_host: Vec<host_poll::Report> = Vec::new();
    let input = read_input();
    'running: while !stop.load(Ordering::Relaxed) {
        let mut dirty = false;
        let first = match input.recv_timeout(INPUT_WAIT) {
            Ok(event) => Some(event),
            Err(RecvTimeoutError::Timeout) => None,
            // The terminal is gone.
            Err(RecvTimeoutError::Disconnected) => break,
        };
        for event in first.into_iter().chain(input.try_iter()) {
            if matches!(event, Event::Resize(..)) {
                previous = None;
            }
            let size = || crossterm::terminal::size().ok();
            match handle(app, event, size, resized, now()) {
                Action::Quit => break 'running,
                Action::Rescan => {
                    workers.command(Command::Force);
                    app.registry
                        .note(now(), "choptop", "rescan requested", Tone::Normal);
                    dirty = true;
                }
                Action::Read => {
                    // A start or stop goes on from a GPU reading and a
                    // scan begun after this, even while paused.
                    app.asked(now());
                    if let Some(poller) = gpu_poller {
                        poller.poll_now();
                    }
                    workers.command(Command::Force);
                    dirty = true;
                }
                Action::Redraw => dirty = true,
                Action::Nothing => {}
            }
        }
        for report in reports.try_iter() {
            receive_scan(app, &mut held_reports, report);
        }
        held_polls.extend(poll_results.try_iter());
        for report in gpu_reports.try_iter() {
            receive_gpu(app, &mut held_gpus, report);
        }
        for report in host_reports.try_iter() {
            hold_host(&mut held_host, report);
        }
        // The loader's answers are not held by a pause, nor are launches.
        for reply in catalog_reports.try_iter() {
            app.catalog_reply(reply);
            dirty = true;
        }
        dirty |= app.supervise(now());
        // A start or stop that waits for a fresh reading, or a launch whose
        // waits are over, goes on from a GPU reading and a scan begun now.
        if app.needs_read() {
            app.asked(now());
            if let Some(poller) = gpu_poller {
                poller.poll_now();
            }
            workers.command(Command::Force);
        }
        let followups = std::mem::take(&mut app.launches.followups);
        if !followups.is_empty() {
            workers.follow(&mut app.registry, followups, now());
        }
        workers.ask(app);
        counters.store(
            app.view == View::Monitor && app.gpus.counters,
            Ordering::Relaxed,
        );
        if !app.paused && !held_host.is_empty() {
            let at = now();
            for report in held_host.drain(..) {
                app.host.apply(report, &mut app.registry, at);
            }
            dirty = true;
        }
        if !app.paused && !held_gpus.is_empty() {
            let at = now();
            for report in held_gpus.drain(..) {
                app.gpus.apply(report, &mut app.registry, at);
            }
            dirty = true;
        }
        if !app.paused && (!held_reports.is_empty() || !held_polls.is_empty()) {
            let at = now();
            let mut followups = Vec::new();
            for report in held_reports.drain(..) {
                followups.extend(take_report(&mut app.registry, report, at));
            }
            let mut polled = held_polls.drain().collect::<Vec<_>>();
            polled.sort_by_key(|(id, _)| *id);
            for (id, result) in polled {
                followups.extend(app.registry.apply(id, result, at));
            }
            workers.follow(&mut app.registry, followups, at);
            dirty = true;
        }
        let at = now();
        if dirty || at - drawn_at >= 1.0 {
            // Cleared before the size is read, so a resize after the read
            // leaves this frame's dialog to confirm no more.
            resized.store(false, Ordering::SeqCst);
            let (width, height) = crossterm::terminal::size()?;
            app.tick(at);
            workers.ask(app);
            app.settle(width, height);
            let frame = ui::draw(app, width, height, at);
            flush(previous.as_ref(), &frame, &mut stdout, color)?;
            stdout.flush()?;
            previous = Some(frame);
            drawn_at = at;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Workers with at most `limit` pollers, which send `results`, and
    /// sending discovery `commands`.
    fn workers(
        limit: usize,
        results: Sender<(u64, Poll)>,
        commands: Option<Sender<Command>>,
    ) -> Workers {
        let (catalog_replies, _) = mpsc::channel();
        Workers {
            pollers: HashMap::new(),
            results,
            commands,
            origin: Instant::now(),
            interval: 1.0,
            bearer: None,
            limit,
            limit_reported: false,
            catalog: None,
            catalog_replies,
            children: catalog::Children::default(),
        }
    }

    #[test]
    fn rows_skipped_at_the_limit_are_polled_once_a_poller_is_free() {
        let mut registry = Registry::new(1.0);
        let first = registry.add_target(Target::parse("127.0.0.1:9").unwrap(), 0.0);
        let second = registry.add_target(Target::parse("127.0.0.2:9").unwrap(), 0.0);
        let (results, _poll_results) = mpsc::channel();
        let mut workers = workers(1, results, None);
        workers.follow(
            &mut registry,
            vec![Followup::Poll(first), Followup::Poll(second)],
            0.0,
        );
        assert_eq!(workers.pollers.keys().collect::<Vec<_>>(), [&first]);
        assert!(registry.events.back().unwrap().what.contains("more than 1"));
        // The first row turns out to be the same server as the second.
        registry.instances[0].alias_of = Some(second);
        workers.follow(&mut registry, vec![Followup::Stop(first)], 1.0);
        assert_eq!(workers.pollers.keys().collect::<Vec<_>>(), [&second]);
    }

    /// A launch is watched by the discovery thread, which runs with
    /// discovery off too, until it listens; its row then has its address.
    #[test]
    fn a_launch_is_watched_until_it_listens() {
        let mut registry = Registry::new(1.0);
        registry.launch_process(1, 200, 7, "Qwen3.8-Flash", 0.0);
        let (commands, received) = mpsc::channel();
        let (results, _poll_results) = mpsc::channel();
        let mut workers = workers(4, results, Some(commands));
        let watch = Followup::Watch {
            pid: 200,
            start_ticks: 7,
        };
        workers.follow(&mut registry, vec![watch], 0.0);
        assert!(matches!(
            received.try_recv(),
            Ok(Command::Watch {
                pid: 200,
                start_ticks: 7
            })
        ));
        let row = registry.launch_row(1).unwrap().id;
        let report = Report::Listening {
            pid: 200,
            start_ticks: 7,
            addresses: vec!["127.0.0.1:9".parse().unwrap()],
        };
        assert_eq!(
            take_report(&mut registry, report, 1.0),
            [Followup::Poll(row)]
        );
        assert_eq!(
            registry.launch_row(1).unwrap().address().as_deref(),
            Some("127.0.0.1:9")
        );
    }

    #[test]
    fn a_paused_gpu_reading_replaces_the_ones_held_before_it() {
        use crate::gpu::Reading;
        let mut held = Vec::new();
        let opened = gpu_poll::Report::Opened {
            driver: None,
            devices: Vec::new(),
        };
        let polled = |at| {
            gpu_poll::Report::Polled(vec![Reading {
                at,
                ..Reading::default()
            }])
        };
        hold_gpu(&mut held, opened.clone());
        hold_gpu(&mut held, polled(1.0));
        hold_gpu(&mut held, polled(2.0));
        assert_eq!(held, [opened, polled(2.0)]);
    }

    #[test]
    fn a_paused_screen_still_updates_the_loaders_readings() {
        use crate::gpu::{Device, Reading};
        use crate::launch::occupancy::{Inputs, Occupancy, occupancy};
        let mut app = App::new(Registry::new(1.0), 1.0, 15.0, vec!["proc"]);
        app.paused = true;
        let (mut held_gpus, mut held_reports) = (Vec::new(), Vec::new());
        let device = Device {
            uuid: Some(format!(
                "GPU-{}-0000-0000-0000-{}",
                "0".repeat(8),
                "0".repeat(12)
            )),
            ..Device::default()
        };
        receive_gpu(
            &mut app,
            &mut held_gpus,
            gpu_poll::Report::Opened {
                driver: None,
                devices: vec![device],
            },
        );
        receive_gpu(
            &mut app,
            &mut held_gpus,
            gpu_poll::Report::Polled(vec![Reading {
                at: 5.0,
                processes: Some(Vec::new()),
                answered: true,
                ..Reading::default()
            }]),
        );
        receive_scan(
            &mut app,
            &mut held_reports,
            Report::Scan {
                found: Vec::new(),
                events: Vec::new(),
                started: 0.0,
            },
        );
        // The screen holds its frame: no GPUs yet, every report held.
        assert!(app.gpus.devices.is_empty());
        assert_eq!((held_gpus.len(), held_reports.len()), (2, 1));
        // The loader reads the latest.
        let gpus = occupancy(&Inputs {
            live: &app.live,
            instances: &app.registry.instances,
            reservations: &app.reservations,
            waited: &[],
            interval: app.interval,
            now: 5.5,
        });
        assert_eq!(gpus[0].occupancy, Occupancy::Free);
    }

    #[test]
    fn a_paused_host_reading_replaces_the_one_held_before_it() {
        use crate::host::Reading;
        let mut held = Vec::new();
        let polled = |at| host_poll::Report::Polled {
            reading: Box::new(Reading {
                at,
                ..Reading::default()
            }),
            chips: Vec::new(),
        };
        let space = |at| host_poll::Report::Space {
            at,
            filesystems: Vec::new(),
        };
        let opened = host_poll::Report::Unavailable("x".to_owned());
        hold_host(&mut held, opened.clone());
        hold_host(&mut held, polled(1.0));
        hold_host(&mut held, space(1.0));
        hold_host(&mut held, polled(2.0));
        hold_host(&mut held, space(6.0));
        assert_eq!(held, [opened, polled(2.0), space(6.0)]);
    }

    #[test]
    fn a_paused_scan_replaces_the_reports_held_before_it() {
        let scan = |events: &[&str]| Report::Scan {
            found: Vec::new(),
            events: events.iter().map(|event| (*event).to_owned()).collect(),
            started: 0.0,
        };
        let mut held = Vec::new();
        hold(&mut held, scan(&["a"]));
        hold(
            &mut held,
            Report::Listening {
                pid: 1,
                start_ticks: 2,
                addresses: Vec::new(),
            },
        );
        hold(&mut held, scan(&[]));
        hold(&mut held, scan(&["b"]));
        assert_eq!(held.len(), 1);
        let Report::Scan { events, .. } = &held[0] else {
            panic!("{held:?}");
        };
        assert_eq!(events, &["a", "b"]);
        // A daemon failing on every scan of a long pause.
        for index in 0..1000 {
            hold(&mut held, scan(&[&format!("failure {index}")]));
        }
        let Report::Scan { events, .. } = &held[0] else {
            panic!("{held:?}");
        };
        assert_eq!(events.len(), EVENT_LIMIT);
        assert_eq!(events.last().unwrap(), "failure 999");
    }
}
