//! The full-screen interface: the terminal, the input loop, and the
//! threads it drives.

use std::collections::HashMap;
use std::io::{self, Write};
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
use crate::discover::{self, Command, Discovery, Report, Schedule};
use crate::gpu::poll::{self as gpu_poll, GpuPoller};
use crate::gpu::{Gpus, Status};
use crate::instances::{EVENT_LIMIT, Followup, Registry, Source, Tone};
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
    /// sampled while the GPU view is open.
    pub gpus: bool,
    pub gpu_counters: bool,
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
                    "chopinfer-top",
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
}

/// Holds `report` while paused. A scan lists every instance, so it replaces
/// the reports held before it, keeping their latest events up to what the
/// event log holds: a long pause holds one scan and a bounded log.
fn hold(held: &mut Vec<Report>, report: Report) {
    let Report::Scan { found, events } = report else {
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

fn take_report(registry: &mut Registry, report: Report, now: f64) -> Vec<Followup> {
    match report {
        Report::Scan { found, events } => {
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
        registry.note(now(), "chopinfer-top", note, Tone::Warning);
    }
    let mut followups = Vec::new();
    for target in setup.targets {
        followups.push(Followup::Poll(registry.add_target(target, now())));
    }
    let (results, poll_results) = mpsc::channel();
    let (reports_sender, reports) = mpsc::channel::<Report>();
    let commands = setup.discovery.map(|discovery| {
        let (commands, receiver) = mpsc::channel();
        discover::spawn(
            discovery,
            Schedule::new(setup.rescan),
            origin,
            receiver,
            reports_sender,
        );
        // The startup scan, before the first frame.
        if let Ok(report) = reports.recv_timeout(FIRST_SCAN) {
            followups.extend(take_report(&mut registry, report, now()));
        }
        commands
    });
    let mut workers = Workers {
        pollers: HashMap::new(),
        results,
        commands,
        origin,
        interval: setup.interval,
        bearer: setup.bearer,
        limit: MAXIMUM_INSTANCES,
        limit_reported: false,
    };
    workers.follow(&mut registry, followups, now());
    let mut app = App::new(registry, setup.interval, setup.rescan, setup.sources);
    // GPM counters are sampled only while the GPU view is open.
    let counters = Arc::new(AtomicBool::new(false));
    let (gpu_results, gpu_reports) = mpsc::channel();
    let _gpu_poller = setup.gpus.then(|| {
        app.gpus = Gpus::new(Status::Opening, setup.gpu_counters, setup.interval);
        GpuPoller::spawn(setup.interval, origin, Arc::clone(&counters), gpu_results)
    });

    let stop = Arc::new(AtomicBool::new(false));
    for signal in [
        signal_hook::consts::SIGTERM,
        signal_hook::consts::SIGHUP,
        signal_hook::consts::SIGINT,
    ] {
        signal_hook::flag::register(signal, Arc::clone(&stop))?;
    }
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
        (&reports, &poll_results, &gpu_reports),
        (&stop, &counters),
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

fn run_loop(
    app: &mut App,
    workers: &mut Workers,
    (reports, poll_results, gpu_reports): (
        &Receiver<Report>,
        &Receiver<(u64, Poll)>,
        &Receiver<gpu_poll::Report>,
    ),
    (stop, counters): (&AtomicBool, &AtomicBool),
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
            match event {
                Event::Key(key) if key.kind == KeyEventKind::Press => match app.key(key) {
                    Action::Quit => break 'running,
                    Action::Rescan => {
                        workers.command(Command::Force);
                        app.registry
                            .note(now(), "chopinfer-top", "rescan requested", Tone::Normal);
                        dirty = true;
                    }
                    Action::Redraw => dirty = true,
                    Action::Nothing => {}
                },
                Event::Mouse(mouse) => dirty |= app.mouse(mouse, now()) != Action::Nothing,
                Event::Resize(_, _) => {
                    previous = None;
                    dirty = true;
                }
                _ => {}
            }
        }
        for report in reports.try_iter() {
            hold(&mut held_reports, report);
        }
        held_polls.extend(poll_results.try_iter());
        for report in gpu_reports.try_iter() {
            hold_gpu(&mut held_gpus, report);
        }
        counters.store(
            app.view == View::Gpus && app.gpus.counters,
            Ordering::Relaxed,
        );
        if !app.paused && !held_gpus.is_empty() {
            for report in held_gpus.drain(..) {
                app.gpus.apply(report);
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
            let (width, height) = crossterm::terminal::size()?;
            app.settle(height);
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

    #[test]
    fn rows_skipped_at_the_limit_are_polled_once_a_poller_is_free() {
        let mut registry = Registry::new(1.0);
        let first = registry.add_target(Target::parse("127.0.0.1:9").unwrap(), 0.0);
        let second = registry.add_target(Target::parse("127.0.0.2:9").unwrap(), 0.0);
        let (results, _poll_results) = mpsc::channel();
        let mut workers = Workers {
            pollers: HashMap::new(),
            results,
            commands: None,
            origin: Instant::now(),
            interval: 1.0,
            bearer: None,
            limit: 1,
            limit_reported: false,
        };
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
    fn a_paused_scan_replaces_the_reports_held_before_it() {
        let scan = |events: &[&str]| Report::Scan {
            found: Vec::new(),
            events: events.iter().map(|event| (*event).to_owned()).collect(),
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
