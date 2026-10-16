//! The instances shown, where they came from, and what happened to them.
//!
//! Explicit targets stay listed. Discovered processes and containers come
//! and go with discovery; one appears as a starting row before it has a
//! listener, since serve opens its listener only after engine construction
//! and warmup. A response carrying the startup nonce makes a row a polled
//! instance. Rows that report the same nonce are one server and are shown
//! once.

use std::collections::VecDeque;
use std::path::PathBuf;

use crate::discover::{Found, Key};
use crate::health::Health;
use crate::http::Endpoint;
use crate::poll::{Answer, Poll};
use crate::stats::Stats;
use crate::targets::Target;

/// Events kept for the events panel.
pub const EVENT_LIMIT: usize = 200;
/// Seconds a process may listen without a valid answer before it is shown
/// as not answering.
const NOT_ANSWERING_SECONDS: f64 = 30.0;
/// Seconds between rereads of the process list of a container row that has
/// never answered.
const REREAD_SECONDS: f64 = 60.0;

/// Where a row came from.
#[derive(Clone, Debug, PartialEq)]
pub enum Source {
    Target(Target),
    Process {
        pid: u32,
        start_ticks: u64,
    },
    Container {
        daemon: PathBuf,
        id: String,
        name: String,
    },
}

impl Source {
    /// `target`, `proc 48211` or `docker glm-tp4`.
    pub fn label(&self) -> String {
        match self {
            Self::Target(_) => "target".to_owned(),
            Self::Process { pid, .. } => format!("proc {pid}"),
            Self::Container { name, .. } => format!("docker {name}"),
        }
    }

    pub fn kind(&self) -> &'static str {
        match self {
            Self::Target(_) => "target",
            Self::Process { .. } => "proc",
            Self::Container { .. } => "docker",
        }
    }

    /// Which row stays visible when two are one server: an explicit target,
    /// then a process, then a container.
    fn rank(&self) -> u8 {
        match self {
            Self::Target(_) => 0,
            Self::Process { .. } => 1,
            Self::Container { .. } => 2,
        }
    }

    pub fn key(&self) -> Option<Key> {
        match self {
            Self::Target(_) => None,
            Self::Process { pid, start_ticks } => Some(Key::Process {
                pid: *pid,
                start_ticks: *start_ticks,
            }),
            Self::Container { daemon, id, .. } => Some(Key::Container {
                daemon: daemon.clone(),
                id: id.clone(),
            }),
        }
    }
}

/// How an event reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tone {
    Normal,
    Good,
    Warning,
    Bad,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Event {
    /// Seconds on chopinfer-top's clock.
    pub at: f64,
    pub who: String,
    pub what: String,
    pub tone: Tone,
}

/// What a row shows as its state.
#[derive(Clone, Debug, PartialEq)]
pub enum State {
    /// Ready, with requests running or waiting.
    Ready,
    /// Ready with nothing running or waiting.
    Idle,
    NotReady(String),
    /// A target not polled yet.
    Connecting,
    /// Discovered, not answering yet: loading or warming up.
    Starting,
    /// Listening for a while without a valid answer.
    NotAnswering,
    Unreachable(String),
    /// Answers without the chopinfer startup nonce header.
    NotChopinfer,
    /// The generation worker failed.
    Failed,
    /// Reuse is unsafe after a commit, rollback or prefix failure.
    Poisoned,
}

impl State {
    pub fn label(&self) -> &'static str {
        match self {
            Self::Ready => "ready",
            Self::Idle => "ready idle",
            Self::NotReady(_) => "not ready",
            Self::Connecting => "connecting",
            Self::Starting => "starting",
            Self::NotAnswering => "not answering",
            Self::Unreachable(_) => "unreachable",
            Self::NotChopinfer => "not chopinfer",
            Self::Failed => "failed",
            Self::Poisoned => "poisoned",
        }
    }

    pub fn tone(&self) -> Tone {
        match self {
            Self::Ready | Self::Idle => Tone::Good,
            Self::NotReady(_) | Self::Connecting | Self::Starting | Self::NotAnswering => {
                Tone::Warning
            }
            Self::Unreachable(_) | Self::NotChopinfer | Self::Failed | Self::Poisoned => Tone::Bad,
        }
    }

    pub fn is_live(&self) -> bool {
        matches!(
            self,
            Self::Ready | Self::Idle | Self::NotReady(_) | Self::Failed | Self::Poisoned
        )
    }
}

/// One row.
#[derive(Clone, Debug)]
pub struct Instance {
    pub id: u64,
    pub source: Source,
    /// Where it is polled, best first.
    pub endpoints: Vec<Endpoint>,
    /// The discovered command line.
    pub arguments: Vec<String>,
    pub model_hint: Option<String>,
    /// The clock reading at which the process started, when known.
    pub born: Option<f64>,
    pub first_seen: f64,
    /// When an address was first known.
    pub listening_since: Option<f64>,
    /// The address that last answered.
    pub endpoint: Option<String>,
    pub health: Option<Health>,
    /// The startup nonce of the last answer, kept even when `/metrics`
    /// could not be read, so restarts and merges do not need statistics.
    pub nonce: Option<String>,
    pub stats: Stats,
    /// Why `/metrics` was not read at the last answer.
    pub problem: Option<String>,
    /// Why the last poll failed.
    pub error: Option<String>,
    pub answered: bool,
    pub reachable: bool,
    pub not_chopinfer: bool,
    /// The visible row this one is the same server as.
    pub alias_of: Option<u64>,
    /// Labels of rows shown as this one.
    pub aliases: Vec<String>,
    pub restarts: u32,
    pub last_restart: Option<f64>,
    /// When a container row that never answered last asked for its process
    /// list to be read again.
    reread_at: f64,
}

impl Instance {
    fn new(id: u64, source: Source, endpoints: Vec<Endpoint>, interval: f64, now: f64) -> Self {
        Self {
            id,
            source,
            listening_since: (!endpoints.is_empty()).then_some(now),
            endpoints,
            arguments: Vec::new(),
            model_hint: None,
            born: None,
            first_seen: now,
            endpoint: None,
            health: None,
            nonce: None,
            stats: Stats::new(interval),
            problem: None,
            error: None,
            answered: false,
            reachable: false,
            not_chopinfer: false,
            alias_of: None,
            aliases: Vec::new(),
            restarts: 0,
            last_restart: None,
            reread_at: f64::NEG_INFINITY,
        }
    }

    pub fn state(&self, now: f64) -> State {
        if self.reachable {
            let gauges = self.stats.latest().map(|snapshot| &snapshot.gauges);
            if gauges.and_then(|gauges| gauges.poisoned) == Some(true) {
                return State::Poisoned;
            }
            if gauges.and_then(|gauges| gauges.worker) == Some("failed") {
                return State::Failed;
            }
            if let Some(reason) = not_ready(self.health.as_ref()) {
                return State::NotReady(reason);
            }
            let busy = gauges.is_some_and(|gauges| {
                gauges.running.unwrap_or(0.0) > 0.0 || gauges.waiting.unwrap_or(0.0) > 0.0
            });
            return if busy { State::Ready } else { State::Idle };
        }
        if self.not_chopinfer {
            return State::NotChopinfer;
        }
        let error = || self.error.clone().unwrap_or_default();
        if self.answered {
            return State::Unreachable(error());
        }
        match &self.source {
            Source::Target(_) if self.error.is_some() => State::Unreachable(error()),
            Source::Target(_) => State::Connecting,
            Source::Container { .. } if self.endpoints.is_empty() => {
                State::Unreachable("no address the host can reach".to_owned())
            }
            Source::Process { .. }
                if self
                    .listening_since
                    .is_some_and(|since| now - since > NOT_ANSWERING_SECONDS) =>
            {
                State::NotAnswering
            }
            _ => State::Starting,
        }
    }

    /// The name it is shown by: the first served model name, else what its
    /// command line says.
    pub fn name(&self) -> Option<&str> {
        self.health
            .as_ref()
            .and_then(Health::model_name)
            .or(self.model_hint.as_deref())
    }

    /// The address that answered, else the first one known.
    pub fn address(&self) -> Option<String> {
        self.endpoint
            .clone()
            .or_else(|| self.endpoints.first().map(Endpoint::authority))
    }

    /// Seconds it has run: the process age when discovery read it, else the
    /// server's metrics clock, which starts when its listener opens.
    pub fn uptime(&self, now: f64) -> Option<f64> {
        if let Some(born) = self.born {
            return Some((now - born).max(0.0));
        }
        self.stats
            .latest()
            .filter(|_| self.reachable)
            .and_then(|snapshot| snapshot.server_seconds)
    }

    pub fn label(&self) -> String {
        match &self.source {
            Source::Target(target) => format!("target {}", target.display()),
            source => source.label(),
        }
    }
}

/// Why a server that answered is not ready: its reason, else its status;
/// `None` when it is ready.
fn not_ready(health: Option<&Health>) -> Option<String> {
    match health {
        Some(health) if health.ready() => None,
        Some(health) => Some(
            health
                .reason
                .clone()
                .unwrap_or_else(|| health.status.clone()),
        ),
        None => Some("unreadable /health".to_owned()),
    }
}

/// What the owner of the pollers and the discovery thread should do.
#[derive(Clone, Debug, PartialEq)]
pub enum Followup {
    /// Start polling the row, or poll its new endpoints.
    Poll(u64),
    /// Stop polling the row.
    Stop(u64),
    /// Run a discovery scan soon.
    Request,
    /// Read a container's process list again at the next scan.
    Invalidate(Key),
}

/// Every row and the event log.
#[derive(Debug)]
pub struct Registry {
    pub instances: Vec<Instance>,
    pub events: VecDeque<Event>,
    next_id: u64,
    interval: f64,
}

impl Registry {
    pub fn new(interval: f64) -> Self {
        Self {
            instances: Vec::new(),
            events: VecDeque::new(),
            next_id: 1,
            interval,
        }
    }

    pub fn note(&mut self, at: f64, who: impl Into<String>, what: impl Into<String>, tone: Tone) {
        if self.events.len() == EVENT_LIMIT {
            self.events.pop_front();
        }
        self.events.push_back(Event {
            at,
            who: who.into(),
            what: what.into(),
            tone,
        });
    }

    pub fn get(&self, id: u64) -> Option<&Instance> {
        self.instances.iter().find(|instance| instance.id == id)
    }

    /// The rows shown: every row that is not another row's server.
    pub fn visible(&self) -> Vec<&Instance> {
        self.instances
            .iter()
            .filter(|instance| instance.alias_of.is_none())
            .collect()
    }

    pub fn add_target(&mut self, target: Target, now: f64) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        let endpoints = vec![target.endpoint()];
        self.instances.push(Instance::new(
            id,
            Source::Target(target),
            endpoints,
            self.interval,
            now,
        ));
        id
    }

    /// Takes a full discovery scan: adds what is new, updates addresses and
    /// removes what is gone.
    pub fn reconcile(&mut self, found: Vec<Found>, now: f64) -> Vec<Followup> {
        let mut followups = Vec::new();
        let keys = found.iter().map(Found::key).collect::<Vec<_>>();
        for item in found {
            let key = item.key();
            let (source, endpoints, arguments, hint, age) = match item {
                Found::Process(process) => (
                    Source::Process {
                        pid: process.pid,
                        start_ticks: process.start_ticks,
                    },
                    process
                        .listening
                        .into_iter()
                        .map(Endpoint::Address)
                        .collect::<Vec<_>>(),
                    process.arguments,
                    process.model_hint,
                    process.age,
                ),
                Found::Container(container) => (
                    Source::Container {
                        daemon: container.daemon,
                        id: container.id,
                        name: container.name,
                    },
                    container
                        .addresses
                        .into_iter()
                        .map(Endpoint::Address)
                        .collect(),
                    container.arguments,
                    container.model_hint,
                    None,
                ),
            };
            if let Some(instance) = self
                .instances
                .iter_mut()
                .find(|instance| instance.source.key().as_ref() == Some(&key))
            {
                instance.arguments = arguments;
                instance.model_hint = hint;
                if instance.endpoints != endpoints {
                    if instance.listening_since.is_none() && !endpoints.is_empty() {
                        instance.listening_since = Some(now);
                    }
                    instance.endpoints = endpoints;
                    if instance.alias_of.is_none() {
                        followups.push(Followup::Poll(instance.id));
                    }
                }
                continue;
            }
            let id = self.next_id;
            self.next_id += 1;
            let mut instance = Instance::new(id, source, endpoints, self.interval, now);
            instance.arguments = arguments;
            instance.model_hint = hint;
            instance.born = age.map(|age| now - age);
            let what = match (&instance.source, instance.endpoints.first()) {
                (Source::Process { .. }, None) => {
                    "discovered: serve process, no listener yet (loading)".to_owned()
                }
                (Source::Process { .. }, Some(endpoint)) => {
                    format!(
                        "discovered: serve process listening on {}",
                        endpoint.authority()
                    )
                }
                (_, Some(endpoint)) => {
                    format!(
                        "discovered: container running serve, reached at {}",
                        endpoint.authority()
                    )
                }
                (_, None) => "discovered: container running serve".to_owned(),
            };
            self.note(now, instance.label(), what, Tone::Normal);
            self.instances.push(instance);
            followups.push(Followup::Poll(id));
        }
        let gone = self
            .instances
            .iter()
            .filter(|instance| {
                instance
                    .source
                    .key()
                    .is_some_and(|key| !keys.contains(&key))
            })
            .map(|instance| instance.id)
            .collect::<Vec<_>>();
        for id in gone {
            followups.extend(self.remove(id, now));
        }
        followups
    }

    fn remove(&mut self, id: u64, now: f64) -> Vec<Followup> {
        let Some(index) = self.instances.iter().position(|instance| instance.id == id) else {
            return Vec::new();
        };
        let instance = self.instances.remove(index);
        let what = match instance.source {
            Source::Container { .. } => "gone: the container stopped or no longer runs serve",
            _ => "gone: the process exited",
        };
        self.note(now, instance.label(), what, Tone::Normal);
        let mut followups = vec![Followup::Stop(id)];
        if let Some(primary) = instance.alias_of
            && let Some(primary) = self.instances.iter_mut().find(|row| row.id == primary)
        {
            let label = instance.label();
            primary.aliases.retain(|alias| *alias != label);
        }
        followups.extend(self.release_aliases(id));
        followups
    }

    /// Shows again the rows that were shown as `id`, and polls them.
    fn release_aliases(&mut self, id: u64) -> Vec<Followup> {
        let mut followups = Vec::new();
        for instance in &mut self.instances {
            if instance.alias_of == Some(id) {
                instance.alias_of = None;
                instance.nonce = None;
                instance.stats = Stats::new(self.interval);
                followups.push(Followup::Poll(instance.id));
            }
        }
        if let Some(primary) = self.instances.iter_mut().find(|row| row.id == id) {
            primary.aliases.clear();
        }
        followups
    }

    /// A loading process now listens on `addresses`.
    pub fn listening(
        &mut self,
        pid: u32,
        start_ticks: u64,
        addresses: Vec<std::net::SocketAddr>,
        now: f64,
    ) -> Vec<Followup> {
        let process = Source::Process { pid, start_ticks };
        let Some(instance) = self
            .instances
            .iter_mut()
            .find(|instance| instance.source == process)
        else {
            return Vec::new();
        };
        let endpoints = addresses
            .into_iter()
            .map(Endpoint::Address)
            .collect::<Vec<_>>();
        if instance.endpoints == endpoints {
            return Vec::new();
        }
        instance.endpoints = endpoints;
        instance.listening_since.get_or_insert(now);
        let id = instance.id;
        let label = instance.label();
        let address = instance.address().unwrap_or_default();
        self.note(now, label, format!("listening on {address}"), Tone::Normal);
        vec![Followup::Poll(id)]
    }

    /// Takes one poll result for row `id`.
    pub fn apply(&mut self, id: u64, poll: Poll, now: f64) -> Vec<Followup> {
        let interval = self.interval;
        let Some(index) = self.instances.iter().position(|instance| instance.id == id) else {
            return Vec::new();
        };
        if self.instances[index].alias_of.is_some() {
            return Vec::new();
        }
        let mut followups = Vec::new();
        let mut notes = Vec::new();
        let instance = &mut self.instances[index];
        let key = instance.source.key();
        let label = instance.label();
        let lost = |followups: &mut Vec<Followup>| {
            followups.push(Followup::Request);
            if let Some(key @ Key::Container { .. }) = &key {
                followups.push(Followup::Invalidate(key.clone()));
            }
        };
        match poll {
            Poll::Answer(answer) => {
                let Answer {
                    endpoint,
                    nonce,
                    health,
                    snapshot,
                    problem,
                } = *answer;
                let new_nonce = instance.nonce.as_ref().is_some_and(|known| *known != nonce);
                if new_nonce {
                    instance.stats = Stats::new(interval);
                }
                instance.nonce = Some(nonce.clone());
                let backwards = match snapshot {
                    Some(snapshot) => instance.stats.push(Some(&nonce), snapshot),
                    None => false,
                };
                if new_nonce || backwards {
                    instance.restarts += 1;
                    instance.last_restart = Some(now);
                    notes.push((
                        if new_nonce {
                            "restarted: new startup nonce; counters re-baselined"
                        } else {
                            "counters went backwards; re-baselined"
                        }
                        .to_owned(),
                        Tone::Warning,
                    ));
                    lost(&mut followups);
                }
                if !instance.answered {
                    notes.push((format!("answering at {endpoint}"), Tone::Good));
                } else if !instance.reachable {
                    notes.push((format!("reachable again at {endpoint}"), Tone::Good));
                }
                if let Some(problem) = &problem
                    && instance.problem.as_ref() != Some(problem)
                {
                    notes.push((problem.clone(), Tone::Warning));
                }
                let before = instance
                    .reachable
                    .then(|| not_ready(instance.health.as_ref()))
                    .flatten();
                let after = not_ready(health.as_ref());
                if after != before {
                    match &after {
                        Some(reason) => notes.push((format!("not ready: {reason}"), Tone::Warning)),
                        None if instance.reachable => notes.push(("ready".to_owned(), Tone::Good)),
                        None => {}
                    }
                }
                instance.answered = true;
                instance.reachable = true;
                instance.not_chopinfer = false;
                instance.endpoint = Some(endpoint);
                instance.health = health;
                instance.problem = problem;
                instance.error = None;
                if new_nonce {
                    followups.extend(self.release_aliases(id));
                }
                followups.extend(self.merge(id, &nonce, now));
            }
            Poll::NotChopinfer { endpoint } => {
                if !instance.not_chopinfer {
                    notes.push((
                        format!("{endpoint} answers without the chopinfer startup nonce header"),
                        Tone::Bad,
                    ));
                }
                let was = instance.reachable;
                instance.not_chopinfer = true;
                instance.reachable = false;
                if was {
                    lost(&mut followups);
                    followups.extend(self.release_aliases(id));
                }
            }
            Poll::Failed(error) => {
                let was = instance.reachable;
                if was {
                    notes.push((format!("unreachable: {error}"), Tone::Bad));
                    lost(&mut followups);
                } else if matches!(instance.source, Source::Target(_))
                    && !instance.answered
                    && instance.error.is_none()
                {
                    notes.push((
                        format!("unreachable: {error}; retrying every 5s"),
                        Tone::Bad,
                    ));
                } else if let Some(key @ Key::Container { .. }) = &key
                    && !instance.answered
                    && now - instance.reread_at >= REREAD_SECONDS
                {
                    // Its cached command may be stale: serve may have
                    // restarted on another port before it ever answered.
                    instance.reread_at = now;
                    followups.push(Followup::Invalidate(key.clone()));
                }
                instance.reachable = false;
                instance.not_chopinfer = false;
                instance.error = Some(error);
                if was {
                    // Rows shown as this one may still reach the server.
                    followups.extend(self.release_aliases(id));
                }
            }
        }
        for (what, tone) in notes {
            self.note(now, label.clone(), what, tone);
        }
        followups
    }

    /// Shows `id` and any other visible row with the same nonce once.
    fn merge(&mut self, id: u64, nonce: &str, now: f64) -> Vec<Followup> {
        let Some(other) = self
            .instances
            .iter()
            .find(|instance| {
                instance.id != id
                    && instance.alias_of.is_none()
                    && instance.reachable
                    && instance.nonce.as_deref() == Some(nonce)
            })
            .map(|instance| instance.id)
        else {
            return Vec::new();
        };
        let order = |id: u64| {
            let instance = self.get(id).expect("a known row");
            (instance.source.rank(), instance.id)
        };
        let (primary, alias) = if order(id) <= order(other) {
            (id, other)
        } else {
            (other, id)
        };
        let alias_label = self.get(alias).expect("a known row").label();
        let primary_label = self.get(primary).expect("a known row").label();
        // Rows already shown as the alias are now shown as the primary, so
        // losing the primary releases them too.
        let mut labels = vec![alias_label.clone()];
        for instance in &mut self.instances {
            if instance.id == alias {
                instance.alias_of = Some(primary);
                labels.append(&mut instance.aliases);
            } else if instance.alias_of == Some(alias) {
                instance.alias_of = Some(primary);
            }
        }
        if let Some(instance) = self.instances.iter_mut().find(|row| row.id == primary) {
            for label in labels {
                if !instance.aliases.contains(&label) {
                    instance.aliases.push(label);
                }
            }
        }
        self.note(
            now,
            alias_label,
            format!("same server as {primary_label}; shown once"),
            Tone::Normal,
        );
        vec![Followup::Stop(alias)]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::discover::docker::ServeContainer;
    use crate::discover::proc::ServeProcess;
    use crate::stats::Snapshot;

    fn process(pid: u32, listening: &[&str]) -> Found {
        Found::Process(ServeProcess {
            pid,
            start_ticks: 7,
            executable: "chopinfer".to_owned(),
            arguments: vec!["chopinfer".to_owned(), "serve".to_owned()],
            model_hint: Some("hint".to_owned()),
            age: Some(10.0),
            listening: listening
                .iter()
                .map(|address| address.parse().unwrap())
                .collect(),
        })
    }

    fn container(id: &str, address: &str) -> Found {
        Found::Container(ServeContainer {
            daemon: PathBuf::from("/var/run/docker.sock"),
            id: id.to_owned(),
            name: format!("c-{id}"),
            arguments: Vec::new(),
            model_hint: None,
            seen_for: 0.0,
            addresses: vec![address.parse().unwrap()],
        })
    }

    fn answer(nonce: &str, generated: f64, ready: bool) -> Poll {
        let health = format!(
            r#"{{"status":"{}","serving":{{"served_model_names":["served"]}}}}"#,
            if ready { "ready" } else { "not_ready" }
        );
        let mut snapshot = Snapshot {
            server_seconds: Some(100.0 + generated),
            ..Snapshot::default()
        };
        snapshot.counters.generation_tokens = generated;
        snapshot.gauges.running = Some(1.0);
        Poll::Answer(Box::new(Answer {
            endpoint: "127.0.0.1:8311".to_owned(),
            nonce: nonce.to_owned(),
            health: Health::parse(health.as_bytes()),
            snapshot: Some(snapshot),
            problem: None,
        }))
    }

    #[test]
    fn a_loading_process_is_a_starting_row_until_it_answers() {
        let mut registry = Registry::new(1.0);
        let followups = registry.reconcile(vec![process(200, &[])], 0.0);
        assert_eq!(followups, [Followup::Poll(1)]);
        let row = registry.get(1).unwrap();
        assert_eq!(row.state(0.0), State::Starting);
        assert_eq!(row.name(), Some("hint"));
        assert_eq!(row.uptime(5.0), Some(15.0));
        // Another process that took its pid is not it.
        assert!(
            registry
                .listening(200, 8, vec!["127.0.0.1:8311".parse().unwrap()], 40.0)
                .is_empty()
        );
        // Its socket opens; it is polled there and answers.
        assert_eq!(
            registry.listening(200, 7, vec!["127.0.0.1:8311".parse().unwrap()], 40.0),
            [Followup::Poll(1)]
        );
        assert_eq!(registry.get(1).unwrap().state(41.0), State::Starting);
        registry.apply(1, Poll::Failed("refused".into()), 41.0);
        assert_eq!(registry.get(1).unwrap().state(80.0), State::NotAnswering);
        assert!(registry.apply(1, answer("n1", 5.0, true), 81.0).is_empty());
        let row = registry.get(1).unwrap();
        assert_eq!(row.state(81.0), State::Ready);
        assert_eq!(row.name(), Some("served"));
        assert_eq!(row.uptime(81.0), Some(91.0));
        // The process exits.
        assert_eq!(registry.reconcile(Vec::new(), 90.0), [Followup::Stop(1)]);
        assert!(registry.instances.is_empty());
        let events = registry
            .events
            .iter()
            .map(|event| event.what.as_str())
            .collect::<Vec<_>>();
        assert_eq!(
            events,
            [
                "discovered: serve process, no listener yet (loading)",
                "listening on 127.0.0.1:8311",
                "answering at 127.0.0.1:8311",
                "gone: the process exited",
            ]
        );
    }

    #[test]
    fn a_new_nonce_restarts_and_a_lost_container_is_invalidated() {
        let mut registry = Registry::new(1.0);
        registry.reconcile(vec![container("aaa", "127.0.0.1:8600")], 0.0);
        registry.apply(1, answer("n1", 5.0, true), 1.0);
        registry.apply(1, answer("n1", 9.0, true), 2.0);
        assert_eq!(registry.get(1).unwrap().stats.len(), 2);
        let key = Key::Container {
            daemon: PathBuf::from("/var/run/docker.sock"),
            id: "aaa".to_owned(),
        };
        let followups = registry.apply(1, answer("n2", 1.0, true), 3.0);
        assert_eq!(
            followups,
            [Followup::Request, Followup::Invalidate(key.clone())]
        );
        let row = registry.get(1).unwrap();
        assert_eq!(
            (row.restarts, row.stats.len(), row.nonce.as_deref()),
            (1, 1, Some("n2"))
        );
        let followups = registry.apply(1, Poll::Failed("refused".into()), 4.0);
        assert_eq!(followups, [Followup::Request, Followup::Invalidate(key)]);
        assert_eq!(
            registry.get(1).unwrap().state(4.0),
            State::Unreachable("refused".into())
        );
        // A second failure is not a new loss.
        assert!(
            registry
                .apply(1, Poll::Failed("refused".into()), 9.0)
                .is_empty()
        );
        assert_eq!(registry.events.back().unwrap().what, "unreachable: refused");
    }

    #[test]
    fn rows_with_one_nonce_are_shown_once() {
        let mut registry = Registry::new(1.0);
        let target = registry.add_target(Target::parse(":8311").unwrap(), 0.0);
        assert_eq!(registry.get(target).unwrap().state(0.0), State::Connecting);
        registry.reconcile(vec![process(200, &["127.0.0.1:8311"])], 0.0);
        registry.apply(2, answer("n1", 5.0, true), 1.0);
        let followups = registry.apply(target, answer("n1", 5.0, true), 1.0);
        assert_eq!(followups, [Followup::Stop(2)]);
        assert_eq!(registry.visible().len(), 1);
        assert_eq!(registry.get(target).unwrap().aliases, ["proc 200"]);
        // Results for the hidden row are ignored.
        assert!(registry.apply(2, answer("n1", 6.0, true), 2.0).is_empty());
        // The server restarts: the target sees a new nonce and the process
        // row is shown again until it reports the same one.
        let followups = registry.apply(target, answer("n2", 1.0, true), 3.0);
        assert!(followups.contains(&Followup::Poll(2)), "{followups:?}");
        assert_eq!(registry.visible().len(), 2);
        assert_eq!(
            registry.apply(2, answer("n2", 1.0, true), 4.0),
            [Followup::Stop(2)]
        );
        // The process exits; the target stays.
        registry.reconcile(Vec::new(), 5.0);
        assert!(registry.get(target).unwrap().aliases.is_empty());
        assert_eq!(registry.visible().len(), 1);
    }

    #[test]
    fn rows_shown_as_an_absorbed_row_follow_it_to_the_new_primary() {
        let mut registry = Registry::new(1.0);
        let ids = [":8311", ":8312", ":8313"]
            .map(|target| registry.add_target(Target::parse(target).unwrap(), 0.0));
        registry.apply(ids[1], answer("a", 1.0, true), 1.0);
        registry.apply(ids[2], answer("a", 1.0, true), 1.0);
        assert_eq!(registry.get(ids[2]).unwrap().alias_of, Some(ids[1]));
        registry.apply(ids[0], answer("a", 1.0, true), 2.0);
        assert_eq!(registry.visible().len(), 1);
        assert_eq!(registry.get(ids[2]).unwrap().alias_of, Some(ids[0]));
        assert_eq!(
            registry.get(ids[0]).unwrap().aliases,
            ["target 127.0.0.1:8312", "target 127.0.0.1:8313"]
        );
        assert!(registry.get(ids[1]).unwrap().aliases.is_empty());
        // The primary restarts: both rows are polled on their own again.
        let followups = registry.apply(ids[0], answer("b", 1.0, true), 3.0);
        assert!(followups.contains(&Followup::Poll(ids[1])), "{followups:?}");
        assert!(followups.contains(&Followup::Poll(ids[2])), "{followups:?}");
        assert_eq!(registry.visible().len(), 3);
    }

    #[test]
    fn the_nonce_is_kept_without_metrics() {
        let without_metrics = |nonce: &str| {
            let Poll::Answer(mut answer) = answer(nonce, 0.0, true) else {
                unreachable!()
            };
            answer.snapshot = None;
            answer.problem = Some("/metrics answered HTTP 503".to_owned());
            Poll::Answer(answer)
        };
        let mut registry = Registry::new(1.0);
        let first = registry.add_target(Target::parse(":8311").unwrap(), 0.0);
        let second = registry.add_target(Target::parse(":8312").unwrap(), 0.0);
        registry.apply(first, without_metrics("a"), 1.0);
        assert_eq!(
            registry.apply(second, without_metrics("a"), 1.0),
            [Followup::Stop(second)]
        );
        // A restart during a metrics outage is counted once.
        registry.apply(first, answer("a", 5.0, true), 2.0);
        registry.apply(first, without_metrics("b"), 3.0);
        registry.apply(first, answer("b", 1.0, true), 4.0);
        let row = registry.get(first).unwrap();
        assert_eq!((row.restarts, row.stats.len()), (1, 1));
        // Two restarts in a row while /metrics fails are both counted.
        registry.apply(first, without_metrics("c"), 5.0);
        registry.apply(first, without_metrics("d"), 6.0);
        assert_eq!(registry.get(first).unwrap().restarts, 3);
    }

    #[test]
    fn a_lost_row_shows_its_aliases_again() {
        let mut registry = Registry::new(1.0);
        let target = registry.add_target(Target::parse(":8311").unwrap(), 0.0);
        registry.reconcile(vec![process(200, &["127.0.0.1:8311"])], 0.0);
        registry.apply(2, answer("n1", 5.0, true), 1.0);
        registry.apply(target, answer("n1", 5.0, true), 1.0);
        assert_eq!(registry.visible().len(), 1);
        // The target's path fails; the process row may still reach the
        // server, so it is polled on its own again.
        let followups = registry.apply(target, Poll::Failed("timed out".into()), 2.0);
        assert!(followups.contains(&Followup::Poll(2)), "{followups:?}");
        assert_eq!(registry.visible().len(), 2);
        assert!(registry.get(target).unwrap().aliases.is_empty());
    }

    #[test]
    fn a_container_that_never_answers_has_its_process_list_reread() {
        let mut registry = Registry::new(1.0);
        registry.reconcile(vec![container("aaa", "127.0.0.1:8600")], 0.0);
        let key = Key::Container {
            daemon: PathBuf::from("/var/run/docker.sock"),
            id: "aaa".to_owned(),
        };
        let refused = || Poll::Failed("connection refused".into());
        assert_eq!(
            registry.apply(1, refused(), 5.0),
            [Followup::Invalidate(key.clone())]
        );
        assert!(registry.apply(1, refused(), 30.0).is_empty());
        assert_eq!(
            registry.apply(1, refused(), 65.0),
            [Followup::Invalidate(key)]
        );
        assert_eq!(registry.get(1).unwrap().state(65.0), State::Starting);
    }

    #[test]
    fn states_follow_health_gauges_and_errors() {
        let mut registry = Registry::new(1.0);
        let id = registry.add_target(Target::parse(":9").unwrap(), 0.0);
        registry.apply(id, Poll::Failed("connection refused".into()), 0.0);
        assert_eq!(
            registry.get(id).unwrap().state(0.0),
            State::Unreachable("connection refused".into())
        );
        assert!(
            registry
                .events
                .back()
                .unwrap()
                .what
                .contains("retrying every 5s")
        );
        registry.apply(id, answer("n", 1.0, false), 1.0);
        assert!(matches!(
            registry.get(id).unwrap().state(1.0),
            State::NotReady(_)
        ));
        assert_eq!(registry.events.back().unwrap().what, "not ready: not_ready");
        let mut idle = answer("n", 2.0, true);
        if let Poll::Answer(answer) = &mut idle
            && let Some(snapshot) = &mut answer.snapshot
        {
            snapshot.gauges.running = Some(0.0);
        }
        registry.apply(id, idle, 2.0);
        assert_eq!(registry.get(id).unwrap().state(2.0), State::Idle);
        assert_eq!(registry.events.back().unwrap().what, "ready");
        let mut poisoned = answer("n", 3.0, true);
        if let Poll::Answer(answer) = &mut poisoned
            && let Some(snapshot) = &mut answer.snapshot
        {
            snapshot.gauges.poisoned = Some(true);
        }
        registry.apply(id, poisoned, 3.0);
        assert_eq!(registry.get(id).unwrap().state(3.0), State::Poisoned);
        registry.apply(
            id,
            Poll::NotChopinfer {
                endpoint: "x".into(),
            },
            4.0,
        );
        assert_eq!(registry.get(id).unwrap().state(4.0), State::NotChopinfer);
    }
}
