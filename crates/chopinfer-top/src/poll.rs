//! Polling one instance: `/health`, then `/metrics`, every interval.

use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender};
use std::thread;
use std::time::{Duration, Instant};

use crate::health::{Health, NONCE_HEADER};
use crate::http::{self, Endpoint, HEALTH_LIMIT, METRICS_LIMIT};
use crate::prometheus::Exposition;
use crate::stats::Snapshot;

/// Seconds between attempts while an instance does not answer.
pub const BACKOFF_SECONDS: f64 = 5.0;

/// A chopinfer server's answer.
#[derive(Clone, Debug)]
pub struct Answer {
    pub endpoint: String,
    pub nonce: String,
    pub health: Option<Health>,
    /// `None` when `/metrics` could not be read; `problem` says why.
    pub snapshot: Option<Snapshot>,
    pub problem: Option<String>,
}

/// The outcome of one poll.
#[derive(Clone, Debug)]
pub enum Poll {
    /// A chopinfer server answered.
    Answer(Box<Answer>),
    /// Something answered without the chopinfer nonce header.
    NotChopinfer { endpoint: String },
    /// Nothing answered.
    Failed(String),
}

impl Poll {
    pub fn answered(&self) -> bool {
        matches!(self, Self::Answer(_))
    }
}

/// Polls the first endpoint that answers, starting with `preferred`, which
/// is updated to the one that did. `seconds` is the client clock reading
/// stored with the snapshot.
pub fn poll(
    endpoints: &[Endpoint],
    preferred: &mut usize,
    bearer: Option<&str>,
    seconds: impl Fn() -> f64,
) -> Poll {
    if endpoints.is_empty() {
        return Poll::Failed("no address yet".to_owned());
    }
    let start = (*preferred).min(endpoints.len() - 1);
    let mut outcome = Poll::Failed("no address answered".to_owned());
    for offset in 0..endpoints.len() {
        let index = (start + offset) % endpoints.len();
        let endpoint = &endpoints[index];
        let shown = endpoint.authority();
        let response = match http::get(endpoint, "/health", bearer, HEALTH_LIMIT) {
            Ok(response) => response,
            Err(error) => {
                if matches!(outcome, Poll::Failed(_)) {
                    outcome = Poll::Failed(format!("{shown}: {error}"));
                }
                continue;
            }
        };
        let Some(nonce) = response.header(NONCE_HEADER).map(str::to_owned) else {
            outcome = Poll::NotChopinfer { endpoint: shown };
            continue;
        };
        *preferred = index;
        let health = Health::parse(&response.body);
        // Every chopinfer response carries the nonce, so a different one
        // means another process answered `/metrics` (a restart or proxy
        // cutover between the two requests); its counters are not this one's.
        let (snapshot, problem) = match http::get(endpoint, "/metrics", bearer, METRICS_LIMIT) {
            Ok(metrics) if metrics.header(NONCE_HEADER) != Some(nonce.as_str()) => (
                None,
                Some("/metrics came from another process; retrying".to_owned()),
            ),
            Ok(metrics) if metrics.status == 200 => {
                let text = String::from_utf8_lossy(&metrics.body);
                (
                    Some(Snapshot::from_exposition(
                        &Exposition::parse(&text),
                        seconds(),
                    )),
                    None,
                )
            }
            Ok(metrics) => (
                None,
                Some(format!("/metrics answered HTTP {}", metrics.status)),
            ),
            Err(error) => (None, Some(format!("/metrics: {error}"))),
        };
        return Poll::Answer(Box::new(Answer {
            endpoint: shown,
            nonce,
            health,
            snapshot,
            problem,
        }));
    }
    outcome
}

/// What a poller is told.
#[derive(Debug)]
pub enum Control {
    /// Poll these endpoints from now on, starting at once.
    Endpoints(Vec<Endpoint>),
    Stop,
}

/// A poller thread and its control channel. Dropping it stops the thread.
#[derive(Debug)]
pub struct Poller {
    control: Sender<Control>,
}

impl Poller {
    /// Polls `endpoints` every `interval` seconds (five while they do not
    /// answer) and sends each result with `id` to `results`.
    pub fn spawn(
        id: u64,
        endpoints: Vec<Endpoint>,
        bearer: Option<String>,
        interval: f64,
        origin: Instant,
        results: Sender<(u64, Poll)>,
    ) -> Self {
        let (control, commands) = mpsc::channel();
        thread::spawn(move || run(id, endpoints, bearer, interval, origin, &commands, &results));
        Self { control }
    }

    pub fn set_endpoints(&self, endpoints: Vec<Endpoint>) {
        let _ = self.control.send(Control::Endpoints(endpoints));
    }
}

impl Drop for Poller {
    fn drop(&mut self) {
        let _ = self.control.send(Control::Stop);
    }
}

fn run(
    id: u64,
    mut endpoints: Vec<Endpoint>,
    bearer: Option<String>,
    interval: f64,
    origin: Instant,
    commands: &Receiver<Control>,
    results: &Sender<(u64, Poll)>,
) {
    let mut preferred = 0;
    loop {
        let wait = if endpoints.is_empty() {
            // Nothing to poll until an address is known.
            Duration::from_secs(3600)
        } else {
            let result = poll(&endpoints, &mut preferred, bearer.as_deref(), || {
                origin.elapsed().as_secs_f64()
            });
            let wait = if result.answered() {
                interval
            } else {
                BACKOFF_SECONDS.max(interval)
            };
            if results.send((id, result)).is_err() {
                return;
            }
            Duration::from_secs_f64(wait)
        };
        match commands.recv_timeout(wait) {
            Ok(Control::Endpoints(new)) => {
                endpoints = new;
                preferred = 0;
            }
            Ok(Control::Stop) | Err(RecvTimeoutError::Disconnected) => return,
            Err(RecvTimeoutError::Timeout) => {}
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::io::{Read, Write};
    use std::net::{SocketAddr, TcpListener};
    use std::sync::{Arc, Mutex};

    /// A fake chopinfer server answering `/health` and `/metrics` from
    /// shared state until dropped.
    pub(crate) struct FakeServer {
        pub address: SocketAddr,
        pub state: Arc<Mutex<FakeServerState>>,
    }

    pub(crate) struct FakeServerState {
        pub nonce: Option<String>,
        /// The nonce `/metrics` answers with when it differs from `nonce`.
        pub metrics_nonce: Option<String>,
        pub health: String,
        pub metrics: String,
    }

    impl FakeServer {
        pub(crate) fn start(nonce: Option<&str>, health: &str, metrics: &str) -> Self {
            let listener = TcpListener::bind("127.0.0.1:0").unwrap();
            let address = listener.local_addr().unwrap();
            let state = Arc::new(Mutex::new(FakeServerState {
                nonce: nonce.map(str::to_owned),
                metrics_nonce: None,
                health: health.to_owned(),
                metrics: metrics.to_owned(),
            }));
            let shared = Arc::clone(&state);
            thread::spawn(move || {
                for socket in listener.incoming() {
                    let Ok(mut socket) = socket else { return };
                    let mut head = Vec::new();
                    let mut byte = [0u8; 1];
                    while !head.ends_with(b"\r\n\r\n") {
                        if socket.read(&mut byte).unwrap_or(0) == 0 {
                            break;
                        }
                        head.push(byte[0]);
                    }
                    let head = String::from_utf8_lossy(&head).into_owned();
                    let path = head.split_whitespace().nth(1).unwrap_or("/").to_owned();
                    let state = shared.lock().unwrap();
                    let (body, nonce) = if path == "/metrics" {
                        (
                            &state.metrics,
                            state.metrics_nonce.as_ref().or(state.nonce.as_ref()),
                        )
                    } else {
                        (&state.health, state.nonce.as_ref())
                    };
                    let nonce = nonce
                        .map(|nonce| format!("{NONCE_HEADER}: {nonce}\r\n"))
                        .unwrap_or_default();
                    let reply = format!(
                        "HTTP/1.1 200 OK\r\n{nonce}Content-Length: {}\r\n\r\n{body}",
                        body.len()
                    );
                    let _ = socket.write_all(reply.as_bytes());
                }
            });
            Self { address, state }
        }
    }

    #[test]
    fn polls_health_then_metrics_and_skips_dead_endpoints() {
        let server = FakeServer::start(
            Some("n1"),
            r#"{"status":"ready","serving":{"served_model_names":["m"]}}"#,
            "vllm:generation_tokens_total 7\n",
        );
        let dead = TcpListener::bind("127.0.0.1:0")
            .unwrap()
            .local_addr()
            .unwrap();
        let endpoints = vec![Endpoint::Address(dead), Endpoint::Address(server.address)];
        let mut preferred = 0;
        match poll(&endpoints, &mut preferred, None, || 1.5) {
            Poll::Answer(answer) => {
                let Answer {
                    nonce,
                    health,
                    snapshot,
                    endpoint,
                    problem,
                } = *answer;
                assert_eq!(nonce, "n1");
                assert_eq!(endpoint, server.address.to_string());
                assert_eq!(health.unwrap().model_name(), Some("m"));
                let snapshot = snapshot.unwrap();
                assert_eq!(snapshot.counters.generation_tokens, 7.0);
                assert_eq!(snapshot.client_seconds, 1.5);
                assert_eq!(problem, None);
            }
            other => panic!("{other:?}"),
        }
        assert_eq!(preferred, 1);
    }

    #[test]
    fn a_server_without_the_nonce_header_is_not_chopinfer() {
        let other = FakeServer::start(None, "{}", "");
        let mut preferred = 0;
        assert!(matches!(
            poll(
                &[Endpoint::Address(other.address)],
                &mut preferred,
                None,
                || 0.0
            ),
            Poll::NotChopinfer { .. }
        ));
        assert!(matches!(
            poll(&[], &mut preferred, None, || 0.0),
            Poll::Failed(_)
        ));
    }

    #[test]
    fn metrics_from_another_process_are_dropped() {
        let server = FakeServer::start(
            Some("a"),
            r#"{"status":"ready"}"#,
            "vllm:generation_tokens_total 7\n",
        );
        server.state.lock().unwrap().metrics_nonce = Some("b".to_owned());
        let mut preferred = 0;
        match poll(
            &[Endpoint::Address(server.address)],
            &mut preferred,
            None,
            || 0.0,
        ) {
            Poll::Answer(answer) => {
                assert_eq!(answer.nonce, "a");
                assert!(answer.snapshot.is_none());
                assert!(answer.problem.unwrap().contains("another process"));
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn a_poller_reports_on_its_interval_and_follows_new_endpoints() {
        let first = FakeServer::start(Some("a"), r#"{"status":"ready"}"#, "");
        let second = FakeServer::start(Some("b"), r#"{"status":"ready"}"#, "");
        let (sender, results) = mpsc::channel();
        let poller = Poller::spawn(
            9,
            vec![Endpoint::Address(first.address)],
            None,
            0.2,
            Instant::now(),
            sender,
        );
        let nonce = |poll: Poll| match poll {
            Poll::Answer(answer) => answer.nonce,
            other => panic!("{other:?}"),
        };
        let (id, result) = results.recv_timeout(Duration::from_secs(5)).unwrap();
        assert_eq!((id, nonce(result)), (9, "a".to_owned()));
        // The server restarts: the next polls carry its new nonce.
        first.state.lock().unwrap().nonce = Some("a2".to_owned());
        let mut seen = String::new();
        for _ in 0..3 {
            seen = nonce(results.recv_timeout(Duration::from_secs(5)).unwrap().1);
            if seen == "a2" {
                break;
            }
        }
        assert_eq!(seen, "a2");
        poller.set_endpoints(vec![Endpoint::Address(second.address)]);
        for _ in 0..3 {
            seen = nonce(results.recv_timeout(Duration::from_secs(5)).unwrap().1);
            if seen == "b" {
                break;
            }
        }
        assert_eq!(seen, "b");
        drop(poller);
        // The thread ends; at most one result was already on its way.
        let mut after = 0;
        while results.recv_timeout(Duration::from_millis(600)).is_ok() {
            after += 1;
        }
        assert!(after <= 1, "{after}");
    }
}
