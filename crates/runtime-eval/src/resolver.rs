//! Bounded hostname lookup for the blocking eval client. One process-wide
//! worker and one queued lookup cap resources even if NSS never returns.
//! Callers stop waiting at their deadline; expired queued work is discarded.

use std::io;
use std::net::{SocketAddr, ToSocketAddrs};
use std::sync::{OnceLock, mpsc};
use std::time::{Duration, Instant};

type Answer = io::Result<SocketAddr>;
struct Lookup {
    host: String,
    port: u16,
    deadline: Instant,
    reply: mpsc::SyncSender<Answer>,
}

#[derive(Debug)]
pub(crate) struct Resolver(mpsc::SyncSender<Lookup>);

impl Resolver {
    fn start(resolve: impl Fn(String, u16) -> Answer + Send + 'static) -> io::Result<Self> {
        let (send, receive) = mpsc::sync_channel::<Lookup>(1);
        std::thread::Builder::new()
            .name("eval-resolver".into())
            .spawn(move || {
                while let Ok(lookup) = receive.recv() {
                    if Instant::now() < lookup.deadline {
                        let answer = resolve(lookup.host, lookup.port);
                        let _ = lookup.reply.send(answer);
                    }
                }
            })?;
        Ok(Self(send))
    }

    pub(crate) fn shared() -> io::Result<&'static Self> {
        static RESOLVER: OnceLock<Result<Resolver, String>> = OnceLock::new();
        RESOLVER
            .get_or_init(|| {
                Self::start(|host, port| {
                    (host.as_str(), port)
                        .to_socket_addrs()?
                        .next()
                        .ok_or_else(|| io::Error::other("hostname has no address"))
                })
                .map_err(|error| error.to_string())
            })
            .as_ref()
            .map_err(|error| io::Error::other(error.clone()))
    }

    pub(crate) fn resolve(&self, host: &str, port: u16, timeout: Duration) -> Answer {
        let deadline = Instant::now()
            .checked_add(timeout)
            .ok_or_else(|| io::Error::other("resolver timeout is out of range"))?;
        let (reply, receive) = mpsc::sync_channel(1);
        self.0
            .try_send(Lookup {
                host: host.into(),
                port,
                deadline,
                reply,
            })
            .map_err(|error| io::Error::other(format!("resolver unavailable: {error}")))?;
        receive
            .recv_timeout(deadline.saturating_duration_since(Instant::now()))
            .map_err(|_| io::Error::new(io::ErrorKind::TimedOut, "hostname resolution timed out"))?
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn a_stalled_lookup_cannot_exceed_the_wait_budget_or_spawn_more_workers() {
        let calls = Arc::new(AtomicUsize::new(0));
        let counted = calls.clone();
        let (release, blocked) = mpsc::channel();
        let (finished, done) = mpsc::channel();
        let resolver = Resolver::start(move |_, _| {
            counted.fetch_add(1, Ordering::SeqCst);
            blocked.recv().unwrap();
            finished.send(()).unwrap();
            Ok("127.0.0.1:1".parse().unwrap())
        })
        .unwrap();
        let started = Instant::now();
        let error = resolver
            .resolve("slow.test", 1, Duration::from_millis(100))
            .unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::TimedOut);
        assert!(started.elapsed() < Duration::from_millis(400));
        // The second lookup expires in the only queue slot. A third cannot
        // add a worker or queued work while the first remains blocked.
        assert!(
            resolver
                .resolve("queued.test", 1, Duration::from_millis(20))
                .is_err()
        );
        assert!(
            resolver
                .resolve("overflow.test", 1, Duration::from_secs(1))
                .unwrap_err()
                .to_string()
                .contains("resolver unavailable")
        );
        release.send(()).unwrap();
        done.recv_timeout(Duration::from_secs(1)).unwrap();
        drop(resolver);
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }
}
