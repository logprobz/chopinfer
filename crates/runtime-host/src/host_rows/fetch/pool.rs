use std::fs::File;
use std::os::unix::fs::FileExt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock, mpsc};

/// Reader threads. One read keeps at most `super::readers()` of its rows
/// in flight, so the window can change between measurement cohorts.
pub(super) const WORKERS: usize = 128;
const QUEUE: usize = 128;
type Reply = (usize, Result<Vec<u8>, String>);

struct Work {
    source: Arc<File>,
    offset: u64,
    length: usize,
    index: usize,
    reply: mpsc::Sender<Reply>,
}

struct Pool {
    send: mpsc::SyncSender<Work>,
}
enum Failure {
    Queue(String),
    Source(String),
}

pub(super) fn fallback(reason: &str) {
    static REPORTED: AtomicBool = AtomicBool::new(false);
    if !REPORTED.swap(true, Ordering::Relaxed) {
        runtime_engine::logging::line(
            serde_json::json!({
                "schema":"chopinfer.mapped-row-fetch.v1","phase":"fallback","reason":reason,
            })
            .to_string(),
        );
    }
}

fn read_one(source: &File, offset: u64, length: usize) -> Result<Vec<u8>, String> {
    let mut bytes = vec![0; length];
    source
        .read_exact_at(&mut bytes, offset)
        .map_err(|error| format!("read mapped row source: {error}"))?;
    Ok(bytes)
}

impl Pool {
    fn new() -> Result<Self, String> {
        let (send, receive) = mpsc::sync_channel::<Work>(QUEUE);
        let receive = Arc::new(Mutex::new(receive));
        for _ in 0..WORKERS {
            let receive = Arc::clone(&receive);
            std::thread::Builder::new()
                .name("mapped-row-read".into())
                .spawn(move || {
                    loop {
                        let work = match receive.lock() {
                            Ok(receiver) => receiver.recv(),
                            Err(_) => break,
                        };
                        let Ok(work) = work else {
                            break;
                        };
                        let result = read_one(&work.source, work.offset, work.length);
                        let _ = work.reply.send((work.index, result));
                    }
                })
                .map_err(|error| format!("start mapped row reader: {error}"))?;
        }
        Ok(Self { send })
    }

    fn read(
        &self,
        source: &Arc<File>,
        requests: &[(u64, usize)],
        window: usize,
    ) -> Result<Vec<Vec<u8>>, Failure> {
        let (reply, receive) = mpsc::channel();
        let mut reply = Some(reply);
        let mut pending = requests.iter().copied().enumerate();
        // Each reply admits the next row. The admission that finds no row
        // left drops this sender, so a lost reply ends the receive loop.
        for _ in 0..window.max(1) {
            self.admit(source, &mut pending, &mut reply)?;
        }
        let mut rows = vec![None; requests.len()];
        let mut failure = None;
        while let Ok((index, result)) = receive.recv() {
            self.admit(source, &mut pending, &mut reply)?;
            match result {
                Ok(bytes) => rows[index] = Some(bytes),
                Err(error) => {
                    if failure.is_none() {
                        failure = Some(error);
                    }
                }
            }
        }
        if let Some(error) = failure {
            return Err(Failure::Source(error));
        }
        rows.into_iter()
            .map(|row| {
                row.ok_or_else(|| Failure::Queue("mapped row read response is absent".into()))
            })
            .collect()
    }

    fn admit(
        &self,
        source: &Arc<File>,
        pending: &mut impl Iterator<Item = (usize, (u64, usize))>,
        reply: &mut Option<mpsc::Sender<Reply>>,
    ) -> Result<(), Failure> {
        let Some(sender) = reply.as_ref() else {
            return Ok(());
        };
        let Some((index, (offset, length))) = pending.next() else {
            *reply = None;
            return Ok(());
        };
        self.send
            .send(Work {
                source: Arc::clone(source),
                offset,
                length,
                index,
                reply: sender.clone(),
            })
            .map_err(|_| Failure::Queue("mapped row read queue stopped".into()))
    }
}

fn serial(source: &File, requests: &[(u64, usize)]) -> Result<Vec<Vec<u8>>, String> {
    requests
        .iter()
        .map(|&(offset, length)| read_one(source, offset, length))
        .collect()
}

pub(super) fn read(
    source: &Arc<File>,
    requests: &[(u64, usize)],
) -> Result<(Vec<Vec<u8>>, bool), String> {
    static POOL: OnceLock<Result<Pool, String>> = OnceLock::new();
    read_using(
        POOL.get_or_init(Pool::new).as_ref().map_err(String::as_str),
        source,
        requests,
    )
}

fn read_using(
    pool: Result<&Pool, &str>,
    source: &Arc<File>,
    requests: &[(u64, usize)],
) -> Result<(Vec<Vec<u8>>, bool), String> {
    match pool {
        Ok(pool) => match pool.read(source, requests, super::readers()) {
            Ok(rows) => return Ok((rows, false)),
            Err(Failure::Source(error)) => return Err(error),
            Err(Failure::Queue(error)) => fallback(&error),
        },
        Err(error) => fallback(error),
    }
    serial(source, requests).map(|rows| (rows, true))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn unavailable_or_closed_queue_uses_exact_serial_reads() {
        let path =
            std::env::temp_dir().join(format!("mapped-pool-fallback-{}", std::process::id()));
        std::fs::write(&path, [1, 9, 2, 8, 3, 7]).unwrap();
        let file = Arc::new(File::open(&path).unwrap());
        let requests = [(1, 3), (0, 2)];
        let expected = (vec![vec![9, 2, 8], vec![1, 9]], true);
        assert_eq!(
            read_using(Err("reader admission unavailable"), &file, &requests).unwrap(),
            expected
        );
        let (send, receive) = mpsc::sync_channel(1);
        drop(receive);
        assert_eq!(
            read_using(Ok(&Pool { send }), &file, &requests).unwrap(),
            expected
        );
        assert!(read_using(Err("reader admission unavailable"), &file, &[(5, 2)]).is_err());
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn every_window_returns_the_rows_in_request_order() {
        let path = std::env::temp_dir().join(format!("mapped-pool-window-{}", std::process::id()));
        let bytes = (0..4096_u32)
            .map(|i| (i * 7 + i / 255) as u8)
            .collect::<Vec<_>>();
        std::fs::write(&path, &bytes).unwrap();
        let file = Arc::new(File::open(&path).unwrap());
        let requests = (0..300_u64)
            .map(|i| ((i * 977) % 4000, 1 + (i % 9) as usize))
            .collect::<Vec<_>>();
        let expected = requests
            .iter()
            .map(|&(offset, length)| bytes[offset as usize..][..length].to_vec())
            .collect::<Vec<_>>();
        let pool = Pool::new().unwrap();
        for window in [1, 2, 7, 64, WORKERS, 1000] {
            assert!(
                pool.read(&file, &requests, window).ok() == Some(expected.clone()),
                "window {window}"
            );
            assert!(pool.read(&file, &requests[..1], window).ok() == Some(expected[..1].to_vec()));
        }
        assert!(matches!(
            pool.read(&file, &[(4095, 2)], 4),
            Err(Failure::Source(_))
        ));
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn a_read_keeps_at_most_its_window_in_flight() {
        let path = std::env::temp_dir().join(format!("mapped-pool-flight-{}", std::process::id()));
        std::fs::write(&path, (0..64_u8).collect::<Vec<_>>()).unwrap();
        let file = Arc::new(File::open(&path).unwrap());
        let (send, receive) = mpsc::sync_channel(QUEUE);
        let requests = (0..10_u64).map(|i| (i * 5, 3)).collect::<Vec<_>>();
        let reader = {
            let (file, requests) = (Arc::clone(&file), requests.clone());
            std::thread::spawn(move || Pool { send }.read(&file, &requests, 3).ok())
        };
        // This thread is the only worker: three rows arrive, and each reply
        // admits exactly one more.
        let wait = std::time::Duration::from_millis(200);
        let mut queued = (0..3)
            .map(|_| receive.recv_timeout(wait).unwrap())
            .collect::<std::collections::VecDeque<_>>();
        while let Some(work) = queued.pop_front() {
            assert!(receive.recv_timeout(wait / 4).is_err());
            let result = read_one(&work.source, work.offset, work.length);
            work.reply.send((work.index, result)).unwrap();
            drop(work);
            if let Ok(next) = receive.recv_timeout(wait) {
                queued.push_back(next);
            }
        }
        let expected = requests
            .iter()
            .map(|&(offset, length)| (offset as u8..offset as u8 + length as u8).collect())
            .collect::<Vec<Vec<u8>>>();
        assert_eq!(reader.join().unwrap(), Some(expected));
        std::fs::remove_file(path).unwrap();
    }
}
