use std::fs::File;
use std::os::unix::fs::FileExt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock, mpsc};

pub(super) const WORKERS: usize = 16;
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

    fn read(&self, source: &Arc<File>, requests: &[(u64, usize)]) -> Result<Vec<Vec<u8>>, Failure> {
        let (reply, receive) = mpsc::channel();
        for (index, &(offset, length)) in requests.iter().enumerate() {
            self.send
                .send(Work {
                    source: Arc::clone(source),
                    offset,
                    length,
                    index,
                    reply: reply.clone(),
                })
                .map_err(|_| Failure::Queue("mapped row read queue stopped".into()))?;
        }
        drop(reply);
        let mut rows = vec![None; requests.len()];
        let mut failure = None;
        for (index, result) in receive {
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
        Ok(pool) => match pool.read(source, requests) {
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
}
