use crate::page_cache::{CacheAdviceV1, advise_v1, read_cached_v1};
use std::fs::File;
use std::os::unix::fs::FileExt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock, mpsc};

/// Reader threads. One read hands its rows to at most `super::readers()` of
/// them, so the window can change between measurement cohorts.
pub(super) const WORKERS: usize = 128;
const QUEUE: usize = 128;
type Reply = (usize, Result<Vec<Vec<u8>>, String>);

/// Consecutive requests of one read for one reader thread, and how it reads
/// them ([`read_share`]).
struct Work {
    source: Arc<File>,
    requests: Vec<(u64, usize)>,
    start: bool,
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
    read_exact(source, &mut bytes, offset)?;
    Ok(bytes)
}

fn read_exact(source: &File, bytes: &mut [u8], offset: u64) -> Result<(), String> {
    source
        .read_exact_at(bytes, offset)
        .map_err(|error| format!("read mapped row source: {error}"))
}

/// Reads one share of a read's requests, in request order. With `start`,
/// rows the page cache holds are copied without waiting, every other row's
/// pages are then asked for (`POSIX_FADV_WILLNEED`, which starts their device
/// reads and returns), and only then does the share wait for them: all its
/// cold rows are in flight at once, so a batch waits about one device
/// latency for them, not one for every `readers()` rows. Without `start`,
/// each row is one blocking read.
/// Either way every row's bytes end as one exact read of its file range; a
/// cached copy that came up short is read again whole.
fn read_share(
    source: &File,
    requests: &[(u64, usize)],
    start: bool,
) -> Result<Vec<Vec<u8>>, String> {
    if !start {
        return requests
            .iter()
            .map(|&(offset, length)| read_one(source, offset, length))
            .collect();
    }
    let mut rows = Vec::with_capacity(requests.len());
    let mut cold = Vec::new();
    for (slot, &(offset, length)) in requests.iter().enumerate() {
        let mut bytes = vec![0; length];
        if !matches!(read_cached_v1(source, &mut bytes, offset), Ok(read) if read == length) {
            cold.push(slot);
        }
        rows.push(bytes);
    }
    super::count_cold_rows(requests.len() - cold.len(), cold.len());
    for &slot in &cold {
        let (offset, length) = requests[slot];
        // A hint only: the read below fetches the row if the advice fails.
        // A zero length would advise the whole file.
        if length > 0 {
            let _ = advise_v1(source, offset, length as u64, CacheAdviceV1::WillNeed);
        }
    }
    for &slot in &cold {
        read_exact(source, &mut rows[slot], requests[slot].0)?;
    }
    Ok(rows)
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
                        let result = read_share(&work.source, &work.requests, work.start);
                        let _ = work.reply.send((work.index, result));
                    }
                })
                .map_err(|error| format!("start mapped row reader: {error}"))?;
        }
        Ok(Self { send })
    }

    /// Started reads split the requests into at most `window` consecutive
    /// shares, all admitted at once. Blocking reads make each row its own
    /// share and keep at most `window` of them in flight.
    fn read(
        &self,
        source: &Arc<File>,
        requests: &[(u64, usize)],
        window: usize,
        start: bool,
    ) -> Result<Vec<Vec<u8>>, Failure> {
        let window = window.max(1);
        let share = if start {
            requests.len().div_ceil(window).max(1)
        } else {
            1
        };
        let (reply, receive) = mpsc::channel();
        let mut reply = Some(reply);
        let mut pending = requests.chunks(share).enumerate();
        // Each reply admits the next share. The admission that finds no
        // share left drops this sender, so a lost reply ends the receive loop.
        for _ in 0..window {
            self.admit(source, &mut pending, start, &mut reply)?;
        }
        let mut shares = vec![None; requests.len().div_ceil(share)];
        let mut failure = None;
        while let Ok((index, result)) = receive.recv() {
            self.admit(source, &mut pending, start, &mut reply)?;
            match result {
                Ok(rows) => shares[index] = Some(rows),
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
        let mut rows = Vec::with_capacity(requests.len());
        for share in shares {
            rows.extend(
                share.ok_or_else(|| Failure::Queue("mapped row read response is absent".into()))?,
            );
        }
        Ok(rows)
    }

    fn admit<'a>(
        &self,
        source: &Arc<File>,
        pending: &mut impl Iterator<Item = (usize, &'a [(u64, usize)])>,
        start: bool,
        reply: &mut Option<mpsc::Sender<Reply>>,
    ) -> Result<(), Failure> {
        let Some(sender) = reply.as_ref() else {
            return Ok(());
        };
        let Some((index, requests)) = pending.next() else {
            *reply = None;
            return Ok(());
        };
        self.send
            .send(Work {
                source: Arc::clone(source),
                requests: requests.to_vec(),
                start,
                index,
                reply: sender.clone(),
            })
            .map_err(|_| Failure::Queue("mapped row read queue stopped".into()))
    }
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
        super::start_reads(),
    )
}

fn read_using(
    pool: Result<&Pool, &str>,
    source: &Arc<File>,
    requests: &[(u64, usize)],
    start: bool,
) -> Result<(Vec<Vec<u8>>, bool), String> {
    match pool {
        Ok(pool) => match pool.read(source, requests, super::readers(), start) {
            Ok(rows) => return Ok((rows, false)),
            Err(Failure::Source(error)) => return Err(error),
            Err(Failure::Queue(error)) => fallback(&error),
        },
        Err(error) => fallback(error),
    }
    // One share on this thread: started reads still keep its cold rows in
    // flight together.
    read_share(source, requests, start).map(|rows| (rows, true))
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
        for start in [false, true] {
            assert_eq!(
                read_using(Err("reader admission unavailable"), &file, &requests, start).unwrap(),
                expected
            );
            let (send, receive) = mpsc::sync_channel(1);
            drop(receive);
            assert_eq!(
                read_using(Ok(&Pool { send }), &file, &requests, start).unwrap(),
                expected
            );
            assert!(
                read_using(Err("reader admission unavailable"), &file, &[(5, 2)], start).is_err()
            );
        }
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
        for start in [false, true] {
            for window in [1, 2, 7, 64, WORKERS, 1000] {
                assert!(
                    pool.read(&file, &requests, window, start).ok() == Some(expected.clone()),
                    "window {window} start {start}"
                );
                assert!(
                    pool.read(&file, &requests[..1], window, start).ok()
                        == Some(expected[..1].to_vec())
                );
                assert!(pool.read(&file, &[], window, start).ok() == Some(Vec::new()));
            }
            assert!(matches!(
                pool.read(&file, &[(0, 2), (4095, 2)], 4, start),
                Err(Failure::Source(_))
            ));
        }
        std::fs::remove_file(path).unwrap();
    }

    fn expected_bytes(requests: &[(u64, usize)]) -> Vec<Vec<u8>> {
        requests
            .iter()
            .map(|&(offset, length)| (offset as u8..offset as u8 + length as u8).collect())
            .collect()
    }

    #[test]
    fn a_blocking_read_keeps_at_most_its_window_in_flight() {
        let path = std::env::temp_dir().join(format!("mapped-pool-flight-{}", std::process::id()));
        std::fs::write(&path, (0..64_u8).collect::<Vec<_>>()).unwrap();
        let file = Arc::new(File::open(&path).unwrap());
        let (send, receive) = mpsc::sync_channel(QUEUE);
        let requests = (0..10_u64).map(|i| (i * 5, 3)).collect::<Vec<_>>();
        let reader = {
            let (file, requests) = (Arc::clone(&file), requests.clone());
            std::thread::spawn(move || Pool { send }.read(&file, &requests, 3, false).ok())
        };
        // This thread is the only worker: three rows arrive, and each reply
        // admits exactly one more.
        let wait = std::time::Duration::from_millis(200);
        let mut queued = (0..3)
            .map(|_| receive.recv_timeout(wait).unwrap())
            .collect::<std::collections::VecDeque<_>>();
        while let Some(work) = queued.pop_front() {
            assert!(receive.recv_timeout(wait / 4).is_err());
            assert_eq!(work.requests.len(), 1);
            assert!(!work.start);
            let result = read_share(&work.source, &work.requests, work.start);
            work.reply.send((work.index, result)).unwrap();
            drop(work);
            if let Ok(next) = receive.recv_timeout(wait) {
                queued.push_back(next);
            }
        }
        assert_eq!(reader.join().unwrap(), Some(expected_bytes(&requests)));
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn a_started_read_admits_every_share_before_any_reply() {
        let path = std::env::temp_dir().join(format!("mapped-pool-shares-{}", std::process::id()));
        std::fs::write(&path, (0..64_u8).collect::<Vec<_>>()).unwrap();
        let file = Arc::new(File::open(&path).unwrap());
        let (send, receive) = mpsc::sync_channel(QUEUE);
        let requests = (0..10_u64).map(|i| (i * 5, 3)).collect::<Vec<_>>();
        let reader = {
            let (file, requests) = (Arc::clone(&file), requests.clone());
            std::thread::spawn(move || Pool { send }.read(&file, &requests, 3, true).ok())
        };
        // Three consecutive shares (4, 4 and 2 rows) arrive with no reply
        // sent, and nothing else does: every row is handed out at once.
        let wait = std::time::Duration::from_millis(200);
        let shares = (0..3)
            .map(|_| receive.recv_timeout(wait).unwrap())
            .collect::<Vec<_>>();
        assert!(receive.recv_timeout(wait / 4).is_err());
        assert_eq!(
            shares
                .iter()
                .map(|w| w.requests.clone())
                .collect::<Vec<_>>(),
            [&requests[..4], &requests[4..8], &requests[8..]].map(<[_]>::to_vec)
        );
        // Replies in reverse order still assemble in request order.
        for work in shares.into_iter().rev() {
            assert!(work.start);
            let result = read_share(&work.source, &work.requests, work.start);
            work.reply.send((work.index, result)).unwrap();
        }
        assert_eq!(reader.join().unwrap(), Some(expected_bytes(&requests)));
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn a_started_row_whose_later_page_is_uncached_is_read_again_whole() {
        let page = crate::host_rows::page_bytes_v1();
        let path = std::env::temp_dir().join(format!("mapped-pool-short-{}", std::process::id()));
        let bytes = (0..3 * page).map(|i| (i % 251) as u8).collect::<Vec<_>>();
        std::fs::write(&path, &bytes).unwrap();
        let file = File::open(&path).unwrap();
        // Evict the file, then cache only its first page (advice reads just
        // that page, and the read waits for it without reading ahead): a row
        // that starts near that page's end continues into an uncached one.
        file.sync_all().unwrap();
        advise_v1(&file, 0, 0, CacheAdviceV1::DontNeed).unwrap();
        advise_v1(&file, 0, page as u64, CacheAdviceV1::WillNeed).unwrap();
        let mut first = [0];
        file.read_exact_at(&mut first, 0).unwrap();
        let row = ((page - 100) as u64, 300);
        let mut probe = vec![0; row.1];
        if let (Ok(read), Ok((0, _))) = (
            read_cached_v1(&file, &mut probe, row.0),
            crate::page_cache::cached_pages_v1(&file, page as u64, 1),
        ) {
            // Where the file system evicted the second page, the cached copy
            // stops at it.
            assert_eq!(read, 100);
        } else {
            eprintln!("skipped short copy: no eviction or residency report on this file system");
        }
        let rows = read_share(&file, &[row, (0, 8), row], true).unwrap();
        let exact = |(offset, length): (u64, usize)| bytes[offset as usize..][..length].to_vec();
        assert_eq!(rows, [exact(row), exact((0, 8)), exact(row)]);
        std::fs::remove_file(path).unwrap();
    }
}
