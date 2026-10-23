use crate::page_cache::{CacheAdviceV1, advise_v1, read_cached_v1};
use std::fs::File;
use std::os::unix::fs::FileExt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock, mpsc};

/// Reader threads. One read hands its rows to at most `super::readers()` of
/// them, so the window can change between measurement cohorts.
pub(super) const WORKERS: usize = 128;
const QUEUE: usize = 128;
type Reply = (usize, Result<Vec<u8>, String>);

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

fn read_exact(source: &File, bytes: &mut [u8], offset: u64) -> Result<(), String> {
    source
        .read_exact_at(bytes, offset)
        .map_err(|error| format!("read mapped row source: {error}"))
}

/// Reads one share of a read's requests, in request order, end to end in
/// one buffer. With `start`, rows the page cache holds are copied without
/// waiting, every other row's pages are then asked for
/// (`POSIX_FADV_WILLNEED`, which starts their device reads and returns), and
/// only then does the share wait for them: all its cold rows are in flight
/// at once, so a batch waits about one device latency for them, not one for
/// every `readers()` rows. Without `start`, each row is one blocking read.
/// Either way every row's bytes end as one exact read of its file range; a
/// cached copy that came up short is read again whole.
fn read_share(source: &File, requests: &[(u64, usize)], start: bool) -> Result<Vec<u8>, String> {
    let mut bytes = vec![0; requests.iter().map(|&(_, length)| length).sum()];
    let mut at = 0;
    if !start {
        for &(offset, length) in requests {
            read_exact(source, &mut bytes[at..at + length], offset)?;
            at += length;
        }
        return Ok(bytes);
    }
    // Each cold row's place in `bytes` and its file range.
    let mut cold = Vec::new();
    for &(offset, length) in requests {
        let row = &mut bytes[at..at + length];
        if !matches!(read_cached_v1(source, row, offset), Ok(read) if read == length) {
            cold.push((at, offset, length));
        }
        at += length;
    }
    super::count_cold_rows(requests.len() - cold.len(), cold.len());
    for &(_, offset, length) in &cold {
        // A hint only: the read below fetches the row if the advice fails.
        // A zero length would advise the whole file.
        if length > 0 {
            let _ = advise_v1(source, offset, length as u64, CacheAdviceV1::WillNeed);
        }
    }
    for &(at, offset, length) in &cold {
        read_exact(source, &mut bytes[at..at + length], offset)?;
    }
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
                        let result = read_share(&work.source, &work.requests, work.start);
                        let _ = work.reply.send((work.index, result));
                    }
                })
                .map_err(|error| format!("start mapped row reader: {error}"))?;
        }
        Ok(Self { send })
    }

    /// Reads every group's requests from its own source as one batch,
    /// group `g`'s rows end to end into `outputs[g]`, which the caller sized
    /// to them. Started reads cut all the groups' requests into consecutive
    /// shares of one size, the rows over `window` rounded up (at most
    /// `window` shares, plus one per further group), and admit them all at
    /// once, so every cold row of every group is in flight together. Blocking
    /// reads make each row its own share and keep at most `window` of them in
    /// flight. Each share is copied to its place as it arrives.
    fn read_into(
        &self,
        groups: &[Group<'_>],
        outputs: &mut [&mut [u8]],
        window: usize,
        start: bool,
    ) -> Result<(), Failure> {
        let window = window.max(1);
        let total: usize = groups.iter().map(|(_, requests)| requests.len()).sum();
        let share = if start {
            total.div_ceil(window).max(1)
        } else {
            1
        };
        // Each share's group, where its bytes start in that group's output,
        // its source and its requests.
        let mut all = Vec::new();
        for (group, &(source, requests)) in groups.iter().enumerate() {
            let mut at = 0;
            for requests in requests.chunks(share) {
                all.push((group, at, source, requests));
                at += requests.iter().map(|&(_, length)| length).sum::<usize>();
            }
        }
        let (reply, receive) = mpsc::channel();
        let mut reply = Some(reply);
        let mut pending = all
            .iter()
            .map(|&(_, _, source, requests)| (source, requests))
            .enumerate();
        // Each reply admits the next share. The admission that finds no
        // share left drops this sender, so a lost reply ends the receive loop.
        for _ in 0..if start { all.len().max(1) } else { window } {
            self.admit(&mut pending, start, &mut reply)?;
        }
        let mut arrived = vec![false; all.len()];
        let mut failure = None;
        while let Ok((index, result)) = receive.recv() {
            self.admit(&mut pending, start, &mut reply)?;
            match result {
                Ok(bytes) => {
                    let (group, at, _, _) = all[index];
                    outputs[group]
                        .get_mut(at..at + bytes.len())
                        .ok_or_else(|| {
                            Failure::Queue("mapped row read overruns its output".into())
                        })?
                        .copy_from_slice(&bytes);
                    arrived[index] = true;
                }
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
        if arrived.contains(&false) {
            return Err(Failure::Queue("mapped row read response is absent".into()));
        }
        Ok(())
    }

    fn admit<'a>(
        &self,
        pending: &mut impl Iterator<Item = (usize, (&'a Arc<File>, &'a [(u64, usize)]))>,
        start: bool,
        reply: &mut Option<mpsc::Sender<Reply>>,
    ) -> Result<(), Failure> {
        let Some(sender) = reply.as_ref() else {
            return Ok(());
        };
        let Some((index, (source, requests))) = pending.next() else {
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

/// One source's requests in a batch read.
pub(super) type Group<'a> = (&'a Arc<File>, &'a [(u64, usize)]);
/// Each group's rows end to end in request order, and whether this thread
/// read them because the reader threads were unavailable.
pub(super) type GroupRows = (Vec<Vec<u8>>, bool);

pub(super) fn read(
    source: &Arc<File>,
    requests: &[(u64, usize)],
) -> Result<(Vec<u8>, bool), String> {
    let (mut rows, serial) = read_groups(&[(source, requests)])?;
    Ok((rows.pop().unwrap_or_default(), serial))
}

/// Reads every group as one batch: one wait for all of them.
pub(super) fn read_groups(groups: &[Group<'_>]) -> Result<GroupRows, String> {
    let mut rows = groups
        .iter()
        .map(|(_, requests)| vec![0; requests.iter().map(|&(_, length)| length).sum()])
        .collect::<Vec<_>>();
    let serial = read_groups_into(
        groups,
        &mut rows.iter_mut().map(Vec::as_mut_slice).collect::<Vec<_>>(),
    )?;
    Ok((rows, serial))
}

/// Reads every group as one batch, group `g`'s rows end to end into
/// `outputs[g]`, and answers whether this thread read them because the
/// reader threads were unavailable. Refuses outputs of another length before
/// any read.
pub(super) fn read_groups_into(
    groups: &[Group<'_>],
    outputs: &mut [&mut [u8]],
) -> Result<bool, String> {
    if outputs.len() != groups.len()
        || groups
            .iter()
            .zip(outputs.iter())
            .any(|((_, requests), output)| {
                requests.iter().map(|&(_, length)| length).sum::<usize>() != output.len()
            })
    {
        return Err("mapped row read outputs differ from their requests".into());
    }
    static POOL: OnceLock<Result<Pool, String>> = OnceLock::new();
    read_using(
        POOL.get_or_init(Pool::new).as_ref().map_err(String::as_str),
        groups,
        outputs,
        super::start_reads(),
    )
}

fn read_using(
    pool: Result<&Pool, &str>,
    groups: &[Group<'_>],
    outputs: &mut [&mut [u8]],
    start: bool,
) -> Result<bool, String> {
    match pool {
        Ok(pool) => match pool.read_into(groups, outputs, super::readers(), start) {
            Ok(()) => return Ok(false),
            Err(Failure::Source(error)) => return Err(error),
            Err(Failure::Queue(error)) => fallback(&error),
        },
        Err(error) => fallback(error),
    }
    // One share per group on this thread: started reads still keep each
    // group's cold rows in flight together.
    for (&(source, requests), output) in groups.iter().zip(outputs.iter_mut()) {
        output.copy_from_slice(&read_share(source, requests, start)?);
    }
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn buffers(groups: &[Group<'_>]) -> Vec<Vec<u8>> {
        groups
            .iter()
            .map(|(_, requests)| vec![0; requests.iter().map(|&(_, length)| length).sum()])
            .collect()
    }

    /// [`read_using`] into new buffers.
    fn read_new(
        pool: Result<&Pool, &str>,
        groups: &[Group<'_>],
        start: bool,
    ) -> Result<GroupRows, String> {
        let mut rows = buffers(groups);
        let serial = read_using(
            pool,
            groups,
            &mut rows.iter_mut().map(Vec::as_mut_slice).collect::<Vec<_>>(),
            start,
        )?;
        Ok((rows, serial))
    }

    /// [`Pool::read_into`] into new buffers.
    fn pool_read(
        pool: &Pool,
        groups: &[Group<'_>],
        window: usize,
        start: bool,
    ) -> Result<Vec<Vec<u8>>, Failure> {
        let mut rows = buffers(groups);
        pool.read_into(
            groups,
            &mut rows.iter_mut().map(Vec::as_mut_slice).collect::<Vec<_>>(),
            window,
            start,
        )?;
        Ok(rows)
    }

    #[test]
    fn unavailable_or_closed_queue_uses_exact_serial_reads() {
        let path =
            std::env::temp_dir().join(format!("mapped-pool-fallback-{}", std::process::id()));
        std::fs::write(&path, [1, 9, 2, 8, 3, 7]).unwrap();
        let file = Arc::new(File::open(&path).unwrap());
        let requests = [(1, 3), (0, 2)];
        let expected = (vec![vec![9, 2, 8, 1, 9], vec![7]], true);
        let groups: [Group<'_>; 2] = [(&file, &requests), (&file, &[(5, 1)])];
        for start in [false, true] {
            assert_eq!(
                read_new(Err("reader admission unavailable"), &groups, start).unwrap(),
                expected
            );
            let (send, receive) = mpsc::sync_channel(1);
            drop(receive);
            assert_eq!(
                read_new(Ok(&Pool { send }), &groups, start).unwrap(),
                expected
            );
            assert!(
                read_new(
                    Err("reader admission unavailable"),
                    &[(&file, &[(5, 2)])],
                    start
                )
                .is_err()
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
                    pool_read(&pool, &[(&file, &requests)], window, start).ok()
                        == Some(vec![expected.concat()]),
                    "window {window} start {start}"
                );
                assert!(
                    pool_read(&pool, &[(&file, &requests[..1])], window, start).ok()
                        == Some(vec![expected[0].clone()])
                );
                assert!(
                    pool_read(&pool, &[(&file, &[])], window, start).ok() == Some(vec![Vec::new()])
                );
                assert!(pool_read(&pool, &[], window, start).ok() == Some(Vec::new()));
                // Groups come back each in its own request order.
                let split = [
                    (&file, &requests[..7]),
                    (&file, &[]),
                    (&file, &requests[7..]),
                ];
                assert!(
                    pool_read(&pool, &split, window, start).ok()
                        == Some(vec![
                            expected[..7].concat(),
                            Vec::new(),
                            expected[7..].concat()
                        ])
                );
            }
            assert!(matches!(
                pool_read(
                    &pool,
                    &[(&file, &[(0, 2)]), (&file, &[(4095, 2)])],
                    4,
                    start
                ),
                Err(Failure::Source(_))
            ));
        }
        // Outputs of another length are refused before any read.
        let mut short = [0_u8; 2];
        assert!(
            read_groups_into(&[(&file, &[(0, 3)])], &mut [&mut short[..]]).is_err()
                && short == [0, 0]
        );
        assert!(read_groups_into(&[(&file, &[(0, 2)])], &mut []).is_err());
        std::fs::remove_file(path).unwrap();
    }

    fn expected_bytes(requests: &[(u64, usize)]) -> Vec<u8> {
        requests
            .iter()
            .flat_map(|&(offset, length)| offset as u8..offset as u8 + length as u8)
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
            std::thread::spawn(move || {
                pool_read(&Pool { send }, &[(&file, &requests)], 3, false)
                    .ok()
                    .map(|mut rows| rows.remove(0))
            })
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
            std::thread::spawn(move || {
                pool_read(&Pool { send }, &[(&file, &requests)], 3, true)
                    .ok()
                    .map(|mut rows| rows.remove(0))
            })
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
    fn a_started_read_of_two_sources_admits_both_before_any_reply() {
        let path = |name: &str| {
            std::env::temp_dir().join(format!("mapped-pool-{name}-{}", std::process::id()))
        };
        std::fs::write(path("codes"), (0..64_u8).collect::<Vec<_>>()).unwrap();
        std::fs::write(path("scales"), (100..164_u8).collect::<Vec<_>>()).unwrap();
        let codes = Arc::new(File::open(path("codes")).unwrap());
        let scales = Arc::new(File::open(path("scales")).unwrap());
        let (send, receive) = mpsc::sync_channel(QUEUE);
        let wide = (0..6_u64).map(|i| (i * 8, 4)).collect::<Vec<_>>();
        let narrow = (0..3_u64).map(|i| (i * 2, 1)).collect::<Vec<_>>();
        let reader = {
            let (codes, scales, wide, narrow) = (
                Arc::clone(&codes),
                Arc::clone(&scales),
                wide.clone(),
                narrow.clone(),
            );
            std::thread::spawn(move || {
                pool_read(
                    &Pool { send },
                    &[(&codes, &wide), (&scales, &narrow)],
                    3,
                    true,
                )
                .ok()
            })
        };
        // Nine rows over a window of three make shares of three: two from the
        // first source and one from the second, all before any reply.
        let wait = std::time::Duration::from_millis(200);
        let shares = (0..3)
            .map(|_| receive.recv_timeout(wait).unwrap())
            .collect::<Vec<_>>();
        assert!(receive.recv_timeout(wait / 4).is_err());
        assert_eq!(
            shares
                .iter()
                .map(|w| (Arc::ptr_eq(&w.source, &scales), w.requests.clone()))
                .collect::<Vec<_>>(),
            [
                (false, wide[..3].to_vec()),
                (false, wide[3..].to_vec()),
                (true, narrow.clone())
            ]
        );
        for work in shares.into_iter().rev() {
            let result = read_share(&work.source, &work.requests, work.start);
            work.reply.send((work.index, result)).unwrap();
        }
        let expected = |bytes: &[u8], requests: &[(u64, usize)]| {
            requests
                .iter()
                .flat_map(|&(offset, length)| bytes[offset as usize..][..length].to_vec())
                .collect::<Vec<_>>()
        };
        assert_eq!(
            reader.join().unwrap(),
            Some(vec![
                expected(&(0..64).collect::<Vec<u8>>(), &wide),
                expected(&(100..164).collect::<Vec<u8>>(), &narrow)
            ])
        );
        for name in ["codes", "scales"] {
            std::fs::remove_file(path(name)).unwrap();
        }
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
        assert_eq!(rows, [exact(row), exact((0, 8)), exact(row)].concat());
        std::fs::remove_file(path).unwrap();
    }
}
