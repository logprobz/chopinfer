use super::*;
use std::os::fd::AsRawFd;
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};

struct Temp(PathBuf);
impl Drop for Temp {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

fn temp(label: &str) -> (Temp, Arc<File>) {
    let path = std::env::temp_dir().join(format!("reader-files-{label}-{}", std::process::id()));
    std::fs::write(&path, (0..4096_u32).map(|i| i as u8).collect::<Vec<_>>()).unwrap();
    let file = Arc::new(File::open(&path).unwrap());
    (Temp(path), file)
}

/// This process's descriptors that name `path`.
pub(crate) fn links(path: &Path) -> Vec<i32> {
    let path = std::fs::canonicalize(path).unwrap();
    std::fs::read_dir("/proc/self/fd")
        .unwrap()
        .filter_map(|entry| {
            let entry = entry.ok()?;
            let target = std::fs::read_link(entry.path()).ok()?;
            (target == path).then(|| entry.file_name().to_str()?.parse().ok())?
        })
        .collect()
}

/// Whether descriptor `fd` closes on exec, from its procfs flags (octal).
pub(crate) fn closes_on_exec(fd: i32) -> bool {
    let info = std::fs::read_to_string(format!("/proc/self/fdinfo/{fd}")).unwrap();
    let flags = info
        .lines()
        .find_map(|line| line.strip_prefix("flags:"))
        .map(|flags| u32::from_str_radix(flags.trim(), 8).unwrap())
        .unwrap();
    flags & libc::O_CLOEXEC as u32 != 0
}

/// The served opener, counting its calls.
fn counted(calls: &AtomicUsize) -> impl Fn(&File) -> io::Result<File> + '_ {
    move |file| {
        calls.fetch_add(1, Ordering::Relaxed);
        reopen(file)
    }
}

#[test]
fn the_budget_is_a_sixteenth_of_the_open_file_limit_up_to_256() {
    for (soft, limit) in [
        (None, 0),
        (Some(15), 0),
        (Some(64), 4),
        (Some(1024), 64),
        (Some(4096), 256),
        (Some(1 << 20), 256),
        (Some(u64::MAX), 256),
    ] {
        assert_eq!(Budget::for_soft_limit(soft).limit(), limit, "{soft:?}");
    }
}

#[cfg(target_os = "linux")]
#[test]
fn reader_descriptions_are_new_bounded_reused_and_closed_with_their_owner() {
    let (temp, file) = temp("bounded");
    let budget = Arc::new(Budget::new(5));
    let calls = AtomicUsize::new(0);
    let readers = ReaderFiles::default();
    assert_eq!(links(&temp.0).len(), 1);
    let three = readers.get_with(&file, 3, &counted(&calls), &budget);
    assert_eq!((three.len(), calls.load(Ordering::Relaxed)), (3, 3));
    let retained = file.metadata().unwrap();
    for description in &three {
        // A new open file description of the same file, not the retained
        // descriptor and not a duplicate of it, that closes on exec.
        let metadata = description.metadata().unwrap();
        assert_eq!(
            (metadata.dev(), metadata.ino()),
            (retained.dev(), retained.ino())
        );
        assert_ne!(description.as_raw_fd(), file.as_raw_fd());
        assert!(closes_on_exec(description.as_raw_fd()));
    }
    // Each reads from its own position: a seek through one moves no other.
    use std::io::{Read, Seek, SeekFrom};
    (&*three[0]).seek(SeekFrom::Start(100)).unwrap();
    let mut byte = [0];
    (&*three[1]).read_exact(&mut byte).unwrap();
    assert_eq!(byte, [0]);
    // Fewer wanted answers the first held, opening nothing.
    let two = readers.get_with(&file, 2, &counted(&calls), &budget);
    assert!(two.iter().zip(&three).all(|(a, b)| Arc::ptr_eq(a, b)));
    assert_eq!(calls.load(Ordering::Relaxed), 3);
    // More stop at the budget, which ends acquisition for good.
    let before = failures();
    let all = readers.get_with(&file, 7, &counted(&calls), &budget);
    assert_eq!((all.len(), calls.load(Ordering::Relaxed)), (5, 5));
    assert!(failures() > before);
    assert_eq!(
        readers.get_with(&file, 7, &counted(&calls), &budget).len(),
        5
    );
    assert_eq!((calls.load(Ordering::Relaxed), budget.held()), (5, 5));
    assert_eq!(links(&temp.0).len(), 6);
    let fds = all.iter().map(|file| file.as_raw_fd()).collect::<Vec<_>>();
    drop((three, two, all));
    drop(readers);
    // The owner closed every description and gave back its budget.
    assert_eq!(budget.held(), 0);
    let left = links(&temp.0);
    assert_eq!(left, [file.as_raw_fd()]);
    assert!(fds.iter().all(|fd| !left.contains(fd)));
}

/// The slots of a dropped owner's descriptions come back only once the
/// descriptions are closed, so concurrent acquisition never holds more open
/// descriptions than the budget.
#[cfg(target_os = "linux")]
#[test]
fn descriptions_close_before_their_slots_are_given_back() {
    let (temp, file) = temp("close-first");
    let path = temp.0.clone();
    let open_at_refund = Arc::new(Mutex::new(Vec::new()));
    let budget = {
        let open_at_refund = Arc::clone(&open_at_refund);
        Arc::new(Budget::watched(8, move |count| {
            open_at_refund
                .lock()
                .unwrap()
                .push((count, links(&path).len()));
        }))
    };
    let readers = ReaderFiles::default();
    assert_eq!(readers.get_with(&file, 4, &reopen, &budget).len(), 4);
    assert_eq!(links(&temp.0).len(), 5);
    drop(readers);
    // Four slots back, with only the retained descriptor still open.
    assert_eq!(*open_at_refund.lock().unwrap(), [(4, 1)]);
    assert_eq!(budget.held(), 0);
}

#[cfg(target_os = "linux")]
#[test]
fn a_failed_open_is_cached_and_leaves_the_retained_descriptor() {
    let (_temp, file) = temp("failures");
    let (other, _) = temp("other-file");
    let fail =
        |code: i32| move |_: &File| -> io::Result<File> { Err(io::Error::from_raw_os_error(code)) };
    type Open = Box<Opener<'static>>;
    let cases: Vec<(&str, Open)> = vec![
        ("procfs absent", Box::new(fail(libc::ENOENT))),
        ("procfs refused", Box::new(fail(libc::EACCES))),
        ("process limit", Box::new(fail(libc::EMFILE))),
        ("system limit", Box::new(fail(libc::ENFILE))),
        // Another file where the magic link should name this one.
        (
            "identity mismatch",
            Box::new(move |_: &File| File::open(&other.0)),
        ),
    ];
    for (case, open) in cases {
        let budget = Arc::new(Budget::new(256));
        let calls = AtomicUsize::new(0);
        let counting = |file: &File| {
            calls.fetch_add(1, Ordering::Relaxed);
            open(file)
        };
        let readers = ReaderFiles::default();
        let before = failures();
        for _ in 0..3 {
            assert!(
                readers.get_with(&file, 24, &counting, &budget).is_empty(),
                "{case}"
            );
        }
        assert_eq!(calls.load(Ordering::Relaxed), 1, "{case}: retried");
        assert!(failures() > before, "{case}");
        assert_eq!(budget.held(), 0, "{case}");
    }
    // A failure after some opens keeps those.
    let budget = Arc::new(Budget::new(256));
    let calls = AtomicUsize::new(0);
    let third_fails = |file: &File| {
        if calls.fetch_add(1, Ordering::Relaxed) == 2 {
            Err(io::Error::from_raw_os_error(libc::EMFILE))
        } else {
            reopen(file)
        }
    };
    let readers = ReaderFiles::default();
    assert_eq!(readers.get_with(&file, 24, &third_fails, &budget).len(), 2);
    assert_eq!(readers.get_with(&file, 24, &third_fails, &budget).len(), 2);
    assert_eq!((calls.load(Ordering::Relaxed), budget.held()), (3, 2));
}
