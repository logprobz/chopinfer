//! Count misses inside one legacy asynchronous callback, excluding other work.
use std::cell::Cell;
use std::sync::atomic::{AtomicU64, Ordering};

thread_local! {
    static MISSES: Cell<Option<u64>> = const { Cell::new(None) };
}

static GATHERS: AtomicU64 = AtomicU64::new(0);
static ROWS: AtomicU64 = AtomicU64::new(0);

pub(crate) fn record(rows: u64) {
    MISSES.with(|count| {
        if let Some(previous) = count.get() {
            count.set(Some(previous.saturating_add(rows)));
        }
    });
}

struct Scope(Option<u64>);
impl Drop for Scope {
    fn drop(&mut self) {
        MISSES.with(|count| count.set(self.0));
    }
}

pub(crate) fn count<T>(copy: impl FnOnce() -> T) -> (T, u64) {
    let scope = Scope(MISSES.with(|count| count.replace(Some(0))));
    let result = copy();
    let misses = MISSES.with(|count| count.get().expect("active miss counter"));
    drop(scope);
    (result, misses)
}

pub(crate) fn observe<T>(copy: impl FnOnce() -> T) -> T {
    let (result, misses) = count(copy);
    GATHERS.fetch_add(1, Ordering::Relaxed);
    ROWS.fetch_add(misses, Ordering::Relaxed);
    result
}

pub(super) fn snapshot() -> (u64, u64) {
    (
        GATHERS.load(Ordering::Relaxed),
        ROWS.load(Ordering::Relaxed),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn callback_scope_excludes_other_threads_and_restores_after_unwind() {
        record(100);
        let (_, misses) = count(|| {
            record(3);
            std::thread::spawn(|| record(200)).join().unwrap();
            assert!(std::panic::catch_unwind(|| count(|| panic!("test unwind"))).is_err());
            record(4);
        });
        assert_eq!(misses, 7);
        assert_eq!(MISSES.with(Cell::get), None);
    }
}
