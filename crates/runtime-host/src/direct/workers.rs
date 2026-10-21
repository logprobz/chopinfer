//! Reusable scoped workers whose shutdown joins every owned thread.

use rayon_core::{ThreadBuilder, ThreadPool, ThreadPoolBuilder};
use std::thread::JoinHandle;

pub(super) struct Workers {
    pool: Option<ThreadPool>,
    threads: Vec<JoinHandle<()>>,
}

impl Workers {
    pub fn new(count: usize) -> Result<Self, String> {
        Self::build(count, |builder, thread| builder.spawn(move || thread.run()))
    }

    fn build(
        count: usize,
        mut spawn: impl FnMut(std::thread::Builder, ThreadBuilder) -> std::io::Result<JoinHandle<()>>,
    ) -> Result<Self, String> {
        let mut threads = Vec::with_capacity(count);
        let built = ThreadPoolBuilder::new()
            .num_threads(count)
            .thread_name(|index| format!("weight-loader-{index}"))
            .spawn_handler(|thread| {
                let mut builder = std::thread::Builder::new();
                if let Some(name) = thread.name() {
                    builder = builder.name(name.to_owned());
                }
                if let Some(size) = thread.stack_size() {
                    builder = builder.stack_size(size);
                }
                threads.push(spawn(builder, thread)?);
                Ok(())
            })
            .build();
        match built {
            Ok(pool) => Ok(Self {
                pool: Some(pool),
                threads,
            }),
            Err(error) => {
                // A failed builder signals any workers it already started.
                for thread in threads {
                    let _ = thread.join();
                }
                Err(format!("start weight workers: {error}"))
            }
        }
    }

    pub fn pool(&self) -> &ThreadPool {
        self.pool.as_ref().expect("workers are live until drop")
    }
}

impl Drop for Workers {
    fn drop(&mut self) {
        // Dropping the pool signals shutdown. Join after that signal, with
        // no source or scratch locks held. Callers have joined their scopes.
        drop(self.pool.take());
        for thread in self.threads.drain(..) {
            let _ = thread.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;
    use std::sync::{Barrier, Mutex};

    #[test]
    fn a_partial_spawn_failure_joins_the_workers_already_started() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicUsize, Ordering};
        let finished = Arc::new(AtomicUsize::new(0));
        let mut attempts = 0;
        let built = Workers::build(4, |builder, thread| {
            attempts += 1;
            if attempts == 3 {
                return Err(std::io::Error::other("injected spawn failure"));
            }
            let finished = finished.clone();
            builder.spawn(move || {
                thread.run();
                finished.fetch_add(1, Ordering::SeqCst);
            })
        });
        assert!(built.is_err());
        assert_eq!(attempts, 3);
        assert_eq!(finished.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn a_panicking_scope_joins_other_tasks_and_leaves_workers_reusable() {
        use std::sync::atomic::{AtomicBool, Ordering};
        let workers = Workers::new(2).unwrap();
        let joined = AtomicBool::new(false);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            workers.pool().in_place_scope(|scope| {
                scope.spawn(|_| panic!("injected packing worker panic"));
                scope.spawn(|_| {
                    joined.store(true, Ordering::SeqCst);
                });
            });
        }));
        assert!(result.is_err());
        assert!(joined.load(Ordering::SeqCst));
        assert_eq!(workers.pool().install(|| 7), 7);
        drop(workers);
    }

    #[test]
    fn scopes_reuse_the_same_bounded_workers() {
        let workers = Workers::new(3).unwrap();
        let run = || {
            let barrier = Barrier::new(3);
            let ids = Mutex::new(HashSet::new());
            workers.pool().in_place_scope(|scope| {
                for _ in 0..3 {
                    scope.spawn(|_| {
                        ids.lock().unwrap().insert(std::thread::current().id());
                        barrier.wait();
                    });
                }
            });
            ids.into_inner().unwrap()
        };
        let first = run();
        assert_eq!(first.len(), 3);
        assert_eq!(first, run());
    }
}
