//! F64 weight matrices and work splitting for the oracle checks.

use std::sync::atomic::{AtomicUsize, Ordering};

/// A row-major f64 weight, [rows, cols]: output features by input features.
#[derive(Clone, Debug, PartialEq)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f64>,
}

impl Matrix {
    pub fn new(rows: usize, cols: usize, data: Vec<f64>) -> Self {
        assert_eq!(data.len(), rows * cols, "matrix shape");
        Self { rows, cols, data }
    }

    pub fn row(&self, row: usize) -> &[f64] {
        &self.data[row * self.cols..(row + 1) * self.cols]
    }
}

fn workers() -> usize {
    std::thread::available_parallelism().map_or(1, usize::from)
}

/// `work(i)` for every `i < count` on all workers, taking tasks in order;
/// results come back in index order.
pub fn par_map<T: Send>(count: usize, work: impl Fn(usize) -> T + Sync) -> Vec<T> {
    let threads = workers().min(count);
    if threads <= 1 {
        return (0..count).map(work).collect();
    }
    let next = AtomicUsize::new(0);
    let (work, next) = (&work, &next);
    let mut indexed = std::thread::scope(|scope| {
        let handles = (0..threads)
            .map(|_| {
                scope.spawn(move || {
                    let mut done = Vec::new();
                    loop {
                        let task = next.fetch_add(1, Ordering::Relaxed);
                        if task >= count {
                            break done;
                        }
                        done.push((task, work(task)));
                    }
                })
            })
            .collect::<Vec<_>>();
        handles
            .into_iter()
            .flat_map(|handle| handle.join().expect("oracle worker"))
            .collect::<Vec<_>>()
    });
    indexed.sort_unstable_by_key(|(task, _)| *task);
    indexed.into_iter().map(|(_, value)| value).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parallel_map_keeps_order() {
        assert_eq!(
            par_map(100, |i| i * 2),
            (0..100).map(|i| i * 2).collect::<Vec<_>>()
        );
    }
}
