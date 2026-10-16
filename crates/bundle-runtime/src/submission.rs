//! Bounded, single-owner submission of independent rank programs.
//!
//! Uploads must be staged before compute starts. Upload stages are bounded to one pinned-ring
//! window; backpressure may wait only for an earlier upload, never compute.
//! Host services stay on their owning thread, including RefCell backends.

/// An independent rank program. `submit` issues at most one bounded stage.
pub trait RankSubmission {
    type Error;
    fn stages(&self) -> usize;
    fn submit(&self, stage: usize) -> Result<(), Self::Error>;
    fn drain(&self) -> Result<(), Self::Error>;
}

/// The first submission error, plus whether any stream could not be drained.
#[derive(Debug, PartialEq, Eq)]
pub struct Failure<E> {
    pub error: E,
    pub undrained: bool,
}

/// Issue one stage per rank before advancing. Journal before calling the
/// backend because a failed service call may already have queued work.
/// Always drain every journaled rank, preserving the first error. Callers
/// must poison and retain scratch when `undrained` is true.
pub fn submit<R: RankSubmission>(
    ranks: &[R],
    mut cancelled: impl FnMut() -> Option<R::Error>,
) -> Result<(), Failure<R::Error>> {
    submit_timed(ranks, &mut cancelled, &mut Timings::default())
}

#[derive(Default, Debug)]
pub struct Timings {
    pub submit: std::time::Duration,
    pub drain: std::time::Duration,
}

pub fn submit_timed<R: RankSubmission>(
    ranks: &[R],
    mut cancelled: impl FnMut() -> Option<R::Error>,
    timings: &mut Timings,
) -> Result<(), Failure<R::Error>> {
    let started = std::time::Instant::now();
    let mut journal = vec![false; ranks.len()];
    let mut first = None;
    'stages: for stage in 0..ranks.iter().map(R::stages).max().unwrap_or(0) {
        if let Some(error) = cancelled() {
            first = Some(error);
            break;
        }
        for (rank, program) in ranks.iter().enumerate() {
            if stage < program.stages() {
                journal[rank] = true;
                if let Err(error) = program.submit(stage) {
                    first = Some(error);
                    break 'stages;
                }
            }
        }
    }
    timings.submit = started.elapsed();
    let started = std::time::Instant::now();
    let mut undrained = false;
    for (program, touched) in ranks.iter().zip(journal) {
        if touched && let Err(error) = program.drain() {
            undrained = true;
            first.get_or_insert(error);
        }
    }
    timings.drain = started.elapsed();
    first.map_or(Ok(()), |error| Err(Failure { error, undrained }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;

    struct Rank<'a> {
        rank: usize,
        log: &'a RefCell<Vec<(usize, usize)>>,
        fail: Option<usize>,
        wait_fail: bool,
    }
    impl RankSubmission for Rank<'_> {
        type Error = &'static str;
        fn stages(&self) -> usize {
            6
        }
        fn submit(&self, stage: usize) -> Result<(), Self::Error> {
            self.log.borrow_mut().push((self.rank, stage));
            if self.fail == Some(stage) {
                Err("submit")
            } else {
                Ok(())
            }
        }
        fn drain(&self) -> Result<(), Self::Error> {
            self.log.borrow_mut().push((self.rank, 6));
            if self.wait_fail { Err("wait") } else { Ok(()) }
        }
    }
    #[test]
    fn peers_advance_before_waits_and_every_failed_rank_is_drained() {
        for failure in [None, Some(0), Some(3), Some(5)] {
            let log = RefCell::new(Vec::new());
            let ranks = [
                Rank {
                    rank: 0,
                    log: &log,
                    fail: None,
                    wait_fail: true,
                },
                Rank {
                    rank: 1,
                    log: &log,
                    fail: failure,
                    wait_fail: true,
                },
            ];
            let result = submit(&ranks, || None).unwrap_err();
            assert!(result.undrained);
            assert_eq!(
                result.error,
                if failure.is_some() { "submit" } else { "wait" }
            );
            let log = log.borrow();
            assert_eq!(&log[log.len() - 2..], &[(0, 6), (1, 6)]);
            for pair in log[..log.len() - 2].chunks_exact(2) {
                assert_eq!(pair[0].0, 0);
                assert_eq!(pair[1], (1, pair[0].1));
            }
        }
    }
    #[test]
    fn failure_journals_exactly_the_ranks_already_touched() {
        for stage in [0, 2, 5] {
            let log = RefCell::new(Vec::new());
            let ranks = (0..4)
                .map(|rank| Rank {
                    rank,
                    log: &log,
                    fail: (rank == 1).then_some(stage),
                    wait_fail: false,
                })
                .collect::<Vec<_>>();
            assert_eq!(submit(&ranks, || None).unwrap_err().error, "submit");
            let mut expected = (0..stage)
                .flat_map(|s| (0..4).map(move |r| (r, s)))
                .collect::<Vec<_>>();
            expected.extend([(0, stage), (1, stage)]);
            expected.extend((0..if stage == 0 { 2 } else { 4 }).map(|r| (r, 6)));
            assert_eq!(*log.borrow(), expected);
        }
    }

    #[test]
    fn cancellation_stops_at_a_boundary_and_drains_before_returning() {
        let log = RefCell::new(Vec::new());
        let ranks = [0, 1].map(|rank| Rank {
            rank,
            log: &log,
            fail: None,
            wait_fail: false,
        });
        let result = submit(&ranks, || (log.borrow().len() == 4).then_some("cancel"));
        assert_eq!(
            result,
            Err(Failure {
                error: "cancel",
                undrained: false
            })
        );
        assert_eq!(
            *log.borrow(),
            [(0, 0), (1, 0), (0, 1), (1, 1), (0, 6), (1, 6)]
        );
    }
}
