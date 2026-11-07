//! The snapshot copy (`kernel_core::snapshot_copy`) through the entry a
//! model's build compiles (`snapshot_copy`): one launch per table, its grid
//! over the op's pages. An op claims what its entry's declaration claims
//! (`snapshot_copy::declared_entries` for the shared symbol), and nothing
//! for an undeclared symbol. Every entry serves every table, so entries of
//! different symbols are a tie.

use kernel_core::backend::{Counts, Launch, Need, OpInfo, Prepared, PreparedOp, Refused, View};
use kernel_core::contract::Refusal;
use kernel_core::properties::Invariance;
use kernel_core::snapshot_copy::{SnapshotCopy, SnapshotCopyBuffers};

use super::check_plan;
use super::select::{Candidate, choose};
use crate::snapshot_copy::{Copy, Entry, declared_entries};

/// The snapshot copy `op` through the registered entry.
pub(super) fn snapshot_copy(
    entries: &[Entry],
    op: &SnapshotCopy,
    need: Need,
) -> Result<Prepared<SnapshotCopyBuffers<View>>, Refused> {
    let (symbol, guarantees) = symbol(entries, need)?;
    let copy = Copy::new(*op, need.capacity, &symbol)?;
    check_plan(&copy.plan()?)?;
    let op = SnapshotCopyOp {
        info: OpInfo::new("sm120.snapshot.copy.v1", guarantees, vec![(symbol, 0)]),
        copy,
    };
    need.check(&op.info)?;
    Ok(Box::new(op))
}

/// The symbol of the entry that serves the op and what the op may claim
/// through it.
fn symbol(entries: &[Entry], need: Need) -> Result<(String, Invariance), Refused> {
    let chosen = choose(
        "snapshot copy",
        entries.iter().map(|entry| {
            let guarantees = declared(&entry.symbol);
            Candidate::new(entry.clone(), &entry.symbol, launch(entry), guarantees)
        }),
        need,
    )?;
    Ok((chosen.entry.symbol, chosen.guarantees))
}

/// The guarantees the shared entry's declaration claims for `symbol`; none
/// for an undeclared entry.
fn declared(symbol: &str) -> Invariance {
    declared_entries()
        .into_iter()
        .find(|entry| entry.symbol == symbol)
        .map_or(Invariance::NONE, |entry| entry.invariance)
}

/// What an entry launches: its symbol, whose code is the one kernel.
fn launch(entry: &Entry) -> String {
    let Entry { symbol } = entry;
    symbol.clone()
}

/// `chop::snapshot::copy`: one thread per vector of the op's pages.
struct SnapshotCopyOp {
    copy: Copy,
    info: OpInfo,
}

impl PreparedOp<SnapshotCopyBuffers<View>> for SnapshotCopyOp {
    fn info(&self) -> &OpInfo {
        &self.info
    }

    /// Checks the segment count against the capacity, the views against
    /// the op's and the table's size against its pages and segments; the
    /// host checks the table's contents.
    fn launches(
        &self,
        buffers: &SnapshotCopyBuffers<View>,
        _workspace: View,
        counts: Counts,
    ) -> Result<Vec<Launch>, Refusal> {
        let op = self.copy.op();
        let segments = counts.rows;
        if segments == 0 || segments > self.copy.capacity() {
            return Err(Refusal(
                "snapshot copy segments exceed the prepared capacity",
            ));
        }
        buffers.table.covers(op.table_bytes(segments)?)?;
        buffers.pool.covers(op.page_bytes)?;
        if buffers.views.iter().any(|view| view.bytes == 0) {
            return Err(Refusal("a snapshot copy's view holds no bytes"));
        }
        let views = buffers
            .views
            .iter()
            .map(|view| view.handle)
            .collect::<Vec<_>>();
        let arguments =
            self.copy
                .arguments(&views, buffers.pool.handle, buffers.table.handle, segments)?;
        Ok(vec![Launch {
            symbol: self.copy.symbol().into(),
            rows: segments,
            plan: self.copy.plan()?,
            arguments,
            collective: false,
            persistent: false,
        }])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{MAX_CAPACITY, Sm120, view};
    use crate::snapshot_copy::{INVARIANCE, SYMBOL, entries};
    use kernel_core::backend::Backend;
    use kernel_core::snapshot_copy::Direction;

    fn shared() -> Sm120 {
        entries().into_iter().fold(Sm120::new(), |backend, entry| {
            backend.with_snapshot_copy(entry)
        })
    }

    /// The Flash snapshot: 84 pieces over three views of 128 native pages.
    fn op(direction: Direction) -> SnapshotCopy {
        SnapshotCopy {
            direction,
            views: 3,
            page_bytes: 56_960,
            pages: 128,
        }
    }

    fn buffers(table: u64) -> SnapshotCopyBuffers<View> {
        SnapshotCopyBuffers {
            views: vec![view(1, 1 << 20), view(2, 1 << 20), view(3, 1 << 20)],
            pool: view(4, 1024 * 56_960),
            table: view(5, table),
        }
    }

    #[test]
    fn copies_launch_what_the_planner_plans() {
        for direction in [Direction::ToPages, Direction::FromPages] {
            let op = op(direction);
            let prepared = shared().snapshot_copy(&op, Need::rows(84)).unwrap();
            assert_eq!(prepared.info().implementation, "sm120.snapshot.copy.v1");
            assert_eq!(prepared.info().guarantees, INVARIANCE);
            assert_eq!(prepared.info().symbols, [(SYMBOL.to_owned(), 0)]);
            let full = buffers(op.table_bytes(84).unwrap());
            let built = Copy::new(op, 84, SYMBOL).unwrap();
            for segments in [1, 44, 84] {
                let launches = prepared
                    .launches(&full, View::EMPTY, Counts::rows(segments))
                    .unwrap();
                assert_eq!(
                    launches,
                    [Launch {
                        symbol: SYMBOL.into(),
                        rows: segments,
                        plan: built.plan().unwrap(),
                        arguments: built.arguments(&[1, 2, 3], 4, 5, segments).unwrap(),
                        collective: false,
                        persistent: false,
                    }]
                );
            }
            for segments in [0, 85] {
                assert!(
                    prepared
                        .launches(&full, View::EMPTY, Counts::rows(segments))
                        .is_err()
                );
            }
            // The table holds the page IDs and the launch's segments, and
            // the op binds each of its views.
            let short = buffers(op.table_bytes(44).unwrap() - 1);
            assert!(
                prepared
                    .launches(&short, View::EMPTY, Counts::rows(44))
                    .is_err()
            );
            let mut missing = full.clone();
            missing.views.pop();
            assert!(
                prepared
                    .launches(&missing, View::EMPTY, Counts::rows(44))
                    .is_err()
            );
            let mut empty = full.clone();
            empty.views[1] = view(2, 0);
            assert!(
                prepared
                    .launches(&empty, View::EMPTY, Counts::rows(44))
                    .is_err()
            );
        }
    }

    #[test]
    fn prepare_refuses_without_an_entry_or_past_the_limits() {
        let op = op(Direction::ToPages);
        assert!(Sm120::new().snapshot_copy(&op, Need::rows(84)).is_err());
        for need in [Need::rows(0), Need::rows(MAX_CAPACITY + 1)] {
            assert!(shared().snapshot_copy(&op, need).is_err());
        }
        let partial = SnapshotCopy {
            page_bytes: 56_968,
            ..op
        };
        assert!(shared().snapshot_copy(&partial, Need::rows(84)).is_err());
        // Two entries of different symbols tie; an undeclared entry claims
        // nothing.
        let tied = shared().with_snapshot_copy(Entry::new("chop_model_snapshot_copy_v1"));
        assert!(tied.snapshot_copy(&op, Need::rows(84)).is_err());
        let own = Sm120::new().with_snapshot_copy(Entry::new("chop_model_snapshot_copy_v1"));
        let prepared = own.snapshot_copy(&op, Need::rows(84)).unwrap();
        assert_eq!(prepared.info().guarantees, Invariance::NONE);
        assert!(
            own.snapshot_copy(&op, Need::rows(84).with_exact(INVARIANCE))
                .is_err()
        );
    }
}
