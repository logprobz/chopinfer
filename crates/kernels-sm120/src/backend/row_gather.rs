//! The gather by index (`kernel_core::row_copy::GatherByIndex`) through the
//! gather entries a model's build compiles (`row::gather`). It moves bytes
//! only; an op claims what its entry's declaration claims
//! (`row::gather::declared_entries` for the shared symbol), and nothing for
//! an undeclared symbol. Every entry serves every row width, so entries of
//! different symbols are a tie. The op is refused when prepared if its
//! stated view alignment is below the kernel's 16-byte vectors, its rows
//! hold a partial vector, no row bucket holds its capacity or its
//! footprint at the capacity overflows u64.

use kernel_core::backend::{Counts, Launch, Need, OpInfo, Prepared, PreparedOp, Refused, View};
use kernel_core::contract::Refusal;
use kernel_core::properties::Invariance;
use kernel_core::row_copy::{GatherByIndex, GatherByIndexBuffers};

use super::select::{Candidate, choose};
use super::{MAX_CAPACITY, check_plan};
use crate::row::gather::{Gather, declared_entries};

/// The gather `op` through a registered gather entry.
pub(super) fn gather_by_index(
    symbols: &[String],
    op: &GatherByIndex,
    need: Need,
) -> Result<Prepared<GatherByIndexBuffers<View>>, Refused> {
    let chosen = choose(
        "gather by index",
        symbols
            .iter()
            .map(|symbol| Candidate::new(symbol.clone(), symbol, symbol.clone(), declared(symbol))),
        need,
    )?;
    if need.capacity == 0 || need.capacity > MAX_CAPACITY {
        return Err(Refused::Rule(format!(
            "an SM120 gather by index serves 1 to {MAX_CAPACITY} rows"
        )));
    }
    let gather = Gather::new(*op, need.capacity, &chosen.entry)?;
    check_plan(&gather.plan(need.capacity)?)?;
    let op = GatherOp {
        info: OpInfo::new(
            "sm120.row.gather.by_index.v1",
            chosen.guarantees,
            vec![(chosen.entry, 0)],
        ),
        gather,
    };
    need.check(&op.info)?;
    Ok(Box::new(op))
}

/// The guarantees the shared gather's declaration claims for `symbol`;
/// none for an undeclared entry.
fn declared(symbol: &str) -> Invariance {
    declared_entries()
        .into_iter()
        .find(|entry| entry.symbol == symbol)
        .map_or(Invariance::NONE, |entry| entry.invariance)
}

/// `chop::row::gather::by_index`: one CTA per row of the smallest row
/// bucket holding the launch's grid rows.
struct GatherOp {
    gather: Gather,
    info: OpInfo,
}

impl PreparedOp<GatherByIndexBuffers<View>> for GatherOp {
    fn info(&self) -> &OpInfo {
        &self.info
    }

    fn launches(
        &self,
        buffers: &GatherByIndexBuffers<View>,
        _workspace: View,
        counts: Counts,
    ) -> Result<Vec<Launch>, Refusal> {
        let (rows, grid) = self.gather.rows(counts.rows, counts.padded_rows)?;
        let op = self.gather.op();
        // The source's rows are the host's: every index names one of them.
        buffers.source.covers(op.row_bytes()?)?;
        buffers.row_indices.covers(u64::from(rows) * 4)?;
        buffers.count.covers(4)?;
        buffers.destination.covers(op.destination_bytes(rows)?)?;
        let plan = self.gather.plan(grid)?;
        let arguments = self.gather.arguments(
            buffers.source.handle,
            buffers.row_indices.handle,
            buffers.count.handle,
            buffers.destination.handle,
        )?;
        Ok(vec![Launch {
            symbol: self.gather.symbol().into(),
            rows: plan.grid[0],
            plan,
            arguments,
            collective: false,
            persistent: false,
        }])
    }
}

#[cfg(test)]
mod tests {
    use kernel_core::backend::Backend;
    use kernel_core::contract::{Argument, DType};

    use super::*;
    use crate::backend::{Sm120, view};
    use crate::row::gather::SYMBOL;

    fn op() -> GatherByIndex {
        GatherByIndex {
            dtype: DType::Bf16,
            width: 6144,
            alignment_bytes: 16,
        }
    }

    fn buffers(rows: u64) -> GatherByIndexBuffers<View> {
        GatherByIndexBuffers {
            source: view(1, 12_288),
            row_indices: view(2, rows * 4),
            count: view(3, 4),
            destination: view(4, rows * 12_288),
        }
    }

    #[test]
    fn the_shared_entry_gathers_over_the_bucket_of_its_grid() {
        let backend = Sm120::new().with_row_gather(SYMBOL);
        let need = Need::rows(8192).with_exact(Invariance::any_rows(8192));
        let prepared = backend.gather_by_index(&op(), need).unwrap();
        assert_eq!(prepared.info().symbols, [(SYMBOL.to_owned(), 0)]);
        let padded = Counts {
            padded_rows: 2048,
            ..Counts::rows(5)
        };
        let [launch] = &prepared.launches(&buffers(5), View::EMPTY, padded).unwrap()[..] else {
            panic!("one launch");
        };
        assert_eq!((launch.rows, launch.plan.grid), (2048, [2048, 1, 1]));
        assert_eq!(
            launch.arguments.0,
            [
                Argument::DeviceBuffer(1),
                Argument::DeviceBuffer(2),
                Argument::DeviceBuffer(3),
                Argument::DeviceBuffer(4),
                Argument::U32(768),
            ]
        );
        assert!(prepared.launches(&buffers(4), View::EMPTY, padded).is_err());
        let too_many = Counts {
            padded_rows: 4,
            ..Counts::rows(5)
        };
        assert!(
            prepared
                .launches(&buffers(5), View::EMPTY, too_many)
                .is_err()
        );
    }

    #[test]
    fn a_backend_without_the_entry_or_an_undeclared_entry_claims_nothing() {
        let need = Need::rows(64).with_exact(Invariance::any_rows(64));
        assert!(Sm120::new().gather_by_index(&op(), need).is_err());
        let undeclared = Sm120::new().with_row_gather("chop_row_gather_other_v1");
        assert!(matches!(
            undeclared.gather_by_index(&op(), need),
            Err(Refused::Inexact { .. })
        ));
        assert!(undeclared.gather_by_index(&op(), Need::rows(64)).is_ok());
    }
}
