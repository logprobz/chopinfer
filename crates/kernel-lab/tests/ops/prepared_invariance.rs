//! Invariance adapters of entries that run through the ops SM120 prepares
//! for them (`kernels_sm120::backend`): each family states, for a declared
//! entry, the op whose launches run it and the kit case over that op
//! (`kernel_lab::kit::cases`), and the harness launches units of the case
//! (`kernel_lab::kit::units`).
//!
//! A family's entries are the lab's at the served shapes. Every case of an
//! entry selects its rows from one source: the case of the entry's op at
//! [`Family::source_rows`], whose builder made its rows valid together. A
//! seed picks a distinct row of the source ([`seed_rows`]); a prompt row r
//! of a chunk case is source row r. So two launches compare the same rows
//! wherever they sit.
//!
//! - Rows: a launch of R rows through the entry's op against each row
//!   alone through the op its one-row bits refer to
//!   ([`Family::reference`]), one row each.
//! - Batch: the probed rows beside rows of other seeds.
//! - Chunk: a prompt in one launch against its chunks, each launched alone
//!   at its own rows.
//!
//! Ops that route row counts to different entries (the Sinkhorn
//! mix-collapse's split and tiles) run the entry at the row counts its
//! family names ([`Family::row_counts`]); chunks of other sizes run the
//! op's other entries, which the entry's claim must equal.

use crate::invariance::{Adapter, Units, row_counts};
use kernel_lab::kit::{LabModules, Lookup, Modules, OpCase};
use kernel_lab::{LabResult, Session};
use kernels_sm120::invariance::{Declared, OpKind};
use std::collections::BTreeSet;

/// How a family launches its declared entries through SM120.
pub trait Family {
    fn kind(&self) -> OpKind;

    /// The case of `entry`'s op at `capacity` rows, prepared so that it
    /// launches the entry at the row counts the family names.
    fn case(&self, entry: &Declared, capacity: u32) -> LabResult<OpCase<'static>>;

    /// The case, over the same roles, of the op whose one-row launch the
    /// entry's rows claim refers to: by default the entry's own op.
    fn reference(&self, entry: &Declared, capacity: u32) -> LabResult<OpCase<'static>> {
        self.case(entry, capacity)
    }

    /// The capacity a reference case of one row may take: one, unless its
    /// fixed images follow the capacity (a KV write's pool), when it takes
    /// the source's.
    fn reference_rows(&self, entry: &Declared) -> u32 {
        let _ = entry;
        1
    }

    /// Rows of the source every case selects from.
    fn source_rows(&self, entry: &Declared) -> u32 {
        let _ = entry;
        512
    }

    fn row_counts(&self, entry: &Declared, max: u32) -> Vec<u32> {
        let _ = entry;
        row_counts(max)
    }

    fn batch_units(&self, entry: &Declared) -> usize {
        let _ = entry;
        4
    }

    fn prompt_rows(&self, entry: &Declared) -> u32 {
        self.source_rows(entry)
    }
}

/// Distinct source rows for `seeds`, a source of `rows` rows: each seed's
/// row modulo `rows`, the next free one when taken.
pub fn seed_rows(seeds: &[u64], rows: u32) -> LabResult<Vec<usize>> {
    let rows = rows as usize;
    if seeds.len() > rows {
        return Err(format!("{} seeds for a source of {rows} rows", seeds.len()));
    }
    let mut taken = BTreeSet::new();
    Ok(seeds
        .iter()
        .map(|&seed| {
            let mut row = (seed % rows as u64) as usize;
            while !taken.insert(row) {
                row = (row + 1) % rows;
            }
            row
        })
        .collect())
}

/// The adapter of a family.
pub struct Prepared<F>(pub F);

impl<F: Family> Prepared<F> {
    fn source(&self, entry: &Declared, rows: u32) -> LabResult<kernel_lab::conformance::Images> {
        Ok(self.0.case(entry, rows)?.images())
    }

    /// The units of one launch of `order`'s source rows through `case`.
    fn launch(
        &self,
        session: &Session,
        functions: &Lookup<'_>,
        case: &OpCase<'static>,
        source: &kernel_lab::conformance::Images,
        order: &[usize],
    ) -> LabResult<Units> {
        let images = case.select(source, order)?;
        case.units(session, functions, &images, order.len() as u32)
    }
}

impl<F: Family> Adapter for Prepared<F> {
    fn kind(&self) -> OpKind {
        self.0.kind()
    }

    fn prompt_rows(&self, entry: &Declared) -> u32 {
        self.0.prompt_rows(entry)
    }

    fn chunks(
        &mut self,
        sessions: &[Session],
        entry: &Declared,
        rows: u32,
        ends: &[u32],
    ) -> LabResult<Units> {
        let source = self.source(entry, rows.max(self.0.source_rows(entry)))?;
        let functions = LabModules.load(&sessions[0])?;
        let mut units = Vec::new();
        let mut start = 0;
        for &end in ends {
            let case = self.0.case(entry, end - start)?;
            let order = (start as usize..end as usize).collect::<Vec<_>>();
            units.extend(self.launch(&sessions[0], &*functions, &case, &source, &order)?);
            start = end;
        }
        Ok(units)
    }

    fn units(&mut self, sessions: &[Session], entry: &Declared, seeds: &[u64]) -> LabResult<Units> {
        let rows = self.0.source_rows(entry);
        let source = self.source(entry, rows)?;
        let order = seed_rows(seeds, rows)?;
        let case = self.0.case(entry, seeds.len() as u32)?;
        let functions = LabModules.load(&sessions[0])?;
        self.launch(&sessions[0], &*functions, &case, &source, &order)
    }

    fn alone(&mut self, sessions: &[Session], entry: &Declared, seeds: &[u64]) -> LabResult<Units> {
        let rows = self.0.source_rows(entry);
        let source = self.source(entry, rows)?;
        let order = seed_rows(seeds, rows)?;
        let reference = self.0.reference(entry, self.0.reference_rows(entry))?;
        let functions = LabModules.load(&sessions[0])?;
        order
            .iter()
            .map(|&row| {
                self.launch(&sessions[0], &*functions, &reference, &source, &[row])?
                    .pop()
                    .ok_or_else(|| format!("{}: no unit", entry.symbol))
            })
            .collect()
    }

    fn row_counts(&self, entry: &Declared, max: u32) -> Vec<u32> {
        self.0.row_counts(entry, max)
    }

    fn batch_units(&self, entry: &Declared) -> usize {
        self.0.batch_units(entry)
    }
}

#[test]
fn seed_rows_are_distinct_and_follow_the_seeds() {
    let rows = seed_rows(&[3, 3, 10, 13], 10).unwrap();
    assert_eq!(rows, [3, 4, 0, 5]);
    assert!(seed_rows(&[1, 2, 3], 2).is_err());
    assert_eq!(seed_rows(&[7], 64).unwrap(), [7]);
}
