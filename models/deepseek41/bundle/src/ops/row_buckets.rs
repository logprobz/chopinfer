//! Prepare small rows separately when a row op otherwise keeps a prefill plan.
use kernel_core::backend::{Need, Prepared, Refused};
use model_deepseek41_contract::dspark_options::SMALL_DRAFT_ROWS;
use model_deepseek41_contract::serving::DECODE_BATCH;
use model_deepseek41_contract::verify_bounds;

use super::{Asks, exact};
use crate::program::Phase;

pub(crate) struct RowBuckets<B> {
    every: Prepared<B>,
    small: Prepared<B>,
    /// Decode and verify rows above `DECODE_BATCH` up to a drafter forward's
    /// small-row limit (`SMALL_DRAFT_ROWS`).
    draft: Prepared<B>,
    draft_rows: u32,
    /// A one-row op, where one row has an entry of its own.
    one: Option<Prepared<B>>,
    /// Decode and verify rows above the draft rows, narrowest first: each
    /// op serves the rows above the previous one up to its own.
    wide: Vec<(u32, Prepared<B>)>,
}
impl<B> RowBuckets<B> {
    pub(crate) fn prepare(
        asks: &Asks,
        what: impl Into<String>,
        every: Need,
        shape: &crate::shape::Shape,
        mut prepare: impl FnMut(Need) -> Result<Prepared<B>, Refused>,
    ) -> Result<Self, Refused> {
        let what = what.into();
        let large = asks.ask(&what, every, &mut prepare)?;
        let small = asks.ask(
            format!("{what} small rows"),
            exact::need(DECODE_BATCH, exact::every_phase(DECODE_BATCH)),
            &mut prepare,
        )?;
        let draft_rows = SMALL_DRAFT_ROWS.max(shape.verify_rows_within(verify_bounds::ROW_BUCKETS));
        let draft = asks.ask(
            format!("{what} draft rows"),
            exact::need(draft_rows, exact::every_phase(draft_rows)),
            prepare,
        )?;
        Ok(Self {
            every: large,
            small,
            draft,
            draft_rows,
            one: None,
            wide: Vec::new(),
        })
    }

    /// Adds an op for each wide verify bucket the instance reaches
    /// ([`crate::shape::Shape::wide_verify_rows`]), which
    /// [`Self::at_phase`] returns for decode and verify rows above the
    /// draft rows. Prefill keeps its plans.
    pub(crate) fn with_wide(
        mut self,
        asks: &Asks,
        what: impl Into<String>,
        shape: &crate::shape::Shape,
        mut prepare: impl FnMut(Need) -> Result<Prepared<B>, Refused>,
    ) -> Result<Self, Refused> {
        let what = what.into();
        for rows in shape.wide_verify_rows() {
            let op = asks.ask(
                format!("{what} wide rows {rows}"),
                exact::need(rows, exact::every_phase(rows)),
                &mut prepare,
            )?;
            self.wide.push((rows, op));
        }
        Ok(self)
    }

    /// Adds a one-row op, which [`Self::at`] returns for one row.
    pub(crate) fn with_one_row(
        mut self,
        asks: &Asks,
        what: impl Into<String>,
        prepare: impl FnMut(Need) -> Result<Prepared<B>, Refused>,
    ) -> Result<Self, Refused> {
        self.one = Some(asks.ask(
            format!("{} one row", what.into()),
            exact::need(1, exact::every_phase(1)),
            prepare,
        )?);
        Ok(self)
    }

    pub(crate) fn at(&self, rows: u32) -> &Prepared<B> {
        if rows == 1
            && let Some(one) = &self.one
        {
            one
        } else if rows <= DECODE_BATCH {
            &self.small
        } else {
            &self.every
        }
    }

    /// [`Self::at`] for `phase`'s rows, except that decode and verify rows
    /// up to `SMALL_DRAFT_ROWS` take the draft-rows op, and wider ones the
    /// narrowest wide op that holds them. A drafter forward of up to
    /// `SMALL_DRAFT_ITEMS` items lowers as a verify phase. Prefill keeps
    /// its plans at every row count.
    pub(crate) fn at_phase(&self, phase: Phase) -> &Prepared<B> {
        let rows = phase.rows();
        if phase.prefill() || rows <= DECODE_BATCH {
            self.at(rows)
        } else if rows <= self.draft_rows {
            &self.draft
        } else if let Some((_, wide)) = self.wide.iter().find(|(bucket, _)| rows <= *bucket) {
            wide
        } else {
            self.at(rows)
        }
    }
}
