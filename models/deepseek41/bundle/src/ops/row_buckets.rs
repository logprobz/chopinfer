//! Prepare small rows separately when a row op otherwise keeps a prefill plan.
use kernel_core::backend::{Need, Prepared, Refused};
use model_deepseek41_contract::serving::DECODE_BATCH;

use super::{Asks, exact};

pub(crate) struct RowBuckets<B> {
    every: Prepared<B>,
    small: Prepared<B>,
    /// A one-row op, where one row has an entry of its own.
    one: Option<Prepared<B>>,
}
impl<B> RowBuckets<B> {
    pub(crate) fn prepare(
        asks: &Asks,
        what: impl Into<String>,
        every: Need,
        mut prepare: impl FnMut(Need) -> Result<Prepared<B>, Refused>,
    ) -> Result<Self, Refused> {
        let what = what.into();
        let large = asks.ask(&what, every, &mut prepare)?;
        let small = asks.ask(
            format!("{what} small rows"),
            exact::need(DECODE_BATCH, exact::every_phase(DECODE_BATCH)),
            prepare,
        )?;
        Ok(Self {
            every: large,
            small,
            one: None,
        })
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
}
