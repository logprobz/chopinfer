//! Prepare small rows separately when a row op otherwise keeps a prefill plan.
use kernel_core::backend::{Need, Prepared, Refused};
use model_deepseek41_contract::serving::DECODE_BATCH;

use super::{Asks, exact};

pub(crate) struct RowBuckets<B> {
    every: Prepared<B>,
    small: Prepared<B>,
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
        })
    }

    pub(crate) fn at(&self, rows: u32) -> &Prepared<B> {
        if rows <= DECODE_BATCH {
            &self.small
        } else {
            &self.every
        }
    }
}
