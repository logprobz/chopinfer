//! Share each staged selected-key tile across a fixed number of heads.
use super::Entry;
use kernel_build::EntrySpec;
use kernel_core::backend::{Launch, View};
use kernel_core::contract::{LaunchPlan, Refusal};

#[derive(Clone, Debug)]
pub struct GroupedEntry {
    pub cache: Entry,
    pub heads: u32,
}

impl GroupedEntry {
    pub fn validate(&self) -> Result<(), Refusal> {
        self.cache.validate()?;
        if !matches!(self.heads, 1 | 2 | 4 | 8) {
            return Err(Refusal("cached head group must be one, two, four or eight"));
        }
        Ok(())
    }

    pub fn symbol(&self) -> String {
        format!(
            "chop_latent_sources_{}_cached_h{}_profile{}_v1",
            self.cache.source.name,
            self.heads,
            u32::from(self.cache.profile)
        )
    }

    pub fn plan(&self, rows: u32) -> Result<LaunchPlan, Refusal> {
        self.validate()?;
        let mut plan = self.cache.plan(rows)?;
        let shape = super::sources::shape(self.cache.source.query)?;
        plan.grid[1] = shape.heads.div_ceil(self.heads);
        crate::backend::check_plan(&plan)?;
        Ok(plan)
    }

    pub fn spec(&self) -> Result<EntrySpec, Refusal> {
        self.validate()?;
        let mut spec = self.cache.spec_group(None, self.heads)?;
        spec.symbol = self.symbol();
        Ok(spec)
    }

    /// Reuse the exact producer, buffers and split layout. Only the consumer's
    /// head grouping changes; the caller retains the original ordered merge.
    pub fn launches(
        &self,
        reference: &Launch,
        decoded: View,
        valid: View,
        stamps: Option<View>,
    ) -> Result<Vec<Launch>, Refusal> {
        self.validate()?;
        let mut calls = self.cache.launches(reference, decoded, valid, stamps)?;
        calls[1].symbol = self.symbol();
        calls[1].plan = self.plan(reference.rows)?;
        Ok(calls)
    }
}
