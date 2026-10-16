//! `reorder`: a row permutation around a transformer block, undone after
//! it.

use serde::{Deserialize, Serialize};

use super::{Boundary, Context, Typed, Variant};
use crate::Patchify;
use crate::policy::{self, Precision};

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NoParams {}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "op")]
pub enum Reorder {
    /// Merge-block rows put in column order: merge blocks walked down each
    /// column of blocks, the 2 x 2 rows of a block kept in order.
    #[serde(rename = "reorder.merge-group-columns.v1")]
    MergeGroupColumns(NoParams),
}

impl Typed for Reorder {
    fn policies(&self) -> &'static [Precision] {
        policy::ANY
    }
}

impl Reorder {
    pub fn check(&self, context: &Context, input: &Boundary) -> Result<(), String> {
        context.admit(&self.id(), Typed::policies(self))?;
        if context.patchify != Some(Patchify::MergeBlocks) || !input.grid {
            return Err(format!("`{}` needs merge-block grid rows", self.id()));
        }
        Ok(())
    }
}
