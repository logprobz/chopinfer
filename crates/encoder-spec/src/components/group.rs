//! `group`: joins an item's segments and pads it to whole groups of rows.

use serde::{Deserialize, Serialize};

use super::{Boundary, Component, Context, Item, Typed};
use crate::policy::{self, ParallelClass, Precision};
use crate::weights::Role;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GroupParams {
    pub rows: u32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "op")]
pub enum Group {
    /// Joins the item's segments in order, then pads the item to a multiple
    /// of `rows` rows by repeating its last row (or code vector): row `r`
    /// reads row `min(r, n - 1)`. Later stages see whole items in whole
    /// groups, in runs of at most the capacity's whole groups.
    #[serde(rename = "group.repeat-last.v1")]
    RepeatLast(GroupParams),
}

impl Group {
    pub fn rows(&self) -> u64 {
        let Group::RepeatLast(params) = self;
        u64::from(params.rows)
    }

    /// The source row of each padded row, for an item of `rows` rows.
    pub fn sources(&self, rows: u64) -> Vec<u64> {
        let padded = rows.div_ceil(self.rows()) * self.rows();
        (0..padded)
            .map(|row| row.min(rows.saturating_sub(1)))
            .collect()
    }
}

impl Typed for Group {
    fn policies(&self) -> &'static [Precision] {
        policy::ANY
    }
}

impl Component for Group {
    fn parallel(&self) -> ParallelClass {
        ParallelClass::GroupCoupled
    }

    fn output(&self, context: &Context, input: &Boundary) -> Result<Boundary, String> {
        let rows = self.rows();
        if rows < 2 {
            return Err("a group has at least two rows".into());
        }
        let run_rows = context.capacity_rows / rows * rows;
        if run_rows == 0 {
            return Err(format!(
                "capacity_rows {} holds no whole group of {rows}",
                context.capacity_rows
            ));
        }
        Ok(Boundary {
            data: input.data.clone(),
            run_rows,
            multiple: rows,
            grid: false,
            segmented: false,
        })
    }

    fn roles(&self, _: &Context, _: &Boundary) -> Result<Vec<Role>, String> {
        Ok(Vec::new())
    }

    fn joins_segments(&self) -> bool {
        true
    }

    fn item(&self, item: Item) -> Result<Item, String> {
        Ok(Item {
            rows: item.rows.div_ceil(self.rows()) * self.rows(),
            grid: None,
        })
    }
}
