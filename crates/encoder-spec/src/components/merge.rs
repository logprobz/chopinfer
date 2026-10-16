//! `merge`: groups of rows become one wider row.

use serde::{Deserialize, Serialize};

use super::{Boundary, Component, Context, Data, Item, Typed, mul};
use crate::Patchify;
use crate::policy::{self, ParallelClass, Precision};
use crate::weights::Role;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ConsecutiveParams {
    pub rows: u32,
}

/// What fills the cells of a group that fall past the grid.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Pad {
    Zero,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SpaceToDepthParams {
    pub factor: u32,
    pub pad: Pad,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "op")]
pub enum Merge {
    /// Each `rows` consecutive rows read as one row: a reinterpretation, no
    /// launch. The rows must already come in whole groups.
    #[serde(rename = "merge.consecutive.v1")]
    Consecutive(ConsecutiveParams),
    /// Each `factor x factor` cell of the row-major patch grid gathered into
    /// one position-major row, zero rows where the cell runs past the grid.
    #[serde(rename = "merge.space-to-depth.v1")]
    SpaceToDepth(SpaceToDepthParams),
}

impl Typed for Merge {
    fn policies(&self) -> &'static [Precision] {
        policy::ANY
    }
}

impl Component for Merge {
    fn parallel(&self) -> ParallelClass {
        ParallelClass::GroupCoupled
    }

    fn output(&self, context: &Context, input: &Boundary) -> Result<Boundary, String> {
        let width = u64::from(input.width()?);
        if input.segmented {
            return Err("merges whole items; join the segments first (a group op)".into());
        }
        let element = match input.data {
            Data::Rows { element, .. } => element,
            Data::Codes { .. } => unreachable!("width() refused codes"),
        };
        let (group, output) = match self {
            Merge::Consecutive(params) => {
                let group = u64::from(params.rows);
                if group < 2 || input.multiple % group != 0 {
                    return Err(format!(
                        "rows come in multiples of {}, not of {group}; declare how an \
                         incomplete group fills (a group op)",
                        input.multiple
                    ));
                }
                (
                    group,
                    Boundary {
                        data: input.data.clone(),
                        run_rows: input.run_rows.div_ceil(group),
                        multiple: input.multiple / group,
                        grid: input.grid,
                        segmented: false,
                    },
                )
            }
            Merge::SpaceToDepth(params) => {
                if context.patchify != Some(Patchify::RowMajor) || !input.grid {
                    return Err("space-to-depth gathers from a row-major patch grid".into());
                }
                let group = mul(u64::from(params.factor), u64::from(params.factor))?;
                if group < 4 {
                    return Err("a space-to-depth cell is at least 2 x 2".into());
                }
                // A cell holds at least one patch, so there are no more cells
                // than patches.
                (
                    group,
                    Boundary {
                        multiple: 1,
                        ..input.clone()
                    },
                )
            }
        };
        let width = mul(width, group)?;
        let width = u32::try_from(width).map_err(|_| format!("merged width {width} overflows"))?;
        Ok(output.with(Data::Rows { width, element }))
    }

    fn roles(&self, _: &Context, _: &Boundary) -> Result<Vec<Role>, String> {
        Ok(Vec::new())
    }

    fn item(&self, item: Item) -> Result<Item, String> {
        match self {
            Merge::Consecutive(params) => {
                let group = u64::from(params.rows);
                if item.rows % group != 0 {
                    return Err(format!(
                        "{} rows are not whole groups of {group}",
                        item.rows
                    ));
                }
                let side = Some(group.isqrt()).filter(|side| side * side == group);
                let grid = match (item.grid, side) {
                    (Some([t, h, w]), Some(side)) if h % side == 0 && w % side == 0 => {
                        Some([t, h / side, w / side])
                    }
                    _ => None,
                };
                Ok(Item {
                    rows: item.rows / group,
                    grid,
                })
            }
            Merge::SpaceToDepth(params) => {
                let [t, h, w] = item.grid.ok_or("space-to-depth needs a grid")?;
                let factor = u64::from(params.factor);
                let (h, w) = (h.div_ceil(factor), w.div_ceil(factor));
                Ok(Item {
                    rows: mul(mul(t, h)?, w)?,
                    grid: Some([t, h, w]),
                })
            }
        }
    }
}
