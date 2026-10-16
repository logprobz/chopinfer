//! `position`: an additive position term per row.

use serde::{Deserialize, Serialize};

use super::{Boundary, Component, Context, Typed, mul};
use crate::policy::{self, ParallelClass, Precision};
use crate::weights::Role;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LearnedGridParams {
    /// The learned table's grid, `[rows, columns]`.
    pub table: [u32; 2],
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "op")]
pub enum Position {
    /// A learned grid resampled to the patch grid by bilinear interpolation
    /// with `align_corners=True`, taps and weights in F32, added in F32.
    #[serde(rename = "position.learned-grid.bilinear-align-corners.f32.v1")]
    LearnedGridBilinearF32(LearnedGridParams),
}

impl Typed for Position {
    fn policies(&self) -> &'static [Precision] {
        policy::PAIRS
    }
}

impl Component for Position {
    fn parallel(&self) -> ParallelClass {
        ParallelClass::RowLocal
    }

    fn output(&self, _: &Context, input: &Boundary) -> Result<Boundary, String> {
        input.width()?;
        let Position::LearnedGridBilinearF32(params) = self;
        if params.table.contains(&0) {
            return Err("`table` has an empty side".into());
        }
        if !input.grid {
            return Err("a learned grid needs rows that follow a patch grid".into());
        }
        Ok(input.clone())
    }

    fn roles(&self, _: &Context, input: &Boundary) -> Result<Vec<Role>, String> {
        let Position::LearnedGridBilinearF32(params) = self;
        let cells = mul(u64::from(params.table[0]), u64::from(params.table[1]))?;
        Ok(vec![Role::new(
            "table",
            vec![cells, u64::from(input.width()?)],
        )])
    }
}
