//! `cast`: a change of row element with no other arithmetic.

use serde::{Deserialize, Serialize};

use super::{Boundary, Component, Context, Data, Element, Typed};
use crate::policy::{self, ParallelClass, Precision};
use crate::weights::Role;

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NoParams {}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "op")]
pub enum Cast {
    /// F32 rows rounded to BF16, round to nearest even: the one rounding
    /// `f32-pairs.v1` allows, at the rows the language model reads, when
    /// the last op cannot fuse it.
    #[serde(rename = "cast.bf16.v1")]
    Bf16(NoParams),
}

impl Typed for Cast {
    fn policies(&self) -> &'static [Precision] {
        policy::PAIRS
    }
}

impl Component for Cast {
    fn parallel(&self) -> ParallelClass {
        ParallelClass::RowLocal
    }

    fn output(&self, _: &Context, input: &Boundary) -> Result<Boundary, String> {
        let width = input.width()?;
        Ok(input.with(Data::Rows {
            width,
            element: Element::Bf16,
        }))
    }

    fn roles(&self, _: &Context, _: &Boundary) -> Result<Vec<Role>, String> {
        Ok(Vec::new())
    }
}
