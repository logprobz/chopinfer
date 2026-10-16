//! `layout`: how an encoder's rows enter the language model.

use serde::{Deserialize, Serialize};

use super::{Boundary, Component, Context, Data, Element, Item, Typed, mul};
use crate::policy::{self, ParallelClass, Precision};
use crate::weights::Role;

/// How the language model positions the media rows.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum PositionRule {
    /// One position per row, as text.
    #[serde(rename = "sequential")]
    Sequential,
    /// Three-axis (time, height, width) positions from the merged grid.
    #[serde(rename = "grid-3axis.v1")]
    Grid3Axis,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LayoutParams {
    /// The language model's hidden size: the width of the rows it reads.
    pub hidden: u32,
    pub positions: PositionRule,
    /// Whether the language model reads a per-row media mask (the rows'
    /// `source_rows >= 0`).
    pub media_mask: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "op")]
pub enum Layout {
    /// The encoder's rows, in order, between the modality's text delimiters.
    #[serde(rename = "layout.tokens.v1")]
    Tokens(LayoutParams),
    /// Learned `start` and `end` rows around the grid, and a learned
    /// `newline` row after every grid row including the last:
    /// `h * (w + 1) + 2` rows per temporal grid.
    #[serde(rename = "layout.learned-delimiters.v1")]
    LearnedDelimiters(LayoutParams),
}

impl Layout {
    pub fn params(&self) -> &LayoutParams {
        match self {
            Layout::Tokens(params) | Layout::LearnedDelimiters(params) => params,
        }
    }
}

impl Typed for Layout {
    fn policies(&self) -> &'static [Precision] {
        policy::ANY
    }
}

impl Component for Layout {
    fn parallel(&self) -> ParallelClass {
        ParallelClass::RowLocal
    }

    fn output(&self, _: &Context, input: &Boundary) -> Result<Boundary, String> {
        let params = self.params();
        let expected = Data::Rows {
            width: params.hidden,
            element: Element::Bf16,
        };
        if input.data != expected {
            return Err(format!(
                "the language model reads bf16 rows of width {} but the last stage writes {}; \
                 rows round to BF16 once, at this boundary",
                params.hidden, input.data
            ));
        }
        if input.segmented {
            return Err("the language model reads whole items; join the segments first".into());
        }
        let needs_grid = params.positions == PositionRule::Grid3Axis
            || matches!(self, Layout::LearnedDelimiters(_));
        if needs_grid && !input.grid {
            return Err("this layout needs rows that follow a grid".into());
        }
        Ok(input.clone())
    }

    fn roles(&self, _: &Context, _: &Boundary) -> Result<Vec<Role>, String> {
        let hidden = vec![u64::from(self.params().hidden)];
        Ok(match self {
            Layout::Tokens(_) => Vec::new(),
            Layout::LearnedDelimiters(_) => ["start", "end", "newline"]
                .map(|name| Role::new(name, hidden.clone()))
                .to_vec(),
        })
    }

    fn item(&self, item: Item) -> Result<Item, String> {
        match self {
            Layout::Tokens(_) => Ok(item),
            Layout::LearnedDelimiters(_) => {
                let [t, h, w] = item.grid.ok_or("learned delimiters need a grid")?;
                let per_grid = mul(h, w + 1)? + 2;
                Ok(Item {
                    rows: mul(t, per_grid)?,
                    grid: None,
                })
            }
        }
    }
}
