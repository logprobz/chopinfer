//! `pad`: rows widened with zero columns, so a GEMM whose K must be a
//! multiple of its tile reads them. Zero columns times zero weight columns
//! add exact zeros, so the GEMM's result is the unpadded one.

use serde::{Deserialize, Serialize};

use super::{Boundary, Component, Context, Data, Element, Typed};
use crate::policy::{self, ParallelClass, Precision};
use crate::weights::Role;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ZeroColumnsParams {
    /// The output width: the input's columns, then zeros up to this many.
    pub columns: u32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "op")]
pub enum Pad {
    /// F32 rows copied unchanged into rows of `columns`, zero past the
    /// input's width.
    #[serde(rename = "pad.zero-columns.v1")]
    ZeroColumns(ZeroColumnsParams),
}

impl Pad {
    pub fn columns(&self) -> u32 {
        match self {
            Pad::ZeroColumns(params) => params.columns,
        }
    }
}

impl Typed for Pad {
    fn policies(&self) -> &'static [Precision] {
        policy::PAIRS
    }
}

impl Component for Pad {
    fn parallel(&self) -> ParallelClass {
        ParallelClass::RowLocal
    }

    fn output(&self, _: &Context, input: &Boundary) -> Result<Boundary, String> {
        let width = input.width()?;
        let Data::Rows {
            element: Element::F32,
            ..
        } = input.data
        else {
            return Err(format!("pads F32 rows, not {}", input.data));
        };
        let columns = self.columns();
        if columns <= width {
            return Err(format!(
                "pads to {columns} columns, not wider than its input's {width}"
            ));
        }
        Ok(input.with(Data::Rows {
            width: columns,
            element: Element::F32,
        }))
    }

    fn roles(&self, _: &Context, _: &Boundary) -> Result<Vec<Role>, String> {
        Ok(Vec::new())
    }
}
