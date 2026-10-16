//! `conv1d`: a 1D convolution over rows as im2col plus a GEMM whose linear
//! variant the precision policy picks.

use serde::{Deserialize, Serialize};

use super::{Boundary, Component, Context, Data, Gelu, Item, Typed, mul};
use crate::policy::{self, ParallelClass, Precision};
use crate::weights::Role;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Conv1dParams {
    pub kernel: u32,
    pub stride: u32,
    /// Zero input rows before the first and after the last.
    pub pad: [u32; 2],
    pub out: u32,
    pub bias: bool,
    /// A GELU in the GEMM epilogue.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gelu: Option<Gelu>,
    /// A GELU as a separate row pass over the stored GEMM output.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gelu_rows: Option<Gelu>,
    /// `bf16-legacy.v1` only: the GEMM output rounds to BF16 before `gelu`.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub round_linear: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "op")]
pub enum Conv1d {
    /// Output row `r` reads padded input rows `r*stride .. r*stride +
    /// kernel`, taps before channels: `(rows + pad - kernel) / stride + 1`
    /// output rows.
    #[serde(rename = "conv1d.im2col.v1")]
    Im2col(Conv1dParams),
}

impl Conv1d {
    pub fn params(&self) -> &Conv1dParams {
        match self {
            Conv1d::Im2col(params) => params,
        }
    }

    /// Output rows for `rows` input rows.
    pub fn rows(&self, rows: u64) -> Result<u64, String> {
        let params = self.params();
        let padded = rows + u64::from(params.pad[0]) + u64::from(params.pad[1]);
        if rows == 0 || padded < u64::from(params.kernel) {
            return Err(format!(
                "{rows} rows are fewer than kernel {}",
                params.kernel
            ));
        }
        Ok((padded - u64::from(params.kernel)) / u64::from(params.stride) + 1)
    }
}

impl Typed for Conv1d {
    fn policies(&self) -> &'static [Precision] {
        policy::ANY
    }
}

impl Component for Conv1d {
    fn parallel(&self) -> ParallelClass {
        ParallelClass::RowHalo
    }

    fn output(&self, context: &Context, input: &Boundary) -> Result<Boundary, String> {
        input.width()?;
        let params = self.params();
        if params.out == 0 || params.kernel == 0 || params.stride == 0 {
            return Err("`out`, `kernel` and `stride` are positive".into());
        }
        if params.pad.iter().any(|&pad| pad >= params.kernel) {
            return Err(format!(
                "padding {:?} reaches past kernel {}",
                params.pad, params.kernel
            ));
        }
        if params.stride > params.kernel {
            return Err(format!(
                "stride {} skips input rows of kernel {}",
                params.stride, params.kernel
            ));
        }
        if params.gelu.is_some() && params.gelu_rows.is_some() {
            return Err("`gelu` and `gelu_rows` exclude each other".into());
        }
        if params.round_linear {
            if context.precision != Precision::Bf16Legacy {
                return Err("`round_linear` is allowed only under bf16-legacy.v1".into());
            }
            if params.gelu.is_none() {
                return Err("`round_linear` needs a fused `gelu`".into());
            }
        }
        Ok(Boundary {
            data: Data::Rows {
                width: params.out,
                element: context.rows_element(),
            },
            run_rows: self.rows(input.run_rows)?,
            multiple: 1,
            grid: false,
            segmented: input.segmented,
        })
    }

    fn roles(&self, _: &Context, input: &Boundary) -> Result<Vec<Role>, String> {
        let params = self.params();
        let width = u64::from(input.width()?);
        let mut roles = vec![Role::new(
            "weight",
            vec![u64::from(params.out), u64::from(params.kernel), width],
        )];
        mul(u64::from(params.kernel), width)?;
        if params.bias {
            roles.push(Role::new("bias", vec![u64::from(params.out)]));
        }
        Ok(roles)
    }

    fn item(&self, item: Item) -> Result<Item, String> {
        Ok(Item {
            rows: self.rows(item.rows)?,
            grid: None,
        })
    }
}
