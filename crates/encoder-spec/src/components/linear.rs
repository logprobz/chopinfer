//! `linear`: one GEMM `rows x in -> rows x out` with its epilogue.

use serde::{Deserialize, Serialize};

use super::{Boundary, Component, Context, Data, Element, Gelu, Typed};
use crate::policy::{self, ParallelClass, Precision};
use crate::weights::Role;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LinearParams {
    /// Output columns: the weight's rows.
    pub out: u32,
    pub bias: bool,
    /// A GELU applied in the epilogue.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gelu: Option<Gelu>,
    /// `linear.pairs.v1` only: `bf16` when this stage writes the rows the
    /// language model reads; F32 otherwise.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output: Option<Element>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "op")]
pub enum Linear {
    /// BF16 operands and output; the epilogue runs on the F32 accumulator.
    #[serde(rename = "linear.bf16.v1")]
    Bf16(LinearParams),
    /// As `linear.bf16.v1`, but the linear output rounds to BF16 before the
    /// activation, as unfused PyTorch does.
    #[serde(rename = "linear.bf16-round-linear.v1")]
    Bf16RoundLinear(LinearParams),
    /// F32 activation as BF16 hi+lo pairs, two MMAs per weight fragment,
    /// F32 accumulate.
    #[serde(rename = "linear.pairs.v1")]
    Pairs(LinearParams),
}

impl Linear {
    pub fn params(&self) -> &LinearParams {
        match self {
            Linear::Bf16(params) | Linear::Bf16RoundLinear(params) | Linear::Pairs(params) => {
                params
            }
        }
    }
}

impl Typed for Linear {
    fn policies(&self) -> &'static [Precision] {
        match self {
            Linear::Bf16(_) | Linear::Bf16RoundLinear(_) => policy::LEGACY,
            Linear::Pairs(_) => policy::PAIRS,
        }
    }
}

impl Component for Linear {
    fn parallel(&self) -> ParallelClass {
        ParallelClass::RowLocal
    }

    fn output(&self, context: &Context, input: &Boundary) -> Result<Boundary, String> {
        input.width()?;
        let params = self.params();
        if params.out == 0 {
            return Err("`out` is zero".into());
        }
        let element = match (self, params.output) {
            (Linear::Bf16RoundLinear(params), _) if params.gelu.is_none() => {
                return Err(
                    "rounding the linear output matters only before an activation; set `gelu`"
                        .into(),
                );
            }
            (Linear::Bf16(_) | Linear::Bf16RoundLinear(_), Some(_)) => {
                return Err("`output` is BF16 by the variant; remove it".into());
            }
            (_, Some(Element::F32)) => return Err("`output: f32` is the default; remove it".into()),
            (_, Some(Element::Bf16)) => Element::Bf16,
            (_, None) => context.rows_element(),
        };
        Ok(input.with(Data::Rows {
            width: params.out,
            element,
        }))
    }

    fn roles(&self, _: &Context, input: &Boundary) -> Result<Vec<Role>, String> {
        let params = self.params();
        let width = u64::from(input.width()?);
        let mut roles = vec![Role::new("weight", vec![u64::from(params.out), width])];
        if params.bias {
            roles.push(Role::new("bias", vec![u64::from(params.out)]));
        }
        Ok(roles)
    }
}
