//! `projector`: a feed-forward stage over each row with no residual, built
//! from slot components.

use serde::{Deserialize, Serialize};

use super::{Boundary, Component, Context, Data, Element, Gelu, Mlp, Norm, Typed};
use crate::policy::{self, ParallelClass, Precision};
use crate::weights::Role;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NormGeluMlpParams {
    pub norm: Norm,
    pub gelu: Gelu,
    pub mlp: Mlp,
    /// `bf16` when this stage writes the rows the language model reads,
    /// rounded once in the down projection; F32 otherwise.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output: Option<Element>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "op")]
pub enum Projector {
    /// `mlp(gelu(norm(x)))`: the norm's F32 rows, then GELU as its own F32
    /// row pass (not a GEMM epilogue), then the MLP slot's GEMMs and row
    /// pass, with no residual add. Weight roles `norm.*` and the MLP's.
    #[serde(rename = "projector.norm-gelu-mlp.v1")]
    NormGeluMlp(Box<NormGeluMlpParams>),
}

impl Projector {
    pub fn params(&self) -> &NormGeluMlpParams {
        match self {
            Projector::NormGeluMlp(params) => params,
        }
    }
}

impl Typed for Projector {
    fn policies(&self) -> &'static [Precision] {
        policy::PAIRS
    }
}

impl Component for Projector {
    fn parallel(&self) -> ParallelClass {
        ParallelClass::RowLocal
    }

    fn output(&self, context: &Context, input: &Boundary) -> Result<Boundary, String> {
        let width = input.width()?;
        let params = self.params();
        params.norm.check(context)?;
        params.mlp.check(context)?;
        let element = match params.output {
            Some(Element::F32) => return Err("`output: f32` is the default; remove it".into()),
            Some(Element::Bf16) => Element::Bf16,
            None => context.rows_element(),
        };
        Ok(input.with(Data::Rows { width, element }))
    }

    fn roles(&self, _: &Context, input: &Boundary) -> Result<Vec<Role>, String> {
        let params = self.params();
        let width = input.width()?;
        let mut roles = params.norm.slot_roles("norm", width);
        roles.extend(params.mlp.roles(width)?);
        Ok(roles)
    }
}
