//! Layer selectors use the owning template's current index, never a global default.
use crate::{Dim, Error, Geometry};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum CurrentLayer {
    #[serde(rename = "current")]
    Current,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum LayerSelector {
    Current(CurrentLayer),
    Index(Dim),
}

impl LayerSelector {
    pub fn current() -> Self {
        Self::Current(CurrentLayer::Current)
    }
    pub fn resolve(&self, geometry: &Geometry, current: Option<u64>) -> Result<u64, Error> {
        let context = || {
            current.ok_or_else(|| {
                Error("current layer selector needs an owning layer template".into())
            })
        };
        match self {
            Self::Current(_) => context(),
            Self::Index(dim) => {
                if dim.uses_current() && current.is_none() {
                    context()?;
                }
                let mut geometry = geometry.clone();
                if let Some(current) = current {
                    geometry.insert("current".into(), current);
                }
                dim.resolve_offset(&geometry)
            }
        }
    }
}

impl Dim {
    fn uses_current(&self) -> bool {
        use crate::DimExpr;
        match self {
            Self::Literal(_) => false,
            Self::Key(key) => key == "current",
            Self::Expr(expression) => match expression.as_ref() {
                DimExpr::Sum { terms } => terms.iter().any(Self::uses_current),
                DimExpr::Product { factors } => factors.iter().any(Self::uses_current),
                DimExpr::Quotient {
                    numerator,
                    denominator,
                } => numerator.uses_current() || denominator.uses_current(),
            },
        }
    }
}
