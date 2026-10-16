//! `rotary`: the rotary position side input of attention, a transformer
//! slot. Rotates q and k in place before the scores. Models register their
//! own rotary variants through `crate::registry`.

use serde::de::Error as _;
use serde::ser::Error as _;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::{Map, Value};

use super::{Registered, Typed, Unresolved, Variant, f32_positive, op_of};
use crate::id::Kind;
use crate::policy::{self, Precision};

/// How rotated dimensions pair up.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Style {
    /// Dimension `d` pairs with `d + rotary_dims / 2`.
    Neox,
}

/// How the host computes the cos and sin table.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum Table {
    /// The reference's F32 math: F32 inverse frequencies and angles, cos and
    /// sin of the F32 angle in f64, rounded to F32.
    #[serde(rename = "table.hf-f32-angle.v1")]
    HfF32Angle,
}

/// Which position a row rotates by.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Positions {
    /// The row's index inside its attention sequence.
    WithinSequence,
}

/// What a row's rotation angle follows.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Axes {
    /// The row's (row, column) in its patch grid: the first half of the
    /// rotated pairs follow the row, the second half the column.
    Grid2d,
    /// The row's index inside its attention sequence.
    Sequence,
}

/// What attention needs of any rotary variant, shared or registered.
pub trait RotaryComponent: Variant {
    /// The head dimensions the rotation touches.
    fn rotary_dims(&self) -> Result<u64, String>;
    /// The position each row rotates by.
    fn axes(&self) -> Axes;
    /// Checks the variant's own parameters, in the precision its table and
    /// kernel compute with.
    fn check(&self) -> Result<(), String>;
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Axial2dParams {
    /// Frequencies per axis; the first `2 * pairs_per_axis` rotated angles
    /// follow the patch row, the next the patch column.
    pub pairs_per_axis: u32,
    /// Rounded to F32 for the inverse frequencies.
    pub theta: f64,
    pub style: Style,
    pub table: Table,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OneDParams {
    /// Frequencies; the first `2 * pairs` head dimensions rotate.
    pub pairs: u32,
    /// Rounded to F32 for the inverse frequencies.
    pub theta: f64,
    pub style: Style,
    pub table: Table,
    pub positions: Positions,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "op")]
pub enum Rotary {
    /// 2D axial rotary over the patch grid, F32 rows, pairs into attention.
    #[serde(rename = "rotary.axial-2d.v1")]
    Axial2d(Axial2dParams),
    /// 1D rotary over one position per row, F32 rows, pairs into attention.
    #[serde(rename = "rotary.1d.v1")]
    OneD(OneDParams),
}

impl Typed for Rotary {
    fn policies(&self) -> &'static [Precision] {
        policy::PAIRS
    }
}

impl RotaryComponent for Rotary {
    fn rotary_dims(&self) -> Result<u64, String> {
        Ok(match self {
            Rotary::Axial2d(params) => 4 * u64::from(params.pairs_per_axis),
            Rotary::OneD(params) => 2 * u64::from(params.pairs),
        })
    }

    fn axes(&self) -> Axes {
        match self {
            Rotary::Axial2d(_) => Axes::Grid2d,
            Rotary::OneD(params) => match params.positions {
                Positions::WithinSequence => Axes::Sequence,
            },
        }
    }

    fn check(&self) -> Result<(), String> {
        let theta = match self {
            Rotary::Axial2d(params) => params.theta,
            Rotary::OneD(params) => params.theta,
        };
        let theta = f32_positive(&format!("`{}`: theta", self.id()), theta)?;
        if theta <= 1.0 {
            return Err(format!("`{}`: theta {theta} is not above 1", self.id()));
        }
        Ok(())
    }
}

/// The rotary slot: a shared variant or a model's registered one.
#[derive(Clone, Debug, PartialEq)]
pub enum RotarySlot {
    Shared(Rotary),
    Registered(Registered<dyn RotaryComponent>),
    Unresolved(Unresolved),
}

impl RotarySlot {
    pub fn component(&self) -> Result<&dyn RotaryComponent, String> {
        match self {
            RotarySlot::Shared(rotary) => Ok(rotary),
            RotarySlot::Registered(registered) => Ok(registered.component.as_ref()),
            RotarySlot::Unresolved(unresolved) => {
                Err(format!("`{}` is not registered", unresolved.id))
            }
        }
    }
}

impl Serialize for RotarySlot {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            RotarySlot::Unresolved(unresolved) => unresolved.value.serialize(serializer),
            other => other
                .component()
                .map_err(S::Error::custom)?
                .to_value()
                .serialize(serializer),
        }
    }
}

impl<'de> Deserialize<'de> for RotarySlot {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let mut object = Map::deserialize(deserializer)?;
        for value in object.values_mut() {
            crate::registry::normalize_json_floats(value).map_err(D::Error::custom)?;
        }
        let (op, kind, namespaced) = op_of(&object).map_err(D::Error::custom)?;
        if kind != Kind::Rotary {
            return Err(D::Error::custom(format!(
                "`{op}` is not a rotary component"
            )));
        }
        if namespaced {
            return Ok(RotarySlot::Unresolved(Unresolved {
                id: op,
                value: object,
            }));
        }
        serde_json::from_value(Value::Object(object))
            .map(RotarySlot::Shared)
            .map_err(D::Error::custom)
    }
}
