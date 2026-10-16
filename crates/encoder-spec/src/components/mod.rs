//! Typed components. One module per kind; each kind's shared variants are a
//! closed set, written in a declaration as `{"op": "<id>", ...}` with the
//! variant's numeric parameters as data. A variant's numerics are fixed by
//! its id: changing them means a new id, never an edit. Model-namespaced
//! variants come from providers (`crate::registry`) and never appear here.

pub mod attention;
pub mod cast;
pub mod conv1d;
pub mod group;
pub mod layout;
pub mod linear;
pub mod merge;
pub mod mlp;
pub mod norm;
pub mod pad;
pub mod position;
pub mod projector;
pub mod reorder;
pub mod rotary;
pub mod transformer;

use std::fmt;
use std::sync::Arc;

use serde::de::{DeserializeOwned, Error as _};
use serde::ser::Error as _;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::{Map, Value};

use crate::id::{self, Kind};
use crate::policy::{ParallelClass, Precision};
use crate::registry::Provider;
use crate::weights::{Role, Weights};
use crate::{Patchify, Sequences};

pub use attention::Attention;
pub use cast::Cast;
pub use conv1d::Conv1d;
pub use group::Group;
pub use layout::Layout;
pub use linear::Linear;
pub use merge::Merge;
pub use mlp::Mlp;
pub use norm::Norm;
pub use pad::Pad;
pub use position::Position;
pub use projector::Projector;
pub use reorder::Reorder;
pub use rotary::{Rotary, RotaryComponent, RotarySlot};
pub use transformer::Transformer;

/// An element type of rows or payloads.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Element {
    F32,
    Bf16,
}

impl Element {
    /// Bytes per value.
    pub fn bytes(self) -> u64 {
        match self {
            Element::F32 => 4,
            Element::Bf16 => 2,
        }
    }
}

/// A GELU form.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Gelu {
    Erf,
    Tanh,
}

/// What each row holds between stages.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Data {
    /// `width` values of `element`.
    Rows { width: u32, element: Element },
    /// One code per entry, entry `s` below `bounds[s]`.
    Codes { bounds: Vec<u32> },
}

impl Data {
    /// Bytes per row: values of the element, or one 32-bit code per entry.
    pub fn row_bytes(&self) -> u64 {
        match self {
            Data::Rows { width, element } => u64::from(*width) * element.bytes(),
            Data::Codes { bounds } => bounds.len() as u64 * 4,
        }
    }
}

impl fmt::Display for Data {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Data::Rows { width, element } => {
                write!(formatter, "{} rows of width {width}", op(element))
            }
            Data::Codes { bounds } => write!(formatter, "{} codes per row", bounds.len()),
        }
    }
}

/// The rows between two stages: their data, and the bounds that size runs.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Boundary {
    pub data: Data,
    /// The most rows one run holds here.
    pub run_rows: u64,
    /// Every item's and every run's rows are a multiple of this.
    pub multiple: u64,
    /// The rows follow a patch grid.
    pub grid: bool,
    /// The rows are still split into the declaration's segments.
    pub segmented: bool,
}

impl Boundary {
    /// The width of rows, refusing codes.
    pub fn width(&self) -> Result<u32, String> {
        match self.data {
            Data::Rows { width, .. } => Ok(width),
            Data::Codes { .. } => Err(format!("expects rows but receives {}", self.data)),
        }
    }

    /// The same rows holding `data`.
    pub fn with(&self, data: Data) -> Boundary {
        Boundary {
            data,
            ..self.clone()
        }
    }
}

/// One item's (or one segment's) rows, for exact geometry.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Item {
    pub rows: u64,
    /// `[t, h, w]` when the rows follow a grid.
    pub grid: Option<[u64; 3]>,
}

/// What a component may consult about the declaration around it.
pub struct Context<'a> {
    pub model: &'a str,
    pub precision: Precision,
    pub patchify: Option<Patchify>,
    pub sequences: &'a Sequences,
    pub capacity_rows: u64,
}

impl Context<'_> {
    /// Admits a component with id `id` whose numerics are defined under
    /// `policies`: the declaration's policy must be one of them, and a
    /// model-namespaced variant must belong to the declaration's model.
    pub fn admit(&self, id: &str, policies: &[Precision]) -> Result<(), String> {
        let parsed = id::parse(id)?;
        if let Some(namespace) = parsed.namespace
            && namespace != self.model
        {
            return Err(format!(
                "`{id}` belongs to model {namespace}; model {} cannot use it",
                self.model
            ));
        }
        if !policies.contains(&self.precision) {
            return Err(format!(
                "`{id}` is not defined under precision policy {}",
                op(&self.precision)
            ));
        }
        Ok(())
    }

    /// The element rows are stored in between ops under the policy.
    pub fn rows_element(&self) -> Element {
        match self.precision {
            Precision::F32Pairs => Element::F32,
            Precision::Bf16Legacy => Element::Bf16,
        }
    }
}

/// The id a component serializes with, or a serialized enum's name.
pub fn op<T: Serialize + ?Sized>(component: &T) -> String {
    match serde_json::to_value(component) {
        Ok(Value::Object(object)) => object.get("op").and_then(Value::as_str).map(str::to_owned),
        Ok(Value::String(text)) => Some(text),
        _ => None,
    }
    .unwrap_or_default()
}

/// `a * b`, refusing overflow.
pub fn mul(a: u64, b: u64) -> Result<u64, String> {
    a.checked_mul(b)
        .ok_or_else(|| format!("{a} x {b} overflows"))
}

/// Checks that the F32 value a kernel or F32 host table receives for
/// `value` is finite and positive.
pub fn f32_positive(name: &str, value: f64) -> Result<f32, String> {
    let effective = value as f32;
    if effective.is_finite() && effective > 0.0 {
        Ok(effective)
    } else {
        Err(format!(
            "{name} {value:?} is {effective:?} in F32, not a finite positive value"
        ))
    }
}

/// What every variant states about itself, shared or registered.
pub trait Variant: fmt::Debug + Send + Sync {
    fn id(&self) -> String;
    /// The variant as written, `op` included: its typed parameters
    /// serialized again. The digest is taken over this.
    fn to_value(&self) -> Value;
    /// The precision policies whose rounding points this variant implements.
    fn policies(&self) -> &'static [Precision];
}

/// A variant whose parameters are a serde enum tagged by `op`.
pub trait Typed: Serialize + fmt::Debug + Send + Sync {
    fn policies(&self) -> &'static [Precision];
}

impl<T: Typed> Variant for T {
    fn id(&self) -> String {
        op(self)
    }

    fn to_value(&self) -> Value {
        serde_json::to_value(self).unwrap_or(Value::Null)
    }

    fn policies(&self) -> &'static [Precision] {
        Typed::policies(self)
    }
}

/// A device region a stage's lowering needs besides the rows it reads and
/// writes, named for the buffer plan.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Scratch {
    pub name: &'static str,
    pub bytes: u64,
}

/// A component that can stand as a stage (or as the output layout).
pub trait Component: Variant {
    fn parallel(&self) -> ParallelClass;
    /// The rows after this stage, or why `input` does not fit it. Checks
    /// nested components and every derived dimension.
    fn output(&self, context: &Context, input: &Boundary) -> Result<Boundary, String>;
    /// The weights the stage reads for `input`, exactly.
    fn roles(&self, context: &Context, input: &Boundary) -> Result<Vec<Role>, String>;
    /// For a stack of blocks, how many: its weight names then take `{i}`.
    fn blocks(&self) -> Option<u32> {
        None
    }
    /// Whether the stage joins the declaration's segments into one item.
    fn joins_segments(&self) -> bool {
        false
    }
    /// One item's (or segment's) rows after this stage.
    fn item(&self, item: Item) -> Result<Item, String> {
        Ok(item)
    }
    /// The scratch regions of a run of `rows` rows of `input`, for a
    /// variant whose lowering the shared buffer plan does not know (a
    /// model's registered variant). Shared variants report none here.
    fn scratch(&self, rows: u64, input: &Boundary) -> Result<Vec<Scratch>, String> {
        let _ = (rows, input);
        Ok(Vec::new())
    }
    /// The estimated FLOPs and activation bytes (weights excluded) of a run
    /// reading `rows[0]` rows of `input` and writing `rows[1]` of `output`,
    /// for a model's registered variant. `None` counts its rows moved once
    /// and no arithmetic.
    fn cost(
        &self,
        rows: [u64; 2],
        input: &Boundary,
        output: &Boundary,
    ) -> Result<Option<crate::cost::Cost>, String> {
        let _ = (rows, input, output);
        Ok(None)
    }
}

/// A variant registered by a model's provider, with the provider that
/// decoded it: validation admits it only in an environment whose registry
/// holds that same provider.
#[derive(Debug)]
pub struct Registered<T: ?Sized> {
    pub component: Arc<T>,
    pub provider: &'static Provider,
}

impl<T: ?Sized> Clone for Registered<T> {
    fn clone(&self) -> Self {
        Registered {
            component: Arc::clone(&self.component),
            provider: self.provider,
        }
    }
}

impl<T: ?Sized + Variant> PartialEq for Registered<T> {
    fn eq(&self, other: &Self) -> bool {
        std::ptr::eq(self.provider, other.provider)
            && self.component.to_value() == other.component.to_value()
    }
}

/// A model-namespaced variant as written, before the environment's registry
/// decodes it. Validation refuses a declaration that still holds one.
#[derive(Clone, Debug, PartialEq)]
pub struct Unresolved {
    pub id: String,
    pub value: Map<String, Value>,
}

/// Reads `op` and its parsed kind from a component object, and whether the
/// id is model-namespaced.
pub(crate) fn op_of(value: &Map<String, Value>) -> Result<(String, Kind, bool), String> {
    let op = value
        .get("op")
        .and_then(Value::as_str)
        .ok_or("a component needs an `op` naming it")?
        .to_owned();
    let parsed = id::parse(&op)?;
    Ok((op.clone(), parsed.kind, parsed.namespace.is_some()))
}

/// A component of a kind that can stand as a stage.
#[derive(Clone, Debug, PartialEq)]
pub enum StageOp {
    Linear(Linear),
    Conv1d(Conv1d),
    Position(Position),
    Norm(Norm),
    Transformer(Transformer),
    Merge(Merge),
    Group(Group),
    Cast(Cast),
    Projector(Projector),
    Pad(Pad),
    Registered(Registered<dyn Component>),
    Unresolved(Unresolved),
}

impl StageOp {
    pub fn component(&self) -> Result<&dyn Component, String> {
        Ok(match self {
            StageOp::Linear(component) => component,
            StageOp::Conv1d(component) => component,
            StageOp::Position(component) => component,
            StageOp::Norm(component) => component,
            StageOp::Transformer(component) => component,
            StageOp::Merge(component) => component,
            StageOp::Group(component) => component,
            StageOp::Cast(component) => component,
            StageOp::Projector(component) => component,
            StageOp::Pad(component) => component,
            StageOp::Registered(registered) => registered.component.as_ref(),
            StageOp::Unresolved(unresolved) => {
                return Err(format!("`{}` is not registered", unresolved.id));
            }
        })
    }

    /// The id as written.
    pub fn id(&self) -> String {
        match self {
            StageOp::Unresolved(unresolved) => unresolved.id.clone(),
            other => other.component().map(|c| c.id()).unwrap_or_default(),
        }
    }
}

impl Serialize for StageOp {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            StageOp::Unresolved(unresolved) => unresolved.value.serialize(serializer),
            other => other
                .component()
                .map_err(S::Error::custom)?
                .to_value()
                .serialize(serializer),
        }
    }
}

impl<'de> Deserialize<'de> for StageOp {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let mut object = Map::deserialize(deserializer)?;
        for value in object.values_mut() {
            crate::registry::normalize_json_floats(value).map_err(D::Error::custom)?;
        }
        let (op, kind, namespaced) = op_of(&object).map_err(D::Error::custom)?;
        if namespaced {
            return Ok(StageOp::Unresolved(Unresolved {
                id: op,
                value: object,
            }));
        }
        let value = Value::Object(object);
        let parsed = match kind {
            Kind::Linear => serde_json::from_value(value).map(StageOp::Linear),
            Kind::Conv1d => serde_json::from_value(value).map(StageOp::Conv1d),
            Kind::Position => serde_json::from_value(value).map(StageOp::Position),
            Kind::Norm => serde_json::from_value(value).map(StageOp::Norm),
            Kind::Transformer => serde_json::from_value(value).map(StageOp::Transformer),
            Kind::Merge => serde_json::from_value(value).map(StageOp::Merge),
            Kind::Group => serde_json::from_value(value).map(StageOp::Group),
            Kind::Cast => serde_json::from_value(value).map(StageOp::Cast),
            Kind::Projector => serde_json::from_value(value).map(StageOp::Projector),
            Kind::Pad => serde_json::from_value(value).map(StageOp::Pad),
            Kind::Quantize | Kind::Embed => {
                return Err(D::Error::custom(format!(
                    "`{op}`: kind {} has no shared variant; a model registers its own",
                    kind.name()
                )));
            }
            Kind::Attention | Kind::Rotary | Kind::Mlp | Kind::Reorder | Kind::Layout => {
                return Err(D::Error::custom(format!(
                    "`{op}`: a component of kind {} is not a stage",
                    kind.name()
                )));
            }
        };
        parsed.map_err(|error| D::Error::custom(format!("`{op}`: {error}")))
    }
}

/// A component and the weights behind its roles, written as one object:
/// the component's fields plus `weights`.
#[derive(Clone, Debug, PartialEq)]
pub struct Stage<C = StageOp> {
    pub op: C,
    pub weights: Weights,
}

impl<C: Serialize> Serialize for Stage<C> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut value = serde_json::to_value(&self.op).map_err(S::Error::custom)?;
        if !self.weights.is_empty() {
            let weights = serde_json::to_value(&self.weights).map_err(S::Error::custom)?;
            value
                .as_object_mut()
                .ok_or_else(|| S::Error::custom("a component serializes to an object"))?
                .insert("weights".into(), weights);
        }
        value.serialize(serializer)
    }
}

impl<'de, C: DeserializeOwned> Deserialize<'de> for Stage<C> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let mut object = Map::deserialize(deserializer)?;
        let weights = match object.remove("weights") {
            Some(weights) => serde_json::from_value(weights)
                .map_err(|error| D::Error::custom(format!("weights: {error}")))?,
            None => Weights::new(),
        };
        let op = serde_json::from_value(Value::Object(object)).map_err(D::Error::custom)?;
        Ok(Self { op, weights })
    }
}
