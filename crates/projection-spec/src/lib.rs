//! Model-neutral projection declarations and checkpoint metadata contracts.

pub mod bind;
mod consumer;
mod contract_validation;
pub mod contracts;
pub use consumer::Consumer;
pub mod derive;
mod drafters;
pub use drafters::Drafter;
pub mod inputs;
pub mod layers;
pub mod lookup;
mod offers;
pub use lookup::{LookupShardTemplate, LookupShards};
pub use offers::offered_weight;
mod segments;
pub mod side;
pub mod support;
pub use segments::{HeadGatherOrder, HeadSegment};
pub use support::{Admission, CheckpointCondition};
pub mod profiles;
pub mod registry;
pub mod validate;

use std::collections::BTreeMap;
use std::fmt;

use contracts::*;
pub use kernel_core::contract::{DType, Epilogues, NormReduction};
pub use kernel_core::precision::format::SliceAxis;
use kernel_core::precision::format::{FormatId, KernelFamily};
use kernel_core::precision::{ExecutionPhase, OperationKind};
pub use layers::{CurrentLayer, LayerSelector};
pub use profiles::{UseOverride, UseProfileOverride, UseProfileRef, UseSelector};
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// The draft declaration schema. Freeze it after its consumers validate.
pub const SCHEMA: &str = "chopinfer.projection.v1";

/// Why a declaration or checkpoint was refused.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Error(pub String);

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Error {}

/// Integer geometry values. Dimensions are positive; offsets may be zero.
pub type Geometry = BTreeMap<String, u64>;

/// A literal, geometry key, or checked integer expression.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Dim {
    Literal(u64),
    Key(String),
    Expr(Box<DimExpr>),
}

/// Dimension arithmetic. Division must be exact; no implicit rounding.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case", deny_unknown_fields)]
pub enum DimExpr {
    Sum { terms: Vec<Dim> },
    Product { factors: Vec<Dim> },
    Quotient { numerator: Dim, denominator: Dim },
}

/// A layer family. Decoder and drafter stems have the same semantics.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LayerStem {
    pub id: String,
    /// Exactly one `{i}` placeholder, replaced by each selected index.
    pub template: String,
    pub indices: LayerIndices,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub execution: Option<ExecutionLayers>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub support: Vec<SupportRule>,
}

/// A half-open index range or a strictly increasing explicit index list.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum LayerIndices {
    Range { start: Dim, end: Dim },
    List { indices: Vec<Dim> },
}

/// A checkpoint name pattern. A stem source uses `{stem}` exactly once.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TensorSource {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stem: Option<String>,
    /// Optional named `{operand}` and integer `{expert}` select source members.
    pub tensor: String,
}

/// An ordered row chunk of one projection, retaining its companion identity.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Part {
    pub id: String,
    pub rows: Dim,
    /// Override the projection head count for this chunk's head-grouped split.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub heads: Option<Dim>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub range: Option<RowRange>,
    #[serde(default)]
    pub scale_origin: [u64; 2],
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub head_runs: Vec<HeadRun>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub transforms: Vec<Variant>,
}

/// Projection semantics, independent of checkpoint encoding and kernels.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum ProjectionKind {
    Dense {},
    ExpertBank {
        experts: Dim,
        top_k: Dim,
    },
    LookupTable {},
    /// `groups` independent dense matrices of `shape` each, without routing
    /// or a sum across groups. The checkpoint tensor stacks the groups along
    /// rows; group `g` reads its own K-wide activation slice `g * K..(g + 1)
    /// * K` and writes output columns `g * rows..(g + 1) * rows`. A rank owns
    /// whole groups. This is the grouped fold GEMM contract
    /// (`kernel_core::mxfp8_gemm`), never one dense matrix over the stack.
    GroupedDense {
        groups: Dim,
    },
}

/// Tensor-parallel ownership and the tile used when padding its split axis.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "class", rename_all = "snake_case", deny_unknown_fields)]
pub enum SplitClass {
    Replicate {},
    Column {
        tile: Dim,
    },
    Row {
        tile: Dim,
    },
    HeadGrouped {
        heads: Dim,
        head_dim: Dim,
        tile: Dim,
    },
    SegmentedHeads {
        segments: Vec<HeadSegment>,
        #[serde(default, skip_serializing_if = "HeadGatherOrder::is_source")]
        order: HeadGatherOrder,
        tile: Dim,
    },
    ExpertOwned {
        tile: Dim,
    },
    ExpertSharded {
        axis: SliceAxis,
        tile: Dim,
    },
    VocabSharded {
        tile: Dim,
    },
}

/// One phase and capacity. Verify capacities are explicit graph buckets.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Use {
    pub phase: ExecutionPhase,
    pub capacity: Dim,
    pub graph: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub contract: Option<UseContract>,
}

/// A namespaced variant with parameters decoded by its registered provider.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Variant {
    pub id: String,
    pub parameters: Value,
    pub numerical_contract: String,
    pub inputs: Vec<Port>,
    pub outputs: Vec<Port>,
}

/// A logical matrix, its checkpoint sources, and its complete declared uses.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Projection {
    pub id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stem: Option<String>,
    /// Another projection whose published port, offers and entries serve
    /// this one. It keeps its own sources, stem and format admission.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub publishes_as: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<TensorSource>,
    /// Rows per operand and K, before tensor-parallel slicing or padding.
    pub shape: [Dim; 2],
    #[serde(
        default = "default_operands",
        skip_serializing_if = "is_default_operands"
    )]
    pub operands: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub parts: Vec<Part>,
    pub kind: ProjectionKind,
    pub split: SplitClass,
    pub epilogues: Epilogues,
    pub numerical_contract: String,
    /// Optional plug-in id. Absence permits the importer to choose by metadata.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub format: Option<FormatId>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub uses: Vec<Use>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub use_profile: Option<UseProfileRef>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub source_transforms: Vec<Variant>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub composition: Option<Composition>,
    /// Formats this projection's port publishes through shared generated
    /// kernel offers, in order. Registered implementations publish their own.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub offers: Vec<FormatId>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub operand_splits: Vec<OperandSplit>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub inputs: Vec<inputs::ActivationInput>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lookup: Option<LookupAssembly>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub assembly: Vec<SourceMember>,
}

/// Ordered projections sharing one input. Mixed formats run separately.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FusionGroup {
    pub id: String,
    pub input: String,
    pub members: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub use_profile: Option<UseProfileRef>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub uses: Vec<Use>,
    #[serde(default)]
    pub eligibility: Eligibility,
    #[serde(default)]
    pub fallback: Fallback,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub outputs: Vec<OutputSegment>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub composition: Option<Composition>,
}

/// The consumer role of a non-projection checkpoint tensor.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SideKind {
    Norm,
    Gate,
    Router,
    Conv,
    State,
    Bias,
    Sink,
    Auxiliary,
}

/// A side tensor bound to a projection or fusion-group consumer id.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SideTensor {
    pub id: String,
    #[serde(default, skip_serializing_if = "side::SidePlacement::is_default")]
    pub placement: side::SidePlacement,
    pub source: TensorSource,
    pub shape: Vec<Dim>,
    pub kind: SideKind,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub dtypes: Vec<DType>,
    pub consumer: Consumer,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub views: Vec<ConsumerView>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub norm: Option<NormContract>,
}

/// The complete projection-owned tensor scope and numerical declarations.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Declaration {
    pub schema: String,
    pub namespace: String,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub use_profiles: BTreeMap<String, Vec<Use>>,
    pub layer_stems: Vec<LayerStem>,
    pub projections: Vec<Projection>,
    pub fusion_groups: Vec<FusionGroup>,
    pub side_tensors: Vec<SideTensor>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub external_tensors: Vec<ExternalTensor>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub drafters: Vec<Drafter>,
}

/// Parse, resolve typed variant parameters, and validate with the caller's
/// geometry and explicitly admitted provider registry.
pub fn parse(
    text: &str,
    geometry: &Geometry,
    registry: &registry::Registry,
) -> Result<Declaration, Error> {
    let mut declaration: Declaration =
        serde_json::from_str(text).map_err(|e| Error(e.to_string()))?;
    declaration.normalize_contracts(geometry, registry)?;
    declaration.validate(geometry, registry)?;
    Ok(declaration)
}

impl Declaration {
    /// Typed JSON with recursively sorted object keys and no whitespace.
    pub fn canonical_json(&self) -> String {
        let value = serde_json::to_value(self).expect("declaration serialization is infallible");
        let mut out = String::new();
        canonical(&value, &mut out);
        out
    }

    /// Lowercase SHA-256 of canonical_json. Geometry is bound separately.
    pub fn digest(&self) -> String {
        use sha2::{Digest as _, Sha256};
        Sha256::digest(self.canonical_json().as_bytes())
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect()
    }
}

fn canonical(value: &Value, out: &mut String) {
    match value {
        Value::Object(object) => {
            let mut keys: Vec<_> = object.keys().collect();
            keys.sort();
            out.push('{');
            for (i, key) in keys.into_iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                out.push_str(&Value::String(key.clone()).to_string());
                out.push(':');
                canonical(&object[key], out);
            }
            out.push('}');
        }
        Value::Array(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                canonical(item, out);
            }
            out.push(']');
        }
        scalar => out.push_str(&scalar.to_string()),
    }
}

fn default_operands() -> Vec<String> {
    vec!["weight".into()]
}
fn is_default_operands(operands: &[String]) -> bool {
    operands == default_operands()
}

impl Projection {
    pub fn operand_count(&self) -> u32 {
        self.operands.len() as u32
    }
    pub fn operand_index(&self, name: &str) -> Option<u64> {
        self.operands
            .iter()
            .position(|id| id == name)
            .map(|i| i as u64)
    }
    /// The kernel family that serves this kind as a matrix operation. A dense
    /// projection may also lower to row stages.
    pub fn family(&self) -> KernelFamily {
        match self.kind {
            ProjectionKind::Dense {} | ProjectionKind::GroupedDense { .. } => KernelFamily::Dense,
            ProjectionKind::ExpertBank { .. } => KernelFamily::GroupedExpert,
            ProjectionKind::LookupTable {} => KernelFamily::Lookup,
        }
    }
    /// The operation kind of one rank's selection: an expert bank keeps the
    /// selected experts and its declared top-k. A selection of several
    /// groups of a grouped dense projection has none: it runs on the grouped
    /// fold GEMM contract (`DerivedOperation::fold_gemm_query`), never as a
    /// dense operation. One group is the dense matrix it is.
    pub fn operation_kind(
        &self,
        geometry: &Geometry,
        selection: &derive::RankSelection,
    ) -> Result<OperationKind, Error> {
        Ok(match &self.kind {
            ProjectionKind::GroupedDense { .. } if selection.experts.len() > 1 => {
                return Err(Error(format!(
                    "projection `{}` selects {} groups, which run on the grouped fold GEMM \
                     contract, not as a dense operation",
                    self.id,
                    selection.experts.len()
                )));
            }
            ProjectionKind::Dense {} | ProjectionKind::GroupedDense { .. } => OperationKind::Dense,
            ProjectionKind::ExpertBank { top_k, .. } => OperationKind::Grouped {
                experts: selection.experts.end - selection.experts.start,
                top_k: u32::try_from(top_k.resolve(geometry)?)
                    .map_err(|_| Error("top_k exceeds u32".into()))?,
            },
            ProjectionKind::LookupTable {} => OperationKind::Lookup,
        })
    }
    /// The owner of execution instances may differ from checkpoint materialization.
    pub fn stem(&self) -> Option<&str> {
        self.stem
            .as_deref()
            .or_else(|| self.source.as_ref().and_then(|s| s.stem.as_deref()))
            .or_else(|| {
                self.lookup
                    .as_ref()
                    .and_then(|l| l.shards.first_source())
                    .and_then(|s| s.stem.as_deref())
            })
    }
}

impl LayerStem {
    /// The executed layer indices: the declared execution subset, else every
    /// materialized index.
    pub fn executed(&self, geometry: &Geometry) -> Result<Vec<u64>, Error> {
        match &self.execution {
            Some(execution) => execution
                .indices
                .iter()
                .map(|d| d.resolve_offset(geometry))
                .collect(),
            None => self.indices.resolve(geometry),
        }
    }

    /// The materialized index an executed layer reads in this stem: its
    /// execution alias's source index, else the same index. An alias that
    /// names another source stem is resolved by derivation, not here.
    pub fn source_index(&self, geometry: &Geometry, executed: u64) -> Result<u64, Error> {
        for alias in self.execution.iter().flat_map(|e| &e.aliases) {
            if alias.execution_index.resolve_offset(geometry)? == executed {
                if alias.source_stem.as_ref().is_some_and(|s| *s != self.id) {
                    return Err(Error(format!(
                        "stem `{}` aliases layer {executed} from another stem",
                        self.id
                    )));
                }
                return alias.source_index.resolve_offset(geometry);
            }
        }
        Ok(executed)
    }
}

impl SideTensor {
    pub fn admitted_dtypes(&self) -> &[DType] {
        if self.dtypes.is_empty() {
            self.norm.as_ref().map_or(&[], |norm| &norm.weight_dtypes)
        } else {
            &self.dtypes
        }
    }
}
