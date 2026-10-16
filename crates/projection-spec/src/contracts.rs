//! Typed composition, ownership and execution contracts for projection data.

use kernel_core::contract::{ActivationFormat, DType};
use kernel_core::precision::ExecutionPhase;
use kernel_core::precision::format::{A16Recipe, FormatId};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

use crate::{Dim, SliceAxis, TensorSource, Variant};

/// A half-open logical range, written as a nonnegative start and positive length.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RowRange {
    pub start: Dim,
    pub length: Dim,
}

/// One ordered head run within a part, before rank ownership is derived.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HeadRun {
    pub id: String,
    pub rows: RowRange,
    pub heads: Dim,
    pub head_dim: Dim,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Interpretation {
    Unsigned,
}

/// A declared data or state port of a registered contract.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Port {
    pub id: String,
    pub binding: String,
    pub dtype: DType,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub interpretation: Option<Interpretation>,
    pub shape: Vec<Dim>,
    pub state: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub format: Option<FormatId>,
}

/// An ordered registered stage with explicit phase adapters and state mutation.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Stage {
    pub id: String,
    pub contract: Variant,
    pub phases: Vec<ExecutionPhase>,
    pub mutates: Vec<String>,
}

/// An immutable numerical identity and format-owned A16 arithmetic recipe.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NumericalRecipe {
    pub numerical_contract: String,
    pub a16_recipe: A16Recipe,
}

/// Format selection applies to the bound source, before source transforms.
/// It does not select a registered implementation.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Composition {
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub implementations: BTreeMap<FormatId, Variant>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub numerical_contracts: BTreeMap<FormatId, NumericalRecipe>,
    pub pre: Vec<Stage>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub implementation: Option<Variant>,
    pub post: Vec<Stage>,
}

impl Composition {
    /// Resolve one recipe from the declaration map for all contributing source
    /// formats. A mixture of mapped and unmapped formats is incompatible.
    pub fn numerical_for_sources(
        &self,
        formats: impl IntoIterator<Item = FormatId>,
    ) -> Result<Option<&NumericalRecipe>, crate::Error> {
        let mut selected: Option<Option<&NumericalRecipe>> = None;
        for format in formats {
            let recipe = self.numerical_contracts.get(&format);
            if selected.is_some_and(|previous| previous != recipe) {
                return Err(crate::Error(
                    "source formats select incompatible numerical recipes".into(),
                ));
            }
            selected = Some(recipe);
        }
        Ok(selected.flatten())
    }

    /// Per-use overrides precede this format-selected default and its fallback.
    pub fn implementation_for(&self, format: FormatId) -> Option<&Variant> {
        self.implementations
            .get(&format)
            .or(self.implementation.as_ref())
    }
}

/// Fusion compatibility predicate. All terms must hold for one use.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "predicate", rename_all = "snake_case", deny_unknown_fields)]
pub enum Eligibility {
    SameFormats {},
    All {
        formats: Vec<FormatId>,
        activation: Option<ActivationFormat>,
        phases: Vec<ExecutionPhase>,
    },
}

impl Default for Eligibility {
    fn default() -> Self {
        Self::SameFormats {}
    }
}

impl Eligibility {
    pub(crate) fn admits(
        &self,
        ids: &std::collections::BTreeSet<FormatId>,
        phase: ExecutionPhase,
        activations: &[Option<ActivationFormat>],
    ) -> bool {
        if ids.is_empty() {
            return false;
        }
        match self {
            Self::SameFormats {} => ids.len() == 1,
            Self::All {
                formats,
                activation,
                phases,
            } => {
                ids.iter().all(|id| formats.contains(id))
                    && (phases.is_empty() || phases.contains(&phase))
                    && activation.is_none_or(|a| activations.iter().all(|v| *v == Some(a)))
            }
        }
    }
}

/// Behavior when a declared fusion predicate does not hold.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Fallback {
    #[default]
    Separate,
    Refuse,
}

/// One member's logical columns in a shared output buffer.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OutputSegment {
    pub buffer: String,
    pub member: String,
    pub offset: Dim,
    pub row_stride: Dim,
    pub live_columns: Dim,
}

/// Per-operand split override in logical coordinates.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OperandSplit {
    pub operand: String,
    pub axis: SliceAxis,
    pub tile: Dim,
}

/// Who owns trailing vocabulary padding, independent of checkpoint shards.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "owner", rename_all = "snake_case", deny_unknown_fields)]
pub enum PaddingOwner {
    LastRank {},
}

/// Companion lifetime for assembled checkpoint lookup rows.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CompanionScope {
    Table,
    Shard,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CompanionScopes {
    pub local: CompanionScope,
    pub global: CompanionScope,
}
impl Default for CompanionScopes {
    fn default() -> Self {
        Self {
            local: CompanionScope::Shard,
            global: CompanionScope::Table,
        }
    }
}

/// Scope follows the actual imported format, with an explicit default.
#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CompanionPolicy {
    pub default: CompanionScopes,
    #[serde(default, skip_serializing_if = "std::collections::BTreeMap::is_empty")]
    pub formats: std::collections::BTreeMap<FormatId, CompanionScopes>,
}
impl CompanionPolicy {
    pub fn for_format(&self, format: FormatId) -> CompanionScopes {
        self.formats.get(&format).copied().unwrap_or(self.default)
    }
    pub fn is_default(&self) -> bool {
        *self == Self::default()
    }
}

/// A checkpoint shard's rows in one logical assembled lookup table.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LookupShard {
    pub id: String,
    pub source: TensorSource,
    pub rows: RowRange,
}

/// An ordered head's variable-sized vocabulary interval.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LookupHead {
    pub id: String,
    pub vocabulary: RowRange,
}

/// Checkpoint shard assembly and head assignment, separate from TP splitting.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LookupAssembly {
    #[serde(default, skip_serializing_if = "CompanionPolicy::is_default")]
    pub companions: CompanionPolicy,
    pub shards: crate::LookupShards,
    pub heads: Vec<LookupHead>,
    pub padding_owner: PaddingOwner,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RowCoordinates {
    #[default]
    Global,
    RankLocal,
}
impl RowCoordinates {
    pub fn is_global(&self) -> bool {
        *self == Self::Global
    }
}

/// A source/member reused in an ordered derived projection assembly.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SourceMember {
    #[serde(default, skip_serializing_if = "RowCoordinates::is_global")]
    pub coordinates: RowCoordinates,
    #[serde(default, skip_serializing_if = "is_false")]
    pub inherit_split: bool,
    pub projection: String,
    pub layer: Option<crate::LayerSelector>,
    pub operand: Option<String>,
    pub rows: RowRange,
}

/// Exact source-to-view conversion admitted by a consumer.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ViewConversion {
    #[default]
    Identity,
    WidenBf16F32,
}

/// A source view bound to one consumer without narrowing the checkpoint value.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ConsumerView {
    pub consumer: crate::Consumer,
    pub dtype: DType,
    pub conversion: ViewConversion,
}

/// Affine storage either repeats one weight vector or stores one per group.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NormAffine {
    Shared,
    PerGroup,
}

/// Shared normalization arithmetic, independent of model-registered variants.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NormContract {
    pub reduction_width: Dim,
    pub groups: Dim,
    pub affine: NormAffine,
    pub reduction: kernel_core::contract::NormReduction,
    pub epsilon_bits: u32,
    pub one_plus_weight: bool,
    pub weight_dtypes: Vec<DType>,
}

/// Per-topology/format/phase admission. Empty selectors match all values.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SupportRule {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub when: Option<crate::CheckpointCondition>,
    pub ranks: Vec<u32>,
    pub formats: Vec<FormatId>,
    pub phases: Vec<ExecutionPhase>,
    pub enabled: bool,
}

/// One execution alias for an already materialized checkpoint layer.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LayerAlias {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_stem: Option<String>,
    pub execution_index: Dim,
    pub source_index: Dim,
}

/// Executed layer subset and aliases, independent of checkpoint materialization.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExecutionLayers {
    pub indices: Vec<Dim>,
    pub aliases: Vec<LayerAlias>,
}

/// Requested capacity and compiled arithmetic capacity are separate identities.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct UseContract {
    pub compiled_capacity: Dim,
    pub row_multiplier: Dim,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chunk_rows: Option<Dim>,
    pub per_item_verify: Option<Dim>,
    pub aggregate_verify: Option<Dim>,
    pub output_dtype: DType,
    pub numerical_contract: String,
    pub activation: Option<ActivationFormat>,
    pub exact_rows: bool,
    pub support: Vec<SupportRule>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub implementation: Option<Variant>,
}

/// A role owned and validated by another declaration, included in closed coverage.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExternalTensor {
    #[serde(default, skip_serializing_if = "is_false")]
    pub optional: bool,
    pub id: String,
    pub owner: String,
    pub image_role: String,
    pub source: TensorSource,
    pub shape: Vec<Dim>,
    pub dtype: DType,
}

fn is_false(value: &bool) -> bool {
    !value
}

impl Port {
    pub fn validate(&self, geometry: &crate::Geometry) -> Result<(), crate::Error> {
        if !crate::registry::valid_name(&self.id) || !crate::registry::valid_name(&self.binding) {
            return Err(crate::Error("invalid variant port id or binding".into()));
        }
        crate::contract_validation::shape(&self.shape, geometry)?;
        if self.interpretation.is_some() && !matches!(self.dtype, DType::I32 | DType::I64) {
            return Err(crate::Error(
                "unsigned interpretation requires I32 or I64 storage".into(),
            ));
        }
        if let Some(format) = self.format {
            kernel_core::precision::format::registry::by_id(format)
                .map_err(|e| crate::Error(e.to_string()))?;
        }
        Ok(())
    }
}
