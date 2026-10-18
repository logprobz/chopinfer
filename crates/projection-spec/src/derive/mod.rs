//! Pure rank-local operations, source selections and required views.
//! Checkpoint IO and persistent record construction stay with callers.

mod adapter;
mod assembly;
mod authentication;
mod execution;
pub mod hooks;
mod index;
mod inputs;
mod operation;
pub mod payload;
mod prepared;
pub(crate) mod selection;
mod views;

pub use adapter::{AdapterValue, ControlBinding, StageAdapter, StageExecution};
pub use authentication::grouped_dense as grouped_dense_matrices;
pub use operation::derive;
pub use selection::Selection as RankSelection;
pub use selection::select as rank_selection;

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::ops::Range;

use kernel_core::contract::{DType, Target};
use kernel_core::precision::format::{A16Recipe, FormatId, KernelFamily};
use kernel_core::precision::layout::PackingPlan;
use kernel_core::precision::{
    ActivationSpec, ArithmeticSpec, MatrixGeometry, OperationSpec, OperationUse, WeightSpec,
};
use kernel_core::weight_transform::{PackedSection, WeightRecord};

use crate::bind::BoundDeclaration;
use crate::registry::Registry;
use crate::{Declaration, Geometry, Projection, Use};

/// One declaration role and execution layer. Global roles have no layer.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub struct InstanceKey {
    pub projection: String,
    pub layer: Option<u64>,
}

/// A source matrix mapped into one logical operand, retaining source coordinates.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SourceMatrix {
    pub weight: WeightSpec,
    pub expert: u32,
    pub operand: u32,
    pub source_expert: u64,
    pub target_rows: Range<u64>,
    pub source_rows: Range<u64>,
    pub target_k: Range<u64>,
    pub source_k: Range<u64>,
}

/// Preserve whether an operation stores rank geometry or uses its source shape.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GeometryMode {
    RankLocal,
    SourceShape,
}

/// Imported operation identity, arithmetic and activation recipes. No recipe is
/// inferred from checkpoint dtype or a declaration's numerical-contract name.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ImportedProjection {
    /// `{i}` names the execution layer; source names remain materialized names.
    pub operation_id: String,
    pub target: Target,
    pub family: KernelFamily,
    pub format: FormatId,
    pub geometry: GeometryMode,
    pub weights: Vec<WeightSpec>,
    pub activations: Vec<ActivationSpec>,
    pub arithmetic: ArithmeticSpec,
    pub a16_recipe: A16Recipe,
    pub matrices: Vec<SourceMatrix>,
    pub preparation: Option<SourcePreparation>,
}

/// A provider's rank-local source preparation. Matrices remain authenticated
/// checkpoint mappings; the effective weights describe the prepared operand.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SourcePreparation {
    pub topology: Topology,
    pub geometry: MatrixGeometry,
    pub records: Vec<VariantRecord>,
}

/// A bypass preserves the complete input, including its format and recipes.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SourceTransformResult {
    Applied(ImportedProjection),
    Bypass,
}

/// Imports indexed by materialized source layer. Execution aliases reuse imports.
pub type ImportedProjections = BTreeMap<InstanceKey, ImportedProjection>;

/// Validated topology coordinates for one rank's pure derivation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Topology {
    pub ranks: u32,
    pub rank: u32,
}

/// One rank's selection from one source matrix. Raw source weights stay unchanged.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OperandSlice {
    pub source: SourceMatrix,
    pub format: FormatId,
    pub rows: Vec<Range<u64>>,
    pub k: Range<u64>,
    /// Destination positions after rank gather, excluding physical padding.
    pub target_rows: Vec<Range<u64>>,
    pub target_k: Range<u64>,
    pub expert: u32,
    pub operand: u32,
    pub geometry: MatrixGeometry,
}

/// A required view's role. Weight sections retain their physical layout contract.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ViewRole {
    Weight(PackedSection),
    Side,
    Input,
    Output,
    State,
}

/// A required view with exact dtype, shape, byte extent and portable layout id.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RequiredView {
    pub id: String,
    pub role: ViewRole,
    pub dtype: DType,
    pub interpretation: Option<crate::contracts::Interpretation>,
    pub shape: Vec<u64>,
    pub bytes: u64,
    pub layout: String,
}

/// One stage's resolved ports and declared state mutations, in declaration order.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DerivedStage {
    pub id: String,
    pub numerical_contract: String,
    pub inputs: Vec<RequiredView>,
    pub outputs: Vec<RequiredView>,
    pub mutations: Vec<String>,
    pub adapter: Option<StageAdapter>,
}

/// A provider-owned legacy record. It uses existing transform ABI identifiers.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VariantRecord {
    pub view: String,
    /// Byte offset within the named view, independent of arena placement.
    pub view_offset: u64,
    pub section: PackedSection,
    pub record: WeightRecord,
    pub tensors: Vec<String>,
}

/// One exact activation gather, with padding supplied by its required view.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct InputSegment {
    pub source: String,
    pub source_columns: Option<u64>,
    pub columns: Range<u64>,
    pub target_columns: Range<u64>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DerivedInput {
    pub operand: u32,
    pub binding: String,
    pub view: RequiredView,
    pub segments: Vec<InputSegment>,
    /// View columns between segments that hold zero padding, such as the
    /// packed tail of each group's K-wide slot of a grouped input. Any
    /// other hole between segments is refused.
    pub padding: Vec<Range<u64>>,
}

/// One operation for one requested use, retaining compiled row and view contracts.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DerivedOperation {
    pub instance: InstanceKey,
    pub source_layer: Option<u64>,
    pub source_instance: InstanceKey,
    pub requested: OperationUse,
    pub compiled_capacity: u32,
    pub row_multiplier: u64,
    pub chunk_rows: Option<u64>,
    pub per_item_verify: Option<u64>,
    pub aggregate_verify: Option<u64>,
    pub exact_rows: bool,
    pub operation: OperationSpec,
    pub logical_geometry: MatrixGeometry,
    pub family: KernelFamily,
    pub a16_recipe: A16Recipe,
    pub slices: Vec<OperandSlice>,
    pub packing: PackingPlan,
    pub views: Vec<RequiredView>,
    pub stages: Vec<DerivedStage>,
    pub inputs: Vec<DerivedInput>,
    pub variant_records: Vec<VariantRecord>,
    /// Authenticated aggregate and part spans available to existing ABI records.
    pub record_sources: Vec<kernel_core::precision::TensorSpan>,
    pub source_preparation: Option<SourcePreparation>,
}

/// One explicit fusion decision and the provider's replacement operations, if any.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DerivedFusion {
    pub id: String,
    pub layer: Option<u64>,
    pub requested: OperationUse,
    pub members: Vec<InstanceKey>,
    pub fused: bool,
    pub operations: Vec<DerivedOperation>,
    pub views: Vec<RequiredView>,
    pub stages: Vec<DerivedStage>,
}

/// The rank's required operations, explicit fusion results and side views.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct RankDerivation {
    pub operations: Vec<DerivedOperation>,
    pub fusions: Vec<DerivedFusion>,
    pub side_views: Vec<RequiredView>,
}

/// Complete immutable input to one rank derivation.
pub struct DerivationInput<'a> {
    pub declaration: &'a Declaration,
    pub bound: &'a BoundDeclaration,
    pub geometry: &'a Geometry,
    pub registry: &'a Registry,
    pub topology: Topology,
    pub imported: &'a ImportedProjections,
    pub variants: &'a dyn hooks::VariantDeriver,
}

/// Context supplied to a registered projection transform, stage or implementation.
#[derive(Clone)]
pub struct ProjectionContext<'a> {
    pub bound: &'a BoundDeclaration,
    pub source_instance: InstanceKey,
    pub selections: Option<&'a [RankSelection]>,
    pub declaration: &'a Declaration,
    pub projection: &'a Projection,
    pub geometry: &'a Geometry,
    pub topology: Topology,
    pub layer: Option<u64>,
    pub source_layer: Option<u64>,
    pub usage: &'a Use,
}

/// The default operation identity template of a projection instance: the
/// projection, then its execution layer when it has a stem, then the output
/// type, joined by dots ([`Declaration::operation_instance`] inverts it).
pub fn default_operation_id(projection: &Projection) -> String {
    if projection.stem().is_some() {
        "{projection}.{i}.{output}".into()
    } else {
        "{projection}.{output}".into()
    }
}

impl Declaration {
    /// The projection and execution layer of an operation identity built
    /// from [`default_operation_id`]. The longest matching projection ID
    /// wins, so a projection whose ID extends another's is never shadowed.
    pub fn operation_instance(&self, id: &str) -> Result<(&Projection, Option<u64>), crate::Error> {
        let unknown = || crate::Error(format!("operation `{id}` names no declared projection"));
        let (projection, rest) = self
            .projections
            .iter()
            .filter_map(|p| Some((p, id.strip_prefix(&p.id)?.strip_prefix('.')?)))
            .max_by_key(|(p, _)| p.id.len())
            .ok_or_else(unknown)?;
        if projection.stem().is_none() {
            return Ok((projection, None));
        }
        let layer = rest
            .split('.')
            .next()
            .and_then(|layer| layer.parse().ok())
            .ok_or_else(unknown)?;
        Ok((projection, Some(layer)))
    }
}

impl Declaration {
    /// The operation identity [`default_operation_id`] gives every executed
    /// instance of each projection `scope` admits, once per output type,
    /// with the declared uses of that output. Uses must be profile-expanded.
    pub fn operations(
        &self,
        geometry: &Geometry,
        scope: impl Fn(&Projection) -> bool,
    ) -> Result<Vec<(String, Vec<OperationUse>)>, crate::Error> {
        let mut operations = vec![];
        for projection in self.projections.iter().filter(|p| scope(p)) {
            let layers = match projection.stem() {
                Some(id) => {
                    let stem = self
                        .layer_stems
                        .iter()
                        .find(|s| s.id == id)
                        .ok_or_else(|| crate::Error(format!("missing stem `{id}`")))?;
                    stem.executed(geometry)?.into_iter().map(Some).collect()
                }
                None => vec![None],
            };
            let mut outputs: Vec<(DType, Vec<OperationUse>)> = vec![];
            for u in &projection.uses {
                let output = u.contract.as_ref().map_or(DType::Bf16, |c| c.output_dtype);
                let usage = OperationUse {
                    phase: u.phase,
                    capacity: u32::try_from(u.capacity.resolve(geometry)?)
                        .map_err(|_| crate::Error("declared capacity exceeds u32".into()))?,
                    graph: u.graph,
                };
                match outputs.iter_mut().find(|(dtype, _)| *dtype == output) {
                    Some((_, uses)) => uses.push(usage),
                    None => outputs.push((output, vec![usage])),
                }
            }
            for layer in layers {
                for (output, uses) in &outputs {
                    let output = match output {
                        DType::Bf16 => "bf16",
                        DType::F32 => "f32",
                        _ => {
                            return Err(crate::Error("unsupported projection output dtype".into()));
                        }
                    };
                    let mut id = default_operation_id(projection)
                        .replace("{projection}", &projection.id)
                        .replace("{output}", output);
                    if let Some(layer) = layer {
                        id = id.replace("{i}", &layer.to_string());
                    }
                    operations.push((id, uses.clone()));
                }
            }
        }
        Ok(operations)
    }
}

impl ProjectionContext<'_> {
    /// Resolve the operation identity with the same rules used by ordinary
    /// derivation. Missing layers leave templates unresolved and are refused.
    pub fn operation_id(&self, template: &str, output: DType) -> Result<String, crate::Error> {
        let output = match output {
            DType::Bf16 => "bf16",
            DType::F32 => "f32",
            _ => return Err(crate::Error("unsupported projection output dtype".into())),
        };
        let mut id = template
            .replace("{output}", output)
            .replace("{projection}", &self.projection.id);
        if let Some(stem) = self
            .projection
            .stem()
            .and_then(|s| self.declaration.layer_stems.iter().find(|d| d.id == s))
        {
            let stem = self.layer.map_or_else(
                || stem.template.clone(),
                |i| stem.template.replace("{i}", &i.to_string()),
            );
            id = id.replace("{stem}", &stem);
        }
        if let Some(layer) = self.layer {
            id = id.replace("{i}", &layer.to_string());
        }
        if id.contains(['{', '}']) {
            return Err(crate::Error(
                "operation id has an unresolved template".into(),
            ));
        }
        Ok(id)
    }

    /// Derive a baseline after checking topology, declaration/use membership,
    /// and authenticated source mappings. Retain the assembly selection and
    /// all declared capacity/output contracts; adapters own only their recipe.
    pub fn ordinary(
        &self,
        imported: &ImportedProjection,
    ) -> Result<DerivedOperation, crate::Error> {
        operation::checked_ordinary(self, imported)
    }
}

/// A fusion provider sees the owning group and the selected source-format recipe.
pub struct FusionContext<'a> {
    pub projection: ProjectionContext<'a>,
    pub group: &'a crate::FusionGroup,
    pub numerical: Option<&'a crate::contracts::NumericalRecipe>,
}
