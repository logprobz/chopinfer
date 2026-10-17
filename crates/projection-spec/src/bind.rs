//! Checkpoint metadata binding. Format-specific companion rules are supplied
//! by the caller, rather than implemented by the declaration schema.

use std::collections::BTreeMap;

use crate::contracts::{ConsumerView, Fallback, SourceMember};
use kernel_core::precision::WeightSpec;
use kernel_core::precision::format::{CompanionBinding, registry as formats};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::validate::{multiply, source_geometry};
use crate::{Error, Geometry, Projection};

/// One safetensors header entry, optionally authenticated by an index shard.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TensorInfo {
    pub dtype: String,
    pub shape: Vec<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub file: Option<String>,
}

/// Caller-supplied headers/index metadata for the declaration's entire scope.
#[derive(Clone, Debug, Default)]
pub struct CheckpointMetadata {
    pub tensors: BTreeMap<String, TensorInfo>,
    pub quantization_config: Value,
    pub validated_owners: BTreeMap<String, std::collections::BTreeSet<String>>,
}

/// An extra companion role, such as one scale tensor for a row chunk.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TensorRequirement {
    pub tensor: String,
    pub dtype: String,
    pub shape: Vec<u64>,
    pub file: Option<String>,
}

/// One expanded projection source and its resolved logical matrix shape.
#[derive(Debug)]
pub struct BindingRequest<'a> {
    pub projection: &'a Projection,
    pub tensor: &'a str,
    pub logical_shape: &'a [u64],
    pub geometry: &'a Geometry,
    pub layer: Option<u64>,
    pub operand: Option<u64>,
    pub expert: Option<u64>,
    pub shard: Option<u64>,
    pub head: Option<u64>,
}

/// An imported operand plus companion roles beyond its standard scale spans.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ImportedOperand {
    /// Full source spans. Independent grids validate through `parts`, not
    /// a uniform-grid interpretation of this aggregate descriptor.
    pub weight: WeightSpec,
    /// Ordered views for a source whose parts restart independent scale grids.
    pub parts: Vec<WeightSpec>,
    pub companions: Vec<TensorRequirement>,
    pub calibrations: Vec<FrameCalibration>,
}

/// The format import hook. It must refuse unsupported quantization recipes
/// and return the exact stored code and required companion metadata.
pub trait CompanionRules {
    fn import(
        &self,
        request: &BindingRequest<'_>,
        metadata: &CheckpointMetadata,
    ) -> Result<ImportedOperand, Error>;
}

/// One resolved static input calibration for an authenticated source frame.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FrameCalibration {
    pub expert: u64,
    pub operand: u64,
    pub part: Option<String>,
    pub scale: kernel_core::precision::ScaleSpec,
}

/// A checked operand and the instance indices that selected its source.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BoundOperand {
    pub projection: String,
    pub layer: Option<u64>,
    pub operand: Option<u64>,
    pub expert: Option<u64>,
    pub shard: Option<u64>,
    pub head: Option<u64>,
    pub imported: ImportedOperand,
    pub format: kernel_core::precision::format::FormatId,
    pub bindings: Vec<CompanionBinding>,
    pub part_bindings: Vec<Vec<CompanionBinding>>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BoundLayerAlias {
    pub execution_stem: String,
    pub execution_index: u64,
    pub source_stem: String,
    pub source_index: u64,
}

/// Binding result. Member order remains the declaration's order.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct BoundDeclaration {
    pub layer_aliases: Vec<BoundLayerAlias>,
    pub conditions: Vec<crate::support::BoundCondition>,
    pub operands: Vec<BoundOperand>,
    pub side_tensors: Vec<String>,
    pub fusion_groups: Vec<BoundFusion>,
    pub assemblies: Vec<BoundAssembly>,
    pub external_tensors: Vec<String>,
    pub side_views: Vec<BoundSideView>,
}

/// One fusion decision for a layer instance. Mixed imported formats preserve
/// the ordered members and require separate execution.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BoundFusion {
    pub id: String,
    pub layer: Option<u64>,
    pub members: Vec<String>,
    pub fused: bool,
    pub phase: kernel_core::precision::ExecutionPhase,
    pub capacity: u64,
}

/// A derived projection's ordered selections from already bound source weights.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BoundAssembly {
    pub projection: String,
    pub layer: Option<u64>,
    pub members: Vec<SourceMember>,
    pub tensors: Vec<Vec<String>>,
}

/// An exact consumer view over a bound side tensor.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BoundSideView {
    pub side: String,
    pub tensor: String,
    pub view: ConsumerView,
}

#[derive(Clone, Debug)]
struct Instance {
    tensor: String,
    layer: Option<u64>,
    operand: Option<u64>,
    expert: Option<u64>,
    shard: Option<u64>,
    head: Option<u64>,
}

fn instances(
    declaration: &crate::Declaration,
    source: &crate::TensorSource,
    counts: &[(&str, u64)],
    geometry: &Geometry,
    operands: &[String],
) -> Result<Vec<Instance>, Error> {
    let layers = if let Some(id) = &source.stem {
        let stem = declaration
            .layer_stems
            .iter()
            .find(|s| s.id == *id)
            .ok_or_else(|| Error(format!("missing stem `{id}`")))?;
        stem.indices
            .resolve(geometry)?
            .into_iter()
            .map(|i| {
                (
                    Some(i),
                    source
                        .tensor
                        .replace("{stem}", &stem.template.replace("{i}", &i.to_string())),
                )
            })
            .collect()
    } else {
        vec![(None, source.tensor.clone())]
    };
    let mut instances: Vec<_> = layers
        .into_iter()
        .map(|(layer, tensor)| Instance {
            tensor,
            layer,
            operand: None,
            expert: None,
            shard: None,
            head: None,
        })
        .collect();
    for &(axis, count) in counts {
        let token = format!("{{{axis}}}");
        if !source.tensor.contains(&token) {
            continue;
        }
        let capacity = multiply(instances.len() as u64, count)?;
        if capacity > crate::validate::MAX_INSTANCES as u64 {
            return Err(Error(format!(
                "tensor `{}` exceeds source instance limit",
                source.tensor
            )));
        }
        let mut next = Vec::with_capacity(capacity as usize);
        for instance in instances {
            for i in 0..count {
                let mut expanded = instance.clone();
                let value = if axis == "operand" {
                    operands
                        .get(i as usize)
                        .ok_or_else(|| Error("source operand name is missing".into()))?
                        .clone()
                } else {
                    i.to_string()
                };
                expanded.tensor = instance.tensor.replace(&token, &value);
                match axis {
                    "operand" => expanded.operand = Some(i),
                    "expert" => expanded.expert = Some(i),
                    "shard" => expanded.shard = Some(i),
                    "head" => expanded.head = Some(i),
                    _ => unreachable!("source axes are internal constants"),
                }
                next.push(expanded);
            }
        }
        instances = next;
    }
    Ok(instances)
}

type ProjectionSource = (
    crate::TensorSource,
    Vec<u64>,
    Vec<(&'static str, u64)>,
    Option<u64>,
);

/// A materialized projection's source patterns: the stored shape, the
/// expanded source axes and, for a sharded lookup, the shard index.
fn projection_sources(
    projection: &Projection,
    geometry: &Geometry,
) -> Result<Vec<ProjectionSource>, Error> {
    if let Some(lookup) = &projection.lookup {
        lookup
            .shards
            .resolve(geometry)?
            .iter()
            .enumerate()
            .map(|(index, shard)| {
                Ok((
                    shard.source.clone(),
                    vec![
                        shard.rows.length.resolve(geometry)?,
                        projection.shape[1].resolve(geometry)?,
                    ],
                    vec![],
                    Some(index as u64),
                ))
            })
            .collect()
    } else {
        let (shape, counts) = source_geometry(projection, geometry)?;
        Ok(vec![(
            projection
                .source
                .clone()
                .ok_or_else(|| Error("projection needs a source".into()))?,
            shape,
            counts,
            None,
        )])
    }
}

/// The declaration member that claims a checkpoint tensor.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SourceOwner {
    Projection(usize),
    Side(usize),
}

/// A claimed checkpoint tensor's owner and the source axes it expands.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SourcePosition {
    pub owner: SourceOwner,
    pub layer: Option<u64>,
    pub operand: Option<u64>,
    pub expert: Option<u64>,
    /// The shard of a sharded lookup.
    pub shard: Option<u64>,
}

impl crate::Declaration {
    /// Every checkpoint tensor the declaration claims as a projection
    /// operand or a side tensor, with the index of its owner. Assembled
    /// projections read resident views of other members and claim none.
    pub fn source_names(
        &self,
        geometry: &Geometry,
    ) -> Result<BTreeMap<String, SourceOwner>, Error> {
        Ok(self
            .source_positions(geometry)?
            .into_iter()
            .map(|(name, position)| (name, position.owner))
            .collect())
    }

    /// [`Self::source_names`] with each tensor's layer, operand, expert
    /// and shard.
    pub fn source_positions(
        &self,
        geometry: &Geometry,
    ) -> Result<BTreeMap<String, SourcePosition>, Error> {
        let mut names = BTreeMap::new();
        let mut claim = |instance: Instance, owner| {
            let position = SourcePosition {
                owner,
                layer: instance.layer,
                operand: instance.operand,
                expert: instance.expert,
                shard: instance.shard,
            };
            if names.insert(instance.tensor.clone(), position).is_some() {
                return Err(Error(format!(
                    "tensor `{}` is claimed twice",
                    instance.tensor
                )));
            }
            Ok(())
        };
        for (index, projection) in self.projections.iter().enumerate() {
            if !projection.assembly.is_empty() {
                continue;
            }
            for (source, _, counts, shard) in projection_sources(projection, geometry)? {
                for mut instance in
                    instances(self, &source, &counts, geometry, &projection.operands)?
                {
                    instance.shard = instance.shard.or(shard);
                    claim(instance, SourceOwner::Projection(index))?;
                }
            }
        }
        for (index, side) in self.side_tensors.iter().enumerate() {
            for instance in instances(self, &side.source, &[], geometry, &[])? {
                claim(instance, SourceOwner::Side(index))?;
            }
        }
        Ok(names)
    }

    /// The ordered operand code names one projection instance reads. An
    /// assembled projection names its first member's source. An expert bank
    /// names its resident bank: the `{expert}` placeholder is removed with
    /// the delimiter before it (`experts.{expert}.down` names
    /// `experts.down`, `specialists_{expert}.down` names `specialists.down`).
    pub fn instance_sources(
        &self,
        geometry: &Geometry,
        projection: &str,
        layer: Option<u64>,
    ) -> Result<Vec<String>, Error> {
        self.sources_of(geometry, projection, layer, None)
    }

    /// The checkpoint names one member `expert` of an expert-bank instance
    /// reads: [`Self::instance_sources`] with the `{expert}` placeholder set.
    pub fn expert_sources(
        &self,
        geometry: &Geometry,
        projection: &str,
        layer: Option<u64>,
        expert: u64,
    ) -> Result<Vec<String>, Error> {
        self.sources_of(geometry, projection, layer, Some(expert))
    }

    fn sources_of(
        &self,
        geometry: &Geometry,
        projection: &str,
        layer: Option<u64>,
        expert: Option<u64>,
    ) -> Result<Vec<String>, Error> {
        let find = |id: &str| {
            self.projections
                .iter()
                .find(|p| p.id == id)
                .ok_or_else(|| Error(format!("unknown projection `{id}`")))
        };
        let mut projection = find(projection)?;
        let mut layer = layer;
        if let Some(member) = projection.assembly.first() {
            layer = member
                .layer
                .as_ref()
                .map(|l| l.resolve(geometry, layer))
                .transpose()?;
            projection = find(&member.projection)?;
        }
        let source = projection.source.as_ref().ok_or_else(|| {
            Error(format!(
                "projection `{}` has no single source",
                projection.id
            ))
        })?;
        let mut name = source.tensor.clone();
        if let Some(id) = &source.stem {
            let stem = self
                .layer_stems
                .iter()
                .find(|s| s.id == *id)
                .ok_or_else(|| Error(format!("missing stem `{id}`")))?;
            let layer = layer
                .filter(|i| {
                    stem.indices
                        .resolve(geometry)
                        .is_ok_and(|all| all.contains(i))
                })
                .ok_or_else(|| Error(format!("layer is outside stem `{id}`")))?;
            name = name.replace("{stem}", &stem.template.replace("{i}", &layer.to_string()));
        } else if layer.is_some() {
            return Err(Error("unstemmed source cannot select a layer".into()));
        }
        if expert.is_some() && !name.contains("{expert}") {
            return Err(Error(format!(
                "projection `{}` has no expert placeholder",
                projection.id
            )));
        }
        name = match expert {
            Some(expert) => name.replace("{expert}", &expert.to_string()),
            None => bank_name(&name),
        };
        Ok(if name.contains("{operand}") {
            projection
                .operands
                .iter()
                .map(|operand| name.replace("{operand}", operand))
                .collect()
        } else {
            vec![name]
        })
    }
}

fn requirement(
    requirement: &TensorRequirement,
    metadata: &CheckpointMetadata,
    claimed: &mut BTreeMap<String, TensorRequirement>,
) -> Result<(), Error> {
    let tensor = &requirement.tensor;
    let found = metadata
        .tensors
        .get(tensor)
        .ok_or_else(|| Error(format!("checkpoint has no tensor `{tensor}`")))?;
    if found.dtype != requirement.dtype
        || found.shape != requirement.shape
        || requirement
            .file
            .as_ref()
            .is_some_and(|file| found.file.as_ref() != Some(file))
    {
        return Err(Error(format!(
            "tensor `{tensor}`: received {} {:?} in {:?}, expected {} {:?} in {:?}",
            found.dtype,
            found.shape,
            found.file,
            requirement.dtype,
            requirement.shape,
            requirement.file
        )));
    }
    if let Some(previous) = claimed.get(tensor)
        && (previous.dtype != requirement.dtype
            || previous.shape != requirement.shape
            || matches!((&previous.file, &requirement.file), (Some(a), Some(b)) if a != b))
    {
        return Err(Error(format!(
            "tensor `{tensor}` has conflicting companion roles"
        )));
    }
    claimed.insert(tensor.clone(), requirement.clone());
    Ok(())
}

fn span_requirement(
    span: &kernel_core::precision::TensorSpan,
    metadata: &CheckpointMetadata,
) -> Result<TensorRequirement, Error> {
    use kernel_core::contract::DType;
    span.validate().map_err(|e| Error(e.to_string()))?;
    if span.offset != 0 {
        return Err(Error(format!(
            "tensor `{}`: checkpoint span must start at zero",
            span.tensor
        )));
    }
    let found = metadata
        .tensors
        .get(&span.tensor)
        .ok_or_else(|| Error(format!("checkpoint has no tensor `{}`", span.tensor)))?;
    // Stored bytes the span reads under another name: UE8M0 scale codes
    // serialized as U8, and packed 4-bit code pairs stored as I8.
    let expected = if (span.dtype == DType::Ue8M0 && found.dtype == "U8")
        || (span.dtype == DType::U8 && found.dtype == "I8")
    {
        found.dtype.as_str()
    } else {
        span.dtype.checkpoint_dtype().ok_or_else(|| {
            Error(format!(
                "tensor `{}`: unsupported checkpoint span dtype {:?}",
                span.tensor, span.dtype
            ))
        })?
    };
    Ok(TensorRequirement {
        tensor: span.tensor.clone(),
        dtype: expected.into(),
        shape: span.shape.clone(),
        file: None,
    })
}

impl crate::Declaration {
    /// Bind every declared role and its required companions. Metadata must
    /// enumerate the complete declaration-owned scope; unclaimed names fail.
    pub fn bind(
        &self,
        geometry: &Geometry,
        registry: &crate::registry::Registry,
        metadata: &CheckpointMetadata,
        rules: &dyn CompanionRules,
    ) -> Result<BoundDeclaration, Error> {
        self.expanded_profiles(geometry)?
            .bind_expanded(geometry, registry, metadata, rules)
    }

    fn bind_expanded(
        &self,
        geometry: &Geometry,
        registry: &crate::registry::Registry,
        metadata: &CheckpointMetadata,
        rules: &dyn CompanionRules,
    ) -> Result<BoundDeclaration, Error> {
        self.validate(geometry, registry)?;
        let mut bound = BoundDeclaration::default();
        let mut claimed = BTreeMap::new();
        let mut primary_claims = BTreeMap::<String, String>::new();
        for projection in &self.projections {
            if !projection.assembly.is_empty() {
                continue;
            }
            for (source, shape, counts, shard) in projection_sources(projection, geometry)? {
                for mut instance in
                    instances(self, &source, &counts, geometry, &projection.operands)?
                {
                    if shard.is_some() {
                        instance.shard = shard;
                    }
                    if bound.operands.len() >= crate::validate::MAX_INSTANCES {
                        return Err(Error("binding exceeds source instance limit".into()));
                    }
                    let tensor = &instance.tensor;
                    if let Some(previous) =
                        primary_claims.insert(tensor.clone(), projection.id.clone())
                    {
                        return Err(Error(format!(
                            "tensor `{tensor}` is claimed by `{previous}` and `{}`",
                            projection.id
                        )));
                    }
                    if !metadata.tensors.contains_key(tensor) {
                        return Err(Error(format!(
                            "checkpoint has no tensor `{tensor}` (projection `{}`)",
                            projection.id
                        )));
                    }
                    let request = BindingRequest {
                        projection,
                        tensor,
                        logical_shape: &shape,
                        geometry,
                        layer: instance.layer,
                        operand: instance.operand,
                        expert: instance.expert,
                        shard: instance.shard,
                        head: instance.head,
                    };
                    let imported = rules
                        .import(&request, metadata)
                        .map_err(|e| Error(format!("tensor `{tensor}`: {e}")))?;
                    let weight = &imported.weight;
                    if weight.codes.tensor != *tensor || weight.logical_shape != shape {
                        return Err(Error(format!(
                            "tensor `{tensor}`: importer returned another source or logical shape"
                        )));
                    }
                    let plugin = match projection.format {
                        Some(id) => formats::by_id(id),
                        None => formats::by_format(weight.format),
                    }
                    .map_err(|e| Error(e.to_string()))?;
                    if projection
                        .composition
                        .as_ref()
                        .and_then(|c| c.implementation_for(plugin.id()))
                        .is_some_and(|default| {
                            projection
                                .uses
                                .iter()
                                .filter_map(|u| u.contract.as_ref())
                                .filter_map(|u| u.implementation.as_ref())
                                .any(|v| v == default)
                        })
                    {
                        return Err(Error(
                            "per-use implementation repeats the format-selected default".into(),
                        ));
                    }
                    if !projection.parts.is_empty() && imported.parts.is_empty() {
                        return Err(Error(format!(
                            "tensor `{tensor}`: declared parts require imported part views"
                        )));
                    }
                    let (bindings, part_bindings) = if imported.parts.is_empty() {
                        (
                            plugin
                                .companions(weight)
                                .map_err(|e| Error(e.to_string()))?,
                            Vec::new(),
                        )
                    } else {
                        (
                            Vec::new(),
                            bind_parts(projection, &imported, metadata, geometry, plugin)
                                .map_err(|e| Error(format!("tensor `{tensor}`: {e}")))?,
                        )
                    };
                    for span in std::iter::once(&weight.codes)
                        .chain(weight.local_scale.iter().map(|s| &s.tensor))
                        .chain(weight.global_scale.iter().map(|s| &s.tensor))
                    {
                        requirement(&span_requirement(span, metadata)?, metadata, &mut claimed)?;
                    }
                    let expert_count = match &projection.kind {
                        crate::ProjectionKind::ExpertBank { experts, .. } => {
                            experts.resolve(geometry)?
                        }
                        _ => 1,
                    };
                    let mut calibration_frames = std::collections::BTreeSet::new();
                    for calibration in &imported.calibrations {
                        let span = &calibration.scale.tensor;
                        span.validate().map_err(|e| Error(e.to_string()))?;
                        let found = metadata
                            .tensors
                            .get(&span.tensor)
                            .ok_or_else(|| Error("calibration has no source header".into()))?;
                        let bytes =
                            kernel_core::precision::shape_elements(&span.tensor, &found.shape)
                                .map_err(|e| Error(e.to_string()))?
                                .checked_mul(4)
                                .ok_or_else(|| Error("calibration source size overflows".into()))?;
                        if matches!(projection.kind, crate::ProjectionKind::LookupTable {})
                            || calibration.expert >= expert_count
                            || calibration.operand >= projection.operand_count().into()
                            || instance.expert.is_some_and(|e| e != calibration.expert)
                            || instance.operand.is_some_and(|o| o != calibration.operand)
                            || calibration
                                .part
                                .as_ref()
                                .is_some_and(|id| !projection.parts.iter().any(|p| p.id == *id))
                            || !calibration_frames.insert((
                                calibration.expert,
                                calibration.operand,
                                calibration.part.clone(),
                            ))
                            || span.dtype != kernel_core::contract::DType::F32
                            || found.dtype != "F32"
                            || span.offset % 4 != 0
                            || span.elements().map_err(|e| Error(e.to_string()))? != 1
                            || span
                                .offset
                                .checked_add(span.byte_length)
                                .is_none_or(|end| end > bytes)
                            || calibration.scale.scope
                                != kernel_core::precision::ScaleScope::Projection
                            || calibration.scale.convention
                                != kernel_core::precision::ScaleConvention::Multiply
                            || !calibration.scale.groups.is_empty()
                        {
                            return Err(Error("calibration frame coordinates or source span disagree with its binding".into()));
                        }
                        requirement(
                            &TensorRequirement {
                                tensor: span.tensor.clone(),
                                dtype: found.dtype.clone(),
                                shape: found.shape.clone(),
                                file: found.file.clone(),
                            },
                            metadata,
                            &mut claimed,
                        )?;
                    }
                    let mut companion_ids = std::collections::BTreeSet::new();
                    for companion in &imported.companions {
                        if companion.tensor == *tensor || !companion_ids.insert(&companion.tensor) {
                            return Err(Error(format!(
                                "tensor `{tensor}`: repeated companion `{}`",
                                companion.tensor
                            )));
                        }
                        kernel_core::precision::shape_elements(&companion.tensor, &companion.shape)
                            .map_err(|e| Error(e.to_string()))?;
                        requirement(companion, metadata, &mut claimed)?;
                    }
                    bound.operands.push(BoundOperand {
                        projection: projection.id.clone(),
                        layer: instance.layer,
                        operand: instance.operand,
                        expert: instance.expert,
                        shard: instance.shard,
                        head: instance.head,
                        imported,
                        format: plugin.id(),
                        bindings,
                        part_bindings,
                    });
                }
            }
        }
        for projection in &self.projections {
            if let Some(lookup) = &projection.lookup {
                let layers: std::collections::BTreeSet<_> = bound
                    .operands
                    .iter()
                    .filter(|b| b.projection == projection.id)
                    .map(|b| b.layer)
                    .collect();
                for layer in layers {
                    for (role, local) in [("local", true), ("global", false)] {
                        let mut companions = Vec::new();
                        for operand in bound
                            .operands
                            .iter()
                            .filter(|b| b.projection == projection.id && b.layer == layer)
                        {
                            let scopes = lookup.companions.for_format(operand.format);
                            let scope = if local { scopes.local } else { scopes.global };
                            let scale = if local {
                                &operand.imported.weight.local_scale
                            } else {
                                &operand.imported.weight.global_scale
                            };
                            if let Some(scale) = scale {
                                companions.push((scope, scale.tensor.tensor.as_str()));
                            }
                        }
                        let mut table = None;
                        for (index, (scope, name)) in companions.iter().enumerate() {
                            match scope {
                                crate::contracts::CompanionScope::Table => {
                                    if table.is_some_and(|previous| previous != *name) {
                                        return Err(Error(format!(
                                            "lookup {role} companion scope requires one shared table tensor"
                                        )));
                                    }
                                    table = Some(*name);
                                }
                                crate::contracts::CompanionScope::Shard => {
                                    if companions.iter().enumerate().any(|(other, (_, source))| {
                                        other != index && source == name
                                    }) {
                                        return Err(Error(format!(
                                            "lookup {role} companion scope refuses sharing across shards"
                                        )));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        for projection in &self.projections {
            if projection.assembly.is_empty() {
                continue;
            }
            let layers = projection
                .stem()
                .map(|id| {
                    self.layer_stems
                        .iter()
                        .find(|s| s.id == id)
                        .ok_or_else(|| Error("assembly owning stem is missing".into()))?
                        .indices
                        .resolve(geometry)
                })
                .transpose()?
                .map_or_else(
                    || vec![None],
                    |layers| layers.into_iter().map(Some).collect(),
                );
            for current in layers {
                let mut tensors = Vec::new();
                for member in &projection.assembly {
                    let layer = member
                        .layer
                        .as_ref()
                        .map(|selector| selector.resolve(geometry, current))
                        .transpose()?;
                    let source = self
                        .projections
                        .iter()
                        .find(|p| p.id == member.projection)
                        .ok_or_else(|| Error("assembly source is missing".into()))?;
                    let operand = member
                        .operand
                        .as_ref()
                        .map(|name| {
                            source
                                .operand_index(name)
                                .ok_or_else(|| Error("assembly source operand is missing".into()))
                        })
                        .transpose()?;
                    let selected: Vec<_> = bound
                        .operands
                        .iter()
                        .filter(|b| {
                            b.projection == member.projection
                                && layer.is_none_or(|l| b.layer == Some(l))
                                && operand
                                    .is_none_or(|o| b.operand.is_none_or(|selected| selected == o))
                        })
                        .map(|b| b.imported.weight.codes.tensor.clone())
                        .collect();
                    if selected.is_empty() {
                        return Err(Error(format!(
                            "assembly `{}` source `{}` has no bound operands",
                            projection.id, member.projection
                        )));
                    }
                    tensors.push(selected);
                }
                bound.assemblies.push(BoundAssembly {
                    projection: projection.id.clone(),
                    layer: current,
                    members: projection.assembly.clone(),
                    tensors,
                });
            }
        }

        for side in &self.side_tensors {
            let shape = side
                .shape
                .iter()
                .map(|d| d.resolve(geometry))
                .collect::<Result<Vec<_>, _>>()?;
            for instance in instances(self, &side.source, &[], geometry, &[])? {
                if bound.operands.len() + bound.side_tensors.len() >= crate::validate::MAX_INSTANCES
                {
                    return Err(Error("binding exceeds source instance limit".into()));
                }
                let tensor = &instance.tensor;
                if let Some(previous) = primary_claims.get(tensor) {
                    if !side.consumer.matches_declared(previous) {
                        return Err(Error(format!(
                            "tensor `{tensor}` is claimed by `{previous}` and `{}`",
                            side.id
                        )));
                    }
                } else {
                    primary_claims.insert(tensor.clone(), side.id.clone());
                }
                let found = metadata.tensors.get(tensor).ok_or_else(|| {
                    Error(format!(
                        "checkpoint has no tensor `{tensor}` (side `{}`)",
                        side.id
                    ))
                })?;
                let admitted = side
                    .admitted_dtypes()
                    .iter()
                    .any(|dtype| dtype.checkpoint_dtype() == Some(found.dtype.as_str()));
                if !admitted {
                    return Err(Error(format!(
                        "tensor `{tensor}`: dtype {} is not admitted for side `{}`",
                        found.dtype, side.id
                    )));
                }
                requirement(
                    &TensorRequirement {
                        tensor: tensor.clone(),
                        dtype: found.dtype.clone(),
                        shape: shape.clone(),
                        file: None,
                    },
                    metadata,
                    &mut claimed,
                )?;
                if side.views.is_empty() {
                    let dtype = side
                        .admitted_dtypes()
                        .iter()
                        .find(|d| d.checkpoint_dtype() == Some(found.dtype.as_str()))
                        .copied()
                        .ok_or_else(|| Error("side source has no admitted dtype".into()))?;
                    bound.side_views.push(BoundSideView {
                        side: side.id.clone(),
                        tensor: tensor.clone(),
                        view: ConsumerView {
                            consumer: side.consumer.clone(),
                            dtype,
                            conversion: crate::contracts::ViewConversion::Identity,
                        },
                    });
                }
                for view in &side.views {
                    bound.side_views.push(BoundSideView {
                        side: side.id.clone(),
                        tensor: tensor.clone(),
                        view: view.clone(),
                    });
                }
                bound.side_tensors.push(tensor.clone());
            }
        }
        let mut present_owners = std::collections::BTreeSet::new();
        for external in &self.external_tensors {
            if instances(self, &external.source, &[], geometry, &[])?
                .iter()
                .any(|instance| metadata.tensors.contains_key(&instance.tensor))
            {
                present_owners.insert(external.owner.as_str());
            }
        }
        for external in &self.external_tensors {
            if external.optional && !present_owners.contains(external.owner.as_str()) {
                continue;
            }
            let shape = crate::contract_validation::shape(&external.shape, geometry)?;
            for instance in instances(self, &external.source, &[], geometry, &[])? {
                if external.optional && !metadata.tensors.contains_key(&instance.tensor) {
                    return Err(Error(format!(
                        "partial external owner `{}` is missing tensor `{}`",
                        external.owner, instance.tensor
                    )));
                }
                if !metadata
                    .validated_owners
                    .get(&external.owner)
                    .is_some_and(|names| names.contains(&instance.tensor))
                {
                    return Err(Error(format!(
                        "external tensor `{}` was not validated by owner `{}`",
                        instance.tensor, external.owner
                    )));
                }
                requirement(
                    &TensorRequirement {
                        tensor: instance.tensor.clone(),
                        dtype: external
                            .dtype
                            .checkpoint_dtype()
                            .ok_or_else(|| Error("unsupported external dtype".into()))?
                            .into(),
                        shape: shape.clone(),
                        file: None,
                    },
                    metadata,
                    &mut claimed,
                )?;
                bound.external_tensors.push(instance.tensor);
            }
        }
        if let Some(tensor) = metadata.tensors.keys().find(|t| !claimed.contains_key(*t)) {
            return Err(Error(format!("unknown checkpoint tensor `{tensor}`")));
        }
        for group in &self.fusion_groups {
            let first = &group.members[0];
            let mut layers: std::collections::BTreeSet<_> = bound
                .operands
                .iter()
                .filter(|b| b.projection == *first)
                .map(|b| b.layer)
                .collect();
            if layers.is_empty() {
                layers.extend(
                    bound
                        .assemblies
                        .iter()
                        .filter(|a| a.projection == *first)
                        .map(|a| a.layer),
                );
            }
            for layer in layers {
                let assembly_names: std::collections::BTreeSet<_> = bound
                    .assemblies
                    .iter()
                    .filter(|a| group.members.contains(&a.projection) && a.layer == layer)
                    .flat_map(|a| a.tensors.iter().flatten())
                    .collect();
                let operands: Vec<_> = bound
                    .operands
                    .iter()
                    .filter(|b| {
                        (group.members.contains(&b.projection) && b.layer == layer)
                            || assembly_names.contains(&b.imported.weight.codes.tensor)
                    })
                    .collect();
                let ids: std::collections::BTreeSet<_> = operands
                    .iter()
                    .map(|b| {
                        let projection = self
                            .projections
                            .iter()
                            .find(|p| p.id == b.projection)
                            .unwrap();
                        projection
                            .format
                            .map(Ok)
                            .unwrap_or_else(|| {
                                formats::by_format(b.imported.weight.format).map(|p| p.id())
                            })
                            .map_err(|e| Error(e.to_string()))
                    })
                    .collect::<Result<_, _>>()?;
                let projection = self.projections.iter().find(|p| p.id == *first).unwrap();
                for use_ in &projection.uses {
                    let activations = group
                        .members
                        .iter()
                        .map(|id| {
                            self.projections
                                .iter()
                                .find(|p| p.id == *id)
                                .unwrap()
                                .uses
                                .iter()
                                .find(|u| {
                                    u.phase == use_.phase
                                        && u.capacity.resolve(geometry).ok()
                                            == use_.capacity.resolve(geometry).ok()
                                        && u.graph == use_.graph
                                })
                                .and_then(|u| u.contract.as_ref())
                                .and_then(|c| c.activation)
                        })
                        .collect::<Vec<_>>();
                    let fused = group.eligibility.admits(&ids, use_.phase, &activations);
                    if !fused && group.fallback == Fallback::Refuse {
                        return Err(Error(format!(
                            "fusion `{}` eligibility failed and fallback refuses",
                            group.id
                        )));
                    }
                    if fused && let Some(composition) = &group.composition {
                        composition.numerical_for_sources(ids.iter().copied())?;
                    }
                    bound.fusion_groups.push(BoundFusion {
                        id: group.id.clone(),
                        layer,
                        members: group.members.clone(),
                        fused,
                        phase: use_.phase,
                        capacity: use_.capacity.resolve(geometry)?,
                    });
                }
            }
        }
        for stem in &self.layer_stems {
            if let Some(execution) = &stem.execution {
                for alias in &execution.aliases {
                    bound.layer_aliases.push(BoundLayerAlias {
                        execution_stem: stem.id.clone(),
                        execution_index: alias.execution_index.resolve_offset(geometry)?,
                        source_stem: alias.source_stem.clone().unwrap_or_else(|| stem.id.clone()),
                        source_index: alias.source_index.resolve_offset(geometry)?,
                    });
                }
            }
        }
        bound.conditions = self.bind_conditions(&bound);
        Ok(bound)
    }
}

/// The safetensors tensor record. Metadata strings are handled separately.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct HeaderTensor {
    dtype: String,
    shape: Vec<u64>,
    data_offsets: [u64; 2],
}

impl CheckpointMetadata {
    /// Adapt parsed safetensors headers and an optional `weight_map` index.
    /// File IO, no-follow checks and source authentication stay with the caller.
    pub fn from_headers(
        headers: &BTreeMap<String, Value>,
        index: Option<&Value>,
        quantization_config: Value,
    ) -> Result<Self, Error> {
        let weight_map = index
            .map(|i| {
                i.get("weight_map")
                    .and_then(Value::as_object)
                    .ok_or_else(|| Error("checkpoint index has no weight_map object".into()))
            })
            .transpose()?;
        let mut tensors = BTreeMap::new();
        for (file, header) in headers {
            let object = header
                .as_object()
                .ok_or_else(|| Error(format!("header `{file}` is not an object")))?;
            let mut spans = Vec::new();
            for (tensor, info) in object {
                if tensor == "__metadata__" {
                    let metadata: BTreeMap<String, String> =
                        serde_json::from_value(info.clone())
                            .map_err(|e| Error(format!("header `{file}` metadata: {e}")))?;
                    drop(metadata);
                    continue;
                }
                crate::validate::pattern(tensor, &[])?;
                let info: HeaderTensor = serde_json::from_value(info.clone())
                    .map_err(|e| Error(format!("tensor `{tensor}`: {e}")))?;
                let dimensions: Vec<_> = info
                    .shape
                    .iter()
                    .copied()
                    .map(crate::Dim::Literal)
                    .collect();
                let values = crate::contract_validation::shape(&dimensions, &Geometry::new())
                    .map_err(|e| Error(format!("tensor `{tensor}`: {e}")))?;
                let elements = values.iter().product::<u64>();
                let bytes = match info.dtype.as_str() {
                    "BOOL" | "U8" | "I8" | "F8_E4M3" | "F8_E5M2" | "F8_E8M0" => 1,
                    "BF16" | "F16" | "I16" | "U16" => 2,
                    "F32" | "I32" | "U32" => 4,
                    "F64" | "I64" | "U64" => 8,
                    _ => {
                        return Err(Error(format!(
                            "tensor `{tensor}`: unknown header dtype {}",
                            info.dtype
                        )));
                    }
                };
                if info.data_offsets[1].checked_sub(info.data_offsets[0])
                    != elements.checked_mul(bytes)
                {
                    return Err(Error(format!(
                        "tensor `{tensor}`: header byte offsets disagree with dtype and shape"
                    )));
                }
                spans.push((info.data_offsets, tensor));
                if let Some(map) = weight_map
                    && map.get(tensor).and_then(Value::as_str) != Some(file.as_str())
                {
                    return Err(Error(format!(
                        "tensor `{tensor}`: header file `{file}` disagrees with index"
                    )));
                }
                if tensors
                    .insert(
                        tensor.clone(),
                        TensorInfo {
                            dtype: info.dtype,
                            shape: info.shape,
                            file: Some(file.clone()),
                        },
                    )
                    .is_some()
                {
                    return Err(Error(format!(
                        "tensor `{tensor}` occurs in multiple headers"
                    )));
                }
            }
            spans.sort_by_key(|(offsets, _)| offsets[0]);
            for pair in spans.windows(2) {
                if pair[0].0[1] > pair[1].0[0] {
                    return Err(Error(format!(
                        "tensor `{}`: header byte offsets overlap",
                        pair[1].1
                    )));
                }
            }
        }
        if let Some(map) = weight_map {
            for tensor in map.keys() {
                if !tensors.contains_key(tensor) {
                    return Err(Error(format!("indexed tensor `{tensor}` has no header")));
                }
            }
        }
        Ok(Self {
            tensors,
            quantization_config,
            validated_owners: BTreeMap::new(),
        })
    }
}

fn bind_parts(
    projection: &Projection,
    imported: &ImportedOperand,
    metadata: &CheckpointMetadata,
    geometry: &Geometry,
    plugin: &dyn kernel_core::precision::format::FormatPlugin,
) -> Result<Vec<Vec<CompanionBinding>>, Error> {
    let weight = &imported.weight;
    if !(2..=3).contains(&weight.logical_shape.len()) {
        return Err(Error("part source must be a matrix or bank".into()));
    }
    let chunks: Vec<_> = if projection.parts.is_empty() {
        vec![("operand", projection.shape[0].resolve(geometry)?, [0, 0])]
    } else {
        projection
            .parts
            .iter()
            .map(|p| Ok((p.id.as_str(), p.rows.resolve(geometry)?, p.scale_origin)))
            .collect::<Result<_, Error>>()?
    };
    let expert_count = if weight.logical_shape.len() == 3 {
        weight.logical_shape[0]
    } else {
        1
    };
    let source_rows = weight.logical_shape[weight.logical_shape.len() - 2];
    let rows = projection.shape[0].resolve(geometry)?;
    if source_rows % rows != 0 {
        return Err(Error(
            "part source rows do not contain whole operands".into(),
        ));
    }
    let operands = source_rows / rows;
    let frames = crate::validate::multiply(
        crate::validate::multiply(expert_count, operands)?,
        chunks.len() as u64,
    )?;
    if frames != imported.parts.len() as u64 {
        return Err(Error(format!(
            "tensor `{}`: independent grids need one declared matrix part per source view",
            weight.codes.tensor
        )));
    }
    weight.codes.validate().map_err(|e| Error(e.to_string()))?;
    let mut code_end = 0u64;
    let mut bindings = Vec::new();
    let mut scale_views: BTreeMap<String, Vec<(u64, u64)>> = BTreeMap::new();
    for (index, spec) in imported.parts.iter().enumerate() {
        let (part_id, part_rows, scale_origin) = chunks[index % chunks.len()];
        let same_source = spec.format == weight.format
            && spec.encoding == weight.encoding
            && spec.layout == weight.layout
            && spec.codes.tensor == weight.codes.tensor
            && spec.codes.dtype == weight.codes.dtype;
        if !same_source
            || spec.logical_shape != vec![part_rows, projection.shape[1].resolve(geometry)?]
            || spec.codes.offset != code_end
        {
            return Err(Error(format!(
                "tensor `{}`: part `{part_id}` source, geometry or code range disagrees",
                weight.codes.tensor
            )));
        }
        spec.validate().map_err(|e| Error(e.to_string()))?;
        if spec
            .local_scale
            .iter()
            .flat_map(|s| &s.groups)
            .any(|g| g.axis < 2 && g.origin != scale_origin[usize::from(g.axis)])
        {
            return Err(Error(format!(
                "part `{part_id}` scale grid origin disagrees with declaration"
            )));
        }
        code_end = spec
            .codes
            .offset
            .checked_add(spec.codes.byte_length)
            .ok_or_else(|| Error("part code range overflows".into()))?;
        if code_end > weight.codes.byte_length {
            return Err(Error("part code span exceeds full source".into()));
        }
        for (view, source) in [
            (&spec.local_scale, &weight.local_scale),
            (&spec.global_scale, &weight.global_scale),
        ] {
            match (view, source) {
                (None, None) => {}
                (Some(view), Some(source)) => {
                    let span = &view.tensor;
                    let full = &source.tensor;
                    full.validate().map_err(|e| Error(e.to_string()))?;
                    let end = span
                        .offset
                        .checked_add(span.byte_length)
                        .ok_or_else(|| Error("part scale range overflows".into()))?;
                    if span.tensor != full.tensor
                        || span.dtype != full.dtype
                        || end > full.byte_length
                    {
                        return Err(Error(format!(
                            "part `{part_id}` companion view exceeds source `{}`",
                            full.tensor
                        )));
                    }
                    if !metadata.tensors.contains_key(&full.tensor) {
                        return Err(Error(format!("checkpoint has no tensor `{}`", full.tensor)));
                    }
                    scale_views
                        .entry(full.tensor.clone())
                        .or_default()
                        .push((span.offset, end));
                }
                _ => {
                    return Err(Error(
                        "part companion roles disagree with full source".into(),
                    ));
                }
            }
        }
        bindings.push(plugin.companions(spec).map_err(|e| Error(e.to_string()))?);
    }
    if code_end != weight.codes.byte_length {
        return Err(Error("part code views do not cover the full source".into()));
    }
    for source in weight.local_scale.iter().chain(&weight.global_scale) {
        let mut spans = scale_views
            .remove(&source.tensor.tensor)
            .unwrap_or_default();
        spans.sort();
        let mut end = 0;
        for (start, next) in spans {
            if start > end {
                return Err(Error(format!(
                    "part companions leave a gap in `{}`",
                    source.tensor.tensor
                )));
            }
            end = end.max(next);
        }
        if end != source.tensor.byte_length {
            return Err(Error(format!(
                "part companions do not cover `{}`",
                source.tensor.tensor
            )));
        }
    }
    Ok(bindings)
}

/// A bank's resident name: `template` without its `{expert}` placeholders.
/// A placeholder that ends a segment goes with the delimiter (`.`, `_`, `-`
/// or `/`) before it, or after it at the start; one joined to the text or
/// placeholder after it, or to text before it, goes alone.
fn bank_name(template: &str) -> String {
    let delimiter = |c: char| matches!(c, '.' | '_' | '-' | '/');
    let mut name = template.to_owned();
    while let Some(at) = name.find("{expert}") {
        let end = at + "{expert}".len();
        let before = name[..at].chars().next_back();
        let after = name[end..].chars().next();
        let range = match (before, after) {
            (_, Some(c)) if !delimiter(c) => at..end,
            (Some(c), _) if delimiter(c) => at - c.len_utf8()..end,
            (None, Some(c)) => at..end + c.len_utf8(),
            _ => at..end,
        };
        name.replace_range(range, "");
    }
    name
}

#[cfg(test)]
mod tests {
    use super::bank_name;

    #[test]
    fn bank_names_drop_only_the_placeholder_and_one_delimiter() {
        for (template, bank) in [
            (
                "layers.{layer}.experts.{expert}.down",
                "layers.{layer}.experts.down",
            ),
            ("specialists_{expert}.down", "specialists.down"),
            ("experts.{expert}", "experts"),
            ("{expert}.down", "down"),
            ("{expert}gate.weight", "gate.weight"),
            ("{expert}date.weight", "date.weight"),
            ("gate{expert}.weight", "gate.weight"),
            ("{operand}{expert}.weight", "{operand}.weight"),
            ("{expert}{operand}.weight", "{operand}.weight"),
            (
                "experts.{expert}{operand}.weight",
                "experts.{operand}.weight",
            ),
            ("{expert}αgate.weight", "αgate.weight"),
            ("{expert}βgate.weight", "βgate.weight"),
            ("gateα{expert}.weight", "gateα.weight"),
            ("experts-{expert}/down", "experts/down"),
            ("{expert}/down", "down"),
        ] {
            assert_eq!(bank_name(template), bank, "{template}");
        }
    }
}
