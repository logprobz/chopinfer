//! Projection derivation inputs built from operands that the caller already
//! admitted against authenticated checkpoint headers. The builder reads no
//! payload and invents no checkpoint recipe: every weight, scale grid and
//! activation comes from the admitted operand of its source tensor.

use std::collections::BTreeMap;

use kernel_core::contract::{ActivationFormat, DType, Target};
use kernel_core::precision::format::{A16Recipe, KernelFamily, registry};
use kernel_core::precision::{ActivationSpec, ArithmeticSpec, Calibration, Rounding, WeightSpec};
use projection_spec::bind::{BindingRequest, BoundDeclaration, CheckpointMetadata, CompanionRules};
use projection_spec::derive::{
    DerivationInput, GeometryMode, ImportedProjection, ImportedProjections, InstanceKey,
    RankDerivation, SourceMatrix, Topology, default_operation_id, hooks::VariantDeriver,
};
use projection_spec::registry::Registry;
use projection_spec::{Declaration, Error, Geometry, Projection, ProjectionKind};

use super::import::ImportedOperand;
use super::projection::metadata::{companion_requirements, static_calibrations};

fn error(e: impl std::fmt::Display) -> Error {
    Error(format!("admitted projection imports: {e}"))
}

/// Admitted operands keyed by their checkpoint tensor name.
pub type Admitted = BTreeMap<String, ImportedOperand>;

/// Companion rules over admitted operands. A projection with declared parts
/// reads an admitted bank whose leading axis holds the parts, such as
/// independently quantized row chunks; each part keeps the bank's scale
/// origin and becomes its own view.
pub struct AdmittedRules<'a>(pub &'a Admitted);

impl CompanionRules for AdmittedRules<'_> {
    fn import(
        &self,
        r: &BindingRequest<'_>,
        metadata: &CheckpointMetadata,
    ) -> Result<projection_spec::bind::ImportedOperand, Error> {
        let admitted = self
            .0
            .get(r.tensor)
            .ok_or_else(|| error("source was not admitted"))?;
        let mut weight = admitted.weight.clone();
        weight.validate().map_err(error)?;
        let mut parts = vec![];
        if !r.projection.parts.is_empty() {
            let [count, rows, k] = weight.logical_shape[..] else {
                return Err(error("independent parts require an admitted bank"));
            };
            if count != r.projection.parts.len() as u64 || r.logical_shape != [count * rows, k] {
                return Err(error("declared parts differ from the admitted bank"));
            }
            // Stored codes keep their own K extent: packed formats hold
            // several codes a byte.
            let [code_count, code_rows, code_k] = weight.codes.shape[..] else {
                return Err(error("independent parts require a stored code bank"));
            };
            if (code_count, code_rows) != (count, rows) {
                return Err(error("stored code bank differs from its parts"));
            }
            for (index, declared) in r.projection.parts.iter().enumerate() {
                if declared.rows.resolve(r.geometry)? != rows || declared.scale_origin != [0, 0] {
                    return Err(error(
                        "declared part changes the admitted scale origin or extent",
                    ));
                }
                let mut part = weight.clone();
                part.logical_shape = vec![rows, k];
                part.codes.shape = vec![rows, code_k];
                part.codes.byte_length /= count;
                part.codes.offset += index as u64 * part.codes.byte_length;
                if let Some(scale) = &mut part.local_scale {
                    let [_, block_rows, block_k] = scale_blocks(scale)?;
                    scale.tensor.shape = vec![rows.div_ceil(block_rows), k.div_ceil(block_k)];
                    scale.tensor.byte_length /= count;
                    scale.tensor.offset += index as u64 * scale.tensor.byte_length;
                    for group in &mut scale.groups {
                        group.axis = group
                            .axis
                            .checked_sub(1)
                            .ok_or_else(|| error("invalid bank scale axis"))?;
                    }
                }
                part.validate().map_err(error)?;
                parts.push(part);
            }
            weight.logical_shape = r.logical_shape.into();
            weight.codes.shape = vec![count * rows, code_k];
            if let Some(scale) = &mut weight.local_scale {
                let [_, block_rows, block_k] = scale_blocks(scale)?;
                scale.tensor.shape = vec![count * rows.div_ceil(block_rows), k.div_ceil(block_k)];
                for group in &mut scale.groups {
                    group.axis -= 1;
                }
            }
        } else if weight.logical_shape != r.logical_shape {
            return Err(error("declared shape differs from admitted source"));
        }
        // A static input calibration is claimed with its frame provenance,
        // as the header importer claims it. Leaving its tensor out of the
        // shown metadata is how a caller ignores it (for example when its
        // activations stay BF16): binding coverage closes over the shown
        // tensors, so an unshown calibration is neither claimed nor framed.
        // A shown one must be the checkpoint's whole header entry before
        // frames are sliced from it.
        let calibration = admitted
            .activation
            .as_ref()
            .and_then(|a| a.quantizer.as_ref())
            .and_then(|q| match &q.calibration {
                Calibration::Static(s) => Some(&s.tensor),
                _ => None,
            });
        let ignored = calibration
            .filter(|full| !metadata.tensors.contains_key(&full.tensor))
            .map(|full| full.tensor.as_str());
        if let Some(full) = calibration.filter(|_| ignored.is_none()) {
            let name = &full.tensor;
            let header = &metadata.tensors[name];
            let whole = full
                .shape
                .iter()
                .try_fold(1u64, |n, d| n.checked_mul(*d))
                .and_then(|n| n.checked_mul(kernel_core::precision::dtype_bits(full.dtype) / 8));
            if full.validate().is_err()
                || full.dtype.checkpoint_dtype() != Some(header.dtype.as_str())
                || full.shape != header.shape
                || full.offset != 0
                || whole != Some(full.byte_length)
            {
                return Err(error(format!(
                    "admitted calibration `{name}` is not its whole checkpoint tensor"
                )));
            }
        }
        let calibrations = match ignored {
            Some(_) => vec![],
            None => static_calibrations(r, admitted.activation.as_ref(), None)?,
        };
        let companions = companion_requirements(
            &admitted.weight,
            admitted
                .source_tensors
                .iter()
                .filter(|t| Some(t.as_str()) != ignored)
                .cloned(),
            metadata,
        )?;
        Ok(projection_spec::bind::ImportedOperand {
            weight,
            parts,
            companions,
            calibrations,
        })
    }
}

/// The block extent of each bank axis of an admitted local scale grid.
fn scale_blocks(scale: &kernel_core::precision::ScaleSpec) -> Result<[u64; 3], Error> {
    let mut blocks = [1; 3];
    for group in &scale.groups {
        *blocks
            .get_mut(group.axis as usize)
            .ok_or_else(|| error("invalid bank scale axis"))? = u64::from(group.size);
    }
    Ok(blocks)
}

/// One imported projection per bound source instance, plus one per
/// executed layer of each assembled projection, which reads its first
/// member's resident operands at the layer the member selects (the
/// assembly's own layer, or a fixed one). Expert banks concatenate their
/// admitted per-expert sources in expert order.
pub fn imported(
    declaration: &Declaration,
    bound: &BoundDeclaration,
    geometry: &Geometry,
    admitted: &Admitted,
) -> Result<ImportedProjections, Error> {
    let mut imported = BTreeMap::new();
    for frame in &bound.operands {
        let key = InstanceKey {
            projection: frame.projection.clone(),
            layer: frame.layer,
        };
        if imported.contains_key(&key) {
            continue;
        }
        let p = declaration
            .projections
            .iter()
            .find(|p| p.id == key.projection)
            .ok_or_else(|| error("missing source declaration"))?;
        let frames: Vec<_> = bound
            .operands
            .iter()
            .filter(|b| b.projection == key.projection && b.layer == key.layer)
            .collect();
        let names = if matches!(p.kind, ProjectionKind::ExpertBank { .. }) {
            declaration.instance_sources(geometry, &p.id, key.layer)?
        } else {
            vec![]
        };
        imported.insert(key, projection(p, &frames, &names, admitted)?);
    }
    for p in declaration
        .projections
        .iter()
        .filter(|p| !p.assembly.is_empty())
    {
        let source = &p.assembly[0];
        let member = |layer: Option<u64>| -> Result<ImportedProjection, Error> {
            let found = match &source.layer {
                Some(selector) => {
                    let key = InstanceKey {
                        projection: source.projection.clone(),
                        layer: Some(selector.resolve(geometry, layer)?),
                    };
                    imported.get(&key)
                }
                None => imported
                    .iter()
                    .find(|(key, _)| key.projection == source.projection)
                    .map(|(_, member)| member),
            };
            let mut own = found
                .ok_or_else(|| error("assembly source was not admitted"))?
                .clone();
            own.operation_id = default_operation_id(p);
            own.arithmetic.numerical_contract = p.numerical_contract.clone();
            Ok(own)
        };
        // A stemmed assembly has one instance per layer, an unstemmed one
        // a single instance.
        let layers = match p.stem() {
            Some(id) => declaration
                .layer_stems
                .iter()
                .find(|s| s.id == id)
                .ok_or_else(|| error("assembly execution stem is absent"))?
                .indices
                .resolve(geometry)?
                .into_iter()
                .map(Some)
                .collect(),
            None => vec![None],
        };
        let owned: Vec<_> = layers
            .into_iter()
            .map(|layer| Ok((layer, member(layer)?)))
            .collect::<Result<_, Error>>()?;
        for (layer, own) in owned {
            imported.insert(
                InstanceKey {
                    projection: p.id.clone(),
                    layer,
                },
                own,
            );
        }
    }
    Ok(imported)
}

/// The source matrices of one stacked expert bank: a single
/// `[experts, operands * rows, k]` source whose experts lead and whose
/// operands (gate then up, in declaration order) stack along rows. Each
/// expert and operand reads its rows of that expert's matrix in place.
fn stacked_matrices(
    p: &Projection,
    frame: &projection_spec::bind::BoundOperand,
    family: KernelFamily,
    [experts, stacked, k]: [u64; 3],
) -> Result<Vec<SourceMatrix>, Error> {
    if family != KernelFamily::GroupedExpert
        || frame.expert.is_some()
        || !frame.imported.parts.is_empty()
    {
        return Err(error(
            "a three-dimensional source must be one stacked expert bank without parts",
        ));
    }
    let operands = if frame.operand.is_some() {
        1
    } else {
        u64::from(p.operand_count())
    };
    if stacked % operands != 0 {
        return Err(error("stacked bank rows do not divide among its operands"));
    }
    let rows = stacked / operands;
    let narrow = |n: u64| u32::try_from(n).map_err(|_| error("bank index exceeds u32"));
    let mut matrices = vec![];
    for expert in 0..experts {
        for local in 0..operands {
            matrices.push(SourceMatrix {
                weight: frame.imported.weight.clone(),
                expert: narrow(expert)?,
                operand: narrow(frame.operand.unwrap_or(local))?,
                source_expert: expert,
                target_rows: 0..rows,
                source_rows: local * rows..(local + 1) * rows,
                target_k: 0..k,
                source_k: 0..k,
            });
        }
    }
    Ok(matrices)
}

/// One instance's import from its bound frames. An expert bank's operands
/// are named `banks`, the declared source names without the expert segment.
fn projection(
    p: &Projection,
    frames: &[&projection_spec::bind::BoundOperand],
    banks: &[String],
    admitted: &Admitted,
) -> Result<ImportedProjection, Error> {
    let family = match p.kind {
        ProjectionKind::Dense {} => KernelFamily::Dense,
        ProjectionKind::ExpertBank { .. } => KernelFamily::GroupedExpert,
        ProjectionKind::LookupTable {} => KernelFamily::Lookup,
    };
    let mut matrices = vec![];
    for frame in frames {
        if let [experts, stacked, k] = frame.imported.weight.logical_shape[..] {
            matrices.extend(stacked_matrices(p, frame, family, [experts, stacked, k])?);
            continue;
        }
        let parts = if frame.imported.parts.is_empty() {
            vec![&frame.imported.weight]
        } else {
            frame.imported.parts.iter().collect()
        };
        let mut start = 0;
        for weight in parts {
            let [rows, k] = weight.logical_shape[..] else {
                return Err(error("source matrix has no two-dimensional frame"));
            };
            matrices.push(SourceMatrix {
                weight: weight.clone(),
                expert: frame.expert.unwrap_or(0) as u32,
                operand: frame.operand.unwrap_or(0) as u32,
                source_expert: 0,
                target_rows: start..start + rows,
                source_rows: 0..rows,
                target_k: 0..k,
                source_k: 0..k,
            });
            start += rows;
        }
    }
    let mut weights: Vec<WeightSpec> = vec![];
    for operand in 0..p.operand_count() {
        let frames: Vec<_> = frames
            .iter()
            .filter(|b| match b.operand {
                Some(o) => o == u64::from(operand),
                // A stacked bank holds every operand of the projection.
                None if family == KernelFamily::GroupedExpert => true,
                None => operand == 0,
            })
            .collect();
        if let [frame] = frames[..]
            && frame.imported.weight.logical_shape.len() == 3
        {
            // Every operand keeps the stacked bank's whole span; its source
            // matrices select the operand's rows.
            weights.push(frame.imported.weight.clone());
        } else if family == KernelFamily::GroupedExpert {
            let sources = frames
                .iter()
                .map(|b| {
                    admitted
                        .get(&b.imported.weight.codes.tensor)
                        .cloned()
                        .ok_or_else(|| error("expert source was not admitted"))
                })
                .collect::<Result<Vec<_>, _>>()?;
            let bank = banks
                .get(operand as usize)
                .ok_or_else(|| error("expert bank has no declared source"))?;
            let bank = bank.strip_suffix(".weight").unwrap_or(bank);
            weights.push(
                super::bank::concatenate_experts(bank, &sources)
                    .map_err(error)?
                    .operand
                    .weight,
            );
        } else {
            let frame = frames
                .first()
                .ok_or_else(|| error("empty source operand"))?;
            weights.push(
                frame
                    .imported
                    .parts
                    .first()
                    .unwrap_or(&frame.imported.weight)
                    .clone(),
            );
        }
    }
    let plugin = registry::by_weight(&weights[0]).map_err(error)?;
    let recipe = p
        .composition
        .as_ref()
        .and_then(|c| c.numerical_contracts.get(&plugin.id()));
    let a16_recipe = recipe.map_or(A16Recipe::ScaledWeight, |r| r.a16_recipe);
    Ok(ImportedProjection {
        operation_id: default_operation_id(p),
        target: Target::Sm120f,
        family,
        format: plugin.id(),
        geometry: GeometryMode::RankLocal,
        weights,
        activations: if family == KernelFamily::Lookup {
            vec![]
        } else {
            vec![ActivationSpec::bf16(); p.operand_count() as usize]
        },
        arithmetic: ArithmeticSpec {
            weight_operand: DType::Bf16,
            accumulator: DType::F32,
            output: p.uses[0]
                .contract
                .as_ref()
                .map_or(DType::Bf16, |c| c.output_dtype),
            decode_order: plugin
                .decode_order(family, ActivationFormat::Bf16, a16_recipe)
                .map_err(error)?,
            rounding: Rounding::NearestEven,
            numerical_contract: recipe.map_or_else(
                || p.numerical_contract.clone(),
                |r| r.numerical_contract.clone(),
            ),
        },
        a16_recipe,
        matrices,
        preparation: None,
    })
}

/// Replace a derived operation's BF16 import activations with the admitted
/// sources' own activation recipes, so precision policy starts from the
/// checkpoint. A quantized activation takes the weight's native MMA operand
/// and its native decode order. Expert banks read the recipe of their first
/// expert's declared source.
pub fn admitted_activations(
    operation: &mut kernel_core::precision::OperationSpec,
    admitted: &Admitted,
    declaration: &Declaration,
    geometry: &Geometry,
) -> Result<(), Error> {
    use kernel_core::contract::WeightFormat;
    use kernel_core::precision::{DecodeOrder, OperationKind};
    if operation.activations.is_empty() {
        return Ok(());
    }
    let (projection, layer) = declaration.operation_instance(&operation.id)?;
    // An assembly reads its first member's materialized sources.
    let materialized = match projection.assembly.first() {
        Some(member) => declaration
            .projections
            .iter()
            .find(|p| p.id == member.projection)
            .ok_or_else(|| error("assembly member is not declared"))?,
        None => projection,
    };
    let first_expert = match projection.kind {
        // A stacked bank is one source for every expert.
        ProjectionKind::ExpertBank { .. }
            if materialized
                .source
                .as_ref()
                .is_some_and(|s| !s.tensor.contains("{expert}")) =>
        {
            Some(declaration.instance_sources(geometry, &projection.id, layer)?)
        }
        ProjectionKind::ExpertBank { .. } => {
            Some(declaration.expert_sources(geometry, &projection.id, layer, 0)?)
        }
        _ => None,
    };
    operation.activations = operation
        .weights
        .iter()
        .enumerate()
        .map(|(index, weight)| {
            let tensor = match &first_expert {
                // A stacked bank names one source for all of its operands.
                Some(names) => match names[..] {
                    [ref only] => only.clone(),
                    _ => names
                        .get(index)
                        .cloned()
                        .ok_or_else(|| error("expert bank operand has no declared source"))?,
                },
                None => weight
                    .codes
                    .tensor
                    .split('@')
                    .next()
                    .unwrap_or_default()
                    .into(),
            };
            admitted
                .get(&tensor)
                .and_then(|o| o.activation.clone())
                .ok_or_else(|| error(format!("missing admitted source activation for {tensor}")))
        })
        .collect::<Result<_, _>>()?;
    let format = operation.activations[0].format;
    if operation.activations.iter().any(|a| a.format != format) {
        return Err(error("fused source activations differ"));
    }
    if format == ActivationFormat::Bf16 {
        return Ok(());
    }
    operation.arithmetic.weight_operand = match operation.weights[0].format {
        WeightFormat::Nvfp4 | WeightFormat::Mxfp4 => DType::Fp4E2M1,
        _ => DType::Fp8E4M3,
    };
    // The checkpoint's recipe is recorded even where no kernel implements it;
    // the BF16 activation policy then binds a weight-reuse order instead.
    let grouped = matches!(operation.kind, OperationKind::Grouped { .. });
    operation.arithmetic.decode_order = match format {
        ActivationFormat::Bf16 => unreachable!("handled above"),
        ActivationFormat::Fp8Group128(_) if grouped => DecodeOrder::SuccessiveScalesAfterBlockMma,
        ActivationFormat::Fp8Group128(_) => DecodeOrder::ScaleProductAfterBlockMma,
        ActivationFormat::Mxfp8 => DecodeOrder::NativeBlockScaledMma,
        ActivationFormat::Nvfp4 => DecodeOrder::NativeBlockScaledMmaThenGlobals,
    };
    Ok(())
}

/// One rank's derivation over the caller's admitted operands.
pub struct RankInputs<'a> {
    /// The expanded declaration of the scope being served.
    pub declaration: &'a Declaration,
    pub geometry: &'a Geometry,
    pub registry: &'a Registry,
    pub variants: &'a dyn VariantDeriver,
    pub metadata: &'a CheckpointMetadata,
    pub admitted: &'a Admitted,
    pub topology: Topology,
}

/// Bind the admitted operands against the declaration, build their imported
/// projections and derive the rank.
pub fn derive(inputs: RankInputs<'_>) -> Result<RankDerivation, Error> {
    let bound = inputs.declaration.bind(
        inputs.geometry,
        inputs.registry,
        inputs.metadata,
        &AdmittedRules(inputs.admitted),
    )?;
    let imported = imported(inputs.declaration, &bound, inputs.geometry, inputs.admitted)?;
    projection_spec::derive::derive(DerivationInput {
        declaration: inputs.declaration,
        bound: &bound,
        geometry: inputs.geometry,
        registry: inputs.registry,
        topology: inputs.topology,
        imported: &imported,
        variants: inputs.variants,
    })
}
