//! Validated views for expert-major, operand-major, part-major source storage.

use super::*;
use crate::quantization::import::import_operand_with_origins;
use kernel_core::contract::WeightFormat;
use kernel_core::precision::WeightSpec;
use kernel_core::precision::format::FormatPlugin;

type PartImport = (
    WeightSpec,
    Vec<WeightSpec>,
    Vec<String>,
    Vec<FrameCalibration>,
);

fn product(a: u64, b: u64) -> Result<u64, Error> {
    a.checked_mul(b)
        .ok_or_else(|| error("frame size overflows"))
}
fn grid(n: u64, origin: u64, group: u64) -> Result<u64, Error> {
    let last = origin
        .checked_add(n)
        .and_then(|v| v.checked_sub(1))
        .ok_or_else(|| error("scale grid overflows"))?;
    Ok(last / group - origin / group + 1)
}
fn view(full: &TensorSpan, offset: u64, shape: Vec<u64>) -> Result<TensorSpan, Error> {
    let bytes = product(
        kernel_core::precision::shape_elements(&full.tensor, &shape).map_err(error)?,
        kernel_core::precision::dtype_bits(full.dtype) / 8,
    )?;
    if offset
        .checked_add(bytes)
        .is_none_or(|end| end > full.byte_length)
    {
        return Err(error(format!(
            "companion or code view exceeds `{}`",
            full.tensor
        )));
    }
    let mut result = full.clone();
    result.shape = shape;
    result.offset = offset;
    result.byte_length = bytes;
    result.validate().map_err(error)?;
    Ok(result)
}

#[derive(Clone, Copy)]
pub(super) struct Frame {
    pub(super) expert: u64,
    pub(super) operand: u64,
    index: u64,
    experts: u64,
    operands: u64,
    frames: u64,
}
pub(super) fn frame(
    r: &BindingRequest<'_>,
    expert: u64,
    operand: u64,
    part: u64,
    chunks: u64,
) -> Result<Frame, Error> {
    let experts = match &r.projection.kind {
        ProjectionKind::ExpertBank { experts, .. } => experts.resolve(r.geometry)?,
        _ => 1,
    };
    let operands = u64::from(r.projection.operand_count());
    let expert = r.expert.unwrap_or(expert);
    let operand = r.operand.unwrap_or(operand);
    Ok(Frame {
        expert,
        operand,
        index: product(
            product(expert, operands)?
                .checked_add(operand)
                .ok_or_else(|| error("frame index overflows"))?,
            chunks,
        )?
        .checked_add(part)
        .ok_or_else(|| error("frame index overflows"))?,
        experts,
        operands,
        frames: product(product(experts, operands)?, chunks)?,
    })
}

pub(super) fn scalar(
    full: &TensorSpan,
    sharing: Option<ScalarSharing>,
    f: Frame,
) -> Result<TensorSpan, Error> {
    let count = full.elements().map_err(error)?;
    let sharing = sharing.unwrap_or(if count == 1 {
        ScalarSharing::Shared
    } else {
        ScalarSharing::Expert
    });
    let (expected, index) = match sharing {
        ScalarSharing::Shared => (1, 0),
        ScalarSharing::Expert => (f.experts, f.expert),
        ScalarSharing::Operand => (f.operands, f.operand),
        ScalarSharing::Frame => (f.frames, f.index),
    };
    if count != expected {
        return Err(error(format!(
            "scalar companion `{}` cardinality disagrees with declared sharing",
            full.tensor
        )));
    }
    let scalar_shape =
        matches!(sharing, ScalarSharing::Shared) && (full.shape.is_empty() || full.shape == [1]);
    if !scalar_shape && full.shape != [expected] {
        return Err(error(format!(
            "scalar companion `{}` shape disagrees with declared sharing",
            full.tensor
        )));
    }
    view(
        full,
        product(index, kernel_core::precision::dtype_bits(full.dtype) / 8)?,
        vec![],
    )
}

pub(super) fn import(
    export: &ExportMetadata,
    role: &OperandRole,
    tensors: &BTreeMap<String, TensorSpan>,
    request: &BindingRequest<'_>,
    hints: &HeaderRole,
    plugin: &dyn FormatPlugin,
) -> Result<PartImport, Error> {
    let shape = &role.logical_shape;
    if !(2..=3).contains(&shape.len()) {
        return Err(error("independent sources need a matrix or bank"));
    }
    let experts = if shape.len() == 3 { shape[0] } else { 1 };
    let rows = request.projection.shape[0].resolve(request.geometry)?;
    let k = *shape.last().unwrap();
    let source_rows = shape[shape.len() - 2];
    if source_rows % rows != 0 {
        return Err(error("source has incomplete operand frames"));
    }
    let operands = source_rows / rows;
    let default_part = projection_spec::Part {
        id: "operand".into(),
        rows: projection_spec::Dim::Literal(rows),
        heads: None,
        range: None,
        scale_origin: [0, 0],
        head_runs: vec![],
        transforms: vec![],
    };
    let chunks = if request.projection.parts.is_empty() {
        vec![&default_part]
    } else {
        request.projection.parts.iter().collect::<Vec<_>>()
    };
    let frames = product(product(experts, operands)?, chunks.len() as u64)?;
    if frames > 1_000_000 {
        return Err(error("too many independent frames"));
    }
    let recipe = export.recipe(&role.targets, role.declared).map_err(error)?;
    let groups = match recipe.weight {
        WeightFormat::Bf16 | WeightFormat::Fp8PerTensor => None,
        WeightFormat::Fp8Block128x128(_) => Some((128, 128)),
        WeightFormat::Mxfp8 | WeightFormat::Mxfp4 => Some((1, 32)),
        WeightFormat::Nvfp4 => Some((1, 16)),
        _ => return Err(error("unsupported independent source weight format")),
    };
    let packed = matches!(recipe.weight, WeightFormat::Nvfp4 | WeightFormat::Mxfp4);
    if packed && k % 2 != 0 {
        return Err(error("packed source K must be even"));
    }
    let code = tensors
        .get(&role.weight)
        .ok_or_else(|| error("missing source codes"))?;
    let mut stored = role.logical_shape.clone();
    *stored.last_mut().unwrap() = if packed { k / 2 } else { k };
    if code.shape != stored {
        return Err(error(
            "code header shape disagrees with declared row-major frames",
        ));
    }
    let stem = role
        .weight
        .strip_suffix(".weight")
        .ok_or_else(|| error("missing weight stem"))?;
    let local_key = if tensors.contains_key(&format!("{stem}.weight_scale_inv")) {
        format!("{stem}.weight_scale_inv")
    } else {
        format!("{stem}.weight_scale")
    };
    let local = tensors.get(&local_key);
    let global_key = format!("{stem}.weight_scale_2");
    let calibration_key = format!("{stem}.input_scale");
    if let (Some(local), Some((br, bk))) = (local, groups) {
        let mut scale_rows = 0;
        let mut columns = None;
        for part in &chunks {
            scale_rows += grid(
                part.rows.resolve(request.geometry)?,
                part.scale_origin[0],
                br,
            )?;
            let width = grid(k, part.scale_origin[1], bk)?;
            if columns.is_some_and(|c| c != width) {
                return Err(error("part scale K grids differ"));
            }
            columns = Some(width);
        }
        let mut stored = if shape.len() == 3 {
            vec![experts]
        } else {
            vec![]
        };
        stored.extend([
            product(scale_rows, operands)?,
            columns.ok_or_else(|| error("missing scale grid"))?,
        ]);
        if local.shape != stored {
            return Err(error(
                "scale header shape disagrees with independent frame grids",
            ));
        }
    }
    if recipe.weight == WeightFormat::Fp8PerTensor
        && local.is_some_and(|s| !s.shape.is_empty() && s.shape != [1])
    {
        return Err(error("scalar local scale header shape is not scalar"));
    }
    let mut code_end = 0;
    let mut scale_end = 0;
    let mut parts = Vec::new();
    let mut consumed = Vec::new();
    let mut calibrations = Vec::new();
    for expert in 0..experts {
        for operand in 0..operands {
            for part in &chunks {
                let part_rows = part.rows.resolve(request.geometry)?;
                let f = frame(
                    request,
                    expert,
                    operand,
                    parts.len() as u64 % chunks.len() as u64,
                    chunks.len() as u64,
                )?;
                let mut selected = tensors.clone();
                let codes = view(
                    code,
                    code_end,
                    vec![part_rows, if packed { k / 2 } else { k }],
                )?;
                code_end = code_end
                    .checked_add(codes.byte_length)
                    .ok_or_else(|| error("code extent overflows"))?;
                selected.insert(role.weight.clone(), codes);
                if let Some(local) = local {
                    let scale = if let Some((br, bk)) = groups {
                        let scale = view(
                            local,
                            scale_end,
                            vec![
                                grid(part_rows, part.scale_origin[0], br)?,
                                grid(k, part.scale_origin[1], bk)?,
                            ],
                        )?;
                        scale_end = scale_end
                            .checked_add(scale.byte_length)
                            .ok_or_else(|| error("scale extent overflows"))?;
                        scale
                    } else {
                        local.clone()
                    };
                    selected.insert(local_key.clone(), scale);
                }
                for (name, sharing) in [
                    (&global_key, hints.global_sharing),
                    (&calibration_key, hints.calibration_sharing),
                ] {
                    if let Some(full) = tensors.get(name) {
                        selected.insert(name.clone(), scalar(full, sharing, f)?);
                    }
                }
                let frame_role = OperandRole {
                    operation: role.operation.clone(),
                    weight: role.weight.clone(),
                    targets: role.targets.clone(),
                    logical_shape: vec![part_rows, k],
                    lookup: role.lookup,
                    shared_scale_stem: None,
                    declared: role.declared,
                };
                let imported =
                    import_operand_with_origins(export, &frame_role, &selected, part.scale_origin)
                        .map_err(error)?;
                plugin.companions(&imported.weight).map_err(error)?;
                if let Some(scale) = imported
                    .activation
                    .as_ref()
                    .and_then(|a| a.quantizer.as_ref())
                    .and_then(|q| {
                        if let kernel_core::precision::Calibration::Static(s) = &q.calibration {
                            Some(s)
                        } else {
                            None
                        }
                    })
                {
                    calibrations.push(FrameCalibration {
                        expert: f.expert,
                        operand: f.operand,
                        part: (!request.projection.parts.is_empty()).then(|| part.id.clone()),
                        scale: scale.clone(),
                    });
                }
                consumed.extend(imported.source_tensors);
                parts.push(imported.weight);
            }
        }
    }
    if code_end != code.byte_length {
        return Err(error("code frames do not cover the source"));
    }
    if groups.is_some() && local.is_some_and(|s| scale_end != s.byte_length) {
        return Err(error("scale frames do not cover the source"));
    }
    let mut aggregate = parts
        .first()
        .ok_or_else(|| error("missing independent frames"))?
        .clone();
    aggregate.codes = code.clone();
    aggregate.logical_shape = role.logical_shape.clone();
    if let Some(scale) = &mut aggregate.local_scale {
        let full = local.ok_or_else(|| error("missing local scale"))?;
        let dtype = scale.tensor.dtype;
        scale.tensor = full.clone();
        scale.tensor.dtype = dtype;
        for group in &mut scale.groups {
            group.axis += (shape.len() - 2) as u8;
            group.origin = 0;
        }
    }
    if let Some(scale) = &mut aggregate.global_scale {
        scale.tensor = tensors
            .get(&global_key)
            .ok_or_else(|| error("missing global scale"))?
            .clone();
        scale.scope = plugin
            .companions(&parts[0])
            .map_err(error)?
            .into_iter()
            .find(|binding| {
                binding.role == kernel_core::precision::format::CompanionRole::GlobalScale
            })
            .ok_or_else(|| error("plug-in omitted the global companion"))?
            .scope;
    }
    // Aggregate spans are provenance. Independent parts, not a fake uniform
    // bank, carry the validated scale geometry and arithmetic contract.
    Ok((aggregate, parts, consumed, calibrations))
}
