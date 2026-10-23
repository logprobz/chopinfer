//! The precision exercise's weight plan: the packed weight image, planned
//! from the checkpoint's safetensors headers and the package's exercise
//! document alone, with no importer.
//!
//! For each operation, in image order, the codes, local-scale and
//! global-scale spans are `PACKED_OPERAND` records over the operation's
//! checkpoint tensors, and the input-scale span, when the checkpoint has a
//! static calibration, copies that tensor unchanged. Bytes between spans are
//! zero. The host executes the records; the image is byte for byte what the
//! SDK's prepared-weight builder writes for the same operation.
use crate::precision::{self as model, Exercise, IMAGE_ALIGNMENT};
use kernel_core::precision::execution::MAX_PRECISION_DOCUMENT_BYTES;
use kernel_core::weight_transform::{PackedOperand, PackedSection, WeightTransform};
use runtime_bundle_abi::weight_plan::{
    CheckpointFileV1, RecordSourceV1, WeightPlanRequestV1, WeightPlanV1, WeightRecordV1,
};
use std::collections::BTreeMap;

/// One checkpoint tensor as its header declares it.
struct Tensor {
    file: u32,
    shape: Vec<u64>,
    byte_length: u64,
}

/// Every tensor of the checkpoint's safetensors headers, by name.
fn tensors(files: &[CheckpointFileV1]) -> Result<BTreeMap<String, Tensor>, String> {
    let mut tensors = BTreeMap::new();
    for (index, file) in files.iter().enumerate() {
        let header: serde_json::Map<String, serde_json::Value> =
            serde_json::from_slice(&file.header)
                .map_err(|e| format!("{}: safetensors header: {e}", file.name))?;
        for (name, entry) in header {
            if name == "__metadata__" {
                continue;
            }
            let invalid = || format!("{}: tensor {name} has an invalid header entry", file.name);
            let shape = entry["shape"]
                .as_array()
                .ok_or_else(invalid)?
                .iter()
                .map(|n| n.as_u64().ok_or_else(invalid))
                .collect::<Result<Vec<_>, _>>()?;
            let (start, end) = match entry["data_offsets"].as_array().map(Vec::as_slice) {
                Some([start, end]) => (
                    start.as_u64().ok_or_else(invalid)?,
                    end.as_u64().ok_or_else(invalid)?,
                ),
                _ => return Err(invalid()),
            };
            let tensor = Tensor {
                file: u32::try_from(index).map_err(|e| e.to_string())?,
                shape,
                byte_length: end.checked_sub(start).ok_or_else(invalid)?,
            };
            if tensors.insert(name.clone(), tensor).is_some() {
                return Err(format!("tensor {name} appears in more than one file"));
            }
        }
    }
    Ok(tensors)
}

/// Plans the packed weight image, the package's only planned image, for the
/// role `model::WEIGHTS_ROLE` or the primary (empty) role; declines every
/// other role.
pub fn plan(request: &WeightPlanRequestV1) -> Result<Option<WeightPlanV1>, String> {
    if !request.image_role.is_empty() && request.image_role != model::WEIGHTS_ROLE {
        return Ok(None);
    }
    if request.ranks != 1 || request.rank != 0 {
        return Err(format!(
            "the precision exercise plans one rank, not rank {} of {}",
            request.rank, request.ranks
        ));
    }
    let document = request
        .component(model::PRECISION_ROLE)
        .ok_or("the package carries no precision exercise document")?;
    if document.len() > MAX_PRECISION_DOCUMENT_BYTES {
        return Err("precision exercise document exceeds its bound".into());
    }
    let exercise: Exercise = serde_json::from_slice(document).map_err(|e| e.to_string())?;
    exercise.validate()?;
    let identity: String = request
        .checkpoint_identity
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    if identity != exercise.checkpoint_identity {
        return Err(
            "the checkpoint's safetensors headers differ from those the package was prepared for"
                .into(),
        );
    }
    let tensors = tensors(&request.files)?;
    let source = |name: &str| -> Result<(RecordSourceV1, &Tensor), String> {
        let tensor = tensors
            .get(name)
            .ok_or_else(|| format!("the checkpoint has no tensor {name}"))?;
        Ok((
            RecordSourceV1 {
                file: tensor.file,
                tensor: name.into(),
            },
            tensor,
        ))
    };
    let mut records = Vec::new();
    for weights in &exercise.weights {
        let packing = &weights.packing;
        let sources = packing
            .sources
            .iter()
            .map(|name| source(name).map(|(source, _)| source))
            .collect::<Result<Vec<_>, _>>()?;
        for (section, span) in [
            (PackedSection::Codes, &weights.codes),
            (PackedSection::LocalScales, &weights.local_scales),
            (PackedSection::GlobalScales, &weights.global_scales),
        ] {
            let transform = WeightTransform::PackedOperand(Box::new(PackedOperand {
                local_encoding: None,
                plan: packing.plan.clone(),
                section,
                decode_order: packing.decode_order,
                matrices: packing.matrices.clone(),
            }));
            records.push(WeightRecordV1 {
                arena_offset: span.offset,
                byte_length: span.bytes,
                transform: transform.id(),
                params: transform.encode_params().map_err(|e| e.to_string())?,
                sources: sources.clone(),
                axis: None,
                ranges: vec![],
                stored_shape: vec![],
            });
        }
        if let Some(name) = &packing.input_scale_tensor {
            let (source, tensor) = source(name)?;
            if tensor.byte_length != weights.input_scale.bytes {
                return Err(format!(
                    "operation {}: calibration {name} differs from its span",
                    weights.operation
                ));
            }
            // A scalar tensor gathers as one element.
            let stored_shape = if tensor.shape.is_empty() {
                vec![1]
            } else {
                tensor.shape.clone()
            };
            records.push(WeightRecordV1 {
                arena_offset: weights.input_scale.offset,
                byte_length: weights.input_scale.bytes,
                transform: WeightTransform::Copy.id(),
                params: vec![],
                sources: vec![source],
                axis: None,
                ranges: vec![],
                stored_shape,
            });
        }
    }
    records.sort_by_key(|record| record.arena_offset);
    Ok(Some(WeightPlanV1 {
        ranks: 1,
        rank: 0,
        image_role: model::WEIGHTS_ROLE.into(),
        alignment: IMAGE_ALIGNMENT,
        arena_bytes: exercise.image_bytes,
        views: vec![],
        components: vec![],
        records,
    }))
}
