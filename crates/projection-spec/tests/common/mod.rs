#![allow(dead_code)]

use std::collections::BTreeMap;

use kernel_core::contract::{DType, WeightFormat};
use kernel_core::precision::{CodeEncoding, TensorSpan, WeightLayout, WeightSpec};
use projection_spec::bind::{
    BindingRequest, CheckpointMetadata, CompanionRules, ImportedOperand, TensorInfo,
};
use projection_spec::registry::Registry;
use projection_spec::{Declaration, Error, Geometry, parse};

pub const FIXTURE: &str = include_str!("../fixtures/synthetic.json");

pub fn geometry() -> Geometry {
    BTreeMap::from([
        ("hidden".into(), 16),
        ("intermediate".into(), 12),
        ("experts".into(), 4),
        ("vocab".into(), 32),
        ("layers".into(), 2),
        ("prefill".into(), 128),
    ])
}

pub fn compact_declaration() -> Declaration {
    parse(FIXTURE, &geometry(), &Registry::default()).unwrap()
}

pub fn declaration() -> Declaration {
    compact_declaration()
        .expanded_profiles(&geometry())
        .unwrap()
}

pub fn info(shape: &[u64], dtype: &str) -> TensorInfo {
    TensorInfo {
        shape: shape.to_vec(),
        dtype: dtype.into(),
        file: Some("weights.safetensors".into()),
    }
}

pub fn metadata() -> CheckpointMetadata {
    let mut tensors = BTreeMap::new();
    for layer in 0..2 {
        for (suffix, shape) in [
            ("q.weight", vec![8, 16]),
            ("k.weight", vec![8, 16]),
            ("out.weight", vec![16, 8]),
            ("feed.gate.weight", vec![12, 16]),
            ("feed.up.weight", vec![12, 16]),
            ("experts.weight", vec![4, 12, 16]),
            ("norm.weight", vec![16]),
            ("router", vec![4, 16]),
        ] {
            tensors.insert(format!("layers.{layer}.{suffix}"), info(&shape, "BF16"));
        }
        tensors.insert(format!("layers.{layer}.gate"), info(&[8], "F32"));
        for expert in 0..4 {
            tensors.insert(
                format!("layers.{layer}.expert.{expert}.out.weight"),
                info(&[16, 12], "BF16"),
            );
        }
    }
    tensors.insert("embed.weight".into(), info(&[32, 16], "BF16"));
    tensors.insert("draft.0.proj.weight".into(), info(&[16, 16], "BF16"));
    tensors.insert("draft.0.conv".into(), info(&[16, 1, 3], "BF16"));
    for shard in 0..2 {
        for head in 0..2 {
            tensors.insert(format!("hash.{shard}.{head}.weight"), info(&[8, 4], "BF16"));
        }
    }
    CheckpointMetadata {
        tensors,
        quantization_config: serde_json::json!({"recipe":"synthetic"}),
        ..Default::default()
    }
}

pub fn bf16(
    request: &BindingRequest<'_>,
    metadata: &CheckpointMetadata,
) -> Result<ImportedOperand, Error> {
    if metadata.quantization_config["recipe"] != "synthetic" {
        return Err(Error("unsupported quantization recipe".into()));
    }
    let info = &metadata.tensors[request.tensor];
    let bytes = info.shape.iter().product::<u64>() * 2;
    let mut imported = ImportedOperand {
        weight: WeightSpec {
            codes: TensorSpan {
                tensor: request.tensor.into(),
                dtype: DType::Bf16,
                shape: info.shape.clone(),
                offset: 0,
                byte_length: bytes,
            },
            logical_shape: request.logical_shape.to_vec(),
            format: WeightFormat::Bf16,
            encoding: CodeEncoding::Bf16LittleEndian,
            layout: WeightLayout::row_major(),
            local_scale: None,
            global_scale: None,
            transforms: vec![],
        },
        parts: vec![],
        companions: vec![],
        calibrations: vec![],
    };
    if !request.projection.parts.is_empty() {
        let frames = request.logical_shape.iter().product::<u64>()
            / (request.projection.shape[0].resolve(request.geometry)?
                * request.projection.shape[1].resolve(request.geometry)?);
        let mut offset = 0;
        for _ in 0..frames {
            for part in &request.projection.parts {
                let rows = part.rows.resolve(request.geometry)?;
                let k = request.projection.shape[1].resolve(request.geometry)?;
                let mut view = imported.weight.clone();
                view.logical_shape = vec![rows, k];
                view.codes.shape = vec![rows, k];
                view.codes.offset = offset;
                view.codes.byte_length = rows * k * 2;
                offset += view.codes.byte_length;
                imported.parts.push(view);
            }
        }
    }
    Ok(imported)
}

pub struct Bf16Rules;
impl CompanionRules for Bf16Rules {
    fn import(
        &self,
        request: &BindingRequest<'_>,
        metadata: &CheckpointMetadata,
    ) -> Result<ImportedOperand, Error> {
        bf16(request, metadata)
    }
}
