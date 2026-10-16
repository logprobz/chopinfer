use crate::backend::{Kernels, Selected};
use kernel_core::contract::DType;
use kernel_core::precision::format::{A16Recipe, KernelFamily, registry};
use kernel_core::precision::{ActivationSpec, ArithmeticSpec, Rounding};
use projection_spec::bind::BoundDeclaration;
use projection_spec::derive::{GeometryMode, ImportedProjection, SourceMatrix};
use projection_spec::{Geometry, Projection, ProjectionKind};

pub(super) fn prototype(
    projection: &Projection,
    layer: Option<u64>,
    bound: &BoundDeclaration,
    geometry: &Geometry,
) -> ImportedProjection {
    let frames: Vec<_> = bound
        .operands
        .iter()
        .filter(|b| b.projection == projection.id && b.layer == layer)
        .collect();
    assert!(!frames.is_empty(), "{} {layer:?}", projection.id);
    let family = match projection.kind {
        ProjectionKind::ExpertBank { .. } => KernelFamily::GroupedExpert,
        ProjectionKind::LookupTable { .. } => KernelFamily::Lookup,
        _ => KernelFamily::Dense,
    };
    let mut matrices = Vec::new();
    for frame in &frames {
        let parts = if frame.imported.parts.is_empty() {
            vec![&frame.imported.weight]
        } else {
            frame.imported.parts.iter().collect()
        };
        let mut target = if family == KernelFamily::Lookup {
            frame.shard.unwrap_or(0) * frame.imported.weight.logical_shape[0]
        } else {
            0
        };
        for (index, weight) in parts.into_iter().enumerate() {
            let n = weight.logical_shape[weight.logical_shape.len() - 2];
            let k = *weight.logical_shape.last().unwrap();
            let start = projection
                .parts
                .get(index)
                .and_then(|p| p.range.as_ref())
                .map_or(target, |r| r.start.resolve_offset(geometry).unwrap());
            matrices.push(SourceMatrix {
                weight: weight.clone(),
                expert: frame.expert.unwrap_or(0) as u32,
                operand: frame.operand.unwrap_or(0) as u32,
                source_expert: 0,
                target_rows: start..start + n,
                source_rows: 0..n,
                target_k: 0..k,
                source_k: 0..k,
            });
            target += n;
        }
    }
    let mut weights = Vec::new();
    for operand in 0..projection.operand_count() {
        let selected: Vec<_> = frames
            .iter()
            .filter(|b| b.operand.unwrap_or(0) == u64::from(operand))
            .collect();
        if family == KernelFamily::GroupedExpert {
            let sources: Vec<_> = selected
                .iter()
                .map(
                    |b| model_compiler_sdk::quantization::import::ImportedOperand {
                        weight: b.imported.weight.clone(),
                        activation: Some(ActivationSpec::bf16()),
                        source_tensors: vec![b.imported.weight.codes.tensor.clone()],
                    },
                )
                .collect();
            let first = &sources[0].weight.codes.tensor;
            let (prefix, tail) = first.split_once(".experts.").unwrap();
            let suffix = tail.split_once('.').unwrap().1.trim_end_matches(".weight");
            weights.push(
                model_compiler_sdk::quantization::bank::concatenate_experts(
                    &format!("{prefix}.experts.{suffix}"),
                    &sources,
                )
                .unwrap()
                .operand
                .weight,
            );
        } else {
            let first = selected[0];
            weights.push(if first.imported.weight.validate().is_ok() {
                first.imported.weight.clone()
            } else {
                first.imported.parts[0].clone()
            });
        }
    }
    let plugin = registry::by_weight(&weights[0]).unwrap();
    let recipe = projection
        .composition
        .as_ref()
        .and_then(|c| c.numerical_contracts.get(&plugin.id()));
    let a16_recipe = recipe.map_or(A16Recipe::ScaledWeight, |r| r.a16_recipe);
    let output = projection.uses[0]
        .contract
        .as_ref()
        .map_or(DType::Bf16, |c| c.output_dtype);
    let arithmetic = ArithmeticSpec {
        weight_operand: DType::Bf16,
        accumulator: DType::F32,
        output,
        decode_order: plugin
            .decode_order(
                family,
                kernel_core::contract::ActivationFormat::Bf16,
                a16_recipe,
            )
            .unwrap(),
        rounding: Rounding::NearestEven,
        numerical_contract: recipe.map_or_else(
            || projection.numerical_contract.clone(),
            |r| r.numerical_contract.clone(),
        ),
    };
    let operation_id = if family == KernelFamily::GroupedExpert {
        weights[0]
            .codes
            .tensor
            .replace(".gate_proj.weight", ".gate_up_proj")
            .trim_end_matches(".weight")
            .into()
    } else if weights.len() == 2 {
        weights[0]
            .codes
            .tensor
            .replace(".gate_proj.weight", ".gate_proj+up_proj")
    } else {
        weights[0].codes.tensor.clone()
    };
    ImportedProjection {
        operation_id: operation_id.trim_end_matches(".weight").into(),
        target: Selected::TARGET,
        family,
        format: plugin.id(),
        geometry: GeometryMode::RankLocal,
        weights,
        activations: if family == KernelFamily::Lookup {
            vec![]
        } else {
            vec![ActivationSpec::bf16(); projection.operand_count() as usize]
        },
        arithmetic,
        a16_recipe,
        matrices,
        preparation: None,
    }
}
