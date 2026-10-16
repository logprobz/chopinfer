//! Pure preparation for registered MiMo behavior. The served loaders retain
//! their current paths; these records reuse their exact source selections.

use super::{RegroupParameters, require};
use crate::Tp;
use crate::backend::{Kernels, Selected};
use crate::catalog::TensorRecord;
use kernel_core::precision::{WeightSpec, dtype_bits};
use kernel_core::weight_transform::{PackedSection, SourceTensor, WeightRecord};
use model_compiler_sdk::quantization::import::ImportedOperand;
use projection_spec::Error;
use projection_spec::derive::VariantRecord;
use runtime_weights::DType;
use std::collections::BTreeMap;
use std::ops::Range;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct QkvPreparation {
    pub weight: WeightSpec,
    pub source_rows: Vec<Range<u64>>,
    pub records: Vec<VariantRecord>,
}

fn error(e: impl std::fmt::Display) -> Error {
    Error(e.to_string())
}

/// Retain the old rank operand and emit the loader's fragment/scale records.
/// The chunk bank owns four independent grids, not one flattened scale grid.
pub fn qkv(
    source: &WeightSpec,
    parameters: &RegroupParameters,
    tp: Tp,
    rank: usize,
) -> Result<QkvPreparation, Error> {
    source.validate().map_err(error)?;
    require(
        matches!(
            source.format,
            kernel_core::contract::WeightFormat::Fp8Block128x128(_)
        ) && source.transforms.is_empty(),
        "QKV regroup requires untransformed block-128 FP8 codes",
    )?;
    require(
        parameters.block_rows == 128
            && parameters.k_block == 128
            && parameters.output_group_rows == 64
            && parameters.chunks == 4,
        "QKV regroup differs from the registered scale grid",
    )?;
    require(rank < tp.count(), "QKV rank is outside the topology")?;
    require(
        parameters.rank_counts.contains(&tp.ranks())
            || parameters.bypass_ranks.contains(&tp.ranks()),
        "QKV regroup has no behavior for this topology",
    )?;
    let chunk_rows = parameters.chunk_rows(tp)?;
    require(
        source.logical_shape
            == [
                u64::from(parameters.chunks),
                chunk_rows,
                u64::from(tp.geometry.hidden()),
            ],
        "QKV source does not match the registered chunk geometry",
    )?;
    let local = source
        .local_scale
        .as_ref()
        .ok_or_else(|| error("QKV local scale is absent"))?;
    require(
        local.convention == kernel_core::precision::ScaleConvention::Multiply
            && local.groups.iter().all(|g| g.origin == 0),
        "QKV source scales must retain the restarting multiplicative grid",
    )?;
    require(
        local.tensor.dtype == kernel_core::contract::DType::F32,
        "served QKV regroup requires F32 scales",
    )?;
    let tensor = |name: &str, dtype, shape: Vec<u64>, offset, bytes| TensorRecord {
        name: name.into(),
        dtype,
        shape,
        file: String::new(),
        file_start: offset,
        bytes,
    };
    let codes = tensor(
        &source.codes.tensor,
        DType::F8E4M3,
        vec![
            u64::from(parameters.chunks) * chunk_rows,
            u64::from(tp.geometry.hidden()),
        ],
        source.codes.offset,
        source.codes.byte_length,
    );
    let scales = tensor(
        &local.tensor.tensor,
        DType::F32,
        vec![
            u64::from(parameters.chunks) * chunk_rows.div_ceil(u64::from(parameters.block_rows)),
            u64::from(tp.geometry.hidden()) / u64::from(parameters.k_block),
        ],
        local.tensor.offset,
        local.tensor.byte_length,
    );
    let tensors = BTreeMap::from([(scales.name.clone(), scales.clone())]);
    let imported = ImportedOperand {
        weight: source.clone(),
        activation: None,
        source_tensors: vec![codes.name.clone(), scales.name.clone()],
    };
    let global = crate::precision::qkv_is_global(tp.geometry, &codes.name).map_err(error)?;
    let mut weight =
        crate::precision::qkv_rank_weight(&imported, &tensors, tp, tp.qkv_rows(global))
            .map_err(error)?;
    if parameters.bypass_ranks.contains(&tp.ranks()) {
        let codes_offset = u64::try_from(rank)
            .map_err(error)?
            .checked_mul(weight.codes.byte_length)
            .ok_or_else(|| error("QKV code offset overflows"))?;
        weight.codes.offset = weight
            .codes
            .offset
            .checked_add(codes_offset)
            .ok_or_else(|| error("QKV code offset overflows"))?;
        if let Some(local) = &mut weight.local_scale {
            let offset = u64::try_from(rank)
                .map_err(error)?
                .checked_mul(local.tensor.byte_length)
                .ok_or_else(|| error("QKV scale offset overflows"))?;
            local.tensor.offset = local
                .tensor
                .offset
                .checked_add(offset)
                .ok_or_else(|| error("QKV scale offset overflows"))?;
        }
    }
    weight.validate().map_err(error)?;
    // The rank's head rows of every chunk, then their 64-row scale groups.
    let rows = if tp.ranks() == 4 {
        let chunk = codes.shape[0] / u64::from(parameters.chunks);
        let first = rank as u64 * chunk;
        std::iter::once(first..first + chunk).collect()
    } else {
        tp.qkv_segments(global, rank)
    };
    let grid = crate::precision::qkv_scale_grid(
        tp.geometry,
        global,
        scales.shape[0],
        scales.shape[1] as usize,
    )
    .map_err(error)?;
    let scale_rows = crate::precision::qkv_rank_scale_rows(grid, &rows, rank, tp).map_err(error)?;
    let k = codes.shape[1];
    let source_tensor = |tensor: &TensorRecord| SourceTensor {
        dtype: tensor.dtype.safetensors_name().into(),
        shape: tensor.shape.clone(),
        byte_length: tensor.bytes,
    };
    let codes_record = model_compiler_sdk::quantization::format::section_record(
        kernel_core::precision::format::registry::by_id(
            kernel_core::precision::format::FormatId::Fp8Block128,
        )
        .map_err(error)?,
        PackedSection::Codes,
        kernel_core::contract::DType::Fp8E4M3,
        kernel_core::contract::DType::Fp8E4M3,
        model_compiler_sdk::quantization::format::RecordLayout::Fragment,
        WeightRecord {
            transform: kernel_core::weight_transform::WeightTransform::Copy,
            sources: vec![source_tensor(&codes)],
            axis: Some(0),
            ranges: rows.clone(),
            stored_shape: vec![rows.iter().map(|r| r.end - r.start).sum(), k],
        },
    )
    .map_err(error)?;
    // TP4 copies its chunk's scale rows; other rank counts one row per group.
    let scale_ranges: Vec<Range<u64>> = if tp.ranks() == 4 {
        let first = scale_rows[0] as u64;
        std::iter::once(first..first + scale_rows.len() as u64).collect()
    } else {
        scale_rows.iter().map(|&r| r as u64..r as u64 + 1).collect()
    };
    let scales_record = WeightRecord {
        transform: kernel_core::weight_transform::WeightTransform::CopyScales,
        sources: vec![source_tensor(&scales)],
        axis: Some(0),
        ranges: scale_ranges.into_iter().collect(),
        stored_shape: vec![scale_rows.len() as u64, scales.shape[1]],
    };
    let mut records = Vec::new();
    for (tensor, section, record) in [
        (&codes, PackedSection::Codes, codes_record),
        (&scales, PackedSection::LocalScales, scales_record),
    ] {
        records.push(VariantRecord {
            view: tensor.name.clone(),
            view_offset: 0,
            section,
            record,
            tensors: vec![tensor.name.clone()],
        });
    }
    // Source offsets belong to the checkpoint spans; records read the tensor
    // relative to its authenticated start and never bake in file offsets.
    require(
        dtype_bits(source.codes.dtype) == 8,
        "QKV codes must be byte-addressable FP8",
    )?;
    Ok(QkvPreparation {
        weight,
        source_rows: rows,
        records,
    })
}

/// Native controls retained beside logical stage ports.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum StageBehavior {
    Routing {
        fused: bool,
        scale_bits: u32,
    },
    Combine {
        parameters: serde_json::Value,
    },
    Mask {
        source: String,
        view: String,
        width: u32,
        controls: Vec<&'static str>,
    },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StageDescriptor {
    pub stage: projection_spec::derive::DerivedStage,
    pub behavior: StageBehavior,
}

pub fn stage_descriptor(
    stage: &projection_spec::contracts::Stage,
    tp: Tp,
    usage: kernel_core::precision::OperationUse,
) -> Result<StageDescriptor, Error> {
    use projection_spec::derive::{DerivedStage, RequiredView, ViewRole};
    require(
        stage.phases.contains(&usage.phase),
        "stage does not cover the requested phase",
    )?;
    require(
        stage.mutates.is_empty(),
        "MiMo stages do not mutate persistent state",
    )?;
    let geometry = super::geometry_for(tp);
    let behavior = match stage.contract.id.as_str() {
        "mimo26/legacy-routing.v1" => {
            super::routing(&stage.contract, &geometry)?;
            StageBehavior::Routing {
                fused: Selected::fused_router(usage),
                scale_bits: 1.0_f32.to_bits(),
            }
        }
        "mimo26/legacy-combine.v1" => {
            let parameters = super::combine(&stage.contract, &geometry)?;
            StageBehavior::Combine { parameters }
        }
        "mimo26/dflash-mask.v1" => {
            super::mask(&stage.contract, &geometry)?;
            let parameters: super::MaskParameters = super::typed(&stage.contract)?;
            require(
                usage.phase == kernel_core::precision::ExecutionPhase::Verify,
                "DFlash mask is a verify stage",
            )?;
            StageBehavior::Mask {
                source: parameters.source,
                view: "mask_embedding.weight".into(),
                width: parameters.width,
                controls: vec!["positions", "prefix_len", "token_batch", "live_tokens"],
            }
        }
        _ => {
            return Err(error(format!(
                "unregistered MiMo stage `{}`",
                stage.contract.id
            )));
        }
    };
    let views = |ports: &[projection_spec::contracts::Port], role| {
        ports
            .iter()
            .map(|port| {
                let shape = port
                    .shape
                    .iter()
                    .map(|d| d.resolve(&geometry))
                    .collect::<Result<Vec<_>, _>>()?;
                let bytes = shape.iter().try_fold(dtype_bits(port.dtype) / 8, |n, v| {
                    n.checked_mul(*v)
                        .ok_or_else(|| error("stage port byte extent overflows"))
                })?;
                Ok(RequiredView {
                    id: port.binding.clone(),
                    role: if port.state { ViewRole::State } else { role },
                    dtype: port.dtype,
                    interpretation: port.interpretation,
                    shape,
                    bytes,
                    layout: "row-major".into(),
                })
            })
            .collect::<Result<Vec<_>, Error>>()
    };
    let mut descriptor = StageDescriptor {
        stage: DerivedStage {
            id: stage.id.clone(),
            numerical_contract: stage.contract.numerical_contract.clone(),
            inputs: views(&stage.contract.inputs, ViewRole::Input)?,
            outputs: views(&stage.contract.outputs, ViewRole::Output)?,
            mutations: stage.mutates.clone(),
            adapter: None,
        },
        behavior,
    };
    use projection_spec::derive::{AdapterValue as Value, StageAdapter, StageExecution};
    let (execution, facts) = match &descriptor.behavior {
        StageBehavior::Routing { fused, scale_bits } => {
            let kernels = Selected::route_kernels(tp, usage, *fused);
            let symbol = kernels.symbol;
            (
                if *fused {
                    StageExecution::Fused {
                        operation: symbol.clone(),
                    }
                } else {
                    StageExecution::Standalone {
                        symbol: symbol.clone(),
                    }
                },
                BTreeMap::from([
                    ("symbol".into(), Value::Text(symbol)),
                    ("scale".into(), Value::F32Bits(*scale_bits)),
                    ("experts".into(), Value::U64(tp.geometry.experts().into())),
                    ("top-k".into(), Value::U64(tp.geometry.top_k().into())),
                    ("capacity".into(), Value::U64(usage.capacity.into())),
                    ("score".into(), Value::Text("sigmoid-correction".into())),
                    ("ties".into(), Value::Text("lower-index".into())),
                    (
                        "route-order".into(),
                        Value::Text("ascending-source-route".into()),
                    ),
                    (
                        "permutation".into(),
                        Value::List(kernels.permutation.into_iter().map(Value::Text).collect()),
                    ),
                    (
                        "controls".into(),
                        Value::List(
                            [
                                ("logits", kernel_core::contract::DType::F32),
                                ("counts", kernel_core::contract::DType::I32),
                                ("offsets", kernel_core::contract::DType::I32),
                                ("sorted-rows", kernel_core::contract::DType::I32),
                                ("padded-rows", kernel_core::contract::DType::I32),
                                ("active-experts", kernel_core::contract::DType::I32),
                                ("live_tokens", kernel_core::contract::DType::I32),
                            ]
                            .map(|(view, dtype)| {
                                Value::Binding(projection_spec::derive::ControlBinding {
                                    view: view.into(),
                                    dtype,
                                    interpretation: (dtype == kernel_core::contract::DType::I32)
                                        .then_some(
                                            projection_spec::contracts::Interpretation::Unsigned,
                                        ),
                                    offset_bytes: 0,
                                    stride_bytes: None,
                                })
                            })
                            .into(),
                        ),
                    ),
                ]),
            )
        }
        StageBehavior::Combine { parameters } => {
            let folded = usage.phase != kernel_core::precision::ExecutionPhase::Prefill
                && usage.capacity <= crate::serving::DECODE_ENTRY_ROWS;
            (
                if folded {
                    StageExecution::Fused {
                        operation: "mlp-reduce-norm".into(),
                    }
                } else {
                    StageExecution::Standalone {
                        symbol: Selected::combine_symbol(tp),
                    }
                },
                parameters
                    .as_object()
                    .ok_or_else(|| error("combine parameters must be an object"))?
                    .iter()
                    .map(|(key, value)| {
                        Ok((
                            key.clone(),
                            Value::Text(
                                value
                                    .as_str()
                                    .ok_or_else(|| error("combine identity must be a string"))?
                                    .into(),
                            ),
                        ))
                    })
                    .collect::<Result<BTreeMap<_, _>, Error>>()?,
            )
        }
        StageBehavior::Mask { .. } => {
            let parameters: super::MaskParameters = super::typed(&stage.contract)?;
            let adapter = parameters.native_adapter(tp);
            (adapter.execution, adapter.facts)
        }
    };
    descriptor.stage.adapter = Some(StageAdapter {
        contract: stage.contract.id.clone(),
        execution,
        facts,
    });
    Ok(descriptor)
}

/// The registered behavior of one listed geometry's declaration.
#[derive(Debug)]
pub struct Variants(pub &'static crate::geometry::Geometry);

impl Variants {
    pub const FLASH: Self = Self(&crate::geometry::V26_FLASH);

    fn tp(&self, context: &projection_spec::derive::ProjectionContext<'_>) -> Result<Tp, Error> {
        Tp::of_geometry(self.0, context.topology.ranks).map_err(error)
    }
}

impl projection_spec::derive::hooks::VariantDeriver for Variants {
    fn transform(
        &self,
        variant: &projection_spec::Variant,
        part: Option<&projection_spec::Part>,
        context: &projection_spec::derive::ProjectionContext<'_>,
        mut imported: projection_spec::derive::ImportedProjection,
    ) -> Result<projection_spec::derive::SourceTransformResult, Error> {
        use projection_spec::derive::{SourcePreparation, SourceTransformResult};
        require(
            variant.id == "mimo26/qkv-scale-regroup.v1" && part.is_none(),
            "unregistered QKV source transform",
        )?;
        super::regroup(variant, context.geometry)?;
        let parameters: RegroupParameters = super::typed(variant)?;
        let tp = self.tp(context)?;
        require(
            context.topology.rank < tp.ranks(),
            "QKV rank is outside topology",
        )?;
        if parameters.bypass_ranks.contains(&tp.ranks()) {
            return Ok(SourceTransformResult::Bypass);
        }
        let original = context
            .bound
            .operands
            .iter()
            .find(|b| {
                b.projection == context.source_instance.projection
                    && b.layer == context.source_layer
            })
            .ok_or_else(|| error("QKV transform has no authenticated source operand"))?;
        require(
            original.imported.parts.len() == parameters.chunks as usize,
            "QKV source must retain every independently scaled chunk",
        )?;
        let mut bank = original.imported.weight.clone();
        let k = *bank
            .logical_shape
            .last()
            .ok_or_else(|| error("QKV source has no K axis"))?;
        let chunk_rows = parameters.chunk_rows(tp)?;
        bank.logical_shape = vec![u64::from(parameters.chunks), chunk_rows, k];
        bank.codes.shape = bank.logical_shape.clone();
        let local = bank
            .local_scale
            .as_mut()
            .ok_or_else(|| error("QKV source has no local scale"))?;
        local.tensor.shape = vec![
            u64::from(parameters.chunks),
            chunk_rows.div_ceil(128),
            k / 128,
        ];
        for axis in &mut local.groups {
            axis.axis += 1;
        }
        let prepared = qkv(&bank, &parameters, tp, context.topology.rank as usize)?;
        let geometry = kernel_core::precision::MatrixGeometry {
            rows: prepared.weight.logical_shape[0],
            k,
        };
        let records = prepared.records;
        let output = context
            .usage
            .contract
            .as_ref()
            .map_or(imported.arithmetic.output, |c| c.output_dtype);
        imported.operation_id = context.operation_id(&imported.operation_id, output)?;
        imported.weights = vec![prepared.weight];
        imported.format = kernel_core::precision::format::FormatId::Fp8Rows64;
        imported.preparation = Some(SourcePreparation {
            topology: context.topology,
            geometry,
            records,
        });
        Ok(SourceTransformResult::Applied(imported))
    }

    fn implementation(
        &self,
        variant: &projection_spec::Variant,
        context: &projection_spec::derive::ProjectionContext<'_>,
        imported: &projection_spec::derive::ImportedProjection,
    ) -> Result<projection_spec::derive::DerivedOperation, Error> {
        match variant.id.as_str() {
            "mimo26/legacy-fused-dense.v1" => {
                super::dense(variant, context.geometry)?;
            }
            _ => {
                return Err(error(format!(
                    "unregistered MiMo implementation `{}`",
                    variant.id
                )));
            }
        }
        let mut derived = context.ordinary(imported)?;
        require(
            derived.operation.arithmetic.numerical_contract == variant.numerical_contract,
            "MiMo implementation changes numerical identity",
        )?;
        legacy::pack(&mut derived)?;
        derived
            .stages
            .push(legacy::stage(variant, &derived, self.tp(context)?)?);
        Ok(derived)
    }

    fn stage(
        &self,
        stage: &projection_spec::contracts::Stage,
        context: &projection_spec::derive::ProjectionContext<'_>,
    ) -> Result<projection_spec::derive::DerivedStage, Error> {
        let tp = self.tp(context)?;
        require(
            context.topology.rank < tp.ranks(),
            "stage rank is outside the topology",
        )?;
        require(
            context.geometry.get("tp") == Some(&u64::from(tp.ranks())),
            "MiMo context topology differs from its geometry",
        )?;
        let capacity =
            u32::try_from(context.usage.capacity.resolve(context.geometry)?).map_err(error)?;
        Ok(stage_descriptor(
            stage,
            tp,
            kernel_core::precision::OperationUse {
                phase: context.usage.phase,
                capacity,
                graph: context.usage.graph,
            },
        )?
        .stage)
    }
}

mod legacy;

#[cfg(test)]
mod tests;
