//! Reuse the rank-image planner for the legacy fragment views and records.
use super::error;
use crate::backend::{Kernels, LegacyStage, Selected};
use crate::catalog::{Catalog, TensorRecord, classify_for};
use crate::contract::Shape;
use crate::placement::ImagePlan;
use kernel_core::contract::DType;
use kernel_core::precision::TensorSpan;
use kernel_core::weight_transform::PackedSection;
use projection_spec::Error;
use projection_spec::derive::{
    DerivedOperation, ImportedProjection, RequiredView, VariantRecord, ViewRole,
};
use std::collections::BTreeMap;

pub(super) fn stage(
    variant: &projection_spec::Variant,
    derived: &DerivedOperation,
    shape: Shape,
) -> Result<projection_spec::derive::DerivedStage, Error> {
    use projection_spec::derive::{
        AdapterValue as Value, DerivedStage, StageAdapter, StageExecution,
    };
    // A use without a projection capacity is refused before any fact.
    crate::precision::projection_capacity(derived.requested).map_err(error)?;
    let mut facts = BTreeMap::from([
        (
            "compiled-capacity".into(),
            Value::U64(derived.compiled_capacity.into()),
        ),
        (
            "weights".into(),
            Value::List(
                derived
                    .views
                    .iter()
                    .filter(|v| matches!(v.role, ViewRole::Weight(_)))
                    .map(|v| Value::Text(v.id.clone()))
                    .collect(),
            ),
        ),
    ]);
    let symbol = if variant.id.ends_with("legacy-fragment-experts.v1") {
        let projection = if derived.operation.epilogues.swiglu.is_some() {
            crate::precision::Projection::GateUp
        } else {
            crate::precision::Projection::Down
        };
        backend_stage(&mut facts, shape, LegacyStage::Experts(projection), derived)?
    } else {
        let p: super::super::DenseParameters = super::super::typed(variant)?;
        if let Some(gate) = p.input_gate {
            facts.insert("input-gate".into(), Value::Text(gate.projection));
            facts.insert(
                "gate-order".into(),
                Value::Text("norm-then-sigmoid-gate".into()),
            );
            backend_stage(&mut facts, shape, LegacyStage::GatedOutput, derived)?
        } else if let Some(mix) = p.stream_mix {
            facts.insert(
                "streams".into(),
                Value::U64(mix.streams.resolve(&super::super::geometry_for(shape)?)?),
            );
            facts.insert(
                "mean-divisor".into(),
                Value::U64(
                    mix.mean_divisor
                        .resolve(&super::super::geometry_for(shape)?)?,
                ),
            );
            facts.insert("activation".into(), Value::Text("sigmoid".into()));
            for (key, view, stride) in [
                ("mix-input", "hc-down", u64::from(shape.hc_rank()) * 2),
                (
                    "precomputed-input",
                    "hc-bottleneck",
                    u64::from(shape.hc_rank()) * 2,
                ),
                ("normalized", "normalized", u64::from(shape.wide()) * 2),
                (
                    "mixed-output",
                    "mixed-output",
                    u64::from(shape.hidden()) * 2,
                ),
            ] {
                facts.insert(
                    key.into(),
                    Value::Binding(projection_spec::derive::ControlBinding {
                        view: view.into(),
                        dtype: DType::Bf16,
                        interpretation: None,
                        offset_bytes: 0,
                        stride_bytes: Some(stride),
                    }),
                );
            }
            facts.insert(
                "live-rows".into(),
                Value::Binding(projection_spec::derive::ControlBinding {
                    view: "live_tokens".into(),
                    dtype: DType::I32,
                    interpretation: Some(projection_spec::contracts::Interpretation::Unsigned),
                    offset_bytes: 0,
                    stride_bytes: None,
                }),
            );
            backend_stage(&mut facts, shape, LegacyStage::StreamMix, derived)?
        } else {
            return Err(error(
                "legacy dense adapter has no registered gate or stream composition",
            ));
        }
    };
    facts.insert("symbol".into(), Value::Text(symbol.clone()));
    Ok(DerivedStage {
        id: format!("{}.legacy", derived.operation.id),
        numerical_contract: variant.numerical_contract.clone(),
        inputs: derived
            .views
            .iter()
            .filter(|v| v.role == ViewRole::Input)
            .cloned()
            .collect(),
        outputs: derived
            .views
            .iter()
            .filter(|v| v.role == ViewRole::Output)
            .cloned()
            .collect(),
        mutations: vec![],
        adapter: Some(StageAdapter {
            contract: variant.id.clone(),
            execution: StageExecution::Fused { operation: symbol },
            facts,
        }),
    })
}

/// The entry the backend runs a legacy adapter `stage` with, its facts
/// added to `facts`.
fn backend_stage(
    facts: &mut BTreeMap<String, projection_spec::derive::AdapterValue>,
    shape: Shape,
    stage: LegacyStage,
    derived: &DerivedOperation,
) -> Result<String, Error> {
    let choice = Selected::legacy_stage(shape, stage, derived.requested).map_err(error)?;
    facts.extend(choice.facts);
    Ok(choice.symbol)
}

fn dtype(dtype: DType) -> Result<runtime_weights::DType, Error> {
    Ok(match dtype {
        DType::Bf16 => runtime_weights::DType::Bf16,
        DType::F32 => runtime_weights::DType::F32,
        DType::Fp8E4M3 => runtime_weights::DType::F8E4M3,
        _ => return Err(error("legacy fragments require BF16, FP8 or F32 sources")),
    })
}

pub(super) fn catalog(
    imported: &ImportedProjection,
    shape: Shape,
    bound: &projection_spec::bind::BoundDeclaration,
) -> Result<Catalog, Error> {
    let mut tensors: BTreeMap<String, TensorRecord> = BTreeMap::new();
    for matrix in &imported.matrices {
        let spans: Vec<&TensorSpan> = std::iter::once(&matrix.weight.codes)
            .chain(matrix.weight.local_scale.iter().map(|s| &s.tensor))
            .collect();
        for part in spans {
            let aggregate = imported
                .weights
                .iter()
                .flat_map(|w| {
                    std::iter::once(&w.codes).chain(w.local_scale.iter().map(|s| &s.tensor))
                })
                .find(|full| full.tensor == part.tensor && full.offset == 0);
            let span = aggregate.unwrap_or_else(|| {
                if part.offset == 0 {
                    part
                } else {
                    bound
                        .operands
                        .iter()
                        .flat_map(|b| {
                            std::iter::once(&b.imported.weight.codes)
                                .chain(b.imported.weight.local_scale.iter().map(|s| &s.tensor))
                        })
                        .find(|full| full.tensor == part.tensor)
                        .unwrap_or(part)
                }
            });
            if span.offset != 0 {
                return Err(error(
                    "legacy fragment source must be a whole checkpoint tensor",
                ));
            }
            let (placement, scale) =
                classify_for(shape.geometry, &span.tensor, shape.ranks() as usize)
                    .map_err(error)?;
            let tensor = TensorRecord {
                name: span.tensor.clone(),
                dtype: dtype(span.dtype)?,
                shape: span.shape.clone(),
                file: String::new(),
                file_start: 0,
                bytes: span.byte_length,
                placement,
                scale,
            };
            if let Some(prior) = tensors.get(&tensor.name) {
                if prior.dtype != tensor.dtype
                    || prior.shape != tensor.shape
                    || prior.bytes != tensor.bytes
                {
                    return Err(error("conflicting legacy source metadata"));
                }
            } else {
                tensors.insert(tensor.name.clone(), tensor);
            }
        }
    }
    Ok(Catalog {
        geometry: shape.geometry,
        tensors,
        ranks: shape.ranks() as usize,
        precision: None,
    })
}

pub(super) fn pack(
    derived: &mut DerivedOperation,
    imported: &ImportedProjection,
    shape: Shape,
    rank: u32,
    bound: &projection_spec::bind::BoundDeclaration,
) -> Result<(), Error> {
    let catalog = catalog(imported, shape, bound)?;
    let plan = ImagePlan::new(&catalog, None).map_err(error)?;
    let index = &plan.ranks[rank as usize];
    derived
        .views
        .retain(|v| !matches!(v.role, ViewRole::Weight(_)));
    derived.variant_records.clear();
    for view in &index.views {
        let section =
            if view.name.ends_with("weight_scale_inv") || view.name.ends_with("weight_scale") {
                PackedSection::LocalScales
            } else {
                PackedSection::Codes
            };
        let stored_dtype = match view.dtype.as_str() {
            "BF16" => DType::Bf16,
            "F32" => DType::F32,
            "F8_E4M3" => DType::Fp8E4M3,
            _ => return Err(error("unsupported legacy view dtype")),
        };
        derived.views.push(RequiredView {
            id: view.name.clone(),
            role: ViewRole::Weight(section),
            dtype: stored_dtype,
            interpretation: None,
            shape: view.shape.clone(),
            bytes: view.bytes,
            layout: view.layout_id.clone(),
        });
    }
    for (name, pieces) in &plan.sources {
        let tensor = &catalog.tensors[name];
        for piece in pieces.iter().filter(|p| p.rank == rank as usize) {
            let view = &index.views[piece.view];
            let section =
                if view.name.ends_with("weight_scale_inv") || view.name.ends_with("weight_scale") {
                    PackedSection::LocalScales
                } else {
                    PackedSection::Codes
                };
            let record = crate::weight_plan::fragment_record(tensor, piece, &view.layout_id)
                .map_err(error)?;
            if record.output_bytes().map_err(error)? != piece.bytes {
                return Err(error("legacy record extent differs from rank placement"));
            }
            derived.variant_records.push(VariantRecord {
                view: view.name.clone(),
                view_offset: piece.offset - view.offset,
                section,
                record,
                tensors: vec![name.clone()],
            });
        }
    }
    derived.packing.layout.id =
        if imported.format == kernel_core::precision::format::FormatId::Bf16 {
            "bf16-fragment"
        } else {
            "fp8-fragment"
        }
        .into();
    if let Some(record) = derived
        .variant_records
        .iter()
        .find(|r| r.section == PackedSection::Codes)
    {
        derived.packing.rows =
            u32::try_from(record.record.stored_shape[record.record.stored_shape.len() - 2])
                .map_err(error)?;
        derived.packing.k = u32::try_from(
            *record
                .record
                .stored_shape
                .last()
                .ok_or_else(|| error("legacy record has no K"))?,
        )
        .map_err(error)?;
    }
    if let Some(view) = derived
        .views
        .iter()
        .find(|v| v.role == ViewRole::Weight(PackedSection::LocalScales))
    {
        derived.packing.local_scale_dtype = Some(view.dtype);
    }
    model_compiler_sdk::quantization::projection::records(derived).map_err(error)?;
    Ok(())
}
