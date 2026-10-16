//! The legacy fused dense entries: their stage and served records.
use super::error;
use crate::Tp;
use crate::backend::{Kernels, Selected};
use kernel_core::contract::DType;
use projection_spec::Error;
use projection_spec::derive::{DerivedOperation, RequiredView, VariantRecord, ViewRole};
use std::collections::BTreeMap;

pub(super) fn stage(
    variant: &projection_spec::Variant,
    derived: &DerivedOperation,
    tp: Tp,
) -> Result<projection_spec::derive::DerivedStage, Error> {
    use projection_spec::derive::{
        AdapterValue as Value, DerivedStage, StageAdapter, StageExecution,
    };
    let p: super::super::DenseParameters = super::super::typed(variant)?;
    let capacity = if derived.requested.phase == kernel_core::precision::ExecutionPhase::Prefill
        || derived.requested.capacity > crate::serving::DECODE_ENTRY_ROWS
    {
        crate::serving::PREFILL_TOKENS
    } else {
        derived.requested.capacity
    };
    let usage = kernel_core::precision::OperationUse {
        capacity,
        ..derived.requested
    };
    let (symbol, entry) = Selected::gate_up_entry(tp, usage, p.operation == "dflash.mlp.gate-up")
        .map_err(|_| error("MiMo legacy dense entry is absent"))?;
    let kernel_core::contract::StaticQuery::Gemm(query) = entry.key.query else {
        return Err(error("MiMo legacy entry has no GEMM query"));
    };
    let geometry = derived.operation.matrix_geometry().map_err(error)?;
    super::require(
        u64::from(query.n) == geometry.rows * 2
            && u64::from(query.k) == geometry.k
            && query.epilogues == derived.operation.epilogues,
        "MiMo legacy dense entry differs from its operation",
    )?;
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
            execution: StageExecution::Fused {
                operation: symbol.clone(),
            },
            facts: BTreeMap::from([
                ("symbol".into(), Value::Text(symbol)),
                ("compiled-capacity".into(), Value::U64(capacity.into())),
                (
                    "scale-order".into(),
                    Value::Text(
                        match derived.operation.arithmetic.decode_order {
                            kernel_core::precision::DecodeOrder::BlockScaleAfterMma => {
                                "block-scale-after-mma"
                            }
                            kernel_core::precision::DecodeOrder::ScaleThenCode => "scale-then-code",
                            _ => {
                                return Err(error(
                                    "legacy gate/up has no native scale-order adapter",
                                ));
                            }
                        }
                        .into(),
                    ),
                ),
                (
                    "operands".into(),
                    Value::List(["gate", "up"].map(|s| Value::Text(s.into())).into()),
                ),
            ]),
        }),
    })
}

/// The served fused records: the shared record policy's gather of each
/// operand, gate then up, in one resident view per section named for both
/// operands.
pub(super) fn pack(derived: &mut DerivedOperation) -> Result<(), Error> {
    use kernel_core::precision::format::registry;
    let format = registry::by_format(derived.packing.format)
        .map_err(error)?
        .id();
    let bound = model_compiler_sdk::quantization::projection::bind_record_policy(
        derived.clone(),
        crate::weight_plan::record_policy(format, derived.family),
    )
    .map_err(error)?;
    derived
        .views
        .retain(|v| !matches!(v.role, ViewRole::Weight(_)));
    derived.variant_records.clear();
    for view in &bound.views {
        let ViewRole::Weight(section) = view.role else {
            continue;
        };
        let records: Vec<_> = bound
            .variant_records
            .iter()
            .filter(|r| r.view == view.id)
            .collect();
        let first = records
            .first()
            .ok_or_else(|| error("fused view has no records"))?;
        let name = crate::weight_plan::resident(&first.tensors[0], true);
        let mut shape = first.record.stored_shape.clone();
        shape[0] *= records.len() as u64;
        derived.views.push(RequiredView {
            id: name.clone(),
            role: ViewRole::Weight(section),
            dtype: match first.record.sources[0].dtype.as_str() {
                "BF16" => DType::Bf16,
                "F32" => DType::F32,
                "F8_E4M3" => DType::Fp8E4M3,
                _ => DType::U8,
            },
            interpretation: None,
            shape,
            bytes: view.bytes,
            layout: view.layout.clone(),
        });
        for record in records {
            derived.variant_records.push(VariantRecord {
                view: name.clone(),
                ..record.clone()
            });
        }
    }
    model_compiler_sdk::quantization::projection::records(derived).map_err(error)?;
    Ok(())
}
