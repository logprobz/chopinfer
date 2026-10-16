//! Pure descriptors for Qwen's registered preparation behavior.

use super::{HashAddressing, HashParameters, require};
use crate::backend::{Kernels, Selected};
use crate::contract::{Ranks, Shape};
use kernel_core::contract::WeightFormat;
use kernel_core::precision::{ExecutionPhase, OperationUse, dtype_bits};
use projection_spec::contracts::Stage;
use projection_spec::derive::hooks::VariantDeriver;
use projection_spec::derive::{DerivedStage, ProjectionContext, RequiredView, ViewRole};
use projection_spec::{Error, Geometry};

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum HashExecution {
    FusedLookup(String),
    Standalone(String),
}

/// Addressing facts used by the current hash launch, separate from logical ports.
#[derive(Debug, PartialEq)]
pub struct HashDescriptor {
    pub stage: DerivedStage,
    pub execution: HashExecution,
    pub addressing: HashAddressing,
    pub history_offset_bytes: u64,
    pub history_item_stride_bytes: u64,
    pub table_start: u64,
    pub table_rows: u64,
    pub local_heads: u32,
    pub ids_row_stride_bytes: u64,
    pub verify_rows: Option<u32>,
    pub controls: Vec<&'static str>,
}

fn error(e: impl std::fmt::Display) -> Error {
    Error(e.to_string())
}

fn ports(stage: &Stage, geometry: &Geometry) -> Result<DerivedStage, Error> {
    let views = |ports: &[projection_spec::contracts::Port], role| {
        ports
            .iter()
            .map(|port| {
                let shape = port
                    .shape
                    .iter()
                    .map(|d| d.resolve(geometry))
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
    Ok(DerivedStage {
        id: stage.id.clone(),
        numerical_contract: stage.contract.numerical_contract.clone(),
        inputs: views(&stage.contract.inputs, ViewRole::Input)?,
        outputs: views(&stage.contract.outputs, ViewRole::Output)?,
        mutations: stage.mutates.clone(),
        adapter: None,
    })
}

pub fn hash_descriptor(
    stage: &Stage,
    shape: Shape,
    rank: u32,
    usage: OperationUse,
    weight: WeightFormat,
) -> Result<HashDescriptor, Error> {
    require(
        stage.contract.id == "qwen38/ngram-hash.v1",
        "unregistered Qwen hash stage",
    )?;
    require(
        stage.phases.contains(&usage.phase),
        "hash stage does not cover the requested phase",
    )?;
    require(
        stage.mutates == ["history"],
        "hash stage must retain its history mutation",
    )?;
    let geometry = super::geometry(shape.ranks())?;
    super::hash(&stage.contract, &geometry)?;
    let parameters: HashParameters = super::typed(&stage.contract)?;
    let (table_start, table_rows) = crate::views::ngram_rows(rank, shape).map_err(error)?;
    let (addressing, controls) = match usage.phase {
        ExecutionPhase::Prefill => (HashAddressing::PrefillContiguous, vec!["item_rows"]),
        ExecutionPhase::Decode => (HashAddressing::DecodeItemStride, vec!["q_len"]),
        ExecutionPhase::Verify => (
            HashAddressing::VerifyCuRows,
            vec!["cu_rows", "q_len", "live_items"],
        ),
    };
    require(
        parameters.phase_addressing.contains(&addressing),
        "hash phase adapter is absent",
    )?;
    let execution = match weight {
        WeightFormat::Fp8PerTensor => HashExecution::FusedLookup(
            crate::precision::legacy_lookup_capability(shape, true, usage)
                .map_err(error)?
                .stages[0]
                .clone(),
        ),
        WeightFormat::Nvfp4 => {
            HashExecution::Standalone(Selected::ngram_hash_symbol(shape, usage.phase))
        }
        _ => {
            return Err(error(
                "hash lookup has no implementation for this weight format",
            ));
        }
    };
    let mut descriptor = HashDescriptor {
        stage: ports(stage, &geometry)?,
        execution,
        addressing,
        history_offset_bytes: shape.ple_conv_bytes(),
        history_item_stride_bytes: shape.ple_entry_bytes(),
        table_start,
        table_rows,
        local_heads: shape.ngram_local_heads,
        ids_row_stride_bytes: u64::from(shape.ngram_local_heads) * 8,
        verify_rows: (usage.phase == ExecutionPhase::Verify)
            .then_some(crate::contract::MAX_VERIFY_ROWS),
        controls,
    };
    use kernel_core::contract::DType;
    use projection_spec::contracts::Interpretation;
    use projection_spec::derive::{
        AdapterValue as Value, ControlBinding, StageAdapter, StageExecution,
    };
    let mut facts = std::collections::BTreeMap::from([
        (
            "addressing".into(),
            Value::Text(
                match usage.phase {
                    ExecutionPhase::Prefill => "prefill-contiguous",
                    ExecutionPhase::Decode => "decode-item-stride",
                    ExecutionPhase::Verify => "verify-cu-rows",
                }
                .into(),
            ),
        ),
        ("rank".into(), Value::U64(rank.into())),
        ("table-start".into(), Value::U64(table_start)),
        ("table-rows".into(), Value::U64(table_rows)),
        (
            "local-heads".into(),
            Value::U64(shape.ngram_local_heads.into()),
        ),
        ("orders".into(), Value::U64(parameters.orders.into())),
        (
            "heads-per-order".into(),
            Value::U64(parameters.heads_per_order.into()),
        ),
        (
            "history-tokens".into(),
            Value::U64(parameters.history_tokens.into()),
        ),
        ("eos".into(), Value::U64(parameters.eos.into())),
        (
            "integer-arithmetic".into(),
            Value::Text("wrapping-u64-xor".into()),
        ),
    ]);
    let binding = |view: &str, dtype, offset_bytes, stride_bytes| {
        Value::Binding(ControlBinding {
            view: view.into(),
            dtype,
            interpretation: Some(Interpretation::Unsigned),
            offset_bytes,
            stride_bytes,
        })
    };
    facts.insert("tokens".into(), binding("tokens", DType::I32, 0, Some(4)));
    for (key, view) in [
        ("history-in", "history-input"),
        ("history-out", "history-output"),
    ] {
        facts.insert(
            key.into(),
            binding(
                view,
                DType::I32,
                shape.ple_conv_bytes(),
                Some(shape.ple_entry_bytes()),
            ),
        );
    }
    facts.insert(
        "ids".into(),
        binding("ids", DType::I64, 0, Some(descriptor.ids_row_stride_bytes)),
    );
    for view in ["multipliers", "sizes", "offsets"] {
        facts.insert(view.into(), binding(view, DType::I64, 0, Some(8)));
    }
    facts.insert("q-len".into(), binding("q_len", DType::I32, 0, Some(4)));
    if usage.phase == ExecutionPhase::Verify {
        facts.insert("cu-rows".into(), binding("cu_rows", DType::I32, 0, Some(4)));
        facts.insert(
            "live-items".into(),
            binding("live_batch", DType::I32, 0, None),
        );
        facts.insert(
            "verify-rows".into(),
            Value::U64(crate::contract::MAX_VERIFY_ROWS.into()),
        );
    }
    descriptor.stage.adapter = Some(StageAdapter {
        contract: stage.contract.id.clone(),
        execution: match &descriptor.execution {
            HashExecution::FusedLookup(operation) => StageExecution::Fused {
                operation: operation.clone(),
            },
            HashExecution::Standalone(symbol) => StageExecution::Standalone {
                symbol: symbol.clone(),
            },
        },
        facts,
    });
    Ok(descriptor)
}

#[derive(Debug, Default)]
pub struct Variants;

impl VariantDeriver for Variants {
    fn fusion(
        &self,
        variant: &projection_spec::Variant,
        context: &projection_spec::derive::FusionContext<'_>,
        members: &[projection_spec::derive::DerivedOperation],
    ) -> Result<projection_spec::derive::DerivedFusion, Error> {
        use projection_spec::derive::{
            AdapterValue as Value, DerivedFusion, DerivedStage, GeometryMode, ImportedProjection,
            StageAdapter, StageExecution,
        };
        require(
            variant.id == "qwen38/legacy-input-fusion.v1",
            "unregistered Qwen input fusion",
        )?;
        super::input_fusion(variant, context.projection.geometry)?;
        let parameters: super::InputFusionParameters = super::typed(variant)?;
        require(
            members
                .iter()
                .map(|m| m.instance.projection.clone())
                .collect::<Vec<_>>()
                == parameters.decode_group
                && parameters.decode_group == context.group.members,
            "legacy fusion membership differs from its registered schedule",
        )?;
        let requested = members
            .first()
            .ok_or_else(|| error("legacy fusion has no members"))?
            .requested;
        require(
            members.iter().all(|m| {
                m.requested == requested
                    && m.operation
                        .weights
                        .iter()
                        .all(|w| w.format == WeightFormat::Bf16)
            }),
            "legacy input fusion requires matching BF16 uses",
        )?;
        let shape = Shape::of(
            Ranks::new(context.projection.topology.ranks)
                .ok_or_else(|| error("unserved Qwen rank count"))?,
        )
        .ok_or_else(|| error("unserved Qwen geometry"))?;
        require(
            context.projection.topology.rank < context.projection.topology.ranks
                && context.projection.geometry.get("tp")
                    == Some(&u64::from(context.projection.topology.ranks)),
            "fusion topology differs from its geometry",
        )?;
        let capacity = crate::precision::projection_capacity(requested).map_err(error)?;
        let groups = if capacity <= crate::contract::DECODE_ENTRY_ROWS {
            vec![parameters.decode_group.clone()]
        } else {
            parameters.prefill_groups
        };
        let mut operations = members.to_vec();
        for operation in &mut operations {
            require(
                operation.operation.arithmetic.numerical_contract == variant.numerical_contract,
                "fusion member changes its numerical identity",
            )?;
            let imported = ImportedProjection {
                operation_id: operation.operation.id.clone(),
                target: operation.operation.target,
                family: operation.family,
                format: kernel_core::precision::format::FormatId::Bf16,
                geometry: GeometryMode::RankLocal,
                weights: operation.operation.weights.clone(),
                activations: operation.operation.activations.clone(),
                arithmetic: operation.operation.arithmetic.clone(),
                a16_recipe: operation.a16_recipe,
                matrices: operation.slices.iter().map(|s| s.source.clone()).collect(),
                preparation: operation.source_preparation.clone(),
            };
            legacy::pack(
                operation,
                &imported,
                shape,
                context.projection.topology.rank,
                context.projection.bound,
            )?;
            operation.operation.geometry = Some(operation.logical_geometry);
            for view in operation
                .views
                .iter_mut()
                .filter(|v| v.role == ViewRole::Output)
            {
                *view
                    .shape
                    .last_mut()
                    .ok_or_else(|| error("legacy output view has no columns"))? =
                    operation.logical_geometry.rows;
                view.bytes = view.shape.iter().product::<u64>()
                    * kernel_core::precision::dtype_bits(view.dtype)
                    / 8;
            }
        }
        let mut stages = Vec::new();
        let mut output_strides = std::collections::BTreeMap::<String, u64>::new();
        let mut outputs = Vec::new();
        for output in &context.group.outputs {
            let member = operations
                .iter()
                .find(|m| m.instance.projection == output.member)
                .ok_or_else(|| error("fusion output has no member"))?;
            let columns = member.logical_geometry.rows;
            let offset = output_strides.entry(output.buffer.clone()).or_default();
            outputs.push((
                output.member.clone(),
                output.buffer.clone(),
                *offset,
                columns,
            ));
            *offset = offset
                .checked_add(columns)
                .ok_or_else(|| error("fusion output stride overflows"))?;
        }
        let output_views: Vec<_> = output_strides
            .iter()
            .map(|(id, stride)| {
                Ok(RequiredView {
                    id: id.clone(),
                    role: ViewRole::Output,
                    dtype: kernel_core::contract::DType::Bf16,
                    interpretation: None,
                    shape: vec![u64::from(capacity), *stride],
                    bytes: u64::from(capacity)
                        .checked_mul(*stride)
                        .and_then(|n| n.checked_mul(2))
                        .ok_or_else(|| error("fusion output bytes overflow"))?,
                    layout: "row-major".into(),
                })
            })
            .collect::<Result<_, Error>>()?;
        // Ports describe full source rows. Native buffers store local heads,
        // including replicated K/V heads when their count is below TP.
        for (index, group) in groups.iter().enumerate() {
            let entry = fusion_entry(&group[0], shape, requested)?;
            let (symbol, geometry) = (
                entry.symbol,
                (entry.segments, entry.k, entry.native_capacity),
            );
            let facts = std::collections::BTreeMap::from([
                (
                    "members".into(),
                    Value::List(group.iter().map(|m| Value::Text(m.clone())).collect()),
                ),
                ("symbol".into(), Value::Text(symbol.clone())),
                (
                    "segments".into(),
                    Value::List(geometry.0.iter().map(|n| Value::U64((*n).into())).collect()),
                ),
                ("k".into(), Value::U64(geometry.1.into())),
                ("native-capacity".into(), Value::U64(geometry.2.into())),
                ("compiled-capacity".into(), Value::U64(capacity.into())),
                (
                    "outputs".into(),
                    Value::List(
                        outputs
                            .iter()
                            .filter(|(member, ..)| group.contains(member))
                            .map(|o| {
                                Value::List(vec![
                                    Value::Text(o.0.clone()),
                                    Value::Text(o.1.clone()),
                                    Value::U64(o.2),
                                    Value::U64(output_strides[&o.1]),
                                    Value::U64(o.3),
                                ])
                            })
                            .collect(),
                    ),
                ),
            ]);
            stages.push(DerivedStage {
                id: format!("{}.legacy.{index}", context.group.id),
                numerical_contract: variant.numerical_contract.clone(),
                inputs: operations
                    .iter()
                    .filter(|m| group.contains(&m.instance.projection))
                    .flat_map(|m| m.views.iter().filter(|v| v.role == ViewRole::Input))
                    .cloned()
                    .collect(),
                outputs: output_views
                    .iter()
                    .filter(|v| {
                        outputs
                            .iter()
                            .any(|(member, buffer, ..)| group.contains(member) && buffer == &v.id)
                    })
                    .cloned()
                    .collect(),
                mutations: vec![],
                adapter: Some(StageAdapter {
                    contract: variant.id.clone(),
                    execution: StageExecution::Fused { operation: symbol },
                    facts,
                }),
            });
        }
        let mut views: Vec<_> = operations.iter().flat_map(|m| m.views.clone()).collect();
        views.extend(output_views);
        Ok(DerivedFusion {
            id: context.group.id.clone(),
            layer: context.projection.layer,
            requested,
            members: members.iter().map(|m| m.instance.clone()).collect(),
            fused: true,
            operations,
            views,
            stages,
        })
    }

    fn implementation(
        &self,
        variant: &projection_spec::Variant,
        context: &ProjectionContext<'_>,
        imported: &projection_spec::derive::ImportedProjection,
    ) -> Result<projection_spec::derive::DerivedOperation, Error> {
        match variant.id.as_str() {
            "qwen38/legacy-fused-dense.v1" => {
                super::dense(variant, context.geometry)?;
            }
            "qwen38/legacy-fragment-experts.v1" => {
                super::experts(variant, context.geometry)?;
            }
            _ => {
                return Err(error(format!(
                    "unregistered Qwen implementation `{}`",
                    variant.id
                )));
            }
        }
        let shape = Shape::of(
            Ranks::new(context.topology.ranks).ok_or_else(|| error("unserved Qwen rank count"))?,
        )
        .ok_or_else(|| error("unserved Qwen geometry"))?;
        let mut derived = context.ordinary(imported)?;
        require(
            derived.operation.arithmetic.numerical_contract == variant.numerical_contract,
            "Qwen implementation changes numerical identity",
        )?;
        if variant.id == "qwen38/legacy-fragment-experts.v1" {
            let projection = if derived.operation.epilogues.swiglu.is_some() {
                crate::precision::Projection::GateUp
            } else {
                crate::precision::Projection::Down
            };
            let native =
                crate::precision::legacy_expert_capability(shape, projection, derived.requested)
                    .map_err(error)?;
            require(
                native.geometry == derived.operation.matrix_geometry().map_err(error)?
                    && native.arithmetic == derived.operation.arithmetic,
                "fragment expert recipe differs from its current compiled capability",
            )?;
        }
        legacy::pack(
            &mut derived,
            imported,
            shape,
            context.topology.rank,
            context.bound,
        )?;
        derived
            .stages
            .push(legacy::stage(variant, &derived, shape)?);
        Ok(derived)
    }

    fn stage(&self, stage: &Stage, context: &ProjectionContext<'_>) -> Result<DerivedStage, Error> {
        let rank_count =
            Ranks::new(context.topology.ranks).ok_or_else(|| error("unserved Qwen rank count"))?;
        let shape =
            Shape::of(rank_count).ok_or_else(|| error("unserved Qwen projection geometry"))?;
        require(
            context.geometry.get("tp") == Some(&u64::from(context.topology.ranks)),
            "Qwen context topology differs from its geometry",
        )?;
        let weights: Vec<_> = context
            .bound
            .operands
            .iter()
            .filter(|operand| {
                operand.projection == context.source_instance.projection
                    && operand.layer == context.source_layer
            })
            .map(|operand| operand.imported.weight.format)
            .collect();
        let weight = *weights
            .first()
            .ok_or_else(|| error("hash stage has no bound lookup operand"))?;
        require(
            weights.iter().all(|other| *other == weight),
            "hash stage has mixed lookup formats",
        )?;
        let capacity =
            u32::try_from(context.usage.capacity.resolve(context.geometry)?).map_err(error)?;
        Ok(hash_descriptor(
            stage,
            shape,
            context.topology.rank,
            OperationUse {
                phase: context.usage.phase,
                capacity,
                graph: context.usage.graph,
            },
            weight,
        )?
        .stage)
    }
}

/// The compiled entry of the registered fusion group led by `member`:
/// its dense role's fused entry, or the hyper-connection input entry, as
/// the backend chooses it.
fn fusion_entry(
    member: &str,
    shape: Shape,
    usage: OperationUse,
) -> Result<crate::backend::FusionChoice, Error> {
    use crate::backend::{FusionMember, Kernels, Selected};
    use crate::precision::DenseRole;
    let role = match crate::projections::ports::get(shape)?.published(member) {
        "linear.qkv" => Some(DenseRole::LinearQkv),
        "linear.z" => Some(DenseRole::LinearZ),
        "linear.a" | "linear.b" => Some(DenseRole::LinearA),
        "full.q" => Some(DenseRole::AttentionQ),
        "full.k" => Some(DenseRole::AttentionK),
        "full.v" => Some(DenseRole::AttentionV),
        "full.index" => Some(DenseRole::IndexQk),
        _ => None,
    };
    let member = match role {
        Some(role) => FusionMember::Dense(role),
        None => FusionMember::HyperConnection(member),
    };
    Selected::legacy_fusion(shape, member, usage).map_err(error)
}

mod legacy;

#[cfg(test)]
mod tests;
