//! Shared schedule defaults and append-only per-instance tuning data.
use super::*;
use kernel_core::contract::{Epilogues, RoutedCapacity};
use kernel_core::precision::ExecutionPhase;

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LookupAddressing {
    pub heads: u32,
    pub ids: row::LookupIds,
    pub scale_dtype: Option<DType>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Key {
    /// A declaration projection ID, supplied only to disambiguate colliding schedules.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub projection: Option<String>,
    pub kind: OperationKind,
    pub format: FormatId,
    pub geometry: MatrixGeometry,
    pub operands: usize,
    pub target: kernel_core::contract::Target,
    pub activation: Option<ActivationFormat>,
    pub weight_operand: DType,
    pub accumulator: DType,
    pub output: DType,
    pub decode_order: DecodeOrder,
    pub rounding: Rounding,
    pub epilogues: Epilogues,
    pub weight_globals: Option<ScaleScope>,
    pub usage: OperationUse,
    pub lookup: Option<LookupAddressing>,
}

pub fn key(operation: &OperationSpec, usage: OperationUse) -> Result<Key, String> {
    operation.validate().map_err(|e| e.to_string())?;
    if !operation.uses.contains(&usage) {
        return Err("undeclared schedule use".into());
    }
    Ok(Key {
        projection: None,
        kind: operation.kind,
        format: registry::by_format(operation.weights[0].format)
            .map_err(|e| e.to_string())?
            .id(),
        geometry: operation.matrix_geometry().map_err(|e| e.to_string())?,
        operands: operation.weights.len(),
        target: operation.target,
        activation: operation.activations.first().map(|a| a.format),
        weight_operand: operation.arithmetic.weight_operand,
        accumulator: operation.arithmetic.accumulator,
        output: operation.arithmetic.output,
        decode_order: operation.arithmetic.decode_order,
        rounding: operation.arithmetic.rounding,
        epilogues: operation.epilogues,
        weight_globals: operation.weights[0].global_scale.as_ref().map(|s| s.scope),
        usage,
        lookup: (operation.kind == OperationKind::Lookup).then_some(LookupAddressing {
            heads: 1,
            ids: row::LookupIds::I32,
            scale_dtype: operation.weights[0]
                .global_scale
                .as_ref()
                .map(|s| s.tensor.dtype),
        }),
    })
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", deny_unknown_fields)]
pub enum Schedule {
    Typed {
        implementation: Implementation,
    },
    Row {
        grid: crate::row::Grid,
        threads: u32,
    },
    Grouped {
        phase: moe::Phase,
        capacity: u32,
        routed_capacity: u32,
        routing: RoutedCapacity,
        knobs: CompileKnobs,
        layout: moe::WeightLayout,
        path: moe::InputPath,
        weight_stages: u8,
        exact: bool,
        route_weight: moe::RouteWeight,
        swiglu_clamp: Option<moe::SwigluClamp>,
        engine: Option<moe::template::ScheduleKnobs>,
        schedule: Option<moe::template::ScheduleKnobs>,
        invariance: crate::invariance::Invariance,
    },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Override {
    pub inputs: Key,
    pub schedule: Schedule,
}

/// Conservative shared schedules. Tuned reductions and registered engines are
/// explicit overrides. Default schedules depend only on portable execution inputs.
pub fn default(operation: &OperationSpec, inputs: &Key) -> Result<Schedule, String> {
    let usage = inputs.usage;
    let capacity = if usage.phase != ExecutionPhase::Prefill && usage.capacity <= 32 {
        usage.capacity.next_power_of_two()
    } else {
        usage.capacity.max(8192)
    };
    let implementation = match inputs.kind {
        OperationKind::GroupedDense { .. } => {
            return Err("grouped dense operations run on the grouped fold GEMM contract, not a generated dense or expert kernel".into());
        }
        OperationKind::Dense => Implementation::Dense {
            kernel: match (inputs.format, capacity <= 32) {
                (FormatId::Bf16, true) => DenseKernel::FragmentDecode { splits: 1 },
                (FormatId::Bf16, false) => DenseKernel::FragmentPrefill { segments: 4 },
                _ => DenseKernel::Tiled { splits: 1 },
            },
            capacity,
            knobs: None,
        },
        OperationKind::Grouped { .. } => Implementation::Grouped {
            phase: if capacity <= 32 {
                moe::Phase::Decode
            } else {
                moe::Phase::Prefill
            },
            capacity,
            knobs: None,
            scales: scale_bindings(operation)?,
            route_weight: moe::RouteWeight::AfterDown,
            swiglu_clamp: None,
        },
        OperationKind::Lookup => {
            let addressing = inputs.lookup.ok_or("missing lookup addressing")?;
            Implementation::Lookup {
                heads: addressing.heads,
                ids: addressing.ids,
                scale_dtype: addressing.scale_dtype,
                // Launch-shape lookups measured slower (kernel-lab
                // `launch_shape_row_stage_timing`).
                shapes: row::ShapeBinding::Compiled,
            }
        }
    };
    Ok(Schedule::Typed { implementation })
}

pub fn resolve(
    operation: &OperationSpec,
    inputs: &Key,
    overrides: &[Override],
) -> Result<Schedule, String> {
    let mut expected = key(operation, inputs.usage)?;
    if operation.kind == OperationKind::Lookup {
        expected.lookup = inputs.lookup;
    }
    if let Some(id) = &inputs.projection {
        if id.is_empty() || *id != operation.id {
            return Err("schedule selector differs from the declaration projection ID".into());
        }
        expected.projection = Some(id.clone());
    }
    if expected != *inputs {
        return Err("schedule inputs disagree with operation".into());
    }
    match overrides.iter().rev().find(|row| row.inputs == *inputs) {
        Some(row) => Ok(row.schedule.clone()),
        None => default(operation, inputs),
    }
}

pub fn offer(
    operation: &OperationSpec,
    inputs: &Key,
    overrides: &[Override],
    aliases: &[Adoption],
) -> Result<GeneratedOffer, String> {
    let Schedule::Typed { implementation } = resolve(operation, inputs, overrides)? else {
        return Err("registered schedule requires its catalog adapter".into());
    };
    super::offer(
        Request {
            operation,
            format: registry::by_format(operation.weights[0].format)
                .map_err(|e| e.to_string())?
                .id(),
            usage: inputs.usage,
            implementation,
        },
        aliases,
    )
}

/// Export-only capture from the independent hand-list reference.
pub fn capture(operation: &OperationSpec, old: &KernelOffer) -> Result<Option<Override>, String> {
    let mut inputs = key(operation, old.usage)?;
    if let Some(lookup) = old.lookup {
        inputs.lookup = Some(LookupAddressing {
            heads: lookup.heads,
            ids: lookup.ids,
            scale_dtype: lookup.global_scale,
        });
    }
    let actual = comparison::implementation(old)?;
    if let Ok(candidate) = offer(operation, &inputs, &[], &[])
        && adopt(&candidate, old).is_ok()
    {
        return Ok(None);
    }
    Ok(Some(Override {
        inputs,
        schedule: Schedule::Typed {
            implementation: actual,
        },
    }))
}

pub fn capture_grouped(
    operation: &OperationSpec,
    usage: OperationUse,
    old: &moe::Variant,
) -> Result<Override, String> {
    Ok(Override {
        inputs: key(operation, usage)?,
        schedule: Schedule::Grouped {
            phase: old.phase,
            capacity: old.tokens_capacity,
            routed_capacity: old.query.m_capacity,
            routing: old.query.grouped.ok_or("missing routing")?.capacity,
            knobs: old.knobs,
            layout: old.layout,
            path: old.path,
            weight_stages: old.weight_stages,
            exact: old.exact,
            route_weight: old.route_weight,
            swiglu_clamp: old.swiglu_clamp,
            engine: old.engine,
            schedule: old.schedule,
            invariance: old.invariance,
        },
    })
}

pub fn capture_row(
    operation: &OperationSpec,
    usage: OperationUse,
    old: &crate::row::RowInstantiation,
) -> Result<Override, String> {
    Ok(Override {
        inputs: key(operation, usage)?,
        schedule: Schedule::Row {
            grid: old.grid,
            threads: old.threads,
        },
    })
}

pub fn grouped(
    operation: &OperationSpec,
    usage: OperationUse,
    base: &moe::Variant,
    overrides: &[Override],
) -> Result<moe::Variant, String> {
    let Schedule::Grouped {
        phase,
        capacity,
        routed_capacity,
        routing,
        knobs,
        layout,
        path,
        weight_stages,
        exact,
        route_weight,
        swiglu_clamp,
        engine,
        schedule,
        invariance,
    } = resolve(operation, &key(operation, usage)?, overrides)?
    else {
        return Err("missing registered grouped schedule override".into());
    };
    let mut result = base.clone();
    result.phase = phase;
    result.tokens_capacity = capacity;
    result.query.m_capacity = routed_capacity;
    result
        .query
        .grouped
        .as_mut()
        .ok_or("missing grouped shape")?
        .capacity = routing;
    result.knobs = knobs;
    result.layout = layout;
    result.path = path;
    result.weight_stages = weight_stages;
    result.exact = exact;
    result.route_weight = route_weight;
    result.swiglu_clamp = swiglu_clamp;
    result.engine = engine;
    result.schedule = schedule;
    result.invariance = invariance;
    Ok(result)
}

pub fn row(
    operation: &OperationSpec,
    usage: OperationUse,
    base: &crate::row::RowInstantiation,
    overrides: &[Override],
) -> Result<crate::row::RowInstantiation, String> {
    let Schedule::Row { grid, threads } = resolve(operation, &key(operation, usage)?, overrides)?
    else {
        return Err("missing registered row schedule override".into());
    };
    let mut result = base.clone();
    result.grid = grid;
    result.threads = threads;
    Ok(result)
}

pub fn append(
    history: &[Override],
    incoming: impl IntoIterator<Item = Override>,
) -> Result<Vec<Override>, String> {
    let mut result = history.to_vec();
    let mut captured: Vec<Override> = Vec::new();
    for record in incoming {
        if let Some(previous) = captured.iter().find(|r| r.inputs == record.inputs) {
            if previous.schedule != record.schedule {
                return Err("conflicting schedules for identical projection inputs".into());
            }
            continue;
        }
        if result.iter().rev().find(|r| r.inputs == record.inputs) != Some(&record) {
            result.push(record.clone());
        }
        captured.push(record);
    }
    Ok(result)
}
