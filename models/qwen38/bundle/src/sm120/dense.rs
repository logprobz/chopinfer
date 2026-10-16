//! Qwen's dense groups on SM120: [`OwnOps::dense_group`] over the dense
//! entries the bundle compiles (the shared fragment family's and Qwen's
//! fused cluster decode, short-K and skinny entries).
//!
//! A projection's entry at a use is chosen at preparation: a graph bucket
//! the decode entries hold (at most `DECODE_ENTRY_ROWS` rows) launches the
//! bucket's cluster entry (`{name}_c{rows}`), the head its fragment decode
//! entry (`head_m{rows}`, `head_f32_m{rows}`) at every use, and every other
//! use the prefill entry over its prepared rows, with the eager schedule
//! only in prefill. The router, the indexer, the hyper-connection, PLE and
//! MTP projections are replicated, so every rank count launches the TP4
//! profile's entries. Equal GEMMs can run different arithmetic (a shared
//! fragment decode entry and a cluster decode entry of one shape), which is
//! why the contract names its projection. A fragment prefill launch that
//! fills every SM is persistent, as `Sm120`'s projections mark theirs.

use kernel_core::precision::{ExecutionPhase, OperationUse};
use model_qwen38_contract::own_ops::dense::{DenseGroup, DenseGroupBuffers, Projection};
use model_qwen38_contract::sm120::dense::{Role, alias};

use super::*;
use crate::contract::DECODE_ENTRY_ROWS;
use crate::sm120::catalog;
use kernel_core::contract::{DType, StaticQuery};

/// The compiled entry a dense group launches.
enum Entry {
    /// A shared fragment-family entry: one weight, one output.
    Fragment(&'static kernel_core::contract::InstantiationSpec),
    /// A Qwen fused entry; the HC producers may append the bottleneck.
    Fused(&'static catalog::qwen::Entry),
}

/// `projection`'s entry at `usage` over up to `capacity` rows, as
/// `lower.rs` named it per step before the groups were prepared.
fn symbol(
    shape: Shape,
    projection: Projection,
    usage: OperationUse,
    capacity: u32,
) -> Result<String, Refused> {
    use Projection::*;
    let decode = usage.phase != ExecutionPhase::Prefill && capacity <= DECODE_ENTRY_ROWS;
    let replicated = shape.geometry.kernel_tag;
    let sharded = shape.tag();
    let either = |name: &str| {
        if decode {
            format!("{name}_c{capacity}")
        } else {
            format!("{name}_prefill")
        }
    };
    // A role's entry is its alias's at the alias's rank count.
    let role = |role: Role| {
        let (profile, name) = alias(shape.profile(), role);
        (profile.tag(), either(name))
    };
    let (tag, name) = match projection {
        LinearIn if decode => (sharded, format!("lin_in_c{capacity}")),
        LinearQkv if !decode => (sharded, "lin_qkv_prefill".into()),
        LinearZ if !decode => (sharded, "lin_z_prefill".into()),
        LinearAb if !decode => (sharded, "lin_ab".into()),
        LinearOutput => role(Role::LinearOutput),
        AttentionIn if decode => (sharded, format!("attn_in_c{capacity}")),
        AttentionQkv if !decode => (sharded, "attn_qkv_prefill".into()),
        IndexQk if !decode => (replicated, "index_qk_prefill".into()),
        AttentionOutput => role(Role::AttentionOutput),
        Router => (replicated, either("router")),
        SharedGateUp => (sharded, either("shared_gate_up")),
        SharedDown => (sharded, either("shared_down")),
        HcIn if decode => (replicated, format!("hc_in_c{capacity}")),
        HcDown => (replicated, either("hc_down")),
        HcInject if !decode => (replicated, "hc_inject".into()),
        HcUp if !decode => (replicated, "hc_up_prefill".into()),
        PleKey => role(Role::PleKey),
        PleValue => role(Role::PleValue),
        Head(DType::Bf16) if capacity <= DECODE_ENTRY_ROWS => {
            (sharded, format!("head_m{capacity}"))
        }
        Head(DType::F32) if capacity <= DECODE_ENTRY_ROWS => {
            (sharded, format!("head_f32_m{capacity}"))
        }
        Head(DType::F32) => (sharded, "head_f32_prefill".into()),
        MtpFc => role(Role::MtpFc),
        _ => {
            return Err(rule(
                "Qwen3.8 has no dense entry for the projection at this use",
            ));
        }
    };
    Ok(format!("chop_gemm_{tag}_{name}_v1"))
}

/// Refuses an entry whose GEMMs are not the group's.
fn check(op: &DenseGroup, entry: &Entry, capacity: u32) -> Result<(), Refused> {
    let (query, segments, outputs) = match entry {
        Entry::Fragment(spec) => {
            let StaticQuery::Gemm(query) = spec.key.query else {
                return Err(rule("the dense entry has no GEMM query"));
            };
            (query, vec![query.n], 1)
        }
        Entry::Fused(entry) => {
            let outputs = if matches!(entry.kind, catalog::qwen::Kind::ClusterDecode { .. }) {
                entry.segments.len()
            } else {
                1
            };
            (entry.query, entry.segments.clone(), outputs)
        }
    };
    if query.k != op.k
        || segments != op.members
        || query.output != op.output
        || query.epilogues.swiglu != op.swiglu
        || outputs != op.strides.len()
        || capacity > query.m_capacity
    {
        return Err(rule("the dense entry computes other GEMMs than the group"));
    }
    if op.bottleneck.is_some() && capacity != query.m_capacity {
        return Err(rule(
            "the bottleneck epilogue serves its entry's whole bucket",
        ));
    }
    Ok(())
}

/// Prepares `op` for steps of `usage` over up to `need.capacity` rows.
pub(super) fn prepare(
    shape: Shape,
    op: &DenseGroup,
    usage: OperationUse,
    need: Need,
) -> Result<Prepared<DenseGroupBuffers<View>>, Refused> {
    op.validate()?;
    let compiled = DenseGroup::of(shape, op.projection);
    if *op != compiled && *op != compiled.clone().with_bottleneck(shape.streams()) {
        return Err(rule(
            "the dense group is compiled for the shape's projection",
        ));
    }
    let capacity = need.capacity;
    let produced = symbol(shape, op.projection, usage, capacity)?;
    let entry = match catalog::qwen::entry(&produced) {
        Some(entry) => Entry::Fused(entry),
        None => Entry::Fragment(
            catalog::dense::entry(&produced)
                .map_err(|_| rule("the bundle has no such dense entry"))?,
        ),
    };
    check(op, &entry, capacity)?;
    let launched = match (&entry, op.bottleneck) {
        (_, None) => produced,
        (Entry::Fused(entry), Some(_)) => {
            kernels_sm120::gemm::fused::hc_precompute_spec(entry)?.symbol
        }
        (Entry::Fragment(_), Some(_)) => {
            return Err(rule("only a fused HC producer writes the bottleneck"));
        }
    };
    let prefill = usage.phase == ExecutionPhase::Prefill;
    let (partial, semaphores) = match &entry {
        Entry::Fragment(spec) => {
            let counts = kernels_sm120::gemm::row_counts(&spec.key, capacity);
            (
                kernels_sm120::gemm::plan(&spec.key, counts)?.workspace_bytes,
                4,
            )
        }
        Entry::Fused(entry) => (
            entry.workspace_capacity()?,
            u64::from(entry.semaphore_words()) * 4,
        ),
    };
    let shared = catalog::dynamic_shared_bytes(&launched)?;
    let op = op.clone();
    let symbol = launched.clone();
    prepare_info(
        "dense_group",
        &launched,
        need,
        |info| {
            info.symbols = vec![(launched.clone(), shared)];
            info.bytes = vec![
                ("partial".into(), partial),
                ("semaphores".into(), semaphores),
            ];
            info.weight_layouts = vec![(
                "weights".into(),
                kernel_core::precision::WeightLayout {
                    id: "bf16-fragment".into(),
                    version: 1,
                },
            )];
        },
        move |b: &DenseGroupBuffers<View>, counts| {
            let rows = rows(counts, capacity)?;
            if b.weights.len() != op.members.len()
                || b.outputs.len() != op.strides.len()
                || b.bottleneck.is_some() != op.bottleneck.is_some()
            {
                return Err(Refusal("a dense group's views are not its contract's"));
            }
            let mut launch = match &entry {
                Entry::Fragment(spec) => fragment(spec, &op, b, rows)?,
                Entry::Fused(entry) => fused(entry, &op, b, rows, prefill)?,
            };
            if let Some(bottleneck) = b.bottleneck {
                if rows != capacity {
                    return Err(Refusal("the bottleneck epilogue launches its whole bucket"));
                }
                bottleneck.covers(bytes(rows, u64::from(op.members[0]) * 2))?;
                launch.symbol = symbol.clone();
                launch.arguments.0.push(buffer(bottleneck));
                launch.arguments.validate()?;
            }
            Ok(launch)
        },
    )
}

/// The bytes `rows` rows of an output of `columns` columns at `stride`
/// take: up to the last row's last column.
fn output_bytes(rows: u32, stride: u32, columns: u32, element: u64) -> u64 {
    (u64::from(rows - 1) * u64::from(stride) + u64::from(columns)) * element
}

/// A fragment-family launch, as `catalog::gemm_entry` builds it.
fn fragment(
    spec: &kernel_core::contract::InstantiationSpec,
    op: &DenseGroup,
    b: &DenseGroupBuffers<View>,
    rows: u32,
) -> Result<Launch, Refusal> {
    let StaticQuery::Gemm(query) = spec.key.query else {
        return Err(Refusal("Qwen3.8 dense entry has no GEMM query"));
    };
    let (weight, output) = (b.weights[0], b.outputs[0]);
    let columns = op.columns(0);
    let element = if query.output == DType::F32 { 4 } else { 2 };
    // Every output holds the entry's full row capacity except a vocabulary
    // head's: logits buffers hold the rows a phase can carry.
    let output_rows = if matches!(op.projection, Projection::Head(_)) {
        rows
    } else {
        query.m_capacity
    };
    if op.strides[0] != columns {
        return Err(Refusal("fragment GEMM output rows are its N wide"));
    }
    weight.covers(u64::from(query.n) * u64::from(query.k) * 2)?;
    b.activation
        .covers(u64::from(query.m_capacity) * u64::from(query.k) * 2)?;
    output.covers(u64::from(output_rows) * u64::from(columns) * element)?;
    b.semaphores.covers(4)?;
    let decode = query.m_capacity <= DECODE_ENTRY_ROWS;
    let maps = kernels_sm120::gemm::fragment_maps(
        &spec.key,
        &query,
        b.activation.handle,
        weight.handle,
        decode,
    );
    let counts = kernels_sm120::gemm::row_counts(&spec.key, rows);
    let plan = kernels_sm120::gemm::plan(&spec.key, counts)?;
    b.partial.covers(plan.workspace_bytes)?;
    let buffers = catalog::dense::GemmBuffers {
        output: output.handle,
        partial: b.partial.handle,
        semaphores: b.semaphores.handle,
        weight_scales: None,
        activation_scales: None,
        residual: None,
        weight_global_scale: None,
        activation_global_scale: None,
    };
    let arguments = kernels_sm120::gemm::arguments(&spec.key, counts, maps, buffers)?;
    // A fragment prefill GEMM runs one block per SM over its grid.
    let persistent = matches!(
        spec.key.dense,
        Some(kernel_core::contract::DenseKernel::FragmentPrefill { .. })
    ) && plan.grid == [kernels_sm120::gemm::SM_COUNT, 1, 1];
    Ok(Launch {
        symbol: spec.symbol.clone(),
        rows,
        plan,
        arguments,
        collective: false,
        persistent,
    })
}

/// A Qwen fused launch, as `catalog::qwen_gemm` builds it; `prefill` takes
/// the entry's eager schedule.
fn fused(
    entry: &catalog::qwen::Entry,
    op: &DenseGroup,
    b: &DenseGroupBuffers<View>,
    rows: u32,
    prefill: bool,
) -> Result<Launch, Refusal> {
    let element = if op.output == DType::F32 { 4 } else { 2 };
    let cluster = matches!(entry.kind, catalog::qwen::Kind::ClusterDecode { .. });
    for (weight, &members) in b.weights.iter().zip(&op.members) {
        weight.covers(u64::from(members.div_ceil(16) * 16) * u64::from(op.k) * 2)?;
    }
    b.activation.covers(bytes(rows, u64::from(op.k) * 2))?;
    for (index, (output, &stride)) in b.outputs.iter().zip(&op.strides).enumerate() {
        // A cluster entry writes each member's rows; the others one row of
        // every member's columns.
        let columns = if cluster {
            op.columns(index)
        } else {
            entry.query.n / (1 + u32::from(op.swiglu.is_some()))
        };
        if stride < columns {
            return Err(Refusal("Qwen3.8 GEMM views do not cover the entry"));
        }
        output.covers(output_bytes(rows, stride, columns, element))?;
    }
    let plan = if prefill {
        entry.prefill_plan(rows)?
    } else {
        entry.plan(rows)?
    };
    b.partial.covers(plan.workspace_bytes)?;
    b.semaphores
        .covers(u64::from(entry.semaphore_words()) * 4)?;
    let mut buffers = catalog::qwen::Buffers {
        activation: b.activation.handle,
        partial: b.partial.handle,
        semaphores: b.semaphores.handle,
        ..catalog::qwen::Buffers::default()
    };
    for (index, weight) in b.weights.iter().enumerate() {
        buffers.weights[index] = weight.handle;
    }
    for (index, (output, &stride)) in b.outputs.iter().zip(&op.strides).enumerate() {
        buffers.outputs[index] = output.handle;
        buffers.output_strides[index] = stride;
    }
    let arguments = if prefill {
        entry.prefill_arguments(rows, buffers)?
    } else {
        entry.arguments(rows, buffers)?
    };
    Ok(Launch {
        symbol: entry.symbol.clone(),
        rows,
        plan,
        arguments,
        collective: false,
        persistent: false,
    })
}
