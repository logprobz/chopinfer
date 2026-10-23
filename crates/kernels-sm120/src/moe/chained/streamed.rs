//! A weight strip feeds every row of its expert once. Independent 128-K
//! MMA chains write scratch; a separate merge folds them in ascending K.
//!
//! A gate/up projection's small and wide queues also have builds reading
//! the bank in 64-row by 128-K tiles (`row-tiles-64x128-groups`); each lane
//! loads the same bytes, so their chains and partials are the row-major
//! queues'.
use super::quantized::Projection;
use crate::moe::{SM_COUNT, ptr};
use kernel_build::{CubinSpec, EntrySpec};
use kernel_core::contract::{LaunchPlan, Refusal, Validate};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Stage {
    Pack,
    Small,
    Wide,
    Merge,
}

pub fn stages(p: &Projection) -> Vec<Stage> {
    let mut stages = vec![Stage::Pack, Stage::Small];
    if p.tokens_capacity > 8 {
        stages.push(Stage::Wide);
    }
    stages.push(Stage::Merge);
    stages
}

pub fn symbol(p: &Projection, stage: Stage) -> String {
    format!(
        "{}_{}_v1",
        p.symbol.trim_end_matches("_v1"),
        match stage {
            Stage::Pack => "pack",
            Stage::Small => "small",
            Stage::Wide => "wide",
            Stage::Merge => "merge",
        }
    )
}

pub fn workspace_bytes(p: &Projection, rows: u32) -> Result<u64, Refusal> {
    let g = p.query.projection.gemm;
    let grouped = g.grouped.expect("routed contract");
    let header = (2 + 4 * u64::from(grouped.experts))
        .checked_mul(4)
        .and_then(|n| n.checked_next_multiple_of(256))
        .ok_or(Refusal("streamed header overflows"))?;
    [
        u64::from(rows),
        u64::from(grouped.top_k),
        if p.gate_up() { 2 } else { 1 },
        u64::from(g.k.div_ceil(128)),
        u64::from(p.columns()),
        4,
    ]
    .into_iter()
    .try_fold(1_u64, |a, b| a.checked_mul(b))
    .and_then(|bytes| bytes.checked_add(header))
    .ok_or(Refusal("streamed partials overflow"))
}

pub fn plan(p: &Projection, stage: Stage, rows: u32) -> Result<LaunchPlan, Refusal> {
    if !p.streamed || rows == 0 || rows > p.max_rows || rows > p.tokens_capacity {
        return Err(Refusal("streamed launch exceeds its selected capacity"));
    }
    let g = p.query.projection.gemm;
    let grouped = g.grouped.expect("routed contract");
    let items = u64::from(grouped.experts.min(rows * grouped.top_k))
        * u64::from(p.columns() / 64)
        * u64::from(g.k.div_ceil(128));
    let (grid, threads) = match stage {
        Stage::Pack => (1, 32),
        Stage::Small => (items.min(u64::from(SM_COUNT * 8)) as u32, 128),
        Stage::Wide => (items.min(u64::from(SM_COUNT * 4)) as u32, 128),
        Stage::Merge => (
            (u64::from(rows) * u64::from(grouped.top_k) * u64::from(p.columns())).div_ceil(256)
                as u32,
            256,
        ),
    };
    let plan = LaunchPlan {
        grid: [grid, 1, 1],
        block: [threads, 1, 1],
        dynamic_shared_bytes: 0,
        workspace_bytes: workspace_bytes(p, rows)?,
        cluster: [1, 1, 1],
        pdl: true,
    };
    plan.validate()?;
    Ok(plan)
}

pub fn entries(p: &Projection) -> Vec<EntrySpec> {
    assert!(p.streamed);
    let op = p.query.projection;
    let g = op.gemm;
    let grouped = g.grouped.expect("routed contract");
    let clamp = op.swiglu_clamp.map_or_else(
        || "chop::moe::clamp::SwigluClamp<false,0u,0u>".into(),
        |c| {
            format!(
                "chop::moe::clamp::SwigluClamp<true,{:#x}u,{:#x}u>",
                c.gate_max_bits, c.up_limit_bits
            )
        },
    );
    let kind = if p.gate_up() { "gate_up" } else { "down" };
    let config = format!(
        "chop::moe::chained::quantized::Config<chop::moe::chained::Projection::{kind},{},{},{},{},64,{},{clamp}>",
        grouped.experts,
        grouped.top_k,
        p.columns(),
        g.k,
        p.tokens_capacity
    );
    stages(p).into_iter().map(|stage| {
        let (params, call, threads) = match stage {
            Stage::Pack => (vec![ptr("counts","const int32_t"), ptr("active","const int32_t"),
                ptr("live","const int32_t"),ptr("scratch","void")],
                format!("pack<{config}>(counts,active,live,scratch)"),32),
            Stage::Small | Stage::Wide => (vec![ptr("input","const uint8_t"),ptr("weight","const uint8_t"),
                ptr("weight_scales","const uint8_t"),ptr("activation_scales","const uint8_t"),
                ptr("offsets","const int32_t"),ptr("sorted_rows","const int32_t"),
                ptr("padded_rows","const int32_t"),ptr("live","const int32_t"),ptr("scratch","void")],
                format!("compute<{config},{}>(input,weight,weight_scales,activation_scales,offsets,sorted_rows,padded_rows,live,scratch)",
                    if stage == Stage::Small {8} else {64}),128),
            Stage::Merge => (vec![ptr("scratch","const void"),ptr("live","const int32_t"),
                ptr("route_weights","const float"),ptr("output","void")],
                format!("merge<{config}>(scratch,live,route_weights,output)"),256),
        };
        EntrySpec::new(symbol(p,stage),params,format!("chop::moe::streamed::{call};"),threads,1)
    }).collect()
}

pub fn cubin_spec(name: &str, projections: &[Projection]) -> CubinSpec {
    super::spec(
        name.into(),
        "chained_streamed.cuh",
        projections.iter().flat_map(entries).collect(),
    )
}

/// The small or wide queue of a gate/up projection reading its bank in row
/// tiles.
pub fn tiled_symbol(p: &Projection, stage: Stage) -> String {
    format!("{}_tiles_v1", symbol(p, stage).trim_end_matches("_v1"))
}

/// The queues of `p` with a row-tiled build: small, and wide when the
/// projection has one.
pub fn tiled_stages(p: &Projection) -> Vec<Stage> {
    stages(p)
        .into_iter()
        .filter(|s| matches!(s, Stage::Small | Stage::Wide))
        .collect()
}

/// The queue's ABI and launch with its bank in row tiles: the same operands,
/// read at the tiled addresses.
pub fn tiled_entry(p: &Projection, stage: Stage) -> Result<EntrySpec, Refusal> {
    let rows = match stage {
        Stage::Small => 8,
        Stage::Wide => 64,
        _ => return Err(Refusal("only the small and wide queues read weights")),
    };
    if !p.gate_up() {
        return Err(Refusal("only the gate/up bank is stored in row tiles"));
    }
    if p.query.projection.gemm.k % 128 != 0 || p.columns() % 64 != 0 {
        return Err(Refusal("row tiles need whole 64-row by 128-K tiles"));
    }
    let mut e = entries(p)
        .into_iter()
        .find(|e| e.symbol == symbol(p, stage))
        .ok_or(Refusal("the projection has no such queue"))?;
    let call = format!(",{rows}>(input,");
    assert_eq!(e.body.matches(&call).count(), 1);
    e.symbol = tiled_symbol(p, stage);
    e.body = e.body.replace(&call, &format!(",{rows},true>(input,"));
    Ok(e)
}

/// [`cubin_spec`] and, after its entries, the tiled queues of each
/// projection in `tiled`.
pub fn tiled_cubin_spec(
    name: &str,
    projections: &[Projection],
    tiled: &[Projection],
) -> Result<CubinSpec, Refusal> {
    let mut spec = cubin_spec(name, projections);
    for p in tiled {
        for stage in tiled_stages(p) {
            spec.entries.push(tiled_entry(p, stage)?);
        }
    }
    Ok(spec)
}
