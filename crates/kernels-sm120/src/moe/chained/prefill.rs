//! Expert-grouped 8/32/64-row tiles with local, ordered 128-K folds.
use super::quantized::Projection;
use crate::moe::{SM_COUNT, ptr};
use kernel_build::{CubinSpec, EntryParam, EntrySpec};
use kernel_core::backend::Launch;
use kernel_core::contract::{Argument, ArgumentList, LaunchPlan, Refusal, Validate};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage {
    Pack,
    Rows8,
    Rows32,
    Rows64,
}
pub const STAGES: [Stage; 4] = [Stage::Pack, Stage::Rows8, Stage::Rows32, Stage::Rows64];
pub const MMA: &str = "chop_throughput_bf16_v1";
pub const READ: &str = "chop_throughput_read_v1";

pub fn symbol(p: &Projection, stage: Stage) -> String {
    format!(
        "{}_{}_v1",
        p.symbol.trim_end_matches("_v1"),
        match stage {
            Stage::Pack => "pack",
            Stage::Rows8 => "r8",
            Stage::Rows32 => "r32",
            Stage::Rows64 => "r64",
        }
    )
}
pub fn negative_symbol(symbol: &str) -> String {
    format!("{}_omit_first_v1", symbol.trim_end_matches("_v1"))
}
pub fn slots(p: &Projection) -> u64 {
    let g = p.query.projection.gemm.grouped.unwrap();
    (u64::from(p.tokens_capacity) * u64::from(g.top_k)).div_ceil(64) + u64::from(g.experts)
}
pub fn workspace_bytes(p: &Projection) -> u64 {
    ((3 + 9 * slots(p)) * 4).next_multiple_of(256)
}
pub fn plan(p: &Projection, stage: Stage, rows: u32) -> Result<LaunchPlan, Refusal> {
    if !p.prefill || rows == 0 || rows > p.max_rows || rows > p.tokens_capacity {
        return Err(Refusal("grouped prefill launch exceeds selected capacity"));
    }
    let g = p.query.projection.gemm.grouped.unwrap();
    let tails = u64::from(g.experts.min(rows * g.top_k));
    let (tiles, ctas, threads) = match stage {
        Stage::Pack => (1, 1, 32),
        Stage::Rows8 => (tails * u64::from(p.columns() / 64), SM_COUNT * 8, 128),
        Stage::Rows32 => (tails * u64::from(p.columns() / 64), SM_COUNT * 4, 128),
        Stage::Rows64 => (
            (u64::from(rows * g.top_k).div_ceil(64) + tails) * u64::from(p.columns() / 64),
            SM_COUNT * 2,
            256,
        ),
    };
    let plan = LaunchPlan {
        grid: [tiles.min(u64::from(ctas)) as u32, 1, 1],
        block: [threads, 1, 1],
        dynamic_shared_bytes: 0,
        workspace_bytes: workspace_bytes(p),
        cluster: [1, 1, 1],
        pdl: true,
    };
    plan.validate()?;
    Ok(plan)
}

pub fn launches(
    p: &Projection,
    rows: u32,
    h: [u64; 10],
    counts: u64,
    active: u64,
    scratch: u64,
) -> Result<Vec<Launch>, Refusal> {
    STAGES
        .into_iter()
        .map(|stage| {
            let handles = if stage == Stage::Pack {
                vec![counts, h[4], active, h[7], scratch]
            } else {
                vec![
                    h[0], h[1], h[2], h[3], h[5], h[6], h[7], scratch, h[8], h[9],
                ]
            };
            let arguments = ArgumentList(
                handles
                    .into_iter()
                    .map(|h| {
                        if h == 0 {
                            Argument::NullDevicePointer
                        } else {
                            Argument::DeviceBuffer(h)
                        }
                    })
                    .collect(),
            );
            arguments.validate()?;
            Ok(Launch {
                symbol: symbol(p, stage),
                rows,
                plan: plan(p, stage, rows)?,
                arguments,
                collective: false,
                persistent: false,
            })
        })
        .collect()
}

pub fn entries(p: &Projection) -> Vec<EntrySpec> {
    assert!(p.prefill);
    let op = p.query.projection;
    let g = op.gemm;
    let grouped = g.grouped.unwrap();
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
    let mut entries = Vec::new();
    for stage in STAGES {
        if stage == Stage::Pack {
            entries.push(EntrySpec::new(symbol(p,stage),vec![ptr("counts","const int32_t"),ptr("offsets","const int32_t"),ptr("active","const int32_t"),ptr("live","const int32_t"),ptr("scratch","void")],
                format!("chop::moe::prefill_quantized::pack<{config}>(counts,offsets,active,live,scratch);"),32,1));
        } else {
            let rows = match stage {
                Stage::Rows8 => 8,
                Stage::Rows32 => 32,
                Stage::Rows64 => 64,
                _ => unreachable!(),
            };
            let params = vec![
                ptr("input", "const uint8_t"),
                ptr("weight", "const uint8_t"),
                ptr("weight_scales", "const uint8_t"),
                ptr("activation_scales", "const uint8_t"),
                ptr("sorted", "const int32_t"),
                ptr("padded", "const int32_t"),
                ptr("live", "const int32_t"),
                ptr("scratch", "const void"),
                ptr("route_weights", "const float"),
                ptr("output", "void"),
            ];
            for mutant in [false, true] {
                if mutant && p.gate_up() {
                    continue;
                }
                let name = symbol(p, stage);
                entries.push(EntrySpec::new(if mutant {negative_symbol(&name)} else {name},params.clone(),
                    format!("chop::moe::prefill_quantized::compute<{config},{rows},{mutant}>(input,weight,weight_scales,activation_scales,sorted,padded,live,scratch,route_weights,output);"),if rows == 64 { 256 } else { 128 },1));
            }
        }
    }
    entries
}

pub fn cubin_spec(name: &str, projections: &[Projection]) -> CubinSpec {
    super::spec(
        name.into(),
        "chained_prefill_quantized.cuh",
        projections.iter().flat_map(entries).collect(),
    )
}
pub fn calibration_cubin_spec(name: &str) -> CubinSpec {
    let mut spec = super::spec(
        name.into(),
        "chained_prefill_quantized.cuh",
        vec![
            EntrySpec::new(
                MMA.into(),
                vec![
                    ptr("output", "float"),
                    ptr("input", "const uint32_t"),
                    EntryParam::U32 {
                        name: "iterations".into(),
                    },
                ],
                "chop::throughput::bf16(output,input,iterations);".into(),
                256,
                1,
            ),
            EntrySpec::new(
                READ.into(),
                vec![
                    ptr("output", "uint32_t"),
                    ptr("input", "const uint4"),
                    EntryParam::U64 {
                        name: "vectors".into(),
                    },
                ],
                "chop::throughput::read(output,input,vectors);".into(),
                256,
                1,
            ),
        ],
    );
    spec.headers = vec!["#include \"common/throughput.cuh\"".into()];
    spec
}
