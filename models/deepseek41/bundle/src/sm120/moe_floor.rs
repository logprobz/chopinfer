//! Prepare the qualified MoE kernel ABI for the generic stage binder.
use crate::ops::moe::floor::{Call, Point, Pointer, Prepared, Source};
use kernel_core::backend::{Launch, OpInfo, Refused};
use kernel_core::contract::{Argument, ArgumentList};
use kernels_sm120::moe::chained::floor::{Entry, Stage};

pub(crate) fn prepare(entry: Entry, scale_bits: u32) -> Result<Prepared, Refused> {
    let scratch = entry
        .workspace_bytes(1)?
        .max(entry.router_workspace_bytes(1)?);
    let ticket_offset = scratch.next_multiple_of(256);
    let mut info = OpInfo::new(
        "sm120.moe.floor.v1",
        kernel_core::properties::Invariance::any_rows(1),
        Vec::new(),
    );
    info.workspace = ticket_offset + 256;
    let mut calls = Vec::new();
    for stage in [
        Stage::RouterPrepare,
        Stage::Router,
        Stage::Metadata,
        Stage::GateSmall,
        Stage::GateFinish,
        Stage::DownSmall,
        Stage::DownFinish,
    ] {
        let spec = entry.entry(stage);
        let pointers = spec
            .params
            .iter()
            .enumerate()
            .filter_map(|(index, p)| match p {
                kernel_build::EntryParam::DevicePointer { name, .. } => Some(Pointer {
                    index,
                    source: match name.as_str() {
                        "scratch" => Source::Workspace,
                        "tickets" => Source::Tickets,
                        _ => Source::View(role(stage, name).into()),
                    },
                    bytes: bytes(&entry, stage, name),
                    alignment: if name == "weight"
                        && matches!(stage, Stage::GateSmall | Stage::DownSmall)
                    {
                        16
                    } else {
                        1
                    },
                }),
                _ => None,
            })
            .collect();
        info.symbols.push((spec.symbol.clone(), 0));
        let mut arguments = ArgumentList::default();
        for p in &spec.params {
            arguments.0.push(match p {
                kernel_build::EntryParam::U32 { name } if name == "rows" => Argument::U32(1),
                kernel_build::EntryParam::U32 { name } if name == "scaling" => {
                    Argument::U32(scale_bits)
                }
                _ => Argument::NullDevicePointer,
            });
        }
        calls.push(Call {
            point: point_of(stage),
            pointers,
            launch: Launch {
                symbol: spec.symbol,
                rows: 1,
                plan: entry.plan(stage, 1)?,
                arguments,
                collective: false,
                persistent: false,
            },
        });
    }
    Ok(Prepared::new(info, calls, ticket_offset))
}

fn point_of(stage: Stage) -> Point {
    match stage {
        Stage::RouterPrepare | Stage::Router => Point::Router,
        Stage::Metadata => Point::Metadata,
        Stage::GateSmall | Stage::GateFinish => Point::Gate,
        Stage::DownSmall => Point::Down,
        Stage::DownFinish => Point::Sum,
        _ => unreachable!(),
    }
}
fn role(stage: Stage, name: &str) -> &str {
    match (stage, name) {
        (Stage::RouterPrepare | Stage::Router, "input") => "wide",
        (Stage::Router, "weight") => "router_weight",
        (Stage::GateSmall, "input") => "input_codes",
        (Stage::GateSmall, "scales") => "input_scales",
        (Stage::GateSmall, "weight") => "gate_weight",
        (Stage::GateSmall, "weight_scales") => "gate_scales",
        (Stage::DownSmall, "input") => "codes",
        (Stage::DownSmall, "weight") => "down_weight",
        (Stage::DownSmall, "weight_scales") => "down_scales",
        _ => name,
    }
}
fn bytes(e: &Entry, s: Stage, n: &str) -> u64 {
    let (x, k, h, i) = (
        u64::from(e.experts()),
        u64::from(e.top_k()),
        u64::from(e.hidden()),
        u64::from(e.intermediate()),
    );
    match n {
        "live" | "padded" | "status" => 4,
        "tickets" => {
            if s == Stage::RouterPrepare || s == Stage::Router {
                4
            } else {
                256
            }
        }
        "scratch" => {
            if s == Stage::RouterPrepare || s == Stage::Router {
                e.router_workspace_bytes(1).unwrap()
            } else {
                e.workspace_bytes(1).unwrap()
            }
        }
        "ids" | "weights" => k * 4,
        "counts" => x * 4,
        "offsets" | "active" => (x + 1) * 4,
        "sorted" => x * 64 * 4,
        "bias" | "alternate_bias" | "scores" | "logits" => x * 4,
        "bias_select" => 1,
        "domain" => x.div_ceil(16) * 4,
        "input" => match s {
            Stage::RouterPrepare | Stage::Router => h * 4,
            Stage::GateSmall => h,
            Stage::DownSmall => k * i,
            _ => unreachable!(),
        },
        "weight" => match s {
            Stage::Router => x * h * 2,
            Stage::GateSmall => x * i * h,
            Stage::DownSmall => x * h * i / 2,
            _ => unreachable!(),
        },
        "weight_scales" => {
            if s == Stage::GateSmall {
                x * i * h / 16
            } else {
                x * h * i / 32
            }
        }
        "scales" => {
            if s == Stage::GateSmall {
                h / 32
            } else {
                k * i / 32
            }
        }
        "intermediate" => k * i * 2,
        "codes" => k * i,
        "shared" | "output" => h * 4,
        "routed" => k * h * 4,
        _ => unreachable!("unknown MoE role {n}"),
    }
}
