//! The qualified one-row MoE pipeline, bound at existing stage boundaries.
use super::*;
use kernel_core::backend::{Launch, Need, OpInfo};
use kernel_core::contract::{Argument, ArgumentList};
use kernels_sm120::moe::chained::floor::{Entry, Stage};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Point {
    Router,
    Metadata,
    Gate,
    Down,
    Sum,
}

struct Call {
    stage: Stage,
    launch: Launch,
    pointers: Vec<(String, u64)>,
    params: Vec<kernel_build::EntryParam>,
}
pub(crate) struct Prepared {
    pub info: OpInfo,
    calls: Vec<Call>,
    ticket_offset: u64,
}
impl Prepared {
    pub(crate) fn new(entry: Entry, scale_bits: u32) -> Result<Self, Refused> {
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
                .filter_map(|p| match p {
                    kernel_build::EntryParam::DevicePointer { name, .. } => {
                        Some((name.clone(), bytes(&entry, stage, name)))
                    }
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
                stage,
                pointers,
                params: spec.params,
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
        Ok(Self {
            info,
            calls,
            ticket_offset,
        })
    }
    pub(crate) fn record(&self, asks: &Asks) {
        asks.record("moe.floor Decode-1", Need::rows(1), &self.info);
    }
    pub(crate) fn launch(
        &self,
        point: Point,
        workspace: DeviceBufferSpanV1,
        mut view: impl FnMut(&str) -> Result<DeviceBufferSpanV1, String>,
    ) -> Result<Vec<Lowered>, String> {
        if workspace.length < self.info.workspace || !workspace.offset.is_multiple_of(256) {
            return Err("MoE pipeline workspace is too small or unaligned".into());
        }
        let mut bind = Bind::default();
        let mut result = Vec::new();
        for call in self.calls.iter().filter(|c| point_of(c.stage) == point) {
            let mut launch = call.launch.clone();
            for (i, param) in call.params.iter().enumerate() {
                let kernel_build::EntryParam::DevicePointer { name, .. } = param else {
                    continue;
                };
                let needed = call.pointers.iter().find(|(n, _)| n == name).unwrap().1;
                let mut span = match name.as_str() {
                    "scratch" => workspace,
                    "tickets" => DeviceBufferSpanV1::new(
                        workspace.buffer,
                        workspace
                            .offset
                            .checked_add(self.ticket_offset)
                            .ok_or("MoE ticket offset overflows")?,
                        256,
                    ),
                    _ => view(role(call.stage, name))?,
                };
                if span.length < needed || span.offset.checked_add(needed).is_none() {
                    return Err(format!("MoE pipeline {name} needs {needed} bytes"));
                }
                if name == "weight"
                    && matches!(call.stage, Stage::GateSmall | Stage::DownSmall)
                    && !span.offset.is_multiple_of(16)
                {
                    return Err("MoE packed weights require 16-byte alignment".into());
                }
                span.length = needed;
                launch.arguments.0[i] = Argument::DeviceBuffer(bind.view(span).handle);
            }
            result.push(bind.lowered(launch));
        }
        Ok(result)
    }
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

#[cfg(test)]
mod tests {
    use super::*;
    use runtime_bundle_abi::BundleDeviceBufferHandleV1;
    #[test]
    fn measured_pipeline_checks_every_bound_and_uses_disjoint_tickets() {
        let e = crate::sm120::moe::floor_entries(&model_deepseek41_contract::geometry::FLASH)
            .unwrap()
            .into_iter()
            .find(|e| e.experts() == 128)
            .unwrap();
        let p = Prepared::new(e, 1f32.to_bits()).unwrap();
        let workspace =
            DeviceBufferSpanV1::new(BundleDeviceBufferHandleV1 { raw: 1 }, 0, p.info.workspace);
        for point in [
            Point::Router,
            Point::Metadata,
            Point::Gate,
            Point::Down,
            Point::Sum,
        ] {
            let good = p
                .launch(point, workspace, |_| {
                    Ok(DeviceBufferSpanV1::new(
                        BundleDeviceBufferHandleV1 { raw: 2 },
                        0,
                        1 << 30,
                    ))
                })
                .unwrap();
            assert!(!good.is_empty());
            assert!(
                p.launch(point, workspace, |_| Ok(DeviceBufferSpanV1::new(
                    BundleDeviceBufferHandleV1 { raw: 2 },
                    0,
                    0
                )))
                .is_err()
            );
            assert!(
                p.launch(
                    point,
                    DeviceBufferSpanV1::new(
                        BundleDeviceBufferHandleV1 { raw: 1 },
                        0,
                        p.info.workspace - 1
                    ),
                    |_| Ok(DeviceBufferSpanV1::new(
                        BundleDeviceBufferHandleV1 { raw: 2 },
                        0,
                        1 << 30
                    ))
                )
                .is_err()
            );
            for op in good {
                for arg in op.launch.arguments.0 {
                    if let Argument::DeviceBuffer(id) = arg {
                        let span = op.launch.bindings.get(id).unwrap();
                        if span.buffer.raw == 1 {
                            assert!(span.offset + span.length <= workspace.length);
                        }
                    }
                }
            }
        }
    }
}
