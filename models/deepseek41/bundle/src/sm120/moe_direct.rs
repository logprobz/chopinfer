//! Qualify exact row/phase buckets and prepare their locked direct MoE ABI.
use crate::ops::moe::direct::{Call, Point, Pointer, Prepared, Source};
use kernel_core::backend::{Launch, OpInfo, Refused};
use kernel_core::contract::{Argument, ArgumentList};
use kernel_core::precision::ExecutionPhase;
use kernels_sm120::moe::chained::{
    bandwidth,
    floor::{Entry, Stage},
};
use model_deepseek41_contract::moe::Experts;

// TP4, 250 W, 2026-10-19, paired full graphs, bitwise in all three routing
// patterns. Direct gains at least 3% in these buckets. Backbone Verify-48
// gains only 1.6% for all-distinct routing and retains the prior selection.
// Backbone C1/C8: 76.16/284.40 -> 65.22/255.17 us per full MoE.
// Drafter C1/C8: 61.65/126.51 -> 47.22/112.38 us against the qualified path.
// Prefetch loses at larger dispersed buckets. No unmeasured rows are inferred.
// Verify 1/2/4/5/8/16/32 also passed complete bitwise graphs on both GPUs.
// Quiet-host confirmation at 250 W passed all 84 newly admitted cases;
// direct/previous time spans 0.79625..0.95890 across three route patterns.
// Full graphs, poisoned replays and clean eager/graph memcheck preserve
// the result. Local-fold candidates remain unselected. A separate measured
// router epilogue may fuse projection reduction, selection and metadata.
fn buckets(set: Experts) -> Vec<(ExecutionPhase, u32)> {
    use ExecutionPhase::{Decode, Verify};
    let mut out = [1, 2, 4, 8, 16, 32].map(|n| (Decode, n)).to_vec();
    out.extend([1, 2, 4, 5, 6, 8, 16, 32, 64].map(|rows| (Verify, rows)));
    if set.count == 128 {
        out.push((Verify, 48));
    }
    out
}

pub(super) fn prepare_set(
    shape: &crate::shape::Shape,
    set: Experts,
) -> Result<Vec<(ExecutionPhase, u32, Prepared)>, Refused> {
    if !super::moe::streamed_qualified(shape.geometry, shape.ranks())
        || !matches!((set.count, set.top_k), (384, 6) | (128, 3))
    {
        return Ok(Vec::new());
    }
    let entry = super::moe_direct_kernels::entries(shape.geometry)
        .map_err(Refused::Rule)?
        .into_iter()
        .find(|e| e.experts() == set.count)
        .expect("declared expert set");
    let scaling = model_deepseek41_contract::moe::router(shape.geometry, set)
        .scaling_factor
        .to_bits();
    buckets(set)
        .into_iter()
        .map(|(phase, rows)| {
            Ok((
                phase,
                rows,
                prepare(
                    &entry,
                    rows,
                    phase,
                    scaling,
                    super::moe_router_finish::qualified(set, phase, rows),
                    super::moe_router_finish::r2(set, phase, rows),
                )?,
            ))
        })
        .collect()
}

fn prepare(
    entry: &Entry,
    rows: u32,
    phase: ExecutionPhase,
    scaling: u32,
    fused_router: bool,
    r2: bool,
) -> Result<Prepared, Refused> {
    let scratch = entry.workspace_bytes(rows)?;
    let routed_offset = if fused_router {
        super::moe_router_finish::layout(entry, rows).routed
    } else {
        0
    };
    let ticket_offset = routed_offset + scratch.next_multiple_of(256);
    let mut info = OpInfo::new(
        "sm120.moe.direct.v1",
        kernel_core::properties::Invariance::any_rows(rows),
        Vec::new(),
    );
    info.workspace = ticket_offset + 256;
    let mut calls = if fused_router {
        super::moe_router_finish::calls(entry, rows, scaling, r2)?
    } else {
        Vec::new()
    };
    info.symbols.extend(calls.iter().map(|call| {
        (
            call.launch.symbol.clone(),
            call.launch.plan.dynamic_shared_bytes,
        )
    }));
    let router = bandwidth::router(entry);
    let mut stages = vec![(None, router.entry()?, router.plan(rows)?)];
    for s in [
        Stage::Metadata,
        Stage::GateSmall,
        Stage::GateWide,
        Stage::GateFinish,
        Stage::DownSmall,
        Stage::DownWide,
        Stage::DownFinish,
    ] {
        if rows <= 8 && matches!(s, Stage::GateWide | Stage::DownWide) {
            continue;
        }
        let direct = matches!(
            s,
            Stage::GateSmall
                | Stage::GateWide
                | Stage::GateFinish
                | Stage::DownSmall
                | Stage::DownWide
        );
        stages.push((
            Some(s),
            if direct {
                bandwidth::entry(entry, s, bandwidth::Loads::Direct)
            } else {
                entry.entry(s)
            },
            if direct {
                bandwidth::plan(entry, s, rows)?
            } else {
                entry.plan(s, rows)?
            },
        ));
    }
    for (stage, mut spec, mut plan) in stages {
        // TP4, 250 W, 2026-10-20: full captured graphs passed bitwise on
        // both GPUs and all four logical ranks. Matching the 80-register
        // small gate's six-CTA residency saves 0.60 ms at AR8, 1.14 ms at
        // AR32 and 0.94..1.17 ms at verify32 across 40 backbone layers.
        // Keep the locked cubin, operands, K128 chains and partial ABI.
        // Other phase/row pairs and the drafter bank retain their prior grid.
        if stage == Some(Stage::GateSmall)
            && entry.experts() == 384
            && matches!(
                (phase, rows),
                (ExecutionPhase::Decode, 8 | 32) | (ExecutionPhase::Verify, 32)
            )
        {
            plan.grid[0] = plan.grid[0].min(kernels_sm120::moe::SM_COUNT * 6);
        }
        // TP4, 250 W, 2026-10-20: the same small-gate body under an
        // eight-CTA launch bound (64 registers, eight resident CTAs) passed
        // 708 captured full graphs bitwise. At 32 rows it saves 0.20..0.25 ms
        // over grid6 across 40 backbone layers, faster in 71..79 of 80 graphs
        // per arm. Decode-8 graphs are within noise (42 of 80) and keep grid6.
        // TP4, 250 W, 2026-10-21, graph-node trace of captured row prefixes
        // over the r2 finish and edge (c1-queue-trace): at backbone Verify-4
        // (DSpark C1 K3) the gate small queue ends 7.1..7.7 us sooner per
        // layer, every case bitwise and every candidate sample faster than
        // every served sample. Decode-1 is flat and keeps the direct gate.
        if stage == Some(Stage::GateSmall)
            && entry.experts() == 384
            && (rows == 32 || (phase == ExecutionPhase::Verify && rows == 4))
        {
            // The positional ABI is identical. Retain the floor binding's
            // view names for scales, sorted rows and padded rows.
            spec.symbol = kernels_sm120::moe::chained::streamed_resident::symbol(&entry.gate);
            plan = kernels_sm120::moe::chained::streamed_resident::plan(&entry.gate, rows)?;
            plan.workspace_bytes = entry.workspace_bytes(rows)?;
        }
        if fused_router && matches!(stage, None | Some(Stage::Metadata)) {
            continue;
        }
        let pointers = spec
            .params
            .iter()
            .enumerate()
            .filter_map(|(index, p)| match p {
                kernel_build::EntryParam::DevicePointer { name, .. } => Some(Pointer {
                    index,
                    source: match name.as_str() {
                        "scratch" => Source::WorkspaceAt(routed_offset),
                        "tickets" => Source::Tickets,
                        _ => Source::View(role(stage, name).into()),
                    },
                    bytes: bytes(entry, stage, name, rows),
                    alignment: if name == "weight" { 16 } else { 1 },
                }),
                _ => None,
            })
            .collect();
        info.symbols
            .push((spec.symbol.clone(), plan.dynamic_shared_bytes));
        let arguments = ArgumentList(
            spec.params
                .iter()
                .map(|p| match p {
                    kernel_build::EntryParam::U32 { name } if name == "scale_bits" => {
                        Argument::U32(scaling)
                    }
                    _ => Argument::NullDevicePointer,
                })
                .collect(),
        );
        calls.push(Call {
            point: match stage {
                None | Some(Stage::Metadata) => Point::Route,
                Some(Stage::GateSmall | Stage::GateWide | Stage::GateFinish) => Point::Gate,
                Some(Stage::DownSmall | Stage::DownWide) => Point::Down,
                Some(Stage::DownFinish) => Point::Sum,
                _ => unreachable!(),
            },
            pointers,
            launch: Launch {
                symbol: spec.symbol,
                rows,
                plan,
                arguments,
                collective: false,
                persistent: false,
            },
        });
    }
    Ok(Prepared::new(info, calls, ticket_offset))
}

fn role(stage: Option<Stage>, name: &str) -> &str {
    match (stage, name) {
        (None, "live_tokens") => "live",
        (Some(Stage::GateSmall | Stage::GateWide), "input") => "input_codes",
        (Some(Stage::GateSmall | Stage::GateWide), "scales") => "input_scales",
        (Some(Stage::GateSmall | Stage::GateWide), "weight") => "gate_weight",
        (Some(Stage::GateSmall | Stage::GateWide), "weight_scales") => "gate_scales",
        (Some(Stage::DownSmall | Stage::DownWide), "input") => "codes",
        (Some(Stage::DownSmall | Stage::DownWide), "weight") => "down_weight",
        (Some(Stage::DownSmall | Stage::DownWide), "weight_scales") => "down_scales",
        _ => name,
    }
}
fn bytes(e: &Entry, s: Option<Stage>, name: &str, rows: u32) -> u64 {
    let (x, k, h, i, r) = (
        u64::from(e.experts()),
        u64::from(e.top_k()),
        u64::from(e.hidden()),
        u64::from(e.intermediate()),
        u64::from(rows),
    );
    let gate = matches!(s, Some(Stage::GateSmall | Stage::GateWide));
    match name {
        "live" | "live_tokens" | "padded" => 4,
        "status" => r * 4,
        "tickets" => 256,
        "scratch" => e.workspace_bytes(rows).unwrap(),
        "ids" | "weights" => r * k * 4,
        "counts" => x * 4,
        "offsets" | "active" => (x + 1) * 4,
        "sorted" => x * 64 * 4,
        "bias" | "alternate_bias" => x * 4,
        "scores" | "logits" => r * x * 4,
        "bias_select" => r,
        "input" => {
            if gate {
                r * h
            } else {
                r * k * i
            }
        }
        "weight" => {
            if gate {
                x * i * h
            } else {
                x * h * i / 2
            }
        }
        "weight_scales" => {
            if gate {
                x * i * h / 16
            } else {
                x * h * i / 32
            }
        }
        "scales" => {
            if gate {
                r * h / 32
            } else {
                r * k * i / 32
            }
        }
        "intermediate" => r * k * i * 2,
        "codes" => r * k * i,
        "shared" | "output" => r * h * 4,
        "routed" => r * k * h * 4,
        _ => unreachable!("unknown direct MoE argument {name}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bundle_runtime::geometry::Ranks;
    use model_deepseek41_contract::geometry::FLASH;
    use runtime_bundle_abi::{BundleDeviceBufferHandleV1, DeviceBufferSpanV1};

    #[test]
    fn direct_admission_keeps_unmeasured_phases_and_weak_buckets_out() {
        for ranks in [Ranks::TP1, Ranks::TP2, Ranks::TP4] {
            let shape = crate::shape::Shape::new(&FLASH, ranks).unwrap();
            for (_, cubin) in super::super::moe_direct_kernels::cubins(&[FLASH]).unwrap() {
                assert_eq!(
                    crate::Kernels::loads(&cubin.name, &shape),
                    ranks == Ranks::TP4
                );
                assert!(!crate::Kernels::drafter_only(&cubin.name, &FLASH));
            }
            for set in model_deepseek41_contract::moe::expert_sets(&FLASH) {
                let ops = prepare_set(&shape, set).unwrap();
                for rows in 1..=192 {
                    for phase in [
                        ExecutionPhase::Prefill,
                        ExecutionPhase::Decode,
                        ExecutionPhase::Verify,
                    ] {
                        let wanted = ranks == Ranks::TP4
                            && match phase {
                                ExecutionPhase::Decode => [1, 2, 4, 8, 16, 32].contains(&rows),
                                ExecutionPhase::Verify => {
                                    [1, 2, 4, 5, 6, 8, 16, 32, 64].contains(&rows)
                                        || (set.count == 128 && rows == 48)
                                }
                                _ => false,
                            };
                        assert_eq!(
                            ops.iter().any(|(p, n, _)| *p == phase && *n == rows),
                            wanted,
                            "{phase:?} rows={rows} experts={}",
                            set.count
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn gate_grid_selects_only_measured_backbone_buckets() {
        let shape = crate::shape::Shape::new(&FLASH, Ranks::TP4).unwrap();
        for set in model_deepseek41_contract::moe::expert_sets(&FLASH) {
            for (phase, rows, op) in prepare_set(&shape, set).unwrap() {
                let workspace = DeviceBufferSpanV1::new(
                    BundleDeviceBufferHandleV1 { raw: 1 },
                    0,
                    op.info.workspace,
                );
                for point in [Point::Gate, Point::Down] {
                    for call in op
                        .launch(point, workspace, |_| {
                            Ok(DeviceBufferSpanV1::new(
                                BundleDeviceBufferHandleV1 { raw: 2 },
                                0,
                                1 << 32,
                            ))
                        })
                        .unwrap()
                    {
                        if call.launch.symbol.contains("_small_") {
                            let resident = if set.count == 384
                                && phase == ExecutionPhase::Decode
                                && rows == 8
                                && point == Point::Gate
                            {
                                6
                            } else {
                                8
                            };
                            assert_eq!(
                                call.launch.symbol.ends_with("_small_resident8_v1"),
                                set.count == 384
                                    && (rows == 32
                                        || (phase == ExecutionPhase::Verify && rows == 4))
                                    && point == Point::Gate
                            );
                            let items = set.count.min(rows * set.top_k)
                                * if point == Point::Gate { 9 * 40 } else { 80 * 5 };
                            assert_eq!(
                                call.launch.plan.grid,
                                [items.min(188 * resident), 1, 1],
                                "{} rows={rows}",
                                call.launch.symbol
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn direct_binding_bounds_every_view_and_keeps_tickets_outside_partials() {
        let shape = crate::shape::Shape::new(&FLASH, Ranks::TP4).unwrap();
        for set in model_deepseek41_contract::moe::expert_sets(&FLASH) {
            for (_, rows, op) in prepare_set(&shape, set).unwrap() {
                let workspace = DeviceBufferSpanV1::new(
                    BundleDeviceBufferHandleV1 { raw: 1 },
                    256,
                    op.info.workspace,
                );
                for point in [
                    Point::RouterPrepare,
                    Point::Route,
                    Point::Gate,
                    Point::Down,
                    Point::Sum,
                ]
                .into_iter()
                .filter(|point| *point != Point::RouterPrepare || op.fuses_router())
                {
                    let views = |_: &str| {
                        Ok(DeviceBufferSpanV1::new(
                            BundleDeviceBufferHandleV1 { raw: 2 },
                            0,
                            1 << 32,
                        ))
                    };
                    let good = op.launch(point, workspace, views).unwrap();
                    assert!(!good.is_empty());
                    assert!(
                        op.launch(
                            point,
                            DeviceBufferSpanV1 {
                                length: op.info.workspace - 1,
                                ..workspace
                            },
                            views
                        )
                        .is_err()
                    );
                    assert!(
                        op.launch(point, workspace, |_| Ok(DeviceBufferSpanV1::new(
                            BundleDeviceBufferHandleV1 { raw: 2 },
                            0,
                            0
                        )))
                        .is_err()
                    );
                    for call in good {
                        assert_eq!(call.launch.rows, rows);
                        if call
                            .launch
                            .symbol
                            .starts_with("chop_moe_sqrt_softplus_row_bias_")
                        {
                            assert_eq!(call.launch.plan.grid, [8, 1, 1]);
                        }
                        assert!(!call.launch.symbol.contains("bandwidth_prefetch"));
                        for arg in &call.launch.arguments.0 {
                            if let Argument::DeviceBuffer(id) = arg {
                                let span = call.launch.bindings.get(*id).unwrap();
                                if span.buffer == workspace.buffer {
                                    assert!(span.offset >= workspace.offset);
                                    assert!(
                                        span.offset + span.length
                                            <= workspace.offset + workspace.length
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
