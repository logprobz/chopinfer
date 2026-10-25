//! Qualify exact row/phase buckets and prepare their locked direct MoE ABI.
use crate::ops::moe::direct::{Call, Point, Pointer, Prepared, Source};
use kernel_core::backend::{Launch, OpInfo, Refused};
use kernel_core::contract::{Argument, ArgumentList};
use kernel_core::precision::ExecutionPhase;
use kernels_sm120::moe::chained::{
    absorb, bandwidth,
    floor::{Entry, Stage},
    strip2,
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
fn buckets(set: Experts, maximum_verify_rows: u32) -> Vec<(ExecutionPhase, u32)> {
    use ExecutionPhase::{Decode, Verify};
    let mut out = [1, 2, 4, 8, 16, 32].map(|n| (Decode, n)).to_vec();
    out.extend([1, 2, 4, 5, 6, 8, 16, 32, 64].map(|rows| (Verify, rows)));
    if set.count == 128 {
        out.push((Verify, 48));
    } else {
        out.extend((33..=maximum_verify_rows.min(63)).map(|rows| (Verify, rows)));
    }
    for rows in 2..=32 {
        if super::moe_router_finish::small_gap(set, Verify, rows) && !out.contains(&(Verify, rows))
        {
            out.push((Verify, rows));
        }
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
    let tiles = super::moe_row_tiles::builds_for(
        shape.geometry,
        super::moe_row_tiles_op::layout(shape.geometry, shape.ranks(), set),
    )
    .map_err(Refused::Rule)?;
    buckets(set, shape.verify_decode_rows())
        .into_iter()
        .map(|(phase, rows)| {
            let wide = super::moe_router_finish::wide(set, phase, rows, shape.verify_decode_rows());
            Ok((
                phase,
                rows,
                prepare(
                    &entry,
                    rows,
                    phase,
                    scaling,
                    super::moe_router_finish::qualified(set, phase, rows) || wide,
                    super::moe_router_finish::r2(set, phase, rows) || wide,
                    &tiles,
                )?,
            ))
        })
        .collect()
}

/// TP4, 250 W, 2026-10-21, captured full MoE graphs (absorb-r1): small
/// queues that also run the wide queues' experts, with no wide launches,
/// passed all 80 AR C32 and DSpark C8 K3 (32-row) cases bitwise on both
/// GPUs, memcheck clean. They save 0.93 ms per AR C32 step (20 of 20 layers
/// faster) and 1.03 ms per verify-32 cycle (18 of 20) over 40 backbone
/// layers; the A/A control moved 0.01..0.02 ms. Same parameters, grids and
/// partial layout as the small queues; the module is the qualified lab
/// module's bytes. Decode-16 and the drafter bank are unmeasured.
fn absorbs(entry: &Entry, phase: ExecutionPhase, rows: u32) -> bool {
    entry.experts() == 384
        && matches!(
            (phase, rows),
            (ExecutionPhase::Decode, 32) | (ExecutionPhase::Verify, 32)
        )
}

/// TP4, 250 W, 2026-10-22, captured full MoE graphs (strip2-r1): the
/// absorbing down small queue with each item covering two adjacent
/// 64-column strips of one expert and 128-K block, staging its rows and
/// activations once for both, passed all 80 cases per GPU bitwise (AR C32
/// and C16, GSM8K and speed workloads), memcheck clean, with no wide down
/// launch. Against serving it saves 0.47 (GSM8K) and 0.82 ms (speed) per
/// Decode-32 step and 0.43 and 0.39 ms per Decode-16 step, which also drops
/// the direct down small and wide queues; 79 of 80 timed layers faster, no
/// slow sample, A/A within 0.02 ms. Each strip keeps its codes, scales,
/// chain, K order and partial address, so the finisher is unchanged. The
/// module is the qualified lab module's bytes. The drafter bank is
/// unmeasured.
///
/// Verify-32 and Verify-16 serve the same queue. TP4, 300 W, 1800 MHz,
/// 2026-10-23, captured full MoE graphs (OV2 strip2-verify), over the
/// DSpark C8 K3 and C16 K1 captures at 32 rows and the C8 capture's 16-row
/// prefixes: all 60 cases per GPU bitwise, memcheck and racecheck clean on
/// the down small queues. Against serving it saves 0.41 (C8 K3) and 0.55 ms
/// (C16 K1) per Verify-32 step and 0.46 ms per Verify-16 step, which also
/// drops the direct down small and wide queues; 59 of 60 timed layers
/// faster, no slow sample, A/A within 0.04 ms. Their router, gate and
/// finisher launches equal the Decode buckets', so the down queue is the
/// only difference either way.
fn strips2(entry: &Entry, _phase: ExecutionPhase, rows: u32) -> bool {
    entry.experts() == 384 && matches!(rows, 16 | 32)
}

fn prepare(
    entry: &Entry,
    rows: u32,
    phase: ExecutionPhase,
    scaling: u32,
    fused_router: bool,
    r2: bool,
    tiles: &std::collections::BTreeMap<String, kernel_build::EntrySpec>,
) -> Result<Prepared, Refused> {
    let wide = fused_router
        && entry.experts() == 384
        && phase == ExecutionPhase::Verify
        && (33..=64).contains(&rows);
    let absorb_queues = absorbs(entry, phase, rows) || wide;
    let strip2_down = strips2(entry, phase, rows) || wide;
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
        if ((rows <= 8 || absorb_queues) && matches!(s, Stage::GateWide | Stage::DownWide))
            || (strip2_down && s == Stage::DownWide)
        {
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
        // Verify-6 (DSpark C1 with five drafts; 6-row prefixes of the same
        // captures, verify6-r1): over r2, the edge and the all-loads finisher
        // the resident8 gate saves 0.33 ms per 40-layer step, 20 of 20
        // layers, no slow sample, every case bitwise. Verify-16 (verify16-r1,
        // 16-row prefixes): 0.80 ms over the same, 20 of 20 layers.
        // Decode-16 (decode16-r1, 16 AR sequences): 0.85 ms, 20 of 20.
        if stage == Some(Stage::GateSmall)
            && entry.experts() == 384
            && (matches!(rows, 16 | 32)
                || (phase == ExecutionPhase::Verify && matches!(rows, 4 | 6)))
        {
            // The positional ABI is identical. Retain the floor binding's
            // view names for scales, sorted rows and padded rows.
            spec.symbol = kernels_sm120::moe::chained::streamed_resident::symbol(&entry.gate);
            plan = kernels_sm120::moe::chained::streamed_resident::plan(&entry.gate, rows)?;
            plan.workspace_bytes = entry.workspace_bytes(rows)?;
        }
        // TP4, 250 W, 2026-10-21, captured row prefixes over the r2 finish
        // and edge (finish-loads-r1): the gate finisher that issues every
        // partial load before its first add passed all 80 cases per GPU
        // bitwise, memcheck clean. It saves 1.1 us per layer at backbone
        // Decode-1 and 1.8 us at Verify-4 (sample means; faster in every
        // timed layer). Same parameters, grid and PDL launch; the module is
        // the qualified lab module's bytes. Over round 10 serving at backbone
        // Decode-8 (finish-backbone-r1) it saves 1.75 us per layer, faster in
        // all 20 timed layers with no slow samples. At 32 rows (five CTAs per
        // SM; 960 CTAs exceed one wave) it is within noise and the served
        // finisher stays. At Verify-6 (verify6-r1) it saves 0.07 ms per
        // 40-layer step over r2 with the edge; the three together save
        // 0.72 ms, 20 of 20 layers. At Verify-16 the fused router with r2,
        // the edge and this finisher save 0.43 ms against the unfused
        // router (verify16-r1), and 0.48 ms at Decode-16 (decode16-r1).
        // Other rows are unmeasured.
        if stage == Some(Stage::GateFinish)
            && entry.experts() == 384
            && matches!(
                (phase, rows),
                (ExecutionPhase::Decode, 1 | 8 | 16) | (ExecutionPhase::Verify, 4 | 6 | 16)
            )
        {
            spec = bandwidth::finish_loads_entry(entry);
        }
        if let Some(s @ (Stage::GateSmall | Stage::DownSmall)) = stage
            && absorb_queues
        {
            // The small queue's positional ABI and view names.
            spec = absorb::entry(entry, s);
            plan = absorb::plan(entry, s, rows)?;
        }
        if stage == Some(Stage::DownSmall) && strip2_down {
            // The absorbing down queue's positional ABI and view names.
            spec = strip2::entry(entry);
            plan = strip2::plan(entry, rows)?;
        }
        if fused_router && matches!(stage, None | Some(Stage::Metadata)) {
            continue;
        }
        // The TP4 backbone gate/up bank in row tiles: each gate queue
        // launches its row-tiled build, which keeps the entry's parameters
        // and launch (`moe_row_tiles`), and no queue reads the tiles as rows.
        if let Some(tiled) = tiles.get(&spec.symbol) {
            spec.symbol.clone_from(&tiled.symbol);
        } else if !tiles.is_empty() && matches!(stage, Some(Stage::GateSmall | Stage::GateWide)) {
            return Err(Refused::Rule(format!(
                "{} would read the row-tiled backbone gate/up bank as rows",
                spec.symbol
            )));
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
    let mut prepared = Prepared::new(info, calls, ticket_offset);
    if fused_router && rows > 32 {
        let layout = kernels_sm120::moe::chained::router_finish64::Layout::new(entry, rows)?;
        prepared = prepared.with_router_planes(crate::ops::moe::direct::RouterPlanes {
            domain_bytes: kernels_sm120::moe::chained::router_finish64::DOMAIN_BYTES,
            partial_offset: layout.partials,
            partial_bytes: layout.partial_bytes,
        });
    }
    Ok(prepared)
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
                                        || (set.count == 384 && (2..=32).contains(&rows))
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
                        if call.launch.symbol.contains("_gate_finish") {
                            assert_eq!(
                                call.launch.symbol.ends_with("_gate_finish_loads_v1"),
                                set.count == 384
                                    && matches!(
                                        (phase, rows),
                                        (ExecutionPhase::Decode, 1 | 8 | 16)
                                            | (ExecutionPhase::Verify, 4 | 6 | 16)
                                    ),
                                "{phase:?} rows={rows}"
                            );
                        }
                        let absorbed = set.count == 384
                            && matches!(
                                (phase, rows),
                                (ExecutionPhase::Decode, 32) | (ExecutionPhase::Verify, 32)
                            );
                        let strips =
                            set.count == 384 && matches!(rows, 16 | 32) && point == Point::Down;
                        assert!(!(absorbed && call.launch.symbol.contains("_wide_")));
                        assert!(!(strips && call.launch.symbol.contains("_wide_")));
                        // TP4 stores the backbone gate/up bank in quad-lane
                        // row tiles: its gate queues launch their quad-lane
                        // builds.
                        let queue = call.launch.symbol.contains("_small_")
                            || call.launch.symbol.contains("_wide_");
                        assert_eq!(
                            queue && call.launch.symbol.contains("_lanes"),
                            queue && set.count == 384 && point == Point::Gate,
                            "{}",
                            call.launch.symbol
                        );
                        if call.launch.symbol.contains("_small_") {
                            assert_eq!(
                                call.launch.symbol.contains("_moe_absorb_"),
                                absorbed && !strips,
                                "{phase:?} rows={rows}"
                            );
                            assert_eq!(
                                call.launch.symbol.contains("_moe_strip2_"),
                                strips,
                                "{phase:?} rows={rows}"
                            );
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
                                call.launch.symbol.contains("_small_resident8_"),
                                set.count == 384
                                    && (rows == 16
                                        || (rows == 32 && !absorbed)
                                        || (phase == ExecutionPhase::Verify
                                            && matches!(rows, 4 | 6)))
                                    && point == Point::Gate
                            );
                            let items = set.count.min(rows * set.top_k)
                                * match point {
                                    Point::Gate => 9 * 40,
                                    _ if strips => 40 * 5,
                                    _ => 80 * 5,
                                };
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

    /// At 16 and 32 rows, Decode and Verify alike, the backbone's down
    /// queues are one two-strip launch with the positional ABI of the
    /// queue it replaced (absorbing at 32, direct at 16), and every launch
    /// of the two buckets is the same.
    #[test]
    fn two_strip_down_queue_serves_decode_and_verify_at_16_and_32_rows() {
        let shape = crate::shape::Shape::new(&FLASH, Ranks::TP4).unwrap();
        let set = model_deepseek41_contract::moe::expert_sets(&FLASH)
            .into_iter()
            .find(|s| s.count == 384)
            .unwrap();
        let entry = super::super::moe_direct_kernels::entries(&FLASH)
            .unwrap()
            .into_iter()
            .find(|e| e.experts() == 384)
            .unwrap();
        assert_eq!(
            strip2::entry(&entry).params,
            bandwidth::entry(&entry, Stage::DownSmall, bandwidth::Loads::Direct).params
        );
        let ops = prepare_set(&shape, set).unwrap();
        let launches = |phase, rows, point| {
            let (_, _, op) = ops
                .iter()
                .find(|(p, n, _)| *p == phase && *n == rows)
                .unwrap();
            let workspace = DeviceBufferSpanV1::new(
                BundleDeviceBufferHandleV1 { raw: 1 },
                0,
                op.info.workspace,
            );
            op.launch(point, workspace, |_| {
                Ok(DeviceBufferSpanV1::new(
                    BundleDeviceBufferHandleV1 { raw: 2 },
                    0,
                    1 << 32,
                ))
            })
            .unwrap()
        };
        let spans = |launch: &bundle_runtime::pipeline::PreparedLaunch| {
            launch
                .arguments
                .0
                .iter()
                .filter_map(|arg| match arg {
                    Argument::DeviceBuffer(id) => Some(launch.bindings.get(*id).unwrap()),
                    _ => None,
                })
                .collect::<Vec<_>>()
        };
        for rows in [16, 32] {
            // The previous Verify down queues: the absorbing small queue at
            // 32 rows, the direct small queue (and a wide one) at 16. The
            // two-strip queue keeps their positional ABI and spans.
            let previous = if rows == 32 {
                absorb::entry(&entry, Stage::DownSmall)
            } else {
                bandwidth::entry(&entry, Stage::DownSmall, bandwidth::Loads::Direct)
            };
            assert_eq!(strip2::entry(&entry).params, previous.params);
            let down = launches(ExecutionPhase::Verify, rows, Point::Down);
            assert_eq!(down.len(), 1, "rows={rows}");
            assert_eq!(down[0].launch.symbol, strip2::symbol(&entry));
            assert_eq!(
                down[0].launch.plan.grid,
                strip2::plan(&entry, rows).unwrap().grid
            );
            assert_eq!(down[0].launch.plan.grid, [1504, 1, 1]);
            for point in [
                Point::RouterPrepare,
                Point::Route,
                Point::Gate,
                Point::Down,
                Point::Sum,
            ] {
                let decode = launches(ExecutionPhase::Decode, rows, point);
                let verify = launches(ExecutionPhase::Verify, rows, point);
                assert_eq!(decode.len(), verify.len(), "{point:?} rows={rows}");
                for (a, b) in decode.iter().zip(&verify) {
                    assert_eq!(a.launch.symbol, b.launch.symbol, "{point:?} rows={rows}");
                    assert_eq!(a.launch.plan.grid, b.launch.plan.grid);
                    assert_eq!(a.launch.arguments, b.launch.arguments);
                    assert_eq!(spans(&a.launch), spans(&b.launch));
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
