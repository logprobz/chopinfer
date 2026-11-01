use super::*;
use crate::lower::{KvExtent, tests::buffers};
use crate::{ops::Ops, shape::Shape};
use bundle_runtime::geometry::{Items, Ranks};
use bundle_runtime::state::Extent;
use bundle_runtime::upload::{DeviceInputs, InputLayout, PhaseArena};
use kernel_core::contract::Argument;
use model_deepseek41_contract::{geometry::FLASH, state::KvStorage};
use runtime_bundle_abi::{BundleDeviceBufferHandleV1, DeviceBufferSpanV1};

fn argument(call: &Lowered, index: usize) -> DeviceBufferSpanV1 {
    let Argument::DeviceBuffer(handle) = call.launch.arguments.0[index] else {
        panic!("expected bound pointer {index}");
    };
    call.launch.bindings.get(handle).unwrap()
}

#[test]
fn every_wide_router_binds_the_named_producer_and_disjoint_metadata_tail() {
    let _metadata = kernel_core::route::Metadata::enable();
    let shape = Shape::new(&FLASH, Ranks::TP4).unwrap().with_verify_rows(64);
    let kernels = crate::Kernels::new(&shape).unwrap();
    let ops = Ops::prepare_set(
        &kernels,
        &shape,
        KvStorage::Native,
        model_deepseek41_contract::moe::expert_sets(&FLASH)[0],
    );
    for rank in 0..4 {
        let mut b = buffers(&shape, rank);
        b.workspace.length = ops.workspace();
        b.sorted_rows.length = ops.moe.as_ref().unwrap().sorted_rows_bytes();
        for (index, name) in [
            "ffn.gate.weight",
            "ffn.gate.weight.f32_domain",
            "ffn.gate.bias",
            "ffn.gate.bias_vl",
            "ffn.experts.gate_up.weight",
            "ffn.experts.gate_up.scale",
            "ffn.experts.down.weight",
            "ffn.experts.down.scale",
        ]
        .into_iter()
        .enumerate()
        {
            b.views.insert(
                layer_view(0, name),
                DeviceBufferSpanV1::new(
                    BundleDeviceBufferHandleV1::from_parts(10000 + index as u32, rank + 1),
                    0,
                    1 << 32,
                ),
            );
        }
        let live =
            DeviceBufferSpanV1::new(BundleDeviceBufferHandleV1::from_parts(999, rank + 1), 0, 4);
        let inputs = DeviceInputs {
            live_tokens: live,
            ..DeviceInputs::within(
                PhaseArena {
                    arena: live,
                    block_tables: live,
                },
                &InputLayout::decode_within(Items::new(1).unwrap()),
            )
        };
        let peers = [b.clone()];
        for rows in 33..=64 {
            let phase = Phase::Verify {
                rows,
                items: rows.div_ceil(6),
            };
            let lowering = Lowering {
                prefill_row_origin: None,
                shape: &shape,
                ops: &ops,
                buffers: &b,
                peers: &peers,
                inputs: &inputs,
                rank,
                kv: KvExtent {
                    storage: KvStorage::Native,
                    pages: 1,
                    extent: Extent {
                        served: 1,
                        bucket: 16,
                        verify_rows: rows,
                    },
                    sequences: 1,
                    max_pages: 1,
                },
                head_rows: 1,
                prefill_hc_fusion: true,
            };
            let lower = |stage| {
                let mut calls = lowering.moe_direct_stage(phase, stage).unwrap().unwrap();
                lowering.tag_routes(phase, stage, &mut calls);
                calls
            };
            let projected = lower(Stage::Project(0, "moe.router"));
            assert_eq!(projected.len(), 2);
            for (call, family) in projected
                .iter()
                .zip(["moe_router_finish", "f32_projection"])
            {
                let tag = call
                    .launch
                    .bindings
                    .route()
                    .expect("selected wide router entry");
                assert_eq!(
                    (tag.class, tag.family),
                    (kernel_core::route::RouteClass::Fast, family)
                );
                assert!(tag.admitted.iter().any(|range| range.contains(&rows)));
            }
            assert!(projected[0].launch.symbol.ends_with("_prepare_v1"));
            assert!(
                projected[1]
                    .launch
                    .symbol
                    .ends_with("_spans_m64_compute_v1")
            );
            assert_eq!(projected[1].launch.arguments.0[5], Argument::U32(rows));
            let domain = argument(&projected[0], 3);
            let partials = argument(&projected[1], 3);
            assert_eq!(domain, argument(&projected[1], 2));
            assert_eq!(domain.length, 256);
            assert_eq!(partials.offset, domain.offset + 256);
            assert_eq!(partials.length, 61440 * u64::from(rows));
            let arrival = argument(&projected[0], 4);
            let route = lower(Stage::Route(0));
            assert_eq!(route.len(), 1);
            assert!(crate::shared_overlap::joins_before_gate(
                &route[0].launch.symbol
            ));
            assert_eq!(partials, argument(&route[0], 0));
            assert_eq!(arrival, argument(&route[0], 18));
            let scratch = argument(&route[0], 16);
            let tickets = argument(&route[0], 17);
            assert!(partials.offset + partials.length <= arrival.offset);
            assert!(arrival.offset + 256 <= scratch.offset);
            assert!(scratch.offset + scratch.length <= tickets.offset);
            assert!(tickets.offset + tickets.length <= b.workspace.offset + b.workspace.length);
            for plane in [domain, partials, arrival, scratch, tickets] {
                assert_eq!(plane.buffer, b.workspace.buffer);
                assert!(plane.offset.is_multiple_of(256));
            }
            for stage in [
                Stage::ExpertsGateUp(0),
                Stage::ExpertsDown(0),
                Stage::RoutedSum(0),
            ] {
                assert!(!lower(stage).is_empty());
            }
            let gate = lower(Stage::ExpertsGateUp(0));
            assert_eq!(gate.len(), 2, "wide gate at {rows} rows");
            assert!(gate[0].launch.symbol.contains("_absorb_lanes_"));
            let down = lower(Stage::ExpertsDown(0));
            assert_eq!(down.len(), 1, "wide down at {rows} rows");
            assert!(down[0].launch.symbol.contains("_moe_strip2_"));
            let direct = ops.moe.as_ref().unwrap().direct(phase).unwrap();
            // The knob entries read width-1 tile scales in place; wider
            // banks expand first.
            for width in [1, 2, 4] {
                assert_eq!(direct.inplace_scales(true, width), width == 1);
                assert_eq!(direct.inplace_scales(false, width), width == 1);
            }
        }
    }
}

#[test]
fn the_wide_router_family_loads_only_with_a_wide_tp4_budget() {
    let cubin = crate::sm120::moe_router64::CUBIN;
    for ranks in [Ranks::TP1, Ranks::TP2, Ranks::TP4] {
        let shape = Shape::new(&FLASH, ranks).unwrap();
        assert!(!crate::Kernels::loads(cubin, &shape));
        for rows in [32, 33, 48, 64] {
            assert_eq!(
                crate::Kernels::loads(cubin, &shape.with_verify_rows(rows)),
                ranks == Ranks::TP4 && rows > 32,
            );
        }
    }
    let specs = crate::sm120::moe_router64::cubins(&[FLASH]).unwrap();
    assert_eq!(specs.len(), 1);
    assert_eq!(specs[0].1.entries.len(), 2);
}

/// An instance loads the wide direct MoE module exactly where it prepares
/// a wide verify bucket: TP4 instances with the `wide_verify` option that
/// serve verify steps above 64 rows, each row count from 65 to its verify
/// rows.
#[test]
fn instances_load_the_wide_direct_family_only_with_a_wide_bucket() {
    let cubin = crate::sm120::moe_direct_wide::CUBIN;
    let set = model_deepseek41_contract::moe::expert_sets(&FLASH)[0];
    for ranks in [Ranks::TP1, Ranks::TP2, Ranks::TP4] {
        let shape = Shape::new(&FLASH, ranks).unwrap();
        for rows in [32, 64, 65, 128, 192] {
            let shape = shape.with_verify_rows(rows);
            assert!(!crate::Kernels::loads(cubin, &shape));
            assert!(crate::sm120::moe_direct::wide_rows(&shape, set).is_empty());
            let wide = shape.with_wide_verify(true);
            let enabled = ranks == Ranks::TP4 && rows > 64;
            assert_eq!(crate::Kernels::loads(cubin, &wide), enabled);
            assert_eq!(
                crate::sm120::moe_direct::wide_rows(&wide, set),
                if enabled {
                    (65..=rows).collect()
                } else {
                    Vec::new()
                }
            );
        }
    }
}

/// With the `wide_verify` option, verify steps of 65 to 192 rows through
/// the wide direct queues keep the router projection and routes of the
/// 8192-row expert chains launch for launch, then bind the pack, the
/// absorbing gate queue and its finisher, the two-strip down queue and the
/// down finisher at the package's verify rows, each pointer within its
/// view and every workspace plane within the workspace, which the wide
/// buckets do not grow.
#[test]
fn every_wide_verify_step_keeps_the_served_routes_and_binds_the_wide_queues() {
    let shape = Shape::new(&FLASH, Ranks::TP4)
        .unwrap()
        .with_verify_rows(192)
        .with_wide_verify(true);
    let kernels = crate::Kernels::new(&shape).unwrap();
    let set = model_deepseek41_contract::moe::expert_sets(&FLASH)[0];
    let wide = Ops::prepare_set(&kernels, &shape, KvStorage::Native, set);
    let mut reference = Ops::prepare_set(&kernels, &shape, KvStorage::Native, set);
    reference.moe.as_mut().unwrap().retain_reference_in_test();
    let largest = wide
        .moe
        .as_ref()
        .unwrap()
        .direct(Phase::Verify {
            rows: 192,
            items: 32,
        })
        .unwrap()
        .info
        .workspace;
    assert!(largest < reference.workspace());
    assert_eq!(wide.workspace(), reference.workspace());
    let rank = 1;
    let mut b = buffers(&shape, rank);
    b.workspace.length = wide.workspace();
    b.sorted_rows.length = wide.moe.as_ref().unwrap().sorted_rows_bytes();
    assert_eq!(
        b.sorted_rows.length,
        reference.moe.as_ref().unwrap().sorted_rows_bytes()
    );
    for (index, name) in [
        "ffn.gate.weight",
        "ffn.gate.weight.f32_domain",
        "ffn.gate.bias",
        "ffn.gate.bias_vl",
        "ffn.experts.gate_up.weight",
        "ffn.experts.gate_up.scale",
        "ffn.experts.down.weight",
        "ffn.experts.down.scale",
    ]
    .into_iter()
    .enumerate()
    {
        b.views.insert(
            layer_view(0, name),
            DeviceBufferSpanV1::new(
                BundleDeviceBufferHandleV1::from_parts(10000 + index as u32, rank + 1),
                0,
                1 << 32,
            ),
        );
    }
    let live = DeviceBufferSpanV1::new(BundleDeviceBufferHandleV1::from_parts(999, rank + 1), 0, 4);
    let inputs = DeviceInputs {
        live_tokens: live,
        ..DeviceInputs::within(
            PhaseArena {
                arena: live,
                block_tables: live,
            },
            &InputLayout::decode_within(Items::new(1).unwrap()),
        )
    };
    let peers = [b.clone()];
    let entry = crate::sm120::moe_direct_wide::entry(&FLASH).unwrap();
    for rows in 65..=192 {
        let phase = Phase::Verify {
            rows,
            items: rows.div_ceil(6),
        };
        assert!(reference.moe.as_ref().unwrap().direct(phase).is_none());
        let direct = wide.moe.as_ref().unwrap().direct(phase).unwrap();
        assert!(!direct.fuses_router() && !direct.has(crate::ops::moe::direct::Point::Route));
        for width in [1, 2, 4] {
            assert!(direct.inplace_scales(true, width));
            assert!(direct.inplace_scales(false, width));
        }
        let lower = |ops: &Ops, stage| {
            let lowering = Lowering {
                prefill_row_origin: None,
                shape: &shape,
                ops,
                buffers: &b,
                peers: &peers,
                inputs: &inputs,
                rank,
                kv: KvExtent {
                    storage: KvStorage::Native,
                    pages: 1,
                    extent: Extent {
                        served: 1,
                        bucket: 32,
                        verify_rows: rows,
                    },
                    sequences: 1,
                    max_pages: 1,
                },
                head_rows: 1,
                prefill_hc_fusion: true,
            };
            lowering.stage(phase, stage).unwrap().unwrap()
        };
        let described = |calls: &[Lowered]| {
            calls
                .iter()
                .map(|c| {
                    let arguments = c
                        .launch
                        .arguments
                        .0
                        .iter()
                        .map(|a| match a {
                            Argument::DeviceBuffer(h) => {
                                format!("{:?}", c.launch.bindings.get(*h).unwrap())
                            }
                            other => format!("{other:?}"),
                        })
                        .collect::<Vec<_>>();
                    (c.launch.symbol.clone(), c.launch.plan, arguments)
                })
                .collect::<Vec<_>>()
        };
        for stage in [Stage::Project(0, "moe.router"), Stage::Route(0)] {
            assert_eq!(
                described(&lower(&wide, stage)),
                described(&lower(&reference, stage))
            );
        }
        let gate = lower(&wide, Stage::ExpertsGateUp(0));
        let symbols = gate
            .iter()
            .map(|c| c.launch.symbol.as_str())
            .collect::<Vec<_>>();
        assert_eq!(
            symbols,
            [
                "chop_deepseek41_moe_wide_t192_e384_k6_pack_v1",
                "chop_deepseek41_moe_absorb_lanes_t192_e384_k6_gate_small_v1",
                "chop_deepseek41_moe_bandwidth_direct_t192_e384_k6_gate_finish_v1",
            ]
        );
        let down = lower(&wide, Stage::ExpertsDown(0));
        assert_eq!(down.len(), 1);
        assert_eq!(
            down[0].launch.symbol,
            "chop_deepseek41_moe_strip2_t192_e384_k6_down_small_v1"
        );
        let sum = lower(&wide, Stage::RoutedSum(0));
        assert_eq!(sum.len(), 1);
        assert_eq!(
            sum[0].launch.symbol,
            "chop_deepseek41_moe_floor_t192_e384_k6_down_finish_v1"
        );
        let scratch = argument(&gate[0], 3);
        let tickets = argument(&gate[0], 4);
        assert_eq!(
            scratch,
            DeviceBufferSpanV1::new(
                b.workspace.buffer,
                b.workspace.offset,
                entry.workspace_bytes(rows).unwrap()
            )
        );
        assert_eq!(tickets, argument(&gate[2], 7));
        assert_eq!(tickets.length, 768);
        assert_eq!(
            tickets.offset,
            scratch.offset + scratch.length.next_multiple_of(256)
        );
        assert!(tickets.offset + tickets.length <= b.workspace.offset + b.workspace.length);
        let prefix = |span: DeviceBufferSpanV1, length| DeviceBufferSpanV1 { length, ..span };
        assert_eq!(
            argument(&gate[0], 0),
            prefix(b.scratch.route_counts, 384 * 4)
        );
        assert_eq!(
            argument(&gate[0], 1),
            prefix(b.scratch.active_experts, 385 * 4)
        );
        let sorted = argument(&gate[1], 5);
        assert_eq!(
            sorted.length,
            u64::from((rows * 6 + 384 * 63).next_multiple_of(64)) * 4
        );
        assert_eq!(sorted.offset, b.sorted_rows.offset);
        for call in gate.iter().chain(&down).chain(&sum) {
            assert_eq!(call.launch.rows, rows);
            for (index, arg) in call.launch.arguments.0.iter().enumerate() {
                if let Argument::DeviceBuffer(h) = arg {
                    let span = call.launch.bindings.get(*h).unwrap();
                    if span.buffer == b.workspace.buffer {
                        assert!(span.offset >= b.workspace.offset, "{index}");
                        assert!(
                            span.offset + span.length <= b.workspace.offset + b.workspace.length
                        );
                    }
                }
            }
        }
    }
}
