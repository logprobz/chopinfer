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
            for width in [1, 2, 4] {
                assert!(direct.inplace_scales(true, width));
                assert!(direct.inplace_scales(false, width));
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
