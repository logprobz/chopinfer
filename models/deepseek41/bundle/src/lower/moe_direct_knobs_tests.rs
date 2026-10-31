use super::*;
use crate::lower::{KvExtent, tests::buffers};
use crate::sm120::moe_direct_knobs::{CUBIN, WIDE_CUBIN};
use crate::{ops::Ops, shape::Shape};
use bundle_runtime::geometry::{Items, Ranks};
use bundle_runtime::state::Extent;
use bundle_runtime::upload::{DeviceInputs, InputLayout, PhaseArena};
use kernel_core::contract::Argument;
use model_deepseek41_contract::execution_options::MoeDirect;
use model_deepseek41_contract::{geometry::FLASH, state::KvStorage};
use runtime_bundle_abi::{BundleDeviceBufferHandleV1, DeviceBufferSpanV1};

fn argument(call: &Lowered, index: usize) -> DeviceBufferSpanV1 {
    let Argument::DeviceBuffer(handle) = call.launch.arguments.0[index] else {
        panic!("expected bound pointer {index}");
    };
    call.launch.bindings.get(handle).unwrap()
}

/// With `moe_direct_gate_tickets`, every absorbing step's gate stage is the
/// one gate queue: it binds the gate finisher's views (route weights,
/// intermediate, codes, their scales, status) and tickets after its own
/// operands, then the instance's strip tickets; no gate finisher runs.
/// Without the strip tickets allocated, the stage refuses.
#[test]
fn the_strip_finishing_gate_binds_the_finisher_views_and_the_strip_tickets() {
    let shape = Shape::new(&FLASH, Ranks::TP4)
        .unwrap()
        .with_verify_rows(192)
        .with_wide_verify(true)
        .with_moe_knobs(MoeDirect {
            gate_tickets: true,
            retire: true,
            ..MoeDirect::default()
        });
    let kernels = crate::Kernels::new(&shape).unwrap();
    let ops = Ops::prepare_set(
        &kernels,
        &shape,
        KvStorage::Native,
        model_deepseek41_contract::moe::expert_sets(&FLASH)[0],
    );
    let moe = ops.moe.as_ref().unwrap();
    assert_eq!(moe.strip_ticket_bytes(), Some(384 * 9 * 4));
    let rank = 0;
    let mut b = buffers(&shape, rank);
    b.workspace.length = ops.workspace();
    b.sorted_rows.length = moe.sorted_rows_bytes();
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
    let tickets = DeviceBufferSpanV1::new(
        BundleDeviceBufferHandleV1::from_parts(20000, rank + 1),
        0,
        384 * 9 * 4,
    );
    b.moe_strip_tickets = Some(tickets);
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
    let mut unallocated = b.clone();
    unallocated.moe_strip_tickets = None;
    let peers = [b.clone()];
    for rows in [32, 48, 64, 65, 128, 192] {
        let phase = Phase::Verify {
            rows,
            items: rows.div_ceil(6),
        };
        let lowering = |buffers| Lowering {
            prefill_row_origin: None,
            shape: &shape,
            ops: &ops,
            buffers,
            peers: &peers,
            inputs: &inputs,
            rank,
            kv: KvExtent {
                storage: KvStorage::Native,
                pages: 1,
                extent: Extent {
                    served: 1,
                    bucket: rows.min(32),
                    verify_rows: rows,
                },
                sequences: 1,
                max_pages: 1,
            },
            head_rows: 1,
            prefill_hc_fusion: true,
        };
        let gate = lowering(&b)
            .moe_direct_stage(phase, Stage::ExpertsGateUp(0))
            .unwrap()
            .unwrap();
        let queues: Vec<_> = gate
            .iter()
            .filter(|c| c.launch.symbol.contains("_gate_small_tickets"))
            .collect();
        assert_eq!(queues.len(), 1, "{rows} rows");
        assert!(queues[0].launch.symbol.contains("_ticketsr_"));
        assert!(
            gate.iter()
                .all(|c| !c.launch.symbol.contains("_gate_finish"))
        );
        let queue = queues[0];
        let s = &b.scratch;
        for (index, view) in [
            (9, s.expert_weights),
            (10, s.expert_intermediate),
            (11, s.intermediate_codes),
            (12, s.intermediate_scales),
            (13, s.row_status),
        ] {
            let bound = argument(queue, index);
            assert_eq!((bound.buffer, bound.offset), (view.buffer, view.offset));
        }
        let bound = argument(queue, 14);
        assert_eq!(bound.buffer, b.workspace.buffer);
        assert_eq!(argument(queue, 15), tickets);
        assert_eq!(queue.launch.arguments.0.len(), 16);
        let refused = lowering(&unallocated).moe_direct_stage(phase, Stage::ExpertsGateUp(0));
        let Err(error) = refused else {
            panic!("the gate stage bound unallocated strip tickets");
        };
        assert!(error.contains("strip tickets"), "{error}");
        // The down queue and both finishers' other stages still bind.
        for stage in [Stage::ExpertsDown(0), Stage::RoutedSum(0)] {
            let calls = lowering(&unallocated)
                .moe_direct_stage(phase, stage)
                .unwrap();
            assert!(!calls.unwrap().is_empty());
        }
    }
}

/// An instance loads a knob module exactly where a `moe_direct_*`
/// option other than loads alone meets a bucket of the absorbing path:
/// the 64-row module on Flash TP4, the wide one with the wide verify
/// buckets as well.
#[test]
fn instances_load_the_knob_modules_only_under_an_option() {
    let options = [
        MoeDirect::default(),
        MoeDirect {
            gate_tickets: true,
            ..MoeDirect::default()
        },
        MoeDirect {
            stream: true,
            keep: true,
            retire: true,
            loads: true,
            ..MoeDirect::default()
        },
        MoeDirect {
            loads: true,
            ..MoeDirect::default()
        },
    ];
    for ranks in [Ranks::TP1, Ranks::TP2, Ranks::TP4] {
        let shape = Shape::new(&FLASH, ranks).unwrap();
        for rows in [32, 64, 65, 128, 192] {
            for wide in [false, true] {
                for option in options {
                    let shape = shape
                        .with_verify_rows(rows)
                        .with_wide_verify(wide)
                        .with_moe_knobs(option);
                    let on = (option.gate_tickets || option.stream) && ranks == Ranks::TP4;
                    assert_eq!(crate::Kernels::loads(CUBIN, &shape), on);
                    assert_eq!(
                        crate::Kernels::loads(WIDE_CUBIN, &shape),
                        on && wide && rows > 64
                    );
                }
            }
        }
    }
}
