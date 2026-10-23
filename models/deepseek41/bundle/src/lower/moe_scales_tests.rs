//! A routed scale bank the rank image stores encoded expands into the raw
//! bank before its first expert launch, on every path that binds it, and
//! no other launch binds the encoded bank.
use crate::buffers::RankBuffers;
use crate::lower::{KvExtent, Lowering, layer_view, tests::buffers};
use crate::ops::{Lowered, Ops};
use crate::program::{Phase, Stage};
use crate::shape::Shape;
use bundle_runtime::geometry::{Items, Ranks};
use bundle_runtime::pipeline::PreparedLaunch;
use bundle_runtime::state::Extent;
use bundle_runtime::upload::{DeviceInputs, InputLayout, PhaseArena};
use kernel_core::contract::Argument;
use model_deepseek41_contract::moe::{self, Experts};
use model_deepseek41_contract::{geometry::FLASH, state::KvStorage};
use runtime_bundle_abi::{BundleDeviceBufferHandleV1, DeviceBufferSpanV1};

const VIEWS: [&str; 14] = [
    "ffn.gate.weight",
    "ffn.gate.weight.f32_domain",
    "ffn.gate.bias",
    "ffn.gate.bias_vl",
    "ffn.experts.gate_up.weight",
    "ffn.experts.gate_up.scale",
    "ffn.experts.down.weight",
    "ffn.experts.down.scale",
    "ffn.shared_experts.w1.weight",
    "ffn.shared_experts.w1.scale",
    "ffn.shared_experts.w3.weight",
    "ffn.shared_experts.w3.scale",
    "ffn.shared_experts.w2.weight",
    "ffn.shared_experts.w2.scale",
];

/// The expanded bank's buffer, apart from every other span.
const EXPANDED: u32 = 20000;

fn span(id: u32, rank: u32, bytes: u64) -> DeviceBufferSpanV1 {
    DeviceBufferSpanV1::new(
        BundleDeviceBufferHandleV1::from_parts(id, rank + 1),
        0,
        bytes,
    )
}

/// Rank `rank`'s buffers with layer 0's routed MoE and shared expert views,
/// every bank raw.
fn rank_buffers(shape: &Shape, ops: &Ops, rank: u32) -> RankBuffers {
    let mut b = buffers(shape, rank);
    b.workspace.length = ops.workspace();
    b.shared_workspace.length = ops.shared_workspace();
    b.sorted_rows.length = ops.moe.as_ref().unwrap().sorted_rows_bytes();
    let mut next = 10000;
    let moe = ops.moe.as_ref().unwrap();
    for name in VIEWS {
        next += 1;
        // A routed scale bank holds its raw bytes, which also bound its
        // encoding; the other views are long enough for any op (the TP1
        // gate/up codes take 4.5 GB).
        let bytes = match name {
            "ffn.experts.gate_up.scale" => moe.scale_bank(true).bytes(),
            "ffn.experts.down.scale" => moe.scale_bank(false).bytes(),
            _ => 1 << 34,
        };
        b.views.insert(layer_view(0, name), span(next, rank, bytes));
    }
    let weights: Vec<_> = b
        .views
        .keys()
        .filter(|name| name.ends_with(".weight"))
        .cloned()
        .collect();
    for weight in weights {
        for (paired, _, sizes) in ops.dense.as_ref().unwrap().stream_weight_ops(&weight) {
            for (role, bytes) in ["packed0", "packed1", "state", "workspace"]
                .into_iter()
                .zip(sizes)
                .filter(|(_, bytes)| *bytes > 0)
            {
                next += 1;
                b.views.insert(
                    crate::ops::fold_stream::name(&weight, paired, role),
                    span(next, rank, bytes),
                );
            }
        }
    }
    b
}

/// `raw` with layer 0's gate/up bank stored encoded at width 1 and its
/// down bank at width 4, and the expanded bank allocated.
fn encoded(shape: &Shape, raw: &RankBuffers, rank: u32) -> RankBuffers {
    let mut b = raw.clone();
    let bytes = crate::buffers::expert_scale_bytes(shape).unwrap();
    b.scratch.expert_scales = Some(span(EXPANDED, rank, bytes));
    for (bank, width) in [("gate_up", 1), ("down", 4)] {
        b.encoded_scales
            .insert(layer_view(0, &format!("ffn.experts.{bank}.scale")), width);
    }
    b
}

fn lower(
    shape: &Shape,
    ops: &Ops,
    b: &RankBuffers,
    rank: u32,
    phase: Phase,
    stages: &[Stage],
) -> Result<Vec<(Stage, Vec<Lowered>)>, String> {
    let live = span(999, rank, 4);
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
    let lowering = Lowering {
        prefill_row_origin: None,
        shape,
        ops,
        buffers: b,
        peers: &peers,
        inputs: &inputs,
        rank,
        kv: KvExtent {
            storage: KvStorage::Native,
            pages: 1,
            extent: Extent {
                served: 1,
                bucket: phase.rows().min(32),
                verify_rows: phase.rows(),
            },
            sequences: 1,
            max_pages: 1,
        },
        head_rows: 1,
        prefill_hc_fusion: true,
    };
    stages
        .iter()
        .map(|&stage| {
            let launches = lowering.stage(phase, stage)?;
            Ok((stage, launches.ok_or("an unlowered stage")?))
        })
        .collect()
}

fn bound(launch: &PreparedLaunch) -> Vec<Option<DeviceBufferSpanV1>> {
    launch
        .arguments
        .0
        .iter()
        .map(|argument| match argument {
            Argument::DeviceBuffer(handle) => Some(launch.bindings.get(*handle).unwrap()),
            _ => None,
        })
        .collect()
}

/// Every phase kind, the direct and grouped paths at TP4 and the grouped
/// path at TP1 and TP2: each projection's launches are its raw lowering's
/// after one expansion, the raw scale bank rebound to the expanded one.
#[test]
fn encoded_scale_banks_expand_before_their_experts() {
    let decode = [1, 2, 4, 8, 16, 32].map(Phase::Decode);
    let verify = [(4, 1), (32, 6), (192, 32)].map(|(rows, items)| Phase::Verify { rows, items });
    let prefill = [1, 512, 8192].map(|rows| Phase::Prefill { rows });
    for ranks in [Ranks::TP1, Ranks::TP2, Ranks::TP4] {
        let shape = Shape::new(&FLASH, ranks).unwrap();
        let kernels = crate::Kernels::new(&shape).unwrap();
        let set = moe::expert_sets(&FLASH)[0];
        let ops = Ops::prepare_set(&kernels, &shape, KvStorage::Native, set);
        let moe_ops = ops.moe.as_ref().unwrap();
        let mut direct = 0;
        for rank in [0, ranks.get() - 1] {
            let raw = rank_buffers(&shape, &ops, rank);
            let encoded = encoded(&shape, &raw, rank);
            let expanded = encoded.scratch.expert_scales.unwrap();
            let scale_view =
                |bank: &str| raw.views[&layer_view(0, &format!("ffn.experts.{bank}.scale"))];
            for phase in decode.iter().chain(&verify).chain(&prefill).copied() {
                let stages = if phase.prefill() {
                    vec![
                        Stage::Route(0),
                        Stage::ExpertsGateUp(0),
                        Stage::ExpertsDown(0),
                    ]
                } else {
                    crate::program::moe(0)
                };
                let at = format!("{ranks:?} rank {rank} {phase:?}");
                let before = lower(&shape, &ops, &raw, rank, phase, &stages)
                    .unwrap_or_else(|e| panic!("{at}: {e}"));
                let after = lower(&shape, &ops, &encoded, rank, phase, &stages)
                    .unwrap_or_else(|e| panic!("{at}: {e}"));
                let is_direct = moe_ops.direct(phase).is_some();
                direct += usize::from(is_direct);
                // Every route op lists the experts with routes, so no
                // expansion writes every expert's scales.
                let listed = is_direct || moe_ops.lists_active(phase, rank);
                assert!(listed, "{at}");
                for ((stage, old), (_, new)) in before.iter().zip(&after) {
                    let (bank, width) = match stage {
                        Stage::ExpertsGateUp(_) => ("gate_up", 1),
                        Stage::ExpertsDown(_) => ("down", 4),
                        _ => {
                            assert_eq!(old.len(), new.len(), "{at} {stage:?}");
                            for (a, b) in old.iter().zip(new) {
                                assert_eq!(a.launch.symbol, b.launch.symbol, "{at}");
                                assert_eq!(bound(&a.launch), bound(&b.launch), "{at}");
                            }
                            continue;
                        }
                    };
                    let shape_of = moe_ops.scale_bank(bank == "gate_up");
                    // The expansion first: the encoded bank into the
                    // expanded one, a PDL launch like the rest.
                    let expand = &new[0].launch;
                    assert_eq!(
                        expand.symbol,
                        kernels_sm120::moe::scale_expand::symbol(width).unwrap(),
                        "{at} {stage:?}"
                    );
                    assert!(expand.plan.pdl, "{at}");
                    let spans = bound(expand);
                    assert_eq!(spans[0], Some(scale_view(bank)), "{at}");
                    assert_eq!(
                        spans[1],
                        listed.then_some(encoded.scratch.active_experts),
                        "{at}"
                    );
                    assert_eq!(spans[2], Some(expanded), "{at}");
                    assert_eq!(
                        expand.arguments.0[3..],
                        [
                            Argument::U32(shape_of.rows),
                            Argument::U32(shape_of.groups),
                            Argument::U32(shape_of.projections),
                            Argument::U32(shape_of.experts),
                        ],
                        "{at}"
                    );
                    let slots = if listed {
                        shape_of.experts.min(phase.rows() * set.top_k)
                    } else {
                        shape_of.experts
                    };
                    assert_eq!(expand.plan.grid[2], slots, "{at}");
                    // Then the raw lowering's launches, the scales rebound.
                    assert_eq!(new.len(), old.len() + 1, "{at} {stage:?}");
                    let raw_scales = scale_view(bank);
                    let mut rebound = 0;
                    for (a, b) in old.iter().zip(&new[1..]) {
                        assert_eq!(a.launch.symbol, b.launch.symbol, "{at}");
                        assert_eq!(a.launch.plan, b.launch.plan, "{at}");
                        assert_eq!(a.launch.rows, b.launch.rows, "{at}");
                        for (x, y) in bound(&a.launch).into_iter().zip(bound(&b.launch)) {
                            match x {
                                Some(x) if x.buffer == raw_scales.buffer => {
                                    rebound += 1;
                                    assert!(x.offset + x.length <= shape_of.bytes(), "{at}");
                                    assert_eq!(
                                        y,
                                        Some(DeviceBufferSpanV1::new(
                                            expanded.buffer,
                                            expanded.offset + x.offset,
                                            x.length,
                                        )),
                                        "{at}"
                                    );
                                }
                                _ => assert_eq!(x, y, "{at}"),
                            }
                        }
                    }
                    assert!(rebound > 0, "{at} {stage:?}");
                }
                // No launch but an expansion binds an encoded bank.
                for (_, launches) in &after {
                    for lowered in launches {
                        if lowered.launch.symbol.starts_with("chop_moe_scale_expand_") {
                            continue;
                        }
                        for bank in ["gate_up", "down"] {
                            assert!(
                                bound(&lowered.launch)
                                    .iter()
                                    .flatten()
                                    .all(|s| s.buffer != scale_view(bank).buffer),
                                "{at}: {} binds the encoded {bank} bank",
                                lowered.launch.symbol
                            );
                        }
                    }
                }
                if !phase.prefill() {
                    // The expansion stays on the routed branch, apart from
                    // the shared expert beside it.
                    let stages: Vec<_> = after
                        .iter()
                        .flat_map(|(s, ops)| std::iter::repeat_n(*s, ops.len()))
                        .collect();
                    let launches: Vec<_> = after
                        .into_iter()
                        .flat_map(|(_, ops)| ops.into_iter().map(|op| op.launch))
                        .collect();
                    let s = &encoded.scratch;
                    let readonly = [s.normed_codes, s.normed_scales, span(999, rank, 4)];
                    crate::shared_overlap::checked_units(&stages, &launches, &readonly)
                        .unwrap_or_else(|e| panic!("{at}: {e}"));
                }
            }
        }
        assert_eq!(direct > 0, ranks == Ranks::TP4, "{ranks:?}");
    }
}

/// The measured one-row floor binds the drafter's banks, which the plan
/// stores raw; an encoded bank there is refused, not read as raw scales.
#[test]
fn the_floor_refuses_an_encoded_scale_bank() {
    let shape = Shape::new(&FLASH, Ranks::TP4).unwrap();
    let kernels = crate::Kernels::new(&shape).unwrap();
    let set = Experts {
        count: 128,
        top_k: 3,
    };
    let mut ops = Ops::prepare_set(&kernels, &shape, KvStorage::Native, set);
    ops.moe.as_mut().unwrap().retain_reference_in_test();
    assert!(ops.moe.as_ref().unwrap().floor(Phase::Decode(1)).is_some());
    let raw = rank_buffers(&shape, &ops, 0);
    let stages = crate::program::moe(0);
    lower(&shape, &ops, &raw, 0, Phase::Decode(1), &stages).unwrap();
    let error = lower(
        &shape,
        &ops,
        &encoded(&shape, &raw, 0),
        0,
        Phase::Decode(1),
        &stages,
    )
    .err()
    .expect("an encoded bank is refused");
    assert!(error.contains("is stored encoded"), "{error}");
}

/// Without its expanded bank, an encoded bank is refused by name.
#[test]
fn an_encoded_bank_needs_its_expanded_bank() {
    let shape = Shape::new(&FLASH, Ranks::TP1).unwrap();
    let kernels = crate::Kernels::new(&shape).unwrap();
    let set = moe::expert_sets(&FLASH)[0];
    let ops = Ops::prepare_set(&kernels, &shape, KvStorage::Native, set);
    let raw = rank_buffers(&shape, &ops, 0);
    let mut b = encoded(&shape, &raw, 0);
    b.scratch.expert_scales = None;
    let error = lower(
        &shape,
        &ops,
        &b,
        0,
        Phase::Decode(1),
        &[Stage::ExpertsGateUp(0)],
    )
    .err()
    .expect("refused");
    assert!(error.contains("no raw bank is allocated"), "{error}");
}
