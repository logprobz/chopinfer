//! The routed scale banks the rank image stores encoded expand into their
//! raw banks in one launch after the routes, on every path that binds
//! them, and no other launch binds an encoded bank.
use crate::buffers::RankBuffers;
use crate::lower::{KvExtent, Lowering, layer_view, tests::buffers};
use crate::ops::{Lowered, Ops};
use crate::program::{Phase, Stage};
use crate::shape::Shape;
use crate::sm120::test_symbols::scale_expand_pair_symbol;
use bundle_runtime::geometry::{Items, Ranks};
use bundle_runtime::pipeline::PreparedLaunch;
use bundle_runtime::state::Extent;
use bundle_runtime::upload::{DeviceInputs, InputLayout, PhaseArena};
use kernel_core::contract::Argument;
use model_deepseek41_contract::moe::{self, Experts};
use model_deepseek41_contract::weight_plan::{RoutedLayout, encoded_scale_layout};
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
            for (role, bytes) in ["state", "workspace"]
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
/// down bank at width 4, and the expanded banks allocated.
fn encoded(shape: &Shape, raw: &RankBuffers, rank: u32) -> RankBuffers {
    let mut b = raw.clone();
    let bytes = crate::buffers::expert_scale_bytes(shape).unwrap();
    b.scratch.expert_scales = Some(span(EXPANDED, rank, bytes));
    for (bank, width) in [("gate_up", 1), ("down", 4)] {
        b.encoded_scales.insert(
            layer_view(0, &format!("ffn.experts.{bank}.scale")),
            crate::buffers::ScaleBankEncoding {
                format: kernel_core::weight_transform::ScaleFormat::RowOffsetV1,
                width,
            },
        );
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
/// path at TP1 and TP2: the routes stage is its raw lowering's, then one
/// expansion of both banks; each projection's launches are its raw
/// lowering's, the raw scale banks rebound to the expanded ones.
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
        // The gate/up bank expands in the layout its consumers read.
        let pair_symbol = match moe_ops.gate_up_layout() {
            RoutedLayout::Rows => scale_expand_pair_symbol(false),
            RoutedLayout::Tiles | RoutedLayout::Lanes => scale_expand_pair_symbol(true),
        };
        let mut direct = 0;
        for rank in [0, ranks.get() - 1] {
            let raw = rank_buffers(&shape, &ops, rank);
            let encoded = encoded(&shape, &raw, rank);
            let expanded = encoded.scratch.expert_scales.unwrap();
            let banks = [true, false].map(|gate_up| moe_ops.scale_bank(gate_up));
            let raw_bank = |at: usize| {
                let bank = if at == 0 { "gate_up" } else { "down" };
                raw.views[&layer_view(0, &format!("ffn.experts.{bank}.scale"))]
            };
            // The gate/up bank, then the down bank at a 256-byte boundary.
            let expanded_bank = |at: usize| {
                let offset = if at == 0 {
                    0
                } else {
                    banks[0].bytes().next_multiple_of(256)
                };
                DeviceBufferSpanV1::new(
                    expanded.buffer,
                    expanded.offset + offset,
                    banks[at].bytes(),
                )
            };
            assert!(
                expanded_bank(1).offset + expanded_bank(1).length
                    <= expanded.offset + expanded.length
            );
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
                let mut rebound = [0, 0];
                for ((stage, old), (_, new)) in before.iter().zip(&after) {
                    let old_launches = if matches!(stage, Stage::Route(_)) {
                        // The routes, then the expansion of both banks: the
                        // encoded banks into the expanded ones, a PDL launch
                        // like the rest, after the routes list the experts.
                        assert_eq!(new.len(), old.len() + 1, "{at}");
                        let expand = &new.last().unwrap().launch;
                        assert_eq!(expand.symbol, pair_symbol, "{at}");
                        assert!(expand.plan.pdl, "{at}");
                        assert!(
                            new[..old.len()].iter().any(|op| bound(&op.launch)
                                .contains(&Some(encoded.scratch.active_experts))),
                            "{at}: no route launch lists the active experts"
                        );
                        let spans = bound(expand);
                        assert_eq!(
                            spans[..5],
                            [
                                Some(raw_bank(0)),
                                Some(raw_bank(1)),
                                Some(encoded.scratch.active_experts),
                                Some(expanded_bank(0)),
                                Some(expanded_bank(1)),
                            ],
                            "{at}"
                        );
                        // Each bank's shape and width; its tile rows
                        // follow from its groups.
                        for (index, (bank, width)) in banks.iter().zip([1, 4]).enumerate() {
                            let first = 5 + 5 * index;
                            assert_eq!(
                                expand.arguments.0[first..first + 4],
                                [bank.rows, bank.groups, bank.projections, width]
                                    .map(Argument::U32),
                                "{at}"
                            );
                        }
                        assert_eq!(
                            expand.arguments.0[15],
                            Argument::U32(banks[0].experts),
                            "{at}"
                        );
                        let slots = banks[0].experts.min(phase.rows() * set.top_k);
                        assert_eq!(expand.plan.grid[2], slots, "{at}");
                        &new[..old.len()]
                    } else {
                        &new[..]
                    };
                    // Otherwise the raw lowering's launches, the scales
                    // rebound.
                    assert_eq!(old.len(), old_launches.len(), "{at} {stage:?}");
                    for (a, b) in old.iter().zip(old_launches) {
                        assert_eq!(a.launch.symbol, b.launch.symbol, "{at}");
                        assert_eq!(a.launch.plan, b.launch.plan, "{at}");
                        assert_eq!(a.launch.rows, b.launch.rows, "{at}");
                        for (x, y) in bound(&a.launch).into_iter().zip(bound(&b.launch)) {
                            let Some(x) = x else {
                                assert_eq!(y, None, "{at}");
                                continue;
                            };
                            match (0..2).find(|&bank| x.buffer == raw_bank(bank).buffer) {
                                Some(bank) => {
                                    rebound[bank] += 1;
                                    let base = expanded_bank(bank);
                                    assert!(x.offset + x.length <= base.length, "{at}");
                                    assert_eq!(
                                        y,
                                        Some(DeviceBufferSpanV1::new(
                                            base.buffer,
                                            base.offset + x.offset,
                                            x.length,
                                        )),
                                        "{at}"
                                    );
                                }
                                None => assert_eq!(Some(x), y, "{at}"),
                            }
                        }
                    }
                }
                assert!(rebound.iter().all(|&n| n > 0), "{at}");
                // No launch but the expansion binds an encoded bank.
                for (_, launches) in &after {
                    for lowered in launches {
                        if lowered.launch.symbol.starts_with("chop_moe_scale_expand_") {
                            continue;
                        }
                        for bank in 0..2 {
                            assert!(
                                bound(&lowered.launch)
                                    .iter()
                                    .flatten()
                                    .all(|s| s.buffer != raw_bank(bank).buffer),
                                "{at}: {} binds an encoded bank",
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
                    crate::shared_overlap::checked_units(&stages, &launches, &readonly, false)
                        .unwrap_or_else(|e| panic!("{at}: {e}"));
                }
            }
        }
        assert_eq!(direct > 0, ranks == Ranks::TP4, "{ranks:?}");
    }
}

/// One encoded bank expands alone; the other binds its placed view.
#[test]
fn a_single_encoded_bank_expands_alone() {
    let shape = Shape::new(&FLASH, Ranks::TP2).unwrap();
    let kernels = crate::Kernels::new(&shape).unwrap();
    let set = moe::expert_sets(&FLASH)[0];
    let ops = Ops::prepare_set(&kernels, &shape, KvStorage::Native, set);
    let raw = rank_buffers(&shape, &ops, 0);
    let mut b = encoded(&shape, &raw, 0);
    let gate_up = layer_view(0, "ffn.experts.gate_up.scale");
    b.encoded_scales.remove(&gate_up);
    let phase = Phase::Decode(4);
    let lowered = lower(&shape, &ops, &b, 0, phase, &crate::program::moe(0)).unwrap();
    let expand = &lowered[1].1.last().unwrap().launch;
    assert_eq!(expand.symbol, scale_expand_pair_symbol(false));
    assert_eq!(bound(expand)[0], None);
    assert_eq!(bound(expand)[3], None);
    assert_eq!(expand.arguments.0[5..10], [0; 5].map(Argument::U32));
    let gate = lowered
        .iter()
        .find(|(stage, _)| *stage == Stage::ExpertsGateUp(0))
        .unwrap();
    assert!(
        gate.1
            .iter()
            .any(|op| bound(&op.launch).contains(&Some(raw.views[&gate_up])))
    );
}

/// The expansion writes the backbone's gate/up bank in the layout its
/// consumers read: through the tiled paired entry where they read row
/// tiles, which needs that bank, and the paired entry otherwise. A rank
/// loads the tiled cubin only then, and create accepts an encoded gate/up
/// bank beside codes in the consumers' layout, as it accepts the raw bank
/// in that layout and refuses it in the other.
#[test]
fn the_expansion_writes_the_gate_up_bank_as_its_consumers_read_it() {
    use bundle_runtime::index::{RankIndex, View as IndexView};
    use kernel_core::backend::View;
    use kernel_core::gated_mlp::GatedMlpProjection;
    use kernel_core::precision::scale_expand::ScaleExpandQuery;
    let set = moe::expert_sets(&FLASH)[0];
    let banks = [GatedMlpProjection::GateUp, GatedMlpProjection::Down]
        .map(|kind| moe::scale_bank(&FLASH, 4, set, kind).unwrap());
    let launch = |gate_up: bool, layout| {
        let bank = |at: usize, width: u32| {
            let b = banks[at];
            let query = ScaleExpandQuery {
                width,
                rows: b.rows,
                groups: b.groups,
                projections: b.projections,
                experts: b.experts,
            };
            let handle = 1 + 2 * at as u64;
            let views = [
                View {
                    handle,
                    bytes: query.minimum_bank_bytes().unwrap(),
                },
                View {
                    handle: handle + 1,
                    bytes: query.output_bytes().unwrap(),
                },
            ];
            (b, width, views)
        };
        let active = View {
            handle: 9,
            bytes: (u64::from(set.count) + 1) * 4,
        };
        crate::Kernels::scale_expand(
            [gate_up.then(|| bank(0, 1)), Some(bank(1, 4))],
            6,
            Some(active),
            1,
            layout,
        )
        .unwrap()
    };
    let rows = launch(true, RoutedLayout::Rows);
    assert_eq!(rows.symbol, scale_expand_pair_symbol(false));
    // Quad-lane tiles store their scales as row tiles do.
    for layout in [RoutedLayout::Tiles, RoutedLayout::Lanes] {
        let tiled = launch(true, layout);
        assert_eq!(tiled.symbol, scale_expand_pair_symbol(true), "{layout:?}");
        assert_eq!(
            (&tiled.plan, &tiled.arguments),
            (&rows.plan, &rows.arguments),
            "{layout:?}"
        );
    }
    for layout in [RoutedLayout::Rows, RoutedLayout::Tiles, RoutedLayout::Lanes] {
        assert_eq!(
            launch(false, layout).symbol,
            scale_expand_pair_symbol(false),
            "{layout:?}"
        );
    }
    // Each expansion cubin, with its entry, and when a rank loads it.
    for (cubin, symbol) in [
        (
            "moe_scale_expand_deepseek41_sm120",
            scale_expand_pair_symbol(false),
        ),
        (
            "moe_scale_expand_tiles_deepseek41_sm120",
            scale_expand_pair_symbol(true),
        ),
    ] {
        let built = crate::CUBINS.iter().find(|c| c.name == cubin).unwrap();
        assert!(built.symbols.contains(&symbol), "{cubin}");
        let tiles = cubin.contains("_tiles_");
        for layout in [RoutedLayout::Rows, RoutedLayout::Tiles, RoutedLayout::Lanes] {
            assert!(!crate::Kernels::loads_scale_expand(cubin, false, layout));
            assert_eq!(
                crate::Kernels::loads_scale_expand(cubin, true, layout),
                !tiles || layout.tiled(),
                "{cubin} {layout:?}"
            );
        }
    }
    for ranks in [Ranks::TP1, Ranks::TP2, Ranks::TP4] {
        let shape = Shape::new(&FLASH, ranks).unwrap();
        let kernels = crate::Kernels::new(&shape).unwrap();
        let ops = Ops::prepare_set(&kernels, &shape, KvStorage::Native, set);
        let layout = ops.moe.as_ref().unwrap().gate_up_layout();
        let other = match layout {
            RoutedLayout::Rows => RoutedLayout::Tiles,
            RoutedLayout::Tiles | RoutedLayout::Lanes => RoutedLayout::Rows,
        };
        let [codes, scales] = layout.view_layouts().unwrap();
        let view = |layout_id: &str| IndexView {
            offset: 0,
            bytes: 0,
            dtype: "U8".into(),
            shape: Vec::new(),
            layout_id: layout_id.into(),
        };
        let index = |gate_up_scale: &str| RankIndex {
            rank: 0,
            image: String::new(),
            image_bytes: 0,
            sha256: None,
            direct_identity: None,
            views: [
                ("gate_up.weight", view(codes)),
                ("gate_up.scale", view(gate_up_scale)),
                ("down.scale", view(&encoded_scale_layout(4))),
            ]
            .into_iter()
            .map(|(bank, view)| (layer_view(0, &format!("ffn.experts.{bank}")), view))
            .collect(),
        };
        for accepted in [encoded_scale_layout(1).as_str(), scales] {
            ops.check_weight_layouts(&shape, &index(accepted))
                .unwrap_or_else(|e| panic!("{ranks:?} {accepted}: {e}"));
        }
        let error = ops
            .check_weight_layouts(&shape, &index(other.view_layouts().unwrap()[1]))
            .unwrap_err();
        assert!(error.contains("but its op reads"), "{ranks:?}: {error}");
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

/// Tile banks change only qualified scale readers and the banks expanded
/// after routing. All other arguments and launch dimensions stay fixed.
#[test]
fn tile_scales_preserve_the_serving_schedule_and_raw_fallbacks() {
    use crate::buffers::ScaleBankEncoding;
    use crate::sm120::test_symbols::{
        scale_tiles_expand_gate_symbol, scale_tiles_expand_pair_symbol,
    };
    use kernel_core::weight_transform::ScaleFormat;
    let shape = Shape::new(&FLASH, Ranks::TP4).unwrap();
    let kernels = crate::Kernels::new(&shape).unwrap();
    let ops = Ops::prepare_set(
        &kernels,
        &shape,
        KvStorage::Native,
        moe::expert_sets(&FLASH)[0],
    );
    let raw = rank_buffers(&shape, &ops, 0);
    let old = encoded(&shape, &raw, 0);
    let phases = [
        (Phase::Decode(1), true, true),
        (Phase::Decode(8), true, true),
        (Phase::Decode(16), false, true),
        (Phase::Decode(32), true, true),
        (Phase::Verify { rows: 4, items: 1 }, true, true),
        (Phase::Verify { rows: 6, items: 1 }, true, true),
        (Phase::Verify { rows: 8, items: 2 }, true, true),
        (Phase::Verify { rows: 16, items: 4 }, false, true),
        (Phase::Verify { rows: 32, items: 8 }, true, true),
        (
            Phase::Verify {
                rows: 48,
                items: 12,
            },
            false,
            true,
        ),
        (
            Phase::Verify {
                rows: 64,
                items: 16,
            },
            false,
            true,
        ),
        (
            Phase::Verify {
                rows: 192,
                items: 32,
            },
            false,
            false,
        ),
        (Phase::Prefill { rows: 1 }, false, false),
        (Phase::Prefill { rows: 16 }, false, false),
        (Phase::Prefill { rows: 512 }, false, false),
    ];
    for width in [1, 2, 4] {
        let mut tile = old.clone();
        let mut encoded_banks = Vec::new();
        for gate in [true, false] {
            let bank = ops.moe.as_ref().unwrap().scale_bank(gate);
            let name = super::scale_bank_name(0, gate);
            tile.encoded_scales.insert(
                name.clone(),
                ScaleBankEncoding {
                    format: ScaleFormat::RowTilesOffsetV1,
                    width,
                },
            );
            let placed = tile.views.get_mut(&name).unwrap();
            placed.length =
                crate::Kernels::encoded_scale_minimum(bank, width, ScaleFormat::RowTilesOffsetV1)
                    .unwrap();
            encoded_banks.push(*placed);
        }
        for (phase, gate, down) in phases {
            let gate = gate && !(phase.rows() == 1 && width == 4);
            let selected = [gate, down];
            let stages = [
                Stage::Route(0),
                Stage::ExpertsGateUp(0),
                Stage::ExpertsDown(0),
            ];
            let before = lower(&shape, &ops, &old, 0, phase, &stages).unwrap();
            let after = lower(&shape, &ops, &tile, 0, phase, &stages).unwrap();
            let at = format!("{phase:?} width {width}");
            for (stage, ((_, old_calls), (_, new_calls))) in before.iter().zip(&after).enumerate() {
                let old_calls = if stage == 0 {
                    &old_calls[..old_calls.len() - 1]
                } else {
                    &old_calls[..]
                };
                let new_calls = if stage == 0 && selected != [true, true] {
                    let expansion = &new_calls.last().unwrap().launch;
                    assert_eq!(
                        expansion.symbol,
                        if matches!(phase, Phase::Decode(16) | Phase::Verify { rows: 16, .. }) {
                            scale_tiles_expand_gate_symbol(width)
                        } else if gate {
                            scale_tiles_expand_pair_symbol(false).into()
                        } else {
                            scale_tiles_expand_pair_symbol(true).into()
                        },
                        "{at}"
                    );
                    let spans = bound(expansion);
                    for bank in 0..2 {
                        assert_eq!(
                            spans[bank],
                            (!selected[bank]).then_some(encoded_banks[bank]),
                            "{at}"
                        );
                        assert_eq!(spans[bank + 3].is_none(), selected[bank], "{at}");
                    }
                    &new_calls[..new_calls.len() - 1]
                } else {
                    &new_calls[..]
                };
                assert_eq!(old_calls.len(), new_calls.len(), "{at}");
                let mut replaced = 0;
                for (old, new) in old_calls.iter().zip(new_calls) {
                    assert_eq!(old.launch.plan, new.launch.plan, "{at}");
                    assert_eq!(old.launch.rows, new.launch.rows, "{at}");
                    let replace = stage > 0
                        && selected[stage - 1]
                        && crate::Kernels::moe_scale_variant(&old.launch.symbol, width).is_some();
                    let expected = if replace {
                        crate::Kernels::moe_scale_variant(&old.launch.symbol, width).unwrap()
                    } else {
                        &old.launch.symbol
                    };
                    assert_eq!(new.launch.symbol, expected, "{at}");
                    for (arg, (a, b)) in bound(&old.launch)
                        .into_iter()
                        .zip(bound(&new.launch))
                        .enumerate()
                    {
                        if replace && arg == 2 {
                            assert_eq!(b, Some(encoded_banks[stage - 1]), "{at}");
                            replaced += 1;
                        } else {
                            assert_eq!(a, b, "{at} argument {arg}");
                        }
                    }
                    for (arg, (a, b)) in old
                        .launch
                        .arguments
                        .0
                        .iter()
                        .zip(&new.launch.arguments.0)
                        .enumerate()
                    {
                        if !matches!(a, Argument::DeviceBuffer(_)) {
                            assert_eq!(a, b, "{at} scalar {arg}");
                        }
                    }
                }
                if stage > 0 {
                    assert_eq!(replaced > 0, selected[stage - 1], "{at}");
                }
            }
        }
    }
}

#[test]
fn raw_and_row_offset_images_do_not_load_tile_scale_modules() {
    use crate::sm120::scale_inplace::{DOWN_CUBIN, EXPAND_CUBIN, GATE_CUBIN};
    use kernel_core::weight_transform::ScaleFormat;
    for format in [
        None,
        Some(ScaleFormat::RowOffsetV1),
        Some(ScaleFormat::RowTilesOffsetV1),
    ] {
        for layout in [RoutedLayout::Rows, RoutedLayout::Tiles, RoutedLayout::Lanes] {
            for cubin in [GATE_CUBIN, DOWN_CUBIN, EXPAND_CUBIN] {
                assert_eq!(
                    crate::Kernels::loads_scale_kernels(cubin, format, layout),
                    format == Some(ScaleFormat::RowTilesOffsetV1)
                );
            }
        }
    }
}
