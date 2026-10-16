//! The Sinkhorn mHC comm epilogue's lowering (`lower/reduce.rs`, switched by
//! the `runtime.sinkhorn_comm_epilogue` model option) against the unfused launches it
//! replaces, rank by rank over the whole program of each phase. With the
//! switch on, each of the 48 sublayer boundaries that reduce BF16 partials
//! (every attention sublayer's end and the three dense MLPs'; the 41 MoE
//! sublayers' rounded all-reduce of F32 partials and the last MLP, which
//! the head collapse follows, stay unfused) lowers its reduce to the
//! unfused all-reduce (`engines::bf16_tuned`), its rounded sums widened
//! into the rank's F32 sums, and then the split epilogue. The epilogue
//! binds exactly the sums, buffers, views, scalars,
//! chain sums and tickets that the unfused round, expand, next split
//! mix-collapse and next input norm bind (in the decode and verify graphs,
//! the next mixer fused with its input norm), and those stages lower to
//! nothing. Every other stage lowers to the
//! same launches. With the switch off nothing changes, which the journal
//! pins show.

use std::collections::BTreeMap;

use bundle_runtime::dispatch;
use bundle_runtime::fake_host::FakeDevice;
use bundle_runtime::pipeline::PreparedLaunch;
use model_glm5next_contract::contract::{Shape, has_mtp_layer};
use model_glm5next_contract::sm120::engines::{self as selected, ModelRow};
use runtime_bundle_abi::KernelArgumentV1;

use crate::journal_tests::{head_rows, inputs, rank_buffers};
use crate::lower::Lowering;
use crate::program::{Phase, Stage, Sublayer, moe_sublayer, stages};
use crate::sm120::engines::Engines;

/// Rows of every reduce entry (`engines::bf16_tuned`): the
/// low-latency entry at one row, the phased one-shot at 2 to 4 rows, the
/// phased two-shot in sending CTAs at 5 to 32 and with workers at 40.
const PHASES: [Phase; 6] = [
    Phase::Prefill { rows: 7 },
    Phase::Prefill { rows: 40 },
    Phase::Decode(1),
    Phase::Decode(4),
    Phase::Decode(16),
    Phase::Verify { rows: 32, items: 8 },
];

/// A launch's resolved arguments by parameter name.
fn named(
    launch: &PreparedLaunch,
    parameters: &BTreeMap<String, Vec<String>>,
) -> BTreeMap<String, KernelArgumentV1> {
    let arguments = dispatch::arguments(&launch.arguments, &launch.bindings).unwrap();
    let names = &parameters[&launch.symbol];
    assert_eq!(arguments.len(), names.len(), "{}", launch.symbol);
    names.iter().cloned().zip(arguments).collect()
}

/// What two launches do: symbol, rows, plan and resolved arguments.
fn same(left: &PreparedLaunch, right: &PreparedLaunch) -> bool {
    left.symbol == right.symbol
        && left.rows == right.rows
        && left.plan == right.plan
        && dispatch::arguments(&left.arguments, &left.bindings).unwrap()
            == dispatch::arguments(&right.arguments, &right.bindings).unwrap()
}

/// The all-reduce launch of `rows` rows, which leaves the rounded sums:
/// the entry SM120's table gives them (`engines::bf16_tuned`).
fn reduce_symbols(rows: u32) -> Vec<String> {
    vec![selected::bf16_tuned(rows).symbol().to_owned()]
}

/// The fused launches of one boundary bind what its unfused launches
/// (`unfused`: the all-reduce's, then expand, mix-collapse and norm, the
/// last two one launch in the decode and verify graphs) bind. The fused
/// boundary keeps the unfused all-reduce launch for launch and widens its
/// rounded sums into the rank's F32 sums, which the epilogue reads.
fn check_boundary(
    parameters: &BTreeMap<String, Vec<String>>,
    phase: Phase,
    unfused: &[PreparedLaunch],
    fused: &[PreparedLaunch],
) {
    let prefill = matches!(phase, Phase::Prefill { .. });
    let rows = phase.rows();
    let symbols = unfused
        .iter()
        .map(|l| l.symbol.as_str())
        .collect::<Vec<_>>();
    let sinkhorn = |op: &str| format!("chop_sinkhorn_hc_glm5next_{op}_v1");
    let mut expected = reduce_symbols(rows);
    let reduce = expected.len();
    let mix_norm = !prefill;
    expected.push(sinkhorn("expand"));
    if mix_norm {
        expected.push(sinkhorn("mix_collapse_norm_tree"));
    } else {
        expected.extend([
            sinkhorn("mix_collapse_split"),
            selected::rmsnorm_symbol(4096),
        ]);
    }
    assert_eq!(symbols, expected);
    let mut fused_expected = reduce_symbols(rows);
    fused_expected.push(ModelRow::Widen.symbol());
    fused_expected.push("chop_sinkhorn_epilogue_glm5next_rows_split_tree_v1".to_owned());
    assert_eq!(
        fused.iter().map(|l| l.symbol.as_str()).collect::<Vec<_>>(),
        fused_expected
    );
    // The all-reduce stays as it is.
    for (left, right) in unfused[..reduce].iter().zip(fused) {
        assert!(same(left, right), "{}", left.symbol);
    }
    let epilogue = fused.last().unwrap();
    assert_eq!(epilogue.rows, rows);
    assert_eq!(epilogue.plan.grid, [16, rows, 1]);
    assert_eq!(epilogue.plan.block, [256, 1, 1]);
    let e = named(epilogue, parameters);
    // The F32 sums the epilogue reads are the all-reduce's rounded sums
    // widened; it rounds them into the same rows at the same live rows.
    let widen = named(&fused[reduce], parameters);
    let round = named(&unfused[reduce - 1], parameters);
    assert_eq!(widen["input"], round["output"]);
    assert_eq!(e["summed"], widen["output"]);
    assert_eq!(e["reduced"], round["output"]);
    assert_eq!(e["live_rows"], widen["live_rows"]);
    if let Some(live_rows) = round.get("live_rows") {
        assert_eq!(e["live_rows"], *live_rows);
    }
    let start = reduce;
    let [expand, mix] = [0, 1].map(|i| named(&unfused[start + i], parameters));
    // The fused mixer binds the norm's arguments by the same names.
    let norm = if mix_norm {
        let mut norm = mix.clone();
        norm.insert("x".into(), mix["collapsed"].clone());
        norm.insert("y".into(), mix["normed"].clone());
        norm
    } else {
        named(&unfused[start + 2], parameters)
    };
    let pairs: [(&str, &BTreeMap<String, KernelArgumentV1>, &str); 19] = [
        ("reduced", &expand, "y"),
        ("streams", &expand, "x"),
        ("expanded", &expand, "out"),
        ("expanded", &mix, "x"),
        ("post", &expand, "post"),
        ("post", &mix, "post"),
        ("comb", &expand, "comb"),
        ("comb", &mix, "comb"),
        ("fn", &mix, "fn"),
        ("base", &mix, "base"),
        ("scale", &mix, "scale"),
        ("collapsed", &mix, "collapsed"),
        ("partials", &mix, "partials"),
        ("tickets", &mix, "tickets"),
        ("collapsed", &norm, "x"),
        ("weight", &norm, "weight"),
        ("normed", &norm, "y"),
        ("eps_bits", &norm, "eps_bits"),
        ("live_rows", &norm, "live_rows"),
    ];
    for (key, launch, theirs) in pairs {
        assert_eq!(e[key], launch[theirs], "{key} against {theirs}");
    }
    for launch in [&expand, &mix] {
        assert_eq!(e["live_rows"], launch["live_rows"]);
    }
}

#[test]
fn each_boundary_binds_what_its_unfused_launches_bind() {
    let shape = Shape::TP4;
    let format = shape
        .kv_formats()
        .into_iter()
        .find(|format| !has_mtp_layer(format))
        .unwrap();
    let device: FakeDevice = FakeDevice::default();
    let engines = Engines::new(shape).unwrap();
    let ops = crate::ops::Ops::for_tests(shape, 1);
    let ranks = (0..shape.ranks())
        .map(|rank| rank_buffers(&device, rank, &format))
        .collect::<Vec<_>>();
    let peers = ranks
        .iter()
        .map(|buffers| buffers.comm.clone())
        .collect::<Vec<_>>();
    let parameters = engines.parameters(shape);
    let layers = shape.geometry.layers();
    // A boundary the epilogue fuses: neither the last MLP nor a MoE
    // sublayer's end.
    let fused = |layer: u32, sublayer| {
        let last = layer + 1 == layers && sublayer == Sublayer::Mlp;
        !(last || moe_sublayer(shape, layer, sublayer))
    };
    // A sublayer whose mixer and input norm the boundary before it fused.
    let fused_into = |layer: u32, sublayer| match sublayer {
        Sublayer::Mlp => fused(layer, Sublayer::Attention),
        Sublayer::Attention => layer > 0 && fused(layer - 1, Sublayer::Mlp),
    };
    for phase in PHASES {
        let one_shot = selected::one_shot(shape, phase.rows()).unwrap();
        assert_eq!(
            one_shot,
            selected::ONE_SHOT_MAX_ROWS
                .checked_sub(phase.rows())
                .is_some(),
            "{phase:?}"
        );
        let program = stages(shape, phase);
        for (rank, buffers) in ranks.iter().enumerate() {
            let inputs = inputs(rank as u32, phase);
            let lowering = Lowering {
                shape,
                format: &format,
                buffers,
                inputs: &inputs,
                ops: &ops,
                rank: rank as u32,
                index_topk: shape.profile.index_topk_max,
                pieces: &[],
                row_offset: 0,
                peers: &peers,
                head_rows: head_rows(phase),
                context_parallel: None,
                draft: None,
                verify_tokens: None,
                projections: None,
            };
            let (mut boundaries, mut unfused_launches, mut fused_launches) = (0, 0, 0);
            for (index, &stage) in program.iter().enumerate() {
                let off = lowering.lower_with(phase, stage, false).unwrap();
                let on = lowering.lower_with(phase, stage, true).unwrap();
                unfused_launches += off.as_ref().map_or(0, Vec::len);
                fused_launches += on.as_ref().map_or(0, Vec::len);
                match stage {
                    Stage::Reduce(layer, sublayer) if fused(layer, sublayer) => {
                        boundaries += 1;
                        let next = match sublayer {
                            Sublayer::Attention => (layer, Sublayer::Mlp),
                            Sublayer::Mlp => (layer + 1, Sublayer::Attention),
                        };
                        assert_eq!(
                            program[index + 1..index + 4],
                            [
                                Stage::HcExpand(layer, sublayer),
                                Stage::HcMix(next.0, next.1),
                                Stage::InputNorm(next.0, next.1),
                            ]
                        );
                        let mut replaced = off.unwrap();
                        for &stage in &program[index + 1..index + 4] {
                            replaced.extend(lowering.lower(phase, stage).unwrap().unwrap());
                        }
                        check_boundary(&parameters, phase, &replaced, &on.unwrap());
                    }
                    Stage::HcExpand(layer, sublayer) if fused(layer, sublayer) => {
                        assert_eq!(on.unwrap().len(), 0);
                    }
                    Stage::HcMix(layer, sublayer) | Stage::InputNorm(layer, sublayer)
                        if fused_into(layer, sublayer) =>
                    {
                        assert_eq!(on.unwrap().len(), 0);
                    }
                    _ => match (off, on) {
                        (None, None) => {}
                        (Some(off), Some(on)) => {
                            assert_eq!(off.len(), on.len(), "{stage:?}");
                            for (left, right) in off.iter().zip(&on) {
                                assert!(same(left, right), "{stage:?} {}", left.symbol);
                            }
                        }
                        (off, on) => panic!(
                            "{stage:?}: lowered {} against {}",
                            off.is_some(),
                            on.is_some()
                        ),
                    },
                }
            }
            assert_eq!(boundaries, 48);
            // The expand, mix-collapse and norm become the widening and
            // the epilogue (in the decode and verify graphs, which fuse the
            // mixer and its norm, as many launches).
            let prefill = matches!(phase, Phase::Prefill { .. });
            let merged = if prefill { 1 } else { 0 };
            assert_eq!(
                unfused_launches - fused_launches,
                merged * boundaries,
                "{phase:?}"
            );
        }
    }
}

/// The drafter's passes have no Sinkhorn mixer: their reduces, norms and
/// residual adds lower alike with the switch on.
#[test]
fn the_drafter_lowers_alike_with_the_switch_on() {
    use crate::program::{DraftPass, drafter_stages};
    let shape = Shape::TP4;
    let format = shape.kv_formats().into_iter().find(has_mtp_layer).unwrap();
    let device: FakeDevice = FakeDevice::default();
    let ops = crate::ops::Ops::for_tests(shape, 1);
    let ranks = (0..shape.ranks())
        .map(|rank| rank_buffers(&device, rank, &format))
        .collect::<Vec<_>>();
    let peers = ranks
        .iter()
        .map(|buffers| buffers.comm.clone())
        .collect::<Vec<_>>();
    for (pass, phase) in [
        (DraftPass::Extend, Phase::Verify { rows: 32, items: 8 }),
        (DraftPass::Step, Phase::Decode(8)),
    ] {
        let inputs = inputs(0, phase);
        let verify = crate::journal_tests::inputs(0, Phase::Verify { rows: 32, items: 8 });
        let lowering = Lowering {
            shape,
            format: &format,
            buffers: &ranks[0],
            inputs: &inputs,
            ops: &ops,
            rank: 0,
            index_topk: shape.profile.index_topk_max,
            pieces: &[],
            row_offset: 0,
            peers: &peers,
            head_rows: head_rows(phase),
            context_parallel: None,
            draft: Some(pass),
            verify_tokens: Some(verify.token_ids),
            projections: None,
        };
        for stage in drafter_stages(shape, pass, phase) {
            let off = lowering.lower_with(phase, stage, false).unwrap();
            let on = lowering.lower_with(phase, stage, true).unwrap();
            match (off, on) {
                (None, None) => {}
                (Some(off), Some(on)) => {
                    assert_eq!(off.len(), on.len(), "{stage:?}");
                    for (left, right) in off.iter().zip(&on) {
                        assert!(same(left, right), "{stage:?}");
                    }
                }
                (off, on) => panic!(
                    "{stage:?}: lowered {} against {}",
                    off.is_some(),
                    on.is_some()
                ),
            }
        }
    }
}
